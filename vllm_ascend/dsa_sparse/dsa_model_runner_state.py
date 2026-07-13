# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ascend model runner 中 DSA 专用的请求状态更新逻辑。

上游 GPUModelRunner 默认每个 KV cache group 都按完整序列单调增长。
DSA 稀疏 decode 不同：Indexer group 需要保持完整 dense 序列，而
MLA/full-resident group 会被压缩成 sparse budget + resident tail。
本文件把这部分状态更新留在 vllm-ascend 内部，避免在 vLLM 主仓加入
Ascend/DSA 专有分支。
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.spec_decode.ngram_proposer_gpu import (
    update_ngram_gpu_tensors_incremental,
    update_scheduler_for_invalid_drafts,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.dsa_sparse.dsa_types import INVALID_SLOT, ReqStage
from vllm_ascend.dsa_sparse.dsa_spec_utils import is_dsa_mla_resident_spec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


def dsa_request_finished_in_worker(self, request_id: str) -> None:
    self.dsa_worker_mgr.request_finished_in_worker(request_id)


def dsa_request_preempted_in_worker(self, request_id: str) -> None:
    self.dsa_worker_mgr.request_preempted_in_worker(request_id)


def normalize_dsa_decode_block_ids(
    self,
    req_id: str,
    req_state: CachedRequestState,
    new_block_ids: tuple[list[int], ...] | None,
    *,
    resumed_from_preemption: bool,
) -> tuple[list[int], ...] | None:
    if new_block_ids is None:
        return None

    expected_num_groups = len(req_state.block_ids)
    if expected_num_groups == 0 or len(new_block_ids) == expected_num_groups:
        return new_block_ids

    kv_cache_groups = self.kv_cache_config.kv_cache_groups
    if len(kv_cache_groups) != expected_num_groups:
        raise RuntimeError(
            "DSA decode block-id normalization got inconsistent KV groups "
            f"for req {req_id}: expected_groups={expected_num_groups}, "
            f"kv_cache_groups={len(kv_cache_groups)}, "
            f"new_block_groups={len(new_block_ids)}")

    full_group_ids = [
        group_id
        for group_id, kv_cache_group in enumerate(kv_cache_groups)
        if is_dsa_mla_resident_spec(kv_cache_group.kv_cache_spec)
    ]
    if len(new_block_ids) == 1 and full_group_ids:
        if resumed_from_preemption:
            raise RuntimeError(
                "Resumed sparse request must refresh every KV group, but got "
                f"only {len(new_block_ids)} block-id group(s) for req {req_id}. "
                f"expected_groups={expected_num_groups} "
                f"full_group_ids={full_group_ids}")
        normalized_block_ids = [
            list(group_block_ids) for group_block_ids in req_state.block_ids
        ]
        normalized_block_ids[full_group_ids[0]] = list(new_block_ids[0])
        return tuple(normalized_block_ids)

    raise RuntimeError(
        "DSA decode block-id normalization could not map scheduler output "
        f"for req {req_id}: expected_groups={expected_num_groups}, "
        f"new_block_groups={len(new_block_ids)}, "
        f"full_group_ids={full_group_ids}")


def update_states(self, scheduler_output: "SchedulerOutput") -> None:
    """Update NPU runner state with DSA sparse cache semantics."""
    for req_id in scheduler_output.finished_req_ids:
        self.dsa_request_finished_in_worker(req_id)
        self.requests.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)
    self.late_interaction_runner.on_requests_finished(
        scheduler_output.finished_req_ids)

    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)

    if scheduler_output.new_block_ids_to_zero:
        self._zero_block_ids(scheduler_output.new_block_ids_to_zero)

    for req_id in scheduler_output.preempted_req_ids or ():
        self.dsa_request_preempted_in_worker(req_id)

    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        self.encoder_cache.pop(mm_hash, None)

    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    unscheduled_req_ids = cached_req_ids - (scheduled_req_ids -
                                            resumed_req_ids)
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)

    is_ngram_gpu = (self.speculative_config is not None
                    and self.speculative_config.use_ngram_gpu())
    if is_ngram_gpu:
        ngram_gpu_new_reqs: list[CachedRequestState] = []

    reqs_to_add: list[CachedRequestState] = []
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        if req_id in self.requests:
            req_state = self._update_streaming_request(req_id, new_req_data)
            reqs_to_add.append(req_state)
            continue

        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params

        if (sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        if self.is_pooling_model:
            assert pooling_params is not None
            task = pooling_params.task
            assert task is not None, "You did not set `task` in the API"

            model = cast(VllmModelForPooling, self.get_model())
            to_update = model.pooler.get_pooling_updates(task)
            to_update.apply(pooling_params)

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt_embeds=new_req_data.prompt_embeds,
            mm_features=new_req_data.mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )
        req_state.context_full_blk_hashes = list(new_req_data.block_hashes or [])
        self.requests[req_id] = req_state
        self.late_interaction_runner.register_request(req_id, pooling_params)

        if sampling_params and sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = (
                self.input_batch.vocab_size
                if sampling_params.prompt_logprobs == -1 else
                sampling_params.prompt_logprobs)

        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        if self.uses_xdrope_dim > 0:
            self._init_xdrope_positions(req_state)

        reqs_to_add.append(req_state)
        if is_ngram_gpu:
            ngram_gpu_new_reqs.append(req_state)

    is_last_rank = get_pp_group().is_last_rank
    req_data = scheduler_output.scheduled_cached_reqs
    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    req_dsa_resident_valid_seq_len = (
        scheduler_output.req_dsa_resident_valid_seq_len)
    req_dsa_stage = scheduler_output.req_dsa_stage

    original_num_spec_per_req: dict[str, int] = {}
    if (self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()):
        for req_id, toks in scheduled_spec_tokens.items():
            original_num_spec_per_req[req_id] = len(toks)
        update_scheduler_for_invalid_drafts(
            self._num_valid_draft_tokens_event,
            self._num_valid_draft_tokens_cpu,
            scheduler_output,
            self.input_batch.req_id_to_index,
        )
    valid_sampled_token_count = self._get_valid_sampled_token_count()

    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_id in req_data.resumed_req_ids
        num_output_tokens = req_data.num_output_tokens[i]
        req_index = self.input_batch.req_id_to_index.get(req_id)
        resident_valid_seq_len = int(
            req_dsa_resident_valid_seq_len[req_id])
        req_stage = ReqStage(req_dsa_stage[req_id])
        is_dsa_sparse_request = (req_stage.is_sparse_decode
                                 and resident_valid_seq_len != INVALID_SLOT)
        if new_block_ids is not None and is_dsa_sparse_request:
            new_block_ids = self._normalize_dsa_decode_block_ids(
                req_id,
                req_state,
                new_block_ids,
                resumed_from_preemption=resumed_from_preemption,
            )
        if i < len(req_data.block_hashes):
            req_state.context_full_blk_hashes = list(req_data.block_hashes[i])

        if req_state.prev_num_draft_len and self.use_async_scheduling:
            if req_index is None:
                req_state.prev_num_draft_len = 0
            else:
                assert self.input_batch.prev_req_id_to_index is not None
                prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                num_accepted = valid_sampled_token_count[prev_req_index] - 1
                num_rejected = req_state.prev_num_draft_len - num_accepted
                num_computed_tokens -= num_rejected
                req_state.output_token_ids.extend([-1] * num_accepted)

                if is_ngram_gpu and num_accepted > 0:
                    self.input_batch.num_tokens_no_spec[req_index] += (
                        num_accepted)

        req_state.num_computed_tokens = num_computed_tokens

        if not is_last_rank:
            if not req_data.new_token_ids:
                new_token_ids: list[int] = []
            else:
                new_token_ids = req_data.new_token_ids[i]
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])
        elif num_output_tokens < len(req_state.output_token_ids):
            del req_state.output_token_ids[num_output_tokens:]
            if req_index is not None:
                end_idx = (self.input_batch.num_prompt_tokens[req_index] +
                           num_output_tokens)
                self.input_batch.num_tokens_no_spec[req_index] = end_idx

        if is_dsa_sparse_request:
            if new_block_ids is not None:
                req_state.block_ids = new_block_ids
        elif not resumed_from_preemption:
            if new_block_ids is not None:
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
        else:
            assert req_index is None
            assert new_block_ids is not None
            req_state.block_ids = new_block_ids

        if req_index is None:
            if self.use_async_scheduling and num_output_tokens > 0:
                resumed_token_ids = req_data.all_token_ids[req_id]
                req_state.output_token_ids = resumed_token_ids[
                    -num_output_tokens:]

            reqs_to_add.append(req_state)
            if is_ngram_gpu:
                ngram_gpu_new_reqs.append(req_state)
            continue

        self.input_batch.num_computed_tokens_cpu[req_index] = (
            num_computed_tokens)
        if new_block_ids is not None:
            if is_dsa_sparse_request:
                self.input_batch.block_table.add_row(new_block_ids, req_index)
            else:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

        if not is_last_rank:
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index, start_token_index:end_token_index] = new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index

        self.input_batch.update_req_spec_token_ids(req_state,
                                                   scheduled_spec_tokens)
        if original_num_spec_per_req:
            orig = original_num_spec_per_req.get(req_id, 0)
            if orig != req_state.prev_num_draft_len:
                req_state.prev_num_draft_len = orig

    for request in reqs_to_add:
        self.input_batch.add_request(request)
        self.input_batch.update_req_spec_token_ids(request,
                                                   scheduled_spec_tokens)

    self.input_batch.condense()
    self._may_reorder_batch(scheduler_output)
    self.input_batch.refresh_metadata()

    if is_ngram_gpu:
        update_ngram_gpu_tensors_incremental(
            self.input_batch,
            self.token_ids_gpu_tensor,
            self.num_tokens_no_spec_gpu,
            ngram_gpu_new_reqs,
            self.device,
            _pinned_idx_buf=self._ngram_pinned_idx_buf,
            _pinned_val_buf=self._ngram_pinned_val_buf,
        )


def update_streaming_request(self, req_id: str,
                             new_req_data: NewRequestData) -> CachedRequestState:
    req_state = GPUModelRunner._update_streaming_request(self, req_id,
                                                         new_req_data)
    req_state.context_full_blk_hashes = list(new_req_data.block_hashes or [])
    return req_state
