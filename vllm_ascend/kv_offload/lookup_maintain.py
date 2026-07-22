#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
from vllm.logger import logger

INDEX_CAPACITY = 144 * 1024
RESIDENT_SLOT_COUNT = 8 * 1024
QUERY_SLOT_COUNT = 2 * 1024
TOTAL_SLOT_COUNT = RESIDENT_SLOT_COUNT + QUERY_SLOT_COUNT
FIXED_MISS_COUNT = 300
FREE_HEAD_STRIDE = 16
SELECTION_TOPK_BLOCK_SIZE = 1
SUPPORTED_MICROBATCHES = (0, 1)
_REQUIRED_INDEX_OPS = (
    "asu_hbm_index_lookup",
    "asu_hbm_index_maintain_aicpu",
    "dmp_lookup_kv_gather",
)
_REQUIRED_ATTENTION_OPS = (
    "npu_dmp_sparse_flash_attention",
    "npu_da_attention_merge",
)


@dataclass
class LookupMaintainWorkspace:
    """Persistent index state for one layer and one microbatch."""

    token_to_slot: torch.Tensor
    slot_to_token: torch.Tensor
    free_slots: torch.Tensor
    free_head: torch.Tensor
    req_pool_entries: torch.Tensor
    request_signature: torch.Tensor
    previous_seq_lens: torch.Tensor
    slot_out: torch.Tensor | None = None
    miss_out: torch.Tensor | None = None
    query_index: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    resident_token_ids: torch.Tensor | None = None
    batch_size: int = 0


@dataclass
class LookupAttentionWorkspace:
    """One microbatch view into a paired 10K resident-KV workspace."""

    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_kv_block_table: torch.Tensor
    full_k_rope: torch.Tensor
    full_kv_cache: torch.Tensor
    hit_sparse_indices: torch.Tensor
    miss_sparse_indices: torch.Tensor
    selected_actual_seq: torch.Tensor
    needs_refill: torch.Tensor
    hit_softmax_max: torch.Tensor
    hit_softmax_sum: torch.Tensor
    miss_softmax_max: torch.Tensor
    miss_softmax_sum: torch.Tensor
    hit_attention_out: torch.Tensor | None = None


@dataclass
class LookupAttentionPairWorkspace:
    """Shared storage used by one combined mb0+mb1 segmented-SFA call."""

    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_kv_block_table: torch.Tensor
    full_k_rope: torch.Tensor
    full_kv_cache: torch.Tensor
    hit_sparse_indices: torch.Tensor
    miss_sparse_indices: torch.Tensor
    selected_actual_seq: torch.Tensor
    needs_refill: torch.Tensor
    hit_softmax_max: torch.Tensor
    hit_softmax_sum: torch.Tensor
    miss_softmax_max: torch.Tensor
    miss_softmax_sum: torch.Tensor
    microbatch_size: int
    token_count_per_microbatch: int
    hit_attention_out: torch.Tensor | None = None


class DMPLookupMaintain:
    """Lookup/Maintain with full-cache hits and a 10K miss-staging pool.

    Scheme 4 owns a miss-only KVGather. It consumes the 2K Lookup result and
    writes the fixed misses into arbitrary staging slots in one launch. Hit
    SFA reads the original vLLM KV cache. Scheme 2 keeps its original
    Dual-Attention KVGather path unchanged.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        num_layers: int,
        max_microbatch_tokens: int,
        max_model_len: int,
        block_size: int = 128,
        custom_ops: Any | None = None,
        attention_ops: Any | None = None,
        maintain_stream: Any | None = None,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_microbatch_tokens <= 0:
            raise ValueError(
                f"max_microbatch_tokens must be positive, got {max_microbatch_tokens}"
            )
        if max_model_len > INDEX_CAPACITY:
            raise ValueError(
                "DMP Lookup/Maintain index capacity is smaller than the model "
                f"context: capacity={INDEX_CAPACITY}, max_model_len={max_model_len}"
            )
        if block_size <= 0 or TOTAL_SLOT_COUNT % block_size != 0:
            raise ValueError(
                f"Lookup/Maintain requires block_size to divide 10240, got {block_size}"
            )

        self.device = device
        self.num_layers = int(num_layers)
        self.max_microbatch_tokens = int(max_microbatch_tokens)
        self.block_size = int(block_size)
        self.custom_ops = custom_ops or self._load_index_ops()
        self.attention_ops = attention_ops or self._load_attention_ops()
        missing_index = [
            name for name in _REQUIRED_INDEX_OPS if not hasattr(self.custom_ops, name)
        ]
        missing_attention = [
            name
            for name in _REQUIRED_ATTENTION_OPS
            if not hasattr(self.attention_ops, name)
        ]
        if missing_index:
            raise RuntimeError(
                "DMP Lookup/Maintain operators are unavailable: "
                + ", ".join(missing_index)
            )
        if missing_attention:
            raise RuntimeError(
                "DMP Lookup/Maintain attention operators are unavailable: "
                + ", ".join(missing_attention)
            )

        self.maintain_stream = maintain_stream or torch.npu.Stream()
        self._workspaces = {
            (layer_idx, microbatch_idx): self._allocate_index_workspace()
            for layer_idx in range(self.num_layers)
            for microbatch_idx in SUPPORTED_MICROBATCHES
        }
        self._attention_workspaces: dict[tuple[Any, ...], LookupAttentionWorkspace] = {}
        self._attention_pair_workspaces: dict[
            tuple[Any, ...], LookupAttentionPairWorkspace
        ] = {}
        self._active_attention_keys: dict[tuple[int, int], tuple[Any, ...]] = {}
        self._active_attention_pair_keys: dict[int, tuple[Any, ...]] = {}
        self._activation_logged = False

    @staticmethod
    def _load_index_ops() -> Any:
        try:
            import_module("dmp_lookup_maintain_custom_ops")
        except ImportError as exc:
            raise RuntimeError(
                "VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=1 requires the "
                "dmp_lookup_maintain_custom_ops extension"
            ) from exc
        return torch.ops.dmp_lookup_maintain

    @staticmethod
    def _load_attention_ops() -> Any:
        try:
            import_module("custom_ops")
        except ImportError as exc:
            raise RuntimeError(
                "DMP Lookup/Maintain requires the pip-cache Dual-Attention "
                "custom_ops wheel"
            ) from exc
        return torch.ops.custom

    def _allocate_index_workspace(self) -> LookupMaintainWorkspace:
        token_to_slot = torch.full(
            (self.max_microbatch_tokens, INDEX_CAPACITY),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        slot_to_token = torch.full(
            (self.max_microbatch_tokens, TOTAL_SLOT_COUNT),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        initial_tokens = torch.arange(
            RESIDENT_SLOT_COUNT, dtype=torch.int32, device=self.device
        )
        token_to_slot[:, :RESIDENT_SLOT_COUNT] = initial_tokens
        slot_to_token[:, :RESIDENT_SLOT_COUNT] = initial_tokens
        free_slots = (
            torch.arange(
                RESIDENT_SLOT_COUNT,
                TOTAL_SLOT_COUNT,
                dtype=torch.int32,
                device=self.device,
            )
            .view(1, -1)
            .expand(self.max_microbatch_tokens, -1)
            .clone()
        )
        return LookupMaintainWorkspace(
            token_to_slot=token_to_slot,
            slot_to_token=slot_to_token,
            free_slots=free_slots,
            free_head=torch.zeros(
                (self.max_microbatch_tokens, FREE_HEAD_STRIDE),
                dtype=torch.int32,
                device=self.device,
            ),
            req_pool_entries=torch.arange(
                self.max_microbatch_tokens,
                dtype=torch.int32,
                device=self.device,
            ),
            request_signature=torch.full(
                (self.max_microbatch_tokens,),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            previous_seq_lens=torch.full(
                (self.max_microbatch_tokens,),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
        )

    @property
    def allocated_tensor_bytes(self) -> int:
        index_bytes = sum(
            tensor.numel() * tensor.element_size()
            for workspace in self._workspaces.values()
            for tensor in (
                workspace.token_to_slot,
                workspace.slot_to_token,
                workspace.free_slots,
                workspace.free_head,
                workspace.req_pool_entries,
                workspace.request_signature,
                workspace.previous_seq_lens,
            )
        )
        attention_bytes = 0
        for workspace in self._attention_pair_workspaces.values():
            tensors = (
                workspace.selection_k_rope,
                workspace.selection_kv_cache,
                workspace.selection_kv_block_table,
                workspace.hit_sparse_indices,
                workspace.miss_sparse_indices,
                workspace.selected_actual_seq,
                workspace.needs_refill,
                workspace.hit_softmax_max,
                workspace.hit_softmax_sum,
                workspace.miss_softmax_max,
                workspace.miss_softmax_sum,
            )
            attention_bytes += sum(t.numel() * t.element_size() for t in tensors)
        return index_bytes + attention_bytes

    def get_workspace(
        self, layer_idx: int, microbatch_idx: int
    ) -> LookupMaintainWorkspace:
        try:
            return self._workspaces[(layer_idx, microbatch_idx)]
        except KeyError as exc:
            raise RuntimeError(
                "DMP Lookup/Maintain workspace is unavailable for "
                f"layer={layer_idx}, microbatch={microbatch_idx}"
            ) from exc

    def get_attention_workspace(
        self, layer_idx: int, microbatch_idx: int
    ) -> LookupAttentionWorkspace:
        try:
            key = self._active_attention_keys[(layer_idx, microbatch_idx)]
            return self._attention_workspaces[key]
        except KeyError as exc:
            raise RuntimeError(
                "DMP Lookup attention workspace is unavailable for "
                f"layer={layer_idx}, microbatch={microbatch_idx}"
            ) from exc

    @staticmethod
    def _normalize_cache(cache: torch.Tensor, name: str) -> torch.Tensor:
        if cache.ndim == 4:
            if cache.shape[2] != 1:
                raise ValueError(f"{name} requires one KV head, got {cache.shape}")
            return cache.squeeze(2)
        if cache.ndim != 3:
            raise ValueError(f"Unexpected {name} shape: {cache.shape}")
        return cache

    @staticmethod
    def _normalize_topk(topk_indices: torch.Tensor) -> torch.Tensor:
        if topk_indices.ndim < 2 or topk_indices.ndim > 4:
            raise RuntimeError(
                "DMP Lookup/Maintain TopK must have rank 2, 3, or 4, got "
                f"{tuple(topk_indices.shape)}"
            )
        batch_size = int(topk_indices.shape[0])
        topk = topk_indices.to(dtype=torch.int32).reshape(batch_size, -1)
        if topk.shape[1] != QUERY_SLOT_COUNT:
            raise RuntimeError(
                "DMP Lookup/Maintain requires TopK width 2048, got "
                f"{tuple(topk_indices.shape)}"
            )
        return topk.contiguous()

    def _attention_pair_workspace_key(
        self,
        layer_idx: int,
        topk: torch.Tensor,
        ql_nope: torch.Tensor,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            layer_idx,
            tuple(topk.shape),
            tuple(ql_nope.shape),
            full_kv_cache.shape[-1],
            full_k_rope.shape[-1],
            full_kv_cache.data_ptr(),
            full_k_rope.data_ptr(),
            topk.device,
            ql_nope.dtype,
            full_kv_cache.dtype,
            full_k_rope.dtype,
        )

    def _allocate_attention_pair_workspace(
        self,
        topk: torch.Tensor,
        ql_nope: torch.Tensor,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
    ) -> LookupAttentionPairWorkspace:
        batch_size = int(topk.shape[0])
        pair_batch_size = len(SUPPORTED_MICROBATCHES) * batch_size
        blocks_per_row = TOTAL_SLOT_COUNT // self.block_size
        selection_num_blocks = pair_batch_size * blocks_per_row
        device = topk.device

        selection_k_rope = torch.empty(
            (selection_num_blocks, self.block_size, full_k_rope.shape[-1]),
            dtype=full_k_rope.dtype,
            device=device,
        )
        selection_kv_cache = torch.empty(
            (selection_num_blocks, self.block_size, full_kv_cache.shape[-1]),
            dtype=full_kv_cache.dtype,
            device=device,
        )
        global_block_table = torch.arange(
            selection_num_blocks, dtype=torch.int32, device=device
        ).view(pair_batch_size, blocks_per_row)
        scalar_shape = (pair_batch_size,)

        lse_shape = (
            len(SUPPORTED_MICROBATCHES) * ql_nope.shape[0],
            ql_nope.shape[1],
        )
        neg_inf = torch.finfo(torch.float32).min
        return LookupAttentionPairWorkspace(
            selection_k_rope=selection_k_rope,
            selection_kv_cache=selection_kv_cache,
            selection_kv_block_table=global_block_table,
            full_k_rope=full_k_rope,
            full_kv_cache=full_kv_cache,
            hit_sparse_indices=torch.empty(
                (pair_batch_size, 1, QUERY_SLOT_COUNT),
                dtype=torch.int32,
                device=device,
            ),
            miss_sparse_indices=torch.empty(
                (pair_batch_size, 1, QUERY_SLOT_COUNT),
                dtype=torch.int32,
                device=device,
            ),
            selected_actual_seq=torch.full(
                scalar_shape, TOTAL_SLOT_COUNT, dtype=torch.int32, device=device
            ),
            needs_refill=torch.empty(scalar_shape, dtype=torch.bool, device=device),
            hit_softmax_max=torch.full(
                lse_shape, neg_inf, dtype=torch.float32, device=device
            ),
            hit_softmax_sum=torch.zeros(lse_shape, dtype=torch.float32, device=device),
            miss_softmax_max=torch.full(
                lse_shape, neg_inf, dtype=torch.float32, device=device
            ),
            miss_softmax_sum=torch.zeros(lse_shape, dtype=torch.float32, device=device),
            microbatch_size=batch_size,
            token_count_per_microbatch=ql_nope.shape[0],
        )

    def _make_microbatch_attention_workspace(
        self,
        pair: LookupAttentionPairWorkspace,
        microbatch_idx: int,
    ) -> LookupAttentionWorkspace:
        batch_size = pair.microbatch_size
        blocks_per_row = TOTAL_SLOT_COUNT // self.block_size
        row_start = microbatch_idx * batch_size
        row_end = row_start + batch_size
        block_start = row_start * blocks_per_row
        block_end = row_end * blocks_per_row
        token_start = microbatch_idx * pair.token_count_per_microbatch
        token_end = token_start + pair.token_count_per_microbatch
        local_block_table = torch.arange(
            batch_size * blocks_per_row,
            dtype=torch.int32,
            device=pair.selection_kv_cache.device,
        ).view(batch_size, blocks_per_row)
        return LookupAttentionWorkspace(
            selection_k_rope=pair.selection_k_rope[block_start:block_end],
            selection_kv_cache=pair.selection_kv_cache[block_start:block_end],
            selection_kv_block_table=local_block_table,
            full_k_rope=pair.full_k_rope,
            full_kv_cache=pair.full_kv_cache,
            hit_sparse_indices=pair.hit_sparse_indices[row_start:row_end],
            miss_sparse_indices=pair.miss_sparse_indices[row_start:row_end],
            selected_actual_seq=pair.selected_actual_seq[row_start:row_end],
            needs_refill=pair.needs_refill[row_start:row_end],
            hit_softmax_max=pair.hit_softmax_max[token_start:token_end],
            hit_softmax_sum=pair.hit_softmax_sum[token_start:token_end],
            miss_softmax_max=pair.miss_softmax_max[token_start:token_end],
            miss_softmax_sum=pair.miss_softmax_sum[token_start:token_end],
        )

    def _get_or_create_attention_workspace(
        self,
        layer_idx: int,
        microbatch_idx: int,
        topk: torch.Tensor,
        ql_nope: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
    ) -> LookupAttentionWorkspace:
        full_kv_cache = self._normalize_cache(kv_cache[0], "full KV cache")
        full_k_rope = self._normalize_cache(kv_cache[1], "full K-rope cache")
        pair_key = self._attention_pair_workspace_key(
            layer_idx,
            topk,
            ql_nope,
            full_kv_cache,
            full_k_rope,
        )
        pair = self._attention_pair_workspaces.get(pair_key)
        if pair is None:
            pair = self._allocate_attention_pair_workspace(
                topk, ql_nope, full_kv_cache, full_k_rope
            )
            self._attention_pair_workspaces[pair_key] = pair
            logger.info(
                "Allocated shared mb0+mb1 10K DMP Lookup KV pool for layer %d",
                layer_idx,
            )
        if (
            pair.microbatch_size != topk.shape[0]
            or pair.token_count_per_microbatch != ql_nope.shape[0]
        ):
            raise RuntimeError(
                "DMP combined attention requires equal mb0/mb1 graph shapes"
            )
        key = (pair_key, microbatch_idx)
        workspace = self._attention_workspaces.get(key)
        if workspace is None:
            workspace = self._make_microbatch_attention_workspace(
                pair, microbatch_idx
            )
            self._attention_workspaces[key] = workspace
        self._active_attention_keys[(layer_idx, microbatch_idx)] = key
        self._active_attention_pair_keys[layer_idx] = pair_key
        return workspace

    @staticmethod
    def _unwrap_operator_output(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def lookup(
        self,
        *,
        layer_idx: int,
        microbatch_idx: int,
        indexer_result: tuple[torch.Tensor, ...],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if microbatch_idx not in SUPPORTED_MICROBATCHES:
            raise RuntimeError(
                "DMP Lookup/Maintain supports microbatches 0 and 1, got "
                f"{microbatch_idx}"
            )
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise RuntimeError(
                f"DMP Lookup/Maintain layer index is out of range: {layer_idx}"
            )

        ql_nope, _, topk_indices, _ = indexer_result
        query_index = self._normalize_topk(topk_indices)
        batch_size = int(query_index.shape[0])
        if batch_size > self.max_microbatch_tokens:
            raise RuntimeError(
                "DMP Lookup/Maintain microbatch exceeds configured capacity: "
                f"batch={batch_size}, capacity={self.max_microbatch_tokens}"
            )

        index_workspace = self.get_workspace(layer_idx, microbatch_idx)
        attention_workspace = self._get_or_create_attention_workspace(
            layer_idx, microbatch_idx, query_index, ql_nope, kv_cache
        )
        seq_lens = attn_metadata.seq_lens.to(dtype=torch.int32).contiguous()
        signature = attn_metadata.block_table[:, 0].to(dtype=torch.int32)
        previous_signature = index_workspace.request_signature[:batch_size]
        previous_seq_lens = index_workspace.previous_seq_lens[:batch_size]
        attention_workspace.needs_refill.copy_(
            (signature != previous_signature) | (seq_lens <= previous_seq_lens)
        )
        previous_signature.copy_(signature)
        previous_seq_lens.copy_(seq_lens)

        slot_out, miss_out, hit_sparse, miss_sparse, resident_token_ids = (
            self.custom_ops.asu_hbm_index_lookup(
                index_workspace.token_to_slot,
                index_workspace.slot_to_token,
                index_workspace.free_slots,
                index_workspace.free_head,
                index_workspace.req_pool_entries[:batch_size],
                query_index,
                seq_lens,
                attention_workspace.needs_refill,
                batch_size,
            )
        )
        index_workspace.slot_out = slot_out
        index_workspace.miss_out = miss_out
        index_workspace.query_index = query_index
        index_workspace.seq_lens = seq_lens
        index_workspace.resident_token_ids = resident_token_ids
        index_workspace.batch_size = batch_size
        attention_workspace.hit_sparse_indices.copy_(hit_sparse.unsqueeze(1))
        attention_workspace.miss_sparse_indices.copy_(miss_sparse.unsqueeze(1))

        attention_workspace.hit_softmax_max.fill_(torch.finfo(torch.float32).min)
        attention_workspace.hit_softmax_sum.zero_()
        attention_workspace.miss_softmax_max.fill_(torch.finfo(torch.float32).min)
        attention_workspace.miss_softmax_sum.zero_()
        attention_workspace.hit_attention_out = None

        if not self._activation_logged:
            logger.info(
                "DMP Lookup/Maintain active: fixed 300 misses, fused int32 "
                "sparse-index output, miss-only KVGather, combined mb0+mb1 "
                "hit SFA before Gather wait, three streams"
            )
            self._activation_logged = True
        return slot_out, miss_out

    def update(
        self,
        *,
        layer_idx: int,
        microbatch_idx: int,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility helper for standalone Lookup/Maintain tests."""
        query_index = self._normalize_topk(topk_indices)
        batch_size = int(query_index.shape[0])
        if batch_size > self.max_microbatch_tokens:
            raise RuntimeError(
                "DMP Lookup/Maintain microbatch exceeds configured capacity: "
                f"batch={batch_size}, capacity={self.max_microbatch_tokens}"
            )
        workspace = self.get_workspace(layer_idx, microbatch_idx)
        seq_lens = torch.full(
            (batch_size,), INDEX_CAPACITY, dtype=torch.int32, device=query_index.device
        )
        slot_out, miss_out, _, _, resident_token_ids = (
            self.custom_ops.asu_hbm_index_lookup(
                workspace.token_to_slot,
                workspace.slot_to_token,
                workspace.free_slots,
                workspace.free_head,
                workspace.req_pool_entries[:batch_size],
                query_index,
                seq_lens,
                torch.ones(batch_size, dtype=torch.bool, device=query_index.device),
                batch_size,
            )
        )
        workspace.slot_out = slot_out
        workspace.miss_out = miss_out
        workspace.query_index = query_index
        workspace.seq_lens = seq_lens
        workspace.resident_token_ids = resident_token_ids
        workspace.batch_size = batch_size
        self.maintain(layer_idx=layer_idx, microbatch_idx=microbatch_idx)
        return slot_out, miss_out

    def maintain(self, *, layer_idx: int, microbatch_idx: int) -> None:
        workspace = self.get_workspace(layer_idx, microbatch_idx)
        if workspace.slot_out is None or workspace.batch_size <= 0:
            raise RuntimeError("Lookup must run before Maintain")
        self.custom_ops.asu_hbm_index_maintain_aicpu(
            workspace.token_to_slot,
            workspace.slot_to_token,
            workspace.free_slots,
            workspace.free_head,
            workspace.req_pool_entries[: workspace.batch_size],
            workspace.slot_out,
            workspace.batch_size,
            layer_idx * len(SUPPORTED_MICROBATCHES) + microbatch_idx,
        )

    def gather(
        self,
        *,
        layer_idx: int,
        microbatch_idx: int,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: Any,
    ) -> None:
        workspace = self.get_attention_workspace(layer_idx, microbatch_idx)
        index_workspace = self.get_workspace(layer_idx, microbatch_idx)
        if (
            index_workspace.slot_out is None
            or index_workspace.miss_out is None
            or index_workspace.query_index is None
            or index_workspace.seq_lens is None
            or index_workspace.resident_token_ids is None
        ):
            raise RuntimeError("Lookup must run before KVGather")
        full_kv_cache = self._normalize_cache(kv_cache[0], "full KV cache")
        full_k_rope = self._normalize_cache(kv_cache[1], "full K-rope cache")
        self.custom_ops.dmp_lookup_kv_gather(
            workspace.selection_k_rope,
            workspace.selection_kv_cache,
            workspace.selection_kv_block_table,
            index_workspace.resident_token_ids,
            index_workspace.query_index,
            index_workspace.slot_out,
            index_workspace.miss_out,
            workspace.needs_refill,
            full_k_rope,
            full_kv_cache,
            attn_metadata.block_table,
            index_workspace.seq_lens,
        )

    def run_hit_attention(
        self,
        *,
        layer_idx: int,
        microbatch_idx: int,
        indexer_result: tuple[torch.Tensor, ...],
        scale: float,
        attn_metadata: Any,
    ) -> None:
        ql_nope, q_pe, _, _ = indexer_result
        workspace = self.get_attention_workspace(layer_idx, microbatch_idx)
        workspace.hit_attention_out = self._unwrap_operator_output(
            self.attention_ops.npu_dmp_sparse_flash_attention(
                ql_nope,
                workspace.full_kv_cache.unsqueeze(2),
                workspace.full_kv_cache.unsqueeze(2),
                workspace.hit_sparse_indices,
                scale,
                SELECTION_TOPK_BLOCK_SIZE,
                block_table=attn_metadata.block_table,
                actual_seq_lengths_query=attn_metadata.cum_query_lens,
                actual_seq_lengths_kv=attn_metadata.seq_lens,
                query_rope=q_pe,
                key_rope=workspace.full_k_rope.unsqueeze(2),
                softmax_max_out=workspace.hit_softmax_max,
                softmax_sum_out=workspace.hit_softmax_sum,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )
        )

    def run_miss_attention_and_merge(
        self,
        *,
        layer_idx: int,
        microbatch_idx: int,
        indexer_result: tuple[torch.Tensor, ...],
        scale: float,
        attn_metadata: Any,
    ) -> torch.Tensor:
        ql_nope, q_pe, _, _ = indexer_result
        workspace = self.get_attention_workspace(layer_idx, microbatch_idx)
        if workspace.hit_attention_out is None:
            raise RuntimeError("Hit attention must run before miss attention")
        miss_attention_out = self._unwrap_operator_output(
            self.attention_ops.npu_dmp_sparse_flash_attention(
                ql_nope,
                workspace.selection_kv_cache.unsqueeze(2),
                workspace.selection_kv_cache.unsqueeze(2),
                workspace.miss_sparse_indices,
                scale,
                SELECTION_TOPK_BLOCK_SIZE,
                block_table=workspace.selection_kv_block_table,
                actual_seq_lengths_query=attn_metadata.cum_query_lens,
                actual_seq_lengths_kv=workspace.selected_actual_seq,
                query_rope=q_pe,
                key_rope=workspace.selection_k_rope.unsqueeze(2),
                softmax_max_out=workspace.miss_softmax_max,
                softmax_sum_out=workspace.miss_softmax_sum,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )
        )
        return self._unwrap_operator_output(
            self.attention_ops.npu_da_attention_merge(
                workspace.hit_attention_out,
                workspace.hit_softmax_max,
                workspace.hit_softmax_sum,
                miss_attention_out,
                workspace.miss_softmax_max,
                workspace.miss_softmax_sum,
            )
        )

    def prepare_combined_attention(
        self,
        *,
        layer_idx: int,
        indexer_results: tuple[tuple[torch.Tensor, ...], ...],
        attn_metadata: tuple[Any, Any],
    ) -> tuple[torch.Tensor, ...]:
        """Merge mb0+mb1 query inputs without waiting for KV Gather."""
        if len(indexer_results) != len(SUPPORTED_MICROBATCHES):
            raise RuntimeError("Combined attention requires exactly two microbatches")
        try:
            pair_key = self._active_attention_pair_keys[layer_idx]
            self._attention_pair_workspaces[pair_key]
        except KeyError as exc:
            raise RuntimeError(
                f"Combined attention workspace is unavailable for layer={layer_idx}"
            ) from exc
        for microbatch_idx in SUPPORTED_MICROBATCHES:
            local_key = self._active_attention_keys.get(
                (layer_idx, microbatch_idx)
            )
            if local_key is None or local_key[0] != pair_key:
                raise RuntimeError(
                    "Lookup must run for mb0 and mb1 before combined attention"
                )

        ql_nope = torch.cat(
            [indexer_result[0] for indexer_result in indexer_results], dim=0
        )
        q_pe = torch.cat(
            [indexer_result[1] for indexer_result in indexer_results], dim=0
        )
        query_offset = indexer_results[0][0].shape[0]
        actual_seq_lengths_query = torch.cat(
            [
                attn_metadata[0].cum_query_lens,
                attn_metadata[1].cum_query_lens + query_offset,
            ]
        )
        full_block_table = torch.cat(
            [metadata.block_table for metadata in attn_metadata], dim=0
        )
        full_seq_lens = torch.cat(
            [metadata.seq_lens for metadata in attn_metadata], dim=0
        )
        return (
            ql_nope,
            q_pe,
            actual_seq_lengths_query,
            full_block_table,
            full_seq_lens,
        )

    def run_combined_hit_attention(
        self,
        *,
        layer_idx: int,
        indexer_results: tuple[tuple[torch.Tensor, ...], ...],
        scale: float,
        attn_metadata: tuple[Any, Any],
        prepared_inputs: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        """Run the combined hit SFA without waiting for miss KVGather."""
        try:
            pair_key = self._active_attention_pair_keys[layer_idx]
            pair = self._attention_pair_workspaces[pair_key]
        except KeyError as exc:
            raise RuntimeError(
                f"Combined attention workspace is unavailable for layer={layer_idx}"
            ) from exc

        if prepared_inputs is None:
            prepared_inputs = self.prepare_combined_attention(
                layer_idx=layer_idx,
                indexer_results=indexer_results,
                attn_metadata=attn_metadata,
            )
        (
            ql_nope,
            q_pe,
            actual_seq_lengths_query,
            full_block_table,
            full_seq_lens,
        ) = prepared_inputs
        pair.hit_attention_out = self._unwrap_operator_output(
            self.attention_ops.npu_dmp_sparse_flash_attention(
                ql_nope,
                pair.full_kv_cache.unsqueeze(2),
                pair.full_kv_cache.unsqueeze(2),
                pair.hit_sparse_indices,
                scale,
                SELECTION_TOPK_BLOCK_SIZE,
                block_table=full_block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=full_seq_lens,
                query_rope=q_pe,
                key_rope=pair.full_k_rope.unsqueeze(2),
                softmax_max_out=pair.hit_softmax_max,
                softmax_sum_out=pair.hit_softmax_sum,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )
        )

    def run_combined_miss_attention_and_merge(
        self,
        *,
        layer_idx: int,
        indexer_results: tuple[tuple[torch.Tensor, ...], ...],
        scale: float,
        attn_metadata: tuple[Any, Any],
        prepared_inputs: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Waited phase: run miss SFA from staging KV, then merge."""
        try:
            pair_key = self._active_attention_pair_keys[layer_idx]
            pair = self._attention_pair_workspaces[pair_key]
        except KeyError as exc:
            raise RuntimeError(
                f"Combined attention workspace is unavailable for layer={layer_idx}"
            ) from exc
        if pair.hit_attention_out is None:
            raise RuntimeError("Combined hit attention must run before miss attention")

        if prepared_inputs is None:
            prepared_inputs = self.prepare_combined_attention(
                layer_idx=layer_idx,
                indexer_results=indexer_results,
                attn_metadata=attn_metadata,
            )
        ql_nope, q_pe, actual_seq_lengths_query, _, _ = prepared_inputs
        miss_attention_out = self._unwrap_operator_output(
            self.attention_ops.npu_dmp_sparse_flash_attention(
                ql_nope,
                pair.selection_kv_cache.unsqueeze(2),
                pair.selection_kv_cache.unsqueeze(2),
                pair.miss_sparse_indices,
                scale,
                SELECTION_TOPK_BLOCK_SIZE,
                block_table=pair.selection_kv_block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=pair.selected_actual_seq,
                query_rope=q_pe,
                key_rope=pair.selection_k_rope.unsqueeze(2),
                softmax_max_out=pair.miss_softmax_max,
                softmax_sum_out=pair.miss_softmax_sum,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )
        )
        return self._unwrap_operator_output(
            self.attention_ops.npu_da_attention_merge(
                pair.hit_attention_out,
                pair.hit_softmax_max,
                pair.hit_softmax_sum,
                miss_attention_out,
                pair.miss_softmax_max,
                pair.miss_softmax_sum,
            )
        )

    def run_combined_attention(
        self,
        *,
        layer_idx: int,
        indexer_results: tuple[tuple[torch.Tensor, ...], ...],
        scale: float,
        attn_metadata: tuple[Any, Any],
        prepared_inputs: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Run hit/miss SFA once for the combined mb0+mb1 indexer result."""
        if prepared_inputs is None:
            prepared_inputs = self.prepare_combined_attention(
                layer_idx=layer_idx,
                indexer_results=indexer_results,
                attn_metadata=attn_metadata,
            )
        self.run_combined_hit_attention(
            layer_idx=layer_idx,
            indexer_results=indexer_results,
            scale=scale,
            attn_metadata=attn_metadata,
            prepared_inputs=prepared_inputs,
        )
        return self.run_combined_miss_attention_and_merge(
            layer_idx=layer_idx,
            indexer_results=indexer_results,
            scale=scale,
            attn_metadata=attn_metadata,
            prepared_inputs=prepared_inputs,
        )
