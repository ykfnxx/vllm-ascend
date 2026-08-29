# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.quantization.methods import (
    AscendW4A8DynamicLinearMethod,
    AscendW8A8DynamicLinearMethod,
    AscendW8A8LinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)

from .config import DSAOffloadConfig
from .constants import INDEX_CAPACITY, QUERY_WIDTH
from .io import make_storage_ids
from .lookup import (
    DSAOffloadBatch,
    IndexCacheCohort,
    load_prefetch_misses,
    make_prefetch_lookup_plan,
)
from .prefetch_coefficients import (
    PREFETCH_GROUP_SIZE,
    PREFETCH_HI_BLOCK_NUM,
    PredictionCoefficientProfile,
    apply_group_predict_coefficients,
    get_active_prefetch_groups,
    get_group_predict_coefficients,
    get_prediction_coefficient_profile,
    pad_prefetch_topk,
)

_QLI_GLM52_CONTRACT = (6144, 2048, 512, 32, 128, 64)
_INVALID_STORAGE_ID = -1


def _linear_method(layer: torch.nn.Module | None) -> Any:
    quant_method = getattr(layer, "quant_method", None)
    return getattr(quant_method, "quant_method", quant_method)


def _linear_output(layer: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output = layer(inputs)
    return output[0] if isinstance(output, tuple) else output


def _is_supported_projection(layer: torch.nn.Module | None) -> bool:
    return isinstance(
        _linear_method(layer),
        (
            AscendW4A8DynamicLinearMethod,
            AscendW8A8DynamicLinearMethod,
            AscendW8A8LinearMethod,
            AscendW8A8MXFP8DynamicLinearMethod,
            UnquantizedLinearMethod,
        ),
    )


def _indexer_cache_parts(
    impl: object,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    indexer = getattr(impl, "indexer", None)
    cache = tuple(indexer.k_cache.kv_cache)
    base_count = 2 if getattr(impl, "enable_sparse_li_c8", False) else 1
    if len(cache) not in {base_count, base_count + 1}:
        raise RuntimeError(
            f"Unexpected Indexer cache layout for {getattr(impl, 'layer_name', None)!r}."
        )
    return cache[:base_count], (
        cache[-1] if len(cache) == base_count + 1 else None
    )


def _prepare_qli(impl: object) -> tuple[bool, torch.Tensor | None]:
    fused_qkv = getattr(impl, "fused_qkv_a_proj", None)
    wq_b = getattr(impl, "wq_b", None)
    wk_weights_proj = getattr(impl, "wk_weights_proj", None)
    q_norm = getattr(impl, "q_a_layernorm", None)
    if not isinstance(_linear_method(fused_qkv), AscendW8A8DynamicLinearMethod):
        return False, None
    if not isinstance(_linear_method(wq_b), AscendW8A8DynamicLinearMethod):
        return False, None
    if not isinstance(_linear_method(wk_weights_proj), UnquantizedLinearMethod):
        return False, None
    if get_ascend_config().weight_nz_mode != 1 or getattr(impl, "is_rope_neox_style", True):
        return False, None
    if fused_qkv is None or wq_b is None or wk_weights_proj is None or q_norm is None:
        return False, None
    tensors = (
        getattr(fused_qkv, "weight", None),
        getattr(fused_qkv, "weight_scale", None),
        getattr(wq_b, "weight", None),
        getattr(wq_b, "weight_scale", None),
        getattr(wk_weights_proj, "weight", None),
        getattr(q_norm, "weight", None),
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False, None
    qkv_weight, qkv_scale, qb_weight, qb_scale, wk_weight, norm_weight = tensors
    assert isinstance(qkv_weight, torch.Tensor)
    assert isinstance(qkv_scale, torch.Tensor)
    assert isinstance(qb_weight, torch.Tensor)
    assert isinstance(qb_scale, torch.Tensor)
    assert isinstance(wk_weight, torch.Tensor)
    assert isinstance(norm_weight, torch.Tensor)
    contract = (
        int(wk_weight.shape[-1]) if wk_weight.ndim == 2 else 0,
        getattr(impl, "q_lora_rank", None),
        getattr(impl, "kv_lora_rank", None),
        getattr(impl, "n_head", None),
        getattr(impl, "head_dim", None),
        getattr(impl, "qk_rope_head_dim", None),
    )
    if contract != _QLI_GLM52_CONTRACT:
        return False, None
    qkv_size = contract[1] + contract[2] + contract[5]
    query_size = contract[3] * contract[4]
    if (
        tuple(qkv_weight.shape) != (contract[0], qkv_size)
        or tuple(qb_weight.shape) != (contract[1], query_size)
        or tuple(wk_weight.shape) != (contract[4] + contract[3], contract[0])
        or tuple(norm_weight.shape) != (contract[1],)
        or qkv_weight.dtype != torch.int8
        or qkv_scale.dtype != torch.bfloat16
        or qb_weight.dtype != torch.int8
        or qb_scale.dtype != torch.bfloat16
        or wk_weight.dtype != torch.bfloat16
        or norm_weight.dtype != torch.bfloat16
        or qkv_scale.numel() != qkv_size
        or qb_scale.numel() != query_size
        or getattr(fused_qkv, "bias", None) is not None
        or getattr(wq_b, "bias", None) is not None
        or getattr(wk_weights_proj, "bias", None) is not None
        or getattr(wq_b, "_chunk_size", 0)
    ):
        return False, None
    norm_bias = getattr(q_norm, "bias", None)
    if norm_bias is None or not getattr(q_norm, "bias_loaded", False):
        norm_bias = torch.zeros_like(norm_weight)
    else:
        norm_bias = norm_bias.data.to(dtype=torch.bfloat16).contiguous()
    return True, norm_bias


@dataclass(frozen=True)
class _PredictionTarget:
    impl: object
    tp_rank: int
    alpha: torch.Tensor
    beta: torch.Tensor
    local_alpha: torch.Tensor
    local_beta: torch.Tensor
    qli_enabled: bool
    qli_norm_bias: torch.Tensor | None

    def compute_topk(
        self,
        *,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: DSAOffloadBatch,
        block_table: torch.Tensor,
        source_rows_before_gather: int | None,
        prefetch_top_k: int,
    ) -> torch.Tensor:
        packed = batch.packed_decode
        if packed is None:
            raise RuntimeError("Grouped prefetch requires packed Decode metadata.")
        impl = self.impl
        if self.qli_enabled and hidden_states.dtype == torch.bfloat16:
            q_norm = impl.q_a_layernorm
            if self.qli_norm_bias is None:
                raise RuntimeError("Prefetch QLI RMSNorm bias was not initialized.")
            rope_width = impl.qk_rope_head_dim // 2
            ranked = source_rows_before_gather is not None and self.alpha.numel() > 1
            query, weights = torch.ops._C_ascend.prefetch_qli_fusion(
                hidden_states,
                impl.fused_qkv_a_proj.weight,
                impl.fused_qkv_a_proj.weight_scale,
                impl.wq_b.weight,
                impl.wq_b.weight_scale,
                q_norm.weight,
                self.qli_norm_bias,
                cos.reshape(hidden_states.shape[0], -1)[:, :rope_width].contiguous(),
                sin.reshape(hidden_states.shape[0], -1)[:, :rope_width].contiguous(),
                impl.wk_weights_proj.weight,
                impl.q_lora_rank,
                impl.n_head,
                impl.head_dim,
                impl.qk_rope_head_dim,
                1.0,
                0.0,
                q_norm.variance_epsilon,
                source_rows_before_gather if ranked else 0,
                self.alpha if ranked else self.local_alpha,
                self.beta if ranked else self.local_beta,
            )
            query = query.index_select(0, packed.token_indices)
            weights = weights.index_select(0, packed.token_indices)
        else:
            predicted_hidden = apply_group_predict_coefficients(
                hidden_states,
                self.alpha,
                self.beta,
                tp_rank=self.tp_rank,
                source_rows_before_gather=source_rows_before_gather,
            ).index_select(0, packed.token_indices).contiguous()
            packed_cos = cos.index_select(0, packed.token_indices).contiguous()
            packed_sin = sin.index_select(0, packed.token_indices).contiguous()
            qkv_lora = _linear_output(impl.fused_qkv_a_proj, predicted_hidden)
            q_c = impl.q_a_layernorm(qkv_lora[..., : impl.q_lora_rank])
            kw = _linear_output(impl.wk_weights_proj, predicted_hidden)
            weights = kw[:, impl.head_dim :]
            query = _linear_output(impl.wq_b, q_c).view(
                -1,
                impl.n_head,
                impl.head_dim,
            )
            if HAS_TRITON:
                query = rope_forward_triton_siso(
                    query,
                    packed_cos,
                    packed_sin,
                    rope_dim=impl.qk_rope_head_dim,
                    is_neox_style=impl.is_rope_neox_style,
                )
            else:
                query_rope, query_nope = torch.split(
                    query,
                    [impl.qk_rope_head_dim, impl.head_dim - impl.qk_rope_head_dim],
                    dim=-1,
                )
                rope_cos = packed_cos.view(-1, 1, 1, impl.qk_rope_head_dim)
                rope_sin = packed_sin.view(-1, 1, 1, impl.qk_rope_head_dim)
                query_rope = query_rope.unsqueeze(2)
                if impl.is_rope_neox_style:
                    query_rope = torch_npu.npu_rotary_mul(
                        query_rope,
                        rope_cos,
                        rope_sin,
                    )
                else:
                    query_rope = torch_npu.npu_interleave_rope(
                        query_rope,
                        rope_cos,
                        rope_sin,
                    )
                query = torch.cat(
                    (query_rope.squeeze(2), query_nope),
                    dim=-1,
                )

        base_cache, key_mean = _indexer_cache_parts(impl)
        query_lengths = packed.query_start_loc[1:]
        historical_lengths = packed.query_positions[
            packed.query_start_loc[:-1].to(torch.int64)
        ].to(torch.int32)
        decode_block_table = block_table.index_select(
            0,
            packed.request_indices.to(torch.int64),
        )
        if impl.enable_sparse_li_c8:
            if key_mean is not None or len(base_cache) != 2:
                raise RuntimeError("C8 grouped prefetch requires key and scale cache only.")
            query_shape = query.shape
            query = query @ type(impl).q_hadamard
            query, query_scale = torch_npu.npu_dynamic_quant(
                query.view(-1, impl.head_dim),
                dst_type=impl.c8_k_cache_dtype,
            )
            topk_indices = torch_npu.npu_quant_lightning_indexer(
                query=query.view(query_shape),
                key=base_cache[0],
                weights=weights,
                query_dequant_scale=query_scale.to(
                    impl.c8_k_scale_cache_dtype
                ).view(query_shape[:-1]),
                key_dequant_scale=base_cache[1].squeeze(2),
                actual_seq_lengths_query=query_lengths,
                actual_seq_lengths_key=historical_lengths,
                block_table=decode_block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=prefetch_top_k,
                sparse_mode=3,
            )
        else:
            if key_mean is None:
                raise RuntimeError("HiCached grouped prefetch requires key mean cache.")
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer_hi_cached(
                query=query,
                key=base_cache[0],
                weights=weights,
                key_mean=key_mean,
                actual_seq_lengths_query=query_lengths,
                actual_seq_lengths_key=historical_lengths,
                block_table=decode_block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=prefetch_top_k,
                sparse_mode=3,
                hi_block_size=int(base_cache[0].shape[1]),
                hi_block_num=PREFETCH_HI_BLOCK_NUM,
                block_pooling_mode="mean",
            )
        topk_indices = pad_prefetch_topk(topk_indices, prefetch_top_k)
        if topk_indices.ndim == 3 and topk_indices.shape[1] == 1:
            topk_indices = topk_indices.squeeze(1)
        return topk_indices.to(torch.int32).contiguous()


@dataclass
class _PreparedPrefetch:
    source_layer_name: str
    batch: DSAOffloadBatch
    topk_indices: torch.Tensor
    compute_done_event: object | None


class GroupedPrefetchController:
    def __init__(
        self,
        *,
        source_layer_name: str,
        target_cohort: IndexCacheCohort,
        target: _PredictionTarget,
        storage_ids: Mapping[int, torch.Tensor],
        stream: object | None,
    ) -> None:
        if len(target_cohort.layer_ids) != PREFETCH_GROUP_SIZE:
            raise ValueError("Grouped prefetch targets must contain four physical layers.")
        self.source_layer_name = source_layer_name
        self.target_cohort = target_cohort
        self.target = target
        self._storage_ids = storage_ids
        self._stream = stream
        self._prepared: _PreparedPrefetch | None = None
        self._compute_done_event: object | None = None
        self._ready_event: object | None = None
        self._ready_without_event = False

    def start(
        self,
        *,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: DSAOffloadBatch,
        block_table: torch.Tensor,
        source_rows_before_gather: int | None,
        prefetch_top_k: int,
    ) -> None:
        if self._prepared is not None or self._ready_event is not None:
            raise RuntimeError(
                f"Overlapping grouped prefetch for {self.target_cohort.leader_layer!r}."
            )

        def compute_topk() -> torch.Tensor:
            return self.target.compute_topk(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                batch=batch,
                block_table=block_table,
                source_rows_before_gather=source_rows_before_gather,
                prefetch_top_k=prefetch_top_k,
            )

        compute_done_event = None
        if self._stream is not None:
            self._stream.wait_event(torch.npu.current_stream().record_event())
            with torch.npu.stream(self._stream):
                topk_indices = compute_topk()
                compute_done_event = torch.npu.current_stream().record_event()
            topk_indices.record_stream(self._stream)
            for tensor in (
                hidden_states,
                cos,
                sin,
                block_table,
                self.target.alpha,
                self.target.beta,
            ):
                if tensor.device.type == "npu":
                    tensor.record_stream(self._stream)
        else:
            topk_indices = compute_topk()
        if topk_indices.shape != (
            batch.packed_decode.token_indices.shape[0],
            QUERY_WIDTH,
        ):
            raise RuntimeError(
                f"Grouped prefetch Top-K has invalid shape {tuple(topk_indices.shape)}."
            )
        self._prepared = _PreparedPrefetch(
            source_layer_name=self.source_layer_name,
            batch=batch,
            topk_indices=topk_indices,
            compute_done_event=compute_done_event,
        )
        self._compute_done_event = compute_done_event

    def wait_for_compute_before_key_write(self) -> None:
        if self._compute_done_event is not None:
            torch.npu.current_stream().wait_event(self._compute_done_event)
            self._compute_done_event = None

    def release_after_exact_load(self, layer_name: str) -> None:
        prepared = self._prepared
        if prepared is None or layer_name != self.source_layer_name:
            return

        def run_prefetch() -> None:
            plan = make_prefetch_lookup_plan(
                semantic_topk=prepared.topk_indices,
                cohort=self.target_cohort,
                batch=prepared.batch,
            )
            for layer_id in self.target_cohort.layer_ids:
                load_prefetch_misses(
                    plan,
                    layer_id,
                    prepared.batch,
                    self._storage_ids[layer_id],
                )

        ready_event = None
        if self._stream is not None:
            self._stream.wait_event(torch.npu.current_stream().record_event())
            if prepared.compute_done_event is not None:
                self._stream.wait_event(prepared.compute_done_event)
            with torch.npu.stream(self._stream):
                run_prefetch()
                ready_event = torch.npu.current_stream().record_event()
        else:
            run_prefetch()
            self._ready_without_event = True
        self._prepared = None
        self._ready_event = ready_event

    def wait_before_exact_lookup(self) -> None:
        if self._prepared is not None:
            raise RuntimeError(
                f"Grouped prefetch for {self.target_cohort.leader_layer!r} was not released."
            )
        if self._ready_event is not None:
            torch.npu.current_stream().wait_event(self._ready_event)
        elif not self._ready_without_event:
            raise RuntimeError(
                f"Grouped prefetch for {self.target_cohort.leader_layer!r} is not ready."
            )
        self._ready_event = None
        self._ready_without_event = False


class GroupedPrefetchRuntime:
    def __init__(
        self,
        *,
        config: DSAOffloadConfig,
        num_hidden_layers: int,
        ordered_layers: Sequence[tuple[str, int, object]],
        cohorts: Sequence[IndexCacheCohort],
        quant_config: object | None,
        max_num_seqs: int,
        block_size: int,
        device: torch.device | str,
    ) -> None:
        self.config = config
        active_groups = get_active_prefetch_groups(num_hidden_layers)
        layers_by_id = {
            layer_id: (layer_name, impl)
            for layer_name, layer_id, impl in ordered_layers
        }
        cohorts_by_first_layer = {
            cohort.layer_ids[0]: cohort for cohort in cohorts
        }
        self.target_layer_names = frozenset(
            layers_by_id[target][0] for target in active_groups.values()
        )
        target_layer_ids = (
            tuple(
                layer_id
                for target in active_groups.values()
                for layer_id in range(target, target + PREFETCH_GROUP_SIZE)
            )
            if config.is_consumer
            else ()
        )
        max_blocks = (
            INDEX_CAPACITY + block_size - 1
        ) // block_size
        self.storage_ids = {
            layer_id: torch.full(
                (max_num_seqs, max_blocks),
                _INVALID_STORAGE_ID,
                dtype=torch.int64,
                device=device,
            )
            for layer_id in target_layer_ids
        }
        self._row_hashes: list[tuple[bytes, ...] | None] = [
            None
        ] * max_num_seqs
        self._prefetch_stream: object | None = None
        self._outgoing: dict[str, GroupedPrefetchController] = {}
        self._incoming: dict[str, GroupedPrefetchController] = {}
        if not config.is_consumer:
            return
        sample = next(iter(self.storage_ids.values()), None)
        # Allocate once before graph capture. Every group serializes prediction,
        # fused prefetch LookupUpdate and Gather on this one graph-stable stream.
        self._prefetch_stream = (
            torch.npu.Stream()
            if sample is not None and sample.device.type == "npu"
            else None
        )
        profile = get_prediction_coefficient_profile(quant_config)
        if not isinstance(profile, PredictionCoefficientProfile):
            raise RuntimeError(
                "Grouped prefetch cannot determine GLM-5.2 prediction coefficients."
            )
        for source_layer_id, target_layer_id in active_groups.items():
            try:
                source_name, source_impl = layers_by_id[source_layer_id]
                target_name, target_impl = layers_by_id[target_layer_id]
                cohort = cohorts_by_first_layer[target_layer_id]
            except KeyError as error:
                raise RuntimeError(
                    f"Cannot bind grouped prefetch {source_layer_id}->{target_layer_id}."
                ) from error
            if (
                not getattr(source_impl, "has_indexer", False)
                or getattr(source_impl, "skip_topk", False)
                or not getattr(target_impl, "has_indexer", False)
                or getattr(target_impl, "skip_topk", False)
                or not _is_supported_projection(
                    getattr(target_impl, "fused_qkv_a_proj", None)
                )
            ):
                raise RuntimeError(
                    f"Grouped prefetch requires source and target Indexer leaders for {source_name}->{target_name}."
                )
            alpha_values, beta_values = get_group_predict_coefficients(
                profile,
                source_layer_id,
            )
            tp_rank = get_tp_group().rank_in_group
            if not 0 <= tp_rank < len(alpha_values):
                raise RuntimeError(
                    f"Grouped prefetch TP rank {tp_rank} is outside "
                    f"coefficient width {len(alpha_values)}."
                )
            q_norm = getattr(target_impl, "q_a_layernorm", None)
            coefficient_weight = getattr(q_norm, "weight", None)
            if not isinstance(coefficient_weight, torch.Tensor):
                raise RuntimeError(
                    f"Grouped prefetch target {target_name!r} has no Q-A "
                    "layer-normalization weight."
                )
            alpha = coefficient_weight.new_tensor(alpha_values)
            beta = coefficient_weight.new_tensor(beta_values)
            qli_enabled, qli_norm_bias = _prepare_qli(target_impl)
            target = _PredictionTarget(
                impl=target_impl,
                tp_rank=tp_rank,
                alpha=alpha,
                beta=beta,
                local_alpha=alpha[tp_rank : tp_rank + 1].contiguous(),
                local_beta=beta[tp_rank : tp_rank + 1].contiguous(),
                qli_enabled=qli_enabled,
                qli_norm_bias=qli_norm_bias,
            )
            controller = GroupedPrefetchController(
                source_layer_name=source_name,
                target_cohort=cohort,
                target=target,
                storage_ids=self.storage_ids,
                stream=self._prefetch_stream,
            )
            self._outgoing[source_name] = controller
            self._incoming[target_name] = controller

    @property
    def fixed_memory_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.storage_ids.values())

    def update_storage_ids(self, batch: DSAOffloadBatch) -> None:
        if not self.storage_ids or batch.hot_cache is None:
            return
        for request_index in batch.decode_request_indices:
            request_id = batch.request_ids[request_index]
            row_id = batch.hot_cache.request_to_row[request_id]
            block_hashes = tuple(batch.block_hashes(request_index))
            if self._row_hashes[row_id] == block_hashes:
                continue
            previous_count = len(self._row_hashes[row_id] or ())
            for layer_id, table in self.storage_ids.items():
                ids = make_storage_ids(
                    block_hashes,
                    layer_id,
                    device=table.device,
                )
                table[row_id, : ids.numel()].copy_(ids)
                if ids.numel() < previous_count:
                    table[row_id, ids.numel() : previous_count].fill_(
                        _INVALID_STORAGE_ID
                    )
            self._row_hashes[row_id] = block_hashes

    def clear_request_row(self, row_id: int) -> None:
        for table in self.storage_ids.values():
            table[row_id].fill_(_INVALID_STORAGE_ID)
        self._row_hashes[row_id] = None

    def start(
        self,
        *,
        layer_name: str,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: DSAOffloadBatch,
        block_table: torch.Tensor,
        source_rows_before_gather: int | None,
    ) -> None:
        controller = self._outgoing.get(layer_name)
        if controller is None or not batch.decode_request_indices:
            return
        logical_rows = batch.query_positions.shape[0]
        controller.start(
            hidden_states=hidden_states[:logical_rows],
            cos=cos[:logical_rows].contiguous(),
            sin=sin[:logical_rows].contiguous(),
            batch=batch,
            block_table=block_table,
            source_rows_before_gather=source_rows_before_gather,
            prefetch_top_k=self.config.prefetch_top_k,
        )

    def wait_for_compute_before_key_write(self, layer_name: str) -> None:
        controller = self._incoming.get(layer_name)
        if controller is not None:
            controller.wait_for_compute_before_key_write()

    def wait_before_exact_lookup(self, layer_name: str) -> None:
        controller = self._incoming.get(layer_name)
        if controller is not None:
            controller.wait_before_exact_lookup()

    def release_after_exact_load(self, layer_name: str) -> None:
        controller = self._outgoing.get(layer_name)
        if controller is not None:
            controller.release_after_exact_load(layer_name)


def create_grouped_prefetch_runtime(
    *,
    config: DSAOffloadConfig,
    num_hidden_layers: int,
    ordered_layers: Sequence[tuple[str, int, object]],
    cohorts: Sequence[IndexCacheCohort],
    quant_config: object | None,
    max_num_seqs: int,
    block_size: int,
    device: torch.device | str,
) -> GroupedPrefetchRuntime | None:
    if not config.enable_prefetch_with_hidden_states:
        return None
    return GroupedPrefetchRuntime(
        config=config,
        num_hidden_layers=num_hidden_layers,
        ordered_layers=ordered_layers,
        cohorts=cohorts,
        quant_config=quant_config,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        device=device,
    )
