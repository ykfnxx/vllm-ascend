"""Dual-Attention (DA): two segmented sparse-attention calls with exported state.

- Attn0 (hit): ``softmax_max_out``, ``softmax_sum_out``
- Attn1 (miss): ``softmax_max_out``, ``softmax_sum_out``
- Merge: fused ``DaAttentionMerge`` custom op when available; Python ``lse_merge`` fallback.

Requires custom SFA with optional LSE buffers plus the ``DaAttentionMerge`` OPP.
Pipeline config: ``BaselineConfig`` (same as baseline indexer/gather/SFA shapes).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch_npu

from baseline import (
    BaselineConfig,
    BaselineRuntime,
    blend_indexer_topk_with_reuse,
    prepare_gather_step,
)


class DualAttentionOutputs(NamedTuple):
    attention_out: torch.Tensor
    attention_out_0: torch.Tensor
    attention_out_1: torch.Tensor
    hit_count: int
    miss_count: int


@dataclass(frozen=True)
class DualAttentionStepMetrics:
    step_id: int
    indexer_ms: float
    gather_ms: float
    sparse_attn_ms: float
    step_ms: float
    attn0_ms: float = 0.0
    attn1_ms: float = 0.0
    hit_count: int = 0
    miss_count: int = 0


def infer_hit_mask_from_block_status(
    selection_topk_indices: torch.Tensor,
    selection_kv_block_status: torch.Tensor,
) -> torch.Tensor:
    """Return ``[T, H_gather, K]`` hit mask; supports indexer ``[T, H_idx, K]`` vs pool ``[T, H_gather, K]``."""
    pool_ids = selection_kv_block_status[..., :-1]
    topk = selection_topk_indices
    if topk.dim() == 2:
        topk = topk.unsqueeze(1)
    if topk.shape[-2] != pool_ids.shape[-2]:
        return (pool_ids.unsqueeze(1) == topk.unsqueeze(-2)).any(dim=-3)
    return (pool_ids.unsqueeze(-2) == topk.unsqueeze(-2)).any(dim=-2)


def pack_sparse_indices(
    column_indices: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack selected columns into sparse indices for SFA.

    Stable compaction: keep natural column order (0..topk-1), move valid entries to the
    front and pad trailing slots with -1. Do not sort by index value (that changes softmax).
    """
    if column_indices.shape != mask.shape:
        raise ValueError(f"shape mismatch: {column_indices.shape} vs {mask.shape}")
    k = column_indices.shape[-1]
    vals = torch.where(mask, column_indices, torch.full_like(column_indices, -1))
    # Valid slots sort before invalid; stable argsort preserves left-to-right column order.
    ar = torch.arange(k, device=mask.device, dtype=torch.int64).view(
        *([1] * (mask.dim() - 1)), k
    )
    sort_key = torch.where(mask, ar, ar + k)
    order = sort_key.argsort(dim=-1, stable=True)
    packed = vals.gather(-1, order.to(vals.dtype))
    counts = mask.sum(dim=-1).amax(dim=-1).to(torch.int32)
    return packed.to(torch.int32), counts


def lse_merge(
    o0: torch.Tensor,
    o1: torch.Tensor,
    m0: torch.Tensor,
    s0: torch.Tensor,
    m1: torch.Tensor,
    s1: torch.Tensor,
) -> torch.Tensor:
    """Merge two partial attention outputs using exported softmax max/sum (Flash-style)."""
    if o0.dim() == 4 and m0.dim() == 2:
        batch_size, seq_len, head_num, _ = o0.shape
        state_shape = (batch_size, seq_len, head_num)
        m0 = m0.reshape(state_shape)
        s0 = s0.reshape(state_shape)
        m1 = m1.reshape(state_shape)
        s1 = s1.reshape(state_shape)
    m = torch.maximum(m0, m1)
    w0 = s0 * torch.exp(m0 - m)
    w1 = s1 * torch.exp(m1 - m)
    denom = (w0 + w1).clamp_min(1e-12)
    while w0.dim() < o0.dim():
        w0 = w0.unsqueeze(-1)
        w1 = w1.unsqueeze(-1)
        denom = denom.unsqueeze(-1)
    return (o0 * w0 + o1 * w1) / denom


def da_attention_merge(
    o0: torch.Tensor,
    o1: torch.Tensor,
    m0: torch.Tensor,
    s0: torch.Tensor,
    m1: torch.Tensor,
    s1: torch.Tensor,
) -> torch.Tensor:
    try:
        import custom_ops  # noqa: F401

        merge_op = torch.ops.custom.npu_da_attention_merge
    except (ImportError, AttributeError):
        merge_op = None
    if merge_op is None:
        return lse_merge(o0.float(), o1.float(), m0.float(), s0.float(), m1.float(), s1.float()).to(o0.dtype)
    return merge_op(o0, m0, s0, o1, m1, s1)


def run_sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    *,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    sparse_block_size: int = 1,
    layout_query: str = "TND",
    layout_kv: str = "PA_BSND",
    sparse_mode: int = 3,
    attention_mode: int = 2,
    prior_softmax_max: torch.Tensor | None = None,
    prior_softmax_sum: torch.Tensor | None = None,
    prior_attention_out: torch.Tensor | None = None,
    softmax_max_out: torch.Tensor | None = None,
    softmax_sum_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """State tensors → custom SFA; otherwise ``torch_npu`` (e.g. full-KV reference in experiments)."""
    kw: dict[str, Any] = {
        "sparse_block_size": sparse_block_size,
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "sparse_mode": sparse_mode,
    }
    if query_rope is not None:
        kw["query_rope"] = query_rope
        kw["key_rope"] = key_rope
    if block_table is not None:
        kw["block_table"] = block_table
    if actual_seq_lengths_query is not None:
        kw["actual_seq_lengths_query"] = actual_seq_lengths_query
    if actual_seq_lengths_kv is not None:
        kw["actual_seq_lengths_kv"] = actual_seq_lengths_kv
    for name, tensor in (
        ("prior_softmax_max", prior_softmax_max),
        ("prior_softmax_sum", prior_softmax_sum),
        ("prior_attention_out", prior_attention_out),
        ("softmax_max_out", softmax_max_out),
        ("softmax_sum_out", softmax_sum_out),
    ):
        if tensor is not None:
            kw[name] = tensor

    has_state = (
        softmax_max_out is not None
        or softmax_sum_out is not None
        or prior_softmax_max is not None
        or prior_softmax_sum is not None
        or prior_attention_out is not None
    )
    use_custom = has_state or os.environ.get("USE_CUSTOM_SFA") == "1"
    if use_custom:
        import custom_ops  # noqa: F401

        out = torch.ops.custom.npu_dmp_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, **kw
        )
    else:
        if query_rope is not None:
            kw["attention_mode"] = attention_mode
        out = torch_npu.npu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, **kw
        )
    return out[0] if isinstance(out, (tuple, list)) else out


def _time_npu_ms(fn) -> tuple[Any, float]:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    val = fn()
    end.record()
    end.synchronize()
    return val, float(start.elapsed_time(end))


def run_dual_sparse_attention(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_k_rope: torch.Tensor,
    selection_kv_block_table: torch.Tensor,
    column_indices: torch.Tensor,
    hit_mask: torch.Tensor,
    scale_value: float,
    *,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    sparse_block_size: int = 1,
    layout_query: str = "TND",
    layout_kv: str = "PA_BSND",
    sparse_mode: int = 3,
    attention_mode: int = 2,
    time: bool = False,
) -> DualAttentionOutputs | tuple[DualAttentionOutputs, float, float]:
    """Attn0 export + Attn1 merge. ``time=True`` → ``(outputs, attn0_ms, attn1_ms)``."""
    miss_mask = ~hit_mask
    hit_indices, _ = pack_sparse_indices(column_indices, hit_mask)
    miss_indices, _ = pack_sparse_indices(column_indices, miss_mask)
    key = selection_kv_cache.unsqueeze(2)
    common = dict(
        query_rope=query_rope,
        key_rope=selection_k_rope.unsqueeze(2),
        block_table=selection_kv_block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
    )
    n_hit = int(hit_mask.sum().item())
    n_miss = int(miss_mask.sum().item())
    t = query.shape
    # Rows with no Attn0 hits keep the softmax identity until Attn1.
    sm_max = torch.full(
        (t[0], t[1]), torch.finfo(torch.float32).min, dtype=torch.float32, device=query.device
    )
    sm_sum = torch.zeros_like(sm_max)
    sm1_max = torch.full_like(sm_max, torch.finfo(torch.float32).min)
    sm1_sum = torch.zeros_like(sm_max)
    use_prior_merge = os.environ.get("DA_KERNEL_PRIOR_MERGE", "0") == "1"

    def _sfa(
        indices,
        *,
        export_max=None,
        export_sum=None,
        prior_attention_out=None,
    ):
        return run_sparse_flash_attention(
            query,
            key,
            key,
            indices,
            scale_value,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            softmax_max_out=export_max,
            softmax_sum_out=export_sum,
            prior_softmax_max=sm_max if prior_attention_out is not None else None,
            prior_softmax_sum=sm_sum if prior_attention_out is not None else None,
            prior_attention_out=prior_attention_out,
            **common,
        )

    def _run_attn0():
        if n_hit == 0:
            return query.new_zeros(query.shape)
        return _sfa(hit_indices, export_max=sm_max, export_sum=sm_sum)

    def _run_attn1(out0: torch.Tensor):
        if n_miss == 0:
            return out0, out0
        if n_hit == 0:
            out1 = _sfa(miss_indices)
            return out1, out1
        if use_prior_merge:
            out1 = _sfa(miss_indices, prior_attention_out=out0)
            return out1, out1
        out1 = _sfa(miss_indices, export_max=sm1_max, export_sum=sm1_sum)
        merged = da_attention_merge(out0, out1, sm_max, sm_sum, sm1_max, sm1_sum)
        return merged, out1

    if time:
        out0, t0 = _time_npu_ms(_run_attn0)
        if n_miss == 0:
            return DualAttentionOutputs(out0, out0, out0, n_hit, n_miss), t0, 0.0
        (final, out1), t1 = _time_npu_ms(lambda: _run_attn1(out0))
        return DualAttentionOutputs(final, out0, out1, n_hit, n_miss), t0, t1

    out0 = _run_attn0()
    final, out1 = _run_attn1(out0)
    return DualAttentionOutputs(final, out0, out1, n_hit, n_miss)


class DualAttentionRunner:
    def __init__(self, config: BaselineConfig | None = None):
        self._rt = BaselineRuntime(config or BaselineConfig())

    def run_step(self, step_id: int) -> DualAttentionStepMetrics:
        rt = self._rt
        torch.npu.synchronize()
        indexer_topk, indexer_ms = _time_npu_ms(lambda: rt.run_indexer())
        topk = blend_indexer_topk_with_reuse(indexer_topk, rt._prev_topk, rt.topk_reuse_rate, rt.rng)
        gather_inputs = rt._gather_inputs._replace(selection_topk_indices=topk)
        status_before = gather_inputs.selection_kv_block_status.clone()
        prepare_gather_step(gather_inputs, rt.topk_reuse_rate, rt.kv_max_seq_len, rt.rng)
        _, gather_ms = _time_npu_ms(lambda: rt.run_gather(gather_inputs))
        rt._prev_topk = topk.detach().clone()
        sparse = rt.make_sparse_attn_inputs(rt.gather_kv_lengths)
        hit_mask = infer_hit_mask_from_block_status(
            gather_inputs.selection_topk_indices, status_before
        )
        cols = torch.arange(rt.index_topk, dtype=torch.int32, device=rt.device).view(1, 1, -1)
        cols = cols.expand(rt.token_count, rt.gather_head_num, -1)
        da_out, attn0_ms, attn1_ms = run_dual_sparse_attention(
            sparse.query,
            sparse.query_rope,
            gather_inputs.selection_kv_cache,
            gather_inputs.selection_k_rope,
            gather_inputs.selection_kv_block_table,
            cols,
            hit_mask,
            sparse.scale_value,
            actual_seq_lengths_query=sparse.actual_seq_lengths_query,
            actual_seq_lengths_kv=sparse.actual_seq_lengths_kv,
            sparse_block_size=sparse.sparse_block_size,
            layout_query=sparse.layout_query,
            layout_kv=sparse.layout_kv,
            sparse_mode=sparse.sparse_mode,
            attention_mode=sparse.attention_mode,
            time=True,
        )
        sparse_attn_ms = attn0_ms + attn1_ms
        return DualAttentionStepMetrics(
            step_id=step_id,
            indexer_ms=indexer_ms,
            gather_ms=gather_ms,
            sparse_attn_ms=sparse_attn_ms,
            step_ms=indexer_ms + gather_ms + sparse_attn_ms,
            attn0_ms=attn0_ms,
            attn1_ms=attn1_ms,
            hit_count=da_out.hit_count,
            miss_count=da_out.miss_count,
        )


def run_dual_attention_pipeline(config: BaselineConfig | None = None) -> list[DualAttentionStepMetrics]:
    runner = DualAttentionRunner(config)
    return [runner.run_step(step_id) for step_id in range(runner._rt.seq_len)]


def print_dual_attention_step(row: DualAttentionStepMetrics) -> None:
    print(
        f"step={row.step_id} indexer_ms={row.indexer_ms:.3f} gather_ms={row.gather_ms:.3f} "
        f"sparse_attn_ms={row.sparse_attn_ms:.3f} step_ms={row.step_ms:.3f} "
        f"attn0={row.attn0_ms:.3f} attn1={row.attn1_ms:.3f} "
        f"hit={row.hit_count} miss={row.miss_count}"
    )


if __name__ == "__main__":
    for row in run_dual_attention_pipeline():
        print_dual_attention_step(row)
