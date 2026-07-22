#!/usr/bin/env python3
# coding=utf-8
"""Probe DA prior path on a fixed hit>0 batch.

Goal:
- Fix one gather batch (same tensors for Attn0/Attn1/full).
- Compare Attn0-exported `sm_max/sm_sum` with Attn1 prior-read inputs
  (the same GM tensors before Attn1 call).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from baseline import (  # noqa: E402
    BaselineConfig,
    BaselineRuntime,
    blend_indexer_topk_with_reuse,
    prepare_gather_step,
)
from dual_attention import (  # noqa: E402
    infer_hit_mask_from_block_status,
    pack_sparse_indices,
    run_sparse_flash_attention,
)


def parse_args():
    p = argparse.ArgumentParser(description="Probe DA prior readback on fixed hit>0 case.")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--kv-max-seq-len", type=int, default=4096)
    p.add_argument("--index-topk", type=int, default=256)
    p.add_argument("--reuse-rate", type=float, default=0.5)
    p.add_argument("--max-probe-steps", type=int, default=8)
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.001)
    return p.parse_args()


def _prepare_one_step(runtime: BaselineRuntime, reuse_rate: float):
    indexer_topk = runtime.run_indexer()
    topk = blend_indexer_topk_with_reuse(indexer_topk, runtime._prev_topk, reuse_rate, runtime.rng)
    gather_inputs = runtime._gather_inputs._replace(selection_topk_indices=topk)
    status_before = gather_inputs.selection_kv_block_status.clone()
    prepare_gather_step(gather_inputs, reuse_rate, runtime.kv_max_seq_len, runtime.rng)
    runtime.run_gather(gather_inputs)
    runtime._prev_topk = topk.detach().clone()
    sparse = runtime.make_sparse_attn_inputs(runtime.gather_kv_lengths)
    hit_mask = infer_hit_mask_from_block_status(gather_inputs.selection_topk_indices, status_before)
    return sparse, gather_inputs, hit_mask


def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    cfg = BaselineConfig(
        device=args.device,
        batch_size=args.batch_size,
        kv_max_seq_len=args.kv_max_seq_len,
        index_topk=args.index_topk,
        topk_reuse_rate=args.reuse_rate,
    )
    rt = BaselineRuntime(cfg)

    probe = None
    for _ in range(args.max_probe_steps):
        sparse, gather_inputs, hit_mask = _prepare_one_step(rt, args.reuse_rate)
        if int(hit_mask.sum().item()) > 0:
            probe = (sparse, gather_inputs, hit_mask)
            break
    if probe is None:
        raise SystemExit("Failed to find hit>0 case; increase --max-probe-steps.")

    sparse, gather_inputs, hit_mask = probe
    miss_mask = ~hit_mask

    cols = torch.arange(rt.index_topk, dtype=torch.int32, device=rt.device).view(1, 1, -1)
    cols = cols.expand(rt.token_count, rt.gather_head_num, -1)
    hit_indices, _ = pack_sparse_indices(cols, hit_mask)
    miss_indices, _ = pack_sparse_indices(cols, miss_mask)

    key = gather_inputs.selection_kv_cache.unsqueeze(2)
    key_rope = gather_inputs.selection_k_rope.unsqueeze(2)
    common = dict(
        query_rope=sparse.query_rope,
        key_rope=key_rope,
        block_table=gather_inputs.selection_kv_block_table,
        actual_seq_lengths_query=sparse.actual_seq_lengths_query,
        actual_seq_lengths_kv=sparse.actual_seq_lengths_kv,
        sparse_block_size=sparse.sparse_block_size,
        layout_query=sparse.layout_query,
        layout_kv=sparse.layout_kv,
        sparse_mode=sparse.sparse_mode,
    )

    sm_max = torch.full(
        (sparse.query.shape[0], sparse.query.shape[1]),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=sparse.query.device,
    )
    sm_sum = torch.zeros_like(sm_max)

    _ = run_sparse_flash_attention(
        sparse.query,
        key,
        key,
        hit_indices,
        sparse.scale_value,
        softmax_max_out=sm_max,
        softmax_sum_out=sm_sum,
        **common,
    )
    torch.npu.synchronize()

    # Attn1 prior reads the same GM buffers written by Attn0 export.
    expected_max = sm_max.clone()
    expected_sum = sm_sum.clone()

    prior_read_max = torch.zeros_like(sm_max)
    prior_read_sum = torch.zeros_like(sm_sum)

    attn1_out = run_sparse_flash_attention(
        sparse.query,
        key,
        key,
        miss_indices,
        sparse.scale_value,
        softmax_max_out=prior_read_max,
        softmax_sum_out=prior_read_sum,
        prior_softmax_max=sm_max,
        prior_softmax_sum=sm_sum,
        **common,
    )
    full_out = run_sparse_flash_attention(
        sparse.query,
        key,
        key,
        sparse.sparse_indices,
        sparse.scale_value,
        **common,
    )
    torch.npu.synchronize()

    export_vs_prior_equal_max = bool(torch.equal(expected_max, prior_read_max))
    export_vs_prior_equal_sum = bool(torch.equal(expected_sum, prior_read_sum))
    finite_max_mask = torch.isfinite(expected_max) & torch.isfinite(prior_read_max)
    finite_sum_mask = torch.isfinite(expected_sum) & torch.isfinite(prior_read_sum)
    export_vs_prior_max = (
        (expected_max[finite_max_mask] - prior_read_max[finite_max_mask]).abs().max().item()
        if finite_max_mask.any()
        else 0.0
    )
    export_vs_prior_sum = (
        (expected_sum[finite_sum_mask] - prior_read_sum[finite_sum_mask]).abs().max().item()
        if finite_sum_mask.any()
        else 0.0
    )

    row_has_hit = hit_mask.any(dim=-1)
    no_hit_rows = ~row_has_hit
    no_hit_cnt = int(no_hit_rows.sum().item())

    if no_hit_cnt > 0:
        no_hit_sm_max = expected_max[no_hit_rows]
        no_hit_sm_sum = expected_sum[no_hit_rows]
        no_hit_max_abs = no_hit_sm_max.abs().max().item()
        no_hit_sum_abs = no_hit_sm_sum.abs().max().item()
    else:
        no_hit_max_abs = 0.0
        no_hit_sum_abs = 0.0

    da_max_diff = (attn1_out.float() - full_out.float()).abs().max().item()
    da_ok = torch.allclose(attn1_out, full_out, rtol=args.rtol, atol=args.atol)

    print(f"hit={int(hit_mask.sum().item())} miss={int(miss_mask.sum().item())}")
    print(
        f"export_vs_attn1_prior_input: equal_max={export_vs_prior_equal_max} "
        f"equal_sum={export_vs_prior_equal_sum} sm_max_finite_diff={export_vs_prior_max:.6g} "
        f"sm_sum_finite_diff={export_vs_prior_sum:.6g}"
    )
    print(
        f"no_hit_rows={no_hit_cnt} no_hit_prior_absmax={no_hit_max_abs:.6g} "
        f"no_hit_prior_abssum={no_hit_sum_abs:.6g}"
    )
    print(f"attn1_vs_full: max_abs_diff={da_max_diff:.6f} allclose={da_ok}")


if __name__ == "__main__":
    main()
