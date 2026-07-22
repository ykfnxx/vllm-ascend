#!/usr/bin/env python3
# coding=utf-8
"""Validate Dual-Attention (DA) export/prior SFA vs single SFA on selection KV (PA_BSND).

Uses BaselineRuntime gather pool + TND query (same contract as src/baseline.py).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
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
    run_dual_sparse_attention,
    run_sparse_flash_attention,
)


def parse_args():
    p = argparse.ArgumentParser(description="DA sparse flash attention vs full SFA")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--kv-max-seq-len", type=int, default=65536)
    p.add_argument("--index-topk", type=int, default=2048)
    p.add_argument("--topk-reuse-rate", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.001)
    return p.parse_args()


def run_one_step(runtime: BaselineRuntime, reuse_rate: float) -> dict:
    indexer_topk = runtime.run_indexer()
    topk = blend_indexer_topk_with_reuse(
        indexer_topk, runtime._prev_topk, reuse_rate, runtime.rng
    )
    gather_inputs = runtime._gather_inputs._replace(selection_topk_indices=topk)
    status_before = gather_inputs.selection_kv_block_status.clone()
    prepare_gather_step(gather_inputs, reuse_rate, runtime.kv_max_seq_len, runtime.rng)
    runtime.run_gather(gather_inputs)
    runtime._prev_topk = topk.detach().clone()

    sparse = runtime.make_sparse_attn_inputs(runtime.gather_kv_lengths)
    hit_mask = infer_hit_mask_from_block_status(
        gather_inputs.selection_topk_indices, status_before
    )

    full_out = run_sparse_flash_attention(
        sparse.query,
        gather_inputs.selection_kv_cache.unsqueeze(2),
        gather_inputs.selection_kv_cache.unsqueeze(2),
        sparse.sparse_indices,
        sparse.scale_value,
        query_rope=sparse.query_rope,
        key_rope=gather_inputs.selection_k_rope.unsqueeze(2),
        block_table=gather_inputs.selection_kv_block_table,
        actual_seq_lengths_query=sparse.actual_seq_lengths_query,
        actual_seq_lengths_kv=sparse.actual_seq_lengths_kv,
        sparse_block_size=sparse.sparse_block_size,
        layout_query=sparse.layout_query,
        layout_kv=sparse.layout_kv,
        sparse_mode=sparse.sparse_mode,
        attention_mode=sparse.attention_mode,
    )

    cols = torch.arange(runtime.index_topk, dtype=torch.int32, device=runtime.device).view(1, 1, -1)
    cols = cols.expand(runtime.token_count, runtime.gather_head_num, -1)
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
    return {
        "full_out": full_out.float().cpu(),
        "da_out": da_out.attention_out.float().cpu(),
        "hit": da_out.hit_count,
        "miss": da_out.miss_count,
        "attn0_ms": attn0_ms,
        "attn1_ms": attn1_ms,
    }


def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    cfg = BaselineConfig(
        device=args.device,
        batch_size=args.batch_size,
        kv_max_seq_len=args.kv_max_seq_len,
        index_topk=args.index_topk,
        topk_reuse_rate=args.topk_reuse_rate,
    )
    runtime = BaselineRuntime(cfg)

    for _ in range(args.warmup):
        run_one_step(runtime, args.topk_reuse_rate)
    torch.npu.synchronize()

    result = run_one_step(runtime, args.topk_reuse_rate)
    ok = torch.allclose(result["da_out"], result["full_out"], rtol=args.rtol, atol=args.atol)
    max_diff = (result["da_out"] - result["full_out"]).abs().max().item()
    print(
        f"hit={result['hit']} miss={result['miss']} "
        f"attn0_ms={result['attn0_ms']:.3f} attn1_ms={result['attn1_ms']:.3f} "
        f"total_ms={result['attn0_ms'] + result['attn1_ms']:.3f}"
    )
    print(f"max_abs_diff={max_diff:.6f} allclose={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
