#!/usr/bin/env python3
# coding=utf-8
"""Probe DA output-O merge: kernel prior path vs host LSE merge."""

from __future__ import annotations

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
    lse_merge,
    pack_sparse_indices,
    run_dual_sparse_attention,
    run_sparse_flash_attention,
)


def _fixed_hit_batch(reuse_rate: float = 0.5):
    rt = BaselineRuntime(
        BaselineConfig(device="npu:0", batch_size=4, kv_max_seq_len=4096, index_topk=256, topk_reuse_rate=reuse_rate)
    )
    for _ in range(8):
        topk = blend_indexer_topk_with_reuse(rt.run_indexer(), rt._prev_topk, reuse_rate, rt.rng)
        gi = rt._gather_inputs._replace(selection_topk_indices=topk)
        st = gi.selection_kv_block_status.clone()
        prepare_gather_step(gi, reuse_rate, rt.kv_max_seq_len, rt.rng)
        rt.run_gather(gi)
        rt._prev_topk = topk.detach().clone()
        sparse = rt.make_sparse_attn_inputs(rt.gather_kv_lengths)
        hit = infer_hit_mask_from_block_status(gi.selection_topk_indices, st)
        if int(hit.sum()) > 0:
            return rt, sparse, gi, hit
    raise RuntimeError("no hit>0 batch")


def main():
    rt, sparse, gi, hit = _fixed_hit_batch()
    miss = ~hit
    cols = torch.arange(rt.index_topk, dtype=torch.int32, device=rt.device).view(1, 1, -1)
    cols = cols.expand(rt.token_count, rt.gather_head_num, -1)
    hit_idx, _ = pack_sparse_indices(cols, hit)
    miss_idx, _ = pack_sparse_indices(cols, miss)
    key = gi.selection_kv_cache.unsqueeze(2)
    common = dict(
        query_rope=sparse.query_rope,
        key_rope=gi.selection_k_rope.unsqueeze(2),
        block_table=gi.selection_kv_block_table,
        actual_seq_lengths_query=sparse.actual_seq_lengths_query,
        actual_seq_lengths_kv=sparse.actual_seq_lengths_kv,
        sparse_block_size=sparse.sparse_block_size,
        layout_query=sparse.layout_query,
        layout_kv=sparse.layout_kv,
        sparse_mode=sparse.sparse_mode,
    )

    sm0_max = torch.full(
        (sparse.query.shape[0], sparse.query.shape[1]),
        torch.finfo(torch.float32).min,
        device=sparse.query.device,
    )
    sm0_sum = torch.zeros_like(sm0_max)
    sm1_max = torch.empty_like(sm0_max)
    sm1_sum = torch.empty_like(sm0_max)

    o0 = run_sparse_flash_attention(
        sparse.query, key, key, hit_idx, sparse.scale_value,
        softmax_max_out=sm0_max, softmax_sum_out=sm0_sum, **common,
    )
    o1_miss = run_sparse_flash_attention(
        sparse.query, key, key, miss_idx, sparse.scale_value,
        softmax_max_out=sm1_max, softmax_sum_out=sm1_sum, **common,
    )
    o1_prior = run_sparse_flash_attention(
        sparse.query, key, key, miss_idx, sparse.scale_value,
        prior_softmax_max=sm0_max, prior_softmax_sum=sm0_sum, prior_attention_out=o0, **common,
    )
    full = run_sparse_flash_attention(
        sparse.query, key, key, sparse.sparse_indices, sparse.scale_value, **common,
    )
    da = run_dual_sparse_attention(
        sparse.query, sparse.query_rope, gi.selection_kv_cache, gi.selection_k_rope,
        gi.selection_kv_block_table, cols, hit, sparse.scale_value,
        actual_seq_lengths_query=sparse.actual_seq_lengths_query,
        actual_seq_lengths_kv=sparse.actual_seq_lengths_kv,
        sparse_block_size=sparse.sparse_block_size,
        layout_query=sparse.layout_query,
        layout_kv=sparse.layout_kv,
        sparse_mode=sparse.sparse_mode,
    )
    torch.npu.synchronize()

    host_merge = lse_merge(
        o0.float(), o1_miss.float(),
        sm0_max.float(), sm0_sum.float(),
        sm1_max.float(), sm1_sum.float(),
    )

    def md(a, b):
        return (a - b).abs().max().item()

    print(f"hit={int(hit.sum())} miss={int(miss.sum())}")
    print(f"full vs o0(hit-only):           max_diff={md(full, o0):.6f}")
    print(f"full vs o1(miss-only):         max_diff={md(full, o1_miss):.6f}")
    print(f"full vs o1(prior kernel):    max_diff={md(full, o1_prior):.6f}")
    print(f"full vs host_lse_merge:        max_diff={md(full, host_merge):.6f}")
    print(f"full vs da_pipeline:           max_diff={md(full, da.attention_out):.6f}")
    print(f"host_merge vs da_pipeline:     max_diff={md(host_merge, da.attention_out):.6f}")
    print(f"host_merge allclose full: {torch.allclose(host_merge, full.float(), rtol=0.02, atol=0.001)}")


if __name__ == "__main__":
    main()
