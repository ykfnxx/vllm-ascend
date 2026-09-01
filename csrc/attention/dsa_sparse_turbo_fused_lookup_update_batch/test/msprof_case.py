#!/usr/bin/env python3
"""Run turbo + turbo_fused A5 lookup for one CSV case, for msprof per-op timing.

Mirror of the turbo baseline's msprof_case.py: one workload (batch /
index_capacity / hit_rate / qpr) launches BOTH
dsa_sparse_turbo_lookup_update_batch (baseline) and
dsa_sparse_turbo_fused_lookup_update_batch (V2, in-kernel classification),
interleaved, so a wrapping `msprof` produces one op_summary row per kernel
launch distinguishable by Op Name.

Workload alignment: the baseline receives the framework-generated history
lookup_mask (valid && token < tail_start); the fused op receives the same
request/query layout via query_positions + verify_starts + tail_starts and
classifies in-kernel.  Both process exactly the same history token set, so
the kernel durations are directly comparable.

Usage (wrapped by msprof):
    msprof --output=... --ai-core=on --aic-mode=task-based --task-time=on \
        --application="python3 msprof_case.py --batch 16 --capacity 131072 \
                       --hit 0.90 --qpr 4 --n 50"
"""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

DEV = "npu:0"
QUERY_WIDTH = 2048
SLOT_COUNT = 10 * 1024
FREE_SLOT_COUNT = 2 * 1024
RESIDENT_SLOTS = SLOT_COUNT - FREE_SLOT_COUNT
NOT_FOUND = -1
BLOCK_SIZE = 128


def build_inputs(batch, index_capacity, hit_rate, qpr, seed):
    """Clean A5 state: resident tail window bidirectionally valid, plus the
    fused op's compact verify/tail anchors and per-query positions."""
    g = torch.Generator("cpu").manual_seed(seed)
    dev = torch.device(DEV)
    total_queries = batch * qpr
    index = torch.full((batch, index_capacity), NOT_FOUND, dtype=torch.int32, device=dev)
    sti = torch.full((batch, SLOT_COUNT), NOT_FOUND, dtype=torch.int32, device=dev)
    resident_start = index_capacity - RESIDENT_SLOTS
    resident_tokens = torch.arange(resident_start, index_capacity, dtype=torch.int32, device=dev)
    slots = torch.arange(RESIDENT_SLOTS, dtype=torch.int32, device=dev)
    for p in range(batch):
        index[p].index_copy_(0, resident_tokens, slots)
        sti[p, :RESIDENT_SLOTS] = slots
    free_slots = torch.zeros((batch, FREE_SLOT_COUNT), dtype=torch.int32, device=dev)
    free_slots[:] = torch.arange(
        RESIDENT_SLOTS, SLOT_COUNT, dtype=torch.int32, device=dev).unsqueeze(0)
    free_head = torch.zeros((batch, 16), dtype=torch.int32, device=dev)

    hit_n = max(1, int(QUERY_WIDTH * hit_rate))
    miss_n = QUERY_WIDTH - hit_n
    query = torch.full((total_queries, QUERY_WIDTH), NOT_FOUND, dtype=torch.int32)
    miss_pool = torch.arange(0, resident_start, dtype=torch.int32)
    miss_perm = torch.randperm(miss_pool.numel(), generator=g)
    flat_miss = miss_pool[miss_perm[: total_queries * miss_n]]
    for tq in range(total_queries):
        perm = torch.randperm(QUERY_WIDTH, generator=g)
        hits = torch.randint(resident_start, index_capacity, (hit_n,), generator=g).int()
        misses = flat_miss[tq * miss_n : (tq + 1) * miss_n]
        query[tq, perm[:hit_n]] = hits
        query[tq, perm[hit_n:]] = misses
    query = query.to(dev)
    req_pool = torch.arange(batch, dtype=torch.int32, device=dev)
    qsl = torch.arange(batch + 1, dtype=torch.int32, device=dev) * qpr
    # fused classification inputs
    verify_start = resident_start + RESIDENT_SLOTS // 2
    verify_starts = torch.full((batch,), verify_start, dtype=torch.int32, device=dev)
    query_positions = (
        torch.arange(total_queries, dtype=torch.int32, device=dev) + verify_start
    )
    # baseline history mask: valid && token < tail_start (same token set the
    # fused kernel classifies as history)
    tail_start = (verify_start // BLOCK_SIZE) * BLOCK_SIZE
    tail_starts = torch.full((batch,), tail_start, dtype=torch.int32, device=dev)
    mask = ((query >= 0) & (query < index_capacity) & (query < tail_start)
            ).to(torch.int32).contiguous()
    return (query, query_positions, verify_starts, tail_starts, index, sti, free_slots,
            free_head, req_pool, qsl, mask)


def clone(*tensors):
    return tuple(t.clone() for t in tensors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--capacity", type=int, required=True)
    ap.add_argument("--hit", type=float, required=True)
    ap.add_argument("--qpr", type=int, default=1, help="queries per request")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    (q, qpos, vstarts, tstarts, idx, sti, fs, fh, rp, qsl, mask) = build_inputs(
        args.batch, args.capacity, args.hit, args.qpr, seed=20260827)
    bs = args.batch
    mtp = 1

    def turbo_args():
        i, s, f, h = clone(idx, sti, fs, fh)
        return (i, s, f, h, rp, qsl, q, mask, bs)

    def fused_args():
        i, s, f, h = clone(idx, sti, fs, fh)
        return (
            i, s, f, h, rp, qsl, q, qpos, vstarts, tstarts,
            bs, BLOCK_SIZE, mtp)

    # warmup both (steady state for msprof)
    for _ in range(20):
        torch.ops._C_ascend.dsa_sparse_turbo_lookup_update_batch(*turbo_args())
    for _ in range(20):
        torch.ops._C_ascend.dsa_sparse_turbo_fused_lookup_update_batch(*fused_args())
    torch.npu.synchronize()

    # timed interleaved loop (one op_summary row per launch)
    for _ in range(args.n):
        torch.ops._C_ascend.dsa_sparse_turbo_lookup_update_batch(*turbo_args())
        torch.ops._C_ascend.dsa_sparse_turbo_fused_lookup_update_batch(*fused_args())
    torch.npu.synchronize()
    print(f"[msprof_case] batch={args.batch} capacity={args.capacity} "
          f"hit={args.hit} qpr={args.qpr} n={args.n} done")


if __name__ == "__main__":
    main()
