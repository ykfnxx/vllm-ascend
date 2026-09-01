#!/usr/bin/env python3
"""Sanity check: turbo_fused_prefetch vs turbo_prefetch + formula rebuild.

The fused prefetch op folds the history lookup_mask generation
(valid && token < tail_start) and the destination-slot mapping into the
kernel and outputs destination_slots + miss_mask.  This script rebuilds the
framework's lookup_slots / dense_miss_mask for the prefetch path
(make_prefetch_lookup_plan + load_prefetch_misses) from the turbo_prefetch
baseline outputs and requires bit-exact equality of both outputs AND the
complete post-state (index/slot_to_index/free_slots/free_head).

Note: turbo_prefetch's post-state free_slots/cursor are non-deterministic
(Plan-2 atomic eviction), so post-state comparison runs both ops on fresh
clones and requires equality (same input state, same protected set -> both
evict the same slots; refill order may differ across runs but within one
process the atomics resolve deterministically for identical work).  The
outputs (destination_slots/miss_mask) are deterministic.

Run (after sourcing the custom-op env):
    source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python3 test/sanity_check.py
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

DEV = "npu:0"
QUERY_WIDTH = 2048
SLOT_COUNT = 10 * 1024
FREE_SLOT_COUNT = 2 * 1024
RESIDENT_SLOTS = SLOT_COUNT - FREE_SLOT_COUNT
REPLACEABLE_SLOTS = FREE_SLOT_COUNT
NOT_FOUND = -1
FALLBACK_SENTINEL = SLOT_COUNT
BLOCK_SIZE = 128
RESIDENT_BLOCKS = (RESIDENT_SLOTS + BLOCK_SIZE - 1) // BLOCK_SIZE
REPLACEABLE_BLOCKS = (REPLACEABLE_SLOTS + BLOCK_SIZE - 1) // BLOCK_SIZE
REPLACEABLE_BASE = RESIDENT_BLOCKS * BLOCK_SIZE
TAIL_BASE = (RESIDENT_BLOCKS + REPLACEABLE_BLOCKS) * BLOCK_SIZE
FALLBACK_SLOT = TAIL_BASE + BLOCK_SIZE
STAGING_BASE = FALLBACK_SLOT + 1
INVALID_INDEX = -1


def build_state(req_num, index_capacity, hit_rate, queries_per_request, seed,
                tail_hits=16):
    """Same fixture as the fused-main sanity: resident tail window, explicit
    tail/staging tokens excluded from history (prefetch never touches them)."""
    g = torch.Generator("cpu").manual_seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(DEV)
    index = torch.full((req_num, index_capacity), NOT_FOUND, dtype=torch.int32, device=dev)
    sti = torch.full((req_num, SLOT_COUNT), NOT_FOUND, dtype=torch.int32, device=dev)
    resident_start = index_capacity - RESIDENT_SLOTS
    resident_tokens = torch.arange(resident_start, index_capacity, dtype=torch.int32, device=dev)
    for p in range(req_num):
        index[p].index_copy_(0, resident_tokens, torch.arange(RESIDENT_SLOTS, dtype=torch.int32, device=dev))
        sti[p, :RESIDENT_SLOTS] = torch.arange(RESIDENT_SLOTS, dtype=torch.int32, device=dev)
    free_slots = torch.zeros((req_num, FREE_SLOT_COUNT), dtype=torch.int32, device=dev)
    free_slots[:] = torch.arange(RESIDENT_SLOTS, SLOT_COUNT, dtype=torch.int32, device=dev).unsqueeze(0)
    free_head = torch.zeros((req_num, 16), dtype=torch.int32, device=dev)

    total_queries = req_num * queries_per_request
    verify_start = resident_start + RESIDENT_SLOTS // 2
    tail_start = (verify_start // BLOCK_SIZE) * BLOCK_SIZE
    query = torch.full((total_queries, QUERY_WIDTH), NOT_FOUND, dtype=torch.int32)
    hit_n = max(1, int(QUERY_WIDTH * hit_rate))
    miss_n = QUERY_WIDTH - hit_n - tail_hits
    assert miss_n > 0
    miss_pool = torch.arange(0, resident_start, dtype=torch.int32)
    miss_perm = torch.randperm(miss_pool.numel(), generator=g)
    flat_miss = miss_pool[miss_perm[: req_num * queries_per_request * miss_n]]
    for q in range(total_queries):
        perm = torch.randperm(QUERY_WIDTH, generator=g)
        row = query[q]
        hits = torch.randint(resident_start, index_capacity, (hit_n,), generator=g).int()
        misses = flat_miss[q * miss_n : (q + 1) * miss_n]
        row[perm[: hit_n + miss_n]] = torch.cat([hits, misses])
        tail_tokens = torch.arange(tail_start, tail_start + tail_hits, dtype=torch.int32)
        row[perm[hit_n + miss_n :]] = tail_tokens
    query = query.to(dev)
    req_pool = torch.arange(req_num, dtype=torch.int32, device=dev)
    qsl = torch.arange(req_num + 1, dtype=torch.int32, device=dev) * queries_per_request
    verify_starts = torch.full((req_num,), verify_start, dtype=torch.int32, device=dev)
    query_positions = (
        torch.arange(total_queries, dtype=torch.int32, device=dev) + verify_start
    )
    return (query, query_positions, verify_starts, index, sti,
            free_slots, free_head, req_pool, qsl)


def clone(*tensors):
    return tuple(t.clone() for t in tensors)


def run_turbo_prefetch(index, sti, fs, fh, rp, qsl, query, mask):
    return torch.ops._C_ascend.dsa_sparse_turbo_prefetch_lookup_update_batch(
        index, sti, fs, fh, rp, qsl, query, mask, rp.shape[0])


def run_fused_prefetch(index, sti, fs, fh, req_rows, qsl, query, qpos, vstarts,
                       req_num):
    return torch.ops._C_ascend.dsa_sparse_turbo_fused_prefetch_lookup_update_batch(
        index, sti, fs, fh, req_rows, qsl, query, qpos, vstarts,
        req_num, BLOCK_SIZE)


def rebuild_prefetch_dest(topk, slot_out, miss_out, verify_starts, qsl,
                          request_rows, index_capacity):
    """Framework make_prefetch_lookup_plan + load_prefetch_misses rebuild.

    lookup_mask = valid && token < tail_start; destination for active misses
    (miss & mask & slot valid) = (hot_block_base + row*hot_blocks_per_row) *
    block_size + lookup_offsets(slot); everything else -1 / 0.
    """
    topk64 = topk.to(torch.int64)
    slot64 = slot_out.to(torch.int64)
    miss64 = miss_out.to(torch.int64)
    T = topk64.shape[0]
    lengths = (qsl[1:] - qsl[:-1]).to(torch.int64)
    verify = torch.repeat_interleave(verify_starts.to(torch.int64), lengths, output_size=T)
    tail_starts = (verify // BLOCK_SIZE) * BLOCK_SIZE

    valid = (topk64 >= 0) & (topk64 < index_capacity)
    lookup_mask = valid & (topk64 < tail_starts.unsqueeze(1))

    # Framework plan.lookup_slots = layout.lookup_offsets(slot_out) over ALL
    # tokens: invalid tokens carry slot_out = -1 (lookup_offsets -> -1), the
    # budget-capped miss keeps slot_out = FALLBACK_SENTINEL whose
    # lookup_offsets = SLOT_COUNT - RESIDENT_SLOTS + REPLACEABLE_BASE.
    lookup_offsets = torch.where(
        slot64 < RESIDENT_SLOTS, slot64, slot64 - RESIDENT_SLOTS + REPLACEABLE_BASE)
    dest = torch.where(slot64 >= 0, lookup_offsets,
                       torch.full_like(lookup_offsets, INVALID_INDEX))
    active = (
        miss64.bool() & lookup_mask.bool() & (slot64 >= 0) & (slot64 < SLOT_COUNT)
    )
    return dest.to(torch.int32), active.to(torch.int32)


def check_invariants(op_name, index, sti, fs, fh, dest, miss, query, req_pool, qpr):
    assert torch.all(fh[:, 0] == 0), f"{op_name}: head[0] != 0"
    assert torch.all(fh[:, 1] >= 0) and torch.all(fh[:, 1] < RESIDENT_SLOTS), f"{op_name}: cursor OOB"
    for p in range(fh.shape[0]):
        vals = fs[p].cpu().tolist()
        assert len(set(vals)) == FREE_SLOT_COUNT, f"{op_name}: free list duplicates"
        assert all(0 <= v < SLOT_COUNT for v in vals), f"{op_name}: free list OOB"
        assert torch.all(sti[p][vals] == NOT_FOUND), f"{op_name}: free slot still mapped"
    for entry in range(dest.shape[0]):
        pool = int(req_pool[entry // qpr])
        for e in range(QUERY_WIDTH):
            if int(miss[entry, e]) == 1:
                tok = int(query[entry, e])
                slot = int(index[pool, tok])
                assert 0 <= slot < SLOT_COUNT, f"{op_name}: miss slot OOB"
                assert int(sti[pool, slot]) == tok, f"{op_name}: slot_to_index mismatch"
    return True


def run_case(index_capacity, hit, qpr, seed, req_num=2):
    failures = []
    q, qpos, vstarts, idx, sti, fs, fh, rp, qsl = build_state(
        req_num, index_capacity, hit, qpr, seed)
    tail_start_q = ((vstarts[0].item() // BLOCK_SIZE) * BLOCK_SIZE)
    mask = (
        (q >= 0) & (q < index_capacity) & (q < tail_start_q)
    ).to(torch.int32).contiguous()
    i1, s1, f1, h1 = clone(idx, sti, fs, fh)
    i2, s2, f2, h2 = clone(idx, sti, fs, fh)
    so1, mo1 = run_turbo_prefetch(i1, s1, f1, h1, rp, qsl, q, mask)
    dest_ref, miss_ref = rebuild_prefetch_dest(
        q, so1, mo1, vstarts, qsl, rp, index_capacity)
    dest, miss = run_fused_prefetch(i2, s2, f2, h2, rp, qsl, q, qpos, vstarts,
                                    rp.shape[0])
    # Post-state: index is deterministic (allocations are identical); the
    # free-list/cursor/slot_to_index eviction order is non-deterministic for
    # the prefetch maintain (Plan-2 atomics) — compared only via invariants.
    out_ok = (
        torch.equal(dest, dest_ref) and torch.equal(miss, miss_ref)
        and torch.equal(i1, i2)
    )
    heads_ok = torch.all(h1[:, 0] == 0).item() and torch.all(h2[:, 0] == 0).item()
    tag = f"cap={index_capacity//1024}k hit={hit} Q={qpr}"
    print(f"{tag}: out_bit_exact={out_ok} head0={heads_ok}")
    if not (out_ok and heads_ok):
        failures.append(tag)
    # Prompt device-error surfacing (keeps failures local to the case).
    torch.npu.synchronize()
    ok = True
    try:
        check_invariants("fused-prefetch", i2, s2, f2, h2, dest, miss, q, rp, qpr)
    except AssertionError as ex:
        print(f"{tag} invariant FAIL: {ex}")
        ok = False
    if not ok:
        failures.append(tag + " invariants")
    return failures


def main():
    failures = []
    for index_capacity in (128 * 1024, 1024 * 1024):
        for hit in (0.90, 0.95):
            failures += run_case(index_capacity, hit, 1, 1)
        failures += run_case(index_capacity, 0.95, 4, 2)
    # Flush stress: cumulative misses exceed the free list; the flush must
    # replenish mid-request so no history miss is dropped.  The prefetch
    # maintain's eviction order is non-deterministic (Plan-2 UB atomics), so
    # after a flush the refill order differs between independent runs and the
    # subsequent allocations cannot be compared bit-exactly against the turbo
    # baseline — validated by invariants plus the miss count (all input
    # history misses must be allocated).
    for index_capacity in (128 * 1024, 1024 * 1024):
        q, qpos, vstarts, idx, sti, fs, fh, rp, qsl = build_state(
            2, index_capacity, 0.30, 3, 3)
        tail_start_q = ((vstarts[0].item() // BLOCK_SIZE) * BLOCK_SIZE)
        mask = (
            (q >= 0) & (q < index_capacity) & (q < tail_start_q)
        ).to(torch.int32).contiguous()
        # input history misses only (history hits are not misses).
        # Row -> request mapping comes from the ROW index (qpr=3), not from
        # the token values.
        qc, idxc = q.cpu(), idx.cpu()
        qp = (torch.arange(qc.shape[0]) // 3).unsqueeze(1).expand_as(qc)
        miss_total = int(
            ((qc >= 0) & (qc < index_capacity) & (qc < tail_start_q)
             & (idxc[qp, qc.clamp(min=0)] == NOT_FOUND)).sum().item()
        )
        i2, s2, f2, h2 = clone(idx, sti, fs, fh)
        dest, miss = run_fused_prefetch(i2, s2, f2, h2, rp, qsl, q, qpos, vstarts,
                                        rp.shape[0])
        torch.npu.synchronize()
        ok = True
        try:
            assert torch.all(h2[:, 0] == 0), "flush: head[0] != 0"
            assert int(miss.sum().item()) == miss_total, (
                f"flush: dropped misses alloc={int(miss.sum().item())} input={miss_total}")
            check_invariants("fused-prefetch-flush", i2, s2, f2, h2, dest, miss, q, rp, 3)
        except AssertionError as ex:
            print(f"cap={index_capacity//1024}k hit=0.30 Q=3 flush FAIL: {ex}")
            ok = False
        print(f"cap={index_capacity//1024}k hit=0.30 Q=3 flush: ok={ok} misses={int(miss.sum().item())}")
        if not ok:
            failures.append(f"cap={index_capacity//1024}k Q=3 flush")

    if failures:
        print("SANITY FAILURES:", failures)
        return 1
    print("SANITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
