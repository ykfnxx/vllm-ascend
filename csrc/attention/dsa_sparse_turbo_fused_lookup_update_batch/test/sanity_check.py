#!/usr/bin/env python3
"""Sanity check: turbo_fused (main) vs turbo + framework formula rebuild.

The fused op folds lookup_mask generation, history/tail/staging
classification and the address mapping into the kernel and outputs
mapped_indices + miss_mask.  This script rebuilds the framework's mapped /
dense_miss_mask from the turbo baseline outputs (slot_out/miss_out) plus the
exact same classification formula (lookup.py make_lookup_plan) and requires
bit-exact equality of both outputs AND the complete post-state
(index/slot_to_index/free_slots/free_head), for MTP and non-MTP.

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
FALLBACK_SENTINEL = SLOT_COUNT  # lookup.py FALLBACK_SENTINEL
BLOCK_SIZE = 128
# Framework layout constants (HotCacheLayout at block_size=128):
RESIDENT_BLOCKS = (RESIDENT_SLOTS + BLOCK_SIZE - 1) // BLOCK_SIZE
REPLACEABLE_BLOCKS = (REPLACEABLE_SLOTS + BLOCK_SIZE - 1) // BLOCK_SIZE
REPLACEABLE_BASE = RESIDENT_BLOCKS * BLOCK_SIZE
TAIL_BASE = (RESIDENT_BLOCKS + REPLACEABLE_BLOCKS) * BLOCK_SIZE
FALLBACK_SLOT = TAIL_BASE + BLOCK_SIZE
STAGING_BASE = FALLBACK_SLOT + 1
INVALID_INDEX = -1


def build_state(req_num, index_capacity, hit_rate, queries_per_request, seed,
                mtp=True, staging_hits=8, tail_hits=16):
    """Fresh state: resident tokens at the tail window of the KV sequence.

    Resident window = the last RESIDENT_SLOTS tokens of [0, index_capacity).
    Each query additionally carries explicit tail (block tail) and staging
    (MTP verify window) tokens so the classification boundaries are
    exercised, not just the history region.
    """
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
    # Anchor the request's verify start inside the resident window so that
    # tail/staging tokens are resident tokens already present in the index
    # table (framework semantics: they are classified, never looked up).
    verify_start = resident_start + RESIDENT_SLOTS // 2
    tail_start = (verify_start // BLOCK_SIZE) * BLOCK_SIZE
    query = torch.full((total_queries, QUERY_WIDTH), NOT_FOUND, dtype=torch.int32)
    hit_n = max(1, int(QUERY_WIDTH * hit_rate))
    special_n = tail_hits + (staging_hits if mtp else 0)
    miss_n = QUERY_WIDTH - hit_n - special_n
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
        row[perm[hit_n + miss_n : hit_n + miss_n + tail_hits]] = tail_tokens
        if mtp:
            staging_tokens = torch.arange(
                verify_start, verify_start + staging_hits, dtype=torch.int32)
            row[perm[hit_n + miss_n + tail_hits :]] = staging_tokens
    query = query.to(dev)
    req_pool = torch.arange(req_num, dtype=torch.int32, device=dev)
    qsl = torch.arange(req_num + 1, dtype=torch.int32, device=dev) * queries_per_request
    verify_starts = torch.full((req_num,), verify_start, dtype=torch.int32, device=dev)
    tail_starts = torch.full((req_num,), tail_start, dtype=torch.int32, device=dev)
    query_positions = (
        torch.arange(total_queries, dtype=torch.int32, device=dev) + verify_start
    )
    return (query, query_positions, verify_starts, tail_starts, index, sti,
            free_slots, free_head, req_pool, qsl)


def clone(*tensors):
    return tuple(t.clone() for t in tensors)


def run_turbo(op_name, index, sti, fs, fh, rp, qsl, query, mask):
    fn = getattr(torch.ops._C_ascend, op_name)
    return fn(index, sti, fs, fh, rp, qsl, query, mask, rp.shape[0])


def run_fused_main(index, sti, fs, fh, req_rows, qsl, query, qpos, vstarts,
                   tstarts, req_num, is_mtp):
    return torch.ops._C_ascend.dsa_sparse_turbo_fused_lookup_update_batch(
        index, sti, fs, fh, req_rows, qsl, query, qpos, vstarts,
        tstarts, req_num, BLOCK_SIZE, is_mtp)


def rebuild_main_mapped(topk, slot_out, miss_out, query_positions,
                        verify_starts, qsl, index_capacity, is_mtp):
    """Framework make_lookup_plan rebuild on top of turbo slot_out/miss_out.

    Mirrors lookup.py exactly: valid/history/tail/staging masks -> mapped
    via the where chain, dense_miss_mask = miss & history & ~fallback.
    """
    topk64 = topk.to(torch.int64)
    slot64 = slot_out.to(torch.int64)
    miss64 = miss_out.to(torch.int64)
    T = topk64.shape[0]
    lengths = (qsl[1:] - qsl[:-1]).to(torch.int64)
    verify = torch.repeat_interleave(verify_starts.to(torch.int64), lengths, output_size=T)
    tail_starts = (verify // BLOCK_SIZE) * BLOCK_SIZE
    current_positions = query_positions.to(torch.int64).unsqueeze(1)

    valid = (topk64 >= 0) & (topk64 < index_capacity)
    history = valid & (topk64 < tail_starts.unsqueeze(1))
    tail = (
        valid & (topk64 >= tail_starts.unsqueeze(1)) & (topk64 < verify.unsqueeze(1))
    )
    staging = (
        valid & is_mtp & (topk64 >= verify.unsqueeze(1)) & (topk64 <= current_positions)
    )
    if not is_mtp:
        tail = (
            valid & (topk64 >= tail_starts.unsqueeze(1)) & (topk64 <= current_positions)
        )
    lookup_mask = history.to(torch.int64)

    # framework: lookup_offsets = layout.lookup_offsets(slot_out)
    lookup_offsets = torch.where(
        slot64 < RESIDENT_SLOTS, slot64, slot64 - RESIDENT_SLOTS + REPLACEABLE_BASE)
    fallback_mask = valid & (slot64 == FALLBACK_SENTINEL)
    lookup_offsets = torch.where(
        fallback_mask, torch.full_like(lookup_offsets, FALLBACK_SLOT), lookup_offsets)
    tail_offsets = TAIL_BASE + topk64 - tail_starts.unsqueeze(1)
    staging_offsets = STAGING_BASE + topk64 - verify.unsqueeze(1)
    mapped = torch.where(
        staging, staging_offsets, torch.where(tail, tail_offsets, lookup_offsets))
    mapped = torch.where(valid, mapped, torch.full_like(mapped, INVALID_INDEX))
    dense_miss_mask = (
        miss64.bool() & lookup_mask.bool() & ~fallback_mask
    ).to(torch.int64)
    return mapped.to(torch.int32), dense_miss_mask.to(torch.int32)


def check_invariants(op_name, index, sti, fs, fh, mapped, miss, query, req_pool, qpr):
    assert torch.all(fh[:, 0] == 0), f"{op_name}: head[0] != 0"
    assert torch.all(fh[:, 1] >= 0) and torch.all(fh[:, 1] < SLOT_COUNT), f"{op_name}: cursor OOB"
    for p in range(fh.shape[0]):
        vals = fs[p].cpu().tolist()
        assert len(set(vals)) == FREE_SLOT_COUNT, f"{op_name}: free list duplicates"
        assert all(0 <= v < SLOT_COUNT for v in vals), f"{op_name}: free list OOB"
        assert torch.all(sti[p][vals] == NOT_FOUND), f"{op_name}: free slot still mapped"
    # every allocated miss: index[token] == slot and slot_to_index[slot] == token
    for entry in range(mapped.shape[0]):
        pool = int(req_pool[entry // qpr])
        for e in range(QUERY_WIDTH):
            if int(miss[entry, e]) == 1:
                tok = int(query[entry, e])
                slot = int(index[pool, tok])
                assert 0 <= slot < SLOT_COUNT, f"{op_name}: miss slot OOB"
                assert int(sti[pool, slot]) == tok, f"{op_name}: slot_to_index mismatch"
    return True


def run_case(index_capacity, hit, qpr, seed, is_mtp, req_num=2):
    failures = []
    q, qpos, vstarts, tstarts, idx, sti, fs, fh, rp, qsl = build_state(
        req_num, index_capacity, hit, qpr, seed, mtp=is_mtp)
    # ---- fused (full classification) vs turbo + framework rebuild ----
    # The turbo baseline receives the framework-generated history lookup_mask
    # (valid && token < tail_start) exactly as in production; tail/staging
    # tokens never enter the turbo state.
    tail_start_q = ((vstarts[0].item() // BLOCK_SIZE) * BLOCK_SIZE)
    mask = (
        (q >= 0) & (q < index_capacity) & (q < tail_start_q)
    ).to(torch.int32).contiguous()
    i1, s1, f1, h1 = clone(idx, sti, fs, fh)
    i2, s2, f2, h2 = clone(idx, sti, fs, fh)
    so1, mo1 = run_turbo("dsa_sparse_turbo_lookup_update_batch",
                         i1, s1, f1, h1, rp, qsl, q, mask)
    mapped_ref, miss_ref = rebuild_main_mapped(
        q, so1, mo1, qpos, vstarts, qsl, index_capacity, is_mtp)
    mapped, miss = run_fused_main(
        i2, s2, f2, h2, rp, qsl, q, qpos, vstarts, tstarts,
        rp.shape[0], int(is_mtp))
    out_ok = (
        torch.equal(mapped, mapped_ref) and torch.equal(miss, miss_ref)
        and torch.equal(i1, i2) and torch.equal(s1, s2)
        and torch.equal(f1, f2) and torch.equal(h1, h2)
    )
    heads_ok = torch.all(h1[:, 0] == 0).item() and torch.all(h2[:, 0] == 0).item()
    tag = f"cap={index_capacity//1024}k hit={hit} Q={qpr} mtp={int(is_mtp)}"
    print(f"{tag}: out_bit_exact={out_ok} head0={heads_ok}")
    if not (out_ok and heads_ok):
        failures.append(tag)
    # ---- invariants on the fused post-state (Q>=2 path) ----
    ok = True
    try:
        check_invariants("fused", i2, s2, f2, h2, mapped, miss, q, rp, qpr)
    except AssertionError as ex:
        print(f"{tag} invariant FAIL: {ex}")
        ok = False
    if not ok:
        failures.append(tag + " invariants")
    return failures


def main():
    failures = []
    # Q=1 MTP/non-MTP bit-exact output + post-state, 128k/1024k, 90%/95%
    for index_capacity in (128 * 1024, 1024 * 1024):
        for hit in (0.90, 0.95):
            for is_mtp in (True, False):
                failures += run_case(index_capacity, hit, 1, 1, is_mtp)
    # Q=4: invariants + bit-exact vs rebuild (classification is exact even
    # though turbo's allocation positions deviate from batch at Q>=2).
    for index_capacity in (128 * 1024, 1024 * 1024):
        for is_mtp in (True, False):
            failures += run_case(index_capacity, 0.95, 4, 2, is_mtp)
    # Flush stress: Q=3 at hit 0.30 -> cumulative misses exceed the free list,
    # the flush must replenish mid-request so no history miss is dropped.
    for index_capacity in (128 * 1024, 1024 * 1024):
        for is_mtp in (True, False):
            q, qpos, vstarts, tstarts, idx, sti, fs, fh, rp, qsl = build_state(
                2, index_capacity, 0.30, 3, 3, mtp=is_mtp)
            tail_start_q = ((vstarts[0].item() // BLOCK_SIZE) * BLOCK_SIZE)
            mask = (
                (q >= 0) & (q < index_capacity) & (q < tail_start_q)
            ).to(torch.int32).contiguous()
            i1, s1, f1, h1 = clone(idx, sti, fs, fh)
            so1, mo1 = run_turbo("dsa_sparse_turbo_lookup_update_batch",
                                 i1, s1, f1, h1, rp, qsl, q, mask)
            mapped_ref, miss_ref = rebuild_main_mapped(
                q, so1, mo1, qpos, vstarts, qsl, index_capacity, is_mtp)
            i2, s2, f2, h2 = clone(idx, sti, fs, fh)
            mapped, miss = run_fused_main(
                i2, s2, f2, h2, rp, qsl, q, qpos, vstarts, tstarts,
                rp.shape[0], int(is_mtp))
            ok = (torch.equal(mapped, mapped_ref) and torch.equal(miss, miss_ref)
                  and torch.all(h2[:, 0] == 0))
            try:
                assert ok, "flush mismatch"
                check_invariants("fused-flush", i2, s2, f2, h2, mapped, miss, q, rp, 3)
            except AssertionError as ex:
                print(f"cap={index_capacity//1024}k hit=0.30 Q=3 mtp={int(is_mtp)} flush FAIL: {ex}")
                ok = False
            print(f"cap={index_capacity//1024}k hit=0.30 Q=3 mtp={int(is_mtp)} "
                  f"flush: ok={ok} misses={int(miss.sum().item())}")
            if not ok:
                failures.append(f"cap={index_capacity//1024}k Q=3 flush mtp={int(is_mtp)}")

    if failures:
        print("SANITY FAILURES:", failures)
        return 1
    print("SANITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
