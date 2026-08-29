/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TURBO_SIMT_H
#define DSA_SPARSE_LOOKUP_UPDATE_TURBO_SIMT_H

#include "../dsa_sparse_turbo_lookup_update_batch_common.h"

#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace DsaSparseTurboLookupUpdateBatch {

constexpr uint32_t kPoolEntryScalar = 0U;
constexpr uint32_t kFreeHeadScalar = 1U;
constexpr uint32_t kCursorScalar = 2U;
constexpr uint32_t kLastVictimScalar = 3U;
constexpr uint32_t kEffectiveScalar = 4U;
constexpr uint32_t kQueryBeginScalar = 5U;
constexpr uint32_t kQueryEndScalar = 6U;
constexpr uint32_t kAllocBaseScalar = 7U;

__simt_callee__ inline void ProtectSlot(
    __ubuf__ uint32_t* protected_bits,
    uint32_t slot)
{
    const uint32_t word = slot >> 5U;
    const uint32_t bit = 1U << (slot & 31U);
    asc_atomic_or(protected_bits + word, bit);
}

__simt_callee__ inline bool IsProtectedSlot(
    __ubuf__ const uint32_t* protected_bits,
    uint32_t slot)
{
    const uint32_t word = slot >> 5U;
    const uint32_t bit = 1U << (slot & 31U);
    return (protected_bits[word] & bit) != 0U;
}

__simt_callee__ inline int32_t BlockExclusiveScan(
    int32_t value,
    __ubuf__ int32_t* warp_totals)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t lane =
        tid & (DSA_SPARSE_TURBO_WARP_SIZE - 1U);
    const uint32_t warp = tid / DSA_SPARSE_TURBO_WARP_SIZE;

    int32_t inclusive = value;
    for (uint32_t delta = 1U;
         delta < DSA_SPARSE_TURBO_WARP_SIZE;
         delta <<= 1U) {
        const int32_t upstream = asc_shfl_up(inclusive, delta);
        if (lane >= delta) {
            inclusive += upstream;
        }
    }
    if (lane == DSA_SPARSE_TURBO_WARP_SIZE - 1U) {
        warp_totals[warp] = inclusive;
    }
    asc_syncthreads();

    if (warp == 0U) {
        int32_t warp_inclusive =
            lane < DSA_SPARSE_TURBO_WARP_COUNT
                ? warp_totals[lane]
                : 0;
        for (uint32_t delta = 1U;
             delta < DSA_SPARSE_TURBO_WARP_SIZE;
             delta <<= 1U) {
            const int32_t upstream =
                asc_shfl_up(warp_inclusive, delta);
            if (lane >= delta) {
                warp_inclusive += upstream;
            }
        }
        if (lane < DSA_SPARSE_TURBO_WARP_COUNT) {
            warp_totals[lane] = warp_inclusive;
        }
    }
    asc_syncthreads();

    const int32_t warp_prefix =
        warp == 0U ? 0 : warp_totals[warp - 1U];
    return warp_prefix + inclusive - value;
}

__simt_callee__ inline void InitializeQueryRange(
    __gm__ int32_t* query_index,
    __gm__ int32_t* lookup_mask,
    __gm__ int32_t* slot_out,
    __gm__ int32_t* miss_out,
    uint32_t query_begin,
    uint32_t query_end,
    uint32_t index_capacity)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_SPARSE_TURBO_QUERY_WIDTH /
        DSA_SPARSE_TURBO_SIMT_THREADS;
    const uint32_t entry_begin = tid * query_chunk;
    for (uint32_t query_id = query_begin;
         query_id < query_end;
         ++query_id) {
        const uint64_t query_base =
            static_cast<uint64_t>(query_id) *
            DSA_SPARSE_TURBO_QUERY_WIDTH;
#pragma unroll
        for (uint32_t local_entry = 0U;
             local_entry < query_chunk;
             ++local_entry) {
            const uint64_t offset =
                query_base + entry_begin + local_entry;
            const int32_t token = query_index[offset];
            const bool active =
                lookup_mask[offset] != 0 && token >= 0 &&
                token < static_cast<int32_t>(index_capacity);
            slot_out[offset] =
                active ? DSA_SPARSE_TURBO_FALLBACK_SLOT
                       : DSA_SPARSE_TURBO_NOT_FOUND;
            miss_out[offset] = 0;
        }
    }
}

__simt_callee__ inline void MaintainRequest(
    __gm__ int32_t* request_index,
    __gm__ int32_t* request_slot_to_index,
    __gm__ int32_t* request_free_slots,
    __gm__ int32_t* request_free_head,
    __gm__ int32_t* query_index,
    __gm__ int32_t* slot_out,
    __gm__ int32_t* miss_out,
    __ubuf__ uint32_t* protected_bits,
    __ubuf__ int32_t* warp_totals,
    __ubuf__ int32_t* scalars,
    __ubuf__ int32_t* alloc_records,
    uint32_t index_capacity);

// Lookup + miss allocation for one query row.  Unlike the fused batch
// baseline, this does NOT maintain (no victim count, no eviction): the
// maintain is consolidated to ONE pass per request (the coarse variant's
// structural lesson — the maintain slot scan dominates the cost, so it must
// run once per transaction, not once per query).  The per-query allocations
// are bounded by the cumulative free-list budget (FREE_SLOT_COUNT) and
// recorded in the allocation ledger, so the request-level maintain can revert
// any allocation that exceeds the evictable victim count (overflow fallback
// with the same output semantics as the baseline's per-query safe_alloc cap).
//
// Plan-1 alignment (coarse design_and_test.md §9.3/§9.7):
//   - the GM free_head[0] progress marker (crash safety condition 2) is
//     written after every query's allocation; the final maintain resets it.
//   - the flush mechanism (risk A/B mitigation): when the cumulative
//     allocation would exceed FREE_SLOT_COUNT, a mid-request maintain
//     replenishes the free list first, so no miss is ever dropped by the
//     budget cap (exact-path safe).
__simt_callee__ inline void ProcessQuery(
    __gm__ int32_t* request_index,
    __gm__ int32_t* request_slot_to_index,
    __gm__ int32_t* request_free_slots,
    __gm__ int32_t* request_free_head,
    __gm__ int32_t* query_index,
    __gm__ int32_t* lookup_mask,
    __gm__ int32_t* slot_out,
    __gm__ int32_t* miss_out,
    __ubuf__ uint32_t* protected_bits,
    __ubuf__ int32_t* warp_totals,
    __ubuf__ int32_t* scalars,
    __ubuf__ int32_t* alloc_records,
    uint32_t query_id,
    uint32_t index_capacity)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_SPARSE_TURBO_QUERY_WIDTH /
        DSA_SPARSE_TURBO_SIMT_THREADS;
    const uint64_t query_base =
        static_cast<uint64_t>(query_id) *
        DSA_SPARSE_TURBO_QUERY_WIDTH;
    const uint32_t query_begin = tid * query_chunk;

    int32_t query_values[query_chunk];
    int32_t query_masks[query_chunk];
    int32_t local_slots[query_chunk];
    int32_t local_miss_candidates[query_chunk];
    int32_t local_misses[query_chunk];
#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        const uint64_t offset =
            query_base + query_begin + local_entry;
        const int32_t token = query_index[offset];
        const int32_t mask = lookup_mask[offset];
        query_values[local_entry] = token;
        query_masks[local_entry] = mask;
        local_slots[local_entry] =
            mask != 0 && token >= 0 &&
                    token < static_cast<int32_t>(index_capacity)
                ? DSA_SPARSE_TURBO_FALLBACK_SLOT
                : DSA_SPARSE_TURBO_NOT_FOUND;
        local_miss_candidates[local_entry] = 0;
        local_misses[local_entry] = 0;
    }

#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        if (query_masks[local_entry] == 0) {
            continue;
        }
        const int32_t token = query_values[local_entry];
        if (token < 0 ||
            token >= static_cast<int32_t>(index_capacity)) {
            continue;
        }
        const int32_t observed =
            request_index[static_cast<uint32_t>(token)];
        if (observed >= 0 &&
            observed < static_cast<int32_t>(
                           DSA_SPARSE_TURBO_SLOT_COUNT)) {
            local_slots[local_entry] = observed;
            ProtectSlot(protected_bits,
                        static_cast<uint32_t>(observed));
        } else {
            local_miss_candidates[local_entry] = 1;
        }
    }

    int32_t local_miss_count = 0;
#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        local_miss_count += local_miss_candidates[local_entry];
    }
    const int32_t miss_prefix =
        BlockExclusiveScan(local_miss_count, warp_totals);
    const int32_t total_misses =
        warp_totals[DSA_SPARSE_TURBO_WARP_COUNT - 1U];

    // Cumulative free-list budget across the request's queries: every query
    // allocates from the free list at [base, base + budget), and the
    // request-level maintain replenishes exactly the cumulative allocation.
    const int32_t base = scalars[kAllocBaseScalar];
    // Plan-1 flush (risk A/B mitigation): if this query's misses would push
    // the cumulative allocation past the free-list end, maintain NOW (evict
    // the accumulated allocations, replenish, head back to 0) and continue —
    // so the budget cap never drops a miss, keeping the exact path safe.
    if (base > 0 &&
        base + total_misses >
            static_cast<int32_t>(DSA_SPARSE_TURBO_FREE_SLOT_COUNT)) {
        MaintainRequest(
            request_index,
            request_slot_to_index,
            request_free_slots,
            request_free_head,
            query_index,
            slot_out,
            miss_out,
            protected_bits,
            warp_totals,
            scalars,
            alloc_records,
            index_capacity);
        if (tid == 0U) {
            scalars[kAllocBaseScalar] = 0;
        }
        asc_syncthreads();
    }
    const int32_t alloc_base = scalars[kAllocBaseScalar];
    int32_t budget = total_misses;
    if (budget >
        static_cast<int32_t>(DSA_SPARSE_TURBO_FREE_SLOT_COUNT) - alloc_base) {
        budget =
            static_cast<int32_t>(DSA_SPARSE_TURBO_FREE_SLOT_COUNT) - alloc_base;
    }

    int32_t local_rank = 0;
#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        if (local_miss_candidates[local_entry] == 0) {
            continue;
        }
        const int32_t miss_rank = miss_prefix + local_rank;
        ++local_rank;
        if (miss_rank >= budget) {
            continue;
        }
        const int32_t token = query_values[local_entry];
        const int32_t slot =
            request_free_slots[
                static_cast<uint32_t>(alloc_base + miss_rank)];
        if (slot < 0 ||
            slot >= static_cast<int32_t>(
                        DSA_SPARSE_TURBO_SLOT_COUNT)) {
            continue;
        }
        request_slot_to_index[
            static_cast<uint32_t>(slot)] = token;
        request_index[
            static_cast<uint32_t>(token)] = slot;
        local_slots[local_entry] = slot;
        local_misses[local_entry] = 1;
        ProtectSlot(protected_bits,
                    static_cast<uint32_t>(slot));
        const uint32_t record_index =
            static_cast<uint32_t>(alloc_base + miss_rank);
        alloc_records[2U * record_index] =
            static_cast<int32_t>(
                query_base + query_begin + local_entry);
        alloc_records[2U * record_index + 1U] = slot;
    }
    asc_threadfence_block();
    asc_syncthreads();
    if (tid == 0U) {
        scalars[kAllocBaseScalar] = alloc_base + budget;
        // Plan-1 crash-safety progress marker: free_head[0] carries the
        // consumed count (non-zero) until the final maintain resets it, so a
        // kernel crash mid-request makes the next invocation fail closed.
        request_free_head[0] = alloc_base + budget;
    }
    asc_syncthreads();

#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        const uint64_t offset =
            query_base + query_begin + local_entry;
        slot_out[offset] = local_slots[local_entry];
        miss_out[offset] = local_misses[local_entry];
    }
}

// Request-level maintain, consolidated from the baseline's per-query passes
// (the coarse variant's design): ONE victim-count scan and ONE eviction pass
// over the request's complete protection set (all queries' hits and
// allocations), instead of one full slot scan per query.  Kept from the
// coarse kernel: the zero-atomic early exit (threads whose victim prefix
// already covers the eviction budget skip the evict pass) and the early
// return when nothing was allocated (a fully-hit request skips both maintain
// scans entirely).  The baseline's bitmask protection table (UBUF, B2) and
// warp-shuffle prefix scans (both stronger than the coarse's uint8 GM table
// and hierarchical 16x16 prefix) are inherited unchanged.
__simt_callee__ inline void MaintainRequest(
    __gm__ int32_t* request_index,
    __gm__ int32_t* request_slot_to_index,
    __gm__ int32_t* request_free_slots,
    __gm__ int32_t* request_free_head,
    __gm__ int32_t* query_index,
    __gm__ int32_t* slot_out,
    __gm__ int32_t* miss_out,
    __ubuf__ uint32_t* protected_bits,
    __ubuf__ int32_t* warp_totals,
    __ubuf__ int32_t* scalars,
    __ubuf__ int32_t* alloc_records,
    uint32_t index_capacity)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count =
        static_cast<uint32_t>(blockDim.x);
    constexpr uint32_t slot_chunk =
        DSA_SPARSE_TURBO_SLOT_COUNT /
        DSA_SPARSE_TURBO_SIMT_THREADS;
    const int32_t cum_alloc = scalars[kAllocBaseScalar];
    if (cum_alloc <= 0) {
        // Coarse early return: a fully-hit request allocated nothing, so
        // there is nothing to replenish — skip both maintain scans.
        return;
    }

    const uint32_t scan_begin = tid * slot_chunk;
    const uint32_t scan_end = scan_begin + slot_chunk;
    const uint32_t cursor =
        static_cast<uint32_t>(scalars[kCursorScalar]);
    int32_t local_victim_count = 0;
    for (uint32_t position = scan_begin;
         position < scan_end;
         ++position) {
        uint32_t slot = cursor + position;
        if (slot >= DSA_SPARSE_TURBO_SLOT_COUNT) {
            slot -= DSA_SPARSE_TURBO_SLOT_COUNT;
        }
        if (!IsProtectedSlot(protected_bits, slot) &&
            request_slot_to_index[slot] !=
                DSA_SPARSE_TURBO_NOT_FOUND) {
            ++local_victim_count;
        }
    }
    const int32_t victim_prefix =
        BlockExclusiveScan(local_victim_count, warp_totals);
    const int32_t total_victims =
        warp_totals[DSA_SPARSE_TURBO_WARP_COUNT - 1U];
    if (tid == 0U) {
        int32_t effective = cum_alloc;
        if (effective > total_victims) {
            effective = total_victims;
        }
        scalars[kEffectiveScalar] = effective;
        scalars[kLastVictimScalar] = DSA_SPARSE_TURBO_NOT_FOUND;
    }
    asc_syncthreads();
    const int32_t effective = scalars[kEffectiveScalar];

    int32_t victim_rank = victim_prefix;
    if (victim_prefix < effective) {
        for (uint32_t position = scan_begin;
             position < scan_end;
             ++position) {
            uint32_t slot = cursor + position;
            if (slot >= DSA_SPARSE_TURBO_SLOT_COUNT) {
                slot -= DSA_SPARSE_TURBO_SLOT_COUNT;
            }
            if (IsProtectedSlot(protected_bits, slot)) {
                continue;
            }
            const int32_t old_token =
                request_slot_to_index[slot];
            if (old_token == DSA_SPARSE_TURBO_NOT_FOUND) {
                continue;
            }
            if (victim_rank < effective) {
                request_slot_to_index[slot] =
                    DSA_SPARSE_TURBO_NOT_FOUND;
                if (old_token >= 0 &&
                    old_token < static_cast<int32_t>(index_capacity) &&
                    request_index[
                        static_cast<uint32_t>(old_token)] ==
                        static_cast<int32_t>(slot)) {
                    request_index[
                        static_cast<uint32_t>(old_token)] =
                        DSA_SPARSE_TURBO_NOT_FOUND;
                }
                // LIFO reverse refill, same as the fused baseline.
                // Free-list positions [effective, cum_alloc) retain their
                // original entries (the reverted allocations' slots), so the
                // list stays full.
                request_free_slots[static_cast<uint32_t>(
                    effective - 1 - victim_rank)] =
                    static_cast<int32_t>(slot);
                if (victim_rank == effective - 1) {
                    scalars[kLastVictimScalar] =
                        static_cast<int32_t>(slot);
                }
            }
            ++victim_rank;
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Overflow fallback: when the evictable victims fall short of the
    // cumulative allocation, revert the trailing allocations (ranks
    // [effective, cum_alloc)) to their pre-request state — index/slot_to_index
    // reset and outputs back to the fallback value — mirroring the baseline's
    // per-query safe_alloc cap.  Their free-list entries were never consumed
    // by the refill, so the free list self-heals without extra work.  Runs
    // only on the rare victim-shortage path.
    if (effective < cum_alloc) {
        for (uint32_t rank = static_cast<uint32_t>(effective) + tid;
             rank < static_cast<uint32_t>(cum_alloc);
             rank += thread_count) {
            const int32_t offset = alloc_records[2U * rank];
            const int32_t slot = alloc_records[2U * rank + 1U];
            const int32_t token = query_index[offset];
            if (request_index[
                    static_cast<uint32_t>(token)] == slot) {
                request_index[
                    static_cast<uint32_t>(token)] =
                    DSA_SPARSE_TURBO_NOT_FOUND;
            }
            request_slot_to_index[
                static_cast<uint32_t>(slot)] =
                DSA_SPARSE_TURBO_NOT_FOUND;
            slot_out[offset] = DSA_SPARSE_TURBO_FALLBACK_SLOT;
            miss_out[offset] = 0;
        }
        asc_threadfence_block();
        asc_syncthreads();
    }

    if (tid == 0U) {
        if (effective > 0) {
            const int32_t last_victim =
                scalars[kLastVictimScalar];
            request_free_head[1] =
                last_victim + 1 >=
                        static_cast<int32_t>(
                            DSA_SPARSE_TURBO_SLOT_COUNT)
                    ? 0
                    : last_victim + 1;
            scalars[kCursorScalar] = request_free_head[1];
        }
        // Self-contained transaction: the head is reset so the next
        // invocation (or the exact-path consumer) can safely consume the
        // free list.
        request_free_head[0] = 0;
    }
    asc_threadfence_block();
    asc_syncthreads();
}

__simt_callee__ inline void ProcessRequest(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* req_pool_entries,
    __gm__ int32_t* query_start_loc,
    __gm__ int32_t* query_index,
    __gm__ int32_t* lookup_mask,
    __gm__ int32_t* slot_out,
    __gm__ int32_t* miss_out,
    __ubuf__ uint32_t* shared_scratch,
    uint32_t req_id,
    uint32_t pool_capacity,
    uint32_t query_num,
    bool reuse_scratch,
    uint32_t index_capacity)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count = static_cast<uint32_t>(blockDim.x);
    __ubuf__ uint32_t* protected_bits = shared_scratch;
    __ubuf__ int32_t* warp_totals =
        reinterpret_cast<__ubuf__ int32_t*>(
            protected_bits + DSA_SPARSE_TURBO_PROTECTED_WORDS);
    __ubuf__ int32_t* scalars =
        warp_totals + DSA_SPARSE_TURBO_WARP_COUNT;
    __ubuf__ int32_t* alloc_records =
        scalars + DSA_SPARSE_TURBO_SHARED_SCALARS;

    for (uint32_t word = tid;
         word < DSA_SPARSE_TURBO_PROTECTED_WORDS;
         word += thread_count) {
        protected_bits[word] = 0U;
    }
    if (tid == 0U) {
        const int32_t pool_entry = req_pool_entries[req_id];
        const int32_t query_begin = query_start_loc[req_id];
        const int32_t query_end = query_start_loc[req_id + 1U];
        scalars[kPoolEntryScalar] = pool_entry;
        scalars[kFreeHeadScalar] = 0;
        scalars[kCursorScalar] = 0;
        scalars[kLastVictimScalar] = DSA_SPARSE_TURBO_NOT_FOUND;
        scalars[kEffectiveScalar] = 0;
        scalars[kQueryBeginScalar] = query_begin;
        scalars[kQueryEndScalar] = query_end;
        scalars[kAllocBaseScalar] = 0;
        if (pool_entry >= 0 &&
            pool_entry < static_cast<int32_t>(pool_capacity)) {
            __gm__ int32_t* request_free_head =
                free_head + static_cast<uint64_t>(pool_entry) *
                                DSA_SPARSE_TURBO_FREE_HEAD_STRIDE;
            scalars[kFreeHeadScalar] = request_free_head[0];
            int32_t cursor = request_free_head[1];
            if (cursor < 0 ||
                cursor >= static_cast<int32_t>(
                              DSA_SPARSE_TURBO_SLOT_COUNT)) {
                cursor = 0;
            }
            scalars[kCursorScalar] = cursor;
        }
    }
    asc_syncthreads();

    const int32_t pool_entry_value = scalars[kPoolEntryScalar];
    const int32_t query_begin_value = scalars[kQueryBeginScalar];
    const int32_t query_end_value = scalars[kQueryEndScalar];
    const bool query_range_valid =
        query_begin_value >= 0 &&
        query_end_value > query_begin_value &&
        query_end_value <= static_cast<int32_t>(query_num);
    if (query_range_valid) {
        InitializeQueryRange(
            query_index,
            lookup_mask,
            slot_out,
            miss_out,
            static_cast<uint32_t>(query_begin_value),
            static_cast<uint32_t>(query_end_value),
            index_capacity);
        asc_syncthreads();
    }
    if (pool_entry_value < 0 ||
        pool_entry_value >= static_cast<int32_t>(pool_capacity) ||
        !query_range_valid ||
        scalars[kFreeHeadScalar] != 0) {
        return;
    }

    const uint32_t pool_entry =
        static_cast<uint32_t>(pool_entry_value);
    __gm__ int32_t* request_index =
        index + static_cast<uint64_t>(pool_entry) *
                    index_capacity;
    __gm__ int32_t* request_slot_to_index =
        slot_to_index + static_cast<uint64_t>(pool_entry) *
                            DSA_SPARSE_TURBO_SLOT_COUNT;
    __gm__ int32_t* request_free_slots =
        free_slots + static_cast<uint64_t>(pool_entry) *
                         DSA_SPARSE_TURBO_FREE_SLOT_COUNT;
    __gm__ int32_t* request_free_head =
        free_head + static_cast<uint64_t>(pool_entry) *
                        DSA_SPARSE_TURBO_FREE_HEAD_STRIDE;

    for (uint32_t query_id =
             static_cast<uint32_t>(query_begin_value);
         query_id < static_cast<uint32_t>(query_end_value);
         ++query_id) {
        ProcessQuery(
            request_index,
            request_slot_to_index,
            request_free_slots,
            request_free_head,
            query_index,
            lookup_mask,
            slot_out,
            miss_out,
            protected_bits,
            warp_totals,
            scalars,
            alloc_records,
            query_id,
            index_capacity);
        asc_syncthreads();
    }

    MaintainRequest(
        request_index,
        request_slot_to_index,
        request_free_slots,
        request_free_head,
        query_index,
        slot_out,
        miss_out,
        protected_bits,
        warp_totals,
        scalars,
        alloc_records,
        index_capacity);
    if (reuse_scratch) {
        asc_syncthreads();
    }
}

__simt_vf__ __launch_bounds__(DSA_SPARSE_TURBO_SIMT_THREADS) inline void
DsaSparseTurboLookupUpdateBatchSimt(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* req_pool_entries,
    __gm__ int32_t* query_start_loc,
    __gm__ int32_t* query_index,
    __gm__ int32_t* lookup_mask,
    __gm__ int32_t* slot_out,
    __gm__ int32_t* miss_out,
    __ubuf__ uint32_t* shared_scratch,
    uint32_t req_num,
    uint32_t pool_capacity,
    uint32_t query_num,
    uint32_t index_capacity)
{
    const uint32_t request_stride = static_cast<uint32_t>(gridDim.x);
    for (uint32_t req_id = static_cast<uint32_t>(blockIdx.x);
         req_id < req_num;
         req_id += request_stride) {
        ProcessRequest(
            index,
            slot_to_index,
            free_slots,
            free_head,
            req_pool_entries,
            query_start_loc,
            query_index,
            lookup_mask,
            slot_out,
            miss_out,
            shared_scratch,
            req_id,
            pool_capacity,
            query_num,
            req_id + request_stride < req_num,
            index_capacity);
    }
}

}  // namespace DsaSparseTurboLookupUpdateBatch

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TURBO_SIMT_H
