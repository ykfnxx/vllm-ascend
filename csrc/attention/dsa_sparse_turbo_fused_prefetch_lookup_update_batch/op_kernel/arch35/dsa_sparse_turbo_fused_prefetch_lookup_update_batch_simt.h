/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_PREFETCH_SIMT_H
#define DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_PREFETCH_SIMT_H

#include "../dsa_sparse_turbo_fused_prefetch_lookup_update_batch_common.h"

#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace DsaSparseTurboFusedPrefetchLookupUpdateBatch {

constexpr uint32_t kPoolEntryScalar = 0U;
constexpr uint32_t kFreeHeadScalar = 1U;
constexpr uint32_t kCursorScalar = 2U;
constexpr uint32_t kLastVictimScalar = 3U;
constexpr uint32_t kEffectiveScalar = 4U;
constexpr uint32_t kQueryBeginScalar = 5U;
constexpr uint32_t kQueryEndScalar = 6U;
constexpr uint32_t kAllocBaseScalar = 7U;
constexpr uint32_t kRefillCounterScalar = 8U;
constexpr uint32_t kVerifyStartScalar = 9U;
constexpr uint32_t kTailStartScalar = 10U;
constexpr uint32_t kRequestRowScalar = 11U;

// Framework-equivalent invalid index sentinel (INVALID_INDEX = -1 in
// vllm_ascend/dsa_offload/lookup.py).
constexpr int32_t kInvalidIndex = -1;

// Framework-equivalent logical slot mapping
// (HotCacheLayout.lookup_offsets): resident slots keep their id, replaceable
// slots are offset by the replaceable base.
__simt_callee__ inline int32_t LookupOffset(
    int32_t slot,
    int32_t replaceable_base)
{
    return slot < static_cast<int32_t>(
                      DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT)
        ? slot
        : slot - static_cast<int32_t>(
                      DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT) +
              replaceable_base;
}

// Logical destination slot, identical to the framework's
// plan.lookup_slots = layout.lookup_offsets(slot_out) over ALL tokens (the
// dense prefetch Gather receives request_rows separately and composes the
// row base in-kernel).  A budget-capped history miss leaves slot_out at the
// FALLBACK sentinel in the framework, whose lookup_offsets value is
// SLOT_COUNT - RESIDENT_SLOTS + replaceable_base.
__simt_callee__ inline int32_t FallbackLogicalSlot(int32_t replaceable_base)
{
    return static_cast<int32_t>(DSA_SPARSE_TURBO_FUSED_SLOT_COUNT) -
           static_cast<int32_t>(DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT) +
           replaceable_base;
}

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
        tid & (DSA_SPARSE_TURBO_FUSED_WARP_SIZE - 1U);
    const uint32_t warp = tid / DSA_SPARSE_TURBO_FUSED_WARP_SIZE;

    int32_t inclusive = value;
    for (uint32_t delta = 1U;
         delta < DSA_SPARSE_TURBO_FUSED_WARP_SIZE;
         delta <<= 1U) {
        const int32_t upstream = asc_shfl_up(inclusive, delta);
        if (lane >= delta) {
            inclusive += upstream;
        }
    }
    if (lane == DSA_SPARSE_TURBO_FUSED_WARP_SIZE - 1U) {
        warp_totals[warp] = inclusive;
    }
    asc_syncthreads();

    if (warp == 0U) {
        int32_t warp_inclusive =
            lane < DSA_SPARSE_TURBO_FUSED_WARP_COUNT
                ? warp_totals[lane]
                : 0;
        for (uint32_t delta = 1U;
             delta < DSA_SPARSE_TURBO_FUSED_WARP_SIZE;
             delta <<= 1U) {
            const int32_t upstream =
                asc_shfl_up(warp_inclusive, delta);
            if (lane >= delta) {
                warp_inclusive += upstream;
            }
        }
        if (lane < DSA_SPARSE_TURBO_FUSED_WARP_COUNT) {
            warp_totals[lane] = warp_inclusive;
        }
    }
    asc_syncthreads();

    const int32_t warp_prefix =
        warp == 0U ? 0 : warp_totals[warp - 1U];
    return warp_prefix + inclusive - value;
}

// Prefetch maintain, identical to the turbo_prefetch baseline: Plan 3 scans
// only the resident ring [0, RESIDENT_SLOT_COUNT); Plan 2 replaces the
// victim-count pass with a single UB-atomic eviction pass (non-deterministic
// refill order — allowed for the prefetch stream, the L+1 exact lookup
// re-derives).  The overflow fallback reverts trailing allocations; the
// reverted outputs reset destination_slots to -1 and miss_mask to 0.
__simt_callee__ inline void PrefetchMaintain(
    __gm__ int32_t* request_index,
    __gm__ int32_t* request_slot_to_index,
    __gm__ int32_t* request_free_slots,
    __gm__ int32_t* request_free_head,
    __gm__ int32_t* query_index,
    __gm__ int32_t* destination_slots,
    __gm__ int32_t* miss_mask,
    __ubuf__ uint32_t* protected_bits,
    __ubuf__ int32_t* warp_totals,
    __ubuf__ int32_t* scalars,
    __ubuf__ int32_t* alloc_records,
    uint32_t index_capacity,
    int32_t replaceable_base)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count =
        static_cast<uint32_t>(blockDim.x);
    (void)warp_totals;
    constexpr uint32_t resident_chunk =
        DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT /
        DSA_SPARSE_TURBO_FUSED_SIMT_THREADS;
    const int32_t cum_alloc = scalars[kAllocBaseScalar];
    if (cum_alloc <= 0) {
        // Coarse early return: a fully-hit request allocated nothing, so
        // there is nothing to replenish — skip the maintain entirely.
        return;
    }

    // Plan 3: scan only the resident ring [0, RESIDENT_SLOT_COUNT), anchored
    // at cursor % RESIDENT_SLOT_COUNT.  Free-region slots are never scanned.
    const uint32_t resident_cursor =
        static_cast<uint32_t>(scalars[kCursorScalar]) %
        DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT;
    const uint32_t scan_begin = tid * resident_chunk;
    const uint32_t scan_end = scan_begin + resident_chunk;
    if (tid == 0U) {
        scalars[kRefillCounterScalar] = 0;
        scalars[kLastVictimScalar] = DSA_SPARSE_TURBO_FUSED_NOT_FOUND;
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Plan 2: single eviction pass with UB-atomic refill ranks.  A thread
    // breaks as soon as it grabs a rank >= cum_alloc (the budget is already
    // covered), so at high hit rates the scan covers only the first few
    // victim-dense chunks.  Refill order is non-deterministic — allowed for
    // the prefetch stream (L+1 re-derives).
    for (uint32_t position = scan_begin;
         position < scan_end;
         ++position) {
        uint32_t slot = resident_cursor + position;
        if (slot >= DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT) {
            slot -= DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT;
        }
        if (IsProtectedSlot(protected_bits, slot)) {
            continue;
        }
        const int32_t old_token =
            request_slot_to_index[slot];
        if (old_token == DSA_SPARSE_TURBO_FUSED_NOT_FOUND) {
            continue;
        }
        const int32_t rank = asc_atomic_add(
            &scalars[kRefillCounterScalar], 1);
        if (rank >= cum_alloc) {
            // Budget covered (or exceeded by this thread's grab): stop.
            break;
        }
        request_slot_to_index[slot] =
            DSA_SPARSE_TURBO_FUSED_NOT_FOUND;
        if (old_token >= 0 &&
            old_token < static_cast<int32_t>(index_capacity) &&
            request_index[
                static_cast<uint32_t>(old_token)] ==
                static_cast<int32_t>(slot)) {
            request_index[
                static_cast<uint32_t>(old_token)] =
                DSA_SPARSE_TURBO_FUSED_NOT_FOUND;
        }
        // Direct (order-independent) refill; free-list positions
        // [refilled, cum_alloc) retain their original entries (the reverted
        // allocations' slots), so the list stays full.
        request_free_slots[static_cast<uint32_t>(rank)] =
            static_cast<int32_t>(slot);
        if (rank == cum_alloc - 1) {
            scalars[kLastVictimScalar] =
                static_cast<int32_t>(slot);
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    const int32_t refilled = scalars[kRefillCounterScalar];
    int32_t effective_value = cum_alloc;
    if (effective_value > refilled) {
        effective_value = refilled;
    }

    // Overflow fallback: when the atomic eviction falls short of the
    // cumulative allocation, revert the trailing allocations (ranks
    // [effective_value, cum_alloc)) to their pre-request state —
    // index/slot_to_index reset and outputs back to invalid (destination -1,
    // miss_mask 0).  Their free-list entries were never consumed by the
    // refill, so the free list self-heals without extra work.  Runs only on
    // the rare victim-shortage path.
    if (effective_value < cum_alloc) {
        for (uint32_t rank =
                 static_cast<uint32_t>(effective_value) + tid;
             rank < static_cast<uint32_t>(cum_alloc);
             rank += thread_count) {
            const int32_t offset = alloc_records[2U * rank];
            const int32_t slot = alloc_records[2U * rank + 1U];
            const int32_t token = query_index[offset];
            if (request_index[
                    static_cast<uint32_t>(token)] == slot) {
                request_index[
                    static_cast<uint32_t>(token)] =
                    DSA_SPARSE_TURBO_FUSED_NOT_FOUND;
            }
            request_slot_to_index[
                static_cast<uint32_t>(slot)] =
                DSA_SPARSE_TURBO_FUSED_NOT_FOUND;
            destination_slots[offset] = FallbackLogicalSlot(replaceable_base);
            miss_mask[offset] = 0;
        }
        asc_threadfence_block();
        asc_syncthreads();
    }

    if (tid == 0U) {
        // Advance the cursor only when the eviction was complete (the thread
        // holding rank cum_alloc-1 wrote the last victim); on a victim
        // shortage the cursor keeps its entry value — the scan anchor is
        // non-deterministic anyway for the prefetch stream.  Cursor stays in
        // the resident ring [0, RESIDENT_SLOT_COUNT).
        if (effective_value == cum_alloc) {
            const int32_t last_victim =
                scalars[kLastVictimScalar];
            request_free_head[1] =
                last_victim + 1 >=
                        static_cast<int32_t>(
                            DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT)
                    ? 0
                    : last_victim + 1;
            scalars[kCursorScalar] = request_free_head[1];
        }
        // Self-contained transaction: the head is reset so the next
        // invocation can safely consume the free list.
        request_free_head[0] = 0;
    }
    asc_threadfence_block();
    asc_syncthreads();
}

// Fused history classification + lookup + miss allocation for one query row
// of the prefetch stream.
//
// The prefetch stream only touches history KV (framework lookup_mask =
// valid && token < tail_start); tail/staging/current-token entries never
// enter the Lookup state.  Non-history entries get destination = -1 and
// miss_mask = 0.  History hits and allocated misses map to the global
// destination slot (row base + logical offset), exactly the framework's
// plan.lookup_slots consumed by the dense prefetch Gather.
//
// The maintain semantics (Plan 2/3, overflow fallback) are inherited
// unchanged from the turbo_prefetch baseline.
__simt_callee__ inline void ProcessQuery(
    __gm__ int32_t* request_index,
    __gm__ int32_t* request_slot_to_index,
    __gm__ int32_t* request_free_slots,
    __gm__ int32_t* request_free_head,
    __gm__ int32_t* query_index,
    __gm__ int32_t* destination_slots,
    __gm__ int32_t* miss_mask,
    __ubuf__ uint32_t* protected_bits,
    __ubuf__ int32_t* warp_totals,
    __ubuf__ int32_t* scalars,
    __ubuf__ int32_t* alloc_records,
    uint32_t query_id,
    uint32_t index_capacity,
    int32_t replaceable_base)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_SPARSE_TURBO_FUSED_QUERY_WIDTH /
        DSA_SPARSE_TURBO_FUSED_SIMT_THREADS;
    const uint64_t query_base =
        static_cast<uint64_t>(query_id) *
        DSA_SPARSE_TURBO_FUSED_QUERY_WIDTH;
    const uint32_t query_begin = tid * query_chunk;

    const int32_t verify_start = scalars[kVerifyStartScalar];
    const int32_t tail_start = scalars[kTailStartScalar];
    (void)verify_start;

    int32_t query_values[query_chunk];
    int32_t query_is_history[query_chunk];
    int32_t local_destinations[query_chunk];
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
        query_values[local_entry] = token;
        local_slots[local_entry] = DSA_SPARSE_TURBO_FUSED_NOT_FOUND;
        local_miss_candidates[local_entry] = 0;
        local_misses[local_entry] = 0;
        if (token < 0 ||
            token >= static_cast<int32_t>(index_capacity)) {
            query_is_history[local_entry] = 0;
            local_destinations[local_entry] = kInvalidIndex;
        } else if (token < tail_start) {
            // History: the only class the prefetch stream services.
            query_is_history[local_entry] = 1;
            local_destinations[local_entry] = kInvalidIndex;
        } else {
            // tail/staging/current token: never prefetched.
            query_is_history[local_entry] = 0;
            local_destinations[local_entry] = kInvalidIndex;
        }
    }

#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        if (query_is_history[local_entry] == 0) {
            continue;
        }
        const int32_t token = query_values[local_entry];
        const int32_t observed =
            request_index[static_cast<uint32_t>(token)];
        if (observed >= 0 &&
            observed < static_cast<int32_t>(
                           DSA_SPARSE_TURBO_FUSED_SLOT_COUNT)) {
            local_slots[local_entry] = observed;
            local_destinations[local_entry] = LookupOffset(
                observed, replaceable_base);
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
        warp_totals[DSA_SPARSE_TURBO_FUSED_WARP_COUNT - 1U];

    // Cumulative free-list budget across the request's queries: every query
    // allocates from the free list at [base, base + budget), and the
    // request-level maintain replenishes exactly the cumulative allocation.
    const int32_t base = scalars[kAllocBaseScalar];
    // Plan-1 flush (risk A/B mitigation): if this query's misses would push
    // the cumulative allocation past the free-list end, maintain NOW (evict
    // the accumulated allocations, replenish, head back to 0) and continue —
    // so the budget cap never drops a miss.
    if (base > 0 &&
        base + total_misses >
            static_cast<int32_t>(DSA_SPARSE_TURBO_FUSED_FREE_SLOT_COUNT)) {
        PrefetchMaintain(
            request_index,
            request_slot_to_index,
            request_free_slots,
            request_free_head,
            query_index,
            destination_slots,
            miss_mask,
            protected_bits,
            warp_totals,
            scalars,
            alloc_records,
            index_capacity,
            replaceable_base);
        if (tid == 0U) {
            scalars[kAllocBaseScalar] = 0;
        }
        asc_syncthreads();
    }
    const int32_t alloc_base = scalars[kAllocBaseScalar];
    int32_t budget = total_misses;
    if (budget >
        static_cast<int32_t>(DSA_SPARSE_TURBO_FUSED_FREE_SLOT_COUNT) - alloc_base) {
        budget =
            static_cast<int32_t>(DSA_SPARSE_TURBO_FUSED_FREE_SLOT_COUNT) - alloc_base;
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
                        DSA_SPARSE_TURBO_FUSED_SLOT_COUNT)) {
            continue;
        }
        request_slot_to_index[
            static_cast<uint32_t>(slot)] = token;
        request_index[
            static_cast<uint32_t>(token)] = slot;
        local_slots[local_entry] = slot;
        local_destinations[local_entry] = LookupOffset(
            slot, replaceable_base);
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
        // A history miss that could not be allocated (budget cap) keeps the
        // framework-equivalent fallback logical slot
        // (lookup_offsets(FALLBACK_SENTINEL)); miss_mask stays 0 so the
        // Gather never consumes it.
        if (query_is_history[local_entry] != 0 &&
            local_miss_candidates[local_entry] != 0 &&
            local_misses[local_entry] == 0) {
            destination_slots[offset] = FallbackLogicalSlot(replaceable_base);
            miss_mask[offset] = 0;
        } else {
            destination_slots[offset] = local_destinations[local_entry];
            miss_mask[offset] = local_misses[local_entry];
        }
    }
}

// Initialize the fused outputs for an invalid request (pool entry out of
// range, bad query range, or a non-zero free_head fail-closed marker), so the
// caller's output tensors are always fully written.
__simt_callee__ inline void InitializeInvalidQueryRange(
    __gm__ int32_t* destination_slots,
    __gm__ int32_t* miss_mask,
    uint32_t query_begin,
    uint32_t query_end)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_SPARSE_TURBO_FUSED_QUERY_WIDTH /
        DSA_SPARSE_TURBO_FUSED_SIMT_THREADS;
    const uint32_t entry_begin = tid * query_chunk;
    for (uint32_t query_id = query_begin;
         query_id < query_end;
         ++query_id) {
        const uint64_t query_base =
            static_cast<uint64_t>(query_id) *
            DSA_SPARSE_TURBO_FUSED_QUERY_WIDTH;
#pragma unroll
        for (uint32_t local_entry = 0U;
             local_entry < query_chunk;
             ++local_entry) {
            const uint64_t offset =
                query_base + entry_begin + local_entry;
            destination_slots[offset] = kInvalidIndex;
            miss_mask[offset] = 0;
        }
    }
}

__simt_callee__ inline void ProcessRequest(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* request_rows,
    __gm__ int32_t* query_start_loc,
    __gm__ int32_t* query_index,
    __gm__ int32_t* query_positions,
    __gm__ int32_t* verify_starts,
    __gm__ int32_t* destination_slots,
    __gm__ int32_t* miss_mask,
    __ubuf__ uint32_t* shared_scratch,
    uint32_t req_id,
    uint32_t pool_capacity,
    uint32_t query_num,
    bool reuse_scratch,
    uint32_t index_capacity,
    int32_t block_size,
    int32_t replaceable_base)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count = static_cast<uint32_t>(blockDim.x);
    __ubuf__ uint32_t* protected_bits = shared_scratch;
    __ubuf__ int32_t* warp_totals =
        reinterpret_cast<__ubuf__ int32_t*>(
            protected_bits + DSA_SPARSE_TURBO_FUSED_PROTECTED_WORDS);
    __ubuf__ int32_t* scalars =
        warp_totals + DSA_SPARSE_TURBO_FUSED_WARP_COUNT;
    __ubuf__ int32_t* alloc_records =
        scalars + DSA_SPARSE_TURBO_FUSED_SHARED_SCALARS;

    for (uint32_t word = tid;
         word < DSA_SPARSE_TURBO_FUSED_PROTECTED_WORDS;
         word += thread_count) {
        protected_bits[word] = 0U;
    }
    if (tid == 0U) {
        const int32_t pool_entry = request_rows[req_id];
        const int32_t query_begin = query_start_loc[req_id];
        const int32_t query_end = query_start_loc[req_id + 1U];
        scalars[kPoolEntryScalar] = pool_entry;
        scalars[kFreeHeadScalar] = 0;
        scalars[kCursorScalar] = 0;
        scalars[kLastVictimScalar] = DSA_SPARSE_TURBO_FUSED_NOT_FOUND;
        scalars[kEffectiveScalar] = 0;
        scalars[kQueryBeginScalar] = query_begin;
        scalars[kQueryEndScalar] = query_end;
        scalars[kAllocBaseScalar] = 0;
        scalars[kRefillCounterScalar] = 0;
        scalars[kVerifyStartScalar] = 0;
        scalars[kTailStartScalar] = 0;
        scalars[kRequestRowScalar] = pool_entry;
        if (pool_entry >= 0 &&
            pool_entry < static_cast<int32_t>(pool_capacity)) {
            __gm__ int32_t* request_free_head =
                free_head + static_cast<uint64_t>(pool_entry) *
                                DSA_SPARSE_TURBO_FUSED_FREE_HEAD_STRIDE;
            scalars[kFreeHeadScalar] = request_free_head[0];
            int32_t cursor = request_free_head[1];
            if (cursor < 0 ||
                cursor >= static_cast<int32_t>(
                              DSA_SPARSE_TURBO_FUSED_SLOT_COUNT)) {
                cursor = 0;
            }
            scalars[kCursorScalar] = cursor;
            // Fused classification anchors: loaded once per request, consumed
            // by every query of the request.  tail_start mirrors the
            // framework's floor(verify_start / block_size) * block_size.
            const int32_t verify_start = verify_starts[req_id];
            scalars[kVerifyStartScalar] = verify_start;
            const int32_t tail_start =
                (verify_start / block_size) * block_size;
            scalars[kTailStartScalar] = tail_start;
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
    if (pool_entry_value < 0 ||
        pool_entry_value >= static_cast<int32_t>(pool_capacity) ||
        !query_range_valid ||
        scalars[kFreeHeadScalar] != 0) {
        // Fail-closed path: outputs are still fully initialized so callers
        // never observe garbage.
        if (query_range_valid) {
            InitializeInvalidQueryRange(
                destination_slots,
                miss_mask,
                static_cast<uint32_t>(query_begin_value),
                static_cast<uint32_t>(query_end_value));
        }
        return;
    }

    const uint32_t pool_entry =
        static_cast<uint32_t>(pool_entry_value);
    __gm__ int32_t* request_index =
        index + static_cast<uint64_t>(pool_entry) *
                    index_capacity;
    __gm__ int32_t* request_slot_to_index =
        slot_to_index + static_cast<uint64_t>(pool_entry) *
                            DSA_SPARSE_TURBO_FUSED_SLOT_COUNT;
    __gm__ int32_t* request_free_slots =
        free_slots + static_cast<uint64_t>(pool_entry) *
                         DSA_SPARSE_TURBO_FUSED_FREE_SLOT_COUNT;
    __gm__ int32_t* request_free_head =
        free_head + static_cast<uint64_t>(pool_entry) *
                        DSA_SPARSE_TURBO_FUSED_FREE_HEAD_STRIDE;

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
            destination_slots,
            miss_mask,
            protected_bits,
            warp_totals,
            scalars,
            alloc_records,
            query_id,
            index_capacity,
            replaceable_base);
        asc_syncthreads();
    }

    PrefetchMaintain(
        request_index,
        request_slot_to_index,
        request_free_slots,
        request_free_head,
        query_index,
        destination_slots,
        miss_mask,
        protected_bits,
        warp_totals,
        scalars,
        alloc_records,
        index_capacity,
        replaceable_base);
    if (reuse_scratch) {
        asc_syncthreads();
    }
}

__simt_vf__ __launch_bounds__(DSA_SPARSE_TURBO_FUSED_SIMT_THREADS) inline void
DsaSparseTurboFusedPrefetchLookupUpdateBatchSimt(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* request_rows,
    __gm__ int32_t* query_start_loc,
    __gm__ int32_t* query_index,
    __gm__ int32_t* query_positions,
    __gm__ int32_t* verify_starts,
    __gm__ int32_t* destination_slots,
    __gm__ int32_t* miss_mask,
    __ubuf__ uint32_t* shared_scratch,
    uint32_t req_num,
    uint32_t pool_capacity,
    uint32_t query_num,
    uint32_t index_capacity,
    int32_t block_size,
    int32_t replaceable_base)
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
            request_rows,
            query_start_loc,
            query_index,
            query_positions,
            verify_starts,
            destination_slots,
            miss_mask,
            shared_scratch,
            req_id,
            pool_capacity,
            query_num,
            req_id + request_stride < req_num,
            index_capacity,
            block_size,
            replaceable_base);
    }
}

}  // namespace DsaSparseTurboFusedPrefetchLookupUpdateBatch

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_PREFETCH_SIMT_H
