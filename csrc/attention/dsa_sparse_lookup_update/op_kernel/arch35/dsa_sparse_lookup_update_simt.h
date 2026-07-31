/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_SIMT_H
#define DSA_SPARSE_LOOKUP_UPDATE_SIMT_H

#include "../dsa_sparse_lookup_update_common.h"

#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace DsaSparseLookupUpdate {

constexpr uint32_t kPoolEntryScalar = 0U;
constexpr uint32_t kFreeHeadScalar = 1U;
constexpr uint32_t kCursorScalar = 2U;
constexpr uint32_t kLastVictimScalar = 3U;

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

// Return the exclusive prefix for this thread. The final warp total contains
// the block-wide sum after this function returns.
__simt_callee__ inline int32_t BlockExclusiveScan(
    int32_t value,
    __ubuf__ int32_t* warp_totals)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t lane = tid & (DSA_SPARSE_WARP_SIZE - 1U);
    const uint32_t warp = tid / DSA_SPARSE_WARP_SIZE;

    int32_t inclusive = value;
    for (uint32_t delta = 1U;
         delta < DSA_SPARSE_WARP_SIZE;
         delta <<= 1U) {
        const int32_t upstream =
            asc_shfl_up(inclusive, delta);
        if (lane >= delta) {
            inclusive += upstream;
        }
    }
    if (lane == DSA_SPARSE_WARP_SIZE - 1U) {
        warp_totals[warp] = inclusive;
    }
    asc_syncthreads();

    if (warp == 0U) {
        int32_t warp_inclusive =
            lane < DSA_SPARSE_WARP_COUNT
                ? warp_totals[lane]
                : 0;
        for (uint32_t delta = 1U;
             delta < DSA_SPARSE_WARP_SIZE;
             delta <<= 1U) {
            const int32_t upstream =
                asc_shfl_up(warp_inclusive, delta);
            if (lane >= delta) {
                warp_inclusive += upstream;
            }
        }
        if (lane < DSA_SPARSE_WARP_COUNT) {
            warp_totals[lane] = warp_inclusive;
        }
    }
    asc_syncthreads();

    const int32_t warp_prefix =
        warp == 0U ? 0 : warp_totals[warp - 1U];
    return warp_prefix + inclusive - value;
}

__simt_vf__ __launch_bounds__(DSA_SPARSE_SIMT_THREADS) inline void
DsaSparseLookupUpdateSimt(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* req_pool_entries,
    __gm__ int32_t* query_index,
    __gm__ int32_t* lookup_mask,
    __gm__ int32_t* slot_out,
    __gm__ int32_t* miss_out,
    __ubuf__ uint32_t* shared_scratch,
    uint32_t req_id,
    uint32_t pool_capacity)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count =
        static_cast<uint32_t>(blockDim.x);
    constexpr uint32_t query_chunk =
        DSA_SPARSE_QUERY_COUNT / DSA_SPARSE_SIMT_THREADS;
    constexpr uint32_t slot_chunk =
        DSA_SPARSE_SLOT_COUNT / DSA_SPARSE_SIMT_THREADS;

    const uint64_t query_base =
        static_cast<uint64_t>(req_id) * DSA_SPARSE_QUERY_COUNT;
    const uint32_t query_begin = tid * query_chunk;

    int32_t query_values[query_chunk];
    int32_t query_masks[query_chunk];
    int32_t local_slots[query_chunk];
    int32_t local_misses[query_chunk];
#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        const uint64_t offset =
            query_base + query_begin + local_entry;
        query_values[local_entry] = query_index[offset];
        query_masks[local_entry] = lookup_mask[offset];
        local_slots[local_entry] = DSA_SPARSE_NOT_FOUND;
        local_misses[local_entry] = 0;
    }

    __ubuf__ uint32_t* protected_bits = shared_scratch;
    __ubuf__ int32_t* warp_totals =
        reinterpret_cast<__ubuf__ int32_t*>(
            protected_bits + DSA_SPARSE_PROTECTED_WORDS);
    __ubuf__ int32_t* scalars =
        warp_totals + DSA_SPARSE_WARP_COUNT;

    for (uint32_t word = tid;
         word < DSA_SPARSE_PROTECTED_WORDS;
         word += thread_count) {
        protected_bits[word] = 0U;
    }
    if (tid == 0U) {
        const int32_t pool_entry_value =
            req_pool_entries[req_id];
        scalars[kPoolEntryScalar] = pool_entry_value;
        scalars[kFreeHeadScalar] = 0;
        scalars[kCursorScalar] = 0;
        scalars[kLastVictimScalar] =
            DSA_SPARSE_NOT_FOUND;
        if (pool_entry_value >= 0 &&
            pool_entry_value <
                static_cast<int32_t>(pool_capacity)) {
            __gm__ int32_t* request_free_head =
                free_head +
                static_cast<uint64_t>(pool_entry_value) *
                    DSA_SPARSE_FREE_HEAD_STRIDE;
            scalars[kFreeHeadScalar] =
                request_free_head[0];
            int32_t cursor = request_free_head[1];
            if (cursor < 0 ||
                cursor >= static_cast<int32_t>(
                              DSA_SPARSE_SLOT_COUNT)) {
                cursor = 0;
            }
            scalars[kCursorScalar] = cursor;
        }
    }
    asc_syncthreads();

    const int32_t pool_entry_value =
        scalars[kPoolEntryScalar];
    if (pool_entry_value < 0 ||
        pool_entry_value >=
            static_cast<int32_t>(pool_capacity)) {
#pragma unroll
        for (uint32_t local_entry = 0U;
             local_entry < query_chunk;
             ++local_entry) {
            const uint64_t offset =
                query_base + query_begin + local_entry;
            slot_out[offset] = local_slots[local_entry];
            miss_out[offset] = local_misses[local_entry];
        }
        return;
    }
    if (scalars[kFreeHeadScalar] != 0) {
        // This fused operator owns the complete lookup/maintain transaction.
        // A non-zero entry head means the row was left mid-transaction by a
        // different producer; fail closed instead of consuming a partial list.
#pragma unroll
        for (uint32_t local_entry = 0U;
             local_entry < query_chunk;
             ++local_entry) {
            const uint64_t offset =
                query_base + query_begin + local_entry;
            slot_out[offset] = local_slots[local_entry];
            miss_out[offset] = local_misses[local_entry];
        }
        return;
    }

    const uint32_t pool_entry =
        static_cast<uint32_t>(pool_entry_value);
    __gm__ int32_t* request_index =
        index +
        static_cast<uint64_t>(pool_entry) *
            DSA_SPARSE_INDEX_CAPACITY;
    __gm__ int32_t* request_slot_to_index =
        slot_to_index +
        static_cast<uint64_t>(pool_entry) *
            DSA_SPARSE_SLOT_COUNT;
    __gm__ int32_t* request_free_slots =
        free_slots +
        static_cast<uint64_t>(pool_entry) *
            DSA_SPARSE_FREE_SLOT_COUNT;
    __gm__ int32_t* request_free_head =
        free_head +
        static_cast<uint64_t>(pool_entry) *
            DSA_SPARSE_FREE_HEAD_STRIDE;

    // First pass: return resident hits and place a deterministic negative
    // claim in index[token] for each missing token. A smaller flat query
    // position always replaces a later claim, so duplicate misses have one
    // stable owner independent of SIMT scheduling.
#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        const uint32_t entry = query_begin + local_entry;
        if (query_masks[local_entry] == 0) {
            continue;
        }
        const int32_t token = query_values[local_entry];
        if (token < 0 ||
            token >= static_cast<int32_t>(
                         DSA_SPARSE_INDEX_CAPACITY)) {
            continue;
        }

        __gm__ int32_t* token_slot =
            request_index + static_cast<uint32_t>(token);
        int32_t observed = *token_slot;
        if (observed >= 0 &&
            observed <
                static_cast<int32_t>(DSA_SPARSE_SLOT_COUNT)) {
            local_slots[local_entry] = observed;
            ProtectSlot(
                protected_bits,
                static_cast<uint32_t>(observed));
            continue;
        }

        const int32_t desired_claim =
            DSA_SPARSE_CLAIM_BASE -
            static_cast<int32_t>(entry);
        while (observed == DSA_SPARSE_NOT_FOUND ||
               observed <= DSA_SPARSE_CLAIM_BASE) {
            if (observed <= DSA_SPARSE_CLAIM_BASE) {
                const int32_t claimed_entry =
                    DSA_SPARSE_CLAIM_BASE - observed;
                if (claimed_entry <=
                    static_cast<int32_t>(entry)) {
                    break;
                }
            }
            const int32_t old = asc_atomic_cas(
                token_slot, observed, desired_claim);
            if (old == observed) {
                break;
            }
            observed = old;
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Count canonical misses per contiguous query chunk. BlockExclusiveScan
    // preserves input order for free-list allocation.
    int32_t local_miss_count = 0;
#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        if (query_masks[local_entry] == 0 ||
            local_slots[local_entry] !=
                DSA_SPARSE_NOT_FOUND) {
            continue;
        }
        const uint32_t entry = query_begin + local_entry;
        const int32_t token = query_values[local_entry];
        if (token < 0 ||
            token >= static_cast<int32_t>(
                         DSA_SPARSE_INDEX_CAPACITY)) {
            continue;
        }
        const int32_t claim =
            DSA_SPARSE_CLAIM_BASE -
            static_cast<int32_t>(entry);
        if (request_index[
                static_cast<uint32_t>(token)] == claim) {
            ++local_miss_count;
        }
    }
    const int32_t miss_prefix =
        BlockExclusiveScan(local_miss_count, warp_totals);
    const int32_t total_misses =
        warp_totals[DSA_SPARSE_WARP_COUNT - 1U];
    const int32_t head_start =
        scalars[kFreeHeadScalar];

    int32_t local_rank = 0;
#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        if (query_masks[local_entry] == 0 ||
            local_slots[local_entry] !=
                DSA_SPARSE_NOT_FOUND) {
            continue;
        }
        const uint32_t entry = query_begin + local_entry;
        const int32_t token = query_values[local_entry];
        if (token < 0 ||
            token >= static_cast<int32_t>(
                         DSA_SPARSE_INDEX_CAPACITY)) {
            continue;
        }
        const int32_t claim =
            DSA_SPARSE_CLAIM_BASE -
            static_cast<int32_t>(entry);
        if (request_index[
                static_cast<uint32_t>(token)] != claim) {
            continue;
        }

        const int32_t miss_rank =
            miss_prefix + local_rank;
        ++local_rank;
        const int32_t free_offset = head_start + miss_rank;
        if (free_offset < 0 ||
            free_offset >=
                static_cast<int32_t>(
                    DSA_SPARSE_FREE_SLOT_COUNT)) {
            request_index[
                static_cast<uint32_t>(token)] =
                DSA_SPARSE_NOT_FOUND;
            continue;
        }
        const int32_t slot =
            request_free_slots[
                static_cast<uint32_t>(free_offset)];
        if (slot < 0 ||
            slot >= static_cast<int32_t>(
                        DSA_SPARSE_SLOT_COUNT)) {
            request_index[
                static_cast<uint32_t>(token)] =
                DSA_SPARSE_NOT_FOUND;
            continue;
        }
        request_slot_to_index[
            static_cast<uint32_t>(slot)] = token;
        request_index[
            static_cast<uint32_t>(token)] = slot;
        local_slots[local_entry] = slot;
        local_misses[local_entry] = 1;
        ProtectSlot(
            protected_bits,
            static_cast<uint32_t>(slot));
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Duplicate followers observe the canonical occurrence's installed slot.
    // Invalid and masked entries retain the initialized (-1, 0) result.
#pragma unroll
    for (uint32_t local_entry = 0U;
         local_entry < query_chunk;
         ++local_entry) {
        if (local_slots[local_entry] !=
                DSA_SPARSE_NOT_FOUND ||
            query_masks[local_entry] == 0) {
            continue;
        }
        const int32_t token = query_values[local_entry];
        if (token < 0 ||
            token >= static_cast<int32_t>(
                         DSA_SPARSE_INDEX_CAPACITY)) {
            continue;
        }
        const int32_t slot =
            request_index[static_cast<uint32_t>(token)];
        if (slot >= 0 &&
            slot <
                static_cast<int32_t>(DSA_SPARSE_SLOT_COUNT)) {
            local_slots[local_entry] = slot;
            ProtectSlot(
                protected_bits,
                static_cast<uint32_t>(slot));
        }
    }
    if (total_misses == 0) {
#pragma unroll
        for (uint32_t local_entry = 0U;
             local_entry < query_chunk;
             ++local_entry) {
            const uint64_t offset =
                query_base + query_begin + local_entry;
            slot_out[offset] = local_slots[local_entry];
            miss_out[offset] = local_misses[local_entry];
        }
        return;
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Fused maintain phase. Scan occupied, non-protected slots in circular
    // cursor order and compact them by per-thread counts. The first M slots
    // replace the M free-list entries consumed above.
    const uint32_t scan_begin = tid * slot_chunk;
    const uint32_t scan_end = scan_begin + slot_chunk;
    const uint32_t cursor =
        static_cast<uint32_t>(scalars[kCursorScalar]);
    int32_t local_victim_count = 0;
    for (uint32_t position = scan_begin;
         position < scan_end;
         ++position) {
        uint32_t slot = cursor + position;
        if (slot >= DSA_SPARSE_SLOT_COUNT) {
            slot -= DSA_SPARSE_SLOT_COUNT;
        }
        if (!IsProtectedSlot(protected_bits, slot) &&
            request_slot_to_index[slot] !=
                DSA_SPARSE_NOT_FOUND) {
            ++local_victim_count;
        }
    }

    const int32_t victim_prefix =
        BlockExclusiveScan(local_victim_count, warp_totals);
    int32_t victim_rank = victim_prefix;
    if (victim_prefix < total_misses) {
        for (uint32_t position = scan_begin;
             position < scan_end;
             ++position) {
            uint32_t slot = cursor + position;
            if (slot >= DSA_SPARSE_SLOT_COUNT) {
                slot -= DSA_SPARSE_SLOT_COUNT;
            }
            if (IsProtectedSlot(protected_bits, slot)) {
                continue;
            }
            const int32_t old_token =
                request_slot_to_index[slot];
            if (old_token == DSA_SPARSE_NOT_FOUND) {
                continue;
            }
            if (victim_rank < total_misses) {
                request_slot_to_index[slot] =
                    DSA_SPARSE_NOT_FOUND;
                if (old_token >= 0 &&
                    old_token <
                        static_cast<int32_t>(
                            DSA_SPARSE_INDEX_CAPACITY) &&
                    request_index[
                        static_cast<uint32_t>(old_token)] ==
                        static_cast<int32_t>(slot)) {
                    request_index[
                        static_cast<uint32_t>(old_token)] =
                        DSA_SPARSE_NOT_FOUND;
                }
                request_free_slots[
                    static_cast<uint32_t>(
                        total_misses - 1 - victim_rank)] =
                    static_cast<int32_t>(slot);
                if (victim_rank == total_misses - 1) {
                    scalars[kLastVictimScalar] =
                        static_cast<int32_t>(slot);
                }
            }
            ++victim_rank;
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    if (tid == 0U) {
        const int32_t last_victim =
            scalars[kLastVictimScalar];
        request_free_head[1] =
            last_victim + 1 >=
                    static_cast<int32_t>(
                        DSA_SPARSE_SLOT_COUNT)
                ? 0
                : last_victim + 1;
        request_free_head[0] = 0;
    }

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

}  // namespace DsaSparseLookupUpdate

#endif  // DSA_SPARSE_LOOKUP_UPDATE_SIMT_H
