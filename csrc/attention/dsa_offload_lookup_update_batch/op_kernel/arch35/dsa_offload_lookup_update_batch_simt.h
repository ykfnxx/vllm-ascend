/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_LOOKUP_UPDATE_BATCH_SIMT_H
#define DSA_OFFLOAD_LOOKUP_UPDATE_BATCH_SIMT_H

#include "../dsa_offload_lookup_update_batch_common.h"

#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"

namespace DsaOffloadLookupUpdateBatch {

constexpr uint32_t kPoolEntryScalar = 0U;
constexpr uint32_t kFreeHeadScalar = 1U;
constexpr uint32_t kCursorScalar = 2U;
constexpr uint32_t kLastVictimScalar = 3U;
constexpr uint32_t kSafeAllocScalar = 4U;
constexpr uint32_t kQueryBeginScalar = 5U;
constexpr uint32_t kQueryEndScalar = 6U;

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
        tid & (DSA_OFFLOAD_BATCH_WARP_SIZE - 1U);
    const uint32_t warp = tid / DSA_OFFLOAD_BATCH_WARP_SIZE;

    int32_t inclusive = value;
    for (uint32_t delta = 1U;
         delta < DSA_OFFLOAD_BATCH_WARP_SIZE;
         delta <<= 1U) {
        const int32_t upstream = asc_shfl_up(inclusive, delta);
        if (lane >= delta) {
            inclusive += upstream;
        }
    }
    if (lane == DSA_OFFLOAD_BATCH_WARP_SIZE - 1U) {
        warp_totals[warp] = inclusive;
    }
    asc_syncthreads();

    if (warp == 0U) {
        int32_t warp_inclusive =
            lane < DSA_OFFLOAD_BATCH_WARP_COUNT
                ? warp_totals[lane]
                : 0;
        for (uint32_t delta = 1U;
             delta < DSA_OFFLOAD_BATCH_WARP_SIZE;
             delta <<= 1U) {
            const int32_t upstream =
                asc_shfl_up(warp_inclusive, delta);
            if (lane >= delta) {
                warp_inclusive += upstream;
            }
        }
        if (lane < DSA_OFFLOAD_BATCH_WARP_COUNT) {
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
    uint32_t query_end)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_OFFLOAD_BATCH_QUERY_WIDTH /
        DSA_OFFLOAD_BATCH_SIMT_THREADS;
    const uint32_t entry_begin = tid * query_chunk;
    for (uint32_t query_id = query_begin;
         query_id < query_end;
         ++query_id) {
        const uint64_t query_base =
            static_cast<uint64_t>(query_id) *
            DSA_OFFLOAD_BATCH_QUERY_WIDTH;
#pragma unroll
        for (uint32_t local_entry = 0U;
             local_entry < query_chunk;
             ++local_entry) {
            const uint64_t offset =
                query_base + entry_begin + local_entry;
            const int32_t token = query_index[offset];
            const bool active =
                lookup_mask[offset] != 0 && token >= 0 &&
                token < static_cast<int32_t>(
                            DSA_OFFLOAD_BATCH_INDEX_CAPACITY);
            slot_out[offset] =
                active ? DSA_OFFLOAD_BATCH_FALLBACK_SLOT
                       : DSA_OFFLOAD_BATCH_NOT_FOUND;
            miss_out[offset] = 0;
        }
    }
}

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
    uint32_t query_id)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_OFFLOAD_BATCH_QUERY_WIDTH /
        DSA_OFFLOAD_BATCH_SIMT_THREADS;
    constexpr uint32_t slot_chunk =
        DSA_OFFLOAD_BATCH_SLOT_COUNT /
        DSA_OFFLOAD_BATCH_SIMT_THREADS;
    const uint64_t query_base =
        static_cast<uint64_t>(query_id) *
        DSA_OFFLOAD_BATCH_QUERY_WIDTH;
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
                    token < static_cast<int32_t>(
                                DSA_OFFLOAD_BATCH_INDEX_CAPACITY)
                ? DSA_OFFLOAD_BATCH_FALLBACK_SLOT
                : DSA_OFFLOAD_BATCH_NOT_FOUND;
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
            token >= static_cast<int32_t>(
                         DSA_OFFLOAD_BATCH_INDEX_CAPACITY)) {
            continue;
        }
        const int32_t observed =
            request_index[static_cast<uint32_t>(token)];
        if (observed >= 0 &&
            observed < static_cast<int32_t>(
                           DSA_OFFLOAD_BATCH_SLOT_COUNT)) {
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
        warp_totals[DSA_OFFLOAD_BATCH_WARP_COUNT - 1U];

    const uint32_t scan_begin = tid * slot_chunk;
    const uint32_t scan_end = scan_begin + slot_chunk;
    const uint32_t cursor =
        static_cast<uint32_t>(scalars[kCursorScalar]);
    int32_t local_victim_count = 0;
    for (uint32_t position = scan_begin;
         position < scan_end;
         ++position) {
        uint32_t slot = cursor + position;
        if (slot >= DSA_OFFLOAD_BATCH_SLOT_COUNT) {
            slot -= DSA_OFFLOAD_BATCH_SLOT_COUNT;
        }
        if (!IsProtectedSlot(protected_bits, slot) &&
            request_slot_to_index[slot] !=
                DSA_OFFLOAD_BATCH_NOT_FOUND) {
            ++local_victim_count;
        }
    }
    const int32_t victim_prefix =
        BlockExclusiveScan(local_victim_count, warp_totals);
    const int32_t total_victims =
        warp_totals[DSA_OFFLOAD_BATCH_WARP_COUNT - 1U];
    if (tid == 0U) {
        int32_t safe_alloc = total_misses;
        if (safe_alloc > total_victims) {
            safe_alloc = total_victims;
        }
        if (safe_alloc > static_cast<int32_t>(
                             DSA_OFFLOAD_BATCH_FREE_SLOT_COUNT)) {
            safe_alloc = static_cast<int32_t>(
                DSA_OFFLOAD_BATCH_FREE_SLOT_COUNT);
        }
        scalars[kSafeAllocScalar] = safe_alloc;
        scalars[kLastVictimScalar] = DSA_OFFLOAD_BATCH_NOT_FOUND;
    }
    asc_syncthreads();
    const int32_t safe_alloc = scalars[kSafeAllocScalar];

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
        if (miss_rank >= safe_alloc) {
            continue;
        }
        const int32_t token = query_values[local_entry];
        const int32_t slot =
            request_free_slots[static_cast<uint32_t>(miss_rank)];
        if (slot < 0 ||
            slot >= static_cast<int32_t>(
                        DSA_OFFLOAD_BATCH_SLOT_COUNT)) {
            continue;
        }
        request_slot_to_index[static_cast<uint32_t>(slot)] = token;
        request_index[static_cast<uint32_t>(token)] = slot;
        local_slots[local_entry] = slot;
        local_misses[local_entry] = 1;
        ProtectSlot(protected_bits, static_cast<uint32_t>(slot));
    }
    asc_threadfence_block();
    asc_syncthreads();

    int32_t victim_rank = victim_prefix;
    if (victim_prefix < safe_alloc) {
        for (uint32_t position = scan_begin;
             position < scan_end;
             ++position) {
            uint32_t slot = cursor + position;
            if (slot >= DSA_OFFLOAD_BATCH_SLOT_COUNT) {
                slot -= DSA_OFFLOAD_BATCH_SLOT_COUNT;
            }
            if (IsProtectedSlot(protected_bits, slot)) {
                continue;
            }
            const int32_t old_token =
                request_slot_to_index[slot];
            if (old_token == DSA_OFFLOAD_BATCH_NOT_FOUND) {
                continue;
            }
            if (victim_rank < safe_alloc) {
                request_slot_to_index[slot] =
                    DSA_OFFLOAD_BATCH_NOT_FOUND;
                if (old_token >= 0 &&
                    old_token < static_cast<int32_t>(
                                    DSA_OFFLOAD_BATCH_INDEX_CAPACITY) &&
                    request_index[
                        static_cast<uint32_t>(old_token)] ==
                        static_cast<int32_t>(slot)) {
                    request_index[
                        static_cast<uint32_t>(old_token)] =
                        DSA_OFFLOAD_BATCH_NOT_FOUND;
                }
                request_free_slots[static_cast<uint32_t>(
                    safe_alloc - 1 - victim_rank)] =
                    static_cast<int32_t>(slot);
                if (victim_rank == safe_alloc - 1) {
                    scalars[kLastVictimScalar] =
                        static_cast<int32_t>(slot);
                }
            }
            ++victim_rank;
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    if (tid == 0U && safe_alloc > 0) {
        const int32_t last_victim = scalars[kLastVictimScalar];
        request_free_head[1] =
            last_victim + 1 >= static_cast<int32_t>(
                                   DSA_OFFLOAD_BATCH_SLOT_COUNT)
                ? 0
                : last_victim + 1;
        request_free_head[0] = 0;
        scalars[kCursorScalar] = request_free_head[1];
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
    bool reuse_scratch)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count = static_cast<uint32_t>(blockDim.x);
    __ubuf__ uint32_t* protected_bits = shared_scratch;
    __ubuf__ int32_t* warp_totals =
        reinterpret_cast<__ubuf__ int32_t*>(
            protected_bits + DSA_OFFLOAD_BATCH_PROTECTED_WORDS);
    __ubuf__ int32_t* scalars =
        warp_totals + DSA_OFFLOAD_BATCH_WARP_COUNT;

    for (uint32_t word = tid;
         word < DSA_OFFLOAD_BATCH_PROTECTED_WORDS;
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
        scalars[kLastVictimScalar] = DSA_OFFLOAD_BATCH_NOT_FOUND;
        scalars[kSafeAllocScalar] = 0;
        scalars[kQueryBeginScalar] = query_begin;
        scalars[kQueryEndScalar] = query_end;
        if (pool_entry >= 0 &&
            pool_entry < static_cast<int32_t>(pool_capacity)) {
            __gm__ int32_t* request_free_head =
                free_head + static_cast<uint64_t>(pool_entry) *
                                DSA_OFFLOAD_BATCH_FREE_HEAD_STRIDE;
            scalars[kFreeHeadScalar] = request_free_head[0];
            int32_t cursor = request_free_head[1];
            if (cursor < 0 ||
                cursor >= static_cast<int32_t>(
                              DSA_OFFLOAD_BATCH_SLOT_COUNT)) {
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
            static_cast<uint32_t>(query_end_value));
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
                    DSA_OFFLOAD_BATCH_INDEX_CAPACITY;
    __gm__ int32_t* request_slot_to_index =
        slot_to_index + static_cast<uint64_t>(pool_entry) *
                            DSA_OFFLOAD_BATCH_SLOT_COUNT;
    __gm__ int32_t* request_free_slots =
        free_slots + static_cast<uint64_t>(pool_entry) *
                         DSA_OFFLOAD_BATCH_FREE_SLOT_COUNT;
    __gm__ int32_t* request_free_head =
        free_head + static_cast<uint64_t>(pool_entry) *
                        DSA_OFFLOAD_BATCH_FREE_HEAD_STRIDE;

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
            query_id);
        asc_syncthreads();
    }
    if (reuse_scratch) {
        asc_syncthreads();
    }
}

__simt_vf__ __launch_bounds__(DSA_OFFLOAD_BATCH_SIMT_THREADS) inline void
DsaOffloadLookupUpdateBatchSimt(
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
    uint32_t query_num)
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
            req_id + request_stride < req_num);
    }
}

}  // namespace DsaOffloadLookupUpdateBatch

#endif  // DSA_OFFLOAD_LOOKUP_UPDATE_BATCH_SIMT_H
