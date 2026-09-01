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
constexpr uint32_t kCursorScalar = 1U;
constexpr uint32_t kLastVictimScalar = 2U;
constexpr uint32_t kSafeAllocScalar = 3U;
constexpr uint32_t kQueryBeginScalar = 4U;
constexpr uint32_t kQueryEndScalar = 5U;
constexpr uint32_t kVerifyStartScalar = 6U;

__simt_callee__ inline void ProtectSlot(
    __ubuf__ uint32_t* protected_bits,
    uint32_t slot)
{
    asc_atomic_or(
        protected_bits + (slot >> 5U),
        1U << (slot & 31U));
}

__simt_callee__ inline bool IsProtectedSlot(
    __ubuf__ const uint32_t* protected_bits,
    uint32_t slot)
{
    return (protected_bits[slot >> 5U] &
            (1U << (slot & 31U))) != 0U;
}

__simt_callee__ inline int32_t MapSlot(
    int32_t slot,
    uint32_t replaceable_base)
{
    return slot < static_cast<int32_t>(DSA_OFFLOAD_RESIDENT_SLOTS)
        ? slot
        : slot - static_cast<int32_t>(DSA_OFFLOAD_RESIDENT_SLOTS) +
              static_cast<int32_t>(replaceable_base);
}

__simt_callee__ inline int32_t BlockExclusiveScan(
    int32_t value,
    __ubuf__ int32_t* warp_totals)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t lane = tid & (DSA_OFFLOAD_WARP_SIZE - 1U);
    const uint32_t warp = tid / DSA_OFFLOAD_WARP_SIZE;
    int32_t inclusive = value;
    for (uint32_t delta = 1U;
         delta < DSA_OFFLOAD_WARP_SIZE;
         delta <<= 1U) {
        const int32_t upstream = asc_shfl_up(inclusive, delta);
        if (lane >= delta) {
            inclusive += upstream;
        }
    }
    if (lane == DSA_OFFLOAD_WARP_SIZE - 1U) {
        warp_totals[warp] = inclusive;
    }
    asc_syncthreads();
    if (warp == 0U) {
        int32_t warp_inclusive =
            lane < DSA_OFFLOAD_WARP_COUNT ? warp_totals[lane] : 0;
        for (uint32_t delta = 1U;
             delta < DSA_OFFLOAD_WARP_SIZE;
             delta <<= 1U) {
            const int32_t upstream =
                asc_shfl_up(warp_inclusive, delta);
            if (lane >= delta) {
                warp_inclusive += upstream;
            }
        }
        if (lane < DSA_OFFLOAD_WARP_COUNT) {
            warp_totals[lane] = warp_inclusive;
        }
    }
    asc_syncthreads();
    return (warp == 0U ? 0 : warp_totals[warp - 1U]) +
           inclusive - value;
}

__simt_callee__ inline void CopyPrefillQueries(
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* mapped_indices,
    __gm__ int32_t* miss_mask,
    uint32_t query_begin,
    uint32_t query_end)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_OFFLOAD_QUERY_WIDTH / DSA_OFFLOAD_SIMT_THREADS;
    const uint32_t entry_begin = tid * query_chunk;
    for (uint32_t query_id = query_begin;
         query_id < query_end;
         ++query_id) {
        const uint64_t query_base =
            static_cast<uint64_t>(query_id) *
            DSA_OFFLOAD_QUERY_WIDTH;
#pragma unroll
        for (uint32_t entry = 0U; entry < query_chunk; ++entry) {
            const uint64_t offset = query_base + entry_begin + entry;
            mapped_indices[offset] = semantic_topk[offset];
            miss_mask[offset] = 0;
        }
    }
}

__simt_callee__ inline void ProcessQuery(
    __gm__ int32_t* request_index,
    __gm__ int32_t* request_slot_to_index,
    __gm__ int32_t* request_free_slots,
    __gm__ int32_t* request_free_head,
    __gm__ int64_t* query_positions,
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* mapped_indices,
    __gm__ int32_t* miss_mask,
    __ubuf__ uint32_t* protected_bits,
    __ubuf__ int32_t* warp_totals,
    __ubuf__ int32_t* scalars,
    uint32_t query_id,
    uint32_t block_size,
    uint32_t replaceable_base,
    uint32_t tail_base,
    uint32_t fallback_slot,
    uint32_t staging_base,
    uint32_t decode_mode)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_OFFLOAD_QUERY_WIDTH / DSA_OFFLOAD_SIMT_THREADS;
    constexpr uint32_t slot_chunk =
        DSA_OFFLOAD_SLOT_COUNT / DSA_OFFLOAD_SIMT_THREADS;
    const uint64_t query_base =
        static_cast<uint64_t>(query_id) * DSA_OFFLOAD_QUERY_WIDTH;
    const uint32_t entry_begin = tid * query_chunk;
    const int64_t current_position = query_positions[query_id];
    const int64_t verify_start =
        static_cast<int64_t>(scalars[kVerifyStartScalar]);
    const int64_t tail_start =
        verify_start / static_cast<int64_t>(block_size) * block_size;

    int32_t tokens[query_chunk];
    int32_t local_mapped[query_chunk];
    int32_t miss_candidates[query_chunk];
    int32_t local_misses[query_chunk];
#pragma unroll
    for (uint32_t entry = 0U; entry < query_chunk; ++entry) {
        const int32_t token =
            semantic_topk[query_base + entry_begin + entry];
        tokens[entry] = token;
        local_mapped[entry] = DSA_OFFLOAD_NOT_FOUND;
        miss_candidates[entry] = 0;
        local_misses[entry] = 0;
        if (token < 0 ||
            token >= static_cast<int32_t>(DSA_OFFLOAD_INDEX_CAPACITY)) {
            continue;
        }
        if (static_cast<int64_t>(token) < tail_start) {
            local_mapped[entry] = static_cast<int32_t>(fallback_slot);
            const int32_t slot =
                request_index[static_cast<uint32_t>(token)];
            if (slot == DSA_OFFLOAD_NOT_FOUND) {
                miss_candidates[entry] = 1;
            } else {
                local_mapped[entry] = MapSlot(slot, replaceable_base);
                ProtectSlot(protected_bits, static_cast<uint32_t>(slot));
            }
        } else if (decode_mode == 0U) {
            if (static_cast<int64_t>(token) <= current_position) {
                local_mapped[entry] =
                    static_cast<int32_t>(tail_base) +
                    token - static_cast<int32_t>(tail_start);
            }
        } else if (static_cast<int64_t>(token) < verify_start) {
            local_mapped[entry] =
                static_cast<int32_t>(tail_base) +
                token - static_cast<int32_t>(tail_start);
        } else if (static_cast<int64_t>(token) <= current_position) {
            local_mapped[entry] =
                static_cast<int32_t>(staging_base) +
                token - static_cast<int32_t>(verify_start);
        }
    }

    int32_t local_miss_count = 0;
#pragma unroll
    for (uint32_t entry = 0U; entry < query_chunk; ++entry) {
        local_miss_count += miss_candidates[entry];
    }
    const int32_t miss_prefix =
        BlockExclusiveScan(local_miss_count, warp_totals);
    const int32_t total_misses =
        warp_totals[DSA_OFFLOAD_WARP_COUNT - 1U];
    if (total_misses == 0) {
#pragma unroll
        for (uint32_t entry = 0U; entry < query_chunk; ++entry) {
            const uint64_t offset = query_base + entry_begin + entry;
            mapped_indices[offset] = local_mapped[entry];
            miss_mask[offset] = 0;
        }
        return;
    }

    const uint32_t scan_begin = tid * slot_chunk;
    const uint32_t scan_end = scan_begin + slot_chunk;
    const uint32_t cursor = static_cast<uint32_t>(scalars[kCursorScalar]);
    int32_t local_victim_count = 0;
    for (uint32_t position = scan_begin;
         position < scan_end;
         ++position) {
        uint32_t slot = cursor + position;
        if (slot >= DSA_OFFLOAD_SLOT_COUNT) {
            slot -= DSA_OFFLOAD_SLOT_COUNT;
        }
        if (!IsProtectedSlot(protected_bits, slot) &&
            request_slot_to_index[slot] != DSA_OFFLOAD_NOT_FOUND) {
            ++local_victim_count;
        }
    }
    const int32_t victim_prefix =
        BlockExclusiveScan(local_victim_count, warp_totals);
    const int32_t total_victims =
        warp_totals[DSA_OFFLOAD_WARP_COUNT - 1U];
    if (tid == 0U) {
        int32_t safe_alloc = total_misses < total_victims
            ? total_misses
            : total_victims;
        if (safe_alloc > static_cast<int32_t>(DSA_OFFLOAD_FREE_SLOT_COUNT)) {
            safe_alloc = static_cast<int32_t>(DSA_OFFLOAD_FREE_SLOT_COUNT);
        }
        scalars[kSafeAllocScalar] = safe_alloc;
        scalars[kLastVictimScalar] = DSA_OFFLOAD_NOT_FOUND;
    }
    asc_syncthreads();
    const int32_t safe_alloc = scalars[kSafeAllocScalar];

    int32_t local_rank = 0;
#pragma unroll
    for (uint32_t entry = 0U; entry < query_chunk; ++entry) {
        if (miss_candidates[entry] == 0) {
            continue;
        }
        const int32_t miss_rank = miss_prefix + local_rank++;
        if (miss_rank >= safe_alloc) {
            continue;
        }
        const int32_t token = tokens[entry];
        const int32_t slot =
            request_free_slots[static_cast<uint32_t>(miss_rank)];
        request_slot_to_index[static_cast<uint32_t>(slot)] = token;
        request_index[static_cast<uint32_t>(token)] = slot;
        local_mapped[entry] = MapSlot(slot, replaceable_base);
        local_misses[entry] = 1;
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
            if (slot >= DSA_OFFLOAD_SLOT_COUNT) {
                slot -= DSA_OFFLOAD_SLOT_COUNT;
            }
            if (IsProtectedSlot(protected_bits, slot)) {
                continue;
            }
            const int32_t old_token = request_slot_to_index[slot];
            if (old_token == DSA_OFFLOAD_NOT_FOUND) {
                continue;
            }
            if (victim_rank < safe_alloc) {
                request_slot_to_index[slot] = DSA_OFFLOAD_NOT_FOUND;
                request_index[static_cast<uint32_t>(old_token)] =
                    DSA_OFFLOAD_NOT_FOUND;
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
        const int32_t next = scalars[kLastVictimScalar] + 1;
        request_free_head[0] = 0;
        request_free_head[1] =
            next == static_cast<int32_t>(DSA_OFFLOAD_SLOT_COUNT)
            ? 0
            : next;
        scalars[kCursorScalar] = request_free_head[1];
    }
    asc_syncthreads();

#pragma unroll
    for (uint32_t entry = 0U; entry < query_chunk; ++entry) {
        const uint64_t offset = query_base + entry_begin + entry;
        mapped_indices[offset] = local_mapped[entry];
        miss_mask[offset] = local_misses[entry];
    }
}

__simt_callee__ inline void ProcessRequest(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* request_rows,
    __gm__ int32_t* query_start_loc,
    __gm__ int64_t* query_positions,
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* mapped_indices,
    __gm__ int32_t* miss_mask,
    __ubuf__ uint32_t* shared_scratch,
    uint32_t request_id,
    uint32_t block_size,
    uint32_t replaceable_base,
    uint32_t tail_base,
    uint32_t fallback_slot,
    uint32_t staging_base,
    uint32_t decode_mode)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    __ubuf__ uint32_t* protected_bits = shared_scratch;
    __ubuf__ int32_t* warp_totals =
        reinterpret_cast<__ubuf__ int32_t*>(
            protected_bits + DSA_OFFLOAD_PROTECTED_WORDS);
    __ubuf__ int32_t* scalars =
        warp_totals + DSA_OFFLOAD_WARP_COUNT;
    for (uint32_t word = tid;
         word < DSA_OFFLOAD_PROTECTED_WORDS;
         word += static_cast<uint32_t>(blockDim.x)) {
        protected_bits[word] = 0U;
    }
    if (tid == 0U) {
        const int32_t pool_entry = request_rows[request_id];
        const int32_t query_begin = query_start_loc[request_id];
        scalars[kPoolEntryScalar] = pool_entry;
        scalars[kQueryBeginScalar] = query_begin;
        scalars[kQueryEndScalar] = query_start_loc[request_id + 1U];
        scalars[kCursorScalar] = pool_entry < 0
            ? 0
            : free_head[
                  static_cast<uint64_t>(pool_entry) *
                      DSA_OFFLOAD_FREE_HEAD_STRIDE +
                  1U];
        scalars[kVerifyStartScalar] =
            static_cast<int32_t>(query_positions[query_begin]);
    }
    asc_syncthreads();

    const uint32_t query_begin =
        static_cast<uint32_t>(scalars[kQueryBeginScalar]);
    const uint32_t query_end =
        static_cast<uint32_t>(scalars[kQueryEndScalar]);
    const int32_t pool_entry_value = scalars[kPoolEntryScalar];
    if (pool_entry_value < 0) {
        CopyPrefillQueries(
            semantic_topk,
            mapped_indices,
            miss_mask,
            query_begin,
            query_end);
        return;
    }

    const uint32_t pool_entry =
        static_cast<uint32_t>(pool_entry_value);
    __gm__ int32_t* request_index =
        index + static_cast<uint64_t>(pool_entry) *
                    DSA_OFFLOAD_INDEX_CAPACITY;
    __gm__ int32_t* request_slot_to_index =
        slot_to_index + static_cast<uint64_t>(pool_entry) *
                            DSA_OFFLOAD_SLOT_COUNT;
    __gm__ int32_t* request_free_slots =
        free_slots + static_cast<uint64_t>(pool_entry) *
                         DSA_OFFLOAD_FREE_SLOT_COUNT;
    __gm__ int32_t* request_free_head =
        free_head + static_cast<uint64_t>(pool_entry) *
                        DSA_OFFLOAD_FREE_HEAD_STRIDE;
    for (uint32_t query_id = query_begin;
         query_id < query_end;
         ++query_id) {
        ProcessQuery(
            request_index,
            request_slot_to_index,
            request_free_slots,
            request_free_head,
            query_positions,
            semantic_topk,
            mapped_indices,
            miss_mask,
            protected_bits,
            warp_totals,
            scalars,
            query_id,
            block_size,
            replaceable_base,
            tail_base,
            fallback_slot,
            staging_base,
            decode_mode);
        asc_syncthreads();
    }
}

__simt_vf__ __launch_bounds__(DSA_OFFLOAD_SIMT_THREADS) inline void
DsaOffloadLookupUpdateBatchSimt(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* request_rows,
    __gm__ int32_t* query_start_loc,
    __gm__ int64_t* query_positions,
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* mapped_indices,
    __gm__ int32_t* miss_mask,
    __ubuf__ uint32_t* shared_scratch,
    uint32_t req_num,
    uint32_t block_size,
    uint32_t replaceable_base,
    uint32_t tail_base,
    uint32_t fallback_slot,
    uint32_t staging_base,
    uint32_t decode_mode)
{
    const uint32_t request_stride =
        static_cast<uint32_t>(gridDim.x);
    for (uint32_t request_id =
             static_cast<uint32_t>(blockIdx.x);
         request_id < req_num;
         request_id += request_stride) {
        ProcessRequest(
            index,
            slot_to_index,
            free_slots,
            free_head,
            request_rows,
            query_start_loc,
            query_positions,
            semantic_topk,
            mapped_indices,
            miss_mask,
            shared_scratch,
            request_id,
            block_size,
            replaceable_base,
            tail_base,
            fallback_slot,
            staging_base,
            decode_mode);
        asc_syncthreads();
    }
}

}  // namespace DsaOffloadLookupUpdateBatch

#endif  // DSA_OFFLOAD_LOOKUP_UPDATE_BATCH_SIMT_H
