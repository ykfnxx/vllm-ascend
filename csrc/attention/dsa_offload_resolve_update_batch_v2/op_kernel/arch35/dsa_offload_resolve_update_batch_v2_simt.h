/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_RESOLVE_UPDATE_BATCH_V2_SIMT_H
#define DSA_OFFLOAD_RESOLVE_UPDATE_BATCH_V2_SIMT_H

#include "../dsa_offload_resolve_update_batch_v2_common.h"
#include "../../../dsa_offload_lookup_update_batch/op_kernel/arch35/dsa_offload_lookup_update_batch_simt.h"

namespace DsaOffloadResolveUpdateBatchV2 {

constexpr int32_t kInvalid = -1;
constexpr int32_t kFallbackRaw = 10 * 1024;
constexpr int32_t kResidentSlots = 8 * 1024;
constexpr int32_t kReplaceableBase = 8 * 1024;
constexpr int32_t kTailBase = 10 * 1024;
constexpr int32_t kFallbackSlot = kTailBase + 128;
constexpr int32_t kStagingBase = kFallbackSlot + 1;

constexpr uint32_t kPoolEntryScalar = 0U;
constexpr uint32_t kFreeHeadScalar = 1U;
constexpr uint32_t kCursorScalar = 2U;
constexpr uint32_t kLastVictimScalar = 3U;
constexpr uint32_t kSafeAllocScalar = 4U;
constexpr uint32_t kQueryBeginScalar = 5U;
constexpr uint32_t kQueryEndScalar = 6U;
constexpr uint32_t kVerifyStartScalar = 7U;
constexpr uint32_t kTailStartScalar = 8U;

enum EntryKind : uint32_t {
    kInvalidEntry = 0U,
    kHistoryEntry = 1U,
    kTailEntry = 2U,
    kStagingEntry = 3U,
};

__simt_callee__ inline EntryKind Classify(
    int32_t token,
    int32_t current_position,
    int32_t verify_start,
    int32_t tail_start,
    uint32_t decode_mode)
{
    if (token < 0 ||
        token >= static_cast<int32_t>(DSA_RESOLVE_V2_INDEX_CAPACITY)) {
        return kInvalidEntry;
    }
    if (token < tail_start) {
        return kHistoryEntry;
    }
    if (decode_mode == 0U) {
        return token <= current_position ? kTailEntry : kInvalidEntry;
    }
    if (token < verify_start) {
        return kTailEntry;
    }
    return token <= current_position ? kStagingEntry : kInvalidEntry;
}

__simt_callee__ inline int32_t ResolveLookupSlot(int32_t slot)
{
    return slot < kResidentSlots
               ? slot
               : slot - kResidentSlots + kReplaceableBase;
}

__simt_callee__ inline void InitializeRange(
    __gm__ int32_t* query_positions,
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* mapped_indices,
    __gm__ int32_t* gather_mask,
    uint32_t query_begin,
    uint32_t query_end,
    bool active_request,
    int32_t verify_start,
    int32_t tail_start,
    uint32_t decode_mode)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t chunk =
        DSA_RESOLVE_V2_QUERY_WIDTH / DSA_RESOLVE_V2_SIMT_THREADS;
    for (uint32_t query_id = query_begin; query_id < query_end; ++query_id) {
        const uint64_t base =
            static_cast<uint64_t>(query_id) * DSA_RESOLVE_V2_QUERY_WIDTH;
        const int32_t current_position = query_positions[query_id];
#pragma unroll
        for (uint32_t local = 0U; local < chunk; ++local) {
            const uint64_t offset = base + tid * chunk + local;
            const EntryKind kind = Classify(
                semantic_topk[offset],
                current_position,
                verify_start,
                tail_start,
                decode_mode);
            mapped_indices[offset] =
                active_request && kind == kHistoryEntry
                    ? kFallbackRaw
                    : kInvalid;
            gather_mask[offset] = 0;
        }
    }
}

__simt_callee__ inline void ResolveRange(
    __gm__ int32_t* query_positions,
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* mapped_indices,
    __gm__ int32_t* gather_mask,
    uint32_t query_begin,
    uint32_t query_end,
    int32_t verify_start,
    int32_t tail_start,
    uint32_t decode_mode)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t chunk =
        DSA_RESOLVE_V2_QUERY_WIDTH / DSA_RESOLVE_V2_SIMT_THREADS;
    for (uint32_t query_id = query_begin; query_id < query_end; ++query_id) {
        const uint64_t base =
            static_cast<uint64_t>(query_id) * DSA_RESOLVE_V2_QUERY_WIDTH;
        const int32_t current_position = query_positions[query_id];
#pragma unroll
        for (uint32_t local = 0U; local < chunk; ++local) {
            const uint64_t offset = base + tid * chunk + local;
            const int32_t token = semantic_topk[offset];
            const EntryKind kind = Classify(
                token,
                current_position,
                verify_start,
                tail_start,
                decode_mode);
            if (kind == kHistoryEntry) {
                const int32_t raw_slot = mapped_indices[offset];
                if (raw_slot == kFallbackRaw) {
                    mapped_indices[offset] = kFallbackSlot;
                    gather_mask[offset] = 0;
                } else if (raw_slot >= 0 &&
                           raw_slot <
                               static_cast<int32_t>(
                                   DSA_RESOLVE_V2_SLOT_COUNT)) {
                    mapped_indices[offset] = ResolveLookupSlot(raw_slot);
                } else {
                    mapped_indices[offset] = kInvalid;
                    gather_mask[offset] = 0;
                }
            } else if (kind == kTailEntry) {
                mapped_indices[offset] = kTailBase + token - tail_start;
                gather_mask[offset] = 0;
            } else if (kind == kStagingEntry) {
                mapped_indices[offset] =
                    kStagingBase + token - verify_start;
                gather_mask[offset] = 0;
            } else {
                mapped_indices[offset] = kInvalid;
                gather_mask[offset] = 0;
            }
        }
    }
}

__simt_callee__ inline void ProcessQuery(
    __gm__ int32_t* request_index,
    __gm__ int32_t* request_slot_to_index,
    __gm__ int32_t* request_free_slots,
    __gm__ int32_t* request_free_head,
    __gm__ int32_t* query_positions,
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* raw_slots,
    __gm__ int32_t* raw_misses,
    __ubuf__ uint32_t* protected_bits,
    __ubuf__ int32_t* warp_totals,
    __ubuf__ int32_t* scalars,
    uint32_t query_id,
    int32_t verify_start,
    int32_t tail_start,
    uint32_t decode_mode)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t query_chunk =
        DSA_RESOLVE_V2_QUERY_WIDTH / DSA_RESOLVE_V2_SIMT_THREADS;
    constexpr uint32_t slot_chunk =
        DSA_RESOLVE_V2_SLOT_COUNT / DSA_RESOLVE_V2_SIMT_THREADS;
    const uint64_t query_base =
        static_cast<uint64_t>(query_id) * DSA_RESOLVE_V2_QUERY_WIDTH;
    const uint32_t entry_begin = tid * query_chunk;
    const int32_t current_position = query_positions[query_id];

    int32_t query_values[query_chunk];
    int32_t history_entries[query_chunk];
    int32_t local_slots[query_chunk];
    int32_t local_miss_candidates[query_chunk];
    int32_t local_misses[query_chunk];
#pragma unroll
    for (uint32_t local = 0U; local < query_chunk; ++local) {
        const uint64_t offset = query_base + entry_begin + local;
        const int32_t token = semantic_topk[offset];
        const bool history =
            Classify(token,
                     current_position,
                     verify_start,
                     tail_start,
                     decode_mode) == kHistoryEntry;
        query_values[local] = token;
        history_entries[local] = history ? 1 : 0;
        local_slots[local] = history ? kFallbackRaw : kInvalid;
        local_miss_candidates[local] = 0;
        local_misses[local] = 0;
    }

#pragma unroll
    for (uint32_t local = 0U; local < query_chunk; ++local) {
        if (history_entries[local] == 0) {
            continue;
        }
        const int32_t token = query_values[local];
        const int32_t observed = request_index[token];
        if (observed >= 0 &&
            observed < static_cast<int32_t>(DSA_RESOLVE_V2_SLOT_COUNT)) {
            local_slots[local] = observed;
            DsaOffloadLookupUpdateBatch::ProtectSlot(
                protected_bits, static_cast<uint32_t>(observed));
        } else {
            local_miss_candidates[local] = 1;
        }
    }

    int32_t local_miss_count = 0;
#pragma unroll
    for (uint32_t local = 0U; local < query_chunk; ++local) {
        local_miss_count += local_miss_candidates[local];
    }
    const int32_t miss_prefix =
        DsaOffloadLookupUpdateBatch::BlockExclusiveScan(
            local_miss_count, warp_totals);
    const int32_t total_misses =
        warp_totals[DSA_RESOLVE_V2_WARP_COUNT - 1U];

    const uint32_t scan_begin = tid * slot_chunk;
    const uint32_t scan_end = scan_begin + slot_chunk;
    const uint32_t cursor = static_cast<uint32_t>(scalars[kCursorScalar]);
    int32_t local_victim_count = 0;
    for (uint32_t position = scan_begin; position < scan_end; ++position) {
        uint32_t slot = cursor + position;
        if (slot >= DSA_RESOLVE_V2_SLOT_COUNT) {
            slot -= DSA_RESOLVE_V2_SLOT_COUNT;
        }
        if (!DsaOffloadLookupUpdateBatch::IsProtectedSlot(
                protected_bits, slot) &&
            request_slot_to_index[slot] != kInvalid) {
            ++local_victim_count;
        }
    }
    const int32_t victim_prefix =
        DsaOffloadLookupUpdateBatch::BlockExclusiveScan(
            local_victim_count, warp_totals);
    const int32_t total_victims =
        warp_totals[DSA_RESOLVE_V2_WARP_COUNT - 1U];
    if (tid == 0U) {
        int32_t safe_alloc = total_misses;
        if (safe_alloc > total_victims) {
            safe_alloc = total_victims;
        }
        if (safe_alloc >
            static_cast<int32_t>(DSA_RESOLVE_V2_FREE_SLOT_COUNT)) {
            safe_alloc = DSA_RESOLVE_V2_FREE_SLOT_COUNT;
        }
        scalars[kSafeAllocScalar] = safe_alloc;
        scalars[kLastVictimScalar] = kInvalid;
    }
    asc_syncthreads();
    const int32_t safe_alloc = scalars[kSafeAllocScalar];

    int32_t local_rank = 0;
#pragma unroll
    for (uint32_t local = 0U; local < query_chunk; ++local) {
        if (local_miss_candidates[local] == 0) {
            continue;
        }
        const int32_t miss_rank = miss_prefix + local_rank++;
        if (miss_rank >= safe_alloc) {
            continue;
        }
        const int32_t token = query_values[local];
        const int32_t slot = request_free_slots[miss_rank];
        if (slot < 0 ||
            slot >= static_cast<int32_t>(DSA_RESOLVE_V2_SLOT_COUNT)) {
            continue;
        }
        request_slot_to_index[slot] = token;
        request_index[token] = slot;
        local_slots[local] = slot;
        local_misses[local] = 1;
        DsaOffloadLookupUpdateBatch::ProtectSlot(
            protected_bits, static_cast<uint32_t>(slot));
    }
    asc_threadfence_block();
    asc_syncthreads();

    int32_t victim_rank = victim_prefix;
    if (victim_prefix < safe_alloc) {
        for (uint32_t position = scan_begin; position < scan_end; ++position) {
            uint32_t slot = cursor + position;
            if (slot >= DSA_RESOLVE_V2_SLOT_COUNT) {
                slot -= DSA_RESOLVE_V2_SLOT_COUNT;
            }
            if (DsaOffloadLookupUpdateBatch::IsProtectedSlot(
                    protected_bits, slot)) {
                continue;
            }
            const int32_t old_token = request_slot_to_index[slot];
            if (old_token == kInvalid) {
                continue;
            }
            if (victim_rank < safe_alloc) {
                request_slot_to_index[slot] = kInvalid;
                if (request_index[old_token] == static_cast<int32_t>(slot)) {
                    request_index[old_token] = kInvalid;
                }
                request_free_slots[safe_alloc - 1 - victim_rank] = slot;
                if (victim_rank == safe_alloc - 1) {
                    scalars[kLastVictimScalar] = slot;
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
            last_victim + 1 >=
                    static_cast<int32_t>(DSA_RESOLVE_V2_SLOT_COUNT)
                ? 0
                : last_victim + 1;
        request_free_head[0] = 0;
        scalars[kCursorScalar] = request_free_head[1];
    }
    asc_syncthreads();

#pragma unroll
    for (uint32_t local = 0U; local < query_chunk; ++local) {
        const uint64_t offset = query_base + entry_begin + local;
        raw_slots[offset] = local_slots[local];
        raw_misses[offset] = local_misses[local];
    }
}

__simt_callee__ inline void ProcessRequest(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* request_rows,
    __gm__ int32_t* query_start_loc,
    __gm__ int32_t* query_positions,
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* mapped_indices,
    __gm__ int32_t* gather_mask,
    __ubuf__ uint32_t* scratch,
    uint32_t request_id,
    uint32_t pool_capacity,
    uint32_t query_num,
    uint32_t block_size,
    uint32_t decode_mode,
    bool reuse_scratch)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count = static_cast<uint32_t>(blockDim.x);
    __ubuf__ uint32_t* protected_bits = scratch;
    __ubuf__ int32_t* warp_totals =
        reinterpret_cast<__ubuf__ int32_t*>(
            protected_bits + DSA_RESOLVE_V2_PROTECTED_WORDS);
    __ubuf__ int32_t* scalars =
        warp_totals + DSA_RESOLVE_V2_WARP_COUNT;

    for (uint32_t word = tid;
         word < DSA_RESOLVE_V2_PROTECTED_WORDS;
         word += thread_count) {
        protected_bits[word] = 0U;
    }
    if (tid == 0U) {
        const int32_t row = request_rows[request_id];
        const int32_t begin = query_start_loc[request_id];
        const int32_t end = query_start_loc[request_id + 1U];
        scalars[kPoolEntryScalar] = row;
        scalars[kFreeHeadScalar] = 0;
        scalars[kCursorScalar] = 0;
        scalars[kLastVictimScalar] = kInvalid;
        scalars[kSafeAllocScalar] = 0;
        scalars[kQueryBeginScalar] = begin;
        scalars[kQueryEndScalar] = end;
        scalars[kVerifyStartScalar] = 0;
        scalars[kTailStartScalar] = 0;
        if (begin >= 0 && end > begin &&
            end <= static_cast<int32_t>(query_num)) {
            const int32_t verify_start = query_positions[begin];
            scalars[kVerifyStartScalar] = verify_start;
            scalars[kTailStartScalar] =
                verify_start / static_cast<int32_t>(block_size) *
                static_cast<int32_t>(block_size);
        }
        if (row >= 0 && row < static_cast<int32_t>(pool_capacity)) {
            __gm__ int32_t* row_head =
                free_head + static_cast<uint64_t>(row) *
                                DSA_RESOLVE_V2_FREE_HEAD_STRIDE;
            scalars[kFreeHeadScalar] = row_head[0];
            int32_t cursor = row_head[1];
            scalars[kCursorScalar] =
                cursor >= 0 &&
                        cursor <
                            static_cast<int32_t>(DSA_RESOLVE_V2_SLOT_COUNT)
                    ? cursor
                    : 0;
        }
    }
    asc_syncthreads();

    const int32_t row_value = scalars[kPoolEntryScalar];
    const int32_t begin_value = scalars[kQueryBeginScalar];
    const int32_t end_value = scalars[kQueryEndScalar];
    const bool range_valid =
        begin_value >= 0 && end_value >= begin_value &&
        end_value <= static_cast<int32_t>(query_num);
    const bool active_request =
        row_value >= 0 && row_value < static_cast<int32_t>(pool_capacity);
    if (range_valid) {
        InitializeRange(
            query_positions,
            semantic_topk,
            mapped_indices,
            gather_mask,
            static_cast<uint32_t>(begin_value),
            static_cast<uint32_t>(end_value),
            active_request,
            scalars[kVerifyStartScalar],
            scalars[kTailStartScalar],
            decode_mode);
        asc_syncthreads();
    }
    if (!range_valid || !active_request) {
        return;
    }

    if (scalars[kFreeHeadScalar] == 0) {
        const uint32_t row = static_cast<uint32_t>(row_value);
        __gm__ int32_t* request_index =
            index + static_cast<uint64_t>(row) *
                        DSA_RESOLVE_V2_INDEX_CAPACITY;
        __gm__ int32_t* request_slot_to_index =
            slot_to_index + static_cast<uint64_t>(row) *
                                DSA_RESOLVE_V2_SLOT_COUNT;
        __gm__ int32_t* request_free_slots =
            free_slots + static_cast<uint64_t>(row) *
                             DSA_RESOLVE_V2_FREE_SLOT_COUNT;
        __gm__ int32_t* request_free_head =
            free_head + static_cast<uint64_t>(row) *
                            DSA_RESOLVE_V2_FREE_HEAD_STRIDE;
        for (uint32_t query_id = static_cast<uint32_t>(begin_value);
             query_id < static_cast<uint32_t>(end_value);
             ++query_id) {
            ProcessQuery(
                request_index,
                request_slot_to_index,
                request_free_slots,
                request_free_head,
                query_positions,
                semantic_topk,
                mapped_indices,
                gather_mask,
                protected_bits,
                warp_totals,
                scalars,
                query_id,
                scalars[kVerifyStartScalar],
                scalars[kTailStartScalar],
                decode_mode);
            asc_syncthreads();
        }
    }

    ResolveRange(
        query_positions,
        semantic_topk,
        mapped_indices,
        gather_mask,
        static_cast<uint32_t>(begin_value),
        static_cast<uint32_t>(end_value),
        scalars[kVerifyStartScalar],
        scalars[kTailStartScalar],
        decode_mode);
    if (reuse_scratch) {
        asc_syncthreads();
    }
}

__simt_vf__ __launch_bounds__(DSA_RESOLVE_V2_SIMT_THREADS) inline void
ResolveUpdateSimt(
    __gm__ int32_t* index,
    __gm__ int32_t* slot_to_index,
    __gm__ int32_t* free_slots,
    __gm__ int32_t* free_head,
    __gm__ int32_t* request_rows,
    __gm__ int32_t* query_start_loc,
    __gm__ int32_t* query_positions,
    __gm__ int32_t* semantic_topk,
    __gm__ int32_t* mapped_indices,
    __gm__ int32_t* gather_mask,
    __ubuf__ uint32_t* scratch,
    uint32_t req_num,
    uint32_t pool_capacity,
    uint32_t query_num,
    uint32_t block_size,
    uint32_t decode_mode)
{
    const uint32_t stride = static_cast<uint32_t>(gridDim.x);
    for (uint32_t request_id = static_cast<uint32_t>(blockIdx.x);
         request_id < req_num;
         request_id += stride) {
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
            gather_mask,
            scratch,
            request_id,
            pool_capacity,
            query_num,
            block_size,
            decode_mode,
            request_id + stride < req_num);
    }
    if (blockIdx.x == 0) {
        const int32_t padding_begin = query_start_loc[req_num];
        if (padding_begin >= 0 &&
            padding_begin < static_cast<int32_t>(query_num)) {
            const uint64_t begin =
                static_cast<uint64_t>(padding_begin) *
                DSA_RESOLVE_V2_QUERY_WIDTH;
            const uint64_t end =
                static_cast<uint64_t>(query_num) *
                DSA_RESOLVE_V2_QUERY_WIDTH;
            for (uint64_t offset =
                     begin + static_cast<uint32_t>(threadIdx.x);
                 offset < end;
                 offset += static_cast<uint32_t>(blockDim.x)) {
                mapped_indices[offset] = kInvalid;
                gather_mask[offset] = 0;
            }
        }
    }
}

}  // namespace DsaOffloadResolveUpdateBatchV2

#endif
