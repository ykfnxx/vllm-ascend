/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_TURBO_RESOLVE_UPDATE_BATCH_V2_SIMT_H
#define DSA_SPARSE_TURBO_RESOLVE_UPDATE_BATCH_V2_SIMT_H

#include "../dsa_sparse_turbo_resolve_update_batch_v2_common.h"
#include "../../../dsa_offload_resolve_update_batch_v2/op_kernel/arch35/dsa_offload_resolve_update_batch_v2_simt.h"
#include "../../../dsa_sparse_turbo_lookup_update_batch/op_kernel/arch35/dsa_sparse_turbo_lookup_update_batch_simt.h"

namespace DsaSparseTurboResolveUpdateBatchV2 {

constexpr uint32_t kPoolEntryScalar = 0U;
constexpr uint32_t kFreeHeadScalar = 1U;
constexpr uint32_t kCursorScalar = 2U;
constexpr uint32_t kLastVictimScalar = 3U;
constexpr uint32_t kEffectiveScalar = 4U;
constexpr uint32_t kQueryBeginScalar = 5U;
constexpr uint32_t kQueryEndScalar = 6U;
constexpr uint32_t kAllocBaseScalar = 7U;
constexpr uint32_t kVerifyStartScalar = 8U;
constexpr uint32_t kTailStartScalar = 9U;

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
    __ubuf__ int32_t* alloc_records,
    uint32_t query_id,
    int32_t verify_start,
    int32_t tail_start,
    uint32_t decode_mode)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    constexpr uint32_t chunk =
        DSA_TURBO_RESOLVE_V2_QUERY_WIDTH /
        DSA_TURBO_RESOLVE_V2_SIMT_THREADS;
    const uint64_t query_base =
        static_cast<uint64_t>(query_id) *
        DSA_TURBO_RESOLVE_V2_QUERY_WIDTH;
    const uint32_t entry_begin = tid * chunk;
    const int32_t current_position = query_positions[query_id];

    int32_t query_values[chunk];
    int32_t history_entries[chunk];
    int32_t local_slots[chunk];
    int32_t local_miss_candidates[chunk];
    int32_t local_misses[chunk];
#pragma unroll
    for (uint32_t local = 0U; local < chunk; ++local) {
        const uint64_t offset = query_base + entry_begin + local;
        const int32_t token = semantic_topk[offset];
        const bool history =
            DsaOffloadResolveUpdateBatchV2::Classify(
                token,
                current_position,
                verify_start,
                tail_start,
                decode_mode) ==
            DsaOffloadResolveUpdateBatchV2::kHistoryEntry;
        query_values[local] = token;
        history_entries[local] = history ? 1 : 0;
        local_slots[local] =
            history ? DsaOffloadResolveUpdateBatchV2::kFallbackRaw
                    : DsaOffloadResolveUpdateBatchV2::kInvalid;
        local_miss_candidates[local] = 0;
        local_misses[local] = 0;
    }

#pragma unroll
    for (uint32_t local = 0U; local < chunk; ++local) {
        if (history_entries[local] == 0) {
            continue;
        }
        const int32_t token = query_values[local];
        const int32_t observed = request_index[token];
        if (observed >= 0 &&
            observed <
                static_cast<int32_t>(DSA_TURBO_RESOLVE_V2_SLOT_COUNT)) {
            local_slots[local] = observed;
            DsaSparseTurboLookupUpdateBatch::ProtectSlot(
                protected_bits, static_cast<uint32_t>(observed));
        } else {
            local_miss_candidates[local] = 1;
        }
    }

    int32_t local_miss_count = 0;
#pragma unroll
    for (uint32_t local = 0U; local < chunk; ++local) {
        local_miss_count += local_miss_candidates[local];
    }
    const int32_t miss_prefix =
        DsaSparseTurboLookupUpdateBatch::BlockExclusiveScan(
            local_miss_count, warp_totals);
    const int32_t total_misses =
        warp_totals[DSA_TURBO_RESOLVE_V2_WARP_COUNT - 1U];

    const int32_t base = scalars[kAllocBaseScalar];
    if (base > 0 &&
        base + total_misses >
            static_cast<int32_t>(
                DSA_TURBO_RESOLVE_V2_FREE_SLOT_COUNT)) {
        DsaSparseTurboLookupUpdateBatch::MaintainRequest(
            request_index,
            request_slot_to_index,
            request_free_slots,
            request_free_head,
            semantic_topk,
            raw_slots,
            raw_misses,
            protected_bits,
            warp_totals,
            scalars,
            alloc_records,
            DSA_TURBO_RESOLVE_V2_INDEX_CAPACITY);
        if (tid == 0U) {
            scalars[kAllocBaseScalar] = 0;
        }
        asc_syncthreads();
    }
    const int32_t alloc_base = scalars[kAllocBaseScalar];
    int32_t budget = total_misses;
    const int32_t remaining =
        DSA_TURBO_RESOLVE_V2_FREE_SLOT_COUNT - alloc_base;
    if (budget > remaining) {
        budget = remaining;
    }

    int32_t local_rank = 0;
#pragma unroll
    for (uint32_t local = 0U; local < chunk; ++local) {
        if (local_miss_candidates[local] == 0) {
            continue;
        }
        const int32_t miss_rank = miss_prefix + local_rank++;
        if (miss_rank >= budget) {
            continue;
        }
        const int32_t token = query_values[local];
        const int32_t slot = request_free_slots[alloc_base + miss_rank];
        if (slot < 0 ||
            slot >=
                static_cast<int32_t>(DSA_TURBO_RESOLVE_V2_SLOT_COUNT)) {
            continue;
        }
        request_slot_to_index[slot] = token;
        request_index[token] = slot;
        local_slots[local] = slot;
        local_misses[local] = 1;
        DsaSparseTurboLookupUpdateBatch::ProtectSlot(
            protected_bits, static_cast<uint32_t>(slot));
        const uint32_t record =
            static_cast<uint32_t>(alloc_base + miss_rank);
        alloc_records[2U * record] = static_cast<int32_t>(
            query_base + entry_begin + local);
        alloc_records[2U * record + 1U] = slot;
    }
    asc_threadfence_block();
    asc_syncthreads();
    if (tid == 0U) {
        scalars[kAllocBaseScalar] = alloc_base + budget;
        request_free_head[0] = alloc_base + budget;
    }
    asc_syncthreads();

#pragma unroll
    for (uint32_t local = 0U; local < chunk; ++local) {
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
            protected_bits + DSA_TURBO_RESOLVE_V2_PROTECTED_WORDS);
    __ubuf__ int32_t* scalars =
        warp_totals + DSA_TURBO_RESOLVE_V2_WARP_COUNT;
    __ubuf__ int32_t* alloc_records =
        scalars + DSA_TURBO_RESOLVE_V2_SHARED_SCALARS;

    for (uint32_t word = tid;
         word < DSA_TURBO_RESOLVE_V2_PROTECTED_WORDS;
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
        scalars[kLastVictimScalar] =
            DsaOffloadResolveUpdateBatchV2::kInvalid;
        scalars[kEffectiveScalar] = 0;
        scalars[kQueryBeginScalar] = begin;
        scalars[kQueryEndScalar] = end;
        scalars[kAllocBaseScalar] = 0;
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
                                DSA_TURBO_RESOLVE_V2_FREE_HEAD_STRIDE;
            scalars[kFreeHeadScalar] = row_head[0];
            const int32_t cursor = row_head[1];
            scalars[kCursorScalar] =
                cursor >= 0 &&
                        cursor < static_cast<int32_t>(
                                     DSA_TURBO_RESOLVE_V2_SLOT_COUNT)
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
        DsaOffloadResolveUpdateBatchV2::InitializeRange(
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
                        DSA_TURBO_RESOLVE_V2_INDEX_CAPACITY;
        __gm__ int32_t* request_slot_to_index =
            slot_to_index + static_cast<uint64_t>(row) *
                                DSA_TURBO_RESOLVE_V2_SLOT_COUNT;
        __gm__ int32_t* request_free_slots =
            free_slots + static_cast<uint64_t>(row) *
                             DSA_TURBO_RESOLVE_V2_FREE_SLOT_COUNT;
        __gm__ int32_t* request_free_head =
            free_head + static_cast<uint64_t>(row) *
                            DSA_TURBO_RESOLVE_V2_FREE_HEAD_STRIDE;

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
                alloc_records,
                query_id,
                scalars[kVerifyStartScalar],
                scalars[kTailStartScalar],
                decode_mode);
            asc_syncthreads();
        }
        DsaSparseTurboLookupUpdateBatch::MaintainRequest(
            request_index,
            request_slot_to_index,
            request_free_slots,
            request_free_head,
            semantic_topk,
            mapped_indices,
            gather_mask,
            protected_bits,
            warp_totals,
            scalars,
            alloc_records,
            DSA_TURBO_RESOLVE_V2_INDEX_CAPACITY);
    }

    DsaOffloadResolveUpdateBatchV2::ResolveRange(
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

__simt_vf__ __launch_bounds__(DSA_TURBO_RESOLVE_V2_SIMT_THREADS) inline void
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
                DSA_TURBO_RESOLVE_V2_QUERY_WIDTH;
            const uint64_t end =
                static_cast<uint64_t>(query_num) *
                DSA_TURBO_RESOLVE_V2_QUERY_WIDTH;
            for (uint64_t offset =
                     begin + static_cast<uint32_t>(threadIdx.x);
                 offset < end;
                 offset += static_cast<uint32_t>(blockDim.x)) {
                mapped_indices[offset] =
                    DsaOffloadResolveUpdateBatchV2::kInvalid;
                gather_mask[offset] = 0;
            }
        }
    }
}

}  // namespace DsaSparseTurboResolveUpdateBatchV2

#endif
