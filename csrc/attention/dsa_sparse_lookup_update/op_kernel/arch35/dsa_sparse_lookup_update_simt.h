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

namespace DsaSparseLookupUpdate {

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
    __gm__ int32_t* workspace,
    uint32_t req_id,
    uint32_t pool_capacity,
    uint32_t workspace_stride)
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
    const uint32_t query_end = query_begin + query_chunk;

    __gm__ int32_t* request_workspace =
        workspace +
        static_cast<uint64_t>(req_id) * workspace_stride;
    __gm__ int32_t* protected_slots = request_workspace;
    __gm__ int32_t* thread_counts =
        protected_slots + DSA_SPARSE_SLOT_COUNT;
    __gm__ int32_t* scalars =
        thread_counts + DSA_SPARSE_SIMT_THREADS;

    for (uint32_t entry = query_begin;
         entry < query_end;
         ++entry) {
        slot_out[query_base + entry] = DSA_SPARSE_NOT_FOUND;
        miss_out[query_base + entry] = 0;
    }
    for (uint32_t slot = tid;
         slot < DSA_SPARSE_SLOT_COUNT;
         slot += thread_count) {
        protected_slots[slot] = 0;
    }
    if (tid < DSA_SPARSE_WORKSPACE_SCALARS) {
        scalars[tid] = 0;
    }
    asc_threadfence_block();
    asc_syncthreads();

    const int32_t pool_entry_value = req_pool_entries[req_id];
    if (pool_entry_value < 0 ||
        pool_entry_value >= static_cast<int32_t>(pool_capacity)) {
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

    if (tid == 0U) {
        scalars[0] = request_free_head[0];
        int32_t cursor = request_free_head[1];
        if (cursor < 0 ||
            cursor >= static_cast<int32_t>(
                          DSA_SPARSE_SLOT_COUNT)) {
            cursor = 0;
        }
        scalars[1] = cursor;
    }
    asc_threadfence_block();
    asc_syncthreads();
    if (scalars[0] != 0) {
        // This fused operator owns the complete lookup/maintain transaction.
        // A non-zero entry head means the row was left mid-transaction by a
        // different producer; fail closed instead of consuming a partial list.
        return;
    }

    // First pass: return resident hits and place a deterministic negative
    // claim in index[token] for each missing token.  A smaller flat query
    // position always replaces a later claim, so duplicate misses have one
    // stable owner independent of SIMT scheduling.
    for (uint32_t entry = query_begin;
         entry < query_end;
         ++entry) {
        const uint64_t offset = query_base + entry;
        if (lookup_mask[offset] == 0) {
            continue;
        }
        const int32_t token = query_index[offset];
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
            slot_out[offset] = observed;
            protected_slots[
                static_cast<uint32_t>(observed)] = 1;
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

    // Count canonical misses per contiguous query chunk.  Prefixing the
    // thread counts preserves input order for free-list allocation.
    int32_t local_miss_count = 0;
    for (uint32_t entry = query_begin;
         entry < query_end;
         ++entry) {
        const uint64_t offset = query_base + entry;
        if (lookup_mask[offset] == 0) {
            continue;
        }
        const int32_t token = query_index[offset];
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
    thread_counts[tid] = local_miss_count;
    asc_threadfence_block();
    asc_syncthreads();

    int32_t miss_prefix = 0;
    for (uint32_t other = 0; other < tid; ++other) {
        miss_prefix += thread_counts[other];
    }
    if (tid == thread_count - 1U) {
        scalars[2] = miss_prefix + local_miss_count;
    }
    asc_threadfence_block();
    asc_syncthreads();

    const int32_t head_start = scalars[0];
    const int32_t total_misses = scalars[2];
    int32_t local_rank = 0;
    for (uint32_t entry = query_begin;
         entry < query_end;
         ++entry) {
        const uint64_t offset = query_base + entry;
        if (lookup_mask[offset] == 0) {
            continue;
        }
        const int32_t token = query_index[offset];
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
        slot_out[offset] = slot;
        miss_out[offset] = 1;
        protected_slots[
            static_cast<uint32_t>(slot)] = 1;
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Duplicate followers observe the canonical occurrence's installed slot.
    // Invalid and masked entries retain the initialized (-1, 0) result.
    for (uint32_t entry = query_begin;
         entry < query_end;
         ++entry) {
        const uint64_t offset = query_base + entry;
        if (slot_out[offset] != DSA_SPARSE_NOT_FOUND ||
            lookup_mask[offset] == 0) {
            continue;
        }
        const int32_t token = query_index[offset];
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
            slot_out[offset] = slot;
            protected_slots[
                static_cast<uint32_t>(slot)] = 1;
        }
    }
    asc_threadfence_block();
    asc_syncthreads();
    if (total_misses == 0) {
        return;
    }

    if (tid == 0U) {
        request_free_head[0] = head_start + total_misses;
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Fused maintain phase.  Scan occupied, non-protected slots in circular
    // cursor order and compact them by per-thread counts.  The first M slots
    // replace the M free-list entries consumed above.
    const uint32_t scan_begin = tid * slot_chunk;
    const uint32_t scan_end = scan_begin + slot_chunk;
    const uint32_t cursor =
        static_cast<uint32_t>(scalars[1]);
    int32_t local_victim_count = 0;
    for (uint32_t position = scan_begin;
         position < scan_end;
         ++position) {
        uint32_t slot = cursor + position;
        if (slot >= DSA_SPARSE_SLOT_COUNT) {
            slot -= DSA_SPARSE_SLOT_COUNT;
        }
        if (protected_slots[slot] == 0 &&
            request_slot_to_index[slot] !=
                DSA_SPARSE_NOT_FOUND) {
            ++local_victim_count;
        }
    }
    thread_counts[tid] = local_victim_count;
    asc_threadfence_block();
    asc_syncthreads();

    int32_t victim_prefix = 0;
    for (uint32_t other = 0; other < tid; ++other) {
        victim_prefix += thread_counts[other];
    }
    int32_t victim_rank = victim_prefix;
    for (uint32_t position = scan_begin;
         position < scan_end;
         ++position) {
        uint32_t slot = cursor + position;
        if (slot >= DSA_SPARSE_SLOT_COUNT) {
            slot -= DSA_SPARSE_SLOT_COUNT;
        }
        if (protected_slots[slot] != 0) {
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
                scalars[3] = static_cast<int32_t>(slot);
            }
        }
        ++victim_rank;
    }
    asc_threadfence_block();
    asc_syncthreads();

    if (tid == 0U) {
        if (total_misses > 0) {
            const int32_t last_victim = scalars[3];
            request_free_head[1] =
                last_victim + 1 >=
                        static_cast<int32_t>(
                            DSA_SPARSE_SLOT_COUNT)
                    ? 0
                    : last_victim + 1;
        }
        request_free_head[0] = 0;
    }
    asc_threadfence_block();
    asc_syncthreads();
}

}  // namespace DsaSparseLookupUpdate

#endif  // DSA_SPARSE_LOOKUP_UPDATE_SIMT_H
