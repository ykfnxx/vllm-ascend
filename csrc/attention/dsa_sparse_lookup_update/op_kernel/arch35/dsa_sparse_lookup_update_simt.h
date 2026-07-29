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
    __gm__ int32_t* token_to_hot,
    __gm__ int32_t* hot_to_token,
    __gm__ int32_t* lru_slots,
    __gm__ int32_t* query_positions,
    __gm__ int32_t* query_to_req_idx,
    __gm__ int32_t* query_to_lane,
    __gm__ uint8_t* query_valid_mask,
    __gm__ int32_t* valid_topk_counts,
    __gm__ int32_t* seq_lens,
    __gm__ int32_t* topk_positions,
    __gm__ int32_t* resolved_hot_indices,
    __gm__ uint8_t* miss_mask,
    __gm__ int32_t* op_workspace,
    uint32_t request_index,
    uint32_t token_position_capacity,
    uint32_t evictable_slot_count,
    uint32_t query_capacity,
    uint32_t query_lane_capacity,
    uint32_t topk_count,
    uint32_t workspace_stride)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t thread_count =
        static_cast<uint32_t>(blockDim.x);

    __gm__ int32_t* request_workspace =
        op_workspace +
        static_cast<uint64_t>(request_index) * workspace_stride;
    __gm__ int32_t* hit_flags = request_workspace;
    __gm__ int32_t* hit_slots =
        hit_flags + evictable_slot_count;
    __gm__ int32_t* evictable_slots =
        hit_slots + evictable_slot_count;
    __gm__ int32_t* thread_hit_counts =
        evictable_slots + evictable_slot_count;
    __gm__ int32_t* thread_evict_counts =
        thread_hit_counts + DSA_SPARSE_SIMT_THREADS;
    __gm__ int32_t* thread_miss_counts =
        thread_evict_counts + DSA_SPARSE_SIMT_THREADS;
    __gm__ int32_t* counters =
        thread_miss_counts + DSA_SPARSE_SIMT_THREADS;

    // Build the fixed request-index/lane -> flat query inverse in the four scalar
    // workspace cells. This accepts a reordered flat query view without a
    // separate pack operator. The graph-input contract guarantees at most one
    // query for each (request index, lane).
    if (tid < DSA_SPARSE_MAX_QUERY_LANES) {
        counters[tid] = DSA_SPARSE_NOT_FOUND;
    }
    asc_threadfence_block();
    asc_syncthreads();
    if (tid == 0U) {
        for (uint32_t query = 0; query < query_capacity; ++query) {
            if (query_to_req_idx[query] !=
                static_cast<int32_t>(request_index)) {
                continue;
            }
            const int32_t lane = query_to_lane[query];
            if (lane >= 0 &&
                lane < static_cast<int32_t>(query_lane_capacity)) {
                counters[static_cast<uint32_t>(lane)] =
                    static_cast<int32_t>(query);
            }
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    int32_t request_queries[DSA_SPARSE_MAX_QUERY_LANES];
    for (uint32_t lane = 0;
         lane < DSA_SPARSE_MAX_QUERY_LANES;
         ++lane) {
        request_queries[lane] = counters[lane];
    }
    int32_t ordered_lanes[DSA_SPARSE_MAX_QUERY_LANES];
    for (uint32_t lane = 0;
         lane < DSA_SPARSE_MAX_QUERY_LANES;
         ++lane) {
        ordered_lanes[lane] = static_cast<int32_t>(lane);
    }
    for (uint32_t left = 0;
         left < query_lane_capacity;
         ++left) {
        for (uint32_t right = left + 1U;
             right < query_lane_capacity;
             ++right) {
            const int32_t left_query =
                request_queries[
                    static_cast<uint32_t>(ordered_lanes[left])];
            const int32_t right_query =
                request_queries[
                    static_cast<uint32_t>(ordered_lanes[right])];
            if ((left_query < 0 && right_query >= 0) ||
                (right_query >= 0 && left_query >= 0 &&
                 right_query < left_query)) {
                const int32_t tmp = ordered_lanes[left];
                ordered_lanes[left] = ordered_lanes[right];
                ordered_lanes[right] = tmp;
            }
        }
    }

    const uint32_t request_entry_count =
        query_lane_capacity * topk_count;
    // Outputs belong to the per-call plan, not to persistent request state.
    // Initialize every mapped query before the inactive-request fast path so
    // padded queries never retain values from an earlier replay.
    for (uint32_t entry = tid;
         entry < request_entry_count;
         entry += thread_count) {
        const uint32_t lane = entry / topk_count;
        const uint32_t rank = entry - lane * topk_count;
        const int32_t query = request_queries[lane];
        if (query < 0) {
            continue;
        }
        const uint64_t output_offset =
            static_cast<uint64_t>(query) * topk_count + rank;
        resolved_hot_indices[output_offset] =
            DSA_SPARSE_NOT_FOUND;
        miss_mask[output_offset] = 0U;
    }
    asc_threadfence_block();
    asc_syncthreads();

    bool has_active_query = false;
    for (uint32_t lane = 0;
         lane < query_lane_capacity;
         ++lane) {
        const int32_t query = request_queries[lane];
        if (query >= 0 && query_valid_mask[query] != 0U) {
            has_active_query = true;
            break;
        }
    }
    if (!has_active_query) {
        return;
    }

    __gm__ int32_t* request_token_to_hot =
        token_to_hot +
        static_cast<uint64_t>(request_index) *
            token_position_capacity;
    __gm__ int32_t* request_hot_to_token =
        hot_to_token +
        static_cast<uint64_t>(request_index) *
            evictable_slot_count;
    __gm__ int32_t* request_lru =
        lru_slots +
        static_cast<uint64_t>(request_index) *
            evictable_slot_count;

    const int32_t request_seq_len = seq_lens[request_index];

    // Current Main-KV positions live in reserved slots for this replay. Remove
    // an evictable mapping left by an earlier replay before ordinary lookup.
    for (uint32_t lane = tid;
         lane < query_lane_capacity;
         lane += thread_count) {
        const int32_t query = request_queries[lane];
        if (query < 0 || query_valid_mask[query] == 0U) {
            continue;
        }
        const int32_t token = query_positions[query];
        if (token < 0 || token >= request_seq_len ||
            token >= static_cast<int32_t>(
                         token_position_capacity)) {
            continue;
        }
        const int32_t stale_slot =
            request_token_to_hot[static_cast<uint32_t>(token)];
        if (stale_slot >= 0 &&
            stale_slot <
                static_cast<int32_t>(evictable_slot_count)) {
            request_token_to_hot[static_cast<uint32_t>(token)] =
                DSA_SPARSE_NOT_FOUND;
            if (request_hot_to_token[
                    static_cast<uint32_t>(stale_slot)] == token) {
                request_hot_to_token[
                    static_cast<uint32_t>(stale_slot)] =
                    DSA_SPARSE_NOT_FOUND;
            }
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    for (uint32_t slot = tid;
         slot < evictable_slot_count;
         slot += thread_count) {
        hit_flags[slot] = 0;
    }
    if (tid < 4U) {
        counters[tid] = 0;
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Resolve every valid [lane, rank] entry. A missing token temporarily
    // stores -(minimum_global_flat_entry + 2) in token_to_hot. The CAS loop
    // monotonically raises the negative value, which deterministically chooses
    // the smallest q*K+k occurrence independently of SIMT scheduling.
    for (uint32_t entry = tid;
         entry < request_entry_count;
         entry += thread_count) {
        const uint32_t lane = entry / topk_count;
        const uint32_t rank = entry - lane * topk_count;
        const int32_t query = request_queries[lane];
        const int32_t valid_topk_count =
            query < 0 ? 0 : valid_topk_counts[query];
        if (query < 0 || query_valid_mask[query] == 0U ||
            valid_topk_count <= 0 ||
            rank >= static_cast<uint32_t>(
                        valid_topk_count)) {
            continue;
        }

        const uint64_t output_offset =
            static_cast<uint64_t>(query) * topk_count + rank;
        const int32_t token = topk_positions[output_offset];
        if (token < 0 || token >= request_seq_len ||
            token >= static_cast<int32_t>(
                         token_position_capacity)) {
            continue;
        }

        int32_t newest_query = DSA_SPARSE_NOT_FOUND;
        int32_t newest_lane = DSA_SPARSE_NOT_FOUND;
        for (uint32_t candidate_lane = 0;
             candidate_lane < query_lane_capacity;
             ++candidate_lane) {
            const int32_t candidate_query =
                request_queries[candidate_lane];
            if (candidate_query < 0 ||
                query_valid_mask[candidate_query] == 0U) {
                continue;
            }
            if (query_positions[candidate_query] == token) {
                if (newest_query == DSA_SPARSE_NOT_FOUND ||
                    candidate_query < newest_query) {
                    newest_query = candidate_query;
                    newest_lane =
                        static_cast<int32_t>(candidate_lane);
                }
            }
        }
        if (newest_lane >= 0) {
            resolved_hot_indices[output_offset] =
                static_cast<int32_t>(evictable_slot_count) +
                newest_lane;
            continue;
        }

        __gm__ int32_t* token_slot =
            request_token_to_hot + static_cast<uint32_t>(token);
        int32_t observed = *token_slot;
        if (observed >= 0 &&
            observed <
                static_cast<int32_t>(evictable_slot_count)) {
            resolved_hot_indices[output_offset] = observed;
            hit_flags[static_cast<uint32_t>(observed)] = 1;
            continue;
        }

        const int32_t global_flat =
            static_cast<int32_t>(output_offset);
        const int32_t desired_claim =
            DSA_SPARSE_CLAIM_BASE - global_flat;
        while (observed == DSA_SPARSE_NOT_FOUND ||
               observed <= DSA_SPARSE_CLAIM_BASE) {
            if (observed <= DSA_SPARSE_CLAIM_BASE) {
                const int32_t claimed_flat =
                    DSA_SPARSE_CLAIM_BASE - observed;
                if (claimed_flat <= global_flat) {
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

    // Only the deterministic minimum flat occurrence owns the payload miss.
    for (uint32_t entry = tid;
         entry < request_entry_count;
         entry += thread_count) {
        const uint32_t lane = entry / topk_count;
        const uint32_t rank = entry - lane * topk_count;
        const int32_t query = request_queries[lane];
        const int32_t valid_topk_count =
            query < 0 ? 0 : valid_topk_counts[query];
        if (query < 0 || query_valid_mask[query] == 0U ||
            valid_topk_count <= 0 ||
            rank >= static_cast<uint32_t>(
                        valid_topk_count)) {
            continue;
        }
        const uint64_t output_offset =
            static_cast<uint64_t>(query) * topk_count + rank;
        const int32_t token = topk_positions[output_offset];
        if (token < 0 || token >= request_seq_len ||
            token >= static_cast<int32_t>(
                         token_position_capacity)) {
            continue;
        }
        const int32_t claim =
            DSA_SPARSE_CLAIM_BASE -
            static_cast<int32_t>(output_offset);
        if (request_token_to_hot[
                static_cast<uint32_t>(token)] == claim) {
            miss_mask[output_offset] = 1U;
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Stable partition of the previous LRU order. All resident selections in
    // this request-wide T*K union are protected from this call's victims. This
    // first list preserves old LRU order; victim selection applies the
    // independent free-before-occupied policy below.
    const uint32_t lru_chunk =
        (evictable_slot_count + thread_count - 1U) /
        thread_count;
    const uint32_t lru_begin = tid * lru_chunk;
    uint32_t lru_end = lru_begin + lru_chunk;
    if (lru_end > evictable_slot_count) {
        lru_end = evictable_slot_count;
    }

    int32_t local_hits = 0;
    int32_t local_evictables = 0;
    for (uint32_t pos = lru_begin; pos < lru_end; ++pos) {
        const uint32_t slot =
            static_cast<uint32_t>(request_lru[pos]);
        if (hit_flags[slot] != 0) {
            ++local_hits;
        } else {
            ++local_evictables;
        }
    }
    thread_hit_counts[tid] = local_hits;
    thread_evict_counts[tid] = local_evictables;
    asc_threadfence_block();
    asc_syncthreads();

    int32_t hit_prefix = 0;
    int32_t evict_prefix = 0;
    for (uint32_t other = 0; other < tid; ++other) {
        hit_prefix += thread_hit_counts[other];
        evict_prefix += thread_evict_counts[other];
    }
    int32_t hit_offset = hit_prefix;
    int32_t evict_offset = evict_prefix;
    for (uint32_t pos = lru_begin; pos < lru_end; ++pos) {
        const int32_t slot = request_lru[pos];
        if (hit_flags[static_cast<uint32_t>(slot)] != 0) {
            hit_slots[hit_offset++] = slot;
        } else {
            evictable_slots[evict_offset++] = slot;
        }
    }
    if (tid == thread_count - 1U) {
        counters[0] = hit_prefix + local_hits;
        counters[1] = evict_prefix + local_evictables;
    }
    asc_threadfence_block();
    asc_syncthreads();

    const uint32_t hit_count =
        static_cast<uint32_t>(counters[0]);
    const uint32_t evictable_count =
        static_cast<uint32_t>(counters[1]);

    // Victim choice is free-first, while each subgroup preserves old LRU
    // order. hit_flags is no longer needed as a bitmap after the stable
    // hit/evictable partition, so reuse its S entries as the candidate list.
    const uint32_t candidate_chunk =
        (evictable_count + thread_count - 1U) /
        thread_count;
    const uint32_t candidate_begin = tid * candidate_chunk;
    uint32_t candidate_end =
        candidate_begin + candidate_chunk;
    if (candidate_end > evictable_count) {
        candidate_end = evictable_count;
    }
    int32_t local_free = 0;
    int32_t local_occupied = 0;
    for (uint32_t pos = candidate_begin;
         pos < candidate_end;
         ++pos) {
        const uint32_t slot =
            static_cast<uint32_t>(evictable_slots[pos]);
        if (request_hot_to_token[slot] ==
            DSA_SPARSE_NOT_FOUND) {
            ++local_free;
        } else {
            ++local_occupied;
        }
    }
    thread_hit_counts[tid] = local_free;
    thread_evict_counts[tid] = local_occupied;
    asc_threadfence_block();
    asc_syncthreads();

    int32_t free_prefix = 0;
    int32_t occupied_prefix = 0;
    int32_t total_free = 0;
    for (uint32_t other = 0; other < tid; ++other) {
        free_prefix += thread_hit_counts[other];
        occupied_prefix += thread_evict_counts[other];
    }
    for (uint32_t other = 0;
         other < thread_count;
         ++other) {
        total_free += thread_hit_counts[other];
    }
    int32_t free_offset = free_prefix;
    int32_t occupied_offset =
        total_free + occupied_prefix;
    for (uint32_t pos = candidate_begin;
         pos < candidate_end;
         ++pos) {
        const int32_t slot = evictable_slots[pos];
        if (request_hot_to_token[
                static_cast<uint32_t>(slot)] ==
            DSA_SPARSE_NOT_FOUND) {
            hit_flags[free_offset++] = slot;
        } else {
            hit_flags[occupied_offset++] = slot;
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Compact canonical misses in global flat [Q,K] order. The four query
    // lanes are sorted by q while their reserved-slot lane identity remains
    // unchanged.
    const uint32_t query_chunk =
        (request_entry_count + thread_count - 1U) /
        thread_count;
    const uint32_t query_begin = tid * query_chunk;
    uint32_t query_end = query_begin + query_chunk;
    if (query_end > request_entry_count) {
        query_end = request_entry_count;
    }

    int32_t local_misses = 0;
    for (uint32_t entry = query_begin;
         entry < query_end;
         ++entry) {
        const uint32_t order_index = entry / topk_count;
        const uint32_t lane = static_cast<uint32_t>(
            ordered_lanes[order_index]);
        const uint32_t rank =
            entry - order_index * topk_count;
        const int32_t query = request_queries[lane];
        if (query < 0) {
            continue;
        }
        const uint64_t output_offset =
            static_cast<uint64_t>(query) * topk_count + rank;
        if (miss_mask[output_offset] != 0U) {
            ++local_misses;
        }
    }
    thread_miss_counts[tid] = local_misses;
    asc_threadfence_block();
    asc_syncthreads();

    int32_t miss_prefix = 0;
    for (uint32_t other = 0; other < tid; ++other) {
        miss_prefix += thread_miss_counts[other];
    }
    int32_t miss_offset = miss_prefix;
    for (uint32_t entry = query_begin;
         entry < query_end;
         ++entry) {
        const uint32_t order_index = entry / topk_count;
        const uint32_t lane = static_cast<uint32_t>(
            ordered_lanes[order_index]);
        const uint32_t rank =
            entry - order_index * topk_count;
        const int32_t query = request_queries[lane];
        if (query < 0) {
            continue;
        }
        const uint64_t output_offset =
            static_cast<uint64_t>(query) * topk_count + rank;
        if (miss_mask[output_offset] == 0U) {
            continue;
        }

        const uint32_t miss_rank =
            static_cast<uint32_t>(miss_offset++);
        const uint32_t victim_slot =
            static_cast<uint32_t>(hit_flags[miss_rank]);
        const int32_t victim_token =
            request_hot_to_token[victim_slot];
        if (victim_token >= 0 &&
            victim_token <
                static_cast<int32_t>(
                    token_position_capacity)) {
            request_token_to_hot[
                static_cast<uint32_t>(victim_token)] =
                DSA_SPARSE_NOT_FOUND;
        }

        const int32_t token = topk_positions[output_offset];
        request_hot_to_token[victim_slot] = token;
        request_token_to_hot[static_cast<uint32_t>(token)] =
            static_cast<int32_t>(victim_slot);
        resolved_hot_indices[output_offset] =
            static_cast<int32_t>(victim_slot);
        hit_slots[hit_count + miss_rank] =
            static_cast<int32_t>(victim_slot);
    }
    if (tid == thread_count - 1U) {
        counters[2] = miss_prefix + local_misses;
    }
    asc_threadfence_block();
    asc_syncthreads();

    const uint32_t miss_count =
        static_cast<uint32_t>(counters[2]);
    const uint32_t stale_count =
        evictable_count - miss_count;

    // Mark allocated victims, then compact untouched evictables in their
    // original LRU order. This keeps victim priority (free-first) independent
    // from the final approximate-LRU stale ordering.
    for (uint32_t slot = tid;
         slot < evictable_slot_count;
         slot += thread_count) {
        hit_flags[slot] = 0;
    }
    asc_threadfence_block();
    asc_syncthreads();
    for (uint32_t rank = tid;
         rank < miss_count;
         rank += thread_count) {
        const uint32_t slot = static_cast<uint32_t>(
            hit_slots[hit_count + rank]);
        hit_flags[slot] = 1;
    }
    asc_threadfence_block();
    asc_syncthreads();

    const uint32_t stale_chunk =
        (evictable_count + thread_count - 1U) /
        thread_count;
    const uint32_t stale_begin = tid * stale_chunk;
    uint32_t stale_end = stale_begin + stale_chunk;
    if (stale_end > evictable_count) {
        stale_end = evictable_count;
    }
    int32_t local_untouched = 0;
    for (uint32_t pos = stale_begin;
         pos < stale_end;
         ++pos) {
        const uint32_t slot =
            static_cast<uint32_t>(evictable_slots[pos]);
        if (hit_flags[slot] == 0) {
            ++local_untouched;
        }
    }
    thread_evict_counts[tid] = local_untouched;
    asc_threadfence_block();
    asc_syncthreads();

    int32_t untouched_prefix = 0;
    for (uint32_t other = 0; other < tid; ++other) {
        untouched_prefix += thread_evict_counts[other];
    }
    int32_t untouched_offset = untouched_prefix;
    for (uint32_t pos = stale_begin;
         pos < stale_end;
         ++pos) {
        const int32_t slot = evictable_slots[pos];
        if (hit_flags[static_cast<uint32_t>(slot)] == 0) {
            request_lru[untouched_offset++] = slot;
        }
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Batch approximate LRU, from LRU to MRU:
    //   untouched stale + newly allocated + selected resident.
    for (uint32_t rank = tid;
         rank < miss_count;
         rank += thread_count) {
        request_lru[stale_count + rank] =
            hit_slots[hit_count + rank];
    }
    for (uint32_t rank = tid;
         rank < hit_count;
         rank += thread_count) {
        request_lru[stale_count + miss_count + rank] =
            hit_slots[rank];
    }
    asc_threadfence_block();
    asc_syncthreads();

    // Duplicate followers reuse the canonical occurrence's installed slot.
    for (uint32_t entry = tid;
         entry < request_entry_count;
         entry += thread_count) {
        const uint32_t lane = entry / topk_count;
        const uint32_t rank = entry - lane * topk_count;
        const int32_t query = request_queries[lane];
        if (query < 0) {
            continue;
        }
        const uint64_t output_offset =
            static_cast<uint64_t>(query) * topk_count + rank;
        if (resolved_hot_indices[output_offset] !=
            DSA_SPARSE_NOT_FOUND) {
            continue;
        }
        const int32_t token = topk_positions[output_offset];
        const int32_t valid_topk_count =
            valid_topk_counts[query];
        if (query_valid_mask[query] != 0U &&
            valid_topk_count > 0 &&
            rank < static_cast<uint32_t>(
                       valid_topk_count) &&
            token >= 0 && token < request_seq_len &&
            token <
                static_cast<int32_t>(
                    token_position_capacity)) {
            resolved_hot_indices[output_offset] =
                request_token_to_hot[
                    static_cast<uint32_t>(token)];
        }
    }
}

}  // namespace DsaSparseLookupUpdate

#endif  // DSA_SPARSE_LOOKUP_UPDATE_SIMT_H
