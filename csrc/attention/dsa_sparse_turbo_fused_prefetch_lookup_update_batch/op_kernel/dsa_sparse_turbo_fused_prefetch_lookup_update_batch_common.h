/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_PREFETCH_COMMON_H
#define DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_PREFETCH_COMMON_H

#include <cstdint>

constexpr uint32_t DSA_SPARSE_TURBO_FUSED_SIMT_THREADS = 256U;
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_FREE_SLOT_COUNT = 2U * 1024U;
// Prefetch maintenance scans resident slots only; the exact path reclaims
// stale mappings in the free-slot region.
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT =
    DSA_SPARSE_TURBO_FUSED_SLOT_COUNT - DSA_SPARSE_TURBO_FUSED_FREE_SLOT_COUNT;
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_QUERY_WIDTH = 2U * 1024U;
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_FREE_HEAD_STRIDE = 16U;
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_WARP_SIZE = 32U;
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_WARP_COUNT =
    DSA_SPARSE_TURBO_FUSED_SIMT_THREADS / DSA_SPARSE_TURBO_FUSED_WARP_SIZE;
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_PROTECTED_WORDS =
    DSA_SPARSE_TURBO_FUSED_SLOT_COUNT / 32U;
// Prefetch scalars (9, including the refill counter) plus the per-request
// tail start shared by every query's classification.
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_SHARED_SCALARS = 10U;
// Per-request allocation ledger, indexed by the global allocation rank
// (rank < FREE_SLOT_COUNT by construction).  Each entry records the flat
// query offset and the assigned slot of one miss allocation, so the
// request-level maintain can revert allocations that exceed the evictable
// victim count (the overflow fallback) without re-deriving per-query state.
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_ALLOC_RECORD_PAIRS =
    DSA_SPARSE_TURBO_FUSED_FREE_SLOT_COUNT;
constexpr uint32_t DSA_SPARSE_TURBO_FUSED_UB_SCRATCH_WORDS =
    DSA_SPARSE_TURBO_FUSED_PROTECTED_WORDS +
    DSA_SPARSE_TURBO_FUSED_WARP_COUNT +
    DSA_SPARSE_TURBO_FUSED_SHARED_SCALARS +
    2U * DSA_SPARSE_TURBO_FUSED_ALLOC_RECORD_PAIRS;
constexpr int32_t DSA_SPARSE_TURBO_FUSED_NOT_FOUND = -1;
constexpr int32_t DSA_SPARSE_TURBO_FUSED_FALLBACK_SLOT =
    static_cast<int32_t>(DSA_SPARSE_TURBO_FUSED_SLOT_COUNT);

static_assert(
    DSA_SPARSE_TURBO_FUSED_QUERY_WIDTH % DSA_SPARSE_TURBO_FUSED_SIMT_THREADS == 0U,
    "query work must divide evenly across SIMT threads");
static_assert(
    DSA_SPARSE_TURBO_FUSED_SLOT_COUNT % DSA_SPARSE_TURBO_FUSED_SIMT_THREADS == 0U,
    "slot scan must divide evenly across SIMT threads");
static_assert(
    DSA_SPARSE_TURBO_FUSED_RESIDENT_SLOT_COUNT %
            DSA_SPARSE_TURBO_FUSED_SIMT_THREADS ==
        0U,
    "resident scan must divide evenly across SIMT threads");
static_assert(
    DSA_SPARSE_TURBO_FUSED_SIMT_THREADS % DSA_SPARSE_TURBO_FUSED_WARP_SIZE == 0U,
    "SIMT thread count must contain complete warps");

struct DsaSparseTurboFusedPrefetchLookupUpdateBatchTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t queryNum;
    // Runtime index width: the KV token space (index table stride), passed by
    // tiling instead of a compile-time 128K constant so the op supports KV
    // sequence lengths beyond 128K (e.g. 1024K) without kernel recompiles.
    uint32_t indexCapacity;
    // Hot Cache layout constants for the destination-slot mapping, derived in
    // host tiling from the attrs (see design_and_test.md §2.1).
    int32_t replaceableBase;
};

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_PREFETCH_COMMON_H
