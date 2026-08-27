/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_LOOKUP_UPDATE_BATCH_COMMON_H
#define DSA_OFFLOAD_LOOKUP_UPDATE_BATCH_COMMON_H

#include <cstdint>

constexpr uint32_t DSA_OFFLOAD_BATCH_SIMT_THREADS = 256U;
constexpr uint32_t DSA_OFFLOAD_BATCH_INDEX_CAPACITY = 128U * 1024U;
constexpr uint32_t DSA_OFFLOAD_BATCH_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t DSA_OFFLOAD_BATCH_FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t DSA_OFFLOAD_BATCH_QUERY_WIDTH = 2U * 1024U;
constexpr uint32_t DSA_OFFLOAD_BATCH_FREE_HEAD_STRIDE = 16U;
constexpr uint32_t DSA_OFFLOAD_BATCH_WARP_SIZE = 32U;
constexpr uint32_t DSA_OFFLOAD_BATCH_WARP_COUNT =
    DSA_OFFLOAD_BATCH_SIMT_THREADS / DSA_OFFLOAD_BATCH_WARP_SIZE;
constexpr uint32_t DSA_OFFLOAD_BATCH_PROTECTED_WORDS =
    DSA_OFFLOAD_BATCH_SLOT_COUNT / 32U;
constexpr uint32_t DSA_OFFLOAD_BATCH_SHARED_SCALARS = 7U;
constexpr uint32_t DSA_OFFLOAD_BATCH_UB_SCRATCH_WORDS =
    DSA_OFFLOAD_BATCH_PROTECTED_WORDS +
    DSA_OFFLOAD_BATCH_WARP_COUNT +
    DSA_OFFLOAD_BATCH_SHARED_SCALARS;
constexpr int32_t DSA_OFFLOAD_BATCH_NOT_FOUND = -1;
constexpr int32_t DSA_OFFLOAD_BATCH_FALLBACK_SLOT =
    static_cast<int32_t>(DSA_OFFLOAD_BATCH_SLOT_COUNT);

static_assert(
    DSA_OFFLOAD_BATCH_QUERY_WIDTH % DSA_OFFLOAD_BATCH_SIMT_THREADS == 0U,
    "query work must divide evenly across SIMT threads");
static_assert(
    DSA_OFFLOAD_BATCH_SLOT_COUNT % DSA_OFFLOAD_BATCH_SIMT_THREADS == 0U,
    "slot scan must divide evenly across SIMT threads");
static_assert(
    DSA_OFFLOAD_BATCH_SIMT_THREADS % DSA_OFFLOAD_BATCH_WARP_SIZE == 0U,
    "SIMT thread count must contain complete warps");

struct DsaOffloadLookupUpdateBatchTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t queryNum;
};

#endif  // DSA_OFFLOAD_LOOKUP_UPDATE_BATCH_COMMON_H
