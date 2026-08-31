/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_TURBO_RESOLVE_UPDATE_BATCH_V2_COMMON_H
#define DSA_SPARSE_TURBO_RESOLVE_UPDATE_BATCH_V2_COMMON_H

#include <cstdint>

constexpr uint32_t DSA_TURBO_RESOLVE_V2_SIMT_THREADS = 256U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_INDEX_CAPACITY = 128U * 1024U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_QUERY_WIDTH = 2U * 1024U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_FREE_HEAD_STRIDE = 16U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_PROTECTED_WORDS =
    DSA_TURBO_RESOLVE_V2_SLOT_COUNT / 32U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_WARP_COUNT = 8U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_SHARED_SCALARS = 10U;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_ALLOC_RECORD_WORDS =
    2U * DSA_TURBO_RESOLVE_V2_FREE_SLOT_COUNT;
constexpr uint32_t DSA_TURBO_RESOLVE_V2_UB_SCRATCH_WORDS =
    DSA_TURBO_RESOLVE_V2_PROTECTED_WORDS +
    DSA_TURBO_RESOLVE_V2_WARP_COUNT +
    DSA_TURBO_RESOLVE_V2_SHARED_SCALARS +
    DSA_TURBO_RESOLVE_V2_ALLOC_RECORD_WORDS;

struct DsaSparseTurboResolveUpdateBatchV2TilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t queryNum;
    uint32_t blockSize;
    uint32_t decodeMode;
};

#endif
