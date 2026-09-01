/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_LOOKUP_UPDATE_COMMON_H
#define DSA_OFFLOAD_LOOKUP_UPDATE_COMMON_H

#include <cstdint>

constexpr uint32_t DSA_OFFLOAD_SIMT_THREADS = 256U;
constexpr uint32_t DSA_OFFLOAD_INDEX_CAPACITY = 128U * 1024U;
constexpr uint32_t DSA_OFFLOAD_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t DSA_OFFLOAD_FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t DSA_OFFLOAD_QUERY_WIDTH = 2U * 1024U;
constexpr uint32_t DSA_OFFLOAD_RESIDENT_SLOTS = 8U * 1024U;
constexpr uint32_t DSA_OFFLOAD_FREE_HEAD_STRIDE = 16U;
constexpr uint32_t DSA_OFFLOAD_WARP_SIZE = 32U;
constexpr uint32_t DSA_OFFLOAD_WARP_COUNT =
    DSA_OFFLOAD_SIMT_THREADS / DSA_OFFLOAD_WARP_SIZE;
constexpr uint32_t DSA_OFFLOAD_PROTECTED_WORDS =
    DSA_OFFLOAD_SLOT_COUNT / 32U;
constexpr uint32_t DSA_OFFLOAD_SHARED_SCALARS = 7U;
constexpr uint32_t DSA_OFFLOAD_UB_SCRATCH_WORDS =
    DSA_OFFLOAD_PROTECTED_WORDS + DSA_OFFLOAD_WARP_COUNT +
    DSA_OFFLOAD_SHARED_SCALARS;
static_assert(
    DSA_OFFLOAD_QUERY_WIDTH % DSA_OFFLOAD_SIMT_THREADS == 0U,
    "query work must divide evenly across SIMT threads");
static_assert(
    DSA_OFFLOAD_SLOT_COUNT % DSA_OFFLOAD_SIMT_THREADS == 0U,
    "slot scan must divide evenly across SIMT threads");
static_assert(
    DSA_OFFLOAD_SIMT_THREADS % DSA_OFFLOAD_WARP_SIZE == 0U,
    "SIMT thread count must contain complete warps");
static_assert(
    DSA_OFFLOAD_SLOT_COUNT % 32U == 0U,
    "protected slot bitset must contain complete words");
constexpr int32_t DSA_OFFLOAD_NOT_FOUND = -1;

struct DsaOffloadLookupUpdateTilingData {
    uint32_t reqNum;
    uint32_t blockSize;
    uint32_t replaceableBase;
    uint32_t tailBase;
    uint32_t fallbackSlot;
    uint32_t stagingBase;
    uint32_t decodeMode;
};

#endif  // DSA_OFFLOAD_LOOKUP_UPDATE_COMMON_H
