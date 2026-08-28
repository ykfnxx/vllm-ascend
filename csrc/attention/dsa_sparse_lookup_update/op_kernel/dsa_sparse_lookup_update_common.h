/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_COMMON_H
#define DSA_SPARSE_LOOKUP_UPDATE_COMMON_H

#include <cstdint>

constexpr uint32_t DSA_SPARSE_SIMT_THREADS = 256U;
constexpr uint32_t DSA_SPARSE_INDEX_CAPACITY = 128U * 1024U;
constexpr uint32_t DSA_SPARSE_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t DSA_SPARSE_FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t DSA_SPARSE_QUERY_COUNT = 2U * 1024U;
constexpr uint32_t DSA_SPARSE_FREE_HEAD_STRIDE = 16U;
constexpr uint32_t DSA_SPARSE_WARP_SIZE = 32U;
constexpr uint32_t DSA_SPARSE_WARP_COUNT =
    DSA_SPARSE_SIMT_THREADS / DSA_SPARSE_WARP_SIZE;
constexpr uint32_t DSA_SPARSE_PROTECTED_WORDS =
    DSA_SPARSE_SLOT_COUNT / 32U;
constexpr uint32_t DSA_SPARSE_SHARED_SCALARS = 4U;
constexpr uint32_t DSA_SPARSE_UB_SCRATCH_WORDS =
    DSA_SPARSE_PROTECTED_WORDS + DSA_SPARSE_WARP_COUNT +
    DSA_SPARSE_SHARED_SCALARS;
static_assert(
    DSA_SPARSE_QUERY_COUNT % DSA_SPARSE_SIMT_THREADS == 0U,
    "query work must divide evenly across SIMT threads");
static_assert(
    DSA_SPARSE_SLOT_COUNT % DSA_SPARSE_SIMT_THREADS == 0U,
    "slot scan must divide evenly across SIMT threads");
static_assert(
    DSA_SPARSE_SIMT_THREADS % DSA_SPARSE_WARP_SIZE == 0U,
    "SIMT thread count must contain complete warps");
static_assert(
    DSA_SPARSE_SLOT_COUNT % 32U == 0U,
    "protected slot bitset must contain complete words");
constexpr int32_t DSA_SPARSE_NOT_FOUND = -1;

struct DsaSparseLookupUpdateTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
};

#endif  // DSA_SPARSE_LOOKUP_UPDATE_COMMON_H
