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
constexpr uint32_t DSA_SPARSE_WORKSPACE_SCALARS = 4U;
constexpr uint32_t DSA_SPARSE_WORKSPACE_STRIDE =
    DSA_SPARSE_SLOT_COUNT + DSA_SPARSE_SIMT_THREADS +
    DSA_SPARSE_WORKSPACE_SCALARS;
static_assert(
    DSA_SPARSE_QUERY_COUNT % DSA_SPARSE_SIMT_THREADS == 0U,
    "query work must divide evenly across SIMT threads");
static_assert(
    DSA_SPARSE_SLOT_COUNT % DSA_SPARSE_SIMT_THREADS == 0U,
    "slot scan must divide evenly across SIMT threads");
constexpr int32_t DSA_SPARSE_NOT_FOUND = -1;
constexpr int32_t DSA_SPARSE_CLAIM_BASE = -2;

struct DsaSparseLookupUpdateTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t workspaceStride;
};

#endif  // DSA_SPARSE_LOOKUP_UPDATE_COMMON_H
