/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_COMMON_H
#define DSA_SPARSE_LOOKUP_UPDATE_COMMON_H

#include <cstdint>

constexpr uint32_t DSA_SPARSE_SIMT_THREADS = 256U;
constexpr uint32_t DSA_SPARSE_MAX_QUERY_LANES = 4U;
constexpr int32_t DSA_SPARSE_NOT_FOUND = -1;
constexpr int32_t DSA_SPARSE_CLAIM_BASE = -2;

struct DsaSparseLookupUpdateTilingData {
    uint32_t tokenPositionCapacity;
    uint32_t evictableSlotCount;
    uint32_t queryCapacity;
    uint32_t requestCapacity;
    uint32_t queryLaneCapacity;
    uint32_t topkCount;
    uint32_t workspaceStride;
};

#endif  // DSA_SPARSE_LOOKUP_UPDATE_COMMON_H
