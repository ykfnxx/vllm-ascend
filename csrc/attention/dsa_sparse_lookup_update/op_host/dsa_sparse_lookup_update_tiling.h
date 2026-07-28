/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TILING_H
#define DSA_SPARSE_LOOKUP_UPDATE_TILING_H

#include <cstdint>

struct DsaSparseLookupUpdateTilingData {
    uint32_t tokenPositionCapacity;
    uint32_t evictableSlotCount;
    uint32_t queryCapacity;
    uint32_t requestCapacity;
    uint32_t queryLaneCapacity;
    uint32_t topkCount;
    uint32_t workspaceStride;
};

struct DsaSparseLookupUpdateCompileInfo {};

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TILING_H
