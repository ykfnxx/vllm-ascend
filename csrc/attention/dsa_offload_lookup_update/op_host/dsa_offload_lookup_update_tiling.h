/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_LOOKUP_UPDATE_TILING_H
#define DSA_OFFLOAD_LOOKUP_UPDATE_TILING_H

#include <cstdint>

struct DsaOffloadLookupUpdateTilingData {
    uint32_t reqNum;
    uint32_t blockSize;
    uint32_t replaceableBase;
    uint32_t tailBase;
    uint32_t fallbackSlot;
    uint32_t stagingBase;
    uint32_t decodeMode;
};

struct DsaOffloadLookupUpdateCompileInfo {};

#endif  // DSA_OFFLOAD_LOOKUP_UPDATE_TILING_H
