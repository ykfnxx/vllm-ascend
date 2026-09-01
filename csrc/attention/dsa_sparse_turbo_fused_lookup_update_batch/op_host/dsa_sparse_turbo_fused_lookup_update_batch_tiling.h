/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_TILING_H
#define DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_TILING_H

#include <cstdint>

struct DsaSparseTurboFusedLookupUpdateBatchTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t queryNum;
    uint32_t indexCapacity;
    int32_t isMtp;
    int32_t replaceableBase;
    int32_t tailBase;
    int32_t fallbackSlot;
    int32_t stagingBase;
};

struct DsaSparseTurboFusedLookupUpdateBatchCompileInfo {};

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_TILING_H
