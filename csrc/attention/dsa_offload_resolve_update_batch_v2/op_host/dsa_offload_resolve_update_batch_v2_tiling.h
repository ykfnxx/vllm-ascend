/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_RESOLVE_UPDATE_BATCH_V2_TILING_H
#define DSA_OFFLOAD_RESOLVE_UPDATE_BATCH_V2_TILING_H

#include <cstdint>

struct DsaOffloadResolveUpdateBatchV2TilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t queryNum;
    uint32_t blockSize;
    uint32_t decodeMode;
};

struct DsaOffloadResolveUpdateBatchV2CompileInfo {};

#endif
