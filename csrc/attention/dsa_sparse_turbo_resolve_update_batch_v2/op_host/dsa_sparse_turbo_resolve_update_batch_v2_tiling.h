/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_TURBO_RESOLVE_UPDATE_BATCH_V2_TILING_H
#define DSA_SPARSE_TURBO_RESOLVE_UPDATE_BATCH_V2_TILING_H

#include <cstdint>

struct DsaSparseTurboResolveUpdateBatchV2TilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t queryNum;
    uint32_t blockSize;
    uint32_t decodeMode;
};

struct DsaSparseTurboResolveUpdateBatchV2CompileInfo {};

#endif
