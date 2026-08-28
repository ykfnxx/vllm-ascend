/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TURBO_TILING_H
#define DSA_SPARSE_LOOKUP_UPDATE_TURBO_TILING_H

#include <cstdint>

struct DsaSparseTurboLookupUpdateBatchTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t queryNum;
    uint32_t indexCapacity;
};

struct DsaSparseTurboLookupUpdateBatchCompileInfo {};

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TURBO_TILING_H
