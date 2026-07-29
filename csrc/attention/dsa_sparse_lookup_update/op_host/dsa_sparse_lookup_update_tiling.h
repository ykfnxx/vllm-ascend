/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TILING_H
#define DSA_SPARSE_LOOKUP_UPDATE_TILING_H

#include <cstdint>

struct DsaSparseLookupUpdateTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
    uint32_t workspaceStride;
};

struct DsaSparseLookupUpdateCompileInfo {};

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TILING_H
