/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_LOOKUP_UPDATE_TILING_H
#define DSA_OFFLOAD_LOOKUP_UPDATE_TILING_H

#include <cstdint>

struct DsaOffloadLookupUpdateTilingData {
    uint32_t reqNum;
    uint32_t poolCapacity;
};

struct DsaOffloadLookupUpdateCompileInfo {};

#endif  // DSA_OFFLOAD_LOOKUP_UPDATE_TILING_H
