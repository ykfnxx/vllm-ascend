/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef ASU_KV_GATHER_DIRECT_V2_TILING_H
#define ASU_KV_GATHER_DIRECT_V2_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsuKvGatherDirectV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, reqNum);
TILING_DATA_FIELD_DEF(uint32_t, queryNum);
TILING_DATA_FIELD_DEF(uint32_t, topkWidth);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, sourceTableWidth);
TILING_DATA_FIELD_DEF(uint32_t, destinationTableWidth);
TILING_DATA_FIELD_DEF(uint32_t, kvRecordElements);
TILING_DATA_FIELD_DEF(uint32_t, ropeRecordElements);
TILING_DATA_FIELD_DEF(uint32_t, poolCapacity);
TILING_DATA_FIELD_DEF(uint32_t, sourcePhysicalBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, destinationPhysicalBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, jitterEnable);
TILING_DATA_FIELD_DEF(uint32_t, jitterSeed);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    AsuKvGatherDirectV2, AsuKvGatherDirectV2TilingData)
}  // namespace optiling

#endif
