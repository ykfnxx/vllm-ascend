#ifndef ASU_KV_GATHER_TILING_H
#define ASU_KV_GATHER_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsuKvGatherTilingData)
TILING_DATA_FIELD_DEF(uint32_t, reqNum);
TILING_DATA_FIELD_DEF(uint32_t, queryCount);
TILING_DATA_FIELD_DEF(uint32_t, residentInitLayout);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, sourceTableWidth);
TILING_DATA_FIELD_DEF(uint32_t, destinationTableWidth);
TILING_DATA_FIELD_DEF(uint32_t, kvRecordElements);
TILING_DATA_FIELD_DEF(uint32_t, ropeRecordElements);
TILING_DATA_FIELD_DEF(uint32_t, sourcePoolCapacity);
TILING_DATA_FIELD_DEF(uint32_t, sourcePhysicalBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, destinationPhysicalBlockCount);
// Swap 延迟抖动由 tiling 数据控制。
TILING_DATA_FIELD_DEF(uint32_t, jitterEnable);
TILING_DATA_FIELD_DEF(uint32_t, jitterSeed);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsuKvGather, AsuKvGatherTilingData)
}  // namespace optiling

#endif  // ASU_KV_GATHER_TILING_H
