#ifndef DMP_LOOKUP_KV_GATHER_TILING_H
#define DMP_LOOKUP_KV_GATHER_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DmpLookupKvGatherTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, selectionBlockSize);
TILING_DATA_FIELD_DEF(uint32_t, selectionBlocksPerRow);
TILING_DATA_FIELD_DEF(uint32_t, fullBlockSize);
TILING_DATA_FIELD_DEF(uint32_t, fullBlocksPerRow);
TILING_DATA_FIELD_DEF(uint32_t, kvDim);
TILING_DATA_FIELD_DEF(uint32_t, ropeDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DmpLookupKvGather, DmpLookupKvGatherTilingData)
}  // namespace optiling

#endif  // DMP_LOOKUP_KV_GATHER_TILING_H
