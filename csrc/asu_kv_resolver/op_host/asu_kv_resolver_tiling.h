#ifndef ASU_KV_RESOLVER_TILING_H
#define ASU_KV_RESOLVER_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AsuKvResolverTilingData)
TILING_DATA_FIELD_DEF(int64_t, topkNumel)
TILING_DATA_FIELD_DEF(int64_t, actualSeqLen)
TILING_DATA_FIELD_DEF(int64_t, blockSize)
TILING_DATA_FIELD_DEF(int64_t, kv0SlotElements)
TILING_DATA_FIELD_DEF(int64_t, kv1SlotElements)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AsuKvResolver, AsuKvResolverTilingData)

} // namespace optiling

#endif // ASU_KV_RESOLVER_TILING_H
