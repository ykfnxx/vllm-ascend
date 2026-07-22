/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef lightning_indexer_decode_update_pool_TILING_H_
#define lightning_indexer_decode_update_pool_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct HMRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct HMTensorParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGHTS_INDEX = 2;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 3;
constexpr uint32_t CACHE_SLOTS_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INDEX = 6;
constexpr uint32_t TOPK_INDEX = 0;
constexpr uint32_t TOPK_SLOTS_INDEX = 1;
constexpr uint32_t MISS_COUNT_INDEX = 2;

constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

constexpr uint32_t DECODE_N2 = 1;
constexpr uint32_t DECODE_G_SIZE = 64;
constexpr uint32_t DECODE_HEAD_DIM = 128;
constexpr uint32_t DECODE_SPARSE_COUNT = 2048;
constexpr uint32_t CACHE_SLOTS_SIZE = 262144;

BEGIN_TILING_DATA_DEF(LIUpdatePoolTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LightningIndexerDecodeUpdatePool, LIUpdatePoolTilingData)

struct LIUpdatePoolCompileInfo {};

struct LIUpdatePoolParaInfo {
    HMRequiredParaInfo query = {nullptr, nullptr};
    HMRequiredParaInfo key = {nullptr, nullptr};
    HMRequiredParaInfo weights = {nullptr, nullptr};
    HMTensorParaInfo reqPoolEntries = {nullptr, nullptr};
    HMRequiredParaInfo cacheSlots = {nullptr, nullptr};
    HMTensorParaInfo actualSeqLengths = {nullptr, nullptr};
    HMTensorParaInfo blockTable = {nullptr, nullptr};
    HMRequiredParaInfo topkIndexOut = {nullptr, nullptr};
    HMRequiredParaInfo topkSlotsOut = {nullptr, nullptr};
    HMRequiredParaInfo missCountOut = {nullptr, nullptr};
};

class LIUpdatePoolTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    LIUpdatePoolParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t n1Size = DECODE_G_SIZE;
    uint32_t n2Size = DECODE_N2;
    uint32_t s2Size = 0;
    uint32_t poolSize = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t usedCoreNum = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
};

class LightningIndexerDecodeUpdatePoolTiling {
public:
    explicit LightningIndexerDecodeUpdatePoolTiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus ParseAndCheck(LIUpdatePoolTilingInfo &tilingInfo);
    ge::graphStatus DoTiling(LIUpdatePoolTilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(LIUpdatePoolTilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(LIUpdatePoolTilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const LIUpdatePoolTilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(LIUpdatePoolTilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    LIUpdatePoolTilingData tilingData_;
};

} // namespace optiling
#endif // lightning_indexer_decode_update_pool_TILING_H_
