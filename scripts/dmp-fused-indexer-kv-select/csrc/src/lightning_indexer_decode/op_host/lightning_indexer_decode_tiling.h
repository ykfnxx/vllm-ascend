/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef LIGHTNING_INDEXER_DECODE_TILING_H_
#define LIGHTNING_INDEXER_DECODE_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct DecodeRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct DecodeTensorParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGHTS_INDEX = 2;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 3;
constexpr uint32_t BLOCK_TABLE_INDEX = 4;
constexpr uint32_t SPARSE_INDICES_INDEX = 0;

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

BEGIN_TILING_DATA_DEF(LIDecodeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LightningIndexerDecode, LIDecodeTilingData)

struct LIDecodeCompileInfo {};

struct LIDecodeParaInfo {
    DecodeRequiredParaInfo query = {nullptr, nullptr};
    DecodeRequiredParaInfo key = {nullptr, nullptr};
    DecodeRequiredParaInfo weights = {nullptr, nullptr};
    DecodeTensorParaInfo actualSeqLengths = {nullptr, nullptr};
    DecodeTensorParaInfo blockTable = {nullptr, nullptr};
    DecodeRequiredParaInfo sparseIndicesOut = {nullptr, nullptr};
};

class LIDecodeTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    LIDecodeParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t n1Size = DECODE_G_SIZE;
    uint32_t n2Size = DECODE_N2;
    uint32_t s2Size = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t usedCoreNum = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
};

class LightningIndexerDecodeTiling {
public:
    explicit LightningIndexerDecodeTiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus ParseAndCheck(LIDecodeTilingInfo &tilingInfo);
    ge::graphStatus DoTiling(LIDecodeTilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(LIDecodeTilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(LIDecodeTilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const LIDecodeTilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(LIDecodeTilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    LIDecodeTilingData tilingData_;
};

} // namespace optiling
#endif // LIGHTNING_INDEXER_DECODE_TILING_H_
