/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_HI_CACHED_TILING_H_
#define LIGHTNING_INDEXER_HI_CACHED_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {

struct LIHiCachedTilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct LIHiCachedTilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class LIHiCachedDataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    BnBsND = 2
};

// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGTHS_INDEX = 2;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
constexpr uint32_t LIGHTNING_INDEXER = 0;
// Attributes Index
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 3;
// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// Input Parameter Limit Constant
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;
constexpr uint32_t HEAD_RATIO_32 = 32;
constexpr uint32_t HEAD_RATIO_64 = 64;

struct LIHiCachedLICompileInfo {};

struct LIHiCachedLiParaInfo {
    LIHiCachedTilingRequiredParaInfo query = {nullptr, nullptr};
    LIHiCachedTilingRequiredParaInfo key = {nullptr, nullptr};
    LIHiCachedTilingRequiredParaInfo weights = {nullptr, nullptr};
    LIHiCachedTilingOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    LIHiCachedTilingOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    LIHiCachedTilingOptionalParaInfo blockTable = {nullptr, nullptr};
    LIHiCachedTilingRequiredParaInfo attenOut = {nullptr, nullptr};

    const char *layOut = nullptr;
    const char *layOutKey = nullptr;
    const int32_t *blockSize = nullptr;
    const int32_t *sparseMode = nullptr;
    const int32_t *sparseCount = nullptr;
};

class LIHiCachedLITilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LIHiCachedLiParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t gSize = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    LIHiCachedDataLayout inputQLayout = LIHiCachedDataLayout::BSND;
    LIHiCachedDataLayout inputKLayout = LIHiCachedDataLayout::BnBsND;
};

class LIHiCachedLIInfoParser {
public:
    explicit LIHiCachedLIInfoParser(gert::TilingContext *context) : context_(context)
    {
    }
    ~LIHiCachedLIInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatchQBsnd();
    ge::graphStatus ValidateInputShapesMatchQTnd();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus CheckBlockCount();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetActualSeqInfo();
    void GenerateInfo(LIHiCachedLITilingInfo &liInfo);
    ge::graphStatus ParseAndCheck(LIHiCachedLITilingInfo &liInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    LIHiCachedLiParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    // Layout
    LIHiCachedDataLayout qLayout_ = LIHiCachedDataLayout::BSND;
    LIHiCachedDataLayout kLayout_ = LIHiCachedDataLayout::BnBsND;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType blockTableType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
};


constexpr uint32_t ATTR_HI_BLOCK_SIZE_INDEX = 4;
constexpr uint32_t ATTR_HI_BLOCK_NUM_INDEX = 5;
constexpr uint32_t ATTR_SINK_INDEX = 6;
constexpr uint32_t ATTR_RECENT_INDEX = 7;
constexpr uint32_t ATTR_BLOCK_POOLING_MODE_INDEX = 8;

constexpr uint32_t BLOCK_POOLING_MODE_MEAN = 0;

BEGIN_TILING_DATA_DEF(LIHiCachedTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, hiBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, hiBlockNum)
TILING_DATA_FIELD_DEF(uint32_t, sink)
TILING_DATA_FIELD_DEF(uint32_t, recent)
TILING_DATA_FIELD_DEF(uint32_t, blockPoolingMode)
TILING_DATA_FIELD_DEF(uint32_t, keyBlockNum)
TILING_DATA_FIELD_DEF(uint32_t, externalHiMaskWordNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LightningIndexerHiCached, LIHiCachedTilingData)

struct LIHiCachedAttrInfo {
    uint32_t hiBlockSize = 128;
    uint32_t hiBlockNum = 64;
    uint32_t sink = 1;
    uint32_t recent = 2;
    uint32_t blockPoolingMode = BLOCK_POOLING_MODE_MEAN;
};

class LightningIndexerHiCachedTiling {
public:
    explicit LightningIndexerHiCachedTiling(gert::TilingContext *context) : context_(context)
    {
    }

    ge::graphStatus DoTiling(LIHiCachedLITilingInfo *tilingInfo, const LIHiCachedAttrInfo &hiAttrInfo);

private:
    gert::TilingContext *context_ = nullptr;
    LIHiCachedTilingData tilingData_;
};

ge::graphStatus ParseLightningIndexerHiCachedAttrs(gert::TilingContext *context, LIHiCachedAttrInfo &hiAttrInfo);

} // namespace optiling

#endif // LIGHTNING_INDEXER_HI_CACHED_TILING_H_
