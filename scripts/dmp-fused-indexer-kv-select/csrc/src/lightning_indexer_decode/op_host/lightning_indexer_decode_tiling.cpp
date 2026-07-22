/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "lightning_indexer_decode_tiling.h"
#include <algorithm>
#include "../op_kernel/lightning_indexer_decode_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
namespace {
template <typename T>
T CeilDiv(T a, T b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}
} // namespace

ge::graphStatus LightningIndexerDecodeTiling::GetNpuInfo(LIDecodeTilingInfo &tilingInfo) const
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("LightningIndexerDecode", "opName got from TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.opName = context_->GetNodeName();
    tilingInfo.platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(tilingInfo.platformInfo == nullptr, OPS_LOG_E(tilingInfo.opName, "GetPlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(tilingInfo.opName, "num of core obtained is 0."),
               return ge::GRAPH_FAILED);

    tilingInfo.socVersion = ascendcPlatform.GetSocVersion();
    OPS_ERR_IF((tilingInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) &&
                   (tilingInfo.socVersion != platform_ascendc::SocVersion::ASCEND910_93),
               OPS_LOG_E(tilingInfo.opName, "SOC Version[%d] is not supported.",
                         static_cast<int32_t>(tilingInfo.socVersion)),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
               OPS_LOG_E(tilingInfo.opName, "workspace size buffer is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(tilingInfo.opName, "raw tiling data is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeTiling::GetTensorInfo(LIDecodeTilingInfo &tilingInfo) const
{
    auto &op = tilingInfo.opParamInfo;
    op.query.desc = context_->GetInputDesc(QUERY_INDEX);
    op.query.shape = context_->GetInputShape(QUERY_INDEX);
    op.key.desc = context_->GetInputDesc(KEY_INDEX);
    op.key.shape = context_->GetInputShape(KEY_INDEX);
    op.weights.desc = context_->GetInputDesc(WEIGHTS_INDEX);
    op.weights.shape = context_->GetInputShape(WEIGHTS_INDEX);
    op.actualSeqLengths.desc = context_->GetInputDesc(ACTUAL_SEQ_K_INDEX);
    op.actualSeqLengths.tensor = context_->GetInputTensor(ACTUAL_SEQ_K_INDEX);
    op.blockTable.desc = context_->GetInputDesc(BLOCK_TABLE_INDEX);
    op.blockTable.tensor = context_->GetInputTensor(BLOCK_TABLE_INDEX);
    op.sparseIndicesOut.desc = context_->GetOutputDesc(SPARSE_INDICES_INDEX);
    op.sparseIndicesOut.shape = context_->GetOutputShape(SPARSE_INDICES_INDEX);

    OPS_ERR_IF(op.query.desc == nullptr || op.query.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "query desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.key.desc == nullptr || op.key.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "key desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.weights.desc == nullptr || op.weights.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "weights desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengths.desc == nullptr || op.actualSeqLengths.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_key desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc == nullptr || op.blockTable.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "block_table desc/tensor is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.sparseIndicesOut.desc == nullptr || op.sparseIndicesOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "sparse_indices desc/shape is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeTiling::CheckDtype(const LIDecodeTilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    ge::DataType qType = op.query.desc->GetDataType();
    ge::DataType kType = op.key.desc->GetDataType();
    ge::DataType wType = op.weights.desc->GetDataType();
    OPS_ERR_IF(qType != kType || qType != wType,
               OPS_LOG_E(tilingInfo.opName, "query/key/weights dtype must match."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(qType != ge::DT_FLOAT16 && qType != ge::DT_BF16,
               OPS_LOG_E(tilingInfo.opName, "query/key/weights dtype must be fp16 or bf16."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_key dtype must be int32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "block_table dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.sparseIndicesOut.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "sparse_indices dtype must be int32."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeTiling::CheckShape(LIDecodeTilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    const auto &qShape = op.query.shape->GetStorageShape();
    const auto &kShape = op.key.shape->GetStorageShape();
    const auto &wShape = op.weights.shape->GetStorageShape();
    const auto &seqShape = op.actualSeqLengths.tensor->GetStorageShape();
    const auto &blockShape = op.blockTable.tensor->GetStorageShape();
    const auto &outShape = op.sparseIndicesOut.shape->GetStorageShape();

    OPS_ERR_IF(qShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "query must be TND [B, 64, 128]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDimNum() != DIM_NUM_FOUR,
               OPS_LOG_E(tilingInfo.opName, "key must be PA_BSND [num_blocks, block_size, 1, 128]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "weights must be [B, 64]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_key must be rank 1."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "block_table must be rank 2."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(outShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "sparse_indices must be [B, 1, 2048]."),
               return ge::GRAPH_FAILED);

    tilingInfo.bSize = static_cast<uint32_t>(qShape.GetDim(0));
    tilingInfo.n1Size = static_cast<uint32_t>(qShape.GetDim(1));
    tilingInfo.n2Size = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_TWO));
    tilingInfo.blockSize = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_ONE));
    tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape.GetDim(DIM_IDX_ONE));
    tilingInfo.s2Size = tilingInfo.blockSize * tilingInfo.maxBlockNumPerBatch;

    OPS_ERR_IF(tilingInfo.bSize == 0, OPS_LOG_E(tilingInfo.opName, "batch size must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqShape.GetShapeSize() != tilingInfo.bSize || blockShape.GetDim(0) != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName,
                         "query batch, actual_seq_lengths_key length, and block_table batch must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(0) == 0, OPS_LOG_E(tilingInfo.opName, "key num_blocks must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.blockSize == 0 || tilingInfo.blockSize > 1024 || tilingInfo.blockSize % 16 != 0,
               OPS_LOG_E(tilingInfo.opName, "key block_size must be a multiple of 16 in (0, 1024]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.s2Size > 262144,
               OPS_LOG_E(tilingInfo.opName, "maxBlockNumPerBatch * blockSize must be <= 262144."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n2Size != DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key N2 must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n1Size != DECODE_G_SIZE,
               OPS_LOG_E(tilingInfo.opName, "decode query N1 must be 64."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape.GetDim(DIM_IDX_TWO) != DECODE_HEAD_DIM || kShape.GetDim(DIM_IDX_THREE) != DECODE_HEAD_DIM,
               OPS_LOG_E(tilingInfo.opName, "head_dim must be 128."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDim(0) != tilingInfo.bSize || wShape.GetDim(1) != DECODE_G_SIZE,
               OPS_LOG_E(tilingInfo.opName, "weights must have shape [B, 64]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(outShape.GetDim(0) != tilingInfo.bSize || outShape.GetDim(1) != DECODE_N2 ||
                   outShape.GetDim(2) != DECODE_SPARSE_COUNT,
               OPS_LOG_E(tilingInfo.opName, "sparse_indices must have shape [B, 1, 2048]."),
               return ge::GRAPH_FAILED);

    tilingInfo.inputQType = op.query.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeTiling::ParseAndCheck(LIDecodeTilingInfo &tilingInfo)
{
    if (GetNpuInfo(tilingInfo) != ge::GRAPH_SUCCESS || GetTensorInfo(tilingInfo) != ge::GRAPH_SUCCESS ||
        CheckDtype(tilingInfo) != ge::GRAPH_SUCCESS || CheckShape(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeTiling::DoTiling(LIDecodeTilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    tilingInfo->usedCoreNum = std::min(tilingInfo->bSize, aicNum);
    uint32_t requestedAivNum = std::min(aivNum, tilingInfo->usedCoreNum * 2U);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(requestedAivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    constexpr uint32_t M_BASE_SIZE = 64;
    constexpr uint32_t S2_BASE_SIZE = 512;
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += M_BASE_SIZE * S2_BASE_SIZE * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * blockDim;
    context_->GetWorkspaceSizes(1)[0] = workspaceSize;

    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    uint32_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint32_t>(tilingInfo->inputQType));
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForLightningIndexerDecode(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLightningIndexerDecode(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("LightningIndexerDecode", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    LIDecodeTilingInfo liInfo;
    LightningIndexerDecodeTiling liTiling(context);
    if (liTiling.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return liTiling.DoTiling(&liInfo);
}

IMPL_OP_OPTILING(LightningIndexerDecode)
    .Tiling(TilingForLightningIndexerDecode)
    .TilingParse<LIDecodeCompileInfo>(TilingPrepareForLightningIndexerDecode);

} // namespace optiling
