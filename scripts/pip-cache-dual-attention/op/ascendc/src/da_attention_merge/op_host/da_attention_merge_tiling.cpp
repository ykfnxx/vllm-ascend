#include "da_attention_merge_tiling.h"

#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/op_def_registry.h"
#include <tiling/platform/platform_ascendc.h>
#include "error/ops_error.h"

using namespace ge;

namespace optiling {
namespace {
constexpr size_t ATTENTION_DIM_NUM = 4;
constexpr size_t ATTENTION_TND_DIM_NUM = 3;
constexpr size_t SOFTMAX_DIM_NUM = 2;
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = 8;
constexpr uint32_t DEFAULT_ROW_BLOCK = 8;

bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool IsNullInput(const gert::TilingContext *context, const T *ptr, const char *name)
{
    if (ptr != nullptr) {
        return false;
    }
    OPS_LOG_E(context->GetNodeName(), "%s is nullptr.", name);
    return true;
}

ge::graphStatus CheckShape(const gert::TilingContext *context)
{
    const auto prevAttentionShapePtr = context->GetInputShape(PREV_ATTENTION_OUT_INPUT_INDEX);
    if (IsNullInput(context, prevAttentionShapePtr, "prev_attention_out shape")) {
        return ge::GRAPH_FAILED;
    }
    const auto prevSoftmaxMaxShapePtr = context->GetInputShape(PREV_SOFTMAX_MAX_INPUT_INDEX);
    if (IsNullInput(context, prevSoftmaxMaxShapePtr, "prev_softmax_max shape")) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape prevAttentionShape = prevAttentionShapePtr->GetStorageShape();
    const gert::Shape prevSoftmaxMaxShape = prevSoftmaxMaxShapePtr->GetStorageShape();

    if (prevAttentionShape.GetDimNum() != ATTENTION_DIM_NUM && prevAttentionShape.GetDimNum() != ATTENTION_TND_DIM_NUM) {
        OPS_LOG_E(context->GetNodeName(), "prev_attention_out must be BSND rank-4 or TND rank-3.");
        return ge::GRAPH_FAILED;
    }
    if (prevSoftmaxMaxShape.GetDimNum() != SOFTMAX_DIM_NUM) {
        OPS_LOG_E(context->GetNodeName(), "prev_softmax_max must be rank-2 [B*S, H].");
        return ge::GRAPH_FAILED;
    }

    const bool isTnd = prevAttentionShape.GetDimNum() == ATTENTION_TND_DIM_NUM;
    const int64_t batchSize = prevAttentionShape.GetDim(0);
    const int64_t seqSize = isTnd ? 1 : prevAttentionShape.GetDim(1);
    const int64_t headNum = isTnd ? prevAttentionShape.GetDim(1) : prevAttentionShape.GetDim(2);
    const int64_t headDim = isTnd ? prevAttentionShape.GetDim(2) : prevAttentionShape.GetDim(3);
    if (batchSize <= 0 || seqSize <= 0 || headNum <= 0 || headDim <= 0) {
        OPS_LOG_E(context->GetNodeName(), "attention dimensions must all be positive.");
        return ge::GRAPH_FAILED;
    }
    if (headDim % 16 != 0) {
        OPS_LOG_E(context->GetNodeName(), "headDim must be aligned to 16 for BSND DaAttentionMerge.");
        return ge::GRAPH_FAILED;
    }
    if (prevSoftmaxMaxShape.GetDim(0) != batchSize * seqSize || prevSoftmaxMaxShape.GetDim(1) != headNum) {
        OPS_LOG_E(context->GetNodeName(), "softmax state must have shape [B*S, H].");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t inputIdx : {PREV_SOFTMAX_SUM_INPUT_INDEX, CUR_SOFTMAX_MAX_INPUT_INDEX, CUR_SOFTMAX_SUM_INPUT_INDEX}) {
        const auto shapePtr = context->GetInputShape(inputIdx);
        if (IsNullInput(context, shapePtr, "softmax state shape")) {
            return ge::GRAPH_FAILED;
        }
        if (!SameShape(shapePtr->GetStorageShape(), prevSoftmaxMaxShape)) {
            OPS_LOG_E(context->GetNodeName(), "all softmax state tensors must have the same shape.");
            return ge::GRAPH_FAILED;
        }
    }

    for (uint32_t inputIdx : {CUR_ATTENTION_OUT_INPUT_INDEX}) {
        const auto shapePtr = context->GetInputShape(inputIdx);
        if (IsNullInput(context, shapePtr, "cur_attention_out shape")) {
            return ge::GRAPH_FAILED;
        }
        if (!SameShape(shapePtr->GetStorageShape(), prevAttentionShape)) {
            OPS_LOG_E(context->GetNodeName(), "attention tensors must have the same shape.");
            return ge::GRAPH_FAILED;
        }
    }

    const auto outShapePtr = context->GetOutputShape(ATTENTION_OUT_OUTPUT_INDEX);
    if (IsNullInput(context, outShapePtr, "attention_out shape")) {
        return ge::GRAPH_FAILED;
    }
    if (!SameShape(outShapePtr->GetStorageShape(), prevAttentionShape)) {
        OPS_LOG_E(context->GetNodeName(), "attention_out must have the same shape as prev_attention_out.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtype(const gert::TilingContext *context)
{
    const auto attentionDtype = context->GetInputDesc(PREV_ATTENTION_OUT_INPUT_INDEX)->GetDataType();
    if (attentionDtype != ge::DT_FLOAT16 && attentionDtype != ge::DT_BF16) {
        OPS_LOG_E(context->GetNodeName(), "attention dtype only supports float16 and bfloat16.");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(CUR_ATTENTION_OUT_INPUT_INDEX)->GetDataType() != attentionDtype ||
        context->GetOutputDesc(ATTENTION_OUT_OUTPUT_INDEX)->GetDataType() != attentionDtype) {
        OPS_LOG_E(context->GetNodeName(), "attention input/output dtypes must match.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t inputIdx : {PREV_SOFTMAX_MAX_INPUT_INDEX, PREV_SOFTMAX_SUM_INPUT_INDEX, CUR_SOFTMAX_MAX_INPUT_INDEX,
                              CUR_SOFTMAX_SUM_INPUT_INDEX}) {
        if (context->GetInputDesc(inputIdx)->GetDataType() != ge::DT_FLOAT) {
            OPS_LOG_E(context->GetNodeName(), "softmax state dtype must be float32.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingDaAttentionMerge(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShape(context) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "DaAttentionMerge shape check failed.");
        return ge::GRAPH_FAILED;
    }
    if (CheckDtype(context) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "DaAttentionMerge dtype check failed.");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape attentionShape = context->GetInputShape(PREV_ATTENTION_OUT_INPUT_INDEX)->GetStorageShape();
    const bool isTnd = attentionShape.GetDimNum() == ATTENTION_TND_DIM_NUM;
    const uint32_t batchSize = static_cast<uint32_t>(attentionShape.GetDim(0));
    const uint32_t seqSize = isTnd ? 1 : static_cast<uint32_t>(attentionShape.GetDim(1));
    const uint32_t headNum = static_cast<uint32_t>(isTnd ? attentionShape.GetDim(1) : attentionShape.GetDim(2));
    const uint32_t headDim = static_cast<uint32_t>(isTnd ? attentionShape.GetDim(2) : attentionShape.GetDim(3));
    const uint32_t totalRows = batchSize * seqSize * headNum;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t maxCoreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    const uint32_t usedCoreNum = std::max<uint32_t>(1, std::min(maxCoreNum, totalRows));
    const uint32_t rowsPerCore = (totalRows + usedCoreNum - 1) / usedCoreNum;

    DaAttentionMergeTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_seqSize(seqSize);
    tiling.set_headNum(headNum);
    tiling.set_headDim(headDim);
    tiling.set_headDimAlign(DaMergeAlign(headDim, static_cast<uint32_t>(16)));
    tiling.set_totalRows(totalRows);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowBlock(std::min(DEFAULT_ROW_BLOCK, rowsPerCore));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    const auto attentionDtype = context->GetInputDesc(PREV_ATTENTION_OUT_INPUT_INDEX)->GetDataType();
    context->SetTilingKey(attentionDtype == ge::DT_FLOAT16 ? DA_MERGE_TILING_KEY_FP16 : DA_MERGE_TILING_KEY_BF16);
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DaAttentionMerge).Tiling(TilingDaAttentionMerge);

} // namespace optiling
