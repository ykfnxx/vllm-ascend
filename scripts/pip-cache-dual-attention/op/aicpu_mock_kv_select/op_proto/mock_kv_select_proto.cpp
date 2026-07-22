#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

namespace ops {
constexpr int32_t INDEX_INPUT_SELECTION_KV_BLOCK_TABLE = 2;
constexpr int32_t INDEX_INPUT_SELECTION_TOPK_INDICES = 4;
constexpr int32_t INDEX_OUTPUT_HIT_SPARSE_INDICES = 0;
constexpr int32_t INDEX_OUTPUT_MISS_TOPK_INDICES = 1;
constexpr int32_t INDEX_OUTPUT_MISS_INSERT_INDICES = 2;
constexpr int32_t INDEX_OUTPUT_HIT_ACTUAL_SEQ = 3;
constexpr int32_t INDEX_OUTPUT_MISS_ACTUAL_SEQ = 4;
constexpr int32_t INDEX_OUTPUT_MISS_COUNT = 5;
constexpr int32_t INDEX_OUTPUT_HIT_COUNT = 6;
constexpr int32_t INDEX_OUTPUT_SELECTION_STATUS_EMPTY = 7;

static ge::graphStatus InferShape4MockKVSelect(gert::InferShapeContext* context)
{
    const gert::Shape* topkShape = context->GetInputShape(INDEX_INPUT_SELECTION_TOPK_INDICES);
    if (topkShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto hitSparseShape = context->GetOutputShape(INDEX_OUTPUT_HIT_SPARSE_INDICES);
    auto missTopkShape = context->GetOutputShape(INDEX_OUTPUT_MISS_TOPK_INDICES);
    auto missInsertShape = context->GetOutputShape(INDEX_OUTPUT_MISS_INSERT_INDICES);
    if (hitSparseShape == nullptr || missTopkShape == nullptr || missInsertShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *hitSparseShape = *topkShape;
    *missTopkShape = *topkShape;
    *missInsertShape = *topkShape;

    const gert::Shape* blockTableShape = context->GetInputShape(INDEX_INPUT_SELECTION_KV_BLOCK_TABLE);
    if (blockTableShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto hitActualSeqShape = context->GetOutputShape(INDEX_OUTPUT_HIT_ACTUAL_SEQ);
    auto missActualSeqShape = context->GetOutputShape(INDEX_OUTPUT_MISS_ACTUAL_SEQ);
    auto missCountShape = context->GetOutputShape(INDEX_OUTPUT_MISS_COUNT);
    auto hitCountShape = context->GetOutputShape(INDEX_OUTPUT_HIT_COUNT);
    auto statusEmptyShape = context->GetOutputShape(INDEX_OUTPUT_SELECTION_STATUS_EMPTY);
    if (hitActualSeqShape == nullptr || missActualSeqShape == nullptr ||
        missCountShape == nullptr || hitCountShape == nullptr || statusEmptyShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *hitActualSeqShape = *blockTableShape;
    hitActualSeqShape->SetDimNum(static_cast<int64_t>(blockTableShape->GetDimNum()) - 1);
    *missActualSeqShape = *hitActualSeqShape;
    *missCountShape = *hitActualSeqShape;
    *hitCountShape = *hitActualSeqShape;
    *statusEmptyShape = *hitActualSeqShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4MockKVSelect(gert::InferDataTypeContext* context)
{
    for (int32_t outputIdx = INDEX_OUTPUT_HIT_SPARSE_INDICES;
         outputIdx <= INDEX_OUTPUT_SELECTION_STATUS_EMPTY; outputIdx++) {
        context->SetOutputDataType(outputIdx, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MockKVSelect)
    .InferShape(InferShape4MockKVSelect)
    .InferDataType(InferDtype4MockKVSelect);
}  // namespace ops
