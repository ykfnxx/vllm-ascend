#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr size_t PREV_ATTENTION_OUT_INPUT_INDEX = 0;
constexpr size_t PREV_SOFTMAX_MAX_INPUT_INDEX = 1;
constexpr size_t PREV_SOFTMAX_SUM_INPUT_INDEX = 2;
constexpr size_t CUR_ATTENTION_OUT_INPUT_INDEX = 3;
constexpr size_t CUR_SOFTMAX_MAX_INPUT_INDEX = 4;
constexpr size_t CUR_SOFTMAX_SUM_INPUT_INDEX = 5;
constexpr size_t ATTENTION_OUT_OUTPUT_INDEX = 0;

static graphStatus InferShapeDaAttentionMerge(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("DaAttentionMerge", "InferShapeContext is nullptr"),
               return GRAPH_FAILED);
    const gert::Shape *prevAttentionShape = context->GetInputShape(PREV_ATTENTION_OUT_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, prevAttentionShape, return GRAPH_FAILED);
    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return GRAPH_FAILED);
    *attentionOutShape = *prevAttentionShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeDaAttentionMerge(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("DaAttentionMerge", "InferDataTypeContext is nullptr"),
               return GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(PREV_ATTENTION_OUT_INPUT_INDEX);
    context->SetOutputDataType(ATTENTION_OUT_OUTPUT_INDEX, inputDataType);
    return GRAPH_SUCCESS;
}

IMPL_OP(DaAttentionMerge).InferShape(InferShapeDaAttentionMerge).InferDataType(InferDataTypeDaAttentionMerge);
} // namespace ops
