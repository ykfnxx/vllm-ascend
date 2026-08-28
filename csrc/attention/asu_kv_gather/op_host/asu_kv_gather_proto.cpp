#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

namespace ops {
namespace {
constexpr uint32_t DESTINATION_KV_INPUT = 0U;
constexpr uint32_t DESTINATION_ROPE_INPUT = 1U;
constexpr uint32_t DESTINATION_KV_OUTPUT = 0U;
constexpr uint32_t DESTINATION_ROPE_OUTPUT = 1U;
}  // namespace

static ge::graphStatus InferShapeAsuKvGather(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("AsuKvGather", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < 2U; ++i) {
        const uint32_t input_index =
            i == 0U ? DESTINATION_KV_INPUT : DESTINATION_ROPE_INPUT;
        const uint32_t output_index =
            i == 0U ? DESTINATION_KV_OUTPUT : DESTINATION_ROPE_OUTPUT;
        const gert::Shape* input_shape = context->GetInputShape(input_index);
        OPS_LOG_E_IF_NULL(context, input_shape, return ge::GRAPH_FAILED);
        gert::Shape* output_shape = context->GetOutputShape(output_index);
        OPS_LOG_E_IF_NULL(context, output_shape, return ge::GRAPH_FAILED);
        output_shape->SetDimNum(input_shape->GetDimNum());
        for (size_t dim = 0; dim < input_shape->GetDimNum(); ++dim) {
            output_shape->SetDim(dim, input_shape->GetDim(dim));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAsuKvGather(
    gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("AsuKvGather", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(
        DESTINATION_KV_OUTPUT,
        context->GetInputDataType(DESTINATION_KV_INPUT));
    context->SetOutputDataType(
        DESTINATION_ROPE_OUTPUT,
        context->GetInputDataType(DESTINATION_ROPE_INPUT));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsuKvGather)
    .InferShape(InferShapeAsuKvGather)
    .InferDataType(InferDataTypeAsuKvGather);
}  // namespace ops
