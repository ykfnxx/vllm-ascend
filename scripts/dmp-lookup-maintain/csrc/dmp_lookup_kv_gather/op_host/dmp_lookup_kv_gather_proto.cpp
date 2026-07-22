#include <register/op_impl_registry.h>

#include "error/ops_error.h"

namespace ops {
namespace {
constexpr uint32_t SELECTION_K_ROPE_INPUT = 0U;
constexpr uint32_t SELECTION_KV_CACHE_INPUT = 1U;
constexpr uint32_t SEQ_LENS_INPUT = 11U;
constexpr uint32_t SELECTION_K_ROPE_OUTPUT = 0U;
constexpr uint32_t SELECTION_KV_CACHE_OUTPUT = 1U;
constexpr uint32_t COPIED_COUNT_OUTPUT = 2U;
}  // namespace

static ge::graphStatus InferShapeDmpLookupKvGather(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("DmpLookupKvGather", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    *context->GetOutputShape(SELECTION_K_ROPE_OUTPUT) =
        *context->GetInputShape(SELECTION_K_ROPE_INPUT);
    *context->GetOutputShape(SELECTION_KV_CACHE_OUTPUT) =
        *context->GetInputShape(SELECTION_KV_CACHE_INPUT);
    *context->GetOutputShape(COPIED_COUNT_OUTPUT) =
        *context->GetInputShape(SEQ_LENS_INPUT);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDmpLookupKvGather(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("DmpLookupKvGather", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(
        SELECTION_K_ROPE_OUTPUT, context->GetInputDataType(SELECTION_K_ROPE_INPUT));
    context->SetOutputDataType(
        SELECTION_KV_CACHE_OUTPUT, context->GetInputDataType(SELECTION_KV_CACHE_INPUT));
    context->SetOutputDataType(COPIED_COUNT_OUTPUT, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DmpLookupKvGather)
    .InferShape(InferShapeDmpLookupKvGather)
    .InferDataType(InferDataTypeDmpLookupKvGather);
}  // namespace ops
