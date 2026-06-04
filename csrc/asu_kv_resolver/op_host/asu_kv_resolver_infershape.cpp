#include <register/op_impl_registry.h>

namespace ops {
constexpr size_t ORIGINAL_TOPK_INDICES_INDEX = 0;
constexpr size_t RESOLVED_KV_SLOTS_INDEX = 0;

static ge::graphStatus InferShapeAsuKvResolver(gert::InferShapeContext* context)
{
    const gert::Shape* topkShape =
        context->GetInputShape(ORIGINAL_TOPK_INDICES_INDEX);
    gert::Shape* outputShape = context->GetOutputShape(RESOLVED_KV_SLOTS_INDEX);
    *outputShape = *topkShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAsuKvResolver(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(RESOLVED_KV_SLOTS_INDEX, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsuKvResolver)
    .InferShape(InferShapeAsuKvResolver)
    .InferDataType(InferDataTypeAsuKvResolver);
} // namespace ops
