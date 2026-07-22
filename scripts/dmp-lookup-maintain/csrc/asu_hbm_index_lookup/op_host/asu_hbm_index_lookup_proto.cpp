#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;

namespace ops {
namespace {
constexpr uint32_t QUERY_INDEX_INPUT = 5U;
constexpr uint32_t SLOT_TO_INDEX_INPUT = 1U;
constexpr uint32_t SLOT_OUT_OUTPUT = 0U;
constexpr uint32_t MISS_OUT_OUTPUT = 1U;
constexpr uint32_t HIT_SPARSE_OUTPUT = 2U;
constexpr uint32_t MISS_SPARSE_OUTPUT = 3U;
constexpr uint32_t RESIDENT_TOKEN_IDS_OUTPUT = 4U;
}  // namespace

static ge::graphStatus InferShapeAsuHbmIndexLookup(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("AsuHbmIndexLookup", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);

    const gert::Shape* query_shape = context->GetInputShape(QUERY_INDEX_INPUT);
    OPS_LOG_E_IF_NULL(context, query_shape, return ge::GRAPH_FAILED);
    gert::Shape* out_shape = context->GetOutputShape(SLOT_OUT_OUTPUT);
    OPS_LOG_E_IF_NULL(context, out_shape, return ge::GRAPH_FAILED);

    out_shape->SetDimNum(query_shape->GetDimNum());
    for (size_t dim = 0; dim < query_shape->GetDimNum(); ++dim) {
        out_shape->SetDim(dim, query_shape->GetDim(dim));
    }
    for (uint32_t output_index = MISS_OUT_OUTPUT;
         output_index <= MISS_SPARSE_OUTPUT;
         ++output_index) {
        gert::Shape* output_shape = context->GetOutputShape(output_index);
        OPS_LOG_E_IF_NULL(context, output_shape, return ge::GRAPH_FAILED);
        output_shape->SetDimNum(query_shape->GetDimNum());
        for (size_t dim = 0; dim < query_shape->GetDimNum(); ++dim) {
            output_shape->SetDim(dim, query_shape->GetDim(dim));
        }
    }
    const gert::Shape* resident_shape = context->GetInputShape(SLOT_TO_INDEX_INPUT);
    OPS_LOG_E_IF_NULL(context, resident_shape, return ge::GRAPH_FAILED);
    gert::Shape* resident_output = context->GetOutputShape(RESIDENT_TOKEN_IDS_OUTPUT);
    OPS_LOG_E_IF_NULL(context, resident_output, return ge::GRAPH_FAILED);
    resident_output->SetDimNum(2);
    resident_output->SetDim(0, query_shape->GetDim(0));
    resident_output->SetDim(1, resident_shape->GetDim(1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAsuHbmIndexLookup(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("AsuHbmIndexLookup", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);

    context->SetOutputDataType(SLOT_OUT_OUTPUT, ge::DT_INT32);
    context->SetOutputDataType(MISS_OUT_OUTPUT, ge::DT_INT32);
    context->SetOutputDataType(HIT_SPARSE_OUTPUT, ge::DT_INT32);
    context->SetOutputDataType(MISS_SPARSE_OUTPUT, ge::DT_INT32);
    context->SetOutputDataType(RESIDENT_TOKEN_IDS_OUTPUT, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsuHbmIndexLookup)
    .InferShape(InferShapeAsuHbmIndexLookup)
    .InferDataType(InferDataTypeAsuHbmIndexLookup);
}  // namespace ops
