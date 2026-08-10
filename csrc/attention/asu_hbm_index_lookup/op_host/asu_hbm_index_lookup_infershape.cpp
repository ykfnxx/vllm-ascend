/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;

namespace ops {
namespace {
constexpr uint32_t QUERY_INDEX_INPUT = 5U;
constexpr uint32_t SLOT_OUT_OUTPUT = 0U;
constexpr uint32_t MISS_OUT_OUTPUT = 1U;
}  // namespace

static ge::graphStatus InferShapeAsuHbmIndexLookup(
    gert::InferShapeContext* context)
{
    OPS_ERR_IF(
        context == nullptr,
        OPS_LOG_E("AsuHbmIndexLookup", "InferShapeContext is nullptr."),
        return ge::GRAPH_FAILED);

    const gert::Shape* query_shape =
        context->GetInputShape(QUERY_INDEX_INPUT);
    OPS_LOG_E_IF_NULL(context, query_shape, return ge::GRAPH_FAILED);
    gert::Shape* out_shape =
        context->GetOutputShape(SLOT_OUT_OUTPUT);
    OPS_LOG_E_IF_NULL(context, out_shape, return ge::GRAPH_FAILED);

    out_shape->SetDimNum(query_shape->GetDimNum());
    for (size_t dim = 0; dim < query_shape->GetDimNum(); ++dim) {
        out_shape->SetDim(dim, query_shape->GetDim(dim));
    }
    gert::Shape* miss_shape =
        context->GetOutputShape(MISS_OUT_OUTPUT);
    OPS_LOG_E_IF_NULL(context, miss_shape, return ge::GRAPH_FAILED);
    miss_shape->SetDimNum(query_shape->GetDimNum());
    for (size_t dim = 0; dim < query_shape->GetDimNum(); ++dim) {
        miss_shape->SetDim(dim, query_shape->GetDim(dim));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAsuHbmIndexLookup(
    gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(
        context == nullptr,
        OPS_LOG_E("AsuHbmIndexLookup",
                  "InferDataTypeContext is nullptr."),
        return ge::GRAPH_FAILED);

    context->SetOutputDataType(SLOT_OUT_OUTPUT, ge::DT_INT32);
    context->SetOutputDataType(MISS_OUT_OUTPUT, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsuHbmIndexLookup)
    .InferShape(InferShapeAsuHbmIndexLookup)
    .InferDataType(InferDataTypeAsuHbmIndexLookup);
}  // namespace ops
