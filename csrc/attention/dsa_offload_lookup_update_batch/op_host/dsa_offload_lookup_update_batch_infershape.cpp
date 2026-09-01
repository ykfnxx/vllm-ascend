/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include <cstddef>
#include <cstdint>

#include "register/op_impl_registry.h"

namespace ops {

namespace {
constexpr size_t kSemanticTopkInput = 7;
constexpr size_t kMappedIndicesOutput = 0;
constexpr size_t kMissMaskOutput = 1;
}

static ge::graphStatus InferShapeForDsaOffloadLookupUpdateBatch(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* query_shape =
        context->GetInputShape(kSemanticTopkInput);
    gert::Shape* slot_shape =
        context->GetOutputShape(kMappedIndicesOutput);
    gert::Shape* miss_shape =
        context->GetOutputShape(kMissMaskOutput);
    if (query_shape == nullptr || slot_shape == nullptr ||
        miss_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    slot_shape->SetDimNum(query_shape->GetDimNum());
    miss_shape->SetDimNum(query_shape->GetDimNum());
    for (size_t dim = 0; dim < query_shape->GetDimNum(); ++dim) {
        const int64_t value = query_shape->GetDim(dim);
        slot_shape->SetDim(dim, value);
        miss_shape->SetDim(dim, value);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDsaOffloadLookupUpdateBatch(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kMappedIndicesOutput, ge::DT_INT32);
    context->SetOutputDataType(kMissMaskOutput, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaOffloadLookupUpdateBatch)
    .InferShape(InferShapeForDsaOffloadLookupUpdateBatch)
    .InferDataType(InferDataTypeForDsaOffloadLookupUpdateBatch);

}  // namespace ops
