/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include <cstddef>
#include <cstdint>

#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kQueryIndexInput = 6;
constexpr size_t kDestinationSlotsOutput = 0;
constexpr size_t kMissMaskOutput = 1;
}

static ge::graphStatus InferShapeForDsaSparseTurboFusedPrefetchLookupUpdateBatch(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* query_shape =
        context->GetInputShape(kQueryIndexInput);
    gert::Shape* destination_shape =
        context->GetOutputShape(kDestinationSlotsOutput);
    gert::Shape* miss_mask_shape =
        context->GetOutputShape(kMissMaskOutput);
    if (query_shape == nullptr || destination_shape == nullptr ||
        miss_mask_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    destination_shape->SetDimNum(query_shape->GetDimNum());
    miss_mask_shape->SetDimNum(query_shape->GetDimNum());
    for (size_t dim = 0; dim < query_shape->GetDimNum(); ++dim) {
        const int64_t value = query_shape->GetDim(dim);
        destination_shape->SetDim(dim, value);
        miss_mask_shape->SetDim(dim, value);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDsaSparseTurboFusedPrefetchLookupUpdateBatch(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kDestinationSlotsOutput, ge::DT_INT32);
    context->SetOutputDataType(kMissMaskOutput, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaSparseTurboFusedPrefetchLookupUpdateBatch)
    .InferShape(InferShapeForDsaSparseTurboFusedPrefetchLookupUpdateBatch)
    .InferDataType(InferDataTypeForDsaSparseTurboFusedPrefetchLookupUpdateBatch);

}  // namespace ops
