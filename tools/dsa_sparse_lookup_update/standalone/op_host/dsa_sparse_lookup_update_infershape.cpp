/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include <cstddef>
#include <cstdint>

#include "register/op_impl_registry.h"

namespace ops {

namespace {
constexpr size_t kQueryIndexInput = 5;
constexpr size_t kSlotOutOutput = 0;
constexpr size_t kMissOutOutput = 1;
}

static ge::graphStatus InferShapeForDsaSparseLookupUpdate(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* query_shape =
        context->GetInputShape(kQueryIndexInput);
    gert::Shape* slot_shape =
        context->GetOutputShape(kSlotOutOutput);
    gert::Shape* miss_shape =
        context->GetOutputShape(kMissOutOutput);
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

static ge::graphStatus InferDataTypeForDsaSparseLookupUpdate(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kSlotOutOutput, ge::DT_INT32);
    context->SetOutputDataType(kMissOutOutput, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaSparseLookupUpdate)
    .InferShape(InferShapeForDsaSparseLookupUpdate)
    .InferDataType(InferDataTypeForDsaSparseLookupUpdate);

}  // namespace ops
