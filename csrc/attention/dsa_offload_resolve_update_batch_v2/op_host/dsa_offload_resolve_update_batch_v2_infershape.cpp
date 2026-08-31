/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include <cstddef>
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kSemanticTopkInput = 7;
}

static ge::graphStatus InferShapeDsaOffloadResolveUpdateBatchV2(
    gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(kSemanticTopkInput);
    if (input == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t output_index = 0; output_index < 2; ++output_index) {
        gert::Shape* output = context->GetOutputShape(output_index);
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeDsaOffloadResolveUpdateBatchV2(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaOffloadResolveUpdateBatchV2)
    .InferShape(InferShapeDsaOffloadResolveUpdateBatchV2)
    .InferDataType(InferDtypeDsaOffloadResolveUpdateBatchV2);
}  // namespace ops
