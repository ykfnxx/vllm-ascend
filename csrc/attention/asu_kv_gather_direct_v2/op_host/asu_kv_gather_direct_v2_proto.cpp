/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeAsuKvGatherDirectV2(
    gert::InferShapeContext* context)
{
    for (uint32_t index = 0; index < 2; ++index) {
        const gert::Shape* input = context->GetInputShape(index);
        gert::Shape* output = context->GetOutputShape(index);
        if (input == nullptr || output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeAsuKvGatherDirectV2(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AsuKvGatherDirectV2)
    .InferShape(InferShapeAsuKvGatherDirectV2)
    .InferDataType(InferDtypeAsuKvGatherDirectV2);
}  // namespace ops
