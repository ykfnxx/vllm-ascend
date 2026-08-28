/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeForDsaSparseLookupUpdate(
    gert::InferShapeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDsaSparseLookupUpdate(
    gert::InferDataTypeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaSparseLookupUpdate)
    .InferShape(InferShapeForDsaSparseLookupUpdate)
    .InferDataType(InferDataTypeForDsaSparseLookupUpdate);

}  // namespace ops
