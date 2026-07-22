/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr int64_t DECODE_SPARSE_COUNT = 2048;

static ge::graphStatus InferShapeLightningIndexerDecode(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("LightningIndexerDecode", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED);
    gert::Shape *outShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, outShape, return ge::GRAPH_FAILED);

    OPS_ERR_IF(queryShape->GetDimNum() != 3,
               OPS_LOG_E(context, "query must be TND [B, N1, D], rank should be 3 but got %zu.",
                         queryShape->GetDimNum()),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyShape->GetDimNum() != 4,
               OPS_LOG_E(context, "key must be PA_BSND [num_blocks, block_size, N2, D], rank should be 4 but got %zu.",
                         keyShape->GetDimNum()),
               return ge::GRAPH_FAILED);

    outShape->SetDimNum(3);
    outShape->SetDim(0, queryShape->GetDim(0));
    outShape->SetDim(1, keyShape->GetDim(2));
    outShape->SetDim(2, DECODE_SPARSE_COUNT);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexerDecode(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("LightningIndexerDecode", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexerDecode)
    .InferShape(InferShapeLightningIndexerDecode)
    .InferDataType(InferDataTypeLightningIndexerDecode);
} // namespace ops
