#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;

namespace ops {
namespace {
constexpr uint32_t HIDDEN_STATES_INPUT = 0U;
constexpr uint32_t Q_LI_OUTPUT = 0U;
constexpr uint32_t WEIGHTS_OUTPUT = 1U;
constexpr uint32_t ATTR_Q_LORA_RANK = 0U;
constexpr uint32_t ATTR_N_HEAD = 1U;
constexpr uint32_t ATTR_HEAD_DIM = 2U;
}  // namespace

static ge::graphStatus InferShapePrefetchQliFusion(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("PrefetchQliFusion", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);

    const gert::Shape* hidden_shape = context->GetInputShape(HIDDEN_STATES_INPUT);
    OPS_LOG_E_IF_NULL(context, hidden_shape, return ge::GRAPH_FAILED);
    gert::Shape* out_shape = context->GetOutputShape(Q_LI_OUTPUT);
    OPS_LOG_E_IF_NULL(context, out_shape, return ge::GRAPH_FAILED);
    gert::Shape* weights_shape = context->GetOutputShape(WEIGHTS_OUTPUT);
    OPS_LOG_E_IF_NULL(context, weights_shape, return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* n_head_attr = attrs->GetAttrPointer<int64_t>(ATTR_N_HEAD);
    OPS_LOG_E_IF_NULL(context, n_head_attr, return ge::GRAPH_FAILED);
    const int64_t* head_dim_attr = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_DIM);
    OPS_LOG_E_IF_NULL(context, head_dim_attr, return ge::GRAPH_FAILED);

    OPS_ERR_IF(hidden_shape->GetDimNum() < 1,
               OPS_LOG_E(context->GetNodeName(), "hidden_states must have rank >= 1."),
               return ge::GRAPH_FAILED);
    int64_t token_num = hidden_shape->GetDim(0);

    out_shape->SetDimNum(3);
    out_shape->SetDim(0, token_num);
    out_shape->SetDim(1, *n_head_attr);
    out_shape->SetDim(2, *head_dim_attr);

    weights_shape->SetDimNum(2);
    weights_shape->SetDim(0, token_num);
    weights_shape->SetDim(1, *n_head_attr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePrefetchQliFusion(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("PrefetchQliFusion", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(Q_LI_OUTPUT, ge::DT_BF16);
    context->SetOutputDataType(WEIGHTS_OUTPUT, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PrefetchQliFusion)
    .InferShape(InferShapePrefetchQliFusion)
    .InferDataType(InferDataTypePrefetchQliFusion);
}  // namespace ops
