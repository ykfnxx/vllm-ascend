import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.custom.npu_da_attention_merge.default)
def convert_npu_da_attention_merge(
    prev_attention_out: Tensor,
    prev_softmax_max: Tensor,
    prev_softmax_sum: Tensor,
    cur_attention_out: Tensor,
    cur_softmax_max: Tensor,
    cur_softmax_sum: Tensor,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "DaAttentionMerge",
        inputs={
            "prev_attention_out": prev_attention_out,
            "prev_softmax_max": prev_softmax_max,
            "prev_softmax_sum": prev_softmax_sum,
            "cur_attention_out": cur_attention_out,
            "cur_softmax_max": cur_softmax_max,
            "cur_softmax_sum": cur_softmax_sum,
        },
        attrs={},
        outputs=["attention_out"],
    )
