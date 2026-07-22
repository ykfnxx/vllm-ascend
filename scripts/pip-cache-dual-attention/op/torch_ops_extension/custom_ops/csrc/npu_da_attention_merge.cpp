#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

at::Tensor construct_da_attention_merge_output_tensor(const at::Tensor &prev_attention_out)
{
    return at::empty(prev_attention_out.sizes(), prev_attention_out.options().dtype(prev_attention_out.dtype()));
}

at::Tensor npu_da_attention_merge_npu(
    const at::Tensor &prev_attention_out,
    const at::Tensor &prev_softmax_max,
    const at::Tensor &prev_softmax_sum,
    const at::Tensor &cur_attention_out,
    const at::Tensor &cur_softmax_max,
    const at::Tensor &cur_softmax_sum)
{
    at::Tensor output = construct_da_attention_merge_output_tensor(prev_attention_out);
    EXEC_NPU_CMD_V1(aclnnDaAttentionMerge,
                    prev_attention_out, prev_softmax_max, prev_softmax_sum,
                    cur_attention_out, cur_softmax_max, cur_softmax_sum,
                    output);
    return output;
}

at::Tensor npu_da_attention_merge_meta(
    const at::Tensor &prev_attention_out,
    const at::Tensor &prev_softmax_max,
    const at::Tensor &prev_softmax_sum,
    const at::Tensor &cur_attention_out,
    const at::Tensor &cur_softmax_max,
    const at::Tensor &cur_softmax_sum)
{
    (void)prev_softmax_max;
    (void)prev_softmax_sum;
    (void)cur_attention_out;
    (void)cur_softmax_max;
    (void)cur_softmax_sum;
    return construct_da_attention_merge_output_tensor(prev_attention_out);
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_da_attention_merge", &custom::npu_da_attention_merge_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_da_attention_merge", &custom::npu_da_attention_merge_meta);
}
