#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("npu_da_attention_merge(Tensor prev_attention_out, Tensor prev_softmax_max, Tensor prev_softmax_sum, "
          "Tensor cur_attention_out, Tensor cur_softmax_max, Tensor cur_softmax_sum) -> Tensor");
}
