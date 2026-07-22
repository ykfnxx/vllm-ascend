/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Sparse-flash-attention-only TORCH_LIBRARY schema (CUSTOM_OPS_SFA_ONLY=1).
 */

#include <torch/extension.h>
#include <torch/library.h>

#if defined(CUSTOM_OPS_SFA_ONLY)
TORCH_LIBRARY(custom, m) {
#else
TORCH_LIBRARY_FRAGMENT(custom, m) {
#endif
    m.def("npu_dmp_sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, *, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? softmax_max_out=None, Tensor? softmax_sum_out=None, Tensor? prior_softmax_max=None, Tensor? prior_softmax_sum=None, Tensor? prior_attention_out=None, str layout_query='BSND', str layout_kv='BSND', int sparse_mode=3) -> Tensor");
}

#if defined(CUSTOM_OPS_SFA_ONLY)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
#endif
