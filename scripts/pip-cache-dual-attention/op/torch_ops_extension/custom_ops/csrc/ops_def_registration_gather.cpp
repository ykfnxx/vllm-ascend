/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * TORCH_LIBRARY schemas for gather-side custom ops (used when CUSTOM_OPS_GATHER_ONLY=1).
 */

#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(custom, m) {
    m.def("npu_gather_selection_kv_cache(Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache, Tensor(c!) "
          "selection_kv_block_table, Tensor(d!) selection_kv_block_status, Tensor selection_topk_indices, Tensor full_k_rope, "
          "Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, Tensor full_q_actual_seq, *, "
          "int selection_topk_block_size=64) -> Tensor");
    m.def("npu_gather_selection_kv_cache_functional(Tensor selection_k_rope, Tensor selection_kv_cache, "
          "Tensor selection_kv_block_table, Tensor selection_kv_block_status, Tensor selection_topk_indices, "
          "Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, "
          "Tensor full_q_actual_seq, *, int selection_topk_block_size=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_kv_select_out(Tensor selection_k_rope, Tensor selection_kv_cache, "
          "Tensor selection_kv_block_table, Tensor selection_kv_block_status, Tensor selection_topk_indices, "
          "Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, "
          "Tensor full_q_actual_seq, Tensor(a!) hit_sparse_indices, Tensor(b!) miss_topk_indices, "
          "Tensor(c!) miss_insert_indices, Tensor(d!) hit_actual_seq, Tensor(e!) miss_actual_seq, "
          "Tensor(f!) miss_count, Tensor(g!) hit_count, Tensor(h!) selection_status_empty, "
          "*, int selection_topk_block_size=64) -> "
          "(Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!), Tensor(h!))");
    m.def("npu_mock_kv_select_out(Tensor selection_k_rope, Tensor selection_kv_cache, "
          "Tensor selection_kv_block_table, Tensor selection_kv_block_status, Tensor selection_topk_indices, "
          "Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, "
          "Tensor full_q_actual_seq, Tensor(a!) hit_sparse_indices, Tensor(b!) miss_topk_indices, "
          "Tensor(c!) miss_insert_indices, Tensor(d!) hit_actual_seq, Tensor(e!) miss_actual_seq, "
          "Tensor(f!) miss_count, Tensor(g!) hit_count, Tensor(h!) selection_status_empty, "
          "*, int selection_topk_block_size=64, int mock_wait_us=25) -> "
          "(Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!), Tensor(h!))");
    m.def("npu_kv_gather_out(Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache, "
          "Tensor(c!) selection_kv_block_table, Tensor(d!) selection_kv_block_status, "
          "Tensor miss_topk_indices, Tensor miss_insert_indices, Tensor full_k_rope, Tensor full_kv_cache, "
          "Tensor full_kv_block_table, Tensor full_kv_actual_seq, Tensor full_q_actual_seq, "
          "Tensor hit_actual_seq, Tensor miss_actual_seq, Tensor miss_count, Tensor hit_count, "
          "Tensor selection_status_empty, Tensor(e!) selection_kv_actual_seq, "
          "*, int selection_topk_block_size=64) -> Tensor(e!)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
