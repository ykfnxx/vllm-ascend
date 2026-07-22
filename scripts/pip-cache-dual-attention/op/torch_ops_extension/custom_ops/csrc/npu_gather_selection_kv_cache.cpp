/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;

// step1, 工具函数，推导输出shape
at::Tensor construct_gather_selection_kv_cache_output_tensor(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache, const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status, const at::Tensor& selection_topk_indices
)
{
    c10::SmallVector<int64_t, SIZE> selection_kv_actual_seq_shape = {selection_kv_block_table.size(0)};
    at::Tensor out = at::empty(selection_kv_actual_seq_shape, selection_topk_indices.options());

    return out;
}

// step2, 为NPU设备实现前向接口
at::Tensor npu_gather_selection_kv_cache_npu(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size)
{
    // construct the output tensor
    at::Tensor selection_kv_actual_seq = construct_gather_selection_kv_cache_output_tensor(selection_k_rope,
                                                                                           selection_kv_cache,
                                                                                           selection_kv_block_table,
                                                                                           selection_kv_block_status,
                                                                                           selection_topk_indices);

    EXEC_NPU_CMD_V1(aclnnGatherSelectionKvCache,
                    selection_k_rope, selection_kv_cache, selection_kv_block_table,
                    selection_kv_block_status, selection_topk_indices, full_k_rope,
                    full_kv_cache, full_kv_block_table,
                    full_kv_actual_seq, full_q_actual_seq, selection_topk_block_size,
                    selection_kv_actual_seq);

    return selection_kv_actual_seq;
}

// step3, 为META设备实现前向接口
at::Tensor npu_gather_selection_kv_cache_meta(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size)
{
    // construct the output tensor
    at::Tensor outputs = construct_gather_selection_kv_cache_output_tensor(selection_k_rope,
                                                                           selection_kv_cache,
                                                                           selection_kv_block_table,
                                                                           selection_kv_block_status,
                                                                           selection_topk_indices);
    return outputs;
}

// step4, 实现函数化前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gather_selection_kv_cache_functional(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size)
{
    at::Tensor selection_kv_actual_seq = construct_gather_selection_kv_cache_output_tensor(selection_k_rope,
                                                                                           selection_kv_cache,
                                                                                           selection_kv_block_table,
                                                                                           selection_kv_block_status,
                                                                                           selection_topk_indices);
    // 保证接口对外为functional
    at::Tensor selection_k_rope_inplace = selection_k_rope.clone();
    at::Tensor selection_kv_cache_inplace = selection_kv_cache.clone();
    at::Tensor selection_kv_block_table_inplace = selection_kv_block_table.clone();
    at::Tensor selection_kv_block_status_inplace = selection_kv_block_status.clone();
    EXEC_NPU_CMD_V1(aclnnGatherSelectionKvCache, selection_k_rope_inplace, selection_kv_cache_inplace,
        selection_kv_block_table_inplace, selection_kv_block_status_inplace, selection_topk_indices,
        full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
        selection_topk_block_size, selection_kv_actual_seq);

    return std::tie(selection_kv_actual_seq, selection_k_rope_inplace, selection_kv_cache_inplace,
        selection_kv_block_table_inplace, selection_kv_block_status_inplace);
}

// step5, 为META设备实现函数化前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gather_selection_kv_cache_functional_meta(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size)
{
    // construct the output tensor
    at::Tensor output5 =
        construct_gather_selection_kv_cache_output_tensor(selection_k_rope,
                                                          selection_kv_cache,
                                                          selection_kv_block_table,
                                                          selection_kv_block_status,
                                                          selection_topk_indices);
    at::Tensor output1 = at::empty_like(selection_k_rope);
    at::Tensor output2 = at::empty_like(selection_kv_cache);
    at::Tensor output3 = at::empty_like(selection_kv_block_table);
    at::Tensor output4 = at::empty_like(selection_kv_block_status);

    return std::make_tuple(output5, output1, output2, output3, output4);
}


std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
npu_kv_select_out_npu(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    at::Tensor& hit_sparse_indices,
    at::Tensor& miss_topk_indices,
    at::Tensor& miss_insert_indices,
    at::Tensor& hit_actual_seq,
    at::Tensor& miss_actual_seq,
    at::Tensor& miss_count,
    at::Tensor& hit_count,
    at::Tensor& selection_status_empty,
    int64_t selection_topk_block_size)
{
    EXEC_NPU_CMD_V1(aclnnKVSelect,
                    selection_k_rope, selection_kv_cache, selection_kv_block_table,
                    selection_kv_block_status, selection_topk_indices, full_k_rope,
                    full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                    full_q_actual_seq, selection_topk_block_size, hit_sparse_indices,
                    miss_topk_indices, miss_insert_indices, hit_actual_seq,
                    miss_actual_seq, miss_count, hit_count, selection_status_empty);
    return std::tie(hit_sparse_indices, miss_topk_indices, miss_insert_indices,
                    hit_actual_seq, miss_actual_seq, miss_count, hit_count,
                    selection_status_empty);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
npu_kv_select_out_meta(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    at::Tensor& hit_sparse_indices,
    at::Tensor& miss_topk_indices,
    at::Tensor& miss_insert_indices,
    at::Tensor& hit_actual_seq,
    at::Tensor& miss_actual_seq,
    at::Tensor& miss_count,
    at::Tensor& hit_count,
    at::Tensor& selection_status_empty,
    int64_t selection_topk_block_size)
{
    return std::tie(hit_sparse_indices, miss_topk_indices, miss_insert_indices,
                    hit_actual_seq, miss_actual_seq, miss_count, hit_count,
                    selection_status_empty);
}

at::Tensor npu_kv_gather_out_npu(
    at::Tensor& selection_k_rope,
    at::Tensor& selection_kv_cache,
    at::Tensor& selection_kv_block_table,
    at::Tensor& selection_kv_block_status,
    const at::Tensor& miss_topk_indices,
    const at::Tensor& miss_insert_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    const at::Tensor& hit_actual_seq,
    const at::Tensor& miss_actual_seq,
    const at::Tensor& miss_count,
    const at::Tensor& hit_count,
    const at::Tensor& selection_status_empty,
    at::Tensor& selection_kv_actual_seq,
    int64_t selection_topk_block_size)
{
    EXEC_NPU_CMD_V1(aclnnKVGather,
                    selection_k_rope, selection_kv_cache, selection_kv_block_table,
                    selection_kv_block_status, miss_topk_indices, miss_insert_indices,
                    full_k_rope, full_kv_cache, full_kv_block_table,
                    full_kv_actual_seq, full_q_actual_seq, hit_actual_seq,
                    miss_actual_seq, miss_count, hit_count, selection_status_empty,
                    selection_topk_block_size, selection_kv_actual_seq);
    return selection_kv_actual_seq;
}

at::Tensor npu_kv_gather_out_meta(
    at::Tensor& selection_k_rope,
    at::Tensor& selection_kv_cache,
    at::Tensor& selection_kv_block_table,
    at::Tensor& selection_kv_block_status,
    const at::Tensor& miss_topk_indices,
    const at::Tensor& miss_insert_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    const at::Tensor& hit_actual_seq,
    const at::Tensor& miss_actual_seq,
    const at::Tensor& miss_count,
    const at::Tensor& hit_count,
    const at::Tensor& selection_status_empty,
    at::Tensor& selection_kv_actual_seq,
    int64_t selection_topk_block_size)
{
    return selection_kv_actual_seq;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_mock_kv_select_out_npu(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    const at::Tensor& hit_sparse_indices,
    const at::Tensor& miss_topk_indices,
    const at::Tensor& miss_insert_indices,
    const at::Tensor& hit_actual_seq,
    const at::Tensor& miss_actual_seq,
    const at::Tensor& miss_count,
    const at::Tensor& hit_count,
    const at::Tensor& selection_status_empty,
    int64_t selection_topk_block_size,
    int64_t mock_wait_us)
{
    EXEC_NPU_CMD_V1(aclnnMockKVSelect,
                    selection_k_rope, selection_kv_cache, selection_kv_block_table,
                    selection_kv_block_status, selection_topk_indices, full_k_rope,
                    full_kv_cache, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
                    selection_topk_block_size, mock_wait_us, hit_sparse_indices, miss_topk_indices,
                    miss_insert_indices, hit_actual_seq, miss_actual_seq, miss_count, hit_count,
                    selection_status_empty);

    return std::make_tuple(hit_sparse_indices, miss_topk_indices, miss_insert_indices, hit_actual_seq,
                           miss_actual_seq, miss_count, hit_count, selection_status_empty);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_mock_kv_select_out_meta(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    const at::Tensor& hit_sparse_indices,
    const at::Tensor& miss_topk_indices,
    const at::Tensor& miss_insert_indices,
    const at::Tensor& hit_actual_seq,
    const at::Tensor& miss_actual_seq,
    const at::Tensor& miss_count,
    const at::Tensor& hit_count,
    const at::Tensor& selection_status_empty,
    int64_t selection_topk_block_size,
    int64_t mock_wait_us)
{
    return std::make_tuple(hit_sparse_indices, miss_topk_indices, miss_insert_indices, hit_actual_seq,
                           miss_actual_seq, miss_count, hit_count, selection_status_empty);
}

}

// step6, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_gather_selection_kv_cache", &custom::npu_gather_selection_kv_cache_npu);
    m.impl("npu_gather_selection_kv_cache_functional", &custom::npu_gather_selection_kv_cache_functional);
    m.impl("npu_kv_select_out", &custom::npu_kv_select_out_npu);
    m.impl("npu_kv_gather_out", &custom::npu_kv_gather_out_npu);
    m.impl("npu_mock_kv_select_out", &custom::npu_mock_kv_select_out_npu);
}

// step7, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_gather_selection_kv_cache", &custom::npu_gather_selection_kv_cache_meta);
    m.impl("npu_gather_selection_kv_cache_functional", &custom::npu_gather_selection_kv_cache_functional_meta);
    m.impl("npu_kv_select_out", &custom::npu_kv_select_out_meta);
    m.impl("npu_kv_gather_out", &custom::npu_kv_gather_out_meta);
    m.impl("npu_mock_kv_select_out", &custom::npu_mock_kv_select_out_meta);
}
