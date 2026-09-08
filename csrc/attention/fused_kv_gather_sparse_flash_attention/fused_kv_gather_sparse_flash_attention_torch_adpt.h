/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FUSED_KV_GATHER_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define FUSED_KV_GATHER_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H

namespace vllm_ascend {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_fused_kv_gather_sparse_flash_attention_output_tensor(
    const at::Tensor &query, const at::Tensor &key,
    const std::string &layout_query_str, const std::string &layout_kv_str, bool return_softmax_lse)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    constexpr int64_t DIM_4 = 4;

    TORCH_CHECK(layout_query_str == "BSND" || layout_query_str == "TND",
                "The layout of query only support BSND and TND, but got ",
                layout_query_str);
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }

    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query_str == "TND") {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1),
                       query.size(DIM_2)};
    } else {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ",
                    query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1),
                       query.size(DIM_2), query.size(DIM_3)};
    }

    at::Tensor attention_output =
        at::empty(output_size, query.options().dtype(query.dtype()));
    at::SmallVector<int64_t, SIZE> softmax_size;
    if (return_softmax_lse) {
        if (query.dim() == DIM_3) {
            const auto kv_head_num =
                layout_kv_str == "PA_BSND" ? key.size(DIM_2) : key.size(DIM_1);
            softmax_size = {
                kv_head_num,
                query.size(DIM_0),
                query.size(DIM_1) / kv_head_num,
            };
        } else {
            softmax_size = {
                query.size(DIM_0),
                key.size(DIM_2),
                query.size(DIM_1),
                query.size(DIM_2) / key.size(DIM_2),
            };
        }
    } else {
        softmax_size = {0};
    }

    at::Tensor softmax_max =
        at::empty(softmax_size, query.options().dtype(at::kFloat));
    at::Tensor softmax_sum =
        at::empty(softmax_size, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        attention_output, softmax_max, softmax_sum);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_kv_gather_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const at::Tensor &host_key, const at::Tensor &host_key_rope,
    const at::Tensor &host_block_table,
    const at::Tensor &query_pool_entries,
    const at::Tensor &hot_source_rows,
    const at::Tensor &route_plan,
    at::Tensor &install_staging_key,
    at::Tensor &install_staging_rope,
    int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    int64_t attention_mode, bool return_softmax_lse)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.");
    TORCH_CHECK(layout_query == "TND" && layout_kv == "PA_BSND" &&
                    sparse_block_size == 1,
                "FusedKvGatherSparseFlashAttention supports TND/PA_BSND and sparse_block_size=1 only");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.dim() == 4 &&
                    key.size(1) == 128 && key.size(2) == 1 && key.size(3) == 512,
                "Hot key/value must use [blocks, 128, 1, 512]");
    TORCH_CHECK(block_table.has_value() && query_rope.has_value() &&
                    key_rope.has_value(),
                "block_table, query_rope and key_rope are required by the fused operator");
    const at::Tensor &query_rope_tensor = query_rope.value();
    const at::Tensor &hot_rope = key_rope.value();
    const at::Tensor &hot_block_table = block_table.value();
    TORCH_CHECK(value.sizes() == key.sizes() &&
                    value.scalar_type() == key.scalar_type(),
                "Hot value must match Hot key shape and dtype");
    TORCH_CHECK(query_rope_tensor.dim() == 3 &&
                    query_rope_tensor.size(0) == query.size(0) &&
                    query_rope_tensor.size(1) == query.size(1) &&
                    query_rope_tensor.size(2) == 64 &&
                    query_rope_tensor.scalar_type() == query.scalar_type(),
                "Query RoPE must match query T/N and use head dimension 64");
    TORCH_CHECK(hot_rope.dim() == 4 && hot_rope.size(0) == key.size(0) &&
                    hot_rope.size(1) == key.size(1) && hot_rope.size(2) == 1 &&
                    hot_rope.size(3) == 64,
                "Hot RoPE must use [blocks, block_size, 1, 64]");
    TORCH_CHECK(host_key.dim() == 3 && host_key_rope.dim() == 3 &&
                    host_key.size(0) == host_key_rope.size(0) &&
                    host_key.size(1) == key.size(1) &&
                    host_key_rope.size(1) == key.size(1) &&
                    host_key.size(2) == 512 && host_key_rope.size(2) == 64,
                "Host KV/RoPE must use paired [blocks, block_size, 512/64] shapes");
    const at::Device device = query.device();
    const std::initializer_list<const at::Tensor *> allInputs = {
        &query, &query_rope_tensor, &key, &value, &sparse_indices,
        &hot_block_table, &hot_rope, &host_key, &host_key_rope,
        &host_block_table, &query_pool_entries,
        &hot_source_rows, &route_plan, &install_staging_key,
        &install_staging_rope};
    for (const at::Tensor *tensor : allInputs) {
        TORCH_CHECK(tensor->device() == device && tensor->is_contiguous(),
                    "all fused SFA inputs must be contiguous on one NPU");
    }
    TORCH_CHECK(host_key.scalar_type() == key.scalar_type() &&
                    host_key_rope.scalar_type() == hot_rope.scalar_type() &&
                    key.scalar_type() == hot_rope.scalar_type(),
                "Hot and Host KV/RoPE dtypes must match");
    for (const at::Tensor *tensor : {&hot_block_table, &host_block_table,
                                     &query_pool_entries, &hot_source_rows,
                                     &route_plan}) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "fused SFA route metadata must use int32");
    }
    TORCH_CHECK(hot_block_table.dim() == 2 &&
                    host_block_table.dim() == 2 &&
                    hot_block_table.size(0) > 0 &&
                    host_block_table.size(0) > 0,
                "Hot and Host block tables must be non-empty rank-two tensors");
    constexpr int64_t QUERY_WIDTH = 2 * 1024;
    // hot_source_rows/query_pool_entries are indexed by sequence (boIdx) in
    // the kernel; TND cu_seqlens carries B+1 entries, so B = size - 1.
    // Length-T per-query-row tensors remain accepted for compatibility.
    TORCH_CHECK(actual_seq_lengths_query.has_value(),
                "actual_seq_lengths_query is required by the fused operator");
    const int64_t num_sequences =
        actual_seq_lengths_query.value().size(0) - 1;
    TORCH_CHECK(route_plan.dim() == 2 && route_plan.size(0) == query.size(0) &&
                    route_plan.size(1) == QUERY_WIDTH &&
                    sparse_indices.dim() == 3 && sparse_indices.size(0) == query.size(0) &&
                    sparse_indices.size(1) == 1 &&
                    sparse_indices.size(2) == route_plan.size(1) &&
                    query_pool_entries.dim() == 1 &&
                    query_pool_entries.size(0) >= num_sequences &&
                    hot_source_rows.sizes() == query_pool_entries.sizes(),
                "route and row metadata must align with TND query rows");
    TORCH_CHECK(install_staging_key.dim() == 4 &&
                    install_staging_rope.dim() == 4 &&
                    install_staging_key.size(0) ==
                        install_staging_rope.size(0) &&
                    install_staging_key.size(1) == key.size(1) &&
                    install_staging_rope.size(1) == key.size(1) &&
                    install_staging_key.size(2) == 1 &&
                    install_staging_rope.size(2) == 1 &&
                    install_staging_key.size(3) == key.size(3) &&
                    install_staging_rope.size(3) == hot_rope.size(3) &&
                    install_staging_key.scalar_type() == key.scalar_type() &&
                    install_staging_rope.scalar_type() == hot_rope.scalar_type() &&
                    install_staging_key.numel() / key.size(3) >=
                        route_plan.numel(),
                "install staging must hold every route as paired [blocks, 128, 1, 512/64]");

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    auto sparse_flash_attention_output =
        construct_fused_kv_gather_sparse_flash_attention_output_tensor(
            query, key, layout_query_str, layout_kv_str, return_softmax_lse);
    at::Tensor attention_output = std::get<0>(sparse_flash_attention_output);
    at::Tensor softmax_max = std::get<1>(sparse_flash_attention_output);
    at::Tensor softmax_sum = std::get<2>(sparse_flash_attention_output);

    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnFusedKvGatherSparseFlashAttention,
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        host_key,
        host_key_rope,
        host_block_table,
        query_pool_entries,
        hot_source_rows,
        route_plan,
        install_staging_key,
        install_staging_rope,
        scale_value,
        sparse_block_size,
        layout_query_ptr,
        layout_kv_ptr,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        return_softmax_lse,
        attention_output,
        softmax_max,
        softmax_sum);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        attention_output, softmax_max, softmax_sum);
}
}  // namespace vllm_ascend

#endif  // FUSED_KV_GATHER_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
