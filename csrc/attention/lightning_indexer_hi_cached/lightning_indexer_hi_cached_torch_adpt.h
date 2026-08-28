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
#ifndef LIGHTNING_INDEXER_HI_CACHED_TORCH_ADPT_H
#define LIGHTNING_INDEXER_HI_CACHED_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <string>

namespace vllm_ascend {

inline at::Tensor npu_lightning_indexer_hi_cached(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &weights,
    const at::Tensor &key_mean,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t hi_block_size,
    int64_t hi_block_num,
    int64_t sink,
    int64_t recent,
    c10::string_view block_pooling_mode)
{
    TORCH_CHECK(query.numel() > 0, "Query is empty.");
    TORCH_CHECK(key.numel() > 0, "Key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Weights is empty.");
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "Query and key must share the same dtype.");
    TORCH_CHECK(key.scalar_type() == weights.scalar_type(), "Key and weights must share the same dtype.");
    for (size_t i = 0; i < query.sizes().size(); ++i) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater than 0, but shape[", i, "] is ",
                    query.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse_count should be greater than 0, but now is ", sparse_count);
    TORCH_CHECK(hi_block_size > 0, "hi_block_size should be greater than 0, but now is ", hi_block_size);
    TORCH_CHECK(hi_block_num > 0,
                "hi_block_num should be greater than 0, but now is ", hi_block_num);
    TORCH_CHECK(hi_block_num <= sparse_count,
                "hi_block_num should be less than or equal to sparse_count, but now is ",
                hi_block_num, " while sparse_count is ", sparse_count);
    TORCH_CHECK(sink >= 0, "sink should be greater than or equal to 0, but now is ", sink);
    TORCH_CHECK(recent >= 1,
                "npu_lightning_indexer_hi_cached requires recent >= 1 because partial tail blocks "
                "are selected through recent instead of key_mean, but now is ",
                recent);
    TORCH_CHECK(sink + recent <= hi_block_num,
                "sink + recent should be less than or equal to hi_block_num, but now sink is ",
                sink, ", recent is ", recent, ", hi_block_num is ", hi_block_num);

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);
    std::string pooling_mode_str = std::string(block_pooling_mode);
    TORCH_CHECK(query_layout_str == "BSND" || query_layout_str == "TND",
                "layout_query only supports BSND or TND, but now is ", query_layout_str);
    TORCH_CHECK(key_layout_str == "PA_BSND",
                "npu_lightning_indexer_hi_cached currently only supports layout_key='PA_BSND', but now is ", key_layout_str);
    TORCH_CHECK(pooling_mode_str == "mean",
                "npu_lightning_indexer_hi_cached currently only supports block_pooling_mode='mean', but now is ",
                pooling_mode_str);
    TORCH_CHECK(actual_seq_lengths_key.has_value(),
                "actual_seq_lengths_key must be provided for npu_lightning_indexer_hi_cached.");
    TORCH_CHECK(block_table.has_value(), "block_table must be provided for npu_lightning_indexer_hi_cached.");
    if (query_layout_str == "TND") {
        TORCH_CHECK(actual_seq_lengths_query.has_value(),
                    "actual_seq_lengths_query must be provided when layout_query='TND'.");
    }
    TORCH_CHECK(key.dim() == 4, "npu_lightning_indexer_hi_cached expects key to be 4-D when layout_key='PA_BSND'.");
    TORCH_CHECK(key.size(1) == hi_block_size,
                "npu_lightning_indexer_hi_cached requires hi_block_size to equal physical page block size, but got "
                "hi_block_size=", hi_block_size, ", block_size=", key.size(1));
    TORCH_CHECK((512 % hi_block_size) == 0,
                "npu_lightning_indexer_hi_cached requires hi_block_size to divide 512, but got ", hi_block_size);
    TORCH_CHECK(key_mean.scalar_type() == key.scalar_type(),
                "key_mean dtype must match key dtype.");
    TORCH_CHECK(key_mean.dim() == 4,
                "key_mean must be 4-D: [kv_blocks, 1, kv_heads, head_dim].");
    TORCH_CHECK(key_mean.size(0) == key.size(0),
                "key_mean dim0 must equal physical key block count.");
    TORCH_CHECK(key_mean.size(1) == 1,
                "key_mean dim1 must be 1 for one mean vector per physical key block.");
    TORCH_CHECK(key_mean.size(2) == key.size(2),
                "key_mean dim2 must equal kv_heads.");
    TORCH_CHECK(key_mean.size(3) == key.size(3),
                "key_mean dim3 must match key head_dim.");

    constexpr int32_t SIZE = 8;
    constexpr int32_t DIM_0 = 0;
    constexpr int32_t DIM_1 = 1;
    constexpr int32_t DIM_2 = 2;

    at::SmallVector<int64_t, SIZE> output_size;
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int64_t n_dim_index = DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }

    at::Tensor output = at::empty(output_size, query.options().dtype(at::kInt));
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());
    char *pooling_mode_ptr = const_cast<char *>(pooling_mode_str.c_str());
    EXEC_NPU_CMD(
        aclnnLightningIndexerHiCached,
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        key_mean,
        query_layout_ptr,
        key_layout_ptr,
        sparse_count,
        sparse_mode,
        hi_block_size,
        hi_block_num,
        sink,
        recent,
        pooling_mode_ptr,
        output);
    return output;
}

} // namespace vllm_ascend

#endif // LIGHTNING_INDEXER_HI_CACHED_TORCH_ADPT_H
