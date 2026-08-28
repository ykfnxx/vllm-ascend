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
#ifndef SCATTER_ND_UPDATE_MEAN_TORCH_ADPT_H
#define SCATTER_ND_UPDATE_MEAN_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <torch/extension.h>

namespace vllm_ascend {

inline void npu_scatter_nd_update_mean(
    at::Tensor &flat_key_cache,
    const at::Tensor &indices,
    const at::Tensor &updates,
    at::Tensor &key_mean,
    int64_t block_size)
{
    TORCH_CHECK(flat_key_cache.dim() == 2,
                "flat_key_cache must be 2-D: [num_blocks * block_size * kv_heads, head_dim].");
    TORCH_CHECK(indices.dim() == 2 && indices.size(1) == 1,
                "indices must be 2-D: [num_updates, 1].");
    TORCH_CHECK(updates.dim() == 2, "updates must be 2-D: [num_updates, head_dim].");
    TORCH_CHECK(indices.size(0) == updates.size(0),
                "indices rows must match updates rows, but got indices.size(0)=",
                indices.size(0), ", updates.size(0)=", updates.size(0));
    TORCH_CHECK(flat_key_cache.size(1) == updates.size(1),
                "flat_key_cache head_dim must match updates head_dim, but got ",
                flat_key_cache.size(1), " and ", updates.size(1));
    TORCH_CHECK(flat_key_cache.scalar_type() == updates.scalar_type(),
                "flat_key_cache and updates must share dtype.");
    TORCH_CHECK(flat_key_cache.scalar_type() == at::kHalf ||
                    flat_key_cache.scalar_type() == at::kBFloat16,
                "npu_scatter_nd_update_mean only supports float16/bfloat16 key cache.");
    TORCH_CHECK(indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt,
                "indices must be int32 or int64 for npu_scatter_nd_update_mean.");
    TORCH_CHECK(key_mean.dim() == 4,
                "key_mean must be 4-D: [num_blocks, 1, kv_heads, head_dim].");
    TORCH_CHECK(key_mean.scalar_type() == flat_key_cache.scalar_type(),
                "key_mean dtype must match flat_key_cache dtype.");
    TORCH_CHECK(block_size > 0, "block_size must be positive.");
    TORCH_CHECK(key_mean.size(1) == 1, "key_mean dim1 must be 1.");
    const int64_t kv_heads = key_mean.size(2);
    TORCH_CHECK(kv_heads > 0, "key_mean dim2 must be positive.");
    TORCH_CHECK(key_mean.size(3) == flat_key_cache.size(1),
                "key_mean head_dim must match flat_key_cache head_dim.");

    const int64_t num_blocks = key_mean.size(0);
    TORCH_CHECK(flat_key_cache.size(0) >= num_blocks * block_size * kv_heads,
                "flat_key_cache rows must cover key_mean blocks * block_size * kv_heads.");
    TORCH_CHECK(flat_key_cache.is_contiguous(), "flat_key_cache must be contiguous.");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous.");
    TORCH_CHECK(updates.is_contiguous(), "updates must be contiguous.");
    TORCH_CHECK(key_mean.is_contiguous(), "key_mean must be contiguous.");

    constexpr int64_t kDecodeFusedScatterRows = 256;
    bool update_cache_in_kernel = updates.size(0) <= kDecodeFusedScatterRows &&
                                  updates.size(0) < block_size * kv_heads;
    if (!update_cache_in_kernel) {
        EXEC_NPU_CMD(aclnnScatterNdUpdate, flat_key_cache, indices, updates);
    }
    EXEC_NPU_CMD(aclnnScatterNdUpdateMean, flat_key_cache, indices, updates, block_size, update_cache_in_kernel,
                 key_mean);
}

} // namespace vllm_ascend

#endif // SCATTER_ND_UPDATE_MEAN_TORCH_ADPT_H
