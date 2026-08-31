/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef ASU_KV_GATHER_DIRECT_V2_TORCH_ADPT_H
#define ASU_KV_GATHER_DIRECT_V2_TORCH_ADPT_H

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor> asu_kv_gather_direct_v2(
    at::Tensor& destination_kv_cache,
    at::Tensor& destination_k_rope,
    const at::Tensor& hot_block_table_pool,
    const at::Tensor& source_kv_cache,
    const at::Tensor& source_k_rope,
    const at::Tensor& source_block_table,
    const at::Tensor& request_rows,
    const at::Tensor& query_start_loc,
    const at::Tensor& semantic_topk,
    const at::Tensor& mapped_indices,
    const at::Tensor& gather_mask,
    int64_t block_size,
    int64_t req_num)
{
    constexpr int64_t kBlockSize = 128;
    constexpr int64_t kQueryWidth = 2 * 1024;
    const auto device = destination_kv_cache.device();

    TORCH_CHECK(block_size == kBlockSize,
                "block_size must be 128");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");
    TORCH_CHECK(destination_kv_cache.device().is_privateuseone() &&
                    destination_k_rope.device() == device &&
                    hot_block_table_pool.device() == device &&
                    source_kv_cache.device() == device &&
                    source_k_rope.device() == device &&
                    source_block_table.device() == device &&
                    request_rows.device() == device &&
                    query_start_loc.device() == device &&
                    semantic_topk.device() == device &&
                    mapped_indices.device() == device &&
                    gather_mask.device() == device,
                "all direct Gather V2 tensors must be on one NPU");
    TORCH_CHECK(destination_kv_cache.is_contiguous() &&
                    destination_k_rope.is_contiguous() &&
                    hot_block_table_pool.is_contiguous() &&
                    source_kv_cache.is_contiguous() &&
                    source_k_rope.is_contiguous() &&
                    source_block_table.is_contiguous() &&
                    request_rows.is_contiguous() &&
                    query_start_loc.is_contiguous() &&
                    semantic_topk.is_contiguous() &&
                    mapped_indices.is_contiguous() &&
                    gather_mask.is_contiguous(),
                "all direct Gather V2 tensors must be contiguous");
    TORCH_CHECK(hot_block_table_pool.scalar_type() == at::kInt &&
                    source_block_table.scalar_type() == at::kInt &&
                    request_rows.scalar_type() == at::kInt &&
                    query_start_loc.scalar_type() == at::kInt &&
                    semantic_topk.scalar_type() == at::kInt &&
                    mapped_indices.scalar_type() == at::kInt &&
                    gather_mask.scalar_type() == at::kInt,
                "all direct Gather V2 metadata tensors must be int32");
    const at::ScalarType kv_dtype = destination_kv_cache.scalar_type();
    const at::ScalarType rope_dtype = destination_k_rope.scalar_type();
    TORCH_CHECK(
        (kv_dtype == at::kHalf && rope_dtype == at::kHalf) ||
            (kv_dtype == at::kBFloat16 && rope_dtype == at::kBFloat16) ||
            (kv_dtype == at::kChar && rope_dtype == at::kBFloat16),
        "direct Gather V2 supports FP16/FP16, BF16/BF16, and INT8/BF16");
    TORCH_CHECK(source_kv_cache.scalar_type() == kv_dtype &&
                    source_k_rope.scalar_type() == rope_dtype,
                "source and destination cache dtypes must match");
    TORCH_CHECK(destination_kv_cache.dim() == 3 &&
                    destination_k_rope.dim() == 3 &&
                    source_kv_cache.dim() == 3 &&
                    source_k_rope.dim() == 3,
                "cache tensors must have shape [blocks, block_size, dim]");
    TORCH_CHECK(destination_kv_cache.size(1) == block_size &&
                    destination_k_rope.size(1) == block_size &&
                    source_kv_cache.size(1) == block_size &&
                    source_k_rope.size(1) == block_size,
                "cache block dimensions must equal block_size");
    TORCH_CHECK(destination_kv_cache.size(0) ==
                    destination_k_rope.size(0) &&
                    source_kv_cache.size(0) == source_k_rope.size(0),
                "KV and RoPE physical block counts must match");
    TORCH_CHECK(destination_kv_cache.size(0) > 0 &&
                    source_kv_cache.size(0) > 0 &&
                    destination_kv_cache.size(2) > 0 &&
                    destination_k_rope.size(2) > 0,
                "cache physical block counts and record dimensions must be positive");
    TORCH_CHECK(destination_kv_cache.size(2) == source_kv_cache.size(2) &&
                    destination_k_rope.size(2) == source_k_rope.size(2),
                "source and destination record dimensions must match");
    TORCH_CHECK(
        destination_kv_cache.size(2) * destination_kv_cache.element_size() %
                    32 ==
                0 &&
            destination_k_rope.size(2) *
                        destination_k_rope.element_size() %
                    32 ==
                0,
        "KV and RoPE record sizes must be 32-byte aligned");
    TORCH_CHECK(hot_block_table_pool.dim() == 2 &&
                    source_block_table.dim() == 2 &&
                    hot_block_table_pool.size(0) ==
                        source_block_table.size(0),
                "block table pools must have shape [P, H]");
    TORCH_CHECK(hot_block_table_pool.size(0) >= req_num &&
                    hot_block_table_pool.size(1) > 0 &&
                    source_block_table.size(1) > 0,
                "block table pools must cover every request and contain blocks");
    TORCH_CHECK(request_rows.dim() == 1 &&
                    request_rows.size(0) == req_num,
                "request_rows must have shape [req_num]");
    TORCH_CHECK(query_start_loc.dim() == 1 &&
                    query_start_loc.size(0) == req_num + 1,
                "query_start_loc must have shape [req_num + 1]");
    TORCH_CHECK(semantic_topk.dim() == 3 &&
                    semantic_topk.size(0) > 0 &&
                    semantic_topk.size(1) == 1 &&
                    semantic_topk.size(2) == kQueryWidth,
                "semantic_topk must have shape [T, 1, 2048]");
    TORCH_CHECK(mapped_indices.sizes() == semantic_topk.sizes() &&
                    gather_mask.sizes() == semantic_topk.sizes(),
                "direct Gather V2 metadata shapes must match");

    EXEC_NPU_CMD(
        aclnnAsuKvGatherDirectV2,
        destination_kv_cache,
        destination_k_rope,
        hot_block_table_pool,
        source_kv_cache,
        source_k_rope,
        source_block_table,
        request_rows,
        query_start_loc,
        semantic_topk,
        mapped_indices,
        gather_mask,
        block_size,
        req_num);
    return {destination_kv_cache, destination_k_rope};
}

}  // namespace vllm_ascend

#endif  // ASU_KV_GATHER_DIRECT_V2_TORCH_ADPT_H
