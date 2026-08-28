/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef ASU_KV_GATHER_TORCH_ADPT_H
#define ASU_KV_GATHER_TORCH_ADPT_H

#include <cstdint>
#include <limits>
#include <tuple>

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> asu_kv_gather(
    at::Tensor& destination_kv_cache,
    at::Tensor& destination_k_rope,
    const at::Tensor& destination_block_table,
    const at::Tensor& source_kv_cache,
    const at::Tensor& source_k_rope,
    const at::Tensor& source_block_table,
    const at::Tensor& req_pool_entries,
    const at::Tensor& token_positions,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_mask,
    int64_t block_size,
    int64_t req_num)
{
    constexpr int64_t UINT32_LIMIT =
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    const at::Device device = destination_kv_cache.device();

    TORCH_CHECK(destination_kv_cache.device().is_privateuseone(),
                "destination_kv_cache must be on NPU");
    TORCH_CHECK(destination_k_rope.device().is_privateuseone(),
                "destination_k_rope must be on NPU");
    TORCH_CHECK(source_kv_cache.device().is_privateuseone(),
                "source_kv_cache must be NPU-addressable swapped memory");
    TORCH_CHECK(source_k_rope.device().is_privateuseone(),
                "source_k_rope must be NPU-addressable swapped memory");
    TORCH_CHECK(destination_block_table.device().is_privateuseone(),
                "destination_block_table must be on NPU");
    TORCH_CHECK(source_block_table.device().is_privateuseone(),
                "source_block_table must be on NPU");
    TORCH_CHECK(req_pool_entries.device().is_privateuseone(),
                "req_pool_entries must be on NPU");
    TORCH_CHECK(token_positions.device().is_privateuseone(),
                "token_positions must be on NPU");
    TORCH_CHECK(destination_slots.device().is_privateuseone(),
                "destination_slots must be on NPU");
    TORCH_CHECK(miss_mask.device().is_privateuseone(),
                "miss_mask must be on NPU");
    TORCH_CHECK(destination_k_rope.device() == device,
                "destination_k_rope must be on the same NPU as destination_kv_cache");
    TORCH_CHECK(destination_block_table.device() == device,
                "destination_block_table must be on the same NPU as destination_kv_cache");
    TORCH_CHECK(source_kv_cache.device() == device,
                "source_kv_cache must be addressable from the destination NPU");
    TORCH_CHECK(source_k_rope.device() == device,
                "source_k_rope must be addressable from the destination NPU");
    TORCH_CHECK(source_block_table.device() == device,
                "source_block_table must be on the same NPU as destination_kv_cache");
    TORCH_CHECK(req_pool_entries.device() == device,
                "req_pool_entries must be on the same NPU as destination_kv_cache");
    TORCH_CHECK(token_positions.device() == device,
                "token_positions must be on the same NPU as destination_kv_cache");
    TORCH_CHECK(destination_slots.device() == device,
                "destination_slots must be on the same NPU as destination_kv_cache");
    TORCH_CHECK(miss_mask.device() == device,
                "miss_mask must be on the same NPU as destination_kv_cache");
    TORCH_CHECK(destination_kv_cache.is_contiguous(),
                "destination_kv_cache must be contiguous");
    TORCH_CHECK(destination_k_rope.is_contiguous(),
                "destination_k_rope must be contiguous");
    TORCH_CHECK(destination_block_table.is_contiguous(),
                "destination_block_table must be contiguous");
    TORCH_CHECK(source_kv_cache.is_contiguous(),
                "source_kv_cache must be contiguous");
    TORCH_CHECK(source_k_rope.is_contiguous(),
                "source_k_rope must be contiguous");
    TORCH_CHECK(source_block_table.is_contiguous(),
                "source_block_table must be contiguous");
    TORCH_CHECK(req_pool_entries.is_contiguous(),
                "req_pool_entries must be contiguous");
    TORCH_CHECK(token_positions.is_contiguous(),
                "token_positions must be contiguous");
    TORCH_CHECK(destination_slots.is_contiguous(),
                "destination_slots must be contiguous");
    TORCH_CHECK(miss_mask.is_contiguous(),
                "miss_mask must be contiguous");
    TORCH_CHECK(destination_block_table.scalar_type() == at::kInt,
                "destination_block_table must be int32");
    TORCH_CHECK(source_block_table.scalar_type() == at::kInt,
                "source_block_table must be int32");
    TORCH_CHECK(req_pool_entries.scalar_type() == at::kInt,
                "req_pool_entries must be int32");
    TORCH_CHECK(token_positions.scalar_type() == at::kInt,
                "token_positions must be int32");
    TORCH_CHECK(destination_slots.scalar_type() == at::kInt,
                "destination_slots must be int32 (original kvgather format).");
    TORCH_CHECK(miss_mask.scalar_type() == at::kInt,
                "miss_mask must be int32");
    const at::ScalarType kv_dtype = destination_kv_cache.scalar_type();
    const at::ScalarType rope_dtype = destination_k_rope.scalar_type();
    const bool supported_dtype_pair =
        (kv_dtype == at::kHalf && rope_dtype == at::kHalf) ||
        (kv_dtype == at::kBFloat16 && rope_dtype == at::kBFloat16) ||
        (kv_dtype == at::kChar && rope_dtype == at::kBFloat16);
    TORCH_CHECK(
        supported_dtype_pair,
        "asu_kv_gather supports (KV, RoPE) dtype pairs "
        "(float16, float16), (bfloat16, bfloat16), and (int8, bfloat16)");
    TORCH_CHECK(destination_kv_cache.dim() == 3,
                "destination_kv_cache must be [blocks, block, dim]");
    TORCH_CHECK(destination_k_rope.dim() == 3,
                "destination_k_rope must be [blocks, block, dim]");
    TORCH_CHECK(source_kv_cache.dim() == 3,
                "source_kv_cache must be [blocks, block, dim]");
    TORCH_CHECK(source_k_rope.dim() == 3,
                "source_k_rope must be [blocks, block, dim]");
    TORCH_CHECK(destination_block_table.dim() == 2,
                "destination_block_table must be [requests, blocks]");
    TORCH_CHECK(source_block_table.dim() == 2,
                "source_block_table must be [pool_capacity, blocks]");
    TORCH_CHECK(req_pool_entries.dim() == 1,
                "req_pool_entries must be [requests]");
    TORCH_CHECK(token_positions.dim() == 2,
                "token_positions must have rank 2");
    TORCH_CHECK(destination_slots.dim() == 2,
                "destination_slots must have rank 2");
    TORCH_CHECK(miss_mask.dim() == 2,
                "miss_mask must have rank 2");
    TORCH_CHECK(block_size > 0, "block_size must be greater than 0");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");
    TORCH_CHECK(block_size <= UINT32_LIMIT,
                "block_size must fit uint32 tiling data");
    TORCH_CHECK(req_num <= UINT32_LIMIT,
                "req_num must fit uint32 tiling data");
    TORCH_CHECK(req_num == destination_block_table.size(0),
                "req_num must equal destination_block_table batch size");
    TORCH_CHECK(req_num == req_pool_entries.size(0),
                "req_num must equal req_pool_entries size");
    const bool dense_layout =
        token_positions.size(0) == req_num &&
        destination_slots.sizes() == token_positions.sizes() &&
        miss_mask.sizes() == token_positions.sizes();
    const bool resident_init_layout =
        token_positions.size(0) == 1 &&
        destination_slots.size(0) == 1 &&
        destination_slots.size(1) == token_positions.size(1) &&
        miss_mask.size(0) == req_num &&
        miss_mask.size(1) == 1;
    TORCH_CHECK(
        dense_layout || resident_init_layout,
        "index metadata must use dense [requests, query_count] layout or "
        "resident initialization [1, query_count], [1, query_count], "
        "[requests, 1] layout");
    TORCH_CHECK(token_positions.size(1) > 0,
                "token_positions query_count must be greater than 0");
    TORCH_CHECK(token_positions.size(1) <= UINT32_LIMIT,
                "token_positions query_count must fit uint32 tiling data");
    TORCH_CHECK(source_block_table.size(0) > 0,
                "source_block_table pool capacity must be greater than 0");
    TORCH_CHECK(source_block_table.size(1) > 0,
                "source_block_table width must be greater than 0");
    TORCH_CHECK(destination_block_table.size(1) > 0,
                "destination_block_table width must be greater than 0");
    TORCH_CHECK(source_block_table.size(0) <= UINT32_LIMIT &&
                    source_block_table.size(1) <= UINT32_LIMIT,
                "source_block_table dimensions must fit uint32 tiling data");
    TORCH_CHECK(destination_block_table.size(1) <= UINT32_LIMIT,
                "destination_block_table width must fit uint32 tiling data");
    TORCH_CHECK(destination_kv_cache.size(0) > 0 &&
                    source_kv_cache.size(0) > 0,
                "source and destination KV caches must contain physical blocks");
    TORCH_CHECK(destination_kv_cache.size(0) <= UINT32_LIMIT &&
                    source_kv_cache.size(0) <= UINT32_LIMIT,
                "source and destination physical block counts must fit uint32 tiling data");
    TORCH_CHECK(destination_kv_cache.size(1) == block_size,
                "destination_kv_cache block dimension must equal block_size");
    TORCH_CHECK(destination_k_rope.size(1) == block_size,
                "destination_k_rope block dimension must equal block_size");
    TORCH_CHECK(source_kv_cache.size(1) == block_size,
                "source_kv_cache block dimension must equal block_size");
    TORCH_CHECK(source_k_rope.size(1) == block_size,
                "source_k_rope block dimension must equal block_size");
    TORCH_CHECK(destination_kv_cache.size(0) ==
                    destination_k_rope.size(0),
                "destination KV and RoPE physical block counts must match");
    TORCH_CHECK(source_kv_cache.size(0) == source_k_rope.size(0),
                "source KV and RoPE physical block counts must match");
    TORCH_CHECK(destination_kv_cache.size(2) > 0,
                "destination KV record dimension must be greater than 0");
    TORCH_CHECK(destination_k_rope.size(2) > 0,
                "destination RoPE record dimension must be greater than 0");
    TORCH_CHECK(destination_kv_cache.size(2) <= UINT32_LIMIT &&
                    destination_k_rope.size(2) <= UINT32_LIMIT,
                "KV and RoPE record dimensions must fit uint32 tiling data");
    TORCH_CHECK(destination_kv_cache.size(2) == source_kv_cache.size(2),
                "source and destination KV record dimensions must match");
    TORCH_CHECK(destination_k_rope.size(2) == source_k_rope.size(2),
                "source and destination RoPE record dimensions must match");
    TORCH_CHECK(destination_kv_cache.scalar_type() ==
                    source_kv_cache.scalar_type(),
                "source and destination KV cache dtypes must match");
    TORCH_CHECK(destination_k_rope.scalar_type() ==
                    source_k_rope.scalar_type(),
                "source and destination RoPE cache dtypes must match");

    const int64_t kv_record_bytes =
        destination_kv_cache.size(2) * destination_kv_cache.element_size();
    const int64_t rope_record_bytes =
        destination_k_rope.size(2) * destination_k_rope.element_size();
    TORCH_CHECK(kv_record_bytes % 32 == 0,
                "KV record size must be 32-byte aligned");
    TORCH_CHECK(rope_record_bytes % 32 == 0,
                "RoPE record size must be 32-byte aligned");

    EXEC_NPU_CMD(aclnnAsuKvGather,
                 destination_kv_cache,
                 destination_k_rope,
                 destination_block_table,
                 source_kv_cache,
                 source_k_rope,
                 source_block_table,
                 req_pool_entries,
                 token_positions,
                 destination_slots,
                 miss_mask,
                 block_size,
                 req_num);
    return std::make_tuple(destination_kv_cache, destination_k_rope);
}

}  // namespace vllm_ascend

#endif  // ASU_KV_GATHER_TORCH_ADPT_H
