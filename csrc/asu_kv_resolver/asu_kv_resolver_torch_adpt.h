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
#ifndef ASU_KV_RESOLVER_TORCH_ADPT_H
#define ASU_KV_RESOLVER_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_asu_resolve_kv_slots_single_req(
    const at::Tensor& original_topk_indices,
    int64_t actual_seq_len,
    int64_t managed_prefix_len,
    at::Tensor& token_state,
    const at::Tensor& asu_record_addr,
    at::Tensor& hbm_slot_of_token,
    at::Tensor& slot_owner_token,
    const at::Tensor& free_slot_stack,
    at::Tensor& free_slot_count,
    const at::Tensor& original_block_table,
    const at::Tensor& original_kv_cache_0,
    const at::Tensor& original_kv_cache_1,
    at::Tensor& managed_kv_cache_0,
    at::Tensor& managed_kv_cache_1,
    int64_t block_size)
{
    auto check_int32 = [](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.scalar_type() == at::kInt,
                    name, " must be int32, but got ", tensor.scalar_type());
    };
    auto check_numel_at_least = [](const at::Tensor& tensor, const char* name,
                                   int64_t min_numel) {
        TORCH_CHECK(tensor.numel() >= min_numel,
                    name, " numel must be at least ", min_numel,
                    ", but got ", tensor.numel());
    };
    auto check_slot_tensor = [](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.dim() >= 1,
                    name, " must have a leading slot dimension");
    };
    auto check_per_slot_shape = [](const at::Tensor& lhs, const char* lhs_name,
                                   const at::Tensor& rhs, const char* rhs_name) {
        TORCH_CHECK(lhs.dim() == rhs.dim(),
                    lhs_name, " and ", rhs_name,
                    " must have the same rank, but got ", lhs.dim(),
                    " and ", rhs.dim());
        for (int64_t dim = 1; dim < lhs.dim(); ++dim) {
            TORCH_CHECK(lhs.size(dim) == rhs.size(dim),
                        lhs_name, " and ", rhs_name,
                        " must have matching per-slot shapes, mismatch at dim ",
                        dim, ": ", lhs.size(dim), " vs ", rhs.size(dim));
        }
    };

    TORCH_CHECK(actual_seq_len >= 0,
                "actual_seq_len must be non-negative, but got ",
                actual_seq_len);
    TORCH_CHECK(managed_prefix_len >= 0,
                "managed_prefix_len must be non-negative, but got ",
                managed_prefix_len);
    TORCH_CHECK(managed_prefix_len <= actual_seq_len,
                "managed_prefix_len must not exceed actual_seq_len, got ",
                managed_prefix_len, " and ", actual_seq_len);
    TORCH_CHECK(block_size > 0,
                "block_size must be positive, but got ", block_size);

    check_int32(original_topk_indices, "original_topk_indices");
    check_int32(token_state, "token_state");
    check_int32(asu_record_addr, "asu_record_addr");
    check_int32(hbm_slot_of_token, "hbm_slot_of_token");
    check_int32(slot_owner_token, "slot_owner_token");
    check_int32(free_slot_stack, "free_slot_stack");
    check_int32(free_slot_count, "free_slot_count");
    check_int32(original_block_table, "original_block_table");

    check_numel_at_least(token_state, "token_state", actual_seq_len);
    check_numel_at_least(asu_record_addr, "asu_record_addr", actual_seq_len);
    check_numel_at_least(hbm_slot_of_token, "hbm_slot_of_token", actual_seq_len);
    check_numel_at_least(free_slot_count, "free_slot_count", 1);
    const int64_t required_blocks =
        actual_seq_len == 0 ? 0 : ((actual_seq_len + block_size - 1) / block_size);
    check_numel_at_least(original_block_table, "original_block_table",
                         required_blocks);

    check_slot_tensor(original_kv_cache_0, "original_kv_cache_0");
    check_slot_tensor(original_kv_cache_1, "original_kv_cache_1");
    check_slot_tensor(managed_kv_cache_0, "managed_kv_cache_0");
    check_slot_tensor(managed_kv_cache_1, "managed_kv_cache_1");
    TORCH_CHECK(original_kv_cache_1.scalar_type() ==
                    original_kv_cache_0.scalar_type() &&
                managed_kv_cache_0.scalar_type() ==
                    original_kv_cache_0.scalar_type() &&
                managed_kv_cache_1.scalar_type() ==
                    original_kv_cache_0.scalar_type(),
                "all KV cache tensors must have the same dtype");
    check_per_slot_shape(original_kv_cache_0, "original_kv_cache_0",
                         managed_kv_cache_0, "managed_kv_cache_0");
    check_per_slot_shape(original_kv_cache_1, "original_kv_cache_1",
                         managed_kv_cache_1, "managed_kv_cache_1");
    TORCH_CHECK(slot_owner_token.numel() >= managed_kv_cache_0.size(0),
                "slot_owner_token must cover managed_kv_cache_0 slots");
    TORCH_CHECK(free_slot_stack.numel() >= managed_kv_cache_0.size(0),
                "free_slot_stack must cover managed_kv_cache_0 slots");

    at::Tensor resolved_kv_slots = at::empty(
        original_topk_indices.sizes(),
        original_topk_indices.options().dtype(at::kInt));
    EXEC_NPU_CMD(
        aclnnAsuKvResolver,
        original_topk_indices,
        token_state,
        asu_record_addr,
        hbm_slot_of_token,
        slot_owner_token,
        free_slot_stack,
        free_slot_count,
        original_block_table,
        original_kv_cache_0,
        original_kv_cache_1,
        managed_kv_cache_0,
        managed_kv_cache_1,
        actual_seq_len,
        managed_prefix_len,
        block_size,
        resolved_kv_slots);
    return resolved_kv_slots;
}

} // namespace vllm_ascend
#endif
