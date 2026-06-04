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
