/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef ASU_HBM_INDEX_MAINTAIN_AICPU_TORCH_ADPT_H
#define ASU_HBM_INDEX_MAINTAIN_AICPU_TORCH_ADPT_H

namespace vllm_ascend {

inline void asu_hbm_index_maintain_aicpu(
    at::Tensor& index,
    at::Tensor& slot_to_index,
    at::Tensor& free_slots,
    at::Tensor& free_head,
    const at::Tensor& req_pool_entries,
    const at::Tensor& last_query_slots,
    int64_t req_num,
    int64_t seed)
{
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(slot_to_index.scalar_type() == at::kInt,
                "slot_to_index must be int32");
    TORCH_CHECK(free_slots.scalar_type() == at::kInt,
                "free_slots must be int32");
    TORCH_CHECK(free_head.scalar_type() == at::kInt,
                "free_head must be int32");
    TORCH_CHECK(req_pool_entries.scalar_type() == at::kInt,
                "req_pool_entries must be int32");
    TORCH_CHECK(last_query_slots.scalar_type() == at::kInt,
                "last_query_slots must be int32");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");

    EXEC_NPU_CMD(
        aclnnAsuHbmIndexMaintainAicpu,
        index,
        slot_to_index,
        free_slots,
        free_head,
        req_pool_entries,
        last_query_slots,
        req_num,
        seed,
        index,
        slot_to_index,
        free_slots,
        free_head);
}

}  // namespace vllm_ascend

#endif  // ASU_HBM_INDEX_MAINTAIN_AICPU_TORCH_ADPT_H
