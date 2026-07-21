/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef ASU_HBM_INDEX_LOOKUP_TORCH_ADPT_H
#define ASU_HBM_INDEX_LOOKUP_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> asu_hbm_index_lookup(
    at::Tensor& index,
    at::Tensor& slot_to_index,
    at::Tensor& free_slots,
    at::Tensor& free_head,
    const at::Tensor& req_pool_entries,
    const at::Tensor& query_index,
    const at::Tensor& lookup_mask,
    int64_t req_num)
{
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(slot_to_index.scalar_type() == at::kInt, "slot_to_index must be int32");
    TORCH_CHECK(free_slots.scalar_type() == at::kInt, "free_slots must be int32");
    TORCH_CHECK(free_head.scalar_type() == at::kInt, "free_head must be int32");
    TORCH_CHECK(req_pool_entries.scalar_type() == at::kInt, "req_pool_entries must be int32");
    TORCH_CHECK(query_index.scalar_type() == at::kInt, "query_index must be int32");
    TORCH_CHECK(lookup_mask.scalar_type() == at::kInt, "lookup_mask must be int32");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");

    at::Tensor slot_out = at::empty_like(query_index);
    at::Tensor miss_out = at::empty_like(query_index);
    EXEC_NPU_CMD(aclnnAsuHbmIndexLookup,
                 index,
                 slot_to_index,
                 free_slots,
                 free_head,
                 req_pool_entries,
                 query_index,
                 lookup_mask,
                 req_num,
                 slot_out,
                 miss_out);
    return std::make_tuple(slot_out, miss_out);
}

}  // namespace vllm_ascend

#endif  // ASU_HBM_INDEX_LOOKUP_TORCH_ADPT_H
