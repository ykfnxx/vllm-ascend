/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TURBO_PREFETCH_TORCH_ADPT_H
#define DSA_SPARSE_LOOKUP_UPDATE_TURBO_PREFETCH_TORCH_ADPT_H

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor>
dsa_sparse_turbo_prefetch_lookup_update_batch(
    at::Tensor& index,
    at::Tensor& slot_to_index,
    at::Tensor& free_slots,
    at::Tensor& free_head,
    const at::Tensor& req_pool_entries,
    const at::Tensor& query_start_loc,
    const at::Tensor& query_index,
    const at::Tensor& lookup_mask,
    int64_t req_num)
{
    constexpr int64_t kIndexCapacity = 128 * 1024;
    constexpr int64_t kSlotCount = 10 * 1024;
    constexpr int64_t kFreeSlotCount = 2 * 1024;
    constexpr int64_t kQueryWidth = 2 * 1024;
    constexpr int64_t kFreeHeadStride = 16;

    TORCH_CHECK(index.scalar_type() == at::kInt,
                "index must be int32");
    TORCH_CHECK(slot_to_index.scalar_type() == at::kInt,
                "slot_to_index must be int32");
    TORCH_CHECK(free_slots.scalar_type() == at::kInt,
                "free_slots must be int32");
    TORCH_CHECK(free_head.scalar_type() == at::kInt,
                "free_head must be int32");
    TORCH_CHECK(req_pool_entries.scalar_type() == at::kInt,
                "req_pool_entries must be int32");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt,
                "query_start_loc must be int32");
    TORCH_CHECK(query_index.scalar_type() == at::kInt,
                "query_index must be int32");
    TORCH_CHECK(lookup_mask.scalar_type() == at::kInt,
                "lookup_mask must be int32");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");
    TORCH_CHECK(index.dim() == 2 &&
                    index.size(1) >= kSlotCount,
                "index must have shape [P, >=10240]");
    const int64_t pool_capacity = index.size(0);
    TORCH_CHECK(req_num <= pool_capacity,
                "req_num must not exceed pool capacity");
    TORCH_CHECK(slot_to_index.dim() == 2 &&
                    slot_to_index.size(0) == pool_capacity &&
                    slot_to_index.size(1) == kSlotCount,
                "slot_to_index must have shape [P, 10240]");
    TORCH_CHECK(free_slots.dim() == 2 &&
                    free_slots.size(0) == pool_capacity &&
                    free_slots.size(1) == kFreeSlotCount,
                "free_slots must have shape [P, 2048]");
    TORCH_CHECK(free_head.dim() == 2 &&
                    free_head.size(0) == pool_capacity &&
                    free_head.size(1) == kFreeHeadStride,
                "free_head must have shape [P, 16]");
    TORCH_CHECK(req_pool_entries.dim() == 1 &&
                    req_pool_entries.size(0) == req_num,
                "req_pool_entries must have shape [req_num]");
    TORCH_CHECK(query_start_loc.dim() == 1 &&
                    query_start_loc.size(0) == req_num + 1,
                "query_start_loc must have shape [req_num + 1]");
    TORCH_CHECK(query_index.dim() == 2 &&
                    query_index.size(0) >= req_num &&
                    query_index.size(1) == kQueryWidth,
                "query_index must have shape [T, 2048] with T >= req_num");
    TORCH_CHECK(lookup_mask.sizes() == query_index.sizes(),
                "lookup_mask must have the same shape as query_index");
    TORCH_CHECK(index.is_contiguous() &&
                    slot_to_index.is_contiguous() &&
                    free_slots.is_contiguous() &&
                    free_head.is_contiguous() &&
                    req_pool_entries.is_contiguous() &&
                    query_start_loc.is_contiguous() &&
                    query_index.is_contiguous() &&
                    lookup_mask.is_contiguous(),
                "all dsa_sparse_turbo_prefetch_lookup_update_batch tensors must be contiguous");
    const auto device = index.device();
    TORCH_CHECK(slot_to_index.device() == device &&
                    free_slots.device() == device &&
                    free_head.device() == device &&
                    req_pool_entries.device() == device &&
                    query_start_loc.device() == device &&
                    query_index.device() == device &&
                    lookup_mask.device() == device,
                "all dsa_sparse_turbo_prefetch_lookup_update_batch tensors must be on one device");

    at::Tensor slot_out = at::empty_like(query_index);
    at::Tensor miss_out = at::empty_like(query_index);
    EXEC_NPU_CMD(
        aclnnDsaSparseTurboPrefetchLookupUpdateBatch,
        index,
        slot_to_index,
        free_slots,
        free_head,
        req_pool_entries,
        query_start_loc,
        query_index,
        lookup_mask,
        req_num,
        slot_out,
        miss_out);
    return std::make_tuple(slot_out, miss_out);
}

}  // namespace vllm_ascend

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TURBO_PREFETCH_TORCH_ADPT_H
