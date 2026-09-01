/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_LOOKUP_UPDATE_TORCH_ADPT_H
#define DSA_OFFLOAD_LOOKUP_UPDATE_TORCH_ADPT_H

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor> dsa_offload_lookup_update(
    at::Tensor& index,
    at::Tensor& slot_to_index,
    at::Tensor& free_slots,
    at::Tensor& free_head,
    const at::Tensor& request_rows,
    const at::Tensor& query_start_loc,
    const at::Tensor& query_positions,
    const at::Tensor& semantic_topk,
    int64_t req_num,
    int64_t block_size,
    int64_t tail_base,
    int64_t fallback_slot,
    int64_t staging_base,
    int64_t decode_mode)
{
    constexpr int64_t kIndexCapacity = 128 * 1024;
    constexpr int64_t kResidentSlots = 8 * 1024;
    constexpr int64_t kSlotCount = 10 * 1024;
    constexpr int64_t kFreeSlotCount = 2 * 1024;
    constexpr int64_t kTopkWidth = 2 * 1024;
    constexpr int64_t kFreeHeadStride = 16;

    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(slot_to_index.scalar_type() == at::kInt,
                "slot_to_index must be int32");
    TORCH_CHECK(free_slots.scalar_type() == at::kInt,
                "free_slots must be int32");
    TORCH_CHECK(free_head.scalar_type() == at::kInt,
                "free_head must be int32");
    TORCH_CHECK(request_rows.scalar_type() == at::kInt,
                "request_rows must be int32");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt,
                "query_start_loc must be int32");
    TORCH_CHECK(query_positions.scalar_type() == at::kLong,
                "query_positions must be int64");
    TORCH_CHECK(semantic_topk.scalar_type() == at::kInt,
                "semantic_topk must be int32");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");
    TORCH_CHECK(block_size > 0, "block_size must be greater than 0");
    TORCH_CHECK(decode_mode == 0 || decode_mode == 1,
                "decode_mode must be 0 or 1");
    TORCH_CHECK(index.dim() == 2 &&
                    index.size(1) == kIndexCapacity,
                "index must have shape [P, 131072]");
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
    TORCH_CHECK(request_rows.dim() == 1 &&
                    request_rows.size(0) == req_num,
                "request_rows must have shape [req_num]");
    TORCH_CHECK(query_start_loc.dim() == 1 &&
                    query_start_loc.size(0) == req_num + 1,
                "query_start_loc must have shape [req_num + 1]");
    TORCH_CHECK(semantic_topk.dim() == 3 &&
                    semantic_topk.size(1) == 1 &&
                    semantic_topk.size(2) == kTopkWidth,
                "semantic_topk must have shape [T, 1, 2048]");
    const int64_t query_num = semantic_topk.size(0);
    TORCH_CHECK(query_positions.dim() == 1 &&
                    query_positions.size(0) == query_num,
                "query_positions must have shape [T]");
    const int64_t replaceable_base =
        (kResidentSlots + block_size - 1) / block_size * block_size;
    const int64_t expected_tail =
        replaceable_base +
        (kFreeSlotCount + block_size - 1) /
            block_size * block_size;
    TORCH_CHECK(tail_base == expected_tail &&
                    fallback_slot == tail_base + block_size &&
                    staging_base == fallback_slot + 1,
                "lookup geometry must match the Hot Cache layout");
    TORCH_CHECK(index.is_contiguous() &&
                    slot_to_index.is_contiguous() &&
                    free_slots.is_contiguous() &&
                    free_head.is_contiguous() &&
                    request_rows.is_contiguous() &&
                    query_start_loc.is_contiguous() &&
                    query_positions.is_contiguous() &&
                    semantic_topk.is_contiguous(),
                "all dsa_offload_lookup_update tensors must be contiguous");
    const auto device = index.device();
    TORCH_CHECK(slot_to_index.device() == device &&
                    free_slots.device() == device &&
                    free_head.device() == device &&
                    request_rows.device() == device &&
                    query_start_loc.device() == device &&
                    query_positions.device() == device &&
                    semantic_topk.device() == device,
                "all dsa_offload_lookup_update tensors must be on one device");

    at::Tensor mapped_indices = at::empty_like(semantic_topk);
    at::Tensor miss_mask = at::empty_like(semantic_topk);
    EXEC_NPU_CMD(
        aclnnDsaOffloadLookupUpdate,
        index,
        slot_to_index,
        free_slots,
        free_head,
        request_rows,
        query_start_loc,
        query_positions,
        semantic_topk,
        req_num,
        block_size,
        tail_base,
        fallback_slot,
        staging_base,
        decode_mode,
        mapped_indices,
        miss_mask);
    return std::make_tuple(mapped_indices, miss_mask);
}

}  // namespace vllm_ascend

#endif  // DSA_OFFLOAD_LOOKUP_UPDATE_TORCH_ADPT_H
