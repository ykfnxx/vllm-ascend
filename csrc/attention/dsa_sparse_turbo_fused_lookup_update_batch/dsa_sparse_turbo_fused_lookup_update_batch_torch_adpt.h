/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_TORCH_ADPT_H
#define DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_TORCH_ADPT_H

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor>
dsa_sparse_turbo_fused_lookup_update_batch(
    at::Tensor& index,
    at::Tensor& slot_to_index,
    at::Tensor& free_slots,
    at::Tensor& free_head,
    const at::Tensor& request_rows,
    const at::Tensor& query_start_loc,
    const at::Tensor& query_index,
    const at::Tensor& query_positions,
    const at::Tensor& verify_starts,
    const at::Tensor& tail_starts,
    int64_t req_num,
    int64_t block_size,
    int64_t is_mtp)
{
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
    TORCH_CHECK(request_rows.scalar_type() == at::kInt,
                "request_rows must be int32");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt,
                "query_start_loc must be int32");
    TORCH_CHECK(query_index.scalar_type() == at::kInt,
                "query_index must be int32");
    TORCH_CHECK(query_positions.scalar_type() == at::kInt,
                "query_positions must be int32");
    TORCH_CHECK(verify_starts.scalar_type() == at::kInt,
                "verify_starts must be int32");
    TORCH_CHECK(tail_starts.scalar_type() == at::kInt,
                "tail_starts must be int32");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");
    TORCH_CHECK(block_size > 0, "block_size must be greater than 0");
    TORCH_CHECK(is_mtp == 0 || is_mtp == 1,
                "is_mtp must be 0 or 1");
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
    TORCH_CHECK(request_rows.dim() == 1 &&
                    request_rows.size(0) == req_num,
                "request_rows must have shape [req_num]");
    TORCH_CHECK(query_start_loc.dim() == 1 &&
                    query_start_loc.size(0) == req_num + 1,
                "query_start_loc must have shape [req_num + 1]");
    TORCH_CHECK(query_index.dim() == 2 &&
                    query_index.size(0) >= req_num &&
                    query_index.size(1) == kQueryWidth,
                "query_index must have shape [T, 2048] with T >= req_num");
    TORCH_CHECK(query_positions.dim() == 1 &&
                    query_positions.size(0) == query_index.size(0),
                "query_positions must have shape [T]");
    TORCH_CHECK(verify_starts.dim() == 1 &&
                    verify_starts.size(0) == req_num,
                "verify_starts must have shape [req_num]");
    TORCH_CHECK(tail_starts.dim() == 1 &&
                    tail_starts.size(0) == req_num,
                "tail_starts must have shape [req_num]");
    TORCH_CHECK(index.is_contiguous() &&
                    slot_to_index.is_contiguous() &&
                    free_slots.is_contiguous() &&
                    free_head.is_contiguous() &&
                    request_rows.is_contiguous() &&
                    query_start_loc.is_contiguous() &&
                    query_index.is_contiguous() &&
                    query_positions.is_contiguous() &&
                    verify_starts.is_contiguous() &&
                    tail_starts.is_contiguous(),
                "all dsa_sparse_turbo_fused_lookup_update_batch tensors must be contiguous");
    const auto device = index.device();
    TORCH_CHECK(slot_to_index.device() == device &&
                    free_slots.device() == device &&
                    free_head.device() == device &&
                    request_rows.device() == device &&
                    query_start_loc.device() == device &&
                    query_index.device() == device &&
                    query_positions.device() == device &&
                    verify_starts.device() == device &&
                    tail_starts.device() == device,
                "all dsa_sparse_turbo_fused_lookup_update_batch tensors must be on one device");

    at::Tensor mapped_indices = at::empty_like(query_index);
    at::Tensor miss_mask = at::empty_like(query_index);
    EXEC_NPU_CMD(
        aclnnDsaSparseTurboFusedLookupUpdateBatch,
        index,
        slot_to_index,
        free_slots,
        free_head,
        request_rows,
        query_start_loc,
        query_index,
        query_positions,
        verify_starts,
        tail_starts,
        req_num,
        block_size,
        is_mtp,
        mapped_indices,
        miss_mask);
    return std::make_tuple(mapped_indices, miss_mask);
}

}  // namespace vllm_ascend

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_TORCH_ADPT_H
