/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_OFFLOAD_RESOLVE_UPDATE_BATCH_V2_TORCH_ADPT_H
#define DSA_OFFLOAD_RESOLVE_UPDATE_BATCH_V2_TORCH_ADPT_H

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor>
dsa_offload_resolve_update_batch_v2(
    at::Tensor& index,
    at::Tensor& slot_to_index,
    at::Tensor& free_slots,
    at::Tensor& free_head,
    const at::Tensor& request_rows,
    const at::Tensor& query_start_loc,
    const at::Tensor& query_positions,
    const at::Tensor& semantic_topk,
    at::Tensor& mapped_indices_out,
    at::Tensor& gather_mask_out,
    int64_t req_num,
    int64_t block_size,
    int64_t decode_mode)
{
    constexpr int64_t kIndexCapacity = 128 * 1024;
    constexpr int64_t kSlotCount = 10 * 1024;
    constexpr int64_t kFreeSlotCount = 2 * 1024;
    constexpr int64_t kQueryWidth = 2 * 1024;
    constexpr int64_t kFreeHeadStride = 16;
    constexpr int64_t kBlockSize = 128;

    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");
    TORCH_CHECK(block_size == kBlockSize,
                "block_size must be 128");
    TORCH_CHECK(decode_mode == 0 || decode_mode == 1,
                "decode_mode must be 0 or 1");
    TORCH_CHECK(index.scalar_type() == at::kInt &&
                    slot_to_index.scalar_type() == at::kInt &&
                    free_slots.scalar_type() == at::kInt &&
                    free_head.scalar_type() == at::kInt &&
                    request_rows.scalar_type() == at::kInt &&
                    query_start_loc.scalar_type() == at::kInt &&
                    query_positions.scalar_type() == at::kInt &&
                    semantic_topk.scalar_type() == at::kInt &&
                    mapped_indices_out.scalar_type() == at::kInt &&
                    gather_mask_out.scalar_type() == at::kInt,
                "all resolve-update V2 tensors must be int32");
    TORCH_CHECK(index.dim() == 2 &&
                    index.size(1) == kIndexCapacity,
                "index must have shape [P, 131072]");
    const int64_t pool_capacity = index.size(0);
    TORCH_CHECK(req_num <= pool_capacity,
                "req_num must not exceed pool capacity");
    TORCH_CHECK(slot_to_index.sizes() ==
                    at::IntArrayRef({pool_capacity, kSlotCount}),
                "slot_to_index must have shape [P, 10240]");
    TORCH_CHECK(free_slots.sizes() ==
                    at::IntArrayRef({pool_capacity, kFreeSlotCount}),
                "free_slots must have shape [P, 2048]");
    TORCH_CHECK(free_head.sizes() ==
                    at::IntArrayRef({pool_capacity, kFreeHeadStride}),
                "free_head must have shape [P, 16]");
    TORCH_CHECK(request_rows.dim() == 1 &&
                    request_rows.size(0) == req_num,
                "request_rows must have shape [req_num]");
    TORCH_CHECK(query_start_loc.dim() == 1 &&
                    query_start_loc.size(0) == req_num + 1,
                "query_start_loc must have shape [req_num + 1]");
    TORCH_CHECK(query_positions.dim() == 1,
                "query_positions must have shape [T]");
    const int64_t query_num = query_positions.size(0);
    TORCH_CHECK(query_num > 0, "query_positions must not be empty");
    TORCH_CHECK(semantic_topk.dim() == 3 &&
                    semantic_topk.size(0) == query_num &&
                    semantic_topk.size(1) == 1 &&
                    semantic_topk.size(2) == kQueryWidth,
                "semantic_topk must have shape [T, 1, 2048]");
    TORCH_CHECK(mapped_indices_out.sizes() == semantic_topk.sizes() &&
                    gather_mask_out.sizes() == semantic_topk.sizes(),
                "resolve-update V2 outputs must match semantic_topk");
    TORCH_CHECK(index.is_contiguous() &&
                    slot_to_index.is_contiguous() &&
                    free_slots.is_contiguous() &&
                    free_head.is_contiguous() &&
                    request_rows.is_contiguous() &&
                    query_start_loc.is_contiguous() &&
                    query_positions.is_contiguous() &&
                    semantic_topk.is_contiguous() &&
                    mapped_indices_out.is_contiguous() &&
                    gather_mask_out.is_contiguous(),
                "all resolve-update V2 tensors must be contiguous");
    const auto device = index.device();
    TORCH_CHECK(device.is_privateuseone(),
                "resolve-update V2 state must be on NPU");
    TORCH_CHECK(slot_to_index.device() == device &&
                    free_slots.device() == device &&
                    free_head.device() == device &&
                    request_rows.device() == device &&
                    query_start_loc.device() == device &&
                    query_positions.device() == device &&
                    semantic_topk.device() == device &&
                    mapped_indices_out.device() == device &&
                    gather_mask_out.device() == device,
                "all resolve-update V2 tensors must be on one device");

    EXEC_NPU_CMD(
        aclnnDsaOffloadResolveUpdateBatchV2,
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
        decode_mode,
        mapped_indices_out,
        gather_mask_out);
    return {mapped_indices_out, gather_mask_out};
}

}  // namespace vllm_ascend

#endif  // DSA_OFFLOAD_RESOLVE_UPDATE_BATCH_V2_TORCH_ADPT_H
