/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef DSA_SPARSE_LOOKUP_UPDATE_TORCH_ADPT_H
#define DSA_SPARSE_LOOKUP_UPDATE_TORCH_ADPT_H

namespace vllm_ascend {

inline void dsa_sparse_lookup_update(
    at::Tensor& token_to_hot,
    at::Tensor& hot_to_token,
    at::Tensor& lru_slots,
    at::Tensor& state_seat_epoch,
    const at::Tensor& row_to_cache_seat,
    const at::Tensor& row_seat_epoch,
    const at::Tensor& query_positions,
    const at::Tensor& query_to_row,
    const at::Tensor& query_to_lane,
    const at::Tensor& query_valid_mask,
    const at::Tensor& valid_topk_counts,
    const at::Tensor& seq_lens,
    const at::Tensor& topk_positions,
    at::Tensor& resolved_hot_indices,
    at::Tensor& miss_mask,
    at::Tensor& workspace)
{
    EXEC_NPU_CMD(
        aclnnDsaSparseLookupUpdate,
        token_to_hot,
        hot_to_token,
        lru_slots,
        state_seat_epoch,
        row_to_cache_seat,
        row_seat_epoch,
        query_positions,
        query_to_row,
        query_to_lane,
        query_valid_mask,
        valid_topk_counts,
        seq_lens,
        topk_positions,
        resolved_hot_indices,
        miss_mask,
        workspace);
}

}  // namespace vllm_ascend

#endif  // DSA_SPARSE_LOOKUP_UPDATE_TORCH_ADPT_H
