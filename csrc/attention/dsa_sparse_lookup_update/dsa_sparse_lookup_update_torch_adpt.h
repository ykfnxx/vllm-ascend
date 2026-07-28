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
    const at::Tensor& query_positions,
    const at::Tensor& query_to_req_idx,
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
        query_positions,
        query_to_req_idx,
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
