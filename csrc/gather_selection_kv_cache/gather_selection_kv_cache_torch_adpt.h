/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef GATHER_SELECTION_KV_CACHE_TORCH_ADPT_H
#define GATHER_SELECTION_KV_CACHE_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_gather_selection_kv_cache(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& req_pool_entries,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& row_modes,
    const at::Tensor& budget_lengths,
    const at::Tensor& tail_valid_token_counts,
    const at::Tensor& resident_tail_starts,
    const at::Tensor& query_position_rows,
    const at::Tensor& attention_indices_out)
{
    TORCH_CHECK(selection_k_rope.device().is_privateuseone(), "selection_k_rope must be on NPU.");
    TORCH_CHECK(selection_kv_cache.device().is_privateuseone(), "selection_kv_cache must be on NPU.");
    TORCH_CHECK(selection_kv_block_table.device().is_privateuseone(), "selection_kv_block_table must be on NPU.");
    TORCH_CHECK(selection_kv_block_status.device().is_privateuseone(), "selection_kv_block_status must be on NPU.");
    TORCH_CHECK(req_pool_entries.device().is_privateuseone(), "req_pool_entries must be on NPU.");
    TORCH_CHECK(selection_topk_indices.device().is_privateuseone(), "selection_topk_indices must be on NPU.");
    TORCH_CHECK(full_kv_block_table.device().is_privateuseone(), "full_kv_block_table must be on NPU.");
    TORCH_CHECK(full_kv_actual_seq.device().is_privateuseone(), "full_kv_actual_seq must be on NPU.");
    TORCH_CHECK(row_modes.device().is_privateuseone(), "row_modes must be on NPU.");
    TORCH_CHECK(budget_lengths.device().is_privateuseone(), "budget_lengths must be on NPU.");
    TORCH_CHECK(tail_valid_token_counts.device().is_privateuseone(),
                "tail_valid_token_counts must be on NPU.");
    TORCH_CHECK(resident_tail_starts.device().is_privateuseone(), "resident_tail_starts must be on NPU.");
    TORCH_CHECK(query_position_rows.device().is_privateuseone(), "query_position_rows must be on NPU.");
    TORCH_CHECK(attention_indices_out.device().is_privateuseone(), "attention_indices_out must be on NPU.");
    TORCH_CHECK(selection_kv_block_table.dim() == 2, "selection_kv_block_table must be [batch, blocks].");
    TORCH_CHECK(selection_kv_block_status.dim() == 4, "selection_kv_block_status must be [pool_capacity, 1, 1, topk+1].");
    TORCH_CHECK(req_pool_entries.dim() == 1, "req_pool_entries must be [batch].");
    TORCH_CHECK(row_modes.dim() == 1, "row_modes must be [batch].");
    TORCH_CHECK(budget_lengths.dim() == 1, "budget_lengths must be [batch].");
    TORCH_CHECK(tail_valid_token_counts.dim() == 1, "tail_valid_token_counts must be [batch].");
    TORCH_CHECK(resident_tail_starts.dim() == 1, "resident_tail_starts must be [batch].");
    TORCH_CHECK(query_position_rows.dim() == 2, "query_position_rows must be [batch, positions].");
    TORCH_CHECK(attention_indices_out.dim() == 2 || attention_indices_out.dim() == 3,
                "attention_indices_out must be [batch, width] or [batch, 1, width].");
    TORCH_CHECK(selection_topk_indices.dim() == 4,
                "selection_topk_indices must be [batch, seq, head, topk].");
    TORCH_CHECK(selection_kv_block_table.size(0) == req_pool_entries.size(0),
                "selection_kv_block_table batch must equal req_pool_entries.");
    TORCH_CHECK(selection_topk_indices.size(0) == req_pool_entries.size(0),
                "selection_topk_indices batch must equal req_pool_entries.");
    TORCH_CHECK(row_modes.size(0) == req_pool_entries.size(0),
                "row_modes batch must equal req_pool_entries.");
    TORCH_CHECK(budget_lengths.size(0) == req_pool_entries.size(0),
                "budget_lengths batch must equal req_pool_entries.");
    TORCH_CHECK(tail_valid_token_counts.size(0) == req_pool_entries.size(0),
                "tail_valid_token_counts batch must equal req_pool_entries.");
    TORCH_CHECK(resident_tail_starts.size(0) == req_pool_entries.size(0),
                "resident_tail_starts batch must equal req_pool_entries.");
    TORCH_CHECK(query_position_rows.size(0) == req_pool_entries.size(0),
                "query_position_rows batch must equal req_pool_entries.");
    TORCH_CHECK(attention_indices_out.size(0) == req_pool_entries.size(0),
                "attention_indices_out batch must equal req_pool_entries.");

    EXEC_NPU_CMD(
        aclnnGatherSelectionKvCache,
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        req_pool_entries,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        row_modes,
        budget_lengths,
        tail_valid_token_counts,
        resident_tail_starts,
        query_position_rows,
        attention_indices_out);
}

}  // namespace vllm_ascend

#endif  // GATHER_SELECTION_KV_CACHE_TORCH_ADPT_H
