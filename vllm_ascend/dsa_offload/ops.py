# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

import torch


@dataclass
class LookupState:
    index: torch.Tensor
    slot_to_index: torch.Tensor
    free_slots: torch.Tensor
    free_head: torch.Tensor


LookupOutput = tuple[torch.Tensor, torch.Tensor]


def lookup_update(
    state: LookupState,
    request_rows: torch.Tensor,
    query_indices: torch.Tensor,
    lookup_mask: torch.Tensor,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_offload_lookup_update(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_indices,
        lookup_mask,
        request_rows.shape[0],
    )


def lookup_update_batch(
    state: LookupState,
    request_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_indices: torch.Tensor,
    lookup_mask: torch.Tensor,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_offload_lookup_update_batch(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_start_loc,
        query_indices,
        lookup_mask,
        request_rows.shape[0],
    )


def resolve_update_batch_v2(
    state: LookupState,
    request_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_positions: torch.Tensor,
    semantic_topk: torch.Tensor,
    mapped_indices_out: torch.Tensor,
    gather_mask_out: torch.Tensor,
    block_size: int,
    decode_mode: int,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_offload_resolve_update_batch_v2(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_start_loc,
        query_positions,
        semantic_topk,
        mapped_indices_out,
        gather_mask_out,
        request_rows.shape[0],
        block_size,
        decode_mode,
    )


def turbo_resolve_update_batch_v2(
    state: LookupState,
    request_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_positions: torch.Tensor,
    semantic_topk: torch.Tensor,
    mapped_indices_out: torch.Tensor,
    gather_mask_out: torch.Tensor,
    block_size: int,
    decode_mode: int,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_sparse_turbo_resolve_update_batch_v2(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_start_loc,
        query_positions,
        semantic_topk,
        mapped_indices_out,
        gather_mask_out,
        request_rows.shape[0],
        block_size,
        decode_mode,
    )


def asu_kv_gather(
    destination_kv_cache: torch.Tensor,
    destination_k_rope: torch.Tensor,
    destination_block_table: torch.Tensor,
    source_kv_cache: torch.Tensor,
    source_k_rope: torch.Tensor,
    source_block_table: torch.Tensor,
    request_rows: torch.Tensor,
    token_positions: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_mask: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_ascend.asu_kv_gather(
        destination_kv_cache,
        destination_k_rope,
        destination_block_table,
        source_kv_cache,
        source_k_rope,
        source_block_table,
        request_rows,
        token_positions,
        destination_slots,
        miss_mask,
        block_size,
        request_rows.shape[0],
    )


def asu_kv_gather_direct_v2(
    destination_kv_cache: torch.Tensor,
    destination_k_rope: torch.Tensor,
    hot_block_table_pool: torch.Tensor,
    source_kv_cache: torch.Tensor,
    source_k_rope: torch.Tensor,
    source_block_table: torch.Tensor,
    request_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    semantic_topk: torch.Tensor,
    mapped_indices: torch.Tensor,
    gather_mask: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_ascend.asu_kv_gather_direct_v2(
        destination_kv_cache,
        destination_k_rope,
        hot_block_table_pool,
        source_kv_cache,
        source_k_rope,
        source_block_table,
        request_rows,
        query_start_loc,
        semantic_topk,
        mapped_indices,
        gather_mask,
        block_size,
        request_rows.shape[0],
    )


def turbo_lookup_update_batch(
    state: LookupState,
    request_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_indices: torch.Tensor,
    lookup_mask: torch.Tensor,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_sparse_turbo_lookup_update_batch(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_start_loc,
        query_indices,
        lookup_mask,
        request_rows.shape[0],
    )


def turbo_prefetch_lookup_update_batch(
    state: LookupState,
    request_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_indices: torch.Tensor,
    lookup_mask: torch.Tensor,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_sparse_turbo_prefetch_lookup_update_batch(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_start_loc,
        query_indices,
        lookup_mask,
        request_rows.shape[0],
    )
