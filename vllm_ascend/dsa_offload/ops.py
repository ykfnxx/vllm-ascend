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


def lookup_update_batch(
    state: LookupState,
    request_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_positions: torch.Tensor,
    semantic_topk: torch.Tensor,
    *,
    block_size: int,
    tail_base: int,
    fallback_slot: int,
    staging_base: int,
    decode_mode: int,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_offload_lookup_update_batch(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_start_loc,
        query_positions,
        semantic_topk,
        request_rows.shape[0],
        block_size,
        tail_base,
        fallback_slot,
        staging_base,
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
