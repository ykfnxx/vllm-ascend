# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from typing import cast

import torch

from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_HEAD_STRIDE,
    DSA_SPARSE_FREE_SLOT_COUNT,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_LOOKUP_SLOT_COUNT,
    DSA_SPARSE_RESIDENT_SLOT_COUNT,
)
from vllm_ascend.ops.dsa_sparse import dsa_sparse_lookup_update

INVALID_INDEX = -1


class DSASparseCoordinator:
    """Per-layer Hot Main cache and per-group lookup plan."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        block_size: int,
        plane_layouts: tuple[tuple[torch.dtype, tuple[int, ...]], ...],
        device: torch.device | str,
        leader: DSASparseCoordinator | None = None,
    ) -> None:
        self.block_size = block_size
        self.hot_stride = DSA_SPARSE_LOOKUP_SLOT_COUNT + block_size
        self.hot_blocks_per_request = self.hot_stride // block_size
        total_hot_blocks = max_num_seqs * self.hot_blocks_per_request
        self.hot_main_cache = tuple(
            torch.empty(
                (total_hot_blocks, block_size, *row_shape),
                dtype=dtype,
                device=device,
            )
            for dtype, row_shape in plane_layouts
        )
        self.leader = leader

        if leader is None:
            self.index = torch.full(
                (max_num_seqs, DSA_SPARSE_INDEX_CAPACITY),
                INVALID_INDEX,
                dtype=torch.int32,
                device=device,
            )
            self.slot_to_index = torch.full(
                (max_num_seqs, DSA_SPARSE_LOOKUP_SLOT_COUNT),
                INVALID_INDEX,
                dtype=torch.int32,
                device=device,
            )
            self.free_slots = (
                torch.arange(
                    DSA_SPARSE_RESIDENT_SLOT_COUNT,
                    DSA_SPARSE_LOOKUP_SLOT_COUNT,
                    dtype=torch.int32,
                    device=device,
                )
                .expand(max_num_seqs, -1)
                .clone()
            )
            self.free_head = torch.zeros(
                (max_num_seqs, DSA_SPARSE_FREE_HEAD_STRIDE),
                dtype=torch.int32,
                device=device,
            )
        else:
            self.index = None
            self.slot_to_index = None
            self.free_slots = None
            self.free_head = None

        self.query_index: torch.Tensor | None = None
        self.slot_out: torch.Tensor | None = None
        self.miss_out: torch.Tensor | None = None
        self.attention_indices: torch.Tensor | None = None
        self.hot_block_table: torch.Tensor | None = None
        self.req_pool_entries: torch.Tensor | None = None

    def initialize_request(self, pool_entry: int) -> None:
        index = cast(torch.Tensor, self.index)
        slot_to_index = cast(torch.Tensor, self.slot_to_index)
        self.reset_request(pool_entry)
        resident_tokens = torch.arange(
            DSA_SPARSE_RESIDENT_SLOT_COUNT,
            dtype=torch.int32,
            device=index.device,
        )
        index[pool_entry, :DSA_SPARSE_RESIDENT_SLOT_COUNT].copy_(
            resident_tokens
        )
        slot_to_index[
            pool_entry, :DSA_SPARSE_RESIDENT_SLOT_COUNT
        ].copy_(resident_tokens)

    def reset_request(self, pool_entry: int) -> None:
        index = cast(torch.Tensor, self.index)
        slot_to_index = cast(torch.Tensor, self.slot_to_index)
        free_slots = cast(torch.Tensor, self.free_slots)
        free_head = cast(torch.Tensor, self.free_head)
        index[pool_entry].fill_(INVALID_INDEX)
        slot_to_index[pool_entry].fill_(INVALID_INDEX)
        free_slots[pool_entry].copy_(
            torch.arange(
                DSA_SPARSE_RESIDENT_SLOT_COUNT,
                DSA_SPARSE_LOOKUP_SLOT_COUNT,
                dtype=torch.int32,
                device=free_slots.device,
            )
        )
        free_head[pool_entry].zero_()

    def build_main_slot_mapping(
        self,
        req_pool_entries: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        query_positions = seq_lens - 1
        tail_offsets = torch.remainder(query_positions, self.block_size)
        return (
            req_pool_entries * self.hot_stride
            + DSA_SPARSE_LOOKUP_SLOT_COUNT
            + tail_offsets
        ).to(torch.int32).contiguous()

    def resolve(
        self,
        topk_indices: torch.Tensor,
        req_pool_entries: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        index = cast(torch.Tensor, self.index)
        slot_to_index = cast(torch.Tensor, self.slot_to_index)
        free_slots = cast(torch.Tensor, self.free_slots)
        free_head = cast(torch.Tensor, self.free_head)
        query_index = topk_indices.squeeze(1).to(torch.int32).contiguous()
        dense_tail_starts = (
            torch.div(
                seq_lens - 1,
                self.block_size,
                rounding_mode="floor",
            )
            * self.block_size
        ).to(torch.int32)
        valid_mask = query_index >= 0
        tail_mask = valid_mask & (
            query_index >= dense_tail_starts.unsqueeze(1)
        )
        lookup_mask = (valid_mask & ~tail_mask).to(torch.int32).contiguous()

        slot_out, miss_out = dsa_sparse_lookup_update(
            index,
            slot_to_index,
            free_slots,
            free_head,
            req_pool_entries,
            query_index,
            lookup_mask,
        )
        tail_slots = (
            DSA_SPARSE_LOOKUP_SLOT_COUNT
            + query_index
            - dense_tail_starts.unsqueeze(1)
        )
        mapped_slots = torch.where(tail_mask, tail_slots, slot_out)

        hot_block_offsets = torch.arange(
            self.hot_blocks_per_request,
            dtype=torch.int32,
            device=req_pool_entries.device,
        )
        self.query_index = query_index
        self.slot_out = slot_out
        self.miss_out = miss_out
        self.attention_indices = torch.where(
            valid_mask,
            mapped_slots,
            torch.full_like(mapped_slots, INVALID_INDEX),
        )
        self.hot_block_table = (
            req_pool_entries.unsqueeze(1) * self.hot_blocks_per_request
            + hot_block_offsets
        ).contiguous()
        self.req_pool_entries = req_pool_entries

    def reuse_leader_plan(self, req_pool_entries: torch.Tensor) -> None:
        leader = self.leader
        assert leader is not None
        assert leader.req_pool_entries is req_pool_entries
        self.query_index = leader.query_index
        self.slot_out = leader.slot_out
        self.miss_out = leader.miss_out
        self.attention_indices = leader.attention_indices
        self.hot_block_table = leader.hot_block_table
        self.req_pool_entries = leader.req_pool_entries

    def mock_store_newest(self) -> None:
        """Backing-store write-through point; payload I/O is not implemented."""

    def mock_load_misses(self) -> None:
        """History miss-load point; payload I/O is not implemented."""


__all__ = ["DSASparseCoordinator"]
