# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch

from vllm_ascend import dsa_sparse_probe
from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_HEAD_STRIDE,
    DSA_SPARSE_FREE_SLOT_COUNT,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_LOOKUP_SLOT_COUNT,
    DSA_SPARSE_RESIDENT_SLOT_COUNT,
)
from vllm_ascend.ops.dsa_sparse import (
    dsa_sparse_lookup_update,
    dsa_sparse_lookup_update_batch,
)

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
        mtp_enabled: bool = False,
        max_verify_tokens_per_request: int = 1,
        cohort_name: str | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("DSA Sparse block_size must be positive.")
        if mtp_enabled and max_verify_tokens_per_request <= 0:
            raise ValueError(
                "DSA Sparse MTP verify capacity must be positive."
            )
        self.block_size = block_size
        self.lookup_capacity = DSA_SPARSE_LOOKUP_SLOT_COUNT
        self.mtp_enabled = mtp_enabled
        self.transient_region_base = self.lookup_capacity
        self.fallback_zero_slot = self.transient_region_base
        self.verify_staging_base = self.fallback_zero_slot + 1
        self.verify_staging_capacity = (
            max_verify_tokens_per_request if mtp_enabled else 0
        )
        transient_slots = (
            1 + self.verify_staging_capacity
            if mtp_enabled
            else block_size
        )
        self.transient_region_span = (
            (transient_slots + block_size - 1) // block_size
        ) * block_size
        self.request_row_stride = (
            self.lookup_capacity + self.transient_region_span
        )
        if self.lookup_capacity % block_size != 0:
            raise ValueError(
                "DSA Sparse lookup capacity must align to block_size."
            )
        if self.request_row_stride % block_size != 0:
            raise ValueError(
                "DSA Sparse request row stride must align to block_size."
            )
        if (
            mtp_enabled
            and self.verify_staging_base + self.verify_staging_capacity
            > self.request_row_stride
        ):
            raise ValueError(
                "DSA Sparse verify staging exceeds the request row."
            )
        self.hot_stride = self.request_row_stride
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
        self.cohort_name = cohort_name
        self.target_step_id = 0

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
        self.query_start_loc: torch.Tensor | None = None
        self.query_positions: torch.Tensor | None = None

    def initialize_request(
        self,
        pool_entry: int,
        resident_token_ids: Iterable[int] | torch.Tensor | None = None,
    ) -> None:
        index = cast(torch.Tensor, self.index)
        slot_to_index = cast(torch.Tensor, self.slot_to_index)
        self.reset_request(pool_entry)
        if resident_token_ids is None:
            resident_tokens = torch.arange(
                DSA_SPARSE_RESIDENT_SLOT_COUNT,
                dtype=torch.int32,
                device=index.device,
            )
        elif isinstance(resident_token_ids, torch.Tensor):
            resident_tokens = resident_token_ids.to(
                device=index.device,
                dtype=torch.int32,
            ).reshape(-1)
        else:
            resident_tokens = torch.tensor(
                list(resident_token_ids),
                dtype=torch.int32,
                device=index.device,
            )
        resident_count = resident_tokens.numel()
        if resident_count > DSA_SPARSE_RESIDENT_SLOT_COUNT:
            raise ValueError(
                "DSA Sparse resident tokens exceed the fixed resident region."
            )
        if resident_count == 0:
            return
        if bool(
            torch.any(
                (resident_tokens < 0)
                | (resident_tokens >= DSA_SPARSE_INDEX_CAPACITY)
            )
        ):
            raise ValueError(
                "DSA Sparse resident token IDs must fit the lookup index."
            )
        if torch.unique(resident_tokens).numel() != resident_count:
            raise ValueError(
                "DSA Sparse resident token IDs must be unique."
            )
        resident_slots = torch.arange(
            resident_count,
            dtype=torch.int32,
            device=index.device,
        )
        index[pool_entry, resident_tokens.long()] = resident_slots
        slot_to_index[pool_entry, :resident_count].copy_(resident_tokens)

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

    def reset_hot_request(self, pool_entry: int) -> None:
        """Wait for prior store completion and restore the read-only fallback."""

        self.wait_for_store()
        if not self.mtp_enabled:
            return
        fallback_global_slot = (
            pool_entry * self.request_row_stride
            + self.fallback_zero_slot
        )
        for plane in self.hot_main_cache:
            plane.view(-1, *plane.shape[2:])[
                fallback_global_slot
            ].zero_()

    def wait_for_store(self) -> None:
        """Mock IO completes synchronously; real backends override this point."""

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

    def build_main_slot_mapping_batch(
        self,
        req_pool_entries: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.mtp_enabled:
            raise RuntimeError(
                "DSA Sparse batch slot mapping requires MTP configuration."
            )
        self.wait_for_store()
        query_lens = (
            query_start_loc[1:] - query_start_loc[:-1]
        ).to(torch.int64)
        total_queries = query_positions.shape[0]
        request_rows = torch.repeat_interleave(
            req_pool_entries,
            query_lens,
            output_size=total_queries,
        )
        request_starts = torch.repeat_interleave(
            query_start_loc[:-1],
            query_lens,
            output_size=total_queries,
        )
        query_offsets = (
            torch.arange(
                total_queries,
                dtype=torch.int32,
                device=query_positions.device,
            )
            - request_starts
        )
        return (
            request_rows * self.request_row_stride
            + self.verify_staging_base
            + query_offsets
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
        self.query_start_loc = None
        self.query_positions = None

    def resolve_batch(
        self,
        topk_indices: torch.Tensor,
        req_pool_entries: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        if not self.mtp_enabled:
            raise RuntimeError(
                "DSA Sparse batch lookup requires MTP configuration."
            )
        index = cast(torch.Tensor, self.index)
        slot_to_index = cast(torch.Tensor, self.slot_to_index)
        free_slots = cast(torch.Tensor, self.free_slots)
        free_head = cast(torch.Tensor, self.free_head)
        query_index = topk_indices.squeeze(1).to(torch.int32).contiguous()
        query_lens = (
            query_start_loc[1:] - query_start_loc[:-1]
        ).to(torch.int64)
        total_queries = query_index.shape[0]
        request_indices = torch.repeat_interleave(
            torch.arange(
                req_pool_entries.shape[0],
                dtype=torch.int32,
                device=req_pool_entries.device,
            ),
            query_lens,
            output_size=total_queries,
        )
        verify_start_positions = query_positions[
            query_start_loc[:-1].long()
        ]
        query_verify_starts = verify_start_positions[
            request_indices.long()
        ].to(torch.int32)
        current_positions = query_positions.to(torch.int32)
        valid_mask = (
            (query_index >= 0)
            & (query_index < DSA_SPARSE_INDEX_CAPACITY)
        )
        history_mask = valid_mask & (
            query_index < query_verify_starts.unsqueeze(1)
        )
        verify_staging_mask = (
            valid_mask
            & (query_index >= query_verify_starts.unsqueeze(1))
            & (query_index <= current_positions.unsqueeze(1))
        )
        lookup_mask = history_mask.to(torch.int32).contiguous()
        slot_out, miss_out = dsa_sparse_lookup_update_batch(
            index,
            slot_to_index,
            free_slots,
            free_head,
            req_pool_entries,
            query_start_loc,
            query_index,
            lookup_mask,
        )
        verify_staging_slots = (
            self.verify_staging_base
            + query_index
            - query_verify_starts.unsqueeze(1)
        )
        mapped_slots = torch.where(
            verify_staging_mask,
            verify_staging_slots,
            torch.where(
                history_mask,
                slot_out,
                torch.full_like(slot_out, INVALID_INDEX),
            ),
        )
        hot_block_offsets = torch.arange(
            self.hot_blocks_per_request,
            dtype=torch.int32,
            device=req_pool_entries.device,
        )
        self.query_index = query_index
        self.slot_out = slot_out
        self.miss_out = miss_out
        self.attention_indices = mapped_slots
        self.hot_block_table = (
            req_pool_entries.unsqueeze(1) * self.hot_blocks_per_request
            + hot_block_offsets
        ).contiguous()
        self.req_pool_entries = req_pool_entries
        self.query_start_loc = query_start_loc
        self.query_positions = query_positions

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
        self.query_start_loc = leader.query_start_loc
        self.query_positions = leader.query_positions

    def load_initial_resident(
        self,
        *,
        layer_name: str,
        remote_request_id: str,
        pool_entry: int,
        resident_token_ids: list[int],
    ) -> None:
        """Initial P/D resident-load boundary; current backend is a mock."""

        if not dsa_sparse_probe.is_enabled():
            return
        resident_count = len(resident_token_ids)
        row_begin = int(pool_entry) * self.request_row_stride
        dsa_sparse_probe.emit(
            "initial_resident_load_mock",
            layer=layer_name,
            cohort=self.cohort_name,
            remote_request_id=remote_request_id,
            pool_entry=int(pool_entry),
            resident_count=resident_count,
            destination_slot_range=[row_begin, row_begin + resident_count],
        )

    def load_history_misses(self, layer_name: str) -> None:
        """Tensor-native history load point; mock performs no payload I/O."""

        if not dsa_sparse_probe.is_enabled():
            return
        miss_out = cast(torch.Tensor, self.miss_out)
        slot_out = cast(torch.Tensor, self.slot_out)
        dsa_sparse_probe.synchronize_device()
        query_lens: list[int] | None = None
        if self.query_start_loc is not None:
            query_lens = (
                self.query_start_loc[1:] - self.query_start_loc[:-1]
            ).detach().cpu().tolist()
        dsa_sparse_probe.emit(
            "history_load_mock",
            target_step_id=self.target_step_id,
            layer=layer_name,
            cohort=self.cohort_name,
            q_i=query_lens,
            history_miss_count=int(miss_out.sum().item()),
            fallback_overflow_count=(
                int(slot_out.eq(self.fallback_zero_slot).sum().item())
                if self.mtp_enabled
                else 0
            ),
        )

    def store_accepted(
        self,
        layer_name: str,
        accepted_input_kv_count: torch.Tensor,
    ) -> None:
        """Tensor-native accepted-prefix store point for the mock backend."""

        if self.query_start_loc is None or self.query_positions is None:
            raise RuntimeError(
                "DSA Sparse accepted store requires an active MTP plan."
            )
        if not dsa_sparse_probe.is_enabled():
            return

        dsa_sparse_probe.synchronize_device()
        query_start_loc = self.query_start_loc.detach().cpu().tolist()
        query_positions = self.query_positions.detach().cpu().tolist()
        accepted = accepted_input_kv_count.detach().cpu().tolist()
        req_pool_entries = cast(
            torch.Tensor,
            self.req_pool_entries,
        ).detach().cpu().tolist()
        query_lens = [
            end - begin
            for begin, end in zip(
                query_start_loc,
                query_start_loc[1:],
            )
        ]
        committed_position_ranges = []
        staging_source_slot_ranges = []
        for pool_entry, query_begin, count in zip(
            req_pool_entries,
            query_start_loc,
            accepted,
        ):
            if count == 0:
                committed_position_ranges.append(None)
            else:
                committed_position_ranges.append(
                    [
                        query_positions[query_begin],
                        query_positions[query_begin + count - 1] + 1,
                    ]
                )
            staging_begin = (
                pool_entry * self.request_row_stride
                + self.verify_staging_base
            )
            staging_source_slot_ranges.append(
                [staging_begin, staging_begin + count]
            )
        dsa_sparse_probe.emit(
            "accepted_store_mock",
            target_step_id=self.target_step_id,
            layer=layer_name,
            cohort=self.cohort_name,
            req_pool_entries=req_pool_entries,
            q_i=query_lens,
            accepted_input_kv_count=accepted,
            committed_kv_count=sum(accepted),
            committed_position_ranges=committed_position_ranges,
            staging_source_slot_ranges=staging_source_slot_ranges,
        )


__all__ = ["DSASparseCoordinator"]
