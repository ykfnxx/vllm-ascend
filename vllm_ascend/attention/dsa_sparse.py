# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch

from vllm_ascend import dsa_sparse_probe
from vllm_ascend.dsa_sparse_backend import (
    DSASparseKVBackend,
    DSASparseStorageKeyEncoder,
)
from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_HEAD_STRIDE,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_KV_TRANSFER_ALIGNMENT,
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
        layer_id: int = 0,
        backend: DSASparseKVBackend | None = None,
        storage_key_encoder: DSASparseStorageKeyEncoder | None = None,
        align_cache_for_kv_transfer: bool = False,
    ) -> None:
        if block_size <= 0:
            raise ValueError("DSA Sparse block_size must be positive.")
        if mtp_enabled and max_verify_tokens_per_request <= 0:
            raise ValueError("DSA Sparse MTP verify capacity must be positive.")
        self.block_size = block_size
        self.layer_id = int(layer_id)
        self.backend = backend
        self.storage_key_encoder = storage_key_encoder
        self.lookup_capacity = DSA_SPARSE_LOOKUP_SLOT_COUNT
        self.mtp_enabled = mtp_enabled
        self.tail_base = self.lookup_capacity
        self.tail_span = block_size
        self.transient_region_base = self.tail_base + self.tail_span
        self.fallback_zero_slot = self.transient_region_base
        self.verify_staging_base = self.fallback_zero_slot + 1
        self.verify_staging_capacity = max_verify_tokens_per_request if mtp_enabled else 0
        transient_slots = 1 + self.verify_staging_capacity
        self.transient_region_span = (
            ((transient_slots + block_size - 1) // block_size) * block_size if mtp_enabled else 0
        )
        self.request_row_stride = self.lookup_capacity + self.tail_span + self.transient_region_span
        if self.lookup_capacity % block_size != 0:
            raise ValueError("DSA Sparse lookup capacity must align to block_size.")
        if self.request_row_stride % block_size != 0:
            raise ValueError("DSA Sparse request row stride must align to block_size.")
        if mtp_enabled and self.verify_staging_base + self.verify_staging_capacity > self.request_row_stride:
            raise ValueError("DSA Sparse verify staging exceeds the request row.")
        self.hot_stride = self.request_row_stride
        self.hot_blocks_per_request = self.hot_stride // block_size
        total_hot_blocks = max_num_seqs * self.hot_blocks_per_request
        self.hot_main_cache = tuple(
            self._allocate_hot_cache_plane(
                (total_hot_blocks, block_size, *row_shape),
                dtype=dtype,
                device=device,
                align_for_kv_transfer=align_cache_for_kv_transfer,
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
        self.query_request_rows: torch.Tensor | None = None
        max_full_blocks = DSA_SPARSE_INDEX_CAPACITY // block_size
        self.storage_ids = torch.full(
            (max_num_seqs, max_full_blocks),
            INVALID_INDEX,
            dtype=torch.int64,
            device=device,
        )
        self._committed_block_counts = [0] * max_num_seqs
        self._put_logical_blocks: list[set[int]] = [set() for _ in range(max_num_seqs)]

    @staticmethod
    def _allocate_hot_cache_plane(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        align_for_kv_transfer: bool,
    ) -> torch.Tensor:
        if not align_for_kv_transfer:
            return torch.empty(shape, dtype=dtype, device=device)

        alignment = DSA_SPARSE_KV_TRANSFER_ALIGNMENT
        element_size = torch.empty((), dtype=dtype).element_size()
        assert alignment % element_size == 0
        numel = 1
        for dimension in shape:
            numel *= dimension
        raw = torch.empty(
            numel + alignment // element_size,
            dtype=dtype,
            device=device,
        )
        offset_bytes = (-raw.data_ptr()) % alignment
        assert offset_bytes % element_size == 0
        offset = offset_bytes // element_size
        cache = raw[offset : offset + numel].view(shape)
        assert cache.data_ptr() % alignment == 0
        return cache

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
            raise ValueError("DSA Sparse resident tokens exceed the fixed resident region.")
        if resident_count == 0:
            return
        if bool(torch.any((resident_tokens < 0) | (resident_tokens >= DSA_SPARSE_INDEX_CAPACITY))):
            raise ValueError("DSA Sparse resident token IDs must fit the lookup index.")
        if torch.unique(resident_tokens).numel() != resident_count:
            raise ValueError("DSA Sparse resident token IDs must be unique.")
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
        self.storage_ids[pool_entry].fill_(INVALID_INDEX)
        self._committed_block_counts[pool_entry] = 0
        self._put_logical_blocks[pool_entry].clear()
        tail_begin = pool_entry * self.request_row_stride + self.tail_base
        tail_end = tail_begin + self.block_size
        for plane in self.hot_main_cache:
            plane.view(-1, *plane.shape[2:])[tail_begin:tail_end].zero_()
        if not self.mtp_enabled:
            return
        fallback_global_slot = pool_entry * self.request_row_stride + self.fallback_zero_slot
        for plane in self.hot_main_cache:
            plane.view(-1, *plane.shape[2:])[fallback_global_slot].zero_()

    def wait_for_store(self) -> None:
        """All supported backends complete PUT before returning."""

    def build_main_slot_mapping(
        self,
        req_pool_entries: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        query_positions = seq_lens - 1
        tail_offsets = torch.remainder(query_positions, self.block_size)
        return (req_pool_entries * self.hot_stride + self.tail_base + tail_offsets).to(torch.int32).contiguous()

    def build_main_slot_mapping_batch(
        self,
        req_pool_entries: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.mtp_enabled:
            raise RuntimeError("DSA Sparse batch slot mapping requires MTP configuration.")
        self.wait_for_store()
        query_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64)
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
            (request_rows * self.request_row_stride + self.verify_staging_base + query_offsets)
            .to(torch.int32)
            .contiguous()
        )

    def set_request_block_hashes(
        self,
        pool_entry: int,
        committed_block_hashes: list[bytes | int] | tuple[bytes | int, ...],
        candidate_block_hashes: list[bytes | int] | tuple[bytes | int, ...] = (),
    ) -> None:
        if self.storage_key_encoder is None:
            return
        committed_ids = self.storage_key_encoder.encode_many(
            committed_block_hashes,
            self.layer_id,
            device=self.storage_ids.device,
        )
        candidate_ids = self.storage_key_encoder.encode_many(
            candidate_block_hashes,
            self.layer_id,
            device=self.storage_ids.device,
        )
        storage_ids = torch.cat((committed_ids, candidate_ids))
        if storage_ids.numel() > self.storage_ids.shape[1]:
            raise ValueError("DSA Sparse block hashes exceed index capacity")
        committed_count = committed_ids.numel()
        current = self.storage_ids[pool_entry]
        old_committed_count = self._committed_block_counts[pool_entry]
        if committed_count < old_committed_count:
            raise RuntimeError("DSA Sparse committed block hashes moved backwards")
        validate_count = min(old_committed_count, committed_count)
        if bool(torch.any(current[:validate_count].ne(committed_ids[:validate_count]))):
            raise RuntimeError("DSA Sparse committed block hash changed")
        for logical_block_idx in self._put_logical_blocks[pool_entry]:
            if logical_block_idx < committed_count and bool(
                current[logical_block_idx].ne(committed_ids[logical_block_idx])
            ):
                raise RuntimeError(
                    "DSA Sparse accepted candidate block hash disagrees with the committed scheduler hash"
                )
        current[committed_count:].fill_(INVALID_INDEX)
        current[: storage_ids.numel()].copy_(storage_ids)
        self._committed_block_counts[pool_entry] = committed_count

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
        tail_mask = valid_mask & (query_index >= dense_tail_starts.unsqueeze(1))
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
        tail_slots = DSA_SPARSE_LOOKUP_SLOT_COUNT + query_index - dense_tail_starts.unsqueeze(1)
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
            req_pool_entries.unsqueeze(1) * self.hot_blocks_per_request + hot_block_offsets
        ).contiguous()
        self.req_pool_entries = req_pool_entries
        self.query_start_loc = None
        self.query_positions = (seq_lens - 1).to(torch.int32)
        self.query_request_rows = req_pool_entries

    def resolve_batch(
        self,
        topk_indices: torch.Tensor,
        req_pool_entries: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        if not self.mtp_enabled:
            raise RuntimeError("DSA Sparse batch lookup requires MTP configuration.")
        index = cast(torch.Tensor, self.index)
        slot_to_index = cast(torch.Tensor, self.slot_to_index)
        free_slots = cast(torch.Tensor, self.free_slots)
        free_head = cast(torch.Tensor, self.free_head)
        query_index = topk_indices.squeeze(1).to(torch.int32).contiguous()
        query_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64)
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
        verify_start_positions = query_positions[query_start_loc[:-1].long()]
        query_verify_starts = verify_start_positions[request_indices.long()].to(torch.int32)
        live_tail_starts = (
            torch.div(
                verify_start_positions,
                self.block_size,
                rounding_mode="floor",
            )
            * self.block_size
        ).to(torch.int32)
        query_live_tail_starts = live_tail_starts[request_indices.long()]
        current_positions = query_positions.to(torch.int32)
        valid_mask = (query_index >= 0) & (query_index < DSA_SPARSE_INDEX_CAPACITY)
        history_mask = valid_mask & (query_index < query_live_tail_starts.unsqueeze(1))
        tail_mask = (
            valid_mask
            & (query_index >= query_live_tail_starts.unsqueeze(1))
            & (query_index < query_verify_starts.unsqueeze(1))
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
        verify_staging_slots = self.verify_staging_base + query_index - query_verify_starts.unsqueeze(1)
        tail_slots = self.tail_base + query_index - query_live_tail_starts.unsqueeze(1)
        mapped_slots = torch.where(
            verify_staging_mask,
            verify_staging_slots,
            torch.where(
                tail_mask,
                tail_slots,
                torch.where(
                    history_mask,
                    slot_out,
                    torch.full_like(slot_out, INVALID_INDEX),
                ),
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
            req_pool_entries.unsqueeze(1) * self.hot_blocks_per_request + hot_block_offsets
        ).contiguous()
        self.req_pool_entries = req_pool_entries
        self.query_start_loc = query_start_loc
        self.query_positions = query_positions
        self.query_request_rows = req_pool_entries[request_indices.long()]

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
        self.query_request_rows = leader.query_request_rows

    def load_initial_resident(
        self,
        *,
        layer_name: str,
        remote_request_id: str,
        pool_entry: int,
        resident_token_ids: list[int],
    ) -> None:
        """Load the selected historical tokens into this physical layer."""

        if resident_token_ids and self.backend is not None:
            token_ids = torch.tensor(
                resident_token_ids,
                dtype=torch.int64,
                device=self.storage_ids.device,
            )
            logical_blocks = torch.div(
                token_ids,
                self.block_size,
                rounding_mode="floor",
            )
            storage_request_ids = self.storage_ids[int(pool_entry), logical_blocks]
            if bool(torch.any(storage_request_ids < 0)):
                raise RuntimeError("DSA Sparse initial resident token has no block hash")
            destination_slots = int(pool_entry) * self.request_row_stride + torch.arange(
                token_ids.numel(),
                dtype=torch.int64,
                device=token_ids.device,
            )
            self.backend.load_tokens_into(
                layer_id=self.layer_id,
                storage_request_ids=storage_request_ids,
                token_offsets_in_block=torch.remainder(token_ids, self.block_size),
                destination_physical_slots=destination_slots,
            )

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
        """Load only lookup misses for this physical layer."""

        miss_out = cast(torch.Tensor, self.miss_out)
        slot_out = cast(torch.Tensor, self.slot_out)
        query_index = cast(torch.Tensor, self.query_index)
        request_rows = cast(torch.Tensor, self.query_request_rows)
        active = miss_out.bool() & (slot_out >= 0) & (slot_out < self.lookup_capacity)
        if self.backend is not None and bool(torch.any(active)):
            expanded_rows = request_rows.unsqueeze(1).expand_as(query_index)
            token_ids = query_index[active].to(torch.int64)
            rows = expanded_rows[active].to(torch.int64)
            logical_blocks = torch.div(
                token_ids,
                self.block_size,
                rounding_mode="floor",
            )
            storage_request_ids = self.storage_ids[rows, logical_blocks]
            if bool(torch.any(storage_request_ids < 0)):
                raise RuntimeError("DSA Sparse history miss has no committed block hash")
            destination_slots = rows * self.request_row_stride + slot_out[active].to(torch.int64)
            self.backend.load_tokens_into(
                layer_id=self.layer_id,
                storage_request_ids=storage_request_ids,
                token_offsets_in_block=torch.remainder(token_ids, self.block_size),
                destination_physical_slots=destination_slots,
            )

        if not dsa_sparse_probe.is_enabled():
            return
        dsa_sparse_probe.synchronize_device()
        query_lens: list[int] | None = None
        if self.query_start_loc is not None:
            query_lens = (self.query_start_loc[1:] - self.query_start_loc[:-1]).detach().cpu().tolist()
        dsa_sparse_probe.emit(
            "history_load_mock",
            target_step_id=self.target_step_id,
            layer=layer_name,
            cohort=self.cohort_name,
            q_i=query_lens,
            history_miss_count=int(miss_out.sum().item()),
            fallback_overflow_count=(int(slot_out.eq(self.fallback_zero_slot).sum().item()) if self.mtp_enabled else 0),
        )

    def _put_filled_tail(
        self,
        pool_entry: int,
        logical_block_idx: int,
    ) -> None:
        if self.backend is None:
            return
        storage_request_id = self.storage_ids[int(pool_entry), int(logical_block_idx)].reshape(1)
        if bool(torch.any(storage_request_id < 0)):
            raise RuntimeError("DSA Sparse filled tail has no block hash")
        source_block_id = torch.tensor(
            [int(pool_entry) * self.hot_blocks_per_request + self.tail_base // self.block_size],
            dtype=torch.int64,
            device=self.storage_ids.device,
        )
        self.backend.put_blocks(
            layer_id=self.layer_id,
            storage_request_ids=storage_request_id,
            source_block_ids=source_block_id,
        )
        self._put_logical_blocks[int(pool_entry)].add(int(logical_block_idx))

    def commit_decode_tail(
        self,
        layer_name: str,
        pool_entries: list[int],
        query_positions: list[int],
    ) -> None:
        if self.query_start_loc is not None:
            raise RuntimeError("DSA Sparse normal decode has an MTP plan")
        if len(pool_entries) != len(query_positions):
            raise ValueError("DSA Sparse Decode tail commit vectors are not aligned")
        for pool_entry, position in zip(pool_entries, query_positions):
            if (int(position) + 1) % self.block_size == 0:
                self._put_filled_tail(int(pool_entry), int(position) // self.block_size)
                if dsa_sparse_probe.is_enabled():
                    dsa_sparse_probe.emit(
                        "tail_block_put",
                        layer=layer_name,
                        pool_entry=int(pool_entry),
                        logical_block_idx=int(position) // self.block_size,
                    )

    def commit_accepted_to_tail(
        self,
        layer_name: str,
        query_start_loc: list[int],
        query_positions: list[int],
        accepted_input_kv_count: list[int],
        req_pool_entries: list[int],
    ) -> None:
        """Copy only accepted verify KV to tail and PUT completed blocks."""

        if self.query_start_loc is None or self.query_positions is None:
            raise RuntimeError("DSA Sparse accepted commit requires an active MTP plan.")
        query_lens = [
            end - begin
            for begin, end in zip(
                query_start_loc,
                query_start_loc[1:],
            )
        ]
        if (
            len(accepted_input_kv_count) != len(query_lens)
            or len(req_pool_entries) != len(query_lens)
            or query_start_loc[-1] != len(query_positions)
            or any(count < 0 or count > query_len for count, query_len in zip(accepted_input_kv_count, query_lens))
        ):
            raise ValueError("DSA Sparse accepted count exceeds query length")
        committed_position_ranges = []
        staging_source_slot_ranges = []
        for pool_entry, query_begin, count in zip(
            req_pool_entries,
            query_start_loc,
            accepted_input_kv_count,
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
            staging_begin = pool_entry * self.request_row_stride + self.verify_staging_base
            staging_source_slot_ranges.append([staging_begin, staging_begin + count])
            copied = 0
            while copied < count:
                position = int(query_positions[query_begin + copied])
                tail_offset = position % self.block_size
                copy_count = min(count - copied, self.block_size - tail_offset)
                source_begin = staging_begin + copied
                destination_begin = int(pool_entry) * self.request_row_stride + self.tail_base + tail_offset
                for plane in self.hot_main_cache:
                    flat = plane.view(-1, *plane.shape[2:])
                    flat[destination_begin : destination_begin + copy_count].copy_(
                        flat[source_begin : source_begin + copy_count]
                    )
                copied += copy_count
                if tail_offset + copy_count == self.block_size:
                    self._put_filled_tail(int(pool_entry), position // self.block_size)
        if dsa_sparse_probe.is_enabled():
            dsa_sparse_probe.emit(
                "accepted_tail_commit",
                target_step_id=self.target_step_id,
                layer=layer_name,
                cohort=self.cohort_name,
                req_pool_entries=req_pool_entries,
                q_i=query_lens,
                accepted_input_kv_count=accepted_input_kv_count,
                committed_kv_count=sum(accepted_input_kv_count),
                committed_position_ranges=committed_position_ranges,
                staging_source_slot_ranges=staging_source_slot_ranges,
            )


__all__ = ["DSASparseCoordinator"]
