"""NPU tensor backend for DSA lookup-resident cache management."""

from __future__ import annotations

from typing import NamedTuple

import torch
from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_kv_backend import (
    DSAKVBackend, compose_dsa_storage_request_ids)
from vllm_ascend.dsa_sparse.dsa_resident_pool import DSAResidentLookupState
from vllm_ascend.dsa_sparse.dsa_types import DSADecodeRowMode

logger = init_logger("vllm.dsa_sparse")


class DSALookupOutput(NamedTuple):
    attention_indices: torch.Tensor


class AscendDSAOpsBackend:
    """Map full-sequence Indexer TopK into resident attention slots.

    Dumped-history token ids use the persistent token-to-slot index; live-tail
    ids map directly into the independent tail block. The KV backend writes
    only historical misses into resident HBM, then AICPU maintenance restores
    the free-slot headroom for the next decode step.
    """

    def __init__(self) -> None:
        self._lookup_call_logged = False
        self._maintain_call_logged = False
        self._resident_init_logged = False

    @staticmethod
    def _normalize_topk(topk: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
        topk = topk.to(device=device, dtype=torch.int32).contiguous()
        if topk.ndim < 2 or topk.ndim > 4:
            raise ValueError(
                "selection_topk_indices must have rank 2, 3, or 4, got "
                f"{tuple(topk.shape)}")
        return topk.reshape(int(topk.shape[0]), -1)

    @staticmethod
    def _as_device_i32(values, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(values):
            return values.to(device=device, dtype=torch.int32).contiguous()
        return torch.tensor(values, dtype=torch.int32, device=device)

    def _initialize_resident_rows(
        self,
        *,
        layer_id: int,
        kv_backend: DSAKVBackend,
        state: DSAResidentLookupState,
        pool_entries: torch.Tensor,
        block_hash_ids: torch.Tensor,
        initialize_rows: torch.Tensor,
        resident_tokens: int,
        selection_block_table: torch.Tensor,
        block_size: int,
    ) -> None:
        pool_rows = pool_entries.to(dtype=torch.long)
        batch_size = int(pool_rows.numel())
        tokens = torch.arange(
            resident_tokens,
            dtype=torch.int32,
            device=pool_entries.device,
        ).view(1, -1).expand(batch_size, -1)
        slots = tokens
        init_mask = initialize_rows.view(-1, 1).expand(-1, resident_tokens)

        self._load_selected_tokens(
            kv_backend=kv_backend,
            layer_id=int(layer_id),
            block_hash_ids=block_hash_ids,
            token_positions=tokens,
            destination_slots=slots,
            load_mask=init_mask,
            destination_block_table=selection_block_table,
            block_size=int(block_size),
        )
        init_batch_rows, init_token_positions = init_mask.nonzero(as_tuple=True)
        init_pool_rows = pool_rows.index_select(0, init_batch_rows)
        init_tokens = tokens[init_batch_rows, init_token_positions].to(
            dtype=torch.long)
        init_slots = slots[init_batch_rows, init_token_positions].to(
            dtype=torch.long)
        state.token_to_slot[init_pool_rows, init_tokens] = init_slots.to(
            dtype=torch.int32)
        state.slot_to_token[init_pool_rows, init_slots] = init_tokens.to(
            dtype=torch.int32)

    @staticmethod
    def _load_selected_tokens(
        *,
        kv_backend: DSAKVBackend,
        layer_id: int,
        block_hash_ids: torch.Tensor,
        token_positions: torch.Tensor,
        destination_slots: torch.Tensor,
        load_mask: torch.Tensor,
        destination_block_table: torch.Tensor,
        block_size: int,
    ) -> None:
        row_indices, token_indices = load_mask.to(
            dtype=torch.bool).nonzero(as_tuple=True)
        if int(row_indices.numel()) == 0:
            return
        token_positions = token_positions[
            row_indices, token_indices].to(dtype=torch.long).contiguous()
        source_logical_blocks = torch.div(
            token_positions, int(block_size), rounding_mode="floor")
        token_offsets_in_block = torch.remainder(
            token_positions, int(block_size)).contiguous()
        selected_block_hash_ids = block_hash_ids[
            row_indices, source_logical_blocks].to(dtype=torch.long)
        storage_request_ids = compose_dsa_storage_request_ids(
            selected_block_hash_ids,
            int(layer_id),
        )

        destination_slots = destination_slots[
            row_indices, token_indices].to(dtype=torch.long).contiguous()
        destination_logical_blocks = torch.div(
            destination_slots, int(block_size), rounding_mode="floor")
        destination_block_offsets = torch.remainder(
            destination_slots, int(block_size))
        destination_physical_blocks = destination_block_table[
            row_indices, destination_logical_blocks].to(dtype=torch.long)
        destination_physical_slots = (
            destination_physical_blocks * int(block_size)
            + destination_block_offsets).contiguous()

        kv_backend.load_tokens_into(
            layer_id=int(layer_id),
            storage_request_ids=storage_request_ids,
            token_offsets_in_block=token_offsets_in_block,
            destination_physical_slots=destination_physical_slots,
        )

    @staticmethod
    def _compose_attention_indices(
        *,
        topk: torch.Tensor,
        mapped_slots: torch.Tensor,
        row_modes: torch.Tensor,
        attention_indices: torch.Tensor,
    ) -> None:
        batch_size, topk_width = topk.shape
        width = int(attention_indices.shape[1])
        padded_topk = torch.full(
            (batch_size, width), -1, dtype=torch.int32, device=topk.device)
        padded_mapped_slots = torch.full_like(padded_topk, -1)
        copy_width = min(topk_width, width)
        padded_topk[:, :copy_width].copy_(topk[:, :copy_width])
        padded_mapped_slots[:, :copy_width].copy_(
            mapped_slots[:, :copy_width])

        sparse_rows = row_modes == int(DSADecodeRowMode.SPARSE)
        dense_rows = row_modes == int(DSADecodeRowMode.DENSE)
        attention_indices.copy_(torch.where(
            dense_rows.view(-1, 1),
            padded_topk,
            torch.where(sparse_rows.view(-1, 1), padded_mapped_slots,
                        torch.full_like(padded_topk, -1)),
        ))

    def lookup_resident_update(
        self,
        *,
        layer_id: int,
        kv_backend: DSAKVBackend,
        selection_topk_indices: torch.Tensor,
        req_pool_entries: torch.Tensor,
        block_hash_ids: torch.Tensor,
        sparse_local_row_indices: torch.Tensor,
        selection_block_table: torch.Tensor,
        block_size: int,
        lookup_state: DSAResidentLookupState,
        resident_tokens: int,
        dense_tail_starts: torch.Tensor,
        resident_tail_starts: torch.Tensor,
        attention_indices_width: int,
        prebuilt_attention_indices: torch.Tensor | None = None,
        row_modes: torch.Tensor,
        lookup_init_mask: torch.Tensor,
        has_lookup_init_rows: bool,
        maintain_seed: int,
    ) -> DSALookupOutput:
        device = selection_block_table.device
        topk = self._normalize_topk(selection_topk_indices, device)
        pool_entries = self._as_device_i32(
            req_pool_entries, device).reshape(-1)
        block_hash_ids = block_hash_ids.to(
            device=device, dtype=torch.long).contiguous()
        if (block_hash_ids.ndim != 2
                or int(block_hash_ids.shape[0]) != int(pool_entries.numel())):
            raise ValueError(
                "DSA block hash ids must align with request pool rows")
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError("DSA block size must be positive")
        row_modes = self._as_device_i32(row_modes, device).reshape(-1)
        lookup_init_mask = lookup_init_mask.to(
            device=device, dtype=torch.bool).reshape(-1)
        selection_block_table = self._as_device_i32(
            selection_block_table, device)
        dense_tail_starts = self._as_device_i32(
            dense_tail_starts, device).reshape(-1)
        resident_tail_starts = self._as_device_i32(
            resident_tail_starts, device).reshape(-1)
        sparse_local_rows = sparse_local_row_indices.to(
            device=device, dtype=torch.long).reshape(-1)

        if has_lookup_init_rows:
            if not self._resident_init_logged:
                logger.info(
                    "DSA sparse invoking resident initialization: "
                    "requests=%d, resident_tokens=%d",
                    int(pool_entries.numel()),
                    int(resident_tokens),
                )
            self._initialize_resident_rows(
                layer_id=int(layer_id),
                kv_backend=kv_backend,
                state=lookup_state,
                pool_entries=pool_entries,
                block_hash_ids=block_hash_ids,
                initialize_rows=lookup_init_mask,
                resident_tokens=int(resident_tokens),
                selection_block_table=selection_block_table,
                block_size=block_size,
            )
            if not self._resident_init_logged:
                logger.info(
                    "DSA sparse completed resident initialization: "
                    "requests=%d, resident_tokens=%d",
                    int(pool_entries.numel()),
                    int(resident_tokens),
                )
                self._resident_init_logged = True
        sparse_topk = topk.index_select(0, sparse_local_rows).contiguous()
        sparse_pool_entries = pool_entries.index_select(
            0, sparse_local_rows).contiguous()
        sparse_block_hash_ids = block_hash_ids.index_select(
            0, sparse_local_rows).contiguous()
        sparse_dense_tail_starts = dense_tail_starts.index_select(
            0, sparse_local_rows).view(-1, 1)
        sparse_resident_tail_starts = resident_tail_starts.index_select(
            0, sparse_local_rows).view(-1, 1)
        sparse_tail_mask = sparse_topk >= sparse_dense_tail_starts
        sparse_history_mask = ~sparse_tail_mask
        sparse_lookup_mask = sparse_history_mask.to(
            dtype=torch.int32).contiguous()
        if not self._lookup_call_logged:
            logger.info(
                "DSA sparse invoking asu_hbm_index_lookup: requests=%d, "
                "query_shape=%s",
                int(sparse_pool_entries.numel()),
                tuple(sparse_topk.shape),
            )
        sparse_slot_out, sparse_miss_out = (
            torch.ops._C_ascend.asu_hbm_index_lookup(
                lookup_state.token_to_slot,
                lookup_state.slot_to_token,
                lookup_state.free_slots,
                lookup_state.free_head,
                sparse_pool_entries,
                sparse_topk,
                sparse_lookup_mask,
                int(sparse_pool_entries.numel()),
            )
        )
        if not self._lookup_call_logged:
            logger.info(
                "DSA sparse completed asu_hbm_index_lookup: requests=%d, "
                "query_shape=%s",
                int(sparse_pool_entries.numel()),
                tuple(sparse_topk.shape),
            )
            self._lookup_call_logged = True
        sparse_misses = (
            sparse_miss_out.to(dtype=torch.bool)
            & sparse_history_mask
        )
        sparse_selection_block_table = selection_block_table.index_select(
            0, sparse_local_rows)
        self._load_selected_tokens(
            kv_backend=kv_backend,
            layer_id=int(layer_id),
            block_hash_ids=sparse_block_hash_ids,
            token_positions=sparse_topk,
            destination_slots=sparse_slot_out,
            load_mask=sparse_misses,
            destination_block_table=sparse_selection_block_table,
            block_size=block_size,
        )
        sparse_tail_slots = (
            sparse_resident_tail_starts
            + sparse_topk
            - sparse_dense_tail_starts
        )
        sparse_mapped_slots = torch.where(
            sparse_tail_mask,
            sparse_tail_slots,
            sparse_slot_out,
        )
        mapped_slots = torch.full_like(topk, -1)
        mapped_slots.index_copy_(
            0, sparse_local_rows, sparse_mapped_slots)

        if torch.is_tensor(prebuilt_attention_indices):
            attention_indices = prebuilt_attention_indices
        else:
            attention_indices = torch.empty(
                (int(pool_entries.numel()), int(attention_indices_width)),
                dtype=torch.int32,
                device=device,
            )
        self._compose_attention_indices(
            topk=topk,
            mapped_slots=mapped_slots,
            row_modes=row_modes,
            attention_indices=attention_indices,
        )
        if not self._maintain_call_logged:
            logger.info(
                "DSA sparse invoking asu_hbm_index_maintain_aicpu: "
                "requests=%d, seed=%d",
                int(sparse_pool_entries.numel()),
                int(maintain_seed),
            )
        torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(
            lookup_state.token_to_slot,
            lookup_state.slot_to_token,
            lookup_state.free_slots,
            lookup_state.free_head,
            sparse_pool_entries,
            sparse_slot_out,
            int(sparse_pool_entries.numel()),
            int(maintain_seed),
        )
        if not self._maintain_call_logged:
            logger.info(
                "DSA sparse completed asu_hbm_index_maintain_aicpu: "
                "requests=%d, seed=%d",
                int(sparse_pool_entries.numel()),
                int(maintain_seed),
            )
            self._maintain_call_logged = True
        return DSALookupOutput(attention_indices=attention_indices)
