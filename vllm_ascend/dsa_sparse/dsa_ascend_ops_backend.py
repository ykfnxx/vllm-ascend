"""NPU tensor backend for DSA lookup-resident cache management."""

from __future__ import annotations

from typing import NamedTuple

import torch
from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_kv_backend import DSAKVBackend
from vllm_ascend.dsa_sparse.dsa_resident_pool import DSAResidentLookupState
from vllm_ascend.dsa_sparse.dsa_types import DSADecodeRowMode

logger = init_logger("vllm.dsa_sparse")


class DSALookupOutput(NamedTuple):
    attention_indices: torch.Tensor


class AscendDSAOpsBackend:
    """Resolve Indexer TopK through a persistent token-to-slot index.

    The AIV lookup operator maps original token ids to persistent resident
    slots and reports actual allocations. The KV backend writes only those
    misses into resident HBM, then the AICPU maintain operator restores the
    free-slot headroom for the next decode step.
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
        initialize_rows: torch.Tensor,
        resident_tokens: int,
        selection_block_table: torch.Tensor,
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

        kv_backend.load_tokens_into(
            layer_id=int(layer_id),
            request_pool_entries=pool_entries,
            token_positions=tokens,
            destination_slots=slots,
            load_mask=init_mask,
            destination_block_table=selection_block_table,
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
    def _compose_attention_indices(
        *,
        topk: torch.Tensor,
        slot_out: torch.Tensor,
        row_modes: torch.Tensor,
        budget_lengths: torch.Tensor,
        tail_valid_token_counts: torch.Tensor,
        resident_tail_starts: torch.Tensor,
        attention_indices: torch.Tensor,
    ) -> None:
        batch_size, topk_width = topk.shape
        width = int(attention_indices.shape[1])
        padded_topk = torch.full(
            (batch_size, width), -1, dtype=torch.int32, device=topk.device)
        padded_slots = torch.full_like(padded_topk, -1)
        copy_width = min(topk_width, width)
        padded_topk[:, :copy_width].copy_(topk[:, :copy_width])
        padded_slots[:, :copy_width].copy_(slot_out[:, :copy_width])

        columns = torch.arange(
            width, dtype=torch.int32, device=topk.device).view(1, -1)
        sparse_rows = row_modes == int(DSADecodeRowMode.SPARSE)
        dense_rows = row_modes == int(DSADecodeRowMode.DENSE)
        budget_mask = columns < budget_lengths.view(-1, 1)
        tail_offsets = columns - budget_lengths.view(-1, 1)
        tail_mask = (
            sparse_rows.view(-1, 1)
            & (tail_offsets >= 0)
            & (tail_offsets < tail_valid_token_counts.view(-1, 1))
        )
        sparse_values = torch.where(
            budget_mask,
            padded_slots,
            torch.where(
                tail_mask,
                resident_tail_starts.view(-1, 1) + tail_offsets,
                torch.full_like(padded_slots, -1),
            ),
        )
        attention_indices.copy_(torch.where(
            dense_rows.view(-1, 1),
            padded_topk,
            torch.where(sparse_rows.view(-1, 1), sparse_values,
                        torch.full_like(padded_topk, -1)),
        ))

    def lookup_resident_update(
        self,
        *,
        layer_id: int,
        kv_backend: DSAKVBackend,
        selection_topk_indices: torch.Tensor,
        req_pool_entries: torch.Tensor,
        sparse_local_row_indices: torch.Tensor,
        selection_block_table: torch.Tensor,
        lookup_state: DSAResidentLookupState,
        resident_tokens: int,
        tail_valid_token_counts: torch.Tensor,
        resident_tail_starts: torch.Tensor,
        budget_lengths: torch.Tensor,
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
        budget_lengths = self._as_device_i32(
            budget_lengths, device).reshape(-1)
        row_modes = self._as_device_i32(row_modes, device).reshape(-1)
        lookup_init_mask = lookup_init_mask.to(
            device=device, dtype=torch.bool).reshape(-1)
        selection_block_table = self._as_device_i32(
            selection_block_table, device)
        tail_valid_token_counts = self._as_device_i32(
            tail_valid_token_counts, device).reshape(-1)
        resident_tail_starts = self._as_device_i32(
            resident_tail_starts, device).reshape(-1)
        sparse_rows = row_modes == int(DSADecodeRowMode.SPARSE)
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
                initialize_rows=lookup_init_mask,
                resident_tokens=int(resident_tokens),
                selection_block_table=selection_block_table,
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
        sparse_misses = sparse_miss_out.to(dtype=torch.bool)
        sparse_selection_block_table = selection_block_table.index_select(
            0, sparse_local_rows)
        kv_backend.load_tokens_into(
            layer_id=int(layer_id),
            request_pool_entries=sparse_pool_entries,
            token_positions=sparse_topk,
            destination_slots=sparse_slot_out,
            load_mask=sparse_misses,
            destination_block_table=sparse_selection_block_table,
        )
        slot_out = torch.full_like(topk, -1)
        slot_out.index_copy_(0, sparse_local_rows, sparse_slot_out)

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
            slot_out=slot_out,
            row_modes=row_modes,
            budget_lengths=budget_lengths,
            tail_valid_token_counts=tail_valid_token_counts,
            resident_tail_starts=resident_tail_starts,
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
