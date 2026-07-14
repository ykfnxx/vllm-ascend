"""NPU tensor backend for DSA lookup-resident cache management."""

from __future__ import annotations

from typing import NamedTuple

import torch

from vllm_ascend.dsa_sparse.dsa_resident_pool import DSAResidentLookupState
from vllm_ascend.dsa_sparse.dsa_types import DSADecodeRowMode


class DSALookupOutput(NamedTuple):
    attention_indices: torch.Tensor


class AscendDSAOpsBackend:
    """Materialize Indexer TopK through a persistent token-to-slot index.

    This framework implementation expresses the lookup state machine with NPU
    tensor operations. It defines the contract for later fused
    lookup/materialize/maintain operators and never invokes the legacy
    gather-selection operator.
    """

    @staticmethod
    def _squeeze_cache_head_dim(cache: torch.Tensor | None,
                                name: str) -> torch.Tensor:
        if not torch.is_tensor(cache):
            raise ValueError(f"{name} is required for DSA lookup resident")
        if cache.ndim == 4 and int(cache.shape[2]) == 1:
            return cache.squeeze(2)
        if cache.ndim == 3:
            return cache
        raise ValueError(
            f"{name} must have shape [blocks, block, 1, dim] or "
            f"[blocks, block, dim], got {tuple(cache.shape)}")

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

    @staticmethod
    def _materialize_pairs(
        *,
        token_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        pair_mask: torch.Tensor,
        selection_block_table: torch.Tensor,
        full_block_table: torch.Tensor,
        selection_kv_cache: torch.Tensor,
        selection_k_rope: torch.Tensor,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
    ) -> None:
        """Copy selected original-token KV records into arbitrary slots."""
        row_indices, pair_indices = pair_mask.nonzero(as_tuple=True)
        tokens = token_ids[row_indices, pair_indices].to(dtype=torch.long)
        slots = slot_ids[row_indices, pair_indices].to(dtype=torch.long)
        block_size = int(selection_kv_cache.shape[1])

        src_logical_blocks = torch.div(
            tokens, block_size, rounding_mode="floor")
        src_offsets = torch.remainder(tokens, block_size)
        dst_logical_blocks = torch.div(
            slots, block_size, rounding_mode="floor")
        dst_offsets = torch.remainder(slots, block_size)
        src_physical_blocks = full_block_table[
            row_indices, src_logical_blocks].to(dtype=torch.long)
        dst_physical_blocks = selection_block_table[
            row_indices, dst_logical_blocks].to(dtype=torch.long)
        src_flat_slots = src_physical_blocks * block_size + src_offsets
        dst_flat_slots = dst_physical_blocks * block_size + dst_offsets

        selection_kv_cache.reshape(-1, selection_kv_cache.shape[-1]).index_copy_(
            0,
            dst_flat_slots,
            full_kv_cache.reshape(-1, full_kv_cache.shape[-1]).index_select(
                0, src_flat_slots),
        )
        selection_k_rope.reshape(-1, selection_k_rope.shape[-1]).index_copy_(
            0,
            dst_flat_slots,
            full_k_rope.reshape(-1, full_k_rope.shape[-1]).index_select(
                0, src_flat_slots),
        )

    def _initialize_resident_rows(
        self,
        *,
        state: DSAResidentLookupState,
        pool_entries: torch.Tensor,
        initialize_rows: torch.Tensor,
        resident_tokens: int,
        selection_block_table: torch.Tensor,
        full_block_table: torch.Tensor,
        selection_kv_cache: torch.Tensor,
        selection_k_rope: torch.Tensor,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
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

        self._materialize_pairs(
            token_ids=tokens,
            slot_ids=slots,
            pair_mask=init_mask,
            selection_block_table=selection_block_table,
            full_block_table=full_block_table,
            selection_kv_cache=selection_kv_cache,
            selection_k_rope=selection_k_rope,
            full_kv_cache=full_kv_cache,
            full_k_rope=full_k_rope,
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
    def _lookup_allocate(
        *,
        state: DSAResidentLookupState,
        topk: torch.Tensor,
        pool_entries: torch.Tensor,
        candidate_lens: torch.Tensor,
        budget_lengths: torch.Tensor,
        sparse_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, topk_width = topk.shape
        pool_rows = pool_entries.to(dtype=torch.long)
        query_columns = torch.arange(
            topk_width, dtype=torch.int32, device=topk.device).view(1, -1)
        valid = (
            sparse_rows.view(-1, 1)
            & (query_columns < budget_lengths.view(-1, 1))
            & (topk >= 0)
            & (topk < candidate_lens.view(-1, 1))
        )
        safe_tokens = torch.where(valid, topk, torch.zeros_like(topk)).to(
            dtype=torch.long)
        expanded_pool_rows = pool_rows.view(-1, 1).expand(
            batch_size, topk_width)
        old_slots = state.token_to_slot[expanded_pool_rows, safe_tokens]
        hits = valid & (old_slots >= 0)
        misses = valid & ~hits

        miss_ranks = misses.to(dtype=torch.int32).cumsum(dim=1) - 1
        old_heads = state.free_head.index_select(0, pool_rows).view(-1, 1)
        free_positions = (old_heads + miss_ranks.clamp_min(0)).to(
            dtype=torch.long)
        allocated_slots = state.free_slots[
            expanded_pool_rows, free_positions]
        slot_out = torch.where(
            hits,
            old_slots,
            torch.where(misses, allocated_slots,
                        torch.full_like(old_slots, -1)),
        )

        miss_batch_rows, miss_columns = misses.nonzero(as_tuple=True)
        miss_pool_rows = pool_rows.index_select(0, miss_batch_rows)
        miss_tokens = topk[miss_batch_rows, miss_columns].to(dtype=torch.long)
        miss_slots = slot_out[miss_batch_rows, miss_columns].to(dtype=torch.long)
        state.token_to_slot[miss_pool_rows, miss_tokens] = miss_slots.to(
            dtype=torch.int32)
        state.slot_to_token[miss_pool_rows, miss_slots] = miss_tokens.to(
            dtype=torch.int32)
        miss_counts = misses.sum(dim=1, dtype=torch.int32)
        state.free_head[pool_rows] = (
            old_heads.reshape(-1) + miss_counts)
        return slot_out.to(dtype=torch.int32), misses, miss_counts

    @staticmethod
    def _maintain_lookup_state(
        *,
        state: DSAResidentLookupState,
        pool_entries: torch.Tensor,
        sparse_rows: torch.Tensor,
        slot_out: torch.Tensor,
        miss_counts: torch.Tensor,
        total_slots: int,
    ) -> None:
        pool_rows = pool_entries.to(dtype=torch.long)
        batch_size = int(pool_rows.numel())
        slot_columns = torch.arange(
            total_slots, dtype=torch.int32,
            device=slot_out.device).view(1, -1)
        cursors = state.evict_cursor.index_select(0, pool_rows).view(-1, 1)
        ordered_slots = torch.remainder(
            slot_columns + cursors, total_slots).to(dtype=torch.long)
        reverse_rows = state.slot_to_token.index_select(0, pool_rows)
        ordered_tokens = reverse_rows.gather(1, ordered_slots)

        protected_counts = torch.zeros(
            (batch_size, total_slots),
            dtype=torch.int32,
            device=slot_out.device,
        )
        valid_slots = sparse_rows.view(-1, 1) & (slot_out >= 0)
        protected_counts.scatter_add_(
            1,
            torch.where(valid_slots, slot_out, torch.zeros_like(slot_out)).to(
                dtype=torch.long),
            valid_slots.to(dtype=torch.int32),
        )
        protected = protected_counts > 0
        eligible = (
            sparse_rows.view(-1, 1)
            & (ordered_tokens >= 0)
            & ~protected.gather(1, ordered_slots)
        )
        victim_ranks = eligible.to(dtype=torch.int32).cumsum(dim=1)
        victim_mask = eligible & (
            victim_ranks <= miss_counts.view(-1, 1))
        victim_batch_rows, victim_order_positions = victim_mask.nonzero(
            as_tuple=True)
        victim_pool_rows = pool_rows.index_select(0, victim_batch_rows)
        victim_slots = ordered_slots[
            victim_batch_rows, victim_order_positions]
        victim_tokens = ordered_tokens[
            victim_batch_rows, victim_order_positions].to(dtype=torch.long)
        free_positions = (
            victim_ranks[victim_batch_rows, victim_order_positions] - 1
        ).to(dtype=torch.long)

        state.token_to_slot[victim_pool_rows, victim_tokens] = -1
        state.slot_to_token[victim_pool_rows, victim_slots] = -1
        state.free_slots[victim_pool_rows, free_positions] = victim_slots.to(
            dtype=torch.int32)
        sparse_pool_rows = pool_rows[sparse_rows]
        state.free_head[sparse_pool_rows] = 0
        state.evict_cursor[sparse_pool_rows] = torch.remainder(
            state.evict_cursor[sparse_pool_rows]
            + miss_counts[sparse_rows],
            total_slots,
        )

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
        selection_topk_indices: torch.Tensor,
        req_pool_entries: torch.Tensor,
        candidate_lens: torch.Tensor,
        selection_block_table: torch.Tensor,
        full_block_table: torch.Tensor,
        nopek_cache_zone: torch.Tensor,
        ropek_cache_zone: torch.Tensor,
        nopek_dram_arena: torch.Tensor,
        ropek_dram_arena: torch.Tensor,
        lookup_state: DSAResidentLookupState,
        resident_tokens: int,
        total_slots: int,
        tail_valid_token_counts: torch.Tensor,
        resident_tail_starts: torch.Tensor,
        budget_lengths: torch.Tensor,
        attention_indices_width: int,
        prebuilt_attention_indices: torch.Tensor | None = None,
        row_modes: torch.Tensor,
        lookup_init_mask: torch.Tensor,
        has_lookup_init_rows: bool,
    ) -> DSALookupOutput:
        selection_k_rope = self._squeeze_cache_head_dim(
            ropek_cache_zone, "ropek_cache_zone")
        selection_kv_cache = self._squeeze_cache_head_dim(
            nopek_cache_zone, "nopek_cache_zone")
        full_k_rope = self._squeeze_cache_head_dim(
            ropek_dram_arena, "ropek_dram_arena")
        full_kv_cache = self._squeeze_cache_head_dim(
            nopek_dram_arena, "nopek_dram_arena")
        device = selection_kv_cache.device
        topk = self._normalize_topk(selection_topk_indices, device)
        pool_entries = self._as_device_i32(
            req_pool_entries, device).reshape(-1)
        candidate_lens = self._as_device_i32(
            candidate_lens, device).reshape(-1)
        budget_lengths = self._as_device_i32(
            budget_lengths, device).reshape(-1)
        row_modes = self._as_device_i32(row_modes, device).reshape(-1)
        lookup_init_mask = lookup_init_mask.to(
            device=device, dtype=torch.bool).reshape(-1)
        selection_block_table = self._as_device_i32(
            selection_block_table, device)
        full_block_table = self._as_device_i32(full_block_table, device)
        tail_valid_token_counts = self._as_device_i32(
            tail_valid_token_counts, device).reshape(-1)
        resident_tail_starts = self._as_device_i32(
            resident_tail_starts, device).reshape(-1)
        sparse_rows = row_modes == int(DSADecodeRowMode.SPARSE)

        if has_lookup_init_rows:
            self._initialize_resident_rows(
                state=lookup_state,
                pool_entries=pool_entries,
                initialize_rows=lookup_init_mask,
                resident_tokens=int(resident_tokens),
                selection_block_table=selection_block_table,
                full_block_table=full_block_table,
                selection_kv_cache=selection_kv_cache,
                selection_k_rope=selection_k_rope,
                full_kv_cache=full_kv_cache,
                full_k_rope=full_k_rope,
            )
        slot_out, misses, miss_counts = self._lookup_allocate(
            state=lookup_state,
            topk=topk,
            pool_entries=pool_entries,
            candidate_lens=candidate_lens,
            budget_lengths=budget_lengths,
            sparse_rows=sparse_rows,
        )
        self._materialize_pairs(
            token_ids=topk,
            slot_ids=slot_out,
            pair_mask=misses,
            selection_block_table=selection_block_table,
            full_block_table=full_block_table,
            selection_kv_cache=selection_kv_cache,
            selection_k_rope=selection_k_rope,
            full_kv_cache=full_kv_cache,
            full_k_rope=full_k_rope,
        )

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
        self._maintain_lookup_state(
            state=lookup_state,
            pool_entries=pool_entries,
            sparse_rows=sparse_rows,
            slot_out=slot_out,
            miss_counts=miss_counts,
            total_slots=int(total_slots),
        )
        return DSALookupOutput(attention_indices=attention_indices)
