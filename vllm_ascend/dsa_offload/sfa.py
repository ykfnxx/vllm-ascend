# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

import torch

from . import lookup as _lookup
from . import pd as _pd

__all__ = [
    "SFAAddressingView",
    "SFAAddressingWorkspace",
    "maybe_start_group_prefetch",
    "prepare_main_slot_mapping",
    "prepare_indexer_cache_write",
    "publish_prefill_layer",
    "resolve_sfa_inputs",
]


@dataclass(frozen=True)
class SFAAddressingView:
    sparse_indices: torch.Tensor
    block_table: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor


@dataclass
class SFAAddressingWorkspace:
    block_table: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        max_num_seqs: int,
        max_block_table_width: int,
        device: torch.device | str,
    ) -> "SFAAddressingWorkspace":
        if max_block_table_width <= 0:
            raise ValueError("DSA Offload SFA block table width must be positive.")
        return cls(
            block_table=torch.zeros(
                (max_num_seqs, max_block_table_width),
                dtype=torch.int32,
                device=device,
            ),
            actual_seq_lengths_kv=torch.zeros(
                max_num_seqs,
                dtype=torch.int32,
                device=device,
            ),
        )

    def compose(
        self,
        *,
        default_block_table: torch.Tensor,
        default_actual_seq_lengths_kv: torch.Tensor,
        batch: _lookup.DSAOffloadBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hot_cache = batch.hot_cache
        if hot_cache is None:
            raise RuntimeError("DSA Offload Decode addressing requires a Hot Cache.")

        num_reqs, default_width = default_block_table.shape
        hot_width = hot_cache.hot_block_table.shape[1]
        required_width = max(default_width, hot_width)
        if num_reqs > self.block_table.shape[0] or required_width > self.block_table.shape[1]:
            raise RuntimeError(
                "DSA Offload SFA addressing workspace is too small: "
                f"requests={num_reqs}/{self.block_table.shape[0]}, "
                f"block_table_width={required_width}/{self.block_table.shape[1]}."
            )
        if default_actual_seq_lengths_kv.shape[0] != num_reqs:
            raise ValueError(
                "DSA Offload SFA KV lengths must have one entry per request: "
                f"requests={num_reqs}, lengths={default_actual_seq_lengths_kv.shape[0]}."
            )

        effective_block_table = self.block_table[:num_reqs, :required_width]
        effective_block_table.zero_()
        effective_block_table[:, :default_width].copy_(default_block_table)
        effective_kv_lengths = self.actual_seq_lengths_kv[:num_reqs]
        effective_kv_lengths.copy_(default_actual_seq_lengths_kv)

        decode_indices = batch.decode_request_indices_tensor
        if decode_indices is None:
            decode_indices = torch.tensor(
                batch.decode_request_indices,
                dtype=torch.int64,
                device=effective_block_table.device,
            )
        elif decode_indices.device != effective_block_table.device:
            raise ValueError("DSA Offload Decode request indices and SFA workspace must be on the same device.")
        hot_rows = batch.request_rows.index_select(0, decode_indices).to(torch.int64)
        effective_block_table[decode_indices] = 0
        effective_block_table[decode_indices, :hot_width] = hot_cache.hot_block_table.index_select(0, hot_rows)
        # Decode sparse indices are remapped into the fixed Hot Cache virtual
        # row, so their validity boundary must use the same virtual address
        # space instead of the request's original logical sequence length.
        effective_kv_lengths[decode_indices] = batch.layout.row_stride
        return effective_block_table, effective_kv_lengths


def _prepare_sfa_addressing_view(
    *,
    batch: _lookup.DSAOffloadBatch,
    default_block_table: torch.Tensor,
    default_actual_seq_lengths_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose only when the current KV-cache metadata source changes."""

    prepared = batch.prepared_step_addressing
    if prepared is None:
        prepared = _lookup.PreparedStepAddressing()
        batch.prepared_step_addressing = prepared
    sources = prepared.sfa_view_sources
    if sources is None or (
        sources[0] is not default_block_table
        or sources[1] is not default_actual_seq_lengths_kv
    ):
        workspace = batch.sfa_workspace
        if workspace is None:
            raise RuntimeError(
                "DSA Offload Decode requires a persistent SFA addressing "
                "workspace."
            )
        (
            prepared.sfa_block_table,
            prepared.sfa_actual_seq_lengths_kv,
        ) = workspace.compose(
            default_block_table=default_block_table,
            default_actual_seq_lengths_kv=default_actual_seq_lengths_kv,
            batch=batch,
        )
        prepared.sfa_view_sources = (
            default_block_table,
            default_actual_seq_lengths_kv,
        )
    assert prepared.sfa_block_table is not None
    assert prepared.sfa_actual_seq_lengths_kv is not None
    return (
        prepared.sfa_block_table,
        prepared.sfa_actual_seq_lengths_kv,
    )


def prepare_main_slot_mapping(
    *,
    batch: _lookup.DSAOffloadBatch | None,
    default_slot_mapping: torch.Tensor,
    default_block_table: torch.Tensor | None = None,
    default_actual_seq_lengths_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if batch is None or not batch.decode_request_indices:
        return default_slot_mapping

    if (default_block_table is None) != (
        default_actual_seq_lengths_kv is None
    ):
        raise ValueError(
            "DSA Offload step addressing requires both block table and KV lengths."
        )

    # Materialize shared Decode metadata on the main stream before grouped
    # prefetch can start on its side stream.  Later exact and prefetch Lookup
    # planning therefore consume the same ready tensors without a new join.
    addressing = _lookup.get_packed_addressing_metadata(batch)
    main_uses_fused_lookup = (
        batch.is_mtp
        and batch.enable_turbo_lookup
        and batch.enable_turbo_fused_lookup
    )
    prefetch_uses_fused_lookup = (
        batch.is_mtp
        and batch.enable_turbo_prefetch_lookup
        and batch.enable_turbo_fused_prefetch_lookup
    )
    if not main_uses_fused_lookup or (
        batch.prefetch_runtime is not None
        and not prefetch_uses_fused_lookup
    ):
        _lookup.get_expanded_lookup_boundaries(batch)
    prepared = batch.prepared_step_addressing
    if prepared is None:
        prepared = _lookup.PreparedStepAddressing()
        batch.prepared_step_addressing = prepared

    slot_mapping_key = id(default_slot_mapping)
    cached_mapping = prepared.main_slot_mappings.get(slot_mapping_key)
    main_slot_mapping = (
        cached_mapping[1]
        if cached_mapping is not None
        and cached_mapping[0] is default_slot_mapping
        else None
    )
    if main_slot_mapping is None:
        main_slot_mapping = default_slot_mapping.clone()
        if batch.graph_query_start_loc is not None:
            total_queries = addressing.query_positions.shape[0]
            query_rows = addressing.query_request_rows_long
            if batch.is_mtp:
                expanded_query_starts = addressing.expanded_query_starts
                assert expanded_query_starts is not None
                row_offsets = (
                    batch.layout.staging_base
                    + torch.arange(
                        total_queries,
                        dtype=torch.int32,
                        device=main_slot_mapping.device,
                    )
                    - expanded_query_starts
                )
            else:
                row_offsets = batch.layout.tail_base + torch.remainder(
                    addressing.query_positions,
                    batch.layout.block_size,
                )
            row_blocks = (
                batch.layout.hot_block_base
                + query_rows * batch.layout.hot_blocks_per_row
            )
            main_slot_mapping[:total_queries] = (
                row_blocks * batch.layout.block_size + row_offsets
            )
        else:
            for request_index in batch.decode_request_indices:
                begin, end = batch.query_ranges[request_index]
                row_id = int(batch.request_rows[request_index].item())
                if batch.is_mtp:
                    row_offsets = batch.layout.staging_base + torch.arange(
                        end - begin,
                        dtype=torch.int64,
                        device=main_slot_mapping.device,
                    )
                else:
                    row_offsets = batch.layout.tail_base + torch.remainder(
                        batch.query_positions[begin:end],
                        batch.layout.block_size,
                    )
                main_slot_mapping[begin:end] = (
                    batch.layout.row_block_base(row_id)
                    * batch.layout.block_size
                    + row_offsets
                )
        prepared.main_slot_mappings[slot_mapping_key] = (
            default_slot_mapping,
            main_slot_mapping,
        )

    if default_block_table is not None:
        _lookup.get_decode_block_table(batch, default_block_table)
        assert default_actual_seq_lengths_kv is not None
        _prepare_sfa_addressing_view(
            batch=batch,
            default_block_table=default_block_table,
            default_actual_seq_lengths_kv=default_actual_seq_lengths_kv,
        )

    return main_slot_mapping


def maybe_start_group_prefetch(
    *,
    layer_name: str,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    default_block_table: torch.Tensor,
    batch: _lookup.DSAOffloadBatch | None,
    source_rows_before_gather: int | None,
) -> None:
    if batch is None or batch.prefetch_runtime is None:
        return
    batch.prefetch_runtime.start(
        layer_name=layer_name,
        hidden_states=hidden_states,
        cos=cos,
        sin=sin,
        batch=batch,
        block_table=default_block_table,
        source_rows_before_gather=source_rows_before_gather,
    )


def prepare_indexer_cache_write(
    *,
    layer_name: str,
    key_cache: torch.Tensor,
    indexer_cache: tuple[torch.Tensor, ...],
    enable_sparse_li_c8: bool,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    block_size: int,
    batch: _lookup.DSAOffloadBatch | None,
) -> bool:
    if batch is None or batch.prefetch_runtime is None:
        return False
    runtime = batch.prefetch_runtime
    if layer_name not in runtime.target_layer_names:
        return False
    runtime.wait_for_compute_before_key_write(layer_name)
    base_count = 2 if enable_sparse_li_c8 else 1
    if len(indexer_cache) != base_count + 1:
        return False
    torch.ops._C_ascend.npu_scatter_nd_update_mean(
        key_cache.view(-1, key.shape[-1]),
        slot_mapping.view(-1, 1),
        key.view(-1, key.shape[-1]).contiguous(),
        indexer_cache[-1],
        block_size,
    )
    return True


def resolve_sfa_inputs(
    *,
    layer_name: str,
    semantic_topk: torch.Tensor,
    default_block_table: torch.Tensor,
    default_actual_seq_lengths_kv: torch.Tensor,
    batch: _lookup.DSAOffloadBatch | None,
) -> SFAAddressingView:
    if batch is None or not batch.decode_request_indices:
        return SFAAddressingView(
            sparse_indices=semantic_topk,
            block_table=default_block_table,
            actual_seq_lengths_kv=default_actual_seq_lengths_kv,
        )

    cohort = next(cohort for cohort in batch.cohorts if layer_name in cohort.layer_names)
    plan = batch.lookup_plans.get(cohort.cohort_id)
    if plan is None:
        if batch.prefetch_runtime is not None:
            batch.prefetch_runtime.wait_before_exact_lookup(layer_name)
        plan = _lookup.make_lookup_plan(
            semantic_topk=semantic_topk,
            cohort=cohort,
            batch=batch,
        )
        batch.lookup_plans[cohort.cohort_id] = plan
    layer_id = cohort.layer_ids[cohort.layer_names.index(layer_name)]
    _lookup.load_plan_misses(plan, layer_id, batch)
    if batch.prefetch_runtime is not None:
        batch.prefetch_runtime.release_after_exact_load(layer_name)
    sfa_block_table, sfa_actual_seq_lengths_kv = _prepare_sfa_addressing_view(
        batch=batch,
        default_block_table=default_block_table,
        default_actual_seq_lengths_kv=default_actual_seq_lengths_kv,
    )
    return SFAAddressingView(
        sparse_indices=plan.mapped_indices,
        block_table=sfa_block_table,
        actual_seq_lengths_kv=sfa_actual_seq_lengths_kv,
    )


def publish_prefill_layer(
    *,
    layer_name: str,
    semantic_topk: torch.Tensor,
    main_cache: tuple[torch.Tensor, ...],
    block_table: torch.Tensor,
    batch: _lookup.DSAOffloadBatch | None,
) -> None:
    if batch is None or not isinstance(
        batch.prefill_state,
        _pd.PrefillPublishState,
    ):
        return
    cohort = next(cohort for cohort in batch.cohorts if layer_name in cohort.layer_names)
    layer_index = cohort.layer_names.index(layer_name)
    batch.prefill_state.publish_layer(
        layer_name=layer_name,
        layer_id=cohort.layer_ids[layer_index],
        semantic_topk=semantic_topk,
        main_cache=main_cache,
        block_table=block_table,
    )
