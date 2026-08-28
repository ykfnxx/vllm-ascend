# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

import torch

from . import lookup as _lookup
from . import pd as _pd

__all__ = [
    "SFAAddressingView",
    "SFAAddressingWorkspace",
    "prepare_main_slot_mapping",
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


def prepare_main_slot_mapping(
    *,
    batch: _lookup.DSAOffloadBatch | None,
    default_slot_mapping: torch.Tensor,
) -> torch.Tensor:
    if batch is None or not batch.decode_request_indices:
        return default_slot_mapping

    main_slot_mapping = default_slot_mapping.clone()
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
        main_slot_mapping[begin:end] = batch.layout.row_block_base(row_id) * batch.layout.block_size + row_offsets
    return main_slot_mapping


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
        plan = _lookup.make_lookup_plan(
            semantic_topk=semantic_topk,
            cohort=cohort,
            batch=batch,
        )
        batch.lookup_plans[cohort.cohort_id] = plan
    layer_id = cohort.layer_ids[cohort.layer_names.index(layer_name)]
    _lookup.load_plan_misses(plan, layer_id, batch)
    workspace = batch.sfa_workspace
    if workspace is None:
        raise RuntimeError("DSA Offload Decode requires a persistent SFA addressing workspace.")
    sfa_block_table, actual_seq_lengths_kv = workspace.compose(
        default_block_table=default_block_table,
        default_actual_seq_lengths_kv=default_actual_seq_lengths_kv,
        batch=batch,
    )
    return SFAAddressingView(
        sparse_indices=plan.mapped_indices,
        block_table=sfa_block_table,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
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
