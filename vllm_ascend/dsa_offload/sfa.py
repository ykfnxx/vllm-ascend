# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch

from . import lookup as _lookup
from . import pd as _pd

__all__ = [
    "prepare_main_slot_mapping",
    "publish_prefill_layer",
    "resolve_sfa_inputs",
]


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
    batch: _lookup.DSAOffloadBatch | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch is None or not batch.decode_request_indices:
        return semantic_topk, default_block_table

    cohort = next(cohort for cohort in batch.cohorts if layer_name in cohort.layer_names)
    plan = batch.lookup_plans.get(cohort.cohort_id)
    if plan is None:
        plan = _lookup.make_lookup_plan(
            semantic_topk=semantic_topk,
            default_block_table=default_block_table,
            cohort=cohort,
            batch=batch,
        )
        batch.lookup_plans[cohort.cohort_id] = plan
    layer_id = cohort.layer_ids[cohort.layer_names.index(layer_name)]
    _lookup.load_plan_misses(plan, layer_id, batch)
    sfa_block_table = default_block_table.clone()
    for request_index in batch.decode_request_indices:
        sfa_block_table[request_index].copy_(plan.hot_block_table[request_index])
    return plan.mapped_indices, sfa_block_table


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
