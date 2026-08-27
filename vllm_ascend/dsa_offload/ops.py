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


def lookup_update(
    state: LookupState,
    request_rows: torch.Tensor,
    query_indices: torch.Tensor,
    lookup_mask: torch.Tensor,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_offload_lookup_update(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_indices,
        lookup_mask,
        request_rows.shape[0],
    )


def lookup_update_batch(
    state: LookupState,
    request_rows: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_indices: torch.Tensor,
    lookup_mask: torch.Tensor,
) -> LookupOutput:
    return torch.ops._C_ascend.dsa_offload_lookup_update_batch(
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
