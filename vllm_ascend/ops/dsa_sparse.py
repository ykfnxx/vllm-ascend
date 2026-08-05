# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Torch binding for the DSASparse lookup/update operator."""

import torch


def dsa_sparse_lookup_update(
    index: torch.Tensor,
    slot_to_index: torch.Tensor,
    free_slots: torch.Tensor,
    free_head: torch.Tensor,
    req_pool_entries: torch.Tensor,
    query_index: torch.Tensor,
    lookup_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C_ascend.dsa_sparse_lookup_update(
        index,
        slot_to_index,
        free_slots,
        free_head,
        req_pool_entries,
        query_index,
        lookup_mask,
        req_pool_entries.shape[0],
    )


__all__ = ["dsa_sparse_lookup_update"]
