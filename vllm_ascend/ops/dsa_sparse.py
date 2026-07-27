# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_ascend.attention.dsa_sparse import (
        DSASparsePlan,
        DSASparseResidencyState,
    )


class DSASparseLookupUpdateTorchOperator:
    """Invoke the Ascend 950 fused lookup/update custom operator."""

    def lookup_update(
        self,
        *,
        state: "DSASparseResidencyState",
        plan: "DSASparsePlan",
    ) -> None:
        try:
            operator = torch.ops._C_ascend.dsa_sparse_lookup_update
        except AttributeError as error:
            raise RuntimeError(
                "dsa_sparse_lookup_update is unavailable. Build and install "
                "the Ascend 950 custom operator package before enabling DSA "
                "Sparse eager Decode."
            ) from error

        operator(
            state.token_to_hot,
            state.hot_to_token,
            state.lru_slots,
            state.state_seat_epoch,
            plan.row_mapping.row_to_cache_seat,
            plan.row_mapping.row_seat_epoch,
            plan.query_positions,
            plan.query_to_row,
            plan.query_to_lane,
            plan.query_valid_mask,
            plan.valid_topk_counts,
            plan.seq_lens,
            plan.topk_positions,
            plan.resolved_hot_indices,
            plan.miss_mask,
            plan.workspace,
        )
