# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Torch binding for the fused DSA Sparse lookup/update operator."""

from __future__ import annotations

import torch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseLookupBatch,
    DSASparseLookupOutput,
    DSASparseLookupState,
)


class TorchDSASparseLookupOperator:
    """Invoke the ASU-shaped lookup ABI with maintain fused in the kernel."""

    def lookup(
        self,
        *,
        state: DSASparseLookupState,
        batch: DSASparseLookupBatch,
    ) -> DSASparseLookupOutput:
        req_num = batch.req_pool_entries.shape[0]
        if req_num <= 0:
            raise ValueError(
                "DSA Sparse lookup requires at least one request."
            )
        slot_out, miss_out = (
            torch.ops._C_ascend.dsa_sparse_lookup_update(
                state.index,
                state.slot_to_index,
                state.free_slots,
                state.free_head,
                batch.req_pool_entries,
                batch.query_index,
                batch.lookup_mask,
                req_num,
            )
        )
        return DSASparseLookupOutput(
            slot_out=slot_out,
            miss_out=miss_out,
        )


__all__ = ["TorchDSASparseLookupOperator"]
