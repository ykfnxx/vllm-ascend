# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Torch bindings for the selectable DSA Sparse lookup backends."""

from __future__ import annotations

import torch

from vllm_ascend import dsa_sparse_probe
from vllm_ascend.attention.dsa_sparse import (
    DSASparseLookupBatch,
    DSASparseLookupOutput,
    DSASparseLookupState,
)
from vllm_ascend.dsa_sparse_config import (
    DSA_SPARSE_ASU_LOOKUP_BACKEND,
    DSA_SPARSE_FUSED_LOOKUP_BACKEND,
    DSASparseLookupBackend,
)


class TorchDSASparseLookupOperator:
    """Invoke the selected implementation of the common lookup ABI."""

    def __init__(
        self,
        lookup_backend: DSASparseLookupBackend = (
            DSA_SPARSE_FUSED_LOOKUP_BACKEND
        ),
    ) -> None:
        self.lookup_backend = lookup_backend
        if lookup_backend == DSA_SPARSE_ASU_LOOKUP_BACKEND:
            self._lookup = self._asu_hbm_index_lookup
            self._maintain_seed = 0
        else:
            self._lookup = self._dsa_sparse_lookup_update

    @staticmethod
    def _asu_hbm_index_lookup(
        *args: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.asu_hbm_index_lookup(*args)

    @staticmethod
    def _dsa_sparse_lookup_update(
        *args: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.dsa_sparse_lookup_update(*args)

    @staticmethod
    def _asu_hbm_index_maintain(
        *args: object,
    ) -> None:
        torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(*args)

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
        slot_out, miss_out = self._lookup(
            state.index,
            state.slot_to_index,
            state.free_slots,
            state.free_head,
            batch.req_pool_entries,
            batch.query_index,
            batch.lookup_mask,
            req_num,
        )
        synchronized = False
        if self.lookup_backend == DSA_SPARSE_ASU_LOOKUP_BACKEND:
            self._asu_hbm_index_maintain(
                state.index,
                state.slot_to_index,
                state.free_slots,
                state.free_head,
                batch.req_pool_entries,
                slot_out,
                req_num,
                self._maintain_seed,
            )
            self._maintain_seed = (
                self._maintain_seed + 1
            ) & 0x7FFFFFFF
            torch.npu.synchronize()
            synchronized = True
        if dsa_sparse_probe.is_enabled():
            if not synchronized:
                dsa_sparse_probe.synchronize_device()
            dsa_sparse_probe.emit(
                "lookup_update_done",
                cohort=state.cohort.name,
                role=state.cohort.role,
                req_num=req_num,
                req_pool_entries_shape=list(
                    batch.req_pool_entries.shape
                ),
                query_index_shape=list(batch.query_index.shape),
                lookup_mask_shape=list(batch.lookup_mask.shape),
                slot_out_shape=list(slot_out.shape),
                miss_out_shape=list(miss_out.shape),
            )
        return DSASparseLookupOutput(
            slot_out=slot_out,
            miss_out=miss_out,
        )


__all__ = ["TorchDSASparseLookupOperator"]
