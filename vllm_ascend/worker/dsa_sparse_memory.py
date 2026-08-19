# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import prod

import torch

from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_HEAD_STRIDE,
    DSA_SPARSE_FREE_SLOT_COUNT,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_LOOKUP_SLOT_COUNT,
)


@dataclass(frozen=True)
class DSASparseFixedHBMBreakdown:
    """Persistent Decode-side Hot Cache and lookup-state bytes."""

    hot_payload_bytes: int
    lookup_state_bytes_per_cohort: int
    cohort_count: int
    lookup_capacity: int
    transient_region_span: int
    fallback_slot_count: int
    verify_staging_capacity: int

    @property
    def lookup_state_bytes(self) -> int:
        return self.lookup_state_bytes_per_cohort * self.cohort_count

    @property
    def core_fixed_tensor_bytes(self) -> int:
        return self.hot_payload_bytes + self.lookup_state_bytes

    @property
    def fixed_hbm_bytes(self) -> int:
        return self.core_fixed_tensor_bytes


def calculate_dsa_sparse_fixed_hbm_bytes(
    max_num_seqs: int,
    block_size: int,
    main_layouts: Iterable[tuple[torch.dtype, int, int]],
    *,
    cohort_count: int,
    max_verify_tokens_per_request: int = 1,
    uses_mtp: bool = False,
) -> DSASparseFixedHBMBreakdown:
    layouts = tuple(main_layouts)
    transient_region_span = _transient_region_span(
        block_size,
        max_verify_tokens_per_request,
        uses_mtp=uses_mtp,
    )
    return DSASparseFixedHBMBreakdown(
        hot_payload_bytes=sum(
            _hot_payload_bytes(
                max_num_seqs,
                transient_region_span,
                dtype,
                num_kv_heads,
                head_size,
            )
            for dtype, num_kv_heads, head_size in layouts
        ),
        lookup_state_bytes_per_cohort=_lookup_state_bytes(max_num_seqs),
        cohort_count=cohort_count,
        lookup_capacity=DSA_SPARSE_LOOKUP_SLOT_COUNT,
        transient_region_span=transient_region_span,
        fallback_slot_count=int(uses_mtp),
        verify_staging_capacity=(
            max_verify_tokens_per_request if uses_mtp else 0
        ),
    )


def reserve_dsa_sparse_fixed_hbm_bytes(
    available_bytes: int,
    fixed_hbm_bytes: int,
    *,
    source: str,
) -> int:
    if fixed_hbm_bytes == 0:
        return available_bytes
    remaining_bytes = available_bytes - fixed_hbm_bytes
    if remaining_bytes < 0:
        raise ValueError(
            "DSA Sparse fixed HBM reservation exceeds the memory available "
            f"from {source}: available={available_bytes}, "
            f"fixed_hbm={fixed_hbm_bytes}."
        )
    return remaining_bytes


def _hot_payload_bytes(
    max_num_seqs: int,
    transient_region_span: int,
    dtype: torch.dtype,
    num_kv_heads: int,
    head_size: int,
) -> int:
    hot_rows = max_num_seqs * (
        DSA_SPARSE_LOOKUP_SLOT_COUNT + transient_region_span
    )
    return _tensor_bytes((hot_rows, num_kv_heads, head_size), dtype)


def _transient_region_span(
    block_size: int,
    max_verify_tokens_per_request: int,
    *,
    uses_mtp: bool,
) -> int:
    if block_size <= 0:
        raise ValueError("DSA Sparse block_size must be positive.")
    if not uses_mtp:
        return block_size
    if max_verify_tokens_per_request <= 0:
        raise ValueError(
            "DSA Sparse MTP max_verify_tokens_per_request must be positive."
        )
    transient_slots = 1 + max_verify_tokens_per_request
    return (
        (transient_slots + block_size - 1) // block_size
    ) * block_size


def _lookup_state_bytes(max_num_seqs: int) -> int:
    return sum(
        (
            _tensor_bytes(
                (max_num_seqs, DSA_SPARSE_INDEX_CAPACITY),
                torch.int32,
            ),
            _tensor_bytes(
                (
                    max_num_seqs,
                    DSA_SPARSE_LOOKUP_SLOT_COUNT,
                ),
                torch.int32,
            ),
            _tensor_bytes(
                (
                    max_num_seqs,
                    DSA_SPARSE_FREE_SLOT_COUNT,
                ),
                torch.int32,
            ),
            _tensor_bytes(
                (
                    max_num_seqs,
                    DSA_SPARSE_FREE_HEAD_STRIDE,
                ),
                torch.int32,
            ),
        )
    )


def _tensor_bytes(
    shape: Sequence[int],
    dtype: torch.dtype,
) -> int:
    return prod(shape, start=1) * _dtype_size_bytes(dtype)


def _dtype_size_bytes(dtype: torch.dtype) -> int:
    if dtype.is_complex:
        return torch.finfo(dtype).bits // 4
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    if dtype == torch.bool:
        return 1
    return torch.iinfo(dtype).bits // 8
