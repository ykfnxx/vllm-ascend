# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import prod

import torch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseCacheConfig,
    DSASparseLayerLayout,
)
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
    backend_auxiliary_bytes: int

    @property
    def lookup_state_bytes(self) -> int:
        return self.lookup_state_bytes_per_cohort * self.cohort_count

    @property
    def core_fixed_tensor_bytes(self) -> int:
        return self.hot_payload_bytes + self.lookup_state_bytes

    @property
    def fixed_hbm_bytes(self) -> int:
        return self.core_fixed_tensor_bytes + self.backend_auxiliary_bytes


def calculate_dsa_sparse_fixed_hbm_bytes(
    config: DSASparseCacheConfig,
    layer_layouts: Iterable[DSASparseLayerLayout],
    *,
    cohort_count: int,
    backend_auxiliary_bytes: int = 0,
) -> DSASparseFixedHBMBreakdown:
    layouts = tuple(layer_layouts)
    return DSASparseFixedHBMBreakdown(
        hot_payload_bytes=sum(
            _hot_payload_bytes(config, layout)
            for layout in layouts
        ),
        lookup_state_bytes_per_cohort=_lookup_state_bytes(config),
        cohort_count=cohort_count,
        backend_auxiliary_bytes=backend_auxiliary_bytes,
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
    config: DSASparseCacheConfig,
    layout: DSASparseLayerLayout,
) -> int:
    hot_rows = config.total_hot_blocks * config.block_size
    return sum(
        _tensor_bytes((hot_rows, *row_shape), dtype)
        for dtype, row_shape in zip(
            layout.plane_dtypes,
            layout.plane_row_shapes,
        )
    )


def _lookup_state_bytes(config: DSASparseCacheConfig) -> int:
    return sum(
        (
            _tensor_bytes(
                (config.max_num_seqs, DSA_SPARSE_INDEX_CAPACITY),
                torch.int32,
            ),
            _tensor_bytes(
                (
                    config.max_num_seqs,
                    DSA_SPARSE_LOOKUP_SLOT_COUNT,
                ),
                torch.int32,
            ),
            _tensor_bytes(
                (
                    config.max_num_seqs,
                    DSA_SPARSE_FREE_SLOT_COUNT,
                ),
                torch.int32,
            ),
            _tensor_bytes(
                (
                    config.max_num_seqs,
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
