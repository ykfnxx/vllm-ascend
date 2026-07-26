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


@dataclass(frozen=True)
class DSASparseFixedHBMBreakdown:
    """Logical tensor bytes reserved outside the vLLM KV block pool.

    ``hot_payload_bytes`` already covers every supplied local Main layer and
    is therefore independent of ``cohort_count``. Residency state and one
    maximum eager plan are core fixed tensors private to each cohort.

    The execution reserve covers the maximum logical bytes simultaneously
    live in ``DSASparseEagerBatchContext``. It consists of context-lifetime
    tensors plus the larger of the begin and lookup scratch phases. PyTorch
    allocator alignment and future custom-operator workspace are not included.
    """

    hot_payload_bytes: int
    residency_state_bytes_per_cohort: int
    eager_plan_bytes_per_cohort: int
    eager_context_bytes_per_cohort: int
    eager_begin_scratch_bytes_per_cohort: int
    eager_lookup_scratch_bytes_per_cohort: int
    cohort_count: int
    backend_auxiliary_bytes: int

    @property
    def residency_state_bytes(self) -> int:
        return self.residency_state_bytes_per_cohort * self.cohort_count

    @property
    def eager_plan_bytes(self) -> int:
        return self.eager_plan_bytes_per_cohort * self.cohort_count

    @property
    def core_fixed_tensor_bytes(self) -> int:
        return self.hot_payload_bytes + self.residency_state_bytes + self.eager_plan_bytes

    @property
    def eager_execution_reserve_bytes_per_cohort(self) -> int:
        return self.eager_context_bytes_per_cohort + max(
            self.eager_begin_scratch_bytes_per_cohort,
            self.eager_lookup_scratch_bytes_per_cohort,
        )

    @property
    def eager_execution_reserve_bytes(self) -> int:
        return self.eager_execution_reserve_bytes_per_cohort * self.cohort_count

    @property
    def fixed_hbm_bytes(self) -> int:
        return self.core_fixed_tensor_bytes + self.eager_execution_reserve_bytes + self.backend_auxiliary_bytes


def calculate_dsa_sparse_fixed_hbm_bytes(
    config: DSASparseCacheConfig,
    layer_layouts: Iterable[DSASparseLayerLayout],
    *,
    cohort_count: int,
    max_sfa_queries: int,
    backend_auxiliary_bytes: int = 0,
) -> DSASparseFixedHBMBreakdown:
    """Calculate the Decode-side fixed HBM reservation without tensors."""

    if isinstance(cohort_count, bool) or not isinstance(cohort_count, int):
        raise TypeError("cohort_count must be an integer.")
    if cohort_count <= 0:
        raise ValueError(f"cohort_count must be positive, got {cohort_count}.")
    if isinstance(backend_auxiliary_bytes, bool) or not isinstance(
        backend_auxiliary_bytes,
        int,
    ):
        raise TypeError("backend_auxiliary_bytes must be an integer.")
    if backend_auxiliary_bytes < 0:
        raise ValueError(f"backend_auxiliary_bytes must be non-negative, got {backend_auxiliary_bytes}.")
    if isinstance(max_sfa_queries, bool) or not isinstance(
        max_sfa_queries,
        int,
    ):
        raise TypeError("max_sfa_queries must be an integer.")
    if max_sfa_queries <= 0:
        raise ValueError(f"max_sfa_queries must be positive, got {max_sfa_queries}.")

    layouts = tuple(layer_layouts)
    if not layouts:
        raise ValueError("At least one local DSA Sparse Main layer layout is required.")
    layer_names = tuple(layout.layer_name for layout in layouts)
    if len(set(layer_names)) != len(layer_names):
        raise ValueError("DSA Sparse Main layer layouts must have unique names.")

    hot_payload_bytes = sum(_hot_payload_bytes(config, layout) for layout in layouts)
    (
        eager_context_bytes,
        eager_begin_scratch_bytes,
        eager_lookup_scratch_bytes,
    ) = _eager_execution_bytes(config, max_sfa_queries)
    return DSASparseFixedHBMBreakdown(
        hot_payload_bytes=hot_payload_bytes,
        residency_state_bytes_per_cohort=_residency_state_bytes(config),
        eager_plan_bytes_per_cohort=_maximum_eager_plan_bytes(config),
        eager_context_bytes_per_cohort=eager_context_bytes,
        eager_begin_scratch_bytes_per_cohort=eager_begin_scratch_bytes,
        eager_lookup_scratch_bytes_per_cohort=eager_lookup_scratch_bytes,
        cohort_count=cohort_count,
        backend_auxiliary_bytes=backend_auxiliary_bytes,
    )


def reserve_dsa_sparse_fixed_hbm_bytes(
    available_bytes: int,
    fixed_hbm_bytes: int,
    *,
    source: str,
) -> int:
    """Deduct one fixed reservation while preserving the baseline path."""

    if isinstance(available_bytes, bool) or not isinstance(available_bytes, int):
        raise TypeError("available_bytes must be an integer.")
    fixed_hbm_bytes = _validate_byte_count("fixed_hbm_bytes", fixed_hbm_bytes)
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


def _residency_state_bytes(config: DSASparseCacheConfig) -> int:
    int32 = torch.int32
    return sum(
        (
            _tensor_bytes(
                (config.max_num_seqs, config.max_model_len),
                int32,
            ),
            _tensor_bytes(
                (config.max_num_seqs, config.managed_hot_width),
                int32,
            ),
            _tensor_bytes(
                (config.max_num_seqs, config.device_buffer_size),
                int32,
            ),
            _tensor_bytes((config.max_num_seqs,), int32),
        )
    )


def _maximum_eager_plan_bytes(config: DSASparseCacheConfig) -> int:
    request_capacity = config.max_num_seqs
    query_lane_capacity = config.max_query_tokens_per_request
    token_capacity = request_capacity * query_lane_capacity
    read_shape = (
        request_capacity,
        query_lane_capacity,
        config.index_topk,
    )
    write_shape = (request_capacity, query_lane_capacity)

    tensor_specs: tuple[tuple[Sequence[int], torch.dtype], ...] = (
        # DSASparseRowMapping
        ((request_capacity,), torch.bool),
        ((request_capacity,), torch.int32),
        ((request_capacity,), torch.int32),
        # DSASparsePlan
        ((token_capacity,), torch.int32),
        ((token_capacity,), torch.int32),
        ((token_capacity,), torch.int32),
        ((token_capacity,), torch.bool),
        ((token_capacity,), torch.int32),
        ((token_capacity, config.index_topk), torch.int32),
        ((request_capacity,), torch.int32),
        (
            (request_capacity, config.max_blocks_per_request),
            torch.int32,
        ),
        (read_shape, torch.int32),
        (read_shape, torch.int32),
        (read_shape, torch.int32),
        (read_shape, torch.bool),
        ((token_capacity, config.index_topk), torch.int32),
        (
            (request_capacity, config.hot_blocks_per_seat),
            torch.int32,
        ),
        ((token_capacity,), torch.int32),
        (write_shape, torch.int32),
        (write_shape, torch.int32),
        (write_shape, torch.bool),
    )
    return sum(_tensor_bytes(shape, dtype) for shape, dtype in tensor_specs)


def _eager_execution_bytes(
    config: DSASparseCacheConfig,
    max_sfa_queries: int,
) -> tuple[int, int, int]:
    """Return context, begin scratch, and lookup scratch bytes per cohort."""

    request_capacity = config.max_num_seqs
    token_capacity = request_capacity * config.max_query_tokens_per_request
    active_query_capacity = min(token_capacity, max_sfa_queries)

    # These tensors survive from begin() until the batch context is closed.
    context_bytes = sum(
        (
            _tensor_bytes((active_query_capacity,), torch.long),
            _tensor_bytes((max_sfa_queries,), torch.int32),
            _tensor_bytes(
                (max_sfa_queries, config.index_topk),
                torch.int32,
            ),
        )
    )

    # All four fixed-shape copies are live together before begin_step()
    # copies them into the preallocated plan.
    begin_scratch_bytes = sum(
        (
            _tensor_bytes((token_capacity,), torch.int32),
            _tensor_bytes((token_capacity,), torch.bool),
            _tensor_bytes((request_capacity,), torch.int32),
            _tensor_bytes(
                (
                    request_capacity,
                    config.max_blocks_per_request,
                ),
                torch.int32,
            ),
            # Advanced indexing of newest_destination_hot_row_ids.
            _tensor_bytes((active_query_capacity,), torch.int32),
        )
    )

    fixed_lookup_bytes = sum(
        (
            _tensor_bytes(
                (token_capacity, config.index_topk),
                torch.int32,
            ),
            _tensor_bytes((token_capacity,), torch.int32),
        )
    )
    count_temporary_bytes = sum(
        (
            _tensor_bytes(
                (active_query_capacity, config.index_topk),
                torch.bool,
            ),
            _tensor_bytes((active_query_capacity,), torch.int32),
        )
    )
    resolved_gather_bytes = _tensor_bytes(
        (active_query_capacity, config.index_topk),
        torch.int32,
    )
    lookup_scratch_bytes = fixed_lookup_bytes + max(
        count_temporary_bytes,
        resolved_gather_bytes,
    )
    return context_bytes, begin_scratch_bytes, lookup_scratch_bytes


def _tensor_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    dimensions = tuple(shape)
    for dimension in dimensions:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise ValueError(f"Tensor dimensions must be non-negative integers, got {dimensions}.")
    return prod(dimensions, start=1) * _dtype_size_bytes(dtype)


def _dtype_size_bytes(dtype: torch.dtype) -> int:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Expected a torch.dtype, got {type(dtype).__name__}.")
    if dtype.is_complex:
        return torch.finfo(dtype).bits // 4
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    if dtype == torch.bool:
        return 1
    return torch.iinfo(dtype).bits // 8


def _validate_byte_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value
