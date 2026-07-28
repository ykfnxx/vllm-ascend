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
    dsa_sparse_lookup_workspace_stride,
)


@dataclass(frozen=True)
class DSASparseFixedHBMBreakdown:
    """Logical tensor bytes reserved outside the vLLM KV block pool.

    ``hot_payload_bytes`` already covers every supplied local Main layer and
    is therefore independent of ``cohort_count``. Residency state and one
    lookup plan are core fixed tensors private to each cohort. Role-level
    request-index metadata is allocated once and shared by every target cohort.

    ``initialization_scratch_bytes`` covers the largest explicit temporary
    tensor used while constructing the fixed state. The execution reserve
    covers the maximum logical bytes simultaneously live in
    ``DSASparseEagerBatchContext``. It consists of one shared metadata staging
    pass plus cohort-private context and lookup scratch tensors. Initialization
    and execution do not overlap, so the fixed reservation uses the larger
    phase peak. The fused custom-operator workspace is included in the core
    lookup plan; PyTorch allocator alignment is not included.
    """

    hot_payload_bytes: int
    batch_metadata_bytes: int
    residency_state_bytes_per_cohort: int
    lookup_plan_bytes_per_cohort: int
    initialization_scratch_bytes: int
    eager_batch_staging_bytes: int
    eager_context_bytes_per_cohort: int
    eager_lookup_scratch_bytes_per_cohort: int
    cohort_count: int
    backend_auxiliary_bytes: int

    @property
    def residency_state_bytes(self) -> int:
        return self.residency_state_bytes_per_cohort * self.cohort_count

    @property
    def lookup_plan_bytes(self) -> int:
        return self.lookup_plan_bytes_per_cohort * self.cohort_count

    @property
    def core_fixed_tensor_bytes(self) -> int:
        return (
            self.hot_payload_bytes
            + self.batch_metadata_bytes
            + self.residency_state_bytes
            + self.lookup_plan_bytes
        )

    @property
    def eager_execution_reserve_bytes_per_cohort(self) -> int:
        return (
            self.eager_context_bytes_per_cohort
            + self.eager_lookup_scratch_bytes_per_cohort
        )

    @property
    def eager_execution_reserve_bytes(self) -> int:
        return (
            self.eager_batch_staging_bytes
            + self.eager_execution_reserve_bytes_per_cohort
            * self.cohort_count
        )

    @property
    def runtime_peak_reserve_bytes(self) -> int:
        return max(
            self.initialization_scratch_bytes,
            self.eager_execution_reserve_bytes,
        )

    @property
    def fixed_hbm_bytes(self) -> int:
        return (
            self.core_fixed_tensor_bytes
            + self.runtime_peak_reserve_bytes
            + self.backend_auxiliary_bytes
        )


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
        eager_batch_staging_bytes,
        eager_context_bytes,
        eager_lookup_scratch_bytes,
    ) = _eager_execution_bytes(config, max_sfa_queries)
    return DSASparseFixedHBMBreakdown(
        hot_payload_bytes=hot_payload_bytes,
        batch_metadata_bytes=_batch_metadata_bytes(config),
        residency_state_bytes_per_cohort=_residency_state_bytes(config),
        lookup_plan_bytes_per_cohort=_maximum_lookup_plan_bytes(config),
        initialization_scratch_bytes=_initialization_scratch_bytes(config),
        eager_batch_staging_bytes=eager_batch_staging_bytes,
        eager_context_bytes_per_cohort=eager_context_bytes,
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
                (config.max_num_seqs, config.device_buffer_size),
                int32,
            ),
            _tensor_bytes(
                (config.max_num_seqs, config.device_buffer_size),
                int32,
            ),
        )
    )


def _batch_metadata_bytes(config: DSASparseCacheConfig) -> int:
    request_capacity = config.max_num_seqs
    query_lane_capacity = config.max_query_tokens_per_request
    token_capacity = request_capacity * query_lane_capacity
    write_shape = (request_capacity, query_lane_capacity)

    tensor_specs: tuple[tuple[Sequence[int], torch.dtype], ...] = (
        # DSASparseBatchMetadata
        ((token_capacity,), torch.int32),
        ((token_capacity,), torch.int32),
        ((token_capacity,), torch.int32),
        ((token_capacity,), torch.bool),
        ((request_capacity,), torch.int32),
        (
            (request_capacity, config.max_blocks_per_request),
            torch.int32,
        ),
        (
            (request_capacity, config.hot_blocks_per_request),
            torch.int32,
        ),
        (write_shape, torch.int32),
        (write_shape, torch.int32),
        (write_shape, torch.bool),
    )
    return sum(_tensor_bytes(shape, dtype) for shape, dtype in tensor_specs)


def _maximum_lookup_plan_bytes(config: DSASparseCacheConfig) -> int:
    request_capacity = config.max_num_seqs
    token_capacity = (
        request_capacity * config.max_query_tokens_per_request
    )
    topk_shape = (token_capacity, config.index_topk)
    tensor_specs: tuple[tuple[Sequence[int], torch.dtype], ...] = (
        ((token_capacity,), torch.int32),
        (topk_shape, torch.int32),
        (topk_shape, torch.int32),
        (topk_shape, torch.bool),
        (
            (
                request_capacity,
                dsa_sparse_lookup_workspace_stride(
                    config.device_buffer_size
                ),
            ),
            torch.int32,
        ),
    )
    return sum(
        _tensor_bytes(shape, dtype)
        for shape, dtype in tensor_specs
    )


def _initialization_scratch_bytes(
    config: DSASparseCacheConfig,
) -> int:
    """Return the largest explicit temporary in fixed-tensor construction."""

    token_capacity = (
        config.max_num_seqs * config.max_query_tokens_per_request
    )
    # DSASparseBatchMetadata.allocate keeps flat_query_indices[Q] alive while
    # constructing query_to_req_idx/query_to_lane. DSASparseResidencyState.allocate
    # keeps arange(S) alive while cloning the expanded LRU initializer.
    return _tensor_bytes(
        (max(token_capacity, config.device_buffer_size),),
        torch.int32,
    )


def _eager_execution_bytes(
    config: DSASparseCacheConfig,
    max_sfa_queries: int,
) -> tuple[int, int, int]:
    """Return shared staging, context, and lookup scratch logical bytes."""

    request_capacity = config.max_num_seqs
    token_capacity = request_capacity * config.max_query_tokens_per_request
    active_query_capacity = min(token_capacity, max_sfa_queries)

    # These tensors survive from begin() until the batch context is closed.
    context_bytes = sum(
        (
            _tensor_bytes((active_query_capacity,), torch.long),
            _tensor_bytes((request_capacity,), torch.long),
            _tensor_bytes((max_sfa_queries,), torch.int32),
            _tensor_bytes(
                (max_sfa_queries, config.index_topk),
                torch.int32,
            ),
        )
    )

    # One role-level staging pass builds the shared fixed metadata.
    batch_staging_bytes = sum(
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
            # query request-index/lane long views used by vectorized addressing.
            2 * _tensor_bytes((token_capacity,), torch.long),
            # seq lens, block indices, physical blocks,
            # destinations, global slots, invalid values.
            6 * _tensor_bytes((token_capacity,), torch.int32),
            # validity mask.
            _tensor_bytes((token_capacity,), torch.bool),
            # safe block and flattened write indices.
            2 * _tensor_bytes((token_capacity,), torch.long),
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
    return batch_staging_bytes, context_bytes, lookup_scratch_bytes


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
