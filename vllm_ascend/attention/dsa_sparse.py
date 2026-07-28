#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from collections import deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

import torch

from vllm_ascend import dsa_sparse_probe
from vllm_ascend.attention.dsa_sparse_io import DSASparseIOOperator

INVALID_INDEX = -1
DSA_SPARSE_SIMT_THREADS = 256
DSA_SPARSE_MAX_QUERY_LANES = 4
DSA_SPARSE_WORKSPACE_COUNTERS = 4


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def dsa_sparse_lookup_workspace_stride(evictable_slots: int) -> int:
    """Return the per-request int32 workspace used by the A5 SIMT op."""

    if evictable_slots <= 0:
        raise ValueError(
            "evictable_slots must be positive, got "
            f"{evictable_slots}."
        )
    return (
        3 * evictable_slots
        + 3 * DSA_SPARSE_SIMT_THREADS
        + DSA_SPARSE_WORKSPACE_COUNTERS
    )


@dataclass(frozen=True)
class DSASparseCacheConfig:
    """Static Decode-side cache dimensions.

    ``device_buffer_size`` counts only evictable slots. Reserved newest slots
    and block alignment are added by :attr:`managed_hot_width` and
    :attr:`hot_stride`.
    """

    max_num_seqs: int
    max_model_len: int
    block_size: int
    device_buffer_size: int
    max_query_tokens_per_request: int
    index_topk: int

    def __post_init__(self) -> None:
        dimensions = {
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "block_size": self.block_size,
            "device_buffer_size": self.device_buffer_size,
            "max_query_tokens_per_request": self.max_query_tokens_per_request,
            "index_topk": self.index_topk,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.device_buffer_size < self.max_topk_union_width:
            raise ValueError(
                "device_buffer_size must cover the complete per-request Top-K "
                f"union, got device_buffer_size={self.device_buffer_size} and "
                f"required={self.max_topk_union_width}."
            )
        if (
            self.max_query_tokens_per_request
            > DSA_SPARSE_MAX_QUERY_LANES
        ):
            raise ValueError(
                "max_query_tokens_per_request exceeds the fused operator "
                f"limit of {DSA_SPARSE_MAX_QUERY_LANES}, got "
                f"{self.max_query_tokens_per_request}."
            )

    @property
    def max_topk_union_width(self) -> int:
        return self.max_query_tokens_per_request * self.index_topk

    @property
    def evictable_hot_slots(self) -> range:
        return range(self.device_buffer_size)

    @property
    def reserved_newest_slots(self) -> range:
        return range(
            self.device_buffer_size,
            self.managed_hot_width,
        )

    @property
    def alignment_padding_slots(self) -> range:
        return range(
            self.managed_hot_width,
            self.hot_stride,
        )

    @property
    def managed_hot_width(self) -> int:
        return self.device_buffer_size + self.max_query_tokens_per_request

    @property
    def hot_stride(self) -> int:
        return _round_up(self.managed_hot_width, self.block_size)

    @property
    def hot_blocks_per_request(self) -> int:
        return self.hot_stride // self.block_size

    @property
    def total_hot_blocks(self) -> int:
        return self.max_num_seqs * self.hot_blocks_per_request

    @property
    def max_blocks_per_request(self) -> int:
        return _round_up(self.max_model_len, self.block_size) // self.block_size


class RequestIndexManager:
    """Allocate one stable Decode request index for each admitted request."""

    def __init__(self, max_num_seqs: int) -> None:
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}.")
        self.max_num_seqs = max_num_seqs
        self._free_indices = deque(range(max_num_seqs))
        self._index_owner: list[Hashable | None] = [None] * max_num_seqs
        self._request_to_index: dict[Hashable, int] = {}

    @property
    def num_free_indices(self) -> int:
        return len(self._free_indices)

    @property
    def active_request_ids(self) -> tuple[Hashable, ...]:
        return tuple(owner for owner in self._index_owner if owner is not None)

    def acquire(self, request_id: Hashable) -> int:
        if request_id in self._request_to_index:
            raise ValueError(
                f"Request {request_id!r} already owns a DSA Sparse request index."
            )
        if not self._free_indices:
            raise RuntimeError("No free DSA Sparse request index is available.")

        request_index = self._free_indices.popleft()
        self._index_owner[request_index] = request_id
        self._request_to_index[request_id] = request_index
        return request_index

    def get_index(self, request_id: Hashable) -> int:
        try:
            return self._request_to_index[request_id]
        except KeyError as exc:
            raise KeyError(
                f"Request {request_id!r} does not own a DSA Sparse request index."
            ) from exc

    def release(self, request_id: Hashable) -> int:
        request_index = self.get_index(request_id)
        del self._request_to_index[request_id]
        self._index_owner[request_index] = None
        self._free_indices.append(request_index)
        return request_index


@dataclass(frozen=True)
class DSASparseResidencyState:
    """Device-resident token-to-hot mapping owned by one residency cohort."""

    cohort: "DSASparseCohortKey"
    token_to_hot: torch.Tensor
    hot_to_token: torch.Tensor
    lru_slots: torch.Tensor

    @classmethod
    def allocate(
        cls,
        config: DSASparseCacheConfig,
        cohort: "DSASparseCohortKey",
        *,
        device: torch.device | str,
    ) -> "DSASparseResidencyState":
        token_to_hot = torch.full(
            (config.max_num_seqs, config.max_model_len),
            INVALID_INDEX,
            dtype=torch.int32,
            device=device,
        )
        hot_to_token = torch.full(
            (config.max_num_seqs, config.device_buffer_size),
            INVALID_INDEX,
            dtype=torch.int32,
            device=device,
        )
        lru_slots = (
            torch.arange(
                config.device_buffer_size,
                dtype=torch.int32,
                device=device,
            )
            .expand(config.max_num_seqs, -1)
            .clone()
        )
        return cls(
            cohort=cohort,
            token_to_hot=token_to_hot,
            hot_to_token=hot_to_token,
            lru_slots=lru_slots,
        )

    def reset_request(self, request_index: int) -> None:
        if not 0 <= request_index < self.token_to_hot.shape[0]:
            raise IndexError(
                f"request_index {request_index} is outside residency capacity "
                f"{self.token_to_hot.shape[0]}."
            )
        self.token_to_hot[request_index].fill_(INVALID_INDEX)
        self.hot_to_token[request_index].fill_(INVALID_INDEX)
        self.lru_slots[request_index].copy_(
            torch.arange(
                self.lru_slots.shape[1],
                dtype=self.lru_slots.dtype,
                device=self.lru_slots.device,
            )
        )


@dataclass(frozen=True)
class DSASparseCohortKey:
    """Ownership boundary for resident state and Main Hot payload."""

    name: str
    role: Literal["target", "draft"]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty.")
        if self.role not in {"target", "draft"}:
            raise ValueError(f"role must be either 'target' or 'draft', got {self.role!r}.")


@dataclass(frozen=True)
class DSASparseLayerKey:
    """Unique layer identity across target and draft residency cohorts."""

    cohort: DSASparseCohortKey
    layer_name: str

    def __post_init__(self) -> None:
        if not self.layer_name:
            raise ValueError("layer_name must not be empty.")


@dataclass(frozen=True)
class DSASparsePlanKey:
    """Shape key shared by eager buffers and the future captured graph path."""

    token_capacity: int
    request_capacity: int
    query_lane_capacity: int
    role: Literal["target", "draft"]

    def __post_init__(self) -> None:
        dimensions = {
            "token_capacity": self.token_capacity,
            "request_capacity": self.request_capacity,
            "query_lane_capacity": self.query_lane_capacity,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        expected_token_capacity = self.request_capacity * self.query_lane_capacity
        if self.token_capacity != expected_token_capacity:
            raise ValueError(
                "token_capacity must equal request_capacity * "
                "query_lane_capacity for a fixed DSA Sparse plan, got "
                f"{self.token_capacity} and {expected_token_capacity}."
            )
        if self.role not in {"target", "draft"}:
            raise ValueError(f"role must be either 'target' or 'draft', got {self.role!r}.")


@dataclass(frozen=True)
class DSASparseBatchMetadata:
    """Role-level fixed inputs shared by every residency cohort.

    Stable request-index query mapping, the Decode block table, the synthetic
    Hot block table, and newest write descriptors describe the batch rather
    than a particular IndexCache cohort. Sharing these tensors prevents one
    metadata staging pass and one HBM copy per cohort.
    """

    key: DSASparsePlanKey
    query_positions: torch.Tensor
    query_to_req_idx: torch.Tensor
    query_to_lane: torch.Tensor
    query_valid_mask: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    hot_block_table: torch.Tensor
    write_global_slots: torch.Tensor
    write_destination_hot_row_ids: torch.Tensor
    write_valid_mask: torch.Tensor

    @classmethod
    def allocate(
        cls,
        config: DSASparseCacheConfig,
        key: DSASparsePlanKey,
        *,
        device: torch.device | str,
        block_table_dtype: torch.dtype = torch.int32,
    ) -> "DSASparseBatchMetadata":
        if key.request_capacity > config.max_num_seqs:
            raise ValueError(
                "request_capacity exceeds max_num_seqs, got "
                f"{key.request_capacity} and {config.max_num_seqs}."
            )
        if key.query_lane_capacity > config.max_query_tokens_per_request:
            raise ValueError(
                "query_lane_capacity exceeds max_query_tokens_per_request, got "
                f"{key.query_lane_capacity} and "
                f"{config.max_query_tokens_per_request}."
            )

        def int_tensor(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.full(
                shape,
                INVALID_INDEX,
                dtype=torch.int32,
                device=device,
            )

        write_shape = (
            key.request_capacity,
            key.query_lane_capacity,
        )
        flat_query_indices = torch.arange(
            key.token_capacity,
            dtype=torch.int32,
            device=device,
        )
        hot_block_table = torch.arange(
            key.request_capacity * config.hot_blocks_per_request,
            dtype=block_table_dtype,
            device=device,
        ).view(
            key.request_capacity,
            config.hot_blocks_per_request,
        )
        return cls(
            key=key,
            query_positions=int_tensor((key.token_capacity,)),
            query_to_req_idx=(
                flat_query_indices // key.query_lane_capacity
            ),
            query_to_lane=flat_query_indices % key.query_lane_capacity,
            query_valid_mask=torch.zeros(
                key.token_capacity,
                dtype=torch.bool,
                device=device,
            ),
            seq_lens=torch.zeros(
                key.request_capacity,
                dtype=torch.int32,
                device=device,
            ),
            block_table=torch.full(
                (
                    key.request_capacity,
                    config.max_blocks_per_request,
                ),
                INVALID_INDEX,
                dtype=block_table_dtype,
                device=device,
            ),
            hot_block_table=hot_block_table,
            write_global_slots=int_tensor(write_shape),
            write_destination_hot_row_ids=int_tensor(write_shape),
            write_valid_mask=torch.zeros(
                write_shape,
                dtype=torch.bool,
                device=device,
            ),
        )


@dataclass(frozen=True)
class DSASparsePlan:
    """Cohort-private lookup outputs and workspace for one execution shape."""

    key: DSASparsePlanKey
    batch_metadata: DSASparseBatchMetadata
    valid_topk_counts: torch.Tensor
    topk_positions: torch.Tensor
    resolved_hot_indices: torch.Tensor
    miss_mask: torch.Tensor
    workspace: torch.Tensor

    @classmethod
    def allocate(
        cls,
        config: DSASparseCacheConfig,
        key: DSASparsePlanKey,
        *,
        device: torch.device | str,
        block_table_dtype: torch.dtype = torch.int32,
        batch_metadata: DSASparseBatchMetadata | None = None,
    ) -> "DSASparsePlan":
        if batch_metadata is None:
            batch_metadata = DSASparseBatchMetadata.allocate(
                config,
                key,
                device=device,
                block_table_dtype=block_table_dtype,
            )
        elif batch_metadata.key != key:
            raise ValueError(
                "Shared DSA Sparse batch metadata must use the same plan key."
            )

        topk_shape = (key.token_capacity, config.index_topk)
        return cls(
            key=key,
            batch_metadata=batch_metadata,
            valid_topk_counts=torch.zeros(
                key.token_capacity,
                dtype=torch.int32,
                device=device,
            ),
            topk_positions=torch.full(
                topk_shape,
                INVALID_INDEX,
                dtype=torch.int32,
                device=device,
            ),
            resolved_hot_indices=torch.full(
                topk_shape,
                INVALID_INDEX,
                dtype=torch.int32,
                device=device,
            ),
            miss_mask=torch.zeros(
                topk_shape,
                dtype=torch.bool,
                device=device,
            ),
            workspace=torch.empty(
                (
                    key.request_capacity,
                    dsa_sparse_lookup_workspace_stride(
                        config.device_buffer_size
                    ),
                ),
                dtype=torch.int32,
                device=device,
            ),
        )

    @property
    def query_positions(self) -> torch.Tensor:
        return self.batch_metadata.query_positions

    @property
    def query_to_req_idx(self) -> torch.Tensor:
        return self.batch_metadata.query_to_req_idx

    @property
    def query_to_lane(self) -> torch.Tensor:
        return self.batch_metadata.query_to_lane

    @property
    def query_valid_mask(self) -> torch.Tensor:
        return self.batch_metadata.query_valid_mask

    @property
    def seq_lens(self) -> torch.Tensor:
        return self.batch_metadata.seq_lens

    @property
    def block_table(self) -> torch.Tensor:
        return self.batch_metadata.block_table

    @property
    def hot_block_table(self) -> torch.Tensor:
        return self.batch_metadata.hot_block_table

    @property
    def write_global_slots(self) -> torch.Tensor:
        return self.batch_metadata.write_global_slots

    @property
    def write_destination_hot_row_ids(self) -> torch.Tensor:
        return self.batch_metadata.write_destination_hot_row_ids

    @property
    def write_valid_mask(self) -> torch.Tensor:
        return self.batch_metadata.write_valid_mask


@dataclass(frozen=True)
class DSASparseLayerLayout:
    """Per-token Main MLA plane layout for one local sparse layer."""

    layer_name: str
    plane_dtypes: tuple[torch.dtype, ...]
    plane_row_shapes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.layer_name:
            raise ValueError("layer_name must not be empty.")
        if not self.plane_dtypes:
            raise ValueError("At least one Main MLA cache plane is required.")
        if len(self.plane_dtypes) != len(self.plane_row_shapes):
            raise ValueError("plane_dtypes and plane_row_shapes must have the same length.")


@dataclass(frozen=True)
class DSASparseLayerHotCache:
    layer_name: str
    planes: tuple[torch.Tensor, ...]

    @classmethod
    def allocate(
        cls,
        layout: DSASparseLayerLayout,
        config: DSASparseCacheConfig,
        *,
        device: torch.device | str,
    ) -> "DSASparseLayerHotCache":
        planes = tuple(
            torch.empty(
                (
                    config.total_hot_blocks,
                    config.block_size,
                    *row_shape,
                ),
                dtype=dtype,
                device=device,
            )
            for dtype, row_shape in zip(
                layout.plane_dtypes,
                layout.plane_row_shapes,
            )
        )
        return cls(layer_name=layout.layer_name, planes=planes)


class DSASparseLookupUpdateOperator(Protocol):
    """Mutation contract for the single fused A5 metadata custom op."""

    def lookup_update(
        self,
        *,
        state: DSASparseResidencyState,
        plan: DSASparsePlan,
    ) -> None: ...


class UnimplementedDSASparseLookupUpdateOperator:
    """Explicit fail-fast implementation for missing custom-op builds."""

    def lookup_update(self, **_: object) -> None:
        raise NotImplementedError(
            "DSA Sparse lookup/update operator is not implemented."
        )


@dataclass(frozen=True)
class DSASparseLayerBinding:
    """Per-layer payload and I/O resources.

    Layers in the same cohort share only the plan and residency state. Their
    Hot Cache planes, backend region, and completion resources stay separate.
    """

    layer_name: str
    cohort: DSASparseCohortKey
    hot_cache: DSASparseLayerHotCache
    io_context: object
    io_region: object
    io_completion: object

    def __post_init__(self) -> None:
        if self.layer_name != self.hot_cache.layer_name:
            raise ValueError("Layer binding name must match its Hot Cache layer name.")

    @property
    def key(self) -> DSASparseLayerKey:
        return DSASparseLayerKey(
            cohort=self.cohort,
            layer_name=self.layer_name,
        )


@dataclass(frozen=True)
class DSASparseCohort:
    key: DSASparseCohortKey
    leader_layer: str
    state: DSASparseResidencyState
    plans: Mapping[DSASparsePlanKey, DSASparsePlan]

    def __post_init__(self) -> None:
        if self.state.cohort != self.key:
            raise ValueError("Residency state owner must match the cohort resource key.")
        for plan_key in self.plans:
            if plan_key.role != self.key.role:
                raise ValueError("Plan role must match the residency cohort role.")


@dataclass
class DSASparseResolution:
    hot_main_cache: tuple[torch.Tensor, ...]
    local_sparse_indices: torch.Tensor
    hot_block_table: torch.Tensor


@dataclass
class DSASparseEagerStep:
    cohort: DSASparseCohort
    plan: DSASparsePlan
    request_ids: tuple[Hashable, ...]
    request_indices: tuple[int, ...]
    lookup_complete: bool = False
    newest_written_layers: set[str] = field(default_factory=set)
    io_completed_layers: set[str] = field(default_factory=set)
    completed_layers: set[str] = field(default_factory=set)


class DSASparseEagerCoordinator:
    """Single eager entry point for the fixed lookup/I/O/SFA sequence."""

    def __init__(
        self,
        config: DSASparseCacheConfig,
        *,
        index_operator: DSASparseLookupUpdateOperator,
        io_operator: DSASparseIOOperator,
        request_index_manager: RequestIndexManager | None = None,
    ) -> None:
        self.config = config
        self.index_operator = index_operator
        self.io_operator = io_operator
        self.request_index_manager = (
            request_index_manager
            or RequestIndexManager(config.max_num_seqs)
        )
        if (
            self.request_index_manager.max_num_seqs
            != config.max_num_seqs
        ):
            raise ValueError(
                "The request-index pool capacity must equal max_num_seqs."
            )
        self._cohorts: dict[DSASparseCohortKey, DSASparseCohort] = {}
        self._layers: dict[DSASparseLayerKey, DSASparseLayerBinding] = {}
        self._active_steps: dict[DSASparseCohortKey, DSASparseEagerStep] = {}
        self._hot_plane_addresses: set[int] = set()
        self._region_identities: set[object] = set()
        self._completion_identities: set[object] = set()
        self._frozen = False
        self._failure: Exception | None = None

    def register_cohort(
        self,
        cohort: DSASparseCohort,
    ) -> None:
        self._require_mutable()
        if cohort.key in self._cohorts:
            raise ValueError(f"DSA Sparse cohort {cohort.key!r} is already registered.")
        request_capacity = self.request_index_manager.max_num_seqs
        state_tensors = (
            cohort.state.token_to_hot,
            cohort.state.hot_to_token,
            cohort.state.lru_slots,
        )
        if any(
            tensor.shape[0] != request_capacity
            for tensor in state_tensors
        ):
            raise ValueError(
                "Every residency-state tensor must cover the complete "
                "request-index pool."
            )
        if any(
            plan_key.request_capacity != request_capacity
            for plan_key in cohort.plans
        ):
            raise ValueError(
                "Every DSA Sparse plan must cover the complete "
                "request-index pool."
            )
        self._cohorts[cohort.key] = cohort

    def acquire_request(self, request_id: Hashable) -> int:
        self._require_healthy()
        request_index = self.request_index_manager.acquire(request_id)
        try:
            for cohort in self._cohorts.values():
                cohort.state.reset_request(request_index)
        except BaseException:
            self.request_index_manager.release(request_id)
            raise
        return request_index

    def request_index(self, request_id: Hashable) -> int:
        self._require_healthy()
        return self.request_index_manager.get_index(request_id)

    def release_request(self, request_id: Hashable) -> int:
        self._require_healthy()
        self.assert_request_idle(request_id)
        return self.request_index_manager.release(request_id)

    def assert_request_idle(self, request_id: Hashable) -> None:
        if any(request_id in step.request_ids for step in self._active_steps.values()):
            raise RuntimeError("Cannot release a DSA Sparse request while its step has pending layer I/O.")

    def register_layer(
        self,
        binding: DSASparseLayerBinding,
    ) -> None:
        self._require_mutable()
        if binding.key in self._layers:
            raise ValueError(f"DSA Sparse layer {binding.key!r} is already registered.")
        cohort = self._get_cohort(binding.cohort)
        if not cohort.leader_layer:
            raise ValueError("DSA Sparse cohort leader layer must not be empty.")
        plane_addresses = {plane.data_ptr() for plane in binding.hot_cache.planes}
        if plane_addresses & self._hot_plane_addresses:
            raise ValueError("Each DSA Sparse layer must own independent Hot Cache planes.")
        region_identity = self._resource_identity(binding.io_region)
        if region_identity in self._region_identities:
            raise ValueError("Each DSA Sparse layer must own an independent I/O region.")
        completion_identity = self._resource_identity(
            binding.io_completion
        )
        if completion_identity in self._completion_identities:
            raise ValueError(
                "Each DSA Sparse layer must own an independent unified I/O "
                "completion resource."
            )
        self._layers[binding.key] = binding
        self._hot_plane_addresses.update(plane_addresses)
        self._region_identities.add(region_identity)
        self._completion_identities.add(completion_identity)
        if dsa_sparse_probe.is_enabled():
            dsa_sparse_probe.emit(
                "hot_cache_registered",
                cohort=binding.cohort.name,
                layer=binding.layer_name,
                hot_cache_ptrs=[
                    plane.data_ptr()
                    for plane in binding.hot_cache.planes
                ],
                hot_cache_shapes=[
                    list(plane.shape)
                    for plane in binding.hot_cache.planes
                ],
            )

    def freeze(self) -> None:
        frozen_cohorts: dict[DSASparseCohortKey, DSASparseCohort] = {}
        for cohort in self._cohorts.values():
            try:
                leader_binding = self._layers[
                    DSASparseLayerKey(
                        cohort=cohort.key,
                        layer_name=cohort.leader_layer,
                    )
                ]
            except KeyError as exc:
                raise ValueError(f"DSA Sparse cohort leader {cohort.leader_layer!r} is not registered.") from exc
            if leader_binding.cohort != cohort.key:
                raise ValueError("DSA Sparse cohort leader must belong to its residency cohort.")
            frozen_cohorts[cohort.key] = DSASparseCohort(
                key=cohort.key,
                leader_layer=cohort.leader_layer,
                state=cohort.state,
                plans=MappingProxyType(dict(cohort.plans)),
            )
        self._cohorts = frozen_cohorts
        self._frozen = True

    def get_layer_binding(
        self,
        cohort_key: DSASparseCohortKey,
        layer_name: str,
    ) -> DSASparseLayerBinding:
        layer_key = DSASparseLayerKey(
            cohort=cohort_key,
            layer_name=layer_name,
        )
        try:
            return self._layers[layer_key]
        except KeyError as exc:
            raise KeyError(f"DSA Sparse layer {layer_key!r} is not registered.") from exc

    def get_cohort(
        self,
        cohort_key: DSASparseCohortKey,
    ) -> DSASparseCohort:
        return self._get_cohort(cohort_key)

    def begin_step(
        self,
        cohort_key: DSASparseCohortKey,
        plan_key: DSASparsePlanKey,
        *,
        request_ids: list[Hashable],
        request_indices: list[int],
        query_positions: torch.Tensor,
        query_valid_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        stage_batch_metadata: bool = True,
    ) -> DSASparseEagerStep:
        self._require_healthy()
        if not self._frozen:
            raise RuntimeError("DSA Sparse coordinator must be frozen before execution.")
        cohort = self._get_cohort(cohort_key)
        try:
            plan = cohort.plans[plan_key]
        except KeyError as exc:
            raise KeyError(f"DSA Sparse plan {plan_key!r} is not registered for cohort {cohort_key!r}.") from exc
        if cohort_key in self._active_steps:
            raise RuntimeError(
                f"Only one DSA Sparse step may mutate a residency cohort at a time, cohort={cohort_key!r}."
            )
        if len(request_indices) != len(request_ids):
            raise ValueError(
                "request_indices must contain one stable index per request."
            )
        if len(set(request_indices)) != len(request_indices):
            raise ValueError("Active DSA Sparse request indices must be unique.")
        for request_id, request_index in zip(
            request_ids,
            request_indices,
        ):
            expected_index = self.request_index(request_id)
            if request_index != expected_index:
                raise RuntimeError(
                    f"Request {request_id!r} uses request index "
                    f"{request_index}, expected {expected_index}."
                )

        if stage_batch_metadata:
            self._copy_exact(
                query_positions,
                plan.query_positions,
                "query_positions",
            )
            self._copy_exact(
                query_valid_mask,
                plan.query_valid_mask,
                "query_valid_mask",
            )
            self._copy_exact(seq_lens, plan.seq_lens, "seq_lens")
            self._copy_exact(
                block_table,
                plan.block_table,
                "block_table",
            )
            self._prepare_batch_metadata(plan)
        plan.topk_positions.fill_(INVALID_INDEX)
        plan.valid_topk_counts.zero_()
        plan.resolved_hot_indices.fill_(INVALID_INDEX)
        plan.miss_mask.zero_()
        step = DSASparseEagerStep(
            cohort=cohort,
            plan=plan,
            request_ids=tuple(request_ids),
            request_indices=tuple(request_indices),
        )
        self._active_steps[cohort_key] = step
        return step

    def submit_newest_write(
        self,
        step: DSASparseEagerStep,
        layer_name: str,
    ) -> None:
        self._assert_active_step(step)
        self._get_step_layer(step, layer_name)
        if layer_name in step.newest_written_layers:
            raise RuntimeError(
                f"Newest Hot Cache payload was already marked written for "
                f"{layer_name!r}."
            )
        step.newest_written_layers.add(layer_name)

    def prepare_lookup(
        self,
        step: DSASparseEagerStep,
        *,
        topk_positions: torch.Tensor,
        valid_topk_counts: torch.Tensor,
    ) -> None:
        self._assert_active_step(step)
        if step.lookup_complete:
            raise RuntimeError("Each DSA Sparse cohort performs lookup only once per step.")
        leader_layer = step.cohort.leader_layer
        self._get_step_layer(step, leader_layer)
        if leader_layer not in step.newest_written_layers:
            raise RuntimeError("The DSA Sparse cohort leader must submit its newest Main KV write before lookup.")
        self._copy_exact(
            topk_positions,
            step.plan.topk_positions,
            "topk_positions",
        )
        self._copy_exact(
            valid_topk_counts,
            step.plan.valid_topk_counts,
            "valid_topk_counts",
        )
        try:
            self.index_operator.lookup_update(
                state=step.cohort.state,
                plan=step.plan,
            )
        except Exception as exc:
            self._poison(exc)
            raise
        step.lookup_complete = True

    def run_layer_attention(
        self,
        step: DSASparseEagerStep,
        layer_name: str,
        attention: Callable[[DSASparseResolution], torch.Tensor],
    ) -> torch.Tensor:
        self._assert_active_step(step)
        binding = self._get_step_layer(step, layer_name)
        if not step.lookup_complete:
            raise RuntimeError("DSA Sparse lookup must complete before layer I/O.")
        if layer_name not in step.newest_written_layers:
            raise RuntimeError("Newest Main KV write must be submitted before lookup/I/O/SFA.")
        if layer_name in step.completed_layers:
            raise RuntimeError(f"DSA Sparse layer {layer_name!r} already completed this step.")

        try:
            self.io_operator.dsa_sparse_io(
                context=binding.io_context,
                region=binding.io_region,
                topk_positions=step.plan.topk_positions,
                resolved_hot_indices=step.plan.resolved_hot_indices,
                miss_mask=step.plan.miss_mask,
                query_to_req_idx=step.plan.query_to_req_idx,
                block_table=step.plan.block_table,
                write_global_slots=step.plan.write_global_slots,
                write_destination_hot_row_ids=(
                    step.plan.write_destination_hot_row_ids
                ),
                write_valid_mask=step.plan.write_valid_mask,
                hot_planes=binding.hot_cache.planes,
                completion=binding.io_completion,
            )
        except Exception as exc:
            self._poison(exc)
            raise
        step.io_completed_layers.add(layer_name)
        resolution = DSASparseResolution(
            hot_main_cache=binding.hot_cache.planes,
            local_sparse_indices=step.plan.resolved_hot_indices,
            hot_block_table=step.plan.hot_block_table,
        )
        output = attention(resolution)
        step.completed_layers.add(layer_name)
        return output

    def finish_step(self, step: DSASparseEagerStep) -> None:
        self._assert_active_step(step)
        expected_layers = {
            layer_key.layer_name for layer_key, binding in self._layers.items() if binding.cohort == step.cohort.key
        }
        if step.completed_layers != expected_layers:
            pending_layers = sorted(expected_layers - step.completed_layers)
            raise RuntimeError(
                "Cannot finish DSA Sparse step before every layer completes "
                f"its unified I/O and SFA call, pending layers: "
                f"{pending_layers}."
            )
        del self._active_steps[step.cohort.key]

    def abort_step(self, step: DSASparseEagerStep) -> None:
        """Retire a failed eager step.

        The unified I/O operator establishes its read/write completion
        dependency within the call, so there is no second eager wait phase.
        """
        self._assert_active_step(step)
        del self._active_steps[step.cohort.key]

    def _prepare_batch_metadata(
        self,
        plan: DSASparsePlan,
    ) -> None:
        """Fill newest-write descriptors using stable request indices."""

        plan.write_global_slots.fill_(INVALID_INDEX)
        plan.write_destination_hot_row_ids.fill_(INVALID_INDEX)
        plan.write_valid_mask.zero_()

        query_req_indices = plan.query_to_req_idx.to(torch.long)
        query_lanes = plan.query_to_lane.to(torch.long)
        query_positions = plan.query_positions
        query_seq_lens = plan.seq_lens[query_req_indices]
        valid = (
            plan.query_valid_mask
            & (query_positions >= 0)
            & (query_positions < query_seq_lens)
            & (query_positions < self.config.max_model_len)
        )

        block_indices = torch.div(
            query_positions,
            self.config.block_size,
            rounding_mode="floor",
        )
        safe_block_indices = block_indices.clamp(
            min=0,
            max=plan.block_table.shape[1] - 1,
        ).to(torch.long)
        physical_blocks = plan.block_table[
            query_req_indices,
            safe_block_indices,
        ].to(torch.int32)
        valid &= physical_blocks >= 0

        destination_rows = (
            plan.query_to_req_idx
            * self.config.hot_stride
            + self.config.device_buffer_size
            + plan.query_to_lane
        ).to(torch.int32)
        global_slots = (
            physical_blocks * self.config.block_size
            + torch.remainder(
                query_positions,
                self.config.block_size,
            )
        ).to(torch.int32)
        invalid_values = torch.full_like(
            query_positions,
            INVALID_INDEX,
        )
        destination_rows = torch.where(
            valid,
            destination_rows,
            invalid_values,
        )
        global_slots = torch.where(
            valid,
            global_slots,
            invalid_values,
        )

        write_indices = (
            query_req_indices * plan.key.query_lane_capacity
            + query_lanes
        )
        plan.write_destination_hot_row_ids.view(-1).index_copy_(
            0,
            write_indices,
            destination_rows,
        )
        plan.write_global_slots.view(-1).index_copy_(
            0,
            write_indices,
            global_slots,
        )
        plan.write_valid_mask.view(-1).index_copy_(
            0,
            write_indices,
            valid,
        )

    def _get_cohort(
        self,
        cohort_key: DSASparseCohortKey,
    ) -> DSASparseCohort:
        try:
            return self._cohorts[cohort_key]
        except KeyError as exc:
            raise KeyError(f"DSA Sparse cohort {cohort_key!r} is not registered.") from exc

    def _get_step_layer(
        self,
        step: DSASparseEagerStep,
        layer_name: str,
    ) -> DSASparseLayerBinding:
        binding = self.get_layer_binding(step.cohort.key, layer_name)
        if binding.cohort != step.cohort.key:
            raise ValueError(f"Layer {layer_name!r} does not belong to step cohort {step.cohort.key!r}.")
        return binding

    def _assert_active_step(self, step: DSASparseEagerStep) -> None:
        if self._active_steps.get(step.cohort.key) is not step:
            raise RuntimeError("DSA Sparse step is not the active cohort owner.")

    def _require_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("DSA Sparse coordinator resources are frozen.")

    def _require_healthy(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "DSA Sparse coordinator is poisoned after an operator "
                "failure."
            ) from self._failure

    def _poison(self, failure: Exception) -> None:
        if self._failure is None:
            self._failure = failure

    @staticmethod
    def _copy_exact(
        source: torch.Tensor,
        destination: torch.Tensor,
        name: str,
    ) -> None:
        if source.shape != destination.shape:
            raise ValueError(f"{name} shape must be {tuple(destination.shape)}, got {tuple(source.shape)}.")
        destination.copy_(source)

    @staticmethod
    def _resource_identity(resource: object) -> object:
        try:
            hash(resource)
        except TypeError:
            return ("identity", id(resource))
        return ("value", type(resource), resource)


@dataclass(frozen=True)
class DSASparseMainWriteTarget:
    hot_main_cache: tuple[torch.Tensor, ...]
    reserved_slot_mapping: torch.Tensor


class DSASparseEagerAttentionContext(Protocol):
    """Layer-facing context consumed by the existing SFA wrapper."""

    @property
    def num_sfa_queries(self) -> int: ...

    def main_write_target(
        self,
        layer_name: str,
    ) -> DSASparseMainWriteTarget: ...

    def submit_newest_write(self, layer_name: str) -> None: ...

    def run_layer_attention(
        self,
        layer_name: str,
        semantic_topk_positions: torch.Tensor,
        attention: Callable[[DSASparseResolution], torch.Tensor],
    ) -> torch.Tensor: ...


class DSASparseEagerBatchContext:
    """Execution-scoped adapter between dynamic eager batches and fixed plans."""

    def __init__(
        self,
        *,
        coordinator: DSASparseEagerCoordinator,
        step: DSASparseEagerStep,
        active_plan_indices: torch.Tensor,
        active_request_indices: torch.Tensor,
        sfa_slot_mapping: torch.Tensor,
        sfa_local_sparse_indices: torch.Tensor,
    ) -> None:
        self.coordinator = coordinator
        self.step = step
        self.active_plan_indices = active_plan_indices
        self.active_request_indices = active_request_indices
        self._sfa_slot_mapping = sfa_slot_mapping
        self._sfa_local_sparse_indices = sfa_local_sparse_indices
        self._closed = False

    @classmethod
    def begin(
        cls,
        coordinator: DSASparseEagerCoordinator,
        cohort_key: DSASparseCohortKey,
        plan_key: DSASparsePlanKey,
        *,
        request_ids: list[Hashable],
        request_indices: list[int],
        query_positions: torch.Tensor,
        query_counts: list[int],
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_sfa_queries: int | None = None,
        stage_batch_metadata: bool = True,
    ) -> "DSASparseEagerBatchContext":
        cohort = coordinator.get_cohort(cohort_key)
        try:
            plan = cohort.plans[plan_key]
        except KeyError as exc:
            raise KeyError(f"DSA Sparse plan {plan_key!r} is not registered for cohort {cohort_key!r}.") from exc
        cls._validate_dynamic_inputs(
            plan,
            request_ids=request_ids,
            request_indices=request_indices,
            query_positions=query_positions,
            query_counts=query_counts,
            seq_lens=seq_lens,
            block_table=block_table,
            num_sfa_queries=num_sfa_queries,
        )

        active_indices = [
            request_index * plan.key.query_lane_capacity + lane
            for request_index, count in zip(
                request_indices,
                query_counts,
            )
            for lane in range(count)
        ]
        num_active_queries = len(active_indices)
        if num_sfa_queries is None:
            num_sfa_queries = num_active_queries
        active_plan_indices = torch.tensor(
            active_indices,
            dtype=torch.long,
            device=plan.query_positions.device,
        )
        active_request_indices = torch.tensor(
            request_indices,
            dtype=torch.long,
            device=plan.seq_lens.device,
        )
        if stage_batch_metadata:
            fixed_query_positions = torch.full_like(
                plan.query_positions,
                INVALID_INDEX,
            )
            fixed_query_valid_mask = torch.zeros_like(
                plan.query_valid_mask
            )
            fixed_query_positions.index_copy_(
                0,
                active_plan_indices,
                query_positions.to(
                    device=fixed_query_positions.device,
                    dtype=fixed_query_positions.dtype,
                ),
            )
            fixed_query_valid_mask.index_fill_(
                0,
                active_plan_indices,
                True,
            )

            fixed_seq_lens = torch.zeros_like(plan.seq_lens)
            fixed_seq_lens.index_copy_(
                0,
                active_request_indices,
                seq_lens,
            )
            fixed_block_table = torch.full_like(
                plan.block_table,
                INVALID_INDEX,
            )
            fixed_block_table.index_copy_(
                0,
                active_request_indices,
                block_table,
            )
        else:
            fixed_query_positions = plan.query_positions
            fixed_query_valid_mask = plan.query_valid_mask
            fixed_seq_lens = plan.seq_lens
            fixed_block_table = plan.block_table

        sfa_slot_mapping = torch.full(
            (num_sfa_queries,),
            INVALID_INDEX,
            dtype=plan.write_destination_hot_row_ids.dtype,
            device=plan.write_destination_hot_row_ids.device,
        )
        sfa_local_sparse_indices = torch.full(
            (
                num_sfa_queries,
                plan.topk_positions.shape[1],
            ),
            INVALID_INDEX,
            dtype=plan.resolved_hot_indices.dtype,
            device=plan.resolved_hot_indices.device,
        )
        step = coordinator.begin_step(
            cohort_key,
            plan_key,
            request_ids=request_ids,
            request_indices=request_indices,
            query_positions=fixed_query_positions,
            query_valid_mask=fixed_query_valid_mask,
            seq_lens=fixed_seq_lens,
            block_table=fixed_block_table,
            stage_batch_metadata=stage_batch_metadata,
        )
        try:
            sfa_slot_mapping[:num_active_queries] = (
                plan.write_destination_hot_row_ids.view(-1)[
                    active_plan_indices
                ]
            )
        except BaseException as exc:
            try:
                coordinator.abort_step(step)
            except BaseException as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"DSA Sparse begin cleanup also failed: {cleanup_exc!r}")
            raise
        return cls(
            coordinator=coordinator,
            step=step,
            active_plan_indices=active_plan_indices,
            active_request_indices=active_request_indices,
            sfa_slot_mapping=sfa_slot_mapping,
            sfa_local_sparse_indices=sfa_local_sparse_indices,
        )

    @property
    def num_active_queries(self) -> int:
        return self.active_plan_indices.shape[0]

    @property
    def num_sfa_queries(self) -> int:
        return self._sfa_slot_mapping.shape[0]

    def main_write_target(
        self,
        layer_name: str,
    ) -> DSASparseMainWriteTarget:
        self._require_open()
        binding = self.coordinator.get_layer_binding(
            self.step.cohort.key,
            layer_name,
        )
        return DSASparseMainWriteTarget(
            hot_main_cache=binding.hot_cache.planes,
            reserved_slot_mapping=self._sfa_slot_mapping,
        )

    def submit_newest_write(self, layer_name: str) -> None:
        self._require_open()
        self.coordinator.submit_newest_write(self.step, layer_name)

    def run_layer_attention(
        self,
        layer_name: str,
        semantic_topk_positions: torch.Tensor,
        attention: Callable[[DSASparseResolution], torch.Tensor],
    ) -> torch.Tensor:
        self._require_open()
        semantic_topk_positions = self._normalize_topk(semantic_topk_positions)
        if not self.step.lookup_complete:
            if layer_name != self.step.cohort.leader_layer:
                raise RuntimeError("DSA Sparse cohort leader must resolve semantic Top-K before follower layers.")
            fixed_topk_positions = torch.full_like(
                self.step.plan.topk_positions,
                INVALID_INDEX,
            )
            fixed_valid_topk_counts = torch.zeros_like(self.step.plan.valid_topk_counts)
            fixed_topk_positions.index_copy_(
                0,
                self.active_plan_indices,
                semantic_topk_positions,
            )
            fixed_valid_topk_counts.index_copy_(
                0,
                self.active_plan_indices,
                (semantic_topk_positions >= 0).sum(
                    dim=-1,
                    dtype=fixed_valid_topk_counts.dtype,
                ),
            )
            self.coordinator.prepare_lookup(
                self.step,
                topk_positions=fixed_topk_positions,
                valid_topk_counts=fixed_valid_topk_counts,
            )

        def active_attention(
            resolution: DSASparseResolution,
        ) -> torch.Tensor:
            self._sfa_local_sparse_indices.fill_(INVALID_INDEX)
            self._sfa_local_sparse_indices[: self.num_active_queries] = resolution.local_sparse_indices[
                self.active_plan_indices
            ]
            active_resolution = DSASparseResolution(
                hot_main_cache=resolution.hot_main_cache,
                local_sparse_indices=self._sfa_local_sparse_indices,
                hot_block_table=resolution.hot_block_table[
                    self.active_request_indices
                ],
            )
            return attention(active_resolution)

        return self.coordinator.run_layer_attention(
            self.step,
            layer_name,
            active_attention,
        )

    def finish(self) -> None:
        self._require_open()
        self.coordinator.finish_step(self.step)
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self.coordinator.abort_step(self.step)
        finally:
            self._closed = True

    def _normalize_topk(
        self,
        semantic_topk_positions: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_topk_positions.ndim == 3 and semantic_topk_positions.shape[1] == 1:
            semantic_topk_positions = semantic_topk_positions.squeeze(1)
        expected_shape = (
            self.num_sfa_queries,
            self.step.plan.topk_positions.shape[1],
        )
        if semantic_topk_positions.shape != expected_shape:
            raise ValueError(
                f"semantic_topk_positions shape must be {expected_shape}, got {tuple(semantic_topk_positions.shape)}."
            )
        return semantic_topk_positions[: self.num_active_queries]

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("DSA Sparse eager batch context is closed.")

    @staticmethod
    def _validate_dynamic_inputs(
        plan: DSASparsePlan,
        *,
        request_ids: list[Hashable],
        request_indices: list[int],
        query_positions: torch.Tensor,
        query_counts: list[int],
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_sfa_queries: int | None,
    ) -> None:
        num_requests = len(request_ids)
        if num_requests > plan.key.request_capacity:
            raise ValueError("request_ids exceeds the DSA Sparse plan request capacity.")
        if len(query_counts) != num_requests:
            raise ValueError("query_counts must contain one entry per request.")
        if len(request_indices) != num_requests:
            raise ValueError(
                "request_indices must contain one entry per request."
            )
        if len(set(request_indices)) != num_requests:
            raise ValueError("request_indices must be unique.")
        if any(
            isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or not 0 <= request_index < plan.key.request_capacity
            for request_index in request_indices
        ):
            raise ValueError(
                "Each request index must fit the DSA Sparse request capacity."
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= plan.key.query_lane_capacity
            for count in query_counts
        ):
            raise ValueError("Each query count must fit the DSA Sparse query-lane capacity.")
        if query_positions.shape != (sum(query_counts),):
            raise ValueError("query_positions must contain exactly the active query lanes.")
        if num_sfa_queries is not None:
            if isinstance(num_sfa_queries, bool) or not isinstance(num_sfa_queries, int):
                raise ValueError("num_sfa_queries must be an integer.")
            if num_sfa_queries < sum(query_counts):
                raise ValueError("num_sfa_queries must cover every active query.")
        if seq_lens.shape != (num_requests,):
            raise ValueError("seq_lens must contain one value per request.")
        expected_block_shape = (
            num_requests,
            plan.block_table.shape[1],
        )
        if block_table.shape != expected_block_shape:
            raise ValueError(f"block_table shape must be {expected_block_shape}, got {tuple(block_table.shape)}.")


class DSASparseEagerContextRouter:
    """Route each sparse layer to its IndexCache residency cohort."""

    def __init__(
        self,
        layer_contexts: Mapping[str, DSASparseEagerBatchContext],
    ) -> None:
        if not layer_contexts:
            raise ValueError("DSA Sparse eager context router requires at least one layer.")
        contexts = tuple({id(context): context for context in layer_contexts.values()}.values())
        num_sfa_queries = contexts[0].num_sfa_queries
        if any(context.num_sfa_queries != num_sfa_queries for context in contexts[1:]):
            raise ValueError("All DSA Sparse layer contexts must use the same SFA query view.")
        self._layer_contexts = MappingProxyType(dict(layer_contexts))
        self._contexts = contexts
        self._num_sfa_queries = num_sfa_queries
        self._closed = False

    @property
    def num_sfa_queries(self) -> int:
        return self._num_sfa_queries

    @property
    def contexts(self) -> tuple[DSASparseEagerBatchContext, ...]:
        return self._contexts

    def context_for(
        self,
        layer_name: str,
    ) -> DSASparseEagerBatchContext:
        try:
            return self._layer_contexts[layer_name]
        except KeyError as exc:
            raise KeyError(f"DSA Sparse layer {layer_name!r} has no eager cohort context.") from exc

    def main_write_target(
        self,
        layer_name: str,
    ) -> DSASparseMainWriteTarget:
        return self.context_for(layer_name).main_write_target(layer_name)

    def submit_newest_write(self, layer_name: str) -> None:
        self.context_for(layer_name).submit_newest_write(layer_name)

    def run_layer_attention(
        self,
        layer_name: str,
        semantic_topk_positions: torch.Tensor,
        attention: Callable[[DSASparseResolution], torch.Tensor],
    ) -> torch.Tensor:
        return self.context_for(layer_name).run_layer_attention(
            layer_name,
            semantic_topk_positions,
            attention,
        )

    def finish(self) -> None:
        if self._closed:
            raise RuntimeError("DSA Sparse eager context router is closed.")
        try:
            for context in self._contexts:
                context.finish()
        except BaseException as exc:
            try:
                self._abort_contexts()
            except BaseException as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"DSA Sparse cohort cleanup also failed: {cleanup_exc!r}")
            raise
        finally:
            self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._abort_contexts()
        finally:
            self._closed = True

    def _abort_contexts(self) -> None:
        first_error: BaseException | None = None
        for context in self._contexts:
            try:
                context.abort()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise RuntimeError("Failed to abort all DSA Sparse eager cohort contexts.") from first_error
