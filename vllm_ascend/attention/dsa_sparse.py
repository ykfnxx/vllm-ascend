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

from vllm_ascend.attention.dsa_sparse_io import DSASparseIOOperator

INVALID_INDEX = -1


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


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
    def hot_blocks_per_seat(self) -> int:
        return self.hot_stride // self.block_size

    @property
    def total_hot_blocks(self) -> int:
        return self.max_num_seqs * self.hot_blocks_per_seat

    @property
    def max_blocks_per_request(self) -> int:
        return _round_up(self.max_model_len, self.block_size) // self.block_size


@dataclass(frozen=True)
class CacheSeatLease:
    seat: int
    epoch: int


@dataclass(frozen=True)
class DSASparseRowMapping:
    """Fixed-size row-to-seat inputs for one eager Decode step."""

    row_active: torch.Tensor
    row_to_cache_seat: torch.Tensor
    row_seat_epoch: torch.Tensor

    @classmethod
    def allocate(
        cls,
        request_capacity: int,
        *,
        device: torch.device | str,
    ) -> "DSASparseRowMapping":
        if request_capacity <= 0:
            raise ValueError(f"request_capacity must be positive, got {request_capacity}.")
        row_to_cache_seat = torch.full(
            (request_capacity,),
            INVALID_INDEX,
            dtype=torch.int32,
            device=device,
        )
        return cls(
            row_active=torch.zeros(
                request_capacity,
                dtype=torch.bool,
                device=device,
            ),
            row_to_cache_seat=row_to_cache_seat,
            row_seat_epoch=torch.full_like(
                row_to_cache_seat,
                INVALID_INDEX,
            ),
        )


class CacheSeatManager:
    """Own stable Decode cache seats on the request control plane."""

    def __init__(self, max_num_seqs: int) -> None:
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}.")
        self.max_num_seqs = max_num_seqs
        self._free_seats = deque(range(max_num_seqs))
        self._seat_owner: list[Hashable | None] = [None] * max_num_seqs
        self._seat_epoch = [0] * max_num_seqs
        self._request_to_lease: dict[Hashable, CacheSeatLease] = {}

    @property
    def num_free_seats(self) -> int:
        return len(self._free_seats)

    @property
    def active_request_ids(self) -> tuple[Hashable, ...]:
        return tuple(owner for owner in self._seat_owner if owner is not None)

    def acquire(self, request_id: Hashable) -> CacheSeatLease:
        if request_id in self._request_to_lease:
            raise ValueError(f"Request {request_id!r} already owns a DSA Sparse cache seat.")
        if not self._free_seats:
            raise RuntimeError("No free DSA Sparse cache seat is available.")

        seat = self._free_seats.popleft()
        self._seat_epoch[seat] += 1
        lease = CacheSeatLease(seat=seat, epoch=self._seat_epoch[seat])
        self._seat_owner[seat] = request_id
        self._request_to_lease[request_id] = lease
        return lease

    def get_lease(self, request_id: Hashable) -> CacheSeatLease:
        try:
            return self._request_to_lease[request_id]
        except KeyError as exc:
            raise KeyError(f"Request {request_id!r} does not own a DSA Sparse cache seat.") from exc

    def release(self, request_id: Hashable) -> CacheSeatLease:
        lease = self.get_lease(request_id)
        del self._request_to_lease[request_id]
        self._seat_owner[lease.seat] = None
        self._free_seats.append(lease.seat)
        return lease

    def pack_rows(
        self,
        request_ids: list[Hashable],
        out: DSASparseRowMapping,
    ) -> DSASparseRowMapping:
        request_capacity = out.row_active.shape[0]
        if len(request_ids) > request_capacity:
            raise ValueError(
                "request_ids exceeds request_capacity, got "
                f"{len(request_ids)} requests for capacity {request_capacity}."
            )
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Each active request may appear in only one DSA Sparse row.")

        out.row_active.fill_(False)
        out.row_to_cache_seat.fill_(INVALID_INDEX)
        out.row_seat_epoch.fill_(INVALID_INDEX)
        for row, request_id in enumerate(request_ids):
            lease = self.get_lease(request_id)
            out.row_active[row] = True
            out.row_to_cache_seat[row] = lease.seat
            out.row_seat_epoch[row] = lease.epoch
        return out


@dataclass(frozen=True)
class DSASparseResidencyState:
    """Device-resident token-to-hot mapping owned by one residency cohort."""

    cohort: "DSASparseCohortKey"
    token_to_hot: torch.Tensor
    hot_to_token: torch.Tensor
    lru_slots: torch.Tensor
    state_seat_epoch: torch.Tensor

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
            (config.max_num_seqs, config.managed_hot_width),
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
        state_seat_epoch = torch.full(
            (config.max_num_seqs,),
            INVALID_INDEX,
            dtype=torch.int32,
            device=device,
        )
        return cls(
            cohort=cohort,
            token_to_hot=token_to_hot,
            hot_to_token=hot_to_token,
            lru_slots=lru_slots,
            state_seat_epoch=state_seat_epoch,
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
class DSASparsePlan:
    """Preallocated lookup and I/O tensors for one execution shape."""

    key: DSASparsePlanKey
    row_mapping: DSASparseRowMapping
    query_positions: torch.Tensor
    query_to_row: torch.Tensor
    query_to_lane: torch.Tensor
    query_valid_mask: torch.Tensor
    valid_topk_counts: torch.Tensor
    topk_positions: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    read_source_global_slots: torch.Tensor
    read_local_hot_slot_ids: torch.Tensor
    read_destination_hot_row_ids: torch.Tensor
    read_valid_mask: torch.Tensor
    resolved_hot_indices: torch.Tensor
    hot_block_table: torch.Tensor
    newest_destination_hot_row_ids: torch.Tensor
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
    ) -> "DSASparsePlan":
        if key.request_capacity > config.max_num_seqs:
            raise ValueError(
                f"request_capacity exceeds max_num_seqs, got {key.request_capacity} and {config.max_num_seqs}."
            )
        if key.query_lane_capacity > config.max_query_tokens_per_request:
            raise ValueError(
                "query_lane_capacity exceeds max_query_tokens_per_request, got "
                f"{key.query_lane_capacity} and "
                f"{config.max_query_tokens_per_request}."
            )

        read_shape = (
            key.request_capacity,
            key.query_lane_capacity,
            config.index_topk,
        )
        write_shape = (
            key.request_capacity,
            key.query_lane_capacity,
        )

        def int_plan(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.full(
                shape,
                INVALID_INDEX,
                dtype=torch.int32,
                device=device,
            )

        return cls(
            key=key,
            row_mapping=DSASparseRowMapping.allocate(
                key.request_capacity,
                device=device,
            ),
            query_positions=int_plan((key.token_capacity,)),
            query_to_row=torch.arange(
                key.token_capacity,
                dtype=torch.int32,
                device=device,
            )
            // key.query_lane_capacity,
            query_to_lane=torch.arange(
                key.token_capacity,
                dtype=torch.int32,
                device=device,
            )
            % key.query_lane_capacity,
            query_valid_mask=torch.zeros(
                key.token_capacity,
                dtype=torch.bool,
                device=device,
            ),
            valid_topk_counts=torch.zeros(
                key.token_capacity,
                dtype=torch.int32,
                device=device,
            ),
            topk_positions=int_plan((key.token_capacity, config.index_topk)),
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
            read_source_global_slots=int_plan(read_shape),
            read_local_hot_slot_ids=int_plan(read_shape),
            read_destination_hot_row_ids=int_plan(read_shape),
            read_valid_mask=torch.zeros(
                read_shape,
                dtype=torch.bool,
                device=device,
            ),
            resolved_hot_indices=int_plan((key.token_capacity, config.index_topk)),
            hot_block_table=torch.full(
                (
                    key.request_capacity,
                    config.hot_blocks_per_seat,
                ),
                INVALID_INDEX,
                dtype=block_table_dtype,
                device=device,
            ),
            newest_destination_hot_row_ids=int_plan((key.token_capacity,)),
            write_global_slots=int_plan(write_shape),
            write_destination_hot_row_ids=int_plan(write_shape),
            write_valid_mask=torch.zeros(
                write_shape,
                dtype=torch.bool,
                device=device,
            ),
        )


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


class DSASparseIndexOperator(Protocol):
    """Mutation contract for the future A5 lookup/state custom operators."""

    def prepare_newest(
        self,
        *,
        state: DSASparseResidencyState,
        plan: DSASparsePlan,
    ) -> None: ...

    def lookup(
        self,
        *,
        state: DSASparseResidencyState,
        plan: DSASparsePlan,
    ) -> None: ...


class UnimplementedDSASparseIndexOperator:
    """Explicit eager stub until the A5 lookup operators are implemented."""

    def prepare_newest(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse newest-state operator is not implemented.")

    def lookup(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse lookup operator is not implemented.")


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
    read_completion: object
    write_completion: object

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
    lookup_complete: bool = False
    write_submitted_layers: set[str] = field(default_factory=set)
    completed_layers: set[str] = field(default_factory=set)


class DSASparseEagerCoordinator:
    """Single eager entry point for the fixed lookup/I/O/SFA sequence."""

    def __init__(
        self,
        config: DSASparseCacheConfig,
        *,
        index_operator: DSASparseIndexOperator,
        io_operator: DSASparseIOOperator,
        seat_manager: CacheSeatManager | None = None,
    ) -> None:
        self.config = config
        self.index_operator = index_operator
        self.io_operator = io_operator
        self.seat_manager = seat_manager or CacheSeatManager(config.max_num_seqs)
        self._cohorts: dict[DSASparseCohortKey, DSASparseCohort] = {}
        self._layers: dict[DSASparseLayerKey, DSASparseLayerBinding] = {}
        self._active_steps: dict[DSASparseCohortKey, DSASparseEagerStep] = {}
        self._hot_plane_addresses: set[int] = set()
        self._region_identities: set[object] = set()
        self._completion_identities: set[object] = set()
        self._frozen = False

    def register_cohort(
        self,
        cohort: DSASparseCohort,
    ) -> None:
        self._require_mutable()
        if cohort.key in self._cohorts:
            raise ValueError(f"DSA Sparse cohort {cohort.key!r} is already registered.")
        self._cohorts[cohort.key] = cohort

    def acquire_request(self, request_id: Hashable) -> CacheSeatLease:
        return self.seat_manager.acquire(request_id)

    def release_request(self, request_id: Hashable) -> CacheSeatLease:
        self.assert_request_idle(request_id)
        return self.seat_manager.release(request_id)

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
        completion_identities = {
            self._resource_identity(binding.read_completion),
            self._resource_identity(binding.write_completion),
        }
        if len(completion_identities) != 2 or completion_identities & self._completion_identities:
            raise ValueError("Each DSA Sparse layer and direction must own an independent completion resource.")
        self._layers[binding.key] = binding
        self._hot_plane_addresses.update(plane_addresses)
        self._region_identities.add(region_identity)
        self._completion_identities.update(completion_identities)

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
        query_positions: torch.Tensor,
        query_valid_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> DSASparseEagerStep:
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

        self.seat_manager.pack_rows(request_ids, plan.row_mapping)
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
        self._copy_exact(block_table, plan.block_table, "block_table")
        plan.topk_positions.fill_(INVALID_INDEX)
        plan.valid_topk_counts.zero_()

        self.index_operator.prepare_newest(
            state=cohort.state,
            plan=plan,
        )
        step = DSASparseEagerStep(
            cohort=cohort,
            plan=plan,
            request_ids=tuple(request_ids),
        )
        self._active_steps[cohort_key] = step
        return step

    def submit_newest_write(
        self,
        step: DSASparseEagerStep,
        layer_name: str,
    ) -> None:
        self._assert_active_step(step)
        binding = self._get_step_layer(step, layer_name)
        if layer_name in step.write_submitted_layers:
            raise RuntimeError(f"Newest write was already submitted for {layer_name!r}.")
        self.io_operator.write_async(
            context=binding.io_context,
            region=binding.io_region,
            destination_global_slots=step.plan.write_global_slots,
            source_hot_row_ids=step.plan.write_destination_hot_row_ids,
            valid_mask=step.plan.write_valid_mask,
            hot_planes=binding.hot_cache.planes,
            completion=binding.write_completion,
        )
        step.write_submitted_layers.add(layer_name)

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
        if leader_layer not in step.write_submitted_layers:
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
        self.index_operator.lookup(
            state=step.cohort.state,
            plan=step.plan,
        )
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
        if layer_name not in step.write_submitted_layers:
            raise RuntimeError("Newest Main KV write must be submitted before lookup/I/O/SFA.")
        if layer_name in step.completed_layers:
            raise RuntimeError(f"DSA Sparse layer {layer_name!r} already completed this step.")

        self.io_operator.read_async(
            context=binding.io_context,
            region=binding.io_region,
            source_global_slots=step.plan.read_source_global_slots,
            destination_hot_row_ids=(step.plan.read_destination_hot_row_ids),
            valid_mask=step.plan.read_valid_mask,
            hot_planes=binding.hot_cache.planes,
            completion=binding.read_completion,
        )
        self.io_operator.wait_read(
            context=binding.io_context,
            completion=binding.read_completion,
            hot_planes=binding.hot_cache.planes,
        )
        resolution = DSASparseResolution(
            hot_main_cache=binding.hot_cache.planes,
            local_sparse_indices=step.plan.resolved_hot_indices,
            hot_block_table=step.plan.hot_block_table,
        )
        output = attention(resolution)
        self.io_operator.wait_write(
            context=binding.io_context,
            completion=binding.write_completion,
            hot_planes=binding.hot_cache.planes,
        )
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
                f"Cannot finish DSA Sparse step before all layer writes are joined, pending layers: {pending_layers}."
            )
        del self._active_steps[step.cohort.key]

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
