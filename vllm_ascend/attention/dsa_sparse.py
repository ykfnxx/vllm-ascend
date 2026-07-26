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
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

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


@dataclass(frozen=True)
class CacheSeatLease:
    seat: int
    epoch: int


@dataclass
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


@dataclass
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


@dataclass
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


@dataclass
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
        block_table: torch.Tensor,
    ) -> None: ...

    def lookup(
        self,
        *,
        state: DSASparseResidencyState,
        plan: DSASparsePlan,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None: ...


class UnimplementedDSASparseIndexOperator:
    """Explicit eager stub until the A5 lookup operators are implemented."""

    def prepare_newest(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse newest-state operator is not implemented.")

    def lookup(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse lookup operator is not implemented.")
