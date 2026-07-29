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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch


@dataclass(frozen=True)
class DSASparseIOCapabilities:
    abi_version: int
    eager_execution: bool
    device_plan: bool
    stable_address: bool
    direct_npu_source_destination: bool
    pd_publication: bool
    portable_block_identity: bool
    decode_block_bind: bool
    indexer_only_transfer: bool
    supported_layouts: frozenset[str]


@dataclass(frozen=True)
class DSASparseStorageLayout:
    layout_name: str
    block_size: int
    rows_per_block: int
    plane_dtypes: tuple[torch.dtype, ...]
    plane_row_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DSASparsePortableBlock:
    request_transfer_id: str
    logical_block_ordinal: int
    content_block_key: str | None = None


@dataclass(frozen=True)
class DSASparseRegionKey:
    deployment_id: str
    instance_id: str
    kv_role: Literal["kv_producer", "kv_consumer"]
    graph_role: Literal["target", "draft"]
    pp_rank: int
    tp_rank: int
    layer_name: str


class DSASparseIOBackend(Protocol):
    """Control-plane backend contract.

    Implementations own storage. The framework owns neither a default backend
    nor a host-memory fallback. Request handles must not be reused while late
    completions can still reference them. ``release_request`` must be
    idempotent for a handle that was already released so stale
    generation-bearing completion cleanup needs no framework-side tombstone.
    """

    def capabilities(self) -> DSASparseIOCapabilities: ...

    def query_capacity(
        self,
        layouts: tuple[DSASparseStorageLayout, ...],
    ) -> int: ...

    def create_context(self, plan_shapes: tuple[tuple[int, ...], ...]) -> object: ...

    def register_region(
        self,
        key: DSASparseRegionKey,
        layout: DSASparseStorageLayout,
    ) -> object: ...

    def begin_publication(
        self,
        request_transfer_id: str,
        portable_blocks: tuple[DSASparsePortableBlock, ...],
    ) -> object: ...

    def bind_publication(
        self,
        publication: object,
        decode_block_table: torch.Tensor,
    ) -> object: ...

    def release_request(self, request_handle: int) -> None: ...

    def freeze(self) -> None: ...

    def close(self) -> None: ...


class DSASparseIOOperator(Protocol):
    """Unified Decode data-plane boundary.

    A production implementation derives history read addresses from
    ``query_index`` and the current Decode block table, loads ``miss_out`` rows
    into ``slot_out``, performs live-tail writes, and establishes the completion
    dependency before returning.
    """

    def dsa_sparse_io(
        self,
        *,
        context: object,
        region: object,
        query_index: torch.Tensor,
        slot_out: torch.Tensor,
        miss_out: torch.Tensor,
        req_pool_entries: torch.Tensor,
        block_table: torch.Tensor,
        write_global_slots: torch.Tensor,
        write_destination_slots: torch.Tensor,
        write_valid_mask: torch.Tensor,
        hot_planes: tuple[torch.Tensor, ...],
        completion: object,
    ) -> None: ...


class UnimplementedDSASparseIOOperator:
    """Explicit eager stub until the backend bridge is implemented."""

    def dsa_sparse_io(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse unified I/O operator is not implemented.")


class MockDSASparseIOOperator:
    """No-op implementation used only by the eager development runtime.

    The mock preserves the final, unconditional one-call-per-layer topology
    and validates the compact tensor contract. It deliberately does not move
    live-tail or history payload.
    """

    def dsa_sparse_io(
        self,
        *,
        context: object,
        region: object,
        query_index: torch.Tensor,
        slot_out: torch.Tensor,
        miss_out: torch.Tensor,
        req_pool_entries: torch.Tensor,
        block_table: torch.Tensor,
        write_global_slots: torch.Tensor,
        write_destination_slots: torch.Tensor,
        write_valid_mask: torch.Tensor,
        hot_planes: tuple[torch.Tensor, ...],
        completion: object,
    ) -> None:
        del context, region, completion
        assert query_index.ndim == 2
        assert slot_out.shape == query_index.shape
        assert miss_out.shape == query_index.shape
        assert req_pool_entries.shape == (query_index.shape[0],)
        assert block_table.shape[0] == query_index.shape[0]
        assert write_global_slots.shape == req_pool_entries.shape
        assert write_destination_slots.shape == req_pool_entries.shape
        assert write_valid_mask.shape == req_pool_entries.shape
        assert query_index.dtype == torch.int32
        assert slot_out.dtype == torch.int32
        assert miss_out.dtype == torch.int32
        assert req_pool_entries.dtype == torch.int32
        assert block_table.dtype == torch.int32
        assert write_global_slots.dtype == torch.int32
        assert write_destination_slots.dtype == torch.int32
        assert write_valid_mask.dtype == torch.bool
        assert query_index.is_contiguous()
        assert slot_out.is_contiguous()
        assert miss_out.is_contiguous()
        assert req_pool_entries.is_contiguous()
        assert block_table.is_contiguous()
        assert write_global_slots.is_contiguous()
        assert write_destination_slots.is_contiguous()
        assert write_valid_mask.is_contiguous()
        tensors = (
            slot_out,
            miss_out,
            req_pool_entries,
            block_table,
            write_global_slots,
            write_destination_slots,
            write_valid_mask,
        )
        assert all(
            tensor.device == query_index.device
            for tensor in tensors
        )
        assert hot_planes
        assert all(
            plane.device == query_index.device
            for plane in hot_planes
        )


DSASparseIOBackendFactory = Callable[
    [dict[str, Any]],
    DSASparseIOBackend,
]


class DSASparseIOBackendRegistry:
    """Initialization-only registry passed explicitly to the coordinator."""

    def __init__(self) -> None:
        self._factories: dict[str, DSASparseIOBackendFactory] = {}
        self._backends: list[DSASparseIOBackend] = []
        self._frozen = False

    def register(
        self,
        name: str,
        factory: DSASparseIOBackendFactory,
    ) -> None:
        if self._frozen:
            raise RuntimeError("DSA Sparse I/O backend registry is frozen.")
        if not name:
            raise ValueError("DSA Sparse I/O backend name must not be empty.")
        if name in self._factories:
            raise ValueError(f"DSA Sparse I/O backend {name!r} is already registered.")
        self._factories[name] = factory

    def create(
        self,
        name: str,
        options: dict[str, Any],
    ) -> DSASparseIOBackend:
        if self._frozen:
            raise RuntimeError("DSA Sparse I/O backend registry is frozen.")
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"DSA Sparse I/O backend {name!r} is not registered.") from exc
        backend = factory(options)
        self._backends.append(backend)
        return backend

    def freeze(self) -> None:
        for backend in self._backends:
            backend.freeze()
        self._frozen = True
