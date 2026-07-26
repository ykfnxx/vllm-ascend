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
    nor a host-memory fallback.
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
    """Eager data-plane call boundary shared with the future I/O bridge."""

    def publish_async(
        self,
        *,
        context: object,
        publication: object,
        region: object,
        portable_block_ids: torch.Tensor,
        source_global_slots: torch.Tensor,
        valid_mask: torch.Tensor,
        full_main_planes: tuple[torch.Tensor, ...],
        completion: object,
    ) -> None: ...

    def wait_publish(
        self,
        *,
        context: object,
        completion: object,
        full_main_planes: tuple[torch.Tensor, ...],
    ) -> None: ...

    def read_async(
        self,
        *,
        context: object,
        region: object,
        source_global_slots: torch.Tensor,
        destination_hot_row_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        hot_planes: tuple[torch.Tensor, ...],
        completion: object,
    ) -> None: ...

    def wait_read(
        self,
        *,
        context: object,
        completion: object,
        hot_planes: tuple[torch.Tensor, ...],
    ) -> None: ...

    def write_async(
        self,
        *,
        context: object,
        region: object,
        destination_global_slots: torch.Tensor,
        source_hot_row_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        hot_planes: tuple[torch.Tensor, ...],
        completion: object,
    ) -> None: ...

    def wait_write(
        self,
        *,
        context: object,
        completion: object,
        hot_planes: tuple[torch.Tensor, ...],
    ) -> None: ...


class UnimplementedDSASparseIOOperator:
    """Explicit eager stub until the backend bridge is implemented."""

    def publish_async(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse I/O publish operator is not implemented.")

    def wait_publish(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse I/O publish wait is not implemented.")

    def read_async(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse I/O read operator is not implemented.")

    def wait_read(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse I/O read wait is not implemented.")

    def write_async(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse I/O write operator is not implemented.")

    def wait_write(self, **_: object) -> None:
        raise NotImplementedError("DSA Sparse I/O write wait is not implemented.")


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
