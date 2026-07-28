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

    A production implementation derives history read addresses from the
    semantic Top-K positions and the current Decode block table, performs the
    newest writes and miss reads, and establishes the completion dependency
    before returning.  Eager and graph execution intentionally share this
    single call shape.
    """

    def dsa_sparse_io(
        self,
        *,
        context: object,
        region: object,
        topk_positions: torch.Tensor,
        resolved_hot_indices: torch.Tensor,
        miss_mask: torch.Tensor,
        query_to_req_idx: torch.Tensor,
        block_table: torch.Tensor,
        write_global_slots: torch.Tensor,
        write_destination_hot_row_ids: torch.Tensor,
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
    and validates the static tensor contract.  It deliberately does not move
    newest or history payload.  Consequently, a miss installed by
    ``dsa_sparse_lookup_update`` does not contain valid payload and this mock
    must not be used to claim model accuracy or multi-step miss correctness.
    """

    def dsa_sparse_io(
        self,
        *,
        context: object,
        region: object,
        topk_positions: torch.Tensor,
        resolved_hot_indices: torch.Tensor,
        miss_mask: torch.Tensor,
        query_to_req_idx: torch.Tensor,
        block_table: torch.Tensor,
        write_global_slots: torch.Tensor,
        write_destination_hot_row_ids: torch.Tensor,
        write_valid_mask: torch.Tensor,
        hot_planes: tuple[torch.Tensor, ...],
        completion: object,
    ) -> None:
        del context, region, completion
        if topk_positions.ndim != 2:
            raise ValueError("topk_positions must be two-dimensional.")
        if topk_positions.shape != resolved_hot_indices.shape:
            raise ValueError(
                "topk_positions and resolved_hot_indices must have the same "
                "shape."
            )
        if miss_mask.shape != topk_positions.shape:
            raise ValueError("miss_mask must have the Top-K tensor shape.")
        if query_to_req_idx.shape != (topk_positions.shape[0],):
            raise ValueError(
                "query_to_req_idx must contain one request index for each "
                "query."
            )
        if block_table.ndim != 2:
            raise ValueError("block_table must be two-dimensional.")
        if write_global_slots.shape != write_destination_hot_row_ids.shape:
            raise ValueError(
                "Newest write source and destination descriptors must have "
                "the same shape."
            )
        if write_valid_mask.shape != write_global_slots.shape:
            raise ValueError("write_valid_mask must have the newest descriptor shape.")
        if write_global_slots.ndim != 2:
            raise ValueError("Newest write descriptors must be two-dimensional.")
        if write_global_slots.shape[0] != block_table.shape[0]:
            raise ValueError(
                "Newest write descriptors and block_table must have the same "
                "request-index capacity."
            )
        if not hot_planes:
            raise ValueError("At least one Hot Cache plane is required.")
        if any(plane.ndim < 2 for plane in hot_planes):
            raise ValueError("Every Hot Cache plane must use a paged layout.")
        hot_page_shape = hot_planes[0].shape[:2]
        if any(
            plane.shape[:2] != hot_page_shape
            for plane in hot_planes[1:]
        ):
            raise ValueError(
                "Every Hot Cache plane must have the same block and row "
                "dimensions."
            )
        if topk_positions.dtype != torch.int32:
            raise TypeError("topk_positions must use int32.")
        if resolved_hot_indices.dtype != torch.int32:
            raise TypeError("resolved_hot_indices must use int32.")
        if miss_mask.dtype != torch.bool:
            raise TypeError("miss_mask must use bool.")
        if query_to_req_idx.dtype != torch.int32:
            raise TypeError("query_to_req_idx must use int32.")
        if block_table.dtype != torch.int32:
            raise TypeError("block_table must use int32.")
        if write_global_slots.dtype != torch.int32:
            raise TypeError("write_global_slots must use int32.")
        if write_destination_hot_row_ids.dtype != torch.int32:
            raise TypeError("write_destination_hot_row_ids must use int32.")
        if write_valid_mask.dtype != torch.bool:
            raise TypeError("write_valid_mask must use bool.")


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
