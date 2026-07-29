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
from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_HEAD_STRIDE,
    DSA_SPARSE_FREE_SLOT_COUNT,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_LOOKUP_SLOT_COUNT,
    DSA_SPARSE_QUERY_WIDTH,
    DSA_SPARSE_RESIDENT_SLOT_COUNT,
)

INVALID_INDEX = -1


@dataclass(frozen=True)
class DSASparseCacheConfig:
    """Static Decode-side dimensions for the ASU lookup contract."""

    max_num_seqs: int
    max_model_len: int
    block_size: int
    index_topk: int

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive.")
        if not 0 < self.max_model_len <= DSA_SPARSE_INDEX_CAPACITY:
            raise ValueError(
                "max_model_len must be in the ASU index range "
                f"(0, {DSA_SPARSE_INDEX_CAPACITY}]."
            )
        if self.index_topk != DSA_SPARSE_QUERY_WIDTH:
            raise ValueError(
                f"index_topk must be {DSA_SPARSE_QUERY_WIDTH}."
            )
        if (
            self.block_size <= 0
            or DSA_SPARSE_RESIDENT_SLOT_COUNT % self.block_size
            or DSA_SPARSE_FREE_SLOT_COUNT % self.block_size
        ):
            raise ValueError(
                "block_size must divide the 8K resident and 2K free regions."
            )

    @property
    def index_capacity(self) -> int:
        return DSA_SPARSE_INDEX_CAPACITY

    @property
    def resident_slot_count(self) -> int:
        return DSA_SPARSE_RESIDENT_SLOT_COUNT

    @property
    def free_slot_count(self) -> int:
        return DSA_SPARSE_FREE_SLOT_COUNT

    @property
    def lookup_slot_count(self) -> int:
        return DSA_SPARSE_LOOKUP_SLOT_COUNT

    @property
    def live_tail_start(self) -> int:
        return DSA_SPARSE_LOOKUP_SLOT_COUNT

    @property
    def hot_stride(self) -> int:
        return DSA_SPARSE_LOOKUP_SLOT_COUNT + self.block_size

    @property
    def hot_blocks_per_request(self) -> int:
        return self.hot_stride // self.block_size

    @property
    def total_hot_blocks(self) -> int:
        return self.max_num_seqs * self.hot_blocks_per_request

    @property
    def max_blocks_per_request(self) -> int:
        return (
            self.max_model_len + self.block_size - 1
        ) // self.block_size


class RequestIndexManager:
    """Allocate one stable Decode request index for each admitted request."""

    def __init__(self, max_num_seqs: int) -> None:
        self.max_num_seqs = max_num_seqs
        self._free_indices = deque(range(max_num_seqs))
        self._index_owner: list[Hashable | None] = [None] * max_num_seqs
        self._request_to_index: dict[Hashable, int] = {}

    @property
    def num_free_indices(self) -> int:
        return len(self._free_indices)

    @property
    def active_request_ids(self) -> tuple[Hashable, ...]:
        return tuple(
            owner for owner in self._index_owner if owner is not None
        )

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
class DSASparseCohortKey:
    """Ownership boundary for shared IndexCache lookup state."""

    name: str
    role: Literal["target", "draft"]


@dataclass(frozen=True)
class DSASparseLayerKey:
    cohort: DSASparseCohortKey
    layer_name: str


@dataclass(frozen=True)
class DSASparseLookupState:
    """Four persistent tensors consumed by ``asu_hbm_index_lookup``."""

    cohort: DSASparseCohortKey
    index: torch.Tensor
    slot_to_index: torch.Tensor
    free_slots: torch.Tensor
    free_head: torch.Tensor

    @classmethod
    def allocate(
        cls,
        config: DSASparseCacheConfig,
        cohort: DSASparseCohortKey,
        *,
        device: torch.device | str,
    ) -> "DSASparseLookupState":
        index = torch.full(
            (config.max_num_seqs, DSA_SPARSE_INDEX_CAPACITY),
            INVALID_INDEX,
            dtype=torch.int32,
            device=device,
        )
        slot_to_index = torch.full(
            (config.max_num_seqs, DSA_SPARSE_LOOKUP_SLOT_COUNT),
            INVALID_INDEX,
            dtype=torch.int32,
            device=device,
        )
        free_slots = (
            torch.arange(
                DSA_SPARSE_RESIDENT_SLOT_COUNT,
                DSA_SPARSE_LOOKUP_SLOT_COUNT,
                dtype=torch.int32,
                device=device,
            )
            .expand(config.max_num_seqs, -1)
            .clone()
        )
        free_head = torch.zeros(
            (config.max_num_seqs, DSA_SPARSE_FREE_HEAD_STRIDE),
            dtype=torch.int32,
            device=device,
        )
        return cls(
            cohort=cohort,
            index=index,
            slot_to_index=slot_to_index,
            free_slots=free_slots,
            free_head=free_head,
        )

    def reset_request(self, request_index: int) -> None:
        self.index[request_index].fill_(INVALID_INDEX)
        self.slot_to_index[request_index].fill_(INVALID_INDEX)
        self.free_slots[request_index].copy_(
            torch.arange(
                DSA_SPARSE_RESIDENT_SLOT_COUNT,
                DSA_SPARSE_LOOKUP_SLOT_COUNT,
                dtype=torch.int32,
                device=self.free_slots.device,
            )
        )
        self.free_head[request_index].zero_()

    def initialize_resident(self, request_index: int) -> None:
        resident_tokens = torch.arange(
            DSA_SPARSE_RESIDENT_SLOT_COUNT,
            dtype=torch.int32,
            device=self.index.device,
        )
        self.index[
            request_index,
            :DSA_SPARSE_RESIDENT_SLOT_COUNT,
        ].copy_(resident_tokens)
        self.slot_to_index[
            request_index,
            :DSA_SPARSE_RESIDENT_SLOT_COUNT,
        ].copy_(resident_tokens)


@dataclass(frozen=True)
class DSASparseLookupBatch:
    req_pool_entries: torch.Tensor
    query_index: torch.Tensor
    lookup_mask: torch.Tensor


@dataclass(frozen=True)
class DSASparseLookupOutput:
    slot_out: torch.Tensor
    miss_out: torch.Tensor


@dataclass(frozen=True)
class DSASparseStepMetadata:
    """Compact active-request metadata shared by all target cohorts."""

    request_ids: tuple[Hashable, ...]
    req_pool_entries: torch.Tensor
    query_positions: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    dense_tail_starts: torch.Tensor
    resident_tail_starts: torch.Tensor
    write_global_slots: torch.Tensor
    write_destination_slots: torch.Tensor
    write_valid_mask: torch.Tensor
    hot_block_table: torch.Tensor


@dataclass(frozen=True)
class DSASparseLayerLayout:
    """Per-token Main MLA plane layout for one local sparse layer."""

    layer_name: str
    plane_dtypes: tuple[torch.dtype, ...]
    plane_row_shapes: tuple[tuple[int, ...], ...]


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


class DSASparseLookupOperator(Protocol):
    """Compact ASU-style lookup protocol."""

    def lookup(
        self,
        *,
        state: DSASparseLookupState,
        batch: DSASparseLookupBatch,
    ) -> DSASparseLookupOutput: ...


class UnimplementedDSASparseLookupOperator:
    """Explicit boundary until the new fused SIMT operator is connected."""

    def lookup(
        self,
        *,
        state: DSASparseLookupState,
        batch: DSASparseLookupBatch,
    ) -> DSASparseLookupOutput:
        del state, batch
        raise NotImplementedError(
            "DSA Sparse ASU-style lookup operator is not implemented."
        )


@dataclass(frozen=True)
class DSASparseLayerBinding:
    """Per-layer Hot Main Cache and I/O resources."""

    layer_name: str
    cohort: DSASparseCohortKey
    hot_cache: DSASparseLayerHotCache
    io_context: object
    io_region: object
    io_completion: object

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
    state: DSASparseLookupState


@dataclass
class DSASparseResolution:
    hot_main_cache: tuple[torch.Tensor, ...]
    attention_indices: torch.Tensor
    hot_block_table: torch.Tensor


@dataclass
class DSASparseEagerStep:
    cohort: DSASparseCohort
    metadata: DSASparseStepMetadata
    lookup_batch: DSASparseLookupBatch | None = None
    lookup_output: DSASparseLookupOutput | None = None
    attention_indices: torch.Tensor | None = None
    newest_written_layers: set[str] = field(default_factory=set)
    io_completed_layers: set[str] = field(default_factory=set)
    completed_layers: set[str] = field(default_factory=set)

    @property
    def lookup_complete(self) -> bool:
        return self.lookup_output is not None


class DSASparseEagerCoordinator:
    """Cohort lookup followed by per-layer I/O and SFA."""

    def __init__(
        self,
        config: DSASparseCacheConfig,
        *,
        lookup_operator: DSASparseLookupOperator,
        io_operator: DSASparseIOOperator,
        request_index_manager: RequestIndexManager | None = None,
    ) -> None:
        self.config = config
        self.lookup_operator = lookup_operator
        self.io_operator = io_operator
        self.request_index_manager = (
            request_index_manager
            or RequestIndexManager(config.max_num_seqs)
        )
        self._cohorts: dict[DSASparseCohortKey, DSASparseCohort] = {}
        self._layers: dict[DSASparseLayerKey, DSASparseLayerBinding] = {}
        self._active_steps: dict[
            DSASparseCohortKey, DSASparseEagerStep
        ] = {}
        self._frozen = False
        self._failure: Exception | None = None

    def register_cohort(self, cohort: DSASparseCohort) -> None:
        if self._frozen:
            raise RuntimeError("DSA Sparse coordinator resources are frozen.")
        if cohort.key in self._cohorts:
            raise ValueError(
                f"DSA Sparse cohort {cohort.key!r} is already registered."
            )
        self._cohorts[cohort.key] = cohort

    def register_layer(self, binding: DSASparseLayerBinding) -> None:
        if self._frozen:
            raise RuntimeError("DSA Sparse coordinator resources are frozen.")
        if binding.key in self._layers:
            raise ValueError(
                f"DSA Sparse layer {binding.key!r} is already registered."
            )
        self._layers[binding.key] = binding
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
        for cohort in self._cohorts.values():
            self.get_layer_binding(
                cohort.key,
                cohort.leader_layer,
            )
        self._frozen = True

    def acquire_request(self, request_id: Hashable) -> int:
        self._require_healthy()
        request_index = self.request_index_manager.acquire(request_id)
        for cohort in self._cohorts.values():
            cohort.state.reset_request(request_index)
            cohort.state.initialize_resident(request_index)
        return request_index

    def request_index(self, request_id: Hashable) -> int:
        self._require_healthy()
        return self.request_index_manager.get_index(request_id)

    def release_request(self, request_id: Hashable) -> int:
        self._require_healthy()
        self.assert_request_idle(request_id)
        request_index = self.request_index_manager.get_index(request_id)
        for cohort in self._cohorts.values():
            cohort.state.reset_request(request_index)
        return self.request_index_manager.release(request_id)

    def assert_request_idle(self, request_id: Hashable) -> None:
        if any(
            request_id in step.metadata.request_ids
            for step in self._active_steps.values()
        ):
            raise RuntimeError(
                "Cannot release a DSA Sparse request while its step is active."
            )

    def get_cohort(
        self,
        cohort_key: DSASparseCohortKey,
    ) -> DSASparseCohort:
        return self._cohorts[cohort_key]

    def get_layer_binding(
        self,
        cohort_key: DSASparseCohortKey,
        layer_name: str,
    ) -> DSASparseLayerBinding:
        return self._layers[
            DSASparseLayerKey(
                cohort=cohort_key,
                layer_name=layer_name,
            )
        ]

    def build_step_metadata(
        self,
        *,
        request_ids: list[Hashable],
        query_positions: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> DSASparseStepMetadata:
        req_pool_entries = torch.tensor(
            [
                self.request_index(request_id)
                for request_id in request_ids
            ],
            dtype=torch.int32,
            device=query_positions.device,
        )
        query_positions = query_positions.to(
            dtype=torch.int32
        ).contiguous()
        seq_lens = seq_lens.to(
            device=query_positions.device,
            dtype=torch.int32,
        ).contiguous()
        block_table = block_table.to(
            device=query_positions.device,
        ).contiguous()
        dense_tail_starts = (
            torch.div(
                query_positions,
                self.config.block_size,
                rounding_mode="floor",
            )
            * self.config.block_size
        ).to(torch.int32)
        resident_tail_starts = torch.full_like(
            query_positions,
            self.config.live_tail_start,
        )

        batch_rows = torch.arange(
            len(request_ids),
            dtype=torch.long,
            device=query_positions.device,
        )
        logical_blocks = torch.div(
            query_positions,
            self.config.block_size,
            rounding_mode="floor",
        ).to(torch.long)
        physical_blocks = block_table[
            batch_rows,
            logical_blocks,
        ].to(torch.int32)
        tail_offsets = torch.remainder(
            query_positions,
            self.config.block_size,
        ).to(torch.int32)
        write_global_slots = (
            physical_blocks * self.config.block_size + tail_offsets
        )
        write_destination_slots = (
            req_pool_entries * self.config.hot_stride
            + self.config.live_tail_start
            + tail_offsets
        )
        write_valid_mask = (
            (query_positions >= 0)
            & (query_positions < seq_lens)
            & (physical_blocks >= 0)
        )
        write_global_slots = torch.where(
            write_valid_mask,
            write_global_slots,
            torch.full_like(write_global_slots, INVALID_INDEX),
        )
        write_destination_slots = torch.where(
            write_valid_mask,
            write_destination_slots,
            torch.full_like(write_destination_slots, INVALID_INDEX),
        )

        hot_block_offsets = torch.arange(
            self.config.hot_blocks_per_request,
            dtype=block_table.dtype,
            device=block_table.device,
        )
        hot_block_table = (
            req_pool_entries.to(block_table.dtype).unsqueeze(1)
            * self.config.hot_blocks_per_request
            + hot_block_offsets
        ).contiguous()
        return DSASparseStepMetadata(
            request_ids=tuple(request_ids),
            req_pool_entries=req_pool_entries,
            query_positions=query_positions,
            seq_lens=seq_lens,
            block_table=block_table,
            dense_tail_starts=dense_tail_starts,
            resident_tail_starts=resident_tail_starts,
            write_global_slots=write_global_slots,
            write_destination_slots=write_destination_slots,
            write_valid_mask=write_valid_mask,
            hot_block_table=hot_block_table,
        )

    def begin_step(
        self,
        cohort_key: DSASparseCohortKey,
        metadata: DSASparseStepMetadata,
    ) -> DSASparseEagerStep:
        self._require_healthy()
        if not self._frozen:
            raise RuntimeError(
                "DSA Sparse coordinator must be frozen before execution."
            )
        if cohort_key in self._active_steps:
            raise RuntimeError(
                "Only one DSA Sparse step may own a cohort at a time."
            )
        step = DSASparseEagerStep(
            cohort=self.get_cohort(cohort_key),
            metadata=metadata,
        )
        self._active_steps[cohort_key] = step
        return step

    def submit_newest_write(
        self,
        step: DSASparseEagerStep,
        layer_name: str,
    ) -> None:
        self._assert_active_step(step)
        self.get_layer_binding(step.cohort.key, layer_name)
        if layer_name in step.newest_written_layers:
            raise RuntimeError(
                f"Newest payload was already submitted for {layer_name!r}."
            )
        step.newest_written_layers.add(layer_name)

    def prepare_lookup(
        self,
        step: DSASparseEagerStep,
        *,
        query_index: torch.Tensor,
    ) -> None:
        self._assert_active_step(step)
        if step.lookup_complete:
            raise RuntimeError(
                "Each DSA Sparse cohort performs lookup once per step."
            )
        leader = step.cohort.leader_layer
        if leader not in step.newest_written_layers:
            raise RuntimeError(
                "The cohort leader must submit the live-tail write before lookup."
            )

        batch_size = step.metadata.req_pool_entries.shape[0]
        assert query_index.shape == (
            batch_size,
            DSA_SPARSE_QUERY_WIDTH,
        )
        assert query_index.dtype == torch.int32
        assert query_index.is_contiguous()
        valid_mask = query_index >= 0
        tail_mask = (
            valid_mask
            & (
                query_index
                >= step.metadata.dense_tail_starts.unsqueeze(1)
            )
        )
        lookup_mask = (
            valid_mask & ~tail_mask
        ).to(torch.int32).contiguous()
        batch = DSASparseLookupBatch(
            req_pool_entries=step.metadata.req_pool_entries,
            query_index=query_index,
            lookup_mask=lookup_mask,
        )
        try:
            output = self.lookup_operator.lookup(
                state=step.cohort.state,
                batch=batch,
            )
        except Exception as exc:
            self._poison(exc)
            raise
        assert output.slot_out.shape == query_index.shape
        assert output.miss_out.shape == query_index.shape
        assert output.slot_out.dtype == torch.int32
        assert output.miss_out.dtype == torch.int32
        assert output.slot_out.device == query_index.device
        assert output.miss_out.device == query_index.device
        assert output.slot_out.is_contiguous()
        assert output.miss_out.is_contiguous()

        tail_slots = (
            step.metadata.resident_tail_starts.unsqueeze(1)
            + query_index
            - step.metadata.dense_tail_starts.unsqueeze(1)
        )
        mapped = torch.where(
            tail_mask,
            tail_slots,
            output.slot_out,
        )
        attention_indices = torch.where(
            valid_mask,
            mapped,
            torch.full_like(mapped, INVALID_INDEX),
        )
        step.lookup_batch = batch
        step.lookup_output = output
        step.attention_indices = attention_indices

    def run_layer_attention(
        self,
        step: DSASparseEagerStep,
        layer_name: str,
        attention: Callable[["DSASparseResolution"], torch.Tensor],
    ) -> torch.Tensor:
        self._assert_active_step(step)
        binding = self.get_layer_binding(step.cohort.key, layer_name)
        if (
            step.lookup_batch is None
            or step.lookup_output is None
            or step.attention_indices is None
        ):
            raise RuntimeError(
                "DSA Sparse lookup must complete before layer I/O."
            )
        if layer_name not in step.newest_written_layers:
            raise RuntimeError(
                "The layer must submit its live-tail write before I/O."
            )
        if layer_name in step.completed_layers:
            raise RuntimeError(
                f"DSA Sparse layer {layer_name!r} already completed."
            )

        try:
            self.io_operator.dsa_sparse_io(
                context=binding.io_context,
                region=binding.io_region,
                query_index=step.lookup_batch.query_index,
                slot_out=step.lookup_output.slot_out,
                miss_out=step.lookup_output.miss_out,
                req_pool_entries=(
                    step.lookup_batch.req_pool_entries
                ),
                block_table=step.metadata.block_table,
                write_global_slots=(
                    step.metadata.write_global_slots
                ),
                write_destination_slots=(
                    step.metadata.write_destination_slots
                ),
                write_valid_mask=step.metadata.write_valid_mask,
                hot_planes=binding.hot_cache.planes,
                completion=binding.io_completion,
            )
        except Exception as exc:
            self._poison(exc)
            raise
        step.io_completed_layers.add(layer_name)
        output = attention(
            DSASparseResolution(
                hot_main_cache=binding.hot_cache.planes,
                attention_indices=step.attention_indices,
                hot_block_table=step.metadata.hot_block_table,
            )
        )
        step.completed_layers.add(layer_name)
        return output

    def finish_step(self, step: DSASparseEagerStep) -> None:
        self._assert_active_step(step)
        expected_layers = {
            key.layer_name
            for key, binding in self._layers.items()
            if binding.cohort == step.cohort.key
        }
        if step.completed_layers != expected_layers:
            raise RuntimeError(
                "Cannot finish DSA Sparse step before every layer completes."
            )
        del self._active_steps[step.cohort.key]

    def abort_step(self, step: DSASparseEagerStep) -> None:
        self._assert_active_step(step)
        del self._active_steps[step.cohort.key]

    def _assert_active_step(self, step: DSASparseEagerStep) -> None:
        if self._active_steps.get(step.cohort.key) is not step:
            raise RuntimeError(
                "DSA Sparse step is not the active cohort owner."
            )

    def _require_healthy(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "DSA Sparse coordinator is poisoned after an operator failure."
            ) from self._failure

    def _poison(self, failure: Exception) -> None:
        if self._failure is None:
            self._failure = failure


@dataclass(frozen=True)
class DSASparseMainWriteTarget:
    hot_main_cache: tuple[torch.Tensor, ...]
    slot_mapping: torch.Tensor


class DSASparseEagerAttentionContext(Protocol):
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
    """One compact active batch for one IndexCache cohort."""

    def __init__(
        self,
        *,
        coordinator: DSASparseEagerCoordinator,
        step: DSASparseEagerStep,
    ) -> None:
        self.coordinator = coordinator
        self.step = step
        self._closed = False

    @classmethod
    def begin(
        cls,
        coordinator: DSASparseEagerCoordinator,
        cohort_key: DSASparseCohortKey,
        *,
        metadata: DSASparseStepMetadata,
    ) -> "DSASparseEagerBatchContext":
        return cls(
            coordinator=coordinator,
            step=coordinator.begin_step(
                cohort_key,
                metadata,
            ),
        )

    @property
    def num_sfa_queries(self) -> int:
        return len(self.step.metadata.request_ids)

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
            slot_mapping=self.step.metadata.write_destination_slots,
        )

    def submit_newest_write(self, layer_name: str) -> None:
        self._require_open()
        self.coordinator.submit_newest_write(
            self.step,
            layer_name,
        )

    def run_layer_attention(
        self,
        layer_name: str,
        semantic_topk_positions: torch.Tensor,
        attention: Callable[[DSASparseResolution], torch.Tensor],
    ) -> torch.Tensor:
        self._require_open()
        if (
            semantic_topk_positions.ndim == 3
            and semantic_topk_positions.shape[1] == 1
        ):
            semantic_topk_positions = (
                semantic_topk_positions.squeeze(1)
            )
        semantic_topk_positions = semantic_topk_positions[
            : self.num_sfa_queries
        ]
        if not self.step.lookup_complete:
            if layer_name != self.step.cohort.leader_layer:
                raise RuntimeError(
                    "The cohort leader must resolve Top-K before followers."
                )
            self.coordinator.prepare_lookup(
                self.step,
                query_index=semantic_topk_positions,
            )
        return self.coordinator.run_layer_attention(
            self.step,
            layer_name,
            attention,
        )

    def finish(self) -> None:
        self._require_open()
        self.coordinator.finish_step(self.step)
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self.coordinator.abort_step(self.step)
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "DSA Sparse eager batch context is closed."
            )


class DSASparseEagerContextRouter:
    """Route each sparse layer to its IndexCache cohort context."""

    def __init__(
        self,
        layer_contexts: Mapping[str, DSASparseEagerBatchContext],
    ) -> None:
        self._layer_contexts = MappingProxyType(
            dict(layer_contexts)
        )
        self._contexts = tuple(
            {
                id(context): context
                for context in layer_contexts.values()
            }.values()
        )
        self._num_sfa_queries = self._contexts[0].num_sfa_queries
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
        return self._layer_contexts[layer_name]

    def main_write_target(
        self,
        layer_name: str,
    ) -> DSASparseMainWriteTarget:
        return self.context_for(layer_name).main_write_target(
            layer_name
        )

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
            raise RuntimeError(
                "DSA Sparse eager context router is closed."
            )
        try:
            for context in self._contexts:
                context.finish()
        except BaseException:
            for context in self._contexts:
                context.abort()
            raise
        finally:
            self._closed = True

    def abort(self) -> None:
        if not self._closed:
            for context in self._contexts:
                context.abort()
            self._closed = True
