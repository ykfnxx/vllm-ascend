# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .constants import (
    FREE_HEAD_STRIDE,
    INDEX_CAPACITY,
    LOOKUP_SLOTS,
    REPLACEABLE_SLOTS,
    RESIDENT_SLOTS,
)
from .io import make_storage_ids, require_block_hashes

if TYPE_CHECKING:
    from .lookup import DSAOffloadBatch


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class HotCacheLayout:
    block_size: int
    max_num_seqs: int
    max_verify_tokens_per_request: int
    hot_block_base: int = 0

    @property
    def resident_blocks(self) -> int:
        return _cdiv(RESIDENT_SLOTS, self.block_size)

    @property
    def replaceable_blocks(self) -> int:
        return _cdiv(REPLACEABLE_SLOTS, self.block_size)

    @property
    def transient_blocks(self) -> int:
        return _cdiv(1 + self.max_verify_tokens_per_request, self.block_size)

    @property
    def hot_blocks_per_row(self) -> int:
        return self.resident_blocks + self.replaceable_blocks + 1 + self.transient_blocks

    @property
    def hot_blocks(self) -> int:
        return self.max_num_seqs * self.hot_blocks_per_row

    @property
    def replaceable_base(self) -> int:
        return self.resident_blocks * self.block_size

    @property
    def tail_base(self) -> int:
        return (self.resident_blocks + self.replaceable_blocks) * self.block_size

    @property
    def fallback_slot(self) -> int:
        return self.tail_base + self.block_size

    @property
    def staging_base(self) -> int:
        return self.fallback_slot + 1

    @property
    def row_stride(self) -> int:
        return self.hot_blocks_per_row * self.block_size

    @property
    def tail_block_offset(self) -> int:
        return self.tail_base // self.block_size

    def row_block_base(self, row_id: int) -> int:
        return self.hot_block_base + row_id * self.hot_blocks_per_row

    def global_slot(self, row_id: int, row_offset: int) -> int:
        return self.row_block_base(row_id) * self.block_size + row_offset

    def lookup_offsets(self, slots: torch.Tensor) -> torch.Tensor:
        return torch.where(
            slots < RESIDENT_SLOTS,
            slots,
            slots - RESIDENT_SLOTS + self.replaceable_base,
        )

    def block_table(self, row_ids: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(
            self.hot_blocks_per_row,
            dtype=torch.int32,
            device=row_ids.device,
        )
        return (
            self.hot_block_base + row_ids.to(torch.int32).unsqueeze(1) * self.hot_blocks_per_row + offsets
        ).contiguous()


@dataclass
class HotCacheState:
    layout: HotCacheLayout
    layer_caches: dict[str, tuple[torch.Tensor, ...]]
    request_to_row: dict[str, int] = field(default_factory=dict)
    ready_requests: set[str] = field(default_factory=set)
    free_rows: deque[int] = field(init=False)
    hot_block_table: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.free_rows = deque(range(self.layout.max_num_seqs))
        if not self.layer_caches:
            raise ValueError("DSA Offload Hot Cache requires at least one layer cache.")
        first_cache_planes = next(iter(self.layer_caches.values()))
        if not first_cache_planes:
            raise ValueError("DSA Offload Hot Cache layers require at least one cache plane.")
        row_ids = torch.arange(
            self.layout.max_num_seqs,
            dtype=torch.int32,
            device=first_cache_planes[0].device,
        )
        # This is the fixed virtual address space consumed by Decode SFA. It is
        # owned by the Hot Cache and deliberately independent of max_model_len
        # and vLLM's scheduler-managed block table width.
        self.hot_block_table = self.layout.block_table(row_ids)

    def admit(self, request_id: str) -> int:
        row_id = self.free_rows.popleft()
        self.request_to_row[request_id] = row_id
        self._clear_transient(row_id)
        return row_id

    def release(self, request_id: str) -> None:
        row_id = self.request_to_row.pop(request_id)
        self.ready_requests.discard(request_id)
        self._clear_transient(row_id)
        self.free_rows.append(row_id)

    def mark_ready(self, request_id: str) -> None:
        self.ready_requests.add(request_id)

    def fail_on_preemption(self, preempted_req_ids: set[str] | None) -> None:
        if preempted_req_ids and preempted_req_ids & self.request_to_row.keys():
            raise RuntimeError("DSA Offload does not support Decode preemption.")

    def _clear_transient(self, row_id: int) -> None:
        begin = self.layout.global_slot(row_id, self.layout.tail_base)
        end = self.layout.global_slot(row_id, self.layout.row_stride)
        for cache_planes in self.layer_caches.values():
            for plane in cache_planes:
                plane.flatten(0, 1)[begin:end].zero_()


def fixed_memory_bytes(
    layout: HotCacheLayout,
    target_specs: Sequence[object],
    cohort_count: int,
    prefetch_layer_count: int = 0,
) -> int:
    hot_payload_bytes = sum(layout.hot_blocks * spec.page_size_bytes for spec in target_specs)
    lookup_ints = layout.max_num_seqs * (INDEX_CAPACITY + LOOKUP_SLOTS + REPLACEABLE_SLOTS + FREE_HEAD_STRIDE)
    storage_id_blocks = _cdiv(INDEX_CAPACITY, layout.block_size)
    storage_id_bytes = (
        prefetch_layer_count
        * layout.max_num_seqs
        * storage_id_blocks
        * 8
    )
    return hot_payload_bytes + cohort_count * lookup_ints * 4 + storage_id_bytes


def resize_target_tensors(
    kv_cache_config: object,
    target_specs: Mapping[str, object],
    layout: HotCacheLayout,
    kv_role: str,
) -> None:
    if kv_role == "kv_producer":
        return
    target_layers = set(target_specs)
    total_blocks = layout.hot_blocks
    if kv_role == "kv_both":
        total_blocks += kv_cache_config.num_blocks
    for tensor in kv_cache_config.kv_cache_tensors:
        matching_layers = target_layers.intersection(tensor.shared_by)
        if matching_layers:
            layer_name = next(iter(matching_layers))
            tensor.size = total_blocks * target_specs[layer_name].page_size_bytes


def validate_target_tensors(
    kv_cache_config: object,
    target_specs: Mapping[str, object],
    layout: HotCacheLayout,
    kv_role: str,
) -> None:
    if kv_role == "kv_producer":
        return
    expected_blocks = layout.hot_blocks + (kv_cache_config.num_blocks if kv_role == "kv_both" else 0)
    target_layers = set(target_specs)
    for tensor in kv_cache_config.kv_cache_tensors:
        matching_layers = target_layers.intersection(tensor.shared_by)
        if matching_layers:
            layer_name = next(iter(matching_layers))
            assert tensor.size == expected_blocks * target_specs[layer_name].page_size_bytes


def _put_tail_block(
    *,
    batch: "DSAOffloadBatch",
    request_index: int,
    logical_block: int,
) -> None:
    hot_cache = batch.hot_cache
    row_id = int(batch.request_rows[request_index].item())
    source_block_id = hot_cache.layout.row_block_base(row_id) + hot_cache.layout.tail_block_offset
    for cohort in batch.cohorts:
        for layer_name, layer_id in zip(cohort.layer_names, cohort.layer_ids):
            device = hot_cache.layer_caches[layer_name][0].device
            block_hashes = batch.block_hashes(request_index)
            request_id = batch.request_ids[request_index]
            require_block_hashes(
                block_hashes,
                logical_block + 1,
                context=f"Decode tail commit for request {request_id}",
            )
            storage_ids = make_storage_ids(
                [block_hashes[logical_block]],
                layer_id,
                device=device,
            )
            batch.io_backend.put_blocks(
                layer_id=layer_id,
                storage_ids=storage_ids,
                source_block_ids=torch.tensor(
                    [source_block_id],
                    dtype=torch.int64,
                    device=device,
                ),
            )


def commit_decode_tail(batch: "DSAOffloadBatch | None") -> None:
    if batch is None or batch.hot_cache is None or batch.is_mtp:
        return
    for request_index in batch.decode_request_indices:
        begin, end = batch.query_ranges[request_index]
        position = int(batch.query_positions[end - 1].item())
        if (position + 1) % batch.layout.block_size == 0:
            _put_tail_block(
                batch=batch,
                request_index=request_index,
                logical_block=position // batch.layout.block_size,
            )


def commit_mtp_tail(
    batch: "DSAOffloadBatch | None",
    accepted_token_counts: Sequence[int],
) -> None:
    if batch is None or batch.hot_cache is None or not batch.is_mtp:
        return
    for request_index in batch.decode_request_indices:
        accepted = accepted_token_counts[request_index]
        if accepted == 0:
            continue
        begin, _ = batch.query_ranges[request_index]
        row_id = int(batch.request_rows[request_index].item())
        row_slot_base = batch.layout.global_slot(row_id, 0)
        copied = 0
        while copied < accepted:
            position = int(batch.query_positions[begin + copied].item())
            tail_offset = position % batch.layout.block_size
            copy_count = min(
                accepted - copied,
                batch.layout.block_size - tail_offset,
            )
            for cache_planes in batch.hot_cache.layer_caches.values():
                for plane in cache_planes:
                    slots = plane.flatten(0, 1)
                    slots[
                        row_slot_base + batch.layout.tail_base + tail_offset : row_slot_base
                        + batch.layout.tail_base
                        + tail_offset
                        + copy_count
                    ].copy_(
                        slots[
                            row_slot_base + batch.layout.staging_base + copied : row_slot_base
                            + batch.layout.staging_base
                            + copied
                            + copy_count
                        ]
                    )
            copied += copy_count
            if tail_offset + copy_count == batch.layout.block_size:
                _put_tail_block(
                    batch=batch,
                    request_index=request_index,
                    logical_block=position // batch.layout.block_size,
                )
