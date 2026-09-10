# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections import deque
from collections.abc import Callable, Mapping, Sequence
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

    def __post_init__(self) -> None:
        if not 1 <= self.max_verify_tokens_per_request <= self.block_size:
            raise ValueError("DSA Offload verification must fit within one block.")

    @property
    def resident_blocks(self) -> int:
        return _cdiv(RESIDENT_SLOTS, self.block_size)

    @property
    def replaceable_blocks(self) -> int:
        return _cdiv(REPLACEABLE_SLOTS, self.block_size)

    @property
    def hot_blocks_per_row(self) -> int:
        return self.resident_blocks + self.replaceable_blocks + 3

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

    def tail_slots(self, positions: torch.Tensor) -> torch.Tensor:
        # Even/odd logical blocks occupy the blocks on either side of fallback.
        return (
            self.tail_base
            + torch.div(positions, self.block_size, rounding_mode="floor").remainder(2)
            * (2 * self.block_size)
            + positions.remainder(self.block_size)
        )

    def tail_block(self, row_id: int, logical_block: int) -> int:
        return self.row_block_base(row_id) + self.tail_block_offset + 2 * (logical_block % 2)

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
    block_hash_resolver: Callable[..., bytes] | None = None,
) -> None:
    hot_cache = batch.hot_cache
    row_id = hot_cache.request_to_row[batch.request_ids[request_index]]
    source_block_id = hot_cache.layout.tail_block(row_id, logical_block)
    block_hashes = batch.block_hashes(request_index)
    request_id = batch.request_ids[request_index]
    if block_hash_resolver is None:
        require_block_hashes(
            block_hashes,
            logical_block + 1,
            context=f"Decode tail commit for request {request_id}",
        )
        block_hash = block_hashes[logical_block]
    else:
        block_hash = block_hash_resolver(
            request_index=request_index,
            logical_block=logical_block,
        )
    for cohort in batch.cohorts:
        for layer_name, layer_id in zip(cohort.layer_names, cohort.layer_ids):
            device = hot_cache.layer_caches[layer_name][0].device
            storage_ids = make_storage_ids(
                [block_hash],
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


def tail_commit_request_indices(batch: "DSAOffloadBatch") -> tuple[int, ...]:
    if batch.hot_cache is None:
        return ()
    if not batch.query_end_positions:
        return batch.decode_request_indices
    block_size = batch.layout.block_size
    candidates = []
    for index in batch.decode_request_indices:
        begin, end = batch.query_ranges[index]
        last = batch.query_end_positions[index]
        # Async scheduling can overestimate positions by the previous draft span.
        first = max(0, last - (end - begin) + 1 - batch.query_position_slack)
        if end > begin and first // block_size != (last + 1) // block_size:
            candidates.append(index)
    return tuple(candidates)


def commit_decode_tail(
    batch: "DSAOffloadBatch | None",
    block_hash_resolver: Callable[..., bytes] | None = None,
) -> None:
    if batch is None or batch.hot_cache is None or batch.is_mtp:
        return
    candidates = tail_commit_request_indices(batch)
    if not candidates:
        return
    if batch.query_end_positions and not batch.query_position_slack:
        positions = [batch.query_end_positions[index] for index in candidates]
    else:
        indices = torch.tensor(
            [batch.query_ranges[index][1] - 1 for index in candidates],
            dtype=torch.int64,
            device=batch.query_positions.device,
        )
        positions = batch.query_positions.index_select(0, indices).to("cpu").tolist()
    for request_index, position in zip(candidates, positions):
        if (position + 1) % batch.layout.block_size == 0:
            _put_tail_block(
                batch=batch,
                request_index=request_index,
                logical_block=position // batch.layout.block_size,
                block_hash_resolver=block_hash_resolver,
            )


def commit_mtp_tail(
    batch: "DSAOffloadBatch | None",
    accepted_token_counts: Sequence[int] | torch.Tensor,
    block_hash_resolver: Callable[..., bytes] | None = None,
) -> None:
    if batch is None or batch.hot_cache is None or not batch.is_mtp:
        return
    candidates = tail_commit_request_indices(batch)
    if not candidates:
        return
    device = batch.query_positions.device
    request_indices = torch.tensor(candidates, dtype=torch.int64, device=device)
    query_indices = torch.tensor(
        [batch.query_ranges[index][0] for index in candidates],
        dtype=torch.int64,
        device=device,
    )
    accepted = torch.as_tensor(accepted_token_counts, dtype=torch.int64, device=device)
    boundaries = torch.stack((
        batch.query_positions.index_select(0, query_indices).to(torch.int64),
        accepted.index_select(0, request_indices),
    ), dim=1).to("cpu").tolist()
    for request_index, (first, count) in zip(candidates, boundaries):
        begin, end = batch.query_ranges[request_index]
        count = min(count, end - begin)
        for logical_block in range(
            first // batch.layout.block_size,
            (first + count) // batch.layout.block_size,
        ):
            _put_tail_block(
                batch=batch,
                request_index=request_index,
                logical_block=logical_block,
                block_hash_resolver=block_hash_resolver,
            )
