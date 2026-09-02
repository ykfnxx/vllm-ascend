# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .constants import QUERY_WIDTH, RESIDENT_SLOTS
from .hot_cache import HotCacheLayout, HotCacheState
from .io import IOBackend, make_storage_ids, require_block_keys
from .lookup import (
    IndexCacheCohort,
    clear_lookup_row,
    initialize_resident_mapping,
)
from .ops import LookupState

DSA_OFFLOAD_PD_HANDOFF_KEY = "dsa_offload_handoff"
DSA_OFFLOAD_PD_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class DSAOffloadPDHandoff:
    remote_request_id: str
    stored_token_count: int
    block_size: int
    layer_topk_by_rank: dict[int, dict[str, list[int]]]
    partial_tail_blocks_by_rank: dict[int, dict[str, int]]
    protocol_version: int = DSA_OFFLOAD_PD_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "remote_request_id": self.remote_request_id,
            "stored_token_count": self.stored_token_count,
            "block_size": self.block_size,
            "layer_topk_by_rank": {str(rank): layers for rank, layers in self.layer_topk_by_rank.items()},
            "partial_tail_blocks_by_rank": {
                str(rank): layers for rank, layers in self.partial_tail_blocks_by_rank.items()
            },
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, raw_handoff: Mapping[str, Any]) -> "DSAOffloadPDHandoff":
        if raw_handoff["protocol_version"] != DSA_OFFLOAD_PD_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported DSA Offload protocol version {raw_handoff['protocol_version']}.")
        handoff = cls(
            remote_request_id=raw_handoff["remote_request_id"],
            stored_token_count=raw_handoff["stored_token_count"],
            block_size=raw_handoff["block_size"],
            layer_topk_by_rank={
                int(rank): {layer_name: list(token_positions) for layer_name, token_positions in layers.items()}
                for rank, layers in raw_handoff["layer_topk_by_rank"].items()
            },
            partial_tail_blocks_by_rank={
                int(rank): {layer_name: int(block_id) for layer_name, block_id in layers.items()}
                for rank, layers in raw_handoff["partial_tail_blocks_by_rank"].items()
            },
            protocol_version=raw_handoff["protocol_version"],
        )
        validate_handoff(handoff, len(handoff.layer_topk_by_rank))
        return handoff


@dataclass(frozen=True)
class DSAOffloadLocalHandoff:
    request_id: str
    stored_token_count: int
    block_size: int
    layer_topk: dict[str, list[int]]
    partial_tail_blocks: dict[str, int]


def validate_handoff(
    handoff: DSAOffloadPDHandoff,
    tp_size: int,
    layer_names: Sequence[str] | None = None,
    block_size: int | None = None,
) -> None:
    if block_size is not None and handoff.block_size != block_size:
        raise ValueError("DSA Offload Prefill and Decode block sizes differ.")
    expected_ranks = set(range(tp_size))
    if set(handoff.layer_topk_by_rank) != expected_ranks:
        raise ValueError("DSA Offload handoff has incomplete TP ranks.")

    expected_layers = set(layer_names) if layer_names is not None else set(handoff.layer_topk_by_rank[0])
    for layers in handoff.layer_topk_by_rank.values():
        if set(layers) != expected_layers:
            raise ValueError("DSA Offload handoff has incomplete target layers.")
        if any(
            len(positions) != QUERY_WIDTH or any(not isinstance(position, int) for position in positions)
            for positions in layers.values()
        ):
            raise ValueError(f"DSA Offload handoff TopK must contain {QUERY_WIDTH} integer positions per layer.")

    if handoff.stored_token_count % handoff.block_size:
        if set(handoff.partial_tail_blocks_by_rank) != expected_ranks or any(
            set(layers) != expected_layers for layers in handoff.partial_tail_blocks_by_rank.values()
        ):
            raise ValueError("DSA Offload partial tail has incomplete ranks or layers.")
    elif handoff.partial_tail_blocks_by_rank:
        raise ValueError("Block-aligned DSA Offload handoff must not contain partial tails.")


def handoff_from_transfer_params(
    kv_transfer_params: Mapping[str, Any] | None,
) -> DSAOffloadPDHandoff | None:
    if kv_transfer_params is None:
        return None
    raw_handoff = kv_transfer_params.get(DSA_OFFLOAD_PD_HANDOFF_KEY)
    if raw_handoff is None:
        return None
    return DSAOffloadPDHandoff.from_dict(raw_handoff)


@dataclass
class DSAOffloadWorkerMetadata:
    request_layer_topk_by_rank: dict[str, dict[int, dict[str, list[int]]]]
    request_partial_tail_blocks_by_rank: dict[str, dict[int, dict[str, int]]] = field(default_factory=dict)

    def aggregate(self, other: "DSAOffloadWorkerMetadata") -> "DSAOffloadWorkerMetadata":
        for request_id, ranks in other.request_layer_topk_by_rank.items():
            self.request_layer_topk_by_rank.setdefault(request_id, {}).update(ranks)
        for request_id, ranks in other.request_partial_tail_blocks_by_rank.items():
            self.request_partial_tail_blocks_by_rank.setdefault(request_id, {}).update(ranks)
        return self


@dataclass
class PrefillPublishState:
    request_ids: tuple[str, ...]
    scheduled_token_counts: tuple[int, ...]
    stored_token_counts: tuple[int, ...]
    publish_requests: tuple[bool, ...]
    committed_block_keys: Mapping[str, Sequence[int]]
    io_backend: IOBackend
    tp_rank: int
    layer_topk: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    partial_tail_blocks: dict[str, dict[str, int]] = field(default_factory=dict)

    def publish_layer(
        self,
        *,
        layer_name: str,
        layer_id: int,
        semantic_topk: torch.Tensor,
        main_cache: tuple[torch.Tensor, ...],
        block_table: torch.Tensor,
    ) -> None:
        block_size = main_cache[0].shape[1]
        token_begin = 0
        layer_topk: dict[str, list[int]] = {}
        layer_tail_blocks: dict[str, int] = {}
        for request_index, (
            request_id,
            scheduled_count,
            stored_token_count,
            should_publish,
        ) in enumerate(
            zip(
                self.request_ids,
                self.scheduled_token_counts,
                self.stored_token_counts,
                self.publish_requests,
            )
        ):
            token_end = token_begin + scheduled_count
            if should_publish:
                full_block_count = stored_token_count // block_size
                if full_block_count:
                    request_keys = self.committed_block_keys[request_id]
                    require_block_keys(
                        request_keys,
                        full_block_count,
                        context=f"Prefill publish for request {request_id}",
                    )
                    source_block_ids = block_table[request_index, :full_block_count].to(torch.int64)
                    storage_ids = make_storage_ids(
                        request_keys[:full_block_count],
                        layer_id,
                        device=source_block_ids.device,
                    )
                    self.io_backend.put_blocks(
                        layer_id=layer_id,
                        storage_ids=storage_ids,
                        source_block_ids=source_block_ids,
                    )
                layer_topk[request_id] = (
                    semantic_topk[token_end - 1].detach().reshape(-1).to(device="cpu", dtype=torch.int32).tolist()
                )
                if stored_token_count % block_size:
                    logical_tail = stored_token_count // block_size
                    layer_tail_blocks[request_id] = int(block_table[request_index, logical_tail].item())
            token_begin = token_end
        self.layer_topk[layer_name] = layer_topk
        self.partial_tail_blocks[layer_name] = layer_tail_blocks

    def worker_metadata(self) -> DSAOffloadWorkerMetadata | None:
        request_topk: dict[str, dict[int, dict[str, list[int]]]] = {}
        request_tails: dict[str, dict[int, dict[str, int]]] = {}
        for layer_name, topk_by_request in self.layer_topk.items():
            for request_id, token_positions in topk_by_request.items():
                request_topk.setdefault(request_id, {}).setdefault(self.tp_rank, {})[layer_name] = token_positions
        for layer_name, tails_by_request in self.partial_tail_blocks.items():
            for request_id, block_id in tails_by_request.items():
                request_tails.setdefault(request_id, {}).setdefault(self.tp_rank, {})[layer_name] = block_id
        if not request_topk:
            return None
        return DSAOffloadWorkerMetadata(request_topk, request_tails)

    def local_handoffs(self, block_size: int) -> tuple[DSAOffloadLocalHandoff, ...]:
        handoffs: list[DSAOffloadLocalHandoff] = []
        for request_id, stored_token_count, should_publish in zip(
            self.request_ids,
            self.stored_token_counts,
            self.publish_requests,
        ):
            if not should_publish:
                continue
            handoffs.append(
                DSAOffloadLocalHandoff(
                    request_id=request_id,
                    stored_token_count=stored_token_count,
                    block_size=block_size,
                    layer_topk={
                        layer_name: topk_by_request[request_id]
                        for layer_name, topk_by_request in self.layer_topk.items()
                    },
                    partial_tail_blocks={
                        layer_name: tails_by_request[request_id]
                        for layer_name, tails_by_request in self.partial_tail_blocks.items()
                        if request_id in tails_by_request
                    },
                )
            )
        return tuple(handoffs)


def build_handoff(
    *,
    request_id: str,
    stored_token_count: int,
    block_size: int,
    layer_topk_by_rank: dict[int, dict[str, list[int]]],
    partial_tail_blocks_by_rank: dict[int, dict[str, int]],
    tp_size: int,
) -> DSAOffloadPDHandoff:
    handoff = DSAOffloadPDHandoff(
        remote_request_id=request_id,
        stored_token_count=stored_token_count,
        block_size=block_size,
        layer_topk_by_rank=layer_topk_by_rank,
        partial_tail_blocks_by_rank=partial_tail_blocks_by_rank,
    )
    validate_handoff(handoff, tp_size)
    return handoff


def select_initial_resident(
    topk_token_positions: Sequence[int],
    stored_token_count: int,
    block_size: int,
) -> list[int]:
    resident_history_end = stored_token_count // block_size * block_size
    target_count = min(RESIDENT_SLOTS, resident_history_end)
    selected: list[int] = []
    selected_set: set[int] = set()
    for position in topk_token_positions:
        if 0 <= position < resident_history_end and position not in selected_set:
            selected.append(position)
            selected_set.add(position)
            if len(selected) == target_count:
                return selected
    for position in range(resident_history_end):
        if position not in selected_set:
            selected.append(position)
            if len(selected) == target_count:
                break
    return selected


def _initialize_hot_row(
    *,
    request_id: str,
    stored_token_count: int,
    block_size: int,
    layer_topk: Mapping[str, Sequence[int]],
    hot_cache: HotCacheState,
    cohorts: Sequence[IndexCacheCohort],
    lookup_states: Mapping[str, LookupState],
    layer_ids: Mapping[str, int],
    committed_block_keys: Sequence[int],
    io_backend: IOBackend,
) -> int:
    full_block_count = stored_token_count // block_size
    require_block_keys(
        committed_block_keys,
        full_block_count,
        context=f"Hot Cache admission for request {request_id}",
    )
    row_id = hot_cache.request_to_row[request_id]
    clear_lookup_row(lookup_states, row_id)
    residents: dict[str, list[int]] = {}
    for cohort in cohorts:
        leader_topk = layer_topk[cohort.leader_layer]
        for follower in cohort.layer_names[1:]:
            if layer_topk[follower] != leader_topk:
                raise ValueError("IndexCache cohort layers must carry identical TopK.")
        residents[cohort.cohort_id] = select_initial_resident(
            leader_topk,
            stored_token_count,
            block_size,
        )

    for layer_name, cache_planes in hot_cache.layer_caches.items():
        layer_id = layer_ids[layer_name]
        cohort = next(cohort for cohort in cohorts if layer_name in cohort.layer_names)
        positions = torch.tensor(
            residents[cohort.cohort_id],
            dtype=torch.int64,
            device=cache_planes[0].device,
        )
        if positions.numel():
            storage_ids = make_storage_ids(
                committed_block_keys,
                layer_id,
                device=cache_planes[0].device,
            )
            logical_blocks = torch.div(
                positions,
                block_size,
                rounding_mode="floor",
            )
            destination_slots = hot_cache.layout.global_slot(row_id, 0) + torch.arange(
                positions.numel(),
                dtype=torch.int64,
                device=positions.device,
            )
            io_backend.get_tokens(
                layer_id=layer_id,
                storage_ids=storage_ids[logical_blocks],
                token_offsets=torch.remainder(positions, block_size),
                destination_slots=destination_slots,
            )

    for cohort in cohorts:
        initialize_resident_mapping(
            lookup_states[cohort.cohort_id],
            row_id,
            residents[cohort.cohort_id],
        )
    return row_id


def admit_from_handoff(
    *,
    request_id: str,
    handoff: DSAOffloadPDHandoff,
    tp_rank: int,
    hot_cache: HotCacheState,
    cohorts: Sequence[IndexCacheCohort],
    lookup_states: Mapping[str, LookupState],
    layer_ids: Mapping[str, int],
    committed_block_keys: Sequence[int],
    io_backend: IOBackend,
) -> int:
    row_id = _initialize_hot_row(
        request_id=request_id,
        stored_token_count=handoff.stored_token_count,
        block_size=handoff.block_size,
        layer_topk=handoff.layer_topk_by_rank[tp_rank],
        hot_cache=hot_cache,
        cohorts=cohorts,
        lookup_states=lookup_states,
        layer_ids=layer_ids,
        committed_block_keys=committed_block_keys,
        io_backend=io_backend,
    )
    hot_cache.mark_ready(request_id)
    return row_id


def admit_local_from_prefill(
    *,
    handoff: DSAOffloadLocalHandoff,
    hot_cache: HotCacheState,
    cohorts: Sequence[IndexCacheCohort],
    lookup_states: Mapping[str, LookupState],
    layer_ids: Mapping[str, int],
    committed_block_keys: Sequence[int],
    io_backend: IOBackend,
) -> int:
    expected_layers = set(layer_ids)
    if handoff.block_size != hot_cache.layout.block_size:
        raise ValueError("DSA Offload local Prefill and Decode block sizes differ.")
    if set(handoff.layer_topk) != expected_layers:
        raise ValueError("DSA Offload local handoff has incomplete target layers.")
    if any(
        len(positions) != QUERY_WIDTH or any(not isinstance(position, int) for position in positions)
        for positions in handoff.layer_topk.values()
    ):
        raise ValueError(f"DSA Offload local handoff TopK must contain {QUERY_WIDTH} integer positions per layer.")
    if handoff.stored_token_count % handoff.block_size:
        if set(handoff.partial_tail_blocks) != expected_layers:
            raise ValueError("DSA Offload local partial tail has incomplete target layers.")
    elif handoff.partial_tail_blocks:
        raise ValueError("Block-aligned DSA Offload local handoff must not contain partial tails.")

    row_id = hot_cache.admit(handoff.request_id)
    try:
        _initialize_hot_row(
            request_id=handoff.request_id,
            stored_token_count=handoff.stored_token_count,
            block_size=handoff.block_size,
            layer_topk=handoff.layer_topk,
            hot_cache=hot_cache,
            cohorts=cohorts,
            lookup_states=lookup_states,
            layer_ids=layer_ids,
            committed_block_keys=committed_block_keys,
            io_backend=io_backend,
        )
        tail_tokens = handoff.stored_token_count % handoff.block_size
        if tail_tokens:
            destination_block = hot_cache.layout.row_block_base(row_id) + hot_cache.layout.tail_block_offset
            for layer_name, source_block in handoff.partial_tail_blocks.items():
                for plane in hot_cache.layer_caches[layer_name]:
                    plane[destination_block, :tail_tokens].copy_(plane[source_block, :tail_tokens])
        hot_cache.mark_ready(handoff.request_id)
    except Exception:
        clear_lookup_row(lookup_states, row_id)
        hot_cache.release(handoff.request_id)
        raise
    return row_id


def make_aux_regions(
    layer_caches: Mapping[str, tuple[torch.Tensor, ...]],
    layout: HotCacheLayout | None,
) -> dict[str, list[dict[str, int]]]:
    regions: dict[str, list[dict[str, int]]] = {}
    for layer_name, cache_planes in layer_caches.items():
        regions[layer_name] = []
        for plane in cache_planes:
            regions[layer_name].append(
                {
                    "base_addr": plane.data_ptr(),
                    "block_stride": plane.stride(0) * plane.element_size(),
                    "token_bytes": plane.stride(1) * plane.element_size(),
                    "hot_block_base": layout.hot_block_base if layout else 0,
                    "hot_blocks_per_row": layout.hot_blocks_per_row if layout else 0,
                    "tail_block_offset": layout.tail_block_offset if layout else 0,
                }
            )
    return regions


def append_partial_tail_transfer(
    *,
    handoff: DSAOffloadPDHandoff | None,
    tp_rank: int,
    row_id: int | None,
    local_regions: Mapping[str, list[dict[str, int]]],
    remote_regions: Mapping[str, list[dict[str, int]]],
    local_addresses: list[int],
    remote_addresses: list[int],
    lengths: list[int],
) -> None:
    if handoff is None:
        return
    if handoff.stored_token_count % handoff.block_size == 0:
        return
    tail_tokens = handoff.stored_token_count % handoff.block_size
    for layer_name, source_block_id in handoff.partial_tail_blocks_by_rank[tp_rank].items():
        for local_plane, remote_plane in zip(local_regions[layer_name], remote_regions[layer_name]):
            target_block_id = (
                local_plane["hot_block_base"]
                + row_id * local_plane["hot_blocks_per_row"]
                + local_plane["tail_block_offset"]
            )
            local_addresses.append(local_plane["base_addr"] + target_block_id * local_plane["block_stride"])
            remote_addresses.append(remote_plane["base_addr"] + source_block_id * remote_plane["block_stride"])
            lengths.append(tail_tokens * local_plane["token_bytes"])
