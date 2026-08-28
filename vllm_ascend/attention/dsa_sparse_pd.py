# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend import dsa_sparse_probe
from vllm_ascend.dsa_sparse_backend import (
    DSASparseKVBackend,
    DSASparseStorageKeyEncoder,
)
from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_QUERY_WIDTH,
    DSA_SPARSE_RESIDENT_SLOT_COUNT,
)

DSA_SPARSE_PD_HANDOFF_KEY = "dsa_sparse_pd_handoff"
DSA_SPARSE_PD_PROTOCOL_VERSION = 2


def build_dsa_sparse_resident_token_ids(
    *,
    topk_token_ids: Iterable[int],
    stored_token_count: int,
    block_size: int,
    resident_token_count: int = DSA_SPARSE_RESIDENT_SLOT_COUNT,
) -> list[int]:
    """Build a TopK-first resident set while preserving score order."""

    stored_token_count = int(stored_token_count)
    block_size = int(block_size)
    resident_token_count = int(resident_token_count)
    if stored_token_count <= 0:
        raise ValueError("DSA Sparse resident initialization requires stored tokens.")
    if block_size <= 0:
        raise ValueError("DSA Sparse resident initialization requires block_size > 0.")
    if not 0 < resident_token_count <= DSA_SPARSE_RESIDENT_SLOT_COUNT:
        raise ValueError("DSA Sparse resident_token_count must fit the 8K region.")

    # The final partial block is always owned by the persistent tail and must
    # not also occupy a lookup-managed resident slot.
    resident_history_end = (stored_token_count // block_size) * block_size
    target_count = min(resident_token_count, resident_history_end)
    selected: list[int] = []
    selected_set: set[int] = set()
    for raw_token_id in topk_token_ids:
        token_id = int(raw_token_id)
        if token_id < 0 or token_id >= resident_history_end or token_id in selected_set:
            continue
        selected.append(token_id)
        selected_set.add(token_id)
        if len(selected) == target_count:
            return selected

    for token_id in range(resident_history_end):
        if token_id in selected_set:
            continue
        selected.append(token_id)
        if len(selected) == target_count:
            break
    return selected


@dataclass(frozen=True)
class DSASparsePDHandoff:
    """Serializable final-Prefill TopK metadata sent from P to D."""

    remote_request_id: str
    stored_token_count: int
    block_size: int
    layer_topk_by_rank: dict[int, dict[str, list[int]]]
    partial_tail_blocks_by_rank: dict[int, dict[str, int]]
    protocol_version: int = DSA_SPARSE_PD_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != DSA_SPARSE_PD_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported DSA Sparse P/D handoff protocol version {self.protocol_version}.")
        if not self.remote_request_id:
            raise ValueError("DSA Sparse P/D remote_request_id must not be empty.")
        if self.stored_token_count <= 0:
            raise ValueError("DSA Sparse P/D stored_token_count must be positive.")
        if self.block_size <= 0:
            raise ValueError("DSA Sparse P/D block_size must be positive.")
        if not self.layer_topk_by_rank:
            raise ValueError("DSA Sparse P/D handoff requires per-rank layer TopK.")
        for rank, layer_topk in self.layer_topk_by_rank.items():
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ValueError("DSA Sparse P/D ranks must be non-negative integers.")
            if not layer_topk:
                raise ValueError(f"DSA Sparse P/D rank {rank} has no layer TopK.")
            for layer_name, token_ids in layer_topk.items():
                if not layer_name:
                    raise ValueError("DSA Sparse P/D layer names must not be empty.")
                if len(token_ids) != DSA_SPARSE_QUERY_WIDTH:
                    raise ValueError(
                        "DSA Sparse P/D layer TopK width must be "
                        f"{DSA_SPARSE_QUERY_WIDTH}, got {len(token_ids)} "
                        f"for {layer_name!r}."
                    )
                if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
                    raise TypeError("DSA Sparse P/D TopK token IDs must be integers.")
        expected_tail_count = self.stored_token_count % self.block_size
        if expected_tail_count:
            if set(self.partial_tail_blocks_by_rank) != set(self.layer_topk_by_rank):
                raise ValueError("DSA Sparse partial-tail ranks must match TopK ranks.")
            for rank, layer_blocks in self.partial_tail_blocks_by_rank.items():
                if set(layer_blocks) != set(self.layer_topk_by_rank[rank]):
                    raise ValueError("DSA Sparse partial-tail layers must match TopK layers.")
                if any(block_id < 0 for block_id in layer_blocks.values()):
                    raise ValueError("DSA Sparse partial-tail block IDs must be non-negative.")
        elif self.partial_tail_blocks_by_rank:
            raise ValueError("DSA Sparse aligned handoff must not carry a partial tail.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "remote_request_id": self.remote_request_id,
            "stored_token_count": self.stored_token_count,
            "block_size": self.block_size,
            "layer_topk_by_rank": {
                str(rank): {layer_name: list(token_ids) for layer_name, token_ids in layer_topk.items()}
                for rank, layer_topk in self.layer_topk_by_rank.items()
            },
            "partial_tail_blocks_by_rank": {
                str(rank): dict(layer_blocks) for rank, layer_blocks in (self.partial_tail_blocks_by_rank.items())
            },
        }

    @classmethod
    def from_dict(cls, raw_handoff: object) -> DSASparsePDHandoff:
        if not isinstance(raw_handoff, dict):
            raise TypeError("DSA Sparse P/D handoff must be a dictionary.")
        raw_topk = raw_handoff.get("layer_topk_by_rank")
        if not isinstance(raw_topk, dict):
            raise TypeError("DSA Sparse P/D layer_topk_by_rank must be a dictionary.")
        layer_topk_by_rank: dict[int, dict[str, list[int]]] = {}
        for raw_rank, raw_layers in raw_topk.items():
            if not isinstance(raw_layers, dict):
                raise TypeError("DSA Sparse P/D rank TopK must be a dictionary.")
            if isinstance(raw_rank, bool):
                raise TypeError("DSA Sparse P/D rank keys must be integers.")
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError) as error:
                raise TypeError("DSA Sparse P/D rank keys must be integers.") from error
            if str(rank) != str(raw_rank):
                raise TypeError("DSA Sparse P/D rank keys must use canonical integers.")
            layers: dict[str, list[int]] = {}
            for layer_name, token_ids in raw_layers.items():
                if not isinstance(layer_name, str):
                    raise TypeError("DSA Sparse P/D layer names must be strings.")
                if not isinstance(token_ids, (list, tuple)):
                    raise TypeError("DSA Sparse P/D layer TopK must be a sequence.")
                if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
                    raise TypeError("DSA Sparse P/D TopK token IDs must be integers.")
                layers[layer_name] = list(token_ids)
            layer_topk_by_rank[rank] = layers

        raw_tail_blocks = raw_handoff.get("partial_tail_blocks_by_rank", {})
        if not isinstance(raw_tail_blocks, dict):
            raise TypeError("DSA Sparse partial_tail_blocks_by_rank must be a dictionary.")
        partial_tail_blocks_by_rank: dict[int, dict[str, int]] = {}
        for raw_rank, raw_layers in raw_tail_blocks.items():
            if not isinstance(raw_layers, dict):
                raise TypeError("DSA Sparse rank partial-tail blocks must be a dictionary.")
            rank = int(raw_rank)
            partial_tail_blocks_by_rank[rank] = {}
            for layer_name, block_id in raw_layers.items():
                if not isinstance(layer_name, str) or isinstance(block_id, bool) or not isinstance(block_id, int):
                    raise TypeError("DSA Sparse partial-tail descriptors are invalid.")
                partial_tail_blocks_by_rank[rank][layer_name] = block_id

        integer_fields: dict[str, object] = {
            "protocol_version": raw_handoff.get("protocol_version", DSA_SPARSE_PD_PROTOCOL_VERSION),
            "stored_token_count": raw_handoff.get("stored_token_count", 0),
            "block_size": raw_handoff.get("block_size", 0),
        }
        for field_name, field_value in integer_fields.items():
            if isinstance(field_value, bool) or not isinstance(field_value, int):
                raise TypeError(f"DSA Sparse P/D {field_name} must be an integer.")
        remote_request_id = raw_handoff.get("remote_request_id", "")
        if not isinstance(remote_request_id, str):
            raise TypeError("DSA Sparse P/D remote_request_id must be a string.")
        return cls(
            protocol_version=integer_fields["protocol_version"],  # type: ignore[arg-type]
            remote_request_id=remote_request_id,
            stored_token_count=integer_fields["stored_token_count"],  # type: ignore[arg-type]
            block_size=integer_fields["block_size"],  # type: ignore[arg-type]
            layer_topk_by_rank=layer_topk_by_rank,
            partial_tail_blocks_by_rank=partial_tail_blocks_by_rank,
        )


def get_dsa_sparse_pd_handoff(
    kv_transfer_params: object,
) -> DSASparsePDHandoff | None:
    if not isinstance(kv_transfer_params, dict):
        return None
    raw_handoff = kv_transfer_params.get(DSA_SPARSE_PD_HANDOFF_KEY)
    if raw_handoff is None:
        return None
    return DSASparsePDHandoff.from_dict(raw_handoff)


class DSASparseProducerAttentionContext(Protocol):
    def publish_layer(
        self,
        layer_name: str,
        semantic_topk_positions: torch.Tensor,
        main_cache: tuple[torch.Tensor, ...],
        block_table: torch.Tensor,
    ) -> None: ...


class DSASparseProducerBatchContext:
    """Capture one final-Prefill TopK row for every SFA layer/request."""

    def __init__(
        self,
        *,
        request_ids: Sequence[str],
        scheduled_token_counts: Sequence[int],
        stored_token_counts: Sequence[int],
        publish_requests: Sequence[bool],
        layer_metadata: Mapping[str, object],
        backend: DSASparseKVBackend | None = None,
        storage_key_encoder: DSASparseStorageKeyEncoder | None = None,
        committed_block_hashes: Mapping[str, Sequence[bytes | int]] | None = None,
    ) -> None:
        self.request_ids = tuple(str(request_id) for request_id in request_ids)
        self.scheduled_token_counts = tuple(int(count) for count in scheduled_token_counts)
        self.stored_token_counts = tuple(int(count) for count in stored_token_counts)
        self.publish_requests = tuple(bool(value) for value in publish_requests)
        self.layer_metadata = dict(layer_metadata)
        self.backend = backend
        self.storage_key_encoder = storage_key_encoder
        self.committed_block_hashes = {
            request_id: list(block_hashes) for request_id, block_hashes in (committed_block_hashes or {}).items()
        }
        self._published_layers: set[str] = set()
        self._layer_topk: dict[str, dict[str, list[int]]] = {}
        self._partial_tail_blocks: dict[str, dict[str, int]] = {}

        request_count = len(self.request_ids)
        if not (
            len(self.scheduled_token_counts)
            == len(self.stored_token_counts)
            == len(self.publish_requests)
            == request_count
        ):
            raise ValueError("DSA Sparse P capture vectors must have equal lengths.")
        if any(count <= 0 for count in self.scheduled_token_counts):
            raise ValueError("DSA Sparse P capture requires scheduled tokens.")
        if not any(self.publish_requests):
            raise ValueError("DSA Sparse P capture requires a final Prefill request.")

    def publish_layer(
        self,
        layer_name: str,
        semantic_topk_positions: torch.Tensor,
        main_cache: tuple[torch.Tensor, ...],
        block_table: torch.Tensor,
    ) -> None:
        if layer_name in self._published_layers:
            raise RuntimeError(f"DSA Sparse layer {layer_name!r} was published twice.")
        if layer_name not in self.layer_metadata:
            raise KeyError(f"Missing DSA Sparse P metadata for layer {layer_name!r}.")
        total_scheduled_tokens = sum(self.scheduled_token_counts)
        if semantic_topk_positions.shape[0] < total_scheduled_tokens:
            raise ValueError("DSA Sparse TopK rows do not cover the Prefill batch.")
        if not main_cache:
            raise ValueError("DSA Sparse P publication requires Main cache planes.")

        if self.backend is not None:
            if self.storage_key_encoder is None:
                raise RuntimeError("DSA Sparse P publication has no storage key encoder.")
            layer_id = extract_layer_index(layer_name)
            for request_index, (
                request_id,
                stored_token_count,
                should_publish,
            ) in enumerate(
                zip(
                    self.request_ids,
                    self.stored_token_counts,
                    self.publish_requests,
                )
            ):
                if not should_publish:
                    continue
                block_size = int(main_cache[0].shape[1])
                full_block_count = stored_token_count // block_size
                if full_block_count == 0:
                    continue
                block_hashes = self.committed_block_hashes.get(request_id, [])
                if len(block_hashes) < full_block_count:
                    raise RuntimeError(
                        f"DSA Sparse P publication is missing committed block hashes: request={request_id!r}."
                    )
                source_block_ids = block_table[request_index, :full_block_count].to(dtype=torch.int64)
                storage_request_ids = self.storage_key_encoder.encode_many(
                    block_hashes[:full_block_count],
                    layer_id,
                    device=source_block_ids.device,
                )
                self.backend.put_blocks(
                    layer_id=layer_id,
                    storage_request_ids=storage_request_ids,
                    source_block_ids=source_block_ids,
                )

        # This is the P-side payload publication boundary. The backend must
        # complete each PUT before TopK is recorded and handed to Decode.
        if dsa_sparse_probe.is_enabled():
            dsa_sparse_probe.emit(
                "initial_publish_mock",
                layer=layer_name,
                request_ids=list(self.request_ids),
                stored_token_counts=list(self.stored_token_counts),
                publish_requests=list(self.publish_requests),
                main_cache_ptrs=[plane.data_ptr() for plane in main_cache],
                block_table_shape=list(block_table.shape),
            )

        layer_topk: dict[str, list[int]] = {}
        layer_partial_tail_blocks: dict[str, int] = {}
        token_row_start = 0
        for request_index, (
            request_id,
            scheduled_token_count,
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
            token_row_end = token_row_start + scheduled_token_count
            if should_publish:
                # This is a one-time final-Prefill control-plane copy. Decode
                # remains tensor-native and does not copy TopK through CPU.
                request_topk = (
                    semantic_topk_positions[token_row_end - 1]
                    .detach()
                    .reshape(-1)
                    .to(device="cpu", dtype=torch.int32)
                    .tolist()
                )
                layer_topk[request_id] = [int(token_id) for token_id in request_topk]
                block_size = int(main_cache[0].shape[1])
                tail_valid_count = stored_token_count % block_size
                if tail_valid_count:
                    logical_block_idx = stored_token_count // block_size
                    layer_partial_tail_blocks[request_id] = int(block_table[request_index, logical_block_idx].item())
            token_row_start = token_row_end
        self._layer_topk[layer_name] = layer_topk
        self._partial_tail_blocks[layer_name] = layer_partial_tail_blocks
        self._published_layers.add(layer_name)

    def layer_topk(self, layer_name: str) -> dict[str, list[int]]:
        if layer_name not in self._published_layers:
            raise RuntimeError(f"DSA Sparse layer {layer_name!r} has not been published.")
        return {request_id: list(token_ids) for request_id, token_ids in self._layer_topk[layer_name].items()}

    def layer_partial_tail_blocks(self, layer_name: str) -> dict[str, int]:
        if layer_name not in self._published_layers:
            raise RuntimeError(f"DSA Sparse layer {layer_name!r} has not been published.")
        return dict(self._partial_tail_blocks[layer_name])


class DSASparseProducerExecution:
    """Attach a P-side capture context to SFA metadata for one forward."""

    def __init__(
        self,
        context: DSASparseProducerBatchContext,
        metadata_objects: Sequence[object],
    ) -> None:
        self.context = context
        self._metadata_objects = tuple(metadata_objects)
        self._closed = False

    def __enter__(self) -> DSASparseProducerBatchContext:
        if self._closed:
            raise RuntimeError("DSA Sparse P capture execution is already closed.")
        return self.context

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, traceback
        if self._closed:
            raise RuntimeError("DSA Sparse P capture execution is already closed.")
        first_error: BaseException | None = None
        for metadata in reversed(self._metadata_objects):
            try:
                current = getattr(metadata, "dsa_sparse_producer_context", None)
                if current is self.context:
                    metadata.dsa_sparse_producer_context = None
                elif current is not None:
                    raise RuntimeError("DSA Sparse P metadata context ownership changed before detach.")
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._closed = True
        if first_error is not None:
            if exc_value is not None and hasattr(exc_value, "add_note"):
                exc_value.add_note(f"DSA Sparse P metadata detach also failed: {first_error!r}")
                return False
            raise RuntimeError("Failed to detach DSA Sparse P metadata.") from first_error
        return False


def begin_dsa_sparse_producer_execution(
    *,
    request_ids: Sequence[str],
    scheduled_token_counts: Sequence[int],
    stored_token_counts: Sequence[int],
    publish_requests: Sequence[bool],
    layer_metadata: Mapping[str, object],
    backend: DSASparseKVBackend | None = None,
    storage_key_encoder: DSASparseStorageKeyEncoder | None = None,
    committed_block_hashes: Mapping[str, Sequence[bytes | int]] | None = None,
) -> DSASparseProducerExecution:
    context = DSASparseProducerBatchContext(
        request_ids=request_ids,
        scheduled_token_counts=scheduled_token_counts,
        stored_token_counts=stored_token_counts,
        publish_requests=publish_requests,
        layer_metadata=layer_metadata,
        backend=backend,
        storage_key_encoder=storage_key_encoder,
        committed_block_hashes=committed_block_hashes,
    )
    metadata_objects: list[object] = []
    seen: set[int] = set()
    for metadata in layer_metadata.values():
        if id(metadata) not in seen:
            metadata_objects.append(metadata)
            seen.add(id(metadata))

    attached: list[object] = []
    try:
        for metadata in metadata_objects:
            if not hasattr(metadata, "dsa_sparse_producer_context"):
                raise TypeError("DSA Sparse P metadata does not expose dsa_sparse_producer_context.")
            if metadata.dsa_sparse_producer_context is not None:
                raise RuntimeError("DSA Sparse P metadata already owns a context.")
            metadata.dsa_sparse_producer_context = context
            attached.append(metadata)
    except BaseException:
        for metadata in reversed(attached):
            if metadata.dsa_sparse_producer_context is context:
                metadata.dsa_sparse_producer_context = None
        raise
    return DSASparseProducerExecution(context, metadata_objects)


__all__ = [
    "DSA_SPARSE_PD_HANDOFF_KEY",
    "DSASparsePDHandoff",
    "DSASparseProducerAttentionContext",
    "DSASparseProducerBatchContext",
    "DSASparseProducerExecution",
    "begin_dsa_sparse_producer_execution",
    "build_dsa_sparse_resident_token_ids",
    "get_dsa_sparse_pd_handoff",
]
