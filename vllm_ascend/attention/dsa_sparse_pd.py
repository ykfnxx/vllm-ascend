# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend import dsa_sparse_probe
from vllm_ascend.attention.dsa_sparse_shm import (
    DSASparseSharedMemoryPayload,
    DSASparseSharedMemoryStore,
)
from vllm_ascend.dsa_sparse_backend import (
    DSASparseKVBackend,
    DSASparseStorageKeyEncoder,
)
from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_QUERY_WIDTH,
    DSA_SPARSE_RESIDENT_SLOT_COUNT,
)

DSA_SPARSE_PD_HANDOFF_KEY = "dsa_sparse_pd_handoff"
DSA_SPARSE_PD_PROTOCOL_VERSION = 5


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
    shared_memory_payloads_by_rank: dict[
        int,
        dict[str, DSASparseSharedMemoryPayload],
    ]
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
        if set(self.shared_memory_payloads_by_rank) != set(
            self.layer_topk_by_rank
        ):
            raise ValueError(
                "DSA Sparse shared-memory ranks must match TopK ranks."
            )
        has_partial_tail = bool(self.stored_token_count % self.block_size)
        for rank, layer_payloads in self.shared_memory_payloads_by_rank.items():
            topk_layers = set(self.layer_topk_by_rank[rank])
            missing_topk_layers = topk_layers - set(layer_payloads)
            if missing_topk_layers:
                raise ValueError(
                    "DSA Sparse shared-memory payloads are missing TopK layers: "
                    f"rank={rank}, missing={sorted(missing_topk_layers)}."
                )
            if any(
                layer_payloads[layer_name].cache_kind != "indexer"
                for layer_name in topk_layers
            ):
                raise ValueError(
                    "DSA Sparse TopK layers require Indexer shared-memory payloads."
                )
            if has_partial_tail and any(
                not layer_payloads[layer_name].tail_planes
                for layer_name in topk_layers
            ):
                raise ValueError(
                    "DSA Sparse partial-tail shared-memory payload is missing "
                    "Main Tail planes."
                )
            if not has_partial_tail and any(
                layer_payloads[layer_name].tail_planes
                for layer_name in topk_layers
            ):
                raise ValueError(
                    "DSA Sparse aligned shared-memory payload must not carry "
                    "Main Tail planes."
                )
            for layer_name in set(layer_payloads) - topk_layers:
                payload = layer_payloads[layer_name]
                if (
                    payload.cache_kind != "mtp_draft"
                    or payload.cache_layer_name != layer_name
                    or payload.tail_planes
                ):
                    raise ValueError(
                        "DSA Sparse non-TopK shared-memory payload must be an "
                        f"MTP draft cache: rank={rank}, layer={layer_name!r}."
                    )

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
            "shared_memory_payloads_by_rank": {
                str(rank): {
                    layer_name: payload.to_dict()
                    for layer_name, payload in layer_payloads.items()
                }
                for rank, layer_payloads in (
                    self.shared_memory_payloads_by_rank.items()
                )
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

        raw_payloads = raw_handoff.get("shared_memory_payloads_by_rank")
        if not isinstance(raw_payloads, dict):
            raise TypeError(
                "DSA Sparse shared_memory_payloads_by_rank must be a dictionary."
            )
        shared_memory_payloads_by_rank: dict[
            int,
            dict[str, DSASparseSharedMemoryPayload],
        ] = {}
        for raw_rank, raw_layers in raw_payloads.items():
            if not isinstance(raw_layers, dict):
                raise TypeError(
                    "DSA Sparse rank shared-memory payloads must be a dictionary."
                )
            if isinstance(raw_rank, bool):
                raise TypeError(
                    "DSA Sparse shared-memory rank keys must be integers."
                )
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "DSA Sparse shared-memory rank keys must be integers."
                ) from error
            if str(rank) != str(raw_rank):
                raise TypeError(
                    "DSA Sparse shared-memory rank keys must use canonical integers."
                )
            shared_memory_payloads_by_rank[rank] = {}
            for layer_name, raw_payload in raw_layers.items():
                if not isinstance(layer_name, str):
                    raise TypeError(
                        "DSA Sparse shared-memory layer names must be strings."
                    )
                shared_memory_payloads_by_rank[rank][layer_name] = (
                    DSASparseSharedMemoryPayload.from_dict(raw_payload)
                )

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
            shared_memory_payloads_by_rank=shared_memory_payloads_by_rank,
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
        indexer_layer_name: str | None = None,
        indexer_cache: tuple[torch.Tensor, ...] = (),
    ) -> None: ...

    def publish_mtp_draft_layer(
        self,
        layer_name: str,
        cache: tuple[torch.Tensor, ...],
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
        shared_memory_store: DSASparseSharedMemoryStore | None = None,
    ) -> None:
        self.request_ids = tuple(str(request_id) for request_id in request_ids)
        self.scheduled_token_counts = tuple(int(count) for count in scheduled_token_counts)
        self.stored_token_counts = tuple(int(count) for count in stored_token_counts)
        self.publish_requests = tuple(bool(value) for value in publish_requests)
        self.layer_metadata = dict(layer_metadata)
        self.backend = backend
        self.storage_key_encoder = storage_key_encoder
        self.shared_memory_store = (
            shared_memory_store or DSASparseSharedMemoryStore()
        )
        self.committed_block_hashes = {
            request_id: list(block_hashes) for request_id, block_hashes in (committed_block_hashes or {}).items()
        }
        self._published_layers: set[str] = set()
        self._layer_topk: dict[str, dict[str, list[int]]] = {}
        self._shared_memory_payloads: dict[
            str,
            dict[str, DSASparseSharedMemoryPayload],
        ] = {}
        self._owned_shared_memory_payloads: list[
            DSASparseSharedMemoryPayload
        ] = []
        self._block_size: int | None = None

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
        indexer_layer_name: str | None = None,
        indexer_cache: tuple[torch.Tensor, ...] = (),
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
        if not indexer_layer_name or not indexer_cache:
            raise RuntimeError(
                "DSA Sparse shared-memory publication requires Indexer "
                "cache metadata."
            )
        if indexer_cache and int(indexer_cache[0].shape[1]) != int(
            main_cache[0].shape[1]
        ):
            raise RuntimeError(
                "DSA Sparse Main and Indexer block sizes do not match."
            )
        block_size = int(main_cache[0].shape[1])
        if self._block_size is None:
            self._block_size = block_size
        elif self._block_size != block_size:
            raise RuntimeError(
                "DSA Sparse P target layers have different block sizes."
            )

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
        layer_shared_memory_payloads: dict[
            str,
            DSASparseSharedMemoryPayload,
        ] = {}
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
                tail_valid_count = stored_token_count % block_size
                block_count = (
                    stored_token_count // block_size + bool(tail_valid_count)
                )
                source_block_ids = block_table[
                    request_index,
                    :block_count,
                ].to(dtype=torch.int64)
                if source_block_ids.numel() != block_count:
                    raise RuntimeError(
                        "DSA Sparse P Indexer publication block table is too "
                        f"short: request={request_id!r}, blocks={block_count}."
                    )
                main_tail_block_id = (
                    int(source_block_ids[-1].item())
                    if tail_valid_count
                    else None
                )
                payload = self.shared_memory_store.publish(
                    cache_kind="indexer",
                    cache_layer_name=indexer_layer_name or "",
                    cache=indexer_cache,
                    cache_block_ids=source_block_ids,
                    logical_num_blocks=int(main_cache[0].shape[0]),
                    main_cache=main_cache,
                    main_tail_block_id=main_tail_block_id,
                    tail_valid_count=tail_valid_count,
                    compute_content_sha256=(
                        dsa_sparse_probe.is_enabled()
                    ),
                )
                layer_shared_memory_payloads[request_id] = payload
                self._owned_shared_memory_payloads.append(payload)
                if dsa_sparse_probe.is_enabled():
                    dsa_sparse_probe.emit(
                        "shared_memory_publish",
                        role="P",
                        request_id=request_id,
                        layer=layer_name,
                        object_name=payload.name,
                        payload_bytes=payload.size,
                        content_sha256=payload.content_sha256,
                        cache_kind=payload.cache_kind,
                        cache_plane_count=len(payload.cache_planes),
                        tail_plane_count=len(payload.tail_planes),
                    )
            token_row_start = token_row_end
        self._layer_topk[layer_name] = layer_topk
        self._shared_memory_payloads[layer_name] = (
            layer_shared_memory_payloads
        )
        self._published_layers.add(layer_name)

    def publish_mtp_draft_layer(
        self,
        layer_name: str,
        cache: tuple[torch.Tensor, ...],
        block_table: torch.Tensor,
    ) -> None:
        if layer_name in self._published_layers:
            raise RuntimeError(
                f"DSA Sparse MTP draft layer {layer_name!r} was published twice."
            )
        if not cache:
            raise ValueError(
                "DSA Sparse MTP draft publication requires cache planes."
            )
        block_size = int(cache[0].shape[1])
        if any(int(plane.shape[1]) != block_size for plane in cache):
            raise RuntimeError(
                "DSA Sparse MTP draft cache planes have different block sizes."
            )
        if self._block_size is None:
            raise RuntimeError(
                "DSA Sparse MTP draft was published before target layers."
            )
        if self._block_size != block_size:
            raise RuntimeError(
                "DSA Sparse target and MTP draft block sizes do not match."
            )
        layer_payloads: dict[str, DSASparseSharedMemoryPayload] = {}
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
            block_count = math.ceil(stored_token_count / block_size)
            source_block_ids = block_table[
                request_index,
                :block_count,
            ].to(dtype=torch.int64)
            if source_block_ids.numel() != block_count:
                raise RuntimeError(
                    "DSA Sparse P MTP draft publication block table is too "
                    f"short: request={request_id!r}, blocks={block_count}."
                )
            payload = self.shared_memory_store.publish(
                cache_kind="mtp_draft",
                cache_layer_name=layer_name,
                cache=cache,
                cache_block_ids=source_block_ids,
                logical_num_blocks=int(cache[0].shape[0]),
                compute_content_sha256=dsa_sparse_probe.is_enabled(),
            )
            layer_payloads[request_id] = payload
            self._owned_shared_memory_payloads.append(payload)
            if dsa_sparse_probe.is_enabled():
                dsa_sparse_probe.emit(
                    "shared_memory_publish",
                    role="P",
                    request_id=request_id,
                    layer=layer_name,
                    object_name=payload.name,
                    payload_bytes=payload.size,
                    content_sha256=payload.content_sha256,
                    cache_kind=payload.cache_kind,
                    cache_plane_count=len(payload.cache_planes),
                    tail_plane_count=0,
                )
        self._layer_topk[layer_name] = {}
        self._shared_memory_payloads[layer_name] = layer_payloads
        self._published_layers.add(layer_name)

    def layer_topk(self, layer_name: str) -> dict[str, list[int]]:
        if layer_name not in self._published_layers:
            raise RuntimeError(f"DSA Sparse layer {layer_name!r} has not been published.")
        return {request_id: list(token_ids) for request_id, token_ids in self._layer_topk[layer_name].items()}

    def layer_shared_memory_payloads(
        self,
        layer_name: str,
    ) -> dict[str, DSASparseSharedMemoryPayload]:
        if layer_name not in self._published_layers:
            raise RuntimeError(f"DSA Sparse layer {layer_name!r} has not been published.")
        return dict(self._shared_memory_payloads[layer_name])

    def release_owned_shared_memory_payloads(self) -> None:
        for payload in self._owned_shared_memory_payloads:
            self.shared_memory_store.unlink(payload)
        self._owned_shared_memory_payloads.clear()

    def transfer_shared_memory_ownership(self) -> None:
        self._owned_shared_memory_payloads.clear()


class DSASparseProducerExecution:
    """Own a P-side capture context through target and deferred MTP work."""

    def __init__(
        self,
        context: DSASparseProducerBatchContext,
        metadata_objects: Sequence[object],
        *,
        defer_completion: bool = False,
    ) -> None:
        self.context = context
        self._metadata_objects = tuple(metadata_objects)
        self._defer_completion = defer_completion
        self._closed = False

    @property
    def is_pending(self) -> bool:
        return not self._closed

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
        if exc_value is None and self._defer_completion:
            return False
        self._complete(
            successful=exc_value is None,
            original_error=exc_value,
        )
        return False

    def finish(self) -> None:
        self._complete(successful=True)

    def abort(self) -> None:
        self._complete(successful=False)

    def _complete(
        self,
        *,
        successful: bool,
        original_error: BaseException | None = None,
    ) -> None:
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
        if successful and first_error is None:
            self.context.transfer_shared_memory_ownership()
        else:
            self.context.release_owned_shared_memory_payloads()
        if first_error is not None:
            if original_error is not None and hasattr(
                original_error,
                "add_note",
            ):
                original_error.add_note(
                    "DSA Sparse P metadata detach also failed: "
                    f"{first_error!r}"
                )
                return
            raise RuntimeError("Failed to detach DSA Sparse P metadata.") from first_error


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
    shared_memory_store: DSASparseSharedMemoryStore | None = None,
    defer_completion: bool = False,
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
        shared_memory_store=shared_memory_store,
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
    return DSASparseProducerExecution(
        context,
        metadata_objects,
        defer_completion=defer_completion,
    )


__all__ = [
    "DSA_SPARSE_PD_HANDOFF_KEY",
    "DSASparsePDHandoff",
    "DSASparseSharedMemoryPayload",
    "DSASparseProducerAttentionContext",
    "DSASparseProducerBatchContext",
    "DSASparseProducerExecution",
    "begin_dsa_sparse_producer_execution",
    "build_dsa_sparse_resident_token_ids",
    "get_dsa_sparse_pd_handoff",
]
