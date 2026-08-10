# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_QUERY_WIDTH,
    DSA_SPARSE_RESIDENT_SLOT_COUNT,
)

DSA_SPARSE_PD_HANDOFF_KEY = "dsa_sparse_pd_handoff"
DSA_SPARSE_PD_PROTOCOL_VERSION = 1


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
        raise ValueError(
            "DSA Sparse resident initialization requires stored tokens."
        )
    if block_size <= 0:
        raise ValueError(
            "DSA Sparse resident initialization requires block_size > 0."
        )
    if not 0 < resident_token_count <= DSA_SPARSE_RESIDENT_SLOT_COUNT:
        raise ValueError(
            "DSA Sparse resident_token_count must fit the 8K region."
        )

    # The final partial block is addressed through the independent dense-tail
    # region, so only complete historical blocks belong in lookup residency.
    dense_tail_start = (stored_token_count // block_size) * block_size
    target_count = min(resident_token_count, dense_tail_start)
    selected: list[int] = []
    selected_set: set[int] = set()
    for raw_token_id in topk_token_ids:
        token_id = int(raw_token_id)
        if (
            token_id < 0
            or token_id >= dense_tail_start
            or token_id in selected_set
        ):
            continue
        selected.append(token_id)
        selected_set.add(token_id)
        if len(selected) == target_count:
            return selected

    for token_id in range(dense_tail_start):
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
    protocol_version: int = DSA_SPARSE_PD_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != DSA_SPARSE_PD_PROTOCOL_VERSION:
            raise ValueError(
                "Unsupported DSA Sparse P/D handoff protocol version "
                f"{self.protocol_version}."
            )
        if not self.remote_request_id:
            raise ValueError(
                "DSA Sparse P/D remote_request_id must not be empty."
            )
        if self.stored_token_count <= 0:
            raise ValueError(
                "DSA Sparse P/D stored_token_count must be positive."
            )
        if self.block_size <= 0:
            raise ValueError("DSA Sparse P/D block_size must be positive.")
        if not self.layer_topk_by_rank:
            raise ValueError(
                "DSA Sparse P/D handoff requires per-rank layer TopK."
            )
        for rank, layer_topk in self.layer_topk_by_rank.items():
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ValueError(
                    "DSA Sparse P/D ranks must be non-negative integers."
                )
            if not layer_topk:
                raise ValueError(
                    f"DSA Sparse P/D rank {rank} has no layer TopK."
                )
            for layer_name, token_ids in layer_topk.items():
                if not layer_name:
                    raise ValueError(
                        "DSA Sparse P/D layer names must not be empty."
                    )
                if len(token_ids) != DSA_SPARSE_QUERY_WIDTH:
                    raise ValueError(
                        "DSA Sparse P/D layer TopK width must be "
                        f"{DSA_SPARSE_QUERY_WIDTH}, got {len(token_ids)} "
                        f"for {layer_name!r}."
                    )
                if any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    for token_id in token_ids
                ):
                    raise TypeError(
                        "DSA Sparse P/D TopK token IDs must be integers."
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "remote_request_id": self.remote_request_id,
            "stored_token_count": self.stored_token_count,
            "block_size": self.block_size,
            "layer_topk_by_rank": {
                str(rank): {
                    layer_name: list(token_ids)
                    for layer_name, token_ids in layer_topk.items()
                }
                for rank, layer_topk in self.layer_topk_by_rank.items()
            },
        }

    @classmethod
    def from_dict(cls, raw_handoff: object) -> DSASparsePDHandoff:
        if not isinstance(raw_handoff, dict):
            raise TypeError("DSA Sparse P/D handoff must be a dictionary.")
        raw_topk = raw_handoff.get("layer_topk_by_rank")
        if not isinstance(raw_topk, dict):
            raise TypeError(
                "DSA Sparse P/D layer_topk_by_rank must be a dictionary."
            )
        layer_topk_by_rank: dict[int, dict[str, list[int]]] = {}
        for raw_rank, raw_layers in raw_topk.items():
            if not isinstance(raw_layers, dict):
                raise TypeError(
                    "DSA Sparse P/D rank TopK must be a dictionary."
                )
            if isinstance(raw_rank, bool):
                raise TypeError(
                    "DSA Sparse P/D rank keys must be integers."
                )
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "DSA Sparse P/D rank keys must be integers."
                ) from error
            if str(rank) != str(raw_rank):
                raise TypeError(
                    "DSA Sparse P/D rank keys must use canonical integers."
                )
            layers: dict[str, list[int]] = {}
            for layer_name, token_ids in raw_layers.items():
                if not isinstance(layer_name, str):
                    raise TypeError(
                        "DSA Sparse P/D layer names must be strings."
                    )
                if not isinstance(token_ids, (list, tuple)):
                    raise TypeError(
                        "DSA Sparse P/D layer TopK must be a sequence."
                    )
                if any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    for token_id in token_ids
                ):
                    raise TypeError(
                        "DSA Sparse P/D TopK token IDs must be integers."
                    )
                layers[layer_name] = list(token_ids)
            layer_topk_by_rank[rank] = layers

        integer_fields: dict[str, object] = {
            "protocol_version": raw_handoff.get(
                "protocol_version", DSA_SPARSE_PD_PROTOCOL_VERSION
            ),
            "stored_token_count": raw_handoff.get("stored_token_count", 0),
            "block_size": raw_handoff.get("block_size", 0),
        }
        for field_name, field_value in integer_fields.items():
            if isinstance(field_value, bool) or not isinstance(
                field_value, int
            ):
                raise TypeError(
                    f"DSA Sparse P/D {field_name} must be an integer."
                )
        remote_request_id = raw_handoff.get("remote_request_id", "")
        if not isinstance(remote_request_id, str):
            raise TypeError(
                "DSA Sparse P/D remote_request_id must be a string."
            )
        return cls(
            protocol_version=integer_fields["protocol_version"],  # type: ignore[arg-type]
            remote_request_id=remote_request_id,
            stored_token_count=integer_fields["stored_token_count"],  # type: ignore[arg-type]
            block_size=integer_fields["block_size"],  # type: ignore[arg-type]
            layer_topk_by_rank=layer_topk_by_rank,
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
    ) -> None: ...


class DSASparseProducerBatchContext:
    """Capture one final-Prefill TopK row for every SFA layer/request."""

    def __init__(
        self,
        *,
        request_ids: Sequence[str],
        scheduled_token_counts: Sequence[int],
        publish_requests: Sequence[bool],
        layer_metadata: Mapping[str, object],
    ) -> None:
        self.request_ids = tuple(str(request_id) for request_id in request_ids)
        self.scheduled_token_counts = tuple(
            int(count) for count in scheduled_token_counts
        )
        self.publish_requests = tuple(bool(value) for value in publish_requests)
        self.layer_metadata = dict(layer_metadata)
        self._published_layers: set[str] = set()
        self._layer_topk: dict[str, dict[str, list[int]]] = {}

        request_count = len(self.request_ids)
        if not (
            len(self.scheduled_token_counts)
            == len(self.publish_requests)
            == request_count
        ):
            raise ValueError(
                "DSA Sparse P capture vectors must have equal lengths."
            )
        if any(count <= 0 for count in self.scheduled_token_counts):
            raise ValueError(
                "DSA Sparse P capture requires scheduled tokens."
            )
        if not any(self.publish_requests):
            raise ValueError(
                "DSA Sparse P capture requires a final Prefill request."
            )

    def publish_layer(
        self,
        layer_name: str,
        semantic_topk_positions: torch.Tensor,
    ) -> None:
        if layer_name in self._published_layers:
            raise RuntimeError(
                f"DSA Sparse layer {layer_name!r} was published twice."
            )
        if layer_name not in self.layer_metadata:
            raise KeyError(
                f"Missing DSA Sparse P metadata for layer {layer_name!r}."
            )
        total_scheduled_tokens = sum(self.scheduled_token_counts)
        if semantic_topk_positions.shape[0] < total_scheduled_tokens:
            raise ValueError(
                "DSA Sparse TopK rows do not cover the Prefill batch."
            )

        layer_topk: dict[str, list[int]] = {}
        token_row_start = 0
        for request_id, scheduled_token_count, should_publish in zip(
            self.request_ids,
            self.scheduled_token_counts,
            self.publish_requests,
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
                layer_topk[request_id] = [
                    int(token_id) for token_id in request_topk
                ]
            token_row_start = token_row_end
        self._layer_topk[layer_name] = layer_topk
        self._published_layers.add(layer_name)

    def layer_topk(self, layer_name: str) -> dict[str, list[int]]:
        if layer_name not in self._published_layers:
            raise RuntimeError(
                f"DSA Sparse layer {layer_name!r} has not been published."
            )
        return {
            request_id: list(token_ids)
            for request_id, token_ids in self._layer_topk[layer_name].items()
        }


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
            raise RuntimeError(
                "DSA Sparse P capture execution is already closed."
            )
        return self.context

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, traceback
        if self._closed:
            raise RuntimeError(
                "DSA Sparse P capture execution is already closed."
            )
        first_error: BaseException | None = None
        for metadata in reversed(self._metadata_objects):
            try:
                current = getattr(
                    metadata, "dsa_sparse_producer_context", None
                )
                if current is self.context:
                    setattr(metadata, "dsa_sparse_producer_context", None)
                elif current is not None:
                    raise RuntimeError(
                        "DSA Sparse P metadata context ownership changed "
                        "before detach."
                    )
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self._closed = True
        if first_error is not None:
            if exc_value is not None and hasattr(exc_value, "add_note"):
                exc_value.add_note(
                    "DSA Sparse P metadata detach also failed: "
                    f"{first_error!r}"
                )
                return False
            raise RuntimeError(
                "Failed to detach DSA Sparse P metadata."
            ) from first_error
        return False


def begin_dsa_sparse_producer_execution(
    *,
    request_ids: Sequence[str],
    scheduled_token_counts: Sequence[int],
    publish_requests: Sequence[bool],
    layer_metadata: Mapping[str, object],
) -> DSASparseProducerExecution:
    context = DSASparseProducerBatchContext(
        request_ids=request_ids,
        scheduled_token_counts=scheduled_token_counts,
        publish_requests=publish_requests,
        layer_metadata=layer_metadata,
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
                raise TypeError(
                    "DSA Sparse P metadata does not expose "
                    "dsa_sparse_producer_context."
                )
            if getattr(metadata, "dsa_sparse_producer_context") is not None:
                raise RuntimeError(
                    "DSA Sparse P metadata already owns a context."
                )
            setattr(metadata, "dsa_sparse_producer_context", context)
            attached.append(metadata)
    except BaseException:
        for metadata in reversed(attached):
            if getattr(metadata, "dsa_sparse_producer_context") is context:
                setattr(metadata, "dsa_sparse_producer_context", None)
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
