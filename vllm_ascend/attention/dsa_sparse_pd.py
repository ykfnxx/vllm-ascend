# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

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
    """Build a TopK-first resident set while preserving score order.

    The last partial block lives in the independent dense-tail region. Valid
    TopK positions are selected first, then the remaining capacity is filled
    in historical token order without re-sorting the TopK prefix.
    """

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

    dense_tail_start = (
        stored_token_count // block_size
    ) * block_size
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
    """Serializable P-to-D control metadata.

    Main payload remains in the configured backend. The connector transports
    only request identity, layout bounds, and final-Prefill per-layer TopK.
    """

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
            raise ValueError(
                "DSA Sparse P/D block_size must be positive."
            )
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
                        f"{DSA_SPARSE_QUERY_WIDTH}, got "
                        f"{len(token_ids)} for {layer_name!r}."
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
    def from_dict(
        cls,
        raw_handoff: object,
    ) -> "DSASparsePDHandoff":
        if not isinstance(raw_handoff, dict):
            raise TypeError(
                "DSA Sparse P/D handoff must be a dictionary."
            )
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
        protocol_version = raw_handoff.get(
            "protocol_version",
            DSA_SPARSE_PD_PROTOCOL_VERSION,
        )
        stored_token_count = raw_handoff.get(
            "stored_token_count",
            0,
        )
        block_size = raw_handoff.get("block_size", 0)
        for field_name, field_value in (
            ("protocol_version", protocol_version),
            ("stored_token_count", stored_token_count),
            ("block_size", block_size),
        ):
            if isinstance(field_value, bool) or not isinstance(
                field_value,
                int,
            ):
                raise TypeError(
                    f"DSA Sparse P/D {field_name} must be an integer."
                )
        remote_request_id = raw_handoff.get(
            "remote_request_id",
            "",
        )
        if not isinstance(remote_request_id, str):
            raise TypeError(
                "DSA Sparse P/D remote_request_id must be a string."
            )
        return cls(
            protocol_version=protocol_version,
            remote_request_id=remote_request_id,
            stored_token_count=stored_token_count,
            block_size=block_size,
            layer_topk_by_rank=layer_topk_by_rank,
        )


def get_dsa_sparse_pd_handoff(
    kv_transfer_params: object,
) -> DSASparsePDHandoff | None:
    if not isinstance(kv_transfer_params, dict):
        return None
    raw_handoff = kv_transfer_params.get(
        DSA_SPARSE_PD_HANDOFF_KEY
    )
    if raw_handoff is None:
        return None
    return DSASparsePDHandoff.from_dict(raw_handoff)


class DSASparseRequestIndexCoordinator(Protocol):
    def acquire_request(self, request_id: Hashable) -> int: ...

    def assert_request_idle(self, request_id: Hashable) -> None: ...

    def release_request(self, request_id: Hashable) -> int: ...


class DSASparseRequestRegionBackend(Protocol):
    """Request-region owner.

    Handles must not be reused while a late completion can still reference
    them, and ``release_request`` must be idempotent for a previously released
    handle. This lets late generation-bearing completions be discarded
    without an unbounded framework-side tombstone set.
    """

    def release_request(self, request_handle: int) -> None: ...


@dataclass(frozen=True)
class DSASparseTransferCompletion:
    request_id: Hashable
    generation: int


@dataclass(frozen=True)
class DSASparseRequestSnapshot:
    request_id: Hashable
    generation: int
    transfer_id: str
    main_region_ready: bool
    indexer_ready: bool
    ready_notified: bool
    admitted: bool
    failed_reason: str | None
    main_region_handle: int | None
    request_index: int | None

    @property
    def ready(self) -> bool:
        return self.main_region_ready and self.indexer_ready and self.failed_reason is None


@dataclass
class _DSASparseRequestState:
    request_id: Hashable
    generation: int
    transfer_id: str
    main_region_ready: bool = False
    indexer_ready: bool = False
    ready_notified: bool = False
    admitted: bool = False
    failed_reason: str | None = None
    main_region_handle: int | None = None
    request_index: int | None = None

    @property
    def ready(self) -> bool:
        return self.main_region_ready and self.indexer_ready and self.failed_reason is None

    def snapshot(self) -> DSASparseRequestSnapshot:
        return DSASparseRequestSnapshot(
            request_id=self.request_id,
            generation=self.generation,
            transfer_id=self.transfer_id,
            main_region_ready=self.main_region_ready,
            indexer_ready=self.indexer_ready,
            ready_notified=self.ready_notified,
            admitted=self.admitted,
            failed_reason=self.failed_reason,
            main_region_handle=self.main_region_handle,
            request_index=self.request_index,
        )


class DSASparsePDLifecycle:
    """Decode-side Main/Indexer ready fan-in and request-index lifecycle."""

    def __init__(
        self,
        *,
        coordinator: DSASparseRequestIndexCoordinator,
        backend: DSASparseRequestRegionBackend,
    ) -> None:
        self._coordinator = coordinator
        self._backend = backend
        self._requests: dict[Hashable, _DSASparseRequestState] = {}
        self._next_generation = 1

    def begin_handoff(
        self,
        request_id: Hashable,
        transfer_id: str,
    ) -> int:
        if request_id in self._requests:
            raise RuntimeError(f"DSA Sparse request {request_id!r} already has an active handoff.")
        if not transfer_id:
            raise ValueError("DSA Sparse transfer_id must not be empty.")
        generation = self._next_generation
        self._next_generation += 1
        self._requests[request_id] = _DSASparseRequestState(
            request_id=request_id,
            generation=generation,
            transfer_id=transfer_id,
        )
        return generation

    def mark_main_region_ready(
        self,
        completion: DSASparseTransferCompletion,
        *,
        request_handle: int,
    ) -> bool:
        state = self._get_current(completion)
        if state is None:
            self._release_region(request_handle)
            return False
        if state.main_region_handle is not None:
            if state.main_region_handle != request_handle:
                raise RuntimeError("DSA Sparse Main region completed twice with different handles.")
            return True
        if state.failed_reason is not None:
            self._release_region(request_handle)
            return False
        self._require_waiting(state)
        state.main_region_handle = request_handle
        state.main_region_ready = True
        return True

    def mark_indexer_ready(
        self,
        completion: DSASparseTransferCompletion,
    ) -> bool:
        state = self._get_current(completion)
        if state is None:
            return False
        self._require_waiting(state)
        state.indexer_ready = True
        return True

    def mark_failed(
        self,
        completion: DSASparseTransferCompletion,
        reason: str,
    ) -> bool:
        state = self._get_current(completion)
        if state is None:
            return False
        self._require_waiting(state)
        if not reason:
            raise ValueError("DSA Sparse handoff failure reason must not be empty.")
        state.failed_reason = reason
        return True

    def take_ready_notifications(self) -> set[DSASparseTransferCompletion]:
        ready: set[DSASparseTransferCompletion] = set()
        for state in self._requests.values():
            if state.ready and not state.ready_notified:
                state.ready_notified = True
                ready.add(
                    DSASparseTransferCompletion(
                        request_id=state.request_id,
                        generation=state.generation,
                    )
                )
        return ready

    def ready_request_ids(
        self,
        notifications: Iterable[DSASparseTransferCompletion],
    ) -> set[Hashable]:
        """Validate generation-bearing notifications at the scheduler edge."""

        ready: set[Hashable] = set()
        for notification in notifications:
            state = self._get_current(notification)
            if state is not None and state.ready and state.ready_notified:
                ready.add(notification.request_id)
        return ready

    def admit(
        self,
        request_id: Hashable,
        generation: int,
    ) -> int:
        state = self._require_generation(request_id, generation)
        if not state.ready:
            raise RuntimeError(
                "DSA Sparse request cannot enter Decode running before Main region and Indexer KV are both ready."
            )
        if state.admitted:
            assert state.request_index is not None
            return state.request_index
        request_index = self._coordinator.acquire_request(request_id)
        state.request_index = request_index
        state.admitted = True
        return request_index

    def preempt(
        self,
        request_id: Hashable,
        generation: int,
    ) -> None:
        self._retire(request_id, generation)

    def finish(
        self,
        request_id: Hashable,
        generation: int,
    ) -> None:
        self._retire(request_id, generation)

    def abort_handoff(
        self,
        request_id: Hashable,
        generation: int,
    ) -> None:
        self._retire(request_id, generation)

    def snapshot(
        self,
        request_id: Hashable,
    ) -> DSASparseRequestSnapshot:
        try:
            state = self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"DSA Sparse request {request_id!r} has no active handoff.") from exc
        return state.snapshot()

    def filter_indexer_completions(
        self,
        completions: Iterable[DSASparseTransferCompletion],
    ) -> set[DSASparseTransferCompletion]:
        """Consume raw Indexer completions and emit only dual-ready requests."""

        for completion in completions:
            self.mark_indexer_ready(completion)
        return self.take_ready_notifications()

    def _retire(
        self,
        request_id: Hashable,
        generation: int,
    ) -> None:
        state = self._require_generation(request_id, generation)
        if state.admitted:
            self._coordinator.assert_request_idle(request_id)
        if state.main_region_handle is not None:
            self._release_region(state.main_region_handle)
        if state.admitted:
            self._coordinator.release_request(request_id)
        del self._requests[request_id]

    def _get_current(
        self,
        completion: DSASparseTransferCompletion,
    ) -> _DSASparseRequestState | None:
        state = self._requests.get(completion.request_id)
        if state is None or state.generation != completion.generation:
            return None
        return state

    def _require_generation(
        self,
        request_id: Hashable,
        generation: int,
    ) -> _DSASparseRequestState:
        try:
            state = self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"DSA Sparse request {request_id!r} has no active handoff.") from exc
        if state.generation != generation:
            raise RuntimeError(
                f"Stale DSA Sparse request generation {generation}; current generation is {state.generation}."
            )
        return state

    def _release_region(self, request_handle: int) -> None:
        self._backend.release_request(request_handle)

    @staticmethod
    def _require_waiting(state: _DSASparseRequestState) -> None:
        if state.admitted:
            raise RuntimeError("DSA Sparse handoff completion cannot mutate an admitted request.")
        if state.failed_reason is not None:
            raise RuntimeError("DSA Sparse handoff is already failed.")
