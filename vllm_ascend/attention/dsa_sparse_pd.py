# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Protocol

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
