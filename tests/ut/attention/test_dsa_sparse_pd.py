# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass, field

import pytest

from vllm_ascend.attention.dsa_sparse import RequestIndexManager
from vllm_ascend.attention.dsa_sparse_pd import (
    DSASparsePDLifecycle,
    DSASparseTransferCompletion,
)


@dataclass
class RecordingCoordinator:
    request_index_manager: RequestIndexManager
    busy_requests: set[str] = field(default_factory=set)

    def acquire_request(self, request_id):
        return self.request_index_manager.acquire(request_id)

    def assert_request_idle(self, request_id):
        if request_id in self.busy_requests:
            raise RuntimeError("pending layer I/O")

    def release_request(self, request_id):
        self.assert_request_idle(request_id)
        return self.request_index_manager.release(request_id)


@dataclass
class RecordingBackend:
    released: list[int] = field(default_factory=list)

    def release_request(self, request_handle):
        self.released.append(request_handle)


def make_lifecycle(max_num_seqs: int = 2):
    coordinator = RecordingCoordinator(RequestIndexManager(max_num_seqs))
    backend = RecordingBackend()
    lifecycle = DSASparsePDLifecycle(
        coordinator=coordinator,
        backend=backend,
    )
    return lifecycle, coordinator, backend


@pytest.mark.parametrize("main_first", [True, False])
def test_ready_fan_in_is_order_independent_and_notifies_once(main_first):
    lifecycle, coordinator, _backend = make_lifecycle()
    generation = lifecycle.begin_handoff("request-a", "transfer-a")
    completion = DSASparseTransferCompletion("request-a", generation)

    if main_first:
        lifecycle.mark_main_region_ready(completion, request_handle=17)
        assert lifecycle.take_ready_notifications() == set()
        lifecycle.mark_indexer_ready(completion)
    else:
        assert lifecycle.filter_indexer_completions([completion]) == set()
        lifecycle.mark_main_region_ready(completion, request_handle=17)

    notifications = lifecycle.take_ready_notifications()
    assert notifications == {completion}
    assert lifecycle.ready_request_ids(notifications) == {"request-a"}
    assert lifecycle.take_ready_notifications() == set()
    assert coordinator.request_index_manager.active_request_ids == ()


def test_admission_allocates_request_index_only_after_dual_ready():
    lifecycle, coordinator, _backend = make_lifecycle()
    generation = lifecycle.begin_handoff("request-a", "transfer-a")
    completion = DSASparseTransferCompletion("request-a", generation)
    lifecycle.mark_main_region_ready(completion, request_handle=17)

    with pytest.raises(RuntimeError, match="both ready"):
        lifecycle.admit("request-a", generation)

    lifecycle.mark_indexer_ready(completion)
    request_index = lifecycle.admit("request-a", generation)

    assert request_index == 0
    assert coordinator.request_index_manager.active_request_ids == ("request-a",)
    assert lifecycle.admit("request-a", generation) == request_index


def test_preempt_releases_region_and_reuses_request_index():
    lifecycle, _coordinator, backend = make_lifecycle(max_num_seqs=1)
    first_generation = lifecycle.begin_handoff("request-a", "transfer-a")
    first_completion = DSASparseTransferCompletion(
        "request-a",
        first_generation,
    )
    lifecycle.mark_main_region_ready(first_completion, request_handle=17)
    lifecycle.mark_indexer_ready(first_completion)
    first_request_index = lifecycle.admit("request-a", first_generation)

    lifecycle.preempt("request-a", first_generation)
    second_generation = lifecycle.begin_handoff("request-a", "transfer-b")
    second_completion = DSASparseTransferCompletion(
        "request-a",
        second_generation,
    )
    lifecycle.mark_main_region_ready(second_completion, request_handle=29)
    lifecycle.mark_indexer_ready(second_completion)
    second_request_index = lifecycle.admit("request-a", second_generation)

    assert backend.released == [17]
    assert second_generation == first_generation + 1
    assert second_request_index == first_request_index


def test_late_completion_from_retired_generation_is_ignored():
    lifecycle, _coordinator, backend = make_lifecycle()
    first_generation = lifecycle.begin_handoff("request-a", "transfer-a")
    first_completion = DSASparseTransferCompletion(
        "request-a",
        first_generation,
    )
    lifecycle.abort_handoff("request-a", first_generation)
    second_generation = lifecycle.begin_handoff("request-a", "transfer-b")

    assert not lifecycle.mark_indexer_ready(first_completion)
    assert not lifecycle.mark_main_region_ready(
        first_completion,
        request_handle=17,
    )
    assert backend.released == [17]
    snapshot = lifecycle.snapshot("request-a")
    assert snapshot.generation == second_generation
    assert not snapshot.indexer_ready
    assert not snapshot.main_region_ready


def test_failed_handoff_never_notifies_or_admits():
    lifecycle, _coordinator, _backend = make_lifecycle()
    generation = lifecycle.begin_handoff("request-a", "transfer-a")
    completion = DSASparseTransferCompletion("request-a", generation)
    lifecycle.mark_main_region_ready(completion, request_handle=17)
    lifecycle.mark_failed(completion, "indexer transfer failed")

    assert lifecycle.take_ready_notifications() == set()
    with pytest.raises(RuntimeError, match="both ready"):
        lifecycle.admit("request-a", generation)


def test_main_completion_after_failure_is_released_without_ready():
    lifecycle, _coordinator, backend = make_lifecycle()
    generation = lifecycle.begin_handoff("request-a", "transfer-a")
    completion = DSASparseTransferCompletion("request-a", generation)
    lifecycle.mark_failed(completion, "indexer transfer failed")

    assert not lifecycle.mark_main_region_ready(
        completion,
        request_handle=17,
    )
    assert backend.released == [17]
    assert lifecycle.take_ready_notifications() == set()


def test_stale_ready_notification_cannot_release_new_generation():
    lifecycle, _coordinator, _backend = make_lifecycle()
    first_generation = lifecycle.begin_handoff("request-a", "transfer-a")
    first_completion = DSASparseTransferCompletion(
        "request-a",
        first_generation,
    )
    lifecycle.mark_main_region_ready(first_completion, request_handle=17)
    lifecycle.mark_indexer_ready(first_completion)
    old_notifications = lifecycle.take_ready_notifications()
    lifecycle.abort_handoff("request-a", first_generation)
    lifecycle.begin_handoff("request-a", "transfer-b")

    assert lifecycle.ready_request_ids(old_notifications) == set()


def test_active_layer_io_blocks_region_and_request_index_release():
    lifecycle, coordinator, backend = make_lifecycle()
    generation = lifecycle.begin_handoff("request-a", "transfer-a")
    completion = DSASparseTransferCompletion("request-a", generation)
    lifecycle.mark_main_region_ready(completion, request_handle=17)
    lifecycle.mark_indexer_ready(completion)
    lifecycle.admit("request-a", generation)
    coordinator.busy_requests.add("request-a")

    with pytest.raises(RuntimeError, match="pending layer I/O"):
        lifecycle.finish("request-a", generation)

    assert backend.released == []
    assert lifecycle.snapshot("request-a").admitted


def test_finish_releases_region_and_removes_request():
    lifecycle, coordinator, backend = make_lifecycle()
    generation = lifecycle.begin_handoff("request-a", "transfer-a")
    completion = DSASparseTransferCompletion("request-a", generation)
    lifecycle.mark_main_region_ready(completion, request_handle=17)
    lifecycle.mark_indexer_ready(completion)
    lifecycle.admit("request-a", generation)

    lifecycle.finish("request-a", generation)

    assert backend.released == [17]
    assert coordinator.request_index_manager.active_request_ids == ()
    with pytest.raises(KeyError, match="no active handoff"):
        lifecycle.snapshot("request-a")


def test_long_request_churn_keeps_only_active_lifecycle_state():
    lifecycle, coordinator, backend = make_lifecycle(max_num_seqs=1)

    for index in range(1_000):
        request_id = f"request-{index}"
        generation = lifecycle.begin_handoff(
            request_id,
            f"transfer-{index}",
        )
        completion = DSASparseTransferCompletion(
            request_id,
            generation,
        )
        lifecycle.mark_main_region_ready(
            completion,
            request_handle=index,
        )
        lifecycle.mark_indexer_ready(completion)
        lifecycle.admit(request_id, generation)
        lifecycle.finish(request_id, generation)

    assert lifecycle._requests == {}
    assert coordinator.request_index_manager.active_request_ids == ()
    assert len(backend.released) == 1_000
