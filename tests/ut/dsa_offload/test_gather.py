# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.dsa_offload import gather as gather_mod
from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout
from vllm_ascend.dsa_offload.lookup import (
    DSAOffloadBatch,
    IndexCacheCohort,
    LookupPlan,
)

COHORT = IndexCacheCohort(
    "l3",
    "l3",
    ("l3", "l4", "l5", "l6"),
    (3, 4, 5, 6),
)


def make_batch(gather_stream) -> DSAOffloadBatch:
    return DSAOffloadBatch(
        layout=HotCacheLayout(4, 1, 2),
        hot_cache=None,
        io_backend=MagicMock(),
        cohorts=(COHORT,),
        lookup_states={},
        request_ids=("decode",),
        request_rows=torch.tensor([0], dtype=torch.int32),
        decode_request_indices=(0,),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([8], dtype=torch.int64),
        is_mtp=False,
        committed_block_keys={"decode": []},
        candidate_block_keys={},
        gather_stream=gather_stream,
    )


def make_plan() -> LookupPlan:
    empty = torch.empty(0, dtype=torch.int64)
    return LookupPlan(
        mapped_indices=torch.zeros((1, 2), dtype=torch.int32),
        miss_positions=empty,
        miss_logical_blocks=empty,
        miss_block_offsets=empty,
        miss_destination_slots=empty,
        miss_batch_indices=empty.to(torch.int32),
        query_request_rows=empty.to(torch.int32),
    )


def make_stream_hub():
    """Minimal torch.npu stream model with a switchable current stream."""
    hub = SimpleNamespace(
        compute=MagicMock(name="compute_stream"),
        gather=MagicMock(name="gather_stream"),
    )
    hub.current = hub.compute
    hub.current_stream = lambda: hub.current

    def stream_ctx(target):
        class _Ctx:
            def __enter__(self):
                hub.current = target
                return target

            def __exit__(self, *args):
                hub.current = hub.compute

        return _Ctx()

    hub.stream = stream_ctx
    return hub


def test_inline_fallback_without_dedicated_stream() -> None:
    batch = make_batch(gather_stream=None)
    plan = make_plan()
    calls = []

    with patch(
        "vllm_ascend.dsa_offload.lookup.load_plan_misses",
        side_effect=lambda plan, layer_id, batch: calls.append(layer_id),
    ):
        gather_mod.issue_leader_gather(plan=plan, layer_id=3, batch=batch)
        gather_mod.issue_follower_gathers(
            plan=plan,
            cohort=COHORT,
            leader_layer_id=3,
            batch=batch,
        )
        # No events were recorded: waiting must be a no-op.
        gather_mod.wait_layer_gather(batch=batch, layer_id=5)

    assert calls == [3, 4, 5, 6]
    assert batch.gather_events == {3: None, 4: None, 5: None, 6: None}


def test_dedicated_stream_issues_cohort_gathers_back_to_back() -> None:
    hub = make_stream_hub()
    entry_event = MagicMock(name="entry_event")
    hub.compute.record_event.return_value = entry_event
    layer_events = [MagicMock(name=f"event:{layer_id}") for layer_id in (3, 4, 5, 6)]
    hub.gather.record_event.side_effect = layer_events

    batch = make_batch(gather_stream=hub.gather)
    plan = make_plan()
    calls = []

    def fake_load(plan, layer_id, batch):
        calls.append((layer_id, hub.current))

    with (
        patch.object(torch.npu, "current_stream", side_effect=hub.current_stream),
        patch.object(torch.npu, "stream", side_effect=hub.stream),
        patch(
            "vllm_ascend.dsa_offload.lookup.load_plan_misses",
            side_effect=fake_load,
        ),
    ):
        gather_mod.issue_leader_gather(plan=plan, layer_id=3, batch=batch)
        gather_mod.issue_follower_gathers(
            plan=plan,
            cohort=COHORT,
            leader_layer_id=3,
            batch=batch,
        )
        # A layer's attention waits only on its own gather event.
        gather_mod.wait_layer_gather(batch=batch, layer_id=5)

    # The dedicated stream enters behind one compute-stream event.
    hub.gather.wait_event.assert_called_once_with(entry_event)
    # Every layer's gather ran on the dedicated stream, leader first.
    assert [call[0] for call in calls] == [3, 4, 5, 6]
    assert all(call[1] is hub.gather for call in calls)
    # Completion events are keyed by layer id.
    assert batch.gather_events == {
        3: layer_events[0],
        4: layer_events[1],
        5: layer_events[2],
        6: layer_events[3],
    }
    # The compute stream waited on exactly the requested layer's event.
    hub.compute.wait_event.assert_called_once_with(layer_events[2])


def test_limit_gather_stream_aiv(monkeypatch) -> None:
    import acl

    stream = SimpleNamespace(npu_stream=12345)
    set_limit = MagicMock(return_value=0)
    monkeypatch.setattr(acl.rt, "set_stream_res_limit", set_limit, raising=False)

    gather_mod.limit_gather_stream_aiv(stream, 4)

    # type 1 = AIV
    set_limit.assert_called_once_with(12345, 1, 4)

    # A non-zero return code degrades to an unlimited stream, no exception.
    monkeypatch.setattr(
        acl.rt, "set_stream_res_limit", MagicMock(return_value=7), raising=False
    )
    gather_mod.limit_gather_stream_aiv(stream, 4)

    # A missing pyACL degrades the same way.
    monkeypatch.setitem(sys.modules, "acl", None)
    gather_mod.limit_gather_stream_aiv(stream, 4)
