# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


def load_scheduler_module(monkeypatch):
    class ConstantList(list):
        pass

    class Scheduler:
        def schedule(self):
            return self.output

        def update_from_output(self, scheduler_output, model_runner_output):
            self.events.append("original")
            return "updated"

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.utils": types.ModuleType("vllm.v1.utils"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.scheduler": types.ModuleType("vllm.v1.core.sched.scheduler"),
    }
    modules["vllm.v1.utils"].ConstantList = ConstantList
    modules["vllm.v1.core.sched.scheduler"].Scheduler = Scheduler
    for module in modules.values():
        module.__path__ = []
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "_dsa_offload_scheduler_test",
        ROOT / "vllm_ascend" / "dsa_offload" / "scheduler.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Scheduler


def make_output():
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="new")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["cached"]),
        kv_connector_metadata=SimpleNamespace(
            requests={
                "load": SimpleNamespace(dsa_offload_handoff=object()),
                "ordinary": SimpleNamespace(dsa_offload_handoff=None),
            }
        ),
        scheduled_spec_decode_tokens={"mtp": [7, 8]},
    )


def make_request(block_hashes, tokens, hasher=None):
    return SimpleNamespace(
        block_hashes=list(block_hashes),
        _all_token_ids=list(tokens),
        all_token_ids=list(tokens),
        _block_hasher=hasher,
        num_tokens=len(tokens),
        num_output_placeholders=0,
    )


def make_fallback_hash_state(module, block_size=2):
    def block_hasher(request):
        start = len(request.block_hashes) * block_size
        hashes = []
        while start + block_size <= len(request._all_token_ids):
            block = request._all_token_ids[start : start + block_size]
            hashes.append(bytes(block))
            start += block_size
        return hashes

    return module._DSAOffloadHashState(block_hasher)


def test_scheduler_attaches_committed_load_only_and_new_candidate_hashes(
    monkeypatch,
) -> None:
    module, Scheduler = load_scheduler_module(monkeypatch)
    seen_candidates = []

    def hash_candidate(candidate):
        seen_candidates.append(candidate)
        return [b"candidate-1"]

    scheduler = Scheduler()
    scheduler.vllm_config = SimpleNamespace(additional_config={"dsa_offload": {}})
    scheduler.requests = {
        "new": make_request([b"new-0"], [1]),
        "cached": make_request([b"cached-0"], [2]),
        "load": make_request([b"load-0"], [3]),
        "ordinary": make_request([], [4]),
        "mtp": make_request([b"committed-0"], [5, 6], hash_candidate),
    }
    scheduler.output = make_output()

    output = scheduler.schedule()

    assert output.scheduled_new_reqs[0].block_hashes == [b"new-0"]
    assert output.scheduled_cached_reqs.block_hashes == [[b"cached-0"]]
    assert output.dsa_offload_connector_block_hashes == {"load": [b"load-0"]}
    assert output.dsa_offload_candidate_block_hashes == {"mtp": [b"candidate-1"]}
    assert seen_candidates[0]._all_token_ids == [5, 6, 7, 8]
    assert seen_candidates[0].block_hashes == [b"committed-0"]


def test_disabled_scheduler_returns_original_output(monkeypatch) -> None:
    _, Scheduler = load_scheduler_module(monkeypatch)
    scheduler = Scheduler()
    scheduler.vllm_config = SimpleNamespace(additional_config={})
    scheduler.output = make_output()
    scheduler.requests = {}

    assert scheduler.schedule() is scheduler.output
    assert not hasattr(scheduler.output, "dsa_offload_candidate_block_hashes")


def test_connector_free_scheduler_has_no_remote_handoff_hashes(
    monkeypatch,
) -> None:
    module, Scheduler = load_scheduler_module(monkeypatch)
    scheduler = Scheduler()
    scheduler.vllm_config = SimpleNamespace(additional_config={"dsa_offload": {}})
    scheduler.requests = {
        "new": make_request([b"new-0"], [1]),
        "cached": make_request([b"cached-0"], [2]),
    }
    scheduler.output = make_output()
    scheduler.output.kv_connector_metadata = None
    scheduler.output.scheduled_spec_decode_tokens = {}

    output = module.attach_block_hashes(scheduler, scheduler.output)

    assert output.dsa_offload_connector_block_hashes == {}


def test_async_scheduler_attaches_incomplete_decode_block_context(
    monkeypatch,
) -> None:
    module, Scheduler = load_scheduler_module(monkeypatch)
    kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils.generate_block_hash_extra_keys = (
        lambda request, start, end, mm_index: (("extra",), mm_index)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.kv_cache_utils",
        kv_cache_utils,
    )

    scheduler = Scheduler()
    scheduler.vllm_config = SimpleNamespace(
        additional_config={"dsa_offload": {}},
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )
    scheduler.block_size = 4
    scheduler.requests = {
        "request": make_request(
            [b"block-0"],
            [1, 2, 3, 4, 5, 6, 7],
            lambda request: [],
        )
    }
    scheduler.requests["request"].num_output_placeholders = 1
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="request")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
    )

    module.attach_block_hashes(scheduler, output)

    assert output.dsa_offload_decode_hash_contexts == {
        "request": (1, b"block-0", (5, 6, 7), ("extra",))
    }


def test_connector_free_scheduler_generates_incremental_dsa_hashes(
    monkeypatch,
) -> None:
    module, Scheduler = load_scheduler_module(monkeypatch)
    scheduler = Scheduler()
    scheduler.vllm_config = SimpleNamespace(additional_config={"dsa_offload": {}})
    request = make_request([], [1, 2, 3])
    scheduler.requests = {"request": request}
    scheduler._vllm_ascend_dsa_offload_hash_state = make_fallback_hash_state(
        module
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="request")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
    )

    module.attach_block_hashes(scheduler, output)

    assert output.scheduled_new_reqs[0].block_hashes == [bytes([1, 2])]
    assert request.block_hashes == []

    request._all_token_ids.append(4)
    request.all_token_ids.append(4)
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(req_ids=["request"])
    module.attach_block_hashes(scheduler, output)

    assert output.scheduled_cached_reqs.block_hashes == [
        [bytes([1, 2]), bytes([3, 4])]
    ]
    assert request.block_hashes == []


def test_connector_free_mtp_uses_dsa_hash_state(monkeypatch) -> None:
    module, Scheduler = load_scheduler_module(monkeypatch)
    scheduler = Scheduler()
    scheduler.vllm_config = SimpleNamespace(additional_config={"dsa_offload": {}})
    request = make_request([], [1, 2, 3])
    scheduler.requests = {"request": request}
    scheduler._vllm_ascend_dsa_offload_hash_state = make_fallback_hash_state(
        module
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="request")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={"request": [4]},
        finished_req_ids=set(),
    )

    module.attach_block_hashes(scheduler, output)

    assert output.scheduled_new_reqs[0].block_hashes == [bytes([1, 2])]
    assert output.dsa_offload_candidate_block_hashes == {
        "request": [bytes([3, 4])]
    }


def test_connector_free_hash_state_is_released(monkeypatch) -> None:
    module, Scheduler = load_scheduler_module(monkeypatch)
    scheduler = Scheduler()
    scheduler.vllm_config = SimpleNamespace(additional_config={"dsa_offload": {}})
    scheduler.requests = {"request": make_request([], [1, 2])}
    state = make_fallback_hash_state(module)
    scheduler._vllm_ascend_dsa_offload_hash_state = state
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="request")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        kv_connector_metadata=None,
        scheduled_spec_decode_tokens={},
        finished_req_ids={"request"},
    )

    module.attach_block_hashes(scheduler, output)

    assert output.scheduled_new_reqs[0].block_hashes == [bytes([1, 2])]
    assert state.committed_by_request == {}


def test_publish_metadata_is_consumed_before_request_finish(monkeypatch) -> None:
    _, Scheduler = load_scheduler_module(monkeypatch)
    scheduler = Scheduler()
    scheduler.vllm_config = SimpleNamespace(additional_config={"dsa_offload": {}})
    scheduler.events = []
    scheduler.connector = SimpleNamespace(
        update_dsa_offload_before_request_finish=lambda output: scheduler.events.append("consume")
    )
    connector_output = SimpleNamespace()
    model_output = SimpleNamespace(kv_connector_output=connector_output)

    result = scheduler.update_from_output(SimpleNamespace(), model_output)

    assert result == "updated"
    assert scheduler.events == ["consume", "original"]
