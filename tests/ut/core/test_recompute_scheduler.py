# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.core.dsa_sparse_scheduler import attach_dsa_sparse_block_hashes
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


def test_dsa_sparse_connector_only_request_carries_block_hashes():
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"dsa_sparse_config": {}},
        ),
        requests={
            "loading": SimpleNamespace(block_hashes=[b"block-0"]),
        },
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        scheduled_spec_decode_tokens={},
        kv_connector_metadata=SimpleNamespace(
            requests={
                "loading": SimpleNamespace(dsa_sparse_handoff=object()),
            }
        ),
    )

    attach_dsa_sparse_block_hashes(scheduler, scheduler_output)

    assert scheduler_output.dsa_sparse_connector_block_hashes == {
        "loading": [b"block-0"]
    }


def test_pd_consumer_first_step_injects_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.is_kv_producer = False
    scheduler.is_hybrid_model = False
    scheduler.is_mtp_kv_consumer = True
    scheduler.num_spec_tokens = 1
    scheduler.max_model_len = 1024
    scheduler.log_stats = False
    scheduler.connector = None

    enqueued_requests = []

    def enqueue_waiting_request(self, request):
        enqueued_requests.append(request)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)

    request = Request(
        request_id="pd-consumer-first-step",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert scheduler.requests[request.request_id] is request
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID]
    assert request.num_tokens_with_spec == request.num_tokens + 1
