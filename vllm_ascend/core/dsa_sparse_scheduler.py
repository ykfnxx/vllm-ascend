# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Scheduler metadata helpers for DSA Sparse persistent Main KV."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler


def _dsa_sparse_enabled(scheduler: Scheduler) -> bool:
    additional_config = getattr(scheduler.vllm_config, "additional_config", None)
    return isinstance(additional_config, dict) and "dsa_sparse_config" in additional_config


def attach_dsa_sparse_block_hashes(
    scheduler: Scheduler,
    scheduler_output: SchedulerOutput,
) -> SchedulerOutput:
    """Attach committed and current-step candidate hashes in place."""

    if not _dsa_sparse_enabled(scheduler):
        return scheduler_output

    for new_request in scheduler_output.scheduled_new_reqs:
        new_request.block_hashes = list(scheduler.requests[new_request.req_id].block_hashes)

    cached_requests = scheduler_output.scheduled_cached_reqs
    cached_requests.block_hashes = [
        list(scheduler.requests[request_id].block_hashes) for request_id in cached_requests.req_ids
    ]

    candidate_hashes: dict[str, list[bytes | int]] = {}
    for request_id, speculative_tokens in scheduler_output.scheduled_spec_decode_tokens.items():
        if not speculative_tokens:
            continue
        request = scheduler.requests[request_id]
        if request._block_hasher is None:
            raise RuntimeError("DSA Sparse MTP requires the scheduler block hasher.")
        candidate = copy.copy(request)
        candidate._all_token_ids = [*request._all_token_ids, *speculative_tokens]
        candidate.all_token_ids = ConstantList(candidate._all_token_ids)
        candidate.block_hashes = list(request.block_hashes)
        candidate_hashes[request_id] = list(candidate._block_hasher(candidate))
    scheduler_output.dsa_candidate_block_hashes = candidate_hashes
    return scheduler_output
