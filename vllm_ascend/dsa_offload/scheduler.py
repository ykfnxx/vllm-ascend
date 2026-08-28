# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy
from typing import TYPE_CHECKING, Any

from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler


def dsa_offload_enabled(scheduler: "Scheduler") -> bool:
    additional_config = scheduler.vllm_config.additional_config
    return "dsa_offload" in additional_config


def attach_block_hashes(
    scheduler: "Scheduler",
    scheduler_output: "SchedulerOutput",
) -> "SchedulerOutput":
    if not dsa_offload_enabled(scheduler):
        return scheduler_output

    for request_data in scheduler_output.scheduled_new_reqs:
        request_data.block_hashes = list(scheduler.requests[request_data.req_id].block_hashes)

    cached = scheduler_output.scheduled_cached_reqs
    cached.block_hashes = [list(scheduler.requests[request_id].block_hashes) for request_id in cached.req_ids]

    connector_metadata = scheduler_output.kv_connector_metadata
    connector_requests = connector_metadata.requests if connector_metadata is not None else {}
    scheduler_output.dsa_offload_connector_block_hashes = {
        request_id: list(scheduler.requests[request_id].block_hashes)
        for request_id, metadata in connector_requests.items()
        if metadata.dsa_offload_handoff is not None
    }

    candidate_hashes: dict[str, list[bytes]] = {}
    for request_id, token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
        if not token_ids:
            continue
        request = scheduler.requests[request_id]
        candidate = copy.copy(request)
        candidate._all_token_ids = [*request._all_token_ids, *token_ids]
        candidate.all_token_ids = ConstantList(candidate._all_token_ids)
        candidate.block_hashes = list(request.block_hashes)
        if request._block_hasher is None:
            raise RuntimeError("DSA Offload MTP requires the scheduler block hasher.")
        candidate_hashes[request_id] = list(request._block_hasher(candidate))
    scheduler_output.dsa_offload_candidate_block_hashes = candidate_hashes
    return scheduler_output


def consume_publish_metadata(
    scheduler: "Scheduler",
    model_runner_output: Any,
) -> None:
    connector_output = model_runner_output.kv_connector_output
    if connector_output is not None and scheduler.connector is not None:
        scheduler.connector.update_dsa_offload_before_request_finish(connector_output)


def install_scheduler_wrappers() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if not getattr(Scheduler.schedule, "_vllm_ascend_dsa_offload", False):
        original_schedule = Scheduler.schedule

        def schedule(self: "Scheduler", *args: Any, **kwargs: Any):
            return attach_block_hashes(
                self,
                original_schedule(self, *args, **kwargs),
            )

        schedule._vllm_ascend_dsa_offload = True
        Scheduler.schedule = schedule

    if not getattr(
        Scheduler.update_from_output,
        "_vllm_ascend_dsa_offload",
        False,
    ):
        original_update_from_output = Scheduler.update_from_output

        def update_from_output(
            self: "Scheduler",
            scheduler_output: "SchedulerOutput",
            model_runner_output: Any,
        ):
            if dsa_offload_enabled(self):
                consume_publish_metadata(self, model_runner_output)
            return original_update_from_output(
                self,
                scheduler_output,
                model_runner_output,
            )

        update_from_output._vllm_ascend_dsa_offload = True
        Scheduler.update_from_output = update_from_output


install_scheduler_wrappers()
