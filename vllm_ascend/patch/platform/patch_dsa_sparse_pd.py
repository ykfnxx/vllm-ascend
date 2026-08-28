# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Patch the two upstream scheduler boundaries needed by DSA Sparse P/D.

vLLM calls ``KVConnector.request_finished`` while processing generated
tokens, but normally forwards worker metadata to the connector only after
that loop. DSA Sparse needs the same-step final-Prefill TopK when constructing
the P-to-D transfer parameters, so consume only this metadata before entering
the original scheduler method. The model runner also needs scheduler block
hashes for persistent Main-KV object identities, so a thin schedule wrapper
attaches metadata prepared by the DSA Sparse scheduler helper.
"""

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.dsa_sparse_scheduler import attach_dsa_sparse_block_hashes

_ORIGINAL_UPDATE_FROM_OUTPUT = Scheduler.update_from_output
_ORIGINAL_SCHEDULE = Scheduler.schedule


def _dsa_sparse_schedule(self: Scheduler, *args, **kwargs):
    scheduler_output = _ORIGINAL_SCHEDULE(self, *args, **kwargs)
    return attach_dsa_sparse_block_hashes(self, scheduler_output)


def _dsa_sparse_update_from_output(
    self: Scheduler,
    scheduler_output,
    model_runner_output,
):
    connector_output = model_runner_output.kv_connector_output
    connector = self.connector
    if connector_output is not None and connector is not None:
        consume = getattr(
            connector,
            "update_dsa_sparse_before_request_finish",
            None,
        )
        if consume is not None:
            consume(connector_output)
    return _ORIGINAL_UPDATE_FROM_OUTPUT(
        self,
        scheduler_output,
        model_runner_output,
    )


if not getattr(
    Scheduler.update_from_output,
    "_vllm_ascend_dsa_sparse_pd",
    False,
):
    _dsa_sparse_update_from_output._vllm_ascend_dsa_sparse_pd = True
    Scheduler.update_from_output = _dsa_sparse_update_from_output

if not getattr(Scheduler.schedule, "_vllm_ascend_dsa_sparse_hashes", False):
    _dsa_sparse_schedule._vllm_ascend_dsa_sparse_hashes = True
    Scheduler.schedule = _dsa_sparse_schedule
