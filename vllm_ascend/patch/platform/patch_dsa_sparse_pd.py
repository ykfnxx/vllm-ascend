# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Preserve final-Prefill DSA TopK before P request teardown.

vLLM calls ``KVConnector.request_finished`` while processing generated
tokens, but normally forwards worker metadata to the connector only after
that loop. DSA Sparse needs the same-step final-Prefill TopK when constructing
the P-to-D transfer parameters, so consume only this metadata before entering
the original scheduler method.
"""

from vllm.v1.core.sched.scheduler import Scheduler


_ORIGINAL_UPDATE_FROM_OUTPUT = Scheduler.update_from_output


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
