# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Cross-layer KV Gather scheduling on a dedicated NPU stream.

Every Decode miss Gather runs on one dedicated, graph-stable stream
(``batch.gather_stream``) instead of the compute stream.  A cohort leader
resolves the shared ``LookupPlan`` and then issues the gathers for the whole
cohort back-to-back:

* ``issue_leader_gather`` runs the leader layer's own gather.  It stays a
  standalone call site because the leader's gather will later be fused with
  neighbouring operators, and that fusion must not disturb the followers.
* ``issue_follower_gathers`` is the independent block that issues the
  remaining layers' gathers right after the leader's lookup instead of
  waiting for each follower layer's forward to reach it.  A follower's
  gather depends only on the cohort-shared plan, the step metadata, and its
  own layer id -- never on the follower layer's own compute.

Each gather records a completion event keyed by layer id, and
``wait_layer_gather`` orders a layer's sparse flash attention behind only its
own gather.  When no dedicated stream is available (e.g. CPU-only test
environments) every gather falls back to running inline on the current
stream and ``wait_layer_gather`` degenerates to a no-op.
"""

import torch
from vllm.logger import logger

from . import lookup as _lookup
from .lookup import DSAOffloadBatch, IndexCacheCohort, LookupPlan

# aclrtDevResLimitType values for acl.rt.set_stream_res_limit: 0 = AIC, 1 = AIV.
_ACL_RT_STREAM_RES_AIV = 1


def limit_gather_stream_aiv(stream, aiv_limit: int) -> None:
    """Cap the dedicated KV Gather stream at ``aiv_limit`` AIV cores.

    Without a per-stream resource limit the runtime lets the gather kernels
    occupy every AIV on the device, starving the compute stream when the two
    overlap.  KV Gather saturates the HBM/RDMA bandwidth with only a few AIV
    cores, so the surplus occupation is pure contention.  Failure to apply
    the limit (old CANN, unsupported device) degrades to an unlimited stream.
    """
    try:
        import acl

        ret = acl.rt.set_stream_res_limit(
            stream.npu_stream, _ACL_RT_STREAM_RES_AIV, aiv_limit
        )
    except Exception as exc:
        logger.warning(
            "DSA Offload cannot limit the KV Gather stream to %d AIV cores "
            "(%s); the stream runs without a resource limit.",
            aiv_limit,
            exc,
        )
        return
    if ret != 0:
        logger.warning(
            "DSA Offload acl.rt.set_stream_res_limit(AIV, %d) failed with "
            "error %d; the KV Gather stream runs without a resource limit.",
            aiv_limit,
            ret,
        )
        return
    logger.info("DSA Offload KV Gather stream limited to %d AIV cores.", aiv_limit)


def issue_leader_gather(
    *,
    plan: LookupPlan,
    layer_id: int,
    batch: DSAOffloadBatch,
) -> None:
    """Run the cohort leader's own KV Gather for this step.

    Kept as a standalone call site: the leader's gather is scheduled to be
    fused with neighbouring operators, and that fusion must not disturb the
    follower-layer early issue in ``issue_follower_gathers``.
    """
    stream = batch.gather_stream
    if stream is None:
        _lookup.load_plan_misses(plan, layer_id, batch)
        batch.gather_events[layer_id] = None
        return
    # The plan tensors were produced on the compute stream; entering the
    # dedicated stream behind this event also orders the gather after the
    # previous step's readers of the destination slots.
    stream.wait_event(torch.npu.current_stream().record_event())
    with torch.npu.stream(stream):
        _lookup.load_plan_misses(plan, layer_id, batch)
        batch.gather_events[layer_id] = torch.npu.current_stream().record_event()
    _record_plan_stream(plan, stream)


def issue_follower_gathers(
    *,
    plan: LookupPlan,
    cohort: IndexCacheCohort,
    leader_layer_id: int,
    batch: DSAOffloadBatch,
) -> None:
    """Issue the cohort's follower-layer KV Gathers back-to-back, early.

    The gather inputs (``plan`` slots/masks, step metadata, per-layer planes)
    are fully determined once the leader's lookup completes, so the follower
    transfers can start immediately and be hidden behind the remaining
    layers' compute.  All of them serialize on the same dedicated stream
    behind the entry event recorded in ``issue_leader_gather``.
    """
    stream = batch.gather_stream
    if stream is not None and leader_layer_id not in batch.gather_events:
        # The leader's gather was absorbed by the fused SFA operator, so the
        # entry fork in ``issue_leader_gather`` never ran.  Fork the compute
        # stream here: without it the side stream is not joined to the
        # capture stream (graph capture fails with "stream not joined"), and
        # the follower gathers would race the compute stream producing plan.
        stream.wait_event(torch.npu.current_stream().record_event())
        _record_plan_stream(plan, stream)
    for layer_id in cohort.layer_ids:
        if layer_id == leader_layer_id:
            continue
        if stream is None:
            _lookup.load_plan_misses(plan, layer_id, batch)
            batch.gather_events[layer_id] = None
            continue
        with torch.npu.stream(stream):
            _lookup.load_plan_misses(plan, layer_id, batch)
            batch.gather_events[layer_id] = torch.npu.current_stream().record_event()


def wait_layer_gather(*, batch: DSAOffloadBatch, layer_id: int) -> None:
    """Order the layer's sparse attention behind only its own KV Gather."""
    event = batch.gather_events.get(layer_id)
    if event is not None:
        torch.npu.current_stream().wait_event(event)


def _record_plan_stream(plan: LookupPlan, stream: object) -> None:
    """Keep plan tensors alive for the dedicated stream's pending work."""
    for tensor in (
        plan.mapped_indices,
        plan.miss_positions,
        plan.miss_logical_blocks,
        plan.miss_block_offsets,
        plan.miss_destination_slots,
        plan.miss_batch_indices,
        plan.query_request_rows,
        plan.query_indices,
        plan.lookup_slots,
        plan.dense_miss_mask,
    ):
        if tensor is not None and tensor.device.type == "npu":
            tensor.record_stream(stream)
