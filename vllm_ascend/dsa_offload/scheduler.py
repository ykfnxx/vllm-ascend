# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from vllm.v1.utils import ConstantList

from .metadata import (
    CommittedBlockUpdate,
    DSAOffloadStepMetadata,
    DecodeHashContext,
    make_block_key,
)
from .pd import DSA_OFFLOAD_PD_HANDOFF_KEY

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler


def is_dsa_offload_handoff_request(request: Any) -> bool:
    """Return whether a request carries a DSA Offload P/D handoff."""
    params = getattr(request, "kv_transfer_params", None)
    return isinstance(params, Mapping) and DSA_OFFLOAD_PD_HANDOFF_KEY in params


@dataclass
class DSAOffloadAdmissionBudget:
    """Scheduler-side mirror of the fixed per-request Hot Cache rows.

    The worker owns the physical rows. The scheduler only needs to ensure that
    requests loading remote KV and requests already decoding never exceed the
    same ``max_num_seqs`` capacity. A row remains reserved across the remote-KV
    wait and decode phases and is released when the request leaves the
    scheduler.
    """

    max_rows: int
    admitted_request_ids: set[str] = field(default_factory=set)

    @property
    def remaining_rows(self) -> int:
        return self.max_rows - len(self.admitted_request_ids)

    def can_admit(self, request_id: str) -> bool:
        return request_id in self.admitted_request_ids or self.remaining_rows > 0

    def admit(self, request_id: str) -> bool:
        if request_id in self.admitted_request_ids:
            return True
        if self.remaining_rows <= 0:
            return False
        self.admitted_request_ids.add(request_id)
        return True

    def is_admitted(self, request_id: str) -> bool:
        return request_id in self.admitted_request_ids

    def sync(self, live_request_ids: set[str]) -> None:
        self.admitted_request_ids.intersection_update(live_request_ids)

    def release(self, request_ids: set[str]) -> None:
        self.admitted_request_ids.difference_update(request_ids)


@dataclass
class _DSAOffloadHashState:
    block_hasher: Callable[[Any], list[bytes]]
    canonical_by_request: dict[str, list[bytes]] = field(
        default_factory=dict
    )
    keys_by_request: dict[str, list[int]] = field(default_factory=dict)
    published_by_request: dict[str, int] = field(default_factory=dict)

    def committed(
        self,
        request_id: str,
        request: Any,
    ) -> tuple[list[bytes], list[int]]:
        canonical = self.canonical_by_request.setdefault(
            request_id,
            list(request.block_hashes),
        )
        keys = self.keys_by_request.setdefault(
            request_id,
            [make_block_key(block_hash) for block_hash in canonical],
        )
        if request.block_hashes:
            upstream = request.block_hashes
            common = min(len(canonical), len(upstream))
            if canonical[:common] != upstream[:common]:
                raise RuntimeError(
                    "DSA Offload block hashes diverged from the scheduler "
                    f"for request {request_id}."
                )
            if len(upstream) > len(canonical):
                appended = upstream[len(canonical) :]
                canonical.extend(appended)
                keys.extend(make_block_key(block_hash) for block_hash in appended)

        if request._block_hasher is None:
            candidate = copy.copy(request)
            candidate.block_hashes = canonical
            appended = self.block_hasher(candidate)
            canonical.extend(appended)
            keys.extend(make_block_key(block_hash) for block_hash in appended)
        return canonical, keys

    def candidate_keys(
        self,
        request_id: str,
        request: Any,
        token_ids: list[int],
    ) -> tuple[int, ...]:
        canonical, _ = self.committed(request_id, request)
        candidate = copy.copy(request)
        candidate._all_token_ids = [*request._all_token_ids, *token_ids]
        candidate.all_token_ids = ConstantList(candidate._all_token_ids)
        candidate.block_hashes = canonical
        return tuple(
            make_block_key(block_hash)
            for block_hash in self.block_hasher(candidate)
        )

    def publish(
        self,
        request_id: str,
        *,
        snapshot: bool,
    ) -> CommittedBlockUpdate | None:
        keys = self.keys_by_request[request_id]
        base_count = 0 if snapshot else self.published_by_request.get(
            request_id,
            0,
        )
        if base_count == len(keys) and not snapshot:
            return None
        self.published_by_request[request_id] = len(keys)
        return base_count, tuple(keys[base_count:])

    def release(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self.canonical_by_request.pop(request_id, None)
            self.keys_by_request.pop(request_id, None)
            self.published_by_request.pop(request_id, None)


def _create_dsa_hash_state(scheduler: "Scheduler") -> _DSAOffloadHashState:
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        init_none_hash,
    )

    caching_hash_fn = get_hash_fn_by_name(
        scheduler.vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)
    return _DSAOffloadHashState(
        get_request_block_hasher(scheduler.block_size, caching_hash_fn)
    )


def _get_dsa_hash_state(scheduler: "Scheduler") -> _DSAOffloadHashState:
    state = getattr(scheduler, "_vllm_ascend_dsa_offload_hash_state", None)
    if state is None:
        state = _create_dsa_hash_state(scheduler)
        scheduler._vllm_ascend_dsa_offload_hash_state = state
    return state


def _committed_hashes(
    scheduler: "Scheduler",
    request_id: str,
) -> list[bytes]:
    request = scheduler.requests[request_id]
    canonical, _ = _get_dsa_hash_state(scheduler).committed(
        request_id,
        request,
    )
    return canonical


def _decode_hash_context(
    scheduler: "Scheduler",
    request_id: str,
    committed_hashes: list[bytes],
) -> DecodeHashContext:
    from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys

    request = scheduler.requests[request_id]
    block_size = scheduler.block_size
    block_index = len(committed_hashes)
    block_start = block_index * block_size
    block_end = block_start + block_size
    known_token_ids = tuple(request._all_token_ids[block_start:block_end])
    extra_keys, _ = generate_block_hash_extra_keys(
        request,
        block_start,
        block_end,
        0 if block_start == 0 else -1,
    )
    parent_hash = committed_hashes[-1] if committed_hashes else None
    return block_index, parent_hash, known_token_ids, extra_keys


def _needs_decode_hash_context(
    scheduler: "Scheduler",
    request_id: str,
    committed_hashes: Sequence[bytes],
) -> bool:
    request = scheduler.requests[request_id]
    next_block_end = (len(committed_hashes) + 1) * scheduler.block_size
    return (
        request.num_tokens + request.num_output_placeholders
        >= next_block_end
    )


def dsa_offload_enabled(scheduler: "Scheduler") -> bool:
    vllm_config = getattr(scheduler, "vllm_config", None)
    additional_config = getattr(vllm_config, "additional_config", None)
    return isinstance(additional_config, Mapping) and "dsa_offload" in additional_config


def dsa_offload_consumer_enabled(scheduler: "Scheduler") -> bool:
    """Return whether this scheduler belongs to a DSA Offload Decode node."""
    if not dsa_offload_enabled(scheduler):
        return False
    transfer_config = getattr(
        getattr(scheduler, "vllm_config", None),
        "kv_transfer_config",
        None,
    )
    return bool(
        transfer_config is not None
        and getattr(transfer_config, "is_kv_consumer", False)
    )


def attach_dsa_offload_metadata(
    scheduler: "Scheduler",
    scheduler_output: "SchedulerOutput",
) -> "SchedulerOutput":
    if not dsa_offload_enabled(scheduler):
        return scheduler_output

    committed_by_request: dict[str, list[bytes]] = {}
    state = _get_dsa_hash_state(scheduler)

    def committed(request_id: str) -> list[bytes]:
        if request_id not in committed_by_request:
            committed_by_request[request_id] = _committed_hashes(
                scheduler,
                request_id,
            )
        return committed_by_request[request_id]

    cached = scheduler_output.scheduled_cached_reqs
    connector_metadata = scheduler_output.kv_connector_metadata
    connector_requests = connector_metadata.requests if connector_metadata is not None else {}
    connector_request_ids = {
        request_id
        for request_id, metadata in connector_requests.items()
        if metadata.dsa_offload_handoff is not None
    }
    scheduled_request_ids = {
        request_data.req_id
        for request_data in scheduler_output.scheduled_new_reqs
    }
    scheduled_request_ids.update(cached.req_ids)
    scheduled_request_ids.update(connector_request_ids)
    for request_id in scheduled_request_ids:
        committed(request_id)

    first_seen_request_ids = {
        request_data.req_id
        for request_data in scheduler_output.scheduled_new_reqs
    }
    first_seen_request_ids.update(connector_request_ids)
    resumed_request_ids = set(getattr(cached, "resumed_req_ids", ()))
    committed_updates: dict[str, CommittedBlockUpdate] = {}
    for request_id in scheduled_request_ids:
        update = state.publish(
            request_id,
            snapshot=(
                request_id in resumed_request_ids
                or (
                    request_id in first_seen_request_ids
                    and request_id not in state.published_by_request
                )
            ),
        )
        if update is not None:
            committed_updates[request_id] = update

    decode_contexts: dict[str, DecodeHashContext] = {}
    scheduler_config = getattr(scheduler.vllm_config, "scheduler_config", None)
    if getattr(scheduler_config, "async_scheduling", False):
        decode_contexts = {
            request_id: _decode_hash_context(
                scheduler,
                request_id,
                committed(request_id),
            )
            for request_id in scheduled_request_ids
            if _needs_decode_hash_context(
                scheduler,
                request_id,
                committed(request_id),
            )
        }

    candidate_keys: dict[str, tuple[int, ...]] = {}
    for request_id, token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
        if not token_ids:
            continue
        request = scheduler.requests[request_id]
        candidate = copy.copy(request)
        candidate._all_token_ids = [*request._all_token_ids, *token_ids]
        candidate.all_token_ids = ConstantList(candidate._all_token_ids)
        if request._block_hasher is None:
            keys = state.candidate_keys(
                request_id,
                request,
                token_ids,
            )
        else:
            candidate.block_hashes = committed(request_id)
            keys = tuple(
                make_block_key(block_hash)
                for block_hash in request._block_hasher(candidate)
            )
        if keys:
            candidate_keys[request_id] = keys
    scheduler_output.dsa_offload_metadata = DSAOffloadStepMetadata(
        committed_updates=committed_updates,
        decode_contexts=decode_contexts,
        candidate_keys=candidate_keys,
    )
    state.release(set(getattr(scheduler_output, "finished_req_ids", ())))
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
            return attach_dsa_offload_metadata(
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
