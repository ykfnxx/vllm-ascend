"""DSA phase barrier and request lifecycle hooks for vLLM v0.18."""

from collections.abc import Iterable
from functools import wraps
from typing import Any

import vllm.v1.core.sched.output as output_mod
import vllm.v1.core.sched.scheduler as scheduler_mod
from vllm.logger import init_logger
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

import vllm_ascend.patch.dsa_sparse.patch_scheduler_output  # noqa: F401
from vllm_ascend.dsa_sparse.dsa_config import (
    DSA_SPARSE_SUPPORTED_ARCHITECTURES,
    is_dsa_sparse_config_enabled,
)
from vllm_ascend.dsa_sparse.dsa_sparse import DSASparseV1
from vllm_ascend.dsa_sparse.dsa_types import DSASparseRole, INVALID_SLOT, ReqStage

logger = init_logger(__name__)


def _is_prefill(request: Request) -> bool:
    return (
        request.num_output_tokens == 0
        and request.num_computed_tokens < request.num_prompt_tokens
    )


def _has_running_prefill(running: Iterable[Request]) -> bool:
    return any(_is_prefill(request) for request in running)


def _is_dsa_enabled(self: Scheduler) -> bool:
    return self.dsa_scheduler_mgr is not None


def _is_decode_request(self: Scheduler, request: Request) -> bool:
    return request.num_output_tokens > 0


def _has_prefill_work(self: Scheduler) -> bool:
    return _is_dsa_enabled(self) and _has_running_prefill(self.running)


def _has_schedulable_waiting_prefill(
    self: Scheduler, token_budget: int
) -> bool:
    if token_budget <= 0 or len(self.running) >= self.max_num_running_reqs:
        return False
    request_queue = self._select_waiting_queue_for_scheduling()
    if request_queue is None:
        return False
    request = request_queue.peek_request()
    if not _is_prefill(request) or self._is_blocked_waiting_status(request.status):
        return False
    num_new_tokens = request.num_tokens - request.num_computed_tokens
    threshold = self.scheduler_config.long_prefill_token_threshold
    if 0 < threshold < num_new_tokens:
        num_new_tokens = threshold
    if (
        not self.scheduler_config.enable_chunked_prefill
        and num_new_tokens > token_budget
    ):
        return False
    return min(num_new_tokens, token_budget) > 0


def _has_ready_decode(self: Scheduler) -> bool:
    for request in self.running:
        if not self._is_dsa_decode_request(request):
            continue
        if (
            request.num_output_placeholders > 0
            and request.num_computed_tokens
            + 2
            - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        ):
            continue
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        num_new_tokens = min(
            num_new_tokens,
            self.max_model_len - 1 - request.num_computed_tokens,
        )
        if num_new_tokens > 0:
            return True
    return False


def _estimate_resident_slots(self: Scheduler, request: Request) -> int:
    return self.dsa_scheduler_mgr.plan_decode_resident_slots(request)


def _install_allocate_slots_wrapper(self: Scheduler) -> None:
    kv_cache_manager = self.kv_cache_manager
    original_allocate_slots = kv_cache_manager.allocate_slots

    @wraps(original_allocate_slots)
    def allocate_slots(
        request: Request, num_new_tokens: int, *args: Any, **kwargs: Any
    ):
        if kv_cache_manager._dsa_inside_allocate_slots:
            return original_allocate_slots(
                request, num_new_tokens, *args, **kwargs
            )
        resident_valid_seq_len = self._estimate_dsa_resident_slots(request)
        kv_cache_manager._dsa_inside_allocate_slots = True
        try:
            return self.dsa_scheduler_mgr.dsa_alloc_slots_wrap(
                kv_cache_manager,
                request,
                resident_valid_seq_len,
                num_new_tokens,
                *args,
                **kwargs,
            )
        finally:
            kv_cache_manager._dsa_inside_allocate_slots = False

    kv_cache_manager._dsa_inside_allocate_slots = False
    kv_cache_manager.allocate_slots = allocate_slots


def _withhold_decode_for_prefill(self: Scheduler):
    withheld = [
        (index, request)
        for index, request in enumerate(self.running)
        if self._is_dsa_decode_request(request)
    ]
    if not withheld:
        return None
    old_max_num_running_reqs = self.max_num_running_reqs
    withheld_requests = {request for _, request in withheld}
    self.running = [
        request for request in self.running if request not in withheld_requests
    ]
    self.max_num_running_reqs -= len(withheld)

    def restore() -> None:
        restored = list(self.running)
        for index, request in withheld:
            if (
                request.request_id in self.requests
                and request.status == RequestStatus.RUNNING
            ):
                restored.insert(min(index, len(restored)), request)
        self.running = restored
        self.max_num_running_reqs = old_max_num_running_reqs

    return restore


def _withhold_waiting_for_decode(self: Scheduler):
    old_waiting = self.waiting
    old_skipped_waiting = self.skipped_waiting
    self.waiting = create_request_queue(self.policy)
    self.skipped_waiting = create_request_queue(self.policy)

    def restore() -> None:
        new_waiting = list(self.waiting)
        new_skipped_waiting = list(self.skipped_waiting)
        for request in reversed(new_waiting):
            old_waiting.prepend_request(request)
        for request in new_skipped_waiting:
            old_skipped_waiting.add_request(request)
        self.waiting = old_waiting
        self.skipped_waiting = old_skipped_waiting

    return restore


def _populate_scheduler_output(self: Scheduler, scheduler_output) -> None:
    scheduler_output.req_dsa_stage = {}
    scheduler_output.req_dsa_resident_valid_seq_len = {}
    scheduler_output.req_dsa_sparse_budget_tokens = {}
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        scheduler_output.req_dsa_stage[req_id] = int(
            ReqStage.coerce(request.dsa_req_stage)
        )
        scheduler_output.req_dsa_resident_valid_seq_len[req_id] = (
            request.dsa_resident_valid_seq_len
        )
        scheduler_output.req_dsa_sparse_budget_tokens[req_id] = (
            request.dsa_sparse_budget_tokens
        )

    for new_request in scheduler_output.scheduled_new_reqs:
        new_request.block_hashes = list(
            self.requests[new_request.req_id].block_hashes
        )

    cached_requests = scheduler_output.scheduled_cached_reqs
    cached_requests.block_hashes = [
        list(self.requests[req_id].block_hashes)
        for req_id in cached_requests.req_ids
    ]


scheduler_mod.NewRequestData = output_mod.NewRequestData
scheduler_mod.CachedRequestData = output_mod.CachedRequestData
scheduler_mod.SchedulerOutput = output_mod.SchedulerOutput

_original_init = Scheduler.__init__
_original_schedule = Scheduler.schedule
_original_preempt_request = Scheduler._preempt_request
_original_update_from_output = Scheduler.update_from_output
_original_add_request = Scheduler.add_request
_original_free_request = Scheduler._free_request


@wraps(_original_init)
def _scheduler_init(self: Scheduler, *args: Any, **kwargs: Any) -> None:
    _original_init(self, *args, **kwargs)
    self.dsa_scheduler_mgr = None
    if (
        self.vllm_config.model_config.architecture in DSA_SPARSE_SUPPORTED_ARCHITECTURES
        and is_dsa_sparse_config_enabled(self.vllm_config)
    ):
        self.dsa_scheduler_mgr = DSASparseV1(
            self.vllm_config, DSASparseRole.SCHEDULER
        )
        _install_allocate_slots_wrapper(self)
        logger.info(
            "DSA sparse scheduler manager enabled: architecture=%s, "
            "block_size=%d, sparse_budget=%d, resident_tokens=%d",
            self.vllm_config.model_config.architecture,
            self.vllm_config.cache_config.block_size,
            self.vllm_config.cache_config.dsa_hbm_sparse_budget,
            self.vllm_config.cache_config.dsa_hbm_resident_tokens,
        )
    elif is_dsa_sparse_config_enabled(self.vllm_config):
        logger.warning(
            "DSA sparse scheduler patch is loaded but its manager is not "
            "enabled: architecture=%s",
            self.vllm_config.model_config.architecture,
        )
    self.dsa_prefill_full_released_req_ids = set()


@wraps(_original_schedule)
def _schedule(self: Scheduler):
    if not _is_dsa_enabled(self):
        return _original_schedule(self)

    token_budget = (
        0
        if self._pause_state == PauseState.PAUSED_ALL
        else self.max_num_scheduled_tokens
    )
    restore = None
    if self._has_dsa_prefill_work() or self._has_schedulable_dsa_waiting_prefill(
        token_budget
    ):
        restore = _withhold_decode_for_prefill(self)
    elif _has_ready_decode(self):
        restore = _withhold_waiting_for_decode(self)
    try:
        scheduler_output = _original_schedule(self)
    finally:
        if restore is not None:
            restore()

    if self.running:
        scheduler_output.num_common_prefix_blocks = (
            self.kv_cache_manager.get_num_common_prefix_blocks(
                self.running[0].request_id
            )
        )
    _populate_scheduler_output(self, scheduler_output)
    return scheduler_output


@wraps(_original_preempt_request)
def _preempt_request(
    self: Scheduler, request: Request, timestamp: float
) -> None:
    _original_preempt_request(self, request, timestamp)
    if _is_dsa_enabled(self):
        self.dsa_prefill_full_released_req_ids.discard(request.request_id)
        request.dsa_req_stage = ReqStage.PREFILL
        request.dsa_next_req_stage = ReqStage.PREFILL
        request.dsa_resident_valid_seq_len = INVALID_SLOT
        request.dsa_sparse_budget_tokens = 0


@wraps(_original_update_from_output)
def _update_from_output(self: Scheduler, scheduler_output, model_runner_output):
    outputs = _original_update_from_output(
        self, scheduler_output, model_runner_output
    )
    if _is_dsa_enabled(self):
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if (
                request is not None
                and not request.is_finished()
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                self._maybe_release_dsa_prefill_full_cache(request)
    return outputs


@wraps(_original_add_request)
def _add_request(self: Scheduler, request: Request) -> None:
    is_new_request = request.request_id not in self.requests
    _original_add_request(self, request)
    if is_new_request and _is_dsa_enabled(self):
        self.dsa_prefill_full_released_req_ids.discard(request.request_id)
        self.dsa_scheduler_mgr.request_begin(
            request.request_id, request.prompt_token_ids
        )


@wraps(_original_free_request)
def _free_request(
    self: Scheduler, request: Request, delay_free_blocks: bool = False
):
    if _is_dsa_enabled(self):
        self.dsa_scheduler_mgr.request_finished_in_scheduler(request.request_id)
        self.dsa_prefill_full_released_req_ids.discard(request.request_id)
    return _original_free_request(self, request, delay_free_blocks)


def _maybe_release_prefill_full_cache(
    self: Scheduler, request: Request
) -> None:
    if request.request_id in self.dsa_prefill_full_released_req_ids:
        return
    if not self.dsa_scheduler_mgr.should_release_full_cache_after_prefill(request):
        return
    if self.dsa_scheduler_mgr.release_prefill_full_cache_except_tail(
        self.kv_cache_manager, request
    ):
        self.dsa_prefill_full_released_req_ids.add(request.request_id)


Scheduler._is_dsa_decode_request = _is_decode_request
Scheduler._has_dsa_prefill_work = _has_prefill_work
Scheduler._has_schedulable_dsa_waiting_prefill = (
    _has_schedulable_waiting_prefill
)
Scheduler._estimate_dsa_resident_slots = _estimate_resident_slots
Scheduler.__init__ = _scheduler_init
Scheduler.schedule = _schedule
Scheduler._preempt_request = _preempt_request
Scheduler.update_from_output = _update_from_output
Scheduler.add_request = _add_request
Scheduler._free_request = _free_request
Scheduler._maybe_release_dsa_prefill_full_cache = (
    _maybe_release_prefill_full_cache
)
logger.info("DSA sparse scheduler patch installed")
