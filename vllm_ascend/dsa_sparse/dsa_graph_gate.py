# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DSA row-mode decode 图模式的 gate 判定。

本文件只描述“当前 model forward 是否满足 DSA 图 replay 条件”，并给出
可预期 eager 回退原因。它不做图 capture/replay 本身，也不修改请求状态。
开启 DSA 图模式后，prefill、dense->sparse 转换、新满块 dump 等非 row-mode
decode 场景仍应走 eager；稳定 dense、稳定 sparse、以及二者混合的
single-token decode batch 都可以进入同一张 DSA row-mode 图。
"""

from dataclasses import dataclass
from typing import Iterable, Mapping

from vllm_ascend.dsa_sparse.dsa_types import ReqStage


DSA_GRAPH_PHASE_ROW_MODE_DECODE = "dsa_row_mode_decode"
DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY = (
    "enable_dsa_row_mode_decode_graph")
DSA_ROW_MODE_DECODE_GRAPH_EXPECTED_EAGER_REASONS = frozenset({
    "empty_batch",
    "total_tokens_mismatch",
    "capture_size_miss",
    "non_single_token_decode",
    "non_row_mode_stage",
    "decode_reaches_full_block_boundary",
})


def is_dsa_row_mode_decode_graph_enabled(
    additional_config: object,
) -> bool:
    """Return whether DSA row-mode decode graph validation is enabled.

    This is intentionally a single switch: if enabled, eligible row-mode
    single-token decode must use the graph. Normal non-graphable stages still
    run eager.
    """
    if not isinstance(additional_config, dict):
        return False
    return additional_config.get(
        DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY) is True


def is_dsa_row_mode_decode_graph_expected_eager(reason: str) -> bool:
    """Return whether a disabled graph gate should continue in eager.

    These reasons describe normal execution phases outside the current graph
    contract, not correctness problems in a graphable row-mode decode.
    """
    return reason in DSA_ROW_MODE_DECODE_GRAPH_EXPECTED_EAGER_REASONS


@dataclass(frozen=True)
class DSAGraphGateDecision:
    disabled: bool
    reason: str
    graph_phase: str | None = None
    row_count: int | None = None
    bad_req_id: str | None = None


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def evaluate_dsa_row_mode_decode_graph(
    *,
    req_ids: Iterable[str],
    scheduled_tokens: Mapping[str, int],
    req_stages: Mapping[str, int],
    resident_lens: Mapping[str, int],
    sparse_budgets: Mapping[str, int],
    total_tokens: int,
    capture_sizes: set[int],
    configured_budget: int,
    resident_graph_limit: int,
    context_lens_before_forward: Mapping[str, int],
    block_size: int,
    resident_sparse_ready: Mapping[str, bool] | None,
    has_full_block_dump: bool,
    full_block_dump_req_id: str | None = None,
) -> DSAGraphGateDecision:
    """Gate DSA row-mode graph replay for stable single-token decode batches.

    DSA split-cache owns decode metadata even for pure dense rows, so dense,
    sparse, and mixed dense/sparse rows all use the same row-mode DSA graph
    once the batch is a stable single-token decode. Transition and
    side-effect-heavy paths still stay eager:
    ENTER_SPARSE_DECODE, multi-token/spec decode, and forwards that dump a
    newly completed full block.
    """

    ordered_req_ids = [str(req_id) for req_id in req_ids]
    row_count = len(ordered_req_ids)
    if row_count <= 0:
        return DSAGraphGateDecision(True, "empty_batch")

    stages_by_req = {
        req_id: ReqStage.coerce(req_stages.get(req_id))
        for req_id in ordered_req_ids
    }
    if _as_int(total_tokens) != row_count:
        return DSAGraphGateDecision(True, "total_tokens_mismatch")

    if _as_int(total_tokens) not in capture_sizes:
        return DSAGraphGateDecision(True, "capture_size_miss")

    if has_full_block_dump:
        return DSAGraphGateDecision(
            True,
            "decode_reaches_full_block_boundary",
            bad_req_id=full_block_dump_req_id,
        )

    for req_id in ordered_req_ids:
        if _as_int(scheduled_tokens.get(req_id)) != 1:
            return DSAGraphGateDecision(
                True, "non_single_token_decode", bad_req_id=req_id)

        stage = stages_by_req[req_id]
        if stage == ReqStage.DENSE_DECODE:
            continue
        if stage != ReqStage.SPARSE_DECODE:
            return DSAGraphGateDecision(
                True, "non_row_mode_stage", bad_req_id=req_id)

        budget = _as_int(configured_budget)
        if budget <= 0:
            return DSAGraphGateDecision(True, "missing_configured_budget")

        if _as_int(resident_graph_limit) <= 0:
            return DSAGraphGateDecision(True, "invalid_resident_graph_limit")

        block = _as_int(block_size)
        if block <= 0:
            return DSAGraphGateDecision(True, "invalid_block_size")

        sparse_budget = _as_int(sparse_budgets.get(req_id))
        if sparse_budget != budget:
            return DSAGraphGateDecision(
                True, "sparse_budget_mismatch", bad_req_id=req_id)

        resident_len = _as_int(resident_lens.get(req_id), default=-1)
        if resident_len <= 0 or resident_len > _as_int(resident_graph_limit):
            return DSAGraphGateDecision(
                True, "resident_len_out_of_graph_limit", bad_req_id=req_id)

        context_len = _as_int(
            context_lens_before_forward.get(req_id), default=-1)
        if context_len <= 0:
            return DSAGraphGateDecision(
                True, "invalid_context_len", bad_req_id=req_id)

        ready_value = (
            None if resident_sparse_ready is None
            else resident_sparse_ready.get(req_id)
        )
        is_resident_ready = (
            resident_len >= sparse_budget >= budget
            if ready_value is None else bool(ready_value)
        )
        if not is_resident_ready:
            return DSAGraphGateDecision(
                True, "resident_budget_below_required", bad_req_id=req_id)

    return DSAGraphGateDecision(
        disabled=False,
        reason="allow_row_mode_decode",
        graph_phase=DSA_GRAPH_PHASE_ROW_MODE_DECODE,
        row_count=row_count,
    )
