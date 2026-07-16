"""DSA 请求级元数据与阶段计划。

本文件定义单个请求在 DSA 稀疏卸载中的轻量元数据视图，包括完整序列长度、
query 位置、HBM sparse budget、resident tail、满块 dump 判定，以及
dense/sparse decode 阶段计划。这里的数据由 scheduler/input-batch 状态
每轮 forward 重新物化，不直接持有 layer 级缓存张量；layer 级 cache zone
解析和稳定性校验放在 dsa_layer_cache_zones.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import torch

from vllm_ascend.dsa_sparse.dsa_types import ReqStage

ReqType = Union[str, int]
QueryPositionRow = list[int] | torch.Tensor


def _resolve_budget_slot_count(
    block_ids: list[int],
    block_size: int,
    max_slots: int | None = None,
) -> int:
    if not block_ids:
        return 0
    block_size = int(block_size)
    total_slots = len(block_ids) * block_size
    if max_slots is None:
        slot_count = total_slots
    else:
        slot_count = max(0, min(int(max_slots), total_slots))
    return max(0, int(slot_count))


@dataclass(frozen=True)
class ReqForwardPlan:
    is_sparse_decode: bool
    budget_slot_count: int
    sparse_budget_tokens: int
    dense_tail_start: int
    tail_valid_token_count: int
    resident_tail_start: int
    resident_tail_end: int
    candidate_range_start: int
    candidate_range_end: int


def _build_req_forward_plan(
    *,
    block_size: int,
    num_prompt_tokens: int,
    num_output_tokens: int,
    resident_valid_seq_len: int,
    req_context_full_blk_hashes: list,
    vllm_budget_block_ids: list[int],
    dsa_sparse_enabled: bool,
    dsa_sparse_budget_tokens: int,
) -> ReqForwardPlan:
    block_size = int(block_size)
    sparse_budget_tokens = max(0, int(dsa_sparse_budget_tokens or 0))
    is_sparse_decode = (
        int(num_output_tokens) > 0
        and int(resident_valid_seq_len) >= 0
        and bool(dsa_sparse_enabled)
        and sparse_budget_tokens > 0
    )
    max_slots = sparse_budget_tokens if bool(dsa_sparse_enabled) else None
    budget_slot_count = _resolve_budget_slot_count(
        vllm_budget_block_ids,
        block_size,
        max_slots=max_slots,
    )

    dense_tokens_before_current_query = (
        int(num_prompt_tokens) + max(int(num_output_tokens) - 1, 0)
    )
    dense_tail_start = (
        dense_tokens_before_current_query // block_size) * block_size

    if is_sparse_decode:
        # Sparse lookup slots occupy every full resident block before the
        # independent dense tail block. The current TopK width is deliberately
        # smaller than this address space and must not define the tail offset.
        resident_tail_start = max(
            0, (len(vllm_budget_block_ids) - 1) * block_size)
        dumped_full_token_end = len(req_context_full_blk_hashes) * block_size
        candidate_range_start = 0
        candidate_range_end = min(dumped_full_token_end, dense_tail_start)
    else:
        total_blocks = len(vllm_budget_block_ids)
        resident_tail_start = max(0, (total_blocks - 1) * block_size)
        candidate_range_start = 0
        candidate_range_end = dense_tail_start

    tail_valid_token_count = 0
    if int(resident_valid_seq_len) >= 0:
        tail_slots = int(resident_valid_seq_len) - resident_tail_start
        if 0 < tail_slots <= block_size:
            tail_valid_token_count = tail_slots
    if tail_valid_token_count <= 0:
        tail_valid_token_count = max(
            0, min(block_size,
                   dense_tokens_before_current_query - dense_tail_start))

    return ReqForwardPlan(
        is_sparse_decode=is_sparse_decode,
        budget_slot_count=budget_slot_count,
        sparse_budget_tokens=sparse_budget_tokens,
        dense_tail_start=dense_tail_start,
        tail_valid_token_count=tail_valid_token_count,
        resident_tail_start=resident_tail_start,
        resident_tail_end=resident_tail_start + tail_valid_token_count,
        candidate_range_start=candidate_range_start,
        candidate_range_end=candidate_range_end,
    )


def _validate_query_position_lengths(
    *,
    request_id: ReqType,
    query_len: int,
    dense_query_positions: QueryPositionRow,
    resident_query_positions: QueryPositionRow,
) -> None:
    dense_query_len = (
        int(dense_query_positions.numel())
        if torch.is_tensor(dense_query_positions)
        else len(dense_query_positions)
    )
    resident_query_len = (
        int(resident_query_positions.numel())
        if torch.is_tensor(resident_query_positions)
        else len(resident_query_positions)
    )
    if dense_query_len and dense_query_len != int(query_len):
        raise RuntimeError(
            f"DSA metadata got mismatched dense_query_positions for req "
            f"{request_id}: {dense_query_len} vs query_len {int(query_len)}")
    if resident_query_len and resident_query_len != int(query_len):
        raise RuntimeError(
            f"DSA metadata got mismatched resident_query_positions for req "
            f"{request_id}: {resident_query_len} vs query_len {int(query_len)}")


@dataclass
class ReqMeta:
    """Per-request row materialized for one model forward.

    ReqMeta is intentionally not a request-lifetime state object. It is rebuilt
    from scheduler/input-batch metadata every model forward, then folded into
    tensorized forward batches in dsa_sparse.py. Long-lived mutable resources
    live elsewhere:
    - HBM sparse resident token rows: DSAResidentTokenPool.
    - Backend-owned KV storage and I/O: DSAKVBackend.
    - Per-layer cache tensor references: dsa_layer_cache_zones.

    The fields below are therefore source metadata for batch construction, not
    layer-wise runtime state. Keeping that boundary clear prevents the old
    request/layer behavior object from creeping back in.
    """

    request_id: ReqType
    index_in_batch: int
    num_prompt_tokens: int
    num_output_tokens: int
    num_scheduled_tokens: int
    num_computed_tokens: int
    # Valid MLA/SFA token count visible in resident HBM; it excludes unused
    # capacity in the allocated resident tail block.
    resident_valid_seq_len: int
    vllm_budget_block_ids: list[int]
    block_size: int
    query_start_loc: int
    query_len: int
    req_context_full_blk_hashes: list
    stage: ReqStage
    dense_query_positions: QueryPositionRow = field(default_factory=list)
    resident_query_positions: QueryPositionRow = field(default_factory=list)
    dsa_sparse_enabled: bool = False
    dsa_sparse_budget_tokens: int = 0
    resident_pool_idx: int = -1
    forward_plan: ReqForwardPlan = field(init=False)

    def __post_init__(self) -> None:
        _validate_query_position_lengths(
            request_id=self.request_id,
            query_len=self.query_len,
            dense_query_positions=self.dense_query_positions,
            resident_query_positions=self.resident_query_positions,
        )
        self.forward_plan = _build_req_forward_plan(
            block_size=self.block_size,
            num_prompt_tokens=self.num_prompt_tokens,
            num_output_tokens=self.num_output_tokens,
            resident_valid_seq_len=self.resident_valid_seq_len,
            req_context_full_blk_hashes=self.req_context_full_blk_hashes,
            vllm_budget_block_ids=self.vllm_budget_block_ids,
            dsa_sparse_enabled=self.dsa_sparse_enabled,
            dsa_sparse_budget_tokens=self.dsa_sparse_budget_tokens,
        )

    @property
    def is_full_block_need_dump_in_decode(self) -> bool:
        assert self.num_output_tokens > 0
        return (self.num_prompt_tokens + self.num_output_tokens) % self.block_size == 0

    @property
    def is_last_prefill_chunk(self):
        return (
                self.num_computed_tokens + self.num_scheduled_tokens
                >= self.num_prompt_tokens and self.num_output_tokens == 0
        )


@dataclass(frozen=True)
class ReqSparseDecodeForwardPlan:
    query_start_loc: int
    query_len: int
    range_start: int
    range_end: int
    budget_slot_count: int

    @classmethod
    def from_req_meta(
            cls,
            req_meta: "ReqMeta") -> "ReqSparseDecodeForwardPlan":
        plan = req_meta.forward_plan
        return cls(
            query_start_loc=int(req_meta.query_start_loc),
            query_len=max(int(req_meta.query_len), 0),
            range_start=int(plan.candidate_range_start),
            range_end=int(plan.candidate_range_end),
            budget_slot_count=int(plan.budget_slot_count),
        )

    def has_valid_topk_window(self) -> bool:
        return (
            self.query_len > 0
            and self.range_start < self.range_end
            and self.budget_slot_count > 0
        )
