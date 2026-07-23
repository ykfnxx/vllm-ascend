"""DSA model forward batch 的数据结构与构造逻辑。

本文件承接 dsa_sparse.py 中“把一轮 model forward 的 ReqMeta 列表整理成
layer hook / lookup-resident / SFA 可消费批量张量”的职责。它持有的是
短生命周期 forward-level batch，而不是请求生命周期状态；请求状态仍由
dsa_sparse.py 和 dsa_req_meta.py 推进，HBM resident 资源由 resident pool
管理，后端 KV I/O 由 worker 级 DSAKVBackend 执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_batch_tensor_utils import (
    build_hbm_block_table_tensor, build_int_tensor, build_padded_int_tensor,
    compute_sparse_attention_indices_width, sort_decode_rows_by_batch_index)
from vllm_ascend.dsa_sparse.dsa_layer_cache_zones import LayerCacheZones
from vllm_ascend.dsa_sparse.dsa_req_meta import (
    QueryPositionRow, ReqMeta, ReqSparseDecodeForwardPlan, ReqType)
from vllm_ascend.dsa_sparse.dsa_resident_pool import (
    DSAResidentLayerResourceView)
from vllm_ascend.dsa_sparse.dsa_types import DSADecodeRowMode, ReqStage

logger = init_logger("vllm.dsa_sparse")


@dataclass
class DSAModelForwardMeta:
    """One-model-forward request row container.

    This object is deliberately short-lived: build_dsa_meta() recreates it for
    every model forward from scheduler/input-batch state. It should not grow
    request-lifetime or layer-lifetime state; persistent resources belong to
    DSAResidentTokenPool, and layer-specific views are derived later from
    DSAForwardSparseDecodeBatch or DSAForwardLayerBatch.
    """

    requests: List[ReqMeta]
    full_block_table_tensor: torch.Tensor | None

    def __init__(self):
        self.requests = []
        self.full_block_table_tensor = None

    def add_request_meta(
            self,
            request_id: ReqType,
            index_in_batch: int,
            num_prompt_tokens: int,
            num_output_tokens: int,
            num_scheduled_tokens: int,
            num_computed_tokens: int,
            resident_valid_seq_len: int,
            vllm_budget_block_ids: list[int],
            indexer_block_ids: list[int],
            block_size,
            query_start_loc: int,
            query_len: int,
            req_context_full_blk_hashes: list = None,
            dense_query_positions: QueryPositionRow | None = None,
            resident_query_positions: QueryPositionRow | None = None,
            stage: ReqStage = ReqStage.PREFILL,
            dsa_sparse_enabled: bool = False,
            dsa_sparse_budget_tokens: int = 0,
            resident_pool_idx: int = -1,
            pd_remote_loaded: bool = False,
    ):
        req_meta = ReqMeta(request_id=request_id,
                           index_in_batch=index_in_batch,
                           num_prompt_tokens=num_prompt_tokens,
                           num_output_tokens=num_output_tokens,
                           num_scheduled_tokens=num_scheduled_tokens,
                           num_computed_tokens=num_computed_tokens,
                           resident_valid_seq_len=resident_valid_seq_len,
                           vllm_budget_block_ids=vllm_budget_block_ids,
                           indexer_block_ids=indexer_block_ids,
                           block_size=block_size,
                           query_start_loc=query_start_loc,
                           query_len=query_len,
                           req_context_full_blk_hashes=req_context_full_blk_hashes,
                           stage=stage,
                           dense_query_positions=(
                               [] if dense_query_positions is None
                               else dense_query_positions),
                           resident_query_positions=(
                               [] if resident_query_positions is None
                               else resident_query_positions),
                           dsa_sparse_enabled=dsa_sparse_enabled,
                           dsa_sparse_budget_tokens=dsa_sparse_budget_tokens,
                           resident_pool_idx=resident_pool_idx,
                           pd_remote_loaded=pd_remote_loaded)
        self.requests.append(req_meta)


@dataclass(frozen=True)
class DSAForwardSparseDecodeBatch:
    """Model-forward-level sparse decode metadata.

    Built once per model forward and reused by every layer's after_indexer
    hook. It contains the tensorized, layer-invariant inputs needed by the
    lookup-resident path, such as resident pool rows, query ranges, HBM block
    tables, and lookup row metadata.

    row-mode decode 刻意拆开两组“行”语义：

    * active_*row_indices_tensor:
      本次 model forward 中所有进入统一 lightning-indexer -> lookup
      -> SFA 路径的 decode 行。这里既包含 DENSE 行，也包含
      SPARSE 行；row_modes_tensor 决定每行使用 dense 还是 sparse
      语义，所以主计算路径必须按 active rows 覆盖整批 decode 行。
    * sparse_*row_indices_tensor:
      active rows 里真正处于 sparse resident 语义的子集。它适合做诊断、
      测试、图 replay 保真，以及后续 sparse-only 账本统计；但不能再用
      它筛掉 dense 行，否则 mixed dense/sparse row-mode batch 会被拆坏。

    local_* row ids 是 DSAForwardSparseDecodeBatch 内部小表的行号，用于
    索引 DSA 自己构造的 tensor，例如 candidate_lens_tensor。
    batch_* row ids 是原始 model/attention batch 大表里的行号，用于索引
    q_li、weights、full-batch lightning-indexer topK 等 attention tensor。

    形象例子：如果原始 batch 是 [prefill_req, dense_req, sparse_req]，
    prefill 行不进入 row-mode decode 小表，那么 DSA batch 只剩
    [dense_req, sparse_req]。此时 active_local=[0, 1]，但
    active_batch=[1, 2]；拿 active_local 去索引 q_li 会错取 prefill 行。
    """

    max_logical_blocks: int
    score_topk_k: int
    resident_pool_indices_tensor: torch.Tensor
    query_position_rows_tensor: torch.Tensor
    tail_valid_token_counts_tensor: torch.Tensor
    dense_tail_starts_tensor: torch.Tensor
    resident_tail_starts_tensor: torch.Tensor
    query_start_locs_tensor: torch.Tensor
    query_lens_tensor: torch.Tensor
    query_last_token_indices_tensor: torch.Tensor
    range_starts_tensor: torch.Tensor
    range_ends_tensor: torch.Tensor
    candidate_lens_tensor: torch.Tensor
    budget_lengths_tensor: torch.Tensor
    batch_hbm_block_table: torch.Tensor
    attention_indices_width: int
    query_last_token_indices_are_identity: bool
    batch_row_indices_tensor: torch.Tensor
    row_modes_tensor: torch.Tensor
    lookup_init_mask_tensor: torch.Tensor
    has_lookup_init_rows: bool
    # active_* 是 row-mode kernel 路径处理的全部 decode 行；sparse_* 仍是
    # 真 sparse 子集。两者都保留，避免图 replay 或后续重构时把“sparse 行”
    # 误扩展成“全部行”，也避免把 dense 行从统一 lookup/SFA 路径里漏掉。
    active_local_row_indices_tensor: torch.Tensor
    active_batch_row_indices_tensor: torch.Tensor
    sparse_row_mask_tensor: torch.Tensor
    sparse_local_row_indices_tensor: torch.Tensor
    sparse_batch_row_indices_tensor: torch.Tensor
    request_ids: list[ReqType]
    batch_row_indices: list[int]

    @classmethod
    def empty(
        cls,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> "DSAForwardSparseDecodeBatch":
        device = torch.device("cpu") if tensor_device is None else torch.device(
            tensor_device)
        empty_i32_table = torch.empty((0, 0), dtype=torch.int32, device=device)
        return cls(
            max_logical_blocks=0,
            score_topk_k=0,
            resident_pool_indices_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            query_position_rows_tensor=torch.empty(
                (0, 0), dtype=torch.int32, device=device),
            tail_valid_token_counts_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            dense_tail_starts_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            resident_tail_starts_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            query_start_locs_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            query_lens_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            query_last_token_indices_tensor=torch.empty(
                (0,), dtype=torch.long, device=device),
            range_starts_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            range_ends_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            candidate_lens_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            budget_lengths_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            batch_hbm_block_table=empty_i32_table,
            attention_indices_width=0,
            query_last_token_indices_are_identity=False,
            batch_row_indices_tensor=torch.empty(
                (0,), dtype=torch.long, device=device),
            row_modes_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            lookup_init_mask_tensor=torch.empty(
                (0,), dtype=torch.bool, device=device),
            has_lookup_init_rows=False,
            active_local_row_indices_tensor=torch.empty(
                (0,), dtype=torch.long, device=device),
            active_batch_row_indices_tensor=torch.empty(
                (0,), dtype=torch.long, device=device),
            sparse_row_mask_tensor=torch.empty(
                (0,), dtype=torch.bool, device=device),
            sparse_local_row_indices_tensor=torch.empty(
                (0,), dtype=torch.long, device=device),
            sparse_batch_row_indices_tensor=torch.empty(
                (0,), dtype=torch.long, device=device),
            request_ids=[],
            batch_row_indices=[],
        )

    def __bool__(self) -> bool:
        return int(self.resident_pool_indices_tensor.numel()) > 0


@dataclass(frozen=True)
class DSALayerRuntimeBatch:
    """Layer-level view for attention_begin/attention_finished.

    This is the current layer's runtime view of DSAForwardLayerBatch: it adds
    the concrete layer id and cache zones needed by begin/finish hooks. It does
    not own long-lived mutable state; it only packages dependencies for the
    current layer invocation.
    """

    layer_id: int
    sparse_decode_guard_request_ids: list[ReqType]
    sparse_decode_guard_pool_indices_tensor: torch.Tensor
    full_block_dump_tables: DSAFullBlockDumpTables
    prefill_done_pool_indices_tensor: torch.Tensor
    cache_zones: LayerCacheZones | None = None

    def __bool__(self) -> bool:
        return (
            int(self.sparse_decode_guard_pool_indices_tensor.numel()) > 0
            or bool(self.full_block_dump_tables)
            or int(self.prefill_done_pool_indices_tensor.numel()) > 0
        )


@dataclass(frozen=True)
class DSAForwardLayerBatch:
    """Model-forward-level plan for attention_begin/attention_finished.

    Built once per model forward to prepare layer hook work that is common
    across layers: sparse-decode dump-ready guards and batched full-block dump
    tables. Each layer turns this into a DSALayerRuntimeBatch when it binds its
    concrete layer id and cache zones.
    """

    sparse_decode_guard_request_ids: list[ReqType]
    sparse_decode_guard_pool_indices_tensor: torch.Tensor
    full_block_dump_tables: DSAFullBlockDumpTables
    prefill_done_pool_indices_tensor: torch.Tensor

    @classmethod
    def empty(
        cls,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> "DSAForwardLayerBatch":
        # These guard tensors index a CPU-side readiness table, so they remain
        # on CPU even when sparse decode resident resources live on NPU.
        return cls(
            sparse_decode_guard_request_ids=[],
            sparse_decode_guard_pool_indices_tensor=torch.empty(
                (0,), dtype=torch.long, device=torch.device("cpu")),
            full_block_dump_tables=DSAFullBlockDumpTables.empty(),
            prefill_done_pool_indices_tensor=torch.empty(
                (0,), dtype=torch.long, device=torch.device("cpu")),
        )

    def __bool__(self) -> bool:
        return (
            int(self.sparse_decode_guard_pool_indices_tensor.numel()) > 0
            or bool(self.full_block_dump_tables)
            or int(self.prefill_done_pool_indices_tensor.numel()) > 0
        )

    def layer_runtime_batch(
        self,
        layer_id: int,
        cache_zones: LayerCacheZones | None = None,
    ) -> DSALayerRuntimeBatch:
        return DSALayerRuntimeBatch(
            layer_id=int(layer_id),
            sparse_decode_guard_request_ids=(
                self.sparse_decode_guard_request_ids),
            sparse_decode_guard_pool_indices_tensor=(
                self.sparse_decode_guard_pool_indices_tensor),
            full_block_dump_tables=self.full_block_dump_tables,
            prefill_done_pool_indices_tensor=(
                self.prefill_done_pool_indices_tensor),
            cache_zones=cache_zones,
        )


@dataclass(frozen=True)
class DSAFullBlockDumpTables:
    """Forward-level rows consumed by the KV backend block put path.

    This table only carries put row metadata. Prefill put readiness is a
    separate tensor in DSAForwardLayerBatch because attention_finished only
    needs resident pool rows to mark readiness; it does not need hashes or
    logical block ids.
    """

    request_ids: list[ReqType]
    request_pool_indices: list[int]
    block_hash_rows: list[list]
    block_id_rows: list[list[int]]
    indexer_block_id_rows: list[list[int]]
    logical_block_index_rows: list[list[int]]
    valid_token_count_rows: list[int]

    def __bool__(self) -> bool:
        return bool(self.request_ids)

    @classmethod
    def empty(cls) -> "DSAFullBlockDumpTables":
        return cls(
            request_ids=[],
            request_pool_indices=[],
            block_hash_rows=[],
            block_id_rows=[],
            indexer_block_id_rows=[],
            logical_block_index_rows=[],
            valid_token_count_rows=[],
        )


def _build_forward_batches_from_dsa_meta(
    dsa_meta: DSAModelForwardMeta | None,
    *,
    tensor_device: torch.device | str | None = None,
    force_decode_row_mode_score_topk: int = 0,
) -> tuple[DSAForwardSparseDecodeBatch, DSAForwardLayerBatch]:
    """Build all model-forward DSA batches with one ReqMeta pass."""
    if dsa_meta is None:
        return (
            DSAForwardSparseDecodeBatch.empty(tensor_device=tensor_device),
            DSAForwardLayerBatch.empty(tensor_device=tensor_device),
        )

    device = torch.device("cpu") if tensor_device is None else torch.device(
        tensor_device)
    resident_pool_indices: list[int] = []
    query_start_locs: list[int] = []
    query_lens: list[int] = []
    query_last_token_indices: list[int] = []
    query_positions: list[QueryPositionRow] = []
    request_ids: list[ReqType] = []
    batch_row_indices: list[int] = []
    sparse_row_mask: list[bool] = []
    row_modes: list[int] = []
    lookup_init_mask: list[bool] = []
    tail_valid_token_counts: list[int] = []
    dense_tail_starts: list[int] = []
    resident_tail_starts: list[int] = []
    range_starts: list[int] = []
    range_ends: list[int] = []
    candidate_lens: list[int] = []
    budget_row_indices: list[int] = []
    budget_lengths: list[int] = []
    hbm_slot_counts: list[int] = []
    attention_indices_width = 0
    max_logical_blocks = 0
    budget_block_size = 0
    score_topk_k = 0

    sparse_decode_guard_request_ids: list[ReqType] = []
    sparse_decode_guard_pool_indices: list[int] = []
    dump_request_ids: list[ReqType] = []
    dump_request_pool_indices: list[int] = []
    dump_block_hash_rows: list[list] = []
    dump_block_id_rows: list[list[int]] = []
    dump_indexer_block_id_rows: list[list[int]] = []
    dump_logical_block_index_rows: list[list[int]] = []
    dump_valid_token_count_rows: list[int] = []
    prefill_done_pool_indices: list[int] = []

    for req_meta in dsa_meta.requests:
        resident_pool_idx = int(req_meta.resident_pool_idx)

        if req_meta.forward_plan.is_sparse_decode:
            sparse_decode_guard_request_ids.append(req_meta.request_id)
            sparse_decode_guard_pool_indices.append(resident_pool_idx)

        block_hashes: list = []
        block_ids: list[int] = []
        indexer_block_ids: list[int] = []
        logical_block_indices: list[int] = []
        valid_token_count = 0
        mark_prefill_done = False
        if (
            req_meta.num_output_tokens == 0
            and req_meta.is_last_prefill_chunk
            and not req_meta.pd_remote_loaded
        ):
            full_hashes = req_meta.req_context_full_blk_hashes
            valid_token_count = int(req_meta.num_prompt_tokens)
            num_blocks = (
                valid_token_count + req_meta.block_size - 1
            ) // req_meta.block_size
            if num_blocks > 0:
                # Prefix-cache hashes only exist for complete blocks. KVIO
                # also persists the final partial block, so keep all row-wise
                # metadata aligned and use ``None`` for its non-cacheable key.
                block_hashes = list(full_hashes[:num_blocks]) + [None] * max(
                    0, num_blocks - len(full_hashes))
                block_ids = [
                    int(block_id)
                    for block_id in req_meta.vllm_budget_block_ids[
                        :num_blocks]
                ]
                indexer_block_ids = [
                    int(block_id)
                    for block_id in req_meta.indexer_block_ids[:num_blocks]
                ]
                logical_block_indices = list(range(num_blocks))
            mark_prefill_done = True
        elif ((req_meta.num_output_tokens > 0 or req_meta.pd_remote_loaded)
              and req_meta.is_full_block_need_dump_in_decode
              and req_meta.req_context_full_blk_hashes):
            logical_block_idx = len(req_meta.req_context_full_blk_hashes) - 1
            block_hashes = [req_meta.req_context_full_blk_hashes[-1]]
            block_ids = [int(req_meta.vllm_budget_block_ids[-1])]
            indexer_block_ids = [int(req_meta.indexer_block_ids[-1])]
            logical_block_indices = [logical_block_idx]
            valid_token_count = (
                logical_block_idx + 1) * req_meta.block_size
            logger.debug(
                "========== DSA DECODE FULL BLOCK DUMP =========="
                " req_id=%s prompt_tokens=%s output_tokens=%s "
                "computed_tokens=%s scheduled_tokens=%s logical_block=%s "
                "hbm_block_id=%s block_size=%s",
                req_meta.request_id,
                req_meta.num_prompt_tokens,
                req_meta.num_output_tokens,
                req_meta.num_computed_tokens,
                req_meta.num_scheduled_tokens,
                logical_block_idx,
                block_ids[0],
                req_meta.block_size,
            )

        if mark_prefill_done:
            prefill_done_pool_indices.append(resident_pool_idx)
        if block_ids:
            dump_request_ids.append(req_meta.request_id)
            dump_request_pool_indices.append(resident_pool_idx)
            dump_block_hash_rows.append(block_hashes)
            dump_block_id_rows.append(block_ids)
            dump_indexer_block_id_rows.append(indexer_block_ids)
            dump_logical_block_index_rows.append(logical_block_indices)
            dump_valid_token_count_rows.append(valid_token_count)

        if req_meta.num_output_tokens <= 0 and not req_meta.pd_remote_loaded:
            continue
        topk_plan = ReqSparseDecodeForwardPlan.from_req_meta(req_meta)
        if resident_pool_idx < 0 or topk_plan.query_len <= 0:
            continue
        is_sparse_row = bool(req_meta.forward_plan.is_sparse_decode)
        if is_sparse_row and not topk_plan.has_valid_topk_window():
            continue

        query_start = int(topk_plan.query_start_loc)
        query_len = int(topk_plan.query_len)
        query_position_row = (
            getattr(req_meta, "resident_query_positions", [])
            if is_sparse_row
            else getattr(req_meta, "dense_query_positions", []))
        tail_valid_count = int(
            req_meta.forward_plan.tail_valid_token_count)
        budget_slot_count = int(topk_plan.budget_slot_count)
        budget_block_size = int(req_meta.block_size)
        range_end = int(topk_plan.range_end)
        batch_row_index = int(req_meta.index_in_batch)
        if batch_row_index < 0 or batch_row_index >= len(dsa_meta.requests):
            raise RuntimeError(
                "DSA sparse decode batch row is out of range: "
                f"req_id={req_meta.request_id}, row={batch_row_index}, "
                f"num_reqs={len(dsa_meta.requests)}")

        request_ids.append(req_meta.request_id)
        batch_row_indices.append(batch_row_index)
        sparse_row_mask.append(is_sparse_row)
        row_modes.append(int(DSADecodeRowMode.SPARSE if is_sparse_row
                             else DSADecodeRowMode.DENSE))
        lookup_init_mask.append(
            is_sparse_row
            and req_meta.stage.is_enter_sparse_decode
            and not req_meta.pd_remote_loaded)
        resident_pool_indices.append(resident_pool_idx)
        query_start_locs.append(query_start)
        query_lens.append(query_len)
        query_last_token_indices.append(
            query_start + max(query_len, 1) - 1)
        query_positions.append(query_position_row)
        tail_valid_token_counts.append(tail_valid_count)
        dense_tail_starts.append(
            int(req_meta.forward_plan.dense_tail_start))
        resident_tail_starts.append(
            int(req_meta.forward_plan.resident_tail_start))
        range_starts.append(int(topk_plan.range_start))
        range_ends.append(range_end)
        candidate_lens.append(max(0, range_end - int(topk_plan.range_start)))
        budget_row_indices.append(batch_row_index)
        budget_lengths.append(budget_slot_count)
        hbm_slot_counts.append(
            int(req_meta.forward_plan.resident_tail_start)
            if is_sparse_row else 0)
        if is_sparse_row:
            # Sparse decode only admits single-token rows whose current query
            # KV is in the independent resident tail. Indexer still selects its
            # TopK over the original full sequence; every selected token is then
            # mapped either through lookup or directly into the live tail.
            attention_indices_width = max(
                attention_indices_width,
                compute_sparse_attention_indices_width(
                    budget_slot_count=budget_slot_count,
                ))
        if is_sparse_row:
            max_logical_blocks = max(
                max_logical_blocks,
                (max(0, range_end) + req_meta.block_size - 1)
                // req_meta.block_size)
            score_topk_k = max(
                score_topk_k,
                int(req_meta.dsa_sparse_budget_tokens))

    # DSA decode graph capture records the row-mode lookup/lightning branch.
    # Pure dense replay therefore still needs a positive topk so the Python
    # control flow reaches the same operator sequence; row_modes tells lookup to
    # leave dense rows as native full-cache rows.
    if request_ids and int(force_decode_row_mode_score_topk) > 0:
        score_topk_k = max(score_topk_k,
                           int(force_decode_row_mode_score_topk))
        attention_indices_width = max(
            attention_indices_width,
            int(force_decode_row_mode_score_topk),
        )

    (
        batch_row_indices,
        request_ids,
        sparse_row_mask,
        row_modes,
        lookup_init_mask,
        resident_pool_indices,
        query_start_locs,
        query_lens,
        query_last_token_indices,
        query_positions,
        tail_valid_token_counts,
        dense_tail_starts,
        resident_tail_starts,
        range_starts,
        range_ends,
        candidate_lens,
        budget_row_indices,
        budget_lengths,
        hbm_slot_counts,
    ) = sort_decode_rows_by_batch_index(
        batch_row_indices,
        request_ids,
        sparse_row_mask,
        row_modes,
        lookup_init_mask,
        resident_pool_indices,
        query_start_locs,
        query_lens,
        query_last_token_indices,
        query_positions,
        tail_valid_token_counts,
        dense_tail_starts,
        resident_tail_starts,
        range_starts,
        range_ends,
        candidate_lens,
        budget_row_indices,
        budget_lengths,
        hbm_slot_counts,
    )

    sparse_local_row_indices = [
        row for row, is_sparse in enumerate(sparse_row_mask) if is_sparse
    ]
    # local 行号描述 DSA forward batch 这张小表，batch 行号描述原始
    # model/attention batch。当前纯 decode 且所有行都有效时二者通常相同；
    # 但 P/D mixed、无效 topK window 行被跳过、或未来请求重组时就会分叉。
    # 例如原始 batch=[prefill_req, dense_req, sparse_req]，prefill 被跳过后：
    # active_local=[0, 1]，active_batch=[1, 2]，sparse_local=[1]，
    # sparse_batch=[2]。
    active_local_row_indices = list(range(len(request_ids)))
    active_batch_rows = list(batch_row_indices)
    true_sparse_batch_rows = [
        batch_row_indices[row] for row in sparse_local_row_indices
    ]
    resident_pool_indices_tensor = build_int_tensor(
        resident_pool_indices, dtype=torch.int32, device=device)
    batch_hbm_block_table = build_hbm_block_table_tensor(
        dsa_meta.full_block_table_tensor,
        budget_row_indices,
        hbm_slot_counts,
        block_size=budget_block_size,
        dtype=torch.int32,
        device=device,
        pad_value=0,
    )

    sparse_batch = DSAForwardSparseDecodeBatch(
        max_logical_blocks=max_logical_blocks,
        score_topk_k=score_topk_k,
        resident_pool_indices_tensor=resident_pool_indices_tensor,
        query_position_rows_tensor=build_padded_int_tensor(
            query_positions,
            dtype=torch.int32,
            device=device,
            pad_value=-1,
        ),
        tail_valid_token_counts_tensor=build_int_tensor(
            tail_valid_token_counts, dtype=torch.int32, device=device),
        dense_tail_starts_tensor=build_int_tensor(
            dense_tail_starts, dtype=torch.int32, device=device),
        resident_tail_starts_tensor=build_int_tensor(
            resident_tail_starts, dtype=torch.int32, device=device),
        query_start_locs_tensor=build_int_tensor(
            query_start_locs, dtype=torch.int32, device=device),
        query_lens_tensor=build_int_tensor(
            query_lens, dtype=torch.int32, device=device),
        query_last_token_indices_tensor=build_int_tensor(
            query_last_token_indices, dtype=torch.long, device=device),
        range_starts_tensor=build_int_tensor(
            range_starts, dtype=torch.int32, device=device),
        range_ends_tensor=build_int_tensor(
            range_ends, dtype=torch.int32, device=device),
        candidate_lens_tensor=build_int_tensor(
            candidate_lens, dtype=torch.int32, device=device),
        budget_lengths_tensor=build_int_tensor(
            budget_lengths, dtype=torch.int32, device=device),
        batch_hbm_block_table=batch_hbm_block_table,
        attention_indices_width=attention_indices_width,
        query_last_token_indices_are_identity=(
            query_last_token_indices == batch_row_indices),
        batch_row_indices_tensor=build_int_tensor(
            batch_row_indices, dtype=torch.long, device=device),
        row_modes_tensor=build_int_tensor(
            row_modes, dtype=torch.int32, device=device),
        lookup_init_mask_tensor=build_int_tensor(
            lookup_init_mask, dtype=torch.bool, device=device),
        has_lookup_init_rows=any(lookup_init_mask),
        active_local_row_indices_tensor=build_int_tensor(
            active_local_row_indices, dtype=torch.long, device=device),
        active_batch_row_indices_tensor=build_int_tensor(
            active_batch_rows, dtype=torch.long, device=device),
        sparse_row_mask_tensor=build_int_tensor(
            sparse_row_mask, dtype=torch.bool, device=device),
        sparse_local_row_indices_tensor=build_int_tensor(
            sparse_local_row_indices, dtype=torch.long, device=device),
        sparse_batch_row_indices_tensor=build_int_tensor(
            true_sparse_batch_rows, dtype=torch.long, device=device),
        request_ids=request_ids,
        batch_row_indices=batch_row_indices,
    )
    layer_batch = DSAForwardLayerBatch(
        sparse_decode_guard_request_ids=sparse_decode_guard_request_ids,
        sparse_decode_guard_pool_indices_tensor=build_int_tensor(
            sparse_decode_guard_pool_indices,
            dtype=torch.long,
            device=torch.device("cpu"),
        ),
        full_block_dump_tables=DSAFullBlockDumpTables(
            request_ids=dump_request_ids,
            request_pool_indices=dump_request_pool_indices,
            block_hash_rows=dump_block_hash_rows,
            block_id_rows=dump_block_id_rows,
            indexer_block_id_rows=dump_indexer_block_id_rows,
            logical_block_index_rows=dump_logical_block_index_rows,
            valid_token_count_rows=dump_valid_token_count_rows,
        ),
        prefill_done_pool_indices_tensor=build_int_tensor(
            prefill_done_pool_indices,
            dtype=torch.long,
            device=torch.device("cpu"),
        ),
    )
    return sparse_batch, layer_batch


@dataclass(frozen=True)
class DSALayerSparseDecodeBatch:
    """Layer-level sparse decode view for after_indexer.

    This combines DSAForwardSparseDecodeBatch with the layer-specific resident
    view. Full-batch query/range/block-table tensors stay on
    DSAForwardSparseDecodeBatch and are passed to lookup-resident from there;
    keeping them out of this layer view avoids repeated layer-wise index_select
    work in the decode hot path.
    """

    layer_id: int
    resident_pool_indices_tensor: torch.Tensor
    budget_lengths_tensor: torch.Tensor
    resident_view: DSAResidentLayerResourceView
    attention_indices_width: int

    def __bool__(self) -> bool:
        return int(self.resident_pool_indices_tensor.numel()) > 0
