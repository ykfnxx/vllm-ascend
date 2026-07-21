# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DSA row-mode decode 图模式的持久 buffer 管理。

本文件只提供 DSAGraphBuffersMixin：它不推进请求生命周期，也不做
scheduler slot 估算，而是负责 FULL graph capture/replay 时需要稳定地址
的 DSA forward batch、layer id tensor、dummy HBM block table，以及
真实 forward 元数据拷贝到 graph-stable buffer 的过程。

DSASparseBase 仍属于算法核心基类，保留在 dsa_sparse.py 中。后续如果
继续适配带新满块 dump 的图模式，优先在这里扩展 graph buffer 形态，
避免把图模式细节重新堆回 hook 主流程。
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_forward_batch import (
    DSAForwardLayerBatch, DSAForwardSparseDecodeBatch)
from vllm_ascend.dsa_sparse.dsa_graph_gate import (
    DSA_GRAPH_PHASE_ROW_MODE_DECODE)
from vllm_ascend.dsa_sparse.dsa_req_meta import ReqType
from vllm_ascend.dsa_sparse.dsa_types import DSADecodeRowMode

logger = init_logger("vllm.dsa_sparse")


_DSA_GRAPH_DUMMY_REQ_ID_PREFIX = "__dsa_graph_dummy_req_"


def _make_dsa_graph_dummy_request_ids(row_count: int) -> list[ReqType]:
    return [
        f"{_DSA_GRAPH_DUMMY_REQ_ID_PREFIX}{row}" for row in range(row_count)
    ]


def _sync_row_mode_decode_graph_batch_row_maps(
    graph_batch: "DSAForwardSparseDecodeBatch",
    *,
    row_count: int,
    request_ids: list[ReqType] | None = None,
    batch_row_indices: list[int] | None = None,
) -> bool:
    """Update graph-stable Python row maps without replacing frozen fields."""
    row_count = int(row_count)
    if row_count <= 0:
        return False

    if request_ids is None:
        request_ids = _make_dsa_graph_dummy_request_ids(row_count)
    else:
        request_ids = list(request_ids)

    if batch_row_indices is None:
        batch_row_indices = list(range(row_count))
    else:
        batch_row_indices = [int(row) for row in batch_row_indices]

    if len(request_ids) != row_count or len(batch_row_indices) != row_count:
        return False

    graph_batch.request_ids[:] = request_ids
    graph_batch.batch_row_indices[:] = batch_row_indices
    return True


class DSAGraphBuffersMixin:
    def get_row_mode_decode_graph_dummy_seq_len(self) -> int:
        """Return a capture-time indexer key length covering replayed requests."""
        max_model_len = int(
            getattr(self._vllm_config.model_config, "max_model_len", 0) or 0)
        if max_model_len > 0:
            return max_model_len
        budget_tokens = int(self._hbm_sparse_budget_tokens or 0)
        return max(1, budget_tokens + int(self._vllm_blk_size))

    def get_row_mode_decode_graph_dummy_resident_seq_len(self) -> int:
        """Return the capture-time resident MLA cache length for SFA."""
        total_slots = int(self._lookup_total_slot_tokens or 0)
        return max(1, total_slots + int(self._vllm_blk_size))

    def _graph_max_logical_blocks(self) -> int:
        max_model_len = int(
            getattr(self._vllm_config.model_config, "max_model_len", 0) or 0)
        max_model_len = max(max_model_len,
                            self.get_row_mode_decode_graph_dummy_seq_len())
        return max(1, (max_model_len + self._vllm_blk_size - 1)
                   // self._vllm_blk_size)

    def _graph_attention_indices_width(self) -> int:
        # Lossless offload preserves the original Indexer TopK cardinality.
        return int(self._hbm_sparse_budget_tokens or 0)

    def _ensure_graph_layer_id_tensors(
        self,
        tensor_device: torch.device | str,
    ) -> torch.Tensor:
        """Return capture-safe device indices for layer-wise index_copy_.

        Building torch.tensor([layer_id], device=npu) inside graph capture
        performs a host-to-device memcpy on the captured stream.  Keep a
        worker-lifetime tensor per device instead, and make capture/replay
        setup create it before the model forward enters the graph region.
        """
        device = torch.device(tensor_device)
        key = str(device)
        cached = self._graph_layer_id_tensors.get(key)
        if cached is not None:
            return cached
        total_layers = int(getattr(self, "total_num_hidden_layers", 0) or 0)
        if total_layers <= 0:
            raise RuntimeError(
                "DSA layer id tensors require total_num_hidden_layers")
        layer_ids = torch.arange(
            total_layers, dtype=torch.long, device=device).reshape(-1, 1)
        self._graph_layer_id_tensors[key] = layer_ids
        return layer_ids

    def _get_layer_id_tensor(
        self,
        layer_id: int,
        tensor_device: torch.device | str,
    ) -> torch.Tensor:
        return self._ensure_graph_layer_id_tensors(tensor_device)[int(
            layer_id)]

    def _fill_graph_dummy_hbm_block_table(
        self,
        graph_batch: DSAForwardSparseDecodeBatch,
        full_block_table_tensor: torch.Tensor | None,
    ) -> None:
        """Fill capture-time HBM block table with legal non-zero ids."""
        row_count = int(graph_batch.batch_hbm_block_table.shape[0])
        block_count = int(graph_batch.batch_hbm_block_table.shape[1])
        device = graph_batch.batch_hbm_block_table.device
        fallback = torch.arange(
            1,
            block_count + 1,
            dtype=graph_batch.batch_hbm_block_table.dtype,
            device=device,
        ).reshape(1, -1).expand(row_count, block_count)

        graph_batch.batch_hbm_block_table.copy_(fallback)
        if not torch.is_tensor(full_block_table_tensor):
            return
        table = full_block_table_tensor.to(
            device=device,
            dtype=graph_batch.batch_hbm_block_table.dtype,
        )
        if table.ndim < 2:
            return
        rows = min(row_count, int(table.shape[0]))
        cols = min(block_count, int(table.shape[1]))
        if rows <= 0 or cols <= 0:
            return
        copied = table[:rows, :cols]
        graph_batch.batch_hbm_block_table[:rows, :cols].copy_(
            torch.where(copied > 0, copied, fallback[:rows, :cols]))

    @staticmethod
    def _copy_tensor_region(
        dst: torch.Tensor,
        src: torch.Tensor,
        *,
        fill_value: int | bool,
    ) -> None:
        rows = min(int(dst.shape[0]), int(src.shape[0]))
        if rows <= 0:
            dst.fill_(fill_value)
            return
        if dst.ndim == 1:
            dst[:rows].copy_(src[:rows].to(device=dst.device,
                                           dtype=dst.dtype))
            if rows < int(dst.shape[0]):
                dst[rows:].fill_(fill_value)
            return
        cols = min(int(dst.shape[1]), int(src.shape[1]))
        if cols <= 0:
            dst.fill_(fill_value)
            return
        dst[:rows, :cols].copy_(
            src[:rows, :cols].to(device=dst.device, dtype=dst.dtype))
        if cols < int(dst.shape[1]):
            dst[:rows, cols:].fill_(fill_value)
        if rows < int(dst.shape[0]):
            dst[rows:].fill_(fill_value)

    def _get_or_create_row_mode_decode_graph_batch(
        self,
        row_count: int,
        *,
        tensor_device: torch.device | str,
        graph_phase: str = DSA_GRAPH_PHASE_ROW_MODE_DECODE,
    ) -> DSAForwardSparseDecodeBatch:
        graph_phase = str(graph_phase)
        row_count = int(row_count)
        if row_count <= 0:
            raise RuntimeError(
                "DSA decode graph requires a positive row count")
        max_reqs = int(self.resident_token_pool.max_reqs)
        if row_count > max_reqs:
            raise RuntimeError(
                "DSA decode graph row count exceeds resident pool "
                f"capacity: row_count={row_count}, max_reqs={max_reqs}")
        key = (str(graph_phase), int(row_count))
        cached = self._graph_row_mode_decode_batches.get(key)
        if cached is not None:
            return cached

        device = torch.device(tensor_device)
        budget_tokens = int(self._hbm_sparse_budget_tokens or 0)
        if budget_tokens <= 0:
            raise RuntimeError(
                "DSA decode graph requires a positive sparse budget")
        block_size = int(self._vllm_blk_size)
        resident_graph_limit = self.get_row_mode_decode_graph_dummy_resident_seq_len()
        budget_blocks = max(
            1, (resident_graph_limit + block_size - 1) // block_size)
        max_logical_blocks = self._graph_max_logical_blocks()
        graph_batch_hbm_block_table = torch.empty(
            (row_count, budget_blocks), dtype=torch.int32, device=device)
        graph_batch = DSAForwardSparseDecodeBatch(
            max_logical_blocks=max_logical_blocks,
            score_topk_k=budget_tokens,
            resident_pool_indices_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            query_position_rows_tensor=torch.empty(
                (row_count, 1), dtype=torch.int32, device=device),
            tail_valid_token_counts_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            dense_tail_starts_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            resident_tail_starts_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            query_start_locs_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            query_lens_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            query_last_token_indices_tensor=torch.empty(
                (row_count,), dtype=torch.long, device=device),
            range_starts_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            range_ends_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            candidate_lens_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            budget_lengths_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            batch_hbm_block_table=graph_batch_hbm_block_table,
            attention_indices_width=self._graph_attention_indices_width(),
            query_last_token_indices_are_identity=True,
            batch_row_indices_tensor=torch.empty(
                (row_count,), dtype=torch.long, device=device),
            row_modes_tensor=torch.empty(
                (row_count,), dtype=torch.int32, device=device),
            lookup_init_mask_tensor=torch.empty(
                (row_count,), dtype=torch.bool, device=device),
            has_lookup_init_rows=True,
            active_local_row_indices_tensor=torch.empty(
                (row_count,), dtype=torch.long, device=device),
            active_batch_row_indices_tensor=torch.empty(
                (row_count,), dtype=torch.long, device=device),
            sparse_row_mask_tensor=torch.empty(
                (row_count,), dtype=torch.bool, device=device),
            sparse_local_row_indices_tensor=torch.empty(
                (row_count,), dtype=torch.long, device=device),
            sparse_batch_row_indices_tensor=torch.empty(
                (row_count,), dtype=torch.long, device=device),
            request_ids=_make_dsa_graph_dummy_request_ids(row_count),
            batch_row_indices=list(range(row_count)),
        )
        self._graph_row_mode_decode_batches[key] = graph_batch
        return graph_batch

    def _reset_row_mode_decode_graph_batch_for_capture(
        self,
        graph_batch: DSAForwardSparseDecodeBatch,
        full_block_table_tensor: torch.Tensor | None = None,
    ) -> None:
        """Fill graph buffers with a valid row-mode dummy decode batch."""
        row_count = int(graph_batch.resident_pool_indices_tensor.numel())
        device = graph_batch.resident_pool_indices_tensor.device
        budget_tokens = int(self._hbm_sparse_budget_tokens or 0)
        dummy_seq_len = self.get_row_mode_decode_graph_dummy_seq_len()
        dummy_range_end = min(
            dummy_seq_len,
            graph_batch.max_logical_blocks * int(self._vllm_blk_size),
        )
        dummy_dense_tail_start = (
            ((dummy_seq_len - 1) // int(self._vllm_blk_size))
            * int(self._vllm_blk_size))
        dummy_tail_count = dummy_seq_len - dummy_dense_tail_start

        row_ids = torch.arange(row_count, dtype=torch.int32, device=device)
        graph_batch.resident_pool_indices_tensor.copy_(row_ids)
        graph_batch.query_start_locs_tensor.copy_(row_ids)
        graph_batch.query_lens_tensor.fill_(1)
        graph_batch.query_last_token_indices_tensor.copy_(row_ids.to(
            dtype=torch.long))
        graph_batch.range_starts_tensor.fill_(0)
        graph_batch.range_ends_tensor.fill_(dummy_range_end)
        graph_batch.candidate_lens_tensor.fill_(dummy_range_end)
        graph_batch.budget_lengths_tensor.fill_(budget_tokens)
        graph_batch.dense_tail_starts_tensor.fill_(dummy_dense_tail_start)
        graph_batch.resident_tail_starts_tensor.fill_(
            int(self._lookup_total_slot_tokens))
        graph_batch.tail_valid_token_counts_tensor.fill_(dummy_tail_count)
        graph_batch.query_position_rows_tensor.fill_(
            int(self._lookup_total_slot_tokens) + dummy_tail_count - 1)
        graph_batch.batch_row_indices_tensor.copy_(row_ids.to(
            dtype=torch.long))
        graph_batch.row_modes_tensor.fill_(int(DSADecodeRowMode.SPARSE))
        graph_batch.lookup_init_mask_tensor.fill_(True)
        graph_batch.active_local_row_indices_tensor.copy_(row_ids.to(
            dtype=torch.long))
        graph_batch.active_batch_row_indices_tensor.copy_(row_ids.to(
            dtype=torch.long))
        graph_batch.sparse_row_mask_tensor.fill_(True)
        graph_batch.sparse_local_row_indices_tensor.copy_(row_ids.to(
            dtype=torch.long))
        graph_batch.sparse_batch_row_indices_tensor.copy_(row_ids.to(
            dtype=torch.long))
        if not _sync_row_mode_decode_graph_batch_row_maps(
                graph_batch, row_count=row_count):
            raise RuntimeError(
                "DSA row-mode graph capture failed to initialize "
                "dummy row-mode batch row maps")
        self._fill_graph_dummy_hbm_block_table(graph_batch,
                                               full_block_table_tensor)

    def prepare_row_mode_decode_graph_capture_batch(
        self,
        row_count: int,
        *,
        tensor_device: torch.device | str,
        full_block_table_tensor: torch.Tensor | None = None,
        graph_phase: str = DSA_GRAPH_PHASE_ROW_MODE_DECODE,
    ):
        """Install persistent DSA buffers while FULL graph is being captured."""
        graph_phase = str(graph_phase)
        graph_batch = self._get_or_create_row_mode_decode_graph_batch(
            row_count, tensor_device=tensor_device, graph_phase=graph_phase)
        self._ensure_graph_layer_id_tensors(tensor_device)
        # DSA 图捕获会用 dummy request/pool row 跑一遍 lookup 路径。
        # Lookup index 是请求生命周期状态，不是普通 graph input buffer；
        # 如果 dummy index 留下来，后续真实请求复用相同 pool_idx 时会被污染。
        self.resident_token_pool.clear_lookup_state_prefix(int(row_count))
        saved_state = (
            self.forward_sparse_decode_batch,
            self.forward_layer_batch,
            self.resident_token_pool.req_hbm_cached_token_counts[
                :int(row_count)].clone(),
        )
        self._reset_row_mode_decode_graph_batch_for_capture(
            graph_batch,
            full_block_table_tensor=full_block_table_tensor,
        )

        resident_tokens = int(self._hbm_resident_tokens or 0)
        row_count = int(row_count)
        if graph_phase != DSA_GRAPH_PHASE_ROW_MODE_DECODE:
            raise RuntimeError(f"Unknown DSA graph phase: {graph_phase}")
        self.resident_token_pool.req_hbm_cached_token_counts[
            :row_count, :].fill_(resident_tokens)

        self.forward_sparse_decode_batch = graph_batch
        self.forward_layer_batch = DSAForwardLayerBatch.empty(
            tensor_device=tensor_device)
        self._forward_sparse_decode_attention_indices_tensor = None
        return saved_state

    def restore_row_mode_decode_graph_capture_batch(self, saved_state) -> None:
        if saved_state is None:
            return
        (
            self.forward_sparse_decode_batch,
            self.forward_layer_batch,
            saved_counts,
        ) = saved_state
        row_count = int(saved_counts.shape[0])
        self.resident_token_pool.req_hbm_cached_token_counts[
            :row_count].copy_(saved_counts)
        # capture restore 只恢复 resident count metadata，lookup index
        # 由 resident pool 持有并按 pool row 清理，避免 capture-only
        # 状态泄漏到 replay。
        self.resident_token_pool.clear_lookup_state_prefix(row_count)

    def prepare_row_mode_decode_graph_replay_batch(
        self,
        row_count: int,
        *,
        graph_phase: str = DSA_GRAPH_PHASE_ROW_MODE_DECODE,
    ) -> bool:
        """Copy current-forward DSA metadata into graph-stable buffers."""
        graph_phase = str(graph_phase)
        real_batch = self.forward_sparse_decode_batch
        if not real_batch:
            return False
        if int(real_batch.resident_pool_indices_tensor.numel()) != int(
                row_count):
            return False
        real_batch_row_indices_tensor = getattr(
            real_batch, "batch_row_indices_tensor", None)
        real_active_local_rows = getattr(
            real_batch, "active_local_row_indices_tensor", None)
        real_active_batch_rows = getattr(
            real_batch, "active_batch_row_indices_tensor", None)
        if (not torch.is_tensor(real_batch_row_indices_tensor)
                or int(real_batch_row_indices_tensor.numel()) != int(
                    row_count)):
            return False
        real_sparse_rows = getattr(
            real_batch, "sparse_batch_row_indices_tensor", None)
        real_sparse_local_rows = getattr(
            real_batch, "sparse_local_row_indices_tensor", None)
        real_sparse_mask = getattr(real_batch, "sparse_row_mask_tensor", None)
        real_row_modes = getattr(real_batch, "row_modes_tensor", None)
        real_lookup_init_mask = getattr(
            real_batch, "lookup_init_mask_tensor", None)
        if (not torch.is_tensor(real_active_local_rows)
                or not torch.is_tensor(real_active_batch_rows)
                or not torch.is_tensor(real_sparse_rows)
                or not torch.is_tensor(real_sparse_local_rows)
                or not torch.is_tensor(real_sparse_mask)
                or not torch.is_tensor(real_row_modes)
                or not torch.is_tensor(real_lookup_init_mask)):
            return False
        if (int(real_active_local_rows.numel()) != int(row_count)
                or int(real_active_batch_rows.numel()) != int(row_count)):
            return False
        if int(real_sparse_rows.numel()) != int(real_sparse_local_rows.numel()):
            return False
        if int(real_sparse_rows.numel()) > int(row_count):
            return False
        if int(real_sparse_mask.numel()) != int(row_count):
            return False
        if int(real_row_modes.numel()) != int(row_count):
            return False
        if int(real_lookup_init_mask.numel()) != int(row_count):
            return False
        tensor_device = real_batch.resident_pool_indices_tensor.device
        graph_batch = self._get_or_create_row_mode_decode_graph_batch(
            row_count, tensor_device=tensor_device, graph_phase=graph_phase)
        if not _sync_row_mode_decode_graph_batch_row_maps(
                graph_batch,
                row_count=row_count,
                request_ids=real_batch.request_ids,
                batch_row_indices=real_batch.batch_row_indices,
        ):
            return False

        if int(real_batch.score_topk_k) != int(graph_batch.score_topk_k):
            return False
        if int(real_batch.max_logical_blocks) > int(
                graph_batch.max_logical_blocks):
            return False
        if int(real_batch.attention_indices_width) > int(
                graph_batch.attention_indices_width):
            return False
        if int(real_batch.query_position_rows_tensor.shape[1]) != int(
                graph_batch.query_position_rows_tensor.shape[1]):
            return False
        if int(real_batch.batch_hbm_block_table.shape[1]) > int(
                graph_batch.batch_hbm_block_table.shape[1]):
            return False
        self._ensure_graph_layer_id_tensors(tensor_device)
        self._copy_tensor_region(
            graph_batch.resident_pool_indices_tensor,
            real_batch.resident_pool_indices_tensor,
            fill_value=-1,
        )
        self._copy_tensor_region(
            graph_batch.query_position_rows_tensor,
            real_batch.query_position_rows_tensor,
            fill_value=-1,
        )
        self._copy_tensor_region(
            graph_batch.tail_valid_token_counts_tensor,
            real_batch.tail_valid_token_counts_tensor,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.dense_tail_starts_tensor,
            real_batch.dense_tail_starts_tensor,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.resident_tail_starts_tensor,
            real_batch.resident_tail_starts_tensor,
            fill_value=int(self._lookup_total_slot_tokens or 0),
        )
        self._copy_tensor_region(
            graph_batch.query_start_locs_tensor,
            real_batch.query_start_locs_tensor,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.query_lens_tensor,
            real_batch.query_lens_tensor,
            fill_value=1,
        )
        self._copy_tensor_region(
            graph_batch.query_last_token_indices_tensor,
            real_batch.query_last_token_indices_tensor,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.range_starts_tensor,
            real_batch.range_starts_tensor,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.range_ends_tensor,
            real_batch.range_ends_tensor,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.candidate_lens_tensor,
            real_batch.candidate_lens_tensor,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.budget_lengths_tensor,
            real_batch.budget_lengths_tensor,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.batch_row_indices_tensor,
            real_batch_row_indices_tensor,
            fill_value=-1,
        )
        self._copy_tensor_region(
            graph_batch.row_modes_tensor,
            real_row_modes,
            fill_value=int(DSADecodeRowMode.PAD),
        )
        self._copy_tensor_region(
            graph_batch.lookup_init_mask_tensor,
            real_lookup_init_mask,
            fill_value=False,
        )
        self._copy_tensor_region(
            graph_batch.active_local_row_indices_tensor,
            real_active_local_rows,
            fill_value=-1,
        )
        self._copy_tensor_region(
            graph_batch.active_batch_row_indices_tensor,
            real_active_batch_rows,
            fill_value=-1,
        )
        self._copy_tensor_region(
            graph_batch.sparse_row_mask_tensor,
            real_sparse_mask,
            fill_value=False,
        )
        self._copy_tensor_region(
            graph_batch.sparse_local_row_indices_tensor,
            real_sparse_local_rows,
            # 图 replay buffer 是固定 shape。即使 sparse_row_mask_tensor 会把
            # padding 行标成 False，capture 后的图里仍可能保留针对整段 index
            # tensor 的 gather/index_select；NPU gather 遇到 -1 会直接 assert。
            # 因此 index 类 padding 必须用安全行号 0，真实有效性只看 mask。
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.sparse_batch_row_indices_tensor,
            real_sparse_rows,
            fill_value=0,
        )
        self._copy_tensor_region(
            graph_batch.batch_hbm_block_table,
            real_batch.batch_hbm_block_table,
            fill_value=0,
        )
        self.forward_sparse_decode_batch = graph_batch
        self.forward_layer_batch = DSAForwardLayerBatch.empty(
            tensor_device=tensor_device)
        self._forward_sparse_decode_attention_indices_tensor = None
        return True
