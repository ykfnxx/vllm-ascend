#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
from vllm.logger import logger

SELECTION_TOPK_BLOCK_SIZE = 1
SUPPORTED_SELECTION_HEADS = 1
TWO_STREAM_MODE = "two"
FOUR_STREAM_MODE = "four"
SUPPORTED_STREAM_MODES = (TWO_STREAM_MODE, FOUR_STREAM_MODE)
LOCAL_KV_BACKEND = "local"
HIXL_KV_BACKEND = "hixl"
SUPPORTED_KV_BACKENDS = (LOCAL_KV_BACKEND, HIXL_KV_BACKEND)
_REQUIRED_ATTENTION_OPS = (
    "npu_dmp_sparse_flash_attention",
    "npu_da_attention_merge",
)
_REQUIRED_LOCAL_KV_OPS = (
    "npu_kv_select_out",
    "npu_kv_gather_out",
)


@dataclass
class DualAttentionPool:
    """Selection-cache storage shared by all graph shapes of one layer."""

    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_kv_block_table: torch.Tensor
    selection_kv_block_status: torch.Tensor
    request_signature: torch.Tensor
    previous_seq_lens: torch.Tensor
    capacity_rows: int
    blocks_per_row: int


@dataclass
class DualAttentionWorkspace:
    """Persistent selection cache and fixed-shape operator outputs."""

    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_kv_block_table: torch.Tensor
    selection_kv_block_status: torch.Tensor
    hit_sparse_indices: torch.Tensor
    miss_topk_indices: torch.Tensor
    miss_insert_indices: torch.Tensor
    hit_actual_seq: torch.Tensor
    miss_actual_seq: torch.Tensor
    miss_count: torch.Tensor
    hit_count: torch.Tensor
    selection_status_empty: torch.Tensor
    selection_kv_actual_seq: torch.Tensor
    selected_actual_seq: torch.Tensor
    full_q_actual_seq: torch.Tensor
    request_signature: torch.Tensor
    previous_seq_lens: torch.Tensor
    hit_softmax_max: torch.Tensor
    hit_softmax_sum: torch.Tensor
    miss_softmax_max: torch.Tensor
    miss_softmax_sum: torch.Tensor
    hit_attention_out: torch.Tensor | None = None
    backend_state: Any = None


class DMPDualAttention:
    """Coordinates split KV selection/gather and segmented SFA for DMP.

    The selection cache is persistent across decode steps. Workspaces are
    keyed by layer and microbatch so graph capture observes stable addresses.
    """

    def __init__(
        self,
        device: torch.device,
        block_size: int,
        *,
        custom_ops: Any | None = None,
        select_stream: Any | None = None,
        gather_stream: Any | None = None,
        stream_mode: str = FOUR_STREAM_MODE,
        max_microbatch_tokens: int | None = None,
        kv_backend: str = LOCAL_KV_BACKEND,
        hixl_backend: Any | None = None,
    ) -> None:
        self.device = device
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if max_microbatch_tokens is not None and max_microbatch_tokens <= 0:
            raise ValueError(f"max_microbatch_tokens must be positive, got {max_microbatch_tokens}")
        self.block_size = block_size
        self.max_microbatch_tokens = max_microbatch_tokens
        self.kv_backend = self._normalize_kv_backend(kv_backend)
        self.hixl_backend = hixl_backend
        if self.kv_backend == HIXL_KV_BACKEND and hixl_backend is None:
            raise ValueError("hixl_backend is required when kv_backend='hixl'")
        self.custom_ops = custom_ops or self._load_custom_ops()
        self._validate_custom_ops()
        self.stream_mode = self._normalize_stream_mode(stream_mode)
        if self.kv_backend == HIXL_KV_BACKEND and self.stream_mode != TWO_STREAM_MODE:
            raise ValueError(
                "The HIXL KV backend currently requires two-stream mode; "
                "four-stream HIXL would race on shared HCOMM workspaces"
            )
        if self.stream_mode == TWO_STREAM_MODE:
            if select_stream is not None and gather_stream is not None and select_stream is not gather_stream:
                raise ValueError("two-stream mode requires one shared auxiliary stream")
            auxiliary_stream = select_stream
            if auxiliary_stream is None:
                auxiliary_stream = gather_stream
            if auxiliary_stream is None:
                auxiliary_stream = torch.npu.Stream()
            self.select_stream = auxiliary_stream
            self.gather_stream = auxiliary_stream
        else:
            self.select_stream = select_stream or torch.npu.Stream()
            self.gather_stream = gather_stream or torch.npu.Stream()
        self._pools: dict[tuple[Any, ...], DualAttentionPool] = {}
        self._workspaces: dict[tuple[Any, ...], DualAttentionWorkspace] = {}
        self._active_workspace_keys: dict[tuple[str, int], tuple[Any, ...]] = {}

    @staticmethod
    def _load_custom_ops() -> Any:
        try:
            import_module("custom_ops")
        except ImportError as exc:
            raise RuntimeError("DMP Dual-Attention requires the pip-cache custom_ops wheel") from exc
        return torch.ops.custom

    def _validate_custom_ops(self) -> None:
        required = list(_REQUIRED_ATTENTION_OPS)
        if self.kv_backend == LOCAL_KV_BACKEND:
            required.extend(_REQUIRED_LOCAL_KV_OPS)
        missing = [op_name for op_name in required if not hasattr(self.custom_ops, op_name)]
        if missing:
            raise RuntimeError("DMP Dual-Attention custom operators are unavailable: " + ", ".join(missing))

    @staticmethod
    def _normalize_kv_backend(kv_backend: str) -> str:
        normalized = str(kv_backend).lower()
        if normalized not in SUPPORTED_KV_BACKENDS:
            raise ValueError(f"Unsupported DMP KV backend {kv_backend!r}; expected one of {SUPPORTED_KV_BACKENDS}")
        return normalized

    @staticmethod
    def _normalize_stream_mode(stream_mode: str) -> str:
        aliases = {"2": TWO_STREAM_MODE, "4": FOUR_STREAM_MODE}
        normalized = aliases.get(str(stream_mode).lower(), str(stream_mode).lower())
        if normalized not in SUPPORTED_STREAM_MODES:
            raise ValueError(f"Unsupported DMP stream mode {stream_mode!r}; expected one of {SUPPORTED_STREAM_MODES}")
        return normalized

    def get_indexer_a_stream(self, main_stream: Any, dmp_stream: Any) -> Any:
        """Select A's indexer stream for the configured topology."""
        if self.stream_mode == TWO_STREAM_MODE:
            return main_stream
        return dmp_stream

    @staticmethod
    def _normalize_cache(cache: torch.Tensor, name: str) -> torch.Tensor:
        if cache.ndim == 4:
            if cache.shape[2] != 1:
                raise ValueError(f"{name} requires one KV head, got shape {cache.shape}")
            return cache.squeeze(2)
        if cache.ndim != 3:
            raise ValueError(f"Unexpected {name} shape: {cache.shape}")
        return cache

    @staticmethod
    def _normalize_topk(topk_indices: torch.Tensor) -> torch.Tensor:
        if topk_indices.ndim == 2:
            topk_indices = topk_indices.unsqueeze(1)
        if topk_indices.ndim != 3:
            raise ValueError(f"Dual-Attention expects top-k indices shaped [T, H, K], got {topk_indices.shape}")
        return topk_indices

    @staticmethod
    def _unwrap_operator_output(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    @staticmethod
    def _workspace_key(
        layer_name: str,
        microbatch_idx: int,
        topk_indices: torch.Tensor,
        ql_nope: torch.Tensor,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            layer_name,
            microbatch_idx,
            tuple(topk_indices.shape),
            tuple(ql_nope.shape),
            full_kv_cache.shape[-1],
            full_k_rope.shape[-1],
            full_kv_cache.data_ptr(),
            full_k_rope.data_ptr(),
            topk_indices.device,
            ql_nope.dtype,
            full_kv_cache.dtype,
            full_k_rope.dtype,
        )

    def _pool_key(
        self,
        layer_name: str,
        microbatch_idx: int,
        topk_indices: torch.Tensor,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
    ) -> tuple[Any, ...]:
        key = (
            layer_name,
            topk_indices.shape[1],
            topk_indices.shape[2],
            full_kv_cache.shape[-1],
            full_k_rope.shape[-1],
            full_kv_cache.data_ptr(),
            full_k_rope.data_ptr(),
            topk_indices.device,
            full_kv_cache.dtype,
            full_k_rope.dtype,
        )
        if self.kv_backend == LOCAL_KV_BACKEND:
            return (layer_name, microbatch_idx, *key[1:])
        return key

    def _allocate_pool(
        self,
        layer_name: str,
        topk_indices: torch.Tensor,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
    ) -> DualAttentionPool:
        token_count, selection_heads, topk = topk_indices.shape
        if selection_heads != SUPPORTED_SELECTION_HEADS:
            raise ValueError(f"DMP Dual-Attention currently supports one selection head, got {selection_heads}")
        selection_capacity = topk * SELECTION_TOPK_BLOCK_SIZE
        if self.kv_backend == HIXL_KV_BACKEND:
            selection_capacity = int(self.hixl_backend.cache_size)
        blocks_per_row = (selection_capacity + self.block_size - 1) // self.block_size
        capacity_tokens = self.max_microbatch_tokens or token_count
        if self.kv_backend == HIXL_KV_BACKEND:
            capacity_tokens = self.hixl_backend.pool_size
        if token_count > capacity_tokens:
            raise RuntimeError(
                "DMP Dual-Attention microbatch exceeds the configured pool "
                f"capacity: tokens={token_count}, capacity={capacity_tokens}"
            )
        capacity_rows = capacity_tokens * selection_heads
        selection_num_blocks = capacity_rows * blocks_per_row
        device = topk_indices.device

        selection_block_table = torch.arange(selection_num_blocks, dtype=torch.int32, device=device).view(
            capacity_rows, blocks_per_row
        )
        status_shape = (capacity_tokens, selection_heads, topk + 1)

        if self.kv_backend == HIXL_KV_BACKEND:
            selection_kv_cache, selection_k_rope = self.hixl_backend.cache_tensors(
                layer_name,
                full_kv_cache.dtype,
                full_k_rope.dtype,
            )
        else:
            selection_k_rope = torch.empty(
                (selection_num_blocks, self.block_size, full_k_rope.shape[-1]),
                dtype=full_k_rope.dtype,
                device=device,
            )
            selection_kv_cache = torch.empty(
                (selection_num_blocks, self.block_size, full_kv_cache.shape[-1]),
                dtype=full_kv_cache.dtype,
                device=device,
            )

        return DualAttentionPool(
            selection_k_rope=selection_k_rope,
            selection_kv_cache=selection_kv_cache,
            selection_kv_block_table=selection_block_table,
            selection_kv_block_status=torch.full(status_shape, -1, dtype=torch.int32, device=device),
            request_signature=torch.full((capacity_tokens,), -1, dtype=torch.int32, device=device),
            previous_seq_lens=torch.full((capacity_tokens,), -1, dtype=torch.int32, device=device),
            capacity_rows=capacity_rows,
            blocks_per_row=blocks_per_row,
        )

    def _allocate_workspace(
        self,
        layer_name: str,
        microbatch_idx: int,
        topk_indices: torch.Tensor,
        ql_nope: torch.Tensor,
        pool: DualAttentionPool,
    ) -> DualAttentionWorkspace:
        token_count, selection_heads, _ = topk_indices.shape
        row_count = token_count * selection_heads
        row_offset = 0
        if self.kv_backend == HIXL_KV_BACKEND:
            if self.max_microbatch_tokens is None:
                raise RuntimeError("max_microbatch_tokens is required for the HIXL backend")
            if microbatch_idx not in (0, 1):
                raise RuntimeError("The HIXL DMP backend currently supports two microbatches")
            row_offset = microbatch_idx * self.max_microbatch_tokens
        if row_offset + row_count > pool.capacity_rows:
            raise RuntimeError(
                "DMP Dual-Attention microbatch exceeds the selection pool "
                f"capacity: offset={row_offset}, rows={row_count}, "
                f"capacity={pool.capacity_rows}"
            )
        block_offset = row_offset * pool.blocks_per_row
        selection_num_blocks = row_count * pool.blocks_per_row
        scalar_shape = (row_count,)
        lse_shape = (ql_nope.shape[0], ql_nope.shape[1])
        neg_inf = torch.finfo(torch.float32).min
        device = topk_indices.device
        selection_block_table = pool.selection_kv_block_table[row_offset : row_offset + row_count]
        if self.kv_backend == HIXL_KV_BACKEND:
            selection_block_table = selection_block_table - block_offset

        workspace = DualAttentionWorkspace(
            selection_k_rope=pool.selection_k_rope[block_offset : block_offset + selection_num_blocks],
            selection_kv_cache=pool.selection_kv_cache[block_offset : block_offset + selection_num_blocks],
            selection_kv_block_table=selection_block_table,
            selection_kv_block_status=pool.selection_kv_block_status[row_offset : row_offset + token_count],
            hit_sparse_indices=torch.empty_like(topk_indices, dtype=torch.int32),
            miss_topk_indices=torch.empty_like(topk_indices, dtype=torch.int32),
            miss_insert_indices=torch.empty_like(topk_indices, dtype=torch.int32),
            hit_actual_seq=torch.empty(scalar_shape, dtype=torch.int32, device=device),
            miss_actual_seq=torch.empty(scalar_shape, dtype=torch.int32, device=device),
            miss_count=torch.empty(scalar_shape, dtype=torch.int32, device=device),
            hit_count=torch.empty(scalar_shape, dtype=torch.int32, device=device),
            selection_status_empty=torch.empty(scalar_shape, dtype=torch.int32, device=device),
            selection_kv_actual_seq=torch.empty(scalar_shape, dtype=torch.int32, device=device),
            selected_actual_seq=torch.empty(scalar_shape, dtype=torch.int32, device=device),
            full_q_actual_seq=torch.ones((token_count,), dtype=torch.int32, device=device),
            request_signature=pool.request_signature[row_offset : row_offset + token_count],
            previous_seq_lens=pool.previous_seq_lens[row_offset : row_offset + token_count],
            hit_softmax_max=torch.full(lse_shape, neg_inf, dtype=torch.float32, device=device),
            hit_softmax_sum=torch.zeros(lse_shape, dtype=torch.float32, device=device),
            miss_softmax_max=torch.full(lse_shape, neg_inf, dtype=torch.float32, device=device),
            miss_softmax_sum=torch.zeros(lse_shape, dtype=torch.float32, device=device),
        )
        if self.kv_backend == HIXL_KV_BACKEND:
            workspace.backend_state = self.hixl_backend.prepare_workspace(layer_name, workspace, microbatch_idx)
        return workspace

    def get_or_create_workspace(
        self,
        layer_name: str,
        microbatch_idx: int,
        topk_indices: torch.Tensor,
        ql_nope: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
    ) -> DualAttentionWorkspace:
        topk_indices = self._normalize_topk(topk_indices)
        full_kv_cache = self._normalize_cache(kv_cache[0], "full KV cache")
        full_k_rope = self._normalize_cache(kv_cache[1], "full K-rope cache")
        key = self._workspace_key(
            layer_name,
            microbatch_idx,
            topk_indices,
            ql_nope,
            full_kv_cache,
            full_k_rope,
        )
        workspace = self._workspaces.get(key)
        if workspace is None:
            pool_key = self._pool_key(
                layer_name,
                microbatch_idx,
                topk_indices,
                full_kv_cache,
                full_k_rope,
            )
            pool = self._pools.get(pool_key)
            if pool is None:
                pool = self._allocate_pool(
                    layer_name,
                    topk_indices,
                    full_kv_cache,
                    full_k_rope,
                )
                self._pools[pool_key] = pool
            workspace = self._allocate_workspace(layer_name, microbatch_idx, topk_indices, ql_nope, pool)
            self._workspaces[key] = workspace
            logger.info(
                "Allocated DMP Dual-Attention workspace for %s microbatch %d",
                layer_name,
                microbatch_idx,
            )
        self._active_workspace_keys[(layer_name, microbatch_idx)] = key
        return workspace

    def get_workspace(self, layer_name: str, microbatch_idx: int) -> DualAttentionWorkspace:
        try:
            key = self._active_workspace_keys[(layer_name, microbatch_idx)]
            return self._workspaces[key]
        except KeyError as exc:
            raise RuntimeError(
                f"Dual-Attention workspace was not prepared for {layer_name} microbatch {microbatch_idx}"
            ) from exc

    @staticmethod
    def _reset_reassigned_requests(
        workspace: DualAttentionWorkspace,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        signature = block_table[:, 0].to(torch.int32)
        seq_lens = seq_lens.to(torch.int32)
        reassigned = (signature != workspace.request_signature) | (seq_lens <= workspace.previous_seq_lens)
        workspace.selection_kv_block_status.masked_fill_(reassigned.view(-1, 1, 1), -1)
        workspace.request_signature.copy_(signature)
        workspace.previous_seq_lens.copy_(seq_lens)

    def select(
        self,
        layer_name: str,
        microbatch_idx: int,
        indexer_result: tuple[torch.Tensor, ...],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: Any,
    ) -> None:
        ql_nope, _, topk_indices, _ = indexer_result
        topk_indices = self._normalize_topk(topk_indices)
        workspace = self.get_or_create_workspace(layer_name, microbatch_idx, topk_indices, ql_nope, kv_cache)
        if self.kv_backend == HIXL_KV_BACKEND:
            self.hixl_backend.select(
                layer_name,
                microbatch_idx,
                workspace,
                topk_indices,
                attn_metadata,
            )
            return
        full_kv_cache = self._normalize_cache(kv_cache[0], "full KV cache")
        full_k_rope = self._normalize_cache(kv_cache[1], "full K-rope cache")
        self._reset_reassigned_requests(workspace, attn_metadata.block_table, attn_metadata.seq_lens)
        workspace.hit_softmax_max.fill_(torch.finfo(torch.float32).min)
        workspace.hit_softmax_sum.zero_()
        workspace.miss_softmax_max.fill_(torch.finfo(torch.float32).min)
        workspace.miss_softmax_sum.zero_()
        workspace.hit_attention_out = None
        torch.clamp(
            attn_metadata.seq_lens,
            min=0,
            max=1,
            out=workspace.full_q_actual_seq,
        )

        self.custom_ops.npu_kv_select_out(
            workspace.selection_k_rope,
            workspace.selection_kv_cache,
            workspace.selection_kv_block_table,
            workspace.selection_kv_block_status,
            topk_indices,
            full_k_rope,
            full_kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            workspace.full_q_actual_seq,
            workspace.hit_sparse_indices,
            workspace.miss_topk_indices,
            workspace.miss_insert_indices,
            workspace.hit_actual_seq,
            workspace.miss_actual_seq,
            workspace.miss_count,
            workspace.hit_count,
            workspace.selection_status_empty,
            selection_topk_block_size=SELECTION_TOPK_BLOCK_SIZE,
        )
        torch.add(
            workspace.hit_actual_seq,
            workspace.miss_actual_seq,
            out=workspace.selected_actual_seq,
        )

    def gather(
        self,
        layer_name: str,
        microbatch_idx: int,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: Any,
    ) -> None:
        workspace = self.get_workspace(layer_name, microbatch_idx)
        if self.kv_backend == HIXL_KV_BACKEND:
            self.hixl_backend.gather(workspace)
            return
        full_kv_cache = self._normalize_cache(kv_cache[0], "full KV cache")
        full_k_rope = self._normalize_cache(kv_cache[1], "full K-rope cache")
        self.custom_ops.npu_kv_gather_out(
            workspace.selection_k_rope,
            workspace.selection_kv_cache,
            workspace.selection_kv_block_table,
            workspace.selection_kv_block_status,
            workspace.miss_topk_indices,
            workspace.miss_insert_indices,
            full_k_rope,
            full_kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            workspace.full_q_actual_seq,
            workspace.hit_actual_seq,
            workspace.miss_actual_seq,
            workspace.miss_count,
            workspace.hit_count,
            workspace.selection_status_empty,
            workspace.selection_kv_actual_seq,
            selection_topk_block_size=SELECTION_TOPK_BLOCK_SIZE,
        )

    def run_hit_attention(
        self,
        layer_name: str,
        microbatch_idx: int,
        indexer_result: tuple[torch.Tensor, ...],
        scale: float,
        attn_metadata: Any,
    ) -> None:
        ql_nope, q_pe, _, _ = indexer_result
        workspace = self.get_workspace(layer_name, microbatch_idx)
        workspace.hit_attention_out = self._unwrap_operator_output(
            self.custom_ops.npu_dmp_sparse_flash_attention(
                ql_nope,
                workspace.selection_kv_cache.unsqueeze(2),
                workspace.selection_kv_cache.unsqueeze(2),
                workspace.hit_sparse_indices,
                scale,
                SELECTION_TOPK_BLOCK_SIZE,
                block_table=workspace.selection_kv_block_table,
                actual_seq_lengths_query=attn_metadata.cum_query_lens,
                actual_seq_lengths_kv=workspace.selected_actual_seq,
                query_rope=q_pe,
                key_rope=workspace.selection_k_rope.unsqueeze(2),
                softmax_max_out=workspace.hit_softmax_max,
                softmax_sum_out=workspace.hit_softmax_sum,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )
        )

    def run_miss_attention_and_merge(
        self,
        layer_name: str,
        microbatch_idx: int,
        indexer_result: tuple[torch.Tensor, ...],
        scale: float,
        attn_metadata: Any,
    ) -> torch.Tensor:
        ql_nope, q_pe, _, _ = indexer_result
        workspace = self.get_workspace(layer_name, microbatch_idx)
        if workspace.hit_attention_out is None:
            raise RuntimeError("Hit attention must run before miss attention and merge")
        miss_attention_out = self._unwrap_operator_output(
            self.custom_ops.npu_dmp_sparse_flash_attention(
                ql_nope,
                workspace.selection_kv_cache.unsqueeze(2),
                workspace.selection_kv_cache.unsqueeze(2),
                workspace.miss_insert_indices,
                scale,
                SELECTION_TOPK_BLOCK_SIZE,
                block_table=workspace.selection_kv_block_table,
                actual_seq_lengths_query=attn_metadata.cum_query_lens,
                actual_seq_lengths_kv=workspace.selection_kv_actual_seq,
                query_rope=q_pe,
                key_rope=workspace.selection_k_rope.unsqueeze(2),
                softmax_max_out=workspace.miss_softmax_max,
                softmax_sum_out=workspace.miss_softmax_sum,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )
        )
        return self._unwrap_operator_output(
            self.custom_ops.npu_da_attention_merge(
                workspace.hit_attention_out,
                workspace.hit_softmax_max,
                workspace.hit_softmax_sum,
                miss_attention_out,
                workspace.miss_softmax_max,
                workspace.miss_softmax_sum,
            )
        )
