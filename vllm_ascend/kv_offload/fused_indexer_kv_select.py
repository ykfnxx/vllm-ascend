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
import torch.nn.functional as F
from vllm.logger import logger

CACHE_SLOTS_CAPACITY = 262144
SELECTION_CACHE_SIZE = 10 * 1024
SPARSE_TOPK = 2048
OPERATOR_QUERY_HEADS = 64
OPERATOR_HEAD_DIM = 128
SUPPORTED_MICROBATCHES = (0, 1)
_REQUIRED_OP = "npu_lightning_indexer_decode_update_pool"
_REQUIRED_GATHER_OP = "dmp_lookup_kv_gather"
_REQUIRED_ATTENTION_OP = "npu_dmp_sparse_flash_attention"


@dataclass
class FusedIndexerKVSelectWorkspace:
    """One microbatch view into a layer-level request pool."""

    req_pool_entries: torch.Tensor
    topk_slots: torch.Tensor | None = None
    miss_count: torch.Tensor | None = None


@dataclass
class FusedIndexerAttentionWorkspace:
    """Persistent 10K local-KV staging used by scheme 3's mock KVIO."""

    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_block_table: torch.Tensor
    resident_token_ids: torch.Tensor
    copy_all_selected: torch.Tensor
    needs_refill: torch.Tensor
    selected_actual_seq: torch.Tensor
    topk_indices: torch.Tensor | None = None
    topk_slots: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None


class DMPFusedIndexerKVSelect:
    """Runs the request-pool fused Lightning Indexer and index update.

    A and B use disjoint rows of one persistent pool per layer. The fused
    operator updates token-to-slot state in place and returns the selected
    token indices, their 10K resident-cache slots, and the miss count.
    """

    def __init__(
        self,
        device: torch.device,
        max_microbatch_tokens: int,
        *,
        custom_ops: Any | None = None,
        gather_ops: Any | None = None,
        attention_ops: Any | None = None,
        block_size: int = 128,
    ) -> None:
        if max_microbatch_tokens <= 0:
            raise ValueError(f"max_microbatch_tokens must be positive, got {max_microbatch_tokens}")
        self.device = device
        self.max_microbatch_tokens = max_microbatch_tokens
        if block_size <= 0 or SELECTION_CACHE_SIZE % block_size != 0:
            raise ValueError(
                "Fused Indexer+Select requires block_size to divide 10240, "
                f"got {block_size}"
            )
        self.block_size = int(block_size)
        self.custom_ops = custom_ops or self._load_custom_ops()
        self.gather_ops = gather_ops or self._load_gather_ops()
        self.attention_ops = attention_ops or self._load_attention_ops()
        if not hasattr(self.custom_ops, _REQUIRED_OP):
            raise RuntimeError(f"Fused Indexer+KVSelect operator is unavailable: {_REQUIRED_OP}")
        if not hasattr(self.gather_ops, _REQUIRED_GATHER_OP):
            raise RuntimeError(
                f"Fused Indexer+Select KVIO operator is unavailable: {_REQUIRED_GATHER_OP}"
            )
        if not hasattr(self.attention_ops, _REQUIRED_ATTENTION_OP):
            raise RuntimeError(
                "Fused Indexer+Select attention operator is unavailable: "
                f"{_REQUIRED_ATTENTION_OP}"
            )
        self.pool_rows = len(SUPPORTED_MICROBATCHES) * max_microbatch_tokens
        self._cache_slot_pools: dict[str, torch.Tensor] = {}
        self._workspaces: dict[tuple[str, int], FusedIndexerKVSelectWorkspace] = {}
        self._attention_workspaces: dict[
            tuple[Any, ...], FusedIndexerAttentionWorkspace
        ] = {}
        self._active_attention_keys: dict[tuple[str, int], tuple[Any, ...]] = {}
        self._activation_logged = False
        self._head_padding_logged = False

    @staticmethod
    def _load_custom_ops() -> Any:
        try:
            import_module("lightning_indexer_decode_custom_ops")
        except ImportError as exc:
            raise RuntimeError(
                "VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=1 requires "
                "the lightning_indexer_decode_custom_ops extension"
            ) from exc
        return torch.ops.custom

    @staticmethod
    def _load_gather_ops() -> Any:
        try:
            import_module("dmp_lookup_maintain_custom_ops")
        except ImportError as exc:
            raise RuntimeError(
                "DMP fused Indexer+Select KVIO requires the "
                "dmp_lookup_maintain_custom_ops extension"
            ) from exc
        return torch.ops.dmp_lookup_maintain

    @staticmethod
    def _load_attention_ops() -> Any:
        try:
            import_module("custom_ops")
        except ImportError as exc:
            raise RuntimeError(
                "DMP fused Indexer+Select requires the Dual-Attention "
                "custom_ops extension"
            ) from exc
        return torch.ops.custom

    def _allocate_workspace(
        self,
        layer_name: str,
        microbatch_idx: int,
    ) -> FusedIndexerKVSelectWorkspace:
        cache_slots = self._cache_slot_pools.get(layer_name)
        if cache_slots is None:
            cache_slots = torch.full(
                (self.pool_rows, CACHE_SLOTS_CAPACITY),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            initial_slots = torch.arange(
                SELECTION_CACHE_SIZE,
                dtype=torch.int32,
                device=self.device,
            )
            cache_slots[:, :SELECTION_CACHE_SIZE] = initial_slots
            self._cache_slot_pools[layer_name] = cache_slots
        row_start = microbatch_idx * self.max_microbatch_tokens
        workspace = FusedIndexerKVSelectWorkspace(
            req_pool_entries=torch.arange(
                row_start,
                row_start + self.max_microbatch_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        )
        self._workspaces[(layer_name, microbatch_idx)] = workspace
        logger.debug(
            "Allocated fused Indexer+Select request-pool rows for %s microbatch %d",
            layer_name,
            microbatch_idx,
        )
        return workspace

    @property
    def allocated_tensor_bytes(self) -> int:
        index_bytes = sum(
            pool.numel() * pool.element_size()
            for pool in self._cache_slot_pools.values()
        ) + sum(
            workspace.req_pool_entries.numel()
            * workspace.req_pool_entries.element_size()
            for workspace in self._workspaces.values()
        )
        attention_bytes = sum(
            tensor.numel() * tensor.element_size()
            for workspace in self._attention_workspaces.values()
            for tensor in (
                workspace.selection_k_rope,
                workspace.selection_kv_cache,
                workspace.selection_block_table,
                workspace.resident_token_ids,
                workspace.copy_all_selected,
                workspace.needs_refill,
                workspace.selected_actual_seq,
            )
        )
        return index_bytes + attention_bytes

    def get_workspace(
        self,
        layer_name: str,
        microbatch_idx: int,
    ) -> FusedIndexerKVSelectWorkspace:
        try:
            return self._workspaces[(layer_name, microbatch_idx)]
        except KeyError as exc:
            raise RuntimeError(
                f"Fused Indexer+KVSelect workspace was not prepared for {layer_name} microbatch {microbatch_idx}"
            ) from exc

    @staticmethod
    def _normalize_cache(cache: torch.Tensor, name: str) -> torch.Tensor:
        if cache.ndim == 4:
            if cache.shape[2] != 1:
                raise RuntimeError(f"{name} requires one KV head, got {cache.shape}")
            return cache.squeeze(2)
        if cache.ndim != 3:
            raise RuntimeError(f"Unexpected {name} shape: {cache.shape}")
        return cache

    def _attention_key(
        self,
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

    def _allocate_attention_workspace(
        self,
        topk_indices: torch.Tensor,
        ql_nope: torch.Tensor,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
    ) -> FusedIndexerAttentionWorkspace:
        batch_size = int(topk_indices.shape[0])
        blocks_per_row = SELECTION_CACHE_SIZE // self.block_size
        num_blocks = batch_size * blocks_per_row
        device = topk_indices.device
        return FusedIndexerAttentionWorkspace(
            selection_k_rope=torch.empty(
                (num_blocks, self.block_size, full_k_rope.shape[-1]),
                dtype=full_k_rope.dtype,
                device=device,
            ),
            selection_kv_cache=torch.empty(
                (num_blocks, self.block_size, full_kv_cache.shape[-1]),
                dtype=full_kv_cache.dtype,
                device=device,
            ),
            selection_block_table=torch.arange(
                num_blocks, dtype=torch.int32, device=device
            ).view(batch_size, blocks_per_row),
            # The current local-HBM KVIO copies all selected 2K tokens every
            # step, so these compatibility inputs are intentionally inert.
            resident_token_ids=torch.zeros(
                (batch_size, SELECTION_CACHE_SIZE),
                dtype=torch.int32,
                device=device,
            ),
            copy_all_selected=torch.ones(
                (batch_size, SPARSE_TOPK), dtype=torch.int32, device=device
            ),
            needs_refill=torch.zeros(
                (batch_size,), dtype=torch.bool, device=device
            ),
            selected_actual_seq=torch.full(
                (batch_size,),
                SELECTION_CACHE_SIZE,
                dtype=torch.int32,
                device=device,
            ),
        )

    def prepare_attention(
        self,
        layer_name: str,
        microbatch_idx: int,
        indexer_result: tuple[torch.Tensor, ...],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: Any,
    ) -> None:
        ql_nope, _, topk_indices, _ = indexer_result
        index_workspace = self.get_workspace(layer_name, microbatch_idx)
        if index_workspace.topk_slots is None:
            raise RuntimeError("Fused Indexer+Select must run before KVIO prepare")
        full_kv_cache = self._normalize_cache(kv_cache[0], "full KV cache")
        full_k_rope = self._normalize_cache(kv_cache[1], "full K-rope cache")
        key = self._attention_key(
            layer_name,
            microbatch_idx,
            topk_indices,
            ql_nope,
            full_kv_cache,
            full_k_rope,
        )
        workspace = self._attention_workspaces.get(key)
        if workspace is None:
            workspace = self._allocate_attention_workspace(
                topk_indices, ql_nope, full_kv_cache, full_k_rope
            )
            self._attention_workspaces[key] = workspace
            logger.info(
                "Allocated scheme-3 10K local KVIO staging for %s microbatch %d",
                layer_name,
                microbatch_idx,
            )
        workspace.topk_indices = topk_indices.reshape(
            topk_indices.shape[0], SPARSE_TOPK
        )
        workspace.topk_slots = index_workspace.topk_slots.reshape(
            topk_indices.shape[0], SPARSE_TOPK
        )
        workspace.seq_lens = attn_metadata.seq_lens.to(
            dtype=torch.int32
        ).contiguous()
        self._active_attention_keys[(layer_name, microbatch_idx)] = key

    def get_attention_workspace(
        self, layer_name: str, microbatch_idx: int
    ) -> FusedIndexerAttentionWorkspace:
        try:
            key = self._active_attention_keys[(layer_name, microbatch_idx)]
            return self._attention_workspaces[key]
        except KeyError as exc:
            raise RuntimeError(
                "Fused Indexer+Select attention workspace was not prepared for "
                f"{layer_name} microbatch {microbatch_idx}"
            ) from exc

    def gather(
        self,
        layer_name: str,
        microbatch_idx: int,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: Any,
    ) -> None:
        workspace = self.get_attention_workspace(layer_name, microbatch_idx)
        if (
            workspace.topk_indices is None
            or workspace.topk_slots is None
            or workspace.seq_lens is None
        ):
            raise RuntimeError("Fused Indexer+Select prepare must run before KVIO")
        full_kv_cache = self._normalize_cache(kv_cache[0], "full KV cache")
        full_k_rope = self._normalize_cache(kv_cache[1], "full K-rope cache")
        self.gather_ops.dmp_lookup_kv_gather(
            workspace.selection_k_rope,
            workspace.selection_kv_cache,
            workspace.selection_block_table,
            workspace.resident_token_ids,
            workspace.topk_indices,
            workspace.topk_slots,
            workspace.copy_all_selected,
            workspace.needs_refill,
            full_k_rope,
            full_kv_cache,
            attn_metadata.block_table,
            workspace.seq_lens,
        )

    def run_attention(
        self,
        layer_name: str,
        microbatch_idx: int,
        indexer_result: tuple[torch.Tensor, ...],
        scale: float,
        attn_metadata: Any,
    ) -> torch.Tensor:
        ql_nope, q_pe, _, _ = indexer_result
        workspace = self.get_attention_workspace(layer_name, microbatch_idx)
        if workspace.topk_slots is None:
            raise RuntimeError("Fused Indexer+Select KVIO must run before attention")
        output = self.attention_ops.npu_dmp_sparse_flash_attention(
            ql_nope,
            workspace.selection_kv_cache.unsqueeze(2),
            workspace.selection_kv_cache.unsqueeze(2),
            workspace.topk_slots.unsqueeze(1),
            scale,
            1,
            block_table=workspace.selection_block_table,
            actual_seq_lengths_query=attn_metadata.cum_query_lens,
            actual_seq_lengths_kv=workspace.selected_actual_seq,
            query_rope=q_pe,
            key_rope=workspace.selection_k_rope.unsqueeze(2),
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    @staticmethod
    def _prepare_operator_inputs(
        query: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or query.shape[2] != OPERATOR_HEAD_DIM:
            raise RuntimeError(f"Fused Indexer+KVSelect requires query [B, N, 128], got {tuple(query.shape)}")
        query_heads = query.shape[1]
        if query_heads <= 0 or query_heads > OPERATOR_QUERY_HEADS:
            raise RuntimeError(f"Fused Indexer+KVSelect supports 1..64 query heads, got {query_heads}")
        if weights.ndim != 2 or tuple(weights.shape) != (query.shape[0], query_heads):
            raise RuntimeError(
                f"Fused Indexer+KVSelect requires weights [B, {query_heads}], got {tuple(weights.shape)}"
            )

        padding_heads = OPERATOR_QUERY_HEADS - query_heads
        if padding_heads:
            # The kernel computes a weighted reduction over its fixed 64-head
            # dimension. Zero query rows and zero weights preserve the score
            # produced by smaller test models exactly.
            query = F.pad(query, (0, 0, 0, padding_heads))
            weights = F.pad(weights, (0, padding_heads))
        return query.contiguous(), weights.contiguous()

    @staticmethod
    def _validate_inputs(
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        if query.ndim != 3 or tuple(query.shape[1:]) != (OPERATOR_QUERY_HEADS, OPERATOR_HEAD_DIM):
            raise RuntimeError(f"Fused Indexer+KVSelect operator input must be [B, 64, 128], got {tuple(query.shape)}")
        if key.ndim != 4 or key.shape[2] != 1 or key.shape[3] != 128:
            raise RuntimeError(
                f"Fused Indexer+KVSelect requires key [num_blocks, block_size, 1, 128], got {tuple(key.shape)}"
            )
        if weights.ndim != 2 or tuple(weights.shape) != (query.shape[0], OPERATOR_QUERY_HEADS):
            raise RuntimeError(f"Fused Indexer+KVSelect requires weights [B, 64], got {tuple(weights.shape)}")
        if actual_seq_lengths_key.shape != (query.shape[0],):
            raise RuntimeError("actual_seq_lengths_key must have one value per query row")
        if block_table.ndim != 2 or block_table.shape[0] != query.shape[0]:
            raise RuntimeError("block_table batch dimension must match query")
        tensors = (
            query,
            key,
            weights,
            actual_seq_lengths_key,
            block_table,
        )
        if not all(tensor.is_contiguous() for tensor in tensors):
            raise RuntimeError("Fused Indexer+KVSelect inputs must be contiguous")

    def select(
        self,
        layer_name: str,
        microbatch_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused top-k indices while retaining KVSelect metadata."""
        if microbatch_idx not in SUPPORTED_MICROBATCHES:
            raise RuntimeError(f"Fused Indexer+KVSelect supports DMP microbatches 0 and 1, got {microbatch_idx}")
        original_query_heads = query.shape[1] if query.ndim == 3 else None
        query, weights = self._prepare_operator_inputs(query, weights)
        self._validate_inputs(
            query,
            key,
            weights,
            actual_seq_lengths_key,
            block_table,
        )
        batch_size = query.shape[0]
        if batch_size > self.max_microbatch_tokens:
            raise RuntimeError(
                "Fused Indexer+KVSelect microbatch exceeds configured capacity: "
                f"batch={batch_size}, capacity={self.max_microbatch_tokens}"
            )
        workspace = self._workspaces.get((layer_name, microbatch_idx))
        if workspace is None:
            workspace = self._allocate_workspace(layer_name, microbatch_idx)

        cache_slots = self._cache_slot_pools[layer_name]
        topk_indices, topk_slots, miss_count = getattr(
            self.custom_ops,
            _REQUIRED_OP,
        )(
            query,
            key,
            weights,
            workspace.req_pool_entries[:batch_size],
            cache_slots,
            actual_seq_lengths_key,
            block_table,
        )
        workspace.topk_slots = topk_slots
        workspace.miss_count = miss_count
        if original_query_heads != OPERATOR_QUERY_HEADS and not self._head_padding_logged:
            logger.info(
                "DMP fused Indexer+KVSelect padded query heads %d -> %d with zeros",
                original_query_heads,
                OPERATOR_QUERY_HEADS,
            )
            self._head_padding_logged = True
        if not self._activation_logged:
            logger.info(
                "DMP fused Indexer+Select request pool active: 10K resident "
                "slots, 2K sparse output; index update is fused on S0"
            )
            self._activation_logged = True
        return topk_indices
