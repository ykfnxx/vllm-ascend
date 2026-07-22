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
SELECTION_CACHE_SIZE = 2048
OPERATOR_QUERY_HEADS = 64
OPERATOR_HEAD_DIM = 128
SUPPORTED_MICROBATCHES = (0, 1)
_REQUIRED_OP = "npu_lightning_indexer_decode_update"


@dataclass
class FusedIndexerKVSelectWorkspace:
    """Per-layer cache-index state used by the fused decode operator."""

    cache_slots: torch.Tensor
    topk_slots: torch.Tensor | None = None
    miss_count: torch.Tensor | None = None


class DMPFusedIndexerKVSelect:
    """Runs fused Lightning Indexer and KVSelect without loading KV data.

    Only ``topk_indices`` is consumed by the existing full-cache SFA path.
    ``topk_slots`` and ``miss_count`` are retained for profiling and a future
    KVGather integration, but do not affect model output in this mode.
    """

    def __init__(
        self,
        device: torch.device,
        max_microbatch_tokens: int,
        *,
        custom_ops: Any | None = None,
    ) -> None:
        if max_microbatch_tokens <= 0:
            raise ValueError(f"max_microbatch_tokens must be positive, got {max_microbatch_tokens}")
        self.device = device
        self.max_microbatch_tokens = max_microbatch_tokens
        self.custom_ops = custom_ops or self._load_custom_ops()
        if not hasattr(self.custom_ops, _REQUIRED_OP):
            raise RuntimeError(f"Fused Indexer+KVSelect operator is unavailable: {_REQUIRED_OP}")
        self._workspaces: dict[tuple[str, int], FusedIndexerKVSelectWorkspace] = {}
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

    def _allocate_workspace(
        self,
        layer_name: str,
        microbatch_idx: int,
    ) -> FusedIndexerKVSelectWorkspace:
        cache_slots = torch.full(
            (self.max_microbatch_tokens, CACHE_SLOTS_CAPACITY),
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
        workspace = FusedIndexerKVSelectWorkspace(cache_slots=cache_slots)
        self._workspaces[(layer_name, microbatch_idx)] = workspace
        logger.debug(
            "Allocated fused Indexer+KVSelect state for %s microbatch %d",
            layer_name,
            microbatch_idx,
        )
        return workspace

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

        topk_indices, topk_slots, miss_count = getattr(
            self.custom_ops,
            _REQUIRED_OP,
        )(
            query,
            key,
            weights,
            workspace.cache_slots[:batch_size],
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
            logger.info("DMP fused Indexer+KVSelect active (KVGather disabled)")
            self._activation_logged = True
        return topk_indices
