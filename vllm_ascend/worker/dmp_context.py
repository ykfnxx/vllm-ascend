#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.kv_offload.block_location import BlockLocationTable
from vllm_ascend.kv_offload.kv_loader import KVLoader

if TYPE_CHECKING:
    from vllm_ascend.kv_offload.dual_attention import DMPDualAttention
    from vllm_ascend.kv_offload.fused_indexer_kv_select import (
        DMPFusedIndexerKVSelect,
    )
    from vllm_ascend.kv_offload.lookup_maintain import DMPLookupMaintain


@dataclass
class DMPSlice:
    """Token index range for one microbatch."""

    start: int  # inclusive
    end: int  # exclusive
    num_padded_tokens: int = 0  # dummy tokens added for 50/50 shape balance

    @property
    def num_real_tokens(self) -> int:
        return self.end - self.start

    @property
    def num_tokens(self) -> int:
        return (self.end - self.start) + self.num_padded_tokens


@dataclass
class DMPContext:
    """Holds all state needed for Dual Microbatch Pipeline across one
    model forward pass.

    Usage:
        # Enter microbatch scope (swaps forward_context.attn_metadata)
        with dmp_ctx.enter_microbatch(0):
            # attention ops see microbatch A's metadata
            ...
    """

    slices: list[DMPSlice]  # [0]=A, [1]=B
    kv_loader: KVLoader
    block_location: BlockLocationTable
    _attn_metadata_list: list  # [0]=A metadata, [1]=B metadata
    _num_tokens_list: list[int]  # [0]=A token count, [1]=B token count
    # 微批次级别的 MC2 padding 和 mask，用于 EP + DMP 组合场景
    _padded_num_tokens_list: list[int]  # [0]=A padded_num_tokens, [1]=B
    _mc2_mask_list: list[Optional[torch.Tensor]]  # [0]=A mc2_mask, [1]=B
    dual_attention: Optional["DMPDualAttention"] = None
    fused_indexer_kv_select: Optional["DMPFusedIndexerKVSelect"] = None
    lookup_maintain: Optional["DMPLookupMaintain"] = None
    _event_cache: dict[str, object] = field(default_factory=dict)
    _active_microbatch_idx: Optional[int] = field(default=None, init=False)

    def get_event(self, tag: str):
        """Return a stable Event for eager execution or graph capture."""
        event = self._event_cache.get(tag)
        if event is None:
            event = torch.npu.Event()
            self._event_cache[tag] = event
        return event

    def prepare_graph_events(self, num_layers: int) -> None:
        """Create all Events before entering the NPU graph capture scope."""
        self.get_event("dmp_fork")
        for layer_idx in range(num_layers):
            self.get_event(f"L{layer_idx}_indexer_A_done")
            self.get_event(f"L{layer_idx}_mlp_done")
            if (
                self.dual_attention is not None
                or self.lookup_maintain is not None
            ):
                for microbatch in ("A", "B"):
                    self.get_event(f"L{layer_idx}_indexer_{microbatch}_done")
                    self.get_event(f"L{layer_idx}_select_{microbatch}_done")
                    self.get_event(f"L{layer_idx}_gather_{microbatch}_done")
                    if self.lookup_maintain is not None:
                        self.get_event(
                            f"L{layer_idx}_maintain_{microbatch}_done"
                        )

    @property
    def active_microbatch_idx(self) -> int:
        if self._active_microbatch_idx is None:
            raise RuntimeError("DMP microbatch context is not active")
        return self._active_microbatch_idx

    @contextmanager
    def enter_microbatch(self, idx: int):
        """Context manager: swap forward_context attn_metadata to
        this microbatch's sliced metadata.

        Also swaps padded_num_tokens and mc2_mask so that MoE comm
        path (MC2/AllToAll) sees microbatch-level values.

        Restores all previous values on exit.
        """
        forward_context = get_forward_context()
        prev_attn_metadata = forward_context.attn_metadata
        prev_num_tokens = getattr(forward_context, "num_tokens", None)
        prev_padded_num_tokens = getattr(forward_context, "padded_num_tokens", None)
        prev_mc2_mask = getattr(forward_context, "mc2_mask", None)
        prev_microbatch_idx = self._active_microbatch_idx
        forward_context.attn_metadata = self._attn_metadata_list[idx]
        forward_context.num_tokens = self._num_tokens_list[idx]
        forward_context.padded_num_tokens = self._padded_num_tokens_list[idx]
        forward_context.mc2_mask = self._mc2_mask_list[idx]
        self._active_microbatch_idx = idx
        try:
            yield
        finally:
            self._active_microbatch_idx = prev_microbatch_idx
            forward_context.attn_metadata = prev_attn_metadata
            if prev_num_tokens is not None:
                forward_context.num_tokens = prev_num_tokens
            if prev_padded_num_tokens is not None:
                forward_context.padded_num_tokens = prev_padded_num_tokens
            if prev_mc2_mask is not None:
                forward_context.mc2_mask = prev_mc2_mask

    def slice_hidden_states(self, hidden_states: torch.Tensor, idx: int) -> torch.Tensor:
        s = self.slices[idx]
        return hidden_states[s.start : s.end]

    def merge_hidden_states(self, hs_a: torch.Tensor, hs_b: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        s_a, s_b = self.slices[0], self.slices[1]
        # Strip padding tokens before writing back to output
        real_a = hs_a[: s_a.num_real_tokens] if s_a.num_padded_tokens > 0 else hs_a
        real_b = hs_b[: s_b.num_real_tokens] if s_b.num_padded_tokens > 0 else hs_b
        output[s_a.start : s_a.end] = real_a
        output[s_b.start : s_b.end] = real_b
        return output
