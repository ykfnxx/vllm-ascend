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

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch_npu
from vllm.logger import init_logger

logger = init_logger(__name__)


class KVLoadOp(ABC):
    """Abstract interface for ASU→HBM KV cache block loading.

    When a real NPU kv_load kernel is available, implement
    _load_blocks_impl to replace PlaceholderKVLoadOp.
    """

    @abstractmethod
    def async_load(
        self,
        kv_cache: list[torch.Tensor],
        block_ids: torch.Tensor,
        block_size: int,
        load_stream: torch_npu.npu.Stream,
    ) -> torch_npu.npu.Event:
        """Async load blocks on load_stream, return completion event."""
        ...


class PlaceholderKVLoadOp(KVLoadOp):
    """No-op placeholder: records an event on current stream without I/O.

    Used for development, debugging, and performance measurement
    (where the event represents zero-cost "instant load").
    """

    def async_load(
        self,
        kv_cache: list[torch.Tensor],
        block_ids: torch.Tensor,
        block_size: int,
        load_stream: torch_npu.npu.Stream,
    ) -> torch_npu.npu.Event:
        event = torch_npu.npu.Event()
        with torch_npu.npu.stream(load_stream):
            event.record(load_stream)
        return event


class SwapBlocksKVLoadOp(KVLoadOp):
    """Concrete KVLoadOp using swap_blocks for CPU→NPU transfer.

    Follows the same pattern as CpuNpuOffloadingHandler in cpu_npu.py.
    """

    def __init__(
        self,
        cpu_caches: list[tuple[torch.Tensor, torch.Tensor]],
        npu_caches: list[torch.Tensor],
        block_size_factor: int,
    ):
        self.cpu_caches = cpu_caches
        self.npu_caches = npu_caches
        self.block_size_factor = block_size_factor

    def async_load(
        self,
        kv_cache: list[torch.Tensor],
        block_ids: torch.Tensor,
        block_size: int,
        load_stream: torch_npu.npu.Stream,
    ) -> torch_npu.npu.Event:
        if block_ids.numel() == 0:
            event = torch_npu.npu.Event()
            event.record(torch_npu.npu.current_stream())
            return event

        # Build src_to_dst mapping (CPU block → NPU block)
        src_blocks = block_ids.cpu().numpy()
        sub_block_count = src_blocks.size * self.block_size_factor
        src_to_dst = np.empty((sub_block_count, 2), dtype=np.int64)
        _expand_block_ids(src_blocks, self.block_size_factor,
                          src_to_dst[:, 0])
        _expand_block_ids(src_blocks, 1, src_to_dst[:, 1])
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        event = torch_npu.npu.Event()
        with torch_npu.npu.stream(load_stream):
            for cpu_pair, npu_tensor in zip(self.cpu_caches, self.npu_caches):
                cpu_key, cpu_value = cpu_pair
                npu_key = npu_tensor[0]
                npu_value = npu_tensor[1]
                torch.ops._C_ascend.swap_blocks(cpu_key, npu_key,
                                                src_to_dst_tensor)
                torch.ops._C_ascend.swap_blocks(cpu_value, npu_value,
                                                src_to_dst_tensor)
            event.record(load_stream)
        return event


def _expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """Convert block IDs to sub-block IDs (same as cpu_npu.py)."""
    assert skip_count < block_size_factor
    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)
    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx
