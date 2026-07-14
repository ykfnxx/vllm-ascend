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

import torch
import torch_npu
from vllm.logger import init_logger

from vllm_ascend.kv_offload.asu_npu import KVLoadOp

logger = init_logger(__name__)


class KVLoader:
    """High-level KV cache loader with tag-based tracking and stream sync.

    Coordinates async KV cache loads for DMP:
    - async_load_blocks(): issues load on S1 (load stream)
    - wait_load_complete(): S0 waits for specific load tag to finish

    Tag format: "L{layer_idx}_A" / "L{layer_idx}_B" for cross-layer uniqueness.
    """

    def __init__(self, kv_load_op: KVLoadOp,
                 load_stream: torch_npu.npu.Stream):
        self.kv_load_op = kv_load_op
        self.load_stream = load_stream
        self._pending_loads: dict[str, torch_npu.npu.Event] = {}

        # graph-mode validation only:
        # cache events by tag, avoid creating a new Event every forward
        self._events: dict[str, torch_npu.npu.Event] = {}

    def _get_event(self, tag: str) -> torch_npu.npu.Event:
        event = self._events.get(tag)
        if event is None:
            event = torch_npu.npu.Event()
            self._events[tag] = event
        return event

    # def async_load_blocks(
    #     self,
    #     block_ids: torch.Tensor,
    #     tag: str,
    #     kv_cache: list[torch.Tensor],
    #     block_size: int,
    # ):
    #     """Issue async load for ASU blocks on S1, tagged for later wait.

    #     If block_ids is empty, records an instant-completion event
    #     (no actual I/O needed).
    #     """
    #     if block_ids.numel() == 0:
    #         # Record completion on the load stream (not current_stream)
    #         # so that wait_load_complete's wait_event is always ordered
    #         # against the correct stream, regardless of which stream
    #         # async_load_blocks was called from.
    #         event = torch_npu.npu.Event()
    #         with torch_npu.npu.stream(self.load_stream):
    #             event.record(self.load_stream)
    #         self._pending_loads[tag] = event
    #         return
    #     event = self.kv_load_op.async_load(kv_cache, block_ids, block_size,
    #                                        self.load_stream)
    #     self._pending_loads[tag] = event

    def async_load_blocks(
        self,
        block_ids: torch.Tensor,
        tag: str,
        kv_cache: list[torch.Tensor],
        block_size: int,
    ):
        """Graph-mode validation path: fake async load on S1.

        This does not perform real KV loading. It only records a fixed event
        on the load stream so graph capture can preserve stream dependency.
        """
        event = self._get_event(tag)

        with torch_npu.npu.stream(self.load_stream):
            event.record(self.load_stream)

        self._pending_loads[tag] = event

    # def wait_load_complete(self, tag: str):
    #     """S0 waits for the load identified by tag to complete.

    #     After waiting, the tag is removed from pending loads.
    #     """
    #     if tag in self._pending_loads:
    #         torch_npu.npu.current_stream().wait_event(self._pending_loads.pop(tag))

    def wait_load_complete(self, tag: str, wait_stream: torch_npu.npu.Stream = None):
        """Wait for the fake load identified by tag."""
        event = self._pending_loads.pop(tag, None)
        if event is not None:
            if wait_stream is None:
                wait_stream = torch_npu.npu.current_stream()
            wait_stream.wait_event(event)

    def has_pending(self, tag: str) -> bool:
        return tag in self._pending_loads
