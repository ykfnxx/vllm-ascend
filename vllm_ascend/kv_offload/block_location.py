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
from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockLocationTable:
    """Tracks which KV cache blocks reside on asu vs HBM.

    location tensor: [num_blocks], int8
      0 = HBM (NPU device memory)
      1 = asu (offloaded, needs load before access)
    """

    def __init__(self, num_blocks: int, device: torch.device):
        self.num_blocks = num_blocks
        self.location = torch.zeros(num_blocks,
                                    dtype=torch.int8,
                                    device=device)

    def update_from_scheduler(self, scheduler_output):
        """Update block locations from scheduler output.

        Called between scheduling steps. Blocks that have been
        swapped out by the scheduler are marked as asu (1).
        """
        # TODO: Integrate with vllm scheduler swap-out events
        # For now, all blocks assumed HBM (PlaceholderKVLoadOp compatible)
        pass

    def mark_blocks_asu(self, block_ids: torch.Tensor):
        """Mark specific blocks as asu-resident."""
        self.location[block_ids] = 1

    def mark_blocks_hbm(self, block_ids: torch.Tensor):
        """Mark specific blocks as HBM-resident (after load completes)."""
        self.location[block_ids] = 0

    def classify_topk_indices(
        self,
        topk_indices: torch.Tensor,
        block_size: int,
        num_real_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Classify topk slot indices into HBM hits and asu misses.

        Args:
            topk_indices: [num_tokens, num_kv_heads, sparse_count] int32
                slot indices from lightning_indexer.
            block_size: KV cache block size.
            num_real_tokens: 若传入，仅对前 num_real_tokens 个 token
                做 classify，跳过 padding token 的无效索引。

        Returns:
            hbm_mask: boolean mask over topk_indices (True = HBM hit)
            asu_block_ids: unique block IDs that are on asu and need loading
        """
        # 过滤 padding token：DMP 微批次填充的 dummy token 的 topk_indices
        # 可能指向无效 block（slot_mapping=-1 的 token 不应参与 classify），
        # 切片后避免越界访问 self.location 及触发无意义的 asu 加载
        if num_real_tokens is not None and num_real_tokens < topk_indices.shape[0]:
            topk_indices = topk_indices[:num_real_tokens]
        # 单次读取 self.location，避免跨层重叠中两次读取间
        # 并发 mark_blocks_hbm 导致 hbm_mask 与 asu_block_ids 不一致
        slot_to_block = topk_indices // block_size
        all_block_locations = self.location[slot_to_block]
        hbm_mask = all_block_locations == 0
        asu_mask = all_block_locations == 1
        asu_block_ids = torch.unique(slot_to_block[asu_mask])
        return hbm_mask, asu_block_ids
