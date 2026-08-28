# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

import torch

from .constants import INDEX_CAPACITY
from .hot_cache import HotCacheLayout
from .ops import asu_kv_gather


@dataclass(frozen=True)
class _LayerCache:
    destination_kv: torch.Tensor
    destination_rope: torch.Tensor
    source_kv: torch.Tensor
    source_rope: torch.Tensor
    source_block_table: torch.Tensor


class KVGatherSimBackend:
    def __init__(self, layout: HotCacheLayout) -> None:
        self._layout = layout
        self._layers: dict[int, _LayerCache] = {}

    @staticmethod
    def _flatten_plane(plane: torch.Tensor, block_size: int) -> torch.Tensor:
        if not plane.is_contiguous():
            raise ValueError("kvgather_sim cache planes must be contiguous")
        if plane.ndim < 3 or plane.shape[1] != block_size:
            raise ValueError(
                "kvgather_sim cache planes must be [blocks, block_size, ...]"
            )
        return plane.view(plane.shape[0], block_size, -1)

    def register_put_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        return

    def register_get_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        if len(cache_planes) != 2:
            raise ValueError("kvgather_sim requires separate KV and RoPE cache planes")
        destination_kv = self._flatten_plane(cache_planes[0], block_size)
        destination_rope = self._flatten_plane(cache_planes[1], block_size)
        if destination_kv.shape[0] != destination_rope.shape[0]:
            raise ValueError("kvgather_sim cache planes must have equal block counts")

        hot_begin = self._layout.hot_block_base
        hot_end = hot_begin + self._layout.hot_blocks
        destination_kv[hot_begin:hot_end].zero_()
        destination_rope[hot_begin:hot_end].zero_()
        source_kv = torch.zeros(
            (1, block_size, destination_kv.shape[2]),
            dtype=destination_kv.dtype,
            device=destination_kv.device,
        )
        source_rope = torch.zeros(
            (1, block_size, destination_rope.shape[2]),
            dtype=destination_rope.dtype,
            device=destination_rope.device,
        )
        source_block_table = torch.zeros(
            (
                self._layout.max_num_seqs,
                (INDEX_CAPACITY + block_size - 1) // block_size,
            ),
            dtype=torch.int32,
            device=destination_kv.device,
        )
        self._layers[int(layer_id)] = _LayerCache(
            destination_kv,
            destination_rope,
            source_kv,
            source_rope,
            source_block_table,
        )

    def finalize_registration(self) -> None:
        return

    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        return

    def get_tokens(
        self,
        *,
        layer_id: int,
        storage_ids: torch.Tensor,
        token_offsets: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None:
        return

    def gather_history_misses(
        self,
        *,
        layer_id: int,
        destination_block_table: torch.Tensor,
        request_rows: torch.Tensor,
        token_positions: torch.Tensor,
        destination_slots: torch.Tensor,
        miss_mask: torch.Tensor,
    ) -> bool:
        layer = self._layers[int(layer_id)]
        asu_kv_gather(
            layer.destination_kv,
            layer.destination_rope,
            destination_block_table,
            layer.source_kv,
            layer.source_rope,
            layer.source_block_table,
            request_rows,
            token_positions,
            destination_slots,
            miss_mask,
            self._layout.block_size,
        )
        return True

    def close(self) -> None:
        self._layers.clear()
