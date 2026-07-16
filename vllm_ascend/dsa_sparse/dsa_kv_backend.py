"""KV backend boundary for DSA sparse offload."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.dsa_sparse")


class DSAKVBackend(ABC):
    """Worker-local DSA KV I/O boundary.

    The framework decides which full blocks to put and which lookup misses to
    load. Implementations own storage, address translation, and I/O ordering.
    Loads write directly into the registered resident cache and return no KV
    tensor.
    """

    @property
    def requires_prefill_put(self) -> bool:
        """Whether sparse decode must wait for this request's prefill put."""
        return True

    @abstractmethod
    def register_layer_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        nopek_cache: torch.Tensor,
        ropek_cache: torch.Tensor,
    ) -> None:
        """Register stable NPU cache tensors for one MLA layer."""

    @abstractmethod
    def put_blocks(
        self,
        *,
        layer_id: int,
        request_ids: list,
        request_pool_indices: list[int],
        logical_block_index_rows: list[list[int]],
        block_key_rows: list[list],
        source_block_id_rows: list[list[int]],
    ) -> None:
        """Write complete HBM blocks to backend-owned storage."""

    @abstractmethod
    def load_tokens_into(
        self,
        *,
        layer_id: int,
        request_pool_entries: torch.Tensor,
        token_positions: torch.Tensor,
        destination_slots: torch.Tensor,
        load_mask: torch.Tensor,
        destination_block_table: torch.Tensor,
    ) -> None:
        """Load selected tokens directly into registered resident HBM slots."""

    @abstractmethod
    def release_request(self, *, request_id, request_pool_idx: int) -> None:
        """Release backend state associated with one request."""

    @abstractmethod
    def close(self) -> None:
        """Close backend resources owned by this worker."""


class MockDSAKVBackend(DSAKVBackend):
    """Storage-free backend that fills miss destinations with mock values."""

    def __init__(self, seed: int = 0) -> None:
        self._layer_caches: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}
        self._random = random.Random(int(seed))
        self._put_logged = False
        self._load_logged = False

    @property
    def requires_prefill_put(self) -> bool:
        # Mock loads synthesize values and do not consume prefill block data.
        return False

    @staticmethod
    def _squeeze_cache_head_dim(cache: torch.Tensor) -> torch.Tensor:
        if cache.ndim == 4:
            return cache.squeeze(2)
        return cache

    def register_layer_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        nopek_cache: torch.Tensor,
        ropek_cache: torch.Tensor,
    ) -> None:
        self._layer_caches[int(layer_id)] = (
            int(block_size),
            self._squeeze_cache_head_dim(nopek_cache),
            self._squeeze_cache_head_dim(ropek_cache),
        )

    def put_blocks(
        self,
        *,
        layer_id: int,
        request_ids: list,
        request_pool_indices: list[int],
        logical_block_index_rows: list[list[int]],
        block_key_rows: list[list],
        source_block_id_rows: list[list[int]],
    ) -> None:
        if not self._put_logged:
            logger.info(
                "DSA mock KV backend accepted block puts without storage: "
                "layer=%d, requests=%d",
                int(layer_id),
                len(request_ids),
            )
            self._put_logged = True

    def load_tokens_into(
        self,
        *,
        layer_id: int,
        request_pool_entries: torch.Tensor,
        token_positions: torch.Tensor,
        destination_slots: torch.Tensor,
        load_mask: torch.Tensor,
        destination_block_table: torch.Tensor,
    ) -> None:
        block_size, nopek_cache, ropek_cache = self._layer_caches[int(
            layer_id)]
        row_indices, token_indices = load_mask.to(
            dtype=torch.bool).nonzero(as_tuple=True)
        slots = destination_slots[row_indices,
                                  token_indices].to(dtype=torch.long)
        logical_blocks = torch.div(slots,
                                   block_size,
                                   rounding_mode="floor")
        block_offsets = torch.remainder(slots, block_size)
        physical_blocks = destination_block_table[
            row_indices, logical_blocks].to(dtype=torch.long)
        physical_slots = physical_blocks * block_size + block_offsets

        nopek_value = self._random.uniform(-1.0, 1.0)
        ropek_value = self._random.uniform(-1.0, 1.0)
        nopek_cache.reshape(-1, nopek_cache.shape[-1]).index_fill_(
            0, physical_slots, nopek_value)
        ropek_cache.reshape(-1, ropek_cache.shape[-1]).index_fill_(
            0, physical_slots, ropek_value)
        if not self._load_logged:
            logger.info(
                "DSA mock KV backend wrote lookup misses directly to resident "
                "HBM: layer=%d, request_rows=%d",
                int(layer_id),
                int(request_pool_entries.numel()),
            )
            self._load_logged = True

    def release_request(self, *, request_id, request_pool_idx: int) -> None:
        return

    def close(self) -> None:
        self._layer_caches.clear()


def create_dsa_kv_backend(vllm_config) -> DSAKVBackend:
    if vllm_config.cache_config.dsa_kv_backend != "mock":
        raise ValueError(
            "Only the mock DSA KV backend is available on this branch")
    return MockDSAKVBackend()
