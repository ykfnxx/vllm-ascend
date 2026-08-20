"""KV backend boundary for DSA sparse offload."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.dsa_sparse")

DSA_STORAGE_LAYER_BITS = 8
_DSA_STORAGE_LAYER_LIMIT = 1 << DSA_STORAGE_LAYER_BITS
_DSA_STORAGE_BLOCK_HASH_BITS = 63 - DSA_STORAGE_LAYER_BITS
_DSA_STORAGE_BLOCK_HASH_MASK = (1 << _DSA_STORAGE_BLOCK_HASH_BITS) - 1


def encode_dsa_block_hash_id(block_hash) -> int:
    """Convert one vLLM block hash to the positive-int64 storage key prefix."""
    if isinstance(block_hash, bool):
        raise TypeError("boolean is not a valid vLLM block hash")
    if isinstance(block_hash, int):
        encoded = int(block_hash)
        if encoded < 0:
            raise ValueError("integer block hash must be non-negative")
    elif isinstance(block_hash, (bytes, bytearray, memoryview)):
        hash_bytes = bytes(block_hash)
        if not hash_bytes:
            raise ValueError("vLLM block hash bytes must not be empty")
        encoded = int.from_bytes(hash_bytes, byteorder="big", signed=False)
    else:
        raise TypeError(
            "vLLM block hash must be an integer or bytes-like value")
    return encoded & _DSA_STORAGE_BLOCK_HASH_MASK


def compose_dsa_storage_request_ids(
    block_hash_ids: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """Pack block-hash prefixes and one physical layer id into int64 keys."""
    layer_id = int(layer_id)
    if layer_id < 0 or layer_id >= _DSA_STORAGE_LAYER_LIMIT:
        raise ValueError(
            f"DSA storage layer id must be in [0, {_DSA_STORAGE_LAYER_LIMIT}), "
            f"got {layer_id}")
    block_hash_ids = block_hash_ids.to(dtype=torch.long)
    return ((block_hash_ids << DSA_STORAGE_LAYER_BITS)
            | layer_id).contiguous()


class DSAKVBackend(ABC):
    """Worker-local DSA KV I/O boundary.

    The framework decides which full blocks to put and which lookup misses to
    load. Implementations own storage, address translation, and I/O ordering.
    Loads write directly into the registered resident cache and return no KV
    tensor. Block-layer identities and block/token descriptors cross this
    boundary as tensors so a backend never has to materialize device metadata
    as Python lists.
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

    def finalize_cache_registration(self) -> None:
        """Finalize cache-region registration before the first transfer."""
        return

    @abstractmethod
    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        """Write flattened complete HBM blocks to backend-owned storage."""

    @abstractmethod
    def load_tokens_into(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        token_offsets_in_block: torch.Tensor,
        destination_physical_slots: torch.Tensor,
    ) -> None:
        """Load selected tokens directly into registered resident HBM slots."""

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
        storage_request_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        _ = source_block_ids
        if not self._put_logged:
            logger.info(
                "DSA mock KV backend accepted block puts without storage: "
                "layer=%d, blocks=%d",
                int(layer_id),
                int(storage_request_ids.numel()),
            )
            self._put_logged = True

    def load_tokens_into(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        token_offsets_in_block: torch.Tensor,
        destination_physical_slots: torch.Tensor,
    ) -> None:
        _, nopek_cache, ropek_cache = self._layer_caches[int(layer_id)]
        request_count = int(storage_request_ids.numel())
        if (int(token_offsets_in_block.numel()) != request_count
                or int(destination_physical_slots.numel()) != request_count):
            raise ValueError(
                "DSA mock token load descriptors must have equal size")
        physical_slots = destination_physical_slots.to(
            dtype=torch.long).reshape(-1)
        nopek_value = self._random.uniform(-1.0, 1.0)
        ropek_value = self._random.uniform(-1.0, 1.0)
        nopek_cache.reshape(-1, nopek_cache.shape[-1]).index_fill_(
            0, physical_slots, nopek_value)
        ropek_cache.reshape(-1, ropek_cache.shape[-1]).index_fill_(
            0, physical_slots, ropek_value)
        if not self._load_logged:
            logger.info(
                "DSA mock KV backend wrote lookup misses directly to resident "
                "HBM: layer=%d, tokens=%d",
                int(layer_id),
                request_count,
            )
            self._load_logged = True

    def close(self) -> None:
        self._layer_caches.clear()


def create_dsa_kv_backend(vllm_config) -> DSAKVBackend:
    backend_name = vllm_config.cache_config.dsa_kv_backend
    if backend_name == "mock":
        return MockDSAKVBackend()
    if backend_name == "kvio":
        from vllm_ascend.dsa_sparse.dsa_kvio_backend import KVIODSAKVBackend

        return KVIODSAKVBackend(
            model_id=int(vllm_config.cache_config.dsa_kvio_model_id),
            pd_flag=int(vllm_config.cache_config.dsa_kvio_pd_flag),
        )
    raise ValueError(f"Unsupported DSA KV backend: {backend_name}")
