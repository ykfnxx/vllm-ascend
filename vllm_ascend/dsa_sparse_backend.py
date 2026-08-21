# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Worker-local payload backend for DSA Sparse Main KV."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

from vllm_ascend.dsa_sparse_config import DSA_SPARSE_KVIO_BACKEND

_STORAGE_KEY_DOMAIN = b"dsa-kvio-decode-v1"
_KVIO_DECODE_NAMESPACE = 1
_KVIO_PUT_OPCODE = 0x05
_KVIO_GET_OPCODE = 0x06


def _normalize_block_hash(block_hash: bytes | int) -> bytes:
    if isinstance(block_hash, bool):
        raise TypeError("boolean is not a valid scheduler block hash")
    if isinstance(block_hash, int):
        if block_hash < 0:
            raise ValueError("scheduler block hash must be non-negative")
        size = max(1, (block_hash.bit_length() + 7) // 8)
        value = block_hash.to_bytes(size, "big", signed=False)
        return b"i" + len(value).to_bytes(4, "big") + value
    if isinstance(block_hash, (bytes, bytearray, memoryview)):
        value = bytes(block_hash)
        if not value:
            raise ValueError("scheduler block hash must not be empty")
        return b"b" + len(value).to_bytes(4, "big") + value
    raise TypeError("scheduler block hash must be bytes-like or an integer")


class DSASparseStorageKeyEncoder:
    """Encode one full block and physical layer into a positive int63 key."""

    def __init__(self) -> None:
        self._identities: dict[int, bytes] = {}

    def encode(self, block_hash: bytes | int, layer_id: int) -> int:
        layer_id = int(layer_id)
        if not 0 <= layer_id < (1 << 32):
            raise ValueError("DSA Sparse physical layer id must fit uint32")
        identity = _STORAGE_KEY_DOMAIN + _normalize_block_hash(block_hash) + layer_id.to_bytes(4, "big", signed=False)
        counter = 0
        while True:
            digest_input = identity if counter == 0 else identity + counter.to_bytes(4, "big", signed=False)
            storage_id = int.from_bytes(
                hashlib.sha256(digest_input).digest()[:8],
                "big",
            ) & ((1 << 63) - 1)
            if storage_id:
                break
            counter += 1
        previous = self._identities.setdefault(storage_id, identity)
        if previous != identity:
            raise RuntimeError("DSA Sparse storage_request_id collision")
        return storage_id

    def encode_many(
        self,
        block_hashes: list[bytes | int] | tuple[bytes | int, ...],
        layer_id: int,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.encode(block_hash, layer_id) for block_hash in block_hashes],
            dtype=torch.int64,
            device=device,
        )


class DSASparseKVBackend:
    """Minimal full-layer-block PUT and layer-token GET boundary."""

    def register_layer_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        raise NotImplementedError

    def finalize_cache_registration(self) -> None:
        return

    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def load_tokens_into(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        token_offsets_in_block: torch.Tensor,
        destination_physical_slots: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return


class MockDSASparseKVBackend(DSASparseKVBackend):
    """Use the production call boundary without external payload storage."""

    def __init__(self) -> None:
        self.layer_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.put_calls: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self.load_calls: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def register_layer_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        del block_size
        if len(cache_planes) != 2:
            raise ValueError("DSA Sparse backend requires two Main KV planes")
        self.layer_caches[int(layer_id)] = (cache_planes[0], cache_planes[1])

    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        self.put_calls.append(
            (
                int(layer_id),
                storage_request_ids.detach().clone(),
                source_block_ids.detach().clone(),
            )
        )

    def load_tokens_into(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        token_offsets_in_block: torch.Tensor,
        destination_physical_slots: torch.Tensor,
    ) -> None:
        self.load_calls.append(
            (
                int(layer_id),
                storage_request_ids.detach().clone(),
                token_offsets_in_block.detach().clone(),
                destination_physical_slots.detach().clone(),
            )
        )

    def close(self) -> None:
        self.layer_caches.clear()


@dataclass(frozen=True)
class _KVIORegion:
    cache_id: int
    cache: torch.Tensor
    token_bytes: int
    block_bytes: int
    storage_offset: int


class KVIODSASparseKVBackend(DSASparseKVBackend):
    """Synchronous tensor-native adapter for the KVIO AIV ABI."""

    def __init__(
        self,
        model_id: int,
        *,
        ops_module: ModuleType | None = None,
        tensor_ops: Any | None = None,
    ) -> None:
        self._model_id = int(model_id)
        self._ops = ops_module
        self._tensor_ops = tensor_ops
        self._pending: dict[int, tuple[int, tuple[torch.Tensor, ...]]] = {}
        self._regions: dict[int, tuple[_KVIORegion, _KVIORegion]] = {}
        self._task_id: torch.Tensor | None = None
        self._model_id_tensor: torch.Tensor | None = None
        self._pd_flag: torch.Tensor | None = None
        self._put_opcode: torch.Tensor | None = None
        self._get_opcode: torch.Tensor | None = None

    def register_layer_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        if self._task_id is not None:
            raise RuntimeError("KVIO cache registration is already finalized")
        if len(cache_planes) != 2:
            raise ValueError("DSA Sparse KVIO requires two Main KV planes")
        if any(not plane.is_contiguous() for plane in cache_planes):
            raise ValueError("DSA Sparse KVIO cache planes must be contiguous")
        self._pending[int(layer_id)] = (int(block_size), cache_planes)

    def finalize_cache_registration(self) -> None:
        if self._task_id is not None:
            return
        if not self._pending:
            raise RuntimeError("KVIO requires at least one registered cache")
        if self._ops is None:
            self._ops = importlib.import_module("rdma_kv_ops")
        if self._tensor_ops is None:
            self._tensor_ops = torch.ops._C_ascend
        addresses: list[int] = []
        lengths: list[int] = []
        device: torch.device | None = None
        for layer_id in sorted(self._pending):
            block_size, planes = self._pending[layer_id]
            regions: list[_KVIORegion] = []
            storage_offset = 0
            for plane in planes:
                if device is None:
                    device = plane.device
                elif plane.device != device:
                    raise ValueError("KVIO cache planes must share one device")
                block_bytes = plane[0].numel() * plane.element_size()
                if block_bytes % block_size:
                    raise ValueError("KVIO block bytes must divide by block_size")
                cache_id = len(addresses)
                regions.append(
                    _KVIORegion(
                        cache_id=cache_id,
                        cache=plane,
                        token_bytes=block_bytes // block_size,
                        block_bytes=block_bytes,
                        storage_offset=storage_offset,
                    )
                )
                storage_offset += block_bytes
                addresses.append(plane.data_ptr())
                lengths.append(plane.numel() * plane.element_size())
            self._regions[layer_id] = (regions[0], regions[1])
        assert self._ops is not None and device is not None
        error_code = int(self._ops.aiv_init(addresses, lengths))
        if error_code:
            raise RuntimeError(f"KVIO aiv_init failed with error code {error_code}")
        kwargs = {"dtype": torch.int64, "device": device}
        self._task_id = torch.ones(1, **kwargs)
        self._model_id_tensor = torch.full((1,), self._model_id, **kwargs)
        self._pd_flag = torch.full((1,), _KVIO_DECODE_NAMESPACE, **kwargs)
        self._put_opcode = torch.full((1,), _KVIO_PUT_OPCODE, **kwargs)
        self._get_opcode = torch.full((1,), _KVIO_GET_OPCODE, **kwargs)

    @staticmethod
    def _interleave(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.stack((first, second), dim=1).reshape(-1).contiguous()

    def _submit(
        self,
        opcode: torch.Tensor,
        cache_ids: torch.Tensor,
        request_ids: torch.Tensor,
        cache_offsets: torch.Tensor,
        storage_offsets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        if self._task_id is None or self._model_id_tensor is None or self._pd_flag is None or self._tensor_ops is None:
            raise RuntimeError("KVIO cache registration is not finalized")
        io_nums = cache_ids.new_full((1,), cache_ids.numel())
        self._tensor_ops.npu_get_put_batch(
            self._task_id,
            self._model_id_tensor,
            self._pd_flag,
            io_nums,
            opcode,
            cache_ids,
            request_ids,
            cache_offsets,
            storage_offsets,
            lengths,
        )
        self._tensor_ops.npu_send_wait(self._task_id, io_nums)
        self._task_id.add_(1)

    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        regions = self._regions[int(layer_id)]
        device = regions[0].cache.device
        request_ids = storage_request_ids.to(device=device, dtype=torch.int64).reshape(-1).contiguous()
        block_ids = source_block_ids.to(device=device, dtype=torch.int64).reshape(-1).contiguous()
        if request_ids.numel() != block_ids.numel():
            raise ValueError("KVIO PUT descriptors must have equal size")
        if not request_ids.numel():
            return
        cache_ids = self._interleave(
            torch.full_like(request_ids, regions[0].cache_id),
            torch.full_like(request_ids, regions[1].cache_id),
        )
        cache_offsets = self._interleave(
            block_ids * regions[0].block_bytes,
            block_ids * regions[1].block_bytes,
        )
        storage_offsets = self._interleave(
            torch.full_like(request_ids, regions[0].storage_offset),
            torch.full_like(request_ids, regions[1].storage_offset),
        )
        lengths = self._interleave(
            torch.full_like(request_ids, regions[0].block_bytes),
            torch.full_like(request_ids, regions[1].block_bytes),
        )
        assert self._put_opcode is not None
        self._submit(
            self._put_opcode,
            cache_ids,
            request_ids.repeat_interleave(2).contiguous(),
            cache_offsets,
            storage_offsets,
            lengths,
        )

    def load_tokens_into(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        token_offsets_in_block: torch.Tensor,
        destination_physical_slots: torch.Tensor,
    ) -> None:
        regions = self._regions[int(layer_id)]
        device = regions[0].cache.device
        request_ids = storage_request_ids.to(device=device, dtype=torch.int64).reshape(-1).contiguous()
        token_offsets = token_offsets_in_block.to(device=device, dtype=torch.int64).reshape(-1).contiguous()
        slots = destination_physical_slots.to(device=device, dtype=torch.int64).reshape(-1).contiguous()
        if request_ids.numel() != token_offsets.numel() or request_ids.numel() != slots.numel():
            raise ValueError("KVIO GET descriptors must have equal size")
        if not request_ids.numel():
            return
        cache_ids = self._interleave(
            torch.full_like(request_ids, regions[0].cache_id),
            torch.full_like(request_ids, regions[1].cache_id),
        )
        cache_offsets = self._interleave(
            slots * regions[0].token_bytes,
            slots * regions[1].token_bytes,
        )
        storage_offsets = self._interleave(
            token_offsets * regions[0].token_bytes + regions[0].storage_offset,
            token_offsets * regions[1].token_bytes + regions[1].storage_offset,
        )
        lengths = self._interleave(
            torch.full_like(request_ids, regions[0].token_bytes),
            torch.full_like(request_ids, regions[1].token_bytes),
        )
        assert self._get_opcode is not None
        self._submit(
            self._get_opcode,
            cache_ids,
            request_ids.repeat_interleave(2).contiguous(),
            cache_offsets,
            storage_offsets,
            lengths,
        )

    def close(self) -> None:
        self._pending.clear()
        self._regions.clear()
        self._task_id = None


def create_dsa_sparse_backend(config: Any) -> DSASparseKVBackend:
    if config.io_backend == DSA_SPARSE_KVIO_BACKEND:
        return KVIODSASparseKVBackend(config.kvio_model_id)
    return MockDSASparseKVBackend()


__all__ = [
    "DSASparseKVBackend",
    "DSASparseStorageKeyEncoder",
    "KVIODSASparseKVBackend",
    "MockDSASparseKVBackend",
    "create_dsa_sparse_backend",
]
