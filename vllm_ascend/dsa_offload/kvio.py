# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib
from dataclasses import dataclass
from typing import Any

import torch

_KVIO_NAMESPACE = 1
_KVIO_PUT_OPCODE = 0x05
_KVIO_GET_OPCODE = 0x06


@dataclass(frozen=True)
class _CacheRegistration:
    block_size: int
    cache_planes: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _KVIORegion:
    cache_id: int
    cache: torch.Tensor
    token_bytes: int
    block_bytes: int
    storage_offset: int


class KVIOBackend:
    def __init__(
        self,
        kvio_model_id: int,
        *,
        ops_module: Any | None = None,
        tensor_ops: Any | None = None,
    ) -> None:
        self._model_id = kvio_model_id
        self._ops = ops_module
        self._tensor_ops = tensor_ops
        self._pending_put: dict[int, _CacheRegistration] = {}
        self._pending_get: dict[int, _CacheRegistration] = {}
        self._put_regions: dict[int, tuple[_KVIORegion, ...]] = {}
        self._get_regions: dict[int, tuple[_KVIORegion, ...]] = {}
        self._task_id: torch.Tensor | None = None
        self._model_id_tensor: torch.Tensor | None = None
        self._namespace: torch.Tensor | None = None
        self._put_opcode: torch.Tensor | None = None
        self._get_opcode: torch.Tensor | None = None
        self._io_tail_event: object | None = None

    def register_put_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        self._register(self._pending_put, layer_id, block_size, cache_planes, "PUT")

    def register_get_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        self._register(self._pending_get, layer_id, block_size, cache_planes, "GET")

    def _register(
        self,
        registrations: dict[int, _CacheRegistration],
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
        direction: str,
    ) -> None:
        if self._task_id is not None:
            raise RuntimeError("KVIO cache registration is finalized")
        if layer_id in registrations:
            raise ValueError(f"KVIO {direction} cache is already registered for layer {layer_id}")
        if block_size <= 0:
            raise ValueError("KVIO block_size must be positive")
        if not cache_planes:
            raise ValueError("KVIO cache_planes must not be empty")
        for plane in cache_planes:
            if plane.ndim < 2 or plane.numel() == 0:
                raise ValueError("KVIO cache planes must not be empty")
            if plane.shape[1] != block_size:
                raise ValueError("KVIO cache plane block dimension must match block_size")
            if not plane.is_contiguous():
                raise ValueError("KVIO cache planes must be contiguous")
            block_bytes = plane[0].numel() * plane.element_size()
            if block_bytes % block_size:
                raise ValueError("KVIO block bytes must divide evenly by block_size")
        registrations[layer_id] = _CacheRegistration(block_size, cache_planes)

    def finalize_registration(self) -> None:
        if self._task_id is not None:
            return
        if not self._pending_put and not self._pending_get:
            raise RuntimeError("KVIO requires at least one registered cache")

        self._validate_shared_layers()
        if self._ops is None:
            self._ops = importlib.import_module("rdma_kv_ops")
        if self._tensor_ops is None:
            self._tensor_ops = torch.ops._C_ascend

        addresses: list[int] = []
        lengths: list[int] = []
        cache_ids: dict[tuple[int, int], int] = {}
        device: torch.device | None = None

        for pending, regions_by_layer in (
            (self._pending_put, self._put_regions),
            (self._pending_get, self._get_regions),
        ):
            for layer_id in sorted(pending):
                registration = pending[layer_id]
                storage_offset = 0
                regions = []
                for plane in registration.cache_planes:
                    if device is None:
                        device = plane.device
                    elif plane.device != device:
                        raise ValueError("KVIO cache planes must share one device")
                    cache_length = plane.numel() * plane.element_size()
                    cache_key = (plane.data_ptr(), cache_length)
                    cache_id = cache_ids.get(cache_key)
                    if cache_id is None:
                        cache_id = len(addresses)
                        cache_ids[cache_key] = cache_id
                        addresses.append(plane.data_ptr())
                        lengths.append(cache_length)
                    block_bytes = plane[0].numel() * plane.element_size()
                    regions.append(
                        _KVIORegion(
                            cache_id=cache_id,
                            cache=plane,
                            token_bytes=block_bytes // registration.block_size,
                            block_bytes=block_bytes,
                            storage_offset=storage_offset,
                        )
                    )
                    storage_offset += block_bytes
                regions_by_layer[layer_id] = tuple(regions)

        error_code = int(self._ops.aiv_init(addresses, lengths))
        if error_code:
            raise RuntimeError(f"KVIO aiv_init failed with error code {error_code}")

        tensor_options = {"dtype": torch.int64, "device": device}
        self._task_id = torch.ones(1, **tensor_options)
        self._model_id_tensor = torch.full((1,), self._model_id, **tensor_options)
        self._namespace = torch.full((1,), _KVIO_NAMESPACE, **tensor_options)
        self._put_opcode = torch.full((1,), _KVIO_PUT_OPCODE, **tensor_options)
        self._get_opcode = torch.full((1,), _KVIO_GET_OPCODE, **tensor_options)

    def _validate_shared_layers(self) -> None:
        for layer_id in self._pending_put.keys() & self._pending_get.keys():
            put_registration = self._pending_put[layer_id]
            get_registration = self._pending_get[layer_id]
            put_layout = (
                put_registration.block_size,
                tuple(
                    plane[0].numel() * plane.element_size() // put_registration.block_size
                    for plane in put_registration.cache_planes
                ),
            )
            get_layout = (
                get_registration.block_size,
                tuple(
                    plane[0].numel() * plane.element_size() // get_registration.block_size
                    for plane in get_registration.cache_planes
                ),
            )
            if put_layout != get_layout:
                raise ValueError(f"KVIO PUT and GET layouts differ for layer {layer_id}")

    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        regions = self._regions_for(self._put_regions, layer_id, "PUT")
        device = regions[0].cache.device
        self._validate_descriptor(storage_ids, device, "storage_ids")
        self._validate_descriptor(source_block_ids, device, "source_block_ids")
        if storage_ids.numel() != source_block_ids.numel():
            raise ValueError("KVIO PUT descriptor tensors must have equal size")
        if storage_ids.numel() == 0:
            return

        cache_ids = self._stack_planes(tuple(torch.full_like(storage_ids, region.cache_id) for region in regions))
        cache_offsets = self._stack_planes(tuple(source_block_ids * region.block_bytes for region in regions))
        storage_offsets = self._stack_planes(
            tuple(torch.full_like(storage_ids, region.storage_offset) for region in regions)
        )
        lengths = self._stack_planes(tuple(torch.full_like(storage_ids, region.block_bytes) for region in regions))
        request_ids = storage_ids.repeat_interleave(len(regions)).contiguous()
        assert self._put_opcode is not None
        self._submit(
            self._put_opcode,
            cache_ids,
            request_ids,
            cache_offsets,
            storage_offsets,
            lengths,
        )

    def get_tokens(
        self,
        *,
        layer_id: int,
        storage_ids: torch.Tensor,
        token_offsets: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None:
        regions = self._regions_for(self._get_regions, layer_id, "GET")
        device = regions[0].cache.device
        self._validate_descriptor(storage_ids, device, "storage_ids")
        self._validate_descriptor(token_offsets, device, "token_offsets")
        self._validate_descriptor(destination_slots, device, "destination_slots")
        if storage_ids.numel() != token_offsets.numel() or storage_ids.numel() != destination_slots.numel():
            raise ValueError("KVIO GET descriptor tensors must have equal size")
        if storage_ids.numel() == 0:
            return

        cache_ids = self._stack_planes(tuple(torch.full_like(storage_ids, region.cache_id) for region in regions))
        cache_offsets = self._stack_planes(tuple(destination_slots * region.token_bytes for region in regions))
        storage_offsets = self._stack_planes(
            tuple(token_offsets * region.token_bytes + region.storage_offset for region in regions)
        )
        lengths = self._stack_planes(tuple(torch.full_like(storage_ids, region.token_bytes) for region in regions))
        request_ids = storage_ids.repeat_interleave(len(regions)).contiguous()
        assert self._get_opcode is not None
        self._submit(
            self._get_opcode,
            cache_ids,
            request_ids,
            cache_offsets,
            storage_offsets,
            lengths,
        )

    def _regions_for(
        self,
        regions_by_layer: dict[int, tuple[_KVIORegion, ...]],
        layer_id: int,
        direction: str,
    ) -> tuple[_KVIORegion, ...]:
        if self._task_id is None:
            raise RuntimeError("KVIO cache registration is not finalized")
        if layer_id not in regions_by_layer:
            raise ValueError(f"KVIO {direction} cache is not registered for layer {layer_id}")
        return regions_by_layer[layer_id]

    @staticmethod
    def _validate_descriptor(tensor: torch.Tensor, device: torch.device, name: str) -> None:
        if tensor.ndim != 1 or tensor.dtype != torch.int64 or tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"KVIO {name} must be a contiguous int64 vector on the cache device")

    @staticmethod
    def _stack_planes(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.stack(values, dim=1).reshape(-1).contiguous()

    def _submit(
        self,
        opcode: torch.Tensor,
        cache_ids: torch.Tensor,
        request_ids: torch.Tensor,
        cache_offsets: torch.Tensor,
        storage_offsets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        assert self._task_id is not None
        assert self._model_id_tensor is not None
        assert self._namespace is not None
        current_stream = None
        if self._task_id.device.type == "npu":
            current_stream = torch.npu.current_stream()
            if self._io_tail_event is not None:
                current_stream.wait_event(self._io_tail_event)
        io_nums = self._task_id.new_full((1,), cache_ids.numel())
        self._tensor_ops.npu_get_put_batch(
            self._task_id,
            self._model_id_tensor,
            self._namespace,
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
        if current_stream is not None:
            self._io_tail_event = current_stream.record_event()

    def close(self) -> None:
        self._pending_put.clear()
        self._pending_get.clear()
        self._put_regions.clear()
        self._get_regions.clear()
        self._task_id = None
        self._model_id_tensor = None
        self._namespace = None
        self._put_opcode = None
        self._get_opcode = None
        self._io_tail_event = None
