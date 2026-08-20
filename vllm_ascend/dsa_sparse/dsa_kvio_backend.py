"""KVIO-backed storage implementation for DSA sparse MLA cache."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_kv_backend import DSAKVBackend

logger = init_logger("vllm.dsa_sparse")

_KVIO_PUT_OPCODE = 0x05
_KVIO_GET_OPCODE = 0x06
_KVIO_PREFILL_PD_FLAG = 0
_KVIO_DECODE_PD_FLAG = 1


@dataclass(frozen=True)
class _KVIOCacheRegion:
    cache_id: int
    cache: torch.Tensor
    token_bytes: int
    block_bytes: int


class KVIODSAKVBackend(DSAKVBackend):
    """Move MLA nope/rope data through the ``rdma_kv_ops`` AIV API.

    KVIO registers each layer's nope and rope cache as two local NPU regions.
    Each integer request id identifies one complete block for one physical
    layer. The remote object stores the nope block followed by the rope block.
    """

    def __init__(
        self,
        *,
        model_id: int,
        pd_flag: int,
        ops_module: ModuleType | None = None,
        tensor_ops: Any | None = None,
    ) -> None:
        self._ops = (
            importlib.import_module("rdma_kv_ops")
            if ops_module is None else ops_module)
        self._tensor_ops = (
            torch.ops._C_ascend if tensor_ops is None else tensor_ops)
        self._model_id = int(model_id)
        self._pd_flag = int(pd_flag)
        self._registered_caches: dict[
            int, tuple[int, torch.Tensor, torch.Tensor]] = {}
        self._regions: dict[int, tuple[_KVIOCacheRegion,
                                       _KVIOCacheRegion]] = {}
        self._initialized = False
        self._task_id_tensor: torch.Tensor | None = None
        self._model_id_tensor: torch.Tensor | None = None
        self._pd_flag_tensor: torch.Tensor | None = None
        self._prefill_pd_flag_tensor: torch.Tensor | None = None
        self._decode_pd_flag_tensor: torch.Tensor | None = None
        self._put_opcode_tensor: torch.Tensor | None = None
        self._get_opcode_tensor: torch.Tensor | None = None

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    @staticmethod
    def _interleave(first: torch.Tensor,
                    second: torch.Tensor) -> torch.Tensor:
        return torch.stack((first, second), dim=1).reshape(-1).contiguous()

    def _submit(
        self,
        *,
        pd_flag: torch.Tensor,
        opcode: torch.Tensor,
        cache_ids: torch.Tensor,
        storage_request_ids: torch.Tensor,
        cache_offsets: torch.Tensor,
        storage_offsets: torch.Tensor,
        block_lengths: torch.Tensor,
    ) -> None:
        if (self._task_id_tensor is None or self._model_id_tensor is None
                or self._pd_flag_tensor is None):
            raise RuntimeError("KVIO tensor controls are not initialized")
        io_nums = cache_ids.new_full((1,), int(cache_ids.numel()))
        self._tensor_ops.npu_get_put_batch(
            self._task_id_tensor,
            self._model_id_tensor,
            pd_flag,
            io_nums,
            opcode,
            cache_ids,
            storage_request_ids,
            cache_offsets,
            storage_offsets,
            block_lengths,
        )
        self._tensor_ops.npu_send_wait(self._task_id_tensor, io_nums)
        self._task_id_tensor.add_(1)

    def register_layer_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        nopek_cache: torch.Tensor,
        ropek_cache: torch.Tensor,
    ) -> None:
        layer_id = int(layer_id)
        if layer_id in self._registered_caches:
            return
        if self._initialized:
            raise RuntimeError(
                "KVIO cache registration cannot change after aiv_init")
        self._registered_caches[layer_id] = (
            int(block_size), nopek_cache, ropek_cache)

    def finalize_cache_registration(self) -> None:
        if self._initialized:
            return

        cache_addresses: list[int] = []
        cache_lengths: list[int] = []
        cache_device: torch.device | None = None
        for layer_id in sorted(self._registered_caches):
            block_size, nopek_cache, ropek_cache = self._registered_caches[
                layer_id]
            layer_regions = []
            for cache in (nopek_cache, ropek_cache):
                if cache_device is None:
                    cache_device = cache.device
                elif cache.device != cache_device:
                    raise RuntimeError(
                        "KVIO registered caches must share one device")
                cache_id = len(cache_addresses)
                cache_bytes = self._tensor_bytes(cache)
                block_bytes = cache_bytes // int(cache.shape[0])
                token_bytes = block_bytes // block_size
                layer_regions.append(
                    _KVIOCacheRegion(
                        cache_id=cache_id,
                        cache=cache,
                        token_bytes=token_bytes,
                        block_bytes=block_bytes,
                    ))
                cache_addresses.append(int(cache.data_ptr()))
                cache_lengths.append(cache_bytes)
            self._regions[layer_id] = (layer_regions[0], layer_regions[1])

        if cache_device is None:
            raise RuntimeError("KVIO requires at least one registered cache")
        error_code = int(self._ops.aiv_init(cache_addresses, cache_lengths))
        if error_code != 0:
            raise RuntimeError(
                f"KVIO aiv_init failed with error code {error_code}")
        tensor_kwargs = {"dtype": torch.long, "device": cache_device}
        self._task_id_tensor = torch.ones((1,), **tensor_kwargs)
        self._model_id_tensor = torch.full(
            (1,), self._model_id, **tensor_kwargs)
        self._pd_flag_tensor = torch.full(
            (1,), self._pd_flag, **tensor_kwargs)
        self._prefill_pd_flag_tensor = torch.full(
            (1,), _KVIO_PREFILL_PD_FLAG, **tensor_kwargs)
        self._decode_pd_flag_tensor = torch.full(
            (1,), _KVIO_DECODE_PD_FLAG, **tensor_kwargs)
        self._put_opcode_tensor = torch.full(
            (1,), _KVIO_PUT_OPCODE, **tensor_kwargs)
        self._get_opcode_tensor = torch.full(
            (1,), _KVIO_GET_OPCODE, **tensor_kwargs)
        self._initialized = True
        logger.info(
            "DSA KVIO backend initialized: cache_regions=%d, model_id=%d, "
            "pd_flag=%d",
            len(cache_addresses),
            self._model_id,
            self._pd_flag,
        )

    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        nopek_region, ropek_region = self._regions[int(layer_id)]
        device = nopek_region.cache.device
        storage_request_ids = storage_request_ids.to(
            device=device, dtype=torch.long).reshape(-1).contiguous()
        source_block_ids = source_block_ids.to(
            device=device, dtype=torch.long).reshape(-1).contiguous()
        block_count = int(storage_request_ids.numel())
        if int(source_block_ids.numel()) != block_count:
            raise ValueError(
                "KVIO block descriptor tensors must have equal size")
        if block_count == 0:
            return

        cache_ids = self._interleave(
            torch.full_like(storage_request_ids, nopek_region.cache_id),
            torch.full_like(storage_request_ids, ropek_region.cache_id),
        )
        request_ids = storage_request_ids.repeat_interleave(2).contiguous()
        cache_offsets = self._interleave(
            source_block_ids * nopek_region.block_bytes,
            source_block_ids * ropek_region.block_bytes,
        )
        storage_offsets = self._interleave(
            torch.zeros_like(storage_request_ids),
            torch.full_like(storage_request_ids, nopek_region.block_bytes),
        )
        block_lengths = self._interleave(
            torch.full_like(storage_request_ids, nopek_region.block_bytes),
            torch.full_like(storage_request_ids, ropek_region.block_bytes),
        )
        if self._put_opcode_tensor is None:
            raise RuntimeError("KVIO PUT opcode tensor is not initialized")
        for pd_flag in (
                self._prefill_pd_flag_tensor,
                self._decode_pd_flag_tensor,
        ):
            assert pd_flag is not None
            self._submit(
                pd_flag=pd_flag,
                opcode=self._put_opcode_tensor,
                cache_ids=cache_ids,
                storage_request_ids=request_ids,
                cache_offsets=cache_offsets,
                storage_offsets=storage_offsets,
                block_lengths=block_lengths,
            )
        logger.debug(
            "DSA KVIO put completed: layer=%d, transfers=%d",
            int(layer_id), int(cache_ids.numel()))

    def load_tokens_into(
        self,
        *,
        layer_id: int,
        storage_request_ids: torch.Tensor,
        token_offsets_in_block: torch.Tensor,
        destination_physical_slots: torch.Tensor,
    ) -> None:
        nopek_region, ropek_region = self._regions[int(layer_id)]
        device = nopek_region.cache.device
        request_ids = storage_request_ids.to(
            device=device, dtype=torch.long).reshape(-1).contiguous()
        token_offsets = token_offsets_in_block.to(
            device=device, dtype=torch.long).reshape(-1).contiguous()
        physical_slots = destination_physical_slots.to(
            device=device, dtype=torch.long).reshape(-1).contiguous()
        token_count = int(request_ids.numel())
        if (int(token_offsets.numel()) != token_count
                or int(physical_slots.numel()) != token_count):
            raise ValueError(
                "KVIO token descriptor tensors must have equal size")
        if token_count == 0:
            return
        cache_ids = self._interleave(
            torch.full_like(request_ids, nopek_region.cache_id),
            torch.full_like(request_ids, ropek_region.cache_id),
        )
        request_ids = request_ids.repeat_interleave(2).contiguous()
        cache_offsets = self._interleave(
            physical_slots * nopek_region.token_bytes,
            physical_slots * ropek_region.token_bytes,
        )
        storage_offsets = self._interleave(
            token_offsets * nopek_region.token_bytes,
            nopek_region.block_bytes
            + token_offsets * ropek_region.token_bytes,
        )
        block_lengths = self._interleave(
            torch.full_like(token_offsets, nopek_region.token_bytes),
            torch.full_like(token_offsets, ropek_region.token_bytes),
        )
        if self._get_opcode_tensor is None:
            raise RuntimeError("KVIO GET opcode tensor is not initialized")
        self._submit(
            pd_flag=self._pd_flag_tensor,
            opcode=self._get_opcode_tensor,
            cache_ids=cache_ids,
            storage_request_ids=request_ids,
            cache_offsets=cache_offsets,
            storage_offsets=storage_offsets,
            block_lengths=block_lengths,
        )
        logger.debug(
            "DSA KVIO get completed: layer=%d, transfers=%d",
            int(layer_id), int(cache_ids.numel()))

    def close(self) -> None:
        if not self._initialized:
            return
        self._initialized = False
        self._regions.clear()
        self._registered_caches.clear()
        self._task_id_tensor = None
        self._model_id_tensor = None
        self._pd_flag_tensor = None
        self._prefill_pd_flag_tensor = None
        self._decode_pd_flag_tensor = None
        self._put_opcode_tensor = None
        self._get_opcode_tensor = None
