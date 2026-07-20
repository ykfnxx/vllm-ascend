"""KVIO-backed storage implementation for DSA sparse MLA cache."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from types import ModuleType

import torch
from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_kv_backend import DSAKVBackend

logger = init_logger("vllm.dsa_sparse")


@dataclass(frozen=True)
class _KVIOCacheRegion:
    cache_id: int
    cache: torch.Tensor
    token_bytes: int
    block_bytes: int
    storage_base: int


class KVIODSAKVBackend(DSAKVBackend):
    """Move MLA nope/rope data through the ``rdma_kv_ops`` AIV API.

    KVIO registers each layer's nope and rope cache as two local NPU regions.
    Remote storage is addressed per integer request id; within that request,
    every registered region owns a contiguous ``max_model_len`` token range.
    """

    def __init__(
        self,
        *,
        model_id: int,
        pd_flag: int,
        max_model_len: int,
        ops_module: ModuleType | None = None,
    ) -> None:
        self._ops = (
            importlib.import_module("rdma_kv_ops")
            if ops_module is None else ops_module)
        self._model_id = int(model_id)
        self._pd_flag = int(pd_flag)
        self._max_model_len = int(max_model_len)
        self._registered_caches: dict[
            int, tuple[int, torch.Tensor, torch.Tensor]] = {}
        self._regions: dict[int, tuple[_KVIOCacheRegion,
                                       _KVIOCacheRegion]] = {}
        self._next_task_id = 1
        self._initialized = False
        self._request_ids_by_pool: dict[int, int] = {}

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def _next_task(self) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    @staticmethod
    def _encode_request_id(request_id) -> int:
        if isinstance(request_id, int):
            return int(request_id)
        digest = hashlib.blake2b(
            str(request_id).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") & 0x7FFFFFFFFFFFFFFF

    def _remote_request_id(self, pool_entry: int) -> int:
        return self._request_ids_by_pool[int(pool_entry)]

    def _check(self, operation: str, error_code) -> None:
        if error_code != self._ops.ErrorCode.SUCCESS:
            raise RuntimeError(
                f"KVIO {operation} failed with error code {error_code}")

    def _wait(self, task_id: int) -> None:
        self._check("aiv_wait", self._ops.aiv_wait([int(task_id)]))

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
        storage_base = 0
        for layer_id in sorted(self._registered_caches):
            block_size, nopek_cache, ropek_cache = self._registered_caches[
                layer_id]
            layer_regions = []
            for cache in (nopek_cache, ropek_cache):
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
                        storage_base=storage_base,
                    ))
                cache_addresses.append(int(cache.data_ptr()))
                cache_lengths.append(cache_bytes)
                storage_base += self._max_model_len * token_bytes
            self._regions[layer_id] = (layer_regions[0], layer_regions[1])

        self._check(
            "aiv_init",
            self._ops.aiv_init(cache_addresses, cache_lengths),
        )
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
        request_ids: list,
        request_pool_indices: list[int],
        logical_block_index_rows: list[list[int]],
        block_key_rows: list[list],
        source_block_id_rows: list[list[int]],
    ) -> None:
        _ = block_key_rows
        nopek_region, ropek_region = self._regions[int(layer_id)]
        cache_ids: list[int] = []
        remote_request_ids: list[int] = []
        cache_offsets: list[int] = []
        storage_offsets: list[int] = []
        request_lengths: list[int] = []

        for request_id, pool_entry, logical_blocks, physical_blocks in zip(
                request_ids,
                request_pool_indices,
                logical_block_index_rows,
                source_block_id_rows,
                strict=True):
            pool_entry = int(pool_entry)
            if pool_entry not in self._request_ids_by_pool:
                self._request_ids_by_pool[pool_entry] = (
                    self._encode_request_id(request_id))
            remote_request_id = self._remote_request_id(pool_entry)
            for logical_block, physical_block in zip(
                    logical_blocks, physical_blocks, strict=True):
                for region in (nopek_region, ropek_region):
                    cache_ids.append(region.cache_id)
                    remote_request_ids.append(remote_request_id)
                    cache_offsets.append(
                        int(physical_block) * region.block_bytes)
                    storage_offsets.append(
                        region.storage_base
                        + int(logical_block) * region.block_bytes)
                    request_lengths.append(region.block_bytes)

        if not cache_ids:
            return
        task_id = self._next_task()
        error_code, kernel_us = self._ops.aiv_put_batch(
            task_id,
            self._model_id,
            self._pd_flag,
            cache_ids,
            remote_request_ids,
            cache_offsets,
            storage_offsets,
            request_lengths,
        )
        self._check("aiv_put_batch", error_code)
        self._wait(task_id)
        logger.debug(
            "DSA KVIO put completed: layer=%d, transfers=%d, kernel_us=%s",
            int(layer_id), len(cache_ids), kernel_us)

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
        nopek_region, ropek_region = self._regions[int(layer_id)]
        row_indices, token_indices = load_mask.to(
            dtype=torch.bool).nonzero(as_tuple=True)
        if int(row_indices.numel()) == 0:
            return

        slots = destination_slots[row_indices,
                                  token_indices].to(dtype=torch.long)
        logical_blocks = torch.div(
            slots, nopek_region.block_bytes // nopek_region.token_bytes,
            rounding_mode="floor")
        block_offsets = torch.remainder(
            slots, nopek_region.block_bytes // nopek_region.token_bytes)
        physical_blocks = destination_block_table[
            row_indices, logical_blocks].to(dtype=torch.long)
        physical_slots = (
            physical_blocks
            * (nopek_region.block_bytes // nopek_region.token_bytes)
            + block_offsets)
        transfer_rows = torch.stack(
            (
                request_pool_entries.index_select(
                    0, row_indices).to(dtype=torch.long),
                token_positions[row_indices,
                                token_indices].to(dtype=torch.long),
                physical_slots,
            ),
            dim=1,
        ).cpu().tolist()

        cache_ids: list[int] = []
        remote_request_ids: list[int] = []
        cache_offsets: list[int] = []
        storage_offsets: list[int] = []
        request_lengths: list[int] = []
        for pool_entry, token_position, physical_slot in transfer_rows:
            remote_request_id = self._remote_request_id(pool_entry)
            for region in (nopek_region, ropek_region):
                cache_ids.append(region.cache_id)
                remote_request_ids.append(remote_request_id)
                cache_offsets.append(int(physical_slot) * region.token_bytes)
                storage_offsets.append(
                    region.storage_base
                    + int(token_position) * region.token_bytes)
                request_lengths.append(region.token_bytes)

        task_id = self._next_task()
        error_code, kernel_us = self._ops.aiv_get_batch(
            task_id,
            self._model_id,
            self._pd_flag,
            cache_ids,
            remote_request_ids,
            cache_offsets,
            storage_offsets,
            request_lengths,
        )
        self._check("aiv_get_batch", error_code)
        self._wait(task_id)
        logger.debug(
            "DSA KVIO get completed: layer=%d, transfers=%d, kernel_us=%s",
            int(layer_id), len(cache_ids), kernel_us)

    def release_request(self, *, request_id, request_pool_idx: int) -> None:
        self._request_ids_by_pool.pop(int(request_pool_idx), None)

    def close(self) -> None:
        if not self._initialized:
            return
        self._ops.aiv_destroy()
        self._initialized = False
        self._regions.clear()
        self._registered_caches.clear()
        self._request_ids_by_pool.clear()
