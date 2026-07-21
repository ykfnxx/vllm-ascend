"""Lightweight KVIO writer for prefill TopK metadata on P-node (kv_producer).

In PD-disaggregated mode the P-node does not run DSA sparse decode logic
(resident pool, lookup, maintain). It only needs to write prefill TopK
positions into KVIO remote storage so that the D-node can read them during
ENTER_SPARSE_DECODE initialization.

To ensure cache_id alignment with the D-node's KVIODSAKVBackend, this writer
registers MLA dummy tensors (zero-filled, matching D-node shapes) before the
real TopK buffers, so the cache_id mapping is identical on both sides:
  cache_id 0..2L-1  : MLA nopek/ropek (dummy on P-node, real on D-node)
  cache_id 2L..2L+L-1: TopK metadata  (real on both P-node and D-node)
"""

from __future__ import annotations

import hashlib
import importlib
from types import ModuleType

import torch
from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_kvio_backend import TOPK_COUNT_DEFAULT

logger = init_logger("vllm.dsa_sparse")


class KVIOTopKMetadataWriter:

    def __init__(
        self,
        *,
        model_id: int,
        pd_flag: int,
        max_model_len: int,
        num_layers: int,
        block_size: int,
        num_blocks: int,
        kv_channels: int,
        head_dim: int,
        topk_count: int = TOPK_COUNT_DEFAULT,
        device: torch.device | str = "npu",
        ops_module: ModuleType | None = None,
    ) -> None:
        self._ops = (
            importlib.import_module("rdma_kv_ops")
            if ops_module is None else ops_module)
        self._model_id = int(model_id)
        self._pd_flag = int(pd_flag)
        self._max_model_len = int(max_model_len)
        self._num_layers = int(num_layers)
        self._block_size = int(block_size)
        self._num_blocks = int(num_blocks)
        self._kv_channels = int(kv_channels)
        self._head_dim = int(head_dim)
        self._topk_count = int(topk_count)
        self._device = torch.device(device)
        self._next_task_id = 1
        self._initialized = False
        self._request_ids_by_pool: dict[int, int] = {}

        self._dummy_mla_caches: list[torch.Tensor] = []
        self._topk_buffers: list[torch.Tensor] = []
        self._topk_regions: list[tuple[int, int, int]] = []

    @staticmethod
    def _encode_request_id(request_id) -> int:
        if isinstance(request_id, int):
            return int(request_id)
        digest = hashlib.blake2b(
            str(request_id).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") & 0x7FFFFFFFFFFFFFFF

    def _next_task(self) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _remote_request_id(self, pool_entry: int) -> int:
        return self._request_ids_by_pool[int(pool_entry)]

    def _check(self, operation: str, error_code) -> None:
        if error_code != self._ops.ErrorCode.SUCCESS:
            raise RuntimeError(
                f"KVIO {operation} failed with error code {error_code}")

    def _wait(self, task_id: int) -> None:
        self._check("aiv_wait", self._ops.aiv_wait([int(task_id)]))

    def finalize_registration(self) -> None:
        if self._initialized:
            return

        cache_addresses: list[int] = []
        cache_lengths: list[int] = []
        storage_base = 0

        mla_element_size = torch.float16.itemsize
        for layer_id in range(self._num_layers):
            nopek_shape = (self._num_blocks, self._block_size, 1,
                           self._kv_channels)
            ropek_shape = (self._num_blocks, self._block_size, 1,
                           self._head_dim)
            nopek_dummy = torch.zeros(nopek_shape, dtype=torch.float16,
                                      device=self._device)
            ropek_dummy = torch.zeros(ropek_shape, dtype=torch.float16,
                                       device=self._device)
            self._dummy_mla_caches.append(nopek_dummy)
            self._dummy_mla_caches.append(ropek_dummy)

            for dummy in (nopek_dummy, ropek_dummy):
                cache_bytes = int(dummy.numel()) * mla_element_size
                block_bytes = cache_bytes // self._num_blocks
                token_bytes = block_bytes // self._block_size
                cache_addresses.append(int(dummy.data_ptr()))
                cache_lengths.append(cache_bytes)
                storage_base += self._max_model_len * token_bytes

        position_dtype = torch.int32
        position_bytes = int(position_dtype.itemsize)
        topk_bytes_per_request = self._topk_count * position_bytes

        for layer_id in range(self._num_layers):
            topk_buffer = torch.zeros(self._max_model_len, self._topk_count,
                                      dtype=position_dtype,
                                      device=self._device)
            self._topk_buffers.append(topk_buffer)

            cache_id = len(cache_addresses)
            cache_bytes = int(topk_buffer.numel()) * position_bytes
            topk_storage_base = storage_base
            self._topk_regions.append(
                (cache_id, topk_storage_base, topk_bytes_per_request))
            cache_addresses.append(int(topk_buffer.data_ptr()))
            cache_lengths.append(cache_bytes)
            storage_base += self._max_model_len * position_bytes

        self._check(
            "aiv_init",
            self._ops.aiv_init(cache_addresses, cache_lengths),
        )
        self._initialized = True
        logger.info(
            "P-node KVIO TopK writer initialized: cache_regions=%d "
            "(MLA_dummy=%d, TopK=%d), model_id=%d, pd_flag=%d",
            len(cache_addresses),
            len(self._dummy_mla_caches),
            len(self._topk_buffers),
            self._model_id,
            self._pd_flag,
        )

    def put_topk(
        self,
        *,
        layer_id: int,
        request_id,
        request_pool_entry: int,
        topk_positions: torch.Tensor,
    ) -> None:
        if not self._initialized:
            raise RuntimeError(
                "KVIOTopKMetadataWriter not initialized; call "
                "finalize_registration() first")
        layer_id = int(layer_id)
        if layer_id >= self._num_layers:
            raise ValueError(
                f"layer_id {layer_id} exceeds num_layers {self._num_layers}")

        pool_entry = int(request_pool_entry)
        if pool_entry not in self._request_ids_by_pool:
            self._request_ids_by_pool[pool_entry] = (
                self._encode_request_id(request_id))

        topk_buffer = self._topk_buffers[layer_id]
        count = min(int(topk_positions.shape[0]), self._topk_count)
        topk_buffer[:count].copy_(topk_positions[:count].to(
            dtype=topk_buffer.dtype, device=topk_buffer.device))

        cache_id, storage_base, _ = self._topk_regions[layer_id]
        position_bytes = int(topk_buffer.element_size())
        remote_request_id = self._remote_request_id(pool_entry)

        cache_ids = [cache_id]
        remote_request_ids = [remote_request_id]
        cache_offsets = [0]
        storage_offsets = [storage_base]
        request_lengths = [count * position_bytes]

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
        self._check("aiv_put_batch (topk)", error_code)
        self._wait(task_id)
        logger.debug(
            "P-node KVIO topk put: layer=%d, pool_entry=%d, count=%d, "
            "kernel_us=%s",
            layer_id, pool_entry, count, kernel_us)

    def close(self) -> None:
        if not self._initialized:
            return
        self._ops.aiv_destroy()
        self._initialized = False
        self._dummy_mla_caches.clear()
        self._topk_buffers.clear()
        self._topk_regions.clear()
        self._request_ids_by_pool.clear()
