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


@dataclass(frozen=True)
class _KVIOLayerRegions:
    indexer: _KVIOCacheRegion | None
    nopek: _KVIOCacheRegion
    ropek: _KVIOCacheRegion


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
        request_namespace: str = "",
        ops_module: ModuleType | None = None,
    ) -> None:
        self._ops = (
            importlib.import_module("rdma_kv_ops")
            if ops_module is None else ops_module)
        self._model_id = int(model_id)
        self._pd_flag = int(pd_flag)
        self._max_model_len = int(max_model_len)
        self._request_namespace = str(request_namespace)
        self._registered_caches: dict[
            int,
            tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None],
        ] = {}
        self._regions: dict[int, _KVIOLayerRegions] = {}
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
    def encode_request_id(
        request_id,
        *,
        namespace: str = "",
    ) -> int:
        if isinstance(request_id, int) and not namespace:
            return int(request_id)
        encoded_request = (
            f"{namespace}\0{request_id}" if namespace else str(request_id))
        digest = hashlib.blake2b(
            encoded_request.encode("utf-8"), digest_size=8).digest()
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
        indexer_cache: torch.Tensor | None = None,
    ) -> None:
        layer_id = int(layer_id)
        if layer_id in self._registered_caches:
            return
        if self._initialized:
            raise RuntimeError(
                "KVIO cache registration cannot change after aiv_init")
        self._registered_caches[layer_id] = (
            int(block_size), nopek_cache, ropek_cache, indexer_cache)

    def finalize_cache_registration(self) -> None:
        if self._initialized:
            return

        cache_addresses: list[int] = []
        cache_lengths: list[int] = []
        storage_base = 0
        for layer_id in sorted(self._registered_caches):
            (
                block_size,
                nopek_cache,
                ropek_cache,
                indexer_cache,
            ) = self._registered_caches[layer_id]
            layer_regions: list[_KVIOCacheRegion] = []
            ordered_caches = (
                (indexer_cache, nopek_cache, ropek_cache)
                if indexer_cache is not None
                else (nopek_cache, ropek_cache)
            )
            for cache in ordered_caches:
                assert cache is not None
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
            if indexer_cache is None:
                self._regions[layer_id] = _KVIOLayerRegions(
                    indexer=None,
                    nopek=layer_regions[0],
                    ropek=layer_regions[1],
                )
            else:
                self._regions[layer_id] = _KVIOLayerRegions(
                    indexer=layer_regions[0],
                    nopek=layer_regions[1],
                    ropek=layer_regions[2],
                )

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
        source_indexer_block_id_rows: list[list[int]] | None = None,
        valid_token_count_rows: list[int] | None = None,
    ) -> None:
        _ = block_key_rows
        regions = self._regions[int(layer_id)]
        cache_ids: list[int] = []
        remote_request_ids: list[int] = []
        cache_offsets: list[int] = []
        storage_offsets: list[int] = []
        request_lengths: list[int] = []

        row_count = len(request_ids)
        if source_indexer_block_id_rows is None:
            source_indexer_block_id_rows = [[] for _ in range(row_count)]
        if valid_token_count_rows is None:
            valid_token_count_rows = [
                (
                    (max(logical_blocks) + 1)
                    * self._registered_caches[int(layer_id)][0]
                    if logical_blocks else 0
                )
                for logical_blocks in logical_block_index_rows
            ]

        block_size = self._registered_caches[int(layer_id)][0]
        for (
            request_id,
            pool_entry,
            logical_blocks,
            physical_blocks,
            indexer_physical_blocks,
            valid_token_count,
        ) in zip(
                request_ids,
                request_pool_indices,
                logical_block_index_rows,
                source_block_id_rows,
                source_indexer_block_id_rows,
                valid_token_count_rows,
                strict=True):
            pool_entry = int(pool_entry)
            if pool_entry not in self._request_ids_by_pool:
                self._request_ids_by_pool[pool_entry] = (
                    self.encode_request_id(
                        request_id,
                        namespace=self._request_namespace,
                    ))
            remote_request_id = self._remote_request_id(pool_entry)
            if regions.indexer is not None and (
                len(indexer_physical_blocks) != len(logical_blocks)
            ):
                raise RuntimeError(
                    "DSA KVIO put requires one Indexer block id per logical "
                    f"block for layer {int(layer_id)}")
            for block_row, (logical_block, physical_block) in enumerate(zip(
                    logical_blocks, physical_blocks, strict=True)):
                token_start = int(logical_block) * block_size
                token_count = min(
                    block_size,
                    max(0, int(valid_token_count) - token_start),
                )
                if token_count <= 0:
                    continue
                region_sources: list[tuple[_KVIOCacheRegion, int]] = []
                if regions.indexer is not None:
                    region_sources.append((
                        regions.indexer,
                        int(indexer_physical_blocks[block_row]),
                    ))
                region_sources.extend((
                    (regions.nopek, int(physical_block)),
                    (regions.ropek, int(physical_block)),
                ))
                for region, source_block in region_sources:
                    cache_ids.append(region.cache_id)
                    remote_request_ids.append(remote_request_id)
                    cache_offsets.append(source_block * region.block_bytes)
                    storage_offsets.append(
                        region.storage_base
                        + token_start * region.token_bytes)
                    request_lengths.append(token_count * region.token_bytes)

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
        regions = self._regions[int(layer_id)]
        nopek_region, ropek_region = regions.nopek, regions.ropek
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

    def bind_request(
        self,
        *,
        request_id,
        request_pool_idx: int,
        remote_request_id: int | None = None,
    ) -> None:
        pool_entry = int(request_pool_idx)
        encoded_request_id = (
            self.encode_request_id(
                request_id,
                namespace=self._request_namespace,
            )
            if remote_request_id is None else int(remote_request_id)
        )
        current = self._request_ids_by_pool.get(pool_entry)
        if current is not None and current != encoded_request_id:
            raise RuntimeError(
                "DSA KVIO resident pool row is already bound to a different "
                f"remote request: pool={pool_entry}, current={current}, "
                f"new={encoded_request_id}")
        self._request_ids_by_pool[pool_entry] = encoded_request_id

    @staticmethod
    def _append_range_transfers(
        *,
        region: _KVIOCacheRegion,
        remote_request_id: int,
        token_start: int,
        token_count: int,
        destination_slot_start: int,
        destination_block_ids: list[int],
        block_size: int,
        cache_ids: list[int],
        remote_request_ids: list[int],
        cache_offsets: list[int],
        storage_offsets: list[int],
        request_lengths: list[int],
    ) -> None:
        token_position = int(token_start)
        destination_slot = int(destination_slot_start)
        remaining = int(token_count)
        while remaining > 0:
            destination_logical_block = destination_slot // block_size
            destination_block_offset = destination_slot % block_size
            if destination_logical_block >= len(destination_block_ids):
                raise RuntimeError(
                    "DSA KVIO P/D destination block table is too short: "
                    f"slot={destination_slot}, blocks="
                    f"{len(destination_block_ids)}, block_size={block_size}")
            transfer_tokens = min(
                remaining, block_size - destination_block_offset)
            physical_block = int(
                destination_block_ids[destination_logical_block])
            cache_ids.append(region.cache_id)
            remote_request_ids.append(int(remote_request_id))
            cache_offsets.append(
                physical_block * region.block_bytes
                + destination_block_offset * region.token_bytes)
            storage_offsets.append(
                region.storage_base + token_position * region.token_bytes)
            request_lengths.append(transfer_tokens * region.token_bytes)
            token_position += transfer_tokens
            destination_slot += transfer_tokens
            remaining -= transfer_tokens

    @classmethod
    def _append_token_transfers(
        cls,
        *,
        region: _KVIOCacheRegion,
        remote_request_id: int,
        token_ids: list[int],
        destination_block_ids: list[int],
        block_size: int,
        cache_ids: list[int],
        remote_request_ids: list[int],
        cache_offsets: list[int],
        storage_offsets: list[int],
        request_lengths: list[int],
    ) -> None:
        """Append arbitrary token reads, coalescing adjacent source tokens."""
        if not token_ids:
            return
        run_token_start = int(token_ids[0])
        run_slot_start = 0
        run_count = 1
        previous_token = run_token_start
        for destination_slot, raw_token_id in enumerate(token_ids[1:], 1):
            token_id = int(raw_token_id)
            if token_id == previous_token + 1:
                run_count += 1
            else:
                cls._append_range_transfers(
                    region=region,
                    remote_request_id=remote_request_id,
                    token_start=run_token_start,
                    token_count=run_count,
                    destination_slot_start=run_slot_start,
                    destination_block_ids=destination_block_ids,
                    block_size=block_size,
                    cache_ids=cache_ids,
                    remote_request_ids=remote_request_ids,
                    cache_offsets=cache_offsets,
                    storage_offsets=storage_offsets,
                    request_lengths=request_lengths,
                )
                run_token_start = token_id
                run_slot_start = destination_slot
                run_count = 1
            previous_token = token_id
        cls._append_range_transfers(
            region=region,
            remote_request_id=remote_request_id,
            token_start=run_token_start,
            token_count=run_count,
            destination_slot_start=run_slot_start,
            destination_block_ids=destination_block_ids,
            block_size=block_size,
            cache_ids=cache_ids,
            remote_request_ids=remote_request_ids,
            cache_offsets=cache_offsets,
            storage_offsets=storage_offsets,
            request_lengths=request_lengths,
        )

    def load_pd_request(
        self,
        *,
        request_pool_idx: int,
        stored_token_count: int,
        layer_resident_token_ids: dict[int, list[int]],
        tail_token_start: int,
        tail_token_count: int,
        tail_slot_start: int,
        indexer_block_ids: list[int],
        resident_block_ids: list[int],
    ) -> None:
        if not self._initialized:
            raise RuntimeError(
                "DSA KVIO P/D load requires finalized cache registration")
        stored_token_count = int(stored_token_count)
        if stored_token_count > self._max_model_len:
            raise ValueError(
                "DSA KVIO P/D stored token count exceeds max_model_len: "
                f"{stored_token_count} > {self._max_model_len}")
        remote_request_id = self._remote_request_id(int(request_pool_idx))
        expected_layer_ids = set(self._regions)
        resident_layer_ids = {
            int(layer_id) for layer_id in layer_resident_token_ids
        }
        if resident_layer_ids != expected_layer_ids:
            raise RuntimeError(
                "DSA KVIO P/D layer TopK does not match registered local "
                f"layers: expected={sorted(expected_layer_ids)}, "
                f"actual={sorted(resident_layer_ids)}"
            )

        for layer_id in sorted(self._regions):
            regions = self._regions[layer_id]
            resident_token_ids = [
                int(token_id)
                for token_id in layer_resident_token_ids[layer_id]
            ]
            if not resident_token_ids:
                raise RuntimeError(
                    f"DSA KVIO P/D layer {layer_id} has no resident tokens"
                )
            if len(set(resident_token_ids)) != len(resident_token_ids):
                raise RuntimeError(
                    f"DSA KVIO P/D layer {layer_id} has duplicate tokens"
                )
            if (
                min(resident_token_ids) < 0
                or max(resident_token_ids) >= int(tail_token_start)
            ):
                raise RuntimeError(
                    "DSA KVIO P/D resident tokens must precede the dense "
                    f"tail: layer={layer_id}, tail_start={tail_token_start}"
                )
            if regions.indexer is None:
                raise RuntimeError(
                    "DSA KVIO P/D requires the Indexer cache to be "
                    f"registered for layer {layer_id}")
            block_size = self._registered_caches[layer_id][0]
            cache_ids: list[int] = []
            remote_request_ids: list[int] = []
            cache_offsets: list[int] = []
            storage_offsets: list[int] = []
            request_lengths: list[int] = []

            self._append_range_transfers(
                region=regions.indexer,
                remote_request_id=remote_request_id,
                token_start=0,
                token_count=stored_token_count,
                destination_slot_start=0,
                destination_block_ids=indexer_block_ids,
                block_size=block_size,
                cache_ids=cache_ids,
                remote_request_ids=remote_request_ids,
                cache_offsets=cache_offsets,
                storage_offsets=storage_offsets,
                request_lengths=request_lengths,
            )
            for region in (regions.nopek, regions.ropek):
                self._append_token_transfers(
                    region=region,
                    remote_request_id=remote_request_id,
                    token_ids=resident_token_ids,
                    destination_block_ids=resident_block_ids,
                    block_size=block_size,
                    cache_ids=cache_ids,
                    remote_request_ids=remote_request_ids,
                    cache_offsets=cache_offsets,
                    storage_offsets=storage_offsets,
                    request_lengths=request_lengths,
                )
                self._append_range_transfers(
                    region=region,
                    remote_request_id=remote_request_id,
                    token_start=tail_token_start,
                    token_count=tail_token_count,
                    destination_slot_start=tail_slot_start,
                    destination_block_ids=resident_block_ids,
                    block_size=block_size,
                    cache_ids=cache_ids,
                    remote_request_ids=remote_request_ids,
                    cache_offsets=cache_offsets,
                    storage_offsets=storage_offsets,
                    request_lengths=request_lengths,
                )

            if not cache_ids:
                continue
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
                "DSA KVIO P/D load completed: layer=%d, transfers=%d, "
                "kernel_us=%s",
                layer_id,
                len(cache_ids),
                kernel_us,
            )

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
