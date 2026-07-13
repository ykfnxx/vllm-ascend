"""DSA 稀疏卸载的 worker 本地 DRAM 热层 KV store core。

本文件定义 DSAHotKVStore 及其块类型枚举，维护 HBM 已卸载满块到 worker
本地 DRAM 后的逻辑块映射关系，包括 NOPE_K/ROPE_K 两类 cache plane 的
块分配、块表查询和 host/device 可见元数据构造。

这里不负责 Ascend swapped-memory arena 的具体分配，也不负责 scheduler
侧 HBM block admission；设备相关初始化入口放在 dsa_ascend_hot_kv_store.py。
"""

import collections
import enum
from dataclasses import dataclass, field

import torch

# DSA hot DRAM is worker-local: the scheduler only owns HBM block
# allocation, while each worker maintains its own logical-block-to-DRAM-pool
# mapping for sparse decode copy plans.

from vllm.logger import init_logger
logger = init_logger(__name__)


class BlockType(enum.Enum):
    NOPE_K = enum.auto()
    ROPE_K = enum.auto()


@dataclass
class _ArenaPoolState:
    hash_to_pool_idx: dict = field(default_factory=dict)
    pool_idx_to_hash: dict[int, object] = field(default_factory=dict)
    pool_ref_counts: dict[int, int] = field(default_factory=dict)
    free_block_ids: list[int] = field(default_factory=list)
    arena: torch.Tensor | None = None


_DRAM_NULL_BLOCK_ID = 0


class DSAHotKVStore:
    def __init__(self):
        # Cache payloads mirror DeepSeek MLA's physical split:
        # layer -> NOPE_K arena and layer -> ROPE_K arena. Request-local
        # logical block tables are stored separately as dense tensors.
        self.block_pools = {
            BlockType.NOPE_K: collections.defaultdict(_ArenaPoolState),
            BlockType.ROPE_K: collections.defaultdict(_ArenaPoolState),
        }
        self._request_to_pool_idx: dict = {}
        self._pool_idx_to_request: dict[int, object] = {}
        self._next_internal_pool_idx = 0
        self._dram_block_table: torch.Tensor | None = None
        self._dram_block_table_version = 0
        self._dram_block_table_device_cache: dict[
            tuple[str, str], tuple[int, torch.Tensor]] = {}
        self._dram_block_ready_tables: dict[int, torch.Tensor] = {}
        self._dram_block_ready_versions: dict[int, int] = {}
        self._dram_block_ready_device_cache: dict[
            tuple[int, str], tuple[int, torch.Tensor]] = {}
        self._request_owned_blocks = collections.defaultdict(set)

    @staticmethod
    def _maybe_pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu" or tensor.is_pinned():
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    @classmethod
    def _allocate_host_arena(cls, block_shape: tuple[int, ...],
                             dtype: torch.dtype,
                             capacity: int) -> torch.Tensor:
        arena = torch.empty((capacity, *block_shape),
                            dtype=dtype,
                            device="cpu")
        return cls._maybe_pin_memory(arena.contiguous())

    @classmethod
    def _to_pinned_cpu_blocks(cls, gpu_blocks: torch.Tensor) -> torch.Tensor:
        cpu_blocks = gpu_blocks.detach().clone().to(device="cpu")
        return cls._maybe_pin_memory(cpu_blocks.contiguous())

    @classmethod
    def _ensure_arena_capacity(cls, pool_state: _ArenaPoolState,
                               block_shape: tuple[int, ...],
                               dtype: torch.dtype,
                               min_capacity: int) -> None:
        current_capacity = 0 if pool_state.arena is None else int(pool_state.arena.shape[0])
        if current_capacity >= min_capacity:
            return

        new_capacity = max(min_capacity, current_capacity * 2, 1)
        new_arena = cls._allocate_host_arena(
            block_shape=block_shape,
            dtype=dtype,
            capacity=new_capacity,
        )
        if pool_state.arena is not None and current_capacity > 0:
            new_arena[:current_capacity].copy_(pool_state.arena)
        pool_state.arena = new_arena
        first_new_block = max(current_capacity, _DRAM_NULL_BLOCK_ID + 1)
        pool_state.free_block_ids.extend(range(first_new_block, new_capacity))

    def preallocate_layer_cache(
            self,
            layer_id: int,
            blk_type: BlockType,
            block_shape: tuple[int, ...],
            dtype: torch.dtype,
            num_blocks: int,
            *,
            max_request_rows: int | None = None,
            max_logical_blocks: int | None = None) -> None:
        """Preallocate a worker-local DRAM hot cache arena for one layer/type."""
        if num_blocks <= 0:
            return
        pool_state = self.block_pools[blk_type][layer_id]
        current_capacity = (
            0 if pool_state.arena is None else int(pool_state.arena.shape[0]))
        # Capacity includes block 0, which is reserved as the IO null block.
        # Treat num_blocks as usable cache blocks to match HBM block-table
        # semantics and to keep block id 0 available for padding/invalid.
        required_capacity = int(num_blocks) + 1
        if current_capacity < required_capacity:
            self._ensure_arena_capacity(
                pool_state=pool_state,
                block_shape=tuple(block_shape),
                dtype=dtype,
                min_capacity=required_capacity,
            )
        if max_request_rows is not None and max_logical_blocks is not None:
            self._ensure_dram_block_table_capacity(
                min_rows=int(max_request_rows),
                min_logical_blocks=int(max_logical_blocks),
            )

    @staticmethod
    def _request_table_key(
            request_id,
            layer_id: int,
            blk_type: BlockType):
        return (request_id, int(layer_id), blk_type)

    def _bump_dram_block_table_version(self) -> None:
        self._dram_block_table_version += 1
        self._dram_block_table_device_cache.clear()

    def _bump_dram_block_ready_version(self, layer_id: int) -> None:
        layer_id = int(layer_id)
        self._dram_block_ready_versions[layer_id] = (
            int(self._dram_block_ready_versions.get(layer_id, 0)) + 1)
        for cache_key in list(self._dram_block_ready_device_cache.keys()):
            if int(cache_key[0]) == layer_id:
                self._dram_block_ready_device_cache.pop(cache_key, None)

    def _ensure_dram_block_table_capacity(
            self,
            min_rows: int,
            min_logical_blocks: int,
            *,
            dtype: torch.dtype = torch.long) -> torch.Tensor:
        min_rows = max(0, int(min_rows))
        min_logical_blocks = max(0, int(min_logical_blocks))
        current = self._dram_block_table
        if (current is not None and int(current.shape[0]) >= min_rows
                and int(current.shape[1]) >= min_logical_blocks):
            return current

        new_rows = max(min_rows,
                       0 if current is None else int(current.shape[0]) * 2,
                       1)
        new_width = max(
            min_logical_blocks,
            0 if current is None else int(current.shape[1]) * 2,
            1)
        new_table = torch.full((new_rows, new_width),
                               _DRAM_NULL_BLOCK_ID,
                               dtype=dtype,
                               device=torch.device("cpu"))
        if current is not None and int(current.numel()) > 0:
            rows = int(current.shape[0])
            cols = int(current.shape[1])
            new_table[:rows, :cols] = current.to(dtype=dtype)
        self._dram_block_table = new_table
        self._bump_dram_block_table_version()
        return new_table

    def _get_dram_pool_idx(self, request_pool_idx: int,
                           logical_block_idx: int) -> int | None:
        table = self._dram_block_table
        if table is None:
            return None
        request_pool_idx = int(request_pool_idx)
        logical_block_idx = int(logical_block_idx)
        if not (0 <= request_pool_idx < int(table.shape[0])):
            return None
        if not (0 <= logical_block_idx < int(table.shape[1])):
            return None
        pool_idx = int(table[request_pool_idx, logical_block_idx].item())
        return None if pool_idx <= _DRAM_NULL_BLOCK_ID else pool_idx

    def _ensure_dram_block_ready_capacity(
            self,
            layer_id: int,
            min_rows: int,
            min_logical_blocks: int) -> torch.Tensor:
        layer_id = int(layer_id)
        min_rows = max(0, int(min_rows))
        min_logical_blocks = max(0, int(min_logical_blocks))
        current = self._dram_block_ready_tables.get(layer_id)
        if (current is not None and int(current.shape[0]) >= min_rows
                and int(current.shape[1]) >= min_logical_blocks):
            return current

        new_rows = max(min_rows,
                       0 if current is None else int(current.shape[0]) * 2,
                       1)
        new_width = max(
            min_logical_blocks,
            0 if current is None else int(current.shape[1]) * 2,
            1)
        new_table = torch.zeros((new_rows, new_width),
                                dtype=torch.bool,
                                device=torch.device("cpu"))
        if current is not None and int(current.numel()) > 0:
            rows = int(current.shape[0])
            cols = int(current.shape[1])
            new_table[:rows, :cols] = current
        self._dram_block_ready_tables[layer_id] = new_table
        self._bump_dram_block_ready_version(layer_id)
        return new_table

    def _clear_pool_idx_tables(self, pool_idx: int) -> None:
        pool_idx = int(pool_idx)
        table = self._dram_block_table
        if table is not None and 0 <= pool_idx < int(table.shape[0]):
            table[pool_idx].fill_(_DRAM_NULL_BLOCK_ID)
            self._bump_dram_block_table_version()
        for layer_id, ready_table in self._dram_block_ready_tables.items():
            if 0 <= pool_idx < int(ready_table.shape[0]):
                ready_table[pool_idx].fill_(False)
                self._bump_dram_block_ready_version(layer_id)

    def bind_request_pool_index(self, request_id, pool_idx: int) -> None:
        pool_idx = int(pool_idx)
        old_pool_idx = self._request_to_pool_idx.get(request_id)
        if old_pool_idx == pool_idx:
            return

        old_request = self._pool_idx_to_request.get(pool_idx)
        if old_request is not None and old_request != request_id:
            self.release_request(old_request)

        if old_pool_idx is not None and int(old_pool_idx) != pool_idx:
            self._pool_idx_to_request.pop(int(old_pool_idx), None)
            self._clear_pool_idx_tables(int(old_pool_idx))

        self._request_to_pool_idx[request_id] = pool_idx
        self._pool_idx_to_request[pool_idx] = request_id
        self._next_internal_pool_idx = max(self._next_internal_pool_idx,
                                           pool_idx + 1)
        self._clear_pool_idx_tables(pool_idx)

    def release_request(self, request_id) -> None:
        pool_idx = self._request_to_pool_idx.pop(request_id, None)
        if pool_idx is None:
            return
        pool_idx = int(pool_idx)
        self._pool_idx_to_request.pop(pool_idx, None)
        self._clear_pool_idx_tables(pool_idx)
        for owned_key in list(self._request_owned_blocks.keys()):
            if owned_key[0] == request_id:
                _, layer_id, blk_type = owned_key
                self._release_request_block_refs_for_key(
                    request_id, int(layer_id), blk_type)

    def _get_request_pool_idx(self, request_id) -> int | None:
        pool_idx = self._request_to_pool_idx.get(request_id)
        return None if pool_idx is None else int(pool_idx)

    def _get_or_assign_request_pool_idx(self, request_id) -> int:
        pool_idx = self._get_request_pool_idx(request_id)
        if pool_idx is not None:
            return pool_idx
        while self._next_internal_pool_idx in self._pool_idx_to_request:
            self._next_internal_pool_idx += 1
        pool_idx = self._next_internal_pool_idx
        self._next_internal_pool_idx += 1
        self._request_to_pool_idx[request_id] = pool_idx
        self._pool_idx_to_request[pool_idx] = request_id
        return pool_idx

    def _request_owned_key(self, request_id, layer_id: int,
                           blk_type: BlockType):
        return self._request_table_key(request_id, layer_id, blk_type)

    def _add_request_block_ref(self, request_id, layer_id: int,
                               blk_type: BlockType, pool_idx: int,
                               blk_hash) -> None:
        owned_key = self._request_owned_key(request_id, layer_id, blk_type)
        pool_idx = int(pool_idx)
        owned_blocks = getattr(self, "_request_owned_blocks", None)
        if owned_blocks is None:
            self._request_owned_blocks = collections.defaultdict(set)
            owned_blocks = self._request_owned_blocks
        if pool_idx in owned_blocks[owned_key]:
            return

        pool_state = self.block_pools[blk_type][layer_id]
        owned_blocks[owned_key].add(pool_idx)
        pool_state.pool_ref_counts[pool_idx] = (
            int(pool_state.pool_ref_counts.get(pool_idx, 0)) + 1)
        if blk_hash is not None:
            pool_state.pool_idx_to_hash[pool_idx] = blk_hash

    def _release_pool_block_ref(self, layer_id: int, blk_type: BlockType,
                                pool_idx: int) -> None:
        pool_state = self.block_pools[blk_type][layer_id]
        pool_idx = int(pool_idx)
        ref_count = int(pool_state.pool_ref_counts.get(pool_idx, 0)) - 1
        if ref_count > 0:
            pool_state.pool_ref_counts[pool_idx] = ref_count
            return

        pool_state.pool_ref_counts.pop(pool_idx, None)
        blk_hash = pool_state.pool_idx_to_hash.pop(pool_idx, None)
        if blk_hash is not None:
            current_pool_idx = pool_state.hash_to_pool_idx.get(blk_hash)
            if current_pool_idx == pool_idx:
                pool_state.hash_to_pool_idx.pop(blk_hash, None)
        if pool_idx not in pool_state.free_block_ids:
            pool_state.free_block_ids.append(pool_idx)
            pool_state.free_block_ids.sort()

    def _release_request_block_refs_for_key(self, request_id, layer_id: int,
                                            blk_type: BlockType) -> None:
        owned_key = self._request_owned_key(request_id, layer_id, blk_type)
        owned_blocks = getattr(self, "_request_owned_blocks", None)
        if not owned_blocks:
            return
        for pool_idx in list(owned_blocks.pop(owned_key, set())):
            self._release_pool_block_ref(layer_id, blk_type, int(pool_idx))

    def _release_request_shared_block_ref(self, request_id, layer_id: int,
                                          pool_idx: int) -> None:
        for blk_type in (BlockType.NOPE_K, BlockType.ROPE_K):
            owned_key = self._request_owned_key(request_id, layer_id, blk_type)
            owned_blocks = getattr(self, "_request_owned_blocks", None)
            owned_set = None if not owned_blocks else owned_blocks.get(
                owned_key)
            if not owned_set or pool_idx not in owned_set:
                continue
            owned_set.remove(pool_idx)
            self._release_pool_block_ref(layer_id, blk_type, int(pool_idx))

    def _allocate_pool_block(self, pool_state: _ArenaPoolState,
                             block_shape: tuple[int, ...],
                             dtype: torch.dtype) -> int:
        if not pool_state.free_block_ids:
            current_capacity = (
                0 if pool_state.arena is None else int(pool_state.arena.shape[0]))
            self._ensure_arena_capacity(
                pool_state=pool_state,
                block_shape=block_shape,
                dtype=dtype,
                min_capacity=max(current_capacity + 1,
                                 _DRAM_NULL_BLOCK_ID + 2),
            )
        if not pool_state.free_block_ids:
            raise RuntimeError("DSA hot DRAM cache has no free block")
        return int(pool_state.free_block_ids.pop(0))

    def _reserve_pool_block(
        self,
        pool_state: _ArenaPoolState,
        pool_idx: int,
        block_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        pool_idx = int(pool_idx)
        if pool_idx == _DRAM_NULL_BLOCK_ID:
            raise RuntimeError("DSA hot DRAM block 0 is reserved as null")
        current_capacity = (
            0 if pool_state.arena is None else int(pool_state.arena.shape[0]))
        if current_capacity <= pool_idx:
            self._ensure_arena_capacity(
                pool_state=pool_state,
                block_shape=block_shape,
                dtype=dtype,
                min_capacity=pool_idx + 1,
            )
        if pool_idx in pool_state.free_block_ids:
            pool_state.free_block_ids.remove(pool_idx)

    def get_arena(self, layer_id: int, blk_type: BlockType) -> torch.Tensor:
        pool_state = self.block_pools[blk_type][layer_id]
        if pool_state.arena is None:
            raise ValueError("DRAM arena is not initialized.")
        return pool_state.arena

    def get_dram_block_table_tensor(
            self,
            num_logical_blocks: int | None = None,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype = torch.long) -> torch.Tensor:
        target_device = (
            torch.device("cpu") if device is None else torch.device(device))
        table = self._dram_block_table
        if table is None:
            width = 0 if num_logical_blocks is None else max(
                0, int(num_logical_blocks))
            table = self._ensure_dram_block_table_capacity(
                min_rows=1,
                min_logical_blocks=width,
                dtype=dtype,
            )
        elif (num_logical_blocks is not None
              and int(table.shape[1]) < int(num_logical_blocks)):
            table = self._ensure_dram_block_table_capacity(
                min_rows=int(table.shape[0]),
                min_logical_blocks=int(num_logical_blocks),
                dtype=dtype,
            )
        if table.dtype != dtype:
            table = table.to(dtype=dtype)
        if target_device.type == "cpu":
            result = table
        else:
            cache_key = (str(target_device), str(dtype))
            cached = self._dram_block_table_device_cache.get(cache_key)
            if (cached is not None
                    and int(cached[0]) == self._dram_block_table_version):
                result = cached[1]
            else:
                result = table.to(device=target_device, non_blocking=True)
                self._dram_block_table_device_cache[cache_key] = (
                    self._dram_block_table_version, result)
        if num_logical_blocks is None:
            return result
        return result[:, :max(0, int(num_logical_blocks))]

    def get_dram_block_ready_tensor(
            self,
            layer_id: int,
            num_logical_blocks: int | None = None,
            *,
            device: torch.device | str | None = None) -> torch.Tensor:
        layer_id = int(layer_id)
        target_device = (
            torch.device("cpu") if device is None else torch.device(device))
        base_rows = 1
        base_width = 0
        if self._dram_block_table is not None:
            base_rows = max(base_rows, int(self._dram_block_table.shape[0]))
            base_width = int(self._dram_block_table.shape[1])
        if num_logical_blocks is not None:
            base_width = max(base_width, max(0, int(num_logical_blocks)))
        ready_table = self._ensure_dram_block_ready_capacity(
            layer_id=layer_id,
            min_rows=base_rows,
            min_logical_blocks=base_width,
        )
        if target_device.type == "cpu":
            result = ready_table
        else:
            cache_key = (layer_id, str(target_device))
            cached = self._dram_block_ready_device_cache.get(cache_key)
            version = int(self._dram_block_ready_versions.get(layer_id, 0))
            if cached is not None and int(cached[0]) == version:
                result = cached[1]
            else:
                result = ready_table.to(device=target_device,
                                        non_blocking=True)
                self._dram_block_ready_device_cache[cache_key] = (
                    version, result)
        if num_logical_blocks is None:
            return result
        return result[:, :max(0, int(num_logical_blocks))]

    def dump_layer_blocks_for_requests(
            self,
            *,
            layer_id: int,
            request_ids: list,
            request_pool_indices: list[int],
            block_hash_rows: list[list] | list[None],
            block_id_rows: list[list[int]],
            logical_block_index_rows: list[list[int]],
            nopek_dev_cache_zone: torch.Tensor,
            ropek_dev_cache_zone: torch.Tensor) -> None:
        row_count = len(request_ids)
        if not (
            row_count == len(request_pool_indices)
            == len(block_hash_rows)
            == len(block_id_rows)
            == len(logical_block_index_rows)
        ):
            raise ValueError(
                "DSA layer dump rows must have matching request, pool, "
                "hash, block, and logical-index lengths")

        nopek_pool = self.block_pools[BlockType.NOPE_K][layer_id]
        ropek_pool = self.block_pools[BlockType.ROPE_K][layer_id]
        dram_block_table_changed = False
        ready_table_changed = False

        for (request_id, request_pool_idx, block_hashes, block_ids,
             logical_block_indices) in zip(
                 request_ids,
                 request_pool_indices,
                 block_hash_rows,
                 block_id_rows,
                 logical_block_index_rows,
             ):
            if block_hashes is not None and len(block_ids) != len(block_hashes):
                raise AssertionError("IDs和Hashes数量必须一致")
            if len(block_ids) != len(logical_block_indices):
                raise AssertionError("IDs和Logical block数量必须一致")
            if not block_ids:
                continue

            self.bind_request_pool_index(request_id, int(request_pool_idx))
            request_table_row_idx = self._get_or_assign_request_pool_idx(
                request_id)
            nopek_blocks = self._to_pinned_cpu_blocks(
                nopek_dev_cache_zone[block_ids])
            ropek_blocks = self._to_pinned_cpu_blocks(
                ropek_dev_cache_zone[block_ids])
            nopek_shape = tuple(nopek_blocks.shape[1:])
            ropek_shape = tuple(ropek_blocks.shape[1:])

            for i, _ in enumerate(block_ids):
                blk_hash = None if block_hashes is None else block_hashes[i]
                logical_block_idx = int(logical_block_indices[i])
                pool_idx = self._get_dram_pool_idx(request_table_row_idx,
                                                   logical_block_idx)
                if pool_idx is None and blk_hash is not None:
                    pool_idx = nopek_pool.hash_to_pool_idx.get(blk_hash)
                if pool_idx is None and blk_hash is not None:
                    pool_idx = ropek_pool.hash_to_pool_idx.get(blk_hash)
                if pool_idx is None:
                    pool_idx = self._allocate_pool_block(
                        pool_state=nopek_pool,
                        block_shape=nopek_shape,
                        dtype=nopek_blocks.dtype,
                    )
                else:
                    self._reserve_pool_block(
                        pool_state=nopek_pool,
                        pool_idx=int(pool_idx),
                        block_shape=nopek_shape,
                        dtype=nopek_blocks.dtype,
                    )
                self._reserve_pool_block(
                    pool_state=ropek_pool,
                    pool_idx=int(pool_idx),
                    block_shape=ropek_shape,
                    dtype=ropek_blocks.dtype,
                )
                if blk_hash is not None:
                    nopek_pool.hash_to_pool_idx[blk_hash] = pool_idx
                    ropek_pool.hash_to_pool_idx[blk_hash] = pool_idx

                old_pool_idx = self._get_dram_pool_idx(
                    request_table_row_idx, logical_block_idx)
                if old_pool_idx != int(pool_idx):
                    if old_pool_idx is not None:
                        self._release_request_shared_block_ref(
                            request_id, layer_id, int(old_pool_idx))
                    logical_table = self._ensure_dram_block_table_capacity(
                        min_rows=request_table_row_idx + 1,
                        min_logical_blocks=logical_block_idx + 1,
                    )
                    logical_table[request_table_row_idx,
                                  logical_block_idx] = int(pool_idx)
                    dram_block_table_changed = True

                self._add_request_block_ref(
                    request_id=request_id,
                    layer_id=layer_id,
                    blk_type=BlockType.NOPE_K,
                    pool_idx=int(pool_idx),
                    blk_hash=blk_hash,
                )
                self._add_request_block_ref(
                    request_id=request_id,
                    layer_id=layer_id,
                    blk_type=BlockType.ROPE_K,
                    pool_idx=int(pool_idx),
                    blk_hash=blk_hash,
                )
                ready_table = self._ensure_dram_block_ready_capacity(
                    layer_id=layer_id,
                    min_rows=request_table_row_idx + 1,
                    min_logical_blocks=logical_block_idx + 1,
                )
                ready_table[request_table_row_idx,
                            logical_block_idx] = False
                nopek_pool.arena[pool_idx].copy_(nopek_blocks[i])
                ropek_pool.arena[pool_idx].copy_(ropek_blocks[i])
                ready_table[request_table_row_idx,
                            logical_block_idx] = True
                ready_table_changed = True

        if dram_block_table_changed:
            self._bump_dram_block_table_version()
        if ready_table_changed:
            self._bump_dram_block_ready_version(layer_id)
