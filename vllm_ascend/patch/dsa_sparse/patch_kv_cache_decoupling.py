"""Route vLLM v0.18 KV groups to independent physical block pools."""

import itertools
from collections.abc import Iterable, Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core import block_pool as block_pool_mod
from vllm.v1.core import kv_cache_coordinator as coordinator_mod
from vllm.v1.core import kv_cache_manager as manager_mod
from vllm.v1.core import single_type_kv_cache_manager as single_manager_mod
from vllm.v1.core.block_pool import AllBlocksCleared, BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.kv_cache_interface import (
    IndexerKVSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
)

from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)


class MultiBlockPool(BlockPool):
    """One block-id namespace and free queue per KV cache group."""

    def __init__(
        self,
        num_gpu_blocks: Sequence[int],
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        super().__init__(
            sum(num_gpu_blocks),
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
        self.block_pools = [
            BlockPool(
                num_blocks,
                enable_caching,
                hash_block_size,
                enable_kv_cache_events,
                metrics_collector,
            )
            for num_blocks in num_gpu_blocks
        ]
        self._bind_shared_cache_state()

    def _bind_shared_cache_state(self) -> None:
        for pool in self.block_pools:
            pool.cached_block_hash_to_block = self.cached_block_hash_to_block
            pool.kv_event_queue = self.kv_event_queue

    def get_new_blocks(
        self, num_blocks: int, pool_id: int
    ) -> list[KVCacheBlock]:
        return self.block_pools[pool_id].get_new_blocks(num_blocks)

    def touch(self, blocks: Sequence[KVCacheBlock], pool_id: int) -> None:
        self.block_pools[pool_id].touch(blocks)

    def free_blocks(
        self, ordered_blocks: Iterable[KVCacheBlock], pool_id: int
    ) -> None:
        self.block_pools[pool_id].free_blocks(ordered_blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        for pool in self.block_pools:
            pool.evict_blocks(
                {
                    block_id
                    for block_id in block_ids
                    if block_id < pool.num_gpu_blocks
                }
            )

    def get_num_free_blocks(self) -> int:
        return sum(pool.get_num_free_blocks() for pool in self.block_pools)

    def get_usage(self) -> float:
        total_blocks = sum(pool.num_gpu_blocks - 1 for pool in self.block_pools)
        free_blocks = sum(pool.get_num_free_blocks() for pool in self.block_pools)
        return 1.0 - free_blocks / total_blocks

    def reset_prefix_cache(self) -> bool:
        used_blocks = sum(
            pool.num_gpu_blocks - pool.get_num_free_blocks()
            for pool in self.block_pools
        )
        if used_blocks != len(self.block_pools):
            return False
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self._bind_shared_cache_state()
        for pool in self.block_pools:
            for block in pool.blocks:
                block.reset_hash()
        if self.metrics_collector:
            self.metrics_collector.reset()
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())
        return True

    def take_events(self) -> list:
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        self._bind_shared_cache_state()
        return events


def _get_new_blocks_from_pool(self, num_blocks: int) -> list[KVCacheBlock]:
    if isinstance(self.block_pool, MultiBlockPool):
        return self.block_pool.get_new_blocks(
            num_blocks, pool_id=self.kv_cache_group_id
        )
    return self.block_pool.get_new_blocks(num_blocks)


def _touch_blocks_in_pool(
    self, blocks: Sequence[KVCacheBlock]
) -> None:
    if isinstance(self.block_pool, MultiBlockPool):
        self.block_pool.touch(blocks, pool_id=self.kv_cache_group_id)
    else:
        self.block_pool.touch(blocks)


def _free_blocks_to_pool(
    self, ordered_blocks: Sequence[KVCacheBlock]
) -> None:
    if isinstance(self.block_pool, MultiBlockPool):
        self.block_pool.free_blocks(
            ordered_blocks, pool_id=self.kv_cache_group_id
        )
    else:
        self.block_pool.free_blocks(ordered_blocks)


def _allocate_new_computed_blocks(
    self,
    request_id: str,
    new_computed_blocks: Sequence[KVCacheBlock],
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None:
    if request_id in self.num_cached_block:
        assert len(new_computed_blocks) == 0
        return

    req_blocks = self.req_to_blocks[request_id]
    assert len(req_blocks) == 0
    total_computed_tokens = (
        num_local_computed_tokens + num_external_computed_tokens
    )
    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    num_skipped_blocks = num_skipped_tokens // self.block_size
    if num_skipped_blocks > 0:
        new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
        num_external_computed_tokens = min(
            total_computed_tokens - num_skipped_tokens,
            num_external_computed_tokens,
        )

    if self.enable_caching:
        self._touch_blocks_in_pool(new_computed_blocks)
    else:
        assert not any(new_computed_blocks)
    req_blocks.extend([self._null_block] * num_skipped_blocks)
    req_blocks.extend(new_computed_blocks)
    self.num_cached_block[request_id] = len(req_blocks)

    if num_external_computed_tokens > 0:
        blocks = self._get_new_blocks_from_pool(
            cdiv(total_computed_tokens, self.block_size) - len(req_blocks)
        )
        req_blocks.extend(blocks)
        if is_dsa_mla_resident_spec(self.kv_cache_spec):
            self.new_block_ids.extend(block.block_id for block in blocks)


def _allocate_new_blocks(
    self, request_id: str, num_tokens: int, num_tokens_main_model: int
) -> list[KVCacheBlock]:
    req_blocks = self.req_to_blocks[request_id]
    num_new_blocks = cdiv(num_tokens, self.block_size) - len(req_blocks)
    if num_new_blocks <= 0:
        return []
    blocks = self._get_new_blocks_from_pool(num_new_blocks)
    req_blocks.extend(blocks)
    if is_dsa_mla_resident_spec(self.kv_cache_spec):
        self.new_block_ids.extend(block.block_id for block in blocks)
    return blocks


def _free(self, request_id: str) -> None:
    req_blocks = self.req_to_blocks.pop(request_id, [])
    self._free_blocks_to_pool(list(reversed(req_blocks)))
    self.num_cached_block.pop(request_id, None)


def _remove_skipped_blocks(
    self, request_id: str, total_computed_tokens: int
) -> None:
    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    if num_skipped_tokens <= 0:
        return
    num_skipped_blocks = min(
        num_skipped_tokens // self.block_size,
        len(self.req_to_blocks[request_id]),
    )
    removed_blocks = []
    blocks = self.req_to_blocks[request_id]
    for index in range(num_skipped_blocks - 1, -1, -1):
        if blocks[index] == self._null_block:
            break
        removed_blocks.append(blocks[index])
        blocks[index] = self._null_block
    self._free_blocks_to_pool(removed_blocks)


class IndexerKVManager(single_manager_mod.SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert is_dsa_indexer_spec(kv_cache_spec)
        computed_blocks = tuple([] for _ in kv_cache_group_ids)
        block_size = kv_cache_spec.block_size * dcp_world_size * pcp_world_size
        for block_hash in itertools.islice(
            block_hashes, max_length // block_size
        ):
            cached_blocks = block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            )
            if not cached_blocks:
                break
            for computed, cached in zip(computed_blocks, cached_blocks):
                computed.append(cached)
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        while (
            block_size != alignment_tokens
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt != len(self.req_to_blocks):
                break
            num_common_blocks += 1
        return num_common_blocks


def _use_group_block_pools(kv_cache_config: KVCacheConfig) -> bool:
    return any(
        is_dsa_indexer_spec(group.kv_cache_spec)
        for group in kv_cache_config.kv_cache_groups
    )


def _coordinator_init(
    self,
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> None:
    self.kv_cache_config = kv_cache_config
    self.max_model_len = max_model_len
    self.enable_caching = enable_caching
    if self._use_group_block_pools(kv_cache_config):
        self.block_pool = MultiBlockPool(
            [group.dsa_num_blocks for group in kv_cache_config.kv_cache_groups],
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
    else:
        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
    self.use_eagle = use_eagle
    self.single_type_managers = tuple(
        single_manager_mod.get_manager_for_kv_cache_spec(
            kv_cache_spec=group.kv_cache_spec,
            block_pool=self.block_pool,
            enable_caching=enable_caching,
            kv_cache_group_id=group_id,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
        )
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
    )


def _get_num_blocks_to_allocate_by_group(
    self,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    num_encoder_tokens: int,
    total_computed_tokens: int,
    num_tokens_main_model: int,
) -> list[int]:
    result = []
    for group_id, manager in enumerate(self.single_type_managers):
        if isinstance(manager, single_manager_mod.CrossAttentionManager):
            result.append(
                manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    num_encoder_tokens,
                )
            )
        else:
            result.append(
                manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[group_id],
                    total_computed_tokens,
                    num_tokens_main_model,
                )
            )
    return result


def _get_num_blocks_to_allocate(self, *args, **kwargs) -> int:
    return sum(self.get_num_blocks_to_allocate_by_group(*args, **kwargs))


def _can_allocate_group_blocks(self, num_blocks: list[int]) -> bool:
    if isinstance(self.block_pool, MultiBlockPool):
        return all(
            required <= pool.get_num_free_blocks()
            for required, pool in zip(num_blocks, self.block_pool.block_pools)
        )
    return sum(num_blocks) <= self.block_pool.get_num_free_blocks()


def _allocate_slots(
    self,
    request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks=None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
):
    if num_new_tokens == 0 and num_external_computed_tokens == 0:
        raise ValueError(
            "num_new_tokens must be greater than 0 when there are no external computed tokens"
        )
    computed_blocks = (
        new_computed_blocks.blocks
        if new_computed_blocks is not None
        else self.empty_kv_cache_blocks.blocks
    )
    num_local_computed_tokens = (
        request.num_computed_tokens + num_new_computed_tokens
    )
    total_computed_tokens = min(
        num_local_computed_tokens + num_external_computed_tokens,
        self.max_model_len,
    )
    num_tokens_main_model = total_computed_tokens + num_new_tokens
    num_tokens_need_slot = min(
        num_tokens_main_model + num_lookahead_tokens, self.max_model_len
    )
    self.coordinator.remove_skipped_blocks(
        request.request_id, total_computed_tokens
    )
    blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate_by_group(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=computed_blocks,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=(
            num_local_computed_tokens + num_external_computed_tokens
        ),
        num_tokens_main_model=num_tokens_main_model,
    )
    if not self._can_allocate_group_blocks(blocks_to_allocate):
        return None
    if (
        computed_blocks is not self.empty_kv_cache_blocks.blocks
        or num_external_computed_tokens > 0
    ):
        self.coordinator.allocate_new_computed_blocks(
            request_id=request.request_id,
            new_computed_blocks=computed_blocks,
            num_local_computed_tokens=num_local_computed_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
        )
    new_blocks = self.coordinator.allocate_new_blocks(
        request.request_id,
        num_tokens_need_slot,
        num_tokens_main_model,
        num_encoder_tokens,
    )
    if not self.enable_caching or delay_cache_blocks:
        return self.create_kv_cache_blocks(new_blocks)
    self.coordinator.cache_blocks(
        request,
        min(total_computed_tokens + num_new_tokens, request.num_tokens),
    )
    return self.create_kv_cache_blocks(new_blocks)


def install_dsa_kv_cache_decoupling_patch() -> None:
    block_pool_mod.MultiBlockPool = MultiBlockPool
    single_manager = single_manager_mod.SingleTypeKVCacheManager
    single_manager._get_new_blocks_from_pool = _get_new_blocks_from_pool
    single_manager._touch_blocks_in_pool = _touch_blocks_in_pool
    single_manager._free_blocks_to_pool = _free_blocks_to_pool
    single_manager.allocate_new_computed_blocks = _allocate_new_computed_blocks
    single_manager.allocate_new_blocks = _allocate_new_blocks
    single_manager.free = _free
    single_manager.remove_skipped_blocks = _remove_skipped_blocks
    single_manager_mod.IndexerKVManager = IndexerKVManager
    single_manager_mod.spec_manager_map[IndexerKVSpec] = IndexerKVManager
    single_manager_mod.spec_manager_map[MLAAttentionSpec] = (
        single_manager_mod.FullAttentionManager
    )

    coordinator = coordinator_mod.KVCacheCoordinator
    coordinator.__init__ = _coordinator_init
    coordinator._use_group_block_pools = staticmethod(_use_group_block_pools)
    coordinator.get_num_blocks_to_allocate_by_group = (
        _get_num_blocks_to_allocate_by_group
    )
    coordinator.get_num_blocks_to_allocate = _get_num_blocks_to_allocate
    coordinator_mod.MultiBlockPool = MultiBlockPool

    manager_mod.KVCacheManager._can_allocate_group_blocks = (
        _can_allocate_group_blocks
    )
    manager_mod.KVCacheManager.allocate_slots = _allocate_slots


install_dsa_kv_cache_decoupling_patch()
