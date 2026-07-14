"""DSA 稀疏卸载热层 KV store 的 Ascend 设备化入口。

本文件负责根据 vLLM 配置、KV cache spec 和用户设置估算 DSA 所需的
worker 本地 DRAM 热层规模，并创建/注册 AscendDSAHotKVStore。这里的
“hot”指推理生命周期内会被 token-wise 换入的 MLA cache 热数据；底层
block/table/ref-count 管理由 dsa_hot_kv_store_core.py 中的 DSAHotKVStore
提供。
"""

from __future__ import annotations

import math

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm_ascend.dsa_sparse.dsa_hot_kv_store_core import (
    BlockType,
    DSAHotKVStore,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)

logger = init_logger(__name__)


class AscendDSAHotKVStore(DSAHotKVStore):
    """Worker-local DRAM store for DSA sparse decode.

    This store is independent from vLLM-Ascend's CPUOffloadingConnector.  It
    owns per-rank NPU-visible swapped DRAM arenas and logical DRAM block
    tables used by DSA to stage MLA cache blocks before selected tokens are
    materialized back to HBM.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.vllm_config = vllm_config

    @classmethod
    def _allocate_host_arena(cls, block_shape: tuple[int, ...],
                             dtype: torch.dtype,
                             capacity: int) -> torch.Tensor:
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError(
                "DSA lookup materialization requires torch_npu swapped-memory "
                "DRAM arenas on Ascend") from exc

        device_index = torch.npu.current_device()
        arena = torch_npu.empty_with_swapped_memory(
            (int(capacity), *tuple(block_shape)),
            dtype=dtype,
            device=torch.device(f"npu:{device_index}"),
        )
        if not arena.is_contiguous():
            raise RuntimeError(
                "torch_npu.empty_with_swapped_memory must return a "
                "contiguous tensor for DSA DRAM arenas")
        return arena

    @classmethod
    def _to_pinned_cpu_blocks(cls, gpu_blocks: torch.Tensor) -> torch.Tensor:
        blocks = gpu_blocks.detach()
        return blocks if blocks.is_contiguous() else blocks.contiguous()

    @staticmethod
    def _layer_id_from_name(layer_name: str) -> int:
        return int(layer_name.split(".")[2])

    def initialize_hot_cache_from_kv_caches(self, kv_caches: dict,
                                            kv_cache_config) -> None:
        """Preallocate request-lifetime DRAM arenas for DSA sparse decode."""
        cache_config = self.vllm_config.cache_config
        if not cache_config.enable_dsa_sparse_cache:
            return

        spec_by_layer = {
            layer_name: group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
        }
        indexer_num_blocks = 0
        for layer_name, cache in kv_caches.items():
            spec = spec_by_layer[layer_name]
            if is_dsa_indexer_spec(spec) and torch.is_tensor(cache):
                indexer_num_blocks = max(indexer_num_blocks,
                                         int(cache.shape[0]))
        if indexer_num_blocks <= 0:
            raise RuntimeError(
                "DSA hot cache initialization requires an IndexerKVSpec "
                "cache tensor")

        multiple = int(cache_config.dsa_hot_cpu_block_multiple)
        hot_num_blocks = indexer_num_blocks * max(1, multiple)
        block_size = int(cache_config.block_size)
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        max_logical_blocks = max(1, math.ceil(max_model_len / block_size) + 1)
        max_request_rows = int(cache_config.dsa_max_active_reqs)

        initialized_layers: list[int] = []
        for layer_name, cache in kv_caches.items():
            spec = spec_by_layer[layer_name]
            if (not isinstance(spec, AttentionSpec)
                    or not is_dsa_mla_resident_spec(spec)):
                continue
            if not isinstance(cache, (tuple, list)) or len(cache) < 2:
                raise RuntimeError(
                    f"DSA MLA cache for {layer_name} must contain noPE and "
                    "RoPE tensors")
            nopek_cache, ropek_cache = cache[0], cache[1]
            if not torch.is_tensor(nopek_cache) or not torch.is_tensor(
                    ropek_cache):
                raise RuntimeError(
                    f"DSA MLA cache for {layer_name} must contain tensors")
            layer_id = self._layer_id_from_name(layer_name)
            self.preallocate_layer_cache(
                layer_id=layer_id,
                blk_type=BlockType.NOPE_K,
                block_shape=tuple(nopek_cache.shape[1:]),
                dtype=nopek_cache.dtype,
                num_blocks=hot_num_blocks,
                max_request_rows=max_request_rows,
                max_logical_blocks=max_logical_blocks,
            )
            self.preallocate_layer_cache(
                layer_id=layer_id,
                blk_type=BlockType.ROPE_K,
                block_shape=tuple(ropek_cache.shape[1:]),
                dtype=ropek_cache.dtype,
                num_blocks=hot_num_blocks,
                max_request_rows=max_request_rows,
                max_logical_blocks=max_logical_blocks,
            )
            initialized_layers.append(layer_id)

        if initialized_layers:
            sample_arena = self.get_arena(initialized_layers[0],
                                          BlockType.NOPE_K)
            logger.debug(
                "Initialized DSA hot DRAM cache: layers=%d, "
                "hot_blocks_per_layer_type=%d, request_rows=%d, "
                "logical_blocks=%d, arena_device=%s",
                len(set(initialized_layers)), hot_num_blocks,
                max_request_rows, max_logical_blocks, sample_arena.device)


def create_dsa_hot_kv_store(vllm_config: VllmConfig) -> AscendDSAHotKVStore:
    return AscendDSAHotKVStore(vllm_config)
