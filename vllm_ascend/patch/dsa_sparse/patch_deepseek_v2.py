"""Use a dedicated KV cache spec for the DeepSeek-V3.2 indexer."""

from functools import wraps

from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.kv_cache_interface import IndexerKVSpec

from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)

_original_get_kv_cache_spec = DeepseekV32IndexerCache.get_kv_cache_spec


@wraps(_original_get_kv_cache_spec)
def _get_kv_cache_spec(self, vllm_config):
    attach_dsa_sparse_cache_attrs(vllm_config)
    if not is_dsa_sparse_config_enabled(vllm_config):
        return _original_get_kv_cache_spec(self, vllm_config)
    return IndexerKVSpec(
        block_size=self.cache_config.block_size,
        num_kv_heads=1,
        head_size=self.head_dim,
        dtype=self.dtype,
    )


DeepseekV32IndexerCache.get_kv_cache_spec = _get_kv_cache_spec
