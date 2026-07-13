"""Bind DSA KV planning and block hashing into vLLM v0.18 EngineCore."""

from functools import wraps

import vllm.v1.engine.core as engine_core_mod
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine.core import EngineCore

from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)

_original_init = EngineCore.__init__


@wraps(_original_init)
def _engine_core_init(self: EngineCore, vllm_config, *args, **kwargs) -> None:
    attach_dsa_sparse_cache_attrs(vllm_config)
    engine_core_mod.get_kv_cache_configs = kv_cache_utils.get_kv_cache_configs
    _original_init(self, vllm_config, *args, **kwargs)
    if (
        is_dsa_sparse_config_enabled(vllm_config)
        and self.request_block_hasher is None
    ):
        block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )
        hash_fn = get_hash_fn_by_name(
            vllm_config.cache_config.prefix_caching_hash_algo
        )
        init_none_hash(hash_fn)
        self.request_block_hasher = get_request_block_hasher(
            block_size, hash_fn
        )


def install_dsa_engine_core_patches() -> None:
    engine_core_mod.get_kv_cache_configs = kv_cache_utils.get_kv_cache_configs
    EngineCore.__init__ = _engine_core_init


install_dsa_engine_core_patches()
