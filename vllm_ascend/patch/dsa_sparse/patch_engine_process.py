"""Install DSA patches before vLLM v0.18 constructs a child EngineCore."""

from functools import wraps

from vllm.v1.engine.core import EngineCoreProc
from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)

_original_run_engine_core = EngineCoreProc.run_engine_core


@wraps(_original_run_engine_core)
def _run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    vllm_config = kwargs["vllm_config"]
    attach_dsa_sparse_cache_attrs(vllm_config)
    if is_dsa_sparse_config_enabled(vllm_config):
        from vllm_ascend.patch.dsa_sparse.patch_runtime import (
            install_dsa_runtime_patches,
        )

        install_dsa_runtime_patches()
    return _original_run_engine_core(
        *args,
        dp_rank=dp_rank,
        local_dp_rank=local_dp_rank,
        **kwargs,
    )


EngineCoreProc.run_engine_core = staticmethod(_run_engine_core)
