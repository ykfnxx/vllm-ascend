"""Install DSA patches before vLLM v0.18 constructs a child EngineCore."""

import os

from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCoreProc
from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)

logger = init_logger(__name__)

_original_run_engine_core = EngineCoreProc.run_engine_core


# Keep this function's module identity so spawn imports the DSA patch module.
def _run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    vllm_config = kwargs["vllm_config"]
    attach_dsa_sparse_cache_attrs(vllm_config)
    if is_dsa_sparse_config_enabled(vllm_config):
        logger.info(
            "DSA sparse EngineCore entry patch active: pid=%d, dp_rank=%d, "
            "local_dp_rank=%d",
            os.getpid(),
            dp_rank,
            local_dp_rank,
        )
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
logger.info(
    "DSA sparse platform patch installed: EngineCoreProc.run_engine_core "
    "is wrapped, pid=%d",
    os.getpid(),
)
