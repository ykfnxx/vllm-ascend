"""Install DSA patches before vLLM v0.18 constructs a child EngineCore."""

from __future__ import annotations

import os

import vllm.v1.engine.core as engine_core_mod
import vllm.v1.engine.core_client as core_client_mod
from vllm.logger import init_logger
from vllm.v1.core import kv_cache_utils as kv_utils
from vllm.v1.core.sched import output as output_mod
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import utils as engine_utils
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.request import Request

from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)

logger = init_logger(__name__)

_DSA_RUN_ENGINE_CORE_WRAPPER_ATTR = (
    "_vllm_ascend_dsa_run_engine_core_wrapper"
)


def _is_dsa_enabled_on_config(vllm_config) -> bool:
    if vllm_config is None:
        return False
    attach_dsa_sparse_cache_attrs(vllm_config)
    return is_dsa_sparse_config_enabled(vllm_config)


def _get_manager_vllm_config(args, kwargs):
    if "vllm_config" in kwargs:
        return kwargs["vllm_config"]
    if len(args) >= 4:
        return args[3]
    return None


def _get_launch_core_vllm_config(args, kwargs):
    if "vllm_config" in kwargs:
        return kwargs["vllm_config"]
    if args:
        return args[0]
    return None


def _get_make_client_vllm_config(args, kwargs):
    if "vllm_config" in kwargs:
        return kwargs["vllm_config"]
    if len(args) >= 3:
        return args[2]
    return None


def _install_dsa_runtime_patches() -> None:
    from vllm_ascend.patch.dsa_sparse.patch_runtime import (
        install_dsa_runtime_patches,
    )

    install_dsa_runtime_patches()


def is_dsa_run_engine_core_wrapper(fn) -> bool:
    return bool(getattr(fn, _DSA_RUN_ENGINE_CORE_WRAPPER_ATTR, False))


def verify_dsa_runtime_patches_installed() -> None:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    from vllm_ascend.patch.dsa_sparse import patch_deepseek_v2
    from vllm_ascend.patch.dsa_sparse import patch_engine_core
    from vllm_ascend.patch.dsa_sparse import patch_kv_cache_utils
    from vllm_ascend.patch.dsa_sparse import patch_request
    from vllm_ascend.patch.dsa_sparse import patch_scheduler
    from vllm_ascend.patch.dsa_sparse import patch_scheduler_output

    checks = {
        "engine_core_proc_entrypoint": is_dsa_run_engine_core_wrapper(
            EngineCoreProc.run_engine_core
        ),
        "kv_cache_configs": (
            kv_utils.get_kv_cache_configs
            is patch_kv_cache_utils._get_kv_cache_configs
        ),
        "engine_core_kv_cache_configs_alias": (
            engine_core_mod.get_kv_cache_configs
            is kv_utils.get_kv_cache_configs
        ),
        "engine_core_init": (
            EngineCore.__init__ is patch_engine_core._engine_core_init
        ),
        "scheduler_init": Scheduler.__init__ is patch_scheduler._scheduler_init,
        "scheduler_schedule": Scheduler.schedule is patch_scheduler._schedule,
        "request_init": Request.__init__ is patch_request._request_init,
        "scheduler_output": (
            output_mod.SchedulerOutput
            is patch_scheduler_output.SchedulerOutput
        ),
        "indexer_cache_spec": (
            DeepseekV32IndexerCache.get_kv_cache_spec
            is patch_deepseek_v2._get_kv_cache_spec
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "DSA sparse runtime patches are incomplete before EngineCore "
            f"startup: {checks}"
        )


def _prepare_dsa_engine_bootstrap(vllm_config, boundary: str) -> bool:
    if not _is_dsa_enabled_on_config(vllm_config):
        return False
    _install_dsa_runtime_patches()
    ensure_dsa_engine_core_entrypoint()
    verify_dsa_runtime_patches_installed()
    logger.info(
        "DSA sparse EngineCore bootstrap verified: boundary=%s, pid=%d",
        boundary,
        os.getpid(),
    )
    return True


def _dsa_sparse_run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    vllm_config = kwargs.get("vllm_config")
    if _prepare_dsa_engine_bootstrap(vllm_config, "run_engine_core"):
        logger.info(
            "DSA sparse EngineCore entry patch active: pid=%d, dp_rank=%d, "
            "local_dp_rank=%d",
            os.getpid(),
            dp_rank,
            local_dp_rank,
        )
    original_run_engine_core = (
        EngineCoreProc._dsa_sparse_original_run_engine_core
    )
    return original_run_engine_core(
        *args,
        dp_rank=dp_rank,
        local_dp_rank=local_dp_rank,
        **kwargs,
    )


setattr(
    _dsa_sparse_run_engine_core,
    _DSA_RUN_ENGINE_CORE_WRAPPER_ATTR,
    True,
)


def ensure_dsa_engine_core_entrypoint() -> None:
    current_run_engine_core = EngineCoreProc.run_engine_core
    if is_dsa_run_engine_core_wrapper(current_run_engine_core):
        return
    EngineCoreProc._dsa_sparse_original_run_engine_core = (
        current_run_engine_core
    )
    EngineCoreProc.run_engine_core = staticmethod(
        _dsa_sparse_run_engine_core
    )
    logger.info(
        "DSA sparse platform patch installed: EngineCoreProc.run_engine_core "
        "is wrapped, pid=%d",
        os.getpid(),
    )


ensure_dsa_engine_core_entrypoint()


if not getattr(EngineCoreClient, "_dsa_sparse_make_client_patched", False):
    _original_make_client = EngineCoreClient.make_client

    def _dsa_sparse_make_client(*args, **kwargs):
        _prepare_dsa_engine_bootstrap(
            _get_make_client_vllm_config(args, kwargs),
            "make_client",
        )
        return _original_make_client(*args, **kwargs)

    EngineCoreClient.make_client = staticmethod(_dsa_sparse_make_client)
    EngineCoreClient._dsa_sparse_make_client_patched = True


if not getattr(
    engine_utils.CoreEngineProcManager,
    "_dsa_sparse_engine_proc_manager_init_patched",
    False,
):
    _original_core_engine_proc_manager_init = (
        engine_utils.CoreEngineProcManager.__init__
    )

    def _dsa_sparse_core_engine_proc_manager_init(
        self, *args, **kwargs
    ):
        _prepare_dsa_engine_bootstrap(
            _get_manager_vllm_config(args, kwargs),
            "CoreEngineProcManager.__init__",
        )
        return _original_core_engine_proc_manager_init(
            self, *args, **kwargs
        )

    engine_utils.CoreEngineProcManager.__init__ = (
        _dsa_sparse_core_engine_proc_manager_init
    )
    setattr(
        engine_utils.CoreEngineProcManager,
        "_dsa_sparse_engine_proc_manager_init_patched",
        True,
    )


if not getattr(engine_utils, "_dsa_sparse_launch_core_engines_patched", False):
    _original_launch_core_engines = engine_utils.launch_core_engines

    def _dsa_sparse_launch_core_engines(*args, **kwargs):
        _prepare_dsa_engine_bootstrap(
            _get_launch_core_vllm_config(args, kwargs),
            "launch_core_engines",
        )
        return _original_launch_core_engines(*args, **kwargs)

    engine_utils.launch_core_engines = _dsa_sparse_launch_core_engines
    core_client_mod.launch_core_engines = _dsa_sparse_launch_core_engines
    engine_utils._dsa_sparse_launch_core_engines_patched = True
