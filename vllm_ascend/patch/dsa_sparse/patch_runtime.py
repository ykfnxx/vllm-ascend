"""Install the fixed vLLM v0.18 DSA patch set in the current process."""


def install_dsa_runtime_patches() -> None:
    import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa: F401
    import vllm_ascend.patch.dsa_sparse.patch_scheduler_output  # noqa: F401
    import vllm_ascend.patch.dsa_sparse.patch_request  # noqa: F401
    import vllm_ascend.patch.dsa_sparse.patch_deepseek_v2  # noqa: F401
    import vllm_ascend.patch.dsa_sparse.patch_kv_cache_utils  # noqa: F401
    import vllm_ascend.patch.dsa_sparse.patch_kv_cache_decoupling  # noqa: F401
    import vllm_ascend.patch.dsa_sparse.patch_scheduler  # noqa: F401
    import vllm_ascend.patch.dsa_sparse.patch_cudagraph_phase  # noqa: F401

    from vllm_ascend.patch.dsa_sparse import patch_engine_core
    from vllm_ascend.patch.dsa_sparse import patch_kv_cache_utils

    patch_kv_cache_utils.install_dsa_kv_cache_utils_patch()
    patch_engine_core.install_dsa_engine_core_patches()


install_dsa_runtime_patches()
