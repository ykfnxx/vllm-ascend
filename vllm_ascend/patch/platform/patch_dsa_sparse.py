"""Platform-stage patches for Ascend DSA sparse-cache offload."""

import vllm_ascend.patch.dsa_sparse.patch_runtime  # noqa: F401
import vllm_ascend.patch.dsa_sparse.patch_engine_process  # noqa: F401
