#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DMP_FUSED_INDEXER_ROOT:-$SCRIPT_DIR/dmp-fused-indexer-kv-select}"
DEVICE="${DMP_FUSED_INDEXER_TEST_DEVICE:-0}"

if [[ ! -f "$ROOT/build_and_install.sh" ]]; then
    echo "Fused operator source is missing: $ROOT" >&2
    exit 1
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

if [[ "${DMP_FUSED_INDEXER_SKIP_BUILD:-0}" != "1" ]]; then
    echo "Building standalone Indexer+KVSelect fusion from: $ROOT"
    (
        cd "$ROOT"
        SOC_VERSION="${SOC_VERSION:-ascend910_9391}" \
        BUILD_JOBS="${BUILD_JOBS:-16}" \
        bash build_and_install.sh
    )
else
    echo "Skipping build; validating existing persistent artifacts."
fi

export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=1
export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=0
source "$SCRIPT_DIR/dmp_fused_indexer_runtime_env.sh"

python3 - "$ROOT" <<'PY'
import ctypes
import os
import sys

root = sys.argv[1]
library_path = os.path.join(
    root, "opp", "vendors", "customize", "op_api", "lib", "libcust_opapi.so"
)
library = ctypes.CDLL(
    library_path,
    mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0),
)
required_symbols = (
    "aclnnLightningIndexerDecodeUpdatePool",
    "aclnnLightningIndexerDecodeUpdatePoolGetWorkspaceSize",
)
missing = [name for name in required_symbols if not hasattr(library, name)]
if missing:
    raise RuntimeError(f"missing operator symbols {missing}: {library_path}")

import lightning_indexer_decode_custom_ops  # noqa: F401,E402
import torch  # noqa: E402

if not hasattr(torch.ops.custom, "npu_lightning_indexer_decode_update_pool"):
    raise RuntimeError("npu_lightning_indexer_decode_update_pool was not registered")
print("Fused Indexer+Select request-pool registration OK")
PY

(
    cd "$ROOT/tests"
    python3 test.py \
        --device "$DEVICE" \
        --bs 2 \
        --min-seqlen 4096 \
        --max-seqlen 4096 \
        --cache-size 10240 \
        --min-miss-count 0 \
        --max-miss-count 64 \
        --warmup 1 \
        --iters 1
)

echo "Fused Indexer+Select request-pool operators are persistent and smoke-tested."
