#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

SOC_VERSION="${SOC_VERSION:-ascend910_9391}"
MAX_JOBS="${MAX_JOBS:-16}"
BUILD_LOG_PATH="${BUILD_LOG_PATH:-/logs/vllm-ascend-build.log}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

[[ "${SOC_VERSION}" =~ ^ascend910_93[0-9]+$ ]] || \
  fail "SOC_VERSION must identify an A3 chip, got: ${SOC_VERSION}"
[[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]] || \
  fail "MAX_JOBS must be a positive integer"

toolkit_env=/usr/local/Ascend/ascend-toolkit/set_env.sh
nnal_env=/usr/local/Ascend/nnal/atb/set_env.sh
[[ -f "${toolkit_env}" ]] || fail "CANN environment script is missing: ${toolkit_env}"
[[ -f "${nnal_env}" ]] || fail "NNAL environment script is missing: ${nnal_env}"

# The openEuler ATB environment script reads variables such as ZSH_VERSION
# without a default. Temporarily disable nounset while sourcing vendor scripts,
# then restore the strict-mode setting for the build itself.
set +u
# shellcheck disable=SC1091
source "${toolkit_env}"
# shellcheck disable=SC1091
source "${nnal_env}"
set -u

export ASCEND_HOME_PATH
export SOC_VERSION
export MAX_JOBS
export COMPILE_CUSTOM_KERNELS=1
export CMAKE_BUILD_TYPE=Release
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/$(uname -m)-linux/devlib:${LD_LIBRARY_PATH:-}"

command -v bisheng >/dev/null 2>&1 || fail "bisheng is missing after loading CANN"
command -v cmake >/dev/null 2>&1 || fail "cmake is required"
command -v g++ >/dev/null 2>&1 || fail "g++ is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"

bundled_hccl_struct_path="${REPO_ROOT}/csrc/utils/inc/kernel/moe_distribute_base.h"
if [[ -n "${HCCL_STRUCT_FILE_PATH:-}" ]]; then
  [[ -f "${HCCL_STRUCT_FILE_PATH}" ]] || \
    fail "HCCL structure file does not exist: ${HCCL_STRUCT_FILE_PATH}"
  printf 'HCCL header: %s (CANN environment)\n' "${HCCL_STRUCT_FILE_PATH}"
else
  [[ -f "${bundled_hccl_struct_path}" ]] || \
    fail "bundled HCCL structure file is missing: ${bundled_hccl_struct_path}"
  printf 'HCCL header: %s (bundled fallback)\n' "${bundled_hccl_struct_path}"
fi

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${BUILD_LOG_PATH}")"

printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Commit:     %s\n' "$(git rev-parse --short HEAD)"
printf 'SoC:        %s\n' "${SOC_VERSION}"
printf 'MAX_JOBS:   %s\n' "${MAX_JOBS}"
printf 'Build log:  %s\n' "${BUILD_LOG_PATH}"

python3 - <<'PY'
import torch
import torch_npu
import vllm

print(f"torch={torch.__version__}")
print(f"torch_npu={getattr(torch_npu, '__version__', '<unknown>')}")
print(f"vllm={getattr(vllm, '__version__', '<unknown>')}")
PY

set -o pipefail
python3 -m pip install \
  -v \
  --no-build-isolation \
  --no-deps \
  -e "${REPO_ROOT}" \
  2>&1 | tee "${BUILD_LOG_PATH}"

python3 - <<'PY'
from pathlib import Path

import vllm_ascend

package_dir = Path(vllm_ascend.__file__).resolve().parent
source_file = package_dir / "dsa_sparse" / "dsa_sparse.py"
bindings = sorted(package_dir.glob("vllm_ascend_C*.so"))
vendor_opp = package_dir / "_cann_ops_custom" / "vendors" / "vllm-ascend"

print(f"vllm_ascend={vllm_ascend.__file__}")
print(f"bindings={[str(path) for path in bindings]}")
print(f"custom_opp={vendor_opp}")

if not bindings:
    raise SystemExit("compiled vllm_ascend_C binding is missing")
if not vendor_opp.is_dir():
    raise SystemExit("packaged custom OPP directory is missing")
source_text = source_file.read_text(encoding="utf-8")
if (
    "DSA local resident initialization is using the final-" not in source_text
    or "Prefill TopK: request=%s" not in source_text
):
    raise SystemExit("installed source does not contain the TopK initialization log")
PY

python3 "${REPO_ROOT}/check_asu_hbm_index_ops.py" --diagnose-aicpu

printf 'A3 vLLM Ascend build and static DSA custom-op checks passed.\n'
