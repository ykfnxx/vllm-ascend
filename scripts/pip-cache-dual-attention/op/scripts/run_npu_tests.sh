#!/usr/bin/env bash
# Run NPU op examples: gather_selection_kv_cache, lightning_indexer, sparse_flash_attention.
set -euo pipefail

OP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/cann-8.5.1/opp/vendors/customize/bin/set_env.bash
set -u

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-8.5.1/opp/vendors/customize/op_api/lib/:\
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/lib:\
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:\
${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

pip3 install -q expecttest hypothesis 2>/dev/null || pip3 install expecttest hypothesis

cd "${OP_ROOT}/examples"

run_one() {
  local script="$1"
  local testcase="${2:-}"
  echo "======== ${script} ${testcase} ========"
  if [[ -n "${testcase}" ]]; then
    python3 "${script}" "${testcase}"
  else
    python3 "${script}"
  fi
}

run_one test_npu_gather_selection_kv_cache.py TestCustomGatherSelectionKvCache

echo "Gather_selection_kv_cache all tests passed."
