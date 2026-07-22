#!/usr/bin/env bash
# Run sparse_flash_attention NPU examples (custom OPP + custom_ops wheel).
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
export USE_CUSTOM_SFA=1

pip3 install -q expecttest hypothesis 2>/dev/null || pip3 install expecttest hypothesis

cd "${OP_ROOT}/examples"
python3 -c "
import sys
from test_npu_sparse_flash_attention import _parse_test_names, _run_tests
_run_tests(_parse_test_names(sys.argv))
" TestCustomSFA

echo "sparse_flash_attention all tests passed."
