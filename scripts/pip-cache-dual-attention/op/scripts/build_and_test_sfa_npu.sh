#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip3 install -q expecttest hypothesis 2>/dev/null || pip3 install expecttest hypothesis

export OPP_OP_NAME=dmp_sparse_flash_attention
bash "${SCRIPT_DIR}/build_opp_sfa.sh"
export CUSTOM_OPS_GATHER_ONLY=
export CUSTOM_OPS_SFA_ONLY=1
bash "${SCRIPT_DIR}/build_torch_ops.sh"
bash "${SCRIPT_DIR}/run_npu_tests_sfa.sh"
