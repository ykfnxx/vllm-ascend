#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip3 install -q expecttest hypothesis 2>/dev/null || pip3 install expecttest hypothesis

bash "${SCRIPT_DIR}/build_opp.sh"
bash "${SCRIPT_DIR}/build_torch_ops.sh"
bash "${SCRIPT_DIR}/run_npu_tests.sh"
