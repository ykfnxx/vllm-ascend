#!/usr/bin/env bash
# Build and install custom OPP from op/ascendc (no cann-recipes-infer).
# OPP_OP_NAME: single op or semicolon-separated list (default: gather_selection_kv_cache).
set -euo pipefail

OP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASCENDC="${OP_ROOT}/ascendc"
INSTALL_OPP="${INSTALL_OPP:-/usr/local/Ascend/cann-8.5.1/opp}"
OPP_OP_NAME="${OPP_OP_NAME:-gather_selection_kv_cache;kv_select;kv_gather}"
OPP_COMPUTE_UNIT="${OPP_COMPUTE_UNIT:-ascend910b}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

cd "${ASCENDC}"
bash build.sh -n "${OPP_OP_NAME}" -c "${OPP_COMPUTE_UNIT}" --disable-check-compatible

RUN_PKG="$(ls -1 "${ASCENDC}/output"/CANN-custom_ops-*.run | head -1)"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${INSTALL_OPP}"

echo "OPP installed to ${INSTALL_OPP}"
