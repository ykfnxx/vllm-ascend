#!/usr/bin/env bash
# Build and install sparse_flash_attention and its fused merge helper.
set -euo pipefail

export OPP_OP_NAME="${OPP_OP_NAME:-dmp_sparse_flash_attention;da_attention_merge}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/build_opp.sh"
