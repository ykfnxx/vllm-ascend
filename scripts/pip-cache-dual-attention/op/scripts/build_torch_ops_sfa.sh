#!/usr/bin/env bash
# Build and install sparse_flash_attention-only custom_ops wheel.
set -euo pipefail

export CUSTOM_OPS_GATHER_ONLY=
export CUSTOM_OPS_SFA_ONLY=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/build_torch_ops.sh"
