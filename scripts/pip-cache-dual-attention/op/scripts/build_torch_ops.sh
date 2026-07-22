#!/usr/bin/env bash
# Build and install custom_ops wheel from op/torch_ops_extension.
# Set CUSTOM_OPS_SFA_ONLY=1 for sparse_flash_attention-only wheel (see build_torch_ops_sfa.sh).
set -euo pipefail

OP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="${OP_ROOT}/torch_ops_extension"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Use ${VAR-default} so an explicitly empty value means "both ops", not gather-only.
export CUSTOM_OPS_GATHER_ONLY="${CUSTOM_OPS_GATHER_ONLY-}"
export CUSTOM_OPS_SFA_ONLY="${CUSTOM_OPS_SFA_ONLY-}"
export USE_NINJA=1
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# setuptools-scm / vcs_versioning fails on git "dubious ownership" under pip-cache mount.
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
cp -r "${EXT}/custom_ops" "${BUILD_DIR}/"
cp "${EXT}/setup.py" "${BUILD_DIR}/"

cd "${BUILD_DIR}"
export USE_NINJA=1
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
python3 setup.py build_ext bdist_wheel
pip3 install dist/*.whl -I
mkdir -p "${EXT}/dist"
cp dist/*.whl "${EXT}/dist/"

echo "custom_ops wheel installed; artifact copied to ${EXT}/dist/"
