#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RUNNER_BUILD_DIR="${SCRIPT_DIR}/build_runner"
INSTALL_ROOT="${SCRIPT_DIR}/.install"
SOC_VERSION="ascend950"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8')"
if ((JOBS > 8)); then
    JOBS=8
fi
CLEAN=0
BUILD_ONLY=0

usage() {
    cat <<'EOF'
Usage: standalone/build.sh [OPTIONS]

Build DsaSparseLookupUpdate as an independent CANN custom-op package. The
operator sources do not include or link any vllm-ascend build target.

Options:
  --install-root PATH  Isolated installation root (default: standalone/.install).
  --jobs N             Parallel build jobs (default: min(host CPUs, 8)).
  --clean              Remove only standalone build directories before build.
  --build-only         Produce the .run package without installing or building
                       the ACLNN runner.
  -h, --help           Show this help.
EOF
}

require_value() {
    if (($# < 2)); then
        echo "ERROR: $1 requires a value." >&2
        exit 2
    fi
}

while (($# > 0)); do
    case "$1" in
        --install-root)
            require_value "$@"
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --jobs)
            require_value "$@"
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --jobs must be a positive integer." >&2
    exit 2
fi

ASCEND_ROOT="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}"
ASC_CMAKE_ROOT="${ASCEND_ROOT}/compiler/tikcpp/ascendc_kernel_cmake"
if [[ ! -d "${ASC_CMAKE_ROOT}" ]]; then
    echo "ERROR: CANN ASC CMake package is missing: ${ASC_CMAKE_ROOT}" >&2
    echo "Source the CANN set_env.sh that provides ASCEND_HOME_PATH." >&2
    exit 1
fi

if ((CLEAN)); then
    rm -rf -- "${BUILD_DIR}" "${RUNNER_BUILD_DIR}"
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_ROOT}" \
    -DASCEND_COMPUTE_UNIT="${SOC_VERSION}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" \
    --target all binary package --parallel "${JOBS}"

mapfile -t installer_records < <(
    find "${BUILD_DIR}" -maxdepth 1 -type f -name '*.run' \
        -printf '%T@ %p\n' | sort -nr
)
if ((${#installer_records[@]} == 0)); then
    echo "ERROR: standalone build produced no .run package." >&2
    exit 1
fi
installer="${installer_records[0]#* }"
echo "Standalone package: ${installer}"

if ((BUILD_ONLY)); then
    exit 0
fi

mkdir -p -- "${INSTALL_ROOT}"
chmod +x -- "${installer}"
"${installer}" --install-path="${INSTALL_ROOT}"

vendor_root="${INSTALL_ROOT}/vendors/dsa_sparse_prof"
if [[ ! -d "${vendor_root}" ]]; then
    echo "ERROR: installation did not create ${vendor_root}." >&2
    exit 1
fi

cmake -S "${SCRIPT_DIR}/runner" -B "${RUNNER_BUILD_DIR}" \
    -DASCEND_HOME_PATH="${ASCEND_ROOT}" \
    -DCUSTOM_OP_ROOT="${vendor_root}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${RUNNER_BUILD_DIR}" --parallel "${JOBS}"

runner="${RUNNER_BUILD_DIR}/dsa_sparse_lookup_update_runner"
if [[ ! -x "${runner}" ]]; then
    echo "ERROR: runner was not produced at ${runner}." >&2
    exit 1
fi

echo "Standalone install root: ${INSTALL_ROOT}"
echo "Standalone runner: ${runner}"
echo "Roofline:"
echo "  bash tools/dsa_sparse_lookup_update/profile_roofline.sh --requests 32 --miss-rate 10"
