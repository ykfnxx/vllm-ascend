#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTALL_ROOT="${SCRIPT_DIR}/.install"
BUILD_ONLY=0
FRESH=0
CMAKE_BUILD_DIR="${REPO_ROOT}/csrc/build"
CMAKE_CACHE="${CMAKE_BUILD_DIR}/CMakeCache.txt"
EXPECTED_CMAKE_OPS="dsa_sparse_lookup_update_batch"

usage() {
    cat <<'EOF'
Usage: build_and_install.sh [--install-root PATH] [--build-only] [--fresh]

Build only dsa_sparse_lookup_update_batch for Ascend 950 and install its
custom-op package into an isolated directory.
EOF
}

while (($# > 0)); do
    case "$1" in
        --install-root)
            if (($# < 2)); then
                echo "ERROR: --install-root requires a path." >&2
                exit 2
            fi
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --fresh)
            FRESH=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${REPO_ROOT}/csrc/build.sh" ]]; then
    echo "ERROR: csrc/build.sh is missing or not executable." >&2
    exit 1
fi

STALE_CONFIGURE_REASON=""
if [[ -f "${CMAKE_CACHE}" ]]; then
    CACHED_OPS="$(
        sed -n 's/^ASCEND_OP_NAME:[^=]*=//p' "${CMAKE_CACHE}"
    )"
    CACHED_OPS="${CACHED_OPS//,/;}"
    if [[ "${CACHED_OPS}" != "${EXPECTED_CMAKE_OPS}" ]]; then
        STALE_CONFIGURE_REASON="cached ASCEND_OP_NAME differs"
    elif grep -Eq \
        "^AICPU_CUST_OBJ_TARGETS:[^=]*=.+$|^ENABLE_AICPU:BOOL=ON$" \
        "${CMAKE_CACHE}"; then
        STALE_CONFIGURE_REASON="cached AICPU targets are incompatible"
    fi
fi

if ((FRESH)) || [[ -n "${STALE_CONFIGURE_REASON}" ]]; then
    if ((FRESH)); then
        echo "Resetting CMake configure state: requested by --fresh."
    else
        echo "Resetting CMake configure state: ${STALE_CONFIGURE_REASON}."
    fi
    rm -f -- "${CMAKE_CACHE}"
    rm -rf -- "${CMAKE_BUILD_DIR}/CMakeFiles"
fi

(
    cd -- "${REPO_ROOT}/csrc"
    bash build.sh \
        --pkg \
        --ops=dsa_sparse_lookup_update_batch \
        --soc=ascend950
)

mapfile -t PACKAGES < <(
    find "${CMAKE_BUILD_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'cann-ops-transformer*.run' \
        -printf '%T@ %p\n' |
        sort -nr
)
if ((${#PACKAGES[@]} == 0)); then
    echo "No custom-op installer was produced." >&2
    exit 1
fi
INSTALLER="${PACKAGES[0]#* }"
echo "Built: ${INSTALLER}"
if ((BUILD_ONLY)); then
    exit 0
fi

mkdir -p -- "${INSTALL_ROOT}"
chmod +x -- "${INSTALLER}"
"${INSTALLER}" --install-path="${INSTALL_ROOT}"
VENDOR_ROOT="${INSTALL_ROOT}/vendors/custom_transformer"
if [[ ! -d "${VENDOR_ROOT}" ]]; then
    echo "ERROR: installation did not create ${VENDOR_ROOT}." >&2
    exit 1
fi
echo "Installed under: ${INSTALL_ROOT}"
echo "Next: python3 ${SCRIPT_DIR}/test_correctness.py --install-root ${INSTALL_ROOT}"
