#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTALL_ROOT="${SCRIPT_DIR}/.install"
BUILD_ONLY=0
FORCE_FRESH_CONFIGURE=0
OPERATOR_SELECTION="all"
SOC_VERSION="ascend950"
CMAKE_BUILD_DIR="${REPO_ROOT}/csrc/build"
CMAKE_CACHE="${CMAKE_BUILD_DIR}/CMakeCache.txt"

usage() {
    cat <<'EOF'
Usage: build_and_install.sh [OPTIONS]

Build selected DSA sparse metadata operators and install them into an isolated
directory.

Options:
  --build-only           Build the .run package without installing it.
  --fresh                Force a fresh CMake configure while preserving
                         downloaded third-party sources.
  --operator NAME        simt, lookup, maintain, legacy, or all
                         (default: all).
  --soc NAME             ascend950 or ascend910_93
                         (default: ascend950).
  --install-root PATH    Install root (default: tools/.../.install).
  -h, --help             Show this message.
EOF
}

while (($# > 0)); do
    case "$1" in
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --fresh)
            FORCE_FRESH_CONFIGURE=1
            shift
            ;;
        --operator)
            if (($# < 2)); then
                echo "ERROR: --operator requires a value." >&2
                exit 2
            fi
            OPERATOR_SELECTION="$2"
            shift 2
            ;;
        --soc)
            if (($# < 2)); then
                echo "ERROR: --soc requires a value." >&2
                exit 2
            fi
            SOC_VERSION="$2"
            shift 2
            ;;
        --install-root)
            if (($# < 2)); then
                echo "ERROR: --install-root requires a path." >&2
                exit 2
            fi
            INSTALL_ROOT="$2"
            shift 2
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

case "${OPERATOR_SELECTION}" in
    simt)
        BUILD_OPS="dsa_sparse_lookup_update"
        EXPECTED_CMAKE_OPS="${BUILD_OPS}"
        EXPECT_AICPU=0
        ;;
    lookup)
        BUILD_OPS="asu_hbm_index_lookup"
        EXPECTED_CMAKE_OPS="${BUILD_OPS}"
        EXPECT_AICPU=0
        ;;
    maintain)
        BUILD_OPS="asu_hbm_index_maintain_aicpu"
        EXPECTED_CMAKE_OPS="${BUILD_OPS}"
        EXPECT_AICPU=1
        ;;
    legacy)
        BUILD_OPS="asu_hbm_index_lookup,asu_hbm_index_maintain_aicpu"
        EXPECTED_CMAKE_OPS="asu_hbm_index_lookup;asu_hbm_index_maintain_aicpu"
        EXPECT_AICPU=1
        ;;
    all)
        BUILD_OPS="dsa_sparse_lookup_update,asu_hbm_index_lookup,asu_hbm_index_maintain_aicpu"
        EXPECTED_CMAKE_OPS="dsa_sparse_lookup_update;asu_hbm_index_lookup;asu_hbm_index_maintain_aicpu"
        EXPECT_AICPU=1
        ;;
    *)
        echo "ERROR: unsupported --operator ${OPERATOR_SELECTION}." >&2
        usage >&2
        exit 2
        ;;
esac

case "${SOC_VERSION}" in
    ascend950 | ascend910_93)
        ;;
    *)
        echo "ERROR: unsupported --soc ${SOC_VERSION}." >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "${SOC_VERSION}" != "ascend950" ]] &&
    [[ "${OPERATOR_SELECTION}" == "simt" ||
       "${OPERATOR_SELECTION}" == "all" ]]; then
    echo "ERROR: the SIMT operator is packaged only for ascend950." >&2
    exit 2
fi

if [[ ! -x "${REPO_ROOT}/csrc/build.sh" ]]; then
    echo "ERROR: csrc/build.sh is missing or not executable under ${REPO_ROOT}." >&2
    exit 1
fi

STALE_CONFIGURE_REASON=""
if [[ -f "${CMAKE_CACHE}" ]]; then
    CACHED_OPS="$(
        sed -n 's/^ASCEND_OP_NAME:[^=]*=//p' "${CMAKE_CACHE}"
    )"
    CACHED_OPS="${CACHED_OPS//,/;}"
    if [[ "${CACHED_OPS}" != "${EXPECTED_CMAKE_OPS}" ]]; then
        STALE_CONFIGURE_REASON="cached ASCEND_OP_NAME differs from ${EXPECTED_CMAKE_OPS}"
    elif ((EXPECT_AICPU == 0)) && grep -Eq \
        "^AICPU_CUST_OBJ_TARGETS:[^=]*=.+$|^ENABLE_AICPU:BOOL=ON$" \
        "${CMAKE_CACHE}"; then
        STALE_CONFIGURE_REASON="cached AICPU targets are incompatible with the AIV-only operator"
    fi
fi

if ((FORCE_FRESH_CONFIGURE)) || [[ -n "${STALE_CONFIGURE_REASON}" ]]; then
    if ((FORCE_FRESH_CONFIGURE)); then
        echo "Resetting CMake configure state: requested by --fresh."
    else
        echo "Resetting CMake configure state: ${STALE_CONFIGURE_REASON}."
    fi
    rm -f -- "${CMAKE_CACHE}"
    rm -rf -- "${CMAKE_BUILD_DIR}/CMakeFiles"
fi

(
    cd -- "${REPO_ROOT}/csrc"
    bash build.sh --pkg --ops="${BUILD_OPS}" --soc="${SOC_VERSION}"
)

mapfile -t INSTALLER_RECORDS < <(
    find "${REPO_ROOT}/csrc/build" \
        -maxdepth 1 \
        -type f \
        -name 'cann-ops-transformer*.run' \
        -printf '%T@ %p\n' |
        sort -nr
)

if ((${#INSTALLER_RECORDS[@]} == 0)); then
    echo "ERROR: no cann-ops-transformer*.run package was produced." >&2
    exit 1
fi

INSTALLER="${INSTALLER_RECORDS[0]#* }"
echo "Single-op package: ${INSTALLER}"

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

echo "Installed ${BUILD_OPS} under ${INSTALL_ROOT}"
if [[ "${OPERATOR_SELECTION}" == "simt" ||
      "${OPERATOR_SELECTION}" == "all" ]]; then
    echo "Correctness:"
    echo "  python3 tools/dsa_sparse_lookup_update/test_correctness.py --install-root ${INSTALL_ROOT}"
    echo "Profile:"
    echo "  python3 tools/dsa_sparse_lookup_update/profile_operator.py --install-root ${INSTALL_ROOT}"
fi
echo "Benchmark:"
echo "  python3 tools/dsa_sparse_lookup_update/benchmark_operator.py --install-root ${INSTALL_ROOT} --operator ${OPERATOR_SELECTION} --concurrency 8"
