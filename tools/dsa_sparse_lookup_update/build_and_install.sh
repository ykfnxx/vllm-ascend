#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTALL_ROOT="${SCRIPT_DIR}/.install"
BUILD_ONLY=0

usage() {
    cat <<'EOF'
Usage: build_and_install.sh [OPTIONS]

Build only dsa_sparse_lookup_update for Ascend 950 and install it into an
isolated directory.

Options:
  --build-only           Build the .run package without installing it.
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

if [[ ! -x "${REPO_ROOT}/csrc/build.sh" ]]; then
    echo "ERROR: csrc/build.sh is missing or not executable under ${REPO_ROOT}." >&2
    exit 1
fi

(
    cd -- "${REPO_ROOT}/csrc"
    bash build.sh --pkg --ops=dsa_sparse_lookup_update --soc=ascend950
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

echo "Installed dsa_sparse_lookup_update under ${INSTALL_ROOT}"
echo "Correctness:"
echo "  python3 tools/dsa_sparse_lookup_update/test_correctness.py --install-root ${INSTALL_ROOT}"
echo "Profile:"
echo "  python3 tools/dsa_sparse_lookup_update/profile_operator.py --install-root ${INSTALL_ROOT}"
