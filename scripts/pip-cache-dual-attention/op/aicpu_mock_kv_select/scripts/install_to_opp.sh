#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${PROJECT_DIR}/build"
CANN_PATH="${ASCEND_HOME_PATH:-}"
VENDOR_NAME=customize_asn
AICPU_KERNEL_SO_NAME=libasn_aicpu_kernels.so
INSTALL_ACLNN=OFF

usage() {
    cat <<'USAGE'
Usage: install_to_opp.sh [--cann <cann_path>] [--vendor <vendor_name>] [--install-aclnn]

Copies MockKVSelect AICPU artifacts into the active CANN OPP vendor directory.
Run build.sh first.

--install-aclnn additionally copies build/output/vendors/<vendor>/op_api/lib/libcust_opapi.so.
This can overwrite an existing vendor op_api library, so it is opt-in.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cann)
            CANN_PATH="$2"
            shift 2
            ;;
        --vendor)
            VENDOR_NAME="$2"
            shift 2
            ;;
        --install-aclnn)
            INSTALL_ACLNN=ON
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${CANN_PATH}" ]]; then
    if [[ -d /usr/local/Ascend/cann-8.5.1 ]]; then
        CANN_PATH=/usr/local/Ascend/cann-8.5.1
    elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
        CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
    else
        echo "CANN path is not set. Use --cann or ASCEND_HOME_PATH." >&2
        exit 1
    fi
fi

OPP_PATH="${ASCEND_OPP_PATH:-${CANN_PATH}/opp}"
SRC_ROOT="${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_impl/cpu"
SRC_SO="${SRC_ROOT}/aicpu_kernel/impl/${AICPU_KERNEL_SO_NAME}"
SRC_JSON="${SRC_ROOT}/config/cust_aicpu_kernel.json"
SRC_OPAPI="${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_api/lib/libcust_opapi.so"
SRC_OPAPI_HEADER="${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_api/include/aclnn_mock_kv_select.h"
SRC_PROTO="${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_proto/lib/linux/$(uname -m)/libcust_opsproto_mock_kv_select.so"
SRC_TF_PLUGIN="${BUILD_DIR}/output/vendors/${VENDOR_NAME}/framework/tensorflow/libcust_tf_parsers.so"
DST_ROOT="${OPP_PATH}/vendors/${VENDOR_NAME}/op_impl/cpu"
OPAPI_DST_ROOT="${OPP_PATH}/vendors/${VENDOR_NAME}/op_api/lib"
OPAPI_HEADER_DST_ROOT="${OPP_PATH}/vendors/${VENDOR_NAME}/op_api/include"
PROTO_DST_ROOT="${OPP_PATH}/vendors/${VENDOR_NAME}/op_proto/lib/linux/$(uname -m)"
TF_PLUGIN_DST_ROOT="${OPP_PATH}/vendors/${VENDOR_NAME}/framework/tensorflow"

if [[ ! -f "${SRC_SO}" || ! -f "${SRC_JSON}" ]]; then
    echo "AICPU artifacts not found. Run ${PROJECT_DIR}/build.sh first." >&2
    exit 1
fi

mkdir -p "${DST_ROOT}/aicpu_kernel/impl" "${DST_ROOT}/config"
cp -f "${SRC_SO}" "${DST_ROOT}/aicpu_kernel/impl/${AICPU_KERNEL_SO_NAME}"
cp -f "${SRC_JSON}" "${DST_ROOT}/config/cust_aicpu_kernel.json"
if [[ -f "${SRC_PROTO}" ]]; then
    mkdir -p "${PROTO_DST_ROOT}"
    cp -f "${SRC_PROTO}" "${PROTO_DST_ROOT}/libcust_opsproto_mock_kv_select.so"
fi
if [[ -f "${SRC_TF_PLUGIN}" ]]; then
    mkdir -p "${TF_PLUGIN_DST_ROOT}"
    cp -f "${SRC_TF_PLUGIN}" "${TF_PLUGIN_DST_ROOT}/libcust_tf_parsers.so"
fi
if [[ "${INSTALL_ACLNN}" == "ON" && -f "${SRC_OPAPI}" ]]; then
    mkdir -p "${OPAPI_DST_ROOT}"
    cp -f "${SRC_OPAPI}" "${OPAPI_DST_ROOT}/libcust_opapi.so"
    if [[ -f "${SRC_OPAPI_HEADER}" ]]; then
        mkdir -p "${OPAPI_HEADER_DST_ROOT}"
        cp -f "${SRC_OPAPI_HEADER}" "${OPAPI_HEADER_DST_ROOT}/aclnn_mock_kv_select.h"
    fi
fi

CONFIG_FILE="${OPP_PATH}/vendors/config.ini"
mkdir -p "${OPP_PATH}/vendors"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    printf 'load_priority=%s\n' "${VENDOR_NAME}" > "${CONFIG_FILE}"
else
    CURRENT_PRIORITY=$(grep -E '^load_priority=' "${CONFIG_FILE}" | cut -d= -f2- || true)
    if [[ -z "${CURRENT_PRIORITY}" ]]; then
        printf 'load_priority=%s\n' "${VENDOR_NAME}" >> "${CONFIG_FILE}"
    else
        UPDATED_PRIORITY=$(printf '%s\n' "${CURRENT_PRIORITY}" | tr ',' '\n' | awk -v vendor="${VENDOR_NAME}" '
            BEGIN { printf "%s", vendor }
            $0 != "" && $0 != vendor { printf ",%s", $0 }
            END { printf "\n" }
        ')
        sed -i "s#^load_priority=.*#load_priority=${UPDATED_PRIORITY}#" "${CONFIG_FILE}"
    fi
fi

BIN_DIR="${OPP_PATH}/vendors/${VENDOR_NAME}/bin"
mkdir -p "${BIN_DIR}"
cat > "${BIN_DIR}/set_env.bash" <<EOF
#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=${OPP_PATH}/vendors/${VENDOR_NAME}:\${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${OPP_PATH}/vendors/${VENDOR_NAME}/op_api/lib/:\${LD_LIBRARY_PATH:-}
EOF

echo "Installed MockKVSelect AICPU artifacts to ${DST_ROOT}"
