#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"
CANN_PATH="${ASCEND_HOME_PATH:-}"
VENDOR_NAME=customize_asn
AICPU_KERNEL_SO_NAME=libasn_aicpu_kernels.so
BUILD_PROTO=ON
GEN_ACLNN=OFF

usage() {
    cat <<'USAGE'
Usage: build.sh [--cann <cann_path>] [--vendor <vendor_name>] [--build-proto] [--no-build-proto] [--gen-aclnn] [--clean]

Builds the MockKVSelect AICPU kernel package artifacts:
  build/output/vendors/<vendor>/op_impl/cpu/aicpu_kernel/impl/libasn_aicpu_kernels.so
  build/output/vendors/<vendor>/op_impl/cpu/config/cust_aicpu_kernel.json
  build/output/vendors/<vendor>/op_api/lib/libcust_opapi.so
  build/output/vendors/<vendor>/op_proto/lib/linux/<arch>/libcust_opsproto_mock_kv_select.so

--gen-aclnn also generates and builds:
  build/output/vendors/<vendor>/op_api/lib/libcust_opapi.so
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
        --build-proto)
            BUILD_PROTO=ON
            shift
            ;;
        --no-build-proto)
            BUILD_PROTO=OFF
            shift
            ;;
        --gen-aclnn)
            GEN_ACLNN=ON
            shift
            ;;
        --clean)
            rm -rf "${BUILD_DIR}"
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

mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_PATH}" \
    -DVENDOR_NAME="${VENDOR_NAME}" \
    -DBUILD_MOCK_KV_SELECT_PROTO="${BUILD_PROTO}"
cmake --build "${BUILD_DIR}" --target package_aicpu -j"$(nproc)"
if [[ "${BUILD_PROTO}" == "ON" ]]; then
    cmake --build "${BUILD_DIR}" --target cust_opsproto_mock_kv_select -j"$(nproc)"
fi

if [[ "${GEN_ACLNN}" == "ON" ]]; then
    OPBUILD_DIR="${BUILD_DIR}/opbuild"
    OPBUILD_BIN="${CANN_PATH}/tools/opbuild/op_build"
    if [[ ! -x "${OPBUILD_BIN}" ]]; then
        OPBUILD_BIN="${CANN_PATH}/toolkit/tools/opbuild/op_build"
    fi
    if [[ ! -x "${OPBUILD_BIN}" ]]; then
        echo "op_build not found under ${CANN_PATH}" >&2
        exit 1
    fi

    rm -rf "${OPBUILD_DIR}"
    mkdir -p "${OPBUILD_DIR}" "${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_api/lib"
    g++ -g -fPIC -shared -std=c++11 \
        "${SCRIPT_DIR}/op_host/mock_kv_select_def.cpp" \
        -D_GLIBCXX_USE_CXX11_ABI=0 \
        -I "${CANN_PATH}/include" \
        -L "${CANN_PATH}/lib64" \
        -lexe_graph -lregister -ltiling_api \
        -o "${OPBUILD_DIR}/libascend_all_ops.so"
    OPS_PROTO_SEPARATE=1 OPS_ACLNN_GEN=1 OPS_PROJECT_NAME=aclnn OPS_PRODUCT_NAME=ascend910_93 \
        "${OPBUILD_BIN}" "${OPBUILD_DIR}/libascend_all_ops.so" "${OPBUILD_DIR}" --compute_unit=ascend910_93
    g++ -fPIC -shared -std=gnu++17 \
        "${OPBUILD_DIR}/aclnn_mock_kv_select.cpp" \
        -I "${OPBUILD_DIR}" \
        -I "${CANN_PATH}/include" \
        -I "${CANN_PATH}/include/aclnn" \
        -I "${CANN_PATH}/include/aclnn_kernels" \
        -L "${CANN_PATH}/lib64" \
        -lnnopbase -lprofapi -lge_common_base -lascend_dump -lascendalog -ldl \
        -o "${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_api/lib/libcust_opapi.so"
fi

cat <<EOF
MockKVSelect AICPU artifacts:
  ${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_impl/cpu/aicpu_kernel/impl/${AICPU_KERNEL_SO_NAME}
  ${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_impl/cpu/config/cust_aicpu_kernel.json
  ${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_api/lib/libcust_opapi.so
EOF
if [[ "${GEN_ACLNN}" == "ON" ]]; then
    echo "  ${BUILD_DIR}/output/vendors/${VENDOR_NAME}/op_api/lib/libcust_opapi.so"
fi
