#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="$SCRIPT_DIR/../start_container_a3.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

MOCK_LOG="$TEST_ROOT/docker.log"
export MOCK_LOG

docker() {
    if [[ "$1" == "image" && "$2" == "inspect" ]]; then
        if [[ "${3:-}" == "--format" ]]; then
            printf '%s\n' "${MOCK_IMAGE_ID:-sha256:a3-image-one}"
        fi
        return 0
    fi

    case "$1" in
        create)
            printf 'create\n' >> "$MOCK_LOG"
            ;;
        start|rm)
            return 0
            ;;
        run)
            printf 'run %s\n' "$*" >> "$MOCK_LOG"
            return 0
            ;;
        cp)
            local source_path="$2"
            local destination="$3"
            case "$source_path" in
                *'/_cann_ops_custom/.')
                    mkdir -p "$destination/op_api/lib"
                    : > "$destination/op_api/lib/libcust_opapi.so"
                    ;;
                *'/_build_info.py')
                    printf '%s\n' '__device_type__ = "A3"' > "$destination"
                    ;;
                *'/_version.py')
                    printf '%s\n' '__version__ = "0.18.0"' > "$destination"
                    ;;
                *':/tmp/dmp-a3-native-files')
                    printf '%s\n' \
                        '/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so' \
                        '/vllm-workspace/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so' \
                        > "$destination"
                    ;;
                *'/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so'|\
                *'/libvllm_ascend_kernels.so')
                    : > "$destination/$(basename "${source_path#*:}")"
                    ;;
                *)
                    printf 'unexpected docker cp: %s -> %s\n' \
                        "$source_path" "$destination" >&2
                    return 1
                    ;;
            esac
            ;;
        *)
            printf 'unexpected docker command: %s\n' "$*" >&2
            return 1
            ;;
    esac
}
export -f docker

SOURCE_ROOT="$TEST_ROOT/repos/vllm-ascend"
RUNTIME_BUNDLE_ROOT="$TEST_ROOT/runtime-bundle"
mkdir -p \
    "$SOURCE_ROOT/vllm_ascend" \
    "$SOURCE_ROOT/scripts" \
    "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/cmake" \
    "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/src/utils/inc/log/inner" \
    "$RUNTIME_BUNDLE_ROOT/dmp-runtime" \
    "$TEST_ROOT/models" \
    "$TEST_ROOT/reduced-models"
: > "$SOURCE_ROOT/requirements.txt"
: > "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/cmake/config.cmake"
: > "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/cmake/func.cmake"
: > "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/src/utils/inc/log/ops_log.h"
: > "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/src/utils/inc/log/inner/dfx_base.h"

run_start() {
    DMP_ROOT="$TEST_ROOT" \
    DMP_A3_SOURCE_ROOT="$SOURCE_ROOT" \
    DMP_A3_SCRIPT_ROOT="$SOURCE_ROOT/scripts" \
    DMP_A3_RUNTIME_BUNDLE_ROOT="$RUNTIME_BUNDLE_ROOT" \
    MODEL_HOST_PATH="$TEST_ROOT/models" \
    REDUCED_MODELS_HOST_PATH="$TEST_ROOT/reduced-models" \
    MOCK_IMAGE_ID="$1" \
        bash "$START_SCRIPT" </dev/null
}

run_start sha256:a3-image-one
compgen -G "$SOURCE_ROOT/vllm_ascend/vllm_ascend_C*.so" >/dev/null
compgen -G "$SOURCE_ROOT/vllm_ascend/*vllm_ascend_kernels*.so" >/dev/null
grep -F -- \
    "-v $SOURCE_ROOT/vllm_ascend:/vllm-workspace/vllm-ascend/vllm_ascend" \
    "$MOCK_LOG" >/dev/null
grep -F -- "-v $SOURCE_ROOT/scripts:/workspace/scripts" "$MOCK_LOG" >/dev/null
grep -F -- \
    "-v $RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/cmake:/workspace/scripts/pip-cache-dual-attention/op/ascendc/cmake" \
    "$MOCK_LOG" >/dev/null
grep -F -- \
    "-v $RUNTIME_BUNDLE_ROOT/dmp-runtime:/workspace/scripts/dmp-runtime" \
    "$MOCK_LOG" >/dev/null
[[ "$(<"$SOURCE_ROOT/vllm_ascend/.dmp-a3-native-image-id")" == \
   "sha256:a3-image-one" ]]
[[ "$(grep -c '^create$' "$MOCK_LOG")" == "1" ]]

run_start sha256:a3-image-one
[[ "$(grep -c '^create$' "$MOCK_LOG")" == "1" ]]

run_start sha256:a3-image-two
[[ "$(<"$SOURCE_ROOT/vllm_ascend/.dmp-a3-native-image-id")" == \
   "sha256:a3-image-two" ]]
[[ "$(grep -c '^create$' "$MOCK_LOG")" == "2" ]]

echo "A3 native-runtime seed test OK."
