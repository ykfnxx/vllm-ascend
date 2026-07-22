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
        start|rm|run)
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

SOURCE_ROOT="$TEST_ROOT/vllm-ascend-0.18.0-copy"
mkdir -p \
    "$SOURCE_ROOT/vllm_ascend" \
    "$TEST_ROOT/scripts" \
    "$TEST_ROOT/models" \
    "$TEST_ROOT/reduced-models"
: > "$SOURCE_ROOT/requirements.txt"

run_start() {
    DMP_ROOT="$TEST_ROOT" \
    MODEL_HOST_PATH="$TEST_ROOT/models" \
    REDUCED_MODELS_HOST_PATH="$TEST_ROOT/reduced-models" \
    MOCK_IMAGE_ID="$1" \
        bash "$START_SCRIPT" </dev/null
}

run_start sha256:a3-image-one
compgen -G "$SOURCE_ROOT/vllm_ascend/vllm_ascend_C*.so" >/dev/null
compgen -G "$SOURCE_ROOT/vllm_ascend/*vllm_ascend_kernels*.so" >/dev/null
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
