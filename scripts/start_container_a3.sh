#!/usr/bin/env bash
set -euo pipefail

PREFIX="${DMP_ROOT:-/root/dmp}"
IMAGE="${DMP_A3_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0-a3-openeuler}"
CONTAINER_NAME="${DMP_CONTAINER_NAME:-vllm-ascend-a3-dmp}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/mnt/models/GLM-5.1-w4a8}"
MODEL_CONTAINER_PATH="${MODEL_CONTAINER_PATH:-/models/GLM-5.1-w4a8}"
REDUCED_MODELS_HOST_PATH="${REDUCED_MODELS_HOST_PATH:-$PREFIX/reduced-models}"
REDUCED_MODELS_CONTAINER_PATH="${REDUCED_MODELS_CONTAINER_PATH:-/models-reduced}"
SOURCE_ROOT="${DMP_A3_SOURCE_ROOT:-/home/ykf/repos/vllm-ascend}"
SCRIPT_ROOT="${DMP_A3_SCRIPT_ROOT:-$SOURCE_ROOT/scripts}"
RUNTIME_BUNDLE_ROOT="${DMP_A3_RUNTIME_BUNDLE_ROOT:-$PREFIX/scripts}"
ASCEND_PACKAGE_ROOT="$SOURCE_ROOT/vllm_ascend"
BUNDLED_OPS="$ASCEND_PACKAGE_ROOT/_cann_ops_custom"
NATIVE_IMAGE_STAMP="$ASCEND_PACKAGE_ROOT/.dmp-a3-native-image-id"
IMAGE_PACKAGE_ROOT="/vllm-workspace/vllm-ascend/vllm_ascend"
DUAL_CMAKE_MOUNT="$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/cmake:/workspace/scripts/pip-cache-dual-attention/op/ascendc/cmake"
DUAL_UTILS_MOUNT="$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/src/utils/inc:/workspace/scripts/pip-cache-dual-attention/op/ascendc/src/utils/inc"
LOOKUP_EXTENSION_MOUNT="$RUNTIME_BUNDLE_ROOT/dmp-lookup-maintain/torch_extension/dmp_lookup_maintain_custom_ops:/workspace/scripts/dmp-lookup-maintain/torch_extension/dmp_lookup_maintain_custom_ops"
FUSED_EXTENSION_MOUNT="$RUNTIME_BUNDLE_ROOT/dmp-fused-indexer-kv-select/torch_extension/lightning_indexer_decode_custom_ops:/workspace/scripts/dmp-fused-indexer-kv-select/torch_extension/lightning_indexer_decode_custom_ops"

for path in "$MODEL_HOST_PATH" "$SOURCE_ROOT/vllm_ascend" "$SCRIPT_ROOT"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path is missing: $path" >&2
        exit 1
    fi
done
runtime_bundle_files=(
    "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/cmake/config.cmake"
    "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/cmake/func.cmake"
    "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/src/utils/inc/log/ops_log.h"
    "$RUNTIME_BUNDLE_ROOT/pip-cache-dual-attention/op/ascendc/src/utils/inc/log/inner/dfx_base.h"
)
for runtime_bundle_file in "${runtime_bundle_files[@]}"; do
    if [[ ! -f "$runtime_bundle_file" ]]; then
        echo "Required Dual-Attention build support is missing: $runtime_bundle_file" >&2
        exit 1
    fi
done
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "A3 image is not present: $IMAGE" >&2
    exit 1
fi
mkdir -p "$REDUCED_MODELS_HOST_PATH"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"

# Binding the Python source hides every native file bundled in the image. Seed
# both the OPP and the two PyTorch extensions, and refresh them if the image ID
# changes so native files from different A3 images are never mixed.
native_runtime_ready() {
    [[ -f "$NATIVE_IMAGE_STAMP" ]] &&
    [[ "$(<"$NATIVE_IMAGE_STAMP")" == "$IMAGE_ID" ]] &&
    [[ -f "$ASCEND_PACKAGE_ROOT/_build_info.py" ]] &&
    grep -q '__device_type__.*A3' "$ASCEND_PACKAGE_ROOT/_build_info.py" &&
    find "$BUNDLED_OPS" -name libcust_opapi.so -print -quit | grep -q . &&
    compgen -G "$ASCEND_PACKAGE_ROOT/vllm_ascend_C*.so" >/dev/null &&
    compgen -G "$ASCEND_PACKAGE_ROOT/*vllm_ascend_kernels*.so" >/dev/null
}

if ! native_runtime_ready; then
    seed_name="dmp-a3-ops-seed-$$"
    seed_stage="$ASCEND_PACKAGE_ROOT/.dmp-a3-native-seed-$$"
    seed_manifest="$seed_stage/native-files.txt"
    echo "Extracting the A3 native runtime from $IMAGE ..."
    mkdir -p "$seed_stage/_cann_ops_custom"
    docker create \
        --name "$seed_name" \
        --entrypoint /bin/bash \
        "$IMAGE" \
        -lc "set -e; find '$IMAGE_PACKAGE_ROOT' -maxdepth 1 -type f \\
            \\( -name 'vllm_ascend_C*.so' -o -name '*vllm_ascend_kernels*.so' \\) \\
            -print | LC_ALL=C sort > /tmp/dmp-a3-native-files; \\
            test -s /tmp/dmp-a3-native-files" >/dev/null
    trap 'docker rm -f "$seed_name" >/dev/null 2>&1 || true; rm -rf "$seed_stage"' EXIT
    docker start -a "$seed_name" >/dev/null
    docker cp \
        "$seed_name:$IMAGE_PACKAGE_ROOT/_cann_ops_custom/." \
        "$seed_stage/_cann_ops_custom/"
    docker cp \
        "$seed_name:$IMAGE_PACKAGE_ROOT/_build_info.py" \
        "$seed_stage/_build_info.py"
    docker cp \
        "$seed_name:$IMAGE_PACKAGE_ROOT/_version.py" \
        "$seed_stage/_version.py" 2>/dev/null || true
    docker cp "$seed_name:/tmp/dmp-a3-native-files" "$seed_manifest"
    while IFS= read -r native_path; do
        case "$native_path" in
            "$IMAGE_PACKAGE_ROOT"/vllm_ascend_C*.so|\
            "$IMAGE_PACKAGE_ROOT"/*vllm_ascend_kernels*.so)
                docker cp "$seed_name:$native_path" "$seed_stage/"
                ;;
            *)
                echo "Unexpected native runtime path from image: $native_path" >&2
                exit 1
                ;;
        esac
    done < "$seed_manifest"

    if ! find "$seed_stage/_cann_ops_custom" -name libcust_opapi.so -print -quit | grep -q . || \
       ! grep -q '__device_type__.*A3' "$seed_stage/_build_info.py" || \
       ! compgen -G "$seed_stage/vllm_ascend_C*.so" >/dev/null || \
       ! compgen -G "$seed_stage/*vllm_ascend_kernels*.so" >/dev/null; then
        echo "The A3 image did not provide a complete vllm-ascend native runtime." >&2
        exit 1
    fi

    rm -rf "$BUNDLED_OPS"
    mv "$seed_stage/_cann_ops_custom" "$BUNDLED_OPS"
    find "$ASCEND_PACKAGE_ROOT" -maxdepth 1 -type f \
        \( -name 'vllm_ascend_C*.so' -o -name '*vllm_ascend_kernels*.so' \) \
        -delete
    find "$seed_stage" -maxdepth 1 -type f \
        \( -name 'vllm_ascend_C*.so' -o -name '*vllm_ascend_kernels*.so' \) \
        -exec mv {} "$ASCEND_PACKAGE_ROOT/" \;
    mv "$seed_stage/_build_info.py" "$ASCEND_PACKAGE_ROOT/_build_info.py"
    if [[ -f "$seed_stage/_version.py" ]]; then
        mv "$seed_stage/_version.py" "$ASCEND_PACKAGE_ROOT/_version.py"
    fi
    printf '%s\n' "$IMAGE_ID" > "$NATIVE_IMAGE_STAMP"
    docker rm "$seed_name" >/dev/null
    rm -rf "$seed_stage"
    trap - EXIT
fi
if ! find "$BUNDLED_OPS" -name libcust_opapi.so -print -quit | grep -q .; then
    echo "A3 bundled libcust_opapi.so was not extracted successfully." >&2
    exit 1
fi
if ! grep -q '__device_type__.*A3' "$SOURCE_ROOT/vllm_ascend/_build_info.py"; then
    echo "The extracted vllm-ascend runtime is not marked for A3." >&2
    exit 1
fi
if ! compgen -G "$ASCEND_PACKAGE_ROOT/vllm_ascend_C*.so" >/dev/null || \
   ! compgen -G "$ASCEND_PACKAGE_ROOT/*vllm_ascend_kernels*.so" >/dev/null; then
    echo "A3 native PyTorch extensions were not extracted successfully." >&2
    exit 1
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

args=(
    run --rm
    --name "$CONTAINER_NAME"
    --shm-size=1g
    --net=host
    --ipc=host
    --security-opt seccomp=unconfined
    --privileged
    -v /dev:/dev
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro
    -v /root/.cache:/root/.cache
    -v "$PREFIX:/dmp-host:ro"
    -v "$MODEL_HOST_PATH:$MODEL_CONTAINER_PATH:ro"
    -v "$REDUCED_MODELS_HOST_PATH:$REDUCED_MODELS_CONTAINER_PATH"
    -v "$SCRIPT_ROOT:/workspace/scripts"
    -v "$DUAL_CMAKE_MOUNT"
    -v "$DUAL_UTILS_MOUNT"
    -v "$SOURCE_ROOT/vllm_ascend:/vllm-workspace/vllm-ascend/vllm_ascend"
    -v "$SOURCE_ROOT/requirements.txt:/vllm-workspace/vllm-ascend/requirements.txt:ro"
)

# Reuse persistent build/runtime directories from the complete offline bundle
# when they are present. Missing artifacts are rebuilt into the active checkout
# instead of masking its source package with an empty bind mount.
if [[ -d "$RUNTIME_BUNDLE_ROOT/dmp-runtime" ]]; then
    args+=(
        -v "$RUNTIME_BUNDLE_ROOT/dmp-runtime:/workspace/scripts/dmp-runtime"
    )
fi
if [[ -d "$RUNTIME_BUNDLE_ROOT/dmp-model-runtime" ]]; then
    args+=(
        -v "$RUNTIME_BUNDLE_ROOT/dmp-model-runtime:/workspace/scripts/dmp-model-runtime"
    )
fi
if [[ -f "$RUNTIME_BUNDLE_ROOT/dmp-lookup-maintain/opp/vendors/customize/op_api/lib/libcust_opapi.so" ]]; then
    args+=(
        -v "$RUNTIME_BUNDLE_ROOT/dmp-lookup-maintain/opp:/workspace/scripts/dmp-lookup-maintain/opp"
    )
fi
if compgen -G \
    "$RUNTIME_BUNDLE_ROOT/dmp-lookup-maintain/torch_extension/dmp_lookup_maintain_custom_ops/*.so" \
    >/dev/null; then
    args+=(
        -v
        "$LOOKUP_EXTENSION_MOUNT"
    )
fi
if [[ -f "$RUNTIME_BUNDLE_ROOT/dmp-fused-indexer-kv-select/opp/vendors/customize/op_api/lib/libcust_opapi.so" ]]; then
    args+=(
        -v "$RUNTIME_BUNDLE_ROOT/dmp-fused-indexer-kv-select/opp:/workspace/scripts/dmp-fused-indexer-kv-select/opp"
    )
fi
if compgen -G \
    "$RUNTIME_BUNDLE_ROOT/dmp-fused-indexer-kv-select/torch_extension/lightning_indexer_decode_custom_ops/*.so" \
    >/dev/null; then
    args+=(
        -v
        "$FUSED_EXTENSION_MOUNT"
    )
fi

if [[ -e /usr/local/bin/npu-smi ]]; then
    args+=(-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro)
elif [[ -e /usr/local/sbin/npu-smi ]]; then
    args+=(-v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro)
fi

echo "Starting $CONTAINER_NAME from $IMAGE"
echo "Source checkout: $SOURCE_ROOT"
echo "Runtime scripts: $SCRIPT_ROOT -> /workspace/scripts"
echo "Runtime bundle overlays: $RUNTIME_BUNDLE_ROOT"
echo "Model: $MODEL_HOST_PATH -> $MODEL_CONTAINER_PATH"
echo "Reduced models: $REDUCED_MODELS_HOST_PATH -> $REDUCED_MODELS_CONTAINER_PATH"
echo "Offline wheels: $PREFIX/*.whl -> /dmp-host/*.whl (read-only)"
docker "${args[@]}" -it "$IMAGE" bash
