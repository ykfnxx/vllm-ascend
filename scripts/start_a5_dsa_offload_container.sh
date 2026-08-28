#!/usr/bin/env bash

set -euo pipefail

IMAGE_HINT="${A5_IMAGE:-dev-26.1.0.day20260712-A5-biaoka-py311-Ubuntu24.04-lts-x86_64}"
CONTAINER_NAME="${A5_CONTAINER_NAME:-vllm-ascend-a5-kvgather-sim}"
VISIBLE_DEVICES="${A5_VISIBLE_DEVICES:-3,5}"
DMP_ROOT="${DMP_ROOT:-/root/dmp}"
REPO_ROOT="${A5_REPO_ROOT:-$DMP_ROOT/vllm-ascend-0.23.0}"
MODEL_PATH="${MODEL_PATH:-/home/y00852214/repos/glm-moe-dsa}"

resolve_image() {
    local ref image_id
    local -a matches=()
    if docker image inspect "$IMAGE_HINT" >/dev/null 2>&1; then
        printf '%s\n' "$IMAGE_HINT"
        return
    fi
    while IFS=$'\t' read -r ref image_id; do
        [[ -n "$ref" && "$ref" != "<none>:<none>" ]] || continue
        if [[ "$ref" == *"$IMAGE_HINT"* || "$image_id" == "$IMAGE_HINT"* ]]; then
            matches+=("$ref")
        fi
    done < <(docker images --no-trunc --format '{{.Repository}}:{{.Tag}}\t{{.ID}}')
    if ((${#matches[@]} != 1)); then
        echo "Set A5_IMAGE to one exact local image; matches=${#matches[@]}." >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

probe_container() {
    docker exec -i "$CONTAINER_NAME" python3 - <<'PY'
import torch_npu

count = torch_npu.npu.device_count()
if count < 2:
    raise RuntimeError(f"container sees {count} NPUs, expected at least 2")
print(
    "A5_CONTAINER_NPU_READY: "
    f"count={count} soc={torch_npu.npu.get_soc_version()}"
)
PY
}

command -v docker >/dev/null || {
    echo "docker is required on the A5 host." >&2
    exit 1
}
[[ -d "$REPO_ROOT/vllm_ascend" ]] || {
    echo "A5 source tree is missing: $REPO_ROOT" >&2
    exit 1
}
[[ -r "$MODEL_PATH/config.json" ]] || {
    echo "A5 model is missing: $MODEL_PATH/config.json" >&2
    exit 1
}
IFS=',' read -r -a DEVICES <<<"$VISIBLE_DEVICES"
[[ "${#DEVICES[@]}" == "2" ]] || {
    echo "A5_VISIBLE_DEVICES must contain Prefill,Decode device IDs." >&2
    exit 1
}
for device in "${DEVICES[@]}"; do
    [[ "$device" =~ ^[0-9]+$ && -e "/dev/davinci$device" ]] || {
        echo "Requested A5 device is missing: /dev/davinci$device" >&2
        exit 1
    }
done

IMAGE="$(resolve_image)"
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    if [[ "${A5_RECREATE_CONTAINER:-0}" == "1" ]]; then
        docker rm -f "$CONTAINER_NAME" >/dev/null
    else
        mounted_repo="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/vllm-workspace/vllm-ascend"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME")"
        mounted_model="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/models/glm-moe-dsa"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME")"
        configured_devices="$(
            docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' \
                "$CONTAINER_NAME" | sed -n 's/^ASCEND_RT_VISIBLE_DEVICES=//p'
        )"
        if [[ "$mounted_repo" != "$REPO_ROOT" || \
              "$mounted_model" != "$MODEL_PATH" || \
              "$configured_devices" != "$VISIBLE_DEVICES" ]]; then
            echo "Existing container mounts/devices differ; set A5_RECREATE_CONTAINER=1 once." >&2
            exit 1
        fi
        docker start "$CONTAINER_NAME" >/dev/null
        probe_container
        echo "A5_KVGATHER_CONTAINER_READY: $CONTAINER_NAME (reused)"
        exit 0
    fi
fi

docker_args=(
    run -d
    --name "$CONTAINER_NAME"
    --restart unless-stopped
    --net host
    --ipc host
    --shm-size 16g
    --security-opt seccomp=unconfined
    --privileged
    -e "ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES"
    -v /dev:/dev
    -v "$REPO_ROOT:/vllm-workspace/vllm-ascend"
    -v "$MODEL_PATH:/models/glm-moe-dsa:ro"
)

add_mount() {
    local source="$1" destination="${2:-$1}"
    [[ -e "$source" ]] && docker_args+=(-v "$source:$destination:ro")
}
add_mount /usr/local/dcmi
add_mount /usr/local/Ascend/driver/lib64
add_mount /usr/local/Ascend/driver/version.info
add_mount /etc/ascend_install.info
if command -v npu-smi >/dev/null 2>&1; then
    npu_smi_path="$(command -v npu-smi)"
    add_mount "$npu_smi_path"
fi

docker_args+=(--entrypoint /bin/bash "$IMAGE" -lc "exec sleep infinity")
docker "${docker_args[@]}" >/dev/null
probe_container
echo "A5_KVGATHER_CONTAINER_READY: $CONTAINER_NAME image=$IMAGE devices=$VISIBLE_DEVICES"
