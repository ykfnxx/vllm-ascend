#!/bin/bash

# =============================================================================
# 配置区域 - 在这里修改容器名、镜像名和挂载路径
# =============================================================================

# 路径前缀
PREFIX=/root/dmp

# 容器和镜像
CONTAINER_NAME="vllm-ascend-env1"
IMAGE="quay.io/ascend/vllm-ascend:v0.18.0-openeuler"

# 宿主机路径 -> 容器内路径
MOUNT_MODELS="$PREFIX/scripts/glm-moe-dsa:/models/GLM-5.1"
MOUNT_CACHE="/root/.cache:/root/.cache"
MOUNT_EXAMPLE="$PREFIX/scripts:/workspace/scripts"
# MOUNT_VLLM_ASCEND="$PREFIX/dmp-vllm-ascend-v0.13.0/vllm_ascend:/vllm-workspace/vllm-ascend/vllm_ascend"
MOUNT_VLLM_ASCEND="$PREFIX/vllm-ascend-0.18.0-copy/vllm_ascend:/vllm-workspace/vllm-ascend/vllm_ascend"

# vllm-ascend-0.18.0
MOUNT_ASCEND_REQ="$PREFIX/vllm-ascend-0.18.0-copy/requirements.txt:/vllm-workspace/vllm-ascend/requirements.txt"
MOUNT_TRANSFORMER="$PREFIX/transformers-5.2.0-py3-none-any.whl:/workspace/transformers-5.2.0-py3-none-any.whl"
MOUNT_HUGGINGFACE="$PREFIX/huggingface_hub-1.22.0-py3-none-any.whl:/workspace/huggingface_hub-1.22.0-py3-none-any.whl"
# MOUNT_VLLM_LOAD="$PREFIX/vllm-patch/deepseek_v2.py:/vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py"

# vllm-ascend-0.13.0
# MOUNT_VLLM="$PREFIX/vllm-patch/default_loader.py:/vllm-workspace/vllm/vllm/model_executor/model_loader/default_loader.py"
# MOUNT_SETUP="$PREFIX/dmp-vllm-ascend-v0.13.0/setup.py:/vllm-workspace/vllm-ascend/setup.py"

# Ascend 驱动相关挂载（一般不需要改）
MOUNT_DEV="/dev:/dev"
MOUNT_DCMI="/usr/local/dcmi:/usr/local/dcmi:ro"
MOUNT_HCCN="/usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool"
MOUNT_NPU_SMI="/usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro"
MOUNT_DRIVER_LIB="/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/:ro"
MOUNT_DRIVER_VER="/usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info"
MOUNT_ASCEND_INFO="/etc/ascend_install.info:/etc/ascend_install.info:ro"

# =============================================================================
# 启动容器
# =============================================================================
docker run --rm \
    --name "$CONTAINER_NAME" \
    --shm-size=1g \
    --net=host \
    --ipc=host \
    --security-opt seccomp=unconfined \
    --privileged \
    -v "$MOUNT_DEV" \
    -v "$MOUNT_DCMI" \
    -v "$MOUNT_HCCN" \
    -v "$MOUNT_NPU_SMI" \
    -v "$MOUNT_DRIVER_LIB" \
    -v "$MOUNT_DRIVER_VER" \
    -v "$MOUNT_ASCEND_INFO" \
    -v "$MOUNT_MODELS" \
    -v "$MOUNT_CACHE" \
    -v "$MOUNT_EXAMPLE" \
    -v "$MOUNT_ASCEND_REQ" \
    -v "$MOUNT_TRANSFORMER" \
    -v "$MOUNT_HUGGINGFACE" \
    -v "$MOUNT_VLLM_ASCEND" \
    -it "$IMAGE" bash -lc '
        if find /workspace/scripts/pip-cache-dual-attention/op/ascendc/output \
            -maxdepth 1 -name "CANN-custom_ops-*.run" -print -quit 2>/dev/null \
            | grep -q .; then
            bash /workspace/scripts/install_dmp_dual_attention_runtime.sh
        else
            echo "DMP operators have not been built yet; run the revision validation script."
        fi
        exec bash
    '
