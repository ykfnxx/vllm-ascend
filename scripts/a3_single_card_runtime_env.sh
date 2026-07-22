#!/usr/bin/env bash

# Keep vLLM's single-card control-plane traffic on the local loopback device.
# This avoids repeated hostname and multi-NIC discovery while vLLM creates its
# world/TP/PP/DP/EP process groups. Multi-card runs are left unchanged.
dmp_configure_a3_single_card_rendezvous() {
    local tp_size="${TENSOR_PARALLEL_SIZE:-1}"
    local ep_enabled="${ENABLE_EXPERT_PARALLEL:-0}"
    local local_rendezvous="${DMP_A3_LOCAL_RENDEZVOUS:-1}"

    if [[ "$local_rendezvous" != "1" || "$tp_size" != "1" || \
          "$ep_enabled" == "1" ]]; then
        return 0
    fi

    export VLLM_HOST_IP="${DMP_A3_LOCAL_HOST_IP:-127.0.0.1}"
    export GLOO_SOCKET_IFNAME="${DMP_A3_GLOO_IFNAME:-lo}"

    if [[ -d /sys/class/net && ! -d "/sys/class/net/$GLOO_SOCKET_IFNAME" ]]; then
        echo "Gloo interface does not exist: $GLOO_SOCKET_IFNAME" >&2
        return 1
    fi
}

# Put the persistent GLM-5 Transformers build ahead of the image packages for
# both the frontend and every child process started by vLLM.
dmp_activate_a3_model_runtime() {
    local script_dir
    local model_runtime
    local runtime_info

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    model_runtime="${DMP_MODEL_RUNTIME_PYTHON_PATH:-$script_dir/dmp-model-runtime/python}"
    export DMP_MODEL_RUNTIME_PYTHON_PATH="$model_runtime"
    export PYTHONPATH="$model_runtime${PYTHONPATH:+:$PYTHONPATH}"

    if ! runtime_info="$(python3 - <<'PY'
from pathlib import Path

import transformers
from transformers import AutoConfig

AutoConfig.for_model("glm_moe_dsa")
print(Path(transformers.__file__).resolve())
PY
    )"; then
        echo "The active Python environment does not support model_type=glm_moe_dsa." >&2
        echo "Expected the persistent runtime at: $model_runtime" >&2
        echo "Run /workspace/scripts/build_a3_dmp_runtime.sh once." >&2
        return 1
    fi

    export DMP_MODEL_RUNTIME_INFO="$runtime_info"
}
