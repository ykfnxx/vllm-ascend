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
