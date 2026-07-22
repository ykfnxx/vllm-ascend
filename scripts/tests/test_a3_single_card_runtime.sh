#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SCRIPT="$SCRIPT_DIR/../a3_single_card_runtime_env.sh"
SUMMARY_SCRIPT="$SCRIPT_DIR/../summarize_distributed_init.py"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

single_card_env="$({
    unset VLLM_HOST_IP GLOO_SOCKET_IFNAME
    TENSOR_PARALLEL_SIZE=1
    ENABLE_EXPERT_PARALLEL=0
    source "$ENV_SCRIPT"
    dmp_configure_a3_single_card_rendezvous || exit 1
    printf '%s %s\n' "$VLLM_HOST_IP" "$GLOO_SOCKET_IFNAME"
})"
[[ "$single_card_env" == "127.0.0.1 lo" ]]

multi_card_env="$({
    VLLM_HOST_IP="10.0.0.1"
    GLOO_SOCKET_IFNAME="eth-test"
    TENSOR_PARALLEL_SIZE=2
    ENABLE_EXPERT_PARALLEL=0
    source "$ENV_SCRIPT"
    dmp_configure_a3_single_card_rendezvous || exit 1
    printf '%s %s\n' "$VLLM_HOST_IP" "$GLOO_SOCKET_IFNAME"
})"
[[ "$multi_card_env" == "10.0.0.1 eth-test" ]]

mkdir -p "$TEST_ROOT/model-runtime/transformers"
cat > "$TEST_ROOT/model-runtime/transformers/__init__.py" <<'PY'
class AutoConfig:
    @classmethod
    def for_model(cls, model_type):
        if model_type != "glm_moe_dsa":
            raise ValueError(model_type)
        return object()
PY

model_runtime_env="$({
    DMP_MODEL_RUNTIME_PYTHON_PATH="$TEST_ROOT/model-runtime"
    source "$ENV_SCRIPT"
    dmp_activate_a3_model_runtime || exit 1
    python3 - <<'PY'
import os
from pathlib import Path

import transformers
from transformers import AutoConfig

AutoConfig.for_model("glm_moe_dsa")
print(Path(transformers.__file__).resolve())
print(os.environ["PYTHONPATH"].split(os.pathsep)[0])
PY
})"
expected_transformers_path="$(
    python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' \
        "$TEST_ROOT/model-runtime/transformers/__init__.py"
)"
grep -q "^$expected_transformers_path$" <<< "$model_runtime_env"
[[ "$(tail -1 <<< "$model_runtime_env")" == "$TEST_ROOT/model-runtime" ]]

cat > "$TEST_ROOT/engine.log" <<'EOF'
(EngineCore pid=1) INFO 07-22 01:54:53 [parallel_state.py:1395] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://80.10.10.32:50161 backend=hccl
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore pid=1) INFO 07-22 02:08:54 [parallel_state.py:1717] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0
EOF

summary="$(python3 "$SUMMARY_SCRIPT" "$TEST_ROOT/engine.log")"
grep -q '^distributed_init_method=tcp://80.10.10.32:50161$' <<< "$summary"
grep -q '^distributed_init_seconds=841$' <<< "$summary"

echo "A3 single-card runtime environment tests passed."
