#!/usr/bin/env bash

set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT="$BUNDLE_DIR/payload/vllm-ascend-0.23.0"
DMP_ROOT="${DMP_ROOT:-/root/dmp}"
TARGET="${A5_REPO_ROOT:-$DMP_ROOT/vllm-ascend-0.23.0}"
COMMIT_FILE="$SNAPSHOT/SNAPSHOT_COMMIT"

bash "$BUNDLE_DIR/verify_bundle.sh"
command -v rsync >/dev/null || {
    echo "rsync is required for source synchronization." >&2
    exit 1
}
[[ -r "$COMMIT_FILE" && -d "$SNAPSHOT/vllm_ascend" && \
   -d "$SNAPSHOT/csrc" ]] || {
    echo "The A5 source payload is incomplete." >&2
    exit 1
}
commit="$(<"$COMMIT_FILE")"
stamp="$(date +%Y%m%d_%H%M%S)"
backup="$DMP_ROOT/backups/a5-kvgather-before-${commit:0:9}-$stamp"
mkdir -p "$backup" "$TARGET"
if [[ -d "$TARGET/vllm_ascend" ]]; then
    rsync -a \
        --exclude='**/__pycache__/***' \
        --exclude='_cann_ops_custom/***' \
        "$TARGET/vllm_ascend/" "$backup/vllm_ascend/"
fi

mkdir -p "$backup/overwritten"
rsync -a --backup --backup-dir="$backup/overwritten" \
    --exclude='/.git/***' \
    --exclude='**/__pycache__/***' \
    --exclude='**/build/***' \
    --exclude='**/build_out/***' \
    --exclude='**/output/***' \
    --exclude='**/dist/***' \
    --exclude='**/*.so' \
    --exclude='**/*.run' \
    --exclude='/csrc/third_party/***' \
    --exclude='/vllm_ascend/_cann_ops_custom/***' \
    --exclude='/scripts/results/***' \
    "$SNAPSHOT/" "$TARGET/"
printf '%s\n' "$commit" >"$TARGET/SNAPSHOT_COMMIT"

echo "A5_KVGATHER_SYNC_OK: commit=$commit target=$TARGET backup=$backup"
echo "Existing extra/untracked files were preserved; overwritten files are recoverable from $backup/overwritten."
exec env \
    DMP_ROOT="$DMP_ROOT" \
    A5_REPO_ROOT="$TARGET" \
    MODEL_PATH="${MODEL_PATH:-/home/y00852214/repos/glm-moe-dsa}" \
    A5_PREFILL_DEVICE="${A5_PREFILL_DEVICE:-3}" \
    A5_DECODE_DEVICE="${A5_DECODE_DEVICE:-5}" \
    A5_PD_HOST_IP="${A5_PD_HOST_IP:-90.90.93.29}" \
    A5_PD_IFNAME="${A5_PD_IFNAME:-ens6f1}" \
    bash "$TARGET/scripts/run_a5_dsa_offload_mtp_kvgathersim_graph_on_host.sh"
