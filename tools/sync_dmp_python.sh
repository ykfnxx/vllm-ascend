#!/bin/bash

set -euo pipefail

SOURCE_ROOT="$(git rev-parse --show-toplevel)"
TARGET_ROOT="${1:-/root/dmp/vllm-ascend-0.18.0-copy}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="$TARGET_ROOT/.dmp_python_backup_$STAMP"

# Only these Python files belong to the DMP overlay. In particular, this list
# excludes csrc, _cann_ops_custom, generated build files, and model assets.
FILES=(
    "vllm_ascend/ascend_forward_context.py"
    "vllm_ascend/attention/sfa_v1.py"
    "vllm_ascend/envs.py"
    "vllm_ascend/kv_offload/asu_npu.py"
    "vllm_ascend/kv_offload/block_location.py"
    "vllm_ascend/kv_offload/kv_loader.py"
    "vllm_ascend/ops/fused_moe/fused_moe.py"
    "vllm_ascend/ops/mla.py"
    "vllm_ascend/patch/worker/patch_deepseek_mtp.py"
    "vllm_ascend/worker/dmp_context.py"
    "vllm_ascend/worker/model_runner_v1.py"
)

if [[ ! -d "$TARGET_ROOT/vllm_ascend" ]]; then
    echo "ERROR: target is not a vllm-ascend source tree: $TARGET_ROOT" >&2
    exit 1
fi

changed=0
for file in "${FILES[@]}"; do
    source_file="$SOURCE_ROOT/$file"
    target_file="$TARGET_ROOT/$file"

    if [[ ! -f "$source_file" ]]; then
        echo "ERROR: source file is missing: $source_file" >&2
        exit 1
    fi
    if cmp -s "$source_file" "$target_file"; then
        printf "UNCHANGED        %s\n" "$file"
        continue
    fi

    if [[ -f "$target_file" ]]; then
        mkdir -p "$BACKUP_ROOT/$(dirname "$file")"
        cp -a "$target_file" "$BACKUP_ROOT/$file"
    fi
    mkdir -p "$(dirname "$target_file")"
    cp -a "$source_file" "$target_file"
    printf "UPDATED          %s\n" "$file"
    changed=1
done

python3 -m py_compile "${FILES[@]/#/$TARGET_ROOT/}"

echo
echo "DMP Python overlay synchronized to: $TARGET_ROOT"
if [[ "$changed" -eq 1 ]]; then
    echo "Previous files backed up under: $BACKUP_ROOT"
else
    echo "No files changed."
fi
echo "Native/custom operator directories were not touched."
