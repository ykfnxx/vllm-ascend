#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_MODEL="${SOURCE_MODEL:-/models/GLM-5.1-w4a8}"
DMP_MIN_MOE_LAYERS="${DMP_MIN_MOE_LAYERS:-1}"

if [[ ! -f "$SOURCE_MODEL/config.json" ]]; then
    echo "Source model is missing: $SOURCE_MODEL" >&2
    exit 1
fi

layer_args=(
    --config "$SOURCE_MODEL/config.json"
    --minimum-moe-layers "$DMP_MIN_MOE_LAYERS"
)
if [[ -n "${REDUCED_LAYERS:-}" ]]; then
    layer_args+=(--layers "$REDUCED_LAYERS")
fi
REDUCED_LAYERS="$(
    python3 "$SCRIPT_DIR/select_reduced_layer_count.py" "${layer_args[@]}"
)"
REDUCED_MODEL_PATH="${REDUCED_MODEL_PATH:-/models-reduced/GLM-5.1-w4a8-${REDUCED_LAYERS}layers-dmp-r2}"

python3 "$SCRIPT_DIR/create_reduced_glm_model.py" \
    --source "$SOURCE_MODEL" \
    --output "$REDUCED_MODEL_PATH" \
    --layers "$REDUCED_LAYERS"

python3 - "$REDUCED_MODEL_PATH" "$REDUCED_LAYERS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
index = json.loads(
    (root / "model.safetensors.index.json").read_text(encoding="utf-8")
)
assert int(config["num_hidden_layers"]) == expected
assert index["weight_map"]
assert "rot.weight" not in index["weight_map"]
marker = json.loads((root / "DMP_REDUCED_MODEL.json").read_text(encoding="utf-8"))
assert int(marker["format_version"]) == 2
missing = sorted(
    {name for name in index["weight_map"].values() if not (root / name).is_file()}
)
assert not missing, missing
print(
    f"REDUCED_MODEL_CHECK_OK: path={root} layers={expected} "
    f"tensors={len(index['weight_map'])}"
)
PY

echo
echo "Use this model for the single-card run:"
echo "MODEL_PATH=$REDUCED_MODEL_PATH"
