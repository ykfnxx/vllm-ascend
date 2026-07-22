#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCENDC_DIR="$ROOT/csrc"
TORCH_EXTENSION_DIR="$ROOT/torch_extension"
RAW_SOC="${SOC_VERSION:-ascend910_9391}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
OP_NAMES="asu_hbm_index_lookup;asu_hbm_index_maintain_aicpu;dmp_lookup_kv_gather"

case "${RAW_SOC,,}" in
    ascend910_9391) SOC="ascend910_93" ;;
    ascend910b1) SOC="ascend910b" ;;
    *) SOC="${RAW_SOC,,}" ;;
esac

export TMPDIR="$ROOT/.tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export OPS_CPU_NUMBER="$BUILD_JOBS"
mkdir -p "$TMPDIR"

echo "[lookup-maintain] root: $ROOT"
echo "[lookup-maintain] raw soc: $RAW_SOC, build soc: $SOC"
echo "[lookup-maintain] operators: $OP_NAMES"

rm -rf "$ASCENDC_DIR/build" "$ASCENDC_DIR/output"
(
    cd "$ASCENDC_DIR"
    bash build.sh -n "$OP_NAMES" -c "$SOC"
)

(
    cd "$TORCH_EXTENSION_DIR"
    rm -rf build
    rm -f dmp_lookup_maintain_custom_ops/*.so
    MAX_JOBS="$BUILD_JOBS" python3 setup.py build_ext --inplace
)

RUN_PKG="$(find "$ASCENDC_DIR/output" -maxdepth 1 \
    -name 'CANN-custom_ops-*.run' -print | LC_ALL=C sort | head -1)"
if [[ -z "$RUN_PKG" ]]; then
    echo "Custom OPP run package was not generated." >&2
    exit 1
fi

INSTALL_OPP="${DMP_LOOKUP_MAINTAIN_INSTALL_OPP_PATH:-$ROOT/opp}"
mkdir -p "$INSTALL_OPP"
chmod +x "$RUN_PKG"
"$RUN_PKG" --quiet --install-path="$INSTALL_OPP"

if [[ ! -f "$INSTALL_OPP/vendors/customize/op_api/lib/libcust_opapi.so" ]]; then
    echo "Lookup/Maintain op-api library was not installed." >&2
    exit 1
fi
AICPU_REPOSITORY="$INSTALL_OPP/vendors/customize/op_impl/aicpu_transformer"
if [[ ! -f "$AICPU_REPOSITORY/op_impl/cpu/config/cust_aicpu_kernel.json" || \
      ! -f "$AICPU_REPOSITORY/op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so" ]]; then
    echo "Lookup/Maintain AICPU repository was not installed correctly: $AICPU_REPOSITORY" >&2
    exit 1
fi
if ! compgen -G \
    "$TORCH_EXTENSION_DIR/dmp_lookup_maintain_custom_ops/*.so" >/dev/null; then
    echo "Lookup/Maintain torch extension was not built." >&2
    exit 1
fi

echo "DMP_LOOKUP_MAINTAIN_INSTALL_OPP_PATH=$INSTALL_OPP" \
    > "$ROOT/.lookup_maintain_env"
echo "[lookup-maintain] build and persistent install complete"
