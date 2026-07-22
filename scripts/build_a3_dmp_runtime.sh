#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
LOG_DIR="$SCRIPT_DIR/logs/a3_runtime_build_$(date +%Y%m%d_%H%M%S)"
DUAL_STAMP="$SCRIPT_DIR/dmp-runtime/.a3-dual-attention-r4"
LOOKUP_STAMP="$SCRIPT_DIR/dmp-lookup-maintain/opp/.a3-lookup-maintain-r5"
FORCE_REBUILD="${DMP_FORCE_REBUILD:-0}"
mkdir -p "$LOG_DIR"
echo "A3 runtime preparation log: $LOG_DIR"

export VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export OPP_COMPUTE_UNIT="${OPP_COMPUTE_UNIT:-ascend910_93}"
export SOC_VERSION="${SOC_VERSION:-ascend910_9391}"

bash "$SCRIPT_DIR/check_a3_single_card_prereqs.sh" \
    | tee "$LOG_DIR/prerequisites.log"

dual_artifacts_present() {
    [[ -f "$SCRIPT_DIR/dmp-runtime/opp/vendors/customize/op_api/lib/libcust_opapi.so" ]] &&
    compgen -G "$SCRIPT_DIR/dmp-runtime/python/custom_ops/*.so" >/dev/null
}

dual_ready() {
    [[ -f "$DUAL_STAMP" ]] &&
    [[ "$(<"$DUAL_STAMP")" == "A3_DUAL_ATTENTION_RUNTIME_REVISION=4" ]] &&
    dual_artifacts_present
}

lookup_artifacts_present() {
    local vendor="$SCRIPT_DIR/dmp-lookup-maintain/opp/vendors/customize"
    local aicpu="$vendor/op_impl/aicpu_transformer"
    [[ -f "$vendor/op_api/lib/libcust_opapi.so" ]] &&
    [[ -f "$aicpu/op_impl/cpu/config/cust_aicpu_kernel.json" ]] &&
    [[ -f "$aicpu/op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so" ]] &&
    compgen -G \
        "$SCRIPT_DIR/dmp-lookup-maintain/torch_extension/dmp_lookup_maintain_custom_ops/*.so" \
        >/dev/null
}

lookup_ready() {
    [[ -f "$LOOKUP_STAMP" ]] &&
    [[ "$(<"$LOOKUP_STAMP")" == "A3_LOOKUP_MAINTAIN_RUNTIME_REVISION=5" ]] &&
    lookup_artifacts_present
}

if [[ "$FORCE_REBUILD" == "1" ]] || ! dual_ready; then
    if [[ "$FORCE_REBUILD" != "1" ]] && dual_artifacts_present; then
        echo "[1/3] Adopting and smoke-testing the existing Dual-Attention runtime."
        DMP_DUAL_ATTENTION_SKIP_BUILD=1 \
            bash "$SCRIPT_DIR/build_dmp_dual_attention_ops.sh" 2>&1 \
            | tee "$LOG_DIR/dual_attention.log"
    else
        echo "[1/3] Building and smoke-testing Dual-Attention operators for A3."
        bash "$SCRIPT_DIR/build_dmp_dual_attention_ops.sh" 2>&1 \
            | tee "$LOG_DIR/dual_attention.log"
    fi
    mkdir -p "$(dirname "$DUAL_STAMP")"
    printf '%s\n' 'A3_DUAL_ATTENTION_RUNTIME_REVISION=4' > "$DUAL_STAMP"
else
    echo "[1/3] Dual-Attention runtime is already complete; skipping rebuild."
fi

if [[ "$FORCE_REBUILD" == "1" ]] || ! lookup_ready; then
    if [[ "$FORCE_REBUILD" != "1" ]] && lookup_artifacts_present; then
        echo "[2/3] Adopting and smoke-testing the existing Lookup/Maintain runtime."
        DMP_LOOKUP_MAINTAIN_SKIP_BUILD=1 \
            bash "$SCRIPT_DIR/build_dmp_lookup_maintain.sh" 2>&1 \
            | tee "$LOG_DIR/lookup_maintain.log"
    else
        echo "[2/3] Building and smoke-testing Lookup/Maintain operators for A3."
        bash "$SCRIPT_DIR/build_dmp_lookup_maintain.sh" 2>&1 \
            | tee "$LOG_DIR/lookup_maintain.log"
    fi
    mkdir -p "$(dirname "$LOOKUP_STAMP")"
    printf '%s\n' 'A3_LOOKUP_MAINTAIN_RUNTIME_REVISION=5' > "$LOOKUP_STAMP"
else
    echo "[2/3] Lookup/Maintain runtime is already complete; skipping rebuild."
fi

# This also deploys the offline Transformers wheels if the persistent model
# runtime was removed independently of the operator builds.
bash "$SCRIPT_DIR/install_dmp_dual_attention_runtime.sh" \
    | tee "$LOG_DIR/persistent_install.log"

echo "[3/3] Validating A3 libraries, Python extensions, and registrations."
bash "$SCRIPT_DIR/validate_a3_dmp_runtime.sh" 2>&1 \
    | tee "$LOG_DIR/final_validation.log"

echo "A3_DMP_RUNTIME_READY: $LOG_DIR"
