#!/usr/bin/env bash
# Build once when native sources change; otherwise reuse the installed runtime.

set -Eeuo pipefail

REPO_ROOT="${A5_REPO_ROOT_IN_CONTAINER:-/vllm-workspace/vllm-ascend}"
DEVICE="${A5_DECODE_DEVICE:-5}"
MAX_JOBS="${MAX_JOBS:-8}"
LOG_ROOT="${A5_LOG_ROOT_IN_CONTAINER:-$REPO_ROOT/scripts/results}"
STAMP="${A5_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$LOG_ROOT/${STAMP}_a5_kvgather_runtime"
FULL_LOG="$OUT_DIR/${STAMP}_runtime.full.log"
RESULT_FILE="$OUT_DIR/${STAMP}_runtime_result.txt"
MARKER="${A5_NATIVE_MARKER:-$REPO_ROOT/.a5-kvgathersim-native-source.sha256}"
CATLASS_HEADER="$REPO_ROOT/csrc/third_party/catlass/include/catlass/catlass.hpp"
CUSTOM_OP_VENDOR_PATH="$REPO_ROOT/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
CUSTOM_OP_API_DIR="$CUSTOM_OP_VENDOR_PATH/op_api/lib"
CUSTOM_OP_API_LIB="$CUSTOM_OP_API_DIR/libcust_opapi.so"
CUSTOM_OP_KERNEL_ROOT="$CUSTOM_OP_VENDOR_PATH/op_impl/ai_core/tbe/kernel"
CUSTOM_OP_INDEX_TOOL="$REPO_ROOT/scripts/a5_kvgather_kernel_index.py"

mkdir -p "$OUT_DIR" "$(dirname "$MARKER")"
exec > >(tee "$FULL_LOG") 2>&1

on_failure() {
    local status=$?
    if ((status != 0)); then
        {
            echo "A5_KVGATHER_RUNTIME_FAILED: status=$status"
            grep -nE -m 80 \
                'CMake Error|FAILED:|error:|Error:|Traceback|RuntimeError|ImportError|No such file|not found' \
                "$FULL_LOG" || true
            echo "full_log=$FULL_LOG"
        } >"$RESULT_FILE"
    fi
}
trap on_failure EXIT

if [[ -r /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
    # Ascend environment scripts are sourced with nounset temporarily disabled.
    set +u
    source /usr/local/Ascend/cann-9.1.0/set_env.sh
    if [[ -r /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
        source /usr/local/Ascend/nnal/atb/set_env.sh
    fi
    set -u
else
    echo "CANN 9.1 environment is missing." >&2
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export MAX_JOBS
export COMPILE_CUSTOM_KERNELS=1
export VLLM_ASCEND_OFFLINE_BUILD=1

configure_custom_op_env() {
    [[ -r "$CUSTOM_OP_API_LIB" ]] || {
        echo "A5_CUSTOM_OP_API_MISSING: $CUSTOM_OP_API_LIB" >&2
        return 1
    }
    if [[ ":${ASCEND_CUSTOM_OPP_PATH:-}:" != *":$CUSTOM_OP_VENDOR_PATH:"* ]]; then
        export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OP_VENDOR_PATH${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    fi
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$CUSTOM_OP_API_DIR:"* ]]; then
        export LD_LIBRARY_PATH="$CUSTOM_OP_API_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    echo "A5_CUSTOM_OP_API_READY: path=$CUSTOM_OP_API_LIB"
}

ensure_custom_op_kernel_index() {
    [[ -r "$CUSTOM_OP_INDEX_TOOL" ]] || {
        echo "A5_CUSTOM_OP_KERNEL_INDEX_TOOL_MISSING: $CUSTOM_OP_INDEX_TOOL" >&2
        return 1
    }
    python3 "$CUSTOM_OP_INDEX_TOOL" \
        --kernel-root "$CUSTOM_OP_KERNEL_ROOT" \
        --repair \
        --repo-root "$REPO_ROOT" \
        --backup-dir "$OUT_DIR/kernel-index-before-repair"
}

NATIVE_FINGERPRINT="$(
    {
        find "$REPO_ROOT/csrc" "$REPO_ROOT/cmake" -type f \
            \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' \
               -o -name '*.h' -o -name '*.hpp' -o -name '*.inc' \
               -o -name '*.py' -o -name '*.sh' -o -name '*.cmake' \
               -o -name '*.json' -o -name 'CMakeLists.txt' \) \
            ! -path '*/third_party/*' ! -path '*/build/*' \
            ! -path '*/build_out/*' ! -path '*/output/*' \
            ! -path '*/_cann_ops_custom/*' -print0
        for build_input in CMakeLists.txt pyproject.toml setup.py; do
            [[ ! -f "$REPO_ROOT/$build_input" ]] || \
                printf '%s\0' "$REPO_ROOT/$build_input"
        done
    } | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
)"

runtime_ok() {
    python3 - "$REPO_ROOT" <<'PY'
import ctypes
import os
import pathlib
import sys

import torch
import torch_npu  # noqa: F401
import vllm_ascend

root = pathlib.Path(sys.argv[1]).resolve()
loaded = pathlib.Path(vllm_ascend.__file__).resolve()
if root not in loaded.parents:
    raise RuntimeError(f"loaded source {loaded} is not below {root}")
custom_op_api = (
    root
    / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
)
handle = ctypes.CDLL(
    str(custom_op_api), mode=os.RTLD_LOCAL | os.RTLD_LAZY
)
for symbol in ("aclnnAsuKvGather", "aclnnAsuKvGatherGetWorkspaceSize"):
    getattr(handle, symbol)
    print(f"A5_CUSTOM_OP_SYMBOL_READY: name={symbol} library={custom_op_api}")

import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

for name in ("dsa_offload_lookup_update_batch", "asu_kv_gather"):
    packet = getattr(torch.ops._C_ascend, name)
    print(f"A5_NATIVE_OP_READY: name={name} schema={packet.default._schema}")
PY
}

operator_smoke() {
    python3 - <<'PY'
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

device = "npu:0"
block_size = 128
source_kv = torch.ones(
    (1, block_size, 16), dtype=torch.bfloat16, device=device
)
source_rope = source_kv.clone()
destination_kv = torch.zeros_like(source_kv)
destination_rope = torch.zeros_like(source_rope)
table = torch.zeros((1, 1), dtype=torch.int32, device=device)
pool = torch.zeros((1,), dtype=torch.int32, device=device)
positions = torch.tensor([[7]], dtype=torch.int32, device=device)
slots = torch.tensor([[11]], dtype=torch.int32, device=device)
miss = torch.ones((1, 1), dtype=torch.int32, device=device)
torch.ops._C_ascend.asu_kv_gather(
    destination_kv,
    destination_rope,
    table,
    source_kv,
    source_rope,
    table,
    pool,
    positions,
    slots,
    miss,
    block_size,
    1,
)
torch.npu.synchronize()
if not torch.equal(destination_kv[0, 11], source_kv[0, 7]):
    raise RuntimeError("asu_kv_gather KV smoke mismatch")
if not torch.equal(destination_rope[0, 11], source_rope[0, 7]):
    raise RuntimeError("asu_kv_gather RoPE smoke mismatch")
print("A5_ASU_KV_GATHER_SIM_SMOKE_OK")
PY
}

ensure_offline_abseil_headers() {
    local abseil_dir="$REPO_ROOT/csrc/third_party/abseil-cpp"
    local required_header="$abseil_dir/absl/base/attributes.h"
    local archive="$REPO_ROOT/csrc/third_party/pkg/abseil-cpp-20230802.1.tar.gz"
    local patch_file="$REPO_ROOT/csrc/cmake/third_party/build/modules/patch/protobuf-hide_absl_symbols.patch"

    if [[ -r "$required_header" && \
          -r "$abseil_dir/absl/log/absl_log.h" && \
          -r "$abseil_dir/absl/base/options.h" && \
          -r "$abseil_dir/absl/hash/internal/hash.h" && \
          -r "$abseil_dir/absl/hash/internal/hash.cc" ]] && \
          grep -qF 'set(BUILD_SHARED_LIBS OFF)' \
              "$abseil_dir/CMakeLists.txt" && \
          grep -qF '#define ABSL_OPTION_INLINE_NAMESPACE_NAME lts_ascend_private' \
              "$abseil_dir/absl/base/options.h"; then
        echo "A5_OFFLINE_ABSEIL_READY: root=$abseil_dir"
        return 0
    fi
    [[ -r "$archive" && -r "$patch_file" ]] || {
        echo "A5_OFFLINE_ABSEIL_REPAIR_INPUT_MISSING: archive=$archive patch=$patch_file" >&2
        return 1
    }

    (
        set -Eeuo pipefail
        local repair_dir source_root entry
        local -a header_matches=()
        repair_dir="$(mktemp -d "$REPO_ROOT/csrc/third_party/.abseil-repair.XXXXXX")"
        cleanup_abseil_repair() {
            case "$repair_dir" in
                "$REPO_ROOT"/csrc/third_party/.abseil-repair.*)
                    rm -rf -- "$repair_dir"
                    ;;
                *)
                    echo "Refusing to clean unexpected repair directory: $repair_dir" >&2
                    ;;
            esac
        }
        trap cleanup_abseil_repair EXIT

        while IFS= read -r entry; do
            case "$entry" in
                /*|../*|*/../*)
                    echo "Unsafe path in offline Abseil archive: $entry" >&2
                    exit 1
                    ;;
            esac
        done < <(tar -tzf "$archive")
        tar -xzf "$archive" -C "$repair_dir"
        while IFS= read -r -d '' entry; do
            header_matches+=("$entry")
        done < <(find "$repair_dir" -type f \
            -path '*/absl/base/attributes.h' -print0)
        [[ "${#header_matches[@]}" == "1" ]] || {
            echo "Offline Abseil archive has ${#header_matches[@]} attributes.h matches; expected 1." >&2
            exit 1
        }
        source_root="${header_matches[0]%/absl/base/attributes.h}"
        [[ -r "$source_root/CMakeLists.txt" && \
           -r "$source_root/absl/log/absl_log.h" ]] || {
            echo "Offline Abseil archive layout is incomplete: $source_root" >&2
            exit 1
        }

        mkdir -p "$abseil_dir"
        cp -a "$source_root/." "$abseil_dir/"
        if patch --dry-run --silent -R -d "$abseil_dir" -p1 \
                <"$patch_file" >/dev/null 2>&1; then
            echo "A5_OFFLINE_ABSEIL_PATCH_ALREADY_APPLIED"
        elif patch --dry-run --silent -d "$abseil_dir" -p1 \
                <"$patch_file" >/dev/null 2>&1; then
            patch --silent -d "$abseil_dir" -p1 <"$patch_file"
            echo "A5_OFFLINE_ABSEIL_PATCH_APPLIED"
        else
            echo "Offline Abseil patch cannot be applied cleanly." >&2
            exit 1
        fi
        [[ -r "$required_header" && \
           -r "$abseil_dir/absl/log/absl_log.h" && \
           -r "$abseil_dir/absl/base/options.h" && \
           -r "$abseil_dir/absl/hash/internal/hash.h" && \
           -r "$abseil_dir/absl/hash/internal/hash.cc" ]] && \
           grep -qF 'set(BUILD_SHARED_LIBS OFF)' \
               "$abseil_dir/CMakeLists.txt" && \
           grep -qF '#define ABSL_OPTION_INLINE_NAMESPACE_NAME lts_ascend_private' \
               "$abseil_dir/absl/base/options.h" || {
            echo "Offline Abseil repair did not restore required headers." >&2
            exit 1
        }
        echo "A5_OFFLINE_ABSEIL_REPAIRED: archive=$archive root=$abseil_dir"
    )
}

if [[ "${A5_FORCE_REBUILD:-0}" != "1" && -r "$MARKER" && \
      "$(<"$MARKER")" == "$NATIVE_FINGERPRINT" ]] && \
      configure_custom_op_env && ensure_custom_op_kernel_index && runtime_ok; then
    operator_smoke
    echo "A5_KVGATHER_RUNTIME_REUSED: native_fingerprint=$NATIVE_FINGERPRINT" \
        | tee "$RESULT_FILE"
    exit 0
fi

# An installed runtime is reused only after binding and kernel validation.
if [[ "${A5_FORCE_REBUILD:-0}" != "1" && ! -e "$MARKER" && ! -L "$MARKER" ]] && \
      configure_custom_op_env && ensure_custom_op_kernel_index && runtime_ok; then
    operator_smoke
    printf '%s\n' "$NATIVE_FINGERPRINT" >"$MARKER"
    {
        echo "A5_KVGATHER_RUNTIME_RECOVERED"
        echo "native_fingerprint=$NATIVE_FINGERPRINT"
        echo "device=$DEVICE"
        echo "full_log=$FULL_LOG"
    } | tee "$RESULT_FILE"
    exit 0
fi

[[ -r "$CATLASS_HEADER" ]] || {
    echo "A5_OFFLINE_DEPENDENCY_MISSING: $CATLASS_HEADER" >&2
    echo "The sync package intentionally does not download dependencies." >&2
    echo "Keep the existing csrc/third_party cache in $REPO_ROOT." >&2
    exit 1
}
for dependency in \
    csrc/third_party/pkg/makeself-release-2.5.0-patch1.tar.gz \
    csrc/third_party/pkg/include.zip \
    csrc/third_party/pkg/protobuf-25.1.tar.gz \
    csrc/third_party/pkg/abseil-cpp-20230802.1.tar.gz \
    csrc/third_party/pkg/libboundscheck-v1.1.16.tar.gz; do
    [[ -r "$REPO_ROOT/$dependency" ]] || {
        echo "A5_OFFLINE_DEPENDENCY_MISSING: $REPO_ROOT/$dependency" >&2
        exit 1
    }
done
echo "A5_OFFLINE_DEPENDENCIES_REUSED: root=$REPO_ROOT/csrc/third_party"
ensure_offline_abseil_headers

echo "A5_KVGATHER_RUNTIME_REBUILD_REQUIRED: native ABI/source changed"
echo "No dependency download will be attempted; the existing offline cache is reused."
python3 -m pip install -e "$REPO_ROOT" --no-deps --no-build-isolation
configure_custom_op_env
ensure_custom_op_kernel_index
runtime_ok
operator_smoke
printf '%s\n' "$NATIVE_FINGERPRINT" >"$MARKER"
{
    echo "A5_KVGATHER_RUNTIME_BUILD_OK"
    echo "native_fingerprint=$NATIVE_FINGERPRINT"
    echo "device=$DEVICE"
    echo "full_log=$FULL_LOG"
} | tee "$RESULT_FILE"
