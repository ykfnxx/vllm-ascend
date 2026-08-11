#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DEVICE="npu:0"
INSTALL_ROOT="${SCRIPT_DIR}/.install"
REQUESTS=32
MISS_RATE=10
MISS_COUNT=""
SEED=1234
PROFILER_WARMUP=10
LAUNCH_COUNT=1
KERNEL_NAME="DsaSparseLookupUpdate"
OUTPUT_ROOT="${SCRIPT_DIR}/roofline_profiles"
TOOL="auto"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: profile_roofline.sh [OPTIONS]

Collect a Roofline profile for the standalone Ascend 950
DsaSparseLookupUpdate SIMT operator. The script wraps benchmark_operator.py
with CANN msopprof (or the compatible "msprof op" entry point) and produces
visualize_data.bin for MindStudio Insight.

Options:
  --device DEVICE         NPU device used by the benchmark (default: npu:0).
  --install-root PATH     Isolated custom-op install root.
  --requests N            Concurrent request rows (default: 32).
  --miss-rate PERCENT     Miss percentage in each 2K query (default: 10).
  --miss-count N          Exact misses in each 2K query; mutually exclusive
                          with --miss-rate.
  --seed N                Random workload seed (default: 1234).
  --warm-up N             msopprof warm-up count (default: 10).
  --launch-count N        Target kernel launches to collect (default: 1).
  --kernel-name NAME      Device kernel name filter
                          (default: DsaSparseLookupUpdate).
  --output-dir PATH       Result root (default: tools/.../roofline_profiles).
  --tool NAME             auto, msopprof, or msprof-op (default: auto).
  --dry-run               Print the resolved command without executing it.
  -h, --help              Show this message.
EOF
}

require_value() {
    if (($# < 2)); then
        echo "ERROR: $1 requires a value." >&2
        exit 2
    fi
}

MISS_RATE_SET=0
MISS_COUNT_SET=0
while (($# > 0)); do
    case "$1" in
        --device)
            require_value "$@"
            DEVICE="$2"
            shift 2
            ;;
        --install-root)
            require_value "$@"
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --requests)
            require_value "$@"
            REQUESTS="$2"
            shift 2
            ;;
        --miss-rate)
            require_value "$@"
            MISS_RATE="$2"
            MISS_RATE_SET=1
            shift 2
            ;;
        --miss-count)
            require_value "$@"
            MISS_COUNT="$2"
            MISS_COUNT_SET=1
            shift 2
            ;;
        --seed)
            require_value "$@"
            SEED="$2"
            shift 2
            ;;
        --warm-up)
            require_value "$@"
            PROFILER_WARMUP="$2"
            shift 2
            ;;
        --launch-count)
            require_value "$@"
            LAUNCH_COUNT="$2"
            shift 2
            ;;
        --kernel-name)
            require_value "$@"
            KERNEL_NAME="$2"
            shift 2
            ;;
        --output-dir)
            require_value "$@"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --tool)
            require_value "$@"
            TOOL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((MISS_RATE_SET && MISS_COUNT_SET)); then
    echo "ERROR: --miss-rate and --miss-count are mutually exclusive." >&2
    exit 2
fi

case "${TOOL}" in
    auto | msopprof | msprof-op)
        ;;
    *)
        echo "ERROR: --tool must be auto, msopprof, or msprof-op." >&2
        exit 2
        ;;
esac

for value_name in REQUESTS SEED PROFILER_WARMUP LAUNCH_COUNT; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${value_name,,} must be a non-negative integer." >&2
        exit 2
    fi
done
if ((REQUESTS == 0 || LAUNCH_COUNT == 0)); then
    echo "ERROR: requests and launch-count must be greater than zero." >&2
    exit 2
fi

if ((MISS_COUNT_SET)); then
    if [[ ! "${MISS_COUNT}" =~ ^[0-9]+$ ]] || ((MISS_COUNT > 2048)); then
        echo "ERROR: miss-count must be an integer in [0, 2048]." >&2
        exit 2
    fi
else
    if [[ ! "${MISS_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "ERROR: miss-rate must be a number in [0, 100]." >&2
        exit 2
    fi
    if ! awk -v value="${MISS_RATE}" 'BEGIN { exit !(value >= 0 && value <= 100) }'; then
        echo "ERROR: miss-rate must be a number in [0, 100]." >&2
        exit 2
    fi
fi

find_msopprof() {
    if command -v msopprof >/dev/null 2>&1; then
        command -v msopprof
        return
    fi

    local ascend_root
    for ascend_root in \
        "${ASCEND_HOME:-}" \
        "${ASCEND_HOME_PATH:-}" \
        "${ASCEND_TOOLKIT_HOME:-}"; do
        if [[ -n "${ascend_root}" &&
              -x "${ascend_root}/tools/msopprof/bin/msopprof" ]]; then
            printf '%s\n' "${ascend_root}/tools/msopprof/bin/msopprof"
            return
        fi
    done
    return 1
}

PROFILE_COMMAND=()
case "${TOOL}" in
    msopprof)
        if MSPROF_BIN="$(find_msopprof)"; then
            PROFILE_COMMAND=("${MSPROF_BIN}")
        elif ((DRY_RUN)); then
            PROFILE_COMMAND=(msopprof)
        else
            echo "ERROR: msopprof is not available in the CANN environment." >&2
            exit 1
        fi
        ;;
    msprof-op)
        if command -v msprof >/dev/null 2>&1; then
            PROFILE_COMMAND=("$(command -v msprof)" op)
        elif ((DRY_RUN)); then
            PROFILE_COMMAND=(msprof op)
        else
            echo "ERROR: msprof is not available in the CANN environment." >&2
            exit 1
        fi
        ;;
    auto)
        if MSPROF_BIN="$(find_msopprof)"; then
            PROFILE_COMMAND=("${MSPROF_BIN}")
        elif command -v msprof >/dev/null 2>&1; then
            PROFILE_COMMAND=("$(command -v msprof)" op)
        elif ((DRY_RUN)); then
            PROFILE_COMMAND=(msopprof)
        else
            echo "ERROR: neither msopprof nor msprof is available." >&2
            echo "Source the CANN set_env.sh script and retry." >&2
            exit 1
        fi
        ;;
esac

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT%/}/${TIMESTAMP}"
BENCHMARK_OUTPUT="${RUN_DIR}/benchmark.json"

PROFILE_COMMAND+=(
    "--output=${RUN_DIR}"
    "--warm-up=${PROFILER_WARMUP}"
    "--launch-count=${LAUNCH_COUNT}"
    "--aic-metrics=Roofline"
    "--kernel-name=${KERNEL_NAME}"
)

APP_COMMAND=(
    python3
    "${SCRIPT_DIR}/benchmark_operator.py"
    --device "${DEVICE}"
    --install-root "${INSTALL_ROOT}"
    --operator simt
    --concurrency "${REQUESTS}"
    --scenario churn
    --seed "${SEED}"
    --warmup 1
    --iterations 1
    --output "${BENCHMARK_OUTPUT}"
)
if ((MISS_COUNT_SET)); then
    APP_COMMAND+=(--miss-count "${MISS_COUNT}")
else
    APP_COMMAND+=(--miss-rate "${MISS_RATE}")
fi

printf 'Roofline command:'
printf ' %q' "${PROFILE_COMMAND[@]}" "${APP_COMMAND[@]}"
printf '\n'

if ((DRY_RUN)); then
    exit 0
fi

mkdir -p -- "${RUN_DIR}"
cd -- "${REPO_ROOT}"
"${PROFILE_COMMAND[@]}" "${APP_COMMAND[@]}"

echo "Roofline profile root: ${RUN_DIR}"
mapfile -t VISUALIZATION_FILES < <(
    find "${RUN_DIR}" -type f -name visualize_data.bin -print | sort
)
if ((${#VISUALIZATION_FILES[@]} == 0)); then
    echo "ERROR: msopprof did not produce visualize_data.bin." >&2
    exit 1
fi

echo "Import one of these files into MindStudio Insight:"
printf '  %s\n' "${VISUALIZATION_FILES[@]}"
