#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCENDC_DIR="${ROOT}/csrc"
TORCH_EXTENSION_DIR="${ROOT}/torch_extension"
ENV_FILE="${ROOT}/.lightning_indexer_decode_env"

RAW_SOC="${SOC_VERSION:-ascend910_9391}"
RAW_SOC_LOWER="${RAW_SOC,,}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
OP_NAME="${LIGHTNING_INDEXER_DECODE_OP_NAME:-ALL}"
OPS_COMPILE_OPTIONS="${LIGHTNING_INDEXER_DECODE_OPS_COMPILE_OPTIONS:-}"
CLEAN_ASCENDC_BUILD="${CLEAN_ASCENDC_BUILD:-1}"
EVICT_EXTRA_SCAN_CHUNKS_RAW="${EVICT_EXTRA_SCAN_CHUNKS:-8}"

if [[ ! "${EVICT_EXTRA_SCAN_CHUNKS_RAW}" =~ ^[0-9]+$ ]]; then
  echo "[lightning-indexer-decode] ERROR: EVICT_EXTRA_SCAN_CHUNKS must be an integer in [0, 512]." >&2
  exit 1
fi
EVICT_EXTRA_SCAN_CHUNKS_VALUE="$((10#${EVICT_EXTRA_SCAN_CHUNKS_RAW}))"
if (( EVICT_EXTRA_SCAN_CHUNKS_VALUE > 512 )); then
  echo "[lightning-indexer-decode] ERROR: EVICT_EXTRA_SCAN_CHUNKS must be an integer in [0, 512]." >&2
  exit 1
fi

case "${RAW_SOC_LOWER}" in
  ascend910_9391)
    SOC="ascend910_93"
    ;;
  ascend910b1)
    SOC="ascend910b"
    ;;
  *)
    SOC="${RAW_SOC_LOWER}"
    ;;
esac

OPS_COMPILE_OPTIONS_COMBINED="${OPS_COMPILE_OPTIONS:+${OPS_COMPILE_OPTIONS} }-DLI_DECODE_UPDATE_EVICT_EXTRA_SCAN_CHUNKS=${EVICT_EXTRA_SCAN_CHUNKS_VALUE}"

export TMPDIR="${ROOT}/.tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

echo "[lightning-indexer-decode] root: ${ROOT}"
echo "[lightning-indexer-decode] raw soc: ${RAW_SOC}, build soc: ${SOC}"
echo "[lightning-indexer-decode] build jobs: ${BUILD_JOBS}"
echo "[lightning-indexer-decode] op name: ${OP_NAME}"
echo "[lightning-indexer-decode] evict extra scan chunks: ${EVICT_EXTRA_SCAN_CHUNKS_VALUE}"
echo "[lightning-indexer-decode] clean ascendc build: ${CLEAN_ASCENDC_BUILD}"
echo "[lightning-indexer-decode] ops compile options: ${OPS_COMPILE_OPTIONS_COMBINED}"
echo "[lightning-indexer-decode] ascendc build dir: ${ASCENDC_DIR}/build"
echo "[lightning-indexer-decode] ascendc output dir: ${ASCENDC_DIR}/output"
echo "[lightning-indexer-decode] torch build dir: ${TORCH_EXTENSION_DIR}/build"

export OPS_CPU_NUMBER="${BUILD_JOBS}"

if [[ "${CLEAN_ASCENDC_BUILD}" == "1" ]]; then
  case "${ASCENDC_DIR}" in
    "${ROOT}/csrc")
      rm -rf "${ASCENDC_DIR}/build" "${ASCENDC_DIR}/output"
      ;;
    *)
      echo "[lightning-indexer-decode] ERROR: refusing to clean unexpected ascendc dir: ${ASCENDC_DIR}" >&2
      exit 1
      ;;
  esac
fi

pushd "${ASCENDC_DIR}" >/dev/null
bash build.sh -n "${OP_NAME}" -c "${SOC}" --ops-compile-options "${OPS_COMPILE_OPTIONS_COMBINED}"
popd >/dev/null

pushd "${TORCH_EXTENSION_DIR}" >/dev/null
rm -rf build
rm -f lightning_indexer_decode_custom_ops/*.so
MAX_JOBS="${BUILD_JOBS}" python3 setup.py build_ext --inplace
popd >/dev/null

ASCENDC_OUTPUT_DIR="${ASCENDC_DIR}/output"
if [[ ! -d "${ASCENDC_OUTPUT_DIR}" ]]; then
  echo "[lightning-indexer-decode] ERROR: output dir was not found after build: ${ASCENDC_OUTPUT_DIR}." >&2
  exit 1
fi

RUN_PKG="$(find "${ASCENDC_OUTPUT_DIR}" -maxdepth 1 -name 'CANN-custom_ops-*.run' | head -n 1)"
if [[ -z "${RUN_PKG}" ]]; then
  echo "[lightning-indexer-decode] ERROR: custom op .run package was not found in ${ASCENDC_OUTPUT_DIR}." >&2
  exit 1
fi

INSTALL_OPP="${LIGHTNING_INDEXER_DECODE_INSTALL_OPP_PATH:-${ROOT}/opp}"
echo "[lightning-indexer-decode] install package: ${RUN_PKG}"
echo "[lightning-indexer-decode] install opp: ${INSTALL_OPP}"
mkdir -p "${INSTALL_OPP}"
export LIGHTNING_INDEXER_DECODE_INSTALL_OPP_PATH="${INSTALL_OPP}"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${INSTALL_OPP}"
{
  printf 'LIGHTNING_INDEXER_DECODE_INSTALL_OPP_PATH=%s\n' "${INSTALL_OPP}"
  printf 'LIGHTNING_INDEXER_DECODE_TORCH_EXTENSION_DIR=%s\n' "${TORCH_EXTENSION_DIR}"
} > "${ENV_FILE}"

if ! compgen -G "${TORCH_EXTENSION_DIR}/lightning_indexer_decode_custom_ops/*.so" >/dev/null; then
  echo "[lightning-indexer-decode] ERROR: torch extension .so was not found after build." >&2
  exit 1
fi

echo "[lightning-indexer-decode] local torch extension: ${TORCH_EXTENSION_DIR}"
echo "[lightning-indexer-decode] tests load it automatically; for external scripts use:"
echo "[lightning-indexer-decode]   PYTHONPATH=${TORCH_EXTENSION_DIR} python your_script.py"
echo "[lightning-indexer-decode] build and install done"
