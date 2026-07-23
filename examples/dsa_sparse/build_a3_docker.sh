#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(
  realpath "$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
)"
CONTAINER_BUILD_SCRIPT="/workspace/vllm-ascend-src/examples/dsa_sparse/compile_a3_container.sh"

IMAGE="quay.io/ascend/vllm-ascend:v0.18.0-a3"
CONTAINER_NAME="vllm-ascend-dsa-build"
LOG_DIR="/tmp/vllm-ascend-a3-build-logs"
MODEL_DIR=""
SOC_VERSION="ascend910_9391"
MAX_JOBS="16"
PULL_IMAGE=1
CREATE_ONLY=0

usage() {
  cat <<'EOF'
Create or reuse an Atlas A3 (8 cards / 16 dies) vLLM Ascend container,
then compile the current source checkout inside it.

Usage:
  build_a3_docker.sh [options]

Options:
  --image IMAGE             Container image
                            (default: quay.io/ascend/vllm-ascend:v0.18.0-a3)
  --container-name NAME     Container name
                            (default: vllm-ascend-dsa-build)
  --log-dir PATH            Host directory for build logs
                            (default: /tmp/vllm-ascend-a3-build-logs)
  --model-dir PATH          Optionally mount a host model directory at /models
  --soc-version VERSION     Exact A3 SoC version
                            (default: ascend910_9391)
  --max-jobs N              Parallel jobs for the extension build (default: 16)
  --skip-pull               Do not pull the image before creating the container
  --create-only             Create/start the container without compiling
  -h, --help                Show this help

The script always mounts the Git repository containing this file at:
  /workspace/vllm-ascend-src

Examples:
  ./examples/dsa_sparse/build_a3_docker.sh \
    --log-dir /data/vllm-logs

  ./examples/dsa_sparse/build_a3_docker.sh \
    --model-dir /data/models \
    --soc-version ascend910_9391
EOF
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || fail "${option} requires a value"
}

while (($# > 0)); do
  case "$1" in
    --image)
      require_value "$1" "${2:-}"
      IMAGE="$2"
      shift 2
      ;;
    --container-name)
      require_value "$1" "${2:-}"
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --log-dir)
      require_value "$1" "${2:-}"
      LOG_DIR="$2"
      shift 2
      ;;
    --model-dir)
      require_value "$1" "${2:-}"
      MODEL_DIR="$2"
      shift 2
      ;;
    --soc-version)
      require_value "$1" "${2:-}"
      SOC_VERSION="$2"
      shift 2
      ;;
    --max-jobs)
      require_value "$1" "${2:-}"
      MAX_JOBS="$2"
      shift 2
      ;;
    --skip-pull)
      PULL_IMAGE=0
      shift
      ;;
    --create-only)
      CREATE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ "${CONTAINER_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
  fail "invalid container name: ${CONTAINER_NAME}"
[[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]] || \
  fail "--max-jobs must be a positive integer"
[[ "${SOC_VERSION}" =~ ^ascend910_93[0-9]+$ ]] || \
  fail "--soc-version must identify an A3 chip, got: ${SOC_VERSION}"

command -v docker >/dev/null 2>&1 || fail "docker is required"
docker info >/dev/null 2>&1 || fail "cannot access the Docker daemon"

[[ -f "${REPO_ROOT}/setup.py" ]] || \
  fail "repository root is missing setup.py: ${REPO_ROOT}"
[[ -x "${REPO_ROOT}/examples/dsa_sparse/compile_a3_container.sh" ]] || \
  fail "container build helper is not executable"

current_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
current_commit="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
if [[ "${current_branch}" != "dev_lookup_maintain_integration_pd" ]]; then
  warn "current branch is ${current_branch}, expected dev_lookup_maintain_integration_pd"
fi

printf 'Source: %s\n' "${REPO_ROOT}"
printf 'Branch: %s\n' "${current_branch}"
printf 'Commit: %s\n' "${current_commit}"
printf 'Image:  %s\n' "${IMAGE}"

git -C "${REPO_ROOT}" submodule update --init --recursive

mkdir -p "${LOG_DIR}"
LOG_DIR="$(realpath "${LOG_DIR}")"
if [[ -n "${MODEL_DIR}" ]]; then
  [[ -d "${MODEL_DIR}" ]] || fail "model directory does not exist: ${MODEL_DIR}"
  MODEL_DIR="$(realpath "${MODEL_DIR}")"
fi

for die_id in {0..15}; do
  [[ -e "/dev/davinci${die_id}" ]] || \
    fail "missing A3 logical device: /dev/davinci${die_id}"
done
for shared_device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ -e "${shared_device}" ]] || fail "missing Ascend device: ${shared_device}"
done

container_id="$(
  docker ps -aq --filter "name=^/${CONTAINER_NAME}$"
)"
if [[ -n "${container_id}" ]]; then
  mounted_source="$(
    docker inspect --format \
      '{{range .Mounts}}{{if eq .Destination "/workspace/vllm-ascend-src"}}{{.Source}}{{end}}{{end}}' \
      "${CONTAINER_NAME}"
  )"
  [[ "${mounted_source}" == "${REPO_ROOT}" ]] || \
    fail "container ${CONTAINER_NAME} mounts ${mounted_source:-<nothing>} instead of ${REPO_ROOT}"
  if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
  printf 'Reusing container: %s\n' "${CONTAINER_NAME}"
else
  if ((PULL_IMAGE)); then
    docker pull "${IMAGE}"
  fi

  docker_args=(
    run -dit
    --name "${CONTAINER_NAME}"
    --net=host
    --shm-size=1g
  )
  for die_id in {0..15}; do
    docker_args+=(--device "/dev/davinci${die_id}")
  done
  docker_args+=(
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
  )

  add_optional_mount() {
    local source_path="$1"
    local destination_path="$2"
    local mount_mode="$3"
    if [[ -e "${source_path}" ]]; then
      docker_args+=(
        -v "${source_path}:${destination_path}:${mount_mode}"
      )
    else
      warn "optional host path is missing and will not be mounted: ${source_path}"
    fi
  }

  add_optional_mount /usr/local/dcmi /usr/local/dcmi ro
  add_optional_mount \
    /usr/local/Ascend/driver/tools/hccn_tool \
    /usr/local/Ascend/driver/tools/hccn_tool \
    ro
  add_optional_mount /usr/local/bin/npu-smi /usr/local/bin/npu-smi ro
  add_optional_mount \
    /usr/local/Ascend/driver/lib64 \
    /usr/local/Ascend/driver/lib64 \
    ro
  add_optional_mount \
    /usr/local/Ascend/driver/version.info \
    /usr/local/Ascend/driver/version.info \
    ro
  add_optional_mount \
    /etc/ascend_install.info \
    /etc/ascend_install.info \
    ro
  add_optional_mount /etc/hccn.conf /etc/hccn.conf ro

  docker_args+=(
    -v "${REPO_ROOT}:/workspace/vllm-ascend-src"
    -v "${LOG_DIR}:/logs"
    -v "/root/.cache:/root/.cache"
    -w /workspace/vllm-ascend-src
  )
  if [[ -n "${MODEL_DIR}" ]]; then
    docker_args+=(-v "${MODEL_DIR}:/models:ro")
  fi

  docker "${docker_args[@]}" "${IMAGE}" bash >/dev/null
  printf 'Created container: %s\n' "${CONTAINER_NAME}"
fi

device_count="$(
  docker exec "${CONTAINER_NAME}" \
    bash -lc 'find /dev -maxdepth 1 -name "davinci[0-9]*" | wc -l'
)"
[[ "${device_count//[[:space:]]/}" == "16" ]] || \
  fail "container sees ${device_count//[[:space:]]/} Davinci devices instead of 16"

if ((CREATE_ONLY)); then
  printf 'Container is ready. Enter with:\n'
  printf '  docker exec -it %q bash\n' "${CONTAINER_NAME}"
  exit 0
fi

docker exec \
  -e "SOC_VERSION=${SOC_VERSION}" \
  -e "MAX_JOBS=${MAX_JOBS}" \
  -e "BUILD_LOG_PATH=/logs/vllm-ascend-build.log" \
  "${CONTAINER_NAME}" \
  bash "${CONTAINER_BUILD_SCRIPT}"

printf 'Build completed. Host log: %s/vllm-ascend-build.log\n' "${LOG_DIR}"
printf 'Enter the container with:\n'
printf '  docker exec -it %q bash\n' "${CONTAINER_NAME}"
