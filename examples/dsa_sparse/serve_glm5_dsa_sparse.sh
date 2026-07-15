#!/usr/bin/env bash

set -euo pipefail

# Usage: ./examples/dsa_sparse/serve_glm5_dsa_sparse.sh /path/to/GLM-5 [vllm options]
MODEL_PATH="$1"
shift

unset VLLM_ASCEND_BALANCE_SCHEDULING
# Keep operator verification on the non-context-parallel DSA path even when
# the parent shell was used for a FlashComm-enabled deployment.
unset VLLM_ASCEND_ENABLE_FLASHCOMM1
unset VLLM_ASCEND_ENABLE_FLASHCOMM

exec vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8077 \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm-5 \
  --max-num-seqs 8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --quantization ascend \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enforce-eager \
  --no-async-scheduling \
  --block-size 128 \
  --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true,"dsa_sparse_config":{"enabled":true}}' \
  "$@"
