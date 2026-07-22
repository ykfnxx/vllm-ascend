# DMP A3 single-card migration revision 9

Extract this archive directly into `/root/dmp`. The first validation uses the
smallest GLM-5.1 W4A8 prefix containing a MoE layer on one A3 card, matching the
earlier reduced-model workflow. The complete model at
`/mnt/models/GLM-5.1-w4a8` is mounted read-only and is never changed.

## Included code

- `vllm-ascend-0.18.0-copy`: vLLM Ascend source with DMP Lookup/Maintain r9.
- `scripts/pip-cache-dual-attention`: scheme-4 SFA/merge operator source.
- `scripts/dmp-lookup-maintain`: Lookup/Maintain/KVGather operator source.
- `scripts/dmp-fused-indexer-kv-select`: `ops_li_update` fused
  Indexer+Select request-pool operator source.
- Scripts for the A3 container, custom-op build, offline model reduction,
  profiling, and 30-line log splitting.

No A2 binary is included. The start script extracts the A3 image's own bundled
vLLM Ascend OPP, `vllm_ascend_C`, and `vllm_ascend_kernels` before bind-mounting
the Python source. The extracted native runtime is keyed by Docker image ID and
is refreshed automatically if the image changes.

## Host commands

```bash
cd /root/dmp
bash verify_bundle.sh
bash scripts/preflight_a3.sh
bash scripts/start_contianer1.sh
```

Defaults:

```text
image:          quay.io/ascend/vllm-ascend:v0.18.0-a3-openeuler
full model:     /mnt/models/GLM-5.1-w4a8
reduced models: /root/dmp/reduced-models
container:      vllm-ascend-a3-dmp
device:         0
```

Place these offline model-runtime wheels directly under `/root/dmp` before
starting the container:

```text
/root/dmp/transformers-5.2.0*.whl
/root/dmp/huggingface_hub-1.22.0*.whl
```

The start script mounts `/root/dmp` read-only at `/dmp-host`. Build/run scripts
locate both wheels automatically and install them into the persistent host
directory `/root/dmp/scripts/dmp-model-runtime/python`. No manual `pip install`
is required, and later containers reuse the installed runtime.

## One-command container run

Run one command after entering the container:

```bash
VISIBLE_DEVICES=0 bash /workspace/scripts/run_a3_single_card_baseline.sh
```

The default remains scheme 4. Run the updated scheme 3 explicitly with:

```bash
DMP_SCHEME=3 VISIBLE_DEVICES=0 \
  bash /workspace/scripts/run_a3_single_card_baseline.sh
```

Scheme 3 runs both fused Indexer+Select calls on S0 and one local 2K KVIO per
microbatch on S1, followed by selected-cache SFA on S0. The fused operator
already updates token-to-slot state. The upstream branch does not provide a
compatible reverse-state AICPU Maintain operator, so scheme 4's Maintain is
not reused in scheme 3.

This entry point checks all host mounts, required build commands, A3 base
operators, offline wheels or persistent Transformers runtime, model paths, and
NPU visibility. It automatically builds and smoke-tests each missing custom-op
runtime, validates all native symbols and PyTorch registrations together,
creates the reduced checkpoint, and only then starts inference. Successful
builds are stamped and skipped on later runs. The reduced-model generator keeps
the dense prefix through the first MoE layer and excludes MTP-only weights such
as `rot.weight`; it does not reuse the obsolete fixed two-layer checkpoint.

Revision 7 validates that the image-native extension registers
`torch.ops._C_ascend.moe_gating_top_k` and the GLM attention operators before
loading model weights. This prevents the former late failure where the source
bind mount hid `vllm_ascend_C*.so` and MoE failed during dummy forward.

Revision 8 keeps single-card vLLM rendezvous on `127.0.0.1` and Gloo on `lo`.
It also reuses a previously smoke-tested operator runtime without launching
extra `torch_npu` validation processes immediately before inference. Set
`DMP_A3_VALIDATE_RUNTIME=1` to request the full validation again.

Revision 9 adds the `ops_li_update` request-pool operator from
`xwLearnsLLM/DSA_offload_ops` commit
`3f5c292a4f069716b683b147da71b3106e2ae2bc`, plus isolated scheme selection so
schemes 2 and 4 retain their existing operator paths.

The first invocation creates a path such as
`/models-reduced/GLM-5.1-w4a8-4layers-dmp-r2` (the exact layer count comes
from `first_k_dense_replace` and `moe_layer_freq`). It reads one source shard at a time,
keeps the smallest prefix containing a MoE layer plus embedding/norm/head tensors, writes a new
safetensors index, and then starts the run. The result persists on the host at
`/root/dmp/reduced-models`, so later containers do not regenerate it.

Default profiling parameters:

```text
TP=1, EP=off, batch=64, prompt=131072, output=10, reduced through first MoE
```

To choose another free A3 card:

```bash
VISIBLE_DEVICES=6 bash /workspace/scripts/run_a3_single_card_baseline.sh
```

The default command now runs the previously agreed profiling workload. To
override it explicitly:

```bash
VISIBLE_DEVICES=6 BATCH_SIZE=64 PROMPT_TOKENS=131072 MAX_TOKENS=10 \
bash /workspace/scripts/run_a3_single_card_baseline.sh
```

If 64 requests do not fit, change only `BATCH_SIZE=48` and retry.

The run directory contains `DISTRIBUTED_INIT_TIMING.txt`. A successful local
rendezvous should show `distributed_init_method=tcp://127.0.0.1:<port>` and a
much smaller `distributed_init_seconds` than the former 841-second startup.

To create a different reduced model, change the layer count. Each count gets a
separate directory:

```bash
REDUCED_LAYERS=4 VISIBLE_DEVICES=6 \
bash /workspace/scripts/run_a3_single_card_baseline.sh
```

Do not raise the layer count, prompt length, and batch size together. First
confirm the reduced-model profile, then change one memory dimension at a time.

## Expected scheme-4 timeline

```text
S0: LI0 -> Lookup0 -> LI1 -> Lookup1 -> preattn -> hit SFA -> wait -> miss SFA -> merge -> update -> MLP
S1:          miss KVGather0                    -> miss KVGather1
S2:          Maintain0                         -> Maintain1
```

Profiling data and 30-line log chunks are written below
`/workspace/scripts/logs`.
