# DSA sparse lookup/update standalone tools

This directory builds, validates, and profiles
`dsa_sparse_lookup_update` without starting a vLLM engine.

The tools exercise only the metadata operator. They do not validate Main-KV
payload I/O, SFA, scheduler admission, or the complete P/D lifecycle.

## Prerequisites

- Ascend 950/A5 device
- CANN toolchain capable of building AscendC SIMT kernels
- The PyTorch and `torch_npu` versions required by this checkout
- A built `vllm_ascend.vllm_ascend_C` extension from this checkout

Run all commands from the repository root.

## Build and install only this operator

```bash
bash tools/dsa_sparse_lookup_update/build_and_install.sh
```

The script invokes:

```bash
cd csrc
bash build.sh --pkg --ops=dsa_sparse_lookup_update --soc=ascend950
```

It installs the resulting package under
`tools/dsa_sparse_lookup_update/.install` so it does not replace the
checkout-wide `vllm_ascend/_cann_ops_custom` directory.

Use `--build-only` to leave the generated `.run` package uninstalled, or
`--install-root PATH` to choose another install root.

## Correctness

```bash
python3 tools/dsa_sparse_lookup_update/test_correctness.py \
  --device npu:0 \
  --random-cases 100
```

The test calls `torch.ops._C_ascend.dsa_sparse_lookup_update` directly and
compares all persistent state and result tensors with the repository's CPU
oracle:

- `token_to_hot`
- `hot_to_token`
- `lru_slots`
- `state_seat_epoch`
- `resolved_hot_indices`
- `miss_mask`

The deterministic cases cover hits, duplicate misses, reserved newest slots,
inactive rows, reordered query metadata, and epoch reset. Random cases add
different residency, LRU, validity, Top-K, and row-to-seat combinations.

To use a package installed somewhere else:

```bash
python3 tools/dsa_sparse_lookup_update/test_correctness.py \
  --install-root /path/to/custom/op/install
```

## Latency and NPU profile

Profile a steady-state lookup:

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --scenario steady \
  --device npu:0
```

Profile the operator-owned cache-seat reset path:

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --scenario fresh \
  --device npu:0
```

Important scenario semantics:

- `steady`: warmup populates the metadata cache, then the same valid Top-K
  selection is measured as resident lookup/LRU maintenance.
- `fresh`: every invocation receives a different seat epoch. The measured
  operator therefore includes its internal `token_to_hot`, `hot_to_token`, and
  LRU reset plus cold allocation.

The default workload uses `T=1`, `K=2048`, and `S=4096`. Override dimensions
with `--seats`, `--rows`, `--max-model-len`, `--slots`, `--lanes`, and
`--topk`.

Each run writes a JSON manifest and, unless `--no-trace` is passed, a parsed
`torch_npu.profiler` trace under:

```text
tools/dsa_sparse_lookup_update/profiles/<timestamp>/
```

The event timing and profiler passes are separate. CPU copies, tensor
allocation, and profiler parsing are outside the event-timed region. Confirm
that the trace contains one of:

- `DsaSparseLookupUpdate`
- `dsa_sparse_lookup_update`
- `aclnnDsaSparseLookupUpdate`
