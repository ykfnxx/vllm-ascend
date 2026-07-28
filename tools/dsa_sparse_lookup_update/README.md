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
- `resolved_hot_indices`
- `miss_mask`

The deterministic cases cover hits, duplicate misses, reserved newest slots,
inactive request indices, and reordered query metadata. Random cases add
different residency, LRU, validity, Top-K, and direct request-index mappings.

To use a package installed somewhere else:

```bash
python3 tools/dsa_sparse_lookup_update/test_correctness.py \
  --install-root /path/to/custom/op/install
```

## Latency and NPU profile

Profile a steady-state lookup:

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:0
```

Warmup populates the metadata cache, then the same valid Top-K selection is
measured as resident lookup/LRU maintenance. Request-index reuse and state
reset belong to the lifecycle control plane and are outside this single-op
profile.

The default workload uses `T=1`, `K=2048`, and `S=4096`. Override dimensions
with `--requests`, `--max-model-len`, `--slots`, `--lanes`, and
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

## 8K resident-cache benchmark

Benchmark one layer with 8192 resident slots per request and a configurable
number of concurrent requests:

```bash
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --concurrency 8
```

`--concurrency N` creates `N` active request indices in one batched operator
invocation. Each index addresses an independent, fully populated 8192-slot
cache directly. The default `Top-K` is 2048 and the default query-lane count is
one.

The benchmark supports:

- `hit`: every Top-K entry is already resident. This measures lookup, result
  production, and LRU hit maintenance against a full 8K cache.
- `churn`: five disjoint 2K token groups rotate through the 8K cache. Every
  measured invocation replaces 2K entries while the cache remains full.
- `both`: run `hit` and `churn` independently. This is the default.

Specify an arbitrary miss percentage:

```bash
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --concurrency 8 \
  --miss-rate 10
```

For the default `Top-K=2048`, 10% is rounded to 205 misses per request. With
eight concurrent requests this produces 1640 misses in each batched operator
invocation. The manifest records the requested percentage, integer miss count,
and effective percentage.

To control the integer count directly, use `--miss-count`:

```bash
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --concurrency 8 \
  --miss-count 200
```

`--miss-rate` and `--miss-count` are mutually exclusive and override
`--scenario`. Before timing, the script executes one validation invocation and
checks that the operator's `miss_mask` contains exactly the requested number
of misses.

For example, run only the resident-hit workload for 32 concurrent requests:

```bash
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --concurrency 32 \
  --scenario hit \
  --warmup 20 \
  --iterations 200
```

The event-timed region contains one batched custom-operator invocation per
sample. Tensor allocation, cache initialization, and host-side result
serialization are excluded. Results are printed and saved under
`tools/dsa_sparse_lookup_update/benchmarks/`.
