# DSA sparse lookup/update standalone tools

This directory builds, checks, benchmarks, and profiles the fused
`dsa_sparse_lookup_update` metadata operator without starting vLLM.

The operator uses the ASU lookup-shaped interface:

```text
(index, slot_to_index, free_slots, free_head,
 req_pool_entries, query_index, lookup_mask, req_num)
    -> (slot_out, miss_out)
```

Lookup and metadata maintenance execute in one SIMT kernel. Payload I/O,
Sparse Flash Attention, scheduler admission, and the complete P/D lifecycle
are outside these standalone tools.

## Build and install

Run from the repository root:

```bash
bash tools/dsa_sparse_lookup_update/build_and_install.sh
```

The script builds only this operator for `ascend950` and installs it under
`tools/dsa_sparse_lookup_update/.install`. Because both the torch schema and
the custom operator ABI changed, rebuild the vLLM-Ascend extension as well as
the single-op package before running the scripts.

## Correctness

```bash
python3 tools/dsa_sparse_lookup_update/test_correctness.py \
  --device npu:0 \
  --requests 2 \
  --random-cases 10
```

The script checks `slot_out`, `miss_out`, and all four persistent state
tensors against the CPU oracle. It covers hits, masked/invalid entries,
duplicate misses, reordered pool rows, fused eviction, free-list refill,
cursor movement, and the final `free_head[:, 0] == 0` invariant.

## Single-operator benchmark

```bash
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --concurrency 8 \
  --miss-rate 10
```

The shape is fixed by the operator contract: 8K resident entries, 2K free
entries, and 2K queries per request. `--concurrency N` controls the number of
independent request rows handled by one operator invocation.

Choose either:

```bash
--miss-rate 10
--miss-count 205
```

or use `--scenario hit`, `--scenario churn`, or the default `both`. The
event-timed interval contains one batched custom-op invocation. Tensor
creation, initial state construction, query-group updates, synchronization,
and JSON serialization are excluded. The script checks both the first and
last measured invocation and fails if the requested miss count has degraded
into hits as persistent metadata changes.

## NPU profile

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:0 \
  --requests 8
```

The profile workload is a steady all-hit lookup over the fixed 2K query
width. It writes a manifest and, unless `--no-trace` is used, a parsed
`torch_npu.profiler` trace under
`tools/dsa_sparse_lookup_update/profiles/<timestamp>/`. The script fails if
the parsed profile does not contain the custom operator name.
