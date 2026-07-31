# DSA sparse lookup/update standalone tools

This directory builds and benchmarks three metadata operators without starting
vLLM:

- `dsa_sparse_lookup_update`: Ascend 950 SIMT fused lookup/update;
- `asu_hbm_index_lookup`: standalone ASU lookup;
- `asu_hbm_index_maintain_aicpu`: standalone AICPU maintenance.

Correctness and profiler tooling remains available for the fused SIMT operator.

The operator uses the ASU lookup-shaped interface:

```text
(index, slot_to_index, free_slots, free_head,
 req_pool_entries, query_index, lookup_mask, req_num)
    -> (slot_out, miss_out)
```

Payload I/O, Sparse Flash Attention, scheduler admission, and the complete P/D
lifecycle are outside these standalone tools.

## Build and install

Run from the repository root:

For all three Ascend 950 operators:

```bash
bash tools/dsa_sparse_lookup_update/build_and_install.sh \
  --operator all \
  --soc ascend950
```

For the two legacy operators on Ascend 910C/A3:

```bash
bash tools/dsa_sparse_lookup_update/build_and_install.sh \
  --operator legacy \
  --soc ascend910_93
```

`--operator` accepts `simt`, `lookup`, `maintain`, `legacy`, or `all`.
`legacy` packages lookup and maintain; `all` additionally packages the SIMT
operator and is available only on Ascend 950. The isolated install root is
`tools/dsa_sparse_lookup_update/.install` by default. Rebuild the
vLLM-Ascend extension as well as the selected custom operator package before
running the scripts.

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

## Independent single-operator benchmark

```bash
# SIMT fused lookup/update only
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --operator simt \
  --concurrency 8 \
  --miss-rate 10

# ASU lookup only
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --operator lookup \
  --concurrency 8 \
  --miss-rate 10

# AICPU maintain only
python3 tools/dsa_sparse_lookup_update/benchmark_operator.py \
  --device npu:0 \
  --operator maintain \
  --concurrency 8 \
  --miss-rate 10
```

The shape is fixed by the operator contract: 8K resident entries, 2K free
entries, and 2K queries per request. `--concurrency N` controls the number of
independent request rows handled by one operator invocation. `--operator
legacy` runs lookup and maintain as separate benchmark phases. `--operator
all` runs all three operators as separate phases. No timed interval contains
more than one operator.

Choose either:

```bash
--miss-rate 10
--miss-count 205
```

or use `--scenario hit`, `--scenario churn`, or the default `both`.

For lookup and SIMT, the script starts with 8K resident tokens and puts the
requested number of absent tokens in the 2K query. For maintain, it directly
constructs the equivalent post-lookup state: `free_head` equals the requested
miss count, those misses occupy free slots, and `last_query_slots` protects the
current query. Maintain setup does not invoke lookup.

Every sample restores its operator-specific state before the start event.
State restoration, tensor creation, synchronization, validation, and JSON
serialization are excluded. The NPU Event interval contains exactly one
selected operator invocation. The script validates the requested miss count,
the expected `free_head`, the occupied-slot count, and protected slots.

`--miss-rate 0` exercises a no-op maintain path because `free_head` is zero.
Use a nonzero miss rate to measure maintain eviction work. Maintain increments
the seed between invocations to match the framework's scan-start behavior.

## NPU profile

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:0 \
  --requests 8 \
  --miss-rate 10
```

Choose either `--miss-rate` or `--miss-count`; omitting both profiles the
all-hit path. For a nonzero miss workload, the script restores all persistent
state before every invocation so every sample performs the requested fused
lookup/update work. State restoration is outside the NPU Event timing window
but appears in the profiler trace; filter the parsed files by
`DsaSparseLookupUpdate` to inspect the custom kernel itself.

The fixed workload has 8K resident entries and a 2K query width. The script
writes a manifest and, unless `--no-trace` is used, a parsed
`torch_npu.profiler` trace under
`tools/dsa_sparse_lookup_update/profiles/<timestamp>/`. The script fails if
the parsed profile does not contain the custom operator name.
