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
unique misses, reordered pool rows, fused eviction, free-list refill, cursor
movement, and the final `free_head[:, 0] == 0` invariant. Active valid query
positions must be unique within each request; duplicate positions are outside
the operator contract.

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

TopK positions are reproducible randomized workloads. For every request, the
tool first samples 8K resident positions from the full 128K logical-position
space and maps them onto the 8K resident Hot Cache slots. It then samples the
hit portion from that resident set and the exact requested number of misses
from its complement, followed by shuffling all 2K query entries. Different
request rows receive different resident sets and TopK orders. Use `--seed` to
reproduce or change the workload. The generator deliberately does not add
duplicate, masked, or invalid positions, so `--miss-count` remains the exact
miss count for the selected workload.

Every sample restores its operator-specific state before the start event.
State restoration, tensor creation, synchronization, validation, and JSON
serialization are excluded. The NPU Event interval contains exactly one
selected operator invocation. The script validates the requested miss count,
the expected `free_head`, the occupied-slot count, and protected slots.

`--miss-rate 0` exercises a no-op maintain path because `free_head` is zero.
Use a nonzero miss rate to measure maintain eviction work. Maintain increments
the seed between invocations to match the framework's scan-start behavior. The
same initial `--seed` also controls randomized TopK generation.

## NPU profile

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:0 \
  --requests 8 \
  --miss-rate 10 \
  --seed 1234
```

Choose either `--miss-rate` or `--miss-count`; omitting both profiles the
all-hit path. In the default `steady` workload, a nonzero-miss run restores
all persistent state before every invocation so every sample performs the
requested fused lookup/update work. State restoration is outside the NPU Event
timing window but appears in the profiler trace; filter the parsed files by
`DsaSparseLookupUpdate` to inspect the custom kernel itself.

The profile workload uses the same unique, per-request randomized TopK
generator as the benchmark. Reusing a seed reproduces the same initial state;
for `step-random`, it also reproduces the generated per-step query schedule.

The fixed workload has 8K resident entries and a 2K query width. The script
writes a manifest and, unless `--no-trace` is used, a parsed
`torch_npu.profiler` trace under
`tools/dsa_sparse_lookup_update/profiles/<timestamp>/`. The script fails if
the parsed profile does not contain the custom operator name.

## Roofline profile

Use the standalone benchmark rather than `profile_operator.py` so that the
outer msopprof session does not conflict with an inner `torch_npu.profiler`
session:

```bash
bash tools/dsa_sparse_lookup_update/profile_roofline.sh \
  --device npu:2 \
  --requests 32 \
  --miss-rate 10
```

The script automatically uses the CANN 9.x `msopprof` executable and falls
back to the compatible `msprof op` entry point. It collects
`DsaSparseLookupUpdate` with `--aic-metrics=Roofline` and prints the generated
`visualize_data.bin` path for import into MindStudio Insight. Use
`--miss-count` instead of `--miss-rate` when an exact miss count is required.
The default result root is
`tools/dsa_sparse_lookup_update/roofline_profiles/`.

The fused operator mutates its index and slot state in place. The wrapper uses
`--replay-mode=application` so every profiler replay also reruns the benchmark
state restoration; kernel-level replay would turn the first replay's misses
into hits on later replays. Since msopprof warm-up is incompatible with
application replay, `--warm-up` runs a standalone benchmark before starting
the profiler instead. Roofline also replays the application for its bound
Default metric collection. Each replay writes its benchmark result to
`benchmark-{pid}-{timestamp_ns}.json`, so replay processes never target the
same JSON path.

Inspect the complete command without accessing an NPU:

```bash
bash tools/dsa_sparse_lookup_update/profile_roofline.sh \
  --device npu:2 \
  --requests 32 \
  --miss-count 205 \
  --dry-run
```

The single-operator profiler supports the same cache-behavior workloads as the
matrix profiler:

```bash
# A new TopK query on every step while one state mapping evolves.
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:2 \
  --requests 32 \
  --miss-rate 10 \
  --workload step-random \
  --profile-iters 20

# An independent state buffer for every profiled step.
python3 tools/dsa_sparse_lookup_update/profile_operator.py \
  --device npu:2 \
  --requests 32 \
  --miss-rate 10 \
  --workload cache-thrash \
  --profile-iters 20
```

`step-random` uses the requested miss count for the initial step and then
allows the fused operator state to evolve. `cache-thrash` keeps all per-step
state buffers alive, so its aggregate GM allocation grows with the number of
profile steps. In both dynamic modes, state restoration is not included in the
trace because no restoration is performed.

## Multi-profile optimization matrix

Run the optimization workload matrix and collect independent hardware metric
profiles with one command:

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator_matrix.py \
  --device npu:0 \
  --install-root tools/dsa_sparse_lookup_update/.install
```

The default matrix uses 32 requests and `0`, `1`, `205`, and `2048` misses per
request. For every workload it records one NPU Event latency distribution and
then creates four separately parsed profiler traces:

- `pipe-utilization`;
- `memory`;
- `resource-conflict`;
- `l2-cache`.

The metric groups are collected in separate profiler sessions because they
use different hardware counters. In the default `steady` workload, state
restoration is outside each Event interval and appears in every nonzero-miss
trace. The dynamic workloads prepare their input schedule before profiling
and do not restore state inside the profiler trace.

Specify a smaller or larger matrix as needed:

```bash
python3 tools/dsa_sparse_lookup_update/profile_operator_matrix.py \
  --device npu:2 \
  --requests 1 8 32 \
  --miss-rates 0 10 100 \
  --metrics pipe-utilization memory resource-conflict \
  --profile-iters 10 \
  --seed 1234
```

The matrix supports three cache-behavior workloads:

```bash
# Reuse one fixed random state and TopK query (default).
python3 tools/dsa_sparse_lookup_update/profile_operator_matrix.py \
  --workload steady \
  --metrics l2-cache \
  --miss-counts 205

# Use a new TopK query on every step while evolving one state mapping.
python3 tools/dsa_sparse_lookup_update/profile_operator_matrix.py \
  --workload step-random \
  --metrics l2-cache \
  --miss-counts 205 \
  --profile-iters 20 \
  --skip-event

# Allocate independent state buffers and use each one once.
python3 tools/dsa_sparse_lookup_update/profile_operator_matrix.py \
  --workload cache-thrash \
  --metrics l2-cache \
  --miss-counts 205 \
  --profile-iters 20 \
  --skip-event
```

`steady` resets the same state before every nonzero-miss invocation and is a
warmed steady-state measurement. `step-random` keeps the initial state
mapping and changes the TopK tensor on every invocation; after the first
invocation the fused operator state is allowed to evolve, so the requested
miss count describes the initial step. `cache-thrash` holds one independent
`index`/slot-state buffer per profile step, which grows the aggregate GM
working set instead of reusing the same addresses. Dynamic workloads do not
restore state inside the profiler trace.

Use either `--miss-counts` or `--miss-rates`. Repeated request counts, miss
counts, and metric names are deduplicated. The output layout is:

```text
profiles/matrix-<timestamp>/
  manifest.json
  req-0032_miss-0000/
    manifest.json
    pipe-utilization/
    memory/
    resource-conflict/
    l2-cache/
```

Each workload manifest records its exact miss count, effective miss rate,
Event timing, trace directories, parsed CSV files, and whether state restore
operations are present in the trace.
