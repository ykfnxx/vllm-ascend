# GLM-5.2 PD Host-KV end-to-end report

Date: 2026-08-30

## Configuration

- Model: `/data/model/GLM-5.2-w4a8`
- Container: `va23-asu-wp`
- Prefill: physical rank 0-7, TP=8
- Decode: physical rank 8-15, TP=8
- KV transport: `LocalShmConnector` plus true Host-backed Main KV/RoPE
- Decode mode: `FULL_DECODE_ONLY`
- Speculation: MTP=3
- Request: 4096 prompt tokens, 4 completion tokens, max model length 8192
- Runtime log directory in the container:
  `/tmp/glm52-fused-pd-e2e-20260830-blocktable-pass`

## Result

- HTTP status: 200
- Usage: 4096 prompt tokens, 4 completion tokens, 4100 total tokens
- TP event consistency: all 8 Decode TP ranks emitted identical probe sequences
- Target steps: 3
- IndexShare cohorts: 21
- Physical sparse-attention layers: 78
- `DsaLookup` plans: 63 (21 cohorts x 3 steps)
- `FusedKvGatherSparseFlashAttention`: 234 (78 layers x 3 steps)
- accepted-tail commits: 234
- independent pre-SFA `history_load_mock`: 0
- independent `hot_cache_sfa_done`: 0

The strict offline validator passed:

```text
PASS: DSA Sparse custom operator and Hot Cache path verified:
{"accepted_tail_commit":234,"cohort_count":21,"completion_tokens":4,
 "fused_kv_gather_sfa":234,"history_load_mock":0,
 "hot_cache_sfa_done":0,"layer_count":78,"lookup_update_done":63,
 "target_steps":3,"tp_rank_count":8}
```

## Representative Decode rank profile

| Operator | Count | Min (us) | Avg (us) | Max (us) |
|---|---:|---:|---:|---:|
| FusedKvGatherSparseFlashAttention | 234 | 107.122 | 116.759 | 126.703 |
| DsaLookup | 63 | 11.460 | 15.537 | 28.341 |
| DsaUpdate | 63 | 112.442 | 335.125 | 940.339 |
| DsaKvGather | 78 | 75.242 | 77.231 | 87.822 |
| AsuKvGather | 234 | 9.220 | 10.782 | 20.900 |

These profile values are from physical device 8 and include profiler overhead.
The request uses one active sequence; it is a functional/path end-to-end test,
not the BS=12 throughput benchmark.

## End-to-end fixes found by the test

1. MLA split-query views (`ql_nope` and `q_pe`) were made contiguous before
   entering the fused kernel. Only small per-step query and route tensors are
   materialized; Hot/Host KV caches are not copied.
2. The persistent Hot block table is pool-indexed, while `AsuKvGather` consumes
   destination rows in query order. A compact install block table is generated
   once per shared IndexShare plan and reused by all physical layers in the
   cohort. This preserves payload-install correctness after `DsaUpdate`.
3. The probe validator now validates TP ranks independently through strict
   identical event sequences and recognizes the fused KV-Gather+SFA profile.

The synthetic repeated-token prompt intentionally tests transport and operator
execution. Its decoded replacement characters are not a model-quality result;
precision remains covered by the nonzero Host/Hot KV operator comparison suite.
