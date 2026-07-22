# DMP A2 validated snapshot

This directory preserves the DMP code and scripts validated on the A2 server.
It uses `quay.io/ascend/vllm-ascend:v0.18.0-openeuler` and CANN 8.5.1.

The default `example1.py` configuration is scheme 4, Lookup/Maintain r9:

```text
S0: LI0 -> Lookup0 -> LI1 -> Lookup1 -> combined hit SFA -> wait -> miss SFA -> merge -> MLP
S1:                  KVGather0                         -> KVGather1
S2:                  Maintain0                        -> Maintain1
```

The runtime modes remain independently selectable through these switches:

```text
VLLM_ASCEND_ENABLE_DMP
VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION
VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT
VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN
```

The A2 run scripts expect the repository source at
`/root/dmp/vllm-ascend-0.18.0-copy` and the scripts at `/root/dmp/scripts`, as
used by the validated server environment. A3 container, reduced-model, and
A3-specific operator changes do not belong to this snapshot and must be kept
on a separate branch.
