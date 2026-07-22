# Scheme 3: Fused Indexer + Select Pool (A3)

Operator source:

- Repository: `xwLearnsLLM/DSA_offload_ops`
- Branch: `ops_li_update`
- Commit: `3f5c292a4f069716b683b147da71b3106e2ae2bc`
- Operator: `npu_lightning_indexer_decode_update_pool`

The operator fuses Lightning Indexer, Select, and the token-to-slot update. A
and B use disjoint rows in one persistent request pool per layer. It returns
`topk_index`, `topk_slots`, and `miss_count`.

Current A3 pipeline:

```text
S0: mb0 fused -> mb1 fused -> pre-attn -> wait KVIO -> SFA0/SFA1 -> MLP
S1:                         KVIO0 -> KVIO1
```

The local KVIO is one launch per microbatch and copies the selected 2K tokens
into their 10K-pool slots. This is the current local-HBM simulation path.

The linked branch does not contain a separate AICPU Maintain operator. Its
token-to-slot update is already part of the fused S0 operator. Scheme 4's
Maintain cannot be reused because it updates a different index/free-list state
and randomly evicts 300 entries. A future S2 requires the matching reverse
`slot-to-token` Maintain implementation from the operator owner.

Run on A3:

```bash
DMP_SCHEME=3 VISIBLE_DEVICES=0 \
  bash /workspace/scripts/run_a3_single_card_baseline.sh
```

The first run builds and smoke-tests the new fused-pool operator. Schemes 2 and
4 keep their existing operators and behavior.
