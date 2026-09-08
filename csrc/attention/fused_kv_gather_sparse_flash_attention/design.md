# FusedKVGatherSparseFlashAttention design

## 1. Operator interface

`fused_kv_gather_sparse_flash_attention` is a route-aware MLA sparse-attention
operator. It accepts query/query-RoPE, Hot KV/RoPE and block table, Host
KV/RoPE and block table, row mappings, `route_plan [R,2048]`, sequence lengths
and the normal SFA scalar attributes. It returns the attention output and the
optional softmax max/sum tensors using the ordinary SFA ABI.

KV/RoPE support BF16 and FP16. Route and block-table tensors are int32. The
first production key supports TND query, PA_BSND sources, KV head count one,
KV/RoPE record widths 512/64, block size 128, sparse block size one and
`sparse_count=2048`.

## 2. Computation

The ordinary SFA AIC matmul, vector softmax and output stages are retained.
Only the existing Vec0 `MergeKv` source-address resolution changes:

```text
for each route rank consumed by SFA Vec0:
    decode kind and source position
    select Hot or Host source row/table
    resolve logical block -> physical block
    DataCopyPad KV and RoPE directly into existing merge UB
    copy the merge tile to SFA's existing kvMerge GM workspace
run the unchanged MM1 -> softmax -> MM2 pipeline
```

POOL_HIT and LIVE_HOT read Hot cache; HOST_MISS reads true Host swapped-memory
cache. INVALID entries terminate/zero-fill the merge stream. `value == key`
for the current MLA latent-cache path.

## 3. Tiling and UB

Block and UB tiling reuse the SFA production tiling. Vec0 already has two
ping-pong merge buffers for 32 records. The fused source resolver adds only a
small route/table tile; KV and RoPE continue using the existing buffers.

| Buffer | Existing size | Count | Purpose |
| --- | ---: | ---: | --- |
| KV merge UB | 32 x 512 x dtype | 2 | Source KV DMA |
| RoPE merge UB | 32 x 64 x dtype | 2 | Source RoPE DMA |
| route tile | 512 x int32 (2 KiB) | 1 | Dedicated Vec0 source kind/index tile |
| valid-size metadata | existing | existing | Cube synchronization |

The route tile must not alias `tmpBuff1`: Vec1 and Vec2 reuse that buffer in
the preloaded SFA pipeline. Route metadata is copied once per S2 tile and the
Hot/Host source-row scalars are loaded once per tile instead of once per
selected token.

Hot and Host maintain independent last-logical-block register caches.  This
groups consecutive tokens from the same physical block without adding an
explicit MTE2-to-scalar synchronization; adjacent table entries continue to
use the A3 hardware DCache's 32-byte cache line.  The production block size is
fixed at 128, allowing logical-block and in-block address generation to use
shifts and masks instead of division and modulo.  Table row bases are computed
once per route tile instead of once per selected token.

Host tiling validates all three block-table domains, physical block counts,
record widths, dtypes and GM address limits independently.

## 4. Workspace

The SFA MM/merge workspace is retained. The MTP selection cache and selection
block table are removed. No workspace may alias persistent Hot or Host cache.

## 5. Performance plan

At BS=12/MTP=3, materializing 48 x 2048 x 576 BF16 records writes roughly
108 MiB to a selection cache which SFA then reads again. Fusion removes about
216 MiB of intermediate HBM traffic and one kernel boundary. Host-source DMA
is unavoidable but can overlap the existing AIC pipeline.

## 6. Correctness and safety

- Independently validate Hot and Host source rows/tables/physical block ids.
- Reject mismatched KV/RoPE physical block counts at adapter and tiling layers.
- Do not use a KV block bound to validate a different RoPE allocation.
- Preserve SFA causal masking, numerical accumulation and output dtype.
- Existing DsaKVGather + ordinary SFA remains the fallback and golden path.

## 7. Implementation checklist

- Add route-aware A3 Vec0 MergeKv specialization.
- Add an independent A5 source loader; do not assume A3 MTE behavior.
- Register a new ACLNN/PyTorch operator and meta implementation.
- Test mixed Hot/Host routes with deterministic nonzero KV/RoPE.
- Compare eager and Graph output to DsaKVGather + ordinary SFA and full HBM.
