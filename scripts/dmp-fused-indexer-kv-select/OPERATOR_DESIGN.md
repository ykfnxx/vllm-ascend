# LightningIndexerDecodeUpdate 算子原理

## 功能

`LightningIndexerDecodeUpdate` 面向 decode 阶段，一次完成 DSA score、top2048、命中判断、evict slot 选择和 cache 索引更新。

## 接口

```python
lightning_indexer_decode_update(
    query,                   # bf16/fp16[B, 64, 128]
    key,                     # bf16/fp16[NUM_BLOCKS, BLOCK, 1, 128]
    weights,                 # bf16/fp16[B, 64]
    cache_slots,             # int32[B, 262144], in/out; -1 表示未缓存
    actual_seq_lengths_key,  # int32[B]
    block_table,             # int32[B, MAX_BLOCKS]
) -> (
    topk_index,              # int32[B, 1, 2048]
    topk_slots,              # int32[B, 1, 2048]
    miss_count,              # int32[B]
)
```

算子原地修改 `cache_slots`。返回后满足：

```text
K = min(actual_seq_lengths_key[b], 2048)
topk_slots[b, 0, k] == cache_slots[b, topk_index[b, 0, k]], 0 <= k < K
```

### Request Pool 接口

```python
lightning_indexer_decode_update_pool(
    query,                   # bf16/fp16[B, 64, 128]
    key,                     # bf16/fp16[NUM_BLOCKS, BLOCK, 1, 128]
    weights,                 # bf16/fp16[B, 64]
    req_pool_entries,        # int32[B], batch -> request pool row
    cache_slots,             # int32[POOL_SIZE, 262144], in/out; -1 表示未缓存
    actual_seq_lengths_key,  # int32[B]
    block_table,             # int32[B, MAX_BLOCKS]
) -> (
    topk_index,              # int32[B, 1, 2048]
    topk_slots,              # int32[B, 1, 2048]
    miss_count,              # int32[B]
)
```

对第 `b` 个请求，算子读写 `cache_slots[req_pool_entries[b]]`。映射值必须位于 `[0, POOL_SIZE)` 且互不重复；未映射的 pool row 保持不变，其余行为与非 pool 接口一致。

## 固定约束

- Decode-only，`query` 固定为 TND，`key` 固定为 PA_BSND。
- `q_heads=64`，`k_heads=1`，`head_dim=128`，`sparse_count=2048`。
- `query/key/weights` dtype 相同，仅支持 `bf16` 和 `fp16`。
- 有效 slot 范围为 `0..16382`；`-1` 表示 miss，`0x3fff` 保留为无效编码。
- `cache_slots` 每行物理容量为 `262144`，逻辑序列长度满足 `0 <= L < 262143`。
- 支持 `ascend910b` 和 `ascend910_93`。

## 序列长度与 Cache 状态

对每个 batch row 独立定义：

```text
L = actual_seq_lengths_key[b]
C = 该请求的物理 cache slot 容量，2048 < C < 16383
K = min(L, 2048)
```

`C` 不作为算子参数传入，而是调用方维护的状态约束。调用算子前必须满足：

- `cache_slots[b, 0:L]` 中有 `min(L, C)` 个有效值，有效 slot 唯一且位于 `[0, C)`。
- `L < C` 时，序列中的全部 `L` 个 token 都已缓存。
- `L >= C` 时，刚好缓存 `C` 个 token，有效 slot 集合为 `0..C-1`。
- `cache_slots[b, L:262144]` 不属于当前序列，算子忽略并保持不变。

算子不负责给 `L<C` 时的未缓存 token 分配空闲 slot；当前接口没有提供空闲 slot 集合，因此这种输入不属于合法状态。

固定长度输出按以下顺序组织：

```text
[old cache 中的 miss token] [old cache 中的 hit token] [padding]
```

前 `K` 项为有效 token，`[K, 2048)` 的 `topk_index` 和 `topk_slots` 均为 `-1`。`miss_count` 只统计前 `K` 个有效 token 在更新前的 miss 数量，padding 不参与排序、统计或 cache 更新。

| 长度范围 | 选择与更新行为 |
| --- | --- |
| `L=0` | 没有有效 token；两个 topk 输出全部为 `-1`，`miss_count=0`，`cache_slots` 不变。 |
| `0<L<=2048` | 全部 `L` 个 token 都被选中；它们均已缓存，因此 `miss_count=0`，`cache_slots` 不变，输出尾部使用 `-1` padding。 |
| `2048<L<C` | 从 `L` 个已缓存 token 中选择 top2048；全部为 hit，`miss_count=0`，`cache_slots` 不变。 |
| `C<=L<262143` | 选择 top2048，按 old cache 的 miss/hit 状态重排，并为每个 miss 淘汰一个不属于 top2048 的 cached token。 |

最后一种情况中，设 `M=miss_count`，则 cached 非 topk token 数量为：

```text
C - (2048 - M) = C - 2048 + M >= M
```

由于 `C>2048`，合法输入必然存在足够的 evict token。更新后仍有 `C` 个有效 slot，且 slot 集合保持为 `0..C-1`。`L=C` 时全部 token 已缓存，因此正常情况下 `miss_count=0`。

## 计算流程

每个请求按 512-token chunk 处理：

1. AIC 计算 64 个 query head 与 paged key 的乘积。
2. AIV 使用 `weights` 对 64 个 head 的结果加权归约，得到每个 token 的 float32 score。
3. 每 4 个 chunk 做一次归并，持续维护固定长度为 2048 的 topk buffer；当 `L<2048` 时，尾部为无效 padding。
4. payload 同时携带 token index 和旧 cache slot：

```text
payload = (slot14 << 18) | token_index
slot == -1  =>  slot14 = 0x3fff
```

5. `L>=2048` 时记录最低分 `thresholdScore`，再按 slot 状态排序，使 miss 在前、hit 在后。
6. 只在前 `K` 个有效位置统计 miss 前缀并写入 `miss_count`；短序列 padding 始终保留在输出尾部。

## Score Workspace

每个 chunk 的 score 都写入 HBM workspace，供 top2048 完成后的 evict 扫描使用。score 不作为输出暴露。

```text
score_stride = AlignUp(block_table.shape[1] * BLOCK, 512)
score_workspace = B * score_stride * sizeof(float)
```

最后一个不足 512 token 的 chunk 使用 `-inf` padding；有效范围由 `actual_seq_lengths_key[b]` 控制。

score 写回由 MTE3 执行，并与当前 chunk 的 payload 处理、局部排序和 topk 归并重叠；在 score UB 被复用前完成同步。

## Evict Candidate

得到 `thresholdScore` 和 `miss_count` 后，使用 `actual_seq_lengths_key[b]` 和 batch row 编号计算确定性 hash，并从 hash 对应的 chunk 开始读取 workspace score 与 `cache_slots`：

```text
slot >= 0  => candidate_key = -score
slot == -1 => candidate_key = -inf
```

每个 chunk 通过 `Sort32 + MrgSort` 排序。candidate buffer 的物理上限为 2048，运行时容量为 `AlignUp(miss_count, 512)`，即 512、1024、1536 或 2048。

扫描按 chunk 编号循环前进，到序列末尾后回到 chunk 0。一次调用最多扫描一整轮，每个 chunk 最多访问一次。

每处理一个 chunk，算子读取对应的 workspace score 和 `cache_slots`，完成排序与归并后检查第 `miss_count-1` 个候选。若该候选满足：

```text
candidate_key > -thresholdScore
```

则已经找到足够多、且 score 严格低于 top2048 阈值的 cached token。算子随后继续扫描若干个尚未访问的 chunk，从更大的局部范围中保留低分候选，再停止扫描。

额外扫描数量由编译时环境变量 `EVICT_EXTRA_SCAN_CHUNKS` 控制，取值范围为 `0..512`，默认值为 `8`。若本轮剩余 chunk 不足指定数量，则只扫描剩余 chunk，不会重复访问已经扫描过的 chunk。

若完整扫描后候选仍不足，算子线性扫描 `cache_slots`，选择不属于当前 top2048 的 cached token 作为 correctness fallback。

## Cache 更新

对每个 miss token：

```text
cache_slots[b, evict_index] = -1
cache_slots[b, miss_index] = evict_slot
topk_slots[b, 0, miss_pos] = evict_slot
```

hit 后缀直接保留旧 slot。最终 `cache_slots` 中有效 slot 的数量和值集合保持不变。

## 调度与 Workspace

调度以请求为单位；一个请求由 1 个 AIC 和 2 个 AIV 调度槽协作处理，请求不会跨 AIC 拆分。除 AscendC 基础 workspace 外，还包含：

```text
2 * 64 * 512 * sizeof(float) * blockDim
+ B * score_stride * sizeof(float)
```
