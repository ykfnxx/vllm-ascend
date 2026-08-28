# AsuKvGatherSim Design

## Purpose

`asu_kv_gather_sim` 是 `asu_kv_gather` 的 swap 延迟抖动仿真版本。数据格式、
输入输出接口、核分配、寻址与校验逻辑与原始 `AsuKvGather` 完全一致，仅有一处差异：

- **swap 读延迟抖动**：每个 AIV 核在整个 pair 循环结束后只忙等一次，忙等时长 =
  对本核实际发射 MTE2 的 miss 数（每 miss 发 KV+Rope 共 2 次 MTE2，N = miss 数 × 2）
  各抽一次 7 级尾延迟后取**最大 delay**。用于复现"多次 swap 读中至少一次命中
  尾延迟"的聚合概率。

## Interface

与原始算子相同：

```text
(destination_kv_cache, destination_k_rope) = asu_kv_gather(
    destination_kv_cache, destination_k_rope,
    destination_block_table,
    source_kv_cache, source_k_rope, source_block_table,
    req_pool_entries, token_positions, destination_slots, miss_mask,
    block_size, req_num)
```

- KV 支持 `float16` / `bfloat16` / `int8`；RoPE 支持 `float16` / `bfloat16`。
- 全部索引张量（block table、pool entries、positions、slots、miss mask）为
  `int32`，与原始 kvgather 数据格式一致（不支持 int16 slot 适配版）。
- 支持 dense `[requests, query_count]` 与 resident-init 紧凑两种元数据布局，
  布局判定与校验逻辑与原版相同。

## Algorithm

扁平 pair 空间大小 `pair_count = req_num * query_count`。核数
`core_count = min(pair_count, 48)`（与原始 asu_kv_gather 完全一致）。
每个核按连续区间处理 pair：

```text
core c 处理 [c*pair_per_core + shift, ...) 连续 pair 区间（步进 1）
```

整个 AIV 核在 pair 循环**结束后**只注入一次抖动（循环内每个 miss 不再逐次抖动）；
循环内累计本核实际发射 MTE2 的 miss 数，抖动时长 = 对 N = miss 数 × 2 抽样后的最大 delay：

```text
for each pair (miss):
    DataCopy sourceKv  -> kvBuffer      # MTE2 load (记 missCount)
    DataCopy sourceRope -> ropeBuffer   # MTE2 load
    Sync<MTE2_MTE3>
    DataCopy kvBuffer   -> destinationKv
    DataCopy ropeBuffer -> destinationRope
    Sync<MTE3_MTE2>

RandomSleepBeforeSwapLoad(missCount × 2)  # 抖动: 整个 AIV 仅一次, 取 N 次抽样最大值
```

每个核只有一组 KV/RoPE 记录 buffer；`req_id/query_id` 顺序步进，跨 req
时回绕，`pool_entry` 随 req 切换重新读取并校验。

## Jitter model

- 7 级尾延迟分布，每级 `(threshold, delayUs)`，阈值按 `threshold = 2^32 / denominator`
  计算，`P(rng < threshold) ≈ 1 / denominator`。采用 CDF 尾部模型采样：从最稀有
  （最小阈值、最长延迟）级别开始检查，保证 `P(delay ≥ 该级 delayUs) = 1 / denominator`：

  | denominator | delayUs | 含义 |
  |---:|---:|---|
  | 1 (基础) | 79 | 99% 基础延迟 |
  | 100 | 169 | 1% |
  | 1000 | 228 | 0.1% |
  | 10000 | 289 | 0.01% |
  | 100000 | 354 | 0.001% |
  | 1000000 | 502 | 0.0001% |
  | 10000000 | 13070 | 0.00001% |

- 聚合概率：`RandomSleepBeforeSwapLoad` 对 N = 本核实际 miss 数 × 2 次 MTE2 各抽
  一次延迟并取最大值，因此 `P(抖动 ≥ 该级 delayUs) = 1 - (1 - 1/denominator)^N`；
  核处理的 miss 越多，命中更高级尾延迟的概率越高（例：N=10 时 `P(≥169us)≈9.6%`，
  N≈43 时 `P(≥169us)≈35%`，N≈546 时 `P(≥228us)≈42%`）。
- 随机源为 LCG（Numerical Recipes 参数 `x = x*1664525 + 1013904223`），
  每核独立种子 `seed ^ (blockIdx * 2654435761)`。
- 忙等为标量循环，`volatile sink` 防止被优化消除，迭代数按
  `delay_us * kItersPerUs`，`kItersPerUs = 189` 已在 Ascend910B2 上经 msprof
  校准。
- 编译期开关 `kSwapJitterEnabled` 与运行时 `tiling.jitterEnable` 双重控制，
  任一关闭即零开销直接返回。

## Tiling

在原版 tiling 数据基础上追加两个字段：

```text
uint32_t jitterEnable;   // 1=注入抖动, 0=关闭
uint32_t jitterSeed;     // 随机种子
```

块 tiling：`min(req_num * query_count, 48)` 核。

## 与原版差异汇总

| 方面 | 原版 asu_kv_gather | sim 版 |
|---|---|---|
| pair -> 核分配 | 连续区间 | 连续区间（一致） |
| pair 循环后（每核一次） | 无 | RandomSleepBeforeSwapLoad(miss 数×2) 取 max |
| tiling 字段 | 原版字段 | + jitterEnable, jitterSeed |

其余部分（索引类型、地址校验、buffer/事件 MTE2_MTE3、MTE3_MTE2、resident-init
布局）均与原始 kvgather 保持一致。关闭 jitter 后（`kSwapJitterEnabled=false`
或 `jitterEnable=0`），输出与原始算子逐字节一致。
