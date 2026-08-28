# AsuKvGatherSim Test Cases

## 概述

`asu_kv_gather_sim` 的正确性以原始 kvgather 的输出为 golden：关闭抖动时二者
必须逐字节一致；开启抖动时 jitter 只增加延迟、不改数据路径，输出仍必须与
golden 一致。性能测试通过 msprof 采集，对比 orig 与 sim（唯一差异为随机抖动）。

## 正确性用例（test_gather_sim_orig.py）

- `query = topk = 2048`；全部索引张量 `int32`；token 取值 `[1, INDEX_SIZE)`，
  random 模式下为完全不同的 tokenId（`replace=False`）。
- 命中率由"顺序 query [1..2048] 预分配前 `n_hit` 个 token 为 resident"精确控制，
  无需 lookup 算子联合测试。
- golden：CPU 参考按 miss_mask 计算期望写入，比较完整 destination KV/RoPE
  （含未触碰的 sentinel 区域），使用 `torch.npu.synchronize()` 后三层对比。

| case | batch | hit ratio | query kind |
|---|---|---:|---|
| hr90_b1_seq | 1 | 90% | sequential |
| hr90_b2_seq | 2 | 90% | sequential |
| hr90_b4_seq | 4 | 90% | sequential |
| hr50_b1_seq | 1 | 50% | sequential |
| hr50_b2_seq | 2 | 50% | sequential |
| hr50_b4_seq | 4 | 50% | sequential |
| hr90_b4_rand | 4 | 90% | random |
| hr50_b4_rand | 4 | 50% | random |

miss 目标 slot 顺序分配自由 slot `[n_hit, n_hit + n_miss)`，保证 slot 合法且
不重叠。

## 性能测试（bench_gather_sim_orig.py + msprof）

- 必须用 msprof 单次采样（读 `op_summary.csv` 的 `Task Duration(us)` 与
  `aiv_scalar_time(us)`），不用 torch-event timing；在空闲 NPU 上运行，避免
  vLLM 负载干扰。
- 矩阵：命中率 90% / 50% × batch `1 2 4 8 16 24 32 64`，query=2048。
- orig 与 sim 使用完全相同输入（唯一差异为 jitter 注入）。
- 预期：每核在循环结束后只忙等一次，时长 = 对 N = 本核实际 miss 数 × 2 次抽样取
  最大 delay；N 随命中率/miss 分布变化（90% b1: N≈10, E≈88us；50% b1: N≈43,
  E≈110us；90% b64: N≈546, P(≥228us)≈42%）；Task Duration ≈ orig + 期望抖动，
  scalar 占比趋近 1（`scalar/task ≈ 0.95~0.99`）。

## 必测回归

- `jitterEnable=0` 或编译期关闭 `kSwapJitterEnabled` 时输出与原始算子逐字节一致；
- `jitterEnable=1` 时输出仍与原始算子一致（jitter 仅增加延迟，不改数据）；
- 连续区间分配（与原版一致）必须保证每个 (req, query) pair 恰好被一个核
  处理一次（覆盖数 = `pair_count`），无重复、无遗漏；
- 抖动主导时 `aiv_scalar_time` 随 `Task Duration` 同步增长，比值趋近 1；
- 命中率为 0% 与 100% 的边界：0% 全部 miss 走完整拷贝路径，100% 零次搬运。
