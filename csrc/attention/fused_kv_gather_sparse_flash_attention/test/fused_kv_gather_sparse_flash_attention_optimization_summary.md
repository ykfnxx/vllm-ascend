# FusedKVGatherSparseFlashAttention A3 优化总结

## 优化目标

将资源从收益有限的独立 `DsaLookup`/LightningIndexer 融合转向
`FusedKVGatherSparseFlashAttention` 的生产热路径。目标 workload 为
GLM-5.2、BS=12、MTP=3、16K/64K、Host miss=10%。

## 实现

- 每个 SFA S2 tile 将连续 `route_plan`（最多 512 个 int32）一次搬入独占
  2 KiB UB，tile 内用 LocalTensor 解析，替代逐 route 的 GM 标量读取。
- `query_pool_entries` 和 `hot_source_rows` 从逐 route 重复读取改为每 tile
  各读取一次。
- 保留 block table 按需读取；整表 DMA 在实测中会增加同步和过量搬运。
- route UB 与 Vec1/Vec2 的 `tmpBuff1` 分离，避免预加载流水中的异步覆盖。

## 迭代记录

| 迭代 | 方案 | BS12/MTP3/16K FP16 | 精度 | 结论 |
| ---- | ---- | -----------------: | ---- | ---- |
| Baseline | 逐项 route/row/table GM 读取 | 221.204 us | PASS | 基线 |
| 1 | route + row + 整行 block table 搬入 tmpBuff1 | 240.773 us | 冒烟 PASS | 整表搬运和同步导致回退，放弃 |
| 2 | 仅 route tile + row 标量缓存，复用 tmpBuff1 | 217.224 us | 18/30 | 暴露 UB 异步复用竞争，放弃 |
| 3 | route tile 使用独占 2 KiB UB | 217.472 us | 30/30 | 最终方案 |

## Baseline 与最终性能

固定 `torch_npu.profiler` schedule：warmup=5、active=5；下表为
`op_statistic.csv` 的算子 Total Time / 5。

| Case | DType | Baseline(us) | Optimized(us) | 降幅 |
| ---- | ----- | -----------: | ------------: | ---: |
| BS1/MTP0/4K | FP16 | 118.482 | 116.906 | 1.33% |
| BS1/MTP0/4K | BF16 | 117.410 | 117.334 | 0.06% |
| BS4/MTP3/16K | FP16 | 123.526 | 123.106 | 0.34% |
| BS4/MTP3/16K | BF16 | 123.502 | 122.198 | 1.06% |
| BS12/MTP3/16K | FP16 | 221.204 | 217.472 | 1.69% |
| BS12/MTP3/16K | BF16 | 221.120 | 216.884 | 1.92% |
| BS12/MTP3/64K | FP16 | 222.192 | 218.628 | 1.60% |
| BS12/MTP3/64K | BF16 | 221.392 | 219.940 | 0.66% |

八组平均延迟由 171.104 us 降至 169.059 us，降低 1.20%；四个
BS=12/MTP=3 生产 case 平均由 221.477 us 降至 218.231 us，降低 1.47%。
相对 `DsaKVGather + SparseFlashAttention` 完整链路，最终生产 case 的
加速比为 3.248–3.267 倍。

## 精度

- 30/30 通过，FP16/BF16 的 MERE、MARE 和 MaxAbsErr 均为 0。
- 覆盖 BS 1–49、MTP 0/3、4K–128K，以及 0%–100% Host miss。
- 首次完整精度测试发现 `tmpBuff1` 复用竞争；使用独占 route UB 后，原
  12 个失败 case 全部恢复且连续完整回归通过。

## 结论

- 高价值收益仍来自消除 selection-cache 往返：BS=12/MTP=3 完整链路约
  3.25 倍；route 批量化是在此基础上的 1%–2% 增量优化。
- block table 在当前随机访问模式下不适合每 tile 整行搬运，特别是 64K
  Host table；下一步应考虑 cache-line 分组或与 payload DMA 地址生成协同。
- SFA 预加载流水中的 UB 生命周期必须按异步阶段建模，不能仅依据函数调用
  顺序复用 buffer。
