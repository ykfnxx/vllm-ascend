# DSA Sparse Lookup Update (Turbo Fused Prefetch Batch) — 阶段三 3B 预取融合算子

## 1. 背景

`dsa_sparse_turbo_fused_prefetch_lookup_update_batch` 是
`dsa_sparse_turbo_prefetch_lookup_update_batch` 的**阶段三 3B 融合变体**
（V2）：在 prefetch turbo（resident 环 + UB 原子单遍驱逐，方案 2/3）之上，
把框架侧预取 `lookup_mask`（`valid && token < tail_start`）生成融合进
kernel，输出 dense Gather 可直接消费的 `destination_slots` 与 `miss_mask`。

设计依据同主融合算子（`dsa_sparse_turbo_fused_lookup_update_batch/design_and_test.md`
§1.1 profiling 数据；`examples/test-feedback-note.md` §阶段三 3B）。

## 2. 设计

### 2.1 接口

```python
destination_slots, miss_mask = torch.ops._C_ascend.dsa_sparse_turbo_fused_prefetch_lookup_update_batch(
    index,            # [pool_capacity, index_capacity] int32（运行时容量）
    slot_to_index,    # [pool_capacity, 10240] int32
    free_slots,       # [pool_capacity, 2048] int32
    free_head,        # [pool_capacity, 16] int32
    request_rows,     # [req_num] int32
    query_start_loc,  # [req_num + 1] int32
    query_indices,    # [query_num, 2048] int32
    query_positions,  # [query_num] int32
    verify_starts,    # [req_num] int32
    req_num,          # int attr
    block_size,       # int attr
)
```

`destination_slots` 输出的是**逻辑 slot offset**（= 框架 `plan.lookup_slots`，
`layout.lookup_offsets(slot_out)` 对全部 token），row 前缀由 dense Gather
内部按 `request_rows` 组合——与 `load_prefetch_misses` 的接线逐位兼容。


### 2.2 核内语义

预取只处理历史 KV（框架 `lookup_mask = valid && token < tail_start`），
tail/staging/当前 token 一律不进 Lookup 状态：

```
token < 0 或 token >= index_capacity              → destination = -1, miss = 0
token >= tail_start（tail/staging/当前 token）    → destination = -1, miss = 0
token < tail_start（history）                     → lookup 路径：
    hit  → destination = lookup_offsets(slot)；miss = 0
    miss 分配成功 → destination = lookup_offsets(新 slot)；miss = 1
    miss 预算截断/溢出撤销 → destination = fallback 逻辑 slot
                             （= lookup_offsets(FALLBACK_SENTINEL)
                             = SLOT_COUNT - RESIDENT_SLOTS + replaceable_base）；miss = 0
```

- `lookup_offsets(slot) = slot < 8192 ? slot : slot - 8192 + replaceable_base`。
- 与框架 `plan.lookup_slots = layout.lookup_offsets(slot_out)`（全量 token）
  逐位一致：hit/miss 分配输出逻辑 offset，fallback 输出
  `lookup_offsets(FALLBACK_SENTINEL)`（miss_mask=0，Gather 不消费）。
- maintain 继承 prefetch turbo：`PrefetchMaintain`（resident 环
  `[0, 8192)` + UB 原子抢位单遍驱逐 + 溢出撤销）。
- 语义边界与 turbo_prefetch 相同：**仅限预取流使用**（方案 3 的 free 区
  stale 临时泄漏由精确路径全量 maintain 回收）。

### 2.3 输出语义

- `destination_slots`：框架 dense 预取 Gather 的 `plan.lookup_slots`
  （逻辑 slot offset，`AsuKvGather` 内部按 `request_rows` 组合 row 前缀）；
- `miss_mask`：history miss 且分配成功 = 1，其余 0。

## 3. 正确性论证

- 分类对拍：`valid && token < tail_start` 与框架
  `make_prefetch_lookup_plan` 的 `lookup_mask` 逐位一致；
- 状态副作用与 turbo_prefetch 完全相同（分类不影响状态机）；
- Q=1 时输出与 turbo_prefetch/batch 位精确（继承）；后状态
  free_slots/cursor 非位精确（驱逐序非确定）——预取流不要求。

## 4. 验证结果（2026-08-31，A5 实机）

`test/sanity_check.py` 全绿：与 turbo_prefetch 基线 + 框架公式重建逐位对拍
（Q=1 输出位精确 + 后状态 index 一致；Q=4 位精确 + 不变量；flush 压力
全分配、head=0、free list 不变量），覆盖 128k/1024k × 90%/95%：

```
cap=128k hit=0.9 Q=1: out_bit_exact=True head0=True   （6 组全 True）
cap=128k hit=0.95 Q=4: out_bit_exact=True head0=True   （2 组全 True）
cap=128k hit=0.30 Q=3 flush: ok=True misses=8508       （2 组 flush 全 ok）
SANITY OK
```

> flush 场景不做逐位对拍（预取驱逐序非确定，Plan-2 UB 原子，flush 后
> 回填顺序传播到后续分配），按不变量 + miss 数验证。

### 4.1 性能对比（msprof op_summary Task Duration，2026-08-31）

`test/msprof_case.py`（turbo_prefetch 与 fused_prefetch 交错）+
`test/run_msprof_compare.py`，Q=1 稳态均值（n=50）：

| case | bs | cap(k) | hit% | turbo_prefetch(us) | fused_prefetch(us) | 加速比 |
|---|---|---|---|---|---|---|
| b001 | 1 | 128 | 90% | 19.670 | 17.112 | **1.149x** |
| b002 | 2 | 128 | 90% | 19.262 | 17.156 | 1.123x |
| b004 | 4 | 128 | 90% | 19.282 | 17.276 | 1.116x |
| b008 | 8 | 128 | 90% | 19.855 | 17.639 | 1.126x |
| b016 | 16 | 128 | 90% | 19.520 | 17.470 | 1.117x |
| b032 | 32 | 128 | 90% | 20.288 | 18.239 | 1.112x |

**Q=1 全面领先 ~1.11-1.15x**：分类融合消除了 `[T,K]` lookup_mask 的 GM
写读与激活判定，输出直接生成逻辑 slot。收益高于主算子（预取路径的 mask
生成占比更大）。

### 4.2 整网复核

预取流整网 profiling（2026-08-31）：`DsaSparseTurboFusedPrefetchLookupUpdateBatch`
4 次 launch 共 136.0us（旧 turbo_prefetch 155.2us，**-12%**），probe 验证
PASS/VALIDATED。

## 5. 文件清单

| 文件 | 说明 |
|---|---|
| `op_kernel/dsa_sparse_turbo_fused_prefetch_lookup_update_batch.cpp` | AIV 内核入口 |
| `op_kernel/dsa_sparse_turbo_fused_prefetch_lookup_update_batch_common.h` | 常量 + TilingData |
| `op_kernel/arch35/dsa_sparse_turbo_fused_prefetch_lookup_update_batch_simt.h` | SIMT 内核（分类融合 + PrefetchMaintain） |
| `op_host/*`（def/tiling/infershape/CMakeLists/op_api） | 注册 + tiling（复制自 turbo_prefetch，改名 + 新输入/attr） |
| `dsa_sparse_turbo_fused_prefetch_lookup_update_batch_torch_adpt.h` | Torch 适配层 |
| `test/sanity_check.py` | 精度/不变量检查 |

注册点：`csrc/torch_binding.cpp`、`csrc/torch_binding_meta.cpp`、
`csrc/build_aclnn.sh`。
