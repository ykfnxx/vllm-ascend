# DSA Sparse Lookup Update (Turbo Fused Batch) — 阶段三 3B 融合算子

## 1. 背景

`dsa_sparse_turbo_fused_lookup_update_batch` 是
`dsa_sparse_turbo_lookup_update_batch` 的**阶段三 3B 融合变体**（V2，新算子名，
不改旧算子 ABI）：在 turbo（request 级 maintain）之上，把框架侧的
lookup_mask 生成、history/tail/staging 分类、地址映射（`torch.where` 链）与
Gather miss mask 全部融合进 kernel，输出 SFA 可直接消费的
`mapped_indices` 与 dense Gather `miss_mask`。

### 1.1 Profiling 依据（decode-profile kernel_details，2026-08-31，GLM-5.2 prefetch）

生产 decode 步中两个 turbo 算子本身合计 ~773us（主 618us/16 launch、预取
155us），而框架侧与 lookup 分类/映射相关的小算子累计约 8.4ms：

| 小算子 | count | total_us | 对应融合目标 |
|---|---|---|---|
| Cast (bool→int32 等) | 968 | 1907.7 | lookup_mask / 各 mask 转换 |
| Index | 211 | 714.7 | verify_starts 展开 |
| IndexPutV2 | 229 | 723.3 | merged_indices 写回 |
| Fill | 379 | 775.7 | fallback/invalid 填充 |
| LogicalAnd | 326 | 495.9 | valid/history/tail/staging |
| SelectV2 (torch.where) | 218 | 460.7 | 三路 where 链 |
| GatherV2 (repeat_interleave) | 292 | 631.6 | query_request_rows 展开 |
| BroadcastTo | 217 | 542.0 | mask 展开 |
| Sub | 171 | 316.6 | query_lengths / tail_starts |
| Add/Adds | 617 | 1387.6 | 地址计算 |

融合核心收益不是减少几次整数比较，而是消除完整 `[T,K]` 中间张量
（`lookup_mask` 等，B=12/Q=6/T=72/K=2048 时单份 576 KiB）的多轮 GM
读写与 kernel launch。完整数据见
`examples/test-feedback-note.md` §阶段三。

### 1.2 与 turbo 的差异

| 项 | turbo（基线） | fused（本算子） |
|---|---|---|
| lookup_mask 输入 | 框架生成 `[T,K]` INT32 后传入 | ❌ 删除；核内按 `verify_starts` 分类生成 |
| 输出 | `slot_out` + `miss_out` | `mapped_indices`（SFA 逻辑地址）+ `miss_mask`（dense Gather） |
| 分类 | 只有 valid 判定（框架 mask 已分类） | 核内完整 history/tail/staging/invalid 分类 |
| 地址映射 | 框架 `lookup_offsets` + `tail_offsets` + `staging_offsets` + where 链 | 核内一次完成 |
| maintain | request 级（turbo 语义） | 不变（继承） |

## 2. 设计

### 2.1 接口

```python
mapped_indices, miss_mask = torch.ops._C_ascend.dsa_sparse_turbo_fused_lookup_update_batch(
    index,            # [pool_capacity, index_capacity] int32（运行时容量）
    slot_to_index,    # [pool_capacity, 10240] int32
    free_slots,       # [pool_capacity, 2048] int32
    free_head,        # [pool_capacity, 16] int32
    request_rows,     # [req_num] int32
    query_start_loc,  # [req_num + 1] int32
    query_indices,    # [query_num, 2048] int32
    query_positions,  # [query_num] int32
    verify_starts,    # [req_num] int32
    tail_starts,      # [req_num] int32
    req_num,          # int attr
    block_size,       # int attr
    is_mtp,           # int attr（1=MTP，0=non-MTP）
)
```

- `query_positions`、`verify_starts` 和 `tail_starts` 直接复用框架每个
  Decode 步在主流生成的 `PackedAddressingMetadata`。算子不再重复执行
  `floor(verify_start / block_size) * block_size`。
- layout 常量（`tail_base`/`staging_base`/`fallback_slot`/`replaceable_base`）
  由 host tiling 按内建常量推导，attr 只传 `block_size`：

```
resident_blocks   = ceil(RESIDENT_SLOTS / block_size)      # 8192
replaceable_blocks = ceil(REPLACEABLE_SLOTS / block_size)   # 2048
replaceable_base  = resident_blocks * block_size
tail_base         = (resident_blocks + replaceable_blocks) * block_size
fallback_slot     = tail_base + block_size
staging_base      = fallback_slot + 1
```

（block_size=128 时分别为 8192/10240/10368/10369，与框架
`HotCacheLayout` 逐项一致。）

### 2.2 核内分类语义（与框架 where 链逐位对齐）

每 request 一次读取 `verify_start = verify_starts[req]` 和
`tail_start = tail_starts[req]`；每个 query 读取
`current_position = query_positions[query_id]`。对每个 Top-K token：

```
token < 0 或 token >= index_capacity          → mapped = INVALID(-1),      miss = 0
token < tail_start                            → history（lookup 路径）
tail_start <= token < verify_start            → tail:    mapped = tail_base + token - tail_start
verify_start <= token <= current_position     → staging(MTP):   staging_base + token - verify_start
                                                tail(non-MTP):  tail_base + token - tail_start
token > current_position                      → mapped = INVALID(-1),      miss = 0
```

history 分支（继承 turbo ProcessQuery）：

- hit（`index[token]` 有效）：`mapped = lookup_offsets(slot)`，`miss = 0`，
  并 protect；
- miss 且分配成功：`mapped = lookup_offsets(新 slot)`，`miss = 1`，
  protect + 记台账；
- miss 但预算截断/溢出撤销：`mapped = fallback_slot`，`miss = 0`
  （对应框架 `fallback_mask → fallback_slot` 路径）。

`lookup_offsets(slot) = slot < 8192 ? slot : slot - 8192 + replaceable_base`
（与框架 `HotCacheLayout.lookup_offsets` 一致）。

### 2.3 输出语义

- `mapped_indices`：SFA `sparse_indices` 直接消费（框架 `plan.mapped_indices`）。
  对 history miss，mapped 同时就是 Gather destination 逻辑 slot；
- `miss_mask`：dense Gather `miss_mask`（框架 `dense_miss_mask` =
  `miss_out & history_mask & ~fallback_mask`）。tail/staging/invalid 一律 0。

### 2.4 maintain 语义

继承 turbo：request 级 `MaintainRequest`（一次 victim 计数 + 一次驱逐）、
flush（累计分配逼近 FREE_SLOT_COUNT 时提前 maintain）、溢出撤销
（alloc ledger）。撤销时 `mapped[offset] = fallback_slot`、`miss_mask[offset] = 0`。

## 3. 正确性论证

- **分类对拍**：核内分类与框架
  `valid_mask / history_mask / tail_mask / staging_mask → mapped` 的 where 链
  逐位一致（见 §2.2 表）；`is_mtp=0` 时 staging 段合并进 tail（框架
  `tail_mask = valid & tail_start <= token <= current_position`）。
- **hit/miss/fallback 输出**：与框架 `slot_out/miss_out → lookup_offsets
  → fallback → where 链` 的映射一致（Q=1 时输出+后状态与 turbo 位精确
  可继承；Q≥2 与 batch 不变量等价）。
- **不变量**：head=0、free list 2048 项不重复且 slot_to_index==-1、
  分配双向一致（继承 turbo 论证）。
- **状态副作用**：index/slot_to_index/free_slots/free_head 的写入与
  turbo 完全相同（分类不影响状态机）。

## 4. 原始分支验证结果（2026-08-31，A5 实机）

以下结果来自精简 ABI 前的同语义 kernel。当前版本新增框架提供的
`tail_starts` 并删除只写不读的线程局部数组，需重新编译后复跑本目录测试；
框架侧已补充参数接线、fallback 选择和 eager logical-slot 回归测试。

`test/sanity_check.py` 全绿：与 turbo 基线 + 框架公式重建逐位对拍
（Q=1 输出+后状态位精确；Q=4 位精确 + 不变量；flush 压力全分配、head=0），
覆盖 128k/1024k × 90%/95% × MTP/non-MTP：

```
cap=128k hit=0.9 Q=1 mtp=1: out_bit_exact=True head0=True   （8 组 Q=1 全 True）
cap=128k hit=0.95 Q=4 mtp=1: out_bit_exact=True head0=True   （4 组 Q=4 全 True）
cap=128k hit=0.30 Q=3 flush: ok=True misses=8460             （4 组 flush 全 ok）
SANITY OK
```

### 4.1 性能对比（msprof op_summary Task Duration，2026-08-31）

`test/msprof_case.py`（turbo 与 fused 交错，工作量对齐：同一 history token 集）
+ `test/run_msprof_compare.py`，稳态均值（n=50）：

| Q | 容量 | hit% | turbo(us) | fused(us) | 加速比 |
|---|---|---|---|---|---|
| 1 | 128k | 90% | 32.4（batch=16 均值） | 31.6 | 1.01-1.03x（6 case） |
| 2 | 128k | 90/95% | 46.3/43.9 | 45.3/42.1 | 1.02-1.04x |
| 4 | 128k | 90/95% | 74.2/69.9 | 70.2/66.0 | 1.06x |

> Q=1 时 kernel 本身分类开销很轻，收益主要在框架侧（消除 mask 生成与
> where 链小算子）；收益随 Q 增长（Q=4 达 1.06x）。框架合入后的整网收益
> 见整网 decode-profile 复核（小算子消除为主）。

### 4.2 整网 decode-profile 复核（2026-08-31，GLM-5.2 prefetch）

框架合入（`dsa_offload.enable_turbo_fused_lookup` / 
`enable_turbo_fused_prefetch_lookup` 默认开）后整网 profiling：

| 指标 | turbo（旧） | fused（新） | 变化 |
|---|---|---|---|
| 主 Lookup kernel（16 launch） | 618.1us | 557.3us | **-10%** |
| 预取 Lookup kernel（4 launch） | 155.2us | 136.0us | **-12%** |
| SelectV2（where 链） | 460.7us | 271.4us | -189us |
| Sub（tail_starts/差分） | 490.1us | 302.5us | -188us |
| Add/Adds | 1964.5us | 1806.3us | -158us |
| LogicalAnd | 539.0us | 456.7us | -82us |
| Fill | 830.4us | 774.9us | -55us |

- probe 验证 PASS/VALIDATED（算子名校验已含 fused 两种形态）；
- 旧 turbo kernel 不再出现；`[T,K]` lookup_mask 的 GM 往返与 where 链
  消除。

**算子序列复核**：LightningIndexer → FusedLookup 之间从 ~24 个 mask 链算子
减到 **6 个**（Cast/Sub/Cast/RepeatInterleave/Cast/Index——均为
`query_request_rows`/`verify_starts` 的必要输入准备）；FusedLookup →
AsuKvGather 之间仅剩 2 个（Cast + GatherV2=block_table 选取，后者是
AsuKvGather 的必要输入）。框架侧小算子（Cast/LogicalAnd/SelectV2/Sub/
GreaterEqual/Less/FloorDiv/Fill/RepeatInterleave）合计消除 ~1.36ms
（41.3→39.9ms），其中 LogicalAnd -278us、Less -281us、SelectV2 -189us、
Sub -186us、GreaterEqual -159us。

## 5. 文件清单

| 文件 | 说明 |
|---|---|
| `op_kernel/dsa_sparse_turbo_fused_lookup_update_batch.cpp` | AIV 内核入口 |
| `op_kernel/dsa_sparse_turbo_fused_lookup_update_batch_common.h` | 常量 + TilingData（+layout 派生字段） |
| `op_kernel/arch35/dsa_sparse_turbo_fused_lookup_update_batch_simt.h` | SIMT 内核（分类融合 + turbo maintain） |
| `op_host/*`（def/tiling/infershape/CMakeLists/op_api） | 注册 + tiling（复制自 turbo，改名 + 新输入/attr） |
| `dsa_sparse_turbo_fused_lookup_update_batch_torch_adpt.h` | Torch 适配层 |
| `test/sanity_check.py` | 精度/不变量检查 |

注册点：`csrc/torch_binding.cpp`（include/def/impl）、
`csrc/torch_binding_meta.cpp`（meta + impl）、`csrc/build_aclnn.sh`
（ascend950 列表）。
