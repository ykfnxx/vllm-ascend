# DSA Offload grouped hidden-state prefetch

本文记录 `dsa-offload-0.23-graph` 分支中 GLM-5.2 grouped hidden-state
prefetch 的实现和流水分析。当前实现支持 `FULL_DECODE_ONLY`，所有 group
共享一条预取流；预测计算、预取 Lookup/Update 和预取 Gather 均在该流上
按顺序执行。

## HiCached 与 Prefetch Lookup 之间的等待

Profiling 中，预取流的 `LightningIndexerHiCached` 与
`DsaSparseTurboPrefetchLookupUpdateBatch` 之间可能出现一段较长的空闲。这不是
两个算子内部执行阻塞，而是当前代码有意把预取拆成两个阶段，并把后半段锚定在
源层精确 Gather 完成之后。

当前流水为：

```text
预取流：
预测 Q → LightningIndexerHiCached → 保存 predicted Top-K → 空闲/等待
                                                    ↓
主流：
源层预处理 → 主 LightningIndexer → 主 LookupUpdateBatch → 主 Gather 完成
                                                    ↓
预取流：
Prefetch LookupUpdateBatch → 目标 group 的四个 Prefetch Gather
```

预取 Top-K 在
[`GroupedPrefetchController.start`](../../vllm_ascend/dsa_offload/prefetch.py)
中提前计算。HiCached 完成后只保存 `topk_indices` 和
`compute_done_event`，不会立即执行预取 Lookup。

源层进入
[`resolve_sfa_inputs`](../../vllm_ascend/dsa_offload/sfa.py) 后，主流先执行：

```text
主 LookupUpdate
  ↓
load_plan_misses：当前源层精确 Gather
  ↓
release_after_exact_load
```

`release_after_exact_load()` 在主流当前位置记录事件。共享预取流等待该事件后，
才执行 `make_prefetch_lookup_plan()`、Prefetch LookupUpdate 和目标 group 的四个
Gather。因此，HiCached 与 Prefetch Lookup 之间的空洞实际覆盖的是源层主流程，
并非预取流仍在计算 Top-K。

### 当前排布的目的

这一锚点属于性能策略：

- 当前请求的精确主 Gather 优先；
- 避免 Prefetch Lookup/Gather 过早与主流程竞争 AIV、MTE 和 HBM；
- 主 Gather 完成后，预取 Lookup/Gather 可以与源层后续 SFA、O-proj 和 MoE
  重叠。

使用 `kvgather_sim` 时，主 Gather 包含真实的 `AsuKvGather` 调用及其模拟延迟，
所以 HiCached 后的等待会比 `mock` 后端更明显。MTP 还会增加主流程 mask 构造和
batch Lookup 的时间，但 MTP 不是该等待存在的根本原因。

### 正确性依赖与性能锚点的区别

预取 Lookup 并不依赖源层主 Gather 的数据结果：

- 源层主 Gather 操作当前 cohort 的 KV；
- Prefetch Lookup 操作下一目标 cohort 的 index、slot 和 free-list；
- 两者使用不同的 Lookup 状态；
- Prefetch Lookup/Gather 真正必须完成的位置，是目标 group 的精确 Lookup
  之前。

因此，“等待源层主 Gather”不是正确性依赖，而是人为选择的流水锚点。当前
正确性保护还包括：

- 目标层精确 current-token indexer-cache 写入前，等待预取 HiCached 完成读取；
- 目标 group 精确 Lookup 前，等待预取 Lookup 和四层 payload Gather 完成，
  避免 index 已命中但 Hot Cache payload 尚未就绪。

## 后续可验证的拆分方案

如果 Profiling 表明当前空闲过大、预取后半段无法在目标 group 前完成，可以把
后半段进一步拆分：

```text
LightningIndexerHiCached 完成
  ↓
提前执行 Prefetch LookupUpdateBatch
  ↓
保存 PrefetchPlan，暂不执行 Gather
  ↓
当前源层主 Gather 完成
  ↓
释放目标 group 的四个 Prefetch Gather
```

该方案能提前掩盖 Prefetch Lookup 的时延，同时继续保证精确主 Gather 优先。
实现时需要将当前单一 ready 状态拆成至少两个图稳定事件：

1. `prefetch_lookup_done`：预测 Lookup/Update 已完成；
2. `prefetch_payload_ready`：目标 group 四层 Gather 已完成。

目标精确 Lookup 只能等待 `prefetch_payload_ready`，不能只等待
`prefetch_lookup_done`。是否采用该排布，需要通过完整 decode 步 profiling
确认提前 Lookup 不会反向拉长主 LightningIndexer、主 Lookup、SFA 或通信窗口。

## Packed metadata 复用与 Turbo Lookup 融合

Profiling 中，主 `LightningIndexer`、预取 `LightningIndexerHiCached` 与对应
LookupUpdate 之间存在较多 `Index`、`RepeatInterleave`、比较、逻辑、Cast 和
地址映射小算子。优化需要区分两类数据：

- 只取决于本 decode 步请求布局和 position 的公共元数据，可以由框架生成一次，
  供所有 cohort、主流程和预取流程复用；
- 依赖主 Top-K 或预测 Top-K 内容的分类结果，不能跨流程共享，最终应融合进
  LookupUpdate 算子。

推荐按“框架生成一次 → 主/预取共享 → Turbo 算子融合”三个阶段实施。前两个阶段
不需要重新编译算子；第三阶段需要修改并重新编译两套 Turbo LookupUpdateBatch
算子。

### 为什么 LookupUpdate 融合 Maintain 后反而多了小算子

这不是 Lookup 与 Maintain 融合本身引入的额外计算，而是两个分支的
Lookup 算子接口边界不同。`dsa-sparse-0.23-graph-prefetch` 在
host-memory + prefetch 路径默认使用 `asu_hbm_index_lookup_opt`；当前
DSA Offload 路径使用 `dsa_offload_lookup_update(_batch)` 或两套 Turbo
LookupUpdateBatch。

| 语义 | 旧 `asu_hbm_index_lookup_opt` | 当前 LookupUpdate |
| --- | --- | --- |
| 状态更新 | Lookup 后再在 AICPU 流执行 Maintain | Lookup 内部同步完成索引状态更新 |
| Lookup 前分类 | 算子输入 `row_modes/dense_tail_starts/resident_tail_starts`，核内分类 history/tail/inactive row | 算子要求框架先生成 `[T,K]` `lookup_mask` |
| MTP packed query | 固定每请求一行 `[B,2048]`，没有 staging 分类 | Batch 接口支持 `[T,2048]` 和 `query_start_loc`，需区分 history/tail/staging |
| Lookup 输出 | `slot_out/miss_out/mapped_slot_out` | 只返回原始 `slot_out/miss_out` |
| SFA 地址 | `mapped_slot_out` 已是 SFA 可消费的 Hot Cache 地址 | 框架再做 slot offset、fallback、tail/staging 映射和多层 `where` |
| Hot Cache block table | step metadata 中已准备 `hot_block_table` | 需把原 KV block table 与 Hot Cache 虚拟行组成 SFA view |

因此，旧算子是“状态修改较窄、地址语义较宽”：Maintain 虽然分离，
但 Lookup 已经在核内完成 Top-K 分类和 SFA 地址映射。当前算子则是
“状态修改较宽、地址语义较窄”：Lookup + Maintain 已融合，但 ABI
仍只处理被 `lookup_mask` 选中的 history token，没有输出最终 SFA 地址。

对应到 profiling，当前小算子可以精确分为：

1. `LightningIndexer → LookupUpdate`：Top-K reshape/Cast，以及
   `valid/history/tail/staging` 比较、逻辑和 `lookup_mask` Cast；
2. `LookupUpdate → Gather/SFA`：`slot_out` 到 resident/replaceable offset 的
   转换、fallback 处理、tail/staging offset、嵌套 `where`、
   `active_misses` 和 dense Gather mask；
3. MTP 额外增加 packed request row、verify/tail/staging 边界。但 MTP
   不是唯一原因；non-MTP 仍需要 history/tail/fallback 映射，只是没有
   staging 分支。

这也规定了优化边界：`query rows/query positions/verify starts/tail starts`
这类与 Top-K 内容无关的紧凑元数据应由框架提前统一生成；逐
Top-K 的 mask、fallback、miss 和最终 `mapped_indices` 应融入 LookupUpdate
V2。不应把 `[T,K]` mask 提前物化并尝试跨主/预取复用，因为两条
路径的 Top-K 不同。

### 公共元数据与 Top-K 相关数据的边界

每个 decode 步可以共享以下元数据：

| 元数据 | 推荐 shape/dtype | 含义 |
| --- | --- | --- |
| `query_start_loc` | `[B+1]`, INT32 | 每个请求在 packed query 中的累计起止位置 |
| `query_lengths` | `[B]`, INT64 | 相邻 `query_start_loc` 的差分 |
| `cumulative_query_lengths` | `[B]`, INT32 | `query_start_loc[1:]`，供主/预取 Indexer 使用 |
| `query_request_rows` | `[T]`, INT32 | 每个 query row 对应的 Hot Cache request row |
| `verify_starts` | `[B]`, INT32 | 每个请求本轮验证区间起点 |
| `tail_starts` | `[B]`, INT32 | `floor(verify_start / block_size) * block_size` |
| `expanded_verify_starts` | `[T]`, INT32 | 融合前供框架逐 Top-K 分类使用 |
| `expanded_tail_starts` | `[T]`, INT32 | 融合前供框架逐 Top-K 分类使用 |
| `expanded_query_starts` | `[T]`, INT32 | MTP staging slot mapping 使用 |
| `gather_destination_block_table` | `[T,H]`, INT32 | 主/预取 Dense Gather 共用的 Hot Cache 虚拟行表 |

这里需要避免一个命名混淆：HiCached 的 `actual_seq_lengths_query` 当前使用的是
`query_start_loc[1:]`，它是每个请求的累计结束位置，而不是差分后的
`query_lengths`。两者不能错误替换。HiCached 的
`actual_seq_lengths_key` 则可以直接复用紧凑的 `verify_starts`。

以下结果依赖各自的 Top-K，不能在主流程与预取流程之间共享：

- `valid_mask`；
- `history_mask`；
- `tail_mask`；
- `staging_mask`；
- `lookup_mask`；
- `fallback_mask`；
- `active_misses`；
- 最终 `mapped_indices`。

主流程 Top-K 和预测 Top-K 不同，因此这些逐 Top-K 结果属于算子融合范围，而不是
公共 packed metadata。

### 阶段一：每个 decode 步只生成一次元数据

当前由 `DSAOffloadBatch` 持有以下结构：

```python
@dataclass
class PackedAddressingMetadata:
    query_lengths: torch.Tensor
    cumulative_query_lengths: torch.Tensor
    query_request_rows: torch.Tensor
    verify_starts: torch.Tensor
    tail_starts: torch.Tensor
    expanded_verify_starts: torch.Tensor | None
    expanded_tail_starts: torch.Tensor | None
    expanded_query_starts: torch.Tensor | None
    gather_destination_block_table: torch.Tensor | None
```

其中 `verify_starts/tail_starts` 是 Fused 主流程与预取流程共同消费的 compact
`[B]` 边界。`expanded_verify_starts/expanded_tail_starts` 只服务旧 Lookup fallback，
按需生成一次并由主流程和预取流程复用；两套 Fused 都开启时不物化这两个 `[T]`
tensor。`expanded_query_starts` 只用于 MTP staging slot mapping，non-MTP 不生成。

Eager 模式下，每个 `DSAOffloadBatch` 只对应一个 decode 步，可以在 batch 创建后
生成一次。图模式下不能在捕图外根据动态 buffer 预先计算；应在图 forward 的固定
入口、主流上生成一次，使这些算子成为图节点，图重放时根据新的
`request_rows`、`query_start_loc` 和 `query_positions` 重新计算。若采用 lazy
`get_packed_addressing_metadata()`，则必须确保新一次捕图会清空 Python 缓存，
不能因为上一次捕图已经保存 tensor 而漏捕元数据生成节点。

公共元数据应同时替换以下位置的重复计算：

1. 主流程 `make_lookup_plan()`；
2. 预取流程 `make_prefetch_lookup_plan()`；
3. 每层 `prepare_main_slot_mapping()`；
4. 预取 HiCached 的 historical length 生成。

否则只修改 Lookup plan 仍会在 slot mapping 中保留相同的差分、展开和 Cast，无法
真正做到每个 decode 步只生成一次。

元数据由主流生成后，预取流可以利用已有的 prefetch-start event 建立依赖，无需
增加一套预取流元数据或额外的全局同步。后续若继续提前 Prefetch Lookup，只需
保证预取流等待 `metadata_ready`。

阶段一主要消除跨 cohort 重复的：

- `Sub`；
- `Cast`；
- `Index`；
- `RepeatInterleave`；
- `FloorDiv`；
- `Mul`。

### 阶段二：主流程和预取流程共享 packed metadata

主流程应简化为：

```text
semantic Top-K + shared packed metadata
  ↓
主流程分类与 LookupUpdate
  ↓
mapped indices + exact Gather plan
```

预取流程应简化为：

```text
predicted Top-K + 同一份 shared packed metadata
  ↓
预取 history 分类与 LookupUpdate
  ↓
Prefetch Gather plan
```

共享范围只包含请求行、query 范围和 position 边界。两条流程仍使用不同 Top-K 和
不同目标 cohort 的 LookupState，不能共享 mask、Lookup 结果或 miss plan。

在当前排布中，Prefetch Lookup 位于源层精确 Gather 之后，所以共享元数据一定已
就绪。把 Lookup 提前后，也只需增加准确的跨流 ready event，不应在预取流重新生成
同一份元数据。

### Lookup 前后地址元数据提前准备

仅统一 `query_lengths/query rows/verify/tail starts` 仍不足以清理
LookupUpdate 到 Gather/SFA 之间的小算子。当前框架进一步在每个 decode 步的主流
入口准备以下结果：

| 结果 | 复用范围 | 正确性边界 |
| --- | --- | --- |
| `main_slot_mapping` | 使用同一原始 slot mapping 的层 | MTP 映射到 staging，非 MTP 映射到 tail；不同 KV cache group 不混用 |
| Decode-only block table | 使用同一原始 block table 的 Indexer/HiCached | 按 block-table tensor 来源缓存，不能跨编号体系复用 |
| Gather destination block table | 主 Gather、预取 Gather、所有物理层 | 来自 DSA Hot Cache 固定虚拟行布局，与各层 KV 内容无关，可以安全共享 |
| SFA block table/KV length view | 当前连续使用同一 block table/KV length 的层 | SFA workspace 是可写 buffer；来源变化立即重新 compose，不能把一份 workspace 当成多组永久缓存 |

这里必须区分“KV 内容”和“KV 地址”：每个物理层的 KV cache 内容肯定不同，但同一
KV cache group 的层通常使用相同的 block 编号和 slot mapping，因此地址元数据可以
复用。`model_runner_v1.py` 会为不同 `kv_cache_gid` 构造不同 block table 和 slot
mapping，所以实现按输入 tensor 来源分组缓存，而不是给整个 batch 强制使用唯一一
份 SFA block table。若层序列从 group A 切到 group B，再回到 group A，单一 SFA
workspace 会在每次来源切换时重新 compose，避免使用已被覆盖的旧视图。

这些准备在每层的 LightningIndexer/LookupUpdate 之前完成；Gather 写入 KV payload
仍是 SFA 的真实数据依赖，但 SFA block table 和 KV length 不依赖 Lookup miss 结果，
无需留在 LookupUpdate 与 SFA 的关键区间。预取流通过已有的 start event 看到主流已
准备好的 packed metadata 和 Gather 行表，不新增全局同步。

本轮实现同时覆盖 MTP 和非 MTP：公共 request metadata、Gather 行表和按来源缓存的
block table 两者共用；只有 `main_slot_mapping` 的 row offset 分别遵循 staging 与
tail 语义。LookupUpdate/LookupUpdateBatch 的选择、Top-K mask、miss/fallback 和地址
映射在普通 fallback 路径保持原实现；MTP Turbo + Fused 路径已由下节的
LookupUpdate V2 在算子内完成。

完成上述准备后，主路径的目标边界是：

```text
step addressing（每步一次、按 cache group 隔离）
  ↓
LightningIndexer
  ↓
Fused LookupUpdate（分类 + 状态更新 + 最终地址映射）
  ↓
Gather（直接复用 destination block table）
  ↓
SFA（直接复用当前 group 的 block table/KV length view）
```

预取路径相应为：

```text
predicted Top-K + shared step addressing
  ↓
Fused Prefetch LookupUpdate（history 分类 + 状态更新 + miss plan）
  ↓
Prefetch Gather（复用同一 Gather destination block table）
```

LookupUpdate 前后的 `valid/history/tail/staging/lookup/fallback` mask、slot 到 Hot
Cache offset 的转换、嵌套 `where` 和 dense miss mask 依赖各自 Top-K，不能作为
公共元数据跨主流程和预取共享。Fused V2 已把它们留在各自算子内即时生成和消费，
不再在框架提前物化 `[T,K]` 中间 tensor；关闭 Fused 时仍保留原公式作为回退和
对拍基线。

### MTP 与 non-MTP 的覆盖关系

当前并不是所有配置都会调用“两套 Turbo LookupUpdateBatch”：

| 配置 | 主 Lookup | 预取 Lookup | 框架元数据复用 | Turbo 融合 |
| --- | --- | --- | --- | --- |
| MTP 开、Turbo/Fused 开、Prefetch 开 | Fused Turbo Batch | Fused Turbo Prefetch Batch | 完整收益 | 完整覆盖 |
| MTP 开、Turbo/Fused 开、Prefetch 关 | Fused Turbo Batch | 无 | 有收益 | 只覆盖主算子 |
| MTP 开、Turbo 关 | 普通 Batch | 普通 Batch | 有收益 | 暂不覆盖 |
| MTP 关、Prefetch 开 | 非 Batch Lookup | 非 Batch Lookup | 有收益 | 两个 Turbo 修改不生效 |
| MTP 关、Prefetch 关 | 非 Batch Lookup | 无 | 收益较小 | 不生效 |

MTP 下，一个请求包含多个 packed query，且需要区分 history、tail、staging，重复
小算子更多，因此是优先目标。non-MTP 通常每请求只有一个 query，没有 staging；
后续若要覆盖，可以新增 non-MTP fused Lookup，或验证 query length 为 1 时统一走
Batch Turbo。第一版不应同时改两条路径，以免扩大正确性验证范围。

### 阶段三：两套 Fused Turbo LookupUpdateBatch 协同接入

`dsa-offload-0.23-graph` 新增的两套阶段 3B 算子已经接入，但没有直接照搬
原 ABI。框架元数据优化后，原算子接口存在三处不匹配：

1. 主算子只接收 `verify_starts`，在 kernel 内重复计算框架已有的
   `tail_starts`；
2. 预取算子接收的 `query_positions` 从未被使用，`verify_starts` 也只用于
   再次计算 `tail_starts`；
3. 原 eager 主 miss plan 把 fused 已输出的逻辑 Hot Cache offset 当成原始
   Lookup slot 再做一次 `lookup_offsets`，存在二次映射风险。

本轮将算子和框架边界收敛为：

```text
框架每 Decode 步一次：
  query_start_loc / query_positions / verify_starts / tail_starts
                             │
              ┌──────────────┴──────────────┐
              │                             │
主 Fused LookupUpdate              预取 Fused LookupUpdate
query_positions[T]                 tail_starts[B]
verify_starts[B]                   （不再传 query_positions）
tail_starts[B]                     （不再传 verify_starts）
              │                             │
mapped_indices + miss_mask         logical_slots + miss_mask
```

`block_size` attr 仍保留，因为 host tiling 需要用它推导
`replaceable_base/tail_base/fallback_slot/staging_base` 等 Hot Cache 布局
常量；它不再用于 kernel 内重复生成每请求 `tail_start`。

选择规则也收紧为：只有 `is_mtp && enable_turbo_* &&
enable_turbo_fused_*` 同时成立才走 Fused V2。关闭 Turbo 时，即使 Fused
开关保留默认值也不会越过普通 Batch fallback。non-MTP 继续走原
`dsa_offload_lookup_update`，其 tail 语义和正确性路径不变。

第三阶段建议拆成 3A 和 3B，先验证 mask 语义，再融合完整地址映射。

#### 3A：在算子内生成 `lookup_mask`

主 Turbo 算子增加以下紧凑输入：

```text
request_rows       [B], INT32
query_start_loc    [B+1], INT32
query_indices      [T,K], INT32
query_positions    [T], INT32
verify_starts      [B], INT32
tail_starts        [B], INT32
```

算子按 request 加载一次框架准备的 `verify_start/tail_start`，然后在读取每个
Top-K token 时直接计算：

```text
valid   = 0 <= token < index_capacity
history = valid && token < tail_start
```

预取 Turbo 只需接收 compact `tail_starts[B]` 并计算
`valid && token < tail_start`。预取只处理历史 KV，不能把 tail、当前 token 或
MTP staging token 写入预取 Lookup 状态。

这一阶段可以消除框架侧的范围比较、LogicalAnd、Bool→INT32 Cast、Contiguous，
以及完整 `[T,K]` `lookup_mask` 的 GM 写入和算子内二次读取。主流程后续的
tail/staging/fallback 和多层 `torch.where` 仍暂时保留。

#### 3B：融合分类、地址映射和 Gather mask

最终主 Turbo 算子应在同一次 query token 读取中完成：

```text
history: token < tail_start

tail:
    tail_start <= token < verify_start

staging（仅 MTP）：
    verify_start <= token <= current_position_of_this_query

invalid:
    token < 0
    或 token >= index_capacity
    或 token > current_position
```

随后在同一个 kernel 内完成：

- history index lookup/update；
- miss slot 分配；
- fallback 判断；
- resident/replaceable slot 到 Hot Cache offset 的转换；
- tail/staging offset 生成；
- 最终 SFA `mapped_indices` 生成；
- dense Gather `miss_mask` 生成。

主算子推荐最终输出：

```text
mapped_indices [T,K], INT32
miss_mask      [T,K], INT32
```

对 history miss，`mapped_indices` 同时就是 Gather destination slot；对 tail、
staging 和 invalid，`miss_mask=0`。这样不必额外输出 `slot_out` 和另一份
destination tensor。

预取 Turbo 只需输出：

```text
destination_slots [T,K], INT32
miss_mask         [T,K], INT32
```

完成 3B 后，框架侧可删除大部分 `valid/history/tail/staging/lookup/fallback`
mask、`active_misses` 和嵌套 `torch.where`。当前 `LookupPlan` 中的
`tail_mask`、`fallback_mask`、`staging_mask` 没有运行时消费者，可在新算子完成
对拍后删除，或仅保留 debug 路径。

### 算子侧性能依据

当前 Turbo kernel 按 request 切分，每个 request 使用一个 AIV block、256 个
SIMT thread。`K=2048` 时每线程大约处理 8 个 Top-K entry。将分类融合后，紧凑的
`verify_start/tail_start` 每个 request 只需读取或计算一次；逐 token 比较可以直接
复用已经读入寄存器的 token，不需要把中间 mask 落到 GM。

例如 `B=12`、每请求 6 个 MTP query、`T=72`、`K=2048` 时，一份 INT32
`lookup_mask` 即为：

```text
72 × 2048 × 4 = 589,824 bytes ≈ 576 KiB
```

主、预取每次 Lookup transaction 都会各自产生并消费一份，还不包括多个 Bool
临时 tensor 和 `torch.where` 的 GM 往返。因此融合的核心收益不是减少几次整数
比较，而是减少 kernel launch、完整 `[T,K]` 临时 tensor 和多轮 GM 读写。

从分层流水看，这不是卡间通信优化；算子也没有新增跨 request 的核间依赖。主要
优化点是 AIV 单核内在读取 Top-K 后立即分类和消费。现有“一 request 对应一
AIV block、`blockDim=min(req_num, aiv_count)`”的切分可以作为第一版基础，融合后
再根据算子 profiling 检查寄存器、UB 和单核时延是否膨胀。

### 实施与验收顺序

1. 已新增 `PackedAddressingMetadata`，保证 eager 每步一次、图内每次 replay 一次；
2. 已让 slot mapping、主 Lookup、预取 Lookup 和 HiCached 共享公共元数据；
3. 已把 Gather 行表、主 slot mapping、Decode block table 和当前 SFA group 的地址
   view 提前准备，并按 KV cache group 来源隔离；
4. 单独 profiling，确认 Lookup 前后元数据小算子数下降且端到端不回退；
5. 已接入新的 Turbo V2 算子并完成 3B 分类、地址映射和 Gather mask 融合；
6. 已按共享元数据边界精简 ABI：主算子增加 `tail_starts`，预取算子删除
   `query_positions/verify_starts`；
7. 已删除 `LookupPlan` 中没有运行时消费者的
   `tail_mask/fallback_mask/staging_mask`；
8. Fused 主流程和预取流程将算子输出的 INT32 `miss_mask` 直接交给 dense Gather；
   图模式不再为未消费的 eager miss plan 执行 `.bool()`，用于消除 Lookup 与 Gather
   之间的 `InplaceCopy/Cast`；
9. 需在 A5 重新编译后，用两份独立 LookupState 对拍全部状态副作用并复核整网
   profiling；最后再决定是否扩展 non-MTP fused 路径。

建议使用新的 V2 算子名，而不是直接修改旧算子 ABI，以避免旧 wheel、自定义算子
包或图缓存误加载。稳定后再删除旧接口。

正确性对拍必须覆盖：

- `index`、`slot_to_index`、`free_slots`、`free_head` 的完整状态；
- `mapped_indices`、destination slot 和 miss mask；
- history/tail/staging/fallback/invalid 边界；
- MTP rejected candidate；
- partial block、slot reuse 和 partial-to-full overwrite；
- eager 与 `FULL_DECODE_ONLY` 图重放；
- Prefetch 开关和 Turbo 开关组合。

性能验收不能只看 Turbo kernel 本身，还应比较：

- HiCached 到 Prefetch Lookup 的墙钟间隔；
- 主/预取 Lookup 前的小算子累计时延和数量；
- 两个 Turbo 算子是否因融合而膨胀；
- 主 Gather miss 数、时延和预取命中效果；
- 稳定 decode 步、闭合层和端到端平均时延。
