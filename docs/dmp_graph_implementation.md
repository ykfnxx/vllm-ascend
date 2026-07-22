# DMP Graph 修改说明

本次复用 vLLM-Ascend 现有的 `FULL_DECODE_ONLY` ACL Graph，只修改 Python
层，使 DMP 双流路径在 Graph capture 时生效。

## 修改位置

### `vllm_ascend/ascend_forward_context.py`

- `set_ascend_forward_context()`：增加 `dmp_context` 参数并写入
  `forward_context`。

### `vllm_ascend/worker/model_runner_v1.py`

- `NPUModelRunner.__init__()`：缓存 DMP Stream 和 Graph Context。
- `_is_dmp_eligible()`：限制当前只处理规则的 DecodeOnly batch。
- `_maybe_create_dmp_slices()`：切分 A/B microbatch 及 attention metadata。
- `_get_or_create_dmp_graph_context()`：按 batch shape 创建并缓存 DMPContext。
- `_dummy_run()`：Graph capture 时提前进入 DMP forward。
- `execute_model()`：Graph replay 时复用 capture 阶段的 DMPContext。

```python
class NPUModelRunner(...):
    def _get_or_create_dmp_graph_context(...):
        # 创建并缓存 DMPContext、Stream 和 Event。
        ...

    def _dummy_run(...):
        # capture 时把 DMPContext 传入模型 forward。
        ...

    def execute_model(...):
        # replay 时取得同一个 DMPContext。
        ...
```

### `vllm_ascend/worker/dmp_context.py`

- `DMPSlice`：记录 A/B microbatch 的 token 范围。
- `DMPContext.get_event()`：缓存跨流 Event。
- `prepare_graph_events()`：在 capture 前创建 Event。
- `enter_microbatch()`：切换当前 microbatch 的 metadata。
- `slice_hidden_states()` / `merge_hidden_states()`：切分和合并输出。

### `vllm_ascend/patch/worker/patch_deepseek_mtp.py`

- `forward_indexer_only()`：单独执行 Indexer。
- `forward_sparse_attn_only()`：单独执行 Sparse Attention。
- `forward_mlp_only()`：单独执行 MLP。
- `forward_mlp_two_mb_once()`：当前合并执行 A/B MLP。
- `dmp_forward()`：实现 S0/S1 调度及 Event 依赖。
- 设置 `IGNORE_COMPILE_KEY`：跳过 TorchDynamo 对双流 Python 控制流的追踪。

```python
def dmp_forward(...):
    # S0：ACL Graph 主 Stream。
    # S1：DMP Stream。
    ...

    # two：A/B Indexer 和计算放 S0，KVSelect/KVGather 放 S1。
    # four：A Indexer 放 S1，Select/Gather 各自使用独立 Stream。
    ...

    # capture 时跳过动态 KV classify/load 和 Host IO。
    ...
```

### `vllm_ascend/attention/sfa_v1.py`

- `AscendSFAImpl.forward_indexer_only()`：拆出 Indexer 阶段。
- `AscendSFAImpl.forward_sparse_attn_only()`：拆出 Sparse Attention 阶段。

### `vllm_ascend/ops/mla.py`

- `AscendMLA.forward_indexer_only()` / `forward_sparse_attn_only()`：调用拆分后的
  MLA 路径。
- `mla_forward_indexer_only()` / `mla_forward_sparse_attn_only()`：增加对应的
  Python custom-op 包装和注册。

### 其他文件

- `vllm_ascend/ops/fused_moe/fused_moe.py`
  - `_encode_layer_name()`：兼容 DMP forward context。
- `vllm_ascend/kv_offload/asu_npu.py`
  - 增加 KV load 接口和当前使用的占位实现。
- `vllm_ascend/kv_offload/block_location.py`
  - 增加 KV block 位置记录和分类。
- `vllm_ascend/kv_offload/kv_loader.py`
  - 增加异步加载及 Event 管理。
- `vllm_ascend/envs.py`
  - 增加 DMP 开关和 `VLLM_ASCEND_DMP_STREAM_MODE` 流拓扑参数。
- `tests/ut/worker/test_model_runner_v1.py`
  - 增加 DMP Graph 条件和 Context 复用测试。

当前完成的是同层双流 Graph；完整跨层 DMP 流水尚未加入。

## Indexer+KVSelect 融合（不加载 KV）

设置 `VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=1` 后，DMP decode
使用 `npu_lightning_indexer_decode_update` 一次完成 top-k 和 KVSelect
索引更新。当前只把 `topk_indices` 交给原 full-cache SFA；`topk_slots` 和
`miss_count` 暂不接 KVGather。因此该模式不依赖 HIXL 或第二张 NPU。

该开关与 `VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION` 互斥；关闭时完全走原
Indexer 路径。

## Dual-Attention 可选路径

默认仍走原 DMP 路径。设置
`VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=1` 后，每层会执行：

```text
Indexer -> KVSelect -> hit SFA
                    -> KVGather -> miss SFA -> DaAttentionMerge
```

- `vllm_ascend/kv_offload/dual_attention.py`：管理 selection cache、固定 shape
  workspace，以及 KVSelect/KVGather/分段 SFA/merge 调用。
- `vllm_ascend/patch/worker/patch_deepseek_mtp.py`：把 Select、Gather 和 hit
  SFA 接入 DMP A/B microbatch 的 Stream/Event 调度。
- `vllm_ascend/attention/sfa_v1.py`：在可选路径中执行 miss SFA 和 merge，
  后续 `v_up_proj`、`o_proj` 与原路径一致。
- `vllm_ascend/worker/model_runner_v1.py`：按最大 microbatch 容量创建共享
  selection pool，多个 Graph batch shape 使用同一 pool 的前缀 view。

该路径需要 `pip-cache` 仓库 `feat/dual-attention` 分支（本次基于
`cc916d0`）中的 KVSelect、KVGather、SparseFlashAttention 和
DaAttentionMerge 算子。容器内编译安装：

```bash
cd /root/dmp/pip-cache
OPP_OP_NAME='gather_selection_kv_cache;kv_select;kv_gather;dmp_sparse_flash_attention;da_attention_merge' \
  bash op/scripts/build_opp.sh
CUSTOM_OPS_GATHER_ONLY= CUSTOM_OPS_SFA_ONLY= \
  bash op/scripts/build_torch_ops.sh
```

启动前加载 OPP 环境并开启两个开关：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/cann-8.5.1/opp/vendors/customize/bin/set_env.bash
set -u
export VLLM_ASCEND_ENABLE_DMP=1
export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=1
export VLLM_ASCEND_DMP_STREAM_MODE=two  # two 或 four
```

先运行 `pip-cache` 自带的算子测试和
`experiments/gather_select_kvcache/test_npu_segmented_sfa.py`，再进行 vLLM
 eager、ACL Graph capture/replay 和 MindStudio profiling 对比。尤其要覆盖全 miss、
部分 hit、全 hit，以及请求槽复用场景。

## HIXL KV 后端（实验）

默认 `VLLM_ASCEND_DMP_KV_BACKEND=local`，继续使用已跑通的
KVSelect/KVGather。只有显式设为 `hixl` 时，才用 `DSA_offload_ops` 的
IndexerUpdate + HIXL submit/PollMany 替换 KV classify/load；DMP 两流/四流调度、
分段 SFA 和 merge 不变。

```bash
export VLLM_ASCEND_DMP_KV_BACKEND=hixl
export VLLM_ASCEND_DMP_HIXL_CONFIG=/workspace/scripts/dmp_hixl_config.json
```

- `vllm_ascend/kv_offload/hixl_dual_attention.py`：在 Graph capture 前建立
  HCOMM、各层 cache/state/session，并为算子提供固定地址输出。
- HIXL 当前只允许 `VLLM_ASCEND_DMP_STREAM_MODE=two`；四流会让 A/B 在共享
  HCOMM workspace 上并发，已直接拒绝。local 后端仍可使用 two 或 four。
- `DSA_offload_ops` 使用 `hixl-li-ready-smoke` 分支；本次只扩展调用方提供
  output tensor，不改 HIXL kernel。
- 该分支依赖 CANN 9.1 和 `libcann_hixl_kernel.json`。只有 CANN 8.5.1 时会
  直接预检失败，不覆盖已有算子碰运气。
- 第二张 NPU 可模拟 SSU，但仓库 smoke 会在其 HBM 中生成合成 KV。它只能验证
  双卡通信、算子和 Graph replay，生成文本没有正确性意义。
- 64 条、单条 128K、全层合成 KV 远超单卡 HBM，不能用 NPU emulator 跑该
  配置。先用包内 `validate_dmp_hixl_model_smoke.sh` 的缩小参数验证；正式 128K
  需要真实 SSU/ASU 存储和真实 prefill KV 写入链路。
# Lookup/Maintain mode

`VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=1` keeps the original Lightning
Indexer and uses a 144K token index with a 10K miss-staging pool. Lookup emits
original token positions for hits and 10K staging slots for misses directly as
`int32`, so there is no post-Lookup Cast. Scheme 4 uses one miss-only KVGather
invocation per microbatch; scheme 2 keeps its original Dual-Attention KVGather
path. mb0/mb1 keep independent Lookup, KVGather, and Maintain calls, and their
Indexer results are combined before segmented SFA. Hit SFA reads the existing
full vLLM KV cache and is enqueued before S0 waits for S1. Only miss SFA reads
the staging pool and waits for KVGather; merge and attention update follow it.
The graph topology is S0=LI0/Lookup0/LI1/Lookup1/combined-preattn/hit-SFA/wait/
miss-SFA/merge/update/MLP, S1=KVGather0/KVGather1, and
S2=Maintain0/Maintain1. The profiling workload fixes 300 misses and 300
maintain evictions per request.
