# LightningIndexerDecode 对齐测试

## 测试目的

当前仓库的 `LightningIndexerDecode` 对原版 LightningIndexer 的接口和 tiling 策略做了 decode-only 简化。本测试用于确认这些改动没有改变算子行为。

Reference 使用 NPU 环境中随 vLLM Ascend 安装的原版算子：

```python
torch.ops._C_ascend.npu_lightning_indexer
```

待测对象是当前仓库编译安装的算子：

```python
torch.ops.custom.npu_lightning_indexer_decode
```

测试重点覆盖 batch 内每个请求的 `actual_seq_lengths_key` 均不相同的场景，并使用 `TND + PA_BSND` Page Attention 模式。

## 为什么使用两个脚本

vLLM Ascend 原版算子和当前仓库算子分别来自不同的 CANN custom OPP 包。两者放在同一个 Python 进程中运行时，先加载的 `binary_info_config.json` 可能固定该进程的 OPP 注册环境，导致另一个 opType 无法找到。

因此测试分成两个独立进程：

1. `generate_vllm_reference.py` 只加载 vLLM Ascend，生成输入并运行原版算子，然后通过 `torch.save` 保存全部输入和 reference 输出。
2. `compare_current_decode.py` 只加载当前仓库扩展，读取完全相同的输入，运行 `LightningIndexerDecode` 并比较结果。

这种方式避免两个 custom OPP 在同一个进程中相互覆盖。

## 输入构造

`generate_vllm_reference.py` 按以下方式构造数据：

- `query`: `[B, 64, 128]`，随机 `bf16` 或 `fp16`。
- `key`: `[NUM_BLOCKS, BLOCK_SIZE, 1, 128]`，随机 paged key cache。
- `weights`: `[B, 64]`，随机值。
- `actual_seq_lengths_query`: `[1, 2, ..., B]`，表示每个请求各有一个 decode query token。
- `actual_seq_lengths_key`: 在 `min_seqlen..max_seqlen` 中生成 `B` 个互不相同的长度，并随机打乱 batch 顺序。
- `block_table`: 每个请求只使用与自身序列长度对应的有效 block；所有有效物理 block ID 全局唯一并随机排列。

原版算子固定使用：

```text
layout_query = TND
layout_key = PA_BSND
sparse_count = 2048
sparse_mode = 3
```

## 运行方法

先运行 vLLM Ascend 原版算子并保存数据：

```bash
python test_compare_li/generate_vllm_reference.py \
  --device 0 \
  --bs 24 \
  --min-seqlen 32768 \
  --max-seqlen 65536 \
  --dtype bf16 \
  --seed 1234
```

默认生成：

```text
test_compare_li/lightning_indexer_reference.pt
```

该文件包含 `query`、`key`、`weights`、序列长度、`block_table`、测试元数据和原版 top2048 输出。文件可能达到数百 MiB，不应提交到 Git。

然后在独立进程中运行当前算子并比较：

```bash
python test_compare_li/compare_current_decode.py --device 0
```

可使用 `--output` 和 `--input` 指定其他数据文件：

```bash
python test_compare_li/generate_vllm_reference.py --output /tmp/li_reference.pt
python test_compare_li/compare_current_decode.py --input /tmp/li_reference.pt
```

## 验收条件

脚本对每个 batch request 独立检查：

1. 两个算子的输出 shape 都是 `[B, 1, 2048]`，dtype 都是 `int32`。
2. 每行恰好包含 2048 个互不重复的 token index。
3. 第 `b` 行所有 index 都满足 `0 <= index < actual_seq_lengths_key[b]`，不会选中 padding token。
4. 将两个算子的每行输出分别排序后，2048 个 index 必须逐项相同，即 top2048 multiset 完全一致。

输出中的：

```text
topk_index_multiset_check=passed
lightning_indexer_decode_alignment_check=passed
```

表示行为对齐测试通过。

`ordered_equal_batches=B/B` 进一步表示所有 batch 的输出顺序也完全相同。顺序一致不是核心语义要求；核心验收条件是每行 top2048 multiset 一致。

## 时延对比

两个算子存在 OPP 注册冲突，因此时延也必须在两个独立 Python 进程中测试。两个脚本使用相同的输入 shape、随机种子、物理 block 分配和 NPU Event 计时方式，默认覆盖 `bs=24/48` 和 `seqlen=65536/131072`。

运行 vLLM Ascend 原版 `LightningIndexer`：

```bash
python test_compare_li/perf_indexer.py \
  --device 0 \
  --batch-sizes 24 48 \
  --seqlens 65536 131072 \
  --warmup 10 \
  --iters 200
```

运行当前仓库的 `LightningIndexerDecode`：

```bash
python test_compare_li/perf_indexer_decode.py \
  --device 0 \
  --batch-sizes 24 48 \
  --seqlens 65536 131072 \
  --warmup 10 \
  --iters 200
```

两份输出都会按 `bs`、`seqlen` 打印整数微秒 `avg_us`，可直接逐行对比。不要在同一个 Python 进程中导入或调用这两个算子。
