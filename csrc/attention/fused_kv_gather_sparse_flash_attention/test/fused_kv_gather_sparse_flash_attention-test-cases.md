# FusedKVGatherSparseFlashAttention 用例设计文档

## 1. 算子标杆

NPU标杆为现有 `DsaKVGather -> npu_sparse_flash_attention`，使用完全相同
的route、Hot/Host非零KV/RoPE、query和sequence lengths。第二标杆为将
所有选中记录预先驻留HBM后的普通SFA。融合输出按现有SFA dtype阈值比较。

## 2. 用例说明

### 2.1 测试配置

```python
SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]

# (category, rows, seq_len, miss_rate, route_layout)
TEST_SHAPES = [
    ("Small", 1, 4096, 0.00, "interleaved"),
    ("Small", 4, 4096, 0.10, "interleaved"),
    ("Decode", 12, 16384, 0.10, "interleaved"),
    ("MTP", 48, 16384, 0.10, "interleaved"),
    ("Production", 48, 65536, 0.10, "interleaved"),
    ("Production", 48, 131072, 0.10, "interleaved"),
    ("Miss", 48, 65536, 0.00, "interleaved"),
    ("Miss", 48, 65536, 1.00, "interleaved"),
]

GENERAL_SHAPES = [
    ("Miss", 48, 65536, 0.01, "interleaved"),
    ("Miss", 48, 65536, 0.25, "interleaved"),
    ("Skew", 48, 65536, 0.10, "tail"),
    ("Rows", 47, 65536, 0.10, "interleaved"),
    ("Rows", 49, 65536, 0.10, "interleaved"),
    ("Live", 48, 65536, 0.10, "mixed_live_hot"),
    ("Invalid", 48, 65536, 0.10, "invalid_suffix"),
]

BOUNDARY_VALUES = [
    "first/last physical Hot block",
    "first/last physical Host block",
    "block offsets 0 and 127",
    "INVALID route suffix",
    "KV/RoPE physical-block mismatch must fail",
]
```

### 2.2 用例覆盖统计

| 类别 | Shape数量 | 边界组合 | dtype数量 | 总用例数 |
| --- | ---: | ---: | ---: | ---: |
| 常规和泛化 | 15 | - | 2 | 30 |
| 失败/边界 | 5 | 组合进常规shape | 2 | 10以上 |
| **总计** | **15** | **5** | **2** | **40以上** |

## 3. 使用说明

所有KV和RoPE使用确定性非零数据。Host源必须由
`torch_npu.empty_with_swapped_memory`创建。分别比较attention output、
softmax max/sum以及Graph replay；性能报告同时展示融合算子和现有两算子链。
