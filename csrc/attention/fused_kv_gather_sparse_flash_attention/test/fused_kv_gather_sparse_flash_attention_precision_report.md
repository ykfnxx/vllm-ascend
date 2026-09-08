# FusedKVGatherSparseFlashAttention 精度验证报告

生成时间：2026-08-30T14:25:14

Golden 为同一非零 Hot/Host 输入下的 `DsaKVGather + SparseFlashAttention`。
Host KV/RoPE 使用 swapped memory；判定标准为 MERE < dtype threshold 且
MARE < 10 × threshold（FP16=2^-10，BF16=2^-7），并且融合算子写入
install staging 的 Host-miss KV/RoPE 必须与 golden payload 逐元素一致。

## 总览

| 指标 | 值 |
| ---- | --: |
| 总用例 | 30 |
| 通过 | 30 |
| 失败 | 0 |
| 通过率 | 100.0% |

## 结果

| Case | Category | DType | MERE | MARE | MaxAbsErr | StagingErr | Cosine | Result |
| ---- | -------- | ----- | ---: | ---: | --------: | ---------: | -----: | ------ |
| BS=1,MTP=0,S=4096,miss=0.00 | Small | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999995232 | PASS |
| BS=1,MTP=0,S=4096,miss=0.00 | Small | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.0000009537 | PASS |
| BS=1,MTP=0,S=4096,miss=0.10 | Small | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999995232 | PASS |
| BS=1,MTP=0,S=4096,miss=0.10 | Small | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.0000011921 | PASS |
| BS=4,MTP=0,S=4096,miss=0.10 | Small | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999991655 | PASS |
| BS=4,MTP=0,S=4096,miss=0.10 | Small | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.0000040531 | PASS |
| BS=12,MTP=0,S=16384,miss=0.10 | Decode | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999973178 | PASS |
| BS=12,MTP=0,S=16384,miss=0.10 | Decode | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.0000054836 | PASS |
| BS=4,MTP=3,S=16384,miss=0.10 | MTP | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999971390 | PASS |
| BS=4,MTP=3,S=16384,miss=0.10 | MTP | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.0000059605 | PASS |
| BS=12,MTP=3,S=16384,miss=0.10 | MTP | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999948740 | PASS |
| BS=12,MTP=3,S=16384,miss=0.10 | MTP | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999896884 | PASS |
| BS=12,MTP=3,S=65536,miss=0.10 | Production | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999947548 | PASS |
| BS=12,MTP=3,S=65536,miss=0.10 | Production | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999880195 | PASS |
| BS=12,MTP=3,S=131072,miss=0.10 | Production | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999939799 | PASS |
| BS=12,MTP=3,S=131072,miss=0.10 | Production | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999892712 | PASS |
| BS=12,MTP=3,S=65536,miss=0.00 | AllHot | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999943972 | PASS |
| BS=12,MTP=3,S=65536,miss=0.00 | AllHot | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999857545 | PASS |
| BS=12,MTP=3,S=65536,miss=1.00 | AllHost | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.0000003576 | PASS |
| BS=12,MTP=3,S=65536,miss=1.00 | AllHost | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.0000003576 | PASS |
| BS=12,MTP=3,S=65536,miss=0.01 | MissRate | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999938011 | PASS |
| BS=12,MTP=3,S=65536,miss=0.01 | MissRate | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999864101 | PASS |
| BS=12,MTP=3,S=65536,miss=0.25 | MissRate | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999942780 | PASS |
| BS=12,MTP=3,S=65536,miss=0.25 | MissRate | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999846816 | PASS |
| BS=47,MTP=0,S=65536,miss=0.10 | OddRows | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999951124 | PASS |
| BS=47,MTP=0,S=65536,miss=0.10 | OddRows | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999895096 | PASS |
| BS=49,MTP=0,S=65536,miss=0.10 | OddRows | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999956489 | PASS |
| BS=49,MTP=0,S=65536,miss=0.10 | OddRows | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999889731 | PASS |
| BS=12,MTP=3,S=65536,miss=0.50 | MissRate | float16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999888539 | PASS |
| BS=12,MTP=3,S=65536,miss=0.50 | MissRate | bfloat16 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.9999853373 | PASS |

## 关键发现

- 30 个精度用例中 30 个通过，覆盖 FP16/BF16、BS 1–49、MTP 0/3。
- 覆盖 0%、1%、10%、25%、50%、100% Host miss，验证 Hot/Host 混合路由。
- 覆盖 4K、16K、64K、128K 序列，包含 GLM-5.2 的 BS=12/MTP=3/64K 生产门禁。
- 对每个 Host miss 同时检查 install staging，防止注意力正确但后续 Hot cache 安装错误。
