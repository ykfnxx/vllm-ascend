#ifndef PREFETCH_QLI_FUSION_TILING_H
#define PREFETCH_QLI_FUSION_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PrefetchQliFusionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tokenNum);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, singleM);   // 每核 M 行段（moe 范式，多核切分）
TILING_DATA_FIELD_DEF(uint32_t, mCore);     // M 维跨核切分数
TILING_DATA_FIELD_DEF(uint32_t, nCore);     // N 维跨核切分数
TILING_DATA_FIELD_DEF(uint32_t, baseN);     // mm 逐 N-block 宽度（SetFixSplit baseN）
TILING_DATA_FIELD_DEF(uint32_t, mChunk);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, qLoraRank);
TILING_DATA_FIELD_DEF(uint32_t, nQkv);
TILING_DATA_FIELD_DEF(uint32_t, nQb);
TILING_DATA_FIELD_DEF(uint32_t, nHead);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, ropeDim);
TILING_DATA_FIELD_DEF(float, alpha);
TILING_DATA_FIELD_DEF(float, beta);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, invQloraRank);
TILING_DATA_FIELD_DEF(float, invHiddenSize);
// rank-aware alpha/beta（GLM-5.2 fused）：0=标量模式，1=per-row rank-aware 向量模式
TILING_DATA_FIELD_DEF(uint32_t, alphaBetaMode);
TILING_DATA_FIELD_DEF(uint32_t, sourceRowsBeforeGather);  // R = row_rank = min(row/R, N-1)
TILING_DATA_FIELD_DEF(uint32_t, alphaVecLen);             // N（alpha_vec/beta_vec 长度）
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling1);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling2);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling3);  // mm3: wk_weights_proj（BF16，N=n_head）
TILING_DATA_FIELD_DEF(uint64_t, wsXqOffset);
TILING_DATA_FIELD_DEF(uint64_t, wsStokOffset);
TILING_DATA_FIELD_DEF(uint64_t, wsQkvOutOffset);
TILING_DATA_FIELD_DEF(uint64_t, wsXcqOffset);
TILING_DATA_FIELD_DEF(uint64_t, wsStok2Offset);
TILING_DATA_FIELD_DEF(uint64_t, wsQfOffset);
TILING_DATA_FIELD_DEF(uint64_t, wsPredOffset);   // predicted_hidden bf16 [T, hidden]（mm3 的 A 输入）
TILING_DATA_FIELD_DEF(uint64_t, wsWeightsOffset);  // mm3 C 输出 bf16 [T, n_head]（AIV 再拷到输出 GM）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PrefetchQliFusion, PrefetchQliFusionTilingData)
}  // namespace optiling

#endif  // PREFETCH_QLI_FUSION_TILING_H
