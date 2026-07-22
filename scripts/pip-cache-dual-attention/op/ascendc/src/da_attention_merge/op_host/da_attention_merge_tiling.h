#ifndef DA_ATTENTION_MERGE_TILING_H
#define DA_ATTENTION_MERGE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t PREV_ATTENTION_OUT_INPUT_INDEX = 0;
constexpr uint32_t PREV_SOFTMAX_MAX_INPUT_INDEX = 1;
constexpr uint32_t PREV_SOFTMAX_SUM_INPUT_INDEX = 2;
constexpr uint32_t CUR_ATTENTION_OUT_INPUT_INDEX = 3;
constexpr uint32_t CUR_SOFTMAX_MAX_INPUT_INDEX = 4;
constexpr uint32_t CUR_SOFTMAX_SUM_INPUT_INDEX = 5;
constexpr uint32_t ATTENTION_OUT_OUTPUT_INDEX = 0;

constexpr uint32_t DA_MERGE_TILING_KEY_FP16 = 1;
constexpr uint32_t DA_MERGE_TILING_KEY_BF16 = 2;

BEGIN_TILING_DATA_DEF(DaAttentionMergeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, headNum)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, headDimAlign)
TILING_DATA_FIELD_DEF(uint32_t, totalRows)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore)
TILING_DATA_FIELD_DEF(uint32_t, rowBlock)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(DaAttentionMerge, DaAttentionMergeTilingData)

template <typename T> inline T DaMergeAlign(T num, T rnd)
{
    return rnd == 0 ? 0 : ((num + rnd - 1) / rnd * rnd);
}

} // namespace optiling

#endif // DA_ATTENTION_MERGE_TILING_H
