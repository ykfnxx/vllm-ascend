/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_common.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_HI_CACHED_COMMON_H
#define LIGHTNING_INDEXER_HI_CACHED_COMMON_H

namespace LICommon {
enum class LI_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

template <typename Q_T, typename K_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          LI_LAYOUT LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BSND, typename... Args>
struct LIType {
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    uint32_t stage1BufIdx = 0;
    uint32_t stage1LocalBlockNum = 0;
    uint32_t stage1BaseBlockIdx = 0;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    // This core owns a Local/Distributed partial-topK merge task.
    bool isLdMergeCore = false;
};

struct ConstInfo {
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t CUBE_BLOCK_ELEM_NUM = 16;
    static constexpr int INVALID_IDX = -1;

    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C1 = 0U;

    uint32_t mBaseSize = 1ULL;
    uint32_t mBaseSizeAlign = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    // Metadata row capacity from tiling bSize. TND decode may have fewer
    // active query rows in s1Size while actual_q/actual_k keep this capacity.
    uint64_t bSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;
    uint64_t hiBlockSize = 0ULL;
    uint64_t hiBlockNum = 0ULL;
    uint64_t kSeqSize = 0ULL;
    // TND: total query rows stored in query. BSND: query sequence length.
    uint64_t s1Size = 1ULL;
    uint32_t kCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t sink = 1U;
    uint32_t recent = 2U;
    uint32_t blockPoolingMode = 0U;
    uint32_t keyBlockNum = 0U;
    // Per-query selection mask workspace. Zero means the mask fits in the
    // unused tail of the Stage1 block-index row.
    uint32_t externalHiMaskWordNum = 0U;
    bool attenMaskFlag = false;

    uint32_t actualLenQDims = 0U;
    uint32_t actualLenDims = 0U;
    bool isAccumSeqS1 = false;
    bool isAccumSeqS2 = false;
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    // This split is assigned to a Local/Distributed partial-topK merge core.
    bool isLdMergeCore = false;
    bool isStage1 = false;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
} // namespace LICommon

#endif // LIGHTNING_INDEXER_HI_CACHED_COMMON_H
