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
 * \file lightning_indexer_kernel.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_HI_CACHED_ARCH35_KERNEL_H
#define LIGHTNING_INDEXER_HI_CACHED_ARCH35_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../lightning_indexer_common.h"
#include "lightning_indexer_service_vector.h"
#include "lightning_indexer_service_cube.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
using namespace matmul;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;
    uint32_t s2LoopEnd = 0U;
    uint32_t actS1Size = 1ULL;
    uint32_t actS2Size = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;
    uint32_t actMBaseSize = 0U;
    uint32_t mBasicSizeTail = 0U;
    uint32_t s2BasicSizeTail = 0U;
};

template <typename LIT>
class LIPreload {
public:
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *keyMean,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                const LIHiCachedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIT::keyLayout;

    using MM1_OUT_T = float;

    LIMatmul<LIT> matmulService;
    LIVector<LIT> vectorService;

    // DAV_3510 uses the same M/N tile geometry as the native LI arch35
    // pipeline. HI Stage1 interprets S2 in physical blocks, while Stage2
    // interprets it in tokens; both remain valid at 128 because the PA block
    // size is also 128.
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t BLOCK_CUBE_SIZE = 16;
    static constexpr int64_t PARTIAL_TOPK_MERGE_PREFETCH_LEN = 2;
    // Ping-pong slots; keep in sync with Host workspace sizing.
    static constexpr uint32_t WORKSPACE_BUFFER_NUM = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<K_T> weightsGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    // workspace
    GlobalTensor<MM1_OUT_T> qkWorkspaceGm;
    GlobalTensor<float> partialTopkGm;
    GlobalTensor<int64_t> partialTopkMetadataGm;
    GlobalTensor<int32_t> blockIndiceGm;
    GlobalTensor<int32_t> externalHiMaskGm;
    GlobalTensor<K_T> keyMeanGm;

    // aic、aiv kernel info
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t stage1UsedCoreNum = 0U;

    LICommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    LICommon::SplitCoreInfo splitCoreInfo{};
    LICommon::SplitCoreInfo splitCoreInfoStage2{};
    bool decodeStage2HiFastPathEligible = false;
    // Uniform, unpadded requests can map query rows without scanning metadata.
    // Zero selects the general mapping for mixed lengths or metadata padding.
    uint32_t decodeQueryTokensPerRequest = 0U;
    uint32_t decodeStage2ChunkSize = S2_BASE_SIZE;
    bool shareMtpStage2Queries = false;
    bool stage2QkToUb = false;
    bool stage1SingleScoreTile = false;
    bool stage1PartialTopkMergeSkippable = false;
    bool stage2PartialTopkMergeSkippable = false;
    bool skipStage1ForHiFullCoverage = false;
    // Number of Stage2 rows that have real work after expanding metadata rows
    // by N2 and gS1 tiles. This is not the metadata batch size.
    uint32_t stage2WorkRowNum = 0U;

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const LIHiCachedTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, LICommon::SplitCoreInfo &info);
    __aicore__ inline void SplitCoreStage1(uint32_t curCoreIdx, uint32_t &coreNum, LICommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t GetGS1MaxVisibleS2(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t GetStage1ScoreTileNum(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    __aicore__ inline uint32_t GetTotalStage1BaseBlockNum();
    __aicore__ inline bool IsHiFullCoverage(uint32_t visibleTokenNum);
    __aicore__ inline bool NeedsStage1ForGS1(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t CountStage2WorkRows();
    __aicore__ inline bool CalcStage1UsesSingleScoreTile();
    __aicore__ inline bool CalcCanSkipStage1PartialTopkMerge();
    __aicore__ inline void InitSplitHints();
    __aicore__ inline bool Stage1UsesSingleScoreTile();
    // Shared arch32/arch35 RunInfo calls cross-core partial TopK merge "LdMerge".
    __aicore__ inline bool CanSkipStage1PartialTopkMerge();
    __aicore__ inline bool CanSkipStage2PartialTopkMerge();
    __aicore__ inline bool InitDecodeStage2FastPathHints();
    __aicore__ inline uint32_t GetDecodeStage2HiFastPathRowNum();
    __aicore__ inline uint32_t GetDecodeStage2GroupNum(uint32_t queryRowNum);
    __aicore__ inline uint32_t MapDecodeStage2HiFastPathRowToBN2(uint32_t rowOrdinal, uint32_t &queryIdx);
    // ================================Process functions================================
    // Stage1 coarse filtering: AIC computes q * mean(k), AIV applies weights
    // and emits the selected HI block list/mask.
    __aicore__ inline void RunStage1();
    // Stage2 token scoring: score original key tokens under the HI block
    // restriction and produce token-level topk.
    __aicore__ inline void RunStage2();
    // Decode/MTP fast path that scores only the fixed HI block list instead
    // of replaying the full S2 tile schedule.
    __aicore__ inline bool RunDecodeStage2HiFastPath();
    __aicore__ inline void RunMtpSharedStage2();
    // One Stage1 S2 tile: q * mean(k), followed by block-score reduction.
    __aicore__ inline void RunStage1Tile(uint32_t loop, uint64_t s2LoopIdx, LICommon::RunInfo &runInfo);
    // One Stage2 S2 tile: dense if HI covers all visible tokens, otherwise
    // packed/masked Stage2 scoring over selected HI block ranges.
    __aicore__ inline void RunStage2Tile(uint32_t loop, uint64_t s2LoopIdx, LICommon::RunInfo &runInfo);
    // Merge Stage1 partial HI block topm from multiple S2 tiles/cores.
    __aicore__ inline void MergeStage1Blocks();
    // Merge Stage2 token topk partials back to the final output tensor.
    __aicore__ inline void MergeStage2Tokens();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline uint32_t GetComputeBatchSize();
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LICommon::RunInfo &runInfo);
    __aicore__ inline uint32_t GetGS1TileMaxVisibleS2(const LICommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::InitTilingData(const LIHiCachedTilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.bSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.s1Size = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.hiBlockSize = tilingData->hiBlockSize;
    constInfo.hiBlockNum = tilingData->hiBlockNum;
    constInfo.sink = tilingData->sink;
    constInfo.recent = tilingData->recent;
    constInfo.blockPoolingMode = tilingData->blockPoolingMode;
    constInfo.keyBlockNum = tilingData->keyBlockNum;
    constInfo.externalHiMaskWordNum = tilingData->externalHiMaskWordNum;
    if (LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    constInfo.s2BaseSize = S2_BASE_SIZE;
    // Match the native DAV_3510 LI query tile: M is 256 for ratio 64 and 128
    // for ratio 32. This reduces the arch35 GM/UB working set without changing
    // the two-stage HI selection semantics.
    constInfo.s1BaseSize = S1_BASE_SIZE;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
    constInfo.mBaseSizeAlign = LICommon::Align(constInfo.mBaseSize, BLOCK_CUBE_SIZE);
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                        __gm__ uint8_t *actualSeqLengths)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.bSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.bSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengths, constInfo.actualLenDims);
    }
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                           GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                           uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.s1Size);
    actS2Size =
        GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize);
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetComputeBatchSize()
{
    if constexpr (LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention) {
        // FULL_DECODE_ONLY capture commonly pads actual_q/actual_k to
        // bSize while query stores only s1Size valid TND rows.
        // actual_q repeats the last cumulative value, actual_k pads 0, and
        // padded rows are appended after the valid prefix. Scheduling only the
        // compute prefix preserves original LI zero-length guards as fallback
        // while avoiding padded metadata work.
        if (constInfo.actualLenQDims != 0 && constInfo.actualLenDims != 0 &&
            constInfo.s1Size > 0 && constInfo.s1Size < constInfo.bSize) {
            return static_cast<uint32_t>(constInfo.s1Size);
        }
    }
    return static_cast<uint32_t>(constInfo.bSize);
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                   uint32_t actS2Size)
{
    if (actS2Size == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = s1Offset + validS2LenBase + constInfo.s1BaseSize;
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2Size));
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetGS1MaxVisibleS2(uint32_t gS1Idx, uint32_t actS1Size,
                                                              uint32_t actS2Size)
{
    if (!constInfo.attenMaskFlag) {
        return actS2Size;
    }
    uint32_t s1BaseIdx = gS1Idx * constInfo.s1BaseSize;
    if (s1BaseIdx >= actS1Size) {
        return 0;
    }
    uint32_t s1LastIdx = Min(s1BaseIdx + constInfo.s1BaseSize, actS1Size) - 1;
    int32_t visibleS2 = static_cast<int32_t>(actS2Size) -
                        (static_cast<int32_t>(actS1Size) - static_cast<int32_t>(s1LastIdx)) + 1;
    visibleS2 = Min(visibleS2, static_cast<int32_t>(actS2Size));
    visibleS2 = Max(visibleS2, 0);
    return static_cast<uint32_t>(visibleS2);
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetStage1ScoreTileNum(uint32_t gS1Idx, uint32_t actS1Size,
                                                                 uint32_t actS2Size)
{
    if (!NeedsStage1ForGS1(gS1Idx, actS1Size, actS2Size)) {
        return 0;
    }
    uint32_t visibleS2 = GetGS1MaxVisibleS2(gS1Idx, actS1Size, actS2Size);
    uint32_t visibleHiBlockNum =
        LICommon::CeilDiv(visibleS2, static_cast<uint32_t>(constInfo.hiBlockSize));
    return LICommon::CeilDiv(visibleHiBlockNum, constInfo.s2BaseSize);
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    uint32_t computeBatchSize = GetComputeBatchSize();
    for (uint32_t bIdx = 0; bIdx < computeBatchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2Size);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::IsHiFullCoverage(uint32_t visibleTokenNum)
{
    uint64_t hiTokenCapacity =
        static_cast<uint64_t>(constInfo.hiBlockNum) * static_cast<uint64_t>(constInfo.hiBlockSize);
    return static_cast<uint64_t>(visibleTokenNum) <= hiTokenCapacity;
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::NeedsStage1ForGS1(uint32_t gS1Idx, uint32_t actS1Size, uint32_t actS2Size)
{
    return !IsHiFullCoverage(GetGS1MaxVisibleS2(gS1Idx, actS1Size, actS2Size));
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::CountStage2WorkRows()
{
    if constexpr (!(LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention)) {
        return 0;
    }
    uint32_t workRowNum = 0;
    uint32_t actS1Size = 0;
    uint32_t actS2Size = 0;
    uint32_t s1GBaseNum = 0;
    uint32_t totalRowNum = GetComputeBatchSize() * static_cast<uint32_t>(constInfo.kHeadNum);
    for (uint32_t bN2Idx = 0; bN2Idx < totalRowNum; ++bN2Idx) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; ++gS1Idx) {
            uint32_t s2BaseNum = constInfo.attenMaskFlag ?
                GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size) :
                CeilDiv(actS2Size, constInfo.s2BaseSize);
            if (s2BaseNum > 0) {
                ++workRowNum;
            }
        }
    }
    return workRowNum;
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::CalcStage1UsesSingleScoreTile()
{
    if constexpr (!(LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention)) {
        return false;
    }
    uint32_t actS1Size = 0;
    uint32_t actS2Size = 0;
    uint32_t maxActS2Size = 0;
    uint32_t computeBatchSize = GetComputeBatchSize();
    for (uint32_t bIdx = 0; bIdx < computeBatchSize; ++bIdx) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        maxActS2Size = LICommon::Max(maxActS2Size, actS2Size);
    }
    uint32_t maxVisibleHiBlockNum =
        LICommon::CeilDiv(maxActS2Size, static_cast<uint32_t>(constInfo.hiBlockSize));
    return maxVisibleHiBlockNum <= constInfo.s2BaseSize;
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::InitSplitHints()
{
    if constexpr (LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention) {
        stage2WorkRowNum = CountStage2WorkRows();
        stage1SingleScoreTile = CalcStage1UsesSingleScoreTile();
        stage1PartialTopkMergeSkippable = CalcCanSkipStage1PartialTopkMerge();
        decodeStage2HiFastPathEligible = InitDecodeStage2FastPathHints();
        // Row-only Stage2 scheduling needs enough independent request rows to
        // keep all cores balanced. Multiple gS1 tiles from one long prefill
        // request are not equivalent: treating them as independent rows makes
        // medium batches such as 16 consistently slower.
        uint32_t computeBatchSize = GetComputeBatchSize();
        stage2PartialTopkMergeSkippable = constInfo.s1Size == computeBatchSize ||
            computeBatchSize >= usedCoreNum;
        skipStage1ForHiFullCoverage = true;
        uint32_t actS1Size = 0;
        uint32_t actS2Size = 0;
        for (uint32_t bIdx = 0; bIdx < computeBatchSize; ++bIdx) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            uint32_t gS1SplitNum = LICommon::CeilDiv(
                actS1Size * static_cast<uint32_t>(constInfo.gSize), constInfo.mBaseSize);
            for (uint32_t gS1Idx = 0; gS1Idx < gS1SplitNum; ++gS1Idx) {
                if (!IsHiFullCoverage(GetGS1MaxVisibleS2(gS1Idx, actS1Size, actS2Size))) {
                    skipStage1ForHiFullCoverage = false;
                    return;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetTotalStage1BaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    uint32_t computeBatchSize = GetComputeBatchSize();
    for (uint32_t bIdx = 0; bIdx < computeBatchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            if (NeedsStage1ForGS1(0, actS1Size, actS2Size)) {
                s2BaseNum = GetStage1ScoreTileNum(0, actS1Size, actS2Size);
                totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            }
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            if (!NeedsStage1ForGS1(s1gIdx, actS1Size, actS2Size)) {
                continue;
            }
            s2BaseNum = GetStage1ScoreTileNum(s1gIdx, actS1Size, actS2Size);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

template <typename LIT>
__aicore__ void inline LIPreload<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, LICommon::SplitCoreInfo &info)
{
    if constexpr (LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention) {
        uint32_t workRowNum = stage2WorkRowNum;
        if (workRowNum == 0) {
            coreNum = 0;
            return;
        }
        if (stage2PartialTopkMergeSkippable) {
            uint32_t actS1Size = 0;
            uint32_t actS2Size = 0;
            uint32_t s1GBaseNum = 0;
            uint32_t totalRowNum = GetComputeBatchSize() * static_cast<uint32_t>(constInfo.kHeadNum);
            uint32_t rowNumPerCore = workRowNum / coreNum;
            uint32_t deal1MoreRowCoreNum = workRowNum % coreNum;
            coreNum = rowNumPerCore == 0 ? deal1MoreRowCoreNum : coreNum;
            if (curCoreIdx >= coreNum) {
                return;
            }
            uint32_t startRowOrdinal = curCoreIdx * rowNumPerCore + Min(curCoreIdx, deal1MoreRowCoreNum);
            uint32_t curCoreRowNum = rowNumPerCore + (curCoreIdx < deal1MoreRowCoreNum ? 1U : 0U);
            uint32_t endRowOrdinal = startRowOrdinal + curCoreRowNum - 1;
            uint32_t rowOrdinal = 0;
            bool foundStart = false;
            for (uint32_t bN2Idx = 0; bN2Idx < totalRowNum; ++bN2Idx) {
                uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
                if (bN2Idx % constInfo.kHeadNum == 0) {
                    GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
                    s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
                }
                for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; ++gS1Idx) {
                    uint32_t s2BaseNum = constInfo.attenMaskFlag ?
                        GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size) :
                        CeilDiv(actS2Size, constInfo.s2BaseSize);
                    if (s2BaseNum == 0) {
                        continue;
                    }
                    if (rowOrdinal == startRowOrdinal) {
                        info.bN2Start = bN2Idx;
                        info.gS1Start = gS1Idx;
                        info.s2Start = 0;
                        foundStart = true;
                    }
                    if (rowOrdinal == endRowOrdinal) {
                        info.bN2End = bN2Idx;
                        info.gS1End = gS1Idx;
                        info.s2End = s2BaseNum - 1;
                        info.isLdMergeCore = false;
                        return;
                    }
                    ++rowOrdinal;
                }
            }
            if (!foundStart) {
                coreNum = 0;
            }
            return;
        }
    }
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    uint32_t totalRowNum = GetComputeBatchSize() * static_cast<uint32_t>(constInfo.kHeadNum);
    for (uint32_t bN2Idx = 0; bN2Idx < totalRowNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            for (uint32_t s2Idx = 0; s2Idx < s2BaseNum;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2BaseNum - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

                    if (coreIdx == curCoreIdx) {
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLdMergeCore = true;
                        }
                        if (coreIdx == coreNum - 1 && info.bN2End != totalRowNum - 1) {
                            info.bN2End = totalRowNum - 1;
                            info.gS1End = 0;
                            info.s2End = 0;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ void inline LIPreload<LIT>::SplitCoreStage1(uint32_t curCoreIdx, uint32_t &coreNum,
                                                       LICommon::SplitCoreInfo &info)
{
    info.isStage1 = true;
    if constexpr (LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention) {
        if (stage1PartialTopkMergeSkippable) {
            uint32_t workRowNum = 0;
            uint32_t actS1Size = 0;
            uint32_t actS2Size = 0;
            uint32_t s1GBaseNum = 0;
            uint32_t totalRowNum = GetComputeBatchSize() * static_cast<uint32_t>(constInfo.kHeadNum);
            for (uint32_t bN2Idx = 0; bN2Idx < totalRowNum; ++bN2Idx) {
                uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
                if (bN2Idx % constInfo.kHeadNum == 0) {
                    GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
                    s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
                }
                for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; ++gS1Idx) {
                    if (NeedsStage1ForGS1(gS1Idx, actS1Size, actS2Size)) {
                        ++workRowNum;
                    }
                }
            }
            if (workRowNum == 0) {
                coreNum = 0;
                return;
            }
            uint32_t rowNumPerCore = workRowNum / coreNum;
            uint32_t deal1MoreRowCoreNum = workRowNum % coreNum;
            coreNum = rowNumPerCore == 0 ? deal1MoreRowCoreNum : coreNum;
            if (curCoreIdx >= coreNum) {
                return;
            }
            uint32_t startRowOrdinal = curCoreIdx * rowNumPerCore +
                                       Min(curCoreIdx, deal1MoreRowCoreNum);
            uint32_t curCoreRowNum = rowNumPerCore + (curCoreIdx < deal1MoreRowCoreNum ? 1U : 0U);
            uint32_t endRowOrdinal = startRowOrdinal + curCoreRowNum - 1;
            uint32_t rowOrdinal = 0;
            bool foundStart = false;
            for (uint32_t bN2Idx = 0; bN2Idx < totalRowNum; ++bN2Idx) {
                uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
                if (bN2Idx % constInfo.kHeadNum == 0) {
                    GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
                    s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
                }
                for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; ++gS1Idx) {
                    if (!NeedsStage1ForGS1(gS1Idx, actS1Size, actS2Size)) {
                        continue;
                    }
                    if (rowOrdinal == startRowOrdinal) {
                        info.bN2Start = bN2Idx;
                        info.gS1Start = gS1Idx;
                        info.s2Start = 0;
                        foundStart = true;
                    }
                    if (rowOrdinal == endRowOrdinal) {
                        info.bN2End = bN2Idx;
                        info.gS1End = gS1Idx;
                        info.s2End = GetStage1ScoreTileNum(gS1Idx, actS1Size, actS2Size) - 1;
                        info.isLdMergeCore = false;
                        return;
                    }
                    ++rowOrdinal;
                }
            }
            if (!foundStart) {
                coreNum = 0;
            }
            return;
        }
    }
    uint32_t totalBlockNum = GetTotalStage1BaseBlockNum();
    if (totalBlockNum == 0) {
        coreNum = 0;
        return;
    }
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;
    if (curCoreIdx >= coreNum) {
        return;
    }

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    uint32_t totalRowNum = GetComputeBatchSize() * static_cast<uint32_t>(constInfo.kHeadNum);
    for (uint32_t bN2Idx = 0; bN2Idx < totalRowNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (!NeedsStage1ForGS1(gS1Idx, actS1Size, actS2Size)) {
                continue;
            }
            s2BaseNum = GetStage1ScoreTileNum(gS1Idx, actS1Size, actS2Size);
            for (uint32_t s2Idx = 0; s2Idx < s2BaseNum;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2BaseNum - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

                    if (coreIdx == curCoreIdx) {
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLdMergeCore = true;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if constexpr (LAYOUT_T == LI_LAYOUT::TND) {
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount +
                    n2Idx * constInfo.sparseCount;
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.s1Size; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.s1Size * constInfo.kHeadNum * constInfo.sparseCount +
                                           s1Idx * constInfo.kHeadNum * constInfo.sparseCount +
                                           n2Idx * constInfo.sparseCount;
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                            __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *keyMean,
                                            __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                            const LIHiCachedTilingData *__restrict tiling,
                                            TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);
    InitSplitHints();

    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfoStage2);
    stage1UsedCoreNum = usedCoreNum;
    SplitCoreStage1(aiCoreIdx, stage1UsedCoreNum, splitCoreInfo);

    pipe = tPipe;
    uint64_t offset = 0;
    uint64_t singleCoreMm1ResSize =
        WORKSPACE_BUFFER_NUM * constInfo.mBaseSizeAlign * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    qkWorkspaceGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    partialTopkGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WORKSPACE_BUFFER_NUM * WORKSPACE_BUFFER_NUM * BASE_TOPK * sizeof(float);

    partialTopkMetadataGm.SetGlobalBuffer((__gm__ int64_t *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WORKSPACE_BUFFER_NUM * PARTIAL_TOPK_METADATA_FIELDS * sizeof(int64_t);

    blockIndiceGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
    // TND row ids are global T offsets, so the stage-1 HI block workspace is
    // [T, N2, sparseCount].  BSND still needs [B, S1, N2, sparseCount].
    uint64_t blockIndiceRows = LAYOUT_T == LI_LAYOUT::TND ? constInfo.s1Size :
                                                             constInfo.bSize * constInfo.s1Size;
    offset += blockIndiceRows * constInfo.kHeadNum * constInfo.sparseCount * sizeof(int32_t);
    externalHiMaskGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
    offset += blockIndiceRows * constInfo.kHeadNum * constInfo.externalHiMaskWordNum * sizeof(int32_t);
    // Host still reserves the legacy mean-workspace tail. Cached HI consumes
    // the external keyMean directly; no kernel descriptor uses that tail.
    keyMeanGm.SetGlobalBuffer((__gm__ K_T *)keyMean);

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    keyGm.SetGlobalBuffer((__gm__ K_T *)key);
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ K_T *)weights);
        vectorService.InitGlobalTensors(qkWorkspaceGm, partialTopkGm, partialTopkMetadataGm, blockIndiceGm,
                                       externalHiMaskGm, weightsGm, indiceOutGm);
    } else {
        matmulService.InitParams(constInfo);
        matmulService.InitGlobalTensors(blockTableGm, keyGm, keyMeanGm, queryGm, qkWorkspaceGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (splitCoreInfo.isStage1) {
        s2BlockNum = GetStage1ScoreTileNum(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    } else if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    if (s2BlockNum == 0) {
        tempLoopInfo.s2LoopEnd = 0;
        return;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.s1Size > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LICommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        LICommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, LICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);
    runInfo.isLdMergeCore = splitCoreInfo.isLdMergeCore;

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.s1Size;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        uint64_t tndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        keyCoreOffset = tndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount +
                              runInfo.n2Idx * constInfo.sparseCount;
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum
    * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetGS1TileMaxVisibleS2(const LICommon::RunInfo &runInfo)
{
    return GetGS1MaxVisibleS2(runInfo.gS1Idx, runInfo.actS1Size, runInfo.actS2Size);
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::CalcCanSkipStage1PartialTopkMerge()
{
    if constexpr (LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention) {
        if (constInfo.s1Size == GetComputeBatchSize() || Stage1UsesSingleScoreTile()) {
            // Pure decode has one query row per request. Keep all Stage1 score
            // tiles for a row on one core and maintain its TopM incrementally,
            // matching arch32 while avoiding legacy cross-core LdMerge.
            return true;
        }
    }
    uint32_t actS1Size = 0;
    uint32_t actS2Size = 0;
    uint32_t computeBatchSize = GetComputeBatchSize();
    for (uint32_t bIdx = 0; bIdx < computeBatchSize; ++bIdx) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        uint32_t gS1SplitNum = LICommon::CeilDiv(
            actS1Size * static_cast<uint32_t>(constInfo.gSize), constInfo.mBaseSize);
        for (uint32_t gS1Idx = 0; gS1Idx < gS1SplitNum; ++gS1Idx) {
            if (GetStage1ScoreTileNum(gS1Idx, actS1Size, actS2Size) > 1) {
                return false;
            }
        }
    }
    return true;
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::CanSkipStage1PartialTopkMerge()
{
    if constexpr (LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention) {
        return stage1PartialTopkMergeSkippable;
    }
    return CalcCanSkipStage1PartialTopkMerge();
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::Stage1UsesSingleScoreTile()
{
    if constexpr (!(LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention)) {
        return false;
    }
    return stage1SingleScoreTile;
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::CanSkipStage2PartialTopkMerge()
{
    if constexpr (LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention) {
        // The compact path finishes its own per-query TopK, including split groups.
        return decodeStage2HiFastPathEligible || stage2PartialTopkMergeSkippable;
    }
    return false;
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::InitDecodeStage2FastPathHints()
{
    if constexpr (!(LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention)) {
        return false;
    }
    // Short MTP requests also have independent per-query HI lists. Schedule
    // those lists directly instead of scanning every physical S2 tile.
    constexpr uint32_t MAX_DECODE_QUERY_TOKENS = 4;
    uint32_t queryTokenNum = 0;
    uint32_t firstQueryLen = 0;
    bool uniformQueryLen = true;
    bool hasMtpQuery = false;
    bool hasFullCoverageQuery = false;
    bool denseMtpCandidates = true;
    // Reinterpret the existing [4 * G, 128] ping-pong slot as [G, 512].
    // Aligned G keeps Fixpipe's padded M rows inside the same allocation.
    bool canBatchMtpBlocks = constInfo.gSize % BLOCK_CUBE_SIZE == 0 &&
                            constInfo.sparseCount == BASE_TOPK &&
                            constInfo.hiBlockSize == S2_BASE_SIZE;
    uint32_t mtpBlocksPerChunk = canBatchMtpBlocks ? S1_BASE_SIZE : 1U;
    uint32_t computeBatchSize = GetComputeBatchSize();
    for (uint32_t bIdx = 0; bIdx < computeBatchSize; ++bIdx) {
        uint32_t actS1Size = 0;
        uint32_t actS2Size = 0;
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        if (actS1Size == 0) {
            uniformQueryLen = false;
            continue;
        }
        if (actS1Size > MAX_DECODE_QUERY_TOKENS || actS2Size < actS1Size ||
            (actS1Size > 1 && constInfo.kHeadNum != 1)) {
            return false;
        }
        // Tiny capture inputs retain the stable tiled path. A full-coverage
        // single query can enumerate logical blocks directly, without reading
        // an unmaterialized Stage1 list. MTP still requires per-query lists.
        uint32_t firstVisibleS2 = constInfo.attenMaskFlag ? actS2Size - actS1Size + 1 : actS2Size;
        if (IsHiFullCoverage(firstVisibleS2) &&
            (actS1Size != 1 || firstVisibleS2 <= constInfo.sparseCount)) {
            return false;
        }
        hasFullCoverageQuery = hasFullCoverageQuery || IsHiFullCoverage(firstVisibleS2);
        // Compare handshakes, not blocks: an MTP chunk can batch four blocks
        // while the general path still synchronizes after each 128-token tile.
        uint32_t visibleBlockNum = LICommon::CeilDiv(firstVisibleS2, static_cast<uint32_t>(constInfo.hiBlockSize));
        denseMtpCandidates = denseMtpCandidates && constInfo.hiBlockNum * 2 + 1 >= visibleBlockNum;
        uint32_t hiChunkNum = LICommon::CeilDiv(static_cast<uint32_t>(constInfo.hiBlockNum), mtpBlocksPerChunk);
        if (actS1Size > 1 && static_cast<uint64_t>(hiChunkNum) * actS1Size > visibleBlockNum) {
            return false;
        }
        hasMtpQuery = hasMtpQuery || actS1Size > 1;
        if (firstQueryLen == 0) {
            firstQueryLen = actS1Size;
        }
        uniformQueryLen = uniformQueryLen && actS1Size == firstQueryLen;
        queryTokenNum += actS1Size;
    }
    if (hasMtpQuery && hasFullCoverageQuery) {
        return false;
    }
    decodeQueryTokensPerRequest = uniformQueryLen ? firstQueryLen : 0;
    decodeStage2ChunkSize = hasMtpQuery ? mtpBlocksPerChunk * S2_BASE_SIZE : S2_BASE_SIZE;
    // Share K only for uniform query pairs with substantial candidate overlap.
    // Enough rows avoid a second partial-TopK merge; the union fits one existing
    // Stage1 output row. Other shapes retain independent query scheduling.
    shareMtpStage2Queries = canBatchMtpBlocks && denseMtpCandidates && uniformQueryLen &&
        (firstQueryLen == 2 || firstQueryLen == 4) && constInfo.gSize >= 16 && constInfo.gSize <= 64 &&
        constInfo.maxBlockNumPerBatch <= BASE_TOPK && constInfo.hiBlockNum > 32 &&
        queryTokenNum >= GetBlockNum();
    return queryTokenNum == constInfo.s1Size && queryTokenNum > 0;
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetDecodeStage2HiFastPathRowNum()
{
    if constexpr (!(LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention)) {
        return 0;
    }
    if (!decodeStage2HiFastPathEligible) {
        return 0;
    }
    return constInfo.s1Size * constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::GetDecodeStage2GroupNum(uint32_t queryRowNum)
{
    // Both idle-core routing and active-core execution must use this split.
    constexpr uint32_t MAX_GROUP_NUM = 8;
    uint32_t blocksPerChunk = decodeStage2ChunkSize / constInfo.hiBlockSize;
    uint32_t hiBlockCount = LICommon::Min(
        static_cast<uint32_t>(constInfo.hiBlockNum), static_cast<uint32_t>(constInfo.sparseCount));
    uint32_t chunkCount = LICommon::Max(1U, LICommon::CeilDiv(hiBlockCount, blocksPerChunk));
    uint32_t groupNum = 1U;
    if (queryRowNum > 0) {
        if (GetBlockNum() >= queryRowNum * MAX_GROUP_NUM) {
            groupNum = MAX_GROUP_NUM;
        } else if (GetBlockNum() >= queryRowNum * 4U) {
            groupNum = 4U;
        } else if (GetBlockNum() >= queryRowNum * 2U) {
            groupNum = 2U;
        }
    }
    return LICommon::Min(groupNum, chunkCount);
}

template <typename LIT>
__aicore__ inline uint32_t LIPreload<LIT>::MapDecodeStage2HiFastPathRowToBN2(uint32_t rowOrdinal,
                                                                       uint32_t &queryIdx)
{
    queryIdx = 0;
    uint32_t totalRowNum = GetComputeBatchSize() * static_cast<uint32_t>(constInfo.kHeadNum);
    if constexpr (!(LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention)) {
        return totalRowNum;
    }
    if (decodeQueryTokensPerRequest != 0) {
        queryIdx = rowOrdinal % decodeQueryTokensPerRequest;
        return rowOrdinal / decodeQueryTokensPerRequest;
    }
    uint32_t workRowOrdinal = 0;
    uint32_t actS1Size = 0;
    uint32_t actS2Size = 0;
    for (uint32_t bN2Idx = 0; bN2Idx < totalRowNum; ++bN2Idx) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        }
        if (actS1Size == 0 || actS2Size == 0) {
            continue;
        }
        if (rowOrdinal < workRowOrdinal + actS1Size) {
            queryIdx = rowOrdinal - workRowOrdinal;
            return bN2Idx;
        }
        workRowOrdinal += actS1Size;
    }
    return totalRowNum;
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::Process()
{
    if (usedCoreNum == 0) {
        ProcessInvalid();
        return;
    }
    if ASCEND_IS_AIV {
        vectorService.SetDecodeStage2HiFastPathEligible(decodeStage2HiFastPathEligible);
    }
    RunStage1();
    MergeStage1Blocks();
    RunStage2();
    MergeStage2Tokens();
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t outputRows = LAYOUT_T == LI_LAYOUT::TND ?
            constInfo.s1Size : constInfo.bSize * constInfo.s1Size;
        uint64_t totalOutputSize = outputRows * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            LICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize > totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::RunStage1()
{
    // Coarse stage over HI blocks. AIC writes q * mean(k); AIV reduces over
    // heads/weights and writes the selected HI block list plus optional mask.
    constexpr bool useStage1BlockMean = LIT::pageAttention;
    if (skipStage1ForHiFullCoverage) {
        return;
    }
    if (aiCoreIdx >= stage1UsedCoreNum) {
        return;
    }

    if ASCEND_IS_AIV {
        if constexpr (!useStage1BlockMean) {
            CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
            CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        }
    } else {
        matmulService.InitPipelineEvents();
    }

    LICommon::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            if constexpr (useStage1BlockMean) {
                if (!NeedsStage1ForGS1(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size)) {
                    splitCoreInfo.s2Start = 0;
                    continue;
                }
            }
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                RunStage1Tile(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIC {
        matmulService.DrainPipelineEvents();
        if constexpr (!useStage1BlockMean) {
            CrossCoreWaitFlag(constInfo.syncV1C1);
            CrossCoreWaitFlag(constInfo.syncV1C1);
        }
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::RunMtpSharedStage2()
{
    uint32_t pairCount = GetDecodeStage2HiFastPathRowNum() / 2;
    uint32_t transferIdx = 0;
    uint32_t lane = tmpBlockIdx & 1;
    for (uint32_t pair = aiCoreIdx; pair < pairCount; pair += GetBlockNum()) {
        uint32_t queryIdx = 0;
        uint32_t bN2Idx = MapDecodeStage2HiFastPathRowToBN2(2 * pair, queryIdx);
        splitCoreInfo.bN2Start = bN2Idx;
        splitCoreInfo.bN2End = bN2Idx;
        splitCoreInfo.gS1Start = 0;
        splitCoreInfo.gS1End = 0;
        splitCoreInfo.s2Start = 0;
        splitCoreInfo.s2End = 0;
        splitCoreInfo.isLdMergeCore = false;
        splitCoreInfo.isStage1 = false;
        CalcGS1LoopParams(bN2Idx);
        CalcS2LoopParams(bN2Idx, 0);
        LICommon::RunInfo info;
        CalcRunInfo(0, 0, info);
        info.tensorQueryOffset += static_cast<uint64_t>(queryIdx) * constInfo.qHeadNum * constInfo.headDim;
        info.tensorWeightsOffset += static_cast<uint64_t>(queryIdx) * constInfo.qHeadNum;
        info.indiceOutOffset += static_cast<uint64_t>(queryIdx) * BASE_TOPK;
        if (constInfo.attenMaskFlag) {
            info.actS2Size -= info.actS1Size - queryIdx - 2;
        }
        info.actS1Size = 2;
        info.actMBaseSize = 2 * constInfo.gSize;
        info.actualSingleProcessSInnerSize = 512;
        info.actualSingleProcessSInnerSizeAlign = 512;
        info.isFirstS2InnerLoop = true;
        info.isLastS2InnerLoop = true;
        info.isAllLoopEnd = true;
        info.isLdMergeCore = false;

        uint32_t sharedBlockCount = 0;
        LICommon::RunInfo queryInfo = info;
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_S>(Stage2QkUb::SHARED_LIST_READY_EVENT);
            sharedBlockCount = blockIndiceGm.GetValue(info.indiceOutOffset + BASE_TOPK);
            CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_S>(Stage2QkUb::SHARED_LIST_BROADCAST_EVENT);
            CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_S>(
                Stage2QkUb::SHARED_LIST_BROADCAST_EVENT + Stage2QkUb::AIV_EVENT_OFFSET);
            matmulService.LoadQueryTile(info);
        } else {
            if (lane == 0) {
                vectorService.BuildMtpSharedBlocks(info);
                CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_MTE3>(Stage2QkUb::SHARED_LIST_READY_EVENT);
            }
            CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_S>(Stage2QkUb::SHARED_LIST_BROADCAST_EVENT);
            sharedBlockCount = vectorService.LoadMtpSharedBlocks(info);
            queryInfo.tensorQueryOffset += static_cast<uint64_t>(lane) * constInfo.qHeadNum * constInfo.headDim;
            queryInfo.tensorWeightsOffset += static_cast<uint64_t>(lane) * constInfo.qHeadNum;
            queryInfo.indiceOutOffset += lane * BASE_TOPK;
            queryInfo.actS1Size = 1;
            queryInfo.actMBaseSize = constInfo.gSize;
            if (constInfo.attenMaskFlag) {
                queryInfo.actS2Size -= 1 - lane;
            }
            vectorService.LoadStage2Weights(queryInfo);
        }

        uint32_t chunks = LICommon::CeilDiv(sharedBlockCount, 4U);
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            uint32_t tiles = LICommon::Min(4U, sharedBlockCount - chunk * 4);
            for (uint32_t local = 0; local < tiles;) {
                LICommon::RunInfo tileInfo = info;
                tileInfo.loop = transferIdx;
                // [2*G,256] must fit the existing 64 KiB L0C slot.
                uint32_t blockPair = constInfo.gSize <= 32 && local + 1 < tiles ? 2U : 1U;
                if ASCEND_IS_AIC {
                    for (uint32_t block = 0; block < blockPair; ++block) {
                        uint32_t slot = (transferIdx + block) % 2;
                        CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::FREE_EVENT + slot);
                        CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(
                            Stage2QkUb::FREE_EVENT + slot + Stage2QkUb::AIV_EVENT_OFFSET);
                    }
                    uint64_t blockOffset = info.indiceOutOffset + chunk * 4 + local;
                    if (blockPair == 2) {
                        matmulService.template ComputeStage2BlockPairToUb<true>(tileInfo, blockIndiceGm, blockOffset);
                    } else {
                        uint32_t logicalBlock = static_cast<uint32_t>(blockIndiceGm.GetValue(blockOffset)) >> 2;
                        uint32_t start = logicalBlock * 128;
                        uint32_t count = LICommon::Min(128U, info.actS2Size - start);
                        matmulService.template ComputeStage2QkRange<true, 2>(
                            tileInfo, start, count, 0, 128, 0);
                    }
                    for (uint32_t block = 0; block < blockPair; ++block) {
                        uint32_t slot = (transferIdx + block) % 2;
                        CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::READY_EVENT + slot);
                        CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(
                            Stage2QkUb::READY_EVENT + slot + Stage2QkUb::AIV_EVENT_OFFSET);
                    }
                } else {
                    for (uint32_t block = 0; block < blockPair; ++block) {
                        uint32_t slot = (transferIdx + block) % 2;
                        CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_V>(Stage2QkUb::READY_EVENT + slot);
                        vectorService.ReduceStage2QkUb(slot, (local + block) * 128, chunk * 4 + local + block);
                        CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_V>(Stage2QkUb::FREE_EVENT + slot);
                    }
                }
                transferIdx += blockPair;
                local += blockPair;
            }
            if ASCEND_IS_AIV {
                vectorService.SelectDecodeStage2HiTokens(
                    queryInfo, chunk, 0, chunks, 0, false, false, false, sharedBlockCount);
            }
        }
        if ASCEND_IS_AIC {
            matmulService.ReleaseQueryTile(info);
        }
    }
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::RunDecodeStage2HiFastPath()
{
    // Each decode/MTP query owns a stable Stage1 list. Schedule only its HI
    // chunks, with the same per-query causal boundary as the tiled path.
    if constexpr (!(LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention)) {
        return false;
    }
    if (!decodeStage2HiFastPathEligible) {
        return false;
    }
    if (shareMtpStage2Queries) {
        RunMtpSharedStage2();
        return true;
    }

    uint32_t totalRowNum = GetDecodeStage2HiFastPathRowNum();
    if (totalRowNum == 0) {
        return false;
    }

    uint32_t hiGroupNum = GetDecodeStage2GroupNum(totalRowNum);
    // When rows already outnumber Cube cores, stride scheduling naturally keeps
    // the hardware busy. Splitting every row again would add partial topK GM
    // traffic and a global merge, so reserve multi-way split for small batches.
    uint32_t totalTaskNum = totalRowNum * hiGroupNum;
    if (aiCoreIdx >= totalTaskNum && hiGroupNum > 1) {
        if ASCEND_IS_AIV {
            SyncAll();
        }
        return true;
    }

    uint32_t stage2Loop = 0;
    for (uint32_t taskIdx = aiCoreIdx; taskIdx < totalTaskNum; taskIdx += GetBlockNum()) {
        uint32_t rowOrdinal = taskIdx / hiGroupNum;
        uint32_t queryIdx = 0;
        uint32_t bN2Idx = MapDecodeStage2HiFastPathRowToBN2(rowOrdinal, queryIdx);
        if (bN2Idx >= GetComputeBatchSize() * static_cast<uint32_t>(constInfo.kHeadNum)) {
            continue;
        }
        uint32_t hiGroupIdx = taskIdx - rowOrdinal * hiGroupNum;
        splitCoreInfo.bN2Start = bN2Idx;
        splitCoreInfo.bN2End = bN2Idx;
        splitCoreInfo.gS1Start = 0;
        splitCoreInfo.gS1End = 0;
        splitCoreInfo.s2Start = 0;
        splitCoreInfo.s2End = 0;
        splitCoreInfo.isLdMergeCore = false;
        splitCoreInfo.isStage1 = false;

        CalcGS1LoopParams(bN2Idx);
        // decodeStage2HiFastPathEligible has already validated every decode row. Keep
        // this hot path branch-light; CalcGS1LoopParams still populates offsets
        // and actual sequence sizes for CalcRunInfo.
        CalcS2LoopParams(bN2Idx, 0);

        LICommon::RunInfo runInfo;
        CalcRunInfo(0, 0, runInfo);
        if (runInfo.actS1Size > 1) {
            runInfo.tensorQueryOffset += static_cast<uint64_t>(queryIdx) * constInfo.qHeadNum * constInfo.headDim;
            runInfo.tensorWeightsOffset += static_cast<uint64_t>(queryIdx) * constInfo.qHeadNum;
            runInfo.indiceOutOffset += static_cast<uint64_t>(queryIdx) * constInfo.kHeadNum * constInfo.sparseCount;
            if (constInfo.attenMaskFlag) {
                runInfo.actS2Size -= runInfo.actS1Size - queryIdx - 1;
            }
            runInfo.actS1Size = 1;
            runInfo.actMBaseSize = constInfo.gSize;
        }
        runInfo.actualSingleProcessSInnerSize = decodeStage2ChunkSize;
        runInfo.actualSingleProcessSInnerSizeAlign = decodeStage2ChunkSize;
        runInfo.isFirstS2InnerLoop = true;
        runInfo.isLastS2InnerLoop = true;
        runInfo.isAllLoopEnd = true;
        runInfo.isLdMergeCore = false;

        uint32_t visibleHiBlockNum =
            LICommon::CeilDiv(runInfo.actS2Size, static_cast<uint32_t>(constInfo.hiBlockSize));
        bool hiFullCoverage = IsHiFullCoverage(runInfo.actS2Size);
        uint32_t hiBlockCount = hiFullCoverage ? visibleHiBlockNum : LICommon::Min(
            static_cast<uint32_t>(constInfo.hiBlockNum),
            LICommon::Min(static_cast<uint32_t>(constInfo.sparseCount), visibleHiBlockNum));
        uint32_t blocksPerChunk = decodeStage2ChunkSize / constInfo.hiBlockSize;
        uint32_t chunkCount = LICommon::CeilDiv(hiBlockCount, blocksPerChunk);
        if (chunkCount == 0) {
            continue;
        }
        uint32_t groupStartChunk = (chunkCount * hiGroupIdx) / hiGroupNum;
        uint32_t groupEndChunk = (chunkCount * (hiGroupIdx + 1U)) / hiGroupNum;
        if (groupStartChunk >= groupEndChunk) {
            continue;
        }
        int64_t partialTopkOffset =
            static_cast<int64_t>(rowOrdinal * hiGroupNum + hiGroupIdx) * BASE_TOPK * 2;

        // Split one row's fixed HI block list into Stage2 token chunks. When
        // decode batch exceeds cube-core count, each core walks multiple rows
        // by stride so large batches still use the HI-only Stage2 path.
        if ASCEND_IS_AIC {
            matmulService.LoadQueryTile(runInfo);
        } else if (stage2QkToUb && (tmpBlockIdx & 1) == 0) {
            vectorService.LoadStage2Weights(runInfo);
        }
        for (uint32_t chunkIdx = groupStartChunk; chunkIdx < groupEndChunk; ++chunkIdx) {
            LICommon::RunInfo chunkRunInfo = runInfo;
            // Keep ping-pong parity across MTP rows with odd chunk counts.
            // The two C/V credits protect the last two chunks, not row ids.
            chunkRunInfo.loop = decodeStage2ChunkSize > constInfo.s2BaseSize ? stage2Loop++ : chunkIdx;
            if (stage2QkToUb) {
                uint32_t hiBlockStart = chunkIdx * blocksPerChunk;
                uint32_t tileCount = LICommon::Min(blocksPerChunk, hiBlockCount - hiBlockStart);
                for (uint32_t localBlockIdx = 0; localBlockIdx < tileCount;) {
                    LICommon::RunInfo tileInfo = chunkRunInfo;
                    tileInfo.loop = chunkRunInfo.loop * blocksPerChunk + localBlockIdx;
                    uint32_t slot = tileInfo.loop % Stage2QkUb::BUFFER_NUM;
                    bool useBlockPair = blocksPerChunk > 1 && localBlockIdx + 1 < tileCount;
                    if (useBlockPair) {
                        if ASCEND_IS_AIC {
                            CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::FREE_EVENT);
                            CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::FREE_EVENT + 1);
                            matmulService.ComputeStage2BlockPairToUb(
                                tileInfo, blockIndiceGm, runInfo.indiceOutOffset + hiBlockStart + localBlockIdx);
                            CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::READY_EVENT + slot);
                            CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::READY_EVENT + (slot ^ 1U));
                        } else if ((tmpBlockIdx & 1) == 0) {
                            for (uint32_t block = 0; block < 2; ++block) {
                                uint32_t blockSlot = (slot + block) % Stage2QkUb::BUFFER_NUM;
                                CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_V>(Stage2QkUb::READY_EVENT + blockSlot);
                                vectorService.ReduceStage2QkUb(
                                    blockSlot, (localBlockIdx + block) * Stage2QkUb::TILE_TOKENS);
                                CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_V>(Stage2QkUb::FREE_EVENT + blockSlot);
                            }
                        }
                        localBlockIdx += 2;
                        continue;
                    }
                    if ASCEND_IS_AIC {
                        CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::FREE_EVENT + slot);
                        int32_t selectedBlock = hiFullCoverage ? hiBlockStart + localBlockIdx :
                            blockIndiceGm.GetValue(runInfo.indiceOutOffset + hiBlockStart + localBlockIdx);
                        uint32_t tokenStart = static_cast<uint32_t>(selectedBlock) * constInfo.hiBlockSize;
                        if (selectedBlock >= 0 && tokenStart < runInfo.actS2Size) {
                            uint32_t tokenCount = LICommon::Min(
                                static_cast<uint32_t>(constInfo.hiBlockSize), runInfo.actS2Size - tokenStart);
                            matmulService.template ComputeStage2QkRange<true>(
                                tileInfo, tokenStart, tokenCount, 0, Stage2QkUb::TILE_TOKENS, 0);
                        }
                        // Publish a credit even for a masked tile: the consumer
                        // follows the same fixed block-list extent on both sides.
                        CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::READY_EVENT + slot);
                    } else if ((tmpBlockIdx & 1) == 0) {
                        CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_V>(Stage2QkUb::READY_EVENT + slot);
                        vectorService.ReduceStage2QkUb(slot, localBlockIdx * Stage2QkUb::TILE_TOKENS);
                        // QK has been reduced into separate score storage. Cube
                        // may reuse this slot while AIV executes the next TopK.
                        CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_V>(Stage2QkUb::FREE_EVENT + slot);
                    }
                    ++localBlockIdx;
                }
                if ASCEND_IS_AIV {
                    if ((tmpBlockIdx & 1) == 0) {
                        bool keepPartialTopk = hiGroupNum > 1 && hiGroupIdx == 0;
                        vectorService.SelectDecodeStage2HiTokens(
                            chunkRunInfo, chunkIdx, groupStartChunk, groupEndChunk, partialTopkOffset,
                            hiGroupNum > 1 && !keepPartialTopk, keepPartialTopk, hiFullCoverage);
                    }
                }
                continue;
            }
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag(constInfo.syncV1C1);
                uint32_t hiBlockStart = chunkIdx * blocksPerChunk;
                for (uint32_t localBlockIdx = 0; localBlockIdx < blocksPerChunk; ++localBlockIdx) {
                    uint32_t hiBlockPos = hiBlockStart + localBlockIdx;
                    if (hiBlockPos >= hiBlockCount) {
                        break;
                    }
                    int32_t selectedBlock = hiFullCoverage ? hiBlockPos :
                        blockIndiceGm.GetValue(runInfo.indiceOutOffset + hiBlockPos);
                    if (selectedBlock < 0) {
                        break;
                    }
                    uint32_t tokenStart = static_cast<uint32_t>(selectedBlock) * constInfo.hiBlockSize;
                    if (tokenStart >= runInfo.actS2Size) {
                        continue;
                    }
                    uint32_t tokenCount = LICommon::Min(
                        static_cast<uint32_t>(constInfo.hiBlockSize), runInfo.actS2Size - tokenStart);
                    uint32_t packedOffset = localBlockIdx * constInfo.hiBlockSize;
                    matmulService.ComputeStage2QkRange(
                        chunkRunInfo, tokenStart, tokenCount, packedOffset, decodeStage2ChunkSize, 0);
                }
                PipeBarrier<PIPE_FIX>();
                CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
            } else {
                CrossCoreWaitFlag(constInfo.syncC1V1);
                if ((tmpBlockIdx & 1) == 0) {
                    bool keepPartialTopk = (hiGroupNum > 1 && hiGroupIdx == 0);
                    vectorService.SelectDecodeStage2HiTokens(chunkRunInfo, static_cast<int32_t>(chunkIdx),
                                                             static_cast<int32_t>(groupStartChunk),
                                                             static_cast<int32_t>(groupEndChunk),
                                                             partialTopkOffset,
                                                             hiGroupNum > 1 && !keepPartialTopk,
                                                             keepPartialTopk,
                                                             hiFullCoverage);
                }
                CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
            }
        }
        if (hiGroupNum > 1) {
            if ASCEND_IS_AIV {
                SyncAll();
                if ((tmpBlockIdx & 1) == 0 && hiGroupIdx == 0) {
                    int64_t partialTopkBaseOffset =
                        static_cast<int64_t>(rowOrdinal * hiGroupNum) * BASE_TOPK * 2;
                    vectorService.MergeDecodeStage2HiPartials(runInfo, partialTopkBaseOffset, hiGroupNum, true);
                }
            }
        }
        if ASCEND_IS_AIC {
            matmulService.ReleaseQueryTile(runInfo);
        }
    }
    return true;
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::RunStage2()
{
    // Stage2 token scoring. The decode HI fast path may consume the fixed
    // HI block list directly; all other modes fall back to the tiled Stage2 path.
    if ASCEND_IS_AIV {
        // Stage switch must synchronize all AIV cores before any core returns
        // early, otherwise SyncAll can deadlock when usedCoreNum < total cores.
        SyncAll();
    }

    splitCoreInfo = splitCoreInfoStage2;
    bool runDecodeStage2HiFastPath = false;
    bool joinDecodeStage2HiIdleAiv = false;
    bool skipDecodeStage2HiIdleAic = false;
    if constexpr (LAYOUT_T == LI_LAYOUT::TND && LIT::pageAttention) {
        uint32_t totalRowNum = GetDecodeStage2HiFastPathRowNum();
        uint32_t hiGroupNum = GetDecodeStage2GroupNum(totalRowNum);
        uint32_t fastPathTaskNum = totalRowNum * hiGroupNum;
        // Stage1 emits a compact HI block list whenever the decode fast path is
        // eligible.  Stage2 must consume the same representation even when the
        // number of rows exceeds the Cube core count; RunDecodeStage2HiFastPath
        // already stride-schedules multiple rows on each core.
        bool useFastPathTopology = decodeStage2HiFastPathEligible && totalRowNum > 0;
        if (useFastPathTopology && aiCoreIdx < fastPathTaskNum) {
            runDecodeStage2HiFastPath = true;
        }
        if ASCEND_IS_AIV {
            joinDecodeStage2HiIdleAiv =
                (useFastPathTopology && aiCoreIdx >= fastPathTaskNum && hiGroupNum > 1);
            runDecodeStage2HiFastPath = runDecodeStage2HiFastPath || joinDecodeStage2HiIdleAiv;
        } else {
            // Metadata-padded decode can have fewer active tasks than the
            // capacity-derived usedCoreNum.  Extra Cube cores must not fall back
            // to the tiled Stage2 path while active cores run the HI fast path,
            // otherwise AIC/AIV flag protocols diverge during graph capture.
            skipDecodeStage2HiIdleAic = useFastPathTopology && aiCoreIdx >= fastPathTaskNum;
        }
    }
    if (skipDecodeStage2HiIdleAic) {
        return;
    }
    if (aiCoreIdx >= usedCoreNum && !runDecodeStage2HiFastPath) {
        return;
    }

    stage2QkToUb = runDecodeStage2HiFastPath && !joinDecodeStage2HiIdleAiv &&
                   constInfo.hiBlockSize == Stage2QkUb::TILE_TOKENS &&
                   constInfo.gSize >= 16 && constInfo.gSize <= 64 && constInfo.gSize % 16 == 0;
    if ASCEND_IS_AIV {
        bool useWideMtpChunks = decodeStage2HiFastPathEligible && decodeStage2ChunkSize > constInfo.s2BaseSize;
        if (!CanSkipStage1PartialTopkMerge() || useWideMtpChunks || stage2QkToUb) {
            // Partial TopK merge resets the pipe to its own scratch buffers.
            // Pure decode skips that merge, so keep the normal vector buffers
            // and avoid a second TPipe reset/init before Stage2.
            pipe->Reset();
            vectorService.InitGlobalTensors(qkWorkspaceGm, partialTopkGm, partialTopkMetadataGm, blockIndiceGm,
                                           externalHiMaskGm, weightsGm, indiceOutGm);
            // Wide MTP Stage2 needs one running TopK row, not the two local
            // Stage1 rows. Reuse that UB budget for the wider score buffer.
            vectorService.InitBuffers(pipe, useWideMtpChunks || stage2QkToUb ? decodeStage2ChunkSize : 0U,
                                       stage2QkToUb, shareMtpStage2Queries);
        }
        if (stage2QkToUb) {
            if ((tmpBlockIdx & 1) == 0) {
                CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_MTE3>(Stage2QkUb::STAGE1_LIST_READY_EVENT);
            }
            if ((tmpBlockIdx & 1) == 0 || shareMtpStage2Queries) {
                CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_V>(Stage2QkUb::FREE_EVENT);
                CrossCoreSetFlag<Stage2QkUb::SYNC_MODE, PIPE_V>(Stage2QkUb::FREE_EVENT + 1);
            }
        } else {
            CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
            CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        }
    } else {
        if (stage2QkToUb) {
            // FIX credits only protect QK UB reuse. Scalar must separately wait
            // for Stage1's GM block list before issuing candidate/key loads.
            CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_S>(Stage2QkUb::STAGE1_LIST_READY_EVENT);
            matmulService.InitStage2QkUb(pipe);
        }
        matmulService.InitPipelineEvents();
    }

    bool handledDecodeStage2HiFastPath = RunDecodeStage2HiFastPath();
    if (!handledDecodeStage2HiFastPath) {
        LICommon::RunInfo runInfo;
        uint32_t gloop = 0;
        for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
            CalcGS1LoopParams(bN2LoopIdx);
            if (tempLoopInfo.curActSeqLenIsZero) {
                DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
                continue;
            }
            for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
                CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
                for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                    RunStage2Tile(gloop, s2LoopIdx, runInfo);
                    ++gloop;
                }
                splitCoreInfo.s2Start = 0;
            }
            if (tempLoopInfo.needDealActS1LessThanS1) {
                DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
            }
            splitCoreInfo.gS1Start = 0;
        }
    }

    if ASCEND_IS_AIC {
        matmulService.DrainPipelineEvents();
        if (stage2QkToUb) {
            CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::FREE_EVENT);
            CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(Stage2QkUb::FREE_EVENT + 1);
            if (shareMtpStage2Queries) {
                CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(
                    Stage2QkUb::FREE_EVENT + Stage2QkUb::AIV_EVENT_OFFSET);
                CrossCoreWaitFlag<Stage2QkUb::SYNC_MODE, PIPE_FIX>(
                    Stage2QkUb::FREE_EVENT + 1 + Stage2QkUb::AIV_EVENT_OFFSET);
            }
        } else {
            CrossCoreWaitFlag(constInfo.syncV1C1);
            CrossCoreWaitFlag(constInfo.syncV1C1);
        }
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::RunStage1Tile(uint32_t loop, uint64_t s2LoopIdx, LICommon::RunInfo &runInfo)
{
    // Stage1 tile is block-granular: s2BaseSize means HI blocks here, not raw
    // tokens. This keeps q * mean(k) scheduling compact for long KV.
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if (IsHiFullCoverage(GetGS1TileMaxVisibleS2(runInfo))) {
        return;
    }
    constexpr bool useStage1BlockMean = LIT::pageAttention;
    LICommon::RunInfo stage1RunInfo = runInfo;
    uint32_t localBlockNum = 0;
    uint32_t localBlockNumAlign = 0;
    if constexpr (useStage1BlockMean) {
        uint32_t maxVisibleS2 = GetGS1TileMaxVisibleS2(runInfo);
        uint32_t totalHiBlockNum =
            LICommon::CeilDiv(maxVisibleS2, static_cast<uint32_t>(constInfo.hiBlockSize));
        uint32_t stage1BaseBlockIdx = runInfo.s2Idx * constInfo.s2BaseSize;
        if (stage1BaseBlockIdx >= totalHiBlockNum) {
            return;
        }
        localBlockNum = LICommon::Min(constInfo.s2BaseSize, totalHiBlockNum - stage1BaseBlockIdx);
        localBlockNumAlign = LICommon::Align(localBlockNum, LICommon::ConstInfo::CUBE_BLOCK_ELEM_NUM);
        stage1RunInfo.stage1BufIdx = runInfo.s2Idx & 1;
        stage1RunInfo.stage1LocalBlockNum = localBlockNum;
        stage1RunInfo.stage1BaseBlockIdx = stage1BaseBlockIdx;
        stage1RunInfo.actualSingleProcessSInnerSize = localBlockNum;
        stage1RunInfo.actualSingleProcessSInnerSizeAlign = localBlockNumAlign;
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        if constexpr (useStage1BlockMean) {
            matmulService.ComputeStage1QkTile(stage1RunInfo, localBlockNumAlign);
        } else {
            matmulService.ComputeQkTile(runInfo, false);
        }
        PipeBarrier<PIPE_FIX>();
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        if constexpr (useStage1BlockMean) {
            CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        }
        CrossCoreWaitFlag(constInfo.syncC1V1);
        if constexpr (useStage1BlockMean) {
            vectorService.SelectStage1HiBlocks(stage1RunInfo);
        } else {
            vectorService.SelectStage1TokenBlocks(runInfo);
            CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        }
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::RunStage2Tile(uint32_t loop, uint64_t s2LoopIdx,
                                                     LICommon::RunInfo &runInfo)
{
    // Stage2 keeps original token-level semantics. It only changes the key
    // range consumed by Cube/AIV according to the Stage1 HI block list.
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        if (IsHiFullCoverage(GetGS1TileMaxVisibleS2(runInfo))) {
            // If HI blocks cover the whole visible sequence, dense stage-2
            // is the exact HISA degenerate case and keeps the efficient
            // original path.
            matmulService.ComputeQkTile(runInfo, true);
        } else {
            if (runInfo.isFirstS2InnerLoop) {
                matmulService.LoadQueryTile(runInfo);
            }

            int32_t hiBlockSize = static_cast<int32_t>(constInfo.hiBlockSize);
            int32_t cuBaseS2Idx = static_cast<int32_t>(runInfo.s2Idx * constInfo.s2BaseSize);
            int32_t cuBaseS1Idx = static_cast<int32_t>(runInfo.gS1Idx * constInfo.s1BaseSize);
            int32_t cuS1ProcNum =
                cuBaseS1Idx + static_cast<int32_t>(constInfo.s1BaseSize) > static_cast<int32_t>(runInfo.actS1Size) ?
                    static_cast<int32_t>(runInfo.actS1Size % constInfo.s1BaseSize) :
                    static_cast<int32_t>(constInfo.s1BaseSize);
            int32_t hiBlockLimit = LICommon::Min(
                static_cast<int32_t>(constInfo.hiBlockNum),
                static_cast<int32_t>(constInfo.sparseCount));
            constexpr int32_t HI_MASK_BITS_PER_WORD = 32;

            // HI stage-2: for each query token, run one G x N matmul where G is the
            // 64 head groups that share the same stage-1 HI blocks. Query is
            // cached once per gS1 tile and reused across HI ranges/S2 loops.
            for (int32_t rowIdx = 0; rowIdx < cuS1ProcNum; ++rowIdx) {
                int32_t cuS1Idx = cuBaseS1Idx + rowIdx;
                int32_t rowRealAcSeq = static_cast<int32_t>(runInfo.actS2Size);
                if (constInfo.attenMaskFlag) {
                    rowRealAcSeq = static_cast<int32_t>(runInfo.actS2Size) -
                                   (static_cast<int32_t>(runInfo.actS1Size) - cuS1Idx) + 1;
                }
                int32_t cuS2Len =
                    cuBaseS2Idx + static_cast<int32_t>(constInfo.s2BaseSize) >= rowRealAcSeq ?
                        rowRealAcSeq - cuBaseS2Idx :
                        static_cast<int32_t>(constInfo.s2BaseSize);
                if (rowRealAcSeq <= 0 || cuS2Len <= 0) {
                    continue;
                }

                int32_t blockBase = cuBaseS2Idx / hiBlockSize;
                int32_t localBlockNum = LICommon::CeilDiv(cuS2Len, hiBlockSize);
                int32_t visibleHiBlockNum = LICommon::CeilDiv(rowRealAcSeq, hiBlockSize);
                int32_t hiBlockCount = LICommon::Min(hiBlockLimit, visibleHiBlockNum);
                bool selectAllBlocks = (hiBlockCount >= visibleHiBlockNum);
                int32_t hiMaskWordCount = LICommon::CeilDiv(
                    LICommon::Max(visibleHiBlockNum, 1), HI_MASK_BITS_PER_WORD);
                uint32_t localSelectedMask = 0;
                uint32_t localMaskLimit =
                    localBlockNum >= HI_MASK_BITS_PER_WORD ?
                        0xffffffffU :
                        ((1U << static_cast<uint32_t>(localBlockNum)) - 1U);

                if (selectAllBlocks) {
                    localSelectedMask = localMaskLimit;
                } else if (constInfo.externalHiMaskWordNum > 0) {
                    int64_t rowOutOffset =
                        static_cast<int64_t>(runInfo.indiceOutOffset) + cuS1Idx * constInfo.sparseCount;
                    int64_t maskRowIdx = rowOutOffset / static_cast<int64_t>(constInfo.sparseCount);
                    int64_t maskRowOffset =
                        maskRowIdx * static_cast<int64_t>(constInfo.externalHiMaskWordNum);
                    int32_t wordIdx = blockBase / HI_MASK_BITS_PER_WORD;
                    int32_t bitShift = blockBase % HI_MASK_BITS_PER_WORD;
                    uint32_t maskWord = static_cast<uint32_t>(
                        externalHiMaskGm.GetValue(maskRowOffset + wordIdx));
                    localSelectedMask = maskWord >> static_cast<uint32_t>(bitShift);
                    if (bitShift + localBlockNum > HI_MASK_BITS_PER_WORD &&
                        wordIdx + 1 < hiMaskWordCount) {
                        uint32_t nextMaskWord = static_cast<uint32_t>(
                            externalHiMaskGm.GetValue(maskRowOffset + wordIdx + 1));
                        localSelectedMask |=
                            nextMaskWord << static_cast<uint32_t>(HI_MASK_BITS_PER_WORD - bitShift);
                    }
                    localSelectedMask &= localMaskLimit;
                } else {
                    int64_t rowOutOffset =
                        static_cast<int64_t>(runInfo.indiceOutOffset) + cuS1Idx * constInfo.sparseCount;
                    int32_t embeddedMaskOffset = LICommon::Align(
                        static_cast<int32_t>(constInfo.hiBlockNum),
                        static_cast<int32_t>(B32_BLOCK_ALIGN_NUM));
                    int64_t maskRowOffset = rowOutOffset + embeddedMaskOffset;
                    int32_t wordIdx = blockBase / HI_MASK_BITS_PER_WORD;
                    int32_t bitShift = blockBase % HI_MASK_BITS_PER_WORD;
                    uint32_t maskWord = static_cast<uint32_t>(
                        blockIndiceGm.GetValue(maskRowOffset + wordIdx));
                    localSelectedMask = maskWord >> static_cast<uint32_t>(bitShift);
                    if (bitShift + localBlockNum > HI_MASK_BITS_PER_WORD &&
                        wordIdx + 1 < hiMaskWordCount) {
                        uint32_t nextMaskWord = static_cast<uint32_t>(
                            blockIndiceGm.GetValue(maskRowOffset + wordIdx + 1));
                        localSelectedMask |=
                            nextMaskWord << static_cast<uint32_t>(HI_MASK_BITS_PER_WORD - bitShift);
                    }
                    localSelectedMask &= localMaskLimit;
                }

                int32_t selectedBlockStart[32];
                int32_t selectedBlockTokenCount[32];
                int32_t selectedBlockCount = 0;
                int32_t packedHiTokenCount = 0;
                bool allLocalBlocksSelected = (localSelectedMask == localMaskLimit);
                if (allLocalBlocksSelected) {
                    selectedBlockCount = localBlockNum;
                    packedHiTokenCount = cuS2Len;
                } else {
                    for (int32_t localBlockIdx = 0; localBlockIdx < localBlockNum; ++localBlockIdx) {
                        if ((localSelectedMask & (1U << static_cast<uint32_t>(localBlockIdx))) == 0) {
                            continue;
                        }
                        int32_t tokenStart = localBlockIdx * hiBlockSize;
                        int32_t tokenEnd = LICommon::Min(tokenStart + hiBlockSize, cuS2Len);
                        int32_t tokenCount = tokenEnd - tokenStart;
                        if (tokenCount <= 0) {
                            continue;
                        }
                        selectedBlockStart[selectedBlockCount] = tokenStart;
                        selectedBlockTokenCount[selectedBlockCount] = tokenCount;
                        packedHiTokenCount += tokenCount;
                        ++selectedBlockCount;
                    }
                }
                if (packedHiTokenCount <= 0) {
                    continue;
                }

                // Merge adjacent selected blocks into longer contiguous ranges to
                // avoid splitting the G x N matmul more than necessary.
                int32_t mergedStart[32];
                int32_t mergedTokenCount[32];
                int32_t mergedCount = 0;
                if (allLocalBlocksSelected) {
                    mergedStart[0] = 0;
                    mergedTokenCount[0] = cuS2Len;
                    mergedCount = 1;
                } else {
                    for (int32_t idx = 0; idx < selectedBlockCount; ++idx) {
                        int32_t tokenStart = selectedBlockStart[idx];
                        int32_t tokenCount = selectedBlockTokenCount[idx];
                        if (tokenCount <= 0) {
                            continue;
                        }
                        if (mergedCount > 0 &&
                            tokenStart == mergedStart[mergedCount - 1] + mergedTokenCount[mergedCount - 1]) {
                            mergedTokenCount[mergedCount - 1] += tokenCount;
                        } else {
                            mergedStart[mergedCount] = tokenStart;
                            mergedTokenCount[mergedCount] = tokenCount;
                            ++mergedCount;
                        }
                    }
                }

                uint64_t s2TileBaseOffset = static_cast<uint64_t>(runInfo.s2Idx) * constInfo.s2BaseSize;
                uint64_t packedDstOffset = 0;
                uint64_t queryMOffset = static_cast<uint64_t>(rowIdx) * constInfo.gSize;
                bool useDenseMaskedStage2 =
                    (mergedCount > 1 && packedHiTokenCount * 4 > cuS2Len * 3) ||
                    (packedHiTokenCount == cuS2Len);
                if (useDenseMaskedStage2) {
                    matmulService.ComputeStage2QkRange(
                        runInfo, s2TileBaseOffset, static_cast<uint64_t>(cuS2Len), 0,
                        runInfo.actualSingleProcessSInnerSizeAlign, queryMOffset);
                } else {
                    for (int32_t idx = 0; idx < mergedCount; ++idx) {
                        uint64_t segSrcOffset = s2TileBaseOffset + static_cast<uint64_t>(mergedStart[idx]);
                        uint64_t segTokenCount = static_cast<uint64_t>(mergedTokenCount[idx]);
                        matmulService.ComputeStage2QkRange(
                            runInfo, segSrcOffset, segTokenCount, packedDstOffset,
                            runInfo.actualSingleProcessSInnerSizeAlign, queryMOffset);
                        packedDstOffset += segTokenCount;
                    }
                }
            }
            if (runInfo.isLastS2InnerLoop) {
                matmulService.ReleaseQueryTile(runInfo);
            }
        }
        // Ensure Stage2 QK writes are globally visible before releasing AIV.
        PipeBarrier<PIPE_FIX>();
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.SelectStage2TokenTopK(runInfo);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::MergeStage1Blocks()
{
    // Stage1 PartialTopkMerge is only needed when multiple coarse S2 tiles
    // contribute partial HI block topm for the same row.
    if ASCEND_IS_AIV {
        if (CanSkipStage1PartialTopkMerge()) {
            return;
        }
        // Close stage-1 writer-side GM traffic before resetting buffers and
        // initializing partial TopK merge buffers on another AIV.
        PipeBarrier<PIPE_ALL>();
        vectorService.InitPartialTopkMergeBuffers(pipe);
        ICachePreLoad(PARTIAL_TOPK_MERGE_PREFETCH_LEN);
        SyncAll();
        if (splitCoreInfo.isLdMergeCore) {
            vectorService.MergeStage1BlockTopM();
        }
        // Make stage-1 PartialTopkMerge/writeback visible to every AIV before
        // stage-2 starts consuming blockIndiceGm.
        SyncAll();
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::MergeStage2Tokens()
{
    // Stage2 PartialTopkMerge combines token-level partial topk from tiled
    // Stage2 scoring.
    if ASCEND_IS_AIV {
        if (CanSkipStage2PartialTopkMerge()) {
            return;
        }
        // Close stage-2 writer-side GM traffic before resetting buffers and
        // initializing partial TopK merge buffers on another AIV.
        PipeBarrier<PIPE_ALL>();
        vectorService.InitPartialTopkMergeBuffers(pipe);
        ICachePreLoad(PARTIAL_TOPK_MERGE_PREFETCH_LEN);
        SyncAll();
        if (splitCoreInfo.isLdMergeCore) {
            vectorService.MergeStage2TokenTopK();
        }
    }
}
} // namespace LIKernel
#endif // LIGHTNING_INDEXER_HI_CACHED_ARCH35_KERNEL_H
