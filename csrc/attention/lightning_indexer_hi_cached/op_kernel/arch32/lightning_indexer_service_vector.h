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
 * \file lightning_indexer_service_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_HI_CACHED_ARCH32_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_HI_CACHED_ARCH32_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../lightning_indexer_common.h"
#include "lightning_indexer_vector.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t LD_MERGE_PARAM_NUM = 16;

template <typename LIT>
class LIVector {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr int32_t MAX_LOCAL_BLOCKS_PER_TILE = 32;

    using MM1_OUT_T = float;

    __aicore__ inline LIVector(){};
    // Stage1 AIV path for q * mean(k): reduce weighted heads into HI block scores
    // and maintain the per-row top-m block list.
    __aicore__ inline void SelectStage1HiBlocks(const LICommon::RunInfo &info);
    // Stage1 fallback for non-PA layouts where block scores are built from token scores.
    __aicore__ inline void SelectStage1TokenBlocks(const LICommon::RunInfo &info);
    // LdMerge means Local/Distributed partial-topK merge, not GM/UB load.
    // Stage1 LdMerge: combine partial HI block top-m from multiple coarse tiles.
    __aicore__ inline void MergeStage1BlockTopM();
    // Stage2 AIV path: reduce weighted token scores and maintain token-level top-k.
    __aicore__ inline void SelectStage2TokenTopK(const LICommon::RunInfo &info);
    // Pure decode Stage2 fast path: score token chunks from the selected HI blocks.
    __aicore__ inline void SelectDecodeStage2HiTokens(const LICommon::RunInfo &info, int32_t chunkIdx,
                                                      int32_t groupStartChunk, int32_t groupEndChunk,
                                                      int64_t partialTopkOffset, bool writePartialTopk,
                                                      bool keepPartialTopk, bool hiFullCoverage);
    // Merge multi-way decode HI partial token top-k results back to the final row.
    __aicore__ inline void MergeDecodeStage2HiPartials(const LICommon::RunInfo &info,
                                                       int64_t partialTopkBaseOffset,
                                                       uint32_t hiGroupNum,
                                                       bool firstPartialInUb);
    // Stage2 LdMerge: combine token-level partial top-k from multiple S2 tiles.
    __aicore__ inline void MergeStage2TokenTopK();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LICommon::ConstInfo &constInfo,
                                      const LIHiCachedTilingData *__restrict tilingData);
    __aicore__ inline void SetDecodeStage2HiFastPathEligible(bool eligible);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<int32_t> blockIndiceGm,
                                                GlobalTensor<int32_t> externalHiMaskGm,
                                                GlobalTensor<K_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                                GlobalTensor<int32_t> blockTableGm, GlobalTensor<K_T> keyGm,
                                                GlobalTensor<K_T> stage1MeanKeyGm,
                                                GlobalTensor<K_T> stage1MeanCacheGm,
                                                GlobalTensor<Q_T> queryGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void InitLdMergeBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<int32_t> blockIndiceGm;
    GlobalTensor<int32_t> externalHiMaskGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<K_T> stage1MeanKeyGm;
    GlobalTensor<K_T> stage1MeanCacheGm;
    GlobalTensor<Q_T> queryGm;

private:
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;

    // Temporary buffers for LdMerge tasks.
    TBuf<> ldMergeInputBuf_;
    TBuf<> ldMergeTmpBuf_;
    TBuf<> ldMergeOutValueBuf_;
    TBuf<> ldMergeOutIdxBuf_;

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t topBlockBufElems_ = 0;
    uint8_t decodeStage2CacheMask_ = 0;
    int32_t stage1ActiveBN2Idx_ = -1;
    int32_t stage1ActiveGS1Idx_ = -1;
    int32_t stage2ActiveBN2Idx_ = -1;
    int32_t stage2ActiveGS1Idx_ = -1;
    bool decodeStage2HiFastPathEligible_ = false;

    // Parameters for LdMerge tasks.
    uint32_t ldMergeListNum_ = 4;
    uint32_t ldMergeParamNum_ = 16;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);
    constexpr static int32_t HI_MASK_BITS_PER_WORD = 32;

    struct LICommon::ConstInfo constInfo_;

    __aicore__ inline void ResetTopBlocks(const LocalTensor<float> &topBuf);
    __aicore__ inline float GetPairScore(const LocalTensor<float> &buf, int32_t pairIdx);
    __aicore__ inline int32_t GetPairIndex(const LocalTensor<float> &buf, int32_t pairIdx);
    __aicore__ inline void SetPair(LocalTensor<float> &buf, int32_t pairIdx, float score, int32_t index);
    __aicore__ inline void InsertTopBlock(LocalTensor<float> &topBuf, float score, int32_t blockId,
                                          int32_t topCount);
    __aicore__ inline int32_t GetSinkBlockCount(int32_t totalBlockNum);
    __aicore__ inline int32_t GetRecentBlockCount(int32_t totalBlockNum, int32_t sinkCount);
    __aicore__ inline int32_t GetHiBlockCount(int32_t totalBlockNum);
    __aicore__ inline int32_t GetScoredBlockCount(int32_t totalBlockNum);
    __aicore__ inline int32_t GetHiMaskOffset();
    __aicore__ inline int32_t GetHiMaskWordCount(int32_t totalBlockNum);
    __aicore__ inline bool UsesExternalHiMask();
    __aicore__ inline int64_t GetExternalHiMaskRowOffset(int64_t blockIndiceRowOffset);
    __aicore__ inline void MarkHiMask(LocalTensor<int32_t> &scratchInt, int32_t maskOffset, int32_t blockId);
    __aicore__ inline void EmitSelectedBlockListToTensor(const LocalTensor<float> &topBuf, int64_t outOffset,
                                                         LocalTensor<int32_t> &scratchInt, int32_t totalBlockNum);
    __aicore__ inline void EmitSelectedBlocksAndEmbeddedMaskToTensor(const LocalTensor<float> &topBuf,
                                                                     int64_t outOffset,
                                                                     LocalTensor<int32_t> &scratchInt,
                                                                     int32_t totalBlockNum);
    __aicore__ inline void EmitSelectedBlocksAndExternalMaskToTensor(const LocalTensor<float> &topBuf,
                                                                     int64_t outOffset,
                                                                     LocalTensor<int32_t> &scratchInt,
                                                                     int32_t totalBlockNum);
    __aicore__ inline void AccumulateStage1MeanBlockScores(const LICommon::RunInfo &info, int32_t innerS1Idx,
                                                           int64_t mmGmOffset, int64_t weightGmOffset,
                                                           int32_t outerG, int32_t mmRowWidth,
                                                           int32_t localHiBlockNumAlign,
                                                           LocalTensor<float> &reduceOutInner,
                                                           LocalTensor<float> &stage1TmpScoreUb,
                                                           LocalTensor<float> &brcBuf);
    __aicore__ inline void AccumulateStage1TokenBlockScores(const LICommon::RunInfo &info, int32_t innerS1Idx,
                                                            int64_t mmGmOffset, int64_t weightGmOffset,
                                                            int32_t outerG, int32_t mmRowWidth, int32_t mmUbStride,
                                                            int32_t localHiBlockNum, int32_t hiBlockSize,
                                                            int32_t cuS2Len, LocalTensor<float> &reduceOutInner);
    template <bool USE_STAGE1_MEAN>
    __aicore__ inline void SelectStage1BlocksImpl(const LICommon::RunInfo &info);
};

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    // Stage1 shares this queue between the output list and block score/index storage.
    uint32_t hiScratchSize =
        (constInfo_.sparseCount + constInfo_.maxBlockNumPerBatch) * sizeof(int32_t);
    outNeedBufSize = hiScratchSize > outNeedBufSize ? hiScratchSize : outNeedBufSize;

    pipe->InitBuffer(inQueue_, 2,
                     groupInner_ * s2BaseSize_ * sizeof(float) + s2BaseSize_ * sizeof(float)); // 69KB mm_out_ub
    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);                                            // 32KB  extract
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2 * sizeof(float));    // 64KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                // 2KB
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));                          // 4KB
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    uint32_t paramBufElemNum = LICommon::Max(static_cast<uint32_t>(constInfo_.hiBlockNum),
                                             constInfo_.externalHiMaskWordNum);
    uint32_t paramBufSize = LICommon::Max(static_cast<uint32_t>(LD_MERGE_PARAM_NUM * sizeof(int64_t)),
                                          paramBufElemNum * sizeof(int32_t));
    pipe->InitBuffer(paramBuf_, paramBufSize);

    //
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    SortedBasicBlock_ = globalTopkUb_[BASE_TOPK * 2 * 2];

    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2);
    LocalTensor<float> tmpfBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpfBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * ldMergeParamNum_ * 2);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * ldMergeParamNum_ +
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * ldMergeParamNum_;
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpfBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * ldMergeParamNum_ * sizeof(int64_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(tmpfBuff);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitLdMergeBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldMergeInputBuf_, 2 * BASE_TOPK * ldMergeListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldMergeTmpBuf_, 2 * BASE_TOPK * ldMergeListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldMergeOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldMergeOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
    uint32_t paramBufSize = LICommon::Max(static_cast<uint32_t>(LD_MERGE_PARAM_NUM * sizeof(int64_t)),
                                          constInfo_.externalHiMaskWordNum * sizeof(int32_t));
    pipe->InitBuffer(paramBuf_, paramBufSize);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitParams(const struct LICommon::ConstInfo &constInfo,
                                                 const LIHiCachedTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;

    groupInner_ = 16;
    blockId_ = GetBlockIdx();
    topBlockBufElems_ = LICommon::Align(
        static_cast<int32_t>(constInfo_.hiBlockNum * VALUE_AND_INDEX_NUM),
        static_cast<int32_t>(B32_VEC_ELM_NUM));
    topBlockBufElems_ = LICommon::Max(topBlockBufElems_, static_cast<int32_t>(B32_VEC_ELM_NUM));
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ResetTopBlocks(const LocalTensor<float> &topBuf)
{
    InitSortOutBuf(topBuf, topBlockBufElems_);
}

template <typename LIT>
__aicore__ inline float LIVector<LIT>::GetPairScore(const LocalTensor<float> &buf, int32_t pairIdx)
{
    return buf.GetValue(pairIdx * VALUE_AND_INDEX_NUM);
}

template <typename LIT>
__aicore__ inline int32_t LIVector<LIT>::GetPairIndex(const LocalTensor<float> &buf, int32_t pairIdx)
{
    return buf.template ReinterpretCast<int32_t>().GetValue(pairIdx * VALUE_AND_INDEX_NUM + 1);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SetPair(LocalTensor<float> &buf, int32_t pairIdx, float score, int32_t index)
{
    buf.SetValue(pairIdx * VALUE_AND_INDEX_NUM, score);
    buf.template ReinterpretCast<int32_t>().SetValue(pairIdx * VALUE_AND_INDEX_NUM + 1, index);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InsertTopBlock(LocalTensor<float> &topBuf, float score, int32_t blockId,
                                                     int32_t topCount)
{
    for (int32_t pos = 0; pos < topCount; ++pos) {
        int32_t curIndex = GetPairIndex(topBuf, pos);
        float curScore = GetPairScore(topBuf, pos);
        if (curIndex >= 0 && score <= curScore) {
            continue;
        }
        for (int32_t shift = topCount - 1; shift > pos; --shift) {
            SetPair(topBuf, shift, GetPairScore(topBuf, shift - 1), GetPairIndex(topBuf, shift - 1));
        }
        SetPair(topBuf, pos, score, blockId);
        return;
    }
}

template <typename LIT>
__aicore__ inline int32_t LIVector<LIT>::GetSinkBlockCount(int32_t totalBlockNum)
{
    return LICommon::Min(static_cast<int32_t>(constInfo_.sink), LICommon::Max(totalBlockNum, 0));
}

template <typename LIT>
__aicore__ inline int32_t LIVector<LIT>::GetRecentBlockCount(int32_t totalBlockNum, int32_t sinkCount)
{
    int32_t remainingBlockNum = LICommon::Max(totalBlockNum - sinkCount, 0);
    return LICommon::Min(static_cast<int32_t>(constInfo_.recent), remainingBlockNum);
}

template <typename LIT>
__aicore__ inline int32_t LIVector<LIT>::GetHiBlockCount(int32_t totalBlockNum)
{
    int32_t hiBlockCount = LICommon::Min(static_cast<int32_t>(constInfo_.hiBlockNum),
                                          static_cast<int32_t>(constInfo_.sparseCount));
    return LICommon::Min(hiBlockCount, LICommon::Max(totalBlockNum, 0));
}

template <typename LIT>
__aicore__ inline int32_t LIVector<LIT>::GetScoredBlockCount(int32_t totalBlockNum)
{
    int32_t sinkCount = GetSinkBlockCount(totalBlockNum);
    int32_t recentCount = GetRecentBlockCount(totalBlockNum, sinkCount);
    int32_t hiBlockCount = GetHiBlockCount(totalBlockNum);
    int32_t middleBlockCount = LICommon::Max(totalBlockNum - sinkCount - recentCount, 0);
    return LICommon::Min(LICommon::Max(hiBlockCount - sinkCount - recentCount, 0), middleBlockCount);
}

template <typename LIT>
__aicore__ inline int32_t LIVector<LIT>::GetHiMaskOffset()
{
    // The mask area is initialized by vector instructions. Keep its UB start
    // 32B-aligned even when hi_block_num is not a multiple of 8 int32 values
    // (for example recall-target values such as 116).
    return LICommon::Align(static_cast<int32_t>(constInfo_.hiBlockNum),
                           static_cast<int32_t>(B32_BLOCK_ALIGN_NUM));
}

template <typename LIT>
__aicore__ inline int32_t LIVector<LIT>::GetHiMaskWordCount(int32_t totalBlockNum)
{
    return LICommon::CeilDiv(LICommon::Max(totalBlockNum, 1), HI_MASK_BITS_PER_WORD);
}

template <typename LIT>
__aicore__ inline bool LIVector<LIT>::UsesExternalHiMask()
{
    return constInfo_.externalHiMaskWordNum > 0;
}

template <typename LIT>
__aicore__ inline int64_t LIVector<LIT>::GetExternalHiMaskRowOffset(int64_t blockIndiceRowOffset)
{
    int64_t rowIdx = blockIndiceRowOffset / static_cast<int64_t>(constInfo_.sparseCount);
    return rowIdx * static_cast<int64_t>(constInfo_.externalHiMaskWordNum);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::MarkHiMask(LocalTensor<int32_t> &scratchInt,
                                                  int32_t maskOffset, int32_t blockId)
{
    if (blockId < 0) {
        return;
    }
    int32_t wordIdx = blockId / HI_MASK_BITS_PER_WORD;
    int32_t bitIdx = blockId % HI_MASK_BITS_PER_WORD;
    uint32_t mask = static_cast<uint32_t>(scratchInt.GetValue(maskOffset + wordIdx));
    mask |= (1U << bitIdx);
    scratchInt.SetValue(maskOffset + wordIdx, static_cast<int32_t>(mask));
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::EmitSelectedBlockListToTensor(const LocalTensor<float> &topBuf,
                                                                    int64_t outOffset,
                                                                    LocalTensor<int32_t> &scratchInt,
                                                                    int32_t totalBlockNum)
{
    int32_t maxOutCount = GetHiBlockCount(totalBlockNum);
    if (maxOutCount <= 0) {
        return;
    }
    Duplicate(scratchInt, constInfo_.INVALID_IDX, maxOutCount);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t sinkCount = GetSinkBlockCount(totalBlockNum);
    int32_t recentCount = GetRecentBlockCount(totalBlockNum, sinkCount);
    int32_t topCount = GetScoredBlockCount(totalBlockNum);
    int32_t outCount = 0;
    for (int32_t blockId = 0; blockId < sinkCount; ++blockId) {
        scratchInt.SetValue(outCount++, blockId);
    }
    for (int32_t pos = 0; pos < topCount; ++pos) {
        int32_t blockId = GetPairIndex(topBuf, pos);
        if (blockId < 0) {
            continue;
        }
        if (outCount >= maxOutCount) {
            break;
        }
        scratchInt.SetValue(outCount++, blockId);
    }
    int32_t recentStart = totalBlockNum - recentCount;
    for (int32_t blockId = recentStart; blockId < totalBlockNum; ++blockId) {
        if (outCount >= maxOutCount) {
            break;
        }
        scratchInt.SetValue(outCount++, blockId);
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(blockIndiceGm[outOffset], scratchInt, maxOutCount);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    PipeBarrier<PIPE_ALL>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::EmitSelectedBlocksAndEmbeddedMaskToTensor(
    const LocalTensor<float> &topBuf, int64_t outOffset, LocalTensor<int32_t> &scratchInt, int32_t totalBlockNum)
{
    int32_t maxOutCount = GetHiBlockCount(totalBlockNum);
    if (maxOutCount <= 0) {
        return;
    }
    int32_t maskOffset = GetHiMaskOffset();
    int32_t maskWordCount = GetHiMaskWordCount(totalBlockNum);
    int32_t copyCount = maskOffset + maskWordCount;
    Duplicate(scratchInt, constInfo_.INVALID_IDX, maskOffset);
    Duplicate(scratchInt[maskOffset], 0, maskWordCount);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t sinkCount = GetSinkBlockCount(totalBlockNum);
    int32_t recentCount = GetRecentBlockCount(totalBlockNum, sinkCount);
    int32_t topCount = GetScoredBlockCount(totalBlockNum);
    int32_t outCount = 0;
    for (int32_t blockId = 0; blockId < sinkCount; ++blockId) {
        scratchInt.SetValue(outCount++, blockId);
        MarkHiMask(scratchInt, maskOffset, blockId);
    }
    for (int32_t pos = 0; pos < topCount; ++pos) {
        int32_t blockId = GetPairIndex(topBuf, pos);
        if (blockId < 0) {
            continue;
        }
        if (outCount >= maxOutCount) {
            break;
        }
        scratchInt.SetValue(outCount++, blockId);
        MarkHiMask(scratchInt, maskOffset, blockId);
    }
    int32_t recentStart = totalBlockNum - recentCount;
    for (int32_t blockId = recentStart; blockId < totalBlockNum; ++blockId) {
        if (outCount >= maxOutCount) {
            break;
        }
        scratchInt.SetValue(outCount++, blockId);
        MarkHiMask(scratchInt, maskOffset, blockId);
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(blockIndiceGm[outOffset], scratchInt, copyCount);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    PipeBarrier<PIPE_ALL>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::EmitSelectedBlocksAndExternalMaskToTensor(
    const LocalTensor<float> &topBuf, int64_t outOffset, LocalTensor<int32_t> &scratchInt, int32_t totalBlockNum)
{
    int32_t maxOutCount = GetHiBlockCount(totalBlockNum);
    if (maxOutCount <= 0) {
        return;
    }
    int32_t maskWordCount = GetHiMaskWordCount(totalBlockNum);
    LocalTensor<int32_t> maskScratch = paramBuf_.Get<int32_t>();
    Duplicate(scratchInt, constInfo_.INVALID_IDX, maxOutCount);
    Duplicate(maskScratch, 0, maskWordCount);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t sinkCount = GetSinkBlockCount(totalBlockNum);
    int32_t recentCount = GetRecentBlockCount(totalBlockNum, sinkCount);
    int32_t topCount = GetScoredBlockCount(totalBlockNum);
    int32_t outCount = 0;
    for (int32_t blockId = 0; blockId < sinkCount; ++blockId) {
        scratchInt.SetValue(outCount++, blockId);
        MarkHiMask(maskScratch, 0, blockId);
    }
    for (int32_t pos = 0; pos < topCount; ++pos) {
        int32_t blockId = GetPairIndex(topBuf, pos);
        if (blockId < 0) {
            continue;
        }
        if (outCount >= maxOutCount) {
            break;
        }
        scratchInt.SetValue(outCount++, blockId);
        MarkHiMask(maskScratch, 0, blockId);
    }
    int32_t recentStart = totalBlockNum - recentCount;
    for (int32_t blockId = recentStart; blockId < totalBlockNum; ++blockId) {
        if (outCount >= maxOutCount) {
            break;
        }
        scratchInt.SetValue(outCount++, blockId);
        MarkHiMask(maskScratch, 0, blockId);
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    LIServiceVec::CopyOut(blockIndiceGm[outOffset], scratchInt, maxOutCount);
    LIServiceVec::CopyOut(externalHiMaskGm[GetExternalHiMaskRowOffset(outOffset)], maskScratch, maskWordCount);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    PipeBarrier<PIPE_ALL>();
}

template <typename LIT>
__aicore__ inline void
LIVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                    GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<int32_t> blockIndiceGm,
                                    GlobalTensor<int32_t> externalHiMaskGm,
                                    GlobalTensor<K_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                    GlobalTensor<int32_t> blockTableGm, GlobalTensor<K_T> keyGm,
                                    GlobalTensor<K_T> stage1MeanKeyGm,
                                    GlobalTensor<K_T> stage1MeanCacheGm,
                                    GlobalTensor<Q_T> queryGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->blockIndiceGm = blockIndiceGm;
    this->externalHiMaskGm = externalHiMaskGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
    this->keyGm = keyGm;
    this->stage1MeanKeyGm = stage1MeanKeyGm;
    this->stage1MeanCacheGm = stage1MeanCacheGm;
    this->queryGm = queryGm;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    LIServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(valueULocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SetDecodeStage2HiFastPathEligible(bool eligible)
{
    decodeStage2HiFastPathEligible_ = eligible;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::AccumulateStage1MeanBlockScores(const LICommon::RunInfo &info,
                                                                      int32_t innerS1Idx, int64_t mmGmOffset,
                                                                      int64_t weightGmOffset, int32_t outerG,
                                                                      int32_t mmRowWidth,
                                                                      int32_t localHiBlockNumAlign,
                                                                      LocalTensor<float> &reduceOutInner,
                                                                      LocalTensor<float> &stage1TmpScoreUb,
                                                                      LocalTensor<float> &brcBuf)
{
    // Stage1 consumes q * mean(k) for HI blocks.  Keep this path separate from
    // the legacy token-block fallback so decode profiling can isolate the real
    // stage1 score cost without extra per-element branch noise.
    for (int32_t outerGidx = 0; outerGidx < outerG; outerGidx++) {
        int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
        LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
        LocalTensor<float> weightsInUb = mmInUb[procGnum * mmRowWidth];
        LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>()[groupInner_];
        LIServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm,
                             mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                 outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign,
                             weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_, procGnum,
                             info.actualSingleProcessSInnerSizeAlign, 0);

        inQueue_.EnQue<float>(mmInUb);
        mmInUb = inQueue_.DeQue<float>();
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        weightsInUb = mmInUb[procGnum * mmRowWidth];
        AscendC::Cast(weightsInUb, weightsInTUb, RoundMode::CAST_NONE, procGnum);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        if (mmRowWidth == 16) {
            Maxs(stage1TmpScoreUb, mmInUb, 0.0f, procGnum * mmRowWidth);
            PipeBarrier<PIPE_V>();
            AscendC::Brcb(brcBuf, weightsInUb,
                          LICommon::CeilDiv(procGnum, static_cast<int32_t>(B32_BLOCK_ALIGN_NUM)),
                          {1, B32_VEC_REPEAT_STRIDE});
            PipeBarrier<PIPE_V>();
            Mul(stage1TmpScoreUb, stage1TmpScoreUb, brcBuf, B32_BLOCK_ALIGN_NUM, procGnum,
                {1, 1, 1, 2, 2, 1});
            Mul(stage1TmpScoreUb[B32_BLOCK_ALIGN_NUM], stage1TmpScoreUb[B32_BLOCK_ALIGN_NUM], brcBuf,
                B32_BLOCK_ALIGN_NUM, procGnum, {1, 1, 1, 2, 2, 1});
            PipeBarrier<PIPE_V>();
            LocalTensor<float> stage1OuterReduceUb = reduceOutInner[localHiBlockNumAlign];
            LIServiceVec::DoReduce(stage1TmpScoreUb, stage1OuterReduceUb, procGnum, mmRowWidth);
            Add(reduceOutInner, reduceOutInner, stage1OuterReduceUb, localHiBlockNumAlign);
            PipeBarrier<PIPE_V>();
        } else {
            Maxs(mmInUb, mmInUb, 0.0f, procGnum * mmRowWidth);
            PipeBarrier<PIPE_V>();
            for (int32_t gIdx = 0; gIdx < procGnum; ++gIdx) {
                float gate = weightsInUb.GetValue(gIdx);
                Axpy(reduceOutInner, mmInUb[gIdx * mmRowWidth], gate, localHiBlockNumAlign);
                PipeBarrier<PIPE_V>();
            }
        }
        inQueue_.FreeTensor(mmInUb);
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::AccumulateStage1TokenBlockScores(const LICommon::RunInfo &info,
                                                                       int32_t innerS1Idx, int64_t mmGmOffset,
                                                                       int64_t weightGmOffset, int32_t outerG,
                                                                       int32_t mmRowWidth, int32_t mmUbStride,
                                                                       int32_t localHiBlockNum, int32_t hiBlockSize,
                                                                       int32_t cuS2Len,
                                                                       LocalTensor<float> &reduceOutInner)
{
    // Fallback path used when stage1 mean blocks are unavailable.  It preserves
    // the original token-block averaging semantics but stays out of the hot
    // q*mean(k) path used by page-attention HI.
    for (int32_t outerGidx = 0; outerGidx < outerG; outerGidx++) {
        int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
        LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
        LocalTensor<float> weightsInUb = mmInUb[procGnum * mmRowWidth];
        LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>()[groupInner_];
        LIServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm,
                             mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                 outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign,
                             weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_, procGnum,
                             info.actualSingleProcessSInnerSizeAlign, mmUbStride);

        inQueue_.EnQue<float>(mmInUb);
        mmInUb = inQueue_.DeQue<float>();
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        weightsInUb = mmInUb[procGnum * s2BaseSize_];
        AscendC::Cast(weightsInUb, weightsInTUb, RoundMode::CAST_NONE, procGnum);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (int32_t gIdx = 0; gIdx < procGnum; ++gIdx) {
            float gate = weightsInUb.GetValue(gIdx);
            int32_t gOffset = gIdx * mmRowWidth;
            for (int32_t localBlockIdx = 0; localBlockIdx < localHiBlockNum; ++localBlockIdx) {
                int32_t tokenStart = localBlockIdx * hiBlockSize;
                int32_t tokenEnd = LICommon::Min(tokenStart + hiBlockSize, cuS2Len);
                float rawBlockSum = 0.0f;
                for (int32_t tokenIdx = tokenStart; tokenIdx < tokenEnd; ++tokenIdx) {
                    rawBlockSum += mmInUb.GetValue(gOffset + tokenIdx);
                }
                float blockMean = rawBlockSum / static_cast<float>(tokenEnd - tokenStart);
                blockMean = blockMean > 0.0f ? blockMean : 0.0f;
                reduceOutInner.SetValue(localBlockIdx,
                                        reduceOutInner.GetValue(localBlockIdx) + blockMean * gate);
            }
        }
        inQueue_.FreeTensor(mmInUb);
    }
}

template <typename LIT>
template <bool USE_STAGE1_MEAN>
__aicore__ inline void LIVector<LIT>::SelectStage1BlocksImpl(const LICommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    bool isNewStage1Group = (info.bN2Idx != stage1ActiveBN2Idx_) || (info.gS1Idx != stage1ActiveGS1Idx_);
    if (isNewStage1Group) {
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        blockS2StartIdx_ = info.s2Idx;
        stage1ActiveBN2Idx_ = info.bN2Idx;
        stage1ActiveGS1Idx_ = info.gS1Idx;
    }
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    constexpr bool useStage1BlockMean = USE_STAGE1_MEAN;
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeq += 1;
        }
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        int32_t hiBlockSize = static_cast<int32_t>(constInfo_.hiBlockSize);
        int32_t totalBlockNum = LICommon::CeilDiv(cuRealAcSeq, hiBlockSize);
        int32_t blockBase = cuBaseS2Idx / hiBlockSize;
        int32_t localHiBlockNum = 0;
        int32_t cuS2Len = 0;
        if constexpr (useStage1BlockMean) {
            blockBase = static_cast<int32_t>(info.stage1BaseBlockIdx);
            if (blockBase < totalBlockNum) {
                localHiBlockNum = LICommon::Min(static_cast<int32_t>(info.stage1LocalBlockNum),
                                                totalBlockNum - blockBase);
                cuS2Len = localHiBlockNum;
            }
        } else {
            cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
            localHiBlockNum = LICommon::CeilDiv(cuS2Len, hiBlockSize);
        }
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = useStage1BlockMean ?
                static_cast<int32_t>(info.actualSingleProcessSInnerSizeAlign) :
                CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmRowWidth = useStage1BlockMean ? static_cast<int32_t>(info.actualSingleProcessSInnerSizeAlign) :
                                                        s2BaseSize_;
            int32_t mmUbStride = useStage1BlockMean ?
                                     0 :
                                     (cuS2LenVecAlign - static_cast<int32_t>(info.actualSingleProcessSInnerSizeAlign)) /
                                         B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            int32_t localHiBlockNumAlign = useStage1BlockMean ?
                static_cast<int32_t>(info.actualSingleProcessSInnerSizeAlign) :
                localHiBlockNum;
            LocalTensor<float> stage1TmpScoreUb = reduceOutBuff;
            PipeBarrier<PIPE_V>();
            Duplicate(reduceOutInner, 0.0f, localHiBlockNumAlign);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            if constexpr (useStage1BlockMean) {
                AccumulateStage1MeanBlockScores(info, innerS1Idx, mmGmOffset, weightGmOffset, outerG, mmRowWidth,
                                                localHiBlockNumAlign, reduceOutInner, stage1TmpScoreUb, brcBuf);
            } else {
                AccumulateStage1TokenBlockScores(info, innerS1Idx, mmGmOffset, weightGmOffset, outerG, mmRowWidth,
                                                 mmUbStride, localHiBlockNum, hiBlockSize, cuS2Len, reduceOutInner);
            }

            bool isS2End = useStage1BlockMean ?
                (blockBase + localHiBlockNum >= totalBlockNum) :
                (cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq);

            LocalTensor<float> rowTopBuf = globalTopkUb_[innerS1Idx * BASE_TOPK * 2];
            int32_t sinkCount = GetSinkBlockCount(totalBlockNum);
            int32_t recentCount = GetRecentBlockCount(totalBlockNum, sinkCount);
            int32_t middleBlockEnd = totalBlockNum - recentCount;
            int32_t scoredBlockCount = GetScoredBlockCount(totalBlockNum);
            constexpr int32_t STAGE1_SORT_LEN = 32;
            int32_t stage1BlockScoreTableRequiredLen = LICommon::Align(totalBlockNum, STAGE1_SORT_LEN);
            int32_t stage1BlockScoreTableLimit =
                (BASE_TOPK * 2 - topBlockBufElems_) / static_cast<int32_t>(VALUE_AND_INDEX_NUM);
            int32_t stage1BlockScoreTableLen =
                LICommon::Min(stage1BlockScoreTableRequiredLen, stage1BlockScoreTableLimit);
            bool isLargeStage1ScoreTable = stage1BlockScoreTableRequiredLen > s2BaseSize_;
            bool allowLargeStage1ScoreTable = info.actS1Size <= 4;
            bool useStage1ScoreTable = useStage1BlockMean && scoredBlockCount > 0 &&
                                       stage1BlockScoreTableRequiredLen <= stage1BlockScoreTableLimit &&
                                       (!isLargeStage1ScoreTable || allowLargeStage1ScoreTable);
            LocalTensor<float> stage1ScoreTable = rowTopBuf[topBlockBufElems_];
            LocalTensor<int32_t> stage1IndexTable =
                stage1ScoreTable[stage1BlockScoreTableLen].template ReinterpretCast<int32_t>();
            if (useStage1ScoreTable && info.s2Idx == blockS2StartIdx_) {
                Duplicate(stage1ScoreTable.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF,
                          stage1BlockScoreTableLen);
                Duplicate(stage1IndexTable, constInfo_.INVALID_IDX, stage1BlockScoreTableLen);
                PipeBarrier<PIPE_V>();
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            }
            if (useStage1ScoreTable) {
                int32_t validStart = LICommon::Min(localHiBlockNum, LICommon::Max(0, sinkCount - blockBase));
                int32_t validEnd = LICommon::Min(localHiBlockNum, LICommon::Max(0, middleBlockEnd - blockBase));
                int32_t validCount = validEnd - validStart;
                if (validCount > 0) {
                    int32_t vecStart = LICommon::Align(validStart, static_cast<int32_t>(B32_BLOCK_ALIGN_NUM));
                    vecStart = LICommon::Min(vecStart, validEnd);
                    int32_t vecEnd = (validEnd / static_cast<int32_t>(B32_BLOCK_ALIGN_NUM)) *
                                     static_cast<int32_t>(B32_BLOCK_ALIGN_NUM);
                    vecEnd = LICommon::Max(vecEnd, vecStart);
                    for (int32_t localBlockIdx = validStart; localBlockIdx < vecStart; ++localBlockIdx) {
                        int32_t globalBlockId = blockBase + localBlockIdx;
                        stage1ScoreTable.SetValue(globalBlockId, reduceOutInner.GetValue(localBlockIdx));
                        stage1IndexTable.SetValue(globalBlockId, globalBlockId);
                    }
                    int32_t vecCount = vecEnd - vecStart;
                    if (vecCount > 0) {
                        Adds(stage1ScoreTable[blockBase + vecStart], reduceOutInner[vecStart], 0.0f, vecCount);
                        ArithProgression<int32_t>(stage1IndexTable[blockBase + vecStart],
                                                  blockBase + vecStart, 1, vecCount);
                    }
                    for (int32_t localBlockIdx = vecEnd; localBlockIdx < validEnd; ++localBlockIdx) {
                        int32_t globalBlockId = blockBase + localBlockIdx;
                        stage1ScoreTable.SetValue(globalBlockId, reduceOutInner.GetValue(localBlockIdx));
                        stage1IndexTable.SetValue(globalBlockId, globalBlockId);
                    }
                    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
                }
            } else if (useStage1BlockMean && scoredBlockCount > 0 && localHiBlockNum <= s2BaseSize_) {
                int32_t localSortLen = LICommon::Align(localHiBlockNum, STAGE1_SORT_LEN);
                LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
                LocalTensor<float> sortScoreUb = tmpSortBuf;
                LocalTensor<int32_t> sortIndiceUbInt = tmpSortBuf[localSortLen].template ReinterpretCast<int32_t>();
                Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, localSortLen);
                Duplicate(sortIndiceUbInt, constInfo_.INVALID_IDX, localSortLen);
                PipeBarrier<PIPE_V>();
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                int32_t validStart = LICommon::Min(localHiBlockNum, LICommon::Max(0, sinkCount - blockBase));
                int32_t validEnd = LICommon::Min(localHiBlockNum, LICommon::Max(0, middleBlockEnd - blockBase));
                int32_t validCount = validEnd - validStart;
                if (validCount > 0) {
                    int32_t vecStart = LICommon::Align(validStart, static_cast<int32_t>(B32_BLOCK_ALIGN_NUM));
                    vecStart = LICommon::Min(vecStart, validEnd);
                    int32_t vecEnd = (validEnd / static_cast<int32_t>(B32_BLOCK_ALIGN_NUM)) *
                                     static_cast<int32_t>(B32_BLOCK_ALIGN_NUM);
                    vecEnd = LICommon::Max(vecEnd, vecStart);
                    for (int32_t localBlockIdx = validStart; localBlockIdx < vecStart; ++localBlockIdx) {
                        int32_t globalBlockId = blockBase + localBlockIdx;
                        sortScoreUb.SetValue(localBlockIdx, reduceOutInner.GetValue(localBlockIdx));
                        sortIndiceUbInt.SetValue(localBlockIdx, globalBlockId);
                    }
                    int32_t vecCount = vecEnd - vecStart;
                    if (vecCount > 0) {
                        Adds(sortScoreUb[vecStart], reduceOutInner[vecStart], 0.0f, vecCount);
                        ArithProgression<int32_t>(sortIndiceUbInt[vecStart], blockBase + vecStart, 1, vecCount);
                    }
                    for (int32_t localBlockIdx = vecEnd; localBlockIdx < validEnd; ++localBlockIdx) {
                        int32_t globalBlockId = blockBase + localBlockIdx;
                        sortScoreUb.SetValue(localBlockIdx, reduceOutInner.GetValue(localBlockIdx));
                        sortIndiceUbInt.SetValue(localBlockIdx, globalBlockId);
                    }
                }
                PipeBarrier<PIPE_V>();
                // AscendC::Sort produces independently sorted 32-element basic blocks.  MrgSort requires
                // each input lane to be globally ordered, so complete the tile-wide merge before combining
                // it with the accumulated Stage1 top blocks.  The score/index halves in tmpSortBuf match the
                // input layout expected by the full SortAll helper; reduceOutBuff is large enough for both
                // the SortAll scratch proposal list.
                LIServiceVec::SortAll(tmpSortBuf, reduceOutBuff, localSortLen);
                // The merge emits both input lists; keep its scratch after the
                // sorted tile instead of overflowing the fixed 512-pair reduce buffer.
                LocalTensor<float> mergeScratch = tmpSortBuf[localSortLen * VALUE_AND_INDEX_NUM];
                LIServiceVec::MergeSort(rowTopBuf, topBlockBufElems_ / VALUE_AND_INDEX_NUM, tmpSortBuf,
                                        localSortLen, mergeScratch);
                outQueue_.FreeTensor(tmpSortBuf);
            } else {
                for (int32_t localBlockIdx = 0; localBlockIdx < localHiBlockNum; ++localBlockIdx) {
                    int32_t globalBlockId = blockBase + localBlockIdx;
                    if (globalBlockId < sinkCount || globalBlockId >= middleBlockEnd) {
                        continue;
                    }
                    InsertTopBlock(rowTopBuf, reduceOutInner.GetValue(localBlockIdx), globalBlockId, scoredBlockCount);
                }
            }

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;
            if (useStage1ScoreTable && (needCopyOutGm || needCopyWsGm)) {
                LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
                LocalTensor<float> stage1SortedPairs = tmpSortBuf;
                LocalTensor<float> stage1SortTmp =
                    tmpSortBuf[stage1BlockScoreTableLen * VALUE_AND_INDEX_NUM];
                LocalTensor<uint32_t> stage1IndexTableUint = stage1IndexTable.template ReinterpretCast<uint32_t>();
                LIServiceVec::SortAll(stage1SortedPairs, stage1ScoreTable, stage1IndexTableUint, stage1SortTmp,
                                      stage1BlockScoreTableLen);
                DataCopy(rowTopBuf, stage1SortedPairs, topBlockBufElems_);
                PipeBarrier<PIPE_V>();
                outQueue_.FreeTensor(tmpSortBuf);
            }

            if (needCopyOutGm) {
                int64_t outOffset = info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount;
                LocalTensor<float> scratch = outQueue_.AllocTensor<float>();
                LocalTensor<int32_t> scratchInt = scratch.template ReinterpretCast<int32_t>();
                if (decodeStage2HiFastPathEligible_) {
                    EmitSelectedBlockListToTensor(rowTopBuf, outOffset, scratchInt, totalBlockNum);
                } else if (UsesExternalHiMask()) {
                    EmitSelectedBlocksAndExternalMaskToTensor(rowTopBuf, outOffset, scratchInt, totalBlockNum);
                } else {
                    EmitSelectedBlocksAndEmbeddedMaskToTensor(rowTopBuf, outOffset, scratchInt, totalBlockNum);
                }
                outQueue_.FreeTensor(scratch);
                ResetTopBlocks(rowTopBuf);
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, topkOut_] float32
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64
                //     16 = [needMerge, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, S1ProcNum, ......]

                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK +
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * ldMergeParamNum_ +
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * ldMergeParamNum_ +
                                       (ldS1Offset + innerS1Idx) * 2 * ldMergeParamNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(0));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount));
                bool isTailReduce = blockS2StartIdx_ == 0;
                if (isTailReduce) {
                    wsInfoOffset += ldMergeParamNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIServiceVec::CopyOut(vec1ResGm[wsOffset], rowTopBuf, topBlockBufElems_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                PipeBarrier<PIPE_ALL>();
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                PipeBarrier<PIPE_ALL>();
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }

    if (LAYOUT_T == LI_LAYOUT::BSND) {
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.s1Size - info.actS1Size;
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.s1Size + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SelectStage1HiBlocks(const LICommon::RunInfo &info)
{
    SelectStage1BlocksImpl<true>(info);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SelectStage1TokenBlocks(const LICommon::RunInfo &info)
{
    SelectStage1BlocksImpl<false>(info);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::MergeStage1BlockTopM()
{
    int32_t cubeNum = GetBlockNum();
    int32_t curCubeId = blockId_ / 2;
    int64_t needMerge;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t isS2End;
    int64_t outOffset = 0;

    LocalTensor<float> mergedTopUb = ldMergeInputBuf_.Get<float>();
    LocalTensor<float> tmpTopUb = ldMergeTmpBuf_.Get<float>();
    LocalTensor<int64_t> paramSlotLocal = paramBuf_.Get<int64_t>();
    LocalTensor<int32_t> ldScratchIdx = ldMergeOutIdxBuf_.Get<int32_t>();

    uint32_t s1MergeStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = curCubeId * s1BaseSize_ * 2 * ldMergeParamNum_;
    bool foundActiveSlot = false;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        int64_t slotInfoOffset = paramGmCoreOffset + innerS1Idx * 2 * ldMergeParamNum_ + ldMergeParamNum_;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(paramSlotLocal, vec1ParamGm[slotInfoOffset],
                    {1, static_cast<uint16_t>(ldMergeParamNum_ * sizeof(int64_t)), 0, 0}, {true, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        needMerge = paramSlotLocal.GetValue(0);
        if (needMerge == 1) {
            foundActiveSlot = true;
            s1MergeStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1MergeStartIdx;
            s1ProcNum++;
        } else if (foundActiveSlot) {
            break;
        }
    }
    if (s1ProcNum == 0) {
        return;
    }

    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1MergeStartIdx = s1MergeStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1MergeStartIdx; innerS1Idx < s1MergeStartIdx + s1VecNum; innerS1Idx++) {
        wsInfoOffset = curCubeId * s1BaseSize_ * 2 * ldMergeParamNum_ + innerS1Idx * 2 * ldMergeParamNum_ + ldMergeParamNum_;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(paramSlotLocal, vec1ParamGm[wsInfoOffset],
                    {1, static_cast<uint16_t>(ldMergeParamNum_ * sizeof(int64_t)), 0, 0}, {true, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        int64_t baseBN2Idx = paramSlotLocal.GetValue(5);
        int64_t baseS1Idx = paramSlotLocal.GetValue(6);
        int64_t baseS2AcSeq = paramSlotLocal.GetValue(1);
        outOffset = paramSlotLocal.GetValue(8);
        int32_t totalBlockNum = LICommon::CeilDiv(static_cast<int32_t>(baseS2AcSeq),
                                                  static_cast<int32_t>(constInfo_.hiBlockSize));
        int32_t scoredBlockCount = GetScoredBlockCount(totalBlockNum);
        wsOffset = curCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(mergedTopUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(topBlockBufElems_ * sizeof(float)), 0, 0}, {true, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        uint64_t mergedCubeMask = (curCubeId < 64) ? (1ULL << curCubeId) : 0ULL;
        bool hasEndPartial = false;
        constexpr int32_t LD_MERGE_METADATA_RETRY_TIMES = 1;
        for (int32_t retry = 0; retry < LD_MERGE_METADATA_RETRY_TIMES; ++retry) {
            bool foundNewPartial = false;
            for (int32_t tmpCubeId = 0; tmpCubeId < cubeNum; ++tmpCubeId) {
                if (tmpCubeId < 64 && ((mergedCubeMask >> tmpCubeId) & 1ULL) != 0) {
                    continue;
                }
                int32_t matchedSlotIdx = -1;
                int32_t matchedHalfIdx = -1;
                int32_t slotIdx = static_cast<int32_t>(innerS1Idx);
                for (int32_t halfIdx = 0; halfIdx < 2; ++halfIdx) {
                    if (tmpCubeId == curCubeId && slotIdx == static_cast<int32_t>(innerS1Idx) && halfIdx == 1) {
                        continue;
                    }
                    wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * ldMergeParamNum_ +
                                   slotIdx * 2 * ldMergeParamNum_ + halfIdx * ldMergeParamNum_;
                    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                    DataCopyPad(paramSlotLocal, vec1ParamGm[wsInfoOffset],
                                {1, static_cast<uint16_t>(ldMergeParamNum_ * sizeof(int64_t)), 0, 0}, {true, 0, 0, 0});
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    needMerge = paramSlotLocal.GetValue(0);
                    int64_t curBN2Idx = paramSlotLocal.GetValue(5);
                    int64_t curS1Idx = paramSlotLocal.GetValue(6);
                    if (needMerge == 1 && curBN2Idx == baseBN2Idx && curS1Idx == baseS1Idx) {
                        matchedSlotIdx = slotIdx;
                        matchedHalfIdx = halfIdx;
                        isS2End = paramSlotLocal.GetValue(4);
                        break;
                    }
                }
                if (matchedSlotIdx < 0) {
                    continue;
                }
                if (tmpCubeId < 64) {
                    mergedCubeMask |= (1ULL << tmpCubeId);
                }
                foundNewPartial = true;
                hasEndPartial = hasEndPartial || (isS2End == 1);
                wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                           matchedSlotIdx * 2 * 2 * BASE_TOPK + matchedHalfIdx * 2 * BASE_TOPK;
                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                DataCopyPad(tmpTopUb, vec1ResGm[wsOffset],
                            {1, static_cast<uint16_t>(topBlockBufElems_ * sizeof(float)), 0, 0},
                            {true, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                if (scoredBlockCount > 0) {
                    int32_t topPairCount = topBlockBufElems_ / VALUE_AND_INDEX_NUM;
                    LocalTensor<float> mergeScratch = tmpTopUb[topBlockBufElems_];
                    LIServiceVec::MergeSort(mergedTopUb, topPairCount, tmpTopUb, topPairCount, mergeScratch);
                }
            }
            if (hasEndPartial && !foundNewPartial) {
                break;
            }
            PipeBarrier<PIPE_ALL>();
        }

        if (decodeStage2HiFastPathEligible_) {
            EmitSelectedBlockListToTensor(mergedTopUb, outOffset, ldScratchIdx, totalBlockNum);
        } else if (UsesExternalHiMask()) {
            EmitSelectedBlocksAndExternalMaskToTensor(mergedTopUb, outOffset, ldScratchIdx, totalBlockNum);
        } else {
            EmitSelectedBlocksAndEmbeddedMaskToTensor(mergedTopUb, outOffset, ldScratchIdx, totalBlockNum);
        }
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SelectDecodeStage2HiTokens(const LICommon::RunInfo &info,
                                                                 int32_t chunkIdx, int32_t groupStartChunk,
                                                                 int32_t groupEndChunk,
                                                                 int64_t partialTopkOffset,
                                                                 bool writePartialTopk,
                                                                 bool keepPartialTopk,
                                                                 bool hiFullCoverage)
{
    // Decode HI path: one query row is owned by the AIV pair for this
    // AIC. Keep the odd AIV in the synchronization protocol but do no work.
    if ((blockId_ & 1) != 0) {
        return;
    }

    int32_t hiBlockSize = static_cast<int32_t>(constInfo_.hiBlockSize);
    int32_t blocksPerChunk = s2BaseSize_ / hiBlockSize;
    int32_t visibleHiBlockNum = LICommon::CeilDiv(static_cast<int32_t>(info.actS2Size), hiBlockSize);
    int32_t hiBlockCount = hiFullCoverage ? visibleHiBlockNum : GetHiBlockCount(visibleHiBlockNum);
    int32_t hiBlockStart = chunkIdx * blocksPerChunk;
    if (hiBlockStart >= hiBlockCount) {
        return;
    }

    int64_t rowOutOffset = info.indiceOutOffset;
    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    int64_t weightGmOffset = info.tensorWeightsOffset;
    int32_t outerG = CeilDiv(gSize_, groupInner_);
    int32_t mmRowWidth = s2BaseSize_;
    int32_t sortDataLen = s2BaseSize_;

    int32_t localChunkIdx = chunkIdx - groupStartChunk;
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> sortScoreUb = reduceOutBuff;
    LocalTensor<float> sortIndiceUb = reduceOutBuff[sortDataLen];
    LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    LocalTensor<int32_t> hiBlockUb = paramBuf_.Get<int32_t>();

    if (localChunkIdx == 0) {
        // Decode owns a single query row here; initialize only that row's
        // running topK instead of the 4-row general vector buffer.
        InitSortOutBuf(globalTopkUb_, BASE_TOPK * 2);
        blockS2StartIdx_ = 0;
        decodeStage2CacheMask_ = 0;
        if (!hiFullCoverage) {
            // The same HI block list is consumed by every 512-token chunk
            // in this row/group.  Copy it once into a small dedicated UB buffer so
            // the hot chunk loop avoids repeated GM scalar reads.
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(hiBlockUb, blockIndiceGm[rowOutOffset],
                        {1, static_cast<uint16_t>(hiBlockCount * sizeof(int32_t)), 0, 0},
                        {true, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        }
    }

    PipeBarrier<PIPE_V>();
    LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
    for (int outerGidx = 0; outerGidx < outerG; ++outerGidx) {
        int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
        LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
        LocalTensor<float> weightsInUb = mmInUb[procGnum * s2BaseSize_];
        LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>()[groupInner_];
        int64_t mmOffset = mmGmOffset + outerGidx * groupInner_ * mmRowWidth;
        LIServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm, mmOffset,
                             weightGmOffset + outerGidx * groupInner_, procGnum, mmRowWidth, 0);

        inQueue_.EnQue<float>(mmInUb);
        mmInUb = inQueue_.DeQue<float>();
        weightsInUb = mmInUb[procGnum * s2BaseSize_];
        LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, weightsInTUb,
                              brcBuf, procGnum, s2BaseSize_, outerGidx);
        inQueue_.FreeTensor(mmInUb);
    }

    int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
    LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], sortScoreUb, gRedCnt, s2BaseSize_);
    outQueue_.FreeTensor(reduceCacheBuf);

    PipeBarrier<PIPE_V>();
    Duplicate(sortIndiceUbInt, LIServiceVec::INVALID_INDEX, sortDataLen);
    PipeBarrier<PIPE_V>();

    bool hasHiBlock = false;
    int32_t selectedTokenStart[MAX_LOCAL_BLOCKS_PER_TILE] = {0};
    int32_t selectedTokenCount[MAX_LOCAL_BLOCKS_PER_TILE] = {0};
    for (int32_t localBlockIdx = 0; localBlockIdx < blocksPerChunk; ++localBlockIdx) {
        int32_t hiBlockPos = hiBlockStart + localBlockIdx;
        if (hiBlockPos >= hiBlockCount) {
            break;
        }
        int32_t selectedBlock = hiFullCoverage ? hiBlockPos : hiBlockUb.GetValue(hiBlockPos);
        if (selectedBlock < 0) {
            break;
        }
        int32_t tokenStart = selectedBlock * hiBlockSize;
        int32_t tokenCount = LICommon::Min(hiBlockSize, static_cast<int32_t>(info.actS2Size) - tokenStart);
        if (tokenCount <= 0) {
            continue;
        }
        selectedTokenStart[localBlockIdx] = tokenStart;
        selectedTokenCount[localBlockIdx] = tokenCount;
        hasHiBlock = true;
    }
    for (int32_t localBlockIdx = 0; localBlockIdx < blocksPerChunk; ++localBlockIdx) {
        int32_t packedStart = localBlockIdx * hiBlockSize;
        int32_t tokenCount = selectedTokenCount[localBlockIdx];
        if (tokenCount <= 0) {
            Duplicate(sortScoreUb[packedStart].template ReinterpretCast<int32_t>(),
                      LIServiceVec::NEG_INF, hiBlockSize);
            continue;
        }
        Adds(sortIndiceUbInt[packedStart], globalTopkIndice_, selectedTokenStart[localBlockIdx], tokenCount);
        if (tokenCount < hiBlockSize) {
            // Keep all vector instructions aligned: back up the whole HI block,
            // clear the whole block to -inf, then restore the valid prefix.
            Adds(brcBuf, sortScoreUb[packedStart], 0.0f, hiBlockSize);
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb[packedStart].template ReinterpretCast<int32_t>(),
                      LIServiceVec::NEG_INF, hiBlockSize);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb[packedStart], brcBuf, 0.0f, tokenCount);
        }
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
    int64_t globalTopkUbCacheIdx = localChunkIdx % 4;
    LocalTensor<float> cacheSortedBlock =
        SortedBasicBlock_[globalTopkUbCacheIdx * s2BaseSize_ * 2];
    if (hasHiBlock) {
        Sort<float, true>(cacheSortedBlock, reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(),
                          tmpSortBuf, s2BaseSize_ / 32);
    } else {
        InitSortOutBuf(cacheSortedBlock, s2BaseSize_ * 2);
    }

    bool isChunkGroupEnd = (globalTopkUbCacheIdx == 3) || (chunkIdx == groupEndChunk - 1);
    if (isChunkGroupEnd) {
        LocalTensor<float> tt = SortedBasicBlock_;
        if (localChunkIdx < 4) {
            MrgBasicBlock(globalTopkUb_, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
        } else {
            if (globalTopkUbCacheIdx > 0) {
                MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                PipeBarrier<PIPE_V>();
                DataCopy(SortedBasicBlock_, tmpSortBuf, (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
            }
            PipeBarrier<PIPE_V>();
            SparseTopK(globalTopkUb_, SortedBasicBlock_, tmpSortBuf, BASE_TOPK,
                       s2BaseSize_ * (globalTopkUbCacheIdx + 1));
        }
    }
    PipeBarrier<PIPE_V>();
    outQueue_.FreeTensor(tmpSortBuf);

    if (chunkIdx == groupEndChunk - 1) {
        if (writePartialTopk) {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            LIServiceVec::CopyOut(vec1ResGm[partialTopkOffset], globalTopkUb_, BASE_TOPK * 2);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            return;
        }
        if (keepPartialTopk) {
            return;
        }
        LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
        LocalTensor<uint32_t> idxULocal = valueULocal.template ReinterpretCast<uint32_t>()[BASE_TOPK];
        ExtractIndex(idxULocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
        PipeBarrier<PIPE_V>();
        InitSortOutBuf(globalTopkUb_, BASE_TOPK * 2);
        outQueue_.EnQue<float>(valueULocal);
        valueULocal = outQueue_.DeQue<float>();
        LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
        LIServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset], idxULocal1, constInfo_.sparseCount);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::MergeDecodeStage2HiPartials(const LICommon::RunInfo &info,
                                                                  int64_t partialTopkBaseOffset,
                                                                  uint32_t hiGroupNum,
                                                                  bool firstPartialInUb)
{
    // The decode HI path may split one row across multiple AIV groups. Merge
    // the partial topk lists once, after all groups have written their local topk.
    LocalTensor<float> mergeTmp = outQueue_.AllocTensor<float>();
    uint32_t firstGmGroupIdx = 1U;
    if (!firstPartialInUb) {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(globalTopkUb_, vec1ResGm[partialTopkBaseOffset],
                    {1, static_cast<uint16_t>(BASE_TOPK * 2 * sizeof(float)), 0, 0},
                    {true, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }
    for (uint32_t groupIdx = firstGmGroupIdx; groupIdx < hiGroupNum; ++groupIdx) {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(SortedBasicBlock_, vec1ResGm[partialTopkBaseOffset + groupIdx * BASE_TOPK * 2],
                    {1, static_cast<uint16_t>(BASE_TOPK * 2 * sizeof(float)), 0, 0},
                    {true, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        LIServiceVec::MergeSortTopN(globalTopkUb_, BASE_TOPK, SortedBasicBlock_, BASE_TOPK, BASE_TOPK, mergeTmp);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> valueULocal = mergeTmp;
    LocalTensor<uint32_t> idxULocal = valueULocal.template ReinterpretCast<uint32_t>()[BASE_TOPK];
    ExtractIndex(idxULocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
    PipeBarrier<PIPE_V>();
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
    LIServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset], idxULocal1, constInfo_.sparseCount);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    outQueue_.FreeTensor(valueULocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SelectStage2TokenTopK(const LICommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    int32_t outerG = CeilDiv(gSize_, groupInner_);
    uint64_t hiTokenNum =
        static_cast<uint64_t>(constInfo_.hiBlockNum) * static_cast<uint64_t>(constInfo_.hiBlockSize);

    bool isNewStage2Group = (info.bN2Idx != stage2ActiveBN2Idx_) || (info.gS1Idx != stage2ActiveGS1Idx_);
    if (isNewStage2Group) {
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2);
        ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        blockS2StartIdx_ = info.s2Idx;
        decodeStage2CacheMask_ = 0;
        stage2ActiveBN2Idx_ = info.bN2Idx;
        stage2ActiveGS1Idx_ = info.gS1Idx;
    }
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeq += 1;
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            bool useHiPackedMm = static_cast<uint64_t>(cuRealAcSeq) > hiTokenNum;
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t hiBlockSize = static_cast<int32_t>(constInfo_.hiBlockSize);
            int32_t blockBase = cuBaseS2Idx / hiBlockSize;
            int32_t localHiBlockNum = LICommon::CeilDiv(cuS2Len, hiBlockSize);
            int64_t rowOutOffset = info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount;
            bool selectAllVisibleTokens = !useHiPackedMm;
            int32_t visibleHiBlockNum = 0;
            int32_t hiBlockCount = 0;
            bool selectAllBlocks = true;
            if (!selectAllVisibleTokens) {
                visibleHiBlockNum = LICommon::CeilDiv(cuRealAcSeq, hiBlockSize);
                hiBlockCount = GetHiBlockCount(visibleHiBlockNum);
                selectAllBlocks = (hiBlockCount >= visibleHiBlockNum);
            }
            uint32_t localSelectedMask = 0;
            uint32_t localMaskLimit =
                localHiBlockNum >= HI_MASK_BITS_PER_WORD ?
                    0xffffffffU :
                    ((1U << static_cast<uint32_t>(localHiBlockNum)) - 1U);
            if (selectAllBlocks) {
                localSelectedMask = localMaskLimit;
            } else if (UsesExternalHiMask()) {
                int32_t maskWordCount = GetHiMaskWordCount(visibleHiBlockNum);
                int32_t wordIdx = blockBase / HI_MASK_BITS_PER_WORD;
                int32_t bitShift = blockBase % HI_MASK_BITS_PER_WORD;
                int64_t maskRowOffset = GetExternalHiMaskRowOffset(rowOutOffset);
                uint32_t maskWord = static_cast<uint32_t>(
                    externalHiMaskGm.GetValue(maskRowOffset + wordIdx));
                localSelectedMask = maskWord >> static_cast<uint32_t>(bitShift);
                if (bitShift + localHiBlockNum > HI_MASK_BITS_PER_WORD && wordIdx + 1 < maskWordCount) {
                    uint32_t nextMaskWord = static_cast<uint32_t>(
                        externalHiMaskGm.GetValue(maskRowOffset + wordIdx + 1));
                    localSelectedMask |= nextMaskWord << static_cast<uint32_t>(HI_MASK_BITS_PER_WORD - bitShift);
                }
                localSelectedMask &= localMaskLimit;
            } else {
                int32_t maskWordCount = GetHiMaskWordCount(visibleHiBlockNum);
                int32_t wordIdx = blockBase / HI_MASK_BITS_PER_WORD;
                int32_t bitShift = blockBase % HI_MASK_BITS_PER_WORD;
                int64_t maskRowOffset = rowOutOffset + GetHiMaskOffset();
                uint32_t maskWord = static_cast<uint32_t>(
                    blockIndiceGm.GetValue(maskRowOffset + wordIdx));
                localSelectedMask = maskWord >> static_cast<uint32_t>(bitShift);
                if (bitShift + localHiBlockNum > HI_MASK_BITS_PER_WORD && wordIdx + 1 < maskWordCount) {
                    uint32_t nextMaskWord = static_cast<uint32_t>(
                        blockIndiceGm.GetValue(maskRowOffset + wordIdx + 1));
                    localSelectedMask |= nextMaskWord << static_cast<uint32_t>(HI_MASK_BITS_PER_WORD - bitShift);
                }
                localSelectedMask &= localMaskLimit;
            }

            int32_t selectedRangeTokenStart[MAX_LOCAL_BLOCKS_PER_TILE];
            int32_t selectedRangeTokenCount[MAX_LOCAL_BLOCKS_PER_TILE];
            int32_t selectedRangePackedStart[MAX_LOCAL_BLOCKS_PER_TILE];
            int32_t selectedRangeCount = 0;
            int32_t packedHiTokenCount = 0;
            if (localSelectedMask == localMaskLimit) {
                selectedRangeCount = 1;
                selectedRangeTokenStart[0] = 0;
                selectedRangeTokenCount[0] = cuS2Len;
                selectedRangePackedStart[0] = 0;
                packedHiTokenCount = cuS2Len;
            } else {
                for (int32_t localBlockIdx = 0; localBlockIdx < localHiBlockNum; ++localBlockIdx) {
                    if ((localSelectedMask & (1U << static_cast<uint32_t>(localBlockIdx))) == 0) {
                        continue;
                    }
                    int32_t tokenStart = localBlockIdx * hiBlockSize;
                    int32_t tokenEnd = LICommon::Min(tokenStart + hiBlockSize, cuS2Len);
                    int32_t tokenCount = tokenEnd - tokenStart;
                    if (tokenCount <= 0) {
                        continue;
                    }
                    if (selectedRangeCount > 0 &&
                        tokenStart == selectedRangeTokenStart[selectedRangeCount - 1] +
                                          selectedRangeTokenCount[selectedRangeCount - 1]) {
                        selectedRangeTokenCount[selectedRangeCount - 1] += tokenCount;
                    } else {
                        selectedRangeTokenStart[selectedRangeCount] = tokenStart;
                        selectedRangeTokenCount[selectedRangeCount] = tokenCount;
                        selectedRangePackedStart[selectedRangeCount] = packedHiTokenCount;
                        ++selectedRangeCount;
                    }
                    packedHiTokenCount += tokenCount;
                }
            }
            bool hasSelectedHiTokens = packedHiTokenCount > 0;
            if (useHiPackedMm) {
                if (packedHiTokenCount == cuS2Len) {
                    useHiPackedMm = false;
                } else if (selectedRangeCount > 1 && packedHiTokenCount * 4 > cuS2Len * 3) {
                    useHiPackedMm = false;
                }
            }
            int32_t mmRowWidth = info.actualSingleProcessSInnerSizeAlign;
            int32_t mmUbStride = (cuS2LenVecAlign - mmRowWidth) / B32_BLOCK_ALIGN_NUM;
            int32_t sortDataLen = cuS2LenVecAlign;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            if (hasSelectedHiTokens) {
                LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
                bool useDenseBaselinePath = !useHiPackedMm && (packedHiTokenCount == cuS2Len);
                int32_t packedSortLenForPadding = LICommon::Align(
                    static_cast<int32_t>(packedHiTokenCount), 128);
                int32_t mmReduceWidth = useHiPackedMm ?
                    LICommon::Min(LICommon::Align(static_cast<int32_t>(packedHiTokenCount),
                                                  static_cast<int32_t>(B32_VEC_ELM_NUM)),
                                  s2BaseSize_) :
                    s2BaseSize_;
                bool needFullSortPadding =
                    packedHiTokenCount < sortDataLen &&
                    (info.actS1Size <= 4 || packedSortLenForPadding >= cuS2LenVecAlign);

                // Every live Stage2 path reaches this block as dense, packed HI,
                // or dense-masked HI.
                {
                    PipeBarrier<PIPE_V>();
                    LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
                    for (int outerGidx = 0; outerGidx < outerG; ++outerGidx) {
                        int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
                        LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
                        LocalTensor<float> weightsInUb = mmInUb[procGnum * s2BaseSize_];
                        LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>()[groupInner_];
                        int64_t mmOffset =
                            mmGmOffset + innerS1Idx * gSize_ * mmRowWidth + outerGidx * groupInner_ * mmRowWidth;
                        if (useHiPackedMm) {
                            LIServiceVec::CopyInWithSrcStride(
                                mmInUb, weightsInTUb, mm1ResGm, weightsGm, mmOffset,
                                weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_, procGnum,
                                mmReduceWidth, mmRowWidth);
                        } else {
                            LIServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm,
                                                 mmOffset,
                                                 weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_, procGnum,
                                                 mmRowWidth, mmUbStride);
                        }

                        inQueue_.EnQue<float>(mmInUb);
                        mmInUb = inQueue_.DeQue<float>();
                        weightsInUb = mmInUb[procGnum * s2BaseSize_];
                        LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, weightsInTUb,
                                              brcBuf, procGnum, mmReduceWidth, outerGidx);
                        inQueue_.FreeTensor(mmInUb);
                    }

                    int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
                    LocalTensor<float> reduceDstUb = reduceOutInner;
                    if (!needFullSortPadding) {
                        reduceDstUb = sortScoreUb;
                    }
                    LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceDstUb, gRedCnt,
                                           mmReduceWidth);
                    outQueue_.FreeTensor(reduceCacheBuf);

                    PipeBarrier<PIPE_V>();
                    if (needFullSortPadding) {
                        Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, sortDataLen);
                        PipeBarrier<PIPE_V>();
                    }

                    if (useDenseBaselinePath) {
                        if (needFullSortPadding) {
                            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
                            PipeBarrier<PIPE_V>();
                        }
                        if (needFullSortPadding) {
                            Duplicate(sortIndiceUbInt, -1, sortDataLen);
                            PipeBarrier<PIPE_V>();
                        }
                        Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
                    } else if (useHiPackedMm) {
                        if (needFullSortPadding) {
                            for (int32_t rangeIdx = 0; rangeIdx < selectedRangeCount; ++rangeIdx) {
                                int32_t tokenCount = selectedRangeTokenCount[rangeIdx];
                                int32_t packedStart = selectedRangePackedStart[rangeIdx];
                                Adds(sortScoreUb[packedStart], reduceOutInner[packedStart], 0.0f, tokenCount);
                            }
                            PipeBarrier<PIPE_V>();
                        }

                        if (needFullSortPadding) {
                            Duplicate(sortIndiceUbInt, -1, sortDataLen);
                            PipeBarrier<PIPE_V>();
                        }
                        for (int32_t rangeIdx = 0; rangeIdx < selectedRangeCount; ++rangeIdx) {
                            int32_t tokenStart = selectedRangeTokenStart[rangeIdx];
                            int32_t tokenCount = selectedRangeTokenCount[rangeIdx];
                            int32_t packedStart = selectedRangePackedStart[rangeIdx];
                            Adds(sortIndiceUbInt[packedStart], globalTopkIndice_[tokenStart],
                                 static_cast<int32_t>(cuBaseS2Idx), tokenCount);
                        }
                    } else {
                        for (int32_t rangeIdx = 0; rangeIdx < selectedRangeCount; ++rangeIdx) {
                            int32_t tokenStart = selectedRangeTokenStart[rangeIdx];
                            int32_t tokenCount = selectedRangeTokenCount[rangeIdx];
                            int32_t packedStart = selectedRangePackedStart[rangeIdx];
                            Adds(sortScoreUb[packedStart], reduceDstUb[tokenStart], 0.0f, tokenCount);
                        }
                        PipeBarrier<PIPE_V>();

                        if (needFullSortPadding) {
                            Duplicate(sortIndiceUbInt, -1, sortDataLen);
                            PipeBarrier<PIPE_V>();
                        }
                        for (int32_t rangeIdx = 0; rangeIdx < selectedRangeCount; ++rangeIdx) {
                            int32_t tokenStart = selectedRangeTokenStart[rangeIdx];
                            int32_t tokenCount = selectedRangeTokenCount[rangeIdx];
                            int32_t packedStart = selectedRangePackedStart[rangeIdx];
                            Adds(sortIndiceUbInt[packedStart], globalTopkIndice_[tokenStart],
                                 static_cast<int32_t>(cuBaseS2Idx), tokenCount);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                }
            }

            if (info.actS1Size > 4) {
                if (hasSelectedHiTokens) {
                    LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
                    LocalTensor<float> rowTopBuf = globalTopkUb_[innerS1Idx * BASE_TOPK * 2];
                    bool isFirstStage2S2Tile = (info.s2Idx == blockS2StartIdx_);
                    int32_t prevRowTopUpper = LICommon::Min(
                        BASE_TOPK, static_cast<int32_t>(info.s2Idx - blockS2StartIdx_) * s2BaseSize_);
                    int32_t packedSortLen = LICommon::Align(
                        static_cast<int32_t>(packedHiTokenCount),
                        128);
                    if (packedSortLen < cuS2LenVecAlign) {
                        LocalTensor<float> compactSortSrc = tmpSortBuf;
                        LocalTensor<float> compactSortTmp = tmpSortBuf[packedSortLen * VALUE_AND_INDEX_NUM];
                        if (packedSortLen == packedHiTokenCount) {
                            LocalTensor<uint32_t> sortIndiceUbUint =
                                sortIndiceUbInt.template ReinterpretCast<uint32_t>();
                            LIServiceVec::SortAll(compactSortSrc, sortScoreUb,
                                                  sortIndiceUbUint, compactSortTmp, packedSortLen);
                        } else {
                            Duplicate(compactSortSrc.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF,
                                      packedSortLen);
                            Duplicate(compactSortSrc[packedSortLen].template ReinterpretCast<int32_t>(),
                                      LIServiceVec::INVALID_INDEX, packedSortLen);
                            PipeBarrier<PIPE_V>();
                            Adds(compactSortSrc, sortScoreUb, 0.0f, packedHiTokenCount);
                            Adds(compactSortSrc.template ReinterpretCast<int32_t>()[packedSortLen], sortIndiceUbInt,
                                 0, packedHiTokenCount);
                            PipeBarrier<PIPE_V>();
                            LIServiceVec::SortAll(compactSortSrc, compactSortTmp, packedSortLen);
                        }
                        PipeBarrier<PIPE_V>();
                        if (isFirstStage2S2Tile) {
                            DataCopy(rowTopBuf, compactSortSrc, packedSortLen * VALUE_AND_INDEX_NUM);
                            PipeBarrier<PIPE_V>();
                        } else {
                            int32_t mergeOutCount = LICommon::Min(BASE_TOPK, prevRowTopUpper + packedSortLen);
                            LIServiceVec::MergeSortTopN(rowTopBuf, prevRowTopUpper, compactSortSrc, packedSortLen,
                                                        mergeOutCount, compactSortTmp);
                        }
                    } else {
                        LIServiceVec::SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);
                        PipeBarrier<PIPE_V>();
                        if (isFirstStage2S2Tile) {
                            DataCopy(rowTopBuf, reduceOutBuff, cuS2LenVecAlign * VALUE_AND_INDEX_NUM);
                            PipeBarrier<PIPE_V>();
                        } else {
                            int32_t mergeOutCount = LICommon::Min(BASE_TOPK, prevRowTopUpper + cuS2LenVecAlign);
                            LIServiceVec::MergeSortTopN(rowTopBuf, prevRowTopUpper, reduceOutBuff, cuS2LenVecAlign,
                                                        mergeOutCount, tmpSortBuf);
                        }
                    }
                    PipeBarrier<PIPE_V>();
                    outQueue_.FreeTensor(tmpSortBuf);
                }
            } else {
                LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
                LocalTensor<float> cacheSortedBlock =
                    SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2];
                if (hasSelectedHiTokens) {
                    if (info.actS1Size == 1) {
                        decodeStage2CacheMask_ |= static_cast<uint8_t>(1U << globalTopkUbCacheIdx);
                    }
                    Sort<float, true>(cacheSortedBlock, reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(),
                                      tmpSortBuf, cuS2LenVecAlign / 32);
                } else {
                    InitSortOutBuf(cacheSortedBlock, s2BaseSize_ * 2);
                }
                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    bool skipEmptyDecodeMerge = (info.actS1Size == 1 && decodeStage2CacheMask_ == 0);
                    if (!skipEmptyDecodeMerge) {
                        LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2];
                        if (info.s2Idx - blockS2StartIdx_ < 4) {
                            MrgBasicBlock(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], tt,
                                          static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                        } else {
                            if (globalTopkUbCacheIdx > 0) {
                                MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                                PipeBarrier<PIPE_V>();
                                DataCopy(SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf,
                                         (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                            }
                            PipeBarrier<PIPE_V>();
                            SparseTopK(globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                       SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf, BASE_TOPK,
                                       s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                        }
                    }
                    decodeStage2CacheMask_ = 0;
                }
                PipeBarrier<PIPE_V>();
                outQueue_.FreeTensor(tmpSortBuf);
            }

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;

            if (needCopyOutGm) {
                LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
                LocalTensor<uint32_t> idxULocal = valueULocal.template ReinterpretCast<uint32_t>()[BASE_TOPK];
                ExtractIndex(idxULocal, globalTopkUb_[innerS1Idx * BASE_TOPK * 2].template ReinterpretCast<uint32_t>(),
                             BASE_TOPK);
                PipeBarrier<PIPE_V>();
                InitSortOutBuf(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], BASE_TOPK * 2);
                outQueue_.EnQue<float>(valueULocal);
                valueULocal = outQueue_.DeQue<float>();
                LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
                LIServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                      idxULocal1, constInfo_.sparseCount);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                outQueue_.FreeTensor(valueULocal);
            } else if (needCopyWsGm) {
                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK +
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * ldMergeParamNum_ +
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * ldMergeParamNum_ +
                                       (ldS1Offset + innerS1Idx) * 2 * ldMergeParamNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(0));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount));
                bool isTailReduce = blockS2StartIdx_ == 0;
                if (isTailReduce) {
                    wsInfoOffset += ldMergeParamNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                PipeBarrier<PIPE_ALL>();
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                PipeBarrier<PIPE_ALL>();
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }

    if (LAYOUT_T == LI_LAYOUT::BSND) {
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.s1Size - info.actS1Size;
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.s1Size + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::MergeStage2TokenTopK()
{
    int32_t cubeNum = GetBlockNum();
    int32_t curCubeId = blockId_ / 2;

    int64_t isS2End;
    int64_t s1Idx;
    uint32_t acc_list_num = 0;
    int64_t needMerge;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    LocalTensor<float> curValueIdxUb = ldMergeInputBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldMergeTmpBuf_.Get<float>();
    LocalTensor<int64_t> paramSlotLocal = paramBuf_.Get<int64_t>();

    uint32_t s1MergeStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = curCubeId * s1BaseSize_ * 2 * ldMergeParamNum_;
    bool foundActiveSlot = false;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        int64_t slotInfoOffset = paramGmCoreOffset + innerS1Idx * 2 * ldMergeParamNum_ + ldMergeParamNum_;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(paramSlotLocal, vec1ParamGm[slotInfoOffset],
                    {1, static_cast<uint16_t>(ldMergeParamNum_ * sizeof(int64_t)), 0, 0}, {true, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        needMerge = paramSlotLocal.GetValue(0);
        if (needMerge == 1) {
            foundActiveSlot = true;
            s1MergeStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1MergeStartIdx;
            s1ProcNum++;
        } else if (foundActiveSlot) {
            break;
        }
    }
    if (s1ProcNum == 0) {
        return;
    }

    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1MergeStartIdx = s1MergeStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1MergeStartIdx; innerS1Idx < s1MergeStartIdx + s1VecNum; innerS1Idx++) {
        acc_list_num = 0;
        valueOffset = 0;

        wsInfoOffset = curCubeId * s1BaseSize_ * 2 * ldMergeParamNum_ + innerS1Idx * 2 * ldMergeParamNum_ + ldMergeParamNum_;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(paramSlotLocal, vec1ParamGm[wsInfoOffset],
                    {1, static_cast<uint16_t>(ldMergeParamNum_ * sizeof(int64_t)), 0, 0}, {true, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        int64_t baseBN2Idx = paramSlotLocal.GetValue(5);
        s1Idx = paramSlotLocal.GetValue(6);
        outOffset = paramSlotLocal.GetValue(8);
        wsOffset = curCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        PipeBarrier<PIPE_V>();
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;
        uint64_t mergedCubeMask = (curCubeId < 64) ? (1ULL << curCubeId) : 0ULL;
        bool hasEndPartial = false;
        constexpr int32_t LD_MERGE_METADATA_RETRY_TIMES = 1;
        for (int32_t retry = 0; retry < LD_MERGE_METADATA_RETRY_TIMES; ++retry) {
            bool foundNewPartial = false;
            for (int32_t tmpCubeId = 0; tmpCubeId < cubeNum; ++tmpCubeId) {
                if (tmpCubeId < 64 && ((mergedCubeMask >> tmpCubeId) & 1ULL) != 0) {
                    continue;
                }
                int32_t matchedSlotIdx = -1;
                int32_t matchedHalfIdx = -1;
                int32_t slotIdx = static_cast<int32_t>(innerS1Idx);
                for (int32_t halfIdx = 0; halfIdx < 2; ++halfIdx) {
                    if (tmpCubeId == curCubeId && slotIdx == static_cast<int32_t>(innerS1Idx) && halfIdx == 1) {
                        continue;
                    }
                    wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * ldMergeParamNum_ +
                                   slotIdx * 2 * ldMergeParamNum_ + halfIdx * ldMergeParamNum_;
                    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                    DataCopyPad(paramSlotLocal, vec1ParamGm[wsInfoOffset],
                                {1, static_cast<uint16_t>(ldMergeParamNum_ * sizeof(int64_t)), 0, 0}, {true, 0, 0, 0});
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    needMerge = paramSlotLocal.GetValue(0);
                    int64_t curBN2Idx = paramSlotLocal.GetValue(5);
                    int64_t curS1Idx = paramSlotLocal.GetValue(6);
                    if (needMerge == 1 && curBN2Idx == baseBN2Idx && curS1Idx == s1Idx) {
                        matchedSlotIdx = slotIdx;
                        matchedHalfIdx = halfIdx;
                        isS2End = paramSlotLocal.GetValue(4);
                        break;
                    }
                }
                if (matchedSlotIdx < 0) {
                    continue;
                }
                if (tmpCubeId < 64) {
                    mergedCubeMask |= (1ULL << tmpCubeId);
                }
                foundNewPartial = true;
                hasEndPartial = hasEndPartial || (isS2End == 1);
                wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                           matchedSlotIdx * 2 * 2 * BASE_TOPK + matchedHalfIdx * 2 * BASE_TOPK;
                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                            {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0},
                            {true, 0, 0, 0});
                valueOffset += 2 * BASE_TOPK;
                acc_list_num++;

                if (acc_list_num == ldMergeListNum_) {
                    AscendC::MrgSort4Info params;
                    params.elementLengths[0] = BASE_TOPK;
                    params.elementLengths[1] = BASE_TOPK;
                    params.elementLengths[2] = BASE_TOPK;
                    params.elementLengths[3] = BASE_TOPK;
                    params.ifExhaustedSuspension = true;
                    params.validBit = 0b1111;
                    params.repeatTimes = 1;

                    AscendC::MrgSortSrcList<float> srcList;
                    srcList.src1 = curValueIdxUb[0];
                    srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                    srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                    srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                    MrgSort(tmpUb, srcList, params);
                    PipeBarrier<PIPE_V>();
                    DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                    PipeBarrier<PIPE_V>();
                    acc_list_num = 1;
                    valueOffset = 2 * BASE_TOPK;
                }
            }
            if (hasEndPartial && !foundNewPartial) {
                break;
            }
            PipeBarrier<PIPE_ALL>();
        }

        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> outValueUb = ldMergeOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldMergeOutIdxBuf_.Get<uint32_t>();

        Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                    {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
}
} // namespace LIKernel
#endif // LIGHTNING_INDEXER_HI_CACHED_ARCH32_SERVICE_VECTOR_H
