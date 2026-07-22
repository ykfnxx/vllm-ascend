/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef lightning_indexer_decode_SERVICE_VECTOR_H
#define lightning_indexer_decode_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_decode_common.h"
#include "lightning_indexer_decode_vector.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;

constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t S2_BASE_SIZE = 512;
constexpr uint32_t G_SIZE = 64;
constexpr uint32_t GROUP_INNER = 16;
constexpr uint32_t OUTER_G = G_SIZE / GROUP_INNER;
constexpr uint32_t TOPK_PAIR_FLOATS = BASE_TOPK * 2;
constexpr uint32_t SORTED_CHUNK_FLOATS = 4 * S2_BASE_SIZE * 2;
constexpr uint32_t SORT_BUFFER_FLOATS = TOPK_PAIR_FLOATS + SORTED_CHUNK_FLOATS;

template <typename LIT>
class LIVector {
public:
    using K_T = typename LIT::keyType;
    using MM1_OUT_T = float;

    __aicore__ inline void ProcessVec(const LICommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t outputOffset);

private:
    static constexpr uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    static constexpr uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> sortedBasicBlock_;
};

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitBuffers(TPipe *pipe)
{
    if ((GetBlockIdx() & 1U) != 0) {
        return;
    }

    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + GROUP_INNER * S2_BASE_SIZE * sizeof(float);
    uint32_t outputScratchSize = TOPK_PAIR_FLOATS * sizeof(float);
    uint32_t outQueueSize = Max(reduceCacheSize, outputScratchSize);

    pipe->InitBuffer(inQueue_, 2,
                     GROUP_INNER * S2_BASE_SIZE * sizeof(float) + S2_BASE_SIZE * sizeof(float));
    pipe->InitBuffer(outQueue_, 1, outQueueSize);
    pipe->InitBuffer(sortOutBuf_, SORT_BUFFER_FLOATS * sizeof(float));
    pipe->InitBuffer(indexBuf_, S2_BASE_SIZE * sizeof(int32_t));
    pipe->InitBuffer(reduceOutBuf_, S2_BASE_SIZE * 2 * sizeof(float));
    pipe->InitBuffer(brcBuf_, GROUP_INNER * 8 * sizeof(float));

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    sortedBasicBlock_ = globalTopkUb_[TOPK_PAIR_FLOATS];
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, S2_BASE_SIZE);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void
LIVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_T> weightsGm,
                                    GlobalTensor<int32_t> indiceOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::CleanInvalidOutput(int64_t outputOffset)
{
    LocalTensor<float> outputLocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> indexLocal = outputLocal.template ReinterpretCast<int32_t>();
    Duplicate(indexLocal, LICommon::ConstInfo::INVALID_IDX, BASE_TOPK);
    outQueue_.EnQue<float>(outputLocal);
    outputLocal = outQueue_.DeQue<float>();
    LIServiceVec::CopyOut(indiceOutGm[outputOffset], indexLocal, BASE_TOPK);
    outQueue_.FreeTensor(outputLocal);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessVec(const LICommon::RunInfo &info)
{
    if ((GetBlockIdx() & 1U) != 0) {
        return;
    }

    int32_t chunkStart = info.s2Idx * S2_BASE_SIZE;
    int32_t chunkSize = info.actualSingleProcessSInnerSize;
    int64_t mmGmOffset = (info.loop % 2) * (G_SIZE * S2_BASE_SIZE);
    int64_t weightGmOffset = static_cast<int64_t>(info.bIdx) * G_SIZE;

    if (info.isFirstS2InnerLoop) {
        InitSortOutBuf(globalTopkUb_, TOPK_PAIR_FLOATS);
    }

    int32_t mmUbStride = (S2_BASE_SIZE - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> reduceOutInner = reduceOutBuff[S2_BASE_SIZE];
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();

    for (int32_t outerGidx = 0; outerGidx < OUTER_G; ++outerGidx) {
        LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
        LocalTensor<float> weightsInUb = mmInUb[GROUP_INNER * S2_BASE_SIZE];
        LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>();
        weightsInTUb = weightsInTUb[GROUP_INNER];
        LIServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm,
                             mmGmOffset + outerGidx * GROUP_INNER * info.actualSingleProcessSInnerSizeAlign,
                             weightGmOffset + outerGidx * GROUP_INNER, GROUP_INNER,
                             info.actualSingleProcessSInnerSizeAlign, mmUbStride);

        inQueue_.EnQue<float>(mmInUb);
        mmInUb = inQueue_.DeQue<float>();
        weightsInUb = mmInUb[GROUP_INNER * S2_BASE_SIZE];
        LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, weightsInTUb,
                              brcBuf, GROUP_INNER, S2_BASE_SIZE, outerGidx);
        inQueue_.FreeTensor(mmInUb);
    }

    LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, GROUP_INNER, S2_BASE_SIZE);
    outQueue_.FreeTensor(reduceCacheBuf);

    LocalTensor<float> sortScoreUb = reduceOutBuff;
    LocalTensor<float> sortIndiceUb = reduceOutBuff[S2_BASE_SIZE];
    PipeBarrier<PIPE_V>();
    Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, S2_BASE_SIZE);
    PipeBarrier<PIPE_V>();
    Adds(sortScoreUb, reduceOutInner, 0.0f, chunkSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
    if (chunkSize != S2_BASE_SIZE) {
        Duplicate(sortIndiceUbInt, -1, S2_BASE_SIZE);
        PipeBarrier<PIPE_V>();
    }
    Adds(sortIndiceUbInt, globalTopkIndice_, chunkStart, chunkSize);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
    uint32_t cachedChunkIdx = info.s2Idx % 4;
    Sort<float, true>(sortedBasicBlock_[cachedChunkIdx * S2_BASE_SIZE * 2], reduceOutBuff,
                      sortIndiceUbInt.template ReinterpretCast<uint32_t>(), tmpSortBuf, S2_BASE_SIZE / 32);
    if (cachedChunkIdx == 3 || info.isLastS2InnerLoop) {
        if (info.s2Idx < 4) {
            MrgBasicBlock(globalTopkUb_, sortedBasicBlock_, static_cast<int64_t>(cachedChunkIdx + 1), S2_BASE_SIZE);
        } else {
            if (cachedChunkIdx > 0) {
                MrgBasicBlock(tmpSortBuf, sortedBasicBlock_, static_cast<int64_t>(cachedChunkIdx + 1), S2_BASE_SIZE);
                PipeBarrier<PIPE_V>();
                DataCopy(sortedBasicBlock_, tmpSortBuf, (cachedChunkIdx + 1) * S2_BASE_SIZE * 2);
            }
            PipeBarrier<PIPE_V>();
            SparseTopK(globalTopkUb_, sortedBasicBlock_, tmpSortBuf, BASE_TOPK,
                       S2_BASE_SIZE * (cachedChunkIdx + 1));
        }
    }
    PipeBarrier<PIPE_V>();
    outQueue_.FreeTensor(tmpSortBuf);

    if (info.isLastS2InnerLoop) {
        LocalTensor<float> outputLocal = outQueue_.AllocTensor<float>();
        LocalTensor<uint32_t> indexLocal = outputLocal.template ReinterpretCast<uint32_t>()[BASE_TOPK];
        ExtractIndex(indexLocal, globalTopkUb_.template ReinterpretCast<uint32_t>(), BASE_TOPK);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue<float>(outputLocal);
        outputLocal = outQueue_.DeQue<float>();
        LIServiceVec::CopyOut(indiceOutGm[static_cast<int64_t>(info.bIdx) * BASE_TOPK],
                              outputLocal.template ReinterpretCast<int32_t>()[BASE_TOPK], BASE_TOPK);
        outQueue_.FreeTensor(outputLocal);
    }
}

} // namespace LIKernel
#endif // lightning_indexer_decode_SERVICE_VECTOR_H
