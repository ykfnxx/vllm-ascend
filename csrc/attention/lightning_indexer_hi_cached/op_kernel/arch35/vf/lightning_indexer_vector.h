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
 * \file lightning_indexer_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_HI_CACHED_ARCH35_VF_VECTOR_H
#define LIGHTNING_INDEXER_HI_CACHED_ARCH35_VF_VECTOR_H

#include "kernel_operator.h"

namespace LIServiceVec {
using namespace AscendC;

constexpr int32_t NEG_INF = 0xFF800000;
constexpr int32_t INVALID_INDEX = -1;
constexpr uint8_t B32_VEC_ELM_NUM = 64;
constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;

template <typename T>
__aicore__ inline void CopyQkAndWeights(
    LocalTensor<float> &qkUb, LocalTensor<T> &weightsUb, GlobalTensor<float> &qkGm,
    GlobalTensor<T> &weightsGm, int64_t qkGmOffset, int64_t weightsGmOffset,
    int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams qkCopyParams;
    qkCopyParams.blockCount = groupInner;
    qkCopyParams.blockLen = s2Inner * sizeof(float);
    qkCopyParams.srcStride = 0;
    qkCopyParams.dstStride = mmUbStride;
    qkCopyParams.rsv = 0;
    AscendC::DataCopyPad(qkUb, qkGm[qkGmOffset], qkCopyParams, padParams);

    AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams weightsCopyParams;
    weightsCopyParams.blockCount = 1;
    weightsCopyParams.blockLen = groupInner * sizeof(T);
    weightsCopyParams.srcStride = 0;
    weightsCopyParams.dstStride = 0;
    weightsCopyParams.rsv = 0;
    AscendC::DataCopyPad(weightsUb, weightsGm[weightsGmOffset], weightsCopyParams, padTParams);
}

template <typename T>
__aicore__ inline void CopyQkAndWeightsWithStride(
    LocalTensor<float> &qkUb, LocalTensor<T> &weightsUb, GlobalTensor<float> &qkGm,
    GlobalTensor<T> &weightsGm, int64_t qkGmOffset, int64_t weightsGmOffset,
    int64_t groupInner, int64_t copyWidth, int64_t srcRowStride)
{
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams qkCopyParams;
    qkCopyParams.blockCount = groupInner;
    qkCopyParams.blockLen = copyWidth * sizeof(float);
    // GM source stride uses bytes; UB destination stride uses 32B blocks.
    qkCopyParams.srcStride = static_cast<uint16_t>((srcRowStride - copyWidth) * sizeof(float));
    qkCopyParams.dstStride = 0;
    qkCopyParams.rsv = 0;
    AscendC::DataCopyPad(qkUb, qkGm[qkGmOffset], qkCopyParams, padParams);

    AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams weightsCopyParams;
    weightsCopyParams.blockCount = 1;
    weightsCopyParams.blockLen = groupInner * sizeof(T);
    weightsCopyParams.srcStride = 0;
    weightsCopyParams.dstStride = 0;
    weightsCopyParams.rsv = 0;
    AscendC::DataCopyPad(weightsUb, weightsGm[weightsGmOffset], weightsCopyParams, padTParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount)
{
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = copyCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, copyParams);
}


template <typename T>
__aicore__ inline void AccumulateWeightedHeads(
    const LocalTensor<float> &reduceCacheBuf, LocalTensor<float> &qkUb,
    LocalTensor<float> &weightsUb, LocalTensor<T> &weightsTUb, LocalTensor<float> &tmpBuff,
    int64_t groupInner, int64_t s2Inner, int32_t outerGidx)
{
    // cast bfloat16_t to float
    if constexpr (!IsSameType<T, float>::value) {
        AscendC::Cast(weightsUb, weightsTUb, RoundMode::CAST_NONE, groupInner);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    AscendC::Brcb(tmpBuff, weightsUb, LICommon::CeilDiv(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], qkUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(qkUb[i * s2Inner], qkUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM], countPerRepeat,
                         repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, qkUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}


__aicore__ inline uint64_t FloorPowerOfTwo(uint64_t value)
{
    if (value <= CONST_TWO) {
        return value;
    } else {
        const uint64_t pow = 63 - clz(value);
        return (1 << pow);
    }
}


__aicore__ inline void ReduceHeads(const LocalTensor<float> &srcTensor, LocalTensor<float> &dstTensor,
                                  int32_t rNum, int32_t aNum)
{
    if (rNum == 1) {
        AscendC::Adds<float>(dstTensor, srcTensor, 0, aNum);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FloorPowerOfTwo(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        AscendC::Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > CONST_TWO) {
        nowRows = nowRows / CONST_TWO;
        AscendC::Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    AscendC::PipeBarrier<PIPE_V>();
}

// Fold the first level before spilling: ((h+16)+h) + ((h+24)+(h+8)).
// This preserves the FP32 tree while halving partial UB traffic for G=32.
__simd_vf__ inline void Stage2WeightedReduce32VF(__ubuf__ float *out,
                                               __ubuf__ float *partial,
                                               __ubuf__ float *qk, __ubuf__ float *weights)
{
    using namespace AscendC::MicroAPI;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t column = 0; column < 128; column += 64) {
#pragma clang loop unroll(disable)
        for (uint16_t head = 0; head < 8; ++head) {
            RegTensor<float> a, b, c, d, weight;
            LoadAlign(a, qk + head * 128 + column);
            Relu(a, a, mask);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, weights + head);
            Mul(a, a, weight, mask);
            LoadAlign(b, qk + (head + 16) * 128 + column);
            Relu(b, b, mask);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, weights + head + 16);
            Mul(b, b, weight, mask);
            Add(a, b, a, mask);
            LoadAlign(c, qk + (head + 8) * 128 + column);
            Relu(c, c, mask);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, weights + head + 8);
            Mul(c, c, weight, mask);
            LoadAlign(d, qk + (head + 24) * 128 + column);
            Relu(d, d, mask);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, weights + head + 24);
            Mul(d, d, weight, mask);
            Add(c, d, c, mask);
            Add(a, a, c, mask);
            StoreAlign(partial + head * 128 + column, a, mask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> values[8];
#pragma unroll
        for (uint16_t head = 0; head < 8; ++head) {
            LoadAlign(values[head], partial + head * 128 + column);
        }
#pragma unroll
        for (uint16_t head = 0; head < 4; ++head) {
            Add(values[head], values[head], values[head + 4], mask);
        }
        Add(values[0], values[0], values[2], mask);
        Add(values[1], values[1], values[3], mask);
        Add(values[0], values[0], values[1], mask);
        StoreAlign(out + column, values[0], mask);
    }
}

// Match AccumulateWeightedHeads/ReduceHeads: separate FP32 Mul/Add, accumulation across
// 16-head groups, then the same 8/4/2/1 tree. BF16 TopK conversion stays later.
template <bool FIRST_GROUP, bool LAST_GROUP, bool APPLY_RELU>
__simd_vf__ inline void Stage2WeightedReduceVF(__ubuf__ float *out,
                                             __ubuf__ float *partial,
                                             __ubuf__ float *qk,
                                             __ubuf__ float *weights,
                                             uint16_t width)
{
    using namespace AscendC::MicroAPI;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> values[16];
    RegTensor<float> weight;
    RegTensor<float> previous;
    for (uint16_t column = 0; column < width; column += 64) {
#pragma unroll
        for (uint16_t head = 0; head < 16; ++head) {
            LoadAlign(values[head], qk + head * width + column);
            if constexpr (APPLY_RELU) {
                Relu(values[head], values[head], mask);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(weight, weights + head);
            Mul(values[head], values[head], weight, mask);
            if constexpr (!FIRST_GROUP) {
                LoadAlign(previous, partial + head * width + column);
                Add(values[head], values[head], previous, mask);
            }
        }
        if constexpr (LAST_GROUP) {
#pragma unroll
            for (uint16_t head = 0; head < 8; ++head) {
                Add(values[head], values[head], values[head + 8], mask);
            }
#pragma unroll
            for (uint16_t head = 0; head < 4; ++head) {
                Add(values[head], values[head], values[head + 4], mask);
            }
            Add(values[0], values[0], values[2], mask);
            Add(values[1], values[1], values[3], mask);
            Add(values[0], values[0], values[1], mask);
            StoreAlign(out + column, values[0], mask);
        } else {
#pragma unroll
            for (uint16_t head = 0; head < 16; ++head) {
                StoreAlign(partial + head * width + column, values[head], mask);
            }
        }
    }
}

template <bool APPLY_RELU = false, typename T>
__aicore__ inline void Stage2WeightedReduce(const AscendC::LocalTensor<float> &out,
                                           const AscendC::LocalTensor<float> &partial,
                                           const AscendC::LocalTensor<float> &qk,
                                           const AscendC::LocalTensor<float> &weights,
                                           const AscendC::LocalTensor<T> &weightsInput,
                                           uint16_t width, bool firstGroup, bool lastGroup)
{
    if constexpr (!AscendC::IsSameType<T, float>::value) {
        AscendC::Cast(weights, weightsInput, AscendC::RoundMode::CAST_NONE, 16);
        AscendC::PipeBarrier<PIPE_V>();
    }
    auto outAddr = (__ubuf__ float *)out.GetPhyAddr();
    auto partialAddr = (__ubuf__ float *)partial.GetPhyAddr();
    auto qkAddr = (__ubuf__ float *)qk.GetPhyAddr();
    auto weightsAddr = (__ubuf__ float *)weights.GetPhyAddr();
    if (firstGroup) {
        if (lastGroup) {
            Stage2WeightedReduceVF<true, true, APPLY_RELU>(outAddr, partialAddr, qkAddr, weightsAddr, width);
        } else {
            Stage2WeightedReduceVF<true, false, APPLY_RELU>(outAddr, partialAddr, qkAddr, weightsAddr, width);
        }
    } else if (lastGroup) {
        Stage2WeightedReduceVF<false, true, APPLY_RELU>(outAddr, partialAddr, qkAddr, weightsAddr, width);
    } else {
        Stage2WeightedReduceVF<false, false, APPLY_RELU>(outAddr, partialAddr, qkAddr, weightsAddr, width);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

} // namespace LIServiceVec
#endif // LIGHTNING_INDEXER_HI_CACHED_ARCH35_VF_VECTOR_H
