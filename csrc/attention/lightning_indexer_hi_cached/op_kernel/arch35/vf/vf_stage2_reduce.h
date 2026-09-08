/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef LIGHTNING_INDEXER_HI_CACHED_VF_STAGE2_REDUCE_H
#define LIGHTNING_INDEXER_HI_CACHED_VF_STAGE2_REDUCE_H

#include "kernel_operator.h"

namespace LIServiceVec {

// Match DoScale/DoReduce: separate FP32 Mul/Add, accumulation across
// 16-head groups, then the same 8/4/2/1 tree. BF16 TopK conversion stays later.
template <bool FIRST_GROUP, bool LAST_GROUP>
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

template <typename T>
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
            Stage2WeightedReduceVF<true, true>(outAddr, partialAddr, qkAddr, weightsAddr, width);
        } else {
            Stage2WeightedReduceVF<true, false>(outAddr, partialAddr, qkAddr, weightsAddr, width);
        }
    } else if (lastGroup) {
        Stage2WeightedReduceVF<false, true>(outAddr, partialAddr, qkAddr, weightsAddr, width);
    } else {
        Stage2WeightedReduceVF<false, false>(outAddr, partialAddr, qkAddr, weightsAddr, width);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

} // namespace LIServiceVec
#endif
