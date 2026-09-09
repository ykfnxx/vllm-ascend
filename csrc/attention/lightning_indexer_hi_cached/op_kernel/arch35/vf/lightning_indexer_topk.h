/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_HI_CACHED_TOPK_H
#define LIGHTNING_INDEXER_HI_CACHED_TOPK_H

#include "kernel_operator.h"
#include "vf_topk_16_gather.h"

namespace topk {
using namespace AscendC;

__simd_vf__ void BuildSortableKeysVFImpl(__ubuf__ uint16_t *keyBuf, uint16_t loopNum)
{
    AscendC::MicroAPI::MaskReg maskAll =
        AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::RegTensor<uint16_t> zero;
    AscendC::MicroAPI::RegTensor<uint16_t> allOne;
    AscendC::MicroAPI::RegTensor<uint16_t> signMask;
    AscendC::MicroAPI::RegTensor<uint16_t> nan;
    AscendC::MicroAPI::Duplicate(zero, static_cast<uint16_t>(0x0000), maskAll);
    AscendC::MicroAPI::Duplicate(allOne, static_cast<uint16_t>(0xFFFF), maskAll);
    AscendC::MicroAPI::Duplicate(signMask, static_cast<uint16_t>(0x8000), maskAll);
    AscendC::MicroAPI::Duplicate(nan, static_cast<uint16_t>(0x7FC0), maskAll);

    for (uint16_t i = 0; i < loopNum; ++i) {
        AscendC::MicroAPI::RegTensor<bfloat16_t> value;
        AscendC::MicroAPI::RegTensor<uint16_t> key;
        AscendC::MicroAPI::RegTensor<uint16_t> sign;
        AscendC::MicroAPI::RegTensor<uint16_t> xorMask;
        AscendC::MicroAPI::MaskReg nanMask;
        AscendC::MicroAPI::MaskReg negativeMask;
        AscendC::MicroAPI::LoadAlign<bfloat16_t>(
            value, reinterpret_cast<__ubuf__ bfloat16_t *>(keyBuf) + i * 128);
        auto &valueBits = reinterpret_cast<AscendC::MicroAPI::RegTensor<uint16_t> &>(value);

        // Match native A5 LI's BF16 sortable-key contract exactly.
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(nanMask, valueBits, nan, maskAll);
        AscendC::MicroAPI::Select(key, allOne, valueBits, nanMask);
        AscendC::MicroAPI::And(sign, key, signMask, maskAll);
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::GT>(negativeMask, sign, zero, maskAll);
        AscendC::MicroAPI::Select(xorMask, allOne, signMask, negativeMask);
        AscendC::MicroAPI::Xor(key, key, xorMask, maskAll);
        AscendC::MicroAPI::StoreAlign<uint16_t>(keyBuf + i * 128, key, maskAll);
    }
}

__aicore__ inline void BuildSortableKeys(const LocalTensor<uint16_t> &keys, uint32_t alignedLen)
{
    BuildSortableKeysVFImpl(reinterpret_cast<__ubuf__ uint16_t *>(keys.GetPhyAddr()),
                            static_cast<uint16_t>(alignedLen / 128));
}

__simd_vf__ void RestoreBfloat16ScoresVFImpl(__ubuf__ uint16_t *keyBuf, uint16_t loopNum)
{
    AscendC::MicroAPI::MaskReg maskAll =
        AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::RegTensor<uint16_t> zero;
    AscendC::MicroAPI::RegTensor<uint16_t> allOne;
    AscendC::MicroAPI::RegTensor<uint16_t> signMask;
    AscendC::MicroAPI::RegTensor<uint16_t> negativeNan;
    AscendC::MicroAPI::Duplicate(zero, static_cast<uint16_t>(0x0000), maskAll);
    AscendC::MicroAPI::Duplicate(allOne, static_cast<uint16_t>(0xFFFF), maskAll);
    AscendC::MicroAPI::Duplicate(signMask, static_cast<uint16_t>(0x8000), maskAll);
    AscendC::MicroAPI::Duplicate(negativeNan, static_cast<uint16_t>(0xFFC0), maskAll);

    for (uint16_t i = 0; i < loopNum; ++i) {
        AscendC::MicroAPI::RegTensor<uint16_t> key;
        AscendC::MicroAPI::RegTensor<uint16_t> sign;
        AscendC::MicroAPI::RegTensor<uint16_t> xorMask;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg positiveMask;
        AscendC::MicroAPI::LoadAlign<uint16_t>(key, keyBuf + i * 128);
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(zeroMask, key, zero, maskAll);
        AscendC::MicroAPI::Select(key, negativeNan, key, zeroMask);
        AscendC::MicroAPI::And(sign, key, signMask, maskAll);
        AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::GT>(positiveMask, sign, zero, maskAll);
        AscendC::MicroAPI::Select(xorMask, signMask, allOne, positiveMask);
        AscendC::MicroAPI::Xor(key, key, xorMask, maskAll);
        AscendC::MicroAPI::StoreAlign<bfloat16_t>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(keyBuf) + i * 128,
            reinterpret_cast<AscendC::MicroAPI::RegTensor<bfloat16_t> &>(key), maskAll);
    }
}

__aicore__ inline void RestoreBfloat16Scores(const LocalTensor<uint16_t> &keys, uint32_t alignedLen)
{
    RestoreBfloat16ScoresVFImpl(reinterpret_cast<__ubuf__ uint16_t *>(keys.GetPhyAddr()),
                                static_cast<uint16_t>(alignedLen / 128));
}

__simd_vf__ void GatherRealIndicesVFImpl(__ubuf__ int32_t *outputIndexBuf,
                                         __ubuf__ uint32_t *selectedPositionBuf,
                                         __ubuf__ int32_t *realIndexBuf,
                                         uint16_t loopNum)
{
    AscendC::MicroAPI::MaskReg maskAll =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < loopNum; ++i) {
        AscendC::MicroAPI::RegTensor<uint32_t> selectedPosition;
        AscendC::MicroAPI::RegTensor<uint32_t> realIndex;
        AscendC::MicroAPI::LoadAlign<uint32_t>(selectedPosition, selectedPositionBuf + i * 64);
        AscendC::MicroAPI::Gather(realIndex, reinterpret_cast<__ubuf__ uint32_t *>(realIndexBuf),
                                  selectedPosition, maskAll);
        AscendC::MicroAPI::StoreAlign<uint32_t>(
            reinterpret_cast<__ubuf__ uint32_t *>(outputIndexBuf + i * 64), realIndex, maskAll);
    }
}

__aicore__ inline void GatherRealIndices(const LocalTensor<int32_t> &outputIndices,
                                         const LocalTensor<uint32_t> &selectedPositions,
                                         const LocalTensor<int32_t> &realIndices,
                                         uint32_t count)
{
    GatherRealIndicesVFImpl(reinterpret_cast<__ubuf__ int32_t *>(outputIndices.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t *>(selectedPositions.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ int32_t *>(realIndices.GetPhyAddr()),
                            static_cast<uint16_t>(count / 64));
}

class LITopk {
public:
    static __aicore__ inline uint32_t GetSharedTmpBufferSize(uint32_t topK, uint32_t trunkLen)
    {
        // Keep the native LI uint16 TopK scratch layout. In particular,
        // tmpIndexLocal needs room for both the aligned TopK prefix and the
        // complete input trunk because the VF filter may append every item
        // tied with the kth value.
        uint32_t scalarSize = (3 * 256 + 64) * sizeof(uint32_t);
        uint32_t filterSize =
            (LICommon::Align(topK, 256U) + trunkLen) * sizeof(uint16_t);
        return scalarSize + filterSize;
    }

    static __aicore__ inline uint32_t GetOutputIndexBufferSize(uint32_t topK)
    {
        return LICommon::Align(topK, 256U) * sizeof(uint32_t);
    }

    __aicore__ inline void Init(uint32_t topK)
    {
        topK_ = topK;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t> &sharedTmpBuffer)
    {
        // HI cached merges tile TopK lists outside LiTopKVF, following the
        // arch32 Stage1/Stage2 flow. The native LI history-index buffers are
        // only needed by its internal multi-trunk merge and are unused here.
        histogramsLocal_ = sharedTmpBuffer;
        idxHighLocal_ = histogramsLocal_[256];
        idxLowLocal_ = idxHighLocal_[256];
        nkValueLocal_ = idxLowLocal_[256];
        tmpIndexLocal_ = nkValueLocal_[64].template ReinterpretCast<uint16_t>();
    }

    __aicore__ inline void operator()(LocalTensor<uint32_t> &selectedPositionLocal,
                                      LocalTensor<uint16_t> &selectedValueLocal,
                                      LocalTensor<uint16_t> &inputLocal,
                                      uint32_t inputLen)
    {
        topkb16gather::LiTopKVF<true>(tmpIndexLocal_, selectedValueLocal, inputLocal,
                                      histogramsLocal_, idxHighLocal_, idxLowLocal_, nkValueLocal_,
                                      topK_, inputLen);
        PipeBarrier<PIPE_V>();
        Cast(selectedPositionLocal, tmpIndexLocal_, RoundMode::CAST_NONE, topK_);
    }

private:
    LocalTensor<uint32_t> histogramsLocal_;
    LocalTensor<uint32_t> idxHighLocal_;
    LocalTensor<uint32_t> idxLowLocal_;
    LocalTensor<uint32_t> nkValueLocal_;
    LocalTensor<uint16_t> tmpIndexLocal_;
    uint32_t topK_ = 0;
};

// Native LI's multi-trunk BF16 TopK, with indices in the compact HI token
// stream. Logical token IDs are recovered only after the final selection.
class LITopkStream {
public:
    static constexpr uint32_t TOPK = 2048;
    static constexpr uint32_t TRUNK_LEN = 16384;

    static __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        return (2 * TOPK + 3 * 256 + 64) * sizeof(uint32_t) +
               (TOPK + TRUNK_LEN) * sizeof(uint16_t);
    }

    __aicore__ inline void InitBuffers(const LocalTensor<uint32_t> &buffer)
    {
        historyIndices_[0] = buffer;
        historyIndices_[1] = buffer[TOPK];
        histograms_ = buffer[2 * TOPK];
        idxHigh_ = histograms_[256];
        idxLow_ = idxHigh_[256];
        nkValue_ = idxLow_[256];
        tmpIndices_ = nkValue_[64].template ReinterpretCast<uint16_t>();
    }

    __aicore__ inline void Select(const LocalTensor<uint16_t> &inputKeys,
                                   const LocalTensor<uint16_t> &historyKeys,
                                   const LocalTensor<uint32_t> &outputIndices,
                                   uint32_t inputCount, uint32_t windowIdx, bool isLastWindow)
    {
        topkb16gather::LiTopKVF<true>(tmpIndices_, historyKeys, inputKeys,
                                     histograms_, idxHigh_, idxLow_, nkValue_, TOPK, inputCount);
        PipeBarrier<PIPE_V>();
        uint32_t nextHistory = (windowIdx + 1) % 2;
        if (windowIdx == 0) {
            Cast(historyIndices_[nextHistory], tmpIndices_, RoundMode::CAST_NONE, TOPK);
        } else {
            topkb16gather::LiTopKGatherVF(historyIndices_[nextHistory], tmpIndices_,
                                          historyIndices_[windowIdx % 2], TOPK, windowIdx * TRUNK_LEN - TOPK);
        }
        PipeBarrier<PIPE_V>();
        if (isLastWindow) {
            DataCopy(outputIndices, historyIndices_[nextHistory], TOPK);
        } else {
            DataCopy(inputKeys, historyKeys, TOPK);
        }
        PipeBarrier<PIPE_V>();
    }

private:
    LocalTensor<uint32_t> historyIndices_[2];
    LocalTensor<uint32_t> histograms_;
    LocalTensor<uint32_t> idxHigh_;
    LocalTensor<uint32_t> idxLow_;
    LocalTensor<uint32_t> nkValue_;
    LocalTensor<uint16_t> tmpIndices_;
};

__simd_vf__ void MapHiTokenIndicesVFImpl(__ubuf__ uint32_t *output,
                                          __ubuf__ uint32_t *positions,
                                          __ubuf__ uint32_t *blocks,
                                          uint32_t blockCount, uint32_t keyLen, uint16_t loops)
{
    using namespace AscendC::MicroAPI;
    MaskReg all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<uint32_t> zero;
    RegTensor<uint32_t> invalid;
    RegTensor<uint32_t> offsetMask;
    Duplicate(zero, 0U, all);
    Duplicate(invalid, 0xFFFFFFFFU, all);
    Duplicate(offsetMask, 127U, all);
    for (uint16_t i = 0; i < loops; ++i) {
        RegTensor<uint32_t> position;
        RegTensor<uint32_t> blockPos;
        RegTensor<uint32_t> blockId;
        RegTensor<uint32_t> offset;
        RegTensor<uint32_t> token;
        MaskReg validBlock;
        MaskReg validToken;
        LoadAlign(position, positions + i * 64);
        ShiftRights(blockPos, position, static_cast<int16_t>(7), all);
        Compares<uint32_t, CMPMODE::LT>(validBlock, blockPos, blockCount, all);
        // Make padding positions safe even before the masked output select.
        Select(blockPos, blockPos, zero, validBlock);
        Gather(blockId, blocks, blockPos, all);
        And(offset, position, offsetMask, all);
        ShiftLefts(token, blockId, static_cast<int16_t>(7), all);
        Add(token, token, offset, all);
        Compares<uint32_t, CMPMODE::LT>(validToken, token, keyLen, all);
        Select(token, token, invalid, validToken);
        Select(token, token, invalid, validBlock);
        StoreAlign(output + i * 64, token, all);
    }
}

__aicore__ inline void MapHiTokenIndices(const LocalTensor<int32_t> &output,
                                          const LocalTensor<uint32_t> &positions,
                                          const LocalTensor<int32_t> &blocks,
                                          uint32_t blockCount, uint32_t keyLen)
{
    MapHiTokenIndicesVFImpl(reinterpret_cast<__ubuf__ uint32_t *>(output.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t *>(positions.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t *>(blocks.GetPhyAddr()),
                            blockCount, keyLen, LITopkStream::TOPK / 64);
}
} // namespace topk

#endif // LIGHTNING_INDEXER_HI_CACHED_TOPK_H
