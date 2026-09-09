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
 * \file lightning_indexer_service_cube.h
 * \brief Stage1 key-mean QK and Stage2 token QK, with GM or direct-UB output.
 */
#ifndef LIGHTNING_INDEXER_HI_CACHED_ARCH35_SERVICE_CUBE_H
#define LIGHTNING_INDEXER_HI_CACHED_ARCH35_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../lightning_indexer_common.h"

namespace LIKernel {
using namespace LICommon;
template <typename LIT>
class LIMatmul {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;

    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitStage2QkUb(TPipe *pipe);
    __aicore__ inline void InitGlobalTensors(
        const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
        const GlobalTensor<K_T> &keyMeanGm, const GlobalTensor<Q_T> &queryGm,
        const GlobalTensor<float> &qkWorkspaceGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitPipelineEvents();
    __aicore__ inline void DrainPipelineEvents();
    __aicore__ inline void ComputeQkTile(const LICommon::RunInfo &runInfo, bool reluPre = false);
    __aicore__ inline void ComputeStage1QkTile(const LICommon::RunInfo &runInfo, uint32_t localBlockNum);
    __aicore__ inline void LoadQueryTile(const LICommon::RunInfo &runInfo);
    __aicore__ inline void ReleaseQueryTile(const LICommon::RunInfo &runInfo);
    template <bool SHARE_QUERY_PAIR = false>
    __aicore__ inline void ComputeStage2BlockPairToUb(const LICommon::RunInfo &runInfo,
                                                     const GlobalTensor<int32_t> &blockIndices,
                                                     uint64_t blockOffset);
    template <bool TO_UB = false, uint32_t QUERY_COUNT = 1>
    __aicore__ inline void ComputeStage2QkRange(
        const LICommon::RunInfo &runInfo, uint64_t s2SrcBaseOffset, uint64_t s2ProcessSize,
        uint64_t s2DstBaseOffset, uint64_t s2DstStride, uint64_t queryMOffset);

    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;         // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID0;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID0;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 128;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t FP16_BLOCK_CUBE = 16;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    static constexpr AscendC::FixpipeConfig STAGE2_QK_UB_CONFIG = {AscendC::CO2Layout::ROW_MAJOR, true};

    __aicore__ inline void CopyQkToGm(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                        uint64_t s2L0RealSize, const LICommon::RunInfo &runInfo, bool reluPre);
    __aicore__ inline void CopyQkToGmWithStride(
        uint64_t s1gGmOffset, uint64_t s2DstGmOffset, uint64_t s1gL0RealSize,
        uint64_t s2L0RealSize, uint64_t s2DstStride, const LICommon::RunInfo &runInfo, bool reluPre);
    __aicore__ inline void CopyQkToUb(uint64_t mSize, uint64_t nSize, const LICommon::RunInfo &runInfo,
                                    uint32_t nOffset = 0, bool shareQueryPair = false);
    __aicore__ inline void ComputeQkToL0C(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                        bool useUnitFlag = true);
    __aicore__ inline void LoadKeyToL0B(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void LoadQueryToL0A(uint64_t s1gL1Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize);
    __aicore__ inline void LoadQueryToL1(uint64_t s1gL1RealSize, uint64_t s1gL1Offset, const LICommon::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset, const LICommon::RunInfo &runInfo);
    __aicore__ inline void LoadKeyMeanToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                         const LICommon::RunInfo &runInfo);
    __aicore__ inline void LoadPagedKeyToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset, const LICommon::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<K_T> keyMeanGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<float> qkWorkspaceGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;
    TBuf<TPosition::VECCALC> stage2QkBuf_;
    LocalTensor<float> stage2QkUb_;
    bool stage2QkToUb_ = false;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t queryL1Mte1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    ConstInfo constInfo_;

private:
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
};

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitStage2QkUb(TPipe *pipe)
{
    // No Cube-side UB was allocated during Stage1. AIV resets its pipe before
    // allocating this same region, so both sides address UB offset zero.
    pipe->InitBuffer(stage2QkBuf_, Stage2QkUb::BufferBytes(constInfo_.gSize));
    stage2QkUb_ = stage2QkBuf_.Get<float>();
    stage2QkToUb_ = true;
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitGlobalTensors(
    const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
    const GlobalTensor<K_T> &keyMeanGm, const GlobalTensor<Q_T> &queryGm,
    const GlobalTensor<float> &qkWorkspaceGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    keyMeanGm_ = keyMeanGm;
    queryGm_ = queryGm;
    qkWorkspaceGm_ = qkWorkspaceGm;
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComputeQkTile(const LICommon::RunInfo &runInfo, bool reluPre)
{
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;
        if (PAGE_ATTENTION) {
            LoadPagedKeyToL1(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
        } else {
            LoadKeyToL1(s2L1RealSize, s2GmOffset, runInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
            uint64_t s1gL1RealSize =
                s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                queryL1Mte2BufIdx_++;
                queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                LoadQueryToL1(s1gL1RealSize, s1gGmOffset, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            } else {
                queryL1Mte1BufIdx_ =
                    queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
            }
            for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t s2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1RealSize; s1gL1Offset += M_BASIC_BLOCK_L0) {
                    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t s1gL0RealSize =
                        s1gL1Offset + M_BASIC_BLOCK_L0 > s1gL1RealSize ? s1gL1RealSize - s1gL1Offset : M_BASIC_BLOCK_L0;
                    LoadQueryToL0A(s1gL1Offset, s1gL1RealSize, s1gL0RealSize);
                    LoadKeyToL0B(s2L1Offset, s2L1RealSize, s2L0RealSize);

                    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                    ComputeQkToL0C(s1gL0RealSize, s2L0RealSize);

                    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                    CopyQkToGm(s1gGmOffset + s1gL1Offset, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize, runInfo,
                               reluPre);
                    l0BufIdx_++;
                }
            }
            if (s2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
            }
        }

        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadQueryTile(const LICommon::RunInfo &runInfo)
{
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
        uint64_t s1gL1RealSize =
            s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
        queryL1Mte2BufIdx_++;
        queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
        WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
        LoadQueryToL1(s1gL1RealSize, s1gGmOffset, runInfo);
        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ReleaseQueryTile(const LICommon::RunInfo &runInfo)
{
    uint64_t queryChunkNum = CeilDiv(static_cast<uint64_t>(runInfo.actMBaseSize), M_BASIC_BLOCK);
    for (uint64_t chunkIdx = 0; chunkIdx < queryChunkNum; ++chunkIdx) {
        uint64_t queryBufIdx = queryL1Mte2BufIdx_ - (queryChunkNum - 1 - chunkIdx);
        SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryBufIdx % QUERY_BUF_NUM);
    }
}

template <typename LIT>
template <bool SHARE_QUERY_PAIR>
__aicore__ inline void LIMatmul<LIT>::ComputeStage2BlockPairToUb(
    const LICommon::RunInfo &runInfo, const GlobalTensor<int32_t> &blockIndices, uint64_t blockOffset)
{
    constexpr uint32_t PAIR_TOKENS = 2 * Stage2QkUb::TILE_TOKENS;
    // BF16 [256,128] occupies all 64 KiB of L0B. Borrow both existing
    // L1 K slots and both L0B credits; no K split or extra allocation.
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    for (uint32_t block = 0; block < 2; ++block) {
        int32_t logicalBlock = blockIndices.GetValue(blockOffset + block);
        if constexpr (SHARE_QUERY_PAIR) {
            logicalBlock >>= 2;
        }
        uint32_t tokenStart = static_cast<uint32_t>(logicalBlock) * Stage2QkUb::TILE_TOKENS;
        if (logicalBlock < 0 || tokenStart >= runInfo.actS2Size) {
            continue;
        }
        uint64_t physicalBlock = blkTableGm_.GetValue(
            runInfo.bIdx * constInfo_.maxBlockNumPerBatch + logicalBlock);
        uint64_t keyOffset = physicalBlock * constInfo_.kCacheBlockSize *
                             constInfo_.kHeadNum * constInfo_.headDim;
        Nd2NzParams copy;
        copy.ndNum = 1;
        copy.nValue = Min(Stage2QkUb::TILE_TOKENS, runInfo.actS2Size - tokenStart);
        copy.dValue = constInfo_.headDim;
        copy.srcDValue = constInfo_.headDim;
        copy.dstNzC0Stride = PAIR_TOKENS;
        copy.dstNzNStride = 1;
        copy.srcNdMatrixStride = 0;
        copy.dstNzMatrixStride = 0;
        DataCopy(keyL1_[block * Stage2QkUb::TILE_TOKENS * FP16_BLOCK_CUBE], keyGm_[keyOffset], copy);
    }
    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
    uint32_t mSize = constInfo_.gSize * (SHARE_QUERY_PAIR ? 2U : 1U);
    LoadQueryToL0A(0, mSize, mSize);
    LoadData2DParamsV2 load;
    load.mStartPosition = 0;
    load.kStartPosition = 0;
    load.mStep = PAIR_TOKENS / BLOCK_CUBE;
    load.kStep = constInfo_.headDim / FP16_BLOCK_CUBE;
    load.srcStride = PAIR_TOKENS / BLOCK_CUBE;
    load.dstStride = PAIR_TOKENS / BLOCK_CUBE;
    load.ifTranspose = false;
    LoadData(keyL0_, keyL1_, load);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
    MmadParams params;
    params.m = mSize;
    params.n = PAIR_TOKENS;
    params.k = constInfo_.headDim;
    params.cmatrixInitVal = true;
    params.cmatrixSource = false;
    params.unitFlag = 0;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], keyL0_, params);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    // Keep the established two 128-token UB slots and reduction order.
    CopyQkToUb(mSize, Stage2QkUb::TILE_TOKENS, runInfo, 0, SHARE_QUERY_PAIR);
    LICommon::RunInfo second = runInfo;
    ++second.loop;
    CopyQkToUb(mSize, Stage2QkUb::TILE_TOKENS, second, Stage2QkUb::TILE_TOKENS, SHARE_QUERY_PAIR);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
    ++l0BufIdx_;
}

template <typename LIT>
template <bool TO_UB, uint32_t QUERY_COUNT>
__aicore__ inline void LIMatmul<LIT>::ComputeStage2QkRange(
    const LICommon::RunInfo &runInfo, uint64_t s2SrcBaseOffset, uint64_t s2ProcessSize, uint64_t s2DstBaseOffset,
    uint64_t s2DstStride, uint64_t queryMOffset)
{
    uint64_t queryChunkNum = CeilDiv(static_cast<uint64_t>(runInfo.actMBaseSize), M_BASIC_BLOCK);
    uint64_t queryChunkIdx = queryMOffset / M_BASIC_BLOCK;
    uint64_t queryChunkBase = queryChunkIdx * M_BASIC_BLOCK;
    uint64_t queryChunkRealSize =
        queryChunkBase + M_BASIC_BLOCK > runInfo.actMBaseSize ? runInfo.actMBaseSize - queryChunkBase : M_BASIC_BLOCK;
    uint64_t queryOffsetInChunk = queryMOffset - queryChunkBase;
    uint64_t queryRealSize =
        Min(static_cast<uint64_t>(constInfo_.gSize * QUERY_COUNT), queryChunkRealSize - queryOffsetInChunk);
    if (queryRealSize == 0) {
        return;
    }
    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_ - (queryChunkNum - 1 - queryChunkIdx);

    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;
        if (PAGE_ATTENTION) {
            LoadPagedKeyToL1(s2L1RealSize, s2SrcBaseOffset + s2GmOffset, runInfo);
        } else {
            LoadKeyToL1(s2L1RealSize, s2GmOffset, runInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
            uint64_t s2L0RealSize =
                s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
            LoadQueryToL0A(queryOffsetInChunk, queryChunkRealSize, queryRealSize);
            LoadKeyToL0B(s2L1Offset, s2L1RealSize, s2L0RealSize);

            SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
            WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

            if constexpr (TO_UB) {
                WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
            }
            ComputeQkToL0C(queryRealSize, s2L0RealSize, !TO_UB);

            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

            if constexpr (TO_UB) {
                CopyQkToUb(queryRealSize, s2L0RealSize, runInfo, 0, QUERY_COUNT == 2);
                SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
            } else {
                CopyQkToGmWithStride(queryMOffset, s2DstBaseOffset + s2GmOffset + s2L1Offset, queryRealSize,
                            s2L0RealSize, s2DstStride, runInfo, true);
            }
            l0BufIdx_++;
        }

        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadKeyToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const LICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t keyGmOffset = runInfo.tensorKeyOffset + (s2GmOffset + s2L1Offset) * constInfo_.headDim;
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;

        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE));
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * BLOCK_CUBE :
                             s2L1Offset * BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadKeyMeanToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                         const LICommon::RunInfo &runInfo)
{
    uint64_t stage1BaseBlock = runInfo.stage1BaseBlockIdx;
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t blockOffsetInTile = s2GmOffset + s2L1Offset;
        if (blockOffsetInTile >= runInfo.stage1LocalBlockNum) {
            ++s2L1Offset;
            continue;
        }

        uint64_t logicalBlockId = stage1BaseBlock + blockOffsetInTile;
        uint64_t maxMte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                   s2L1RealSize - s2L1Offset :
                                   S2_BASIC_BLOCK_L0 - s2L1Offset;
        maxMte2Size = LICommon::Min(maxMte2Size,
                                    static_cast<uint64_t>(runInfo.stage1LocalBlockNum) - blockOffsetInTile);

        uint64_t phyBlockId = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + logicalBlockId);
        uint64_t s2Mte2Size = 1;
        while (s2Mte2Size < maxMte2Size) {
            uint64_t nextLogicalBlockId = logicalBlockId + s2Mte2Size;
            uint64_t nextPhyBlockId =
                blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + nextLogicalBlockId);
            if (nextPhyBlockId != phyBlockId + s2Mte2Size) {
                break;
            }
            ++s2Mte2Size;
        }

        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE));
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        uint64_t keyGmOffset = (phyBlockId * constInfo_.kHeadNum + runInfo.n2Idx) * constInfo_.headDim;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * BLOCK_CUBE :
                             s2L1Offset * BLOCK_CUBE)],
                 keyMeanGm_[keyGmOffset], nd2nzPara);
        s2L1Offset += s2Mte2Size;
    }
}

// blkNum, blkSize, N2, D
template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadPagedKeyToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const LICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                   constInfo_.kCacheBlockSize * constInfo_.kHeadNum * constInfo_.headDim +
                               s2BlkOffset * constInfo_.headDim;
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset :
                                                                              s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE));
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * BLOCK_CUBE :
                             s2L1Offset * BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComputeStage1QkTile(const LICommon::RunInfo &runInfo, uint32_t localBlockNum)
{
    if (localBlockNum == 0) {
        return;
    }
    uint64_t s2ProcessSize = localBlockNum;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;
        LoadKeyMeanToL1(s2L1RealSize, s2GmOffset, runInfo);

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
            uint64_t s1gL1RealSize =
                s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                queryL1Mte2BufIdx_++;
                queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                LoadQueryToL1(s1gL1RealSize, s1gGmOffset, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            } else {
                queryL1Mte1BufIdx_ =
                    queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
            }
            for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t s2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1RealSize; s1gL1Offset += M_BASIC_BLOCK_L0) {
                    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t s1gL0RealSize =
                        s1gL1Offset + M_BASIC_BLOCK_L0 > s1gL1RealSize ? s1gL1RealSize - s1gL1Offset : M_BASIC_BLOCK_L0;
                    LoadQueryToL0A(s1gL1Offset, s1gL1RealSize, s1gL0RealSize);
                    LoadKeyToL0B(s2L1Offset, s2L1RealSize, s2L0RealSize);

                    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                    ComputeQkToL0C(s1gL0RealSize, s2L0RealSize);

                    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                    CopyQkToGm(s1gGmOffset + s1gL1Offset, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize, runInfo,
                         false);
                    l0BufIdx_++;
                }
            }
            if (s2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
            }
        }

        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
}

// batch, s1, n2, g, d
template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadQueryToL1(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                  const LICommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize;
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE);
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             queryGm_[runInfo.tensorQueryOffset + s1gGmOffset * constInfo_.headDim], nd2nzPara);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadQueryToL0A(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                   uint64_t s1gL0RealSize)
{
    LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
    loadDataParams.kStartPosition = 0;
    loadDataParams.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadDataParams.kStep = CeilDiv(constInfo_.headDim, FP16_BLOCK_CUBE);
    loadDataParams.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
    loadDataParams.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadDataParams.ifTranspose = false;
    LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], loadDataParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadKeyToL0B(uint64_t s2L1Offset, uint64_t s2L1RealSize,
                                                 uint64_t s2L0RealSize)
{
    LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
    loadDataParams.kStartPosition = 0;
    loadDataParams.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadDataParams.kStep = CeilDiv(constInfo_.headDim, FP16_BLOCK_CUBE);
    loadDataParams.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadDataParams.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadDataParams.ifTranspose = false;
    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET], loadDataParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComputeQkToL0C(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                   bool useUnitFlag)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = useUnitFlag ? 0b11 : 0;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::CopyQkToGm(
    uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
    uint64_t s2L0RealSize, const LICommon::RunInfo &runInfo, bool reluPre)
{
    CopyQkToGmWithStride(s1gGmOffset, s2GmOffset, s1gL0RealSize, s2L0RealSize,
                        runInfo.actualSingleProcessSInnerSizeAlign, runInfo, reluPre);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::CopyQkToGmWithStride(
    uint64_t s1gGmOffset, uint64_t s2DstGmOffset, uint64_t s1gL0RealSize,
    uint64_t s2L0RealSize, uint64_t s2DstStride, const LICommon::RunInfo &runInfo, bool reluPre)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = s2DstStride;
    intriParams.srcStride = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0b11;
    intriParams.reluPre = reluPre ? 1 : 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(qkWorkspaceGm_[(runInfo.loop % 2) * constInfo_.mBaseSizeAlign * constInfo_.s2BaseSize +
                                  s1gGmOffset * intriParams.dstStride + s2DstGmOffset],
                      cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], intriParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::CopyQkToUb(
    uint64_t mSize, uint64_t nSize, const LICommon::RunInfo &runInfo, uint32_t nOffset, bool shareQueryPair)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
    params.mSize = CeilAlign(mSize, BLOCK_CUBE);
    params.nSize = CeilAlign(nSize, BLOCK_CUBE);
    params.srcStride = params.mSize;
    params.dstStride = Stage2QkUb::TILE_TOKENS;
    params.dualDstCtl = shareQueryPair ? 1 : 0;
    params.subBlockId = 0;
    // C310 dual-destination Fixpipe cannot apply ReLU. Both direct-UB paths
    // apply it in AIV before weighting, as native LI does.
    params.reluEn = false;
    // Match native LI's explicit M/FIX events, not legacy unitFlag signalling.
    params.unitFlag = 0;
    uint32_t offset = (runInfo.loop % Stage2QkUb::BUFFER_NUM) *
                      constInfo_.gSize * Stage2QkUb::TILE_TOKENS;
    Fixpipe<float, float, STAGE2_QK_UB_CONFIG>(stage2QkUb_[offset],
                                  cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET + nOffset * params.mSize], params);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitPipelineEvents()
{
    if (stage2QkToUb_) {
        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT);
        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
    }
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::DrainPipelineEvents()
{
    if (stage2QkToUb_) {
        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT);
        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
    }
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}
} // namespace LIKernel
#endif // LIGHTNING_INDEXER_HI_CACHED_ARCH35_SERVICE_CUBE_H
