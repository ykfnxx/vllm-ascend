/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "kernel_operator.h"

using namespace AscendC;

namespace optiling {
// Keep this ABI mirror in sync with op_host/scatter_nd_update_mean_tiling.h.
struct ScatterNdUpdateMeanTilingData {
    uint32_t numUpdates;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t kvHeads;
    uint32_t numBlocks;
    uint32_t updateCacheInKernel;
    float invBlockSize;
};
} // namespace optiling

namespace {

template <typename T, typename IndexT>
class ScatterNdUpdateMeanKernel {
public:
    __aicore__ inline void Init(GM_ADDR flatKeyCache, GM_ADDR indices, GM_ADDR updates, GM_ADDR keyMean,
                                const optiling::ScatterNdUpdateMeanTilingData *tilingData)
    {
        flatKeyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(flatKeyCache));
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IndexT *>(indices));
        keyMeanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(keyMean));

        numUpdates_ = tilingData->numUpdates;
        headDim_ = tilingData->headDim;
        blockSize_ = tilingData->blockSize;
        kvHeads_ = tilingData->kvHeads;
        numBlocks_ = tilingData->numBlocks;
        updateCacheInKernel_ = tilingData->updateCacheInKernel != 0U;
        invBlockSize_ = tilingData->invBlockSize;
        useContiguousBlockMean_ = (blockSize_ == 128U && headDim_ == 128U && kvHeads_ == 1U);

        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(updates));
        pipe_.InitBuffer(inputBuf_, AlignUp32(headDim_ * sizeof(T)));
        pipe_.InitBuffer(inputFloatBuf_, AlignUp32(headDim_ * sizeof(float)));
        pipe_.InitBuffer(meanFloatBuf_, AlignUp32(headDim_ * sizeof(float)));
        pipe_.InitBuffer(meanCastBuf_, AlignUp32(headDim_ * sizeof(T)));
        if (useContiguousBlockMean_) {
            pipe_.InitBuffer(blockInputBuf_, AlignUp32(blockSize_ * headDim_ * sizeof(T)));
            pipe_.InitBuffer(blockFloatBuf_, AlignUp32(blockSize_ * headDim_ * sizeof(float)));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        for (uint32_t i = coreIdx; i < numUpdates_; i += coreNum) {
            uint32_t block = 0;
            uint32_t head = 0;
            uint32_t offset = 0;
            if (!DecodeTouchedRow(i, block, head, offset)) {
                continue;
            }
            if (updateCacheInKernel_) {
                CopyUpdateToCache(i, block, head, offset);
            }
            if (offset + 1U == blockSize_) {
                uint32_t startIdx = 0;
                if (!updateCacheInKernel_ && TryFindFullBlockRunStart(i, block, head, startIdx)) {
                    ComputeBlockMeanFromUpdates(startIdx, block, head);
                } else {
                    ComputeBlockMean(block, head);
                }
            }
        }
    }

private:
    __aicore__ inline void DecodeFlatRow(int64_t flatRow, uint32_t &block, uint32_t &head, uint32_t &offset) const
    {
        const uint32_t flatRowU32 = static_cast<uint32_t>(flatRow);
        head = flatRowU32 % kvHeads_;
        const uint32_t slot = flatRowU32 / kvHeads_;
        block = slot / blockSize_;
        offset = slot - block * blockSize_;
    }

    __aicore__ inline bool DecodeTouchedRow(uint32_t updateIdx, uint32_t &block, uint32_t &head, uint32_t &offset)
    {
        int64_t flatRow = static_cast<int64_t>(indicesGm_.GetValue(updateIdx));
        if (flatRow < 0) {
            return false;
        }
        DecodeFlatRow(flatRow, block, head, offset);
        if (block >= numBlocks_ || head >= kvHeads_) {
            return false;
        }
        return true;
    }

    __aicore__ inline void ComputeBlockMean(uint32_t block, uint32_t head)
    {
        if (useContiguousBlockMean_) {
            ComputeContiguousBlockMeanFromCache(block, head);
            return;
        }
        ComputeBlockMeanFromCacheRows(block, head);
    }

    __aicore__ inline void CopyUpdateToCache(uint32_t updateIdx, uint32_t block, uint32_t head, uint32_t offset)
    {
        LocalTensor<T> input = inputBuf_.Get<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headDim_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        uint64_t dstRow = (static_cast<uint64_t>(block) * blockSize_ + offset) * kvHeads_ + head;
        DataCopyPad(input, updatesGm_[static_cast<uint64_t>(updateIdx) * headDim_], copyParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopyPad(flatKeyCacheGm_[dstRow * headDim_], input, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void ComputeBlockMeanFromCacheRows(uint32_t block, uint32_t head)
    {
        LocalTensor<T> input = inputBuf_.Get<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> meanFloat = meanFloatBuf_.Get<float>();

        Duplicate(meanFloat, 0.0f, headDim_);
        PipeBarrier<PIPE_V>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = headDim_ * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadParams padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        for (uint32_t s = 0; s < blockSize_; ++s) {
            uint64_t row = (static_cast<uint64_t>(block) * blockSize_ + s) * kvHeads_ + head;
            DataCopyPad(input, flatKeyCacheGm_[row * headDim_], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(inputFloat, input, RoundMode::CAST_NONE, headDim_);
            PipeBarrier<PIPE_V>();
            Add(meanFloat, meanFloat, inputFloat, headDim_);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }

        StoreMean(block, head, meanFloat, copyParams, eventIdVToMte3);
    }

    __aicore__ inline void ComputeContiguousBlockMeanFromCache(uint32_t block, uint32_t head)
    {
        (void)head;
        LocalTensor<T> blockInput = blockInputBuf_.Get<T>();
        LocalTensor<float> blockFloat = blockFloatBuf_.Get<float>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = blockSize_ * headDim_ * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadParams padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        uint64_t blockBaseRow = static_cast<uint64_t>(block) * blockSize_;
        DataCopyPad(blockInput, flatKeyCacheGm_[blockBaseRow * headDim_], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Cast(blockFloat, blockInput, RoundMode::CAST_NONE, blockSize_ * headDim_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

        ReduceContiguousBlock(blockFloat);

        copyParams.blockLen = headDim_ * sizeof(T);
        StoreMean(block, head, blockFloat, copyParams, eventIdVToMte3);
    }

    __aicore__ inline void ComputeBlockMeanFromUpdates(uint32_t startIdx, uint32_t block, uint32_t head)
    {
        if (useContiguousBlockMean_) {
            ComputeContiguousBlockMeanFromUpdates(startIdx, block, head);
            return;
        }
        ComputeBlockMeanFromUpdateRows(startIdx, block, head);
    }

    __aicore__ inline void ComputeBlockMeanFromUpdateRows(uint32_t startIdx, uint32_t block, uint32_t head)
    {
        LocalTensor<T> input = inputBuf_.Get<T>();
        LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
        LocalTensor<float> meanFloat = meanFloatBuf_.Get<float>();

        Duplicate(meanFloat, 0.0f, headDim_);
        PipeBarrier<PIPE_V>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = headDim_ * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadParams padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        (void)block;
        (void)head;
        for (uint32_t s = 0; s < blockSize_; ++s) {
            DataCopyPad(input, updatesGm_[static_cast<uint64_t>(startIdx + s) * headDim_], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(inputFloat, input, RoundMode::CAST_NONE, headDim_);
            PipeBarrier<PIPE_V>();
            Add(meanFloat, meanFloat, inputFloat, headDim_);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }

        StoreMean(block, head, meanFloat, copyParams, eventIdVToMte3);
    }

    __aicore__ inline void ComputeContiguousBlockMeanFromUpdates(uint32_t startIdx, uint32_t block, uint32_t head)
    {
        (void)head;
        LocalTensor<T> blockInput = blockInputBuf_.Get<T>();
        LocalTensor<float> blockFloat = blockFloatBuf_.Get<float>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = blockSize_ * headDim_ * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadParams padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        DataCopyPad(blockInput, updatesGm_[static_cast<uint64_t>(startIdx) * headDim_], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Cast(blockFloat, blockInput, RoundMode::CAST_NONE, blockSize_ * headDim_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

        ReduceContiguousBlock(blockFloat);

        copyParams.blockLen = headDim_ * sizeof(T);
        StoreMean(block, head, blockFloat, copyParams, eventIdVToMte3);
    }

    __aicore__ inline void StoreMean(uint32_t block, uint32_t head, LocalTensor<float> meanFloat,
                                     DataCopyParams &copyParams, event_t eventIdVToMte3)
    {
        Muls(meanFloat, meanFloat, invBlockSize_, headDim_);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> meanCast = meanCastBuf_.Get<T>();
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            Cast(meanCast, meanFloat, RoundMode::CAST_RINT, headDim_);
        } else {
            Cast(meanCast, meanFloat, RoundMode::CAST_ROUND, headDim_);
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        copyParams.blockLen = headDim_ * sizeof(T);
        uint64_t meanOffset = (static_cast<uint64_t>(block) * kvHeads_ + head) * headDim_;
        DataCopyPad(keyMeanGm_[meanOffset], meanCast, copyParams);
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

    __aicore__ inline void ReduceContiguousBlock(LocalTensor<float> blockFloat)
    {
        for (uint32_t stride = blockSize_ >> 1U; stride > 0U; stride >>= 1U) {
            Add(blockFloat, blockFloat, blockFloat[stride * headDim_], stride * headDim_);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline bool TryFindFullBlockRunStart(uint32_t updateIdx, uint32_t block, uint32_t head,
                                                    uint32_t &startIdx)
    {
        if (updateIdx + 1U < blockSize_) {
            return false;
        }
        startIdx = updateIdx + 1U - blockSize_;
        if (startIdx + blockSize_ > numUpdates_) {
            return false;
        }
        return IsFullBlockRun(startIdx, block, head);
    }

    __aicore__ inline bool IsFullBlockRun(uint32_t startIdx, uint32_t block, uint32_t head)
    {
        if (startIdx + blockSize_ > numUpdates_) {
            return false;
        }
        for (uint32_t s = 0; s < blockSize_; ++s) {
            uint32_t runBlock = 0;
            uint32_t runHead = 0;
            uint32_t runOffset = 0;
            if (!DecodeTouchedRow(startIdx + s, runBlock, runHead, runOffset) ||
                runBlock != block || runHead != head || runOffset != s) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint32_t AlignUp32(uint32_t size) const
    {
        constexpr uint32_t align = 32U;
        return (size + align - 1U) / align * align;
    }

    GlobalTensor<T> flatKeyCacheGm_;
    GlobalTensor<IndexT> indicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> keyMeanGm_;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> inputBuf_;
    TBuf<QuePosition::VECCALC> inputFloatBuf_;
    TBuf<QuePosition::VECCALC> meanFloatBuf_;
    TBuf<QuePosition::VECCALC> meanCastBuf_;
    TBuf<QuePosition::VECCALC> blockInputBuf_;
    TBuf<QuePosition::VECCALC> blockFloatBuf_;
    uint32_t numUpdates_ = 0;
    uint32_t headDim_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t numBlocks_ = 0;
    bool updateCacheInKernel_ = false;
    float invBlockSize_ = 0.0f;
    bool useContiguousBlockMean_ = false;
};

} // namespace
