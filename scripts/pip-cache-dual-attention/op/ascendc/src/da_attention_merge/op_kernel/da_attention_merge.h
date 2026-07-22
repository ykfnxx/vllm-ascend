#ifndef DA_ATTENTION_MERGE_H
#define DA_ATTENTION_MERGE_H

#include "kernel_operator.h"

namespace DaAttentionMergeNs {
using namespace AscendC;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = 8;
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = 64;
constexpr uint32_t REPEAT_STRIDE_UP_BOUND = 256;
constexpr uint32_t SYNC_EVENT_ID = 0;
constexpr float DENOM_EPS = 1.0e-12f;

template <typename T> __aicore__ inline T CeilAlign(T num, T rnd)
{
    return rnd == 0 ? 0 : ((num + rnd - 1) / rnd * rnd);
}

template <typename OUT_T>
class DaAttentionMergeKernel {
public:
    __aicore__ inline DaAttentionMergeKernel(TPipe *pipe, const DaAttentionMergeTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR prevAttentionOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum,
                                GM_ADDR curAttentionOut, GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum,
                                GM_ADDR attentionOut)
    {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        headDim_ = tiling_->headDim;
        headDimAlign_ = tiling_->headDimAlign;
        rowBlock_ = tiling_->rowBlock;
        rowAlign_ = CeilAlign(rowBlock_, FP32_BLOCK_ELEMENT_NUM);
        tileElements_ = rowBlock_ * headDimAlign_;

        prevAttentionGm_.SetGlobalBuffer((__gm__ OUT_T *)prevAttentionOut);
        curAttentionGm_.SetGlobalBuffer((__gm__ OUT_T *)curAttentionOut);
        attentionOutGm_.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
        prevMaxGm_.SetGlobalBuffer((__gm__ float *)prevSoftmaxMax);
        prevSumGm_.SetGlobalBuffer((__gm__ float *)prevSoftmaxSum);
        curMaxGm_.SetGlobalBuffer((__gm__ float *)curSoftmaxMax);
        curSumGm_.SetGlobalBuffer((__gm__ float *)curSoftmaxSum);

        pipe_->InitBuffer(prevAttentionBuf_, tileElements_ * sizeof(OUT_T));
        pipe_->InitBuffer(curAttentionBuf_, tileElements_ * sizeof(OUT_T));
        pipe_->InitBuffer(outputBuf_, tileElements_ * sizeof(OUT_T));
        pipe_->InitBuffer(prevFloatBuf_, tileElements_ * sizeof(float));
        pipe_->InitBuffer(curFloatBuf_, tileElements_ * sizeof(float));
        pipe_->InitBuffer(stateBuf_, rowAlign_ * 7 * sizeof(float) + rowAlign_ * FP32_BLOCK_ELEMENT_NUM * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        const uint32_t startRow = blockIdx_ * tiling_->rowsPerCore;
        const uint32_t endRow = Min(startRow + tiling_->rowsPerCore, tiling_->totalRows);
        for (uint32_t row = startRow; row < endRow; row += rowBlock_) {
            const uint32_t rowCount = Min(rowBlock_, endRow - row);
            ProcessRows(row, rowCount);
        }
    }

private:
    template <typename A, typename B> __aicore__ inline A Min(A lhs, B rhs)
    {
        return lhs < static_cast<A>(rhs) ? lhs : static_cast<A>(rhs);
    }

    __aicore__ inline void PrepareWeights(uint32_t rowStart, uint32_t rowCount)
    {
        const uint32_t rowCountAlign = CeilAlign(rowCount, FP32_BLOCK_ELEMENT_NUM);
        LocalTensor<float> state = stateBuf_.Get<float>();
        prevMaxUb_ = state;
        prevSumUb_ = state[rowAlign_];
        curMaxUb_ = state[rowAlign_ * 2];
        curSumUb_ = state[rowAlign_ * 3];
        denomUb_ = state[rowAlign_ * 4];
        prevWeightUb_ = state[rowAlign_ * 5];
        curWeightUb_ = state[rowAlign_ * 6];
        rowWeightBrcbUb_ = state[rowAlign_ * 7];

        DataCopyExtParams stateCopyParams{1, static_cast<uint32_t>(rowCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> statePadParams{true, 0, static_cast<uint8_t>(rowCountAlign - rowCount), 0.0f};
        DataCopyPad(prevMaxUb_, prevMaxGm_[rowStart], stateCopyParams, statePadParams);
        DataCopyPad(prevSumUb_, prevSumGm_[rowStart], stateCopyParams, statePadParams);
        DataCopyPad(curMaxUb_, curMaxGm_[rowStart], stateCopyParams, statePadParams);
        DataCopyPad(curSumUb_, curSumGm_[rowStart], stateCopyParams, statePadParams);
        SetFlag<HardEvent::MTE2_V>(SYNC_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(SYNC_EVENT_ID);
        pipe_barrier(PIPE_V);

        Max(denomUb_, prevMaxUb_, curMaxUb_, rowCountAlign);
        pipe_barrier(PIPE_V);
        Sub(prevWeightUb_, prevMaxUb_, denomUb_, rowCountAlign);
        Sub(curWeightUb_, curMaxUb_, denomUb_, rowCountAlign);
        pipe_barrier(PIPE_V);
        Exp(prevWeightUb_, prevWeightUb_, rowCountAlign);
        Exp(curWeightUb_, curWeightUb_, rowCountAlign);
        pipe_barrier(PIPE_V);
        Mul(prevWeightUb_, prevWeightUb_, prevSumUb_, rowCountAlign);
        Mul(curWeightUb_, curWeightUb_, curSumUb_, rowCountAlign);
        pipe_barrier(PIPE_V);
        Add(denomUb_, prevWeightUb_, curWeightUb_, rowCountAlign);
        pipe_barrier(PIPE_V);
        Maxs(denomUb_, denomUb_, DENOM_EPS, rowCountAlign);
        pipe_barrier(PIPE_V);
        Div(prevWeightUb_, prevWeightUb_, denomUb_, rowCountAlign);
        Div(curWeightUb_, curWeightUb_, denomUb_, rowCountAlign);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ProcessRows(uint32_t rowStart, uint32_t rowCount)
    {
        PrepareWeights(rowStart, rowCount);

        const uint32_t attenOffset = rowStart * headDim_;
        const uint32_t tileCount = rowCount * headDimAlign_;
        DataCopyExtParams attentionCopyParams{static_cast<uint16_t>(rowCount),
                                              static_cast<uint32_t>(headDim_ * sizeof(OUT_T)), 0, 0, 0};
        DataCopyPadExtParams<OUT_T> attentionPadParams{false, 0, 0, 0};

        LocalTensor<OUT_T> prevAttentionUb = prevAttentionBuf_.Get<OUT_T>();
        LocalTensor<OUT_T> curAttentionUb = curAttentionBuf_.Get<OUT_T>();
        LocalTensor<OUT_T> outputUb = outputBuf_.Get<OUT_T>();
        LocalTensor<float> prevFloatUb = prevFloatBuf_.Get<float>();
        LocalTensor<float> curFloatUb = curFloatBuf_.Get<float>();

        DataCopyPad(prevAttentionUb, prevAttentionGm_[attenOffset], attentionCopyParams, attentionPadParams);
        DataCopyPad(curAttentionUb, curAttentionGm_[attenOffset], attentionCopyParams, attentionPadParams);
        SetFlag<HardEvent::MTE2_V>(SYNC_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(SYNC_EVENT_ID);
        pipe_barrier(PIPE_V);
        Cast(prevFloatUb, prevAttentionUb, RoundMode::CAST_NONE, tileCount);
        Cast(curFloatUb, curAttentionUb, RoundMode::CAST_NONE, tileCount);
        pipe_barrier(PIPE_V);

        Brcb(rowWeightBrcbUb_, prevWeightUb_, (rowCount + 7) / 8, {1, 8});
        pipe_barrier(PIPE_V);
        RowMuls(prevFloatUb, prevFloatUb, rowWeightBrcbUb_, rowCount, headDimAlign_, headDim_);
        pipe_barrier(PIPE_V);

        Brcb(rowWeightBrcbUb_, curWeightUb_, (rowCount + 7) / 8, {1, 8});
        pipe_barrier(PIPE_V);
        RowMuls(curFloatUb, curFloatUb, rowWeightBrcbUb_, rowCount, headDimAlign_, headDim_);
        pipe_barrier(PIPE_V);
        Add(prevFloatUb, prevFloatUb, curFloatUb, tileCount);
        pipe_barrier(PIPE_V);

        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) {
            Cast(outputUb, prevFloatUb, RoundMode::CAST_RINT, tileCount);
        } else {
            Cast(outputUb, prevFloatUb, RoundMode::CAST_NONE, tileCount);
        }
        pipe_barrier(PIPE_V);

        SetFlag<HardEvent::V_MTE3>(SYNC_EVENT_ID);
        WaitFlag<HardEvent::V_MTE3>(SYNC_EVENT_ID);
        DataCopyPad(attentionOutGm_[attenOffset], outputUb, attentionCopyParams);
        SetFlag<HardEvent::MTE3_V>(SYNC_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(SYNC_EVENT_ID);
    }

    __aicore__ inline void RowMuls(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
    {
        uint32_t dLoop = actualColumnCount / FP32_REPEAT_ELEMENT_NUM;
        uint32_t dRemain = actualColumnCount % FP32_REPEAT_ELEMENT_NUM;

        if (columnCount < REPEAT_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
            BinaryRepeatParams repeatParams;
            repeatParams.src0BlkStride = 1;
            repeatParams.src1BlkStride = 0;
            repeatParams.dstBlkStride = 1;
            repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
            repeatParams.src1RepStride = 1;
            repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;

            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, FP32_REPEAT_ELEMENT_NUM, dealRowCount, repeatParams);
                offset += FP32_REPEAT_ELEMENT_NUM;
            }
            if (dRemain > 0) {
                Mul(dstUb[dLoop * FP32_REPEAT_ELEMENT_NUM], src0Ub[dLoop * FP32_REPEAT_ELEMENT_NUM], src1Ub,
                    dRemain, dealRowCount, repeatParams);
            }
        } else {
            BinaryRepeatParams repeatParams;
            repeatParams.src0RepStride = 8;
            repeatParams.src0BlkStride = 1;
            repeatParams.src1RepStride = 0;
            repeatParams.src1BlkStride = 0;
            repeatParams.dstRepStride = 8;
            repeatParams.dstBlkStride = 1;

            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * FP32_BLOCK_ELEMENT_NUM],
                    FP32_REPEAT_ELEMENT_NUM, dLoop, repeatParams);
                if (dRemain > 0) {
                    Mul(dstUb[i * columnCount + dLoop * FP32_REPEAT_ELEMENT_NUM],
                        src0Ub[i * columnCount + dLoop * FP32_REPEAT_ELEMENT_NUM],
                        src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dRemain, 1, repeatParams);
                }
            }
        }
    }

    TPipe *pipe_;
    const DaAttentionMergeTilingData *tiling_;
    uint32_t blockIdx_ = 0;
    uint32_t headDim_ = 0;
    uint32_t headDimAlign_ = 0;
    uint32_t rowBlock_ = 0;
    uint32_t rowAlign_ = 0;
    uint32_t tileElements_ = 0;

    GlobalTensor<OUT_T> prevAttentionGm_;
    GlobalTensor<OUT_T> curAttentionGm_;
    GlobalTensor<OUT_T> attentionOutGm_;
    GlobalTensor<float> prevMaxGm_;
    GlobalTensor<float> prevSumGm_;
    GlobalTensor<float> curMaxGm_;
    GlobalTensor<float> curSumGm_;

    TBuf<TPosition::VECCALC> prevAttentionBuf_;
    TBuf<TPosition::VECCALC> curAttentionBuf_;
    TBuf<TPosition::VECCALC> outputBuf_;
    TBuf<TPosition::VECCALC> prevFloatBuf_;
    TBuf<TPosition::VECCALC> curFloatBuf_;
    TBuf<TPosition::VECCALC> stateBuf_;

    LocalTensor<float> prevMaxUb_;
    LocalTensor<float> prevSumUb_;
    LocalTensor<float> curMaxUb_;
    LocalTensor<float> curSumUb_;
    LocalTensor<float> denomUb_;
    LocalTensor<float> prevWeightUb_;
    LocalTensor<float> curWeightUb_;
    LocalTensor<float> rowWeightBrcbUb_;
};

} // namespace DaAttentionMergeNs

#endif // DA_ATTENTION_MERGE_H
