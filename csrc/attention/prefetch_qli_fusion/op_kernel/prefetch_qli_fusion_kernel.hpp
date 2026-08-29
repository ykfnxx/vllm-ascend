#ifndef PREFETCH_QLI_FUSION_KERNEL_HPP
#define PREFETCH_QLI_FUSION_KERNEL_HPP

// This operator keeps its Cast, reduction, arithmetic, and DataCopy helpers local.

#include "kernel_operator.h"

__aicore__ inline uint32_t PqfCeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : ((x + y - 1) / y);
}

__aicore__ inline uint32_t PqfRoundUp(uint32_t x, uint32_t y = 16)
{
    return y == 0 ? x : (x + y - 1) / y * y;
}

// bf16/half -> fp32 (CAST_NONE + PipeBarrier)
template <typename T>
__aicore__ inline void PqfCastFrom16To32(const AscendC::LocalTensor<float>& out,
                                         const AscendC::LocalTensor<T>& in, uint32_t count)
{
    AscendC::Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
}

// fp32 -> half/bf16 (half: CAST_NONE; bf16: CAST_RINT)
template <typename T>
__aicore__ inline void PqfCastFrom32To16(const AscendC::LocalTensor<T>& out,
                                         const AscendC::LocalTensor<float>& in, uint32_t count)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        AscendC::Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
    } else { // bfloat16_t
        AscendC::Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// half -> int8: clamp to [-128,127] then Cast CAST_RINT
__aicore__ inline void PqfCastFromF16ToI8(const AscendC::LocalTensor<int8_t>& out,
                                          const AscendC::LocalTensor<half>& in, half quantMin,
                                          uint32_t count)
{
    AscendC::Maxs(in, in, quantMin, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(in, in, (half)127, count);
    AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    AscendC::Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
#else
    AscendC::Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
#endif
    AscendC::PipeBarrier<PIPE_V>();
}

// fp32 -> int8：先 fp32->int16(CAST_RINT 四舍六入五成双，与 npu_dynamic_quant 一致)
// -> int16->half(CAST_NONE 整数精确) -> half->int8(CAST_RINT)。
// 关键：不能 fp32->half->int8——fp32->half 丢尾数（half 10bit mantissa）会让
// x/scale 在 .5 边界差 1 级（如 25.49->half 25.5->int8 26，正确应为 25）。
// 复用 tmpHalf 缓冲：先按 int16 视图做 fp32->int16，再按 half 视图做 int16->half。
__aicore__ inline void PqfCastFromF32ToI8(const AscendC::LocalTensor<int8_t>& out,
                                          const AscendC::LocalTensor<half>& tmpHalf,
                                          const AscendC::LocalTensor<float>& in, uint32_t count)
{
    auto tmpInt16 = tmpHalf.ReinterpretCast<int16_t>();
    AscendC::Cast(tmpInt16, in, AscendC::RoundMode::CAST_RINT, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(tmpHalf, tmpInt16, AscendC::RoundMode::CAST_NONE, count);  // int16->half 整数精确
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(tmpHalf, tmpHalf, (half)-128, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(tmpHalf, tmpHalf, (half)127, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(out, tmpHalf, AscendC::RoundMode::CAST_RINT, count);
    AscendC::PipeBarrier<PIPE_V>();
}

#endif  // PREFETCH_QLI_FUSION_KERNEL_HPP
