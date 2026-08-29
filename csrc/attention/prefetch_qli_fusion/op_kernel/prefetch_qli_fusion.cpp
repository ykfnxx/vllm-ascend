/*
 * Prefetch QLI fusion kernel.
 *
 * AIC cores execute the three matmul stages. AIV cores execute per-token
 * quantization, RMSNorm, dequantization, RoPE, and output conversion. The
 * stages exchange GM results through four global barriers. A single UB arena
 * is reused across vector phases, and GM/UB dependencies use PIPE_ALL.
 */

#include "kernel_operator.h"
#include "prefetch_qli_fusion_kernel.hpp"

#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {

using A_TYPE_MM = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
// Quantized weights use [K, N] FRACTAL_NZ storage.
using B_TYPE_MM = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t, false>;
using C_TYPE_MM  = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using BIAS_TYPE  = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

// MatmulImpl uses unit flags with multi-data loading enabled.
constexpr MatmulConfig PQF_MM_CFG{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                  0, 0, 0, 0, true};

// Quantized matmul keeps C in int32; AIV applies scale factors in fp32.
using mmT = matmul::MatmulImpl<A_TYPE_MM, B_TYPE_MM, C_TYPE_MM, BIAS_TYPE, PQF_MM_CFG>;

// The BF16 weight projection consumes contiguous [N_wk, K] rows.
using A_TYPE_MM3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using B_TYPE_MM3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;  // [N_wk,K] + isTransB
using C_TYPE_MM3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BIAS_TYPE_MM3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using mmT3 = matmul::MatmulImpl<A_TYPE_MM3, B_TYPE_MM3, C_TYPE_MM3, BIAS_TYPE_MM3, PQF_MM_CFG>;

constexpr uint32_t FP32_ELEMS_PER_32B = 8;   // 32B / 4B
constexpr uint16_t BF16_ELEMS_PER_32B = 16;  // 32B / 2B
constexpr uint32_t INT8_ELEMS_PER_32B = 32;  // 32B / 1B
constexpr uint32_t INT32_ELEMS_PER_32B = 8;  // 32B / 4B
// rank-aware alpha/beta 系数向量的 UB 容量（实际 N 由 tiling.alphaVecLen 决定，GLM-5.2 N=16）
constexpr uint32_t MAX_ALPHA_VEC = 64U;

// ============ 各 phase UB 布局（byte 偏移，全部 32B 对齐） ============

struct P1Offsets {
    uint32_t hpBf16, hpFp32, absFp32, maxFp32, tmpHalf, xqInt8;
    uint32_t alphaVecBf16, betaVecBf16, alphaVecFp32, betaVecFp32, end;
};
__aicore__ inline P1Offsets ComputeP1Offsets(uint32_t hs)
{
    const uint32_t hsB = PqfRoundUp(hs, BF16_ELEMS_PER_32B);
    const uint32_t hsF = PqfRoundUp(hs, FP32_ELEMS_PER_32B);
    const uint32_t hsI = PqfRoundUp(hs, INT8_ELEMS_PER_32B);
    const uint32_t avB = PqfRoundUp(MAX_ALPHA_VEC, BF16_ELEMS_PER_32B);
    const uint32_t avF = PqfRoundUp(MAX_ALPHA_VEC, FP32_ELEMS_PER_32B);
    P1Offsets o;
    o.hpBf16  = 0;
    o.hpFp32  = o.hpBf16 + hsB * 2;
    o.absFp32 = o.hpFp32 + hsF * 4;
    o.maxFp32 = o.absFp32 + hsF * 4;
    o.tmpHalf = o.maxFp32 + FP32_ELEMS_PER_32B * 4;
    o.xqInt8  = o.tmpHalf + hsB * 2;
    // rank-aware 系数（仅 Phase 1 使用；挂在 P1 尾部，Phase 3 RoPE 覆盖前已不再需要）
    o.alphaVecBf16 = o.xqInt8 + hsI;
    o.betaVecBf16  = o.alphaVecBf16 + avB * 2;
    o.alphaVecFp32 = o.betaVecBf16 + avB * 2;
    o.betaVecFp32  = o.alphaVecFp32 + avF * 4;
    o.end     = o.betaVecFp32 + avF * 4;
    return o;
}

struct P2Offsets {
    uint32_t qkvInt32, qkvFp32, wsQkvBf16, wsQkvFp32, qrawHalf, qrawFp32, gammaFp32, betaFp32,
        sqFp32, sumFp32, rmsFp32, outFp32, sTokRow, gammaBf16, betaBf16, qcFp32, absFp32,
        maxFp32, tmpHalf2, xcqInt8, end;
};
__aicore__ inline P2Offsets ComputeP2Offsets(uint32_t ql)
{
    const uint32_t qlB = PqfRoundUp(ql, BF16_ELEMS_PER_32B);
    const uint32_t qlF = PqfRoundUp(ql, FP32_ELEMS_PER_32B);
    const uint32_t qlI = PqfRoundUp(ql, INT8_ELEMS_PER_32B);
    P2Offsets o;
    o.qkvInt32  = 0;
    o.qkvFp32   = o.qkvInt32 + qlB * 4;             // int32 行段 [:ql]
    o.wsQkvBf16 = o.qkvFp32 + qlF * 4;
    o.wsQkvFp32 = o.wsQkvBf16 + qlB * 2;
    o.qrawHalf  = o.wsQkvFp32 + qlF * 4;
    o.qrawFp32  = o.qrawHalf + qlB * 2;
    o.gammaFp32 = o.qrawFp32 + qlF * 4;
    o.betaFp32  = o.gammaFp32 + qlF * 4;
    o.sqFp32    = o.betaFp32 + qlF * 4;
    o.sumFp32   = o.sqFp32 + qlF * 4;
    o.rmsFp32   = o.sumFp32 + FP32_ELEMS_PER_32B * 4;
    o.outFp32   = o.rmsFp32 + qlF * 4;
    o.sTokRow   = o.outFp32 + qlF * 4;
    o.gammaBf16 = o.sTokRow + FP32_ELEMS_PER_32B * 4;
    o.betaBf16  = o.gammaBf16 + qlB * 2;
    o.qcFp32    = o.betaBf16 + qlB * 2;
    o.absFp32   = o.qcFp32 + qlF * 4;
    o.maxFp32   = o.absFp32 + qlF * 4;
    o.tmpHalf2  = o.maxFp32 + FP32_ELEMS_PER_32B * 4;
    o.xcqInt8   = o.tmpHalf2 + qlB * 2;
    o.end       = o.xcqInt8 + qlI;
    return o;
}

// RoPE vector-phase UB layout.
// Processing order:
//   qfInt32 -> Cast -> X(全行 fp32) -> 反量化/round_bf16（wsQbBf16 复用为 bf16 全行中间）
//   Gather 拆偶/奇 -> aBuf/bBuf；broadcast-Mul + 矢量 Mul/Sub/Add 旋转
//   Gather 交织 -> ropeBuf（X 前 16K）；Cast -> ropeBf16（复用 aBuf 空间）
//   rope 段 DataCopyPad 写回；nope 段直接用 wsQbBf16 的 round_bf16 全行 strided 写回
// 复用（不新增 UB）：
//   qfInt32[0,32K) Cast 后死 -> aBuf[0,8K)+bBuf[8K,16K)+t3[16K,24K)+t4[24K,32K)
//     （交织后 aBuf/bBuf/t3/t4 死 -> ropeBf16[0,8K) 复用 aBuf 空间）
//   X(qfFp32)[32K,64K) gather 后 rope 段死 -> ropeBuf[0,16K)
//   wsQbBf16[64K,80K) 反量化后死 -> round_bf16(qf) 全行 bf16（nope 写回源）
//   wsQbFp32[80K,112K) 常驻（per-chunk 反量化系数，不可复用）
//   offset arrays: aOffsets(8K) + interleaveOffsets(16K)
struct P3VecOffsets {
    uint32_t qfInt32, qfFp32, wsQbBf16, wsQbFp32, cosHalf, sinHalf, cosFp32, sinFp32,
        sTok2, cosFull32, sinFull32, aOffsets, interleaveOffsets, end;
};
__aicore__ inline P3VecOffsets ComputeP3VecOffsets(uint32_t nb, uint32_t ropeElems, uint32_t halfRd)
{
    const uint32_t nbB = PqfRoundUp(nb, BF16_ELEMS_PER_32B);
    const uint32_t nbF = PqfRoundUp(nb, FP32_ELEMS_PER_32B);
    const uint32_t hrB = PqfRoundUp(halfRd, BF16_ELEMS_PER_32B);
    const uint32_t hrF = PqfRoundUp(halfRd, FP32_ELEMS_PER_32B);
    const uint32_t ropeF = PqfRoundUp(ropeElems, FP32_ELEMS_PER_32B);
    const uint32_t ropeRepFp32 = 64;  // 910B 矢量 repeat = 64 fp32 = 256B
    P3VecOffsets o;
    uint32_t cur = 0;
    o.qfInt32  = cur; cur += nbB * 4;   // 32KB（raw int32 行段）
    o.qfFp32   = cur; cur += nbF * 4;   // 32KB（X 全行 fp32）
    o.wsQbBf16 = cur; cur += nbB * 2;   // 16KB（反量化系数 bf16，后复用 round_bf16 中间）
    o.wsQbFp32 = cur; cur += nbF * 4;   // 32KB（反量化系数 fp32，常驻）
    o.cosHalf  = cur; cur += hrB * 2;
    o.sinHalf  = cur; cur += hrB * 2;
    o.cosFp32  = cur; cur += hrF * 4;
    o.sinFp32  = cur; cur += hrF * 4;
    o.sTok2    = cur; cur += FP32_ELEMS_PER_32B * 4;
    o.cosFull32 = cur; cur += ropeRepFp32 * sizeof(float);  // 256B（broadcast Mul 用）
    o.sinFull32 = cur; cur += ropeRepFp32 * sizeof(float);  // 256B
    o.aOffsets = cur; cur += ropeF * sizeof(uint32_t);      // 8KB
    o.interleaveOffsets = cur; cur += 2 * ropeF * sizeof(uint32_t);  // 16KB
    o.end = cur;
    return o;
}

// ============ GM <-> UB 搬运 helper ============

// GM -> UB 连续搬运 count 个 T 元素（MTE2 + PipeBarrier<PIPE_ALL>）。
// 关键：DAV_2201 AIC 核上 SetFlag/WaitFlag 对 V 相关事件（MTE2_V/V_MTE3/V_S/S_V）是
// no-op（asc/include/.../kernel_operator_block_sync_intf.h L28-56），必须用
// PipeBarrier<PIPE_ALL>（kernel_reg.h 2201 分支仅 PIPE_V no-op，PIPE_ALL 走 pipe_barrier）。
template <typename T>
__aicore__ inline void PqfCopyIn(LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count)
{
    uint32_t bytes = count * sizeof(T);
    if ((bytes & 31U) == 0U) {
        // 32B 对齐用基础 DataCopy（AIC 核 GM->UB 更可靠的 MTE2 通路）
        DataCopy(dst, src, count);
    } else {
        DataCopyPad(dst, src, DataCopyExtParams{1, bytes, 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, 0});
    }
    PipeBarrier<PIPE_ALL>();  // 等 MTE2 落地，vector 读才可见（AIC 核唯一可靠全同步）
}

// UB -> GM 连续搬运 count 个 T 元素（V 已就绪后 MTE3 + PIPE_ALL 全同步）
template <typename T>
__aicore__ inline void PqfCopyOut(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
{
    uint32_t bytes = count * sizeof(T);
    PipeBarrier<PIPE_ALL>();  // 等前序 V 写完成，再搬出（V->MTE3 事件在 AIC no-op，用 PIPE_ALL）
    if ((bytes & 31U) == 0U) {
        DataCopy(dst, src, count);  // 32B 对齐基础 DataCopy（UB->GM）
    } else {
        DataCopyPad(dst, src, DataCopyExtParams{1, bytes, 0, 0, 0});  // UB->GM 3 参版（4 参版仅 950）
    }
    PipeBarrier<PIPE_ALL>();  // 等 MTE3 搬完，buffer 可复用
}

// N-axis slice [nStart, nStart+nLen) assigned to one core.
// 策略（保证 SetSingleShape 的 curN ∈ {baseN, 全局尾块}，均为已验证取值）：
//   先把 n 切成完整 baseN 宽块（n/baseN 块），按「baseBlocks + 前 rem 核多 1 块」均衡连续分配；
//   全局余数 tail（n%baseN）追加到最后一个持有完整块的核上。
//   boundary 都在 baseN 的整数倍上，B/C GM 偏移天然 32B 对齐；块数不均时仅相差 1 块（均衡）。
__aicore__ inline void ComputeNSlice(uint32_t n, uint32_t baseN, uint32_t nCore, uint32_t core,
                                     uint32_t& nStart, uint32_t& nLen)
{
    uint32_t fullBlocks = baseN == 0 ? 0 : n / baseN;
    uint32_t tail = baseN == 0 ? n : n % baseN;
    if (nCore == 0 || nCore > fullBlocks) {
        // 核数超过完整块数：前 fullBlocks 个核各 1 块，其余核 0 块
        uint32_t bStart = (core < fullBlocks) ? core : fullBlocks;
        uint32_t bEnd = (core + 1U < fullBlocks) ? (core + 1U) : fullBlocks;
        nStart = bStart * baseN;
        nLen = (bEnd > bStart) ? ((bEnd - bStart) * baseN) : 0U;
        if (tail > 0U && bEnd == fullBlocks && bEnd > bStart) {
            nLen += tail;  // 全局余数块追加到最后一个持块核
        }
        return;
    }
    uint32_t baseBlocks = fullBlocks / nCore;
    uint32_t remBlocks = fullBlocks % nCore;
    uint32_t blocksPerCore = baseBlocks + ((core < remBlocks) ? 1U : 0U);
    // 本核起始块 = 之前所有核的块数（前 rem 核多 1 块）
    uint32_t prevBlocks = (core < remBlocks)
                              ? (core * (baseBlocks + 1U))
                              : (remBlocks * (baseBlocks + 1U) + (core - remBlocks) * baseBlocks);
    uint32_t bEnd = prevBlocks + blocksPerCore;
    nStart = prevBlocks * baseN;
    nLen = blocksPerCore * baseN;
    if (tail > 0U && bEnd == fullBlocks && blocksPerCore > 0U) {
        nLen += tail;  // 全局余数块追加到最后一个持块核
    }
}

class KernelPrefetchQliFusion {
public:
    __aicore__ inline KernelPrefetchQliFusion() {}

    __aicore__ inline void Init(GM_ADDR hiddenStates, GM_ADDR wqkv, GM_ADDR wsQkv,
                                GM_ADDR wqb, GM_ADDR wsQb, GM_ADDR gamma1,
                                GM_ADDR beta1, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR wkWeights, GM_ADDR alphaVec, GM_ADDR betaVec,
                                GM_ADDR qLi, GM_ADDR weights, GM_ADDR workspace,
                                const PrefetchQliFusionTilingData& tiling,
                                TPipe* pipe);

    // AIC executes matmul while AIV executes quantization, RMSNorm, and RoPE.
    // SyncAll<false>() orders the alternating stages.
    // 计算链：quant1(AIV) -> mm1(AIC) -> RMSNorm+quant2(AIV) -> mm2(AIC) -> RoPE(AIV)
    __aicore__ inline void ProcessAic(uint32_t blk);
    __aicore__ inline void ProcessAiv(uint32_t aivBlk);

private:
    __aicore__ inline void RunMatmul1(uint32_t mStart, uint32_t mLen, uint32_t nStart, uint32_t nLen);
    __aicore__ inline void RunMatmul2(uint32_t mStart, uint32_t mLen, uint32_t nStart, uint32_t nLen);
    // mm3：wk_weights_proj（BF16，N=n_head，仅 blk==0 核算全 M weights）
    __aicore__ inline void RunMatmul3();
    __aicore__ inline void ProcessChunkQuant1(uint32_t mStart, uint32_t mLen);
    __aicore__ inline void ProcessChunkRmsNorm(uint32_t mStart, uint32_t mLen);
    __aicore__ inline void ProcessChunkRoPE(uint32_t mStart, uint32_t mLen);
    // Build RoPE gather offsets after transient vector phases release the arena.
    __aicore__ inline void BuildRopeOffsets();
    __aicore__ inline float ReduceMaxRow(LocalTensor<float>& absLocal,
                                         LocalTensor<float>& maxFp32,
                                         uint32_t rowLenAligned);
    __aicore__ inline float ReduceSumRow(LocalTensor<float>& sqLocal,
                                         LocalTensor<float>& sumFp32,
                                         uint32_t rowLenAligned);

    // 从 arena 按字节偏移切出指定 dtype 的视图
    template <typename T>
    __aicore__ inline LocalTensor<T> Ub(uint32_t byteOffset)
    {
        return ubArena_[byteOffset].template ReinterpretCast<T>();
    }

    // Tiling 标量
    uint32_t tokenNum_{0};
    uint32_t usedCoreNum_{1};
    uint32_t singleM_{0};   // 每核 M 行段
    uint32_t mCore_{1};     // M 维跨核切分数
    uint32_t nCore_{1};     // N 维跨核切分数
    uint32_t baseN_{0};     // mm 逐 N-block 宽度（SetFixSplit baseN）
    uint32_t mChunk_{4};
    uint32_t hiddenSize_{0};
    uint32_t qLoraRank_{0};
    uint32_t nQkv_{0};
    uint32_t nQb_{0};
    uint32_t nHead_{0};
    uint32_t headDim_{0};
    uint32_t ropeDim_{0};
    uint32_t halfRd_{0};       // ropeDim/2
    uint32_t ropeElems_{0};    // halfRd * nHead（每个 token 的 rope 元素数）
    float alpha_{1.0f};
    float beta_{0.0f};
    float eps_{1e-6f};
    float invQLoraRank_{0};
    float invHiddenSize_{0};
    // rank-aware alpha/beta（GLM-5.2 fused）：0=标量模式，1=per-row rank-aware 向量模式
    uint32_t alphaBetaMode_{0};
    uint32_t sourceRowsBeforeGather_{0};  // R = row_rank = min(row/R, N-1)
    uint32_t alphaVecLen_{0};             // N（alpha_vec/beta_vec 长度）

    // workspace 偏移
    uint64_t wsXqOffset_{0};
    uint64_t wsStokOffset_{0};
    uint64_t wsQkvOutOffset_{0};
    uint64_t wsXcqOffset_{0};
    uint64_t wsStok2Offset_{0};
    uint64_t wsQfOffset_{0};
    uint64_t wsPredOffset_{0};  // predicted_hidden bf16 [T, hidden]（mm3 的 A 输入）
    uint64_t wsWeightsOffset_{0};  // mm3 C 输出 bf16 [T, n_head]

    TCubeTiling cubeTiling1_;
    TCubeTiling cubeTiling2_;
    TCubeTiling cubeTiling3_;  // mm3: wk_weights_proj（BF16）

    // Matmul instances are initialized once and process explicit N-axis blocks.
    mmT mm1_;
    mmT mm2_;
    mmT3 mm3_;  // wk_weights_proj（BF16）

    GlobalTensor<bfloat16_t> hiddenGm_;
    GlobalTensor<int8_t> wqkvGm_;
    GlobalTensor<bfloat16_t> wsQkvGm_;
    GlobalTensor<int8_t> wqbGm_;
    GlobalTensor<bfloat16_t> wsQbGm_;
    GlobalTensor<bfloat16_t> gamma1Gm_;
    GlobalTensor<bfloat16_t> beta1Gm_;
    GlobalTensor<bfloat16_t> cosGm_;
    GlobalTensor<bfloat16_t> sinGm_;
    GlobalTensor<bfloat16_t> alphaVecGm_;  // rank-aware 模式：alpha 系数向量 [N] bf16
    GlobalTensor<bfloat16_t> betaVecGm_;   // rank-aware 模式：beta 系数向量 [N] bf16
    GlobalTensor<bfloat16_t> wkWeightsGm_;  // wk_weights_proj 权重（已行偏移到 head_dim 起）
    GlobalTensor<bfloat16_t> wsPredGm_;     // workspace：predicted_hidden bf16 [T, hidden]
    GlobalTensor<bfloat16_t> wsWeightsGm_;  // workspace：mm3 C 输出 bf16 [T, n_head]
    GlobalTensor<bfloat16_t> qLiGm_;
    GlobalTensor<bfloat16_t> weightsGm_;    // 输出：weights bf16 [T, n_head]

    GlobalTensor<int8_t> wsXqGm_;
    GlobalTensor<float> wsStokGm_;
    GlobalTensor<int32_t> wsQkvOutGm_;
    GlobalTensor<int8_t> wsXcqGm_;
    GlobalTensor<float> wsStok2Gm_;
    GlobalTensor<int32_t> wsQfGm_;

    TPipe* pipe_{nullptr};

    __gm__ bfloat16_t* hiddenStates_{nullptr};
    __gm__ uint8_t* workspace_{nullptr};

    // VECIN supports both GM-to-UB and UB-to-GM DataCopyPad for the shared arena.
    TBuf<TPosition::VECIN> ubArenaBuf_;
    LocalTensor<uint8_t> ubArena_;
    uint32_t ubArenaBytes_{0};
};

// ============ Init ============
__aicore__ inline void KernelPrefetchQliFusion::Init(
    GM_ADDR hiddenStates, GM_ADDR wqkv, GM_ADDR wsQkv, GM_ADDR wqb, GM_ADDR wsQb,
    GM_ADDR gamma1, GM_ADDR beta1, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR wkWeights, GM_ADDR alphaVec, GM_ADDR betaVec,
    GM_ADDR qLi, GM_ADDR weights, GM_ADDR workspace,
    const PrefetchQliFusionTilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    tokenNum_ = tiling.tokenNum;
    usedCoreNum_ = tiling.usedCoreNum;
    singleM_ = tiling.singleM;
    mCore_ = tiling.mCore;
    nCore_ = tiling.nCore;
    baseN_ = tiling.baseN;
    mChunk_ = tiling.mChunk;
    hiddenSize_ = tiling.hiddenSize;
    qLoraRank_ = tiling.qLoraRank;
    nQkv_ = tiling.nQkv;
    nQb_ = tiling.nQb;
    nHead_ = tiling.nHead;
    headDim_ = tiling.headDim;
    ropeDim_ = tiling.ropeDim;
    halfRd_ = ropeDim_ / 2;
    ropeElems_ = halfRd_ * nHead_;
    alpha_ = tiling.alpha;
    beta_ = tiling.beta;
    eps_ = tiling.eps;
    invQLoraRank_ = tiling.invQloraRank;
    invHiddenSize_ = tiling.invHiddenSize;
    alphaBetaMode_ = tiling.alphaBetaMode;
    sourceRowsBeforeGather_ = tiling.sourceRowsBeforeGather;
    alphaVecLen_ = tiling.alphaVecLen;
    cubeTiling1_ = tiling.cubeTiling1;
    cubeTiling2_ = tiling.cubeTiling2;
    cubeTiling3_ = tiling.cubeTiling3;

    wsXqOffset_ = tiling.wsXqOffset;
    wsStokOffset_ = tiling.wsStokOffset;
    wsQkvOutOffset_ = tiling.wsQkvOutOffset;
    wsXcqOffset_ = tiling.wsXcqOffset;
    wsStok2Offset_ = tiling.wsStok2Offset;
    wsQfOffset_ = tiling.wsQfOffset;
    wsPredOffset_ = tiling.wsPredOffset;
    wsWeightsOffset_ = tiling.wsWeightsOffset;

    hiddenStates_ = reinterpret_cast<__gm__ bfloat16_t*>(hiddenStates);  // DEBUG
    hiddenGm_.SetGlobalBuffer(hiddenStates_, static_cast<uint64_t>(tokenNum_) * hiddenSize_);
    wqkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(wqkv));
    wsQkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wsQkv), nQkv_);
    wqbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(wqb));
    wsQbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wsQb), nQb_);
    gamma1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gamma1), qLoraRank_);
    beta1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(beta1), qLoraRank_);
    cosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos),
                           static_cast<uint64_t>(tokenNum_) * (ropeDim_ / 2));
    sinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sin),
                           static_cast<uint64_t>(tokenNum_) * (ropeDim_ / 2));
    // rank-aware 模式：系数向量直接读 GM 输入（可选输入缺失时 alphaVec 为 null，仅在 mode 1 使用）
    if (alphaBetaMode_ == 1U) {
        alphaVecGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(alphaVec), alphaVecLen_);
        betaVecGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(betaVec), alphaVecLen_);
    }
    // wk_weights_proj 权重：逻辑 [head_dim+n_head, hidden]，mm3 只消费后半 n_head 行（weights 列），
    // 行偏移在 RunMatmul3 用下标给出（与 mm1 的 wqkvGm_[bOff] 一致）
    wkWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wkWeights),
                                 static_cast<uint64_t>(headDim_ + nHead_) * hiddenSize_);
    qLiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(qLi),
                           static_cast<uint64_t>(tokenNum_) * nHead_ * headDim_);
    weightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(weights),
                               static_cast<uint64_t>(tokenNum_) * nHead_);

    __gm__ uint8_t* ws = reinterpret_cast<__gm__ uint8_t*>(workspace);
    workspace_ = ws;  // DEBUG
    wsXqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(ws + wsXqOffset_),
                            static_cast<uint64_t>(tokenNum_) * hiddenSize_);
    wsStokGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + wsStokOffset_), tokenNum_);
    wsQkvOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ws + wsQkvOutOffset_),
                                static_cast<uint64_t>(tokenNum_) * nQkv_);
    wsXcqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(ws + wsXcqOffset_),
                             static_cast<uint64_t>(tokenNum_) * qLoraRank_);
    wsStok2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + wsStok2Offset_), tokenNum_);
    wsQfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ws + wsQfOffset_),
                            static_cast<uint64_t>(tokenNum_) * nQb_);
    wsPredGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws + wsPredOffset_),
                              static_cast<uint64_t>(tokenNum_) * hiddenSize_);
    wsWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws + wsWeightsOffset_),
                                 static_cast<uint64_t>(tokenNum_) * nHead_);

    // The arena covers the largest vector phase.
    const uint32_t hs = hiddenSize_;
    const uint32_t ql = qLoraRank_;
    const uint32_t nb = nQb_;
    const P1Offsets p1 = ComputeP1Offsets(hs);
    const P2Offsets p2 = ComputeP2Offsets(ql);
    const P3VecOffsets p3 = ComputeP3VecOffsets(nb, ropeElems_, halfRd_);
    uint32_t arena = p1.end;
    if (p2.end > arena) { arena = p2.end; }
    if (p3.end > arena) { arena = p3.end; }
    ubArenaBytes_ = PqfRoundUp(arena, FP32_ELEMS_PER_32B * 4);

    // Allocate the vector arena before MatmulImpl reserves its buffers.
    pipe_->InitBuffer(ubArenaBuf_, ubArenaBytes_);
    ubArena_ = ubArenaBuf_.Get<uint8_t>();

    // rank-aware 模式：把 alpha_vec/beta_vec（bf16 [N]）预载到 UB（P1 尾段）并 cast fp32。
    // 仅 Phase 1（quant1）使用；Phase 3（RoPE）覆盖前已结束，跨 chunk 复用无需重复搬运。
    if (alphaBetaMode_ == 1U && alphaVecLen_ > 0U) {
        const P1Offsets p1Init = ComputeP1Offsets(hiddenSize_);
        auto alphaVecBf16 = Ub<bfloat16_t>(p1Init.alphaVecBf16);
        auto betaVecBf16 = Ub<bfloat16_t>(p1Init.betaVecBf16);
        auto alphaVecFp32 = Ub<float>(p1Init.alphaVecFp32);
        auto betaVecFp32 = Ub<float>(p1Init.betaVecFp32);
        const uint32_t n = PqfRoundUp(alphaVecLen_, BF16_ELEMS_PER_32B);
        PqfCopyIn(alphaVecBf16, alphaVecGm_, alphaVecLen_);
        PqfCopyIn(betaVecBf16, betaVecGm_, alphaVecLen_);
        PqfCastFrom16To32(alphaVecFp32, alphaVecBf16, n);
        PqfCastFrom16To32(betaVecFp32, betaVecBf16, n);
        PipeBarrier<PIPE_ALL>();
    }

    // Only AIC cores own cube buffers and initialize the matmul instances.
    if ASCEND_IS_AIC {
        mm1_.Init(&cubeTiling1_, pipe);
        mm2_.Init(&cubeTiling2_, pipe);
        mm3_.Init(&cubeTiling3_, pipe);
    }
}

// ============ ReduceMax 单行（{1, N}，AR：归约内轴 N） ============
__aicore__ inline float KernelPrefetchQliFusion::ReduceMaxRow(
    LocalTensor<float>& absLocal, LocalTensor<float>& maxFp32, uint32_t rowLenAligned)
{
    uint32_t srcShape[2] = {1, rowLenAligned};
    ReduceMax<float, AscendC::Pattern::Reduce::AR>(maxFp32, absLocal, srcShape, true);
    PipeBarrier<PIPE_ALL>();  // V_S 事件在 AIC no-op，用 PIPE_ALL 等 ReduceMax 写盘后再标量读
    float maxVal = maxFp32.GetValue(0);
    PipeBarrier<PIPE_ALL>();  // S_V 事件在 AIC no-op
    return maxVal;
}

// ============ ReduceSum 单行（{1, N}，AR：归约内轴 N） ============
// ReduceSum<AR> preserves the platform reduction behavior used by RMSNorm.
__aicore__ inline float KernelPrefetchQliFusion::ReduceSumRow(
    LocalTensor<float>& sqLocal, LocalTensor<float>& sumFp32, uint32_t rowLenAligned)
{
    uint32_t srcShape[2] = {1, rowLenAligned};
    ReduceSum<float, AscendC::Pattern::Reduce::AR>(sumFp32, sqLocal, srcShape, true);
    PipeBarrier<PIPE_ALL>();  // V_S 事件在 AIC no-op，用 PIPE_ALL 等 ReduceSum 写盘后再标量读
    float sumVal = sumFp32.GetValue(0);
    PipeBarrier<PIPE_ALL>();  // S_V 事件在 AIC no-op
    return sumVal;
}

// ============ RoPE gather offsets ============
// aOffsets[h*halfRd+i]     = (h*headDim + 2i)*4        （偶列 a；b 列复用 + srcBaseAddr=4）
// interleaveOffsets[h*2*halfRd+2i]     = (h*halfRd + i)*4
// interleaveOffsets[h*2*halfRd+2i+1]   = ropeBytes + (h*halfRd + i)*4   （nb 段）
// The offsets depend only on the tensor shape.
__aicore__ inline void KernelPrefetchQliFusion::BuildRopeOffsets()
{
    const uint32_t halfRd = halfRd_;
    const uint32_t ropeElems = ropeElems_;
    const uint32_t hd = headDim_;
    const uint32_t ropeBytes = ropeElems * sizeof(float);  // na/nb 段字节偏移
    const P3VecOffsets o = ComputeP3VecOffsets(nQb_, ropeElems, halfRd);
    auto aOff = Ub<uint32_t>(o.aOffsets);
    auto interOff = Ub<uint32_t>(o.interleaveOffsets);

    for (uint32_t h = 0; h < nHead_; ++h) {
        for (uint32_t i = 0; i < halfRd; ++i) {
            const uint32_t evenOff = (h * hd + 2 * i) * sizeof(float);       // 偶列字节
            const uint32_t ropeIdx = (h * halfRd + i) * sizeof(float);       // aAndB 内 a 段字节
            aOff.SetValue(h * halfRd + i, evenOff);
            interOff.SetValue(2 * (h * halfRd + i), ropeIdx);
            interOff.SetValue(2 * (h * halfRd + i) + 1, ropeBytes + ropeIdx);  // b 段
        }
    }
    PipeBarrier<PIPE_ALL>();  // 标量写完成，后续 Gather 可读
}

// ============ mm1（moe 范式逐 N-block，raw int32 输出，手动 B/C GM 偏移） ============
// Iterate over baseN blocks within this core's N-axis slice.
// curN ∈ {baseN, 全局尾块}（ComputeNSlice 保证），每核只加载 N/nCore 列权重。
__aicore__ inline void KernelPrefetchQliFusion::RunMatmul1(uint32_t mStart, uint32_t mLen,
                                                           uint32_t nStart, uint32_t nLen)
{
    if (nLen == 0U) { return; }  // 本核无 mm1 N 分片
    const uint32_t n = nQkv_;
    const uint32_t k = hiddenSize_;
    const uint32_t baseN = baseN_;

    GlobalTensor<int8_t> aGm = wsXqGm_[static_cast<uint64_t>(mStart) * k];
    uint32_t local = 0U;
    while (local < nLen) {
        uint32_t curN = (nLen - local > baseN) ? baseN : (nLen - local);  // 块宽（尾块 < baseN）
        // ND transB: B 存 [N,K] 行优先，本块起始列 (nStart+local)，元素偏移 (nStart+local)*K
        uint64_t bOff = static_cast<uint64_t>(nStart + local) * k;
        mm1_.SetOrgShape(tokenNum_, n, k);
        mm1_.SetSingleShape(mLen, curN, k);
        mm1_.SetTensorA(aGm, false);
        mm1_.SetTensorB(wqkvGm_[bOff], false);  // B=[K,N] NZ，N 列偏移 nStart*K
        mm1_.template IterateAll<false>(
            wsQkvOutGm_[static_cast<uint64_t>(mStart) * n + nStart + local], 0);
        local += baseN;
    }
    PipeBarrier<PIPE_ALL>();  // 等本核全部 N-block 写盘完成
    mm1_.End();
}

// ============ mm2（moe 范式逐 N-block，raw int32 输出，手动 B/C GM 偏移） ============
// Iterate over baseN blocks within this core's N-axis slice.
__aicore__ inline void KernelPrefetchQliFusion::RunMatmul2(uint32_t mStart, uint32_t mLen,
                                                           uint32_t nStart, uint32_t nLen)
{
    if (nLen == 0U) { return; }  // 本核无 mm2 N 分片
    const uint32_t n = nQb_;
    const uint32_t k = qLoraRank_;
    const uint32_t baseN = baseN_;

    GlobalTensor<int8_t> aGm = wsXcqGm_[static_cast<uint64_t>(mStart) * k];
    uint32_t local = 0U;
    while (local < nLen) {
        uint32_t curN = (nLen - local > baseN) ? baseN : (nLen - local);
        uint64_t bOff = static_cast<uint64_t>(nStart + local) * k;  // ND transB: (nStart+local)*K
        mm2_.SetOrgShape(tokenNum_, n, k);
        mm2_.SetSingleShape(mLen, curN, k);
        mm2_.SetTensorA(aGm, false);
        mm2_.SetTensorB(wqbGm_[bOff], false);  // B=[K,N] NZ，N 列偏移 nStart*K
        mm2_.template IterateAll<false>(
            wsQfGm_[static_cast<uint64_t>(mStart) * n + nStart + local], 0);
        local += baseN;
    }
    PipeBarrier<PIPE_ALL>();
    mm2_.End();
}

// ============ mm3：wk_weights_proj（BF16，N=n_head，全 M） ============
// 仅 blk==0 的 AIC 核计算全部 T 行的 weights（N=n_head=32 很小）。
// A = wsPred（predicted_hidden bf16 [T, hidden]，B1 后 AIV 已写完）
// B = wkWeightsGm_（已行偏移到 head_dim 起，只含 weights 列对应权重）
// C = wsWeightsGm_（workspace [T, n_head] bf16，AIV Phase3 再拷到输出 GM）
__aicore__ inline void KernelPrefetchQliFusion::RunMatmul3()
{
    const uint32_t n = nHead_;
    const uint32_t k = hiddenSize_;
    const uint32_t m = tokenNum_;
    if (m == 0U || n == 0U) { return; }

    GlobalTensor<bfloat16_t> aGm = wsPredGm_;
    const uint64_t bOff = static_cast<uint64_t>(headDim_) * hiddenSize_;  // B=[N_wk,K] ND，weights 行偏移 headDim*K
    mm3_.SetOrgShape(m, n, k);
    mm3_.SetSingleShape(m, n, k);
    mm3_.SetTensorA(aGm, false);
    mm3_.SetTensorB(wkWeightsGm_[bOff], true);  // B=[N_wk, K=hidden] + isTransB，下标取 weights 行
    mm3_.template IterateAll<false>(wsWeightsGm_, 0);
    PipeBarrier<PIPE_ALL>();  // 等 weights 写盘完成
    mm3_.End();
}

// ============ per-token 量化①（H -> xq + s_tok -> GM） ============
__aicore__ inline void KernelPrefetchQliFusion::ProcessChunkQuant1(uint32_t mStart, uint32_t mLen)
{
    const uint32_t hs = hiddenSize_;
    const P1Offsets o = ComputeP1Offsets(hs);

    auto hpBf16 = Ub<bfloat16_t>(o.hpBf16);
    auto hpFp32 = Ub<float>(o.hpFp32);
    auto absFp32 = Ub<float>(o.absFp32);
    auto maxFp32 = Ub<float>(o.maxFp32);
    auto tmpHalf = Ub<half>(o.tmpHalf);
    auto xqInt8 = Ub<int8_t>(o.xqInt8);

    const uint32_t hsB = PqfRoundUp(hs, BF16_ELEMS_PER_32B);
    const uint32_t hsF = PqfRoundUp(hs, FP32_ELEMS_PER_32B);

    // rank-aware 系数（Phase 1 使用，Init 已预载）
    auto alphaVecFp32 = Ub<float>(o.alphaVecFp32);
    auto betaVecFp32 = Ub<float>(o.betaVecFp32);

    for (uint32_t r = 0; r < mLen; ++r) {
        PqfCopyIn(hpBf16, hiddenGm_[static_cast<uint64_t>(mStart + r) * hs], hs);
        PqfCastFrom16To32(hpFp32, hpBf16, hsB);
        if (alphaBetaMode_ == 1U) {
            // rank-aware：row_rank = min(row / source_rows_before_gather, N-1)，
            // 每行取所属 rank 的 alpha/beta（GLM-5.2 apply_group_predict_coefficients）。
            const uint32_t row = mStart + r;
            const uint32_t rawRank = (sourceRowsBeforeGather_ > 0U) ? (row / sourceRowsBeforeGather_) : 0U;
            const uint32_t rowRank = (alphaVecLen_ > 0U && rawRank < alphaVecLen_)
                                         ? rawRank
                                         : ((alphaVecLen_ > 0U) ? (alphaVecLen_ - 1U) : 0U);
            PipeBarrier<PIPE_ALL>();  // S_V 事件在 AIC no-op，PIPE_ALL 兜底
            const float a = alphaVecFp32.GetValue(rowRank);
            const float b = betaVecFp32.GetValue(rowRank);
            PipeBarrier<PIPE_ALL>();
            Muls(hpFp32, hpFp32, a, hsF);
            PipeBarrier<PIPE_V>();
            Adds(hpFp32, hpFp32, b, hsF);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(hpFp32, hpFp32, alpha_, hsF);
            PipeBarrier<PIPE_V>();
            Adds(hpFp32, hpFp32, beta_, hsF);
            PipeBarrier<PIPE_V>();
        }
        // 对齐 golden：`Hp = (H*alpha+beta).to(bf16)` 先取 bf16 再量化（P6 alpha/beta 非平凡时
        // 避免 fp32 Hp 与 golden bf16 Hp 的 scale/量化差异经下游放大）。
        PqfCastFrom32To16(hpBf16, hpFp32, hsF);  // round_bf16(Hp)，复用输入 buffer
        PqfCopyOut(wsPredGm_[static_cast<uint64_t>(mStart + r) * hs], hpBf16, hs);  // ★ predicted_hidden -> wsPred（mm3 A）
        PqfCastFrom16To32(hpFp32, hpBf16, hsB);  // 回 fp32 做量化
        Abs(absFp32, hpFp32, hsF);
        PipeBarrier<PIPE_V>();
        float maxVal = ReduceMaxRow(absFp32, maxFp32, hsF);
        float scale = (maxVal == 0.0f) ? 1.0f : (maxVal / 127.0f);
        float invScale = 1.0f / scale;
        Muls(hpFp32, hpFp32, invScale, hsF);
        PipeBarrier<PIPE_V>();
        PqfCastFromF32ToI8(xqInt8, tmpHalf, hpFp32, hsF);
        PqfCopyOut(wsXqGm_[static_cast<uint64_t>(mStart + r) * hs], xqInt8, hs);
        maxFp32.SetValue(0, scale);
        PipeBarrier<PIPE_ALL>();  // S_V 事件在 AIC/AIV 均需 PIPE_ALL 兜底，让标量写对 MTE3 可见
        PqfCopyOut(wsStokGm_[mStart + r], maxFp32, 1);
    }
}

// ============ slice + per-channel/per-token 反量化 + RMSNorm + 量化② ============
__aicore__ inline void KernelPrefetchQliFusion::ProcessChunkRmsNorm(uint32_t mStart, uint32_t mLen)
{
    const uint32_t ql = qLoraRank_;
    const P2Offsets o = ComputeP2Offsets(ql);

    auto qkvInt32 = Ub<int32_t>(o.qkvInt32);
    auto qkvFp32 = Ub<float>(o.qkvFp32);
    auto wsQkvBf16 = Ub<bfloat16_t>(o.wsQkvBf16);
    auto wsQkvFp32 = Ub<float>(o.wsQkvFp32);
    // q_raw and q_c use BF16 at the quantized matmul and RMSNorm boundary.
    auto qrawHalf = Ub<bfloat16_t>(o.qrawHalf);
    auto qrawFp32 = Ub<float>(o.qrawFp32);
    auto gammaFp32 = Ub<float>(o.gammaFp32);
    auto betaFp32 = Ub<float>(o.betaFp32);
    auto sqFp32 = Ub<float>(o.sqFp32);
    auto sumFp32 = Ub<float>(o.sumFp32);
    auto rmsFp32 = Ub<float>(o.rmsFp32);
    auto outFp32 = Ub<float>(o.outFp32);
    auto sTokRow = Ub<float>(o.sTokRow);
    auto gammaBf16 = Ub<bfloat16_t>(o.gammaBf16);
    auto betaBf16 = Ub<bfloat16_t>(o.betaBf16);
    auto qcFp32 = Ub<float>(o.qcFp32);
    auto absFp32 = Ub<float>(o.absFp32);
    auto maxFp32_2 = Ub<float>(o.maxFp32);
    auto tmpHalf2 = Ub<half>(o.tmpHalf2);
    auto xcqInt8 = Ub<int8_t>(o.xcqInt8);

    const uint32_t qlB = PqfRoundUp(ql, BF16_ELEMS_PER_32B);
    const uint32_t qlF = PqfRoundUp(ql, FP32_ELEMS_PER_32B);

    // gamma/beta 常驻 UB（每 chunk 加载一次）
    PqfCopyIn(gammaBf16, gamma1Gm_, ql);
    PqfCopyIn(betaBf16, beta1Gm_, ql);
    PqfCastFrom16To32(gammaFp32, gammaBf16, qlB);
    PqfCastFrom16To32(betaFp32, betaBf16, qlB);
    // per-channel 反量化系数 ws_qkv[:ql]（每 chunk 加载一次）
    PqfCopyIn(wsQkvBf16, wsQkvGm_[0], ql);
    PqfCastFrom16To32(wsQkvFp32, wsQkvBf16, qlB);

    for (uint32_t r = 0; r < mLen; ++r) {
        // slice qkv_out[r, :ql] (raw int32) -> UB int32 -> fp32
        // 注意：int32->float 的 Cast roundMode 必须是 CAST_RINT（CAST_NONE 不支持，
        // 见 Cast.md 910B 表 `int32_t|float` 行）。
        PqfCopyIn(qkvInt32, wsQkvOutGm_[static_cast<uint64_t>(mStart + r) * nQkv_], ql);
        Cast(qkvFp32, qkvInt32, RoundMode::CAST_RINT, ql);
        PipeBarrier<PIPE_V>();
        // per-channel 反量化: qkvFp32 *= ws_qkv
        Mul(qkvFp32, qkvFp32, wsQkvFp32, qlF);
        PipeBarrier<PIPE_V>();
        // per-token 补乘 s_tok[r]
        PqfCopyIn(sTokRow, wsStokGm_[mStart + r], 1);
        float s = sTokRow.GetValue(0);
        Muls(qkvFp32, qkvFp32, s, qlF);
        PipeBarrier<PIPE_V>();
        // fp32 -> bf16 (q_raw)
        PqfCastFrom32To16(qrawHalf, qkvFp32, qlF);
        // === RMSNorm ===
        PqfCastFrom16To32(outFp32, qrawHalf, qlB);
        Mul(sqFp32, outFp32, outFp32, qlF);
        PipeBarrier<PIPE_V>();
        float sum = ReduceSumRow(sqFp32, sumFp32, qlF);
        float mean = sum * invQLoraRank_;
        float rms = sqrt(mean + eps_);
        Duplicate(rmsFp32, rms, qlF);
        PipeBarrier<PIPE_V>();
        Div(outFp32, outFp32, rmsFp32, qlF);
        PipeBarrier<PIPE_V>();
        // Round the normalized value to BF16 before the affine transform.
        PqfCastFrom32To16(qrawHalf, outFp32, qlF);   // round_bf16(q_raw/rms)
        PqfCastFrom16To32(outFp32, qrawHalf, qlB);   // 回 fp32 做 gamma/beta 仿射
        Mul(outFp32, outFp32, gammaFp32, qlF);
        PipeBarrier<PIPE_V>();
        Add(outFp32, outFp32, betaFp32, qlF);
        PipeBarrier<PIPE_V>();
        // q_c bf16（与设计链一致：bf16 中间）
        PqfCastFrom32To16(qrawHalf, outFp32, qlF);
        // === 量化②：q_c -> xcq + s_tok2 -> GM ===
        PqfCastFrom16To32(qcFp32, qrawHalf, qlB);
        Abs(absFp32, qcFp32, qlF);
        PipeBarrier<PIPE_V>();
        float maxVal = ReduceMaxRow(absFp32, maxFp32_2, qlF);
        float scale = (maxVal == 0.0f) ? 1.0f : (maxVal / 127.0f);
        float invScale = 1.0f / scale;
        Muls(qcFp32, qcFp32, invScale, qlF);
        PipeBarrier<PIPE_V>();
        PqfCastFromF32ToI8(xcqInt8, tmpHalf2, qcFp32, qlF);
        PqfCopyOut(wsXcqGm_[static_cast<uint64_t>(mStart + r) * ql], xcqInt8, ql);
        maxFp32_2.SetValue(0, scale);
        PipeBarrier<PIPE_ALL>();  // S_V 事件在 AIC no-op
        PqfCopyOut(wsStok2Gm_[mStart + r], maxFp32_2, 1);
    }
}

// ============ mm2 输出 + per-channel/per-token 反量化 + 矢量 interleave RoPE -> q_li ============
// Vector RoPE preserves the fp32 operation order used by the reference path:
//   a = qf[2i], b = qf[2i+1], na = a*cos - b*sin, nb = b*cos + a*sin
// 矢量流程（每行）：
//   X(全行 fp32, round_bf16) --Gather(aOff, base=0)-> aBuf
//                                --Gather(aOff, base=4)-> bBuf   （b 列 = a 列偏移+1）
//   broadcast-Mul cos/sin（cosFull32=[cos(32),cos(32)]，src1RepStride=0）
//   na = a*cos - b*sin；nb = b*cos + a*sin
//   --Gather(interleaveOffsets)-> ropeBuf（交织）
//   Cast -> ropeBf16；nope 段直接用 wsQbBf16 的 round_bf16 全行 strided DataCopyPad 写回
// 复用（不新增 UB）：
//   qfInt32 空间：aBuf[0,8K)+bBuf[8K,16K)+t3[16K,24K)+t4[24K,32K)；交织后 ropeBf16[0,8K)
//   X 空间：ropeBuf[0,16K)；X[16K,32K) 仅作 gather 源（nope 段留在内）
//   wsQbBf16：round_bf16(qf) 全行 bf16（nope 写回源，stride-2 段）
//   wsQbFp32：反量化系数（常驻）
__aicore__ inline void KernelPrefetchQliFusion::ProcessChunkRoPE(uint32_t mStart, uint32_t mLen)
{
    const uint32_t nb = nQb_;
    const uint32_t nh = nHead_;
    const uint32_t hd = headDim_;
    const uint32_t rd = ropeDim_;
    const uint32_t halfRd = halfRd_;
    const uint32_t ropeElems = ropeElems_;
    const uint32_t ropeBytes = ropeElems * sizeof(float);
    const uint32_t ropeRepFp32 = 64;  // 910B 矢量 repeat = 64 fp32
    const P3VecOffsets o = ComputeP3VecOffsets(nb, ropeElems, halfRd);

    auto qfInt32 = Ub<int32_t>(o.qfInt32);
    auto X = Ub<float>(o.qfFp32);
    auto wsQbBf16 = Ub<bfloat16_t>(o.wsQbBf16);
    auto wsQbFp32 = Ub<float>(o.wsQbFp32);
    auto cosHalf = Ub<bfloat16_t>(o.cosHalf);
    auto sinHalf = Ub<bfloat16_t>(o.sinHalf);
    auto cosFp32 = Ub<float>(o.cosFp32);
    auto sinFp32 = Ub<float>(o.sinFp32);
    auto sTok2Row = Ub<float>(o.sTok2);
    auto cosFull32 = Ub<float>(o.cosFull32);
    auto sinFull32 = Ub<float>(o.sinFull32);
    auto aOff = Ub<uint32_t>(o.aOffsets);
    auto interOff = Ub<uint32_t>(o.interleaveOffsets);

    // 子视图（复用 qfInt32 / X 空间）
    const uint32_t aq = o.qfInt32;
    auto aBuf   = Ub<float>(aq + 0 * ropeBytes);  // 8KB
    auto bBuf   = Ub<float>(aq + 1 * ropeBytes);  // 8KB
    auto t3     = Ub<float>(aq + 2 * ropeBytes);  // 8KB
    auto t4     = Ub<float>(aq + 3 * ropeBytes);  // 8KB
    auto ropeBf16 = Ub<bfloat16_t>(aq + 0 * ropeBytes);  // 交织后复用 aBuf 空间（8KB bf16）
    const uint32_t xq = o.qfFp32;
    auto ropeBuf  = Ub<float>(xq + 0);               // 16KB

    const uint32_t nbB = PqfRoundUp(nb, BF16_ELEMS_PER_32B);
    const uint32_t nbF = PqfRoundUp(nb, FP32_ELEMS_PER_32B);
    const uint32_t hrB = PqfRoundUp(halfRd, BF16_ELEMS_PER_32B);
    const uint32_t ropeF = PqfRoundUp(ropeElems, FP32_ELEMS_PER_32B);
    const uint32_t repTimes = ropeF / ropeRepFp32;

    // per-channel 反量化系数 ws_qb（每 chunk 加载一次，常驻 wsQbFp32；wsQbBf16 后复用 round_bf16）
    PqfCopyIn(wsQbBf16, wsQbGm_[0], nb);
    PqfCastFrom16To32(wsQbFp32, wsQbBf16, nbB);

    // broadcast-Mul 参数：src1RepStride=0 → cos/sin 全 head 广播（cosFull32 = [cos(32),cos(32)]）
    const BinaryRepeatParams brp{1, 1, 1, 8, 8, 0};

    for (uint32_t r = 0; r < mLen; ++r) {
        // ---- 1. int32 -> fp32 + per-channel/per-token 反量化 ----
        PqfCopyIn(qfInt32, wsQfGm_[static_cast<uint64_t>(mStart + r) * nb], nb);
        Cast(X, qfInt32, RoundMode::CAST_RINT, nb);
        PipeBarrier<PIPE_V>();
        Mul(X, X, wsQbFp32, nbF);   // per-channel
        PipeBarrier<PIPE_V>();
        PqfCopyIn(sTok2Row, wsStok2Gm_[mStart + r], 1);
        float s2 = sTok2Row.GetValue(0);
        Muls(X, X, s2, nbF);        // per-token
        PipeBarrier<PIPE_V>();
        // 对齐 golden：round_bf16(qf)（wsQbBf16 复用为 bf16 全行中间，同时也是 nope 写回源）
        PqfCastFrom32To16(wsQbBf16, X, nbF);  // wsQbBf16 = round_bf16(qf) 全行 bf16
        PqfCastFrom16To32(X, wsQbBf16, nbB);  // X = round_bf16 精度回 fp32 供 gather/旋转

        // ---- 2. cos/sin（本行，[halfRd]）+ 复制成 broadcast 用 64 fp32 ----
        PqfCopyIn(cosHalf, cosGm_[static_cast<uint64_t>(mStart + r) * halfRd], halfRd);
        PqfCopyIn(sinHalf, sinGm_[static_cast<uint64_t>(mStart + r) * halfRd], halfRd);
        PqfCastFrom16To32(cosFp32, cosHalf, hrB);
        PqfCastFrom16To32(sinFp32, sinHalf, hrB);
        // cosFull32 = [cos(32), cos(32)]（910B 无 3 参 Copy，用 mask/repeat 版）
        {
            const CopyRepeatParams crp{1, 1, 8, 8};
            Copy(cosFull32[0], cosFp32, halfRd, 1, crp);
            Copy(cosFull32[halfRd], cosFp32, halfRd, 1, crp);
            Copy(sinFull32[0], sinFp32, halfRd, 1, crp);
            Copy(sinFull32[halfRd], sinFp32, halfRd, 1, crp);
        }
        PipeBarrier<PIPE_V>();

        // ---- 3. 拆偶/奇：Gather a、b（b 用 srcBaseAddr=4 复用 aOff） ----
        Gather(aBuf, X, aOff, /*srcBaseAddr=*/0, ropeElems);
        Gather(bBuf, X, aOff, /*srcBaseAddr=*/sizeof(float), ropeElems);
        PipeBarrier<PIPE_V>();

        // ---- 4. 矢量交织旋转 ----
        //   t3 = a*sin；t4 = b*cos；t4 = nb = b*cos + a*sin
        //   aBuf = a*cos；bBuf = b*sin；aBuf = na = a*cos - b*sin
        Mul(t3, aBuf, sinFull32, ropeRepFp32, repTimes, brp);
        PipeBarrier<PIPE_V>();
        Mul(t4, bBuf, cosFull32, ropeRepFp32, repTimes, brp);
        PipeBarrier<PIPE_V>();
        Add(t4, t4, t3, ropeF);
        PipeBarrier<PIPE_V>();
        Mul(aBuf, aBuf, cosFull32, ropeRepFp32, repTimes, brp);
        PipeBarrier<PIPE_V>();
        Mul(bBuf, bBuf, sinFull32, ropeRepFp32, repTimes, brp);
        PipeBarrier<PIPE_V>();
        Sub(aBuf, aBuf, bBuf, ropeF);
        PipeBarrier<PIPE_V>();
        // nb(t4) 移到 bBuf（与 aBuf 连续成 aAndB=[na,nb]，供交织 Gather）
        {
            const CopyRepeatParams crp{1, 1, 8, 8};
            Copy(bBuf, t4, ropeRepFp32, repTimes, crp);
        }
        PipeBarrier<PIPE_ALL>();

        // ---- 5. 交织回 ropeBuf（Gather，src=aBuf 基址 = aAndB=[na,nb]） ----
        Gather(ropeBuf, aBuf, interOff, /*srcBaseAddr=*/0, 2 * ropeElems);
        PipeBarrier<PIPE_V>();

        // ---- 6. 写回 q_li（bf16）：rope 段（Cast 后 strided）+ nope 段（wsQbBf16 全行 bf16 strided） ----
        PqfCastFrom32To16(ropeBf16, ropeBuf, 2 * ropeElems);
        PipeBarrier<PIPE_ALL>();  // V 完成，MTE3 可读
        // rope 段：nh heads × rd bf16，head stride = hd bf16
        DataCopyPad(qLiGm_[static_cast<uint64_t>(mStart + r) * nb], ropeBf16,
                    DataCopyExtParams{static_cast<uint16_t>(nh),
                                      static_cast<uint32_t>(rd * sizeof(uint16_t)), 0,
                                      static_cast<uint32_t>((hd - rd) * sizeof(uint16_t)), 0});
        PipeBarrier<PIPE_ALL>();  // MTE3 完成
        // nope 段：源 = wsQbBf16 全行 bf16 的 [rd : hd) 段（每 head 跳过 rd 个 bf16 起读）
        DataCopyPad(qLiGm_[static_cast<uint64_t>(mStart + r) * nb + rd], wsQbBf16[rd],
                    DataCopyExtParams{static_cast<uint16_t>(nh),
                                      static_cast<uint32_t>((hd - rd) * sizeof(uint16_t)),
                                      static_cast<uint32_t>(rd * sizeof(uint16_t) / 32U),
                                      static_cast<uint32_t>((hd - rd) * sizeof(uint16_t)), 0});
        PipeBarrier<PIPE_ALL>();
    }
}

// ============ ProcessAic（AIC 核：两段 matmul） ============
// SyncAll<false>() 全局 barrier 序列（所有 AIC/AIV 核按同一顺序调用）：
//   B1: quant1 完成 -> mm1 可开始
//   B2: mm1 完成 -> RMSNorm/quant2 可开始
//   B3: quant2 完成 -> mm2 可开始
//   B4: mm2 完成 -> RoPE 可开始
// The 2D grid assigns one M segment and one N slice to each AIC core:
//   mCoreIdx = blk / nCore；nCoreIdx = blk % nCore。每核只加载 N/nCore 列权重。
// AIV 侧从 GM 读完整行段做后处理（跨核归并在 GM，B2/B4 barrier 保证所有分片写盘完成）。
__aicore__ inline void KernelPrefetchQliFusion::ProcessAic(uint32_t blk)
{
    if (tokenNum_ == 0) { return; }
    if (blk >= usedCoreNum_) {
        // 超出 AIC 网格的块：仅参与 4 个 barrier
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        return;
    }
    const uint32_t nCore = (nCore_ == 0) ? 1U : nCore_;
    uint32_t mCoreIdx = blk / nCore;
    uint32_t nCoreIdx = blk % nCore;
    uint32_t mStart = mCoreIdx * singleM_;
    uint32_t mEnd = mStart + singleM_;
    if (mEnd > tokenNum_) { mEnd = tokenNum_; }
    uint32_t totalRows = (mEnd > mStart) ? (mEnd - mStart) : 0;

    // N 分片（mm1/mm2 各自独立，共享 nCore 切分数）
    uint32_t nStart1 = 0, nLen1 = 0, nStart2 = 0, nLen2 = 0;
    ComputeNSlice(nQkv_, baseN_, nCore, nCoreIdx, nStart1, nLen1);
    ComputeNSlice(nQb_, baseN_, nCore, nCoreIdx, nStart2, nLen2);

    if (totalRows == 0) {
        // 无行段的 AIC 核：仅参与 4 个 barrier
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        return;
    }

    // B1: 等 AIV 完成 quant1（xq/s_tok/predicted_hidden 落 GM）
    AscendC::SyncAll<false>();
    // mm1（moe 范式逐 N-block，raw int32 输出到 workspace 本核 N 分片）
    RunMatmul1(mStart, totalRows, nStart1, nLen1);
    // mm3（wk_weights_proj，BF16）：仅 blk==0 核算全 M weights（N=n_head 很小），
    // A=wsPred（B1 后 AIV 已写 predicted_hidden），C 直接写 weights 输出 GM
    if (blk == 0U) {
        RunMatmul3();
    }
    // B2: mm1/mm3 完成 -> AIV 可开始 RMSNorm
    AscendC::SyncAll<false>();
    // B3: 等 AIV 完成 quant2（xcq/s_tok2 落 GM）
    AscendC::SyncAll<false>();
    // mm2（本核 N 分片）
    RunMatmul2(mStart, totalRows, nStart2, nLen2);
    // B4: mm2 完成 -> AIV 可开始 RoPE
    AscendC::SyncAll<false>();
}

// ============ ProcessAiv（AIV 核：vector 后处理） ============
// AIV row assignment is independent of the AIC M/N grid.
// 所有 AIV 核按连续行区间均分 [0, tokenNum)，与 AIC 的 M/N 网格无关 —— AIV 从 GM 读
// 完整行段做 quant1/RMSNorm/RoPE，跨核依赖完全由 B1-B4 全局 barrier 保障（N 分片
// AIC 在 B2/B4 前已把本核分片写盘 GM，AIV 在 barrier 后读到完整行）。
__aicore__ inline void KernelPrefetchQliFusion::ProcessAiv(uint32_t aivBlk)
{
    if (tokenNum_ == 0) { return; }
    uint32_t totalAiv = 2U * usedCoreNum_;   // MIX 1:2：AIV 核数 = 2 × AIC 核数
    if (totalAiv == 0) { totalAiv = 1U; }
    uint32_t perAiv = (tokenNum_ + totalAiv - 1U) / totalAiv;  // ceil 行/核
    uint32_t vStart = aivBlk * perAiv;
    uint32_t vLen = 0;
    if (vStart < tokenNum_) {
        // 防下溢：vStart >= T 的核无行段（仅参与 barrier）
        vLen = (vStart + perAiv > tokenNum_) ? (tokenNum_ - vStart) : perAiv;
    }
    if (vLen == 0) {
        // 无行段的 AIV 核：仅参与 4 个 barrier
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        return;
    }

    // Phase 1: per-token 量化①（本核行段 -> xq/s_tok GM）
    for (uint32_t off = 0; off < vLen; off += mChunk_) {
        uint32_t mLen = (vLen - off > mChunk_) ? mChunk_ : (vLen - off);
        ProcessChunkQuant1(vStart + off, mLen);
    }
    // B1: quant1 完成 -> AIC 可开始 mm1
    AscendC::SyncAll<false>();
    // B2: 等 AIC 完成 mm1（qkv_out 落 GM）
    AscendC::SyncAll<false>();
    // Phase 2: slice + per-channel/per-token 反量化 + RMSNorm + 量化②
    for (uint32_t off = 0; off < vLen; off += mChunk_) {
        uint32_t mLen = (vLen - off > mChunk_) ? mChunk_ : (vLen - off);
        ProcessChunkRmsNorm(vStart + off, mLen);
    }
    // B3: quant2 完成 -> AIC 可开始 mm2
    AscendC::SyncAll<false>();
    // B4: 等 AIC 完成 mm2（qf 落 GM）
    AscendC::SyncAll<false>();
    // Build offsets after quantization and RMSNorm release their shared UB ranges.
    BuildRopeOffsets();
    // Phase 3: per-channel/per-token 反量化 + interleave RoPE -> q_li
    for (uint32_t off = 0; off < vLen; off += mChunk_) {
        uint32_t mLen = (vLen - off > mChunk_) ? mChunk_ : (vLen - off);
        ProcessChunkRoPE(vStart + off, mLen);
    }
    // Phase 3b: mm3 weights（wsWeights workspace）→ 输出 GM（weightsGm_）
    // wsWeights 由 blk==0 的 AIC 核在 B1 后写入；本核读它已过 B2/B3/B4 barrier，安全。
    if (vLen > 0U) {
        const P3VecOffsets o3 = ComputeP3VecOffsets(nQb_, ropeElems_, halfRd_);
        auto wCopy = Ub<bfloat16_t>(o3.qfInt32);  // 复用 RoPE 后的 free UB 区
        for (uint32_t r = 0; r < vLen; ++r) {
            PqfCopyIn(wCopy, wsWeightsGm_[static_cast<uint64_t>(vStart + r) * nHead_], nHead_);
            PqfCopyOut(weightsGm_[static_cast<uint64_t>(vStart + r) * nHead_], wCopy, nHead_);
        }
    }
}

}  // namespace

extern "C" __global__ __aicore__ void prefetch_qli_fusion(
    GM_ADDR hiddenStates, GM_ADDR wqkv, GM_ADDR wsQkv, GM_ADDR wqb, GM_ADDR wsQb,
    GM_ADDR gamma1, GM_ADDR beta1, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR wkWeights, GM_ADDR alphaVec, GM_ADDR betaVec,
    GM_ADDR qLi, GM_ADDR weights, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelPrefetchQliFusion op;
    op.Init(hiddenStates, wqkv, wsQkv, wqb, wsQb, gamma1, beta1, cos, sin,
            wkWeights, alphaVec, betaVec,
            qLi, weights, workspace, tilingData, &pipe);
    uint32_t blk = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        op.ProcessAiv(blk);
    } else {
        op.ProcessAic(blk);
    }
}
