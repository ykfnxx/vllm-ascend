/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

#include "util_regbase.h"
#include "sparse_flash_attention_common_arch35.h"

#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#if __has_include("../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h")
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#else
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_flashupdate_new.h")
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#else
#include "../../common/arch35/vf/vf_flashupdate_new.h"
#endif

#if __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../../common/buffers_policy.h"
#endif
#if __has_include("../../common/op_kernel/buffer_manager.h")
#include "../../common/op_kernel/buffer_manager.h"
#else
#include "../../common/buffer_manager.h"
#endif
#if __has_include("../../common/op_kernel/buffer.h")
#include "../../common/op_kernel/buffer.h"
#else
#include "../../common/buffer.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {

TEMPLATES_DEF
class SFAVectorService {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr uint32_t dVTemplateTypeInput = 576;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;

    // GM→UB route plan 预取窗口大小 (tokens)。
    // route plan 是 request 内连续 [K] int32 数组; 窗口按该尺寸滑窗连续 DataCopyPad 进
    // gatherPlanBuf, 窗口内用 UB GetValue 替代 per-token 串行标量 GM 随机读 (gather 阶段
    // 每 token 原本 2 次 GM 随机读)。窗口数据由 RefillRouteWindow 维护, 由 GetRealCmpS2Idx 消费。
    static constexpr uint32_t kRoutePrefetchWindowTokens = 1024U;

    // ==================== Functions ======================
    __aicore__ inline SFAVectorService() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const SparseFlashAttentionTilingDataMla *__restrict tiling,
                                        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
                                        __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            if (actualSeqLengthsQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ);
            }
            if (actualSeqLengths != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths);
            }
            this->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                       __gm__ uint8_t *softmaxSum, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                            __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *softmaxMax,
                                            __gm__ uint8_t *softmaxSum);
    __aicore__ inline void InitDsaSources(__gm__ uint8_t *hotKey, __gm__ uint8_t *hotKeyRope,
                                          __gm__ uint8_t *hotBlockTable, __gm__ uint8_t *hostKey,
                                          __gm__ uint8_t *hostKeyRope, __gm__ uint8_t *hostBlockTable,
                                          __gm__ uint8_t *queryPoolEntries, __gm__ uint8_t *hotSourceRows,
                                          __gm__ uint8_t *routePlan,
                                          __gm__ uint8_t *installStagingKey,
                                          __gm__ uint8_t *installStagingRope);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

private:
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
        int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo,
        ConstInfo &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb,
        int64_t startRow, int64_t keyOffset, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInRouteSingle(LocalTensor<KV_T> kvInUb,
        int64_t startRow, int64_t packedRoute, const RunInfo &runInfo,
        ConstInfo &constInfo);
    // GM→UB route plan 滑动窗口预取 (A5 vector 优化, 替代 per-token 串行标量 GM 随机读)。
    // 每次拉取最多 kRoutePrefetchWindowTokens 个连续 int32 到 gatherPlanBuf 窗口,
    // GetRealCmpS2Idx 在窗口内用 UB GetValue, 非窗口内回退 GM。
    __aicore__ inline void RefillRouteWindow(int64_t routeGmBase,
        int64_t topkBS1Idx, const ConstInfo &constInfo);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo,
        ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
        int64_t vec2S1Idx, int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void CopyFALseToGm(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &constInfo);

    TPipe *tPipe;
    const SparseFlashAttentionTilingDataMla *__restrict tilingData;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<KV_T> dsaHotKeyGm;
    GlobalTensor<KV_T> dsaHotRopeGm;
    GlobalTensor<KV_T> dsaHostKeyGm;
    GlobalTensor<KV_T> dsaHostRopeGm;
    GlobalTensor<int32_t> dsaHotBlockTableGm;
    GlobalTensor<int32_t> dsaHostBlockTableGm;
    GlobalTensor<int32_t> dsaQueryPoolEntriesGm;
    GlobalTensor<int32_t> dsaHotSourceRowsGm;
    GlobalTensor<int32_t> dsaRoutePlanGm;
    GlobalTensor<KV_T> dsaInstallStagingKeyGm;
    GlobalTensor<KV_T> dsaInstallStagingRopeGm;
    bool dsaRouteEnabled = false;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    LocalTensor<float> lseUb;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TBuf<> stage0OutBuf[2];
    TBuf<> stage2OutBuf;
    // GM→UB route plan 预取窗口缓冲区 (尺寸 = kRoutePrefetchWindowTokens * int32)。
    TBuf<> gatherPlanBuf;
    // ---- route 预取窗口状态 ----
    LocalTensor<int32_t> routePrefetchUb;   // 窗口数据的 UB 视图 (gatherPlanBuf 的 int32 视图)
    bool routePrefetchActive = false;       // 本 request 预取会话是否有效 (拉满后置 false)
    int64_t routeAlignedElem = 0;           // 当前窗口首元素在 GM 的绝对索引 (32B floor 对齐)
    int64_t routeWindowElems = 0;           // 当前窗口实际拉取的有效元素数 (>= 0, 8 的整数倍)
    int64_t routePrefetched = 0;            // 相对 sparseS2Start 已推进的元素数 (下一窗口起点)
    uint32_t routeFillEvtId = 0;            // MTE2→scalar 事件: 窗口 DataCopyPad 写全后再允许标量读
    int64_t hotSourceRowCached = INT64_MIN; // 每 request 缓存一次的 hot source row (INT64_MIN=未缓存)
    int64_t hostPoolEntryCached = INT64_MIN;
    TEventID mte3ToVAttnOutId; // 存放MTE3_V的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID vToMte3AttnOutId; // 存放V_MTE3的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID mte3ToVLseOutId; // 存放MTE3_V的eventId, 用于V1 LSE拷出阶段的同步
    TEventID vToMte3LseOutId; // 存放V_MTE3的eventId, 用于V1 LSE拷出阶段的同步
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> dequantScaleBuff;
    TBuf<> lseBuf;

    TEventID mte2ToV;
    TEventID mte2ToMte3[2];
    TEventID mte3ToMte2[2];

    bool isSinks = false;
    T negativeFloatScalar;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;

    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
SFAVectorService<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
    int64_t s2IdxInBase, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = runInfo.boIdx == 0 ? 0 : cuSeqlensQGm.GetValue(runInfo.boIdx - 1);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.sparseBlockCount +
            runInfo.s1oIdx * constInfo.sparseBlockCount; // B, S1, N2(1), K
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount;
    int64_t topkKIdx = s2IdxInBase + cmpS2LoopCnt * constInfo.s2BaseSize;
    if (unlikely(topkKIdx >= constInfo.sparseBlockCount)) {
        token0Idx = -1;
    } else {
        if (dsaRouteEnabled) {
            const int64_t ubIdx = topkBS1Idx + topkKIdx - this->routeAlignedElem;
            const int32_t packed =
                (this->routePrefetchActive && ubIdx >= 0 &&
                 ubIdx < this->routeWindowElems)  // 窗口命中读 UB, 越界回退 GM
                    ? this->routePrefetchUb.GetValue(ubIdx)
                    : dsaRoutePlanGm.GetValue(topkBS1Idx + topkKIdx);
            token0Idx = (static_cast<uint32_t>(packed) >> 29U) == 0U
                            ? -1
                            : static_cast<int64_t>(static_cast<uint32_t>(packed));
        } else {
            token0Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
        }
    }
    topkKIdx += 1;
    if (unlikely((topkKIdx >= constInfo.sparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        if (dsaRouteEnabled) {
            const int64_t ubIdx = topkBS1Idx + topkKIdx - this->routeAlignedElem;
            const int32_t packed =
                (this->routePrefetchActive && ubIdx >= 0 &&
                 ubIdx < this->routeWindowElems)  // 窗口命中读 UB, 越界回退 GM
                    ? this->routePrefetchUb.GetValue(ubIdx)
                    : dsaRoutePlanGm.GetValue(topkBS1Idx + topkKIdx);
            token1Idx = (static_cast<uint32_t>(packed) >> 29U) == 0U
                            ? -1
                            : static_cast<int64_t>(static_cast<uint32_t>(packed));
        } else {
            token1Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
int64_t SFAVectorService<TEMPLATE_ARGS>::GetkeyOffset(
    int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) *
            static_cast<int64_t>(blockSize) +
            blkTableOffset; // BlockNum, BlockSize, N(1), D
    } else {
        if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
            realkeyOffset = (runInfo.boIdx * constInfo.s2Size + s2Idx); // BSN(1)D
        } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            int64_t batchKvStart = (runInfo.boIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(runInfo.boIdx - 1);
            realkeyOffset = (batchKvStart + s2Idx);
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
SFAVectorService<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
    int64_t keyOffset, ConstInfo &constInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;

    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    uint32_t combineBytes = 512 * sizeof(KV_T);
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * 512], intriParams, padParams);

    intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeRope *sizeof(KV_T);
    intriParams.dstStride = 512 / BUFFER_SIZE_BYTE_32B;
    DataCopyPad(kvInUb[startRow * 576 + 512], keyRopeGm[keyOffset * 64], intriParams, padParams); // combineDimAlign
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
uint32_t SFAVectorService<TEMPLATE_ARGS>::CopyInRouteSingle(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t packedRoute,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (packedRoute < 0) {
        return 0;
    }
    constexpr uint32_t ROUTE_KIND_SHIFT = 29U;
    constexpr uint32_t ROUTE_INDEX_MASK = (1U << ROUTE_KIND_SHIFT) - 1U;
    constexpr uint32_t SRC_POOL_HIT = 1U;
    constexpr uint32_t SRC_HOST_MISS = 2U;
    constexpr uint32_t SRC_LIVE_HOT = 3U;
    const uint32_t route = static_cast<uint32_t>(packedRoute);
    const uint32_t kind = route >> ROUTE_KIND_SHIFT;
    const bool useHot = kind == SRC_POOL_HIT || kind == SRC_LIVE_HOT;
    if (!useHot && kind != SRC_HOST_MISS) {
        return 0;
    }
    const uint32_t tableWidth = useHot ? tilingData->baseParams.dsaHotTableWidth
                                       : tilingData->baseParams.dsaHostTableWidth;
    const uint32_t poolCapacity = useHot ? tilingData->baseParams.dsaHotPoolCapacity
                                         : tilingData->baseParams.dsaHostPoolCapacity;
    const uint32_t physicalBlocks = useHot ? tilingData->baseParams.dsaHotPhysicalBlocks
                                           : tilingData->baseParams.dsaHostPhysicalBlocks;
    const int32_t sourceRow = useHot
                                  ? (this->hotSourceRowCached != INT64_MIN
                                         ? static_cast<int32_t>(this->hotSourceRowCached)
                                         : dsaHotSourceRowsGm.GetValue(runInfo.boIdx))
                                  : (this->hostPoolEntryCached != INT64_MIN
                                         ? static_cast<int32_t>(this->hostPoolEntryCached)
                                         : dsaQueryPoolEntriesGm.GetValue(runInfo.boIdx));
    if (sourceRow < 0 || static_cast<uint32_t>(sourceRow) >= poolCapacity) {
        return 0;
    }
    const uint32_t sourcePosition = route & ROUTE_INDEX_MASK;
    const uint32_t logicalBlock = sourcePosition / blockSize;
    const uint32_t blockOffset = sourcePosition % blockSize;
    if (logicalBlock >= tableWidth) {
        return 0;
    }
    const uint64_t tableIndex = static_cast<uint64_t>(sourceRow) * tableWidth + logicalBlock;
    const int32_t physicalBlock = useHot ? dsaHotBlockTableGm.GetValue(tableIndex)
                                         : dsaHostBlockTableGm.GetValue(tableIndex);
    if (physicalBlock < 0 || static_cast<uint32_t>(physicalBlock) >= physicalBlocks) {
        return 0;
    }
    const uint64_t sourceToken = static_cast<uint64_t>(physicalBlock) * blockSize + blockOffset;
    DataCopyExtParams params;
    params.blockCount = 1;
    params.dstStride = 0;
    params.srcStride = 0;
    params.blockLen = constInfo.dSizeNope * sizeof(KV_T);
    DataCopyPadExtParams<KV_T> padParams;
    if (useHot) {
        DataCopyPad(kvInUb[startRow * 576], dsaHotKeyGm[sourceToken * constInfo.dSizeNope], params, padParams);
    } else {
        DataCopyPad(kvInUb[startRow * 576], dsaHostKeyGm[sourceToken * constInfo.dSizeNope], params, padParams);
    }
    params.blockLen = constInfo.dSizeRope * sizeof(KV_T);
    params.dstStride = constInfo.dSizeNope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
    if (useHot) {
        DataCopyPad(kvInUb[startRow * 576 + 512], dsaHotRopeGm[sourceToken * constInfo.dSizeRope], params, padParams);
    } else {
        DataCopyPad(kvInUb[startRow * 576 + 512], dsaHostRopeGm[sourceToken * constInfo.dSizeRope], params, padParams);
    }
    return 1;
}

// GM→UB route plan 预取窗口填充: 拉取 [routeGmBase + routePrefetched, +win) 段。
// - GM 源 32B floor 对齐 (DataCopyPad 源地址需 32B 对齐), 对齐裕量数据不消费,
//   对应元素由 GetRealCmpS2Idx 的越界判断走 GM fallback;
// - 拉取长度 clamp 到 request 末尾, 避免跨出 route plan 数组越界;
// - 只拉完整的 8-token(32B) 段, 尾部不足 8 个不拉 (对应元素读 GM), 避免尾 burst 越出窗口末尾;
// - Set+Wait MTE2_S: 窗口数据由 MTE2 写入、由标量 GetValue 读取, 必须等 MTE2→scalar 事件
//   才返回, 保证标量读看到写全的数据。
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::RefillRouteWindow(
    int64_t routeGmBase, int64_t topkBS1Idx, const ConstInfo &constInfo)
{
    const int64_t avail = sparseCalSize - routePrefetched;
    if (avail <= 0) {
        routePrefetchActive = false;  // 本 request 的 route 已全部拉取, 后续读走 GM
        return;
    }
    const int64_t gmElem = routeGmBase + routePrefetched;
    const int64_t gmByte = gmElem * 4;
    const int64_t alignedStart = (gmByte & ~INT64_C(31)) >> 2;  // GM 源 32B floor 对齐
    routeAlignedElem = alignedStart;
    int64_t win = avail;
    if (win > kRoutePrefetchWindowTokens) {
        win = kRoutePrefetchWindowTokens;
    }
    // clamp 拉取上界到 request 内合法范围 (route plan 是 [req] 的 K 个 int32), 防跨 buffer 越界
    const int64_t reqEnd = topkBS1Idx + constInfo.sparseBlockCount;
    int64_t copyEnd = alignedStart + win;
    if (copyEnd > reqEnd) {
        copyEnd = reqEnd;
    }
    // 只拉完整的 8-token(32B) 段, 尾部不足 8 个不拉 (对应元素读 GM fallback),
    // 避免 DataCopyPad 尾 burst 越出窗口末尾。
    const int64_t copyElems8 = ((copyEnd - alignedStart) >> 3) << 3;
    if (copyElems8 <= 0) {
        routeWindowElems = 0;
        routePrefetched += win;
        return;
    }
    DataCopyExtParams cdp;
    cdp.blockCount = static_cast<uint16_t>(copyElems8 / 8);
    cdp.blockLen = static_cast<uint16_t>(32);  // 每块 32B = 8 个 int32
    cdp.srcStride = 0;
    cdp.dstStride = 0;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(routePrefetchUb[0], dsaRoutePlanGm[alignedStart], cdp, padParams);
    SetFlag<HardEvent::MTE2_S>(routeFillEvtId);
    WaitFlag<HardEvent::MTE2_S>(routeFillEvtId);
    routeWindowElems = copyElems8;
    routePrefetched += win;
}
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
uint32_t SFAVectorService<TEMPLATE_ARGS>::CopyInKvSparse(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
    int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (dsaRouteEnabled) {
        uint32_t copied = CopyInRouteSingle(kvInUb, startRow, token0Idx, runInfo, constInfo);
        copied += CopyInRouteSingle(kvInUb, startRow + copied, token1Idx, runInfo, constInfo);
        return copied;
    }
    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    int64_t blkTableSrcStride =
        ((keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) :
        (keyOffset1 - keyOffset0)) - constInfo.sparseBlockSize);
    int64_t keySrcStride = blkTableSrcStride * constInfo.dSizeNope * sizeof(KV_T);
    int64_t keyRopeSrcStride = blkTableSrcStride * constInfo.dSizeRope * sizeof(KV_T);
    if (unlikely(keyOffset1 < 0)) {
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
    } else if (keySrcStride >= INT32_MAX || keySrcStride < 0 || constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, constInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeNope *sizeof(KV_T);
        intriParams.dstStride = constInfo.dSizeRope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }
        DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * constInfo.dSizeNope],
                    intriParams, padParams); // combineDimAlign

        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeRope *sizeof(KV_T);
        intriParams.dstStride = constInfo.dSizeNope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        intriParams.srcStride = keyRopeSrcStride;
        DataCopyPad(kvInUb[startRow * 576 + 512], keyRopeGm[keyOffset * constInfo.dSizeRope],
                    intriParams, padParams); // combineDimAlign
    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float* ubDstAddr, __ubuf__ int8_t* ubSrcAddr, uint32_t dealRowCount)
{
    MicroAPI::RegTensor<fp8_e8m0_t> vScale0;
    MicroAPI::RegTensor<fp8_e8m0_t> vScale1;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res0;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res1;
    MicroAPI::RegTensor<float> vScalefp32Res0;
    MicroAPI::RegTensor<float> vScalefp32Res1;
    __ubuf__ int8_t* ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float* ubDstAddrTmp = ubDstAddr;
    MicroAPI::MaskReg bf16TypeMaskAll = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t>&)vScale0, ubScaleSrcAddrTemp, 640);

        MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        MicroAPI::StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vScalefp32Res0, 64, bf16TypeMaskAll);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> kvOutUb,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * 576], kvOutUb, dealRow * 576);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = 0;
            } else {
                sparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2);
            }
        }
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize);
        sparseS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
        sparseCalSize = sparseS2End - sparseS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    blockSize = constInfo.oriBlockSize;
    maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;

    CalSparseCalSize(runInfo, constInfo);
    ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo, startPos);
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE3>(15);
        CrossCoreWaitFlag<0, PIPE_MTE3>(15);
    }
    outputL1.SetCrossCore();
    v0ResGm.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    if (sparseCalSize == 0) {
        return;
    }
    bool meetEnd = false;
    int64_t s2Start = sparseS2Start;
    int64_t s2 = sparseS2Start;
    int64_t token0Idx;
    int64_t token1Idx; // 拷贝进入的两个token的index
    // 处理一个s2的base块
    uint32_t pingPong = 0;
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        const uint64_t actualSeqQPrefixSum = runInfo.boIdx == 0
            ? 0 : cuSeqlensQGm.GetValue(runInfo.boIdx - 1);
        topkBS1Idx = (actualSeqQPrefixSum + runInfo.s1oIdx) *
            constInfo.sparseBlockCount;
    } else {
        topkBS1Idx = runInfo.boIdx * constInfo.s1Size *
            constInfo.sparseBlockCount +
            runInfo.s1oIdx * constInfo.sparseBlockCount;
    }
    // ===== GM→UB route plan 预取窗口 (A5 vector 优化) =====
    // route plan 是该 request 内连续 [K] int32 数组; 整段连续 DataCopyPad 进
    // gatherPlanBuf 窗口 (连续 burst 远快于逐点标量 GetValue), 窗口内用 UB
    // GetValue 替代每 token 的串行标量 GM 随机读, 消除主要 stall 来源之一。
    routePrefetchActive = dsaRouteEnabled;
    if (dsaRouteEnabled) {
        routePrefetchUb = this->gatherPlanBuf.template Get<int32_t>();
        routePrefetched = 0;
        const int64_t routeGmBase = topkBS1Idx +
            static_cast<int64_t>(runInfo.s2LoopCount) * constInfo.s2BaseSize +
            sparseS2Start;
        RefillRouteWindow(routeGmBase, topkBS1Idx, constInfo);
        hotSourceRowCached = dsaHotSourceRowsGm.GetValue(runInfo.boIdx);
        hostPoolEntryCached = dsaQueryPoolEntriesGm.GetValue(runInfo.boIdx);
    }
    while ((s2 < sparseS2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        uint32_t hostMissMask = 0U;
        uint32_t stagingRouteIndices[16];
        // 窗口推进: s2 已消费到当前窗口末尾, 拉下一窗。
        if (routePrefetchActive &&
            (s2 - sparseS2Start) >= routePrefetched) {
            const int64_t routeGmBase = topkBS1Idx +
                static_cast<int64_t>(runInfo.s2LoopCount) *
                    constInfo.s2BaseSize +
                sparseS2Start;
            RefillRouteWindow(routeGmBase, topkBS1Idx, constInfo);
        }
        // 1、copy kv in, gm ->ub
        LocalTensor<Q_T> stage0OutUb = this->stage0OutBuf[pingPong].template Get<Q_T>();
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
        while (dealRow < Min(16, sparseCalSize) && s2 < sparseS2End) { // 拷贝满16行或者遇到-1
            const int64_t pairS2 = s2;
            s2 += 2; // 每次搬运2行
            if (dsaRouteEnabled) {
                GetRealCmpS2Idx(token0Idx, token1Idx, pairS2, runInfo,
                                constInfo);
                if (token0Idx == -1 && token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
                const uint32_t copied0 = CopyInRouteSingle(
                    stage0OutUb, dealRow, token0Idx, runInfo, constInfo);
                if (copied0 != 0U) {
                    stagingRouteIndices[dealRow] = static_cast<uint32_t>(
                        runInfo.s2LoopCount * constInfo.s2BaseSize + pairS2);
                    if ((static_cast<uint32_t>(token0Idx) >> 29U) == 2U) {
                        hostMissMask |= 1U << dealRow;
                    }
                    dealRow += copied0;
                }
                const uint32_t copied1 = CopyInRouteSingle(
                    stage0OutUb, dealRow, token1Idx, runInfo, constInfo);
                if (copied1 != 0U) {
                    stagingRouteIndices[dealRow] = static_cast<uint32_t>(
                        runInfo.s2LoopCount * constInfo.s2BaseSize + pairS2 + 1);
                    if ((static_cast<uint32_t>(token1Idx) >> 29U) == 2U) {
                        hostMissMask |= 1U << dealRow;
                    }
                    dealRow += copied1;
                }
                if (token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
            } else {
                GetRealCmpS2Idx(token0Idx, token1Idx, pairS2, runInfo,
                                constInfo);
                if (token0Idx == -1 && token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
                dealRow += CopyInKvSparse(stage0OutUb, dealRow, token0Idx,
                                          token1Idx, runInfo, constInfo);
                if (token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
            }
        }
        if (dealRow  == 0) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            pingPong ^= 1;
            return;
        }
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
        // 2、copy kv out, ub -> l1
        CopyOutKvUb2Gm(v0ResGm, stage0OutUb, dealRow, s2Start, runInfo, constInfo);
        if (dsaRouteEnabled && hostMissMask != 0U) {
            for (int64_t row = 0; row < dealRow; ++row) {
                if ((hostMissMask & (1U << row)) == 0U) {
                    continue;
                }
                const uint64_t stagingToken = static_cast<uint64_t>(
                    topkBS1Idx + stagingRouteIndices[row]);
                DataCopy(dsaInstallStagingKeyGm[
                             stagingToken * constInfo.dSizeNope],
                         stage0OutUb[row * 576], constInfo.dSizeNope);
                DataCopy(dsaInstallStagingRopeGm[
                             stagingToken * constInfo.dSizeRope],
                         stage0OutUb[row * 576 + 512],
                         constInfo.dSizeRope);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
        s2Start += dealRow;
        pingPong ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (runInfo.s2LoopCount == 0) { // sink 丢失首token信息，sink会增加首token信息，维度是n1
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSize, // 实际的计算有效元素，
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>(s2BaseSize * constInfo.dSizeV);
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * \
            (BLOCK_BYTE / sizeof(Q_T)) * (runInfo.mRealSize - runInfo.halfMRealSize)],
            stage1CastTensor, {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize,
            (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
            (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    if (runInfo.s2LoopCount != 0) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if (constInfo.returnSoftmaxLse && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        CopyFALseToGm(runInfo, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

    WaitFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::Bmm2DataCopyOut (RunInfo &runInfo, ConstInfo &constInfo,
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3AttnOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3AttnOutId);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;
    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (constInfo.aivIdx * singleCoreSize < totalOutputSize && singleInitOutputSize > 0) {
            matmul::InitOutput<OUTPUT_T>(
                this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
        }
    }

    if (constInfo.returnSoftmaxLse) {
        uint64_t totalReturnSoftmaxSize = 0;
        if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
            totalReturnSoftmaxSize = constInfo.bSize *  constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            totalReturnSoftmaxSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize; // (N2,T1,G)
        }
        if (coreNum != 0 && totalReturnSoftmaxSize > 0) {
            uint64_t singleCoreSoftmaxSize = (totalReturnSoftmaxSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
            uint64_t tailSoftmaxSize = totalReturnSoftmaxSize - constInfo.aivIdx * singleCoreSoftmaxSize;
            uint64_t singleInitSoftmaxSize = tailSoftmaxSize < singleCoreSoftmaxSize ?
                                             tailSoftmaxSize : singleCoreSoftmaxSize;
            if (constInfo.aivIdx * singleCoreSoftmaxSize < totalReturnSoftmaxSize && singleInitSoftmaxSize > 0) {
                matmul::InitOutput<float>(this->softmaxSumGm[constInfo.aivIdx * singleCoreSoftmaxSize], singleInitSoftmaxSize, 0);
                matmul::InitOutput<float>(this->softmaxMaxGm[constInfo.aivIdx * singleCoreSoftmaxSize], singleInitSoftmaxSize, 0);
            }
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                                  __gm__ uint8_t *softmaxSum, ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)(softmaxSum));
        this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)(softmaxMax));
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                                       __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                                       __gm__ uint8_t *blockTable, __gm__ uint8_t *softmaxMax,
                                                       __gm__ uint8_t *softmaxSum)
{
    keyGm.SetGlobalBuffer((__gm__ KV_T *)(key));
    if constexpr (isPa) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);;
    }
    sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    keyRopeGm.SetGlobalBuffer((__gm__ KV_T *)(keyRope));
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::InitDsaSources(
    __gm__ uint8_t *hotKey, __gm__ uint8_t *hotKeyRope,
    __gm__ uint8_t *hotBlockTable, __gm__ uint8_t *hostKey,
    __gm__ uint8_t *hostKeyRope, __gm__ uint8_t *hostBlockTable,
    __gm__ uint8_t *queryPoolEntries, __gm__ uint8_t *hotSourceRows,
    __gm__ uint8_t *routePlan, __gm__ uint8_t *installStagingKey,
    __gm__ uint8_t *installStagingRope)
{
    dsaHotKeyGm.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(hotKey));
    dsaHotRopeGm.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(hotKeyRope));
    dsaHostKeyGm.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(hostKey));
    dsaHostRopeGm.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(hostKeyRope));
    dsaHotBlockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(hotBlockTable));
    dsaHostBlockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(hostBlockTable));
    dsaQueryPoolEntriesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryPoolEntries));
    dsaHotSourceRowsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(hotSourceRows));
    dsaRoutePlanGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(routePlan));
    dsaInstallStagingKeyGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ KV_T *>(installStagingKey));
    dsaInstallStagingRopeGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ KV_T *>(installStagingRope));
    dsaRouteEnabled = true;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::CopyFALseToGm(RunInfo &runInfo, ConstInfo &constInfo)
{
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();

    size_t alignedSize = (sizeof(float) * runInfo.halfMRealSize + 31) / 32 * 32 / sizeof(float);

    int64_t lseOffset = runInfo.softmaxLseOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = sizeof(float) * runInfo.halfMRealSize;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    // 拷贝 softmaxMaxUb -> GM
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    DataCopy(lseUb, maxUb, alignedSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxMaxGm[lseOffset], lseUb, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);

    // 拷贝 softmaxSumUb -> GM
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    DataCopy(lseUb, sumUb, alignedSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxSumGm[lseOffset], lseUb, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo &constInfo)
{
    LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512); // sinksBuf内存申请512B
    tPipe->InitBuffer(lseBuf, 512); // lseBuf内存申请512B
    lseUb = this->lseBuf.template Get<float>();

    tPipe->InitBuffer(stage0OutBuf[0], 576 * 16 * sizeof(KV_T));
    tPipe->InitBuffer(stage0OutBuf[1], 576 * 16 * sizeof(KV_T));
    tPipe->InitBuffer(gatherPlanBuf,
        kRoutePrefetchWindowTokens * sizeof(uint32_t));

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVAttnOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);

    vToMte3AttnOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();

    // MTE2_S: MTE2→scalar(program) pipe 事件。窗口 UB 读走 GetValue 标量管线,
    // 必须等 MTE2_S 才能保证 GetValue 看到 DataCopyPad 写全的窗口数据。
    // (MTE2_V 只管 MTE2→vector, 标量读会拿到上次窗口的陈旧数据。)
    routeFillEvtId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
    mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte3ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    mte3ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
    mte2ToMte3[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToMte3[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    // TODO参数整改
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    sharedParams.bSize = sparseAttnSharedkvBaseParams.batchSize;
    sharedParams.n2Size = 1;
    sharedParams.gSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup;
    sharedParams.s1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    sharedParams.s2Size = sparseAttnSharedkvBaseParams.seqSize;
    sharedParams.sparseBlockCount = sparseAttnSharedkvBaseParams.sparseBlockCount;
    sharedParams.cmpRatio = 1; // 走sparse， 但不压缩
    sharedParams.oriMaskMode = sparseAttnSharedkvBaseParams.sparseMode;
    sharedParams.oriWinLeft = -1;
    sharedParams.oriWinRight = 0;
    sharedParams.layoutType = sparseAttnSharedkvBaseParams.outputLayout;
    sharedParams.dSizeRope = 64;
    sharedParams.softmaxScale = sparseAttnSharedkvBaseParams.scaleValue;
    sharedParams.dSize = 512;
    sharedParams.dSizeVInput = 512;
    sharedParams.usedCoreNum = this->tilingData->singleCoreParams.usedCoreNum;

    // pageAttention, rope在C侧搬运时使用
    if constexpr (isPa) {
        sharedParams.oriBlockSize = sparseAttnSharedkvBaseParams.blockSize;
        sharedParams.oriMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams.maxBlockNumPerBatch;
    }

    // actQ->TND, actKV pa场景任意layout均有
    sharedParams.isActualSeqLengthsNull = sparseAttnSharedkvBaseParams.isActualLenDimsNull;
    sharedParams.isActualSeqLengthsKVNull = sparseAttnSharedkvBaseParams.isActualLenDimsKVNull;
    sharedParams.returnSoftmaxLse = sparseAttnSharedkvBaseParams.returnSoftmaxLse;
    sharedParams.needInit = 0;
    for (uint32_t bIdx = 0; bIdx < sharedParams.bSize; bIdx++) {
        int64_t s2Size;
        if constexpr (KV_LAYOUT_T == SFA_LAYOUT::TND) {
            s2Size = bIdx == 0 ? actualSeqLengthsKVGm.GetValue(bIdx) : \
            actualSeqLengthsKVGm.GetValue(bIdx) - actualSeqLengthsKVGm.GetValue(bIdx - 1);
        } else {
            if (sharedParams.isActualSeqLengthsKVNull) {
                s2Size = sharedParams.s2Size;
            } else {
                s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
            }
        }
        int64_t s1Size;
        if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            s1Size = bIdx == 0 ? cuSeqlensQGm.GetValue(bIdx) : \
            cuSeqlensQGm.GetValue(bIdx) - cuSeqlensQGm.GetValue(bIdx - 1);
        } else {
            if (sharedParams.isActualSeqLengthsNull) {
                s1Size = sharedParams.s1Size;
            } else {
                s1Size = cuSeqlensQGm.GetValue(bIdx);
            }
        }
        if (s1Size > s2Size || (LAYOUT_T == SFA_LAYOUT::BSND && s1Size < sharedParams.s1Size)) {
            sharedParams.needInit = 1;
            break;
        }
    }

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
#pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}


TEMPLATES_DEF class SFAVectorServiceDummy {
public:
    __aicore__ inline SFAVectorServiceDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                       __gm__ uint8_t *softmaxSum, ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                            __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *softmaxMax,
                                            __gm__ uint8_t *softmaxSum) {}
    __aicore__ inline void InitDsaSources(__gm__ uint8_t *hotKey, __gm__ uint8_t *hotKeyRope,
                                          __gm__ uint8_t *hotBlockTable, __gm__ uint8_t *hostKey,
                                          __gm__ uint8_t *hostKeyRope, __gm__ uint8_t *hostBlockTable,
                                          __gm__ uint8_t *queryPoolEntries, __gm__ uint8_t *hotSourceRows,
                                          __gm__ uint8_t *routePlan,
                                          __gm__ uint8_t *installStagingKey,
                                          __gm__ uint8_t *installStagingRope) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const SparseFlashAttentionTilingDataMla *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, __gm__ uint8_t *actualSeqLengthsQ,
        __gm__ uint8_t *actualSeqLengths) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
};
}
#endif // SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
