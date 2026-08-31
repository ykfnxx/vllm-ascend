/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
struct JitterLevel {
    uint32_t threshold;
    uint32_t delayUs;
};

constexpr JitterLevel kJitterLevels[] = {
    {0xFFFFFFFFU / 100U + 1U, 169U},
    {0xFFFFFFFFU / 1000U + 1U, 228U},
    {0xFFFFFFFFU / 10000U + 1U, 289U},
    {0xFFFFFFFFU / 100000U + 1U, 354U},
    {0xFFFFFFFFU / 1000000U + 1U, 502U},
    {0xFFFFFFFFU / 10000000U + 1U, 13070U},
};
constexpr uint32_t kJitterLevelCount =
    sizeof(kJitterLevels) / sizeof(kJitterLevels[0]);
constexpr uint32_t kBaseDelayUs = 79U;
constexpr uint32_t kItersPerUs = 189U;

template <typename KvT, typename RopeT>
class KernelAsuKvGatherDirectV2 {
public:
    __aicore__ inline void Init(
        GM_ADDR destinationKvCache,
        GM_ADDR destinationKRope,
        GM_ADDR hotBlockTablePool,
        GM_ADDR sourceKvCache,
        GM_ADDR sourceKRope,
        GM_ADDR sourceBlockTable,
        GM_ADDR requestRows,
        GM_ADDR queryStartLoc,
        GM_ADDR semanticTopk,
        GM_ADDR mappedIndices,
        GM_ADDR gatherMask,
        const AsuKvGatherDirectV2TilingData* tiling,
        TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        destinationKv_.SetGlobalBuffer(
            reinterpret_cast<__gm__ KvT*>(destinationKvCache));
        destinationRope_.SetGlobalBuffer(
            reinterpret_cast<__gm__ RopeT*>(destinationKRope));
        hotBlockTable_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(hotBlockTablePool));
        sourceKv_.SetGlobalBuffer(
            reinterpret_cast<__gm__ KvT*>(sourceKvCache));
        sourceRope_.SetGlobalBuffer(
            reinterpret_cast<__gm__ RopeT*>(sourceKRope));
        sourceBlockTable_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(sourceBlockTable));
        requestRows_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(requestRows));
        queryStartLoc_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
        semanticTopk_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(semanticTopk));
        mappedIndices_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(mappedIndices));
        gatherMask_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(gatherMask));
        pipe_->InitBuffer(
            kvBuffer_, tiling_->kvRecordElements * sizeof(KvT));
        pipe_->InitBuffer(
            ropeBuffer_, tiling_->ropeRecordElements * sizeof(RopeT));
        mte2ToMte3_ = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE2_MTE3));
        mte3ToMte2_ = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        mte3Done_ = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE3_V));
        rngState_ = tiling_->jitterSeed ^ (GetBlockIdx() * 2654435761U);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core_id = GetBlockIdx();
        const uint32_t core_count = GetBlockNum();
        const uint64_t pair_count =
            static_cast<uint64_t>(tiling_->queryNum) *
            tiling_->topkWidth;
        const uint64_t pairs_per_core = pair_count / core_count;
        const uint64_t extra = pair_count % core_count;
        const uint64_t pair_begin =
            static_cast<uint64_t>(core_id) * pairs_per_core +
            (core_id < extra ? core_id : extra);
        const uint64_t pair_end =
            pair_begin + pairs_per_core + (core_id < extra ? 1U : 0U);
        if (pair_begin == pair_end) {
            return;
        }

        auto kv_local = kvBuffer_.Get<KvT>();
        auto rope_local = ropeBuffer_.Get<RopeT>();
        kv_local.SetSize(tiling_->kvRecordElements);
        rope_local.SetSize(tiling_->ropeRecordElements);

        uint32_t query_id =
            static_cast<uint32_t>(pair_begin / tiling_->topkWidth);
        uint32_t request_id = 0U;
        while (request_id < tiling_->reqNum &&
               queryStartLoc_.GetValue(request_id + 1U) <=
                   static_cast<int32_t>(query_id)) {
            ++request_id;
        }
        int32_t row = request_id < tiling_->reqNum
                          ? requestRows_.GetValue(request_id)
                          : -1;
        uint32_t miss_count = 0U;

        for (uint64_t pair = pair_begin; pair < pair_end; ++pair) {
            const uint32_t current_query =
                static_cast<uint32_t>(pair / tiling_->topkWidth);
            if (current_query != query_id) {
                query_id = current_query;
                while (request_id < tiling_->reqNum &&
                       queryStartLoc_.GetValue(request_id + 1U) <=
                           static_cast<int32_t>(query_id)) {
                    ++request_id;
                }
                row = request_id < tiling_->reqNum
                          ? requestRows_.GetValue(request_id)
                          : -1;
            }
            if (row < 0 ||
                static_cast<uint32_t>(row) >= tiling_->poolCapacity ||
                gatherMask_.GetValue(pair) == 0) {
                continue;
            }

            const int32_t token = semanticTopk_.GetValue(pair);
            const int32_t mapped = mappedIndices_.GetValue(pair);
            if (token < 0 || mapped < 0) {
                continue;
            }
            const uint32_t source_logical =
                static_cast<uint32_t>(token) / tiling_->blockSize;
            const uint32_t source_offset =
                static_cast<uint32_t>(token) % tiling_->blockSize;
            const uint32_t destination_logical =
                static_cast<uint32_t>(mapped) / tiling_->blockSize;
            const uint32_t destination_offset =
                static_cast<uint32_t>(mapped) % tiling_->blockSize;
            if (source_logical >= tiling_->sourceTableWidth ||
                destination_logical >= tiling_->destinationTableWidth) {
                continue;
            }

            const uint64_t source_table_offset =
                static_cast<uint64_t>(row) * tiling_->sourceTableWidth +
                source_logical;
            const uint64_t destination_table_offset =
                static_cast<uint64_t>(row) *
                    tiling_->destinationTableWidth +
                destination_logical;
            const int32_t source_block =
                sourceBlockTable_.GetValue(source_table_offset);
            const int32_t destination_block =
                hotBlockTable_.GetValue(destination_table_offset);
            if (source_block < 0 || destination_block < 0 ||
                static_cast<uint32_t>(source_block) >=
                    tiling_->sourcePhysicalBlockCount ||
                static_cast<uint32_t>(destination_block) >=
                    tiling_->destinationPhysicalBlockCount) {
                continue;
            }

            const uint64_t source_token =
                static_cast<uint64_t>(source_block) *
                    tiling_->blockSize +
                source_offset;
            const uint64_t destination_token =
                static_cast<uint64_t>(destination_block) *
                    tiling_->blockSize +
                destination_offset;
            ++miss_count;
            DataCopy(
                kv_local,
                sourceKv_[source_token * tiling_->kvRecordElements],
                tiling_->kvRecordElements);
            DataCopy(
                rope_local,
                sourceRope_[source_token * tiling_->ropeRecordElements],
                tiling_->ropeRecordElements);
            Sync<HardEvent::MTE2_MTE3>(mte2ToMte3_);
            DataCopy(
                destinationKv_[destination_token *
                               tiling_->kvRecordElements],
                kv_local,
                tiling_->kvRecordElements);
            DataCopy(
                destinationRope_[destination_token *
                                 tiling_->ropeRecordElements],
                rope_local,
                tiling_->ropeRecordElements);
            Sync<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        }
        SleepForSwapLatency(miss_count * 2U);
    }

private:
    __aicore__ inline void SleepForSwapLatency(uint32_t transfer_count)
    {
        if (tiling_->jitterEnable == 0U || transfer_count == 0U) {
            return;
        }
        uint32_t max_delay = 0U;
        for (uint32_t transfer = 0U;
             transfer < transfer_count;
             ++transfer) {
            rngState_ = rngState_ * 1664525U + 1013904223U;
            uint32_t delay = kBaseDelayUs;
            for (uint32_t level = kJitterLevelCount; level > 0U; --level) {
                if (rngState_ < kJitterLevels[level - 1U].threshold) {
                    delay = kJitterLevels[level - 1U].delayUs;
                    break;
                }
            }
            if (delay > max_delay) {
                max_delay = delay;
            }
        }
        Sync<HardEvent::MTE3_V>(mte3Done_);
        const uint32_t iterations = max_delay * kItersPerUs;
        volatile uint32_t sink = 0U;
        for (uint32_t index = 0U; index < iterations; ++index) {
            sink += index;
        }
        (void)sink;
    }

    template <HardEvent event>
    __aicore__ inline void Sync(event_t event_id)
    {
        SetFlag<event>(event_id);
        WaitFlag<event>(event_id);
    }

    const AsuKvGatherDirectV2TilingData* tiling_;
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> kvBuffer_;
    TBuf<TPosition::VECCALC> ropeBuffer_;
    event_t mte2ToMte3_;
    event_t mte3ToMte2_;
    event_t mte3Done_;
    uint32_t rngState_;
    GlobalTensor<KvT> destinationKv_;
    GlobalTensor<RopeT> destinationRope_;
    GlobalTensor<int32_t> hotBlockTable_;
    GlobalTensor<KvT> sourceKv_;
    GlobalTensor<RopeT> sourceRope_;
    GlobalTensor<int32_t> sourceBlockTable_;
    GlobalTensor<int32_t> requestRows_;
    GlobalTensor<int32_t> queryStartLoc_;
    GlobalTensor<int32_t> semanticTopk_;
    GlobalTensor<int32_t> mappedIndices_;
    GlobalTensor<int32_t> gatherMask_;
};
}  // namespace

extern "C" __global__ __aicore__ void asu_kv_gather_direct_v2(
    GM_ADDR destinationKvCache,
    GM_ADDR destinationKRope,
    GM_ADDR hotBlockTablePool,
    GM_ADDR sourceKvCache,
    GM_ADDR sourceKRope,
    GM_ADDR sourceBlockTable,
    GM_ADDR requestRows,
    GM_ADDR queryStartLoc,
    GM_ADDR semanticTopk,
    GM_ADDR mappedIndices,
    GM_ADDR gatherMask,
    GM_ADDR destinationKvCacheOut,
    GM_ADDR destinationKRopeOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)destinationKvCacheOut;
    (void)destinationKRopeOut;
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelAsuKvGatherDirectV2<
        DTYPE_SOURCE_KV_CACHE,
        DTYPE_SOURCE_K_ROPE> op;
    op.Init(destinationKvCache,
            destinationKRope,
            hotBlockTablePool,
            sourceKvCache,
            sourceKRope,
            sourceBlockTable,
            requestRows,
            queryStartLoc,
            semanticTopk,
            mappedIndices,
            gatherMask,
            &tilingData,
            &pipe);
    op.Process();
}
