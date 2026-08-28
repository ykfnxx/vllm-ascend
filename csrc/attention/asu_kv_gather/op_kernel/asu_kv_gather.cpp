#include "kernel_operator.h"

using namespace AscendC;

namespace {

// ---- swap 延迟抖动配置 (编译期) ----
// 模拟从 swap 读取 KV/RoPE 的概率性延迟抖动: 7 级尾延迟分布。
// 每级 = (32bit 均匀随机数阈值, 延迟 us); 阈值由概率分母按
// threshold = 2^32/denominator 计算 (P(rng<threshold) ≈ 1/denominator)。
// 概率分母依次为 100, 1000, ..., 10000000 (10x 递减), 对应真实 swap
// 读的 99% 基础延迟 + 1%/0.1%/... 尾延迟分位
// (CDF 尾部模型: P(delay >= level.delayUs) = 1/分母, 采样从最小阈值开始)。
struct SwapJitterLevel {
    uint32_t threshold;
    uint32_t delayUs;
};
constexpr SwapJitterLevel kSwapJitterLevels[] = {
    {0xFFFFFFFFU / 100U + 1U, 169U},
    {0xFFFFFFFFU / 1000U + 1U, 228U},
    {0xFFFFFFFFU / 10000U + 1U, 289U},
    {0xFFFFFFFFU / 100000U + 1U, 354U},
    {0xFFFFFFFFU / 1000000U + 1U, 502U},
    {0xFFFFFFFFU / 10000000U + 1U, 13070U},
};
constexpr uint32_t kSwapJitterLevelCount =
    sizeof(kSwapJitterLevels) / sizeof(kSwapJitterLevels[0]);
constexpr uint32_t kSwapJitterBaseUs = 79U;
// 编译期开关: true=注入随机忙等, false=关闭
constexpr bool kSwapJitterEnabled = true;
// 忙等迭代数/微秒: 已在 238 (Ascend910B2) 上通过 msprof 校准 (2026-08-18)。
constexpr uint32_t kItersPerUs = 189U;

template <typename KvT, typename RopeT, bool ResidentInit>
class KernelAsuKvGather {
public:
    __aicore__ inline void Init(
        GM_ADDR destinationKvCache,
        GM_ADDR destinationKRope,
        GM_ADDR destinationBlockTable,
        GM_ADDR sourceKvCache,
        GM_ADDR sourceKRope,
        GM_ADDR sourceBlockTable,
        GM_ADDR reqPoolEntries,
        GM_ADDR tokenPositions,
        GM_ADDR destinationSlots,
        GM_ADDR missMask,
        const AsuKvGatherTilingData* tiling,
        TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        destinationKvGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ KvT*>(destinationKvCache));
        destinationRopeGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ RopeT*>(destinationKRope));
        destinationBlockTableGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(destinationBlockTable));
        sourceKvGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ KvT*>(sourceKvCache));
        sourceRopeGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ RopeT*>(sourceKRope));
        sourceBlockTableGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(sourceBlockTable));
        reqPoolEntriesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(reqPoolEntries));
        tokenPositionsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(tokenPositions));
        destinationSlotsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(destinationSlots));
        missMaskGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(missMask));
        pipe_->InitBuffer(
            kvBuffer_, tiling_->kvRecordElements * sizeof(KvT));
        pipe_->InitBuffer(
            ropeBuffer_, tiling_->ropeRecordElements * sizeof(RopeT));
        mte2ToMte3Event_ = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE2_MTE3));
        mte3ToMte2Event_ = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        mte3DoneEvent_ = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE3_V));
        InitJitterRng();
    }

    __aicore__ inline void Process()
    {
        const uint32_t core_id = GetBlockIdx();
        const uint32_t core_count = GetBlockNum();
        auto kv_local = kvBuffer_.Get<KvT>();
        auto rope_local = ropeBuffer_.Get<RopeT>();
        kv_local.SetSize(tiling_->kvRecordElements);
        rope_local.SetSize(tiling_->ropeRecordElements);

        const uint64_t pair_count =
            static_cast<uint64_t>(tiling_->reqNum) * tiling_->queryCount;
        // 连续区间分配: 与原始 asu_kv_gather 完全一致,
        // 核 core_id 处理 [pair_begin, pair_end) 连续 pair 区间
        const uint64_t pairs_per_core = pair_count / core_count;
        const uint64_t tail_core_count = pair_count % core_count;
        const uint64_t core_pair_count =
            pairs_per_core + (core_id < tail_core_count ? 1U : 0U);
        const uint64_t pair_begin =
            static_cast<uint64_t>(core_id) * pairs_per_core +
            (core_id < tail_core_count ? core_id : tail_core_count);
        const uint64_t pair_end = pair_begin + core_pair_count;
        uint32_t req_id =
            static_cast<uint32_t>(pair_begin / tiling_->queryCount);
        uint32_t query_id =
            static_cast<uint32_t>(pair_begin % tiling_->queryCount);
        int32_t pool_entry = reqPoolEntriesGm_.GetValue(req_id);
        bool valid_pool_entry =
            pool_entry >= 0 &&
            static_cast<uint32_t>(pool_entry) <
                tiling_->sourcePoolCapacity;

        uint32_t missCount = 0U;

        for (uint64_t pair_offset = pair_begin;
             pair_offset < pair_end;
             ++pair_offset) {
            uint64_t index_offset = pair_offset;
            uint64_t mask_offset = pair_offset;
            if constexpr (ResidentInit) {
                index_offset = query_id;
                mask_offset = req_id;
            }
            if (valid_pool_entry &&
                missMaskGm_.GetValue(mask_offset) != 0) {
                const int32_t token = tokenPositionsGm_.GetValue(index_offset);
                const int32_t slot = destinationSlotsGm_.GetValue(index_offset);
                if (token >= 0 && slot >= 0) {
                    const uint32_t source_logical_block =
                        static_cast<uint32_t>(token) / tiling_->blockSize;
                    const uint32_t source_block_offset =
                        static_cast<uint32_t>(token) % tiling_->blockSize;
                    const uint32_t destination_logical_block =
                        static_cast<uint32_t>(slot) / tiling_->blockSize;
                    const uint32_t destination_block_offset =
                        static_cast<uint32_t>(slot) % tiling_->blockSize;
                    if (source_logical_block < tiling_->sourceTableWidth &&
                        destination_logical_block <
                            tiling_->destinationTableWidth) {
                        const int32_t source_physical_block =
                            sourceBlockTableGm_.GetValue(
                                static_cast<uint64_t>(pool_entry) *
                                        tiling_->sourceTableWidth +
                                source_logical_block);
                        const int32_t destination_physical_block =
                            destinationBlockTableGm_.GetValue(
                                static_cast<uint64_t>(req_id) *
                                        tiling_->destinationTableWidth +
                                destination_logical_block);
                        if (source_physical_block >= 0 &&
                            destination_physical_block >= 0 &&
                            static_cast<uint32_t>(source_physical_block) <
                                tiling_->sourcePhysicalBlockCount &&
                            static_cast<uint32_t>(destination_physical_block) <
                                tiling_->destinationPhysicalBlockCount) {
                            const uint64_t source_token_offset =
                                (static_cast<uint64_t>(source_physical_block) *
                                     tiling_->blockSize +
                                 source_block_offset);
                            const uint64_t destination_token_offset =
                                (static_cast<uint64_t>(
                                     destination_physical_block) *
                                     tiling_->blockSize +
                                 destination_block_offset);

                            ++missCount;  // 本核实际发射 MTE2 的 miss 数 (每 miss 发 KV+Rope 共 2 次)

                            DataCopy(
                                kv_local,
                                sourceKvGm_[source_token_offset *
                                            tiling_->kvRecordElements],
                                tiling_->kvRecordElements);
                            DataCopy(
                                rope_local,
                                sourceRopeGm_[source_token_offset *
                                              tiling_->ropeRecordElements],
                                tiling_->ropeRecordElements);
                            Sync<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
                            DataCopy(
                                destinationKvGm_[destination_token_offset *
                                                 tiling_->kvRecordElements],
                                kv_local,
                                tiling_->kvRecordElements);
                            DataCopy(
                                destinationRopeGm_[
                                    destination_token_offset *
                                    tiling_->ropeRecordElements],
                                rope_local,
                                tiling_->ropeRecordElements);
                            Sync<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
                        }
                    }
                }
            }

            ++query_id;
            if (query_id == tiling_->queryCount &&
                pair_offset + 1U < pair_end) {
                query_id = 0U;
                ++req_id;
                pool_entry = reqPoolEntriesGm_.GetValue(req_id);
                valid_pool_entry =
                    pool_entry >= 0 &&
                    static_cast<uint32_t>(pool_entry) <
                        tiling_->sourcePoolCapacity;
            }
        }

        // 整个 AIV 只注入一次抖动 (循环外): 抽样次数 = 本核实际 miss 数 × 2
        // (每 miss 发射 KV+Rope 共 2 次 MTE2), 取 N 次抽样中的最大 delay 作为本核实际抖动时间
        RandomSleepBeforeSwapLoad(missCount * 2U);
    }

private:
    // 初始化 swap 抖动随机流: seed 混合 block_id, 各核随机流独立
    __aicore__ inline void InitJitterRng()
    {
        rngState_ =
            tiling_->jitterSeed ^ (GetBlockIdx() * 2654435761U);
    }

    // 每个 AIV 核只在 pair 循环前调用一次: 对本核循环内 MTE2 发射总个数
    // (循环次数 × 2) 各抽一次 7 级尾延迟, 取最大 delay 作为本核实际抖动时间。
    // 反映"多次 swap 读中至少一次命中尾延迟"的聚合概率:
    // P(抖动 >= 某级 delayUs) = 1 - (1 - 1/denominator)^N。
    // 编译期开关 kSwapJitterEnabled 关闭时直接返回, 零开销。
    __aicore__ inline void RandomSleepBeforeSwapLoad(uint32_t mte2Count)
    {
        if (!kSwapJitterEnabled || tiling_->jitterEnable == 0U ||
            mte2Count == 0U) {
            return;
        }
        uint32_t max_delay_us = 0U;
        for (uint32_t i = 0U; i < mte2Count; ++i) {
            // LCG 一步 (Numerical Recipes 经典参数)
            rngState_ = rngState_ * 1664525U + 1013904223U;
            const uint32_t r = rngState_;
            // CDF 尾部模型: 从最稀有(最小阈值)开始检查, 越稀有延迟越长,
            // 保证 P(delay >= level.delayUs) = 1/denominator。
            uint32_t delay_us = kSwapJitterBaseUs;
            for (uint32_t k = kSwapJitterLevelCount; k > 0U; --k) {
                if (r < kSwapJitterLevels[k - 1U].threshold) {
                    delay_us = kSwapJitterLevels[k - 1U].delayUs;
                    break;
                }
            }
            if (delay_us > max_delay_us) {
                max_delay_us = delay_us;
            }
        }
        // 抖动必须等整 MTE3 写回真正落盘成功后再执行: 用 MTE3_V 同步
        // (V 等待 MTE3 写回完成), 避免抖动被 MTE3 排水掩盖。
        Sync<HardEvent::MTE3_V>(mte3DoneEvent_);
        BusyWaitUs(max_delay_us);
    }

    // 标量忙等: 迭代次数 = us * kItersPerUs (volatile 防止优化消除)
    __aicore__ inline void BusyWaitUs(uint32_t us)
    {
        const uint32_t iters = us * kItersPerUs;
        volatile uint32_t sink = 0U;
        for (uint32_t i = 0U; i < iters; ++i) {
            sink += i;
        }
        (void)sink;
    }

    template <HardEvent event>
    __aicore__ inline void Sync(event_t event_id)
    {
        SetFlag<event>(event_id);
        WaitFlag<event>(event_id);
    }

    const AsuKvGatherTilingData* tiling_;
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> kvBuffer_;
    TBuf<TPosition::VECCALC> ropeBuffer_;
    event_t mte2ToMte3Event_;
    event_t mte3ToMte2Event_;
    event_t mte3DoneEvent_;
    uint32_t rngState_ = 0U;
    GlobalTensor<KvT> destinationKvGm_;
    GlobalTensor<RopeT> destinationRopeGm_;
    GlobalTensor<int32_t> destinationBlockTableGm_;
    GlobalTensor<KvT> sourceKvGm_;
    GlobalTensor<RopeT> sourceRopeGm_;
    GlobalTensor<int32_t> sourceBlockTableGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> tokenPositionsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> missMaskGm_;
};

}  // namespace

extern "C" __global__ __aicore__ void asu_kv_gather(
    GM_ADDR destinationKvCache,
    GM_ADDR destinationKRope,
    GM_ADDR destinationBlockTable,
    GM_ADDR sourceKvCache,
    GM_ADDR sourceKRope,
    GM_ADDR sourceBlockTable,
    GM_ADDR reqPoolEntries,
    GM_ADDR tokenPositions,
    GM_ADDR destinationSlots,
    GM_ADDR missMask,
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
    if (tilingData.residentInitLayout != 0U) {
        KernelAsuKvGather<
            DTYPE_SOURCE_KV_CACHE,
            DTYPE_SOURCE_K_ROPE,
            true> op;
        op.Init(destinationKvCache,
                destinationKRope,
                destinationBlockTable,
                sourceKvCache,
                sourceKRope,
                sourceBlockTable,
                reqPoolEntries,
                tokenPositions,
                destinationSlots,
                missMask,
                &tilingData,
                &pipe);
        op.Process();
    } else {
        KernelAsuKvGather<
            DTYPE_SOURCE_KV_CACHE,
            DTYPE_SOURCE_K_ROPE,
            false> op;
        op.Init(destinationKvCache,
                destinationKRope,
                destinationBlockTable,
                sourceKvCache,
                sourceKRope,
                sourceBlockTable,
                reqPoolEntries,
                tokenPositions,
                destinationSlots,
                missMask,
                &tilingData,
                &pipe);
        op.Process();
    }
}
