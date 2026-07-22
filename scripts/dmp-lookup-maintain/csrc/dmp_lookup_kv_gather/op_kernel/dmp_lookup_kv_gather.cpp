#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t TOTAL_SLOT_COUNT = 10U * 1024U;
constexpr uint32_t QUERY_COUNT = 2U * 1024U;

template <typename T>
class KernelDmpLookupKvGather {
public:
    __aicore__ inline KernelDmpLookupKvGather() {}

    __aicore__ inline void Init(
        GM_ADDR selectionKRope,
        GM_ADDR selectionKvCache,
        GM_ADDR selectionBlockTable,
        GM_ADDR residentTokenIds,
        GM_ADDR queryIndex,
        GM_ADDR slotOut,
        GM_ADDR missOut,
        GM_ADDR needsRefill,
        GM_ADDR fullKRope,
        GM_ADDR fullKvCache,
        GM_ADDR fullBlockTable,
        GM_ADDR seqLens,
        GM_ADDR copiedCount,
        const DmpLookupKvGatherTilingData* tiling,
        TPipe* pipe)
    {
        pipe_ = pipe;
        batchSize_ = tiling->batchSize;
        selectionBlockSize_ = tiling->selectionBlockSize;
        selectionBlocksPerRow_ = tiling->selectionBlocksPerRow;
        fullBlockSize_ = tiling->fullBlockSize;
        fullBlocksPerRow_ = tiling->fullBlocksPerRow;
        kvDim_ = tiling->kvDim;
        ropeDim_ = tiling->ropeDim;

        selectionKRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(selectionKRope));
        selectionKvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(selectionKvCache));
        selectionBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selectionBlockTable));
        residentTokenIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(residentTokenIds));
        queryIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryIndex));
        slotOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotOut));
        missOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missOut));
        needsRefillGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(needsRefill));
        fullKRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(fullKRope));
        fullKvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(fullKvCache));
        fullBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(fullBlockTable));
        seqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
        copiedCountGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(copiedCount));

        pipe_->InitBuffer(kvBuffer_, kvDim_ * sizeof(T));
        pipe_->InitBuffer(ropeBuffer_, ropeDim_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        for (uint32_t row = coreId; row < batchSize_; row += blockNum) {
            const int32_t seqLen = seqLensGm_.GetValue(row);
            int32_t copied = 0;
            // Hit SFA reads the full vLLM KV cache directly. This operator
            // only stages the fixed 300 misses for miss SFA.
            const uint32_t queryBase = row * QUERY_COUNT;
            for (uint32_t i = 0; i < QUERY_COUNT; ++i) {
                if (missOutGm_.GetValue(queryBase + i) == 0) {
                    continue;
                }
                const int32_t token = queryIndexGm_.GetValue(queryBase + i);
                const int32_t slot = slotOutGm_.GetValue(queryBase + i);
                if (token >= 0 && token < seqLen && slot >= 0 &&
                    slot < static_cast<int32_t>(TOTAL_SLOT_COUNT) &&
                    CopyToken(row, token, static_cast<uint32_t>(slot))) {
                    ++copied;
                }
            }
            copiedCountGm_.SetValue(row, copied);
        }
    }

private:
    __aicore__ inline bool CopyToken(uint32_t row, int32_t token, uint32_t slot)
    {
        const uint32_t sourceLogicalBlock =
            static_cast<uint32_t>(token) / fullBlockSize_;
        if (sourceLogicalBlock >= fullBlocksPerRow_) {
            return false;
        }
        const uint32_t targetLogicalBlock = slot / selectionBlockSize_;
        const int32_t sourcePhysicalBlock = fullBlockTableGm_.GetValue(
            row * fullBlocksPerRow_ + sourceLogicalBlock);
        const int32_t targetPhysicalBlock = selectionBlockTableGm_.GetValue(
            row * selectionBlocksPerRow_ + targetLogicalBlock);
        if (sourcePhysicalBlock < 0 || targetPhysicalBlock < 0) {
            return false;
        }

        const uint32_t sourceOffset = static_cast<uint32_t>(token) % fullBlockSize_;
        const uint32_t targetOffset = slot % selectionBlockSize_;
        const uint64_t sourceKvBase =
            (static_cast<uint64_t>(sourcePhysicalBlock) * fullBlockSize_ + sourceOffset) * kvDim_;
        const uint64_t sourceRopeBase =
            (static_cast<uint64_t>(sourcePhysicalBlock) * fullBlockSize_ + sourceOffset) * ropeDim_;
        const uint64_t targetKvBase =
            (static_cast<uint64_t>(targetPhysicalBlock) * selectionBlockSize_ + targetOffset) * kvDim_;
        const uint64_t targetRopeBase =
            (static_cast<uint64_t>(targetPhysicalBlock) * selectionBlockSize_ + targetOffset) * ropeDim_;

        auto kvLocal = kvBuffer_.Get<T>();
        auto ropeLocal = ropeBuffer_.Get<T>();
        kvLocal.SetSize(kvDim_);
        ropeLocal.SetSize(ropeDim_);
        DataCopy(kvLocal, fullKvCacheGm_[sourceKvBase], kvDim_);
        DataCopy(ropeLocal, fullKRopeGm_[sourceRopeBase], ropeDim_);
        SyncPipelines<HardEvent::MTE2_MTE3>();
        DataCopy(selectionKvCacheGm_[targetKvBase], kvLocal, kvDim_);
        DataCopy(selectionKRopeGm_[targetRopeBase], ropeLocal, ropeDim_);
        SyncPipelines<HardEvent::MTE3_MTE2>();
        return true;
    }

    template <HardEvent event>
    __aicore__ inline void SyncPipelines()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(event));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> kvBuffer_;
    TBuf<TPosition::VECCALC> ropeBuffer_;
    GlobalTensor<T> selectionKRopeGm_;
    GlobalTensor<T> selectionKvCacheGm_;
    GlobalTensor<int32_t> selectionBlockTableGm_;
    GlobalTensor<int32_t> residentTokenIdsGm_;
    GlobalTensor<int32_t> queryIndexGm_;
    GlobalTensor<int32_t> slotOutGm_;
    GlobalTensor<int32_t> missOutGm_;
    GlobalTensor<bool> needsRefillGm_;
    GlobalTensor<T> fullKRopeGm_;
    GlobalTensor<T> fullKvCacheGm_;
    GlobalTensor<int32_t> fullBlockTableGm_;
    GlobalTensor<int32_t> seqLensGm_;
    GlobalTensor<int32_t> copiedCountGm_;
    uint32_t batchSize_;
    uint32_t selectionBlockSize_;
    uint32_t selectionBlocksPerRow_;
    uint32_t fullBlockSize_;
    uint32_t fullBlocksPerRow_;
    uint32_t kvDim_;
    uint32_t ropeDim_;
};
}  // namespace

extern "C" __global__ __aicore__ void dmp_lookup_kv_gather(
    GM_ADDR selectionKRope,
    GM_ADDR selectionKvCache,
    GM_ADDR selectionBlockTable,
    GM_ADDR residentTokenIds,
    GM_ADDR queryIndex,
    GM_ADDR slotOut,
    GM_ADDR missOut,
    GM_ADDR needsRefill,
    GM_ADDR fullKRope,
    GM_ADDR fullKvCache,
    GM_ADDR fullBlockTable,
    GM_ADDR seqLens,
    GM_ADDR selectionKRopeOut,
    GM_ADDR selectionKvCacheOut,
    GM_ADDR copiedCount,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)selectionKRopeOut;
    (void)selectionKvCacheOut;
    (void)workspace;
    if (g_coreType == AIC) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelDmpLookupKvGather<DTYPE_SELECTION_K_ROPE> op;
    op.Init(selectionKRope,
            selectionKvCache,
            selectionBlockTable,
            residentTokenIds,
            queryIndex,
            slotOut,
            missOut,
            needsRefill,
            fullKRope,
            fullKvCache,
            fullBlockTable,
            seqLens,
            copiedCount,
            &tilingData,
            &pipe);
    op.Process();
}
