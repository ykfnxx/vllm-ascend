#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t INDEX_SIZE = 128U * 1024U;
constexpr uint32_t SLOT_COUNT = 10U * 1024U;
constexpr uint32_t FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t QUERY_COUNT = 2U * 1024U;
constexpr uint32_t INDEX_TILE_LEN = 16U * 1024U;
constexpr uint32_t FREE_HEADS_PER_CACHE_LINE = 16U;
constexpr int32_t NOT_FOUND = -1;

class KernelAsuHbmIndexLookup {
public:
    __aicore__ inline KernelAsuHbmIndexLookup() {}

    __aicore__ inline void Init(GM_ADDR index,
                                GM_ADDR slotToIndex,
                                GM_ADDR freeSlots,
                                GM_ADDR freeHead,
                                GM_ADDR reqPoolEntries,
                                GM_ADDR queryIndex,
                                GM_ADDR slotOut,
                                GM_ADDR missOut,
                                uint32_t reqNum,
                                TPipe* pipe)
    {
        pipe_ = pipe;
        reqNum_ = reqNum;

        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        slotToIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotToIndex));
        freeSlotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(freeSlots));
        freeHeadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(freeHead));
        reqPoolEntriesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(reqPoolEntries), reqNum_);
        queryIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryIndex), reqNum_ * QUERY_COUNT);
        slotOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotOut), reqNum_ * QUERY_COUNT);
        missOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missOut), reqNum_ * QUERY_COUNT);

        pipe_->InitBuffer(queryBuf_, QUERY_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(indexBuf_, INDEX_TILE_LEN * sizeof(int32_t));
        pipe_->InitBuffer(outBuf_, QUERY_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(missBuf_, QUERY_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(candidateBuf_, QUERY_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(deltaBuf_, QUERY_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(clampBuf_, QUERY_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(helperBuf_, QUERY_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(offsetBuf_, QUERY_COUNT * sizeof(uint32_t));
        pipe_->InitBuffer(maskBuf_, QUERY_COUNT * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        auto queryTile = queryBuf_.Get<int32_t>();
        auto indexTile = indexBuf_.Get<int32_t>();
        auto indexTileFloat = indexBuf_.Get<float>();
        auto outTile = outBuf_.Get<int32_t>();
        auto outTileFloat = outBuf_.Get<float>();
        auto missTile = missBuf_.Get<int32_t>();
        auto candidateTileFloat = candidateBuf_.Get<float>();
        auto deltaTile = deltaBuf_.Get<int32_t>();
        auto clampTile = clampBuf_.Get<int32_t>();
        auto helperTile = helperBuf_.Get<int32_t>();
        auto offsetTile = offsetBuf_.Get<int32_t>();
        auto offsetTileU32 = offsetBuf_.Get<uint32_t>();
        auto maskTile = maskBuf_.Get<uint8_t>();

        queryTile.SetSize(QUERY_COUNT);
        indexTile.SetSize(INDEX_TILE_LEN);
        indexTileFloat.SetSize(INDEX_TILE_LEN);
        outTile.SetSize(QUERY_COUNT);
        outTileFloat.SetSize(QUERY_COUNT);
        missTile.SetSize(QUERY_COUNT);
        candidateTileFloat.SetSize(QUERY_COUNT);
        deltaTile.SetSize(QUERY_COUNT);
        clampTile.SetSize(QUERY_COUNT);
        helperTile.SetSize(QUERY_COUNT);
        offsetTile.SetSize(QUERY_COUNT);
        offsetTileU32.SetSize(QUERY_COUNT);
        maskTile.SetSize(QUERY_COUNT);

        for (uint32_t reqId = 0; reqId < reqNum_; ++reqId) {
            uint32_t poolEntry = static_cast<uint32_t>(reqPoolEntriesGm_.GetValue(reqId));
            // Keep all free_head entries in one cache line on the same core.
            uint32_t freeHeadCacheLine = poolEntry / FREE_HEADS_PER_CACHE_LINE;
            if (freeHeadCacheLine % blockNum != coreId) {
                continue;
            }

            uint32_t indexReqBase = poolEntry * INDEX_SIZE;
            uint32_t slotReqBase = poolEntry * SLOT_COUNT;
            uint32_t freeReqBase = poolEntry * FREE_SLOT_COUNT;
            uint32_t queryReqBase = reqId * QUERY_COUNT;
            int32_t freeHead = freeHeadGm_.GetValue(poolEntry);

            DataCopy(queryTile, queryIndexGm_[queryReqBase], QUERY_COUNT);
            SyncPipelines<HardEvent::MTE2_V>();
            Duplicate(outTile, NOT_FOUND, QUERY_COUNT);
            Duplicate(missTile, static_cast<int32_t>(0), QUERY_COUNT);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t indexBase = 0; indexBase < INDEX_SIZE; indexBase += INDEX_TILE_LEN) {
                if (indexBase != 0U) {
                    SyncPipelines<HardEvent::V_MTE2>();
                }
                DataCopy(indexTile, indexGm_[indexReqBase + indexBase], INDEX_TILE_LEN);
                SyncPipelines<HardEvent::MTE2_V>();

                Adds(deltaTile, queryTile, -static_cast<int32_t>(indexBase), QUERY_COUNT);
                Relu(clampTile, deltaTile, QUERY_COUNT);
                Adds(helperTile, clampTile, -static_cast<int32_t>(INDEX_TILE_LEN - 1U), QUERY_COUNT);
                Relu(helperTile, helperTile, QUERY_COUNT);
                Muls(helperTile, helperTile, static_cast<int32_t>(-1), QUERY_COUNT);
                Add(clampTile, clampTile, helperTile, QUERY_COUNT);

                Muls(offsetTile, clampTile, static_cast<int32_t>(sizeof(int32_t)), QUERY_COUNT);
                Muls(helperTile, clampTile, static_cast<int32_t>(-1), QUERY_COUNT);
                Add(helperTile, helperTile, deltaTile, QUERY_COUNT);
                CompareScalar(maskTile, helperTile, static_cast<int32_t>(0), CMPMODE::EQ, QUERY_COUNT);

                Gather(candidateTileFloat, indexTileFloat, offsetTileU32, 0, QUERY_COUNT);
                Select(outTileFloat, maskTile, candidateTileFloat, outTileFloat,
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, QUERY_COUNT);
                PipeBarrier<PIPE_ALL>();
            }

            SyncPipelines<HardEvent::V_S>();
            for (uint32_t i = 0; i < QUERY_COUNT; ++i) {
                int32_t slot = outTile.GetValue(i);
                if (slot == NOT_FOUND) {
                    int32_t indexId = queryTile.GetValue(i);
                    slot = indexGm_.GetValue(indexReqBase + static_cast<uint32_t>(indexId));
                    if (slot == NOT_FOUND) {
                        slot = freeSlotsGm_.GetValue(freeReqBase + static_cast<uint32_t>(freeHead));
                        ++freeHead;
                        indexGm_.SetValue(indexReqBase + static_cast<uint32_t>(indexId), slot);
                        slotToIndexGm_.SetValue(slotReqBase + static_cast<uint32_t>(slot), indexId);
                        missTile.SetValue(i, static_cast<int32_t>(1));
                    }
                    outTile.SetValue(i, slot);
                }
            }

            // Scalar reads queryTile and may update outTile. Order both
            // dependencies before the next request reuses these buffers.
            SyncPipelines<HardEvent::S_MTE2>();
            SyncPipelines<HardEvent::S_MTE3>();
            DataCopy(slotOutGm_[queryReqBase], outTile, QUERY_COUNT);
            DataCopy(missOutGm_[queryReqBase], missTile, QUERY_COUNT);
            SyncPipelines<HardEvent::MTE3_V>();
            freeHeadGm_.SetValue(poolEntry, freeHead);
        }

        DataCacheCleanAndInvalid<int32_t,
                                 CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(freeHeadGm_);
    }

private:
    template <HardEvent event>
    __aicore__ inline void SyncPipelines()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(event));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    TPipe* pipe_;
    TBuf<TPosition::VECIN> queryBuf_;
    TBuf<TPosition::VECIN> indexBuf_;
    TBuf<TPosition::VECOUT> outBuf_;
    TBuf<TPosition::VECOUT> missBuf_;
    TBuf<TPosition::VECCALC> candidateBuf_;
    TBuf<TPosition::VECCALC> deltaBuf_;
    TBuf<TPosition::VECCALC> clampBuf_;
    TBuf<TPosition::VECCALC> helperBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;

    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<int32_t> slotToIndexGm_;
    GlobalTensor<int32_t> freeSlotsGm_;
    GlobalTensor<int32_t> freeHeadGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> queryIndexGm_;
    GlobalTensor<int32_t> slotOutGm_;
    GlobalTensor<int32_t> missOutGm_;
    uint32_t reqNum_;
};

}  // namespace

extern "C" __global__ __aicore__ void asu_hbm_index_lookup(GM_ADDR index,
                                                            GM_ADDR slotToIndex,
                                                            GM_ADDR freeSlots,
                                                            GM_ADDR freeHead,
                                                            GM_ADDR reqPoolEntries,
                                                            GM_ADDR queryIndex,
                                                            GM_ADDR slotOut,
                                                            GM_ADDR missOut,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelAsuHbmIndexLookup op;
    op.Init(index,
            slotToIndex,
            freeSlots,
            freeHead,
            reqPoolEntries,
            queryIndex,
            slotOut,
            missOut,
            tilingData.reqNum,
            &pipe);
    op.Process();
}
