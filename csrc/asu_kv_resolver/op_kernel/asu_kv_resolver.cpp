#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr int32_t ASU_ONLY = 0;
constexpr int32_t HBM_RESIDENT = 1;

template <typename T>
class AsuKvResolverKernel {
public:
    __aicore__ inline AsuKvResolverKernel() {}

    __aicore__ inline void Init(GM_ADDR originalTopkIndices,
                                GM_ADDR tokenState,
                                GM_ADDR asuRecordAddr,
                                GM_ADDR hbmSlotOfToken,
                                GM_ADDR slotOwnerToken,
                                GM_ADDR freeSlotStack,
                                GM_ADDR freeSlotCount,
                                GM_ADDR originalBlockTable,
                                GM_ADDR originalKvCache0,
                                GM_ADDR originalKvCache1,
                                GM_ADDR managedKvCache0,
                                GM_ADDR managedKvCache1,
                                GM_ADDR resolvedKvSlots,
                                const AsuKvResolverTilingData* tilingData)
    {
        topkNumel_ = tilingData->topkNumel;
        blockSize_ = tilingData->blockSize;
        kv0SlotElements_ = tilingData->kv0SlotElements;
        kv1SlotElements_ = tilingData->kv1SlotElements;

        originalTopkIndicesGm_.SetGlobalBuffer(
            (__gm__ int32_t*)originalTopkIndices, topkNumel_);
        tokenStateGm_.SetGlobalBuffer((__gm__ int32_t*)tokenState);
        asuRecordAddrGm_.SetGlobalBuffer((__gm__ int32_t*)asuRecordAddr);
        hbmSlotOfTokenGm_.SetGlobalBuffer((__gm__ int32_t*)hbmSlotOfToken);
        slotOwnerTokenGm_.SetGlobalBuffer((__gm__ int32_t*)slotOwnerToken);
        freeSlotStackGm_.SetGlobalBuffer((__gm__ int32_t*)freeSlotStack);
        freeSlotCountGm_.SetGlobalBuffer((__gm__ int32_t*)freeSlotCount);
        originalBlockTableGm_.SetGlobalBuffer(
            (__gm__ int32_t*)originalBlockTable);
        originalKvCache0Gm_.SetGlobalBuffer((__gm__ T*)originalKvCache0);
        originalKvCache1Gm_.SetGlobalBuffer((__gm__ T*)originalKvCache1);
        managedKvCache0Gm_.SetGlobalBuffer((__gm__ T*)managedKvCache0);
        managedKvCache1Gm_.SetGlobalBuffer((__gm__ T*)managedKvCache1);
        resolvedKvSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t*)resolvedKvSlots, topkNumel_);
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < topkNumel_; ++i) {
            int32_t tokenId = originalTopkIndicesGm_.GetValue(i);
            int32_t state = tokenStateGm_.GetValue(tokenId);
            int32_t slot = hbmSlotOfTokenGm_.GetValue(tokenId);

            if (state != HBM_RESIDENT) {
                int32_t sourceSlot =
                    state == ASU_ONLY ? asuRecordAddrGm_.GetValue(tokenId)
                                      : ResolveTailSourceSlot(tokenId);
                slot = PopFreeSlot();
                CopyFullKvPair(sourceSlot, slot);
                tokenStateGm_.SetValue(tokenId, HBM_RESIDENT);
                hbmSlotOfTokenGm_.SetValue(tokenId, slot);
                slotOwnerTokenGm_.SetValue(slot, tokenId);
            }

            resolvedKvSlotsGm_.SetValue(i, slot);
        }
    }

private:
    __aicore__ inline int32_t ResolveTailSourceSlot(int32_t tokenId)
    {
        int64_t logicalBlock = tokenId / blockSize_;
        int64_t offset = tokenId - logicalBlock * blockSize_;
        int32_t physicalBlock = originalBlockTableGm_.GetValue(logicalBlock);
        return static_cast<int32_t>(physicalBlock * blockSize_ + offset);
    }

    __aicore__ inline int32_t PopFreeSlot()
    {
        int32_t nextCount = freeSlotCountGm_.GetValue(0) - 1;
        freeSlotCountGm_.SetValue(0, nextCount);
        return freeSlotStackGm_.GetValue(nextCount);
    }

    __aicore__ inline void CopyFullKvPair(int32_t sourceSlot,
                                          int32_t managedSlot)
    {
        CopySlot(originalKvCache0Gm_, managedKvCache0Gm_, sourceSlot,
                 managedSlot, kv0SlotElements_);
        CopySlot(originalKvCache1Gm_, managedKvCache1Gm_, sourceSlot,
                 managedSlot, kv1SlotElements_);
    }

    __aicore__ inline void CopySlot(GlobalTensor<T>& src,
                                    GlobalTensor<T>& dst,
                                    int32_t sourceSlot,
                                    int32_t managedSlot,
                                    int64_t slotElements)
    {
        int64_t srcOffset = static_cast<int64_t>(sourceSlot) * slotElements;
        int64_t dstOffset = static_cast<int64_t>(managedSlot) * slotElements;
        for (int64_t j = 0; j < slotElements; ++j) {
            dst.SetValue(dstOffset + j, src.GetValue(srcOffset + j));
        }
    }

private:
    int64_t topkNumel_;
    int64_t blockSize_;
    int64_t kv0SlotElements_;
    int64_t kv1SlotElements_;

    GlobalTensor<int32_t> originalTopkIndicesGm_;
    GlobalTensor<int32_t> tokenStateGm_;
    GlobalTensor<int32_t> asuRecordAddrGm_;
    GlobalTensor<int32_t> hbmSlotOfTokenGm_;
    GlobalTensor<int32_t> slotOwnerTokenGm_;
    GlobalTensor<int32_t> freeSlotStackGm_;
    GlobalTensor<int32_t> freeSlotCountGm_;
    GlobalTensor<int32_t> originalBlockTableGm_;
    GlobalTensor<T> originalKvCache0Gm_;
    GlobalTensor<T> originalKvCache1Gm_;
    GlobalTensor<T> managedKvCache0Gm_;
    GlobalTensor<T> managedKvCache1Gm_;
    GlobalTensor<int32_t> resolvedKvSlotsGm_;
};
} // namespace

extern "C" __global__ __aicore__ void asu_kv_resolver(
    GM_ADDR originalTopkIndices,
    GM_ADDR tokenState,
    GM_ADDR asuRecordAddr,
    GM_ADDR hbmSlotOfToken,
    GM_ADDR slotOwnerToken,
    GM_ADDR freeSlotStack,
    GM_ADDR freeSlotCount,
    GM_ADDR originalBlockTable,
    GM_ADDR originalKvCache0,
    GM_ADDR originalKvCache1,
    GM_ADDR managedKvCache0,
    GM_ADDR managedKvCache1,
    GM_ADDR resolvedKvSlots,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        if constexpr (DTYPE_ORIGINAL_KV_CACHE_0 == DT_FLOAT16) {
            AsuKvResolverKernel<half> kernel;
            kernel.Init(originalTopkIndices, tokenState, asuRecordAddr,
                        hbmSlotOfToken, slotOwnerToken, freeSlotStack,
                        freeSlotCount, originalBlockTable, originalKvCache0,
                        originalKvCache1, managedKvCache0, managedKvCache1,
                        resolvedKvSlots, &tilingData);
            kernel.Process();
        } else if constexpr (DTYPE_ORIGINAL_KV_CACHE_0 == DT_BF16) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
            AsuKvResolverKernel<bfloat16_t> kernel;
            kernel.Init(originalTopkIndices, tokenState, asuRecordAddr,
                        hbmSlotOfToken, slotOwnerToken, freeSlotStack,
                        freeSlotCount, originalBlockTable, originalKvCache0,
                        originalKvCache1, managedKvCache0, managedKvCache1,
                        resolvedKvSlots, &tilingData);
            kernel.Process();
#endif
        } else {
            AsuKvResolverKernel<float> kernel;
            kernel.Init(originalTopkIndices, tokenState, asuRecordAddr,
                        hbmSlotOfToken, slotOwnerToken, freeSlotStack,
                        freeSlotCount, originalBlockTable, originalKvCache0,
                        originalKvCache1, managedKvCache0, managedKvCache1,
                        resolvedKvSlots, &tilingData);
            kernel.Process();
        }
    }
}
