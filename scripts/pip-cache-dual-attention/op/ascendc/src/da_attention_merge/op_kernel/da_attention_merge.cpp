#include "kernel_operator.h"
#include "da_attention_merge.h"

using namespace AscendC;
using namespace DaAttentionMergeNs;

extern "C" __global__ __aicore__ void da_attention_merge(
    GM_ADDR prevAttentionOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum,
    GM_ADDR curAttentionOut, GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum,
    GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    const DaAttentionMergeTilingData *__restrict tilingPtr = &tilingData;

    if (TILING_KEY_IS(1)) {
        DaAttentionMergeKernel<half> op(&pipe, tilingPtr);
        op.Init(prevAttentionOut, prevSoftmaxMax, prevSoftmaxSum, curAttentionOut, curSoftmaxMax, curSoftmaxSum,
                attentionOut);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        DaAttentionMergeKernel<bfloat16_t> op(&pipe, tilingPtr);
        op.Init(prevAttentionOut, prevSoftmaxMax, prevSoftmaxSum, curAttentionOut, curSoftmaxMax, curSoftmaxSum,
                attentionOut);
        op.Process();
#endif
    }
}
