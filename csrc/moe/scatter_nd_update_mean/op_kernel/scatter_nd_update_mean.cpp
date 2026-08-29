/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel_operator.h"

#if (__CCE_AICORE__ == 310)
#include "arch35/scatter_nd_update_mean_kernel.h"
#else
#include "arch32/scatter_nd_update_mean_kernel.h"
#endif

extern "C" __global__ __aicore__ void scatter_nd_update_mean(
    GM_ADDR flatKeyCache, GM_ADDR indices, GM_ADDR updates, GM_ADDR keyMean, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    REGISTER_TILING_DEFAULT(optiling::ScatterNdUpdateMeanTilingData);
    GET_TILING_DATA_WITH_STRUCT(optiling::ScatterNdUpdateMeanTilingData, tilingData, tiling);
    ScatterNdUpdateMeanKernel<DTYPE_FLAT_KEY_CACHE, DTYPE_INDICES> op;
    op.Init(flatKeyCache, indices, updates, keyMean, &tilingData);
    op.Process();
}
