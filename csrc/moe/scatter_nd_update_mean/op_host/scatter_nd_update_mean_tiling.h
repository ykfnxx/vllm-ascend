#pragma once

#include <cstdint>

namespace optiling {
struct ScatterNdUpdateMeanTilingData {
    uint32_t numUpdates;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t kvHeads;
    uint32_t numBlocks;
    uint32_t updateCacheInKernel;
    float invBlockSize;
};
} // namespace optiling
