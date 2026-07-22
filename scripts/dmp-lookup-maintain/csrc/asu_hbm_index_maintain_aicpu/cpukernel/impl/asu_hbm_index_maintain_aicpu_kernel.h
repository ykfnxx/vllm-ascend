#ifndef ASU_HBM_INDEX_MAINTAIN_AICPU_KERNEL_H
#define ASU_HBM_INDEX_MAINTAIN_AICPU_KERNEL_H

#include "cpu_kernel.h"

namespace aicpu {
class AsuHbmIndexMaintainAicpuCpuKernel : public CpuKernel {
public:
    ~AsuHbmIndexMaintainAicpuCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;
};
}  // namespace aicpu

#endif  // ASU_HBM_INDEX_MAINTAIN_AICPU_KERNEL_H
