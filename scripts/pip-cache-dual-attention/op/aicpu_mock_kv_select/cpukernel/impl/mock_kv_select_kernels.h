#ifndef MOCK_KV_SELECT_KERNELS_H_
#define MOCK_KV_SELECT_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class MockKVSelectCpuKernel : public CpuKernel {
public:
    ~MockKVSelectCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu

#endif  // MOCK_KV_SELECT_KERNELS_H_
