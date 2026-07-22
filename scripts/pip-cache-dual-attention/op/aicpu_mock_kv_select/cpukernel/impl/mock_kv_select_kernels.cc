#include "mock_kv_select_kernels.h"

#include <cstdint>
#include <ctime>

#include "cpu_kernel_register.h"

namespace {
const char *MOCK_KV_SELECT = "MockKVSelect";
const char *MOCK_WAIT_US_ATTR = "mock_wait_us";
constexpr uint64_t DEFAULT_MOCK_WAIT_US = 25;

uint64_t NowNs()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

void BusyWaitUs(uint64_t waitUs)
{
    const uint64_t endNs = NowNs() + waitUs * 1000ULL;
    while (NowNs() < endNs) {
        asm volatile("" ::: "memory");
    }
}

uint64_t GetMockWaitUs(aicpu::CpuKernelContext &ctx)
{
    aicpu::AttrValue *attr = ctx.GetAttr(MOCK_WAIT_US_ATTR);
    if (attr == nullptr) {
        return DEFAULT_MOCK_WAIT_US;
    }
    const int64_t waitUs = attr->GetInt();
    return waitUs > 0 ? static_cast<uint64_t>(waitUs) : 0;
}
}  // namespace

namespace aicpu {
uint32_t MockKVSelectCpuKernel::Compute(CpuKernelContext &ctx)
{
    BusyWaitUs(GetMockWaitUs(ctx));
    return 0;
}

REGISTER_CPU_KERNEL(MOCK_KV_SELECT, MockKVSelectCpuKernel);
}  // namespace aicpu

extern "C" uint32_t RunCpuKernel(void *param);

extern "C" __attribute__((visibility("default"))) uint32_t MockKVSelect(void *param)
{
    return RunCpuKernel(param);
}
