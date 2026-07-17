#include "asu_hbm_index_maintain_aicpu_kernel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "cpu_tensor.h"
#include "cpu_types.h"

namespace {
const char* const OP_TYPE = "AsuHbmIndexMaintainAicpu";

constexpr uint32_t INDEX_INPUT = 0U;
constexpr uint32_t SLOT_TO_INDEX_INPUT = 1U;
constexpr uint32_t FREE_SLOTS_INPUT = 2U;
constexpr uint32_t FREE_HEAD_INPUT = 3U;
constexpr uint32_t REQ_POOL_ENTRIES_INPUT = 4U;
constexpr uint32_t LAST_QUERY_SLOTS_INPUT = 5U;
constexpr uint32_t INDEX_OUTPUT = 0U;
constexpr uint32_t SLOT_TO_INDEX_OUTPUT = 1U;
constexpr uint32_t FREE_SLOTS_OUTPUT = 2U;
constexpr uint32_t FREE_HEAD_OUTPUT = 3U;

constexpr uint32_t INDEX_SIZE = 128U * 1024U;
constexpr uint32_t SLOT_COUNT = 10U * 1024U;
constexpr uint32_t FREE_SLOT_COUNT = 2U * 1024U;
constexpr uint32_t QUERY_COUNT = 2U * 1024U;
constexpr uint32_t FREE_HEAD_STRIDE = 16U;
constexpr uint32_t PROTECTED_WORD_BITS = 64U;
constexpr uint32_t PROTECTED_WORD_COUNT =
    (SLOT_COUNT + PROTECTED_WORD_BITS - 1U) / PROTECTED_WORD_BITS;
constexpr uint32_t MAX_REQUEST_WORKERS = 32U;
constexpr int32_t NOT_FOUND = -1;
static_assert(FREE_HEAD_STRIDE * sizeof(int32_t) == 64U,
              "free_head row must occupy one 64-byte cache line");

uint32_t Hash32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

void ClearProtectedSlots(uint64_t* protected_slots)
{
    for (uint32_t i = 0; i < PROTECTED_WORD_COUNT; ++i) {
        protected_slots[i] = 0ULL;
    }
}

void MarkProtectedSlot(uint64_t* protected_slots, int32_t slot)
{
    const uint32_t slot_id = static_cast<uint32_t>(slot);
    protected_slots[slot_id / PROTECTED_WORD_BITS] |=
        1ULL << (slot_id % PROTECTED_WORD_BITS);
}

bool IsProtectedSlot(const uint64_t* protected_slots, uint32_t slot)
{
    return (protected_slots[slot / PROTECTED_WORD_BITS] &
            (1ULL << (slot % PROTECTED_WORD_BITS))) != 0ULL;
}

void CopyState(const int32_t* source, int32_t* destination, uint64_t element_count)
{
    if (source != destination) {
        std::memcpy(destination, source, element_count * sizeof(int32_t));
    }
}

using RequestTaskFunction = void (*)(void*, uint32_t);

class ParallelJob {
public:
    ParallelJob(RequestTaskFunction function,
                void* context,
                uint32_t task_count)
        : function_(function), context_(context), remaining_(task_count)
    {
    }

    void Run(uint32_t req_id)
    {
        function_(context_, req_id);
        if (remaining_.fetch_sub(1U) == 1U) {
            std::lock_guard<std::mutex> lock(done_mutex_);
            done_condition_.notify_one();
        }
    }

    void Wait()
    {
        std::unique_lock<std::mutex> lock(done_mutex_);
        done_condition_.wait(lock, [this]() { return remaining_.load() == 0U; });
    }

private:
    RequestTaskFunction function_;
    void* context_;
    std::atomic<uint32_t> remaining_;
    std::mutex done_mutex_;
    std::condition_variable done_condition_;
};

struct RequestTask {
    ParallelJob* job;
    uint32_t req_id;
};

// Reuse workers across the per-layer maintain calls in each decode step.
class RequestThreadPool {
public:
    static RequestThreadPool& Instance()
    {
        static RequestThreadPool pool;
        return pool;
    }

    void ParallelFor(uint32_t req_num,
                     RequestTaskFunction function,
                     void* context)
    {
        if (req_num == 1U) {
            function(context, 0U);
            return;
        }

        ParallelJob job(function, context, req_num - 1U);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for (uint32_t req_id = 1U; req_id < req_num; ++req_id) {
                tasks_.push_back({&job, req_id});
            }
        }
        task_condition_.notify_all();

        function(context, 0U);
        job.Wait();
    }

private:
    RequestThreadPool() : stopping_(false)
    {
        const uint32_t hardware_threads = std::max(
            1U, static_cast<uint32_t>(std::thread::hardware_concurrency()));
        const uint32_t worker_num = std::min(
            MAX_REQUEST_WORKERS,
            hardware_threads > 1U ? hardware_threads - 1U : 1U);
        workers_.reserve(worker_num);
        for (uint32_t worker_id = 0U; worker_id < worker_num; ++worker_id) {
            workers_.emplace_back(&RequestThreadPool::WorkerLoop, this);
        }
    }

    ~RequestThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        task_condition_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    RequestThreadPool(const RequestThreadPool&) = delete;
    RequestThreadPool& operator=(const RequestThreadPool&) = delete;

    void WorkerLoop()
    {
        while (true) {
            RequestTask task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                task_condition_.wait(
                    lock,
                    [this]() { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = tasks_.front();
                tasks_.pop_front();
            }
            task.job->Run(task.req_id);
        }
    }

    std::mutex queue_mutex_;
    std::condition_variable task_condition_;
    std::deque<RequestTask> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_;
};

void MaintainOneRequest(int32_t* index,
                        int32_t* slot_to_index,
                        int32_t* free_slots,
                        int32_t* free_head,
                        const int32_t* req_pool_entries,
                        const int32_t* last_query_slots,
                        uint32_t req_id,
                        uint32_t seed)
{
    const uint32_t pool_entry =
        static_cast<uint32_t>(req_pool_entries[req_id]);
    int32_t* req_index = index + pool_entry * INDEX_SIZE;
    int32_t* req_slot_to_index = slot_to_index + pool_entry * SLOT_COUNT;
    int32_t* req_free_slots = free_slots + pool_entry * FREE_SLOT_COUNT;
    const int32_t* req_last_query_slots =
        last_query_slots + req_id * QUERY_COUNT;
    int32_t head = free_head[pool_entry * FREE_HEAD_STRIDE];
    if (head == 0) {
        return;
    }

    uint64_t protected_slots[PROTECTED_WORD_COUNT];
    uint32_t slot = Hash32(seed ^ pool_entry) % SLOT_COUNT;
    ClearProtectedSlots(protected_slots);
    for (uint32_t i = 0; i < QUERY_COUNT; ++i) {
        MarkProtectedSlot(protected_slots, req_last_query_slots[i]);
    }

    while (head > 0) {
        const int32_t index_id = req_slot_to_index[slot];
        if (index_id != NOT_FOUND &&
            !IsProtectedSlot(protected_slots, slot)) {
            req_slot_to_index[slot] = NOT_FOUND;
            req_index[static_cast<uint32_t>(index_id)] = NOT_FOUND;
            --head;
            req_free_slots[static_cast<uint32_t>(head)] =
                static_cast<int32_t>(slot);
        }
        ++slot;
        if (slot == SLOT_COUNT) {
            slot = 0;
        }
    }

    free_head[pool_entry * FREE_HEAD_STRIDE] = head;
}

struct MaintainTaskContext {
    int32_t* index;
    int32_t* slot_to_index;
    int32_t* free_slots;
    int32_t* free_head;
    const int32_t* req_pool_entries;
    const int32_t* last_query_slots;
    uint32_t seed;
};

void MaintainRequestTask(void* context, uint32_t req_id)
{
    auto* task = static_cast<MaintainTaskContext*>(context);
    MaintainOneRequest(task->index,
                       task->slot_to_index,
                       task->free_slots,
                       task->free_head,
                       task->req_pool_entries,
                       task->last_query_slots,
                       req_id,
                       task->seed);
}

void MaintainEviction(int32_t* index,
                      int32_t* slot_to_index,
                      int32_t* free_slots,
                      int32_t* free_head,
                      const int32_t* req_pool_entries,
                      const int32_t* last_query_slots,
                      uint32_t req_num,
                      uint32_t seed)
{
    MaintainTaskContext context = {
        index,
        slot_to_index,
        free_slots,
        free_head,
        req_pool_entries,
        last_query_slots,
        seed,
    };
    if (req_num == 1U) {
        MaintainRequestTask(&context, 0U);
        return;
    }
    RequestThreadPool::Instance().ParallelFor(
        req_num, MaintainRequestTask, &context);
}
}  // namespace

namespace aicpu {
uint32_t AsuHbmIndexMaintainAicpuCpuKernel::Compute(CpuKernelContext& ctx)
{
    const uint32_t req_num =
        static_cast<uint32_t>(ctx.GetAttr("req_num")->GetInt());
    const uint32_t seed =
        static_cast<uint32_t>(ctx.GetAttr("seed")->GetInt());

    const auto* index_input =
        reinterpret_cast<const int32_t*>(ctx.Input(INDEX_INPUT)->GetData());
    const auto* slot_to_index_input = reinterpret_cast<const int32_t*>(
        ctx.Input(SLOT_TO_INDEX_INPUT)->GetData());
    const auto* free_slots_input = reinterpret_cast<const int32_t*>(
        ctx.Input(FREE_SLOTS_INPUT)->GetData());
    const auto* free_head_input = reinterpret_cast<const int32_t*>(
        ctx.Input(FREE_HEAD_INPUT)->GetData());
    const auto* req_pool_entries = reinterpret_cast<const int32_t*>(
        ctx.Input(REQ_POOL_ENTRIES_INPUT)->GetData());
    const auto* last_query_slots = reinterpret_cast<const int32_t*>(
        ctx.Input(LAST_QUERY_SLOTS_INPUT)->GetData());

    auto* index =
        reinterpret_cast<int32_t*>(ctx.Output(INDEX_OUTPUT)->GetData());
    auto* slot_to_index = reinterpret_cast<int32_t*>(
        ctx.Output(SLOT_TO_INDEX_OUTPUT)->GetData());
    auto* free_slots = reinterpret_cast<int32_t*>(
        ctx.Output(FREE_SLOTS_OUTPUT)->GetData());
    auto* free_head = reinterpret_cast<int32_t*>(
        ctx.Output(FREE_HEAD_OUTPUT)->GetData());

    CopyState(index_input, index, static_cast<uint64_t>(req_num) * INDEX_SIZE);
    CopyState(slot_to_index_input,
              slot_to_index,
              static_cast<uint64_t>(req_num) * SLOT_COUNT);
    CopyState(free_slots_input,
              free_slots,
              static_cast<uint64_t>(req_num) * FREE_SLOT_COUNT);
    CopyState(free_head_input,
              free_head,
              static_cast<uint64_t>(req_num) * FREE_HEAD_STRIDE);

    MaintainEviction(index,
                     slot_to_index,
                     free_slots,
                     free_head,
                     req_pool_entries,
                     last_query_slots,
                     req_num,
                     seed);
    return 0U;
}

REGISTER_CPU_KERNEL(OP_TYPE, AsuHbmIndexMaintainAicpuCpuKernel);
}  // namespace aicpu
