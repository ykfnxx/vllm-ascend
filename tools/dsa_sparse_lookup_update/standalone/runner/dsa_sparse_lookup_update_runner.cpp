/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_dsa_sparse_lookup_update.h"

namespace {

constexpr int32_t kInvalidIndex = -1;
constexpr int64_t kIndexCapacity = 128 * 1024;
constexpr int64_t kResidentSlotCount = 8 * 1024;
constexpr int64_t kFreeSlotCount = 2 * 1024;
constexpr int64_t kSlotCount = 10 * 1024;
constexpr int64_t kQueryCount = 2 * 1024;
constexpr int64_t kFreeHeadStride = 16;

struct Options {
    int32_t device = 0;
    int64_t requests = 32;
    int64_t missCount = 205;
    uint32_t seed = 1234;
    bool quiet = false;
};

struct TensorStorage {
    aclTensor* tensor = nullptr;
    void* deviceAddress = nullptr;
    size_t bytes = 0;
};

void CheckAcl(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with ACL status " +
            std::to_string(status));
    }
}

int64_t ParseInt64(const char* value, const char* name)
{
    size_t consumed = 0;
    const std::string text(value);
    const int64_t parsed = std::stoll(text, &consumed, 10);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return parsed;
}

double ParseDouble(const char* value, const char* name)
{
    size_t consumed = 0;
    const std::string text(value);
    const double parsed = std::stod(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(name) + " must be a number");
    }
    return parsed;
}

int32_t ParseDevice(const char* value)
{
    std::string text(value);
    constexpr const char* prefix = "npu:";
    if (text.rfind(prefix, 0) == 0) {
        text.erase(0, 4);
    }
    const int64_t parsed = ParseInt64(text.c_str(), "device");
    if (parsed < 0 || parsed > INT32_MAX) {
        throw std::invalid_argument("device must be a non-negative int32");
    }
    return static_cast<int32_t>(parsed);
}

void Usage(const char* program)
{
    std::cout
        << "Usage: " << program << " [OPTIONS]\n"
        << "  --device N|npu:N   Device id (default: 0)\n"
        << "  --requests N       Concurrent request rows (default: 32)\n"
        << "  --miss-rate P      Miss percentage per 2048 queries (default: 10)\n"
        << "  --miss-count N     Exact misses per request; exclusive with --miss-rate\n"
        << "  --seed N           Workload seed (default: 1234)\n"
        << "  --quiet            Suppress the workload summary\n";
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    bool missRateSet = false;
    bool missCountSet = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        auto requireValue = [&]() -> const char* {
            if (++index >= argc) {
                throw std::invalid_argument(argument + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--device") {
            options.device = ParseDevice(requireValue());
        } else if (argument == "--requests") {
            options.requests = ParseInt64(requireValue(), "requests");
        } else if (argument == "--miss-rate") {
            const double rate = ParseDouble(requireValue(), "miss-rate");
            if (rate < 0.0 || rate > 100.0) {
                throw std::invalid_argument("miss-rate must be in [0, 100]");
            }
            options.missCount = static_cast<int64_t>(
                std::floor(kQueryCount * rate / 100.0 + 0.5));
            missRateSet = true;
        } else if (argument == "--miss-count") {
            options.missCount = ParseInt64(requireValue(), "miss-count");
            missCountSet = true;
        } else if (argument == "--seed") {
            const int64_t seed = ParseInt64(requireValue(), "seed");
            if (seed < 0 || seed > UINT32_MAX) {
                throw std::invalid_argument("seed must fit uint32");
            }
            options.seed = static_cast<uint32_t>(seed);
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "-h" || argument == "--help") {
            Usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (missRateSet && missCountSet) {
        throw std::invalid_argument(
            "--miss-rate and --miss-count are mutually exclusive");
    }
    if (options.requests <= 0 || options.requests > INT32_MAX) {
        throw std::invalid_argument("requests must be a positive int32");
    }
    if (options.missCount < 0 || options.missCount > kQueryCount) {
        throw std::invalid_argument("miss-count must be in [0, 2048]");
    }
    return options;
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2;
         index >= 0;
         --index) {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    return strides;
}

TensorStorage CreateTensor(
    const std::vector<int32_t>& hostData,
    const std::vector<int64_t>& shape)
{
    TensorStorage storage;
    storage.bytes = hostData.size() * sizeof(int32_t);
    CheckAcl(
        aclrtMalloc(
            &storage.deviceAddress,
            storage.bytes,
            ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc");
    CheckAcl(
        aclrtMemcpy(
            storage.deviceAddress,
            storage.bytes,
            hostData.data(),
            storage.bytes,
            ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy host to device");
    const std::vector<int64_t> strides = ContiguousStrides(shape);
    storage.tensor = aclCreateTensor(
        shape.data(),
        shape.size(),
        ACL_INT32,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        storage.deviceAddress);
    if (storage.tensor == nullptr) {
        aclrtFree(storage.deviceAddress);
        throw std::runtime_error("aclCreateTensor returned null");
    }
    return storage;
}

void DestroyTensor(TensorStorage& storage)
{
    if (storage.tensor != nullptr) {
        aclDestroyTensor(storage.tensor);
        storage.tensor = nullptr;
    }
    if (storage.deviceAddress != nullptr) {
        aclrtFree(storage.deviceAddress);
        storage.deviceAddress = nullptr;
    }
}

struct Workload {
    std::vector<int32_t> index;
    std::vector<int32_t> slotToIndex;
    std::vector<int32_t> freeSlots;
    std::vector<int32_t> freeHead;
    std::vector<int32_t> reqPoolEntries;
    std::vector<int32_t> queryIndex;
    std::vector<int32_t> lookupMask;
    std::vector<int32_t> slotOut;
    std::vector<int32_t> missOut;
};

Workload MakeWorkload(const Options& options)
{
    const size_t requests = static_cast<size_t>(options.requests);
    Workload workload{
        std::vector<int32_t>(requests * kIndexCapacity, kInvalidIndex),
        std::vector<int32_t>(requests * kSlotCount, kInvalidIndex),
        std::vector<int32_t>(requests * kFreeSlotCount, kInvalidIndex),
        std::vector<int32_t>(requests * kFreeHeadStride, 0),
        std::vector<int32_t>(requests, 0),
        std::vector<int32_t>(requests * kQueryCount, kInvalidIndex),
        std::vector<int32_t>(requests * kQueryCount, 1),
        std::vector<int32_t>(requests * kQueryCount, kInvalidIndex),
        std::vector<int32_t>(requests * kQueryCount, 0),
    };

    std::mt19937 generator(options.seed);
    std::vector<int32_t> positions(kIndexCapacity);
    std::iota(positions.begin(), positions.end(), 0);
    for (size_t request = 0; request < requests; ++request) {
        std::shuffle(positions.begin(), positions.end(), generator);
        const size_t indexBase = request * kIndexCapacity;
        const size_t slotBase = request * kSlotCount;
        const size_t freeBase = request * kFreeSlotCount;
        const size_t queryBase = request * kQueryCount;

        for (int64_t slot = 0; slot < kResidentSlotCount; ++slot) {
            const int32_t token = positions[slot];
            workload.index[indexBase + token] = static_cast<int32_t>(slot);
            workload.slotToIndex[slotBase + slot] = token;
        }
        for (int64_t offset = 0; offset < kFreeSlotCount; ++offset) {
            workload.freeSlots[freeBase + offset] =
                static_cast<int32_t>(kResidentSlotCount + offset);
        }
        workload.reqPoolEntries[request] = static_cast<int32_t>(request);

        std::vector<int32_t> hits(
            positions.begin(), positions.begin() + kResidentSlotCount);
        std::shuffle(hits.begin(), hits.end(), generator);
        const int64_t hitCount = kQueryCount - options.missCount;
        for (int64_t offset = 0; offset < hitCount; ++offset) {
            workload.queryIndex[queryBase + offset] = hits[offset];
        }
        for (int64_t offset = 0; offset < options.missCount; ++offset) {
            workload.queryIndex[queryBase + hitCount + offset] =
                positions[kResidentSlotCount + offset];
        }
        std::shuffle(
            workload.queryIndex.begin() + queryBase,
            workload.queryIndex.begin() + queryBase + kQueryCount,
            generator);
    }
    return workload;
}

std::vector<int32_t> CopyToHost(const TensorStorage& storage)
{
    std::vector<int32_t> result(storage.bytes / sizeof(int32_t));
    CheckAcl(
        aclrtMemcpy(
            result.data(),
            storage.bytes,
            storage.deviceAddress,
            storage.bytes,
            ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy device to host");
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    aclrtStream stream = nullptr;
    void* workspace = nullptr;
    std::vector<TensorStorage> tensors;
    int32_t activeDevice = -1;
    try {
        const Options options = ParseOptions(argc, argv);
        Workload workload = MakeWorkload(options);

        CheckAcl(aclInit(nullptr), "aclInit");
        CheckAcl(aclrtSetDevice(options.device), "aclrtSetDevice");
        activeDevice = options.device;
        CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");

        tensors.reserve(9);
        tensors.push_back(CreateTensor(
            workload.index, {options.requests, kIndexCapacity}));
        tensors.push_back(CreateTensor(
            workload.slotToIndex, {options.requests, kSlotCount}));
        tensors.push_back(CreateTensor(
            workload.freeSlots, {options.requests, kFreeSlotCount}));
        tensors.push_back(CreateTensor(
            workload.freeHead, {options.requests, kFreeHeadStride}));
        tensors.push_back(CreateTensor(
            workload.reqPoolEntries, {options.requests}));
        tensors.push_back(CreateTensor(
            workload.queryIndex, {options.requests, kQueryCount}));
        tensors.push_back(CreateTensor(
            workload.lookupMask, {options.requests, kQueryCount}));
        tensors.push_back(CreateTensor(
            workload.slotOut, {options.requests, kQueryCount}));
        tensors.push_back(CreateTensor(
            workload.missOut, {options.requests, kQueryCount}));

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        CheckAcl(
            aclnnDsaSparseLookupUpdateGetWorkspaceSize(
                tensors[0].tensor,
                tensors[1].tensor,
                tensors[2].tensor,
                tensors[3].tensor,
                tensors[4].tensor,
                tensors[5].tensor,
                tensors[6].tensor,
                options.requests,
                tensors[7].tensor,
                tensors[8].tensor,
                &workspaceSize,
                &executor),
            "aclnnDsaSparseLookupUpdateGetWorkspaceSize");
        if (workspaceSize > 0) {
            CheckAcl(
                aclrtMalloc(
                    &workspace,
                    workspaceSize,
                    ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc workspace");
        }
        CheckAcl(
            aclnnDsaSparseLookupUpdate(
                workspace,
                workspaceSize,
                executor,
                stream),
            "aclnnDsaSparseLookupUpdate");
        CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

        const std::vector<int32_t> slotOut = CopyToHost(tensors[7]);
        const std::vector<int32_t> missOut = CopyToHost(tensors[8]);
        const int64_t actualMisses = std::count(
            missOut.begin(), missOut.end(), 1);
        const int64_t expectedMisses = options.requests * options.missCount;
        if (actualMisses != expectedMisses) {
            throw std::runtime_error(
                "operator produced " + std::to_string(actualMisses) +
                " misses, expected " + std::to_string(expectedMisses));
        }
        if (std::any_of(
                slotOut.begin(), slotOut.end(),
                [](int32_t slot) { return slot < 0 || slot >= kSlotCount; })) {
            throw std::runtime_error("operator produced an invalid slot_out value");
        }

        if (!options.quiet) {
            std::cout
                << "DsaSparseLookupUpdate standalone workload: requests="
                << options.requests
                << ", resident_entries_per_request=" << kResidentSlotCount
                << ", queries_per_request=" << kQueryCount
                << ", misses_per_request=" << options.missCount
                << ", seed=" << options.seed << '\n';
        }

        if (workspace != nullptr) {
            aclrtFree(workspace);
            workspace = nullptr;
        }
        for (auto& tensor : tensors) {
            DestroyTensor(tensor);
        }
        aclrtDestroyStream(stream);
        stream = nullptr;
        aclrtResetDevice(options.device);
        activeDevice = -1;
        aclFinalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        for (auto& tensor : tensors) {
            DestroyTensor(tensor);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (activeDevice >= 0) {
            aclrtResetDevice(activeDevice);
        }
        aclFinalize();
        return 1;
    }
}
