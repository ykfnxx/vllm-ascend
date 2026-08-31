# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

INDEX_CAPACITY = 128 * 1024
RESIDENT_SLOTS = 8 * 1024
LOOKUP_SLOTS = 10 * 1024
QUERY_WIDTH = 2 * 1024
FREE_HEAD_STRIDE = 16
BLOCK_SIZE = 128
HOT_BLOCKS_PER_ROW = 82


def measure_us(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--queries-per-request", type=int, default=4)
    parser.add_argument("--gather-misses-per-query", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--turbo", action="store_true")
    args = parser.parse_args()
    if args.turbo and not args.mtp:
        raise ValueError("--turbo requires --mtp")
    if not 0 <= args.gather_misses_per_query <= QUERY_WIDTH:
        raise ValueError("gather misses must be in [0, 2048]")

    request_count = args.requests
    query_count = request_count * args.queries_per_request
    request_rows = torch.arange(
        request_count, dtype=torch.int32, device="npu"
    )
    query_start_loc = (
        torch.arange(request_count + 1, dtype=torch.int32, device="npu")
        * args.queries_per_request
    )
    query_positions = (
        16512
        + torch.arange(
            args.queries_per_request, dtype=torch.int32, device="npu"
        ).repeat(request_count)
    )
    semantic_topk = torch.full(
        (query_count, 1, QUERY_WIDTH),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    semantic_topk[:, 0, :32] = torch.arange(
        32, dtype=torch.int32, device="npu"
    )

    index = torch.full(
        (request_count, INDEX_CAPACITY),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    slot_to_index = torch.full(
        (request_count, LOOKUP_SLOTS),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    resident = torch.arange(RESIDENT_SLOTS, dtype=torch.int32, device="npu")
    index[:, :RESIDENT_SLOTS] = resident
    slot_to_index[:, :RESIDENT_SLOTS] = resident
    free_slots = torch.arange(
        RESIDENT_SLOTS, LOOKUP_SLOTS, dtype=torch.int32, device="npu"
    ).expand(request_count, -1).clone()
    free_head = torch.zeros(
        (request_count, FREE_HEAD_STRIDE), dtype=torch.int32, device="npu"
    )
    mapped_out = torch.empty_like(semantic_topk)
    gather_out = torch.empty_like(semantic_topk)
    lookup_name = (
        "dsa_sparse_turbo_resolve_update_batch_v2"
        if args.turbo
        else "dsa_offload_resolve_update_batch_v2"
    )

    def lookup() -> tuple[torch.Tensor, torch.Tensor]:
        return getattr(torch.ops._C_ascend, lookup_name)(
            index,
            slot_to_index,
            free_slots,
            free_head,
            request_rows,
            query_start_loc,
            query_positions,
            semantic_topk,
            mapped_out,
            gather_out,
            request_count,
            BLOCK_SIZE,
            int(args.mtp),
        )

    destination_blocks = request_count * HOT_BLOCKS_PER_ROW
    destination_kv = torch.zeros(
        (destination_blocks, BLOCK_SIZE, 512),
        dtype=torch.int8,
        device="npu",
    )
    destination_rope = torch.zeros(
        (destination_blocks, BLOCK_SIZE, 64),
        dtype=torch.bfloat16,
        device="npu",
    )
    source_kv = torch.zeros(
        (1, BLOCK_SIZE, 512), dtype=torch.int8, device="npu"
    )
    source_rope = torch.zeros(
        (1, BLOCK_SIZE, 64), dtype=torch.bfloat16, device="npu"
    )
    hot_table = torch.arange(
        destination_blocks, dtype=torch.int32, device="npu"
    ).view(request_count, HOT_BLOCKS_PER_ROW)
    source_table = torch.zeros(
        (request_count, 1024), dtype=torch.int32, device="npu"
    )
    gather_topk = torch.zeros_like(semantic_topk)
    gather_mapped = torch.full_like(semantic_topk, -1)
    gather_mask = torch.zeros_like(semantic_topk)
    miss_count = args.gather_misses_per_query
    if miss_count:
        gather_mapped[:, 0, :miss_count] = 8192 + torch.arange(
            miss_count, dtype=torch.int32, device="npu"
        )
        gather_mask[:, 0, :miss_count] = 1

    def gather(
        topk: torch.Tensor,
        mapped: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.asu_kv_gather_direct_v2(
            destination_kv,
            destination_rope,
            hot_table,
            source_kv,
            source_rope,
            source_table,
            request_rows,
            query_start_loc,
            topk,
            mapped,
            mask,
            BLOCK_SIZE,
            request_count,
        )

    def active_gather() -> tuple[torch.Tensor, torch.Tensor]:
        return gather(gather_topk, gather_mapped, gather_mask)

    def cascade() -> tuple[torch.Tensor, torch.Tensor]:
        mapped, mask = lookup()
        return gather(semantic_topk, mapped, mask)

    result = {
        "requests": request_count,
        "queries": query_count,
        "decode_mode": "mtp" if args.mtp else "normal",
        "lookup_operator": lookup_name,
        "gather_misses_per_query": miss_count,
        "lookup_us": measure_us(lookup, args.warmup, args.iterations),
        "active_gather_us": measure_us(
            active_gather, args.warmup, args.iterations
        ),
        "lookup_to_gather_cascade_us": measure_us(
            cascade, args.warmup, args.iterations
        ),
    }
    if args.profile_dir is not None:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=False,
            profile_memory=False,
            with_modules=False,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(args.profile_dir),
                worker_name="dsa_offload_abi_v2",
            ),
        ) as profiler:
            for _ in range(args.profile_iterations):
                cascade()
                profiler.step()
        result["profile_dir"] = str(args.profile_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
