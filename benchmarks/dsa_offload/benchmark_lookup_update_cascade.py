# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import argparse
import statistics

import torch

from vllm_ascend.dsa_offload.ops import LookupState, lookup_update_batch
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

INDEX_CAPACITY = 128 * 1024
RESIDENT_SLOTS = 8 * 1024
LOOKUP_SLOTS = 10 * 1024


def make_inputs(
    batch_size: int,
    queries_per_request: int,
    decode_mode: int,
) -> tuple[LookupState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = LookupState(
        index=torch.full(
            (batch_size, INDEX_CAPACITY),
            -1,
            dtype=torch.int32,
            device="npu",
        ),
        slot_to_index=torch.full(
            (batch_size, LOOKUP_SLOTS),
            -1,
            dtype=torch.int32,
            device="npu",
        ),
        free_slots=torch.arange(
            RESIDENT_SLOTS,
            LOOKUP_SLOTS,
            dtype=torch.int32,
            device="npu",
        )
        .expand(batch_size, LOOKUP_SLOTS - RESIDENT_SLOTS)
        .clone(),
        free_head=torch.zeros(
            (batch_size, 16), dtype=torch.int32, device="npu"
        ),
    )
    resident = torch.arange(
        RESIDENT_SLOTS, dtype=torch.int32, device="npu"
    )
    state.index[:, :RESIDENT_SLOTS] = resident
    state.slot_to_index[:, :RESIDENT_SLOTS] = resident
    query_num = batch_size * queries_per_request
    request_rows = torch.arange(
        batch_size, dtype=torch.int32, device="npu"
    )
    query_start_loc = torch.arange(
        0,
        query_num + 1,
        queries_per_request,
        dtype=torch.int32,
        device="npu",
    )
    query_positions = torch.arange(
        16384,
        16384 + queries_per_request,
        dtype=torch.int64,
        device="npu",
    ).repeat(batch_size)
    generator = torch.Generator(device="npu")
    generator.manual_seed(7)
    semantic_topk = torch.randint(
        0,
        RESIDENT_SLOTS,
        (query_num, 1, 2048),
        dtype=torch.int32,
        device="npu",
        generator=generator,
    )
    if decode_mode == 0 and queries_per_request != 1:
        raise ValueError("normal Decode requires one query per request")
    return (
        state,
        request_rows,
        query_start_loc,
        query_positions,
        semantic_topk,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--queries-per-request", type=int, default=1)
    parser.add_argument("--decode-mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    inputs = make_inputs(
        args.batch_size,
        args.queries_per_request,
        args.decode_mode,
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        state, request_rows, query_start_loc, positions, topk = inputs
        return lookup_update_batch(
            state,
            request_rows,
            query_start_loc,
            positions,
            topk,
            block_size=128,
            tail_base=10240,
            fallback_slot=10368,
            staging_base=10369,
            decode_mode=args.decode_mode,
        )

    for _ in range(args.warmup):
        run()
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iterations)]
    for start, end in zip(starts, ends):
        start.record()
        run()
        end.record()
    torch.npu.synchronize()
    latencies_us = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends)
    ]
    ordered = sorted(latencies_us)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    queries = args.batch_size * args.queries_per_request
    median = statistics.median(latencies_us)
    print(
        f"requests={args.batch_size} queries={queries} "
        f"mode={args.decode_mode} min_us={ordered[0]:.3f} "
        f"median_us={median:.3f} p95_us={p95:.3f} "
        f"median_queries_per_s={queries * 1e6 / median:.1f}"
    )


if __name__ == "__main__":
    main()
