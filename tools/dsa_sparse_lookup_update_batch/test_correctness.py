#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

from common import (
    FREE_HEAD_STRIDE,
    FREE_SLOT_COUNT,
    INDEX_CAPACITY,
    INVALID_INDEX,
    QUERY_WIDTH,
    REPO_ROOT,
    RESIDENT_SLOT_COUNT,
    SLOT_COUNT,
    BatchOperatorInputs,
    invoke,
    load_runtime,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.ut.ops.dsa_sparse_lookup_update_batch_reference import (  # noqa: E402
    dsa_sparse_lookup_update_batch_reference,
)
from tests.ut.ops.dsa_sparse_lookup_update_reference import (  # noqa: E402
    DSASparseLookupUpdateState,
)


def _state(requests: int) -> DSASparseLookupUpdateState:
    index = [[INVALID_INDEX] * INDEX_CAPACITY for _ in range(requests)]
    slot_to_index = [[INVALID_INDEX] * SLOT_COUNT for _ in range(requests)]
    for row in range(requests):
        for token in range(RESIDENT_SLOT_COUNT):
            index[row][token] = token
            slot_to_index[row][token] = token
    return DSASparseLookupUpdateState(
        index=index,
        slot_to_index=slot_to_index,
        free_slots=[
            list(range(RESIDENT_SLOT_COUNT, SLOT_COUNT))
            for _ in range(requests)
        ],
        free_head=[
            [0] * FREE_HEAD_STRIDE for _ in range(requests)
        ],
    )


def _queries(
    requests: int,
    queries_per_request: int,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    query_start_loc = [
        request * queries_per_request
        for request in range(requests + 1)
    ]
    query_index = []
    lookup_mask = []
    for request in range(requests):
        resident_hit = request % RESIDENT_SLOT_COUNT
        initial_previous_hit = (
            resident_hit + 1
        ) % RESIDENT_SLOT_COUNT
        for query in range(queries_per_request):
            miss_token = 9000 + request * 64 + query
            previous_miss = (
                miss_token - 1 if query > 0 else initial_previous_hit
            )
            active = [resident_hit, miss_token, previous_miss, -1]
            valid_active = [token for token in active if token >= 0]
            if len(valid_active) != len(set(valid_active)):
                raise AssertionError(
                    "correctness workload contains duplicate active tokens: "
                    f"request={request}, query={query}, tokens={active}"
                )
            query_index.append(
                active + [INVALID_INDEX] * (QUERY_WIDTH - len(active))
            )
            lookup_mask.append(
                [1, 1, 1, 1] + [0] * (QUERY_WIDTH - len(active))
            )
    return query_start_loc, query_index, lookup_mask


def _tensor(runtime: Any, values: Any) -> Any:
    return runtime.torch.tensor(
        values,
        dtype=runtime.torch.int32,
        device=runtime.device,
    ).contiguous()


def _assert_equal(runtime: Any, name: str, actual: Any, expected: Any) -> None:
    expected_tensor = runtime.torch.tensor(expected, dtype=runtime.torch.int32)
    actual_tensor = actual.detach().cpu()
    if not runtime.torch.equal(actual_tensor, expected_tensor):
        mismatch = actual_tensor.ne(expected_tensor).nonzero()[:10].tolist()
        raise AssertionError(f"{name} mismatch at {mismatch}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--queries-per-request", type=int, default=4)
    args = parser.parse_args()
    if args.requests <= 0 or args.queries_per_request <= 0:
        raise ValueError("requests and queries-per-request must be positive")

    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
    )
    state = _state(args.requests)
    expected_state = copy.deepcopy(state)
    query_start_loc, query_index, lookup_mask = _queries(
        args.requests,
        args.queries_per_request,
    )
    req_pool_entries = list(reversed(range(args.requests)))
    expected_slot_out, expected_miss_out = (
        dsa_sparse_lookup_update_batch_reference(
            expected_state,
            req_pool_entries=req_pool_entries,
            query_start_loc=query_start_loc,
            query_index=query_index,
            lookup_mask=lookup_mask,
            fallback_slot=SLOT_COUNT,
        )
    )
    inputs = BatchOperatorInputs(
        index=_tensor(runtime, state.index),
        slot_to_index=_tensor(runtime, state.slot_to_index),
        free_slots=_tensor(runtime, state.free_slots),
        free_head=_tensor(runtime, state.free_head),
        req_pool_entries=_tensor(runtime, req_pool_entries),
        query_start_loc=_tensor(runtime, query_start_loc),
        query_index=_tensor(runtime, query_index),
        lookup_mask=_tensor(runtime, lookup_mask),
    )
    slot_out, miss_out = invoke(runtime, inputs)
    runtime.torch.npu.synchronize()

    for name, actual, expected in (
        ("slot_out", slot_out, expected_slot_out),
        ("miss_out", miss_out, expected_miss_out),
        ("index", inputs.index, expected_state.index),
        ("slot_to_index", inputs.slot_to_index, expected_state.slot_to_index),
        ("free_slots", inputs.free_slots, expected_state.free_slots),
        ("free_head", inputs.free_head, expected_state.free_head),
    ):
        _assert_equal(runtime, name, actual, expected)
    print(
        "PASS: dsa_sparse_lookup_update_batch matches the CPU oracle "
        f"for B={args.requests}, q={args.queries_per_request}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
