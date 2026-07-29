#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import (
    FREE_HEAD_STRIDE,
    FREE_SLOT_COUNT,
    INDEX_CAPACITY,
    INVALID_INDEX,
    QUERY_COUNT,
    REPO_ROOT,
    RESIDENT_SLOT_COUNT,
    SLOT_COUNT,
    OperatorInputs,
    invoke,
    load_runtime,
    validate_requests,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.ut.ops.dsa_sparse_lookup_update_reference import (  # noqa: E402
    DSASparseLookupUpdateState,
    dsa_sparse_lookup_update_reference,
)


@dataclass
class CorrectnessCase:
    name: str
    state: DSASparseLookupUpdateState
    req_pool_entries: list[int]
    query_index: list[list[int]]
    lookup_mask: list[list[int]]


def _make_resident_state(
    requests: int,
) -> DSASparseLookupUpdateState:
    index = [
        [INVALID_INDEX] * INDEX_CAPACITY
        for _ in range(requests)
    ]
    slot_to_index = [
        [INVALID_INDEX] * SLOT_COUNT
        for _ in range(requests)
    ]
    for row in range(requests):
        index[row][:RESIDENT_SLOT_COUNT] = range(
            RESIDENT_SLOT_COUNT
        )
        slot_to_index[row][:RESIDENT_SLOT_COUNT] = range(
            RESIDENT_SLOT_COUNT
        )
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


def _handcrafted_case(requests: int) -> CorrectnessCase:
    query = []
    mask = []
    for row in range(requests):
        values = [row, 9000 + row, 9000 + row, -1]
        values.extend(
            [INVALID_INDEX] * (QUERY_COUNT - len(values))
        )
        active = [1, 1, 1, 1]
        active.extend([0] * (QUERY_COUNT - len(active)))
        query.append(values)
        mask.append(active)
    return CorrectnessCase(
        name="hit_duplicate_miss_masked_padding_reordered_rows",
        state=_make_resident_state(requests),
        req_pool_entries=list(reversed(range(requests))),
        query_index=query,
        lookup_mask=mask,
    )


def _random_case(
    *,
    seed: int,
    requests: int,
) -> CorrectnessCase:
    rng = random.Random(seed)
    state = _make_resident_state(requests)
    query: list[list[int]] = []
    mask: list[list[int]] = []
    for _ in range(requests):
        miss_tokens = [
            RESIDENT_SLOT_COUNT + rank
            for rank in range(QUERY_COUNT)
        ]
        values = []
        active = []
        for entry in range(QUERY_COUNT):
            selector = rng.random()
            if selector < 0.45:
                token = rng.randrange(RESIDENT_SLOT_COUNT)
            elif selector < 0.8:
                token = miss_tokens[
                    rng.randrange(0, entry + 1)
                ]
            elif selector < 0.9:
                token = INVALID_INDEX
            else:
                token = INDEX_CAPACITY + entry
            values.append(token)
            active.append(0 if rng.random() < 0.1 else 1)
        query.append(values)
        mask.append(active)
    req_pool_entries = list(range(requests))
    rng.shuffle(req_pool_entries)
    return CorrectnessCase(
        name=f"random_seed_{seed}",
        state=state,
        req_pool_entries=req_pool_entries,
        query_index=query,
        lookup_mask=mask,
    )


def _tensor(
    runtime: Any,
    values: Any,
) -> Any:
    return runtime.torch.tensor(
        values,
        dtype=runtime.torch.int32,
        device=runtime.device,
    ).contiguous()


def _make_inputs(
    runtime: Any,
    case: CorrectnessCase,
) -> OperatorInputs:
    return OperatorInputs(
        index=_tensor(runtime, case.state.index),
        slot_to_index=_tensor(
            runtime, case.state.slot_to_index
        ),
        free_slots=_tensor(runtime, case.state.free_slots),
        free_head=_tensor(runtime, case.state.free_head),
        req_pool_entries=_tensor(
            runtime, case.req_pool_entries
        ),
        query_index=_tensor(runtime, case.query_index),
        lookup_mask=_tensor(runtime, case.lookup_mask),
    )


def _assert_equal(
    runtime: Any,
    *,
    case_name: str,
    tensor_name: str,
    actual: Any,
    expected: Any,
) -> None:
    actual_cpu = actual.detach().cpu()
    expected_cpu = runtime.torch.tensor(
        expected, dtype=runtime.torch.int32
    )
    if runtime.torch.equal(actual_cpu, expected_cpu):
        return
    mismatch = (actual_cpu != expected_cpu).nonzero()
    details = []
    for location in mismatch[:10].tolist():
        index = tuple(location)
        details.append(
            {
                "index": location,
                "actual": actual_cpu[index].item(),
                "expected": expected_cpu[index].item(),
            }
        )
    raise AssertionError(
        f"{case_name}: {tensor_name} mismatch at {details}; "
        f"total mismatches={mismatch.shape[0]}."
    )


def _run_case(runtime: Any, case: CorrectnessCase) -> None:
    expected_state = copy.deepcopy(case.state)
    expected_slots, expected_misses = (
        dsa_sparse_lookup_update_reference(
            expected_state,
            req_pool_entries=case.req_pool_entries,
            query_index=case.query_index,
            lookup_mask=case.lookup_mask,
        )
    )
    inputs = _make_inputs(runtime, case)
    actual_slots, actual_misses = invoke(runtime, inputs)
    runtime.torch.npu.synchronize()

    comparisons = (
        ("index", inputs.index, expected_state.index),
        (
            "slot_to_index",
            inputs.slot_to_index,
            expected_state.slot_to_index,
        ),
        (
            "free_slots",
            inputs.free_slots,
            expected_state.free_slots,
        ),
        (
            "free_head",
            inputs.free_head,
            expected_state.free_head,
        ),
        ("slot_out", actual_slots, expected_slots),
        ("miss_out", actual_misses, expected_misses),
    )
    for name, actual, expected in comparisons:
        _assert_equal(
            runtime,
            case_name=case.name,
            tensor_name=name,
            actual=actual,
            expected=expected,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the fused Ascend 950 "
            "dsa_sparse_lookup_update operator with its CPU oracle."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--random-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--requests", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    validate_requests(args.requests)
    if args.random_cases < 0:
        raise ValueError("random-cases must be non-negative")
    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
    )
    cases = [_handcrafted_case(args.requests)]
    cases.extend(
        _random_case(
            seed=args.seed + index,
            requests=args.requests,
        )
        for index in range(args.random_cases)
    )
    with runtime.torch.inference_mode():
        for index, case in enumerate(cases, start=1):
            _run_case(runtime, case)
            print(f"PASS {index}/{len(cases)}: {case.name}")
    print(
        "PASS: fused lookup/update matched the CPU oracle for "
        f"{len(cases)} case(s) on {args.device}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
