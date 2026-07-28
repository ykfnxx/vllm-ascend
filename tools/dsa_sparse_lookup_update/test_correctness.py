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
    INVALID_INDEX,
    REPO_ROOT,
    OperatorInputs,
    invoke,
    load_runtime,
    validate_dimensions,
    workspace_stride,
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
    query_positions: list[int]
    query_to_req_idx: list[int]
    query_to_lane: list[int]
    query_valid_mask: list[bool]
    valid_topk_counts: list[int]
    seq_lens: list[int]
    topk_positions: list[list[int]]


def _make_empty_state(
    *,
    requests: int,
    max_model_len: int,
    slots: int,
) -> DSASparseLookupUpdateState:
    return DSASparseLookupUpdateState(
        token_to_hot=[
            [INVALID_INDEX] * max_model_len for _ in range(requests)
        ],
        hot_to_token=[[INVALID_INDEX] * slots for _ in range(requests)],
        lru_slots=[list(range(slots)) for _ in range(requests)],
    )


def _install(
    state: DSASparseLookupUpdateState,
    *,
    request_index: int,
    slot: int,
    token: int,
) -> None:
    state.token_to_hot[request_index][token] = slot
    state.hot_to_token[request_index][slot] = token


def _handcrafted_cases() -> list[CorrectnessCase]:
    mixed_state = _make_empty_state(
        requests=2,
        max_model_len=16,
        slots=8,
    )
    _install(mixed_state, request_index=0, slot=0, token=2)
    _install(mixed_state, request_index=0, slot=3, token=9)
    _install(mixed_state, request_index=1, slot=1, token=4)
    mixed_state.lru_slots[0] = [1, 4, 0, 2, 5, 6, 7, 3]
    mixed_state.lru_slots[1] = [7, 6, 5, 4, 3, 2, 0, 1]

    mixed = CorrectnessCase(
        name="mixed_hit_duplicate_reserved_inactive_reordered",
        state=mixed_state,
        query_positions=[15, 12, 14, 13],
        query_to_req_idx=[0, 1, 0, 1],
        query_to_lane=[1, 0, 0, 1],
        query_valid_mask=[True, False, True, False],
        valid_topk_counts=[4, 4, 4, 2],
        seq_lens=[16, 14],
        topk_positions=[
            [9, 5, 15, 5],
            [4, 3, 2, 1],
            [2, 5, 14, INVALID_INDEX],
            [0, 1, 2, 3],
        ],
    )

    resident_state = _make_empty_state(
        requests=1,
        max_model_len=16,
        slots=4,
    )
    _install(resident_state, request_index=0, slot=0, token=2)
    _install(resident_state, request_index=0, slot=2, token=7)
    resident_state.lru_slots[0] = [2, 1, 3, 0]
    resident = CorrectnessCase(
        name="direct_request_index_resident_reuse",
        state=resident_state,
        query_positions=[15],
        query_to_req_idx=[0],
        query_to_lane=[0],
        query_valid_mask=[True],
        valid_topk_counts=[4],
        seq_lens=[16],
        topk_positions=[[2, 7, 15, 5]],
    )
    return [mixed, resident]


def _random_state(
    rng: random.Random,
    *,
    requests: int,
    max_model_len: int,
    slots: int,
) -> DSASparseLookupUpdateState:
    state = _make_empty_state(
        requests=requests,
        max_model_len=max_model_len,
        slots=slots,
    )
    for request_index in range(requests):
        occupied = rng.randrange(0, slots + 1)
        resident_slots = rng.sample(range(slots), occupied)
        resident_tokens = rng.sample(range(max_model_len), occupied)
        for slot, token in zip(resident_slots, resident_tokens):
            _install(
                state,
                request_index=request_index,
                slot=slot,
                token=token,
            )
        state.lru_slots[request_index] = rng.sample(range(slots), slots)
    return state


def _random_case(
    *,
    seed: int,
    requests: int,
    max_model_len: int,
    slots: int,
    lanes: int,
    topk: int,
) -> CorrectnessCase:
    rng = random.Random(seed)
    state = _random_state(
        rng,
        requests=requests,
        max_model_len=max_model_len,
        slots=slots,
    )

    request_lane_pairs = [
        (request_index, lane)
        for request_index in range(requests)
        for lane in range(lanes)
    ]
    rng.shuffle(request_lane_pairs)
    query_to_req_idx = [
        request_index for request_index, _ in request_lane_pairs
    ]
    query_to_lane = [lane for _, lane in request_lane_pairs]
    query_positions = [
        max_model_len - lanes + lane
        for _, lane in request_lane_pairs
    ]
    query_valid_mask = [
        rng.random() >= 0.15 for _ in request_lane_pairs
    ]
    valid_topk_counts = [
        topk if rng.random() >= 0.25 else rng.randrange(0, topk + 1)
        for _ in request_lane_pairs
    ]
    seq_lens = [max_model_len] * requests

    topk_positions: list[list[int]] = []
    for request_index, lane in request_lane_pairs:
        resident_tokens = [
            token
            for token in state.hot_to_token[request_index]
            if token != INVALID_INDEX
        ]
        current_positions = [
            max_model_len - lanes + candidate_lane
            for candidate_lane in range(lanes)
        ]
        candidates = resident_tokens + current_positions
        request_topk = []
        for rank in range(topk):
            selector = rng.random()
            if candidates and selector < 0.35:
                token = rng.choice(candidates)
            elif selector < 0.85:
                token = rng.randrange(0, max_model_len - lanes)
            elif selector < 0.925:
                token = INVALID_INDEX
            else:
                token = max_model_len + rank + lane
            request_topk.append(token)
        topk_positions.append(request_topk)

    return CorrectnessCase(
        name=f"random_seed_{seed}",
        state=state,
        query_positions=query_positions,
        query_to_req_idx=query_to_req_idx,
        query_to_lane=query_to_lane,
        query_valid_mask=query_valid_mask,
        valid_topk_counts=valid_topk_counts,
        seq_lens=seq_lens,
        topk_positions=topk_positions,
    )


def _to_device_tensor(
    torch: Any,
    values: Any,
    *,
    dtype: Any,
    device: Any,
) -> Any:
    return torch.tensor(values, dtype=dtype, device=device).contiguous()


def _make_operator_inputs(runtime: Any, case: CorrectnessCase) -> OperatorInputs:
    torch = runtime.torch
    device = runtime.device
    requests = len(case.state.token_to_hot)
    slots = len(case.state.hot_to_token[0])
    query_count = len(case.query_positions)
    topk = len(case.topk_positions[0])

    return OperatorInputs(
        token_to_hot=_to_device_tensor(
            torch,
            case.state.token_to_hot,
            dtype=torch.int32,
            device=device,
        ),
        hot_to_token=_to_device_tensor(
            torch,
            case.state.hot_to_token,
            dtype=torch.int32,
            device=device,
        ),
        lru_slots=_to_device_tensor(
            torch,
            case.state.lru_slots,
            dtype=torch.int32,
            device=device,
        ),
        query_positions=_to_device_tensor(
            torch,
            case.query_positions,
            dtype=torch.int32,
            device=device,
        ),
        query_to_req_idx=_to_device_tensor(
            torch,
            case.query_to_req_idx,
            dtype=torch.int32,
            device=device,
        ),
        query_to_lane=_to_device_tensor(
            torch,
            case.query_to_lane,
            dtype=torch.int32,
            device=device,
        ),
        query_valid_mask=_to_device_tensor(
            torch,
            case.query_valid_mask,
            dtype=torch.bool,
            device=device,
        ),
        valid_topk_counts=_to_device_tensor(
            torch,
            case.valid_topk_counts,
            dtype=torch.int32,
            device=device,
        ),
        seq_lens=_to_device_tensor(
            torch,
            case.seq_lens,
            dtype=torch.int32,
            device=device,
        ),
        topk_positions=_to_device_tensor(
            torch,
            case.topk_positions,
            dtype=torch.int32,
            device=device,
        ),
        resolved_hot_indices=torch.full(
            (query_count, topk),
            0x5A5A5A5A,
            dtype=torch.int32,
            device=device,
        ),
        miss_mask=torch.ones(
            (query_count, topk),
            dtype=torch.bool,
            device=device,
        ),
        workspace=torch.empty(
            (requests, workspace_stride(slots)),
            dtype=torch.int32,
            device=device,
        ),
    )


def _assert_tensor_equal(
    torch: Any,
    *,
    case_name: str,
    tensor_name: str,
    actual: Any,
    expected_values: Any,
    dtype: Any,
) -> None:
    actual_cpu = actual.detach().cpu()
    expected = torch.tensor(expected_values, dtype=dtype)
    if torch.equal(actual_cpu, expected):
        return

    mismatch = (actual_cpu != expected).nonzero()
    locations = mismatch[:10].tolist()
    details = []
    for location in locations:
        index = tuple(location)
        details.append(
            {
                "index": location,
                "actual": actual_cpu[index].item(),
                "expected": expected[index].item(),
            }
        )
    raise AssertionError(
        f"{case_name}: {tensor_name} mismatch at {details}; "
        f"total mismatches={mismatch.shape[0]}."
    )


def _run_case(runtime: Any, case: CorrectnessCase) -> None:
    expected_state = copy.deepcopy(case.state)
    expected_resolved, expected_miss = dsa_sparse_lookup_update_reference(
        expected_state,
        query_positions=case.query_positions,
        query_to_req_idx=case.query_to_req_idx,
        query_to_lane=case.query_to_lane,
        query_valid_mask=case.query_valid_mask,
        valid_topk_counts=case.valid_topk_counts,
        seq_lens=case.seq_lens,
        topk_positions=case.topk_positions,
    )

    inputs = _make_operator_inputs(runtime, case)
    invoke(runtime, inputs)
    runtime.torch.npu.synchronize()

    comparisons = (
        (
            "token_to_hot",
            inputs.token_to_hot,
            expected_state.token_to_hot,
            runtime.torch.int32,
        ),
        (
            "hot_to_token",
            inputs.hot_to_token,
            expected_state.hot_to_token,
            runtime.torch.int32,
        ),
        (
            "lru_slots",
            inputs.lru_slots,
            expected_state.lru_slots,
            runtime.torch.int32,
        ),
        (
            "resolved_hot_indices",
            inputs.resolved_hot_indices,
            expected_resolved,
            runtime.torch.int32,
        ),
        (
            "miss_mask",
            inputs.miss_mask,
            expected_miss,
            runtime.torch.bool,
        ),
    )
    for tensor_name, actual, expected, dtype in comparisons:
        _assert_tensor_equal(
            runtime.torch,
            case_name=case.name,
            tensor_name=tensor_name,
            actual=actual,
            expected_values=expected,
            dtype=dtype,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Ascend 950 dsa_sparse_lookup_update operator with "
            "the repository CPU oracle."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--random-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--topk", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.random_cases < 0:
        raise ValueError(f"random-cases must be non-negative, got {args.random_cases}.")
    validate_dimensions(
        requests=args.requests,
        max_model_len=args.max_model_len,
        slots=args.slots,
        lanes=args.lanes,
        topk=args.topk,
    )
    if args.max_model_len < args.slots:
        raise ValueError(
            "random correctness generation requires max-model-len >= slots, "
            f"got {args.max_model_len} < {args.slots}."
        )

    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
    )
    cases = _handcrafted_cases()
    cases.extend(
        _random_case(
            seed=args.seed + case_index,
            requests=args.requests,
            max_model_len=args.max_model_len,
            slots=args.slots,
            lanes=args.lanes,
            topk=args.topk,
        )
        for case_index in range(args.random_cases)
    )

    with runtime.torch.inference_mode():
        for case_index, case in enumerate(cases, start=1):
            _run_case(runtime, case)
            if case_index == len(cases) or case_index % 25 == 0:
                print(f"PASS {case_index}/{len(cases)}: {case.name}")

    print(
        "PASS: dsa_sparse_lookup_update matched the CPU oracle for "
        f"{len(cases)} cases on {args.device}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
