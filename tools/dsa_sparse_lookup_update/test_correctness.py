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
    row_to_cache_seat: list[int]
    row_seat_epoch: list[int]
    query_positions: list[int]
    query_to_row: list[int]
    query_to_lane: list[int]
    query_valid_mask: list[bool]
    valid_topk_counts: list[int]
    seq_lens: list[int]
    topk_positions: list[list[int]]


def _make_empty_state(
    *,
    seats: int,
    max_model_len: int,
    slots: int,
    epochs: list[int] | None = None,
) -> DSASparseLookupUpdateState:
    return DSASparseLookupUpdateState(
        token_to_hot=[[INVALID_INDEX] * max_model_len for _ in range(seats)],
        hot_to_token=[[INVALID_INDEX] * slots for _ in range(seats)],
        lru_slots=[list(range(slots)) for _ in range(seats)],
        state_seat_epoch=list(epochs) if epochs is not None else [0] * seats,
    )


def _install(
    state: DSASparseLookupUpdateState,
    *,
    seat: int,
    slot: int,
    token: int,
) -> None:
    state.token_to_hot[seat][token] = slot
    state.hot_to_token[seat][slot] = token


def _handcrafted_cases() -> list[CorrectnessCase]:
    mixed_state = _make_empty_state(
        seats=2,
        max_model_len=16,
        slots=8,
        epochs=[7, 3],
    )
    _install(mixed_state, seat=0, slot=0, token=2)
    _install(mixed_state, seat=0, slot=3, token=9)
    _install(mixed_state, seat=1, slot=1, token=4)
    mixed_state.lru_slots[0] = [1, 4, 0, 2, 5, 6, 7, 3]
    mixed_state.lru_slots[1] = [7, 6, 5, 4, 3, 2, 0, 1]

    mixed = CorrectnessCase(
        name="mixed_hit_duplicate_reserved_inactive_reordered",
        state=mixed_state,
        row_to_cache_seat=[0, INVALID_INDEX],
        row_seat_epoch=[7, 99],
        query_positions=[15, 12, 14, 13],
        query_to_row=[0, 1, 0, 1],
        query_to_lane=[1, 0, 0, 1],
        query_valid_mask=[True, True, True, False],
        valid_topk_counts=[4, 4, 4, 2],
        seq_lens=[16, 14],
        topk_positions=[
            [9, 5, 15, 5],
            [4, 3, 2, 1],
            [2, 5, 14, INVALID_INDEX],
            [0, 1, 2, 3],
        ],
    )

    reset_state = _make_empty_state(
        seats=1,
        max_model_len=16,
        slots=4,
        epochs=[11],
    )
    _install(reset_state, seat=0, slot=0, token=2)
    _install(reset_state, seat=0, slot=2, token=7)
    reset_state.lru_slots[0] = [2, 1, 3, 0]
    reset = CorrectnessCase(
        name="seat_epoch_reset",
        state=reset_state,
        row_to_cache_seat=[0],
        row_seat_epoch=[12],
        query_positions=[15],
        query_to_row=[0],
        query_to_lane=[0],
        query_valid_mask=[True],
        valid_topk_counts=[4],
        seq_lens=[16],
        topk_positions=[[2, 7, 15, 5]],
    )
    return [mixed, reset]


def _random_state(
    rng: random.Random,
    *,
    seats: int,
    max_model_len: int,
    slots: int,
) -> DSASparseLookupUpdateState:
    state = _make_empty_state(
        seats=seats,
        max_model_len=max_model_len,
        slots=slots,
        epochs=[rng.randrange(0, 8) for _ in range(seats)],
    )
    for seat in range(seats):
        occupied = rng.randrange(0, slots + 1)
        resident_slots = rng.sample(range(slots), occupied)
        resident_tokens = rng.sample(range(max_model_len), occupied)
        for slot, token in zip(resident_slots, resident_tokens):
            _install(state, seat=seat, slot=slot, token=token)
        state.lru_slots[seat] = rng.sample(range(slots), slots)
    return state


def _random_case(
    *,
    seed: int,
    seats: int,
    rows: int,
    max_model_len: int,
    slots: int,
    lanes: int,
    topk: int,
) -> CorrectnessCase:
    rng = random.Random(seed)
    state = _random_state(
        rng,
        seats=seats,
        max_model_len=max_model_len,
        slots=slots,
    )

    row_to_cache_seat = [INVALID_INDEX] * rows
    active_count = rng.randrange(1, rows + 1)
    active_rows = rng.sample(range(rows), active_count)
    active_seats = rng.sample(range(seats), active_count)
    for row, seat in zip(active_rows, active_seats):
        row_to_cache_seat[row] = seat

    row_seat_epoch = []
    for row, seat in enumerate(row_to_cache_seat):
        if seat == INVALID_INDEX:
            row_seat_epoch.append(rng.randrange(0, 16))
        elif rng.random() < 0.25:
            row_seat_epoch.append(state.state_seat_epoch[seat] + 1)
        else:
            row_seat_epoch.append(state.state_seat_epoch[seat])

    row_lane_pairs = [(row, lane) for row in range(rows) for lane in range(lanes)]
    rng.shuffle(row_lane_pairs)
    query_to_row = [row for row, _ in row_lane_pairs]
    query_to_lane = [lane for _, lane in row_lane_pairs]
    query_positions = [
        max_model_len - lanes + lane
        for _, lane in row_lane_pairs
    ]
    query_valid_mask = [rng.random() >= 0.15 for _ in row_lane_pairs]
    valid_topk_counts = [
        topk if rng.random() >= 0.25 else rng.randrange(0, topk + 1)
        for _ in row_lane_pairs
    ]
    seq_lens = [max_model_len] * rows

    topk_positions: list[list[int]] = []
    for row, lane in row_lane_pairs:
        seat = row_to_cache_seat[row]
        resident_tokens = (
            [
                token
                for token in state.hot_to_token[seat]
                if token != INVALID_INDEX
            ]
            if seat != INVALID_INDEX
            else []
        )
        current_positions = [
            max_model_len - lanes + candidate_lane
            for candidate_lane in range(lanes)
        ]
        candidates = resident_tokens + current_positions
        row_topk = []
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
            row_topk.append(token)
        topk_positions.append(row_topk)

    return CorrectnessCase(
        name=f"random_seed_{seed}",
        state=state,
        row_to_cache_seat=row_to_cache_seat,
        row_seat_epoch=row_seat_epoch,
        query_positions=query_positions,
        query_to_row=query_to_row,
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
    rows = len(case.row_to_cache_seat)
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
        state_seat_epoch=_to_device_tensor(
            torch,
            case.state.state_seat_epoch,
            dtype=torch.int32,
            device=device,
        ),
        row_to_cache_seat=_to_device_tensor(
            torch,
            case.row_to_cache_seat,
            dtype=torch.int32,
            device=device,
        ),
        row_seat_epoch=_to_device_tensor(
            torch,
            case.row_seat_epoch,
            dtype=torch.int32,
            device=device,
        ),
        query_positions=_to_device_tensor(
            torch,
            case.query_positions,
            dtype=torch.int32,
            device=device,
        ),
        query_to_row=_to_device_tensor(
            torch,
            case.query_to_row,
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
            (rows, workspace_stride(slots)),
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
        row_to_cache_seat=case.row_to_cache_seat,
        row_seat_epoch=case.row_seat_epoch,
        query_positions=case.query_positions,
        query_to_row=case.query_to_row,
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
            "state_seat_epoch",
            inputs.state_seat_epoch,
            expected_state.state_seat_epoch,
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
    parser.add_argument("--seats", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
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
        seats=args.seats,
        rows=args.rows,
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
            seats=args.seats,
            rows=args.rows,
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
