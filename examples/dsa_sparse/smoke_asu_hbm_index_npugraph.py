#!/usr/bin/env python3
"""Smoke-test the ASU HBM index operators with npugraph_ex."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


INDEX_SIZE = 128 * 1024
SLOT_COUNT = 10 * 1024
FREE_SLOT_COUNT = 2 * 1024
QUERY_COUNT = 2 * 1024
FREE_HEAD_STRIDE = 16
RESIDENT_COUNT = SLOT_COUNT - FREE_SLOT_COUNT
HIT_COUNT = QUERY_COUNT // 2
NOT_FOUND = -1
REQ_NUM = 1
SEED = 20260717


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile lookup and AICPU maintain into one npugraph_ex graph, "
            "then run capture and replay smoke checks."
        )
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="NPU device id (default: 0)",
    )
    return parser.parse_args()


def configure_custom_opp() -> tuple[Path, Path]:
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm_ascend is not installed in this environment")

    package_dir = Path(spec.origin).resolve().parent
    vendor_opp = (
        package_dir / "_cann_ops_custom" / "vendors" / "vllm-ascend"
    )
    aicpu_opp = vendor_opp / "op_impl" / "aicpu_transformer"
    if not vendor_opp.is_dir():
        raise RuntimeError(f"custom OPP directory does not exist: {vendor_opp}")
    if not aicpu_opp.is_dir():
        raise RuntimeError(f"AICPU OPP directory does not exist: {aicpu_opp}")

    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    custom_paths = f"{aicpu_opp}:{vendor_opp}"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{custom_paths}:{current}" if current else custom_paths
    )
    return package_dir, aicpu_opp


def load_custom_ops(torch) -> Path:
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C as extension

    for op_name in (
        "asu_hbm_index_lookup",
        "asu_hbm_index_maintain_aicpu",
    ):
        qualified_name = f"_C_ascend::{op_name}"
        if not hasattr(torch.ops._C_ascend, op_name):
            raise RuntimeError(f"PyTorch operator is not registered: {qualified_name}")
        for dispatch_key in ("PrivateUse1", "Meta"):
            if not torch._C._dispatch_has_kernel_for_dispatch_key(
                qualified_name, dispatch_key
            ):
                raise RuntimeError(
                    f"{qualified_name} has no {dispatch_key} implementation"
                )

    return Path(extension.__file__).resolve()


def build_initial_state(torch):
    index = torch.full((REQ_NUM, INDEX_SIZE), NOT_FOUND, dtype=torch.int32)
    slot_to_index = torch.full(
        (REQ_NUM, SLOT_COUNT), NOT_FOUND, dtype=torch.int32
    )
    free_slots = torch.arange(
        RESIDENT_COUNT, SLOT_COUNT, dtype=torch.int32
    ).unsqueeze(0)
    free_head = torch.zeros(
        (REQ_NUM, FREE_HEAD_STRIDE), dtype=torch.int32
    )
    req_pool_entries = torch.zeros(REQ_NUM, dtype=torch.int32)
    query_index = torch.empty((REQ_NUM, QUERY_COUNT), dtype=torch.int32)
    lookup_mask = torch.ones_like(query_index)

    resident_tokens = torch.arange(RESIDENT_COUNT, dtype=torch.int32)
    resident_slots = torch.arange(RESIDENT_COUNT, dtype=torch.int32)
    miss_tokens = torch.arange(
        RESIDENT_COUNT,
        RESIDENT_COUNT + HIT_COUNT,
        dtype=torch.int32,
    )

    index[0, resident_tokens.long()] = resident_slots
    slot_to_index[0, :RESIDENT_COUNT] = resident_tokens
    query_index[0, 0::2] = resident_tokens[:HIT_COUNT]
    query_index[0, 1::2] = miss_tokens

    expected_slots = torch.empty_like(query_index)
    expected_slots[0, 0::2] = resident_slots[:HIT_COUNT]
    expected_slots[0, 1::2] = free_slots[0, :HIT_COUNT]
    expected_misses = torch.zeros_like(query_index)
    expected_misses[0, 1::2] = 1

    state = (
        index,
        slot_to_index,
        free_slots,
        free_head,
        req_pool_entries,
        query_index,
        lookup_mask,
    )
    expected = (expected_slots, expected_misses)
    return state, expected


def assert_equal(torch, name: str, actual, expected) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = (actual != expected).nonzero()
    first = tuple(int(value) for value in mismatch[0])
    raise AssertionError(
        f"{name} mismatch at {first}: actual={int(actual[first])}, "
        f"expected={int(expected[first])}, "
        f"total_mismatches={int(mismatch.shape[0])}"
    )


def validate_result(torch, state, outputs, expected) -> None:
    index, slot_to_index, _, free_head, _, query_index, _ = state
    slot_out, miss_out = outputs
    expected_slots, expected_misses = expected

    slot_out_cpu = slot_out.cpu()
    miss_out_cpu = miss_out.cpu()
    query_index_cpu = query_index.cpu()
    assert_equal(torch, "slot output", slot_out_cpu, expected_slots)
    assert_equal(torch, "miss output", miss_out_cpu, expected_misses)
    assert_equal(
        torch,
        "free head after maintain",
        free_head.cpu(),
        torch.zeros((REQ_NUM, FREE_HEAD_STRIDE), dtype=torch.int32),
    )

    query_slots = index[0].index_select(0, query_index[0].long()).cpu()
    assert_equal(torch, "token-to-slot state", query_slots, slot_out_cpu[0])
    slot_tokens = slot_to_index[0].index_select(0, slot_out[0].long()).cpu()
    assert_equal(torch, "slot-to-token state", slot_tokens, query_index_cpu[0])


def reset_state(state, initial_state) -> None:
    for tensor, initial_tensor in zip(state, initial_state, strict=True):
        tensor.copy_(initial_tensor)


def main() -> None:
    args = parse_args()
    package_dir, aicpu_opp = configure_custom_opp()

    import torch

    extension_path = load_custom_ops(torch)
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)

    class ASUHbmIndexModel(torch.nn.Module):
        def forward(
            self,
            index,
            slot_to_index,
            free_slots,
            free_head,
            req_pool_entries,
            query_index,
            lookup_mask,
        ):
            slot_out, miss_out = (
                torch.ops._C_ascend.asu_hbm_index_lookup(
                    index,
                    slot_to_index,
                    free_slots,
                    free_head,
                    req_pool_entries,
                    query_index,
                    lookup_mask,
                    REQ_NUM,
                )
            )
            torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(
                index,
                slot_to_index,
                free_slots,
                free_head,
                req_pool_entries,
                slot_out,
                REQ_NUM,
                SEED,
            )
            return slot_out, miss_out

    host_state, expected = build_initial_state(torch)
    initial_state = tuple(tensor.to(device) for tensor in host_state)
    state = tuple(tensor.clone() for tensor in initial_state)

    model = ASUHbmIndexModel().npu()
    compiled_model = torch.compile(
        model,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
    )

    print(f"[INFO] vllm_ascend package={package_dir}")
    print(f"[INFO] extension={extension_path}")
    print(f"[INFO] AICPU OPP={aicpu_opp}")
    print(f"[INFO] device={device}")

    outputs = compiled_model(*state)
    torch.npu.synchronize()
    validate_result(torch, state, outputs, expected)
    print("[PASS] npugraph_ex capture and first forward")

    reset_state(state, initial_state)
    outputs = compiled_model(*state)
    torch.npu.synchronize()
    validate_result(torch, state, outputs, expected)
    print("[PASS] npugraph_ex replay")
    print("[PASS] ASU HBM lookup and AICPU maintain smoke test")


if __name__ == "__main__":
    main()
