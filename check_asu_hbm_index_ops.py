#!/usr/bin/env python3
"""Check the ASU HBM lookup and AICPU maintain custom operators on NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


INDEX_SIZE = 128 * 1024
SLOT_COUNT = 10 * 1024
FREE_SLOT_COUNT = 2 * 1024
QUERY_COUNT = 2 * 1024
RESIDENT_COUNT = SLOT_COUNT - FREE_SLOT_COUNT
HIT_COUNT = QUERY_COUNT // 2
MISS_COUNT = QUERY_COUNT - HIT_COUNT
REQ_NUM = 2
POOL_NUM = 18
REQ_POOL_ENTRY_VALUES = (17, 1)
NOT_FOUND = -1
UINT32_MASK = (1 << 32) - 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run lookup and AICPU maintain, then compare their state with a "
            "CPU reference implementation."
        ))
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="NPU device id (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260714,
        help="Eviction seed passed to the maintain operator",
    )
    parser.add_argument(
        "--diagnose-aicpu",
        action="store_true",
        help=(
            "Inspect the packaged AICPU JSON and shared library without "
            "running either operator"
        ),
    )
    return parser.parse_args()


def _configure_custom_opp() -> Path:
    repo_root = Path(__file__).resolve().parent
    vendor_opp = (
        repo_root
        / "vllm_ascend"
        / "_cann_ops_custom"
        / "vendors"
        / "vllm-ascend"
    )
    custom_opp = vendor_opp / "op_impl" / "aicpu_transformer"
    if not custom_opp.is_dir():
        raise RuntimeError(
            f"AICPU custom OPP directory does not exist: {custom_opp}. "
            "Build vllm-ascend with custom kernels first."
        )

    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    custom_paths = f"{custom_opp}:{vendor_opp}"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{custom_paths}:{current}" if current else custom_paths
    )
    return custom_opp


def _hash32(value: int) -> int:
    value &= UINT32_MASK
    value ^= value >> 16
    value = (value * 0x7FEB352D) & UINT32_MASK
    value ^= value >> 15
    value = (value * 0x846CA68B) & UINT32_MASK
    value ^= value >> 16
    return value & UINT32_MASK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readelf_symbols(shared_library: Path) -> str:
    try:
        result = subprocess.run(
            ["readelf", "--dyn-syms", "--wide", str(shared_library)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("[FAIL] readelf is not installed")
        return ""

    if result.returncode != 0:
        print(f"[FAIL] readelf failed: {result.stderr.strip()}")
        return ""
    return result.stdout


def _diagnose_aicpu_package(custom_opp: Path) -> None:
    op_type = "AsuHbmIndexMaintainAicpu"
    json_path = custom_opp / "op_impl/cpu/config/cust_aicpu_kernel.json"
    shared_library = (
        custom_opp
        / "op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so"
    )
    vendor_config = custom_opp.parents[2] / "config.ini"
    failed = False

    print(f"[INFO] ASCEND_CUSTOM_OPP_PATH={os.environ['ASCEND_CUSTOM_OPP_PATH']}")
    print(
        f"[INFO] global vendor config={vendor_config} "
        f"exists={vendor_config.is_file()} required=False"
    )
    repository_suffix = custom_opp.name.rsplit("_", 1)[-1]
    print(f"[INFO] AICPU repository suffix={repository_suffix}")
    if repository_suffix != "transformer":
        print("[FAIL] CANN 8.5 will reject this AICPU repository suffix")
        failed = True
    print(f"[INFO] AICPU JSON={json_path} exists={json_path.is_file()}")
    print(
        f"[INFO] AICPU library={shared_library} "
        f"exists={shared_library.is_file()}"
    )

    if not json_path.is_file():
        print("[FAIL] packaged AICPU JSON is missing")
        failed = True
    else:
        try:
            package_info = json.loads(json_path.read_text(encoding="utf-8"))
            op_info = package_info[op_type]["opInfo"]
        except (json.JSONDecodeError, KeyError) as error:
            print(f"[FAIL] invalid AICPU JSON entry: {error}")
            failed = True
        else:
            for field in (
                "engine",
                "opKernelLib",
                "kernelSo",
                "functionName",
                "userDefined",
            ):
                print(f"[INFO] JSON {field}={op_info.get(field)}")
            if op_info.get("kernelSo") != shared_library.name:
                print("[FAIL] JSON kernelSo does not match the packaged library")
                failed = True
            if op_info.get("functionName") != "RunCpuKernel":
                print("[FAIL] JSON functionName is not RunCpuKernel")
                failed = True

    if not shared_library.is_file():
        print("[FAIL] packaged AICPU library is missing")
        failed = True
    else:
        print(
            f"[INFO] AICPU library size={shared_library.stat().st_size} "
            f"sha256={_sha256(shared_library)}"
        )
        symbols = _readelf_symbols(shared_library)
        run_cpu_kernel = re.search(r"\bRunCpuKernel\b", symbols) is not None
        direct_kernel = re.search(rf"\b{op_type}\b", symbols) is not None
        print(f"[INFO] dynamic symbol RunCpuKernel={run_cpu_kernel}")
        print(f"[INFO] dynamic symbol {op_type}={direct_kernel}")
        if not run_cpu_kernel:
            print("[FAIL] AICPU library does not export RunCpuKernel")
            failed = True

        binary = shared_library.read_bytes()
        registered_name = op_type.encode("ascii") in binary
        print(f"[INFO] embedded registered op name {op_type}={registered_name}")
        if not registered_name:
            print("[FAIL] AICPU registration name is absent from the library")
            failed = True

    if failed:
        raise SystemExit(1)
    print("[PASS] packaged AICPU metadata and binary entry are consistent")


def _build_initial_state(torch):
    index = torch.full(
        (POOL_NUM, INDEX_SIZE), NOT_FOUND, dtype=torch.int32
    )
    slot_to_index = torch.full(
        (POOL_NUM, SLOT_COUNT), NOT_FOUND, dtype=torch.int32
    )
    free_slots = torch.arange(
        RESIDENT_COUNT, SLOT_COUNT, dtype=torch.int32
    ).repeat(POOL_NUM, 1)
    free_head = torch.zeros(POOL_NUM, dtype=torch.int32)
    req_pool_entries = torch.tensor(
        REQ_POOL_ENTRY_VALUES, dtype=torch.int32
    )
    query_index = torch.empty(
        (REQ_NUM, QUERY_COUNT), dtype=torch.int32
    )
    resident_slots = torch.arange(RESIDENT_COUNT, dtype=torch.int32)

    for req_id in range(REQ_NUM):
        pool_entry = int(req_pool_entries[req_id])
        token_base = req_id * 2 * RESIDENT_COUNT
        resident_tokens = torch.arange(
            token_base,
            token_base + RESIDENT_COUNT,
            dtype=torch.int32,
        )
        miss_tokens = torch.arange(
            token_base + RESIDENT_COUNT,
            token_base + RESIDENT_COUNT + MISS_COUNT,
            dtype=torch.int32,
        )

        index[pool_entry, resident_tokens.long()] = resident_slots
        slot_to_index[pool_entry, :RESIDENT_COUNT] = resident_tokens
        query_index[req_id, 0::2] = resident_tokens[:HIT_COUNT]
        query_index[req_id, 1::2] = miss_tokens

    return (
        index,
        slot_to_index,
        free_slots,
        free_head,
        query_index,
        req_pool_entries,
    )


def _lookup_reference(
    index,
    slot_to_index,
    free_slots,
    free_head,
    query_index,
    req_pool_entries,
):
    slot_out = query_index.new_empty(query_index.shape)
    for req_id in range(REQ_NUM):
        pool_entry = int(req_pool_entries[req_id])
        head = int(free_head[pool_entry])
        for query_id in range(QUERY_COUNT):
            index_id = int(query_index[req_id, query_id])
            slot = int(index[pool_entry, index_id])
            if slot == NOT_FOUND:
                slot = int(free_slots[pool_entry, head])
                head += 1
                index[pool_entry, index_id] = slot
                slot_to_index[pool_entry, slot] = index_id
            slot_out[req_id, query_id] = slot
        free_head[pool_entry] = head
    return slot_out


def _maintain_reference(
    index,
    slot_to_index,
    free_slots,
    free_head,
    last_query_slots,
    req_pool_entries,
    seed: int,
) -> None:
    for req_id in range(REQ_NUM):
        pool_entry = int(req_pool_entries[req_id])
        head = int(free_head[pool_entry])
        if head == 0:
            continue

        protected = set(int(slot) for slot in last_query_slots[req_id])
        slot = _hash32((seed & UINT32_MASK) ^ pool_entry) % SLOT_COUNT
        while head > 0:
            index_id = int(slot_to_index[pool_entry, slot])
            if index_id != NOT_FOUND and slot not in protected:
                slot_to_index[pool_entry, slot] = NOT_FOUND
                index[pool_entry, index_id] = NOT_FOUND
                head -= 1
                free_slots[pool_entry, head] = slot
            slot += 1
            if slot == SLOT_COUNT:
                slot = 0
        free_head[pool_entry] = head


def _assert_equal(torch, name: str, actual, expected) -> None:
    if torch.equal(actual, expected):
        print(f"[PASS] {name}")
        return

    mismatch = (actual != expected).nonzero()
    first = tuple(int(value) for value in mismatch[0])
    raise AssertionError(
        f"{name} mismatch at {first}: "
        f"actual={int(actual[first])}, expected={int(expected[first])}; "
        f"total mismatches={int(mismatch.shape[0])}"
    )


def _load_ops(torch) -> None:
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    for op_name in (
        "asu_hbm_index_lookup",
        "asu_hbm_index_maintain_aicpu",
    ):
        if not hasattr(torch.ops._C_ascend, op_name):
            raise RuntimeError(f"torch operator is not registered: {op_name}")
    print("[PASS] lookup and maintain operators are registered")


def main() -> None:
    args = _parse_args()
    custom_opp = _configure_custom_opp()

    if args.diagnose_aicpu:
        _diagnose_aicpu_package(custom_opp)
        return

    import torch

    _load_ops(torch)
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)

    initial_state = _build_initial_state(torch)
    expected_index = initial_state[0].clone()
    expected_slot_to_index = initial_state[1].clone()
    expected_free_slots = initial_state[2].clone()
    expected_free_head = initial_state[3].clone()
    query_index = initial_state[4]
    req_pool_entries = initial_state[5]

    expected_slot_out = _lookup_reference(
        expected_index,
        expected_slot_to_index,
        expected_free_slots,
        expected_free_head,
        query_index,
        req_pool_entries,
    )

    index = initial_state[0].to(device)
    slot_to_index = initial_state[1].to(device)
    free_slots = initial_state[2].to(device)
    free_head = initial_state[3].to(device)
    query_index_npu = query_index.to(device)
    req_pool_entries_npu = req_pool_entries.to(device)

    slot_out = torch.ops._C_ascend.asu_hbm_index_lookup(
        index,
        slot_to_index,
        free_slots,
        free_head,
        req_pool_entries_npu,
        query_index_npu,
        REQ_NUM,
    )
    torch.npu.synchronize()

    actual_slot_out = slot_out.cpu()
    _assert_equal(torch, "lookup slot output", actual_slot_out, expected_slot_out)
    _assert_equal(
        torch,
        "lookup free head",
        free_head.cpu(),
        expected_free_head,
    )
    _assert_equal(
        torch,
        "lookup token-to-slot index",
        index.cpu(),
        expected_index,
    )
    _assert_equal(
        torch,
        "lookup slot-to-token index",
        slot_to_index.cpu(),
        expected_slot_to_index,
    )

    torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(
        index,
        slot_to_index,
        free_slots,
        free_head,
        req_pool_entries_npu,
        slot_out,
        REQ_NUM,
        args.seed,
    )
    torch.npu.synchronize()

    _maintain_reference(
        expected_index,
        expected_slot_to_index,
        expected_free_slots,
        expected_free_head,
        expected_slot_out,
        req_pool_entries,
        args.seed,
    )

    _assert_equal(torch, "maintain token-to-slot index", index.cpu(), expected_index)
    _assert_equal(
        torch,
        "maintain slot-to-token index",
        slot_to_index.cpu(),
        expected_slot_to_index,
    )
    _assert_equal(
        torch,
        "maintain free slots",
        free_slots.cpu(),
        expected_free_slots,
    )
    _assert_equal(
        torch,
        "maintain free head",
        free_head.cpu(),
        expected_free_head,
    )

    print(
        "ASU HBM index custom-op check passed: "
        f"device={device}, requests={REQ_NUM}, pool_entries="
        f"{REQ_POOL_ENTRY_VALUES}, hits/request={HIT_COUNT}, "
        f"misses/request={MISS_COUNT}, seed={args.seed}, custom_opp={custom_opp}"
    )


if __name__ == "__main__":
    main()
