# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INVALID_INDEX = -1
SIMT_THREADS = 256
MAX_QUERY_LANES = 4
WORKSPACE_COUNTERS = 4

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
DEFAULT_INSTALL_ROOT = TOOL_DIR / ".install"
CHECKOUT_INSTALL_ROOT = REPO_ROOT / "vllm_ascend" / "_cann_ops_custom"
CUSTOM_OP_VENDOR = "custom_transformer"


@dataclass
class Runtime:
    torch: Any
    torch_npu: Any
    operator: Any
    device: Any
    install_root: Path | None


@dataclass
class OperatorInputs:
    token_to_hot: Any
    hot_to_token: Any
    lru_slots: Any
    state_seat_epoch: Any
    row_to_cache_seat: Any
    row_seat_epoch: Any
    query_positions: Any
    query_to_row: Any
    query_to_lane: Any
    query_valid_mask: Any
    valid_topk_counts: Any
    seq_lens: Any
    topk_positions: Any
    resolved_hot_indices: Any
    miss_mask: Any
    workspace: Any

    def arguments(self) -> tuple[Any, ...]:
        return (
            self.token_to_hot,
            self.hot_to_token,
            self.lru_slots,
            self.state_seat_epoch,
            self.row_to_cache_seat,
            self.row_seat_epoch,
            self.query_positions,
            self.query_to_row,
            self.query_to_lane,
            self.query_valid_mask,
            self.valid_topk_counts,
            self.seq_lens,
            self.topk_positions,
            self.resolved_hot_indices,
            self.miss_mask,
            self.workspace,
        )


def workspace_stride(evictable_slots: int) -> int:
    if evictable_slots <= 0:
        raise ValueError(f"evictable_slots must be positive, got {evictable_slots}.")
    return 3 * evictable_slots + 3 * SIMT_THREADS + WORKSPACE_COUNTERS


def validate_dimensions(
    *,
    seats: int,
    rows: int,
    max_model_len: int,
    slots: int,
    lanes: int,
    topk: int,
) -> None:
    dimensions = {
        "seats": seats,
        "rows": rows,
        "max_model_len": max_model_len,
        "slots": slots,
        "lanes": lanes,
        "topk": topk,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if rows > seats:
        raise ValueError(f"rows must not exceed seats, got rows={rows}, seats={seats}.")
    if lanes > MAX_QUERY_LANES:
        raise ValueError(f"lanes must be at most {MAX_QUERY_LANES}, got {lanes}.")
    if slots < lanes * topk:
        raise ValueError(
            "slots must cover the complete per-row Top-K union, got "
            f"slots={slots}, lanes={lanes}, topk={topk}."
        )


def _prepend_env_path(name: str, path: Path) -> None:
    path_text = str(path)
    current = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if path_text not in current:
        current.insert(0, path_text)
        os.environ[name] = ":".join(current)


def _resolve_install_root(install_root: str | Path | None) -> Path | None:
    if install_root is not None:
        resolved = Path(install_root).expanduser().resolve()
        vendor_root = resolved / "vendors" / CUSTOM_OP_VENDOR
        if not vendor_root.is_dir():
            raise RuntimeError(
                f"{resolved} does not contain vendors/{CUSTOM_OP_VENDOR}. "
                "Run build_and_install.sh first or pass the correct --install-root."
            )
        return resolved

    for candidate in (DEFAULT_INSTALL_ROOT, CHECKOUT_INSTALL_ROOT):
        if (candidate / "vendors" / CUSTOM_OP_VENDOR).is_dir():
            return candidate.resolve()
    return None


def load_runtime(*, device: str, install_root: str | Path | None) -> Runtime:
    resolved_install_root = _resolve_install_root(install_root)
    if resolved_install_root is not None:
        vendor_root = resolved_install_root / "vendors" / CUSTOM_OP_VENDOR
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_root)
        vendor_lib = vendor_root / "op_api" / "lib"
        if vendor_lib.is_dir():
            _prepend_env_path("LD_LIBRARY_PATH", vendor_lib)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        torch = importlib.import_module("torch")
        torch_npu = importlib.import_module("torch_npu")
    except ImportError as error:
        raise RuntimeError(
            "PyTorch/torch_npu is unavailable. Run this tool in the Ascend 950 "
            "vLLM-Ascend build environment."
        ) from error

    try:
        torch.npu.set_device(device)
    except Exception as error:
        raise RuntimeError(f"Unable to select {device}: {error}") from error

    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
        operator = torch.ops._C_ascend.dsa_sparse_lookup_update
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        install_hint = (
            f" from {resolved_install_root}" if resolved_install_root is not None else ""
        )
        raise RuntimeError(
            "Unable to load torch.ops._C_ascend.dsa_sparse_lookup_update"
            f"{install_hint}. Build the extension and install the single-op package."
        ) from error

    return Runtime(
        torch=torch,
        torch_npu=torch_npu,
        operator=operator,
        device=torch.device(device),
        install_root=resolved_install_root,
    )


def invoke(runtime: Runtime, inputs: OperatorInputs) -> None:
    runtime.operator(*inputs.arguments())


def make_profile_inputs(
    runtime: Runtime,
    *,
    seats: int,
    rows: int,
    max_model_len: int,
    slots: int,
    lanes: int,
    topk: int,
) -> OperatorInputs:
    validate_dimensions(
        seats=seats,
        rows=rows,
        max_model_len=max_model_len,
        slots=slots,
        lanes=lanes,
        topk=topk,
    )
    required_valid_tokens = lanes * topk + lanes
    if max_model_len < required_valid_tokens:
        raise ValueError(
            "max_model_len is too small for a unique valid Top-K union and "
            f"reserved query positions; need at least {required_valid_tokens}, "
            f"got {max_model_len}."
        )

    torch = runtime.torch
    device = runtime.device
    query_count = rows * lanes

    token_to_hot = torch.full(
        (seats, max_model_len),
        INVALID_INDEX,
        dtype=torch.int32,
        device=device,
    )
    hot_to_token = torch.full(
        (seats, slots),
        INVALID_INDEX,
        dtype=torch.int32,
        device=device,
    )
    lru_slots = (
        torch.arange(slots, dtype=torch.int32, device=device)
        .expand(seats, -1)
        .clone()
    )
    state_seat_epoch = torch.full(
        (seats,),
        INVALID_INDEX,
        dtype=torch.int32,
        device=device,
    )
    row_to_cache_seat = torch.arange(rows, dtype=torch.int32, device=device)
    row_seat_epoch = torch.zeros(rows, dtype=torch.int32, device=device)

    query_to_row = torch.arange(rows, dtype=torch.int32, device=device).repeat_interleave(lanes)
    query_to_lane = torch.arange(lanes, dtype=torch.int32, device=device).repeat(rows)
    query_positions = (
        torch.arange(lanes, dtype=torch.int32, device=device)
        .repeat(rows)
        .add(max_model_len - lanes)
    )
    query_valid_mask = torch.ones(query_count, dtype=torch.bool, device=device)
    valid_topk_counts = torch.full(
        (query_count,),
        topk,
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full(
        (rows,),
        max_model_len,
        dtype=torch.int32,
        device=device,
    )

    row_topk = torch.arange(lanes * topk, dtype=torch.int32, device=device).reshape(lanes, topk)
    topk_positions = row_topk.repeat(rows, 1)
    resolved_hot_indices = torch.full(
        (query_count, topk),
        INVALID_INDEX,
        dtype=torch.int32,
        device=device,
    )
    miss_mask = torch.zeros(
        (query_count, topk),
        dtype=torch.bool,
        device=device,
    )
    workspace = torch.empty(
        (rows, workspace_stride(slots)),
        dtype=torch.int32,
        device=device,
    )

    return OperatorInputs(
        token_to_hot=token_to_hot,
        hot_to_token=hot_to_token,
        lru_slots=lru_slots,
        state_seat_epoch=state_seat_epoch,
        row_to_cache_seat=row_to_cache_seat,
        row_seat_epoch=row_seat_epoch,
        query_positions=query_positions,
        query_to_row=query_to_row,
        query_to_lane=query_to_lane,
        query_valid_mask=query_valid_mask,
        valid_topk_counts=valid_topk_counts,
        seq_lens=seq_lens,
        topk_positions=topk_positions,
        resolved_hot_indices=resolved_hot_indices,
        miss_mask=miss_mask,
        workspace=workspace,
    )
