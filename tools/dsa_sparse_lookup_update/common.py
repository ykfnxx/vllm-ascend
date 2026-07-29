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
INDEX_CAPACITY = 128 * 1024
RESIDENT_SLOT_COUNT = 8 * 1024
FREE_SLOT_COUNT = 2 * 1024
SLOT_COUNT = 10 * 1024
QUERY_COUNT = 2 * 1024
FREE_HEAD_STRIDE = 16

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
DEFAULT_INSTALL_ROOT = TOOL_DIR / ".install"
CHECKOUT_INSTALL_ROOT = (
    REPO_ROOT / "vllm_ascend" / "_cann_ops_custom"
)
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
    index: Any
    slot_to_index: Any
    free_slots: Any
    free_head: Any
    req_pool_entries: Any
    query_index: Any
    lookup_mask: Any

    @property
    def req_num(self) -> int:
        return self.req_pool_entries.shape[0]

    def arguments(self) -> tuple[Any, ...]:
        return (
            self.index,
            self.slot_to_index,
            self.free_slots,
            self.free_head,
            self.req_pool_entries,
            self.query_index,
            self.lookup_mask,
            self.req_num,
        )


def validate_requests(requests: int) -> None:
    if requests <= 0:
        raise ValueError(
            f"requests must be positive, got {requests}."
        )


def _prepend_env_path(name: str, path: Path) -> None:
    path_text = str(path)
    current = [
        entry
        for entry in os.environ.get(name, "").split(":")
        if entry
    ]
    if path_text not in current:
        current.insert(0, path_text)
        os.environ[name] = ":".join(current)


def _resolve_install_root(
    install_root: str | Path | None,
) -> Path | None:
    if install_root is not None:
        resolved = Path(install_root).expanduser().resolve()
        vendor_root = resolved / "vendors" / CUSTOM_OP_VENDOR
        if not vendor_root.is_dir():
            raise RuntimeError(
                f"{resolved} does not contain "
                f"vendors/{CUSTOM_OP_VENDOR}. Run "
                "build_and_install.sh first or pass the correct "
                "--install-root."
            )
        return resolved

    for candidate in (
        DEFAULT_INSTALL_ROOT,
        CHECKOUT_INSTALL_ROOT,
    ):
        if (
            candidate / "vendors" / CUSTOM_OP_VENDOR
        ).is_dir():
            return candidate.resolve()
    return None


def load_runtime(
    *,
    device: str,
    install_root: str | Path | None,
) -> Runtime:
    resolved_install_root = _resolve_install_root(install_root)
    if resolved_install_root is not None:
        vendor_root = (
            resolved_install_root
            / "vendors"
            / CUSTOM_OP_VENDOR
        )
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
            "PyTorch/torch_npu is unavailable. Run this tool in "
            "the Ascend 950 vLLM-Ascend build environment."
        ) from error

    try:
        torch.npu.set_device(device)
    except Exception as error:
        raise RuntimeError(
            f"Unable to select {device}: {error}"
        ) from error

    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
        operator = (
            torch.ops._C_ascend.dsa_sparse_lookup_update
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
    ) as error:
        install_hint = (
            f" from {resolved_install_root}"
            if resolved_install_root is not None
            else ""
        )
        raise RuntimeError(
            "Unable to load "
            "torch.ops._C_ascend.dsa_sparse_lookup_update"
            f"{install_hint}. Rebuild both the extension binding "
            "and the single-op package."
        ) from error

    return Runtime(
        torch=torch,
        torch_npu=torch_npu,
        operator=operator,
        device=torch.device(device),
        install_root=resolved_install_root,
    )


def invoke(
    runtime: Runtime,
    inputs: OperatorInputs,
) -> tuple[Any, Any]:
    return runtime.operator(*inputs.arguments())


def make_profile_inputs(
    runtime: Runtime,
    *,
    requests: int,
) -> OperatorInputs:
    validate_requests(requests)
    torch = runtime.torch
    device = runtime.device

    index = torch.full(
        (requests, INDEX_CAPACITY),
        INVALID_INDEX,
        dtype=torch.int32,
        device=device,
    )
    slot_to_index = torch.full(
        (requests, SLOT_COUNT),
        INVALID_INDEX,
        dtype=torch.int32,
        device=device,
    )
    resident = torch.arange(
        RESIDENT_SLOT_COUNT,
        dtype=torch.int32,
        device=device,
    ).expand(requests, -1)
    index[:, :RESIDENT_SLOT_COUNT].copy_(resident)
    slot_to_index[:, :RESIDENT_SLOT_COUNT].copy_(resident)
    free_slots = (
        torch.arange(
            RESIDENT_SLOT_COUNT,
            SLOT_COUNT,
            dtype=torch.int32,
            device=device,
        )
        .expand(requests, -1)
        .clone()
    )
    free_head = torch.zeros(
        (requests, FREE_HEAD_STRIDE),
        dtype=torch.int32,
        device=device,
    )
    req_pool_entries = torch.arange(
        requests,
        dtype=torch.int32,
        device=device,
    )
    query_index = (
        torch.arange(
            QUERY_COUNT,
            dtype=torch.int32,
            device=device,
        )
        .expand(requests, -1)
        .clone()
    )
    lookup_mask = torch.ones(
        (requests, QUERY_COUNT),
        dtype=torch.int32,
        device=device,
    )
    return OperatorInputs(
        index=index,
        slot_to_index=slot_to_index,
        free_slots=free_slots,
        free_head=free_head,
        req_pool_entries=req_pool_entries,
        query_index=query_index,
        lookup_mask=lookup_mask,
    )
