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
SIMT_OPERATOR = "dsa_sparse_lookup_update"
LOOKUP_OPERATOR = "asu_hbm_index_lookup"
MAINTAIN_OPERATOR = "asu_hbm_index_maintain_aicpu"
SUPPORTED_OPERATORS = frozenset(
    (SIMT_OPERATOR, LOOKUP_OPERATOR, MAINTAIN_OPERATOR)
)


@dataclass
class Runtime:
    torch: Any
    torch_npu: Any
    operator: Any
    operator_name: str
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


@dataclass
class MaintainInputs:
    index: Any
    slot_to_index: Any
    free_slots: Any
    free_head: Any
    req_pool_entries: Any
    last_query_slots: Any

    @property
    def req_num(self) -> int:
        return self.req_pool_entries.shape[0]

    def arguments(self, seed: int) -> tuple[Any, ...]:
        return (
            self.index,
            self.slot_to_index,
            self.free_slots,
            self.free_head,
            self.req_pool_entries,
            self.last_query_slots,
            self.req_num,
            seed,
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
    operator_name: str = SIMT_OPERATOR,
) -> Runtime:
    if operator_name not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unsupported operator {operator_name!r}; expected one of "
            f"{sorted(SUPPORTED_OPERATORS)}."
        )
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
            "an Ascend vLLM-Ascend build environment."
        ) from error

    try:
        torch.npu.set_device(device)
    except Exception as error:
        raise RuntimeError(
            f"Unable to select {device}: {error}"
        ) from error

    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
        operator = getattr(torch.ops._C_ascend, operator_name)
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
            f"Unable to load torch.ops._C_ascend.{operator_name}"
            f"{install_hint}. Rebuild both the extension binding "
            "and the single-op package."
        ) from error

    return Runtime(
        torch=torch,
        torch_npu=torch_npu,
        operator=operator,
        operator_name=operator_name,
        device=torch.device(device),
        install_root=resolved_install_root,
    )


def invoke(
    runtime: Runtime,
    inputs: OperatorInputs,
) -> tuple[Any, Any]:
    if runtime.operator_name == MAINTAIN_OPERATOR:
        raise ValueError(
            "Use invoke_maintain() for asu_hbm_index_maintain_aicpu."
        )
    return runtime.operator(*inputs.arguments())


def invoke_maintain(
    runtime: Runtime,
    inputs: MaintainInputs,
    *,
    seed: int,
) -> None:
    if runtime.operator_name != MAINTAIN_OPERATOR:
        raise ValueError(
            "invoke_maintain() requires "
            "asu_hbm_index_maintain_aicpu."
        )
    runtime.operator(*inputs.arguments(seed))


def make_profile_inputs(
    runtime: Runtime,
    *,
    requests: int,
    miss_count: int = 0,
) -> OperatorInputs:
    validate_requests(requests)
    _validate_miss_count(miss_count)
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
    hit_count = QUERY_COUNT - miss_count
    query_row = torch.cat(
        (
            torch.arange(
                hit_count,
                dtype=torch.int32,
                device=device,
            ),
            torch.arange(
                RESIDENT_SLOT_COUNT,
                RESIDENT_SLOT_COUNT + miss_count,
                dtype=torch.int32,
                device=device,
            ),
        )
    )
    query_index = query_row.expand(requests, -1).clone()
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


def make_maintain_profile_inputs(
    runtime: Runtime,
    *,
    requests: int,
    miss_count: int,
) -> MaintainInputs:
    _validate_miss_count(miss_count)
    lookup_inputs = make_profile_inputs(
        runtime,
        requests=requests,
        miss_count=miss_count,
    )
    torch = runtime.torch
    device = runtime.device
    hit_count = QUERY_COUNT - miss_count
    allocated_slots = torch.arange(
        RESIDENT_SLOT_COUNT,
        RESIDENT_SLOT_COUNT + miss_count,
        dtype=torch.int32,
        device=device,
    )
    if miss_count:
        lookup_inputs.index[
            :, RESIDENT_SLOT_COUNT : RESIDENT_SLOT_COUNT + miss_count
        ].copy_(allocated_slots.expand(requests, -1))
        lookup_inputs.slot_to_index[
            :, RESIDENT_SLOT_COUNT : RESIDENT_SLOT_COUNT + miss_count
        ].copy_(allocated_slots.expand(requests, -1))
        lookup_inputs.free_head[:, 0].fill_(miss_count)
    last_query_row = torch.cat(
        (
            torch.arange(
                hit_count,
                dtype=torch.int32,
                device=device,
            ),
            allocated_slots,
        )
    )
    return MaintainInputs(
        index=lookup_inputs.index,
        slot_to_index=lookup_inputs.slot_to_index,
        free_slots=lookup_inputs.free_slots,
        free_head=lookup_inputs.free_head,
        req_pool_entries=lookup_inputs.req_pool_entries,
        last_query_slots=last_query_row.expand(requests, -1).clone(),
    )


def clone_operator_inputs(
    inputs: OperatorInputs,
) -> OperatorInputs:
    return OperatorInputs(
        index=inputs.index.clone(),
        slot_to_index=inputs.slot_to_index.clone(),
        free_slots=inputs.free_slots.clone(),
        free_head=inputs.free_head.clone(),
        req_pool_entries=inputs.req_pool_entries.clone(),
        query_index=inputs.query_index.clone(),
        lookup_mask=inputs.lookup_mask.clone(),
    )


def clone_maintain_inputs(
    inputs: MaintainInputs,
) -> MaintainInputs:
    return MaintainInputs(
        index=inputs.index.clone(),
        slot_to_index=inputs.slot_to_index.clone(),
        free_slots=inputs.free_slots.clone(),
        free_head=inputs.free_head.clone(),
        req_pool_entries=inputs.req_pool_entries.clone(),
        last_query_slots=inputs.last_query_slots.clone(),
    )


def restore_operator_inputs(
    destination: OperatorInputs,
    source: OperatorInputs,
) -> None:
    for name in (
        "index",
        "slot_to_index",
        "free_slots",
        "free_head",
        "req_pool_entries",
        "query_index",
        "lookup_mask",
    ):
        getattr(destination, name).copy_(getattr(source, name))


def restore_maintain_inputs(
    destination: MaintainInputs,
    source: MaintainInputs,
) -> None:
    for name in (
        "index",
        "slot_to_index",
        "free_slots",
        "free_head",
        "req_pool_entries",
        "last_query_slots",
    ):
        getattr(destination, name).copy_(getattr(source, name))


def _validate_miss_count(miss_count: int) -> None:
    if not 0 <= miss_count <= QUERY_COUNT:
        raise ValueError(
            f"miss_count must be in [0, {QUERY_COUNT}], "
            f"got {miss_count}."
        )
