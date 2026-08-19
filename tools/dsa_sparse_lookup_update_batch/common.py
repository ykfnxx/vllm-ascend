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
QUERY_WIDTH = 2 * 1024
FREE_HEAD_STRIDE = 16
BATCH_OPERATOR = "dsa_sparse_lookup_update_batch"

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
DEFAULT_INSTALL_ROOT = TOOL_DIR / ".install"
CUSTOM_OP_VENDOR = "custom_transformer"


@dataclass
class Runtime:
    torch: Any
    torch_npu: Any
    operator: Any
    device: Any
    install_root: Path | None


@dataclass
class BatchOperatorInputs:
    index: Any
    slot_to_index: Any
    free_slots: Any
    free_head: Any
    req_pool_entries: Any
    query_start_loc: Any
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
            self.query_start_loc,
            self.query_index,
            self.lookup_mask,
            self.req_num,
        )


def _prepend_env_path(name: str, path: Path) -> None:
    entries = [
        entry
        for entry in os.environ.get(name, "").split(":")
        if entry
    ]
    path_text = str(path)
    if path_text not in entries:
        entries.insert(0, path_text)
        os.environ[name] = ":".join(entries)


def load_runtime(
    *,
    device: str,
    install_root: str | Path | None,
) -> Runtime:
    resolved_install_root = (
        Path(install_root).expanduser().resolve()
        if install_root is not None
        else DEFAULT_INSTALL_ROOT.resolve()
    )
    if resolved_install_root.is_dir():
        vendor_root = (
            resolved_install_root / "vendors" / CUSTOM_OP_VENDOR
        )
        if not vendor_root.is_dir():
            raise RuntimeError(
                f"{resolved_install_root} does not contain "
                f"vendors/{CUSTOM_OP_VENDOR}."
            )
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_root)
        vendor_lib = vendor_root / "op_api" / "lib"
        if vendor_lib.is_dir():
            _prepend_env_path("LD_LIBRARY_PATH", vendor_lib)
    elif install_root is not None:
        raise RuntimeError(
            f"Custom-op install root does not exist: {resolved_install_root}"
        )
    else:
        resolved_install_root = None

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        torch = importlib.import_module("torch")
        torch_npu = importlib.import_module("torch_npu")
        torch.npu.set_device(device)
        importlib.import_module("vllm_ascend.vllm_ascend_C")
        operator = getattr(torch.ops._C_ascend, BATCH_OPERATOR)
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "Unable to load the batch operator. Rebuild the Python extension "
            "and install dsa_sparse_lookup_update_batch first."
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
    inputs: BatchOperatorInputs,
) -> tuple[Any, Any]:
    return runtime.operator(*inputs.arguments())


def make_profile_inputs(
    runtime: Runtime,
    *,
    requests: int,
    queries_per_request: int,
    miss_count_per_query: int,
) -> BatchOperatorInputs:
    if requests <= 0 or queries_per_request <= 0:
        raise ValueError("requests and queries_per_request must be positive")
    if not 0 <= miss_count_per_query <= QUERY_WIDTH:
        raise ValueError(
            f"miss_count_per_query must be in [0, {QUERY_WIDTH}]"
        )
    max_miss_token = (
        RESIDENT_SLOT_COUNT
        + queries_per_request * miss_count_per_query
    )
    if max_miss_token > INDEX_CAPACITY:
        raise ValueError("query workload exceeds the 128K logical index")

    torch = runtime.torch
    index = torch.full(
        (requests, INDEX_CAPACITY),
        INVALID_INDEX,
        dtype=torch.int32,
        device=runtime.device,
    )
    slot_to_index = torch.full(
        (requests, SLOT_COUNT),
        INVALID_INDEX,
        dtype=torch.int32,
        device=runtime.device,
    )
    resident_tokens = torch.arange(
        RESIDENT_SLOT_COUNT,
        dtype=torch.int32,
        device=runtime.device,
    ).expand(requests, -1)
    resident_slots = resident_tokens.clone()
    index.scatter_(1, resident_tokens.long(), resident_slots)
    slot_to_index[:, :RESIDENT_SLOT_COUNT].copy_(resident_tokens)
    free_slots = (
        torch.arange(
            RESIDENT_SLOT_COUNT,
            SLOT_COUNT,
            dtype=torch.int32,
            device=runtime.device,
        )
        .expand(requests, -1)
        .clone()
    )
    free_head = torch.zeros(
        (requests, FREE_HEAD_STRIDE),
        dtype=torch.int32,
        device=runtime.device,
    )
    req_pool_entries = torch.arange(
        requests,
        dtype=torch.int32,
        device=runtime.device,
    )
    query_start_loc = torch.arange(
        0,
        (requests + 1) * queries_per_request,
        queries_per_request,
        dtype=torch.int32,
        device=runtime.device,
    )

    hit_count = QUERY_WIDTH - miss_count_per_query
    rows = []
    for query_id in range(requests * queries_per_request):
        local_query = query_id % queries_per_request
        hits = torch.arange(
            hit_count,
            dtype=torch.int32,
            device=runtime.device,
        )
        misses = torch.arange(
            RESIDENT_SLOT_COUNT
            + local_query * miss_count_per_query,
            RESIDENT_SLOT_COUNT
            + (local_query + 1) * miss_count_per_query,
            dtype=torch.int32,
            device=runtime.device,
        )
        rows.append(torch.cat((hits, misses)))
    query_index = torch.stack(rows).contiguous()
    lookup_mask = torch.ones_like(query_index)
    return BatchOperatorInputs(
        index=index,
        slot_to_index=slot_to_index,
        free_slots=free_slots,
        free_head=free_head,
        req_pool_entries=req_pool_entries,
        query_start_loc=query_start_loc,
        query_index=query_index,
        lookup_mask=lookup_mask,
    )


def clone_inputs(inputs: BatchOperatorInputs) -> BatchOperatorInputs:
    return BatchOperatorInputs(
        **{
            name: getattr(inputs, name).clone()
            for name in BatchOperatorInputs.__dataclass_fields__
        }
    )


def restore_inputs(
    destination: BatchOperatorInputs,
    source: BatchOperatorInputs,
) -> None:
    for name in BatchOperatorInputs.__dataclass_fields__:
        getattr(destination, name).copy_(getattr(source, name))
