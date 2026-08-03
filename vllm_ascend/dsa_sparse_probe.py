# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend

DSA_SPARSE_PROBE_LOG_PREFIX = "DSA_SPARSE_PROBE "
DSA_SPARSE_PD_LOG_PREFIX = "DSA_SPARSE_PD "


def is_enabled() -> bool:
    """Return whether the test-only eager-path probe is enabled."""

    return envs_ascend.VLLM_ASCEND_DSA_SPARSE_RUNTIME_PROBE


def pd_trace_is_enabled() -> bool:
    """Return whether P/D TopK handoff tracing is enabled."""

    return envs_ascend.VLLM_ASCEND_DSA_SPARSE_PD_TRACE


def synchronize_device() -> None:
    """Make a probe completion event mean that device work has completed."""

    torch.npu.synchronize()


def emit(event: str, **fields: Any) -> None:
    """Emit one stable, machine-readable probe event."""

    payload = {
        "event": event,
        **fields,
    }
    logger.info(
        "%s%s",
        DSA_SPARSE_PROBE_LOG_PREFIX,
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def emit_pd_handoff(
    event: str,
    *,
    role: str,
    request_id: object,
    handoff: Any,
    **fields: Any,
) -> None:
    """Emit one complete, comparable P/D TopK handoff event.

    The full serialized handoff is retained for exact diagnosis. Hashes make
    the common equality check cheap to read before inspecting the TopK lists.
    """

    if not pd_trace_is_enabled():
        return
    handoff_dict = handoff.to_dict()
    canonical_handoff = json.dumps(
        handoff_dict,
        sort_keys=True,
        separators=(",", ":"),
    )
    layer_topk_hashes = {
        rank: {
            layer_name: hashlib.sha256(
                json.dumps(
                    token_ids,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for layer_name, token_ids in layers.items()
        }
        for rank, layers in handoff_dict["layer_topk_by_rank"].items()
    }
    payload = {
        "event": event,
        "role": role,
        "request_id": str(request_id),
        "remote_request_id": handoff_dict["remote_request_id"],
        "handoff_sha256": hashlib.sha256(
            canonical_handoff.encode("utf-8")
        ).hexdigest(),
        "layer_topk_sha256_by_rank": layer_topk_hashes,
        "handoff": handoff_dict,
        **fields,
    }
    logger.info(
        "%s%s",
        DSA_SPARSE_PD_LOG_PREFIX,
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
