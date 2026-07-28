# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import json
from typing import Any

import torch
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend

DSA_SPARSE_PROBE_LOG_PREFIX = "DSA_SPARSE_PROBE "


def is_enabled() -> bool:
    """Return whether the test-only eager-path probe is enabled."""

    return envs_ascend.VLLM_ASCEND_DSA_SPARSE_RUNTIME_PROBE


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
