# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

DSASparseKVRole = Literal["kv_producer", "kv_consumer"]


@dataclass(frozen=True)
class DSASparseConfig:
    """Validated framework configuration for the eager DSA Sparse milestone."""

    io_backend: str
    io_backend_options: Mapping[str, Any]
    kv_role: DSASparseKVRole
    device_buffer_size: int | None
    max_query_tokens_per_request: int
    index_topk: int

    @property
    def is_producer(self) -> bool:
        return self.kv_role == "kv_producer"

    @property
    def is_consumer(self) -> bool:
        return self.kv_role == "kv_consumer"


def load_dsa_sparse_config(vllm_config: object) -> DSASparseConfig | None:
    """Parse and validate ``additional_config.dsa_sparse_config``.

    The section's presence enables the feature. This development milestone is
    intentionally eager-only; graph execution must fail during configuration
    instead of silently selecting a baseline or full-Main fallback.
    """

    additional_config = getattr(vllm_config, "additional_config", None)
    if additional_config is None:
        return None
    if not isinstance(additional_config, dict):
        raise TypeError("additional_config must be a dictionary.")
    if "dsa_sparse_config" not in additional_config:
        return None

    raw_config = additional_config["dsa_sparse_config"]
    if not isinstance(raw_config, dict):
        raise TypeError("dsa_sparse_config must be a dictionary.")

    allowed_keys = {
        "io_backend",
        "io_backend_options",
        "device_buffer_size",
    }
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown dsa_sparse_config fields: {unknown_keys}.")

    io_backend = raw_config.get("io_backend")
    if not isinstance(io_backend, str) or not io_backend.strip():
        raise ValueError("dsa_sparse_config.io_backend must be a non-empty string.")

    io_backend_options = raw_config.get("io_backend_options", {})
    if not isinstance(io_backend_options, dict):
        raise TypeError("dsa_sparse_config.io_backend_options must be a dictionary.")

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        raise ValueError("DSA Sparse requires model_config.")
    if not bool(getattr(model_config, "enforce_eager", False)):
        raise ValueError("This DSA Sparse milestone supports only the eager execution path.")

    model_type = _get_model_type(model_config)
    if model_type != "glm_moe_dsa":
        raise ValueError(
            f"DSA Sparse currently supports only GLM-5 sparse attention (model_type='glm_moe_dsa'), got {model_type!r}."
        )

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_role not in {"kv_producer", "kv_consumer"}:
        raise ValueError("DSA Sparse requires a P/D-only role: kv_role must be 'kv_producer' or 'kv_consumer'.")

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        raise ValueError("DSA Sparse requires parallel_config.")
    _require_parallel_size(parallel_config, "pipeline_parallel_size")
    _require_parallel_size(parallel_config, "decode_context_parallel_size")
    _require_parallel_size(parallel_config, "prefill_context_parallel_size")

    max_query_tokens_per_request = _get_max_query_tokens_per_request(vllm_config)
    index_topk = _get_index_topk(model_config)
    device_buffer_size = raw_config.get("device_buffer_size")
    if kv_role == "kv_producer":
        if device_buffer_size is not None:
            raise ValueError("dsa_sparse_config.device_buffer_size is Decode-only and must not be set for kv_producer.")
    else:
        if isinstance(device_buffer_size, bool) or not isinstance(device_buffer_size, int):
            raise ValueError("dsa_sparse_config.device_buffer_size must be a positive integer for kv_consumer.")
        if device_buffer_size <= 0:
            raise ValueError("dsa_sparse_config.device_buffer_size must be a positive integer for kv_consumer.")
        minimum_size = max_query_tokens_per_request * index_topk
        if device_buffer_size < minimum_size:
            raise ValueError(
                "dsa_sparse_config.device_buffer_size must hold the complete "
                f"per-request Top-K union, expected at least {minimum_size}, "
                f"got {device_buffer_size}."
            )

    return DSASparseConfig(
        io_backend=io_backend,
        io_backend_options=MappingProxyType(dict(io_backend_options)),
        kv_role=kv_role,
        device_buffer_size=device_buffer_size,
        max_query_tokens_per_request=max_query_tokens_per_request,
        index_topk=index_topk,
    )


def _get_model_type(model_config: object) -> str | None:
    for config_name in ("hf_text_config", "hf_config"):
        config = getattr(model_config, config_name, None)
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str):
            return model_type
    return None


def _get_index_topk(model_config: object) -> int:
    for config_name in ("hf_text_config", "hf_config"):
        config = getattr(model_config, config_name, None)
        index_topk = getattr(config, "index_topk", None)
        if isinstance(index_topk, bool):
            continue
        if isinstance(index_topk, int) and index_topk > 0:
            return index_topk
    raise ValueError("DSA Sparse requires a positive GLM-5 index_topk.")


def _get_max_query_tokens_per_request(vllm_config: object) -> int:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_speculative_tokens = (
        getattr(speculative_config, "num_speculative_tokens", 0) if speculative_config is not None else 0
    )
    if isinstance(num_speculative_tokens, bool) or not isinstance(
        num_speculative_tokens,
        int,
    ):
        raise ValueError("num_speculative_tokens must be an integer for DSA Sparse.")
    if not 0 <= num_speculative_tokens <= 3:
        raise ValueError("DSA Sparse eager currently supports 0 to 3 speculative tokens.")
    return 1 + num_speculative_tokens


def _require_parallel_size(parallel_config: object, field: str) -> None:
    size = getattr(parallel_config, field, 1)
    if size != 1:
        raise ValueError(f"DSA Sparse requires {field}=1, got {size}.")
