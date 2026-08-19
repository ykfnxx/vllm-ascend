# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass
from typing import Literal

from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_SLOT_COUNT,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_QUERY_WIDTH,
    DSA_SPARSE_RESIDENT_SLOT_COUNT,
)

DSASparseKVRole = Literal["kv_producer", "kv_consumer"]
DSA_SPARSE_MOCK_IO_BACKEND = "mock"
DSA_SPARSE_MAX_VERIFY_TOKENS_PER_REQUEST = 16


@dataclass(frozen=True)
class DSASparseConfig:
    """Validated configuration for the eager DSASparse path."""

    io_backend: str
    kv_role: DSASparseKVRole
    index_topk: int
    speculative_method: str | None
    max_verify_tokens_per_request: int

    @property
    def is_producer(self) -> bool:
        return self.kv_role == "kv_producer"

    @property
    def is_consumer(self) -> bool:
        return self.kv_role == "kv_consumer"

    @property
    def uses_mtp(self) -> bool:
        return self.speculative_method == "mtp"


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

    allowed_keys = {"io_backend"}
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown dsa_sparse_config fields: {unknown_keys}.")

    io_backend = raw_config.get("io_backend")
    if not isinstance(io_backend, str) or not io_backend.strip():
        raise ValueError("dsa_sparse_config.io_backend must be a non-empty string.")
    if io_backend != DSA_SPARSE_MOCK_IO_BACKEND:
        raise ValueError(
            "The current DSA Sparse Graph-out milestone supports only "
            "io_backend='mock'; no concrete I/O backend is implemented."
        )

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

    speculative_method, max_verify_tokens_per_request = (
        _get_supported_speculative_config(vllm_config)
    )
    index_topk = _get_index_topk(model_config)
    if index_topk != DSA_SPARSE_QUERY_WIDTH:
        raise ValueError(
            f"DSA Sparse requires index_topk={DSA_SPARSE_QUERY_WIDTH}, "
            f"got {index_topk}."
        )
    max_model_len = getattr(model_config, "max_model_len", None)
    if (
        not isinstance(max_model_len, int)
        or not 0 < max_model_len <= DSA_SPARSE_INDEX_CAPACITY
    ):
        raise ValueError(
            "DSA Sparse max_model_len must fit the 128K ASU index."
        )
    cache_config = getattr(vllm_config, "cache_config", None)
    block_size = getattr(cache_config, "block_size", None)
    if (
        not isinstance(block_size, int)
        or block_size <= 0
        or DSA_SPARSE_RESIDENT_SLOT_COUNT % block_size
        or DSA_SPARSE_FREE_SLOT_COUNT % block_size
    ):
        raise ValueError(
            "DSA Sparse block_size must divide the 8K resident and 2K free regions."
        )

    return DSASparseConfig(
        io_backend=io_backend,
        kv_role=kv_role,
        index_topk=index_topk,
        speculative_method=speculative_method,
        max_verify_tokens_per_request=max_verify_tokens_per_request,
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


def _get_supported_speculative_config(
    vllm_config: object,
) -> tuple[str | None, int]:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return None, 1
    num_speculative_tokens = (
        getattr(speculative_config, "num_speculative_tokens", 0)
    )
    if isinstance(num_speculative_tokens, bool) or not isinstance(
        num_speculative_tokens,
        int,
    ):
        raise ValueError("num_speculative_tokens must be an integer for DSA Sparse.")
    if num_speculative_tokens < 0:
        raise ValueError(
            "num_speculative_tokens must not be negative for DSA Sparse."
        )
    if num_speculative_tokens == 0:
        return None, 1

    speculative_method = getattr(speculative_config, "method", None)
    if speculative_method != "mtp":
        raise ValueError(
            "DSA Sparse supports speculative decoding only with method='mtp'."
        )
    max_verify_tokens_per_request = num_speculative_tokens + 1
    if (
        max_verify_tokens_per_request
        > DSA_SPARSE_MAX_VERIFY_TOKENS_PER_REQUEST
    ):
        raise ValueError(
            "DSA Sparse MTP requires num_speculative_tokens + 1 <= "
            f"{DSA_SPARSE_MAX_VERIFY_TOKENS_PER_REQUEST}, got "
            f"{max_verify_tokens_per_request}."
        )
    return speculative_method, max_verify_tokens_per_request


def _require_parallel_size(parallel_config: object, field: str) -> None:
    size = getattr(parallel_config, field, 1)
    if size != 1:
        raise ValueError(f"DSA Sparse requires {field}=1, got {size}.")
