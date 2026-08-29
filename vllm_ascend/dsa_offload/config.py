# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass
from typing import Any, Literal

from .constants import INDEX_CAPACITY, QUERY_WIDTH

PREFETCH_TOP_K_MIN = 128

DSAOffloadRole = Literal["kv_producer", "kv_consumer", "kv_both"]


@dataclass(frozen=True)
class DSAOffloadConfig:
    io_backend: Literal["kvio", "mock", "kvgather_sim"]
    kvio_model_id: int
    max_verify_tokens_per_request: int
    kv_role: DSAOffloadRole
    kv_transfer_config: Any | None
    enable_prefetch_with_hidden_states: bool
    prefetch_top_k: int
    enable_turbo_lookup: bool
    enable_turbo_prefetch_lookup: bool

    @property
    def has_connector(self) -> bool:
        return self.kv_transfer_config is not None

    @property
    def is_producer(self) -> bool:
        return self.kv_role in {"kv_producer", "kv_both"}

    @property
    def is_consumer(self) -> bool:
        return self.kv_role in {"kv_consumer", "kv_both"}


def load_dsa_offload_config(vllm_config: object) -> DSAOffloadConfig | None:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict) or "dsa_offload" not in additional_config:
        return None

    raw_config = additional_config["dsa_offload"]
    if not isinstance(raw_config, dict):
        raise ValueError("additional_config.dsa_offload must be an object.")
    io_backend = raw_config.get("io_backend", "mock")
    if io_backend not in {"kvio", "mock", "kvgather_sim"}:
        raise ValueError(
            "dsa_offload.io_backend must be 'kvio', 'mock', or 'kvgather_sim'."
        )

    enable_prefetch = raw_config.get("enable_prefetch_with_hidden_states", False)
    if not isinstance(enable_prefetch, bool):
        raise TypeError("dsa_offload.enable_prefetch_with_hidden_states must be a boolean.")
    prefetch_top_k = raw_config.get("prefetch_top_k", QUERY_WIDTH)
    if isinstance(prefetch_top_k, bool) or not isinstance(prefetch_top_k, int):
        raise TypeError("dsa_offload.prefetch_top_k must be an integer.")
    if not PREFETCH_TOP_K_MIN <= prefetch_top_k <= QUERY_WIDTH:
        raise ValueError(
            "dsa_offload.prefetch_top_k must be in "
            f"[{PREFETCH_TOP_K_MIN}, {QUERY_WIDTH}]."
        )
    enable_turbo_lookup = raw_config.get("enable_turbo_lookup", True)
    if not isinstance(enable_turbo_lookup, bool):
        raise TypeError("dsa_offload.enable_turbo_lookup must be a boolean.")
    enable_turbo_prefetch_lookup = raw_config.get(
        "enable_turbo_prefetch_lookup",
        True,
    )
    if not isinstance(enable_turbo_prefetch_lookup, bool):
        raise TypeError(
            "dsa_offload.enable_turbo_prefetch_lookup must be a boolean."
        )

    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if get_ascend_device_type() != AscendDeviceType.A5:
        raise ValueError("DSA Offload requires Ascend A5.")

    model_config = vllm_config.model_config
    if enable_prefetch and not bool(getattr(model_config, "enforce_eager", False)):
        raise ValueError("DSA Offload hidden-state prefetch requires eager execution.")
    hf_config = getattr(model_config, "hf_text_config", None)
    if hf_config is None:
        hf_config = model_config.hf_config
    if hf_config.model_type != "glm_moe_dsa":
        raise ValueError("DSA Offload supports only the GLM-5 family.")
    if hf_config.index_topk != QUERY_WIDTH:
        raise ValueError(f"DSA Offload requires index_topk={QUERY_WIDTH}.")
    cache_config = getattr(vllm_config, "cache_config", None)
    block_size = getattr(cache_config, "block_size", 0)
    max_supported_len = INDEX_CAPACITY + (
        block_size if isinstance(block_size, int) and block_size > 0 else 0
    )
    if model_config.max_model_len > max_supported_len:
        raise ValueError(
            "DSA Offload max_model_len must fit the index capacity plus one "
            f"tail block ({max_supported_len})."
        )

    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        # A connector-free engine owns both phases of every request. Remote
        # producer/consumer roles still require a connector to carry the P/D
        # handoff and the partial target-cache tail.
        kv_role: DSAOffloadRole = "kv_both"
    else:
        kv_connector = kv_transfer_config.kv_connector
        if kv_connector not in {"MooncakeConnectorV1", "LocalShmConnector"}:
            raise ValueError("DSA Offload supports only MooncakeConnectorV1 or LocalShmConnector.")
        kv_role = kv_transfer_config.kv_role
        if kv_role not in {"kv_producer", "kv_consumer", "kv_both"}:
            raise ValueError("DSA Offload requires kv_producer, kv_consumer, or kv_both.")
        if kv_connector == "LocalShmConnector" and kv_role == "kv_both":
            raise ValueError(
                "LocalShmConnector supports only separate kv_producer and "
                "kv_consumer engines; omit kv_transfer_config for local kv_both."
            )

        prefill = kv_transfer_config.get_from_extra_config("prefill", {})
        decode = kv_transfer_config.get_from_extra_config("decode", {})
        if "tp_size" not in prefill or "tp_size" not in decode:
            raise ValueError("DSA Offload connectors require prefill.tp_size and decode.tp_size.")
        if prefill["tp_size"] != decode["tp_size"]:
            raise ValueError("DSA Offload requires equal Prefill and Decode TP sizes.")

    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.pipeline_parallel_size != 1
        or parallel_config.prefill_context_parallel_size != 1
        or parallel_config.decode_context_parallel_size != 1
    ):
        raise ValueError("DSA Offload does not support PP, PCP, or DCP.")

    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        max_verify_tokens = 1
    else:
        if speculative_config.method != "mtp":
            raise ValueError("DSA Offload supports speculative decoding only with MTP.")
        max_verify_tokens = speculative_config.num_speculative_tokens + 1
        if max_verify_tokens > 16:
            raise ValueError("DSA Offload supports at most 16 verification tokens per request.")

    kvio_model_id = raw_config.get("kvio_model_id", 0)
    if kvio_model_id < 0:
        raise ValueError("dsa_offload.kvio_model_id must be non-negative.")

    return DSAOffloadConfig(
        io_backend=io_backend,
        kvio_model_id=kvio_model_id,
        max_verify_tokens_per_request=max_verify_tokens,
        kv_role=kv_role,
        kv_transfer_config=kv_transfer_config,
        enable_prefetch_with_hidden_states=enable_prefetch,
        prefetch_top_k=prefetch_top_k,
        enable_turbo_lookup=enable_turbo_lookup,
        enable_turbo_prefetch_lookup=enable_turbo_prefetch_lookup,
    )


def reserve_fixed_memory(available_bytes: int, fixed_bytes: int) -> int:
    remaining_bytes = available_bytes - fixed_bytes
    if remaining_bytes < 0:
        raise ValueError(
            f"DSA Offload fixed cache exceeds available KV memory: available={available_bytes}, fixed={fixed_bytes}."
        )
    return remaining_bytes
