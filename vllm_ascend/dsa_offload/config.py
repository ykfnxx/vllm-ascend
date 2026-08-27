# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass
from typing import Any, Literal

from .constants import INDEX_CAPACITY, QUERY_WIDTH

DSAOffloadRole = Literal["kv_producer", "kv_consumer", "kv_both"]


@dataclass(frozen=True)
class DSAOffloadConfig:
    io_backend: Literal["kvio", "mock"]
    kvio_model_id: int
    max_verify_tokens_per_request: int
    kv_transfer_config: Any

    @property
    def kv_role(self) -> DSAOffloadRole:
        return self.kv_transfer_config.kv_role

    @property
    def is_producer(self) -> bool:
        return self.kv_transfer_config.is_kv_producer

    @property
    def is_consumer(self) -> bool:
        return self.kv_transfer_config.is_kv_consumer


def load_dsa_offload_config(vllm_config: object) -> DSAOffloadConfig | None:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict) or "dsa_offload" not in additional_config:
        return None

    raw_config = additional_config["dsa_offload"]
    io_backend = raw_config["io_backend"]
    if io_backend not in {"kvio", "mock"}:
        raise ValueError("dsa_offload.io_backend must be 'kvio' or 'mock'.")

    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if get_ascend_device_type() != AscendDeviceType.A5:
        raise ValueError("DSA Offload requires Ascend A5.")

    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_text_config", None)
    if hf_config is None:
        hf_config = model_config.hf_config
    if hf_config.model_type != "glm_moe_dsa":
        raise ValueError("DSA Offload supports only the GLM-5 family.")
    if hf_config.index_topk != QUERY_WIDTH:
        raise ValueError(f"DSA Offload requires index_topk={QUERY_WIDTH}.")
    if model_config.max_model_len > INDEX_CAPACITY:
        raise ValueError(f"DSA Offload max_model_len must not exceed {INDEX_CAPACITY}.")

    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or kv_transfer_config.kv_connector != "MooncakeConnectorV1":
        raise ValueError("DSA Offload requires MooncakeConnectorV1.")
    kv_role = kv_transfer_config.kv_role
    if kv_role not in {"kv_producer", "kv_consumer", "kv_both"}:
        raise ValueError("DSA Offload requires kv_producer, kv_consumer, or kv_both.")

    prefill = kv_transfer_config.get_from_extra_config("prefill", {})
    decode = kv_transfer_config.get_from_extra_config("decode", {})
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
        kv_transfer_config=kv_transfer_config,
    )


def reserve_fixed_memory(available_bytes: int, fixed_bytes: int) -> int:
    remaining_bytes = available_bytes - fixed_bytes
    if remaining_bytes < 0:
        raise ValueError(
            f"DSA Offload fixed cache exceeds available KV memory: available={available_bytes}, fixed={fixed_bytes}."
        )
    return remaining_bytes
