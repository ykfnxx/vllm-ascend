# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import sys
import types
from types import SimpleNamespace

import pytest

from vllm_ascend.dsa_offload.config import (
    load_dsa_offload_config,
    reserve_fixed_memory,
)


class TransferConfig:
    kv_connector = "MooncakeConnectorV1"
    kv_role = "kv_both"

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_role in {"kv_producer", "kv_both"}

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_role in {"kv_consumer", "kv_both"}

    def get_from_extra_config(self, key, default):
        return {"tp_size": 4}


def make_config(**overrides):
    config = SimpleNamespace(
        additional_config={"dsa_offload": {"io_backend": "mock", "kvio_model_id": 9}},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa", index_topk=2048),
            max_model_len=128 * 1024,
        ),
        kv_transfer_config=TransferConfig(),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=7),
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


@pytest.fixture(autouse=True)
def a5_device(monkeypatch):
    utils = types.ModuleType("vllm_ascend.utils")
    utils.AscendDeviceType = SimpleNamespace(A5="a5")
    utils.get_ascend_device_type = lambda: "a5"
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)


def test_feature_gate_and_valid_config() -> None:
    disabled = make_config(additional_config={})
    assert load_dsa_offload_config(disabled) is None

    vllm_config = make_config()
    config = load_dsa_offload_config(vllm_config)

    assert config is not None
    assert config.io_backend == "mock"
    assert config.kvio_model_id == 9
    assert config.max_verify_tokens_per_request == 8
    assert config.kv_transfer_config is vllm_config.kv_transfer_config
    assert config.has_connector
    assert config.kv_role == "kv_both"
    assert config.is_producer and config.is_consumer


def test_connector_free_config_uses_local_mixed_role() -> None:
    config = load_dsa_offload_config(make_config(kv_transfer_config=None))

    assert config is not None
    assert not config.has_connector
    assert config.kv_role == "kv_both"
    assert config.is_producer and config.is_consumer


def test_io_backend_defaults_to_mock() -> None:
    vllm_config = make_config()
    vllm_config.additional_config = {"dsa_offload": {}}

    config = load_dsa_offload_config(vllm_config)

    assert config is not None
    assert config.io_backend == "mock"


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_local_shm_connector_is_supported_for_split_pd(role: str) -> None:
    vllm_config = make_config()
    vllm_config.kv_transfer_config.kv_connector = "LocalShmConnector"
    vllm_config.kv_transfer_config.kv_role = role

    config = load_dsa_offload_config(vllm_config)

    assert config is not None
    assert config.has_connector
    assert config.kv_role == role


def test_local_shm_connector_rejects_mixed_role() -> None:
    vllm_config = make_config()
    vllm_config.kv_transfer_config.kv_connector = "LocalShmConnector"

    with pytest.raises(ValueError, match="omit kv_transfer_config"):
        load_dsa_offload_config(vllm_config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda config: config.additional_config["dsa_offload"].update(io_backend="memory"),
            "io_backend",
        ),
        (
            lambda config: setattr(config.model_config.hf_text_config, "index_topk", 1024),
            "index_topk=2048",
        ),
        (
            lambda config: setattr(config.kv_transfer_config, "kv_connector", "MooncakeLayerwiseConnector"),
            "MooncakeConnectorV1",
        ),
        (
            lambda config: setattr(config.parallel_config, "pipeline_parallel_size", 2),
            "PP, PCP, or DCP",
        ),
        (
            lambda config: setattr(config.speculative_config, "method", "eagle"),
            "only with MTP",
        ),
        (
            lambda config: setattr(config.speculative_config, "num_speculative_tokens", 16),
            "at most 16",
        ),
    ],
)
def test_fixed_configuration_contract(change, message) -> None:
    config = make_config()
    change(config)

    with pytest.raises(ValueError, match=message):
        load_dsa_offload_config(config)


def test_fixed_memory_reservation() -> None:
    assert reserve_fixed_memory(100, 40) == 60
    with pytest.raises(ValueError, match="exceeds available"):
        reserve_fixed_memory(39, 40)
