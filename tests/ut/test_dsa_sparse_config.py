# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest

from vllm_ascend.dsa_sparse_config import load_dsa_sparse_config


def make_vllm_config(
    *,
    io_backend: str = "mock",
    role: str = "kv_consumer",
    enforce_eager: bool = True,
    device_buffer_size: int | None = 32,
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    num_speculative_tokens: int = 0,
    model_type: str = "glm_moe_dsa",
    index_topk: int = 8,
):
    dsa_config = {
        "io_backend": io_backend,
        "io_backend_options": {"namespace": "test"},
    }
    if device_buffer_size is not None:
        dsa_config["device_buffer_size"] = device_buffer_size
    return SimpleNamespace(
        additional_config={"dsa_sparse_config": dsa_config},
        model_config=SimpleNamespace(
            enforce_eager=enforce_eager,
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                index_topk=index_topk,
            ),
        ),
        kv_transfer_config=SimpleNamespace(kv_role=role),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
        ),
    )


def test_absent_config_keeps_dsa_sparse_disabled():
    vllm_config = SimpleNamespace(additional_config={})

    assert load_dsa_sparse_config(vllm_config) is None


def test_consumer_config_is_immutable_and_derives_runtime_shape():
    config = load_dsa_sparse_config(make_vllm_config())

    assert config is not None
    assert config.is_consumer
    assert not config.is_producer
    assert config.max_query_tokens_per_request == 1
    assert config.index_topk == 8
    assert config.device_buffer_size == 32
    with pytest.raises(TypeError):
        config.io_backend_options["namespace"] = "changed"


def test_producer_keeps_full_cache_and_has_no_device_buffer():
    config = load_dsa_sparse_config(
        make_vllm_config(
            role="kv_producer",
            device_buffer_size=None,
            num_speculative_tokens=3,
        )
    )

    assert config is not None
    assert config.is_producer
    assert config.device_buffer_size is None
    assert config.max_query_tokens_per_request == 1


@pytest.mark.parametrize("role", [None, "kv_both"])
def test_only_pd_roles_are_supported(role):
    with pytest.raises(ValueError, match="P/D-only role"):
        load_dsa_sparse_config(make_vllm_config(role=role))


def test_graph_execution_is_rejected_in_eager_milestone():
    with pytest.raises(ValueError, match="only the eager execution path"):
        load_dsa_sparse_config(make_vllm_config(enforce_eager=False))


def test_graph_out_milestone_rejects_concrete_io_backend():
    with pytest.raises(ValueError, match="only io_backend='mock'"):
        load_dsa_sparse_config(
            make_vllm_config(io_backend="vendor"),
        )


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("pipeline_parallel_size", {"pp": 2}),
        ("decode_context_parallel_size", {"dcp": 2}),
        ("prefill_context_parallel_size", {"pcp": 2}),
    ],
)
def test_unsupported_parallel_modes_fail_fast(field, overrides):
    with pytest.raises(ValueError, match=rf"{field}=1"):
        load_dsa_sparse_config(make_vllm_config(**overrides))


def test_tp_is_not_restricted_by_core_config():
    vllm_config = make_vllm_config()
    vllm_config.parallel_config.tensor_parallel_size = 16

    assert load_dsa_sparse_config(vllm_config) is not None


@pytest.mark.parametrize("device_buffer_size", [None, 0, 7])
def test_consumer_requires_complete_topk_union(device_buffer_size):
    with pytest.raises(ValueError, match="device_buffer_size"):
        load_dsa_sparse_config(make_vllm_config(device_buffer_size=device_buffer_size))


def test_producer_rejects_decode_hot_cache_size():
    with pytest.raises(ValueError, match="Decode-only"):
        load_dsa_sparse_config(
            make_vllm_config(
                role="kv_producer",
                device_buffer_size=32,
            )
        )


@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3])
def test_speculative_decode_waits_for_draft_hot_cache_runtime(
    num_speculative_tokens,
):
    with pytest.raises(ValueError, match="target decode only"):
        load_dsa_sparse_config(
            make_vllm_config(
                num_speculative_tokens=num_speculative_tokens,
            )
        )


def test_only_glm5_sparse_model_is_supported():
    with pytest.raises(ValueError, match="glm_moe_dsa"):
        load_dsa_sparse_config(make_vllm_config(model_type="deepseek_v3"))
