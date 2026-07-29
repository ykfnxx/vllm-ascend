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
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    num_speculative_tokens: int = 0,
    model_type: str = "glm_moe_dsa",
    index_topk: int = 2048,
    max_model_len: int = 4096,
    block_size: int = 128,
    extra_dsa_fields: dict | None = None,
):
    dsa_config = {
        "io_backend": io_backend,
        "io_backend_options": {"namespace": "test"},
        **(extra_dsa_fields or {}),
    }
    return SimpleNamespace(
        additional_config={"dsa_sparse_config": dsa_config},
        model_config=SimpleNamespace(
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                index_topk=index_topk,
            ),
        ),
        cache_config=SimpleNamespace(block_size=block_size),
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
    assert load_dsa_sparse_config(
        SimpleNamespace(additional_config={})
    ) is None


def test_config_exposes_only_fixed_lookup_contract_inputs():
    config = load_dsa_sparse_config(make_vllm_config())

    assert config is not None
    assert config.is_consumer
    assert config.index_topk == 2048
    assert not hasattr(config, "device_buffer_size")
    assert not hasattr(config, "max_query_tokens_per_request")
    with pytest.raises(TypeError):
        config.io_backend_options["namespace"] = "changed"


def test_device_buffer_size_is_removed_from_user_configuration():
    with pytest.raises(ValueError, match="Unknown"):
        load_dsa_sparse_config(
            make_vllm_config(
                extra_dsa_fields={"device_buffer_size": 8192}
            )
        )


@pytest.mark.parametrize("role", [None, "kv_both"])
def test_only_pd_roles_are_supported(role):
    with pytest.raises(ValueError, match="P/D-only role"):
        load_dsa_sparse_config(make_vllm_config(role=role))


def test_graph_execution_is_rejected():
    with pytest.raises(ValueError, match="only the eager execution path"):
        load_dsa_sparse_config(make_vllm_config(enforce_eager=False))


def test_only_mock_io_is_supported():
    with pytest.raises(ValueError, match="only io_backend='mock'"):
        load_dsa_sparse_config(make_vllm_config(io_backend="vendor"))


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("pipeline_parallel_size", {"pp": 2}),
        ("decode_context_parallel_size", {"dcp": 2}),
        ("prefill_context_parallel_size", {"pcp": 2}),
    ],
)
def test_unsupported_parallel_modes_fail(field, overrides):
    with pytest.raises(ValueError, match=rf"{field}=1"):
        load_dsa_sparse_config(make_vllm_config(**overrides))


@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3])
def test_speculative_decode_is_rejected(num_speculative_tokens):
    with pytest.raises(ValueError, match="target decode only"):
        load_dsa_sparse_config(
            make_vllm_config(
                num_speculative_tokens=num_speculative_tokens
            )
        )


def test_index_topk_is_fixed_to_2048():
    with pytest.raises(ValueError, match="index_topk=2048"):
        load_dsa_sparse_config(make_vllm_config(index_topk=1024))


def test_model_length_must_fit_128k_index():
    with pytest.raises(ValueError, match="128K"):
        load_dsa_sparse_config(
            make_vllm_config(max_model_len=128 * 1024 + 1)
        )


def test_block_size_must_partition_lookup_regions():
    with pytest.raises(ValueError, match="block_size"):
        load_dsa_sparse_config(make_vllm_config(block_size=192))


def test_only_glm5_sparse_model_is_supported():
    with pytest.raises(ValueError, match="glm_moe_dsa"):
        load_dsa_sparse_config(
            make_vllm_config(model_type="deepseek_v3")
        )
