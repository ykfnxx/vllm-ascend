# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import torch

from vllm_ascend.dsa_offload.constants import QUERY_WIDTH
from vllm_ascend.dsa_offload.prefetch_coefficients import (
    PredictionCoefficientProfile,
    apply_group_predict_coefficients,
    get_active_prefetch_groups,
    get_group_predict_coefficients,
    get_prediction_coefficient_profile,
    pad_prefetch_topk,
)


def test_glm52_prefetch_topology_supports_complete_reduced_models() -> None:
    assert get_active_prefetch_groups(10) == {2: 6}
    assert get_active_prefetch_groups(14) == {2: 6, 6: 10}
    assert tuple(get_active_prefetch_groups(78)) == tuple(range(2, 71, 4))


def test_glm52_prefetch_profile_and_cp16_coefficients() -> None:
    quant_config = SimpleNamespace(
        quant_description={
            "model_quant_type": "W8A8_DYNAMIC",
            "model.layers.2.mlp.experts.weight": "W4A8_DYNAMIC",
        }
    )
    profile = get_prediction_coefficient_profile(quant_config)

    assert profile is PredictionCoefficientProfile.GLM52_W4A8
    alpha, beta = get_group_predict_coefficients(profile, 2)
    assert len(alpha) == len(beta) == 16
    assert alpha[:2] == (1.426778885800295, 0.037143112479008)
    assert beta[:2] == (-0.0014946481492795387, 0.0017293299924049557)

    w8_alpha, w8_beta = get_group_predict_coefficients(
        PredictionCoefficientProfile.GLM52_W8A8,
        2,
    )
    assert w8_alpha == (0.7627772709274597, *((1.0,) * 15))
    assert w8_beta == (-0.0008454544838177624, *((0.0,) * 15))


def test_group_coefficients_and_reduced_topk_keep_runtime_shapes() -> None:
    hidden_states = torch.ones((5, 2), dtype=torch.float32)
    predicted = apply_group_predict_coefficients(
        hidden_states,
        torch.tensor([2.0, 3.0]),
        torch.tensor([0.0, 1.0]),
        tp_rank=0,
        source_rows_before_gather=2,
    )
    topk = torch.arange(8, dtype=torch.int32).view(2, 4)
    padded = pad_prefetch_topk(topk, 4)

    expected = torch.tensor([[2.0], [2.0], [4.0], [4.0], [4.0]])
    assert torch.equal(predicted, expected.expand_as(hidden_states))
    assert padded.shape == (2, QUERY_WIDTH)
    assert torch.equal(padded[:, :4], topk)
    assert padded[:, 4:].eq(-1).all()
