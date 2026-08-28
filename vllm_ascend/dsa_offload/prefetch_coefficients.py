# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from enum import Enum

import torch

from .constants import QUERY_WIDTH

PREFETCH_GROUP_SIZE = 4
PREFETCH_FIRST_SOURCE_LAYER = 2
PREFETCH_MIN_MODEL_LAYERS = 10
PREFETCH_FULL_MODEL_LAYERS = 78
PREFETCH_HI_BLOCK_NUM = 128

PREFETCH_GROUPS = {
    source_layer: source_layer + PREFETCH_GROUP_SIZE
    for source_layer in range(2, 71, PREFETCH_GROUP_SIZE)
}

_GLM52_CP16_RANK0_COEFFICIENTS = {
    2: (0.7534633881797883, -2.920227705064452e-04),
    6: (0.4499160381418932, -1.955804766109186e-04),
    10: (1.6020963736094129, 7.208523719933957e-05),
    14: (0.8304060356565585, -1.322604365853837e-04),
    18: (1.6038112812692056, 1.196267575676426e-03),
    22: (0.4419906564740327, -6.541664564005124e-04),
    26: (0.9371220941284848, -7.473514107508483e-04),
    30: (0.6124603104839230, 1.007539321950684e-06),
    34: (0.7465077245054130, -2.728459027765521e-03),
    38: (0.6274350133401467, 9.492132727839263e-06),
    42: (0.8155560462553153, -7.210319026522409e-04),
    46: (0.8539307745847096, 1.520687708528885e-05),
    50: (0.8590300601050843, 3.122042472507400e-04),
    54: (0.8956555880140493, -9.653218853237830e-04),
    58: (0.8125425466498852, 2.634971726632664e-06),
    62: (1.9782859302617941, 2.304399797461859e-04),
    66: (0.2036059106430181, -5.008542623083366e-04),
    70: (1.9192340944289839, 2.502599919443691e-04),
}


class PredictionCoefficientProfile(Enum):
    GLM52_W4A8 = "glm52_w4a8"
    GLM52_W8A8 = "glm52_w8a8"


def get_active_prefetch_groups(num_hidden_layers: int) -> dict[int, int]:
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int):
        raise ValueError("Grouped prefetch requires integer num_hidden_layers.")
    if not PREFETCH_MIN_MODEL_LAYERS <= num_hidden_layers <= PREFETCH_FULL_MODEL_LAYERS:
        raise ValueError(
            "Grouped prefetch supports num_hidden_layers in "
            f"[{PREFETCH_MIN_MODEL_LAYERS}, {PREFETCH_FULL_MODEL_LAYERS}]."
        )
    if (num_hidden_layers - PREFETCH_FIRST_SOURCE_LAYER) % PREFETCH_GROUP_SIZE:
        raise ValueError(
            "Grouped prefetch requires two standalone layers followed by "
            "complete four-layer cohorts."
        )
    return {
        source: target
        for source, target in PREFETCH_GROUPS.items()
        if target + PREFETCH_GROUP_SIZE <= num_hidden_layers
    }


def get_prediction_coefficient_profile(
    quant_config: object | None,
) -> PredictionCoefficientProfile | None:
    quant_description = getattr(quant_config, "quant_description", None)
    if not isinstance(quant_description, dict):
        return None
    quant_types = {
        value.upper()
        for value in quant_description.values()
        if isinstance(value, str)
    }
    if any(quant_type.startswith("W4A8") for quant_type in quant_types):
        return PredictionCoefficientProfile.GLM52_W4A8
    if any(quant_type.startswith("W8A8") for quant_type in quant_types):
        return PredictionCoefficientProfile.GLM52_W8A8
    return None


def get_group_predict_coefficients(
    profile: PredictionCoefficientProfile,
    source_layer_id: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if profile not in {
        PredictionCoefficientProfile.GLM52_W4A8,
        PredictionCoefficientProfile.GLM52_W8A8,
    }:
        raise ValueError(f"Unsupported prediction coefficient profile: {profile!r}.")
    try:
        rank0_alpha, rank0_beta = _GLM52_CP16_RANK0_COEFFICIENTS[
            source_layer_id
        ]
    except KeyError as error:
        raise ValueError(
            f"Missing grouped prefetch coefficients for source layer {source_layer_id}."
        ) from error
    return (
        (rank0_alpha, *((1.0,) * 15)),
        (rank0_beta, *((0.0,) * 15)),
    )


def apply_group_predict_coefficients(
    hidden_states: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    tp_rank: int,
    source_rows_before_gather: int | None,
) -> torch.Tensor:
    if source_rows_before_gather is None:
        return hidden_states * alpha[tp_rank] + beta[tp_rank]
    row_rank = torch.div(
        torch.arange(
            hidden_states.shape[0],
            dtype=torch.int64,
            device=hidden_states.device,
        ),
        source_rows_before_gather,
        rounding_mode="floor",
    ).clamp_max(alpha.shape[0] - 1)
    return (
        hidden_states * alpha[row_rank].unsqueeze(1)
        + beta[row_rank].unsqueeze(1)
    )


def pad_prefetch_topk(
    topk_indices: torch.Tensor,
    prefetch_top_k: int,
) -> torch.Tensor:
    padding_count = QUERY_WIDTH - prefetch_top_k
    if padding_count == 0:
        return topk_indices
    padding = torch.full(
        (*topk_indices.shape[:-1], padding_count),
        -1,
        dtype=topk_indices.dtype,
        device=topk_indices.device,
    )
    return torch.cat((topk_indices, padding), dim=-1)
