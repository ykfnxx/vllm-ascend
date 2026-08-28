# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_commit_accepted_uses_valid_sampler_prefix_for_every_layer():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        dsa_sparse_config=SimpleNamespace(
            is_consumer=True,
            uses_mtp=True,
        )
    )
    runner.attn_state = AscendAttentionState.SpecDecoding
    leader = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
        query_positions=torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64),
        req_pool_entries=torch.tensor([4, 7], dtype=torch.int32),
    )
    first_layer = MagicMock()
    second_layer = MagicMock()
    runner._dsa_sparse_leader_coordinators = (leader,)
    runner._dsa_sparse_coordinators = (
        ("layer-0", first_layer),
        ("layer-1", second_layer),
    )
    runner.input_batch = SimpleNamespace(vocab_size=100)
    runner.discard_request_mask = SimpleNamespace(gpu=torch.tensor([False, True]))
    runner._dsa_sparse_target_step_id = 7
    sampled_token_ids = torch.tensor(
        [
            [10, 11, -1],
            [12, 100, -1],
        ],
        dtype=torch.int64,
    )

    runner._store_dsa_sparse_accepted(sampled_token_ids)

    for layer_name, coordinator in runner._dsa_sparse_coordinators:
        coordinator.commit_accepted_to_tail.assert_called_once()
        call = coordinator.commit_accepted_to_tail.call_args
        assert call.args[0] == layer_name
        assert call.args[1] == [0, 3, 5]
        assert call.args[2] == [10, 11, 12, 20, 21]
        assert call.args[3] == [2, 0]
        assert call.args[4] == [4, 7]
    assert runner._dsa_sparse_target_step_id == 8
