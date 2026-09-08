# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest

from vllm_ascend.dsa_offload.metadata import BlockHashUpdate, apply_block_hash_update


def test_block_hash_update_appends_suffix_and_accepts_worker_overlap() -> None:
    committed = [11]
    apply_block_hash_update("request", committed, BlockHashUpdate(base_count=1, hashes=(12,)))
    assert committed == [11, 12]

    worker_ahead = [11, 12]
    apply_block_hash_update("request", worker_ahead, BlockHashUpdate(base_count=1, hashes=(12,)))
    assert worker_ahead == [11, 12]


def test_block_hash_update_rejects_gap_and_divergence() -> None:
    with pytest.raises(RuntimeError, match="has a gap"):
        apply_block_hash_update("request", [11], BlockHashUpdate(base_count=2, hashes=(13,)))

    with pytest.raises(RuntimeError, match="diverged from the scheduler"):
        apply_block_hash_update("request", [11, 12], BlockHashUpdate(base_count=1, hashes=(13,)))
