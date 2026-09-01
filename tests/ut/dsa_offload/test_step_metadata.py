# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest

from vllm_ascend.dsa_offload.metadata import apply_committed_update


def test_committed_update_appends_suffix_and_accepts_worker_overlap() -> None:
    committed = [11]
    apply_committed_update("request", committed, (1, (12,)))
    assert committed == [11, 12]

    worker_ahead = [11, 12]
    apply_committed_update("request", worker_ahead, (1, (12,)))
    assert worker_ahead == [11, 12]


def test_committed_update_rejects_gap_and_divergence() -> None:
    with pytest.raises(RuntimeError, match="has a gap"):
        apply_committed_update("request", [11], (2, (13,)))

    with pytest.raises(RuntimeError, match="diverged from the scheduler"):
        apply_committed_update("request", [11, 12], (1, (13,)))
