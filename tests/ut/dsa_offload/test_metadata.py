# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest

from vllm_ascend.dsa_offload.metadata import (
    BlockHashUpdate,
    apply_block_hash_update,
)


def test_apply_block_hash_update() -> None:
    committed = [b"stale"]
    apply_block_hash_update(
        "request",
        committed,
        BlockHashUpdate(0, (b"block-0",), replace=True),
    )
    assert committed == [b"block-0"]

    committed.append(b"block-1")
    apply_block_hash_update(
        "request",
        committed,
        BlockHashUpdate(1, (b"block-1", b"block-2")),
    )
    assert committed == [b"block-0", b"block-1", b"block-2"]

    with pytest.raises(RuntimeError, match="has a gap"):
        apply_block_hash_update(
            "request",
            [b"block-0"],
            BlockHashUpdate(2, (b"block-2",)),
        )

    with pytest.raises(RuntimeError, match="diverged from the scheduler"):
        apply_block_hash_update(
            "request",
            [b"block-0", b"worker"],
            BlockHashUpdate(1, (b"scheduler",)),
        )
