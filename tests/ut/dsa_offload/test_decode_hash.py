# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.dsa_offload.decode_hash import DecodeBlockHashState


def test_resolve_builds_missing_hash_from_context_and_worker_tokens() -> None:
    calls = []

    def block_hasher(parent_hash, token_ids, extra_keys):
        calls.append((parent_hash, list(token_ids), extra_keys))
        return b"generated"

    state = DecodeBlockHashState(4, block_hasher)
    state.update_contexts(
        {"request": (0, None, (10, 11), ("extra",))}
    )
    batch = SimpleNamespace(
        request_ids=("request",),
        query_ranges=((0, 3),),
        query_positions=torch.tensor([2, 3, 4]),
    )
    committed = {"request": []}

    block_hash = state.resolve(
        batch=batch,
        query_token_ids=torch.tensor([12, 13, 99]),
        committed_block_hashes=committed,
        request_index=0,
        logical_block=0,
    )

    assert block_hash == b"generated"
    assert committed == {"request": [b"generated"]}
    assert calls == [(None, [10, 11, 12, 13], ("extra",))]


def test_resolve_rejects_incomplete_worker_token_context() -> None:
    state = DecodeBlockHashState(4, lambda *_: b"unused")
    state.update_contexts({"request": (0, None, (10, 11), None)})
    batch = SimpleNamespace(
        request_ids=("request",),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([3]),
    )

    with pytest.raises(RuntimeError, match=r"missing_token_offsets=\[2\]"):
        state.resolve(
            batch=batch,
            query_token_ids=torch.tensor([13]),
            committed_block_hashes={"request": []},
            request_index=0,
            logical_block=0,
        )


def test_reconcile_keeps_worker_hash_until_scheduler_catches_up() -> None:
    state = DecodeBlockHashState(4, lambda *_: b"unused")

    assert state.reconcile(
        "request",
        [b"block-0", b"block-1"],
        [b"block-0"],
    ) == [b"block-0", b"block-1"]
    assert state.reconcile(
        "request",
        [b"block-0", b"block-1"],
        [b"block-0", b"block-1", b"block-2"],
    ) == [b"block-0", b"block-1", b"block-2"]

    with pytest.raises(RuntimeError, match="diverged from the scheduler"):
        state.reconcile(
            "request",
            [b"block-0", b"worker"],
            [b"block-0", b"scheduler"],
        )
