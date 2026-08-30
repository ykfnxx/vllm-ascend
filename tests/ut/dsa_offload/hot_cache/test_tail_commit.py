# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.hot_cache import (
    HotCacheLayout,
    HotCacheState,
    commit_decode_tail,
    commit_mtp_tail,
)
from vllm_ascend.dsa_offload.io import make_storage_id
from vllm_ascend.dsa_offload.lookup import DSAOffloadBatch, IndexCacheCohort


def make_batch(spy_io, *, position: int, is_mtp: bool, committed, candidate):
    layout = HotCacheLayout(4, 1, 3)
    cache = torch.zeros((layout.hot_blocks, 4, 1))
    hot_cache = HotCacheState(layout, {"layer": (cache,)})
    row = hot_cache.admit("request")
    cohort = IndexCacheCohort("layer", "layer", ("layer",), (6,))
    batch = DSAOffloadBatch(
        layout=layout,
        hot_cache=hot_cache,
        io_backend=spy_io,
        cohorts=(cohort,),
        lookup_states={},
        request_ids=("request",),
        request_rows=torch.tensor([row], dtype=torch.int32),
        decode_request_indices=(0,),
        query_ranges=((0, 3 if is_mtp else 1),),
        query_positions=torch.tensor(
            [position, position + 1, position + 2] if is_mtp else [position],
            dtype=torch.int64,
        ),
        is_mtp=is_mtp,
        committed_block_hashes={"request": committed},
        candidate_block_hashes={"request": candidate},
    )
    return batch, cache


def test_decode_put_happens_only_when_tail_becomes_full(spy_io) -> None:
    partial, _ = make_batch(
        spy_io,
        position=2,
        is_mtp=False,
        committed=[b"block-0"],
        candidate=[],
    )
    commit_decode_tail(partial)
    assert spy_io.put_calls == []

    full, _ = make_batch(
        spy_io,
        position=3,
        is_mtp=False,
        committed=[b"block-0"],
        candidate=[],
    )
    events = ["model"]
    original_put = spy_io.put_blocks

    def ordered_put(**kwargs):
        events.append("put")
        original_put(**kwargs)

    spy_io.put_blocks = ordered_put
    commit_decode_tail(full)
    assert events == ["model", "put"]
    assert spy_io.put_calls[-1]["storage_ids"].tolist() == [make_storage_id(b"block-0", 6)]
    assert spy_io.put_calls[-1]["source_block_ids"].tolist() == [full.layout.tail_block_offset]


def test_decode_tail_commit_rejects_missing_block_hash(spy_io) -> None:
    batch, _ = make_batch(
        spy_io,
        position=3,
        is_mtp=False,
        committed=[],
        candidate=[],
    )

    with pytest.raises(
        RuntimeError,
        match=r"Decode tail commit for request request requires 1 block hashes",
    ):
        commit_decode_tail(batch)


def test_decode_tail_commit_uses_worker_hash_resolver(spy_io) -> None:
    batch, _ = make_batch(
        spy_io,
        position=3,
        is_mtp=False,
        committed=[],
        candidate=[],
    )

    commit_decode_tail(
        batch,
        lambda **kwargs: b"resolved-0",
    )

    assert spy_io.put_calls[-1]["storage_ids"].tolist() == [
        make_storage_id(b"resolved-0", 6)
    ]


def test_mtp_commits_only_accepted_prefix_then_puts_candidate_block(spy_io) -> None:
    batch, cache = make_batch(
        spy_io,
        position=3,
        is_mtp=True,
        committed=[],
        candidate=[b"candidate-0"],
    )
    slots = cache.flatten(0, 1)
    slots[batch.layout.staging_base : batch.layout.staging_base + 3, 0] = torch.tensor([11.0, 12.0, 99.0])
    events = ["accepted"]
    original_put = spy_io.put_blocks

    def put_after_copy(**kwargs):
        assert slots[batch.layout.tail_base + 3, 0].item() == 11
        events.append("put")
        original_put(**kwargs)

    spy_io.put_blocks = put_after_copy

    commit_mtp_tail(batch, [2])

    assert events == ["accepted", "put"]
    assert slots[batch.layout.tail_base + 3, 0].item() == 11
    assert slots[batch.layout.tail_base, 0].item() == 12
    assert slots[batch.layout.tail_base + 1, 0].item() == 0
    assert spy_io.put_calls[-1]["storage_ids"].tolist() == [make_storage_id(b"candidate-0", 6)]
