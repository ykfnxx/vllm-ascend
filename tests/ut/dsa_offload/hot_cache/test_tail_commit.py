# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.hot_cache import (
    HotCacheLayout,
    HotCacheState,
    commit_decode_tail,
    commit_mtp_tail,
    tail_commit_request_indices,
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
        query_end_positions=(position + 2 if is_mtp else position,),
    )
    return batch, cache


def test_decode_put_happens_only_when_tail_becomes_full(spy_io) -> None:
    partial, _ = make_batch(
        spy_io,
        position=2,
        is_mtp=False,
        committed=[b"\x01" * 32],
        candidate=[],
    )
    commit_decode_tail(partial)
    assert spy_io.put_calls == []

    full, _ = make_batch(
        spy_io,
        position=3,
        is_mtp=False,
        committed=[b"\x01" * 32],
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
    assert spy_io.put_calls[-1]["storage_ids"].tolist() == [
        make_storage_id(b"\x01" * 32, 6)
    ]
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
        lambda **kwargs: b"\x02" * 32,
    )

    assert spy_io.put_calls[-1]["storage_ids"].tolist() == [
        make_storage_id(b"\x02" * 32, 6)
    ]


@pytest.mark.parametrize("position", [3, 7, 11])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_mtp_puts_only_confirmed_block_without_moving_kv(spy_io, position, accepted) -> None:
    hashes = [bytes([index + 1]) * 32 for index in range(position // 4 + 1)]
    batch, cache = make_batch(
        spy_io,
        position=position,
        is_mtp=True,
        committed=hashes[:-1],
        candidate=hashes[-1:],
    )
    slots = cache.flatten(0, 1)
    slots[batch.layout.tail_slots(batch.query_positions), 0] = torch.tensor([11.0, 12.0, 99.0])
    before = cache.clone()
    original_put = spy_io.put_blocks

    def put_from_original_block(**kwargs):
        assert kwargs["source_block_ids"].tolist() == [batch.layout.tail_block(0, position // 4)]
        assert cache[kwargs["source_block_ids"][0], 3, 0] == 11
        original_put(**kwargs)

    spy_io.put_blocks = put_from_original_block
    commit_mtp_tail(batch, torch.tensor([accepted]))
    assert torch.equal(cache, before)
    assert len(spy_io.put_calls) == int(accepted > 0)
    if accepted:
        assert spy_io.put_calls[-1]["storage_ids"].tolist() == [make_storage_id(hashes[-1], 6)]


def test_mtp_boundary_uses_device_position_after_async_rejection(spy_io) -> None:
    batch, _ = make_batch(
        spy_io, position=7, is_mtp=True,
        committed=[b"\x01" * 32], candidate=[b"\x02" * 32],
    )
    batch.query_end_positions = (10,)
    batch.query_position_slack = 3
    assert tail_commit_request_indices(batch) == (0,)
    commit_mtp_tail(batch, torch.tensor([1]))
    assert spy_io.put_calls[-1]["source_block_ids"].tolist() == [batch.layout.tail_block(0, 1)]


def test_mtp_rejected_suffix_is_overwritten_on_retry(spy_io) -> None:
    batch, cache = make_batch(
        spy_io, position=3, is_mtp=True,
        committed=[], candidate=[b"\x01" * 32],
    )
    slots = cache.flatten(0, 1)
    slots[batch.layout.tail_slots(batch.query_positions), 0] = torch.tensor([11.0, 99.0, 99.0])
    commit_mtp_tail(batch, torch.tensor([1]))
    first_block = cache[batch.layout.tail_block(0, 0)].clone()
    batch.query_positions = torch.tensor([4, 5, 6])
    batch.query_end_positions = (6,)
    slots[batch.layout.tail_slots(batch.query_positions), 0] = torch.tensor([12.0, 13.0, 14.0])
    assert tail_commit_request_indices(batch) == ()
    commit_mtp_tail(batch, torch.tensor([2]))
    assert len(spy_io.put_calls) == 1
    assert torch.equal(cache[batch.layout.tail_block(0, 0)], first_block)
    assert cache[batch.layout.tail_block(0, 1), :2, 0].tolist() == [12.0, 13.0]
