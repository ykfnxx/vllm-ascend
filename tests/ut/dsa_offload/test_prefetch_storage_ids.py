# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.dsa_offload import prefetch
from vllm_ascend.dsa_offload.io import make_storage_ids
from vllm_ascend.dsa_offload.lookup import DSAOffloadBatch


def _hashes(values):
    return tuple(value.encode() for value in values)


def _runtime():
    runtime = object.__new__(prefetch.GroupedPrefetchRuntime)
    runtime.storage_ids = {
        layer: torch.full((2, 8), -1, dtype=torch.int64) for layer in (6, 7)
    }
    runtime._row_hashes = [None, None]
    return runtime


def _batch(committed, candidate=()):
    batch = object.__new__(DSAOffloadBatch)
    batch.request_ids = ("request",)
    batch.decode_request_indices = (0,)
    batch.hot_cache = SimpleNamespace(request_to_row={"request": 0})
    batch.committed_block_hashes = {"request": list(_hashes(committed))}
    batch.candidate_block_hashes = {"request": list(_hashes(candidate))}
    return batch


@pytest.mark.parametrize(
    "previous,committed,candidate,changed",
    [
        (None, "AB", "", ["AB"]),
        ("AB", "AB", "", []),
        ("AB", "ABCD", "", ["CD"]),
        ("ABC", "AB", "D", ["D"]),
        ("ABCD", "ABC", "D", []),  # Candidate promotion.
        ("ABCD", "ABC", "E", ["E"]),
        ("ABCD", "AB", "E", ["E"]),
        ("ABCD", "AB", "", []),  # Rejected candidate tail.
        ("ABCD", "XBCY", "", ["X", "Y"]),  # Snapshot, separate ranges.
        ("ABCD", "AXYD", "", ["XY"]),  # Adjacent changes are merged.
        ("AB", "", "", []),
        (None, "", "", []),
    ],
)
def test_storage_ids_update_only_changed_ranges(previous, committed, candidate, changed):
    runtime = _runtime()
    if previous is not None:
        runtime.update_storage_ids(_batch(previous))
    addresses = [table.data_ptr() for table in runtime.storage_ids.values()]

    with patch.object(prefetch, "make_storage_ids", wraps=make_storage_ids) as pack:
        runtime.update_storage_ids(_batch(committed, candidate))

    assert [(call.args[1], call.args[0]) for call in pack.call_args_list] == [
        (layer, _hashes(values)) for layer in runtime.storage_ids for values in changed
    ]
    effective = _hashes(committed + candidate)
    for layer, table in runtime.storage_ids.items():
        expected = torch.full_like(table, -1)
        expected[0, : len(effective)] = make_storage_ids(effective, layer, device="cpu")
        assert torch.equal(table, expected)
    assert runtime._row_hashes[0] == effective
    assert addresses == [table.data_ptr() for table in runtime.storage_ids.values()]


def test_storage_ids_local_confirmation_and_row_reuse():
    runtime = _runtime()
    batch = _batch("AB", "C")
    runtime.update_storage_ids(batch)
    # Local confirmation followed by scheduler confirmation leaves the same
    # effective sequence, so neither step should repackage the IDs.
    batch.committed_block_hashes["request"].append(b"C")
    batch.candidate_block_hashes.clear()
    with patch.object(prefetch, "make_storage_ids", wraps=make_storage_ids) as pack:
        runtime.update_storage_ids(batch)
        runtime.update_storage_ids(_batch("ABC"))
        pack.assert_not_called()

    runtime.clear_request_row(0)
    assert runtime._row_hashes[0] is None
    assert all(table.eq(-1).all() for table in runtime.storage_ids.values())
    with patch.object(prefetch, "make_storage_ids", wraps=make_storage_ids) as pack:
        runtime.update_storage_ids(_batch("ABC"))
        assert pack.call_count == len(runtime.storage_ids)
