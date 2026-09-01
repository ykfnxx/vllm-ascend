# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch

from vllm_ascend.dsa_offload.io import make_storage_id, make_storage_ids
from vllm_ascend.dsa_offload.metadata import make_block_key


def test_block_key_is_stable_positive_int63() -> None:
    canonical_hash = bytes(range(32))

    block_key = make_block_key(canonical_hash)

    assert block_key == 718236936765396657
    assert block_key == make_block_key(canonical_hash)
    assert block_key != make_block_key(canonical_hash + b"next")
    assert 0 < block_key < 1 << 63


def test_storage_id_is_stable_positive_int63() -> None:
    block_key = make_block_key(bytes(range(32)))

    storage_id = make_storage_id(block_key, 7)

    assert storage_id == 203820441002039781
    assert storage_id == make_storage_id(block_key, 7)
    assert 0 < storage_id < 1 << 63


def test_storage_id_uses_block_key_and_physical_layer() -> None:
    first = make_block_key(b"first")
    second = make_block_key(b"second")

    assert make_storage_id(first, 3) != make_storage_id(second, 3)
    assert make_storage_id(first, 3) != make_storage_id(first, 4)


def test_storage_ids_builds_int64_tensor_on_requested_device() -> None:
    block_keys = [make_block_key(b"block-0"), make_block_key(b"block-1")]

    storage_ids = make_storage_ids(block_keys, 2, device="cpu")

    assert storage_ids.dtype == torch.int64
    assert storage_ids.device.type == "cpu"
    assert storage_ids.tolist() == [
        make_storage_id(block_key, 2) for block_key in block_keys
    ]
