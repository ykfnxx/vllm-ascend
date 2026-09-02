# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch

from vllm_ascend.dsa_offload.io import make_storage_id, make_storage_ids


def test_storage_id_is_stable_positive_int63() -> None:
    block_hash = bytes(range(32))

    storage_id = make_storage_id(block_hash, 7)

    assert storage_id == make_storage_id(block_hash, 7)
    assert 0 < storage_id < 1 << 63


def test_storage_id_uses_complete_hash_and_physical_layer() -> None:
    prefix = b"shared-prefix"

    assert make_storage_id(prefix + b"a", 3) != make_storage_id(prefix + b"b", 3)
    assert make_storage_id(prefix + b"a", 3) != make_storage_id(prefix + b"a", 4)


def test_storage_ids_builds_int64_tensor_on_requested_device() -> None:
    hashes = [b"block-0", b"block-1"]

    storage_ids = make_storage_ids(hashes, 2, device="cpu")

    assert storage_ids.dtype == torch.int64
    assert storage_ids.device.type == "cpu"
    assert storage_ids.tolist() == [make_storage_id(block_hash, 2) for block_hash in hashes]
