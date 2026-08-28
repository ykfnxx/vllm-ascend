# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch

from vllm_ascend.dsa_sparse_backend import (
    DSASparseStorageKeyEncoder,
    KVIODSASparseKVBackend,
)


class FakeKVIOOps:
    def __init__(self) -> None:
        self.init_args = None
        self.calls = []
        self.waits = []

    def aiv_init(self, addresses, lengths):
        self.init_args = (addresses, lengths)
        return 0

    def npu_get_put_batch(self, *args):
        self.calls.append(tuple(arg.clone() for arg in args))

    def npu_send_wait(self, *args):
        self.waits.append(tuple(arg.clone() for arg in args))


def _values(call):
    return tuple(arg.tolist() for arg in call)


def _make_backend():
    ops = FakeKVIOOps()
    backend = KVIODSASparseKVBackend(
        7,
        ops_module=ops,
        tensor_ops=ops,
    )
    nope = torch.zeros((4, 2, 1, 3), dtype=torch.float32)
    rope = torch.zeros((4, 2, 1, 1), dtype=torch.float32)
    backend.register_layer_cache(
        layer_id=3,
        block_size=2,
        cache_planes=(nope, rope),
    )
    backend.finalize_cache_registration()
    return backend, ops, nope, rope


def test_storage_key_uses_full_hash_and_physical_layer():
    encoder = DSASparseStorageKeyEncoder()
    first = encoder.encode(b"block-hash", 3)

    assert first == encoder.encode(b"block-hash", 3)
    assert first != encoder.encode(b"block-hash", 4)
    assert first != encoder.encode(b"block-hash-other", 3)
    assert 0 < first < 1 << 63


def test_kvio_registers_two_planes_and_translates_put_get_descriptors():
    backend, ops, nope, rope = _make_backend()

    assert ops.init_args == (
        [nope.data_ptr(), rope.data_ptr()],
        [nope.numel() * nope.element_size(), rope.numel() * rope.element_size()],
    )

    backend.put_blocks(
        layer_id=3,
        storage_request_ids=torch.tensor([41], dtype=torch.int64),
        source_block_ids=torch.tensor([2], dtype=torch.int64),
    )
    backend.load_tokens_into(
        layer_id=3,
        storage_request_ids=torch.tensor([41], dtype=torch.int64),
        token_offsets_in_block=torch.tensor([1], dtype=torch.int64),
        destination_physical_slots=torch.tensor([4], dtype=torch.int64),
    )

    assert _values(ops.calls[0]) == (
        [1],
        [7],
        [1],
        [2],
        [0x05],
        [0, 1],
        [41, 41],
        [48, 16],
        [0, 24],
        [24, 8],
    )
    assert _values(ops.calls[1]) == (
        [2],
        [7],
        [1],
        [2],
        [0x06],
        [0, 1],
        [41, 41],
        [48, 16],
        [12, 28],
        [12, 4],
    )
    assert [_values(wait) for wait in ops.waits] == [([1], [2]), ([2], [2])]


def test_kvio_translates_one_packed_main_plane():
    ops = FakeKVIOOps()
    backend = KVIODSASparseKVBackend(
        7,
        ops_module=ops,
        tensor_ops=ops,
    )
    packed_main = torch.zeros((4, 2, 1, 13), dtype=torch.uint8)
    backend.register_layer_cache(
        layer_id=3,
        block_size=2,
        cache_planes=(packed_main,),
    )
    backend.finalize_cache_registration()

    backend.put_blocks(
        layer_id=3,
        storage_request_ids=torch.tensor([41], dtype=torch.int64),
        source_block_ids=torch.tensor([2], dtype=torch.int64),
    )
    backend.load_tokens_into(
        layer_id=3,
        storage_request_ids=torch.tensor([41], dtype=torch.int64),
        token_offsets_in_block=torch.tensor([1], dtype=torch.int64),
        destination_physical_slots=torch.tensor([4], dtype=torch.int64),
    )

    assert ops.init_args == (
        [packed_main.data_ptr()],
        [packed_main.numel() * packed_main.element_size()],
    )
    assert _values(ops.calls[0]) == (
        [1],
        [7],
        [1],
        [1],
        [0x05],
        [0],
        [41],
        [52],
        [0],
        [26],
    )
    assert _values(ops.calls[1]) == (
        [2],
        [7],
        [1],
        [1],
        [0x06],
        [0],
        [41],
        [52],
        [13],
        [13],
    )
    assert [_values(wait) for wait in ops.waits] == [([1], [1]), ([2], [1])]
