# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.kvio import KVIOBackend


class FakeKVIOOps:
    def __init__(self, error_code: int = 0) -> None:
        self.error_code = error_code
        self.init_calls: list[tuple[list[int], list[int]]] = []

    def aiv_init(self, addresses: list[int], lengths: list[int]) -> int:
        self.init_calls.append((addresses, lengths))
        return self.error_code


class FakeTensorOps:
    def __init__(self) -> None:
        self.batch_calls: list[tuple[torch.Tensor, ...]] = []
        self.wait_calls: list[tuple[torch.Tensor, ...]] = []

    def npu_get_put_batch(self, *arguments: torch.Tensor) -> None:
        self.batch_calls.append(tuple(argument.clone() for argument in arguments))

    def npu_send_wait(self, *arguments: torch.Tensor) -> None:
        self.wait_calls.append(tuple(argument.clone() for argument in arguments))


def make_backend(error_code: int = 0) -> tuple[KVIOBackend, FakeKVIOOps, FakeTensorOps]:
    kvio_ops = FakeKVIOOps(error_code)
    tensor_ops = FakeTensorOps()
    return KVIOBackend(17, ops_module=kvio_ops, tensor_ops=tensor_ops), kvio_ops, tensor_ops


def int64(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)


def descriptors(tensor_ops: FakeTensorOps) -> tuple[list[int], ...]:
    call = tensor_ops.batch_calls[-1]
    return tuple(tensor.tolist() for tensor in call[5:10])


def test_two_plane_put_descriptors() -> None:
    backend, kvio_ops, tensor_ops = make_backend()
    nope = torch.empty((4, 2, 3), dtype=torch.float16)
    rope = torch.empty((4, 2, 1), dtype=torch.float32)
    backend.register_put_cache(layer_id=5, block_size=2, cache_planes=(nope, rope))
    backend.finalize_registration()

    backend.put_blocks(
        layer_id=5,
        storage_ids=int64([101, 202]),
        source_block_ids=int64([2, 1]),
    )

    assert kvio_ops.init_calls == [([nope.data_ptr(), rope.data_ptr()], [48, 32])]
    assert descriptors(tensor_ops) == (
        [0, 1, 0, 1],
        [101, 101, 202, 202],
        [24, 16, 12, 8],
        [0, 12, 0, 12],
        [12, 8, 12, 8],
    )
    assert tuple(tensor.tolist() for tensor in tensor_ops.batch_calls[0][:5]) == (
        [1],
        [17],
        [1],
        [4],
        [5],
    )
    assert tensor_ops.wait_calls[0][1].tolist() == [4]


def test_two_plane_get_descriptors() -> None:
    backend, _, tensor_ops = make_backend()
    nope = torch.empty((4, 2, 3), dtype=torch.float16)
    rope = torch.empty((4, 2, 1), dtype=torch.float32)
    backend.register_get_cache(layer_id=5, block_size=2, cache_planes=(nope, rope))
    backend.finalize_registration()

    backend.get_tokens(
        layer_id=5,
        storage_ids=int64([101, 202]),
        token_offsets=int64([1, 0]),
        destination_slots=int64([3, 1]),
    )

    assert descriptors(tensor_ops) == (
        [0, 1, 0, 1],
        [101, 101, 202, 202],
        [18, 12, 6, 4],
        [6, 16, 0, 12],
        [6, 4, 6, 4],
    )
    assert tensor_ops.batch_calls[0][4].tolist() == [6]


def test_single_packed_plane_is_shared_between_put_and_get() -> None:
    backend, kvio_ops, tensor_ops = make_backend()
    packed = torch.empty((3, 2, 5), dtype=torch.uint8)
    backend.register_put_cache(layer_id=1, block_size=2, cache_planes=(packed,))
    backend.register_get_cache(layer_id=1, block_size=2, cache_planes=(packed,))
    backend.finalize_registration()

    backend.put_blocks(layer_id=1, storage_ids=int64([11]), source_block_ids=int64([2]))
    backend.get_tokens(
        layer_id=1,
        storage_ids=int64([11]),
        token_offsets=int64([1]),
        destination_slots=int64([4]),
    )

    assert kvio_ops.init_calls == [([packed.data_ptr()], [30])]
    assert descriptors(tensor_ops) == ([0], [11], [20], [5], [5])
    assert tensor_ops.batch_calls[-1][0].tolist() == [2]


def test_separate_put_and_get_tensors_have_distinct_cache_ids() -> None:
    backend, kvio_ops, tensor_ops = make_backend()
    put_cache = torch.empty((3, 2, 4), dtype=torch.float16)
    get_cache = torch.empty((2, 2, 4), dtype=torch.float16)
    backend.register_put_cache(layer_id=1, block_size=2, cache_planes=(put_cache,))
    backend.register_get_cache(layer_id=1, block_size=2, cache_planes=(get_cache,))
    backend.finalize_registration()

    backend.get_tokens(
        layer_id=1,
        storage_ids=int64([7]),
        token_offsets=int64([0]),
        destination_slots=int64([1]),
    )

    assert kvio_ops.init_calls == [([put_cache.data_ptr(), get_cache.data_ptr()], [48, 32])]
    assert descriptors(tensor_ops)[0] == [1]


def test_empty_batch_does_not_submit() -> None:
    backend, _, tensor_ops = make_backend()
    cache = torch.empty((2, 2, 4), dtype=torch.float16)
    backend.register_put_cache(layer_id=1, block_size=2, cache_planes=(cache,))
    backend.finalize_registration()

    backend.put_blocks(layer_id=1, storage_ids=int64([]), source_block_ids=int64([]))

    assert tensor_ops.batch_calls == []
    assert tensor_ops.wait_calls == []


def test_registration_and_descriptor_contract_errors() -> None:
    backend, _, _ = make_backend()
    cache = torch.empty((2, 2, 4), dtype=torch.float16)
    backend.register_put_cache(layer_id=1, block_size=2, cache_planes=(cache,))

    with pytest.raises(ValueError, match="already registered"):
        backend.register_put_cache(layer_id=1, block_size=2, cache_planes=(cache,))
    with pytest.raises(RuntimeError, match="not finalized"):
        backend.put_blocks(layer_id=1, storage_ids=int64([1]), source_block_ids=int64([0]))

    backend.finalize_registration()
    with pytest.raises(RuntimeError, match="finalized"):
        backend.register_get_cache(layer_id=1, block_size=2, cache_planes=(cache,))
    with pytest.raises(ValueError, match="not registered"):
        backend.get_tokens(
            layer_id=1,
            storage_ids=int64([1]),
            token_offsets=int64([0]),
            destination_slots=int64([0]),
        )
    with pytest.raises(ValueError, match="contiguous int64 vector"):
        backend.put_blocks(
            layer_id=1,
            storage_ids=torch.tensor([1], dtype=torch.int32),
            source_block_ids=int64([0]),
        )


def test_layout_and_initialization_errors() -> None:
    backend, _, _ = make_backend()
    cache = torch.empty((2, 2, 4), dtype=torch.float16)

    with pytest.raises(ValueError, match="must not be empty"):
        backend.register_put_cache(layer_id=1, block_size=2, cache_planes=())
    with pytest.raises(ValueError, match="block dimension"):
        backend.register_put_cache(layer_id=1, block_size=3, cache_planes=(cache,))

    noncontiguous = torch.empty((2, 2, 4), dtype=torch.float16).transpose(1, 2)
    with pytest.raises(ValueError, match="contiguous"):
        backend.register_put_cache(layer_id=1, block_size=4, cache_planes=(noncontiguous,))

    mismatch_backend, _, _ = make_backend()
    mismatch_backend.register_put_cache(layer_id=1, block_size=2, cache_planes=(cache,))
    mismatch_backend.register_get_cache(
        layer_id=1,
        block_size=2,
        cache_planes=(torch.empty((2, 2, 3), dtype=torch.float16),),
    )
    with pytest.raises(ValueError, match="layouts differ"):
        mismatch_backend.finalize_registration()

    failing_backend, _, _ = make_backend(error_code=9)
    failing_backend.register_put_cache(layer_id=1, block_size=2, cache_planes=(cache,))
    with pytest.raises(RuntimeError, match="error code 9"):
        failing_backend.finalize_registration()
