import torch

from vllm_ascend.dsa_sparse.dsa_ascend_ops_backend import AscendDSAOpsBackend
from vllm_ascend.dsa_sparse.dsa_kv_backend import (
    MockDSAKVBackend, compose_dsa_storage_request_ids,
    encode_dsa_block_hash_id)
from vllm_ascend.dsa_sparse.dsa_kvio_backend import KVIODSAKVBackend


class FakeKVIOOps:
    class ErrorCode:
        SUCCESS = 0

    def __init__(self):
        self.init_args = None
        self.put_calls = []
        self.get_calls = []
        self.wait_calls = []

    @staticmethod
    def _clone_args(args):
        return tuple(
            arg.clone() if torch.is_tensor(arg) else arg for arg in args)

    def aiv_init(self, cache_addresses, cache_lengths):
        self.init_args = (cache_addresses, cache_lengths)
        return self.ErrorCode.SUCCESS

    def npu_get_put_batch(self, *args):
        cloned_args = self._clone_args(args)
        opcode = int(cloned_args[4].reshape(-1)[0])
        if opcode == 0x05:
            self.put_calls.append(cloned_args)
        elif opcode == 0x06:
            self.get_calls.append(cloned_args)
        else:
            raise AssertionError(f"unexpected KVIO opcode {opcode}")
        return self.ErrorCode.SUCCESS

    def npu_send_wait(self, *args):
        self.wait_calls.append(self._clone_args(args))
        return self.ErrorCode.SUCCESS


def tensor_call_values(call):
    return tuple(arg.tolist() if torch.is_tensor(arg) else arg for arg in call)


def test_storage_request_id_combines_block_hash_and_layer():
    block_hash_id = encode_dsa_block_hash_id(b"vllm-block-hash")
    assert block_hash_id == encode_dsa_block_hash_id(b"vllm-block-hash")
    assert encode_dsa_block_hash_id(42) == 42

    block_hash_ids = torch.tensor([block_hash_id], dtype=torch.long)
    layer_3_id = compose_dsa_storage_request_ids(block_hash_ids, 3)
    layer_4_id = compose_dsa_storage_request_ids(block_hash_ids, 4)
    assert layer_3_id.tolist() == [(block_hash_id << 8) | 3]
    assert layer_4_id.tolist() == [(block_hash_id << 8) | 4]

    for invalid_block_hash in (-1, b""):
        try:
            encode_dsa_block_hash_id(invalid_block_hash)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid block hash must be rejected")


def test_mock_backend_writes_only_lookup_miss_destinations():
    backend = MockDSAKVBackend(seed=0)
    nopek_cache = torch.zeros((4, 2, 1, 3), dtype=torch.float32)
    ropek_cache = torch.zeros((4, 2, 1, 1), dtype=torch.float32)
    backend.register_layer_cache(
        layer_id=0,
        block_size=2,
        nopek_cache=nopek_cache,
        ropek_cache=ropek_cache,
    )

    backend.load_tokens_into(
        layer_id=0,
        storage_request_ids=torch.tensor([44, 77], dtype=torch.long),
        token_offsets_in_block=torch.tensor([1, 0], dtype=torch.long),
        destination_physical_slots=torch.tensor([4, 6], dtype=torch.long),
    )

    expected_rows = torch.tensor([4, 6])
    nopek_rows = nopek_cache.reshape(-1, nopek_cache.shape[-1])
    ropek_rows = ropek_cache.reshape(-1, ropek_cache.shape[-1])
    assert torch.all(nopek_rows.index_select(0, expected_rows) != 0)
    assert torch.all(ropek_rows.index_select(0, expected_rows) != 0)

    untouched_rows = torch.tensor([0, 1, 2, 3, 5, 7])
    assert torch.count_nonzero(nopek_rows.index_select(0, untouched_rows)) == 0
    assert torch.count_nonzero(ropek_rows.index_select(0, untouched_rows)) == 0

    registered_nopek = backend._layer_caches[0][1]
    registered_ropek = backend._layer_caches[0][2]
    assert registered_nopek.data_ptr() == nopek_cache.data_ptr()
    assert registered_ropek.data_ptr() == ropek_cache.data_ptr()


def test_ops_backend_compacts_token_layer_load_descriptors():
    ops = FakeKVIOOps()
    backend = KVIODSAKVBackend(
        model_id=7,
        pd_flag=1,
        ops_module=ops,
        tensor_ops=ops,
    )
    nopek_cache = torch.zeros((4, 2, 1, 3), dtype=torch.float32)
    ropek_cache = torch.zeros((4, 2, 1, 1), dtype=torch.float32)
    backend.register_layer_cache(
        layer_id=3,
        block_size=2,
        nopek_cache=nopek_cache,
        ropek_cache=ropek_cache,
    )
    backend.finalize_cache_registration()

    AscendDSAOpsBackend._load_selected_tokens(
        kv_backend=backend,
        layer_id=3,
        block_hash_ids=torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
        token_positions=torch.tensor([[1, 2], [3, 0]], dtype=torch.int32),
        destination_slots=torch.tensor([[0, 3], [1, 0]], dtype=torch.int32),
        load_mask=torch.tensor([[True, True], [True, False]]),
        destination_block_table=torch.tensor(
            [[2, 0], [1, 3]], dtype=torch.int32),
        block_size=2,
    )

    expected_request_ids = compose_dsa_storage_request_ids(
        torch.tensor([10, 11, 21], dtype=torch.long), 3).repeat_interleave(2)
    assert tensor_call_values(ops.get_calls[0]) == (
        [1],
        [7],
        [1],
        [6],
        [0x06],
        [0, 1, 0, 1, 0, 1],
        expected_request_ids.tolist(),
        [48, 16, 12, 4, 36, 12],
        [12, 28, 0, 24, 12, 28],
        [12, 4, 12, 4, 12, 4],
    )
    assert tensor_call_values(ops.wait_calls[0]) == ([1], [6])


def test_mock_backend_put_and_close_are_storage_free():
    backend = MockDSAKVBackend()
    nopek_cache = torch.zeros((1, 2, 1, 3), dtype=torch.float32)
    ropek_cache = torch.zeros((1, 2, 1, 1), dtype=torch.float32)
    backend.register_layer_cache(
        layer_id=3,
        block_size=2,
        nopek_cache=nopek_cache,
        ropek_cache=ropek_cache,
    )

    backend.put_blocks(
        layer_id=3,
        storage_request_ids=torch.tensor([42], dtype=torch.long),
        source_block_ids=torch.tensor([0], dtype=torch.long),
    )
    assert set(backend.__dict__) == {
        "_layer_caches",
        "_random",
        "_put_logged",
        "_load_logged",
    }
    assert backend._layer_caches[3][1].data_ptr() == nopek_cache.data_ptr()
    assert backend._layer_caches[3][2].data_ptr() == ropek_cache.data_ptr()

    backend.close()
    assert backend._layer_caches == {}


def test_kvio_backend_translates_block_put_offsets_and_waits():
    ops = FakeKVIOOps()
    backend = KVIODSAKVBackend(
        model_id=7,
        pd_flag=1,
        ops_module=ops,
        tensor_ops=ops,
    )
    nopek_cache = torch.zeros((4, 2, 1, 3), dtype=torch.float32)
    ropek_cache = torch.zeros((4, 2, 1, 1), dtype=torch.float32)
    backend.register_layer_cache(
        layer_id=3,
        block_size=2,
        nopek_cache=nopek_cache,
        ropek_cache=ropek_cache,
    )
    backend.finalize_cache_registration()

    assert ops.init_args == (
        [nopek_cache.data_ptr(), ropek_cache.data_ptr()],
        [nopek_cache.numel() * nopek_cache.element_size(),
         ropek_cache.numel() * ropek_cache.element_size()],
    )

    block_hash_id = encode_dsa_block_hash_id(b"block-0")
    remote_request_id = int(
        compose_dsa_storage_request_ids(
            torch.tensor([block_hash_id], dtype=torch.long), 3)[0])
    backend.put_blocks(
        layer_id=3,
        storage_request_ids=torch.tensor(
            [remote_request_id], dtype=torch.long),
        source_block_ids=torch.tensor([2], dtype=torch.long),
    )

    assert tensor_call_values(ops.put_calls[0]) == (
        [1],
        [7],
        [0],
        [2],
        [0x05],
        [0, 1],
        [remote_request_id, remote_request_id],
        [48, 16],
        [0, 24],
        [24, 8],
    )
    assert tensor_call_values(ops.wait_calls[0]) == ([1], [2])
    assert tensor_call_values(ops.put_calls[1]) == (
        [2],
        [7],
        [1],
        [2],
        [0x05],
        [0, 1],
        [remote_request_id, remote_request_id],
        [48, 16],
        [0, 24],
        [24, 8],
    )
    assert tensor_call_values(ops.wait_calls[1]) == ([2], [2])


def test_kvio_backend_translates_token_get_offsets_and_closes():
    ops = FakeKVIOOps()
    backend = KVIODSAKVBackend(
        model_id=7,
        pd_flag=1,
        ops_module=ops,
        tensor_ops=ops,
    )
    nopek_cache = torch.zeros((4, 2, 1, 3), dtype=torch.float32)
    ropek_cache = torch.zeros((4, 2, 1, 1), dtype=torch.float32)
    backend.register_layer_cache(
        layer_id=3,
        block_size=2,
        nopek_cache=nopek_cache,
        ropek_cache=ropek_cache,
    )
    backend.finalize_cache_registration()
    first_block_hash_id = encode_dsa_block_hash_id(b"block-0")
    second_block_hash_id = encode_dsa_block_hash_id(b"block-1")
    storage_request_ids = compose_dsa_storage_request_ids(
        torch.tensor(
            [first_block_hash_id, second_block_hash_id], dtype=torch.long),
        3,
    )
    remote_request_id = int(storage_request_ids[0])
    backend.put_blocks(
        layer_id=3,
        storage_request_ids=torch.tensor(
            [remote_request_id], dtype=torch.long),
        source_block_ids=torch.tensor([0], dtype=torch.long),
    )
    ops.put_calls.clear()
    ops.wait_calls.clear()

    backend.load_tokens_into(
        layer_id=3,
        storage_request_ids=storage_request_ids,
        token_offsets_in_block=torch.tensor([1, 0], dtype=torch.long),
        destination_physical_slots=torch.tensor([4, 1], dtype=torch.long),
    )

    assert tensor_call_values(ops.get_calls[0]) == (
        [3],
        [7],
        [1],
        [4],
        [0x06],
        [0, 1, 0, 1],
        [int(storage_request_ids[0]), int(storage_request_ids[0]),
         int(storage_request_ids[1]), int(storage_request_ids[1])],
        [48, 16, 12, 4],
        [12, 28, 0, 24],
        [12, 4, 12, 4],
    )
    assert tensor_call_values(ops.wait_calls[0]) == ([3], [4])

    backend.close()
    backend.close()
