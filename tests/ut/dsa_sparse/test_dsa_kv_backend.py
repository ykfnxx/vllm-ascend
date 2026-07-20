import hashlib

import torch

from vllm_ascend.dsa_sparse.dsa_kv_backend import MockDSAKVBackend
from vllm_ascend.dsa_sparse.dsa_kvio_backend import KVIODSAKVBackend


class FakeKVIOOps:
    class ErrorCode:
        SUCCESS = 0

    def __init__(self):
        self.init_args = None
        self.put_calls = []
        self.get_calls = []
        self.wait_calls = []
        self.destroy_calls = 0

    def aiv_init(self, cache_addresses, cache_lengths):
        self.init_args = (cache_addresses, cache_lengths)
        return self.ErrorCode.SUCCESS

    def aiv_put_batch(self, *args):
        self.put_calls.append(args)
        return self.ErrorCode.SUCCESS, 11.0

    def aiv_get_batch(self, *args):
        self.get_calls.append(args)
        return self.ErrorCode.SUCCESS, 13.0

    def aiv_wait(self, task_ids):
        self.wait_calls.append(task_ids)
        return self.ErrorCode.SUCCESS

    def aiv_destroy(self):
        self.destroy_calls += 1


def encode_request_id(request_id: str) -> int:
    digest = hashlib.blake2b(
        request_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & 0x7FFFFFFFFFFFFFFF


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
        request_pool_entries=torch.tensor([4, 7], dtype=torch.int32),
        token_positions=torch.tensor([[9, 10], [11, 12]], dtype=torch.int32),
        destination_slots=torch.tensor([[0, 3], [1, 2]], dtype=torch.int32),
        load_mask=torch.tensor([[True, False], [False, True]]),
        destination_block_table=torch.tensor([[2, 0], [1, 3]], dtype=torch.int32),
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


def test_mock_backend_put_release_and_close_are_storage_free():
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
        request_ids=[42],
        request_pool_indices=[0],
        logical_block_index_rows=[[0]],
        block_key_rows=[["block-0"]],
        source_block_id_rows=[[0]],
    )
    backend.release_request(request_id="request-0", request_pool_idx=0)

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
        max_model_len=8,
        ops_module=ops,
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

    request_id = "request-0"
    remote_request_id = encode_request_id(request_id)
    backend.put_blocks(
        layer_id=3,
        request_ids=[request_id],
        request_pool_indices=[1],
        logical_block_index_rows=[[1]],
        block_key_rows=[["block-1"]],
        source_block_id_rows=[[2]],
    )

    assert ops.put_calls == [(
        1,
        7,
        1,
        [0, 1],
        [remote_request_id, remote_request_id],
        [48, 16],
        [24, 104],
        [24, 8],
    )]
    assert ops.wait_calls == [[1]]


def test_kvio_backend_translates_token_get_offsets_and_closes():
    ops = FakeKVIOOps()
    backend = KVIODSAKVBackend(
        model_id=7,
        pd_flag=1,
        max_model_len=8,
        ops_module=ops,
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
    request_id = "request-0"
    remote_request_id = encode_request_id(request_id)
    backend.put_blocks(
        layer_id=3,
        request_ids=[request_id],
        request_pool_indices=[1],
        logical_block_index_rows=[[0]],
        block_key_rows=[["block-0"]],
        source_block_id_rows=[[0]],
    )
    ops.put_calls.clear()
    ops.wait_calls.clear()

    backend.load_tokens_into(
        layer_id=3,
        request_pool_entries=torch.tensor([1], dtype=torch.int32),
        token_positions=torch.tensor([[5, 6]], dtype=torch.int32),
        destination_slots=torch.tensor([[0, 3]], dtype=torch.int32),
        load_mask=torch.tensor([[True, True]]),
        destination_block_table=torch.tensor([[2, 0]], dtype=torch.int32),
    )

    assert ops.get_calls == [(
        2,
        7,
        1,
        [0, 1, 0, 1],
        [remote_request_id] * 4,
        [48, 16, 12, 4],
        [60, 116, 72, 120],
        [12, 4, 12, 4],
    )]
    assert ops.wait_calls == [[2]]

    backend.close()
    backend.close()
    assert ops.destroy_calls == 1
