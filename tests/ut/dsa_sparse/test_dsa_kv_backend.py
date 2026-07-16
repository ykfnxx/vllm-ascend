import torch

from vllm_ascend.dsa_sparse.dsa_kv_backend import MockDSAKVBackend


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
        request_ids=["request-0"],
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
