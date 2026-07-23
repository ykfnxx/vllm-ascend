from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.dsa_sparse.dsa_ascend_ops_backend import (
    AscendDSAOpsBackend,
)
from vllm_ascend.dsa_sparse.dsa_config import (
    validate_dsa_kv_transfer_config,
)
from vllm_ascend.dsa_sparse.dsa_forward_batch import (
    DSAModelForwardMeta,
    _build_forward_batches_from_dsa_meta,
)
from vllm_ascend.dsa_sparse.dsa_pd import (
    DSA_KVIO_CONNECTOR_NAME,
    DSA_KVIO_PD_LAYER_TOPK_KEY,
    DSA_KVIO_PD_MANIFEST_KEY,
    DSA_KVIO_PD_PROTOCOL_VERSION,
    DSA_KVIO_PD_STATE_READY,
    DSA_MOONCAKE_CONNECTOR_NAME,
    DSA_PD_INITIAL_TRANSPORT_MOONCAKE,
    DSAKVIOPDManifest,
    DSAKVIOPDRequest,
    build_pd_resident_token_ids,
    get_dsa_kvio_layer_topk,
    get_dsa_kvio_pd_manifest,
    serialize_dsa_kvio_layer_topk,
)
from vllm_ascend.dsa_sparse.dsa_req_meta import ReqMeta
from vllm_ascend.dsa_sparse.dsa_resident_pool import (
    DSAResidentLookupState,
    DSAResidentTokenPool,
)
from vllm_ascend.dsa_sparse.dsa_sparse import DSASparseV1
from vllm_ascend.dsa_sparse.dsa_types import ReqStage


def test_pd_manifest_builds_compact_resident_and_tail_plan():
    manifest = DSAKVIOPDManifest.build(
        remote_request_id=123,
        model_id=7,
        stored_token_count=10_371,
        block_size=128,
        index_capacity=128 * 1024,
        resident_tokens=8 * 1024,
        free_slot_tokens=2 * 1024,
        producer_world_size=2,
        layout_fingerprint=456,
        generation=789,
    )

    assert manifest.protocol_version == DSA_KVIO_PD_PROTOCOL_VERSION
    assert manifest.state == DSA_KVIO_PD_STATE_READY
    assert manifest.generation == 789
    assert manifest.producer_world_size == 2
    assert manifest.layout_fingerprint == 456
    assert manifest.logical_block_count == 82
    assert manifest.resident_slot_start == 0
    assert manifest.resident_token_count == 8 * 1024
    assert manifest.tail_token_start == 10_368
    assert manifest.tail_token_count == 3
    assert manifest.tail_slot_start == 10 * 1024
    assert get_dsa_kvio_pd_manifest({
        DSA_KVIO_PD_MANIFEST_KEY: manifest.to_dict()
    }) == manifest


def test_pd_manifest_rejects_inconsistent_tail_metadata():
    manifest = DSAKVIOPDManifest.build(
        remote_request_id=123,
        model_id=7,
        stored_token_count=10_371,
        block_size=128,
        index_capacity=128 * 1024,
        resident_tokens=8 * 1024,
        free_slot_tokens=2 * 1024,
        producer_world_size=1,
        layout_fingerprint=456,
    ).to_dict()
    manifest["tail_token_count"] = 4

    with pytest.raises(ValueError, match="tail range"):
        DSAKVIOPDManifest.from_dict(manifest)


def test_pd_manifest_rejects_layout_larger_than_index_capacity():
    with pytest.raises(ValueError, match="exceeds lookup index capacity"):
        DSAKVIOPDManifest.build(
            remote_request_id=123,
            model_id=7,
            stored_token_count=512,
            block_size=128,
            index_capacity=1024,
            resident_tokens=768,
            free_slot_tokens=512,
            producer_world_size=1,
            layout_fingerprint=456,
        )


def test_pd_manifest_rejects_non_ready_state():
    manifest = DSAKVIOPDManifest.build(
        remote_request_id=123,
        model_id=7,
        stored_token_count=10_371,
        block_size=128,
        index_capacity=128 * 1024,
        resident_tokens=8 * 1024,
        free_slot_tokens=2 * 1024,
        producer_world_size=1,
        layout_fingerprint=456,
    ).to_dict()
    manifest["state"] = 0

    with pytest.raises(ValueError, match="not ready"):
        DSAKVIOPDManifest.from_dict(manifest)


def _make_pd_transfer_config(
    *,
    connector=DSA_KVIO_CONNECTOR_NAME,
    role="kv_producer",
    backend="kvio",
    prefix_caching=False,
):
    topology = {
        "prefill": {"tp_size": 1, "dp_size": 1},
        "decode": {"tp_size": 1, "dp_size": 1},
    }
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            dsa_kv_backend=backend,
            enable_prefix_caching=prefix_caching,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector,
            kv_role=role,
            get_from_extra_config=(
                lambda name, default: topology.get(name, default)
            ),
        ),
    )


def test_dsa_kv_transfer_validation_allows_supported_pd_connectors():
    validate_dsa_kv_transfer_config(
        _make_pd_transfer_config(role="kv_producer")
    )
    validate_dsa_kv_transfer_config(
        _make_pd_transfer_config(role="kv_consumer")
    )
    validate_dsa_kv_transfer_config(
        _make_pd_transfer_config(
            connector=DSA_MOONCAKE_CONNECTOR_NAME,
            role="kv_producer",
        )
    )
    validate_dsa_kv_transfer_config(
        _make_pd_transfer_config(
            connector=DSA_MOONCAKE_CONNECTOR_NAME,
            role="kv_consumer",
        )
    )
    validate_dsa_kv_transfer_config(SimpleNamespace(
        cache_config=SimpleNamespace(),
        kv_transfer_config=None,
    ))


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (
            _make_pd_transfer_config(connector="MooncakeConnectorV1"),
            "supports KV transfer only",
        ),
        (
            _make_pd_transfer_config(backend="mock"),
            "kv_backend.*kvio",
        ),
        (
            _make_pd_transfer_config(role="kv_both"),
            "kv_role",
        ),
        (
            _make_pd_transfer_config(
                role="kv_consumer",
                prefix_caching=True,
            ),
            "prefix caching",
        ),
    ],
)
def test_dsa_kv_transfer_validation_rejects_unsupported_modes(
    config,
    error,
):
    with pytest.raises(ValueError, match=error):
        validate_dsa_kv_transfer_config(config)


def test_dsa_mooncake_rejects_mismatched_pd_topology():
    config = _make_pd_transfer_config(
        connector=DSA_MOONCAKE_CONNECTOR_NAME
    )
    topology = {
        "prefill": {"tp_size": 2, "dp_size": 1},
        "decode": {"tp_size": 1, "dp_size": 1},
    }
    config.kv_transfer_config.get_from_extra_config = (
        lambda name, default: topology.get(name, default)
    )

    with pytest.raises(ValueError, match="matching P/D TP and DP"):
        validate_dsa_kv_transfer_config(config)


def test_pd_resident_initialization_prioritizes_topk_and_excludes_tail():
    token_ids = build_pd_resident_token_ids(
        topk_token_ids=[9, 3, 3, 16, -1, 8],
        stored_token_count=18,
        block_size=4,
        resident_token_count=6,
    )

    assert token_ids == [9, 3, 8, 0, 1, 2]
    assert 16 not in token_ids
    assert all(token_id < 16 for token_id in token_ids)


def test_pd_layer_topk_round_trips_json_rank_and_layer_keys():
    layer_topk_by_rank = {
        0: {2: [9, 3]},
        1: {7: [8, 4]},
    }
    transfer_params = {
        DSA_KVIO_PD_LAYER_TOPK_KEY: serialize_dsa_kvio_layer_topk(
            layer_topk_by_rank
        )
    }

    assert get_dsa_kvio_layer_topk(transfer_params) == layer_topk_by_rank


def test_prefill_capture_uses_last_query_row_and_keeps_valid_prompt_topk():
    manager = DSASparseV1.__new__(DSASparseV1)
    manager._prefill_layer_topk = {}
    manager.dsa_meta = SimpleNamespace(requests=[SimpleNamespace(
        request_id="prefill-request",
        is_last_prefill_chunk=True,
        query_start_loc=0,
        query_len=2,
        num_prompt_tokens=18,
        forward_plan=SimpleNamespace(dense_tail_start=16),
    )])
    topk_indices = torch.tensor([
        [[1, 2, 3, 4, 5]],
        [[9, 3, 3, 16, 18]],
    ], dtype=torch.int32)

    manager.capture_prefill_last_token_topk(
        "model.layers.3.self_attn", topk_indices
    )

    assert manager.take_pd_prefill_layer_topk() == {
        "prefill-request": {3: [9, 3, 16]}
    }
    assert manager.take_pd_prefill_layer_topk() == {}


def test_non_pd_initialization_uses_per_layer_final_prefill_topk():
    manager = DSASparseV1.__new__(DSASparseV1)
    manager._hbm_resident_tokens = 6
    manager._local_topk_init_logged_requests = set()
    manager._prefill_layer_topk = {
        "local-request": {
            3: [9, 3, 16, 4],
            4: [7, 6],
        }
    }
    req_meta = SimpleNamespace(
        request_id="local-request",
        block_size=4,
        stage=ReqStage.ENTER_SPARSE_DECODE,
        pd_remote_loaded=False,
        forward_plan=SimpleNamespace(
            is_sparse_decode=True,
            dense_tail_start=16,
            tail_valid_token_count=2,
        ),
    )
    manager.dsa_meta = SimpleNamespace(requests=[req_meta])
    forward_batch = SimpleNamespace(
        has_lookup_init_rows=True,
        request_ids=["local-request"],
        batch_hbm_block_table=torch.zeros((1, 2), dtype=torch.int32),
    )

    layer3_tokens, initialized = (
        manager._build_local_resident_initial_token_ids(
            layer_id=3,
            forward_batch=forward_batch,
        )
    )
    assert layer3_tokens.tolist() == [[9, 3, 4, 0, 1, 2]]
    assert initialized == ["local-request"]
    with patch("vllm_ascend.dsa_sparse.dsa_sparse.logger") as mock_logger:
        manager._consume_local_prefill_layer_topk(
            layer_id=3,
            request_ids=initialized,
        )
        mock_logger.info.assert_called_once()
        assert "final-Prefill TopK" in mock_logger.info.call_args.args[0]
        mock_logger.debug.assert_called_once()
    assert manager._prefill_layer_topk == {
        "local-request": {4: [7, 6]}
    }

    layer4_tokens, initialized = (
        manager._build_local_resident_initial_token_ids(
            layer_id=4,
            forward_batch=forward_batch,
        )
    )
    assert layer4_tokens.tolist() == [[7, 6, 0, 1, 2, 3]]
    with patch("vllm_ascend.dsa_sparse.dsa_sparse.logger") as mock_logger:
        manager._consume_local_prefill_layer_topk(
            layer_id=4,
            request_ids=initialized,
        )
        mock_logger.info.assert_not_called()
        mock_logger.debug.assert_called_once()
    assert manager._prefill_layer_topk == {}


def test_non_pd_initialization_requires_final_prefill_topk():
    manager = DSASparseV1.__new__(DSASparseV1)
    manager._hbm_resident_tokens = 6
    manager._prefill_layer_topk = {}
    manager.dsa_meta = SimpleNamespace(requests=[SimpleNamespace(
        request_id="local-request",
        block_size=4,
        stage=ReqStage.ENTER_SPARSE_DECODE,
        pd_remote_loaded=False,
        forward_plan=SimpleNamespace(
            is_sparse_decode=True,
            dense_tail_start=16,
            tail_valid_token_count=2,
        ),
    )])
    forward_batch = SimpleNamespace(
        has_lookup_init_rows=True,
        request_ids=["local-request"],
        batch_hbm_block_table=torch.zeros((1, 2), dtype=torch.int32),
    )

    with pytest.raises(RuntimeError, match="missing the final Prefill TopK"):
        manager._build_local_resident_initial_token_ids(
            layer_id=3,
            forward_batch=forward_batch,
        )


def test_resident_row_initialization_maps_selected_tokens_to_slots():
    loaded: dict = {}
    kv_backend = SimpleNamespace(
        load_tokens_into=lambda **kwargs: loaded.update(kwargs)
    )
    lookup_state = DSAResidentLookupState(
        token_to_slot=torch.full((1, 32), -1, dtype=torch.int32),
        slot_to_token=torch.full((1, 6), -1, dtype=torch.int32),
        free_slots=torch.tensor([[4, 5]], dtype=torch.int32),
        free_head=torch.zeros((1, 16), dtype=torch.int32),
    )
    initial_tokens = torch.tensor([[9, 3, 4, 0]], dtype=torch.int32)

    AscendDSAOpsBackend()._initialize_resident_rows(
        layer_id=3,
        kv_backend=kv_backend,
        state=lookup_state,
        pool_entries=torch.tensor([0], dtype=torch.int32),
        initialize_rows=torch.tensor([True]),
        resident_tokens=4,
        selection_block_table=torch.tensor([[7]], dtype=torch.int32),
        initial_resident_token_ids=initial_tokens,
    )

    assert loaded["token_positions"].tolist() == [[9, 3, 4, 0]]
    assert loaded["destination_slots"].tolist() == [[0, 1, 2, 3]]
    assert lookup_state.slot_to_token[0, :4].tolist() == [9, 3, 4, 0]
    assert lookup_state.token_to_slot[0, [9, 3, 4, 0]].tolist() == [
        0, 1, 2, 3
    ]


def test_pd_first_d_query_uses_sparse_layout_without_reinitializing_lookup():
    req_meta = ReqMeta(
        request_id="decode-request",
        index_in_batch=0,
        num_prompt_tokens=10_372,
        num_output_tokens=0,
        num_scheduled_tokens=1,
        num_computed_tokens=10_371,
        resident_valid_seq_len=10_244,
        vllm_budget_block_ids=list(range(81)),
        indexer_block_ids=list(range(82)),
        block_size=128,
        query_start_loc=0,
        query_len=1,
        req_context_full_blk_hashes=list(range(81)),
        stage=ReqStage.ENTER_SPARSE_DECODE,
        dense_query_positions=[10_371],
        resident_query_positions=[10_243],
        dsa_sparse_enabled=True,
        dsa_sparse_budget_tokens=2_048,
        resident_pool_idx=0,
        pd_remote_loaded=True,
    )

    assert req_meta.forward_plan.is_sparse_decode is True
    assert req_meta.forward_plan.dense_tail_start == 10_368
    assert req_meta.forward_plan.resident_tail_start == 10_240
    assert req_meta.forward_plan.tail_valid_token_count == 4
    assert req_meta.forward_plan.candidate_range_end == 10_368

    forward_meta = DSAModelForwardMeta()
    forward_meta.requests.append(req_meta)
    forward_meta.full_block_table_tensor = torch.arange(
        81, dtype=torch.int32).view(1, -1)
    sparse_batch, layer_batch = _build_forward_batches_from_dsa_meta(
        forward_meta,
        tensor_device="cpu",
    )

    assert sparse_batch.request_ids == ["decode-request"]
    assert sparse_batch.sparse_row_mask_tensor.tolist() == [True]
    assert sparse_batch.lookup_init_mask_tensor.tolist() == [False]
    assert layer_batch.sparse_decode_guard_request_ids == ["decode-request"]


def test_pd_initializes_independent_lookup_mapping_per_layer():
    pool = DSAResidentTokenPool(
        max_reqs=1,
        num_layers=2,
        index_capacity=32,
        resident_tokens=8,
        free_slot_tokens=2,
        device="cpu",
    )
    pool_idx = pool.acquire("decode-request")
    pool.initialize_request_layer_mappings(
        request_id="decode-request",
        layer_token_ids={0: [1, 4, 7], 1: [2, 5, 8]},
    )

    layer0 = pool.get_layer_lookup_state(0)
    layer1 = pool.get_layer_lookup_state(1)
    assert layer0.slot_to_token[pool_idx, :3].tolist() == [1, 4, 7]
    assert layer1.slot_to_token[pool_idx, :3].tolist() == [2, 5, 8]
    assert layer0.token_to_slot[pool_idx, [1, 4, 7]].tolist() == [0, 1, 2]
    assert layer1.token_to_slot[pool_idx, [2, 5, 8]].tolist() == [0, 1, 2]
    assert layer0.token_to_slot[pool_idx, 2].item() == -1
    assert layer1.token_to_slot[pool_idx, 1].item() == -1
    assert pool.req_hbm_cached_token_counts[pool_idx].tolist() == [3, 3]


def test_pd_initialization_failure_releases_worker_local_state():
    manager = DSASparseV1.__new__(DSASparseV1)
    cleared = []
    released_backend = []
    released_pool = []
    manager._clear_full_dump_done = cleared.append
    manager.kv_backend = SimpleNamespace(
        release_request=lambda **kwargs: released_backend.append(kwargs)
    )
    manager.resident_token_pool = SimpleNamespace(
        release=released_pool.append
    )
    manager._pd_initialized_requests = {"decode-request"}

    manager._rollback_pd_request_initialization(
        request_id="decode-request",
        resident_pool_idx=3,
    )

    assert cleared == ["decode-request"]
    assert released_backend == [{
        "request_id": "decode-request",
        "request_pool_idx": 3,
    }]
    assert released_pool == ["decode-request"]
    assert manager._pd_initialized_requests == set()


def test_pd_mooncake_initialization_binds_kvio_without_duplicate_get():
    manager = DSASparseV1.__new__(DSASparseV1)
    manager._vllm_blk_size = 128
    manager._parallel_rank = 0
    manager._pd_initialized_requests = set()
    manager.full_dump_done_by_pool = torch.zeros(
        (1, 1), dtype=torch.bool
    )
    initialized_mappings = []
    manager.resident_token_pool = SimpleNamespace(
        index_capacity=128 * 1024,
        resident_tokens=8 * 1024,
        free_slot_tokens=2 * 1024,
        initialize_request_layer_mappings=(
            lambda **kwargs: initialized_mappings.append(kwargs)
        ),
    )
    bound = []
    loaded = []
    manager.kv_backend = SimpleNamespace(
        bind_request=lambda **kwargs: bound.append(kwargs),
        load_pd_request=lambda **kwargs: loaded.append(kwargs),
    )
    manifest = DSAKVIOPDManifest.build(
        remote_request_id=123,
        model_id=7,
        stored_token_count=10_371,
        block_size=128,
        index_capacity=128 * 1024,
        resident_tokens=8 * 1024,
        free_slot_tokens=2 * 1024,
        producer_world_size=1,
        layout_fingerprint=456,
    )
    request = DSAKVIOPDRequest(
        request_id="decode-request",
        manifest=manifest,
        indexer_block_ids=list(range(82)),
        resident_block_ids=list(range(81)),
        layer_topk_by_rank={0: {0: [9_000, 17]}},
        initial_transport=DSA_PD_INITIAL_TRANSPORT_MOONCAKE,
    )

    manager._initialize_pd_request(
        request_id="decode-request",
        resident_pool_idx=0,
        pd_request=request,
    )

    assert bound == [{
        "request_id": "decode-request",
        "request_pool_idx": 0,
        "remote_request_id": 123,
    }]
    assert loaded == []
    assert initialized_mappings[0]["request_id"] == "decode-request"
    assert manager.full_dump_done_by_pool[0].tolist() == [True]
    assert manager._pd_initialized_requests == {"decode-request"}


def test_pd_first_d_query_dumps_a_tail_that_becomes_full():
    req_meta = ReqMeta(
        request_id="decode-request",
        index_in_batch=0,
        num_prompt_tokens=10_496,
        num_output_tokens=0,
        num_scheduled_tokens=1,
        num_computed_tokens=10_495,
        resident_valid_seq_len=10_368,
        vllm_budget_block_ids=list(range(81)),
        indexer_block_ids=list(range(82)),
        block_size=128,
        query_start_loc=0,
        query_len=1,
        req_context_full_blk_hashes=list(range(82)),
        stage=ReqStage.ENTER_SPARSE_DECODE,
        dense_query_positions=[10_495],
        resident_query_positions=[10_367],
        dsa_sparse_enabled=True,
        dsa_sparse_budget_tokens=2_048,
        resident_pool_idx=0,
        pd_remote_loaded=True,
    )
    forward_meta = DSAModelForwardMeta()
    forward_meta.requests.append(req_meta)
    forward_meta.full_block_table_tensor = torch.arange(
        81, dtype=torch.int32).view(1, -1)

    _, layer_batch = _build_forward_batches_from_dsa_meta(
        forward_meta,
        tensor_device="cpu",
    )

    dump = layer_batch.full_block_dump_tables
    assert dump.logical_block_index_rows == [[81]]
    assert dump.block_id_rows == [[80]]
    assert dump.indexer_block_id_rows == [[81]]
    assert dump.valid_token_count_rows == [10_496]
