from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer import dsa_kvio_connector as connector_module
from vllm_ascend.distributed.kv_transfer.dsa_kvio_connector import (
    DSAKVIOConnector,
    DSAKVIOConnectorMetadata,
    DSAKVIOConnectorWorkerMetadata,
)
from vllm_ascend.dsa_sparse.dsa_pd import (
    DSA_KVIO_PD_LAYER_TOPK_KEY,
    DSA_KVIO_PD_MANIFEST_KEY,
    get_dsa_kvio_layer_topk,
)


def _make_vllm_config(
    *,
    producer: bool,
    world_size: int = 1,
    model: str = "test-model",
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2,
            cache_dtype="auto",
            dsa_kv_backend="kvio",
            dsa_kvio_model_id=7,
            enable_dsa_sparse_cache=True,
        ),
        model_config=SimpleNamespace(
            model=model,
            revision="test-revision",
            dtype="float16",
            max_model_len=32,
            get_total_num_hidden_layers=lambda: world_size,
        ),
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=producer,
            is_kv_consumer=not producer,
            engine_id=("prefill-engine" if producer else "decode-engine"),
        ),
        parallel_config=SimpleNamespace(
            rank=0,
            world_size=world_size,
            tensor_parallel_size=world_size,
            pipeline_parallel_size=1,
        ),
    )


def _make_kv_cache_config(monkeypatch):
    monkeypatch.setattr(
        connector_module,
        "is_dsa_indexer_spec",
        lambda spec: spec == "indexer",
    )
    monkeypatch.setattr(
        connector_module,
        "is_dsa_mla_resident_spec",
        lambda spec: spec == "resident",
    )
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec="indexer"),
            SimpleNamespace(kv_cache_spec="resident"),
        ])


def test_connector_handoff_builds_manifest_and_forwards_d_block_tables(
    monkeypatch,
):
    kv_cache_config = _make_kv_cache_config(monkeypatch)
    producer = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    p_request = SimpleNamespace(
        request_id="prefill-request",
        num_computed_tokens=10_371,
        num_prompt_tokens=10_371,
        output_token_ids=[42],
        kv_transfer_params={"do_remote_decode": True},
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    producer.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=DSAKVIOConnectorWorkerMetadata({
                "prefill-request": {0: {0: [9_000, 17, 10_368]}},
            })
        )
    )
    delay_free, transfer_params = producer.request_finished_all_groups(
        p_request,
        ([], []),
    )

    assert delay_free is False
    assert transfer_params is not None
    assert DSA_KVIO_PD_MANIFEST_KEY in transfer_params
    assert DSA_KVIO_PD_LAYER_TOPK_KEY in transfer_params
    assert transfer_params["do_remote_decode"] is False
    assert transfer_params["last_token_id"] == 42
    assert get_dsa_kvio_layer_topk(transfer_params) == {
        0: {0: [9_000, 17, 10_368]}
    }

    consumer = DSAKVIOConnector(
        _make_vllm_config(producer=False),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    d_request = SimpleNamespace(
        request_id="decode-request",
        prompt_token_ids=[*range(10_371), 42],
        kv_transfer_params=transfer_params,
    )
    assert consumer.get_num_new_matched_tokens(d_request, 0) == (
        10_371,
        False,
    )

    blocks = SimpleNamespace(get_block_ids=lambda: [[3, 4, 5], [8, 9]])
    consumer.update_state_after_alloc(d_request, blocks, 10_371)
    metadata = consumer.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={"decode-request": 1}))

    assert isinstance(metadata, DSAKVIOConnectorMetadata)
    assert len(metadata.dsa_requests) == 1
    pd_request = metadata.dsa_requests[0]
    assert pd_request.request_id == "decode-request"
    assert pd_request.indexer_block_ids == [3, 4, 5]
    assert pd_request.resident_block_ids == [8, 9]
    assert pd_request.manifest.stored_token_count == 10_371
    assert pd_request.layer_topk_by_rank == {
        0: {0: [9_000, 17, 10_368]}
    }


def test_consumer_rejects_local_prefix_mixed_with_remote_state(monkeypatch):
    kv_cache_config = _make_kv_cache_config(monkeypatch)
    producer = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    producer.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=DSAKVIOConnectorWorkerMetadata({
                "prefill-request": {0: {0: [9_000, 17]}},
            })
        )
    )
    _, transfer_params = producer.request_finished_all_groups(
        SimpleNamespace(
            request_id="prefill-request",
            num_computed_tokens=10_371,
            num_prompt_tokens=10_371,
            output_token_ids=[42],
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        ),
        ([], []),
    )
    consumer = DSAKVIOConnector(
        _make_vllm_config(producer=False),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    d_request = SimpleNamespace(
        request_id="decode-request",
        prompt_token_ids=[*range(10_371), 42],
        kv_transfer_params=transfer_params,
    )

    with pytest.raises(RuntimeError, match="prefix-cache"):
        consumer.get_num_new_matched_tokens(d_request, 2)


def test_worker_metadata_aggregates_ranks_and_early_consume_clears_output(
    monkeypatch,
):
    kv_cache_config = _make_kv_cache_config(monkeypatch)
    connector = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    worker_metadata = DSAKVIOConnectorWorkerMetadata({
        "prefill-request": {0: {0: [9, 3]}},
    })
    worker_metadata.aggregate(DSAKVIOConnectorWorkerMetadata({
        "prefill-request": {1: {4: [8, 2]}},
    }))
    connector_output = SimpleNamespace(
        kv_connector_worker_meta=worker_metadata
    )

    connector.update_dsa_prefill_seeds_before_request_finish(
        connector_output
    )

    assert connector_output.kv_connector_worker_meta is None
    assert connector._producer_layer_topk_by_request == {
        "prefill-request": {
            0: {0: [9, 3]},
            1: {4: [8, 2]},
        }
    }


def test_producer_rejects_handoff_without_final_prefill_topk(monkeypatch):
    connector = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        _make_kv_cache_config(monkeypatch),
    )

    with pytest.raises(RuntimeError, match="without the last Prefill"):
        connector.request_finished_all_groups(
            SimpleNamespace(
                request_id="prefill-request",
                num_computed_tokens=10_371,
                num_prompt_tokens=10_371,
                output_token_ids=[42],
                kv_transfer_params={"do_remote_decode": True},
                status=RequestStatus.FINISHED_LENGTH_CAPPED,
            ),
            ([], []),
        )


def test_producer_does_not_publish_unrequested_handoff(monkeypatch):
    connector = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        _make_kv_cache_config(monkeypatch),
    )
    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=DSAKVIOConnectorWorkerMetadata({
                "prefill-request": {0: {0: [9, 3]}},
            })
        )
    )

    delay_free, transfer_params = connector.request_finished_all_groups(
        SimpleNamespace(
            request_id="prefill-request",
            num_computed_tokens=10_371,
            num_prompt_tokens=10_371,
            output_token_ids=[42],
            kv_transfer_params=None,
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        ),
        ([], []),
    )

    assert delay_free is False
    assert transfer_params is None


def test_producer_does_not_publish_short_sparse_handoff(monkeypatch):
    connector = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        _make_kv_cache_config(monkeypatch),
    )

    delay_free, transfer_params = connector.request_finished_all_groups(
        SimpleNamespace(
            request_id="short-prefill-request",
            num_computed_tokens=10_367,
            num_prompt_tokens=10_367,
            output_token_ids=[42],
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        ),
        ([], []),
    )

    assert delay_free is False
    assert transfer_params is None


def test_producer_does_not_handoff_after_eos(monkeypatch):
    connector = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        _make_kv_cache_config(monkeypatch),
    )

    delay_free, transfer_params = connector.request_finished_all_groups(
        SimpleNamespace(
            request_id="stopped-prefill-request",
            num_computed_tokens=10_371,
            num_prompt_tokens=10_371,
            output_token_ids=[42],
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_STOPPED,
        ),
        ([], []),
    )

    assert delay_free is False
    assert transfer_params is None


def test_producer_requires_exactly_one_handoff_token(monkeypatch):
    connector = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        _make_kv_cache_config(monkeypatch),
    )
    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=DSAKVIOConnectorWorkerMetadata({
                "prefill-request": {0: {0: [9, 3]}},
            })
        )
    )

    with pytest.raises(RuntimeError, match="exactly one token"):
        connector.request_finished_all_groups(
            SimpleNamespace(
                request_id="prefill-request",
                num_computed_tokens=10_371,
                num_prompt_tokens=10_371,
                output_token_ids=[42, 43],
                kv_transfer_params={"do_remote_decode": True},
                status=RequestStatus.FINISHED_LENGTH_CAPPED,
            ),
            ([], []),
        )


def test_consumer_rejects_layout_fingerprint_mismatch(monkeypatch):
    kv_cache_config = _make_kv_cache_config(monkeypatch)
    producer = DSAKVIOConnector(
        _make_vllm_config(producer=True, model="producer-model"),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    producer.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=DSAKVIOConnectorWorkerMetadata({
                "prefill-request": {0: {0: [9, 3]}},
            })
        )
    )
    _, transfer_params = producer.request_finished_all_groups(
        SimpleNamespace(
            request_id="prefill-request",
            num_computed_tokens=10_371,
            num_prompt_tokens=10_371,
            output_token_ids=[42],
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        ),
        ([], []),
    )
    consumer = DSAKVIOConnector(
        _make_vllm_config(producer=False, model="consumer-model"),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )

    with pytest.raises(ValueError, match="layout fingerprint"):
        consumer.get_num_new_matched_tokens(
            SimpleNamespace(
                request_id="decode-request",
                prompt_token_ids=[*range(10_371), 42],
                kv_transfer_params=transfer_params,
            ),
            0,
        )


def test_consumer_rejects_wrong_router_handoff_token(monkeypatch):
    kv_cache_config = _make_kv_cache_config(monkeypatch)
    producer = DSAKVIOConnector(
        _make_vllm_config(producer=True),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    producer.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=DSAKVIOConnectorWorkerMetadata({
                "prefill-request": {0: {0: [9, 3]}},
            })
        )
    )
    _, transfer_params = producer.request_finished_all_groups(
        SimpleNamespace(
            request_id="prefill-request",
            num_computed_tokens=10_371,
            num_prompt_tokens=10_371,
            output_token_ids=[42],
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        ),
        ([], []),
    )
    consumer = DSAKVIOConnector(
        _make_vllm_config(producer=False),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )

    with pytest.raises(ValueError, match="handoff token mismatch"):
        consumer.get_num_new_matched_tokens(
            SimpleNamespace(
                request_id="decode-request",
                prompt_token_ids=[*range(10_371), 43],
                kv_transfer_params=transfer_params,
            ),
            0,
        )
