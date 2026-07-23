import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.request import RequestStatus

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("mooncake.engine", fake_engine)

from vllm_ascend.distributed.kv_transfer import (  # noqa: E402
    dsa_kvio_connector as kvio_connector_module,
)
from vllm_ascend.distributed.kv_transfer import (  # noqa: E402
    dsa_mooncake_connector as connector_module,
)
from vllm_ascend.distributed.kv_transfer.dsa_kvio_connector import (  # noqa: E402
    DSAKVIOConnectorWorkerMetadata,
)
from vllm_ascend.distributed.kv_transfer.dsa_mooncake_connector import (  # noqa: E402
    DSAMooncakeConnector,
    DSAMooncakeConnectorMetadata,
    _DSAMooncakeCacheRegion,
    _append_contiguous_range_mapping,
    _append_token_mapping,
)
from vllm_ascend.dsa_sparse.dsa_pd import (  # noqa: E402
    DSA_PD_INITIAL_TRANSPORT_KEY,
    DSA_PD_INITIAL_TRANSPORT_MOONCAKE,
)


def _make_vllm_config(*, producer: bool):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=128,
            cache_dtype="auto",
            dsa_kv_backend="kvio",
            dsa_kvio_model_id=7,
            enable_dsa_sparse_cache=True,
        ),
        model_config=SimpleNamespace(
            model="test-model",
            revision="test-revision",
            dtype="float16",
            max_model_len=131_072,
            get_total_num_hidden_layers=lambda: 1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=producer,
            is_kv_consumer=not producer,
            engine_id=("prefill-engine" if producer else "decode-engine"),
            kv_port=31_000,
        ),
        parallel_config=SimpleNamespace(
            rank=0,
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
        ),
    )


def _make_kv_cache_config(monkeypatch):
    for module in (kvio_connector_module, connector_module):
        monkeypatch.setattr(
            module,
            "is_dsa_indexer_spec",
            lambda spec: spec == "indexer",
        )
        monkeypatch.setattr(
            module,
            "is_dsa_mla_resident_spec",
            lambda spec: spec == "resident",
        )
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec="indexer"),
            SimpleNamespace(kv_cache_spec="resident"),
        ]
    )


def test_scheduler_builds_mooncake_handoff_and_delays_p_blocks(monkeypatch):
    kv_cache_config = _make_kv_cache_config(monkeypatch)
    producer = DSAMooncakeConnector(
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
    p_request = SimpleNamespace(
        request_id="prefill-request",
        num_computed_tokens=10_371,
        num_prompt_tokens=10_371,
        output_token_ids=[42],
        kv_transfer_params={
            "do_remote_decode": True,
            "do_remote_prefill": False,
        },
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    prompt_blocks = 82
    delay_free, params = producer.request_finished_all_groups(
        p_request,
        (
            list(range(100, 100 + prompt_blocks)),
            list(range(200, 200 + prompt_blocks)),
        ),
    )

    assert delay_free is True
    assert params is not None
    assert params[DSA_PD_INITIAL_TRANSPORT_KEY] == (
        DSA_PD_INITIAL_TRANSPORT_MOONCAKE
    )
    assert params["remote_block_ids"][0] == list(
        range(100, 100 + prompt_blocks)
    )
    assert params["remote_block_ids"][1] == list(
        range(200, 200 + prompt_blocks)
    )
    assert params["remote_engine_id"] == "prefill-engine"
    assert params["remote_request_id"] == "prefill-request"
    assert params["remote_port"] == 31_000

    send_metadata = producer.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={})
    )
    assert isinstance(send_metadata, DSAMooncakeConnectorMetadata)
    assert set(send_metadata.requests_to_send) == {"prefill-request"}

    consumer = DSAMooncakeConnector(
        _make_vllm_config(producer=False),
        KVConnectorRole.SCHEDULER,
        kv_cache_config,
    )
    d_request = SimpleNamespace(
        request_id="decode-request",
        prompt_token_ids=[*range(10_371), 42],
        kv_transfer_params=params,
    )
    assert consumer.get_num_new_matched_tokens(d_request, 0) == (
        10_371,
        False,
    )
    local_blocks = SimpleNamespace(
        get_block_ids=lambda: [
            list(range(prompt_blocks)),
            list(range(81)),
        ]
    )
    consumer.update_state_after_alloc(
        d_request, local_blocks, 10_371
    )
    recv_metadata = consumer.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={"decode-request": 1})
    )
    pd_request = recv_metadata.dsa_requests[0]
    assert pd_request.initial_transport == (
        DSA_PD_INITIAL_TRANSPORT_MOONCAKE
    )
    assert pd_request.remote_indexer_block_ids == list(
        range(100, 100 + prompt_blocks)
    )
    assert pd_request.remote_resident_block_ids == list(
        range(200, 200 + prompt_blocks)
    )
    assert pd_request.remote_request_id == "prefill-request"


def test_address_plan_coalesces_dense_ranges_but_not_sparse_reordering():
    tensor = torch.zeros((4, 2, 2), dtype=torch.uint8)
    region = _DSAMooncakeCacheRegion(
        layer_id=0,
        component="indexer",
        tensor=tensor,
        num_blocks=4,
        block_bytes=4,
        token_bytes=2,
    )
    transfers = []
    _append_contiguous_range_mapping(
        transfers,
        local_region=region,
        remote_base_address=10_000,
        remote_block_ids=[3, 4],
        local_block_ids=[1, 2],
        source_token_start=0,
        destination_slot_start=0,
        token_count=4,
        block_size=2,
    )

    assert len(transfers) == 1
    assert transfers[0].local_address == tensor.data_ptr() + 4
    assert transfers[0].remote_address == 10_000 + 12
    assert transfers[0].length == 8

    sparse_transfers = []
    _append_token_mapping(
        sparse_transfers,
        local_region=region,
        remote_base_address=10_000,
        remote_block_ids=[3, 4],
        local_block_ids=[1, 2],
        source_token_ids=[3, 0],
        destination_slots=[0, 1],
        block_size=2,
    )

    assert len(sparse_transfers) == 2


def test_address_plan_rejects_short_remote_block_table():
    tensor = torch.zeros((4, 2, 2), dtype=torch.uint8)
    region = _DSAMooncakeCacheRegion(
        layer_id=0,
        component="indexer",
        tensor=tensor,
        num_blocks=4,
        block_bytes=4,
        token_bytes=2,
    )

    with pytest.raises(RuntimeError, match="remote block table"):
        _append_token_mapping(
            [],
            local_region=region,
            remote_base_address=10_000,
            remote_block_ids=[],
            local_block_ids=[1],
            source_token_ids=[0],
            destination_slots=[0],
            block_size=2,
        )
