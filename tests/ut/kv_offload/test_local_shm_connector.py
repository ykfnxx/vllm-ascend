# SPDX-License-Identifier: Apache-2.0

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm_ascend.attention.dsa_sparse_pd import DSASparsePDHandoff  # noqa: E402
from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.local_shm_connector import (  # noqa: E402
    LocalShmConnectorWorker,
    LocalShmSendSpec,
    _cache_planes,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import ReqMeta  # noqa: E402
from vllm_ascend.dsa_sparse_constants import DSA_SPARSE_QUERY_WIDTH  # noqa: E402


def _worker(shm_dir: Path, *, engine_id: str) -> LocalShmConnectorWorker:
    worker = LocalShmConnectorWorker.__new__(LocalShmConnectorWorker)
    worker.engine_id = engine_id
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.num_blocks = 4
    worker.block_size = 2
    worker.shm_dir = shm_dir
    worker.timeout = 0.1
    worker._layer_group_ids = {"indexer": 0}
    worker._group_tokens_per_block = {0: 2}
    worker._dsa_sparse_tail_layouts = {}
    worker._dsa_sparse_request_rows = {}
    return worker


def test_rank_local_mmap_transfers_scheduler_kv_and_packed_c8_tail(tmp_path: Path):
    packed_head_dim = get_sfa_qsfa_packed_head_dim(512, 64)
    source_kv = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    source_main = (
        torch.arange(4 * 2 * packed_head_dim, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(4, 2, 1, packed_head_dim)
        .view(torch.float8_e4m3fn)
    )
    destination_kv = torch.zeros_like(source_kv)
    destination_hot = torch.zeros((2, 2, 1, packed_head_dim), dtype=torch.float8_e4m3fn)

    handoff = DSASparsePDHandoff(
        remote_request_id="remote-request",
        stored_token_count=3,
        block_size=2,
        layer_topk_by_rank={0: {"main": list(range(DSA_SPARSE_QUERY_WIDTH))}},
        partial_tail_blocks_by_rank={0: {"main": 1}},
    )
    send_spec = LocalShmSendSpec(
        remote_block_ids=([2, 0],),
        dsa_sparse_handoff=handoff,
    )

    producer = _worker(tmp_path, engine_id="prefill")
    producer.kv_caches = {"indexer": (source_kv,)}
    producer._dsa_sparse_aux_caches = {"main": (source_main,)}
    producer._publish("remote-request", send_spec)

    consumer = _worker(tmp_path, engine_id="decode")
    consumer.kv_caches = {"indexer": (destination_kv,)}
    consumer._dsa_sparse_aux_caches = {"main": (destination_hot,)}
    consumer._dsa_sparse_tail_layouts = {"main": (2, 1)}
    consumer._dsa_sparse_request_rows = {"local-request": 0}
    meta = ReqMeta(
        local_block_ids=([1, 3],),
        local_full_block_ids=([1, 3],),
        num_external_tokens=4,
        num_computed_tokens=0,
        remote_block_ids=([2, 0],),
        remote_host="127.0.0.1",
        remote_port=0,
        remote_engine_id="prefill",
        remote_request_id="remote-request",
        remote_pcp_size=1,
        remote_dcp_size=1,
        remote_ptp_size=1,
        remote_multi_nodes_meta_mapping={},
        num_prompt_blocks=2,
        remote_block_size=2,
        dsa_sparse_handoff=handoff,
    )
    consumer._load("local-request", meta)

    torch.testing.assert_close(destination_kv[1], source_kv[2])
    torch.testing.assert_close(destination_kv[3], source_kv[0])
    assert torch.equal(
        destination_hot[1, 0].view(torch.uint8),
        source_main[1, 0].view(torch.uint8),
    )
    assert not list(tmp_path.iterdir())


def test_manifest_rejects_wrong_tp_rank(tmp_path: Path):
    consumer = _worker(tmp_path, engine_id="decode")
    meta = types.SimpleNamespace(
        remote_engine_id="prefill",
        remote_request_id="request",
        remote_block_size=2,
    )
    manifest = {
        "version": 1,
        "engine_id": "prefill",
        "request_id": "request",
        "tp_rank": 1,
        "tp_size": 1,
        "block_size": 2,
    }

    with pytest.raises(RuntimeError, match="tp_rank"):
        consumer._validate_manifest(manifest, meta)


def test_payload_plane_validation_rejects_missing_plane(tmp_path: Path):
    consumer = _worker(tmp_path, engine_id="decode")
    consumer.kv_caches = {}
    consumer._dsa_sparse_aux_caches = {
        "main": (
            torch.empty((2, 2, 1, 13), dtype=torch.uint8),
            torch.empty((2, 2, 1), dtype=torch.float32),
        )
    }
    consumer._dsa_sparse_tail_layouts = {"main": (2, 1)}
    handoff = DSASparsePDHandoff(
        remote_request_id="remote-request",
        stored_token_count=3,
        block_size=2,
        layer_topk_by_rank={0: {"main": list(range(DSA_SPARSE_QUERY_WIDTH))}},
        partial_tail_blocks_by_rank={0: {"main": 1}},
    )
    meta = types.SimpleNamespace(
        remote_block_ids=([],),
        dsa_sparse_handoff=handoff,
    )
    manifest = {
        "records": [
            {
                "kind": "tail",
                "layer_name": "main",
                "plane_index": 0,
            }
        ]
    }

    with pytest.raises(RuntimeError, match="plane layouts do not match"):
        consumer._validate_payload_plane_layouts(manifest, meta)


def test_cache_planes_rejects_empty_cache():
    with pytest.raises(ValueError, match="at least one tensor plane"):
        _cache_planes(())
