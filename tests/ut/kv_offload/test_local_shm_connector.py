# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm_ascend.distributed.kv_transfer.kv_p2p.local_shm_connector import (  # noqa: E402
    LocalShmConnectorWorker,
    LocalShmSendSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import ReqMeta  # noqa: E402
from vllm_ascend.dsa_offload.constants import QUERY_WIDTH  # noqa: E402
from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout  # noqa: E402
from vllm_ascend.dsa_offload.pd import DSAOffloadPDHandoff  # noqa: E402


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
    worker._dsa_offload_aux_caches = {}
    worker._dsa_offload_layout = None
    worker._dsa_offload_request_rows = {}
    return worker


def test_rank_local_mmap_transfers_scheduler_kv_and_partial_tail(tmp_path: Path):
    source_kv = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    source_main = torch.arange(100, 124, dtype=torch.float32).reshape(4, 2, 3)
    destination_kv = torch.zeros_like(source_kv)
    layout = HotCacheLayout(2, 1, 0)
    destination_hot = torch.zeros((layout.hot_blocks, 2, 3), dtype=torch.float32)

    handoff = DSAOffloadPDHandoff(
        remote_request_id="remote-request",
        stored_token_count=3,
        block_size=2,
        layer_topk_by_rank={0: {"main": list(range(QUERY_WIDTH))}},
        partial_tail_blocks_by_rank={0: {"main": 1}},
    )
    send_spec = LocalShmSendSpec(
        remote_block_ids=([2, 0],),
        dsa_offload_handoff=handoff,
    )

    producer = _worker(tmp_path, engine_id="prefill")
    producer.kv_caches = {"indexer": (source_kv,)}
    producer._dsa_offload_aux_caches = {"main": (source_main,)}
    producer._publish("remote-request", send_spec)

    consumer = _worker(tmp_path, engine_id="decode")
    consumer.kv_caches = {"indexer": (destination_kv,)}
    consumer._dsa_offload_aux_caches = {"main": (destination_hot,)}
    consumer._dsa_offload_layout = layout
    consumer._dsa_offload_request_rows = {"local-request": 0}
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
        dsa_offload_handoff=handoff,
    )
    consumer._load("local-request", meta)

    torch.testing.assert_close(destination_kv[1], source_kv[2])
    torch.testing.assert_close(destination_kv[3], source_kv[0])
    tail_block = layout.tail_block_offset
    torch.testing.assert_close(destination_hot[tail_block, 0], source_main[1, 0])
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
