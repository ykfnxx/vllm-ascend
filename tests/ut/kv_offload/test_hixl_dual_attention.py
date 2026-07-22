import json

import pytest

from vllm_ascend.kv_offload.hixl_dual_attention import (
    HixlBackendConfig,
    _channel_partitions,
    _ready_group_sqe_capacity,
)


def test_hixl_config_requires_endpoint_fields(tmp_path):
    config_path = tmp_path / "hixl.json"
    config_path.write_text(json.dumps({"npu_phy_dev": 5}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="npu_ip, kernel_json"):
        HixlBackendConfig.from_json(str(config_path))


def test_hixl_config_preserves_local_and_ssu_endpoints(tmp_path):
    config_path = tmp_path / "hixl.json"
    config_path.write_text(
        json.dumps(
            {
                "npu_phy_dev": 5,
                "npu_ip": "192.168.1.5",
                "ssu_phy_dev": 6,
                "ssu_ip": "192.168.1.6",
                "kernel_json": "/tmp/kernel.json",
                "source_mode": "synthetic",
            }
        ),
        encoding="utf-8",
    )

    config = HixlBackendConfig.from_json(str(config_path))

    assert config.npu_phy_dev == 5
    assert config.ssu_phy_dev == 6
    assert config.source_mode == "synthetic"


def test_ready_group_capacity_covers_parallel_channel_layout():
    _, planned_counts, _ = _channel_partitions(
        total_slots=32 * 2048,
        slots_per_sqe=55,
        channel_count=7,
    )
    capacity = _ready_group_sqe_capacity(
        batch_size=32,
        topk=2048,
        slots_per_sqe=55,
        channel_count=7,
        group_size=12,
    )

    assert capacity >= sum(planned_counts)
    assert capacity % 7 == 0
