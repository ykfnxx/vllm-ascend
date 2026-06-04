import importlib.util
from pathlib import Path

import torch


def _load_asu_kv_cache_manager_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root / "vllm_ascend" / "attention" /
        "asu_kv_cache_manager.py"
    )
    spec = importlib.util.spec_from_file_location("asu_kv_cache_manager",
                                                  module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_asu_kv_cache_feature_gate_defaults_off(monkeypatch):
    module = _load_asu_kv_cache_manager_module()

    monkeypatch.delenv("VLLM_ASCEND_ENABLE_ASU_KV_CACHE", raising=False)
    assert module.is_asu_kv_cache_enabled() is False

    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ASU_KV_CACHE", "1")
    assert module.is_asu_kv_cache_enabled() is True


def test_manager_allocates_per_layer_managed_cache_from_pa_cache():
    module = _load_asu_kv_cache_manager_module()
    original_kv_caches = {
        "layers.0.self_attn": (
            torch.zeros((2, 4, 1, 8), dtype=torch.bfloat16),
            torch.zeros((2, 4, 1, 2), dtype=torch.bfloat16),
            torch.zeros((2, 4, 1, 4), dtype=torch.bfloat16),
        ),
        "layers.1.self_attn": (
            torch.zeros((2, 4, 1, 8), dtype=torch.bfloat16),
            torch.zeros((2, 4, 1, 2), dtype=torch.bfloat16),
            torch.zeros((2, 4, 1, 4), dtype=torch.bfloat16),
        ),
    }

    manager = module.ASUFullKVCacheManagerFunctional.from_kv_caches(
        original_kv_caches,
        max_seq_len=8,
        block_size=4,
        managed_slot_count=6,
        enabled=True,
    )

    layer0 = manager.get_layer_context("layers.0.self_attn")
    layer1 = manager.get_layer_context("layers.1.self_attn")
    assert layer0.managed_kv_cache_0.shape == (6, 1, 8)
    assert layer0.managed_kv_cache_1.shape == (6, 1, 2)
    assert layer0.managed_kv_cache_0.dtype is torch.bfloat16
    assert layer1.managed_kv_cache_0.data_ptr() != layer0.managed_kv_cache_0.data_ptr()
    assert layer0.free_slot_count.item() == 6
    assert layer0.free_slot_stack.tolist() == [0, 1, 2, 3, 4, 5]


def test_manager_resets_single_request_state_before_decode():
    module = _load_asu_kv_cache_manager_module()
    original_kv_caches = {
        "layers.0.self_attn": (
            torch.zeros((3, 4, 1, 8), dtype=torch.float16),
            torch.zeros((3, 4, 1, 2), dtype=torch.float16),
            torch.zeros((3, 4, 1, 4), dtype=torch.float16),
        ),
    }
    manager = module.ASUFullKVCacheManagerFunctional.from_kv_caches(
        original_kv_caches,
        max_seq_len=12,
        block_size=4,
        managed_slot_count=12,
        enabled=True,
    )
    original_block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)

    manager.reset_single_request(
        req_id="req-1",
        actual_seq_len=6,
        managed_prefix_len=4,
        original_block_table=original_block_table,
    )

    layer = manager.get_layer_context("layers.0.self_attn")
    assert layer.req_id == "req-1"
    assert layer.actual_seq_len == 6
    assert layer.managed_prefix_len == 4
    assert layer.token_state[:8].tolist() == [
        module.ASU_ONLY,
        module.ASU_ONLY,
        module.ASU_ONLY,
        module.ASU_ONLY,
        module.TAIL_HBM,
        module.TAIL_HBM,
        module.INVALID,
        module.INVALID,
    ]
    assert layer.asu_record_addr[:4].tolist() == [8, 9, 10, 11]
    assert layer.free_slot_count.item() == 12
    assert layer.slot_owner_token[:3].tolist() == [
        module.INVALID,
        module.INVALID,
        module.INVALID,
    ]
