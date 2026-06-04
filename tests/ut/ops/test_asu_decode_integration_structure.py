from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_model_runner_allocates_asu_manager_after_kv_cache_init():
    model_runner = (
        _repo_root() / "vllm_ascend" / "worker" / "model_runner_v1.py"
    ).read_text()

    assert "ASUFullKVCacheManagerFunctional" in model_runner
    assert "self.asu_kv_cache_manager" in model_runner
    assert "ASUFullKVCacheManagerFunctional.from_kv_caches" in model_runner
    assert "attn_metadata_i.asu_kv_cache_manager" in model_runner


def test_sfa_decode_path_has_asu_resolver_and_asu_sfa_branch():
    sfa_v1 = (_repo_root() / "vllm_ascend" / "attention" /
              "sfa_v1.py").read_text()

    assert "asu_kv_cache_manager" in sfa_v1
    assert "def _use_asu_kv_cache" in sfa_v1
    assert "def _execute_asu_sparse_flash_attention_process" in sfa_v1
    assert "npu_asu_resolve_kv_slots_single_req" in sfa_v1
    assert "npu_sparse_flash_attention_asu" in sfa_v1
    assert "npu_sparse_flash_attention(" in sfa_v1
    assert "AscendAttentionState.DecodeOnly" in sfa_v1
    assert "self.enable_dsa_cp" in sfa_v1
