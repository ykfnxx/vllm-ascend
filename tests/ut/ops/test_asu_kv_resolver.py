import importlib.util
from pathlib import Path

import torch


def _load_asu_kv_resolver_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "vllm_ascend" / "ops" / "asu_kv_resolver.py"
    spec = importlib.util.spec_from_file_location("asu_kv_resolver", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


asu_kv_resolver = _load_asu_kv_resolver_module()
ASU_ONLY = asu_kv_resolver.ASU_ONLY
HBM_RESIDENT = asu_kv_resolver.HBM_RESIDENT
INVALID = asu_kv_resolver.INVALID
TAIL_HBM = asu_kv_resolver.TAIL_HBM
asu_resolve_kv_slots_single_req = (
    asu_kv_resolver.asu_resolve_kv_slots_single_req
)


def test_resolver_installs_asu_and_tail_tokens_into_managed_cache():
    original_topk_indices = torch.tensor([[[1, 1, 5]]], dtype=torch.int64)
    original_kv_cache_0 = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    original_kv_cache_1 = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    managed_kv_cache_0 = torch.full((4, 2), -1.0)
    managed_kv_cache_1 = torch.full((4, 3), -1.0)
    token_state = torch.tensor(
        [ASU_ONLY, ASU_ONLY, ASU_ONLY, ASU_ONLY, TAIL_HBM, TAIL_HBM],
        dtype=torch.int32,
    )
    asu_record_addr = torch.tensor([0, 4, 0, 0, -1, -1], dtype=torch.int64)
    hbm_slot_of_token = torch.full((6,), -1, dtype=torch.int64)
    slot_owner_token = torch.full((4,), INVALID, dtype=torch.int64)
    free_slot_stack = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    free_slot_count = torch.tensor(4, dtype=torch.int64)
    original_block_table = torch.tensor([2, 0, 3], dtype=torch.int64)

    resolved_kv_slots = asu_resolve_kv_slots_single_req(
        original_topk_indices,
        actual_seq_len=6,
        managed_prefix_len=4,
        token_state=token_state,
        asu_record_addr=asu_record_addr,
        hbm_slot_of_token=hbm_slot_of_token,
        slot_owner_token=slot_owner_token,
        free_slot_stack=free_slot_stack,
        free_slot_count=free_slot_count,
        original_block_table=original_block_table,
        original_kv_cache_0=original_kv_cache_0,
        original_kv_cache_1=original_kv_cache_1,
        managed_kv_cache_0=managed_kv_cache_0,
        managed_kv_cache_1=managed_kv_cache_1,
        block_size=2,
    )

    assert resolved_kv_slots.tolist() == [[[3, 3, 2]]]
    assert free_slot_count.item() == 2
    assert token_state.tolist() == [
        ASU_ONLY,
        HBM_RESIDENT,
        ASU_ONLY,
        ASU_ONLY,
        TAIL_HBM,
        HBM_RESIDENT,
    ]
    assert hbm_slot_of_token.tolist() == [-1, 3, -1, -1, -1, 2]
    assert slot_owner_token.tolist() == [INVALID, INVALID, 5, 1]
    torch.testing.assert_close(managed_kv_cache_0[3], original_kv_cache_0[4])
    torch.testing.assert_close(managed_kv_cache_1[3], original_kv_cache_1[4])
    torch.testing.assert_close(managed_kv_cache_0[2], original_kv_cache_0[7])
    torch.testing.assert_close(managed_kv_cache_1[2], original_kv_cache_1[7])


def test_resolver_reuses_resident_slots_without_consuming_free_slots():
    original_topk_indices = torch.tensor([[2]], dtype=torch.int32)
    original_kv_cache_0 = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    original_kv_cache_1 = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    managed_kv_cache_0 = torch.full((3, 2), -1.0)
    managed_kv_cache_1 = torch.full((3, 3), -1.0)
    managed_kv_cache_0[1] = torch.tensor([101.0, 102.0])
    managed_kv_cache_1[1] = torch.tensor([201.0, 202.0, 203.0])
    token_state = torch.tensor([ASU_ONLY, ASU_ONLY, HBM_RESIDENT],
                               dtype=torch.int32)
    asu_record_addr = torch.tensor([0, 1, 2], dtype=torch.int64)
    hbm_slot_of_token = torch.tensor([-1, -1, 1], dtype=torch.int64)
    slot_owner_token = torch.tensor([INVALID, 2, INVALID], dtype=torch.int64)
    free_slot_stack = torch.tensor([0, 2, -1], dtype=torch.int64)
    free_slot_count = torch.tensor(2, dtype=torch.int64)
    original_block_table = torch.tensor([0, 1], dtype=torch.int64)

    resolved_kv_slots = asu_resolve_kv_slots_single_req(
        original_topk_indices,
        actual_seq_len=3,
        managed_prefix_len=3,
        token_state=token_state,
        asu_record_addr=asu_record_addr,
        hbm_slot_of_token=hbm_slot_of_token,
        slot_owner_token=slot_owner_token,
        free_slot_stack=free_slot_stack,
        free_slot_count=free_slot_count,
        original_block_table=original_block_table,
        original_kv_cache_0=original_kv_cache_0,
        original_kv_cache_1=original_kv_cache_1,
        managed_kv_cache_0=managed_kv_cache_0,
        managed_kv_cache_1=managed_kv_cache_1,
        block_size=2,
    )

    assert resolved_kv_slots.tolist() == [[1]]
    assert free_slot_count.item() == 2
    torch.testing.assert_close(managed_kv_cache_0[1],
                               torch.tensor([101.0, 102.0]))
    torch.testing.assert_close(managed_kv_cache_1[1],
                               torch.tensor([201.0, 202.0, 203.0]))
