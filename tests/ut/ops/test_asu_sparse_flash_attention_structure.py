from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_sparse_flash_attention_asu_variant_is_registered():
    repo_root = _repo_root()
    torch_binding = (repo_root / "csrc" / "torch_binding.cpp").read_text()
    torch_binding_meta = (
        repo_root / "csrc" / "torch_binding_meta.cpp"
    ).read_text()
    adapter = (
        repo_root / "csrc" / "sparse_flash_attention" /
        "sparse_flash_attention_torch_adpt.h"
    ).read_text()

    assert "npu_sparse_flash_attention_asu" in torch_binding
    assert "resolved_kv_slots" in torch_binding
    assert "managed_key_rope" in torch_binding
    assert "npu_sparse_flash_attention_asu_meta" in torch_binding_meta
    assert "npu_sparse_flash_attention_asu" in adapter
    assert "aclnnSparseFlashAttentionAsu" in adapter


def test_sparse_flash_attention_asu_variant_has_cann_host_and_kernel_inputs():
    repo_root = _repo_root()
    op_root = repo_root / "csrc" / "sparse_flash_attention"
    op_def = (op_root / "op_host" / "sparse_flash_attention_def.cpp").read_text()
    op_proto = (
        op_root / "op_host" / "sparse_flash_attention_proto.cpp"
    ).read_text()
    tiling_header = (
        op_root / "op_host" / "sparse_flash_attention_tiling.h"
    ).read_text()
    kernel = (op_root / "op_kernel" / "sparse_flash_attention.cpp").read_text()
    vector_service = (
        op_root / "op_kernel" / "sparse_flash_attention_service_vector_mla.h"
    ).read_text()
    cube_service = (
        op_root / "op_kernel" / "sparse_flash_attention_service_cube_mla.h"
    ).read_text()

    assert "class SparseFlashAttentionAsu" in op_def
    assert 'this->Input("resolved_kv_slots")' in op_def
    assert 'this->Input("managed_key_rope")' in op_def
    assert "IMPL_OP(SparseFlashAttentionAsu)" in op_proto
    assert "RESOLVED_KV_SLOTS_INPUT_INDEX" in tiling_header
    assert "MANAGED_KEY_ROPE_INPUT_INDEX" in tiling_header
    assert "sparse_flash_attention_asu" in kernel
    assert "resolvedKvSlots" in kernel
    assert "GetResolvedSlot" in vector_service
    assert "resolvedSlotsGm" in cube_service
