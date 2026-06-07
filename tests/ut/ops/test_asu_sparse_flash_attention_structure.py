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
        repo_root / "csrc" / "sparse_flash_attention_asu" /
        "sparse_flash_attention_asu_torch_adpt.h"
    ).read_text()
    original_adapter = (
        repo_root / "csrc" / "sparse_flash_attention" /
        "sparse_flash_attention_torch_adpt.h"
    ).read_text()

    assert "sparse_flash_attention_asu/sparse_flash_attention_asu_torch_adpt.h" in torch_binding
    assert "npu_sparse_flash_attention_asu" in torch_binding
    assert "resolved_kv_slots" in torch_binding
    assert "managed_key_rope" in torch_binding
    assert "npu_sparse_flash_attention_asu_meta" in torch_binding_meta
    assert "npu_sparse_flash_attention_asu" in adapter
    assert "aclnnSparseFlashAttentionAsu" in adapter
    assert "npu_sparse_flash_attention_asu" not in original_adapter


def test_sparse_flash_attention_asu_variant_has_cann_host_and_kernel_inputs():
    repo_root = _repo_root()
    asu_root = repo_root / "csrc" / "sparse_flash_attention_asu"
    asu_cmake = (asu_root / "op_host" / "CMakeLists.txt").read_text()
    op_def = (
        asu_root / "op_host" / "sparse_flash_attention_asu_def.cpp"
    ).read_text()
    op_proto = (
        asu_root / "op_host" / "sparse_flash_attention_asu_proto.cpp"
    ).read_text()
    asu_tiling = (
        asu_root / "op_host" / "sparse_flash_attention_asu_tiling.cpp"
    ).read_text()
    tiling_header = (
        asu_root / "op_host" / "sparse_flash_attention_asu_tiling.h"
    ).read_text()
    tiling_key = (
        asu_root / "op_kernel" / "sparse_flash_attention_template_tiling_key.h"
    ).read_text()
    kernel = (asu_root / "op_kernel" / "sparse_flash_attention.cpp").read_text()
    vector_service = (
        asu_root / "op_kernel" / "sparse_flash_attention_service_vector_mla.h"
    ).read_text()
    cube_service = (
        asu_root / "op_kernel" / "sparse_flash_attention_service_cube_mla.h"
    ).read_text()
    build_aclnn = (repo_root / "csrc" / "build_aclnn.sh").read_text()

    assert "OP_NAME SparseFlashAttentionAsu" in asu_cmake
    assert "sparse_flash_attention_asu_depends" not in asu_cmake
    assert "../../sparse_flash_attention" not in asu_cmake
    assert "class SparseFlashAttentionAsu" in op_def
    assert 'this->Input("resolved_kv_slots")' in op_def
    assert 'this->Input("managed_key_rope")' in op_def
    assert "IMPL_OP(SparseFlashAttentionAsu)" in op_proto
    assert "IMPL_OP_OPTILING(SparseFlashAttentionAsu)" in asu_tiling
    assert "IMPL_OP_OPTILING(SparseFlashAttention)" not in asu_tiling
    assert "REGISTER_TILING_DATA_CLASS(SparseFlashAttention," not in tiling_header
    assert "RESOLVED_KV_SLOTS_INPUT_INDEX" in tiling_header
    assert "MANAGED_KEY_ROPE_INPUT_INDEX" in tiling_header
    assert "ASCENDC_TPL_ARGS_DECL(SparseFlashAttentionAsu" in tiling_key
    assert "sparse_flash_attention_asu" in kernel
    assert "sparse_flash_attention(__gm__" not in kernel
    assert "resolvedKvSlots" in kernel
    assert "GetResolvedSlot" in vector_service
    assert "resolvedSlotsGm" in cube_service
    assert "sparse_flash_attention_asu" in build_aclnn


def test_original_sparse_flash_attention_has_no_asu_variant_changes():
    sfa_root = _repo_root() / "csrc" / "sparse_flash_attention"
    source_text = "\n".join(
        path.read_text()
        for path in sfa_root.rglob("*")
        if path.suffix in {".cpp", ".h"}
    )

    assert "SparseFlashAttentionAsu" not in source_text
    assert "sparse_flash_attention_asu" not in source_text
    assert "resolved_kv_slots" not in source_text
    assert "resolvedKvSlots" not in source_text
    assert "MANAGED_KEY_ROPE_INPUT_INDEX" not in source_text
    assert "asuResolved" not in source_text
