from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestAsuHbmIndexCsrcWiring(unittest.TestCase):

    def test_lookup_operator_sources_live_under_vllm_ascend_csrc(self):
        lookup_dir = REPO_ROOT / "csrc" / "asu_hbm_index_lookup"

        self.assertTrue((lookup_dir / "asu_hbm_index_lookup_torch_adpt.h").is_file())
        self.assertTrue((lookup_dir / "op_host" / "CMakeLists.txt").is_file())
        self.assertTrue((lookup_dir / "op_host" / "asu_hbm_index_lookup_def.cpp").is_file())
        self.assertTrue((lookup_dir / "op_host" / "asu_hbm_index_lookup_proto.cpp").is_file())
        self.assertTrue((lookup_dir / "op_host" / "asu_hbm_index_lookup_tiling.cpp").is_file())
        self.assertTrue((lookup_dir / "op_kernel" / "asu_hbm_index_lookup.cpp").is_file())

    def test_maintain_aicpu_sources_live_under_vllm_ascend_csrc(self):
        maintain_dir = REPO_ROOT / "csrc" / "asu_hbm_index_maintain_aicpu"

        self.assertFalse((maintain_dir / "CMakeLists.txt").exists())
        self.assertFalse((maintain_dir / "build.sh").exists())
        self.assertTrue((maintain_dir / "op_host" / "CMakeLists.txt").is_file())
        self.assertTrue((maintain_dir / "op_host" / "asu_hbm_index_maintain_aicpu_def.cpp").is_file())
        self.assertTrue((maintain_dir / "op_host" / "asu_hbm_index_maintain_aicpu_proto.cpp").is_file())
        self.assertTrue((maintain_dir / "asu_hbm_index_maintain_aicpu_torch_adpt.h").is_file())
        self.assertTrue((maintain_dir / "op_kernel" / "asu_hbm_index_maintain_aicpu.cpp").is_file())
        self.assertTrue((maintain_dir / "op_kernel" / "asu_hbm_index_maintain_aicpu_kernel.aicpu").is_file())
        self.assertTrue((maintain_dir / "README.md").is_file())

    def test_torch_binding_registers_lookup_and_aicpu_maintain_only(self):
        binding = (REPO_ROOT / "csrc" / "torch_binding.cpp").read_text()

        self.assertIn('#include "asu_hbm_index_lookup/asu_hbm_index_lookup_torch_adpt.h"', binding)
        self.assertIn(
            '#include "asu_hbm_index_maintain_aicpu/asu_hbm_index_maintain_aicpu_torch_adpt.h"',
            binding,
        )
        self.assertIn("asu_hbm_index_lookup(Tensor(a!) index", binding)
        self.assertIn("&vllm_ascend::asu_hbm_index_lookup", binding)
        self.assertIn("asu_hbm_index_maintain_aicpu(Tensor(a!) index", binding)
        self.assertIn("&vllm_ascend::asu_hbm_index_maintain_aicpu", binding)
        self.assertNotIn("asu_hbm_index_maintain(Tensor(a!) index", binding)
        self.assertNotIn("&vllm_ascend::asu_hbm_index_maintain);", binding)


if __name__ == "__main__":
    unittest.main()
