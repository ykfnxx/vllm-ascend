import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import standard_run_config as config  # noqa: E402


class TestDmpSchemeRuntimeIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.original_sys_path = sys.path.copy()

    def tearDown(self) -> None:
        sys.path[:] = self.original_sys_path

    @staticmethod
    def _base_environment() -> dict[str, str]:
        return {
            "DMP_FUSED_INDEXER_OPP_PATH": "/test/fused-opp",
            "DMP_FUSED_INDEXER_PYTHON_PATH": "/test/fused-python",
            "DMP_LOOKUP_MAINTAIN_OPP_PATH": "/test/lookup-opp",
            "DMP_LOOKUP_MAINTAIN_PYTHON_PATH": "/test/lookup-python",
            "DMP_DUAL_ATTENTION_OPP_PATH": "/test/dual-opp",
            "DMP_DUAL_ATTENTION_PYTHON_PATH": "/test/dual-python",
            "ASCEND_CUSTOM_OPP_PATH": (
                "/test/fused-opp:/test/lookup-opp/op_impl/aicpu_transformer:"
                "/test/lookup-opp:/test/dual-opp:/keep/opp"
            ),
            "PYTHONPATH": (
                "/test/fused-python:/test/lookup-python:"
                "/test/dual-python:/keep/python"
            ),
            "LD_LIBRARY_PATH": (
                "/test/fused-opp/op_api/lib:/test/lookup-opp/op_api/lib:"
                "/test/dual-opp/op_api/lib:/keep/lib"
            ),
        }

    def test_scheme3_removes_scheme4_aicpu_state(self) -> None:
        environment = self._base_environment()
        environment.update(
            {
                "VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT": "1",
                "VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN": "0",
                "VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION": "0",
            }
        )
        with patch.dict(os.environ, environment, clear=True):
            vendor_root = config.configure_dmp_runtime("0")

            self.assertEqual(vendor_root, "/test/fused-opp")
            self.assertEqual(
                os.environ["ASCEND_CUSTOM_OPP_PATH"],
                "/test/fused-opp:/test/lookup-opp:/test/dual-opp:/keep/opp",
            )
            self.assertEqual(
                os.environ["PYTHONPATH"],
                "/test/fused-python:/test/lookup-python:"
                "/test/dual-python:/keep/python",
            )
            self.assertEqual(
                os.environ["LD_LIBRARY_PATH"],
                f"{config.DEFAULT_VLLM_ASCEND_OP_API_PATH}:/keep/lib",
            )
            self.assertEqual(
                os.environ["VLLM_ASCEND_DMP_SERIALIZE_MAINTAIN"], "0"
            )

    def test_scheme4_removes_scheme3_state(self) -> None:
        environment = self._base_environment()
        environment.update(
            {
                "VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT": "0",
                "VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN": "1",
                "VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION": "0",
            }
        )
        with patch.dict(os.environ, environment, clear=True):
            vendor_root = config.configure_dmp_runtime("0")

            self.assertEqual(vendor_root, "/test/lookup-opp")
            self.assertEqual(
                os.environ["ASCEND_CUSTOM_OPP_PATH"],
                "/test/lookup-opp/op_impl/aicpu_transformer:"
                "/test/lookup-opp:/test/dual-opp:/keep/opp",
            )
            self.assertEqual(
                os.environ["PYTHONPATH"],
                "/test/lookup-python:/test/dual-python:/keep/python",
            )
            self.assertEqual(
                os.environ["LD_LIBRARY_PATH"],
                f"{config.DEFAULT_VLLM_ASCEND_OP_API_PATH}:/keep/lib",
            )
            self.assertEqual(
                os.environ["VLLM_ASCEND_DMP_SERIALIZE_MAINTAIN"], "0"
            )

    def test_scheme4_allows_serial_maintain(self) -> None:
        environment = self._base_environment()
        environment.update(
            {
                "VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT": "0",
                "VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN": "1",
                "VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION": "0",
                "VLLM_ASCEND_DMP_SERIALIZE_MAINTAIN": "1",
            }
        )
        with patch.dict(os.environ, environment, clear=True):
            vendor_root = config.configure_dmp_runtime("0")

            self.assertEqual(vendor_root, "/test/lookup-opp")
            self.assertEqual(
                os.environ["VLLM_ASCEND_DMP_SERIALIZE_MAINTAIN"], "1"
            )

    def test_serial_maintain_requires_scheme4(self) -> None:
        environment = self._base_environment()
        environment.update(
            {
                "VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT": "1",
                "VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN": "0",
                "VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION": "0",
                "VLLM_ASCEND_DMP_SERIALIZE_MAINTAIN": "1",
            }
        )
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(
                ValueError, "requires VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=1"
            ):
                config.configure_dmp_runtime("0")


if __name__ == "__main__":
    unittest.main()
