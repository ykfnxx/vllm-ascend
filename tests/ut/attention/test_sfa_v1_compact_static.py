from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]


class AscendSFACompactStaticTest(unittest.TestCase):

    def test_forward_uses_compact_sfa_inputs_between_indexer_and_sfa(self):
        source = (REPO_ROOT / "vllm_ascend/attention/sfa_v1.py").read_text()

        indexer_pos = source.index("topk_indices = self.indexer_select_post_process")
        compact_pos = source.index("prepare_compact_sfa_inputs")
        sfa_pos = source.index("attn_output = self._execute_sparse_flash_attention_process")

        self.assertLess(indexer_pos, compact_pos)
        self.assertLess(compact_pos, sfa_pos)
        self.assertIn("sfa_topk_indices", source)
        self.assertIn("sfa_attn_metadata", source)

    def test_compact_sfa_envs_are_declared(self):
        source = (REPO_ROOT / "vllm_ascend/envs.py").read_text()

        self.assertIn("VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA", source)
        self.assertIn("VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS", source)

    def test_model_runner_passes_compact_config_to_manager(self):
        source = (REPO_ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text()

        self.assertIn("envs.VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA", source)
        self.assertIn("compact_sfa_enabled=envs.VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA", source)
        self.assertIn("max_pinned_reqs=envs.VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS", source)
        self.assertIn("register_static_offload_block_pool", source)
        self.assertIn("scheduler_output.finished_req_ids", source)
        self.assertIn("release_request(req_id)", source)


if __name__ == "__main__":
    unittest.main()
