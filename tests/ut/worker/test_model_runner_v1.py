import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestNPUModelRunnerDMP(unittest.TestCase):

    @staticmethod
    def _metadata(attn_state, num_reqs):
        return {
            "layer": SimpleNamespace(
                attn_state=attn_state,
                cum_query_lens=torch.arange(1, num_reqs + 1),
            )
        }

    @patch("vllm_ascend.worker.model_runner_v1.enable_dsa_cp", return_value=False)
    def test_dmp_graph_eligibility_requires_uniform_decode(self, _mock_enable_dsa_cp):
        runner = NPUModelRunner.__new__(NPUModelRunner)

        self.assertTrue(
            runner._is_dmp_eligible(
                self._metadata(AscendAttentionState.DecodeOnly, 4),
                4,
            )
        )
        self.assertFalse(
            runner._is_dmp_eligible(
                self._metadata(AscendAttentionState.DecodeOnly, 3),
                3,
            )
        )
        self.assertFalse(
            runner._is_dmp_eligible(
                self._metadata(AscendAttentionState.DecodeOnly, 2),
                4,
            )
        )
        self.assertFalse(
            runner._is_dmp_eligible(
                self._metadata(AscendAttentionState.SpecDecoding, 4),
                4,
            )
        )

    def test_dmp_graph_context_is_reused(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._dmp_graph_contexts = {}
        runner.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=4)
        )
        expected_context = MagicMock()
        runner._maybe_create_dmp_slices = MagicMock(return_value=expected_context)
        batch_descriptor = object()
        attn_metadata = object()

        first = runner._get_or_create_dmp_graph_context(
            batch_descriptor,
            attn_metadata,
            4,
        )
        second = runner._get_or_create_dmp_graph_context(
            batch_descriptor,
            attn_metadata,
            4,
        )

        self.assertIs(first, expected_context)
        self.assertIs(second, expected_context)
        runner._maybe_create_dmp_slices.assert_called_once_with(
            attn_metadata,
            4,
            None,
        )
        expected_context.prepare_graph_events.assert_called_once_with(4)


if __name__ == "__main__":
    unittest.main()
