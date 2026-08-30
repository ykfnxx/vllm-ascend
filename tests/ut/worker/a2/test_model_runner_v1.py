import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadataBuilder,
)
from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.dsa_offload.external_main import (
    add_decode_external_main_cache,
)
from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerAcceptedTokens(unittest.TestCase):
    @patch("vllm_ascend.worker.model_runner_v1.mamba_utils.postprocess_mamba_align_gpu")
    def test_postprocess_writes_accepted_counts_to_independent_snapshot(self, mock_postprocess):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_async_scheduling = True
        runner.speculative_config = object()
        runner.model_config = SimpleNamespace(is_hybrid=True)
        runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
        runner.num_accepted_tokens = SimpleNamespace(
            cpu=torch.zeros(2, dtype=torch.int32),
            gpu=torch.zeros(2, dtype=torch.int32),
        )
        persistent_counts = torch.ones(2, dtype=torch.int32)
        runner.input_batch = SimpleNamespace(num_accepted_tokens_cpu_tensor=persistent_counts)
        runner.kv_cache_config = object()
        runner.compilation_config = SimpleNamespace(static_forward_context={})
        runner.model = SimpleNamespace(get_mamba_state_copy_func=lambda: ())
        runner.num_accepted_tokens_event = MagicMock()
        runner._get_mamba_bufs = MagicMock()

        runner._update_states_after_model_execute(
            torch.tensor([[10, 11, -1], [20, -1, -1]]),
            MagicMock(),
        )

        self.assertIs(
            mock_postprocess.call_args.kwargs["num_accepted_tokens_cpu_tensor"],
            runner.num_accepted_tokens.cpu,
        )
        self.assertIsNot(
            mock_postprocess.call_args.kwargs["num_accepted_tokens_cpu_tensor"],
            persistent_counts,
        )
        runner.num_accepted_tokens_event.record.assert_called_once_with()

    @patch(
        "vllm_ascend.worker.model_runner_v1.GPUModelRunner._update_states_after_model_execute",
        autospec=True,
    )
    def test_non_async_postprocess_delegates_to_upstream(self, mock_postprocess):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_async_scheduling = False
        output_token_ids = torch.tensor([[10, -1]])
        scheduler_output = MagicMock()

        runner._update_states_after_model_execute(output_token_ids, scheduler_output)

        mock_postprocess.assert_called_once_with(runner, output_token_ids, scheduler_output)

    def test_remap_uses_snapshot_after_persistent_row_is_overwritten(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        previous_counts = np.ones(16, dtype=np.int32)
        previous_counts[4] = 3
        previous_counts[11] = 4
        persistent_counts = np.ones(16, dtype=np.int32)

        runner.num_accepted_tokens = SimpleNamespace(np=previous_counts)
        runner.prev_positions = SimpleNamespace(np=np.array([11, -1, 4] + [-1] * 13, dtype=np.int64))
        runner.input_batch = SimpleNamespace(num_accepted_tokens_cpu=persistent_counts)
        runner.use_async_scheduling = True

        runner._sync_num_accepted_tokens(num_reqs=3, has_prev_mapping=True)

        np.testing.assert_array_equal(previous_counts[:3], [4, 1, 3])
        np.testing.assert_array_equal(persistent_counts[:3], [4, 1, 3])

    def test_async_without_previous_mapping_initializes_current_rows(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        snapshot = np.array([0, 4, 3, 9], dtype=np.int32)
        persistent_counts = np.array([7, 8, 6, 5], dtype=np.int32)
        runner.num_accepted_tokens = SimpleNamespace(np=snapshot)
        runner.input_batch = SimpleNamespace(num_accepted_tokens_cpu=persistent_counts)
        runner.use_async_scheduling = True

        runner._sync_num_accepted_tokens(num_reqs=3, has_prev_mapping=False)

        np.testing.assert_array_equal(snapshot, [1, 1, 1, 9])
        np.testing.assert_array_equal(persistent_counts, [1, 1, 1, 5])

    def test_non_async_sync_uses_condensed_input_batch_rows(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        snapshot = np.array([4, 2, 9], dtype=np.int32)
        persistent_counts = np.array([2, 1, 7], dtype=np.int32)
        runner.num_accepted_tokens = SimpleNamespace(np=snapshot)
        runner.input_batch = SimpleNamespace(num_accepted_tokens_cpu=persistent_counts)
        runner.use_async_scheduling = False

        runner._sync_num_accepted_tokens(num_reqs=2, has_prev_mapping=False)

        np.testing.assert_array_equal(snapshot, [2, 1, 9])
        np.testing.assert_array_equal(persistent_counts, [2, 1, 7])


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_compress = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.sfa_dcp_replicated_indexer_size = 1
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.enable_hamming_sparse = False
        runner.shared_kv_cache_layers = {}
        runner.kv_caches = []
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.dsa_offload_config = None
        runner._dsa_offload_main_specs = {}
        runner._dsa_offload_cohorts = ()
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        runner.c8_k_cache_dtype = torch.float8_e4m3fn
        runner.c8_k_scale_cache_dtype = torch.float32
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

    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_sparse_layer_without_indexer_allocates_only_mla_kv_cache(
        self,
        mock_get_layers,
        _mock_has_ec_transfer,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 16
        runner.kv_cache_dtype = torch.bfloat16
        runner.ascend_config = MagicMock()
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        runner.vllm_config.quant_config = None
        runner.vllm_config.cache_config.cache_dtype = "auto"

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.impl = SimpleNamespace(
            has_indexer=False,
            enable_sparse_sfa_c8=False,
            enable_sparse_li_c8=False,
        )
        attn_module.kv_lora_rank = 512
        attn_module.qk_rope_head_dim = 64
        layer_name = "model.layers.1.self_attn.attn"
        mock_get_layers.return_value = {layer_name: attn_module}

        spec = runner.get_kv_cache_spec()[layer_name]
        self.assertEqual(spec.head_size, 512 + 64)
        self.assertEqual(spec.page_size_bytes, 16 * (512 + 64) * 2)

        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=spec.page_size_bytes * 2,
                    shared_by=[layer_name],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                )
            ],
        )

        raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
        raw_k_cache, raw_v_cache = raw_caches[layer_name]

        self.assertEqual(raw_k_cache.numel(), 2 * 16 * 512 * 2)
        self.assertEqual(raw_v_cache.numel(), 2 * 16 * 64 * 2)

    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_sparse_main_and_indexer_use_independent_specs(
        self,
        mock_get_layers,
        _mock_has_ec_transfer,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 128
        runner.kv_cache_dtype = torch.bfloat16
        runner.sfa_dcp_replicated_indexer_size = 3
        runner.ascend_config = MagicMock()
        runner.ascend_config.is_sparse_li_c8_layer.return_value = True
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"

        main_layer_name = "model.layers.0.self_attn.attn"
        main_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(main_module)
        main_module.impl = SimpleNamespace(enable_sparse_sfa_c8=True)

        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        indexer_module = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
        torch.nn.Module.__init__(indexer_module)
        indexer_module.prefix = indexer_layer_name
        mock_get_layers.return_value = {
            main_layer_name: main_module,
            indexer_layer_name: indexer_module,
        }

        specs = runner.get_kv_cache_spec()

        main_spec = specs[main_layer_name]
        indexer_spec = specs[indexer_layer_name]
        packed_head_dim = get_sfa_qsfa_packed_head_dim(512, 64)
        self.assertIsInstance(main_spec, AscendMLAAttentionSpec)
        self.assertTrue(main_spec.cache_sparse_sfa_c8)
        self.assertEqual(main_spec.page_size_bytes, 128 * packed_head_dim)
        self.assertIsInstance(indexer_spec, AscendSFAIndexerCacheSpec)
        self.assertTrue(indexer_spec.cache_sparse_li_c8)
        self.assertEqual(indexer_spec.sfa_dcp_replicated_indexer_size, 3)
        self.assertEqual(indexer_spec.page_size_bytes, 3 * 128 * (128 + 4))
        runner.ascend_config.is_sparse_li_c8_layer.assert_called_once_with(
            indexer_module.prefix,
        )

    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_dsa_offload_decode_omits_main_from_scheduler_spec(
        self,
        mock_get_layers,
        _mock_has_ec_transfer,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 128
        runner.kv_cache_dtype = torch.bfloat16
        runner.dsa_offload_config = SimpleNamespace(
            kv_role="kv_consumer",
            enable_prefetch_with_hidden_states=False,
        )
        runner.ascend_config = MagicMock()
        runner.ascend_config.is_sparse_li_c8_layer.return_value = False
        runner.model_config.hf_text_config = SimpleNamespace(
            num_hidden_layers=1,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"

        main_layer_name = "model.layers.0.self_attn.attn"
        main_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(main_module)
        main_module.impl = SimpleNamespace(
            enable_sparse_sfa_c8=False,
            skip_topk=False,
        )
        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        indexer_module = DeepseekV32IndexerCache.__new__(
            DeepseekV32IndexerCache
        )
        torch.nn.Module.__init__(indexer_module)
        indexer_module.prefix = indexer_layer_name
        mock_get_layers.return_value = {
            main_layer_name: main_module,
            indexer_layer_name: indexer_module,
        }

        scheduler_specs = runner.get_kv_cache_spec()

        self.assertEqual(set(scheduler_specs), {indexer_layer_name})
        self.assertEqual(
            set(runner._dsa_offload_main_specs),
            {main_layer_name},
        )
        self.assertIsInstance(
            runner._dsa_offload_main_specs[main_layer_name],
            AscendMLAAttentionSpec,
        )

    def test_dsa_offload_decode_restores_fixed_main_cache_worker_side(self):
        block_size = 16
        main_layer_name = "model.layers.0.self_attn.attn"
        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        main_spec = AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512 + 64,
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        )
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        )
        scheduler_num_blocks = 100
        kv_cache_config = KVCacheConfig(
            num_blocks=scheduler_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=indexer_spec.page_size_bytes
                    * scheduler_num_blocks,
                    shared_by=[indexer_layer_name],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[indexer_layer_name],
                    kv_cache_spec=indexer_spec,
                ),
            ],
        )
        layout = HotCacheLayout(
            block_size=block_size,
            max_num_seqs=2,
            max_verify_tokens_per_request=1,
        )

        group_id = add_decode_external_main_cache(
            kv_cache_config,
            {main_layer_name: main_spec},
            layout.hot_blocks,
        )

        self.assertEqual(group_id, 0)
        self.assertEqual(kv_cache_config.num_blocks, scheduler_num_blocks)
        group = kv_cache_config.kv_cache_groups[0]
        self.assertEqual(
            set(group.layer_names),
            {main_layer_name, indexer_layer_name},
        )
        self.assertIsInstance(
            group.kv_cache_spec,
            UniformTypeKVCacheSpecs,
        )
        main_tensors = [
            tensor
            for tensor in kv_cache_config.kv_cache_tensors
            if tensor.shared_by == [main_layer_name]
        ]
        self.assertEqual(len(main_tensors), 1)
        self.assertEqual(
            main_tensors[0].size,
            layout.hot_blocks * main_spec.page_size_bytes,
        )

    def _build_sparse_cache_config(self, main_c8: bool, indexer_c8: bool, dcp_size: int):
        block_size = 16
        main_layer_name = "model.layers.0.self_attn.attn"
        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        packed_head_dim = get_sfa_qsfa_packed_head_dim(512, 64)
        main_spec = AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=packed_head_dim if main_c8 else 512 + 64,
            dtype=torch.float8_e4m3fn if main_c8 else torch.bfloat16,
            cache_dtype_str="auto",
            cache_sparse_sfa_c8=main_c8,
        )
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float8_e4m3fn if indexer_c8 else torch.bfloat16,
            scale_dim=1 if indexer_c8 else 0,
            scale_dtype=torch.float32 if indexer_c8 else torch.int8,
            cache_dtype_str="auto",
            cache_sparse_li_c8=indexer_c8,
            sfa_dcp_replicated_indexer_size=dcp_size,
        )
        specs = {
            main_layer_name: main_spec,
            indexer_layer_name: indexer_spec,
        }
        group_spec = UniformTypeKVCacheSpecs(
            block_size=block_size,
            kv_cache_specs=specs,
        )
        num_blocks = 2
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=main_spec.page_size_bytes * num_blocks,
                    shared_by=[main_layer_name],
                ),
                KVCacheTensor(
                    size=indexer_spec.page_size_bytes * num_blocks,
                    shared_by=[indexer_layer_name],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[main_layer_name, indexer_layer_name],
                    kv_cache_spec=group_spec,
                )
            ],
        )
        return (
            main_layer_name,
            indexer_layer_name,
            main_spec,
            indexer_spec,
            kv_cache_config,
        )

    def test_main_and_indexer_specs_have_independent_page_sizes_and_merge(self):
        dcp_size = 4
        (
            _main_layer_name,
            _indexer_layer_name,
            main_bf16_spec,
            indexer_bf16_spec,
            _kv_cache_config,
        ) = self._build_sparse_cache_config(False, False, dcp_size)
        (
            _main_layer_name,
            _indexer_layer_name,
            main_c8_spec,
            indexer_c8_spec,
            _kv_cache_config,
        ) = self._build_sparse_cache_config(True, True, dcp_size)

        packed_head_dim = get_sfa_qsfa_packed_head_dim(512, 64)
        self.assertEqual(main_bf16_spec.page_size_bytes, 16 * (512 + 64) * 2)
        self.assertEqual(main_c8_spec.page_size_bytes, 16 * packed_head_dim)
        self.assertEqual(indexer_bf16_spec.page_size_bytes, dcp_size * 16 * 128 * 2)
        self.assertEqual(indexer_c8_spec.page_size_bytes, dcp_size * 16 * (128 + 4))

        self.assertEqual(
            AscendMLAAttentionSpec.merge([main_c8_spec, main_c8_spec]),
            main_c8_spec,
        )
        self.assertEqual(
            AscendSFAIndexerCacheSpec.merge([indexer_c8_spec, indexer_c8_spec]),
            indexer_c8_spec,
        )

    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_initialize_attn_backend_routes_indexer_cache(self, mock_get_layers):
        runner = self._build_runner()
        runner.attn_groups = []
        runner._check_and_update_cudagraph_mode = MagicMock()
        runner.calculate_reorder_batch_threshold = MagicMock()
        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        indexer_layer = MagicMock()
        indexer_layer.get_attn_backend.side_effect = AssertionError
        mock_get_layers.return_value = {indexer_layer_name: indexer_layer}
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=indexer_spec.page_size_bytes * 2,
                    shared_by=[indexer_layer_name],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[indexer_layer_name],
                    kv_cache_spec=indexer_spec,
                ),
            ],
        )

        runner.initialize_attn_backend(kv_cache_config)

        self.assertEqual(len(runner.attn_groups), 1)
        self.assertEqual(len(runner.attn_groups[0]), 1)
        indexer_group = runner.attn_groups[0][0]
        self.assertIs(indexer_group.backend, AscendSFAIndexerBackend)
        self.assertEqual(indexer_group.layer_names, [indexer_layer_name])
        self.assertIs(indexer_group.kv_cache_spec, indexer_spec)
        self.assertIsInstance(
            indexer_group.get_metadata_builder(),
            AscendSFAIndexerMetadataBuilder,
        )
        indexer_layer.get_attn_backend.assert_not_called()

    def test_sparse_main_and_indexer_allocate_and_reshape_four_layouts(self):
        num_blocks = 2
        dcp_size = 3
        packed_head_dim = get_sfa_qsfa_packed_head_dim(512, 64)

        for main_c8, indexer_c8 in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(main_c8=main_c8, indexer_c8=indexer_c8):
                runner = self._build_runner()
                runner.use_sparse = True
                runner.vllm_config.quant_config = None
                runner.model_config.hf_text_config = SimpleNamespace(index_head_dim=128)
                runner._get_attention_kv_cache_dims = lambda _layer_name, _spec: (512, 64)
                backend = MagicMock()
                backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    head_size,
                )
                (
                    main_layer_name,
                    indexer_layer_name,
                    main_spec,
                    indexer_spec,
                    kv_cache_config,
                ) = self._build_sparse_cache_config(main_c8, indexer_c8, dcp_size)

                raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
                raw_main_cache = raw_caches[main_layer_name]
                raw_indexer_cache = raw_caches[indexer_layer_name]

                self.assertEqual(
                    sum(tensor.numel() for tensor in raw_main_cache),
                    main_spec.page_size_bytes * num_blocks,
                )
                self.assertEqual(
                    sum(tensor.numel() for tensor in raw_indexer_cache),
                    indexer_spec.page_size_bytes * num_blocks,
                )
                self.assertNotEqual(
                    raw_main_cache[0].data_ptr(),
                    raw_indexer_cache[0].data_ptr(),
                )

                runner._kv_cache_spec_attn_group_iterator = MagicMock(
                    return_value=[
                        SimpleNamespace(
                            kv_cache_spec=main_spec,
                            backend=backend,
                            layer_names=[main_layer_name],
                        ),
                        SimpleNamespace(
                            kv_cache_spec=indexer_spec,
                            backend=backend,
                            layer_names=[indexer_layer_name],
                        ),
                    ],
                )
                caches = runner._reshape_kv_cache_tensors(
                    kv_cache_config,
                    raw_caches,
                )
                main_cache = caches[main_layer_name]
                indexer_cache = caches[indexer_layer_name]

                if main_c8:
                    self.assertEqual(len(main_cache), 1)
                    self.assertEqual(
                        main_cache[0].shape,
                        (num_blocks, 16, 1, packed_head_dim),
                    )
                    self.assertEqual(main_cache[0].dtype, torch.float8_e4m3fn)
                else:
                    self.assertEqual(len(main_cache), 2)
                    self.assertEqual(main_cache[0].shape, (num_blocks, 16, 1, 512))
                    self.assertEqual(main_cache[1].shape, (num_blocks, 16, 1, 64))
                    self.assertEqual(main_cache[0].dtype, torch.bfloat16)
                    self.assertEqual(main_cache[1].dtype, torch.bfloat16)

                self.assertEqual(
                    indexer_cache[0].shape,
                    (num_blocks * dcp_size, 16, 1, 128),
                )
                if indexer_c8:
                    self.assertEqual(len(indexer_cache), 2)
                    self.assertEqual(indexer_cache[0].dtype, torch.float8_e4m3fn)
                    self.assertEqual(
                        indexer_cache[1].shape,
                        (num_blocks * dcp_size, 16, 1, 1),
                    )
                    self.assertEqual(indexer_cache[1].dtype, torch.float32)
                else:
                    self.assertEqual(len(indexer_cache), 1)
                    self.assertEqual(indexer_cache[0].dtype, torch.bfloat16)

    def test_sparse_initialize_kv_cache_tensors_binds_four_layouts(self):
        dcp_size = 3

        for main_c8, indexer_c8 in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(main_c8=main_c8, indexer_c8=indexer_c8):
                runner = self._build_runner()
                runner.use_sparse = True
                runner.vllm_config.quant_config = None
                runner.model_config.hf_text_config = SimpleNamespace(
                    index_head_dim=128,
                    model_type="glm4",
                )
                runner._get_attention_kv_cache_dims = lambda _layer_name, _spec: (512, 64)
                backend = MagicMock()
                backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    head_size,
                )
                (
                    main_layer_name,
                    indexer_layer_name,
                    main_spec,
                    indexer_spec,
                    kv_cache_config,
                ) = self._build_sparse_cache_config(main_c8, indexer_c8, dcp_size)
                runner._kv_cache_spec_attn_group_iterator = MagicMock(
                    return_value=[
                        SimpleNamespace(
                            kv_cache_spec=main_spec,
                            backend=backend,
                            layer_names=[main_layer_name],
                        ),
                        SimpleNamespace(
                            kv_cache_spec=indexer_spec,
                            backend=backend,
                            layer_names=[indexer_layer_name],
                        ),
                    ],
                )
                main_module = SimpleNamespace(kv_cache=None)
                indexer_module = SimpleNamespace(kv_cache=None)
                runner.compilation_config = SimpleNamespace(
                    static_forward_context={
                        main_layer_name: main_module,
                        indexer_layer_name: indexer_module,
                    },
                )

                caches = runner.initialize_kv_cache_tensors(kv_cache_config)

                self.assertIs(main_module.kv_cache, caches[main_layer_name])
                self.assertIs(indexer_module.kv_cache, caches[indexer_layer_name])
                self.assertEqual(len(runner.kv_caches), 2)
                self.assertEqual(
                    sum(cache is caches[main_layer_name] for cache in runner.kv_caches),
                    1,
                )
                self.assertEqual(
                    sum(cache is caches[indexer_layer_name] for cache in runner.kv_caches),
                    1,
                )
                self.assertEqual(len(caches[main_layer_name]), 1 if main_c8 else 2)
                self.assertEqual(len(caches[indexer_layer_name]), 2 if indexer_c8 else 1)


class TestNPUModelRunnerOutputTokenIds(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = MagicMock()
        runner.use_compress = False
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.get_ascend_config")
    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable, mock_get_ascend_config):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_reduce_sample = False
        mock_get_ascend_config.return_value = mock_ascend_config

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.sampling_metadata.top_k = None
        input_batch.num_reqs = 2
        input_batch.top_k_cpu = None
        input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }
        input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        input_batch.async_copy_ready_event = MagicMock()
        input_batch.async_copy_ready_event.synchronize = MagicMock()

        # Simulate the real behavior of InputBatch.update_async_output_token_ids
        def mock_update_output_token_ids():
            output_token_ids = input_batch.sampling_metadata.output_token_ids
            sampled_ids = input_batch.sampled_token_ids_cpu.tolist()

            for index, req_id in enumerate(input_batch.prev_req_id_to_index):
                prev_index = input_batch.prev_req_id_to_index[req_id]
                req_output = output_token_ids[index]
                if req_output and req_output[-1] == -1:
                    req_output[-1] = sampled_ids[prev_index]

        input_batch.update_async_output_token_ids.side_effect = mock_update_output_token_ids

        # Build runner and inject dependencies
        runner = self._build_runner()
        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value=MagicMock())

        # Call sample method
        logits = torch.randn(2, 32000)
        runner._sample(logits=logits, spec_decode_metadata=None)

        # Verify sampler and update_async_output_token_ids were called
        runner.sampler.assert_called_once()
        input_batch.update_async_output_token_ids.assert_called_once()

        # Verify output_token_ids were updated before sampler is called
        call_kwargs = runner.sampler.call_args[1]
        actual_sampling_metadata = call_kwargs["sampling_metadata"]
        actual_output_token_ids = actual_sampling_metadata.output_token_ids
        self.assertEqual(actual_output_token_ids[0], [1, 2, 3, 6])
        self.assertEqual(actual_output_token_ids[1], [4, 5, 7])

    def test_placeholder_spec_tokens_are_sanitized_only_for_forward(self):
        runner = self._build_runner()
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, 33, -1])

    def test_placeholder_sanitization_is_scoped_to_current_forward(self):
        runner = self._build_runner()
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=2,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, -1])

    def test_mtp3_placeholder_metadata_is_preserved_before_sanitizing_forward(self):
        runner = self._build_runner()
        runner.pcp_size = 1
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner._arange_scratch = np.empty(8, dtype=np.int32)
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, -1, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, -1, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1, -1, -1]},
        )

        spec_decode_metadata = runner._calc_spec_decode_metadata(
            num_draft_tokens=np.array([3], dtype=np.int32),
            cu_num_scheduled_tokens=np.array([4], dtype=np.int32),
            num_pcp_pads=None,
        )
        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(spec_decode_metadata.draft_token_ids.tolist(), [-1, -1, -1])
        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 0, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, -1, -1])


class TestNPUModelRunnerDebugger(unittest.TestCase):
    def _build_runner(self, debugger=None):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.debugger = debugger or MagicMock()
        runner.model = MagicMock()
        runner.model_config = MagicMock()
        runner.model_config.enforce_eager = False
        runner._debugger_started = True
        runner._debugger_step_dummy_data_before_execute = False
        runner.use_compress = False
        return runner

    def test_finalize_dump_data_stops_stop_capable_debugger(self):
        runner = self._build_runner()

        runner._finalize_dump_data()

        runner.debugger.stop.assert_called_once_with()
        runner.debugger.step.assert_called_once_with()
        self.assertFalse(runner._debugger_started)

    def test_finalize_dump_data_steps_graph_debugger_without_stop(self):
        debugger = MagicMock(spec=["start", "step"])
        runner = self._build_runner(debugger)

        runner._finalize_dump_data()

        debugger.step.assert_called_once_with()
        self.assertTrue(runner._debugger_started)

    def test_start_dump_data_noop_when_already_started(self):
        runner = self._build_runner(MagicMock(spec=["start", "step"]))

        runner._start_dump_data()

        runner.debugger.start.assert_not_called()
        runner.debugger.step.assert_not_called()
        self.assertTrue(runner._debugger_started)


class TestCorrectOptimisticSeqLensCpu(unittest.TestCase):
    """Regression tests for async spec-decode seq_lens correction.

    The helper must synchronize the device->host copy event *before* reading
    ``valid_sampled_token_count_cpu``. Reading it early consumes stale counts
    and corrupts the CPU seq_lens, which surfaced as an accuracy regression on
    DeepSeek-V4 (its compressed-KV slot mapping is built from these seq_lens).
    """

    def _build_runner(self, optimistic, prev_positions, prev_drafts, counts_cpu):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.optimistic_seq_lens_cpu = optimistic
        runner.prev_positions = SimpleNamespace(np=prev_positions)
        runner.prev_num_draft_tokens = SimpleNamespace(np=prev_drafts)
        runner.valid_sampled_token_count_cpu = counts_cpu
        return runner

    def test_synchronizes_before_host_read(self):
        num_reqs = 3
        # Optimistic (all drafts assumed accepted):
        #   prev_computed=[100,200,50], prev_drafts=[2,3,1], sched=[3,4,2]
        #   optimistic = prev_computed + (prev_drafts + 1) + sched
        optimistic = torch.tensor([106, 208, 54], dtype=torch.int64)
        prev_positions = np.array([0, 1, 2], dtype=np.int64)
        prev_drafts = np.array([2, 3, 1], dtype=np.int32)

        # CPU buffer initially holds STALE counts (== drafts + 1, i.e. "all
        # accepted"). If the helper reads before synchronizing, the correction
        # is a no-op and the assertion below fails.
        counts_cpu = torch.tensor([3, 4, 2], dtype=torch.int32)
        # The true counts that the async copy delivers on synchronize().
        true_counts = np.array([2, 1, 2], dtype=np.int32)

        runner = self._build_runner(optimistic, prev_positions, prev_drafts, counts_cpu)
        event = MagicMock()
        event.synchronize.side_effect = lambda: counts_cpu.copy_(torch.from_numpy(true_counts))
        runner.valid_sampled_token_count_event = event

        runner._correct_optimistic_seq_lens_cpu(num_reqs)

        event.synchronize.assert_called_once()
        # correction = (prev_drafts + 1 - true_counts) = [1, 3, 0]
        # corrected  = optimistic - correction          = [105, 205, 54]
        np.testing.assert_array_equal(optimistic.numpy(), np.array([105, 205, 54]))

    def test_asserts_event_present(self):
        runner = self._build_runner(
            torch.tensor([10], dtype=torch.int64),
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int32),
            torch.tensor([1], dtype=torch.int32),
        )
        runner.valid_sampled_token_count_event = None
        with self.assertRaises(AssertionError):
            runner._correct_optimistic_seq_lens_cpu(1)


if __name__ == "__main__":
    unittest.main()
