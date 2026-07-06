import sys
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm.distributed.parallel_state import GroupCoordinator

if 'torch_npu._inductor' not in sys.modules:
    sys.modules['torch_npu._inductor'] = MagicMock()

from vllm_ascend.attention.sfa_v1 import (AscendSFABackend, AscendSFAImpl,
                                          AscendSFAMetadata,
                                          AscendSFAMetadataBuilder)
from vllm_ascend.utils import enable_dsa_cp


class TestAscendSFABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(),
                         AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):

    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestAscendSFAMetadataBuilder(TestBase):

    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config",
                             return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(self, kv_cache_spec, layer_names, vllm_config,
                             device, metadata_cls, supports_dcp_with_varlen):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__",
            mock_parent_init)
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100
        common_attn_metadata.max_query_len = 10
        common_attn_metadata.req_ids = ["req-0", "req-1"]
        common_attn_metadata.token_req_indices_cpu = torch.tensor([0, 1], dtype=torch.int32)
        common_attn_metadata.token_positions_cpu = torch.tensor([0, 0], dtype=torch.int64)
        common_attn_metadata.prefill_lens_cpu = torch.tensor([1, 1], dtype=torch.int32)

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)
        assert metadata.num_decodes == 0
        assert metadata.num_decode_tokens == 0
        assert metadata.num_prefills == 10
        assert metadata.req_ids == ["req-0", "req-1"]
        assert torch.equal(metadata.token_req_indices_cpu, torch.tensor([0, 1], dtype=torch.int32))
        assert torch.equal(metadata.token_positions_cpu, torch.tensor([0, 0], dtype=torch.int64))
        assert torch.equal(metadata.prefill_lens_cpu, torch.tensor([1, 1], dtype=torch.int32))

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
            self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100
        common_attn_metadata.max_query_len = 10

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly


class TestAscendSFAOffloadKVCacheV0Wiring(TestBase):

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_scatter_nd_update_")
    @patch("vllm_ascend.attention.sfa_v1._EXTRA_CTX")
    def test_forward_calls_real_ops_lookup_between_indexer_and_sfa(
        self,
        mock_extra_ctx,
        mock_scatter_update,
        mock_wait_for_kv,
        mock_get_weight_prefetch_method,
        mock_save_kv,
    ):
        manager = MagicMock()
        call_order = []
        sfa_topk_indices = []
        topk_indices = torch.tensor([[[0]]], dtype=torch.int32)
        mock_extra_ctx.offload_kv_cache_v0 = manager
        mock_extra_ctx.capturing = False
        manager.persist_prefill_kv_to_microkv.side_effect = lambda **kwargs: call_order.append("persist")
        manager.validate_topk_with_real_hbm_index_ops.side_effect = lambda **kwargs: call_order.append("lookup")

        prefetch_method = MagicMock()
        mock_get_weight_prefetch_method.return_value = prefetch_method

        impl = object.__new__(AscendSFAImpl)
        impl.enable_dsa_cp = False
        impl.enable_dsa_cp_with_layer_shard = False
        impl.enable_dsa_cp_with_o_proj_tp = False
        impl.enable_mlapo = False
        impl.use_sparse_c8_indexer = False
        impl.is_kv_producer = False
        impl.q_lora_rank = 2
        impl.kv_lora_rank = 2
        impl.qk_rope_head_dim = 1
        impl.fused_qkv_a_proj = MagicMock(return_value=(torch.ones(1, 5), None))
        impl.q_a_layernorm = MagicMock(side_effect=lambda x: x)
        impl.indexer_select_pre_process = MagicMock(return_value=(torch.ones(1, 3), None))
        impl.exec_kv = MagicMock(return_value=(None, None))
        impl._q_proj_and_k_up_proj = MagicMock(return_value=(torch.ones(1, 1, 2), torch.ones(1, 1, 1)))
        impl.rope_single = MagicMock(side_effect=lambda q_pe, cos, sin: q_pe)
        impl._get_full_kv = MagicMock(side_effect=lambda k_li, attn_metadata: k_li)
        impl.indexer_select_post_process = MagicMock(
            side_effect=lambda **kwargs: call_order.append("indexer") or topk_indices
        )
        def execute_sparse_flash_attention(*args):
            call_order.append("sfa")
            sfa_topk_indices.append(args[3])
            return torch.ones(1, 3)

        impl._execute_sparse_flash_attention_process = MagicMock(side_effect=execute_sparse_flash_attention)
        impl._v_up_proj = MagicMock(return_value=torch.ones(1, 3))
        impl.o_proj = MagicMock(return_value=(torch.ones(1, 3), None))
        impl.o_proj.weight = torch.ones(1, 3)

        metadata = AscendSFAMetadata(
            num_actual_tokens=1,
            slot_mapping=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
            cum_query_lens=torch.tensor([1], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            sin=torch.zeros(1, 1),
            cos=torch.zeros(1, 1),
            num_input_tokens=1,
            attn_state=AscendAttentionState.DecodeOnly,
            num_decode_tokens=1,
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([1], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([1], dtype=torch.int32),
        )
        kv_cache = (
            torch.zeros(1, 1, 1, 2),
            torch.zeros(1, 1, 1, 1),
            torch.zeros(1, 1, 3),
        )
        output = torch.zeros(1, 3)

        impl.forward("model.layers.0.self_attn", torch.ones(1, 3), kv_cache, metadata, output=output)

        self.assertEqual(call_order, ["persist", "indexer", "lookup", "sfa"])
        self.assertIs(sfa_topk_indices[0], topk_indices)
        manager.persist_prefill_kv_to_microkv.assert_called_once()
        manager.validate_topk_with_real_hbm_index_ops.assert_called_once()
        self.assertIs(impl._execute_sparse_flash_attention_process.call_args.args[3], topk_indices)
