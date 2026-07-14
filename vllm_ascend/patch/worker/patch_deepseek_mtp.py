import torch
import torch.nn as nn
import vllm
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP, DeepSeekMultiTokenPredictorLayer

MTP_ROT_WEIGHT_NAME = "rot.weight"


def get_spec_layer_idx_from_weight_name(config: DeepseekV2Config | DeepseekV3Config, weight_name: str) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and config.num_nextn_predict_layers > 0:
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}.") or weight_name.startswith(MTP_ROT_WEIGHT_NAME):
                return layer_idx + i
    return None


class AscendDeepSeekMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__(vllm_config, prefix)
        quant_description = getattr(vllm_config.quant_config, "quant_description", None)
        self.is_rot_used = quant_description.get("is_rot_used", False) if quant_description is not None else False
        self.target_model_type = vllm_config.speculative_config.target_model_config.hf_text_config.model_type
        if self.is_rot_used and self.target_model_type == "glm_moe_dsa":
            self.rot = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        if self.is_rot_used and self.target_model_type == "glm_moe_dsa":
            previous_hidden_states = self.rot(previous_hidden_states)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
        hidden_states = residual + hidden_states
        return hidden_states


class AscendDeepSeekMTP(DeepSeekMTP):
    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if name != MTP_ROT_WEIGHT_NAME:
            return super()._rewrite_spec_layer_name(spec_layer, name)
        else:
            return f"model.layers.{spec_layer}.rot.weight"


vllm.model_executor.models.deepseek_v2.get_spec_layer_idx_from_weight_name = get_spec_layer_idx_from_weight_name
vllm.model_executor.models.deepseek_mtp.get_spec_layer_idx_from_weight_name = get_spec_layer_idx_from_weight_name
vllm.model_executor.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer = AscendDeepSeekMultiTokenPredictorLayer
vllm.model_executor.models.deepseek_mtp.DeepSeekMTP = AscendDeepSeekMTP


from vllm_ascend import envs  # noqa: E402

if envs.VLLM_ASCEND_ENABLE_DMP:
    from itertools import islice

    import torch_npu  # noqa: F401
    from vllm.distributed import get_pp_group
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2Model, IntermediateTensors

    # DMP requires MLA with sparse indexer, which is only available in
    # DeepSeek V3+ (DeepseekV3Config / DeepseekV3ProcessingConfig).
    # Guard against accidental application to V2 or Lite models.
    _dmp_config_type = type(DeepseekV2Model.__init__)  # placeholder check
    # Check at patch-application time: DeepseekV2Model shares the class
    # across V2/V3, so we verify at forward time instead (see dmp_forward).

    # DMP's dmp_forward contains cross-stream dispatch logic (npu.Stream,
    # npu.Event) and Python-level control flow that is incompatible with
    # TorchDynamo tracing.  When TorchCompileWithNoGuardsWrapper compiles
    # DeepseekV2Model.forward with fullgraph=True, Dynamo inlines
    # dmp_forward and misinterprets `self` as a Tensor.  Setting
    # _ignore_compile_vllm = True makes do_not_compile = True so that
    # __call__ directly invokes self.forward (dmp_forward) without going
    # through Dynamo.  Graph capture is handled by ACLGraphWrapper
    # (torch.npu.NPUGraph) at a higher level, which correctly records
    # the NPU ops including cross-stream operations.
    from vllm.compilation.decorators import IGNORE_COMPILE_KEY
    setattr(DeepseekV2Model, IGNORE_COMPILE_KEY, True)

    # Store original forward for fallback (prefill, non-DMP decode)
    _original_deepseek_v2_model_forward = DeepseekV2Model.forward

    def _dmp_input_layernorm(layer, hidden_states, residual):
        """Apply input_layernorm, matching DecoderLayer.forward() logic."""
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states, residual)
        return hidden_states, residual

    def _dmp_post_attn_and_mlp(layer, hidden_states, residual,
                                llama_4_scaling=None):
        """Post-attention layernorm + MLP, matching DecoderLayer.forward()
        tail logic."""
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual)
        if layer.mlp is not None:
            hidden_states = layer.mlp(hidden_states)
        return hidden_states, residual

    def forward_indexer_only(self, positions, hidden_states, residual,
                             llama_4_scaling=None):
        """Layer-level indexer phase: input_layernorm → self_attn.forward_indexer_only().

        Returns (SFAIndexerResult, residual).
        """
        hidden_states, residual = _dmp_input_layernorm(
            self, hidden_states, residual)
        # FP16 overflow fix (same as DecoderLayer.forward)
        if (hidden_states.dtype == torch.float16
                and self.routed_scaling_factor is not None):
            hidden_states = hidden_states * (
                1.0 / self.routed_scaling_factor)

        indexer_out = self.self_attn.mla_attn.forward_indexer_only(
            positions, hidden_states)
        return indexer_out, residual

    def forward_sparse_attn_and_mlp(self, hidden_states, residual,
                                    indexer_out, llama_4_scaling=None):
        """Layer-level sparse-attn + MLP phase:
        self_attn.forward_sparse_attn_only() → residual add
        → post_attn_layernorm → mlp.

        Returns (hidden_states, residual).
        """
        attn_output = self.self_attn.mla_attn.forward_sparse_attn_only(indexer_out)

        # FP16 overflow fix for attention output
        if (attn_output.dtype == torch.float16
                and self.routed_scaling_factor is not None
                and self.shared_expert_scaling_factor is not None):
            attn_output = attn_output * (
                1.0 / self.routed_scaling_factor)

        hidden_states = residual + attn_output
        hidden_states, residual = _dmp_post_attn_and_mlp(
            self, hidden_states, residual, llama_4_scaling)
        return hidden_states, residual

    def forward_sparse_attn_only(self, hidden_states, residual,
                                 indexer_out, llama_4_scaling=None):
        """Layer-level sparse-attn phase WITHOUT MoE:
        self_attn.forward_sparse_attn_only() → residual add
        → post_attn_layernorm.

        Used for cross-layer DMP: mbB SparseAttn and mbB MoE are
        separated so that mbB MoE overlaps with mbA indexer on S1.

        Returns (hidden_states, residual) after post_attn_layernorm.
        """
        attn_output = self.self_attn.mla_attn.forward_sparse_attn_only(indexer_out)

        # FP16 overflow fix for attention output
        if (attn_output.dtype == torch.float16
                and self.routed_scaling_factor is not None
                and self.shared_expert_scaling_factor is not None):
            attn_output = attn_output * (
                1.0 / self.routed_scaling_factor)

        hidden_states = residual + attn_output
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        return hidden_states, residual

    def forward_mlp_only(self, hidden_states, residual,
                         llama_4_scaling=None):
        """Layer-level MoE-only phase.

        Used for cross-layer DMP: mbB MoE runs on S0 while mbA indexer
        for the next layer runs on S1.

        Returns (hidden_states, residual).
        """
        if self.mlp is not None:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_mlp_two_mb_once(layer, hs_a, res_a, hs_b, res_b):
        n_a = hs_a.shape[0]
        n_b = hs_b.shape[0]

        hs = torch.cat([hs_a, hs_b], dim=0)

        if res_a is None and res_b is None:
            res = None
        else:
            assert res_a is not None and res_b is not None
            res = torch.cat([res_a, res_b], dim=0)

        hs, res = layer.forward_mlp_only(hs, res)

        hs_a, hs_b = torch.split(hs, [n_a, n_b], dim=0)

        if res is None:
            res_a = None
            res_b = None
        else:
            res_a, res_b = torch.split(res, [n_a, n_b], dim=0)

        return hs_a, res_a, hs_b, res_b

    # NOTE: DeepseekV2Model compilation is disabled via _ignore_compile_vllm
    # (see above) because dmp_forward's cross-stream logic is incompatible
    # with TorchDynamo.  The @torch.compiler.disable decorator below is kept
    # as a secondary guard; it is NOT effective under fullgraph=True (Dynamo
    # ignores graph-break requests), but would protect if someone removes
    # the _ignore_compile_vllm flag.
    @torch.compiler.disable
    def dmp_forward(self, input_ids, positions, intermediate_tensors,
                    inputs_embeds=None):
        """Cross-layer DMP interleaved forward.

        Per-layer, two microbatches (A/B) execute with cross-layer
        overlapping: mbB MoE(L) on S0 || mbA Indexer+Load(L+1) on S1.

        Timeline for layer L (L > 0, not last):
          Step 1: S0 waits for S1's mbA indexer+load(L) to complete
          Step 2: S0 runs mbA SparseAttn + MoE, records moe_done_A
          Step 3: S0 runs mbB Indexer + Classify + async_load B
          Step 4: S0 runs mbB SparseAttn (no MoE yet)
          Step 5: S1 launches mbA Indexer+Classify+Load(L+1) overlapping
                  with S0's mbB MoE

        Replaces DeepseekV2Model.forward when DMP is enabled.
        """

        forward_context = get_forward_context()
        dmp_ctx = getattr(forward_context, 'dmp_context', None)
        is_capturing = getattr(forward_context, 'capturing', False)

        # torch._dynamo.config.reorderable_logging_functions.add(print)
        # print("[DMP] enter dmp_forward, dmp_ctx is None=%s", dmp_ctx is None)

        # DMP requires MLA with sparse indexer (DeepSeek V3+ only).
        # Fall back to original forward for non-V3 models.
        hf_config = getattr(self, 'config', None)
        if hf_config is not None and not hasattr(hf_config, 'index_topk'):
            # logger.debug("[DMP] fallback because no index_topk")
            # npu_print("[DMP] 4 fall back!")
            return _original_deepseek_v2_model_forward(
                self, input_ids, positions, intermediate_tensors, inputs_embeds)

        # Fallback: prefill or non-DMP decode (dmp_context not set)
        if dmp_ctx is None:
            # npu_print("[DMP] 3 fall back!")
            print("[DEBUG] out dmp_forward 2!")
            return _original_deepseek_v2_model_forward(
                self, input_ids, positions, intermediate_tensors, inputs_embeds)

        # npu_print("[DMP] 2 enter dmp_forward!")

        # print("[DMP] enter dmp_forward, dmp_ctx is None=%s 1", dmp_ctx is None)

        # Standard embed + pipeline parallel entry (same as original)
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Llama-4 scaling factor (computed once, same as original)
        llama_4_scaling = None
        hf_config = getattr(self, 'config', None)
        if hf_config is not None and hasattr(hf_config,
                                             'llama_4_scaling'):
            llama_4_scaling = hf_config.llama_4_scaling

        def _get_block_size(config, default=128):
            try:
                return object.__getattribute__(config, "cache_config").block_size
            except AttributeError:
                return default

        block_size = _get_block_size(self.config)

        # print("[DMP] enter dmp_forward, dmp_ctx is None=%s 1", dmp_ctx is None)

        # Split hidden_states into A/B microbatches
        s_a, s_b = dmp_ctx.slices[0], dmp_ctx.slices[1]
        hs_a = dmp_ctx.slice_hidden_states(hidden_states, 0)
        hs_b = dmp_ctx.slice_hidden_states(hidden_states, 1)
        # Defensive: slice positions per microbatch (MLA uses cos/sin from
        # attn_metadata, but other backends may read positions directly).
        pos_a = positions[s_a.start:s_a.end]
        pos_b = positions[s_b.start:s_b.end]
        if residual is not None:
            res_a = dmp_ctx.slice_hidden_states(residual, 0)
            res_b = dmp_ctx.slice_hidden_states(residual, 1)
        else:
            res_a = None
            res_b = None

        # Pad smaller microbatch with zeros for 50/50 shape balance
        # (required for graph mode shape consistency)
        if s_a.num_padded_tokens > 0:
            pad_a = torch.zeros(s_a.num_padded_tokens, hs_a.shape[1],
                                dtype=hs_a.dtype, device=hs_a.device)
            hs_a = torch.cat([hs_a, pad_a])
            if res_a is not None:
                pad_res_a = torch.zeros(s_a.num_padded_tokens,
                                        res_a.shape[1],
                                        dtype=res_a.dtype,
                                        device=res_a.device)
                res_a = torch.cat([res_a, pad_res_a])
        if s_b.num_padded_tokens > 0:
            pad_b = torch.zeros(s_b.num_padded_tokens, hs_b.shape[1],
                                dtype=hs_b.dtype, device=hs_b.device)
            hs_b = torch.cat([hs_b, pad_b])
            if res_b is not None:
                pad_res_b = torch.zeros(s_b.num_padded_tokens,
                                        res_b.shape[1],
                                        dtype=res_b.dtype,
                                        device=res_b.device)
                res_b = torch.cat([res_b, pad_res_b])
        # Pad positions to match padded token count (dummy positions for
        # padding tokens; exact values don't matter since slot_mapping=-1
        # prevents KV writes for padding tokens).
        if s_a.num_padded_tokens > 0:
            pad_pos_a = torch.zeros(s_a.num_padded_tokens,
                                    dtype=pos_a.dtype, device=pos_a.device)
            pos_a = torch.cat([pos_a, pad_pos_a])
        if s_b.num_padded_tokens > 0:
            pad_pos_b = torch.zeros(s_b.num_padded_tokens,
                                    dtype=pos_b.dtype, device=pos_b.device)
            pos_b = torch.cat([pos_b, pad_pos_b])

        # Cross-layer DMP state
        s0 = torch.npu.current_stream()
        s1 = dmp_ctx.kv_loader.load_stream   # S1: load stream (reused for indexer)

        # Make S1 a child of the graph's capture stream. Every layer rejoins
        # S0 through indexer_done before its output is consumed.
        fork_event = dmp_ctx.get_event("dmp_fork")
        fork_event.record(s0)
        s1.wait_event(fork_event)
        previous_layer_done = None

        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        num_layers = len(layers)

        # Per-layer interleaved DMP execution
        # Both eager and graph-capture use the same dual-stream path.
        # SSD offload (classify_topk_indices + async_load_blocks +
        # wait_load_complete) is skipped during capture since it involves
        # variable-length ops and host-side IO that cannot be captured.
        print("[DMP] dmp_forward graph_capture={}!".format(is_capturing))
        for layer_idx in range(num_layers):
            layer = layers[layer_idx]
            forward_context.layer_idx = self.start_layer + layer_idx

            # ── Step 1: Consume S1's cross-layer mbA indexer+load ──
            # if pending_indexer_load_a is not None and not first:
            #     # pending_indexer_load_a.wait(s0)
            #     s0.wait_event(pending_indexer_load_a)
            #     res_a = pending_res_a_next
            #     pending_indexer_load_a = None
            #     pending_res_a_next = None

            # if layer_idx == 0:
                # First layer: mbA indexer on S1 (no cross-layer overlap)
            with dmp_ctx.enter_microbatch(0), torch.npu.stream(s1):
                if previous_layer_done is not None:
                    s1.wait_event(previous_layer_done)
                indexer_out_a, res_a = layer.forward_indexer_only(
                    pos_a, hs_a, res_a, llama_4_scaling)
                if not is_capturing:
                    topk_a = indexer_out_a[2]
                    if topk_a is not None:
                        _, asu_a = \
                            dmp_ctx.block_location.classify_topk_indices(
                                topk_a, block_size,
                                num_real_tokens=dmp_ctx.slices[
                                    0].num_real_tokens)
                        dmp_ctx.kv_loader.async_load_blocks(
                            asu_a,
                            tag=f"L{layer_idx}_A",
                            kv_cache=layer.self_attn.mla_attn.mla_attn.kv_cache,
                            block_size=block_size)
                indexer_a_done = dmp_ctx.get_event(
                    f"L{layer_idx}_indexer_A_done")
                indexer_a_done.record(s1)

            # mbB indexer on S0
            with torch.npu.stream(s0), dmp_ctx.enter_microbatch(1):
                indexer_out_b, res_b = layer.forward_indexer_only(
                    pos_b, hs_b, res_b, llama_4_scaling)
                if not is_capturing:
                    topk_b = indexer_out_b[2]
                    if topk_b is not None:
                        _, asu_b = \
                            dmp_ctx.block_location.classify_topk_indices(
                                topk_b, block_size,
                                num_real_tokens=dmp_ctx.slices[
                                    1].num_real_tokens)
                        dmp_ctx.kv_loader.async_load_blocks(
                            asu_b,
                            tag=f"L{layer_idx}_B",
                            kv_cache=layer.self_attn.mla_attn.mla_attn.kv_cache,
                            block_size=block_size)

            # mbA SFA on S0
            s0.wait_event(indexer_a_done)
            if not is_capturing:
                dmp_ctx.kv_loader.wait_load_complete(f"L{layer_idx}_A")

            with torch.npu.stream(s0), dmp_ctx.enter_microbatch(0):
                # s0.wait_event(li0_event)
                hs_a, res_a = layer.forward_sparse_attn_only(
                        hs_a, res_a, indexer_out_a, llama_4_scaling)
                # sfa0_event.record(s0)

            # mbB SFA on S0
            if not is_capturing:
                dmp_ctx.kv_loader.wait_load_complete(f"L{layer_idx}_B")
            with torch.npu.stream(s0), dmp_ctx.enter_microbatch(1):
                # s0.wait_event(li1_event)
                hs_b, res_b = layer.forward_sparse_attn_only(
                        hs_b, res_b, indexer_out_b, llama_4_scaling)
                # sfa1_event.record(s0)

            # mbA MLP on S0
            # with torch.npu.stream(s0):
            #     with dmp_ctx.enter_microbatch(0):
            #         # s0.wait_event(sfa0_event)
            #         hs_a, res_a = layer.forward_mlp_only(
            #                 hs_a, res_a, llama_4_scaling)


            # ── Step 5: Cross-layer overlap (not last layer) ──
            # if not is_last_layer:
            #     next_layer = layers[layer_idx + 1]

            #     with dmp_ctx.enter_microbatch(0):
            #         with torch.npu.stream(s1):
            #             # moe_done_a.wait(s1)
            #             s1.wait_event(moe_done_a)

            #             indexer_out_a_next, res_a_next = \
            #                 next_layer.forward_indexer_only(
            #                     pos_a, hs_a, res_a, llama_4_scaling)

            #             # if not is_capturing:
            #             topk_a_next = indexer_out_a_next[2]
            #             if topk_a_next is not None:
            #                 _, asu_a_next = \
            #                     dmp_ctx.block_location.classify_topk_indices(
            #                         topk_a_next, block_size,
            #                         num_real_tokens=dmp_ctx.slices[
            #                             0].num_real_tokens)
            #                 dmp_ctx.kv_loader.async_load_blocks(
            #                     asu_a_next,
            #                     tag=f"L{layer_idx + 1}_A",
            #                     kv_cache=next_layer.self_attn.mla_attn.mla_attn.kv_cache,
            #                     block_size=block_size)

                # Record S1 completion event for next iteration
                # first = False
                # pending_indexer_load_a = torch.npu.Event()
                # pending_indexer_load_a.record(s1)
                # indexer_out_a = indexer_out_a_next
                # pending_res_a_next = res_a_next

            # mbB MLP on S0 (overlaps with S1 above)
            with torch.npu.stream(s0), dmp_ctx.enter_microbatch(1):
                # s0.wait_event(sfa1_event)
                # hs_b, res_b = layer.forward_mlp_only(
                #         hs_b, res_b, llama_4_scaling)
                hs_a, res_a, hs_b, res_b = forward_mlp_two_mb_once(
                    layer, hs_a, res_a, hs_b, res_b
                )

            # The next layer's S1 indexer consumes hs_a/res_a produced by
            # this MLP, so it must wait for S0 explicitly.
            previous_layer_done = dmp_ctx.get_event(
                f"L{layer_idx}_mlp_done")
            previous_layer_done.record(s0)

        # S1 join S0, then merge microbatches
        # join_event = torch.npu.Event()
        # join_event.record(s1)
        # join_event.wait(s0)
        # s0.wait_event(join_event)
        # DEBUG
        # s0.wait_stream(s1)

        num_real_tokens = s_a.num_real_tokens + s_b.num_real_tokens
        merged_shape = (num_real_tokens, hs_a.shape[1])
        hidden_states = dmp_ctx.merge_hidden_states(
            hs_a, hs_b, torch.empty(merged_shape,
                                    dtype=hs_a.dtype,
                                    device=hs_a.device))
        if res_a is not None and res_b is not None:
            residual = dmp_ctx.merge_hidden_states(
                res_a, res_b, torch.empty(merged_shape,
                                          dtype=res_a.dtype,
                                          device=res_a.device))
        else:
            residual = None

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        print("[DEBUG] out dmp_forward!")
        return hidden_states

    # Apply monkey-patches
    DeepseekV2Model.forward = dmp_forward

    DeepseekV2DecoderLayer.forward_indexer_only = forward_indexer_only
    DeepseekV2DecoderLayer.forward_sparse_attn_and_mlp = forward_sparse_attn_and_mlp
    DeepseekV2DecoderLayer.forward_sparse_attn_only = forward_sparse_attn_only
    DeepseekV2DecoderLayer.forward_mlp_only = forward_mlp_only
