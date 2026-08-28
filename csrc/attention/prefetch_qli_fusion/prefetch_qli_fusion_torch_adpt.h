#ifndef PREFETCH_QLI_FUSION_TORCH_ADPT_H
#define PREFETCH_QLI_FUSION_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> prefetch_qli_fusion(
    const at::Tensor& hidden_states,
    const at::Tensor& wqkv, const at::Tensor& ws_qkv,
    const at::Tensor& wqb,  const at::Tensor& ws_qb,
    const at::Tensor& gamma1, const at::Tensor& beta1,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& wk_weights_proj,
    int64_t q_lora_rank, int64_t n_head, int64_t head_dim,
    int64_t qk_rope_head_dim, double alpha, double beta, double eps,
    int64_t source_rows_before_gather,
    const c10::optional<at::Tensor>& alpha_vec,
    const c10::optional<at::Tensor>& beta_vec)
{
    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16,
                "hidden_states must be bfloat16");
    TORCH_CHECK(hidden_states.dim() == 2,
                "hidden_states must be [T, hidden_size]");
    TORCH_CHECK(wqkv.scalar_type() == at::kChar,
                "wqkv must be int8 (FRACTAL_NZ)");
    TORCH_CHECK(wqb.scalar_type() == at::kChar,
                "wqb must be int8 (FRACTAL_NZ)");
    TORCH_CHECK(gamma1.scalar_type() == at::kBFloat16,
                "gamma1 must be bfloat16");
    TORCH_CHECK(beta1.scalar_type() == at::kBFloat16,
                "beta1 must be bfloat16");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16, "cos must be bfloat16");
    TORCH_CHECK(sin.scalar_type() == at::kBFloat16, "sin must be bfloat16");
    TORCH_CHECK(wk_weights_proj.scalar_type() == at::kBFloat16,
                "wk_weights_proj must be bfloat16");
    // rank-aware 模式：alpha_vec/beta_vec 必须同时提供（同为 bf16 [N]）
    if (alpha_vec.has_value() != beta_vec.has_value()) {
        TORCH_CHECK(false, "alpha_vec and beta_vec must be provided together");
    }
    if (alpha_vec.has_value()) {
        TORCH_CHECK(alpha_vec->scalar_type() == at::kBFloat16,
                    "alpha_vec must be bfloat16");
        TORCH_CHECK(beta_vec->scalar_type() == at::kBFloat16,
                    "beta_vec must be bfloat16");
        TORCH_CHECK(alpha_vec->dim() == 1 && beta_vec->dim() == 1 &&
                        alpha_vec->size(0) == beta_vec->size(0),
                    "alpha_vec/beta_vec must be 1-D with equal length [N]");
    }

    const int64_t T = hidden_states.size(0);
    at::Tensor q_li = at::empty({T, n_head, head_dim}, hidden_states.options());
    at::Tensor weights = at::empty({T, n_head}, hidden_states.options());

    EXEC_NPU_CMD(aclnnPrefetchQliFusion,
                 hidden_states, wqkv, ws_qkv, wqb, ws_qb,
                 gamma1, beta1, cos, sin,
                 wk_weights_proj, alpha_vec, beta_vec,
                 q_lora_rank, n_head, head_dim, qk_rope_head_dim,
                 alpha, beta, eps, source_rows_before_gather,
                 q_li, weights);
    return {q_li, weights};
}

}  // namespace vllm_ascend

#endif  // PREFETCH_QLI_FUSION_TORCH_ADPT_H
