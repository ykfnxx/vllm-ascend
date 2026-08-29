#include "register/op_def_registry.h"

namespace ops {
class PrefetchQliFusion : public OpDef {
public:
    explicit PrefetchQliFusion(const char* name) : OpDef(name)
    {
        this->Input("hidden_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wqkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("ws_qkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wqb")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("ws_qb")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        // wk_weights_proj 融合（weights-only projection）：BF16 非量化权重 [head_dim+n_head, hidden]。
        // kernel 内部只算后半 n_head 列（weights），输出 [T, n_head]，与 q_li 共享 predicted_hidden。
        this->Input("wk_weights_proj")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        // rank-aware alpha/beta（GLM-5.2 grouped prefetch 融合）：可选 per-rank 系数向量 [N] bf16。
        // 提供时进入 rank-aware 模式：predicted_hidden[row] = hidden[row]*alpha_vec[row_rank]
        // + beta_vec[row_rank]，row_rank = min(row / source_rows_before_gather, N-1)。
        // 缺失时保持标量模式（alpha/beta 属性），行为不变。
        this->Input("alpha_vec")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta_vec")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("q_li")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("q_lora_rank").AttrType(REQUIRED).Int();
        this->Attr("n_head").AttrType(REQUIRED).Int();
        this->Attr("head_dim").AttrType(REQUIRED).Int();
        this->Attr("qk_rope_head_dim").AttrType(REQUIRED).Int();
        this->Attr("alpha").AttrType(REQUIRED).Float();
        this->Attr("beta").AttrType(REQUIRED).Float();
        this->Attr("eps").AttrType(REQUIRED).Float();
        // rank-aware 模式下行 rank 划分：row_rank = min(row / source_rows_before_gather, N-1)
        this->Attr("source_rows_before_gather").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(PrefetchQliFusion);
}  // namespace ops
