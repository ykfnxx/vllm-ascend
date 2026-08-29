#include "prefetch_qli_fusion_tiling.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace {
constexpr uint32_t HIDDEN_STATES_INPUT = 0U;
constexpr uint32_t WQKV_INPUT = 1U;
constexpr uint32_t WS_QKV_INPUT = 2U;
constexpr uint32_t WQB_INPUT = 3U;
constexpr uint32_t WS_QB_INPUT = 4U;
constexpr uint32_t GAMMA1_INPUT = 5U;
constexpr uint32_t BETA1_INPUT = 6U;
constexpr uint32_t COS_INPUT = 7U;
constexpr uint32_t SIN_INPUT = 8U;
constexpr uint32_t WK_WEIGHTS_PROJ_INPUT = 9U;  // wk_weights_proj 权重 [head_dim+n_head, hidden] bf16
constexpr uint32_t ALPHA_VEC_INPUT = 10U;  // 可选：rank-aware alpha 系数向量 [N] bf16
constexpr uint32_t BETA_VEC_INPUT = 11U;  // 可选：rank-aware beta 系数向量 [N] bf16
constexpr uint32_t Q_LI_OUTPUT = 0U;
constexpr uint32_t WEIGHTS_OUTPUT = 1U;

constexpr uint32_t ATTR_Q_LORA_RANK = 0U;
constexpr uint32_t ATTR_N_HEAD = 1U;
constexpr uint32_t ATTR_HEAD_DIM = 2U;
constexpr uint32_t ATTR_QK_ROPE_HEAD_DIM = 3U;
constexpr uint32_t ATTR_ALPHA = 4U;
constexpr uint32_t ATTR_BETA = 5U;
constexpr uint32_t ATTR_EPS = 6U;
constexpr uint32_t ATTR_SOURCE_ROWS_BEFORE_GATHER = 7U;  // 可选：row_rank = min(row/R, N-1)

constexpr uint32_t M_CHUNK = 4U;
constexpr uint32_t ALIGN_16 = 16U;
constexpr uint32_t ALIGN_32 = 32U;
constexpr uint64_t WS_ALIGN = 32U;  // workspace 段对齐（matmul GM 输入/输出要求 32B）

// ============ moe GMMSetMMTiling 常量（对齐 csrc/moe_grouped_matmul） ============
constexpr uint32_t BEST_BASEN = 256;        // moe BEST_BASEN
constexpr uint32_t MAX_BASEM = 256;         // moe MAX_BASEM
constexpr uint64_t L1_PARTA_SIZE = 256UL * 1024UL;  // moe L1_PARTA_SIZE
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;       // moe DOUBLE_BUFFER_L0A_L0B
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2; // moe DOUBLE_BUFFER_STEPKA_STEPKB
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint32_t MM_DATA_TYPE_SIZE = 1;   // int8 元素宽 1B（moe CalMMTiling 对 bf16/fp16 用 2，本算子 int8 用 1）

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

inline uint32_t RoundUp(uint32_t v, uint32_t align)
{
    return align == 0 ? v : (v + align - 1) / align * align;
}

inline uint32_t SixteenAlignUp(uint32_t v)
{
    return (v + 15U) & ~15U;
}

inline uint32_t SixteenAlignDown(uint32_t v)
{
    return v & ~15U;
}

// ============ moe CalMMTiling：计算 baseM/baseN/baseK（int8 元素宽 1B） ============
// 返回 false 表示参数非法
static bool CalMMTiling(uint32_t m, uint64_t l0aSize, uint64_t l0bSize, uint64_t l0cSize,
                        uint32_t& baseM, uint32_t& baseN, uint32_t& baseK)
{
    baseN = BEST_BASEN;
    baseK = static_cast<uint32_t>((l0bSize / DOUBLE_BUFFER_L0A_L0B) /
                                  (static_cast<uint64_t>(baseN) * MM_DATA_TYPE_SIZE));
    baseK = SixteenAlignDown(baseK);
    uint32_t maxBaseM = static_cast<uint32_t>(l0cSize /
                                              (static_cast<uint64_t>(baseN) * FP32_DATATYPE_SIZE));
    baseM = std::min<uint32_t>(static_cast<uint32_t>((l0aSize / DOUBLE_BUFFER_L0A_L0B) /
                                                     (static_cast<uint64_t>(baseK) * MM_DATA_TYPE_SIZE)),
                               maxBaseM);
    baseM = baseM > m ? SixteenAlignUp(m) : SixteenAlignDown(baseM);
    if (baseM > MAX_BASEM) {
        baseM = MAX_BASEM;
    }
    if (baseM == 0 || baseK == 0) {
        return false;
    }
    return true;
}

// ============ moe CalcStepKaKb：根据 L1 分段计算 stepKa/stepKb ============
// 返回 false 表示 step==0 非法
static bool CalcStepKaKb(uint64_t l1Size, uint32_t baseM, uint32_t baseN, uint32_t baseK,
                         uint32_t& stepKa, uint32_t& stepKb)
{
    if (l1Size < L1_PARTA_SIZE) {
        return false;
    }
    uint64_t l1ASize = baseM > baseN ? L1_PARTA_SIZE : l1Size - L1_PARTA_SIZE;
    uint64_t l1BSize = l1Size - l1ASize;
    stepKa = static_cast<uint32_t>((l1ASize / 2UL) /
                                   (static_cast<uint64_t>(baseM) * baseK * MM_DATA_TYPE_SIZE));
    stepKb = static_cast<uint32_t>((l1BSize / 2UL) /
                                   (static_cast<uint64_t>(baseN) * baseK * MM_DATA_TYPE_SIZE));
    if (stepKa == 0 || stepKb == 0) {
        return false;
    }
    if (stepKa > stepKb) {
        stepKa = stepKa / stepKb * stepKb;
    } else if (stepKa < stepKb) {
        stepKb = stepKb / stepKa * stepKa;
    }
    return true;
}

// ============ mm3：wk_weights_proj 简化 bf16 tiling ============
// N=n_head 很小（32），用简单 matmul tiling（全 N 单块，不设 FixSplit），
// 避免 moe int8 参数在 bf16（2B）下导致 L0/L1 地址越界。
static ge::graphStatus RunMatmul3Tiling(platform_ascendc::PlatformAscendC& platform,
                                        uint32_t M, uint32_t K, uint32_t N,
                                        TCubeTiling& cubeTiling)
{
    matmul_tiling::MultiCoreMatmulTiling tilingApi(platform);
    tilingApi.SetDim(1);
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_BF16, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_BF16, true);  // B=[N_wk,K] ND + isTransB
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_BF16);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                          matmul_tiling::DataType::DT_BF16);
    tilingApi.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    tilingApi.SetShape(M, N, K);  // 全 N 单块
    uint64_t l1_size = 0, l0c_size = 0, ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    tilingApi.SetBufferSpace(static_cast<int64_t>(l1_size), static_cast<int64_t>(l0c_size),
                             static_cast<int64_t>(ub_size));
    tilingApi.EnableBias(false);
    int64_t res = tilingApi.GetTiling(cubeTiling);
    if (res == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============ Matmul tiling ============
static ge::graphStatus RunMatmulTiling(platform_ascendc::PlatformAscendC& platform,
                                       uint32_t coreNumAic, uint32_t M, uint32_t K, uint32_t N,
                                       uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                       uint32_t stepKa, uint32_t stepKb,
                                       TCubeTiling& cubeTiling, bool isBf16 = false)
{
    matmul_tiling::MultiCoreMatmulTiling tilingApi(platform);
    // Kernel-level SetSingleShape partitions M; the tiling object stays single-core.
    tilingApi.SetDim(1);
    if (isBf16) {
        // mm3：wk_weights_proj（BF16 非量化），C 直接 bf16 输出，无 bias
        tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_BF16, false);
        tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_BF16, false);  // B=[K,N] ND
        tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_BF16);
        tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                              matmul_tiling::DataType::DT_BF16);
    } else {
        tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_INT8, false);
        // B uses [K, N] FRACTAL_NZ and C keeps the raw int32 accumulator.
        tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ,
                           matmul_tiling::DataType::DT_INT8, false);  // [K,N] NZ
        tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_INT32);
        tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                              matmul_tiling::DataType::DT_INT32);
    }
    tilingApi.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    tilingApi.SetShape(M, baseN, K);   // N=baseN：逐 N-block（moe 范式）
    tilingApi.SetFixSplit(static_cast<int32_t>(baseM), static_cast<int32_t>(baseN),
                          static_cast<int32_t>(baseK));
    uint64_t l1_size = 0, l0c_size = 0, ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    // Use the platform-reported buffer sizes.
    tilingApi.SetBufferSpace(static_cast<int64_t>(l1_size), static_cast<int64_t>(l0c_size),
                             static_cast<int64_t>(ub_size));
    tilingApi.EnableBias(false);
    int64_t res = tilingApi.GetTiling(cubeTiling);
    if (res == -1) {
        return ge::GRAPH_FAILED;
    }
    // 显式覆写 base/step（moe GMMSetMMTiling L170-180）
    cubeTiling.set_shareMode(0);
    cubeTiling.set_dbL0C(1);  // 禁用 L0C double buffer
    cubeTiling.set_baseM(static_cast<int32_t>(baseM));
    cubeTiling.set_baseN(static_cast<int32_t>(baseN));
    cubeTiling.set_baseK(static_cast<int32_t>(baseK));
    cubeTiling.set_stepKa(static_cast<int32_t>(stepKa));
    cubeTiling.set_depthA1(static_cast<int32_t>(stepKa * DOUBLE_BUFFER_STEPKA_STEPKB * 1));
    cubeTiling.set_stepKb(static_cast<int32_t>(stepKb));
    cubeTiling.set_depthB1(static_cast<int32_t>(stepKb * DOUBLE_BUFFER_STEPKA_STEPKB * 1));
    cubeTiling.set_stepM(1);
    cubeTiling.set_stepN(1);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus CheckContracts(gert::TilingContext* context)
{
    const auto* hidden_desc = context->GetInputDesc(HIDDEN_STATES_INPUT);
    OPS_LOG_E_IF_NULL(context, hidden_desc, return ge::GRAPH_FAILED);
    OPS_ERR_IF(hidden_desc->GetDataType() != ge::DT_BF16,
               OPS_LOG_E(context->GetNodeName(), "hidden_states must be bfloat16."),
               return ge::GRAPH_FAILED);

    const auto* wqkv_desc = context->GetInputDesc(WQKV_INPUT);
    OPS_LOG_E_IF_NULL(context, wqkv_desc, return ge::GRAPH_FAILED);
    OPS_ERR_IF(wqkv_desc->GetDataType() != ge::DT_INT8,
               OPS_LOG_E(context->GetNodeName(), "wqkv must be int8."),
               return ge::GRAPH_FAILED);

    const auto* wqb_desc = context->GetInputDesc(WQB_INPUT);
    OPS_LOG_E_IF_NULL(context, wqb_desc, return ge::GRAPH_FAILED);
    OPS_ERR_IF(wqb_desc->GetDataType() != ge::DT_INT8,
               OPS_LOG_E(context->GetNodeName(), "wqb must be int8."),
               return ge::GRAPH_FAILED);

    const auto* gamma1_desc = context->GetInputDesc(GAMMA1_INPUT);
    OPS_LOG_E_IF_NULL(context, gamma1_desc, return ge::GRAPH_FAILED);
    OPS_ERR_IF(gamma1_desc->GetDataType() != ge::DT_BF16,
               OPS_LOG_E(context->GetNodeName(), "gamma1 must be bfloat16."),
               return ge::GRAPH_FAILED);

    const auto* beta1_desc = context->GetInputDesc(BETA1_INPUT);
    OPS_LOG_E_IF_NULL(context, beta1_desc, return ge::GRAPH_FAILED);
    OPS_ERR_IF(beta1_desc->GetDataType() != ge::DT_BF16,
               OPS_LOG_E(context->GetNodeName(), "beta1 must be bfloat16."),
               return ge::GRAPH_FAILED);

    const auto* q_li_desc = context->GetOutputDesc(Q_LI_OUTPUT);
    OPS_LOG_E_IF_NULL(context, q_li_desc, return ge::GRAPH_FAILED);
    OPS_ERR_IF(q_li_desc->GetDataType() != ge::DT_BF16,
               OPS_LOG_E(context->GetNodeName(), "q_li must be bfloat16."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
}  // namespace

static ge::graphStatus PrefetchQliFusionTilingFunc(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("PrefetchQliFusion", "TilingContext is nullptr."),
               return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* q_lora_rank_attr = attrs->GetAttrPointer<int64_t>(ATTR_Q_LORA_RANK);
    OPS_LOG_E_IF_NULL(context, q_lora_rank_attr, return ge::GRAPH_FAILED);
    const int64_t* n_head_attr = attrs->GetAttrPointer<int64_t>(ATTR_N_HEAD);
    OPS_LOG_E_IF_NULL(context, n_head_attr, return ge::GRAPH_FAILED);
    const int64_t* head_dim_attr = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_DIM);
    OPS_LOG_E_IF_NULL(context, head_dim_attr, return ge::GRAPH_FAILED);
    const int64_t* qk_rope_head_dim_attr = attrs->GetAttrPointer<int64_t>(ATTR_QK_ROPE_HEAD_DIM);
    OPS_LOG_E_IF_NULL(context, qk_rope_head_dim_attr, return ge::GRAPH_FAILED);
    const float* alpha_attr = attrs->GetAttrPointer<float>(ATTR_ALPHA);
    OPS_LOG_E_IF_NULL(context, alpha_attr, return ge::GRAPH_FAILED);
    const float* beta_attr = attrs->GetAttrPointer<float>(ATTR_BETA);
    OPS_LOG_E_IF_NULL(context, beta_attr, return ge::GRAPH_FAILED);
    const float* eps_attr = attrs->GetAttrPointer<float>(ATTR_EPS);
    OPS_LOG_E_IF_NULL(context, eps_attr, return ge::GRAPH_FAILED);

    // rank-aware alpha/beta（GLM-5.2 fused）：alpha_vec 输入存在 => 向量模式，
    // 每行 row_rank = min(row / source_rows_before_gather, N-1)。缺失 => 标量模式。
    const bool rankAware = (context->GetInputShape(ALPHA_VEC_INPUT) != nullptr);
    uint32_t sourceRowsBeforeGather = 0U;
    uint32_t alphaVecLen = 0U;
    if (rankAware) {
        const int64_t* srbg_attr = attrs->GetAttrPointer<int64_t>(ATTR_SOURCE_ROWS_BEFORE_GATHER);
        sourceRowsBeforeGather = (srbg_attr != nullptr) ? static_cast<uint32_t>(*srbg_attr) : 0U;
        const gert::StorageShape* alpha_vec_shape = context->GetInputShape(ALPHA_VEC_INPUT);
        if (alpha_vec_shape != nullptr) {
            const gert::Shape& alpha_vec_storage = alpha_vec_shape->GetStorageShape();
            alphaVecLen = (alpha_vec_storage.GetDimNum() >= 1)
                              ? static_cast<uint32_t>(alpha_vec_storage.GetDim(0))
                              : 0U;
        }
    }

    if (CheckContracts(context) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Tensor contract check failed.");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* hidden_shape = context->GetInputShape(HIDDEN_STATES_INPUT);
    OPS_LOG_E_IF_NULL(context, hidden_shape, return ge::GRAPH_FAILED);
    const gert::Shape& hidden = hidden_shape->GetStorageShape();
    OPS_ERR_IF(hidden.GetDimNum() < 2,
               OPS_LOG_E(context->GetNodeName(), "hidden_states rank must be >= 2."),
               return ge::GRAPH_FAILED);
    uint32_t T = static_cast<uint32_t>(hidden.GetDim(0));
    uint32_t hiddenSize = static_cast<uint32_t>(hidden.GetDim(1));

    uint32_t qLoraRank = static_cast<uint32_t>(*q_lora_rank_attr);
    uint32_t nHead = static_cast<uint32_t>(*n_head_attr);
    uint32_t headDim = static_cast<uint32_t>(*head_dim_attr);
    uint32_t ropeDim = static_cast<uint32_t>(*qk_rope_head_dim_attr);

    const gert::StorageShape* ws_qkv_shape = context->GetInputShape(WS_QKV_INPUT);
    OPS_LOG_E_IF_NULL(context, ws_qkv_shape, return ge::GRAPH_FAILED);
    const gert::Shape& ws_qkv_storage = ws_qkv_shape->GetStorageShape();
    OPS_ERR_IF(ws_qkv_storage.GetDimNum() < 1,
               OPS_LOG_E(context->GetNodeName(), "ws_qkv must have rank >= 1."),
               return ge::GRAPH_FAILED);
    uint32_t nQkv = static_cast<uint32_t>(ws_qkv_storage.GetDim(0));

    const gert::StorageShape* ws_qb_shape = context->GetInputShape(WS_QB_INPUT);
    OPS_LOG_E_IF_NULL(context, ws_qb_shape, return ge::GRAPH_FAILED);
    const gert::Shape& ws_qb_storage = ws_qb_shape->GetStorageShape();
    OPS_ERR_IF(ws_qb_storage.GetDimNum() < 1,
               OPS_LOG_E(context->GetNodeName(), "ws_qb must have rank >= 1."),
               return ge::GRAPH_FAILED);
    uint32_t nQb = static_cast<uint32_t>(ws_qb_storage.GetDim(0));

    fe::PlatFormInfos* platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
    uint32_t coreNumAic = ascendc_platform.GetCoreNumAic();
    OPS_ERR_IF(coreNumAic == 0, OPS_LOG_E(context->GetNodeName(), "AIC core count is 0."),
               return ge::GRAPH_FAILED);
    uint64_t l0aSize = 0, l0bSize = 0, l0cSize = 0, l1Size = 0;
    ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize);
    ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize);
    ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);

    // ============ moe CalMMTiling：baseM/baseN/baseK（两段 matmul 的 M 均为 T） ============
    uint32_t baseM = 0, baseN = 0, baseK = 0;
    if (!CalMMTiling(T, l0aSize, l0bSize, l0cSize, baseM, baseN, baseK)) {
        OPS_LOG_E(context->GetNodeName(), "CalMMTiling failed.");
        return ge::GRAPH_FAILED;
    }
    uint32_t stepKa = 0, stepKb = 0;
    if (!CalcStepKaKb(l1Size, baseM, baseN, baseK, stepKa, stepKb)) {
        OPS_LOG_E(context->GetNodeName(), "CalcStepKaKb failed.");
        return ge::GRAPH_FAILED;
    }

    // ============ 2D AIC grid (mCore × nCore) ============
    // N-axis partitioning limits each core to its own weight slice.
    //   - 小 T（T < baseM，weight-bound）：mCore=1（M 不切），nCore 用满所有 AIC 核；
    //   - 大 T：按 baseM 粒度 M 切（singleM ≤ baseM，对齐 moe 范式），剩余核做 N 切。
    // AIV 侧不改计算语义：AIV 独立按全行段从 GM 读完整 qkv/qf 行（跨核归并在 GM 完成，
    // 由 B2/B4 全局 barrier 保证所有 N 分片 AIC 写盘完成），RMSNorm/RoPE 仍消费完整行。
    uint32_t mCore = 1U;
    if (T >= baseM) {
        mCore = CeilDiv(T, baseM);
        if (mCore > coreNumAic) { mCore = coreNumAic; }
    }
    if (mCore == 0) { mCore = 1U; }
    // nCore = 剩余 AIC 核做 N 切分。不额外按 baseN 块数封顶：mm2(N=8192) 有 32 个 baseN 块，
    // 20 核内每核都能分到 ≥1 个块；mm1 块数少时部分核 mm1 分片为空（仅参与 mm2），
    // 权重按核均衡分配由 ComputeNSlice 的「baseBlocks + 余数」策略保证。
    uint32_t nCore = coreNumAic / mCore;
    if (nCore < 1U) { nCore = 1U; }
    // Cap nCore at 16 to avoid barrier-only cores when mm1 has fewer N blocks.
    if (nCore > 16U) { nCore = 16U; }
    uint32_t usedCoreNum = mCore * nCore;
    if (usedCoreNum == 0) { usedCoreNum = 1U; }
    uint32_t singleM = CeilDiv(T, mCore);
    if (singleM == 0) { singleM = 1U; }

    PrefetchQliFusionTilingData tiling;
    tiling.set_tokenNum(T);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_singleM(singleM);
    tiling.set_mCore(mCore);
    tiling.set_nCore(nCore);
    tiling.set_baseN(baseN);
    tiling.set_mChunk(M_CHUNK);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_qLoraRank(qLoraRank);
    tiling.set_nQkv(nQkv);
    tiling.set_nQb(nQb);
    tiling.set_nHead(nHead);
    tiling.set_headDim(headDim);
    tiling.set_ropeDim(ropeDim);
    tiling.set_alpha(*alpha_attr);
    tiling.set_beta(*beta_attr);
    tiling.set_eps(*eps_attr);
    tiling.set_invQloraRank(1.0f / static_cast<float>(qLoraRank));
    tiling.set_invHiddenSize(1.0f / static_cast<float>(hiddenSize));
    tiling.set_alphaBetaMode(rankAware ? 1U : 0U);
    tiling.set_sourceRowsBeforeGather(sourceRowsBeforeGather);
    tiling.set_alphaVecLen(alphaVecLen);

    // 两段 W8A8 Matmul tiling（moe SetFixSplit 范式；C=int32 raw，反量化移 UB）
    if (RunMatmulTiling(ascendc_platform, usedCoreNum, T, hiddenSize, nQkv,
                        baseM, baseN, baseK, stepKa, stepKb,
                        tiling.cubeTiling1) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "matmul1 tiling failed.");
        return ge::GRAPH_FAILED;
    }
    if (RunMatmulTiling(ascendc_platform, usedCoreNum, T, qLoraRank, nQb,
                        baseM, baseN, baseK, stepKa, stepKb,
                        tiling.cubeTiling2) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "matmul2 tiling failed.");
        return ge::GRAPH_FAILED;
    }
    // mm3：wk_weights_proj（BF16），M=T, K=hiddenSize, N=n_head（只算 weights 列）
    // 用简化的 bf16 matmul tiling（不走 moe SetFixSplit，避免 bf16 下 int8 参数溢出）
    if (RunMatmul3Tiling(ascendc_platform, T, hiddenSize, nHead,
                         tiling.cubeTiling3) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "matmul3(wk_weights_proj) tiling failed.");
        return ge::GRAPH_FAILED;
    }

    // workspace 分配（每段 32B 对齐，满足 matmul GM 输入/输出对齐要求）:
    //   xq (int8) [T, hiddenSize], s_tok (fp32) [T],
    //   qkv_out (int32) [T, nQkv], xcq (int8) [T, qLoraRank], s_tok2 (fp32) [T],
    //   qf (int32) [T, nQb]
    // 尾部追加 matmul libapi 同步 workspace（moe 范式，moe_grouped_matmul_cpu.cpp L267/L304）。
    uint64_t totalWs = 0;
    uint64_t wsXq = totalWs; totalWs += RoundUp(static_cast<uint64_t>(T) * hiddenSize, WS_ALIGN);
    uint64_t wsStok = totalWs; totalWs += RoundUp(static_cast<uint64_t>(T) * sizeof(float), WS_ALIGN);
    uint64_t wsQkvOut = totalWs;
    totalWs += RoundUp(static_cast<uint64_t>(T) * nQkv * sizeof(int32_t), WS_ALIGN);
    uint64_t wsXcq = totalWs; totalWs += RoundUp(static_cast<uint64_t>(T) * qLoraRank, WS_ALIGN);
    uint64_t wsStok2 = totalWs; totalWs += RoundUp(static_cast<uint64_t>(T) * sizeof(float), WS_ALIGN);
    uint64_t wsQf = totalWs;
    totalWs += RoundUp(static_cast<uint64_t>(T) * nQb * sizeof(int32_t), WS_ALIGN);
    // wsPred：predicted_hidden bf16 [T, hiddenSize]（mm3 的 A 输入，AIV quant1 写出）
    uint64_t wsPred = totalWs;
    totalWs += RoundUp(static_cast<uint64_t>(T) * hiddenSize * 2U, WS_ALIGN);  // bf16 2B
    // wsWeights：mm3 C 输出 bf16 [T, n_head]（AIV 再拷到输出 GM）
    uint64_t wsWeights = totalWs;
    totalWs += RoundUp(static_cast<uint64_t>(T) * nHead * 2U, WS_ALIGN);  // bf16 2B
    totalWs = RoundUp(totalWs, 64U);  // libapi 段 64B 对齐
    totalWs += static_cast<uint64_t>(ascendc_platform.GetLibApiWorkSpaceSize());

    tiling.set_wsXqOffset(wsXq);
    tiling.set_wsStokOffset(wsStok);
    tiling.set_wsQkvOutOffset(wsQkvOut);
    tiling.set_wsXcqOffset(wsXcq);
    tiling.set_wsStok2Offset(wsStok2);
    tiling.set_wsQfOffset(wsQf);
    tiling.set_wsPredOffset(wsPred);
    tiling.set_wsWeightsOffset(wsWeights);

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL(context, workSpaces, return ge::GRAPH_FAILED);
    workSpaces[0] = static_cast<size_t>(totalWs);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct PrefetchQliFusionCompileInfo {};

static ge::graphStatus TilingParseForPrefetchQliFusion(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PrefetchQliFusion)
    .Tiling(PrefetchQliFusionTilingFunc)
    .TilingParse<PrefetchQliFusionCompileInfo>(TilingParseForPrefetchQliFusion);
}  // namespace optiling
