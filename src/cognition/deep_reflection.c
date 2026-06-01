/**
 * @file deep_reflection.c
 * @brief 深度反思引擎完整实现
 *
 * 多维度认知审视：信念冲突检测、认识论评估、逻辑一致性检验、
 * 多段落反思、知识综合、假设生成与冲突解决。
 *
 * ============================================================
 * 【模块职责 - ZSFWS-028 认知三模块边界】
 * ============================================================
 * 本模块（深度反思）的核心职责：主动内省式认知一致性审查系统
 *
 * 独特功能：
 *   - 多维度自我反思（dr_reflect / dr_reflect_multi_passage）
 *   - 认识论确定性评估——审视"我有多确定"（dr_epistemic_assessment）
 *   - 知识库内部冲突检测与解决（dr_detect_knowledge_conflicts / dr_resolve_conflicts）
 *   - 从反思结果生成新认知假设（dr_generate_hypotheses）
 *   - 认知根因分析——寻找思维过程本身的逻辑矛盾（dr_root_cause_analysis）
 *   - 检测信念体系中自相矛盾的命题（dr_detect_contradictions）
 *   - 多视角综合以消除分歧（dr_generate_synthesis）
 *   - 反思结果转化为行动计划（dr_chain_to_plan）
 *   - 与元认知层集成（dr_integrate_with_metacognition）
 *
 * 与 deep_correction.c 的区别：
 *   - deep_reflection 是"主动反思"：无外部错误触发，定期/按需审视内部一致性
 *   - deep_correction 是"被动修复"：必须有错误报告才能触发
 *   - reflection关注"我有没有自相矛盾？"，correction关注"这个错误是什么导致的？"
 *
 * 与 deep_thought_chain.c 的区别：
 *   - deep_reflection 审视已有的信念/知识，不创造新推理链
 *   - deep_thought_chain 从零开始构建解决新问题的推理路径
 *   - reflection输出的是"认知状态审查报告"，thought_chain输出的是"推理路径"
 *
 * ⚠️  功能重叠提示：
 *   - dr_root_cause_analysis（认知根因）与 dc_analyze_root_cause（错误根因）：
 *     建议：两者可共享贝叶斯推理底层的概率传播机制，但输入不同
 *   - dr_chain_to_plan（反思→计划）与 dtc_chain_to_plan（思考链→计划）：
 *     建议：reflection的chain_to_plan应附加"反思来源"标记，thought_chain的不需
 *   - dr_generate_hypotheses 与 dc_generate_hypotheses：
 *     建议：reflection假设需标记为"探索性假设"（无关联错误），correction假设标记为"修正性假设"（有关联error_id）
 * ============================================================
 */
#include "selflnn/cognition/deep_reflection.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/laplace.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>

#define DR_SIGMOID(x) (1.0f / (1.0f + expf(-(x))))
#define DR_TANH(x) (tanhf(x))
#define DR_MAX(a,b) (((a)>(b))?(a):(b))
#define DR_MIN(a,b) (((a)<(b))?(a):(b))
#define DR_CLAMP(v,lo,hi) DR_MIN(DR_MAX((v),(lo)),(hi))
#define DR_ABS(x) (((x)>=0)?(x):(-(x)))

struct DeepReflectionEngine {
    DRConfig config;
    LNN* reflection_net;
    LNN* synthesis_net;
    LNN* conflict_net;
    LNN* epistemic_net;
    LNN* hypothesis_net;
    int owns_reflection;   /* 所有权标记 */
    int owns_synthesis;
    int owns_conflict;
    int owns_epistemic;
    int owns_hypothesis;
    int conflict_net_warmed;        /**< ZSFZS-F017: conflict_net是否已完成预热 */
    int lnn_conflict_epochs;        /**< ZSFZS-F017: conflict_net预热迭代次数 */
    float* layer_embeddings;
    size_t layer_embed_count;
    float* knowledge_base;
    size_t knowledge_size;
    float* conflict_history;
    size_t conflict_count;
    int initialized;
};

/* F-023修复：使用统一余弦相似度替代本地实现 */
#define cosine_sim_dr(a, b, dim) math_cosine_similarity((a), (b), (dim))

static float euclidean_dist_dr(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

static float bayesian_conflict_score(const float* belief_a, const float* belief_b,
                                      float prior_a, float prior_b, size_t dim) {
    float sim = cosine_sim_dr(belief_a, belief_b, dim);
    float evidence_ratio = (1.0f - sim) / (sim + 1e-10f);
    float posterior_a = (prior_a * (1.0f - sim)) / (prior_a * (1.0f - sim) + (1.0f - prior_a) * sim + 1e-10f);
    float posterior_b = (prior_b * (1.0f - sim)) / (prior_b * (1.0f - sim) + (1.0f - prior_b) * sim + 1e-10f);
    float divergence = DR_ABS(posterior_a - prior_a) + DR_ABS(posterior_b - prior_b);
    return DR_CLAMP(divergence * evidence_ratio, 0.0f, 1.0f);
}

static void weighted_embedding_fusion(const float** embeddings, const float* weights,
                                       size_t count, size_t dim, float* out) {
    memset(out, 0, dim * sizeof(float));
    float total_weight = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float w = (weights) ? weights[i] : 1.0f;
        for (size_t j = 0; j < dim; j++) {
            out[j] += embeddings[i][j] * w;
        }
        total_weight += w;
    }
    if (total_weight > 1e-10f) {
        float inv = 1.0f / total_weight;
        for (size_t j = 0; j < dim; j++) out[j] *= inv;
    }
}

static void generate_perspective_embedding(LNN* lnn, const float* base, int perspective_idx,
                                            float diversity, size_t dim, float* out) {
    /* F-002修复: 使用LNN前向传播生成真实语义多视角嵌入
     * 通过频域调制输入嵌入来产生不同视角，而非确定性数学噪声 */
    float modulated_input[DR_EMBED_DIM];
    for (size_t j = 0; j < dim; j++) {
        float phase = (float)(perspective_idx) * 2.0f * 3.14159265f / 5.0f;
        float modulator = 0.5f + 0.5f * cosf(phase + (float)j * 0.15f);
        modulated_input[j] = base[j] * (0.3f + modulator * diversity);
    }
    if (lnn) {
        lnn_forward(lnn, modulated_input, out);
    } else {
        memcpy(out, modulated_input, dim * sizeof(float));
    }
    float norm = 0.0f;
    for (size_t j = 0; j < dim; j++) norm += out[j] * out[j];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (size_t j = 0; j < dim; j++) out[j] /= norm;
    }
}

static const char* layer_names[] = {
    "描述性分析",
    "分析性分析",
    "批判性分析",
    "因果性分析",
    "认知论分析",
    "元认知分析",
    "变革性重构"
};

static const char* perspective_names[] = {
    "客观事实视角",
    "逻辑推理视角",
    "经验归纳视角",
    "因果推断视角",
    "反事实假设视角",
    "多智能体协商视角",
    "第一性原理视角",
    "系统整体视角"
};

/* ===================================================================
 * S-014修复: 深度反思5维分析——用真实算法替代简化评分公式
 *
 * 原实现仅用cosine_similarity+layer_index计算评分，
 * 新实现包含四个独立分析维度:
 *   1. 信念网络一致性检测（逻辑一致性检验）
 *   2. 假设检验（反例生成与验证）
 *   3. 矛盾检测（跨知识库矛盾发现）
 *   4. 风险评估（决策路径失败概率）
 * =================================================================== */

/* ---- 信念网络一致性检测 ----
 * 检验当前层嵌入与知识库中信念三元组的逻辑一致性
 * 算法: 对知识库中每对信念(h,t)和关系r, 验证h+r≈t的一致性
 * 返回: 一致性评分 [0,1]，越高表示越一致 */
static float dr_belief_consistency_check(DeepReflectionEngine* engine,
                                          const float* layer_embedding,
                                          size_t edim) {
    if (!engine || !layer_embedding) return 0.5f;
    if (!engine->knowledge_base || engine->knowledge_size == 0) return 0.5f;
    /* 知识库存储为连续的嵌入向量，每edim维为一个信念节点
     * 检验: 当前embedding与每个KB信念的余弦相似度分布 */
    size_t kb_nodes = engine->knowledge_size;
    if (kb_nodes > 1024) kb_nodes = 1024;
    float total_consistency = 0.0f;
    float total_weight = 0.0f;
    int pair_count = 0;
    /* 遍历KB中连续的三元组 (subject, predicate, object) 结构
     * 即 knowledge_base 中每3个连续嵌入构成一个三元组信念 */
    for (size_t t = 0; t + 2 < kb_nodes && t < 1021; t += 3) {
        const float* subj = engine->knowledge_base + t * edim;
        const float* pred = engine->knowledge_base + (t + 1) * edim;
        const float* obj = engine->knowledge_base + (t + 2) * edim;
        /* 信念传导: subj + pred ≈ obj 的语义一致性
         * 同时检验 layer_embedding 与每个信念节点的关系 */
        float sim_subj = cosine_sim_dr(layer_embedding, subj, edim);
        float sim_pred = cosine_sim_dr(layer_embedding, pred, edim);
        float sim_obj = cosine_sim_dr(layer_embedding, obj, edim);
        /* 三元组内部一致性: subj通过pred映射到obj的保真度 */
        float subj_pred_proj[DR_EMBED_DIM];
        for (size_t j = 0; j < edim; j++) {
            subj_pred_proj[j] = subj[j] + pred[j];
        }
        float sim_proj_obj = cosine_sim_dr(subj_pred_proj, obj, edim);
        /* 综合一致性: 当前嵌入与三元组的亲和度 + 三元组内部保真度 */
        float triad_consistency = (sim_subj + sim_pred + sim_obj) * 0.25f + sim_proj_obj * 0.25f;
        float weight = (sim_subj + sim_obj) * 0.5f + 0.1f;
        total_consistency += triad_consistency * weight;
        total_weight += weight;
        pair_count++;
    }
    if (pair_count == 0 || total_weight < 1e-10f) return 0.5f;
    float belief_score = total_consistency / total_weight;
    return DR_CLAMP(belief_score, 0.0f, 1.0f);
}

/* ---- 假设检验 ----
 * 生成反例假设，验证当前假设的鲁棒性
 * 算法: 对当前嵌入生成2个扰动变体（正变体和反例变体），
 * 通过hypothesis_net推理，比较输出一致性
 * 返回: 假设强度评分 [0,1]，越高表示假设越稳健 */
static float dr_hypothesis_test(DeepReflectionEngine* engine,
                                 const float* layer_embedding,
                                 size_t edim, float* hypothesis_out_embed) {
    if (!engine || !layer_embedding || !engine->hypothesis_net) return 0.5f;
    /* 生成正变体: 小幅正扰动 */
    float pos_variant[DR_EMBED_DIM];
    for (size_t j = 0; j < edim; j++) {
        float pert = layer_embedding[j] * 0.08f;
        pos_variant[j] = layer_embedding[j] + pert;
    }
    /* 生成反例变体: 翻转部分特征符号 */
    float neg_variant[DR_EMBED_DIM];
    for (size_t j = 0; j < edim; j++) {
        float pert = -layer_embedding[j] * 0.12f;
        neg_variant[j] = layer_embedding[j] + pert;
    }
    /* 分别推理 */
    float pos_out[DR_EMBED_DIM] = {0};
    float neg_out[DR_EMBED_DIM] = {0};
    lnn_forward(engine->hypothesis_net, pos_variant, pos_out);
    lnn_forward(engine->hypothesis_net, neg_variant, neg_out);
    /* 计算反例与正例输出的分歧度 */
    float pos_neg_divergence = 1.0f - cosine_sim_dr(pos_out, neg_out, edim);
    /* 使用reflection_net进一步验证 */
    float ref_pos_out[DR_EMBED_DIM] = {0};
    float ref_neg_out[DR_EMBED_DIM] = {0};
    if (engine->reflection_net) {
        lnn_forward(engine->reflection_net, pos_out, ref_pos_out);
        lnn_forward(engine->reflection_net, neg_out, ref_neg_out);
    } else {
        memcpy(ref_pos_out, pos_out, edim * sizeof(float));
        memcpy(ref_neg_out, neg_out, edim * sizeof(float));
    }
    float ref_divergence = 1.0f - cosine_sim_dr(ref_pos_out, ref_neg_out, edim);
    /* 原始假设输出（使用原始嵌入） */
    float orig_out[DR_EMBED_DIM] = {0};
    float orig_in[DR_EMBED_DIM];
    memcpy(orig_in, layer_embedding, edim * sizeof(float));
    lnn_forward(engine->hypothesis_net, orig_in, orig_out);
    if (hypothesis_out_embed) {
        memcpy(hypothesis_out_embed, orig_out, edim * sizeof(float));
    }
    float orig_pos_sim = cosine_sim_dr(orig_out, pos_out, edim);
    /* 假设强度: 正变体与原始一致 + 反例有低分歧度 → 假设稳健 */
    float hyp_strength = orig_pos_sim * 0.5f +
                         (1.0f - pos_neg_divergence) * 0.25f +
                         (1.0f - ref_divergence) * 0.25f;
    return DR_CLAMP(hyp_strength, 0.0f, 1.0f);
}

/* ---- 矛盾检测 ----
 * 跨知识库/跨层检测相互矛盾的断言
 * 算法: 使用conflict_net对当前嵌入与知识库条目做二分类
 * 检测高度矛盾对，结合历史矛盾记录
 * 返回: 矛盾强度 [0,1]，越高表示矛盾越严重
 *       contradiction_pairs: 矛盾层索引对（输出） */
static float dr_cross_contradiction_check(DeepReflectionEngine* engine,
                                           const float* layer_embedding,
                                           const float* prev_embeddings,
                                           int num_prev, size_t edim,
                                           int* out_contradiction_count) {
    if (!engine || !layer_embedding) { if (out_contradiction_count) *out_contradiction_count = 0; return 0.0f; }
    float max_conflict = 0.0f;
    int conflict_count = 0;
    /* 与之前各层的嵌入配对，送入conflict_net检测矛盾 */
    for (int p = 0; p < num_prev && p < 12; p++) {
        if (!prev_embeddings) break;
        float con_input[DR_EMBED_DIM * 2];
        memcpy(con_input, layer_embedding, edim * sizeof(float));
        memcpy(con_input + edim, prev_embeddings + (size_t)p * edim, edim * sizeof(float));
        float con_out[1] = {0};
        if (engine->conflict_net) {
            /* ZSFZS-F017修复: conflict_net未充分训练时，使用Xavier初始化后
             * 先执行一次无监督预热（基于余弦距离的自蒸馏），
             * 然后用预热后的网络检测矛盾。预热在首次调用时自动触发。 */
            if (!engine->conflict_net_warmed && engine->lnn_conflict_epochs < 3) {
                /* 自蒸馏预热：让网络输出接近余弦距离，提供合理的初始嵌入 */
                float* temp_out = (float*)calloc(edim * 2, sizeof(float));
                if (temp_out) {
                    for (size_t j = 0; j < edim; j++) {
                        temp_out[j] = layer_embedding[j];
                        temp_out[edim + j] = prev_embeddings ? prev_embeddings[j] : 0.0f;
                    }
                    float cos_dist = 1.0f - cosine_sim_dr(layer_embedding,
                        prev_embeddings ? prev_embeddings : layer_embedding, edim);
                    float target[1] = { cos_dist };
                    /* 单步SGD预热，学习率0.01 */
                    for (int wu = 0; wu < 5; wu++) {
                        lnn_forward(engine->conflict_net, temp_out, con_out);
                        float err = con_out[0] - target[0];
                        float grad[1] = { err * 0.01f };
                        lnn_backward(engine->conflict_net, grad, NULL);
                    }
                    engine->lnn_conflict_epochs++;
                    if (engine->lnn_conflict_epochs >= 3) engine->conflict_net_warmed = 1;
                    free(temp_out);
                }
            }
            lnn_forward(engine->conflict_net, con_input, con_out);
        } else {
            /* 无conflict_net时使用余弦距离作为替代 */
            con_out[0] = 1.0f - cosine_sim_dr(layer_embedding, prev_embeddings + (size_t)p * edim, edim);
        }
        float conflict = DR_SIGMOID(con_out[0]);
        if (conflict > 0.4f) {
            conflict_count++;
        }
        if (conflict > max_conflict) max_conflict = conflict;
    }
    /* 检查知识库中是否已有类似矛盾记录 */
    if (engine->conflict_history && engine->conflict_count > 0) {
        size_t hist_limit = engine->conflict_count < 64 ? engine->conflict_count : 64;
        for (size_t h = 0; h < hist_limit; h++) {
            float hist_conflict = engine->conflict_history[h * 4 + 2];
            if (hist_conflict > max_conflict) max_conflict = hist_conflict;
        }
    }
    if (out_contradiction_count) *out_contradiction_count = conflict_count;
    return DR_CLAMP(max_conflict, 0.0f, 1.0f);
}

/* ---- 风险评估 ----
 * 评估当前决策路径的失败概率
 * 算法: 综合认知不确定性、假设强度、矛盾程度、以及深度进展趋势
 * 使用epistemic_net获取认知状态，结合历史趋势预测风险
 * 返回: 失败概率 [0,1] */
static float dr_risk_assessment(DeepReflectionEngine* engine,
                                 const float* layer_embedding,
                                 float depth_score, float coherence_score,
                                 float hypothesis_strength, float contradiction_level,
                                 size_t edim, int layer_idx, float prev_risk) {
    if (!engine || !layer_embedding) return 0.1f;
    /* 认知不确定性: 通过epistemic_net获取 */
    float epi_out[3] = {0};
    if (engine->epistemic_net) {
        float epi_in[DR_EMBED_DIM];
        memcpy(epi_in, layer_embedding, edim * sizeof(float));
        lnn_forward(engine->epistemic_net, epi_in, epi_out);
    }
    float epistemic_uncertainty = (engine->epistemic_net) ? DR_SIGMOID(epi_out[1]) : 0.3f;
    float epistemic_confidence = (engine->epistemic_net) ? DR_SIGMOID(epi_out[0]) : 0.5f;
    /* 风险成分:
     *   不确定性贡献 + 低一致性贡献 + 低假设强度 + 高矛盾 */
    float risk = 0.0f;
    risk += epistemic_uncertainty * 0.30f;
    risk += (1.0f - coherence_score) * 0.25f;
    risk += (1.0f - hypothesis_strength) * 0.20f;
    risk += contradiction_level * 0.15f;
    /* 深度惩罚: 更深的层如果一致性低，风险更大 */
    float depth_risk = depth_score * (1.0f - coherence_score) * 0.10f;
    risk += depth_risk;
    /* 趋势因子: 与前一层的风险进行比较 */
    if (layer_idx > 0 && prev_risk > 0.0f) {
        float trend = (risk - prev_risk);
        if (trend > 0) {
            risk += trend * 0.15f;
        }
    }
    return DR_CLAMP(risk, 0.01f, 1.0f);
}

DeepReflectionEngine* dr_engine_create(DRConfig config) {
    DeepReflectionEngine* engine = (DeepReflectionEngine*)safe_calloc(1, sizeof(DeepReflectionEngine));
    if (!engine) return NULL;

    engine->config = config;
    engine->initialized = 1;
    engine->layer_embed_count = 0;
    engine->knowledge_size = 0;
    engine->conflict_count = 0;

    size_t edim = (config.embedding_dim > 0) ? config.embedding_dim : DR_EMBED_DIM;

    engine->owns_reflection = 1;
    engine->owns_synthesis = 1;
    engine->owns_conflict = 1;
    engine->owns_epistemic = 1;
    engine->owns_hypothesis = 1;

    LNNConfig ref_cfg = {0};
    ref_cfg.input_size = (int)edim;
    ref_cfg.hidden_size = 512;
    ref_cfg.output_size = (int)edim;
    ref_cfg.num_layers = 2;
    engine->reflection_net = lnn_create(&ref_cfg);

    LNNConfig syn_cfg = {0};
    syn_cfg.input_size = (int)(edim * 2);
    syn_cfg.hidden_size = 512;
    syn_cfg.output_size = (int)edim;
    syn_cfg.num_layers = 2;
    engine->synthesis_net = lnn_create(&syn_cfg);
    engine->owns_synthesis = 1;

    LNNConfig con_cfg = {0};
    con_cfg.input_size = (int)(edim * 2);
    con_cfg.hidden_size = 256;
    con_cfg.output_size = 1;
    con_cfg.num_layers = 1;
    engine->conflict_net = lnn_create(&con_cfg);
    engine->owns_conflict = 1;

    LNNConfig epi_cfg = {0};
    epi_cfg.input_size = (int)edim;
    epi_cfg.hidden_size = 256;
    epi_cfg.output_size = 3;
    epi_cfg.num_layers = 1;
    engine->epistemic_net = lnn_create(&epi_cfg);
    engine->owns_epistemic = 1;

    LNNConfig hyp_cfg = {0};
    hyp_cfg.input_size = (int)edim;
    hyp_cfg.hidden_size = 512;
    hyp_cfg.output_size = (int)edim;
    hyp_cfg.num_layers = 2;
    engine->hypothesis_net = lnn_create(&hyp_cfg);
    engine->owns_hypothesis = 1;

    engine->layer_embeddings = (float*)safe_calloc(DR_MAX_CHAIN_LEN * edim, sizeof(float));
    engine->knowledge_base = (float*)safe_calloc(1024 * edim, sizeof(float));
    engine->conflict_history = (float*)safe_calloc(DR_MAX_CONTRADICTIONS * 4, sizeof(float));
    
    return engine;
}

void dr_engine_destroy(DeepReflectionEngine* engine) {
    if (!engine) return;
    if (engine->reflection_net && engine->owns_reflection) lnn_free(engine->reflection_net);
    if (engine->synthesis_net && engine->owns_synthesis) lnn_free(engine->synthesis_net);
    if (engine->conflict_net && engine->owns_conflict) lnn_free(engine->conflict_net);
    if (engine->epistemic_net && engine->owns_epistemic) lnn_free(engine->epistemic_net);
    if (engine->hypothesis_net && engine->owns_hypothesis) lnn_free(engine->hypothesis_net);
    safe_free((void**)&engine->layer_embeddings);
    safe_free((void**)&engine->knowledge_base);
    safe_free((void**)&engine->conflict_history);
    safe_free((void**)&engine);
}

int dr_reflect(DeepReflectionEngine* engine,
                const char* topic,
                const float* context_data,
                size_t context_size,
                DRReflectionChain* chain_out) {
    if (!engine || !chain_out) return -1;
    if (!topic && !context_data) return -2;

    memset(chain_out, 0, sizeof(DRReflectionChain));

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;

    float base_embed[DR_EMBED_DIM] = {0};
    if (context_data && context_size > 0) {
        size_t copy_dim = (context_size < DR_EMBED_DIM) ? context_size : DR_EMBED_DIM;
        memcpy(base_embed, context_data, copy_dim * sizeof(float));
    } else if (topic) {
        /* N-004修复: 使用字符bigram哈希编码替代ASCII/255简化 */
        float topic_input[DR_EMBED_DIM];
        memset(topic_input, 0, sizeof(topic_input));
        size_t tlen = strlen(topic);
        /* 双字节哈希编码：相邻字符对→特征维度索引，均值叠加 */
        int hash_count = 0;
        for (size_t i = 0; i + 1 < tlen && i < DR_EMBED_DIM * 4; i++) {
            uint32_t h = ((uint32_t)(unsigned char)topic[i] << 8) | (uint32_t)(unsigned char)topic[i+1];
            h = h * 2654435761u;
            size_t idx = (size_t)(h % DR_EMBED_DIM);
            topic_input[idx] += 1.0f;
            hash_count++;
        }
        if (hash_count > 0) {
            float inv = 1.0f / sqrtf((float)hash_count + 1e-8f);
            for (size_t i = 0; i < DR_EMBED_DIM; i++)
                topic_input[i] *= inv;
        }
        if (engine->reflection_net) {
            lnn_forward(engine->reflection_net, topic_input, base_embed);
        } else {
            memcpy(base_embed, topic_input, DR_EMBED_DIM * sizeof(float));
        }
    }

    int max_layers = engine->config.max_iterations;
    if (max_layers > DR_MAX_CHAIN_LEN) max_layers = DR_MAX_CHAIN_LEN;

    memcpy(engine->layer_embeddings, base_embed, edim * sizeof(float));
    engine->layer_embed_count = 1;

    const float* prev_embeddings[DR_MAX_CHAIN_LEN];
    float prev_weights[DR_MAX_CHAIN_LEN];

    /* S-014: 跨层累积风险追踪 */
    float prev_risk = 0.0f;

    for (int layer = 0; layer < max_layers; layer++) {
        DRLayerResult* lr = &chain_out->layers[layer];
        lr->layer = (DRReflectionLayer)(layer % 7);

        float layer_input[DR_EMBED_DIM] = {0};
        for (size_t i = 0; i < engine->layer_embed_count; i++) {
            for (size_t j = 0; j < edim; j++) {
                layer_input[j] += engine->layer_embeddings[i * edim + j];
            }
        }
        float inv_count = 1.0f / (float)(engine->layer_embed_count > 0 ? engine->layer_embed_count : 1);
        for (size_t j = 0; j < edim; j++) {
            layer_input[j] *= inv_count;
        }

        /* M-009修复: 基于内容嵌入方差确定视角数量（高方差需多视角） */
        float embed_var = 0.0f, embed_mean = 0.0f;
        for (size_t j = 0; j < edim; j++) embed_mean += layer_input[j];
        embed_mean /= (float)edim;
        for (size_t j = 0; j < edim; j++) {
            float d = layer_input[j] - embed_mean;
            embed_var += d * d;
        }
        embed_var /= (float)edim;
        int num_perspectives = (embed_var > 0.1f) ? 6 : (embed_var > 0.01f ? 4 : 3);

        float multi_persp_in[DR_EMBED_DIM] = {0};
        for (size_t p = 0; p < (size_t)num_perspectives; p++) {
            float persp_emb[DR_EMBED_DIM];
            generate_perspective_embedding(engine->reflection_net, layer_input, (int)p, 0.3f + embed_var * 2.0f, edim, persp_emb);
            float persp_out[DR_EMBED_DIM] = {0};
            lnn_forward(engine->reflection_net, persp_emb, persp_out);
            for (size_t j = 0; j < edim; j++) {
                multi_persp_in[j] += persp_out[j];
            }
        }
        for (size_t j = 0; j < edim; j++) {
            multi_persp_in[j] /= (float)num_perspectives;
            lr->content_embedding[j] = multi_persp_in[j];
        }

        if (layer < 7) {
            lr->depth_score = (float)(layer + 1) / 7.0f;
        } else {
            lr->depth_score = 1.0f;
        }

        float sim_to_prev = 0.0f;
        if (layer > 0) {
            sim_to_prev = cosine_sim_dr(
                chain_out->layers[layer - 1].content_embedding,
                lr->content_embedding, edim);
        }
        lr->novelty_score = 1.0f - sim_to_prev;

        lr->coherence_score = 0.0f;
        for (size_t i = 0; i < engine->layer_embed_count; i++) {
            lr->coherence_score += cosine_sim_dr(
                engine->layer_embeddings + i * edim,
                lr->content_embedding, edim);
        }
        if (engine->layer_embed_count > 0) {
            lr->coherence_score /= (float)engine->layer_embed_count;
        }

        float epistemic_out[3] = {0};
        lnn_forward(engine->epistemic_net, lr->content_embedding, epistemic_out);
        float epistemic_conf = DR_SIGMOID(epistemic_out[0]);
        float epistemic_uncertainty = DR_SIGMOID(epistemic_out[1]);
        float epistemic_novelty = DR_SIGMOID(epistemic_out[2]);

        /* ===== S-014修复: 5维深度分析 ===== */
        /* 维度1: 信念网络一致性检测 */
        float belief_consistency = dr_belief_consistency_check(
            engine, lr->content_embedding, edim);
        /* 信念一致性与余弦一致性融合为增强一致性 */
        lr->coherence_score = lr->coherence_score * 0.4f + belief_consistency * 0.6f;

        /* 维度2: 假设检验 */
        float hyp_out_embed[DR_EMBED_DIM];
        float hypothesis_strength = dr_hypothesis_test(
            engine, lr->content_embedding, edim, hyp_out_embed);
        /* 假设强度影响新颖性：强假设降低伪新颖性 */
        lr->novelty_score = lr->novelty_score * 0.6f + (1.0f - hypothesis_strength) * 0.2f + lr->novelty_score * hypothesis_strength * 0.2f;
        lr->novelty_score = DR_CLAMP(lr->novelty_score, 0.0f, 1.0f);

        /* 维度3: 矛盾检测 */
        float* prev_layer_flat = NULL;
        if (layer > 0) {
            prev_layer_flat = (float*)safe_calloc((size_t)layer * edim, sizeof(float));
            if (prev_layer_flat) {
                for (int p = 0; p < layer; p++) {
                    memcpy(prev_layer_flat + (size_t)p * edim,
                           chain_out->layers[p].content_embedding,
                           edim * sizeof(float));
                }
            }
        }
        int cross_contradiction_count = 0;
        float contradiction_level = dr_cross_contradiction_check(
            engine, lr->content_embedding,
            prev_layer_flat, layer, edim, &cross_contradiction_count);
        if (prev_layer_flat) safe_free((void**)&prev_layer_flat);
        /* 矛盾标记: 超过阈值则标记，同时考虑跨层矛盾数量 */
        if (contradiction_level > 0.45f || cross_contradiction_count >= 3) {
            lr->contradiction_flag = 1.0f;
        } else if (lr->coherence_score < 0.25f) {
            lr->contradiction_flag = 1.0f;
        } else {
            lr->contradiction_flag = 0.0f;
        }

        /* 维度4: 风险评估 */
        float risk = dr_risk_assessment(engine, lr->content_embedding,
            lr->depth_score, lr->coherence_score,
            hypothesis_strength, contradiction_level,
            edim, layer, prev_risk);
        prev_risk = risk;
        /* 高风险触发更深层分析需求 */
        lr->needs_deeper = (lr->depth_score < engine->config.depth_threshold ||
                           lr->novelty_score > engine->config.novelty_threshold ||
                           risk > 0.4f);

        lr->reflection_text = (char*)safe_calloc(2048, 1);
        int perspective_idx = layer % 8;
        snprintf(lr->reflection_text, 2048,
            "[%s][%s] 深度:%.2f 新颖性:%.2f 一致性:%.2f "
            "信念一致性:%.2f 假设强度:%.2f 矛盾度:%.2f 风险:%.2f "
            "认知置信度:%.2f 不确定性:%.2f - %s",
            layer_names[lr->layer], perspective_names[perspective_idx],
            lr->depth_score, lr->novelty_score, lr->coherence_score,
            belief_consistency, hypothesis_strength, contradiction_level, risk,
            epistemic_conf, epistemic_uncertainty,
            topic ? topic : "上下文反思");
        lr->text_len = strlen(lr->reflection_text);

        prev_embeddings[layer] = lr->content_embedding;
        prev_weights[layer] = 1.0f - lr->novelty_score * 0.5f;

        if (engine->layer_embed_count < DR_MAX_CHAIN_LEN) {
            memcpy(engine->layer_embeddings + engine->layer_embed_count * edim,
                   lr->content_embedding, edim * sizeof(float));
            engine->layer_embed_count++;
        }

        chain_out->insight_scores[layer] = lr->novelty_score * 0.4f +
                                           lr->depth_score * 0.3f +
                                           epistemic_conf * 0.3f;
        chain_out->num_insights = (size_t)(layer + 1);

        chain_out->num_layers++;
    }

    float total_depth = 0.0f;
    for (size_t i = 0; i < chain_out->num_layers; i++) {
        total_depth += chain_out->layers[i].depth_score;
    }
    chain_out->overall_depth = total_depth / (float)(chain_out->num_layers > 0 ? chain_out->num_layers : 1);

    float transformative = 0.0f;
    float novelty_sum = 0.0f;
    for (size_t i = 0; i < chain_out->num_layers; i++) {
        novelty_sum += chain_out->layers[i].novelty_score;
        if (i > 0) {
            float change = DR_ABS(chain_out->layers[i].depth_score - chain_out->layers[i - 1].depth_score);
            transformative += change;
        }
    }
    chain_out->transformative_potential = DR_CLAMP(
        (novelty_sum / (float)chain_out->num_layers) * 0.6f +
        (transformative / (float)(chain_out->num_layers > 1 ? chain_out->num_layers - 1 : 1)) * 0.4f,
        0.0f, 1.0f);

    chain_out->self_consistency = 0.0f;
    int pair_count = 0;
    for (size_t i = 0; i < chain_out->num_layers && i < 12; i++) {
        for (size_t j = i + 1; j < chain_out->num_layers && j < 12; j++) {
            chain_out->self_consistency += cosine_sim_dr(
                chain_out->layers[i].content_embedding,
                chain_out->layers[j].content_embedding, edim);
            pair_count++;
        }
    }
    if (pair_count > 0) chain_out->self_consistency /= (float)pair_count;

    if (engine->config.enable_causal_analysis) {
        float root_cause_score = 0.0f;
        for (size_t i = 1; i < chain_out->num_layers; i++) {
            float cause_effect = cosine_sim_dr(
                chain_out->layers[i - 1].content_embedding,
                chain_out->layers[i].content_embedding, edim);
            root_cause_score += cause_effect * chain_out->layers[i].depth_score;
        }
        if (chain_out->num_layers > 1) {
            float avg_cause = root_cause_score / (float)(chain_out->num_layers - 1);
            chain_out->self_consistency = (chain_out->self_consistency + avg_cause) * 0.5f;
        }
    }

    chain_out->synthesis = (char*)safe_calloc(4096, 1);
    char insight_buffer[2048] = {0};
    size_t insight_offset = 0;
    for (size_t i = 0; i < chain_out->num_layers && i < 5; i++) {
        if (chain_out->layers[i].novelty_score > engine->config.novelty_threshold) {
            int n = snprintf(insight_buffer + insight_offset,
                sizeof(insight_buffer) - insight_offset,
                "  洞察%zu: %s (新颖性:%.2f 深度:%.2f)\n",
                i + 1, layer_names[chain_out->layers[i].layer],
                chain_out->layers[i].novelty_score,
                chain_out->layers[i].depth_score);
            if (n > 0) insight_offset += (size_t)n;
            if (insight_offset >= sizeof(insight_buffer) - 128) break;
        }
    }

    snprintf(chain_out->synthesis, 4096,
        "深度反思综合结果:\n"
        "  反思层数: %zu  |  整体深度: %.2f  |  变革潜力: %.2f  |  自我一致性: %.2f\n"
        "  ---\n"
        "  主题: %s\n"
        "  === 关键洞察 ===\n"
        "%s"
        "  === 总结 ===\n"
        "  多视角分析覆盖 %zu 个认知层面, "
        "最高深度 %.2f, "
        "认知稳定性 %s",
        chain_out->num_layers, chain_out->overall_depth,
        chain_out->transformative_potential,
        chain_out->self_consistency,
        topic ? topic : "（无主题）",
        insight_buffer,
        chain_out->num_layers,
        chain_out->overall_depth,
        (chain_out->self_consistency > 0.6f) ? "高" :
        (chain_out->self_consistency > 0.3f) ? "中" : "需修正");
    chain_out->synthesis_len = strlen(chain_out->synthesis);

    return 0;
}

int dr_reflect_multi_passage(DeepReflectionEngine* engine,
                              const char* topic,
                              const float** passages, size_t num_passages,
                              const size_t* passage_sizes,
                              DRReflectionChain* chain_out) {
    if (!engine || !chain_out) return -1;
    if (!topic && (!passages || num_passages == 0)) return -2;

    memset(chain_out, 0, sizeof(DRReflectionChain));

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;

    float fused_embed[DR_EMBED_DIM] = {0};
    if (passages && num_passages > 0) {
        const float** emb_ptr = (const float**)safe_calloc(num_passages, sizeof(float*));
        float* weights = (float*)safe_calloc(num_passages, sizeof(float));
        for (size_t p = 0; p < num_passages; p++) {
            emb_ptr[p] = passages[p];
            float mag = 0.0f;
            for (size_t j = 0; j < (passage_sizes ? passage_sizes[p] : edim) && j < edim; j++) {
                mag += passages[p][j] * passages[p][j];
            }
            weights[p] = sqrtf(mag) + 0.1f;
        }
        weighted_embedding_fusion(emb_ptr, weights, num_passages, edim, fused_embed);
        safe_free((void**)&emb_ptr);
        safe_free((void**)&weights);
    } else if (topic) {
        /* N-004修复: bigram哈希编码替代ASCII/255 */
        size_t tlen = strlen(topic);
        int hash_count = 0;
        for (size_t i = 0; i + 1 < tlen && i < edim * 4; i++) {
            uint32_t h = ((uint32_t)(unsigned char)topic[i] << 8) | (uint32_t)(unsigned char)topic[i+1];
            h = h * 2654435761u;
            size_t idx = (size_t)(h % edim);
            fused_embed[idx] += 1.0f;
            hash_count++;
        }
        if (hash_count > 0) {
            float inv = 1.0f / sqrtf((float)hash_count + 1e-8f);
            for (size_t i = 0; i < edim; i++) fused_embed[i] *= inv;
        }
    }

    int max_layers = engine->config.max_iterations;
    if (max_layers > DR_MAX_CHAIN_LEN) max_layers = DR_MAX_CHAIN_LEN;

    memcpy(engine->layer_embeddings, fused_embed, edim * sizeof(float));
    engine->layer_embed_count = 1;

    /* P1-035修复：使用时间混合种子替代固定42增加随机多样性 */
    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)&topic);
    if (topic) {
        size_t tlen = strlen(topic);
        for (size_t i = 0; i < tlen; i++)
            seed = seed * 31 + (unsigned char)topic[i];
    }
    seed ^= (unsigned int)((uint64_t)time(NULL) & 0xFFFFFFFFu);

    for (int layer = 0; layer < max_layers; layer++) {
        DRLayerResult* lr = &chain_out->layers[layer];
        lr->layer = (DRReflectionLayer)(layer % 7);

        float layer_input[DR_EMBED_DIM] = {0};
        for (size_t i = 0; i < engine->layer_embed_count; i++) {
            for (size_t j = 0; j < edim; j++) {
                layer_input[j] += engine->layer_embeddings[i * edim + j];
            }
        }
        float inv_count = 1.0f / (float)(engine->layer_embed_count > 0 ? engine->layer_embed_count : 1);
        for (size_t j = 0; j < edim; j++) {
            layer_input[j] *= inv_count;
        }

        float hyp_input[DR_EMBED_DIM];
        memcpy(hyp_input, layer_input, edim * sizeof(float));
        /* P0-004修复: 使用generate_perspective_embedding生成有意义的假设变体
           替代伪随机模运算扰动（原算法仅给输入加噪声，不代表真正的假设生成） */
        float hyp_perturb[DR_EMBED_DIM] = {0};
        int hyp_perspective = (int)(layer * 3 + seed * 7) % 8 + 2;
        generate_perspective_embedding(engine->reflection_net, layer_input,
            hyp_perspective, 0.15f, edim, hyp_perturb);
        for (size_t j = 0; j < edim; j++) {
            hyp_input[j] = layer_input[j] * 0.8f + hyp_perturb[j] * 0.2f;
        }

        float hyp_out[DR_EMBED_DIM] = {0};
        lnn_forward(engine->hypothesis_net, hyp_input, hyp_out);

        float persp_out[DR_EMBED_DIM] = {0};
        lnn_forward(engine->reflection_net, hyp_out, persp_out);

        for (size_t j = 0; j < edim; j++) {
            lr->content_embedding[j] = persp_out[j] * 0.7f + hyp_out[j] * 0.3f;
        }

        lr->depth_score = (layer < 7) ? (float)(layer + 1) / 7.0f : 1.0f;

        float sim_prev = (layer > 0) ? cosine_sim_dr(
            chain_out->layers[layer - 1].content_embedding,
            lr->content_embedding, edim) : 0.0f;
        lr->novelty_score = 1.0f - sim_prev;

        lr->coherence_score = 0.0f;
        for (size_t i = 0; i < engine->layer_embed_count; i++) {
            lr->coherence_score += cosine_sim_dr(
                engine->layer_embeddings + i * edim,
                lr->content_embedding, edim);
        }
        if (engine->layer_embed_count > 0)
            lr->coherence_score /= (float)engine->layer_embed_count;

        /* ZSFQQ-Q030: S-014 5维深度分析增强（多段落反射版本）
         * 与 dr_reflect() 对齐，为多段落反射添加完整的5维分析 */
        float s014_belief_consistency = 0.0f;
        float s014_hypothesis_strength = 0.0f;
        float s014_contradiction_level = 0.0f;
        float s014_risk_score = 0.0f;

        /* 维度1: 信念一致性（跨段落嵌入余弦一致性） */
        {
            int bc_count = 0;
            for (size_t i = 0; i < engine->layer_embed_count; i++) {
                float sim = cosine_sim_dr(engine->layer_embeddings + i * edim,
                    lr->content_embedding, edim);
                s014_belief_consistency += sim;
                bc_count++;
            }
            if (bc_count > 0) s014_belief_consistency /= (float)bc_count;
        }
        /* 维度2: 假设检验强度（基于内容嵌入的激活幅度） */
        {
            float hyp_mag = 0.0f;
            for (size_t i = 0; i < edim; i++) {
                float v = lr->content_embedding[i];
                hyp_mag += v * v;
            }
            s014_hypothesis_strength = sqrtf(hyp_mag) / sqrtf((float)edim);
        }
        /* 维度3: 矛盾检测（与先前层的最低一致性即为矛盾度） */
        {
            float min_consistency = 1.0f;
            for (size_t i = 0; i < engine->layer_embed_count; i++) {
                float sim = cosine_sim_dr(engine->layer_embeddings + i * edim,
                    lr->content_embedding, edim);
                if (sim < min_consistency) min_consistency = sim;
            }
            s014_contradiction_level = 1.0f - min_consistency;
        }
        /* 维度4: 风险评估（基于新颖性和矛盾度） */
        s014_risk_score = lr->novelty_score * 0.4f + s014_contradiction_level * 0.4f +
                          (1.0f - s014_belief_consistency) * 0.2f;

        /* 将S-014分析结果融合到coherence_score和contradiction_flag */
        lr->coherence_score = s014_belief_consistency * 0.6f + lr->coherence_score * 0.4f;
        lr->contradiction_flag = (s014_contradiction_level > 0.5f) ? 1.0f : 0.0f;
        lr->depth_score = lr->depth_score * 0.7f + s014_hypothesis_strength * 0.3f;

        lr->reflection_text = (char*)safe_calloc(2048, 1);
        snprintf(lr->reflection_text, 2048,
            "[%s][多段落反射S-014] 深度:%.2f 新颖:%.2f 一致:%.2f "
            "信念:%.2f 假设:%.2f 矛盾:%.2f 风险:%.2f - %s",
            layer_names[lr->layer], lr->depth_score,
            lr->novelty_score, lr->coherence_score,
            s014_belief_consistency, s014_hypothesis_strength,
            s014_contradiction_level, s014_risk_score,
            topic ? topic : "多段落反思");
        lr->text_len = strlen(lr->reflection_text);

        lr->contradiction_flag = (lr->coherence_score < 0.25f) ? 1.0f : 0.0f;
        lr->needs_deeper = (lr->depth_score < engine->config.depth_threshold);

        if (engine->layer_embed_count < DR_MAX_CHAIN_LEN) {
            memcpy(engine->layer_embeddings + engine->layer_embed_count * edim,
                   lr->content_embedding, edim * sizeof(float));
            engine->layer_embed_count++;
        }

        chain_out->insight_scores[layer] = lr->novelty_score * 0.5f + lr->depth_score * 0.5f;
        chain_out->num_insights = (size_t)(layer + 1);
        chain_out->num_layers++;
    }

    float total_depth = 0.0f;
    for (size_t i = 0; i < chain_out->num_layers; i++)
        total_depth += chain_out->layers[i].depth_score;
    chain_out->overall_depth = total_depth / (float)(chain_out->num_layers > 0 ? chain_out->num_layers : 1);

    chain_out->self_consistency = 0.0f;
    int pc = 0;
    for (size_t i = 0; i < chain_out->num_layers && i < 10; i++) {
        for (size_t j = i + 1; j < chain_out->num_layers && j < 10; j++) {
            chain_out->self_consistency += cosine_sim_dr(
                chain_out->layers[i].content_embedding,
                chain_out->layers[j].content_embedding, edim);
            pc++;
        }
    }
    if (pc > 0) chain_out->self_consistency /= (float)pc;

    float tv = 0.0f;
    for (size_t i = 1; i < chain_out->num_layers; i++) {
        tv += DR_ABS(chain_out->layers[i].depth_score - chain_out->layers[i - 1].depth_score);
    }
    chain_out->transformative_potential = DR_CLAMP(tv / (float)(chain_out->num_layers > 1 ? chain_out->num_layers - 1 : 1), 0.0f, 1.0f);

    chain_out->synthesis = (char*)safe_calloc(4096, 1);
    snprintf(chain_out->synthesis, 4096,
        "多段落深度反思:\n  段落数: %zu  |  反思层数: %zu  |  深度: %.2f  |  变革潜力: %.2f\n  主题: %s",
        num_passages, chain_out->num_layers, chain_out->overall_depth,
        chain_out->transformative_potential, topic ? topic : "（无主题）");
    chain_out->synthesis_len = strlen(chain_out->synthesis);

    return 0;
}

int dr_epistemic_assessment(DeepReflectionEngine* engine,
                             DRReflectionChain* chain,
                             float* knowledge_certainty_out,
                             float* knowledge_coverage_out,
                             float* knowledge_boundary_out) {
    if (!engine || !chain) return -1;

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;

    float certainty = 0.0f;
    float coverage = 0.0f;
    float boundary = 0.0f;

    for (size_t i = 0; i < chain->num_layers; i++) {
        float epi_in[DR_EMBED_DIM];
        memcpy(epi_in, chain->layers[i].content_embedding, edim * sizeof(float));
        float epi_out[3] = {0};
        lnn_forward(engine->epistemic_net, epi_in, epi_out);

        float c = DR_SIGMOID(epi_out[0]);
        float cov = DR_SIGMOID(epi_out[1]);
        float b = 1.0f - DR_SIGMOID(epi_out[2]);

        certainty += c * chain->layers[i].depth_score;
        coverage += cov;
        boundary += b;
    }

    float inv = 1.0f / (float)(chain->num_layers > 0 ? chain->num_layers : 1);
    if (knowledge_certainty_out) *knowledge_certainty_out = DR_CLAMP(certainty * inv, 0.0f, 1.0f);
    if (knowledge_coverage_out) *knowledge_coverage_out = DR_CLAMP(coverage * inv, 0.0f, 1.0f);
    if (knowledge_boundary_out) *knowledge_boundary_out = DR_CLAMP(boundary * inv, 0.0f, 1.0f);

    return 0;
}

int dr_detect_knowledge_conflicts(DeepReflectionEngine* engine,
                                    DRReflectionChain* chain,
                                    DRConflictInfo* conflicts_out,
                                    size_t* num_conflicts) {
    if (!engine || !chain || !conflicts_out || !num_conflicts) return -1;
    if (!engine->config.enable_contradiction_detection) {
        *num_conflicts = 0;
        return 0;
    }

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    size_t count = 0;
    size_t max_conflicts = (*num_conflicts < DR_MAX_CONTRADICTIONS) ? *num_conflicts : DR_MAX_CONTRADICTIONS;

    for (size_t i = 0; i < chain->num_layers && count < max_conflicts; i++) {
        float prior_i = chain->layers[i].depth_score * 0.5f + chain->layers[i].coherence_score * 0.5f;

        for (size_t j = i + 1; j < chain->num_layers && count < max_conflicts; j++) {
            float prior_j = chain->layers[j].depth_score * 0.5f + chain->layers[j].coherence_score * 0.5f;

            float bcs = bayesian_conflict_score(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding,
                prior_i, prior_j, edim);

            float con_input[DR_EMBED_DIM * 2];
            memcpy(con_input, chain->layers[i].content_embedding, edim * sizeof(float));
            memcpy(con_input + edim, chain->layers[j].content_embedding, edim * sizeof(float));
            float con_out[1] = {0};
            lnn_forward(engine->conflict_net, con_input, con_out);
            float nn_conflict = DR_SIGMOID(con_out[0]);

            float combined_conflict = bcs * 0.6f + nn_conflict * 0.4f;

            if (combined_conflict > 0.35f) {
                DRConflictInfo* ci = &conflicts_out[count];
                ci->layer_a = i;
                ci->layer_b = j;
                ci->conflict_score = combined_conflict;
                ci->bayesian_score = bcs;
                ci->nn_score = nn_conflict;
                ci->similarity = cosine_sim_dr(
                    chain->layers[i].content_embedding,
                    chain->layers[j].content_embedding, edim);

                snprintf(ci->description, sizeof(ci->description),
                    "知识冲突: [层%zu:%s] vs [层%zu:%s] "
                    "(冲突:%.2f 贝叶斯:%.2f 网络:%.2f 相似度:%.2f)",
                    i, layer_names[chain->layers[i].layer],
                    j, layer_names[chain->layers[j].layer],
                    combined_conflict, bcs, nn_conflict, ci->similarity);

                ci->resolution = DR_RESOLVE_UNRESOLVED;

                if (engine->conflict_count < DR_MAX_CONTRADICTIONS) {
                    engine->conflict_history[engine->conflict_count * 4 + 0] = (float)i;
                    engine->conflict_history[engine->conflict_count * 4 + 1] = (float)j;
                    engine->conflict_history[engine->conflict_count * 4 + 2] = combined_conflict;
                    engine->conflict_history[engine->conflict_count * 4 + 3] = (float)ci->resolution;
                    engine->conflict_count++;
                }

                count++;
            }
        }
    }

    *num_conflicts = count;
    return 0;
}

int dr_resolve_conflicts(DeepReflectionEngine* engine,
                          DRConflictInfo* conflicts,
                          size_t num_conflicts,
                          DRReflectionChain* chain_out) {
    if (!engine || !conflicts || !chain_out) return -1;

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    int resolved_count = 0;

    for (size_t c = 0; c < num_conflicts; c++) {
        DRConflictInfo* ci = &conflicts[c];
        if (ci->resolution != DR_RESOLVE_UNRESOLVED) continue;

        size_t idx_a = ci->layer_a;
        size_t idx_b = ci->layer_b;

        float belief_a[DR_EMBED_DIM], belief_b[DR_EMBED_DIM];
        memcpy(belief_a, chain_out->layers[idx_a].content_embedding, edim * sizeof(float));
        memcpy(belief_b, chain_out->layers[idx_b].content_embedding, edim * sizeof(float));

        float confidence_a = chain_out->layers[idx_a].depth_score;
        float confidence_b = chain_out->layers[idx_b].depth_score;
        float total_conf = confidence_a + confidence_b + 1e-10f;

        float merged[DR_EMBED_DIM];
        for (size_t j = 0; j < edim; j++) {
            merged[j] = (belief_a[j] * confidence_a + belief_b[j] * confidence_b) / total_conf;
        }

        memcpy(chain_out->layers[idx_a].content_embedding, merged, edim * sizeof(float));
        chain_out->layers[idx_a].contradiction_flag = 0.0f;
        chain_out->layers[idx_a].coherence_score =
            (chain_out->layers[idx_a].coherence_score + chain_out->layers[idx_b].coherence_score) * 0.5f;
        chain_out->layers[idx_b].contradiction_flag = 0.0f;

        ci->resolution = DR_RESOLVE_MERGED;

        if (engine->conflict_count > 0 && c < engine->conflict_count) {
            engine->conflict_history[c * 4 + 3] = (float)DR_RESOLVE_MERGED;
        }

        resolved_count++;
    }

    float merged_depth = 0.0f;
    for (size_t i = 0; i < chain_out->num_layers; i++) {
        merged_depth += chain_out->layers[i].depth_score;
    }
    chain_out->overall_depth = merged_depth / (float)(chain_out->num_layers > 0 ? chain_out->num_layers : 1);

    return resolved_count;
}

int dr_generate_hypotheses(DeepReflectionEngine* engine,
                            DRReflectionChain* chain,
                            float* hypotheses_out,
                            size_t* num_hypotheses,
                            size_t max_hypotheses) {
    if (!engine || !chain || !hypotheses_out || !num_hypotheses) return -1;

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    size_t count = 0;
    size_t max_h = (max_hypotheses < 16) ? max_hypotheses : 16;

    /* P1-035修复：使用时间混合种子替代固定42 */
    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)chain ^ count);
    for (size_t i = 0; i < chain->num_layers && count < max_h; i++) {
        if (chain->layers[i].novelty_score < 0.3f) continue;

        float base[DR_EMBED_DIM];
        memcpy(base, chain->layers[i].content_embedding, edim * sizeof(float));

        for (size_t variant = 0; variant < 2 && count < max_h; variant++) {
            float hyp_in[DR_EMBED_DIM];
            for (size_t j = 0; j < edim; j++) {
                /* Box-Muller变换生成标准正态噪声：从两个均匀分布伪随机数生成高斯分布 */
                unsigned int s_a = seed + (unsigned int)(i * 37 + variant * 73 + j * 101);
                unsigned int s_b = seed + (unsigned int)(i * 53 + variant * 97 + j * 151);
                float u1 = (float)(s_a % 1000000) / 1000000.0f + 1e-7f;
                float u2 = (float)(s_b % 1000000) / 1000000.0f;
                float noise = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2) * 0.15f;
                hyp_in[j] = base[j] + noise;
            }

            float hyp_out[DR_EMBED_DIM] = {0};
            lnn_forward(engine->hypothesis_net, hyp_in, hyp_out);

            float* out = hypotheses_out + count * edim;
            memcpy(out, hyp_out, edim * sizeof(float));
            count++;
        }
    }

    *num_hypotheses = count;
    return 0;
}

int dr_root_cause_analysis(DeepReflectionEngine* engine,
                            DRReflectionChain* chain,
                            size_t* root_layer_idx,
                            float* root_confidence) {
    if (!engine || !chain || !root_layer_idx || !root_confidence) return -1;
    if (chain->num_layers == 0) return -2;

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    float best_cause_score = -1.0f;
    size_t best_idx = 0;

    for (size_t i = 0; i < chain->num_layers; i++) {
        float downstream_influence = 0.0f;
        int influence_count = 0;

        for (size_t j = i + 1; j < chain->num_layers; j++) {
            float sim = cosine_sim_dr(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding, edim);
            float influence = sim * chain->layers[j].depth_score * (1.0f - chain->layers[j].novelty_score);
            downstream_influence += influence;
            influence_count++;
        }

        for (size_t j = 0; j < i; j++) {
            float sim = cosine_sim_dr(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding, edim);
            float influence = sim * chain->layers[i].depth_score * (1.0f - chain->layers[i].novelty_score);
            downstream_influence += influence * 0.3f;
            influence_count++;
        }

        float avg_influence = (influence_count > 0) ?
            downstream_influence / (float)influence_count : 0.0f;

        float cause_score = avg_influence * chain->layers[i].depth_score *
                           (1.0f + chain->layers[i].coherence_score) * 0.5f;

        if (cause_score > best_cause_score) {
            best_cause_score = cause_score;
            best_idx = i;
        }
    }

    *root_layer_idx = best_idx;
    *root_confidence = DR_CLAMP(best_cause_score, 0.0f, 1.0f);
    return 0;
}

int dr_get_insights(DeepReflectionEngine* engine,
                     DRReflectionChain* chain,
                     char** insights_out,
                     size_t* num_insights) {
    if (!engine || !chain || !insights_out || !num_insights) return -1;

    size_t count = 0;
    for (size_t i = 0; i < chain->num_layers && count < DR_MAX_INSIGHTS; i++) {
        if (chain->insight_scores[i] > engine->config.novelty_threshold) {
            insights_out[count] = chain->layers[i].reflection_text;
            count++;
        }
    }

    if (count == 0 && chain->num_layers > 0) {
        insights_out[0] = chain->layers[chain->num_layers - 1].reflection_text;
        count = 1;
    }

    *num_insights = count;
    return 0;
}

int dr_detect_contradictions(DeepReflectionEngine* engine,
                              DRReflectionChain* chain,
                              char** contradictions_out,
                              size_t* num_contradictions) {
    if (!engine || !chain || !contradictions_out || !num_contradictions) return -1;
    if (!engine->config.enable_contradiction_detection) {
        *num_contradictions = 0;
        return 0;
    }

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;
    size_t count = 0;
    size_t max_out = (*num_contradictions < DR_MAX_CONTRADICTIONS) ? *num_contradictions : DR_MAX_CONTRADICTIONS;

    for (size_t i = 0; i < chain->num_layers && count < max_out; i++) {
        for (size_t j = i + 1; j < chain->num_layers && count < max_out; j++) {
            float sim = cosine_sim_dr(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding, edim);

            float bayesian = bayesian_conflict_score(
                chain->layers[i].content_embedding,
                chain->layers[j].content_embedding,
                chain->layers[i].depth_score,
                chain->layers[j].depth_score, edim);

            float combined_conflict = bayesian * 0.7f + (1.0f - sim) * 0.3f;

            if (combined_conflict > 0.4f && chain->layers[i].depth_score > 0.3f &&
                chain->layers[j].depth_score > 0.3f) {
                char* buf = (char*)safe_calloc(512, 1);
                snprintf(buf, 512, "矛盾: 层%zu[%s:%.2f] vs 层%zu[%s:%.2f] "
                         "(贝叶斯冲突:%.2f 相似度:%.2f 综合:%.2f)",
                         i, layer_names[chain->layers[i].layer],
                         chain->layers[i].depth_score,
                         j, layer_names[chain->layers[j].layer],
                         chain->layers[j].depth_score,
                         bayesian, sim, combined_conflict);
                contradictions_out[count] = buf;
                count++;
            }
        }
    }

    *num_contradictions = count;
    return 0;
}

int dr_generate_synthesis(DeepReflectionEngine* engine,
                           DRReflectionChain* chain,
                           char* synthesis_out,
                           size_t max_len) {
    if (!engine || !chain || !synthesis_out) return -1;

    if (chain->synthesis) {
        strncpy(synthesis_out, chain->synthesis, max_len - 1);
        synthesis_out[max_len - 1] = '\0';
        return 0;
    }

    size_t edim = (engine->config.embedding_dim > 0) ? engine->config.embedding_dim : DR_EMBED_DIM;

    const float* top_embs[3];
    float top_weights[3];
    size_t num_top = (chain->num_layers < 3) ? chain->num_layers : 3;
    for (size_t i = 0; i < num_top; i++) {
        top_embs[i] = chain->layers[i].content_embedding;
        top_weights[i] = chain->layers[i].depth_score;
    }

    float syn_input[DR_EMBED_DIM] = {0};
    weighted_embedding_fusion(top_embs, top_weights, num_top, edim, syn_input);

    float syn_output[DR_EMBED_DIM] = {0};
    lnn_forward(engine->synthesis_net, syn_input, syn_output);

    snprintf(synthesis_out, max_len,
        "深度反思综合 (%zu层, 深度%.2f, 变革潜力%.2f, 一致性%.2f)",
        chain->num_layers, chain->overall_depth,
        chain->transformative_potential, chain->self_consistency);

    return 0;
}

int dr_chain_to_plan(DeepReflectionEngine* engine,
                      DRReflectionChain* chain,
                      float* plan_actions,
                      size_t* num_actions) {
    if (!engine || !chain || !plan_actions || !num_actions) return -1;

    size_t actions = (chain->num_layers < 16) ? chain->num_layers : 16;

    size_t root_idx = 0;
    float root_conf = 0.0f;
    dr_root_cause_analysis(engine, chain, &root_idx, &root_conf);

    for (size_t i = 0; i < actions; i++) {
        size_t layer_idx = i * chain->num_layers / actions;
        if (layer_idx >= chain->num_layers) layer_idx = chain->num_layers - 1;

        DRLayerResult* lr = &chain->layers[layer_idx];
        float* action = plan_actions + i * 5;
        action[0] = (float)lr->layer;
        action[1] = lr->depth_score;
        action[2] = lr->novelty_score;
        action[3] = lr->coherence_score;
        action[4] = (layer_idx == root_idx) ? 1.0f : 0.0f;
    }

    *num_actions = actions;
    return 0;
}

int dr_integrate_with_metacognition(DeepReflectionEngine* engine,
                                     MetacognitionSystem* meta_system,
                                     DRReflectionChain* chain) {
    if (!engine || !meta_system || !chain) return -1;

    float ref_quality = chain->overall_depth * 0.4f +
                        chain->self_consistency * 0.3f +
                        chain->transformative_potential * 0.3f;

    size_t root_idx = 0;
    float root_conf = 0.0f;
    dr_root_cause_analysis(engine, chain, &root_idx, &root_conf);

    float certainty = 0.0f, coverage = 0.0f, boundary = 0.0f;
    dr_epistemic_assessment(engine, chain, &certainty, &coverage, &boundary);

    char assessment[2048];
    snprintf(assessment, sizeof(assessment),
        "深度反思整合评估:\n"
        "  反思质量: %.2f  |  认知确定性: %.2f  |  知识覆盖率: %.2f  |  知识边界: %.2f\n"
        "  根因层: %zu (置信度:%.2f)\n"
        "  整体深度: %.2f  |  一致性: %.2f  |  变革潜力: %.2f",
        ref_quality, certainty, coverage, boundary,
        root_idx, root_conf,
        chain->overall_depth, chain->self_consistency,
        chain->transformative_potential);

    metacognition_neutral_self_assessment(meta_system, 0, assessment, strlen(assessment));

    return 0;
}

void dr_chain_free(DRReflectionChain* chain) {
    if (!chain) return;
    for (size_t i = 0; i < chain->num_layers; i++) {
        safe_free((void**)&chain->layers[i].reflection_text);
    }
    safe_free((void**)&chain->synthesis);
    memset(chain, 0, sizeof(DRReflectionChain));
}
