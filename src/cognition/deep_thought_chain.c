/**
 * @file deep_thought_chain.c
 * @brief 深度思考链系统完整实现
 *
 * 多分支推理链构建与评估：束搜索、自一致性重排序、
 * 分支扩展、剪枝回溯、思维合并与最优路径选择。
 *
 * ============================================================
 * 【模块职责 - ZSFWS-028 认知三模块边界】
 * ============================================================
 * 本模块（深度思考链）的核心职责：构建式推理链生成与优化系统
 *
 * 独特功能：
 *   - 多步骤推理链构建（dtc_reason_chain）：观察→分析→假设→推理→评估→综合→结论→行动
 *   - 推理分支并行扩展（dtc_expand_branch）
 *   - 束搜索探索最优推理路径（dtc_beam_search）
 *   - 自一致性重排序验证推理可靠性（dtc_self_consistency_rerank）
 *   - 多路径思维合并以消除分歧（dtc_merge_thoughts）
 *   - 推理链质量评估（dtc_evaluate_chain）
 *   - 低质量分支剪枝（dtc_prune_chain）
 *   - 推理失败时回溯到分支点重试（dtc_backtrack）
 *   - 选择全局最优推理路径（dtc_get_best_path）
 *   - 推理链导出为文本描述或行动计划（dtc_chain_to_text / dtc_chain_to_plan）
 *
 * 与 deep_correction.c 的区别：
 *   - deep_thought_chain 面向未来问题构建新推理路径，是"创造式"的
 *   - deep_correction 面向已发生的错误构建修复路径，是"补救式"的
 *   - thought_chain有回溯（backtrack到分支点），correction有回滚（rollback修正操作）
 *
 * 与 deep_reflection.c 的区别：
 *   - deep_thought_chain 从零构建新推理，关注"如何解决新问题"
 *   - deep_reflection 审视已有信念，关注"我现有的认知是否正确"
 *   - thought_chain输出是"推理路径/计划"，reflection输出是"认知审查报告"
 *
 * ⚠️  功能重叠提示：
 *   - dtc_chain_to_plan（思考链→计划）与 dr_chain_to_plan（反思→计划）：
 *     建议：thought_chain的plan通常更具体、分步骤，reflection的plan更宏观
 *     两个plan可通过一个统一的"计划融合层"合并为执行计划
 *   - dtc_merge_thoughts（思维合并）与 dr_generate_synthesis（反思综合）：
 *     建议：两者本质都是多视角融合，可共享融合算法基础层，但输入类型不同
 * ============================================================
 */
#include "selflnn/cognition/deep_thought_chain.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/laplace.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/secure_random.h"   /* N-003: 安全随机数替代LCG */
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>

#define DTC_SIGMOID(x) (1.0f / (1.0f + expf(-(x))))
#define DTC_TANH(x) (tanhf(x))
#define DTC_MAX(a,b) (((a)>(b))?(a):(b))
#define DTC_MIN(a,b) (((a)<(b))?(a):(b))
#define DTC_CLAMP(v,lo,hi) DTC_MIN(DTC_MAX((v),(lo)),(hi))
#define DTC_ABS(x) (((x)>=0)?(x):(-(x)))

struct DTCSystem {
    DTCConfig config;
    LNN* thought_net;
    LNN* evaluation_net;
    LNN* merge_net;
    int owns_thought;
    int owns_evaluation;
    int owns_merge;
    int initialized;
    
    /* 拉普拉斯分析器（频域思维链稳定性分析与增强） */
    LaplaceAnalyzer* laplace_analyzer;      /**< 拉普拉斯分析器 */
    float* laplace_spectrum_buffer;         /**< 频谱分析缓冲区 */
};

static const char* step_names[] = {
    "观察",
    "分析",
    "假设",
    "推理",
    "评估",
    "综合",
    "结论",
    "行动"
};

/* F-023修复：使用统一余弦相似度替代本地实现 */
#define cosine_sim_dtc(a, b, dim) math_cosine_similarity((a), (b), (dim))

static float euclidean_dist_dtc(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

typedef struct {
    float score;
    size_t start_idx;
    size_t end_idx;
    float coherence;
    float avg_confidence;
} BeamPath;

static int compare_beam_paths(const void* a, const void* b) {
    const BeamPath* pa = (const BeamPath*)a;
    const BeamPath* pb = (const BeamPath*)b;
    if (pb->score > pa->score) return 1;
    if (pb->score < pa->score) return -1;
    return 0;
}

DTCSystem* dtc_system_create(DTCConfig config) {
    DTCSystem* system = (DTCSystem*)safe_calloc(1, sizeof(DTCSystem));
    if (!system) return NULL;

    system->config = config;
    system->initialized = 1;

    system->owns_thought = system->owns_evaluation = system->owns_merge = 1;

    LNNConfig thought_cfg = {0};
    thought_cfg.input_size = DTC_EMBED_DIM;
    thought_cfg.hidden_size = 512;
    thought_cfg.output_size = DTC_EMBED_DIM;
    thought_cfg.num_layers = 2;
    system->thought_net = lnn_create(&thought_cfg);
    system->owns_thought = 1;

    LNNConfig eval_cfg = {0};
    eval_cfg.input_size = DTC_EMBED_DIM * 2;
    eval_cfg.hidden_size = 256;
    eval_cfg.output_size = 3;
    eval_cfg.num_layers = 2;
    system->evaluation_net = lnn_create(&eval_cfg);
    system->owns_evaluation = 1;

    LNNConfig merge_cfg = {0};
    merge_cfg.input_size = DTC_EMBED_DIM * 2;
    merge_cfg.hidden_size = 512;
    merge_cfg.output_size = DTC_EMBED_DIM;
    merge_cfg.num_layers = 1;
    system->merge_net = lnn_create(&merge_cfg);
    system->owns_merge = 1;
    
    /* 初始化拉普拉斯分析器（频域思维链稳定性分析） */
    {
        LaplaceConfig lap_cfg;
        memset(&lap_cfg, 0, sizeof(lap_cfg));
        lap_cfg.num_samples = 256;
        lap_cfg.sample_rate = 1000.0f;
        lap_cfg.max_frequency = 100.0f;
        lap_cfg.min_frequency = 0.1f;
        lap_cfg.enable_stability = 1;
        lap_cfg.enable_frequency = 1;
        lap_cfg.enable_optimization = 1;
        lap_cfg.cutoff_frequency = 50.0f;
        lap_cfg.filter_order = 2;
        lap_cfg.alpha = 0.95f;
        lap_cfg.beta = 0.05f;
        system->laplace_analyzer = laplace_analyzer_create(&lap_cfg);
        system->laplace_spectrum_buffer = (float*)safe_malloc(256 * sizeof(float));
        if (system->laplace_spectrum_buffer) {
            memset(system->laplace_spectrum_buffer, 0, 256 * sizeof(float));
        }
    }
    
    return system;
}

void dtc_system_destroy(DTCSystem* system) {
    if (!system) return;
    if (system->thought_net && system->owns_thought) lnn_free(system->thought_net);
    if (system->evaluation_net && system->owns_evaluation) lnn_free(system->evaluation_net);
    if (system->merge_net && system->owns_merge) lnn_free(system->merge_net);
    if (system->laplace_analyzer) {
        laplace_analyzer_free(system->laplace_analyzer);
        system->laplace_analyzer = NULL;
    }
    safe_free((void**)&system->laplace_spectrum_buffer);
    safe_free((void**)&system);
}

int dtc_reason_chain(DTCSystem* system,
                      const float* input_data,
                      size_t input_dim,
                      const char* query,
                      DTCChainResult* result_out) {
    if (!system || !result_out) return -1;
    if (!input_data && !query) return -2;

    memset(result_out, 0, sizeof(DTCChainResult));

    float base_embed[DTC_EMBED_DIM] = {0};
    if (input_data && input_dim > 0) {
        size_t copy_dim = input_dim < DTC_EMBED_DIM ? input_dim : DTC_EMBED_DIM;
        memcpy(base_embed, input_data, copy_dim * sizeof(float));
    } else if (query) {
        /* N-004修复: 使用字符bigram哈希编码替代ASCII/255简化 */
        float query_input[DTC_EMBED_DIM];
        memset(query_input, 0, sizeof(query_input));
        size_t qlen = strlen(query);
        int hash_count = 0;
        for (size_t i = 0; i + 1 < qlen && i < DTC_EMBED_DIM * 4; i++) {
            uint32_t h = ((uint32_t)(unsigned char)query[i] << 8) | (uint32_t)(unsigned char)query[i+1];
            h = h * 2654435761u;
            size_t idx = (size_t)(h % DTC_EMBED_DIM);
            query_input[idx] += 1.0f;
            hash_count++;
        }
        if (hash_count > 0) {
            float inv = 1.0f / sqrtf((float)hash_count + 1e-8f);
            for (size_t i = 0; i < DTC_EMBED_DIM; i++)
                query_input[i] *= inv;
        }
        if (system->thought_net) {
            lnn_forward(system->thought_net, query_input, base_embed);
        } else {
            memcpy(base_embed, query_input, DTC_EMBED_DIM * sizeof(float));
        }
    }

    float norm = 0.0f;
    for (size_t i = 0; i < DTC_EMBED_DIM; i++) norm += base_embed[i] * base_embed[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (size_t i = 0; i < DTC_EMBED_DIM; i++) base_embed[i] /= norm;
    }

    size_t max_steps = system->config.max_depth;
    if (max_steps > DTC_MAX_CHAIN_LEN) max_steps = DTC_MAX_CHAIN_LEN;

    float current_embed[DTC_EMBED_DIM];
    memcpy(current_embed, base_embed, DTC_EMBED_DIM * sizeof(float));
    unsigned int seed = (unsigned int)(query ? strlen(query) : 42);
    /* S-014修复: 使用当前嵌入能量+时间混合种子生成认知噪声
     * 替代确定性公式((seed+step*53+j*7)%1000)/1000 */
    seed ^= (unsigned int)((uint64_t)time(NULL) & 0xFFFFFFFFu);

    for (size_t step = 0; step < max_steps; step++) {
        DTCThoughtNode* node = &result_out->nodes[step];
        node->step_type = (DTCStepType)(step % 8);
        node->parent_index = (step > 0) ? step - 1 : 0;

        /* 基于嵌入统计特性的认知噪声模型 */
        float embed_var = 0.0f;
        for (size_t j = 0; j < DTC_EMBED_DIM; j++) {
            float d = current_embed[j];
            embed_var += d * d;
        }
        embed_var /= (float)DTC_EMBED_DIM;
        float noise_scale = system->config.temperature * sqrtf(embed_var + 1e-6f);

        float thought_input[DTC_EMBED_DIM];
        for (size_t j = 0; j < DTC_EMBED_DIM; j++) {
            /* N-003修复: 使用密码学安全随机数替代LCG弱随机
             * Box-Muller变换: sqrt(-2ln(U1))*cos(2π*U2) 生成标准正态分布噪声 */
            float u1 = secure_random_float();
            float u2 = secure_random_float();
            float noise = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.2831853f * u2);
            thought_input[j] = current_embed[j] + noise * noise_scale * 0.1f;
        }

        float thought_out[DTC_EMBED_DIM] = {0};
        lnn_forward(system->thought_net, thought_input, thought_out);
        memcpy(node->thought_embedding, thought_out, DTC_EMBED_DIM * sizeof(float));

        float eval_input[DTC_EMBED_DIM * 2];
        memcpy(eval_input, current_embed, DTC_EMBED_DIM * sizeof(float));
        memcpy(eval_input + DTC_EMBED_DIM, thought_out, DTC_EMBED_DIM * sizeof(float));

        float eval_out[3] = {0};
        lnn_forward(system->evaluation_net, eval_input, eval_out);

        node->confidence = DTC_SIGMOID(eval_out[0]);
        node->uncertainty = DTC_SIGMOID(eval_out[1]);
        float raw_branching = DTC_SIGMOID(eval_out[2]);
        node->branching_factor = DTC_CLAMP(raw_branching * 2.0f, 0.0f, 1.0f);

        node->thought_text = (char*)safe_calloc(1024, 1);
        /* M-011修复: 思维文本融入LNN输出统计分析而非纯格式字符串 */
        float embed_energy = 0.0f;
        int active_dims = 0;
        for (size_t j = 0; j < DTC_EMBED_DIM && j < 32; j++) {
            embed_energy += thought_out[j] * thought_out[j];
            if (fabsf(thought_out[j]) > 0.2f) active_dims++;
        }
        snprintf(node->thought_text, 1024,
            "[%s] 置信:%.2f 能量:%.3f 激活:%d/%d 分支:%.2f",
            step_names[node->step_type],
            node->confidence, sqrtf(embed_energy), active_dims, 32, node->branching_factor);
        node->text_len = strlen(node->thought_text);

        node->has_branches = (node->branching_factor > system->config.branching_threshold &&
                             step > 0) ? 1 : 0;

        float momentum = 0.3f + node->confidence * 0.2f;
        for (size_t j = 0; j < DTC_EMBED_DIM; j++) {
            current_embed[j] = current_embed[j] * (1.0f - momentum) + thought_out[j] * momentum;
        }

        result_out->num_nodes = step + 1;

        if (node->confidence > system->config.confidence_threshold &&
            node->step_type == DTC_STEP_CONCLUDE) {
            break;
        }
    }

    result_out->chain_coherence = 0.0f;
    int pair_count = 0;
    for (size_t i = 1; i < result_out->num_nodes; i++) {
        float sim = cosine_sim_dtc(
            result_out->nodes[i].thought_embedding,
            result_out->nodes[i - 1].thought_embedding, DTC_EMBED_DIM);
        float conf_weight = result_out->nodes[i].confidence;
        result_out->chain_coherence += sim * conf_weight;
        pair_count++;
    }
    if (pair_count > 0) result_out->chain_coherence /= (float)pair_count;

    result_out->reasoning_depth = (float)result_out->num_nodes / (float)DTC_MAX_CHAIN_LEN;

    float avg_conf = 0.0f;
    for (size_t i = 0; i < result_out->num_nodes; i++) {
        avg_conf += result_out->nodes[i].confidence;
    }
    result_out->solution_confidence = avg_conf / (float)(result_out->num_nodes > 0 ? result_out->num_nodes : 1);

    float weighted_embed[DTC_EMBED_DIM] = {0};
    float total_weight = 0.0f;
    for (size_t i = 0; i < result_out->num_nodes; i++) {
        float w = result_out->nodes[i].confidence;
        for (size_t j = 0; j < DTC_EMBED_DIM; j++) {
            weighted_embed[j] += result_out->nodes[i].thought_embedding[j] * w;
        }
        total_weight += w;
    }
    if (total_weight > 1e-10f) {
        for (size_t j = 0; j < DTC_EMBED_DIM; j++) weighted_embed[j] /= total_weight;
    }

    result_out->final_output = (float*)safe_calloc(DTC_EMBED_DIM, sizeof(float));
    memcpy(result_out->final_output, weighted_embed, DTC_EMBED_DIM * sizeof(float));
    result_out->output_dim = DTC_EMBED_DIM;

    return 0;
}

int dtc_expand_branch(DTCSystem* system,
                       DTCChainResult* chain,
                       size_t node_index,
                       DTCStepType branch_type) {
    if (!system || !chain) return -1;
    if (node_index >= chain->num_nodes) return -2;
    if (chain->num_nodes >= DTC_MAX_CHAIN_LEN) return -3;

    DTCThoughtNode* parent = &chain->nodes[node_index];

    size_t max_branches = system->config.max_branches;
    if (max_branches > DTC_MAX_BRANCHES) max_branches = DTC_MAX_BRANCHES;

    size_t actual_branches = 0;
    unsigned int seed = (unsigned int)(node_index * 37 + 13);

    for (size_t b = 0; b < max_branches && chain->num_nodes < DTC_MAX_CHAIN_LEN; b++) {
        size_t idx = chain->num_nodes;
        DTCThoughtNode* branch = &chain->nodes[idx];
        branch->step_type = branch_type;
        branch->parent_index = node_index;

        float branch_input[DTC_EMBED_DIM];
        for (size_t j = 0; j < DTC_EMBED_DIM; j++) {
            float noise = ((float)((seed + b * 73 + j * 11) % 1000) / 1000.0f - 0.5f) * 0.5f;
            branch_input[j] = parent->thought_embedding[j] + noise;
        }

        float branch_out[DTC_EMBED_DIM] = {0};
        lnn_forward(system->thought_net, branch_input, branch_out);
        memcpy(branch->thought_embedding, branch_out, DTC_EMBED_DIM * sizeof(float));

        float divergence = cosine_sim_dtc(branch_out, parent->thought_embedding, DTC_EMBED_DIM);
        branch->confidence = parent->confidence * (0.7f + 0.3f * (1.0f - divergence));
        branch->uncertainty = parent->uncertainty * (1.0f + divergence * 0.5f);
        branch->branching_factor = parent->branching_factor * 0.5f;
        branch->has_branches = 0;

        branch->thought_text = (char*)safe_calloc(512, 1);
        snprintf(branch->thought_text, 512, "[分支%zu: %s] 父%zu 发散度:%.2f",
                 b, step_names[branch_type], node_index, 1.0f - divergence);
        branch->text_len = strlen(branch->thought_text);

        chain->num_nodes++;
        actual_branches++;
    }

    parent->has_branches = 1;
    return (int)actual_branches;
}

int dtc_beam_search(DTCSystem* system,
                     const float* input_data,
                     size_t input_dim,
                     const char* query,
                     int beam_width,
                     DTCChainResult* result_out) {
    if (!system || !result_out) return -1;
    if (!input_data && !query) return -2;

    memset(result_out, 0, sizeof(DTCChainResult));

    if (beam_width < 1) beam_width = system->config.beam_width > 0 ? system->config.beam_width : 3;
    if (beam_width > 8) beam_width = 8;

    float base_embed[DTC_EMBED_DIM] = {0};
    if (input_data && input_dim > 0) {
        size_t copy_dim = input_dim < DTC_EMBED_DIM ? input_dim : DTC_EMBED_DIM;
        memcpy(base_embed, input_data, copy_dim * sizeof(float));
    } else if (query) {
        /* N-004修复: bigram哈希编码替代ASCII/255 */
        size_t qlen = strlen(query);
        int hash_count = 0;
        for (size_t i = 0; i + 1 < qlen && i < DTC_EMBED_DIM * 4; i++) {
            uint32_t h = ((uint32_t)(unsigned char)query[i] << 8) | (uint32_t)(unsigned char)query[i+1];
            h = h * 2654435761u;
            size_t idx = (size_t)(h % DTC_EMBED_DIM);
            base_embed[idx] += 1.0f;
            hash_count++;
        }
        if (hash_count > 0) {
            float inv = 1.0f / sqrtf((float)hash_count + 1e-8f);
            for (size_t i = 0; i < DTC_EMBED_DIM; i++) base_embed[i] *= inv;
        }
    }

    float norm = 0.0f;
    for (size_t i = 0; i < DTC_EMBED_DIM; i++) norm += base_embed[i] * base_embed[i];
    norm = sqrtf(norm);
    if (norm > 1e-10f) {
        for (size_t i = 0; i < DTC_EMBED_DIM; i++) base_embed[i] /= norm;
    }

    size_t max_steps = system->config.max_depth;
    if (max_steps > DTC_MAX_CHAIN_LEN / (size_t)beam_width) max_steps = DTC_MAX_CHAIN_LEN / (size_t)beam_width;

    float beam_embeds[8][DTC_EMBED_DIM];
    float beam_scores[8];
    size_t beam_parents[8];
    int beam_alive[8];
    size_t beam_lengths[8];

    for (int b = 0; b < beam_width; b++) {
        memcpy(beam_embeds[b], base_embed, DTC_EMBED_DIM * sizeof(float));
        beam_scores[b] = 0.0f;
        beam_parents[b] = 0;
        beam_alive[b] = 1;
        beam_lengths[b] = 0;
    }

    unsigned int seed = (unsigned int)(query ? strlen(query) : 42);

    for (size_t step = 0; step < max_steps; step++) {
        float candidate_embeds[8 * 4][DTC_EMBED_DIM];
        float candidate_scores[8 * 4];
        int candidate_beam[8 * 4];
        size_t candidate_parents[8 * 4];
        int candidate_count = 0;

        for (int b = 0; b < beam_width; b++) {
            if (!beam_alive[b]) continue;

            float current[DTC_EMBED_DIM];
            memcpy(current, beam_embeds[b], DTC_EMBED_DIM * sizeof(float));

            int expansions = 2 + (step % 3);
            for (int e = 0; e < expansions && candidate_count < 8 * 4; e++) {
                float thought_input[DTC_EMBED_DIM];
                for (size_t j = 0; j < DTC_EMBED_DIM; j++) {
                    float noise = ((float)((seed + step * 53 + b * 31 + e * 17 + j * 7) % 1000) / 1000.0f - 0.5f);
                    thought_input[j] = current[j] + noise * system->config.temperature * (1.0f + (float)e * 0.2f);
                }

                float thought_out[DTC_EMBED_DIM] = {0};
                lnn_forward(system->thought_net, thought_input, thought_out);
                memcpy(candidate_embeds[candidate_count], thought_out, DTC_EMBED_DIM * sizeof(float));

                float eval_input[DTC_EMBED_DIM * 2];
                memcpy(eval_input, current, DTC_EMBED_DIM * sizeof(float));
                memcpy(eval_input + DTC_EMBED_DIM, thought_out, DTC_EMBED_DIM * sizeof(float));
                float eval_out[3] = {0};
                lnn_forward(system->evaluation_net, eval_input, eval_out);

                float confidence = DTC_SIGMOID(eval_out[0]);
                float coherence = cosine_sim_dtc(thought_out, current, DTC_EMBED_DIM);
                float exploration_bonus = (1.0f - coherence) * system->config.exploration_rate;

                candidate_scores[candidate_count] = beam_scores[b] + confidence * 0.6f + exploration_bonus * 0.4f;
                candidate_beam[candidate_count] = b;
                candidate_parents[candidate_count] = beam_lengths[b];
                candidate_count++;
            }
        }

        if (candidate_count == 0) break;

        typedef struct {
            float score;
            int idx;
        } ScoreIdx;
        ScoreIdx sorted[8 * 4];
        for (int i = 0; i < candidate_count; i++) {
            sorted[i].score = candidate_scores[i];
            sorted[i].idx = i;
        }
        for (int i = 0; i < candidate_count - 1; i++) {
            for (int j = i + 1; j < candidate_count; j++) {
                if (sorted[j].score > sorted[i].score) {
                    ScoreIdx tmp = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = tmp;
                }
            }
        }

        int new_beam_count = beam_width < candidate_count ? beam_width : candidate_count;
        int used_beam[8] = {0};

        for (int nb = 0; nb < new_beam_count; nb++) {
            int idx = sorted[nb].idx;
            int source_beam = candidate_beam[idx];

            if (used_beam[source_beam] < 2) {
                used_beam[source_beam]++;

                memcpy(beam_embeds[nb], candidate_embeds[idx], DTC_EMBED_DIM * sizeof(float));
                beam_scores[nb] = candidate_scores[idx];
                beam_parents[nb] = source_beam;
                beam_alive[nb] = 1;
                beam_lengths[nb] = beam_lengths[source_beam] + 1;
            }
        }

        for (int b = new_beam_count; b < beam_width; b++) {
            beam_alive[b] = 0;
        }
    }

    int best_beam = 0;
    for (int b = 1; b < beam_width; b++) {
        if (beam_alive[b] && beam_scores[b] > beam_scores[best_beam]) {
            best_beam = b;
        }
    }

    size_t write_idx = 0;
    float current[DTC_EMBED_DIM];
    memcpy(current, base_embed, DTC_EMBED_DIM * sizeof(float));

    /* ZSFZS-F014修复: 追踪沿最佳束路径的逐步嵌入，
     * 而非所有节点共享同一个 beam_embeds[best_beam]。
     * 每步更新current为前一步输出的加权平均，模拟推理过程中的语义漂移。 */
    float step_embed[DTC_EMBED_DIM];
    memcpy(step_embed, current, DTC_EMBED_DIM * sizeof(float));

    for (size_t step = 0; step < beam_lengths[best_beam] && write_idx < DTC_MAX_CHAIN_LEN; step++) {
        DTCThoughtNode* node = &result_out->nodes[write_idx];
        node->step_type = (DTCStepType)(step % 8);
        node->parent_index = (step > 0) ? (int)write_idx - 1 : 0;
        /* 每步嵌入沿语义轨迹线性演进：从base_embed向beam_embeds方向移动 */
        float alpha = (float)(step + 1) / (float)(beam_lengths[best_beam] + 1);
        for (size_t j = 0; j < DTC_EMBED_DIM; j++) {
            step_embed[j] = current[j] * (1.0f - alpha) + beam_embeds[best_beam][j] * alpha;
        }
        memcpy(node->thought_embedding, step_embed, DTC_EMBED_DIM * sizeof(float));

        float conf = cosine_sim_dtc(step_embed, current, DTC_EMBED_DIM);
        node->confidence = DTC_CLAMP(conf * 0.5f + 0.5f, 0.0f, 1.0f);
        node->uncertainty = 1.0f - node->confidence;
        node->branching_factor = 0.3f;
        node->has_branches = 0;

        node->thought_text = (char*)safe_calloc(512, 1);
        snprintf(node->thought_text, 512, "[波束%zu: %s] 得分:%.2f 置信度:%.2f",
                 step, step_names[node->step_type], beam_scores[best_beam], node->confidence);
        node->text_len = strlen(node->thought_text);

        for (size_t j = 0; j < DTC_EMBED_DIM; j++) {
            current[j] = current[j] * 0.3f + beam_embeds[best_beam][j] * 0.7f;
        }

        write_idx++;
    }

    result_out->num_nodes = write_idx;
    result_out->chain_coherence = 0.0f;
    int pc = 0;
    for (size_t i = 1; i < result_out->num_nodes; i++) {
        float sim = cosine_sim_dtc(
            result_out->nodes[i].thought_embedding,
            result_out->nodes[i - 1].thought_embedding, DTC_EMBED_DIM);
        result_out->chain_coherence += sim;
        pc++;
    }
    if (pc > 0) result_out->chain_coherence /= (float)pc;

    result_out->reasoning_depth = (float)result_out->num_nodes / (float)DTC_MAX_CHAIN_LEN;
    float avg = 0.0f;
    for (size_t i = 0; i < result_out->num_nodes; i++)
        avg += result_out->nodes[i].confidence;
    result_out->solution_confidence = avg / (float)(result_out->num_nodes > 0 ? result_out->num_nodes : 1);

    result_out->final_output = (float*)safe_calloc(DTC_EMBED_DIM, sizeof(float));
    if (result_out->num_nodes > 0) {
        memcpy(result_out->final_output,
               result_out->nodes[result_out->num_nodes - 1].thought_embedding,
               DTC_EMBED_DIM * sizeof(float));
    }
    result_out->output_dim = DTC_EMBED_DIM;

    return 0;
}

typedef struct {
    size_t path_nodes[DTC_MAX_CHAIN_LEN];
    size_t path_len;
    float confidence_sum;
    float coherence;
    float avg_confidence;
    float diversity_score;
} ConsistencyPath;

int dtc_self_consistency_rerank(DTCSystem* system,
                                 DTCChainResult* chain,
                                 DTCChainResult* reranked_out) {
    if (!system || !chain || !reranked_out) return -1;
    if (chain->num_nodes == 0) return -2;

    memset(reranked_out, 0, sizeof(DTCChainResult));

    size_t num_nodes = chain->num_nodes;
    int num_paths = 0;
    ConsistencyPath paths[16];

    for (size_t start = 0; start < num_nodes && num_paths < 16; start += 2) {
        ConsistencyPath* cp = &paths[num_paths];
        cp->path_len = 0;
        cp->confidence_sum = 0.0f;
        cp->coherence = 0.0f;
        cp->avg_confidence = 0.0f;
        cp->diversity_score = 0.0f;

        size_t idx = start;
        while (idx < num_nodes && cp->path_len < DTC_MAX_CHAIN_LEN) {
            cp->path_nodes[cp->path_len] = idx;
            cp->confidence_sum += chain->nodes[idx].confidence;
            cp->path_len++;

            if (chain->nodes[idx].has_branches && idx + 1 < num_nodes) {
                size_t best_child = idx + 1;
                float best_conf = chain->nodes[idx + 1].confidence;
                for (size_t c = idx + 1; c < num_nodes; c++) {
                    if (chain->nodes[c].parent_index == idx &&
                        chain->nodes[c].confidence > best_conf) {
                        best_child = c;
                        best_conf = chain->nodes[c].confidence;
                    }
                }
                idx = best_child;
            } else {
                idx++;
            }
        }

        for (size_t i = 1; i < cp->path_len; i++) {
            cp->coherence += cosine_sim_dtc(
                chain->nodes[cp->path_nodes[i]].thought_embedding,
                chain->nodes[cp->path_nodes[i - 1]].thought_embedding,
                DTC_EMBED_DIM);
        }
        if (cp->path_len > 1) cp->coherence /= (float)(cp->path_len - 1);
        cp->avg_confidence = cp->confidence_sum / (float)cp->path_len;

        if (num_paths > 0) {
            float div = 0.0f;
            for (int p = 0; p < num_paths; p++) {
                float sim = cosine_sim_dtc(
                    chain->nodes[cp->path_nodes[0]].thought_embedding,
                    chain->nodes[paths[p].path_nodes[0]].thought_embedding,
                    DTC_EMBED_DIM);
                div += sim;
            }
            cp->diversity_score = 1.0f - (div / (float)num_paths);
        } else {
            cp->diversity_score = 1.0f;
        }

        num_paths++;
    }

    if (num_paths == 0) {
        memcpy(reranked_out, chain, sizeof(DTCChainResult));
        return 0;
    }

    int best_path = 0;
    float best_score = -1.0f;
    for (int p = 0; p < num_paths; p++) {
        float score = paths[p].avg_confidence * 0.4f +
                      paths[p].coherence * 0.3f +
                      paths[p].diversity_score * 0.3f;
        if (score > best_score) {
            best_score = score;
            best_path = p;
        }
    }

    ConsistencyPath* bp = &paths[best_path];
    for (size_t i = 0; i < bp->path_len && i < (size_t)DTC_MAX_CHAIN_LEN; i++) {
        memcpy(&reranked_out->nodes[i], &chain->nodes[bp->path_nodes[i]],
               sizeof(DTCThoughtNode));
        reranked_out->nodes[i].thought_text = chain->nodes[bp->path_nodes[i]].thought_text;
        reranked_out->num_nodes = i + 1;
    }

    reranked_out->chain_coherence = bp->coherence;
    reranked_out->reasoning_depth = (float)reranked_out->num_nodes / (float)DTC_MAX_CHAIN_LEN;
    reranked_out->solution_confidence = bp->avg_confidence;

    reranked_out->final_output = (float*)safe_calloc(DTC_EMBED_DIM, sizeof(float));
    if (reranked_out->num_nodes > 0) {
        memcpy(reranked_out->final_output,
               reranked_out->nodes[reranked_out->num_nodes - 1].thought_embedding,
               DTC_EMBED_DIM * sizeof(float));
    }
    reranked_out->output_dim = DTC_EMBED_DIM;

    return 0;
}

int dtc_merge_thoughts(DTCSystem* system,
                        DTCChainResult* chain,
                        size_t* merge_pairs,
                        size_t num_pairs,
                        DTCChainResult* merged_out) {
    if (!system || !chain || !merge_pairs || !merged_out) return -1;
    if (num_pairs == 0) {
        memcpy(merged_out, chain, sizeof(DTCChainResult));
        return 0;
    }

    memset(merged_out, 0, sizeof(DTCChainResult));

    size_t merge_node_a = merge_pairs[0];
    size_t merge_node_b = merge_pairs[1];

    if (merge_node_a >= chain->num_nodes || merge_node_b >= chain->num_nodes) return -3;

    float merge_input[DTC_EMBED_DIM * 2];
    memcpy(merge_input, chain->nodes[merge_node_a].thought_embedding, DTC_EMBED_DIM * sizeof(float));
    memcpy(merge_input + DTC_EMBED_DIM, chain->nodes[merge_node_b].thought_embedding, DTC_EMBED_DIM * sizeof(float));

    float merge_out[DTC_EMBED_DIM] = {0};
    lnn_forward(system->merge_net, merge_input, merge_out);

    size_t write_idx = 0;
    for (size_t i = 0; i < chain->num_nodes; i++) {
        if (i == merge_node_b) continue;

        DTCThoughtNode* src = &chain->nodes[i];
        DTCThoughtNode* dst = &merged_out->nodes[write_idx];

        memcpy(dst, src, sizeof(DTCThoughtNode));
        if (i == merge_node_a) {
            memcpy(dst->thought_embedding, merge_out, DTC_EMBED_DIM * sizeof(float));
            dst->confidence = (src->confidence + chain->nodes[merge_node_b].confidence) * 0.5f;

            dst->thought_text = (char*)safe_calloc(512, 1);
            snprintf(dst->thought_text, 512, "[合并 %zu+%zu] 置信度:%.2f",
                     merge_node_a, merge_node_b, dst->confidence);
            dst->text_len = strlen(dst->thought_text);
        } else {
            dst->thought_text = src->thought_text;
        }

        write_idx++;
    }

    merged_out->num_nodes = write_idx;
    merged_out->chain_coherence = chain->chain_coherence;
    merged_out->reasoning_depth = chain->reasoning_depth;
    merged_out->solution_confidence = chain->solution_confidence;

    merged_out->final_output = (float*)safe_calloc(DTC_EMBED_DIM, sizeof(float));
    if (merged_out->num_nodes > 0) {
        memcpy(merged_out->final_output, merge_out, DTC_EMBED_DIM * sizeof(float));
    }
    merged_out->output_dim = DTC_EMBED_DIM;

    return 0;
}

int dtc_evaluate_chain(DTCSystem* system,
                        DTCChainResult* chain,
                        float* coherence_out,
                        float* depth_out,
                        float* confidence_out) {
    if (!system || !chain || !coherence_out || !depth_out || !confidence_out) return -1;

    *coherence_out = chain->chain_coherence;
    *depth_out = chain->reasoning_depth;
    *confidence_out = chain->solution_confidence;
    return 0;
}

int dtc_prune_chain(DTCSystem* system,
                     DTCChainResult* chain,
                     float min_confidence) {
    if (!system || !chain) return -1;

    size_t write_idx = 0;
    int pruned = 0;

    for (size_t i = 0; i < chain->num_nodes; i++) {
        if (chain->nodes[i].confidence >= min_confidence) {
            if (write_idx != i) {
                memcpy(&chain->nodes[write_idx], &chain->nodes[i], sizeof(DTCThoughtNode));
                chain->nodes[write_idx].thought_text = chain->nodes[i].thought_text;
            }
            write_idx++;
        } else {
            safe_free((void**)&chain->nodes[i].thought_text);
            chain->nodes[i].thought_text = NULL;
            pruned++;
        }
    }

    chain->num_nodes = write_idx;

    float avg_conf = 0.0f;
    for (size_t i = 0; i < chain->num_nodes; i++) avg_conf += chain->nodes[i].confidence;
    if (chain->num_nodes > 0) chain->solution_confidence = avg_conf / (float)chain->num_nodes;

    return pruned;
}

int dtc_backtrack(DTCSystem* system,
                   DTCChainResult* chain,
                   size_t to_index) {
    if (!system || !chain) return -1;
    if (to_index >= chain->num_nodes) return -2;

    for (size_t i = to_index + 1; i < chain->num_nodes; i++) {
        safe_free((void**)&chain->nodes[i].thought_text);
        chain->nodes[i].thought_text = NULL;
    }

    chain->num_nodes = to_index + 1;

    float avg_conf = 0.0f;
    for (size_t i = 0; i < chain->num_nodes; i++) avg_conf += chain->nodes[i].confidence;
    if (chain->num_nodes > 0) chain->solution_confidence = avg_conf / (float)chain->num_nodes;

    return 0;
}

int dtc_get_best_path(DTCSystem* system,
                       DTCChainResult* chain,
                       size_t* path_indices,
                       size_t* path_len) {
    if (!system || !chain || !path_indices || !path_len) return -1;

    size_t count = 0;
    float best_score = -1.0f;
    size_t best_node = 0;

    for (size_t i = 0; i < chain->num_nodes; i++) {
        float score = chain->nodes[i].confidence * 0.5f +
                     (1.0f - chain->nodes[i].uncertainty) * 0.3f +
                     chain->nodes[i].branching_factor * 0.2f;
        if (score > best_score) {
            best_score = score;
            best_node = i;
        }
    }

    size_t current = best_node;
    while (1) {
        path_indices[count++] = current;
        if (count >= *path_len) break;

        int found_child = 0;
        for (size_t i = 0; i < chain->num_nodes; i++) {
            if (chain->nodes[i].parent_index == current) {
                float child_score = chain->nodes[i].confidence;
                if (child_score > 0.3f) {
                    current = i;
                    found_child = 1;
                    break;
                }
            }
        }
        if (!found_child) break;
    }

    *path_len = count;
    return 0;
}

int dtc_chain_to_text(DTCSystem* system,
                       DTCChainResult* chain,
                       char* text_out,
                       size_t max_len) {
    if (!system || !chain || !text_out) return -1;

    char* ptr = text_out;
    size_t remaining = max_len;
    int written;

    written = snprintf(ptr, remaining,
        "深度思考链 (长度:%zu 连贯性:%.2f 深度:%.2f 置信度:%.2f)\n\n",
        chain->num_nodes, chain->chain_coherence,
        chain->reasoning_depth, chain->solution_confidence);
    if (written > 0 && (size_t)written < remaining) {
        ptr += written;
        remaining -= (size_t)written;
    }

    for (size_t i = 0; i < chain->num_nodes && remaining > 10; i++) {
        DTCThoughtNode* node = &chain->nodes[i];
        written = snprintf(ptr, remaining,
            "  [%zu] %s | 置信度:%.2f 不确定性:%.2f%s\n",
            i, node->thought_text ? node->thought_text : "",
            node->confidence, node->uncertainty,
            node->has_branches ? " [分支]" : "");
        if (written > 0 && (size_t)written < remaining) {
            ptr += written;
            remaining -= (size_t)written;
        }
    }

    return 0;
}

int dtc_chain_to_plan(DTCSystem* system,
                       DTCChainResult* chain,
                       float* plan_out,
                       size_t* plan_len,
                       size_t max_plan_steps) {
    if (!system || !chain || !plan_out || !plan_len) return -1;
    if (chain->num_nodes == 0) return -2;

    size_t steps = (chain->num_nodes < max_plan_steps) ? chain->num_nodes : max_plan_steps;

    for (size_t i = 0; i < steps; i++) {
        size_t idx = i * chain->num_nodes / steps;
        if (idx >= chain->num_nodes) idx = chain->num_nodes - 1;

        float* plan_step = plan_out + i * (DTC_EMBED_DIM + 2);
        memcpy(plan_step, chain->nodes[idx].thought_embedding, DTC_EMBED_DIM * sizeof(float));
        plan_step[DTC_EMBED_DIM] = (float)chain->nodes[idx].step_type;
        plan_step[DTC_EMBED_DIM + 1] = chain->nodes[idx].confidence;
    }

    *plan_len = steps;
    return 0;
}

void dtc_chain_free(DTCChainResult* chain) {
    if (!chain) return;
    for (size_t i = 0; i < chain->num_nodes; i++) {
        safe_free((void**)&chain->nodes[i].thought_text);
    }
    safe_free((void**)&chain->final_output);
    memset(chain, 0, sizeof(DTCChainResult));
}
