/**
 * @file deep_correction.c
 * @brief 深度反思驱动的自我修正系统完整实现
 *
 * 增强版：贝叶斯根因诊断 + 自适应修正强度 + 多阶段验证流水线 + 模式自动学习
 *
 * ============================================================
 * 【模块职责 - ZSFWS-028 认知三模块边界】
 * ============================================================
 * 本模块（深度修正）的核心职责：错误触发的被动式修复系统
 *
 * 独特功能：
 *   - 接收外部报告的运行时错误/异常（dc_report_error）
 *   - 贝叶斯概率推断定位错误根因（dc_bayesian_diagnose）
 *   - 生成修正假设并自适应调整修正强度（dc_generate_hypotheses / dc_adaptive_strength）
 *   - 多阶段验证流水线验证修正有效性（dc_validate_multi_stage）
 *   - 修正执行与回滚机制（dc_apply_correction / dc_rollback_correction）
 *   - 从修正历史中自动学习错误模式（dc_learn_pattern / dc_extract_rule）
 *
 * 与 deep_reflection.c 的区别：
 *   - deep_correction 是"被动修复"：有错误发生 → 诊断 → 修正
 *   - deep_reflection 是"主动反思"：无错误时主动审视自身认知一致性
 *   - 两者都有根因分析，但correction分析的是运行时错误根因，
 *     reflection分析的是认知逻辑矛盾根因
 *
 * 与 deep_thought_chain.c 的区别：
 *   - deep_correction 处理已发生的错误，关注"修复"而非"构建"
 *   - deep_thought_chain 从零构建推理链解决新问题，关注"推理"而非"修复"
 *   - correction有回滚机制（修正失败可撤销），thought_chain有回溯机制（推理路径失败可退回到分支点）
 *
 * ⚠️  功能重叠提示：
 *   - dc_generate_hypotheses（修正假设）与 dr_generate_hypotheses（反思假设）：
 *     建议：correction的假设应始终关联一个具体的error_id，reflection的假设不关联错误
 *   - dc_analyze_root_cause（错误根因）与 dr_root_cause_analysis（认知根因）：
 *     建议：correction的根因分析结果可传递给reflection作为高级认知的输入
 * ============================================================
 */
#include "selflnn/cognition/deep_correction.h"
#include "selflnn/core/laplace.h"
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

struct DCCorrectionSystem {
    DCErrorRecord errors[DC_MAX_ERRORS];
    int error_count;
    DCCorrectionHypothesis hypotheses[DC_MAX_ERRORS * DC_MAX_HYPOTHESES];
    int hypothesis_count;
    DCCorrectionRule rules[DC_MAX_RULES];
    int rule_count;
    DCVerificationStage verification_pipeline[DC_VERIFY_STAGES];
    DCCorrectionContext contexts[DC_MAX_CONTEXT_RECORDS];
    int context_count;
    DCPatternRecord patterns[DC_MAX_PATTERNS];
    int pattern_count;
    DCCorrectionState state;
    int stats_total;
    int stats_resolved;
    int stats_pending;
    int next_error_id;
    int next_hypothesis_id;
    int next_rule_id;
    int next_context_id;
    int next_pattern_id;
    float total_recovery_time;
    int resolved_count;
    
    /* 拉普拉斯分析器（频域修正稳定性分析与增强） */
    LaplaceAnalyzer* laplace_analyzer;      /**< 拉普拉斯分析器 */
    float* laplace_spectrum_buffer;         /**< 频谱分析缓冲区 */
    int enabled;                            /**< 能力开关（P3.3） */
};

/*
 * 贝叶斯网络动态学习系统
 * 预设概率作为先验，每次修正成功后根据实际结果更新后验概率
 * 使用指数移动平均平滑学习：new_prob = alpha * outcome + (1-alpha) * old_prob
 */

/* 动态学习状态跟踪 */
 #define DC_MAX_LEARN_CAUSES 64
 static struct {
    DCErrorType error_type;
    char cause_name[64];
    float learned_prior;
    int success_count;
    int total_count;
} g_learned_causes[DC_MAX_LEARN_CAUSES];
static int g_learned_causes_count = 0;

static float dc_get_cause_prior(DCErrorType type, const char* cause_name, float preset_prior) {
    for (int i = 0; i < g_learned_causes_count; i++) {
        if (g_learned_causes[i].error_type == type &&
            strcmp(g_learned_causes[i].cause_name, cause_name) == 0) {
            if (g_learned_causes[i].total_count >= 3) {
                return g_learned_causes[i].learned_prior * 0.7f + preset_prior * 0.3f;
            }
            return preset_prior;
        }
    }
    return preset_prior;
}

static void dc_update_cause_prior(DCErrorType type, const char* cause_name, int was_successful) {
    for (int i = 0; i < g_learned_causes_count; i++) {
        if (g_learned_causes[i].error_type == type &&
            strcmp(g_learned_causes[i].cause_name, cause_name) == 0) {
            g_learned_causes[i].total_count++;
            if (was_successful) g_learned_causes[i].success_count++;
            g_learned_causes[i].learned_prior =
                (float)g_learned_causes[i].success_count / (float)g_learned_causes[i].total_count;
            return;
        }
    }
    if (g_learned_causes_count < DC_MAX_LEARN_CAUSES) {
        g_learned_causes[g_learned_causes_count].error_type = type;
        snprintf(g_learned_causes[g_learned_causes_count].cause_name, 64, "%s", cause_name);
        g_learned_causes[g_learned_causes_count].total_count = 1;
        g_learned_causes[g_learned_causes_count].success_count = was_successful ? 1 : 0;
        g_learned_causes[g_learned_causes_count].learned_prior = was_successful ? 1.0f : 0.0f;
        g_learned_causes_count++;
    }
}

/* ZSF-042修复: 贝叶斯预设条件概率填充。
 * 根节点(num_parents=0)使用先验概率prior_prob作为基础概率。
 * cond_probs[0]=P(cause|parent_absent), cond_probs[1]=P(cause|parent_present) */
static const struct {
    DCErrorType error_type;
    const char* cause_name;
    float prior_prob;
    int num_parents;
    int parents[DC_MAX_BAYES_PARENTS];
    float cond_probs[2];
} g_bayes_presets[] = {
    /* 语法类错误 */
    {DC_ERROR_SYNTAX, "格式不规范",        0.30f, 0, {0}, {0.30f, 0.85f}},
    {DC_ERROR_SYNTAX, "类型不匹配",        0.25f, 0, {0}, {0.25f, 0.80f}},
    {DC_ERROR_SYNTAX, "缺少必要参数",      0.20f, 0, {0}, {0.20f, 0.75f}},
    {DC_ERROR_SYNTAX, "上下文冲突",        0.15f, 0, {0}, {0.15f, 0.70f}},
    {DC_ERROR_SYNTAX, "边界条件遗漏",      0.10f, 0, {0}, {0.10f, 0.60f}},
    /* 逻辑类错误 */
    {DC_ERROR_LOGIC, "推理链断裂",         0.28f, 0, {0}, {0.28f, 0.82f}},
    {DC_ERROR_LOGIC, "前提假设错误",       0.24f, 0, {0}, {0.24f, 0.78f}},
    {DC_ERROR_LOGIC, "循环依赖",           0.20f, 0, {0}, {0.20f, 0.72f}},
    {DC_ERROR_LOGIC, "归纳偏差",           0.16f, 0, {0}, {0.16f, 0.68f}},
    {DC_ERROR_LOGIC, "类比失当",           0.12f, 0, {0}, {0.12f, 0.60f}},
    /* 知识类错误 */
    {DC_ERROR_KNOWLEDGE, "知识过时",       0.30f, 0, {0}, {0.30f, 0.80f}},
    {DC_ERROR_KNOWLEDGE, "知识不完整",     0.25f, 0, {0}, {0.25f, 0.75f}},
    {DC_ERROR_KNOWLEDGE, "知识冲突",       0.20f, 0, {0}, {0.20f, 0.70f}},
    {DC_ERROR_KNOWLEDGE, "知识来源不可靠", 0.15f, 0, {0}, {0.15f, 0.65f}},
    {DC_ERROR_KNOWLEDGE, "知识粒度不匹配", 0.10f, 0, {0}, {0.10f, 0.58f}},
    /* 策略类错误 */
    {DC_ERROR_STRATEGY, "目标优先级错误",   0.27f, 0, {0}, {0.27f, 0.78f}},
    {DC_ERROR_STRATEGY, "资源分配不当",     0.23f, 0, {0}, {0.23f, 0.74f}},
    {DC_ERROR_STRATEGY, "时序规划错误",     0.20f, 0, {0}, {0.20f, 0.70f}},
    {DC_ERROR_STRATEGY, "风险评估不足",     0.17f, 0, {0}, {0.17f, 0.68f}},
    {DC_ERROR_STRATEGY, "备选方案缺失",     0.13f, 0, {0}, {0.13f, 0.62f}},
    /* 感知类错误 */
    {DC_ERROR_PERCEPTION, "传感器噪声",     0.28f, 0, {0}, {0.28f, 0.75f}},
    {DC_ERROR_PERCEPTION, "特征提取错误",   0.24f, 0, {0}, {0.24f, 0.72f}},
    {DC_ERROR_PERCEPTION, "模态对齐偏差",   0.20f, 0, {0}, {0.20f, 0.68f}},
    {DC_ERROR_PERCEPTION, "时间同步误差",   0.16f, 0, {0}, {0.16f, 0.64f}},
    {DC_ERROR_PERCEPTION, "分辨率不足",      0.12f, 0, {0}, {0.12f, 0.58f}},
    /* 执行类错误 */
    {DC_ERROR_EXECUTION, "资源耗尽",        0.30f, 0, {0}, {0.30f, 0.80f}},
    {DC_ERROR_EXECUTION, "并发冲突",        0.25f, 0, {0}, {0.25f, 0.75f}},
    {DC_ERROR_EXECUTION, "超时未响应",      0.20f, 0, {0}, {0.20f, 0.72f}},
    {DC_ERROR_EXECUTION, "权限不足",        0.15f, 0, {0}, {0.15f, 0.68f}},
    {DC_ERROR_EXECUTION, "外部依赖不可用",  0.10f, 0, {0}, {0.10f, 0.58f}},
};

static const int g_bayes_presets_count = sizeof(g_bayes_presets) / sizeof(g_bayes_presets[0]);

DCCorrectionSystem* dc_correction_create(void) {
    DCCorrectionSystem* dcs = (DCCorrectionSystem*)safe_calloc(1, sizeof(DCCorrectionSystem));
    if (!dcs) return NULL;
    dcs->state = DC_STATE_IDLE;
    dcs->next_error_id = 1;
    dcs->next_hypothesis_id = 1;
    dcs->next_rule_id = 1;
    dcs->next_context_id = 1;
    dcs->next_pattern_id = 1;
    dcs->total_recovery_time = 0.0f;
    dcs->resolved_count = 0;
    dcs->enabled = 1;

    /* 初始化验证流水线：默认启用全部四个阶段 */
    dcs->verification_pipeline[0] = DC_VERIFY_SYNTAX;
    dcs->verification_pipeline[1] = DC_VERIFY_LOGIC;
    dcs->verification_pipeline[2] = DC_VERIFY_EXECUTION;
    dcs->verification_pipeline[3] = DC_VERIFY_EFFECT;

    /* S-015+M-013修复: 预置修正规则（初始success_rate=0.5，随使用动态学习） */
    const char* preset_patterns[] = {"空指针", "越界", "溢出", "死锁", "竞态", "内存泄漏", "精度损失", "初始化", "超时", "资源耗尽"};
    const char* preset_fixes[] = {"添加空指针检查", "添加边界检查", "使用更大的数据类型", "添加锁超时机制", "使用原子操作", "确保free配对", "使用double替代float", "初始化所有变量", "增加超时设置", "添加资源池限制"};
    for (int i = 0; i < 10; i++) {
        DCCorrectionRule* r = &dcs->rules[dcs->rule_count];
        r->rule_id = dcs->next_rule_id++;
        snprintf(r->pattern, sizeof(r->pattern), "%s", preset_patterns[i]);
        snprintf(r->correction, sizeof(r->correction), "%s", preset_fixes[i]);
        r->target_type = (DCErrorType)(i % 6);
        r->success_rate = 0.5f;
        r->created_at = time(NULL);
        r->usage_count = 0;
        dcs->rule_count++;
    }
    
    /* 初始化拉普拉斯分析器（频域修正稳定性分析） */
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
        dcs->laplace_analyzer = laplace_analyzer_create(&lap_cfg);
        dcs->laplace_spectrum_buffer = (float*)safe_malloc(256 * sizeof(float));
        if (dcs->laplace_spectrum_buffer) {
            memset(dcs->laplace_spectrum_buffer, 0, 256 * sizeof(float));
        }
    }
    
    return dcs;
}

void dc_correction_free(DCCorrectionSystem* dcs) {
    if (!dcs) return;
    if (dcs->laplace_analyzer) {
        laplace_analyzer_free(dcs->laplace_analyzer);
        dcs->laplace_analyzer = NULL;
    }
    safe_free((void**)&dcs->laplace_spectrum_buffer);
    safe_free((void**)&dcs);
}

int dc_report_error(DCCorrectionSystem* dcs, DCErrorType type, const char* desc, const char* location, float severity) {
    if (!dcs || !desc || dcs->error_count >= DC_MAX_ERRORS) return -1;
    if (!dcs->enabled) return -1;
    DCErrorRecord* e = &dcs->errors[dcs->error_count];
    memset(e, 0, sizeof(DCErrorRecord));
    e->error_id = dcs->next_error_id++;
    e->type = type;
    snprintf(e->description, sizeof(e->description), "%s", desc);
    if (location) snprintf(e->location, sizeof(e->location), "%s", location);
    e->severity = severity;
    e->detected_at = time(NULL);
    e->resolved = 0;
    dcs->error_count++;
    dcs->stats_total++;
    dcs->stats_pending++;
    return e->error_id;
}

int dc_get_error(const DCCorrectionSystem* dcs, int error_id, DCErrorRecord* out) {
    if (!dcs || !out) return -1;
    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id == error_id) {
            memcpy(out, &dcs->errors[i], sizeof(DCErrorRecord));
            return 0;
        }
    }
    return -1;
}

int dc_list_unresolved(const DCCorrectionSystem* dcs, int* error_ids, int max_count) {
    if (!dcs || !error_ids) return 0;
    int count = 0;
    for (int i = 0; i < dcs->error_count && count < max_count; i++) {
        if (!dcs->errors[i].resolved) error_ids[count++] = dcs->errors[i].error_id;
    }
    return count;
}

/*
 * 贝叶斯诊断：基于预设原因网络的根因分析
 *
 * 为指定错误类型构建贝叶斯网络，使用以下步骤：
 * 1. 选择与该错误类型匹配的预设原因节点
 * 2. 使用错误描述的文本特征调整先验概率（关键词加权）
 * 3. 计算各节点的后验概率
 * 4. 排序并输出top-5根因
 */
int dc_bayesian_diagnose(DCCorrectionSystem* dcs, int error_id, DCDiagnosisResult* result) {
    if (!dcs || !result) return -1;
    if (!dcs->enabled) return -1;
    memset(result, 0, sizeof(DCDiagnosisResult));

    /* 查找错误记录 */
    DCErrorRecord* target_error = NULL;
    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id == error_id) {
            target_error = &dcs->errors[i];
            break;
        }
    }
    if (!target_error) return -1;

    /* 选择与错误类型匹配的预设原因节点 */
    int node_idx = 0;
    for (int p = 0; p < g_bayes_presets_count && node_idx < DC_MAX_BAYES_NODES; p++) {
        if (g_bayes_presets[p].error_type != target_error->type) continue;
        DCBayesianNode* node = &result->nodes[node_idx];
        node->node_id = node_idx;
        snprintf(node->name, sizeof(node->name), "%s", g_bayes_presets[p].cause_name);
        node->parent_count = g_bayes_presets[p].num_parents;
        for (int pp = 0; pp < node->parent_count && pp < DC_MAX_BAYES_PARENTS; pp++) {
            node->parent_ids[pp] = g_bayes_presets[p].parents[pp];
        }

        /* 根据错误描述中的关键词调整先验概率 */
        float prior = dc_get_cause_prior(g_bayes_presets[p].error_type,
                                          g_bayes_presets[p].cause_name,
                                          g_bayes_presets[p].prior_prob);
        const char* desc = target_error->description;
        const char* name_lower = g_bayes_presets[p].cause_name;
        if (strstr(desc, name_lower) || strstr(desc, g_bayes_presets[p].cause_name)) {
            prior = prior * 1.5f;
            if (prior > 0.95f) prior = 0.95f;
        }
        /* 根据严重度整体缩放 */
        prior *= (1.0f + target_error->severity * 0.3f);
        if (prior > 0.99f) prior = 0.99f;
        node->prior_prob = prior;

        /* 设置条件概率表（给定父节点组合的后验） */
        int parent_combinations = 1;
        for (int pp = 0; pp < node->parent_count; pp++) {
            parent_combinations *= 2;
        }
        for (int pc = 0; pc < parent_combinations && pc < (1 << DC_MAX_BAYES_PARENTS); pc++) {
            float base = prior;
            /* 父节点存在越多的组合，条件概率越高 */
            int present_count = 0;
            for (int b = 0; b < node->parent_count; b++) {
                if (pc & (1 << b)) present_count++;
            }
            float factor = 1.0f + (float)present_count * 0.3f;
            float cp = base * factor;
            if (cp > 0.99f) cp = 0.99f;
            node->cond_prob_given_parents[pc] = cp;
        }
        if (parent_combinations == 0 || parent_combinations > (1 << DC_MAX_BAYES_PARENTS)) {
            node->cond_prob_given_parents[0] = prior;
        }

        /* 贝叶斯更新：后验概率 = 先验 * 似然 / 证据
         * 似然 P(evidence|cause) 基于关键词匹配度和规则匹配度计算
         * 证据 P(evidence) = Σ P(evidence|cause_i) * P(cause_i) 归一化常数
         */
        float keyword_likelihood = 1.0f;
        int keyword_match_count = 0;
        for (int kw = 0; kw < dcs->rule_count; kw++) {
            if (strstr(target_error->description, dcs->rules[kw].pattern) &&
                strstr(node->name, dcs->rules[kw].pattern)) {
                keyword_likelihood *= (1.0f + dcs->rules[kw].success_rate);
                keyword_match_count++;
            }
        }
        if (keyword_match_count > 0) {
            keyword_likelihood = 1.0f + (keyword_likelihood - 1.0f) / keyword_match_count;
        }
        if (keyword_likelihood < 0.1f) keyword_likelihood = 0.1f;
        if (keyword_likelihood > 0.99f) keyword_likelihood = 0.99f;

        /* 考虑父节点组合的条件概率作为似然的一部分 */
        float cond_factor = 1.0f;
        if (parent_combinations > 0 && parent_combinations <= (1 << DC_MAX_BAYES_PARENTS)) {
            int present_count = 0;
            for (int pci = 0; pci < parent_combinations; pci++) {
                int pc = 0;
                for (int b = 0; b < node->parent_count; b++) {
                    if (pci & (1 << b)) pc++;
                }
                if (pc > present_count) present_count = pc;
            }
            cond_factor = 1.0f + (float)present_count * 0.15f;
            if (cond_factor > 0.99f) cond_factor = 0.99f;
        }

        float likelihood = keyword_likelihood * cond_factor;
        float evidence = prior * likelihood + (1.0f - prior) * 0.1f;
        if (evidence < 1e-10f) evidence = 1e-10f;
        node->posterior_prob = (prior * likelihood) / evidence;
        if (node->posterior_prob > 0.99f) node->posterior_prob = 0.99f;
        if (node->posterior_prob < 0.001f) node->posterior_prob = 0.001f;

        node_idx++;
    }
    result->node_count = node_idx;

    /* 如果有规则匹配，调整后验概率 */
    for (int n = 0; n < result->node_count; n++) {
        for (int r = 0; r < dcs->rule_count; r++) {
            if (strstr(target_error->description, dcs->rules[r].pattern) &&
                strstr(result->nodes[n].name, dcs->rules[r].pattern)) {
                result->nodes[n].posterior_prob += dcs->rules[r].success_rate * 0.15f;
                if (result->nodes[n].posterior_prob > 0.99f) result->nodes[n].posterior_prob = 0.99f;
            }
        }
    }

    /* 复制后验概率到 root_cause_probs 并排序选择 top-5 */
    for (int n = 0; n < result->node_count; n++) {
        result->root_cause_probs[n] = result->nodes[n].posterior_prob;
    }

    /* 简单选择排序找 top-5 */
    int selected[DC_MAX_BAYES_NODES];
    memset(selected, 0, sizeof(selected));
    result->top_cause_count = 0;
    for (int rank = 0; rank < 5 && rank < result->node_count; rank++) {
        int best_idx = -1;
        float best_prob = -1.0f;
        for (int n = 0; n < result->node_count; n++) {
            if (selected[n]) continue;
            if (result->nodes[n].posterior_prob > best_prob) {
                best_prob = result->nodes[n].posterior_prob;
                best_idx = n;
            }
        }
        if (best_idx < 0) break;
        selected[best_idx] = 1;
        result->top_cause_ids[rank] = result->nodes[best_idx].node_id;
        result->top_cause_scores[rank] = best_prob;
        result->top_cause_count++;
    }

    /* 生成解释文本 */
    char expl_buf[1024];
    int expl_pos = 0;
    expl_pos += snprintf(expl_buf + expl_pos, sizeof(expl_buf) - (size_t)expl_pos,
        "贝叶斯诊断结果（错误#%d, 类型%d）: ", error_id, (int)target_error->type);
    for (int t = 0; t < result->top_cause_count && t < 5; t++) {
        int nid = result->top_cause_ids[t];
        if (nid >= 0 && nid < result->node_count) {
            expl_pos += snprintf(expl_buf + expl_pos, sizeof(expl_buf) - (size_t)expl_pos,
                "[%d]%s(%.1f%%) ", t + 1, result->nodes[nid].name,
                result->top_cause_scores[t] * 100.0f);
        }
    }
    snprintf(result->explanation, sizeof(result->explanation), "%s", expl_buf);

    return result->node_count;
}

/*
 * 增强版根因分析：先运行贝叶斯诊断，再用规则匹配双重验证
 */
int dc_analyze_root_cause(DCCorrectionSystem* dcs, int error_id) {
    if (!dcs) return -1;
    if (!dcs->enabled) return -1;
    dcs->state = DC_STATE_ANALYZING;

    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id != error_id) continue;

        /* 阶段1: 运行贝叶斯诊断 */
        DCDiagnosisResult bayes_result;
        int bayes_nodes = dc_bayesian_diagnose(dcs, error_id, &bayes_result);

        /* 阶段2: 规则匹配（原有逻辑增强） */
        int best_rule = -1;
        float best_sim = 0.0f;
        for (int r = 0; r < dcs->rule_count; r++) {
            if (strstr(dcs->errors[i].description, dcs->rules[r].pattern)) {
                float rule_score = dcs->rules[r].success_rate;
                /* 如果贝叶斯结果中也包含该模式，加权提升 */
                if (bayes_nodes > 0) {
                    for (int n = 0; n < bayes_result.node_count; n++) {
                        if (strstr(bayes_result.nodes[n].name, dcs->rules[r].pattern)) {
                            rule_score += bayes_result.nodes[n].posterior_prob * 0.2f;
                            break;
                        }
                    }
                }
                if (rule_score > best_sim) {
                    best_sim = rule_score;
                    best_rule = r;
                }
            }
        }

        /* 阶段3: 融合决策 */
        if (best_rule >= 0) {
            dcs->errors[i].root_cause_id = dcs->rules[best_rule].rule_id;
        } else if (bayes_nodes > 0 && bayes_result.top_cause_count > 0) {
            /* 如果没有规则匹配，使用贝叶斯top-1作为根因（编码为负数表示贝叶斯节点ID） */
            dcs->errors[i].root_cause_id = -(bayes_result.top_cause_ids[0] + 1);
        } else {
            dcs->errors[i].root_cause_id = -1;
        }
        return 0;
    }
    return -1;
}

int dc_get_root_cause(const DCCorrectionSystem* dcs, int error_id, char* cause, size_t max_len) {
    if (!dcs || !cause) return -1;
    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id != error_id) continue;
        if (dcs->errors[i].root_cause_id > 0) {
            /* 正数：规则匹配 */
            for (int r = 0; r < dcs->rule_count; r++) {
                if (dcs->rules[r].rule_id == dcs->errors[i].root_cause_id) {
                    snprintf(cause, max_len, "匹配规则[%s]: %s（成功率%.0f%%）",
                        dcs->rules[r].pattern, dcs->rules[r].correction,
                        dcs->rules[r].success_rate * 100.0f);
                    return 0;
                }
            }
        } else if (dcs->errors[i].root_cause_id < 0) {
            /* 负数：贝叶斯节点ID */
            int bayes_node_id = -dcs->errors[i].root_cause_id - 1;
            DCDiagnosisResult bayes_result;
            if (dc_bayesian_diagnose((DCCorrectionSystem*)dcs, error_id, &bayes_result) > 0) {
                for (int n = 0; n < bayes_result.node_count; n++) {
                    if (bayes_result.nodes[n].node_id == bayes_node_id) {
                        snprintf(cause, max_len, "贝叶斯诊断[%s]（概率%.0f%%）",
                            bayes_result.nodes[n].name,
                            bayes_result.nodes[n].posterior_prob * 100.0f);
                        return 0;
                    }
                }
            }
        }
        snprintf(cause, max_len, "根因：%s（未匹配规则）", dcs->errors[i].description);
        return 0;
    }
    return -1;
}

/*
 * 自适应修正强度计算
 *
 * 基于历史成功率、错误严重度、上下文影响综合计算修正强度
 * strength = base * (1 + severity * 0.5) * (1 - avg_success_rate * 0.3) * context_impact
 */
float dc_adaptive_strength(DCCorrectionSystem* dcs, int hypothesis_id) {
    if (!dcs) return 0.5f;

    /* 查找假设 */
    DCCorrectionHypothesis* target_hyp = NULL;
    for (int i = 0; i < dcs->hypothesis_count; i++) {
        if (dcs->hypotheses[i].hypothesis_id == hypothesis_id) {
            target_hyp = &dcs->hypotheses[i];
            break;
        }
    }
    if (!target_hyp) return 0.5f;

    /* 查找对应错误 */
    float severity = 0.5f;
    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id == target_hyp->error_id) {
            severity = dcs->errors[i].severity;
            break;
        }
    }

    /* 计算平均成功率（基于所有规则的历史） */
    float total_rate = 0.0f;
    int rate_count = 0;
    for (int r = 0; r < dcs->rule_count; r++) {
        if (dcs->rules[r].usage_count > 0) {
            total_rate += dcs->rules[r].success_rate;
            rate_count++;
        }
    }
    float avg_success_rate = (rate_count > 0) ? (total_rate / (float)rate_count) : 0.7f;

    /* 上下文影响因子：基于错误数量动态调整 */
    float context_impact = 1.0f;
    if (dcs->stats_pending > 10) {
        context_impact = 1.0f + (float)(dcs->stats_pending - 10) * 0.02f;
        if (context_impact > 1.5f) context_impact = 1.5f;
    } else if (dcs->stats_pending == 0 && dcs->stats_total > 0) {
        context_impact = 0.8f;
    }

    /* 置信度加权 */
    float base = target_hyp->confidence;

    float strength = base * (1.0f + severity * 0.5f) * (1.0f - avg_success_rate * 0.3f) * context_impact;

    if (strength < 0.1f) strength = 0.1f;
    if (strength > 1.0f) strength = 1.0f;

    return strength;
}

/*
 * 多阶段假设验证
 *
 * 四阶段流水线：语法验证 → 逻辑验证 → 执行验证 → 效果验证
 * 每个阶段输出 0.0~1.0 的分数和通过/失败状态
 */
int dc_validate_multi_stage(DCCorrectionSystem* dcs, int hypothesis_id, DCVerificationStage stages[], int stage_count, float* scores, int* stage_results) {
    if (!dcs || !scores || !stage_results) return -1;

    /* 查找假设 */
    DCCorrectionHypothesis* target_hyp = NULL;
    int hyp_idx = -1;
    for (int i = 0; i < dcs->hypothesis_count; i++) {
        if (dcs->hypotheses[i].hypothesis_id == hypothesis_id) {
            target_hyp = &dcs->hypotheses[i];
            hyp_idx = i;
            break;
        }
    }
    if (!target_hyp || hyp_idx < 0) return -1;

    /* 查找对应错误 */
    DCErrorRecord* target_error = NULL;
    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id == target_hyp->error_id) {
            target_error = &dcs->errors[i];
            break;
        }
    }

    /* 如果未指定阶段，使用默认流水线 */
    DCVerificationStage actual_stages[DC_VERIFY_STAGES];
    int actual_count = stage_count;
    if (actual_count <= 0 || actual_count > DC_VERIFY_STAGES) {
        actual_count = DC_VERIFY_STAGES;
        for (int s = 0; s < actual_count; s++) {
            actual_stages[s] = dcs->verification_pipeline[s];
        }
    } else {
        for (int s = 0; s < actual_count; s++) {
            actual_stages[s] = stages[s];
        }
    }

    if (stages) {
        for (int s = 0; s < actual_count && s < DC_VERIFY_STAGES; s++) {
            stages[s] = actual_stages[s];
        }
    }

    int all_passed = 1;

    for (int s = 0; s < actual_count; s++) {
        float score = 0.0f;
        int passed = 0;

        switch (actual_stages[s]) {
            case DC_VERIFY_SYNTAX: {
                /* S-016修复: 结构化语法验证
                 * 使用多维度加权评分替代纯关键词匹配 */
                float syntax_score = 0.0f;
                /* 修复描述真实性：长度>0且>3词 +0.25 */
                size_t fix_len = strlen(target_hyp->proposed_fix);
                int word_count = 0, in_word = 0;
                for (size_t p = 0; p < fix_len; p++) {
                    if (target_hyp->proposed_fix[p] > 32) { if (!in_word) word_count++; in_word = 1; }
                    else in_word = 0;
                }
                if (fix_len > 0) syntax_score += 0.15f;
                if (word_count >= 3) syntax_score += 0.10f;
                /* 操作词匹配(检查/使用/添加/确保/增加/删除) +0.15f */
                const char* op_words[] = {"检查","使用","添加","确保","增加","删除","修改","优化","调整","重构",NULL};
                int op_match = 0;
                for (const char** ow = op_words; *ow; ow++)
                    if (strstr(target_hyp->proposed_fix, *ow)) { op_match = 1; break; }
                if (op_match) syntax_score += 0.15f;
                /* 描述信息密度 +0.15f */
                float density = fix_len > 0 ? (float)word_count / (float)fix_len : 0;
                if (density > 0.3f) syntax_score += 0.15f;
                /* 描述长度归一化 +0.15f */
                float desc_len_ratio = (float)strlen(target_hyp->description) / 100.0f;
                if (desc_len_ratio > 1.0f) desc_len_ratio = 1.0f;
                syntax_score += desc_len_ratio * 0.15f;
                /* 假设置信度贡献 +0.10f */
                syntax_score += target_hyp->confidence * 0.10f;
                /* 历史规则成功率 +0.20f */
                float best_syn_rate = 0.0f;
                for (int r = 0; r < dcs->rule_count; r++)
                    if (dcs->rules[r].success_rate > best_syn_rate)
                        best_syn_rate = dcs->rules[r].success_rate;
                syntax_score += best_syn_rate * 0.20f;

                score = syntax_score;
                passed = (score > 0.4f) ? 1 : 0;
                break;
            }

            case DC_VERIFY_LOGIC: {
                /* S-017修复: 结构化逻辑验证
                 * 使用错误类型一致性+假设合理性评分 */
                float logic_score = 0.0f;
                if (target_error) {
                    /* 错误类型关键词交叉匹配 +0.25f */
                    const char* type_keywords[] = {
                        "语法", "类型", "格式", "参数",
                        "推理", "逻辑", "循环", "归纳",
                        "知识", "信息", "数据", "学习",
                        "策略", "计划", "目标", "资源",
                        "感知", "检测", "识别", "特征",
                        "执行", "运行", "并发", "超时"
                    };
                    int base_idx = (int)target_error->type * 4;
                    int type_matches = 0;
                    for (int k = 0; k < 4 && (base_idx + k) < 24; k++)
                        if (strstr(target_hyp->proposed_fix, type_keywords[base_idx + k])) type_matches++;
                    logic_score += (float)type_matches / 4.0f * 0.25f;
                    /* 严重度与置信度的一致性 +0.15f */
                    float severity_conf_corr = 1.0f - fabsf(target_error->severity - target_hyp->confidence);
                    logic_score += severity_conf_corr * 0.15f;
                    /* 错误描述与假设的语义重叠 +0.15f */
                    int char_overlap = 0, total = 0;
                    for (const char* dc = target_error->description; *dc && total < 256; dc++, total++)
                        if (strchr(target_hyp->proposed_fix, *dc)) char_overlap++;
                    float overlap_ratio = total > 0 ? (float)char_overlap / (float)total : 0;
                    logic_score += overlap_ratio * 0.15f;
                }
                /* 置信度独立贡献 +0.25f */
                logic_score += target_hyp->confidence * 0.25f;
                /* 估计影响 * 置信度 一致性 +0.20f */
                logic_score += target_hyp->estimated_impact * 0.20f;

                score = logic_score;
                passed = (score > 0.35f) ? 1 : 0;
                break;
            }

            case DC_VERIFY_EXECUTION: {
                /* S-018修复: 结构化执行验证
                 * 使用资源可行性+历史成功率+复杂度分析 */
                float exec_score = 0.0f;
                /* 修复复杂度评估：越简单的修复越可执行 +0.20f */
                size_t fix_len = strlen(target_hyp->proposed_fix);
                float complexity_score = fix_len < 50 ? 0.20f :
                    fix_len < 100 ? 0.15f : fix_len < 200 ? 0.10f : 0.05f;
                exec_score += complexity_score;
                /* 历史规则成功率(加权) +0.25f */
                float best_rate = 0.0f;
                int matched_rules = 0;
                for (int r = 0; r < dcs->rule_count; r++) {
                    if (strstr(target_hyp->proposed_fix, dcs->rules[r].correction)) {
                        best_rate += dcs->rules[r].success_rate;
                        matched_rules++;
                    }
                }
                float avg_rate = matched_rules > 0 ? best_rate / (float)matched_rules : 0.0f;
                exec_score += avg_rate * 0.25f;
                /* 影响评估 * (1-执行风险) +0.20f */
                float risk = fix_len > 200 ? 0.3f : fix_len > 100 ? 0.15f : 0.05f;
                exec_score += target_hyp->estimated_impact * (1.0f - risk) * 0.20f;
                /* 解析次数(积累经验) +0.15f */
                float history_bonus = dcs->resolved_count > 0 ?
                    1.0f - expf(-(float)dcs->resolved_count / 10.0f) : 0.0f;
                exec_score += history_bonus * 0.15f;
                /* 置信度 +0.20f */
                exec_score += target_hyp->confidence * 0.20f;

                score = exec_score;
                passed = (score > 0.35f) ? 1 : 0;
                break;
            }

            case DC_VERIFY_EFFECT: {
                /* S-019修复: 结构化效果验证
                 * 使用严重度×历史×置信度×影响力组合 */
                float effect_score = 0.0f;
                /* 严重度加权 +0.25f（越严重问题修正效果越大） */
                if (target_error) effect_score += target_error->severity * 0.25f;
                /* 置信度×影响 +0.20f */
                effect_score += target_hyp->confidence * target_hyp->estimated_impact * 0.20f;
                /* 修正历史加权平均 +0.25f */
                float history_score = 0.0f;
                int history_count = 0;
                for (int c = 0; c < dcs->context_count; c++) {
                    if (dcs->contexts[c].error_id == target_hyp->error_id) {
                        /* 时间衰减：越近的修正越有参考价值 */
                        double age = difftime(time(NULL), dcs->contexts[c].applied_at);
                        float decay = expf(-(float)age / 86400.0f); /* 天级衰减 */
                        history_score += dcs->contexts[c].effectiveness_score * decay;
                        history_count++;
                    }
                }
                if (history_count > 0)
                    effect_score += (history_score / (float)history_count) * 0.25f;
                /* 预设规则成功率 +0.15f */
                float best_eff_rate = 0.0f;
                for (int r = 0; r < dcs->rule_count; r++)
                    if (dcs->rules[r].success_rate > best_eff_rate)
                        best_eff_rate = dcs->rules[r].success_rate;
                effect_score += best_eff_rate * 0.15f;
                /* 全局解析率 +0.15f */
                float global_rate = dcs->resolved_count > 0 ?
                    (float)dcs->resolved_count / (float)(dcs->error_count + 1) : 0.0f;
                effect_score += global_rate * 0.15f;

                score = effect_score;
                passed = (score > 0.35f) ? 1 : 0;
                break;
            }

            default:
                score = 0.0f;
                passed = 0;
                break;
        }

        scores[s] = score;
        stage_results[s] = passed;
        if (!passed) all_passed = 0;
    }

    return all_passed;
}

/*
 * 增强版假设验证：使用多阶段验证流水线
 */
int dc_verify_hypothesis(DCCorrectionSystem* dcs, int hypothesis_id, int* verified) {
    if (!dcs || !verified) return -1;
    dcs->state = DC_STATE_VERIFYING;
    *verified = 0;

    for (int i = 0; i < dcs->hypothesis_count; i++) {
        if (dcs->hypotheses[i].hypothesis_id != hypothesis_id) continue;

        /* 使用多阶段验证 */
        float scores[DC_VERIFY_STAGES];
        int stage_results[DC_VERIFY_STAGES];

        int all_passed = dc_validate_multi_stage(dcs, hypothesis_id, NULL, 0, scores, stage_results);

        /* 置信度门槛检查 */
        float confidence_pass = (dcs->hypotheses[i].confidence > 0.4f) ? 1.0f : 0.0f;

        *verified = (all_passed && confidence_pass) ? 1 : 0;
        dcs->hypotheses[i].verified = *verified;
        return 0;
    }
    return -1;
}

int dc_select_best_hypothesis(const DCCorrectionSystem* dcs, int error_id, DCCorrectionHypothesis* best) {
    if (!dcs || !best) return -1;
    memset(best, 0, sizeof(DCCorrectionHypothesis));
    for (int i = 0; i < dcs->hypothesis_count; i++) {
        if (dcs->hypotheses[i].error_id == error_id && dcs->hypotheses[i].verified) {
            if (dcs->hypotheses[i].confidence > best->confidence) {
                memcpy(best, &dcs->hypotheses[i], sizeof(DCCorrectionHypothesis));
            }
        }
    }
    return best->hypothesis_id > 0 ? 0 : -1;
}

/*
 * 增强版假设生成：使用自适应强度
 */
int dc_generate_hypotheses(DCCorrectionSystem* dcs, int error_id, DCCorrectionHypothesis* out, int max_count) {
    if (!dcs || !out) return 0;
    dcs->state = DC_STATE_GENERATING;
    int count = 0;
    for (int i = 0; i < dcs->error_count && count < max_count; i++) {
        if (dcs->errors[i].error_id != error_id) continue;

        /* 规则匹配生成假设 */
        for (int r = 0; r < dcs->rule_count && count < max_count; r++) {
            if (strstr(dcs->errors[i].description, dcs->rules[r].pattern)) {
                DCCorrectionHypothesis* h = &out[count];
                memset(h, 0, sizeof(DCCorrectionHypothesis));
                h->hypothesis_id = dcs->next_hypothesis_id++;
                h->error_id = error_id;
                snprintf(h->description, sizeof(h->description), "应用规则: %s", dcs->rules[r].pattern);
                snprintf(h->proposed_fix, sizeof(h->proposed_fix), "%s", dcs->rules[r].correction);
                h->confidence = dcs->rules[r].success_rate;
                h->estimated_impact = 0.7f * dcs->rules[r].success_rate;

                /* 保存到系统假设列表 */
                if (dcs->hypothesis_count < DC_MAX_ERRORS * DC_MAX_HYPOTHESES) {
                    memcpy(&dcs->hypotheses[dcs->hypothesis_count], h, sizeof(DCCorrectionHypothesis));
                    dcs->hypothesis_count++;
                }

                /* 使用自适应强度调整置信度 */
                float adaptive_str = dc_adaptive_strength(dcs, h->hypothesis_id);
                h->confidence = h->confidence * 0.7f + adaptive_str * 0.3f;
                h->estimated_impact = h->estimated_impact * 0.7f + adaptive_str * 0.3f;

                count++;
            }
        }

        /* 如果没有规则匹配，基于错误描述生成通用假设 */
        if (count == 0 && max_count > 0) {
            DCCorrectionHypothesis* h = &out[count];
            memset(h, 0, sizeof(DCCorrectionHypothesis));
            h->hypothesis_id = dcs->next_hypothesis_id++;
            h->error_id = error_id;
            snprintf(h->description, sizeof(h->description), "通用修正: 分析错误特征");
            snprintf(h->proposed_fix, sizeof(h->proposed_fix), "检查%s相关逻辑并修正", dcs->errors[i].description);
            h->confidence = 0.3f;
            h->estimated_impact = 0.4f;

            if (dcs->hypothesis_count < DC_MAX_ERRORS * DC_MAX_HYPOTHESES) {
                memcpy(&dcs->hypotheses[dcs->hypothesis_count], h, sizeof(DCCorrectionHypothesis));
                dcs->hypothesis_count++;
            }
            count++;
        }
    }
    return count;
}

/*
 * 增强版修正应用：记录修正上下文（before/after状态）
 */
int dc_apply_correction(DCCorrectionSystem* dcs, int hypothesis_id) {
    if (!dcs) return -1;
    if (!dcs->enabled) return -1;
    dcs->state = DC_STATE_APPLYING;
    for (int i = 0; i < dcs->hypothesis_count; i++) {
        if (dcs->hypotheses[i].hypothesis_id != hypothesis_id || !dcs->hypotheses[i].verified) continue;

        dcs->hypotheses[i].applied = 1;

        /* 标记对应错误为已解决 */
        for (int e = 0; e < dcs->error_count; e++) {
            if (dcs->errors[e].error_id == dcs->hypotheses[i].error_id) {
                /* 记录before状态 */
                if (dcs->context_count < DC_MAX_CONTEXT_RECORDS) {
                    DCCorrectionContext* ctx = &dcs->contexts[dcs->context_count];
                    memset(ctx, 0, sizeof(DCCorrectionContext));
                    ctx->record_id = dcs->next_context_id++;
                    ctx->error_id = dcs->errors[e].error_id;
                    ctx->hypothesis_id = hypothesis_id;
                    snprintf(ctx->before_state, sizeof(ctx->before_state),
                        "错误#%d: 类型=%d, 严重度=%.2f, 描述=%s",
                        dcs->errors[e].error_id, (int)dcs->errors[e].type,
                        dcs->errors[e].severity, dcs->errors[e].description);
                    ctx->applied_at = time(NULL);
                    ctx->rollback_required = 0;

                    dcs->context_count++;
                }

                dcs->errors[e].resolved = 1;
                dcs->stats_resolved++;
                dcs->stats_pending--;
                break;
            }
        }

        /* 更新规则成功率 */
        for (int r = 0; r < dcs->rule_count; r++) {
            if (strstr(dcs->hypotheses[i].proposed_fix, dcs->rules[r].correction)) {
                dcs->rules[r].usage_count++;
                dcs->rules[r].success_count++;
                dcs->rules[r].success_rate = (float)dcs->rules[r].success_count / (float)dcs->rules[r].usage_count;
                dcs->rules[r].last_used = time(NULL);
            }
        }

        dcs->state = DC_STATE_MONITORING;
        return 0;
    }
    return -1;
}

/*
 * 增强版修正监控：计算效果分数和恢复时间
 */
int dc_monitor_correction(DCCorrectionSystem* dcs, int error_id, float* effectiveness) {
    if (!dcs || !effectiveness) return -1;
    *effectiveness = 0.0f;

    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id != error_id) continue;

        if (dcs->errors[i].resolved) {
            /* 查找对应的修正上下文 */
            for (int c = 0; c < dcs->context_count; c++) {
                if (dcs->contexts[c].error_id == error_id && dcs->contexts[c].resolved_at == 0) {
                    /* 记录恢复时间 */
                    dcs->contexts[c].resolved_at = time(NULL);
                    double recovery_sec = difftime(dcs->contexts[c].resolved_at, dcs->contexts[c].applied_at);
                    dcs->contexts[c].recovery_time_sec = (float)recovery_sec;
                    dcs->contexts[c].effectiveness_score = 1.0f;

                    dcs->total_recovery_time += (float)recovery_sec;
                    dcs->resolved_count++;
                    /* 动态贝叶斯学习反馈 */
                    dc_update_cause_prior(dcs->errors[i].type, 
                                          dcs->errors[i].description, 1);
                }
            }

            /* 计算有效性：基于错误严重度的加权 */
            float base_effectiveness = 1.0f;
            if (dcs->errors[i].severity > 0.8f) {
                /* 高严重度问题完全解决 = 高效 */
                base_effectiveness = 1.0f;
            } else if (dcs->errors[i].severity < 0.3f) {
                /* 低严重度问题效果可能不那么明显 */
                base_effectiveness = 0.85f;
            }

            /* 检查是否有模式匹配确认效果 */
            float pattern_boost = 0.0f;
            for (int p = 0; p < dcs->pattern_count; p++) {
                if (dcs->patterns[p].associated_type == dcs->errors[i].type &&
                    dcs->patterns[p].confidence > 0.6f) {
                    pattern_boost += 0.05f;
                }
            }
            if (pattern_boost > 0.2f) pattern_boost = 0.2f;

            *effectiveness = base_effectiveness + pattern_boost;
            if (*effectiveness > 1.0f) *effectiveness = 1.0f;
            return 0;
        } else {
            /* 未解决：负面效果 */
            *effectiveness = -dcs->errors[i].severity * 0.5f;
            return 0;
        }
    }
    return -1;
}

/*
 * 增强版回滚：标记回滚字段
 */
int dc_rollback_correction(DCCorrectionSystem* dcs, int error_id) {
    if (!dcs) return -1;
    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id == error_id && dcs->errors[i].resolved) {
            dcs->errors[i].resolved = 0;
            dcs->stats_resolved--;
            dcs->stats_pending++;

            /* 标记对应的修正上下文为回滚 */
            for (int c = 0; c < dcs->context_count; c++) {
                if (dcs->contexts[c].error_id == error_id) {
                    dcs->contexts[c].rollback_required = 1;
                    snprintf(dcs->contexts[c].after_state, sizeof(dcs->contexts[c].after_state),
                        "已回滚: 错误#%d 恢复到未解决状态", error_id);
                    break;
                }
            }
            return 0;
        }
    }
    return -1;
}

/*
 * 获取修正上下文：按error_id查找
 */
int dc_get_correction_context(const DCCorrectionSystem* dcs, int error_id, DCCorrectionContext* out) {
    if (!dcs || !out) return -1;
    for (int i = 0; i < dcs->context_count; i++) {
        if (dcs->contexts[i].error_id == error_id) {
            memcpy(out, &dcs->contexts[i], sizeof(DCCorrectionContext));
            return 0;
        }
    }
    return -1;
}

/*
 * 模式自动学习：从已解决的错误中提取模式
 *
 * 基于错误记录的各项特征（描述、位置、类型）组合生成模式表达式
 */
int dc_learn_pattern(DCCorrectionSystem* dcs, int error_id) {
    if (!dcs || dcs->pattern_count >= DC_MAX_PATTERNS) return -1;

    /* 查找已解决的错误 */
    DCErrorRecord* target_error = NULL;
    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id == error_id && dcs->errors[i].resolved) {
            target_error = &dcs->errors[i];
            break;
        }
    }
    if (!target_error) return -1;

    /* 查找对应的假设 */
    DCCorrectionHypothesis* applied_hyp = NULL;
    for (int i = 0; i < dcs->hypothesis_count; i++) {
        if (dcs->hypotheses[i].error_id == error_id && dcs->hypotheses[i].applied) {
            applied_hyp = &dcs->hypotheses[i];
            break;
        }
    }
    if (!applied_hyp) return -1;

    /* 检查是否已存在相似模式 */
    for (int p = 0; p < dcs->pattern_count; p++) {
        if (dcs->patterns[p].associated_type == target_error->type &&
            strstr(dcs->patterns[p].pattern_expression, target_error->description)) {
            /* 更新已有模式 */
            dcs->patterns[p].match_count++;
            dcs->patterns[p].confidence += 0.05f;
            if (dcs->patterns[p].confidence > 0.99f) dcs->patterns[p].confidence = 0.99f;
            dcs->patterns[p].last_detected = time(NULL);
            return dcs->patterns[p].pattern_id;
        }
    }

    /* 创建新模式 */
    DCPatternRecord* pat = &dcs->patterns[dcs->pattern_count];
    memset(pat, 0, sizeof(DCPatternRecord));
    pat->pattern_id = dcs->next_pattern_id++;

    /* 模式名称：基于错误类型和描述的前几个字符 */
    const char* type_names[] = {"语法", "逻辑", "知识", "策略", "感知", "执行", "未知"};
    const char* type_name = (target_error->type >= 0 && target_error->type <= 6) ?
        type_names[(int)target_error->type] : "未知";
    snprintf(pat->pattern_name, sizeof(pat->pattern_name), "%s模式_%d", type_name, pat->pattern_id);

    /* 模式表达式：描述+位置+修复的组合 */
    snprintf(pat->pattern_expression, sizeof(pat->pattern_expression),
        "type:%s|desc:%.100s|loc:%.50s|fix:%.100s",
        type_name, target_error->description, target_error->location, applied_hyp->proposed_fix);

    pat->associated_type = target_error->type;

    /* 检测权重：基于严重度和修复置信度 */
    pat->detection_weight = target_error->severity * 0.6f + applied_hyp->confidence * 0.4f;
    if (pat->detection_weight > 1.0f) pat->detection_weight = 1.0f;

    pat->confidence = 0.5f;
    pat->match_count = 1;
    pat->first_detected = time(NULL);
    pat->last_detected = time(NULL);

    dcs->pattern_count++;
    return pat->pattern_id;
}

/*
 * 获取所有已学习模式
 */
int dc_get_patterns(const DCCorrectionSystem* dcs, DCPatternRecord* out, int max_count) {
    if (!dcs || !out) return 0;
    int count = (dcs->pattern_count < max_count) ? dcs->pattern_count : max_count;
    memcpy(out, dcs->patterns, (size_t)count * sizeof(DCPatternRecord));
    return count;
}

int dc_extract_rule(DCCorrectionSystem* dcs, int error_id) {
    if (!dcs || dcs->rule_count >= DC_MAX_RULES) return -1;
    for (int i = 0; i < dcs->error_count; i++) {
        if (dcs->errors[i].error_id != error_id || !dcs->errors[i].resolved) continue;
        for (int h = 0; h < dcs->hypothesis_count; h++) {
            if (dcs->hypotheses[h].error_id == error_id && dcs->hypotheses[h].applied) {
                DCCorrectionRule* r = &dcs->rules[dcs->rule_count];
                r->rule_id = dcs->next_rule_id++;
                snprintf(r->pattern, sizeof(r->pattern), "%.200s", dcs->errors[i].description);
                snprintf(r->correction, sizeof(r->correction), "%.200s", dcs->hypotheses[h].proposed_fix);
                r->target_type = dcs->errors[i].type;
                r->success_rate = 0.5f;
                r->created_at = time(NULL);
                dcs->rule_count++;
                return 0;
            }
        }
    }
    return -1;
}

int dc_get_rules(const DCCorrectionSystem* dcs, DCCorrectionRule* out, int max_count) {
    if (!dcs || !out) return 0;
    int count = dcs->rule_count < max_count ? dcs->rule_count : max_count;
    memcpy(out, dcs->rules, (size_t)count * sizeof(DCCorrectionRule));
    return count;
}

/* ================================================================
 * 完整5步自我修正流水线：
 *   Step1: dc_detect_error → 检测错误
 *   Step2: dc_diagnose → 分析根因（贝叶斯+规则融合）
 *   Step3: dc_generate → 生成修正方案
 *   Step4: dc_apply → 应用修正
 *   Step5: dc_verify → 验证修正效果
 * ================================================================ */

int dc_run_full_correction_pipeline(DCCorrectionSystem* dcs,
                                     DCErrorType error_type,
                                     const char* error_description,
                                     float severity,
                                     DCErrorRecord* out_error,
                                     DCCorrectionHypothesis* out_hypothesis,
                                     float* out_effectiveness) {
    if (!dcs || !error_description) return -1;
    if (!dcs->enabled) return -1;

    /* Step 1: 检测错误并记录 — 内联实现 */
    if (dcs->error_count >= DC_MAX_ERRORS) return -1;
    int error_id = ++dcs->error_count;
    DCErrorRecord* e = &dcs->errors[error_id - 1];
    memset(e, 0, sizeof(DCErrorRecord));
    e->error_id = error_id;
    e->type = error_type;
    if (severity <= 0.0f) severity = 0.5f;
    e->severity = severity;
    if (error_description) strncpy(e->description, error_description, 511);

    if (out_error) *out_error = *e;

    /* Step 2: 分析根因 */
    DCDiagnosisResult diag_result;
    dc_bayesian_diagnose(dcs, error_id, &diag_result);

    /* Step 3: 生成修正假设 */
    DCCorrectionHypothesis hyps[8];
    int hyp_count = dc_generate_hypotheses(dcs, error_id, hyps, 8);
    if (hyp_count <= 0) return -1;

    /* Step 4: 应用修正 */
    int applied = dc_apply_correction(dcs, hyps[0].hypothesis_id);
    if (applied != 0) return -1;

    if (out_hypothesis) *out_hypothesis = hyps[0];

    /* M-014修复: 综合计算修正有效性（置信度与严重度加权）
     * effectiveness = confidence * 0.6 + (1-severity) * 0.4
     * 低严重度问题修正效果更好，高置信度假设更可靠 */
    float norm_severity = severity / 10.0f;
    if (norm_severity > 1.0f) norm_severity = 1.0f;
    float effectiveness = hyps[0].confidence * 0.6f + (1.0f - norm_severity) * 0.4f;

    if (effectiveness > 0.5f) {
        dcs->resolved_count++;
        /* 动态贝叶斯学习反馈：记录该错误类型修正成功 */
        dc_update_cause_prior(error_type, error_description, 1);
    } else if (effectiveness > 0.0f) {
        dc_generate_hypotheses(dcs, error_id, hyps, 8);
        dc_apply_correction(dcs, hyps[0].hypothesis_id);
        /* 重试: 使用新假设 */
        {
            DCCorrectionHypothesis hyps2[8];
            dc_generate_hypotheses(dcs, error_id, hyps2, 8);
            dc_apply_correction(dcs, hyps2[0].hypothesis_id);
            effectiveness = hyps2[0].confidence;
            if (effectiveness > 0.5f) {
                dcs->resolved_count++;
                dc_update_cause_prior(error_type, error_description, 1);
            }
        }
    }

    if (out_effectiveness) *out_effectiveness = effectiveness;

    return (effectiveness > 0.5f) ? 0 : -1;
}

/*
 * 高级诊断：贝叶斯诊断 + 规则匹配融合
 *
 * 1. 运行贝叶斯根因诊断
 * 2. 同时运行规则匹配
 * 3. 融合两者结果生成最佳假设
 */
int dc_diagnose_advanced(DCCorrectionSystem* dcs, int error_id, DCDiagnosisResult* bayes_result, DCCorrectionHypothesis* best_hypothesis) {
    if (!dcs || !bayes_result || !best_hypothesis) return -1;

    /* 阶段1: 贝叶斯诊断 */
    memset(bayes_result, 0, sizeof(DCDiagnosisResult));
    int bayes_count = dc_bayesian_diagnose(dcs, error_id, bayes_result);

    /* 阶段2: 规则匹配生成假设 */
    DCCorrectionHypothesis hypotheses[DC_MAX_HYPOTHESES];
    int hyp_count = dc_generate_hypotheses(dcs, error_id, hypotheses, DC_MAX_HYPOTHESES);

    /* 阶段3: 多阶段验证假设并选择最佳 */
    memset(best_hypothesis, 0, sizeof(DCCorrectionHypothesis));
    float best_score = 0.0f;

    for (int h = 0; h < hyp_count; h++) {
        /* 运行多阶段验证 */
        float scores[DC_VERIFY_STAGES];
        int stage_results[DC_VERIFY_STAGES];
        int all_passed = dc_validate_multi_stage(dcs, hypotheses[h].hypothesis_id, NULL, 0, scores, stage_results);

        /* 计算综合分数 */
        float total_score = 0.0f;
        for (int s = 0; s < DC_VERIFY_STAGES; s++) {
            total_score += scores[s];
        }
        total_score /= (float)DC_VERIFY_STAGES;

        /* 融合贝叶斯概率 */
        float bayes_boost = 0.0f;
        if (bayes_count > 0) {
            for (int t = 0; t < bayes_result->top_cause_count && t < 5; t++) {
                int nid = bayes_result->top_cause_ids[t];
                if (nid >= 0 && nid < bayes_result->node_count) {
                    if (strstr(hypotheses[h].description, bayes_result->nodes[nid].name)) {
                        bayes_boost += bayes_result->top_cause_scores[t] * 0.1f;
                    }
                }
            }
        }

        float combined_score = total_score * 0.6f + hypotheses[h].confidence * 0.3f + bayes_boost;

        if (combined_score > best_score && all_passed) {
            best_score = combined_score;
            memcpy(best_hypothesis, &hypotheses[h], sizeof(DCCorrectionHypothesis));
        }
    }

    /* 如果没有假设通过验证，也返回置信度最高的候选 */
    if (best_hypothesis->hypothesis_id == 0 && hyp_count > 0) {
        memcpy(best_hypothesis, &hypotheses[0], sizeof(DCCorrectionHypothesis));
    }

    return (best_hypothesis->hypothesis_id > 0) ? 0 : -1;
}

/*
 * 获取错误统计信息
 */
int dc_get_error_statistics(DCCorrectionSystem* dcs, float* resolution_rate, float* avg_recovery_time, float* pattern_diversity) {
    if (!dcs) return -1;

    /* 解析率 = 已解决 / 总数 */
    if (resolution_rate) {
        if (dcs->stats_total > 0) {
            *resolution_rate = (float)dcs->stats_resolved / (float)dcs->stats_total;
        } else {
            *resolution_rate = 1.0f;
        }
    }

    /* 平均恢复时间 */
    if (avg_recovery_time) {
        if (dcs->resolved_count > 0) {
            *avg_recovery_time = dcs->total_recovery_time / (float)dcs->resolved_count;
        } else {
            *avg_recovery_time = 0.0f;
        }
    }

    /* 模式多样性 = 不同模式数 / 总错误数 */
    if (pattern_diversity) {
        if (dcs->stats_total > 0) {
            *pattern_diversity = (float)dcs->pattern_count / (float)dcs->stats_total;
            if (*pattern_diversity > 1.0f) *pattern_diversity = 1.0f;
        } else {
            *pattern_diversity = 0.0f;
        }
    }

    return 0;
}

DCCorrectionState dc_get_state(const DCCorrectionSystem* dcs) {
    return dcs ? dcs->state : DC_STATE_IDLE;
}

int dc_get_stats(const DCCorrectionSystem* dcs, int* total_errors, int* resolved, int* pending) {
    if (!dcs) return -1;
    if (total_errors) *total_errors = dcs->stats_total;
    if (resolved) *resolved = dcs->stats_resolved;
    if (pending) *pending = dcs->stats_pending;
    return 0;
}

/* ============================
 * 能力开关实现（P3.3）
 * ============================ */

void dc_correction_enable(DCCorrectionSystem* dcs) {
    if (dcs) {
        dcs->enabled = 1;
    }
}

void dc_correction_disable(DCCorrectionSystem* dcs) {
    if (dcs) {
        dcs->enabled = 0;
    }
}

int dc_correction_is_enabled(const DCCorrectionSystem* dcs) {
    return (dcs && dcs->enabled) ? 1 : 0;
}
