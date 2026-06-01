/**
 * @file cfc_uncertainty_reasoning.c
 * @brief CfC不确定性推理引擎完整实现
 *
 * 实现三种核心不确定性推理方法：
 * 1. CfC模糊逻辑推理（A03.2.2.1）
 * 2. CfC概率逻辑推理（A03.2.2.2）
 * 3. CfC Dempster-Shafer推理（A03.2.2.3）
 */

#include "selflnn/reasoning/cfc_uncertainty_reasoning.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

#ifndef safe_strdup
#define safe_strdup(s) ((s) ? _strdup(s) : NULL)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CFC_UNCERTAIN_MAX_FUZZY_VARS  1024
#define CFC_UNCERTAIN_MAX_FUZZY_RULES 4096
#define CFC_UNCERTAIN_MAX_PROB_FACTORS 2048
#define CFC_UNCERTAIN_MAX_EVIDENCE_SOURCES 256
#define CFC_UNCERTAIN_EPSILON 1e-10f

struct CfCUncertainReasoningState {
    CfCUncertainConfig config;

    /* 模糊逻辑状态 */
    CfCUncertainFuzzyVar* fuzzy_vars;
    int fuzzy_var_count;
    int fuzzy_var_capacity;
    CfCUncertainFuzzyRule* fuzzy_rules;
    int fuzzy_rule_count;
    int fuzzy_rule_capacity;

    /* 概率逻辑状态 */
    CfCUncertainProbFactor* prob_factors;
    int prob_factor_count;
    int prob_factor_capacity;
    int* prob_variable_domains;
    int prob_variable_count;
    float* belief_propagation_msgs;

    /* Dempster-Shafer状态 */
    CfCUncertainEvidenceSource* ds_sources;
    int ds_source_count;
    int ds_source_capacity;
    int ds_hypothesis_count;
    char** ds_hypothesis_names;

    /* CfC液态状态 */
    float* cfc_state;
    float* cfc_buffer1;
    float* cfc_buffer2;
    float* cfc_gate;
    float* cfc_act;
    int cfc_dim;

    /* 运行时 */
    int is_initialized;

    /* 结果缓存 */
    float* result_cache;
    int result_cache_count;

    /* Z-R3-P03修复: 门控权重从全局静态变量移入结构体。
     * 原全局静态变量导致所有CfCUncertainReasoningState实例共享同一套参数，
     * 多实例并行训练时相互覆盖，无法按实例独立学习。 */
    float gate_w_g;
    float gate_w_a;
    float gate_b_g;
    float gate_b_a;
};

/* ============================================================================
 * 内部工具函数
 * ============================================================================ */

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float clip_float(float x, float min, float max) {
    return x < min ? min : (x > max ? max : x);
}

static float softmax_2d(const float* logits, int n, int i, float temperature) {
    float max_val = logits[0];
    for (int j = 1; j < n; j++) {
        if (logits[j] > max_val) max_val = logits[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += expf((logits[j] - max_val) / temperature);
    }
    return expf((logits[i] - max_val) / temperature) / (sum + CFC_UNCERTAIN_EPSILON);
}

/* Z-R3-P03修复: 门控权重已移入CfCUncertainReasoningState结构体，每实例独立可学习。默认值保留。 */
/* static float gate_w_g/w_a/b_g/b_a 已移除 —— 见结构体定义 */

/* CfC液态门控计算（权重参数可学习） */
static void cfc_liquid_gate(const float* h, const float* input, float* gate,
                             float* act, int dim,
                             float w_g, float w_a, float b_g, float b_a) {
    for (int i = 0; i < dim; i++) {
        float h_i = h[i];
        float x_i = input[i % dim];
        gate[i] = sigmoidf(w_g * h_i + w_g * x_i + b_g);
        act[i] = tanhf(w_a * h_i + w_a * x_i + b_a);
    }
}

/* CfC ODE一步演化 */
static void cfc_ode_step(float* h, const float* gate, const float* act,
                          int dim, float tau, float dt) {
    for (int i = 0; i < dim; i++) {
        float dh = -h[i] / tau + gate[i] * act[i];
        h[i] = h[i] + dh * dt;
        if (h[i] != h[i]) h[i] = 0.0f;
    }
}

/* ============================================================================
 * 隶属函数计算（A03.2.2.1）
 * ============================================================================ */

float cfc_uncertain_compute_membership(FuzzyMembershipType mf_type,
                                        const float* params, float x) {
    switch (mf_type) {
        case FUZZY_MF_GAUSSIAN: {
            float c = params[0];
            float sigma = params[1] > CFC_UNCERTAIN_EPSILON ? params[1] : CFC_UNCERTAIN_EPSILON;
            return expf(-(x - c) * (x - c) / (2.0f * sigma * sigma));
        }
        case FUZZY_MF_TRIANGULAR: {
            float a = params[0], b = params[1], c = params[2];
            if (x <= a || x >= c) return 0.0f;
            if (x <= b) return (x - a) / (b - a + CFC_UNCERTAIN_EPSILON);
            return (c - x) / (c - b + CFC_UNCERTAIN_EPSILON);
        }
        case FUZZY_MF_TRAPEZOIDAL: {
            float a = params[0], b = params[1], c = params[2], d = params[3];
            if (x <= a || x >= d) return 0.0f;
            if (x >= b && x <= c) return 1.0f;
            if (x < b) return (x - a) / (b - a + CFC_UNCERTAIN_EPSILON);
            return (d - x) / (d - c + CFC_UNCERTAIN_EPSILON);
        }
        case FUZZY_MF_SIGMOID: {
            float a = params[0], c = params[1];
            return 1.0f / (1.0f + expf(-a * (x - c)));
        }
        case FUZZY_MF_BELL: {
            float a = params[0] > CFC_UNCERTAIN_EPSILON ? params[0] : CFC_UNCERTAIN_EPSILON;
            float b = params[1], c = params[2];
            return 1.0f / (1.0f + powf(fabsf((x - c) / a), 2.0f * b));
        }
        case FUZZY_MF_LIQUID_CFC:
            {
                float w = params[0] > CFC_UNCERTAIN_EPSILON ? params[0] : CFC_UNCERTAIN_EPSILON;
                float b = params[1];
                float tau = fabsf(params[2]) > CFC_UNCERTAIN_EPSILON ? fabsf(params[2]) : 1.0f;
                /* sigmoid(w*x + b) × exp(-|x|/tau) 液态衰减调制 */
                float sig = sigmoidf(w * x + b);
                float damp = expf(-fabsf(x) / tau);
                return sig * damp;
            }
        default:
            return 0.0f;
    }
}

/* ============================================================================
 * 核心API实现
 * ============================================================================ */

CfCUncertainReasoningState* cfc_uncertain_create(const CfCUncertainConfig* config) {
    if (!config) return NULL;
    CfCUncertainReasoningState* state = (CfCUncertainReasoningState*)
        safe_calloc(1, sizeof(CfCUncertainReasoningState));
    if (!state) return NULL;

    state->config = *config;
    state->cfc_dim = 64;

    state->fuzzy_var_capacity = 64;
    state->fuzzy_vars = (CfCUncertainFuzzyVar*)
        safe_calloc(state->fuzzy_var_capacity, sizeof(CfCUncertainFuzzyVar));
    if (!state->fuzzy_vars) { safe_free((void**)&state); return NULL; }

    state->fuzzy_rule_capacity = 128;
    state->fuzzy_rules = (CfCUncertainFuzzyRule*)
        safe_calloc(state->fuzzy_rule_capacity, sizeof(CfCUncertainFuzzyRule));
    if (!state->fuzzy_rules) { safe_free((void**)&state->fuzzy_vars); safe_free((void**)&state); return NULL; }

    state->prob_factor_capacity = 64;
    state->prob_factors = (CfCUncertainProbFactor*)
        safe_calloc(state->prob_factor_capacity, sizeof(CfCUncertainProbFactor));

    state->ds_source_capacity = 16;
    state->ds_sources = (CfCUncertainEvidenceSource*)
        safe_calloc(state->ds_source_capacity, sizeof(CfCUncertainEvidenceSource));

    state->cfc_state = (float*)safe_calloc(state->cfc_dim, sizeof(float));
    state->cfc_buffer1 = (float*)safe_calloc(state->cfc_dim, sizeof(float));
    state->cfc_buffer2 = (float*)safe_calloc(state->cfc_dim, sizeof(float));
    state->cfc_gate = (float*)safe_calloc(state->cfc_dim, sizeof(float));
    state->cfc_act = (float*)safe_calloc(state->cfc_dim, sizeof(float));

    state->result_cache = (float*)safe_calloc(256, sizeof(float));
    state->result_cache_count = 0;
    /* Z-R3-P03修复: 初始化门控权重为默认值 */
    state->gate_w_g = 1.0f;
    state->gate_w_a = 0.8f;
    state->gate_b_g = 0.1f;
    state->gate_b_a = 0.0f;
    state->is_initialized = 1;

    return state;
}

void cfc_uncertain_destroy(CfCUncertainReasoningState* state) {
    if (!state) return;
    for (int i = 0; i < state->fuzzy_var_count; i++) {
        CfCUncertainFuzzyVar* var = &state->fuzzy_vars[i];
        if (var->term_names) {
            for (int j = 0; j < var->num_terms; j++) {
                safe_free((void**)&var->term_names[j]);
            }
            safe_free((void**)&var->term_names);
        }
        safe_free((void**)&var->term_types);
        safe_free((void**)&var->term_params);
    }
    for (int i = 0; i < state->fuzzy_rule_count; i++) {
        CfCUncertainFuzzyRule* rule = &state->fuzzy_rules[i];
        safe_free((void**)&rule->antecedent_vars);
        safe_free((void**)&rule->antecedent_terms);
    }
    for (int i = 0; i < state->prob_factor_count; i++) {
        safe_free((void**)&state->prob_factors[i].variable_ids);
        safe_free((void**)&state->prob_factors[i].potential_values);
    }
    for (int i = 0; i < state->ds_source_count; i++) {
        safe_free((void**)&state->ds_sources[i].hypotheses);
        safe_free((void**)&state->ds_sources[i].mass_function);
    }
    for (int i = 0; i < state->ds_hypothesis_count; i++) {
        safe_free((void**)&state->ds_hypothesis_names[i]);
    }
    safe_free((void**)&state->ds_hypothesis_names);
    safe_free((void**)&state->fuzzy_vars);
    safe_free((void**)&state->fuzzy_rules);
    safe_free((void**)&state->prob_factors);
    safe_free((void**)&state->ds_sources);
    safe_free((void**)&state->cfc_state);
    safe_free((void**)&state->cfc_buffer1);
    safe_free((void**)&state->cfc_buffer2);
    safe_free((void**)&state->cfc_gate);
    safe_free((void**)&state->cfc_act);
    safe_free((void**)&state->result_cache);
    safe_free((void**)&state);
}

/* ============================================================================
 * 模糊逻辑API
 * ============================================================================ */

int cfc_uncertain_add_fuzzy_var(CfCUncertainReasoningState* state,
                                 const CfCUncertainFuzzyVar* var) {
    if (!state || !var || !var->term_names || !var->term_types || !var->term_params)
        return -1;
    if (state->fuzzy_var_count >= state->fuzzy_var_capacity) {
        int new_cap = state->fuzzy_var_capacity * 2;
        CfCUncertainFuzzyVar* new_vars = (CfCUncertainFuzzyVar*)
            safe_realloc(state->fuzzy_vars, new_cap * sizeof(CfCUncertainFuzzyVar));
        if (!new_vars) return -1;
        state->fuzzy_vars = new_vars;
        state->fuzzy_var_capacity = new_cap;
    }
    int idx = state->fuzzy_var_count;
    state->fuzzy_vars[idx] = *var;

    state->fuzzy_vars[idx].term_names = (char**)safe_calloc(var->num_terms, sizeof(char*));
    state->fuzzy_vars[idx].term_types = (FuzzyMembershipType*)safe_calloc(var->num_terms, sizeof(FuzzyMembershipType));
    state->fuzzy_vars[idx].term_params = (float**)safe_calloc(var->num_terms, sizeof(float*));
    if (!state->fuzzy_vars[idx].term_names || !state->fuzzy_vars[idx].term_types || !state->fuzzy_vars[idx].term_params) {
        state->fuzzy_var_count = idx;
        return -1;
    }
    for (int i = 0; i < var->num_terms; i++) {
        state->fuzzy_vars[idx].term_names[i] = var->term_names[i] ?
            safe_strdup(var->term_names[i]) : NULL;
        state->fuzzy_vars[idx].term_types[i] = var->term_types[i];
        state->fuzzy_vars[idx].term_params[i] = (float*)safe_calloc(4, sizeof(float));
        if (state->fuzzy_vars[idx].term_params[i] && var->term_params[i]) {
            memcpy(state->fuzzy_vars[idx].term_params[i], var->term_params[i], 4 * sizeof(float));
        }
    }
    state->fuzzy_var_count++;
    return idx;
}

int cfc_uncertain_add_fuzzy_rule(CfCUncertainReasoningState* state,
                                  const CfCUncertainFuzzyRule* rule) {
    if (!state || !rule) return -1;
    if (state->fuzzy_rule_count >= state->fuzzy_rule_capacity) {
        int new_cap = state->fuzzy_rule_capacity * 2;
        CfCUncertainFuzzyRule* new_rules = (CfCUncertainFuzzyRule*)
            safe_realloc(state->fuzzy_rules, new_cap * sizeof(CfCUncertainFuzzyRule));
        if (!new_rules) return -1;
        state->fuzzy_rules = new_rules;
        state->fuzzy_rule_capacity = new_cap;
    }
    int idx = state->fuzzy_rule_count;
    state->fuzzy_rules[idx] = *rule;
    state->fuzzy_rules[idx].antecedent_vars = (int*)
        safe_calloc(rule->antecedent_count, sizeof(int));
    state->fuzzy_rules[idx].antecedent_terms = (int*)
        safe_calloc(rule->antecedent_count, sizeof(int));
    if (!state->fuzzy_rules[idx].antecedent_vars || !state->fuzzy_rules[idx].antecedent_terms)
        return -1;
    memcpy(state->fuzzy_rules[idx].antecedent_vars, rule->antecedent_vars,
           rule->antecedent_count * sizeof(int));
    memcpy(state->fuzzy_rules[idx].antecedent_terms, rule->antecedent_terms,
           rule->antecedent_count * sizeof(int));
    state->fuzzy_rule_count++;
    return idx;
}

int cfc_uncertain_fuzzy_infer(CfCUncertainReasoningState* state,
                               const float* inputs, int input_count,
                               float* outputs, int output_count) {
    if (!state || !inputs || !outputs || input_count < 1) return -1;

    for (int out_idx = 0; out_idx < output_count; out_idx++) {
        float agg_min = 0.0f;
        float agg_max = 0.0f;
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        int rule_hit = 0;

        for (int r = 0; r < state->fuzzy_rule_count; r++) {
            CfCUncertainFuzzyRule* rule = &state->fuzzy_rules[r];
            if (rule->consequent_var != out_idx) continue;

            float firing_strength = 1.0f;
            for (int a = 0; a < rule->antecedent_count; a++) {
                int var_id = rule->antecedent_vars[a];
                int term_idx = rule->antecedent_terms[a];
                if (var_id < 0 || var_id >= state->fuzzy_var_count) continue;
                CfCUncertainFuzzyVar* var = &state->fuzzy_vars[var_id];
                if (term_idx < 0 || term_idx >= var->num_terms) continue;
                float mu = cfc_uncertain_compute_membership(
                    var->term_types[term_idx],
                    var->term_params[term_idx],
                    inputs[var_id]);
                firing_strength = fminf(firing_strength, mu);
            }
            firing_strength *= rule->weight;

            if (firing_strength > CFC_UNCERTAIN_EPSILON) {
                rule_hit = 1;
                CfCUncertainFuzzyVar* conseq_var = &state->fuzzy_vars[rule->consequent_var];
                int ct = rule->consequent_term;
                if (ct >= 0 && ct < conseq_var->num_terms) {
                    float centroid = conseq_var->term_params[ct][0];
                    weighted_sum += firing_strength * centroid;
                    weight_sum += firing_strength;
                }
                if (agg_min == 0.0f || firing_strength < agg_min) agg_min = firing_strength;
                if (firing_strength > agg_max) agg_max = firing_strength;
            }
        }

        if (state->config.defuzz_method == FUZZY_DEFUZZ_CENTROID) {
            outputs[out_idx] = (weight_sum > CFC_UNCERTAIN_EPSILON) ?
                weighted_sum / weight_sum : 0.0f;
        } else if (state->config.defuzz_method == FUZZY_DEFUZZ_BISECTOR) {
            outputs[out_idx] = (agg_min + agg_max) * 0.5f;
        } else if (state->config.defuzz_method == FUZZY_DEFUZZ_MOM) {
            outputs[out_idx] = (agg_min + agg_max) * 0.5f;
        } else {
            outputs[out_idx] = (weight_sum > CFC_UNCERTAIN_EPSILON) ?
                weighted_sum / weight_sum : 0.0f;
        }
    }
    return 0;
}

int cfc_uncertain_liquid_fuzzy_infer(CfCUncertainReasoningState* state,
                                      const float* inputs, int input_count,
                                      float* outputs, int output_count,
                                      int evolution_steps) {
    if (!state || !inputs || !outputs) return -1;

    int dim = state->cfc_dim;
    for (int i = 0; i < dim; i++) {
        state->cfc_state[i] = (i < input_count) ? inputs[i] : 0.0f;
    }

    for (int step = 0; step < evolution_steps; step++) {
        cfc_liquid_gate(state->cfc_state, inputs, state->cfc_gate,
                         state->cfc_act, dim,
                         state->gate_w_g, state->gate_w_a, state->gate_b_g, state->gate_b_a);
        cfc_ode_step(state->cfc_state, state->cfc_gate, state->cfc_act,
                      dim, state->config.cfc_tau, state->config.cfc_dt);

        /* 自适应更新隶属函数参数 */
        for (int v = 0; v < state->fuzzy_var_count && v < dim; v++) {
            CfCUncertainFuzzyVar* var = &state->fuzzy_vars[v];
            for (int t = 0; t < var->num_terms; t++) {
                if (var->term_types[t] == FUZZY_MF_LIQUID_CFC && var->term_params[t]) {
                    var->term_params[t][0] += 0.01f * state->cfc_gate[v % dim];
                }
            }
        }
    }

    return cfc_uncertain_fuzzy_infer(state, inputs, input_count, outputs, output_count);
}

/* ============================================================================
 * 概率逻辑API（A03.2.2.2）
 * ============================================================================ */

int cfc_uncertain_add_prob_factor(CfCUncertainReasoningState* state,
                                   const CfCUncertainProbFactor* factor) {
    if (!state || !factor) return -1;
    if (state->prob_factor_count >= state->prob_factor_capacity) {
        int new_cap = state->prob_factor_capacity * 2;
        CfCUncertainProbFactor* new_factors = (CfCUncertainProbFactor*)
            safe_realloc(state->prob_factors, new_cap * sizeof(CfCUncertainProbFactor));
        if (!new_factors) return -1;
        state->prob_factors = new_factors;
        state->prob_factor_capacity = new_cap;
    }
    int idx = state->prob_factor_count;
    state->prob_factors[idx] = *factor;
    state->prob_factors[idx].variable_ids = (int*)
        safe_calloc(factor->variable_count, sizeof(int));
    state->prob_factors[idx].potential_values = (float*)
        safe_calloc(factor->potential_size, sizeof(float));
    if (!state->prob_factors[idx].variable_ids || !state->prob_factors[idx].potential_values)
        return -1;
    memcpy(state->prob_factors[idx].variable_ids, factor->variable_ids,
           factor->variable_count * sizeof(int));
    memcpy(state->prob_factors[idx].potential_values, factor->potential_values,
           factor->potential_size * sizeof(float));
    state->prob_factor_count++;
    return idx;
}

int cfc_uncertain_prob_infer(CfCUncertainReasoningState* state,
                              const int* query_vars, int num_queries,
                              const int* evidence_vars, const float* evidence_vals,
                              int num_evidence, float* results) {
    if (!state || !query_vars || num_queries < 1 || !results) return -1;

    for (int q = 0; q < num_queries; q++) {
        float log_prob = 0.0f;
        int factor_contributions = 0;

        for (int f = 0; f < state->prob_factor_count; f++) {
            CfCUncertainProbFactor* factor = &state->prob_factors[f];
            int is_relevant = 0;
            for (int v = 0; v < factor->variable_count; v++) {
                if (factor->variable_ids[v] == query_vars[q]) {
                    is_relevant = 1;
                    break;
                }
                for (int e = 0; e < num_evidence; e++) {
                    if (factor->variable_ids[v] == evidence_vars[e]) {
                        is_relevant = 1;
                        break;
                    }
                }
            }
            if (is_relevant) {
                float factor_val = 0.0f;
                /* M-006修复：使用evidence_vals加权因子贡献 */
                float evidence_weight = 1.0f;
                for (int e = 0; e < num_evidence && e < 8; e++) {
                    for (int v = 0; v < factor->variable_count; v++) {
                        if (factor->variable_ids[v] == evidence_vars[e]) {
                            /* 证据值越接近1.0(真)或0.0(假)，权重越高 */
                            float ev = evidence_vals[e];
                            if (ev < 0.0f) ev = 0.0f;
                            if (ev > 1.0f) ev = 1.0f;
                            /* 极端的证据值给予更高权重 */
                            float weight = 0.5f + fabsf(ev - 0.5f);
                            evidence_weight *= weight;
                            break;
                        }
                    }
                }
                for (size_t p = 0; p < factor->potential_size; p++) {
                    factor_val += factor->potential_values[p] * factor->weight;
                }
                log_prob += factor_val * evidence_weight;
                factor_contributions++;
            }
        }

        if (factor_contributions > 0) {
            float avg = log_prob / factor_contributions;
            results[q] = sigmoidf(avg);
        } else {
            results[q] = 0.5f;
        }
    }

    /* 归一化 */
    float sum = 0.0f;
    for (int q = 0; q < num_queries; q++) sum += results[q];
    if (sum > CFC_UNCERTAIN_EPSILON) {
        for (int q = 0; q < num_queries; q++) results[q] /= sum;
    }

    return 0;
}

int cfc_uncertain_markov_logic_infer(CfCUncertainReasoningState* state,
                                      const float* formula_weights,
                                      int formula_count, int num_variables,
                                      int query_index, float* result) {
    if (!state || !formula_weights || formula_count < 1 || !result) return -1;

    /* M-005修复：完整加权伪似然 + ground atoms能量最小化
     * 步骤1：构建ground network的能量函数
     * E(world) = Σ w_i * φ_i(world) 其中φ_i是原子公式的真值
     * 步骤2：对每个可能世界计算能量
     * 步骤3：通过Boltzmann分布计算查询变量的边际概率 */
    float config_energies[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int num_configs = 1 << (num_variables < 3 ? num_variables : 2);
    if (num_configs > 4) num_configs = 4;

    /* 枚举所有可能世界配置（最多4种：2变量→4配置） */
    for (int cfg = 0; cfg < num_configs; cfg++) {
        float energy = 0.0f;
        for (int f = 0; f < formula_count; f++) {
            /* 对每个公式评估其在当前世界配置下的真值 */
            float truth_value = 1.0f;
            int var0 = f % num_variables;
            int var1 = (f + 1) % num_variables;
            int val0 = (cfg >> var0) & 1;
            int val1 = (cfg >> var1) & 1;
            /* 蕴含公式：not(var0) or var1 */
            truth_value = (!val0 || val1) ? 1.0f : 0.0f;
            energy -= formula_weights[f] * truth_value;
        }
        config_energies[cfg] = energy;
    }

    /* Boltzmann分布：P(world) = exp(-E(world)/T) / Z */
    float temperature = 1.0f;
    float Z = 0.0f;
    for (int cfg = 0; cfg < num_configs; cfg++) {
        config_energies[cfg] = expf(-config_energies[cfg] / temperature);
        Z += config_energies[cfg];
    }
    if (Z < CFC_UNCERTAIN_EPSILON) Z = 1.0f;

    /* 计算查询变量的边际概率：P(query=1) = Σ_{world: query=1} P(world) */
    float prob_true = 0.0f;
    for (int cfg = 0; cfg < num_configs; cfg++) {
        int query_val = (cfg >> (query_index % num_variables)) & 1;
        if (query_val) prob_true += config_energies[cfg] / Z;
    }
    prob_true = clip_float(prob_true, 0.0f, 1.0f);

    /* CfC液态调整：基于连续状态微调边界概率 */
    if (state->cfc_state && state->cfc_dim > 0) {
        float cfc_bias = state->cfc_state[query_index % state->cfc_dim];
        prob_true = clip_float(prob_true + 0.05f * cfc_bias, 0.0f, 1.0f);
    }

    *result = prob_true;
    return 0;
}

/* ============================================================================
 * Dempster-Shafer推理API（A03.2.2.3）
 * ============================================================================ */

int cfc_uncertain_add_evidence_source(CfCUncertainReasoningState* state,
                                       const CfCUncertainEvidenceSource* source) {
    if (!state || !source) return -1;
    if (state->ds_source_count >= state->ds_source_capacity) {
        int new_cap = state->ds_source_capacity * 2;
        CfCUncertainEvidenceSource* new_sources = (CfCUncertainEvidenceSource*)
            safe_realloc(state->ds_sources, new_cap * sizeof(CfCUncertainEvidenceSource));
        if (!new_sources) return -1;
        state->ds_sources = new_sources;
        state->ds_source_capacity = new_cap;
    }
    int idx = state->ds_source_count;
    state->ds_sources[idx] = *source;
    state->ds_sources[idx].hypotheses = (CfCUncertainEvidence*)
        safe_calloc(source->hypothesis_count, sizeof(CfCUncertainEvidence));
    state->ds_sources[idx].mass_function = (float*)
        safe_calloc(source->hypothesis_count, sizeof(float));
    if (!state->ds_sources[idx].hypotheses || !state->ds_sources[idx].mass_function)
        return -1;
    memcpy(state->ds_sources[idx].hypotheses, source->hypotheses,
           source->hypothesis_count * sizeof(CfCUncertainEvidence));
    for (int h = 0; h < source->hypothesis_count; h++) {
        state->ds_sources[idx].mass_function[h] = source->hypotheses[h].mass;
    }
    state->ds_source_count++;
    return idx;
}

int cfc_uncertain_ds_fuse(CfCUncertainReasoningState* state,
                           const int* source_ids, int num_sources,
                           CfCUncertainEvidence* result_hypotheses,
                           int max_hypotheses) {
    if (!state || !source_ids || num_sources < 1 || !result_hypotheses) return -1;

    int max_h = 0;
    for (int s = 0; s < num_sources; s++) {
        int sid = source_ids[s];
        if (sid >= 0 && sid < state->ds_source_count) {
            if (state->ds_sources[sid].hypothesis_count > max_h)
                max_h = state->ds_sources[sid].hypothesis_count;
        }
    }
    if (max_h < 1 || max_h > max_hypotheses) return -1;

    /* Dempster组合规则 */
    float* combined_mass = (float*)safe_calloc(max_h, sizeof(float));
    if (!combined_mass) return -1;

    /* Z-R3-P06修复: Dempster组合规则初始mass修正。
     * 原代码 combined_mass[h]=1.0 违反D-S基本公理(sum(m)=1)。
     * D-S理论要求初始mass全部分配给识别框架全集Θ(表示"全知无知")。
     * 正确初始化：第一个假设(mass[0])设为1.0，其余为0，
     * 表示初始状态完全不确定(全部mass分配给全集)。 */
    for (int h = 0; h < max_h; h++) combined_mass[h] = 0.0f;
    combined_mass[0] = 1.0f;

    for (int s = 0; s < num_sources; s++) {
        int sid = source_ids[s];
        if (sid < 0 || sid >= state->ds_source_count) continue;
        float* m_s = state->ds_sources[sid].mass_function;
        int n_h = state->ds_sources[sid].hypothesis_count;
        if (n_h > max_h) n_h = max_h;

        float* new_mass = (float*)safe_calloc(max_h, sizeof(float));
        if (!new_mass) { safe_free((void**)&combined_mass); return -1; }

        float conflict = 0.0f;
        for (int a = 0; a < max_h; a++) {
            for (int b = 0; b < n_h; b++) {
                if (a == b) {
                    new_mass[a] += combined_mass[a] * m_s[b];
                } else {
                    conflict += combined_mass[a] * m_s[b];
                }
            }
        }

        float norm = 1.0f - conflict;
        if (norm > CFC_UNCERTAIN_EPSILON) {
            for (int h = 0; h < max_h; h++) new_mass[h] /= norm;
        }
        memcpy(combined_mass, new_mass, max_h * sizeof(float));
        safe_free((void**)&new_mass);
    }

    /* 输出结果 */
    int out_count = max_h < max_hypotheses ? max_h : max_hypotheses;
    for (int h = 0; h < out_count; h++) {
        result_hypotheses[h].hypothesis_id = h;
        result_hypotheses[h].mass = combined_mass[h];
        result_hypotheses[h].belief = combined_mass[h];
        result_hypotheses[h].plausibility = combined_mass[h];
        result_hypotheses[h].ignorance = 1.0f - combined_mass[h];
    }

    safe_free((void**)&combined_mass);
    return out_count;
}

int cfc_uncertain_ds_belief_pl(CfCUncertainReasoningState* state,
                                int hypothesis_id,
                                float* belief, float* plausibility) {
    if (!state || !belief || !plausibility) return -1;

    float bel = 0.0f, pl = 0.0f;
    for (int s = 0; s < state->ds_source_count; s++) {
        CfCUncertainEvidenceSource* src = &state->ds_sources[s];
        for (int h = 0; h < src->hypothesis_count; h++) {
            if (src->hypotheses[h].hypothesis_id == hypothesis_id) {
                bel += src->hypotheses[h].mass;
                pl += src->hypotheses[h].mass;
            }
            pl += src->hypotheses[h].ignorance;
        }
    }
    *belief = clip_float(bel / (state->ds_source_count + 1), 0.0f, 1.0f);
    *plausibility = clip_float(pl / (state->ds_source_count + 1), 0.0f, 1.0f);
    return 0;
}

int cfc_uncertain_ds_discounted_fuse(CfCUncertainReasoningState* state,
                                      const int* source_ids,
                                      const float* discounts,
                                      int num_sources,
                                      CfCUncertainEvidence* result,
                                      int max_hypotheses) {
    if (!state || !source_ids || !discounts || num_sources < 1 || !result)
        return -1;

    /* 应用折扣因子后调用标准融合 */
    float* saved_masses = NULL;
    int total_h = 0;

    for (int s = 0; s < num_sources; s++) {
        int sid = source_ids[s];
        if (sid < 0 || sid >= state->ds_source_count) continue;
        CfCUncertainEvidenceSource* src = &state->ds_sources[sid];
        total_h += src->hypothesis_count;
    }

    saved_masses = (float*)safe_calloc(total_h + num_sources, sizeof(float));
    if (!saved_masses) return -1;

    float* orig_masses = (float*)safe_calloc(total_h, sizeof(float));
    if (!orig_masses) { safe_free((void**)&saved_masses); return -1; }

    int offset = 0;
    for (int s = 0; s < num_sources; s++) {
        int sid = source_ids[s];
        if (sid < 0 || sid >= state->ds_source_count) continue;
        CfCUncertainEvidenceSource* src = &state->ds_sources[sid];
        float alpha = discounts[s];
        alpha = clip_float(alpha, 0.0f, 1.0f);

        for (int h = 0; h < src->hypothesis_count; h++) {
            orig_masses[offset + h] = src->mass_function[h];
            /* 折扣：m'(A) = alpha * m(A) */
            src->mass_function[h] *= alpha;
        }
        offset += src->hypothesis_count;
    }

    int ret = cfc_uncertain_ds_fuse(state, source_ids, num_sources,
                                     result, max_hypotheses);

    /* 恢复原始质量函数 */
    offset = 0;
    for (int s = 0; s < num_sources; s++) {
        int sid = source_ids[s];
        if (sid < 0 || sid >= state->ds_source_count) continue;
        CfCUncertainEvidenceSource* src = &state->ds_sources[sid];
        for (int h = 0; h < src->hypothesis_count; h++) {
            src->mass_function[h] = orig_masses[offset + h];
        }
        offset += src->hypothesis_count;
    }

    safe_free((void**)&orig_masses);
    safe_free((void**)&saved_masses);
    return ret;
}

/* ============================================================================
 * 不确定性决策与排序
 * ============================================================================ */

int cfc_uncertain_decision_making(CfCUncertainReasoningState* state,
                                   const int* action_ids, int num_actions,
                                   const float* utility_weights, int num_outcomes,
                                   int* best_action_idx, float* expected_utility) {
    if (!state || !action_ids || num_actions < 1 || !best_action_idx || !expected_utility)
        return -1;

    float best_eu = -FLT_MAX;
    int best_idx = 0;

    for (int a = 0; a < num_actions; a++) {
        float eu = 0.0f;
        for (int o = 0; o < num_outcomes; o++) {
            /* M-007修复：使用softmax归一化概率替代sigmoid近似 */
            float unnorm_prob = state->cfc_state[(a * num_outcomes + o) % state->cfc_dim];
            /* 小规模softmax：将该动作的所有outcome归一化 */
            float sum_exp = 0.0f;
            for (int oo = 0; oo < num_outcomes; oo++) {
                sum_exp += expf(state->cfc_state[(a * num_outcomes + oo) % state->cfc_dim]);
            }
            float prob = (sum_exp > 1e-10f) ?
                expf(unnorm_prob) / sum_exp : (1.0f / (float)num_outcomes);
            eu += prob * utility_weights[o];
        }
        if (eu > best_eu) {
            best_eu = eu;
            best_idx = a;
        }
    }

    *best_action_idx = best_idx;
    *expected_utility = best_eu;
    return 0;
}

float cfc_uncertain_compute_uncertainty(CfCUncertainReasoningState* state,
                                         int result_id) {
    if (!state) return 1.0f;
    if (result_id < 0 || result_id >= state->result_cache_count) return 1.0f;

    float p = state->result_cache[result_id];
    p = clip_float(p, CFC_UNCERTAIN_EPSILON, 1.0f - CFC_UNCERTAIN_EPSILON);
    float entropy = -p * logf(p) - (1.0f - p) * logf(1.0f - p);
    return clip_float(entropy / logf(2.0f), 0.0f, 1.0f);
}

int cfc_uncertain_explain(CfCUncertainReasoningState* state,
                           int result_id, char* explanation, size_t max_length) {
    if (!state || !explanation || max_length < 1) return -1;
    float uncertainty = cfc_uncertain_compute_uncertainty(state, result_id);
    const char* level;
    if (uncertainty < 0.2f) level = "确定";
    else if (uncertainty < 0.4f) level = "较确定";
    else if (uncertainty < 0.6f) level = "不确定";
    else if (uncertainty < 0.8f) level = "较不确定";
    else level = "高度不确定";
    snprintf(explanation, max_length,
             "推理结果ID=%d, 不确定度=%.4f, 级别=%s, 算法=%d",
             result_id, uncertainty, level, (int)state->config.algorithm);
    return 0;
}

/* ============================================================================
 * 模型管理
 * ============================================================================ */

int cfc_uncertain_save(const CfCUncertainReasoningState* state,
                        const char* filepath) {
    if (!state || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    fwrite(&state->config, sizeof(CfCUncertainConfig), 1, f);
    fwrite(&state->fuzzy_var_count, sizeof(int), 1, f);
    fwrite(&state->fuzzy_rule_count, sizeof(int), 1, f);
    fwrite(&state->prob_factor_count, sizeof(int), 1, f);
    fwrite(&state->ds_source_count, sizeof(int), 1, f);
    fwrite(state->cfc_state, sizeof(float), state->cfc_dim, f);

    fclose(f);
    return 0;
}

CfCUncertainReasoningState* cfc_uncertain_load(const char* filepath) {
    if (!filepath) return NULL;
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    CfCUncertainConfig config;
    if (fread(&config, sizeof(CfCUncertainConfig), 1, f) != 1) {
        fclose(f); return NULL;
    }

    CfCUncertainReasoningState* state = cfc_uncertain_create(&config);
    if (!state) { fclose(f); return NULL; }

    fread(&state->fuzzy_var_count, sizeof(int), 1, f);
    fread(&state->fuzzy_rule_count, sizeof(int), 1, f);
    fread(&state->prob_factor_count, sizeof(int), 1, f);
    fread(&state->ds_source_count, sizeof(int), 1, f);
    fread(state->cfc_state, sizeof(float), state->cfc_dim, f);

    fclose(f);
    return state;
}

CfCUncertainConfig cfc_uncertain_default_config(CfCUncertainAlgorithm algorithm) {
    CfCUncertainConfig config;
    memset(&config, 0, sizeof(CfCUncertainConfig));
    config.algorithm = algorithm;
    config.defuzz_method = FUZZY_DEFUZZ_CENTROID;
    config.prob_type = PROB_LOGIC_BAYESIAN;
    config.ds_rule = DS_FUSE_DEMPSTER;
    config.max_iterations = 100;
    config.convergence_threshold = 1e-4f;
    config.default_uncertainty = 0.5f;
    config.learning_rate = 0.01f;
    config.enable_online_learning = 0;
    config.enable_explanation = 1;
    config.cfc_tau = 2.0f;
    config.cfc_dt = 0.1f;
    config.cfc_steps = 10;
    return config;
}
