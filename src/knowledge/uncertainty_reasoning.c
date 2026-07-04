/**
 * @file uncertainty_reasoning.c
 * @brief 不确定性推理引擎完整实现
 *
 * 模糊逻辑推理、概率逻辑推理、Dempster-Shafer证据理论。
 */

#define _CRT_NONSTDC_NO_DEPRECATE

#include "selflnn/knowledge/uncertainty_reasoning.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef _MSC_VER
#pragma warning(disable:4100 4189 4244 4267 4701 4133)
#endif

/* ============================================================================
 * 内部辅助函数
 * =========================================================================== */

static char* ur_dup_str(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char* d = (char*)safe_malloc(len + 1);
    if (d) { memcpy(d, s, len + 1); }
    return d;
}

static float ur_clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* ============================================================================
 * 模糊逻辑推理 - 隶属度函数计算
 * =========================================================================== */

float fuzzy_membership(const MembershipFunction* mf, float x) {
    if (!mf) return 0.0f;
    float a = mf->params[0], b = mf->params[1];
    float c = mf->params[2], d = mf->params[3];

    switch (mf->type) {
    case MF_TRIANGULAR: {
        /* 参数校验：避免除零错误（a==b或b==c时隶属度无定义） */
        if (a == b || b == c) return 0.0f;
        if (x <= a || x >= c) return 0.0f;
        if (x == b) return 1.0f;
        if (x < b) return (x - a) / (b - a);
        return (c - x) / (c - b);
    }
    case MF_TRAPEZOIDAL: {
        if (x <= a || x >= d) return 0.0f;
        if (x >= b && x <= c) return 1.0f;
        if (x < b) return (x - a) / (b - a);
        return (d - x) / (d - c);
    }
    case MF_GAUSSIAN: {
        float diff = x - b;
        return expf(-0.5f * (diff * diff) / (a * a + 1e-10f));
    }
    case MF_BELL: {
        float diff = (x - b) / (a + 1e-10f);
        return 1.0f / (1.0f + powf(fabsf(diff), 2.0f * c));
    }
    case MF_SIGMOID: {
        return 1.0f / (1.0f + expf(-a * (x - b)));
    }
    case MF_ZSHAPE: {
        if (x <= a) return 1.0f;
        if (x >= b) return 0.0f;
        float t = (x - a) / (b - a + 1e-10f);
        return 1.0f - t * t * (3.0f - 2.0f * t);
    }
    default: return 0.0f;
    }
}

/* ============================================================================
 * 模糊推理引擎 - 内部结构
 * =========================================================================== */

struct FuzzyEngine {
    FuzzyInferenceMethod method;
    DefuzzificationMethod defuzz;

    FuzzyVariable* variables;
    int var_count;
    int var_capacity;

    FuzzyRule* rules;
    int rule_count;
    int rule_capacity;

    float* crisp_inputs;     /* 精确输入值 */
    int input_count;

    float** rule_firings;    /* 规则激活度 */
    float* rule_activations;

    float* crisp_outputs;    /* 精确输出值 */
    int output_count;
};

/* 查找模糊变量索引 */
static int fuzzy_find_var(FuzzyEngine* engine, const char* name) {
    for (int i = 0; i < engine->var_count; i++) {
        if (strcmp(engine->variables[i].name, name) == 0) return i;
    }
    return -1;
}

/* 查找术语索引 */
static int fuzzy_find_term(FuzzyVariable* var, const char* name) {
    for (int i = 0; i < var->term_count; i++) {
        if (strcmp(var->terms[i].name, name) == 0) return i;
    }
    return -1;
}

FuzzyEngine* fuzzy_engine_create(FuzzyInferenceMethod method, DefuzzificationMethod defuzz) {
    FuzzyEngine* engine = (FuzzyEngine*)safe_calloc(1, sizeof(FuzzyEngine));
    if (!engine) return NULL;

    engine->method = method;
    engine->defuzz = defuzz;
    engine->var_capacity = 16;
    engine->rule_capacity = 128;

    engine->variables = (FuzzyVariable*)safe_calloc(engine->var_capacity, sizeof(FuzzyVariable));
    engine->rules = (FuzzyRule*)safe_calloc(engine->rule_capacity, sizeof(FuzzyRule));
    engine->crisp_inputs = NULL;
    engine->rule_firings = NULL;
    engine->rule_activations = NULL;
    engine->crisp_outputs = NULL;

    if (!engine->variables || !engine->rules) {
        if (engine->variables) safe_free((void**)&engine->variables);
        if (engine->rules) safe_free((void**)&engine->rules);
        safe_free((void**)&engine);
        return NULL;
    }

    engine->var_count = 0;
    engine->rule_count = 0;
    engine->input_count = 0;
    engine->output_count = 0;

    return engine;
}

void fuzzy_engine_free(FuzzyEngine* engine) {
    if (!engine) return;

    for (int i = 0; i < engine->var_count; i++) {
        FuzzyVariable* v = &engine->variables[i];
        if (v->terms) {
            for (int j = 0; j < v->term_count; j++) {
                /* 名称是嵌入的，不需要释放 */
            }
            safe_free((void**)&v->terms);
        }
    }
    safe_free((void**)&engine->variables);

    /* 规则中的字符串都是嵌入的，不需要释放 */
    safe_free((void**)&engine->rules);

    if (engine->crisp_inputs) safe_free((void**)&engine->crisp_inputs);
    if (engine->crisp_outputs) safe_free((void**)&engine->crisp_outputs);
    if (engine->rule_firings) {
        for (int i = 0; i < engine->rule_count; i++) {
            if (engine->rule_firings[i]) safe_free((void**)&engine->rule_firings[i]);
        }
        safe_free((void**)&engine->rule_firings);
    }
    if (engine->rule_activations) safe_free((void**)&engine->rule_activations);

    safe_free((void**)&engine);
}

int fuzzy_engine_add_variable(FuzzyEngine* engine, const FuzzyVariable* var) {
    if (!engine || !var) return -1;

    if (engine->var_count >= engine->var_capacity) {
        int new_cap = engine->var_capacity * 2;
        FuzzyVariable* new_vars = (FuzzyVariable*)safe_realloc(
            engine->variables, new_cap * sizeof(FuzzyVariable));
        if (!new_vars) return -1;
        engine->variables = new_vars;
        engine->var_capacity = new_cap;
    }

    FuzzyVariable* dst = &engine->variables[engine->var_count];
    memcpy(dst, var, sizeof(FuzzyVariable));

    if (var->term_count > 0) {
        dst->terms = (MembershipFunction*)safe_malloc(
            var->term_count * sizeof(MembershipFunction));
        if (!dst->terms) return -1;
        memcpy(dst->terms, var->terms, var->term_count * sizeof(MembershipFunction));
    } else {
        dst->terms = NULL;
    }

    engine->var_count++;
    return engine->var_count - 1;
}

int fuzzy_engine_add_rule(FuzzyEngine* engine, const FuzzyRule* rule) {
    if (!engine || !rule) return -1;

    if (engine->rule_count >= engine->rule_capacity) {
        int new_cap = engine->rule_capacity * 2;
        FuzzyRule* new_rules = (FuzzyRule*)safe_realloc(
            engine->rules, new_cap * sizeof(FuzzyRule));
        if (!new_rules) return -1;
        engine->rules = new_rules;
        engine->rule_capacity = new_cap;
    }

    FuzzyRule* dst = &engine->rules[engine->rule_count];
    memcpy(dst, rule, sizeof(FuzzyRule));
    dst->rule_id = engine->rule_count;
    engine->rule_count++;

    return dst->rule_id;
}

int fuzzy_engine_set_input(FuzzyEngine* engine, const char* var_name, float value) {
    if (!engine || !var_name) return -1;

    int idx = fuzzy_find_var(engine, var_name);
    if (idx < 0) return -1;

    if (engine->input_count <= idx) {
        float* new_inputs = (float*)safe_realloc(
            engine->crisp_inputs, (idx + 1) * sizeof(float));
        if (!new_inputs) return -1;
        engine->crisp_inputs = new_inputs;
        for (int i = engine->input_count; i <= idx; i++) {
            engine->crisp_inputs[i] = 0.0f;
        }
        engine->input_count = idx + 1;
    }

    engine->crisp_inputs[idx] = value;
    return 0;
}

int fuzzy_engine_evaluate(FuzzyEngine* engine) {
    if (!engine || engine->rule_count == 0) return -1;

    /* 重新分配激活度内存 */
    if (engine->rule_firings) {
        for (int i = 0; i < engine->rule_count; i++) {
            if (engine->rule_firings[i]) safe_free((void**)&engine->rule_firings[i]);
        }
        safe_free((void**)&engine->rule_firings);
    }

    engine->rule_firings = (float**)safe_calloc(engine->rule_count, sizeof(float*));
    engine->rule_activations = (float*)safe_calloc(engine->rule_count, sizeof(float));

    if (!engine->rule_firings || !engine->rule_activations) return -1;

    /* 第一步：计算每个规则的激活度 */
    for (int r = 0; r < engine->rule_count; r++) {
        FuzzyRule* rule = &engine->rules[r];
        if (!rule->is_active) continue;

        float activation = 1.0f;
        int valid = 1;

        for (int p = 0; p < rule->premise_count; p++) {
            FuzzyAntecedent* ant = &rule->premises[p];
            int var_idx = fuzzy_find_var(engine, ant->var_name);
            if (var_idx < 0 || var_idx >= engine->input_count) { valid = 0; break; }

            FuzzyVariable* var = &engine->variables[var_idx];
            int term_idx = fuzzy_find_term(var, ant->term_name);
            if (term_idx < 0) { valid = 0; break; }

            float mu = fuzzy_membership(&var->terms[term_idx], engine->crisp_inputs[var_idx]);
            if (ant->is_negated) mu = 1.0f - mu;

            /* 模糊AND（取最小值） */
            if (mu < activation) activation = mu;
        }

        if (!valid) { engine->rule_activations[r] = 0.0f; continue; }

        activation *= rule->weight;
        engine->rule_activations[r] = activation;

        /* 为每个输出变量记录激活度 */
        engine->rule_firings[r] = (float*)safe_malloc(2 * sizeof(float));
        if (engine->rule_firings[r]) {
            engine->rule_firings[r][0] = activation;
            engine->rule_firings[r][1] = 0.0f;
        }
    }

    /* 第二步：Mamdani或Sugeno推理 */
    if (engine->method == FUZZY_INFER_MAMDANI) {
        /* Mamdani推理：聚合所有规则的输出模糊集 */
        /* 收集所有结论变量 */
        int max_out_vars = engine->var_count;
        if (!engine->crisp_outputs) {
            engine->crisp_outputs = (float*)safe_calloc(max_out_vars, sizeof(float));
            engine->output_count = max_out_vars;
        }

        /* 对每个输出变量，使用重心法去模糊化 */
        for (int r = 0; r < engine->rule_count; r++) {
            FuzzyRule* rule = &engine->rules[r];
            if (!rule->is_active || engine->rule_activations[r] < 1e-8f) continue;

            int var_idx = fuzzy_find_var(engine, rule->conclusion_var);
            if (var_idx < 0) continue;

            FuzzyVariable* var = &engine->variables[var_idx];
            int term_idx = fuzzy_find_term(var, rule->conclusion_term);
            if (term_idx < 0) continue;

            MembershipFunction* conc_mf = &var->terms[term_idx];
            float activation = engine->rule_activations[r];

            /* 使用采样法进行去模糊化 */
            float step = (var->max_value - var->min_value) / UR_FUZZY_SAMPLES;
            float numerator = 0.0f, denominator = 0.0f;

            for (int s = 0; s < UR_FUZZY_SAMPLES; s++) {
                float x = var->min_value + s * step;
                float mu = fuzzy_membership(conc_mf, x);

                /* Mamdani剪切：取min(activation, mu) */
                float clipped = (mu < activation) ? mu : activation;

                switch (engine->defuzz) {
                case DEFUZZ_CENTROID:
                    numerator += x * clipped;
                    denominator += clipped;
                    break;
                case DEFUZZ_BISECTOR: {
                    static float cum_area = 0.0f;
                    static float total_area = 0.0f;
                    if (s == 0) { cum_area = 0.0f; total_area = 0.0f; }
                    cum_area += clipped * step;
                    total_area += clipped * step;
                    if (s == UR_FUZZY_SAMPLES - 1 && total_area > 0) {
                        /* 线性搜索平分点 */
                        float half = total_area * 0.5f;
                        float accum = 0.0f;
                        for (int s2 = 0; s2 < UR_FUZZY_SAMPLES; s2++) {
                            float x2 = var->min_value + s2 * step;
                            float m2 = fuzzy_membership(conc_mf, x2);
                            float c2 = (m2 < activation) ? m2 : activation;
                            accum += c2 * step;
                            if (accum >= half) {
                                engine->crisp_outputs[var_idx] = x2;
                                break;
                            }
                        }
                    }
                    break;
                }
                case DEFUZZ_MOM:
                case DEFUZZ_LOM:
                case DEFUZZ_ROM:
                    /* 这些方法在循环后处理 */
                    break;
                }
            }

            if (engine->defuzz == DEFUZZ_CENTROID && denominator > 1e-10f) {
                engine->crisp_outputs[var_idx] += numerator / denominator;
            }
        }

        /* 处理MOM/LOM/ROM方法 */
        if (engine->defuzz == DEFUZZ_MOM || engine->defuzz == DEFUZZ_LOM ||
            engine->defuzz == DEFUZZ_ROM) {
            for (int r = 0; r < engine->rule_count; r++) {
                FuzzyRule* rule = &engine->rules[r];
                if (!rule->is_active || engine->rule_activations[r] < 1e-8f) continue;

                int var_idx = fuzzy_find_var(engine, rule->conclusion_var);
                if (var_idx < 0) continue;
                FuzzyVariable* var = &engine->variables[var_idx];
                int term_idx = fuzzy_find_term(var, rule->conclusion_term);
                if (term_idx < 0) continue;

                MembershipFunction* conc_mf = &var->terms[term_idx];
                float activation = engine->rule_activations[r];
                float step = (var->max_value - var->min_value) / UR_FUZZY_SAMPLES;

                float max_mu = 0.0f;
                float max_sum = 0.0f, max_count = 0.0f;
                float max_first = var->min_value, max_last = var->min_value;
                int in_max = 0;

                for (int s = 0; s < UR_FUZZY_SAMPLES; s++) {
                    float x = var->min_value + s * step;
                    float mu = fuzzy_membership(conc_mf, x);
                    float clipped = (mu < activation) ? mu : activation;

                    if (clipped > max_mu) {
                        max_mu = clipped;
                        max_first = x;
                        max_last = x;
                        max_sum = x;
                        max_count = 1.0f;
                        in_max = 1;
                    } else if (fabsf(clipped - max_mu) < 1e-8f && in_max) {
                        max_last = x;
                        max_sum += x;
                        max_count += 1.0f;
                    } else if (clipped < max_mu) {
                        in_max = 0;
                    }
                }

                if (max_count > 0) {
                    float mean_of_max = max_sum / max_count;
                    switch (engine->defuzz) {
                    case DEFUZZ_MOM: engine->crisp_outputs[var_idx] = mean_of_max; break;
                    case DEFUZZ_LOM: engine->crisp_outputs[var_idx] = max_last; break;
                    case DEFUZZ_ROM: engine->crisp_outputs[var_idx] = max_first; break;
                    default: break;
                    }
                }
            }
        }
    } else {
        /* Sugeno推理：加权平均 */
        int max_out_vars = engine->var_count;
        if (!engine->crisp_outputs) {
            engine->crisp_outputs = (float*)safe_calloc(max_out_vars, sizeof(float));
            engine->output_count = max_out_vars;
        }

        float* weight_sum = (float*)safe_calloc(max_out_vars, sizeof(float));
        if (!weight_sum) return -1;

        for (int r = 0; r < engine->rule_count; r++) {
            FuzzyRule* rule = &engine->rules[r];
            if (!rule->is_active || engine->rule_activations[r] < 1e-8f) continue;

            int var_idx = fuzzy_find_var(engine, rule->conclusion_var);
            if (var_idx < 0) continue;

            FuzzyVariable* var = &engine->variables[var_idx];
            int term_idx = fuzzy_find_term(var, rule->conclusion_term);
            if (term_idx < 0) continue;

            MembershipFunction* conc_mf = &var->terms[term_idx];
            float activation = engine->rule_activations[r];

            /* Sugeno输出：取隶属度函数的中心值 */
            float center = (conc_mf->params[0] + conc_mf->params[2]) * 0.5f;
            if (conc_mf->type == MF_GAUSSIAN) center = conc_mf->params[1];

            engine->crisp_outputs[var_idx] += activation * center;
            weight_sum[var_idx] += activation;
        }

        for (int i = 0; i < max_out_vars; i++) {
            if (weight_sum[i] > 1e-10f) {
                engine->crisp_outputs[i] /= weight_sum[i];
            }
        }

        safe_free((void**)&weight_sum);
    }

    return 0;
}

int fuzzy_engine_get_output(FuzzyEngine* engine, const char* var_name, float* output) {
    if (!engine || !var_name || !output) return -1;

    int idx = fuzzy_find_var(engine, var_name);
    if (idx < 0 || idx >= engine->output_count) return -1;

    *output = engine->crisp_outputs[idx];
    return 0;
}

void fuzzy_engine_reset(FuzzyEngine* engine) {
    if (!engine) return;

    if (engine->crisp_inputs) {
        memset(engine->crisp_inputs, 0, engine->input_count * sizeof(float));
    }
    if (engine->crisp_outputs) {
        memset(engine->crisp_outputs, 0, engine->output_count * sizeof(float));
    }
    if (engine->rule_activations) {
        memset(engine->rule_activations, 0, engine->rule_count * sizeof(float));
    }
}

int fuzzy_engine_get_stats(FuzzyEngine* engine, int* var_count, int* rule_count) {
    if (!engine) return -1;
    if (var_count) *var_count = engine->var_count;
    if (rule_count) *rule_count = engine->rule_count;
    return 0;
}

/* ============================================================================
 * 概率逻辑推理引擎
 * =========================================================================== */

struct ProbEngine {
    BayesianNetwork network;
    int* evidence;       /* -1表示未观测，否则为观测值索引 */
    int evidence_count;
};

/* 查找变量索引 */
static int prob_find_var(ProbEngine* engine, const char* name) {
    for (int i = 0; i < engine->network.var_count; i++) {
        if (strcmp(engine->network.variables[i].name, name) == 0) return i;
    }
    return -1;
}

/* 查找CPT */
static int prob_find_cpt(ProbEngine* engine, const char* name) {
    for (int i = 0; i < engine->network.cpt_count; i++) {
        if (strcmp(engine->network.cpts[i].var_name, name) == 0) return i;
    }
    return -1;
}

/* 获取变量的值数量 */
static int prob_var_value_count(ProbVariable* var) {
    if (var->type == PROB_VAR_BOOLEAN) return 2;
    if (var->type == PROB_VAR_DISCRETE) return var->value_count;
    return 1;
}

/* 变量消元：对因子列表进行求和消元（支持共享变量优化） */
static float* prob_factor_multiply(float** factors, int* factor_sizes,
                                    int* factor_vars, int* factor_var_counts,
                                    int factor_count, int* out_size) {
    if (factor_count == 0) { *out_size = 0; return NULL; }
    if (factor_count == 1) {
        *out_size = factor_sizes[0];
        float* result = (float*)safe_malloc(factor_sizes[0] * sizeof(float));
        if (result) memcpy(result, factors[0], factor_sizes[0] * sizeof(float));
        return result;
    }

    /* 使用共享变量信息优化因子乘法：
     * 如果两个因子有共享变量，只对非共享维度进行外积 */
    int shared_vars = 0;
    if (factor_vars && factor_var_counts && factor_count >= 2) {
        int sz_a = factor_var_counts[0];
        int sz_b = factor_var_counts[1];
        for (int va = 0; va < sz_a && !shared_vars; va++) {
            for (int vb = 0; vb < sz_b; vb++) {
                if (factor_vars[va] == factor_vars[va + sz_a > 0 ? factor_var_counts[factor_count-1] : 0]) {
                    shared_vars = 1; break;
                }
            }
        }
    }

    float* cur = (float*)safe_malloc(factor_sizes[0] * sizeof(float));
    if (!cur) { *out_size = 0; return NULL; }
    memcpy(cur, factors[0], factor_sizes[0] * sizeof(float));
    int cur_size = factor_sizes[0];

    for (int f = 1; f < factor_count; f++) {
        int new_size = cur_size * factor_sizes[f];
        /* 如果有共享变量，可以压缩乘法（仅对独立维度做外积） */
        if (shared_vars && f == 1) {
            int effective = factor_sizes[f] > 2 ? factor_sizes[f] / 2 : factor_sizes[f];
            new_size = cur_size * effective;
        }
        float* new_cur = (float*)safe_malloc(new_size * sizeof(float));
        if (!new_cur) { safe_free((void**)&cur); *out_size = 0; return NULL; }

        for (int i = 0; i < cur_size; i++) {
            for (int j = 0; j < factor_sizes[f] && (i * factor_sizes[f] + j) < new_size; j++) {
                new_cur[i * factor_sizes[f] + j] = cur[i] * factors[f][j];
            }
        }
        safe_free((void**)&cur);
        cur = new_cur;
        cur_size = new_size;
    }

    *out_size = cur_size;
    return cur;
}

/* 获取CPT条目中对应父节点赋值的概率 */
static float prob_get_cpt_prob(CPTEntry* entry, int child_value) {
    if (!entry || child_value < 0 || child_value >= entry->prob_count) return 0.0f;
    return entry->probs[child_value];
}

/* DFS辅助函数（通过参数传递状态，避免嵌套函数） */
static void prob_dfs_visit(BayesianNetwork* net, int v, int* visited,
                           int* order, int* order_idx) {
    visited[v] = 1;
    for (int u = 0; u < net->var_count; u++) {
        if (net->adj_matrix[v * net->adj_capacity + u] && !visited[u]) {
            prob_dfs_visit(net, u, visited, order, order_idx);
        }
    }
    order[(*order_idx)++] = v;
}

/* 构建所有变量的拓扑顺序（DFS） */
static int* prob_topological_sort(BayesianNetwork* net, int* out_count) {
    int n = net->var_count;
    int* visited = (int*)safe_calloc(n, sizeof(int));
    int* order = (int*)safe_malloc(n * sizeof(int));
    int order_idx = 0;

    if (!visited || !order) {
        safe_free((void**)&visited);
        safe_free((void**)&order);
        *out_count = 0;
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            prob_dfs_visit(net, i, visited, order, &order_idx);
        }
    }

    safe_free((void**)&visited);
    *out_count = order_idx;
    return order;
}

/* K-007修复：使用加密安全随机数替代rand() */
static float prob_random(void) {
    return secure_random_float();
}

/* 根据概率分布采样 */
static int prob_sample_from_dist(float* probs, int count) {
    float r = prob_random();
    float cum = 0.0f;
    for (int i = 0; i < count; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return count - 1;
}

ProbEngine* prob_engine_create(void) {
    ProbEngine* engine = (ProbEngine*)safe_calloc(1, sizeof(ProbEngine));
    if (!engine) return NULL;

    engine->network.variables = NULL;
    engine->network.var_count = 0;
    engine->network.var_capacity = 0;
    engine->network.cpts = NULL;
    engine->network.cpt_count = 0;
    engine->network.adj_matrix = NULL;
    engine->network.adj_capacity = 0;
    engine->evidence = NULL;
    engine->evidence_count = 0;

    return engine;
}

void prob_engine_free(ProbEngine* engine) {
    if (!engine) return;

    for (int i = 0; i < engine->network.var_count; i++) {
        ProbVariable* v = &engine->network.variables[i];
        if (v->value_names) {
            for (int j = 0; j < v->value_count; j++) {
                if (v->value_names[j]) safe_free((void**)&v->value_names[j]);
            }
            safe_free((void**)&v->value_names);
        }
        if (v->prior_probs) safe_free((void**)&v->prior_probs);
    }
    safe_free((void**)&engine->network.variables);

    for (int i = 0; i < engine->network.cpt_count; i++) {
        ConditionalProbabilityTable* cpt = &engine->network.cpts[i];
        if (cpt->parent_names) {
            for (int j = 0; j < cpt->parent_count; j++) {
                if (cpt->parent_names[j]) safe_free((void**)&cpt->parent_names[j]);
            }
            safe_free((void**)&cpt->parent_names);
        }
        if (cpt->entries) {
            for (int j = 0; j < cpt->entry_count; j++) {
                if (cpt->entries[j].probs) safe_free((void**)&cpt->entries[j].probs);
            }
            safe_free((void**)&cpt->entries);
        }
    }
    safe_free((void**)&engine->network.cpts);
    if (engine->network.adj_matrix) safe_free((void**)&engine->network.adj_matrix);
    if (engine->evidence) safe_free((void**)&engine->evidence);

    safe_free((void**)&engine);
}

int prob_engine_load_network(ProbEngine* engine, const BayesianNetwork* network) {
    if (!engine || !network) return -1;

    /* 释放旧网络 */
    prob_engine_free(engine);

    /* 重新初始化 */
    engine->network.var_count = network->var_count;
    engine->network.var_capacity = network->var_count;
    engine->network.cpt_count = network->cpt_count;

    /* 复制变量 */
    engine->network.variables = (ProbVariable*)safe_calloc(
        engine->network.var_count, sizeof(ProbVariable));
    if (!engine->network.variables) return -1;

    for (int i = 0; i < engine->network.var_count; i++) {
        ProbVariable* dst = &engine->network.variables[i];
        memcpy(dst, &network->variables[i], sizeof(ProbVariable));

        if (network->variables[i].value_count > 0) {
            dst->value_names = (char**)safe_calloc(
                network->variables[i].value_count, sizeof(char*));
            if (dst->value_names) {
                for (int j = 0; j < network->variables[i].value_count; j++) {
                    if (network->variables[i].value_names[j]) {
                        dst->value_names[j] = ur_dup_str(network->variables[i].value_names[j]);
                    }
                }
            }
        }

        if (network->variables[i].prior_probs) {
            int vc = prob_var_value_count(&network->variables[i]);
            dst->prior_probs = (float*)safe_malloc(vc * sizeof(float));
            if (dst->prior_probs) {
                memcpy(dst->prior_probs, network->variables[i].prior_probs, vc * sizeof(float));
            }
        }
    }

    /* 复制CPT */
    engine->network.cpts = (ConditionalProbabilityTable*)safe_calloc(
        engine->network.cpt_count, sizeof(ConditionalProbabilityTable));
    if (!engine->network.cpts) return -1;

    for (int i = 0; i < engine->network.cpt_count; i++) {
        ConditionalProbabilityTable* dst = &engine->network.cpts[i];
        memcpy(dst, &network->cpts[i], sizeof(ConditionalProbabilityTable));

        if (network->cpts[i].parent_count > 0) {
            dst->parent_names = (char**)safe_calloc(
                network->cpts[i].parent_count, sizeof(char*));
            if (dst->parent_names) {
                for (int j = 0; j < network->cpts[i].parent_count; j++) {
                    if (network->cpts[i].parent_names[j]) {
                        dst->parent_names[j] = ur_dup_str(network->cpts[i].parent_names[j]);
                    }
                }
            }
        }

        if (network->cpts[i].entry_count > 0) {
            dst->entries = (CPTEntry*)safe_calloc(
                network->cpts[i].entry_count, sizeof(CPTEntry));
            if (dst->entries) {
                for (int j = 0; j < network->cpts[i].entry_count; j++) {
                    CPTEntry* edst = &dst->entries[j];
                    memcpy(edst, &network->cpts[i].entries[j], sizeof(CPTEntry));
                    if (network->cpts[i].entries[j].prob_count > 0) {
                        edst->probs = (float*)safe_malloc(
                            network->cpts[i].entries[j].prob_count * sizeof(float));
                        if (edst->probs) {
                            memcpy(edst->probs, network->cpts[i].entries[j].probs,
                                   network->cpts[i].entries[j].prob_count * sizeof(float));
                        }
                    }
                }
            }
        }
    }

    /* 复制邻接矩阵 */
    int adj_size = network->var_count * network->var_count;
    engine->network.adj_capacity = network->var_count;
    engine->network.adj_matrix = (int*)safe_calloc(adj_size, sizeof(int));
    if (engine->network.adj_matrix && network->adj_matrix) {
        memcpy(engine->network.adj_matrix, network->adj_matrix, adj_size * sizeof(int));
    }

    /* 初始化证据数组 */
    engine->evidence_count = engine->network.var_count;
    engine->evidence = (int*)safe_malloc(engine->evidence_count * sizeof(int));
    if (engine->evidence) {
        for (int i = 0; i < engine->evidence_count; i++) {
            engine->evidence[i] = -1;
        }
    }

    return 0;
}

int prob_engine_set_evidence(ProbEngine* engine, const char* var_name, int value_index) {
    if (!engine || !var_name) return -1;

    int idx = prob_find_var(engine, var_name);
    if (idx < 0) return -1;

    if (idx >= engine->evidence_count) return -1;
    engine->evidence[idx] = value_index;
    return 0;
}

int prob_engine_infer(ProbEngine* engine, const char* query_var, float* result_probs, int max_probs) {
    if (!engine || !query_var || !result_probs || max_probs <= 0) return -1;

    int qidx = prob_find_var(engine, query_var);
    if (qidx < 0) return -1;

    int qvc = prob_var_value_count(&engine->network.variables[qidx]);
    if (qvc > max_probs) qvc = max_probs;

    /* 变量消元算法（枚举法：对所有变量组合进行全概率求和） */
    int n = engine->network.var_count;
    int* var_sizes = (int*)safe_malloc(n * sizeof(int));
    int total_combinations = 1;

    for (int i = 0; i < n; i++) {
        var_sizes[i] = prob_var_value_count(&engine->network.variables[i]);
        if (engine->evidence[i] >= 0) var_sizes[i] = 1;
        total_combinations *= var_sizes[i];
    }

    if (total_combinations > 1048576) {
        safe_free((void**)&var_sizes);
        return prob_engine_likelihood_weighting(engine, query_var, 10000, result_probs, max_probs);
    }

    /* 使用迭代计数器枚举所有变量组合，替代递归嵌套函数 */
    int* assignment = (int*)safe_calloc(n, sizeof(int));
    float* joint_probs = (float*)safe_calloc(total_combinations, sizeof(float));
    float Z = 0.0f;
    int combo_idx = 0;

    /* 初始化赋值：证据变量固定，非证据变量从0开始 */
    for (int i = 0; i < n; i++) {
        if (engine->evidence[i] >= 0) {
            assignment[i] = engine->evidence[i];
        } else {
            assignment[i] = 0;
        }
    }

    /* 迭代枚举所有组合 */
    while (1) {
        float prob = 1.0f;
        for (int i = 0; i < n; i++) {
            int cpt_idx = prob_find_cpt(engine, engine->network.variables[i].name);
            if (cpt_idx >= 0) {
                ConditionalProbabilityTable* cpt = &engine->network.cpts[cpt_idx];
                int entry_idx = 0;
                int stride = 1;
                for (int p = cpt->parent_count - 1; p >= 0; p--) {
                    int pidx = prob_find_var(engine, cpt->parent_names[p]);
                    if (pidx >= 0) {
                        entry_idx += assignment[pidx] * stride;
                        stride *= prob_var_value_count(&engine->network.variables[pidx]);
                    }
                }
                if (entry_idx < cpt->entry_count) {
                    prob *= prob_get_cpt_prob(&cpt->entries[entry_idx], assignment[i]);
                }
            } else {
                if (engine->network.variables[i].prior_probs) {
                    prob *= engine->network.variables[i].prior_probs[assignment[i]];
                }
            }
        }
        joint_probs[combo_idx] = prob;
        Z += prob;
        combo_idx++;

        /* 递增到下一个组合 */
        int carry = 1;
        for (int i = n - 1; i >= 0 && carry; i--) {
            if (engine->evidence[i] >= 0) continue; /* 证据变量跳过 */
            assignment[i]++;
            if (assignment[i] >= prob_var_value_count(&engine->network.variables[i])) {
                assignment[i] = 0;
            } else {
                carry = 0;
            }
        }
        if (carry) break;
    }

    /* 第二次遍历：计算查询变量的边缘后验分布 */
    for (int i = 0; i < n; i++) {
        if (engine->evidence[i] >= 0) {
            var_sizes[i] = 1;
        } else {
            var_sizes[i] = prob_var_value_count(&engine->network.variables[i]);
        }
    }

    memset(result_probs, 0, qvc * sizeof(float));

    /* 重置赋值 */
    for (int i = 0; i < n; i++) {
        if (engine->evidence[i] >= 0) {
            assignment[i] = engine->evidence[i];
        } else {
            assignment[i] = 0;
        }
    }

    combo_idx = 0;
    while (1) {
        if (Z > 1e-20f) {
            result_probs[assignment[qidx]] += joint_probs[combo_idx] / Z;
        }
        combo_idx++;

        int carry = 1;
        for (int i = n - 1; i >= 0 && carry; i--) {
            if (engine->evidence[i] >= 0) continue;
            assignment[i]++;
            if (assignment[i] >= prob_var_value_count(&engine->network.variables[i])) {
                assignment[i] = 0;
            } else {
                carry = 0;
            }
        }
        if (carry) break;
    }

    safe_free((void**)&var_sizes);
    safe_free((void**)&assignment);
    safe_free((void**)&joint_probs);

    return qvc;
}

int prob_engine_rejection_sampling(ProbEngine* engine, const char* query_var,
                                    int sample_count, float* result_probs, int max_probs) {
    if (!engine || !query_var || !result_probs || sample_count <= 0 || max_probs <= 0) return -1;

    int qidx = prob_find_var(engine, query_var);
    if (qidx < 0) return -1;

    int qvc = prob_var_value_count(&engine->network.variables[qidx]);
    if (qvc > max_probs) qvc = max_probs;

    /* 拓扑排序 */
    int topo_count = 0;
    int* topo = prob_topological_sort(&engine->network, &topo_count);
    if (!topo) {
        /* 拓扑排序失败——回退到顺序节点遍历（降级）
         * 顺序遍历使采样质量下降但保证系统可继续运行 */
        log_debug("[不确定性推理] prob_topological_sort返回NULL，使用顺序节点遍历回退");
        topo = (int*)safe_malloc(engine->network.var_count * sizeof(int));
        if (!topo) return -1;
        for (int i = 0; i < engine->network.var_count; i++) topo[i] = i;
        topo_count = engine->network.var_count;
    }

    memset(result_probs, 0, qvc * sizeof(float));
    int accepted = 0;
    int n = engine->network.var_count;

    for (int s = 0; s < sample_count; s++) {
        int* sample = (int*)safe_malloc(n * sizeof(int));
        if (!sample) { safe_free((void**)&topo); return -1; }

        for (int ti = 0; ti < topo_count; ti++) {
            int v = topo[ti];
            int vc = prob_var_value_count(&engine->network.variables[v]);

            if (engine->evidence[v] >= 0) {
                sample[v] = engine->evidence[v];
                continue;
            }

            int cpt_idx = prob_find_cpt(engine, engine->network.variables[v].name);
            if (cpt_idx >= 0) {
                ConditionalProbabilityTable* cpt = &engine->network.cpts[cpt_idx];
                int entry_idx = 0;
                int stride = 1;
                for (int p = cpt->parent_count - 1; p >= 0; p--) {
                    int pidx = prob_find_var(engine, cpt->parent_names[p]);
                    if (pidx >= 0) {
                        entry_idx += sample[pidx] * stride;
                        stride *= prob_var_value_count(&engine->network.variables[pidx]);
                    }
                }
                if (entry_idx < cpt->entry_count) {
                    sample[v] = prob_sample_from_dist(cpt->entries[entry_idx].probs, vc);
                } else {
                    sample[v] = 0;
                }
            } else {
                if (engine->network.variables[v].prior_probs) {
                    sample[v] = prob_sample_from_dist(engine->network.variables[v].prior_probs, vc);
                } else {
                    sample[v] = 0;
                }
            }
        }

        /* 拒绝检查：证据是否匹配 */
        int consistent = 1;
        for (int i = 0; i < n; i++) {
            if (engine->evidence[i] >= 0 && sample[i] != engine->evidence[i]) {
                consistent = 0;
                break;
            }
        }

        if (consistent) {
            result_probs[sample[qidx]] += 1.0f;
            accepted++;
        }

        safe_free((void**)&sample);
    }

    if (accepted > 0) {
        for (int i = 0; i < qvc; i++) {
            result_probs[i] /= (float)accepted;
        }
    }

    safe_free((void**)&topo);
    return qvc;
}

int prob_engine_likelihood_weighting(ProbEngine* engine, const char* query_var,
                                      int sample_count, float* result_probs, int max_probs) {
    if (!engine || !query_var || !result_probs || sample_count <= 0 || max_probs <= 0) return -1;

    int qidx = prob_find_var(engine, query_var);
    if (qidx < 0) return -1;

    int qvc = prob_var_value_count(&engine->network.variables[qidx]);
    if (qvc > max_probs) qvc = max_probs;

    int topo_count = 0;
    int* topo = prob_topological_sort(&engine->network, &topo_count);
    if (!topo) {
        topo = (int*)safe_malloc(engine->network.var_count * sizeof(int));
        if (!topo) return -1;
        for (int i = 0; i < engine->network.var_count; i++) topo[i] = i;
        topo_count = engine->network.var_count;
    }

    memset(result_probs, 0, qvc * sizeof(float));
    float weight_sum = 0.0f;
    int n = engine->network.var_count;

    for (int s = 0; s < sample_count; s++) {
        int* sample = (int*)safe_malloc(n * sizeof(int));
        if (!sample) { safe_free((void**)&topo); return -1; }

        float weight = 1.0f;

        for (int ti = 0; ti < topo_count; ti++) {
            int v = topo[ti];
            int vc = prob_var_value_count(&engine->network.variables[v]);

            int cpt_idx = prob_find_cpt(engine, engine->network.variables[v].name);
            float* dist = NULL;
            int dist_size = vc;

            if (cpt_idx >= 0) {
                ConditionalProbabilityTable* cpt = &engine->network.cpts[cpt_idx];
                int entry_idx = 0;
                int stride = 1;
                for (int p = cpt->parent_count - 1; p >= 0; p--) {
                    int pidx = prob_find_var(engine, cpt->parent_names[p]);
                    if (pidx >= 0) {
                        entry_idx += sample[pidx] * stride;
                        stride *= prob_var_value_count(&engine->network.variables[pidx]);
                    }
                }
                if (entry_idx < cpt->entry_count) {
                    dist = cpt->entries[entry_idx].probs;
                    dist_size = cpt->entries[entry_idx].prob_count;
                }
            } else {
                dist = engine->network.variables[v].prior_probs;
            }

            if (engine->evidence[v] >= 0) {
                sample[v] = engine->evidence[v];
                if (dist && engine->evidence[v] < dist_size) {
                    weight *= dist[engine->evidence[v]];
                }
            } else {
                if (dist) {
                    sample[v] = prob_sample_from_dist(dist, dist_size);
                } else {
                    sample[v] = 0;
                }
            }
        }

        result_probs[sample[qidx]] += weight;
        weight_sum += weight;
        safe_free((void**)&sample);
    }

    if (weight_sum > 1e-20f) {
        for (int i = 0; i < qvc; i++) {
            result_probs[i] /= weight_sum;
        }
    }

    safe_free((void**)&topo);
    return qvc;
}

void prob_engine_reset(ProbEngine* engine) {
    if (!engine || !engine->evidence) return;
    for (int i = 0; i < engine->evidence_count; i++) {
        engine->evidence[i] = -1;
    }
}

int prob_engine_get_stats(ProbEngine* engine, int* var_count, int* cpt_count) {
    if (!engine) return -1;
    if (var_count) *var_count = engine->network.var_count;
    if (cpt_count) *cpt_count = engine->network.cpt_count;
    return 0;
}

/* ============================================================================
 * Dempster-Shafer 证据理论引擎
 * =========================================================================== */

struct DSEngine {
    DSFrame frame;
    DSBodyOfEvidence* evidence_list;
    int evidence_count;
    int evidence_capacity;
};

DSEngine* ds_engine_create(void) {
    DSEngine* engine = (DSEngine*)safe_calloc(1, sizeof(DSEngine));
    if (!engine) return NULL;

    engine->frame.element_names = NULL;
    engine->frame.element_count = 0;
    engine->evidence_list = NULL;
    engine->evidence_count = 0;
    engine->evidence_capacity = 16;

    engine->evidence_list = (DSBodyOfEvidence*)safe_calloc(
        engine->evidence_capacity, sizeof(DSBodyOfEvidence));

    if (!engine->evidence_list) {
        safe_free((void**)&engine);
        return NULL;
    }

    return engine;
}

void ds_engine_free(DSEngine* engine) {
    if (!engine) return;

    if (engine->frame.element_names) {
        for (int i = 0; i < engine->frame.element_count; i++) {
            if (engine->frame.element_names[i]) safe_free((void**)&engine->frame.element_names[i]);
        }
        safe_free((void**)&engine->frame.element_names);
    }

    for (int i = 0; i < engine->evidence_count; i++) {
        DSBodyOfEvidence* ev = &engine->evidence_list[i];
        if (ev->focal_elements) {
            safe_free((void**)&ev->focal_elements);
        }
        if (ev->frame && ev->frame != &engine->frame) {
            /* 外部框架不需要释放 */
        }
    }
    safe_free((void**)&engine->evidence_list);
    safe_free((void**)&engine);
}

int ds_engine_set_frame(DSEngine* engine, const DSFrame* frame) {
    if (!engine || !frame) return -1;

    if (engine->frame.element_names) {
        for (int i = 0; i < engine->frame.element_count; i++) {
            if (engine->frame.element_names[i]) safe_free((void**)&engine->frame.element_names[i]);
        }
        safe_free((void**)&engine->frame.element_names);
    }

    engine->frame.element_count = frame->element_count;
    engine->frame.element_names = (char**)safe_calloc(frame->element_count, sizeof(char*));
    if (!engine->frame.element_names) return -1;

    for (int i = 0; i < frame->element_count; i++) {
        engine->frame.element_names[i] = ur_dup_str(frame->element_names[i]);
    }

    strncpy(engine->frame.name, frame->name, UR_MAX_TERM_LEN - 1);
    engine->frame.name[UR_MAX_TERM_LEN - 1] = '\0';

    return 0;
}

int ds_engine_add_evidence(DSEngine* engine, const DSBodyOfEvidence* evidence) {
    if (!engine || !evidence) return -1;

    if (engine->evidence_count >= engine->evidence_capacity) {
        int new_cap = engine->evidence_capacity * 2;
        DSBodyOfEvidence* new_list = (DSBodyOfEvidence*)safe_realloc(
            engine->evidence_list, new_cap * sizeof(DSBodyOfEvidence));
        if (!new_list) return -1;
        engine->evidence_list = new_list;
        engine->evidence_capacity = new_cap;
    }

    DSBodyOfEvidence* dst = &engine->evidence_list[engine->evidence_count];
    dst->frame = &engine->frame;
    dst->focal_count = evidence->focal_count;
    dst->focal_capacity = evidence->focal_count;
    dst->total_mass = evidence->total_mass;

    dst->focal_elements = (DSElement*)safe_malloc(
        evidence->focal_count * sizeof(DSElement));
    if (!dst->focal_elements) return -1;

    memcpy(dst->focal_elements, evidence->focal_elements,
           evidence->focal_count * sizeof(DSElement));

    engine->evidence_count++;
    return engine->evidence_count - 1;
}

int ds_combine_two(const DSBodyOfEvidence* ev1, const DSBodyOfEvidence* ev2,
                   DSBodyOfEvidence* result) {
    if (!ev1 || !ev2 || !result) return -1;

    int max_focal = ev1->focal_count * ev2->focal_count;
    DSElement* combined = (DSElement*)safe_calloc(max_focal, sizeof(DSElement));
    if (!combined) return -1;

    float K = 0.0f;
    int combined_count = 0;

    for (int i = 0; i < ev1->focal_count; i++) {
        for (int j = 0; j < ev2->focal_count; j++) {
            int intersection = ev1->focal_elements[i].element_mask &
                               ev2->focal_elements[j].element_mask;
            float product = ev1->focal_elements[i].mass *
                            ev2->focal_elements[j].mass;

            if (intersection == 0) {
                K += product;
            } else {
                int found = -1;
                for (int k = 0; k < combined_count; k++) {
                    if (combined[k].element_mask == intersection) {
                        found = k;
                        break;
                    }
                }

                if (found >= 0) {
                    combined[found].mass += product;
                } else {
                    combined[combined_count].element_mask = intersection;
                    combined[combined_count].mass = product;
                    snprintf(combined[combined_count].description, sizeof(combined[combined_count].description),
                             "组合焦元 0x%X", intersection);
                    combined_count++;
                }
            }
        }
    }

    float normalization = 1.0f - K;
    if (normalization < 1e-20f) {
        safe_free((void**)&combined);
        return -1;
    }

    for (int i = 0; i < combined_count; i++) {
        combined[i].mass /= normalization;
    }

    result->focal_elements = combined;
    result->focal_count = combined_count;
    result->focal_capacity = max_focal;
    result->frame = ev1->frame;

    result->total_mass = 0.0f;
    for (int i = 0; i < combined_count; i++) {
        result->total_mass += combined[i].mass;
    }

    /* 计算信度和似真度 */
    for (int i = 0; i < combined_count; i++) {
        int mask = combined[i].element_mask;
        combined[i].belief = ds_belief(mask, result);
        combined[i].plausibility = ds_plausibility(mask, result);
    }

    return 0;
}

int ds_combine_yager(const DSBodyOfEvidence* ev1, const DSBodyOfEvidence* ev2,
                     DSBodyOfEvidence* result) {
    if (!ev1 || !ev2 || !result) return -1;

    int max_focal = ev1->focal_count * ev2->focal_count + 1;
    DSElement* combined = (DSElement*)safe_calloc(max_focal, sizeof(DSElement));
    if (!combined) return -1;

    float K = 0.0f;
    int combined_count = 0;
    int full_mask = 0;

    if (ev1->frame) {
        for (int e = 0; e < ev1->frame->element_count; e++) {
            full_mask |= (1 << e);
        }
    }

    for (int i = 0; i < ev1->focal_count; i++) {
        for (int j = 0; j < ev2->focal_count; j++) {
            int intersection = ev1->focal_elements[i].element_mask &
                               ev2->focal_elements[j].element_mask;
            float product = ev1->focal_elements[i].mass *
                            ev2->focal_elements[j].mass;

            if (intersection == 0) {
                K += product;
            } else {
                int found = -1;
                for (int k = 0; k < combined_count; k++) {
                    if (combined[k].element_mask == intersection) {
                        found = k;
                        break;
                    }
                }
                if (found >= 0) {
                    combined[found].mass += product;
                } else {
                    combined[combined_count].element_mask = intersection;
                    combined[combined_count].mass = product;
                    combined_count++;
                }
            }
        }
    }

    /* Yager：将冲突分配给全集（未知） */
    if (K > 1e-20f && full_mask > 0) {
        int found = -1;
        for (int k = 0; k < combined_count; k++) {
            if (combined[k].element_mask == full_mask) {
                found = k;
                break;
            }
        }
        if (found >= 0) {
            combined[found].mass += K;
        } else {
            combined[combined_count].element_mask = full_mask;
            combined[combined_count].mass = K;
            snprintf(combined[combined_count].description, sizeof(combined[combined_count].description),
                     "Yager全集(冲突=%.4f)", K);
            combined_count++;
        }
    }

    result->focal_elements = combined;
    result->focal_count = combined_count;
    result->focal_capacity = max_focal;
    result->frame = ev1->frame;

    result->total_mass = 0.0f;
    for (int i = 0; i < combined_count; i++) {
        result->total_mass += combined[i].mass;
    }

    for (int i = 0; i < combined_count; i++) {
        int mask = combined[i].element_mask;
        combined[i].belief = ds_belief(mask, result);
        combined[i].plausibility = ds_plausibility(mask, result);
    }

    return 0;
}

float ds_conflict_coefficient(const DSBodyOfEvidence* ev1, const DSBodyOfEvidence* ev2) {
    if (!ev1 || !ev2) return 0.0f;

    float K = 0.0f;
    for (int i = 0; i < ev1->focal_count; i++) {
        for (int j = 0; j < ev2->focal_count; j++) {
            if ((ev1->focal_elements[i].element_mask &
                 ev2->focal_elements[j].element_mask) == 0) {
                K += ev1->focal_elements[i].mass * ev2->focal_elements[j].mass;
            }
        }
    }
    return K;
}

float ds_belief(int element_mask, const DSBodyOfEvidence* evidence) {
    if (!evidence || !evidence->focal_elements) return 0.0f;

    float bel = 0.0f;
    for (int i = 0; i < evidence->focal_count; i++) {
        if ((evidence->focal_elements[i].element_mask & element_mask) ==
            evidence->focal_elements[i].element_mask) {
            bel += evidence->focal_elements[i].mass;
        }
    }
    return bel;
}

float ds_plausibility(int element_mask, const DSBodyOfEvidence* evidence) {
    if (!evidence || !evidence->focal_elements) return 0.0f;

    float pl = 0.0f;
    for (int i = 0; i < evidence->focal_count; i++) {
        if ((evidence->focal_elements[i].element_mask & element_mask) != 0) {
            pl += evidence->focal_elements[i].mass;
        }
    }
    return pl;
}

int ds_belief_interval(int element_mask, const DSBodyOfEvidence* evidence,
                       float* belief, float* plausibility) {
    if (!evidence || !belief || !plausibility) return -1;

    *belief = ds_belief(element_mask, evidence);
    *plausibility = ds_plausibility(element_mask, evidence);
    return 0;
}

int ds_decide(const DSBodyOfEvidence* evidence, char* decision, size_t decision_size) {
    if (!evidence || !decision || decision_size == 0) return -1;

    int best_idx = -1;
    float best_belpl = -1.0f;

    for (int i = 0; i < evidence->focal_count; i++) {
        float bel = evidence->focal_elements[i].belief;
        float pl = evidence->focal_elements[i].plausibility;
        float avg = (bel + pl) * 0.5f;

        if (avg > best_belpl) {
            best_belpl = avg;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        snprintf(decision, decision_size, "无法决策");
        return -1;
    }

    snprintf(decision, decision_size, "%s (Bel=%.4f, Pl=%.4f, 信度区间=[%.4f, %.4f])",
             evidence->focal_elements[best_idx].description,
             evidence->focal_elements[best_idx].belief,
             evidence->focal_elements[best_idx].plausibility,
             evidence->focal_elements[best_idx].belief,
             evidence->focal_elements[best_idx].plausibility);

    return 0;
}

int ds_engine_combine(DSEngine* engine, DSBodyOfEvidence* result) {
    if (!engine || !result || engine->evidence_count == 0) return -1;

    if (engine->evidence_count == 1) {
        memcpy(result, &engine->evidence_list[0], sizeof(DSBodyOfEvidence));
        result->focal_elements = (DSElement*)safe_malloc(
            engine->evidence_list[0].focal_count * sizeof(DSElement));
        if (!result->focal_elements) return -1;
        memcpy(result->focal_elements, engine->evidence_list[0].focal_elements,
               engine->evidence_list[0].focal_count * sizeof(DSElement));
        return 0;
    }

    DSBodyOfEvidence temp;
    int ret = ds_combine_two(&engine->evidence_list[0], &engine->evidence_list[1], &temp);
    if (ret != 0) return -1;

    for (int i = 2; i < engine->evidence_count; i++) {
        DSBodyOfEvidence new_temp;
        ret = ds_combine_two(&temp, &engine->evidence_list[i], &new_temp);
        if (temp.focal_elements) safe_free((void**)&temp.focal_elements);
        if (ret != 0) return -1;
        temp = new_temp;
    }

    memcpy(result, &temp, sizeof(DSBodyOfEvidence));
    return 0;
}

void ds_engine_reset(DSEngine* engine) {
    if (!engine) return;

    for (int i = 0; i < engine->evidence_count; i++) {
        DSBodyOfEvidence* ev = &engine->evidence_list[i];
        if (ev->focal_elements) safe_free((void**)&ev->focal_elements);
        ev->focal_elements = NULL;
        ev->focal_count = 0;
    }
    engine->evidence_count = 0;
}

int ds_engine_get_stats(DSEngine* engine, int* frame_size, int* evidence_count) {
    if (!engine) return -1;
    if (frame_size) *frame_size = engine->frame.element_count;
    if (evidence_count) *evidence_count = engine->evidence_count;
    return 0;
}

/* ============================================================================
 * 综合不确定性推理引擎
 * =========================================================================== */

struct UncertaintyEngine {
    URConfig config;
    FuzzyEngine* fuzzy_engine;
    ProbEngine* prob_engine;
    DSEngine* ds_engine;
    int initialized;
};

UncertaintyEngine* uncertainty_engine_create(const URConfig* config) {
    UncertaintyEngine* engine = (UncertaintyEngine*)safe_calloc(1, sizeof(UncertaintyEngine));
    if (!engine) return NULL;

    if (config) {
        memcpy(&engine->config, config, sizeof(URConfig));
    } else {
        engine->config.method = UR_METHOD_FUZZY;
        engine->config.fuzzy_method = FUZZY_INFER_MAMDANI;
        engine->config.defuzz_method = DEFUZZ_CENTROID;
        engine->config.prob_sample_count = 5000;
        engine->config.ds_conflict_threshold = 0.3f;
        engine->config.network_name[0] = '\0';
    }

    engine->fuzzy_engine = NULL;
    engine->prob_engine = NULL;
    engine->ds_engine = NULL;
    engine->initialized = 0;

    switch (engine->config.method) {
    case UR_METHOD_FUZZY:
        engine->fuzzy_engine = fuzzy_engine_create(
            engine->config.fuzzy_method, engine->config.defuzz_method);
        if (engine->fuzzy_engine) engine->initialized = 1;
        break;
    case UR_METHOD_PROBABILISTIC:
        engine->prob_engine = prob_engine_create();
        if (engine->prob_engine) engine->initialized = 1;
        break;
    case UR_METHOD_DS_EVIDENCE:
        engine->ds_engine = ds_engine_create();
        if (engine->ds_engine) engine->initialized = 1;
        break;
    case UR_METHOD_HYBRID:
        engine->fuzzy_engine = fuzzy_engine_create(FUZZY_INFER_MAMDANI, DEFUZZ_CENTROID);
        engine->prob_engine = prob_engine_create();
        engine->ds_engine = ds_engine_create();
        if (engine->fuzzy_engine || engine->prob_engine || engine->ds_engine) {
            engine->initialized = 1;
        }
        break;
    }

    return engine;
}

void uncertainty_engine_free(UncertaintyEngine* engine) {
    if (!engine) return;

    if (engine->fuzzy_engine) fuzzy_engine_free(engine->fuzzy_engine);
    if (engine->prob_engine) prob_engine_free(engine->prob_engine);
    if (engine->ds_engine) ds_engine_free(engine->ds_engine);

    safe_free((void**)&engine);
}

int uncertainty_engine_reason(UncertaintyEngine* engine, const char** facts,
                               size_t fact_count, URResult* result) {
    if (!engine || !result) return -1;
    memset(result, 0, sizeof(URResult));

    clock_t start = clock();

    switch (engine->config.method) {
    case UR_METHOD_FUZZY: {
        if (!engine->fuzzy_engine) return -1;

        for (size_t i = 0; i < fact_count; i++) {
            char var_name[UR_MAX_TERM_LEN];
            float value;
            if (sscanf(facts[i], "%63[^=]=%f", var_name, &value) == 2) {
                fuzzy_engine_set_input(engine->fuzzy_engine, var_name, value);
            }
        }

        fuzzy_engine_evaluate(engine->fuzzy_engine);
        result->fuzzy_output = 0.0f;

        for (int i = 0; i < engine->fuzzy_engine->var_count; i++) {
            float out;
            if (fuzzy_engine_get_output(engine->fuzzy_engine,
                                         engine->fuzzy_engine->variables[i].name, &out) == 0) {
                result->fuzzy_output += out;
            }
        }

        result->confidence = 0.85f;
        snprintf(result->trace, sizeof(result->trace),
                 "模糊推理: %zu个输入, %d个变量, %d条规则",
                 fact_count, engine->fuzzy_engine->var_count, engine->fuzzy_engine->rule_count);
        break;
    }
    case UR_METHOD_PROBABILISTIC: {
        if (!engine->prob_engine) return -1;

        for (size_t i = 0; i < fact_count; i++) {
            char var_name[UR_MAX_TERM_LEN];
            int val;
            if (sscanf(facts[i], "%63[^=]=%d", var_name, &val) == 2) {
                prob_engine_set_evidence(engine->prob_engine, var_name, val);
            }
        }

        if (engine->prob_engine->network.var_count > 0) {
            float probs[16];
            int pcount = prob_engine_likelihood_weighting(
                engine->prob_engine,
                engine->prob_engine->network.variables[0].name,
                engine->config.prob_sample_count,
                probs, 16);

            if (pcount > 0) {
                result->prob_distribution = (float*)safe_malloc(pcount * sizeof(float));
                if (result->prob_distribution) {
                    memcpy(result->prob_distribution, probs, pcount * sizeof(float));
                    result->prob_size = pcount;
                }
                result->confidence = probs[0];
                for (int i = 1; i < pcount; i++) {
                    if (probs[i] > result->confidence) result->confidence = probs[i];
                }
            }
        }

        snprintf(result->trace, sizeof(result->trace),
                 "概率推理: %zu个证据, %d个变量, %d个CPT",
                 fact_count, engine->prob_engine->network.var_count,
                 engine->prob_engine->network.cpt_count);
        break;
    }
    case UR_METHOD_DS_EVIDENCE: {
        if (!engine->ds_engine) return -1;

        DSBodyOfEvidence combined;
        memset(&combined, 0, sizeof(DSBodyOfEvidence));
        int ret = ds_engine_combine(engine->ds_engine, &combined);
        if (ret == 0) {
            result->ds_result = (DSBodyOfEvidence*)safe_malloc(sizeof(DSBodyOfEvidence));
            if (result->ds_result) {
                memcpy(result->ds_result, &combined, sizeof(DSBodyOfEvidence));
            }

            result->confidence = 0.0f;
            for (int i = 0; i < combined.focal_count; i++) {
                float avg = (combined.focal_elements[i].belief +
                             combined.focal_elements[i].plausibility) * 0.5f;
                if (avg > result->confidence) result->confidence = avg;
            }
        }

        snprintf(result->trace, sizeof(result->trace),
                 "DS证据推理: %d个证据体, %d个焦元",
                 engine->ds_engine->evidence_count,
                 engine->ds_engine->evidence_count > 0 ?
                 engine->ds_engine->evidence_list[0].focal_count : 0);
        break;
    }
    case UR_METHOD_HYBRID: {
        result->confidence = 0.0f;
        float fuzzy_conf = 0.0f;
        int fuzzy_valid = 0;

        /* 模糊推理部分 */
        if (engine->fuzzy_engine) {
            for (size_t i = 0; i < fact_count; i++) {
                char var_name[UR_MAX_TERM_LEN];
                float value;
                if (sscanf(facts[i], "%63[^=]=%f", var_name, &value) == 2) {
                    fuzzy_engine_set_input(engine->fuzzy_engine, var_name, value);
                }
            }
            fuzzy_engine_evaluate(engine->fuzzy_engine);
            for (int i = 0; i < engine->fuzzy_engine->var_count; i++) {
                float out;
                if (fuzzy_engine_get_output(engine->fuzzy_engine,
                                             engine->fuzzy_engine->variables[i].name, &out) == 0) {
                    fuzzy_conf += out;
                }
            }
            fuzzy_valid = 1;
        }

        /* 概率推理部分 */
        float prob_conf = 0.0f;
        int prob_valid = 0;
        if (engine->prob_engine && engine->prob_engine->network.var_count > 0) {
            float probs[16];
            int pcount = prob_engine_likelihood_weighting(
                engine->prob_engine,
                engine->prob_engine->network.variables[0].name,
                engine->config.prob_sample_count, probs, 16);
            if (pcount > 0) {
                prob_conf = probs[0];
                for (int i = 1; i < pcount; i++) {
                    if (probs[i] > prob_conf) prob_conf = probs[i];
                }
                prob_valid = 1;
            }
        }

        /* DS证据推理部分 */
        float ds_conf = 0.0f;
        int ds_valid = 0;
        if (engine->ds_engine && engine->ds_engine->evidence_count > 0) {
            DSBodyOfEvidence combined;
            memset(&combined, 0, sizeof(DSBodyOfEvidence));
            if (ds_engine_combine(engine->ds_engine, &combined) == 0) {
                for (int i = 0; i < combined.focal_count; i++) {
                    float avg = (combined.focal_elements[i].belief +
                                 combined.focal_elements[i].plausibility) * 0.5f;
                    if (avg > ds_conf) ds_conf = avg;
                }
                ds_valid = 1;
                result->ds_result = (DSBodyOfEvidence*)safe_malloc(sizeof(DSBodyOfEvidence));
                if (result->ds_result) memcpy(result->ds_result, &combined, sizeof(DSBodyOfEvidence));
            }
        }

        /* 融合三种推理结果 */
        int valid_count = fuzzy_valid + prob_valid + ds_valid;
        if (valid_count > 0) {
            result->confidence = (fuzzy_conf + prob_conf + ds_conf) / (float)valid_count;
            result->fuzzy_output = fuzzy_conf;
        }

        snprintf(result->trace, sizeof(result->trace),
                 "混合不确定性推理: 模糊=%s, 概率=%s, DS=%s, 综合置信度=%.4f",
                 fuzzy_valid ? "有效" : "无效",
                 prob_valid ? "有效" : "无效",
                 ds_valid ? "有效" : "无效",
                 result->confidence);
        break;
    }
    }

    result->inference_time_ms = (int)((clock() - start) * 1000 / CLOCKS_PER_SEC);
    return 0;
}

LogicInferenceResult* uncertainty_integrated_reason(UncertaintyEngine* uncertainty_engine,
                                                     ReasoningEngine* logic_engine,
                                                     const char** facts, size_t fact_count) {
    if (!uncertainty_engine || !logic_engine || !facts || fact_count == 0) return NULL;

    /* 先执行不确定性推理 */
    URResult ur_result;
    memset(&ur_result, 0, sizeof(URResult));
    uncertainty_engine_reason(uncertainty_engine, facts, fact_count, &ur_result);

    /* 构建增强事实（加入不确定性推理结果） */
    size_t enhanced_count = fact_count + 5;
    const char** enhanced_facts = (const char**)safe_malloc(enhanced_count * sizeof(char*));
    if (!enhanced_facts) {
        uncertainty_result_free(&ur_result);
        return NULL;
    }

    /* 复制原始事实指针（注意：这些是外部字符串不能释放） */
    for (size_t i = 0; i < fact_count; i++) {
        enhanced_facts[i] = facts[i];
    }

    /* 添加不确定性推理结果作为新事实 */
    char conf_str[128];
    snprintf(conf_str, sizeof(conf_str), "不确定性推理置信度=%f", ur_result.confidence);
    enhanced_facts[fact_count] = conf_str;

    char fuzzy_str[128];
    snprintf(fuzzy_str, sizeof(fuzzy_str), "模糊输出=%f", ur_result.fuzzy_output);
    enhanced_facts[fact_count + 1] = fuzzy_str;

    if (ur_result.ds_result && ur_result.ds_result->focal_count > 0) {
        char ds_str[256];
        snprintf(ds_str, sizeof(ds_str), "DS最佳焦元置信度=%f,%f",
                 ur_result.ds_result->focal_elements[0].belief,
                 ur_result.ds_result->focal_elements[0].plausibility);
        enhanced_facts[fact_count + 2] = ds_str;
    } else {
        enhanced_facts[fact_count + 2] = "DS无结果";
    }

    /* 执行逻辑推理 */
    LogicInferenceResult* logic_result = logic_reasoning_engine_forward_chain(
        logic_engine, enhanced_facts, fact_count + 3, NULL, 0);

    /* 将不确定性置信度传播到逻辑推理结果 */
    if (logic_result && ur_result.confidence > 0.0f) {
        logic_result->overall_confidence = (logic_result->overall_confidence + ur_result.confidence) * 0.5f;
    }

    safe_free((void**)&enhanced_facts);
    uncertainty_result_free(&ur_result);

    return logic_result;
}

void uncertainty_result_free(URResult* result) {
    if (!result) return;
    if (result->prob_distribution) safe_free((void**)&result->prob_distribution);
    if (result->ds_result) {
        if (result->ds_result->focal_elements) {
            safe_free((void**)&result->ds_result->focal_elements);
        }
        safe_free((void**)&result->ds_result);
    }
    memset(result, 0, sizeof(URResult));
}

void uncertainty_engine_reset(UncertaintyEngine* engine) {
    if (!engine) return;

    if (engine->fuzzy_engine) fuzzy_engine_reset(engine->fuzzy_engine);
    if (engine->prob_engine) prob_engine_reset(engine->prob_engine);
    if (engine->ds_engine) ds_engine_reset(engine->ds_engine);

    engine->initialized = 0;
    if (engine->fuzzy_engine || engine->prob_engine || engine->ds_engine) {
        engine->initialized = 1;
    }
}

/* ============================================================================
 * 马尔可夫逻辑网络 (Markov Logic Network)
 *
 * MLN将一阶逻辑与概率图模型结合：
 * - 每个公式(w, F)定义了一个势函数 exp(w · 满足F的世界数)
 * - 概率分布: P(ω) = (1/Z) · exp(Σ w_i · n_i(ω))
 *   其中n_i(ω)是公式F_i在ω中满足的真值指派数量
 * - 推理通过MCMC吉布斯采样近似计算
 * ============================================================================ */

#define MLN_MAX_FORMULAS  64
#define MLN_MAX_CONSTANTS 128
#define MLN_MAX_GROUNDED 512
#define MLN_SAMPLES       1000
#define MLN_BURNIN        200

typedef struct {
    char predicate[64];
    char args[4][32];
    int arity;
    float weight;
    int is_negated;
} MLNFormula;

typedef struct {
    int bool_values[MLN_MAX_GROUNDED];
    int num_bools;
} MLNGrounding;

/* ========== 线程安全：MLN公式计数器锁 ========== */
#ifdef _WIN32
static CRITICAL_SECTION g_mln_fc_lock;
/* K-182: volatile + InterlockedCompareExchange 消除DCL竞态 */
static volatile LONG g_mln_fc_lock_init = 0;
static void mln_fc_lock_init(void) {
    if (!g_mln_fc_lock_init) {
        if (InterlockedCompareExchange(&g_mln_fc_lock_init, 2, 0) == 0) {
            InitializeCriticalSection(&g_mln_fc_lock);
            g_mln_fc_lock_init = 1;
        } else {
            while (g_mln_fc_lock_init != 1) { Sleep(0); }
        }
    }
}
#define MLN_FC_LOCK() do { mln_fc_lock_init(); EnterCriticalSection(&g_mln_fc_lock); } while(0)
#define MLN_FC_UNLOCK() LeaveCriticalSection(&g_mln_fc_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_mln_fc_lock = PTHREAD_MUTEX_INITIALIZER;
#define MLN_FC_LOCK() pthread_mutex_lock(&g_mln_fc_lock)
#define MLN_FC_UNLOCK() pthread_mutex_unlock(&g_mln_fc_lock)
#endif

static MLNFormula mln_formulas[MLN_MAX_FORMULAS];
static int mln_formula_count = 0;

int mln_add_formula(const char* predicate, int arity,
                    const char** arg_names, float weight, int is_negated) {
    MLN_FC_LOCK();
    if (mln_formula_count >= MLN_MAX_FORMULAS) { MLN_FC_UNLOCK(); return -1; }
    MLNFormula* f = &mln_formulas[mln_formula_count++];
    MLN_FC_UNLOCK();
    strncpy(f->predicate, predicate, 63);
    f->arity = (arity > 0 && arity <= 4) ? arity : 2;
    for (int i = 0; i < f->arity && i < 4; i++) {
        if (arg_names && arg_names[i]) strncpy(f->args[i], arg_names[i], 31);
    }
    f->weight = weight;
    f->is_negated = is_negated;
    return 0;
}

static float mln_energy(const MLNGrounding* g, int n_ground) {
    float e = 0.0f;
    MLN_FC_LOCK();
    int count = mln_formula_count;
    MLN_FC_UNLOCK();
    for (int f_idx = 0; f_idx < count; f_idx++) {
        MLNFormula* f = &mln_formulas[f_idx];
        int satisfied = 0;
        for (int g_idx = 0; g_idx < n_ground; g_idx++) {
            int tv = g->bool_values[g_idx];
            if (f->is_negated) tv = !tv;
            if (tv) satisfied++;
        }
        e -= f->weight * (float)satisfied;
    }
    return e;
}

int mln_mcmc_inference(const char** evidence, int ev_count,
                        const char** query_vars, int query_count,
                        float* marginal_probs, int max_vars) {
    if (!query_vars || !marginal_probs || query_count <= 0) return -1;

    int n_vars = query_count < max_vars ? query_count : max_vars;
    MLNGrounding g;
    g.num_bools = n_vars;

    /* 初始随机赋值（K-007修复：使用安全随机数） */
    for (int i = 0; i < n_vars; i++) {
        g.bool_values[i] = (secure_random_float() < 0.5f) ? 1 : 0;
    }

    int* counts = (int*)safe_calloc((size_t)n_vars, sizeof(int));
    if (!counts) return -1;

    int total_samples = MLN_SAMPLES + MLN_BURNIN;
    for (int sample = 0; sample < total_samples; sample++) {
        /* 吉布斯采样：逐个变量更新 */
        for (int v = 0; v < n_vars; v++) {
            float e_on, e_off;

            g.bool_values[v] = 1;
            e_on = mln_energy(&g, n_vars);

            g.bool_values[v] = 0;
            e_off = mln_energy(&g, n_vars);

            /* P(v=1) = exp(-E_on) / (exp(-E_on) + exp(-E_off)) */
            float max_e = (e_on > e_off) ? e_on : e_off;
            float p_on = expf(-(e_on - max_e));
            float p_off = expf(-(e_off - max_e));
            float prob = p_on / (p_on + p_off + 1e-15f);

            /* K-007修复：使用安全随机数 */
            g.bool_values[v] = (secure_random_float() < prob) ? 1 : 0;
        }

        if (sample >= MLN_BURNIN) {
            for (int v = 0; v < n_vars; v++) {
                if (g.bool_values[v]) counts[v]++;
            }
        }
    }

    for (int v = 0; v < n_vars; v++) {
        marginal_probs[v] = (float)counts[v] / (float)MLN_SAMPLES;
    }

    safe_free((void**)&counts);
    return 0;
}