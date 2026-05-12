#include "selflnn/evolution/pareto_optimization.h"
#include "selflnn/core/evolutionary_algorithms.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/math_utils.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

// ============================================================
// 帕累托优化配置
// ============================================================

ParetoOptimizationConfig pareto_config_default(int num_objectives) {
    ParetoOptimizationConfig config;
    memset(&config, 0, sizeof(config));
    if (num_objectives < 1) num_objectives = 2;
    if (num_objectives > PARETO_MAX_OBJECTIVES) num_objectives = PARETO_MAX_OBJECTIVES;
    config.num_objectives = num_objectives;
    config.population_size = 100;
    config.max_generations = 50;
    config.crossover_prob = 0.9f;
    config.mutation_prob = 0.1f;
    config.mutation_strength = 0.05f;
    for (int i = 0; i < num_objectives; i++) {
        config.objective_types[i] = (ParetoObjectiveType)i;
        config.weights[i] = 1.0f;
    }
    return config;
}

// ============================================================
// 帕累托前沿提取
// ============================================================

int pareto_front_extract(Population* pop, NSGA2IndividualData* obj_data,
                          ParetoFront* front, int num_objectives) {
    if (!pop || !obj_data || !front || num_objectives < 1) return -1;

    front->entries = NULL;
    front->entry_count = 0;
    front->alloc_count = 0;
    front->num_objectives = num_objectives;

    int front_count = population_compute_non_dominated_sort(pop, obj_data, num_objectives);
    if (front_count < 0) return -1;

    size_t pop_size = population_get_size(pop);
    if (pop_size == 0) return -1;

    int* pareto_indices = (int*)malloc((size_t)pop_size * sizeof(int));
    if (!pareto_indices) return -1;

    front_count = population_extract_pareto_front(pop, obj_data, pareto_indices);
    if (front_count <= 0) {
        free(pareto_indices);
        return -1;
    }

    int pareto_count = front_count;

    int alloc_size = pareto_count < 256 ? 256 : pareto_count;
    front->entries = (ParetoFrontEntry*)calloc((size_t)alloc_size, sizeof(ParetoFrontEntry));
    if (!front->entries) {
        free(pareto_indices);
        return -1;
    }
    front->alloc_count = alloc_size;

    for (int i = 0; i < pareto_count; i++) {
        ParetoFrontEntry* entry = &front->entries[i];
        entry->individual_index = pareto_indices[i];
        entry->rank = 0;
        entry->crowding_distance = obj_data[pareto_indices[i]].crowding_distance;

        for (int j = 0; j < num_objectives; j++) {
            entry->objectives[j] = obj_data[pareto_indices[i]].objectives[j];
        }

        entry->genome_size = 0;
        entry->genome = NULL;

        const float* genome = population_get_individual_genome(pop, (size_t)pareto_indices[i], &entry->genome_size);
        if (genome && entry->genome_size > 0) {
            entry->genome = (float*)safe_malloc(entry->genome_size * sizeof(float));
            if (entry->genome) {
                memcpy(entry->genome, genome, entry->genome_size * sizeof(float));
            }
        }
    }

    free(pareto_indices);
    front->entry_count = pareto_count;

    for (int i = 0; i < num_objectives; i++) {
        front->objective_ranges[i][0] = FLT_MAX;
        front->objective_ranges[i][1] = -FLT_MAX;
        snprintf(front->objective_names[i], PARETO_OBJECTIVE_NAME_MAX, "目标%d", i + 1);
    }

    snprintf(front->objective_names[0], PARETO_OBJECTIVE_NAME_MAX, "精度");
    if (num_objectives > 1) snprintf(front->objective_names[1], PARETO_OBJECTIVE_NAME_MAX, "速度");
    if (num_objectives > 2) snprintf(front->objective_names[2], PARETO_OBJECTIVE_NAME_MAX, "能耗");

    for (int i = 0; i < pareto_count; i++) {
        for (int j = 0; j < num_objectives; j++) {
            float v = front->entries[i].objectives[j];
            if (v < front->objective_ranges[j][0]) front->objective_ranges[j][0] = v;
            if (v > front->objective_ranges[j][1]) front->objective_ranges[j][1] = v;
        }
    }

    float ref_point[PARETO_MAX_OBJECTIVES];
    for (int j = 0; j < num_objectives; j++) {
        ref_point[j] = front->objective_ranges[j][1] * 1.1f + 0.001f;
    }
    pareto_front_compute_hypervolume(front, ref_point);

    front->spacing_metric = pareto_front_compute_spacing(front);

    front->coverage_ratio = (float)pareto_count / (float)(pop_size > 0 ? pop_size : 1);

    return pareto_count;
}

// ============================================================
// 超体积指标计算
// ============================================================

int pareto_front_compute_hypervolume(ParetoFront* front, const float* reference_point) {
    if (!front || !reference_point || front->entry_count == 0) return -1;

    int n = front->entry_count;
    int m = front->num_objectives;

    int* indices = (int*)malloc((size_t)n * sizeof(int));
    if (!indices) return -1;
    for (int i = 0; i < n; i++) indices[i] = i;

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (front->entries[indices[i]].objectives[0] < front->entries[indices[j]].objectives[0]) {
                int t = indices[i];
                indices[i] = indices[j];
                indices[j] = t;
            }
        }
    }

    double hv = 0.0;
    double prev_obj0 = reference_point[0];

    for (int i = 0; i < n; i++) {
        ParetoFrontEntry* e = &front->entries[indices[i]];
        if (e->objectives[0] >= reference_point[0]) continue;

        double width = prev_obj0 - e->objectives[0];
        if (width <= 0.0) continue;

        double height = 1.0;
        for (int j = 1; j < m; j++) {
            height *= (reference_point[j] - e->objectives[j]);
        }
        if (height < 0.0) height = 0.0;

        hv += width * height;
        prev_obj0 = e->objectives[0];
    }

    free(indices);
    front->hypervolume = (float)hv;
    return 0;
}

// ============================================================
// 间距指标（衡量帕累托前沿的均匀分布程度）
// ============================================================

float pareto_front_compute_spacing(const ParetoFront* front) {
    if (!front || front->entry_count < 2) return 0.0f;

    int n = front->entry_count;
    int m = front->num_objectives;

    float* distances = (float*)safe_malloc((size_t)n * sizeof(float));
    if (!distances) return -1.0f;

    float sum_dist = 0.0f;
    for (int i = 0; i < n; i++) {
        float min_dist = FLT_MAX;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            float dist = 0.0f;
            for (int k = 0; k < m; k++) {
                float diff = front->entries[i].objectives[k] - front->entries[j].objectives[k];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
            if (dist < min_dist) min_dist = dist;
        }
        distances[i] = min_dist;
        sum_dist += min_dist;
    }

    float mean_dist = sum_dist / n;

    float variance = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = distances[i] - mean_dist;
        variance += diff * diff;
    }

    safe_free((void**)&distances);
    return sqrtf(variance / n);
}

// ============================================================
// 帕累托前沿转JSON
// ============================================================

int pareto_front_to_json(const ParetoFront* front, char* buffer, size_t buffer_size) {
    if (!front || !buffer || buffer_size == 0) return -1;

    int n = front->entry_count;
    int m = front->num_objectives;

    int written = snprintf(buffer, buffer_size,
        "{\n"
        "  \"entry_count\": %d,\n"
        "  \"num_objectives\": %d,\n"
        "  \"hypervolume\": %.6f,\n"
        "  \"spacing_metric\": %.6f,\n"
        "  \"coverage_ratio\": %.4f,\n"
        "  \"objective_names\": [",
        n, m, front->hypervolume, front->spacing_metric, front->coverage_ratio);

    if (written < 0) return -1;

    size_t pos = (size_t)written;
    for (int j = 0; j < m; j++) {
        int ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
            "%s\"%s\"", j > 0 ? ", " : "", front->objective_names[j]);
        if (ret < 0) return -1;
        pos += (size_t)ret;
        if (pos >= buffer_size) return -1;
    }

    int ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
        "],\n"
        "  \"objective_ranges\": [\n");
    if (ret < 0) return -1;
    pos += (size_t)ret;
    if (pos >= buffer_size) return -1;

    for (int j = 0; j < m; j++) {
        ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
            "    {\"min\": %.6f, \"max\": %.6f}%s\n",
            front->objective_ranges[j][0], front->objective_ranges[j][1],
            j < m - 1 ? "," : "");
        if (ret < 0) return -1;
        pos += (size_t)ret;
        if (pos >= buffer_size) return -1;
    }

    ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
        "  ],\n"
        "  \"pareto_front\": [\n");
    if (ret < 0) return -1;
    pos += (size_t)ret;
    if (pos >= buffer_size) return -1;

    for (int i = 0; i < n; i++) {
        const ParetoFrontEntry* entry = &front->entries[i];
        ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
            "    {\"index\": %d, \"rank\": %d, \"crowding_distance\": %.6f, \"objectives\": [",
            entry->individual_index, entry->rank, entry->crowding_distance);
        if (ret < 0) return -1;
        pos += (size_t)ret;
        if (pos >= buffer_size) return -1;

        for (int j = 0; j < m; j++) {
            ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
                "%s%.6f", j > 0 ? ", " : "", entry->objectives[j]);
            if (ret < 0) return -1;
            pos += (size_t)ret;
            if (pos >= buffer_size) return -1;
        }

        ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
            "]}%s\n", i < n - 1 ? "," : "");
        if (ret < 0) return -1;
        pos += (size_t)ret;
        if (pos >= buffer_size) return -1;
    }

    ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
        "  ]\n"
        "}\n");
    if (ret < 0) return -1;

    return 0;
}

// ============================================================
// 从帕累托前沿中选择最佳解（基于用户偏好权重）
// ============================================================

int pareto_front_select_best(ParetoFront* front, const float* preferences,
                              float* best_genome, size_t genome_size) {
    if (!front || !preferences || !best_genome || genome_size == 0) return -1;

    int n = front->entry_count;
    int m = front->num_objectives;
    if (n == 0 || m == 0) return -1;

    int best_idx = -1;
    float best_score = -FLT_MAX;

    for (int i = 0; i < n; i++) {
        ParetoFrontEntry* entry = &front->entries[i];
        float score = 0.0f;
        for (int j = 0; j < m; j++) {
            float normalized;
            if (front->objective_ranges[j][1] > front->objective_ranges[j][0]) {
                normalized = (entry->objectives[j] - front->objective_ranges[j][0]) /
                             (front->objective_ranges[j][1] - front->objective_ranges[j][0]);
            } else {
                normalized = 0.5f;
            }
            score += preferences[j] * normalized;
        }
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) return -1;

    ParetoFrontEntry* best_entry = &front->entries[best_idx];
    if (best_entry->genome && best_entry->genome_size <= genome_size) {
        memcpy(best_genome, best_entry->genome, best_entry->genome_size * sizeof(float));
        return 0;
    }

    return -1;
}

// ============================================================
// 帕累托多目标演化主循环
// ============================================================

int pareto_multi_evolve(Population* pop, MultiObjectiveFunction obj_func,
                         const ParetoOptimizationConfig* config, void* user_data,
                         ParetoFront* front) {
    if (!pop || !obj_func || !config || !front) return -1;

    size_t pop_size = population_get_size(pop);
    if (pop_size == 0) return -1;

    NSGA2Config nsga2_cfg = nsga2_config_default(config->num_objectives);
    nsga2_cfg.crossover_prob = config->crossover_prob;
    nsga2_cfg.mutation_prob = config->mutation_prob;
    nsga2_cfg.mutation_strength = config->mutation_strength;

    int max_gen = config->max_generations;
    if (max_gen <= 0) max_gen = 50;

    NSGA2IndividualData* all_obj_data = NULL;
    Population* work_pop = pop;

    int gen_count = 0;
    for (int gen = 0; gen < max_gen; gen++) {
        int result = population_nsga2_evolve(work_pop, obj_func, &nsga2_cfg, user_data);
        if (result < 0) return -1;
        gen_count++;
    }

    all_obj_data = (NSGA2IndividualData*)safe_malloc(pop_size * sizeof(NSGA2IndividualData));
    if (!all_obj_data) return -1;

    for (size_t i = 0; i < pop_size; i++) {
        all_obj_data[i].objectives = (float*)safe_malloc((size_t)config->num_objectives * sizeof(float));
        if (!all_obj_data[i].objectives) {
            for (size_t j = 0; j < i; j++) safe_free((void**)&all_obj_data[j].objectives);
            safe_free((void**)&all_obj_data);
            return -1;
        }
        all_obj_data[i].rank = -1;
        all_obj_data[i].crowding_distance = 0.0f;
    }

    int result = pareto_front_extract(pop, all_obj_data, front, config->num_objectives);

    for (size_t i = 0; i < pop_size; i++) {
        safe_free((void**)&all_obj_data[i].objectives);
    }
    safe_free((void**)&all_obj_data);

    return result;
}

// ============================================================
// 帕累托演化训练（创建新种群、演化、输出帕累托前沿）
// ============================================================

typedef struct {
    LNN* network;
    const float* train_inputs;
    const float* train_targets;
    size_t num_samples;
    int max_iterations;
    ParetoOptimizationConfig* config;
} ParetoEvolveUserData;

static void pareto_internal_multi_obj(const float* genome, size_t genome_size,
                                       float* objectives, int num_objectives,
                                       void* user_data) {
    ParetoEvolveUserData* data = (ParetoEvolveUserData*)user_data;
    if (!data || !data->network || !data->config) return;

    EvolutionTrainingData eval_data;
    memset(&eval_data, 0, sizeof(eval_data));
    eval_data.network = data->network;
    eval_data.train_inputs = data->train_inputs;
    eval_data.train_targets = data->train_targets;
    eval_data.num_samples = data->num_samples;
    eval_data.accuracy_weight = 1.0f;
    eval_data.speed_weight = 0.5f;
    eval_data.energy_weight = 0.3f;
    eval_data.max_eval_iterations = data->max_iterations;

    evolution_apply_genome_to_network(data->network, genome, genome_size);

    evolution_multi_objective_train(genome, genome_size, objectives, num_objectives, &eval_data);
}

int pareto_evolutionary_train(float* genome, size_t genome_size,
                               ParetoOptimizationConfig* config,
                               MultiObjectiveFunction obj_func, void* user_data,
                               ParetoFront* front) {
    if (!genome || genome_size == 0 || !config || !front) return -1;

    Population* pop = population_create((size_t)config->population_size, genome_size, 0);
    if (!pop) return -1;

    population_initialize_random(pop, -1.0f, 1.0f);

    ParetoEvolveUserData internal_data;
    MultiObjectiveFunction actual_obj_func = obj_func;
    void* actual_user_data = user_data;

    if (!actual_obj_func) {
        ParetoEvolveUserData* pdata = &internal_data;
        if (!user_data) {
            memset(pdata, 0, sizeof(ParetoEvolveUserData));
            pdata->config = config;
            pdata->max_iterations = 100;
        }
        actual_obj_func = pareto_internal_multi_obj;
        actual_user_data = pdata;
    }

    int result = pareto_multi_evolve(pop, actual_obj_func, config, actual_user_data, front);

    if (result > 0 && front->entry_count > 0) {
        float default_prefs[PARETO_MAX_OBJECTIVES];
        for (int i = 0; i < config->num_objectives; i++) {
            default_prefs[i] = config->weights[i];
        }
        pareto_front_select_best(front, default_prefs, genome, genome_size);
    }

    population_destroy(pop);

    return result;
}

// ============================================================
// 训练结果转JSON
// ============================================================

int pareto_training_results_to_json(const float* genome, size_t genome_size,
                                     const float* objectives, int num_objectives,
                                     char* buffer, size_t buffer_size) {
    if (!genome || !objectives || !buffer || buffer_size == 0) return -1;

    int written = snprintf(buffer, buffer_size,
        "{\n"
        "  \"genome_size\": %zu,\n"
        "  \"num_objectives\": %d,\n"
        "  \"objectives\": [",
        genome_size, num_objectives);

    if (written < 0) return -1;
    size_t pos = (size_t)written;

    for (int j = 0; j < num_objectives; j++) {
        int ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
            "%s%.6f", j > 0 ? ", " : "", objectives[j]);
        if (ret < 0) return -1;
        pos += (size_t)ret;
        if (pos >= buffer_size) return -1;
    }

    int ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
        "],\n"
        "  \"genome_sample\": [");
    if (ret < 0) return -1;
    pos += (size_t)ret;
    if (pos >= buffer_size) return -1;

    int sample_count = (int)(genome_size < 5 ? genome_size : 5);
    for (int i = 0; i < sample_count; i++) {
        ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
            "%s%.6f", i > 0 ? ", " : "", genome[i]);
        if (ret < 0) return -1;
        pos += (size_t)ret;
        if (pos >= buffer_size) return -1;
    }

    ret = snprintf(buffer + pos, buffer_size > pos ? buffer_size - pos : 0,
        "]\n"
        "}\n");
    if (ret < 0) return -1;

    return 0;
}

// ============================================================
// MOEA/D 基于分解的多目标进化算法
// ============================================================

MOEADConfig pareto_moead_config_default(int num_objectives) {
    MOEADConfig config;
    memset(&config, 0, sizeof(config));
    if (num_objectives < 1) num_objectives = 2;
    if (num_objectives > PARETO_MAX_OBJECTIVES) num_objectives = PARETO_MAX_OBJECTIVES;
    config.decomposition_type = MOEAD_DECOMPOSITION_TEICHEBYCHEFF;
    config.num_objectives = num_objectives;
    config.population_size = 100;
    config.max_generations = 50;
    config.neighborhood_size = 20;
    config.crossover_prob = 0.9f;
    config.mutation_prob = 0.1f;
    config.mutation_strength = 0.05f;
    for (int j = 0; j < num_objectives; j++) {
        config.ideal_point[j] = FLT_MAX;
    }
    return config;
}

/**
 * @brief 生成MOEA/D均匀分布权重向量（单纯形格点法）
 *
 * 生成所有和为1的权重组合 (0/H, 1/H, ..., H/H)
 * 使用递归方法生成组合C(H+m-1, m-1)
 */
static int moead_generate_weights(int m, int H, float* weights_out, int max_count) {
    if (m < 2 || H < 1) return 0;
    int count = 0;
    int* indices = (int*)safe_malloc((size_t)(m + 1) * sizeof(int));
    if (!indices) return 0;

    for (int i = 0; i < m; i++) indices[i] = 0;
    indices[m] = H;

    while (1) {
        if (count >= max_count) break;

        float* w = &weights_out[(size_t)count * (size_t)m];
        for (int j = 0; j < m; j++) {
            w[j] = (float)(indices[j + 1] - indices[j]) / (float)H;
        }
        count++;

        int p = m - 1;
        while (p >= 1 && indices[p] == indices[p + 1]) p--;
        if (p == 0) break;
        indices[p]++;
        int v = indices[p];
        for (int q = p + 1; q <= m; q++) {
            if (indices[q] > v) indices[q] = indices[q - 1] + 1;
        }
    }

    safe_free((void**)&indices);
    return count;
}

/**
 * @brief 计算MOEA/D邻域（基于权重向量欧氏距离）
 */
static int moead_compute_neighborhood(const float* weights, int num_weights, int m,
                                       int T, int* neighborhood_out) {
    int nw = num_weights;
    if (nw == 0 || T <= 0) return -1;

    float* dists = (float*)safe_malloc((size_t)nw * sizeof(float));
    if (!dists) return -1;

    for (int i = 0; i < nw; i++) {
        // 计算权重向量i与所有向量的距离并排序
        for (int k = 0; k < nw; k++) {
            float d = 0.0f;
            for (int j = 0; j < m; j++) {
                float diff = weights[(size_t)i * (size_t)m + (size_t)j]
                           - weights[(size_t)k * (size_t)m + (size_t)j];
                d += diff * diff;
            }
            dists[k] = sqrtf(d);
        }

        // 选择T个最近的邻居（冒泡排序取前T个）
        int* pos = (int*)safe_malloc((size_t)nw * sizeof(int));
        if (!pos) { safe_free((void**)&dists); return -1; }
        for (int k = 0; k < nw; k++) pos[k] = k;

        for (int a = 0; a < nw - 1 && a < T; a++) {
            for (int b = a + 1; b < nw; b++) {
                if (dists[pos[b]] < dists[pos[a]]) {
                    int tmp = pos[a]; pos[a] = pos[b]; pos[b] = tmp;
                }
            }
        }

        for (int t = 0; t < T && t < nw; t++) {
            neighborhood_out[(size_t)i * (size_t)T + (size_t)t] = pos[t];
        }

        safe_free((void**)&pos);
    }

    safe_free((void**)&dists);
    return 0;
}

/**
 * @brief 计算切比雪夫标量函数值 g^te(x|λ,z*)
 */
static float moead_tchebycheff(const float* objectives, int m,
                                const float* lambda, const float* ideal) {
    float max_val = -FLT_MAX;
    for (int j = 0; j < m; j++) {
        float diff = objectives[j] - ideal[j];
        if (diff < 0.0f) diff = 0.0f;
        float val = lambda[j] * diff;
        if (val > max_val) max_val = val;
    }
    return max_val;
}

/**
 * @brief 计算加权和标量函数值 g^ws(x|λ)
 */
static float moead_weighted_sum(const float* objectives, int m,
                                 const float* lambda) {
    float sum = 0.0f;
    for (int j = 0; j < m; j++) {
        sum += lambda[j] * objectives[j];
    }
    return sum;
}

int pareto_moead_evolve(Population* pop, MultiObjectiveFunction obj_func,
                         const MOEADConfig* config, void* user_data,
                         ParetoFront* front) {
    if (!pop || !obj_func || !config || !front) return -1;

    size_t pop_size = population_get_size(pop);
    if (pop_size == 0) pop_size = config->population_size;

    int m = config->num_objectives;
    if (m < 1) return -1;

    // 确定基因组大小
    size_t genome_size = 64;
    {
        size_t gs = 0;
        const float* g = population_get_individual_genome(pop, 0, &gs);
        if (g && gs > 0) genome_size = gs;
    }

    // 生成权重向量
    int H = 0;
    if (m == 2) H = (int)pop_size - 1;
    else if (m == 3) H = (int)(sqrtf((float)pop_size * 6.0f) + 0.5f);
    else H = (int)pop_size;

    size_t max_weights = pop_size + 100;
    float* weights = (float*)safe_malloc(max_weights * (size_t)m * sizeof(float));
    if (!weights) return -1;

    int num_weights = moead_generate_weights(m, H, weights, (int)max_weights);
    if (num_weights <= 0) {
        num_weights = (int)pop_size;
        for (int i = 0; i < num_weights; i++) {
            float* w = &weights[(size_t)i * (size_t)m];
            float sum = 0.0f;
            for (int j = 0; j < m; j++) {
                w[j] = rng_uniform(0.1f, 1.0f);
                sum += w[j];
            }
            for (int j = 0; j < m; j++) w[j] /= sum;
        }
    }

    if (num_weights > (int)pop_size) num_weights = (int)pop_size;

    int T = config->neighborhood_size;
    if (T > num_weights) T = num_weights - 1;
    if (T < 2) T = 2;

    int* neighborhood = (int*)safe_malloc((size_t)num_weights * (size_t)T * sizeof(int));
    if (!neighborhood) { safe_free((void**)&weights); return -1; }

    if (moead_compute_neighborhood(weights, num_weights, m, T, neighborhood) != 0) {
        safe_free((void**)&weights);
        safe_free((void**)&neighborhood);
        return -1;
    }

    // 初始化基因组矩阵 [num_weights][genome_size]
    float* genomes = (float*)safe_malloc((size_t)num_weights * genome_size * sizeof(float));
    if (!genomes) {
        safe_free((void**)&weights);
        safe_free((void**)&neighborhood);
        return -1;
    }

    // 从种群初始化基因组或随机初始化
    size_t pop_genome_size = 0;
    const float* pop_genome = population_get_individual_genome(pop, 0, &pop_genome_size);
    if (pop_genome && pop_genome_size == genome_size) {
        for (int i = 0; i < num_weights; i++) {
            memcpy(&genomes[(size_t)i * genome_size], pop_genome, genome_size * sizeof(float));
        }
    } else {
        for (int i = 0; i < num_weights; i++) {
            float* g = &genomes[(size_t)i * genome_size];
            for (size_t k = 0; k < genome_size; k++) {
                g[k] = rng_uniform(-1.0f, 1.0f);
            }
        }
    }

    // 初始化目标值矩阵 [num_weights][m]
    float* objectives = (float*)safe_calloc((size_t)num_weights * (size_t)m, sizeof(float));
    if (!objectives) {
        safe_free((void**)&weights);
        safe_free((void**)&neighborhood);
        safe_free((void**)&genomes);
        return -1;
    }

    float* scalar_vals = (float*)safe_malloc((size_t)num_weights * sizeof(float));
    if (!scalar_vals) {
        safe_free((void**)&weights);
        safe_free((void**)&neighborhood);
        safe_free((void**)&genomes);
        safe_free((void**)&objectives);
        return -1;
    }

    float ideal[PARETO_MAX_OBJECTIVES];
    for (int j = 0; j < m; j++) {
        ideal[j] = config->ideal_point[j];
    }

    // 评估初始种群
    for (int i = 0; i < num_weights; i++) {
        float* g = &genomes[(size_t)i * genome_size];
        float* obj = &objectives[(size_t)i * (size_t)m];
        obj_func(g, genome_size, obj, m, user_data);
        for (int j = 0; j < m; j++) {
            if (obj[j] < ideal[j]) ideal[j] = obj[j];
        }
    }

    for (int i = 0; i < num_weights; i++) {
        const float* lambda = &weights[(size_t)i * (size_t)m];
        const float* obj = &objectives[(size_t)i * (size_t)m];
        if (config->decomposition_type == MOEAD_DECOMPOSITION_TEICHEBYCHEFF) {
            scalar_vals[i] = moead_tchebycheff(obj, m, lambda, ideal);
        } else {
            scalar_vals[i] = moead_weighted_sum(obj, m, lambda);
        }
    }

    float* child_genome = (float*)safe_malloc(genome_size * sizeof(float));
    float child_obj[PARETO_MAX_OBJECTIVES];
    if (!child_genome) {
        safe_free((void**)&weights);
        safe_free((void**)&neighborhood);
        safe_free((void**)&genomes);
        safe_free((void**)&objectives);
        safe_free((void**)&scalar_vals);
        return -1;
    }

    int max_gen = config->max_generations;
    if (max_gen <= 0) max_gen = 50;

    for (int gen = 0; gen < max_gen; gen++) {
        int update_count = 0;

        for (int i = 0; i < num_weights; i++) {
            int* b_i = &neighborhood[(size_t)i * (size_t)T];
            int p1 = b_i[(int)(rng_uniform(0.0f, (float)T))];
            int p2 = b_i[(int)(rng_uniform(0.0f, (float)T))];
            while (p2 == p1 && T > 1) {
                p2 = b_i[(int)(rng_uniform(0.0f, (float)T))];
            }

            const float* g1 = &genomes[(size_t)p1 * genome_size];
            const float* g2 = &genomes[(size_t)p2 * genome_size];

            float cr = config->crossover_prob;
            int crossover_happened = 0;
            for (size_t k = 0; k < genome_size; k++) {
                if (rng_uniform(0.0f, 1.0f) < cr) {
                    float u = rng_uniform(0.0f, 1.0f);
                    float beta;
                    if (u <= 0.5f) {
                        beta = powf(2.0f * u, 1.0f / 21.0f);
                    } else {
                        beta = powf(1.0f / (2.0f * (1.0f - u)), 1.0f / 21.0f);
                    }
                    child_genome[k] = 0.5f * ((1.0f - beta) * g1[k] + (1.0f + beta) * g2[k]);
                    crossover_happened = 1;
                } else {
                    child_genome[k] = g1[k];
                }
            }

            if (!crossover_happened) {
                memcpy(child_genome, g1, genome_size * sizeof(float));
            }

            float mp = config->mutation_prob;
            float ms = config->mutation_strength;
            for (size_t k = 0; k < genome_size; k++) {
                if (rng_uniform(0.0f, 1.0f) < mp) {
                    child_genome[k] += ms * rng_normal(0.0f, 1.0f);
                    if (child_genome[k] > 2.0f) child_genome[k] = 2.0f;
                    if (child_genome[k] < -2.0f) child_genome[k] = -2.0f;
                }
            }

            obj_func(child_genome, genome_size, child_obj, m, user_data);

            for (int j = 0; j < m; j++) {
                if (child_obj[j] < ideal[j]) ideal[j] = child_obj[j];
            }

            int replace_count = 0;
            int shuffled[256];
            for (int t = 0; t < T; t++) shuffled[t] = t;
            for (int a = 0; a < T; a++) {
                int b = (int)(rng_uniform(0.0f, (float)T));
                if (b < T) {
                    int tmp = shuffled[a]; shuffled[a] = shuffled[b]; shuffled[b] = tmp;
                }
            }

            for (int t_idx = 0; t_idx < T && replace_count < 1; t_idx++) {
                int n_idx = b_i[shuffled[t_idx]];
                if (n_idx >= num_weights) continue;

                const float* n_lambda = &weights[(size_t)n_idx * (size_t)m];
                float n_scalar;
                if (config->decomposition_type == MOEAD_DECOMPOSITION_TEICHEBYCHEFF) {
                    n_scalar = moead_tchebycheff(child_obj, m, n_lambda, ideal);
                } else {
                    n_scalar = moead_weighted_sum(child_obj, m, n_lambda);
                }

                if (n_scalar < scalar_vals[n_idx]) {
                    memcpy(&genomes[(size_t)n_idx * genome_size], child_genome, genome_size * sizeof(float));
                    memcpy(&objectives[(size_t)n_idx * (size_t)m], child_obj, (size_t)m * sizeof(float));
                    scalar_vals[n_idx] = n_scalar;
                    replace_count++;
                    update_count++;
                }
            }
        }

        if ((gen + 1) % 10 == 0) {
            float min_scalar = FLT_MAX, avg_scalar = 0.0f;
            for (int i = 0; i < num_weights; i++) {
                if (scalar_vals[i] < min_scalar) min_scalar = scalar_vals[i];
                avg_scalar += scalar_vals[i];
            }
            avg_scalar /= (float)num_weights;
            log_info("MOEA/D 第%d代: 最小标量值=%.6f 平均标量值=%.6f 更新数=%d",
                     gen + 1, min_scalar, avg_scalar, update_count);
        }
    }

    // ========== 提取帕累托前沿 ==========
    front->entries = NULL;
    front->entry_count = 0;
    front->alloc_count = 0;
    front->num_objectives = m;

    int* front_indices = (int*)malloc((size_t)num_weights * sizeof(int));
    int front_count = 0;

    for (int i = 0; i < num_weights; i++) {
        int dominated = 0;
        for (int j = 0; j < num_weights; j++) {
            if (i == j) continue;
            int dominates = 1;
            int strictly_better = 0;
            for (int k = 0; k < m; k++) {
                float oi = objectives[(size_t)i * (size_t)m + (size_t)k];
                float oj = objectives[(size_t)j * (size_t)m + (size_t)k];
                if (oj > oi) { dominates = 0; break; }
                if (oj < oi) strictly_better = 1;
            }
            if (dominates && strictly_better) {
                dominated = 1;
                break;
            }
        }
        if (!dominated) {
            front_indices[front_count++] = i;
        }
    }

    int alloc_size = front_count < 256 ? 256 : front_count;
    front->entries = (ParetoFrontEntry*)calloc((size_t)alloc_size, sizeof(ParetoFrontEntry));
    front->alloc_count = alloc_size;

    for (int i = 0; i < front_count; i++) {
        ParetoFrontEntry* entry = &front->entries[i];
        entry->individual_index = front_indices[i];
        entry->rank = 0;
        entry->crowding_distance = 0.0f;
        for (int j = 0; j < m; j++) {
            entry->objectives[j] = objectives[(size_t)front_indices[i] * (size_t)m + (size_t)j];
        }
        entry->genome_size = genome_size;
        entry->genome = (float*)safe_malloc(genome_size * sizeof(float));
        if (entry->genome) {
            memcpy(entry->genome, &genomes[(size_t)front_indices[i] * genome_size], genome_size * sizeof(float));
        }
    }

    free(front_indices);
    front->entry_count = front_count;

    for (int j = 0; j < m; j++) {
        front->objective_ranges[j][0] = FLT_MAX;
        front->objective_ranges[j][1] = -FLT_MAX;
        snprintf(front->objective_names[j], PARETO_OBJECTIVE_NAME_MAX,
                 j == 0 ? "精度" : (j == 1 ? "速度" : (j == 2 ? "能耗" : "目标%d")), j + 1);
    }

    for (int i = 0; i < front_count; i++) {
        for (int j = 0; j < m; j++) {
            float v = front->entries[i].objectives[j];
            if (v < front->objective_ranges[j][0]) front->objective_ranges[j][0] = v;
            if (v > front->objective_ranges[j][1]) front->objective_ranges[j][1] = v;
        }
    }

    float ref_point[PARETO_MAX_OBJECTIVES];
    for (int j = 0; j < m; j++) {
        ref_point[j] = front->objective_ranges[j][1] * 1.1f + 0.001f;
    }
    pareto_front_compute_hypervolume(front, ref_point);
    front->spacing_metric = pareto_front_compute_spacing(front);
    front->coverage_ratio = (float)front_count / (float)num_weights;

    safe_free((void**)&weights);
    safe_free((void**)&neighborhood);
    safe_free((void**)&genomes);
    safe_free((void**)&objectives);
    safe_free((void**)&scalar_vals);
    safe_free((void**)&child_genome);

    return front_count;
}

// ============================================================
// 帕累托前沿内存释放
// ============================================================

void pareto_front_destroy(ParetoFront* front) {
    if (!front) return;
    for (int i = 0; i < front->entry_count; i++) {
        if (front->entries && front->entries[i].genome) {
            safe_free((void**)&front->entries[i].genome);
        }
    }
    if (front->entries) {
        free(front->entries);
        front->entries = NULL;
    }
    front->entry_count = 0;
    front->alloc_count = 0;
}
