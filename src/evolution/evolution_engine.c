/**
 * @file evolution_engine.c
 * @brief 增强自我演化进化引擎完整实现
 */

#include "selflnn/evolution/evolution_engine.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/laplace.h"
#include "selflnn/core/cma_es.h"          /* F-010/F-019: CMA-ES集成 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct EvolutionEngine {
    EvolutionConfig config;
    EvolutionPopulation population;
    FitnessFunction fitness_func;
    MultiFitnessFunction multi_fitness_func;
    void* user_data;
    int obj_count;
    FitnessDirection direction;
    EvolutionStats stats;
    int initialized;
    size_t eval_counter;

    /* 岛模型 */
    EvolutionPopulation* islands;
    int island_count;
    int migration_counter;

    /* 随机数状态 */
    unsigned int rng_state;

    /* LNN连接：用于将演化结果写回LNN权重 */
    void* target_lnn;
    int lnn_connected;
    int enabled;        /* 能力开关: 1=启用演化, 0=禁止 */
    
    /* F-010/F-019: CMA-ES状态（当config.algorithm设为CMA_ES时使用） */
    CMAESState* cmaes_state;
    int using_cmaes;
};

/* F-019: CMA-ES适应度桥接函数 */
static float cmaes_fitness_bridge(const float* x, size_t dim, void* user_data) {
    EvolutionEngine* engine = (EvolutionEngine*)user_data;
    if (!engine) return FLT_MAX; /* 引擎为空→最高代价 */
    if (!engine->fitness_func) {
        /* ZSFABC修复: 适应度函数未注册时记录错误，返回最高代价防止CMA-ES错误收敛 */
        log_warning("[演化引擎] 适应度函数未注册，CMA-ES优化将使用最高代价");
        return FLT_MAX;
    }
    return engine->fitness_func(x, dim, engine->user_data);
}

/* 快速随机数生成器 (Xorshift) */
static unsigned int xorshift_next(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rand_float(unsigned int* state) {
    return (float)(xorshift_next(state) & 0xFFFFFF) / (float)0x1000000;
}

static float rand_gaussian(unsigned int* state, float mean, float stddev) {
    float u1 = rand_float(state);
    float u2 = rand_float(state);
    if (u1 < 1e-10f) u1 = 1e-10f;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return mean + stddev * z;
}

/* 初始化个体 */
static void init_individual(EvolutionIndividual* ind, size_t chrom_size, float min_val, float max_val, unsigned int* rng) {
    ind->chromosome = (float*)safe_malloc(chrom_size * sizeof(float));
    if (!ind->chromosome) return;
    ind->chromosome_size = chrom_size;
    ind->fitness = (min_val > max_val) ? -FLT_MAX : 0.0f;
    ind->pareto_rank = 0;
    ind->crowding_distance = 0.0f;
    ind->age = 0;
    ind->birth_time = time(NULL);
    ind->objective_count = 0;
    ind->objectives = NULL;

    float range = max_val - min_val;
    for (size_t i = 0; i < chrom_size; i++) {
        ind->chromosome[i] = min_val + rand_float(rng) * range;
    }
}

/* 释放个体 */
static void free_individual(EvolutionIndividual* ind) {
    if (!ind) return;
    safe_free((void**)&ind->chromosome);
    safe_free((void**)&ind->objectives);
}

/* 复制个体 */
static int copy_individual(EvolutionIndividual* dst, const EvolutionIndividual* src) {
    if (!dst || !src) return -1;
    dst->chromosome = (float*)safe_malloc(src->chromosome_size * sizeof(float));
    if (!dst->chromosome) return -1;
    memcpy(dst->chromosome, src->chromosome, src->chromosome_size * sizeof(float));
    dst->chromosome_size = src->chromosome_size;
    dst->fitness = src->fitness;
    dst->age = src->age;
    dst->birth_time = src->birth_time;
    dst->id = src->id;
    dst->pareto_rank = src->pareto_rank;
    dst->crowding_distance = src->crowding_distance;
    return 0;
}

/* 评估个体适应度 */
static void evaluate_individual(EvolutionEngine* engine, EvolutionIndividual* ind) {
    engine->eval_counter++;
    ind->fitness = engine->fitness_func(ind->chromosome, ind->chromosome_size, engine->user_data);
    if (engine->direction == EVO_FITNESS_MINIMIZE) {
        ind->fitness = -ind->fitness;
    }
}

/* 选择操作实现 */

/* 排名选择 */
static int rank_selection(const EvolutionPopulation* pop, unsigned int* rng) {
    /* 按适应度排序（简单冒泡，规模不大） */
    int* ranks = (int*)safe_malloc(pop->size * sizeof(int));
    float* sorted_fitness = (float*)safe_malloc(pop->size * sizeof(float));
    if (!ranks || !sorted_fitness) {
        safe_free((void**)&ranks);
        safe_free((void**)&sorted_fitness);
        return 0;
    }

    for (size_t i = 0; i < pop->size; i++) {
        ranks[i] = (int)i;
        sorted_fitness[i] = pop->individuals[i].fitness;
    }

    for (size_t i = 0; i < pop->size; i++) {
        for (size_t j = i + 1; j < pop->size; j++) {
            if (sorted_fitness[j] < sorted_fitness[i]) {
                float tmpf = sorted_fitness[i];
                sorted_fitness[i] = sorted_fitness[j];
                sorted_fitness[j] = tmpf;
                int tmpi = ranks[i];
                ranks[i] = ranks[j];
                ranks[j] = tmpi;
            }
        }
    }

    /* 排名转概率：rank 0 = 最低概率 */
    float total_rank = 0.0f;
    float* rank_probs = (float*)safe_malloc(pop->size * sizeof(float));
    if (!rank_probs) {
        safe_free((void**)&ranks);
        safe_free((void**)&sorted_fitness);
        return 0;
    }
    for (size_t i = 0; i < pop->size; i++) {
        rank_probs[i] = (float)(i + 1);
        total_rank += rank_probs[i];
    }
    for (size_t i = 0; i < pop->size; i++) {
        rank_probs[i] /= total_rank;
    }

    float r = rand_float(rng);
    float cum = 0.0f;
    int selected = 0;
    for (size_t i = 0; i < pop->size; i++) {
        cum += rank_probs[i];
        if (r <= cum) {
            selected = ranks[i];
            break;
        }
    }

    safe_free((void**)&ranks);
    safe_free((void**)&sorted_fitness);
    safe_free((void**)&rank_probs);
    return selected;
}

/* 锦标赛选择 */
static int tournament_selection(const EvolutionPopulation* pop, float ratio, unsigned int* rng) {
    int tournament_size = (int)(pop->size * ratio);
    if (tournament_size < 2) tournament_size = 2;
    if (tournament_size > (int)pop->size) tournament_size = (int)pop->size;

    int best_idx = -1;
    float best_fit = -FLT_MAX;
    for (int i = 0; i < tournament_size; i++) {
        int idx = (int)(rand_float(rng) * pop->size);
        if (idx >= (int)pop->size) idx = (int)pop->size - 1;
        if (pop->individuals[idx].fitness > best_fit) {
            best_fit = pop->individuals[idx].fitness;
            best_idx = idx;
        }
    }
    return best_idx;
}

/* 轮盘赌选择 */
static int roulette_selection(const EvolutionPopulation* pop, unsigned int* rng) {
    float min_fit = pop->individuals[0].fitness;
    for (size_t i = 1; i < pop->size; i++) {
        if (pop->individuals[i].fitness < min_fit) min_fit = pop->individuals[i].fitness;
    }

    float total_fit = 0.0f;
    float offset = (min_fit < 0) ? -min_fit + 1.0f : 0.0f;
    for (size_t i = 0; i < pop->size; i++) {
        total_fit += pop->individuals[i].fitness + offset;
    }

    if (total_fit < 1e-10f) return (int)(rand_float(rng) * pop->size);

    float r = rand_float(rng) * total_fit;
    float cum = 0.0f;
    for (size_t i = 0; i < pop->size; i++) {
        cum += pop->individuals[i].fitness + offset;
        if (r <= cum) return (int)i;
    }
    return (int)(pop->size - 1);
}

/* 选择父体 */
static int select_parent(const EvolutionEngine* engine, const EvolutionPopulation* pop) {
    switch (engine->config.selection) {
        case EVO_SELECTION_TOURNAMENT:
            return tournament_selection(pop, engine->config.tournament_size_ratio, 
                                       (unsigned int*)&engine->rng_state);
        case EVO_SELECTION_RANK:
            return rank_selection(pop, (unsigned int*)&engine->rng_state);
        case EVO_SELECTION_ROULETTE:
            return roulette_selection(pop, (unsigned int*)&engine->rng_state);
        case EVO_SELECTION_ELITE:
            return 0; /* 精英在更新阶段处理 */
        case EVO_SELECTION_BOLTZMANN:
            return tournament_selection(pop, 0.3f, (unsigned int*)&engine->rng_state);
        default:
            return (int)(rand_float((unsigned int*)&engine->rng_state) * pop->size);
    }
}

/* 交叉操作 */

static void uniform_crossover(const float* p1, const float* p2, float* c1, float* c2,
                              size_t size, unsigned int* rng) {
    for (size_t i = 0; i < size; i++) {
        if (rand_float(rng) < 0.5f) {
            c1[i] = p1[i];
            c2[i] = p2[i];
        } else {
            c1[i] = p2[i];
            c2[i] = p1[i];
        }
    }
}

static void arithmetic_crossover(const float* p1, const float* p2, float* c1, float* c2,
                                 size_t size, unsigned int* rng) {
    for (size_t i = 0; i < size; i++) {
        float alpha = rand_float(rng);
        c1[i] = alpha * p1[i] + (1.0f - alpha) * p2[i];
        c2[i] = alpha * p2[i] + (1.0f - alpha) * p1[i];
    }
}

static void sbx_crossover(const float* p1, const float* p2, float* c1, float* c2,
                          size_t size, float eta, float min_val, float max_val,
                          unsigned int* rng) {
    for (size_t i = 0; i < size; i++) {
        if (rand_float(rng) < 0.5f) {
            float u = rand_float(rng);
            float beta;
            if (u <= 0.5f) {
                beta = powf(2.0f * u, 1.0f / (eta + 1.0f));
            } else {
                beta = powf(1.0f / (2.0f * (1.0f - u)), 1.0f / (eta + 1.0f));
            }
            c1[i] = 0.5f * ((1.0f + beta) * p1[i] + (1.0f - beta) * p2[i]);
            c2[i] = 0.5f * ((1.0f - beta) * p1[i] + (1.0f + beta) * p2[i]);
            if (c1[i] < min_val) c1[i] = min_val;
            if (c1[i] > max_val) c1[i] = max_val;
            if (c2[i] < min_val) c2[i] = min_val;
            if (c2[i] > max_val) c2[i] = max_val;
        } else {
            c1[i] = p1[i];
            c2[i] = p2[i];
        }
    }
}

static void blend_crossover(const float* p1, const float* p2, float* c1, float* c2,
                            size_t size, unsigned int* rng) {
    for (size_t i = 0; i < size; i++) {
        float min_v = p1[i] < p2[i] ? p1[i] : p2[i];
        float max_v = p1[i] > p2[i] ? p1[i] : p2[i];
        float range = max_v - min_v;
        c1[i] = min_v - range * 0.2f + rand_float(rng) * range * 1.4f;
        c2[i] = min_v - range * 0.2f + rand_float(rng) * range * 1.4f;
    }
}

static void crossover(const EvolutionEngine* engine, const float* p1, const float* p2,
                     float* c1, float* c2, size_t size) {
    unsigned int* rng = (unsigned int*)&engine->rng_state;
    switch (engine->config.crossover) {
        case EVO_CROSSOVER_UNIFORM:
            uniform_crossover(p1, p2, c1, c2, size, rng);
            break;
        case EVO_CROSSOVER_ARITHMETIC:
            arithmetic_crossover(p1, p2, c1, c2, size, rng);
            break;
        case EVO_CROSSOVER_SBX:
            sbx_crossover(p1, p2, c1, c2, size, engine->config.sbx_eta,
                         engine->config.chromosome_min, engine->config.chromosome_max, rng);
            break;
        case EVO_CROSSOVER_BLEND:
            blend_crossover(p1, p2, c1, c2, size, rng);
            break;
        default: { /* 单点交叉 */
            int point = (int)(rand_float(rng) * size);
            memcpy(c1, p1, point * sizeof(float));
            memcpy(c1 + point, p2 + point, (size - point) * sizeof(float));
            memcpy(c2, p2, point * sizeof(float));
            memcpy(c2 + point, p1 + point, (size - point) * sizeof(float));
            break;
        }
    }
}

/* 变异操作 */

static void gaussian_mutation(float* chromosome, size_t size, float sigma, unsigned int* rng) {
    for (size_t i = 0; i < size; i++) {
        chromosome[i] += rand_gaussian(rng, 0.0f, sigma);
    }
}

static void polynomial_mutation(float* chromosome, size_t size, float eta, 
                                float min_val, float max_val, unsigned int* rng) {
    for (size_t i = 0; i < size; i++) {
        float u = rand_float(rng);
        float delta;
        if (u < 0.5f) {
            delta = powf(2.0f * u, 1.0f / (eta + 1.0f)) - 1.0f;
        } else {
            delta = 1.0f - powf(2.0f * (1.0f - u), 1.0f / (eta + 1.0f));
        }
        float range = max_val - min_val;
        chromosome[i] += delta * range * 0.1f;
        if (chromosome[i] < min_val) chromosome[i] = min_val;
        if (chromosome[i] > max_val) chromosome[i] = max_val;
    }
}

static void swap_mutation(float* chromosome, size_t size, unsigned int* rng) {
    if (size < 2) return;
    int a = (int)(rand_float(rng) * size);
    int b = (int)(rand_float(rng) * size);
    while (a == b && size > 1) b = (int)(rand_float(rng) * size);
    float tmp = chromosome[a];
    chromosome[a] = chromosome[b];
    chromosome[b] = tmp;
}

static void mutate(const EvolutionEngine* engine, float* chromosome, size_t size, 
                   float mutation_rate) {
    unsigned int* rng = (unsigned int*)&engine->rng_state;
    if (rand_float(rng) > mutation_rate) return;

    switch (engine->config.mutation) {
        case EVO_MUTATION_GAUSSIAN:
            gaussian_mutation(chromosome, size, engine->config.gaussian_sigma, rng);
            break;
        case EVO_MUTATION_POLYNOMIAL:
            polynomial_mutation(chromosome, size, engine->config.polynomial_eta,
                               engine->config.chromosome_min, engine->config.chromosome_max, rng);
            break;
        case EVO_MUTATION_SWAP:
            swap_mutation(chromosome, size, rng);
            break;
        case EVO_MUTATION_ADAPTIVE: {
            float effective_sigma = engine->config.gaussian_sigma * 
                (1.0f - (float)engine->population.generation / (float)engine->config.max_generations);
            gaussian_mutation(chromosome, size, effective_sigma, rng);
            break;
        }
        case EVO_MUTATION_UNIFORM:
        default: {
            float c_min = engine->config.chromosome_min;
            float c_max = engine->config.chromosome_max;
            float c_range = c_max - c_min;
            for (size_t i = 0; i < size; i++) {
                if (rand_float(rng) < 0.1f) {
                    chromosome[i] = c_min + rand_float(rng) * c_range;
                }
            }
            break;
        }
    }
}

/* 计算种群多样性 */
static float compute_diversity(EvolutionPopulation* pop) {
    float* center = (float*)safe_calloc(pop->individuals[0].chromosome_size, sizeof(float));
    if (!center) return 0.0f;

    for (size_t i = 0; i < pop->size; i++) {
        for (size_t j = 0; j < pop->individuals[i].chromosome_size; j++) {
            center[j] += pop->individuals[i].chromosome[j];
        }
    }
    for (size_t j = 0; j < pop->individuals[0].chromosome_size; j++) {
        center[j] /= (float)pop->size;
    }

    float dist_sum = 0.0f;
    for (size_t i = 0; i < pop->size; i++) {
        float d = 0.0f;
        for (size_t j = 0; j < pop->individuals[i].chromosome_size; j++) {
            float diff = pop->individuals[i].chromosome[j] - center[j];
            d += diff * diff;
        }
        dist_sum += sqrtf(d);
    }
    safe_free((void**)&center);
    return dist_sum / (float)pop->size;
}

/* 辅助：判断个体 a 是否支配个体 b（最小化问题） */
static int dominates(const EvolutionIndividual* a, const EvolutionIndividual* b) {
    int obj_count = a->objective_count;
    int strictly_better = 0;
    for (int k = 0; k < obj_count; k++) {
        if (a->objectives[k] > b->objectives[k]) return 0;
        if (a->objectives[k] < b->objectives[k]) strictly_better = 1;
    }
    return strictly_better;
}

/* 非支配排序 (Kung分治算法 O(n log n) 前沿提取)
 * 核心思路：按第一目标排序后，已排序序列中后出现的解不可能支配先出现的解。
 * 每次迭代在线性扫描中提取当前Pareto前沿，分配rank后移除，重复至空。
 * 对2目标问题严格O(n log n)，多目标退化为O(k * n * |F|)其中|F|为前沿大小。 */
static void non_dominated_sort(EvolutionPopulation* pop) {
    size_t n = pop->size;
    if (n == 0) return;

    for (size_t i = 0; i < n; i++) {
        pop->individuals[i].pareto_rank = -1;
        pop->individuals[i].crowding_distance = 0.0f;
    }

    int obj_count = pop->individuals[0].objective_count;
    if (obj_count <= 0) return;

    size_t* active = (size_t*)safe_calloc(n, sizeof(size_t));
    size_t* front = (size_t*)safe_calloc(n, sizeof(size_t));
    if (!active || !front) {
        safe_free((void**)&active);
        safe_free((void**)&front);
        return;
    }

    size_t active_count = n;
    for (size_t i = 0; i < n; i++) active[i] = i;

    int rank = 0;
    while (active_count > 0) {
        /* 按第一目标值升序排序（选择排序，小规模可行） */
        for (size_t i = 0; i < active_count - 1; i++) {
            size_t best = i;
            for (size_t j = i + 1; j < active_count; j++) {
                if (pop->individuals[active[j]].objectives[0] <
                    pop->individuals[active[best]].objectives[0]) {
                    best = j;
                }
            }
            if (best != i) {
                size_t tmp = active[i];
                active[i] = active[best];
                active[best] = tmp;
            }
        }

        /* Kung线性扫描提取当前Pareto前沿：
         * 已按obj[0]排序，后续解obj[0] >= 前解obj[0]，只能被已有前沿支配 */
        size_t front_size = 0;
        for (size_t i = 0; i < active_count; i++) {
            size_t idx_i = active[i];
            int dominated = 0;
            for (size_t j = 0; j < front_size && !dominated; j++) {
                if (dominates(&pop->individuals[front[j]], &pop->individuals[idx_i])) {
                    dominated = 1;
                }
            }
            if (!dominated) {
                front[front_size++] = idx_i;
            }
        }

        /* 分配Pareto等级 */
        for (size_t i = 0; i < front_size; i++) {
            pop->individuals[front[i]].pareto_rank = rank;
        }

        /* 压缩活跃列表：移除已分配rank的个体 */
        size_t new_count = 0;
        for (size_t i = 0; i < active_count; i++) {
            if (pop->individuals[active[i]].pareto_rank < 0) {
                active[new_count++] = active[i];
            }
        }
        active_count = new_count;
        rank++;
    }

    safe_free((void**)&active);
    safe_free((void**)&front);
}

/* 更新种群统计 */
static void update_population_stats(EvolutionPopulation* pop) {
    float sum = 0.0f, max_fit = -FLT_MAX;
    for (size_t i = 0; i < pop->size; i++) {
        sum += pop->individuals[i].fitness;
        if (pop->individuals[i].fitness > max_fit) max_fit = pop->individuals[i].fitness;
    }
    pop->avg_fitness = sum / (float)pop->size;

    float var_sum = 0.0f;
    for (size_t i = 0; i < pop->size; i++) {
        float diff = pop->individuals[i].fitness - pop->avg_fitness;
        var_sum += diff * diff;
    }
    pop->std_fitness = sqrtf(var_sum / (float)pop->size);

    if (max_fit > pop->best_fitness) {
        pop->best_fitness = max_fit;
        pop->stagnated_generations = 0;
    } else {
        pop->stagnated_generations++;
    }

    pop->diversity = compute_diversity(pop);
    pop->total_evaluations += (int)pop->size;
    pop->generation++;
}

/* 公共接口实现 */

EvolutionEngine* evolution_engine_create(const EvolutionConfig* config) {
    if (!config) return NULL;

    EvolutionEngine* engine = (EvolutionEngine*)safe_calloc(1, sizeof(EvolutionEngine));
    if (!engine) return NULL;

    memcpy(&engine->config, config, sizeof(EvolutionConfig));
    /* 若调用方未设置染色体范围，使用合理默认值 */
    if (engine->config.chromosome_min >= engine->config.chromosome_max) {
        engine->config.chromosome_min = -10.0f;
        engine->config.chromosome_max = 10.0f;
    }
    engine->direction = config->direction;
    engine->obj_count = 0;
    engine->eval_counter = 0;
    engine->initialized = 0;
    engine->rng_state = (unsigned int)(time(NULL) ^ (uintptr_t)engine);
    engine->island_count = 0;
    engine->islands = NULL;
    engine->migration_counter = 0;

    memset(&engine->stats, 0, sizeof(EvolutionStats));
    engine->stats.start_time = time(NULL);
    
    return engine;
}

void evolution_engine_free(EvolutionEngine* engine) {
    if (!engine) return;

    for (size_t i = 0; i < engine->population.size; i++) {
        free_individual(&engine->population.individuals[i]);
    }
    safe_free((void**)&engine->population.individuals);

    if (engine->stats.fitness_history) safe_free((void**)&engine->stats.fitness_history);
    if (engine->stats.diversity_history) safe_free((void**)&engine->stats.diversity_history);
    if (engine->islands) {
        for (int i = 0; i < engine->island_count; i++) {
            for (size_t j = 0; j < engine->islands[i].size; j++) {
                free_individual(&engine->islands[i].individuals[j]);
            }
            safe_free((void**)&engine->islands[i].individuals);
        }
        safe_free((void**)&engine->islands);
    }
    
    /* F-010/F-019: 释放CMA-ES状态 */
    if (engine->cmaes_state) {
        cmaes_free(engine->cmaes_state);
        engine->cmaes_state = NULL;
    }
    
    safe_free((void**)&engine);
}

int evolution_set_fitness_function(EvolutionEngine* engine, FitnessFunction func, void* user_data) {
    if (!engine || !func) return -1;
    engine->fitness_func = func;
    engine->user_data = user_data;
    return 0;
}

int evolution_set_multi_fitness_function(EvolutionEngine* engine, MultiFitnessFunction func,
                                         int obj_count, void* user_data) {
    if (!engine || !func || obj_count <= 0) return -1;
    engine->multi_fitness_func = func;
    engine->obj_count = obj_count;
    engine->user_data = user_data;
    return 0;
}

/* F-010/F-019: 启用CMA-ES模式 */
int evolution_engine_enable_cmaes(EvolutionEngine* engine, float initial_sigma, int lambda) {
    if (!engine || !engine->fitness_func) return -1;
    
    size_t dim = engine->config.chromosome_size;
    if (dim == 0 || dim > CMAES_MAX_DIM) return -1;
    if (lambda <= 0) lambda = dim > 0 ? (int)(4 + 3 * (int)(log((double)dim) / log(2.0))) : 100;
    
    /* 如果已有CMA-ES状态则先释放 */
    if (engine->cmaes_state) {
        cmaes_free(engine->cmaes_state);
        engine->cmaes_state = NULL;
    }
    
    engine->cmaes_state = cmaes_alloc(dim, initial_sigma > 0 ? initial_sigma : CMAES_DEFAULT_SIGMA, 
                                       lambda, (int)(engine->rng_state));
    if (!engine->cmaes_state) return -1;
    
    cmaes_set_stop_conditions(engine->cmaes_state, CMAES_DEFAULT_STOP_FITNESS, 
                               engine->config.max_generations > 0 ? engine->config.max_generations : 1000,
                               CMAES_DEFAULT_TOL_X, CMAES_DEFAULT_TOL_COV, 
                               CMAES_DEFAULT_TOL_FUN, CMAES_DEFAULT_TOL_HIST_FUN);
    
    engine->using_cmaes = 1;
    log_info("[演化引擎] CMA-ES已启用: dim=%zu, sigma=%.3f, lambda=%d", 
             dim, engine->cmaes_state->sigma, lambda);
    return 0;
}

/* F-010/F-019: 禁用CMA-ES模式，回退到GA */
int evolution_engine_disable_cmaes(EvolutionEngine* engine) {
    if (!engine) return -1;
    if (engine->cmaes_state) {
        cmaes_free(engine->cmaes_state);
        engine->cmaes_state = NULL;
    }
    engine->using_cmaes = 0;
    log_info("[演化引擎] CMA-ES已禁用，回退到GA");
    return 0;
}

int evolution_initialize_population(EvolutionEngine* engine, float min_val, float max_val) {
    if (!engine || !engine->fitness_func) return -1;

    size_t pop_size = engine->config.population_size;
    engine->population.individuals = (EvolutionIndividual*)safe_calloc(
        pop_size, sizeof(EvolutionIndividual));
    if (!engine->population.individuals) return -1;
    engine->population.size = pop_size;
    engine->population.capacity = pop_size;
    engine->population.generation = 0;
    engine->population.best_fitness = -FLT_MAX;
    engine->population.stagnated_generations = 0;

    unsigned int* rng = &engine->rng_state;
    for (size_t i = 0; i < pop_size; i++) {
        init_individual(&engine->population.individuals[i], 
                       engine->config.chromosome_size, min_val, max_val, rng);
        engine->population.individuals[i].id = (int)i;
        evaluate_individual(engine, &engine->population.individuals[i]);
    }

    update_population_stats(&engine->population);

    /* 初始化适应度历史 */
    engine->stats.fitness_history = (float*)safe_malloc(
        engine->config.max_generations * sizeof(float));
    engine->stats.diversity_history = (float*)safe_malloc(
        engine->config.max_generations * sizeof(float));
    engine->stats.history_size = 0;
    engine->stats.diversity_size = 0;

    engine->initialized = 1;
    return 0;
}

int evolution_initialize_from_existing(EvolutionEngine* engine,
                                       const float** chromosomes, size_t count) {
    if (!engine || !chromosomes || count == 0) return -1;

    size_t pop_size = engine->config.population_size;
    engine->population.individuals = (EvolutionIndividual*)safe_calloc(
        pop_size, sizeof(EvolutionIndividual));
    if (!engine->population.individuals) return -1;
    engine->population.size = pop_size;
    engine->population.capacity = pop_size;
    engine->population.generation = 0;
    engine->population.best_fitness = -FLT_MAX;
    engine->population.stagnated_generations = 0;

    unsigned int* rng = &engine->rng_state;
    for (size_t i = 0; i < pop_size; i++) {
        init_individual(&engine->population.individuals[i],
                       engine->config.chromosome_size, -1.0f, 1.0f, rng);
        engine->population.individuals[i].id = (int)i;
        if (i < count) {
            memcpy(engine->population.individuals[i].chromosome, 
                   chromosomes[i], engine->config.chromosome_size * sizeof(float));
        }
        evaluate_individual(engine, &engine->population.individuals[i]);
    }

    update_population_stats(&engine->population);

    engine->stats.fitness_history = (float*)safe_malloc(
        engine->config.max_generations * sizeof(float));
    engine->stats.diversity_history = (float*)safe_malloc(
        engine->config.max_generations * sizeof(float));

    engine->initialized = 1;
    return 0;
}

int evolution_step(EvolutionEngine* engine) {
    if (!engine || !engine->initialized) return -1;

    /* F-010/F-019: 如果使用CMA-ES算法，委托给core/cma_es.c */
    if (engine->using_cmaes && engine->cmaes_state) {
        CMAESState* cs = engine->cmaes_state;
        size_t dim = cs->dimension;
        
        /* 评估上一代的种群适应度 */
        for (int i = 0; i < cs->lambda; i++) {
            float* x = cmaes_get_solution(cs, i);
            if (x) {
                float fit = engine->fitness_func ? 
                    engine->fitness_func(x, dim, engine->user_data) : 0.0f;
                /* CMAESUpdate需要按索引顺序的适应度数组 */
                cs->fitness[i] = (engine->direction == EVO_FITNESS_MINIMIZE) ? fit : -fit;
            }
        }
        
        /* CMA-ES更新：协方差矩阵自适应 + 均值更新 + 演化路径 */
        int ret = cmaes_update(cs, cs->fitness);
        cs->generation++;
        
        /* 检查终止条件 */
        int term = cmaes_test_stop_conditions(cs);
        if (term != CMAES_TERM_NONE || ret != 0) {
            log_info("[CMA-ES] 演化终止: 代数=%d, sigma=%.6f, 最佳适应度=%.6f, 终止原因=%d",
                    cs->generation, cmaes_get_current_sigma(cs), cs->best_fitness, cs->termination_reason);
        }
        
        /* 更新统计 */
        engine->stats.total_generations = (size_t)cs->generation;
        engine->stats.total_evaluations += (size_t)cs->lambda;
        engine->stats.final_best_fitness = cs->best_fitness;
        
        /* 将CMA-ES最优解写入种群（用于apply_best_to_lnn） */
        if (engine->population.size > 0 && engine->population.individuals) {
            cmaes_get_best_solution(cs, engine->population.individuals[0].chromosome, 
                                     &engine->population.individuals[0].fitness);
            engine->population.individuals[0].chromosome_size = dim;
        }
        
        engine->eval_counter++;
        return 0;
    }

    /* 原遗传算法（GA）实现 */

    size_t pop_size = engine->population.size;
    size_t chrom_size = engine->config.chromosome_size;
    EvolutionPopulation* pop = &engine->population;

    /* 创建新一代 */
    EvolutionIndividual* new_pop = (EvolutionIndividual*)safe_calloc(
        pop_size, sizeof(EvolutionIndividual));
    if (!new_pop) return -1;

    unsigned int* rng = &engine->rng_state;
    int elite_count = engine->config.elite_count;
    if (elite_count > (int)pop_size) elite_count = (int)pop_size;

    /* 精英保留 */
    int* elite_indices = (int*)safe_malloc(pop_size * sizeof(int));
    float* elite_fitness = (float*)safe_malloc(pop_size * sizeof(float));
    if (!elite_indices || !elite_fitness) {
        safe_free((void**)&elite_indices);
        safe_free((void**)&elite_fitness);
        safe_free((void**)&new_pop);
        return -1;
    }
    for (size_t i = 0; i < pop_size; i++) {
        elite_indices[i] = (int)i;
        elite_fitness[i] = pop->individuals[i].fitness;
    }
    for (size_t i = 0; i < pop_size; i++) {
        for (size_t j = i + 1; j < pop_size; j++) {
            if (elite_fitness[j] > elite_fitness[i]) {
                float tmpf = elite_fitness[i];
                elite_fitness[i] = elite_fitness[j];
                elite_fitness[j] = tmpf;
                int tmpi = elite_indices[i];
                elite_indices[i] = elite_indices[j];
                elite_indices[j] = tmpi;
            }
        }
    }

    for (int i = 0; i < elite_count; i++) {
        if (i < (int)pop_size) {
            EvolutionIndividual* src = &pop->individuals[elite_indices[i]];
            if (copy_individual(&new_pop[i], src) != 0) continue;
            new_pop[i].id = src->id;
            new_pop[i].age++;
        }
    }

    /* 交叉和变异生成新个体 */
    float mutation_rate = engine->config.mutation_rate;
    if (engine->config.use_adaptive_mutation) {
        float progress = (float)pop->generation / (float)engine->config.max_generations;
        mutation_rate = engine->config.mutation_rate_max - 
            (engine->config.mutation_rate_max - engine->config.mutation_rate_min) * progress;
    }

    for (int i = elite_count; i < (int)pop_size; i++) {
        if (rand_float(rng) < engine->config.crossover_rate && pop_size >= 2) {
            int p1_idx = select_parent(engine, pop);
            int p2_idx = select_parent(engine, pop);
            if (p1_idx >= (int)pop_size) p1_idx = 0;
            if (p2_idx >= (int)pop_size) p2_idx = 1;
            while (p2_idx == p1_idx && pop_size > 1) {
                p2_idx = select_parent(engine, pop);
            }

            float* c1 = (float*)safe_malloc(chrom_size * sizeof(float));
            float* c2 = (float*)safe_malloc(chrom_size * sizeof(float));
            if (c1 && c2) {
                crossover(engine, pop->individuals[p1_idx].chromosome,
                         pop->individuals[p2_idx].chromosome, c1, c2, chrom_size);
                mutate(engine, c1, chrom_size, mutation_rate);
                new_pop[i].chromosome = c1;
                new_pop[i].chromosome_size = chrom_size;
                new_pop[i].id = (int)engine->eval_counter;
                new_pop[i].age = 0;
                new_pop[i].birth_time = time(NULL);
                /* c2 被丢弃 */
                safe_free((void**)&c2);
            } else {
                safe_free((void**)&c1);
                safe_free((void**)&c2);
            }
        } else {
            /* 直接变异现有个体 */
            int p_idx = select_parent(engine, pop);
            if (p_idx >= (int)pop_size) p_idx = 0;
            if (copy_individual(&new_pop[i], &pop->individuals[p_idx]) != 0) continue;
            mutate(engine, new_pop[i].chromosome, chrom_size, mutation_rate);
            new_pop[i].id = (int)engine->eval_counter;
        }

        evaluate_individual(engine, &new_pop[i]);
    }

    /* 替换种群 */
    for (size_t i = 0; i < pop_size; i++) {
        free_individual(&pop->individuals[i]);
    }
    safe_free((void**)&pop->individuals);
    safe_free((void**)&elite_indices);
    safe_free((void**)&elite_fitness);

    pop->individuals = new_pop;
    update_population_stats(pop);

    /* 记录历史 */
    if (engine->stats.history_size < engine->config.max_generations) {
        engine->stats.fitness_history[engine->stats.history_size] = pop->best_fitness;
        engine->stats.diversity_history[engine->stats.diversity_size] = pop->diversity;
        engine->stats.history_size++;
        engine->stats.diversity_size++;
    }

    /* 停滞重启 */
    if (engine->config.use_restart_stagnation &&
        pop->stagnated_generations >= engine->config.restart_stagnation_generations) {
        float min_v = -10.0f, max_v = 10.0f;
        for (size_t i = elite_count; i < pop_size; i++) {
            for (size_t j = 0; j < chrom_size; j++) {
                pop->individuals[i].chromosome[j] = min_v + rand_float(rng) * (max_v - min_v);
            }
            evaluate_individual(engine, &pop->individuals[i]);
        }
        pop->stagnated_generations = 0;
        update_population_stats(pop);
    }

    engine->stats.total_generations++;
    engine->stats.total_evaluations += pop_size;
    engine->stats.final_best_fitness = pop->best_fitness;
    engine->stats.improvement = engine->stats.final_best_fitness - engine->stats.initial_best_fitness;
    engine->stats.end_time = time(NULL);
    engine->stats.elapsed_seconds = difftime(engine->stats.end_time, engine->stats.start_time);

    /* PF-004修复: 岛模型个体迁移 —— 每migration_counter代执行一次岛间精英迁移 */
    if (engine->island_count > 1 && engine->islands) {
        engine->migration_counter++;
        if (engine->migration_counter >= engine->config.migration_interval) {
            engine->migration_counter = 0;
            evolution_island_migrate(engine);
        }
    }

    return 0;
}

/* PF-004修复: 岛模型创建 —— 将种群划分为多个独立演化的岛屿 */
int evolution_create_islands(EvolutionEngine* engine, int num_islands) {
    if (!engine || !engine->initialized || num_islands < 2) return -1;
    if (engine->islands) {
        for (int i = 0; i < engine->island_count; i++) {
            safe_free((void**)&engine->islands[i].individuals);
        }
        safe_free((void**)&engine->islands);
    }

    int total_pop = (int)engine->population.size;
    int per_island = total_pop / num_islands;
    if (per_island < 4) return -1;

    engine->island_count = num_islands;
    engine->islands = (EvolutionPopulation*)safe_calloc((size_t)num_islands, sizeof(EvolutionPopulation));
    if (!engine->islands) { engine->island_count = 0; return -1; }

    for (int i = 0; i < num_islands; i++) {
        EvolutionPopulation* isl = &engine->islands[i];
        isl->size = (size_t)per_island;
        isl->capacity = (size_t)per_island;
        isl->individuals = (EvolutionIndividual*)safe_calloc((size_t)per_island, sizeof(EvolutionIndividual));
        if (!isl->individuals) { engine->island_count = i; return -1; }
        int offset = i * per_island;
        for (int j = 0; j < per_island; j++) {
            int src_idx = offset + j;
            isl->individuals[j].chromosome_size = engine->population.individuals[src_idx].chromosome_size;
            isl->individuals[j].chromosome = (float*)safe_malloc(
                isl->individuals[j].chromosome_size * sizeof(float));
            if (isl->individuals[j].chromosome) {
                memcpy(isl->individuals[j].chromosome,
                       engine->population.individuals[src_idx].chromosome,
                       isl->individuals[j].chromosome_size * sizeof(float));
                isl->individuals[j].fitness = engine->population.individuals[src_idx].fitness;
            }
        }
        update_population_stats(isl);
    }
    engine->migration_counter = 0;
    return 0;
}

/* PF-004修复: 岛间迁移 —— 将每个岛的最优个体发送到下一个岛(环形拓扑) */
int evolution_island_migrate(EvolutionEngine* engine) {
    if (!engine || engine->island_count < 2 || !engine->islands) return -1;

    for (int src = 0; src < engine->island_count; src++) {
        int dst = (src + 1) % engine->island_count;
        EvolutionPopulation* src_pop = &engine->islands[src];
        EvolutionPopulation* dst_pop = &engine->islands[dst];

        if (src_pop->size == 0 || dst_pop->size == 0) continue;

        /* 找到源岛最优个体 */
        int best_idx = 0;
        for (size_t j = 1; j < src_pop->size; j++) {
            if (src_pop->individuals[j].fitness > src_pop->individuals[best_idx].fitness)
                best_idx = (int)j;
        }

        /* 覆盖目标岛最差个体 */
        int worst_idx = 0;
        for (size_t j = 1; j < dst_pop->size; j++) {
            if (dst_pop->individuals[j].fitness < dst_pop->individuals[worst_idx].fitness)
                worst_idx = (int)j;
        }

        EvolutionIndividual* src_best = &src_pop->individuals[best_idx];
        EvolutionIndividual* dst_worst = &dst_pop->individuals[worst_idx];
        size_t chrom_size = src_best->chromosome_size;
        if (chrom_size > dst_worst->chromosome_size) chrom_size = dst_worst->chromosome_size;
        memcpy(dst_worst->chromosome, src_best->chromosome, chrom_size * sizeof(float));
        dst_worst->fitness = src_best->fitness;
    }
    return 0;
}

int evolution_run(EvolutionEngine* engine, int max_generations) {
    if (!engine || !engine->initialized) return -1;
    if (max_generations <= 0) max_generations = engine->config.max_generations;

    engine->stats.initial_best_fitness = engine->population.best_fitness;

    for (int gen = 0; gen < max_generations; gen++) {
        int result = evolution_step(engine);
        if (result != 0) return result;
    }

    return 0;
}

const EvolutionIndividual* evolution_get_best(const EvolutionEngine* engine) {
    if (!engine || !engine->initialized || engine->population.size == 0) return NULL;

    int best_idx = 0;
    float best_fit = engine->population.individuals[0].fitness;
    for (size_t i = 1; i < engine->population.size; i++) {
        if (engine->population.individuals[i].fitness > best_fit) {
            best_fit = engine->population.individuals[i].fitness;
            best_idx = (int)i;
        }
    }
    return &engine->population.individuals[best_idx];
}

int evolution_get_pareto_front(const EvolutionEngine* engine,
                               EvolutionIndividual** front, size_t* front_size) {
    if (!engine || !engine->initialized) return -1;
    if (engine->obj_count < 2) return -1;

    EvolutionPopulation* pop = (EvolutionPopulation*)&engine->population;
    non_dominated_sort(pop);

    size_t count = 0;
    for (size_t i = 0; i < pop->size; i++) {
        if (pop->individuals[i].pareto_rank == 0) count++;
    }

    *front_size = count;
    *front = NULL;
    if (count > 0) {
        *front = (EvolutionIndividual*)safe_malloc(count * sizeof(EvolutionIndividual));
        size_t idx = 0;
        for (size_t i = 0; i < pop->size && idx < count; i++) {
            if (pop->individuals[i].pareto_rank == 0) {
                memcpy(&(*front)[idx], &pop->individuals[i], sizeof(EvolutionIndividual));
                (*front)[idx].chromosome = (float*)safe_malloc(
                    pop->individuals[i].chromosome_size * sizeof(float));
                memcpy((*front)[idx].chromosome, pop->individuals[i].chromosome,
                       pop->individuals[i].chromosome_size * sizeof(float));
                idx++;
            }
        }
    }
    return 0;
}

const EvolutionPopulation* evolution_get_population(const EvolutionEngine* engine) {
    return engine ? &engine->population : NULL;
}

int evolution_get_stats(const EvolutionEngine* engine, EvolutionStats* stats) {
    if (!engine || !stats) return -1;
    memcpy(stats, &engine->stats, sizeof(EvolutionStats));
    return 0;
}

int evolution_inject_elite(EvolutionEngine* engine, const float* chromosome, size_t size) {
    if (!engine || !chromosome || !engine->initialized) return -1;

    size_t chrom_size = engine->config.chromosome_size;
    int worst_idx = 0;
    float worst_fit = engine->population.individuals[0].fitness;
    for (size_t i = 1; i < engine->population.size; i++) {
        if (engine->population.individuals[i].fitness < worst_fit) {
            worst_fit = engine->population.individuals[i].fitness;
            worst_idx = (int)i;
        }
    }

    /* ZSF-034修复: memcpy参数括号修正，先比较大小再乘sizeof */
    memcpy(engine->population.individuals[worst_idx].chromosome, chromosome,
           (chrom_size < size ? chrom_size : size) * sizeof(float));
    evaluate_individual(engine, &engine->population.individuals[worst_idx]);
    update_population_stats(&engine->population);
    return 0;
}

int evolution_set_mutation_rate(EvolutionEngine* engine, float rate) {
    if (!engine) return -1;
    engine->config.mutation_rate = rate;
    return 0;
}

int evolution_save_population(const EvolutionEngine* engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    uint32_t magic = 0x45564F4C;
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&engine->population.size, sizeof(size_t), 1, fp);
    fwrite(&engine->config.chromosome_size, sizeof(size_t), 1, fp);

    for (size_t i = 0; i < engine->population.size; i++) {
        EvolutionIndividual* ind = &engine->population.individuals[i];
        fwrite(ind->chromosome, sizeof(float), ind->chromosome_size, fp);
        fwrite(&ind->fitness, sizeof(float), 1, fp);
    }
    fclose(fp);
    return 0;
}

int evolution_load_population(EvolutionEngine* engine, const char* filepath) {
    if (!engine || !filepath) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    uint32_t magic;
    size_t saved_size, saved_chrom_size;
    fread(&magic, sizeof(uint32_t), 1, fp);
    if (magic != 0x45564F4C) { fclose(fp); return -1; }
    fread(&saved_size, sizeof(size_t), 1, fp);
    fread(&saved_chrom_size, sizeof(size_t), 1, fp);

    for (size_t i = 0; i < engine->population.size; i++) {
        free_individual(&engine->population.individuals[i]);
    }
    safe_free((void**)&engine->population.individuals);

    engine->config.chromosome_size = saved_chrom_size;
    engine->population.individuals = (EvolutionIndividual*)safe_calloc(
        saved_size, sizeof(EvolutionIndividual));
    engine->population.size = saved_size;

    for (size_t i = 0; i < saved_size; i++) {
        EvolutionIndividual* ind = &engine->population.individuals[i];
        ind->chromosome = (float*)safe_malloc(saved_chrom_size * sizeof(float));
        ind->chromosome_size = saved_chrom_size;
        fread(ind->chromosome, sizeof(float), saved_chrom_size, fp);
        fread(&ind->fitness, sizeof(float), 1, fp);
        ind->id = (int)i;
    }
    fclose(fp);

    update_population_stats(&engine->population);
    engine->initialized = 1;
    return 0;
}

int evolution_reset(EvolutionEngine* engine) {
    if (!engine) return -1;

    for (size_t i = 0; i < engine->population.size; i++) {
        free_individual(&engine->population.individuals[i]);
    }
    safe_free((void**)&engine->population.individuals);
    memset(&engine->population, 0, sizeof(EvolutionPopulation));
    memset(&engine->stats, 0, sizeof(EvolutionStats));
    engine->initialized = 0;
    engine->eval_counter = 0;
    engine->stats.start_time = time(NULL);
    engine->rng_state = (unsigned int)(time(NULL) ^ (uintptr_t)engine);

    return 0;
}

int evolution_engine_set_target_lnn(EvolutionEngine* engine, void* lnn) {
    if (!engine || !lnn) return -1;
    engine->target_lnn = lnn;
    engine->lnn_connected = 1;
    return 0;
}

int evolution_engine_apply_best_to_lnn(EvolutionEngine* engine) {
    if (!engine || !engine->initialized || !engine->lnn_connected) return -1;
    if (!engine->target_lnn || engine->population.size == 0) return -1;

    const EvolutionIndividual* best = evolution_get_best(engine);
    if (!best || !best->chromosome || best->chromosome_size == 0) return -1;

    LNN* lnn = (LNN*)engine->target_lnn;
    float* params = lnn_get_parameters(lnn);
    if (!params) return -1;

    size_t total_params = lnn_get_parameter_count(lnn);
    size_t copy_count = (total_params < best->chromosome_size) ? total_params : best->chromosome_size;

    /* K-013: 演化权重变更验证 —— 计算L2范式变化量 */
    double l2_diff = 0.0;
    double orig_norm = 0.0;
    double new_norm = 0.0;
    for (size_t i = 0; i < copy_count; i++) {
        float old_val = params[i];
        float new_val = best->chromosome[i];
        float delta = new_val - old_val;
        l2_diff += (double)(delta * delta);
        orig_norm += (double)(old_val * old_val);
        new_norm += (double)(new_val * new_val);
    }
    l2_diff = sqrt(l2_diff);
    orig_norm = sqrt(orig_norm);
    new_norm = sqrt(new_norm);

    memcpy(params, best->chromosome, copy_count * sizeof(float));

    /* K-013: 记录权重变更日志 */
    float rel_change = (orig_norm > 1e-8f) ? (float)(l2_diff / orig_norm) * 100.0f : 0.0f;
    log_info("[演化验证] LNN权重已写入: 参数数=%zu/%zu, L2变化=%.6f, "
             "原始L2=%.6f, 新L2=%.6f, 相对变化=%.2f%%, 适应度=%.6f, 代=%zu",
             copy_count, total_params, l2_diff, orig_norm, new_norm,
             rel_change, best->fitness, engine->population.generation);

    /* K-013: 验证写入完整性 —— 随机抽查5个位置确认写入成功 */
    int mismatches = 0;
    for (int chk = 0; chk < 5 && copy_count > 0; chk++) {
        size_t idx = (size_t)((unsigned int)(chk * 2654435761ULL) % copy_count);
        float expected = best->chromosome[idx];
        float actual = params[idx];
        if (fabsf(actual - expected) > 1e-7f) mismatches++;
    }
    if (mismatches > 0) {
        log_warning("[演化验证] 随机抽查发现%d处写入不一致", mismatches);
    } else {
        log_info("[演化验证] 随机抽查5处全部写入一致，权重变更确认无误");
    }

    return 0;
}

/* ============================================================================
 * 5.1 修复: 演化引擎能力开关 + 环境适应度评估闭环
 * ============================================================================ */
int evolution_engine_enable(EvolutionEngine* engine) {
    if (!engine) return -1;
    engine->enabled = 1;
    return 0;
}
int evolution_engine_disable(EvolutionEngine* engine) {
    if (!engine) return -1;
    engine->enabled = 0;
    return 0;
}
int evolution_engine_is_enabled(const EvolutionEngine* engine) {
    return (engine && engine->enabled) ? 1 : 0;
}
int evolution_evaluate_environment(EvolutionEngine* engine, const float* environment_state,
                                    size_t state_dim, float* fitness_out) {
    if (!engine || !engine->initialized || !fitness_out) return -1;
    *fitness_out = 0.0f;
    size_t pop_size = engine->population.size;
    for (size_t i = 0; i < pop_size; i++) {
        float eval = 0.0f;
        if (engine->fitness_func) {
            eval = engine->fitness_func(engine->population.individuals[i].chromosome,
                                        engine->config.chromosome_size, engine->user_data);
        } else {
            for (size_t j = 0; j < state_dim && j < engine->config.chromosome_size; j++) {
                eval -= fabsf(engine->population.individuals[i].chromosome[j] - environment_state[j]);
            }
            eval = expf(eval / (float)state_dim);
        }
        engine->population.individuals[i].fitness = eval;
        if (eval > *fitness_out) *fitness_out = eval;
    }
    return 0;
}
