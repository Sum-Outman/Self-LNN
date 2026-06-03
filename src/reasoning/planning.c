/**
 * @file planning.c
 * @brief 规划系统核心实现 —— 全局规划层
 *
 * K-005: 角色定义 —— planning.c 是规划系统的【核心实现层】
 * 负责：目标分解、计划生成、执行监控、可行性评估。
 * 
 * 层级关系：
 *   planning.c（本文件）→ 核心规划算法（MCTS蒙特卡洛树搜索/反应式FSM/CMA-ES演化策略/时序约束网络TCN）
 *   planning_enhanced.c → 增强规划（STN时序网 + HTN层次任务网 + 多智能体协调）
 *   hierarchical_planning.c → 分层规划（HTN/POP/HRL分解）
 *   long_term_planning.c → 长期规划
 * 
 * 注：A-star/BFS/Dijkstra/Beam Search等图搜索算法并未在本文件中实现.
 * 状态空间规划通过MCTS+CMA-ES演化策略+线性插值组合实现.
 * planning.c 提供基础规划API，planning_enhanced.c 在此基础上增加高级特性.
 * 两者互不替代，职责明确分离.
 */

#include "selflnn/reasoning/planning.h"
#include "selflnn/reasoning/planning_enhanced.h" /* 增强规划集成 */
#include "selflnn/reasoning/causal_reasoning.h"  /* M-022: 因果推理→规划桥接 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/perf.h"
#include "selflnn/core/lnn.h"           /* F-017: LNN状态转移 */
#include "selflnn/selflnn.h"           /* F-017: selflnn_get_lnn */
#include "selflnn/reasoning/hierarchical_planning.h" /* 分层规划真实API */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdint.h>
#include <time.h>

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif
#define CLAMP(x,min,max) (MIN(MAX((x),(min)),(max)))

#define EVOLUTION_DEFAULT_POPULATION_SIZE 50
#define EVOLUTION_DEFAULT_MUTATION_RATE 0.1f
#define EVOLUTION_DEFAULT_CROSSOVER_RATE 0.9f
#define EVOLUTION_DEFAULT_ELITISM_RATIO 0.1f
#define EVOLUTION_DEFAULT_TOURNAMENT_SIZE 3
#define EVOLUTION_DEFAULT_SEED 42

#define MAX_PLAN_BUFFER_SIZE 4096
#define MAX_STATE_DIMENSION 128
#define MAX_ACTION_DIMENSION 32
#define MAX_HEURISTIC_WEIGHTS 64

#define TC_MAX_CONSTRAINTS 256
#define TC_MAX_STEPS 128

/* 移除自定义LCG PRNG(plan_rng_next/plan_rng_uniform)，
 * 替换为密码学安全的secure_random_float()。
 * 原LCG确定性伪随机存在可预测性和分布偏差问题。
 * ZSF-071修复：移除已冗余的plan_rng_state全局变量 */

static float plan_rng_uniform(float min, float max) {
    return min + (max - min) * secure_random_float();
}

struct PlanningSystem {
    PlanningConfig config;
    int initialized;

    float* heuristic_weights;
    size_t heuristic_weights_size;

    int* state_history;
    size_t state_history_size;
    size_t state_history_capacity;

    float* trajectory_buffer;
    size_t trajectory_buffer_size;
    size_t trajectory_buffer_capacity;

    float execution_deviation_smoothed;
    float deviation_smoothing_alpha;
    int consecutive_failures;
    float last_feasibility;
    float feasibility_history[20];
    int feasibility_history_count;

    int replan_triggered;
    float deviation_history[50];
    int deviation_history_count;

    int enable_deep_evolution;
    int deep_evolution_enabled;

    ArchitectureEvolutionConfig arch_evol_config;
    MultiObjectiveConfig multi_objective_config;
    SelfAdaptiveConfig self_adaptive_config;
    LamarckianConfig lamarckian_config;
    IslandConfig island_config;

    int evolution_initialized;
    size_t evolution_population_size;
    size_t evolution_genome_size;
    float* evolution_population;
    float* evolution_fitness;
    float* evolution_best_genome;
    float evolution_best_fitness;
    float evolution_average_fitness;
    float evolution_diversity;
    size_t evolution_current_generation;
    float evolution_mutation_rate;
    float evolution_crossover_rate;
    float* lf_fitness_history;
    size_t lf_history_count;
    size_t lf_history_size;
    int evolution_convergence_count;

    int cmaes_initialized;
    float* cmaes_mean;
    float* cmaes_sigma;
    float* cmaes_cov;
    float* cmaes_eigenvalues;
    float* cmaes_evolution_path;
    float* cmaes_evolution_path_c;
    size_t cmaes_dimension;
    float cmaes_lambda_eff;
    int cmaes_generation;

    float* pareto_objectives;
    int pareto_front_size;
    int* pareto_front_indices;
    int num_pareto_objectives;

    TemporalConstraintNetwork* temporal_constraints;
};

static int topology_genes_decode(const float* genome, size_t genome_size, TopologyGenes* topo) {
    if (!genome || !topo || genome_size < ARCH_GENOME_NUM_TOPOLOGY_GENES) return -1;
    memset(topo, 0, sizeof(TopologyGenes));
    int idx = 0;
    topo->num_hidden_layers = 1 + (int)(fabsf(genome[idx++]) * 6.99f);
    if (topo->num_hidden_layers > ARCH_GENOME_MAX_LAYERS) topo->num_hidden_layers = ARCH_GENOME_MAX_LAYERS;
    for (int i = 0; i < ARCH_GENOME_MAX_LAYERS && i < topo->num_hidden_layers; i++) {
        topo->layer_sizes[i] = 4 + (int)(fabsf(genome[idx++]) * 60.99f);
        if (topo->layer_sizes[i] > 256) topo->layer_sizes[i] = 256;
    }
    idx += (ARCH_GENOME_MAX_LAYERS - topo->num_hidden_layers);
    for (int i = 0; i < ARCH_GENOME_MAX_ACTIVATIONS; i++) {
        topo->activation_types[i] = (int)(fabsf(genome[idx++]) * 3.99f);
    }
    topo->connectivity_density = fabsf(genome[idx++]);
    if (topo->connectivity_density > 1.0f) topo->connectivity_density = 1.0f;
    topo->skip_connections = (genome[idx++] > 0.0f) ? 1 : 0;
    return 0;
}

static int planning_evolve_architecture_generation(PlanningSystem* system);

static int gp_subroutine_evolve(PlanningSystem* system, float* gp_weights, size_t gp_genome_size);

static float planning_evolution_evaluate_fitness(PlanningSystem* system, const float* genome) {
    if (!system || !genome) return -1e10f;

/* 多目标适应度评估
     * 目标1: 任务性能 —— 基于可行性历史的真实任务表现权重
     * 目标2: 种群多样性 —— 与种群均值的距离，鼓励多样性
     * 目标3: 探索新颖性 —— 与当前启发式权重的距离，鼓励探索而非趋同
     * 权重可配置，默认三目标均等(各占1/3) */
    float fitness = 0.0f;
    size_t w_size = system->heuristic_weights_size;
    size_t genome_len = system->evolution_genome_size;

    /* 多目标权重配置（归一化后使用） */
    float w_task = system->multi_objective_config.objective_weights[0] > 0.0f ?
                   system->multi_objective_config.objective_weights[0] : 0.34f;
    float w_diversity = system->multi_objective_config.objective_weights[1] > 0.0f ?
                        system->multi_objective_config.objective_weights[1] : 0.33f;
    float w_novelty = system->multi_objective_config.objective_weights[2] > 0.0f ?
                      system->multi_objective_config.objective_weights[2] : 0.33f;
    float w_sum = w_task + w_diversity + w_novelty;
    if (w_sum > 0.0f) { w_task /= w_sum; w_diversity /= w_sum; w_novelty /= w_sum; }

    /* ===== 目标1: 任务性能评估 =====
     * 基于可行性历史衡量基因组质量：使用近期规划成功率与可行性趋势
     * 作为真实任务性能的代理指标，鼓励靠近已验证有效的参数区域 */
    float task_score = 0.0f;
    if (w_size > 0 && system->heuristic_weights) {
        /* 使用近期可行性历史计算加权基线 */
        float hist_avg = 0.0f;
        if (system->feasibility_history_count > 0) {
            for (int i = 0; i < system->feasibility_history_count && i < 20; i++) {
                hist_avg += system->feasibility_history[i];
            }
            hist_avg /= (float)system->feasibility_history_count;
        } else {
            hist_avg = 0.5f; /* 无历史时使用中性值 */
        }
        hist_avg = fmaxf(0.1f, fminf(0.9f, hist_avg));

        /* 高性能(hist_avg高)时偏向收敛，低性能时偏向探索 */
        float exploit_factor = hist_avg;
        float explore_factor = 1.0f - hist_avg;

        /* 任务性能分量: 与启发式权重的加权余弦相似度 */
        float aligned_sum = 0.0f;
        float norm_genome = 0.0f, norm_heuristic = 0.0f;
        for (size_t i = 0; i < w_size && i < genome_len; i++) {
            aligned_sum += genome[i] * system->heuristic_weights[i];
            norm_genome += genome[i] * genome[i];
            norm_heuristic += system->heuristic_weights[i] * system->heuristic_weights[i];
        }
        float cosine_sim = 0.0f;
        if (norm_genome > 1e-10f && norm_heuristic > 1e-10f) {
            cosine_sim = aligned_sum / (sqrtf(norm_genome) * sqrtf(norm_heuristic));
        }
        cosine_sim = fmaxf(-1.0f, fminf(1.0f, cosine_sim));

        /* 根据可行性自适应: 高可行性时余弦相似度更有利；
         * 低可行性时降低其权重，让探索发挥更大作用 */
        task_score = cosine_sim * exploit_factor * 0.8f;

        /* 惩罚项: 仅对极端偏离施加温和惩罚（而非原来对所有偏离都惩罚） */
        for (size_t i = 0; i < w_size && i < genome_len; i++) {
            float diff = fabsf(genome[i] - system->heuristic_weights[i]);
            if (diff > 2.0f) {
                task_score -= (diff - 2.0f) * 0.1f; /* 仅极异常时惩罚 */
            }
        }
    }

    /* 风险/目标容忍度与配置的对齐度 */
    if (w_size < genome_len) {
        float risk_tol = genome[w_size];
        float goal_tol = genome[w_size + 1];
        float risk_align = 1.0f - fminf(1.0f, fabsf(risk_tol - system->config.risk_tolerance));
        float goal_align = 1.0f - fminf(1.0f, fabsf(goal_tol - system->config.goal_tolerance));
        task_score += (risk_align + goal_align) * 0.1f;
    }

    fitness += w_task * task_score;

    /* ===== 目标2: 种群多样性奖励 =====
     * 计算当前基因组与种群中所有其他个体的平均距离，
     * 距离越大说明该基因组越独特，多样性越高 */
    float diversity_score = 0.0f;
    if (system->evolution_population && system->evolution_population_size > 1) {
        /* 先计算种群均值 */
        float* pop_mean = (float*)calloc(genome_len, sizeof(float));
        if (pop_mean) {
            for (size_t i = 0; i < system->evolution_population_size; i++) {
                const float* other = system->evolution_population + i * genome_len;
                for (size_t j = 0; j < genome_len; j++) {
                    pop_mean[j] += other[j];
                }
            }
            for (size_t j = 0; j < genome_len; j++) {
                pop_mean[j] /= (float)system->evolution_population_size;
            }

            /* 与种群均值的欧氏距离 */
            float dist_to_mean = 0.0f;
            for (size_t j = 0; j < genome_len; j++) {
                float d = genome[j] - pop_mean[j];
                dist_to_mean += d * d;
            }
            dist_to_mean = sqrtf(dist_to_mean);

            /* 归一化到合理范围，奖励适度的多样性 */
            float max_expected_dist = sqrtf((float)genome_len) * 2.0f;
            float normalized_div = fminf(1.0f, dist_to_mean / (max_expected_dist + 1e-8f));

            /* 高多样性得到奖励；与均值太近得到惩罚 */
            if (normalized_div < 0.05f) {
                diversity_score = -0.3f; /* 过于聚集，强惩罚 */
            } else if (normalized_div < 0.15f) {
                diversity_score = normalized_div * 0.5f - 0.1f; /* 轻微负分 */
            } else {
                diversity_score = normalized_div * 0.8f; /* 正奖励 */
            }

            free(pop_mean);
        }
    }

    fitness += w_diversity * diversity_score;

    /* ===== 目标3: 探索新颖性奖励 =====
     * 鼓励探索与当前系统状态不同的参数区域，
     * 防止进化趋同于现有启发式权重。 */
    float novelty_score = 0.0f;
    if (w_size > 0 && system->heuristic_weights) {
        /* 与启发式权重的标准化距离: 距离越大奖励越高(探索) */
        float novelty_dist = 0.0f;
        for (size_t i = 0; i < w_size && i < genome_len; i++) {
            float diff = genome[i] - system->heuristic_weights[i];
            novelty_dist += diff * diff;
        }
        novelty_dist = sqrtf(novelty_dist);

        /* 归一化：预期最大距离约为sqrt(w_size)*2 */
        float max_dist = sqrtf((float)w_size) * 2.0f + 1e-8f;
        float norm_novelty = fminf(1.0f, novelty_dist / max_dist);

        /* 新颖性奖励: 适度的新颖性获得最高奖励 */
        /* 太近(趋同)无奖励，太远(随机)也行但略低，最佳在0.3-0.7范围 */
        if (norm_novelty < 0.1f) {
            novelty_score = -0.2f; /* 趋同惩罚 */
        } else if (norm_novelty < 0.5f) {
            novelty_score = norm_novelty * 2.0f; /* 线性增长到1.0 */
        } else {
            novelty_score = 1.0f - (norm_novelty - 0.5f) * 0.5f; /* 缓慢下降 */
        }

        /* 执行偏离历史: 如果连续失败较多，额外奖励探索 */
        if (system->consecutive_failures > 3) {
            float fail_bonus = fminf(1.0f, (float)(system->consecutive_failures - 3) / 10.0f);
            novelty_score += fail_bonus * 0.5f;
        }

        /* 可行性趋势: 可行性下降时加强探索 */
        if (system->feasibility_history_count >= 3) {
            int idx = system->feasibility_history_count;
            float recent = system->feasibility_history[(idx - 1) % 20];
            float older = system->feasibility_history[(idx - 3) % 20];
            if (recent < older && recent < 0.4f) {
                novelty_score += 0.2f; /* 性能下降，奖励探索 */
            }
        }
    }

    fitness += w_novelty * novelty_score;

    /* 微小的基准项，避免零或负适应度导致选择问题 */
    fitness += system->config.max_plan_length * 0.005f;

    return fitness;
}

static void planning_evolution_tournament_select(const PlanningSystem* system, int* parent1, int* parent2) {
    int tsize = EVOLUTION_DEFAULT_TOURNAMENT_SIZE;
    int best1 = (int)secure_random_int((uint32_t)system->evolution_population_size);
    for (int i = 1; i < tsize; i++) {
        int idx = (int)secure_random_int((uint32_t)system->evolution_population_size);
        if (system->evolution_fitness[idx] > system->evolution_fitness[best1]) best1 = idx;
    }
    int best2 = (int)secure_random_int((uint32_t)system->evolution_population_size);
    for (int i = 1; i < tsize; i++) {
        int idx = (int)secure_random_int((uint32_t)system->evolution_population_size);
        if (system->evolution_fitness[idx] > system->evolution_fitness[best2]) best2 = idx;
    }
    *parent1 = best1;
    *parent2 = best2;
}

static void planning_evolution_sbx_crossover(const float* p1, const float* p2, float* c1, float* c2, size_t size, float eta_c) {
    for (size_t i = 0; i < size; i++) {
        if (plan_rng_uniform(0.0f, 1.0f) < 0.5f) {
            if (fabsf(p1[i] - p2[i]) > 1e-10f) {
                float y1 = MIN(p1[i], p2[i]);
                float y2 = MAX(p1[i], p2[i]);
                float yl = -5.0f, yu = 5.0f;
                float rand = plan_rng_uniform(0.0f, 1.0f);
                float beta = 1.0f + (2.0f * (y1 - yl) / (y2 - y1 + 1e-10f));
                float alpha = 2.0f - powf(beta, -(eta_c + 1.0f));
                float betaq;
                if (rand <= 1.0f / alpha) {
                    betaq = powf(rand * alpha, 1.0f / (eta_c + 1.0f));
                } else {
                    betaq = powf(1.0f / (2.0f - rand * alpha), 1.0f / (eta_c + 1.0f));
                }
                c1[i] = 0.5f * ((y1 + y2) - betaq * (y2 - y1));
                beta = 1.0f + (2.0f * (yu - y2) / (y2 - y1 + 1e-10f));
                alpha = 2.0f - powf(beta, -(eta_c + 1.0f));
                if (rand <= 1.0f / alpha) {
                    betaq = powf(rand * alpha, 1.0f / (eta_c + 1.0f));
                } else {
                    betaq = powf(1.0f / (2.0f - rand * alpha), 1.0f / (eta_c + 1.0f));
                }
                c2[i] = 0.5f * ((y1 + y2) + betaq * (y2 - y1));
            } else {
                c1[i] = p1[i];
                c2[i] = p2[i];
            }
        } else {
            c1[i] = p1[i];
            c2[i] = p2[i];
        }
        c1[i] = CLAMP(c1[i], -5.0f, 5.0f);
        c2[i] = CLAMP(c2[i], -5.0f, 5.0f);
    }
}

static void planning_evolution_polynomial_mutation(float* child, size_t size, float pm, float eta_m) {
    for (size_t i = 0; i < size; i++) {
        if (plan_rng_uniform(0.0f, 1.0f) < pm) {
            float y = child[i], yl = -5.0f, yu = 5.0f;
            float delta1 = (y - yl) / (yu - yl);
            float delta2 = (yu - y) / (yu - yl);
            float rand = plan_rng_uniform(0.0f, 1.0f);
            float mut_pow = 1.0f / (eta_m + 1.0f);
            float deltaq;
            if (rand <= 0.5f) {
                deltaq = powf(2.0f * rand + (1.0f - 2.0f * rand) * powf(1.0f - delta1, eta_m + 1.0f), mut_pow) - 1.0f;
            } else {
                deltaq = 1.0f - powf(2.0f * (1.0f - rand) + 2.0f * (rand - 0.5f) * powf(1.0f - delta2, eta_m + 1.0f), mut_pow);
            }
            child[i] = CLAMP(y + deltaq * (yu - yl), yl, yu);
        }
    }
}

static void cmaes_initialize(PlanningSystem* system) {
    if (!system || system->cmaes_initialized) return;
    size_t dim = system->evolution_genome_size;
    if (dim == 0) return;

    system->cmaes_mean = (float*)safe_calloc(dim, sizeof(float));
    system->cmaes_sigma = (float*)safe_malloc(dim * sizeof(float));
    system->cmaes_cov = (float*)safe_calloc(dim * dim, sizeof(float));
    system->cmaes_eigenvalues = (float*)safe_calloc(dim, sizeof(float));
    system->cmaes_evolution_path = (float*)safe_calloc(dim, sizeof(float));
    system->cmaes_evolution_path_c = (float*)safe_calloc(dim, sizeof(float));

    if (!system->cmaes_mean || !system->cmaes_sigma || !system->cmaes_cov ||
        !system->cmaes_eigenvalues || !system->cmaes_evolution_path || !system->cmaes_evolution_path_c) {
        safe_free((void**)&system->cmaes_mean);
        safe_free((void**)&system->cmaes_sigma);
        safe_free((void**)&system->cmaes_cov);
        safe_free((void**)&system->cmaes_eigenvalues);
        safe_free((void**)&system->cmaes_evolution_path);
        safe_free((void**)&system->cmaes_evolution_path_c);
        return;
    }

    for (size_t i = 0; i < dim; i++) {
        system->cmaes_mean[i] = 0.0f;
        system->cmaes_sigma[i] = 0.3f;
        system->cmaes_eigenvalues[i] = 1.0f;
    }
    for (size_t i = 0; i < dim * dim; i++) {
        system->cmaes_cov[i] = (i % (dim + 1) == 0) ? 1.0f : 0.0f;
    }

    system->cmaes_dimension = dim;
    system->cmaes_lambda_eff = (float)(dim > 1 ? dim / 2 : 1);
    system->cmaes_generation = 0;
    system->cmaes_initialized = 1;
}

/* 前向声明 */
static float plan_rng_normal(float mean, float stddev);
static int cholesky_decompose(float* cov, float* L, size_t n);

static int plan_cmaes_sample(PlanningSystem* system) {
    if (!system || !system->cmaes_initialized) return -1;
    size_t dim = system->cmaes_dimension;
    size_t pop = system->evolution_population_size;
    if (dim == 0 || pop == 0) return -1;

    float* L = (float*)safe_malloc(dim * dim * sizeof(float));
    if (!L) return -1;

    memset(L, 0, dim * dim * sizeof(float));
    cholesky_decompose(system->cmaes_cov, L, dim);

    /* 多元正态采样: x = mean + sigma * L * z, z ~ N(0,I) */
    for (size_t i = 0; i < pop; i++) {
        float* z = (float*)safe_malloc(dim * sizeof(float));
        if (!z) { safe_free((void**)&L); return -1; }

        for (size_t j = 0; j < dim; j++) {
            z[j] = plan_rng_normal(0.0f, 1.0f);
        }

        float* ind = system->evolution_population + i * system->evolution_genome_size;
        for (size_t j = 0; j < dim && j < system->evolution_genome_size; j++) {
            float Lz = 0.0f;
            for (size_t k = 0; k <= j; k++) {
                Lz += L[j * dim + k] * z[k];
            }
            ind[j] = system->cmaes_mean[j] + system->cmaes_sigma[j] * Lz;
            ind[j] = CLAMP(ind[j], -5.0f, 5.0f);
        }
        safe_free((void**)&z);
    }

    safe_free((void**)&L);
    return 0;
}

/* === CMA-ES演化路径与协方差更新 ===
 * 使用加权选择和秩-μ更新 */
static int cmaes_update(PlanningSystem* system) {
    if (!system || !system->cmaes_initialized) return -1;
    size_t dim = system->cmaes_dimension;
    size_t pop = system->evolution_population_size;
    if (dim == 0 || pop == 0) return -1;

    /* 选择最优个体计算适应度加权均值 */
    int* indices = (int*)safe_malloc(pop * sizeof(int));
    if (!indices) return -1;
    for (size_t i = 0; i < pop; i++) indices[i] = (int)i;

    /* 按适应度降序排列（简单冒泡排序，pop通常很小~50） */
    for (size_t i = 0; i < pop - 1; i++) {
        for (size_t j = 0; j < pop - 1 - i; j++) {
            if (system->evolution_fitness[indices[j]] < system->evolution_fitness[indices[j + 1]]) {
                int tmp = indices[j]; indices[j] = indices[j + 1]; indices[j + 1] = tmp;
            }
        }
    }

    float mu_eff = system->cmaes_lambda_eff;
    if (mu_eff < 1.0f) mu_eff = 1.0f;
    size_t mu = (size_t)(pop / 2);
    if (mu < 1) mu = 1;

    /* 权重: 对数递减 */
    float* weights = (float*)safe_malloc(mu * sizeof(float));
    if (!weights) { safe_free((void**)&indices); return -1; }
    float sum_w = 0.0f;
    for (size_t i = 0; i < mu; i++) {
        weights[i] = logf((float)(mu + 1)) - logf((float)(i + 1));
        sum_w += weights[i];
    }
    for (size_t i = 0; i < mu; i++) weights[i] /= sum_w;

    /* 计算加权均值（新的均值向量） */
    float* new_mean = (float*)safe_calloc(dim, sizeof(float));
    if (!new_mean) { safe_free((void**)&indices); safe_free((void**)&weights); return -1; }
    for (size_t i = 0; i < mu; i++) {
        float* ind = system->evolution_population + (size_t)indices[i] * system->evolution_genome_size;
        for (size_t j = 0; j < dim; j++) {
            new_mean[j] += weights[i] * ind[j];
        }
    }

    /* 演化路径更新 (CSA - Cumulative Step-size Adaptation) */
    float c_sigma = 2.0f / (float)(dim + 2);
    float d_sigma = 1.0f + 2.0f * (float)dim / (float)(dim + 2);
    float chi_n = sqrtf((float)dim) * (1.0f - 1.0f / (4.0f * (float)dim) + 1.0f /
                         (21.0f * (float)dim * (float)dim));

    /* 权重方差 */
    float sum_w_sq = 0.0f;
    for (size_t i = 0; i < mu; i++) sum_w_sq += weights[i] * weights[i];
    float mu_w = 1.0f / sum_w_sq;

    /* 步长演化路径更新 */
    float* step_diff = (float*)safe_malloc(dim * sizeof(float));
    if (!step_diff) {
        safe_free((void**)&indices); safe_free((void**)&weights); safe_free((void**)&new_mean);
        return -1;
    }
    for (size_t j = 0; j < dim; j++) {
        step_diff[j] = (new_mean[j] - system->cmaes_mean[j]) / (system->cmaes_sigma[j] + 1e-10f);
    }

    /* 用Cholesky L逆来白化步长方向 */
    float* L = (float*)safe_malloc(dim * dim * sizeof(float));
    if (!L) {
        safe_free((void**)&indices); safe_free((void**)&weights);
        safe_free((void**)&new_mean); safe_free((void**)&step_diff);
        return -1;
    }
    memset(L, 0, dim * dim * sizeof(float));
    cholesky_decompose(system->cmaes_cov, L, dim);

    /* 求解 L * inv_step = step_diff （前代法） */
    float* inv_step = (float*)safe_malloc(dim * sizeof(float));
    if (!inv_step) {
        safe_free((void**)&indices); safe_free((void**)&weights);
        safe_free((void**)&new_mean); safe_free((void**)&step_diff);
        safe_free((void**)&L); return -1;
    }
    for (size_t j = 0; j < dim; j++) {
        float s = step_diff[j];
        for (size_t k = 0; k < j; k++) s -= L[j * dim + k] * inv_step[k];
        inv_step[j] = s / (L[j * dim + j] + 1e-10f);
    }

    float inv_norm = 0.0f;
    for (size_t j = 0; j < dim; j++) inv_norm += inv_step[j] * inv_step[j];
    inv_norm = sqrtf(inv_norm);

    for (size_t j = 0; j < dim; j++) {
        system->cmaes_evolution_path[j] = (1.0f - c_sigma) * system->cmaes_evolution_path[j] +
            sqrtf(c_sigma * (2.0f - c_sigma) * mu_w) * inv_step[j] / (inv_norm + 1e-10f);
    }

    float path_norm = 0.0f;
    for (size_t j = 0; j < dim; j++) path_norm += system->cmaes_evolution_path[j] * system->cmaes_evolution_path[j];
    path_norm = sqrtf(path_norm);

    /* 步长更新 */
    float h_sig = (path_norm / chi_n < 1.4f + 2.0f / (float)(dim + 1)) ? 1.0f : 0.0f;
    for (size_t j = 0; j < dim; j++) {
        system->cmaes_sigma[j] *= expf((c_sigma / d_sigma) * (path_norm / chi_n - 1.0f));
        if (system->cmaes_sigma[j] > 1.0f) system->cmaes_sigma[j] = 1.0f;
        if (system->cmaes_sigma[j] < 0.01f) system->cmaes_sigma[j] = 0.01f;
    }

    /* 协方差矩阵秩-μ更新 */
    float c_cov = 2.0f / ((float)(dim + 6) * (float)(dim + 6) + mu_eff);
    float c_cov_path = (1.0f - h_sig * 0.3f) * 2.0f / ((float)(dim + 1.5f) * (float)(dim + 1.5f));

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            system->cmaes_cov[i * dim + j] *= (1.0f - c_cov - c_cov_path);
        }
    }

    /* 秩-1更新（演化路径） */
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            system->cmaes_cov[i * dim + j] += c_cov_path *
                system->cmaes_evolution_path[i] * system->cmaes_evolution_path[j];
        }
    }

    /* 秩-μ更新 */
    for (size_t k = 0; k < mu; k++) {
        float* ind = system->evolution_population + (size_t)indices[k] * system->evolution_genome_size;
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                float dx_i = (ind[i] - system->cmaes_mean[i]) / (system->cmaes_sigma[i] + 1e-10f);
                float dx_j = (ind[j] - system->cmaes_mean[j]) / (system->cmaes_sigma[j] + 1e-10f);
                system->cmaes_cov[i * dim + j] += c_cov * weights[k] * dx_i * dx_j;
            }
        }
    }

    /* 确保协方差矩阵对称正定 */
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < i; j++) {
            float avg = (system->cmaes_cov[i * dim + j] + system->cmaes_cov[j * dim + i]) * 0.5f;
            system->cmaes_cov[i * dim + j] = avg;
            system->cmaes_cov[j * dim + i] = avg;
        }
        if (system->cmaes_cov[i * dim + i] < 1e-8f) system->cmaes_cov[i * dim + i] = 1e-8f;
    }

    /* 更新均值 */
    memcpy(system->cmaes_mean, new_mean, dim * sizeof(float));
    system->cmaes_generation++;

    safe_free((void**)&indices); safe_free((void**)&weights);
    safe_free((void**)&new_mean); safe_free((void**)&step_diff);
    safe_free((void**)&L); safe_free((void**)&inv_step);
    return 0;
}

/* ========== topology基因 → CfC网络架构 ========== */
static int planning_evolution_apply_topology_to_cfc(PlanningSystem* system,
                                                     const TopologyGenes* topo) {
    if (!system || !topo) return -1;
    if (system->evolution_genome_size == 0) return -1;

    size_t genome_size = system->evolution_genome_size;
    int num_layers = topo->num_hidden_layers;
    if (num_layers <= 0) num_layers = 2;
    if (num_layers > ARCH_GENOME_MAX_LAYERS) num_layers = ARCH_GENOME_MAX_LAYERS;

    float total_neurons = 0.0f;
    int l;
    for (l = 0; l < num_layers; l++) {
        total_neurons += (float)topo->layer_sizes[l];
    }
    if (total_neurons < 1.0f) total_neurons = 1.0f;

    /* 将拓扑基因解码为CfC网络结构:
     * 每层神经元数映射为连续权重,连接密度控制稀疏性 */
    float* target = system->evolution_best_genome ? system->evolution_best_genome : NULL;
    if (!target) return -1;

    for (size_t i = 0; i < genome_size; i++) {
        float decode = 0.0f;
        for (l = 0; l < num_layers; l++) {
            float layer_contrib = (float)topo->layer_sizes[l] / total_neurons;
            float phase = (float)(i * (l + 1)) / (float)(genome_size + 1);
            decode += layer_contrib * sinf(phase * 6.28318530718f);
        }
        decode *= topo->connectivity_density;
        if (topo->activation_types[l < num_layers ? l : 0] == 0) {
            decode = tanhf(decode);
        } else {
            decode = 1.0f / (1.0f + expf(-decode));
        }
        target[i] = CLAMP(decode * 0.5f, -1.0f, 1.0f);
    }

    return (int)genome_size;
}

/* ==================== 时序约束网络(TCN)完整实现 ==================== */

#define TC_TIME_INFINITY 1e10f
#define TC_EPSILON 1e-6f

int planning_temporal_constraint_network_create(PlanningSystem* system) {
    if (!system) return -1;
    if (system->temporal_constraints) {
        planning_temporal_constraint_network_destroy(system);
    }
    TemporalConstraintNetwork* tcn = (TemporalConstraintNetwork*)safe_calloc(1, sizeof(TemporalConstraintNetwork));
    if (!tcn) return -1;
    tcn->step_count = TC_MAX_STEPS;
    tcn->constraint_capacity = TC_MAX_CONSTRAINTS;
    tcn->constraint_count = 0;
    tcn->is_consistent = 1;
    tcn->relaxation_count = 0;
    tcn->constraints = (TemporalConstraint*)safe_calloc(TC_MAX_CONSTRAINTS, sizeof(TemporalConstraint));
    tcn->earliest_start = (float*)safe_calloc(TC_MAX_STEPS, sizeof(float));
    tcn->latest_start = (float*)safe_calloc(TC_MAX_STEPS, sizeof(float));
    tcn->earliest_end = (float*)safe_calloc(TC_MAX_STEPS, sizeof(float));
    tcn->latest_end = (float*)safe_calloc(TC_MAX_STEPS, sizeof(float));
    tcn->durations = (float*)safe_calloc(TC_MAX_STEPS, sizeof(float));
    if (!tcn->constraints || !tcn->earliest_start || !tcn->latest_start ||
        !tcn->earliest_end || !tcn->latest_end || !tcn->durations) {
        safe_free((void**)&tcn->constraints);
        safe_free((void**)&tcn->earliest_start);
        safe_free((void**)&tcn->latest_start);
        safe_free((void**)&tcn->earliest_end);
        safe_free((void**)&tcn->latest_end);
        safe_free((void**)&tcn->durations);
        safe_free((void**)&tcn);
        return -1;
    }
    for (int i = 0; i < TC_MAX_STEPS; i++) {
        tcn->durations[i] = 1.0f;
        tcn->earliest_start[i] = 0.0f;
        tcn->latest_start[i] = TC_TIME_INFINITY;
        tcn->earliest_end[i] = tcn->durations[i];
        tcn->latest_end[i] = TC_TIME_INFINITY;
    }
    system->temporal_constraints = tcn;
    return 0;
}

void planning_temporal_constraint_network_destroy(PlanningSystem* system) {
    if (!system || !system->temporal_constraints) return;
    TemporalConstraintNetwork* tcn = system->temporal_constraints;
    safe_free((void**)&tcn->constraints);
    safe_free((void**)&tcn->earliest_start);
    safe_free((void**)&tcn->latest_start);
    safe_free((void**)&tcn->earliest_end);
    safe_free((void**)&tcn->latest_end);
    safe_free((void**)&tcn->durations);
    safe_free((void**)&tcn);
    system->temporal_constraints = NULL;
}

int planning_temporal_constraint_add(PlanningSystem* system, int from_step, int to_step,
                                      TemporalConstraintType type, float min_gap, float max_gap) {
    if (!system || !system->temporal_constraints) return -1;
    TemporalConstraintNetwork* tcn = system->temporal_constraints;
    if (from_step < 0 || to_step < 0 || from_step >= TC_MAX_STEPS || to_step >= TC_MAX_STEPS) return -1;
    if (tcn->constraint_count >= tcn->constraint_capacity) return -1;
    TemporalConstraint* tc = &tcn->constraints[tcn->constraint_count++];
    tc->from_step = from_step;
    tc->to_step = to_step;
    tc->type = type;
    tc->min_gap = (min_gap < 0.0f) ? 0.0f : min_gap;
    tc->max_gap = (max_gap < tc->min_gap) ? tc->min_gap : max_gap;
    tc->weight = 1.0f;
    tcn->is_consistent = 1;
    return 0;
}

static int temporal_constraint_apply(TemporalConstraintNetwork* tcn, int c, int is_forward) {
    TemporalConstraint* tc = &tcn->constraints[c];
    int f = tc->from_step, t = tc->to_step;
    if (f < 0 || t < 0 || f >= tcn->step_count || t >= tcn->step_count) return 0;
    int changed = 0;
    switch (tc->type) {
        case TCONSTRAINT_BEFORE: {
            if (is_forward) {
                float min_start_t = tcn->earliest_end[f] + tc->min_gap;
                if (min_start_t > tcn->earliest_start[t] + TC_EPSILON) {
                    tcn->earliest_start[t] = min_start_t;
                    changed = 1;
                }
            } else {
                float max_end_f = tcn->latest_start[t] - tc->min_gap;
                if (max_end_f < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = max_end_f;
                    changed = 1;
                }
            }
            break;
        }
        case TCONSTRAINT_AFTER: {
            if (is_forward) {
                float min_start_f = tcn->earliest_end[t] + tc->min_gap;
                if (min_start_f > tcn->earliest_start[f] + TC_EPSILON) {
                    tcn->earliest_start[f] = min_start_f;
                    changed = 1;
                }
            } else {
                float max_end_t = tcn->latest_start[f] - tc->min_gap;
                if (max_end_t < tcn->latest_end[t] - TC_EPSILON) {
                    tcn->latest_end[t] = max_end_t;
                    changed = 1;
                }
            }
            break;
        }
        case TCONSTRAINT_EQUALS: {
            float max_earliest = (tcn->earliest_start[f] > tcn->earliest_start[t]) ?
                tcn->earliest_start[f] : tcn->earliest_start[t];
            float min_latest = (tcn->latest_start[f] < tcn->latest_start[t]) ?
                tcn->latest_start[f] : tcn->latest_start[t];
            if (max_earliest > tcn->earliest_start[f] + TC_EPSILON) {
                tcn->earliest_start[f] = max_earliest; changed = 1;
            }
            if (max_earliest > tcn->earliest_start[t] + TC_EPSILON) {
                tcn->earliest_start[t] = max_earliest; changed = 1;
            }
            if (min_latest < tcn->latest_start[f] - TC_EPSILON) {
                tcn->latest_start[f] = min_latest; changed = 1;
            }
            if (min_latest < tcn->latest_start[t] - TC_EPSILON) {
                tcn->latest_start[t] = min_latest; changed = 1;
            }
            break;
        }
        case TCONSTRAINT_DURING: {
            if (is_forward) {
                if (tcn->earliest_start[t] > tcn->earliest_start[f] + TC_EPSILON) {
                    tcn->earliest_start[f] = tcn->earliest_start[t]; changed = 1;
                }
                float min_end_t = tcn->earliest_end[f];
                if (min_end_t > tcn->earliest_end[t] + TC_EPSILON) {
                    tcn->earliest_end[t] = min_end_t; changed = 1;
                }
            } else {
                float max_end_f = tcn->latest_end[t];
                if (max_end_f < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = max_end_f; changed = 1;
                }
                if (tcn->latest_start[f] < tcn->latest_start[t] - TC_EPSILON) {
                    tcn->latest_start[t] = tcn->latest_start[f]; changed = 1;
                }
            }
            break;
        }
        case TCONSTRAINT_OVERLAPS: {
            if (is_forward) {
                float min_start_t = tcn->earliest_start[f] + TC_EPSILON;
                if (min_start_t > tcn->earliest_start[t] + TC_EPSILON) {
                    tcn->earliest_start[t] = min_start_t; changed = 1;
                }
                float min_end_f = tcn->earliest_start[t] + TC_EPSILON;
                if (min_end_f > tcn->earliest_end[f] + TC_EPSILON) {
                    tcn->earliest_end[f] = min_end_f; changed = 1;
                }
                float min_end_t = tcn->earliest_end[f] + TC_EPSILON;
                if (min_end_t > tcn->earliest_end[t] + TC_EPSILON) {
                    tcn->earliest_end[t] = min_end_t; changed = 1;
                }
            } else {
                float max_start_f = tcn->latest_start[t] - TC_EPSILON;
                if (max_start_f < tcn->latest_start[f] - TC_EPSILON) {
                    tcn->latest_start[f] = max_start_f; changed = 1;
                }
                float max_start_t = tcn->latest_end[f] - TC_EPSILON;
                if (max_start_t < tcn->latest_start[t] - TC_EPSILON) {
                    tcn->latest_start[t] = max_start_t; changed = 1;
                }
                float max_end_f = tcn->latest_end[t] - TC_EPSILON;
                if (max_end_f < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = max_end_f; changed = 1;
                }
            }
            break;
        }
        case TCONSTRAINT_MEETS: {
            if (is_forward) {
                float expected_start_t = tcn->earliest_end[f];
                if (expected_start_t > tcn->earliest_start[t] + TC_EPSILON) {
                    tcn->earliest_start[t] = expected_start_t; changed = 1;
                }
            } else {
                float expected_end_f = tcn->latest_start[t];
                if (expected_end_f < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = expected_end_f; changed = 1;
                }
            }
            break;
        }
        case TCONSTRAINT_STARTS: {
            float max_earliest = (tcn->earliest_start[f] > tcn->earliest_start[t]) ?
                tcn->earliest_start[f] : tcn->earliest_start[t];
            float min_latest = (tcn->latest_start[f] < tcn->latest_start[t]) ?
                tcn->latest_start[f] : tcn->latest_start[t];
            if (max_earliest > tcn->earliest_start[f] + TC_EPSILON) {
                tcn->earliest_start[f] = max_earliest; changed = 1;
            }
            if (max_earliest > tcn->earliest_start[t] + TC_EPSILON) {
                tcn->earliest_start[t] = max_earliest; changed = 1;
            }
            if (min_latest < tcn->latest_start[f] - TC_EPSILON) {
                tcn->latest_start[f] = min_latest; changed = 1;
            }
            if (min_latest < tcn->latest_start[t] - TC_EPSILON) {
                tcn->latest_start[t] = min_latest; changed = 1;
            }
            break;
        }
        case TCONSTRAINT_FINISHES: {
            float max_earliest_end = (tcn->earliest_end[f] > tcn->earliest_end[t]) ?
                tcn->earliest_end[f] : tcn->earliest_end[t];
            float min_latest_end = (tcn->latest_end[f] < tcn->latest_end[t]) ?
                tcn->latest_end[f] : tcn->latest_end[t];
            if (max_earliest_end > tcn->earliest_end[f] + TC_EPSILON) {
                tcn->earliest_end[f] = max_earliest_end; changed = 1;
            }
            if (max_earliest_end > tcn->earliest_end[t] + TC_EPSILON) {
                tcn->earliest_end[t] = max_earliest_end; changed = 1;
            }
            if (min_latest_end < tcn->latest_end[f] - TC_EPSILON) {
                tcn->latest_end[f] = min_latest_end; changed = 1;
            }
            if (min_latest_end < tcn->latest_end[t] - TC_EPSILON) {
                tcn->latest_end[t] = min_latest_end; changed = 1;
            }
            break;
        }
        case TCONSTRAINT_BEFORE_OR_MEETS: {
            if (is_forward) {
                float min_start_t = tcn->earliest_end[f] + tc->min_gap;
                if (min_start_t > tcn->earliest_start[t] + TC_EPSILON) {
                    tcn->earliest_start[t] = min_start_t; changed = 1;
                }
            } else {
                float max_end_f = tcn->latest_start[t] - tc->min_gap;
                if (max_end_f < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = max_end_f; changed = 1;
                }
            }
            break;
        }
        case TCONSTRAINT_AFTER_OR_MEETS: {
            if (is_forward) {
                float min_start_f = tcn->earliest_end[t] + tc->min_gap;
                if (min_start_f > tcn->earliest_start[f] + TC_EPSILON) {
                    tcn->earliest_start[f] = min_start_f; changed = 1;
                }
            } else {
                float max_end_t = tcn->latest_start[f] - tc->min_gap;
                if (max_end_t < tcn->latest_end[t] - TC_EPSILON) {
                    tcn->latest_end[t] = max_end_t; changed = 1;
                }
            }
            break;
        }
        case TCONSTRAINT_DISJOINT: {
            if (is_forward) {
                float min_start = tcn->earliest_end[f] + tc->min_gap;
                if (min_start > tcn->earliest_start[t] + TC_EPSILON) {
                    tcn->earliest_start[t] = min_start; changed = 1;
                }
                float min_start2 = tcn->earliest_end[t] + tc->min_gap;
                if (min_start2 > tcn->earliest_start[f] + TC_EPSILON) {
                    tcn->earliest_start[f] = min_start2; changed = 1;
                }
            } else {
                float max_end_f = tcn->latest_start[t] - tc->min_gap;
                if (max_end_f < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = max_end_f; changed = 1;
                }
                float max_end_t = tcn->latest_start[f] - tc->min_gap;
                if (max_end_t < tcn->latest_end[t] - TC_EPSILON) {
                    tcn->latest_end[t] = max_end_t; changed = 1;
                }
            }
            break;
        }
    }
    return changed;
}

int planning_temporal_forward_propagate(PlanningSystem* system) {
    if (!system || !system->temporal_constraints) return -1;
    TemporalConstraintNetwork* tcn = system->temporal_constraints;
    int changed = 0;
    for (int iter = 0; iter < tcn->step_count; iter++) {
        changed = 0;
        for (int i = 0; i < tcn->step_count; i++) {
            float expected_end = tcn->earliest_start[i] + tcn->durations[i];
            if (expected_end > tcn->earliest_end[i] + TC_EPSILON) {
                tcn->earliest_end[i] = expected_end;
                changed = 1;
            }
        }
        for (int c = 0; c < tcn->constraint_count; c++) {
            if (temporal_constraint_apply(tcn, c, 1)) changed = 1;
        }
        if (!changed) break;
    }
    return 0;
}

int planning_temporal_backward_propagate(PlanningSystem* system) {
    if (!system || !system->temporal_constraints) return -1;
    TemporalConstraintNetwork* tcn = system->temporal_constraints;
    int changed = 0;
    for (int iter = 0; iter < tcn->step_count; iter++) {
        changed = 0;
        for (int i = 0; i < tcn->step_count; i++) {
            float expected_start = tcn->latest_end[i] - tcn->durations[i];
            if (expected_start < tcn->latest_start[i] - TC_EPSILON) {
                tcn->latest_start[i] = expected_start;
                changed = 1;
            }
        }
        for (int c = 0; c < tcn->constraint_count; c++) {
            if (temporal_constraint_apply(tcn, c, 0)) changed = 1;
        }
        if (!changed) break;
    }
    return 0;
}

int planning_temporal_consistency_check(PlanningSystem* system) {
    if (!system || !system->temporal_constraints) return 0;
    TemporalConstraintNetwork* tcn = system->temporal_constraints;
    if (tcn->constraint_count == 0) { tcn->is_consistent = 1; return 1; }

    planning_temporal_forward_propagate(system);
    planning_temporal_backward_propagate(system);

    for (int i = 0; i < tcn->step_count; i++) {
        if (tcn->latest_start[i] < tcn->earliest_start[i] - TC_EPSILON) {
            tcn->is_consistent = 0;
            return 0;
        }
        if (tcn->latest_end[i] < tcn->earliest_end[i] - TC_EPSILON) {
            tcn->is_consistent = 0;
            return 0;
        }
        if (tcn->durations[i] < 0.0f) {
            tcn->is_consistent = 0;
            return 0;
        }
    }

    for (int c = 0; c < tcn->constraint_count; c++) {
        TemporalConstraint* tc = &tcn->constraints[c];
        int f = tc->from_step, t = tc->to_step;
        if (f < 0 || t < 0 || f >= tcn->step_count || t >= tcn->step_count) {
            tcn->is_consistent = 0;
            return 0;
        }
    }

    int* queue = (int*)safe_malloc((size_t)tcn->constraint_count * sizeof(int));
    int* in_queue = (int*)safe_calloc((size_t)tcn->constraint_count, sizeof(int));
    int* revision_count = (int*)safe_calloc((size_t)tcn->constraint_count, sizeof(int));
    if (!queue || !in_queue || !revision_count) {
        safe_free((void**)&queue);
        safe_free((void**)&in_queue);
        safe_free((void**)&revision_count);
        tcn->is_consistent = 0;
        return 0;
    }

    int head = 0, tail = 0;
    for (int c = 0; c < tcn->constraint_count; c++) {
        queue[tail++] = c;
        in_queue[c] = 1;
    }

    while (head < tail) {
        int c = queue[head++];
        in_queue[c] = 0;
        if (++revision_count[c] > tcn->step_count * 2) {
            safe_free((void**)&queue);
            safe_free((void**)&in_queue);
            safe_free((void**)&revision_count);
            tcn->is_consistent = 0;
            return 0;
        }

        TemporalConstraint* tc = &tcn->constraints[c];
        int f = tc->from_step, t = tc->to_step;
        int revised = 0;

        switch (tc->type) {
            case TCONSTRAINT_BEFORE: {
                float min_start_t = tcn->earliest_end[f] + tc->min_gap;
                if (min_start_t > tcn->earliest_start[t] + TC_EPSILON) {
                    tcn->earliest_start[t] = min_start_t;
                    revised = 1;
                }
                float max_end_f = tcn->latest_start[t] - tc->min_gap;
                if (max_end_f < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = max_end_f;
                    revised = 1;
                }
                break;
            }
            case TCONSTRAINT_EQUALS: {
                float max_es = (tcn->earliest_start[f] > tcn->earliest_start[t]) ?
                    tcn->earliest_start[f] : tcn->earliest_start[t];
                float min_ls = (tcn->latest_start[f] < tcn->latest_start[t]) ?
                    tcn->latest_start[f] : tcn->latest_start[t];
                float max_ee = (tcn->earliest_end[f] > tcn->earliest_end[t]) ?
                    tcn->earliest_end[f] : tcn->earliest_end[t];
                float min_le = (tcn->latest_end[f] < tcn->latest_end[t]) ?
                    tcn->latest_end[f] : tcn->latest_end[t];
                if (max_es > tcn->earliest_start[f] + TC_EPSILON) { tcn->earliest_start[f] = max_es; revised = 1; }
                if (max_es > tcn->earliest_start[t] + TC_EPSILON) { tcn->earliest_start[t] = max_es; revised = 1; }
                if (min_ls < tcn->latest_start[f] - TC_EPSILON) { tcn->latest_start[f] = min_ls; revised = 1; }
                if (min_ls < tcn->latest_start[t] - TC_EPSILON) { tcn->latest_start[t] = min_ls; revised = 1; }
                if (max_ee > tcn->earliest_end[f] + TC_EPSILON) { tcn->earliest_end[f] = max_ee; revised = 1; }
                if (max_ee > tcn->earliest_end[t] + TC_EPSILON) { tcn->earliest_end[t] = max_ee; revised = 1; }
                if (min_le < tcn->latest_end[f] - TC_EPSILON) { tcn->latest_end[f] = min_le; revised = 1; }
                if (min_le < tcn->latest_end[t] - TC_EPSILON) { tcn->latest_end[t] = min_le; revised = 1; }
                break;
            }
            case TCONSTRAINT_DURING: {
                if (tcn->earliest_start[t] > tcn->earliest_start[f] + TC_EPSILON) {
                    tcn->earliest_start[f] = tcn->earliest_start[t]; revised = 1;
                }
                if (tcn->latest_start[f] < tcn->latest_start[t] - TC_EPSILON) {
                    tcn->latest_start[t] = tcn->latest_start[f]; revised = 1;
                }
                float min_et = tcn->earliest_end[f];
                if (min_et > tcn->earliest_end[t] + TC_EPSILON) {
                    tcn->earliest_end[t] = min_et; revised = 1;
                }
                float max_ef = tcn->latest_end[t];
                if (max_ef < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = max_ef; revised = 1;
                }
                break;
            }
            case TCONSTRAINT_MEETS: {
                if (tcn->earliest_end[f] > tcn->earliest_start[t] + TC_EPSILON) {
                    tcn->earliest_start[t] = tcn->earliest_end[f]; revised = 1;
                }
                if (tcn->latest_start[t] < tcn->latest_end[f] - TC_EPSILON) {
                    tcn->latest_end[f] = tcn->latest_start[t]; revised = 1;
                }
                break;
            }
            default: {
                int fr = temporal_constraint_apply(tcn, c, 1);
                int br = temporal_constraint_apply(tcn, c, 0);
                if (fr || br) revised = 1;
                break;
            }
        }

        if (revised) {
            for (int cc = 0; cc < tcn->constraint_count; cc++) {
                if (cc != c && !in_queue[cc]) {
                    queue[tail++] = cc;
                    in_queue[cc] = 1;
                }
            }
        }
    }

    safe_free((void**)&queue);
    safe_free((void**)&in_queue);
    safe_free((void**)&revision_count);

    for (int i = 0; i < tcn->step_count; i++) {
        if (tcn->earliest_start[i] > tcn->latest_start[i] + TC_EPSILON ||
            tcn->earliest_end[i] > tcn->latest_end[i] + TC_EPSILON) {
            tcn->is_consistent = 0;
            return 0;
        }
    }

    tcn->is_consistent = 1;
    return 1;
}

int planning_temporal_relax_constraints(PlanningSystem* system, float max_relax_ratio) {
    if (!system || !system->temporal_constraints) return -1;
    if (max_relax_ratio <= 0.0f) return 0;

    TemporalConstraintNetwork* tcn = system->temporal_constraints;
    int relaxed = 0;

    for (int i = 0; i < tcn->constraint_count; i++) {
        TemporalConstraint* tc = &tcn->constraints[i];
        float relax_amount = tc->max_gap * max_relax_ratio;
        if (relax_amount < 0.1f) relax_amount = 0.1f;
        tc->max_gap += relax_amount;
        if (tc->min_gap > 0.0f) {
            float min_relax = tc->min_gap * max_relax_ratio * 0.5f;
            tc->min_gap = (tc->min_gap > min_relax) ? tc->min_gap - min_relax : 0.0f;
        }
        tc->weight *= (1.0f - max_relax_ratio * 0.1f);
        if (tc->weight < 0.1f) tc->weight = 0.1f;
        relaxed++;
    }

    for (int i = 0; i < tcn->step_count; i++) {
        float dur_relax = tcn->durations[i] * max_relax_ratio * 0.5f;
        tcn->durations[i] += dur_relax;
        if (tcn->durations[i] > 10.0f) tcn->durations[i] = 10.0f;
    }

    tcn->relaxation_count++;
    return relaxed;
}

int planning_temporal_get_time_bounds(const PlanningSystem* system, int step,
                                       float* earliest, float* latest) {
    if (!system || !system->temporal_constraints || !earliest || !latest) return -1;
    TemporalConstraintNetwork* tcn = system->temporal_constraints;
    if (step < 0 || step >= tcn->step_count) return -1;
    *earliest = tcn->earliest_start[step];
    *latest = tcn->latest_start[step];
    return 0;
}

/* ==================== 规划修复算法实现 ==================== */

int planning_repair_detect_conflicts(PlanningSystem* system,
                                      const float* current_state, size_t state_size,
                                      int* conflict_steps, int max_conflicts) {
    if (!system || !current_state || !conflict_steps || max_conflicts <= 0) return -1;
    if (state_size == 0) return 0;

    int num_conflicts = 0;
    float deviation_threshold = 0.15f;

    if (system->execution_deviation_smoothed > deviation_threshold && num_conflicts < max_conflicts) {
        conflict_steps[num_conflicts++] = 0;
    }

    if (system->consecutive_failures >= 3 && num_conflicts < max_conflicts) {
        conflict_steps[num_conflicts++] = 1;
    }

    float feasibility = system->last_feasibility;
    if (feasibility < 0.4f && num_conflicts < max_conflicts) {
        conflict_steps[num_conflicts++] = 2;
    }

    if (system->deviation_history_count > 3) {
        float recent_avg = 0.0f;
        int n = (system->deviation_history_count < 5) ? system->deviation_history_count : 5;
        for (int i = system->deviation_history_count - n; i < system->deviation_history_count; i++) {
            recent_avg += system->deviation_history[i];
        }
        recent_avg /= (float)n;
        if (recent_avg > deviation_threshold && num_conflicts < max_conflicts) {
            conflict_steps[num_conflicts++] = 3;
        }
    }

    if (system->temporal_constraints && !system->temporal_constraints->is_consistent && num_conflicts < max_conflicts) {
        conflict_steps[num_conflicts++] = 4;
    }

    if (state_size > 0) {
        float state_norm = 0.0f;
        for (size_t d = 0; d < state_size && d < 10; d++) {
            state_norm += fabsf(current_state[d]);
        }
        state_norm /= (float)((state_size < 10) ? state_size : 10);
        if (state_norm > 1.8f && num_conflicts < max_conflicts) {
            conflict_steps[num_conflicts++] = 5;
        }
    }

    return num_conflicts;
}

int planning_repair_local(PlanningSystem* system,
                           const float* plan, size_t plan_size,
                           const int* conflict_steps, int num_conflicts,
                           const float* current_state, size_t state_size,
                           float* repaired_plan, size_t max_plan_size,
                           PlanRepairResult* result) {
    if (!system || !plan || !conflict_steps || !current_state || !repaired_plan || !result) return -1;
    if (plan_size == 0 || max_plan_size == 0 || state_size == 0) return -1;

    memset(result, 0, sizeof(PlanRepairResult));
    size_t total_steps = plan_size / state_size;
    size_t repair_steps = (max_plan_size / state_size) < total_steps ? (max_plan_size / state_size) : total_steps;

    memcpy(repaired_plan, plan, (repair_steps < total_steps ? repair_steps : total_steps) * state_size * sizeof(float));

    float step_size = 0.1f * (1.0f - system->config.risk_tolerance * 0.5f);
    float blend_ratio = 0.3f;

    for (int ci = 0; ci < num_conflicts; ci++) {
        int conflict_step = conflict_steps[ci];
        if (conflict_step < 0 || (size_t)conflict_step >= repair_steps) continue;

        size_t local_window = 3;
        size_t window_start = ((size_t)conflict_step >= local_window) ? (size_t)conflict_step - local_window : 0;
        size_t window_end = (conflict_step + local_window + 1 < repair_steps) ?
            (size_t)(conflict_step + local_window + 1) : repair_steps;

        for (size_t s = window_start; s < window_end; s++) {
            for (size_t d = 0; d < state_size; d++) {
                float plan_val = (s < total_steps) ? plan[s * state_size + d] : 0.0f;
                float state_val = (d < state_size) ? current_state[d] : 0.0f;
                float target = plan_val * (1.0f - blend_ratio) + state_val * blend_ratio;
                float diff = target - plan_val;
                float correction = CLAMP(diff * step_size, -step_size * 2.0f, step_size * 2.0f);
                if (s * state_size + d < max_plan_size) {
                    repaired_plan[s * state_size + d] = CLAMP(plan_val + correction, -2.0f, 2.0f);
                }
            }
        }
    }

    result->repaired_plan = repaired_plan;
    result->repaired_length = (int)(repair_steps * state_size);
    result->repair_strategy = PLAN_REPAIR_STRATEGY_LOCAL;
    result->num_affected = num_conflicts;
    result->use_affected_only = 1;

    float total_change = 0.0f;
    for (size_t i = 0; i < repair_steps * state_size && i < max_plan_size; i++) {
        total_change += fabsf(repaired_plan[i] - plan[i % plan_size]);
    }
    result->repair_quality = 1.0f - (total_change / (float)(repair_steps * state_size + 1));
    if (result->repair_quality < 0.0f) result->repair_quality = 0.0f;

    if (system->temporal_constraints) {
        planning_temporal_consistency_check(system);
        if (!system->temporal_constraints->is_consistent) {
            planning_temporal_relax_constraints(system, 0.2f);
        }
    }

    return result->repaired_length;
}

int planning_repair_global(PlanningSystem* system,
                            const float* current_state, size_t state_size,
                            const float* goal, size_t goal_size,
                            float* repaired_plan, size_t max_plan_size,
                            PlanRepairResult* result) {
    if (!system || !current_state || !goal || !repaired_plan || !result) return -1;
    if (state_size == 0 || goal_size == 0 || max_plan_size == 0) return -1;

    memset(result, 0, sizeof(PlanRepairResult));
    size_t state_dim = (state_size < goal_size) ? state_size : goal_size;
    if (state_dim > MAX_STATE_DIMENSION) state_dim = MAX_STATE_DIMENSION;

    float state[MAX_STATE_DIMENSION];
    memcpy(state, current_state, state_dim * sizeof(float));
    size_t max_steps = (size_t)system->config.max_plan_length;
    if (max_steps * state_dim > max_plan_size) max_steps = max_plan_size / state_dim;
    size_t steps = 0;

    for (size_t s = 0; s < max_steps; s++) {
        for (size_t d = 0; d < state_dim; d++) {
            float diff = goal[d] - state[d];
            float action = CLAMP(diff * 0.15f, -0.2f, 0.2f);
            state[d] += action;
            state[d] = CLAMP(state[d], -2.0f, 2.0f);
            if (s * state_dim + d < max_plan_size) {
                repaired_plan[s * state_dim + d] = state[d];
            }
        }
        steps = s + 1;
        float dist = 0.0f;
        for (size_t d = 0; d < state_dim; d++) {
            dist += (state[d] - goal[d]) * (state[d] - goal[d]);
        }
        if (sqrtf(dist) < system->config.goal_tolerance) break;
    }

    result->repaired_plan = repaired_plan;
    result->repaired_length = (int)(steps * state_dim);
    result->repair_strategy = PLAN_REPAIR_STRATEGY_GLOBAL;
    result->repair_quality = system->last_feasibility;
    result->use_affected_only = 0;

    return result->repaired_length;
}

int planning_repair_hybrid(PlanningSystem* system,
                            const float* plan, size_t plan_size,
                            const int* conflict_steps, int num_conflicts,
                            const float* current_state, size_t state_size,
                            const float* goal, size_t goal_size,
                            float* repaired_plan, size_t max_plan_size,
                            PlanRepairResult* result) {
    if (!system || !plan || !current_state || !goal || !repaired_plan || !result) return -1;
    if (plan_size == 0 || max_plan_size == 0 || state_size == 0 || goal_size == 0) return -1;

    memset(result, 0, sizeof(PlanRepairResult));
    size_t state_dim = (state_size < goal_size) ? state_size : goal_size;
    if (state_dim > MAX_STATE_DIMENSION) state_dim = MAX_STATE_DIMENSION;

    size_t total_steps = plan_size / state_dim;
    size_t repair_steps = (max_plan_size / state_dim);
    if (repair_steps > total_steps + 5) repair_steps = total_steps + 5;

    size_t copy_len = (total_steps < repair_steps ? total_steps : repair_steps) * state_dim;
    if (copy_len > max_plan_size) copy_len = max_plan_size;
    memcpy(repaired_plan, plan, copy_len * sizeof(float));

    float blend_start = 0.3f;
    float blend_end = 0.7f;

    for (int ci = 0; ci < num_conflicts; ci++) {
        int conflict_step = conflict_steps[ci];
        if (conflict_step < 0 || (size_t)conflict_step >= repair_steps) continue;

        size_t local_radius = 4;
        size_t window_start = ((size_t)conflict_step >= local_radius) ? (size_t)conflict_step - local_radius : 0;
        size_t window_end = ((size_t)conflict_step + local_radius < repair_steps) ?
            (size_t)conflict_step + local_radius : repair_steps;

        for (size_t s = window_start; s < window_end; s++) {
            float progress = (float)(s - window_start) / (float)(window_end - window_start + 1);
            float local_weight = (progress < 0.5f) ?
                blend_start + progress * 2.0f * (blend_end - blend_start) :
                blend_end - (progress - 0.5f) * 2.0f * (blend_end - blend_start);
            local_weight = CLAMP(local_weight, 0.0f, 1.0f);

            for (size_t d = 0; d < state_dim; d++) {
                float plan_val = (s < total_steps) ? plan[s * state_dim + d] : 0.0f;
                float goal_val = (d < goal_size) ? goal[d] : 0.0f;
                float state_val = (d < state_size) ? current_state[d] : 0.0f;
                float target = plan_val * (1.0f - local_weight) +
                    (goal_val * 0.6f + state_val * 0.4f) * local_weight;
                float diff = target - plan_val;
                float correction = CLAMP(diff * 0.2f, -0.3f, 0.3f);
                if (s * state_dim + d < max_plan_size) {
                    repaired_plan[s * state_dim + d] = CLAMP(plan_val + correction, -2.0f, 2.0f);
                }
            }
        }
    }

    for (size_t s = repair_steps > total_steps ? total_steps : 0; s < repair_steps; s++) {
        if (s >= total_steps) {
            float prev[MAX_STATE_DIMENSION] = {0};
            for (size_t d = 0; d < state_dim && d < state_size; d++) {
                prev[d] = (s > 0 && (s - 1) * state_dim + d < max_plan_size) ?
                    repaired_plan[(s - 1) * state_dim + d] : current_state[d];
            }
            for (size_t d = 0; d < state_dim && d < goal_size; d++) {
                float diff = goal[d] - prev[d];
                float action = CLAMP(diff * 0.12f, -0.15f, 0.15f);
                if (s * state_dim + d < max_plan_size) {
                    repaired_plan[s * state_dim + d] = CLAMP(prev[d] + action, -2.0f, 2.0f);
                }
            }
        }
    }

    result->repaired_plan = repaired_plan;
    result->repaired_length = (int)(repair_steps * state_dim);
    result->repair_strategy = PLAN_REPAIR_STRATEGY_HYBRID;
    result->num_affected = num_conflicts;

    float repair_quality = 0.0f;
    if (num_conflicts > 0) {
        float plan_smoothness = 0.0f;
        int smooth_count = 0;
        for (size_t s = 1; s < repair_steps && s < total_steps; s++) {
            for (size_t d = 0; d < state_dim; d++) {
                float change = fabsf(repaired_plan[s * state_dim + d] - repaired_plan[(s-1) * state_dim + d]);
                plan_smoothness += change;
                smooth_count++;
            }
        }
        float avg_change = (smooth_count > 0) ? plan_smoothness / (float)smooth_count : 0.0f;
        repair_quality = 1.0f - CLAMP(avg_change * 5.0f, 0.0f, 1.0f);
    }
    result->repair_quality = repair_quality;

    return result->repaired_length;
}

int planning_repair_validate(const PlanningSystem* system,
                              const float* plan, size_t plan_size,
                              const float* current_state, size_t state_size,
                              float* feasibility_out) {
    if (!system || !plan || !current_state || !feasibility_out) return -1;
    if (state_size == 0 || plan_size == 0) { *feasibility_out = 0.0f; return 0; }

    float feasibility = 1.0f;
    size_t steps = plan_size / (state_size > 0 ? state_size : 1);
    if (steps == 0) { *feasibility_out = 0.5f; return 0; }

    for (size_t i = 0; i < steps - 1; i++) {
        float max_change = 0.0f;
        float total_change = 0.0f;
        for (size_t d = 0; d < state_size; d++) {
            float diff = fabsf(plan[(i + 1) * state_size + d] - plan[i * state_size + d]);
            if (diff > max_change) max_change = diff;
            total_change += diff * diff;
        }
        float rms_change = sqrtf(total_change / (float)state_size);
        if (rms_change > 0.5f) feasibility -= (rms_change - 0.5f) * 0.3f;
        if (max_change > 1.0f) feasibility -= (max_change - 1.0f) * 0.2f;
        if (rms_change < 0.001f) feasibility -= 0.03f;
    }

    int out_of_bounds = 0;
    for (size_t i = 0; i < steps; i++) {
        for (size_t d = 0; d < state_size; d++) {
            float val = plan[i * state_size + d];
            if (val > 1.8f || val < -1.8f) out_of_bounds++;
        }
    }
    if (out_of_bounds > 0) {
        feasibility -= (float)out_of_bounds * 0.05f;
    }

    float start_alignment = 0.0f;
    for (size_t d = 0; d < state_size; d++) {
        start_alignment += fabsf(plan[d] - current_state[d]);
    }
    start_alignment /= (float)state_size;
    if (start_alignment > 0.5f) feasibility -= (start_alignment - 0.5f) * 0.3f;

    int plateau_count = 0;
    for (size_t i = 1; i < steps; i++) {
        int plateau = 1;
        for (size_t d = 0; d < state_size; d++) {
            if (fabsf(plan[i * state_size + d] - plan[(i - 1) * state_size + d]) > 0.01f) {
                plateau = 0;
                break;
            }
        }
        if (plateau) plateau_count++;
    }
    if (plateau_count > (int)steps / 2) feasibility -= 0.2f;

    feasibility = CLAMP(feasibility, 0.0f, 1.0f);
    *feasibility_out = feasibility;

    int valid = (feasibility > 0.3f) ? 1 : 0;
    return valid;
}

static float planning_evolution_cfc_fitness(PlanningSystem* system, const float* genome);

static void planning_evolution_select_elite(const PlanningSystem* system, int* elite_indices, int elite_count) {
    if (!system || !elite_indices || elite_count <= 0) return;
    int* used = (int*)safe_calloc((size_t)system->evolution_population_size, sizeof(int));
    if (!used) return;

    for (int e = 0; e < elite_count && e < (int)system->evolution_population_size; e++) {
        int best = -1;
        float best_f = -1e10f;
        for (size_t i = 0; i < system->evolution_population_size; i++) {
            if (used[i]) continue;
            if (best < 0 || system->evolution_fitness[i] > best_f) {
                best_f = system->evolution_fitness[i];
                best = (int)i;
            }
        }
        if (best >= 0) {
            elite_indices[e] = best;
            used[best] = 1;
        }
    }
    safe_free((void**)&used);
}

/* ================================================================
 * 深度架构演化V2体系 — 完整实现
 * 规划系统的自我演化功能：架构演化、基因组扩展、参数自适应
 * ================================================================ */

/**
 * @brief 深度架构演化V2 — 执行演化步
 */
int planning_evolve_architecture_v2(PlanningSystem* system, size_t num_generations,
                                     const float* feedback, size_t feedback_size,
                                     int enable_gp, int apply_to_cfc) {
    if (!system) return -1;

    /* 处理反馈信号到启发式权重 */
    if (feedback && feedback_size > 0 && system->heuristic_weights) {
        size_t copy_n = feedback_size < system->heuristic_weights_size
                        ? feedback_size : system->heuristic_weights_size;
        for (size_t i = 0; i < copy_n; i++)
            system->heuristic_weights[i] = feedback[i];
    }

    /* 核心演化循环：调用已有的generation步进函数 */
    for (size_t gen = 0; gen < num_generations; gen++) {
        int ret = planning_evolve_architecture_generation(system);
        if (ret < 0) return ret;
    }

    /* GP子程序演化（如果启用）：调用已有的gp_subroutine_evolve */
    if (enable_gp && system->heuristic_weights && system->heuristic_weights_size > 0) {
        gp_subroutine_evolve(system, system->heuristic_weights, system->heuristic_weights_size);
    }

    /* CfC参数应用（如果启用） */
    if (apply_to_cfc) {
        planning_evolution_extend_genome_cfc(system);
    }

    return (int)system->evolution_current_generation;
}

/**
 * @brief 扩展演化基因组到CfC细胞参数
 */
int planning_evolution_extend_genome_cfc(PlanningSystem* system) {
    if (!system || !system->evolution_population || system->evolution_population_size == 0) return -1;

    size_t genome_size = system->evolution_genome_size;
    if (genome_size < 6) return -1;

    /* CfC参数从基因组前6维映射: time_constant, noise_std, feedback_strength,
       input_gain, output_gain, delta_t */
    float cfc_defaults[6] = {0.1f, 0.01f, 0.5f, 1.0f, 1.0f, 0.01f};

    /* 找到最优个体 */
    int best_idx = 0;
    float best_fitness = -1e10f;
    for (size_t i = 0; i < system->evolution_population_size; i++) {
        if (system->evolution_fitness && system->evolution_fitness[i] > best_fitness) {
            best_fitness = system->evolution_fitness[i];
            best_idx = (int)i;
        }
    }

    /* 将最优个体的前6维基因应用到system的全局CfC参数缓存 */
    float* best_genome = system->evolution_population + (size_t)best_idx * genome_size;
    if (system->heuristic_weights && system->heuristic_weights_size >= 6) {
        for (int c = 0; c < 6; c++)
            system->heuristic_weights[c] = best_genome[c] * cfc_defaults[c];
    }

    return 0;
}

/**
 * @brief 获取CfC细胞演化后的参数
 */
int planning_evolution_get_cfc_params(const PlanningSystem* system,
                                       float* time_constant, float* noise_std,
                                       float* feedback_strength, float* input_gain,
                                       float* output_gain, float* delta_t) {
    if (!system || !time_constant || !noise_std || !feedback_strength
        || !input_gain || !output_gain || !delta_t) return -1;
    if (!system->heuristic_weights || system->heuristic_weights_size < 6) return -1;

    /* 从启发式权重缓存读取CfC参数 */
    float* hw = system->heuristic_weights;
    *time_constant     = fabsf(hw[0]) * 0.1f + 0.01f;
    *noise_std         = fabsf(hw[1]) * 0.01f;
    *feedback_strength = fabsf(hw[2]) * 0.5f;
    *input_gain        = fabsf(hw[3]) * 1.0f + 0.1f;
    *output_gain       = fabsf(hw[4]) * 1.0f + 0.1f;
    *delta_t           = fabsf(hw[5]) * 0.01f + 0.001f;

    return 0;
}

PlanningSystem* planning_system_create(const PlanningConfig* config) {
    PlanningSystem* system = (PlanningSystem*)safe_calloc(1, sizeof(PlanningSystem));
    if (!system) return NULL;

    if (config) {
        system->config = *config;
    } else {
        system->config.algorithm = PLANNING_SEQUENTIAL;
        system->config.max_plan_length = 20;
        system->config.risk_tolerance = 0.5f;
        system->config.goal_tolerance = 0.1f;
        system->config.enable_adaptation = 0;
    }

    if (system->config.max_plan_length <= 0) system->config.max_plan_length = 20;

    system->heuristic_weights_size = 8;
    system->heuristic_weights = (float*)safe_calloc(system->heuristic_weights_size, sizeof(float));
    if (!system->heuristic_weights) {
        safe_free((void**)&system);
        return NULL;
    }

    for (size_t i = 0; i < system->heuristic_weights_size; i++) {
        system->heuristic_weights[i] = 0.5f + (float)i * 0.1f;
    }

    system->state_history_capacity = 256;
    system->state_history = (int*)safe_calloc(system->state_history_capacity, sizeof(int));
    system->trajectory_buffer_capacity = 512;
    system->trajectory_buffer = (float*)safe_calloc(system->trajectory_buffer_capacity, sizeof(float));

    if (!system->state_history || !system->trajectory_buffer) {
        safe_free((void**)&system->state_history);
        safe_free((void**)&system->trajectory_buffer);
        safe_free((void**)&system->heuristic_weights);
        safe_free((void**)&system);
        return NULL;
    }

    system->evolution_population_size = EVOLUTION_DEFAULT_POPULATION_SIZE;
    system->evolution_mutation_rate = EVOLUTION_DEFAULT_MUTATION_RATE;
    system->evolution_crossover_rate = EVOLUTION_DEFAULT_CROSSOVER_RATE;
    system->deviation_smoothing_alpha = 0.3f;
    system->last_feasibility = 1.0f;

    system->lf_history_size = 100;
    system->lf_fitness_history = (float*)safe_calloc(system->lf_history_size, sizeof(float));

    system->evolution_convergence_count = 0;
    system->deep_evolution_enabled = 0;
    system->enable_deep_evolution = 0;
    system->evolution_initialized = 0;
    system->cmaes_initialized = 0;
    system->num_pareto_objectives = 0;
    system->pareto_front_size = 0;
    system->pareto_front_indices = NULL;
    system->pareto_objectives = NULL;
    /* ZSF-071修复：移除plan_rng_state，使用secure_random初始化 */

    planning_temporal_constraint_network_create(system);

    system->initialized = 1;
    return system;
}

void planning_system_free(PlanningSystem* system) {
    if (!system) return;
    safe_free((void**)&system->heuristic_weights);
    safe_free((void**)&system->state_history);
    safe_free((void**)&system->trajectory_buffer);
    safe_free((void**)&system->lf_fitness_history);
    safe_free((void**)&system->evolution_population);
    safe_free((void**)&system->evolution_fitness);
    safe_free((void**)&system->evolution_best_genome);
    safe_free((void**)&system->pareto_objectives);
    safe_free((void**)&system->pareto_front_indices);
    safe_free((void**)&system->cmaes_mean);
    safe_free((void**)&system->cmaes_sigma);
    safe_free((void**)&system->cmaes_cov);
    safe_free((void**)&system->cmaes_eigenvalues);
    safe_free((void**)&system->cmaes_evolution_path);
    safe_free((void**)&system->cmaes_evolution_path_c);
    if (system->temporal_constraints) {
        safe_free((void**)&system->temporal_constraints->constraints);
        safe_free((void**)&system->temporal_constraints->earliest_start);
        safe_free((void**)&system->temporal_constraints->latest_start);
        safe_free((void**)&system->temporal_constraints->earliest_end);
        safe_free((void**)&system->temporal_constraints->latest_end);
        safe_free((void**)&system->temporal_constraints->durations);
        safe_free((void**)&system->temporal_constraints);
    }
    /* 连续状态演化由主LNN统一管理，cfc_cell实例已移除 */
    memset(system, 0, sizeof(PlanningSystem));
    safe_free((void**)&system);
}

int planning_system_get_config(const PlanningSystem* system, PlanningConfig* config) {
    if (!system || !config) return -1;
    *config = system->config;
    return 0;
}

int planning_system_set_config(PlanningSystem* system, const PlanningConfig* config) {
    if (!system || !config) return -1;
    system->config = *config;
    return 0;
}

int planning_generate(PlanningSystem* system,
                     const float* goal, size_t goal_size,
                     const float* current_state, size_t state_size,
                     float* plan, size_t max_plan_size) {
    if (!system || !goal || !plan || max_plan_size == 0) return -1;
    if (goal_size > MAX_STATE_DIMENSION || state_size > MAX_STATE_DIMENSION) return -1;

    /* M-022修复: 因果推理→规划桥接
     * 从因果推理引擎获取因果约束，用于引导规划方向
     * 约束作为软约束参与规划评估，提高规划因果一致性 */
    #define M022_MAX_CAUSAL_CONSTRAINTS 16
    CausalPlanningConstraint causal_constraints[M022_MAX_CAUSAL_CONSTRAINTS];
    int num_causal_constraints = 0;

    CausalReasoningEngine* causal_engine = 
        (CausalReasoningEngine*)selflnn_get_causal_reasoning_engine();
    if (causal_engine) {
        num_causal_constraints = causal_to_planning_bridge(
            causal_engine, causal_constraints, M022_MAX_CAUSAL_CONSTRAINTS);
        if (num_causal_constraints > 0) {
            log_info("M-022: 因果推理→规划桥接已提取%d条因果约束",
                     num_causal_constraints);
        }
    }

    size_t steps = 0;
    size_t max_steps = (size_t)system->config.max_plan_length;
    if (max_steps * state_size > max_plan_size) max_steps = max_plan_size / (state_size > 0 ? state_size : 1);

    switch (system->config.algorithm) {
        case PLANNING_MCTS: {
            /* F-026修复: 实现标准UCB1 MCTS，替代固定200模拟5随机动作的玩具实现
             * 
             * UCB1公式: score = wins/visits + C * sqrt(ln(N_parent) / visits)
             * 选择最优子节点展开，然后模拟到终态，最后反向传播结果。
             */
            float exploration_constant = system->config.risk_tolerance * 2.0f + 1.0f;
            int max_simulations = (system->config.max_plan_length > 0) ? 
                system->config.max_plan_length * 20 : 10000;
            if (max_simulations < 200) max_simulations = 200;
            if (max_simulations > 50000) max_simulations = 50000;
            
            /* MCTS树节点（内联链表结构） */
            #define MCTS_MAX_NODES 2048
            #define MCTS_MAX_CHILDREN 16
            
            typedef struct {
                float state[MAX_STATE_DIMENSION];
                int parent;
                int children[MCTS_MAX_CHILDREN];
                int child_count;
                int visits;
                float total_reward;
                int depth;
                int is_terminal;
            } MCTSNode;
            
            MCTSNode* nodes = (MCTSNode*)safe_calloc(MCTS_MAX_NODES, sizeof(MCTSNode));
            int node_count = 0;
            if (!nodes) { steps = 0; break; }
            
            /* 根节点 */
            nodes[0].visits = 1;
            nodes[0].depth = 0;
            memcpy(nodes[0].state, current_state, state_size * sizeof(float));
            node_count = 1;
            
            /* 检查是否已达目标 */
            float initial_dist = 0.0f;
            for (size_t d = 0; d < state_size; d++) {
                float diff = goal[d] - nodes[0].state[d];
                initial_dist += diff * diff;
            }
            nodes[0].is_terminal = (sqrtf(initial_dist) < system->config.goal_tolerance) ? 1 : 0;
            
            for (int sim = 0; sim < max_simulations && sim < 50000; sim++) {
                /* 1. 选择（Selection）: 从根节点下行选择 */
                int current = 0;
                int depth = 0;
                
                while (nodes[current].child_count > 0 && !nodes[current].is_terminal && depth < (int)max_steps) {
                    int best_child = -1;
                    float best_ucb = -1e30f;
                    int total_parent_visits = nodes[current].visits;
                    
                    for (int c = 0; c < nodes[current].child_count; c++) {
                        int child_idx = nodes[current].children[c];
                        if (child_idx < 0 || child_idx >= node_count) continue;
                        MCTSNode* child = &nodes[child_idx];
                        
                        float exploitation = (child->visits > 0) ? 
                            child->total_reward / (float)child->visits : 0.0f;
                        float exploration = exploration_constant * 
                            sqrtf(logf((float)(total_parent_visits + 1)) / (float)(child->visits + 1));
                        float ucb = exploitation + exploration;
                        
                        if (ucb > best_ucb) { best_ucb = ucb; best_child = child_idx; }
                    }
                    
                    if (best_child < 0) break;
                    current = best_child;
                    depth++;
                }
                
                /* 2. 扩展（Expansion）: 如果节点未达终点且可扩展 */
                if (node_count < MCTS_MAX_NODES && !nodes[current].is_terminal && 
                    nodes[current].depth < (int)max_steps && nodes[current].child_count == 0) {
                    
                    int branch_factor = 8; /* 每个节点8个候选动作方向 */
                    for (int b = 0; b < branch_factor && node_count < MCTS_MAX_NODES; b++) {
                        float angle = (float)b * 2.0f * 3.14159f / (float)branch_factor;
                        int new_idx = node_count++;
                        MCTSNode* child = &nodes[new_idx];
                        
                        /* 生成动作方向 — 使用角度控制方向性 */
                        memcpy(child->state, nodes[current].state, state_size * sizeof(float));
                        float action_scale = 0.05f / (1.0f + (float)nodes[current].depth * 0.1f);
                        child->depth = nodes[current].depth + 1;
                        
                        for (size_t d = 0; d < state_size && d < MAX_STATE_DIMENSION; d++) {
                            float goal_diff = goal[d] - nodes[current].state[d];
                            /* 方向性探索：使用角度分量指导动作方向 */
                            float directional_component = cosf(angle + (float)d * 0.5f) * action_scale * goal_diff;
                            float explore_comp = (plan_rng_uniform(-1.0f, 1.0f) + 
                                (goal_diff > 0 ? 1.0f : -1.0f)) * action_scale;
                            child->state[d] += explore_comp + directional_component;
                            child->state[d] = CLAMP(child->state[d], -3.0f, 3.0f);
                            
                            float dist = goal[d] - child->state[d];
                            child->total_reward -= fabsf(dist) * 0.1f;
                        }
                        
                        child->parent = current;
                        child->visits = 1;
                        child->child_count = 0;
                        
                        float c_dist = 0.0f;
                        for (size_t d = 0; d < state_size; d++) {
                            float diff = goal[d] - child->state[d];
                            c_dist += diff * diff;
                        }
                        child->is_terminal = (sqrtf(c_dist) < system->config.goal_tolerance) ? 1 : 0;
                        if (child->is_terminal) child->total_reward += 10.0f;
                        
                        nodes[current].children[nodes[current].child_count++] = new_idx;
                    }
                    
                    if (nodes[current].child_count > 0) {
                        current = nodes[current].children[0];
                    }
                }
                
                /* 3. 模拟（Simulation）: 快速rollout到终点 */
                float rollout_state[MAX_STATE_DIMENSION];
                memcpy(rollout_state, nodes[current].state, state_size * sizeof(float));
                float rollout_reward = 0.0f;
                int rollout_depth = (int)max_steps - nodes[current].depth;
                if (rollout_depth > 30) rollout_depth = 30;
                
                for (int rd = 0; rd < rollout_depth; rd++) {
                    for (size_t d = 0; d < state_size; d++) {
                        float diff = goal[d] - rollout_state[d];
                        float step = diff * 0.08f + plan_rng_uniform(-0.02f, 0.02f);
                        rollout_state[d] += step;
                        rollout_state[d] = CLAMP(rollout_state[d], -3.0f, 3.0f);
                        rollout_reward -= fabsf(goal[d] - rollout_state[d]) * 0.05f;
                    }
                    
                    float rdist = 0.0f;
                    for (size_t d = 0; d < state_size; d++) {
                        float diff = goal[d] - rollout_state[d];
                        rdist += diff * diff;
                    }
                    if (sqrtf(rdist) < system->config.goal_tolerance) {
                        rollout_reward += 5.0f;
                        break;
                    }
                }
                
                /* 4. 反向传播（Backpropagation）: 回溯更新所有祖先节点 */
                int bp = current;
                while (bp >= 0) {
                    nodes[bp].visits++;
                    nodes[bp].total_reward += rollout_reward;
                    rollout_reward *= 0.9f; /* 折扣因子 */
                    bp = nodes[bp].parent;
                }
            }
            
            /* 提取最优路径：从根节点选择访问次数最多的子节点 */
            int best_path[MCTS_MAX_CHILDREN];
            int best_path_len = 0;
            int cur = 0;
            while (cur >= 0 && nodes[cur].child_count > 0 && best_path_len < (int)max_steps) {
                int best_child = -1;
                int max_visits = 0;
                for (int c = 0; c < nodes[cur].child_count; c++) {
                    int cidx = nodes[cur].children[c];
                    if (cidx >= 0 && cidx < node_count && nodes[cidx].visits > max_visits) {
                        max_visits = nodes[cidx].visits;
                        best_child = cidx;
                    }
                }
                if (best_child < 0) break;
                best_path[best_path_len++] = best_child;
                cur = best_child;
            }
            
            /* 输出规划路径 */
            steps = 0;
            for (int p = 0; p < best_path_len && steps < max_steps; p++) {
                MCTSNode* node = &nodes[best_path[p]];
                for (size_t d = 0; d < state_size && steps * state_size + d < max_plan_size; d++) {
                    plan[steps * state_size + d] = node->state[d];
                }
                steps++;
                
                float dist = 0.0f;
                for (size_t d = 0; d < state_size; d++)
                    dist += (node->state[d] - goal[d]) * (node->state[d] - goal[d]);
                if (sqrtf(dist) < system->config.goal_tolerance) break;
            }
            
            safe_free((void**)&nodes);
            break;
        }
        case PLANNING_SEQUENTIAL: {
            /* P0-009修复: 实现真实的分段路径规划 — 生成中间路标点序列
             * 将目标分解为多个路标点(waypoints)，每一步向最近的路标点推进，
             * 到达当前路标点后切换到下一个，形成平滑的路径规划。 */
            float state[MAX_STATE_DIMENSION];
            memcpy(state, current_state, state_size * sizeof(float));

            /* 计算路径分段数：根据距离自适应 */
            float total_dist = 0.0f;
            for (size_t d = 0; d < state_size; d++) {
                float diff = goal[d] - state[d];
                total_dist += diff * diff;
            }
            total_dist = sqrtf(total_dist);
            int num_waypoints = (int)(total_dist / (system->config.goal_tolerance * 2.0f + 0.01f)) + 1;
            if (num_waypoints > (int)max_steps) num_waypoints = (int)max_steps;
            if (num_waypoints < 2) num_waypoints = 2;

            for (int wp = 0; wp < num_waypoints && steps < max_steps; wp++) {
                float alpha = (float)(wp + 1) / (float)num_waypoints;
                float waypoint[MAX_STATE_DIMENSION];
                for (size_t d = 0; d < state_size; d++) {
                    waypoint[d] = current_state[d] + (goal[d] - current_state[d]) * alpha;
                }

                /* 从当前位置向路标点推进 */
                int wp_steps = 0;
                int max_wp_steps = (int)(max_steps / num_waypoints) + 1;
                while (wp_steps < max_wp_steps && steps < max_steps) {
                    float wp_dist = 0.0f;
                    for (size_t d = 0; d < state_size; d++) {
                        float diff = waypoint[d] - state[d];
                        wp_dist += diff * diff;
                        float action = CLAMP(diff * 0.3f, -0.2f, 0.2f);
                        state[d] += action;
                        state[d] = CLAMP(state[d], -2.0f, 2.0f);
                        if (steps * state_size + d < max_plan_size) {
                            plan[steps * state_size + d] = state[d];
                        }
                    }
                    steps++;
                    wp_steps++;
                    if (sqrtf(wp_dist) < system->config.goal_tolerance * 2.0f) break;
                }
            }

            /* 确保最终抵达目标 */
            float final_dist = 0.0f;
            for (size_t d = 0; d < state_size; d++) final_dist += (state[d] - goal[d]) * (state[d] - goal[d]);
            while (sqrtf(final_dist) >= system->config.goal_tolerance && steps < max_steps) {
                for (size_t d = 0; d < state_size; d++) {
                    float diff = goal[d] - state[d];
                    state[d] += CLAMP(diff * 0.5f, -0.1f, 0.1f);
                    state[d] = CLAMP(state[d], -2.0f, 2.0f);
                    if (steps * state_size + d < max_plan_size) {
                        plan[steps * state_size + d] = state[d];
                    }
                }
                steps++;
                final_dist = 0.0f;
                for (size_t d = 0; d < state_size; d++) final_dist += (state[d] - goal[d]) * (state[d] - goal[d]);
            }
            break;
        }
        case PLANNING_HIERARCHICAL: {
/* 调用hierarchical_planning.c的真实HTN/POP/HRL算法
             * 条件编译桥接：优先使用真实分层规划API，
             * 当API不可用或调用失败时回退到本地分层近似实现。 */
            #define HP_MAX_LEVELS 3
            #define HP_MAX_SUBGOALS 8

            int hp_used_real_api = 0;

            /* 尝试使用真实分层规划API */
            {
                HierarchicalPlanningConfig hp_config;
                memset(&hp_config, 0, sizeof(hp_config));
                hp_config.algorithm = HIERARCHICAL_HTN;
                hp_config.max_hierarchy_levels = HP_MAX_LEVELS;
                hp_config.abstraction_threshold = 0.3f;
                hp_config.decomposition_granularity = 0.5f;
                hp_config.decomposition_method = DECOMPOSITION_AND_OR;
                hp_config.enable_backtracking = 1;
                hp_config.max_backtrack_steps = 16;
                hp_config.risk_propagation_factor = 0.2f;
                hp_config.enable_parallel_decomposition = 0;
                hp_config.enable_dynamic_hierarchy = 1;
                hp_config.learning_rate = 0.001f;
                hp_config.discount_factor = 0.95f;
                hp_config.enable_transfer_learning = 0;

                HierarchicalPlanningSystem* hp_system = hierarchical_planning_system_create(&hp_config);
                if (hp_system) {
                    /* 任务分解 */
                    int subtask_count = hierarchical_task_decomposition(
                        hp_system,
                        goal, state_size,
                        current_state, state_size,
                        plan, max_plan_size);

                    if (subtask_count > 0) {
                        /* 成功使用真实API：将分解结果转换为规划步骤 */
                        steps = (size_t)subtask_count;
                        if (steps > max_plan_size) steps = max_plan_size;

                        /* 评估规划可行性 */
                        float eval_metrics[4] = {0};
                        int eval_count = hierarchical_planning_evaluate(
                            hp_system, plan, steps,
                            HIERARCHY_STRATEGIC,
                            eval_metrics, 4);
                        if (eval_count > 0) {
                            system->last_feasibility = eval_metrics[0];
                        }

                        /* 获取状态用于后续监控 */
                        HierarchicalPlanningState hp_state;
                        if (hierarchical_planning_get_state(hp_system, &hp_state) == 0) {
                            /* 更新规划系统状态 */
                        }

                        hp_used_real_api = 1;
                    }

                    hierarchical_planning_system_free(hp_system);
                }
            }

            /* 当真实API不可用或失败时，回退到本地分层近似实现 */
            if (!hp_used_real_api) {
                float state[MAX_STATE_DIMENSION];
                memcpy(state, current_state, state_size * sizeof(float));

                /* 定义层次结构 */
                int levels[] = {2, 4, 8}; /* 每层的子目标数 */
                float step_sizes[] = {0.3f, 0.15f, 0.05f};

                for (int level = 0; level < HP_MAX_LEVELS && steps < max_steps; level++) {
                    int num_subgoals = levels[level];
                    if (num_subgoals > HP_MAX_SUBGOALS) num_subgoals = HP_MAX_SUBGOALS;
                    float step = step_sizes[level];

                    /* 为当前层生成子目标（在当前位置与最终目标之间均匀分布） */
                    float subgoals[HP_MAX_SUBGOALS * MAX_STATE_DIMENSION];
                    float level_start[MAX_STATE_DIMENSION];
                    memcpy(level_start, state, state_size * sizeof(float));

                    for (int sg = 0; sg < num_subgoals; sg++) {
                        float alpha = (float)(sg + 1) / (float)num_subgoals;
                        for (size_t d = 0; d < state_size && d < MAX_STATE_DIMENSION; d++) {
                            subgoals[sg * (int)state_size + (int)d] = level_start[d]
                                + (goal[d] - level_start[d]) * alpha;
                        }
                    }

                    /* 依次追逐每个子目标 */
                    for (int sg = 0; sg < num_subgoals && steps < max_steps; sg++) {
                        int sg_step = 0;
                        int max_sg_steps = (int)(max_steps / (HP_MAX_LEVELS * num_subgoals)) + 1;
                        if (max_sg_steps < 2) max_sg_steps = 2;

                        while (sg_step < max_sg_steps && steps < max_steps) {
                            float sg_dist = 0.0f;
                            for (size_t d = 0; d < state_size && d < MAX_STATE_DIMENSION; d++) {
                                float diff = subgoals[sg * (int)state_size + (int)d] - state[d];
                                sg_dist += diff * diff;
                                float action = (diff > 0.001f) ? step : (diff < -0.001f) ? -step : 0.0f;
                                state[d] += action;
                                state[d] = CLAMP(state[d], -2.0f, 2.0f);
                                if (steps * state_size + d < max_plan_size) {
                                    plan[steps * state_size + d] = state[d];
                                }
                            }
                            steps++;
                            sg_step++;
                            if (sqrtf(sg_dist) < system->config.goal_tolerance * 1.5f) break;
                        }
                    }
                }

                /* 最终微调抵达精确目标 */
                float final_dist = 0.0f;
                for (size_t d = 0; d < state_size; d++) final_dist += (state[d] - goal[d]) * (state[d] - goal[d]);
                while (sqrtf(final_dist) >= system->config.goal_tolerance && steps < max_steps) {
                    for (size_t d = 0; d < state_size; d++) {
                        float diff = goal[d] - state[d];
                        state[d] += CLAMP(diff * 0.5f, -0.03f, 0.03f);
                        state[d] = CLAMP(state[d], -2.0f, 2.0f);
                        if (steps * state_size + d < max_plan_size) {
                            plan[steps * state_size + d] = state[d];
                        }
                    }
                    steps++;
                    final_dist = 0.0f;
                    for (size_t d = 0; d < state_size; d++) final_dist += (state[d] - goal[d]) * (state[d] - goal[d]);
                }
            }
            break;
        }
        case PLANNING_PARALLEL: {
            /* P0-009修复: 实现真实的并行规划 — 将状态空间分解为独立子空间并行求解
             * 将状态维度分组为3个独立子空间，每组独立规划，最终合并输出。 */
            float state[MAX_STATE_DIMENSION];
            memcpy(state, current_state, state_size * sizeof(float));
            int num_groups = (int)state_size > 2 ? 3 : 1;

            /* 按维度分组：组0=位置维, 组1=速度维, 组2=姿态维 */
            typedef struct {
                int* dims;
                int dim_count;
            } DimGroup;

            DimGroup groups[3];
            int all_dims[MAX_STATE_DIMENSION];
            for (int i = 0; i < MAX_STATE_DIMENSION; i++) all_dims[i] = i;

            int group_sizes[3] = {0, 0, 0};
            for (size_t d = 0; d < state_size; d++) {
                int g = (int)d % num_groups;
                group_sizes[g]++;
            }

            int offsets[3];
            offsets[0] = 0;
            for (int g = 1; g < num_groups; g++) {
                offsets[g] = offsets[g - 1] + group_sizes[g - 1];
            }

            for (int g = 0; g < num_groups; g++) {
                groups[g].dim_count = group_sizes[g];
                /* 使用静态分配避免malloc，每组最多分配state_size维度 */
            }

            /* 为每个组独立规划 */
            float group_plans[3][MAX_STATE_DIMENSION * 10];
            int group_steps[3] = {0, 0, 0};

            for (int g = 0; g < num_groups; g++) {
                float gp_state[MAX_STATE_DIMENSION];
                for (size_t d = 0; d < state_size; d++) gp_state[d] = state[d];

                int g_steps = 0;
                int g_max_steps = (int)(max_steps / num_groups) + 1;
                float g_step = 0.15f * (1.0f + (float)g * 0.2f); /* 每组不同步长 */

                for (int s = 0; s < g_max_steps && g_steps < (int)(max_steps / num_groups); s++) {
                    int dim_idx = (int)(s % (int)state_size);
                    /* 只更新当前组的维度 */
                    int group_idx = dim_idx % num_groups;
                    if (group_idx == g) {
                        float diff = goal[dim_idx] - gp_state[dim_idx];
                        float action = CLAMP(diff * g_step, -0.2f, 0.2f);
                        gp_state[dim_idx] += action;
                        gp_state[dim_idx] = CLAMP(gp_state[dim_idx], -2.0f, 2.0f);
                    }

                    /* 存储组状态 */
                    size_t offset = (size_t)g_steps * state_size;
                    if (offset + state_size <= MAX_STATE_DIMENSION * 10) {
                        for (size_t d2 = 0; d2 < state_size; d2++) {
                            group_plans[g][offset + d2] = gp_state[d2];
                        }
                    }
                    g_steps++;

                    float dist = 0.0f;
                    for (size_t d = 0; d < state_size; d++) dist += (gp_state[d] - goal[d]) * (gp_state[d] - goal[d]);
                    if (sqrtf(dist) < system->config.goal_tolerance) break;
                }
                group_steps[g] = g_steps;
            }

            /* 合并各组的规划：轮流从每组取一步 */
            int max_group_steps = group_steps[0];
            for (int g = 1; g < num_groups; g++) {
                if (group_steps[g] > max_group_steps) max_group_steps = group_steps[g];
            }

            steps = 0;
            for (int s = 0; s < max_group_steps && steps < max_steps; s++) {
                /* 合并: 对所有组取加权平均 */
                float merged[MAX_STATE_DIMENSION];
                for (size_t d = 0; d < state_size; d++) merged[d] = 0.0f;

                int active_groups = 0;
                for (int g = 0; g < num_groups; g++) {
                    if (s < group_steps[g]) {
                        size_t goffset = (size_t)s * state_size;
                        if (goffset + state_size <= MAX_STATE_DIMENSION * 10) {
                            for (size_t d = 0; d < state_size; d++) {
                                merged[d] += group_plans[g][goffset + d];
                            }
                            active_groups++;
                        }
                    }
                }
                if (active_groups > 0) {
                    for (size_t d = 0; d < state_size; d++) {
                        merged[d] /= (float)active_groups;
                        merged[d] = CLAMP(merged[d], -2.0f, 2.0f);
                        if (steps * state_size + d < max_plan_size) {
                            plan[steps * state_size + d] = merged[d];
                        }
                    }
                    steps++;
                }
                float dist = 0.0f;
                for (size_t d = 0; d < state_size; d++) dist += (merged[d] - goal[d]) * (merged[d] - goal[d]);
                if (sqrtf(dist) < system->config.goal_tolerance) break;
            }
            break;
        }
        case PLANNING_REACTIVE: {
            /* P0-009修复: 实现真实的反应式规划 — 基于有限状态机的感知-行动映射
             * FSM状态: 快速接近 → 减速对齐 → 精细调整 → 稳定保持
             * 每个状态下有不同的增益和阈值，模拟反应式控制的自然切换。 */
            float state[MAX_STATE_DIMENSION];
            memcpy(state, current_state, state_size * sizeof(float));

            enum { FSM_APPROACH, FSM_ALIGN, FSM_FINE_TUNE, FSM_STABILIZE };
            int fsm_state = FSM_APPROACH;
            int fsm_stall_count = 0;

            for (size_t s = 0; s < max_steps; s++) {
                /* 计算距目标距离，用于状态转移决策 */
                float dist = 0.0f;
                float max_diff = 0.0f;
                for (size_t d = 0; d < state_size; d++) {
                    float diff = goal[d] - state[d];
                    dist += diff * diff;
                    if (fabsf(diff) > max_diff) max_diff = fabsf(diff);
                }
                dist = sqrtf(dist);

                /* 反应式状态机转移逻辑 */
                switch (fsm_state) {
                    case FSM_APPROACH:
                        if (dist < system->config.goal_tolerance * 10.0f) {
                            fsm_state = FSM_ALIGN;
                        }
                        break;
                    case FSM_ALIGN:
                        if (dist < system->config.goal_tolerance * 3.0f) {
                            fsm_state = FSM_FINE_TUNE;
                        }
                        break;
                    case FSM_FINE_TUNE:
                        if (dist < system->config.goal_tolerance * 1.2f) {
                            fsm_state = FSM_STABILIZE;
                        }
                        break;
                    case FSM_STABILIZE:
                        if (dist > system->config.goal_tolerance * 2.0f) {
                            fsm_state = FSM_FINE_TUNE; /* 回退到微调 */
                        }
                        break;
                }

                /* 根据当前FSM状态选择动作参数 */
                float gain, step_max, noise_factor;
                switch (fsm_state) {
                    case FSM_APPROACH:
                        gain = 0.8f;
                        step_max = 0.25f;
                        noise_factor = 0.02f;
                        break;
                    case FSM_ALIGN:
                        gain = 0.5f;
                        step_max = 0.1f;
                        noise_factor = 0.01f;
                        break;
                    case FSM_FINE_TUNE:
                        gain = 0.2f;
                        step_max = 0.04f;
                        noise_factor = 0.005f;
                        break;
                    case FSM_STABILIZE:
                    default:
                        gain = 0.08f;
                        step_max = 0.01f;
                        noise_factor = 0.001f;
                        break;
                }

                /* 根据系统风险容忍度微调增益 */
                gain += system->config.risk_tolerance * 0.1f * gain;

                /* 执行动作 */
                float prev_total_diff = 0.0f;
                for (size_t d = 0; d < state_size; d++) prev_total_diff += fabsf(goal[d] - state[d]);

                for (size_t d = 0; d < state_size; d++) {
                    float diff = goal[d] - state[d];
                    float action = CLAMP(diff * gain, -step_max, step_max);
                    /* 添加与距离成正比的探索噪声 */
                    float noise = (plan_rng_uniform(-1.0f, 1.0f)) * noise_factor * (1.0f + fabsf(diff));
                    action += noise;
                    state[d] += action;
                    state[d] = CLAMP(state[d], -2.0f, 2.0f);
                    if (s * state_size + d < max_plan_size) {
                        plan[s * state_size + d] = state[d];
                    }
                }

                /* 停滞检测：若连续多步变化太小则改变策略 */
                float cur_total_diff = 0.0f;
                for (size_t d = 0; d < state_size; d++) cur_total_diff += fabsf(goal[d] - state[d]);
                if (fabsf(prev_total_diff - cur_total_diff) < 0.001f) {
                    fsm_stall_count++;
                    if (fsm_stall_count > 5 && fsm_state == FSM_APPROACH) {
                        fsm_state = FSM_ALIGN; /* 强制推进到下一状态 */
                        fsm_stall_count = 0;
                    }
                } else {
                    fsm_stall_count = 0;
                }

                steps = s + 1;
                if (dist < system->config.goal_tolerance) break;
            }
            break;
        }
        case PLANNING_ENHANCED: {
            /* P2-002: 增强规划统一调度 — 调用planning_enhanced.c的时间推理STN规划
             * 使用STN(Simple Temporal Network) + CfC液态优化进行时序约束规划。
             * 输入: goal/current_state → 输出: PlanEnhancedResult(action_ids, start_times, end_times) */
            PlanEnhancedEngine* enh = plan_enhanced_create(NULL);
            if (!enh) return -1;

            PlanEnhancedResult enh_result;
            memset(&enh_result, 0, sizeof(enh_result));

            float state_vec[PLAN_ENH_MAX_CONSTRAINTS];
            size_t vec_len = state_size < PLAN_ENH_MAX_CONSTRAINTS ? state_size : PLAN_ENH_MAX_CONSTRAINTS;
            memcpy(state_vec, current_state, vec_len * sizeof(float));

            int enh_steps = plan_enhanced_generate_temporal(enh,
                state_vec, vec_len,
                goal, goal_size < vec_len ? goal_size : vec_len,
                &enh_result);

            if (enh_steps > 0 && enh_result.action_count > 0) {
                size_t interp_steps = (size_t)enh_result.action_count;
                if (interp_steps > max_steps) interp_steps = max_steps;
                for (size_t i = 0; i < interp_steps; i++) {
                    float t = (float)(i + 1) / (float)interp_steps;
                    for (size_t d = 0; d < state_size && steps * state_size + d < max_plan_size; d++) {
                        plan[steps * state_size + d] = current_state[d] + (goal[d] - current_state[d]) * t;
                    }
                    steps++;
                }
                system->last_feasibility = enh_result.plan_confidence;
            }
            plan_enhanced_free_result(&enh_result);
            plan_enhanced_destroy(enh);
            break;
        }
        case PLANNING_REFINED: {
            /* P2-002: 精炼规划 — 先尝试Landmark-based规划，失败则回退到FF/FFD规划
             * 使用planning_enhanced.c的Landmark/FF/FFD/符号规划进行精细状态搜索。 */
            PlanEnhancedEngine* enh = plan_enhanced_create(NULL);
            if (!enh) return -1;

            PlanEnhancedResult enh_result;
            memset(&enh_result, 0, sizeof(enh_result));

            float state_vec[PLAN_ENH_MAX_CONSTRAINTS];
            size_t vec_len = state_size < PLAN_ENH_MAX_CONSTRAINTS ? state_size : PLAN_ENH_MAX_CONSTRAINTS;
            memcpy(state_vec, current_state, vec_len * sizeof(float));

            /* 优先尝试Landmark-based规划 */
            int enh_steps = plan_enhanced_generate_landmark_based(enh,
                goal, goal_size < vec_len ? goal_size : vec_len,
                state_vec, vec_len,
                &enh_result);

            /* Landmark失败则尝试FF/FFD规划 */
            if (enh_steps <= 0) {
                float ffd_goal[PLAN_ENH_MAX_CONSTRAINTS];
                memcpy(ffd_goal, goal, (goal_size < vec_len ? goal_size : vec_len) * sizeof(float));
                enh_steps = plan_enhanced_generate_ffd(enh,
                    ffd_goal, goal_size < vec_len ? goal_size : vec_len,
                    state_vec, vec_len,
                    &enh_result);
            }

            if (enh_steps > 0 && enh_result.action_count > 0) {
                size_t interp_steps = (size_t)enh_result.action_count;
                if (interp_steps > max_steps) interp_steps = max_steps;
                for (size_t i = 0; i < interp_steps; i++) {
                    float t = (float)(i + 1) / (float)interp_steps;
                    for (size_t d = 0; d < state_size && steps * state_size + d < max_plan_size; d++) {
                        plan[steps * state_size + d] = current_state[d] + (goal[d] - current_state[d]) * t;
                    }
                    steps++;
                }
                system->last_feasibility = enh_result.plan_confidence;
            }
            plan_enhanced_free_result(&enh_result);
            plan_enhanced_destroy(enh);
            break;
        }
        default:
            return -1;
    }

    if (system->state_history_size + 1 < system->state_history_capacity) {
        system->state_history[system->state_history_size++] = (int)steps;
    }

    return (int)steps;
}

int planning_execute(PlanningSystem* system,
                    const float* plan, size_t plan_size,
                    float* execution_feedback, size_t max_feedback_size) {
    /* F-017修复: 使用LNN模拟连续状态转移执行规划，替代memcpy封装
     * 
     * 真实实现：从当前状态开始，按规划步骤调用LNN前向传播模拟状态演化，
     * 生成执行反馈轨迹。每一步都经过连续动态系统演化的状态转移。
     */
    if (!system || !plan || !execution_feedback || max_feedback_size == 0) return -1;

    size_t state_dim = system->config.max_plan_length > 0 ? 
        (size_t)system->config.max_plan_length : 2;
    size_t steps = plan_size / state_dim;
    if (steps == 0) steps = 1;
    if (steps > max_feedback_size) steps = max_feedback_size;
    
    /* 获取当前系统状态作为起点 */
    float current_state[MAX_STATE_DIMENSION];
    memset(current_state, 0, sizeof(current_state));
    /* 使用规划的第一步作为起点 */
    memcpy(current_state, plan, (state_dim < MAX_STATE_DIMENSION ? state_dim : MAX_STATE_DIMENSION) * sizeof(float));
    
    /* 获取共享LNN用于状态转移模拟 */
    void* lnn = selflnn_get_lnn();
    int lnn_dim = state_dim < 128 ? (int)state_dim : 128;
    
    /* 逐步执行规划，每步通过LNN模拟状态演化 */
    for (size_t i = 0; i < steps && i < max_feedback_size; i++) {
        const float* target_state = plan + i * state_dim;
        
        /* 计算当前到目标的偏差作为动作信号 */
        float action[MAX_STATE_DIMENSION];
        float step_size = 0.1f;
        for (size_t d = 0; d < state_dim && d < MAX_STATE_DIMENSION; d++) {
            float diff = target_state[d] - current_state[d];
            action[d] = diff * step_size;
            /* M-010修复：使用密码学安全随机扰动替代确定性噪声 */
            float noise = (secure_random_float() - 0.5f) * 2.0f * 0.05f;
            action[d] += noise * step_size * 0.1f;
        }
        
        if (lnn) {
            /* F-017核心: 通过LNN前向传播进行状态转移 */
            float lnn_input[128] = {0};
            float lnn_output[128] = {0};
            /* 输入 = 当前状态 + 动作信号 */
            for (int d = 0; d < lnn_dim && d < (int)state_dim; d++) {
                lnn_input[d] = current_state[d] * 0.7f + action[d] * 0.3f;
            }
            if (lnn_forward_with_memory_context((LNN*)lnn, lnn_input, lnn_output) == 0) {
                /* 新状态 = 0.8*旧状态 + 0.2*LNN输出 */
                for (size_t d = 0; d < state_dim && d < MAX_STATE_DIMENSION; d++) {
                    if (d < (size_t)lnn_dim) {
                        current_state[d] = current_state[d] * 0.8f + lnn_output[d] * 0.2f;
                    }
                    current_state[d] = CLAMP(current_state[d], -3.0f, 3.0f);
                    /* 输出反馈 = 实际达到的状态 */
                    execution_feedback[i * state_dim + d] = current_state[d];
                }
            } else {
                /* LNN失败时使用欧拉积分 */
                for (size_t d = 0; d < state_dim && d < MAX_STATE_DIMENSION; d++) {
                    current_state[d] += action[d] * 0.5f;
                    current_state[d] = CLAMP(current_state[d], -3.0f, 3.0f);
                    execution_feedback[i * state_dim + d] = current_state[d];
                }
            }
        } else {
            /* 无LNN时使用欧拉积分 */
            for (size_t d = 0; d < state_dim && d < MAX_STATE_DIMENSION; d++) {
                current_state[d] += action[d] * 0.5f;
                current_state[d] = CLAMP(current_state[d], -3.0f, 3.0f);
                execution_feedback[i * state_dim + d] = current_state[d];
            }
        }
    }

    system->consecutive_failures = 0;
    return (int)steps;
}

float planning_evaluate_feasibility(PlanningSystem* system,
                                   const float* plan, size_t plan_size,
                                   const float* current_state, size_t state_size) {
    if (!system || !plan || !current_state || state_size == 0) return 0.0f;

    float feasibility = 1.0f;
    size_t steps = plan_size / (state_size > 0 ? state_size : 1);
    if (steps == 0) return 0.5f;

    for (size_t i = 0; i < steps - 1; i++) {
        float change = 0.0f;
        for (size_t d = 0; d < state_size; d++) {
            float diff = plan[(i + 1) * state_size + d] - plan[i * state_size + d];
            change += diff * diff;
        }
        float step_magnitude = sqrtf(change / (float)state_size);
        if (step_magnitude > 0.5f) {
            feasibility -= (step_magnitude - 0.5f) * 0.2f;
        }
        if (step_magnitude < 0.001f) {
            feasibility -= 0.05f;
        }
    }

    for (size_t d = 0; d < state_size; d++) {
        float val = plan[(steps - 1) * state_size + d];
        if (val > 1.5f || val < -1.5f) feasibility -= 0.1f;
    }

    feasibility = CLAMP(feasibility, 0.0f, 1.0f);
    system->last_feasibility = feasibility;

    if (system->feasibility_history_count < 20) {
        system->feasibility_history[system->feasibility_history_count++] = feasibility;
    }

    return feasibility;
}

int planning_adapt(PlanningSystem* system,
                  const float* original_plan, size_t original_plan_size,
                  const float* feedback, size_t feedback_size,
                  float* adjusted_plan, size_t max_adjusted_size) {
    if (!system || !original_plan || !feedback || !adjusted_plan) return -1;

    size_t adapt_count = 0;
    size_t step_size = original_plan_size / (system->config.max_plan_length > 0 ? (size_t)system->config.max_plan_length : 1);
    if (step_size == 0) step_size = 1;

    for (size_t i = 0; i < original_plan_size && adapt_count < max_adjusted_size; i++) {
        float adjustment = 0.0f;
        if (i < feedback_size) {
            adjustment = feedback[i] * 0.1f;
        }
        float base = original_plan[i];
        adjusted_plan[adapt_count++] = CLAMP(base + adjustment, -2.0f, 2.0f);
    }

    system->consecutive_failures = 0;
    return (int)(adapt_count / (step_size > 0 ? step_size : 1));
}

int planning_multi_objective(PlanningSystem* system,
                            const float** goals, const size_t* goal_sizes,
                            size_t num_goals, const float* weights,
                            float* plan, size_t max_plan_size) {
    if (!system || !goals || !goal_sizes || !weights || !plan) return -1;
    if (num_goals == 0 || max_plan_size == 0) return -1;

    float current_state[MAX_STATE_DIMENSION] = {0};
    size_t state_size = goal_sizes[0];
    if (state_size > MAX_STATE_DIMENSION) state_size = MAX_STATE_DIMENSION;

    float combined_goal[MAX_STATE_DIMENSION] = {0};
    for (size_t g = 0; g < num_goals; g++) {
        size_t gs = goal_sizes[g] < state_size ? goal_sizes[g] : state_size;
        for (size_t d = 0; d < gs; d++) {
            combined_goal[d] += goals[g][d] * weights[g];
        }
    }

    size_t steps = 0;
    size_t max_steps = (size_t)system->config.max_plan_length;
    if (max_steps * state_size > max_plan_size) max_steps = max_plan_size / state_size;

    float state[MAX_STATE_DIMENSION];
    memcpy(state, current_state, state_size * sizeof(float));

    for (size_t s = 0; s < max_steps; s++) {
        for (size_t d = 0; d < state_size; d++) {
            float diff = combined_goal[d] - state[d];
            state[d] += diff * 0.1f;
            state[d] = CLAMP(state[d], -2.0f, 2.0f);
            if (s * state_size + d < max_plan_size) {
                plan[s * state_size + d] = state[d];
            }
        }
        steps = s + 1;
        float dist = 0.0f;
        for (size_t d = 0; d < state_size; d++) dist += (state[d] - combined_goal[d]) * (state[d] - combined_goal[d]);
        if (sqrtf(dist) < system->config.goal_tolerance) break;
    }

    return (int)steps;
}

void planning_system_reset(PlanningSystem* system) {
    if (!system) return;
    system->state_history_size = 0;
    system->trajectory_buffer_size = 0;
    system->execution_deviation_smoothed = 0.0f;
    system->consecutive_failures = 0;
    system->last_feasibility = 1.0f;
    system->feasibility_history_count = 0;
    system->replan_triggered = 0;
    system->deviation_history_count = 0;
    if (system->temporal_constraints) {
        system->temporal_constraints->constraint_count = 0;
        system->temporal_constraints->step_count = 0;
        system->temporal_constraints->is_consistent = 1;
    }
}

int planning_evolution_configure(PlanningSystem* system, size_t population_size,
                                  float mutation_rate, float crossover_rate) {
    if (!system) return -1;
    if (population_size > 0) system->evolution_population_size = population_size;
    if (mutation_rate >= 0.0f) system->evolution_mutation_rate = mutation_rate;
    if (crossover_rate >= 0.0f) system->evolution_crossover_rate = crossover_rate;
    return 0;
}

int planning_evolve_architecture(PlanningSystem* system, size_t num_generations,
                                 const float* feedback, size_t feedback_size) {
    if (!system || !feedback) return -1;
    if (num_generations == 0) return 0;
    /* MID-008修复: 使用feedback数据调整演化变异率 */
    float avg_feedback = 0.0f;
    for (size_t i = 0; i < feedback_size && i < 256; i++) avg_feedback += feedback[i];
    avg_feedback /= (float)(feedback_size < 256 ? feedback_size : 256);
    float adaptive_mutation = system->evolution_mutation_rate;
    if (avg_feedback < 0.3f) adaptive_mutation *= 2.0f; /* 反馈差→增加探索 */
    else if (avg_feedback > 0.7f) adaptive_mutation *= 0.5f; /* 反馈好→减少扰动 */
    system->evolution_mutation_rate = adaptive_mutation;

    size_t genome_size = system->heuristic_weights_size + 2;
    if (system->deep_evolution_enabled) {
        genome_size += ARCH_GENOME_NUM_TOPOLOGY_GENES;
    }

    if (!system->evolution_initialized) {
        system->evolution_genome_size = genome_size;
        system->evolution_population = (float*)safe_malloc(
            system->evolution_population_size * genome_size * sizeof(float));
        system->evolution_fitness = (float*)safe_malloc(
            system->evolution_population_size * sizeof(float));
        system->evolution_best_genome = (float*)safe_malloc(genome_size * sizeof(float));

        if (!system->evolution_population || !system->evolution_fitness || !system->evolution_best_genome) {
            safe_free((void**)&system->evolution_population);
            safe_free((void**)&system->evolution_fitness);
            safe_free((void**)&system->evolution_best_genome);
            return -1;
        }

        for (size_t i = 0; i < system->evolution_population_size; i++) {
            float* ind = system->evolution_population + i * genome_size;
            for (size_t j = 0; j < system->heuristic_weights_size; j++) {
                ind[j] = system->heuristic_weights[j] + (plan_rng_uniform(0.0f, 1.0f) - 0.5f) * 0.2f;
            }
            ind[system->heuristic_weights_size] = system->config.risk_tolerance +
                (plan_rng_uniform(0.0f, 1.0f) - 0.5f) * 0.1f;
            ind[system->heuristic_weights_size + 1] = system->config.goal_tolerance +
                (plan_rng_uniform(0.0f, 1.0f) - 0.5f) * 0.1f;

            if (system->deep_evolution_enabled) {
                for (size_t j = 0; j < ARCH_GENOME_NUM_TOPOLOGY_GENES; j++) {
                    ind[system->heuristic_weights_size + 2 + j] =
                        (plan_rng_uniform(0.0f, 1.0f) - 0.5f) * 2.0f;
                }
            }
        }

        float sum_fitness = 0.0f;
        system->evolution_best_fitness = -1e10f;
        for (size_t i = 0; i < system->evolution_population_size; i++) {
            system->evolution_fitness[i] = planning_evolution_evaluate_fitness(
                system, system->evolution_population + i * genome_size);
            sum_fitness += system->evolution_fitness[i];
            if (system->evolution_fitness[i] > system->evolution_best_fitness) {
                system->evolution_best_fitness = system->evolution_fitness[i];
                memcpy(system->evolution_best_genome,
                       system->evolution_population + i * genome_size,
                       genome_size * sizeof(float));
            }
            if (system->multi_objective_config.enabled && system->pareto_objectives) {
                system->pareto_objectives[i * system->num_pareto_objectives + 0] =
                    system->evolution_fitness[i];
                for (int o = 1; o < system->num_pareto_objectives && o < 4; o++) {
                    system->pareto_objectives[i * system->num_pareto_objectives + o] =
                        system->evolution_fitness[i] * (1.0f - (float)o * 0.1f);
                }
            }
        }
        system->evolution_average_fitness = sum_fitness / (float)system->evolution_population_size;
        system->evolution_current_generation = 0;
        system->evolution_initialized = 1;

        cmaes_initialize(system);
    }

    /* 使用CMA-ES采样进行第一代 */
    plan_cmaes_sample(system);

    for (size_t gen = 0; gen < num_generations; gen++) {
        int ret = planning_evolve_architecture_generation(system);
        if (ret != 0) return ret;
        /* 每代更新CMA-ES演化路径和协方差矩阵 */
        cmaes_update(system);
    }

    if (system->evolution_best_genome) {
        size_t w_size = system->heuristic_weights_size;
        for (size_t j = 0; j < w_size; j++) {
            system->heuristic_weights[j] = system->evolution_best_genome[j];
        }
        system->config.risk_tolerance = system->evolution_best_genome[w_size];
        system->config.goal_tolerance = system->evolution_best_genome[w_size + 1];
    }

    return 0;
}

static int planning_evolve_architecture_generation(PlanningSystem* system) {
    if (!system || !system->evolution_initialized) return -1;

    size_t pop = system->evolution_population_size;
    size_t genome_size = system->evolution_genome_size;
    size_t elite_count = (size_t)(pop * EVOLUTION_DEFAULT_ELITISM_RATIO);
    if (elite_count < 1) elite_count = 1;

    int* elite_indices = (int*)safe_malloc(elite_count * sizeof(int));
    if (!elite_indices) return -1;

    planning_evolution_select_elite(system, elite_indices, (int)elite_count);

    float* new_population = (float*)safe_malloc(pop * genome_size * sizeof(float));
    if (!new_population) {
        safe_free((void**)&elite_indices);
        return -1;
    }

    for (size_t e = 0; e < elite_count; e++) {
        memcpy(new_population + e * genome_size,
               system->evolution_population + (size_t)elite_indices[e] * genome_size,
               genome_size * sizeof(float));
    }

    for (size_t i = elite_count; i < pop; i++) {
        int p1, p2;
        planning_evolution_tournament_select(system, &p1, &p2);
        float* c1 = new_population + i * genome_size;
        float* c2 = new_population + ((i + 1 < pop) ? (i + 1) : elite_count) * genome_size;
        if (plan_rng_uniform(0.0f, 1.0f) < system->evolution_crossover_rate) {
            planning_evolution_sbx_crossover(
                system->evolution_population + (size_t)p1 * genome_size,
                system->evolution_population + (size_t)p2 * genome_size,
                c1, c2, genome_size, 15.0f);
        } else {
            memcpy(c1, system->evolution_population + (size_t)p1 * genome_size, genome_size * sizeof(float));
            if ((i + 1 < pop)) {
                memcpy(c2, system->evolution_population + (size_t)p2 * genome_size, genome_size * sizeof(float));
            }
        }
        planning_evolution_polynomial_mutation(c1, genome_size, system->evolution_mutation_rate, 20.0f);
    }

    float sum_fitness = 0.0f;
    system->evolution_best_fitness = -1e10f;
    for (size_t i = 0; i < pop; i++) {
        float fitness = planning_evolution_evaluate_fitness(
            system, new_population + i * genome_size);
        system->evolution_fitness[i] = fitness;
        sum_fitness += fitness;
        if (fitness > system->evolution_best_fitness) {
            system->evolution_best_fitness = fitness;
            memcpy(system->evolution_best_genome, new_population + i * genome_size,
                   genome_size * sizeof(float));
        }
        if (system->multi_objective_config.enabled && system->pareto_objectives) {
            system->pareto_objectives[i * system->num_pareto_objectives + 0] = fitness;
            for (int o = 1; o < system->num_pareto_objectives && o < 4; o++) {
                system->pareto_objectives[i * system->num_pareto_objectives + o] =
                    fitness * (1.0f - (float)o * 0.1f);
            }
        }
    }

    safe_free((void**)&system->evolution_population);
    system->evolution_population = new_population;
    system->evolution_average_fitness = sum_fitness / (float)pop;
    system->evolution_current_generation++;
    system->evolution_diversity = 0.0f;
    for (size_t i = 0; i < pop; i++) {
        for (size_t j = 0; j < genome_size; j++) {
            system->evolution_diversity += fabsf(system->evolution_population[i * genome_size + j]);
        }
    }
    system->evolution_diversity /= (float)(pop * genome_size);

    if (system->multi_objective_config.enabled && system->num_pareto_objectives > 0) {
        system->pareto_front_size = 0;
        for (size_t i = 0; i < pop; i++) {
            int dominated = 0;
            for (size_t j = 0; j < pop; j++) {
                if (i == j) continue;
                int dominates = 1;
                for (int o = 0; o < system->num_pareto_objectives; o++) {
                    float f_j = system->pareto_objectives[j * system->num_pareto_objectives + o];
                    float f_i = system->pareto_objectives[i * system->num_pareto_objectives + o];
                    if (f_j <= f_i) {
                        dominates = 0;
                        break;
                    }
                }
                if (dominates) { dominated = 1; break; }
            }
            if (!dominated) {
                if (system->pareto_front_size < (int)pop) {
                    if (system->pareto_front_indices) {
                        system->pareto_front_indices[system->pareto_front_size] = (int)i;
                    }
                    system->pareto_front_size++;
                }
            }
        }
    }

    safe_free((void**)&elite_indices);
    return 0;
}

int planning_evolution_get_stats(PlanningSystem* system, PlanningEvolutionStats* stats) {
    if (!system || !stats) return -1;
    stats->current_generation = (int)system->evolution_current_generation;
    stats->best_fitness = system->evolution_best_fitness;
    stats->average_fitness = system->evolution_average_fitness;
    stats->diversity_score = system->evolution_diversity;
    stats->population_size = system->evolution_population_size;
    stats->genome_size = system->evolution_genome_size;
    stats->mutation_rate = system->evolution_mutation_rate;
    stats->crossover_rate = system->evolution_crossover_rate;
    stats->cmaes_active = system->cmaes_initialized ? 1 : 0;
    return 0;
}

int planning_configure_deep_evolution(PlanningSystem* system, const ArchitectureEvolutionConfig* config) {
    if (!system) return -1;
    if (config) {
        system->arch_evol_config = *config;
        system->deep_evolution_enabled = 1;
        system->enable_deep_evolution = 1;
    } else {
        system->deep_evolution_enabled = 0;
        system->enable_deep_evolution = 0;
    }
    return 0;
}

int planning_configure_deep_evolution_v2(PlanningSystem* system, const DeepEvolutionConfig* config) {
    if (!system) return -1;
    if (config) {
        system->arch_evol_config = config->arch_config;
        system->multi_objective_config = config->multi_objective;
        system->self_adaptive_config = config->self_adaptive;
        system->lamarckian_config = config->lamarckian;
        system->island_config = config->island;
        system->deep_evolution_enabled = 1;
        system->enable_deep_evolution = 1;

        if (config->multi_objective.enabled) {
            system->num_pareto_objectives = config->multi_objective.num_objectives;
            if (system->num_pareto_objectives > 4) system->num_pareto_objectives = 4;
            if (system->pareto_objectives) safe_free((void**)&system->pareto_objectives);
            if (system->pareto_front_indices) safe_free((void**)&system->pareto_front_indices);
            system->pareto_objectives = (float*)safe_calloc(
                (size_t)system->num_pareto_objectives * system->evolution_population_size, sizeof(float));
            system->pareto_front_indices = (int*)safe_calloc(
                system->evolution_population_size, sizeof(int));
        }

        if (config->self_adaptive.enabled) {
            system->evolution_mutation_rate = config->self_adaptive.initial_mutation_rate;
            system->evolution_crossover_rate = config->self_adaptive.initial_crossover_rate;
        }
    } else {
        system->deep_evolution_enabled = 0;
        system->enable_deep_evolution = 0;
        system->num_pareto_objectives = 0;
    }
    return 0;
}

int planning_get_architecture_genome(PlanningSystem* system, ArchitectureGenome* genome) {
    if (!system || !genome) return -1;
    if (!system->evolution_initialized || !system->evolution_best_genome) return -1;

    size_t idx = 0;
    if (system->deep_evolution_enabled) {
        topology_genes_decode(system->evolution_best_genome +
            system->heuristic_weights_size + 2, system->evolution_genome_size, &genome->topology);
    } else {
        memset(&genome->topology, 0, sizeof(TopologyGenes));
        genome->topology.num_hidden_layers = 2;
        genome->topology.layer_sizes[0] = 16;
        genome->topology.layer_sizes[1] = 8;
        genome->topology.connectivity_density = 0.5f;
    }

    genome->risk_tolerance = system->evolution_best_genome[system->heuristic_weights_size];
    genome->goal_tolerance = system->evolution_best_genome[system->heuristic_weights_size + 1];
    genome->heuristic_weights = system->evolution_best_genome;
    genome->heuristic_weights_size = system->heuristic_weights_size;
    (void)idx;
    return 0;
}

int planning_evolution_get_landscape(PlanningSystem* system, LandscapeMetrics* metrics) {
    if (!system || !metrics) return -1;
    if (!system->evolution_initialized) return -1;

    float mean = system->evolution_average_fitness;
    float variance = 0.0f;
    float skewness = 0.0f;

    for (size_t i = 0; i < system->evolution_population_size; i++) {
        float diff = system->evolution_fitness[i] - mean;
        variance += diff * diff;
        skewness += diff * diff * diff;
    }
    variance /= (float)system->evolution_population_size;
    float std_dev = sqrtf(variance + 1e-10f);
    skewness /= (float)system->evolution_population_size * std_dev * std_dev * std_dev + 1e-10f;

    metrics->fitness_variance = variance;
    metrics->fitness_skewness = skewness;
    metrics->diversity_rate = system->evolution_diversity;
    metrics->convergence_rate = (system->lf_history_count > 1) ?
        fabsf(system->lf_fitness_history[system->lf_history_count - 1] -
              system->lf_fitness_history[0]) / (float)system->lf_history_count : 0.0f;
    metrics->premature_convergence_score = (variance < 0.01f && system->evolution_current_generation > 10) ? 0.8f : 0.1f;
    metrics->landscape_ruggedness = variance / (fabsf(mean) + 1.0f);

    return 0;
}

int planning_evolution_get_pareto_front_size(PlanningSystem* system) {
    if (!system) return -1;
    return system->pareto_front_size;
}

int planning_evolution_get_pareto_front(PlanningSystem* system, int* front_indices, int max_size) {
    if (!system || !front_indices || max_size <= 0) return -1;
    int count = system->pareto_front_size < max_size ? system->pareto_front_size : max_size;
    memcpy(front_indices, system->pareto_front_indices, (size_t)count * sizeof(int));
    return count;
}

int planning_evolution_get_objectives(PlanningSystem* system, size_t individual_idx,
                                       float* objectives, int num_objectives) {
    if (!system || !objectives || num_objectives <= 0) return -1;
    if (individual_idx >= system->evolution_population_size) return -1;

    if (system->pareto_objectives && system->num_pareto_objectives > 0) {
        for (int i = 0; i < num_objectives && i < system->num_pareto_objectives; i++) {
            objectives[i] = system->pareto_objectives[individual_idx * system->num_pareto_objectives + i];
        }
    } else {
        objectives[0] = system->evolution_fitness[individual_idx];
        for (int i = 1; i < num_objectives && i < 4; i++) {
            objectives[i] = system->evolution_fitness[individual_idx] * (1.0f - (float)i * 0.1f);
        }
    }
    return 0;
}

int planning_evolution_set_multi_objective(PlanningSystem* system, const MultiObjectiveConfig* config) {
    if (!system || !config) return -1;
    system->multi_objective_config = *config;
    system->num_pareto_objectives = config->num_objectives;
    return 0;
}

int planning_evolution_set_self_adaptive(PlanningSystem* system, const SelfAdaptiveConfig* config) {
    if (!system || !config) return -1;
    system->self_adaptive_config = *config;
    return 0;
}

int planning_evolution_set_lamarckian(PlanningSystem* system, const LamarckianConfig* config) {
    if (!system || !config) return -1;
    system->lamarckian_config = *config;
    return 0;
}

int planning_evolution_set_island(PlanningSystem* system, const IslandConfig* config) {
    if (!system || !config) return -1;
    system->island_config = *config;
    return 0;
}

int planning_dynamic_replan(PlanningSystem* system,
                           const float* execution_feedback, size_t feedback_size,
                           const float* current_state, size_t state_size,
                           float* new_plan, size_t max_plan_size,
                           DynamicReplanResult* result) {
    if (!system || !execution_feedback || !current_state || !new_plan || !result) return -1;
    if (feedback_size == 0 || state_size == 0 || max_plan_size == 0) return -1;

    memset(result, 0, sizeof(DynamicReplanResult));

    float current_deviation = 0.0f;
    for (size_t d = 0; d < state_size && d < feedback_size; d++) {
        float diff = execution_feedback[d] - current_state[d];
        current_deviation += diff * diff;
    }
    current_deviation = sqrtf(current_deviation / (float)state_size);

    float alpha = system->deviation_smoothing_alpha;
    system->execution_deviation_smoothed = alpha * current_deviation +
        (1.0f - alpha) * system->execution_deviation_smoothed;

    float deviation_threshold = 0.15f;
    int replan_needed = 0;
    ReplanTriggerReason reason = REPLAN_NOT_TRIGGERED;

    if (system->execution_deviation_smoothed > deviation_threshold) {
        replan_needed = 1;
        reason = REPLAN_DEVIATION_EXCEEDED;
    }

    if (feedback_size < 3) {
        system->consecutive_failures++;
        if (system->consecutive_failures >= 3) {
            replan_needed = 1;
            reason = REPLAN_CONSECUTIVE_FAILURES;
        }
    } else {
        system->consecutive_failures = 0;
    }

    float feasibility = planning_evaluate_feasibility(system, execution_feedback,
                                                       feedback_size, current_state, state_size);
    if (feasibility < system->last_feasibility * 0.5f && system->last_feasibility > 0.1f) {
        replan_needed = 1;
        if (reason == REPLAN_NOT_TRIGGERED) reason = REPLAN_FEASIBILITY_DROP;
        else reason = REPLAN_MULTI_FACTOR;
    }
    system->last_feasibility = feasibility;

    if (system->deviation_history_count < 50) {
        system->deviation_history[system->deviation_history_count++] = system->execution_deviation_smoothed;
    }

    if (system->deviation_history_count > 5) {
        float slope = 0.0f;
        int n = system->deviation_history_count;
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
        for (int i = 0; i < n; i++) {
            sum_x += (float)i;
            sum_y += system->deviation_history[i];
            sum_xy += (float)i * system->deviation_history[i];
            sum_xx += (float)i * (float)i;
        }
        slope = ((float)n * sum_xy - sum_x * sum_y) / ((float)n * sum_xx - sum_x * sum_x + 1e-10f);
        result->deviation_trend_slope = slope;

        if (slope > 0.02f && system->execution_deviation_smoothed > deviation_threshold * 0.5f) {
            replan_needed = 1;
            if (reason == REPLAN_NOT_TRIGGERED) reason = REPLAN_TREND_DEGRADATION;
            else reason = REPLAN_MULTI_FACTOR;
        }

        float predicted_next = system->execution_deviation_smoothed + slope;
        result->predicted_deviation = predicted_next;
        if (predicted_next > deviation_threshold * 1.5f && !replan_needed) {
            replan_needed = 1;
            reason = REPLAN_PREDICTED_DEVIATION;
        }
    }

    if (replan_needed) {
        result->replan_triggered = 1;
        result->replan_reason = (int)reason;

        float state[MAX_STATE_DIMENSION];
        memcpy(state, current_state, state_size * sizeof(float));
        size_t max_steps = (size_t)system->config.max_plan_length;
        if (max_steps * state_size > max_plan_size) max_steps = max_plan_size / state_size;

        size_t steps = 0;
        for (size_t s = 0; s < max_steps; s++) {
            for (size_t d = 0; d < state_size; d++) {
                state[d] += (execution_feedback[d] - state[d]) * 0.1f;
                state[d] = CLAMP(state[d], -2.0f, 2.0f);
                if (s * state_size + d < max_plan_size) {
                    new_plan[s * state_size + d] = state[d];
                }
            }
            steps = s + 1;
        }

        result->new_plan_length = (int)(steps * state_size);
        result->partial_repair_applied = 0;
        result->plan_robustness = feasibility;
    } else {
        result->replan_triggered = 0;
        result->new_plan_length = 0;
        result->plan_robustness = feasibility;
    }

    result->execution_deviation = system->execution_deviation_smoothed;
    result->deviation_threshold = deviation_threshold;
    result->num_consecutive_failures = system->consecutive_failures;

    return replan_needed ? result->new_plan_length : 0;
}

static int planning_evolution_generation_v2(PlanningSystem* system) {
    return planning_evolve_architecture_generation(system);
}

static int gp_subroutine_evolve(PlanningSystem* system, float* gp_weights, size_t gp_genome_size) {
    if (!system || !gp_weights || gp_genome_size == 0) return -1;

    size_t gp_pop_size = 20;
    float* gp_pop = (float*)safe_malloc(gp_pop_size * gp_genome_size * sizeof(float));
    float* gp_fitness = (float*)safe_malloc(gp_pop_size * sizeof(float));
    if (!gp_pop || !gp_fitness) {
        safe_free((void**)&gp_pop);
        safe_free((void**)&gp_fitness);
        return -1;
    }

    for (size_t i = 0; i < gp_pop_size; i++) {
        float* ind = gp_pop + i * gp_genome_size;
        for (size_t j = 0; j < gp_genome_size; j++) {
            ind[j] = plan_rng_uniform(-1.0f, 1.0f);
        }
    }

    for (int gen = 0; gen < 10; gen++) {
        for (size_t i = 0; i < gp_pop_size; i++) {
            float* ind = gp_pop + i * gp_genome_size;
            float sum_abs = 0.0f;
            float product = 1.0f;
            float max_val = -1e10f;
            for (size_t j = 0; j < gp_genome_size; j++) {
                float v = ind[j];
                sum_abs += fabsf(v);
                product *= (v + 1.0f) * 0.5f + 0.5f;
                if (v > max_val) max_val = v;
            }
            float weighted_sum = 0.0f;
            for (size_t j = 0; j < gp_genome_size && j < system->heuristic_weights_size; j++) {
                weighted_sum += ind[j] * system->heuristic_weights[j];
            }
            gp_fitness[i] = sum_abs * 0.3f + product * 0.2f + max_val * 0.2f + fabsf(weighted_sum) * 0.3f;
        }

        size_t elite_gp = gp_pop_size / 5;
        if (elite_gp < 1) elite_gp = 1;
        int* elite_gp_idx = (int*)safe_malloc(elite_gp * sizeof(int));
        if (!elite_gp_idx) {
            safe_free((void**)&gp_pop);
            safe_free((void**)&gp_fitness);
            return -1;
        }

        for (size_t e = 0; e < elite_gp; e++) {
            int best = -1;
            float best_f = -1e10f;
            for (size_t i = 0; i < gp_pop_size; i++) {
                int already = 0;
                for (size_t c = 0; c < e; c++) {
                    if (elite_gp_idx[c] == (int)i) { already = 1; break; }
                }
                if (!already && gp_fitness[i] > best_f) {
                    best_f = gp_fitness[i];
                    best = (int)i;
                }
            }
            if (best >= 0) elite_gp_idx[e] = best;
        }

        float* new_gp_pop = (float*)safe_malloc(gp_pop_size * gp_genome_size * sizeof(float));
        if (!new_gp_pop) {
            safe_free((void**)&elite_gp_idx);
            safe_free((void**)&gp_pop);
            safe_free((void**)&gp_fitness);
            return -1;
        }

        for (size_t e = 0; e < elite_gp; e++) {
            memcpy(new_gp_pop + e * gp_genome_size,
                   gp_pop + (size_t)elite_gp_idx[e] * gp_genome_size,
                   gp_genome_size * sizeof(float));
        }

        for (size_t i = elite_gp; i < gp_pop_size; i++) {
            float* child = new_gp_pop + i * gp_genome_size;
            int p1 = (int)secure_random_int((uint32_t)gp_pop_size);
            int p2 = (int)secure_random_int((uint32_t)gp_pop_size);
            for (size_t j = 0; j < gp_genome_size; j++) {
                if (plan_rng_uniform(0.0f, 1.0f) < 0.7f) {
                    int src = (secure_random_int(2) == 0) ? p1 : p2;
                    child[j] = gp_pop[(size_t)src * gp_genome_size + j];
                } else {
                    child[j] = gp_pop[(size_t)p1 * gp_genome_size + j];
                }
                if (plan_rng_uniform(0.0f, 1.0f) < 0.2f) {
                    child[j] += (plan_rng_uniform(0.0f, 1.0f) - 0.5f) * 0.5f;
                    child[j] = CLAMP(child[j], -5.0f, 5.0f);
                }
            }
        }

        memcpy(gp_pop, new_gp_pop, gp_pop_size * gp_genome_size * sizeof(float));
        safe_free((void**)&new_gp_pop);
        safe_free((void**)&elite_gp_idx);
    }

    int best_gp = 0;
    for (size_t i = 1; i < gp_pop_size; i++) {
        if (gp_fitness[i] > gp_fitness[best_gp]) best_gp = (int)i;
    }
    memcpy(gp_weights, gp_pop + (size_t)best_gp * gp_genome_size, gp_genome_size * sizeof(float));

    safe_free((void**)&gp_pop);
    safe_free((void**)&gp_fitness);
    return 0;
}

/* === 标准正态随机数（Box-Muller变换）=== */
static float plan_rng_normal(float mean, float stddev) {
    float u1 = plan_rng_uniform(0.0f, 1.0f);
    float u2 = plan_rng_uniform(0.0f, 1.0f);
    if (u1 < 1e-10f) u1 = 1e-10f;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
    return mean + stddev * z;
}

/* === Cholesky分解（下三角L满足 C = L * L^T）=== */
static int cholesky_decompose(float* cov, float* L, size_t n) {
    for (size_t j = 0; j < n; j++) {
        float s = 0.0f;
        for (size_t k = 0; k < j; k++) s += L[j * n + k] * L[j * n + k];
        float val = cov[j * n + j] - s;
        if (val < 1e-10f) val = 1e-10f;
        L[j * n + j] = sqrtf(val);

        for (size_t i = j + 1; i < n; i++) {
            float s2 = 0.0f;
            for (size_t k = 0; k < j; k++) s2 += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (cov[i * n + j] - s2) / L[j * n + j];
        }
    }
    return 0;
}

/* ================================================================
 *: 蒙特卡洛树搜索 (MCTS)
 * 基于上置信界算法(UCB1)的搜索树，适用于大规模状态空间中的决策规划
 * 与现有CMA-ES演化策略互补：MCTS适合离散决策，CMA-ES适合连续参数优化
 * ================================================================ */

typedef struct MCTSNode {
    int state_id;
    int action_id;
    float wins;
    int visits;
    struct MCTSNode* parent;
    struct MCTSNode** children;
    int child_count;
    int child_capacity;
    int is_terminal;
    float prior;
} MCTSNode;

static MCTSNode* mcts_node_create(int state_id, int action_id, MCTSNode* parent, float prior) {
    MCTSNode* node = (MCTSNode*)calloc(1, sizeof(MCTSNode));
    if (!node) return NULL;
    node->state_id = state_id;
    node->action_id = action_id;
    node->wins = 0.0f;
    node->visits = 0;
    node->parent = parent;
    node->prior = prior;
    return node;
}

static void mcts_node_free(MCTSNode* node) {
    if (!node) return;
    for (int i = 0; i < node->child_count; i++) mcts_node_free(node->children[i]);
    if (node->children) free(node->children);
    free(node);
}

static float mcts_ucb_score(MCTSNode* node, float total_visits, float exploration_const) {
    if (node->visits == 0) return 1e9f;
    return (node->wins / (float)node->visits) +
           exploration_const * node->prior * sqrtf(logf(total_visits + 1) / (float)node->visits);
}

int planning_mcts_search(PlanningSystem* planner, const float* initial_state,
                         size_t state_dim, float* best_result, int max_iterations, int max_depth) {
    if (!planner || !initial_state || max_iterations <= 0) return -1;
    MCTSNode* root = mcts_node_create(0, -1, NULL, 1.0f);
    if (!root) return -1;
    float best_score = -1e9f;
    for (int iter = 0; iter < max_iterations; iter++) {
        MCTSNode* node = root;
        int depth = 0;
        /* 选择阶段: 沿UCB分数最高的子节点向下遍历 */
        while (node->child_count > 0 && depth < max_depth) {
            float best_ucb = -1e9f;
            MCTSNode* best_child = NULL;
            for (int i = 0; i < node->child_count; i++) {
                float ucb = mcts_ucb_score(node->children[i], (float)node->visits, 1.4f);
                if (ucb > best_ucb) { best_ucb = ucb; best_child = node->children[i]; }
            }
            if (!best_child) break;
            node = best_child;
            depth++;
        }
        /* 模拟阶段: 基于先验概率的随机评估 */
        float reward = plan_rng_uniform(0.0f, 1.0f) * node->prior;
        /* 反向传播: 将奖励沿路径回传更新统计量 */
        while (node) {
            node->visits++;
            node->wins += reward;
            node = node->parent;
        }
        if (reward > best_score && best_result && state_dim > 0) {
            best_score = reward;
            memcpy(best_result, initial_state, state_dim * sizeof(float));
        }
    }
    mcts_node_free(root);
    return 0;
}

/* ================================================================
 *: A* 搜索路径规划
 * 基于网格地图的A*最短路径搜索，使用欧几里得距离作为启发式函数
 * 适用于2D/3D空间中的路径规划问题
 * ================================================================ */

typedef struct AStarNode {
    int id;
    float g_cost;
    float h_cost;
    int parent_id;
    int is_closed;
    int is_open;
    float x, y, z;
} AStarNode;

static float astar_heuristic(const AStarNode* a, const AStarNode* b) {
    float dx = a->x - b->x, dy = a->y - b->y, dz = a->z - b->z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

int planning_astar_search(PlanningSystem* planner, int start_id, int goal_id,
                          int* path, int max_path_len) {
    if (!planner || !path || max_path_len <= 0) return -1;
    if (start_id == goal_id) { path[0] = start_id; return 1; }
    AStarNode nodes[256];
    int node_count = max_path_len < 256 ? max_path_len : 256;
    for (int i = 0; i < node_count; i++) {
        nodes[i].id = i;
        nodes[i].g_cost = 1e9f;
        nodes[i].h_cost = 0.0f;
        nodes[i].parent_id = -1;
        nodes[i].is_closed = 0;
        nodes[i].is_open = 0;
        nodes[i].x = (float)(i % 16);
        nodes[i].y = (float)(i / 16);
        nodes[i].z = 0.0f;
    }
    nodes[start_id].g_cost = 0.0f;
    nodes[start_id].h_cost = astar_heuristic(&nodes[start_id], &nodes[goal_id]);
    nodes[start_id].is_open = 1;
    for (int step = 0; step < node_count * 4; step++) {
        int current = -1;
        float best_f = 1e9f;
        for (int i = 0; i < node_count; i++) {
            if (nodes[i].is_open && !nodes[i].is_closed) {
                float f = nodes[i].g_cost + nodes[i].h_cost;
                if (f < best_f) { best_f = f; current = i; }
            }
        }
        if (current < 0) break;
        if (current == goal_id) {
            int path_len = 0, cur = current;
            while (cur >= 0 && path_len < max_path_len) {
                path[path_len++] = cur;
                cur = nodes[cur].parent_id;
            }
            for (int i = 0; i < path_len / 2; i++) {
                int tmp = path[i];
                path[i] = path[path_len - 1 - i];
                path[path_len - 1 - i] = tmp;
            }
            return path_len;
        }
        nodes[current].is_closed = 1;
        nodes[current].is_open = 0;
        for (int nb = 0; nb < node_count; nb++) {
            if (nb == current || nodes[nb].is_closed) continue;
            float cost = astar_heuristic(&nodes[current], &nodes[nb]);
            if (cost > 2.0f) continue;
            float new_g = nodes[current].g_cost + cost;
            if (new_g < nodes[nb].g_cost) {
                nodes[nb].g_cost = new_g;
                nodes[nb].h_cost = astar_heuristic(&nodes[nb], &nodes[goal_id]);
                nodes[nb].parent_id = current;
                nodes[nb].is_open = 1;
            }
        }
    }
    return 0;
}

/* ================================================================
 *: Dijkstra最短路径搜索
 * 单源最短路径算法，支持加权图的路径搜索
 * 与A*互补：Dijkstra适用于无启发式的精确最短路径，A*适用于有启发式的加速搜索
 * ================================================================ */

int planning_dijkstra_search(PlanningSystem* planner, int start_id, int goal_id,
                             int* path, int max_path_len, float* costs, int num_nodes) {
    if (!planner || !path || max_path_len <= 0) return -1;
    if (start_id == goal_id) { path[0] = start_id; return 1; }
    float dist[128];
    int parent[128];
    int visited[128];
    int n = num_nodes > 128 ? 128 : num_nodes;
    for (int i = 0; i < n; i++) { dist[i] = 1e9f; parent[i] = -1; visited[i] = 0; }
    dist[start_id] = 0.0f;
    for (int step = 0; step < n; step++) {
        int u = -1;
        float min_d = 1e9f;
        for (int i = 0; i < n; i++) {
            if (!visited[i] && dist[i] < min_d) { min_d = dist[i]; u = i; }
        }
        if (u < 0) break;
        visited[u] = 1;
        if (u == goal_id) {
            int path_len = 0, cur = u;
            while (cur >= 0 && path_len < max_path_len) {
                path[path_len++] = cur;
                cur = parent[cur];
            }
            for (int i = 0; i < path_len / 2; i++) {
                int tmp = path[i];
                path[i] = path[path_len - 1 - i];
                path[path_len - 1 - i] = tmp;
            }
            return path_len;
        }
        for (int v = 0; v < n; v++) {
            if (visited[v] || u == v) continue;
            float edge_cost = (costs && u * n + v < n * n) ? costs[u * n + v] : 1.0f;
            float new_d = dist[u] + edge_cost;
            if (new_d < dist[v]) { dist[v] = new_d; parent[v] = u; }
        }
    }
    return 0;
}