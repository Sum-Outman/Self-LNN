#ifndef SELFLNN_PARETO_OPTIMIZATION_H
#define SELFLNN_PARETO_OPTIMIZATION_H

#include "selflnn/core/evolutionary_algorithms.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PARETO_MAX_OBJECTIVES 8
#define PARETO_MAX_FRONT_SIZE 256
#define PARETO_OBJECTIVE_NAME_MAX 32

typedef struct {
    int individual_index;
    float objectives[PARETO_MAX_OBJECTIVES];
    int rank;
    float crowding_distance;
    float* genome;
    size_t genome_size;
} ParetoFrontEntry;

typedef struct {
    ParetoFrontEntry* entries;
    int entry_count;
    int alloc_count;
    int num_objectives;
    char objective_names[PARETO_MAX_OBJECTIVES][PARETO_OBJECTIVE_NAME_MAX];
    float objective_ranges[PARETO_MAX_OBJECTIVES][2];
    float hypervolume;
    float spacing_metric;
    float coverage_ratio;
} ParetoFront;

typedef enum {
    PARETO_OBJ_ACCURACY = 0,
    PARETO_OBJ_SPEED = 1,
    PARETO_OBJ_ENERGY = 2,
    PARETO_OBJ_MEMORY = 3,
    PARETO_OBJ_ROBUSTNESS = 4,
    PARETO_OBJ_GENERALIZATION = 5,
    PARETO_OBJ_COMPLEXITY = 6,
    PARETO_OBJ_LATENCY = 7
} ParetoObjectiveType;

typedef struct {
    ParetoObjectiveType objective_types[PARETO_MAX_OBJECTIVES];
    float weights[PARETO_MAX_OBJECTIVES];
    int num_objectives;
    int population_size;
    int max_generations;
    float crossover_prob;
    float mutation_prob;
    float mutation_strength;
} ParetoOptimizationConfig;

ParetoOptimizationConfig pareto_config_default(int num_objectives);

int pareto_front_extract(Population* pop, NSGA2IndividualData* obj_data,
                          ParetoFront* front, int num_objectives);

int pareto_front_compute_hypervolume(ParetoFront* front, const float* reference_point);

float pareto_front_compute_spacing(const ParetoFront* front);

int pareto_front_to_json(const ParetoFront* front, char* buffer, size_t buffer_size);

int pareto_front_select_best(ParetoFront* front, const float* preferences,
                              float* best_genome, size_t genome_size);

int pareto_multi_evolve(Population* pop, MultiObjectiveFunction obj_func,
                         const ParetoOptimizationConfig* config, void* user_data,
                         ParetoFront* front);

int pareto_evolutionary_train(float* genome, size_t genome_size,
                               ParetoOptimizationConfig* config,
                               MultiObjectiveFunction obj_func, void* user_data,
                               ParetoFront* front);

int pareto_training_results_to_json(const float* genome, size_t genome_size,
                                     const float* objectives, int num_objectives,
                                     char* buffer, size_t buffer_size);

/**
 * @brief MOEA/D 分解方法类型
 */
typedef enum {
    MOEAD_DECOMPOSITION_TEICHEBYCHEFF = 0,  /**< 切比雪夫分解 */
    MOEAD_DECOMPOSITION_WEIGHTED_SUM = 1    /**< 加权和分解 */
} MOEADDecompositionType;

/**
 * @brief MOEA/D 配置结构
 */
typedef struct {
    MOEADDecompositionType decomposition_type;  /**< 分解方法 */
    int num_objectives;                         /**< 目标数 */
    size_t population_size;                     /**< 种群大小 */
    int max_generations;                        /**< 最大代数 */
    int neighborhood_size;                      /**< 邻域大小 */
    float crossover_prob;                       /**< 交叉概率 */
    float mutation_prob;                        /**< 变异概率 */
    float mutation_strength;                    /**< 变异强度 */
    float ideal_point[PARETO_MAX_OBJECTIVES];   /**< 理想点（自动更新） */
} MOEADConfig;

/**
 * @brief 创建默认MOEA/D配置
 * 
 * @param num_objectives 目标数
 * @return MOEADConfig 默认配置
 */
MOEADConfig pareto_moead_config_default(int num_objectives);

/**
 * @brief MOEA/D多目标演化主循环
 * 
 * 基于分解的多目标进化算法，将多目标问题分解为多个标量子问题并行优化。
 * 
 * @param pop 种群指针（用于初始化和演化）
 * @param obj_func 多目标评估函数
 * @param config MOEA/D配置
 * @param user_data 用户数据
 * @param front 输出帕累托前沿
 * @return int 成功返回前沿个数，失败返回-1
 */
int pareto_moead_evolve(Population* pop, MultiObjectiveFunction obj_func,
                         const MOEADConfig* config, void* user_data,
                         ParetoFront* front);

void pareto_front_destroy(ParetoFront* front);

#ifdef __cplusplus
}
#endif

#endif
