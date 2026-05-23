/**
 * @file swarm_intelligence.h
 * @brief 群体智能优化器 —— 兼容性桥接头文件
 *
 * ZSFWS-修复: 原 multisystem/swarm_intelligence.c 已合并至 robot/swarm_control.c。
 * 本头文件提供 multisystem_control 模块所需的 Swarm/SwarmConfig/MSSwarmState 等
 * 类型定义。实现由 robot/swarm_control.c 中的桥接函数提供。
 *
 * 注意: 为避免与 robot/swarm_control.h 的 SwarmState 名称冲突，
 * 本模块使用 MSSwarmState (MultiSystem Swarm State)。
 */

#ifndef SELFLNN_MULTISYSTEM_SWARM_INTELLIGENCE_H
#define SELFLNN_MULTISYSTEM_SWARM_INTELLIGENCE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 枚举类型
 * ================================================================ */

typedef enum {
    SWARM_ALGORITHM_ACO = 0,
    SWARM_ALGORITHM_ABC = 1,
    SWARM_ALGORITHM_PSO = 4,
    SWARM_ALGORITHM_FSS = 5,
    SWARM_ALGORITHM_BAT = 6,
    SWARM_ALGORITHM_FIREFLY = 7,
    SWARM_ALGORITHM_CUCKOO = 8,
    SWARM_ALGORITHM_GWO = 9,
    SWARM_ALGORITHM_WOA = 10,
    SWARM_ALGORITHM_MARL = 11,
    SWARM_ALGORITHM_HYBRID = 12
} SwarmAlgorithmType;

typedef enum {
    SWARM_TOPOLOGY_STAR = 0,
    SWARM_TOPOLOGY_MESH = 1,
    SWARM_TOPOLOGY_RING = 2,
    SWARM_TOPOLOGY_GLOBAL = 3,
    SWARM_TOPOLOGY_RANDOM = 4,
    SWARM_TOPOLOGY_HIERARCHICAL = 5
} SwarmTopologyType;

typedef enum {
    SWARM_CONVERGENCE_MAX_ITERATIONS = 0,
    SWARM_CONVERGENCE_THRESHOLD = 1,
    SWARM_CONVERGENCE_STAGNATION = 2,
    SWARM_CONVERGENCE_FITNESS_THRESHOLD = 3,
    SWARM_CONVERGENCE_POSITION_CHANGE = 4,
    SWARM_CONVERGENCE_VELOCITY_CHANGE = 5
} SwarmConvergenceCondition;

/* ================================================================
 * 常量宏
 * ================================================================ */

#define SWARM_MAX_SIZE          10000
#define SWARM_MAX_DIMENSIONS    1000
#define SWARM_MAX_ITERATIONS    1000000

/* ================================================================
 * 兼容性类型
 * ================================================================ */

/** @brief Swarm 句柄（实现层映射为 robot 模块的 SwarmController） */
typedef struct Swarm Swarm;

/** @brief 群体个体结构体 */
typedef struct {
    int id;
    int is_active;
    float* position;
    float* velocity;
    float* best_position;
    float fitness;
    float best_fitness;
    float current_fitness;
    void* custom_data;
} SwarmIndividual;

/** @brief SwarmConfig —— multisystem_control 所需的群体配置 */
typedef struct {
    int algorithm_type;
    int swarm_size;
    int dimensions;
    int convergence_condition;
    int max_iterations;
    int topology_type;
    int neighborhood_size;
    float inertia_weight;
    float cognitive_weight;
    float social_weight;
    float exploration_factor;
    float exploitation_factor;
    int enable_logging;
    int log_frequency;
    int enable_parallel_evaluation;
    int num_threads;
    float (*fitness_func)(const float*, int, void*);
    float* lower_bounds;
    float* upper_bounds;
    void* user_data;
    float fitness_threshold;
    float pheromone_evaporation;
    void (*constraint_func)(float*, int, void*);
    void (*iteration_callback)(void*, int, void*);
    void (*convergence_callback)(void*, int, void*);
} SwarmConfig;

/** @brief MSSwarmState —— 避免与 robot/swarm_control.h 的 SwarmState 冲突 */
typedef struct {
    float* best_position;
    float best_fitness;
    int iteration;
    int converged;
    float average_fitness;
    float fitness_std_dev;
    float position_diversity;
    float velocity_diversity;
    int is_converged;
    int convergence_reason;
    float computation_time_ms;
} MSSwarmState;

/** @brief SwarmState —— 内部状态（用于 swarm_get_state） */
typedef MSSwarmState SwarmState;

/** @brief SwarmResult —— 优化结果 */
typedef struct {
    float best_fitness;
    int iterations;
    float total_time_ms;
    int convergence_reason;
    float* best_position;
    int num_positions;
    float fitness;
    float* positions;
    char error_message[256];
    MSSwarmState final_state;
} SwarmResult;

/* ================================================================
 * 兼容性函数声明（实现见 robot/swarm_control.c）
 * ================================================================ */

Swarm*  swarm_create(const SwarmConfig* config);
void    swarm_free(Swarm* swarm);
int     swarm_initialize(Swarm* swarm, const float* initial_positions);
int     swarm_optimize(Swarm* swarm, SwarmResult* result);
void    swarm_result_init(SwarmResult* result);
void    swarm_result_free(SwarmResult* result);
int     swarm_iterate(Swarm* swarm);
int     swarm_get_ms_state(Swarm* swarm, MSSwarmState* state);
int     swarm_get_best_solution(Swarm* swarm, float* position, float* fitness);
int     swarm_reset(Swarm* swarm);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MULTISYSTEM_SWARM_INTELLIGENCE_H */
