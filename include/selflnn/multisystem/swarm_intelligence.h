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
 * 兼容性类型
 * ================================================================ */

/** @brief Swarm 句柄（实现层映射为 robot 模块的 SwarmController） */
typedef struct Swarm Swarm;

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
} SwarmConfig;

#define SWARM_ALGORITHM_PSO                4
#define SWARM_CONVERGENCE_MAX_ITERATIONS   0
#define SWARM_CONVERGENCE_THRESHOLD        1
#define SWARM_CONVERGENCE_STAGNATION       2
#define SWARM_TOPOLOGY_RING                2
#define SWARM_TOPOLOGY_STAR                0
#define SWARM_TOPOLOGY_MESH                1

/** @brief MSSwarmState —— 避免与 robot/swarm_control.h 的 SwarmState 冲突 */
typedef struct {
    float best_position[64];
    float best_fitness;
    int iteration;
    int converged;
} MSSwarmState;

/** @brief SwarmResult —— 优化结果 */
typedef struct {
    float* positions;
    int num_positions;
    float fitness;
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
