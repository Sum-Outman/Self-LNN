/**
 * @file swarm_intelligence.h
 * @brief 群体智能系统
 * 
 * 实现多种群体智能算法，用于多机器人协调、优化和决策：
 * 1. 粒子群优化（Particle Swarm Optimization, PSO）
 * 2. 蚁群算法（Ant Colony Optimization, ACO）
 * 3. 蜂群算法（Artificial Bee Colony, ABC）
 * 4. 鱼群算法（Fish School Search, FSS）
 * 5. 蝙蝠算法（Bat Algorithm）
 * 6. 萤火虫算法（Firefly Algorithm）
 * 7. 布谷鸟搜索（Cuckoo Search）
 * 8. 灰狼优化器（Grey Wolf Optimizer, GWO）
 * 9. 鲸鱼优化算法（Whale Optimization Algorithm, WOA）
 * 10. 多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）
 * 
 *  ，提供完整的群体智能算法实现。
 */

#ifndef SELFLNN_SWARM_INTELLIGENCE_H
#define SELFLNN_SWARM_INTELLIGENCE_H

#include "selflnn/core/errors.h"
#include "selflnn/core/tensor.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 常量定义 ========== */

/** @brief 最大群体大小 */
#define SWARM_MAX_SIZE 1000

/** @brief 最大维度数 */
#define SWARM_MAX_DIMENSIONS 100

/** @brief 最大迭代次数 */
#define SWARM_MAX_ITERATIONS 10000

/** @brief 算法类型枚举 */
typedef enum {
    SWARM_ALGORITHM_PSO = 0,           /**< 粒子群优化 */
    SWARM_ALGORITHM_ACO = 1,           /**< 蚁群算法 */
    SWARM_ALGORITHM_ABC = 2,           /**< 人工蜂群算法 */
    SWARM_ALGORITHM_FSS = 3,           /**< 鱼群算法 */
    SWARM_ALGORITHM_BAT = 4,           /**< 蝙蝠算法 */
    SWARM_ALGORITHM_FIREFLY = 5,       /**< 萤火虫算法 */
    SWARM_ALGORITHM_CUCKOO = 6,        /**< 布谷鸟搜索 */
    SWARM_ALGORITHM_GWO = 7,           /**< 灰狼优化器 */
    SWARM_ALGORITHM_WOA = 8,           /**< 鲸鱼优化算法 */
    SWARM_ALGORITHM_MARL = 9,          /**< 多智能体强化学习 */
    SWARM_ALGORITHM_HYBRID = 10        /**< 混合算法 */
} SwarmAlgorithmType;

/** @brief 收敛条件枚举 */
typedef enum {
    SWARM_CONVERGENCE_MAX_ITERATIONS = 0,  /**< 最大迭代次数 */
    SWARM_CONVERGENCE_FITNESS_THRESHOLD = 1, /**< 适应度阈值 */
    SWARM_CONVERGENCE_POSITION_CHANGE = 2, /**< 位置变化阈值 */
    SWARM_CONVERGENCE_VELOCITY_CHANGE = 3, /**< 速度变化阈值 */
    SWARM_CONVERGENCE_STAGNATION = 4       /**< 停滞迭代次数 */
} SwarmConvergenceCondition;

/** @brief 通信拓扑枚举 */
typedef enum {
    SWARM_TOPOLOGY_GLOBAL = 0,         /**< 全局拓扑（全连接） */
    SWARM_TOPOLOGY_RING = 1,           /**< 环状拓扑 */
    SWARM_TOPOLOGY_STAR = 2,           /**< 星形拓扑 */
    SWARM_TOPOLOGY_MESH = 3,           /**< 网状拓扑 */
    SWARM_TOPOLOGY_RANDOM = 4,         /**< 随机拓扑 */
    SWARM_TOPOLOGY_HIERARCHICAL = 5    /**< 分层拓扑 */
} SwarmTopologyType;

/* ========== 数据结构定义 ========== */

/** @brief 粒子/个体结构 */
typedef struct {
    float* position;                   /**< 当前位置（维度数组） */
    float* velocity;                   /**< 当前速度（维度数组） */
    float* best_position;              /**< 历史最佳位置 */
    float best_fitness;                /**< 历史最佳适应度 */
    float current_fitness;             /**< 当前适应度 */
    int id;                            /**< 个体ID */
    int is_active;                     /**< 是否活跃 */
    void* custom_data;                 /**< 算法特定数据 */
} SwarmIndividual;

/** @brief 群体结构 */
typedef struct Swarm Swarm;

/** @brief 适应度函数类型 */
typedef float (*SwarmFitnessFunction)(const float* position, int dimensions, void* user_data);

/** @brief 约束函数类型 */
typedef void (*SwarmConstraintFunction)(float* position, int dimensions, void* user_data);

/** @brief 回调函数类型 */
typedef void (*SwarmCallbackFunction)(const Swarm* swarm, int iteration, void* user_data);

/** @brief 群体配置结构 */
typedef struct {
    SwarmAlgorithmType algorithm_type; /**< 算法类型 */
    int swarm_size;                    /**< 群体大小 */
    int dimensions;                    /**< 问题维度 */
    float* lower_bounds;               /**< 下界数组 */
    float* upper_bounds;               /**< 上界数组 */
    
    /* 收敛条件 */
    SwarmConvergenceCondition convergence_condition; /**< 收敛条件类型 */
    int max_iterations;               /**< 最大迭代次数 */
    float fitness_threshold;          /**< 适应度阈值 */
    float position_change_threshold;  /**< 位置变化阈值 */
    float velocity_change_threshold;  /**< 速度变化阈值 */
    int stagnation_iterations;        /**< 停滞迭代次数 */
    
    /* 通信拓扑 */
    SwarmTopologyType topology_type;  /**< 拓扑类型 */
    int neighborhood_size;            /**< 邻居大小（用于局部拓扑） */
    
    /* 算法特定参数 */
    float inertia_weight;             /**< 惯性权重（PSO） */
    float cognitive_weight;           /**< 认知权重（PSO） */
    float social_weight;              /**< 社会权重（PSO） */
    float pheromone_evaporation;      /**< 信息素蒸发率（ACO） */
    float exploration_factor;         /**< 探索因子 */
    float exploitation_factor;        /**< 开发因子 */
    
    /* 回调函数 */
    SwarmFitnessFunction fitness_func; /**< 适应度函数 */
    SwarmConstraintFunction constraint_func; /**< 约束函数 */
    SwarmCallbackFunction iteration_callback; /**< 迭代回调 */
    SwarmCallbackFunction convergence_callback; /**< 收敛回调 */
    void* user_data;                   /**< 用户数据 */
    
    /* 并行计算 */
    int enable_parallel_evaluation;   /**< 是否启用并行评估 */
    int num_threads;                  /**< 线程数 */
    
    /* 日志和调试 */
    int enable_logging;               /**< 是否启用日志 */
    int log_frequency;                /**< 日志频率（迭代次数） */
} SwarmConfig;

/** @brief 群体状态结构 */
typedef struct {
    int iteration;                     /**< 当前迭代次数 */
    float best_fitness;                /**< 全局最佳适应度 */
    float* best_position;              /**< 全局最佳位置 */
    float average_fitness;             /**< 平均适应度 */
    float fitness_std_dev;             /**< 适应度标准差 */
    float position_diversity;          /**< 位置多样性 */
    float velocity_diversity;          /**< 速度多样性 */
    int is_converged;                  /**< 是否已收敛 */
    int convergence_reason;            /**< 收敛原因 */
    float computation_time_ms;         /**< 计算时间（毫秒） */
} SwarmState;

/** @brief 群体结果结构 */
typedef struct {
    float best_fitness;                /**< 最佳适应度 */
    float* best_position;              /**< 最佳位置 */
    int iterations;                    /**< 总迭代次数 */
    float total_time_ms;               /**< 总时间（毫秒） */
    int convergence_reason;            /**< 收敛原因 */
    SwarmState final_state;            /**< 最终状态 */
    char error_message[256];           /**< 错误信息 */
} SwarmResult;

/* ========== 函数声明 ========== */

/**
 * @brief 创建群体智能系统
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create(const SwarmConfig* config);

/**
 * @brief 释放群体智能系统
 * 
 * @param swarm 群体指针
 */
void swarm_free(Swarm* swarm);

/**
 * @brief 初始化群体
 * 
 * @param swarm 群体指针
 * @param initial_positions 初始位置数组（可选）
 * @return int 成功返回0，失败返回-1
 */
int swarm_initialize(Swarm* swarm, const float* initial_positions);

/**
 * @brief 执行群体智能优化
 * 
 * @param swarm 群体指针
 * @param result 结果指针
 * @return int 成功返回0，失败返回-1
 */
int swarm_optimize(Swarm* swarm, SwarmResult* result);

/**
 * @brief 执行单次迭代
 * 
 * @param swarm 群体指针
 * @return int 成功返回0，失败返回-1
 */
int swarm_iterate(Swarm* swarm);

/**
 * @brief 获取群体状态
 * 
 * @param swarm 群体指针
 * @param state 状态指针
 * @return int 成功返回0，失败返回-1
 */
int swarm_get_state(const Swarm* swarm, SwarmState* state);

/**
 * @brief 获取个体信息
 * 
 * @param swarm 群体指针
 * @param individual_id 个体ID
 * @param individual 个体指针
 * @return int 成功返回0，失败返回-1
 */
int swarm_get_individual(const Swarm* swarm, int individual_id, SwarmIndividual* individual);

/**
 * @brief 更新个体位置
 * 
 * @param swarm 群体指针
 * @param individual_id 个体ID
 * @param new_position 新位置数组
 * @return int 成功返回0，失败返回-1
 */
int swarm_update_individual_position(Swarm* swarm, int individual_id, const float* new_position);

/**
 * @brief 获取最佳解
 * 
 * @param swarm 群体指针
 * @param position 位置数组（输出）
 * @param fitness 适应度（输出）
 * @return int 成功返回0，失败返回-1
 */
int swarm_get_best_solution(const Swarm* swarm, float* position, float* fitness);

/**
 * @brief 重置群体
 * 
 * @param swarm 群体指针
 * @return int 成功返回0，失败返回-1
 */
int swarm_reset(Swarm* swarm);

/**
 * @brief 保存群体状态到文件
 * 
 * @param swarm 群体指针
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int swarm_save_state(const Swarm* swarm, const char* filename);

/**
 * @brief 从文件加载群体状态
 * 
 * @param swarm 群体指针
 * @param filename 文件名
 * @return int 成功返回0，失败返回-1
 */
int swarm_load_state(Swarm* swarm, const char* filename);

/**
 * @brief 设置算法参数
 * 
 * @param swarm 群体指针
 * @param param_name 参数名
 * @param value 参数值
 * @return int 成功返回0，失败返回-1
 */
int swarm_set_parameter(Swarm* swarm, const char* param_name, float value);

/**
 * @brief 获取算法参数
 * 
 * @param swarm 群体指针
 * @param param_name 参数名
 * @param value 参数值（输出）
 * @return int 成功返回0，失败返回-1
 */
int swarm_get_parameter(const Swarm* swarm, const char* param_name, float* value);

/**
 * @brief 获取算法类型名称
 * 
 * @param algorithm_type 算法类型
 * @return const char* 算法名称
 */
const char* swarm_algorithm_name(SwarmAlgorithmType algorithm_type);

/**
 * @brief 获取拓扑类型名称
 * 
 * @param topology_type 拓扑类型
 * @return const char* 拓扑名称
 */
const char* swarm_topology_name(SwarmTopologyType topology_type);

/**
 * @brief 获取收敛条件名称
 * 
 * @param condition 收敛条件
 * @return const char* 条件名称
 */
const char* swarm_convergence_condition_name(SwarmConvergenceCondition condition);

/**
 * @brief 初始化群体结果
 * 
 * @param result 结果指针
 */
void swarm_result_init(SwarmResult* result);

/**
 * @brief 释放群体结果内存
 * 
 * @param result 结果指针
 */
void swarm_result_free(SwarmResult* result);

/**
 * @brief 初始化群体状态
 * 
 * @param state 状态指针
 */
void swarm_state_init(SwarmState* state);

/**
 * @brief 释放群体状态内存
 * 
 * @param state 状态指针
 */
void swarm_state_free(SwarmState* state);

/* ========== 算法特定函数 ========== */

/**
 * @brief 创建粒子群优化器
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_pso(const SwarmConfig* config);

/**
 * @brief 创建蚁群算法
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_aco(const SwarmConfig* config);

/**
 * @brief 创建人工蜂群算法
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_abc(const SwarmConfig* config);

/**
 * @brief 创建鱼群算法
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_fss(const SwarmConfig* config);

/**
 * @brief 创建蝙蝠算法
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_bat(const SwarmConfig* config);

/**
 * @brief 创建萤火虫算法
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_firefly(const SwarmConfig* config);

/**
 * @brief 创建布谷鸟搜索
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_cuckoo(const SwarmConfig* config);

/**
 * @brief 创建灰狼优化器
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_gwo(const SwarmConfig* config);

/**
 * @brief 创建鲸鱼优化算法
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_woa(const SwarmConfig* config);

/**
 * @brief 创建多智能体强化学习
 * 
 * @param config 配置参数
 * @return Swarm* 群体指针，失败返回NULL
 */
Swarm* swarm_create_marl(const SwarmConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SWARM_INTELLIGENCE_H */