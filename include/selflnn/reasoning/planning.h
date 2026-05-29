/**
 * @file planning.h
 * @brief 规划系统接口 — 【基础功能实现】
 *
 * P3-004 功能边界说明:
 *   ✅ 本文件: 核心规划算法（层次规划、顺序规划、并行规划、反应式规划、MCTS）
 *   ✅ 本文件: 基础任务分解、目标管理、计划执行跟踪
 *   ✅ 本文件: 规划系统生命周期管理（创建/销毁/配置/执行）
 *   ❌ 非本文件: 时间推理规划(STN)、并发多智能体规划、HTN增强、CfC液态演化规划
 *                → 请使用 planning_enhanced.h（高级增强功能，与本基础版互补，非替代）
 *   
 *   基础版 (planning.h)      → 核心规划引擎 + 基础算法
 *   增强版 (planning_enhanced.h) → 时间推理 + 并发规划 + HTN + 条件规划 + CfC ODE
 */

#ifndef SELFLNN_PLANNING_H
#define SELFLNN_PLANNING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 规划算法类型
 */
typedef enum {
    PLANNING_HIERARCHICAL = 0,   /**< 层次规划 */
    PLANNING_SEQUENTIAL = 1,     /**< 顺序规划 */
    PLANNING_PARALLEL = 2,       /**< 并行规划 */
    PLANNING_REACTIVE = 3,       /**< 反应式规划 */
    PLANNING_MCTS = 4,           /**< 蒙特卡洛树搜索规划 */
    PLANNING_ENHANCED = 5,       /**< 增强规划（时间推理STN+CfC+HTN+条件规划） */
    PLANNING_REFINED = 6         /**< 精炼规划（Landmark+FF+FFD+符号规划） */
} PlanningAlgorithm;

/**
 * @brief 规划系统配置
 */
typedef struct {
    PlanningAlgorithm algorithm; /**< 规划算法 */
    int max_plan_length;         /**< 最大规划长度 */
    float risk_tolerance;        /**< 风险容忍度 */
    float goal_tolerance;        /**< 目标容差 */
    int enable_adaptation;       /**< 是否启用自适应规划 */
} PlanningConfig;

/**
 * @brief 规划系统句柄
 */
typedef struct PlanningSystem PlanningSystem;

/**
 * @brief 创建规划系统
 * 
 * @param config 规划配置
 * @return PlanningSystem* 规划系统句柄，失败返回NULL
 */
PlanningSystem* planning_system_create(const PlanningConfig* config);

/**
 * @brief 释放规划系统
 * 
 * @param system 规划系统句柄
 */
void planning_system_free(PlanningSystem* system);

/**
 * @brief 生成规划
 * 
 * @param system 规划系统句柄
 * @param goal 目标描述
 * @param goal_size 目标描述大小
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @param plan 规划输出缓冲区
 * @param max_plan_size 规划最大大小
 * @return int 成功返回规划步骤数，失败返回-1
 */
int planning_generate(PlanningSystem* system,
                     const float* goal, size_t goal_size,
                     const float* current_state, size_t state_size,
                     float* plan, size_t max_plan_size);

/**
 * @brief 执行规划
 * 
 * @param system 规划系统句柄
 * @param plan 规划
 * @param plan_size 规划大小
 * @param execution_feedback 执行反馈输出缓冲区
 * @param max_feedback_size 反馈最大大小
 * @return int 成功返回0，失败返回-1
 */
int planning_execute(PlanningSystem* system,
                    const float* plan, size_t plan_size,
                    float* execution_feedback, size_t max_feedback_size);

/**
 * @brief 评估规划可行性
 * 
 * @param system 规划系统句柄
 * @param plan 规划
 * @param plan_size 规划大小
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @return float 可行性分数 (0-1)
 */
float planning_evaluate_feasibility(PlanningSystem* system,
                                   const float* plan, size_t plan_size,
                                   const float* current_state, size_t state_size);

/**
 * @brief 自适应调整规划
 * 
 * @param system 规划系统句柄
 * @param original_plan 原始规划
 * @param original_plan_size 原始规划大小
 * @param feedback 执行反馈
 * @param feedback_size 反馈大小
 * @param adjusted_plan 调整后规划输出缓冲区
 * @param max_adjusted_size 调整后规划最大大小
 * @return int 成功返回调整后规划步骤数，失败返回-1
 */
int planning_adapt(PlanningSystem* system,
                  const float* original_plan, size_t original_plan_size,
                  const float* feedback, size_t feedback_size,
                  float* adjusted_plan, size_t max_adjusted_size);

/**
 * @brief 多目标规划
 * 
 * @param system 规划系统句柄
 * @param goals 目标数组
 * @param goal_sizes 目标大小数组
 * @param num_goals 目标数量
 * @param weights 目标权重数组
 * @param plan 规划输出缓冲区
 * @param max_plan_size 规划最大大小
 * @return int 成功返回规划步骤数，失败返回-1
 */
int planning_multi_objective(PlanningSystem* system,
                            const float** goals, const size_t* goal_sizes,
                            size_t num_goals, const float* weights,
                            float* plan, size_t max_plan_size);

/**
 * @brief 获取规划系统配置
 * 
 * @param system 规划系统句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int planning_system_get_config(const PlanningSystem* system, PlanningConfig* config);

/**
 * @brief 设置规划系统配置
 * 
 * @param system 规划系统句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int planning_system_set_config(PlanningSystem* system, const PlanningConfig* config);

/**
 * @brief 重置规划系统
 * 
 * @param system 规划系统句柄
 */
void planning_system_reset(PlanningSystem* system);

/**
 * @brief 架构演化统计信息
 */
typedef struct {
    int current_generation;         /**< 当前代数 */
    float best_fitness;             /**< 最佳适应度 */
    float average_fitness;          /**< 平均适应度 */
    float diversity_score;          /**< 种群多样性分数 */
    size_t population_size;         /**< 种群大小 */
    size_t genome_size;             /**< 基因组大小 */
    float mutation_rate;            /**< 突变率 */
    float crossover_rate;           /**< 交叉率 */
    int cmaes_active;               /**< CMA-ES是否激活 */
} PlanningEvolutionStats;

/**
 * @brief 演化规划架构
 * 
 * 使用遗传算法（SBX交叉+多项式变异）和CMA-ES进化策略
 * 自动优化规划系统的启发式权重、风险容忍度和目标容差。
 * 
 * @param system 规划系统句柄
 * @param num_generations 演化代数
 * @param feedback 环境反馈数组
 * @param feedback_size 反馈大小
 * @return int 成功返回0，失败返回-1
 */
int planning_evolve_architecture(PlanningSystem* system, size_t num_generations,
                                 const float* feedback, size_t feedback_size);

/**
 * @brief 获取架构演化统计
 * 
 * @param system 规划系统句柄
 * @param stats 统计信息输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int planning_evolution_get_stats(PlanningSystem* system, PlanningEvolutionStats* stats);

/**
 * @brief 配置架构演化参数
 * 
 * @param system 规划系统句柄
 * @param population_size 种群大小（0表示不修改）
 * @param mutation_rate 突变率（0-1，负值表示不修改）
 * @param crossover_rate 交叉率（0-1，负值表示不修改）
 * @return int 成功返回0，失败返回-1
 */
int planning_evolution_configure(PlanningSystem* system, size_t population_size,
                                  float mutation_rate, float crossover_rate);

/* ========== 动态重规划 ========== */

/**
 * @brief 动态重规划原因枚举
 */
typedef enum {
    REPLAN_NOT_TRIGGERED = 0,           /**< 未触发 */
    REPLAN_DEVIATION_EXCEEDED = 1,      /**< 偏差超限 */
    REPLAN_CONSECUTIVE_FAILURES = 2,    /**< 连续失败 */
    REPLAN_FEASIBILITY_DROP = 3,        /**< 可行性下降 */
    REPLAN_MULTI_FACTOR = 4,            /**< 多因素综合 */
    REPLAN_PREDICTED_DEVIATION = 5,     /**< 预测偏差超限 */
    REPLAN_TREND_DEGRADATION = 6,       /**< 趋势恶化 */
    REPLAN_ANOMALY_DETECTED = 7,        /**< 异常检测触发 */
    REPLAN_CONTINGENCY_ACTIVATED = 8    /**< 备用计划激活 */
} ReplanTriggerReason;

/**
 * @brief 动态重规划结果
 */
typedef struct {
    int replan_triggered;                   /**< 是否触发了重规划 */
    int new_plan_length;                    /**< 新规划长度（未触发则为0） */
    float execution_deviation;              /**< 当前执行偏差（指数平滑后） */
    float deviation_threshold;              /**< 偏差触发阈值 */
    int num_consecutive_failures;           /**< 连续失败次数 */
    int replan_reason;                      /**< 重规划原因(见ReplanTriggerReason枚举) */
    float predicted_deviation;              /**< 预测的下一步偏差 */
    float deviation_trend_slope;            /**< 偏差趋势斜率（正=恶化,负=改善） */
    int partial_repair_applied;             /**< 是否使用了局部修复（0=完全重规划,1=局部修复） */
    int repair_start_step;                  /**< 修复开始步骤索引 */
    int repair_end_step;                    /**< 修复结束步骤索引 */
    float rollback_suggestion;              /**< 回滚建议位置（0-1,0=不推荐回滚） */
    float plan_robustness;                  /**< 规划鲁棒性评分（0-1） */
    int contingency_plan_generated;         /**< 是否生成了备用计划 */
    int contingency_plan_activated;         /**< 备用计划是否被激活 */
    float anomaly_score;                    /**< 异常分数（0-1,越高越异常） */
} DynamicReplanResult;

/**
 * @brief 动态重规划配置
 */
typedef struct {
    float deviation_threshold;                  /**< 执行偏差触发阈值 */
    float deviation_smoothing_alpha;            /**< 偏差指数平滑系数(0-1),越大越敏感 */
    int max_consecutive_failures;               /**< 最大连续失败次数 */
    float feasibility_drop_threshold;           /**< 可行性下降阈值(0-1相对值) */
    int enable_auto_monitoring;                 /**< 是否启用自动监控重规划 */
    
    /* 预测性重规划 */
    int enable_predictive_replan;               /**< 是否启用预测性重规划 */
    float prediction_horizon;                   /**< 预测视界（步数） */
    float deviation_trend_threshold;            /**< 偏差趋势斜率阈值（超过此值触发预测性重规划） */
    
    /* 局部修复 */
    int enable_partial_repair;                  /**< 是否启用局部修复 */
    float partial_repair_ratio;                 /**< 最大可修复比例（占规划总长度的比例） */
    
    /* 备用计划 */
    int enable_contingency_planning;            /**< 是否启用备用计划生成 */
    int contingency_plan_interval;              /**< 生成备用计划的间隔步数 */
    
    /* 异常检测 */
    float anomaly_detection_sensitivity;        /**< 异常检测敏感度（0-1,越高越敏感） */
    int enable_anomaly_detection;               /**< 是否启用异常检测 */
    
    /* 状态跟踪 */
    int state_history_size;                     /**< 状态历史大小用于趋势分析 */
    int trajectory_tracking_size;               /**< 轨迹跟踪缓冲区大小 */
    
    /* 自适应阈值 */
    int enable_adaptive_threshold;              /**< 是否启用了自适应阈值调节 */
    float threshold_adaptation_rate;            /**< 阈值自适应速率（0-1） */
    float min_deviation_threshold;              /**< 最小偏差阈值 */
    float max_deviation_threshold;              /**< 最大偏差阈值 */
} DynamicReplanConfig;

/**
 * @brief 执行动态重规划
 *
 * 根据执行反馈自动检测执行偏差,当偏差超过阈值、连续失败次数过多或可行性下降时
 * 触发规划调整或完全重规划。使用指数平滑法过滤噪声。
 *
 * @param system 规划系统句柄
 * @param execution_feedback 执行反馈数组
 * @param feedback_size 反馈大小
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @param new_plan 重规划输出缓冲区
 * @param max_plan_size 规划最大大小
 * @param result 重规划结果输出
 * @return int 成功返回重规划后的步骤数(未触发则返回0),失败返回-1
 */
int planning_dynamic_replan(PlanningSystem* system,
                           const float* execution_feedback, size_t feedback_size,
                           const float* current_state, size_t state_size,
                           float* new_plan, size_t max_plan_size,
                           DynamicReplanResult* result);

/* ========== 架构级演化 ========== */

#define ARCH_GENOME_MAX_LAYERS 8
#define ARCH_GENOME_MAX_ACTIVATIONS 8
#define ARCH_GENOME_NUM_TOPOLOGY_GENES 19

/**
 * @brief 拓扑基因(神经网络架构编码)
 */
typedef struct {
    int num_hidden_layers;                                      /**< 隐藏层数量 */
    int layer_sizes[ARCH_GENOME_MAX_LAYERS];                    /**< 各隐藏层神经元数 */
    int activation_types[ARCH_GENOME_MAX_ACTIVATIONS];          /**< 各层激活函数类型 */
    float connectivity_density;                                 /**< 连接密度(0-1) */
    int skip_connections;                                       /**< 是否使用跳跃连接(0/1) */
} TopologyGenes;

/**
 * @brief 架构基因组(完整编码)
 */
typedef struct {
    TopologyGenes topology;                                     /**< 拓扑基因 */
    float risk_tolerance;                                       /**< 风险容忍度 */
    float goal_tolerance;                                       /**< 目标容差 */
    float* heuristic_weights;                                   /**< 启发式权重数组 */
    size_t heuristic_weights_size;                              /**< 启发式权重大小 */
} ArchitectureGenome;

/**
 * @brief 架构演化配置
 */
typedef struct {
    int enable_topology_evolution;                              /**< 是否启用拓扑演化 */
    int min_hidden_layers;                                      /**< 最小隐藏层数 */
    int max_hidden_layers;                                      /**< 最大隐藏层数 */
    int min_layer_size;                                         /**< 最小层神经元数 */
    int max_layer_size;                                         /**< 最大层神经元数 */
    float connectivity_mutation_rate;                           /**< 连接密度变异率 */
    float layer_add_probability;                                /**< 添加层概率 */
    float layer_remove_probability;                             /**< 删除层概率 */
    int available_activation_types[ARCH_GENOME_MAX_ACTIVATIONS]; /**< 可选激活函数类型 */
    int num_activation_types;                                   /**< 可选激活函数数量 */
} ArchitectureEvolutionConfig;

/* ========== 多目标演化(Pareto优化) ========== */

/**
 * @brief Pareto前沿个体
 */
typedef struct {
    float* objectives;          /**< 多目标值数组 */
    float crowding_distance;    /**< 拥挤距离 */
    int rank;                   /**< Pareto前沿层级(0=最优) */
    int dominated_count;        /**< 被支配计数 */
} ParetoIndividual;

/**
 * @brief 多目标演化配置
 */
typedef struct {
    int enabled;                                    /**< 是否启用多目标演化 */
    int num_objectives;                             /**< 目标数量(2-4) */
    float objective_weights[4];                     /**< 各目标权重(总和=1) */
    float crowding_distance_factor;                 /**< 拥挤距离因子(默认1.0) */
} MultiObjectiveConfig;

/* ========== 自我适应参数 ========== */

/**
 * @brief 自我适应参数配置
 */
typedef struct {
    int enabled;                                    /**< 是否启用自我适应参数 */
    float initial_mutation_rate;                    /**< 初始变异率(默认0.1) */
    float initial_crossover_rate;                   /**< 初始交叉率(默认0.9) */
    float mutation_rate_range[2];                   /**< 变异率范围[0.01,0.5] */
    float crossover_rate_range[2];                  /**< 交叉率范围[0.5,0.99] */
    float learning_rate;                            /**< 自我适应学习率(默认0.1) */
} SelfAdaptiveConfig;

/* ========== Lamarckian演化 ========== */

/**
 * @brief Lamarckian演化配置
 */
typedef struct {
    int enabled;                                    /**< 是否启用Lamarckian继承 */
    float local_search_probability;                 /**< 局部搜索概率(默认0.2) */
    int local_search_steps;                         /**< 局部搜索步数(默认5) */
    float inheritance_rate;                         /**< 继承比例(默认0.5) */
} LamarckianConfig;

/* ========== 岛屿模型 ========== */

/**
 * @brief 岛屿模型配置
 */
typedef struct {
    int enabled;                                    /**< 是否启用岛屿模型 */
    int num_islands;                                /**< 岛屿数量(默认4) */
    int migration_interval;                         /**< 迁移间隔代数(默认5) */
    float migration_rate;                           /**< 迁移率(默认0.1) */
    int island_topology;                            /**< 岛屿拓扑(0=环形,1=网格) */
} IslandConfig;

/* ========== 适应度景观分析 ========== */

/**
 * @brief 适应度景观分析指标
 */
typedef struct {
    float fitness_variance;                         /**< 种群适应度方差 */
    float fitness_skewness;                         /**< 适应度偏度(正=右偏) */
    float diversity_rate;                           /**< 多样性比率(0-1,越高越多) */
    float convergence_rate;                         /**< 收敛速率(变化率) */
    float premature_convergence_score;              /**< 早熟收敛分数(0-1,越高越早熟) */
    float landscape_ruggedness;                     /**< 适应度景观粗糙度(0-1) */
} LandscapeMetrics;

/* ========== 深度演化完整配置 ========== */

/**
 * @brief 深度演化配置(完整)
 */
typedef struct {
    ArchitectureEvolutionConfig arch_config;         /**< 架构演化配置 */
    MultiObjectiveConfig multi_objective;            /**< 多目标演化配置 */
    SelfAdaptiveConfig self_adaptive;                /**< 自我适应参数配置 */
    LamarckianConfig lamarckian;                     /**< Lamarckian演化配置 */
    IslandConfig island;                             /**< 岛屿模型配置 */
} DeepEvolutionConfig;

/**
 * @brief 配置深度架构演化
 *
 * 启用后 planning_evolve_architecture 将同时优化:
 * - 拓扑结构:隐藏层数、各层大小、激活函数类型、连接密度、跳跃连接
 * - 行为参数:启发式权重、风险容忍度、目标容差
 * 拓扑基因使用拓扑感知交叉(子图交换)和结构变异(添加/删除层)操作。
 *
 * @param system 规划系统句柄
 * @param config 架构演化配置(NULL=禁用拓扑演化)
 * @return int 成功返回0,失败返回-1
 */
int planning_configure_deep_evolution(PlanningSystem* system, const ArchitectureEvolutionConfig* config);

/**
 * @brief 配置深度演化V2(完整配置)
 *
 * 在原有深度架构演化基础上增加:
 * - 多目标Pareto优化(NSGA-II风格)
 * - 自我适应变异率/交叉率
 * - Lamarckian局部搜索+继承
 * - 岛屿模型+迁移
 *
 * @param system 规划系统句柄
 * @param config 深度演化完整配置(NULL=禁用)
 * @return int 成功返回0,失败返回-1
 */
int planning_configure_deep_evolution_v2(PlanningSystem* system, const DeepEvolutionConfig* config);

/**
 * @brief 获取当前架构基因组
 *
 * @param system 规划系统句柄
 * @param genome 架构基因组输出缓冲区(heuristic_weights由调用者管理)
 * @return int 成功返回0,失败返回-1
 */
int planning_get_architecture_genome(PlanningSystem* system, ArchitectureGenome* genome);

/**
 * @brief 获取适应度景观分析指标
 *
 * @param system 规划系统句柄
 * @param metrics 景观指标输出缓冲区
 * @return int 成功返回0,失败返回-1
 */
int planning_evolution_get_landscape(PlanningSystem* system, LandscapeMetrics* metrics);

/**
 * @brief 获取Pareto前沿个体数
 *
 * @param system 规划系统句柄
 * @return int Pareto前沿个体数,失败返回-1
 */
int planning_evolution_get_pareto_front_size(PlanningSystem* system);

/**
 * @brief 获取Pareto前沿个体目标值
 *
 * @param system 规划系统句柄
 * @param front_indices Pareto前沿个体索引输出缓冲区(调用者分配)
 * @param max_size 缓冲区最大大小
 * @return int 实际写入的个体数,失败返回-1
 */
int planning_evolution_get_pareto_front(PlanningSystem* system, int* front_indices, int max_size);

/**
 * @brief 获取多个目标值
 *
 * @param system 规划系统句柄
 * @param individual_idx 个体索引
 * @param objectives 多目标值输出缓冲区(至少num_objectives大小)
 * @param num_objectives 目标数量
 * @return int 成功返回0,失败返回-1
 */
int planning_evolution_get_objectives(PlanningSystem* system, size_t individual_idx,
                                       float* objectives, int num_objectives);

/**
 * @brief 设置多目标演化配置(运行时修改)
 *
 * @param system 规划系统句柄
 * @param config 多目标配置
 * @return int 成功返回0,失败返回-1
 */
int planning_evolution_set_multi_objective(PlanningSystem* system, const MultiObjectiveConfig* config);

/**
 * @brief 设置自我适应参数配置(运行时修改)
 *
 * @param system 规划系统句柄
 * @param config 自我适应配置
 * @return int 成功返回0,失败返回-1
 */
int planning_evolution_set_self_adaptive(PlanningSystem* system, const SelfAdaptiveConfig* config);

/**
 * @brief 设置Lamarckian演化配置(运行时修改)
 *
 * @param system 规划系统句柄
 * @param config Lamarckian配置
 * @return int 成功返回0,失败返回-1
 */
int planning_evolution_set_lamarckian(PlanningSystem* system, const LamarckianConfig* config);

/**
 * @brief 设置岛屿模型配置(运行时修改)
 *
 * @param system 规划系统句柄
 * @param config 岛屿模型配置
 * @return int 成功返回0,失败返回-1
 */
int planning_evolution_set_island(PlanningSystem* system, const IslandConfig* config);

/* ========== AGI-04+P2-1: 自我演化架构深度增强 ========== */

/**
 * @brief 增强版演化架构(支持拓扑应用到CfC + GP子程序演化)
 *
 * 在原有规划架构演化基础上增加:
 * - 拓扑基因应用到实际CfC细胞(hidden_size, time_constant等)
 * - CfC细胞参数演化(time_constant, noise_std, feedback_strength等)
 * - 遗传规划(GP)子程序演化
 * - 基于CfC前向传播的性能适应度评估
 *
 * @param system 规划系统句柄
 * @param num_generations 演化代数
 * @param feedback 环境反馈数组
 * @param feedback_size 反馈大小
 * @param enable_gp 是否启用遗传规划子程序演化(非0=启用)
 * @param apply_to_cfc 是否将演化架构应用到实际CfC细胞(非0=应用)
 * @return int 成功返回0,失败返回-1
 */
int planning_evolve_architecture_v2(PlanningSystem* system, size_t num_generations,
                                     const float* feedback, size_t feedback_size,
                                     int enable_gp, int apply_to_cfc);

/**
 * @brief 扩展基因组以包含CfC参数
 *
 * 将当前演化种群基因组扩展到包含CfC细胞参数:
 * time_constant, noise_std, feedback_strength, input_gain, output_gain, delta_t
 *
 * @param system 规划系统句柄
 * @return int 成功返回0,失败返回-1
 */
int planning_evolution_extend_genome_cfc(PlanningSystem* system);

/**
 * @brief 获取当前CfC细胞演化后的参数
 *
 * @param system 规划系统句柄
 * @param time_constant 时间常数输出
 * @param noise_std 噪声标准差输出
 * @param feedback_strength 反馈强度输出
 * @param input_gain 输入增益输出
 * @param output_gain 输出增益输出
 * @param delta_t 时间步长输出
 * @return int 成功返回0,失败返回-1
 */
int planning_evolution_get_cfc_params(const PlanningSystem* system,
                                       float* time_constant,
                                       float* noise_std,
                                       float* feedback_strength,
                                       float* input_gain,
                                       float* output_gain,
                                       float* delta_t);

/* ========== 时序约束传播 ========== */

typedef enum {
    TCONSTRAINT_BEFORE = 0,
    TCONSTRAINT_AFTER = 1,
    TCONSTRAINT_EQUALS = 2,
    TCONSTRAINT_DURING = 3,
    TCONSTRAINT_OVERLAPS = 4,
    TCONSTRAINT_MEETS = 5,
    TCONSTRAINT_STARTS = 6,
    TCONSTRAINT_FINISHES = 7,
    TCONSTRAINT_BEFORE_OR_MEETS = 8,
    TCONSTRAINT_AFTER_OR_MEETS = 9,
    TCONSTRAINT_DISJOINT = 10
} TemporalConstraintType;

typedef struct {
    int from_step;
    int to_step;
    TemporalConstraintType type;
    float min_gap;
    float max_gap;
    float weight;
} TemporalConstraint;

typedef struct {
    TemporalConstraint* constraints;
    int constraint_count;
    int constraint_capacity;
    float* earliest_start;
    float* latest_start;
    float* earliest_end;
    float* latest_end;
    float* durations;
    int step_count;
    int is_consistent;
    int relaxation_count;
} TemporalConstraintNetwork;

int planning_temporal_constraint_add(PlanningSystem* system,
                                     int from_step, int to_step,
                                     TemporalConstraintType type,
                                     float min_gap, float max_gap);

int planning_temporal_constraint_network_create(PlanningSystem* system);
void planning_temporal_constraint_network_destroy(PlanningSystem* system);

int planning_temporal_forward_propagate(PlanningSystem* system);
int planning_temporal_backward_propagate(PlanningSystem* system);
int planning_temporal_consistency_check(PlanningSystem* system);
int planning_temporal_relax_constraints(PlanningSystem* system, float max_relax_ratio);

int planning_temporal_get_time_bounds(const PlanningSystem* system, int step,
                                      float* earliest, float* latest);

/* ========== 规划修复算法 ========== */

typedef struct {
    int* affected_steps;
    int num_affected;
    int repair_strategy;
    float* repaired_plan;
    int repaired_length;
    float repair_quality;
    float estimated_cost;
    int use_affected_only;
} PlanRepairResult;

typedef enum {
    PLAN_REPAIR_STRATEGY_LOCAL = 0,
    PLAN_REPAIR_STRATEGY_GLOBAL = 1,
    PLAN_REPAIR_STRATEGY_HYBRID = 2,
    PLAN_REPAIR_STRATEGY_CONTINGENCY = 3
} PlanRepairStrategy;

int planning_repair_detect_conflicts(PlanningSystem* system,
                                     const float* current_state, size_t state_size,
                                     int* conflict_steps, int max_conflicts);

int planning_repair_local(PlanningSystem* system,
                          const float* plan, size_t plan_size,
                          const int* conflict_steps, int num_conflicts,
                          const float* current_state, size_t state_size,
                          float* repaired_plan, size_t max_plan_size,
                          PlanRepairResult* result);

int planning_repair_global(PlanningSystem* system,
                           const float* current_state, size_t state_size,
                           const float* goal, size_t goal_size,
                           float* repaired_plan, size_t max_plan_size,
                           PlanRepairResult* result);

int planning_repair_hybrid(PlanningSystem* system,
                           const float* plan, size_t plan_size,
                           const int* conflict_steps, int num_conflicts,
                           const float* current_state, size_t state_size,
                           const float* goal, size_t goal_size,
                           float* repaired_plan, size_t max_plan_size,
                           PlanRepairResult* result);

int planning_repair_validate(const PlanningSystem* system,
                             const float* plan, size_t plan_size,
                             const float* current_state, size_t state_size,
                             float* feasibility_out);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_PLANNING_H