/**
 * @file planning_enhanced.h
 * @brief 增强规划引擎接口（时间推理 + 并发规划 + 条件规划）
 *
 * 基于CfC液态神经网络的增强规划系统，扩展基础的规划能力：
 * 1. A04.2.2 时间推理规划 — STN(Simple Temporal Network) + CfC液态时序推理
 * 2. A04.2.2 并发多智能体规划 — 多智能体协同时序规划
 * 3. A04.3.1 HTN层次任务网规划增强 — 任务分解 + 方法选择
 * 4. 条件规划 — 感知信息决策规划
 * 5. 偏序规划 — 部分顺序规划
 *
 * 100% 纯C实现，无第三方库依赖。
 */

#ifndef SELFLNN_PLANNING_ENHANCED_H
#define SELFLNN_PLANNING_ENHANCED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 配置常量
 * ============================================================================ */
#define PLAN_ENH_MAX_TASKS          65536   /**< 最大任务数 */
#define PLAN_ENH_MAX_ACTIONS        262144  /**< 最大动作数 */
#define PLAN_ENH_MAX_CONSTRAINTS    131072  /**< 最大约束数 */
#define PLAN_ENH_MAX_AGENTS         256     /**< 最大智能体数 */
#define PLAN_ENH_MAX_METHODS        4096    /**< 最大分解方法数 */
#define PLAN_ENH_MAX_PLAN_DEPTH     1024    /**< 最大规划深度 */
#define PLAN_ENH_MAX_CONDITIONS     512     /**< 最大条件数 */

/* ============================================================================
 * 类型定义
 * ============================================================================ */

/**
 * @brief 增强规划算法类型
 */
typedef enum {
    PLAN_ENH_TEMPORAL = 0,                  /**< 时间推理规划 (STN/STPU) */
    PLAN_ENH_CONCURRENT,                     /**< 并发多智能体规划 */
    PLAN_ENH_HTN,                            /**< 层次任务网规划 */
    PLAN_ENH_CONDITIONAL,                    /**< 条件规划 */
    PLAN_ENH_PARTIAL_ORDER,                  /**< 偏序规划 */
    PLAN_ENH_LIQUID_CFC                      /**< CfC液态演化规划 */
} PlanEnhancedAlgorithm;

/**
 * @brief 时间约束类型（A04.2.2）
 */
typedef enum {
    TIME_CONSTRAINT_BOUNDED = 0,            /**< [min, max] 时间窗口 */
    TIME_CONSTRAINT_AT,                      /**< 精确时间点 */
    TIME_CONSTRAINT_AFTER,                   /**< 在时间点之后 */
    TIME_CONSTRAINT_BEFORE,                  /**< 在时间点之前 */
    TIME_CONSTRAINT_BETWEEN,                 /**< 在两个事件之间 */
    TIME_CONSTRAINT_DURATION,                /**< 持续时间约束 */
    TIME_CONSTRAINT_RECURRING                /**< 周期性约束 */
} TimeConstraintType;

/**
 * @brief 时序约束
 */
typedef struct {
    int constraint_id;                      /**< 约束ID */
    TimeConstraintType type;                /**< 约束类型 */
    int from_action;                        /**< 源动作ID */
    int to_action;                          /**< 目标动作ID（可选，-1表示无） */
    float min_time;                         /**< 最小时间 */
    float max_time;                         /**< 最大时间 */
    float duration;                         /**< 持续时间 */
    float period;                           /**< 周期（周期性约束） */
    float flexibility;                      /**< 柔性度(0~1, 1=完全柔性) */
    char description[128];                  /**< 约束描述 */
} PlanTimeConstraint;

/**
 * @brief 规划动作
 */
typedef struct {
    int action_id;                          /**< 动作ID */
    char name[64];                          /**< 动作名称 */
    float duration_min;                     /**< 最小持续时间 */
    float duration_max;                     /**< 最大持续时间 */
    float cost;                             /**< 执行成本 */
    float* preconditions;                    /**< 前提条件向量 */
    size_t precond_size;                    /**< 前提条件大小 */
    float* effects;                          /**< 效果向量 */
    size_t effect_size;                     /**< 效果大小 */
    int num_required_resources;             /**< 所需资源数 */
    int required_agent_type;                /**< 所需智能体类型(-1=任意) */
    int is_parallelizable;                  /**< 是否可并行执行 */
    float cfc_activation;                   /**< CfC液态激活度 */
} PlanEnhancedAction;

/**
 * @brief HTN任务（A04.3.1）
 */
typedef struct {
    int task_id;                            /**< 任务ID */
    char name[64];                          /**< 任务名称 */
    int is_primitive;                       /**< 是否为原语任务(1=是,0=复合任务) */
    int* subtasks;                           /**< 子任务ID数组 */
    int subtask_count;                      /**< 子任务数 */
    int* decomposition_methods;             /**< 分解方法ID数组 */
    int method_count;                       /**< 分解方法数 */
    float priority;                         /**< 优先级(0~1) */
    float deadline;                         /**< 截止时间(-1=无) */
    float* preconditions;                    /**< 前提条件 */
    size_t precond_size;                    /**< 前提条件大小 */
} PlanHTNTask;

/**
 * @brief HTN分解方法
 */
typedef struct {
    int method_id;                          /**< 方法ID */
    char name[64];                          /**< 方法名称 */
    int task_id;                            /**< 所属任务ID */
    int* subtask_ids;                        /**< 子任务ID数组 */
    int subtask_count;                      /**< 子任务数 */
    float* applicability_conditions;         /**< 适用条件向量 */
    size_t cond_size;                       /**< 条件大小 */
    float cost_estimate;                    /**< 估计成本 */
    float success_probability;              /**< 成功概率 */
    float priority;                         /**< 方法优先级 */
} PlanHTNMethod;

/**
 * @brief 智能体规格
 */
typedef struct {
    int agent_id;                           /**< 智能体ID */
    char name[64];                          /**< 智能体名称 */
    int agent_type;                         /**< 智能体类型 */
    float capabilities[32];                 /**< 能力向量 */
    int capability_count;                   /**< 能力数 */
    float max_speed;                        /**< 最大速度 */
    float max_load;                         /**< 最大负载 */
    float energy_per_action;                /**< 每动作能耗 */
    float position[3];                      /**< 位置[x,y,z] */
    int is_available;                       /**< 是否可用 */
} PlanAgent;

/**
 * @brief 时间规划结果
 */
typedef struct {
    int* action_ids;                         /**< 动作ID序列 */
    float* start_times;                      /**< 开始时间序列 */
    float* end_times;                        /**< 结束时间序列 */
    int action_count;                       /**< 动作数 */
    int* assigned_agents;                    /**< 分配智能体ID */
    int agent_count;                        /**< 智能体数 */
    float total_cost;                       /**< 总成本 */
    float total_duration;                   /**< 总持续时间 */
    float makespan;                         /**< 完成时间 */
    float plan_confidence;                  /**< 规划置信度 */
    int is_feasible;                        /**< 是否可行 */
    char plan_summary[256];                 /**< 规划摘要 */
} PlanEnhancedResult;

/**
 * @brief 增强规划配置
 */
typedef struct {
    PlanEnhancedAlgorithm algorithm;        /**< 规划算法 */
    int max_actions;                        /**< 最大动作数 */
    int max_plan_length;                    /**< 最大规划长度 */
    float time_horizon;                     /**< 时间范围 */
    float deadline;                         /**< 全局截止时间 */
    int num_agents;                         /**< 智能体数 */
    float risk_tolerance;                   /**< 风险容忍度 */
    float resource_limit;                   /**< 资源上限 */
    int enable_parallel;                    /**< 启用并行 */
    int enable_replanning;                  /**< 启用重规划 */
    int enable_temporal_reasoning;          /**< 启用时间推理 */
    int enable_conditional_branches;        /**< 启用条件分支 */
    float cfc_tau;                          /**< CfC时间常数 */
    float cfc_dt;                           /**< CfC步长 */
    int cfc_steps;                          /**< CfC积分步数 */
} PlanEnhancedConfig;

/**
 * @brief 增强规划引擎句柄
 */
typedef struct PlanEnhancedEngine PlanEnhancedEngine;

/* ============================================================================
 * A04.2.2 — 时间推理规划核心API
 * ============================================================================ */

/**
 * @brief 创建增强规划引擎
 *
 * @param config 规划配置
 * @return PlanEnhancedEngine* 成功返回句柄，失败返回NULL
 */
PlanEnhancedEngine* plan_enhanced_create(const PlanEnhancedConfig* config);

/**
 * @brief 销毁增强规划引擎
 *
 * @param engine 规划引擎句柄
 */
void plan_enhanced_destroy(PlanEnhancedEngine* engine);

/**
 * @brief 添加规划动作
 *
 * @param engine 规划引擎
 * @param action 动作定义
 * @return int 成功返回动作ID，失败返回-1
 */
int plan_enhanced_add_action(PlanEnhancedEngine* engine,
                              const PlanEnhancedAction* action);

/**
 * @brief 添加时间约束
 *
 * @param engine 规划引擎
 * @param constraint 时间约束
 * @return int 成功返回约束ID，失败返回-1
 */
int plan_enhanced_add_time_constraint(PlanEnhancedEngine* engine,
                                       const PlanTimeConstraint* constraint);

/**
 * @brief 添加HTN任务（A04.3.1）
 *
 * @param engine 规划引擎
 * @param task HTN任务定义
 * @return int 成功返回任务ID，失败返回-1
 */
int plan_enhanced_add_htn_task(PlanEnhancedEngine* engine,
                                const PlanHTNTask* task);

/**
 * @brief 添加HTN分解方法
 *
 * @param engine 规划引擎
 * @param method 分解方法定义
 * @return int 成功返回方法ID，失败返回-1
 */
int plan_enhanced_add_htn_method(PlanEnhancedEngine* engine,
                                  const PlanHTNMethod* method);

/**
 * @brief 注册智能体
 *
 * @param engine 规划引擎
 * @param agent 智能体定义
 * @return int 成功返回智能体ID，失败返回-1
 */
int plan_enhanced_register_agent(PlanEnhancedEngine* engine,
                                  const PlanAgent* agent);

/**
 * @brief 生成时间规划
 *
 * 使用STN(Simple Temporal Network) + 差分约束求解：
 *   最小化 C = Σ cost(action_i)
 *   约束: t_j - t_i ∈ [d_min, d_max]
 *        duration_i ∈ [dur_min, dur_max]
 *   CfC ODE优化: dt/dτ = -∇C(t) / τ
 *
 * @param engine 规划引擎
 * @param goal_state 目标状态向量
 * @param goal_size 目标状态大小
 * @param initial_state 初始状态向量
 * @param state_size 状态大小
 * @param result 规划结果输出
 * @return int 成功返回0，失败返回-1
 */
int plan_enhanced_generate_temporal(PlanEnhancedEngine* engine,
                                     const float* goal_state, size_t goal_size,
                                     const float* initial_state, size_t state_size,
                                     PlanEnhancedResult* result);

/**
 * @brief 生成并发多智能体规划
 *
 * 多智能体并发规划使用冲突检测与资源约束：
 *   1. 为每个智能体独立生成子规划
 *   2. 检测动作间资源冲突和时序冲突
 *   3. 使用CfC门控协调解决冲突
 *      c_gate = σ(W_c · [h_i; h_j; r_shared])
 *   4. 调整时序窗口消解冲突
 *
 * @param engine 规划引擎
 * @param goal_states 各智能体目标状态 [num_agents][goal_size]
 * @param goal_size 目标状态大小
 * @param results 各智能体规划结果数组 [num_agents]
 * @param max_results 最大结果数
 * @return int 实际生成的结果数，失败返回-1
 */
int plan_enhanced_generate_concurrent(PlanEnhancedEngine* engine,
                                       const float* goal_states, size_t goal_size,
                                       PlanEnhancedResult* results, int max_results);

/**
 * @brief 生成HTN层次规划（A04.3.1）
 *
 * 自上而下的任务分解 + 方法选择：
 *   1. 从根任务开始
 *   2. 对每个复合任务，选择最合适的方法
 *   3. 递归分解直到所有子任务为原语动作
 *   4. 方法选择使用CfC液态评估：
 *      m* = argmax P(success|m) / cost(m)
 *
 * @param engine 规划引擎
 * @param root_task_id 根任务ID
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @param result 规划结果输出
 * @return int 成功返回0，失败返回-1
 */
int plan_enhanced_generate_htn(PlanEnhancedEngine* engine,
                                int root_task_id,
                                const float* current_state, size_t state_size,
                                PlanEnhancedResult* result);

/**
 * @brief 执行STN时间一致性检查
 *
 * 使用Floyd-Warshall / Bellman-Ford检测时间网络一致性：
 *   差分约束图: x_j - x_i ≤ w_ij
 *   一致性条件: 无负环
 *   CfC加速: d_dist/dt = -dist/τ + f_cfc(dist, w)
 *
 * @param engine 规划引擎
 * @param constraints 约束数组
 * @param num_constraints 约束数
 * @return int 一致返回1，不一致返回0，错误返回-1
 */
int plan_enhanced_check_temporal_consistency(PlanEnhancedEngine* engine,
                                              const PlanTimeConstraint* constraints,
                                              int num_constraints);

/**
 * @brief 偏序规划生成
 *
 * 生成动作间的偏序关系而非全序：
 *   动作a必须在动作b之前当且仅当 a产生的前提被b使用
 *   使用CfC排序：P(before|a,b) = σ(W_o·[h_a; h_b])
 *
 * @param engine 规划引擎
 * @param goal_state 目标状态
 * @param goal_size 目标状态大小
 * @param initial_state 初始状态
 * @param state_size 状态大小
 * @param partial_order_pairs 输出偏序对 [(action_before, action_after)]
 * @param max_pairs 最大偏序对数
 * @return int 偏序对数，失败返回-1
 */
int plan_enhanced_generate_partial_order(PlanEnhancedEngine* engine,
                                          const float* goal_state, size_t goal_size,
                                          const float* initial_state, size_t state_size,
                                          int* partial_order_pairs, int max_pairs);

/* ============================================================================
 * 条件规划API
 * ============================================================================ */

/**
 * @brief 生成条件规划（带感知分支）
 *
 * 条件规划树：
 *   root → [感知] → 分支1 (条件C1) → 动作序列1
 *                  → 分支2 (条件C2) → 动作序列2
 *   CfC条件评估: P(Ck|percept) = softmax(W_p · h_percept)
 *
 * @param engine 规划引擎
 * @param goal_state 目标状态
 * @param goal_size 目标状态大小
 * @param initial_state 初始状态
 * @param state_size 状态大小
 * @param sensing_vars 感知变量索引数组
 * @param num_sensing_vars 感知变量数
 * @param results 分支规划结果数组
 * @param max_branches 最大分支数
 * @return int 分支数，失败返回-1
 */
int plan_enhanced_generate_conditional(PlanEnhancedEngine* engine,
                                        const float* goal_state, size_t goal_size,
                                        const float* initial_state, size_t state_size,
                                        const int* sensing_vars, int num_sensing_vars,
                                        PlanEnhancedResult* results, int max_branches);

/* ============================================================================
 * 重规划与自适应
 * ============================================================================ */

/**
 * @brief 执行重规划（当环境变化时）
 *
 * 增量式重规划：
 *   1. 检测已失效的动作
 *   2. 标记受影响的后继动作
 *   3. 只修复受影响的部分
 *   4. CfC快速适应: h_new = h_old + Δh(env_change)
 *
 * @param engine 规划引擎
 * @param original_result 原始规划结果
 * @param state_delta 状态变化向量
 * @param delta_size 变化大小
 * @param new_result 新规划结果输出
 * @return int 成功返回0，失败返回-1
 */
int plan_enhanced_replan(PlanEnhancedEngine* engine,
                          const PlanEnhancedResult* original_result,
                          const float* state_delta, size_t delta_size,
                          PlanEnhancedResult* new_result);

/**
 * @brief 执行规划可行性验证
 *
 * 从多个维度验证规划可行性：
 *   1. 资源可行性：Σ resource_usage ≤ resource_limit
 *   2. 时间可行性：所有时间约束一致
 *   3. 因果可行性：前提条件满足
 *   4. 物理可行性：运动学和动力学约束
 *
 * @param engine 规划引擎
 * @param result 规划结果
 * @return float 可行性分数(0~1)
 */
float plan_enhanced_validate_feasibility(PlanEnhancedEngine* engine,
                                          const PlanEnhancedResult* result);

/* ============================================================================
 * A04.2.2 — FF/FFD启发式规划API
 * ============================================================================ */

/**
 * @brief 计算FF启发式值（删除效应放松的规划图启发式）
 *
 * 在删除放松的规划图中，从当前状态到目标状态所需的最少动作数。
 * 对每个未满足的目标命题，找到第一个能使其成立的动作为止。
 *
 * @param engine 规划引擎
 * @param state 当前状态向量
 * @param state_size 状态大小
 * @param goal_state 目标状态向量
 * @param goal_size 目标大小
 * @return float FF启发式值(h值，越小越好)，失败返回-1
 */
float plan_enhanced_ff_heuristic(PlanEnhancedEngine* engine,
                                  const float* state, size_t state_size,
                                  const float* goal_state, size_t goal_size);

/**
 * @brief 计算FFD启发式值（删除放松的规划图启发式+优选动作）
 *
 * FFD扩展：在规划图构建过程中偏好具有更高"帮助度"的动作。
 * 帮助度：一个动作同时有助于满足多个未达成目标的程度。
 *
 * @param engine 规划引擎
 * @param state 当前状态向量
 * @param state_size 状态大小
 * @param goal_state 目标状态向量
 * @param goal_size 目标大小
 * @return float FFD启发式值，失败返回-1
 */
float plan_enhanced_ffd_heuristic(PlanEnhancedEngine* engine,
                                   const float* state, size_t state_size,
                                   const float* goal_state, size_t goal_size);

/**
 * @brief 使用FF算法生成规划（Fast Forward规划器）
 *
 * 前向状态空间搜索 + FF启发式引导：
 *   1. 从初始状态开始
 *   2. 应用可行动作生成后继状态
 *   3. 使用FF启发式评估每个后继
 *   4. 选择h值最小的方向继续搜索
 *   5. 到达目标状态时停止
 *   6. 提取动作序列作为规划
 *
 * @param engine 规划引擎
 * @param goal_state 目标状态向量
 * @param goal_size 目标大小
 * @param initial_state 初始状态向量
 * @param state_size 状态大小
 * @param result 规划结果输出
 * @return int 成功返回0，失败返回-1
 */
int plan_enhanced_generate_ff(PlanEnhancedEngine* engine,
                               const float* goal_state, size_t goal_size,
                               const float* initial_state, size_t state_size,
                               PlanEnhancedResult* result);

/**
 * @brief 使用FFD算法生成规划（Fast Forward with Dead-end detection）
 *
 * 贪婪最佳优先搜索 + FFD启发式 + 死端检测：
 *   1. 使用FFD启发式引导搜索方向
 *   2. 检测不可解状态（死端）并触发回溯
 *   3. 偏好帮助多个目标的动作
 *   4. 目标计数启发式辅助修剪
 *
 * @param engine 规划引擎
 * @param goal_state 目标状态向量
 * @param goal_size 目标大小
 * @param initial_state 初始状态向量
 * @param state_size 状态大小
 * @param result 规划结果输出
 * @return int 成功返回0，失败返回-1
 */
int plan_enhanced_generate_ffd(PlanEnhancedEngine* engine,
                                const float* goal_state, size_t goal_size,
                                const float* initial_state, size_t state_size,
                                PlanEnhancedResult* result);

/**
 * @brief 释放规划结果资源
 *
 * @param result 规划结果指针
 */
void plan_enhanced_free_result(PlanEnhancedResult* result);

/**
 * @brief 保存增强规划模型
 *
 * @param engine 规划引擎
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int plan_enhanced_save(const PlanEnhancedEngine* engine, const char* filepath);

/**
 * @brief 加载增强规划模型
 *
 * @param filepath 文件路径
 * @return PlanEnhancedEngine* 成功返回句柄，失败返回NULL
 */
PlanEnhancedEngine* plan_enhanced_load(const char* filepath);

/**
 * @brief 获取默认增强规划配置
 *
 * @param algorithm 规划算法类型
 * @return PlanEnhancedConfig 默认配置
 */
PlanEnhancedConfig plan_enhanced_default_config(PlanEnhancedAlgorithm algorithm);

/* ============================================================================
 * A04.2.2 — Landmark-based规划API
 * ============================================================================ */

/**
 * @brief Landmark类型
 */
typedef enum {
    PLAN_LANDMARK_FACT = 0,              /**< 事实Landmark（状态中的命题必须为真） */
    PLAN_LANDMARK_ACTION,                 /**< 动作Landmark（动作必须被执行） */
    PLAN_LANDMARK_DISJUNCTIVE             /**< 析取Landmark（多选一达成） */
} PlanLandmarkType;

/**
 * @brief Landmark节点
 */
typedef struct {
    int landmark_id;                     /**< Landmark ID */
    PlanLandmarkType type;               /**< Landmark类型 */
    float* fact_vector;                  /**< 事实向量（必须满足的状态子集） */
    size_t fact_size;                    /**< 事实向量大小 */
    float importance;                    /**< 重要性(0~1) */
    int is_achieved;                     /**< 是否已达成 */
    int* predecessors;                   /**< 前置Landmark ID数组 */
    int pred_count;                      /**< 前置数量 */
    int* successors;                     /**< 后置Landmark ID数组 */
    int succ_count;                      /**< 后置数量 */
    float discovery_cost;                /**< 发现代价估计 */
} PlanLandmark;

/**
 * @brief Landmark图（Landmark Graph）
 *
 * Landmark之间的顺序关系图，用于启发式评估和规划引导。
 * 通过因果分析从规划问题中提取landmark及它们的序关系。
 */
typedef struct {
    PlanLandmark* landmarks;             /**< Landmark数组 */
    int landmark_count;                  /**< Landmark数量 */
    int landmark_capacity;               /**< Landmark容量 */
    float* initial_state;                /**< 初始状态备份 */
    size_t state_size;                   /**< 状态维度 */
    float* goal_state;                   /**< 目标状态备份 */
    int* landmark_order_matrix;          /**< 序关系矩阵 [landmark_count][landmark_count] */
    int* level_of_landmark;              /**< Landmark层次（在规划图中的层数） */
} PlanLandmarkGraph;

/**
 * @brief 从规划问题中提取Landmark图
 *
 * 使用因果分析（Causal Analysis）方法提取Landmark：
 *   1. 目标命题 = 初始Landmark集合
 *   2. 对每个Landmark p，找到首次使p成立的动作为a
 *   3. a的所有前提条件 = 新的Landmark候选
 *   4. 递归直到初始状态
 *   5. 构建Landmark之间的序关系
 *
 * @param engine 规划引擎
 * @param initial_state 初始状态向量
 * @param state_size 状态大小
 * @param goal_state 目标状态向量
 * @param goal_size 目标大小
 * @return PlanLandmarkGraph* 成功返回Landmark图，失败返回NULL
 */
PlanLandmarkGraph* plan_enhanced_extract_landmarks(PlanEnhancedEngine* engine,
                                                    const float* initial_state, size_t state_size,
                                                    const float* goal_state, size_t goal_size);

/**
 * @brief 销毁Landmark图
 *
 * @param graph Landmark图指针
 */
void plan_enhanced_destroy_landmark_graph(PlanLandmarkGraph* graph);

/**
 * @brief 获取Landmark数量
 *
 * @param graph Landmark图
 * @return int Landmark数量，失败返回-1
 */
int plan_enhanced_get_landmark_count(const PlanLandmarkGraph* graph);

/**
 * @brief 获取指定Landmark
 *
 * @param graph Landmark图
 * @param index Landmark索引
 * @param landmark 输出Landmark
 * @return int 成功返回0，失败返回-1
 */
int plan_enhanced_get_landmark(const PlanLandmarkGraph* graph, int index,
                                PlanLandmark* landmark);

/**
 * @brief 计算Landmark启发式值
 *
 * 从当前状态到达成所有Landmark所需的剩余Landmark计数：
 *   h_LM(s) = |{ l : l ∈ landmarks ∧ l ∉ achieved(s) }|
 *
 * @param engine 规划引擎
 * @param state 当前状态向量
 * @param state_size 状态大小
 * @param graph Landmark图
 * @return float Landmark启发式值，失败返回-1
 */
float plan_enhanced_landmark_heuristic(PlanEnhancedEngine* engine,
                                        const float* state, size_t state_size,
                                        const PlanLandmarkGraph* graph);

/**
 * @brief 使用Landmark-based算法生成规划
 *
 * A*搜索 + Landmark计数启发式：
 *   1. 提取Landmark图
 *   2. 使用Landmark启发式引导搜索
 *   3. 优先探索已达成更多Landmark的状态
 *   4. 支持Landmark序关系约束
 *
 * @param engine 规划引擎
 * @param goal_state 目标状态向量
 * @param goal_size 目标大小
 * @param initial_state 初始状态向量
 * @param state_size 状态大小
 * @param result 规划结果输出
 * @return int 成功返回0，失败返回-1
 */
int plan_enhanced_generate_landmark_based(PlanEnhancedEngine* engine,
                                           const float* goal_state, size_t goal_size,
                                           const float* initial_state, size_t state_size,
                                           PlanEnhancedResult* result);

/* ============================================================================
 * A04.2.2 — 符号规划API (Symbolic/STRIPS Planning)
 * ============================================================================ */

/**
 * @brief 符号动作（STRIPS风格）
 *
 * 使用离散符号表示的动作，支持正向/负向前提条件和添加/删除效果。
 * 状态表示为布尔向量，true=命题成立，false=不成立。
 */
typedef struct {
    int action_id;                       /**< 动作ID（关联PlanEnhancedAction） */
    int* positive_preconds;              /**< 正向前提条件（必须为真的命题索引） */
    int pos_precond_count;               /**< 正向前提条件数 */
    int* negative_preconds;              /**< 负向前提条件（必须为假的命题索引） */
    int neg_precond_count;               /**< 负向前提条件数 */
    int* add_effects;                    /**< 添加效果（设为真的命题索引） */
    int add_effect_count;                /**< 添加效果数 */
    int* del_effects;                    /**< 删除效果（设为假的命题索引） */
    int del_effect_count;                /**< 删除效果数 */
    float cost;                          /**< 动作成本 */
} PlanSymbolicAction;

/**
 * @brief 符号状态（布尔状态向量 + CfC液态嵌入）
 */
typedef struct {
    unsigned char* bool_state;           /**< 布尔状态向量（0/1） */
    size_t state_size;                   /**< 状态维度 */
    float* cfc_embedding;                /**< CfC液态嵌入（连续辅助） */
    size_t embed_dim;                    /**< 嵌入维度 */
    float g_value;                       /**< A* g值（已耗代价） */
    float h_value;                       /**< A* h值（启发式估计） */
} PlanSymbolicState;

/**
 * @brief 符号规划器
 */
typedef struct {
    PlanSymbolicAction* actions;         /**< 符号动作数组 */
    int action_count;                    /**< 符号动作数 */
    size_t state_size;                   /**< 状态维度 */
    int* goal_props;                     /**< 目标命题索引数组 */
    int goal_count;                      /**< 目标命题数 */
    int* initial_true_props;             /**< 初始真命题索引数组 */
    int initial_true_count;              /**< 初始真命题数 */
    float* cfc_state;                    /**< CfC液态状态（可选） */
    int cfc_dim;                         /**< CfC维度 */
} PlanSymbolicPlanner;

/**
 * @brief 创建符号规划器（从增强引擎自动转换）
 *
 * 将PlanEnhancedAction转换为PlanSymbolicAction：
 *   正向前提: precondition[i] > 0.5
 *   负向前提: precondition[i] < -0.5
 *   添加效果: effect[i] > 0.5
 *   删除效果: effect[i] < -0.5
 *
 * @param engine 增强规划引擎
 * @param state_size 状态维度
 * @return PlanSymbolicPlanner* 成功返回符号规划器，失败返回NULL
 */
PlanSymbolicPlanner* plan_enhanced_create_symbolic_planner(PlanEnhancedEngine* engine,
                                                            size_t state_size);

/**
 * @brief 销毁符号规划器
 *
 * @param planner 符号规划器
 */
void plan_enhanced_destroy_symbolic_planner(PlanSymbolicPlanner* planner);

/**
 * @brief 检查符号动作是否可应用
 *
 * @param planner 符号规划器
 * @param action_index 动作索引
 * @param state 布尔状态向量
 * @return int 可应用返回1，否则返回0
 */
int plan_enhanced_symbolic_action_applicable(const PlanSymbolicPlanner* planner,
                                              int action_index,
                                              const unsigned char* state);

/**
 * @brief 应用符号动作（状态推进）
 *
 * 在布尔状态下应用动作的效果：
 *   对所有add_effects[i]: state[add_effects[i]] = 1
 *   对所有del_effects[i]: state[del_effects[i]] = 0
 *
 * @param planner 符号规划器
 * @param action_index 动作索引
 * @param state 输入状态（将被修改）
 */
void plan_enhanced_symbolic_apply_action(const PlanSymbolicPlanner* planner,
                                          int action_index,
                                          unsigned char* state);

/**
 * @brief 计算符号规划启发式（max/SUM/FF风格）
 *
 * 使用删除放松的规划图启发式：
 *   忽略删除效果，在放松空间中计算到达目标的最少动作层数
 *
 * @param planner 符号规划器
 * @param state 当前布尔状态
 * @return float 启发式值(h)，失败返回-1
 */
float plan_enhanced_symbolic_heuristic(const PlanSymbolicPlanner* planner,
                                        const unsigned char* state);

/**
 * @brief 生成符号规划（STRIPS前向搜索 + A*）
 *
 * @param planner 符号规划器
 * @param result 规划结果输出（action_id序列）
 * @param max_depth 最大搜索深度
 * @return int 成功返回0（找到规划），无解返回1，失败返回-1
 */
int plan_enhanced_generate_symbolic(PlanSymbolicPlanner* planner,
                                     PlanEnhancedResult* result,
                                     int max_depth);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_PLANNING_ENHANCED_H */

