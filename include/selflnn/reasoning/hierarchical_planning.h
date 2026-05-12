/**
 * @file hierarchical_planning.h
 * @brief 分层规划系统接口
 * 
 * 分层规划（Hierarchical Planning）系统实现，支持：
 * 1. 抽象层次分解（Abstract Hierarchy Decomposition）
 * 2. 任务网络规划（Hierarchical Task Network, HTN）
 * 3. 部分有序规划（Partial Order Planning, POP）
 * 4. 分层强化学习（Hierarchical Reinforcement Learning, HRL）
 * 
 *  ，提供完整的分层规划算法。
 */

#ifndef SELFLNN_HIERARCHICAL_PLANNING_H
#define SELFLNN_HIERARCHICAL_PLANNING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 分层规划算法类型
 */
typedef enum {
    HIERARCHICAL_HTN = 0,          /**< 分层任务网络（HTN） */
    HIERARCHICAL_POP = 1,          /**< 部分有序规划（POP） */
    HIERARCHICAL_HRL = 2,          /**< 分层强化学习（HRL） */
    HIERARCHICAL_ABSTRACT = 3,     /**< 抽象层次分解 */
    HIERARCHICAL_MIXED = 4         /**< 混合分层规划 */
} HierarchicalPlanningAlgorithm;

/**
 * @brief 规划层次结构
 */
typedef enum {
    HIERARCHY_STRATEGIC = 0,       /**< 战略层：长期目标 */
    HIERARCHY_TACTICAL = 1,        /**< 战术层：中期计划 */
    HIERARCHY_OPERATIONAL = 2,     /**< 操作层：短期行动 */
    HIERARCHY_EXECUTION = 3        /**< 执行层：实时控制 */
} PlanningHierarchy;

/**
 * @brief 任务分解方法
 */
typedef enum {
    DECOMPOSITION_AND_OR = 0,      /**< AND/OR分解 */
    DECOMPOSITION_SEQUENTIAL = 1,  /**< 顺序分解 */
    DECOMPOSITION_PARALLEL = 2,    /**< 并行分解 */
    DECOMPOSITION_CONDITIONAL = 3  /**< 条件分解 */
} TaskDecompositionMethod;

/**
 * @brief 分层规划配置
 */
typedef struct {
    HierarchicalPlanningAlgorithm algorithm;   /**< 分层规划算法 */
    int max_hierarchy_levels;                  /**< 最大层次数 */
    float abstraction_threshold;               /**< 抽象阈值 */
    float decomposition_granularity;           /**< 分解粒度 */
    TaskDecompositionMethod decomposition_method; /**< 任务分解方法 */
    int enable_backtracking;                   /**< 是否启用回溯 */
    int max_backtrack_steps;                   /**< 最大回溯步数 */
    float risk_propagation_factor;             /**< 风险传播因子 */
    int enable_parallel_decomposition;         /**< 是否启用并行分解 */
    int enable_dynamic_hierarchy;              /**< 是否启用动态层次调整 */
    float learning_rate;                       /**< 学习率（HRL使用） */
    float discount_factor;                     /**< 折扣因子（HRL使用） */
    int enable_transfer_learning;              /**< 是否启用迁移学习 */
} HierarchicalPlanningConfig;

/**
 * @brief 分层任务节点
 */
typedef struct HierarchicalTaskNode HierarchicalTaskNode;

/**
 * @brief 任务网络
 */
typedef struct {
    HierarchicalTaskNode* root;                /**< 根任务节点 */
    size_t node_count;                         /**< 节点数量 */
    PlanningHierarchy hierarchy_level;         /**< 层次级别 */
    float completion_score;                    /**< 完成度评分 */
    float feasibility_score;                   /**< 可行性评分 */
    float risk_score;                          /**< 风险评分 */
} TaskNetwork;

/**
 * @brief 分层规划状态
 */
typedef struct {
    TaskNetwork* current_network;              /**< 当前任务网络 */
    PlanningHierarchy current_level;           /**< 当前层次 */
    float* abstract_state;                     /**< 抽象状态表示 */
    size_t abstract_state_size;                /**< 抽象状态大小 */
    int decomposition_depth;                   /**< 分解深度 */
    int backtrack_count;                       /**< 回溯次数 */
    float cumulative_reward;                   /**< 累积奖励（HRL） */
    int is_initialized;                        /**< 是否已初始化 */
} HierarchicalPlanningState;

/**
 * @brief 分层规划系统句柄
 */
typedef struct HierarchicalPlanningSystem HierarchicalPlanningSystem;

/**
 * @brief 创建分层规划系统
 * 
 * @param config 分层规划配置
 * @return HierarchicalPlanningSystem* 分层规划系统句柄，失败返回NULL
 */
HierarchicalPlanningSystem* hierarchical_planning_system_create(
    const HierarchicalPlanningConfig* config);

/**
 * @brief 释放分层规划系统
 * 
 * @param system 分层规划系统句柄
 */
void hierarchical_planning_system_free(HierarchicalPlanningSystem* system);

/**
 * @brief 生成分层规划
 * 
 * @param system 分层规划系统句柄
 * @param goal 目标描述
 * @param goal_size 目标描述大小
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @param hierarchy_level 规划层次
 * @param plan 规划输出缓冲区
 * @param max_plan_size 规划最大大小
 * @return int 成功返回规划步骤数，失败返回-1
 */
int hierarchical_planning_generate(HierarchicalPlanningSystem* system,
                                  const float* goal, size_t goal_size,
                                  const float* current_state, size_t state_size,
                                  PlanningHierarchy hierarchy_level,
                                  float* plan, size_t max_plan_size);

/**
 * @brief 任务分解
 * 
 * @param system 分层规划系统句柄
 * @param task 任务描述
 * @param task_size 任务描述大小
 * @param context 上下文信息
 * @param context_size 上下文大小
 * @param subtasks 子任务输出缓冲区
 * @param max_subtasks 最大子任务数
 * @return int 成功返回子任务数，失败返回-1
 */
int hierarchical_task_decomposition(HierarchicalPlanningSystem* system,
                                   const float* task, size_t task_size,
                                   const float* context, size_t context_size,
                                   float* subtasks, size_t max_subtasks);

/**
 * @brief 抽象状态提取
 * 
 * @param system 分层规划系统句柄
 * @param concrete_state 具体状态
 * @param state_size 状态大小
 * @param hierarchy_level 抽象层次
 * @param abstract_state 抽象状态输出缓冲区
 * @param max_abstract_size 抽象状态最大大小
 * @return int 成功返回抽象状态大小，失败返回-1
 */
int hierarchical_state_abstraction(HierarchicalPlanningSystem* system,
                                  const float* concrete_state, size_t state_size,
                                  PlanningHierarchy hierarchy_level,
                                  float* abstract_state, size_t max_abstract_size);

/**
 * @brief 规划评估与优化
 * 
 * @param system 分层规划系统句柄
 * @param plan 规划
 * @param plan_size 规划大小
 * @param hierarchy_level 规划层次
 * @param evaluation_metrics 评估指标输出缓冲区
 * @param max_metrics 最大指标数
 * @return int 成功返回评估指标数，失败返回-1
 */
int hierarchical_planning_evaluate(HierarchicalPlanningSystem* system,
                                  const float* plan, size_t plan_size,
                                  PlanningHierarchy hierarchy_level,
                                  float* evaluation_metrics, size_t max_metrics);

/**
 * @brief 层次间协调
 * 
 * @param system 分层规划系统句柄
 * @param high_level_plan 高层规划
 * @param high_level_size 高层规划大小
 * @param low_level_plan 低层规划
 * @param low_level_size 低层规划大小
 * @param coordinated_plan 协调后规划输出缓冲区
 * @param max_coordinated_size 协调规划最大大小
 * @return int 成功返回协调规划大小，失败返回-1
 */
int hierarchical_planning_coordinate(HierarchicalPlanningSystem* system,
                                    const float* high_level_plan, size_t high_level_size,
                                    const float* low_level_plan, size_t low_level_size,
                                    float* coordinated_plan, size_t max_coordinated_size);

/**
 * @brief 动态层次调整
 * 
 * @param system 分层规划系统句柄
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @param current_plan 当前规划
 * @param plan_size 规划大小
 * @param new_hierarchy_level 新层次级别输出
 * @return int 成功返回0，失败返回-1
 */
int hierarchical_dynamic_adjustment(HierarchicalPlanningSystem* system,
                                   const float* current_state, size_t state_size,
                                   const float* current_plan, size_t plan_size,
                                   PlanningHierarchy* new_hierarchy_level);

/**
 * @brief 获取分层规划状态
 * 
 * @param system 分层规划系统句柄
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int hierarchical_planning_get_state(HierarchicalPlanningSystem* system,
                                   HierarchicalPlanningState* state);

/**
 * @brief 重置分层规划系统
 * 
 * @param system 分层规划系统句柄
 * @return int 成功返回0，失败返回-1
 */
int hierarchical_planning_reset(HierarchicalPlanningSystem* system);

#ifdef __cplusplus
}
#endif

/* ========== P2-1: HTN规划器深度增强 ========== */

/**
 * @brief HTN复合任务方法
 *
 * 定义如何将复合任务分解为子任务序列。
 * 每个方法对应一种分解模式，包含前提条件和子任务签名。
 */
typedef struct {
    float* task_signature;              /**< 匹配的任务签名 */
    size_t signature_size;              /**< 签名大小 */
    float match_threshold;              /**< 匹配阈值 */
    float** subtask_signatures;         /**< 子任务签名数组 */
    size_t* subtask_sizes;              /**< 子任务大小数组 */
    int** ordering_constraints;         /**< 排序约束矩阵(n×n: -1=前,0=无,1=后,2=并行) */
    size_t subtask_count;               /**< 子任务数量 */
    float* preconditions;               /**< 方法前提条件 */
    size_t preconditions_size;          /**< 前提条件大小 */
    float* expected_effects;            /**< 方法预期效果 */
    size_t expected_effects_size;       /**< 预期效果大小 */
    float applicability_score;          /**< 适用性分数(自动学习) */
    float average_cost;                 /**< 平均执行成本 */
    float average_duration;             /**< 平均执行时长 */
    int is_primitive_method;            /**< 是否为原始方法 */
    int use_count;                      /**< 使用次数计数 */
    float success_rate;                 /**< 成功率(自动统计) */
} HTNMethod;

/**
 * @brief HTN原始操作符
 *
 * 定义不可再分解的原始操作，包含前提条件和效果。
 */
typedef struct {
    float* operator_signature;          /**< 操作符签名 */
    size_t signature_size;              /**< 签名大小 */
    float* preconditions;               /**< 前提条件 */
    size_t preconditions_size;          /**< 前提条件大小 */
    float* effects;                     /**< 效果 */
    size_t effects_size;                /**< 效果大小 */
    float cost;                         /**< 执行成本 */
    float duration;                     /**< 执行时长 */
    int use_count;                      /**< 使用次数 */
} HTNOperator;

/**
 * @brief HTN资源类型枚举
 */
typedef enum {
    HTN_RESOURCE_COMPUTATIONAL = 0,    /**< 计算资源 */
    HTN_RESOURCE_MEMORY = 1,           /**< 内存资源 */
    HTN_RESOURCE_TIME = 2,             /**< 时间资源 */
    HTN_RESOURCE_ENERGY = 3,           /**< 能量资源 */
    HTN_RESOURCE_COMMUNICATION = 4,    /**< 通信资源 */
    HTN_RESOURCE_SENSOR = 5,           /**< 传感器资源 */
    HTN_RESOURCE_ACTUATOR = 6,         /**< 执行器资源 */
    HTN_RESOURCE_CUSTOM = 7            /**< 自定义资源 */
} HTNResourceType;

/**
 * @brief HTN资源描述
 */
typedef struct {
    HTNResourceType resource_type;      /**< 资源类型 */
    char name[64];                      /**< 资源名称 */
    float total_capacity;               /**< 总容量 */
    float available;                    /**< 当前可用量 */
    float* allocation_history;          /**< 分配历史 */
    size_t history_size;                /**< 历史大小 */
    size_t history_capacity;            /**< 历史容量 */
    float refill_rate;                  /**< 补充速率(per step) */
} HTNResource;

/**
 * @brief HTN方法选择器配置
 */
typedef struct {
    float precondition_weight;          /**< 前提条件匹配权重(默认0.3) */
    float effect_weight;                /**< 效果匹配权重(默认0.2) */
    float cost_weight;                  /**< 成本权重(默认0.15) */
    float diversity_weight;             /**< 多样性权重(默认0.1) */
    float state_alignment_weight;       /**< 状态对齐权重(默认0.15) */
    float success_rate_weight;          /**< 成功率权重(默认0.1) */
    int top_k_methods;                  /**< 选择前K个方法(默认3) */
    float exploration_rate;             /**< 探索率(默认0.1) */
} HTNMethodSelectorConfig;

/**
 * @brief HTN规划域
 *
 * 包含方法库、操作符库和资源池。
 */
typedef struct {
    HTNMethod* methods;                 /**< 方法数组 */
    size_t method_count;                /**< 方法数量 */
    size_t method_capacity;             /**< 方法容量 */
    HTNOperator* operators;             /**< 操作符数组 */
    size_t operator_count;              /**< 操作符数量 */
    size_t operator_capacity;           /**< 操作符容量 */
    HTNResource* resources;             /**< 资源数组 */
    size_t resource_count;              /**< 资源数量 */
    size_t resource_capacity;           /**< 资源容量 */
} HTNPlanningDomain;

/**
 * @brief 任务交互类型枚举
 */
typedef enum {
    TASK_INTERACTION_NONE = 0,          /**< 无交互 */
    TASK_INTERACTION_POSITIVE = 1,      /**< 正面交互(因果支持) */
    TASK_INTERACTION_NEGATIVE = 2,      /**< 负面交互(因果冲突) */
    TASK_INTERACTION_RESOURCE = 3,      /**< 资源冲突 */
    TASK_INTERACTION_ORDERING = 4,      /**< 排序约束冲突 */
    TASK_INTERACTION_MUTEX = 5          /**< 互斥约束 */
} TaskInteractionType;

/**
 * @brief 任务交互描述
 */
typedef struct {
    int from_task;                      /**< 来源任务索引 */
    int to_task;                        /**< 目标任务索引 */
    TaskInteractionType type;           /**< 交互类型 */
    float strength;                     /**< 交互强度(0-1) */
    float* resolution_action;           /**< 解决方案 */
    size_t resolution_size;             /**< 解决方案大小 */
    char description[128];              /**< 交互描述 */
} TaskInteraction;

/**
 * @brief HTN分解结果
 */
typedef struct {
    int total_tasks;                    /**< 总任务数 */
    int primitive_count;                /**< 原始任务数 */
    int compound_count;                 /**< 复合任务数 */
    float avg_decomposition_depth;      /**< 平均分解深度 */
    float total_cost;                   /**< 总成本估计 */
    float total_duration;               /**< 总时长估计 */
    float feasibility_score;            /**< 可行性评分 */
    TaskInteraction* interactions;       /**< 检测到的交互 */
    int interaction_count;              /**< 交互数量 */
} HTNDecompositionResult;

/**
 * @brief 创建HTN规划域
 *
 * @param domain 规划域输出
 * @param max_methods 最大方法数
 * @param max_operators 最大操作符数
 * @param max_resources 最大资源数
 * @return int 成功返回0,失败返回-1
 */
int htn_domain_create(HTNPlanningDomain* domain, size_t max_methods,
                      size_t max_operators, size_t max_resources);

/**
 * @brief 销毁HTN规划域
 *
 * @param domain 规划域句柄
 */
void htn_domain_destroy(HTNPlanningDomain* domain);

/**
 * @brief 添加复合任务方法到规划域
 *
 * @param domain 规划域句柄
 * @param method 方法描述
 * @return int 成功返回方法索引,失败返回-1
 */
int htn_add_method(HTNPlanningDomain* domain, const HTNMethod* method);

/**
 * @brief 添加原始操作符到规划域
 *
 * @param domain 规划域句柄
 * @param op 操作符描述
 * @return int 成功返回操作符索引,失败返回-1
 */
int htn_add_operator(HTNPlanningDomain* domain, const HTNOperator* op);

/**
 * @brief 添加资源到规划域
 *
 * @param domain 规划域句柄
 * @param resource 资源描述
 * @return int 成功返回资源索引,失败返回-1
 */
int htn_add_resource(HTNPlanningDomain* domain, const HTNResource* resource);

/**
 * @brief 检测任务网络中所有交互
 *
 * 分析任务网络中所有任务之间的交互关系，包括:
 * - 因果支持(一个任务的效果提供另一个任务的前提条件)
 * - 因果冲突(一个任务的效果破坏另一个任务的前提条件)
 * - 资源冲突(多个任务竞争同一资源)
 * - 排序约束冲突(现有排序违反约束)
 *
 * @param system 分层规划系统句柄
 * @param interactions 交互输出缓冲区
 * @param max_interactions 最大交互数
 * @return int 成功返回实际交互数,失败返回-1
 */
int htn_detect_task_interactions(HierarchicalPlanningSystem* system,
                                 TaskInteraction* interactions,
                                 size_t max_interactions);

/**
 * @brief 解决资源冲突
 *
 * 通过调整任务排序或分配替代资源解决资源冲突。
 *
 * @param system 分层规划系统句柄
 * @param interactions 已检测到的交互
 * @param num_interactions 交互数量
 * @return int 成功解决的冲突数,失败返回-1
 */
int htn_resolve_resource_conflicts(HierarchicalPlanningSystem* system,
                                   const TaskInteraction* interactions,
                                   size_t num_interactions);

/**
 * @brief 设置方法选择器配置
 *
 * @param system 分层规划系统句柄
 * @param config 方法选择器配置
 * @return int 成功返回0,失败返回-1
 */
int htn_set_method_selector_config(HierarchicalPlanningSystem* system,
                                   const HTNMethodSelectorConfig* config);

/**
 * @brief 获取方法选择器配置
 *
 * @param system 分层规划系统句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0,失败返回-1
 */
int htn_get_method_selector_config(const HierarchicalPlanningSystem* system,
                                   HTNMethodSelectorConfig* config);

/**
 * @brief 获取HTN规划域
 *
 * @param system 分层规划系统句柄
 * @param domain 规划域输出缓冲区
 * @return int 成功返回0,失败返回-1
 */
int htn_get_domain(const HierarchicalPlanningSystem* system,
                   HTNPlanningDomain* domain);

/**
 * @brief 获取上次HTN分解结果
 *
 * @param system 分层规划系统句柄
 * @param result 分解结果输出缓冲区
 * @return int 成功返回0,失败返回-1
 */
int htn_get_decomposition_result(const HierarchicalPlanningSystem* system,
                                 HTNDecompositionResult* result);

/**
 * @brief 启用/禁用增强HTN规划
 *
 * @param system 分层规划系统句柄
 * @param enable 1=启用增强HTN,0=使用默认HTN
 * @return int 成功返回0,失败返回-1
 */
int htn_set_enhanced_enabled(HierarchicalPlanningSystem* system, int enable);

/**
 * @brief 增强HTN分解(使用方法库)
 *
 * 使用方法库中的复合任务方法进行智能分解:
 * 1. 匹配任务签名找到适用方法
 * 2. 使用方法选择器(多目标加权)选择最优方法
 * 3. 递归分解所有非原始子任务
 * 4. 检测并解决任务交互和资源冲突
 * 5. 生成优化的部分有序任务网络
 *
 * @param system 分层规划系统句柄
 * @param goal 目标描述
 * @param goal_size 目标描述大小
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @param plan 规划输出缓冲区
 * @param max_plan_size 规划最大大小
 * @return int 成功返回规划步骤数,失败返回-1
 */
int htn_enhanced_decompose(HierarchicalPlanningSystem* system,
                           const float* goal, size_t goal_size,
                           const float* current_state, size_t state_size,
                           float* plan, size_t max_plan_size);

/**
 * @brief 初始化默认HTN方法库
 *
 * 根据当前系统配置初始化一组默认的复合任务方法:
 * - 战略层: 长期目标分解(2-3个战术目标)
 * - 战术层: 中期计划分解(3-5个操作步骤)
 * - 操作层: 短期行动分解(2-4个执行步骤)
 * - 资源管理方法
 * - 并行执行方法
 *
 * @param system 分层规划系统句柄
 * @return int 成功返回方法数,失败返回-1
 */
int htn_init_default_method_library(HierarchicalPlanningSystem* system);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_HIERARCHICAL_PLANNING_H */