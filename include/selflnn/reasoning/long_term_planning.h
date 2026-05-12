/**
 * @file long_term_planning.h
 * @brief 长期规划和目标分解优化系统
 * 
 * 长期规划系统，支持目标分解、时序规划、资源优化和自适应重规划。
 * 提供分层目标分解（HGD）、时序逻辑规划（TLP）、资源受限项目调度（RCPS）等算法。
 */

#ifndef SELFLNN_REASONING_LONG_TERM_PLANNING_H
#define SELFLNN_REASONING_LONG_TERM_PLANNING_H

#include "selflnn/reasoning/planning.h"
#include "selflnn/core/cfc_network.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 时间范围
 */
typedef enum {
    TIME_HORIZON_SHORT = 0,       /**< 短期（小时/天） */
    TIME_HORIZON_MEDIUM,          /**< 中期（周/月） */
    TIME_HORIZON_LONG,            /**< 长期（年/多年） */
    TIME_HORIZON_STRATEGIC        /**< 战略期（多年/十年） */
} TimeHorizon;

/**
 * @brief 目标类型
 */
typedef enum {
    GOAL_TYPE_ACHIEVEMENT = 0,    /**< 达成型目标 */
    GOAL_TYPE_MAINTENANCE,        /**< 维持型目标 */
    GOAL_TYPE_AVOIDANCE,          /**< 避免型目标 */
    GOAL_TYPE_OPTIMIZATION,       /**< 优化型目标 */
    GOAL_TYPE_EXPLORATION         /**< 探索型目标 */
} GoalType;

/**
 * @brief 目标优先级
 */
typedef enum {
    GOAL_PRIORITY_CRITICAL = 0,   /**< 关键（必须完成） */
    GOAL_PRIORITY_HIGH,           /**< 高优先级 */
    GOAL_PRIORITY_MEDIUM,         /**< 中优先级 */
    GOAL_PRIORITY_LOW,            /**< 低优先级 */
    GOAL_PRIORITY_OPTIONAL        /**< 可选（可放弃） */
} GoalPriority;

/**
 * @brief 目标状态
 */
typedef enum {
    GOAL_STATE_PENDING = 0,       /**< 待处理 */
    GOAL_STATE_ACTIVE,            /**< 进行中 */
    GOAL_STATE_SUSPENDED,         /**< 已暂停 */
    GOAL_STATE_COMPLETED,         /**< 已完成 */
    GOAL_STATE_FAILED,            /**< 已失败 */
    GOAL_STATE_ABORTED            /**< 已中止 */
} GoalState;

/**
 * @brief 目标
 */
typedef struct {
    char* goal_id;                /**< 目标ID */
    char* description;            /**< 目标描述 */
    GoalType goal_type;           /**< 目标类型 */
    GoalPriority priority;        /**< 目标优先级 */
    GoalState state;              /**< 目标状态 */
    
    // 时间约束
    TimeHorizon horizon;          /**< 时间范围 */
    float start_time;             /**< 开始时间 */
    float deadline;               /**< 截止时间 */
    float estimated_duration;     /**< 预计持续时间 */
    
    // 成功条件
    float success_threshold;      /**< 成功阈值（0-1） */
    float progress;               /**< 当前进度（0-1） */
    
    // 资源需求
    float required_resources[8];  /**< 所需资源（CPU、内存、能源等） */
    int resource_count;           /**< 资源类型数量 */
    
    // 依赖关系
    char** prerequisite_goals;    /**< 先决条件目标ID数组 */
    int prerequisite_count;       /**< 先决条件数量 */
    
    // 子目标
    char** subgoal_ids;           /**< 子目标ID数组 */
    int subgoal_count;            /**< 子目标数量 */
    
    // 评估指标
    float difficulty;             /**< 难度（0-1） */
    float importance;             /**< 重要性（0-1） */
    float urgency;                /**< 紧急程度（0-1） */
    
    // 历史记录
    float creation_time;          /**< 创建时间 */
    float last_update_time;       /**< 最后更新时间 */
    int update_count;             /**< 更新次数 */
} Goal;

/**
 * @brief 冲突解决策略
 */
typedef enum {
    CONFLICT_RESOLUTION_RESCHEDULE = 0,      /**< 重新调度：调整动作时间 */
    CONFLICT_RESOLUTION_REALLOCATE_RESOURCES, /**< 重新分配资源 */
    CONFLICT_RESOLUTION_PRIORITIZE,          /**< 优先级处理：优先执行高优先级动作 */
    CONFLICT_RESOLUTION_PARALLELIZE,         /**< 并行化：允许动作并行执行 */
    CONFLICT_RESOLUTION_SKIP,                /**< 跳过：跳过低优先级动作 */
    CONFLICT_RESOLUTION_ADAPT                /**< 自适应：动态调整计划 */
} ConflictResolutionStrategy;

/**
 * @brief 规划动作
 */
typedef struct {
    char* action_id;              /**< 动作ID */
    char* description;            /**< 动作描述 */
    
    // 时间参数
    float start_time;             /**< 开始时间 */
    float duration;               /**< 持续时间 */
    float end_time;               /**< 结束时间（计算得出） */
    
    // 资源消耗
    float resource_consumption[8]; /**< 资源消耗 */
    float resource_production[8];  /**< 资源产出 */
    
    // 前置条件
    char** preconditions;         /**< 前置条件描述数组 */
    int precondition_count;       /**< 前置条件数量 */
    
    // 效果
    char** effects;               /**< 效果描述数组 */
    int effect_count;             /**< 效果数量 */
    
    // 执行状态
    int executed;                 /**< 是否已执行 */
    float execution_progress;     /**< 执行进度（0-1） */
    float success_probability;    /**< 成功概率（0-1） */
    
    // 关联目标
    char* associated_goal;        /**< 关联目标ID */
    
    // 依赖管理
    char** dependencies;          /**< 依赖动作ID数组 */
    int dependency_count;         /**< 依赖数量 */
    char* action_name;            /**< 动作名称 */
} PlanAction;

/**
 * @brief 长期规划配置
 */
typedef struct {
    // 时间设置
    TimeHorizon default_horizon;  /**< 默认时间范围 */
    float time_resolution;        /**< 时间分辨率（小时） */
    float max_planning_time;      /**< 最大规划时间（秒） */
    
    // 优化目标
    float weight_completion_time; /**< 完成时间权重 */
    float weight_resource_usage;  /**< 资源使用权重 */
    float weight_goal_achievement; /**< 目标达成权重 */
    float weight_risk_minimization; /**< 风险最小化权重 */
    
    // 分解设置
    int max_decomposition_depth;  /**< 最大分解深度 */
    float decomposition_threshold; /**< 分解阈值 */
    int enable_adaptive_decomposition; /**< 启用自适应分解 */
    
    // 重规划设置
    int enable_replanning;        /**< 启用重规划 */
    float replanning_interval;    /**< 重规划间隔（小时） */
    float replanning_threshold;   /**< 重规划阈值 */
    
    // 资源约束
    float available_resources[8]; /**< 可用资源 */
    int resource_type_count;      /**< 资源类型数量 */
    
    // 不确定性处理
    float uncertainty_tolerance;  /**< 不确定性容忍度 */
    int enable_contingency_planning; /**< 启用应急规划 */
    int contingency_branch_count; /**< 应急分支数量 */
} LongTermPlanningConfig;

/**
 * @brief 长期规划系统
 */
typedef struct LongTermPlanningSystem LongTermPlanningSystem;

/**
 * @brief 创建长期规划系统
 * 
 * @param config 规划配置
 * @return LongTermPlanningSystem* 规划系统指针，失败返回NULL
 */
LongTermPlanningSystem* long_term_planning_system_create(
    const LongTermPlanningConfig* config);

/**
 * @brief 销毁长期规划系统
 * 
 * @param system 规划系统
 */
void long_term_planning_system_destroy(LongTermPlanningSystem* system);

/**
 * @brief 添加目标到规划系统
 * 
 * @param system 规划系统
 * @param goal 目标
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_add_goal(LongTermPlanningSystem* system, const Goal* goal);

/**
 * @brief 从规划系统移除目标
 * 
 * @param system 规划系统
 * @param goal_id 目标ID
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_remove_goal(LongTermPlanningSystem* system, const char* goal_id);

/**
 * @brief 目标分解（分层目标分解）
 * 
 * @param system 规划系统
 * @param goal_id 目标ID
 * @param max_depth 最大分解深度
 * @return char** 子目标ID数组，失败返回NULL
 */
char** long_term_planning_decompose_goal(LongTermPlanningSystem* system,
                                        const char* goal_id, int max_depth);

/**
 * @brief 生成计划
 * 
 * @param system 规划系统
 * @param goal_ids 目标ID数组
 * @param goal_count 目标数量
 * @param plan_actions 计划动作输出数组
 * @param max_actions 最大动作数
 * @return int 生成的计划动作数量，失败返回-1
 */
int long_term_planning_generate_plan(LongTermPlanningSystem* system,
                                    const char** goal_ids, int goal_count,
                                    PlanAction** plan_actions, int max_actions);

/**
 * @brief 优化计划
 * 
 * @param system 规划系统
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param optimized_actions 优化后动作输出数组
 * @return int 优化后动作数量，失败返回-1
 */
int long_term_planning_optimize_plan(LongTermPlanningSystem* system,
                                     PlanAction** plan_actions, int action_count,
                                     PlanAction*** optimized_actions);

/**
 * @brief 执行计划
 * 
 * @param system 规划系统
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param progress_callback 进度回调函数
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_execute_plan(LongTermPlanningSystem* system,
                                   PlanAction** plan_actions, int action_count,
                                   void (*progress_callback)(float));

/**
 * @brief 监控计划执行
 * 
 * @param system 规划系统
 * @param goal_id 目标ID
 * @param progress 进度输出（0-1）
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_monitor_progress(LongTermPlanningSystem* system,
                                       const char* goal_id, float* progress);

/**
 * @brief 评估计划可行性
 * 
 * @param system 规划系统
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @return float 可行性分数（0-1），失败返回-1
 */
float long_term_planning_evaluate_feasibility(LongTermPlanningSystem* system,
                                             PlanAction** plan_actions,
                                             int action_count);

/**
 * @brief 重规划（适应环境变化）
 * 
 * @param system 规划系统
 * @param changed_conditions 变化的条件数组
 * @param condition_count 条件数量
 * @param updated_plan 更新后的计划输出
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_replan(LongTermPlanningSystem* system,
                             const char** changed_conditions, int condition_count,
                             PlanAction*** updated_plan);

/**
 * @brief 计算资源需求
 * 
 * @param system 规划系统
 * @param goal_id 目标ID
 * @param resource_needs 资源需求输出数组
 * @param resource_count 资源类型数量
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_calculate_resource_needs(LongTermPlanningSystem* system,
                                               const char* goal_id,
                                               float* resource_needs,
                                               int resource_count);

/**
 * @brief 检测冲突
 * 
 * @param system 规划系统
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param conflicts 冲突输出数组
 * @param max_conflicts 最大冲突数
 * @return int 检测到的冲突数量，失败返回-1
 */
int long_term_planning_detect_conflicts(LongTermPlanningSystem* system,
                                       PlanAction** plan_actions, int action_count,
                                       int* conflicts, int max_conflicts);

/**
 * @brief 解决冲突
 * 
 * @param system 规划系统
 * @param conflicts 冲突数组
 * @param conflict_count 冲突数量
 * @param resolution_strategy 解决策略
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_resolve_conflicts(LongTermPlanningSystem* system,
                                        int* conflicts, int conflict_count,
                                        int resolution_strategy);

/**
 * @brief 获取默认长期规划配置
 * 
 * @param config 配置输出
 */
void long_term_planning_default_config(LongTermPlanningConfig* config);

/**
 * @brief 创建目标
 * 
 * @param goal_id 目标ID
 * @param description 目标描述
 * @param goal_type 目标类型
 * @return Goal* 目标指针，失败返回NULL
 */
Goal* goal_create(const char* goal_id, const char* description, GoalType goal_type);

/**
 * @brief 销毁目标
 * 
 * @param goal 目标
 */
void goal_destroy(Goal* goal);

/**
 * @brief 创建计划动作
 * 
 * @param action_id 动作ID
 * @param description 动作描述
 * @param duration 持续时间
 * @return PlanAction* 计划动作指针，失败返回NULL
 */
PlanAction* plan_action_create(const char* action_id, const char* description,
                              float duration);

/**
 * @brief 销毁计划动作
 * 
 * @param action 计划动作
 */
void plan_action_destroy(PlanAction* action);

/**
 * @brief 计算目标关键路径
 * 
 * @param system 规划系统
 * @param goal_id 目标ID
 * @param critical_path 关键路径输出数组
 * @param max_path_length 最大路径长度
 * @return int 关键路径长度，失败返回-1
 */
int long_term_planning_calculate_critical_path(LongTermPlanningSystem* system,
                                              const char* goal_id,
                                              char** critical_path,
                                              int max_path_length);

/**
 * @brief 评估目标风险
 * 
 * @param system 规划系统
 * @param goal_id 目标ID
 * @return float 风险分数（0-1），失败返回-1
 */
float long_term_planning_assess_risk(LongTermPlanningSystem* system,
                                    const char* goal_id);

/**
 * @brief 生成应急计划
 * 
 * @param system 规划系统
 * @param goal_id 目标ID
 * @param risk_factor 风险因素
 * @param contingency_plan 应急计划输出
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_generate_contingency_plan(LongTermPlanningSystem* system,
                                                const char* goal_id,
                                                float risk_factor,
                                                PlanAction*** contingency_plan);

/* ============================================================================
 * Allen时间区间代数
 * =========================================================================== */

/**
 * @brief Allen时间区间关系枚举（13种）
 */
typedef enum {
    ALLEN_BEFORE = 0,        /**< X < Y: X在Y之前 */
    ALLEN_AFTER,             /**< Y < X: X在Y之后 */
    ALLEN_MEETS,             /**< X m Y: X结束时刻等于Y开始时刻 */
    ALLEN_MET_BY,            /**< Y m X: X被Y相接 */
    ALLEN_OVERLAPS,          /**< X o Y: X与Y重叠 */
    ALLEN_OVERLAPPED_BY,     /**< Y o X: X被Y重叠 */
    ALLEN_STARTS,            /**< X s Y: X与Y同时开始 */
    ALLEN_STARTED_BY,        /**< Y s X: X被Y开始 */
    ALLEN_DURING,            /**< X d Y: X在Y期间内 */
    ALLEN_CONTAINS,          /**< Y d X: X包含Y */
    ALLEN_FINISHES,          /**< X f Y: X与Y同时结束 */
    ALLEN_FINISHED_BY,       /**< Y f X: X被Y结束 */
    ALLEN_EQUALS             /**< X = Y: X与Y时间相等 */
} AllenRelation;

/**
 * @brief 时间区间
 */
typedef struct {
    float start;             /**< 开始时间 */
    float end;               /**< 结束时间 */
} TimeInterval;

/**
 * @brief Allen约束
 */
typedef struct {
    AllenRelation relation;  /**< 时间关系 */
    char* interval_a_id;     /**< 区间A的ID */
    char* interval_b_id;     /**< 区间B的ID */
    TimeInterval interval_a; /**< 区间A */
    TimeInterval interval_b; /**< 区间B */
    float confidence;        /**< 置信度（0-1） */
} AllenConstraint;

/**
 * @brief 判断两个时间区间之间的Allen关系
 * 
 * @param a 区间A
 * @param b 区间B
 * @return AllenRelation Allen关系
 */
AllenRelation allen_calculate_relation(const TimeInterval* a, const TimeInterval* b);

/**
 * @brief 获取Allen关系的逆关系
 * 
 * @param relation 原关系
 * @return AllenRelation 逆关系
 */
AllenRelation allen_inverse_relation(AllenRelation relation);

/**
 * @brief 判断两个Allen关系是否兼容（可以通过组合推导）
 * 
 * @param r1 关系1
 * @param r2 关系2
 * @return int 兼容返回1，否则返回0
 */
int allen_relations_composable(AllenRelation r1, AllenRelation r2);

/**
 * @brief 组合两个Allen关系（Allen组合表）
 * 
 * @param r1 关系1
 * @param r2 关系2
 * @param result 组合结果集合（位掩码）
 */
void allen_compose_relations(AllenRelation r1, AllenRelation r2, unsigned int* result);

/**
 * @brief 添加Allen时间约束到规划系统
 * 
 * @param system 规划系统
 * @param constraint Allen约束
 * @return int 成功返回0，失败返回-1
 */
int long_term_planning_add_allen_constraint(LongTermPlanningSystem* system,
                                           const AllenConstraint* constraint);

/**
 * @brief 检查计划中的Allen约束一致性
 * 
 * @param system 规划系统
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param violations 违规约束索引输出数组
 * @param max_violations 最大违规数
 * @return int 违规数量，0表示全部通过
 */
int long_term_planning_check_allen_consistency(LongTermPlanningSystem* system,
                                              PlanAction** plan_actions,
                                              int action_count,
                                              int* violations,
                                              int max_violations);

/* ============================================================================
 * LTL时序逻辑
 * =========================================================================== */

/**
 * @brief LTL时序算子
 */
typedef enum {
    LTL_GLOBALLY = 0,        /**< G φ: 全局成立（Always/Globally） */
    LTL_FINALLY,             /**< F φ: 最终成立（Finally/Eventually） */
    LTL_NEXT,                /**< X φ: 下一步成立（NeXt） */
    LTL_UNTIL,               /**< φ U ψ: φ成立直到ψ成立（Until） */
    LTL_RELEASE,             /**< φ R ψ: φ释放ψ（Release：ψ成立直到φ成立，包含φ永不成立时） */
    LTL_WEAK_UNTIL           /**< φ W ψ: 弱直到（Weak Until：ψ可能永不成立） */
} LTLOperator;

/**
 * @brief LTL原子命题判断函数类型
 * 
 * @param state_index 状态索引
 * @param context 上下文数据
 * @return int 命题成立返回1，否则返回0
 */
typedef int (*LTLPropositionFunc)(int state_index, void* context);

/**
 * @brief LTL公式节点
 */
typedef struct LTLFormulaNode {
    LTLOperator op;                  /**< 算子类型 */
    int is_atomic;                   /**< 是否为原子命题 */
    LTLPropositionFunc prop_func;    /**< 原子命题判断函数 */
    void* prop_context;              /**< 原子命题上下文 */
    struct LTLFormulaNode* left;     /**< 左子公式 */
    struct LTLFormulaNode* right;    /**< 右子公式 */
    float time_bound;                /**< 时间边界（可选，≤0表示无边界） */
} LTLFormula;

/**
 * @brief LTL模型检查结果
 */
typedef struct {
    int satisfied;                   /**< 是否满足 */
    int violating_state;             /**< 违规状态索引（-1表示无） */
    float satisfaction_degree;       /**< 满足程度（0-1） */
    char description[256];           /**< 结果描述 */
} LTLResult;

/**
 * @brief 创建LTL原子命题公式
 * 
 * @param prop_func 命题判断函数
 * @param context 上下文
 * @return LTLFormula* 公式指针，失败返回NULL
 */
LTLFormula* ltl_create_atomic(LTLPropositionFunc prop_func, void* context);

/**
 * @brief 创建LTL一元算子公式（G、F、X）
 * 
 * @param op 算子类型（LTL_GLOBALLY、LTL_FINALLY、LTL_NEXT）
 * @param operand 操作数公式
 * @return LTLFormula* 公式指针，失败返回NULL
 */
LTLFormula* ltl_create_unary(LTLOperator op, LTLFormula* operand);

/**
 * @brief 创建LTL二元算子公式（U、R、W）
 * 
 * @param op 算子类型（LTL_UNTIL、LTL_RELEASE、LTL_WEAK_UNTIL）
 * @param left 左操作数
 * @param right 右操作数
 * @return LTLFormula* 公式指针，失败返回NULL
 */
LTLFormula* ltl_create_binary(LTLOperator op, LTLFormula* left, LTLFormula* right);

/**
 * @brief 销毁LTL公式
 * 
 * @param formula 公式
 */
void ltl_destroy(LTLFormula* formula);

/**
 * @brief 在状态序列上检查LTL公式
 * 
 * @param formula LTL公式
 * @param state_count 状态数量
 * @param context 上下文
 * @return LTLResult 检查结果
 */
LTLResult ltl_model_check(const LTLFormula* formula, int state_count, void* context);

/**
 * @brief 在状态序列上检查有界LTL公式（Bounded Model Checking）
 * 
 * @param formula LTL公式
 * @param state_count 状态数量
 * @param bound 时间边界
 * @param context 上下文
 * @return LTLResult 检查结果
 */
LTLResult ltl_bounded_model_check(const LTLFormula* formula, int state_count,
                                  int bound, void* context);

/**
 * @brief 将LTL公式转换为Büchi自动机（用于模型检查）
 * 
 * @param formula LTL公式
 * @param state_count 自动机状态数量输出
 * @return int** 转移表，失败返回NULL
 */
int** ltl_to_buchi(const LTLFormula* formula, int* state_count);

/**
 * @brief 在计划上检查LTL时序属性
 * 
 * @param system 规划系统
 * @param formula LTL公式
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @return LTLResult 检查结果
 */
LTLResult long_term_planning_check_ltl_property(LongTermPlanningSystem* system,
                                               const LTLFormula* formula,
                                               PlanAction** plan_actions,
                                               int action_count);

/* ============================================================================
 * CTL (Computation Tree Logic) 模型检查
 * =========================================================================== */

/**
 * @brief CTL时序算子（分支时间逻辑）
 */
typedef enum {
    CTL_AG = 0,        /**< AG φ: 所有路径上全局成立 */
    CTL_EG,            /**< EG φ: 存在一条路径全局成立 */
    CTL_AF,            /**< AF φ: 所有路径上最终成立 */
    CTL_EF,            /**< EF φ: 存在一条路径最终成立 */
    CTL_AX,            /**< AX φ: 所有路径的下一步成立 */
    CTL_EX,            /**< EX φ: 存在一条路径的下一步成立 */
    CTL_AU,            /**< A[φ U ψ]: 所有路径上φ成立直到ψ成立 */
    CTL_EU             /**< E[φ U ψ]: 存在一条路径φ成立直到ψ成立 */
} CTLOperator;

/**
 * @brief Kripke结构状态
 */
typedef struct {
    int state_id;                   /**< 状态ID */
    int* propositions;              /**< 原子命题标签（位掩码数组） */
    int prop_count;                 /**< 命题数量 */
    int* transitions;               /**< 转移目标状态索引数组 */
    int transition_count;           /**< 转移数量 */
    void* user_data;                /**< 用户自定义数据 */
} KripkeState;

/**
 * @brief Kripke结构（用于CTL模型检查）
 */
typedef struct {
    KripkeState* states;            /**< 状态数组 */
    int state_count;                /**< 状态数量 */
    int proposition_count;          /**< 原子命题总数 */
    char** proposition_names;       /**< 命题名称数组 */
    int** transition_matrix;        /**< 转移矩阵 [state_count][state_count], 1=有转移 */
} KripkeStructure;

/**
 * @brief CTL公式节点
 */
typedef struct CTLFormulaNode {
    CTLOperator op;                  /**< 算子类型 */
    int is_atomic;                   /**< 是否为原子命题 */
    int atomic_prop_index;           /**< 原子命题索引 */
    struct CTLFormulaNode* left;     /**< 左子公式 */
    struct CTLFormulaNode* right;    /**< 右子公式 */
    float time_bound;                /**< 时间边界（≤0表示无边界） */
} CTLFormulaNode;

/**
 * @brief CTL模型检查结果
 */
typedef struct {
    int satisfied;                   /**< 是否满足 */
    int* satisfying_states;          /**< 满足公式的状态索引数组 */
    int satisfying_count;            /**< 满足状态数量 */
    float satisfaction_degree;       /**< 满足程度（0-1） */
    char description[256];           /**< 结果描述 */
    int* counterexample_path;        /**< 反例路径状态索引数组 */
    int counterexample_length;       /**< 反例路径长度 */
} CTLResult;

/**
 * @brief 创建CTL原子命题公式
 *
 * @param prop_index 原子命题索引
 * @return CTLFormulaNode* 公式指针，失败返回NULL
 */
CTLFormulaNode* ctl_create_atomic(int prop_index);

/**
 * @brief 创建CTL一元算子公式（AG、EG、AF、EF、AX、EX）
 *
 * @param op 算子类型
 * @param operand 操作数公式
 * @return CTLFormulaNode* 公式指针，失败返回NULL
 */
CTLFormulaNode* ctl_create_unary(CTLOperator op, CTLFormulaNode* operand);

/**
 * @brief 创建CTL二元算子公式（AU、EU）
 *
 * @param op 算子类型（CTL_AU或CTL_EU）
 * @param left 左操作数
 * @param right 右操作数
 * @return CTLFormulaNode* 公式指针，失败返回NULL
 */
CTLFormulaNode* ctl_create_binary(CTLOperator op, CTLFormulaNode* left, CTLFormulaNode* right);

/**
 * @brief 销毁CTL公式树
 *
 * @param formula 公式指针
 */
void ctl_destroy_formula(CTLFormulaNode* formula);

/**
 * @brief 创建Kripke结构
 *
 * @param state_count 状态数量
 * @param proposition_count 原子命题数量
 * @return KripkeStructure* Kripke结构指针，失败返回NULL
 */
KripkeStructure* ctl_create_kripke(int state_count, int proposition_count);

/**
 * @brief 销毁Kripke结构
 *
 * @param ks Kripke结构指针
 */
void ctl_destroy_kripke(KripkeStructure* ks);

/**
 * @brief 添加Kripke状态转移
 *
 * @param ks Kripke结构
 * @param from 源状态索引
 * @param to 目标状态索引
 * @return int 成功返回0，失败返回-1
 */
int ctl_add_transition(KripkeStructure* ks, int from, int to);

/**
 * @brief 设置Kripke状态的原子命题标签
 *
 * @param ks Kripke结构
 * @param state_idx 状态索引
 * @param prop_idx 命题索引
 * @param value 命题值（0或1）
 * @return int 成功返回0，失败返回-1
 */
int ctl_set_proposition(KripkeStructure* ks, int state_idx, int prop_idx, int value);

/**
 * @brief 执行CTL模型检查
 *
 * 使用递归标记算法（labeling algorithm）计算所有满足CTL公式的状态。
 * 对于AG/EG使用最大不动点语义，AF/EF/EU使用最小不动点语义。
 *
 * @param ks Kripke结构
 * @param formula CTL公式
 * @return CTLResult 检查结果
 */
CTLResult ctl_model_check(const KripkeStructure* ks, const CTLFormulaNode* formula);

/**
 * @brief 从计划构建Kripke结构
 *
 * 将计划动作序列转换为Kripke结构，每个时间步对应一个状态，
 * 动作执行效果对应原子命题标签。
 *
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param proposition_names 命题名称数组
 * @param prop_name_count 命题名称数量
 * @return KripkeStructure* Kripke结构指针，失败返回NULL
 */
KripkeStructure* ctl_build_kripke_from_plan(PlanAction** plan_actions,
                                           int action_count,
                                           const char** proposition_names,
                                           int prop_name_count);

/**
 * @brief 在计划上检查CTL时序属性
 *
 * @param system 规划系统
 * @param formula CTL公式
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @return CTLResult 检查结果
 */
CTLResult long_term_planning_check_ctl_property(LongTermPlanningSystem* system,
                                               const CTLFormulaNode* formula,
                                               PlanAction** plan_actions,
                                               int action_count);

/* ============================================================================
 * 鲁棒性分析
 * =========================================================================== */

/**
 * @brief 鲁棒性指标类型
 */
typedef enum {
    ROBUSTNESS_TEMPORAL = 0,       /**< 时间鲁棒性：时间偏移容忍度 */
    ROBUSTNESS_RESOURCE,           /**< 资源鲁棒性：资源波动容忍度 */
    ROBUSTNESS_STRUCTURAL,         /**< 结构鲁棒性：动作顺序偏差容忍度 */
    ROBUSTNESS_PARAMETRIC          /**< 参数鲁棒性：参数变化容忍度 */
} RobustnessType;

/**
 * @brief 时间松弛度
 */
typedef struct {
    float total_slack;             /**< 总松弛时间 */
    float* action_slacks;          /**< 每个动作的松弛时间 */
    int action_count;              /**< 动作数量 */
    float critical_path_duration;  /**< 关键路径总时长 */
    float* earliest_start;         /**< 最早开始时间 */
    float* latest_start;           /**< 最晚开始时间 */
    float* earliest_finish;        /**< 最早完成时间 */
    float* latest_finish;          /**< 最晚完成时间 */
    int* on_critical_path;         /**< 是否在关键路径上 */
} TemporalSlack;

/**
 * @brief 资源松弛度
 */
typedef struct {
    float* resource_headroom;      /**< 每种资源的余量 */
    int resource_count;            /**< 资源类型数量 */
    float* peak_demands;           /**< 峰值需求 */
    float* avg_demands;            /**< 平均需求 */
    float* utilizations;           /**< 利用率 */
    float min_headroom;            /**< 最小资源余量 */
    int bottleneck_resource;       /**< 瓶颈资源索引 */
} ResourceSlack;

/**
 * @brief 敏感性分析结果
 */
typedef struct {
    float** sensitivity_matrix;    /**< 敏感性矩阵 [param_count][metric_count] */
    int param_count;               /**< 参数数量 */
    int metric_count;              /**< 指标数量 */
    char** param_names;            /**< 参数名称 */
    char** metric_names;           /**< 指标名称 */
    float* critical_params;        /**< 关键参数排序（最敏感优先） */
    int* critical_param_indices;   /**< 关键参数索引 */
    int critical_param_count;      /**< 关键参数数量 */
} SensitivityResult;

/**
 * @brief 蒙特卡洛鲁棒性估计结果
 */
typedef struct {
    float success_probability;     /**< 成功概率 */
    float mean_duration;           /**< 平均持续时长 */
    float std_duration;            /**< 持续时长标准差 */
    float mean_cost;               /**< 平均成本 */
    float std_cost;                /**< 成本标准差 */
    float* percentile_durations;   /**< 分位时长 [5, 25, 50, 75, 95] */
    int percentile_count;          /**< 分位数量（固定5） */
    float risk_value;              /**< 风险值（失败率×损失） */
    int total_simulations;         /**< 总模拟次数 */
    int success_count;             /**< 成功次数 */
} MonteCarloResult;

/**
 * @brief 完整鲁棒性分析结果
 */
typedef struct {
    float overall_robustness;      /**< 整体鲁棒性分数（0-1） */
    TemporalSlack* temporal;       /**< 时间松弛度 */
    ResourceSlack* resource;       /**< 资源松弛度 */
    SensitivityResult* sensitivity; /**< 敏感性分析 */
    MonteCarloResult* monte_carlo; /**< 蒙特卡洛估计 */
    char** recommendations;        /**< 改进建议数组 */
    int recommendation_count;      /**< 建议数量 */
} RobustnessResult;

/**
 * @brief 分析计划时间松弛度（关键路径法）
 *
 * 计算每个动作的最早/最晚开始/完成时间及总松弛时间。
 * 使用前向/后向传递算法。
 *
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param dependencies 依赖矩阵 [action_count][action_count]，1=依赖
 * @return TemporalSlack* 时间松弛度结果，失败返回NULL
 */
TemporalSlack* plan_robustness_temporal_slack(PlanAction** plan_actions,
                                             int action_count,
                                             const int* dependencies);

/**
 * @brief 释放时间松弛度
 *
 * @param slack 时间松弛度指针
 */
void plan_robustness_destroy_temporal_slack(TemporalSlack* slack);

/**
 * @brief 分析计划资源松弛度
 *
 * 计算每种资源的余量、峰值需求、利用率和瓶颈资源。
 *
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param available_resources 可用资源数组
 * @param resource_count 资源类型数量
 * @return ResourceSlack* 资源松弛度结果，失败返回NULL
 */
ResourceSlack* plan_robustness_resource_slack(PlanAction** plan_actions,
                                             int action_count,
                                             const float* available_resources,
                                             int resource_count);

/**
 * @brief 释放资源松弛度
 *
 * @param slack 资源松弛度指针
 */
void plan_robustness_destroy_resource_slack(ResourceSlack* slack);

/**
 * @brief 执行敏感性分析
 *
 * 对计划参数进行微小扰动，观察对关键指标的影响。
 * 使用有限差分法计算敏感性梯度。
 *
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param param_names 参数名称数组
 * @param param_count 参数数量
 * @param metric_names 指标名称数组
 * @param metric_count 指标数量
 * @param base_values 参数基值数组
 * @param perturbations 扰动幅度数组
 * @param evaluate_func 评估函数：输入(参数值数组, 参数数量)，输出(指标值数组, 指标数量)
 * @return SensitivityResult* 敏感性分析结果，失败返回NULL
 */
SensitivityResult* plan_robustness_sensitivity_analysis(
    PlanAction** plan_actions,
    int action_count,
    const char** param_names, int param_count,
    const char** metric_names, int metric_count,
    const float* base_values,
    const float* perturbations,
    void (*evaluate_func)(const float* params, int param_count,
                          float* metrics, int metric_count));

/**
 * @brief 释放敏感性分析结果
 *
 * @param result 敏感性分析结果指针
 */
void plan_robustness_destroy_sensitivity_result(SensitivityResult* result);

/**
 * @brief 执行蒙特卡洛鲁棒性估计
 *
 * 通过随机采样参数分布，模拟大量计划执行场景，
 * 统计成功概率、时长分布和风险值。
 *
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param sample_func 采样函数：输入(采样结果数组, 参数数量)
 * @param evaluate_func 评估函数：输入(采样参数, 参数数量)，输出(成功标志, 时长, 成本)
 * @param param_count 采样参数数量
 * @param num_simulations 模拟次数
 * @return MonteCarloResult* 蒙特卡洛结果，失败返回NULL
 */
MonteCarloResult* plan_robustness_monte_carlo(
    PlanAction** plan_actions,
    int action_count,
    void (*sample_func)(float* samples, int param_count),
    int (*evaluate_func)(const float* params, int param_count,
                         float* duration, float* cost),
    int param_count,
    int num_simulations);

/**
 * @brief 释放蒙特卡洛结果
 *
 * @param result 蒙特卡洛结果指针
 */
void plan_robustness_destroy_monte_carlo_result(MonteCarloResult* result);

/**
 * @brief 执行完整的鲁棒性分析
 *
 * 综合分析时间松弛度、资源松弛度、敏感性和蒙特卡洛估计，
 * 生成整体鲁棒性分数和改进建议。
 *
 * @param system 规划系统
 * @param plan_actions 计划动作数组
 * @param action_count 动作数量
 * @param num_monte_carlo_samples 蒙特卡洛采样次数（0=跳过蒙特卡洛）
 * @return RobustnessResult* 完整分析结果，失败返回NULL
 */
RobustnessResult* long_term_planning_analyze_robustness(
    LongTermPlanningSystem* system,
    PlanAction** plan_actions,
    int action_count,
    int num_monte_carlo_samples);

/**
 * @brief 释放完整鲁棒性分析结果
 *
 * @param result 分析结果指针
 */
void long_term_planning_destroy_robustness_result(RobustnessResult* result);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_REASONING_LONG_TERM_PLANNING_H */