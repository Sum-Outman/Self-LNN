/**
 * @file bdi_model.h
 * @brief BDI（信念-愿望-意图）模型完整接口
 *
 * 完整的BDI认知架构，实现智能体的信念(Belief)、愿望(Desire)、意图(Intention)
 * 推理循环。包含目标管理、手段-目的推理、计划库、意图重考虑和BDI执行循环。
 *
 * BDI循环: 感知 → 信念更新 → 选项生成 → 愿望过滤 → 意图承诺 → 计划执行
 *
 * 原始代码位置: src/cognition/self_cognition.c
 * 深度实现日期: 2026-07-22 (L-009修复)
 */

#ifndef SELFLNN_BDI_MODEL_H
#define SELFLNN_BDI_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 基础工具函数（保留原有API兼容）
 * ============================================================================ */

/**
 * @brief 贝叶斯信念更新
 * 信念 = 信念 * prior_weight + 观测 * posterior_weight
 */
void bdi_update_belief_bayesian(float* belief, size_t dim,
                                 const float* observation, float certainty);

/**
 * @brief 从欲望和信念计算意图（哈达玛积 + L1归一化）
 */
void bdi_compute_intention_from_desire_belief(float* intention, const float* desire,
                                               const float* belief, size_t dim);

/**
 * @brief 初始化信念向量为均等分布（最大熵）
 */
void bdi_init_belief_uniform(float* belief, size_t dim);

/**
 * @brief 初始化意图向量为零
 */
void bdi_init_intention_zero(float* intention, size_t dim);

/* ============================================================================
 * BDI完整架构类型定义
 * ============================================================================ */

/** @brief 目标状态 */
typedef enum {
    BDI_GOAL_INACTIVE = 0,     /**< 未激活 */
    BDI_GOAL_ACTIVE = 1,       /**< 已激活，等待处理 */
    BDI_GOAL_PURSUING = 2,     /**< 正在追求 */
    BDI_GOAL_ACHIEVED = 3,     /**< 已达成 */
    BDI_GOAL_FAILED = 4,       /**< 失败 */
    BDI_GOAL_SUSPENDED = 5     /**< 挂起 */
} BDIGoalState;

/** @brief 计划步骤类型 */
typedef enum {
    BDI_STEP_ACTION = 0,       /**< 执行动作 */
    BDI_STEP_SUBGOAL = 1,      /**< 子目标 */
    BDI_STEP_CONDITION = 2,    /**< 条件检查 */
    BDI_STEP_PARALLEL = 3      /**< 并行分支 */
} BDIStepType;

/** @brief 意图状态 */
typedef enum {
    BDI_INTENTION_PENDING = 0,  /**< 待定 */
    BDI_INTENTION_COMMITTED = 1,/**< 已承诺 */
    BDI_INTENTION_EXECUTING = 2,/**< 执行中 */
    BDI_INTENTION_COMPLETED = 3,/**< 已完成 */
    BDI_INTENTION_DROPPED = 4   /**< 已放弃 */
} BDIIntentionState;

/** @brief BDI目标 */
typedef struct {
    int goal_id;                /**< 目标ID */
    char* name;                 /**< 目标名称 */
    char* description;          /**< 目标描述 */
    BDIGoalState state;         /**< 目标状态 */
    float priority;             /**< 优先级 (0.0-1.0) */
    float urgency;              /**< 紧迫度 (0.0-1.0) */
    long deadline;              /**< 截止时间（Unix时间戳，0=无截止） */
    long created_time;          /**< 创建时间 */
    long achieved_time;         /**< 达成时间 */
    float* precondition;        /**< 前置条件向量（信念空间） */
    float* success_condition;   /**< 成功条件向量 */
    size_t condition_dim;       /**< 条件向量维度 */
    int parent_goal_id;         /**< 父目标ID（-1=顶级目标） */
    int retry_count;            /**< 重试次数 */
    int max_retries;            /**< 最大重试次数 */
} BDIGoal;

/** @brief BDI计划步骤 */
typedef struct {
    BDIStepType type;           /**< 步骤类型 */
    char* action_name;          /**< 动作名称 */
    char* action_params;        /**< 动作参数 */
    float* expected_effect;     /**< 预期效果向量 */
    size_t effect_dim;          /**< 效果向量维度 */
    float cost;                 /**< 执行成本 */
    float duration;             /**< 预计耗时 */
} BDIPlanStep;

/** @brief BDI计划 */
typedef struct {
    int plan_id;                /**< 计划ID */
    char* name;                 /**< 计划名称 */
    BDIPlanStep* steps;         /**< 计划步骤 */
    size_t step_count;          /**< 步骤数量 */
    size_t step_capacity;       /**< 步骤容量 */
    float* precondition;        /**< 前置条件向量 */
    float* expected_outcome;    /**< 预期结果向量 */
    size_t condition_dim;       /**< 条件向量维度 */
    float success_rate;         /**< 历史成功率 (0.0-1.0) */
    int use_count;              /**< 使用次数 */
    int success_count;          /**< 成功次数 */
    long last_used;             /**< 最后使用时间 */
} BDIPlan;

/** @brief BDI意图 */
typedef struct {
    int intention_id;           /**< 意图ID */
    int goal_id;                /**< 关联目标ID */
    int plan_id;                /**< 关联计划ID */
    BDIIntentionState state;    /**< 意图状态 */
    float commitment;           /**< 承诺强度 (0.0-1.0) */
    int current_step;           /**< 当前执行步骤 */
    long committed_time;        /**< 承诺时间 */
    long last_executed;         /**< 最后执行时间 */
} BDIIntention;

/** @brief BDI配置 */
typedef struct {
    int belief_dim;              /**< 信念向量维度 */
    int desire_dim;              /**< 欲望向量维度 */
    int intention_dim;           /**< 意图向量维度 */
    int max_goals;               /**< 最大目标数 */
    int max_plans;               /**< 最大计划数 */
    int max_intentions;          /**< 最大意图数 */
    int max_plan_steps;          /**< 每计划最大步骤数 */
    float intention_reconsideration_rate; /**< 意图重考虑频率 (0.0-1.0) */
    float goal_decay_rate;       /**< 目标紧迫度衰减率 */
    float commitment_threshold;  /**< 承诺阈值 (0.0-1.0) */
    float drop_threshold;        /**< 放弃意图阈值 */
    int max_retries;             /**< 最大重试次数 */
    long step_timeout_ms;        /**< 步骤超时（毫秒） */
} BDIConfig;

/** @brief BDI统计 */
typedef struct {
    int goals_created;           /**< 创建的目标数 */
    int goals_achieved;          /**< 达成的目标数 */
    int goals_failed;            /**< 失败的目标数 */
    int plans_created;           /**< 创建的计划数 */
    int plans_executed;          /**< 执行的计划数 */
    int plans_succeeded;         /**< 成功的计划数 */
    int intentions_committed;    /**< 承诺的意图数 */
    int intentions_dropped;      /**< 放弃的意图数 */
    int intentions_completed;    /**< 完成的意图数 */
    int reconsiderations;       /**< 重考虑次数 */
    int cycles_executed;         /**< 执行的BDI循环数 */
    long total_plan_time_ms;     /**< 总计划执行时间 */
    long last_cycle_time_ms;     /**< 上次循环耗时 */
    float avg_plan_success_rate; /**< 平均计划成功率 */
    float avg_goal_achievement_time_ms; /**< 平均目标达成时间 */
} BDIStats;

/** @brief BDI模型（不透明句柄） */
typedef struct BDIModel BDIModel;

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

/**
 * @brief 创建BDI模型
 * @param config BDI配置（NULL则使用默认配置）
 * @return BDIModel* 模型句柄，失败返回NULL
 */
BDIModel* bdi_model_create(const BDIConfig* config);

/**
 * @brief 初始化BDI模型（分配内部缓冲区）
 * @param model BDI模型句柄
 * @return int 成功返回0，失败返回-1
 */
int bdi_model_init(BDIModel* model);

/**
 * @brief 销毁BDI模型
 * @param model BDI模型句柄
 */
void bdi_model_destroy(BDIModel* model);

/**
 * @brief 重置BDI模型（清空所有信念/欲望/目标/计划/意图）
 * @param model BDI模型句柄
 * @return int 成功返回0
 */
int bdi_model_reset(BDIModel* model);

/* ============================================================================
 * 信念管理
 * ============================================================================ */

/**
 * @brief 更新信念向量（贝叶斯融合）
 * @param model BDI模型
 * @param observation 观测向量
 * @param certainty 观测确定性 (0.0-1.0)
 * @return int 成功返回0
 */
int bdi_model_update_belief(BDIModel* model, const float* observation, float certainty);

/**
 * @brief 获取当前信念向量
 * @param model BDI模型
 * @param belief_out 输出缓冲区
 * @param dim 缓冲区维度
 * @return int 成功返回实际维度，失败返回-1
 */
int bdi_model_get_belief(const BDIModel* model, float* belief_out, size_t dim);

/**
 * @brief 直接设置信念向量
 * @param model BDI模型
 * @param belief 信念向量
 * @param dim 向量维度
 * @return int 成功返回0
 */
int bdi_model_set_belief(BDIModel* model, const float* belief, size_t dim);

/* ============================================================================
 * 欲望管理
 * ============================================================================ */

/**
 * @brief 获取当前欲望向量
 * @param model BDI模型
 * @param desire_out 输出缓冲区
 * @param dim 缓冲区维度
 * @return int 成功返回实际维度，失败返回-1
 */
int bdi_model_get_desire(const BDIModel* model, float* desire_out, size_t dim);

/**
 * @brief 设置欲望向量
 * @param model BDI模型
 * @param desire 欲望向量
 * @param dim 向量维度
 * @return int 成功返回0
 */
int bdi_model_set_desire(BDIModel* model, const float* desire, size_t dim);

/**
 * @brief 增强特定欲望维度
 * @param model BDI模型
 * @param dim_index 维度索引
 * @param increment 增量 (0.0-1.0)
 * @return int 成功返回0
 */
int bdi_model_boost_desire(BDIModel* model, size_t dim_index, float increment);

/* ============================================================================
 * 意图管理
 * ============================================================================ */

/**
 * @brief 从当前信念和欲望计算意图向量
 * @param model BDI模型
 * @return int 成功返回0
 */
int bdi_model_compute_intention(BDIModel* model);

/**
 * @brief 获取当前意图向量
 * @param model BDI模型
 * @param intention_out 输出缓冲区
 * @param dim 缓冲区维度
 * @return int 成功返回实际维度
 */
int bdi_model_get_intention(const BDIModel* model, float* intention_out, size_t dim);

/**
 * @brief 意图重考虑：检查是否应放弃当前意图
 * @param model BDI模型
 * @return int 放弃的意图数量
 */
int bdi_model_reconsider_intentions(BDIModel* model);

/* ============================================================================
 * 目标管理
 * ============================================================================ */

/**
 * @brief 添加新目标
 * @param model BDI模型
 * @param name 目标名称
 * @param description 目标描述
 * @param priority 优先级 (0.0-1.0)
 * @param deadline 截止时间（0=无截止）
 * @param precondition 前置条件向量（NULL=无条件）
 * @param success_condition 成功条件向量（NULL=无条件）
 * @param cond_dim 条件向量维度
 * @return int 目标ID，失败返回-1
 */
int bdi_model_add_goal(BDIModel* model, const char* name, const char* description,
                       float priority, long deadline,
                       const float* precondition, const float* success_condition,
                       size_t cond_dim);

/**
 * @brief 移除目标
 * @param model BDI模型
 * @param goal_id 目标ID
 * @return int 成功返回0
 */
int bdi_model_remove_goal(BDIModel* model, int goal_id);

/**
 * @brief 目标优先级排序（按优先级×紧迫度）
 * @param model BDI模型
 * @return int 排序后的活跃目标数
 */
int bdi_model_prioritize_goals(BDIModel* model);

/**
 * @brief 获取最高优先级目标
 * @param model BDI模型
 * @return int 目标ID，无活跃目标返回-1
 */
int bdi_model_get_top_goal(BDIModel* model);

/**
 * @brief 更新目标状态
 * @param model BDI模型
 * @param goal_id 目标ID
 * @param state 新状态
 * @return int 成功返回0
 */
int bdi_model_set_goal_state(BDIModel* model, int goal_id, BDIGoalState state);

/**
 * @brief 目标重考虑：检查是否应放弃不可达目标
 * @param model BDI模型
 * @return int 重考虑的目标数
 */
int bdi_model_reconsider_goals(BDIModel* model);

/* ============================================================================
 * 计划库管理
 * ============================================================================ */

/**
 * @brief 添加计划到计划库
 * @param model BDI模型
 * @param name 计划名称
 * @param steps 步骤数组
 * @param step_count 步骤数量
 * @param precondition 前置条件向量
 * @param expected_outcome 预期结果向量
 * @param cond_dim 条件向量维度
 * @return int 计划ID，失败返回-1
 */
int bdi_model_add_plan(BDIModel* model, const char* name,
                       const BDIPlanStep* steps, size_t step_count,
                       const float* precondition, const float* expected_outcome,
                       size_t cond_dim);

/**
 * @brief 手段-目的推理：为给定目标寻找最佳计划
 * 评估所有计划的前置条件与当前信念的匹配度，返回最合适的计划。
 * @param model BDI模型
 * @param goal_id 目标ID
 * @return int 最佳计划ID，无合适计划返回-1
 */
int bdi_model_means_end_reasoning(BDIModel* model, int goal_id);

/**
 * @brief 执行计划的一个步骤
 * @param model BDI模型
 * @param plan_id 计划ID
 * @param step_index 步骤索引
 * @param action_result 动作执行结果输出（NULL=不关心）
 * @return int 成功返回0，计划完成返回1，失败返回-1
 */
int bdi_model_execute_step(BDIModel* model, int plan_id, int step_index,
                           float* action_result);

/**
 * @brief 记录计划执行结果
 * @param model BDI模型
 * @param plan_id 计划ID
 * @param success 是否成功
 */
void bdi_model_record_plan_result(BDIModel* model, int plan_id, int success);

/* ============================================================================
 * BDI执行循环
 * ============================================================================ */

/**
 * @brief 执行一次完整的BDI推理循环
 * 
 * 循环步骤:
 *   1. 信念更新（如果提供了观测）
 *   2. 意图重考虑
 *   3. 目标优先级排序
 *   4. 手段-目的推理（为最高优先级目标选计划）
 *   5. 意图承诺
 *   6. 计划步骤执行
 *
 * @param model BDI模型
 * @param observation 观测向量（NULL=跳过信念更新）
 * @param certainty 观测确定性
 * @return int 执行的动作数，失败返回-1
 */
int bdi_model_step(BDIModel* model, const float* observation, float certainty);

/**
 * @brief 获取当前正在执行的意图
 * @param model BDI模型
 * @param intention_out 输出意图信息
 * @return int 成功返回0，无活跃意图返回-1
 */
int bdi_model_get_current_intention(const BDIModel* model, BDIIntention* intention_out);

/* ============================================================================
 * LNN集成
 * ============================================================================ */

/**
 * @brief 将BDI模型连接到LNN网络
 * 连接后，信念更新和意图计算将使用LNN进行非线性状态演化。
 * @param model BDI模型
 * @param lnn LNN网络句柄
 * @return int 成功返回0
 */
int bdi_model_set_lnn(BDIModel* model, void* lnn);

/* ============================================================================
 * 统计查询
 * ============================================================================ */

/**
 * @brief 获取BDI运行统计
 * @param model BDI模型
 * @param stats 输出统计信息
 * @return int 成功返回0
 */
int bdi_model_get_stats(const BDIModel* model, BDIStats* stats);

/**
 * @brief 获取默认BDI配置
 * @param config 输出配置
 */
void bdi_config_get_default(BDIConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_BDI_MODEL_H */