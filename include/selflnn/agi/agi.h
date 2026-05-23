#ifndef SELFLNN_AGI_H
#define SELFLNN_AGI_H

#include <stddef.h>
#include <time.h>
#include "selflnn/core/common.h"
#include "selflnn/core/lnn.h"
#include "selflnn/knowledge/knowledge.h"
#include "selflnn/reasoning/reasoning.h"
#include "selflnn/reasoning/planning.h"
#include "selflnn/core/decision_engine.h"
#include "selflnn/learning/learning.h"
#include "selflnn/memory/memory_manager.h"
#include "selflnn/metacognition.h"
#include "selflnn/self_cognition.h"
#include "selflnn/cognition/deep_reflection.h"
#include "selflnn/cognition/deep_thought_chain.h"
#include "selflnn/cognition/deep_correction.h"
#include "selflnn/multimodal/dialogue.h"
#include "selflnn/multimodal/liquid_vision.h"
#include "selflnn/multimodal/depth_estimation.h"
#include "selflnn/multisystem/multisystem_control.h"
#include "selflnn/core/unified_lnn_state.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGI_MAX_GOALS 32
#define AGI_MAX_SUBGOALS 128
#define AGI_MAX_TASKS 256
#define AGI_MAX_COGNITIVE_STATES 1024
#define AGI_MAX_DIALOGUE_HISTORY 512
#define AGI_MAX_PLANS 64
#define AGI_NAME_LEN 128
#define AGI_DESC_LEN 512
#define AGI_STATE_VECTOR_DIM 256

typedef enum {
    AGI_STATE_IDLE = 0,
    AGI_STATE_PERCEIVE = 1,
    AGI_STATE_REASON = 2,
    AGI_STATE_PLAN = 3,
    AGI_STATE_DECIDE = 4,
    AGI_STATE_EXECUTE = 5,
    AGI_STATE_LEARN = 6,
    AGI_STATE_REFLECT = 7,
    AGI_STATE_ERROR = 8,
    AGI_STATE_SLEEP = 9
} AGIState;

typedef enum {
    AGI_GOAL_ACHIEVED = 0,
    AGI_GOAL_IN_PROGRESS = 1,
    AGI_GOAL_FAILED = 2,
    AGI_GOAL_SUSPENDED = 3,
    AGI_GOAL_PENDING = 4,
    AGI_GOAL_CANCELLED = 5
} AIGoalStatus;

typedef enum {
    AGI_TASK_PENDING = 0,
    AGI_TASK_RUNNING = 1,
    AGI_TASK_COMPLETED = 2,
    AGI_TASK_FAILED = 3,
    AGI_TASK_CANCELLED = 4
} AGITaskStatus;

typedef enum {
    AGI_PRIORITY_CRITICAL = 0,
    AGI_PRIORITY_HIGH = 1,
    AGI_PRIORITY_NORMAL = 2,
    AGI_PRIORITY_LOW = 3,
    AGI_PRIORITY_BACKGROUND = 4
} AGIPriority;

typedef struct {
    int goal_id;
    char name[AGI_NAME_LEN];
    char description[AGI_DESC_LEN];
    float priority;
    AIGoalStatus status;
    float progress;
    float deadline;
    time_t created_at;
    time_t updated_at;
    int parent_goal_id;
    int subgoal_ids[AGI_MAX_SUBGOALS];
    int subgoal_count;
    float state_vector[AGI_STATE_VECTOR_DIM];
    int state_vector_dim;
} AIGoal;

typedef struct {
    int task_id;
    char name[AGI_NAME_LEN];
    char description[AGI_DESC_LEN];
    AGITaskStatus status;
    AGIPriority priority;
    float progress;
    time_t created_at;
    time_t started_at;
    time_t completed_at;
    int goal_id;
    float* action_sequence;
    int action_count;
    int current_action_index;
    void* result_data;
    size_t result_size;
    int error_code;
    char error_message[AGI_DESC_LEN];
} AGITask;

typedef struct {
    AGIState state;
    float confidence;
    float curiosity;
    float cognitive_load;
    float attention_focus;
    int active_goal_count;
    int active_task_count;
    int completed_task_count;
    int failed_task_count;
    int knowledge_count;
    int memory_count;
    float learning_progress;
    int reflection_count;
    time_t last_state_change;
    time_t uptime;
    int cycle_count;
} AGICognitiveStatus;

typedef struct {
    int enable_self_decision;
    int enable_autonomous_execution;
    int enable_self_learning;
    int enable_self_evolution;
    int enable_imitation_learning;
    int enable_self_correction;
    int enable_reflection;
    int enable_curiosity;
    int enable_planning;
    int enable_dialogue;
    int max_tasks;
    int max_goals;
    float learning_rate;
    float exploration_rate;
    float reflection_interval;
    float cognitive_load_limit;
    int state_vector_dim;
    int knowledge_capacity;
    int memory_capacity;
    int reasoning_depth;
    int planning_horizon;
} AGIConfig;

#define AGI_CONFIG_DEFAULT { \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    AGI_MAX_TASKS, AGI_MAX_GOALS, \
    0.01f, 0.1f, 300.0f, 0.8f, \
    AGI_STATE_VECTOR_DIM, 10000, 10000, \
    3, 10 \
}

typedef struct AGISystem AGISystem;

AGISystem* agi_system_create(const AGIConfig* config);

void agi_system_free(AGISystem* system);

int agi_system_set_knowledge_base(AGISystem* system, KnowledgeBase* kb);
int agi_system_set_reasoning_engine(AGISystem* system, ReasoningEngine* engine);
int agi_system_set_decision_engine(AGISystem* system, DecisionEngine* engine);
int agi_system_set_planning_system(AGISystem* system, PlanningSystem* planner);
int agi_system_set_learning_engine(AGISystem* system, LearningEngine* learner);
int agi_system_set_memory_manager(AGISystem* system, MemoryManager* memory);
int agi_system_set_metacognition(AGISystem* system, MetacognitionSystem* meta);
int agi_system_set_self_cognition(AGISystem* system, SelfCognitionSystem* self_cog);
int agi_system_set_reflection(AGISystem* system, DeepReflectionEngine* reflection);
int agi_system_set_thought_chain(AGISystem* system, DTCSystem* dtc);
int agi_system_set_correction(AGISystem* system, DCCorrectionSystem* correction);
int agi_system_set_dialogue(AGISystem* system, DialogueProcessor* dialogue);
int agi_system_set_lnn(AGISystem* system, LNN* lnn);
int agi_system_set_unified_lnn(AGISystem* system, UnifiedLNNState* state);
int agi_system_set_vision_processor(AGISystem* system, CfcVisionProcessor* vision);
int agi_system_set_depth_estimator(AGISystem* system, DepthEstimator* estimator);
int agi_system_set_multisystem_control(AGISystem* system, MultiSystemControlEngine* engine);

int agi_system_set_config(AGISystem* system, const AGIConfig* config);
int agi_system_get_config(const AGISystem* system, AGIConfig* config);

int agi_system_add_goal(AGISystem* system, const char* name, const char* description, float priority, float deadline, const float* state_vector, int state_vector_dim);
int agi_system_update_goal(AGISystem* system, int goal_id, float progress, AIGoalStatus status);
int agi_system_get_goal(const AGISystem* system, int goal_id, AIGoal* goal);
int agi_system_remove_goal(AGISystem* system, int goal_id);
int agi_system_list_goals(const AGISystem* system, int* goal_ids, int max_count);

int agi_system_add_task(AGISystem* system, const char* name, const char* description, AGIPriority priority, int goal_id);
int agi_system_update_task(AGISystem* system, int task_id, float progress, AGITaskStatus status);
int agi_system_get_task(const AGISystem* system, int task_id, AGITask* task);
int agi_system_remove_task(AGISystem* system, int task_id);
int agi_system_list_tasks(const AGISystem* system, int* task_ids, int max_count);

int agi_system_cognitive_cycle(AGISystem* system, const float* sensory_input, int input_dim, float* output, int output_dim);
int agi_system_cognitive_cycle_multimodal(AGISystem* system,
    const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES],
    const int raw_dims[UNIFIED_LNN_MAX_MODALITIES],
    const int modality_present[UNIFIED_LNN_MAX_MODALITIES],
    float* output, int output_dim);

int agi_system_perceive(AGISystem* system, const float* sensory_input, int input_dim, float* state_vector);
int agi_system_perceive_multimodal(AGISystem* system,
    const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES],
    const int raw_dims[UNIFIED_LNN_MAX_MODALITIES],
    const int modality_present[UNIFIED_LNN_MAX_MODALITIES],
    float* state_vector, int state_vector_dim);
int agi_system_reason(AGISystem* system, const float* state_vector, int state_dim, float* reasoning_result);
int agi_system_plan(AGISystem* system, const float* reasoning_result, int result_dim, float* plan, int* plan_steps);
int agi_system_decide(AGISystem* system, const float* plan_options, int num_options, int* chosen_option);
int agi_system_execute(AGISystem* system, int chosen_option, float* execution_output, int output_dim);
int agi_system_learn(AGISystem* system, const float* experience, int exp_dim, float reward);

int agi_system_self_reflect(AGISystem* system, char* reflection_text, size_t max_len);
int agi_system_generate_dialogue(AGISystem* system, const char* input_text, char* response, size_t max_len);
int agi_system_set_autonomous_mode(AGISystem* system, int enable);
int agi_system_get_status(const AGISystem* system, AGICognitiveStatus* status);

int agi_system_save_state(AGISystem* system, const char* filepath);
int agi_system_load_state(AGISystem* system, const char* filepath);

int agi_system_enable_capability(AGISystem* system, const char* capability_name, int enable);
int agi_system_is_capability_enabled(const AGISystem* system, const char* capability_name);

/* ============================================================================
 * 多目标效用理论决策 & 博弈论决策 & 元决策引擎
 * ============================================================================ */

#define AGI_MAX_PLAYERS 8
#define AGI_MAX_STRATEGIES 16
#define AGI_MAX_PAYOFF_DIM 4

/**
 * @brief 多目标聚合方法
 */
typedef enum {
    AGI_MAUT_WEIGHTED_SUM = 0,       /**< 加权和（经典线性聚合） */
    AGI_MAUT_LEXICOGRAPHIC = 1,      /**< 字典序（按优先级排序） */
    AGI_MAUT_TOPSIS = 2,             /**< TOPSIS（理想解距离法） */
    AGI_MAUT_GOAL_PROGRAMMING = 3,   /**< 目标规划（距离目标值最小化） */
    AGI_MAUT_AHP = 4                 /**< 层次分析法（成对比较矩阵） */
} AGIMAUTMethod;

/**
 * @brief MAUT多目标决策结果
 */
typedef struct {
    int best_alternative;            /**< 最优方案索引 */
    float* utility_scores;           /**< 各方案综合效用分 [num_alternatives] */
    float* ranking;                  /**< 排序值 (1=最优) */
    float consistency_ratio;         /**< 一致性比率 (AHP方法, <0.1为可接受) */
    float overall_confidence;        /**< 整体决策置信度 */
    int* pareto_optimal;             /**< 帕累托最优标志 [num_alternatives] */
    int num_pareto_optimal;          /**< 帕累托最优方案数 */
    char decision_rationale[512];    /**< 决策理由 */
} MAUTResult;

/**
 * @brief 博弈论解概念枚举
 */
typedef enum {
    AGI_GAME_NASH_PURE = 0,          /**< 纯策略纳什均衡 */
    AGI_GAME_NASH_MIXED = 1,         /**< 混合策略纳什均衡 */
    AGI_GAME_MINIMAX = 2,            /**< 极小化极大（零和博弈） */
    AGI_GAME_STACKELBERG = 3,        /**< 斯塔克尔伯格均衡（领导者-追随者） */
    AGI_GAME_CORRELATED = 4,         /**< 相关均衡 */
    AGI_GAME_COOPERATIVE = 5,        /**< 合作博弈（纳什讨价还价解） */
    AGI_GAME_PARETO_BARGAINING = 6   /**< 帕累托讨价还价 */
} AGIGameConcept;

/**
 * @brief 博弈论玩家定义
 */
typedef struct {
    char name[64];                   /**< 玩家名称 */
    int num_strategies;              /**< 策略数 */
    float* payoff_matrix;            /**< 收益矩阵 [num_strategies × (total_strategies)] */
    float reservation_price;         /**< 保留价格（合作博弈） */
    float bargaining_power;          /**< 谈判力 (0-1) */
    int is_cooperative;              /**< 是否合作 */
} AGIPlayer;

/**
 * @brief 博弈论求解结果
 */
typedef struct {
    AGIGameConcept concept;          /**< 使用的解概念 */
    int num_players;                 /**< 玩家数 */
    int* equilibrium_profile;        /**< 均衡策略组合 [num_players] */
    float* mixed_strategy;           /**< 混合策略概率分布 [num_players × max_strategies] */
    float* equilibrium_payoffs;      /**< 均衡收益 [num_players] */
    float social_welfare;            /**< 社会福利 (总收益) */
    float fairness_index;            /**< 公平指数 (0-1, 1=完全公平) */
    float nash_product;              /**< 纳什积（合作博弈） */
    int is_unique;                   /**< 均衡是否唯一 */
    int num_equilibria_found;        /**< 发现的均衡数 */
    float computation_time_ms;       /**< 计算耗时 */
    char rationale[512];             /**< 解算理由 */
} AGIGameResult;

/**
 * @brief 元决策策略类型
 */
typedef enum {
    AGI_META_DECISION_MAUT = 0,      /**< 多目标效用法 */
    AGI_META_DECISION_GAME = 1,      /**< 博弈论法（有多方交互时） */
    AGI_META_DECISION_BAYESIAN = 2,  /**< 贝叶斯决策（有概率分布时） */
    AGI_META_DECISION_RULE_BASED = 3,/**< 规则推理（少量方案时） */
    AGI_META_DECISION_HEURISTIC = 4, /**< 启发式搜索（大规模方案时） */
    AGI_META_DECISION_LEARNED = 5    /**< 学习策略（有历史数据时） */
} AGIMetaDecisionStrategy;

/**
 * @brief 元决策引擎上下文
 */
typedef struct {
    int num_alternatives;            /**< 方案数 */
    int num_objectives;              /**< 目标数 */
    int num_players;                 /**< 博弈玩家数 (>0 触发博弈论) */
    int has_probabilities;           /**< 是否有概率分布 */
    int has_historical_data;         /**< 是否有历史决策数据 */
    float problem_scale;             /**< 问题规模评分 (0-1) */
    float urgency;                   /**< 紧迫度 (0-1, 1=急需) */
    float risk_tolerance;            /**< 风险容忍度 (0-1) */
    float available_time_ms;         /**< 可用计算时间 */
} MetaDecisionContext;

/**
 * @brief 元决策结果
 */
typedef struct {
    AGIMetaDecisionStrategy chosen_strategy; /**< 选择的策略 */
    float strategy_confidence;       /**< 策略选择置信度 */
    MAUTResult maut_result;          /**< MAUT结果（若使用） */
    AGIGameResult game_result;       /**< 博弈论结果（若使用） */
    int best_alternative;            /**< 最终最优方案 */
    float overall_utility;           /**< 最终综合效用 */
    int execution_path;              /**< 执行路径 (0=标准,1=快速,2=深度) */
    char strategy_rationale[512];    /**< 策略选择理由 */
} MetaDecisionResult;

/* ---- 多目标效用理论 API ---- */

/**
 * @brief 执行MAUT多目标决策分析
 *
 * 支持5种聚合方法：加权和、字典序、TOPSIS、目标规划、AHP。
 * 自动计算帕累托前沿和各方案的综合效用排名。
 *
 * @param system AGI系统句柄
 * @param attribute_matrix 属性矩阵 [num_alts × num_obj] (行=方案, 列=目标)
 * @param num_alts 方案数
 * @param num_obj 目标数
 * @param weights 目标权重 [num_obj]
 * @param is_max 最大化标志 [num_obj] (1=最大化, 0=最小化)
 * @param method 聚合方法
 * @param result 输出结果
 * @return 0成功, -1失败
 */
int agi_maut_decide(AGISystem* system,
                    const float* attribute_matrix,
                    int num_alts, int num_obj,
                    const float* weights,
                    const int* is_max,
                    AGIMAUTMethod method,
                    MAUTResult* result);

/**
 * @brief 释放MAUT结果
 */
void agi_maut_result_free(MAUTResult* result);

/* ---- 博弈论决策 API ---- */

/**
 * @brief 执行博弈论决策分析
 *
 * 支持纯/混合纳什均衡、极小化极大、斯塔克尔伯格、
 * 相关均衡、合作博弈（纳什讨价还价解）6种解概念。
 *
 * @param system AGI系统句柄
 * @param players 玩家定义数组 [num_players]
 * @param num_players 玩家数
 * @param concept 解概念类型
 * @param result 输出结果
 * @return 0成功, -1失败
 */
int agi_game_decide(AGISystem* system,
                    const AGIPlayer* players,
                    int num_players,
                    AGIGameConcept concept,
                    AGIGameResult* result);

/**
 * @brief 释放博弈论结果
 */
void agi_game_result_free(AGIGameResult* result);

/* ---- 元决策引擎 API ---- */

/**
 * @brief 元决策引擎 —— 自动选择最优决策策略
 *
 * 基于问题上下文（方案数、目标数、有无对手/概率/历史数据）
 * 自动选择最优决策策略（MAUT/博弈论/贝叶斯/规则/启发式/学习），
 * 执行决策并返回综合结果。
 *
 * @param system AGI系统句柄
 * @param attribute_matrix 属性矩阵 [num_alts × num_obj]
 * @param num_alts 方案数
 * @param num_obj 目标数
 * @param weights 目标权重 [num_obj]
 * @param is_max 最大化标志 [num_obj]
 * @param players 博弈玩家 (NULL=无对手)
 * @param num_players 玩家数
 * @param context 元决策上下文
 * @param result 输出综合结果
 * @return 0成功, -1失败
 */
int agi_meta_decide(AGISystem* system,
                    const float* attribute_matrix,
                    int num_alts, int num_obj,
                    const float* weights,
                    const int* is_max,
                    const AGIPlayer* players,
                    int num_players,
                    const MetaDecisionContext* context,
                    MetaDecisionResult* result);

/**
 * @brief 释放元决策结果
 */
void agi_meta_decision_result_free(MetaDecisionResult* result);

int agi_system_plan_execution(AGISystem* system, int goal_id, float* action_sequence, int* num_actions);
int agi_system_execute_plan_step(AGISystem* system, int task_id);
int agi_system_monitor_execution(AGISystem* system, int task_id, float* feedback);

int agi_system_imitate(AGISystem* system, const float* demonstration, int demo_dim, const float* target_output, int target_dim);
int agi_system_self_evolve(AGISystem* system, float* new_parameters, int* param_count);

int agi_system_set_image_input(AGISystem* system,
    const float* image_data, int width, int height, int channels);

int agi_system_process_image(AGISystem* system,
    const float* image_data, int width, int height, int channels,
    float* visual_features, int max_features);
int agi_system_process_stereo(AGISystem* system,
    const float* left_image, const float* right_image,
    int width, int height, int channels,
    float* depth_features, int max_features,
    StereoCalibration* calibration);

#ifdef __cplusplus
}
#endif

#endif
