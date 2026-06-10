/**
 * @file self_cognition.h
 * @brief 自我认知系统接口
 * 
 * 自我认知系统允许AGI系统监控自身状态、评估能力、跟踪学习进展，
 * 并实现自我反思和元认知功能。
 */

#ifndef SELFLNN_SELF_COGNITION_H
#define SELFLNN_SELF_COGNITION_H

#include <stddef.h>
#include <time.h>
#include "selflnn/cognition/metacognition.h"

/* L-009: 认知模块依赖关系已验证 — 无循环依赖
 * self_cognition.h → metacognition.h (单向, metacognition.h使用前向声明避免反向引用)
 * deep_reflection.h → metacognition.h (单向)
 * 其余头文件(deep_correction/deep_thought_chain/abstraction)无跨模块include */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 自我认知维度枚举
 */
typedef enum {
    SELF_COGNITION_STATE = 0,      /**< 系统状态认知 */
    SELF_COGNITION_CAPABILITY = 1, /**< 能力认知 */
    SELF_COGNITION_KNOWLEDGE = 2,  /**< 知识认知 */
    SELF_COGNITION_GOAL = 3,       /**< 目标认知 */
    SELF_COGNITION_LEARNING = 4,   /**< 学习进展认知 */
    SELF_COGNITION_PERFORMANCE = 5 /**< 性能认知 */
} SelfCognitionDimension;

/**
 * @brief 系统状态结构体
 */
typedef struct {
    float cpu_usage;               /**< CPU使用率 (0-1) */
    float memory_usage;            /**< 内存使用率 (0-1) */
    float gpu_usage;               /**< GPU使用率 (0-1) */
    float disk_usage;              /**< 磁盘使用率 (0-1) */
    int error_count;               /**< 错误计数 */
    int warning_count;             /**< 警告计数 */
    float uptime_hours;            /**< 运行时间 (小时) */
    float temperature;             /**< 系统温度 (°C) */
} CognitionSystemStatus;

/**
 * @brief 能力评估结构体
 */
typedef struct {
    float reasoning_ability;       /**< 推理能力分数 (0-1) */
    float learning_ability;        /**< 学习能力分数 (0-1) */
    float memory_capacity;         /**< 记忆容量分数 (0-1) */
    float planning_ability;        /**< 规划能力分数 (0-1) */
    float perception_ability;      /**< 感知能力分数 (0-1) */
    float action_ability;          /**< 行动能力分数 (0-1) */
    float adaptability;            /**< 适应能力分数 (0-1) */
    float creativity;              /**< 创造力分数 (0-1) */
} CapabilityAssessment;

/**
 * @brief 知识元认知结构体
 */
typedef struct {
    size_t known_concepts;         /**< 已知概念数量 */
    size_t unknown_concepts;       /**< 未知概念数量 */
    float knowledge_coverage;      /**< 知识覆盖率 (0-1) */
    float knowledge_confidence;    /**< 知识置信度 (0-1) */
    float knowledge_freshness;     /**< 知识新鲜度 (0-1) */
    float knowledge_consistency;   /**< 知识一致性 (0-1) */
} KnowledgeMetacognition;

/**
 * @brief 目标自省结构体
 */
typedef struct {
    char current_goal[256];        /**< 当前目标描述 */
    float goal_priority;           /**< 目标优先级 (0-1) */
    float goal_progress;           /**< 目标进展 (0-1) */
    float goal_feasibility;        /**< 目标可行性 (0-1) */
    int subgoal_count;             /**< 子目标数量 */
    float goal_confidence;         /**< 目标置信度 (0-1) */
} GoalIntrospection;

/**
 * @brief 学习进展结构体
 */
typedef struct {
    float learning_rate;           /**< 学习速率 (0-1) */
    float learning_efficiency;     /**< 学习效率 (0-1) */
    size_t training_samples;       /**< 训练样本数量 */
    size_t test_samples;           /**< 测试样本数量 */
    float training_accuracy;       /**< 训练准确率 (0-1) */
    float test_accuracy;           /**< 测试准确率 (0-1) */
    float generalization;          /**< 泛化能力 (0-1) */
    float evolution_progress;      /**< 进化进展 (0-1) */
} LearningProgress;

/**
 * @brief 自我认知配置结构体
 */
typedef struct {
    int enable_continuous_monitoring; /**< 是否启用持续监控 */
    float update_interval_sec;        /**< 更新间隔 (秒) */
    int enable_self_reflection;       /**< 是否启用自我反思 */
    int enable_capability_assessment; /**< 是否启用能力评估 */
    int enable_knowledge_tracking;    /**< 是否启用知识跟踪 */
    int enable_self_correction;       /**< 是否启用自我修正 (可关闭/启动) */
} SelfCognitionConfig;

/**
 * @brief 自我认知系统句柄
 */
typedef struct SelfCognitionSystem SelfCognitionSystem;

/**
 * @brief 创建自我认知系统
 * 
 * @param config 自我认知配置
 * @return SelfCognitionSystem* 自我认知系统句柄，失败返回NULL
 */
SelfCognitionSystem* self_cognition_create(const SelfCognitionConfig* config);

/**
 * @brief 释放自我认知系统
 * 
 * @param system 自我认知系统句柄
 */
void self_cognition_free(SelfCognitionSystem* system);

/**
 * @brief 更新自我认知状态
 * 
 * @param system 自我认知系统句柄
 * @param dimension 要更新的认知维度
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_update(SelfCognitionSystem* system, SelfCognitionDimension dimension);

/**
 * @brief 获取系统状态
 * 
 * @param system 自我认知系统句柄
 * @param status 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_status(SelfCognitionSystem* system, CognitionSystemStatus* status);

/**
 * @brief 获取能力评估
 * 
 * @param system 自我认知系统句柄
 * @param assessment 评估输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_capability(SelfCognitionSystem* system, CapabilityAssessment* assessment);

/**
 * @brief 获取知识元认知
 * 
 * @param system 自我认知系统句柄
 * @param metacognition 元认知输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_knowledge(SelfCognitionSystem* system, KnowledgeMetacognition* metacognition);

/**
 * @brief 获取目标自省
 * 
 * @param system 自我认知系统句柄
 * @param introspection 自省输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_goal(SelfCognitionSystem* system, GoalIntrospection* introspection);

/**
 * @brief 获取学习进展
 * 
 * @param system 自我认知系统句柄
 * @param progress 进展输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_learning(SelfCognitionSystem* system, LearningProgress* progress);

/**
 * @brief 执行自我反思
 * 
 * @param system 自我认知系统句柄
 * @param reflection 反思结果输出缓冲区
 * @param max_reflection_size 最大反思结果大小
 * @return int 成功返回反思结果大小，失败返回-1
 */
int self_cognition_reflect(SelfCognitionSystem* system, char* reflection, size_t max_reflection_size);

/**
 * @brief 评估自身限制
 * 
 * @param system 自我认知系统句柄
 * @param limitations 限制描述输出缓冲区
 * @param max_limitations_size 最大限制描述大小
 * @return int 成功返回限制描述大小，失败返回-1
 */
int self_cognition_assess_limitations(SelfCognitionSystem* system, char* limitations, size_t max_limitations_size);

/**
 * @brief 生成自我改进建议
 * 
 * @param system 自我认知系统句柄
 * @param suggestions 建议输出缓冲区
 * @param max_suggestions_size 最大建议大小
 * @return int 成功返回建议大小，失败返回-1
 */
int self_cognition_generate_suggestions(SelfCognitionSystem* system, char* suggestions, size_t max_suggestions_size);

/**
 * @brief 获取自我认知配置
 * 
 * @param system 自我认知系统句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_config(const SelfCognitionSystem* system, SelfCognitionConfig* config);

/**
 * @brief 设置自我认知配置
 * 
 * @param system 自我认知系统句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_set_config(SelfCognitionSystem* system, const SelfCognitionConfig* config);

/**
 * @brief 重置自我认知系统
 * 
 * @param system 自我认知系统句柄
 */
void self_cognition_reset(SelfCognitionSystem* system);

/**
 * @brief 决策类型枚举
 */
typedef enum {
    DECISION_EXPLORE = 0,           /**< 探索决策 - 探索新知识或环境 */
    DECISION_EXPLOIT = 1,           /**< 利用决策 - 利用现有知识 */
    DECISION_LEARN = 2,             /**< 学习决策 - 启动学习过程 */
    DECISION_PLAN = 3,              /**< 规划决策 - 生成新规划 */
    DECISION_ADAPT = 4,             /**< 适应决策 - 适应环境变化 */
    DECISION_REST = 5               /**< 休息决策 - 降低活动水平 */
} CogDecisionType;

/**
 * @brief 决策结果结构体
 */
typedef struct {
    CogDecisionType type;              /**< 决策类型 */
    char description[256];          /**< 决策描述 */
    float confidence;               /**< 决策置信度 (0-1) */
    float expected_value;           /**< 期望价值 */
    float risk_level;               /**< 风险等级 (0-1) */
    int estimated_duration_sec;     /**< 预计持续时间 (秒) */
} SelfDecisionResult;

/**
 * @brief 执行状态枚举
 */
typedef enum {
    EXECUTION_PENDING = 0,          /**< 执行待开始 */
    EXECUTION_RUNNING = 1,          /**< 执行中 */
    EXECUTION_PAUSED = 2,           /**< 执行暂停 */
    EXECUTION_COMPLETED = 3,        /**< 执行完成 */
    EXECUTION_FAILED = 4,           /**< 执行失败 */
    EXECUTION_CANCELLED = 5         /**< 执行取消 */
} ExecutionStatus;

/**
 * @brief 执行状态结构体
 */
typedef struct {
    ExecutionStatus status;         /**< 执行状态 */
    float progress;                 /**< 执行进度 (0-1) */
    char feedback[512];             /**< 执行反馈 */
    int elapsed_time_sec;           /**< 已用时间 (秒) */
    int remaining_time_sec;         /**< 剩余时间 (秒) */
} ExecutionState;

/**
 * @brief 基于当前状态和目标做出决策
 * 
 * @param system 自我认知系统句柄
 * @param goal_priority 目标优先级 (0-1)
 * @param risk_tolerance 风险容忍度 (0-1)
 * @param decision 决策输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_make_decision(SelfCognitionSystem* system, 
                                 float goal_priority,
                                 float risk_tolerance,
                                 SelfDecisionResult* decision);

/**
 * @brief 执行决策
 * 
 * @param system 自我认知系统句柄
 * @param decision 要执行的决策
 * @param execution_state 执行状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_execute_decision(SelfCognitionSystem* system,
                                    const SelfDecisionResult* decision,
                                    ExecutionState* execution_state);

/**
 * @brief 监控执行状态
 * 
 * @param system 自我认知系统句柄
 * @param execution_state 执行状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_monitor_execution(SelfCognitionSystem* system,
                                     ExecutionState* execution_state);

/**
 * @brief 停止当前执行
 * 
 * @param system 自我认知系统句柄
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_stop_execution(SelfCognitionSystem* system);

/**
 * @brief 获取执行历史
 * 
 * @param system 自我认知系统句柄
 * @param history 历史记录输出缓冲区
 * @param max_history_entries 最大历史记录条目数
 * @return int 成功返回历史记录条目数，失败返回-1
 */
int self_cognition_get_execution_history(SelfCognitionSystem* system,
                                         ExecutionState* history,
                                         size_t max_history_entries);

/* === 自我意识系统API（用于测试兼容性） === */

/**
 * @brief 目标优先级枚举
 */
typedef enum {
    COG_GOAL_PRIORITY_LOW = 0,
    COG_GOAL_PRIORITY_MEDIUM = 1,
    COG_GOAL_PRIORITY_HIGH = 2,
    COG_GOAL_PRIORITY_CRITICAL = 3
} SelfCognitionGoalPriority;

/**
 * @brief 目标结构体
 */
typedef struct {
    char goal_id[64];              /**< 目标ID */
    char description[256];         /**< 目标描述 */
    SelfCognitionGoalPriority priority;         /**< 目标优先级 */
    time_t deadline;               /**< 截止时间 */
} SelfCognitionGoal;

/**
 * @brief 反思结果结构体
 */
typedef struct {
    int reflection_depth;          /**< 反思深度 */
    size_t improvements_identified; /**< 发现的改进点数量 */
    char* reflection_summary;      /**< 反思摘要 */
} ReflectionResult;

/**
 * @brief 计划结果结构体
 */
typedef struct {
    size_t step_count;             /**< 计划步骤数 */
    float estimated_completion_time; /**< 预计完成时间（秒） */
    char* plan_summary;            /**< 计划摘要 */
} CognitionPlanResult;

/**
 * @brief 自我意识系统配置
 */
typedef struct {
    int enable_continuous_monitoring; /**< 启用持续监控 */
    int enable_error_analysis;        /**< 启用错误分析 */
} SelfAwarenessConfig;

/**
 * @brief 错误严重程度枚举
 */
typedef enum {
    ERROR_SEVERITY_LOW = 0,
    ERROR_SEVERITY_MEDIUM = 1,
    ERROR_SEVERITY_HIGH = 2,
    ERROR_SEVERITY_CRITICAL = 3
} ErrorSeverity;

/**
 * @brief 系统错误结构体
 */
typedef struct {
    int error_code;                 /**< 错误代码 */
    char* error_message;            /**< 错误消息 */
    ErrorSeverity severity;         /**< 错误严重程度 */
    time_t timestamp;               /**< 错误时间戳 */
} SelfCognitionSystemError;

/**
 * @brief 自我意识系统句柄（不透明指针，内部实现在self_cognition.c中）
 */
typedef struct SelfAwarenessSystem SelfAwarenessSystem;



/**
 * @brief 错误分析结果结构体
 */
typedef struct {
    size_t root_causes_identified;  /**< 识别的根本原因数量 */
    size_t solutions_proposed;      /**< 提出的解决方案数量 */
    char* analysis_summary;         /**< 分析摘要 */
} ErrorAnalysisResult;



/**
 * @brief 创建自我意识系统
 * 
 * @param config 系统配置
 * @return SelfAwarenessSystem* 系统句柄，失败返回NULL
 */
SelfAwarenessSystem* self_awareness_system_create(const SelfAwarenessConfig* config);

/**
 * @brief 释放自我意识系统
 * 
 * @param system 系统句柄
 */
void self_awareness_system_free(SelfAwarenessSystem* system);

/**
 * @brief 执行自我反思
 * 
 * @param system 系统句柄
 * @param reflection_prompt 反思提示
 * @return ReflectionResult* 反思结果，失败返回NULL
 */
ReflectionResult* self_awareness_reflect(SelfAwarenessSystem* system, const char* reflection_prompt);

/**
 * @brief 释放反思结果
 * 
 * @param result 反思结果
 */
void reflection_result_free(ReflectionResult* result);

/**
 * @brief 规划目标
 * 
 * @param system 系统句柄
 * @param goal 目标
 * @return CognitionPlanResult* 计划结果，失败返回NULL
 */
CognitionPlanResult* self_awareness_plan_goal(SelfAwarenessSystem* system, const void* goal);

/**
 * @brief 销毁自我意识系统（self_awareness_system_free的别名）
 * 
 * @param system 系统句柄
 */
#define self_awareness_system_destroy self_awareness_system_free

/**
 * @brief 分析系统错误
 * 
 * @param system 系统句柄
 * @param error 系统错误
 * @return ErrorAnalysisResult* 错误分析结果，失败返回NULL
 */
ErrorAnalysisResult* self_awareness_analyze_error(SelfAwarenessSystem* system, const void* error);

/**
 * @brief 释放错误分析结果
 * 
 * @param result 错误分析结果
 */
void error_analysis_result_free(ErrorAnalysisResult* result);

/**
 * @brief 释放计划结果
 * 
 * @param result 计划结果
 */
void plan_result_free(CognitionPlanResult* result);

/* ============================
 * 自我修正系统API
 * ============================ */

/**
 * @brief 自我修正类型枚举
 */
typedef enum {
    SELF_CORRECTION_PARAMETER = 0,        /**< 参数修正 - 调整模型参数 */
    SELF_CORRECTION_ALGORITHM = 1,        /**< 算法修正 - 调整计算算法 */
    SELF_CORRECTION_ARCHITECTURE = 2,     /**< 架构修正 - 调整网络架构 */
    SELF_CORRECTION_POLICY = 3,           /**< 策略修正 - 调整决策策略 */
    SELF_CORRECTION_MEMORY = 4,           /**< 内存修正 - 优化内存使用 */
    SELF_CORRECTION_PERFORMANCE = 5       /**< 性能修正 - 优化性能表现 */
} SelfCorrectionType;

/**
 * @brief 自我修正结果结构体
 */
typedef struct {
    SelfCorrectionType type;              /**< 修正类型 */
    char description[512];                /**< 修正描述 */
    float correction_strength;            /**< 修正强度 (0-1) */
    float expected_improvement;           /**< 预期改进 (0-1) */
    float confidence;                     /**< 修正置信度 (0-1) */
    time_t correction_time;               /**< 修正时间 */
    int correction_id;                    /**< 修正ID（唯一标识） */
    float weight_change_l2; /**< 权重实际L2变更量 */
} SelfCorrectionResult;

/**
 * @brief 执行自我修正
 * 
 * @param system 自我认知系统句柄
 * @param issue_description 问题描述
 * @param issue_severity 问题严重程度 (0-1)
 * @param correction_result 修正结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_perform_correction(SelfCognitionSystem* system,
                                     const char* issue_description,
                                     float issue_severity,
                                     SelfCorrectionResult* correction_result);

/**
 * @brief 获取自我修正历史
 * 
 * @param system 自我认知系统句柄
 * @param history 历史记录输出缓冲区
 * @param max_history_entries 最大历史记录条目数
 * @return int 成功返回历史记录条目数，失败返回-1
 */
int self_cognition_get_correction_history(SelfCognitionSystem* system,
                                         SelfCorrectionResult* history,
                                         size_t max_history_entries);

/**
 * @brief 评估自我修正效果
 * 
 * @param system 自我认知系统句柄
 * @param correction_id 修正ID
 * @param effectiveness 效果评估输出缓冲区 (0-1)
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_assess_correction_effectiveness(SelfCognitionSystem* system,
                                                  int correction_id,
                                                  float* effectiveness);

/**
 * @brief 启用或禁用自我修正功能
 * 
 * @param system 自我认知系统句柄
 * @param enable 启用标志 (1=启用, 0=禁用)
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_enable_self_correction(SelfCognitionSystem* system, int enable);

/* ============================
 * 深度自我认知系统API
 * ============================ */

/**
 * @brief 自我模型配置结构体
 */
typedef struct {
    size_t state_encoding_size;      /**< 状态编码维度 */
    size_t prediction_horizon;       /**< 预测时间步数 */
    float learning_rate;             /**< 学习率 */
    float regularization_strength;   /**< 正则化强度 */
    int enable_online_learning;      /**< 是否启用在线学习 */
    int enable_uncertainty_modeling; /**< 是否启用不确定性建模 */
    int enable_counterfactual;       /**< 是否启用反事实推理 */
} SelfModelConfig;

/**
 * @brief 自我模型状态编码
 */
typedef struct {
    float* encoded_state;            /**< 编码后的状态向量 */
    size_t state_size;               /**< 状态向量大小 */
    float encoding_confidence;       /**< 编码置信度 (0-1) */
    time_t encoding_time;            /**< 编码时间 */
} SelfModelState;

/**
 * @brief 自我预测结果
 */
typedef struct {
    float* predicted_states;         /**< 预测状态序列 */
    size_t prediction_steps;         /**< 预测步数 */
    float* prediction_confidences;   /**< 预测置信度序列 */
    float* uncertainty_estimates;    /**< 不确定性估计序列 */
    float overall_confidence;        /**< 整体置信度 (0-1) */
    float prediction_error;          /**< 预测误差 */
} SelfPredictionResult;

/**
 * @brief 元认知推理类型
 */
typedef enum {
    METACOGNITION_REFLECTIVE = 0,    /**< 反思性推理 - 分析过去行为 */
    METACOGNITION_PROSPECTIVE = 1,   /**< 前瞻性推理 - 预测未来状态 */
    METACOGNITION_EVALUATIVE = 2,    /**< 评估性推理 - 评估当前状态 */
    METACOGNITION_REGULATIVE = 3,    /**< 调节性推理 - 调节行为策略 */
    METACOGNITION_CREATIVE = 4       /**< 创造性推理 - 生成新解决方案 */
} MetacognitionType;

/**
 * @brief 元认知推理结果
 */
typedef struct {
    MetacognitionType type;          /**< 推理类型 */
    char* reasoning_process;         /**< 推理过程描述 */
    size_t reasoning_length;         /**< 推理过程长度 */
    char* conclusions;               /**< 推理结论 */
    size_t conclusions_length;       /**< 推理结论长度 */
    float reasoning_confidence;      /**< 推理置信度 (0-1) */
    char* improvement_suggestions;   /**< 改进建议 */
    size_t suggestions_length;       /**< 改进建议长度 */
} MetacognitionResult;

/**
 * @brief 深度反思层级
 */
typedef enum {
    REFLECTION_LEVEL_SURFACE = 0,    /**< 表层反思 - 描述性分析 */
    REFLECTION_LEVEL_PRACTICAL = 1,  /**< 实践反思 - 操作性分析 */
    REFLECTION_LEVEL_PROCESS = 2,    /**< 过程反思 - 过程性分析 */
    REFLECTION_LEVEL_PREMISE = 3,    /**< 前提反思 - 基础假设分析 */
    REFLECTION_LEVEL_TRANSFORMATIVE = 4 /**< 变革反思 - 根本性重构 */
} ReflectionLevel;

/**
 * @brief 深度反思结果
 */
typedef struct {
    ReflectionLevel level;           /**< 反思层级 */
    char* reflection_content;        /**< 反思内容 */
    size_t content_length;           /**< 内容长度 */
    char* insights_gained;           /**< 获得的洞见 */
    size_t insights_length;          /**< 洞见长度 */
    char* action_plans;              /**< 行动计划 */
    size_t plans_length;             /**< 计划长度 */
    float reflection_depth;          /**< 反思深度评分 (0-1) */
    float transformative_potential;  /**< 变革潜力评分 (0-1) */
} DeepReflectionResult;

/**
 * @brief 创建深度自我认知系统
 * 
 * @param base_system 基础自我认知系统
 * @param model_config 自我模型配置
 * @return SelfCognitionSystem* 增强的自我认知系统，失败返回NULL
 */
SelfCognitionSystem* deep_self_cognition_create(SelfCognitionSystem* base_system, 
                                                const SelfModelConfig* model_config);

/**
 * @brief 训练自我模型
 * 
 * @param system 自我认知系统句柄
 * @param training_data 训练数据（状态序列）
 * @param data_size 数据大小
 * @param epochs 训练轮数
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_train_model(SelfCognitionSystem* system, 
                              const float* training_data, 
                              size_t data_size, 
                              int epochs);

/**
 * @brief 编码当前系统状态到自我模型
 * 
 * @param system 自我认知系统句柄
 * @param encoded_state 编码状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_encode_state(SelfCognitionSystem* system, 
                               SelfModelState* encoded_state);

/**
 * @brief 预测未来系统状态
 * 
 * @param system 自我认知系统句柄
 * @param steps 预测步数
 * @param prediction 预测结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_predict_future(SelfCognitionSystem* system, 
                                 int steps, 
                                 SelfPredictionResult* prediction);

/**
 * @brief 执行元认知推理
 * 
 * @param system 自我认知系统句柄
 * @param reasoning_type 推理类型
 * @param context 推理上下文
 * @param result 推理结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_metacognitive_reasoning(SelfCognitionSystem* system,
                                          MetacognitionType reasoning_type,
                                          const char* context,
                                          MetacognitionResult* result);

/**
 * @brief 执行深度自我反思
 * 
 * @param system 自我认知系统句柄
 * @param reflection_level 反思层级
 * @param reflection_prompt 反思提示
 * @param result 反思结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_deep_reflection(SelfCognitionSystem* system,
                                  ReflectionLevel reflection_level,
                                  const char* reflection_prompt,
                                  DeepReflectionResult* result);

/**
 * @brief 生成自我改进计划
 * 
 * @param system 自我认知系统句柄
 * @param issue_description 问题描述
 * @param improvement_plan 改进计划输出缓冲区
 * @param max_plan_size 最大计划大小
 * @return int 成功返回计划大小，失败返回-1
 */
int self_cognition_generate_improvement_plan(SelfCognitionSystem* system,
                                            const char* issue_description,
                                            char* improvement_plan,
                                            size_t max_plan_size);

/**
 * @brief 【自我编程闭环】认知评估能力缺口后委托编程模块生成代码
 *
 * 桥接 cognitive→programming 的完整闭环:
 *   Intent → synthesize_code → compile → sandbox_execute → self_improve_code
 *
 * 调用后系统元模型根据 learning_signal 更新能力评分。
 *
 * @param system 认知系统句柄
 * @param intent 编程意图(需求描述+函数签名+I/O示例)
 * @param result 输出编程闭环结果(调用者用 programming_closure_free 释放)
 * @return 0=成功, -1=失败
 */
int self_cognition_delegate_programming(SelfCognitionSystem* system,
                                        void* intent,
                                        void* result);

/**
 * @brief 检测"需要代码生成"的认知状态并自主触发
 *
 * 扫描当前能力评分，如果某个维度低于阈值，
 * 自动创建 ProgrammingIntent 并执行整个闭环。
 *
 * @param system 认知系统句柄
 * @param trigger_threshold 触发阈值(0.0-1.0, 能力评分低于此值触发)
 * @param plan_output 输出的改进计划文本
 * @param plan_size 计划缓冲区大小
 * @return 0=无触发, >0=触发的闭环数量, -1=失败
 */
int self_cognition_autonomous_code_generation(SelfCognitionSystem* system,
                                              float trigger_threshold,
                                              char* plan_output,
                                              size_t plan_size);

/**
 * @brief 评估自我认知准确性
 * 
 * @param system 自我认知系统句柄
 * @param accuracy 准确性评分输出缓冲区 (0-1)
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_assess_accuracy(SelfCognitionSystem* system, float* accuracy);

/**
 * @brief 释放元认知推理结果
 * 
 * @param result 推理结果
 */
void metacognition_result_free(MetacognitionResult* result);

/**
 * @brief 释放深度反思结果
 * 
 * @param result 反思结果
 */
void deep_reflection_result_free(DeepReflectionResult* result);

/**
 * @brief 释放自我预测结果
 * 
 * @param result 预测结果
 */
void self_prediction_result_free(SelfPredictionResult* result);

/**
 * @brief 释放自我模型状态
 * 
 * @param state 模型状态
 */
void self_model_state_free(SelfModelState* state);

/**
 * @brief 深度自我认知系统统计信息
 */
typedef struct {
    size_t state_history_size;          /**< 状态历史大小 */
    size_t state_history_capacity;      /**< 状态历史容量 */
    size_t metacognition_history_size;  /**< 元认知历史大小 */
    size_t reflection_results_size;     /**< 反思结果大小 */
    size_t prediction_history_size;     /**< 预测历史大小 */
    int model_training_epochs;          /**< 模型训练轮次 */
    float model_training_loss;          /**< 模型训练损失 */
    float model_accuracy;               /**< 模型准确性 */
    int is_model_trained;               /**< 模型是否已训练 */
    time_t last_model_update;           /**< 最后模型更新时间 */
    size_t encoded_state_size;          /**< 编码状态大小 */
    int is_model_initialized;           /**< 模型是否已初始化 */
    float state_history_usage;          /**< 状态历史使用率 (0-1) */
    float avg_state_value;              /**< 平均状态值 */
    time_t current_time;                /**< 当前时间 */
    double model_age_seconds;           /**< 模型年龄（秒） */
    float training_frequency;           /**< 训练频率（轮次/小时） */
} DeepSelfCognitionStats;

/* ============================
 * 连续身份跟踪系统API
 * ============================ */

/**
 * @brief 身份签名结构体
 * 
 * 唯一标识AGI自身体份的指纹，包含连续跟踪所需的全部信息。
 * 身份签名是系统内部状态的确定性哈希，确保身份可被持续跟踪。
 */
typedef struct {
    char identity_hash[128];         /**< 身份哈希值（SHA-256风格的64字节hex字符串） */
    time_t creation_time;            /**< 身份创建时间 */
    time_t last_update_time;         /**< 最后更新时间 */
    size_t update_count;             /**< 更新次数 */
    float identity_stability;        /**< 身份稳定性 (0-1)，1=完全稳定 */
    float identity_evolution_rate;   /**< 身份演化速率 (0-1)，0=无变化 */
    float continuity_score;          /**< 连续分数 (0-1)，1=完全连续 */
    size_t discontinuity_events;     /**< 不连续事件计数 */
    char last_discontinuity_reason[256]; /**< 最后不连续原因 */
    float core_identity_fingerprint[16]; /**< 核心身份指纹向量（16维） */
    float identity_confidence;       /**< 身份置信度 (0-1) */
    float self_consistency;          /**< 自我一致性 (0-1) */
} IdentitySignature;

/**
 * @brief 身份演化历史记录
 */
typedef struct {
    char snapshot_hash[128];         /**< 快照哈希 */
    time_t snapshot_time;            /**< 快照时间 */
    float stability_at_snapshot;     /**< 快照时稳定性 */
    float evolution_from_previous;   /**< 与前一次快照的演化程度 */
    float core_vector_delta[16];     /**< 核心向量变化 */
    char change_description[256];    /**< 变化描述 */
} IdentitySnapshot;

/**
 * @brief 获取当前身份签名
 * 
 * 计算并返回当前系统的确定性身份指纹。
 * 身份指纹基于系统内部状态（权重、配置、历史等）生成，
 * 即使在系统重启后也能保持一致性（只要内部状态一致）。
 * 
 * @param system 自我认知系统句柄
 * @param signature 身份签名输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_identity_signature(SelfCognitionSystem* system,
                                          IdentitySignature* signature);

/**
 * @brief 更新身份跟踪
 * 
 * 基于当前系统状态更新身份签名，跟踪身份演化。
 * 记录身份变化并检测潜在的身份不连续性。
 * 应在每次系统状态更新后调用。
 * 
 * @param system 自我认知系统句柄
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_update_identity(SelfCognitionSystem* system);

/**
 * @brief 检查身份连续性
 * 
 * 检测当前身份是否与历史身份保持连续。
 * 通过比较当前身份签名与历史快照的相似度判断。
 * 
 * @param system 自我认知系统句柄
 * @param continuity_score 连续性评分输出 (0-1)，1=完全连续
 * @param discontinuity_reason 如果不连续，输出原因描述
 * @param max_reason_size 原因缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_check_identity_continuity(SelfCognitionSystem* system,
                                             float* continuity_score,
                                             char* discontinuity_reason,
                                             size_t max_reason_size);

/**
 * @brief 获取身份演化指标
 * 
 * 返回身份的演化速率、稳定性、连续性和总体健康度指标。
 * 
 * @param system 自我认知系统句柄
 * @param evolution_rate 演化速率输出 (0-1)
 * @param stability 稳定性输出 (0-1)
 * @param self_consistency 自我一致性输出 (0-1)
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_identity_evolution(SelfCognitionSystem* system,
                                          float* evolution_rate,
                                          float* stability,
                                          float* self_consistency);

/**
 * @brief 获取身份历史快照数
 * 
 * @param system 自我认知系统句柄
 * @return size_t 快照数量，失败返回0
 */
size_t self_cognition_get_identity_snapshot_count(SelfCognitionSystem* system);

// ==================== 元认知系统集成接口 ====================

/**
 * @brief 执行元认知监控
 *
 * @param system 自我认知系统句柄
 * @param input_data 输入数据
 * @param data_size 数据大小
 * @param result 监控结果输出
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_metacognition_monitor(SelfCognitionSystem* system,
                                        const float* input_data, size_t data_size,
                                        MetacognitionMonitoringResult* result);

/**
 * @brief 获取元认知评估——桥接到决策引擎 (M-020修复)
 *
 * 返回最近一次元认知监控的评估结果，包含confidence/trend/requires_action等指标。
 * AGI认知循环调用此函数获取元认知洞见后，传入决策引擎用于调整决策权重、
 * 风险偏好和行动优先级。
 *
 * @param system 自我认知系统句柄
 * @param result 元认知监控结果输出（从内部缓存复制）
 * @return int 成功返回0（有缓存数据），返回-1表示无缓存数据（需先调用监控）
 */
int self_cognition_get_metacognition_assessment(SelfCognitionSystem* system,
                                                MetacognitionMonitoringResult* result);

/**
 * @brief 更新自我模型
 *
 * @param system 自我认知系统句柄
 * @param new_data 新数据
 * @param data_size 数据大小
 * @param ground_truth 真实值（可选）
 * @param truth_size 真实值大小
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_update_self_model(SelfCognitionSystem* system,
                                    const float* new_data, size_t data_size,
                                    const float* ground_truth, size_t truth_size);

/**
 * @brief 执行预测性自我认知
 *
 * @param system 自我认知系统句柄
 * @param current_state 当前状态
 * @param state_size 状态大小
 * @param prediction_horizon 预测时间范围（秒）
 * @param result 预测结果输出
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_predictive_self(SelfCognitionSystem* system,
                                  const float* current_state, size_t state_size,
                                  float prediction_horizon,
                                  PredictiveSelfResult* result);

/**
 * @brief 获取元认知自我模型状态
 *
 * @param system 自我认知系统句柄
 * @param state 自我模型状态输出
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_self_model_state(SelfCognitionSystem* system,
                                       MetacognitionModelState* state);

/**
 * @brief 执行客观理性自我评估（不涉及情感，纯系统状态分析）
 *
 * @param system 自我认知系统句柄
 * @param assessment_type 评估类型
 * @param assessment_result 评估结果输出缓冲区
 * @param result_size 结果缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_neutral_assessment(SelfCognitionSystem* system,
                                     int assessment_type,
                                     char* assessment_result,
                                     size_t result_size);

/**
 * @brief 设置液态神经网络实例
 *
 * 将液态神经网络（LNN）实例连接到自我认知系统，使认知功能
 * 能够利用LNN进行深度自我认知处理（如能力评估、状态预测等）。
 * 所有模态的统一状态演化通过同一个LNN连续动态系统完成。
 *
 * @param system 自我认知系统句柄
 * @param lnn 液态神经网络实例指针（可传NULL解除关联）
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_set_lnn_instance(SelfCognitionSystem* system, LNN* lnn);

/**
 * @brief P0-001: 设置架构控制器实例（用于运行时架构修正）
 * 
 * 将全局架构控制器引用注入到自我认知系统，
 * 使得架构修正（SELF_CORRECTION_ARCHITECTURE）能够
 * 通过 arch_controller_submit_change 真正执行网络结构变更。
 * 
 * @param system 自我认知系统句柄
 * @param arch_controller 架构控制器实例
 */
void self_cognition_set_arch_controller(SelfCognitionSystem* system, void* arch_controller);

/* ===================== 认知记忆系统 API（P1-03修复） ===================== */

typedef struct {
    time_t timestamp;
    uint32_t content_hash;
    float significance;
    int retrieval_count;
    float forgetting_factor;
    int consolidated;
} CognitiveMemoryFragment;

int self_cognition_memory_consolidate(SelfCognitionSystem* system);
int self_cognition_memory_retrieve(SelfCognitionSystem* system, const char* query,
                                    CognitiveMemoryFragment* results, int max_results);

/* ===================== 迭代式元认知循环 API（P1-04修复） ===================== */
int self_cognition_iterative_reflection(SelfCognitionSystem* system, int max_iterations);

/* ============================
 * AGI深度自我认知增强模块
 * ============================ */

/* ---- 心智理论（Theory of Mind） ---- */

#define SELFLNN_MAX_TRACKED_AGENTS 16
#define SELFLNN_AGENT_STATE_DIM 8
#define SELFLNN_TOM_OBS_HISTORY 32

/**
 * @brief 其他智能体心智状态
 */
typedef struct {
    float belief[SELFLNN_AGENT_STATE_DIM];         /**< 信念状态 */
    float intention[SELFLNN_AGENT_STATE_DIM];      /**< 意图 */
    float desire[SELFLNN_AGENT_STATE_DIM];         /**< 欲望/目标 */
    float estimated_capability[SELFLNN_AGENT_STATE_DIM]; /**< 能力估计 */
    float knowledge_certainty;                     /**< 知识确定性 (0-1) */
    float cooperativeness;                         /**< 合作倾向 (0-1) */
    float trust_level;                             /**< 信任水平 (0-1) */
    float reciprocity;                             /**< 互惠程度 (0-1) */
    float observed_actions[SELFLNN_TOM_OBS_HISTORY * 4]; /**< 观测动作历史 */
    size_t obs_count;                              /**< 观测计数 */
    int recursive_level;                           /**< 递归心智层级 (0-3) */
    time_t last_observed;                          /**< 最后观测时间 */
    int is_active;                                 /**< 是否活跃 */
    char agent_id[64];                             /**< 智能体ID */
} AgentMentalState;

/**
 * @brief 心智理论状态
 */
typedef struct {
    AgentMentalState agents[SELFLNN_MAX_TRACKED_AGENTS]; /**< 跟踪的智能体数组 */
    size_t active_agent_count;                          /**< 活跃智能体数 */
    int max_recursive_level;                            /**< 最大递归层级 */
    float theory_update_rate;                           /**< 理论更新率 (0-1) */
    int tom_enabled;                                    /**< 是否启用 */
} TheoryOfMindState;

/**
 * @brief 初始化心智理论系统
 */
int self_cognition_init_theory_of_mind(SelfCognitionSystem* system);

/**
 * @brief 注册/更新一个智能体的心智模型
 */
int self_cognition_update_agent_model(SelfCognitionSystem* system,
                                      const char* agent_id,
                                      const float* observed_action, size_t action_dim,
                                      const float* context_state, size_t state_dim,
                                      float interaction_feedback);

/**
 * @brief 预测指定智能体的下一步行为
 */
int self_cognition_predict_agent_action(SelfCognitionSystem* system,
                                        const char* agent_id,
                                        float* predicted_action, size_t action_dim,
                                        float* confidence);

/**
 * @brief 获取指定智能体的心智状态
 */
int self_cognition_get_agent_mental_state(SelfCognitionSystem* system,
                                          const char* agent_id,
                                          AgentMentalState* state_out);

/**
 * @brief 执行递归心智推理
 * "我认为你认为我认为..."
 */
int self_cognition_recursive_mind_reading(SelfCognitionSystem* system,
                                          const char* agent_id,
                                          int recursion_depth,
                                          float* inferred_belief, size_t belief_dim);

/* ---- 因果自我模型（Causal Self-Model） ---- */

#define SELFLNN_MAX_CAUSAL_LINKS 256
#define SELFLNN_CAUSAL_STATE_DIM 16

/**
 * @brief 因果链
 */
typedef struct {
    float cause_state[SELFLNN_CAUSAL_STATE_DIM];     /**< 原因状态 */
    float action_taken[SELFLNN_CAUSAL_STATE_DIM];    /**< 执行动作 */
    float effect_state[SELFLNN_CAUSAL_STATE_DIM];    /**< 效果状态 */
    float causal_strength;                           /**< 因果强度 (0-1) */
    float confidence;                                /**< 置信度 (0-1) */
    int occurrence_count;                            /**< 出现次数 */
    float avg_effect_magnitude;                      /**< 平均影响幅度 */
    time_t last_observed;                            /**< 最后观测时间 */
    int is_active;                                   /**< 是否活跃 */
    char cause_label[64];                            /**< 原因标签 */
    char effect_label[64];                           /**< 效果标签 */
} CausalLink;

/**
 * @brief 因果自我模型状态
 */
typedef struct {
    CausalLink links[SELFLNN_MAX_CAUSAL_LINKS];      /**< 因果链接数组 */
    size_t link_count;                               /**< 链接数 */
    size_t write_index;                              /**< 循环写入索引 */
    int causal_model_enabled;                        /**< 是否启用 */
    float learning_rate;                             /**< 学习率 (0-1) */
} CausalSelfModel;

/**
 * @brief 初始化因果自我模型
 */
int self_cognition_init_causal_model(SelfCognitionSystem* system);

/**
 * @brief 记录一个因果事件
 * 观测到 (state_before, action) -> state_after 的因果关系
 */
int self_cognition_record_causal_event(SelfCognitionSystem* system,
                                       const float* state_before, size_t state_dim,
                                       const float* action, size_t action_dim,
                                       const float* state_after, size_t effect_dim);

/**
 * @brief 预测执行动作后的效果
 */
int self_cognition_predict_effect(SelfCognitionSystem* system,
                                  const float* state_before, size_t state_dim,
                                  const float* action, size_t action_dim,
                                  float* predicted_effect, size_t effect_dim,
                                  float* confidence);

/**
 * @brief 识别给定效果的最可能原因
 */
int self_cognition_identify_cause(SelfCognitionSystem* system,
                                  const float* current_state, size_t state_dim,
                                  const float* effect, size_t effect_dim,
                                  char* cause_description, size_t desc_size);

/**
 * @brief 执行反事实推理
 * "如果当时执行了不同的动作，现在会怎样？"
 */
int self_cognition_counterfactual_reasoning(SelfCognitionSystem* system,
                                            const float* current_state, size_t state_dim,
                                            const float* counterfactual_action, size_t action_dim,
                                            float* imagined_effect, size_t effect_dim);

/**
 * @brief 多智能体反事实模拟
 *
 * 在指定情景下，对多个智能体的信念/欲望/意图进行反事实推演。
 * 替换目标智能体的动作后，重新模拟所有智能体的交互结果。
 *
 * @param system 自我认知系统句柄
 * @param target_agent_id 目标智能体ID
 * @param counterfactual_action 反事实动作向量
 * @param action_dim 动作维度
 * @param context_state 当前环境上下文状态
 * @param context_dim 上下文维度
 * @param simulated_outcome [out] 模拟结果状态
 * @param outcome_dim 结果维度
 * @param affected_agents [out] 受影响的其他智能体ID列表
 * @param max_affected 最大受影响智能体数量
 * @param affected_count [out] 实际受影响智能体数量
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_multi_agent_counterfactual_simulation(
    SelfCognitionSystem* system,
    const char* target_agent_id,
    const float* counterfactual_action, size_t action_dim,
    const float* context_state, size_t context_dim,
    float* simulated_outcome, size_t outcome_dim,
    char (*affected_agents)[64], size_t max_affected,
    size_t* affected_count);

/**
 * @brief 反事实情景树探索
 *
 * 对给定情景，生成多个反事实分支并逐一模拟，
 * 返回各分支的潜在结果和置信度，用于决策评估。
 *
 * @param system 自我认知系统句柄
 * @param current_state 当前状态
 * @param state_dim 状态维度
 * @param candidate_actions 候选动作矩阵 [n_candidates * action_dim]
 * @param candidate_descriptions 候选动作描述字符串数组
 * @param n_candidates 候选动作数量
 * @param action_dim 动作维度
 * @param outcomes [out] 模拟结果矩阵 [n_candidates * outcome_dim]
 * @param confidences [out] 各结果置信度数组
 * @param outcome_dim 结果维度
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_explore_counterfactual_scenarios(
    SelfCognitionSystem* system,
    const float* current_state, size_t state_dim,
    const float* candidate_actions, size_t n_candidates, size_t action_dim,
    const char* const* candidate_descriptions,
    float* outcomes, float* confidences, size_t outcome_dim);

/* ---- 自我叙事（Self-Narrative） ---- */

#define SELFLNN_MAX_NARRATIVE_EVENTS 128
#define SELFLNN_NARRATIVE_DESC_LEN 256

/**
 * @brief 自我叙事事件类型
 */
typedef enum {
    NARRATIVE_EVENT_MILESTONE = 0,      /**< 里程碑 */
    NARRATIVE_EVENT_LEARNING = 1,       /**< 学习事件 */
    NARRATIVE_EVENT_FAILURE = 2,        /**< 失败事件 */
    NARRATIVE_EVENT_ADAPTATION = 3,     /**< 适应事件 */
    NARRATIVE_EVENT_IDENTITY_SHIFT = 4, /**< 身份转变 */
    NARRATIVE_EVENT_GOAL_CHANGE = 5,    /**< 目标变化 */
    NARRATIVE_EVENT_CAPABILITY_JUMP = 6 /**< 能力跃迁 */
} NarrativeEventType;

/**
 * @brief 自我叙事事件
 */
typedef struct {
    NarrativeEventType event_type;                  /**< 事件类型 */
    time_t timestamp;                               /**< 时间戳 */
    float significance;                             /**< 重要性 (0-1) */
    char description[SELFLNN_NARRATIVE_DESC_LEN];   /**< 事件描述 */
    float state_before;                             /**< 之前状态值 */
    float state_after;                              /**< 之后状态值 */
    char causal_context[SELFLNN_NARRATIVE_DESC_LEN]; /**< 因果背景 */
} NarrativeEvent;

/**
 * @brief 自我叙事状态
 */
typedef struct {
    NarrativeEvent events[SELFLNN_MAX_NARRATIVE_EVENTS]; /**< 事件数组 */
    size_t event_count;                                  /**< 事件数 */
    size_t write_index;                                  /**< 循环写入索引 */
    int narrative_enabled;                               /**< 是否启用 */
    char current_arc[256];                               /**< 当前叙事弧线 */
    float narrative_coherence;                           /**< 叙事连贯性 (0-1) */
    size_t reflection_triggers;                          /**< 触发反思次数 */
} SelfNarrativeState;

/**
 * @brief 初始化自我叙事系统
 */
int self_cognition_init_narrative(SelfCognitionSystem* system);

/**
 * @brief 记录一个自我叙事事件
 */
int self_cognition_record_narrative_event(SelfCognitionSystem* system,
                                          NarrativeEventType event_type,
                                          const char* description,
                                          float significance,
                                          float state_before, float state_after);

/**
 * @brief 生成完整的自我叙事文本
 */
int self_cognition_generate_narrative(SelfCognitionSystem* system,
                                      char* narrative_text, size_t max_text_size);

/**
 * @brief 识别自我发展中的关键转折点
 */
int self_cognition_identify_turning_points(SelfCognitionSystem* system,
                                           NarrativeEvent* turning_points,
                                           size_t max_points,
                                           size_t* point_count);

/**
 * @brief 计算自我叙事连贯性评分
 */
float self_cognition_assess_narrative_coherence(SelfCognitionSystem* system);

/* ============================
 * 统一功能开关API（增强T3）
 * ============================ */

/**
 * @brief AGI核心功能类型枚举
 *
 * 统一标识系统中所有可运行时启停的核心功能。
 */
typedef enum {
    FEATURE_SELF_DECISION = 0,           /**< 自我决策 */
    FEATURE_AUTONOMOUS_EXECUTION = 1,    /**< 自主执行 */
    FEATURE_SELF_LEARNING = 2,           /**< 自我学习 */
    FEATURE_SELF_EVOLUTION = 3,          /**< 自我演化进化 */
    FEATURE_SELF_CORRECTION = 4,         /**< 自我修正 */
    FEATURE_IMITATION_LEARNING = 5,      /**< 模仿学习 */
    FEATURE_MULTI_ROBOT = 6,             /**< 多机器人控制 */
    FEATURE_GPU_COMPUTING = 7,           /**< GPU计算 */
    FEATURE_GPU_TRAINING = 8,            /**< GPU训练 */
    FEATURE_CFC_AGI = 9,                /**< CfC全模态AGI */
    FEATURE_CONCURRENCY = 10,           /**< 高并发 */
    FEATURE_KNOWLEDGE_BASE = 11,        /**< 知识库 */
    FEATURE_DIALOGUE = 12,              /**< 对话能力 */
    FEATURE_VISION = 13,                /**< 视觉识别 */
    FEATURE_SLAM = 14,                  /**< SLAM定位 */
    FEATURE_TTS = 15,                   /**< 语音合成 */
    FEATURE_METACOGNITION = 16,         /**< 元认知 */
    FEATURE_PLANNING = 17,              /**< 计划能力 */
    FEATURE_COUNT = 18                  /**< 功能总数 */
} FeatureType;

/**
 * @brief 统一启用或禁用指定核心功能
 *
 * 通过FeatureType枚举统一管理所有核心功能的运行时启停。
 * 替换各个模块分散的enable/disable开关，提供单一入口。
 * 持久化到系统配置中，重启后保持状态。
 *
 * @param system 自我认知系统句柄
 * @param feature 要启停的功能类型
 * @param enable 1=启用，0=禁用
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_enable_feature(SelfCognitionSystem* system,
                                  FeatureType feature,
                                  int enable);

/**
 * @brief 查询指定功能当前是否启用
 *
 * @param system 自我认知系统句柄
 * @param feature 要查询的功能类型
 * @return int 启用返回1，禁用返回0，失败返回-1
 */
int self_cognition_is_feature_enabled(SelfCognitionSystem* system,
                                      FeatureType feature);

/**
 * @brief 获取所有功能的启用状态
 *
 * 将所有功能的当前启用状态写入status数组。
 * status数组长度必须为FEATURE_COUNT。
 *
 * @param system 自我认知系统句柄
 * @param status 状态输出数组（长度FEATURE_COUNT），每个元素1=启用，0=禁用
 * @return int 成功返回0，失败返回-1
 */
int self_cognition_get_all_feature_states(SelfCognitionSystem* system,
                                          int* status);

/**
 * @brief 自我认知验证闭环反馈
 *
 * 将自我认知系统的评价结果与实际系统性能进行对比验证，
 * 生成认知准确性分数，用于持续改进自我认知模型。
 * 这是自我认知系统的核心验证机制。
 *
 * @param system 自我认知系统句柄
 * @param actual_performance 实际性能指标数组(感知/推理/规划/学习/执行/知识/泛化)
 * @param num_metrics 性能指标数量(≤7)
 * @param feedback 输出验证反馈报告(可NULL)
 * @param max_feedback_size 报告缓冲区大小
 * @return 认知准确性(0-1)，-1失败
 */
float self_cognition_verify_feedback(SelfCognitionSystem* system,
                                      const float* actual_performance,
                                      size_t num_metrics,
                                      char* feedback,
                                      size_t max_feedback_size);

/* 能力开关（P3.3） */
void self_cognition_system_enable(SelfCognitionSystem* system);
void self_cognition_system_disable(SelfCognitionSystem* system);
int self_cognition_system_is_enabled(const SelfCognitionSystem* system);

/* ============================================================================
 *: LNN训练就绪检查
 *
 * 检查全局共享LNN是否已完成训练（如加载了检查点或经过至少一轮训练）。
 * 未训练的LNN（随机权重）会导致自我认知的评估/反思/修正产生无意义的
 * 随机噪声输出，可能误导错误的自我修正决策。
 *
 * 检查逻辑：
 *   1. self_model_lnn 是否已训练（is_model_trained标志）
 *   2. 共享LNN是否经过前向传播（forward_count > 0）
 *   3. 共享LNN的平均激活度是否非零（avg_activation ≠ 0.0）
 *
 * @param system 自我认知系统句柄
 * @return 1=LNN已就绪可安全使用, 0=LNN未就绪需保守模式
 * ============================================================================ */
int self_cognition_is_lnn_ready(SelfCognitionSystem* system);

/* ============================================================================
 * 闭环自我认知反馈系统（P1-05 修复）
 *
 * 实现完整的"预测→实测→偏差分析→模型校准"闭环。
 * 解决元认知监控与深度反思/深度思考链集成不完整的问题。
 * ============================================================================ */

/**
 * @brief 闭环反馈校准上下文
 * 记录一次完整的预测-实测闭环操作的所有数据
 */
typedef struct {
    float* predicted_state;         /**< 预测状态 [state_dim] */
    float* actual_state;            /**< 实测状态 [state_dim] */
    float* deviation;               /**< 逐维偏差 [state_dim] */
    size_t state_dim;               /**< 状态维度 */
    float max_deviation;            /**< 最大偏差 */
    float avg_deviation;            /**< 平均偏差 */
    float calibration_factor;       /**< 校准因子（应用于模型） */
    float confidence_shift;         /**< 置信度变化 (可正可负) */
    time_t timestamp;               /**< 时间戳 */
    int calibration_applied;        /**< 是否已应用校准 */
    int triggered_deep_reflection;  /**< 是否触发了深度反思 */
    int triggered_retraining;       /**< 是否触发了重训练 */
    char insight[256];              /**< 闭环产生的洞见 */
} ClosedLoopFeedback;

/**
 * @brief 执行完整的闭环自我认知反馈
 *
 * 核心闭环流程：
 *   1. 使用当前自我模型预测未来状态
 *   2. 接收实际测量值
 *   3. 计算预测-实测偏差
 *   4. 自动校准自我模型（调整置信度/触发深度反思/触发重训练）
 *   5. 生成闭环反馈报告
 *
 * @param system 自我认知系统句柄
 * @param actual_state 实际观测到的系统状态 [state_dim]
 * @param state_dim 状态维度
 * @param feedback 闭环反馈输出
 * @return 0成功，-1失败
 */
int self_cognition_closed_loop_feedback(SelfCognitionSystem* system,
                                        const float* actual_state,
                                        size_t state_dim,
                                        ClosedLoopFeedback* feedback);

/**
 * @brief 释放闭环反馈
 */
void closed_loop_feedback_free(ClosedLoopFeedback* feedback);

/**
 * @brief 将深度反思/深度思考链的洞见集成到自我模型
 *
 * 从深度反思引擎和深度思考链系统中提取洞见，
 * 转化为自我模型的校准参数，实现元认知↔深度认知的闭环。
 *
 * @param system 自我认知系统句柄
 * @param insights 洞见文本数组
 * @param insight_scores 洞见分数数组
 * @param num_insights 洞见数量
 * @param applied_count 输出实际应用的洞见数量
 * @return 0成功，-1失败
 */
int self_cognition_integrate_deep_insights(SelfCognitionSystem* system,
                                           const char* const* insights,
                                           const float* insight_scores,
                                           size_t num_insights,
                                           size_t* applied_count);

/**
 * @brief 获取自我模型的校准状态
 */
typedef struct {
    float model_accuracy;           /**< 模型当前准确性 */
    float calibration_health;       /**< 校准健康度 (0-1), <0.3需重校准 */
    float drift_rate;               /**< 预测漂移率 (0-1), >0.5需关注 */
    int consecutive_deviations;     /**< 连续大偏差次数 */
    time_t last_calibration;        /**< 上次校准时间 */
    int calibration_count;          /**< 累积校准次数 */
    int retraining_count;           /**< 累积重训练次数 */
    int deep_reflection_triggers;   /**< 深度反思触发次数 */
    float average_calibration_gain; /**< 平均校准增益 */
} SelfCalibrationStatus;

/**
 * @brief 获取自我模型校准状态
 */
int self_cognition_get_calibration_status(SelfCognitionSystem* system,
                                          SelfCalibrationStatus* status);

/**
 * @brief 性能预测与实测对比闭环
 *
 * 对特定能力维度进行预测→实测→校准的窄带闭环。
 * 适用于高频(>1Hz)运行时在线校准。
 *
 * @param system 自我认知系统句柄
 * @param capability_dim 能力维度索引 (0=推理,1=学习,2=记忆,3=规划,4=感知,5=执行)
 * @param predicted_score 预测分数
 * @param actual_score 实测分数
 * @param calibrated_score 输出校准后分数
 * @return 0成功，-1失败
 */
int self_cognition_calibrate_capability(SelfCognitionSystem* system,
                                        int capability_dim,
                                        float predicted_score,
                                        float actual_score,
                                        float* calibrated_score);

/* ============================================================================
 * 深度思维链集成增强（P1-05 修复）
 *
 * 为自我认知系统提供深度思维链(DTC)的原生支持，
 * 避免每次需要通过metacognition间接调用。
 * ============================================================================ */

/**
 * @brief 使用深度思维链进行自我反思性推理
 *
 * 构建一个自我反思查询，通过深度思维链的
 * 多步推理(observe→analyze→hypothesize→reason→evaluate→synthesize→conclude)
 * 生成关于自身状态、能力和改进方向的深层洞见。
 *
 * @param system 自我认知系统句柄
 * @param reasoning_output 推理结果文本输出
 * @param max_output_size 最大文本长度
 * @param confidence 输出推理置信度
 * @return 0成功，-1失败
 */
int self_cognition_deep_thought_self_reflection(SelfCognitionSystem* system,
                                                char* reasoning_output,
                                                size_t max_output_size,
                                                float* confidence);

/* ============================================================================
 * 连续自我校准循环（P1-05 修复）
 *
 * 实现完整的 self_model_predict → measure → compare →
 * calibrate → re-predict 循环，用于在线持续优化自我认知
 * ============================================================================ */

/**
 * @brief 自动校准循环配置
 */
typedef struct {
    float calibration_interval_sec;  /**< 校准间隔 (秒) */
    float drift_threshold;           /**< 漂移阈值，超此值触发深度校准 */
    int max_consecutive_deviations;  /**< 最大连续偏差容忍 */
    int enable_auto_retraining;      /**< 是否允许自动重训练 */
    int enable_deep_reflection_trigger; /**< 大偏差是否触发深度反思 */
    float auto_retrain_threshold;    /**< 自动重训练触发阈值 */
} AutoCalibrationConfig;

/**
 * @brief 获取默认自动校准配置
 */
AutoCalibrationConfig self_cognition_auto_calibration_default(void);

/**
 * @brief 执行一次自动校准循环
 *
 * 基于当前自我模型的准确性状态和最近的预测偏差，
 * 自动决定是否需要调整校准参数或触发深度自我分析。
 * 应定期调用（建议频率：1-10Hz）。
 *
 * @param system 自我认知系统句柄
 * @param config 自动校准配置
 * @param recent_deviations 最近的预测偏差数组（按时间顺序）
 * @param num_deviations 偏差数量
 * @param action_taken 输出执行的动作描述
 * @param action_size 描述缓冲区大小
 * @return 0无动作，1已执行校准，2已触发深度反思，3已触发重训练，-1失败
 */
int self_cognition_auto_calibration_cycle(SelfCognitionSystem* system,
                                          const AutoCalibrationConfig* config,
                                          const float* recent_deviations,
                                          size_t num_deviations,
                                          char* action_taken,
                                          size_t action_size);

/* ============================================================================
 * ZSF-009 修复: 将.c中已实现的公共函数声明到头文件
 * ============================================================================ */

/**
 * @brief 获取自我模型状态
 * @param system 自我认知系统句柄
 * @param model_state 输出模型状态
 * @return 0成功，-1失败
 */
int self_cognition_get_model_status(SelfCognitionSystem* system,
                                   SelfModelState* model_state);

/**
 * @brief 评估自我模型准确性
 * @param system 自我认知系统句柄
 * @return 模型准确性评分 (0.0-1.0)
 */
float self_cognition_assess_model_accuracy(SelfCognitionSystem* system);

/**
 * @brief 获取深度统计信息（供监控和调试使用）
 * @param system 自我认知系统句柄
 * @param total_reflections 输出总反思次数
 * @param total_corrections 输出总修正次数
 * @param avg_reflection_score 输出平均反思得分
 * @param avg_correction_depth 输出平均修正深度
 * @return 0成功，-1失败
 */
int self_cognition_get_deep_stats(SelfCognitionSystem* system,
                                  int* total_reflections,
                                  int* total_corrections,
                                  float* avg_reflection_score,
                                  float* avg_correction_depth);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_SELF_COGNITION_H