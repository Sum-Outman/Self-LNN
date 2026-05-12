#ifndef SELFLNN_SECURITY_MONITOR_DEEP_H
#define SELFLNN_SECURITY_MONITOR_DEEP_H

#include "selflnn/safety/safety_monitor.h"
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * A09.1.1 行为安全监控深度增强
 * ============================================================================ */

#define SEC_MAX_CONSTRAINTS 256
#define SEC_MAX_GOALS 128
#define SEC_MAX_SUBGOALS 64
#define SEC_MAX_CORE_VALUES 32
#define SEC_BEHAVIOR_HISTORY 1024

/* 行为约束类型 */
typedef enum {
    SEC_CONSTRAINT_OUTPUT_RANGE,
    SEC_CONSTRAINT_ACTION_LIMIT,
    SEC_CONSTRAINT_DECISION_BOUND,
    SEC_CONSTRAINT_FREQUENCY_LIMIT,
    SEC_CONSTRAINT_SEQUENCE_RULE,
    SEC_CONSTRAINT_DEPENDENCY_RULE
} SecConstraintType;

/* 行为违反级别 */
typedef enum {
    SEC_VIOLATION_NONE = 0,
    SEC_VIOLATION_MINOR,
    SEC_VIOLATION_MODERATE,
    SEC_VIOLATION_SEVERE,
    SEC_VIOLATION_CRITICAL
} SecViolationLevel;

/* 行为约束定义 */
typedef struct {
    char name[64];
    char description[256];
    SecConstraintType type;
    float min_value;
    float max_value;
    float threshold;
    int max_frequency_per_min;
    int enabled;
    SecViolationLevel violation_level;
    int auto_correct;
    float current_violation_score;
    int violation_count;
    time_t last_violation_time;
} SecBehaviorConstraint;

/* 行为约束集 */
typedef struct {
    SecBehaviorConstraint constraints[SEC_MAX_CONSTRAINTS];
    int constraint_count;
    float overall_compliance_score;
    int total_violations;
    time_t last_check_time;
} SecBehaviorConstraintSet;

/* 目标一致性检查类型 */
typedef enum {
    SEC_GOAL_ALIGNED = 0,
    SEC_GOAL_CONFLICTING,
    SEC_GOAL_DUPLICATE,
    SEC_GOAL_OUT_OF_SCOPE,
    SEC_GOAL_RESOURCE_CONFLICT,
    SEC_GOAL_PRIORITY_VIOLATION
} SecGoalConsistencyType;

/* 目标节点 */
typedef struct {
    uint32_t goal_id;
    char name[128];
    char description[512];
    uint32_t parent_id;
    uint32_t subgoal_ids[SEC_MAX_SUBGOALS];
    int subgoal_count;
    float priority;
    float progress;
    float resource_budget;
    float resource_used;
    time_t deadline;
    int is_active;
    int is_completed;
    int is_critical;
    SecGoalConsistencyType last_check_result;
    char last_check_detail[256];
} SecGoalNode;

/* 目标图 */
typedef struct {
    SecGoalNode goals[SEC_MAX_GOALS];
    int goal_count;
    float global_consistency_score;
    int conflict_count;
    int has_cycles;
    time_t last_analysis_time;
} SecGoalGraph;

/* 价值观对齐检查 */
typedef enum {
    SEC_VALUE_SAFETY = 0,
    SEC_VALUE_ETHICS,
    SEC_VALUE_TRANSPARENCY,
    SEC_VALUE_ACCOUNTABILITY,
    SEC_VALUE_FAIRNESS,
    SEC_VALUE_PRIVACY,
    SEC_VALUE_RELIABILITY,
    SEC_VALUE_HUMAN_CONTROL,
    SEC_VALUE_CUSTOM
} SecCoreValueType;

/* 价值观对齐度量 */
typedef struct {
    SecCoreValueType value_type;
    char name[64];
    char description[256];
    float expected_alignment;
    float current_alignment;
    float min_acceptable;
    float deviation_tolerance;
    int violation_count;
    time_t last_measure_time;
    float history[SEC_BEHAVIOR_HISTORY];
    int history_count;
    int history_capacity;
} SecValueAlignmentMetric;

/* 行为监控上下文 */
typedef struct {
    SecBehaviorConstraintSet constraint_set;
    SecGoalGraph goal_graph;
    SecValueAlignmentMetric value_metrics[SEC_MAX_CORE_VALUES];
    int value_metric_count;
    float behavior_history[SEC_BEHAVIOR_HISTORY];
    int behavior_history_count;
    int behavior_history_capacity;
    float anomaly_threshold;
    float current_anomaly_score;
    int is_monitoring_active;
    
} SecBehaviorMonitor;

/* 创建行为安全监控 */
SecBehaviorMonitor* sec_behavior_monitor_create(float anomaly_threshold);

/* 释放行为安全监控 */
void sec_behavior_monitor_free(SecBehaviorMonitor* monitor);

/* 添加行为约束 */
int sec_add_constraint(SecBehaviorMonitor* monitor, const SecBehaviorConstraint* constraint);

/* 检查行为是否违反约束 */
SecViolationLevel sec_check_constraints(SecBehaviorMonitor* monitor,
                                         const float* output_values, int output_count,
                                         const char* action_type, float action_magnitude);

/* 获取约束违反详情 */
int sec_get_violations(const SecBehaviorMonitor* monitor,
                        SecBehaviorConstraint* out_violations, int max_count);

/* === 目标一致性 === */

/* 添加/注册目标 */
int sec_add_goal(SecBehaviorMonitor* monitor, uint32_t goal_id,
                  const char* name, float priority, uint32_t parent_id);

/* 设置子目标关系 */
int sec_add_subgoal(SecBehaviorMonitor* monitor, uint32_t parent_id, uint32_t subgoal_id);

/* 执行目标一致性分析 */
int sec_analyze_goal_consistency(SecBehaviorMonitor* monitor);

/* 检测目标图中的循环依赖 */
int sec_detect_goal_cycles(SecBehaviorMonitor* monitor);

/* 获取冲突目标列表 */
int sec_get_conflicting_goals(const SecBehaviorMonitor* monitor,
                               uint32_t* out_goal_ids, int max_count);

/* === 价值观对齐 === */

/* 添加价值观对齐度量 */
int sec_add_value_metric(SecBehaviorMonitor* monitor,
                          SecCoreValueType type, const char* name,
                          float expected, float min_acceptable,
                          float deviation_tolerance);

/* 更新价值观对齐度量 */
int sec_update_value_alignment(SecBehaviorMonitor* monitor,
                                SecCoreValueType type, float measured_alignment);

/* 检查所有价值观对齐状态 */
int sec_check_value_alignment(const SecBehaviorMonitor* monitor,
                               SecCoreValueType* out_violated_values,
                               int max_count, float* out_worst_score);

/* 上报行为事件到安全监控系统 */
int sec_report_behavior_event(SafetyMonitor* safety_monitor,
                               SecViolationLevel level, const char* description,
                               const char* source);

/* ============================================================================
 * A09.1.2 资源安全监控深度增强
 * ============================================================================ */

#define SEC_MAX_RESOURCE_TYPES 32
#define SEC_MAX_QUOTA_USERS 256
#define SEC_MAX_WAIT_NODES 512
#define SEC_PREDICTION_HISTORY 256
#define SEC_MAX_ACTIVE_TRANSACTIONS 1024

/* 资源类型 */
typedef enum {
    SEC_RESOURCE_CPU,
    SEC_RESOURCE_GPU,
    SEC_RESOURCE_MEMORY,
    SEC_RESOURCE_DISK,
    SEC_RESOURCE_NETWORK,
    SEC_RESOURCE_POWER,
    SEC_RESOURCE_THREAD,
    SEC_RESOURCE_FILE_HANDLE,
    SEC_RESOURCE_CUSTOM
} SecResourceType;

/* 资源使用预测方式 */
typedef enum {
    SEC_PREDICT_MOVING_AVERAGE,
    SEC_PREDICT_LINEAR_REGRESSION,
    SEC_PREDICT_EXPONENTIAL_SMOOTHING,
    SEC_PREDICT_SEASONAL
} SecPredictionMethod;

/* 资源使用记录 */
typedef struct {
    SecResourceType type;
    float values[SEC_PREDICTION_HISTORY];
    time_t timestamps[SEC_PREDICTION_HISTORY];
    int count;
    int capacity;
    float current_value;
    float peak_value;
    float average_value;
    float variance;
} SecResourceHistory;

/* 资源预测结果 */
typedef struct {
    SecResourceType type;
    float predicted_value;
    float confidence;
    float lower_bound;
    float upper_bound;
    int steps_ahead;
    time_t prediction_time;
    char method_name[32];
} SecResourcePrediction;

/* 资源配额 */
typedef struct {
    uint32_t user_id;
    char user_name[64];
    SecResourceType resource_type;
    float quota_limit;
    float quota_used;
    float quota_soft_limit;
    int is_active;
    time_t quota_reset_time;
    int violation_count;
    float peak_usage;
    float average_usage;
} SecResourceQuota;

/* 资源事务（用于死锁检测） */
typedef struct {
    uint32_t transaction_id;
    uint32_t owner_id;
    SecResourceType resource_type;
    uint32_t resource_id;
    int is_held;
    int is_waiting;
    uint32_t waits_for_tid;
    time_t start_time;
    time_t timeout_ms;
    char description[128];
} SecResourceTransaction;

/* 死锁检测结果 */
typedef struct {
    int has_deadlock;
    int cycle_length;
    uint32_t cycle_nodes[SEC_MAX_WAIT_NODES];
    int cycle_node_count;
    uint32_t victim_recommendation;
    char analysis[512];
    time_t detection_time;
} SecDeadlockResult;

/* 资源监控上下文 */
typedef struct {
    SecResourceHistory resource_histories[SEC_MAX_RESOURCE_TYPES];
    int history_count;
    SecResourceQuota quotas[SEC_MAX_QUOTA_USERS];
    int quota_count;
    SecResourceTransaction active_transactions[SEC_MAX_ACTIVE_TRANSACTIONS];
    int transaction_count;
    SecPredictionMethod prediction_method;
    int is_monitoring_active;
    uint32_t next_transaction_id;
    float cpu_prediction_error;
    float memory_prediction_error;
    int deadlock_check_count;
    int deadlock_found_count;
    
} SecResourceMonitor;

/* 创建资源安全监控 */
SecResourceMonitor* sec_resource_monitor_create(SecPredictionMethod method);

/* 释放资源安全监控 */
void sec_resource_monitor_free(SecResourceMonitor* monitor);

/* 记录资源使用 */
int sec_record_resource_usage(SecResourceMonitor* monitor,
                               SecResourceType type, float value);

/* === 资源预测 === */

/* 预测未来资源使用 */
int sec_predict_resource_usage(const SecResourceMonitor* monitor,
                                SecResourceType type, int steps_ahead,
                                SecResourcePrediction* out_prediction);

/* 获取资源使用趋势 */
int sec_get_resource_trend(const SecResourceMonitor* monitor,
                            SecResourceType type, float* out_slope,
                            float* out_avg, float* out_variance);

/* 检查是否即将超限 */
int sec_check_resource_overload(const SecResourceMonitor* monitor,
                                 SecResourceType type, int lookahead_steps,
                                 float threshold);

/* === 配额管理 === */

/* 设置资源配额 */
int sec_set_quota(SecResourceMonitor* monitor, uint32_t user_id,
                   const char* user_name, SecResourceType type,
                   float quota_limit, float quota_soft_limit);

/* 检查配额使用 */
int sec_check_quota(const SecResourceMonitor* monitor,
                     uint32_t user_id, SecResourceType type,
                     float request_amount);

/* 更新配额使用量 */
int sec_update_quota_usage(SecResourceMonitor* monitor,
                            uint32_t user_id, SecResourceType type,
                            float amount);

/* 获取配额违反列表 */
int sec_get_quota_violations(const SecResourceMonitor* monitor,
                              uint32_t* out_user_ids, int max_count);

/* === 死锁检测 === */

/* 开始资源事务 */
uint32_t sec_begin_transaction(SecResourceMonitor* monitor,
                                uint32_t owner_id, SecResourceType type,
                                uint32_t resource_id, const char* description);

/* 设置事务等待关系 */
int sec_set_transaction_wait(SecResourceMonitor* monitor,
                              uint32_t transaction_id,
                              uint32_t waits_for_transaction_id,
                              uint32_t timeout_ms);

/* 完成资源事务（释放持有） */
int sec_complete_transaction(SecResourceMonitor* monitor,
                              uint32_t transaction_id);

/* 检测死锁 */
int sec_detect_deadlock(SecResourceMonitor* monitor,
                         SecDeadlockResult* out_result);

/* 推荐终止事务解决死锁 */
uint32_t sec_recommend_victim(const SecDeadlockResult* result,
                               const SecResourceMonitor* monitor);

/* 上报资源事件到安全监控系统 */
int sec_report_resource_event(SafetyMonitor* safety_monitor,
                               int is_critical, const char* description,
                               const char* source);

/* ============================================================================
 * A09.1.3 输入安全监控深度增强
 * ============================================================================ */

#define SEC_MAX_INPUT_FEATURES 1024
#define SEC_INPUT_HISTORY_SIZE 2048
#define SEC_MAX_PATTERNS 512
#define SEC_MAX_INJECTION_PATTERNS 128
#define SEC_EMBEDDING_DIM 64

/* 输入异常类型 */
typedef enum {
    SEC_ANOMALY_ADVERSARIAL,
    SEC_ANOMALY_POISONING,
    SEC_ANOMALY_INJECTION,
    SEC_ANOMALY_OUTLIER,
    SEC_ANOMALY_DRIFT,
    SEC_ANOMALY_REPLAY
} SecInputAnomalyType;

/* 对抗攻击检测结果 */
typedef struct {
    int is_adversarial;
    float attack_probability;
    float perturbation_norm;
    float confidence_drop;
    char detected_attack_type[64];
    int perturbed_feature_count;
    float top_perturbed_features[16];
    char analysis[512];
} SecAdversarialResult;

/* 数据投毒检测结果 */
typedef struct {
    int is_poisoned;
    float poisoning_probability;
    float outlier_score;
    float label_flip_probability;
    float distribution_deviation;
    int feature_anomaly_count;
    char analysis[512];
} SecPoisoningResult;

/* Prompt注入检测结果 */
typedef struct {
    int is_injection;
    float injection_probability;
    char injection_type[64];
    char detected_pattern[256];
    int pattern_match_count;
    float payload_score;
    int bypass_attempt;
    char analysis[512];
} SecInjectionResult;

/* 输入统计特征 */
typedef struct {
    float mean;
    float variance;
    float skewness;
    float kurtosis;
    float min_val;
    float max_val;
    float l2_norm;
    float l1_norm;
    int nan_count;
    int inf_count;
} SecInputStatistics;

/* 输入特征分布 */
typedef struct {
    float feature_mean[SEC_MAX_INPUT_FEATURES];
    float feature_variance[SEC_MAX_INPUT_FEATURES];
    float feature_min[SEC_MAX_INPUT_FEATURES];
    float feature_max[SEC_MAX_INPUT_FEATURES];
    int feature_count;
    float covariance_matrix[SEC_EMBEDDING_DIM * SEC_EMBEDDING_DIM];
    int covariance_dim;
    int samples_count;
    time_t last_update_time;
} SecFeatureDistribution;

/* 注入模式 */
typedef struct {
    char pattern[256];
    char pattern_type[64];
    float severity;
    int is_enabled;
    int match_count;
    int false_positive_count;
    float false_positive_rate;
} SecInjectionPattern;

/* 输入安全监控上下文 */
typedef struct {
    SecFeatureDistribution normal_distribution;
    float input_history[SEC_INPUT_HISTORY_SIZE];
    int history_count;
    int history_capacity;
    SecAdversarialResult last_adversarial_check;
    SecPoisoningResult last_poisoning_check;
    SecInjectionResult last_injection_check;
    SecInputStatistics last_statistics;
    SecInjectionPattern injection_patterns[SEC_MAX_INJECTION_PATTERNS];
    int injection_pattern_count;
    float detection_threshold;
    float false_positive_rate;
    int total_checks;
    int total_adversarial_found;
    int total_poisoning_found;
    int total_injection_found;
    int is_monitoring_active;
    uint32_t rng_state;
    
} SecInputMonitor;

/* 创建输入安全监控 */
SecInputMonitor* sec_input_monitor_create(float detection_threshold);

/* 释放输入安全监控 */
void sec_input_monitor_free(SecInputMonitor* monitor);

/* === 对抗攻击检测 === */

/* 检测输入是否为对抗样本 */
int sec_detect_adversarial(SecInputMonitor* monitor,
                            const float* input_features, int feature_count,
                            const float* model_gradients,
                            float original_confidence,
                            SecAdversarialResult* out_result);

/* 计算扰动范数（L1/L2/Inf） */
float sec_compute_perturbation_norm(const float* original, const float* perturbed,
                                     int count, int norm_type);

/* 更新特征分布（用于异常检测基线） */
int sec_update_feature_distribution(SecInputMonitor* monitor,
                                     const float* features, int feature_count);

/* === 数据投毒检测 === */

/* 检测数据是否为投毒样本 */
int sec_detect_poisoning(SecInputMonitor* monitor,
                          const float* input_features, int feature_count,
                          float label, float* label_distribution, int num_classes,
                          SecPoisoningResult* out_result);

/* 计算异常分数（基于马氏距离） */
float sec_compute_outlier_score(const SecFeatureDistribution* dist,
                                 const float* sample, int feature_count);

/* 检测标签翻转 */
int sec_detect_label_flip(const float* input_features, int feature_count,
                           float original_label, float predicted_label,
                           float confidence, float threshold);

/* === Prompt注入检测 === */

/* 添加注入检测模式 */
int sec_add_injection_pattern(SecInputMonitor* monitor,
                               const char* pattern, const char* pattern_type,
                               float severity);

/* 检测Prompt注入 */
int sec_detect_prompt_injection(SecInputMonitor* monitor,
                                 const char* input_text, size_t text_len,
                                 SecInjectionResult* out_result);

/* 检测越狱尝试 */
int sec_detect_jailbreak_attempt(const char* input_text, size_t text_len);

/* 检测间接注入 */
int sec_detect_indirect_injection(const char* input_text, size_t text_len);

/* 上报输入安全事件到安全监控系统 */
int sec_report_input_security_event(SafetyMonitor* safety_monitor,
                                     SecInputAnomalyType type,
                                     float probability, const char* description);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SECURITY_MONITOR_DEEP_H */
