/**
 * @file deep_correction.h
 * @brief 深度反思驱动的自我修正系统接口
 *
 * 增强版：贝叶斯根因诊断 + 自适应修正强度 + 多阶段验证流水线 + 模式自动学习
 */

#ifndef SELFLNN_DEEP_CORRECTION_H
#define SELFLNN_DEEP_CORRECTION_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DC_MAX_ERRORS 256
#define DC_MAX_HYPOTHESES 16
#define DC_MAX_RULES 128
#define DC_MAX_TRACE 512
#define DC_MAX_BAYES_NODES 32
#define DC_MAX_BAYES_PARENTS 8
#define DC_MAX_PATTERNS 64
#define DC_MAX_CONTEXT_RECORDS 128
#define DC_VERIFY_STAGES 4

typedef enum {
    DC_ERROR_SYNTAX = 0,
    DC_ERROR_LOGIC = 1,
    DC_ERROR_KNOWLEDGE = 2,
    DC_ERROR_STRATEGY = 3,
    DC_ERROR_PERCEPTION = 4,
    DC_ERROR_EXECUTION = 5,
    DC_ERROR_UNKNOWN = 6
} DCErrorType;

typedef enum {
    DC_STATE_IDLE = 0,
    DC_STATE_ANALYZING = 1,
    DC_STATE_GENERATING = 2,
    DC_STATE_VERIFYING = 3,
    DC_STATE_APPLYING = 4,
    DC_STATE_MONITORING = 5
} DCCorrectionState;

typedef enum {
    DC_VERIFY_SYNTAX = 0,
    DC_VERIFY_LOGIC = 1,
    DC_VERIFY_EXECUTION = 2,
    DC_VERIFY_EFFECT = 3
} DCVerificationStage;

typedef struct {
    int error_id;
    DCErrorType type;
    char description[512];
    char location[128];
    char context[512];
    float severity;
    time_t detected_at;
    int root_cause_id;
    int resolved;
} DCErrorRecord;

typedef struct {
    int hypothesis_id;
    int error_id;
    char description[512];
    char proposed_fix[512];
    float confidence;
    float estimated_impact;
    int verified;
    int applied;
} DCCorrectionHypothesis;

typedef struct {
    int rule_id;
    char pattern[256];
    char correction[256];
    DCErrorType target_type;
    float success_rate;
    int usage_count;
    int success_count;
    time_t created_at;
    time_t last_used;
} DCCorrectionRule;

typedef struct {
    int node_id;
    char name[64];
    int parent_ids[DC_MAX_BAYES_PARENTS];
    int parent_count;
    float prior_prob;
    float cond_prob_given_parents[1 << DC_MAX_BAYES_PARENTS];
    float posterior_prob;
} DCBayesianNode;

typedef struct {
    DCBayesianNode nodes[DC_MAX_BAYES_NODES];
    int node_count;
    float root_cause_probs[DC_MAX_BAYES_NODES];
    int top_cause_ids[5];
    float top_cause_scores[5];
    int top_cause_count;
    char explanation[1024];
} DCDiagnosisResult;

typedef struct {
    int record_id;
    int error_id;
    int hypothesis_id;
    char before_state[1024];
    char after_state[1024];
    float effectiveness_score;
    float recovery_time_sec;
    int rollback_required;
    time_t applied_at;
    time_t resolved_at;
} DCCorrectionContext;

typedef struct {
    int pattern_id;
    char pattern_name[128];
    char pattern_expression[512];
    DCErrorType associated_type;
    float detection_weight;
    float confidence;
    int match_count;
    time_t first_detected;
    time_t last_detected;
} DCPatternRecord;

typedef struct DCCorrectionSystem DCCorrectionSystem;

DCCorrectionSystem* dc_correction_create(void);
void dc_correction_free(DCCorrectionSystem* dcs);

/* 错误管理 */
int dc_report_error(DCCorrectionSystem* dcs, DCErrorType type, const char* desc, const char* location, float severity);
int dc_get_error(const DCCorrectionSystem* dcs, int error_id, DCErrorRecord* out);
int dc_list_unresolved(const DCCorrectionSystem* dcs, int* error_ids, int max_count);

/* 根因分析（增强版：贝叶斯+规则融合） */
int dc_analyze_root_cause(DCCorrectionSystem* dcs, int error_id);
int dc_get_root_cause(const DCCorrectionSystem* dcs, int error_id, char* cause, size_t max_len);
int dc_bayesian_diagnose(DCCorrectionSystem* dcs, int error_id, DCDiagnosisResult* result);

/* 假设生成与验证（增强版：自适应强度+多阶段验证） */
int dc_generate_hypotheses(DCCorrectionSystem* dcs, int error_id, DCCorrectionHypothesis* out, int max_count);
int dc_verify_hypothesis(DCCorrectionSystem* dcs, int hypothesis_id, int* verified);
int dc_select_best_hypothesis(const DCCorrectionSystem* dcs, int error_id, DCCorrectionHypothesis* best);
int dc_validate_multi_stage(DCCorrectionSystem* dcs, int hypothesis_id, DCVerificationStage stages[], int stage_count, float* scores, int* stage_results);

/* 自适应修正强度 */
float dc_adaptive_strength(DCCorrectionSystem* dcs, int hypothesis_id);

/* 修正应用（增强版：带上下文记录） */
int dc_apply_correction(DCCorrectionSystem* dcs, int hypothesis_id);
int dc_monitor_correction(DCCorrectionSystem* dcs, int error_id, float* effectiveness);
int dc_rollback_correction(DCCorrectionSystem* dcs, int error_id);
int dc_get_correction_context(const DCCorrectionSystem* dcs, int error_id, DCCorrectionContext* out);

/* 模式自动学习 */
int dc_learn_pattern(DCCorrectionSystem* dcs, int error_id);
int dc_get_patterns(const DCCorrectionSystem* dcs, DCPatternRecord* out, int max_count);

/* 规则积累 */
int dc_extract_rule(DCCorrectionSystem* dcs, int error_id);
int dc_get_rules(const DCCorrectionSystem* dcs, DCCorrectionRule* out, int max_count);

/* 高级诊断统计 */
int dc_diagnose_advanced(DCCorrectionSystem* dcs, int error_id, DCDiagnosisResult* bayes_result, DCCorrectionHypothesis* best_hypothesis);
int dc_get_error_statistics(DCCorrectionSystem* dcs, float* resolution_rate, float* avg_recovery_time, float* pattern_diversity);
int dc_run_full_correction_pipeline(DCCorrectionSystem* dcs,
                                    DCErrorType error_type,
                                    const char* error_description,
                                    float severity,
                                    DCErrorRecord* out_error,
                                    DCCorrectionHypothesis* out_hypothesis,
                                    float* out_effectiveness);

/* 状态查询 */
DCCorrectionState dc_get_state(const DCCorrectionSystem* dcs);
int dc_get_stats(const DCCorrectionSystem* dcs, int* total_errors, int* resolved, int* pending);

/* ============================
 * 能力开关控制（P3.3）
 * ============================ */

/**
 * @brief 启用自我修正系统
 * 
 * @param dcs 自我修正系统句柄
 */
void dc_correction_enable(DCCorrectionSystem* dcs);

/**
 * @brief 禁用自我修正系统
 * 
 * @param dcs 自我修正系统句柄
 */
void dc_correction_disable(DCCorrectionSystem* dcs);

/**
 * @brief 检查自我修正系统是否已启用
 * 
 * @param dcs 自我修正系统句柄
 * @return 1表示启用，0表示禁用
 */
int dc_correction_is_enabled(const DCCorrectionSystem* dcs);

#ifdef __cplusplus
}
#endif
#endif
