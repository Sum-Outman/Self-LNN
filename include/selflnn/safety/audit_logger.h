#ifndef SELFLNN_AUDIT_LOGGER_H
#define SELFLNN_AUDIT_LOGGER_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 配置常量
 * ============================================================================ */
#define AUDIT_MAX_OPERATION_LOG    8192
#define AUDIT_MAX_DECISION_LOG     4096
#define AUDIT_MAX_CHANGE_LOG       2048
#define AUDIT_MAX_REPORT_LENGTH    65536
#define AUDIT_MAX_COMPLIANCE_RULES 256
#define AUDIT_MAX_ANALYSIS_WINDOW  1000
#define AUDIT_MAX_CATEGORIES       64

/* ============================================================================
 * A09.3.1 审计日志类型定义
 * ============================================================================ */

/* 操作日志类型 */
typedef enum {
    AUDIT_OP_SYSTEM_START = 0,
    AUDIT_OP_SYSTEM_STOP,
    AUDIT_OP_SYSTEM_RESTART,
    AUDIT_OP_MODEL_LOAD,
    AUDIT_OP_MODEL_UNLOAD,
    AUDIT_OP_TRAINING_START,
    AUDIT_OP_TRAINING_STOP,
    AUDIT_OP_INFERENCE,
    AUDIT_OP_DECISION_EXECUTE,
    AUDIT_OP_RESOURCE_ALLOCATE,
    AUDIT_OP_RESOURCE_RELEASE,
    AUDIT_OP_CONFIG_CHANGE,
    AUDIT_OP_USER_LOGIN,
    AUDIT_OP_USER_LOGOUT,
    AUDIT_OP_PERMISSION_CHANGE,
    AUDIT_OP_DATA_ACCESS,
    AUDIT_OP_DATA_MODIFY,
    AUDIT_OP_COMMUNICATION_SEND,
    AUDIT_OP_COMMUNICATION_RECV,
    AUDIT_OP_CUSTOM
} AuditOperationType;

/* 操作日志条目 */
typedef struct {
    uint64_t log_id;              /* P2-02修复: 使用uint64_t防止回绕溢出 */
    time_t timestamp;
    AuditOperationType op_type;
    char operation_name[64];
    char operator_name[64];
    char target[128];
    char before_state[256];
    char after_state[256];
    char result[128];
    int success;
    float duration_ms;
    char detail[512];
} AuditOperationEntry;

/* 变更日志类型 */
typedef enum {
    AUDIT_CHANGE_CONFIG = 0,
    AUDIT_CHANGE_PARAMETER,
    AUDIT_CHANGE_MODEL_WEIGHT,
    AUDIT_CHANGE_RULE,
    AUDIT_CHANGE_POLICY,
    AUDIT_CHANGE_PERMISSION,
    AUDIT_CHANGE_MODULE_STATE,
    AUDIT_CHANGE_HARDWARE_STATE,
    AUDIT_CHANGE_CUSTOM
} AuditChangeType;

/* 变更日志条目 */
typedef struct {
    uint64_t change_id;           /* P2-02修复: 使用uint64_t防止回绕溢出 */
    time_t timestamp;
    AuditChangeType change_type;
    char change_name[64];
    char change_initiator[64];
    char component[128];
    char old_value[512];
    char new_value[512];
    char reason[256];
    int approved;
    char approver[64];
    int rollback_possible;
    char rollback_instruction[256];
} AuditChangeEntry;

/* 决策日志条目（扩展自 decision_engine.h） */
typedef struct {
    uint64_t decision_id;         /* P2-02修复: 使用uint64_t防止回绕溢出 */
    time_t timestamp;
    char decision_name[128];
    char context[512];
    float confidence;
    float utility;
    int alternative_count;
    char chosen_alternative[128];
    char reasoning_path[1024];
    int is_autonomous;
    float execution_time_ms;
    int has_error;
    char error_message[256];
    int human_override;
    char override_reason[256];
} AuditDecisionEntry;

/* 审计日志分类 */
typedef struct {
    char category_name[64];
    AuditOperationType* op_types;
    int op_type_count;
    int is_critical;
    int retention_days;
} AuditCategory;

/* ============================================================================
 * A09.3.2 审计分析类型定义
 * ============================================================================ */

/* 异常行为模式 */
typedef enum {
    AUDIT_ANOMALY_NONE = 0,
    AUDIT_ANOMALY_FREQUENCY_SPIKE,
    AUDIT_ANOMALY_PATTERN_DEVIATION,
    AUDIT_ANOMALY_SEQUENCE_VIOLATION,
    AUDIT_ANOMALY_RESOURCE_SURGE,
    AUDIT_ANOMALY_UNAUTHORIZED_ACCESS,
    AUDIT_ANOMALY_DATA_LEAK,
    AUDIT_ANOMALY_PRIVILEGE_ESCALATION,
    AUDIT_ANOMALY_TIMING_ANOMALY
} AuditAnomalyType;

/* 异常分析结果 */
typedef struct {
    AuditAnomalyType anomaly_type;
    float anomaly_score;
    char description[256];
    time_t first_occurrence;
    time_t last_occurrence;
    int occurrence_count;
    char affected_components[256];
    int severity;
    int requires_attention;
} AuditAnomalyResult;

/* 审计报告生成选项 */
typedef struct {
    int include_operations;
    int include_decisions;
    int include_changes;
    int include_analysis;
    int include_compliance;
    time_t start_time;
    time_t end_time;
    char report_title[128];
    char report_format[16];
    int max_entries_per_section;
} AuditReportOptions;

/* 合规规则 */
typedef enum {
    AUDIT_COMPLIANCE_MANDATORY_LOG = 0,
    AUDIT_COMPLIANCE_APPROVAL_REQUIRED,
    AUDIT_COMPLIANCE_RETENTION_PERIOD,
    AUDIT_COMPLIANCE_ACCESS_CONTROL,
    AUDIT_COMPLIANCE_DATA_PROTECTION,
    AUDIT_COMPLIANCE_AUDIT_FREQUENCY,
    AUDIT_COMPLIANCE_CUSTOM
} AuditComplianceType;

/* 审计数据类别（M-004合规检查按数据源区分） */
typedef enum {
    AUDIT_DATA_ACCESS_LOG = 0,      /**< 访问日志 */
    AUDIT_DATA_CONFIG_CHANGE = 1,   /**< 配置变更 */
    AUDIT_DATA_SYSTEM_INTEGRITY = 2,/**< 系统完整性 */
    AUDIT_DATA_ERROR_LOG = 3        /**< 错误日志 */
} AuditDataCategory;

/* 审计统一日志事件类型 */
typedef enum {
    AUDIT_EVENT_ACCESS_VIOLATION = 0, /**< 访问违规 */
    AUDIT_EVENT_CONFIG_MODIFY = 1,    /**< 配置修改 */
    AUDIT_EVENT_INTEGRITY_BREACH = 2, /**< 完整性破坏 */
    AUDIT_EVENT_ERROR_OCCURRENCE = 3, /**< 错误事件 */
    AUDIT_EVENT_SYSTEM_ERROR = 3      /**< 系统严重错误（同ERROR_OCCURRENCE） */
} AuditUnifiedEventType;

/* 审计记录授权状态 */
typedef enum {
    AUDIT_STATUS_OK = 0,              /**< 正常/已授权 */
    AUDIT_STATUS_AUTHORIZED = 0,      /**< 已授权（同OK） */
    AUDIT_STATUS_UNAUTHORIZED = 1     /**< 未授权 */
} AuditAuthorizationStatus;

/* 统一审计日志记录（供M-004合规检查使用） */
typedef struct {
    uint64_t id;                  /* P2-02修复: 使用uint64_t防止回绕溢出 */
    time_t timestamp;
    AuditUnifiedEventType event_type;
    AuditDataCategory data_type; /**< 所属数据类别 */
    AuditAuthorizationStatus status;
    char source[128];
    char detail[256];
} AuditUnifiedLogRecord;

/* 合规规则 */
typedef struct {
    int rule_id;
    AuditComplianceType type;
    AuditDataCategory data_type; /**< 关联的数据类别 */
    char rule_name[64];
    char description[256];
    char requirement[512];
    int is_enabled;
    int is_automatic;
    int criticality;
} AuditComplianceRule;

/* 合规检查结果 */
typedef struct {
    int rule_id;
    char rule_name[64];
    int passed;
    float compliance_score;
    char evidence[512];
    char recommendation[256];
    time_t check_time;
} AuditComplianceResult;

/* 审计日志系统句柄 */
typedef struct AuditLogger AuditLogger;

/* ============================================================================
 * 核心API：创建与销毁
 * ============================================================================ */

/**
 * @brief 创建审计日志系统
 */
AuditLogger* audit_logger_create(void);

/**
 * @brief 销毁审计日志系统
 */
void audit_logger_free(AuditLogger* logger);

/* ============================================================================
 * A09.3.1 操作日志
 * ============================================================================ */

/**
 * @brief 记录操作日志
 */
uint64_t audit_log_operation(AuditLogger* logger, AuditOperationType op_type,
                          const char* op_name, const char* operator_name,
                          const char* target, const char* before_state,
                          const char* after_state, const char* result,
                          int success, float duration_ms, const char* detail);

/**
 * @brief 查询操作日志（按时间范围）
 */
int audit_query_operations(const AuditLogger* logger, time_t start, time_t end,
                            AuditOperationEntry* entries, int max_count);

/**
 * @brief 查询操作日志（按类型）
 */
int audit_query_operations_by_type(const AuditLogger* logger, AuditOperationType type,
                                    AuditOperationEntry* entries, int max_count);

/**
 * @brief 查询操作日志（按操作者）
 */
int audit_query_operations_by_operator(const AuditLogger* logger, const char* operator_name,
                                        AuditOperationEntry* entries, int max_count);

/* ============================================================================
 * A09.3.1 决策日志
 * ============================================================================ */

/**
 * @brief 记录决策日志
 */
uint64_t audit_log_decision(AuditLogger* logger,
                         const char* decision_name, const char* context,
                         float confidence, float utility,
                         int alternative_count, const char* chosen_alternative,
                         const char* reasoning_path, int is_autonomous,
                         float execution_time_ms, int has_error,
                         const char* error_message, int human_override,
                         const char* override_reason);

/**
 * @brief 查询决策日志
 */
int audit_query_decisions(const AuditLogger* logger, time_t start, time_t end,
                           AuditDecisionEntry* entries, int max_count);

/* ============================================================================
 * A09.3.1 变更日志
 * ============================================================================ */

/**
 * @brief 记录变更日志
 */
uint64_t audit_log_change(AuditLogger* logger, AuditChangeType change_type,
                       const char* change_name, const char* initiator,
                       const char* component, const char* old_value,
                       const char* new_value, const char* reason,
                       int approved, const char* approver,
                       int rollback_possible, const char* rollback_instruction);

/**
 * @brief 查询变更日志（按组件）
 */
int audit_query_changes_by_component(const AuditLogger* logger, const char* component,
                                      AuditChangeEntry* entries, int max_count);

/**
 * @brief 查询变更日志（按时间范围）
 */
int audit_query_changes(const AuditLogger* logger, time_t start, time_t end,
                         AuditChangeEntry* entries, int max_count);

/* ============================================================================
 * A09.3.2 审计分析
 * ============================================================================ */

/**
 * @brief 执行异常行为分析
 *
 * 分析指定时间窗口内的操作/决策/变更日志，
 * 检测频率突增、模式偏离、序列违规等异常模式。
 */
int audit_analyze_anomalies(AuditLogger* logger, time_t window_start, time_t window_end,
                             AuditAnomalyResult* results, int max_count);

/**
 * @brief 检测操作频率异常
 */
int audit_detect_frequency_anomaly(const AuditLogger* logger, AuditOperationType type,
                                    time_t window_start, time_t window_end,
                                    float* out_baseline, float* out_current,
                                    float* out_deviation);

/**
 * @brief 生成审计报告
 *
 * 根据报告选项生成结构化的审计报告字符串。
 * 包含操作摘要、决策分析、变更追踪、异常事件和合规状态。
 */
int audit_generate_report(const AuditLogger* logger, const AuditReportOptions* options,
                           char* report_buffer, size_t buffer_size);

/* ============================================================================
 * A09.3.2 合规检查
 * ============================================================================ */

/**
 * @brief 添加合规规则
 */
int audit_add_compliance_rule(AuditLogger* logger, const AuditComplianceRule* rule);

/**
 * @brief 执行合规检查
 *
 * 遍历所有启用的合规规则并逐一检查。
 * 返回每条规则的检查结果和总体合规评分。
 */
int audit_check_compliance(const AuditLogger* logger,
                            AuditComplianceResult* results, int max_count,
                            float* out_overall_score);

/**
 * @brief 检查日志保留期合规
 */
int audit_check_retention_compliance(const AuditLogger* logger,
                                      AuditComplianceResult* result);

/* ============================================================================
 * 辅助API
 * ============================================================================ */

/**
 * @brief 获取审计日志统计
 */
int audit_get_statistics(const AuditLogger* logger, size_t* out_op_count,
                          size_t* out_decision_count, size_t* out_change_count,
                          time_t* out_earliest, time_t* out_latest);

/**
 * @brief 清除旧日志（按保留天数）
 */
int audit_purge_old_logs(AuditLogger* logger, int retention_days);

/**
 * @brief 导出审计日志为JSON
 */
int audit_export_json(const AuditLogger* logger, time_t start, time_t end,
                       char* json_buffer, size_t buffer_size);

/**
 * @brief 创建默认合规规则集
 */
int audit_setup_default_compliance(AuditLogger* logger);

/* ============================================================================
 * P0-001修复: 文件持久化API
 * ============================================================================ */

/**
 * @brief 将内存中的审计日志刷写到磁盘文件
 *
 * 将操作日志、决策日志、变更日志以JSON行格式写入
 * logs/audit_YYYYMMDD.jsonl 文件。
 * 使用追加模式(fopen "a")，每次写入后调用fflush确保数据落盘。
 * 线程安全：内部持有锁读取缓冲区，文件I/O在锁外执行。
 *
 * @param logger 审计日志系统句柄
 * @return int 成功返回写入的日志条数，失败返回-1
 */
int audit_flush_to_file(AuditLogger* logger);

/**
 * @brief 按日轮转审计日志文件
 *
 * 检查当前日期是否与上次刷写日期不同。
 * 如果进入新的一天，清理超过7天的旧日志文件，
 * 保留最近7天的审计日志。
 *
 * @param logger 审计日志系统句柄
 * @return int 返回清理的文件数量，0表示无需清理
 */
int audit_log_rotate(AuditLogger* logger);

/**
 * @brief 记录系统启动事件并立即持久化
 *
 * 在系统启动时调用，记录一次"系统启动"操作日志，
 * 并立即调用audit_flush_to_file()将日志刷写到磁盘。
 * 确保系统崩溃后启动事件仍可追溯。
 *
 * @param logger 审计日志系统句柄
 * @return int 成功返回操作日志ID，失败返回-1
 */
int audit_log_startup(AuditLogger* logger);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_AUDIT_LOGGER_H */
