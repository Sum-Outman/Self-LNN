#ifndef SELFLNN_DECISION_ENGINE_H
#define SELFLNN_DECISION_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file decision_engine.h
 * @brief 决策引擎核心接口
 * 
 * 决策引擎提供完整的效用函数计算和多目标决策能力，
 * 支持基于多属性效用理论（MAUT）的决策框架，
 * 能够处理连续和离散决策空间，支持约束优化和帕累托最优。
 */

/**
 * @brief 决策类型枚举
 */
typedef enum {
    DECISION_TYPE_SINGLE_OBJECTIVE = 0,   /**< 单目标决策 */
    DECISION_TYPE_MULTI_OBJECTIVE = 1,    /**< 多目标决策 */
    DECISION_TYPE_CONSTRAINED = 2,        /**< 约束决策 */
    DECISION_TYPE_SEQUENTIAL = 3,         /**< 序列决策 */
    DECISION_TYPE_GAME_THEORETIC = 4      /**< 博弈论决策 */
} DecisionType;

/**
 * @brief 自定义效用函数指针类型
 * 
 * @param value 输入值
 * @param params 函数参数数组
 * @param params_size 参数数组大小
 * @param user_data 用户数据指针
 * @return float 效用值
 */
typedef float (*CustomUtilityFunction)(float value, const float* params, size_t params_size, void* user_data);

/**
 * @brief 效用函数类型枚举
 */
typedef enum {
    UTILITY_LINEAR = 0,                   /**< 线性效用函数 */
    UTILITY_EXPONENTIAL = 1,              /**< 指数效用函数 */
    UTILITY_LOGARITHMIC = 2,              /**< 对数效用函数 */
    UTILITY_SIGMOID = 3,                  /**< S型效用函数 */
    UTILITY_QUADRATIC = 4,                /**< 二次效用函数 */
    UTILITY_CUSTOM = 5                    /**< 自定义效用函数 */
} UtilityFunctionType;

/**
 * @brief 决策目标结构体
 */
typedef struct {
    char* name;                           /**< 目标名称 */
    float weight;                         /**< 目标权重 (0-1) */
    float min_value;                      /**< 最小值 */
    float max_value;                      /**< 最大值 */
    int is_maximization;                  /**< 是否为最大化目标 (1=最大化, 0=最小化) */
    UtilityFunctionType utility_type;     /**< 效用函数类型 */
    float* utility_params;                /**< 效用函数参数数组 */
    size_t params_size;                   /**< 参数数组大小 */
    CustomUtilityFunction custom_function; /**< 自定义效用函数指针（仅当utility_type为UTILITY_CUSTOM时有效） */
    void* custom_user_data;               /**< 自定义函数用户数据指针 */
} DecisionObjective;

/**
 * @brief 决策约束结构体
 */
typedef struct {
    char* name;                           /**< 约束名称 */
    float lower_bound;                    /**< 下界 */
    float upper_bound;                    /**< 上界 */
    int is_equality;                      /**< 是否为等式约束 (1=等式, 0=不等式) */
    float penalty_weight;                 /**< 违反约束的惩罚权重 */
    int attribute_index;                  /**< 属性索引（-1表示使用自定义函数计算） */
} DecisionConstraint;

/**
 * @brief 决策备选方案结构体
 */
typedef struct {
    char* id;                             /**< 方案ID */
    float* attribute_values;              /**< 属性值数组 */
    size_t num_attributes;                /**< 属性数量 */
    float* objective_values;              /**< 目标值数组（计算后） */
    size_t num_objectives;                /**< 目标数量 */
    float overall_utility;                /**< 总体效用值 */
    float constraint_violation;           /**< 约束违反程度 (0=无违反) */
    int is_feasible;                      /**< 是否可行 (1=可行) */
} DecisionAlternative;

/**
 * @brief 决策结果结构体
 */
typedef struct {
    DecisionAlternative* best_alternative; /**< 最佳方案 */
    DecisionAlternative* alternatives;     /**< 所有方案数组 */
    size_t num_alternatives;               /**< 方案数量 */
    float* pareto_front;                   /**< 帕累托前沿（多目标决策） */
    size_t pareto_size;                    /**< 帕累托前沿大小 */
    float decision_confidence;             /**< 决策置信度 (0-1) */
    float computation_time_ms;             /**< 计算时间（毫秒） */
} DecisionResult;

/**
 * @brief 决策引擎配置结构体
 */
typedef struct {
    DecisionType decision_type;            /**< 决策类型 */
    size_t max_alternatives;               /**< 最大备选方案数 */
    size_t max_objectives;                 /**< 最大目标数 */
    size_t max_constraints;                /**< 最大约束数 */
    float epsilon;                         /**< 数值计算容差 */
    int enable_pareto_optimization;        /**< 是否启用帕累托优化 */
    int enable_uncertainty_modeling;       /**< 是否启用不确定性建模 */
    int enable_explanation;                /**< 是否启用决策解释 */
} DecisionEngineConfig;

/**
 * @brief 决策引擎句柄
 */
typedef struct DecisionEngine DecisionEngine;

/**
 * @brief 创建决策引擎
 * 
 * @param config 引擎配置
 * @return DecisionEngine* 引擎句柄，失败返回NULL
 */
DecisionEngine* decision_engine_create(const DecisionEngineConfig* config);

/**
 * @brief 销毁决策引擎
 * 
 * @param engine 引擎句柄
 */
void decision_engine_destroy(DecisionEngine* engine);

/**
 * @brief 设置决策目标
 * 
 * @param engine 引擎句柄
 * @param objectives 目标数组
 * @param num_objectives 目标数量
 * @return int 成功返回0，失败返回错误码
 */
int decision_engine_set_objectives(DecisionEngine* engine,
                                   const DecisionObjective* objectives,
                                   size_t num_objectives);

/**
 * @brief 设置决策约束
 * 
 * @param engine 引擎句柄
 * @param constraints 约束数组
 * @param num_constraints 约束数量
 * @return int 成功返回0，失败返回错误码
 */
int decision_engine_set_constraints(DecisionEngine* engine,
                                    const DecisionConstraint* constraints,
                                    size_t num_constraints);

/**
 * @brief 添加决策备选方案
 * 
 * @param engine 引擎句柄
 * @param alternatives 方案数组
 * @param num_alternatives 方案数量
 * @return int 成功返回0，失败返回错误码
 */
int decision_engine_add_alternatives(DecisionEngine* engine,
                                     const DecisionAlternative* alternatives,
                                     size_t num_alternatives);

/**
 * @brief 清空所有决策备选方案
 *
 * 清除引擎中所有已添加的备选方案，释放相关内存，
 * 用于在每次认知循环决策前重置方案列表。
 *
 * @param engine 引擎句柄
 */
void decision_engine_clear_alternatives(DecisionEngine* engine);

/**
 * @brief 执行决策分析
 * 
 * @param engine 引擎句柄
 * @param result 决策结果输出（需要调用decision_result_free释放）
 * @return int 成功返回0，失败返回错误码
 */
int decision_engine_analyze(DecisionEngine* engine, DecisionResult* result);

/**
 * @brief 计算多目标帕累托前沿
 * 
 * @param engine 引擎句柄
 * @param alternatives 方案数组
 * @param num_alternatives 方案数量
 * @param pareto_front 帕累托前沿输出数组
 * @param max_pareto_size 最大帕累托前沿大小
 * @return int 实际帕累托前沿大小，失败返回-1
 */
int decision_engine_compute_pareto_front(DecisionEngine* engine,
                                         const DecisionAlternative* alternatives,
                                         size_t num_alternatives,
                                         DecisionAlternative* pareto_front,
                                         size_t max_pareto_size);

/**
 * @brief 计算效用函数值
 * 
 * @param utility_type 效用函数类型
 * @param value 输入值
 * @param params 函数参数数组
 * @param params_size 参数数组大小
 * @return float 效用值
 */
float decision_engine_compute_utility(UtilityFunctionType utility_type,
                                      float value,
                                      const float* params,
                                      size_t params_size);

/**
 * @brief 释放决策结果内存
 * 
 * @param result 决策结果
 */
void decision_result_free(DecisionResult* result);

/**
 * @brief 释放决策备选方案内存
 * 
 * @param alternative 决策备选方案
 */
void decision_alternative_free(DecisionAlternative* alternative);

/* ============================================================================
 * 决策审计日志系统
 * ============================================================================ */

/**
 * @brief 决策日志条目类型
 */
typedef enum {
    DECISION_LOG_TYPE_ANALYSIS = 0,   /**< 决策分析 */
    DECISION_LOG_TYPE_EXECUTION = 1,  /**< 决策执行 */
    DECISION_LOG_TYPE_EVALUATION = 2, /**< 决策评估 */
    DECISION_LOG_TYPE_OVERRIDE = 3,   /**< 决策覆盖（手动干预） */
    DECISION_LOG_TYPE_ERROR = 4       /**< 决策错误 */
} DecisionLogType;

/**
 * @brief 决策日志条目
 */
typedef struct {
    DecisionLogType log_type;         /**< 日志类型 */
    char timestamp[32];               /**< 时间戳字符串 */
    char description[256];            /**< 决策描述 */
    char objective_labels[512];       /**< 目标标签（JSON数组字符串） */
    float objective_values[16];       /**< 当前目标值 */
    size_t objective_count;           /**< 目标数量 */
    char chosen_alternative[128];     /**< 选择的方案描述 */
    float chosen_utility;             /**< 选择方案的效用值 */
    int total_alternatives;           /**< 总方案数 */
    int is_autonomous;                /**< 是否自主决策 */
    float execution_time_ms;          /**< 决策执行耗时(ms) */
} DecisionLogEntry;

/**
 * @brief 决策审计日志
 */
typedef struct {
    DecisionLogEntry entries[256];   /**< 日志条目环形缓冲区 */
    size_t entry_count;              /**< 当前条目数 */
    size_t max_entries;              /**< 最大条目数 */
    int is_enabled;                  /**< 是否启用审计日志 */
} DecisionAuditLog;

/**
 * @brief 初始化决策审计日志
 *
 * @param log 审计日志句柄
 * @param max_entries 最大条目数（默认256）
 * @return int 成功返回0，失败返回-1
 */
int decision_audit_log_init(DecisionAuditLog* log, size_t max_entries);

/**
 * @brief 记录决策日志条目
 *
 * @param log 审计日志句柄
 * @param log_type 日志类型
 * @param description 决策描述
 * @param objective_values 目标值数组
 * @param objective_labels 目标标签字符串（逗号分隔）
 * @param objective_count 目标数量
 * @param chosen_alternative 选择的方案描述
 * @param chosen_utility 选择方案的效用值
 * @param total_alternatives 总方案数
 * @param is_autonomous 是否自主决策
 * @param execution_time_ms 执行耗时(ms)
 * @return int 成功返回0，失败返回-1
 */
int decision_audit_log_record(DecisionAuditLog* log,
                              DecisionLogType log_type,
                              const char* description,
                              const float* objective_values,
                              const char* objective_labels,
                              size_t objective_count,
                              const char* chosen_alternative,
                              float chosen_utility,
                              int total_alternatives,
                              int is_autonomous,
                              float execution_time_ms);

/**
 * @brief 导出决策审计日志为JSON字符串
 *
 * @param log 审计日志句柄
 * @param json_buffer [out] JSON输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回写入字符数，失败返回-1
 */
int decision_audit_log_export_json(const DecisionAuditLog* log,
                                   char* json_buffer, size_t buffer_size);

/**
 * @brief 获取最近的N条决策日志
 *
 * @param log 审计日志句柄
 * @param entries [out] 日志条目输出数组
 * @param max_count 最大获取条数
 * @return int 实际获取的条目数，失败返回-1
 */
int decision_audit_log_recent(const DecisionAuditLog* log,
                              DecisionLogEntry* entries, size_t max_count);

/**
 * @brief 清除决策审计日志
 *
 * @param log 审计日志句柄
 */
void decision_audit_log_clear(DecisionAuditLog* log);

/* ============================
 * 能力开关控制（P3.3）
 * ============================ */

/**
 * @brief 启用决策引擎
 * 
 * @param engine 决策引擎句柄
 */
void decision_engine_enable(DecisionEngine* engine);

/**
 * @brief 禁用决策引擎
 * 
 * @param engine 决策引擎句柄
 */
void decision_engine_disable(DecisionEngine* engine);

/**
 * @brief 检查决策引擎是否已启用
 * 
 * @param engine 决策引擎句柄
 * @return 1表示启用，0表示禁用
 */
int decision_engine_is_enabled(const DecisionEngine* engine);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_DECISION_ENGINE_H */