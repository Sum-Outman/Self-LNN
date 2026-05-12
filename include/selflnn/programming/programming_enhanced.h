#ifndef SELFLNN_PROGRAMMING_ENHANCED_H
#define SELFLNN_PROGRAMMING_ENHANCED_H

#include <stddef.h>
#include <stdint.h>
#include "selflnn/programming/self_programming.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 配置常量
 * ============================================================================ */
#define PENH_MAX_REFACTORING_ACTIONS   256
#define PENH_MAX_PERF_ISSUES           256
#define PENH_MAX_VULNERABILITIES       256
#define PENH_MAX_FUNCTIONS             1024
#define PENH_MAX_VARIABLES             4096
#define PENH_MAX_PATHS                 1024

/* ============================================================================
 * A09.4.1 代码重构引擎
 * ============================================================================ */

/* 重构操作类型 */
typedef enum {
    PENH_REFACTOR_RENAME_VARIABLE = 0,
    PENH_REFACTOR_EXTRACT_FUNCTION,
    PENH_REFACTOR_INLINE_FUNCTION,
    PENH_REFACTOR_EXTRACT_CONSTANT,
    PENH_REFACTOR_SIMPLIFY_CONDITIONAL,
    PENH_REFACTOR_SPLIT_LOOP,
    PENH_REFACTOR_MERGE_LOOPS,
    PENH_REFACTOR_REMOVE_DEAD_CODE,
    PENH_REFACTOR_MOVE_FUNCTION,
    PENH_REFACTOR_INTRODUCE_VARIABLE,
    PENH_REFACTOR_ENCAPSULATE_FIELD,
    PENH_REFACTOR_CUSTOM
} PenhRefactorType;

/* 重构操作 */
typedef struct {
    int action_id;
    PenhRefactorType type;
    char description[256];
    char source_location[128];
    char old_value[256];
    char new_value[256];
    float estimated_improvement;
    int risk_level;
    int applies_to_ast;
    int applies_to_source;
    char source_transform[1024];
} PenhRefactorAction;

/* 重构计划 */
typedef struct {
    PenhRefactorAction actions[PENH_MAX_REFACTORING_ACTIONS];
    int action_count;
    char plan_name[128];
    char plan_description[512];
    float total_improvement;
    int total_risk;
} PenhRefactorPlan;

/* 重构引擎句柄 */
typedef struct PenhRefactorEngine PenhRefactorEngine;

/**
 * @brief 创建重构引擎
 */
PenhRefactorEngine* penh_refactor_engine_create(void);

/**
 * @brief 销毁重构引擎
 */
void penh_refactor_engine_free(PenhRefactorEngine* engine);

/**
 * @brief 分析AST生成重构计划
 *
 * 遍历AST检测可重构的模式并生成操作序列。
 * 包括：长函数提取、重复代码合并、复杂条件简化、死代码移除等。
 */
int penh_analyze_refactoring(PenhRefactorEngine* engine, const ASTNode* ast,
                              PenhRefactorPlan* plan);

/**
 * @brief 检测可提取函数（代码块过长或重复）
 */
int penh_detect_extractable(const ASTNode* ast, int min_lines,
                             PenhRefactorAction* actions, int max_count);

/**
 * @brief 检测可简化条件表达式
 */
int penh_detect_simplifiable_conditional(const ASTNode* ast,
                                          PenhRefactorAction* actions, int max_count);

/**
 * @brief 检测可移除死代码
 */
int penh_detect_dead_code(const ASTNode* ast,
                           PenhRefactorAction* actions, int max_count);

/**
 * @brief 执行重构操作（在源代码层面进行字符串替换/变换）
 */
int penh_apply_refactoring(const PenhRefactorAction* action,
                            const char* source_code, char* output, size_t output_size);

/**
 * @brief 预览重构效果
 */
int penh_preview_refactoring(const PenhRefactorAction* action,
                              const char* source_code, char* preview, size_t preview_size);

/**
 * @brief 批量执行重构计划
 */
int penh_apply_refactor_plan(const PenhRefactorPlan* plan,
                              const char* source_code, char* output, size_t output_size);

/* ============================================================================
 * A09.4.2 性能分析
 * ============================================================================ */

/* 性能瓶颈类型 */
typedef enum {
    PENH_PERF_DEEP_RECURSION = 0,
    PENH_PERF_INFINITE_RECURSION,
    PENH_PERF_HIGH_COMPLEXITY_LOOP,
    PENH_PERF_REDUNDANT_COMPUTATION,
    PENH_PERF_POOR_MEMORY_LOCALITY,
    PENH_PERF_UNNECESSARY_ALLOCATION,
    PENH_PERF_INEFFICIENT_ALGORITHM,
    PENH_PERF_EXCESSIVE_INLINING,
    PENH_PERF_LARGE_STACK_FRAME,
    PENH_PERF_CUSTOM
} PenhPerfIssueType;

/* 性能问题 */
typedef struct {
    int issue_id;
    PenhPerfIssueType type;
    char function_name[128];
    int line;
    char description[256];
    float severity;
    char recommendation[256];
    float estimated_speedup;
} PenhPerfIssue;

/* 性能分析结果 */
typedef struct {
    PenhPerfIssue issues[PENH_MAX_PERF_ISSUES];
    int issue_count;
    float overall_score;
    char summary[1024];
} PenhPerfResult;

/**
 * @brief 执行性能分析
 *
 * 对源代码进行静态性能分析，检测性能瓶颈：
 * - 递归深度和无限递归风险
 * - 循环复杂度（嵌套深度、迭代范围）
 * - 冗余计算（重复子表达式）
 * - 内存访问模式
 * - 算法效率（基于操作模式估算时间复杂度）
 */
int penh_analyze_performance(const char* source_code, const ASTNode* ast,
                              PenhPerfResult* result);

/**
 * @brief 检测递归性能问题（深度递归和无限递归）
 */
int penh_detect_recursion_issues(const ASTNode* ast,
                                  PenhPerfIssue* issues, int max_count);

/**
 * @brief 检测循环性能问题（高复杂度循环）
 */
int penh_detect_loop_issues(const ASTNode* ast,
                             PenhPerfIssue* issues, int max_count);

/**
 * @brief 检测冗余计算
 */
int penh_detect_redundant_computation(const ASTNode* ast,
                                       PenhPerfIssue* issues, int max_count);

/* ============================================================================
 * A09.4.2 安全漏洞检测
 * ============================================================================ */

/* 安全漏洞类型 */
typedef enum {
    PENH_VULN_BUFFER_OVERFLOW = 0,
    PENH_VULN_NULL_DEREFERENCE,
    PENH_VULN_DIVIDE_BY_ZERO,
    PENH_VULN_INTEGER_OVERFLOW,
    PENH_VULN_UNINITIALIZED_VAR,
    PENH_VULN_MEMORY_LEAK,
    PENH_VULN_USE_AFTER_FREE,
    PENH_VULN_FORMAT_STRING,
    PENH_VULN_COMMAND_INJECTION,
    PENH_VULN_PATH_TRAVERSAL,
    PENH_VULN_UNCHECKED_RETURN,
    PENH_VULN_CUSTOM
} PenhVulnerabilityType;

/* 安全漏洞 */
typedef struct {
    int vuln_id;
    PenhVulnerabilityType type;
    int line;
    char function_name[128];
    char description[256];
    float severity;
    char recommendation[256];
    char code_snippet[256];
} PenhVulnerability;

/* 安全扫描结果 */
typedef struct {
    PenhVulnerability vulns[PENH_MAX_VULNERABILITIES];
    int vuln_count;
    float overall_risk_score;
    char summary[1024];
} PenhSecurityResult;

/**
 * @brief 执行安全漏洞扫描
 *
 * 对源代码进行静态安全分析，检测常见安全漏洞：
 * - 缓冲区溢出（数组访问越界、strcpy等）
 * - 空指针解引用
 * - 除零错误
 * - 整数溢出
 * - 未初始化变量使用
 * - 内存泄漏（malloc无free路径）
 * - 释放后使用
 * - 格式化字符串漏洞
 * - 命令注入
 * - 路径遍历
 */
int penh_scan_security(const char* source_code, const ASTNode* ast,
                        PenhSecurityResult* result);

/**
 * @brief 检测缓冲区溢出风险
 */
int penh_detect_buffer_overflow(const char* source_code,
                                 PenhVulnerability* vulns, int max_count);

/**
 * @brief 检测空指针解引用
 */
int penh_detect_null_dereference(const ASTNode* ast,
                                  PenhVulnerability* vulns, int max_count);

/**
 * @brief 检测内存泄漏
 */
int penh_detect_memory_leak(const ASTNode* ast,
                             PenhVulnerability* vulns, int max_count);

/**
 * @brief 检测除零错误
 */
int penh_detect_divide_by_zero(const ASTNode* ast,
                                PenhVulnerability* vulns, int max_count);

/* ============================================================================
 * 辅助API
 * ============================================================================ */

/**
 * @brief 获取安全漏洞类型名称
 */
const char* penh_vulnerability_type_name(PenhVulnerabilityType type);

/**
 * @brief 获取性能问题类型名称
 */
const char* penh_perf_issue_type_name(PenhPerfIssueType type);

/**
 * @brief 获取重构操作类型名称
 */
const char* penh_refactor_type_name(PenhRefactorType type);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_PROGRAMMING_ENHANCED_H */
