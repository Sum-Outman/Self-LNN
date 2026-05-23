#ifndef SELFLNN_PRODUCT_DESIGN_ENHANCED_H
#define SELFLNN_PRODUCT_DESIGN_ENHANCED_H

#include <stddef.h>
#include "selflnn/product_design/product_design.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 配置常量
 * ============================================================================ */
#define PDE_MAX_REQUIREMENTS          1024
#define PDE_MAX_TRACE_LINKS           4096
#define PDE_MAX_OBJECTIVES            16
#define PDE_MAX_PARETO_SOLUTIONS      1024
#define PDE_MAX_SAMPLES               10000
#define PDE_MAX_CONSTRAINTS           256
#define PDE_DESCRIPTION_LEN           256

/* ============================================================================
 * A09.5.1 需求分析引擎
 * ============================================================================ */

/* 需求状态 */
typedef enum {
    PDE_REQ_DRAFT = 0,
    PDE_REQ_VALIDATED,
    PDE_REQ_APPROVED,
    PDE_REQ_IMPLEMENTED,
    PDE_REQ_VERIFIED,
    PDE_REQ_REJECTED,
    PDE_REQ_CHANGED
} PdeRequirementStatus;

/* 需求优先级 */
typedef enum {
    PDE_PRIORITY_CRITICAL = 0,
    PDE_PRIORITY_HIGH,
    PDE_PRIORITY_MEDIUM,
    PDE_PRIORITY_LOW,
    PDE_PRIORITY_OPTIONAL
} PdeRequirementPriority;

/* 验证结果严重性 */
typedef enum {
    PDE_SEVERITY_INFO = 0,
    PDE_SEVERITY_WARNING,
    PDE_SEVERITY_ERROR,
    PDE_SEVERITY_CRITICAL
} PdeValidationSeverity;

/* 验证问题 */
typedef struct {
    int issue_id;
    PdeValidationSeverity severity;
    char category[64];
    char description[PDE_DESCRIPTION_LEN];
    char location[128];
    char suggestion[PDE_DESCRIPTION_LEN];
} PdeValidationIssue;

/* 验证报告 */
typedef struct {
    PdeValidationIssue issues[256];
    int issue_count;
    int passed;
    float completeness_score;
    float consistency_score;
    float feasibility_score;
    char summary[512];
} PdeValidationReport;

/* 需求条目 */
typedef struct {
    int req_id;
    char title[128];
    char description[PDE_DESCRIPTION_LEN];
    PdeRequirementStatus status;
    PdeRequirementPriority priority;
    int source_line;
    char source_origin[128];
    float estimated_effort;
    float business_value;
    int parent_id;
    int dependency_ids[32];
    int dependency_count;
    int version;
    char change_history[1024];
} PdeRequirementItem;

/* 追踪链接 */
typedef struct {
    int link_id;
    int source_req_id;
    int target_req_id;
    char link_type[32];
    char description[PDE_DESCRIPTION_LEN];
} PdeTraceLink;

/* 需求追踪器句柄 */
typedef struct PdeRequirementTracker PdeRequirementTracker;

/**
 * @brief 创建需求追踪器
 */
PdeRequirementTracker* pde_tracker_create(void);

/**
 * @brief 销毁需求追踪器
 */
void pde_tracker_destroy(PdeRequirementTracker* tracker);

/**
 * @brief 从需求文本提取并注册需求条目
 *
 * 解析原始需求文本，自动分割为独立需求条目，
 * 分配唯一ID，设置初始状态为DRAFT。
 */
int pde_tracker_register_requirements(PdeRequirementTracker* tracker,
                                       const char* requirement_text);

/**
 * @brief 获取已注册的需求数量
 */
int pde_tracker_get_requirement_count(const PdeRequirementTracker* tracker);

/**
 * @brief 获取需求条目
 */
const PdeRequirementItem* pde_tracker_get_requirement(
    const PdeRequirementTracker* tracker, int req_id);

/**
 * @brief 设置需求状态
 */
int pde_tracker_set_status(PdeRequirementTracker* tracker,
                            int req_id, PdeRequirementStatus status);

/**
 * @brief 设置需求优先级
 */
int pde_tracker_set_priority(PdeRequirementTracker* tracker,
                              int req_id, PdeRequirementPriority priority);

/**
 * @brief 添加需求依赖关系
 */
int pde_tracker_add_dependency(PdeRequirementTracker* tracker,
                                int req_id, int depends_on_id);

/**
 * @brief 添加追踪链接
 */
int pde_tracker_add_trace_link(PdeRequirementTracker* tracker,
                                int source_id, int target_id,
                                const char* link_type);

/* ============================================================================
 * A09.5.1 需求验证
 * ============================================================================ */

/**
 * @brief 全面验证需求集
 *
 * 执行三项验证：
 * - 完整性检查：检查是否有缺失的必要字段（标题、描述等）
 * - 一致性检查：检查需求间是否有冲突（互斥、循环依赖等）
 * - 可行性检查：根据预估工作量和价值评估可行性
 */
int pde_validate_requirements(const PdeRequirementTracker* tracker,
                               PdeValidationReport* report);

/**
 * @brief 检查需求完整性
 */
int pde_check_completeness(const PdeRequirementTracker* tracker,
                            PdeValidationIssue* issues, int max_count);

/**
 * @brief 检查需求一致性（冲突检测、循环依赖检测）
 */
int pde_check_consistency(const PdeRequirementTracker* tracker,
                           PdeValidationIssue* issues, int max_count);

/**
 * @brief 检查需求可行性
 */
int pde_check_feasibility(const PdeRequirementTracker* tracker,
                           PdeValidationIssue* issues, int max_count);

/**
 * @brief 获取需求追踪报告（文本格式）
 */
int pde_generate_trace_report(const PdeRequirementTracker* tracker,
                               char* output, size_t output_size);

/* ============================================================================
 * A09.5.2 多目标设计优化
 * ============================================================================ */

/* 优化目标类型 */
typedef enum {
    PDE_OBJ_MINIMIZE_COST = 0,
    PDE_OBJ_MAXIMIZE_FEASIBILITY,
    PDE_OBJ_MAXIMIZE_INNOVATION,
    PDE_OBJ_MAXIMIZE_MARKET_POTENTIAL,
    PDE_OBJ_MINIMIZE_TIME,
    PDE_OBJ_MINIMIZE_COMPLEXITY,
    PDE_OBJ_CUSTOM
} PdeObjectiveType;

/* 优化目标 */
typedef struct {
    int obj_id;
    PdeObjectiveType type;
    char name[64];
    int maximize;
    float weight;
    float target_value;
    float current_best;
} PdeObjective;

/* Pareto 前沿解 */
typedef struct {
    int solution_id;
    DesignParameter* parameters;
    size_t param_count;
    float objective_values[PDE_MAX_OBJECTIVES];
    int objective_count;
    ProductSpec* spec;
    float crowding_distance;
    int rank;
} PdeParetoSolution;

/* 多目标优化结果 */
typedef struct {
    PdeParetoSolution pareto_front[PDE_MAX_PARETO_SOLUTIONS];
    int pareto_count;
    PdeObjective objectives[PDE_MAX_OBJECTIVES];
    int objective_count;
    int total_evaluations;
    char summary[1024];
} PdeMultiObjectiveResult;

/* 多目标优化器句柄 */
typedef struct PdeMultiObjectiveOptimizer PdeMultiObjectiveOptimizer;

/**
 * @brief 创建多目标优化器
 */
PdeMultiObjectiveOptimizer* pde_moo_create(void);

/**
 * @brief 销毁多目标优化器
 */
void pde_moo_destroy(PdeMultiObjectiveOptimizer* optimizer);

/**
 * @brief 添加优化目标
 */
int pde_moo_add_objective(PdeMultiObjectiveOptimizer* optimizer,
                           PdeObjectiveType type, const char* name,
                           int maximize, float weight);

/**
 * @brief 执行 NSGA-II 风格多目标优化
 *
 * 使用非支配排序 + 拥挤度距离的进化多目标优化：
 * - 种群初始化（参数空间随机采样）
 * - 非支配排序（Pareto 前沿分层）
 * - 拥挤度距离计算（保持解多样性）
 * - 锦标赛选择 + SBX 交叉 + 多项式变异
 * - 精英保留策略
 */
int pde_moo_optimize(PdeMultiObjectiveOptimizer* optimizer,
                      const ProductRequirement* requirement,
                      const DesignParameter* design_params,
                      size_t param_count,
                      const ConstraintSpec* constraints,
                      size_t constraint_count,
                      int population_size,
                      int generations,
                      PdeMultiObjectiveResult* result);

/**
 * @brief 从优化结果中选择最佳折衷解
 *
 * 使用 TOPSIS 方法从 Pareto 前沿中选择最优点：
 * 归一化 + 加权理想解距离排序
 */
int pde_moo_select_compromise(const PdeMultiObjectiveResult* result,
                               DesignParameter* best_params,
                               size_t* param_count);

/* ============================================================================
 * A09.5.2 设计空间探索
 * ============================================================================ */

/* 采样方法 */
typedef enum {
    PDE_SAMPLE_GRID = 0,
    PDE_SAMPLE_RANDOM,
    PDE_SAMPLE_LATIN_HYPERCUBE,
    PDE_SAMPLE_SOBOL
} PdeSampleMethod;

/* 敏感性分析结果 */
typedef struct {
    char param_name[64];
    float sensitivity_index;
    float main_effect;
    float interaction_effect;
    int rank;
} PdeSensitivityResult;

/**
 * @brief 执行设计空间探索
 *
 * 在参数空间中进行采样探索，评估各参数组合的设计质量。
 * 支持网格采样、随机采样、拉丁超立方采样。
 */
int pde_explore_design_space(const ProductRequirement* requirement,
                              const DesignParameter* params,
                              size_t param_count,
                              const ConstraintSpec* constraints,
                              size_t constraint_count,
                              PdeSampleMethod method,
                              int sample_count,
                              DesignVariant* results,
                              int* result_count);

/**
 * @brief 执行全局敏感性分析
 *
 * 基于 Morris 方法的 elementary effects 分析：
 * 计算每个参数对设计质量的主效应和交互效应，
 * 识别关键设计参数。
 */
int pde_sensitivity_analysis(const ProductRequirement* requirement,
                              const DesignParameter* params,
                              size_t param_count,
                              const ConstraintSpec* constraints,
                              size_t constraint_count,
                              int trajectories,
                              PdeSensitivityResult* results,
                              int* result_count);

/**
 * @brief 执行局部搜索优化
 *
 * 从初始设计参数出发，在邻域内迭代搜索更优解。
 * 每次沿所有参数方向进行小步探索，接受改进解。
 */
int pde_local_search(const ProductRequirement* requirement,
                      DesignParameter* params,
                      size_t param_count,
                      const ConstraintSpec* constraints,
                      size_t constraint_count,
                      int max_iterations,
                      float step_size,
                      DesignEvaluation* final_evaluation);

/* ============================================================================
 * ZSFWS-036: 工业设计种子案例库接口
 * ============================================================================ */

/**
 * @brief 加载种子案例到产品设计引擎
 *
 * 在引擎初始化时加载50+个跨领域工业设计先验案例，
 * 为设计优化、类比推理和设计空间探索提供基线参考。
 * 案例涵盖：消费电子(10)、机械设计(10)、软件界面(10)、
 * 建筑结构(5)、机器人设计(5)、其他(10)。
 *
 * @param engine 产品设计引擎句柄
 * @return 成功加载的案例数量，失败返回-1
 */
int product_design_load_seed_cases(struct ProductDesignEngine* engine);

/**
 * @brief 获取指定类别的种子案例数量
 *
 * @param category 案例类别 (0=消费电子,1=机械设计,2=软件界面,3=建筑结构,4=机器人设计,5=其他)
 * @return 该类别案例数量，无效类别返回0
 */
int product_design_get_seed_case_count_by_category(int category);

/**
 * @brief 获取种子案例条目详情
 *
 * @param index 案例索引（0~54）
 * @param name 输出：案例名称缓冲区
 * @param name_size 名称缓冲区大小
 * @param category_name 输出：类别名称缓冲区
 * @param cat_size 类别名称缓冲区大小
 * @param tags 输出：特征标签缓冲区（逗号分隔）
 * @param tags_size 特征标签缓冲区大小
 * @param constraints 输出：设计约束摘要缓冲区
 * @param cons_size 约束摘要缓冲区大小
 * @return 成功返回0，无效索引返回-1
 */
int product_design_get_seed_case(int index,
                                  char* name, size_t name_size,
                                  char* category_name, size_t cat_size,
                                  char* tags, size_t tags_size,
                                  char* constraints, size_t cons_size);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_PRODUCT_DESIGN_ENHANCED_H */