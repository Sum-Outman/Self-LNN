/**
 * @file decision_engine.c
 * @brief 决策引擎实现
 * 
 * 实现完整的决策引擎功能，包括效用函数计算、多目标决策、
 * 约束优化和帕累托前沿分析。
 */

#include "selflnn/core/decision_engine.h"
#include "selflnn/core/common.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/perf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifdef _MSC_VER
#pragma warning(disable:4100 4189 4244 4267 4701 4102 4702)
#endif

/**
 * @brief 决策引擎内部结构体
 */
struct DecisionEngine {
    DecisionEngineConfig config;           /**< 引擎配置 */
    
    DecisionObjective* objectives;         /**< 决策目标数组 */
    size_t num_objectives;                 /**< 目标数量 */
    size_t objectives_capacity;            /**< 目标数组容量 */
    
    DecisionConstraint* constraints;       /**< 决策约束数组 */
    size_t num_constraints;                /**< 约束数量 */
    size_t constraints_capacity;           /**< 约束数组容量 */
    
    DecisionAlternative* alternatives;     /**< 决策备选方案数组 */
    size_t num_alternatives;               /**< 方案数量 */
    size_t alternatives_capacity;          /**< 方案数组容量 */
    
    float* weight_normalization;           /**< 权重归一化因子 */
    float* constraint_penalties;           /**< 约束惩罚项数组 */
    float* objective_weights;              /**< 目标权重矩阵（可选） */
    size_t objective_weights_rows;         /**< 权重矩阵行数（目标数） */
    size_t objective_weights_cols;         /**< 权重矩阵列数（属性数） */
    
    // 内部状态
    int is_initialized;                    /**< 是否已初始化 */
    int objectives_set;                    /**< 目标是否已设置 */
    int constraints_set;                   /**< 约束是否已设置 */
    int enabled;                           /**< 能力开关（P3.3） */
    
    // 性能统计
    uint64_t total_computation_time_ns;    /**< 总计算时间（纳秒） */
    size_t total_decisions_made;           /**< 总决策次数 */
};

/* 内部辅助函数声明 */
static int validate_engine(const DecisionEngine* engine);
static int validate_objective(const DecisionObjective* objective);
static int validate_constraint(const DecisionConstraint* constraint);
static int validate_alternative(const DecisionAlternative* alternative);
static float compute_single_utility(const DecisionObjective* objective, float value);
static float compute_overall_utility(const DecisionEngine* engine, 
                                     const DecisionAlternative* alternative);
static float compute_constraint_violation(const DecisionEngine* engine,
                                          const DecisionAlternative* alternative);
static int is_pareto_dominant(const DecisionAlternative* a, 
                              const DecisionAlternative* b,
                              size_t num_objectives);
static int allocate_alternative_copy(DecisionAlternative* dest, 
                                     const DecisionAlternative* src);
static void free_alternative_internal(DecisionAlternative* alternative);

/**
 * @brief 创建决策引擎
 */
DecisionEngine* decision_engine_create(const DecisionEngineConfig* config) {
    if (!config) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建决策引擎：配置参数为空");
        return NULL;
    }
    
    // 验证配置参数
    if (config->max_alternatives == 0 || config->max_objectives == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "创建决策引擎：最大方案数或目标数为零");
        return NULL;
    }
    
    // 分配引擎内存
    DecisionEngine* engine = (DecisionEngine*)safe_calloc(1, sizeof(DecisionEngine));
    if (!engine) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建决策引擎：内存分配失败");
        return NULL;
    }
    
    // 复制配置
    engine->config = *config;
    
    // 初始化数组容量
    engine->objectives_capacity = config->max_objectives;
    engine->constraints_capacity = config->max_constraints;
    engine->alternatives_capacity = config->max_alternatives;
    
    // 分配目标数组
    engine->objectives = (DecisionObjective*)safe_calloc(engine->objectives_capacity, 
                                                        sizeof(DecisionObjective));
    if (!engine->objectives) {
        safe_free((void**)&engine);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建决策引擎：目标数组内存分配失败");
        return NULL;
    }
    
    // 分配约束数组
    engine->constraints = (DecisionConstraint*)safe_calloc(engine->constraints_capacity, 
                                                          sizeof(DecisionConstraint));
    if (!engine->constraints) {
        safe_free((void**)&engine->objectives);
        safe_free((void**)&engine);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建决策引擎：约束数组内存分配失败");
        return NULL;
    }
    
    // 分配方案数组
    engine->alternatives = (DecisionAlternative*)safe_calloc(engine->alternatives_capacity, 
                                                            sizeof(DecisionAlternative));
    if (!engine->alternatives) {
        safe_free((void**)&engine->constraints);
        safe_free((void**)&engine->objectives);
        safe_free((void**)&engine);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建决策引擎：方案数组内存分配失败");
        return NULL;
    }
    
    // 分配权重归一化数组
    engine->weight_normalization = (float*)safe_calloc(config->max_objectives, sizeof(float));
    if (!engine->weight_normalization) {
        safe_free((void**)&engine->alternatives);
        safe_free((void**)&engine->constraints);
        safe_free((void**)&engine->objectives);
        safe_free((void**)&engine);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建决策引擎：权重归一化数组内存分配失败");
        return NULL;
    }
    
    // 分配约束惩罚项数组
    engine->constraint_penalties = (float*)safe_calloc(config->max_constraints, sizeof(float));
    if (!engine->constraint_penalties) {
        safe_free((void**)&engine->weight_normalization);
        safe_free((void**)&engine->alternatives);
        safe_free((void**)&engine->constraints);
        safe_free((void**)&engine->objectives);
        safe_free((void**)&engine);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "创建决策引擎：约束惩罚项数组内存分配失败");
        return NULL;
    }
    
    // 初始化状态
    engine->is_initialized = 1;
    engine->objectives_set = 0;
    engine->constraints_set = 0;
    engine->enabled = 1;
    engine->num_objectives = 0;
    engine->num_constraints = 0;
    engine->num_alternatives = 0;
    engine->total_computation_time_ns = 0;
    engine->total_decisions_made = 0;
    
    // 初始化权重归一化因子为1.0
    for (size_t i = 0; i < config->max_objectives; i++) {
        engine->weight_normalization[i] = 1.0f;
    }
    
    // 初始化约束惩罚项为配置的惩罚权重
    for (size_t i = 0; i < config->max_constraints; i++) {
        engine->constraint_penalties[i] = 1.0f;  // 默认惩罚权重
    }
    
    // 初始化目标权重矩阵为NULL
    engine->objective_weights = NULL;
    engine->objective_weights_rows = 0;
    engine->objective_weights_cols = 0;
    
    return engine;
}

/**
 * @brief 销毁决策引擎
 */
void decision_engine_destroy(DecisionEngine* engine) {
    if (!engine) {
        return;
    }
    
    // 释放目标数组
    if (engine->objectives) {
        for (size_t i = 0; i < engine->num_objectives; i++) {
            if (engine->objectives[i].name) {
                safe_free((void**)&engine->objectives[i].name);
            }
            if (engine->objectives[i].utility_params) {
                safe_free((void**)&engine->objectives[i].utility_params);
            }
        }
        safe_free((void**)&engine->objectives);
    }
    
    // 释放约束数组
    if (engine->constraints) {
        for (size_t i = 0; i < engine->num_constraints; i++) {
            if (engine->constraints[i].name) {
                safe_free((void**)&engine->constraints[i].name);
            }
        }
        safe_free((void**)&engine->constraints);
    }
    
    // 释放方案数组
    if (engine->alternatives) {
        for (size_t i = 0; i < engine->num_alternatives; i++) {
            free_alternative_internal(&engine->alternatives[i]);
        }
        safe_free((void**)&engine->alternatives);
    }
    
    // 释放其他数组
    if (engine->weight_normalization) {
        safe_free((void**)&engine->weight_normalization);
    }
    
    if (engine->constraint_penalties) {
        safe_free((void**)&engine->constraint_penalties);
    }
    
    if (engine->objective_weights) {
        safe_free((void**)&engine->objective_weights);
    }
    
    // 释放引擎本身
    safe_free((void**)&engine);
}

/**
 * @brief 设置决策目标
 */
int decision_engine_set_objectives(DecisionEngine* engine,
                                   const DecisionObjective* objectives,
                                   size_t num_objectives) {
    if (!validate_engine(engine) || !objectives || num_objectives == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置决策目标：参数无效");
        return -1;
    }
    if (!engine->enabled) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置决策目标：决策引擎已禁用");
        return -1;
    }
    
    if (num_objectives > engine->config.max_objectives) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置决策目标：目标数量超过最大限制");
        return -1;
    }
    
    // 验证每个目标
    for (size_t i = 0; i < num_objectives; i++) {
        if (!validate_objective(&objectives[i])) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "设置决策目标：目标验证失败");
            return -1;
        }
    }
    
    // 清空现有目标
    for (size_t i = 0; i < engine->num_objectives; i++) {
        if (engine->objectives[i].name) {
            safe_free((void**)&engine->objectives[i].name);
        }
        if (engine->objectives[i].utility_params) {
            safe_free((void**)&engine->objectives[i].utility_params);
        }
    }
    
    // 复制新目标
    engine->num_objectives = 0;
    float total_weight = 0.0f;
    
    for (size_t i = 0; i < num_objectives; i++) {
        DecisionObjective* dest = &engine->objectives[i];
        const DecisionObjective* src = &objectives[i];
        
        // 复制名称
        dest->name = string_duplicate_nullable(src->name);
        if (!dest->name && src->name) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "设置决策目标：复制目标名称内存分配失败");
            return -1;
        }
        
        // 复制基本字段
        dest->weight = src->weight;
        dest->min_value = src->min_value;
        dest->max_value = src->max_value;
        dest->is_maximization = src->is_maximization;
        dest->utility_type = src->utility_type;
        dest->params_size = src->params_size;
        dest->custom_function = src->custom_function;
        dest->custom_user_data = src->custom_user_data;
        
        // 复制效用函数参数
        if (src->utility_params && src->params_size > 0) {
            dest->utility_params = (float*)safe_malloc(src->params_size * sizeof(float));
            if (!dest->utility_params) {
                safe_free((void**)&dest->name);
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                      "设置决策目标：复制效用函数参数内存分配失败");
                return -1;
            }
            memcpy(dest->utility_params, src->utility_params, src->params_size * sizeof(float));
        } else {
            dest->utility_params = NULL;
        }
        
        total_weight += src->weight;
        engine->num_objectives++;
    }
    
    // 归一化权重
    if (total_weight > 0.0f) {
        for (size_t i = 0; i < engine->num_objectives; i++) {
            engine->objectives[i].weight /= total_weight;
        }
    }
    
    engine->objectives_set = 1;
    return 0;
}

/**
 * @brief 设置决策约束
 */
int decision_engine_set_constraints(DecisionEngine* engine,
                                    const DecisionConstraint* constraints,
                                    size_t num_constraints) {
    if (!validate_engine(engine) || !constraints) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置决策约束：参数无效");
        return -1;
    }
    if (!engine->enabled) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置决策约束：决策引擎已禁用");
        return -1;
    }
    
    if (num_constraints > engine->config.max_constraints) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "设置决策约束：约束数量超过最大限制");
        return -1;
    }
    
    // 验证每个约束
    for (size_t i = 0; i < num_constraints; i++) {
        if (!validate_constraint(&constraints[i])) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "设置决策约束：约束验证失败");
            return -1;
        }
    }
    
    // 清空现有约束
    for (size_t i = 0; i < engine->num_constraints; i++) {
        if (engine->constraints[i].name) {
            safe_free((void**)&engine->constraints[i].name);
        }
    }
    
    // 复制新约束
    engine->num_constraints = 0;
    
    for (size_t i = 0; i < num_constraints; i++) {
        DecisionConstraint* dest = &engine->constraints[i];
        const DecisionConstraint* src = &constraints[i];
        
        // 复制名称
        dest->name = string_duplicate_nullable(src->name);
        if (!dest->name && src->name) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "设置决策约束：复制约束名称内存分配失败");
            return -1;
        }
        
        // 复制其他字段
        dest->lower_bound = src->lower_bound;
        dest->upper_bound = src->upper_bound;
        dest->is_equality = src->is_equality;
        dest->penalty_weight = src->penalty_weight;
        dest->attribute_index = src->attribute_index;
        
        // 更新约束惩罚项
        if (i < engine->config.max_constraints) {
            engine->constraint_penalties[i] = src->penalty_weight;
        }
        
        engine->num_constraints++;
    }
    
    engine->constraints_set = 1;
    return 0;
}

/**
 * @brief 添加决策备选方案
 */
int decision_engine_add_alternatives(DecisionEngine* engine,
                                     const DecisionAlternative* alternatives,
                                     size_t num_alternatives) {
    if (!validate_engine(engine) || !alternatives || num_alternatives == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "添加决策备选方案：参数无效");
        return -1;
    }
    if (!engine->enabled) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "添加决策备选方案：决策引擎已禁用");
        return -1;
    }
    
    if (engine->num_alternatives + num_alternatives > engine->config.max_alternatives) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "添加决策备选方案：方案数量超过最大限制");
        return -1;
    }
    
    // 验证每个方案
    for (size_t i = 0; i < num_alternatives; i++) {
        if (!validate_alternative(&alternatives[i])) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "添加决策备选方案：方案验证失败");
            return -1;
        }
    }
    
    // 复制方案
    for (size_t i = 0; i < num_alternatives; i++) {
        DecisionAlternative* dest = &engine->alternatives[engine->num_alternatives];
        const DecisionAlternative* src = &alternatives[i];
        
        if (allocate_alternative_copy(dest, src) != 0) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "添加决策备选方案：复制方案内存分配失败");
            return -1;
        }
        
        engine->num_alternatives++;
    }
    
    return 0;
}

/**
 * @brief 执行决策分析
 */
int decision_engine_analyze(DecisionEngine* engine, DecisionResult* result) {
    if (!validate_engine(engine) || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "执行决策分析：参数无效");
        return -1;
    }
    if (!engine->enabled) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "执行决策分析：决策引擎已禁用");
        return -1;
    }
    
    if (!engine->objectives_set) {
        selflnn_set_last_error(SELFLNN_ERROR_NOT_INITIALIZED, __func__, __FILE__, __LINE__,
                              "执行决策分析：决策目标未设置");
        return -1;
    }
    
    if (engine->num_alternatives == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "执行决策分析：无备选方案");
        return -1;
    }
    
    // 开始性能计时
    PerfTimer timer;
    perf_timer_start(&timer);
    
    // 初始化结果结构
    memset(result, 0, sizeof(DecisionResult));
    
    // 为每个方案计算目标值和效用
    for (size_t i = 0; i < engine->num_alternatives; i++) {
        DecisionAlternative* alt = &engine->alternatives[i];
        
        // 分配目标值数组
        if (!alt->objective_values) {
            alt->objective_values = (float*)safe_malloc(engine->num_objectives * sizeof(float));
            if (!alt->objective_values) {
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                      "执行决策分析：分配目标值数组内存失败");
                return -1;
            }
            alt->num_objectives = engine->num_objectives;
        }
        
        // M-028修复：计算每个目标的值并归一化
        for (size_t j = 0; j < engine->num_objectives && j < alt->num_attributes; j++) {
            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;
            for (size_t k = 0; k < alt->num_attributes; k++) {
                float weight = 1.0f;
                if (engine->objective_weights && j < engine->num_objectives && k < alt->num_attributes) {
                    weight = engine->objective_weights[j * alt->num_attributes + k];
                }
                weighted_sum += weight * alt->attribute_values[k];
                weight_sum += weight;
            }
            /* 归一化目标值（除以权重和，使各目标值量纲一致） */
            alt->objective_values[j] = (weight_sum > 1e-10f) ?
                weighted_sum / weight_sum : weighted_sum;
            /* 调用效用函数进行非线性变换 */
            if (engine->objectives && j < engine->num_objectives) {
                alt->objective_values[j] = compute_single_utility(&engine->objectives[j],
                    alt->objective_values[j]);
            }
        }
        
        // 计算约束违反程度
        alt->constraint_violation = compute_constraint_violation(engine, alt);
        alt->is_feasible = (alt->constraint_violation <= engine->config.epsilon) ? 1 : 0;
        
        // 计算总体效用（考虑约束违反惩罚）
        alt->overall_utility = compute_overall_utility(engine, alt);
        if (!alt->is_feasible) {
            // 对不可行方案施加惩罚
            alt->overall_utility -= alt->constraint_violation * 100.0f;
        }
    }
    
    // 根据决策类型选择最佳方案
    int best_index = -1;
    float best_utility = -FLT_MAX;
    
    if (engine->config.decision_type == DECISION_TYPE_SINGLE_OBJECTIVE) {
        // 单目标决策：选择总体效用最高的可行方案
        for (size_t i = 0; i < engine->num_alternatives; i++) {
            DecisionAlternative* alt = &engine->alternatives[i];
            if (alt->is_feasible && alt->overall_utility > best_utility) {
                best_utility = alt->overall_utility;
                best_index = (int)i;
            }
        }
        
        // 如果没有可行方案，选择约束违反最小的方案
        if (best_index == -1) {
            float min_violation = FLT_MAX;
            for (size_t i = 0; i < engine->num_alternatives; i++) {
                DecisionAlternative* alt = &engine->alternatives[i];
                if (alt->constraint_violation < min_violation) {
                    min_violation = alt->constraint_violation;
                    best_index = (int)i;
                }
            }
        }
    } else if (engine->config.decision_type == DECISION_TYPE_MULTI_OBJECTIVE) {
        // 多目标决策：计算帕累托前沿
        if (engine->config.enable_pareto_optimization) {
            // 分配帕累托前沿数组
            size_t max_pareto = engine->num_alternatives;
            DecisionAlternative* pareto_front = (DecisionAlternative*)safe_malloc(
                max_pareto * sizeof(DecisionAlternative));
            if (!pareto_front) {
                selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                      "执行决策分析：分配帕累托前沿数组内存失败");
                return -1;
            }
            
            // 计算帕累托前沿
            result->pareto_size = decision_engine_compute_pareto_front(
                engine, engine->alternatives, engine->num_alternatives,
                pareto_front, max_pareto);
            
            if (result->pareto_size > 0) {
                result->pareto_front = (float*)safe_malloc(result->pareto_size * sizeof(float));
                if (!result->pareto_front) {
                    safe_free((void**)&pareto_front);
                    selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                          "执行决策分析：分配帕累托前沿值数组内存失败");
                    return -1;
                }
                
                // 复制帕累托前沿方案的总体效用
                for (size_t i = 0; i < result->pareto_size; i++) {
                    result->pareto_front[i] = pareto_front[i].overall_utility;
                }
                
                // 选择帕累托前沿中效用最高的方案作为最佳方案
                best_utility = -FLT_MAX;
                for (size_t i = 0; i < result->pareto_size; i++) {
                    if (pareto_front[i].overall_utility > best_utility) {
                        best_utility = pareto_front[i].overall_utility;
                        best_index = (int)i;
                    }
                }
            }
            
            // 释放临时帕累托前沿数组
            for (size_t i = 0; i < result->pareto_size; i++) {
                free_alternative_internal(&pareto_front[i]);
            }
            safe_free((void**)&pareto_front);
        } else {
            // 不使用帕累托优化：使用加权和法
            best_utility = -FLT_MAX;
            for (size_t i = 0; i < engine->num_alternatives; i++) {
                DecisionAlternative* alt = &engine->alternatives[i];
                if (alt->is_feasible && alt->overall_utility > best_utility) {
                    best_utility = alt->overall_utility;
                    best_index = (int)i;
                }
            }
        }
    } else if (engine->config.decision_type == DECISION_TYPE_CONSTRAINED) {
        // 约束决策：使用罚函数法处理约束优化问题
        // 目标 = 总体效用 - 约束违反惩罚之和
        // 每个约束可能有不同的惩罚权重
        for (size_t i = 0; i < engine->num_alternatives; i++) {
            DecisionAlternative* alt = &engine->alternatives[i];
            
            // 计算加权约束违反惩罚
            float total_penalty = 0.0f;
            for (size_t c = 0; c < engine->num_constraints; c++) {
                DecisionConstraint* constraint = &engine->constraints[c];
                float value = (constraint->attribute_index >= 0 && 
                              (size_t)constraint->attribute_index < alt->num_attributes)
                              ? alt->attribute_values[constraint->attribute_index] : 0.0f;
                
                float violation = 0.0f;
                if (constraint->is_equality) {
                    // 等式约束：|value - (lower_bound+upper_bound)/2|
                    float target = (constraint->lower_bound + constraint->upper_bound) * 0.5f;
                    violation = fabsf(value - target);
                } else {
                    // 不等式约束：低于下界或高于上界的部分
                    if (value < constraint->lower_bound) {
                        violation = constraint->lower_bound - value;
                    } else if (value > constraint->upper_bound) {
                        violation = value - constraint->upper_bound;
                    }
                }
                total_penalty += violation * constraint->penalty_weight;
            }
            
            float penalized_utility = alt->overall_utility - total_penalty;
            if (penalized_utility > best_utility) {
                best_utility = penalized_utility;
                best_index = (int)i;
            }
        }
    } else if (engine->config.decision_type == DECISION_TYPE_SEQUENTIAL) {
        // 序列决策：基于余弦相似度的效用累加评估（非决策树结构）
        // 使用多阶段评估：当前状态向量与各候选方案向量的余弦相似度 × 效用权重
        // 使用动态规划评估决策序列的累积效用
        // 每个方案视为一个决策步骤，考虑时间折扣因子
        
        size_t num_alts = engine->num_alternatives;
        if (num_alts == 0) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "序列决策：无可选方案");
            return -1;
        }
        
        float discount_factor = 0.9f;
        float gamma = discount_factor;
        
        // 为每个方案计算累积折扣效用
        float* cumulative_utilities = (float*)safe_malloc(num_alts * sizeof(float));
        if (!cumulative_utilities) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "序列决策：分配累积效用数组失败");
            return -1;
        }
        
        for (size_t i = 0; i < num_alts; i++) {
            DecisionAlternative* alt = &engine->alternatives[i];
            cumulative_utilities[i] = alt->overall_utility;
            
            // 状态转移评估：基于方案间的相似度预测后续效用
            // 使用属性相似度作为状态转移概率的近似
            for (size_t j = 0; j < num_alts; j++) {
                if (i == j) continue;
                DecisionAlternative* next = &engine->alternatives[j];
                
                float similarity = 0.0f;
                size_t min_attrs = (alt->num_attributes < next->num_attributes) 
                                   ? alt->num_attributes : next->num_attributes;
                if (min_attrs > 0) {
                    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
                    for (size_t k = 0; k < min_attrs; k++) {
                        dot += alt->attribute_values[k] * next->attribute_values[k];
                        norm1 += alt->attribute_values[k] * alt->attribute_values[k];
                        norm2 += next->attribute_values[k] * next->attribute_values[k];
                    }
                    float denom = sqrtf(norm1) * sqrtf(norm2);
                    similarity = (denom > 1e-10f) ? dot / denom : 0.0f;
                }
                
                if (similarity > 0.5f) {
                    cumulative_utilities[i] += gamma * similarity * next->overall_utility;
                }
            }
            
            if (cumulative_utilities[i] > best_utility) {
                best_utility = cumulative_utilities[i];
                best_index = (int)i;
            }
        }
        
        safe_free((void**)&cumulative_utilities);
    } else if (engine->config.decision_type == DECISION_TYPE_GAME_THEORETIC) {
        // 博弈论决策：纳什均衡求解
        // 构建效用矩阵，使用迭代最佳响应算法寻找纯策略纳什均衡
        // 支持零和博弈和一般和博弈
        
        size_t num_alts = engine->num_alternatives;
        if (num_alts < 2) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "博弈论决策：至少需要2个方案");
            return -1;
        }
        
        float* payoff_matrix = (float*)safe_malloc(num_alts * num_alts * sizeof(float));
        if (!payoff_matrix) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "博弈论决策：分配效用矩阵失败");
            return -1;
        }
        
        // 构建效用矩阵：第i个智能体选择方案i，第j个智能体选择方案j的效用
        for (size_t i = 0; i < num_alts; i++) {
            for (size_t j = 0; j < num_alts; j++) {
                float payoff = 0.0f;
                if (i == j) {
                    // 合作情况：自身效用
                    payoff = engine->alternatives[i].overall_utility;
                } else {
                    // 竞争情况：效用差值（零和博弈近似）
                    float self_util = engine->alternatives[i].overall_utility;
                    float other_util = engine->alternatives[j].overall_utility;
                    payoff = self_util - other_util * 0.5f;
                }
                payoff_matrix[i * num_alts + j] = payoff;
            }
        }
        
        // 迭代最佳响应（IBR）：寻找纯策略纳什均衡
        int* strategy_profile = (int*)safe_calloc(num_alts, sizeof(int));
        int* best_responses = (int*)safe_malloc(num_alts * sizeof(int));
        if (!strategy_profile || !best_responses) {
            safe_free((void**)&payoff_matrix);
            safe_free((void**)&strategy_profile);
            safe_free((void**)&best_responses);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "博弈论决策：分配策略数组失败");
            return -1;
        }
        
        // 初始化策略：每个智能体的初始策略索引
        for (size_t i = 0; i < num_alts; i++) {
            strategy_profile[i] = (int)i;
        }
        
        int converged = 0;
        for (int iter = 0; iter < 100 && !converged; iter++) {
            converged = 1;
            for (size_t i = 0; i < num_alts; i++) {
                float best_payoff = -FLT_MAX;
                int best_action = strategy_profile[i];
                
                for (size_t a = 0; a < num_alts; a++) {
                    float current_payoff = payoff_matrix[a * num_alts + strategy_profile[i]];
                    if (current_payoff > best_payoff) {
                        best_payoff = current_payoff;
                        best_action = (int)a;
                    }
                }
                
                best_responses[i] = best_action;
                if (best_action != strategy_profile[i]) {
                    converged = 0;
                }
            }
            
            if (!converged) {
                for (size_t i = 0; i < num_alts; i++) {
                    strategy_profile[i] = best_responses[i];
                }
            }
        }
        
        // 计算每个方案的纳什均衡得分
        for (size_t i = 0; i < num_alts; i++) {
            float nash_score = 0.0f;
            int is_equilibrium = 1;
            
            for (size_t j = 0; j < num_alts; j++) {
                if (strategy_profile[j] != (int)i) {
                    is_equilibrium = 0;
                    break;
                }
            }
            
            nash_score = payoff_matrix[i * num_alts + strategy_profile[i]];
            if (is_equilibrium) {
                nash_score += 10.0f;
            }
            
            if (nash_score > best_utility) {
                best_utility = nash_score;
                best_index = (int)i;
            }
        }
        
        safe_free((void**)&payoff_matrix);
        safe_free((void**)&strategy_profile);
        safe_free((void**)&best_responses);
    }
    
    // 设置最佳方案
    if (best_index >= 0) {
        // 复制最佳方案
        result->best_alternative = (DecisionAlternative*)safe_malloc(sizeof(DecisionAlternative));
        if (!result->best_alternative) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "执行决策分析：分配最佳方案内存失败");
            return -1;
        }
        
        if (allocate_alternative_copy(result->best_alternative, 
                                     &engine->alternatives[best_index]) != 0) {
            safe_free((void**)&result->best_alternative);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "执行决策分析：复制最佳方案内存失败");
            return -1;
        }
    }
    
    // 复制所有方案到结果中
    result->alternatives = (DecisionAlternative*)safe_malloc(
        engine->num_alternatives * sizeof(DecisionAlternative));
    if (!result->alternatives) {
        if (result->best_alternative) {
            decision_alternative_free(result->best_alternative);
        }
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "执行决策分析：分配方案数组内存失败");
        return -1;
    }
    
    result->num_alternatives = engine->num_alternatives;
    for (size_t i = 0; i < engine->num_alternatives; i++) {
        if (allocate_alternative_copy(&result->alternatives[i], 
                                     &engine->alternatives[i]) != 0) {
            // 清理已分配的内存
            for (size_t j = 0; j < i; j++) {
                free_alternative_internal(&result->alternatives[j]);
            }
            safe_free((void**)&result->alternatives);
            if (result->best_alternative) {
                decision_alternative_free(result->best_alternative);
            }
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "执行决策分析：复制方案数组内存失败");
            return -1;
        }
    }
    
    // 计算决策置信度（基于最佳方案与其他方案的效用差距）
    if (engine->num_alternatives > 1 && best_index >= 0) {
        float second_best_utility = -FLT_MAX;
        for (size_t i = 0; i < engine->num_alternatives; i++) {
            if ((int)i != best_index && engine->alternatives[i].overall_utility > second_best_utility) {
                second_best_utility = engine->alternatives[i].overall_utility;
            }
        }
        
        if (best_utility > second_best_utility) {
            float utility_range = best_utility - second_best_utility;
            float max_utility_range = fabsf(best_utility) + fabsf(second_best_utility);
            if (max_utility_range > 0.0f) {
                result->decision_confidence = utility_range / max_utility_range;
            } else {
                result->decision_confidence = 0.5f;
            }
        } else {
            result->decision_confidence = 0.5f;
        }
    } else {
        result->decision_confidence = 1.0f;
    }
    
    // 计算计算时间
    uint64_t elapsed_ns = perf_timer_stop(&timer);
    result->computation_time_ms = (float)elapsed_ns / 1000000.0f;
    
    // 更新引擎统计
    engine->total_computation_time_ns += elapsed_ns;
    engine->total_decisions_made++;
    
    return 0;
}

/**
 * @brief 计算多目标帕累托前沿
 */
int decision_engine_compute_pareto_front(DecisionEngine* engine,
                                         const DecisionAlternative* alternatives,
                                         size_t num_alternatives,
                                         DecisionAlternative* pareto_front,
                                         size_t max_pareto_size) {
    if (!engine || !alternatives || !pareto_front || num_alternatives == 0 || max_pareto_size == 0) {
        return -1;
    }
    if (!engine->enabled) {
        return -1;
    }
    
    size_t pareto_count = 0;
    
    // 遍历所有方案
    for (size_t i = 0; i < num_alternatives; i++) {
        const DecisionAlternative* current = &alternatives[i];
        int is_pareto_efficient = 1;
        
        // 检查当前方案是否被其他方案支配
        for (size_t j = 0; j < num_alternatives; j++) {
            if (i == j) continue;
            
            const DecisionAlternative* other = &alternatives[j];
            
            // 检查other是否支配current
            if (is_pareto_dominant(other, current, engine->num_objectives)) {
                is_pareto_efficient = 0;
                break;
            }
        }
        
        // 如果是帕累托有效方案，添加到前沿
        if (is_pareto_efficient && pareto_count < max_pareto_size) {
            // 复制方案到帕累托前沿
            if (allocate_alternative_copy(&pareto_front[pareto_count], current) == 0) {
                pareto_count++;
            }
        }
    }
    
    return (int)pareto_count;
}

/**
 * @brief 计算效用函数值
 */
float decision_engine_compute_utility(UtilityFunctionType utility_type,
                                      float value,
                                      const float* params,
                                      size_t params_size) {
    // 参数验证
    if (isnan(value) || isinf(value)) {
        return 0.0f;
    }
    
    switch (utility_type) {
        case UTILITY_LINEAR: {
            // 线性效用函数: u(x) = a*x + b
            float a = (params && params_size >= 1) ? params[0] : 1.0f;
            float b = (params && params_size >= 2) ? params[1] : 0.0f;
            return a * value + b;
        }
        
        case UTILITY_EXPONENTIAL: {
            // 指数效用函数: u(x) = a * exp(b*x) + c
            float a = (params && params_size >= 1) ? params[0] : 1.0f;
            float b = (params && params_size >= 2) ? params[1] : 0.1f;
            float c = (params && params_size >= 3) ? params[2] : 0.0f;
            return a * expf(b * value) + c;
        }
        
        case UTILITY_LOGARITHMIC: {
            // 对数效用函数: u(x) = a * log(b*x + c) + d
            if (value <= 0.0f) return 0.0f;
            float a = (params && params_size >= 1) ? params[0] : 1.0f;
            float b = (params && params_size >= 2) ? params[1] : 1.0f;
            float c = (params && params_size >= 3) ? params[2] : 1.0f;
            /* ZSFA-FIX-F005: 修复off-by-one，params[4]→params[3]，索引从0开始第4个参数应为索引3 */
            float d = (params && params_size >= 4) ? params[3] : 0.0f;
            return a * logf(b * value + c) + d;
        }
        
        case UTILITY_SIGMOID: {
            // S型效用函数: u(x) = a / (1 + exp(-b*(x - c))) + d
            float a = (params && params_size >= 1) ? params[0] : 1.0f;
            float b = (params && params_size >= 2) ? params[1] : 1.0f;
            float c = (params && params_size >= 3) ? params[2] : 0.0f;
            float d = (params && params_size >= 4) ? params[3] : 0.0f;
            return a / (1.0f + expf(-b * (value - c))) + d;
        }
        
        case UTILITY_QUADRATIC: {
            // 二次效用函数: u(x) = a*x² + b*x + c
            float a = (params && params_size >= 1) ? params[0] : 0.0f;
            float b = (params && params_size >= 2) ? params[1] : 1.0f;
            float c = (params && params_size >= 3) ? params[2] : 0.0f;
            return a * value * value + b * value + c;
        }
        
        case UTILITY_CUSTOM: {
            // 自定义效用函数：完整实现支持参数化多项式函数
            // 如果没有参数，使用线性函数作为默认
            // 如果提供参数，使用多项式：sum(params[i] * value^i)
            if (!params || params_size == 0) {
                // 默认线性函数
                return value;
            }
            
            // 计算多项式值
            float result = 0.0f;
            float power = 1.0f;
            for (size_t i = 0; i < params_size; i++) {
                result += params[i] * power;
                power *= value;
            }
            return result;
        }
        
        default:
            return value;  // 默认返回原始值
    }
}

/**
 * @brief 释放决策结果内存
 */
void decision_result_free(DecisionResult* result) {
    if (!result) {
        return;
    }
    
    if (result->best_alternative) {
        decision_alternative_free(result->best_alternative);
    }
    
    if (result->alternatives) {
        for (size_t i = 0; i < result->num_alternatives; i++) {
            decision_alternative_free(&result->alternatives[i]);
        }
        safe_free((void**)&result->alternatives);
    }
    
    if (result->pareto_front) {
        safe_free((void**)&result->pareto_front);
    }
    
    memset(result, 0, sizeof(DecisionResult));
}

/**
 * @brief 释放决策备选方案内存
 */
void decision_alternative_free(DecisionAlternative* alternative) {
    if (!alternative) {
        return;
    }
    
    free_alternative_internal(alternative);
}

/* ==================== 内部辅助函数实现 ==================== */

/**
 * @brief 验证引擎有效性
 */
static int validate_engine(const DecisionEngine* engine) {
    return (engine && engine->is_initialized) ? 1 : 0;
}

/**
 * @brief 验证目标有效性
 */
static int validate_objective(const DecisionObjective* objective) {
    if (!objective) {
        return 0;
    }
    
    // 检查权重非负
    if (objective->weight < 0.0f) {
        return 0;
    }
    
    // 检查值范围有效性
    if (objective->min_value > objective->max_value) {
        return 0;
    }
    
    // 检查效用函数参数
    if (objective->utility_params && objective->params_size == 0) {
        return 0;
    }
    
    return 1;
}

/**
 * @brief 验证约束有效性
 */
static int validate_constraint(const DecisionConstraint* constraint) {
    if (!constraint) {
        return 0;
    }
    
    // 检查边界有效性
    if (constraint->lower_bound > constraint->upper_bound) {
        return 0;
    }
    
    // 检查惩罚权重非负
    if (constraint->penalty_weight < 0.0f) {
        return 0;
    }
    
    return 1;
}

/**
 * @brief 验证方案有效性
 */
static int validate_alternative(const DecisionAlternative* alternative) {
    if (!alternative) {
        return 0;
    }
    
    // 检查属性数组
    if (!alternative->attribute_values || alternative->num_attributes == 0) {
        return 0;
    }
    
    return 1;
}

/**
 * @brief 计算单个目标的效用值
 */
static float compute_single_utility(const DecisionObjective* objective, float value) {
    // 归一化值到[0,1]范围
    float normalized_value = 0.0f;
    if (objective->max_value > objective->min_value) {
        normalized_value = (value - objective->min_value) / (objective->max_value - objective->min_value);
        normalized_value = fmaxf(0.0f, fminf(1.0f, normalized_value));
    }
    
    // 如果不是最大化目标，则取反
    if (!objective->is_maximization) {
        normalized_value = 1.0f - normalized_value;
    }
    
    // 计算效用
    float utility = 0.0f;
    
    if (objective->utility_type == UTILITY_CUSTOM && objective->custom_function) {
        // 自定义效用函数：调用用户提供的函数
        utility = objective->custom_function(normalized_value,
                                            objective->utility_params,
                                            objective->params_size,
                                            objective->custom_user_data);
    } else {
        // 标准效用函数
        utility = decision_engine_compute_utility(objective->utility_type,
                                                 normalized_value,
                                                 objective->utility_params,
                                                 objective->params_size);
    }
    
    return utility;
}

/**
 * @brief 计算总体效用
 */
static float compute_overall_utility(const DecisionEngine* engine, 
                                     const DecisionAlternative* alternative) {
    if (!engine || !alternative || !alternative->objective_values) {
        return 0.0f;
    }
    
    /* ZSF-033修复: objective_values已通过compute_single_utility(L543)进行了
     * 非线性效用变换，此处仅做加权求和，不再重复调用compute_single_utility。
     * 原代码在此处再次调用导致效用函数被应用两次（双重变换）。 */
    float total_utility = 0.0f;
    
    for (size_t i = 0; i < engine->num_objectives && i < alternative->num_objectives; i++) {
        total_utility += engine->objectives[i].weight * alternative->objective_values[i];
    }
    
    return total_utility;
}

/**
 * @brief 计算约束违反程度
 */
static float compute_constraint_violation(const DecisionEngine* engine,
                                          const DecisionAlternative* alternative) {
    if (!engine || !alternative || engine->num_constraints == 0) {
        return 0.0f;
    }
    
    float total_violation = 0.0f;
    
    // 约束评估：完整实现，支持属性索引映射和多属性约束
    for (size_t i = 0; i < engine->num_constraints; i++) {
        const DecisionConstraint* constraint = &engine->constraints[i];
        
        // 获取约束对应的属性值
        float value = 0.0f;
        int attr_idx = constraint->attribute_index;
        
        if (attr_idx >= 0 && attr_idx < (int)alternative->num_attributes) {
            // 使用指定的属性索引
            value = alternative->attribute_values[attr_idx];
        } else if (attr_idx == -1) {
            // 自定义约束：使用所有属性值的加权平均进行综合评估
            // 权重基于约束的惩罚权重，各属性权重均等分配
            if (alternative->num_attributes > 0) {
                float weighted_sum = 0.0f;
                for (size_t k = 0; k < alternative->num_attributes; k++) {
                    weighted_sum += alternative->attribute_values[k];
                }
                value = weighted_sum / (float)alternative->num_attributes;
            } else {
                value = 0.0f;
            }
        } else {
            // 无效属性索引，跳过该约束
            continue;
        }
        
        float violation = 0.0f;
        
        if (constraint->is_equality) {
            // 等式约束：|value - target|，target取上下界平均值
            float target = (constraint->lower_bound + constraint->upper_bound) / 2.0f;
            violation = fabsf(value - target);
        } else {
            // 不等式约束
            if (value < constraint->lower_bound) {
                violation = constraint->lower_bound - value;
            } else if (value > constraint->upper_bound) {
                violation = value - constraint->upper_bound;
            }
        }
        
        total_violation += violation * constraint->penalty_weight;
    }
    
    return total_violation;
}

/**
 * @brief 检查方案A是否支配方案B
 */
static int is_pareto_dominant(const DecisionAlternative* a, 
                              const DecisionAlternative* b,
                              size_t num_objectives) {
    if (!a || !b || !a->objective_values || !b->objective_values) {
        return 0;
    }
    
    int at_least_one_better = 0;
    
    for (size_t i = 0; i < num_objectives; i++) {
        float a_val = a->objective_values[i];
        float b_val = b->objective_values[i];
        
        if (a_val < b_val) {
            // A在目标i上比B差，所以A不支配B
            return 0;
        } else if (a_val > b_val) {
            // A在目标i上比B好
            at_least_one_better = 1;
        }
        // 如果相等，继续检查下一个目标
    }
    
    return at_least_one_better;
}

/**
 * @brief 分配并复制方案
 */
static int allocate_alternative_copy(DecisionAlternative* dest, 
                                     const DecisionAlternative* src) {
    if (!dest || !src) {
        return -1;
    }

    memset(dest, 0, sizeof(DecisionAlternative));

    /* F-038: 使用统一深拷贝宏，与copy_knowledge_entry共享同一模式 */
    DEEP_COPY_STRING_SAFE(dest->id, src->id, safe_free);

    DEEP_COPY_FLOAT_ARRAY(dest->attribute_values, dest->num_attributes,
                          src->attribute_values, src->num_attributes);

    DEEP_COPY_FLOAT_ARRAY(dest->objective_values, dest->num_objectives,
                          src->objective_values, src->num_objectives);

    DEEP_COPY_SCALAR(dest->overall_utility, src->overall_utility);
    DEEP_COPY_SCALAR(dest->constraint_violation, src->constraint_violation);
    DEEP_COPY_SCALAR(dest->is_feasible, src->is_feasible);

    return 0;

deep_copy_cleanup:
    DEEP_COPY_CLEANUP_FREE(dest->id);
    DEEP_COPY_CLEANUP_FREE(dest->attribute_values);
    DEEP_COPY_CLEANUP_FREE(dest->objective_values);
    return -1;
}

/**
 * @brief 释放方案内部内存
 */
static void free_alternative_internal(DecisionAlternative* alternative) {
    if (!alternative) {
        return;
    }
    
    if (alternative->id) {
        safe_free((void**)&alternative->id);
    }
    
    if (alternative->attribute_values) {
        safe_free((void**)&alternative->attribute_values);
    }
    
    if (alternative->objective_values) {
        safe_free((void**)&alternative->objective_values);
    }
    
    memset(alternative, 0, sizeof(DecisionAlternative));
}

/* ============================================================================
 * 决策审计日志实现
 * ============================================================================ */

int decision_audit_log_init(DecisionAuditLog* log, size_t max_entries) {
    if (!log) return -1;
    memset(log, 0, sizeof(DecisionAuditLog));
    log->max_entries = (max_entries > 0 && max_entries <= 256) ? max_entries : 256;
    log->is_enabled = 1;
    log->entry_count = 0;
    return 0;
}

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
                              float execution_time_ms) {
    if (!log || !log->is_enabled || !description) return -1;

    size_t idx = log->entry_count % log->max_entries;
    DecisionLogEntry* entry = &log->entries[idx];
    memset(entry, 0, sizeof(DecisionLogEntry));

    entry->log_type = log_type;

    time_t now = time(NULL);
    struct tm* tm_info;
#ifdef _WIN32
    tm_info = localtime(&now);
#else
    struct tm tm_buf;
    tm_info = localtime_r(&now, &tm_buf);
#endif
    if (tm_info) {
        strftime(entry->timestamp, sizeof(entry->timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    } else {
        snprintf(entry->timestamp, sizeof(entry->timestamp), "%ld", (long)now);
    }

    strncpy(entry->description, description, sizeof(entry->description) - 1);
    entry->description[sizeof(entry->description) - 1] = '\0';

    if (objective_labels) {
        strncpy(entry->objective_labels, objective_labels, sizeof(entry->objective_labels) - 1);
        entry->objective_labels[sizeof(entry->objective_labels) - 1] = '\0';
    }

    entry->objective_count = (objective_count > 16) ? 16 : objective_count;
    if (objective_values) {
        for (size_t i = 0; i < entry->objective_count; i++) {
            entry->objective_values[i] = objective_values[i];
        }
    }

    if (chosen_alternative) {
        strncpy(entry->chosen_alternative, chosen_alternative, sizeof(entry->chosen_alternative) - 1);
        entry->chosen_alternative[sizeof(entry->chosen_alternative) - 1] = '\0';
    }
    entry->chosen_utility = chosen_utility;
    entry->total_alternatives = total_alternatives;
    entry->is_autonomous = is_autonomous;
    entry->execution_time_ms = execution_time_ms;

    log->entry_count++;
    return 0;
}

int decision_audit_log_export_json(const DecisionAuditLog* log,
                                   char* json_buffer, size_t buffer_size) {
    if (!log || !json_buffer || buffer_size == 0) return -1;

    size_t pos = 0;
    int written = snprintf(json_buffer + pos, buffer_size - pos,
                          "{\"enabled\":%d,\"count\":%zu,\"entries\":[",
                          log->is_enabled, log->entry_count);
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    size_t start = (log->entry_count > log->max_entries)
        ? log->entry_count - log->max_entries : 0;
    size_t total = log->entry_count < log->max_entries ? log->entry_count : log->max_entries;

    const char* type_names[] = {
        "analysis", "execution", "evaluation", "override", "error"
    };

    for (size_t i = 0; i < total; i++) {
        size_t idx = (start + i) % log->max_entries;
        const DecisionLogEntry* e = &log->entries[idx];

        if (i > 0) {
            written = snprintf(json_buffer + pos, buffer_size - pos, ",");
            if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
            pos += (size_t)written;
        }

        written = snprintf(json_buffer + pos, buffer_size - pos,
                          "{\"type\":\"%s\",\"time\":\"%s\",\"desc\":\"%s\","
                          "\"choices\":\"%s\",\"utility\":%.4f,"
                          "\"alts\":%d,\"auto\":%d,\"exec_ms\":%.2f}",
                          type_names[e->log_type], e->timestamp, e->description,
                          e->chosen_alternative, e->chosen_utility,
                          e->total_alternatives, e->is_autonomous, e->execution_time_ms);
        if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
        pos += (size_t)written;
    }

    written = snprintf(json_buffer + pos, buffer_size - pos, "]}");
    if (written < 0 || (size_t)written >= buffer_size - pos) return -1;
    pos += (size_t)written;

    return (int)pos;
}

int decision_audit_log_recent(const DecisionAuditLog* log,
                              DecisionLogEntry* entries, size_t max_count) {
    if (!log || !entries || max_count == 0) return -1;

    size_t available = log->entry_count < log->max_entries ? log->entry_count : log->max_entries;
    size_t count = available < max_count ? available : max_count;

    size_t start = (log->entry_count > log->max_entries)
        ? log->entry_count - log->max_entries : 0;

    for (size_t i = 0; i < count; i++) {
        size_t idx = (start + (available - count) + i) % log->max_entries;
        memcpy(&entries[i], &log->entries[idx], sizeof(DecisionLogEntry));
    }

    return (int)count;
}

void decision_audit_log_clear(DecisionAuditLog* log) {
    if (!log) return;
    memset(log->entries, 0, sizeof(log->entries));
    log->entry_count = 0;
}

/* ============================
 * 能力开关实现（P3.3）
 * ============================ */

void decision_engine_enable(DecisionEngine* engine) {
    if (engine) {
        engine->enabled = 1;
    }
}

void decision_engine_disable(DecisionEngine* engine) {
    if (engine) {
        engine->enabled = 0;
    }
}

int decision_engine_is_enabled(const DecisionEngine* engine) {
    return (engine && engine->enabled) ? 1 : 0;
}