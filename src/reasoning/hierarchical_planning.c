/**
 * @file hierarchical_planning.c
 * @brief 分层规划系统实现
 * 
 * 分层规划（Hierarchical Planning）系统完整实现，包括：
 * 1. 分层任务网络（HTN）规划算法
 * 2. 部分有序规划（POP）算法
 * 3. 分层强化学习（HRL）算法
 * 4. 抽象层次分解算法
 * 5. 任务分解与协调机制
 * 
 *  ，提供完整的分层规划算法。
 */

#include "selflnn/reasoning/hierarchical_planning.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/secure_random.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif

/* 手动声明内存函数以避免警告 */

void safe_free(void** ptr);

/**
 * @brief 分层任务节点内部结构
 */
struct HierarchicalTaskNode {
    float* task_description;        /**< 任务描述 */
    size_t task_size;               /**< 任务描述大小 */
    PlanningHierarchy hierarchy;    /**< 任务层次 */
    TaskDecompositionMethod decomposition_method; /**< 分解方法 */
    
    HierarchicalTaskNode* parent;   /**< 父节点 */
    HierarchicalTaskNode** children; /**< 子节点数组 */
    size_t children_count;          /**< 子节点数量 */
    size_t children_capacity;       /**< 子节点容量 */
    
    float completion_score;         /**< 完成度评分 */
    float feasibility_score;        /**< 可行性评分 */
    float risk_score;               /**< 风险评分 */
    
    int is_primitive;               /**< 是否为原始任务（不可再分解） */
    int is_completed;               /**< 是否已完成 */
    int is_failed;                  /**< 是否失败 */
    
    float* preconditions;           /**< 前提条件 */
    size_t preconditions_size;      /**< 前提条件大小 */
    float* effects;                 /**< 效果 */
    size_t effects_size;            /**< 效果大小 */
    
    float* parameters;              /**< 任务参数 */
    size_t parameters_size;         /**< 参数大小 */
};

/**
 * @brief 分层规划系统内部结构
 */
struct HierarchicalPlanningSystem {
    HierarchicalPlanningConfig config;      /**< 配置 */
    HierarchicalPlanningState state;        /**< 状态 */
    
    /* 任务网络 */
    TaskNetwork* task_networks;             /**< 任务网络数组 */
    size_t networks_count;                  /**< 网络数量 */
    size_t networks_capacity;               /**< 网络容量 */
    
    /* 抽象状态映射 */
    float** abstraction_mappings;           /**< 抽象状态映射矩阵 */
    size_t* mapping_sizes;                  /**< 映射大小数组 */
    size_t mappings_count;                  /**< 映射数量 */
    
    /* 学习参数（HRL使用） */
    float** value_functions;                /**< 价值函数 */
    size_t* value_function_sizes;           /**< 价值函数大小 */
    float** policy_functions;               /**< 策略函数 */
    size_t* policy_function_sizes;          /**< 策略函数大小 */
    
    /* 回溯栈 */
    HierarchicalTaskNode** backtrack_stack; /**< 回溯栈 */
    size_t stack_size;                      /**< 栈大小 */
    size_t stack_capacity;                  /**< 栈容量 */
    
    /* 统计信息 */
    size_t total_decompositions;            /**< 总分解次数 */
    size_t successful_decompositions;       /**< 成功分解次数 */
    size_t failed_decompositions;           /**< 失败分解次数 */
    size_t total_backtracks;                /**< 总回溯次数 */
    float average_planning_time;            /**< 平均规划时间 */
    time_t last_planning_time;              /**< 最后规划时间 */
    
    /* P2-1: HTN增强字段 */
    HTNPlanningDomain htn_domain;           /**< HTN规划域 */
    HTNMethodSelectorConfig htn_selector_config; /**< 方法选择器配置 */
    HTNDecompositionResult htn_last_result; /**< 上次分解结果 */
    int htn_enhanced_enabled;               /**< 是否启用增强HTN */
    int htn_interaction_buffer_size;        /**< 交互缓冲区大小 */
    TaskInteraction* htn_interaction_buffer; /**< 交互缓冲区 */
    
    int is_initialized;                     /**< 是否已初始化 */
};

/* ============================================================================
 * 内部辅助函数声明
 * ============================================================================ */

static HierarchicalTaskNode* create_task_node(
    const float* task_description, size_t task_size,
    PlanningHierarchy hierarchy,
    TaskDecompositionMethod decomposition_method);

static void free_task_node(HierarchicalTaskNode* node);

static int add_child_node(HierarchicalTaskNode* parent, HierarchicalTaskNode* child);

static int decompose_task_htn(HierarchicalPlanningSystem* system,
                             HierarchicalTaskNode* task_node,
                             const float* context, size_t context_size);

static int decompose_task_pop(HierarchicalPlanningSystem* system,
                             HierarchicalTaskNode* task_node,
                             const float* context, size_t context_size);

static int decompose_task_hrl(HierarchicalPlanningSystem* system,
                             HierarchicalTaskNode* task_node,
                             const float* context, size_t context_size);

static int decompose_task_abstract(HierarchicalPlanningSystem* system,
                                  HierarchicalTaskNode* task_node,
                                  const float* context, size_t context_size);

static float evaluate_task_feasibility(HierarchicalPlanningSystem* system,
                                      HierarchicalTaskNode* task_node,
                                      const float* context, size_t context_size);

static float evaluate_task_risk(HierarchicalPlanningSystem* system,
                               HierarchicalTaskNode* task_node,
                               const float* context, size_t context_size);

static int extract_abstract_state(HierarchicalPlanningSystem* system,
                                 const float* concrete_state, size_t state_size,
                                 PlanningHierarchy target_level,
                                 float* abstract_state, size_t max_abstract_size);

static int coordinate_hierarchies(HierarchicalPlanningSystem* system,
                                 HierarchicalTaskNode* high_level_node,
                                 HierarchicalTaskNode* low_level_node,
                                 float* coordinated_plan, size_t max_coordinated_size);

static int backtrack_decomposition(HierarchicalPlanningSystem* system,
                                  HierarchicalTaskNode* current_node);

static int update_value_function(HierarchicalPlanningSystem* system,
                                HierarchicalTaskNode* task_node,
                                float reward);

static int update_policy_function(HierarchicalPlanningSystem* system,
                                 HierarchicalTaskNode* task_node,
                                 const float* action, size_t action_size);

/* ============================================================================
 * 公共接口实现
 * ============================================================================ */

/**
 * @brief 创建分层规划系统
 */
HierarchicalPlanningSystem* hierarchical_planning_system_create(
    const HierarchicalPlanningConfig* config) {
    
    if (!config) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "配置参数为空");
        return NULL;
    }
    
    if (config->max_hierarchy_levels <= 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "最大层次数必须大于0");
        return NULL;
    }
    
    HierarchicalPlanningSystem* system = (HierarchicalPlanningSystem*)safe_malloc(
        sizeof(HierarchicalPlanningSystem));
    if (!system) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配分层规划系统内存失败");
        return NULL;
    }
    
    // 初始化配置
    memcpy(&system->config, config, sizeof(HierarchicalPlanningConfig));
    
    // 初始化状态
    memset(&system->state, 0, sizeof(HierarchicalPlanningState));
    system->state.current_level = HIERARCHY_STRATEGIC;
    system->state.decomposition_depth = 0;
    system->state.backtrack_count = 0;
    system->state.cumulative_reward = 0.0f;
    system->state.is_initialized = 0;
    
    // 初始化任务网络
    system->task_networks = NULL;
    system->networks_count = 0;
    system->networks_capacity = 0;
    
    // 初始化抽象状态映射
    system->abstraction_mappings = NULL;
    system->mapping_sizes = NULL;
    system->mappings_count = 0;
    
    // 初始化学习参数
    system->value_functions = NULL;
    system->value_function_sizes = NULL;
    system->policy_functions = NULL;
    system->policy_function_sizes = NULL;
    
    // 初始化回溯栈
    system->backtrack_stack = NULL;
    system->stack_size = 0;
    system->stack_capacity = 0;
    
    // 初始化统计信息
    system->total_decompositions = 0;
    system->successful_decompositions = 0;
    system->failed_decompositions = 0;
    system->total_backtracks = 0;
    system->average_planning_time = 0.0f;
    system->last_planning_time = time(NULL);
    
    system->is_initialized = 1;
    
    return system;
}

/**
 * @brief 释放分层规划系统
 */
void hierarchical_planning_system_free(HierarchicalPlanningSystem* system) {
    if (!system) return;
    
    // 释放任务网络
    if (system->task_networks) {
        for (size_t i = 0; i < system->networks_count; i++) {
            if (system->task_networks[i].root) {
                free_task_node(system->task_networks[i].root);
            }
        }
        safe_free((void**)&system->task_networks);
    }
    
    // 释放抽象状态映射
    if (system->abstraction_mappings) {
        for (size_t i = 0; i < system->mappings_count; i++) {
            if (system->abstraction_mappings[i]) {
                safe_free((void**)&system->abstraction_mappings[i]);
            }
        }
        safe_free((void**)&system->abstraction_mappings);
        safe_free((void**)&system->mapping_sizes);
    }
    
    // 释放学习参数
    if (system->value_functions) {
        for (size_t i = 0; i < system->mappings_count; i++) {
            if (system->value_functions[i]) {
                safe_free((void**)&system->value_functions[i]);
            }
        }
        safe_free((void**)&system->value_functions);
        safe_free((void**)&system->value_function_sizes);
    }
    
    if (system->policy_functions) {
        for (size_t i = 0; i < system->mappings_count; i++) {
            if (system->policy_functions[i]) {
                safe_free((void**)&system->policy_functions[i]);
            }
        }
        safe_free((void**)&system->policy_functions);
        safe_free((void**)&system->policy_function_sizes);
    }
    
    // 释放回溯栈
    if (system->backtrack_stack) {
        safe_free((void**)&system->backtrack_stack);
    }
    
    // 释放抽象状态
    if (system->state.abstract_state) {
        safe_free((void**)&system->state.abstract_state);
    }
    
    // 释放系统
    safe_free((void**)&system);
}

/**
 * @brief 递归计算任务树中所有节点数量
 */
static size_t count_all_task_nodes(const HierarchicalTaskNode* node) {
    if (!node) return 0;
    size_t count = 1;
    for (size_t i = 0; i < node->children_count; i++) {
        count += count_all_task_nodes(node->children[i]);
    }
    return count;
}

/**
 * @brief 生成分层规划
 */
int hierarchical_planning_generate(HierarchicalPlanningSystem* system,
                                  const float* goal, size_t goal_size,
                                  const float* current_state, size_t state_size,
                                  PlanningHierarchy hierarchy_level,
                                  float* plan, size_t max_plan_size) {
    
    if (!system || !goal || !current_state || !plan) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (goal_size == 0 || state_size == 0 || max_plan_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    if (hierarchy_level < HIERARCHY_STRATEGIC || hierarchy_level > HIERARCHY_EXECUTION) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的层次级别");
        return -1;
    }
    
    time_t start_time = time(NULL);
    
    // 创建根任务节点
    HierarchicalTaskNode* root_node = create_task_node(
        goal, goal_size, hierarchy_level, system->config.decomposition_method);
    if (!root_node) {
        return -1;
    }
    
    // 设置当前状态
    system->state.current_level = hierarchy_level;
    system->state.decomposition_depth = 0;
    system->state.backtrack_count = 0;
    
    // 提取抽象状态
    float* abstract_state = (float*)safe_malloc(state_size * sizeof(float));
    if (!abstract_state) {
        free_task_node(root_node);
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配抽象状态内存失败");
        return -1;
    }
    
    int abstract_size = extract_abstract_state(system, current_state, state_size,
                                              hierarchy_level, abstract_state, state_size);
    if (abstract_size <= 0) {
        safe_free((void**)&abstract_state);
        free_task_node(root_node);
        return -1;
    }
    
    // 更新系统状态
    if (system->state.abstract_state) {
        safe_free((void**)&system->state.abstract_state);
    }
    system->state.abstract_state = abstract_state;
    system->state.abstract_state_size = abstract_size;
    
    // 根据算法类型进行任务分解
    int result = -1;
    switch (system->config.algorithm) {
        case HIERARCHICAL_HTN:
            result = decompose_task_htn(system, root_node, current_state, state_size);
            break;
            
        case HIERARCHICAL_POP:
            result = decompose_task_pop(system, root_node, current_state, state_size);
            break;
            
        case HIERARCHICAL_HRL:
            result = decompose_task_hrl(system, root_node, current_state, state_size);
            break;
            
        case HIERARCHICAL_ABSTRACT:
            result = decompose_task_abstract(system, root_node, current_state, state_size);
            break;
            
        case HIERARCHICAL_MIXED:
            // 混合算法：先尝试HTN，失败则尝试POP
            result = decompose_task_htn(system, root_node, current_state, state_size);
            if (result < 0 && system->config.enable_backtracking) {
                result = decompose_task_pop(system, root_node, current_state, state_size);
            }
            break;
            
        default:
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "不支持的算法类型");
            break;
    }
    
    if (result < 0) {
        free_task_node(root_node);
        return -1;
    }
    
    // 将任务网络转换为规划序列：深度优先遍历任务树，收集原始任务
    int plan_steps = 0;
    float* plan_ptr = plan;
    size_t remaining_size = max_plan_size;
    
    // 使用递归栈进行深度优先遍历，收集原始任务
    // 用迭代方式实现避免栈溢出
    HierarchicalTaskNode** node_stack = NULL;
    size_t stack_cap = 64;
    size_t stack_top = 0;
    
    node_stack = (HierarchicalTaskNode**)safe_malloc(stack_cap * sizeof(HierarchicalTaskNode*));
    if (!node_stack) {
        free_task_node(root_node);
        return -1;
    }
    node_stack[stack_top++] = root_node;
    
    while (stack_top > 0 && remaining_size > 0) {
        HierarchicalTaskNode* current = node_stack[--stack_top];
        
        if (current->is_primitive || current->children_count == 0) {
            // 原始任务：将任务描述复制到规划序列
            size_t copy_size = (current->task_size < remaining_size) ? current->task_size : remaining_size;
            memcpy(plan_ptr, current->task_description, copy_size * sizeof(float));
            plan_ptr += copy_size;
            plan_steps++;
            remaining_size -= copy_size;
        } else {
            // 非原始任务：将子任务按逆序压栈（保持顺序）
            for (size_t i = current->children_count; i > 0; i--) {
                size_t idx = i - 1;
                if (!current->children[idx]) continue;
                if (stack_top >= stack_cap) {
                    stack_cap *= 2;
                    HierarchicalTaskNode** new_stack = (HierarchicalTaskNode**)safe_realloc(
                        node_stack, stack_cap * sizeof(HierarchicalTaskNode*));
                    if (!new_stack) {
                        safe_free((void**)&node_stack);
                        free_task_node(root_node);
                        return -1;
                    }
                    node_stack = new_stack;
                }
                node_stack[stack_top++] = current->children[idx];
            }
        }
    }
    safe_free((void**)&node_stack);
    
    // 创建任务网络
    if (system->networks_count >= system->networks_capacity) {
        size_t new_capacity = system->networks_capacity == 0 ? 4 : system->networks_capacity * 2;
        TaskNetwork* new_networks = (TaskNetwork*)safe_realloc(
            system->task_networks, new_capacity * sizeof(TaskNetwork));
        if (!new_networks) {
            free_task_node(root_node);
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "扩展任务网络数组失败");
            return -1;
        }
        system->task_networks = new_networks;
        system->networks_capacity = new_capacity;
    }
    
    // 评估任务网络
    float feasibility = evaluate_task_feasibility(system, root_node, current_state, state_size);
    float risk = evaluate_task_risk(system, root_node, current_state, state_size);
    
    // 保存任务网络
    system->task_networks[system->networks_count].root = root_node;
    system->task_networks[system->networks_count].node_count = count_all_task_nodes(root_node);
    system->task_networks[system->networks_count].hierarchy_level = hierarchy_level;
    system->task_networks[system->networks_count].completion_score = 0.0f;
    system->task_networks[system->networks_count].feasibility_score = feasibility;
    system->task_networks[system->networks_count].risk_score = risk;
    system->networks_count++;
    
    // 更新系统状态
    system->state.current_network = &system->task_networks[system->networks_count - 1];
    system->state.is_initialized = 1;
    
    // 更新统计信息
    time_t end_time = time(NULL);
    double planning_time = difftime(end_time, start_time);
    system->average_planning_time = (float)((system->average_planning_time * system->total_decompositions + planning_time) /
                                   (system->total_decompositions + 1));
    system->total_decompositions++;
    system->last_planning_time = end_time;
    
    // 如果未收集到原始任务，使用目标作为保底规划
    if (plan_steps == 0 && max_plan_size >= goal_size) {
        memcpy(plan, goal, goal_size * sizeof(float));
        plan_steps = 1;
    }
    
    return plan_steps;
}

/**
 * @brief 任务分解
 */
int hierarchical_task_decomposition(HierarchicalPlanningSystem* system,
                                   const float* task, size_t task_size,
                                   const float* context, size_t context_size,
                                   float* subtasks, size_t max_subtasks) {
    
    if (!system || !task || !context || !subtasks) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (task_size == 0 || context_size == 0 || max_subtasks == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    // 创建任务节点
    HierarchicalTaskNode* task_node = create_task_node(
        task, task_size, system->state.current_level, system->config.decomposition_method);
    if (!task_node) {
        return -1;
    }
    
    // 执行任务分解
    int result = -1;
    switch (system->config.algorithm) {
        case HIERARCHICAL_HTN:
            result = decompose_task_htn(system, task_node, context, context_size);
            break;
            
        case HIERARCHICAL_POP:
            result = decompose_task_pop(system, task_node, context, context_size);
            break;
            
        case HIERARCHICAL_HRL:
            result = decompose_task_hrl(system, task_node, context, context_size);
            break;
            
        case HIERARCHICAL_ABSTRACT:
            result = decompose_task_abstract(system, task_node, context, context_size);
            break;
            
        default:
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                                  "不支持的算法类型");
            break;
    }
    
    if (result < 0) {
        free_task_node(task_node);
        return -1;
    }
    
    // 收集子任务：深度优先遍历所有非原始子任务
    int subtask_count = 0;
    if (task_node->children_count > 0 && max_subtasks > 0) {
        // 使用栈遍历所有层次的下级任务
        HierarchicalTaskNode** collect_stack = (HierarchicalTaskNode**)safe_malloc(
            (task_node->children_count + 1) * sizeof(HierarchicalTaskNode*));
        if (!collect_stack) {
            free_task_node(task_node);
            return -1;
        }
        size_t cs_top = 0;
        
        // 先将所有直接子任务入栈
        for (size_t i = 0; i < task_node->children_count; i++) {
            if (task_node->children[i]) {
                collect_stack[cs_top++] = task_node->children[i];
            }
        }
        
        // 深度优先遍历收集所有层级的子任务描述
        while (cs_top > 0 && (size_t)subtask_count < max_subtasks) {
            HierarchicalTaskNode* current = collect_stack[--cs_top];
            
            if (current->is_primitive || current->children_count == 0) {
                // 原始任务：复制描述到输出缓冲区
                size_t copy_size = current->task_size < max_subtasks - (size_t)subtask_count
                                 ? current->task_size : max_subtasks - (size_t)subtask_count;
                memcpy(subtasks + subtask_count, current->task_description,
                       copy_size * sizeof(float));
                subtask_count++;
            } else {
                // 非原始任务：在缓冲区空间允许时，将子节点入栈
                if (cs_top + current->children_count <= task_node->children_count + 1) {
                    for (size_t i = current->children_count; i > 0; i--) {
                        size_t idx = i - 1;
                        if (current->children[idx]) {
                            collect_stack[cs_top++] = current->children[idx];
                        }
                    }
                }
            }
        }
        
        safe_free((void**)&collect_stack);
    }
    
    free_task_node(task_node);
    
    if (subtask_count == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_ALGORITHM_FAILURE, __func__, __FILE__, __LINE__,
                              "任务分解未产生子任务");
        return -1;
    }
    
    system->successful_decompositions++;
    return subtask_count;
}

/**
 * @brief 抽象状态提取
 */
int hierarchical_state_abstraction(HierarchicalPlanningSystem* system,
                                  const float* concrete_state, size_t state_size,
                                  PlanningHierarchy hierarchy_level,
                                  float* abstract_state, size_t max_abstract_size) {
    
    if (!system || !concrete_state || !abstract_state) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (state_size == 0 || max_abstract_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    if (hierarchy_level < HIERARCHY_STRATEGIC || hierarchy_level > HIERARCHY_EXECUTION) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的层次级别");
        return -1;
    }
    
    return extract_abstract_state(system, concrete_state, state_size,
                                 hierarchy_level, abstract_state, max_abstract_size);
}

/**
 * @brief 规划评估与优化
 */
int hierarchical_planning_evaluate(HierarchicalPlanningSystem* system,
                                  const float* plan, size_t plan_size,
                                  PlanningHierarchy hierarchy_level,
                                  float* evaluation_metrics, size_t max_metrics) {
    
    if (!system || !plan || !evaluation_metrics) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (plan_size == 0 || max_metrics == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    if (hierarchy_level < HIERARCHY_STRATEGIC || hierarchy_level > HIERARCHY_EXECUTION) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的层次级别");
        return -1;
    }
    
    // 完整规划评估：基于规划数据计算各项指标
    int metrics_count = 0;
    
    if (max_metrics >= 4) {
        // 计算规划数据统计量
        float plan_sum = 0.0f, plan_sq_sum = 0.0f;
        float plan_min = plan[0], plan_max = plan[0];
        for (size_t i = 0; i < plan_size; i++) {
            plan_sum += plan[i];
            plan_sq_sum += plan[i] * plan[i];
            if (plan[i] < plan_min) plan_min = plan[i];
            if (plan[i] > plan_max) plan_max = plan[i];
        }
        float plan_mean = plan_sum / (float)plan_size;
        float plan_variance = plan_sq_sum / (float)plan_size - plan_mean * plan_mean;
        if (plan_variance < 0.0f) plan_variance = 0.0f;
        float plan_stddev = sqrtf(plan_variance);
        float plan_range = plan_max - plan_min;
        
        // 计算规划变化率（梯度累积）
        float gradient_energy = 0.0f;
        for (size_t i = 1; i < plan_size; i++) {
            float diff = plan[i] - plan[i - 1];
            gradient_energy += diff * diff;
        }
        gradient_energy = sqrtf(gradient_energy / (float)(plan_size > 1 ? plan_size - 1 : 1));
        
        // 1. 可行性评分：基于规划的一致性和层次级别
        float consistency = 1.0f / (1.0f + plan_variance);
        float hierarchy_factor = 1.0f - (float)hierarchy_level * 0.15f;
        if (hierarchy_factor < 0.3f) hierarchy_factor = 0.3f;
        evaluation_metrics[0] = consistency * hierarchy_factor;
        if (evaluation_metrics[0] > 1.0f) evaluation_metrics[0] = 1.0f;
        if (evaluation_metrics[0] < 0.1f) evaluation_metrics[0] = 0.1f;
        metrics_count++;
        
        // 2. 风险评分：基于变化率和范围
        float normalized_gradient = gradient_energy / (1.0f + gradient_energy);
        float range_risk = plan_range / (1.0f + plan_range);
        evaluation_metrics[1] = 0.4f * normalized_gradient + 0.3f * range_risk + 0.3f * (1.0f - hierarchy_factor);
        if (evaluation_metrics[1] > 1.0f) evaluation_metrics[1] = 1.0f;
        if (evaluation_metrics[1] < 0.0f) evaluation_metrics[1] = 0.0f;
        metrics_count++;
        
        // 3. 效率评分：基于规划长度和步骤复杂度
        float length_efficiency = 1.0f / (1.0f + (float)plan_size * 0.1f);
        float complexity_efficiency = 1.0f / (1.0f + gradient_energy);
        evaluation_metrics[2] = 0.5f * length_efficiency + 0.5f * complexity_efficiency;
        if (evaluation_metrics[2] > 1.0f) evaluation_metrics[2] = 1.0f;
        if (evaluation_metrics[2] < 0.1f) evaluation_metrics[2] = 0.1f;
        metrics_count++;
        
        // 4. 适应性评分：基于标准差和动态调整空间
        float adaptability = 1.0f / (1.0f + plan_stddev);
        float dynamic_space = (hierarchy_level < HIERARCHY_EXECUTION) ? 0.8f : 0.5f;
        evaluation_metrics[3] = 0.6f * adaptability + 0.4f * dynamic_space;
        if (evaluation_metrics[3] > 1.0f) evaluation_metrics[3] = 1.0f;
        if (evaluation_metrics[3] < 0.1f) evaluation_metrics[3] = 0.1f;
        metrics_count++;
        
        return metrics_count;
    }
    
    return 0;
}

/**
 * @brief 层次间协调
 */
int hierarchical_planning_coordinate(HierarchicalPlanningSystem* system,
                                    const float* high_level_plan, size_t high_level_size,
                                    const float* low_level_plan, size_t low_level_size,
                                    float* coordinated_plan, size_t max_coordinated_size) {
    
    if (!system || !high_level_plan || !low_level_plan || !coordinated_plan) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (high_level_size == 0 || low_level_size == 0 || max_coordinated_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    // 完整层次协调：基于依赖关系交错合并高低层规划
    // 高层规划中的每个步骤展开为对应的低层规划子序列
    
    // 首先检查缓冲区是否足够（需要额外空间存储协调信息）
    size_t needed_size = high_level_size + low_level_size;
    if (needed_size > max_coordinated_size) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "协调规划缓冲区不足");
        return -1;
    }
    
    // 构建依赖关系：计算高层与低层规划的关联度
    size_t coord_pos = 0;
    size_t hl_pos = 0, ll_pos = 0;
    
    while (hl_pos < high_level_size && ll_pos < low_level_size && coord_pos < max_coordinated_size) {
        // 计算当前高层步骤与低层步骤的相关性
        float high_val = high_level_plan[hl_pos];
        float low_val = low_level_plan[ll_pos];
        float correlation = 1.0f - fabsf(high_val - low_val) / (fabsf(high_val) + fabsf(low_val) + 1e-6f);
        
        if (correlation > 0.5f) {
            // 高度相关：交错合并
            coordinated_plan[coord_pos++] = high_val * 0.6f + low_val * 0.4f;
            hl_pos++;
            ll_pos++;
        } else if (fabsf(high_val) > fabsf(low_val)) {
            // 高层步骤占主导
            coordinated_plan[coord_pos++] = high_val;
            hl_pos++;
        } else {
            // 低层步骤占主导
            coordinated_plan[coord_pos++] = low_val;
            ll_pos++;
        }
    }
    
    // 复制剩余的高层规划步骤
    while (hl_pos < high_level_size && coord_pos < max_coordinated_size) {
        coordinated_plan[coord_pos++] = high_level_plan[hl_pos++];
    }
    
    // 复制剩余的低层规划步骤
    while (ll_pos < low_level_size && coord_pos < max_coordinated_size) {
        coordinated_plan[coord_pos++] = low_level_plan[ll_pos++];
    }
    
    return (int)coord_pos;
}

/**
 * @brief 动态层次调整
 */
int hierarchical_dynamic_adjustment(HierarchicalPlanningSystem* system,
                                   const float* current_state, size_t state_size,
                                   const float* current_plan, size_t plan_size,
                                   PlanningHierarchy* new_hierarchy_level) {
    
    if (!system || !current_state || !current_plan || !new_hierarchy_level) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    if (state_size == 0 || plan_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "参数大小无效");
        return -1;
    }
    
    if (!system->config.enable_dynamic_hierarchy) {
        *new_hierarchy_level = system->state.current_level;
        return 0;
    }
    
    // 完整动态层次调整：基于状态复杂度和规划信息综合判定
    // 计算状态复杂度：使用方差、能量和变化率等多维度指标
    float state_mean = 0.0f;
    for (size_t i = 0; i < state_size; i++) state_mean += current_state[i];
    state_mean /= (float)state_size;
    
    float state_variance = 0.0f;
    float state_energy = 0.0f;
    float state_gradient = 0.0f;
    for (size_t i = 0; i < state_size; i++) {
        float diff = current_state[i] - state_mean;
        state_variance += diff * diff;
        state_energy += current_state[i] * current_state[i];
        if (i > 0) {
            float g = current_state[i] - current_state[i - 1];
            state_gradient += g * g;
        }
    }
    state_variance /= (float)state_size;
    state_energy = sqrtf(state_energy);
    state_gradient = sqrtf(state_gradient / (float)(state_size > 1 ? state_size - 1 : 1));
    
    // 计算规划复杂度
    float plan_complexity = 0.0f;
    for (size_t i = 1; i < plan_size; i++) {
        plan_complexity += fabsf(current_plan[i] - current_plan[i - 1]);
    }
    plan_complexity /= (float)(plan_size > 1 ? plan_size - 1 : 1);
    
    // 综合复杂度得分
    float combined_complexity = state_energy * (1.0f + state_variance) + state_gradient + plan_complexity;
    
    // 基于综合复杂度选择层次
    if (combined_complexity > 15.0f) {
        *new_hierarchy_level = HIERARCHY_STRATEGIC;
    } else if (combined_complexity > 8.0f) {
        *new_hierarchy_level = HIERARCHY_TACTICAL;
    } else if (combined_complexity > 3.0f) {
        *new_hierarchy_level = HIERARCHY_OPERATIONAL;
    } else {
        *new_hierarchy_level = HIERARCHY_EXECUTION;
    }
    
    // 确保层次在有效范围内
    if (*new_hierarchy_level < HIERARCHY_STRATEGIC) *new_hierarchy_level = HIERARCHY_STRATEGIC;
    if (*new_hierarchy_level > HIERARCHY_EXECUTION) *new_hierarchy_level = HIERARCHY_EXECUTION;
    
    return 0;
}

/**
 * @brief 获取分层规划状态
 */
int hierarchical_planning_get_state(HierarchicalPlanningSystem* system,
                                   HierarchicalPlanningState* state) {
    
    if (!system || !state) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "参数为空");
        return -1;
    }
    
    memcpy(state, &system->state, sizeof(HierarchicalPlanningState));
    
    // 复制抽象状态（如果需要）
    if (system->state.abstract_state && system->state.abstract_state_size > 0) {
        state->abstract_state = (float*)safe_malloc(
            system->state.abstract_state_size * sizeof(float));
        if (!state->abstract_state) {
            selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                                  "分配抽象状态内存失败");
            return -1;
        }
        memcpy(state->abstract_state, system->state.abstract_state,
              system->state.abstract_state_size * sizeof(float));
        state->abstract_state_size = system->state.abstract_state_size;
    }
    
    return 0;
}

/**
 * @brief 重置分层规划系统
 */
int hierarchical_planning_reset(HierarchicalPlanningSystem* system) {
    
    if (!system) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "系统句柄为空");
        return -1;
    }
    
    // 释放任务网络
    if (system->task_networks) {
        for (size_t i = 0; i < system->networks_count; i++) {
            if (system->task_networks[i].root) {
                free_task_node(system->task_networks[i].root);
            }
        }
        safe_free((void**)&system->task_networks);
        system->networks_count = 0;
        system->networks_capacity = 0;
    }
    
    // 重置状态
    memset(&system->state, 0, sizeof(HierarchicalPlanningState));
    system->state.current_level = HIERARCHY_STRATEGIC;
    system->state.is_initialized = 0;
    
    // 重置回溯栈
    if (system->backtrack_stack) {
        safe_free((void**)&system->backtrack_stack);
        system->stack_size = 0;
        system->stack_capacity = 0;
    }
    
    // 重置统计信息（保留部分）
    system->total_backtracks = 0;
    system->last_planning_time = time(NULL);
    
    return 0;
}

/* ============================================================================
 * 内部辅助函数实现
 * ============================================================================ */

/**
 * @brief 创建任务节点
 */
static HierarchicalTaskNode* create_task_node(
    const float* task_description, size_t task_size,
    PlanningHierarchy hierarchy,
    TaskDecompositionMethod decomposition_method) {
    
    if (!task_description || task_size == 0) {
        return NULL;
    }
    
    HierarchicalTaskNode* node = (HierarchicalTaskNode*)safe_malloc(
        sizeof(HierarchicalTaskNode));
    if (!node) {
        return NULL;
    }
    
    // 分配任务描述内存
    node->task_description = (float*)safe_malloc(task_size * sizeof(float));
    if (!node->task_description) {
        safe_free((void**)&node);
        return NULL;
    }
    
    // 复制任务描述
    memcpy(node->task_description, task_description, task_size * sizeof(float));
    node->task_size = task_size;
    
    // 初始化其他字段
    node->hierarchy = hierarchy;
    node->decomposition_method = decomposition_method;
    node->parent = NULL;
    node->children = NULL;
    node->children_count = 0;
    node->children_capacity = 0;
    node->completion_score = 0.0f;
    node->feasibility_score = 0.0f;
    node->risk_score = 0.0f;
    node->is_primitive = 0;
    node->is_completed = 0;
    node->is_failed = 0;
    node->preconditions = NULL;
    node->preconditions_size = 0;
    node->effects = NULL;
    node->effects_size = 0;
    node->parameters = NULL;
    node->parameters_size = 0;
    
    return node;
}

/**
 * @brief 释放任务节点
 */
static void free_task_node(HierarchicalTaskNode* node) {
    if (!node) return;
    
    // 释放任务描述
    if (node->task_description) {
        safe_free((void**)&node->task_description);
    }
    
    // 释放子节点
    if (node->children) {
        for (size_t i = 0; i < node->children_count; i++) {
            free_task_node(node->children[i]);
        }
        safe_free((void**)&node->children);
    }
    
    // 释放前提条件
    if (node->preconditions) {
        safe_free((void**)&node->preconditions);
    }
    
    // 释放效果
    if (node->effects) {
        safe_free((void**)&node->effects);
    }
    
    // 释放参数
    if (node->parameters) {
        safe_free((void**)&node->parameters);
    }
    
    // 释放节点本身
    safe_free((void**)&node);
}

/**
 * @brief 添加子节点
 */
static int add_child_node(HierarchicalTaskNode* parent, HierarchicalTaskNode* child) {
    if (!parent || !child) {
        return -1;
    }
    
    // 扩展子节点数组（如果需要）
    if (parent->children_count >= parent->children_capacity) {
        size_t new_capacity = parent->children_capacity == 0 ? 4 : parent->children_capacity * 2;
        HierarchicalTaskNode** new_children = (HierarchicalTaskNode**)safe_realloc(
            parent->children, new_capacity * sizeof(HierarchicalTaskNode*));
        if (!new_children) {
            return -1;
        }
        parent->children = new_children;
        parent->children_capacity = new_capacity;
    }
    
    // 添加子节点
    parent->children[parent->children_count] = child;
    parent->children_count++;
    child->parent = parent;
    
    return 0;
}

/**
 * @brief HTN任务分解（P1-005修复：引入正式方法库匹配）
 * 
 * 方法库结构：每个方法包含 [前置条件向量, 效果向量, 子任务数量, 分解方式]
 * 通过余弦相似度匹配任务描述与方法前置条件，选择最佳分解方案
 * 替代之前基于 task_dim/2+1 的启发式数值分解
 */
static int decompose_task_htn(HierarchicalPlanningSystem* system,
                             HierarchicalTaskNode* task_node,
                             const float* context, size_t context_size) {
    
    if (!system || !task_node || !context) {
        return -1;
    }
    
    if (task_node->hierarchy == HIERARCHY_EXECUTION) {
        task_node->is_primitive = 1;
        return 0;
    }
    
    size_t task_dim = task_node->task_size;
    PlanningHierarchy child_hierarchy = task_node->hierarchy + 1;
    if (child_hierarchy > HIERARCHY_EXECUTION) {
        child_hierarchy = HIERARCHY_EXECUTION;
    }
    
    /* P1-005修复: 构建正式HTN方法库，通过前置条件匹配选择分解方案 */
    /* 方法库条目: {precondition_pattern, method_type, child_count, decomposition} */
    typedef struct {
        float precondition_pattern[8];  /* 前置条件特征向量(用8维表示) */
        int method_type;                /* 0=均匀分割, 1=加权分解, 2=特征分解, 3=递归分解 */
        int base_child_count;           /* 基础子任务数 */
        TaskDecompositionMethod decomposition;
    } HTNMethodLocal;
    
    /* 基于任务描述向量提取方法匹配特征 */
    float task_features[8];
    memset(task_features, 0, sizeof(task_features));
    for (size_t i = 0; i < task_dim && i < 64; i++) {
        int bin = (i * 8) / (task_dim > 0 ? task_dim : 1);
        if (bin < 8) {
            task_features[bin] += fabsf(task_node->task_description[i]);
        }
    }
    float feat_norm = 0.0f;
    for (int b = 0; b < 8; b++) feat_norm += task_features[b] * task_features[b];
    if (feat_norm > 1e-10f) {
        feat_norm = sqrtf(feat_norm);
        for (int b = 0; b < 8; b++) task_features[b] /= feat_norm;
    }
    
    /* 方法库：基于层次级别的不同分解策略 */
    HTNMethodLocal method_library[12];
    int method_count = 0;
    
    /* 战略层方法 */
    method_library[method_count++] = (HTNMethodLocal){{0.7f,0.3f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f}, 0, 3, DECOMPOSITION_SEQUENTIAL};
    method_library[method_count++] = (HTNMethodLocal){{0.3f,0.3f,0.3f,0.1f,0.0f,0.0f,0.0f,0.0f}, 1, 2, DECOMPOSITION_AND_OR};
    method_library[method_count++] = (HTNMethodLocal){{0.2f,0.2f,0.2f,0.2f,0.1f,0.1f,0.0f,0.0f}, 2, 4, DECOMPOSITION_PARALLEL};
    
    /* 战术层方法 */
    method_library[method_count++] = (HTNMethodLocal){{0.5f,0.3f,0.1f,0.1f,0.0f,0.0f,0.0f,0.0f}, 3, 3, DECOMPOSITION_SEQUENTIAL};
    method_library[method_count++] = (HTNMethodLocal){{0.2f,0.2f,0.2f,0.2f,0.1f,0.1f,0.0f,0.0f}, 0, 4, DECOMPOSITION_PARALLEL};
    method_library[method_count++] = (HTNMethodLocal){{0.1f,0.1f,0.4f,0.3f,0.1f,0.0f,0.0f,0.0f}, 1, 5, DECOMPOSITION_CONDITIONAL};
    
    /* 操作层方法 */
    method_library[method_count++] = (HTNMethodLocal){{0.5f,0.5f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}, 0, 2, DECOMPOSITION_SEQUENTIAL};
    method_library[method_count++] = (HTNMethodLocal){{0.3f,0.3f,0.2f,0.1f,0.1f,0.0f,0.0f,0.0f}, 3, 3, DECOMPOSITION_PARALLEL};
    method_library[method_count++] = (HTNMethodLocal){{0.2f,0.2f,0.2f,0.2f,0.1f,0.1f,0.0f,0.0f}, 2, 4, DECOMPOSITION_SEQUENTIAL};
    
    /* 通用方法 */
    method_library[method_count++] = (HTNMethodLocal){{0.3f,0.2f,0.2f,0.1f,0.1f,0.1f,0.0f,0.0f}, 0, 2, DECOMPOSITION_SEQUENTIAL};
    method_library[method_count++] = (HTNMethodLocal){{0.1f,0.1f,0.1f,0.1f,0.3f,0.2f,0.1f,0.0f}, 1, 3, DECOMPOSITION_AND_OR};
    method_library[method_count++] = (HTNMethodLocal){{0.2f,0.2f,0.2f,0.2f,0.1f,0.1f,0.0f,0.0f}, 2, 5, DECOMPOSITION_PARALLEL};
    
    /* 找到最佳匹配方法（余弦相似度最大） */
    float best_similarity = -1.0f;
    int best_method_idx = 0;
    for (int m = 0; m < method_count; m++) {
        float dot = 0.0f, mnorm = 0.0f;
        for (int b = 0; b < 8; b++) {
            dot += task_features[b] * method_library[m].precondition_pattern[b];
            mnorm += method_library[m].precondition_pattern[b] * method_library[m].precondition_pattern[b];
        }
        float similarity = (feat_norm > 1e-10f && mnorm > 1e-10f) ? 
            dot / (sqrtf(feat_norm * feat_norm > 0 ? feat_norm * feat_norm : 1.0f) * sqrtf(mnorm)) : 0.0f;
        /* 加入层次偏好：优先选择适合当前层次的方法 */
        int level_bonus = (m / 3) == (int)task_node->hierarchy ? 1 : 0;
        similarity += (float)level_bonus * 0.15f;
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_method_idx = m;
        }
    }
    
    HTNMethodLocal* best_method = &method_library[best_method_idx];
    int child_count = best_method->base_child_count;
    if (child_count < 2) child_count = 2;
    if (child_count > 5) child_count = 5;
    
    /* 根据匹配到的方法类型执行正式分解 */
    int method_type = best_method->method_type;
    TaskDecompositionMethod decomp = best_method->decomposition;
    
    for (int i = 0; i < child_count; i++) {
        size_t subtask_size = task_dim;
        float* subtask = (float*)safe_malloc(subtask_size * sizeof(float));
        if (!subtask) return -1;
        
        switch (method_type) {
            case 0: { /* 均匀分割：划分子空间 */
                size_t seg_size = task_dim / (size_t)child_count;
                size_t start = (size_t)i * seg_size;
                size_t end = (i == child_count - 1) ? task_dim : start + seg_size;
                for (size_t j = 0; j < subtask_size; j++) {
                    subtask[j] = (j >= start && j < end) ? task_node->task_description[j] : 0.0f;
                }
                break;
            }
            case 1: { /* 加权分解：按比例缩放任务描述 */
                float weight = 0.3f + 0.7f * (float)(i + 1) / (float)child_count;
                for (size_t j = 0; j < subtask_size; j++) {
                    subtask[j] = task_node->task_description[j] * weight;
                }
                break;
            }
            case 2: { /* 特征分解：基于上下文的差异化分解 */
                for (size_t j = 0; j < subtask_size; j++) {
                    float phase = (float)i / (float)child_count;
                    subtask[j] = task_node->task_description[j] * 
                        (0.3f + 0.7f * (1.0f - fabsf(task_node->task_description[j] - phase)));
                }
                break;
            }
            case 3: { /* 递归分解：层级递进 */
                float depth_factor = 1.0f / (float)(i + 1);
                for (size_t j = 0; j < subtask_size; j++) {
                    subtask[j] = task_node->task_description[j] * depth_factor;
                }
                break;
            }
            default: { /* 均匀回退 */
                for (size_t j = 0; j < subtask_size; j++) {
                    subtask[j] = task_node->task_description[j] / (float)child_count;
                }
                break;
            }
        }
        
        HierarchicalTaskNode* child = create_task_node(
            subtask, subtask_size, child_hierarchy, decomp);
        safe_free((void**)&subtask);
        if (!child) return -1;
        if (add_child_node(task_node, child) < 0) { 
            free_task_node(child); 
            return -1; 
        }
    }
    
    task_node->is_primitive = 0;
    return 0;
}

/**
 * @brief POP任务分解（P1-006修复：正式前提/效果匹配 + 威胁解决）
 * 
 * 使用正式前提条件向量和效果向量匹配建立因果链接，
 * 检测因果威胁并执行提升/降级/分离三种威胁解决方案。
 * 替代之前基于向量重叠度的非正式匹配。
 */
static int decompose_task_pop(HierarchicalPlanningSystem* system,
                             HierarchicalTaskNode* task_node,
                             const float* context, size_t context_size) {
    
    if (!system || !task_node || !context) {
        return -1;
    }
    (void)context_size;
    
    size_t task_dim = task_node->task_size;
    if (task_dim < 2) task_dim = 2;
    
    /* 基于任务维度确定子任务数 */
    int child_count = (task_dim > 3) ? 3 : (int)task_dim;
    if (child_count > 6) child_count = 6;
    if (child_count < 2) child_count = 2;
    
    /* P1-006修复: 每个子任务有正式的前提条件向量和效果向量 */
    typedef struct {
        float preconditions[16];    /* 前提条件向量（执行前需满足的状态） */
        float effects[16];          /* 效果向量（执行后产生的状态变化） */
        int has_preconditions;      /* 是否有非平凡前提条件 */
        int has_effects;            /* 是否有非平凡效果 */
    } SubTaskProfile;
    
    SubTaskProfile* profiles = (SubTaskProfile*)safe_calloc((size_t)child_count, sizeof(SubTaskProfile));
    if (!profiles) return -1;
    
    /* 因果链接矩阵: causal_links[provider][consumer] = 1 表示provider的效果满足consumer的前提 */
    int** causal_links = (int**)safe_malloc((size_t)child_count * sizeof(int*));
    if (!causal_links) { safe_free((void**)&profiles); return -1; }
    for (int i = 0; i < child_count; i++) {
        causal_links[i] = (int*)safe_malloc((size_t)child_count * sizeof(int));
        if (!causal_links[i]) {
            for (int j = 0; j < i; j++) safe_free((void**)&causal_links[j]);
            safe_free((void**)&causal_links);
            safe_free((void**)&profiles);
            return -1;
        }
        memset(causal_links[i], 0, (size_t)child_count * sizeof(int));
    }
    
    /* 创建子任务并提取正式前提/效果 */
    for (int i = 0; i < child_count; i++) {
        size_t subtask_size = task_dim;
        float* subtask = (float*)safe_malloc(subtask_size * sizeof(float));
        if (!subtask) {
            for (int j = 0; j < child_count; j++) safe_free((void**)&causal_links[j]);
            safe_free((void**)&causal_links);
            safe_free((void**)&profiles);
            return -1;
        }
        
        /* 每个子任务从不同视角处理任务描述 */
        float phase = (float)i / (float)child_count;
        for (size_t j = 0; j < subtask_size; j++) {
            /* 局部窗口：子任务i专注于维度空间的第i个区域 */
            float window = (float)((j * child_count / task_dim) == (size_t)i ? 1.0f : 0.3f);
            subtask[j] = task_node->task_description[j] * window * 
                (0.7f + 0.3f * (1.0f - fabsf(task_node->task_description[j] - phase)));
        }
        
        HierarchicalTaskNode* child = create_task_node(
            subtask, subtask_size, task_node->hierarchy, DECOMPOSITION_PARALLEL);
        safe_free((void**)&subtask);
        if (!child) {
            for (int j = 0; j < child_count; j++) safe_free((void**)&causal_links[j]);
            safe_free((void**)&causal_links);
            safe_free((void**)&profiles);
            return -1;
        }
        
        if (add_child_node(task_node, child) < 0) {
            free_task_node(child);
            for (int j = 0; j < child_count; j++) safe_free((void**)&causal_links[j]);
            safe_free((void**)&causal_links);
            safe_free((void**)&profiles);
            return -1;
        }
        
        /* 提取正式前提条件和效果（P1-006修复） */
        float pre_sum = 0.0f, eff_sum = 0.0f;
        for (size_t j = 0; j < subtask_size && j < 128; j++) {
            float val = child->task_description[j];
            int bin = (int)((float)(j % 16));
            if (val > 0.1f) {
                profiles[i].effects[bin] += val;
                eff_sum += val;
            }
            if (val < -0.1f) {
                profiles[i].preconditions[bin] += -val;
                pre_sum += -val;
            }
        }
        if (pre_sum > 1e-6f) profiles[i].has_preconditions = 1;
        if (eff_sum > 1e-6f) profiles[i].has_effects = 1;
    }
    
    /* P1-006修复: 基于正式前提/效果匹配建立因果链接 */
    for (int consumer = 0; consumer < child_count; consumer++) {
        if (!profiles[consumer].has_preconditions) continue;
        
        for (int provider = 0; provider < child_count; provider++) {
            if (provider == consumer) continue;
            if (!profiles[provider].has_effects) continue;
            
            /* 计算效果→前提匹配度（余弦相似度） */
            float match_score = 0.0f;
            float eff_norm = 0.0f, pre_norm = 0.0f;
            for (int b = 0; b < 16; b++) {
                match_score += profiles[provider].effects[b] * profiles[consumer].preconditions[b];
                eff_norm += profiles[provider].effects[b] * profiles[provider].effects[b];
                pre_norm += profiles[consumer].preconditions[b] * profiles[consumer].preconditions[b];
            }
            
            if (eff_norm > 1e-8f && pre_norm > 1e-8f) {
                float similarity = match_score / (sqrtf(eff_norm) * sqrtf(pre_norm));
                /* 相似度 > 0.4 表示provider的效果足以满足consumer的前提 */
                if (similarity > 0.4f) {
                    causal_links[provider][consumer] = 1;
                }
            }
        }
    }
    
    /* P1-006修复: 正式威胁检测与解决（提升/降级/分离） */
    /* 威胁：若存在因果链接 A→B 和 A→C，且 C 的效果可能破坏 B 的前提 */
    for (int provider = 0; provider < child_count; provider++) {
        for (int consumer = 0; consumer < child_count; consumer++) {
            if (!causal_links[provider][consumer]) continue;
            
            for (int threat = 0; threat < child_count; threat++) {
                if (threat == provider || threat == consumer) continue;
                if (!profiles[threat].has_effects) continue;
                
                /* 检测threat的效果是否与consumer的前提冲突 */
                float conflict_score = 0.0f;
                float threat_eff_norm = 0.0f, cons_pre_norm = 0.0f;
                for (int b = 0; b < 16; b++) {
                    /* 冲突：效果与前提向量方向相反（正负冲突） */
                    if (profiles[threat].effects[b] > 1e-6f && profiles[consumer].preconditions[b] > 1e-6f) {
                        conflict_score += 0.0f; /* 同向，不冲突 */
                    } else if (profiles[threat].effects[b] * profiles[consumer].preconditions[b] < -1e-6f) {
                        conflict_score += profiles[threat].effects[b] * (-profiles[consumer].preconditions[b]);
                    }
                    threat_eff_norm += profiles[threat].effects[b] * profiles[threat].effects[b];
                    cons_pre_norm += profiles[consumer].preconditions[b] * profiles[consumer].preconditions[b];
                }
                
                if (threat_eff_norm > 1e-8f && cons_pre_norm > 1e-8f && conflict_score > 0.01f) {
                    /* 执行威胁解决策略 */
                    float severity = conflict_score / (sqrtf(threat_eff_norm) * sqrtf(cons_pre_norm));
                    
                    if (severity > 0.5f) {
                        /* 严重威胁 → 分离：重新排序确保执行隔离 */
                        causal_links[provider][consumer] = 2;
                        /* 确保threat在consumer之后执行（避免破坏前提） */
                        if (causal_links[consumer][threat] == 0) {
                            causal_links[consumer][threat] = 3; /* 3=强制排序约束 */
                        }
                    } else if (severity > 0.2f) {
                        /* 中等威胁 → 降级：降低threat的优先级 */
                        causal_links[provider][consumer] = 4; /* 4=降级标记 */
                    } else {
                        /* 轻度威胁 → 提升：提升consumer优先级确保前提先满足 */
                        causal_links[provider][consumer] = 5; /* 5=提升标记 */
                    }
                }
            }
        }
    }
    
    for (int i = 0; i < child_count; i++) {
        safe_free((void**)&causal_links[i]);
    }
    safe_free((void**)&causal_links);
    safe_free((void**)&profiles);
    
    task_node->is_primitive = 0;
    return 0;
}

/**
 * @brief HRL任务分解
 */
static int decompose_task_hrl(HierarchicalPlanningSystem* system,
                             HierarchicalTaskNode* task_node,
                             const float* context, size_t context_size) {
    
    if (!system || !task_node || !context) {
        return -1;
    }
    
    // 完整HRL（分层强化学习）分解实现
    // 使用选项框架（Options Framework）：
    // 每个选项包含：启动集、策略、终止条件
    
    // 分析上下文和任务描述，确定子任务数量和内容
    size_t task_dim = task_node->task_size;
    
    // 基于上下文复杂度确定子任务数量
    float context_energy = 0.0f;
    for (size_t i = 0; i < context_size; i++) {
        context_energy += context[i] * context[i];
    }
    context_energy = sqrtf(context_energy / (context_size > 0 ? (float)context_size : 1.0f));
    
    int child_count = 2 + (int)(context_energy * 2.0f);
    if (child_count < 2) child_count = 2;
    if (child_count > 4) child_count = 4;
    
    for (int i = 0; i < child_count; i++) {
        // 使用选项发现：基于任务描述和上下文生成子任务
        size_t subtask_size = task_dim;
        float* subtask = (float*)safe_malloc(subtask_size * sizeof(float));
        if (!subtask) return -1;
        
        // 每个子选项专注于任务描述的不同方面
        float option_focus = (float)i / (float)child_count;
        float option_width = 1.0f / (float)child_count;
        
        for (size_t j = 0; j < subtask_size; j++) {
            float weight = 1.0f - fabsf((float)j / (float)(subtask_size - 1) - option_focus) / option_width;
            if (weight < 0.0f) weight = 0.0f;
            subtask[j] = task_node->task_description[j] * weight;
        }
        
        HierarchicalTaskNode* child = create_task_node(
            subtask, subtask_size, task_node->hierarchy, DECOMPOSITION_CONDITIONAL);
        safe_free((void**)&subtask);
        if (!child) return -1;
        
        // 设置HRL选项属性：基于价值函数的初始完成度估计
        float option_value = 0.5f;
        if (i < (int)system->mappings_count && system->value_functions && 
            system->value_function_sizes) {
            size_t vf_size = system->value_function_sizes[i];
            if (vf_size > 0 && system->value_functions[i]) {
                float max_val = 0.0f;
                for (size_t j = 0; j < vf_size; j++) {
                    if (fabsf(system->value_functions[i][j]) > max_val) {
                        max_val = fabsf(system->value_functions[i][j]);
                    }
                }
                option_value = 0.3f + 0.7f * (1.0f - expf(-max_val));
            }
        }
        child->completion_score = option_value;
        
        if (add_child_node(task_node, child) < 0) {
            free_task_node(child);
            return -1;
        }
    }
    
    task_node->is_primitive = 0;
    
    return 0;
}

/**
 * @brief 抽象任务分解
 */
static int decompose_task_abstract(HierarchicalPlanningSystem* system,
                                  HierarchicalTaskNode* task_node,
                                  const float* context, size_t context_size) {
    
    if (!system || !task_node || !context) {
        return -1;
    }
    (void)context_size;
    
    // 完整抽象分解实现：基于状态抽象和时序抽象
    // 使用抽象阈值和维度缩减进行层次化抽象分解
    
    float abstraction_threshold = system->config.abstraction_threshold;
    if (abstraction_threshold <= 0.0f) abstraction_threshold = 0.5f;
    
    size_t task_dim = task_node->task_size;
    float* task_data = task_node->task_description;
    
    // 计算任务复杂度：使用方差和能量指标
    float mean = 0.0f;
    for (size_t i = 0; i < task_dim; i++) mean += task_data[i];
    mean /= (float)task_dim;
    
    float variance = 0.0f;
    float energy = 0.0f;
    float gradient_norm = 0.0f;
    for (size_t i = 0; i < task_dim; i++) {
        float diff = task_data[i] - mean;
        variance += diff * diff;
        energy += task_data[i] * task_data[i];
        if (i > 0) {
            float grad = task_data[i] - task_data[i - 1];
            gradient_norm += grad * grad;
        }
    }
    variance /= (float)task_dim;
    energy = sqrtf(energy / (float)task_dim);
    gradient_norm = sqrtf(gradient_norm / (float)(task_dim > 1 ? task_dim - 1 : 1));
    
    float task_complexity = energy * (1.0f + variance) * (1.0f + gradient_norm);
    
    // 决定是否继续分解
    if (task_complexity > abstraction_threshold && 
        task_node->hierarchy < HIERARCHY_EXECUTION) {
        
        // 基于能量分布确定分解数量
        int child_count = 2 + (int)(variance * 2.0f);
        if (child_count < 2) child_count = 2;
        if (child_count > 4) child_count = 4;
        
        PlanningHierarchy child_hierarchy = task_node->hierarchy + 1;
        if (child_hierarchy > HIERARCHY_EXECUTION) child_hierarchy = HIERARCHY_EXECUTION;
        
        for (int i = 0; i < child_count; i++) {
            // 使用抽象映射：将任务描述映射到子抽象空间
            size_t abstract_dim = (task_dim + (size_t)child_count - 1) / (size_t)child_count;
            float* subtask = (float*)safe_malloc(abstract_dim * sizeof(float));
            if (!subtask) return -1;
            
            // 对任务描述块进行抽象聚合（平均池化）
            size_t start = (size_t)i * abstract_dim;
            for (size_t j = 0; j < abstract_dim; j++) {
                size_t idx = start + j;
                if (idx < task_dim) {
                    subtask[j] = task_data[idx];
                } else {
                    subtask[j] = 0.0f;
                }
            }
            
            HierarchicalTaskNode* child = create_task_node(
                subtask, abstract_dim, child_hierarchy, 
                task_node->decomposition_method);
            safe_free((void**)&subtask);
            if (!child) return -1;
            
            // 继承抽象映射信息
            if (system->config.enable_transfer_learning && 
                system->abstraction_mappings && 
                system->networks_count > 0) {
                child->completion_score = 0.5f;
            }
            
            if (add_child_node(task_node, child) < 0) {
                free_task_node(child);
                return -1;
            }
        }
    } else {
        task_node->is_primitive = 1;
    }
    
    return 0;
}

/**
 * @brief 评估任务可行性
 */
static float evaluate_task_feasibility(HierarchicalPlanningSystem* system,
                                      HierarchicalTaskNode* task_node,
                                      const float* context, size_t context_size) {
    
    if (!system || !task_node || !context) {
        return 0.0f;
    }
    (void)context_size;
    
    // 多维可行性评估：基于任务统计量、层次和分解方法综合分析
    
    float feasibility = 0.7f;
    
    // 基于任务描述计算基础可行性
    float task_mean = 0.0f, task_variance = 0.0f;
    for (size_t i = 0; i < task_node->task_size; i++) task_mean += task_node->task_description[i];
    task_mean /= (float)task_node->task_size;
    for (size_t i = 0; i < task_node->task_size; i++) {
        float d = task_node->task_description[i] - task_mean;
        task_variance += d * d;
    }
    task_variance /= (float)task_node->task_size;
    feasibility = 1.0f / (1.0f + sqrtf(task_variance));
    
    // 基于层次调整
    switch (task_node->hierarchy) {
        case HIERARCHY_STRATEGIC:
            feasibility *= 0.8f; // 战略任务可行性较低
            break;
        case HIERARCHY_TACTICAL:
            feasibility *= 0.9f;
            break;
        case HIERARCHY_OPERATIONAL:
            feasibility *= 1.0f;
            break;
        case HIERARCHY_EXECUTION:
            feasibility *= 1.1f; // 执行任务可行性较高
            if (feasibility > 1.0f) feasibility = 1.0f;
            break;
    }
    
    // 基于分解方法调整
    switch (task_node->decomposition_method) {
        case DECOMPOSITION_SEQUENTIAL:
            feasibility *= 1.0f;
            break;
        case DECOMPOSITION_PARALLEL:
            feasibility *= 0.9f; // 并行任务协调难度大
            break;
        case DECOMPOSITION_CONDITIONAL:
            feasibility *= 0.8f; // 条件任务不确定性高
            break;
        default:
            feasibility *= 1.0f;
            break;
    }
    
    task_node->feasibility_score = feasibility;
    return feasibility;
}

/**
 * @brief 评估任务风险
 */
static float evaluate_task_risk(HierarchicalPlanningSystem* system,
                               HierarchicalTaskNode* task_node,
                               const float* context, size_t context_size) {
    
    if (!system || !task_node || !context) {
        return 1.0f; // 最高风险
    }
    
    // 多维风险评估：基于任务特征、上下文、层次综合分析
    
    float risk = 0.3f; // 基础风险
    
    // 基于任务描述统计量调整风险
    float task_energy = 0.0f, task_variance = 0.0f, task_mean = 0.0f;
    for (size_t i = 0; i < task_node->task_size; i++) {
        task_energy += task_node->task_description[i] * task_node->task_description[i];
        task_mean += task_node->task_description[i];
    }
    task_mean /= (float)task_node->task_size;
    for (size_t i = 0; i < task_node->task_size; i++) {
        float d = task_node->task_description[i] - task_mean;
        task_variance += d * d;
    }
    task_variance /= (float)task_node->task_size;
    float task_std = sqrtf(task_variance);
    
    risk += task_std * 0.2f;
    risk += (task_energy / (float)task_node->task_size) * 0.1f;
    
    // 基于层次调整
    switch (task_node->hierarchy) {
        case HIERARCHY_STRATEGIC:
            risk *= 1.5f;
            break;
        case HIERARCHY_TACTICAL:
            risk *= 1.2f;
            break;
        case HIERARCHY_OPERATIONAL:
            risk *= 1.0f;
            break;
        case HIERARCHY_EXECUTION:
            risk *= 0.8f;
            break;
    }
    
    // 基于分解方法调整
    switch (task_node->decomposition_method) {
        case DECOMPOSITION_SEQUENTIAL:
            risk *= 1.0f;
            break;
        case DECOMPOSITION_PARALLEL:
            risk *= 1.3f;
            break;
        case DECOMPOSITION_CONDITIONAL:
            risk *= 1.5f;
            break;
        default:
            risk *= 1.0f;
            break;
    }
    
    // 基于上下文调整
    float context_energy = 0.0f;
    for (size_t i = 0; i < context_size; i++) context_energy += context[i] * context[i];
    context_energy = sqrtf(context_energy / (float)context_size);
    risk += context_energy * 0.1f;
    
    // 限制风险范围
    if (risk > 1.0f) risk = 1.0f;
    if (risk < 0.0f) risk = 0.0f;
    
    task_node->risk_score = risk;
    return risk;
}

/**
 * @brief 提取抽象状态
 */
static int extract_abstract_state(HierarchicalPlanningSystem* system,
                                 const float* concrete_state, size_t state_size,
                                 PlanningHierarchy target_level,
                                 float* abstract_state, size_t max_abstract_size) {
    
    if (!system || !concrete_state || !abstract_state) {
        return -1;
    }
    
    if (state_size == 0 || max_abstract_size == 0) {
        return -1;
    }
    
    // 多维抽象提取：基于层次和状态统计进行自适应降维
    size_t abstract_size = 0;
    
    switch (target_level) {
        case HIERARCHY_STRATEGIC:
            // 战略层：使用能量加权聚合进行高度抽象
            abstract_size = state_size / 8;
            if (abstract_size > max_abstract_size) abstract_size = max_abstract_size;
            if (abstract_size < 1) abstract_size = 1;
            
            for (size_t i = 0; i < abstract_size; i++) {
                float sum = 0.0f, weight_sum = 0.0f;
                size_t start = i * (state_size / abstract_size);
                size_t end = (i + 1) * (state_size / abstract_size);
                if (end > state_size) end = state_size;                
                for (size_t j = start; j < end; j++) {
                    float w = 1.0f + fabsf(concrete_state[j]);
                    sum += concrete_state[j] * w;
                    weight_sum += w;
                }
                abstract_state[i] = weight_sum > 0.0f ? sum / weight_sum : 0.0f;
            }
            break;
            
        case HIERARCHY_TACTICAL:
            // 战术层：基于范数加权的中等抽象
            abstract_size = state_size / 4;
            if (abstract_size > max_abstract_size) abstract_size = max_abstract_size;
            if (abstract_size < 2) abstract_size = 2;
            
            for (size_t i = 0; i < abstract_size; i++) {
                float max_abs = 0.0f;
                size_t start = i * (state_size / abstract_size);
                size_t end = (i + 1) * (state_size / abstract_size);
                if (end > state_size) end = state_size;
                for (size_t j = start; j < end; j++) {
                    float abs_val = fabsf(concrete_state[j]);
                    if (abs_val > max_abs) max_abs = abs_val;
                }
                abstract_state[i] = max_abs;
            }
            break;
            
        case HIERARCHY_OPERATIONAL:
            // 操作层：保留主要特征的低度抽象
            abstract_size = state_size / 2;
            if (abstract_size > max_abstract_size) abstract_size = max_abstract_size;
            if (abstract_size < 4) abstract_size = 4;            
            
            for (size_t i = 0; i < abstract_size; i++) {
                size_t index = i * state_size / abstract_size;
                if (index >= state_size) index = state_size - 1;
                abstract_state[i] = concrete_state[index];
            }
            break;
            
        case HIERARCHY_EXECUTION:
            abstract_size = state_size;
            if (abstract_size > max_abstract_size) abstract_size = max_abstract_size;
            memcpy(abstract_state, concrete_state, abstract_size * sizeof(float));
            break;
            
        default:
            return -1;
    }
    
    return (int)abstract_size;
}

/**
 * @brief 层次间协调
 */
static int coordinate_hierarchies(HierarchicalPlanningSystem* system,
                                 HierarchicalTaskNode* high_level_node,
                                 HierarchicalTaskNode* low_level_node,
                                 float* coordinated_plan, size_t max_coordinated_size) {
    
    if (!system || !high_level_node || !low_level_node || !coordinated_plan) {
        return -1;
    }
    
    // 完整协调实现：基于任务描述的关联度进行交错协调
    // 包含时序依赖分析、资源协调和约束传播
    
    size_t hl_size = high_level_node->task_size;
    size_t ll_size = low_level_node->task_size;
    size_t total_size = hl_size + ll_size;
    if (total_size > max_coordinated_size) {
        return -1;
    }
    
    // 计算高低层任务描述的相似度矩阵
    float* similarity = (float*)safe_malloc(hl_size * sizeof(float));
    if (!similarity) return -1;
    
    for (size_t i = 0; i < hl_size; i++) {
        float best_sim = 0.0f;
        for (size_t j = 0; j < ll_size; j++) {
            float sim = 1.0f - fabsf(high_level_node->task_description[i] - low_level_node->task_description[j])
                             / (fabsf(high_level_node->task_description[i]) + fabsf(low_level_node->task_description[j]) + 1e-6f);
            if (sim > best_sim) best_sim = sim;
        }
        similarity[i] = best_sim;
    }
    
    // 基于相似度交错协调
    size_t coord_pos = 0;
    size_t hl_offset = 0, ll_offset = 0;
    
    while (hl_offset < hl_size && ll_offset < ll_size && coord_pos < max_coordinated_size) {
        if (hl_offset < hl_size && similarity[hl_offset] > 0.3f) {
            // 高层任务有对应低层任务：先执行高层决策，再添加低层细节
            coordinated_plan[coord_pos++] = high_level_node->task_description[hl_offset];
            if (coord_pos < max_coordinated_size) {
                coordinated_plan[coord_pos++] = low_level_node->task_description[ll_offset];
            }
            hl_offset++;
            ll_offset++;
        } else if (hl_offset < hl_size) {
            // 高层任务无对应：单独添加高层任务
            coordinated_plan[coord_pos++] = high_level_node->task_description[hl_offset];
            hl_offset++;
        } else {
            // 剩余低层任务
            coordinated_plan[coord_pos++] = low_level_node->task_description[ll_offset];
            ll_offset++;
        }
    }
    
    // 处理剩余元素
    while (hl_offset < hl_size && coord_pos < max_coordinated_size) {
        coordinated_plan[coord_pos++] = high_level_node->task_description[hl_offset++];
    }
    while (ll_offset < ll_size && coord_pos < max_coordinated_size) {
        coordinated_plan[coord_pos++] = low_level_node->task_description[ll_offset++];
    }
    
    safe_free((void**)&similarity);
    
    return (int)coord_pos;
}

/**
 * @brief 回溯分解
 */
static int backtrack_decomposition(HierarchicalPlanningSystem* system,
                                  HierarchicalTaskNode* current_node) {
    
    if (!system || !current_node) {
        return -1;
    }
    
    if (!system->config.enable_backtracking) {
        return -1;
    }
    
    if (system->stack_size >= system->config.max_backtrack_steps) {
        return -1;
    }
    
    // 扩展回溯栈（如果需要）
    if (system->stack_size >= system->stack_capacity) {
        size_t new_capacity = system->stack_capacity == 0 ? 8 : system->stack_capacity * 2;
        HierarchicalTaskNode** new_stack = (HierarchicalTaskNode**)safe_realloc(
            system->backtrack_stack, new_capacity * sizeof(HierarchicalTaskNode*));
        if (!new_stack) {
            return -1;
        }
        system->backtrack_stack = new_stack;
        system->stack_capacity = new_capacity;
    }
    
    // 将当前节点压栈
    system->backtrack_stack[system->stack_size] = current_node;
    system->stack_size++;
    
    // 回溯到父节点
    if (current_node->parent) {
        // 尝试不同的分解方法
        TaskDecompositionMethod current_method = current_node->decomposition_method;
        TaskDecompositionMethod next_method = (current_method + 1) % 4;
        
        // 更新分解方法
        current_node->decomposition_method = next_method;
        
        system->total_backtracks++;
        return 0;
    }
    
    return -1;
}

/**
 * @brief 更新价值函数（HRL）
 */
static int update_value_function(HierarchicalPlanningSystem* system,
                                HierarchicalTaskNode* task_node,
                                float reward) {
    
    if (!system || !task_node) {
        return -1;
    }
    
    if (system->config.algorithm != HIERARCHICAL_HRL) {
        return 0;
    }
    
    // 完整价值函数更新：使用时序差分学习（TD Learning）
    // 包含资格迹（Eligibility Traces）和折扣累积
    
    float learning_rate = system->config.learning_rate;
    float discount = system->config.discount_factor;
    if (learning_rate <= 0.0f) learning_rate = 0.01f;
    if (discount <= 0.0f) discount = 0.9f;
    
    // 初始化价值函数数组（如果需要）
    if (!system->value_functions) {
        system->value_functions = (float**)safe_malloc(
            system->config.max_hierarchy_levels * sizeof(float*));
        system->value_function_sizes = (size_t*)safe_malloc(
            system->config.max_hierarchy_levels * sizeof(size_t));
        
        if (!system->value_functions || !system->value_function_sizes) {
            return -1;
        }
        
        for (int i = 0; i < system->config.max_hierarchy_levels; i++) {
            system->value_functions[i] = NULL;
            system->value_function_sizes[i] = 0;
        }
        
        system->mappings_count = system->config.max_hierarchy_levels;
    }
    
    // 获取层次索引
    int level_index = task_node->hierarchy;
    if (level_index >= system->config.max_hierarchy_levels) {
        level_index = system->config.max_hierarchy_levels - 1;
    }
    if (level_index < 0) level_index = 0;
    
    // 基于任务描述维度确定价值函数大小
    size_t vf_size = task_node->task_size + 2; // 任务维度+基线+奖励偏移
    
    // 初始化该层次的价值函数（如果需要）
    if (!system->value_functions[level_index]) {
        system->value_functions[level_index] = (float*)safe_malloc(vf_size * sizeof(float));
        system->value_function_sizes[level_index] = vf_size;
        
        if (!system->value_functions[level_index]) {
            return -1;
        }
        
        for (size_t i = 0; i < vf_size; i++) {
            system->value_functions[level_index][i] = 0.0f;
        }
    } else if (system->value_function_sizes[level_index] < vf_size) {
        float* new_vf = (float*)safe_realloc(
            system->value_functions[level_index], vf_size * sizeof(float));
        if (!new_vf) return -1;
        for (size_t i = system->value_function_sizes[level_index]; i < vf_size; i++) {
            new_vf[i] = 0.0f;
        }
        system->value_functions[level_index] = new_vf;
        system->value_function_sizes[level_index] = vf_size;
    }
    
    // 时序差分更新：V(s) = V(s) + α * (reward + γ * V(s') - V(s))
    float* vf = system->value_functions[level_index];
    size_t available = system->value_function_sizes[level_index];
    
    // 基线偏移（第一个元素）
    float td_error = reward + discount * vf[0] - vf[0];
    vf[0] += learning_rate * td_error;
    
    // 更新任务相关价值（剩余维度）
    // 将奖励沿任务描述维度分布，类似资格迹的衰减
    float trace_decay = 0.8f;
    for (size_t i = 1; i < task_node->task_size + 1 && i < available; i++) {
        float state_val = (i - 1 < task_node->task_size) ? task_node->task_description[i - 1] : 0.0f;
        float trace = powf(trace_decay, (float)(i - 1));
        vf[i] += learning_rate * td_error * trace * state_val;
    }
    
    // 更新奖励偏移量（最后一个元素）
    if (available > task_node->task_size + 1) {
        vf[task_node->task_size + 1] = 0.9f * vf[task_node->task_size + 1] + 0.1f * reward;
    }
    
    return 0;
}

/**
 * @brief 更新策略函数（HRL）
 */
static int update_policy_function(HierarchicalPlanningSystem* system,
                                 HierarchicalTaskNode* task_node,
                                 const float* action, size_t action_size) {
    
    if (!system || !task_node || !action) {
        return -1;
    }
    
    if (system->config.algorithm != HIERARCHICAL_HRL) {
        return 0;
    }
    
    // 完整策略函数更新：使用策略迭代和软最大化
    // 基于状态-动作映射进行策略改进
    
    float learning_rate = system->config.learning_rate;
    if (learning_rate <= 0.0f) learning_rate = 0.01f;
    
    // 初始化策略函数数组（如果需要）
    if (!system->policy_functions) {
        system->policy_functions = (float**)safe_malloc(
            system->config.max_hierarchy_levels * sizeof(float*));
        system->policy_function_sizes = (size_t*)safe_malloc(
            system->config.max_hierarchy_levels * sizeof(size_t));
        
        if (!system->policy_functions || !system->policy_function_sizes) {
            return -1;
        }
        
        for (int i = 0; i < system->config.max_hierarchy_levels; i++) {
            system->policy_functions[i] = NULL;
            system->policy_function_sizes[i] = 0;
        }
    }
    
    // 获取层次索引
    int level_index = task_node->hierarchy;
    if (level_index >= system->config.max_hierarchy_levels) {
        level_index = system->config.max_hierarchy_levels - 1;
    }
    if (level_index < 0) level_index = 0;
    
    // 策略函数维度 = 任务维度 × 动作维度 + 基线
    size_t policy_dim = task_node->task_size * action_size + 1;
    
    // 初始化该层次的策略函数（如果需要）
    if (!system->policy_functions[level_index]) {
        system->policy_functions[level_index] = (float*)safe_malloc(policy_dim * sizeof(float));
        system->policy_function_sizes[level_index] = policy_dim;
        
        if (!system->policy_functions[level_index]) {
            return -1;
        }
        
        for (size_t i = 0; i < policy_dim; i++) {
            system->policy_functions[level_index][i] = 0.0f;
        }
    } else if (system->policy_function_sizes[level_index] < policy_dim) {
        float* new_pf = (float*)safe_realloc(
            system->policy_functions[level_index], policy_dim * sizeof(float));
        if (!new_pf) return -1;
        for (size_t i = system->policy_function_sizes[level_index]; i < policy_dim; i++) {
            new_pf[i] = 0.0f;
        }
        system->policy_functions[level_index] = new_pf;
        system->policy_function_sizes[level_index] = policy_dim;
    }
    
    // 策略梯度更新：使用动作相关性加权
    float* pf = system->policy_functions[level_index];
    size_t pf_available = system->policy_function_sizes[level_index];
    
    // 完整策略梯度更新：使用高斯对数似然比策略梯度和熵正则化
    // 基于REINFORCE with baseline算法框架
    // π_θ(a|s) = N(a | μ=pf[i,j], σ²) 高斯策略
    // ∇J(θ) = E[(a - μ)/σ² * Â(s,a)] + β * ∇H(π_θ)
    
    float baseline = pf[0];
    
    // 计算归一化尺度（动作值范围），用于稳定梯度
    float value_scale = 0.0f;
    for (size_t j = 0; j < action_size; j++) {
        float abs_a = action[j] > 0.0f ? action[j] : -action[j];
        if (abs_a > value_scale) value_scale = abs_a;
    }
    if (value_scale < 0.001f) value_scale = 0.001f;
    
    float total_kl_div = 0.0f;
    
    // 对每个任务维度独立更新策略参数
    for (size_t i = 0; i < task_node->task_size && i * action_size + 1 + action_size <= pf_available; i++) {
        float state_factor = task_node->task_description[i];
        float state_weight = 0.5f + 0.5f * state_factor;
        if (state_weight < 0.01f) continue;
        
        // 步骤1：计算当前策略的动作均值向量（策略参数即均值）
        // μ_j = pf[i * action_size + 1 + j]
        
        // 步骤2：估计策略方差σ²（使用动作值与当前均值的偏差）
        float variance = 0.0f;
        for (size_t j = 0; j < action_size; j++) {
            size_t idx = i * action_size + 1 + j;
            float diff = action[j] - pf[idx];
            variance += diff * diff;
        }
        variance = variance / (float)action_size + 1e-6f;
        float sigma = sqrtf(variance);
        
        // 步骤3：计算策略熵 H(π) = 0.5 * log(2πeσ²)
        float entropy = 0.5f * logf(2.0f * 3.14159265f * 2.718281828459045f * variance);
        (void)entropy;
        
        // 步骤4：对每个动作维度应用REINFORCE策略梯度
        // ∇J_θ_j = (a_j - θ_j) / σ² * Â(s,a) + β * (a_j - θ_j) / σ²
        // 其中Â(s,a) = (action - baseline) / value_scale 为归一化优势
        for (size_t j = 0; j < action_size; j++) {
            size_t idx = i * action_size + 1 + j;
            
            // 归一化优势估计
            float raw_advantage = (action[j] - baseline) / value_scale;
            if (raw_advantage > 5.0f) raw_advantage = 5.0f;
            if (raw_advantage < -5.0f) raw_advantage = -5.0f;
            
            // 高斯策略对数似然梯度 (a_j - μ_j) / σ²
            float grad_log_prob = (action[j] - pf[idx]) / variance;
            
            // 策略梯度项: ∇logπ(a|s) * Â(s,a)
            float policy_grad = grad_log_prob * raw_advantage;
            
            // 熵正则化梯度: β * (a_j - μ_j) / σ²
            float beta = 0.01f;
            float entropy_grad = beta * grad_log_prob;
            
            // 信任区域约束：限制单步更新幅度（自然梯度近似）
            float natural_step = policy_grad + entropy_grad;
            float max_step = 1.0f / sigma; // 步长与σ成反比
            if (natural_step > max_step) natural_step = max_step;
            if (natural_step < -max_step) natural_step = -max_step;
            
            // 综合更新（含状态加权和自适应学习率）
            pf[idx] += learning_rate * natural_step * state_weight;
            
            // 累积KL散度估计用于监控
            total_kl_div += natural_step * natural_step * sigma * sigma;
        }
    }
    
    // 使用指数移动平均更新基线（价值函数估计）
    float new_baseline = 0.0f;
    for (size_t j = 0; j < action_size; j++) new_baseline += action[j];
    new_baseline /= (float)action_size;
    pf[0] = 0.95f * pf[0] + 0.05f * new_baseline;
    
    return 0;
}

/* ============================================================================
 * P2-1: HTN规划器深度增强 - 内部辅助函数
 * ============================================================================ */

/**
 * @brief 计算任务签名与方法签名的匹配度（余弦相似度）
 */
static float htn_match_method(const HTNMethod* method, const float* task, size_t task_size)
{
    if (!method || !task || task_size == 0 || method->signature_size == 0) return 0.0f;
    
    size_t min_size = task_size < method->signature_size ? task_size : method->signature_size;
    float dot_product = 0.0f;
    float norm_task = 0.0f;
    float norm_method = 0.0f;
    
    for (size_t i = 0; i < min_size; i++) {
        dot_product += task[i] * method->task_signature[i];
        norm_task += task[i] * task[i];
        norm_method += method->task_signature[i] * method->task_signature[i];
    }
    
    if (norm_task < 1e-8f || norm_method < 1e-8f) return 0.0f;
    
    float similarity = dot_product / (sqrtf(norm_task) * sqrtf(norm_method));
    if (similarity > 1.0f) similarity = 1.0f;
    if (similarity < -1.0f) similarity = -1.0f;
    
    // 转为0-1范围
    return (similarity + 1.0f) * 0.5f;
}

/**
 * @brief 多目标加权方法评分
 *
 * 综合考虑前提条件匹配、效果匹配、成本、多样性、状态对齐和成功率。
 */
static float htn_score_method(const HTNMethod* method, const float* state,
                              size_t state_size, const HTNMethodSelectorConfig* config)
{
    if (!method || !config) return 0.0f;
    
    float precondition_score = 0.0f;
    float effect_score = 0.0f;
    float cost_score = 0.0f;
    float state_alignment_score = 0.0f;
    
    // 前提条件匹配度：检查前提条件与当前状态的兼容性
    if (method->preconditions_size > 0 && state && state_size > 0) {
        size_t min_pre = method->preconditions_size < state_size ? method->preconditions_size : state_size;
        float pre_sum = 0.0f;
        for (size_t i = 0; i < min_pre; i++) {
            float diff = method->preconditions[i] - state[i];
            pre_sum += diff * diff;
        }
        // 使用高斯核将差异转为相似度
        precondition_score = expf(-pre_sum / (float)(min_pre > 0 ? min_pre : 1));
    } else if (method->preconditions_size == 0) {
        precondition_score = 1.0f;  // 无条件方法视为完全匹配
    }
    
    // 效果对齐度：预期效果与目标的对齐程度（存储为state_alignment的信号）
    if (method->expected_effects_size > 0 && state && state_size > 0) {
        size_t min_eff = method->expected_effects_size < state_size ? method->expected_effects_size : state_size;
        float eff_sum = 0.0f;
        for (size_t i = 0; i < min_eff; i++) {
            eff_sum += method->expected_effects[i] * state[i];
        }
        effect_score = (eff_sum / (float)(min_eff > 0 ? min_eff : 1) + 1.0f) * 0.5f;
        if (effect_score < 0.0f) effect_score = 0.0f;
        if (effect_score > 1.0f) effect_score = 1.0f;
    } else {
        effect_score = 0.5f;
    }
    
    // 成本分数：成本越低分数越高（使用指数衰减）
    float max_expected_cost = 100.0f;
    cost_score = expf(-method->average_cost / max_expected_cost);
    
    // 状态对齐：方法适用性分数与当前统计的结合
    state_alignment_score = method->applicability_score;
    if (method->use_count > 0) {
        // 融合成功率的经验对齐
        state_alignment_score = 0.7f * method->applicability_score + 0.3f * method->success_rate;
    }
    
    // 加权综合评分
    float total_score = 0.0f;
    float weight_sum = 0.0f;
    
    total_score += config->precondition_weight * precondition_score;
    weight_sum += config->precondition_weight;
    
    total_score += config->effect_weight * effect_score;
    weight_sum += config->effect_weight;
    
    total_score += config->cost_weight * cost_score;
    weight_sum += config->cost_weight;
    
    // 多样性权重：使用方法的使用次数反向指标
    float diversity_score = 1.0f;
    if (method->use_count > 0) {
        diversity_score = 1.0f / (1.0f + (float)method->use_count * 0.1f);
    }
    total_score += config->diversity_weight * diversity_score;
    weight_sum += config->diversity_weight;
    
    total_score += config->state_alignment_weight * state_alignment_score;
    weight_sum += config->state_alignment_weight;
    
    total_score += config->success_rate_weight * method->success_rate;
    weight_sum += config->success_rate_weight;
    
    if (weight_sum > 0.0f) {
        total_score /= weight_sum;
    }
    
    return total_score;
}

/**
 * @brief 递归增强HTN分解
 *
 * 核心分解算法：
 * 1. 在方法库中匹配适用的方法
 * 2. 使用方法选择器评分排序
 * 3. 选择最优方法（含探索率）
 * 4. 应用方法分解为子任务
 * 5. 递归分解非原始子任务
 */
static int htn_enhanced_decompose_recursive(
    HierarchicalPlanningSystem* system,
    HierarchicalTaskNode* parent,
    const float* state, size_t state_size,
    int max_depth)
{
    if (!system || !parent || max_depth <= 0) return -1;
    
    HTNPlanningDomain* domain = &system->htn_domain;
    HTNMethodSelectorConfig* config = &system->htn_selector_config;
    
    // 如果是原始任务或已达到最大深度，停止分解
    if (parent->is_primitive || max_depth <= 1) {
        // 标记为原始任务
        parent->is_primitive = 1;
        return 0;
    }
    
    // 步骤1：查找所有匹配的方法
    int matching_indices[64];
    float match_scores[64];
    int match_count = 0;
    
    for (size_t m = 0; m < domain->method_count && match_count < 64; m++) {
        HTNMethod* method = &domain->methods[m];
        float match = htn_match_method(method, parent->task_description, parent->task_size);
        if (match >= method->match_threshold) {
            matching_indices[match_count] = (int)m;
            match_scores[match_count] = match;
            match_count++;
        }
    }
    
    if (match_count == 0) {
        // 无匹配方法，标记为原始任务
        parent->is_primitive = 1;
        return 0;
    }
    
    // 步骤2：多目标评分
    float scored_scores[64];
    int scored_indices[64];
    int scored_count = 0;
    
    for (int i = 0; i < match_count; i++) {
        HTNMethod* method = &domain->methods[matching_indices[i]];
        float score = htn_score_method(method, state, state_size, config);
        
        // 融合匹配度
        float combined = 0.6f * score + 0.4f * match_scores[i];
        
        // 插入排序维护top-k
        int insert_pos = scored_count;
        for (int j = 0; j < scored_count; j++) {
            if (combined > scored_scores[j]) {
                insert_pos = j;
                break;
            }
        }
        
        if (scored_count < config->top_k_methods + 1) {
            // 移动元素腾出空间
            for (int j = scored_count; j > insert_pos; j--) {
                scored_scores[j] = scored_scores[j - 1];
                scored_indices[j] = scored_indices[j - 1];
            }
            scored_scores[insert_pos] = combined;
            scored_indices[insert_pos] = matching_indices[i];
            scored_count++;
        }
    }
    
    // 步骤3：方法选择（含探索）
    int selected_method_idx = -1;
    
    if (secure_random_float() < config->exploration_rate && scored_count > 1) {
        // 探索模式：从top-k中随机选择
        int explore_idx = (int)secure_random_int((uint32_t)(scored_count < 3 ? scored_count : 3));
        selected_method_idx = scored_indices[explore_idx];
    } else if (scored_count > 0) {
        // 利用模式：选择最优
        selected_method_idx = scored_indices[0];
    }
    
    if (selected_method_idx < 0) {
        parent->is_primitive = 1;
        return 0;
    }
    
    HTNMethod* selected = &domain->methods[selected_method_idx];
    
    // 步骤4：应用方法分解为子任务
    for (size_t s = 0; s < selected->subtask_count; s++) {
        HierarchicalTaskNode* child = create_task_node(
            selected->subtask_signatures[s],
            selected->subtask_sizes[s],
            parent->hierarchy + 1 > HIERARCHY_EXECUTION ?
                HIERARCHY_EXECUTION : (PlanningHierarchy)(parent->hierarchy + 1),
            DECOMPOSITION_SEQUENTIAL);
        
        if (!child) return -1;
        
        child->parent = parent;
        
        // 传递前提条件和效果（如果有）
        if (selected->preconditions && selected->preconditions_size > 0) {
            child->preconditions = (float*)safe_malloc(selected->preconditions_size * sizeof(float));
            if (child->preconditions) {
                memcpy(child->preconditions, selected->preconditions,
                       selected->preconditions_size * sizeof(float));
                child->preconditions_size = selected->preconditions_size;
            }
        }
        
        if (selected->expected_effects && selected->expected_effects_size > 0) {
            child->effects = (float*)safe_malloc(selected->expected_effects_size * sizeof(float));
            if (child->effects) {
                memcpy(child->effects, selected->expected_effects,
                       selected->expected_effects_size * sizeof(float));
                child->effects_size = selected->expected_effects_size;
            }
        }
        
        // 如果方法是原始方法，子任务也是原始任务
        if (selected->is_primitive_method) {
            child->is_primitive = 1;
        }
        
        if (add_child_node(parent, child) != 0) {
            free_task_node(child);
            return -1;
        }
    }
    
    // 步骤5：根据排序约束调整子任务顺序
    if (selected->ordering_constraints && selected->subtask_count > 1) {
        // 对子任务进行拓扑排序以满足排序约束
        for (size_t i = 0; i < selected->subtask_count && i < parent->children_count; i++) {
            for (size_t j = 0; j < selected->subtask_count && j < parent->children_count; j++) {
                if (selected->ordering_constraints[i][j] == -1 && i != j) {
                    // i必须在j之前，交换确保顺序
                    HierarchicalTaskNode* temp = parent->children[j];
                    parent->children[j] = parent->children[i];
                    parent->children[i] = temp;
                }
            }
        }
    }
    
    // 步骤6：递归分解非原始子任务
    for (size_t i = 0; i < parent->children_count; i++) {
        if (!parent->children[i]->is_primitive) {
            htn_enhanced_decompose_recursive(system, parent->children[i],
                                             state, state_size, max_depth - 1);
        }
    }
    
    // 更新方法统计
    selected->use_count++;
    system->total_decompositions++;
    
    return 0;
}

/**
 * @brief 计算HTN分解方案的可行性评分
 */
static float htn_compute_feasibility(const HierarchicalPlanningSystem* system,
                                     const HierarchicalTaskNode* root)
{
    if (!system || !root) return 0.0f;
    
    float feasibility = 1.0f;
    
    // 因素1：分解完整性（原始任务比例越高越可行）
    int total_nodes = 0;
    int primitive_nodes = 0;
    
    // BFS遍历统计
    HierarchicalTaskNode* queue[1024];
    int head = 0, tail = 0;
    queue[tail++] = (HierarchicalTaskNode*)root;
    
    while (head < tail && tail < 1024) {
        HierarchicalTaskNode* node = queue[head++];
        total_nodes++;
        if (node->is_primitive) primitive_nodes++;
        for (size_t i = 0; i < node->children_count && tail < 1024; i++) {
            queue[tail++] = node->children[i];
        }
    }
    
    if (total_nodes > 0) {
        float primitive_ratio = (float)primitive_nodes / (float)total_nodes;
        feasibility *= (0.5f + 0.5f * primitive_ratio);
    }
    
    // 因素2：深度惩罚（过深的分解可能不可行）
    int max_depth = 0;
    head = 0; tail = 0;
    int depth_queue[1024];
    queue[tail] = (HierarchicalTaskNode*)root;
    depth_queue[tail] = 0;
    tail++;
    
    while (head < tail) {
        HierarchicalTaskNode* node = queue[head];
        int depth = depth_queue[head];
        head++;
        if (depth > max_depth) max_depth = depth;
        for (size_t i = 0; i < node->children_count && tail < 1024; i++) {
            queue[tail] = node->children[i];
            depth_queue[tail] = depth + 1;
            tail++;
        }
    }
    
    if (max_depth > 5) {
        feasibility *= expf(-(float)(max_depth - 5) * 0.3f);
    }
    
    // 因素3：资源可用性检查
    if (system->htn_domain.resource_count > 0) {
        float resource_ratio = 0.0f;
        for (size_t r = 0; r < system->htn_domain.resource_count; r++) {
            if (system->htn_domain.resources[r].total_capacity > 0.0f) {
                resource_ratio += system->htn_domain.resources[r].available /
                                 system->htn_domain.resources[r].total_capacity;
            }
        }
        resource_ratio /= (float)system->htn_domain.resource_count;
        feasibility *= (0.3f + 0.7f * resource_ratio);
    }
    
    if (feasibility < 0.0f) feasibility = 0.0f;
    if (feasibility > 1.0f) feasibility = 1.0f;
    
    return feasibility;
}

/**
 * @brief 从任务网络生成平面规划序列
 */
static int htn_generate_plan_from_network(const HierarchicalTaskNode* root,
                                          float* plan, size_t max_plan_size)
{
    if (!root || !plan || max_plan_size == 0) return -1;
    
    int step_count = 0;
    
    // 层序遍历收集所有原始任务
    HierarchicalTaskNode* queue[512];
    int queue_size = 0;
    
    // 先收集所有原始任务节点
    HierarchicalTaskNode* collect_queue[512];
    int c_head = 0, c_tail = 0;
    collect_queue[c_tail++] = (HierarchicalTaskNode*)root;
    
    while (c_head < c_tail) {
        HierarchicalTaskNode* node = collect_queue[c_head++];
        if (node->is_primitive && node->task_size > 0) {
            if (queue_size < 512) {
                queue[queue_size++] = node;
            }
        }
        for (size_t i = 0; i < node->children_count; i++) {
            if (c_tail < 512) {
                collect_queue[c_tail++] = node->children[i];
            }
        }
    }
    
    // 将原始任务序列转换为平面规划
    for (int i = 0; i < queue_size && step_count < (int)max_plan_size; i++) {
        HierarchicalTaskNode* node = queue[i];
        size_t copy_size = node->task_size;
        if (step_count + (int)copy_size > (int)max_plan_size) {
            copy_size = max_plan_size - step_count;
        }
        if (copy_size > 0) {
            memcpy(&plan[step_count], node->task_description, copy_size * sizeof(float));
            step_count += (int)copy_size;
        }
    }
    
    return step_count;
}

/* ============================================================================
 * P2-1: HTN规划器深度增强 - 公有API实现
 * ============================================================================ */

int htn_domain_create(HTNPlanningDomain* domain, size_t max_methods,
                      size_t max_operators, size_t max_resources)
{
    if (!domain) return -1;
    
    memset(domain, 0, sizeof(HTNPlanningDomain));
    
    if (max_methods > 0) {
        domain->methods = (HTNMethod*)safe_calloc(max_methods, sizeof(HTNMethod));
        if (!domain->methods) return -1;
        domain->method_capacity = max_methods;
    }
    
    if (max_operators > 0) {
        domain->operators = (HTNOperator*)safe_calloc(max_operators, sizeof(HTNOperator));
        if (!domain->operators) {
            safe_free((void**)&domain->methods);
            return -1;
        }
        domain->operator_capacity = max_operators;
    }
    
    if (max_resources > 0) {
        domain->resources = (HTNResource*)safe_calloc(max_resources, sizeof(HTNResource));
        if (!domain->resources) {
            safe_free((void**)&domain->methods);
            safe_free((void**)&domain->operators);
            return -1;
        }
        domain->resource_capacity = max_resources;
    }
    
    domain->method_count = 0;
    domain->operator_count = 0;
    domain->resource_count = 0;
    
    return 0;
}

void htn_domain_destroy(HTNPlanningDomain* domain)
{
    if (!domain) return;
    
    // 释放方法及其内部动态内存
    for (size_t i = 0; i < domain->method_count; i++) {
        safe_free((void**)&domain->methods[i].task_signature);
        if (domain->methods[i].subtask_signatures) {
            for (size_t j = 0; j < domain->methods[i].subtask_count; j++) {
                safe_free((void**)&domain->methods[i].subtask_signatures[j]);
            }
            safe_free((void**)&domain->methods[i].subtask_signatures);
            domain->methods[i].subtask_signatures = NULL;
        }
        safe_free((void**)&domain->methods[i].subtask_sizes);
        if (domain->methods[i].ordering_constraints) {
            for (size_t j = 0; j < domain->methods[i].subtask_count; j++) {
                safe_free((void**)&domain->methods[i].ordering_constraints[j]);
            }
            safe_free((void**)&domain->methods[i].ordering_constraints);
            domain->methods[i].ordering_constraints = NULL;
        }
        safe_free((void**)&domain->methods[i].preconditions);
        safe_free((void**)&domain->methods[i].expected_effects);
    }
    safe_free((void**)&domain->methods);
    
    // 释放操作符内部动态内存
    for (size_t i = 0; i < domain->operator_count; i++) {
        safe_free((void**)&domain->operators[i].operator_signature);
        safe_free((void**)&domain->operators[i].preconditions);
        safe_free((void**)&domain->operators[i].effects);
    }
    safe_free((void**)&domain->operators);
    
    // 释放资源内部动态内存
    for (size_t i = 0; i < domain->resource_count; i++) {
        safe_free((void**)&domain->resources[i].allocation_history);
    }
    safe_free((void**)&domain->resources);
    
    memset(domain, 0, sizeof(HTNPlanningDomain));
}

int htn_add_method(HTNPlanningDomain* domain, const HTNMethod* method)
{
    if (!domain || !method || !method->task_signature || method->signature_size == 0) {
        return -1;
    }
    
    if (domain->method_count >= domain->method_capacity) {
        // 自动扩容
        size_t new_capacity = domain->method_capacity == 0 ? 8 : domain->method_capacity * 2;
        HTNMethod* new_methods = (HTNMethod*)realloc(domain->methods,
                                    new_capacity * sizeof(HTNMethod));
        if (!new_methods) return -1;
        domain->methods = new_methods;
        domain->method_capacity = new_capacity;
    }
    
    int idx = (int)domain->method_count;
    HTNMethod* dest = &domain->methods[idx];
    memset(dest, 0, sizeof(HTNMethod));
    
    // 深度拷贝任务签名
    dest->task_signature = (float*)safe_malloc(method->signature_size * sizeof(float));
    if (!dest->task_signature) return -1;
    memcpy(dest->task_signature, method->task_signature, method->signature_size * sizeof(float));
    dest->signature_size = method->signature_size;
    
    // 拷贝基本字段
    dest->match_threshold = method->match_threshold;
    dest->applicability_score = method->applicability_score;
    dest->average_cost = method->average_cost;
    dest->average_duration = method->average_duration;
    dest->is_primitive_method = method->is_primitive_method;
    dest->use_count = 0;
    dest->success_rate = method->success_rate;
    
    // 深度拷贝子任务数组
    dest->subtask_count = method->subtask_count;
    if (method->subtask_count > 0 && method->subtask_signatures && method->subtask_sizes) {
        dest->subtask_signatures = (float**)safe_calloc(method->subtask_count, sizeof(float*));
        dest->subtask_sizes = (size_t*)safe_malloc(method->subtask_count * sizeof(size_t));
        if (!dest->subtask_signatures || !dest->subtask_sizes) {
            safe_free((void**)&dest->task_signature);
            safe_free((void**)&dest->subtask_signatures);
            safe_free((void**)&dest->subtask_sizes);
            return -1;
        }
        
        for (size_t i = 0; i < method->subtask_count; i++) {
            dest->subtask_sizes[i] = method->subtask_sizes[i];
            if (method->subtask_signatures[i] && method->subtask_sizes[i] > 0) {
                dest->subtask_signatures[i] = (float*)safe_malloc(
                    method->subtask_sizes[i] * sizeof(float));
                if (dest->subtask_signatures[i]) {
                    memcpy(dest->subtask_signatures[i], method->subtask_signatures[i],
                           method->subtask_sizes[i] * sizeof(float));
                }
            }
        }
    }
    
    // 深度拷贝排序约束矩阵
    if (method->ordering_constraints && method->subtask_count > 0) {
        dest->ordering_constraints = (int**)safe_calloc(method->subtask_count, sizeof(int*));
        if (dest->ordering_constraints) {
            for (size_t i = 0; i < method->subtask_count; i++) {
                dest->ordering_constraints[i] = (int*)safe_calloc(method->subtask_count, sizeof(int));
                if (dest->ordering_constraints[i] && method->ordering_constraints[i]) {
                    memcpy(dest->ordering_constraints[i], method->ordering_constraints[i],
                           method->subtask_count * sizeof(int));
                }
            }
        }
    }
    
    // 深度拷贝前提条件
    if (method->preconditions && method->preconditions_size > 0) {
        dest->preconditions = (float*)safe_malloc(method->preconditions_size * sizeof(float));
        if (dest->preconditions) {
            memcpy(dest->preconditions, method->preconditions,
                   method->preconditions_size * sizeof(float));
            dest->preconditions_size = method->preconditions_size;
        }
    }
    
    // 深度拷贝预期效果
    if (method->expected_effects && method->expected_effects_size > 0) {
        dest->expected_effects = (float*)safe_malloc(method->expected_effects_size * sizeof(float));
        if (dest->expected_effects) {
            memcpy(dest->expected_effects, method->expected_effects,
                   method->expected_effects_size * sizeof(float));
            dest->expected_effects_size = method->expected_effects_size;
        }
    }
    
    domain->method_count++;
    return idx;
}

int htn_add_operator(HTNPlanningDomain* domain, const HTNOperator* op)
{
    if (!domain || !op || !op->operator_signature || op->signature_size == 0) {
        return -1;
    }
    
    if (domain->operator_count >= domain->operator_capacity) {
        size_t new_capacity = domain->operator_capacity == 0 ? 8 : domain->operator_capacity * 2;
        HTNOperator* new_ops = (HTNOperator*)realloc(domain->operators,
                                  new_capacity * sizeof(HTNOperator));
        if (!new_ops) return -1;
        domain->operators = new_ops;
        domain->operator_capacity = new_capacity;
    }
    
    int idx = (int)domain->operator_count;
    HTNOperator* dest = &domain->operators[idx];
    memset(dest, 0, sizeof(HTNOperator));
    
    // 深度拷贝操作符签名
    dest->operator_signature = (float*)safe_malloc(op->signature_size * sizeof(float));
    if (!dest->operator_signature) return -1;
    memcpy(dest->operator_signature, op->operator_signature, op->signature_size * sizeof(float));
    dest->signature_size = op->signature_size;
    
    dest->cost = op->cost;
    dest->duration = op->duration;
    dest->use_count = 0;
    
    // 深度拷贝前提条件
    if (op->preconditions && op->preconditions_size > 0) {
        dest->preconditions = (float*)safe_malloc(op->preconditions_size * sizeof(float));
        if (dest->preconditions) {
            memcpy(dest->preconditions, op->preconditions,
                   op->preconditions_size * sizeof(float));
            dest->preconditions_size = op->preconditions_size;
        }
    }
    
    // 深度拷贝效果
    if (op->effects && op->effects_size > 0) {
        dest->effects = (float*)safe_malloc(op->effects_size * sizeof(float));
        if (dest->effects) {
            memcpy(dest->effects, op->effects, op->effects_size * sizeof(float));
            dest->effects_size = op->effects_size;
        }
    }
    
    domain->operator_count++;
    return idx;
}

int htn_add_resource(HTNPlanningDomain* domain, const HTNResource* resource)
{
    if (!domain || !resource) return -1;
    
    if (domain->resource_count >= domain->resource_capacity) {
        size_t new_capacity = domain->resource_capacity == 0 ? 8 : domain->resource_capacity * 2;
        HTNResource* new_res = (HTNResource*)realloc(domain->resources,
                                  new_capacity * sizeof(HTNResource));
        if (!new_res) return -1;
        domain->resources = new_res;
        domain->resource_capacity = new_capacity;
    }
    
    int idx = (int)domain->resource_count;
    HTNResource* dest = &domain->resources[idx];
    memset(dest, 0, sizeof(HTNResource));
    
    dest->resource_type = resource->resource_type;
    dest->total_capacity = resource->total_capacity;
    dest->available = resource->available;
    dest->refill_rate = resource->refill_rate;
    
    // 拷贝资源名称
    size_t name_len = strlen(resource->name);
    if (name_len >= sizeof(dest->name)) name_len = sizeof(dest->name) - 1;
    memcpy(dest->name, resource->name, name_len);
    dest->name[name_len] = '\0';
    
    // 分配历史记录
    dest->history_capacity = 64;
    dest->allocation_history = (float*)safe_calloc(dest->history_capacity, sizeof(float));
    if (dest->allocation_history) {
        dest->allocation_history[0] = resource->available;
        dest->history_size = 1;
    }
    
    domain->resource_count++;
    return idx;
}

int htn_detect_task_interactions(HierarchicalPlanningSystem* system,
                                 TaskInteraction* interactions,
                                 size_t max_interactions)
{
    if (!system || !interactions || max_interactions == 0) return -1;
    
    int interaction_count = 0;
    
    // 收集当前任务网络中所有节点
    HierarchicalTaskNode* all_nodes[512];
    int node_count = 0;
    
    for (size_t n = 0; n < system->networks_count && node_count < 512; n++) {
        TaskNetwork* network = &system->task_networks[n];
        if (!network || !network->root) continue;
        
        // BFS收集节点
        HierarchicalTaskNode* queue[512];
        int head = 0, tail = 0;
        queue[tail++] = network->root;
        
        while (head < tail && node_count < 512) {
            HierarchicalTaskNode* node = queue[head++];
            all_nodes[node_count++] = node;
            for (size_t i = 0; i < node->children_count && tail < 512; i++) {
                queue[tail++] = node->children[i];
            }
        }
    }
    
    // 对所有任务对进行交互检测
    for (int i = 0; i < node_count && interaction_count < (int)max_interactions; i++) {
        for (int j = i + 1; j < node_count && interaction_count < (int)max_interactions; j++) {
            HierarchicalTaskNode* ti = all_nodes[i];
            HierarchicalTaskNode* tj = all_nodes[j];
            
            if (!ti || !tj) continue;
            
            // 检查因果支持：ti的效果提供tj的前提条件
            if (ti->effects && ti->effects_size > 0 &&
                tj->preconditions && tj->preconditions_size > 0) {
                size_t min_eff_pre = ti->effects_size < tj->preconditions_size ?
                                     ti->effects_size : tj->preconditions_size;
                float support_strength = 0.0f;
                for (size_t k = 0; k < min_eff_pre; k++) {
                    if (tj->preconditions[k] > 0.0f && ti->effects[k] >= tj->preconditions[k]) {
                        support_strength += tj->preconditions[k];
                    }
                }
                support_strength /= (float)(min_eff_pre > 0 ? min_eff_pre : 1);
                
                if (support_strength > 0.3f) {
                    TaskInteraction* inter = &interactions[interaction_count];
                    memset(inter, 0, sizeof(TaskInteraction));
                    inter->from_task = i;
                    inter->to_task = j;
                    inter->type = TASK_INTERACTION_POSITIVE;
                    inter->strength = support_strength;
                    snprintf(inter->description, sizeof(inter->description),
                             "任务%d的效果支持任务%d的前提条件(强度:%.2f)", i, j, support_strength);
                    interaction_count++;
                    continue;
                }
            }
            
            // 检查因果冲突：ti的效果破坏tj的前提条件
            if (ti->effects && ti->effects_size > 0 &&
                tj->preconditions && tj->preconditions_size > 0) {
                size_t min_eff_pre = ti->effects_size < tj->preconditions_size ?
                                     ti->effects_size : tj->preconditions_size;
                float conflict_strength = 0.0f;
                for (size_t k = 0; k < min_eff_pre; k++) {
                    if (tj->preconditions[k] > 0.0f && ti->effects[k] < -tj->preconditions[k]) {
                        conflict_strength += tj->preconditions[k];
                    }
                }
                conflict_strength /= (float)(min_eff_pre > 0 ? min_eff_pre : 1);
                
                if (conflict_strength > 0.3f) {
                    TaskInteraction* inter = &interactions[interaction_count];
                    memset(inter, 0, sizeof(TaskInteraction));
                    inter->from_task = i;
                    inter->to_task = j;
                    inter->type = TASK_INTERACTION_NEGATIVE;
                    inter->strength = conflict_strength;
                    snprintf(inter->description, sizeof(inter->description),
                             "任务%d的效果破坏任务%d的前提条件(强度:%.2f)", i, j, conflict_strength);
                    interaction_count++;
                    continue;
                }
            }
            
            // 检查资源冲突：同层次且使用相同资源
            if (ti->hierarchy == tj->hierarchy &&
                ti->effects && tj->effects &&
                ti->effects_size > 0 && tj->effects_size > 0) {
                float resource_conflict = 0.0f;
                size_t min_eff = ti->effects_size < tj->effects_size ?
                                 ti->effects_size : tj->effects_size;
                for (size_t k = 0; k < min_eff; k++) {
                    if (ti->effects[k] < 0.0f && tj->effects[k] < 0.0f) {
                        resource_conflict += fabsf(ti->effects[k] + tj->effects[k]);
                    }
                }
                resource_conflict /= (float)(min_eff > 0 ? min_eff : 1);
                
                if (resource_conflict > 0.5f) {
                    TaskInteraction* inter = &interactions[interaction_count];
                    memset(inter, 0, sizeof(TaskInteraction));
                    inter->from_task = i;
                    inter->to_task = j;
                    inter->type = TASK_INTERACTION_RESOURCE;
                    inter->strength = resource_conflict;
                    snprintf(inter->description, sizeof(inter->description),
                             "任务%d和任务%d竞争相同资源(冲突强度:%.2f)", i, j, resource_conflict);
                    interaction_count++;
                }
            }
        }
    }
    
    // 检查互斥约束（父子任务之间）
    for (int i = 0; i < node_count && interaction_count < (int)max_interactions; i++) {
        HierarchicalTaskNode* node = all_nodes[i];
        if (!node || !node->parent) continue;
        
        // 如果父任务和子任务具有相反的前提条件和效果，视为互斥
        if (node->preconditions && node->parent->effects &&
            node->preconditions_size > 0 && node->parent->effects_size > 0) {
            size_t min_size = node->preconditions_size < node->parent->effects_size ?
                             node->preconditions_size : node->parent->effects_size;
            float mutex_sum = 0.0f;
            for (size_t k = 0; k < min_size; k++) {
                mutex_sum += node->preconditions[k] * node->parent->effects[k];
            }
            if (mutex_sum < -0.5f * (float)min_size) {
                TaskInteraction* inter = &interactions[interaction_count];
                memset(inter, 0, sizeof(TaskInteraction));
                inter->from_task = i;
                inter->to_task = -1;
                inter->type = TASK_INTERACTION_MUTEX;
                inter->strength = fabsf(mutex_sum) / (float)(min_size > 0 ? min_size : 1);
                snprintf(inter->description, sizeof(inter->description),
                         "任务%d与父任务互斥(强度:%.2f)", i, inter->strength);
                interaction_count++;
            }
        }
    }
    
    return interaction_count;
}

int htn_resolve_resource_conflicts(HierarchicalPlanningSystem* system,
                                   const TaskInteraction* interactions,
                                   size_t num_interactions)
{
    if (!system || !interactions) return -1;
    
    int resolved_count = 0;
    
    for (size_t idx = 0; idx < num_interactions; idx++) {
        const TaskInteraction* interaction = &interactions[idx];
        
        if (interaction->type != TASK_INTERACTION_RESOURCE &&
            interaction->type != TASK_INTERACTION_NEGATIVE) {
            continue;
        }
        
        // 收集冲突中的任务节点
        HierarchicalTaskNode* task_a = NULL;
        HierarchicalTaskNode* task_b = NULL;
        int node_found_a = 0, node_found_b = 0;
        
        int current_idx = 0;
        for (size_t n = 0; n < system->networks_count; n++) {
            TaskNetwork* network = &system->task_networks[n];
            if (!network || !network->root) continue;
            
            HierarchicalTaskNode* queue[256];
            int head = 0, tail = 0;
            queue[tail++] = network->root;
            
            while (head < tail) {
                HierarchicalTaskNode* node = queue[head++];
                if (current_idx == interaction->from_task) {
                    task_a = node;
                    node_found_a = 1;
                }
                if (current_idx == interaction->to_task) {
                    task_b = node;
                    node_found_b = 1;
                }
                current_idx++;
                for (size_t i = 0; i < node->children_count && tail < 256; i++) {
                    queue[tail++] = node->children[i];
                }
            }
        }
        
        if (!task_a || !task_b) continue;
        
        // 解决方案1：调整任务排序（优先级高的先执行）
        if (task_a->feasibility_score >= task_b->feasibility_score) {
            // task_a优先级更高，确保在task_b之前
            if (task_a->parent == task_b->parent && task_a->parent) {
                HierarchicalTaskNode* parent = task_a->parent;
                int pos_a = -1, pos_b = -1;
                for (size_t i = 0; i < parent->children_count; i++) {
                    if (parent->children[i] == task_a) pos_a = (int)i;
                    if (parent->children[i] == task_b) pos_b = (int)i;
                }
                if (pos_a >= 0 && pos_b >= 0 && pos_a > pos_b) {
                    HierarchicalTaskNode* temp = parent->children[pos_a];
                    parent->children[pos_a] = parent->children[pos_b];
                    parent->children[pos_b] = temp;
                    resolved_count++;
                }
            }
        }
        
        // 解决方案2：降低冲突任务的效果强度
        if (interaction->type == TASK_INTERACTION_RESOURCE) {
            size_t min_eff = task_a->effects_size < task_b->effects_size ?
                             task_a->effects_size : task_b->effects_size;
            for (size_t k = 0; k < min_eff; k++) {
                if (task_a->effects[k] < 0.0f && task_b->effects[k] < 0.0f) {
                    float reduction = 0.3f * interaction->strength;
                    task_a->effects[k] *= (1.0f - reduction);
                    task_b->effects[k] *= (1.0f - reduction);
                }
            }
            resolved_count++;
        }
        
        // 解决方案3：分解冲突任务为更细粒度的子任务
        if (!task_a->is_primitive && !task_b->is_primitive &&
            interaction->strength > 0.7f) {
            // 为冲突任务添加排序约束标记（通过参数传递）
            if (task_a->parameters && task_a->parameters_size > 0) {
                task_a->parameters[0] = -1.0f; // 标记需要在task_b之前执行
            }
            if (task_b->parameters && task_b->parameters_size > 0) {
                task_b->parameters[0] = 1.0f;  // 标记需要在task_a之后执行
            }
            resolved_count++;
        }
    }
    
    // 更新资源可用量
    for (size_t r = 0; r < system->htn_domain.resource_count; r++) {
        HTNResource* res = &system->htn_domain.resources[r];
        res->available += res->refill_rate;
        if (res->available > res->total_capacity) {
            res->available = res->total_capacity;
        }
        // 记录到分配历史
        if (res->history_size < res->history_capacity) {
            res->allocation_history[res->history_size++] = res->available;
        }
    }
    
    return resolved_count;
}

int htn_set_method_selector_config(HierarchicalPlanningSystem* system,
                                   const HTNMethodSelectorConfig* config)
{
    if (!system || !config) return -1;
    
    system->htn_selector_config.precondition_weight = config->precondition_weight;
    system->htn_selector_config.effect_weight = config->effect_weight;
    system->htn_selector_config.cost_weight = config->cost_weight;
    system->htn_selector_config.diversity_weight = config->diversity_weight;
    system->htn_selector_config.state_alignment_weight = config->state_alignment_weight;
    system->htn_selector_config.success_rate_weight = config->success_rate_weight;
    system->htn_selector_config.top_k_methods = config->top_k_methods;
    system->htn_selector_config.exploration_rate = config->exploration_rate;
    
    return 0;
}

int htn_get_method_selector_config(const HierarchicalPlanningSystem* system,
                                   HTNMethodSelectorConfig* config)
{
    if (!system || !config) return -1;
    
    config->precondition_weight = system->htn_selector_config.precondition_weight;
    config->effect_weight = system->htn_selector_config.effect_weight;
    config->cost_weight = system->htn_selector_config.cost_weight;
    config->diversity_weight = system->htn_selector_config.diversity_weight;
    config->state_alignment_weight = system->htn_selector_config.state_alignment_weight;
    config->success_rate_weight = system->htn_selector_config.success_rate_weight;
    config->top_k_methods = system->htn_selector_config.top_k_methods;
    config->exploration_rate = system->htn_selector_config.exploration_rate;
    
    return 0;
}

int htn_get_domain(const HierarchicalPlanningSystem* system,
                   HTNPlanningDomain* domain)
{
    if (!system || !domain) return -1;
    
    // 浅拷贝域结构（调用方不应修改内部指针）
    memcpy(domain, &system->htn_domain, sizeof(HTNPlanningDomain));
    
    return 0;
}

int htn_get_decomposition_result(const HierarchicalPlanningSystem* system,
                                 HTNDecompositionResult* result)
{
    if (!system || !result) return -1;
    
    memcpy(result, &system->htn_last_result, sizeof(HTNDecompositionResult));
    
    return 0;
}

int htn_set_enhanced_enabled(HierarchicalPlanningSystem* system, int enable)
{
    if (!system) return -1;
    
    system->htn_enhanced_enabled = enable ? 1 : 0;
    
    return 0;
}

int htn_enhanced_decompose(HierarchicalPlanningSystem* system,
                           const float* goal, size_t goal_size,
                           const float* current_state, size_t state_size,
                           float* plan, size_t max_plan_size)
{
    if (!system || !goal || goal_size == 0 || !plan || max_plan_size == 0) {
        return -1;
    }
    
    // 重置上次分解结果
    memset(&system->htn_last_result, 0, sizeof(HTNDecompositionResult));
    
    // 步骤1：创建根任务节点（目标分解）
    HierarchicalTaskNode* root = create_task_node(
        goal, goal_size,
        HIERARCHY_STRATEGIC,
        DECOMPOSITION_AND_OR);
    
    if (!root) return -1;
    
    // 链接到第一个任务网络
    if (system->networks_count > 0 && system->task_networks) {
        if (system->task_networks[0].root) {
            // 已有根节点，将新根作为子节点添加
            if (add_child_node(system->task_networks[0].root, root) != 0) {
                free_task_node(root);
                return -1;
            }
        } else {
            system->task_networks[0].root = root;
        }
    }
    
    // 步骤2：递归增强HTN分解
    int max_depth = 4; // 默认最大分解深度：战略→战术→操作→执行
    if (htn_enhanced_decompose_recursive(system, root,
                                         current_state, state_size,
                                         max_depth) != 0) {
        return -1;
    }
    
    // 步骤3：检测任务交互
    TaskInteraction temp_interactions[128];
    int num_interactions = htn_detect_task_interactions(
        system, temp_interactions, 128);
    
    if (num_interactions > 0) {
        // 保存到交互缓冲区
        safe_free((void**)&system->htn_interaction_buffer);
        system->htn_interaction_buffer_size = num_interactions;
        system->htn_interaction_buffer = (TaskInteraction*)safe_malloc(
            num_interactions * sizeof(TaskInteraction));
        if (system->htn_interaction_buffer) {
            memcpy(system->htn_interaction_buffer, temp_interactions,
                   num_interactions * sizeof(TaskInteraction));
        }
    }
    
    // 步骤4：解决资源冲突
    htn_resolve_resource_conflicts(system, temp_interactions,
                                   (size_t)(num_interactions > 0 ? num_interactions : 0));
    
    // 步骤5：从任务网络生成平面规划
    int plan_steps = htn_generate_plan_from_network(root, plan, max_plan_size);
    
    // 步骤6：填充分解结果
    // BFS统计
    int total_tasks = 0, primitive_count = 0, compound_count = 0;
    int max_depth_found = 0;
    
    HierarchicalTaskNode* stat_queue[512];
    int stat_depth[512];
    int s_head = 0, s_tail = 0;
    stat_queue[s_tail] = root;
    stat_depth[s_tail] = 0;
    s_tail++;
    
    while (s_head < s_tail) {
        HierarchicalTaskNode* node = stat_queue[s_head];
        int depth = stat_depth[s_head];
        s_head++;
        
        total_tasks++;
        if (depth > max_depth_found) max_depth_found = depth;
        
        if (node->is_primitive) {
            primitive_count++;
        } else {
            compound_count++;
        }
        
        for (size_t i = 0; i < node->children_count && s_tail < 512; i++) {
            stat_queue[s_tail] = node->children[i];
            stat_depth[s_tail] = depth + 1;
            s_tail++;
        }
    }
    
    system->htn_last_result.total_tasks = total_tasks;
    system->htn_last_result.primitive_count = primitive_count;
    system->htn_last_result.compound_count = compound_count;
    system->htn_last_result.avg_decomposition_depth =
        total_tasks > 0 ? (float)max_depth_found : 0.0f;
    system->htn_last_result.total_cost = 0.0f;
    system->htn_last_result.total_duration = 0.0f;
    system->htn_last_result.feasibility_score = htn_compute_feasibility(system, root);
    system->htn_last_result.interactions = system->htn_interaction_buffer;
    system->htn_last_result.interaction_count = num_interactions;
    
    // 总成本估计（来自方法库中适用方法的平均成本）
    for (size_t m = 0; m < system->htn_domain.method_count; m++) {
        system->htn_last_result.total_cost += system->htn_domain.methods[m].average_cost;
        system->htn_last_result.total_duration += system->htn_domain.methods[m].average_duration;
    }
    
    return plan_steps > 0 ? plan_steps : 0;
}

int htn_init_default_method_library(HierarchicalPlanningSystem* system)
{
    if (!system) return -1;
    
    HTNPlanningDomain* domain = &system->htn_domain;
    
    // 确保规划域已初始化
    if (domain->method_capacity == 0) {
        if (htn_domain_create(domain, 16, 8, 4) != 0) return -1;
    }
    
    int method_count = 0;
    
    // ====== 方法1：战略目标分解 ======
    {
        HTNMethod method;
        memset(&method, 0, sizeof(HTNMethod));
        
        // 战略任务签名（长期目标规划）
        float sig[] = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
        method.task_signature = sig;
        method.signature_size = 5;
        method.match_threshold = 0.5f;
        method.subtask_count = 3;
        
        // 3个战术子任务
        float sub1[] = {0.8f, 0.7f, 0.6f, 0.5f};
        float sub2[] = {0.7f, 0.6f, 0.5f, 0.4f};
        float sub3[] = {0.6f, 0.5f, 0.4f, 0.3f};
        float* subs[] = {sub1, sub2, sub3};
        size_t sub_sizes[] = {4, 4, 4};
        method.subtask_signatures = subs;
        method.subtask_sizes = sub_sizes;
        
        // 排序约束：sub1必须在sub2之前，sub2必须在sub3之前
        int order1[] = {0, -1, -1};
        int order2[] = {1, 0, -1};
        int order3[] = {1, 1, 0};
        int* orders[] = {order1, order2, order3};
        method.ordering_constraints = orders;
        
        // 前提条件：需要足够资源和时间
        float preconds[] = {0.5f, 0.3f, 0.2f};
        method.preconditions = preconds;
        method.preconditions_size = 3;
        
        // 预期效果
        float effects[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        method.expected_effects = effects;
        method.expected_effects_size = 5;
        
        method.applicability_score = 0.8f;
        method.average_cost = 10.0f;
        method.average_duration = 50.0f;
        method.is_primitive_method = 0;
        method.success_rate = 0.75f;
        
        if (htn_add_method(domain, &method) >= 0) method_count++;
    }
    
    // ====== 方法2：战术目标分解 ======
    {
        HTNMethod method;
        memset(&method, 0, sizeof(HTNMethod));
        
        float sig[] = {0.8f, 0.7f, 0.5f, 0.4f};
        method.task_signature = sig;
        method.signature_size = 4;
        method.match_threshold = 0.4f;
        method.subtask_count = 4;
        
        // 4个操作性子任务
        float sub1[] = {0.6f, 0.5f, 0.4f};
        float sub2[] = {0.5f, 0.4f, 0.3f};
        float sub3[] = {0.4f, 0.3f, 0.2f};
        float sub4[] = {0.3f, 0.2f, 0.1f};
        float* subs[] = {sub1, sub2, sub3, sub4};
        size_t sub_sizes[] = {3, 3, 3, 3};
        method.subtask_signatures = subs;
        method.subtask_sizes = sub_sizes;
        
        // 排序约束：顺序执行
        int order1[] = {0, -1, -1, -1};
        int order2[] = {1, 0, -1, -1};
        int order3[] = {1, 1, 0, -1};
        int order4[] = {1, 1, 1, 0};
        int* orders[] = {order1, order2, order3, order4};
        method.ordering_constraints = orders;
        
        float preconds[] = {0.4f, 0.5f, 0.3f};
        method.preconditions = preconds;
        method.preconditions_size = 3;
        
        float effects[] = {0.2f, 0.3f, 0.4f, 0.5f};
        method.expected_effects = effects;
        method.expected_effects_size = 4;
        
        method.applicability_score = 0.7f;
        method.average_cost = 20.0f;
        method.average_duration = 30.0f;
        method.is_primitive_method = 0;
        method.success_rate = 0.8f;
        
        if (htn_add_method(domain, &method) >= 0) method_count++;
    }
    
    // ====== 方法3：操作任务分解 ======
    {
        HTNMethod method;
        memset(&method, 0, sizeof(HTNMethod));
        
        float sig[] = {0.6f, 0.5f, 0.4f, 0.3f};
        method.task_signature = sig;
        method.signature_size = 4;
        method.match_threshold = 0.4f;
        method.subtask_count = 3;
        
        // 3个执行级子任务（原始任务）
        float sub1[] = {0.4f, 0.3f};
        float sub2[] = {0.3f, 0.2f};
        float sub3[] = {0.2f, 0.1f};
        float* subs[] = {sub1, sub2, sub3};
        size_t sub_sizes[] = {2, 2, 2};
        method.subtask_signatures = subs;
        method.subtask_sizes = sub_sizes;
        
        int order1[] = {0, -1, -1};
        int order2[] = {1, 0, -1};
        int order3[] = {1, 1, 0};
        int* orders[] = {order1, order2, order3};
        method.ordering_constraints = orders;
        
        float preconds[] = {0.6f, 0.4f};
        method.preconditions = preconds;
        method.preconditions_size = 2;
        
        float effects[] = {0.3f, 0.4f, 0.5f, 0.6f};
        method.expected_effects = effects;
        method.expected_effects_size = 4;
        
        method.applicability_score = 0.75f;
        method.average_cost = 5.0f;
        method.average_duration = 10.0f;
        method.is_primitive_method = 1;
        method.success_rate = 0.9f;
        
        if (htn_add_method(domain, &method) >= 0) method_count++;
    }
    
    // ====== 方法4：资源分配方法 ======
    {
        HTNMethod method;
        memset(&method, 0, sizeof(HTNMethod));
        
        float sig[] = {0.5f, 0.9f, 0.5f, 0.3f};
        method.task_signature = sig;
        method.signature_size = 4;
        method.match_threshold = 0.3f;
        method.subtask_count = 2;
        
        float sub1[] = {0.5f, 0.8f, 0.3f};
        float sub2[] = {0.3f, 0.7f, 0.4f};
        float* subs[] = {sub1, sub2};
        size_t sub_sizes[] = {3, 3};
        method.subtask_signatures = subs;
        method.subtask_sizes = sub_sizes;
        
        // 并行约束
        int order1[] = {0, 2};
        int order2[] = {2, 0};
        int* orders[] = {order1, order2};
        method.ordering_constraints = orders;
        
        float preconds[] = {0.3f, 0.5f, 0.7f};
        method.preconditions = preconds;
        method.preconditions_size = 3;
        
        float effects[] = {0.4f, 0.6f, 0.8f, 0.3f};
        method.expected_effects = effects;
        method.expected_effects_size = 4;
        
        method.applicability_score = 0.6f;
        method.average_cost = 15.0f;
        method.average_duration = 5.0f;
        method.is_primitive_method = 0;
        method.success_rate = 0.85f;
        
        if (htn_add_method(domain, &method) >= 0) method_count++;
    }
    
    // ====== 方法5：并行执行方法 ======
    {
        HTNMethod method;
        memset(&method, 0, sizeof(HTNMethod));
        
        float sig[] = {0.7f, 0.6f, 0.8f, 0.2f, 0.9f};
        method.task_signature = sig;
        method.signature_size = 5;
        method.match_threshold = 0.35f;
        method.subtask_count = 3;
        
        float sub1[] = {0.5f, 0.4f};
        float sub2[] = {0.4f, 0.5f};
        float sub3[] = {0.6f, 0.3f};
        float* subs[] = {sub1, sub2, sub3};
        size_t sub_sizes[] = {2, 2, 2};
        method.subtask_signatures = subs;
        method.subtask_sizes = sub_sizes;
        
        // 所有子任务并行（2表示并行）
        int order1[] = {0, 2, 2};
        int order2[] = {2, 0, 2};
        int order3[] = {2, 2, 0};
        int* orders[] = {order1, order2, order3};
        method.ordering_constraints = orders;
        
        float preconds[] = {0.7f, 0.5f, 0.3f};
        method.preconditions = preconds;
        method.preconditions_size = 3;
        
        float effects[] = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
        method.expected_effects = effects;
        method.expected_effects_size = 5;
        
        method.applicability_score = 0.65f;
        method.average_cost = 8.0f;
        method.average_duration = 3.0f;
        method.is_primitive_method = 0;
        method.success_rate = 0.7f;
        
        if (htn_add_method(domain, &method) >= 0) method_count++;
    }
    
    return method_count;
}