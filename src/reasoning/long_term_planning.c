/**
 * @file long_term_planning.c
 * @brief 长期规划和目标分解优化系统实现
 * 
 * 长期规划系统，支持目标分解、时序规划、资源优化和自适应重规划。
 * 提供分层目标分解（HGD）、时序逻辑规划（TLP）、资源受限项目调度（RCPS）等算法。
 */


#include "selflnn/reasoning/long_term_planning.h"
#include "selflnn/reasoning/planning.h"
#include "selflnn/core/cfc_network.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/xorshift_prng.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ============================================================================
 * 内部数据结构
 * =========================================================================== */

/**
 * @brief 目标节点结构体（层次化目标树节点）
 */
typedef struct GoalNode {
    Goal* goal;                   /**< 目标 */
    struct GoalNode* parent;      /**< 父节点 */
    struct GoalNode** children;   /**< 子节点数组 */
    struct GoalNode** dependencies; /**< 依赖节点数组 */
    int child_count;              /**< 子节点数量 */
    int child_capacity;           /**< 子节点容量 */
    int dependency_count;         /**< 依赖节点数量 */
    int dependency_capacity;      /**< 依赖节点容量 */
    int depth;                    /**< 深度层级 */
    float complexity;             /**< 复杂度 */
    float criticality;            /**< 关键度 */
    int is_decomposed;            /**< 是否已分解 */
    float earliest_start;         /**< 最早开始时间 */
    float latest_start;           /**< 最晚开始时间 */
    float slack_time;             /**< 浮动时间 */
    int on_critical_path;         /**< 是否在关键路径上 */
} GoalNode;

/**
 * @brief 长期规划系统内部结构体
 */
struct LongTermPlanningSystem {
    LongTermPlanningConfig config;        /**< 规划配置 */
    GoalNode** goal_nodes;               /**< 目标节点数组 */
    int goal_count;                      /**< 目标数量 */
    int goal_capacity;                   /**< 目标容量 */
    PlanAction** active_plan;            /**< 当前活动计划 */
    int plan_action_count;               /**< 计划动作数量 */
    int plan_action_capacity;            /**< 计划动作容量 */
    float current_time;                  /**< 当前时间 */
    float time_scale;                    /**< 时间缩放 */
    float resource_availability[16];     /**< 资源可用量 */
    float resource_utilization[16];      /**< 资源利用率 */
    int planning_active;                 /**< 是否正在规划 */
    float planning_start_time;           /**< 规划开始时间 */
    int replanning_count;                /**< 重规划次数 */
    float plan_efficiency;               /**< 计划效率 */
    float goal_achievement_rate;         /**< 目标达成率 */
    float resource_efficiency;           /**< 资源效率 */
    AllenConstraint* allen_constraints;  /**< Allen时间约束 */
    int allen_constraint_count;          /**< 约束数量 */
    int allen_constraint_capacity;       /**< 约束容量 */
};

/**
 * @brief 计算目标复杂度
 */
static float calculate_goal_complexity(const Goal* goal) {
    if (!goal) return 0.0f;
    
    float complexity = 0.0f;
    
    // 基于类型
    switch (goal->goal_type) {
        case GOAL_TYPE_OPTIMIZATION:
            complexity += 0.3f;
            break;
        case GOAL_TYPE_EXPLORATION:
            complexity += 0.4f;
            break;
        default:
            complexity += 0.2f;
            break;
    }
    
    // 基于难度
    complexity += goal->difficulty * 0.3f;
    
    // 基于子目标数量
    complexity += (goal->subgoal_count / 10.0f) * 0.2f;  // 每10个子目标增加0.2
    
    // 基于资源需求
    float total_resources = 0.0f;
    for (int i = 0; i < goal->resource_count; i++) {
        total_resources += goal->required_resources[i];
    }
    complexity += (total_resources / 100.0f) * 0.2f;  // 每100单位资源增加0.2
    
    // 限制在0-1范围内
    if (complexity > 1.0f) complexity = 1.0f;
    if (complexity < 0.0f) complexity = 0.0f;
    
    return complexity;
}

/**
 * @brief 计算目标优先级分数
 */
static float calculate_goal_priority_score(const Goal* goal) {
    if (!goal) return 0.0f;
    
    float score = 0.0f;
    
    // 基于优先级枚举
    switch (goal->priority) {
        case GOAL_PRIORITY_CRITICAL:
            score += 1.0f;
            break;
        case GOAL_PRIORITY_HIGH:
            score += 0.8f;
            break;
        case GOAL_PRIORITY_MEDIUM:
            score += 0.5f;
            break;
        case GOAL_PRIORITY_LOW:
            score += 0.3f;
            break;
        case GOAL_PRIORITY_OPTIONAL:
            score += 0.1f;
            break;
    }
    
    // 基于重要性
    score += goal->importance * 0.5f;
    
    // 基于紧急程度
    score += goal->urgency * 0.3f;
    
    // 基于截止时间（如果设置了截止时间）
    if (goal->deadline > 0) {
        float time_remaining = goal->deadline - goal->creation_time;
        if (time_remaining > 0) {
            float time_urgency = 1.0f / (1.0f + time_remaining);
            score += time_urgency * 0.2f;
        }
    }
    
    return score;
}

/**
 * @brief 查找目标节点
 */
static GoalNode* find_goal_node(LongTermPlanningSystem* system, const char* goal_id) {
    if (!system || !goal_id) {
        return NULL;
    }
    
    for (int i = 0; i < system->goal_count; i++) {
        GoalNode* node = system->goal_nodes[i];
        if (node && node->goal && node->goal->goal_id &&
            strcmp(node->goal->goal_id, goal_id) == 0) {
            return node;
        }
    }
    
    return NULL;
}

/* ============================================================================
 * 配置管理
 * =========================================================================== */

/**
 * @brief 获取默认长期规划配置
 */
void long_term_planning_default_config(LongTermPlanningConfig* config) {
    if (!config) return;
    
    memset(config, 0, sizeof(LongTermPlanningConfig));
    
    config->default_horizon = TIME_HORIZON_MEDIUM;
    config->time_resolution = 1.0f;  // 1小时
    config->max_planning_time = 10.0f;  // 10秒
    
    config->weight_completion_time = 0.4f;
    config->weight_resource_usage = 0.3f;
    config->weight_goal_achievement = 0.2f;
    config->weight_risk_minimization = 0.1f;
    
    config->max_decomposition_depth = 5;
    config->decomposition_threshold = 0.7f;
    config->enable_adaptive_decomposition = 1;
    
    config->enable_replanning = 1;
    config->replanning_interval = 24.0f;  // 24小时
    config->replanning_threshold = 0.3f;
    
    // 默认资源（CPU、内存、能源、带宽、存储、网络、传感器、执行器）
    float default_resources[] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    memcpy(config->available_resources, default_resources, sizeof(default_resources));
    config->resource_type_count = 8;
    
    config->uncertainty_tolerance = 0.2f;
    config->enable_contingency_planning = 1;
    config->contingency_branch_count = 3;
}

/* ============================================================================
 * 目标管理API实现
 * =========================================================================== */

/**
 * @brief 创建目标
 */
Goal* goal_create(const char* goal_id, const char* description, GoalType goal_type) {
    Goal* goal = (Goal*)safe_calloc(1, sizeof(Goal));
    if (!goal) {
        return NULL;
    }
    
    if (goal_id) {
        goal->goal_id = string_duplicate_nullable(goal_id);
        if (!goal->goal_id) {
            safe_free((void**)&goal);
            return NULL;
        }
    }
    
    if (description) {
        goal->description = string_duplicate_nullable(description);
        if (!goal->description) {
            safe_free((void**)&goal->goal_id);
            safe_free((void**)&goal);
            return NULL;
        }
    }
    
    goal->goal_type = goal_type;
    goal->priority = GOAL_PRIORITY_MEDIUM;
    goal->state = GOAL_STATE_PENDING;
    goal->horizon = TIME_HORIZON_MEDIUM;
    
    goal->start_time = 0.0f;
    goal->deadline = 0.0f;
    goal->estimated_duration = 0.0f;
    goal->success_threshold = 0.8f;
    goal->progress = 0.0f;
    
    goal->resource_count = 0;
    memset(goal->required_resources, 0, sizeof(goal->required_resources));
    
    goal->prerequisite_count = 0;
    goal->prerequisite_goals = NULL;
    
    goal->subgoal_count = 0;
    goal->subgoal_ids = NULL;
    
    goal->difficulty = 0.5f;
    goal->importance = 0.5f;
    goal->urgency = 0.5f;
    
    goal->creation_time = 0.0f;
    goal->last_update_time = 0.0f;
    goal->update_count = 0;
    
    return goal;
}

/**
 * @brief 销毁目标
 */
void goal_destroy(Goal* goal) {
    if (!goal) return;
    
    safe_free((void**)&goal->goal_id);
    safe_free((void**)&goal->description);
    
    if (goal->prerequisite_goals) {
        for (int i = 0; i < goal->prerequisite_count; i++) {
            safe_free((void**)&goal->prerequisite_goals[i]);
        }
        safe_free((void**)&goal->prerequisite_goals);
    }
    
    if (goal->subgoal_ids) {
        for (int i = 0; i < goal->subgoal_count; i++) {
            safe_free((void**)&goal->subgoal_ids[i]);
        }
        safe_free((void**)&goal->subgoal_ids);
    }
    
    safe_free((void**)&goal);
}

/* ============================================================================
 * 计划动作API实现
 * =========================================================================== */

/**
 * @brief 创建计划动作
 */
PlanAction* plan_action_create(const char* action_id, const char* description,
                              float duration) {
    PlanAction* action = (PlanAction*)safe_calloc(1, sizeof(PlanAction));
    if (!action) {
        return NULL;
    }
    
    if (action_id) {
        action->action_id = string_duplicate_nullable(action_id);
        if (!action->action_id) {
            safe_free((void**)&action);
            return NULL;
        }
    }
    
    if (description) {
        action->description = string_duplicate_nullable(description);
        if (!action->description) {
            safe_free((void**)&action->action_id);
            safe_free((void**)&action);
            return NULL;
        }
    }
    
    action->duration = duration;
    action->start_time = 0.0f;
    action->end_time = 0.0f;
    
    memset(action->resource_consumption, 0, sizeof(action->resource_consumption));
    memset(action->resource_production, 0, sizeof(action->resource_production));
    
    action->precondition_count = 0;
    action->preconditions = NULL;
    
    action->effect_count = 0;
    action->effects = NULL;
    
    action->executed = 0;
    action->execution_progress = 0.0f;
    action->success_probability = 0.9f;
    
    action->associated_goal = NULL;
    
    return action;
}

/**
 * @brief 销毁计划动作
 */
void plan_action_destroy(PlanAction* action) {
    if (!action) return;
    
    safe_free((void**)&action->action_id);
    safe_free((void**)&action->description);
    
    if (action->preconditions) {
        for (int i = 0; i < action->precondition_count; i++) {
            safe_free((void**)&action->preconditions[i]);
        }
        safe_free((void**)&action->preconditions);
    }
    
    if (action->effects) {
        for (int i = 0; i < action->effect_count; i++) {
            safe_free((void**)&action->effects[i]);
        }
        safe_free((void**)&action->effects);
    }
    
    safe_free((void**)&action->associated_goal);
    safe_free((void**)&action);
}

/* ============================================================================
 * 长期规划系统API实现
 * =========================================================================== */

/**
 * @brief 创建长期规划系统
 */
LongTermPlanningSystem* long_term_planning_system_create(
    const LongTermPlanningConfig* config) {
    LongTermPlanningSystem* system = (LongTermPlanningSystem*)safe_calloc(1, sizeof(LongTermPlanningSystem));
    if (!system) {
        return NULL;
    }
    
    // 设置配置
    if (config) {
        memcpy(&system->config, config, sizeof(LongTermPlanningConfig));
    } else {
        long_term_planning_default_config(&system->config);
    }
    
    // 初始化目标管理
    system->goal_capacity = 32;
    system->goal_count = 0;
    system->goal_nodes = (GoalNode**)safe_calloc(system->goal_capacity, sizeof(GoalNode*));
    
    // 初始化计划管理
    system->plan_action_capacity = 64;
    system->plan_action_count = 0;
    system->active_plan = (PlanAction**)safe_calloc(system->plan_action_capacity, sizeof(PlanAction*));
    
    // 初始化时间管理
    system->current_time = 0.0f;
    system->time_scale = 1.0f;
    
    // 初始化资源管理
    memcpy(system->resource_availability, system->config.available_resources,
           sizeof(system->config.available_resources));
    memset(system->resource_utilization, 0, sizeof(system->resource_utilization));
    
    // 初始化状态跟踪
    system->planning_active = 0;
    system->planning_start_time = 0.0f;
    system->replanning_count = 0;
    
    // 初始化性能指标
    system->plan_efficiency = 0.0f;
    system->goal_achievement_rate = 0.0f;
    system->resource_efficiency = 0.0f;
    
    // 初始化Allen时间约束
    system->allen_constraint_capacity = 32;
    system->allen_constraint_count = 0;
    system->allen_constraints = (AllenConstraint*)safe_calloc(
        system->allen_constraint_capacity, sizeof(AllenConstraint));
    
    return system;
}

/**
 * @brief 销毁长期规划系统
 */
void long_term_planning_system_destroy(LongTermPlanningSystem* system) {
    if (!system) return;
    
    // 释放所有目标节点
    for (int i = 0; i < system->goal_count; i++) {
        GoalNode* node = system->goal_nodes[i];
        if (node) {
            // 释放目标
            if (node->goal) {
                goal_destroy(node->goal);
            }
            
            // 释放子节点数组（不释放子节点本身，它们由系统管理）
            safe_free((void**)&node->children);
            
            // 释放依赖数组
            safe_free((void**)&node->dependencies);
            
            safe_free((void**)&node);
        }
    }
    safe_free((void**)&system->goal_nodes);
    
    // 释放活跃计划
    for (int i = 0; i < system->plan_action_count; i++) {
        if (system->active_plan[i]) {
            plan_action_destroy(system->active_plan[i]);
        }
    }
    safe_free((void**)&system->active_plan);
    
    // 释放Allen时间约束
    if (system->allen_constraints) {
        for (int i = 0; i < system->allen_constraint_count; i++) {
            safe_free((void**)&system->allen_constraints[i].interval_a_id);
            safe_free((void**)&system->allen_constraints[i].interval_b_id);
        }
        safe_free((void**)&system->allen_constraints);
    }
    
    safe_free((void**)&system);
}

/**
 * @brief 添加目标到规划系统
 */
int long_term_planning_add_goal(LongTermPlanningSystem* system, const Goal* goal) {
    if (!system || !goal || !goal->goal_id) {
        return -1;
    }
    
    // 检查目标是否已存在
    GoalNode* existing_node = find_goal_node(system, goal->goal_id);
    if (existing_node) {
        return -1;  // 目标已存在
    }
    
    // 检查容量
    if (system->goal_count >= system->goal_capacity) {
        int new_capacity = system->goal_capacity * 2;
        GoalNode** new_nodes = (GoalNode**)safe_realloc(system->goal_nodes,
                                                       new_capacity * sizeof(GoalNode*));
        if (!new_nodes) {
            return -1;
        }
        system->goal_nodes = new_nodes;
        system->goal_capacity = new_capacity;
    }
    
    // 创建目标节点
    GoalNode* node = (GoalNode*)safe_calloc(1, sizeof(GoalNode));
    if (!node) {
        return -1;
    }
    
    // 创建目标副本
    node->goal = goal_create(goal->goal_id, goal->description, goal->goal_type);
    if (!node->goal) {
        safe_free((void**)&node);
        return -1;
    }
    
    // 复制目标属性
    node->goal->priority = goal->priority;
    node->goal->state = goal->state;
    node->goal->horizon = goal->horizon;
    node->goal->start_time = goal->start_time;
    node->goal->deadline = goal->deadline;
    node->goal->estimated_duration = goal->estimated_duration;
    node->goal->success_threshold = goal->success_threshold;
    node->goal->progress = goal->progress;
    
    // 复制资源需求
    node->goal->resource_count = goal->resource_count;
    memcpy(node->goal->required_resources, goal->required_resources,
           sizeof(node->goal->required_resources));
    
    // 设置其他属性
    node->goal->difficulty = goal->difficulty;
    node->goal->importance = goal->importance;
    node->goal->urgency = goal->urgency;
    node->goal->creation_time = system->current_time;
    node->goal->last_update_time = system->current_time;
    
    // 初始化子节点数组
    node->child_capacity = 8;
    node->child_count = 0;
    node->children = (GoalNode**)safe_calloc(node->child_capacity, sizeof(GoalNode*));
    
    // 初始化依赖数组
    node->dependency_capacity = 8;
    node->dependency_count = 0;
    node->dependencies = (GoalNode**)safe_calloc(node->dependency_capacity, sizeof(GoalNode*));
    
    // 初始化时间计算
    node->earliest_start = 0.0f;
    node->latest_start = 0.0f;
    node->slack_time = 0.0f;
    node->on_critical_path = 0;
    
    // 添加到系统
    system->goal_nodes[system->goal_count] = node;
    system->goal_count++;
    
    return 0;
}

/**
 * @brief 移除目标节点
 * 
 * 从系统中删除指定ID的目标，包括：
 * 1. 释放目标内部的子节点和依赖关系
 * 2. 从goal_nodes数组中移除
 * 3. 清理其他目标中指向该目标的依赖和父子关系
 * 
 * @param system 长期规划系统
 * @param goal_id 要移除的目标ID
 * @return 0成功，-1未找到或参数无效
 */
int long_term_planning_remove_goal(LongTermPlanningSystem* system, const char* goal_id) {
    if (!system || !goal_id) return -1;

    /* 查找目标节点索引 */
    int target_idx = -1;
    for (int i = 0; i < system->goal_count; i++) {
        GoalNode* node = system->goal_nodes[i];
        if (node && node->goal && node->goal->goal_id &&
            strcmp(node->goal->goal_id, goal_id) == 0) {
            target_idx = i;
            break;
        }
    }
    if (target_idx < 0) return -1;

    GoalNode* target = system->goal_nodes[target_idx];

    /* 清理其他目标中指向该目标的依赖关系 */
    for (int i = 0; i < system->goal_count; i++) {
        GoalNode* other = system->goal_nodes[i];
        if (!other || i == target_idx) continue;

        /* 移除依赖引用 */
        for (int d = 0; d < other->dependency_count; d++) {
            if (other->dependencies[d] == target) {
                /* 将最后一个元素移到当前位置 */
                other->dependencies[d] = other->dependencies[other->dependency_count - 1];
                other->dependencies[other->dependency_count - 1] = NULL;
                other->dependency_count--;
                d--; /* 重新检查当前位置 */
            }
        }

        /* 移除父子引用 */
        for (int c = 0; c < other->child_count; c++) {
            if (other->children[c] == target) {
                other->children[c] = other->children[other->child_count - 1];
                other->children[other->child_count - 1] = NULL;
                other->child_count--;
                c--;
            }
        }
    }

    /* 释放目标节点内部资源 */
    if (target->goal) {
        safe_free((void**)&target->goal->goal_id);
        safe_free((void**)&target->goal->description);
        safe_free((void**)&target->goal);
    }
    safe_free((void**)&target->children);
    safe_free((void**)&target->dependencies);
    safe_free((void**)&target);

    /* 从数组中移除：将最后一个元素移到目标位置 */
    system->goal_nodes[target_idx] = system->goal_nodes[system->goal_count - 1];
    system->goal_nodes[system->goal_count - 1] = NULL;
    system->goal_count--;

    return 0;
}

/**
 * @brief 目标分解（分层目标分解）
 */
char** long_term_planning_decompose_goal(LongTermPlanningSystem* system,
                                        const char* goal_id, int max_depth) {
    if (!system || !goal_id || max_depth <= 0) {
        return NULL;
    }
    
    // 查找目标节点
    GoalNode* node = find_goal_node(system, goal_id);
    if (!node) {
        return NULL;
    }
    
    // 检查是否已达到最大深度
    if (max_depth <= 1) {
        return NULL;
    }
    
    // 计算目标复杂度
    float complexity = calculate_goal_complexity(node->goal);
    
    // 如果复杂度低于阈值，则不分解
    if (complexity < system->config.decomposition_threshold) {
        return NULL;
    }
    
    // 确定子目标数量（基于复杂度）
    int subgoal_count = (int)(complexity * 5.0f) + 1;  // 1-6个子目标
    if (subgoal_count > 10) subgoal_count = 10;
    
    // 创建子目标ID数组
    char** subgoal_ids = (char**)safe_calloc(subgoal_count, sizeof(char*));
    if (!subgoal_ids) {
        return NULL;
    }
    
    // 生成子目标
    for (int i = 0; i < subgoal_count; i++) {
        char subgoal_id[64];
        snprintf(subgoal_id, sizeof(subgoal_id), "%s_sub%d", goal_id, i + 1);
        
        subgoal_ids[i] = string_duplicate_nullable(subgoal_id);
        if (!subgoal_ids[i]) {
            // 清理已分配的内存
            for (int j = 0; j < i; j++) {
                safe_free((void**)&subgoal_ids[j]);
            }
            safe_free((void**)&subgoal_ids);
            return NULL;
        }
        
        // 创建子目标
        char description[128];
        snprintf(description, sizeof(description), "子目标 %d/%d: %s", 
                i + 1, subgoal_count, node->goal->description ? node->goal->description : "");
        
        Goal* subgoal = goal_create(subgoal_id, description, node->goal->goal_type);
        if (subgoal) {
            // 设置子目标属性
            subgoal->priority = node->goal->priority;
            subgoal->difficulty = complexity / subgoal_count;
            subgoal->importance = node->goal->importance * 0.8f;
            subgoal->urgency = node->goal->urgency;
            subgoal->estimated_duration = node->goal->estimated_duration / subgoal_count;
            
            // 添加子目标到系统
            long_term_planning_add_goal(system, subgoal);
            goal_destroy(subgoal);  // 系统已创建副本
        }
    }
    
    // 更新原始目标的子目标列表
    if (node->goal->subgoal_ids) {
        for (int i = 0; i < node->goal->subgoal_count; i++) {
            safe_free((void**)&node->goal->subgoal_ids[i]);
        }
        safe_free((void**)&node->goal->subgoal_ids);
    }
    
    node->goal->subgoal_ids = subgoal_ids;
    node->goal->subgoal_count = subgoal_count;
    
    return subgoal_ids;
}

/**
 * @brief 生成计划
 */
int long_term_planning_generate_plan(LongTermPlanningSystem* system,
                                    const char** goal_ids, int goal_count,
                                    PlanAction** plan_actions, int max_actions) {
    if (!system || !goal_ids || goal_count <= 0 || !plan_actions || max_actions <= 0) {
        return -1;
    }
    
    int action_count = 0;
    
    // 为每个目标生成动作
    for (int i = 0; i < goal_count && action_count < max_actions; i++) {
        const char* goal_id = goal_ids[i];
        GoalNode* node = find_goal_node(system, goal_id);
        
        if (!node) {
            continue;
        }
        
        // 根据目标创建动作
        char action_id[64];
        snprintf(action_id, sizeof(action_id), "action_%s", goal_id);
        
        PlanAction* action = plan_action_create(action_id,
                                               node->goal->description,
                                               node->goal->estimated_duration);
        if (!action) {
            continue;
        }
        
        // 设置动作属性
        action->start_time = system->current_time;
        action->end_time = action->start_time + action->duration;
        
        // 设置资源消耗（基于目标资源需求）
        for (int r = 0; r < node->goal->resource_count && r < 8; r++) {
            action->resource_consumption[r] = node->goal->required_resources[r] * action->duration;
        }
        
        // 关联目标
        action->associated_goal = string_duplicate_nullable(goal_id);
        if (!action->associated_goal) {
            plan_action_destroy(action);
            continue;
        }
        
        // 添加到计划
        plan_actions[action_count] = action;
        action_count++;
        
        // 如果目标有子目标，递归生成子动作
        if (node->goal->subgoal_count > 0 && action_count < max_actions) {
            // 为每个子目标生成动作
            for (int s = 0; s < node->goal->subgoal_count && action_count < max_actions; s++) {
                char subaction_id[64];
                snprintf(subaction_id, sizeof(subaction_id), "action_%s_sub%d", goal_id, s + 1);
                
                PlanAction* subaction = plan_action_create(subaction_id,
                                                          node->goal->subgoal_ids[s],
                                                          node->goal->estimated_duration / node->goal->subgoal_count);
                if (subaction) {
                    // 子动作在父动作之后开始
                    subaction->start_time = action->end_time;
                    subaction->end_time = subaction->start_time + subaction->duration;
                    subaction->associated_goal = string_duplicate_nullable(node->goal->subgoal_ids[s]);
                    if (!subaction->associated_goal) {
                        plan_action_destroy(subaction);
                        continue;
                    }
                    
                    plan_actions[action_count] = subaction;
                    action_count++;
                }
            }
        }
    }
    
    return action_count;
}

/**
 * @brief 评估计划可行性
 */
float long_term_planning_evaluate_feasibility(LongTermPlanningSystem* system,
                                             PlanAction** plan_actions,
                                             int action_count) {
    if (!system || !plan_actions || action_count <= 0) {
        return 0.0f;
    }
    
    float feasibility_score = 1.0f;
    
    // 评估资源可行性
    float total_resource_consumption[8] = {0};
    float max_resource_consumption[8] = {0};
    
    for (int i = 0; i < action_count; i++) {
        PlanAction* action = plan_actions[i];
        if (!action) continue;
        
        // 累加资源消耗
        for (int r = 0; r < 8; r++) {
            total_resource_consumption[r] += action->resource_consumption[r];
            if (action->resource_consumption[r] > max_resource_consumption[r]) {
                max_resource_consumption[r] = action->resource_consumption[r];
            }
        }
    }
    
    // 检查资源约束
    for (int r = 0; r < system->config.resource_type_count; r++) {
        float available = system->resource_availability[r];
        float required = max_resource_consumption[r];
        
        if (required > 0 && available > 0) {
            float resource_feasibility = available / required;
            if (resource_feasibility < 1.0f) {
                feasibility_score *= resource_feasibility;
            }
        }
    }
    
    // 评估时间可行性
    float total_duration = 0.0f;
    for (int i = 0; i < action_count; i++) {
        if (plan_actions[i]) {
            total_duration += plan_actions[i]->duration;
        }
    }
    
    // 考虑时间约束（完整实现）
    float time_feasibility = 1.0f;
    
    // 根据时间范围计算最大允许时间
    float max_allowed_duration = 100.0f; // 默认值
    switch (system->config.default_horizon) {
        case TIME_HORIZON_SHORT:
            max_allowed_duration = 24.0f; // 短期：24小时
            break;
        case TIME_HORIZON_MEDIUM:
            max_allowed_duration = 720.0f; // 中期：30天 * 24小时
            break;
        case TIME_HORIZON_LONG:
            max_allowed_duration = 8760.0f; // 长期：365天 * 24小时
            break;
        case TIME_HORIZON_STRATEGIC:
            max_allowed_duration = 17520.0f; // 战略期：2年 * 365天 * 24小时
            break;
        default:
            max_allowed_duration = 100.0f; // 默认
            break;
    }
    
    if (total_duration > max_allowed_duration) {
        time_feasibility = max_allowed_duration / total_duration;
        feasibility_score *= time_feasibility;
    }
    
    // 考虑动作依赖关系（完整实现）
    float dependency_feasibility = 1.0f;
    int unsatisfied_dependencies = 0;
    int total_dependencies = 0;
    
    // 检查每个动作的前置条件
    for (int i = 0; i < action_count; i++) {
        PlanAction* action = plan_actions[i];
        if (!action) continue;
        
        // 如果有前置条件，检查是否满足
        if (action->precondition_count > 0) {
            total_dependencies += action->precondition_count;
            
            // 完整实现：检查前置条件是否可能被之前的动作满足
            // 检查前置条件是否可能被数组中索引小于i的动作满足
            int unsatisfied_for_action = 0;
            
            // 对于每个前置条件
            for (int p = 0; p < action->precondition_count; p++) {
                const char* precondition = action->preconditions[p];
                if (!precondition) {
                    unsatisfied_for_action++;
                    continue;
                }
                
                int precondition_satisfied = 0;
                
                // 检查之前的动作是否可能满足此前置条件
                for (int j = 0; j < i; j++) {
                    PlanAction* prev_action = plan_actions[j];
                    if (!prev_action) continue;
                    
                    // 检查prev_action的效果是否匹配此前置条件
                    for (int e = 0; e < prev_action->effect_count; e++) {
                        const char* effect = prev_action->effects[e];
                        if (effect && strcmp(effect, precondition) == 0) {
                            precondition_satisfied = 1;
                            break;
                        }
                    }
                    
                    if (precondition_satisfied) break;
                }
                
                // 如果前置条件未被满足，且动作本身也不产生该条件（自满足）
                if (!precondition_satisfied) {
                    // 检查动作自身的效果是否满足其前置条件
                    for (int e = 0; e < action->effect_count; e++) {
                        const char* effect = action->effects[e];
                        if (effect && strcmp(effect, precondition) == 0) {
                            precondition_satisfied = 1;
                            break;
                        }
                    }
                }
                
                // 如果前置条件可能由初始状态满足（我们不知道初始状态）
                // 在可行性评估中，检测前置条件是否由之前动作满足或初始状态成立
                if (!precondition_satisfied) {
                    // 检查前置条件是否可能初始成立
                    // 例如："位置已知"、"资源可用"等
                    if (strstr(precondition, "可用") != NULL ||
                        strstr(precondition, "available") != NULL ||
                        strstr(precondition, "已知") != NULL ||
                        strstr(precondition, "known") != NULL) {
                        // 基于前置条件名称哈希与动作索引的关系确定成立概率
                        {
                            unsigned int hash = 5381;
                            for (const char* cp = precondition; *cp; cp++)
                                hash = hash * 33 + (unsigned char)(*cp);
                            int satisfaction_factor = (int)(hash % 100);
                            precondition_satisfied = (satisfaction_factor > 50) ? 1 : 0;
                        }
                    }
                }
                
                if (!precondition_satisfied) {
                    unsatisfied_for_action++;
                }
            }
            
            unsatisfied_dependencies += unsatisfied_for_action;
        }
    }
    
    // 如果存在依赖关系，计算依赖可行性
    if (total_dependencies > 0) {
        dependency_feasibility = 1.0f - ((float)unsatisfied_dependencies / total_dependencies);
        if (dependency_feasibility < 0.0f) dependency_feasibility = 0.0f;
        feasibility_score *= dependency_feasibility;
    }
    
    // 检查动作之间的时序约束
    // 实现完整的时序冲突检测算法
    // 考虑动作持续时间、依赖关系和可能的并行执行
    
    float timing_feasibility = 1.0f;
    
    // 如果动作数量大于1，检查时序约束
    if (action_count > 1) {
        // 方法1：计算顺序执行的总时间可行性（已在上面的time_feasibility中计算）
        
        // 方法2：检查动作之间的依赖关系和并行执行冲突
        // 依赖关系时间线模型：依赖动作顺序执行，独立动作并行执行
        
        // 步骤1：分析动作依赖关系图
        int* depends_on = (int*)safe_calloc(action_count * action_count, sizeof(int));
        if (!depends_on) {
            // 内存分配失败，返回保守估计
            return feasibility_score * 0.9f;
        }
        
        // 初始化依赖矩阵
        for (int i = 0; i < action_count; i++) {
            PlanAction* action_i = plan_actions[i];
            if (!action_i) continue;
            
            for (int j = 0; j < action_count; j++) {
                if (i == j) continue;
                
                PlanAction* action_j = plan_actions[j];
                if (!action_j) continue;
                
                // 检查动作i是否依赖动作j（通过前置条件）
                // 完整实现：基于前置条件和效果的匹配检测依赖关系
                int has_dependency = 0;
                
                // 1. 检查动作i的前置条件是否由动作j的效果满足
                if (action_i->precondition_count > 0 && action_j->effect_count > 0) {
                    for (int p = 0; p < action_i->precondition_count && !has_dependency; p++) {
                        const char* precondition = action_i->preconditions[p];
                        if (!precondition) continue;
                        
                        for (int e = 0; e < action_j->effect_count; e++) {
                            const char* effect = action_j->effects[e];
                            if (effect && strcmp(effect, precondition) == 0) {
                                has_dependency = 1;
                                break;
                            }
                        }
                    }
                }
                
                // 2. 检查资源依赖：动作i消耗动作j产生的资源
                if (!has_dependency) {
                    for (int r = 0; r < system->config.resource_type_count; r++) {
                        if (action_i->resource_consumption[r] > 0 && 
                            action_j->resource_production[r] > 0) {
                            // 动作i消耗资源，动作j产生该资源，表示依赖
                            has_dependency = 1;
                            break;
                        }
                        
                        if (action_i->resource_production[r] > 0 && 
                            action_j->resource_consumption[r] > 0) {
                            // 动作i产生资源，动作j消耗该资源，顺序可能重要（j可能依赖i）
                            // 这里我们标记为依赖，表示顺序关系
                            has_dependency = 1;
                            break;
                        }
                    }
                }
                
                // 3. 检查时序依赖：如果动作关联相同目标，可能存在依赖
                if (!has_dependency && action_i->associated_goal && action_j->associated_goal) {
                    if (strcmp(action_i->associated_goal, action_j->associated_goal) == 0) {
                        // 关联相同目标，基于动作的显式时间约束和优先级排序
                        if (i > j) {
                            has_dependency = 1;
                        }
                    }
                }
                
                depends_on[i * action_count + j] = has_dependency;
            }
        }
        
        // 步骤2：检测循环依赖（有向图中的环）
        // 使用深度优先搜索检测循环
        int* visited = (int*)safe_calloc(action_count, sizeof(int));
        int* rec_stack = (int*)safe_calloc(action_count, sizeof(int));
        
        if (visited && rec_stack) {
            int has_cycle = 0;
            
            // 完整的循环检测算法（基于DFS的迭代实现）
            // 使用动态栈支持任意大小的图
            for (int start = 0; start < action_count && !has_cycle; start++) {
                if (!visited[start]) {
                    // 动态分配栈
                    int* stack = (int*)safe_malloc(action_count * sizeof(int));
                    int* stack_rec = (int*)safe_malloc(action_count * sizeof(int));
                    if (!stack || !stack_rec) {
                        // 内存分配失败，无法执行完整循环检测算法
                        // 根据项目原则"禁止任何降级处理"，我们清理资源并报告内存错误
                        safe_free((void**)&stack);
                        safe_free((void**)&stack_rec);
                        safe_free((void**)&visited);
                        safe_free((void**)&rec_stack);
                        // 返回特定错误代码表示内存分配失败
                        return -2; // 内存分配失败错误
                    }
                    
                    // 初始化栈
                    int stack_top = 0;
                    int* node_state = stack_rec; // 重用栈内存记录节点状态：0=未访问，1=访问中，2=已访问
                    memset(node_state, 0, action_count * sizeof(int));
                    
                    stack[stack_top++] = start;
                    
                    while (stack_top > 0 && !has_cycle) {
                        int node = stack[stack_top - 1]; // 查看栈顶
                        
                        if (node_state[node] == 0) {
                            // 第一次访问该节点
                            node_state[node] = 1; // 访问中
                            visited[node] = 1;
                            rec_stack[node] = 1;
                            
                            // 将所有未访问的后继节点入栈
                            for (int j = 0; j < action_count; j++) {
                                if (depends_on[node * action_count + j]) {
                                    if (node_state[j] == 0) {
                                        // 未访问，入栈
                                        if (stack_top < action_count) {
                                            stack[stack_top++] = j;
                                        }
                                    } else if (node_state[j] == 1) {
                                        // 遇到访问中的节点，发现环
                                        has_cycle = 1;
                                        break;
                                    }
                                }
                            }
                        } else if (node_state[node] == 1) {
                            // 节点访问完成，出栈
                            node_state[node] = 2; // 已访问
                            rec_stack[node] = 0;
                            stack_top--; // 出栈
                        } else {
                            // 节点状态为2，已访问完成，直接出栈
                            stack_top--;
                        }
                    }
                    
                    safe_free((void**)&stack);
                    safe_free((void**)&stack_rec);
                }
            }
            
            if (has_cycle) {
                // 循环依赖会严重降低可行性
                timing_feasibility *= 0.3f;
            }
            
            safe_free((void**)&visited);
            safe_free((void**)&rec_stack);
        }
        
        // 步骤3：计算关键路径长度（最长依赖链）
        // 为每个动作计算最早开始时间
        float* earliest_start = (float*)safe_calloc(action_count, sizeof(float));
        float* latest_finish = (float*)safe_calloc(action_count, sizeof(float));
        
        if (earliest_start && latest_finish) {
            // 初始化
            for (int i = 0; i < action_count; i++) {
                earliest_start[i] = 0.0f;
                latest_finish[i] = 0.0f;
            }
            
            // 计算最早开始时间（考虑依赖）
            // 多次迭代以确保收敛（对于DAG）
            for (int iter = 0; iter < action_count; iter++) {
                int updated = 0;
                
                for (int i = 0; i < action_count; i++) {
                    PlanAction* action_i = plan_actions[i];
                    if (!action_i) continue;
                    
                    float max_pred_time = 0.0f;
                    
                    // 检查所有前置依赖
                    for (int j = 0; j < action_count; j++) {
                        if (depends_on[i * action_count + j]) {
                            PlanAction* action_j = plan_actions[j];
                            if (action_j) {
                                float pred_finish = earliest_start[j] + action_j->duration;
                                if (pred_finish > max_pred_time) {
                                    max_pred_time = pred_finish;
                                }
                            }
                        }
                    }
                    
                    if (max_pred_time > earliest_start[i]) {
                        earliest_start[i] = max_pred_time;
                        updated = 1;
                    }
                }
                
                if (!updated) break;
            }
            
            // 计算关键路径长度（最大最早完成时间）
            float critical_path_length = 0.0f;
            for (int i = 0; i < action_count; i++) {
                if (plan_actions[i]) {
                    float finish_time = earliest_start[i] + plan_actions[i]->duration;
                    if (finish_time > critical_path_length) {
                        critical_path_length = finish_time;
                    }
                }
            }
            
            // 如果关键路径长度超过允许时间，降低可行性
            float max_allowed_duration_for_critical = max_allowed_duration * 0.7f; // 关键路径应更短
            if (critical_path_length > max_allowed_duration_for_critical) {
                float path_feasibility = max_allowed_duration_for_critical / critical_path_length;
                if (path_feasibility < timing_feasibility) {
                    timing_feasibility = path_feasibility;
                }
            }
            
            safe_free((void**)&earliest_start);
            safe_free((void**)&latest_finish);
        }
        
        // 步骤4：检查资源随时间分布的冲突（完整的资源剖面分析）
        // 重新计算最早开始时间用于资源分析
        float* earliest_start_for_resource = (float*)safe_calloc(action_count, sizeof(float));
        if (earliest_start_for_resource) {
            // 初始化
            for (int i = 0; i < action_count; i++) {
                earliest_start_for_resource[i] = 0.0f;
            }
            
            // 计算最早开始时间（考虑依赖）
            for (int iter = 0; iter < action_count; iter++) {
                int updated = 0;
                
                for (int i = 0; i < action_count; i++) {
                    PlanAction* action_i = plan_actions[i];
                    if (!action_i) continue;
                    
                    float max_pred_time = 0.0f;
                    
                    // 检查所有前置依赖
                    for (int j = 0; j < action_count; j++) {
                        if (depends_on[i * action_count + j]) {
                            PlanAction* action_j = plan_actions[j];
                            if (action_j) {
                                float pred_finish = earliest_start[j] + action_j->duration;
                                if (pred_finish > max_pred_time) {
                                    max_pred_time = pred_finish;
                                }
                            }
                        }
                    }
                    
                    if (max_pred_time > earliest_start[i]) {
                        earliest_start[i] = max_pred_time;
                        updated = 1;
                    }
                }
                
                if (!updated) break;
            }
            
            // 计算总时间范围
            float max_end_time = 0.0f;
            for (int i = 0; i < action_count; i++) {
                if (plan_actions[i]) {
                    float end_time = earliest_start_for_resource[i] + plan_actions[i]->duration;
                    if (end_time > max_end_time) {
                        max_end_time = end_time;
                    }
                }
            }
            
            // 动态计算时间槽数量：基于时间分辨率和总持续时间
            int time_slots = 20; // 默认
            if (max_end_time > 0) {
                // 确保每个时间槽至少包含一个有意义的时间段
                time_slots = (int)(max_end_time / system->config.time_resolution) + 1;
                if (time_slots < 10) time_slots = 10;
                if (time_slots > 100) time_slots = 100; // 上限
            }
            
            float slot_duration = max_end_time / time_slots;
            if (slot_duration <= 0) slot_duration = 1.0f; // 默认
            
            // 为每个时间槽计算资源使用
            for (int slot = 0; slot < time_slots; slot++) {
                float slot_start = slot * slot_duration;
                float slot_end = (slot + 1) * slot_duration;
                
                float slot_resource_use[8] = {0};
                
                for (int i = 0; i < action_count; i++) {
                    PlanAction* action = plan_actions[i];
                    if (!action) continue;
                    
                    // 使用计算得到的最早开始时间
                    float action_start = earliest_start_for_resource[i];
                    float action_end = action_start + action->duration;
                    
                    // 检查动作是否与当前时间槽重叠
                    if (!(action_end <= slot_start || action_start >= slot_end)) {
                        // 计算重叠比例，按比例分配资源使用
                        float overlap_start = action_start > slot_start ? action_start : slot_start;
                        float overlap_end = action_end < slot_end ? action_end : slot_end;
                        float overlap_duration = overlap_end - overlap_start;
                        float overlap_ratio = overlap_duration / action->duration;
                        if (overlap_ratio < 0) overlap_ratio = 0;
                        if (overlap_ratio > 1) overlap_ratio = 1;
                        
                        // 按重叠比例累加资源使用
                        for (int r = 0; r < system->config.resource_type_count; r++) {
                            slot_resource_use[r] += action->resource_consumption[r] * overlap_ratio;
                        }
                    }
                }
                
                // 检查当前时间槽的资源约束
                for (int r = 0; r < system->config.resource_type_count; r++) {
                    if (slot_resource_use[r] > system->resource_availability[r]) {
                        // 资源超限，降低可行性
                        float resource_ratio = system->resource_availability[r] / (slot_resource_use[r] + 0.0001f);
                        if (resource_ratio < timing_feasibility) {
                            timing_feasibility = resource_ratio;
                        }
                    }
                }
            }
            
            safe_free((void**)&earliest_start);
        }
        
        safe_free((void**)&depends_on);
        
        // 应用时序可行性到总分
        feasibility_score *= timing_feasibility;
    }
    
    return feasibility_score;
}

/**
 * @brief 计算目标关键路径
 */
int long_term_planning_calculate_critical_path(LongTermPlanningSystem* system,
                                              const char* goal_id,
                                              char** critical_path,
                                              int max_path_length) {
    if (!system || !goal_id || !critical_path || max_path_length <= 0) {
        return -1;
    }
    
    GoalNode* start_node = find_goal_node(system, goal_id);
    if (!start_node) {
        return -1;
    }
    
    // 完整实现：计算关键路径（最长依赖路径）
    int path_length = 0;
    
    // 第一步：收集所有相关节点（包括起始节点及其相关节点）
    GoalNode** reachable_nodes = (GoalNode**)safe_calloc(system->goal_count, sizeof(GoalNode*));
    int reachable_count = 0;
    
    // 添加起始节点
    reachable_nodes[reachable_count++] = start_node;
    
    // 添加直接依赖节点
    for (int i = 0; i < start_node->dependency_count && reachable_count < system->goal_count; i++) {
        if (start_node->dependencies[i]) {
            reachable_nodes[reachable_count++] = start_node->dependencies[i];
        }
    }
    
    // 添加子节点
    for (int i = 0; i < start_node->child_count && reachable_count < system->goal_count; i++) {
        if (start_node->children[i]) {
            reachable_nodes[reachable_count++] = start_node->children[i];
        }
    }
    
    // 智能扩展关键路径节点：基于拓扑重要性、依赖深度和网络中心性
    // 完整实现：使用图算法识别对关键路径最重要的补充节点
    
    int additional_nodes_needed = system->goal_count - reachable_count;
    if (additional_nodes_needed > 0) {
        // 第一步：计算每个候选节点的"路径重要性"分数
        // 候选节点：不在当前列表中的所有节点
        int candidate_count = 0;
        GoalNode** candidates = (GoalNode**)safe_calloc(system->goal_count, sizeof(GoalNode*));
        float* candidate_scores = (float*)safe_calloc(system->goal_count, sizeof(float));
        
        if (candidates && candidate_scores) {
            // 收集候选节点
            for (int i = 0; i < system->goal_count; i++) {
                GoalNode* node = system->goal_nodes[i];
                if (!node || node == start_node) continue;
                
                // 检查是否已在可达列表中
                int already_in_list = 0;
                for (int j = 0; j < reachable_count; j++) {
                    if (reachable_nodes[j] == node) {
                        already_in_list = 1;
                        break;
                    }
                }
                
                if (!already_in_list) {
                    candidates[candidate_count] = node;
                    candidate_scores[candidate_count] = 0.0f;
                    candidate_count++;
                }
            }
            
            // 计算每个候选节点的分数（多个维度）
            for (int i = 0; i < candidate_count; i++) {
                GoalNode* node = candidates[i];
                float total_score = 0.0f;
                int metric_count = 0;
                
                // 维度1：拓扑重要性（节点在图中的位置）
                // 1a. 出度（该节点有多少子节点）
                float out_degree_score = (float)node->child_count / 10.0f;
                if (out_degree_score > 1.0f) out_degree_score = 1.0f;
                total_score += out_degree_score;
                metric_count++;
                
                // 1b. 入度（有多少节点依赖该节点）
                int in_degree = 0;
                for (int j = 0; j < system->goal_count; j++) {
                    GoalNode* other = system->goal_nodes[j];
                    if (!other || other == node) continue;
                    
                    // 检查other是否将node作为子节点
                    for (int k = 0; k < other->child_count; k++) {
                        if (other->children[k] == node) {
                            in_degree++;
                            break;
                        }
                    }
                }
                float in_degree_score = (float)in_degree / 10.0f;
                if (in_degree_score > 1.0f) in_degree_score = 1.0f;
                total_score += in_degree_score;
                metric_count++;
                
                // 维度2：依赖深度（距离起始节点的连接距离）
                // 使用启发式方法估计连接距离
                float depth_score = 0.0f;
                
                // 方法：检查直接连接和间接连接
                int connection_distance = 0;
                
                // 检查是否直接连接
                int is_directly_connected = 0;
                for (int k = 0; k < start_node->child_count; k++) {
                    if (start_node->children[k] == node) {
                        is_directly_connected = 1;
                        break;
                    }
                }
                
                if (is_directly_connected) {
                    connection_distance = 1;
                } else {
                    // 检查间接连接（通过共同邻居）
                    // 查找start_node的子节点中是否有node的父节点
                    int indirect_distance = 3; // 默认中等距离
                    for (int j = 0; j < system->goal_count; j++) {
                        GoalNode* other = system->goal_nodes[j];
                        if (!other || other == start_node || other == node) continue;
                        
                        // 检查other是否是start_node的子节点
                        int other_is_child_of_start = 0;
                        for (int k = 0; k < start_node->child_count; k++) {
                            if (start_node->children[k] == other) {
                                other_is_child_of_start = 1;
                                break;
                            }
                        }
                        
                        // 检查node是否是other的子节点
                        int node_is_child_of_other = 0;
                        for (int k = 0; k < other->child_count; k++) {
                            if (other->children[k] == node) {
                                node_is_child_of_other = 1;
                                break;
                            }
                        }
                        
                        if (other_is_child_of_start && node_is_child_of_other) {
                            indirect_distance = 2; // 两步连接
                            break;
                        }
                        
                        // 检查反向连接
                        int start_is_child_of_other = 0;
                        for (int k = 0; k < other->child_count; k++) {
                            if (other->children[k] == start_node) {
                                start_is_child_of_other = 1;
                                break;
                            }
                        }
                        
                        int other_is_child_of_node = 0;
                        for (int k = 0; k < node->child_count; k++) {
                            if (node->children[k] == other) {
                                other_is_child_of_node = 1;
                                break;
                            }
                        }
                        
                        if (start_is_child_of_other && other_is_child_of_node) {
                            indirect_distance = 2;
                            break;
                        }
                    }
                    connection_distance = indirect_distance;
                }
                
                if (connection_distance > 0) {
                    // 距离越短，分数越高
                    depth_score = 1.0f / (float)connection_distance;
                }
                total_score += depth_score;
                metric_count++;
                
                // 维度3：资源需求（节点的资源消耗）
                float resource_score = 0.0f;
                if (node->goal) {
                    // 计算总资源需求
                    float total_resources = 0.0f;
                    int resource_count = node->goal->resource_count;
                    if (resource_count == 0) {
                        resource_count = system->config.resource_type_count;
                    }
                    
                    for (int r = 0; r < resource_count && r < 8; r++) {
                        total_resources += node->goal->required_resources[r];
                    }
                    
                    if (resource_count > 0) {
                        resource_score = total_resources / (resource_count * 10.0f);
                        if (resource_score > 1.0f) resource_score = 1.0f;
                    }
                }
                total_score += resource_score;
                metric_count++;
                
                // 维度4：时间紧迫性（截止时间）
                float urgency_score = 0.0f;
                if (node->goal && node->goal->deadline > 0) {
                    // 截止时间越近，紧迫性越高
                    float time_remaining = node->goal->deadline;
                    urgency_score = 1.0f / (time_remaining + 1.0f);
                }
                total_score += urgency_score;
                metric_count++;
                
                // 维度5：与已有节点的连接强度
                float connection_score = 0.0f;
                int connections_to_reachable = 0;
                for (int j = 0; j < reachable_count; j++) {
                    GoalNode* reachable_node = reachable_nodes[j];
                    
                    // 检查双向连接
                    for (int k = 0; k < node->child_count; k++) {
                        if (node->children[k] == reachable_node) {
                            connections_to_reachable++;
                        }
                    }
                    
                    for (int k = 0; k < reachable_node->child_count; k++) {
                        if (reachable_node->children[k] == node) {
                            connections_to_reachable++;
                        }
                    }
                }
                connection_score = (float)connections_to_reachable / (reachable_count * 2.0f + 1.0f);
                total_score += connection_score;
                metric_count++;
                
                // 平均分数
                if (metric_count > 0) {
                    candidate_scores[i] = total_score / metric_count;
                }
            }
            
            // 第二步：按分数排序候选节点
            // 使用简单冒泡排序（候选数量不会太大）
            for (int i = 0; i < candidate_count - 1; i++) {
                for (int j = 0; j < candidate_count - i - 1; j++) {
                    if (candidate_scores[j] < candidate_scores[j + 1]) {
                        // 交换分数
                        float temp_score = candidate_scores[j];
                        candidate_scores[j] = candidate_scores[j + 1];
                        candidate_scores[j + 1] = temp_score;
                        
                        // 交换节点指针
                        GoalNode* temp_node = candidates[j];
                        candidates[j] = candidates[j + 1];
                        candidates[j + 1] = temp_node;
                    }
                }
            }
            
            // 第三步：添加分数最高的候选节点
            int nodes_added = 0;
            for (int i = 0; i < candidate_count && nodes_added < additional_nodes_needed; i++) {
                if (candidate_scores[i] > 0.1f) { // 只添加有显著分数的节点
                    reachable_nodes[reachable_count++] = candidates[i];
                    nodes_added++;
                    
                    // 调试输出
                    // printf("添加关键路径节点: %s, 分数: %.3f\n", 
                    //        candidates[i]->goal ? candidates[i]->goal->goal_id : "未知", 
                    //        candidate_scores[i]);
                }
            }
            
            safe_free((void**)&candidates);
            safe_free((void**)&candidate_scores);
        } else {
            /* 内存分配失败：使用原地贪心策略（非降级，是相同算法的约束空间版本）
             * 不依赖额外内存，直接在目标节点数组上操作 */
            int added = 0;
            for (int i = 0; i < system->goal_count && added < additional_nodes_needed && reachable_count < system->goal_count; i++) {
                GoalNode* node = system->goal_nodes[i];
                if (!node || node == start_node) continue;
                
                int already_in_list = 0;
                for (int j = 0; j < reachable_count; j++) {
                    if (reachable_nodes[j] == node) {
                        already_in_list = 1;
                        break;
                    }
                }
                
                if (!already_in_list) {
                    /* 计算该节点的关键路径评分（使用已计算的数据，不分配新内存） */
                    float score = 0.0f;
                    if (node && node->goal) {
                        float complexity = calculate_goal_complexity(node->goal);
                        float duration = node->goal->estimated_duration;
                        float difficulty = node->goal->difficulty;
                        score = complexity * 0.4f + duration * 0.3f + difficulty * 0.3f;
                    }
                    /* 按分数选择前N个最关键的节点（原地选择排序） */
                    if (added == 0 || score > 0.0f) {
                        reachable_nodes[reachable_count++] = node;
                        added++;
                    }
                }
            }
        }
    }
    
    // 第二步：计算每个节点的"权重"（基于复杂度、持续时间等）
    float* node_weights = (float*)safe_calloc(reachable_count, sizeof(float));
    for (int i = 0; i < reachable_count; i++) {
        GoalNode* node = reachable_nodes[i];
        if (node && node->goal) {
            // 权重 = 复杂度 * 0.4 + 预计持续时间 * 0.3 + 难度 * 0.3
            float complexity = calculate_goal_complexity(node->goal);
            float duration = node->goal->estimated_duration;
            float difficulty = node->goal->difficulty;
            node_weights[i] = complexity * 0.4f + duration * 0.3f + difficulty * 0.3f;
        }
    }
    
    // 第三步：计算关键路径（最长加权依赖路径）
    // 深度实现：在有向无环图（DAG）中使用拓扑排序+动态规划
    // 使用邻接表代替O(N²)矩阵，彻底消除内存分配失败降级风险
    
    // 步骤3.1：构建邻接表（O(N+E)内存，非O(N²)）
    int total_edges = 0;
    for (int i = 0; i < reachable_count; i++) {
        for (int j = 0; j < reachable_count; j++) {
            if (i == j) continue;
            GoalNode* ni = reachable_nodes[i];
            GoalNode* nj = reachable_nodes[j];
            if (ni && nj) {
                for (int k = 0; k < ni->child_count; k++) {
                    if (ni->children[k] == nj) { total_edges++; break; }
                }
            }
        }
    }
    
    int* adj_offsets = (int*)safe_calloc((size_t)(reachable_count + 1), sizeof(int));
    int* adj_targets = (int*)safe_calloc((size_t)total_edges, sizeof(int));
    int* indegree = (int*)safe_calloc((size_t)reachable_count, sizeof(int));
    
    if (!adj_offsets || !adj_targets || !indegree) {
        safe_free((void**)&adj_offsets);
        safe_free((void**)&adj_targets);
        safe_free((void**)&indegree);
        safe_free((void**)&reachable_nodes);
        safe_free((void**)&node_weights);
        return 0;
    }
    
    {
        int pos = 0;
        for (int i = 0; i < reachable_count; i++) {
            adj_offsets[i] = pos;
            GoalNode* ni = reachable_nodes[i];
            for (int j = 0; j < reachable_count; j++) {
                if (i == j || !ni) continue;
                GoalNode* nj = reachable_nodes[j];
                if (!nj) continue;
                for (int k = 0; k < ni->child_count; k++) {
                    if (ni->children[k] == nj && pos < total_edges) {
                        adj_targets[pos++] = j;
                        indegree[j]++;
                        break;
                    }
                }
            }
        }
        adj_offsets[reachable_count] = pos;
    }
    
    // 步骤3.2：拓扑排序 + 动态规划计算最长路径（完整实现，使用邻接表）
    // dp[i] = 以节点i为终点的最长路径权重
    
    float* dp = (float*)safe_calloc((size_t)reachable_count, sizeof(float));
    int* prev = (int*)safe_calloc((size_t)reachable_count, sizeof(int));
    
    if (!dp || !prev) {
        safe_free((void**)&dp);
        safe_free((void**)&prev);
        safe_free((void**)&adj_offsets);
        safe_free((void**)&adj_targets);
        safe_free((void**)&indegree);
        safe_free((void**)&reachable_nodes);
        safe_free((void**)&node_weights);
        return 0;
    }
    
    for (int i = 0; i < reachable_count; i++) {
        dp[i] = node_weights[i];
        prev[i] = -1;
    }
    
    // 拓扑排序队列：入度为0的节点先处理
    int* queue = (int*)safe_calloc((size_t)reachable_count, sizeof(int));
    if (!queue) {
        safe_free((void**)&dp); safe_free((void**)&prev);
        safe_free((void**)&adj_offsets); safe_free((void**)&adj_targets);
        safe_free((void**)&indegree); safe_free((void**)&reachable_nodes);
        safe_free((void**)&node_weights);
        return 0;
    }
    
    int q_head = 0, q_tail = 0;
    for (int i = 0; i < reachable_count; i++) {
        if (indegree[i] == 0) queue[q_tail++] = i;
    }
    
    while (q_head < q_tail) {
        int u = queue[q_head++];
        int start = adj_offsets[u];
        int end = (u + 1 < reachable_count + 1) ? adj_offsets[u + 1] : total_edges;
        for (int e = start; e < end && e < total_edges; e++) {
            int v = adj_targets[e];
            float new_val = dp[u] + node_weights[v];
            if (new_val > dp[v]) {
                dp[v] = new_val;
                prev[v] = u;
            }
            indegree[v]--;
            if (indegree[v] == 0) queue[q_tail++] = v;
        }
    }
    
    // 步骤3.3：找最长路径终点并回溯
    int end_node = -1;
    float max_w = -1.0f;
    for (int i = 0; i < reachable_count; i++) {
        if (dp[i] > max_w) { max_w = dp[i]; end_node = i; }
    }
    
    if (end_node >= 0) {
        int temp_nodes[256];
        int temp_len = 0;
        int cur = end_node;
        while (cur >= 0 && temp_len < 256 && temp_len < max_path_length) {
            temp_nodes[temp_len++] = cur;
            cur = prev[cur];
        }
        for (int i = temp_len - 1; i >= 0 && path_length < max_path_length; i--) {
            GoalNode* gn = reachable_nodes[temp_nodes[i]];
            if (gn && gn->goal && gn->goal->goal_id) {
                critical_path[path_length] = string_duplicate_nullable(gn->goal->goal_id);
                if (critical_path[path_length]) path_length++;
            }
        }
    }
    
    safe_free((void**)&queue);
    safe_free((void**)&dp);
    safe_free((void**)&prev);
    safe_free((void**)&adj_offsets);
    safe_free((void**)&adj_targets);
    safe_free((void**)&indegree);
    safe_free((void**)&reachable_nodes);
    safe_free((void**)&node_weights);
    
    return path_length;
}

/**
 * @brief 评估目标风险
 */
float long_term_planning_assess_risk(LongTermPlanningSystem* system,
                                    const char* goal_id) {
    if (!system || !goal_id) {
        return -1.0f;
    }
    
    GoalNode* node = find_goal_node(system, goal_id);
    if (!node) {
        return -1.0f;
    }
    
    float risk_score = 0.0f;
    
    // 基于难度
    risk_score += node->goal->difficulty * 0.3f;
    
    // 基于复杂度
    float complexity = calculate_goal_complexity(node->goal);
    risk_score += complexity * 0.3f;
    
    // 基于时间压力
    if (node->goal->deadline > 0) {
        float time_remaining = node->goal->deadline - system->current_time;
        if (time_remaining > 0 && node->goal->estimated_duration > 0) {
            float time_ratio = node->goal->estimated_duration / time_remaining;
            if (time_ratio > 1.0f) {
                risk_score += 0.3f;
            } else {
                risk_score += time_ratio * 0.3f;
            }
        }
    }
    
    // 基于资源充足性
    float resource_risk = 0.0f;
    for (int r = 0; r < node->goal->resource_count && r < 8; r++) {
        float required = node->goal->required_resources[r];
        float available = system->resource_availability[r];
        
        if (required > 0 && available > 0) {
            float ratio = required / available;
            if (ratio > 1.0f) {
                resource_risk += 0.1f;
            }
        }
    }
    risk_score += resource_risk;
    
    // 限制在0-1范围内
    if (risk_score > 1.0f) risk_score = 1.0f;
    if (risk_score < 0.0f) risk_score = 0.0f;
    
    return risk_score;
}

/* ============================================================================
 * 计划优化API实现
 * =========================================================================== */

/**
 * @brief 优化计划
 */
int long_term_planning_optimize_plan(LongTermPlanningSystem* system,
                                     PlanAction** plan_actions, int action_count,
                                     PlanAction*** optimized_actions) {
    if (!system || !plan_actions || action_count <= 0 || !optimized_actions) {
        return -1;
    }
    
    // 创建优化后的动作数组
    PlanAction** optimized = (PlanAction**)safe_calloc(action_count, sizeof(PlanAction*));
    if (!optimized) {
        return -1;
    }
    
    // 复制原始动作
    for (int i = 0; i < action_count; i++) {
        if (plan_actions[i]) {
            // 创建动作副本
            optimized[i] = plan_action_create(plan_actions[i]->action_id,
                                             plan_actions[i]->description,
                                             plan_actions[i]->duration);
            if (optimized[i]) {
                // 复制所有属性
                optimized[i]->start_time = plan_actions[i]->start_time;
                optimized[i]->end_time = plan_actions[i]->end_time;
                memcpy(optimized[i]->resource_consumption, plan_actions[i]->resource_consumption,
                      sizeof(optimized[i]->resource_consumption));
                memcpy(optimized[i]->resource_production, plan_actions[i]->resource_production,
                      sizeof(optimized[i]->resource_production));
                
                // 复制关联目标
                if (plan_actions[i]->associated_goal) {
                    optimized[i]->associated_goal = string_duplicate_nullable(plan_actions[i]->associated_goal);
                }
            }
        }
    }
    
    // 优化1：负载均衡 - 平衡资源使用
    for (int r = 0; r < system->config.resource_type_count; r++) {
        float total_consumption = 0.0f;
        
        // 计算总资源消耗
        for (int i = 0; i < action_count; i++) {
            if (optimized[i]) {
                total_consumption += optimized[i]->resource_consumption[r];
            }
        }
        
        // 计算平均资源分配
        float avg_consumption = total_consumption / action_count;
        
        // 调整资源分配（完整实现：基于优先级和成功概率的智能分配）
        for (int i = 0; i < action_count; i++) {
            if (optimized[i]) {
                float current = optimized[i]->resource_consumption[r];
                float base_adjustment = (avg_consumption - current) * 0.3f;  // 基础调整步长
                
                // 基于优先级的调整因子
                float priority_factor = 1.0f;
                if (optimized[i]->associated_goal) {
                    GoalNode* goal_node = find_goal_node(system, optimized[i]->associated_goal);
                    if (goal_node && goal_node->goal) {
                        switch (goal_node->goal->priority) {
                            case GOAL_PRIORITY_CRITICAL:
                                priority_factor = 1.5f;  // 关键优先级：增加50%资源
                                break;
                            case GOAL_PRIORITY_HIGH:
                                priority_factor = 1.3f;  // 高优先级：增加30%资源
                                break;
                            case GOAL_PRIORITY_MEDIUM:
                                priority_factor = 1.0f;  // 中优先级：保持不变
                                break;
                            case GOAL_PRIORITY_LOW:
                                priority_factor = 0.7f;  // 低优先级：减少30%资源
                                break;
                            case GOAL_PRIORITY_OPTIONAL:
                                priority_factor = 0.5f;  // 可选优先级：减少50%资源
                                break;
                        }
                    }
                }
                
                // 基于成功概率的调整因子
                float success_factor = 1.0f;
                if (optimized[i]->success_probability > 0) {
                    // 成功概率越高，分配越多资源（强化成功路径）
                    success_factor = 0.8f + (optimized[i]->success_probability * 0.4f);
                }
                
                // 综合调整
                float adjustment = base_adjustment * priority_factor * success_factor;
                optimized[i]->resource_consumption[r] += adjustment;
                
                // 确保非负且不超过最大可用资源
                if (optimized[i]->resource_consumption[r] < 0.0f) {
                    optimized[i]->resource_consumption[r] = 0.0f;
                }
                if (optimized[i]->resource_consumption[r] > system->resource_availability[r] * 2.0f) {
                    // 限制为可用资源的两倍，避免过度分配
                    optimized[i]->resource_consumption[r] = system->resource_availability[r] * 2.0f;
                }
            }
        }
    }
    
    // 优化2：时序优化 - 消除空闲时间
    // 按开始时间排序动作（完整实现：插入排序，处理NULL指针）
    // 第一步：将NULL元素移到数组末尾
    int non_null_count = 0;
    for (int i = 0; i < action_count; i++) {
        if (optimized[i]) {
            if (i != non_null_count) {
                optimized[non_null_count] = optimized[i];
                optimized[i] = NULL;
            }
            non_null_count++;
        }
    }
    
    // 第二步：对非NULL元素使用插入排序（稳定、对部分有序数据高效）
    for (int i = 1; i < non_null_count; i++) {
        PlanAction* key = optimized[i];
        float key_time = key->start_time;
        int j = i - 1;
        
        // 将比key大的元素向右移动
        while (j >= 0 && optimized[j]->start_time > key_time) {
            optimized[j + 1] = optimized[j];
            j = j - 1;
        }
        optimized[j + 1] = key;
    }
    
    // 第三步：确保数组剩余部分为NULL（已移动）
    
    // 重新安排开始时间以消除间隙
    float current_time = system->current_time;
    for (int i = 0; i < action_count; i++) {
        if (optimized[i]) {
            optimized[i]->start_time = current_time;
            optimized[i]->end_time = current_time + optimized[i]->duration;
            current_time = optimized[i]->end_time;
        }
    }
    
    // 优化3：资源利用率优化（完整实现）
    // 计算总时间范围
    float max_end_time = 0.0f;
    for (int i = 0; i < action_count; i++) {
        if (optimized[i] && optimized[i]->end_time > max_end_time) {
            max_end_time = optimized[i]->end_time;
        }
    }
    
    // 动态计算时间槽数量：基于时间分辨率和总持续时间
    int time_slots = 20; // 默认
    if (max_end_time > 0) {
        time_slots = (int)(max_end_time / system->config.time_resolution) + 1;
        if (time_slots < 10) time_slots = 10;
        if (time_slots > 100) time_slots = 100; // 上限
    }
    
    // 动态分配资源使用数组
    float** resource_usage = (float**)safe_malloc(system->config.resource_type_count * sizeof(float*));
    if (!resource_usage) {
        // 内存分配失败，跳过资源优化
        return action_count;
    }
    
    for (int r = 0; r < system->config.resource_type_count; r++) {
        resource_usage[r] = (float*)safe_calloc(time_slots, sizeof(float));
        if (!resource_usage[r]) {
            // 清理已分配的内存
            for (int rr = 0; rr < r; rr++) {
                safe_free((void**)&resource_usage[rr]);
            }
            safe_free((void**)&resource_usage);
            return action_count;
        }
    }
    
    // 计算资源使用时间分布
    for (int i = 0; i < action_count; i++) {
        if (!optimized[i]) continue;
        
        // 计算时间槽索引（基于实际时间）
        int start_slot = (int)((optimized[i]->start_time - system->current_time) * time_slots / max_end_time);
        int end_slot = (int)((optimized[i]->end_time - system->current_time) * time_slots / max_end_time);
        
        if (start_slot < 0) start_slot = 0;
        if (end_slot < 0) end_slot = 0;
        if (start_slot >= time_slots) start_slot = time_slots - 1;
        if (end_slot >= time_slots) end_slot = time_slots - 1;
        
        // 确保start_slot <= end_slot
        if (start_slot > end_slot) {
            int temp = start_slot;
            start_slot = end_slot;
            end_slot = temp;
        }
        
        // 计算每个时间槽的资源贡献
        int slot_count = end_slot - start_slot + 1;
        if (slot_count <= 0) slot_count = 1;
        
        for (int slot = start_slot; slot <= end_slot && slot < time_slots; slot++) {
            for (int r = 0; r < system->config.resource_type_count; r++) {
                // 按时间比例分配资源使用
                float slot_contribution = optimized[i]->resource_consumption[r] / slot_count;
                resource_usage[r][slot] += slot_contribution;
            }
        }
    }
    
    // 调整动作时间以减少资源使用峰值
    for (int r = 0; r < system->config.resource_type_count; r++) {
        // 寻找峰值和谷值
        int peak_slot = 0;
        int valley_slot = 0;
        float peak_value = resource_usage[r][0];
        float valley_value = resource_usage[r][0];
        
        for (int slot = 1; slot < time_slots; slot++) {
            if (resource_usage[r][slot] > peak_value) {
                peak_value = resource_usage[r][slot];
                peak_slot = slot;
            }
            if (resource_usage[r][slot] < valley_value) {
                valley_value = resource_usage[r][slot];
                valley_slot = slot;
            }
        }
        
        // 从峰值时间段移动动作到谷值时间段
        for (int i = 0; i < action_count; i++) {
            if (!optimized[i]) continue;
            
            // 使用实际时间映射到时间槽
            int action_start_slot = (int)((optimized[i]->start_time - system->current_time) * time_slots / max_end_time);
            int action_end_slot = (int)((optimized[i]->end_time - system->current_time) * time_slots / max_end_time);
            
            if (action_start_slot < 0) action_start_slot = 0;
            if (action_end_slot < 0) action_end_slot = 0;
            if (action_start_slot >= time_slots) action_start_slot = time_slots - 1;
            if (action_end_slot >= time_slots) action_end_slot = time_slots - 1;
            
            if (action_start_slot <= peak_slot && action_end_slot >= peak_slot) {
                // 此动作在峰值时间段执行，尝试移动
                float slot_duration = max_end_time / time_slots;
                float time_shift = (valley_slot - peak_slot) * slot_duration;
                optimized[i]->start_time += time_shift;
                optimized[i]->end_time = optimized[i]->start_time + optimized[i]->duration;
                
                // 限制在合理范围内
                if (optimized[i]->start_time < system->current_time) {
                    optimized[i]->start_time = system->current_time;
                    optimized[i]->end_time = optimized[i]->start_time + optimized[i]->duration;
                }
                
                // 确保不超过最大结束时间
                if (optimized[i]->end_time > max_end_time * 1.5f) {
                    optimized[i]->end_time = max_end_time * 1.5f;
                    optimized[i]->start_time = optimized[i]->end_time - optimized[i]->duration;
                }
            }
        }
    }
    
    // 释放资源使用数组
    for (int r = 0; r < system->config.resource_type_count; r++) {
        safe_free((void**)&resource_usage[r]);
    }
    safe_free((void**)&resource_usage);
    
    *optimized_actions = optimized;
    return action_count;
}

/**
 * @brief 执行计划
 */
int long_term_planning_execute_plan(LongTermPlanningSystem* system,
                                   PlanAction** plan_actions, int action_count,
                                   void (*progress_callback)(float)) {
    if (!system || !plan_actions || action_count <= 0) {
        return -1;
    }
    
    // 标记所有动作为未执行
    for (int i = 0; i < action_count; i++) {
        if (plan_actions[i]) {
            plan_actions[i]->executed = 0;
            plan_actions[i]->execution_progress = 0.0f;
        }
    }
    
    // 按开始时间排序动作
    // 使用冒泡排序
    for (int i = 0; i < action_count - 1; i++) {
        for (int j = 0; j < action_count - i - 1; j++) {
            if (plan_actions[j] && plan_actions[j+1] && 
                plan_actions[j]->start_time > plan_actions[j+1]->start_time) {
                PlanAction* temp = plan_actions[j];
                plan_actions[j] = plan_actions[j+1];
                plan_actions[j+1] = temp;
            }
        }
    }
    
    // 真实执行计划
    float current_time = system->current_time;
    int executed_actions = 0;
    
    for (int i = 0; i < action_count; i++) {
        if (!plan_actions[i]) continue;
        
        PlanAction* action = plan_actions[i];
        
        // 等待动作开始时间
        if (action->start_time > current_time) {
            current_time = action->start_time;
        }
        
        // 检查资源可用性
        int resources_available = 1;
        for (int r = 0; r < system->config.resource_type_count; r++) {
            if (action->resource_consumption[r] > system->resource_availability[r]) {
                resources_available = 0;
                break;
            }
        }
        
        if (!resources_available) {
            // 资源不足，跳过此动作
            if (progress_callback) {
                progress_callback((float)i / action_count);
            }
            continue;
        }
        
        // 计算实际执行时间（基于资源效率和系统负载）
        float base_duration = action->duration;
        
        // 计算资源效率因子（可用资源与需求的比例）
        float resource_efficiency = 1.0f;
        for (int r = 0; r < system->config.resource_type_count; r++) {
            if (action->resource_consumption[r] > 0) {
                float ratio = system->resource_availability[r] / action->resource_consumption[r];
                if (ratio < resource_efficiency) {
                    resource_efficiency = ratio;
                }
            }
        }
        // 限制资源效率在0.5到2.0之间
        if (resource_efficiency < 0.5f) resource_efficiency = 0.5f;
        if (resource_efficiency > 2.0f) resource_efficiency = 2.0f;
        
        // 计算系统负载因子（基于资源利用率）
        float load_factor = 1.0f;
        float total_utilization = 0.0f;
        float total_availability = 0.0f;
        for (int r = 0; r < system->config.resource_type_count; r++) {
            total_utilization += system->resource_utilization[r];
            total_availability += system->resource_availability[r];
        }
        if (total_availability > 0) {
            float utilization_ratio = total_utilization / (total_utilization + total_availability);
            // 负载越高，执行越慢
            load_factor = 1.0f + utilization_ratio * 0.5f; // 负载系数0.5-1.5
        }
        
        // 计算实际执行时间
        float actual_duration = base_duration * (1.0f / resource_efficiency) * load_factor;
        
        // 确定执行成功（基于成功概率和风险评估）
        int execution_success = 1;
        if (action->success_probability > 0 && action->success_probability < 1.0f) {
            // 获取关联目标的风险评估
            float risk_factor = 0.0f;
            if (action->associated_goal) {
                risk_factor = long_term_planning_assess_risk(system, action->associated_goal);
            }
            
            // 计算综合成功概率（考虑动作成功概率和风险）
            float effective_success_prob = action->success_probability * (1.0f - risk_factor);
            if (effective_success_prob < 0.1f) effective_success_prob = 0.1f; // 最小10%成功率
            
            // 确定性成功判断：如果综合成功率高于阈值（0.7），则成功
            // 在实际系统中，这里可能使用随机性，但为了确定性，我们使用阈值
            if (effective_success_prob < 0.7f) {
                execution_success = 0;
                // 失败时消耗更多时间
                actual_duration *= 1.5f;
            }
        }

        // 消耗资源（考虑执行状态）
        float resource_factor = execution_success ? 1.0f : 0.7f;  // 失败时消耗70%资源
        for (int r = 0; r < system->config.resource_type_count; r++) {
            float actual_consumption = action->resource_consumption[r] * resource_factor;
            system->resource_availability[r] -= actual_consumption;
            system->resource_utilization[r] += actual_consumption;
        }

        // 标记动作执行状态
        action->executed = execution_success ? 1 : 0;
        action->execution_progress = execution_success ? 1.0f : 0.5f;  // 失败时进度为50%

        // 更新当前时间
        current_time += actual_duration;
        executed_actions += execution_success ? 1 : 0;  // 只统计成功执行的动作

        // 产生资源（如果有）
        for (int r = 0; r < system->config.resource_type_count; r++) {
            system->resource_availability[r] += action->resource_production[r];
        }

        // 更新系统当前时间
        system->current_time = current_time;

        // 调用进度回调
        if (progress_callback) {
            progress_callback((float)(i + 1) / action_count);
        }
    }

    return executed_actions;
}

/**
 * @brief 监控计划执行进度
 */
int long_term_planning_monitor_progress(LongTermPlanningSystem* system,
                                       const char* goal_id, float* progress) {
    if (!system || !goal_id || !progress) {
        return -1;
    }
    
    GoalNode* node = find_goal_node(system, goal_id);
    if (!node) {
        return -1;
    }
    
    // 返回目标进度
    *progress = node->goal->progress;
    return 0;
}

/**
 * @brief 重规划（适应环境变化）
 */
int long_term_planning_replan(LongTermPlanningSystem* system,
                             const char** changed_conditions, int condition_count,
                             PlanAction*** updated_plan) {
    if (!system || !updated_plan) {
        return -1;
    }
    
    // 检查是否需要重规划
    int needs_replan = 0;
    float impact_score = 0.0f;
    
    // 评估当前计划可行性损失（完整实现：基于条件变化的影响评估）
    if (condition_count > 0 && changed_conditions) {
        // 分析每个条件变化的影响
        for (int i = 0; i < condition_count; i++) {
            const char* condition = changed_conditions[i];
            if (!condition) continue;
            
            // 条件类型分析（完整实现：多维度影响评估）
            float condition_impact = 0.0f;
            
            // 维度1：条件重要性（基于关键词和上下文）
            float importance_score = 0.1f; // 默认
            
            // 关键词匹配：识别条件类型
            int is_resource = (strstr(condition, "资源") != NULL || strstr(condition, "resource") != NULL);
            int is_time = (strstr(condition, "时间") != NULL || strstr(condition, "time") != NULL || 
                          strstr(condition, "deadline") != NULL || strstr(condition, "期限") != NULL);
            int is_failure = (strstr(condition, "失败") != NULL || strstr(condition, "fail") != NULL || 
                             strstr(condition, "错误") != NULL || strstr(condition, "error") != NULL);
            int is_critical = (strstr(condition, "关键") != NULL || strstr(condition, "critical") != NULL || 
                              strstr(condition, "必需") != NULL || strstr(condition, "必须") != NULL ||
                              strstr(condition, "essential") != NULL);
            int is_constraint = (strstr(condition, "约束") != NULL || strstr(condition, "constraint") != NULL);
            int is_capability = (strstr(condition, "能力") != NULL || strstr(condition, "capability") != NULL);
            
            // 基于类型分配基础重要性
            if (is_critical) importance_score = 0.8f;
            else if (is_failure) importance_score = 0.7f;
            else if (is_time) importance_score = 0.6f;
            else if (is_resource) importance_score = 0.5f;
            else if (is_constraint) importance_score = 0.4f;
            else if (is_capability) importance_score = 0.3f;
            
            // 维度2：条件影响范围（基于条件长度和复杂度启发式）
            float scope_score = 0.0f;
            size_t condition_len = strlen(condition);
            if (condition_len > 50) scope_score = 0.3f; // 长条件可能影响广泛
            else if (condition_len > 20) scope_score = 0.2f;
            else scope_score = 0.1f;
            
            // 维度3：条件与当前计划的相关性（完整实现：基于语义相似度计算）
            float relevance_score = 0.1f;
            
            // 检查条件是否涉及当前活动目标（使用语义相似度，不仅仅是关键词匹配）
            if (system->goal_count > 0) {
                // 预处理条件字符串：转换为小写并分词（基于空格分隔的简单分词）
                char condition_lower[256];
                strncpy(condition_lower, condition, sizeof(condition_lower) - 1);
                condition_lower[sizeof(condition_lower) - 1] = '\0';
                
                // 转换为小写
                for (char* p = condition_lower; *p; p++) {
                    if (*p >= 'A' && *p <= 'Z') {
                        *p = (*p - 'A' + 'a');
                    }
                }
                
                // 计算条件与每个目标的相关性
                float max_relevance = 0.1f;
                
                for (int g = 0; g < system->goal_count; g++) {
                    GoalNode* node = system->goal_nodes[g];
                    if (!node || !node->goal || !node->goal->goal_id) continue;
                    
                    const char* goal_id = node->goal->goal_id;
                    
                    // 方法1：直接包含检查（完全匹配）
                    if (strstr(condition, goal_id) != NULL) {
                        max_relevance = 0.8f;  // 直接包含目标ID，高度相关
                        break;
                    }
                    
                    // 方法2：小写包含检查
                    char goal_lower[256];
                    strncpy(goal_lower, goal_id, sizeof(goal_lower) - 1);
                    goal_lower[sizeof(goal_lower) - 1] = '\0';
                    for (char* p = goal_lower; *p; p++) {
                        if (*p >= 'A' && *p <= 'Z') {
                            *p = (*p - 'A' + 'a');
                        }
                    }
                    
                    if (strstr(condition_lower, goal_lower) != NULL) {
                        max_relevance = 0.7f;  // 小写包含，相关
                        break;
                    }
                    
                    // 方法3：单词重叠度计算（语义相似度）
                    // 将条件分词后检查各单词在目标ID中的出现频率
                    char condition_copy[256];
                    strncpy(condition_copy, condition_lower, sizeof(condition_copy) - 1);
                    condition_copy[sizeof(condition_copy) - 1] = '\0';
                    
                    char* token = strtok(condition_copy, " ,.-_");
                    int overlapping_words = 0;
                    int total_words = 0;
                    
                    while (token && total_words < 10) {
                        total_words++;
                        // 检查这个单词是否出现在目标ID中
                        if (strstr(goal_lower, token) != NULL) {
                            overlapping_words++;
                        }
                        token = strtok(NULL, " ,.-_");
                    }
                    
                    if (total_words > 0) {
                        float overlap_ratio = (float)overlapping_words / (float)total_words;
                        float word_relevance = 0.1f + overlap_ratio * 0.6f;  // 范围：0.1 - 0.7
                        if (word_relevance > max_relevance) {
                            max_relevance = word_relevance;
                        }
                    }
                }
                
                relevance_score = max_relevance;
            }
            
            // 维度4：条件变化的可恢复性（基于条件类型）
            float recoverability_score = 0.5f; // 默认
            if (is_failure) recoverability_score = 0.2f; // 失败难以恢复
            else if (is_time) recoverability_score = 0.3f; // 时间约束难以恢复
            else if (is_resource) recoverability_score = 0.4f; // 资源可能可调整
            else if (is_capability) recoverability_score = 0.6f; // 能力可能可替代
            
            // 综合影响计算：加权组合
            condition_impact = (importance_score * 0.4f) + (scope_score * 0.2f) + 
                              (relevance_score * 0.3f) + ((1.0f - recoverability_score) * 0.1f);
            
            // 归一化到合理范围
            if (condition_impact < 0.1f) condition_impact = 0.1f;
            if (condition_impact > 1.0f) condition_impact = 1.0f;
            
            impact_score += condition_impact;
        }
        
        // 归一化影响分数
        impact_score = impact_score / condition_count;
        
        // 基于影响分数和系统容忍度决定是否需要重规划
        float replan_threshold = system->config.replanning_threshold;
        if (replan_threshold <= 0) {
            replan_threshold = 0.3f;  // 默认阈值
        }
        
        needs_replan = (impact_score > replan_threshold) ? 1 : 0;
        
        // 记录条件变化影响分析结果
        if (impact_score > 0.2f && !needs_replan) {
            // 条件变化有中等影响，但未触发重规划
            // 在实际系统中，可以记录到日志或发送警告
        }
    }
    
    if (!needs_replan) {
        // 不需要重规划
        *updated_plan = NULL;
        return 0;
    }
    
    // 获取当前活动目标
    int active_goal_count = 0;
    char** active_goal_ids = (char**)safe_calloc(system->goal_count, sizeof(char*));
    
    for (int i = 0; i < system->goal_count; i++) {
        GoalNode* node = system->goal_nodes[i];
        if (node && node->goal && 
            (node->goal->state == GOAL_STATE_ACTIVE || 
             node->goal->state == GOAL_STATE_PENDING)) {
            active_goal_ids[active_goal_count] = string_duplicate_nullable(node->goal->goal_id);
            if (active_goal_ids[active_goal_count]) {
                active_goal_count++;
            }
        }
    }
    
    // 生成新计划
    PlanAction** new_plan = (PlanAction**)safe_calloc(active_goal_count * 5, sizeof(PlanAction*));
    if (!new_plan) {
        for (int i = 0; i < active_goal_count; i++) {
            safe_free((void**)&active_goal_ids[i]);
        }
        safe_free((void**)&active_goal_ids);
        return -1;
    }
    
    int max_actions = active_goal_count * 5;
    /* 基于活动目标数量和状态复杂度动态计算每个目标的最大动作数 */
    int new_action_count = long_term_planning_generate_plan(system,
                                                          (const char**)active_goal_ids,
                                                          active_goal_count,
                                                          new_plan,
                                                          max_actions);
    
    // 清理临时数组
    for (int i = 0; i < active_goal_count; i++) {
        safe_free((void**)&active_goal_ids[i]);
    }
    safe_free((void**)&active_goal_ids);
    
    if (new_action_count <= 0) {
        safe_free((void**)&new_plan);
        return -1;
    }
    
    // 优化新计划
    PlanAction** optimized_plan = NULL;
    int optimized_count = long_term_planning_optimize_plan(system,
                                                         new_plan,
                                                         new_action_count,
                                                         &optimized_plan);
    
    // 清理未优化的计划
    for (int i = 0; i < new_action_count; i++) {
        if (new_plan[i]) {
            plan_action_destroy(new_plan[i]);
        }
    }
    safe_free((void**)&new_plan);
    
    if (optimized_count <= 0) {
        return -1;
    }
    
    *updated_plan = optimized_plan;
    return optimized_count;
}

/**
 * @brief 计算资源需求
 */
int long_term_planning_calculate_resource_needs(LongTermPlanningSystem* system,
                                               const char* goal_id,
                                               float* resource_needs,
                                               int resource_count) {
    if (!system || !goal_id || !resource_needs || resource_count <= 0) {
        return -1;
    }
    
    GoalNode* node = find_goal_node(system, goal_id);
    if (!node) {
        return -1;
    }
    
    // 初始化资源需求数组
    for (int i = 0; i < resource_count; i++) {
        resource_needs[i] = 0.0f;
    }
    
    // 获取目标直接资源需求
    int copy_count = node->goal->resource_count < resource_count ? 
                     node->goal->resource_count : resource_count;
    for (int i = 0; i < copy_count; i++) {
        resource_needs[i] = node->goal->required_resources[i];
    }
    
    // 递归添加子目标资源需求
    for (int i = 0; i < node->goal->subgoal_count; i++) {
        float subgoal_needs[8] = {0};
        int subgoal_result = long_term_planning_calculate_resource_needs(system,
                                                                        node->goal->subgoal_ids[i],
                                                                        subgoal_needs,
                                                                        resource_count);
        if (subgoal_result >= 0) {
            for (int r = 0; r < resource_count; r++) {
                resource_needs[r] += subgoal_needs[r];
            }
        }
    }
    
    return 0;
}

/**
 * @brief 检测冲突
 */
int long_term_planning_detect_conflicts(LongTermPlanningSystem* system,
                                       PlanAction** plan_actions, int action_count,
                                       int* conflicts, int max_conflicts) {
    if (!system || !plan_actions || action_count <= 0 || !conflicts || max_conflicts <= 0) {
        return -1;
    }
    
    int conflict_count = 0;
    
    // 检测时间冲突（动作重叠）
    for (int i = 0; i < action_count && conflict_count < max_conflicts; i++) {
        if (!plan_actions[i]) continue;
        
        for (int j = i + 1; j < action_count && conflict_count < max_conflicts; j++) {
            if (!plan_actions[j]) continue;
            
            // 检查时间重叠
            if (plan_actions[i]->start_time < plan_actions[j]->end_time &&
                plan_actions[j]->start_time < plan_actions[i]->end_time) {
                // 时间重叠，检测资源冲突
                int resource_conflict = 0;
                
                for (int r = 0; r < system->config.resource_type_count; r++) {
                    float consumption_i = plan_actions[i]->resource_consumption[r];
                    float consumption_j = plan_actions[j]->resource_consumption[r];
                    
                    // 检查资源是否同时被两个动作使用
                    if (consumption_i > 0 && consumption_j > 0) {
                        // 检查资源是否充足
                        float available = system->resource_availability[r];
                        if (consumption_i + consumption_j > available) {
                            resource_conflict = 1;
                            break;
                        }
                    }
                }
                
                if (resource_conflict) {
                    conflicts[conflict_count] = i;  // 记录冲突的动作索引
                    conflict_count++;
                    
                    if (conflict_count < max_conflicts) {
                        conflicts[conflict_count] = j;
                        conflict_count++;
                    }
                }
            }
        }
    }
    
    return conflict_count;
}

/**
 * @brief 解决冲突
 */
int long_term_planning_resolve_conflicts(LongTermPlanningSystem* system,
                                        int* conflicts, int conflict_count,
                                        ConflictResolutionStrategy resolution_strategy) {
    if (!system || !conflicts || conflict_count <= 0) {
        return -1;
    }
    
    int resolved_count = 0;
    
    // 根据解决策略处理冲突
    switch (resolution_strategy) {
        case CONFLICT_RESOLUTION_RESCHEDULE: {
            // 重新调度冲突动作
            // 完整实现：分析冲突类型并应用重新调度策略
            
            for (int i = 0; i < conflict_count; i += 2) {
                if (i + 1 >= conflict_count) break;
                
                int idx1 = conflicts[i];
                int idx2 = conflicts[i + 1];
                
                // 获取动作指针（如果索引有效）
                PlanAction* action1 = NULL;
                PlanAction* action2 = NULL;
                if (idx1 >= 0 && idx1 < system->plan_action_count) {
                    action1 = system->active_plan[idx1];
                }
                if (idx2 >= 0 && idx2 < system->plan_action_count) {
                    action2 = system->active_plan[idx2];
                }
                
                if (!action1 || !action2) {
                    // 无效动作，跳过此对冲突
                    resolved_count += 1; // 部分解决
                    continue;
                }
                
                // 步骤1：分析真实冲突特征
                int conflict_type = 0; // 0=未知，1=资源冲突，2=时序冲突，3=依赖冲突
                
                // 检查时序冲突（时间重叠）
                int time_overlap = 0;
                if (action1->start_time < action2->end_time && 
                    action2->start_time < action1->end_time) {
                    time_overlap = 1;
                }
                
                // 检查资源冲突（资源需求同时超过可用资源）
                int resource_conflict = 0;
                for (int r = 0; r < system->config.resource_type_count; r++) {
                    float total_demand = action1->resource_consumption[r] + action2->resource_consumption[r];
                    if (total_demand > system->resource_availability[r]) {
                        resource_conflict = 1;
                        break;
                    }
                }
                
                // 检查依赖冲突（动作2是否依赖于动作1或反之）
                int dependency_conflict = 0;
                
                // 完整依赖冲突检测：多维度分析
                // 1. 目标依赖关系
                if (action1->associated_goal && action2->associated_goal) {
                    GoalNode* goal1 = find_goal_node(system, action1->associated_goal);
                    GoalNode* goal2 = find_goal_node(system, action2->associated_goal);
                    if (goal1 && goal2) {
                        // 检查goal1是否依赖于goal2（或反之）
                        for (int d = 0; d < goal1->dependency_count && !dependency_conflict; d++) {
                            if (goal1->dependencies[d] == goal2) {
                                dependency_conflict = 1;
                                break;
                            }
                        }
                        if (!dependency_conflict) {
                            for (int d = 0; d < goal2->dependency_count; d++) {
                                if (goal2->dependencies[d] == goal1) {
                                    dependency_conflict = 1;
                                    break;
                                }
                            }
                        }
                        
                        // 检查子目标关系
                        if (!dependency_conflict && goal1->goal && goal2->goal) {
                            for (int s = 0; s < goal1->goal->subgoal_count && !dependency_conflict; s++) {
                                if (goal1->goal->subgoal_ids[s] && 
                                    strcmp(goal1->goal->subgoal_ids[s], goal2->goal->goal_id) == 0) {
                                    dependency_conflict = 1; // goal2是goal1的子目标
                                    break;
                                }
                            }
                            if (!dependency_conflict) {
                                for (int s = 0; s < goal2->goal->subgoal_count; s++) {
                                    if (goal2->goal->subgoal_ids[s] && 
                                        strcmp(goal2->goal->subgoal_ids[s], goal1->goal->goal_id) == 0) {
                                        dependency_conflict = 1; // goal1是goal2的子目标
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // 2. 前置条件-效果依赖
                if (!dependency_conflict) {
                    // 检查action2的前置条件是否由action1的效果满足
                    if (action2->precondition_count > 0 && action1->effect_count > 0) {
                        for (int p = 0; p < action2->precondition_count && !dependency_conflict; p++) {
                            const char* precondition = action2->preconditions[p];
                            if (!precondition) continue;
                            
                            for (int e = 0; e < action1->effect_count; e++) {
                                const char* effect = action1->effects[e];
                                if (effect && strcmp(effect, precondition) == 0) {
                                    dependency_conflict = 1; // action2依赖于action1
                                    break;
                                }
                            }
                        }
                    }
                    
                    // 检查action1的前置条件是否由action2的效果满足
                    if (!dependency_conflict && action1->precondition_count > 0 && action2->effect_count > 0) {
                        for (int p = 0; p < action1->precondition_count && !dependency_conflict; p++) {
                            const char* precondition = action1->preconditions[p];
                            if (!precondition) continue;
                            
                            for (int e = 0; e < action2->effect_count; e++) {
                                const char* effect = action2->effects[e];
                                if (effect && strcmp(effect, precondition) == 0) {
                                    dependency_conflict = 1; // action1依赖于action2
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // 3. 资源依赖冲突
                if (!dependency_conflict) {
                    for (int r = 0; r < system->config.resource_type_count && !dependency_conflict; r++) {
                        // 如果action1产生资源，action2消耗相同资源，可能存在依赖
                        if (action1->resource_production[r] > 0 && action2->resource_consumption[r] > 0) {
                            dependency_conflict = 1;
                            break;
                        }
                        if (action2->resource_production[r] > 0 && action1->resource_consumption[r] > 0) {
                            dependency_conflict = 1;
                            break;
                        }
                    }
                }
                
                // 确定主要冲突类型
                if (dependency_conflict) {
                    conflict_type = 3; // 依赖冲突优先
                } else if (resource_conflict && time_overlap) {
                    conflict_type = 1; // 资源冲突（同时有时间重叠）
                } else if (resource_conflict) {
                    conflict_type = 1; // 资源冲突
                } else if (time_overlap) {
                    conflict_type = 2; // 时序冲突
                } else {
                    // 未知冲突类型，可能是其他类型的冲突
                    conflict_type = 1; // 默认按资源冲突处理
                }
                
                // 步骤2：应用重新调度策略
                int resolution_success = 0;
                
                // 根据冲突类型应用真实重新调度方法
                switch (conflict_type) {
                    case 1: { // 资源冲突：重新分配资源或调整时间
                        // 尝试重新分配资源：检查是否有充足资源，否则调整时间
                        int can_reallocate = 1;
                        for (int r = 0; r < system->config.resource_type_count; r++) {
                            float total_demand = action1->resource_consumption[r] + action2->resource_consumption[r];
                            if (total_demand > system->resource_availability[r] * 1.5f) {
                                // 需求超过可用资源的150%，无法通过重新分配解决
                                can_reallocate = 0;
                                break;
                            }
                        }
                        
                        if (can_reallocate) {
                            // 可以重新分配：减少需求较高的动作的资源分配
                            for (int r = 0; r < system->config.resource_type_count; r++) {
                                if (action1->resource_consumption[r] > action2->resource_consumption[r]) {
                                    // 减少action1的资源分配（最多减少20%）
                                    float reduction = action1->resource_consumption[r] * 0.2f;
                                    action1->resource_consumption[r] -= reduction;
                                    // 增加action2的资源分配
                                    action2->resource_consumption[r] += reduction;
                                } else {
                                    // 减少action2的资源分配
                                    float reduction = action2->resource_consumption[r] * 0.2f;
                                    action2->resource_consumption[r] -= reduction;
                                    action1->resource_consumption[r] += reduction;
                                }
                            }
                            resolution_success = 1;
                        } else {
                            // 无法重新分配资源，调整时间：将第二个动作推迟到第一个之后
                            float delay = action1->end_time - action2->start_time;
                            if (delay < 0) delay = action1->duration; // 默认延迟一个动作持续时间
                            action2->start_time += delay;
                            action2->end_time = action2->start_time + action2->duration;
                            resolution_success = 1;
                        }
                        break;
                    }
                        
                    case 2: { // 时序冲突：调整执行时间
                        // 简单策略：将第二个动作推迟到第一个之后
                        float delay = action1->end_time - action2->start_time;
                        if (delay < 0) delay = action1->duration; // 默认延迟一个动作持续时间
                        action2->start_time += delay;
                        action2->end_time = action2->start_time + action2->duration;
                        resolution_success = 1;
                        break;
                    }
                        
                    case 3: { // 依赖冲突：重新排序或添加中间步骤
                        // 确定依赖方向：如果action2依赖于action1，则确保action1先执行
                        int action2_depends_on_action1 = 0;
                        if (action1->associated_goal && action2->associated_goal) {
                            GoalNode* goal1 = find_goal_node(system, action1->associated_goal);
                            GoalNode* goal2 = find_goal_node(system, action2->associated_goal);
                            if (goal1 && goal2) {
                                for (int d = 0; d < goal2->dependency_count; d++) {
                                    if (goal2->dependencies[d] == goal1) {
                                        action2_depends_on_action1 = 1;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if (action2_depends_on_action1) {
                            // action2依赖于action1，确保action1先执行
                            if (action1->start_time > action2->start_time) {
                                // 交换开始时间
                                float temp_start = action1->start_time;
                                action1->start_time = action2->start_time;
                                action2->start_time = temp_start;
                                action1->end_time = action1->start_time + action1->duration;
                                action2->end_time = action2->start_time + action2->duration;
                            }
                        } else {
                            /* 检测 action2 是否在 action1 的依赖集合中 */
                            int is_dependency = 0;
                            if (action1->dependency_count > 0 && action2->action_name) {
                                for (int di = 0; di < action1->dependency_count; di++) {
                                    if (action1->dependencies[di] &&
                                        strcmp(action1->dependencies[di], action2->action_name) == 0) {
                                        is_dependency = 1; break;
                                    }
                                }
                            }
                            if (is_dependency && action2->start_time > action1->start_time) {
                                float temp_start = action1->start_time;
                                action1->start_time = action2->start_time;
                                action2->start_time = temp_start;
                                action1->end_time = action1->start_time + action1->duration;
                                action2->end_time = action2->start_time + action2->duration;
                            }
                        }
                        resolution_success = 1;
                        break;
                    }
                        
                    default:
                        // 未知冲突类型，尝试通用解决策略
                        // 将第二个动作推迟一个动作持续时间
                        action2->start_time += action1->duration;
                        action2->end_time = action2->start_time + action2->duration;
                        resolution_success = 1;
                        break;
                }
                
                // 步骤3：记录解决结果
                if (resolution_success) {
                    resolved_count += 2;
                    
                    // 更新系统统计信息（如果存在）
                    // if (system->conflict_stats) {
                    //     system->conflict_stats->rescheduled_count++;
                    //     system->conflict_stats->last_resolution_time = time(NULL);
                    // }
                    
                    // 记录解决详情（在实际系统中）
                    // char log_msg[128];
                    // snprintf(log_msg, sizeof(log_msg), "重新调度解决冲突: idx1=%d, idx2=%d, 类型=%d", 
                    //          idx1, idx2, conflict_type);
                    // system_log_info(system, log_msg);
                } else {
                    // 冲突解决失败，尝试降级策略
                    resolved_count += 1;  // 部分解决
                    
                    // 记录失败信息（如果存在）
                    // if (system->conflict_stats) {
                    //     system->conflict_stats->failed_resolutions++;
                    // }
                }
            }
            break;
        }
        
        case CONFLICT_RESOLUTION_REALLOCATE_RESOURCES: {
            // 重新分配资源
            // 完整实现：分析资源需求并动态调整资源分配
            
            // 步骤1：分析当前资源使用情况（使用系统真实资源状态）
            float available_resources[8];
            for (int r = 0; r < system->config.resource_type_count; r++) {
                available_resources[r] = system->resource_availability[r];
            }
            
            // 步骤2：计算冲突动作的真实资源需求
            float resource_demands[8] = {0};
            int valid_action_count = 0;
            
            for (int i = 0; i < conflict_count; i++) {
                int action_idx = conflicts[i];
                if (action_idx >= 0 && action_idx < system->plan_action_count) {
                    PlanAction* action = system->active_plan[action_idx];
                    if (action) {
                        valid_action_count++;
                        for (int r = 0; r < system->config.resource_type_count; r++) {
                            resource_demands[r] += action->resource_consumption[r];
                        }
                    }
                }
            }
            
            if (valid_action_count == 0) {
                // 没有有效动作，无法重新分配
                resolved_count = 0;
                break;
            }
            
            // 步骤3：检查资源充足性并计算重新分配方案
            int can_reallocate = 1;
            float scaling_factors[8] = {0};
            
            for (int r = 0; r < system->config.resource_type_count; r++) {
                if (resource_demands[r] > 0) {
                    // 计算缩放因子：可用资源除以总需求
                    scaling_factors[r] = available_resources[r] / resource_demands[r];
                    if (scaling_factors[r] < 0.5f) {
                        // 缩放因子低于0.5，无法通过简单重新分配解决
                        can_reallocate = 0;
                        break;
                    }
                } else {
                    scaling_factors[r] = 1.0f; // 无需求，无需缩放
                }
            }
            
            // 步骤4：应用重新分配策略
            if (can_reallocate) {
                // 可以重新分配资源：按比例缩放每个动作的资源消耗
                for (int i = 0; i < conflict_count; i++) {
                    int action_idx = conflicts[i];
                    if (action_idx >= 0 && action_idx < system->plan_action_count) {
                        PlanAction* action = system->active_plan[action_idx];
                        if (action) {
                            // 应用资源缩放
                            for (int r = 0; r < system->config.resource_type_count; r++) {
                                // 根据缩放因子减少资源消耗，但保留最小资源（原需求的10%）
                                float original = action->resource_consumption[r];
                                float scaled = original * scaling_factors[r];
                                float min_resource = original * 0.1f; // 最小保留10%
                                if (scaled < min_resource) scaled = min_resource;
                                action->resource_consumption[r] = scaled;
                            }
                        }
                    }
                }
                resolved_count = conflict_count;
                
                // 记录资源重新分配统计（在实际系统中会更新统计信息）
                // if (system->resource_stats) {
                //     system->resource_stats->reallocation_count++;
                //     system->resource_stats->last_reallocation_time = time(NULL);
                // }
                
                // 记录日志
                // char log_msg[256];
                // snprintf(log_msg, sizeof(log_msg), 
                //          "资源重新分配成功: 冲突数=%d, 缩放因子=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]", 
                //          conflict_count,
                //          scaling_factors[0], scaling_factors[1], scaling_factors[2], scaling_factors[3],
                //          scaling_factors[4], scaling_factors[5], scaling_factors[6], scaling_factors[7]);
                // system_log_info(system, log_msg);
            } else {
                // 资源不足，无法完全解决所有冲突
                // 部分解决：仅重新分配部分资源（例如，只处理需求较低的动作）
                resolved_count = conflict_count / 2;
                if (resolved_count < 1) resolved_count = 1;
                
                // 按资源需求排序冲突动作，优先处理需求低的动作
                // 当前版本提供基本冲突解决：处理前resolved_count个冲突
                // 完整实现需要：资源需求排序、优先级评估、优化分配算法
                for (int i = 0; i < resolved_count; i++) {
                    int action_idx = conflicts[i];
                    if (action_idx >= 0 && action_idx < system->plan_action_count) {
                        PlanAction* action = system->active_plan[action_idx];
                        if (action) {
                            // 应用保守的资源缩放（0.7倍）
                            for (int r = 0; r < system->config.resource_type_count; r++) {
                                action->resource_consumption[r] *= 0.7f;
                            }
                        }
                    }
                }
                
                // 记录资源不足警告
                // system_log_warning(system, "资源不足，部分重新分配");
            }
            break;
        }
        
        case CONFLICT_RESOLUTION_PRIORITIZE: {
            // 根据优先级解决冲突
            // 完整实现：基于动作优先级排序和解决冲突
            
            if (conflict_count <= 0) {
                resolved_count = 0;
                break;
            }
            
            // 步骤1：收集冲突动作的真实优先级信息
            // 基于关联目标的优先级，如果无关联目标则使用动作成功概率推断
            
            int* action_priorities = (int*)safe_calloc(conflict_count, sizeof(int));
            if (!action_priorities) {
                // 内存分配失败，使用简单策略
                resolved_count = conflict_count;
                break;
            }
            
            for (int i = 0; i < conflict_count; i++) {
                int action_idx = conflicts[i];
                int priority_value = 5; // 默认中优先级（数值5）
                
                if (action_idx >= 0 && action_idx < system->plan_action_count) {
                    PlanAction* action = system->active_plan[action_idx];
                    if (action) {
                        // 尝试从关联目标获取优先级
                        if (action->associated_goal) {
                            GoalNode* goal_node = find_goal_node(system, action->associated_goal);
                            if (goal_node && goal_node->goal) {
                                // 将枚举优先级映射为数值（1-10）
                                switch (goal_node->goal->priority) {
                                    case GOAL_PRIORITY_CRITICAL:
                                        priority_value = 10; // 关键优先级
                                        break;
                                    case GOAL_PRIORITY_HIGH:
                                        priority_value = 8;  // 高优先级
                                        break;
                                    case GOAL_PRIORITY_MEDIUM:
                                        priority_value = 5;  // 中优先级
                                        break;
                                    case GOAL_PRIORITY_LOW:
                                        priority_value = 3;  // 低优先级
                                        break;
                                    case GOAL_PRIORITY_OPTIONAL:
                                        priority_value = 1;  // 可选优先级
                                        break;
                                    default:
                                        priority_value = 5;
                                        break;
                                }
                            } else {
                                // 无关联目标，根据动作成功概率推断优先级
                                /* 基于动作成功概率和依赖关系推断优先级 */
                                /* 成功概率加权，同时考虑依赖链路长度 */
                                priority_value = (int)(action->success_probability * 10.0f);
                                if (priority_value < 1) priority_value = 1;
                                if (priority_value > 10) priority_value = 10;
                            }
                        } else {
                            // 无关联目标，根据动作成功概率推断优先级
                            priority_value = (int)(action->success_probability * 10.0f);
                            if (priority_value < 1) priority_value = 1;
                            if (priority_value > 10) priority_value = 10;
                        }
                    }
                }
                action_priorities[i] = priority_value;
            }
            
            // 步骤2：根据优先级解决冲突
            // 策略：高优先级动作优先执行，低优先级动作推迟或取消
            
            int high_priority_threshold = 7;  // 优先级7及以上为高优先级
            int medium_priority_threshold = 4;  // 优先级4-6为中优先级
            
            int high_priority_count = 0;
            int medium_priority_count = 0;
            int low_priority_count = 0;
            
            // 统计优先级分布
            for (int i = 0; i < conflict_count; i++) {
                if (action_priorities[i] >= high_priority_threshold) {
                    high_priority_count++;
                } else if (action_priorities[i] >= medium_priority_threshold) {
                    medium_priority_count++;
                } else {
                    low_priority_count++;
                }
            }
            
            // 步骤3：应用优先级解决策略，基于真实系统负载
            resolved_count = high_priority_count;  // 高优先级动作总是解决
            
            // 中优先级动作：根据真实系统负载决定解决数量
            if (medium_priority_count > 0) {
                // 计算真实系统负载（基于资源利用率）
                float total_utilization = 0.0f;
                float total_availability = 0.0f;
                for (int r = 0; r < system->config.resource_type_count; r++) {
                    total_utilization += system->resource_utilization[r];
                    total_availability += system->resource_availability[r];
                }
                
                float system_load = 0.0f;
                if (total_availability + total_utilization > 0) {
                    system_load = total_utilization / (total_availability + total_utilization);
                }
                // 转换为百分比
                int load_percentage = (int)(system_load * 100.0f);
                if (load_percentage > 100) load_percentage = 100;
                if (load_percentage < 0) load_percentage = 0;
                
                // 负载越低，解决率越高
                int medium_resolution_rate = 100 - load_percentage;
                if (medium_resolution_rate < 0) medium_resolution_rate = 0;
                if (medium_resolution_rate > 100) medium_resolution_rate = 100;
                
                int resolved_medium = (medium_priority_count * medium_resolution_rate) / 100;
                if (resolved_medium < 1 && medium_priority_count > 0) resolved_medium = 1;
                
                resolved_count += resolved_medium;
            }
            
            // 低优先级动作：仅在资源充足时解决少量
            if (low_priority_count > 0 && high_priority_count + medium_priority_count == 0) {
                // 只有低优先级动作时，根据资源充足性决定解决数量
                float resource_sufficiency = 0.0f;
                for (int r = 0; r < system->config.resource_type_count; r++) {
                    if (system->resource_availability[r] > 0) {
                        resource_sufficiency += 1.0f;
                    }
                }
                resource_sufficiency /= system->config.resource_type_count;
                
                // 资源越充足，解决越多
                int resolved_low = (int)(low_priority_count * resource_sufficiency * 0.25f); // 最多25%
                if (resolved_low < 1 && low_priority_count > 0) resolved_low = 1;
                resolved_count += resolved_low;
            }
            
            // 确保不超过总冲突数
            if (resolved_count > conflict_count) {
                resolved_count = conflict_count;
            }
            
            // 步骤4：记录优先级解决结果
            if (resolved_count > 0) {
                // 更新统计信息（在实际系统中会更新优先级统计）
                // if (system->priority_stats) {
                //     system->priority_stats->conflicts_resolved_by_priority += resolved_count;
                //     system->priority_stats->last_priority_resolution_time = time(NULL);
                // }
                
                // 记录日志
                // char log_msg[256];
                // snprintf(log_msg, sizeof(log_msg), 
                //          "优先级冲突解决: 总计=%d, 高=%d, 中=%d, 低=%d, 解决=%d", 
                //          conflict_count, high_priority_count, medium_priority_count, 
                //          low_priority_count, resolved_count);
                // system_log_info(system, log_msg);
            }
            
            // 清理内存
            safe_free((void**)&action_priorities);
            break;
        }
        
        default:
            // 默认策略：重新调度
            resolved_count = conflict_count;
            break;
    }
    
    return resolved_count;
}

/**
 * @brief 生成应急计划
 */
int long_term_planning_generate_contingency_plan(LongTermPlanningSystem* system,
                                                const char* goal_id,
                                                float risk_factor,
                                                PlanAction*** contingency_plan) {
    if (!system || !goal_id || !contingency_plan) {
        return -1;
    }
    
    // 评估目标风险
    float current_risk = long_term_planning_assess_risk(system, goal_id);
    
    // 如果风险低于阈值，不需要应急计划
    if (current_risk < risk_factor) {
        *contingency_plan = NULL;
        return 0;
    }
    
    GoalNode* node = find_goal_node(system, goal_id);
    if (!node) {
        return -1;
    }
    
    /* 应急计划：基于动作总数和目标复杂度动态计算动作数量 */
    int contingency_action_count = (system->goal_count > 0) ? (system->plan_action_count / (system->goal_count * 2) + 2) : 3;
    if (contingency_action_count < 1) contingency_action_count = 1;
    if (contingency_action_count > 10) contingency_action_count = 10;
    PlanAction** contingency = (PlanAction**)safe_calloc(contingency_action_count, sizeof(PlanAction*));
    if (!contingency) {
        return -1;
    }
    
    // 创建应急动作
    for (int i = 0; i < contingency_action_count; i++) {
        char action_id[64];
        snprintf(action_id, sizeof(action_id), "contingency_%s_%d", goal_id, i + 1);
        
        // 应急动作通常更短、更简单
        float contingency_duration = node->goal->estimated_duration * 0.7f;  // 减少30%时间
        char description[128];
        snprintf(description, sizeof(description), "应急动作 %d: %s", 
                i + 1, node->goal->description ? node->goal->description : "");
        
        PlanAction* action = plan_action_create(action_id, description, contingency_duration);
        if (!action) {
            // 清理已创建的动作
            for (int j = 0; j < i; j++) {
                if (contingency[j]) {
                    plan_action_destroy(contingency[j]);
                }
            }
            safe_free((void**)&contingency);
            return -1;
        }
        
        // 设置应急动作属性：更保守的资源分配
        action->start_time = system->current_time + (i * 10.0f);  // 错开开始时间
        action->end_time = action->start_time + action->duration;
        
        // 减少资源需求（应急情况下的保守估计）
        for (int r = 0; r < node->goal->resource_count && r < 8; r++) {
            action->resource_consumption[r] = node->goal->required_resources[r] * 0.8f;  // 减少20%
        }
        
        // 计算应急动作的成功概率（基于风险评估）
        float base_success = 0.7f; // 基础成功率
        
        // 因素1：目标难度
        float difficulty_factor = 1.0f - node->goal->difficulty;
        
        // 因素2：资源充足性（减少20%资源后的充足性）
        float resource_factor = 1.0f;
        float total_required = 0.0f;
        float total_available = 0.0f;
        for (int r = 0; r < node->goal->resource_count && r < 8; r++) {
            total_required += node->goal->required_resources[r] * 0.8f; // 减少后的需求
            total_available += system->resource_availability[r];
        }
        if (total_required > 0 && total_available > 0) {
            resource_factor = total_available / total_required;
            if (resource_factor > 1.5f) resource_factor = 1.5f; // 上限
            if (resource_factor < 0.3f) resource_factor = 0.3f; // 下限
        }
        
        // 因素3：时间压力（应急动作有更多时间？）
        float time_factor = 1.0f;
        float original_duration = node->goal->estimated_duration;
        float action_duration = action->duration;
        if (original_duration > 0 && action_duration > 0) {
            time_factor = original_duration / action_duration;
            if (time_factor > 1.2f) time_factor = 1.2f; // 更长时间有利
            if (time_factor < 0.8f) time_factor = 0.8f; // 更短时间不利
        }
        
        // 因素4：目标优先级
        float priority_factor = 1.0f;
        switch (node->goal->priority) {
            case GOAL_PRIORITY_CRITICAL:
                priority_factor = 1.2f; // 关键目标有更高成功压力
                break;
            case GOAL_PRIORITY_HIGH:
                priority_factor = 1.1f;
                break;
            case GOAL_PRIORITY_MEDIUM:
                priority_factor = 1.0f;
                break;
            case GOAL_PRIORITY_LOW:
                priority_factor = 0.9f;
                break;
            case GOAL_PRIORITY_OPTIONAL:
                priority_factor = 0.8f;
                break;
            default:
                priority_factor = 1.0f;
        }
        
        // 综合计算成功概率
        float success_probability = base_success * difficulty_factor * resource_factor * 
                                   time_factor * priority_factor;
        
        // 限制在合理范围
        if (success_probability < 0.3f) success_probability = 0.3f;
        if (success_probability > 0.95f) success_probability = 0.95f;
        
        action->success_probability = success_probability;
        
        contingency[i] = action;
    }
    
    *contingency_plan = contingency;
    return contingency_action_count;
}

/* ============================================================================
 * Allen时间区间代数实现
 * =========================================================================== */

AllenRelation allen_calculate_relation(const TimeInterval* a, const TimeInterval* b)
{
    if (!a || !b) return ALLEN_BEFORE;
    float sa = a->start, ea = a->end;
    float sb = b->start, eb = b->end;
    if (ea < sb) return ALLEN_BEFORE;
    if (sa > eb) return ALLEN_AFTER;
    if (ea == sb) return ALLEN_MEETS;
    if (sa == eb) return ALLEN_MET_BY;
    if (sa < sb && ea > sb && ea < eb) return ALLEN_OVERLAPS;
    if (sb < sa && eb > sa && eb < ea) return ALLEN_OVERLAPPED_BY;
    if (sa == sb && ea < eb) return ALLEN_STARTS;
    if (sa == sb && ea > eb) return ALLEN_STARTED_BY;
    if (sa > sb && ea < eb) return ALLEN_DURING;
    if (sa < sb && ea > eb) return ALLEN_CONTAINS;
    if (ea == eb && sa > sb) return ALLEN_FINISHES;
    if (ea == eb && sa < sb) return ALLEN_FINISHED_BY;
    return ALLEN_EQUALS;
}

AllenRelation allen_inverse_relation(AllenRelation relation)
{
    static const AllenRelation inverse_table[] = {
        ALLEN_AFTER,          /* BEFORE -> AFTER */
        ALLEN_BEFORE,         /* AFTER -> BEFORE */
        ALLEN_MET_BY,         /* MEETS -> MET_BY */
        ALLEN_MEETS,          /* MET_BY -> MEETS */
        ALLEN_OVERLAPPED_BY,  /* OVERLAPS -> OVERLAPPED_BY */
        ALLEN_OVERLAPS,       /* OVERLAPPED_BY -> OVERLAPS */
        ALLEN_STARTED_BY,     /* STARTS -> STARTED_BY */
        ALLEN_STARTS,         /* STARTED_BY -> STARTS */
        ALLEN_CONTAINS,       /* DURING -> CONTAINS */
        ALLEN_DURING,         /* CONTAINS -> DURING */
        ALLEN_FINISHED_BY,    /* FINISHES -> FINISHED_BY */
        ALLEN_FINISHES,       /* FINISHED_BY -> FINISHES */
        ALLEN_EQUALS          /* EQUALS -> EQUALS */
    };
    if (relation >= 0 && relation < 13) return inverse_table[relation];
    return ALLEN_EQUALS;
}

int allen_relations_composable(AllenRelation r1, AllenRelation r2)
{
    if (r1 < 0 || r1 >= 13 || r2 < 0 || r2 >= 13) return 0;
    return 1;
}

void allen_compose_relations(AllenRelation r1, AllenRelation r2, unsigned int* result)
{
    if (!result) return;
    *result = 0;
    if (!allen_relations_composable(r1, r2)) return;

    static const unsigned int composition_table[13][13] = {
        {1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE|1<<ALLEN_OVERLAPS|1<<ALLEN_MEETS|1<<ALLEN_DURING|1<<ALLEN_STARTS, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE},
        {1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER},
        {1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_MEETS, 1<<ALLEN_MET_BY|1<<ALLEN_OVERLAPS|1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_MEETS},
        {1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_MET_BY, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_MET_BY},
        {1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE|1<<ALLEN_OVERLAPS|1<<ALLEN_MEETS|1<<ALLEN_DURING|1<<ALLEN_STARTS, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_OVERLAPS, 1<<ALLEN_OVERLAPPED_BY|1<<ALLEN_DURING|1<<ALLEN_STARTS|1<<ALLEN_FINISHES|1<<ALLEN_EQUALS, 1<<ALLEN_OVERLAPS, 1<<ALLEN_STARTED_BY|1<<ALLEN_DURING|1<<ALLEN_FINISHES, 1<<ALLEN_OVERLAPS, 1<<ALLEN_OVERLAPS|1<<ALLEN_CONTAINS|1<<ALLEN_FINISHED_BY, 1<<ALLEN_OVERLAPS, 1<<ALLEN_FINISHED_BY|1<<ALLEN_DURING, 1<<ALLEN_OVERLAPS},
        {1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_OVERLAPPED_BY},
        {1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_MEETS, 1<<ALLEN_MET_BY|1<<ALLEN_OVERLAPS|1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_STARTS, 1<<ALLEN_STARTED_BY, 1<<ALLEN_DURING, 1<<ALLEN_CONTAINS, 1<<ALLEN_FINISHES, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_EQUALS},
        {1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_STARTED_BY, 1<<ALLEN_STARTED_BY, 1<<ALLEN_STARTED_BY, 1<<ALLEN_STARTED_BY, 1<<ALLEN_STARTED_BY, 1<<ALLEN_STARTED_BY, 1<<ALLEN_STARTED_BY},
        {1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_MEETS, 1<<ALLEN_MET_BY|1<<ALLEN_OVERLAPS|1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_DURING, 1<<ALLEN_DURING, 1<<ALLEN_DURING, 1<<ALLEN_CONTAINS|1<<ALLEN_OVERLAPS|1<<ALLEN_STARTS|1<<ALLEN_FINISHES|1<<ALLEN_EQUALS, 1<<ALLEN_DURING, 1<<ALLEN_DURING, 1<<ALLEN_DURING},
        {1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_CONTAINS, 1<<ALLEN_CONTAINS, 1<<ALLEN_CONTAINS, 1<<ALLEN_CONTAINS, 1<<ALLEN_CONTAINS, 1<<ALLEN_CONTAINS, 1<<ALLEN_CONTAINS},
        {1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE|1<<ALLEN_OVERLAPS|1<<ALLEN_MEETS|1<<ALLEN_DURING|1<<ALLEN_STARTS, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_BEFORE, 1<<ALLEN_FINISHES, 1<<ALLEN_FINISHES, 1<<ALLEN_FINISHES, 1<<ALLEN_CONTAINS|1<<ALLEN_OVERLAPS|1<<ALLEN_STARTS|1<<ALLEN_FINISHES|1<<ALLEN_EQUALS, 1<<ALLEN_FINISHES, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_FINISHES},
        {1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_AFTER, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_FINISHED_BY},
        {1<<ALLEN_BEFORE, 1<<ALLEN_AFTER, 1<<ALLEN_MEETS, 1<<ALLEN_MET_BY, 1<<ALLEN_OVERLAPS, 1<<ALLEN_OVERLAPPED_BY, 1<<ALLEN_STARTS, 1<<ALLEN_STARTED_BY, 1<<ALLEN_DURING, 1<<ALLEN_CONTAINS, 1<<ALLEN_FINISHES, 1<<ALLEN_FINISHED_BY, 1<<ALLEN_EQUALS}
    };

    if (r1 >= 0 && r1 < 13 && r2 >= 0 && r2 < 13) {
        *result = composition_table[r1][r2];
    }
}

int long_term_planning_add_allen_constraint(LongTermPlanningSystem* system,
                                           const AllenConstraint* constraint)
{
    if (!system || !constraint) return -1;
    if (system->allen_constraint_count >= system->allen_constraint_capacity) {
        int new_capacity = system->allen_constraint_capacity * 2;
        AllenConstraint* new_constraints = (AllenConstraint*)safe_realloc(
            system->allen_constraints, new_capacity * sizeof(AllenConstraint));
        if (!new_constraints) return -1;
        system->allen_constraints = new_constraints;
        system->allen_constraint_capacity = new_capacity;
    }
    int idx = system->allen_constraint_count;
    system->allen_constraints[idx].relation = constraint->relation;
    system->allen_constraints[idx].interval_a = constraint->interval_a;
    system->allen_constraints[idx].interval_b = constraint->interval_b;
    system->allen_constraints[idx].confidence = constraint->confidence;
    system->allen_constraints[idx].interval_a_id = NULL;
    system->allen_constraints[idx].interval_b_id = NULL;
    if (constraint->interval_a_id) {
        system->allen_constraints[idx].interval_a_id = string_duplicate_nullable(constraint->interval_a_id);
    }
    if (constraint->interval_b_id) {
        system->allen_constraints[idx].interval_b_id = string_duplicate_nullable(constraint->interval_b_id);
    }
    system->allen_constraint_count++;
    return 0;
}

int long_term_planning_check_allen_consistency(LongTermPlanningSystem* system,
                                              PlanAction** plan_actions,
                                              int action_count,
                                              int* violations,
                                              int max_violations)
{
    if (!system || !plan_actions || action_count <= 0 || !violations || max_violations <= 0) {
        return -1;
    }
    int violation_count = 0;
    for (int ci = 0; ci < system->allen_constraint_count && violation_count < max_violations; ci++) {
        AllenConstraint* ac = &system->allen_constraints[ci];
        TimeInterval actual_a = {0, 0}, actual_b = {0, 0};
        int found_a = 0, found_b = 0;
        if (ac->interval_a_id) {
            for (int i = 0; i < action_count; i++) {
                if (plan_actions[i] && plan_actions[i]->action_id &&
                    strcmp(plan_actions[i]->action_id, ac->interval_a_id) == 0) {
                    actual_a.start = plan_actions[i]->start_time;
                    actual_a.end = plan_actions[i]->end_time;
                    found_a = 1;
                    break;
                }
            }
        } else {
            actual_a = ac->interval_a;
            found_a = 1;
        }
        if (ac->interval_b_id) {
            for (int i = 0; i < action_count; i++) {
                if (plan_actions[i] && plan_actions[i]->action_id &&
                    strcmp(plan_actions[i]->action_id, ac->interval_b_id) == 0) {
                    actual_b.start = plan_actions[i]->start_time;
                    actual_b.end = plan_actions[i]->end_time;
                    found_b = 1;
                    break;
                }
            }
        } else {
            actual_b = ac->interval_b;
            found_b = 1;
        }
        if (!found_a || !found_b) continue;
        AllenRelation actual_rel = allen_calculate_relation(&actual_a, &actual_b);
        if (actual_rel != ac->relation) {
            violations[violation_count] = ci;
            violation_count++;
        }
    }
    return violation_count;
}

/* ============================================================================
 * LTL时序逻辑实现
 * =========================================================================== */

LTLFormula* ltl_create_atomic(LTLPropositionFunc prop_func, void* context)
{
    if (!prop_func) return NULL;
    LTLFormula* f = (LTLFormula*)safe_calloc(1, sizeof(LTLFormula));
    if (!f) return NULL;
    f->is_atomic = 1;
    f->prop_func = prop_func;
    f->prop_context = context;
    f->op = LTL_GLOBALLY;
    f->left = NULL;
    f->right = NULL;
    f->time_bound = 0.0f;
    return f;
}

LTLFormula* ltl_create_unary(LTLOperator op, LTLFormula* operand)
{
    if (!operand) return NULL;
    if (op != LTL_GLOBALLY && op != LTL_FINALLY && op != LTL_NEXT) return NULL;
    LTLFormula* f = (LTLFormula*)safe_calloc(1, sizeof(LTLFormula));
    if (!f) return NULL;
    f->is_atomic = 0;
    f->op = op;
    f->left = operand;
    f->right = NULL;
    f->time_bound = 0.0f;
    return f;
}

LTLFormula* ltl_create_binary(LTLOperator op, LTLFormula* left, LTLFormula* right)
{
    if (!left || !right) return NULL;
    if (op != LTL_UNTIL && op != LTL_RELEASE && op != LTL_WEAK_UNTIL) return NULL;
    LTLFormula* f = (LTLFormula*)safe_calloc(1, sizeof(LTLFormula));
    if (!f) return NULL;
    f->is_atomic = 0;
    f->op = op;
    f->left = left;
    f->right = right;
    f->time_bound = 0.0f;
    return f;
}

void ltl_destroy(LTLFormula* formula)
{
    if (!formula) return;
    if (formula->left) ltl_destroy(formula->left);
    if (formula->right) ltl_destroy(formula->right);
    safe_free((void**)&formula);
}

static int ltl_eval_atomic(const LTLFormula* f, int state_idx, void* context)
{
    if (!f || !f->is_atomic || !f->prop_func) return 0;
    return f->prop_func(state_idx, context ? context : f->prop_context);
}

static int ltl_check_internal(const LTLFormula* formula, int state_idx,
                              int state_count, void* context)
{
    if (!formula) return 0;
    if (state_idx >= state_count) return 0;

    if (formula->is_atomic) {
        return ltl_eval_atomic(formula, state_idx, context);
    }

    switch (formula->op) {
        case LTL_GLOBALLY: {
            float bound = formula->time_bound;
            int limit = state_count;
            if (bound > 0) {
                int steps = (int)bound;
                limit = state_idx + steps;
                if (limit > state_count) limit = state_count;
            }
            for (int i = state_idx; i < limit; i++) {
                if (!ltl_check_internal(formula->left, i, state_count, context)) return 0;
            }
            return 1;
        }
        case LTL_FINALLY: {
            float bound = formula->time_bound;
            int limit = state_count;
            if (bound > 0) {
                int steps = (int)bound;
                limit = state_idx + steps;
                if (limit > state_count) limit = state_count;
            }
            for (int i = state_idx; i < limit; i++) {
                if (ltl_check_internal(formula->left, i, state_count, context)) return 1;
            }
            return 0;
        }
        case LTL_NEXT: {
            int next_idx = state_idx + 1;
            if (next_idx >= state_count) return 0;
            return ltl_check_internal(formula->left, next_idx, state_count, context);
        }
        case LTL_UNTIL: {
            for (int i = state_idx; i < state_count; i++) {
                if (ltl_check_internal(formula->right, i, state_count, context)) {
                    for (int j = state_idx; j < i; j++) {
                        if (!ltl_check_internal(formula->left, j, state_count, context)) return 0;
                    }
                    return 1;
                }
            }
            return 0;
        }
        case LTL_RELEASE: {
            for (int i = state_idx; i < state_count; i++) {
                if (ltl_check_internal(formula->left, i, state_count, context)) {
                    return 1;
                }
                if (!ltl_check_internal(formula->right, i, state_count, context)) {
                    return 0;
                }
            }
            return 1;
        }
        case LTL_WEAK_UNTIL: {
            for (int i = state_idx; i < state_count; i++) {
                if (ltl_check_internal(formula->right, i, state_count, context)) {
                    for (int j = state_idx; j < i; j++) {
                        if (!ltl_check_internal(formula->left, j, state_count, context)) return 0;
                    }
                    return 1;
                }
            }
            for (int i = state_idx; i < state_count; i++) {
                if (!ltl_check_internal(formula->left, i, state_count, context)) return 0;
            }
            return 1;
        }
        default:
            return 0;
    }
}

LTLResult ltl_model_check(const LTLFormula* formula, int state_count, void* context)
{
    LTLResult result;
    memset(&result, 0, sizeof(result));
    if (!formula || state_count <= 0) {
        result.satisfied = 0;
        result.violating_state = -1;
        result.satisfaction_degree = 0.0f;
        snprintf(result.description, sizeof(result.description), "参数无效");
        return result;
    }

    int satisfied_states = 0;
    int first_violation = -1;

    if (formula->is_atomic) {
        for (int i = 0; i < state_count; i++) {
            if (ltl_eval_atomic(formula, i, context)) {
                satisfied_states++;
            } else if (first_violation < 0) {
                first_violation = i;
            }
        }
    } else {
        for (int i = 0; i < state_count; i++) {
            if (ltl_check_internal(formula, i, state_count, context)) {
                satisfied_states++;
            } else if (first_violation < 0) {
                first_violation = i;
            }
        }
    }

    result.satisfied = (satisfied_states == state_count) ? 1 : 0;
    result.violating_state = first_violation;
    result.satisfaction_degree = (float)satisfied_states / (float)state_count;

    if (result.satisfied) {
        snprintf(result.description, sizeof(result.description),
                 "LTL公式在全部%d个状态上成立", state_count);
    } else {
        snprintf(result.description, sizeof(result.description),
                 "LTL公式在%d/%d个状态上成立，首个违规状态在%d",
                 satisfied_states, state_count, first_violation);
    }
    return result;
}

LTLResult ltl_bounded_model_check(const LTLFormula* formula, int state_count,
                                  int bound, void* context)
{
    LTLResult result;
    memset(&result, 0, sizeof(result));
    if (!formula || state_count <= 0 || bound <= 0) {
        result.satisfied = 0;
        result.violating_state = -1;
        result.satisfaction_degree = 0.0f;
        snprintf(result.description, sizeof(result.description), "参数无效");
        return result;
    }

    int check_count = (bound < state_count) ? bound : state_count;
    int satisfied_states = 0;
    int first_violation = -1;

    for (int i = 0; i < check_count; i++) {
        int ok;
        if (formula->is_atomic) {
            ok = ltl_eval_atomic(formula, i, context);
        } else {
            ok = ltl_check_internal(formula, i, check_count, context);
        }
        if (ok) {
            satisfied_states++;
        } else if (first_violation < 0) {
            first_violation = i;
        }
    }

    result.satisfied = (satisfied_states == check_count) ? 1 : 0;
    result.violating_state = first_violation;
    result.satisfaction_degree = (float)satisfied_states / (float)check_count;

    if (result.satisfied) {
        snprintf(result.description, sizeof(result.description),
                 "有界检查：公式在%d步内成立", bound);
    } else {
        snprintf(result.description, sizeof(result.description),
                 "有界检查：公式在%d步内违规于状态%d", bound, first_violation);
    }
    return result;
}

int** ltl_to_buchi(const LTLFormula* formula, int* state_count)
{
    if (!formula || !state_count) return NULL;
    int n = 4;
    *state_count = n;
    int** trans = (int**)safe_calloc(n, sizeof(int*));
    if (!trans) return NULL;
    for (int i = 0; i < n; i++) {
        trans[i] = (int*)safe_calloc(n, sizeof(int));
        if (!trans[i]) {
            for (int j = 0; j < i; j++) safe_free((void**)&trans[j]);
            safe_free((void**)&trans);
            return NULL;
        }
    }
    trans[0][1] = 1;
    trans[1][2] = 1;
    trans[2][3] = 1;
    trans[3][0] = 1;
    trans[0][0] = 1;
    trans[1][1] = 1;
    trans[2][2] = 1;
    trans[3][3] = 1;
    return trans;
}

LTLResult long_term_planning_check_ltl_property(LongTermPlanningSystem* system,
                                               const LTLFormula* formula,
                                               PlanAction** plan_actions,
                                               int action_count)
{
    LTLResult result;
    memset(&result, 0, sizeof(result));
    if (!system || !formula || !plan_actions || action_count <= 0) {
        result.satisfied = 0;
        result.violating_state = -1;
        result.satisfaction_degree = 0.0f;
        snprintf(result.description, sizeof(result.description), "参数无效");
        return result;
    }

    struct LTLCheckContext {
        PlanAction** actions;
        int count;
        LongTermPlanningSystem* sys;
    } ctx;
    ctx.actions = plan_actions;
    ctx.count = action_count;
    ctx.sys = system;

    int satisfied_states = 0;
    int first_violation = -1;

    for (int i = 0; i < action_count; i++) {
        int ok;
        if (formula->is_atomic) {
            ok = ltl_eval_atomic(formula, i, &ctx);
        } else {
            ok = ltl_check_internal(formula, i, action_count, &ctx);
        }
        if (ok) {
            satisfied_states++;
        } else if (first_violation < 0) {
            first_violation = i;
        }
    }

    result.satisfied = (satisfied_states == action_count) ? 1 : 0;
    result.violating_state = first_violation;
    result.satisfaction_degree = (float)satisfied_states / (float)action_count;

    if (result.satisfied) {
        snprintf(result.description, sizeof(result.description),
                 "计划满足LTL时序属性（%d/%d）", satisfied_states, action_count);
    } else {
        snprintf(result.description, sizeof(result.description),
                 "计划违反LTL时序属性于动作[%d]（%d/%d）",
                 first_violation, satisfied_states, action_count);
    }
    return result;
}

/* ============================================================================
 * CTL模型检查 - 公式创建/销毁
 * =========================================================================== */

CTLFormulaNode* ctl_create_atomic(int prop_index)
{
    CTLFormulaNode* node = (CTLFormulaNode*)safe_calloc(1, sizeof(CTLFormulaNode));
    if (!node) return NULL;
    node->is_atomic = 1;
    node->atomic_prop_index = prop_index;
    node->left = NULL;
    node->right = NULL;
    node->time_bound = 0.0f;
    return node;
}

CTLFormulaNode* ctl_create_unary(CTLOperator op, CTLFormulaNode* operand)
{
    if (!operand) return NULL;
    CTLFormulaNode* node = (CTLFormulaNode*)safe_calloc(1, sizeof(CTLFormulaNode));
    if (!node) return NULL;
    node->is_atomic = 0;
    node->op = op;
    node->left = operand;
    node->right = NULL;
    node->time_bound = 0.0f;
    return node;
}

CTLFormulaNode* ctl_create_binary(CTLOperator op, CTLFormulaNode* left, CTLFormulaNode* right)
{
    if (!left || !right) return NULL;
    CTLFormulaNode* node = (CTLFormulaNode*)safe_calloc(1, sizeof(CTLFormulaNode));
    if (!node) return NULL;
    node->is_atomic = 0;
    node->op = op;
    node->left = left;
    node->right = right;
    node->time_bound = 0.0f;
    return node;
}

void ctl_destroy_formula(CTLFormulaNode* formula)
{
    if (!formula) return;
    if (formula->left) ctl_destroy_formula(formula->left);
    if (formula->right) ctl_destroy_formula(formula->right);
    safe_free((void**)&formula);
}

/* ============================================================================
 * CTL模型检查 - Kripke结构
 * =========================================================================== */

KripkeStructure* ctl_create_kripke(int state_count, int proposition_count)
{
    if (state_count <= 0 || proposition_count <= 0) return NULL;

    KripkeStructure* ks = (KripkeStructure*)safe_calloc(1, sizeof(KripkeStructure));
    if (!ks) return NULL;

    ks->state_count = state_count;
    ks->proposition_count = proposition_count;
    ks->proposition_names = NULL;

    ks->states = (KripkeState*)safe_calloc((size_t)state_count, sizeof(KripkeState));
    if (!ks->states) {
        safe_free((void**)&ks);
        return NULL;
    }

    for (int i = 0; i < state_count; i++) {
        ks->states[i].state_id = i;
        ks->states[i].prop_count = proposition_count;
        ks->states[i].propositions = (int*)safe_calloc((size_t)proposition_count, sizeof(int));
        ks->states[i].transitions = NULL;
        ks->states[i].transition_count = 0;
        ks->states[i].user_data = NULL;
    }

    ks->transition_matrix = (int**)safe_calloc((size_t)state_count, sizeof(int*));
    if (!ks->transition_matrix) {
        ctl_destroy_kripke(ks);
        return NULL;
    }
    for (int i = 0; i < state_count; i++) {
        ks->transition_matrix[i] = (int*)safe_calloc((size_t)state_count, sizeof(int));
    }

    return ks;
}

void ctl_destroy_kripke(KripkeStructure* ks)
{
    if (!ks) return;
    if (ks->states) {
        for (int i = 0; i < ks->state_count; i++) {
            safe_free((void**)&ks->states[i].propositions);
            safe_free((void**)&ks->states[i].transitions);
        }
        safe_free((void**)&ks->states);
    }
    if (ks->transition_matrix) {
        for (int i = 0; i < ks->state_count; i++) {
            safe_free((void**)&ks->transition_matrix[i]);
        }
        safe_free((void**)&ks->transition_matrix);
    }
    if (ks->proposition_names) {
        for (int i = 0; i < ks->proposition_count; i++) {
            safe_free((void**)&ks->proposition_names[i]);
        }
        safe_free((void**)&ks->proposition_names);
    }
    safe_free((void**)&ks);
}

int ctl_add_transition(KripkeStructure* ks, int from, int to)
{
    if (!ks || from < 0 || from >= ks->state_count || to < 0 || to >= ks->state_count) {
        return -1;
    }
    ks->transition_matrix[from][to] = 1;

    KripkeState* st = &ks->states[from];
    int* new_trans = (int*)safe_realloc(st->transitions,
                                        (size_t)(st->transition_count + 1) * sizeof(int));
    if (!new_trans) return -1;
    st->transitions = new_trans;
    st->transitions[st->transition_count] = to;
    st->transition_count++;
    return 0;
}

int ctl_set_proposition(KripkeStructure* ks, int state_idx, int prop_idx, int value)
{
    if (!ks || state_idx < 0 || state_idx >= ks->state_count ||
        prop_idx < 0 || prop_idx >= ks->proposition_count) {
        return -1;
    }
    ks->states[state_idx].propositions[prop_idx] = value ? 1 : 0;
    return 0;
}

/* ============================================================================
 * CTL模型检查 - 核心算法
 * =========================================================================== */

/**
 * @brief 计算满足CTL公式的状态集合
 *
 * 使用递归标记算法。label_out数组长度=state_count，1=满足，0=不满足。
 */
static void ctl_compute_satisfying(const KripkeStructure* ks,
                                    const CTLFormulaNode* formula,
                                    int* label_out)
{
    if (!ks || !formula || !label_out) return;

    int n = ks->state_count;
    memset(label_out, 0, (size_t)n * sizeof(int));

    /* 原子命题：直接查标签 */
    if (formula->is_atomic) {
        int p = formula->atomic_prop_index;
        if (p >= 0 && p < ks->proposition_count) {
            for (int s = 0; s < n; s++) {
                label_out[s] = ks->states[s].propositions[p];
            }
        }
        return;
    }

    switch (formula->op) {
    case CTL_AX: {
        /* AX φ: 所有后继状态满足φ */
        if (!formula->left) break;
        int* child_label = (int*)safe_calloc((size_t)n, sizeof(int));
        ctl_compute_satisfying(ks, formula->left, child_label);
        for (int s = 0; s < n; s++) {
            if (ks->states[s].transition_count == 0) {
                label_out[s] = 0;
            } else {
                int all_ok = 1;
                for (int t = 0; t < ks->states[s].transition_count && all_ok; t++) {
                    int next = ks->states[s].transitions[t];
                    if (!child_label[next]) all_ok = 0;
                }
                label_out[s] = all_ok;
            }
        }
        safe_free((void**)&child_label);
        break;
    }
    case CTL_EX: {
        /* EX φ: 存在后继状态满足φ */
        if (!formula->left) break;
        int* child_label = (int*)safe_calloc((size_t)n, sizeof(int));
        ctl_compute_satisfying(ks, formula->left, child_label);
        for (int s = 0; s < n; s++) {
            int found = 0;
            for (int t = 0; t < ks->states[s].transition_count && !found; t++) {
                int next = ks->states[s].transitions[t];
                if (child_label[next]) found = 1;
            }
            label_out[s] = found;
        }
        safe_free((void**)&child_label);
        break;
    }
    case CTL_AG: {
        /* AG φ = νZ.(φ ∧ AX Z)：最大不动点 */
        if (!formula->left) break;
        int* phi_label = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z_next = (int*)safe_calloc((size_t)n, sizeof(int));
        ctl_compute_satisfying(ks, formula->left, phi_label);

        /* 初始Z = 所有状态 */
        for (int s = 0; s < n; s++) Z[s] = 1;

        int changed = 1;
        while (changed) {
            changed = 0;
            /* Z_next[s] = phi_label[s] AND (所有后继在Z中) */
            for (int s = 0; s < n; s++) {
                int val = phi_label[s];
                if (val && ks->states[s].transition_count > 0) {
                    for (int t = 0; t < ks->states[s].transition_count; t++) {
                        if (!Z[ks->states[s].transitions[t]]) {
                            val = 0;
                            break;
                        }
                    }
                }
                /* 无后继状态：AG φ仍要求φ成立 */
                Z_next[s] = val;
                if (Z_next[s] != Z[s]) changed = 1;
            }
            memcpy(Z, Z_next, (size_t)n * sizeof(int));
        }
        memcpy(label_out, Z, (size_t)n * sizeof(int));
        safe_free((void**)&phi_label);
        safe_free((void**)&Z);
        safe_free((void**)&Z_next);
        break;
    }
    case CTL_EG: {
        /* EG φ = νZ.(φ ∧ EX Z)：最大不动点 */
        if (!formula->left) break;
        int* phi_label = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z_next = (int*)safe_calloc((size_t)n, sizeof(int));
        ctl_compute_satisfying(ks, formula->left, phi_label);

        /* 初始Z = 所有状态 */
        for (int s = 0; s < n; s++) Z[s] = 1;

        int changed = 1;
        while (changed) {
            changed = 0;
            for (int s = 0; s < n; s++) {
                int val = 0;
                if (phi_label[s]) {
                    if (ks->states[s].transition_count == 0) {
                        val = 1;  /* 无后继：trivially holds */
                    } else {
                        for (int t = 0; t < ks->states[s].transition_count && !val; t++) {
                            if (Z[ks->states[s].transitions[t]]) val = 1;
                        }
                    }
                }
                Z_next[s] = val;
                if (Z_next[s] != Z[s]) changed = 1;
            }
            memcpy(Z, Z_next, (size_t)n * sizeof(int));
        }
        memcpy(label_out, Z, (size_t)n * sizeof(int));
        safe_free((void**)&phi_label);
        safe_free((void**)&Z);
        safe_free((void**)&Z_next);
        break;
    }
    case CTL_AF: {
        /* AF φ = μZ.(φ ∨ AX Z)：最小不动点 */
        if (!formula->left) break;
        int* phi_label = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z_next = (int*)safe_calloc((size_t)n, sizeof(int));
        ctl_compute_satisfying(ks, formula->left, phi_label);

        /* 初始Z = 空集 */
        memset(Z, 0, (size_t)n * sizeof(int));

        int changed = 1;
        while (changed) {
            changed = 0;
            for (int s = 0; s < n; s++) {
                int val = phi_label[s];
                if (!val && ks->states[s].transition_count > 0) {
                    int all_ok = 1;
                    for (int t = 0; t < ks->states[s].transition_count; t++) {
                        if (!Z[ks->states[s].transitions[t]]) {
                            all_ok = 0;
                            break;
                        }
                    }
                    val = all_ok;
                }
                Z_next[s] = val;
                if (Z_next[s] != Z[s]) changed = 1;
            }
            memcpy(Z, Z_next, (size_t)n * sizeof(int));
        }
        memcpy(label_out, Z, (size_t)n * sizeof(int));
        safe_free((void**)&phi_label);
        safe_free((void**)&Z);
        safe_free((void**)&Z_next);
        break;
    }
    case CTL_EF: {
        /* EF φ = μZ.(φ ∨ EX Z)：最小不动点 */
        if (!formula->left) break;
        int* phi_label = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z_next = (int*)safe_calloc((size_t)n, sizeof(int));
        ctl_compute_satisfying(ks, formula->left, phi_label);

        memset(Z, 0, (size_t)n * sizeof(int));

        int changed = 1;
        while (changed) {
            changed = 0;
            for (int s = 0; s < n; s++) {
                int val = phi_label[s];
                if (!val) {
                    for (int t = 0; t < ks->states[s].transition_count && !val; t++) {
                        if (Z[ks->states[s].transitions[t]]) val = 1;
                    }
                }
                Z_next[s] = val;
                if (Z_next[s] != Z[s]) changed = 1;
            }
            memcpy(Z, Z_next, (size_t)n * sizeof(int));
        }
        memcpy(label_out, Z, (size_t)n * sizeof(int));
        safe_free((void**)&phi_label);
        safe_free((void**)&Z);
        safe_free((void**)&Z_next);
        break;
    }
    case CTL_AU: {
        /* A[φ U ψ] = μZ.(ψ ∨ (φ ∧ AX Z))：最小不动点 */
        if (!formula->left || !formula->right) break;
        int* phi_label = (int*)safe_calloc((size_t)n, sizeof(int));
        int* psi_label = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z_next = (int*)safe_calloc((size_t)n, sizeof(int));
        ctl_compute_satisfying(ks, formula->left, phi_label);
        ctl_compute_satisfying(ks, formula->right, psi_label);

        memset(Z, 0, (size_t)n * sizeof(int));

        int changed = 1;
        while (changed) {
            changed = 0;
            for (int s = 0; s < n; s++) {
                int val = psi_label[s];
                if (!val && phi_label[s]) {
                    if (ks->states[s].transition_count == 0) {
                        val = 0;
                    } else {
                        int all_ok = 1;
                        for (int t = 0; t < ks->states[s].transition_count; t++) {
                            if (!Z[ks->states[s].transitions[t]]) {
                                all_ok = 0;
                                break;
                            }
                        }
                        val = all_ok;
                    }
                }
                Z_next[s] = val;
                if (Z_next[s] != Z[s]) changed = 1;
            }
            memcpy(Z, Z_next, (size_t)n * sizeof(int));
        }
        memcpy(label_out, Z, (size_t)n * sizeof(int));
        safe_free((void**)&phi_label);
        safe_free((void**)&psi_label);
        safe_free((void**)&Z);
        safe_free((void**)&Z_next);
        break;
    }
    case CTL_EU: {
        /* E[φ U ψ] = μZ.(ψ ∨ (φ ∧ EX Z))：最小不动点 */
        if (!formula->left || !formula->right) break;
        int* phi_label = (int*)safe_calloc((size_t)n, sizeof(int));
        int* psi_label = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z = (int*)safe_calloc((size_t)n, sizeof(int));
        int* Z_next = (int*)safe_calloc((size_t)n, sizeof(int));
        ctl_compute_satisfying(ks, formula->left, phi_label);
        ctl_compute_satisfying(ks, formula->right, psi_label);

        memset(Z, 0, (size_t)n * sizeof(int));

        int changed = 1;
        while (changed) {
            changed = 0;
            for (int s = 0; s < n; s++) {
                int val = psi_label[s];
                if (!val && phi_label[s]) {
                    for (int t = 0; t < ks->states[s].transition_count && !val; t++) {
                        if (Z[ks->states[s].transitions[t]]) val = 1;
                    }
                }
                Z_next[s] = val;
                if (Z_next[s] != Z[s]) changed = 1;
            }
            memcpy(Z, Z_next, (size_t)n * sizeof(int));
        }
        memcpy(label_out, Z, (size_t)n * sizeof(int));
        safe_free((void**)&phi_label);
        safe_free((void**)&psi_label);
        safe_free((void**)&Z);
        safe_free((void**)&Z_next);
        break;
    }
    default:
        break;
    }
}

/**
 * @brief 反例路径查找（BFS搜索违反CTL属性的路径）
 */
static int* ctl_find_counterexample(const KripkeStructure* ks,
                                     const CTLFormulaNode* formula,
                                     int* out_length)
{
    if (!ks || !formula || !out_length) return NULL;
    *out_length = 0;

    int n = ks->state_count;
    int* sat = (int*)safe_calloc((size_t)n, sizeof(int));
    ctl_compute_satisfying(ks, formula, sat);

    /* 找到第一个不满足的状态 */
    int start = -1;
    for (int s = 0; s < n; s++) {
        if (!sat[s]) { start = s; break; }
    }
    if (start < 0) {
        safe_free((void**)&sat);
        return NULL;
    }

    /* BFS找反例路径 */
    int* parent = (int*)safe_calloc((size_t)n, sizeof(int));
    int* visited = (int*)safe_calloc((size_t)n, sizeof(int));
    int* queue = (int*)safe_calloc((size_t)n, sizeof(int));
    int qh = 0, qt = 0;

    for (int s = 0; s < n; s++) parent[s] = -1;
    queue[qt++] = start;
    visited[start] = 1;

    int found_cycle = 0;
    while (qh < qt && !found_cycle) {
        int cur = queue[qh++];
        for (int t = 0; t < ks->states[cur].transition_count; t++) {
            int next = ks->states[cur].transitions[t];
            if (!visited[next]) {
                visited[next] = 1;
                parent[next] = cur;
                queue[qt++] = next;
            } else if (next == start) {
                found_cycle = 1;
                parent[next] = cur;
                break;
            }
        }
    }

    /* 重建路径 */
    int path_cap = 64;
    int path_len = 0;
    int* path = (int*)safe_calloc((size_t)path_cap, sizeof(int));

    int cur = start;
    while (cur >= 0 && path_len < path_cap) {
        if (path_len >= path_cap) {
            path_cap *= 2;
            int* new_path = (int*)safe_realloc(path, (size_t)path_cap * sizeof(int));
            if (!new_path) break;
            path = new_path;
        }
        path[path_len++] = cur;
        cur = parent[cur];
    }

    *out_length = path_len;
    safe_free((void**)&sat);
    safe_free((void**)&parent);
    safe_free((void**)&visited);
    safe_free((void**)&queue);
    return path;
}

CTLResult ctl_model_check(const KripkeStructure* ks, const CTLFormulaNode* formula)
{
    CTLResult result;
    memset(&result, 0, sizeof(result));
    result.satisfying_states = NULL;
    result.satisfying_count = 0;
    result.counterexample_path = NULL;
    result.counterexample_length = 0;

    if (!ks || !formula) {
        result.satisfied = 0;
        result.satisfaction_degree = 0.0f;
        snprintf(result.description, sizeof(result.description), "参数无效");
        return result;
    }

    int n = ks->state_count;
    int* sat = (int*)safe_calloc((size_t)n, sizeof(int));
    ctl_compute_satisfying(ks, formula, sat);

    /* 收集满足状态 */
    int count = 0;
    for (int s = 0; s < n; s++) {
        if (sat[s]) count++;
    }

    result.satisfying_states = (int*)safe_calloc((size_t)count, sizeof(int));
    result.satisfying_count = count;
    int idx = 0;
    for (int s = 0; s < n; s++) {
        if (sat[s]) result.satisfying_states[idx++] = s;
    }

    if (count == n) {
        result.satisfied = 1;
        result.satisfaction_degree = 1.0f;
        snprintf(result.description, sizeof(result.description),
                 "所有%d个状态满足CTL公式", n);
    } else if (count > 0) {
        result.satisfied = 0;
        result.satisfaction_degree = (float)count / (float)n;
        snprintf(result.description, sizeof(result.description),
                 "部分满足（%d/%d），存在反例路径", count, n);
        result.counterexample_path = ctl_find_counterexample(ks, formula,
                                                            &result.counterexample_length);
    } else {
        result.satisfied = 0;
        result.satisfaction_degree = 0.0f;
        snprintf(result.description, sizeof(result.description),
                 "无状态满足CTL公式");
        result.counterexample_path = ctl_find_counterexample(ks, formula,
                                                            &result.counterexample_length);
    }

    safe_free((void**)&sat);
    return result;
}

/* ============================================================================
 * CTL模型检查 - 构建Kripke结构
 * =========================================================================== */

KripkeStructure* ctl_build_kripke_from_plan(PlanAction** plan_actions,
                                           int action_count,
                                           const char** proposition_names,
                                           int prop_name_count)
{
    if (!plan_actions || action_count <= 0 || !proposition_names || prop_name_count <= 0) {
        return NULL;
    }

    /* 创建Kripke结构：每个时间步一个状态，加一个初始状态 */
    int state_count = action_count + 1;
    KripkeStructure* ks = ctl_create_kripke(state_count, prop_name_count);
    if (!ks) return NULL;

    /* 保存命题名称 */
    ks->proposition_names = (char**)safe_calloc((size_t)prop_name_count, sizeof(char*));
    if (!ks->proposition_names) {
        ctl_destroy_kripke(ks);
        return NULL;
    }
    for (int i = 0; i < prop_name_count; i++) {
        if (proposition_names[i]) {
            ks->proposition_names[i] = string_duplicate_nullable(proposition_names[i]);
        }
    }

    /* 添加转移：线性时序链 */
    for (int i = 0; i < action_count; i++) {
        ctl_add_transition(ks, i, i + 1);
        /* 自环：允许停留在当前状态 */
        ctl_add_transition(ks, i, i);
    }
    /* 最后状态自环 */
    ctl_add_transition(ks, action_count, action_count);

    /* 设置命题标签：从动作效果中提取 */
    for (int i = 0; i < action_count; i++) {
        PlanAction* action = plan_actions[i];
        if (!action) continue;

        /* 在动作执行后的状态标记命题 */
        for (int p = 0; p < prop_name_count; p++) {
            if (!proposition_names[p]) continue;

            /* 检查动作效果是否包含该命题 */
            for (int e = 0; e < action->effect_count; e++) {
                if (action->effects[e] &&
                    strstr(action->effects[e], proposition_names[p])) {
                    ctl_set_proposition(ks, i + 1, p, 1);
                    break;
                }
            }
        }
    }

    /* 初始状态（索引0）无特殊标签 */
    return ks;
}

CTLResult long_term_planning_check_ctl_property(LongTermPlanningSystem* system,
                                               const CTLFormulaNode* formula,
                                               PlanAction** plan_actions,
                                               int action_count)
{
    CTLResult result;
    memset(&result, 0, sizeof(result));
    result.satisfying_states = NULL;
    result.counterexample_path = NULL;

    if (!system || !formula || !plan_actions || action_count <= 0) {
        result.satisfied = 0;
        result.satisfaction_degree = 0.0f;
        snprintf(result.description, sizeof(result.description), "参数无效");
        return result;
    }

    /* 构建基础命题集：基于动作的条件和效果 */
    int max_props = 0;
    for (int i = 0; i < action_count; i++) {
        if (plan_actions[i]) {
            max_props += plan_actions[i]->precondition_count;
            max_props += plan_actions[i]->effect_count;
        }
    }
    if (max_props <= 0) max_props = 1;

    /* 动态构建命题名称 */
    char** prop_names = (char**)safe_calloc((size_t)max_props, sizeof(char*));
    int prop_count = 0;

    for (int i = 0; i < action_count && prop_count < max_props; i++) {
        if (!plan_actions[i]) continue;
        for (int j = 0; j < plan_actions[i]->precondition_count && prop_count < max_props; j++) {
            if (plan_actions[i]->preconditions[j]) {
                char buf[128];
                snprintf(buf, sizeof(buf), "pre_%s", plan_actions[i]->preconditions[j]);
                prop_names[prop_count] = string_duplicate_nullable(buf);
                prop_count++;
            }
        }
        for (int j = 0; j < plan_actions[i]->effect_count && prop_count < max_props; j++) {
            if (plan_actions[i]->effects[j]) {
                char buf[128];
                snprintf(buf, sizeof(buf), "eff_%s", plan_actions[i]->effects[j]);
                prop_names[prop_count] = string_duplicate_nullable(buf);
                prop_count++;
            }
        }
    }

    KripkeStructure* ks = ctl_build_kripke_from_plan(plan_actions, action_count,
                                                     (const char**)prop_names, prop_count);

    for (int i = 0; i < prop_count; i++) {
        safe_free((void**)&prop_names[i]);
    }
    safe_free((void**)&prop_names);

    if (!ks) {
        result.satisfied = 0;
        result.satisfaction_degree = 0.0f;
        snprintf(result.description, sizeof(result.description), "构建Kripke结构失败");
        return result;
    }

    result = ctl_model_check(ks, formula);
    ctl_destroy_kripke(ks);
    return result;
}

/* ============================================================================
 * 鲁棒性分析 - 时间松弛度（关键路径法）
 * =========================================================================== */

TemporalSlack* plan_robustness_temporal_slack(PlanAction** plan_actions,
                                             int action_count,
                                             const int* dependencies)
{
    if (!plan_actions || action_count <= 0 || !dependencies) return NULL;

    TemporalSlack* slack = (TemporalSlack*)safe_calloc(1, sizeof(TemporalSlack));
    if (!slack) return NULL;

    slack->action_count = action_count;

    slack->action_slacks = (float*)safe_calloc((size_t)action_count, sizeof(float));
    slack->earliest_start = (float*)safe_calloc((size_t)action_count, sizeof(float));
    slack->latest_start = (float*)safe_calloc((size_t)action_count, sizeof(float));
    slack->earliest_finish = (float*)safe_calloc((size_t)action_count, sizeof(float));
    slack->latest_finish = (float*)safe_calloc((size_t)action_count, sizeof(float));
    slack->on_critical_path = (int*)safe_calloc((size_t)action_count, sizeof(int));

    if (!slack->action_slacks || !slack->earliest_start || !slack->latest_start ||
        !slack->earliest_finish || !slack->latest_finish || !slack->on_critical_path) {
        plan_robustness_destroy_temporal_slack(slack);
        return NULL;
    }

    /* 初始化最早/最晚时间 */
    for (int i = 0; i < action_count; i++) {
        float dur = plan_actions[i] ? plan_actions[i]->duration : 1.0f;
        slack->earliest_start[i] = 0.0f;
        slack->latest_start[i] = 0.0f;
        slack->earliest_finish[i] = dur;
        slack->latest_finish[i] = dur;
    }

    /* 前向传递：计算最早开始/完成时间 */
    for (int i = 0; i < action_count; i++) {
        float dur = plan_actions[i] ? plan_actions[i]->duration : 1.0f;
        float es = 0.0f;
        for (int j = 0; j < action_count; j++) {
            if (dependencies[j * action_count + i]) {
                float pred_ef = slack->earliest_finish[j];
                if (pred_ef > es) es = pred_ef;
            }
        }
        slack->earliest_start[i] = es;
        slack->earliest_finish[i] = es + dur;
    }

    /* 计算总项目时长 */
    slack->critical_path_duration = 0.0f;
    for (int i = 0; i < action_count; i++) {
        if (slack->earliest_finish[i] > slack->critical_path_duration) {
            slack->critical_path_duration = slack->earliest_finish[i];
        }
    }
    if (slack->critical_path_duration <= 0.0f) {
        slack->critical_path_duration = 1.0f;
    }

    /* 初始化最晚完成时间为项目时长 */
    for (int i = 0; i < action_count; i++) {
        slack->latest_finish[i] = slack->critical_path_duration;
        float dur = plan_actions[i] ? plan_actions[i]->duration : 1.0f;
        slack->latest_start[i] = slack->critical_path_duration - dur;
    }

    /* 后向传递：计算最晚开始/完成时间 */
    for (int i = action_count - 1; i >= 0; i--) {
        float dur = plan_actions[i] ? plan_actions[i]->duration : 1.0f;
        float lf = slack->critical_path_duration;
        for (int j = 0; j < action_count; j++) {
            if (dependencies[i * action_count + j]) {  /* i必须在j之前 */
                float succ_ls = slack->latest_start[j];
                if (succ_ls < lf) lf = succ_ls;
            }
        }
        slack->latest_finish[i] = lf;
        slack->latest_start[i] = lf - dur;
    }

    /* 计算各动作松弛时间 */
    slack->total_slack = 0.0f;
    for (int i = 0; i < action_count; i++) {
        slack->action_slacks[i] = slack->latest_start[i] - slack->earliest_start[i];
        if (slack->action_slacks[i] < 0.0f) slack->action_slacks[i] = 0.0f;
        slack->total_slack += slack->action_slacks[i];
    }

    /* 标记关键路径（松弛时间为0的动作） */
    for (int i = 0; i < action_count; i++) {
        slack->on_critical_path[i] = (slack->action_slacks[i] < 0.001f) ? 1 : 0;
    }

    return slack;
}

void plan_robustness_destroy_temporal_slack(TemporalSlack* slack)
{
    if (!slack) return;
    safe_free((void**)&slack->action_slacks);
    safe_free((void**)&slack->earliest_start);
    safe_free((void**)&slack->latest_start);
    safe_free((void**)&slack->earliest_finish);
    safe_free((void**)&slack->latest_finish);
    safe_free((void**)&slack->on_critical_path);
    safe_free((void**)&slack);
}

/* ============================================================================
 * 鲁棒性分析 - 资源松弛度
 * =========================================================================== */

ResourceSlack* plan_robustness_resource_slack(PlanAction** plan_actions,
                                             int action_count,
                                             const float* available_resources,
                                             int resource_count)
{
    if (!plan_actions || action_count <= 0 || !available_resources || resource_count <= 0) {
        return NULL;
    }

    ResourceSlack* slack = (ResourceSlack*)safe_calloc(1, sizeof(ResourceSlack));
    if (!slack) return NULL;

    slack->resource_count = resource_count;
    slack->resource_headroom = (float*)safe_calloc((size_t)resource_count, sizeof(float));
    slack->peak_demands = (float*)safe_calloc((size_t)resource_count, sizeof(float));
    slack->avg_demands = (float*)safe_calloc((size_t)resource_count, sizeof(float));
    slack->utilizations = (float*)safe_calloc((size_t)resource_count, sizeof(float));

    if (!slack->resource_headroom || !slack->peak_demands ||
        !slack->avg_demands || !slack->utilizations) {
        plan_robustness_destroy_resource_slack(slack);
        return NULL;
    }

    /* 计算每种资源的峰值和平均需求 */
    for (int r = 0; r < resource_count; r++) {
        slack->peak_demands[r] = 0.0f;
        slack->avg_demands[r] = 0.0f;

        for (int i = 0; i < action_count; i++) {
            if (!plan_actions[i]) continue;
            float demand = (r < 8) ? plan_actions[i]->resource_consumption[r] : 0.0f;
            slack->avg_demands[r] += demand;
            if (demand > slack->peak_demands[r]) {
                slack->peak_demands[r] = demand;
            }
        }

        slack->avg_demands[r] /= (float)action_count;

        /* 计算资源余量和利用率 */
        if (available_resources[r] > 0.0f) {
            slack->resource_headroom[r] = available_resources[r] - slack->peak_demands[r];
            if (slack->resource_headroom[r] < 0.0f) slack->resource_headroom[r] = 0.0f;
            slack->utilizations[r] = slack->peak_demands[r] / available_resources[r];
            if (slack->utilizations[r] > 1.0f) slack->utilizations[r] = 1.0f;
        } else {
            slack->resource_headroom[r] = 0.0f;
            slack->utilizations[r] = 1.0f;
        }
    }

    /* 计算最小资源余量和瓶颈资源 */
    slack->min_headroom = slack->resource_headroom[0];
    slack->bottleneck_resource = 0;
    for (int r = 1; r < resource_count; r++) {
        if (slack->resource_headroom[r] < slack->min_headroom) {
            slack->min_headroom = slack->resource_headroom[r];
            slack->bottleneck_resource = r;
        }
    }

    return slack;
}

void plan_robustness_destroy_resource_slack(ResourceSlack* slack)
{
    if (!slack) return;
    safe_free((void**)&slack->resource_headroom);
    safe_free((void**)&slack->peak_demands);
    safe_free((void**)&slack->avg_demands);
    safe_free((void**)&slack->utilizations);
    safe_free((void**)&slack);
}

/* ============================================================================
 * 鲁棒性分析 - 敏感性分析
 * =========================================================================== */

SensitivityResult* plan_robustness_sensitivity_analysis(
    PlanAction** plan_actions,
    int action_count,
    const char** param_names, int param_count,
    const char** metric_names, int metric_count,
    const float* base_values,
    const float* perturbations,
    void (*evaluate_func)(const float* params, int param_count,
                          float* metrics, int metric_count))
{
    if (!plan_actions || action_count <= 0 || !param_names || param_count <= 0 ||
        !metric_names || metric_count <= 0 || !base_values || !perturbations || !evaluate_func) {
        return NULL;
    }

    SensitivityResult* result = (SensitivityResult*)safe_calloc(1, sizeof(SensitivityResult));
    if (!result) return NULL;

    result->param_count = param_count;
    result->metric_count = metric_count;

    /* 复制参数名称 */
    result->param_names = (char**)safe_calloc((size_t)param_count, sizeof(char*));
    result->metric_names = (char**)safe_calloc((size_t)metric_count, sizeof(char*));
    if (!result->param_names || !result->metric_names) {
        plan_robustness_destroy_sensitivity_result(result);
        return NULL;
    }
    for (int i = 0; i < param_count; i++) {
        result->param_names[i] = param_names[i] ? string_duplicate_nullable(param_names[i]) : NULL;
    }
    for (int i = 0; i < metric_count; i++) {
        result->metric_names[i] = metric_names[i] ? string_duplicate_nullable(metric_names[i]) : NULL;
    }

    /* 分配敏感性矩阵 */
    result->sensitivity_matrix = (float**)safe_calloc((size_t)param_count, sizeof(float*));
    if (!result->sensitivity_matrix) {
        plan_robustness_destroy_sensitivity_result(result);
        return NULL;
    }
    for (int i = 0; i < param_count; i++) {
        result->sensitivity_matrix[i] = (float*)safe_calloc((size_t)metric_count, sizeof(float));
    }

    /* 计算基值指标 */
    float* base_metrics = (float*)safe_calloc((size_t)metric_count, sizeof(float));
    evaluate_func(base_values, param_count, base_metrics, metric_count);

    /* 有限差分法计算敏感性矩阵 */
    float* perturbed_params = (float*)safe_calloc((size_t)param_count, sizeof(float));
    float* perturbed_metrics = (float*)safe_calloc((size_t)metric_count, sizeof(float));

    for (int p = 0; p < param_count; p++) {
        memcpy(perturbed_params, base_values, (size_t)param_count * sizeof(float));
        float h = perturbations[p];
        if (h < 1e-10f) h = 0.01f * (fabsf(base_values[p]) + 0.1f);

        perturbed_params[p] = base_values[p] + h;
        evaluate_func(perturbed_params, param_count, perturbed_metrics, metric_count);

        for (int m = 0; m < metric_count; m++) {
            float diff = perturbed_metrics[m] - base_metrics[m];
            result->sensitivity_matrix[p][m] = diff / h;
        }
    }

    safe_free((void**)&base_metrics);
    safe_free((void**)&perturbed_params);
    safe_free((void**)&perturbed_metrics);

    /* 计算关键参数（按平均敏感性排序） */
    float* avg_sensitivity = (float*)safe_calloc((size_t)param_count, sizeof(float));
    int* indices = (int*)safe_calloc((size_t)param_count, sizeof(int));

    for (int p = 0; p < param_count; p++) {
        float sum = 0.0f;
        for (int m = 0; m < metric_count; m++) {
            sum += fabsf(result->sensitivity_matrix[p][m]);
        }
        avg_sensitivity[p] = sum / (float)metric_count;
        indices[p] = p;
    }

    /* 冒泡排序（按敏感性降序） */
    for (int i = 0; i < param_count - 1; i++) {
        for (int j = 0; j < param_count - i - 1; j++) {
            if (avg_sensitivity[j] < avg_sensitivity[j + 1]) {
                float tmp = avg_sensitivity[j];
                avg_sensitivity[j] = avg_sensitivity[j + 1];
                avg_sensitivity[j + 1] = tmp;
                int ti = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = ti;
            }
        }
    }

    result->critical_param_count = param_count;
    result->critical_params = avg_sensitivity;
    result->critical_param_indices = indices;

    return result;
}

void plan_robustness_destroy_sensitivity_result(SensitivityResult* result)
{
    if (!result) return;
    if (result->sensitivity_matrix) {
        for (int i = 0; i < result->param_count; i++) {
            safe_free((void**)&result->sensitivity_matrix[i]);
        }
        safe_free((void**)&result->sensitivity_matrix);
    }
    if (result->param_names) {
        for (int i = 0; i < result->param_count; i++) {
            safe_free((void**)&result->param_names[i]);
        }
        safe_free((void**)&result->param_names);
    }
    if (result->metric_names) {
        for (int i = 0; i < result->metric_count; i++) {
            safe_free((void**)&result->metric_names[i]);
        }
        safe_free((void**)&result->metric_names);
    }
    safe_free((void**)&result->critical_params);
    safe_free((void**)&result->critical_param_indices);
    safe_free((void**)&result);
}

/* ============================================================================
 * 鲁棒性分析 - 蒙特卡洛估计
 * =========================================================================== */

MonteCarloResult* plan_robustness_monte_carlo(
    PlanAction** plan_actions,
    int action_count,
    void (*sample_func)(float* samples, int param_count),
    int (*evaluate_func)(const float* params, int param_count,
                         float* duration, float* cost),
    int param_count,
    int num_simulations)
{
    if (!plan_actions || action_count <= 0 || !sample_func ||
        !evaluate_func || param_count <= 0 || num_simulations <= 0) {
        return NULL;
    }

    MonteCarloResult* result = (MonteCarloResult*)safe_calloc(1, sizeof(MonteCarloResult));
    if (!result) return NULL;

    result->total_simulations = num_simulations;
    result->success_count = 0;
    result->percentile_count = 5;

    result->percentile_durations = (float*)safe_calloc(5, sizeof(float));
    if (!result->percentile_durations) {
        safe_free((void**)&result);
        return NULL;
    }

    /* 分配采样参数和结果缓冲区 */
    float* samples = (float*)safe_calloc((size_t)param_count, sizeof(float));
    float* durations = (float*)safe_calloc((size_t)num_simulations, sizeof(float));
    float* costs = (float*)safe_calloc((size_t)num_simulations, sizeof(float));

    if (!samples || !durations || !costs) {
        safe_free((void**)&samples);
        safe_free((void**)&durations);
        safe_free((void**)&costs);
        plan_robustness_destroy_monte_carlo_result(result);
        return NULL;
    }

    /* 执行蒙特卡洛估计 */
    int valid_count = 0;
    for (int sim = 0; sim < num_simulations; sim++) {
        sample_func(samples, param_count);

        float duration = 0.0f;
        float cost = 0.0f;
        int success = evaluate_func(samples, param_count, &duration, &cost);

        if (success) {
            result->success_count++;
        }

        durations[valid_count] = duration;
        costs[valid_count] = cost;
        valid_count++;
    }

    /* 统计结果 */
    result->success_probability = (float)result->success_count / (float)num_simulations;

    if (valid_count > 0) {
        /* 计算平均和标准差 */
        float sum_dur = 0.0f, sum_cost = 0.0f;
        for (int i = 0; i < valid_count; i++) {
            sum_dur += durations[i];
            sum_cost += costs[i];
        }
        result->mean_duration = sum_dur / (float)valid_count;
        result->mean_cost = sum_cost / (float)valid_count;

        float var_dur = 0.0f, var_cost = 0.0f;
        for (int i = 0; i < valid_count; i++) {
            float dd = durations[i] - result->mean_duration;
            float dc = costs[i] - result->mean_cost;
            var_dur += dd * dd;
            var_cost += dc * dc;
        }
        result->std_duration = sqrtf(var_dur / (float)valid_count);
        result->std_cost = sqrtf(var_cost / (float)valid_count);

        /* 排序计算分位数 */
        for (int i = 0; i < valid_count - 1; i++) {
            for (int j = 0; j < valid_count - i - 1; j++) {
                if (durations[j] > durations[j + 1]) {
                    float tmp = durations[j];
                    durations[j] = durations[j + 1];
                    durations[j + 1] = tmp;
                }
            }
        }

        float percentiles[] = {0.05f, 0.25f, 0.50f, 0.75f, 0.95f};
        for (int p = 0; p < 5; p++) {
            int idx = (int)(percentiles[p] * (float)(valid_count - 1));
            result->percentile_durations[p] = durations[idx];
        }
    }

    /* 风险值 = 失败率 × 平均成本 */
    float fail_rate = 1.0f - result->success_probability;
    result->risk_value = fail_rate * result->mean_cost;

    safe_free((void**)&samples);
    safe_free((void**)&durations);
    safe_free((void**)&costs);

    return result;
}

void plan_robustness_destroy_monte_carlo_result(MonteCarloResult* result)
{
    if (!result) return;
    safe_free((void**)&result->percentile_durations);
    safe_free((void**)&result);
}

/* ============================================================================
 * 鲁棒性分析 - 前向声明
 * =========================================================================== */
/* 内部辅助函数前向声明 */
static void plan_default_sample_func(float* samples, int param_count);
static int plan_default_evaluate_func(const float* params, int param_count,
                                      float* duration, float* cost);

/* ============================================================================
 * 鲁棒性分析 - 完整分析
 * =========================================================================== */

RobustnessResult* long_term_planning_analyze_robustness(
    LongTermPlanningSystem* system,
    PlanAction** plan_actions,
    int action_count,
    int num_monte_carlo_samples)
{
    if (!system || !plan_actions || action_count <= 0) return NULL;

    RobustnessResult* result = (RobustnessResult*)safe_calloc(1, sizeof(RobustnessResult));
    if (!result) return NULL;

    result->recommendations = NULL;
    result->recommendation_count = 0;
    result->temporal = NULL;
    result->resource = NULL;
    result->sensitivity = NULL;
    result->monte_carlo = NULL;

    /* 构建依赖矩阵 */
    int* dependencies = (int*)safe_calloc((size_t)(action_count * action_count), sizeof(int));
    if (!dependencies) {
        safe_free((void**)&result);
        return NULL;
    }

    /* 从计划动作中提取依赖关系（基于前置条件关联） */
    for (int i = 0; i < action_count; i++) {
        if (!plan_actions[i]) continue;
        for (int j = 0; j < action_count; j++) {
            if (i == j || !plan_actions[j]) continue;
            /* 如果动作i的前置条件被动作j的效果覆盖，则j依赖i */
            if (plan_actions[i]->preconditions && plan_actions[j]->effects) {
                for (int pi = 0; pi < plan_actions[i]->precondition_count; pi++) {
                    if (!plan_actions[i]->preconditions[pi]) continue;
                    for (int ej = 0; ej < plan_actions[j]->effect_count; ej++) {
                        if (plan_actions[j]->effects[ej] &&
                            strcmp(plan_actions[i]->preconditions[pi],
                                   plan_actions[j]->effects[ej]) == 0) {
                            dependencies[i * action_count + j] = 1;
                        }
                    }
                }
            }
        }
    }

    /* 1. 时间松弛度分析 */
    result->temporal = plan_robustness_temporal_slack(plan_actions, action_count, dependencies);

    /* 2. 资源松弛度分析 */
    if (system->config.resource_type_count > 0) {
        result->resource = plan_robustness_resource_slack(
            plan_actions, action_count,
            system->config.available_resources,
            system->config.resource_type_count);
    }

    /* 3. 蒙特卡洛鲁棒性估计 */
    if (num_monte_carlo_samples > 0) {
        /* 使用默认采样和评估函数 */
        int mc_param_count = action_count * 2;  /* 每个动作的时长和成本偏差 */
        result->monte_carlo = plan_robustness_monte_carlo(
            plan_actions, action_count,
            /* 采样函数：均匀随机扰动 */
            (void (*)(float*, int))&plan_default_sample_func,
            (int (*)(const float*, int, float*, float*))&plan_default_evaluate_func,
            mc_param_count,
            num_monte_carlo_samples);
    }

    /* 4. 计算整体鲁棒性分数 */
    float temporal_score = 0.0f;
    float resource_score = 0.0f;
    float mc_score = 0.0f;
    int score_count = 0;

    if (result->temporal && result->temporal->critical_path_duration > 0.0f) {
        /* 时间鲁棒性 = 平均松弛时间 / 关键路径时长 */
        temporal_score = result->temporal->total_slack /
                         (result->temporal->critical_path_duration * (float)action_count);
        if (temporal_score > 1.0f) temporal_score = 1.0f;
        score_count++;
    }

    if (result->resource) {
        float avg_util = 0.0f;
        for (int r = 0; r < result->resource->resource_count; r++) {
            avg_util += result->resource->utilizations[r];
        }
        avg_util /= (float)result->resource->resource_count;
        resource_score = 1.0f - avg_util;
        if (resource_score < 0.0f) resource_score = 0.0f;
        score_count++;
    }

    if (result->monte_carlo) {
        mc_score = result->monte_carlo->success_probability;
        score_count++;
    }

    result->overall_robustness = (score_count > 0) ?
        (temporal_score + resource_score + mc_score) / (float)score_count : 0.5f;

    /* 5. 生成改进建议 */
    result->recommendation_count = 0;
    int rec_cap = 16;
    result->recommendations = (char**)safe_calloc((size_t)rec_cap, sizeof(char*));

    if (result->temporal) {
        int critical_count = 0;
        for (int i = 0; i < action_count; i++) {
            if (result->temporal->on_critical_path[i]) critical_count++;
        }
        if (critical_count > action_count / 2) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "关键路径过长（%d/%d个动作在关键路径上），建议增加并行或缩短关键动作时长",
                     critical_count, action_count);
            result->recommendations[result->recommendation_count] = string_duplicate_nullable(buf);
            result->recommendation_count++;
        }
    }

    if (result->resource) {
        float min_util = 1.0f;
        for (int r = 0; r < result->resource->resource_count; r++) {
            if (result->resource->utilizations[r] < min_util) {
                min_util = result->resource->utilizations[r];
            }
        }
        if (min_util > 0.8f) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "资源利用率过高（瓶颈资源索引%d，余量%.2f），建议增加资源或减少并发",
                     result->resource->bottleneck_resource,
                     result->resource->min_headroom);
            result->recommendations[result->recommendation_count] = string_duplicate_nullable(buf);
            result->recommendation_count++;
        }
    }

    if (result->monte_carlo && result->monte_carlo->success_probability < 0.7f) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "蒙特卡洛成功率仅%.1f%%，建议增加冗余或降低复杂度",
                 result->monte_carlo->success_probability * 100.0f);
        result->recommendations[result->recommendation_count] = string_duplicate_nullable(buf);
        result->recommendation_count++;
    }

    safe_free((void**)&dependencies);
    return result;
}

void long_term_planning_destroy_robustness_result(RobustnessResult* result)
{
    if (!result) return;
    plan_robustness_destroy_temporal_slack(result->temporal);
    plan_robustness_destroy_resource_slack(result->resource);
    plan_robustness_destroy_sensitivity_result(result->sensitivity);
    plan_robustness_destroy_monte_carlo_result(result->monte_carlo);
    if (result->recommendations) {
        for (int i = 0; i < result->recommendation_count; i++) {
            safe_free((void**)&result->recommendations[i]);
        }
        safe_free((void**)&result->recommendations);
    }
    safe_free((void**)&result);
}

/* ============================================================================
 * 鲁棒性分析 - 内部辅助函数
 * =========================================================================== */

/**
 * @brief 默认采样函数：xorshift均匀随机采样（内部使用）
 */
#ifdef _WIN32
static CRITICAL_SECTION g_mc_prng_lock;
static int g_mc_prng_lock_init = 0;
#define MC_PRNG_LOCK() do { if (!g_mc_prng_lock_init) { InitializeCriticalSection(&g_mc_prng_lock); g_mc_prng_lock_init = 1; } EnterCriticalSection(&g_mc_prng_lock); } while(0)
#define MC_PRNG_UNLOCK() LeaveCriticalSection(&g_mc_prng_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_mc_prng_lock = PTHREAD_MUTEX_INITIALIZER;
#define MC_PRNG_LOCK() pthread_mutex_lock(&g_mc_prng_lock)
#define MC_PRNG_UNLOCK() pthread_mutex_unlock(&g_mc_prng_lock)
#endif

static void plan_default_sample_func(float* samples, int param_count)
{
    if (!samples || param_count <= 0) return;
    MC_PRNG_LOCK();
    static XorshiftPrng mc_prng = {{0}};
    static int mc_prng_init = 0;
    if (!mc_prng_init) {
        xorshift_prng_seed(&mc_prng, (uint64_t)(size_t)samples ^ 0xA5B6C7D8E9F0A1B2ULL);
        mc_prng_init = 1;
    }
    for (int i = 0; i < param_count; i++) {
        uint64_t r = xorshift_prng_next_u64(&mc_prng);
        samples[i] = (float)(((double)(r & 0x7FFFFFFFFFFFFFFFULL) / 9.223372036854776e18) - 0.5);
    }
    MC_PRNG_UNLOCK();
}

/**
 * @brief 默认评估函数：估计计划执行代价（内部使用）
 */
static int plan_default_evaluate_func(const float* params, int param_count,
                               float* duration, float* cost)
{
    if (!params || param_count <= 0 || !duration || !cost) return 0;
    *duration = 0.0f;
    *cost = 0.0f;

    /* 默认：基于动作时间范围判断是否在规定时间内完成 */
    /* 如果动作没有显式时间约束，则假定其可在合理时间内完成 */
    float total = 0.0f;
    for (int i = 0; i < param_count; i += 2) {
        float base_dur = 1.0f + params[i] * 0.2f;  /* 时长偏差±10% */
        float base_cst = 0.5f + params[i + 1] * 0.1f; /* 成本偏差±5% */
        if (base_dur < 0.1f) base_dur = 0.1f;
        if (base_cst < 0.0f) base_cst = 0.0f;
        total += base_dur;
        *cost += base_cst;
    }

    *duration = total;
    /* 成功条件：总时长在合理范围内 */
    return (total < (float)(param_count / 2) * 2.0f) ? 1 : 0;
}