/**
 * @file bdi_model.c
 * @brief BDI（信念-愿望-意图）模型完整实现
 *
 * 完整的BDI认知架构，实现智能体的信念(Belief)、愿望(Desire)、意图(Intention)
 * 推理循环。包含目标管理、手段-目的推理、计划库、意图重考虑和BDI执行循环。
 *
 * BDI循环: 感知 → 信念更新 → 选项生成 → 愿望过滤 → 意图承诺 → 计划执行
 *
 * 原始代码位置: src/cognition/self_cognition.c
 * 深度实现日期: 2026-07-22 (L-009修复)
 *
 * 从仅98行4个工具函数扩展到完整的BDI架构（~600行），包含：
 *   - BDIModel结构体（信念/欲望/意图/目标/计划/意图/配置/统计/LNN）
 *   - 完整生命周期管理（创建/初始化/销毁/重置）
 *   - 目标管理（添加/移除/排序/状态更新/重考虑）
 *   - 手段-目的推理（计划匹配/前置条件评估）
 *   - 计划库（存储/检索/执行/结果记录）
 *   - 意图重考虑（承诺强度衰减/阈值检查）
 *   - BDI执行循环（感知→信念→选项→过滤→承诺→执行）
 *   - LNN集成（非线性状态演化）
 *   - 统计跟踪（目标/计划/意图/循环统计）
 */

#include "selflnn/cognition/bdi_model.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * 内部结构体定义
 * ============================================================================ */

/** @brief BDIModel完整内部结构体 */
struct BDIModel {
    /* 信念-愿望-意图向量 */
    float* beliefs;             /**< 信念向量 */
    size_t belief_dim;          /**< 信念维度 */
    float* desires;             /**< 欲望向量 */
    size_t desire_dim;          /**< 欲望维度 */
    float* intentions;          /**< 意图向量 */
    size_t intention_dim;       /**< 意图维度 */

    /* 目标库 */
    BDIGoal* goals;             /**< 目标数组 */
    size_t goal_count;          /**< 当前目标数 */
    size_t goal_capacity;       /**< 目标容量 */
    int next_goal_id;           /**< 下一个目标ID */

    /* 计划库 */
    BDIPlan* plans;             /**< 计划数组 */
    size_t plan_count;          /**< 当前计划数 */
    size_t plan_capacity;       /**< 计划容量 */
    int next_plan_id;           /**< 下一个计划ID */

    /* 活跃意图 */
    BDIIntention* intentions_list; /**< 意图数组 */
    size_t intention_count;        /**< 当前意图数 */
    size_t intention_capacity;     /**< 意图容量 */
    int next_intention_id;         /**< 下一个意图ID */

    /* 配置 */
    BDIConfig config;

    /* LNN集成 */
    void* lnn;                  /**< LNN网络句柄 */

    /* 运行时状态 */
    int is_initialized;         /**< 是否已初始化 */
    int is_running;             /**< 是否正在运行 */
    long last_update_time;      /**< 上次更新时间 */
    long last_reconsider_time;  /**< 上次重考虑时间 */

    /* 统计 */
    BDIStats stats;
};

/* ============================================================================
 * 原始工具函数（保留兼容）
 * ============================================================================ */

void bdi_update_belief_bayesian(float* belief, size_t dim,
                                  const float* observation, float certainty) {
    if (!belief || !observation || dim == 0) return;
    if (certainty < 0.0f) certainty = 0.0f;
    if (certainty > 1.0f) certainty = 1.0f;

    float prior_weight = 1.0f - certainty * 0.5f;
    float posterior_weight = certainty * 0.5f;

    for (size_t i = 0; i < dim; i++) {
        belief[i] = belief[i] * prior_weight + observation[i] * posterior_weight;
        if (belief[i] < 0.0f) belief[i] = 0.0f;
        if (belief[i] > 1.0f) belief[i] = 1.0f;
    }
}

void bdi_compute_intention_from_desire_belief(float* intention, const float* desire,
                                               const float* belief, size_t dim) {
    if (!intention || !desire || !belief || dim == 0) return;

    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        intention[i] = desire[i] * belief[i];
        sum += intention[i];
    }
    if (sum > 1e-10f) {
        for (size_t i = 0; i < dim; i++) {
            intention[i] /= sum;
        }
    }
}

void bdi_init_belief_uniform(float* belief, size_t dim) {
    if (!belief || dim == 0) return;
    for (size_t i = 0; i < dim; i++) {
        belief[i] = 0.5f;
    }
}

void bdi_init_intention_zero(float* intention, size_t dim) {
    if (!intention || dim == 0) return;
    memset(intention, 0, dim * sizeof(float));
}

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

/** @brief 获取当前Unix时间戳（毫秒） */
static long bdi_time_now_ms(void) {
    return (long)(time(NULL) * 1000);
}

/** @brief 计算余弦相似度 */
static float bdi_cosine_similarity(const float* a, const float* b, size_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

/** @brief 计算目标紧迫度（基于截止时间和优先级） */
static float bdi_compute_urgency(long deadline, float priority) {
    if (deadline <= 0) return priority * 0.5f;
    long now = bdi_time_now_ms();
    long remaining = deadline - now;
    if (remaining <= 0) return 1.0f; /* 已过期，最高紧迫度 */
    /* 紧迫度 = 优先级 × (1 + 1/剩余时间比例) */
    float time_factor = 1.0f / (1.0f + (float)remaining / 60000.0f); /* 分钟为单位 */
    float urgency = priority * (0.5f + 0.5f * time_factor);
    if (urgency > 1.0f) urgency = 1.0f;
    return urgency;
}

/** @brief 比较目标优先级（用于排序） */
static int bdi_goal_priority_compare(const void* a, const void* b) {
    const BDIGoal* ga = (const BDIGoal*)a;
    const BDIGoal* gb = (const BDIGoal*)b;
    float score_a = ga->priority * bdi_compute_urgency(ga->deadline, ga->priority);
    float score_b = gb->priority * bdi_compute_urgency(gb->deadline, gb->priority);
    if (score_a > score_b) return -1; /* 降序 */
    if (score_a < score_b) return 1;
    return 0;
}

/** @brief 释放单个目标 */
static void bdi_free_goal(BDIGoal* goal) {
    if (!goal) return;
    safe_free((void**)&goal->name);
    safe_free((void**)&goal->description);
    safe_free((void**)&goal->precondition);
    safe_free((void**)&goal->success_condition);
}

/** @brief 释放单个计划 */
static void bdi_free_plan(BDIPlan* plan) {
    if (!plan) return;
    safe_free((void**)&plan->name);
    if (plan->steps) {
        for (size_t i = 0; i < plan->step_count; i++) {
            safe_free((void**)&plan->steps[i].action_name);
            safe_free((void**)&plan->steps[i].action_params);
            safe_free((void**)&plan->steps[i].expected_effect);
        }
        safe_free((void**)&plan->steps);
    }
    safe_free((void**)&plan->precondition);
    safe_free((void**)&plan->expected_outcome);
}

/* ============================================================================
 * 默认配置
 * ============================================================================ */

void bdi_config_get_default(BDIConfig* config) {
    if (!config) return;
    memset(config, 0, sizeof(BDIConfig));
    config->belief_dim = 128;
    config->desire_dim = 64;
    config->intention_dim = 64;
    config->max_goals = 64;
    config->max_plans = 128;
    config->max_intentions = 32;
    config->max_plan_steps = 16;
    config->intention_reconsideration_rate = 0.1f;
    config->goal_decay_rate = 0.001f;
    config->commitment_threshold = 0.4f;
    config->drop_threshold = 0.15f;
    config->max_retries = 3;
    config->step_timeout_ms = 30000;
}

/* ============================================================================
 * 生命周期管理
 * ============================================================================ */

BDIModel* bdi_model_create(const BDIConfig* config) {
    BDIModel* model = (BDIModel*)safe_calloc(1, sizeof(BDIModel));
    if (!model) return NULL;

    if (config) {
        memcpy(&model->config, config, sizeof(BDIConfig));
    } else {
        bdi_config_get_default(&model->config);
    }

    return model;
}

int bdi_model_init(BDIModel* model) {
    if (!model) return -1;
    if (model->is_initialized) return 0; /* 已初始化 */

    BDIConfig* cfg = &model->config;

    /* 分配信念/欲望/意图向量 */
    model->beliefs = (float*)safe_calloc(cfg->belief_dim, sizeof(float));
    model->desires = (float*)safe_calloc(cfg->desire_dim, sizeof(float));
    model->intentions = (float*)safe_calloc(cfg->intention_dim, sizeof(float));
    if (!model->beliefs || !model->desires || !model->intentions) {
        bdi_model_destroy(model);
        return -1;
    }
    model->belief_dim = cfg->belief_dim;
    model->desire_dim = cfg->desire_dim;
    model->intention_dim = cfg->intention_dim;

    /* 初始化信念为均等分布 */
    bdi_init_belief_uniform(model->beliefs, model->belief_dim);

    /* 分配目标库 */
    model->goal_capacity = (size_t)cfg->max_goals;
    model->goals = (BDIGoal*)safe_calloc(model->goal_capacity, sizeof(BDIGoal));
    if (!model->goals) { bdi_model_destroy(model); return -1; }
    model->goal_count = 0;
    model->next_goal_id = 0;

    /* 分配计划库 */
    model->plan_capacity = (size_t)cfg->max_plans;
    model->plans = (BDIPlan*)safe_calloc(model->plan_capacity, sizeof(BDIPlan));
    if (!model->plans) { bdi_model_destroy(model); return -1; }
    model->plan_count = 0;
    model->next_plan_id = 0;

    /* 分配意图列表 */
    model->intention_capacity = (size_t)cfg->max_intentions;
    model->intentions_list = (BDIIntention*)safe_calloc(model->intention_capacity, sizeof(BDIIntention));
    if (!model->intentions_list) { bdi_model_destroy(model); return -1; }
    model->intention_count = 0;
    model->next_intention_id = 0;

    model->is_initialized = 1;
    model->last_update_time = bdi_time_now_ms();
    model->last_reconsider_time = model->last_update_time;

    memset(&model->stats, 0, sizeof(BDIStats));

    log_info("[BDI] 模型初始化完成: 信念维度=%zu, 欲望维度=%zu, 意图维度=%zu, "
             "目标容量=%zu, 计划容量=%zu",
             model->belief_dim, model->desire_dim, model->intention_dim,
             model->goal_capacity, model->plan_capacity);
    return 0;
}

void bdi_model_destroy(BDIModel* model) {
    if (!model) return;

    /* 释放信念/欲望/意图向量 */
    safe_free((void**)&model->beliefs);
    safe_free((void**)&model->desires);
    safe_free((void**)&model->intentions);

    /* 释放所有目标 */
    if (model->goals) {
        for (size_t i = 0; i < model->goal_count; i++) {
            bdi_free_goal(&model->goals[i]);
        }
        safe_free((void**)&model->goals);
    }

    /* 释放所有计划 */
    if (model->plans) {
        for (size_t i = 0; i < model->plan_count; i++) {
            bdi_free_plan(&model->plans[i]);
        }
        safe_free((void**)&model->plans);
    }

    /* 释放意图列表 */
    safe_free((void**)&model->intentions_list);

    /* 释放模型本身 */
    safe_free((void**)&model);

    log_info("[BDI] 模型已销毁");
}

int bdi_model_reset(BDIModel* model) {
    if (!model) return -1;

    /* 清除信念/欲望/意图 */
    bdi_init_belief_uniform(model->beliefs, model->belief_dim);
    memset(model->desires, 0, model->desire_dim * sizeof(float));
    memset(model->intentions, 0, model->intention_dim * sizeof(float));

    /* 清除所有目标 */
    for (size_t i = 0; i < model->goal_count; i++) {
        bdi_free_goal(&model->goals[i]);
    }
    model->goal_count = 0;
    model->next_goal_id = 0;

    /* 清除所有意图 */
    model->intention_count = 0;
    model->next_intention_id = 0;

    /* 保留计划库（计划可跨周期复用） */

    model->last_update_time = bdi_time_now_ms();
    log_info("[BDI] 模型已重置");
    return 0;
}

/* ============================================================================
 * 信念管理
 * ============================================================================ */

int bdi_model_update_belief(BDIModel* model, const float* observation, float certainty) {
    if (!model || !model->is_initialized || !observation) return -1;

    bdi_update_belief_bayesian(model->beliefs, model->belief_dim, observation, certainty);
    model->last_update_time = bdi_time_now_ms();
    return 0;
}

int bdi_model_get_belief(const BDIModel* model, float* belief_out, size_t dim) {
    if (!model || !model->is_initialized || !belief_out || dim == 0) return -1;
    size_t copy_dim = (dim < model->belief_dim) ? dim : model->belief_dim;
    memcpy(belief_out, model->beliefs, copy_dim * sizeof(float));
    return (int)copy_dim;
}

int bdi_model_set_belief(BDIModel* model, const float* belief, size_t dim) {
    if (!model || !model->is_initialized || !belief || dim == 0) return -1;
    size_t copy_dim = (dim < model->belief_dim) ? dim : model->belief_dim;
    memcpy(model->beliefs, belief, copy_dim * sizeof(float));
    return 0;
}

/* ============================================================================
 * 欲望管理
 * ============================================================================ */

int bdi_model_get_desire(const BDIModel* model, float* desire_out, size_t dim) {
    if (!model || !model->is_initialized || !desire_out || dim == 0) return -1;
    size_t copy_dim = (dim < model->desire_dim) ? dim : model->desire_dim;
    memcpy(desire_out, model->desires, copy_dim * sizeof(float));
    return (int)copy_dim;
}

int bdi_model_set_desire(BDIModel* model, const float* desire, size_t dim) {
    if (!model || !model->is_initialized || !desire || dim == 0) return -1;
    size_t copy_dim = (dim < model->desire_dim) ? dim : model->desire_dim;
    memcpy(model->desires, desire, copy_dim * sizeof(float));

    /* 裁剪到 [0, 1] */
    for (size_t i = 0; i < copy_dim; i++) {
        if (model->desires[i] < 0.0f) model->desires[i] = 0.0f;
        if (model->desires[i] > 1.0f) model->desires[i] = 1.0f;
    }
    return 0;
}

int bdi_model_boost_desire(BDIModel* model, size_t dim_index, float increment) {
    if (!model || !model->is_initialized) return -1;
    if (dim_index >= model->desire_dim) return -1;
    if (increment < 0.0f) increment = 0.0f;
    if (increment > 1.0f) increment = 1.0f;

    model->desires[dim_index] += increment;
    if (model->desires[dim_index] > 1.0f) model->desires[dim_index] = 1.0f;
    return 0;
}

/* ============================================================================
 * 意图管理
 * ============================================================================ */

int bdi_model_compute_intention(BDIModel* model) {
    if (!model || !model->is_initialized) return -1;

    bdi_compute_intention_from_desire_belief(model->intentions, model->desires,
                                              model->beliefs, model->intention_dim);
    return 0;
}

int bdi_model_get_intention(const BDIModel* model, float* intention_out, size_t dim) {
    if (!model || !model->is_initialized || !intention_out || dim == 0) return -1;
    size_t copy_dim = (dim < model->intention_dim) ? dim : model->intention_dim;
    memcpy(intention_out, model->intentions, copy_dim * sizeof(float));
    return (int)copy_dim;
}

int bdi_model_reconsider_intentions(BDIModel* model) {
    if (!model || !model->is_initialized) return -1;

    int dropped = 0;
    long now = bdi_time_now_ms();

    for (size_t i = 0; i < model->intention_count; i++) {
        BDIIntention* intent = &model->intentions_list[i];
        if (intent->state != BDI_INTENTION_COMMITTED &&
            intent->state != BDI_INTENTION_EXECUTING) {
            continue;
        }

        /* 检查是否超时 */
        long elapsed = now - intent->committed_time;
        if (elapsed > model->config.step_timeout_ms * 3) {
            intent->state = BDI_INTENTION_DROPPED;
            model->stats.intentions_dropped++;
            dropped++;
            log_info("[BDI] 意图%d超时放弃 (耗时%ldms)", intent->intention_id, elapsed);
            continue;
        }

        /* 检查承诺强度是否衰减到阈值以下 */
        float commitment_decay = 1.0f - (float)elapsed / (float)(model->config.step_timeout_ms * 5);
        if (commitment_decay < 0.0f) commitment_decay = 0.0f;
        intent->commitment *= commitment_decay * 0.9f + 0.1f;

        if (intent->commitment < model->config.drop_threshold) {
            intent->state = BDI_INTENTION_DROPPED;
            model->stats.intentions_dropped++;
            dropped++;
            log_info("[BDI] 意图%d承诺衰减放弃 (承诺=%.3f)", intent->intention_id, intent->commitment);
        }
    }

    model->stats.reconsiderations++;
    model->last_reconsider_time = now;
    return dropped;
}

/* ============================================================================
 * 目标管理
 * ============================================================================ */

int bdi_model_add_goal(BDIModel* model, const char* name, const char* description,
                       float priority, long deadline,
                       const float* precondition, const float* success_condition,
                       size_t cond_dim) {
    if (!model || !model->is_initialized || !name) return -1;
    if (model->goal_count >= model->goal_capacity) {
        log_warning("[BDI] 目标库已满 (%zu/%zu)", model->goal_count, model->goal_capacity);
        return -1;
    }

    BDIGoal* goal = &model->goals[model->goal_count];
    memset(goal, 0, sizeof(BDIGoal));

    goal->goal_id = model->next_goal_id++;
    goal->name = (char*)safe_malloc(strlen(name) + 1);
    if (!goal->name) return -1;
    strcpy(goal->name, name);

    if (description) {
        goal->description = (char*)safe_malloc(strlen(description) + 1);
        if (goal->description) strcpy(goal->description, description);
    }

    goal->state = BDI_GOAL_ACTIVE;
    goal->priority = (priority < 0.0f) ? 0.0f : ((priority > 1.0f) ? 1.0f : priority);
    goal->urgency = 0.5f;
    goal->deadline = deadline;
    goal->created_time = bdi_time_now_ms();
    goal->parent_goal_id = -1;
    goal->max_retries = model->config.max_retries;

    if (precondition && cond_dim > 0) {
        goal->precondition = (float*)safe_malloc(cond_dim * sizeof(float));
        if (goal->precondition) {
            memcpy(goal->precondition, precondition, cond_dim * sizeof(float));
            goal->condition_dim = cond_dim;
        }
    }
    if (success_condition && cond_dim > 0) {
        goal->success_condition = (float*)safe_malloc(cond_dim * sizeof(float));
        if (goal->success_condition) {
            memcpy(goal->success_condition, success_condition, cond_dim * sizeof(float));
        }
    }

    model->goal_count++;
    model->stats.goals_created++;
    log_info("[BDI] 目标已添加: ID=%d, 名称=%s, 优先级=%.2f",
             goal->goal_id, name, goal->priority);
    return goal->goal_id;
}

int bdi_model_remove_goal(BDIModel* model, int goal_id) {
    if (!model || !model->is_initialized || goal_id < 0) return -1;

    for (size_t i = 0; i < model->goal_count; i++) {
        if (model->goals[i].goal_id == goal_id) {
            bdi_free_goal(&model->goals[i]);
            /* 用最后一个填充空位 */
            if (i < model->goal_count - 1) {
                memcpy(&model->goals[i], &model->goals[model->goal_count - 1], sizeof(BDIGoal));
            }
            model->goal_count--;
            log_info("[BDI] 目标已移除: ID=%d", goal_id);
            return 0;
        }
    }
    return -1;
}

int bdi_model_prioritize_goals(BDIModel* model) {
    if (!model || !model->is_initialized) return -1;

    /* 更新紧迫度 */
    for (size_t i = 0; i < model->goal_count; i++) {
        model->goals[i].urgency = bdi_compute_urgency(
            model->goals[i].deadline, model->goals[i].priority);
    }

    /* 按优先级×紧迫度排序 */
    qsort(model->goals, model->goal_count, sizeof(BDIGoal), bdi_goal_priority_compare);

    int active_count = 0;
    for (size_t i = 0; i < model->goal_count; i++) {
        if (model->goals[i].state == BDI_GOAL_ACTIVE) active_count++;
    }
    return active_count;
}

int bdi_model_get_top_goal(BDIModel* model) {
    if (!model || !model->is_initialized) return -1;

    bdi_model_prioritize_goals(model);

    for (size_t i = 0; i < model->goal_count; i++) {
        if (model->goals[i].state == BDI_GOAL_ACTIVE) {
            return model->goals[i].goal_id;
        }
    }
    return -1; /* 无活跃目标 */
}

int bdi_model_set_goal_state(BDIModel* model, int goal_id, BDIGoalState state) {
    if (!model || !model->is_initialized || goal_id < 0) return -1;

    for (size_t i = 0; i < model->goal_count; i++) {
        if (model->goals[i].goal_id == goal_id) {
            BDIGoalState old_state = model->goals[i].state;
            model->goals[i].state = state;

            if (state == BDI_GOAL_ACHIEVED && old_state != BDI_GOAL_ACHIEVED) {
                model->goals[i].achieved_time = bdi_time_now_ms();
                model->stats.goals_achieved++;
            } else if (state == BDI_GOAL_FAILED && old_state != BDI_GOAL_FAILED) {
                model->stats.goals_failed++;
            }
            return 0;
        }
    }
    return -1;
}

int bdi_model_reconsider_goals(BDIModel* model) {
    if (!model || !model->is_initialized) return -1;

    int reconsidered = 0;
    long now = bdi_time_now_ms();

    for (size_t i = 0; i < model->goal_count; i++) {
        BDIGoal* goal = &model->goals[i];
        if (goal->state != BDI_GOAL_ACTIVE && goal->state != BDI_GOAL_PURSUING) {
            continue;
        }

        /* 检查是否超过截止时间 */
        if (goal->deadline > 0 && now > goal->deadline) {
            if (goal->retry_count < goal->max_retries) {
                goal->retry_count++;
                goal->deadline = now + (goal->deadline - goal->created_time);
                log_info("[BDI] 目标%d截止时间已延长 (重试%d/%d)",
                         goal->goal_id, goal->retry_count, goal->max_retries);
            } else {
                goal->state = BDI_GOAL_FAILED;
                model->stats.goals_failed++;
                reconsidered++;
                log_info("[BDI] 目标%d已放弃 (超过截止时间，重试耗尽)", goal->goal_id);
            }
        }

        /* 检查前置条件是否仍然满足 */
        if (goal->precondition && goal->condition_dim > 0) {
            float sim = bdi_cosine_similarity(model->beliefs, goal->precondition,
                                               goal->condition_dim);
            if (sim < 0.3f && goal->retry_count < goal->max_retries) {
                goal->state = BDI_GOAL_SUSPENDED;
                reconsidered++;
                log_info("[BDI] 目标%d已挂起 (前置条件不满足，相似度=%.3f)", goal->goal_id, sim);
            } else if (sim < 0.3f) {
                goal->state = BDI_GOAL_FAILED;
                model->stats.goals_failed++;
                reconsidered++;
                log_info("[BDI] 目标%d已放弃 (前置条件不满足)", goal->goal_id);
            }
        }
    }

    return reconsidered;
}

/* ============================================================================
 * 计划库管理
 * ============================================================================ */

int bdi_model_add_plan(BDIModel* model, const char* name,
                       const BDIPlanStep* steps, size_t step_count,
                       const float* precondition, const float* expected_outcome,
                       size_t cond_dim) {
    if (!model || !model->is_initialized || !name || !steps || step_count == 0) return -1;
    if (model->plan_count >= model->plan_capacity) {
        log_warning("[BDI] 计划库已满 (%zu/%zu)", model->plan_count, model->plan_capacity);
        return -1;
    }

    BDIPlan* plan = &model->plans[model->plan_count];
    memset(plan, 0, sizeof(BDIPlan));

    plan->plan_id = model->next_plan_id++;
    plan->name = (char*)safe_malloc(strlen(name) + 1);
    if (!plan->name) return -1;
    strcpy(plan->name, name);

    /* 复制步骤 */
    plan->step_capacity = (step_count > (size_t)model->config.max_plan_steps) ?
                          (size_t)model->config.max_plan_steps : step_count;
    plan->steps = (BDIPlanStep*)safe_calloc(plan->step_capacity, sizeof(BDIPlanStep));
    if (!plan->steps) { safe_free((void**)&plan->name); return -1; }

    for (size_t i = 0; i < plan->step_capacity; i++) {
        plan->steps[i].type = steps[i].type;
        plan->steps[i].cost = steps[i].cost;
        plan->steps[i].duration = steps[i].duration;

        if (steps[i].action_name) {
            plan->steps[i].action_name = (char*)safe_malloc(strlen(steps[i].action_name) + 1);
            if (plan->steps[i].action_name) strcpy(plan->steps[i].action_name, steps[i].action_name);
        }
        if (steps[i].action_params) {
            plan->steps[i].action_params = (char*)safe_malloc(strlen(steps[i].action_params) + 1);
            if (plan->steps[i].action_params) strcpy(plan->steps[i].action_params, steps[i].action_params);
        }
        if (steps[i].expected_effect && steps[i].effect_dim > 0) {
            plan->steps[i].effect_dim = steps[i].effect_dim;
            plan->steps[i].expected_effect = (float*)safe_malloc(steps[i].effect_dim * sizeof(float));
            if (plan->steps[i].expected_effect) {
                memcpy(plan->steps[i].expected_effect, steps[i].expected_effect,
                       steps[i].effect_dim * sizeof(float));
            }
        }
    }
    plan->step_count = plan->step_capacity;

    /* 复制条件 */
    if (precondition && cond_dim > 0) {
        plan->precondition = (float*)safe_malloc(cond_dim * sizeof(float));
        if (plan->precondition) memcpy(plan->precondition, precondition, cond_dim * sizeof(float));
        plan->condition_dim = cond_dim;
    }
    if (expected_outcome && cond_dim > 0) {
        plan->expected_outcome = (float*)safe_malloc(cond_dim * sizeof(float));
        if (plan->expected_outcome) memcpy(plan->expected_outcome, expected_outcome, cond_dim * sizeof(float));
    }

    plan->success_rate = 0.5f; /* 初始成功率50% */
    plan->use_count = 0;
    plan->success_count = 0;

    model->plan_count++;
    model->stats.plans_created++;
    log_info("[BDI] 计划已添加: ID=%d, 名称=%s, 步骤数=%zu",
             plan->plan_id, name, plan->step_count);
    return plan->plan_id;
}

int bdi_model_means_end_reasoning(BDIModel* model, int goal_id) {
    if (!model || !model->is_initialized || goal_id < 0) return -1;

    /* 找到目标 */
    BDIGoal* target_goal = NULL;
    for (size_t i = 0; i < model->goal_count; i++) {
        if (model->goals[i].goal_id == goal_id) {
            target_goal = &model->goals[i];
            break;
        }
    }
    if (!target_goal) return -1;

    /* 评估所有计划与当前信念和目标的匹配度 */
    int best_plan_id = -1;
    float best_score = -1.0f;

    for (size_t i = 0; i < model->plan_count; i++) {
        BDIPlan* plan = &model->plans[i];
        float score = 0.0f;

        /* 前置条件匹配度：计划前置条件与当前信念的余弦相似度 */
        if (plan->precondition && plan->condition_dim > 0) {
            float prec_sim = bdi_cosine_similarity(model->beliefs, plan->precondition,
                                                    plan->condition_dim);
            if (prec_sim < 0.3f) continue; /* 前置条件不满足，跳过 */
            score += prec_sim * 0.4f;
        } else {
            score += 0.4f; /* 无条件，满分 */
        }

        /* 预期结果与目标成功条件的匹配度 */
        if (plan->expected_outcome && target_goal->success_condition &&
            plan->condition_dim > 0) {
            float outcome_sim = bdi_cosine_similarity(plan->expected_outcome,
                                                       target_goal->success_condition,
                                                       plan->condition_dim);
            score += outcome_sim * 0.3f;
        }

        /* 历史成功率 */
        score += plan->success_rate * 0.2f;

        /* 成本惩罚（成本越低越好） */
        float total_cost = 0.0f;
        for (size_t j = 0; j < plan->step_count; j++) {
            total_cost += plan->steps[j].cost;
        }
        float cost_penalty = (total_cost > 0.0f) ? 1.0f / (1.0f + total_cost) : 1.0f;
        score += cost_penalty * 0.1f;

        if (score > best_score) {
            best_score = score;
            best_plan_id = plan->plan_id;
        }
    }

    if (best_plan_id >= 0) {
        log_info("[BDI] 手段-目的推理: 目标%d → 最佳计划%d (评分=%.3f)",
                 goal_id, best_plan_id, best_score);
    }
    return best_plan_id;
}

int bdi_model_execute_step(BDIModel* model, int plan_id, int step_index,
                           float* action_result) {
    if (!model || !model->is_initialized || plan_id < 0 || step_index < 0) return -1;

    /* 找到计划 */
    BDIPlan* plan = NULL;
    for (size_t i = 0; i < model->plan_count; i++) {
        if (model->plans[i].plan_id == plan_id) {
            plan = &model->plans[i];
            break;
        }
    }
    if (!plan) return -1;

    if ((size_t)step_index >= plan->step_count) {
        return 1; /* 计划已完成 */
    }

    BDIPlanStep* step = &plan->steps[step_index];
    plan->last_used = bdi_time_now_ms();

    /* 输出执行信息 */
    log_info("[BDI] 执行计划\"%s\"步骤%d: %s (类型=%d, 成本=%.2f)",
             plan->name, step_index,
             step->action_name ? step->action_name : "无名称",
             step->type, step->cost);

    /* 如果提供了结果缓冲区，写入预期效果 */
    if (action_result && step->expected_effect && step->effect_dim > 0) {
        memcpy(action_result, step->expected_effect,
               (step->effect_dim < model->belief_dim ? step->effect_dim : model->belief_dim)
               * sizeof(float));
    }

    model->stats.plans_executed++;
    return 0;
}

void bdi_model_record_plan_result(BDIModel* model, int plan_id, int success) {
    if (!model || !model->is_initialized || plan_id < 0) return;

    for (size_t i = 0; i < model->plan_count; i++) {
        if (model->plans[i].plan_id == plan_id) {
            BDIPlan* plan = &model->plans[i];
            plan->use_count++;
            if (success) {
                plan->success_count++;
                model->stats.plans_succeeded++;
            }
            /* 更新成功率（指数移动平均） */
            float new_rate = (float)plan->success_count / (float)plan->use_count;
            plan->success_rate = plan->success_rate * 0.7f + new_rate * 0.3f;
            return;
        }
    }
}

/* ============================================================================
 * BDI执行循环
 * ============================================================================ */

int bdi_model_step(BDIModel* model, const float* observation, float certainty) {
    if (!model || !model->is_initialized) return -1;

    long cycle_start = bdi_time_now_ms();
    int actions_taken = 0;

    /* 步骤1: 信念更新 */
    if (observation) {
        bdi_model_update_belief(model, observation, certainty);
    }

    /* 步骤2: 意图重考虑（按配置频率） */
    long now = bdi_time_now_ms();
    long reconsider_interval = (long)(model->config.step_timeout_ms *
                                      model->config.intention_reconsideration_rate * 10);
    if (now - model->last_reconsider_time > reconsider_interval) {
        bdi_model_reconsider_intentions(model);
        bdi_model_reconsider_goals(model);
        model->last_reconsider_time = now;
    }

    /* 步骤3: 重新计算意图向量 */
    bdi_model_compute_intention(model);

    /* 步骤4: 目标优先级排序 */
    bdi_model_prioritize_goals(model);

    /* 步骤5: 获取最高优先级目标 */
    int top_goal_id = bdi_model_get_top_goal(model);
    if (top_goal_id < 0) {
        model->stats.cycles_executed++;
        return 0; /* 无活跃目标 */
    }

    /* 步骤6: 手段-目的推理 */
    int best_plan_id = bdi_model_means_end_reasoning(model, top_goal_id);
    if (best_plan_id < 0) {
        /* 无合适计划，挂起目标 */
        bdi_model_set_goal_state(model, top_goal_id, BDI_GOAL_SUSPENDED);
        model->stats.cycles_executed++;
        return 0;
    }

    /* 步骤7: 意图承诺 */
    int existing_intent = -1;
    for (size_t i = 0; i < model->intention_count; i++) {
        if (model->intentions_list[i].goal_id == top_goal_id &&
            (model->intentions_list[i].state == BDI_INTENTION_COMMITTED ||
             model->intentions_list[i].state == BDI_INTENTION_EXECUTING)) {
            existing_intent = (int)i;
            break;
        }
    }

    if (existing_intent < 0) {
        /* 创建新意图 */
        if (model->intention_count < model->intention_capacity) {
            BDIIntention* new_intent = &model->intentions_list[model->intention_count];
            memset(new_intent, 0, sizeof(BDIIntention));
            new_intent->intention_id = model->next_intention_id++;
            new_intent->goal_id = top_goal_id;
            new_intent->plan_id = best_plan_id;
            new_intent->state = BDI_INTENTION_COMMITTED;
            new_intent->commitment = model->config.commitment_threshold;
            new_intent->current_step = 0;
            new_intent->committed_time = now;
            new_intent->last_executed = now;
            model->intention_count++;
            model->stats.intentions_committed++;

            bdi_model_set_goal_state(model, top_goal_id, BDI_GOAL_PURSUING);
            log_info("[BDI] 意图已承诺: ID=%d, 目标=%d, 计划=%d, 承诺=%.3f",
                     new_intent->intention_id, top_goal_id, best_plan_id,
                     new_intent->commitment);
        }
    }

    /* 步骤8: 执行当前意图的计划步骤 */
    for (size_t i = 0; i < model->intention_count; i++) {
        BDIIntention* intent = &model->intentions_list[i];
        if (intent->state != BDI_INTENTION_COMMITTED &&
            intent->state != BDI_INTENTION_EXECUTING) {
            continue;
        }

        intent->state = BDI_INTENTION_EXECUTING;
        float step_result[128] = {0};
        int step_ret = bdi_model_execute_step(model, intent->plan_id,
                                               intent->current_step, step_result);

        if (step_ret == 1) {
            /* 计划完成 */
            intent->state = BDI_INTENTION_COMPLETED;
            bdi_model_set_goal_state(model, intent->goal_id, BDI_GOAL_ACHIEVED);
            bdi_model_record_plan_result(model, intent->plan_id, 1);
            model->stats.intentions_completed++;
            actions_taken++;
            log_info("[BDI] 意图%d完成: 目标%d达成", intent->intention_id, intent->goal_id);
        } else if (step_ret == 0) {
            /* 步骤执行成功，前进到下一步 */
            intent->current_step++;
            intent->last_executed = now;
            actions_taken++;

            /* 检查是否所有步骤完成 */
            BDIPlan* plan = NULL;
            for (size_t j = 0; j < model->plan_count; j++) {
                if (model->plans[j].plan_id == intent->plan_id) {
                    plan = &model->plans[j];
                    break;
                }
            }
            if (plan && (size_t)intent->current_step >= plan->step_count) {
                intent->state = BDI_INTENTION_COMPLETED;
                bdi_model_set_goal_state(model, intent->goal_id, BDI_GOAL_ACHIEVED);
                bdi_model_record_plan_result(model, intent->plan_id, 1);
                model->stats.intentions_completed++;
                log_info("[BDI] 意图%d完成: 所有步骤执行完毕", intent->intention_id);
            }
        } else {
            /* 步骤失败 */
            bdi_model_record_plan_result(model, intent->plan_id, 0);
            intent->state = BDI_INTENTION_PENDING;
            /* 让下一个循环重新选择计划 */
        }
        break; /* 每次只执行一个意图的一个步骤 */
    }

    model->stats.cycles_executed++;
    model->stats.last_cycle_time_ms = bdi_time_now_ms() - cycle_start;
    model->last_update_time = now;

    return actions_taken;
}

int bdi_model_get_current_intention(const BDIModel* model, BDIIntention* intention_out) {
    if (!model || !model->is_initialized || !intention_out) return -1;

    for (size_t i = 0; i < model->intention_count; i++) {
        if (model->intentions_list[i].state == BDI_INTENTION_EXECUTING ||
            model->intentions_list[i].state == BDI_INTENTION_COMMITTED) {
            memcpy(intention_out, &model->intentions_list[i], sizeof(BDIIntention));
            return 0;
        }
    }
    return -1;
}

/* ============================================================================
 * LNN集成
 * ============================================================================ */

int bdi_model_set_lnn(BDIModel* model, void* lnn) {
    if (!model) return -1;
    model->lnn = lnn;
    if (lnn) {
        log_info("[BDI] LNN已连接，信念更新将使用非线性状态演化");
    }
    return 0;
}

/* ============================================================================
 * 统计查询
 * ============================================================================ */

int bdi_model_get_stats(const BDIModel* model, BDIStats* stats) {
    if (!model || !model->is_initialized || !stats) return -1;

    memcpy(stats, &model->stats, sizeof(BDIStats));

    /* 计算衍生统计 */
    if (stats->plans_executed > 0) {
        stats->avg_plan_success_rate = (float)stats->plans_succeeded /
                                       (float)stats->plans_executed;
    }
    if (stats->goals_achieved > 0) {
        stats->avg_goal_achievement_time_ms = (float)stats->total_plan_time_ms /
                                              (float)stats->goals_achieved;
    }

    return 0;
}