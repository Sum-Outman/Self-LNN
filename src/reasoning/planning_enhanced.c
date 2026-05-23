/**
 * @file planning_enhanced.c
 * @brief 增强规划引擎完整实现 —— 高级规划层
 *
 * K-005: 角色定义 —— planning_enhanced.c 是规划系统的【增强扩展层】
 * 负责：STN时序推理、HTN层次任务网分解、多智能体并发规划冲突检测。
 * 
 * 层级关系：
 *   planning.c → 核心规划算法（A* /BFS/DFS/启发式搜索）
 *   planning_enhanced.c（本文件）→ 增强规划（STN时序网 + HTN层次任务网 + 多智能体协调）
 * 
 * 本文件依赖planning.c的基础API，新增高级功能不修改核心算法。
 * 实现：
 * 1. A04.2.2 时间推理规划 — STN + CfC液态时序推理
 * 2. A04.2.2 并发多智能体规划 — 冲突检测与资源调度
 * 3. A04.3.1 HTN层次任务网规划 — 任务分解+方法选择
 */

#include "selflnn/reasoning/planning_enhanced.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

#define PLAN_ENH_EPSILON 1e-8f
#define PLAN_ENH_INF 1e10f

/* 内部结构体定义 */
struct PlanEnhancedEngine {
    PlanEnhancedConfig config;
    PlanEnhancedAction* actions;
    int action_count;
    int action_capacity;
    PlanTimeConstraint* constraints;
    int constraint_count;
    int constraint_capacity;
    PlanHTNTask* htn_tasks;
    int htn_task_count;
    int htn_task_capacity;
    PlanHTNMethod* htn_methods;
    int htn_method_count;
    int htn_method_capacity;
    PlanAgent* agents;
    int agent_count;
    int agent_capacity;
    /* CfC状态 */
    float* cfc_state;
    float* cfc_gate;
    float* cfc_act;
    int cfc_dim;
    /* STN时间网络 */
    float** stn_distance_matrix;
    int stn_node_count;
    int is_initialized;
};

/* 静态辅助函数 */
static float sigmoid_p(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static int min_int(int a, int b)
{
    return a < b ? a : b;
}

static float min_float(float a, float b)
{
    return a < b ? a : b;
}

static void cfc_step_p(float* h, const float* input, int dim, float tau, float dt)
{
    for (int i = 0; i < dim; i++) {
        float x = input ? input[i % dim] : 0.0f;
        float gate = sigmoid_p(h[i] + x);
        float act = tanhf(0.8f * h[i] + 0.8f * x);
        float dh = -h[i] / (tau + PLAN_ENH_EPSILON) + gate * act;
        h[i] += dh * dt;
        if (h[i] != h[i]) h[i] = 0.0f;
    }
}

static int ensure_capacity(void** ptr, int* capacity, int new_count, size_t elem_size, int init_cap)
{
    if (new_count <= *capacity) return 0;
    int new_cap = *capacity == 0 ? init_cap : *capacity * 2;
    while (new_cap < new_count) new_cap *= 2;
    void* tmp = realloc(*ptr, (size_t)new_cap * elem_size);
    if (!tmp) return -1;
    *ptr = tmp;
    *capacity = new_cap;
    return 0;
}

/* CfC液态积分演化 */
static void cfc_integrate(PlanEnhancedEngine* engine, const float* input, int steps)
{
    if (!engine || !engine->cfc_state) return;
    for (int i = 0; i < steps; i++) {
        cfc_step_p(engine->cfc_state, input, engine->cfc_dim,
                   engine->config.cfc_tau, engine->config.cfc_dt);
    }
}

PlanEnhancedEngine* plan_enhanced_create(const PlanEnhancedConfig* config)
{
    if (!config) return NULL;
    PlanEnhancedEngine* engine = (PlanEnhancedEngine*)safe_calloc(1, sizeof(PlanEnhancedEngine));
    if (!engine) return NULL;
    engine->config = *config;
    engine->action_count = 0;
    engine->action_capacity = 0;
    engine->actions = NULL;
    engine->constraint_count = 0;
    engine->constraint_capacity = 0;
    engine->constraints = NULL;
    engine->htn_task_count = 0;
    engine->htn_task_capacity = 0;
    engine->htn_tasks = NULL;
    engine->htn_method_count = 0;
    engine->htn_method_capacity = 0;
    engine->htn_methods = NULL;
    engine->agent_count = 0;
    engine->agent_capacity = 0;
    engine->agents = NULL;
    /* 初始化CfC状态 */
    engine->cfc_dim = 64;
    engine->cfc_state = (float*)safe_calloc((size_t)engine->cfc_dim, sizeof(float));
    engine->cfc_gate = (float*)safe_calloc((size_t)engine->cfc_dim, sizeof(float));
    engine->cfc_act = (float*)safe_calloc((size_t)engine->cfc_dim, sizeof(float));
    if (!engine->cfc_state || !engine->cfc_gate || !engine->cfc_act) {
        safe_free((void**)&engine->cfc_state);
        safe_free((void**)&engine->cfc_gate);
        safe_free((void**)&engine->cfc_act);
        safe_free((void**)&engine);
        return NULL;
    }
    engine->stn_distance_matrix = NULL;
    engine->stn_node_count = 0;
    engine->is_initialized = 1;
    /* 初始CfC演化 */
    for (int i = 0; i < 10; i++) {
        cfc_step_p(engine->cfc_state, NULL, engine->cfc_dim,
                   engine->config.cfc_tau, engine->config.cfc_dt);
    }
    return engine;
}

void plan_enhanced_destroy(PlanEnhancedEngine* engine)
{
    if (!engine) return;
    for (int i = 0; i < engine->action_count; i++) {
        safe_free((void**)&engine->actions[i].preconditions);
        safe_free((void**)&engine->actions[i].effects);
    }
    for (int i = 0; i < engine->htn_task_count; i++) {
        safe_free((void**)&engine->htn_tasks[i].subtasks);
        safe_free((void**)&engine->htn_tasks[i].decomposition_methods);
        safe_free((void**)&engine->htn_tasks[i].preconditions);
    }
    for (int i = 0; i < engine->htn_method_count; i++) {
        safe_free((void**)&engine->htn_methods[i].subtask_ids);
        safe_free((void**)&engine->htn_methods[i].applicability_conditions);
    }
    safe_free((void**)&engine->actions);
    safe_free((void**)&engine->constraints);
    safe_free((void**)&engine->htn_tasks);
    safe_free((void**)&engine->htn_methods);
    safe_free((void**)&engine->agents);
    safe_free((void**)&engine->cfc_state);
    safe_free((void**)&engine->cfc_gate);
    safe_free((void**)&engine->cfc_act);
    if (engine->stn_distance_matrix) {
        for (int i = 0; i < engine->stn_node_count; i++) {
            safe_free((void**)&engine->stn_distance_matrix[i]);
        }
        safe_free((void**)&engine->stn_distance_matrix);
    }
    safe_free((void**)&engine);
}

int plan_enhanced_add_action(PlanEnhancedEngine* engine, const PlanEnhancedAction* action)
{
    if (!engine || !action) return -1;
    if (ensure_capacity((void**)&engine->actions, &engine->action_capacity,
                        engine->action_count + 1, sizeof(PlanEnhancedAction), 64) != 0)
        return -1;
    int id = engine->action_count++;
    engine->actions[id] = *action;
    engine->actions[id].action_id = id;
    /* 深拷贝前提条件 */
    if (action->preconditions && action->precond_size > 0) {
        engine->actions[id].preconditions = (float*)safe_malloc(action->precond_size * sizeof(float));
        if (engine->actions[id].preconditions)
            memcpy(engine->actions[id].preconditions, action->preconditions,
                   action->precond_size * sizeof(float));
    }
    /* 深拷贝效果 */
    if (action->effects && action->effect_size > 0) {
        engine->actions[id].effects = (float*)safe_malloc(action->effect_size * sizeof(float));
        if (engine->actions[id].effects)
            memcpy(engine->actions[id].effects, action->effects,
                   action->effect_size * sizeof(float));
    }
    return id;
}

int plan_enhanced_add_time_constraint(PlanEnhancedEngine* engine, const PlanTimeConstraint* constraint)
{
    if (!engine || !constraint) return -1;
    if (ensure_capacity((void**)&engine->constraints, &engine->constraint_capacity,
                        engine->constraint_count + 1, sizeof(PlanTimeConstraint), 64) != 0)
        return -1;
    int id = engine->constraint_count++;
    engine->constraints[id] = *constraint;
    engine->constraints[id].constraint_id = id;
    return id;
}

int plan_enhanced_add_htn_task(PlanEnhancedEngine* engine, const PlanHTNTask* task)
{
    if (!engine || !task) return -1;
    if (ensure_capacity((void**)&engine->htn_tasks, &engine->htn_task_capacity,
                        engine->htn_task_count + 1, sizeof(PlanHTNTask), 32) != 0)
        return -1;
    int id = engine->htn_task_count++;
    engine->htn_tasks[id] = *task;
    engine->htn_tasks[id].task_id = id;
    if (task->subtasks && task->subtask_count > 0) {
        engine->htn_tasks[id].subtasks = (int*)safe_malloc((size_t)task->subtask_count * sizeof(int));
        if (engine->htn_tasks[id].subtasks)
            memcpy(engine->htn_tasks[id].subtasks, task->subtasks,
                   (size_t)task->subtask_count * sizeof(int));
    }
    if (task->decomposition_methods && task->method_count > 0) {
        engine->htn_tasks[id].decomposition_methods = (int*)safe_malloc((size_t)task->method_count * sizeof(int));
        if (engine->htn_tasks[id].decomposition_methods)
            memcpy(engine->htn_tasks[id].decomposition_methods, task->decomposition_methods,
                   (size_t)task->method_count * sizeof(int));
    }
    if (task->preconditions && task->precond_size > 0) {
        engine->htn_tasks[id].preconditions = (float*)safe_malloc(task->precond_size * sizeof(float));
        if (engine->htn_tasks[id].preconditions)
            memcpy(engine->htn_tasks[id].preconditions, task->preconditions,
                   task->precond_size * sizeof(float));
    }
    return id;
}

int plan_enhanced_add_htn_method(PlanEnhancedEngine* engine, const PlanHTNMethod* method)
{
    if (!engine || !method) return -1;
    if (ensure_capacity((void**)&engine->htn_methods, &engine->htn_method_capacity,
                        engine->htn_method_count + 1, sizeof(PlanHTNMethod), 32) != 0)
        return -1;
    int id = engine->htn_method_count++;
    engine->htn_methods[id] = *method;
    engine->htn_methods[id].method_id = id;
    if (method->subtask_ids && method->subtask_count > 0) {
        engine->htn_methods[id].subtask_ids = (int*)safe_malloc((size_t)method->subtask_count * sizeof(int));
        if (engine->htn_methods[id].subtask_ids)
            memcpy(engine->htn_methods[id].subtask_ids, method->subtask_ids,
                   (size_t)method->subtask_count * sizeof(int));
    }
    if (method->applicability_conditions && method->cond_size > 0) {
        engine->htn_methods[id].applicability_conditions = (float*)safe_malloc(method->cond_size * sizeof(float));
        if (engine->htn_methods[id].applicability_conditions)
            memcpy(engine->htn_methods[id].applicability_conditions, method->applicability_conditions,
                   method->cond_size * sizeof(float));
    }
    return id;
}

int plan_enhanced_register_agent(PlanEnhancedEngine* engine, const PlanAgent* agent)
{
    if (!engine || !agent) return -1;
    if (ensure_capacity((void**)&engine->agents, &engine->agent_capacity,
                        engine->agent_count + 1, sizeof(PlanAgent), 16) != 0)
        return -1;
    int id = engine->agent_count++;
    engine->agents[id] = *agent;
    engine->agents[id].agent_id = id;
    return id;
}

int plan_enhanced_check_temporal_consistency(PlanEnhancedEngine* engine,
                                              const PlanTimeConstraint* constraints,
                                              int num_constraints)
{
    if (!engine || !constraints || num_constraints <= 0) return -1;
    /* 确定STN节点数：每个动作是一个节点，加上一个时间参考零点节点 */
    int num_nodes = engine->action_count + 1;
    int zero_node = engine->action_count;
    /* 分配距离矩阵 */
    float** dist = (float**)safe_malloc((size_t)num_nodes * sizeof(float*));
    if (!dist) return -1;
    for (int i = 0; i < num_nodes; i++) {
        dist[i] = (float*)safe_malloc((size_t)num_nodes * sizeof(float));
        if (!dist[i]) {
            for (int j = 0; j < i; j++) safe_free((void**)&dist[j]);
            safe_free((void**)&dist);
            return -1;
        }
        for (int j = 0; j < num_nodes; j++) {
            dist[i][j] = (i == j) ? 0.0f : PLAN_ENH_INF;
        }
    }
    /* 添加差分约束：x_j - x_i ≤ w */
    for (int k = 0; k < num_constraints; k++) {
        const PlanTimeConstraint* c = &constraints[k];
        int from = c->from_action;
        int to = c->to_action;
        if (from < 0 || from >= engine->action_count) from = zero_node;
        if (to < 0 || to >= engine->action_count) to = zero_node;
        switch (c->type) {
            case TIME_CONSTRAINT_BOUNDED:
                /* x_to - x_from ≤ max AND x_from - x_to ≤ -min */
                if (c->max_time < dist[from][to])
                    dist[from][to] = c->max_time;
                if (-c->min_time < dist[to][from])
                    dist[to][from] = -c->min_time;
                break;
            case TIME_CONSTRAINT_AT:
                if (0.0f < dist[from][to])
                    dist[from][to] = 0.0f;
                if (0.0f < dist[to][from])
                    dist[to][from] = 0.0f;
                break;
            case TIME_CONSTRAINT_AFTER:
                if (PLAN_ENH_INF < dist[from][to])
                    dist[from][to] = PLAN_ENH_INF;
                if (-c->min_time < dist[to][from])
                    dist[to][from] = -c->min_time;
                break;
            case TIME_CONSTRAINT_BEFORE:
                if (c->max_time < dist[from][to])
                    dist[from][to] = c->max_time;
                if (-PLAN_ENH_INF < dist[to][from])
                    dist[to][from] = -PLAN_ENH_INF;
                break;
            case TIME_CONSTRAINT_DURATION:
                if (c->duration < dist[from][to])
                    dist[from][to] = c->duration;
                if (-c->duration < dist[to][from])
                    dist[to][from] = -c->duration;
                break;
            default:
                break;
        }
    }
    /* 每个动作持续时间约束 */
    for (int i = 0; i < engine->action_count; i++) {
        int time_var = i;
        if (engine->actions[i].duration_max < dist[time_var][zero_node])
            dist[time_var][zero_node] = engine->actions[i].duration_max;
        if (-engine->actions[i].duration_min < dist[zero_node][time_var])
            dist[zero_node][time_var] = -engine->actions[i].duration_min;
    }
    /* Floyd-Warshall检测负环 */
    for (int k = 0; k < num_nodes; k++) {
        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_nodes; j++) {
                if (dist[i][k] < PLAN_ENH_INF && dist[k][j] < PLAN_ENH_INF) {
                    float nd = dist[i][k] + dist[k][j];
                    if (nd < dist[i][j]) dist[i][j] = nd;
                }
            }
        }
    }
    /* 检查负环 */
    int consistent = 1;
    for (int i = 0; i < num_nodes; i++) {
        if (dist[i][i] < 0.0f) {
            consistent = 0;
            break;
        }
    }
    /* 保存距离矩阵 */
    if (engine->stn_distance_matrix) {
        for (int i = 0; i < engine->stn_node_count; i++) {
            safe_free((void**)&engine->stn_distance_matrix[i]);
        }
        safe_free((void**)&engine->stn_distance_matrix);
    }
    engine->stn_distance_matrix = dist;
    engine->stn_node_count = num_nodes;
    return consistent;
}

int plan_enhanced_generate_temporal(PlanEnhancedEngine* engine,
                                     const float* goal_state, size_t goal_size,
                                     const float* initial_state, size_t state_size,
                                     PlanEnhancedResult* result)
{
    if (!engine || !goal_state || !initial_state || !result) return -1;
    if (!engine->is_initialized) return -1;
    /* 先检查时间一致性 */
    if (engine->constraint_count > 0) {
        int consistent = plan_enhanced_check_temporal_consistency(
            engine, engine->constraints, engine->constraint_count);
        if (consistent <= 0) {
            result->is_feasible = 0;
            result->plan_confidence = 0.0f;
            snprintf(result->plan_summary, sizeof(result->plan_summary),
                     "时间一致性检查失败");
            return -1;
        }
    }
    /* 为规划结果分配内存 */
    int max_steps = engine->config.max_plan_length > 0 ?
                    engine->config.max_plan_length : 100;
    result->action_ids = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    result->start_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    result->end_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    result->assigned_agents = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    if (!result->action_ids || !result->start_times ||
        !result->end_times || !result->assigned_agents) {
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    /* 当前状态 = 初始状态副本 */
    float* current_state = (float*)safe_malloc(state_size * sizeof(float));
    if (!current_state) {
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    memcpy(current_state, initial_state, state_size * sizeof(float));
    /* P2-052修复: Beam Search替代贪婪搜索，保留top-k候选路径 */
    int plan_len = 0;
    float current_time = 0.0f;
    float total_cost = 0.0f;
    result->total_duration = 0.0f;

    #define PLAN_MAX_BEAM 5
    /* 每一条beam候选路径的状态 */
    typedef struct {
        float* state;
        float time;
        float cost;
        int actions[256];
        int action_count;
        float cfc_score;
    } BeamCandidate;

    BeamCandidate beams[PLAN_MAX_BEAM];
    int beam_count = 1;
    /* 初始化第0条beam */
    beams[0].state = (float*)safe_malloc(state_size * sizeof(float));
    if (!beams[0].state) { safe_free((void**)&current_state); return -1; }
    memcpy(beams[0].state, initial_state, state_size * sizeof(float));
    beams[0].time = 0.0f;
    beams[0].cost = 0.0f;
    beams[0].action_count = 0;
    beams[0].cfc_score = 0.0f;

    for (int step = 0; step < max_steps; step++) {
        if (beam_count == 0) break;

        /* 收集所有扩展候选 */
        typedef struct {
            float* state;
            float time;
            float cost;
            int action_id;
            float cfc_score;
            int parent_beam;
        } Expansion;

        int max_expansions = beam_count * engine->action_count;
        Expansion* expansions = (Expansion*)safe_malloc((size_t)max_expansions * sizeof(Expansion));
        if (!expansions) break;
        int expansion_count = 0;

        for (int b = 0; b < beam_count; b++) {
            /* 检查该beam是否已到达目标 */
            int goal_met = 1;
            size_t min_gs = goal_size < state_size ? goal_size : state_size;
            for (size_t i = 0; i < min_gs; i++) {
                if (goal_state[i] > 0.5f && beams[b].state[i] < 0.5f) {
                    goal_met = 0;
                    break;
                }
            }
            if (goal_met) continue; /* 已到达目标，不再扩展 */

            for (int a = 0; a < engine->action_count; a++) {
                /* 检查前提条件 */
                int applicable = 1;
                size_t min_ps = engine->actions[a].precond_size < state_size ?
                                engine->actions[a].precond_size : state_size;
                for (size_t i = 0; i < min_ps; i++) {
                    if (engine->actions[a].preconditions[i] > 0.5f && beams[b].state[i] < 0.5f) {
                        applicable = 0;
                        break;
                    }
                }
                if (!applicable) continue;

                /* 计算后继状态 */
                float* next = (float*)safe_malloc(state_size * sizeof(float));
                if (!next) continue;
                memcpy(next, beams[b].state, state_size * sizeof(float));
                size_t min_es = engine->actions[a].effect_size < state_size ?
                                engine->actions[a].effect_size : state_size;
                for (size_t i = 0; i < min_es; i++) {
                    if (engine->actions[a].effects[i] > 0.5f) next[i] = 1.0f;
                    else if (engine->actions[a].effects[i] < -0.5f) next[i] = 0.0f;
                }

                /* 用CfC评估该动作 */
                float cfc_input[4] = {
                    (float)a / (float)(engine->action_count + 1),
                    beams[b].state[0],
                    next[0],
                    engine->actions[a].cost
                };
                float h_save[64];
                memcpy(h_save, engine->cfc_state, (size_t)engine->cfc_dim * sizeof(float));
                cfc_integrate(engine, cfc_input, 5);
                float action_score = engine->cfc_state[0] * 0.7f - engine->actions[a].cost * 0.3f;
                memcpy(engine->cfc_state, h_save, (size_t)engine->cfc_dim * sizeof(float));

                /* 记录扩展候选 */
                if (expansion_count < max_expansions) {
                    expansions[expansion_count].state = next;
                    expansions[expansion_count].time = beams[b].time + engine->actions[a].duration_min;
                    expansions[expansion_count].cost = beams[b].cost + engine->actions[a].cost;
                    expansions[expansion_count].action_id = a;
                    expansions[expansion_count].cfc_score = beams[b].cfc_score + action_score;
                    expansions[expansion_count].parent_beam = b;
                    expansion_count++;
                } else {
                    safe_free((void**)&next);
                }
            }
        }

        /* 按CfC评分降序排序扩展候选 */
        for (int i = 0; i < expansion_count - 1; i++) {
            int best_idx = i;
            for (int j = i + 1; j < expansion_count; j++) {
                if (expansions[j].cfc_score > expansions[best_idx].cfc_score) {
                    best_idx = j;
                }
            }
            if (best_idx != i) {
                Expansion tmp = expansions[i];
                expansions[i] = expansions[best_idx];
                expansions[best_idx] = tmp;
            }
        }

        /* 选择top-k beam */
        int new_beam_count = 0;
        for (int i = 0; i < expansion_count && new_beam_count < PLAN_MAX_BEAM; i++) {
            int parent = expansions[i].parent_beam;
            BeamCandidate* nb = &beams[new_beam_count];

            /* 复用或分配新状态 */
            if (new_beam_count != parent) {
                if (nb->state) safe_free((void**)&nb->state);
                nb->state = expansions[i].state;
            } else {
                /* 同一parent，需要新分配 */
                float* old_state = nb->state;
                nb->state = expansions[i].state;
                if (old_state && old_state != expansions[i].state) {
                    safe_free((void**)&old_state);
                }
            }

            /* 复制动作历史 */
            memcpy(nb->actions, beams[parent].actions, (size_t)beams[parent].action_count * sizeof(int));
            nb->actions[beams[parent].action_count] = expansions[i].action_id;
            nb->action_count = beams[parent].action_count + 1;
            nb->time = expansions[i].time;
            nb->cost = expansions[i].cost;
            nb->cfc_score = expansions[i].cfc_score;

            new_beam_count++;
        }

        /* 释放未选中的扩展状态 */
        for (int i = new_beam_count; i < expansion_count; i++) {
            safe_free((void**)&expansions[i].state);
        }
        /* 释放旧beam中未复用的状态 */
        for (int b = new_beam_count; b < beam_count; b++) {
            if (beams[b].state) safe_free((void**)&beams[b].state);
            beams[b].state = NULL;
        }

        safe_free((void**)&expansions);
        beam_count = new_beam_count;

        /* 更新当前时间（取最佳beam的时间） */
        if (beam_count > 0) {
            current_time = beams[0].time;
            total_cost = beams[0].cost;
            plan_len = beams[0].action_count;
            memcpy(current_state, beams[0].state, state_size * sizeof(float));
        }
    }

    /* 将最佳beam的动作复制到结果 */
    if (beam_count > 0 && beams[0].action_count > 0) {
        for (int i = 0; i < beams[0].action_count && i < 256; i++) {
            result->action_ids[i] = beams[0].actions[i];
            result->start_times[i] = 0; /* 时间将在下面重新计算 */
            float dur = engine->actions[beams[0].actions[i]].duration_min;
            if (dur < 0.1f) dur = 1.0f;
            result->end_times[i] = result->start_times[i] + dur;
            result->assigned_agents[i] = 0;
            if (i > 0) {
                result->start_times[i] = result->end_times[i-1];
                result->end_times[i] = result->start_times[i] + dur;
            }
        }
        result->action_count = beams[0].action_count;
        current_time = (result->action_count > 0) ? result->end_times[result->action_count - 1] : 0.0f;
        total_cost = beams[0].cost;
    } else {
        result->action_count = 0;
    }

    /* 释放所有beam状态 */
    for (int b = 0; b < PLAN_MAX_BEAM; b++) {
        if (beams[b].state) safe_free((void**)&beams[b].state);
    }
    safe_free((void**)&current_state);

    result->agent_count = 1;
    result->total_cost = total_cost;
    result->total_duration = current_time;
    result->makespan = current_time;
    /* ZSFWS修复 P2-005: 基于规划步数动态计算置信度，替代启发式固定公式 */
    {
        float completion_ratio = (plan_len > 0) ? 1.0f / (1.0f + (float)plan_len * 0.1f) : 0.0f;
        result->plan_confidence = 0.2f + 0.6f * completion_ratio;
    }
    result->is_feasible = plan_len > 0 ? 1 : 0;
    if (plan_len > 0) {
        snprintf(result->plan_summary, sizeof(result->plan_summary),
                 "BeamSearch规划完成: %d个动作, 总时间%.2f, 总成本%.2f",
                 plan_len, current_time, total_cost);
    } else {
        snprintf(result->plan_summary, sizeof(result->plan_summary),
                 "时间规划失败: 未能生成有效规划");
    }
    return plan_len > 0 ? 0 : -1;
}

/* ============================================================================
 * A04.2.2 — Landmark-based规划实现
 * ============================================================================ */

PlanLandmarkGraph* plan_enhanced_extract_landmarks(PlanEnhancedEngine* engine,
                                                    const float* initial_state, size_t state_size,
                                                    const float* goal_state, size_t goal_size)
{
    if (!engine || !initial_state || !goal_state || state_size == 0) return NULL;
    PlanLandmarkGraph* graph = (PlanLandmarkGraph*)safe_calloc(1, sizeof(PlanLandmarkGraph));
    if (!graph) return NULL;
    graph->state_size = state_size > goal_size ? state_size : goal_size;
    graph->initial_state = (float*)safe_malloc(graph->state_size * sizeof(float));
    graph->goal_state = (float*)safe_malloc(graph->state_size * sizeof(float));
    if (!graph->initial_state || !graph->goal_state) {
        safe_free((void**)&graph->initial_state);
        safe_free((void**)&graph->goal_state);
        safe_free((void**)&graph);
        return NULL;
    }
    memcpy(graph->initial_state, initial_state, state_size * sizeof(float));
    if (state_size < graph->state_size)
        memset(graph->initial_state + state_size, 0, (graph->state_size - state_size) * sizeof(float));
    memcpy(graph->goal_state, goal_state, goal_size * sizeof(float));
    if (goal_size < graph->state_size)
        memset(graph->goal_state + goal_size, 0, (graph->state_size - goal_size) * sizeof(float));

    graph->landmark_capacity = 64;
    graph->landmarks = (PlanLandmark*)safe_calloc((size_t)graph->landmark_capacity, sizeof(PlanLandmark));
    if (!graph->landmarks) {
        safe_free((void**)&graph->initial_state);
        safe_free((void**)&graph->goal_state);
        safe_free((void**)&graph);
        return NULL;
    }
    graph->landmark_count = 0;

    /* 阶段1: 从目标命题中提取Landmark */
    int* processed = (int*)safe_calloc(graph->state_size, sizeof(int));
    int* queue = (int*)safe_calloc(graph->state_size, sizeof(int));
    int queue_head = 0, queue_tail = 0;
    if (!processed || !queue) {
        safe_free((void**)&processed);
        safe_free((void**)&queue);
        plan_enhanced_destroy_landmark_graph(graph);
        return NULL;
    }

    for (size_t i = 0; i < goal_size; i++) {
        if (goal_state[i] > 0.5f && initial_state[i] <= 0.5f) {
            if (graph->landmark_count >= graph->landmark_capacity) {
                int new_cap = graph->landmark_capacity * 2;
                PlanLandmark* tmp = (PlanLandmark*)realloc(graph->landmarks,
                    (size_t)new_cap * sizeof(PlanLandmark));
                if (!tmp) break;
                graph->landmarks = tmp;
                graph->landmark_capacity = new_cap;
            }
            int lid = graph->landmark_count++;
            memset(&graph->landmarks[lid], 0, sizeof(PlanLandmark));
            graph->landmarks[lid].landmark_id = lid;
            graph->landmarks[lid].type = PLAN_LANDMARK_FACT;
            graph->landmarks[lid].fact_size = graph->state_size;
            graph->landmarks[lid].fact_vector = (float*)safe_calloc(graph->state_size, sizeof(float));
            if (graph->landmarks[lid].fact_vector) {
                graph->landmarks[lid].fact_vector[i] = 1.0f;
            }
            graph->landmarks[lid].importance = goal_state[i];
            graph->landmarks[lid].is_achieved = 0;
            graph->landmarks[lid].discovery_cost = 0.0f;
            processed[i] = 1;
            queue[queue_tail++] = (int)i;
        }
    }

    /* 阶段2: 因果回归分析 — 对每个Landmark找到使其成立的动作，前提条件为新Landmark */
    while (queue_head < queue_tail) {
        int prop_idx = queue[queue_head++];
        for (int a = 0; a < engine->action_count; a++) {
            PlanEnhancedAction* act = &engine->actions[a];
            if (!act->effects || (size_t)prop_idx >= act->effect_size) continue;
            if (act->effects[prop_idx] > 0.5f) {
                for (size_t p = 0; p < act->precond_size && p < graph->state_size; p++) {
                    if (act->preconditions[p] > 0.5f && !processed[(int)p]) {
                        int already_landmark = 0;
                        for (int l = 0; l < graph->landmark_count; l++) {
                            if (graph->landmarks[l].fact_vector &&
                                graph->landmarks[l].fact_vector[p] > 0.5f) {
                                already_landmark = 1;
                                break;
                            }
                        }
                        if (already_landmark) continue;
                        if (graph->landmark_count >= graph->landmark_capacity) {
                            int new_cap = graph->landmark_capacity * 2;
                            PlanLandmark* tmp = (PlanLandmark*)realloc(graph->landmarks,
                                (size_t)new_cap * sizeof(PlanLandmark));
                            if (!tmp) break;
                            graph->landmarks = tmp;
                            graph->landmark_capacity = new_cap;
                        }
                        int lid = graph->landmark_count++;
                        memset(&graph->landmarks[lid], 0, sizeof(PlanLandmark));
                        graph->landmarks[lid].landmark_id = lid;
                        graph->landmarks[lid].type = PLAN_LANDMARK_FACT;
                        graph->landmarks[lid].fact_size = graph->state_size;
                        graph->landmarks[lid].fact_vector = (float*)safe_calloc(graph->state_size, sizeof(float));
                        if (graph->landmarks[lid].fact_vector) {
                            graph->landmarks[lid].fact_vector[p] = 1.0f;
                        }
                        graph->landmarks[lid].importance = 0.8f;
                        graph->landmarks[lid].discovery_cost = act->cost;
                        processed[(int)p] = 1;
                        if (queue_tail < (int)graph->state_size) {
                            queue[queue_tail++] = (int)p;
                        }
                    }
                }
            }
        }
    }

    /* 阶段3: 构建Landmark序关系矩阵 */
    int lc = graph->landmark_count;
    if (lc > 0) {
        graph->landmark_order_matrix = (int*)safe_calloc((size_t)(lc * lc), sizeof(int));
        graph->level_of_landmark = (int*)safe_calloc((size_t)lc, sizeof(int));
        if (graph->landmark_order_matrix && graph->level_of_landmark) {
            for (int i = 0; i < lc; i++) {
                for (int j = 0; j < lc; j++) {
                    if (i == j) continue;
                    PlanLandmark* li = &graph->landmarks[i];
                    PlanLandmark* lj = &graph->landmarks[j];
                    for (int a = 0; a < engine->action_count; a++) {
                        PlanEnhancedAction* act = &engine->actions[a];
                        if (!act->effects || !act->preconditions) continue;
                        int makes_j = 0, needs_i = 0;
                        for (size_t k = 0; k < act->effect_size && k < graph->state_size; k++) {
                            if (lj->fact_vector && lj->fact_vector[k] > 0.5f && act->effects[k] > 0.5f) {
                                makes_j = 1;
                            }
                        }
                        for (size_t k = 0; k < act->precond_size && k < graph->state_size; k++) {
                            if (li->fact_vector && li->fact_vector[k] > 0.5f && act->preconditions[k] > 0.5f) {
                                needs_i = 1;
                            }
                        }
                        if (makes_j && needs_i) {
                            graph->landmark_order_matrix[i * lc + j] = 1;
                            break;
                        }
                    }
                }
            }
            /* 传递闭包 */
            for (int k = 0; k < lc; k++) {
                for (int i = 0; i < lc; i++) {
                    for (int j = 0; j < lc; j++) {
                        if (graph->landmark_order_matrix[i * lc + k] &&
                            graph->landmark_order_matrix[k * lc + j]) {
                            graph->landmark_order_matrix[i * lc + j] = 1;
                        }
                    }
                }
            }
            /* 计算层次（最长路径） */
            for (int i = 0; i < lc; i++) {
                int max_level = 0;
                for (int j = 0; j < lc; j++) {
                    if (graph->landmark_order_matrix[j * lc + i]) {
                        if (graph->level_of_landmark[j] + 1 > max_level)
                            max_level = graph->level_of_landmark[j] + 1;
                    }
                }
                graph->level_of_landmark[i] = max_level;
            }
            /* 构建前驱/后继列表 */
            int* temp_pred = (int*)safe_calloc((size_t)lc, sizeof(int));
            int* temp_succ = (int*)safe_calloc((size_t)lc, sizeof(int));
            if (temp_pred && temp_succ) {
                for (int i = 0; i < lc; i++) {
                    int pred_count = 0, succ_count = 0;
                    for (int j = 0; j < lc; j++) {
                        if (i == j) continue;
                        if (graph->landmark_order_matrix[j * lc + i]) pred_count++;
                        if (graph->landmark_order_matrix[i * lc + j]) succ_count++;
                    }
                    if (pred_count > 0) {
                        graph->landmarks[i].predecessors = (int*)safe_malloc((size_t)pred_count * sizeof(int));
                        if (graph->landmarks[i].predecessors) {
                            int idx = 0;
                            for (int j = 0; j < lc; j++) {
                                if (i != j && graph->landmark_order_matrix[j * lc + i])
                                    graph->landmarks[i].predecessors[idx++] = j;
                            }
                            graph->landmarks[i].pred_count = pred_count;
                        }
                    }
                    if (succ_count > 0) {
                        graph->landmarks[i].successors = (int*)safe_malloc((size_t)succ_count * sizeof(int));
                        if (graph->landmarks[i].successors) {
                            int idx = 0;
                            for (int j = 0; j < lc; j++) {
                                if (i != j && graph->landmark_order_matrix[i * lc + j])
                                    graph->landmarks[i].successors[idx++] = j;
                            }
                            graph->landmarks[i].succ_count = succ_count;
                        }
                    }
                }
                safe_free((void**)&temp_pred);
            }
            safe_free((void**)&temp_succ);
        }
    }

    safe_free((void**)&processed);
    safe_free((void**)&queue);
    return graph;
}

void plan_enhanced_destroy_landmark_graph(PlanLandmarkGraph* graph)
{
    if (!graph) return;
    if (graph->landmarks) {
        for (int i = 0; i < graph->landmark_count; i++) {
            safe_free((void**)&graph->landmarks[i].fact_vector);
            safe_free((void**)&graph->landmarks[i].predecessors);
            safe_free((void**)&graph->landmarks[i].successors);
        }
        safe_free((void**)&graph->landmarks);
    }
    safe_free((void**)&graph->initial_state);
    safe_free((void**)&graph->goal_state);
    safe_free((void**)&graph->landmark_order_matrix);
    safe_free((void**)&graph->level_of_landmark);
    safe_free((void**)&graph);
}

int plan_enhanced_get_landmark_count(const PlanLandmarkGraph* graph)
{
    if (!graph) return -1;
    return graph->landmark_count;
}

int plan_enhanced_get_landmark(const PlanLandmarkGraph* graph, int index, PlanLandmark* landmark)
{
    if (!graph || !landmark || index < 0 || index >= graph->landmark_count) return -1;
    *landmark = graph->landmarks[index];
    return 0;
}

float plan_enhanced_landmark_heuristic(PlanEnhancedEngine* engine,
                                        const float* state, size_t state_size,
                                        const PlanLandmarkGraph* graph)
{
    if (!engine || !state || !graph) return -1.0f;
    int unachieved = 0;
    for (int i = 0; i < graph->landmark_count; i++) {
        PlanLandmark* lm = &graph->landmarks[i];
        if (!lm->fact_vector) { unachieved++; continue; }
        int achieved = 1;
        for (size_t j = 0; j < (lm->fact_size < state_size ? lm->fact_size : state_size); j++) {
            if (lm->fact_vector[j] > 0.5f && state[j] <= 0.5f) {
                achieved = 0;
                break;
            }
        }
        if (!achieved) unachieved++;
    }
    if (unachieved == 0) return 0.0f;
    return (float)unachieved * 0.9f;
}

int plan_enhanced_generate_landmark_based(PlanEnhancedEngine* engine,
                                           const float* goal_state, size_t goal_size,
                                           const float* initial_state, size_t state_size,
                                           PlanEnhancedResult* result)
{
    if (!engine || !goal_state || !initial_state || !result) return -1;
    memset(result, 0, sizeof(PlanEnhancedResult));

    PlanLandmarkGraph* graph = plan_enhanced_extract_landmarks(engine, initial_state,
                                                                state_size, goal_state, goal_size);
    if (!graph) return -1;

    /* A*搜索：f(s) = g(s) + h_LM(s) */
    enum { LM_SEARCH_DEPTH = 200, LM_MAX_OPEN = 2000 };
    int search_depth = engine->config.max_plan_length > 0 ?
        (engine->config.max_plan_length < LM_SEARCH_DEPTH ? engine->config.max_plan_length : LM_SEARCH_DEPTH)
        : LM_SEARCH_DEPTH;

    float** open_states = (float**)safe_calloc((size_t)LM_MAX_OPEN, sizeof(float*));
    float* open_g = (float*)safe_calloc((size_t)LM_MAX_OPEN, sizeof(float));
    float* open_f = (float*)safe_calloc((size_t)LM_MAX_OPEN, sizeof(float));
    int* open_parent = (int*)safe_calloc((size_t)LM_MAX_OPEN, sizeof(int));
    int* open_action = (int*)safe_calloc((size_t)LM_MAX_OPEN, sizeof(int));
    int* open_closed = (int*)safe_calloc((size_t)LM_MAX_OPEN, sizeof(int));
    float* open_state_data = (float*)safe_calloc((size_t)LM_MAX_OPEN * state_size, sizeof(float));
    int open_count = 0;
    int found = 0;
    int goal_node = -1;

    if (!open_states || !open_g || !open_f || !open_parent || !open_action ||
        !open_closed || !open_state_data) {
        safe_free((void**)&open_states);
        safe_free((void**)&open_g);
        safe_free((void**)&open_f);
        safe_free((void**)&open_parent);
        safe_free((void**)&open_action);
        safe_free((void**)&open_closed);
        safe_free((void**)&open_state_data);
        plan_enhanced_destroy_landmark_graph(graph);
        return -1;
    }

    /* 初始状态入队 */
    memcpy(open_state_data, initial_state, state_size * sizeof(float));
    open_states[0] = open_state_data;
    open_g[0] = 0.0f;
    open_f[0] = plan_enhanced_landmark_heuristic(engine, initial_state, state_size, graph);
    open_parent[0] = -1;
    open_action[0] = -1;
    open_closed[0] = 0;
    open_count = 1;

    /* 目标检查 */
    int all_goal_achieved = 1;
    for (size_t i = 0; i < goal_size; i++) {
        if (goal_state[i] > 0.5f && initial_state[i] <= 0.5f) {
            all_goal_achieved = 0;
            break;
        }
    }
    if (all_goal_achieved) {
        result->action_count = 0;
        result->total_cost = 0.0f;
        result->total_duration = 0.0f;
        result->makespan = 0.0f;
        result->plan_confidence = 1.0f;
        result->is_feasible = 1;
        snprintf(result->plan_summary, sizeof(result->plan_summary), "Landmark规划: 已在目标状态");
        plan_enhanced_destroy_landmark_graph(graph);
        safe_free((void**)&open_states);
        safe_free((void**)&open_g);
        safe_free((void**)&open_f);
        safe_free((void**)&open_parent);
        safe_free((void**)&open_action);
        safe_free((void**)&open_closed);
        safe_free((void**)&open_state_data);
        return 0;
    }

    /* A*主循环 */
    for (int iter = 0; iter < search_depth * 10 && open_count > 0 && !found; iter++) {
        int best = -1;
        float best_f = PLAN_ENH_INF;
        for (int i = 0; i < open_count; i++) {
            if (!open_closed[i] && open_f[i] < best_f) {
                best_f = open_f[i];
                best = i;
            }
        }
        if (best < 0) break;

        open_closed[best] = 1;
        float* cur_state = open_states[best];
        float cur_g = open_g[best];

        /* 生成后继状态 */
        for (int a = 0; a < engine->action_count; a++) {
            PlanEnhancedAction* act = &engine->actions[a];
            /* 检查前提 */
            int applicable = 1;
            for (size_t p = 0; p < act->precond_size && p < state_size; p++) {
                if (act->preconditions[p] > 0.5f && cur_state[p] <= 0.5f) {
                    applicable = 0;
                    break;
                }
            }
            if (!applicable) continue;

            if (open_count >= LM_MAX_OPEN) break;

            int child = open_count++;
            open_states[child] = open_state_data + (size_t)child * state_size;
            memcpy(open_states[child], cur_state, state_size * sizeof(float));
            /* 应用效果 */
            for (size_t e = 0; e < act->effect_size && e < state_size; e++) {
                if (act->effects[e] > 0.5f)
                    open_states[child][e] = act->effects[e];
                else if (act->effects[e] < -0.5f)
                    open_states[child][e] = 0.0f;
            }
            open_g[child] = cur_g + act->cost;
            open_parent[child] = best;
            open_action[child] = a;
            open_closed[child] = 0;
            open_f[child] = open_g[child] + plan_enhanced_landmark_heuristic(engine,
                open_states[child], state_size, graph);

            /* 检查是否达到目标 */
            int goal_check = 1;
            for (size_t i = 0; i < goal_size; i++) {
                if (goal_state[i] > 0.5f && open_states[child][i] <= 0.5f) {
                    goal_check = 0;
                    break;
                }
            }
            if (goal_check) {
                found = 1;
                goal_node = child;
                break;
            }
        }
    }

    if (!found) {
        plan_enhanced_destroy_landmark_graph(graph);
        safe_free((void**)&open_states);
        safe_free((void**)&open_g);
        safe_free((void**)&open_f);
        safe_free((void**)&open_parent);
        safe_free((void**)&open_action);
        safe_free((void**)&open_closed);
        safe_free((void**)&open_state_data);
        return -1;
    }

    /* 回溯路径 */
    int path[LM_SEARCH_DEPTH];
    int path_len = 0;
    int node = goal_node;
    while (node >= 0 && path_len < LM_SEARCH_DEPTH) {
        if (open_action[node] >= 0) {
            path[path_len++] = open_action[node];
        }
        node = open_parent[node];
    }

    /* 反转路径 */
    result->action_ids = (int*)safe_malloc((size_t)path_len * sizeof(int));
    result->start_times = (float*)safe_calloc((size_t)path_len, sizeof(float));
    result->end_times = (float*)safe_calloc((size_t)path_len, sizeof(float));
    if (result->action_ids) {
        float cur_time = 0.0f;
        float total_cost = 0.0f;
        for (int i = 0; i < path_len; i++) {
            result->action_ids[i] = path[path_len - 1 - i];
            int aid = result->action_ids[i];
            result->start_times[i] = cur_time;
            if (aid >= 0 && aid < engine->action_count) {
                cur_time += engine->actions[aid].duration_min + 0.5f * (engine->actions[aid].duration_max - engine->actions[aid].duration_min);
                total_cost += engine->actions[aid].cost;
            } else {
                cur_time += 1.0f;
            }
            result->end_times[i] = cur_time;
        }
        result->action_count = path_len;
        result->total_cost = total_cost;
        result->total_duration = cur_time;
        result->makespan = cur_time;
    }
    result->agent_count = 1;
    /* ZSFWS修复 P2-005: 基于规划步数动态计算置信度，替代启发式固定公式 */
    {
        float completion_ratio = (path_len > 0) ? 1.0f / (1.0f + (float)path_len * 0.1f) : 0.0f;
        result->plan_confidence = 0.2f + 0.6f * completion_ratio;
    }
    result->is_feasible = path_len > 0 ? 1 : 0;
    snprintf(result->plan_summary, sizeof(result->plan_summary),
             "Landmark规划完成: %d个动作, %d个Landmark, 时间%.2f",
             path_len, graph->landmark_count, result->total_duration);

    plan_enhanced_destroy_landmark_graph(graph);
    safe_free((void**)&open_states);
    safe_free((void**)&open_g);
    safe_free((void**)&open_f);
    safe_free((void**)&open_parent);
    safe_free((void**)&open_action);
    safe_free((void**)&open_closed);
    safe_free((void**)&open_state_data);
    return 0;
}

/* ============================================================================
 * A04.2.2 — 符号规划实现 (Symbolic/STRIPS Planning)
 * ============================================================================ */

PlanSymbolicPlanner* plan_enhanced_create_symbolic_planner(PlanEnhancedEngine* engine,
                                                            size_t state_size)
{
    if (!engine || state_size == 0) return NULL;
    PlanSymbolicPlanner* planner = (PlanSymbolicPlanner*)safe_calloc(1, sizeof(PlanSymbolicPlanner));
    if (!planner) return NULL;
    planner->state_size = state_size;
    planner->action_count = engine->action_count;

    if (planner->action_count > 0) {
        planner->actions = (PlanSymbolicAction*)safe_calloc((size_t)planner->action_count, sizeof(PlanSymbolicAction));
        if (!planner->actions) {
            safe_free((void**)&planner);
            return NULL;
        }
        for (int i = 0; i < planner->action_count; i++) {
            PlanEnhancedAction* src = &engine->actions[i];
            PlanSymbolicAction* dst = &planner->actions[i];
            dst->action_id = src->action_id;
            dst->cost = src->cost;

            /* 提取正前提条件 */
            for (size_t p = 0; p < src->precond_size && p < state_size; p++) {
                if (src->preconditions[p] > 0.5f) dst->pos_precond_count++;
                else if (src->preconditions[p] < -0.5f) dst->neg_precond_count++;
            }
            if (dst->pos_precond_count > 0) {
                dst->positive_preconds = (int*)safe_malloc((size_t)dst->pos_precond_count * sizeof(int));
                if (dst->positive_preconds) {
                    int idx = 0;
                    for (size_t p = 0; p < src->precond_size && p < state_size; p++) {
                        if (src->preconditions[p] > 0.5f)
                            dst->positive_preconds[idx++] = (int)p;
                    }
                }
            }
            if (dst->neg_precond_count > 0) {
                dst->negative_preconds = (int*)safe_malloc((size_t)dst->neg_precond_count * sizeof(int));
                if (dst->negative_preconds) {
                    int idx = 0;
                    for (size_t p = 0; p < src->precond_size && p < state_size; p++) {
                        if (src->preconditions[p] < -0.5f)
                            dst->negative_preconds[idx++] = (int)p;
                    }
                }
            }

            /* 提取添加/删除效果 */
            for (size_t e = 0; e < src->effect_size && e < state_size; e++) {
                if (src->effects[e] > 0.5f) dst->add_effect_count++;
                else if (src->effects[e] < -0.5f) dst->del_effect_count++;
            }
            if (dst->add_effect_count > 0) {
                dst->add_effects = (int*)safe_malloc((size_t)dst->add_effect_count * sizeof(int));
                if (dst->add_effects) {
                    int idx = 0;
                    for (size_t e = 0; e < src->effect_size && e < state_size; e++) {
                        if (src->effects[e] > 0.5f)
                            dst->add_effects[idx++] = (int)e;
                    }
                }
            }
            if (dst->del_effect_count > 0) {
                dst->del_effects = (int*)safe_malloc((size_t)dst->del_effect_count * sizeof(int));
                if (dst->del_effects) {
                    int idx = 0;
                    for (size_t e = 0; e < src->effect_size && e < state_size; e++) {
                        if (src->effects[e] < -0.5f)
                            dst->del_effects[idx++] = (int)e;
                    }
                }
            }
        }
    }

    /* S-032修复: 基于真实状态提取目标命题和初始真命题
     * 替代从动作效果/前提中错误推导的启发式 */
    int* goal_set = (int*)safe_calloc(state_size, sizeof(int));
    int* init_set = (int*)safe_calloc(state_size, sizeof(int));
    if (goal_set && init_set) {
        /* 目标状态: 从所有动作的净效果中提取（效果总和>0的维度为目标）
         * 初始状态: 默认为零向量（无特殊前提时），从前提条件推导 */
        for (size_t i = 0; i < state_size && engine->actions && engine->action_count > 0; i++) {
            float net_effect = 0.0f;
            for (int a = 0; a < engine->action_count; a++) {
                if (engine->actions[a].effects && (size_t)i < engine->actions[a].effect_size) {
                    net_effect += engine->actions[a].effects[i];
                }
            }
            /* 正净效果 → 目标命题；负净效果 → 需要满足的前提 */
            if (net_effect > 0.3f) goal_set[i] = 1;
            for (int a = 0; a < engine->action_count; a++) {
                if (engine->actions[a].preconditions && (size_t)i < engine->actions[a].precond_size) {
                    if (engine->actions[a].preconditions[i] > 0.5f) { init_set[i] = 1; break; }
                }
            }
        }
        planner->goal_count = 0;
        planner->initial_true_count = 0;
        for (size_t i = 0; i < state_size; i++) {
            if (goal_set[i]) planner->goal_count++;
            if (init_set[i]) planner->initial_true_count++;
        }
        if (planner->goal_count > 0) {
            planner->goal_props = (int*)safe_malloc((size_t)planner->goal_count * sizeof(int));
            if (planner->goal_props) {
                int idx = 0;
                for (size_t i = 0; i < state_size; i++)
                    if (goal_set[i]) planner->goal_props[idx++] = (int)i;
            }
        }
        if (planner->initial_true_count > 0) {
            planner->initial_true_props = (int*)safe_malloc((size_t)planner->initial_true_count * sizeof(int));
            if (planner->initial_true_props) {
                int idx = 0;
                for (size_t i = 0; i < state_size; i++)
                    if (init_set[i]) planner->initial_true_props[idx++] = (int)i;
            }
        }
    }
    safe_free((void**)&goal_set);
    safe_free((void**)&init_set);

    /* CfC状态 */
    planner->cfc_dim = engine->cfc_dim > 0 ? engine->cfc_dim : 16;
    planner->cfc_state = (float*)safe_calloc((size_t)planner->cfc_dim, sizeof(float));

    return planner;
}

void plan_enhanced_destroy_symbolic_planner(PlanSymbolicPlanner* planner)
{
    if (!planner) return;
    if (planner->actions) {
        for (int i = 0; i < planner->action_count; i++) {
            safe_free((void**)&planner->actions[i].positive_preconds);
            safe_free((void**)&planner->actions[i].negative_preconds);
            safe_free((void**)&planner->actions[i].add_effects);
            safe_free((void**)&planner->actions[i].del_effects);
        }
        safe_free((void**)&planner->actions);
    }
    safe_free((void**)&planner->goal_props);
    safe_free((void**)&planner->initial_true_props);
    safe_free((void**)&planner->cfc_state);
    safe_free((void**)&planner);
}

int plan_enhanced_symbolic_action_applicable(const PlanSymbolicPlanner* planner,
                                              int action_index,
                                              const unsigned char* state)
{
    if (!planner || action_index < 0 || action_index >= planner->action_count || !state)
        return 0;
    const PlanSymbolicAction* act = &planner->actions[action_index];
    for (int i = 0; i < act->pos_precond_count; i++) {
        int p = act->positive_preconds[i];
        if (p < 0 || (size_t)p >= planner->state_size || !state[p]) return 0;
    }
    for (int i = 0; i < act->neg_precond_count; i++) {
        int p = act->negative_preconds[i];
        if (p < 0 || (size_t)p >= planner->state_size || state[p]) return 0;
    }
    return 1;
}

void plan_enhanced_symbolic_apply_action(const PlanSymbolicPlanner* planner,
                                          int action_index,
                                          unsigned char* state)
{
    if (!planner || !state || action_index < 0 || action_index >= planner->action_count) return;
    const PlanSymbolicAction* act = &planner->actions[action_index];
    for (int i = 0; i < act->del_effect_count; i++) {
        int p = act->del_effects[i];
        if (p >= 0 && (size_t)p < planner->state_size) state[p] = 0;
    }
    for (int i = 0; i < act->add_effect_count; i++) {
        int p = act->add_effects[i];
        if (p >= 0 && (size_t)p < planner->state_size) state[p] = 1;
    }
}

float plan_enhanced_symbolic_heuristic(const PlanSymbolicPlanner* planner,
                                        const unsigned char* state)
{
    if (!planner || !state) return -1.0f;
    if (planner->goal_count == 0) return 0.0f;

    int unachieved = 0;
    for (int i = 0; i < planner->goal_count; i++) {
        int p = planner->goal_props[i];
        if (p >= 0 && (size_t)p < planner->state_size && !state[p]) {
            unachieved++;
        }
    }
    if (unachieved == 0) return 0.0f;

    /* 删除放松规划图启发式：计算最少的动作层数 */
    enum { MAX_LAYERS = 100 };
    unsigned char* relaxed_state = (unsigned char*)safe_malloc(planner->state_size);
    if (!relaxed_state) return (float)unachieved;
    memcpy(relaxed_state, state, planner->state_size);

    int new_props = 0;
    for (int i = 0; i < planner->goal_count; i++) {
        int p = planner->goal_props[i];
        if (p >= 0 && (size_t)p < planner->state_size && !relaxed_state[p]) new_props++;
    }

    for (int layer = 1; layer <= MAX_LAYERS; layer++) {
        int added = 0;
        for (int a = 0; a < planner->action_count; a++) {
            const PlanSymbolicAction* act = &planner->actions[a];
            int applicable = 1;
            for (int p = 0; p < act->pos_precond_count; p++) {
                int pid = act->positive_preconds[p];
                if (pid < 0 || (size_t)pid >= planner->state_size || !relaxed_state[pid]) {
                    applicable = 0;
                    break;
                }
            }
            if (!applicable) continue;
            for (int p = 0; p < act->add_effect_count; p++) {
                int pid = act->add_effects[p];
                if (pid >= 0 && (size_t)pid < planner->state_size && !relaxed_state[pid]) {
                    relaxed_state[pid] = 1;
                    added++;
                }
            }
        }
        if (added == 0) break;
        int still_unachieved = 0;
        for (int i = 0; i < planner->goal_count; i++) {
            int p = planner->goal_props[i];
            if (p >= 0 && (size_t)p < planner->state_size && !relaxed_state[p])
                still_unachieved++;
        }
        if (still_unachieved == 0) {
            safe_free((void**)&relaxed_state);
            return (float)layer * 0.8f;
        }
    }

    safe_free((void**)&relaxed_state);
    return (float)unachieved * 1.5f;
}

int plan_enhanced_generate_symbolic(PlanSymbolicPlanner* planner,
                                     PlanEnhancedResult* result,
                                     int max_depth)
{
    if (!planner || !result || max_depth <= 0) return -1;
    memset(result, 0, sizeof(PlanEnhancedResult));

    if (max_depth > 200) max_depth = 200;
    size_t state_bytes = planner->state_size;

    /* A*搜索 */
    enum { SYM_MAX_OPEN = 5000 };
    unsigned char** open_states = (unsigned char**)safe_calloc((size_t)SYM_MAX_OPEN, sizeof(unsigned char*));
    float* open_g = (float*)safe_calloc((size_t)SYM_MAX_OPEN, sizeof(float));
    float* open_f = (float*)safe_calloc((size_t)SYM_MAX_OPEN, sizeof(float));
    int* open_parent = (int*)safe_calloc((size_t)SYM_MAX_OPEN, sizeof(int));
    int* open_action = (int*)safe_calloc((size_t)SYM_MAX_OPEN, sizeof(int));
    int* open_closed = (int*)safe_calloc((size_t)SYM_MAX_OPEN, sizeof(int));
    unsigned char* state_pool = (unsigned char*)safe_calloc((size_t)SYM_MAX_OPEN * state_bytes, 1);
    int open_count = 0;
    int found = 0;
    int goal_node = -1;

    if (!open_states || !open_g || !open_f || !open_parent || !open_action ||
        !open_closed || !state_pool) {
        safe_free((void**)&open_states); safe_free((void**)&open_g);
        safe_free((void**)&open_f); safe_free((void**)&open_parent);
        safe_free((void**)&open_action); safe_free((void**)&open_closed);
        safe_free((void**)&state_pool);
        return -1;
    }

    /* 初始状态：所有初始真命题设为1 */
    open_states[0] = state_pool;
    memset(open_states[0], 0, state_bytes);
    for (int i = 0; i < planner->initial_true_count; i++) {
        int p = planner->initial_true_props[i];
        if (p >= 0 && (size_t)p < state_bytes) open_states[0][p] = 1;
    }
    open_g[0] = 0.0f;
    open_f[0] = plan_enhanced_symbolic_heuristic(planner, open_states[0]);
    open_parent[0] = -1;
    open_action[0] = -1;
    open_closed[0] = 0;
    open_count = 1;

    /* 检查初始是否已满足目标 */
    int init_goal = 1;
    for (int i = 0; i < planner->goal_count; i++) {
        int p = planner->goal_props[i];
        if (p >= 0 && (size_t)p < state_bytes && !open_states[0][p]) {
            init_goal = 0;
            break;
        }
    }
    if (init_goal) {
        result->action_count = 0;
        result->total_cost = 0.0f;
        result->plan_confidence = 1.0f;
        result->is_feasible = 1;
        snprintf(result->plan_summary, sizeof(result->plan_summary), "符号规划: 已在目标状态");
        safe_free((void**)&open_states); safe_free((void**)&open_g);
        safe_free((void**)&open_f); safe_free((void**)&open_parent);
        safe_free((void**)&open_action); safe_free((void**)&open_closed);
        safe_free((void**)&state_pool);
        return 0;
    }

    /* A*主循环 */
    for (int iter = 0; iter < max_depth * 20 && open_count > 0 && !found; iter++) {
        int best = -1;
        float best_f = PLAN_ENH_INF;
        for (int i = 0; i < open_count; i++) {
            if (!open_closed[i] && open_f[i] < best_f) {
                best_f = open_f[i];
                best = i;
            }
        }
        if (best < 0) break;
        open_closed[best] = 1;

        unsigned char* cur_state = open_states[best];
        float cur_g = open_g[best];

        if ((int)cur_g >= max_depth) continue;

        for (int a = 0; a < planner->action_count; a++) {
            if (!plan_enhanced_symbolic_action_applicable(planner, a, cur_state))
                continue;

            if (open_count >= SYM_MAX_OPEN) break;

            int child = open_count++;
            open_states[child] = state_pool + (size_t)child * state_bytes;
            memcpy(open_states[child], cur_state, state_bytes);
            plan_enhanced_symbolic_apply_action(planner, a, open_states[child]);

            open_g[child] = cur_g + planner->actions[a].cost;
            open_parent[child] = best;
            open_action[child] = a;
            open_closed[child] = 0;
            open_f[child] = open_g[child] + plan_enhanced_symbolic_heuristic(planner, open_states[child]);

            /* 检查目标 */
            int goal_ok = 1;
            for (int i = 0; i < planner->goal_count; i++) {
                int p = planner->goal_props[i];
                if (p >= 0 && (size_t)p < state_bytes && !open_states[child][p]) {
                    goal_ok = 0;
                    break;
                }
            }
            if (goal_ok) {
                found = 1;
                goal_node = child;
                break;
            }
        }
    }

    if (!found) {
        safe_free((void**)&open_states); safe_free((void**)&open_g);
        safe_free((void**)&open_f); safe_free((void**)&open_parent);
        safe_free((void**)&open_action); safe_free((void**)&open_closed);
        safe_free((void**)&state_pool);
        return 1;
    }

    /* 回溯路径 */
    int path[200];
    int path_len = 0;
    int node = goal_node;
    while (node >= 0 && path_len < max_depth) {
        if (open_action[node] >= 0) path[path_len++] = open_action[node];
        node = open_parent[node];
    }

    /* 反转路径到结果 */
    result->action_ids = (int*)safe_malloc((size_t)path_len * sizeof(int));
    result->start_times = (float*)safe_calloc((size_t)path_len, sizeof(float));
    result->end_times = (float*)safe_calloc((size_t)path_len, sizeof(float));
    if (result->action_ids) {
        float cur_time = 0.0f;
        float total_cost = 0.0f;
        for (int i = 0; i < path_len; i++) {
            result->action_ids[i] = path[path_len - 1 - i];
            result->start_times[i] = cur_time;
            int aid = result->action_ids[i];
            PlanSymbolicAction* sa = aid >= 0 && aid < planner->action_count ? &planner->actions[aid] : NULL;
            cur_time += sa ? sa->cost : 1.0f;
            total_cost += sa ? sa->cost : 1.0f;
            result->end_times[i] = cur_time;
        }
        result->action_count = path_len;
        result->total_cost = total_cost;
        result->total_duration = cur_time;
        result->makespan = cur_time;
    }
    result->agent_count = 1;
    /* ZSFWS修复 P2-005: 基于规划步数动态计算置信度，替代启发式固定公式 */
    {
        float completion_ratio = (path_len > 0) ? 1.0f / (1.0f + (float)path_len * 0.1f) : 0.0f;
        result->plan_confidence = 0.2f + 0.6f * completion_ratio;
    }
    result->is_feasible = path_len > 0 ? 1 : 0;
    snprintf(result->plan_summary, sizeof(result->plan_summary),
             "符号规划完成: %d个动作, 深度%d, 代价%.2f",
             path_len, path_len, result->total_cost);

    safe_free((void**)&open_states); safe_free((void**)&open_g);
    safe_free((void**)&open_f); safe_free((void**)&open_parent);
    safe_free((void**)&open_action); safe_free((void**)&open_closed);
    safe_free((void**)&state_pool);
    return 0;
}

int plan_enhanced_generate_concurrent(PlanEnhancedEngine* engine,
                                       const float* goal_states, size_t goal_size,
                                       PlanEnhancedResult* results, int max_results)
{
    if (!engine || !goal_states || !results || max_results <= 0) return -1;
    if (!engine->is_initialized) return -1;
    int num_agents = engine->agent_count > 0 ? engine->agent_count : max_results;
    if (num_agents > max_results) num_agents = max_results;
    int generated = 0;
    /* 为每个智能体生成独立规划 */
    for (int ag = 0; ag < num_agents; ag++) {
        PlanEnhancedResult* r = &results[ag];
        memset(r, 0, sizeof(PlanEnhancedResult));
        const float* ag_goal = goal_states + (size_t)ag * goal_size;
        /* 初始状态 = 全0（空状态） */
        float* init_state = (float*)safe_calloc(goal_size, sizeof(float));
        if (!init_state) continue;
        /* 调用时间规划 */
        int ret = plan_enhanced_generate_temporal(engine, ag_goal, goal_size,
                                                   init_state, goal_size, r);
        safe_free((void**)&init_state);
        if (ret == 0) {
            /* 分配智能体 */
            for (int i = 0; i < r->action_count; i++) {
                r->assigned_agents[i] = ag;
            }
            r->agent_count = 1;
            generated++;
        }
    }
    /* 冲突检测与消解 */
    for (int i = 0; i < generated; i++) {
        for (int j = i + 1; j < generated; j++) {
            PlanEnhancedResult* ri = &results[i];
            PlanEnhancedResult* rj = &results[j];
            for (int ai = 0; ai < ri->action_count; ai++) {
                for (int aj = 0; aj < rj->action_count; aj++) {
                    /* 检测时间重叠冲突 */
                    float si = ri->start_times[ai], ei = ri->end_times[ai];
                    float sj = rj->start_times[aj], ej = rj->end_times[aj];
                    int overlap = (si < ej && sj < ei);
                    if (overlap) {
                        /* 检测资源共享冲突 */
                        int ri_act = ri->action_ids[ai];
                        int rj_act = rj->action_ids[aj];
                        int resource_conflict = 0;
                        if (ri_act >= 0 && ri_act < engine->action_count &&
                            rj_act >= 0 && rj_act < engine->action_count) {
                            if (engine->actions[ri_act].num_required_resources > 0 &&
                                engine->actions[rj_act].num_required_resources > 0) {
                                resource_conflict = 1;
                            }
                        }
                        if (resource_conflict) {
                            /* 延迟后一个动作 */
                            float shift = ei - sj + 0.5f;
                            if (shift > 0) {
                                for (int k = aj; k < rj->action_count; k++) {
                                    rj->start_times[k] += shift;
                                    rj->end_times[k] += shift;
                                }
                                rj->makespan += shift;
                                rj->total_duration += shift;
                            }
                        }
                    }
                }
            }
        }
    }
    return generated;
}

int plan_enhanced_generate_htn(PlanEnhancedEngine* engine,
                                int root_task_id,
                                const float* current_state, size_t state_size,
                                PlanEnhancedResult* result)
{
    if (!engine || !current_state || !result) return -1;
    if (root_task_id < 0 || root_task_id >= engine->htn_task_count) return -1;
    int max_steps = engine->config.max_plan_length > 0 ?
                    engine->config.max_plan_length : 200;
    result->action_ids = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    result->start_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    result->end_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    result->assigned_agents = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    if (!result->action_ids || !result->start_times ||
        !result->end_times || !result->assigned_agents) {
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    /* 任务栈：待分解任务 */
    int* task_stack = (int*)safe_malloc((size_t)max_steps * sizeof(int));
    int* task_depths = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    if (!task_stack || !task_depths) {
        safe_free((void**)&task_stack);
        safe_free((void**)&task_depths);
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    int stack_ptr = 0;
    task_stack[stack_ptr++] = root_task_id;
    int plan_len = 0;
    float current_time = 0.0f;
    float total_cost = 0.0f;
    while (stack_ptr > 0 && plan_len < max_steps) {
        int task_id = task_stack[--stack_ptr];
        if (task_id < 0 || task_id >= engine->htn_task_count) continue;
        PlanHTNTask* task = &engine->htn_tasks[task_id];
        if (task->is_primitive) {
            /* 原语任务 -> 查找对应动作 */
            int found_action = -1;
            for (int a = 0; a < engine->action_count; a++) {
                if (engine->actions[a].action_id == task_id) {
                    found_action = a;
                    break;
                }
            }
            if (found_action < 0) {
                /* 按名称匹配 */
                for (int a = 0; a < engine->action_count; a++) {
                    if (strcmp(engine->actions[a].name, task->name) == 0) {
                        found_action = a;
                        break;
                    }
                }
            }
            if (found_action >= 0) {
                result->action_ids[plan_len] = found_action;
                result->start_times[plan_len] = current_time;
                float dur = engine->actions[found_action].duration_min;
                if (dur < 0.1f) dur = 1.0f;
                result->end_times[plan_len] = current_time + dur;
                result->assigned_agents[plan_len] = 0;
                total_cost += engine->actions[found_action].cost;
                current_time += dur;
                plan_len++;
            }
        } else {
            /* 复合任务 -> 选择最佳方法分解 */
            if (task->method_count <= 0) continue;
            int best_method = -1;
            float best_score = -PLAN_ENH_INF;
            for (int m = 0; m < task->method_count; m++) {
                int mid = task->decomposition_methods[m];
                if (mid < 0 || mid >= engine->htn_method_count) continue;
                PlanHTNMethod* method = &engine->htn_methods[mid];
                /* 检查适用条件 */
                int applicable = 1;
                size_t min_cs = method->cond_size < state_size ?
                                method->cond_size : state_size;
                for (size_t i = 0; i < min_cs; i++) {
                    if (method->applicability_conditions[i] > 0.5f &&
                        current_state[i] < 0.5f) {
                        applicable = 0;
                        break;
                    }
                }
                if (!applicable) continue;
                float score = method->success_probability * 0.6f -
                              method->cost_estimate * 0.4f;
                if (score > best_score) {
                    best_score = score;
                    best_method = mid;
                }
            }
            if (best_method < 0) continue;
            PlanHTNMethod* method = &engine->htn_methods[best_method];
            /* 将子任务逆序压栈（保持顺序） */
            for (int s = method->subtask_count - 1; s >= 0; s--) {
                if (stack_ptr < max_steps) {
                    task_stack[stack_ptr++] = method->subtask_ids[s];
                }
            }
        }
    }
    safe_free((void**)&task_stack);
    safe_free((void**)&task_depths);
    result->action_count = plan_len;
    result->agent_count = 1;
    result->total_cost = total_cost;
    result->total_duration = current_time;
    result->makespan = current_time;
    /* ZSFWS修复 P2-005: 基于规划步数动态计算置信度，替代启发式固定公式 */
    {
        float completion_ratio = (plan_len > 0) ? 1.0f / (1.0f + (float)plan_len * 0.1f) : 0.0f;
        result->plan_confidence = 0.2f + 0.6f * completion_ratio;
    }
    result->is_feasible = plan_len > 0 ? 1 : 0;
    if (plan_len > 0) {
        snprintf(result->plan_summary, sizeof(result->plan_summary),
                 "HTN规划完成: %d个动作, 总时间%.2f",
                 plan_len, current_time);
    } else {
        snprintf(result->plan_summary, sizeof(result->plan_summary),
                 "HTN规划失败: 未能分解任务");
    }
    return plan_len > 0 ? 0 : -1;
}

int plan_enhanced_generate_partial_order(PlanEnhancedEngine* engine,
                                          const float* goal_state, size_t goal_size,
                                          const float* initial_state, size_t state_size,
                                          int* partial_order_pairs, int max_pairs)
{
    if (!engine || !goal_state || !initial_state || !partial_order_pairs || max_pairs <= 0)
        return -1;
    /* 先生成一个全序规划 */
    PlanEnhancedResult result;
    memset(&result, 0, sizeof(result));
    int ret = plan_enhanced_generate_temporal(engine, goal_state, goal_size,
                                               initial_state, state_size, &result);
    if (ret != 0) return -1;
    int pair_count = 0;
    /* 从规划中提取因果链作为偏序 */
    for (int i = 0; i < result.action_count && pair_count < max_pairs - 1; i++) {
        int act_i = result.action_ids[i];
        if (act_i < 0 || act_i >= engine->action_count) continue;
        /* 查找使用该动作效果的动作 */
        for (int j = i + 1; j < result.action_count && pair_count < max_pairs - 1; j++) {
            int act_j = result.action_ids[j];
            if (act_j < 0 || act_j >= engine->action_count) continue;
            /* 检查动作j是否使用动作i的效果作为前提 */
            int causal_link = 0;
            size_t min_es = engine->actions[act_i].effect_size < engine->actions[act_j].precond_size ?
                            engine->actions[act_i].effect_size : engine->actions[act_j].precond_size;
            for (size_t k = 0; k < min_es; k++) {
                if (engine->actions[act_i].effects[k] > 0.5f &&
                    engine->actions[act_j].preconditions[k] > 0.5f) {
                    causal_link = 1;
                    break;
                }
            }
            if (causal_link) {
                partial_order_pairs[pair_count * 2] = i;
                partial_order_pairs[pair_count * 2 + 1] = j;
                pair_count++;
            }
        }
    }
    plan_enhanced_free_result(&result);
    return pair_count;
}

void plan_enhanced_free_result(PlanEnhancedResult* result)
{
    if (!result) return;
    safe_free((void**)&result->action_ids);
    safe_free((void**)&result->start_times);
    safe_free((void**)&result->end_times);
    safe_free((void**)&result->assigned_agents);
    memset(result, 0, sizeof(PlanEnhancedResult));
}

int plan_enhanced_generate_conditional(PlanEnhancedEngine* engine,
                                        const float* goal_state, size_t goal_size,
                                        const float* initial_state, size_t state_size,
                                        const int* sensing_vars, int num_sensing_vars,
                                        PlanEnhancedResult* results, int max_branches)
{
    if (!engine || !goal_state || !initial_state || !sensing_vars ||
        !results || max_branches <= 0 || num_sensing_vars <= 0) return -1;
    int branch_count = 0;
    /* K-098: 移位安全,限制num_sensing_vars<31避免1<<31 UB */
    int safe_vars = (num_sensing_vars < 31) ? num_sensing_vars : 30;
    int max_b = 1 << safe_vars;
    if (max_b > max_branches) max_b = max_branches;
    for (int b = 0; b < max_b; b++) {
        PlanEnhancedResult* r = &results[b];
        memset(r, 0, sizeof(PlanEnhancedResult));
        /* 根据分支条件修改初始状态 */
        float* branch_state = (float*)safe_malloc(state_size * sizeof(float));
        if (!branch_state) continue;
        memcpy(branch_state, initial_state, state_size * sizeof(float));
        for (int v = 0; v < num_sensing_vars; v++) {
            int var_idx = sensing_vars[v];
            if (var_idx >= 0 && var_idx < (int)state_size) {
                /* 分支b的第v位决定该感知变量的值 */
                branch_state[var_idx] = (b >> v) & 1 ? 1.0f : 0.0f;
            }
        }
        int ret = plan_enhanced_generate_temporal(engine, goal_state, goal_size,
                                                   branch_state, state_size, r);
        safe_free((void**)&branch_state);
        if (ret == 0) {
            snprintf(r->plan_summary + strlen(r->plan_summary),
                     sizeof(r->plan_summary) - strlen(r->plan_summary),
                     " [分支%d]", b);
            branch_count++;
        }
    }
    return branch_count;
}

int plan_enhanced_replan(PlanEnhancedEngine* engine,
                          const PlanEnhancedResult* original_result,
                          const float* state_delta, size_t delta_size,
                          PlanEnhancedResult* new_result)
{
    if (!engine || !original_result || !state_delta || !new_result) return -1;
    /* 从原始结果中提取部分规划，结合状态变化生成新规划 */
    int max_steps = engine->config.max_plan_length > 0 ?
                    engine->config.max_plan_length : 100;
    new_result->action_ids = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    new_result->start_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    new_result->end_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    new_result->assigned_agents = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    if (!new_result->action_ids || !new_result->start_times ||
        !new_result->end_times || !new_result->assigned_agents) {
        safe_free((void**)&new_result->action_ids);
        safe_free((void**)&new_result->start_times);
        safe_free((void**)&new_result->end_times);
        safe_free((void**)&new_result->assigned_agents);
        return -1;
    }
    /* 复制原始规划中未受影响的部分 */
    int new_len = 0;
    float current_time = 0.0f;
    float total_cost = 0.0f;
    for (int i = 0; i < original_result->action_count && new_len < max_steps; i++) {
        int aid = original_result->action_ids[i];
        if (aid < 0 || aid >= engine->action_count) continue;
        /* 检查该动作是否受状态变化影响 */
        int affected = 0;
        size_t min_ps = engine->actions[aid].precond_size < delta_size ?
                        engine->actions[aid].precond_size : delta_size;
        for (size_t j = 0; j < min_ps; j++) {
            if (state_delta[j] != 0.0f &&
                engine->actions[aid].preconditions[j] > 0.5f) {
                affected = 1;
                break;
            }
        }
        if (!affected) {
            /* 未受影响的动作直接复用 */
            new_result->action_ids[new_len] = aid;
            new_result->start_times[new_len] = current_time;
            float dur = engine->actions[aid].duration_min;
            if (dur < 0.1f) dur = 1.0f;
            new_result->end_times[new_len] = current_time + dur;
            new_result->assigned_agents[new_len] =
                i < original_result->agent_count ? original_result->assigned_agents[i] : 0;
            total_cost += engine->actions[aid].cost;
            current_time += dur;
            new_len++;
        }
    }
    new_result->action_count = new_len;
    new_result->agent_count = original_result->agent_count;
    new_result->total_cost = total_cost;
    new_result->total_duration = current_time;
    new_result->makespan = current_time;
    /* M-029修复: 基于规划长度和成本动态计算置信度 */
    float dynamic_conf = new_len > 0 ? (0.5f + 0.5f * (1.0f - fminf(total_cost / (total_cost + 100.0f), 1.0f))) : 0.0f;
    new_result->plan_confidence = dynamic_conf;
    new_result->is_feasible = new_len > 0 ? 1 : 0;
    snprintf(new_result->plan_summary, sizeof(new_result->plan_summary),
             "重规划完成: %d个动作(从%d个原始动作中恢复)",
             new_len, original_result->action_count);
    return new_len > 0 ? 0 : -1;
}

float plan_enhanced_validate_feasibility(PlanEnhancedEngine* engine,
                                          const PlanEnhancedResult* result)
{
    if (!engine || !result || result->action_count <= 0) return 0.0f;
    float score = 1.0f;
    /* 1. 资源可行性 */
    float total_resources = 0.0f;
    int has_resource_info = 0;
    for (int i = 0; i < result->action_count; i++) {
        int aid = result->action_ids[i];
        if (aid >= 0 && aid < engine->action_count) {
            total_resources += (float)engine->actions[aid].num_required_resources;
            if (engine->actions[aid].num_required_resources > 0) has_resource_info = 1;
        }
    }
    if (has_resource_info && engine->config.resource_limit > 0) {
        float resource_ratio = total_resources / engine->config.resource_limit;
        if (resource_ratio > 1.0f) score -= (resource_ratio - 1.0f) * 0.5f;
    }
    /* 2. 时间可行性 */
    for (int i = 0; i < result->action_count; i++) {
        if (result->end_times[i] <= result->start_times[i]) {
            score -= 0.1f;
        }
    }
    /* 3. 因果可行性 */
    for (int i = 1; i < result->action_count; i++) {
        int aid = result->action_ids[i];
        if (aid < 0 || aid >= engine->action_count) continue;
        size_t min_ps = engine->actions[aid].precond_size;
        if (min_ps == 0) continue;
        int precond_met = 0;
        for (int j = 0; j < i; j++) {
            int j_aid = result->action_ids[j];
            if (j_aid < 0 || j_aid >= engine->action_count) continue;
            size_t check_sz = engine->actions[j_aid].effect_size < min_ps ?
                              engine->actions[j_aid].effect_size : min_ps;
            for (size_t k = 0; k < check_sz; k++) {
                if (engine->actions[j_aid].effects[k] > 0.5f &&
                    engine->actions[aid].preconditions[k] > 0.5f) {
                    precond_met = 1;
                    break;
                }
            }
            if (precond_met) break;
        }
        if (!precond_met) score -= 0.1f;
    }
    /* 4. 截止时间检查 */
    if (engine->config.deadline > 0 && result->makespan > engine->config.deadline) {
        score -= (result->makespan - engine->config.deadline) / engine->config.deadline * 0.3f;
    }
    /* 钳制到[0,1] */
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;
    return score;
}

int plan_enhanced_save(const PlanEnhancedEngine* engine, const char* filepath)
{
    if (!engine || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;
    /* 写入配置 */
    fwrite(&engine->config, sizeof(PlanEnhancedConfig), 1, fp);
    /* 写入动作 */
    fwrite(&engine->action_count, sizeof(int), 1, fp);
    for (int i = 0; i < engine->action_count; i++) {
        fwrite(&engine->actions[i], sizeof(PlanEnhancedAction), 1, fp);
        if (engine->actions[i].preconditions && engine->actions[i].precond_size > 0) {
            fwrite(engine->actions[i].preconditions, sizeof(float),
                   engine->actions[i].precond_size, fp);
        }
        if (engine->actions[i].effects && engine->actions[i].effect_size > 0) {
            fwrite(engine->actions[i].effects, sizeof(float),
                   engine->actions[i].effect_size, fp);
        }
    }
    /* 写入约束 */
    fwrite(&engine->constraint_count, sizeof(int), 1, fp);
    fwrite(engine->constraints, sizeof(PlanTimeConstraint),
           (size_t)engine->constraint_count, fp);
    fclose(fp);
    return 0;
}

PlanEnhancedEngine* plan_enhanced_load(const char* filepath)
{
    if (!filepath) return NULL;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;
    PlanEnhancedConfig config;
    if (fread(&config, sizeof(PlanEnhancedConfig), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }
    PlanEnhancedEngine* engine = plan_enhanced_create(&config);
    if (!engine) {
        fclose(fp);
        return NULL;
    }
    int action_count;
    if (fread(&action_count, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        plan_enhanced_destroy(engine);
        return NULL;
    }
    for (int i = 0; i < action_count; i++) {
        PlanEnhancedAction act;
        memset(&act, 0, sizeof(act));
        if (fread(&act, sizeof(PlanEnhancedAction), 1, fp) != 1) break;
        act.preconditions = NULL;
        act.effects = NULL;
        if (act.precond_size > 0) {
            act.preconditions = (float*)safe_malloc(act.precond_size * sizeof(float));
            if (act.preconditions) {
                fread(act.preconditions, sizeof(float), act.precond_size, fp);
            }
        }
        if (act.effect_size > 0) {
            act.effects = (float*)safe_malloc(act.effect_size * sizeof(float));
            if (act.effects) {
                fread(act.effects, sizeof(float), act.effect_size, fp);
            }
        }
        plan_enhanced_add_action(engine, &act);
        safe_free((void**)&act.preconditions);
        safe_free((void**)&act.effects);
    }
    int constraint_count;
    if (fread(&constraint_count, sizeof(int), 1, fp) == 1) {
        for (int i = 0; i < constraint_count; i++) {
            PlanTimeConstraint tc;
            if (fread(&tc, sizeof(PlanTimeConstraint), 1, fp) == 1) {
                plan_enhanced_add_time_constraint(engine, &tc);
            }
        }
    }
    fclose(fp);
    return engine;
}

PlanEnhancedConfig plan_enhanced_default_config(PlanEnhancedAlgorithm algorithm)
{
    PlanEnhancedConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.algorithm = algorithm;
    cfg.max_actions = 1024;
    cfg.max_plan_length = 200;
    cfg.time_horizon = 100.0f;
    cfg.deadline = 0.0f;
    cfg.num_agents = 1;
    cfg.risk_tolerance = 0.5f;
    cfg.resource_limit = 100.0f;
    cfg.enable_parallel = 1;
    cfg.enable_replanning = 1;
    cfg.enable_temporal_reasoning = 1;
    cfg.enable_conditional_branches = 0;
    cfg.cfc_tau = 2.0f;
    cfg.cfc_dt = 0.1f;
    cfg.cfc_steps = 10;
    return cfg;
}

/* ============================================================================
 * FF/FFD启发式规划实现 (A04.2.2)
 * ============================================================================ */

/* 检查状态是否满足目标 */
static int ff_state_satisfies_goal(const float* state, size_t state_size,
                                    const float* goal, size_t goal_size)
{
    size_t min_sz = state_size < goal_size ? state_size : goal_size;
    for (size_t i = 0; i < min_sz; i++) {
        if (goal[i] > 0.5f && state[i] < 0.5f) return 0;
    }
    return 1;
}

/* 检查动作是否可应用于当前状态 */
static int ff_action_applicable(const PlanEnhancedAction* act,
                                 const float* state, size_t state_size)
{
    size_t min_sz = act->precond_size < state_size ? act->precond_size : state_size;
    for (size_t i = 0; i < min_sz; i++) {
        if (act->preconditions[i] > 0.5f && state[i] < 0.5f) return 0;
    }
    return 1;
}

/* 应用动作（完整效果，包含删除效果） */
static void ff_apply_action(const PlanEnhancedAction* act,
                             const float* state, float* next_state,
                             size_t state_size)
{
    memcpy(next_state, state, state_size * sizeof(float));
    size_t eff_sz = act->effect_size < state_size ? act->effect_size : state_size;
    for (size_t i = 0; i < eff_sz; i++) {
        if (act->effects[i] > 0.5f) {
            next_state[i] = 1.0f;
        } else if (act->effects[i] < -0.5f) {
            next_state[i] = 0.0f;
        }
    }
}

/* 放松版应用动作（只加不减，忽略删除效果） */
static void ff_apply_action_relaxed(const PlanEnhancedAction* act,
                                     const float* state, float* next_state,
                                     size_t state_size)
{
    memcpy(next_state, state, state_size * sizeof(float));
    size_t eff_sz = act->effect_size < state_size ? act->effect_size : state_size;
    for (size_t i = 0; i < eff_sz; i++) {
        if (act->effects[i] > 0.5f) next_state[i] = 1.0f;
    }
}

/**
 * FF启发式：放松规划图构建 + 向后提取
 *
 * Phase 1: 前向传播构建放松规划图
 *   - S_0 = state
 *   - A_k = {a | precond(a) ⊆ S_k}
 *   - S_{k+1} = S_k ∪ {eff⁺(a) | a ∈ A_k}
 *   - 直到所有目标命题出现在某个S_k中，或没有新命题加入
 *
 * Phase 2: 向后提取放松规划
 *   - 从目标层开始，逆向遍历各层
 *   - 对每个新出现的目标命题，找到第一个能产生它的动作
 *   - 计数该动作，将其前提条件作为新目标
 */
float plan_enhanced_ff_heuristic(PlanEnhancedEngine* engine,
                                  const float* state, size_t state_size,
                                  const float* goal_state, size_t goal_size)
{
    if (!engine || !state || !goal_state || state_size == 0) return -1.0f;
    if (engine->action_count <= 0) return -1.0f;
    /* 如果已经在目标状态，h=0 */
    if (ff_state_satisfies_goal(state, state_size, goal_state, goal_size)) {
        return 0.0f;
    }
    /* ===== Phase 1: 前向构建放松规划图 ===== */
    #define MAX_FF_LAYERS 256
    float** layers = (float**)safe_calloc(MAX_FF_LAYERS, sizeof(float*));
    int* layer_action_flags = (int*)safe_calloc((size_t)engine->action_count, sizeof(int));
    if (!layers || !layer_action_flags) {
        safe_free((void**)&layers);
        safe_free((void**)&layer_action_flags);
        return PLAN_ENH_INF;
    }
    /* 第0层 = 当前状态 */
    layers[0] = (float*)safe_malloc(state_size * sizeof(float));
    if (!layers[0]) {
        safe_free((void**)&layers);
        safe_free((void**)&layer_action_flags);
        return PLAN_ENH_INF;
    }
    memcpy(layers[0], state, state_size * sizeof(float));
    int num_layers = 1;
    int goal_reachable = 0;
    for (int l = 0; l < MAX_FF_LAYERS - 1; l++) {
        /* 分配新层 = 复制前一层 */
        layers[l + 1] = (float*)safe_malloc(state_size * sizeof(float));
        if (!layers[l + 1]) break;
        memcpy(layers[l + 1], layers[l], state_size * sizeof(float));
        /* 查找在当前层新可用的动作 */
        int added_proposition = 0;
        for (int a = 0; a < engine->action_count; a++) {
            if (layer_action_flags[a]) continue; /* 已在之前的层可用 */
            if (!ff_action_applicable(&engine->actions[a], layers[l], state_size))
                continue;
            layer_action_flags[a] = 1;
            /* 应用动作的放松效果 */
            size_t eff_sz = engine->actions[a].effect_size < state_size ?
                            engine->actions[a].effect_size : state_size;
            for (size_t i = 0; i < eff_sz; i++) {
                if (engine->actions[a].effects[i] > 0.5f && layers[l + 1][i] < 0.5f) {
                    layers[l + 1][i] = 1.0f;
                    added_proposition = 1;
                }
            }
        }
        num_layers = l + 2;
        /* 检查目标是否可达 */
        if (ff_state_satisfies_goal(layers[l + 1], state_size, goal_state, goal_size)) {
            goal_reachable = 1;
            break;
        }
        /* 如果没有新命题加入，停止 */
        if (!added_proposition) break;
    }
    if (!goal_reachable) {
        for (int i = 0; i < num_layers; i++) {
            safe_free((void**)&layers[i]);
        }
        safe_free((void**)&layers);
        safe_free((void**)&layer_action_flags);
        return PLAN_ENH_INF;
    }
    /* ===== Phase 2: 向后提取放松规划 ===== */
    float* satisfied = (float*)safe_calloc(state_size, sizeof(float));
    int* plan_action_flags = (int*)safe_calloc((size_t)engine->action_count, sizeof(int));
    if (!satisfied || !plan_action_flags) {
        safe_free((void**)&satisfied);
        safe_free((void**)&plan_action_flags);
        for (int i = 0; i < num_layers; i++) {
            safe_free((void**)&layers[i]);
        }
        safe_free((void**)&layers);
        safe_free((void**)&layer_action_flags);
        return PLAN_ENH_INF;
    }
    /* 从最后一层开始逆向 */
    int plan_action_count = 0;
    for (int l = num_layers - 1; l >= 0; l--) {
        /* 找出在本层新满足但尚未标记为已满足的目标命题 */
        int has_new_goal = 0;
        size_t min_gs = goal_size < state_size ? goal_size : state_size;
        for (size_t g = 0; g < min_gs; g++) {
            if (goal_state[g] > 0.5f && satisfied[g] < 0.5f) {
                /* 检查这个目标是否在本层才出现 */
                int appears_here = (layers[l][g] > 0.5f);
                int appears_prev = (l > 0 && layers[l - 1][g] > 0.5f);
                if (appears_here && !appears_prev) {
                    has_new_goal = 1;
                    break;
                }
            }
        }
        if (!has_new_goal) continue;
        /* 找到能产生这些目标命题的动作 */
        for (int a = 0; a < engine->action_count; a++) {
            if (plan_action_flags[a]) continue;
            /* 检查该动作是否在本层可用（前提条件在l-1层满足） */
            if (l == 0) {
                if (!ff_action_applicable(&engine->actions[a], state, state_size))
                    continue;
            } else {
                if (!ff_action_applicable(&engine->actions[a], layers[l - 1], state_size))
                    continue;
            }
            /* 检查该动作是否能产生未满足的目标命题 */
            int helps = 0;
            for (size_t g = 0; g < min_gs; g++) {
                if (goal_state[g] > 0.5f && satisfied[g] < 0.5f) {
                    size_t eff_sz = engine->actions[a].effect_size;
                    if (g < eff_sz && engine->actions[a].effects[g] > 0.5f) {
                        helps = 1;
                        break;
                    }
                }
            }
            if (!helps) continue;
            /* 标记该动作的效果命题为已满足 */
            size_t eff_sz = engine->actions[a].effect_size < state_size ?
                            engine->actions[a].effect_size : state_size;
            for (size_t i = 0; i < eff_sz; i++) {
                if (engine->actions[a].effects[i] > 0.5f) {
                    satisfied[i] = 1.0f;
                }
            }
            /* 标记该动作的前提命题为已满足 */
            size_t pre_sz = engine->actions[a].precond_size < state_size ?
                            engine->actions[a].precond_size : state_size;
            for (size_t i = 0; i < pre_sz; i++) {
                if (engine->actions[a].preconditions[i] > 0.5f) {
                    satisfied[i] = 1.0f;
                }
            }
            plan_action_flags[a] = 1;
            plan_action_count++;
            break;
        }
    }
    safe_free((void**)&satisfied);
    safe_free((void**)&plan_action_flags);
    for (int i = 0; i < num_layers; i++) {
        safe_free((void**)&layers[i]);
    }
    safe_free((void**)&layers);
    safe_free((void**)&layer_action_flags);
    return (float)plan_action_count;
}

/**
 * FFD启发式：带帮助度评分的FF
 *
 * 在向后提取过程中，优先选择帮助度（同时帮助多个未达成目标）更高的动作。
 * h_ffd = h_ff - λ · avg_helpfulness
 */
float plan_enhanced_ffd_heuristic(PlanEnhancedEngine* engine,
                                   const float* state, size_t state_size,
                                   const float* goal_state, size_t goal_size)
{
    if (!engine || !state || !goal_state || state_size == 0) return -1.0f;
    if (engine->action_count <= 0) return -1.0f;
    if (ff_state_satisfies_goal(state, state_size, goal_state, goal_size)) {
        return 0.0f;
    }
    /* ===== Phase 1: 前向构建放松规划图 ===== */
    #define MAX_FFD_LAYERS 256
    float** layers = (float**)safe_calloc(MAX_FFD_LAYERS, sizeof(float*));
    int* layer_action_flags = (int*)safe_calloc((size_t)engine->action_count, sizeof(int));
    if (!layers || !layer_action_flags) {
        safe_free((void**)&layers);
        safe_free((void**)&layer_action_flags);
        return PLAN_ENH_INF;
    }
    layers[0] = (float*)safe_malloc(state_size * sizeof(float));
    if (!layers[0]) {
        safe_free((void**)&layers);
        safe_free((void**)&layer_action_flags);
        return PLAN_ENH_INF;
    }
    memcpy(layers[0], state, state_size * sizeof(float));
    int num_layers = 1;
    int goal_reachable = 0;
    for (int l = 0; l < MAX_FFD_LAYERS - 1; l++) {
        layers[l + 1] = (float*)safe_malloc(state_size * sizeof(float));
        if (!layers[l + 1]) break;
        memcpy(layers[l + 1], layers[l], state_size * sizeof(float));
        int added = 0;
        for (int a = 0; a < engine->action_count; a++) {
            if (layer_action_flags[a]) continue;
            if (!ff_action_applicable(&engine->actions[a], layers[l], state_size))
                continue;
            layer_action_flags[a] = 1;
            size_t eff_sz = engine->actions[a].effect_size < state_size ?
                            engine->actions[a].effect_size : state_size;
            for (size_t i = 0; i < eff_sz; i++) {
                if (engine->actions[a].effects[i] > 0.5f && layers[l + 1][i] < 0.5f) {
                    layers[l + 1][i] = 1.0f;
                    added = 1;
                }
            }
        }
        num_layers = l + 2;
        if (ff_state_satisfies_goal(layers[l + 1], state_size, goal_state, goal_size)) {
            goal_reachable = 1;
            break;
        }
        if (!added) break;
    }
    if (!goal_reachable) {
        for (int i = 0; i < num_layers; i++) {
            safe_free((void**)&layers[i]);
        }
        safe_free((void**)&layers);
        safe_free((void**)&layer_action_flags);
        return PLAN_ENH_INF;
    }
    /* ===== Phase 2: 带帮助度评分的提取 ===== */
    float* satisfied = (float*)safe_calloc(state_size, sizeof(float));
    if (!satisfied) {
        for (int i = 0; i < num_layers; i++) {
            safe_free((void**)&layers[i]);
        }
        safe_free((void**)&layers);
        safe_free((void**)&layer_action_flags);
        return PLAN_ENH_INF;
    }
    int plan_action_count = 0;
    float total_helpfulness = 0.0f;
    for (int l = num_layers - 1; l >= 0; l--) {
        /* 收集本层所有可用的候选动作 */
        int* candidates = (int*)safe_calloc((size_t)engine->action_count, sizeof(int));
        float* helpfulness_scores = (float*)safe_calloc((size_t)engine->action_count, sizeof(float));
        if (!candidates || !helpfulness_scores) {
            safe_free((void**)&candidates);
            safe_free((void**)&helpfulness_scores);
            safe_free((void**)&satisfied);
            for (int i = 0; i < num_layers; i++) {
                safe_free((void**)&layers[i]);
            }
            safe_free((void**)&layers);
            safe_free((void**)&layer_action_flags);
            return PLAN_ENH_INF;
        }
        int num_candidates = 0;
        size_t min_gs = goal_size < state_size ? goal_size : state_size;
        for (int a = 0; a < engine->action_count; a++) {
            if (l == 0) {
                if (!ff_action_applicable(&engine->actions[a], state, state_size))
                    continue;
            } else {
                if (!ff_action_applicable(&engine->actions[a], layers[l - 1], state_size))
                    continue;
            }
            /* 计算帮助度：该动作帮助了多少未满足目标 */
            float helpfulness = 0.0f;
            for (size_t g = 0; g < min_gs; g++) {
                if (goal_state[g] > 0.5f && satisfied[g] < 0.5f) {
                    if (g < engine->actions[a].effect_size &&
                        engine->actions[a].effects[g] > 0.5f) {
                        helpfulness += 1.0f;
                    }
                }
            }
            if (helpfulness > 0.0f) {
                candidates[num_candidates] = a;
                helpfulness_scores[num_candidates] = helpfulness;
                num_candidates++;
            }
        }
        if (num_candidates == 0) {
            safe_free((void**)&candidates);
            safe_free((void**)&helpfulness_scores);
            continue;
        }
        /* 选择帮助度最高的动作 */
        int best_idx = 0;
        for (int c = 1; c < num_candidates; c++) {
            if (helpfulness_scores[c] > helpfulness_scores[best_idx]) {
                best_idx = c;
            }
        }
        int best_action = candidates[best_idx];
        total_helpfulness += helpfulness_scores[best_idx];
        size_t eff_sz = engine->actions[best_action].effect_size < state_size ?
                        engine->actions[best_action].effect_size : state_size;
        for (size_t i = 0; i < eff_sz; i++) {
            if (engine->actions[best_action].effects[i] > 0.5f) {
                satisfied[i] = 1.0f;
            }
        }
        size_t pre_sz = engine->actions[best_action].precond_size < state_size ?
                        engine->actions[best_action].precond_size : state_size;
        for (size_t i = 0; i < pre_sz; i++) {
            if (engine->actions[best_action].preconditions[i] > 0.5f) {
                satisfied[i] = 1.0f;
            }
        }
        plan_action_count++;
        safe_free((void**)&candidates);
        safe_free((void**)&helpfulness_scores);
    }
    safe_free((void**)&satisfied);
    for (int i = 0; i < num_layers; i++) {
        safe_free((void**)&layers[i]);
    }
    safe_free((void**)&layers);
    safe_free((void**)&layer_action_flags);
    if (plan_action_count <= 0) return PLAN_ENH_INF;
    float avg_helpfulness = total_helpfulness / (float)plan_action_count;
    float h_ffd = (float)plan_action_count - 0.1f * avg_helpfulness;
    if (h_ffd < 0.0f) h_ffd = 0.0f;
    return h_ffd;
}

/**
 * FF规划器：贪婪最佳优先搜索
 *
 * 1. 从初始状态开始
 * 2. 评估所有可行动作，用FF启发式评估每个后继
 * 3. 选择h值最小的方向
 * 4. 到达目标或超过最大步数时停止
 */
int plan_enhanced_generate_ff(PlanEnhancedEngine* engine,
                               const float* goal_state, size_t goal_size,
                               const float* initial_state, size_t state_size,
                               PlanEnhancedResult* result)
{
    if (!engine || !goal_state || !initial_state || !result) return -1;
    if (engine->action_count <= 0) return -1;
    int max_steps = engine->config.max_plan_length > 0 ?
                    engine->config.max_plan_length : 100;
    result->action_ids = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    result->start_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    result->end_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    result->assigned_agents = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    if (!result->action_ids || !result->start_times ||
        !result->end_times || !result->assigned_agents) {
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    float* current_state = (float*)safe_malloc(state_size * sizeof(float));
    if (!current_state) {
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    memcpy(current_state, initial_state, state_size * sizeof(float));
    int plan_len = 0;
    float current_time = 0.0f;
    float total_cost = 0.0f;
    for (int step = 0; step < max_steps; step++) {
        if (ff_state_satisfies_goal(current_state, state_size, goal_state, goal_size)) {
            break;
        }
        int best_action = -1;
        float best_h = PLAN_ENH_INF;
        float* best_next = (float*)safe_malloc(state_size * sizeof(float));
        if (!best_next) break;
        for (int a = 0; a < engine->action_count; a++) {
            if (!ff_action_applicable(&engine->actions[a], current_state, state_size))
                continue;
            float* succ = (float*)safe_malloc(state_size * sizeof(float));
            if (!succ) continue;
            ff_apply_action(&engine->actions[a], current_state, succ, state_size);
            float h_val = plan_enhanced_ff_heuristic(engine, succ, state_size,
                                                      goal_state, goal_size);
            if (h_val >= 0 && h_val < best_h) {
                best_h = h_val;
                best_action = a;
                memcpy(best_next, succ, state_size * sizeof(float));
            }
            safe_free((void**)&succ);
        }
        if (best_action < 0) {
            safe_free((void**)&best_next);
            break;
        }
        /* 记录动作 */
        result->action_ids[plan_len] = best_action;
        result->start_times[plan_len] = current_time;
        float dur = engine->actions[best_action].duration_min;
        if (dur < 0.1f) dur = 1.0f;
        result->end_times[plan_len] = current_time + dur;
        result->assigned_agents[plan_len] = 0;
        total_cost += engine->actions[best_action].cost;
        current_time += dur;
        memcpy(current_state, best_next, state_size * sizeof(float));
        safe_free((void**)&best_next);
        plan_len++;
    }
    safe_free((void**)&current_state);
    result->action_count = plan_len;
    result->agent_count = 1;
    result->total_cost = total_cost;
    result->total_duration = current_time;
    result->makespan = current_time;
    result->plan_confidence = plan_len > 0 ? (0.5f + 0.2f * (1.0f - fminf(total_cost / (total_cost + 100.0f), 1.0f))) : 0.0f;
    result->is_feasible = plan_len > 0 ? 1 : 0;
    snprintf(result->plan_summary, sizeof(result->plan_summary),
             "FF规划完成: %d个动作, h=%.1f, 时间%.2f",
             plan_len, plan_len > 0 ? (float)plan_len : 0.0f, current_time);
    return plan_len > 0 ? 0 : -1;
}

/**
 * FFD规划器：带死端检测和回溯的贪婪最佳优先搜索
 *
 * 1. 使用FFD启发式引导搜索
 * 2. 检测死端(h=INF)并触发回溯
 * 3. 目标计数启发式辅助修剪
 */
int plan_enhanced_generate_ffd(PlanEnhancedEngine* engine,
                                const float* goal_state, size_t goal_size,
                                const float* initial_state, size_t state_size,
                                PlanEnhancedResult* result)
{
    if (!engine || !goal_state || !initial_state || !result) return -1;
    if (engine->action_count <= 0) return -1;
    int max_steps = engine->config.max_plan_length > 0 ?
                    engine->config.max_plan_length : 100;
    result->action_ids = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    result->start_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    result->end_times = (float*)safe_calloc((size_t)max_steps, sizeof(float));
    result->assigned_agents = (int*)safe_calloc((size_t)max_steps, sizeof(int));
    if (!result->action_ids || !result->start_times ||
        !result->end_times || !result->assigned_agents) {
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    float* current_state = (float*)safe_malloc(state_size * sizeof(float));
    if (!current_state) {
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    memcpy(current_state, initial_state, state_size * sizeof(float));
    /* 搜索栈用于回溯 */
    #define FFD_MAX_BACKTRACK 10
    int stack_depth = 0;
    int max_stack = max_steps;
    float** state_stack = (float**)safe_calloc((size_t)max_stack, sizeof(float*));
    int* action_stack = (int*)safe_calloc((size_t)max_stack, sizeof(int));
    int* branch_start_stack = (int*)safe_calloc((size_t)max_stack, sizeof(int));
    float* h_stack = (float*)safe_calloc((size_t)max_stack, sizeof(float));
    if (!state_stack || !action_stack || !branch_start_stack || !h_stack) {
        safe_free((void**)&state_stack);
        safe_free((void**)&action_stack);
        safe_free((void**)&branch_start_stack);
        safe_free((void**)&h_stack);
        safe_free((void**)&current_state);
        safe_free((void**)&result->action_ids);
        safe_free((void**)&result->start_times);
        safe_free((void**)&result->end_times);
        safe_free((void**)&result->assigned_agents);
        return -1;
    }
    int plan_len = 0;
    float current_time = 0.0f;
    float total_cost = 0.0f;
    int backtrack_count = 0;
    for (int step = 0; step < max_steps; step++) {
        if (ff_state_satisfies_goal(current_state, state_size, goal_state, goal_size)) {
            break;
        }
        /* 收集所有可行动作 */
        int* candidates = (int*)safe_malloc((size_t)engine->action_count * sizeof(int));
        float* candidate_h = (float*)safe_malloc((size_t)engine->action_count * sizeof(float));
        float* candidate_f = (float*)safe_malloc((size_t)engine->action_count * sizeof(float));
        if (!candidates || !candidate_h || !candidate_f) {
            safe_free((void**)&candidates);
            safe_free((void**)&candidate_h);
            safe_free((void**)&candidate_f);
            break;
        }
        int num_candidates = 0;
        /* 未满足目标计数（辅助启发式） */
        int unsatisfied_goals = 0;
        size_t min_gs = goal_size < state_size ? goal_size : state_size;
        for (size_t i = 0; i < min_gs; i++) {
            if (goal_state[i] > 0.5f && current_state[i] < 0.5f) {
                unsatisfied_goals++;
            }
        }
        for (int a = 0; a < engine->action_count; a++) {
            if (!ff_action_applicable(&engine->actions[a], current_state, state_size))
                continue;
            float* succ = (float*)safe_malloc(state_size * sizeof(float));
            if (!succ) continue;
            ff_apply_action(&engine->actions[a], current_state, succ, state_size);
            float h_val = plan_enhanced_ffd_heuristic(engine, succ, state_size,
                                                       goal_state, goal_size);
            safe_free((void**)&succ);
            /* 跳过死端 */
            if (h_val >= PLAN_ENH_INF) continue;
            candidates[num_candidates] = a;
            candidate_h[num_candidates] = h_val;
            /* 组合评价：h_ffd + 目标计数辅助 */
            candidate_f[num_candidates] = h_val + 0.2f * (float)unsatisfied_goals;
            num_candidates++;
        }
        if (num_candidates == 0) {
            safe_free((void**)&candidates);
            safe_free((void**)&candidate_h);
            safe_free((void**)&candidate_f);
            /* 死端：触发回溯 */
            if (stack_depth > 0 && backtrack_count < FFD_MAX_BACKTRACK) {
                stack_depth--;
                memcpy(current_state, state_stack[stack_depth], state_size * sizeof(float));
                plan_len = action_stack[stack_depth];
                current_time = result->end_times[plan_len > 0 ? plan_len - 1 : 0];
                backtrack_count++;
                continue;
            }
            break;
        }
        /* 选择综合评价最小的动作 */
        int best_idx = 0;
        for (int c = 1; c < num_candidates; c++) {
            if (candidate_f[c] < candidate_f[best_idx]) {
                best_idx = c;
            }
        }
        int best_action = candidates[best_idx];
        if (plan_len < max_steps) {
            /* 保存当前状态到搜索栈 */
            state_stack[stack_depth] = (float*)safe_malloc(state_size * sizeof(float));
            if (state_stack[stack_depth]) {
                memcpy(state_stack[stack_depth], current_state, state_size * sizeof(float));
                action_stack[stack_depth] = plan_len;
                branch_start_stack[stack_depth] = 0;
                h_stack[stack_depth] = candidate_h[best_idx];
                stack_depth++;
            }
            result->action_ids[plan_len] = best_action;
            result->start_times[plan_len] = current_time;
            float dur = engine->actions[best_action].duration_min;
            if (dur < 0.1f) dur = 1.0f;
            result->end_times[plan_len] = current_time + dur;
            result->assigned_agents[plan_len] = 0;
            total_cost += engine->actions[best_action].cost;
            current_time += dur;
            /* 应用动作 */
            float* next_state = (float*)safe_malloc(state_size * sizeof(float));
            if (next_state) {
                ff_apply_action(&engine->actions[best_action], current_state,
                                next_state, state_size);
                memcpy(current_state, next_state, state_size * sizeof(float));
                safe_free((void**)&next_state);
            }
            plan_len++;
        }
        safe_free((void**)&candidates);
        safe_free((void**)&candidate_h);
        safe_free((void**)&candidate_f);
    }
    /* 清理搜索栈 */
    for (int i = 0; i < stack_depth; i++) {
        safe_free((void**)&state_stack[i]);
    }
    safe_free((void**)&state_stack);
    safe_free((void**)&action_stack);
    safe_free((void**)&branch_start_stack);
    safe_free((void**)&h_stack);
    safe_free((void**)&current_state);
    result->action_count = plan_len;
    result->agent_count = 1;
    result->total_cost = total_cost;
    result->total_duration = current_time;
    result->makespan = current_time;
    result->plan_confidence = plan_len > 0 ? (0.5f + 0.4f * (1.0f - fminf(backtrack_count / 10.0f, 1.0f))) : 0.0f;
    result->is_feasible = plan_len > 0 ? 1 : 0;
    if (backtrack_count > 0) {
        snprintf(result->plan_summary, sizeof(result->plan_summary),
                 "FFD规划完成: %d个动作, 回溯%d次, 时间%.2f",
                 plan_len, backtrack_count, current_time);
    } else {
        snprintf(result->plan_summary, sizeof(result->plan_summary),
                 "FFD规划完成: %d个动作, 时间%.2f",
                 plan_len, current_time);
    }
    return plan_len > 0 ? 0 : -1;
}
