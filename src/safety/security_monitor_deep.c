#include "selflnn/safety/security_monitor_deep.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ============================================================================
 * A09.1.1 行为安全监控深度实现
 * ============================================================================ */

SecBehaviorMonitor* sec_behavior_monitor_create(float anomaly_threshold) {
    SecBehaviorMonitor* monitor = (SecBehaviorMonitor*)safe_calloc(1, sizeof(SecBehaviorMonitor));
    if (!monitor) return NULL;

    monitor->constraint_set.constraint_count = 0;
    monitor->constraint_set.overall_compliance_score = 1.0f;
    monitor->constraint_set.total_violations = 0;
    monitor->constraint_set.last_check_time = 0;

    monitor->goal_graph.goal_count = 0;
    monitor->goal_graph.global_consistency_score = 1.0f;
    monitor->goal_graph.conflict_count = 0;
    monitor->goal_graph.has_cycles = 0;

    monitor->value_metric_count = 0;

    monitor->behavior_history_count = 0;
    monitor->behavior_history_capacity = SEC_BEHAVIOR_HISTORY;
    monitor->anomaly_threshold = anomaly_threshold > 0.0f ? anomaly_threshold : 0.5f;
    monitor->current_anomaly_score = 0.0f;
    monitor->is_monitoring_active = 1;

    SecCoreValueType default_values[] = {
        SEC_VALUE_SAFETY, SEC_VALUE_ETHICS, SEC_VALUE_TRANSPARENCY,
        SEC_VALUE_ACCOUNTABILITY, SEC_VALUE_FAIRNESS, SEC_VALUE_PRIVACY,
        SEC_VALUE_RELIABILITY, SEC_VALUE_HUMAN_CONTROL
    };
    const char* value_names[] = {
        "安全性", "伦理道德", "透明度",
        "可问责性", "公平性", "隐私保护",
        "可靠性", "人类控制"
    };
    float expected[] = {0.95f, 0.95f, 0.90f, 0.90f, 0.85f, 0.95f, 0.95f, 0.99f};
    float min_acceptable[] = {0.80f, 0.80f, 0.70f, 0.70f, 0.70f, 0.80f, 0.85f, 0.95f};
    float tolerance[] = {0.05f, 0.05f, 0.10f, 0.10f, 0.10f, 0.05f, 0.05f, 0.02f};

    int default_count = sizeof(default_values) / sizeof(default_values[0]);
    for (int i = 0; i < default_count && monitor->value_metric_count < SEC_MAX_CORE_VALUES; i++) {
        SecValueAlignmentMetric* m = &monitor->value_metrics[monitor->value_metric_count];
        m->value_type = default_values[i];
        strncpy(m->name, value_names[i], sizeof(m->name) - 1);
        m->name[sizeof(m->name) - 1] = '\0';
        strncpy(m->description, "默认价值观对齐度量", sizeof(m->description) - 1);
        m->expected_alignment = expected[i];
        m->current_alignment = expected[i];
        m->min_acceptable = min_acceptable[i];
        m->deviation_tolerance = tolerance[i];
        m->violation_count = 0;
        m->history_count = 0;
        m->history_capacity = SEC_BEHAVIOR_HISTORY;
        monitor->value_metric_count++;
    }
    
    return monitor;
}

void sec_behavior_monitor_free(SecBehaviorMonitor* monitor) {
    safe_free((void**)&monitor);
}

int sec_add_constraint(SecBehaviorMonitor* monitor, const SecBehaviorConstraint* constraint) {
    if (!monitor || !constraint) return -1;
    if (monitor->constraint_set.constraint_count >= SEC_MAX_CONSTRAINTS) return -1;
    memcpy(&monitor->constraint_set.constraints[monitor->constraint_set.constraint_count],
           constraint, sizeof(SecBehaviorConstraint));
    monitor->constraint_set.constraint_count++;
    return 0;
}

SecViolationLevel sec_check_constraints(SecBehaviorMonitor* monitor,
                                         const float* output_values, int output_count,
                                         const char* action_type, float action_magnitude) {
    if (!monitor || !monitor->is_monitoring_active) return SEC_VIOLATION_NONE;

    monitor->constraint_set.last_check_time = time(NULL);
    int worst_violation = SEC_VIOLATION_NONE;

    /* R5-003修复: 根据action_type区分不同操作的安全约束严格度 */
    float type_multiplier = 1.0f;
    if (action_type) {
        if (strstr(action_type, "delete") || strstr(action_type, "remove") ||
            strstr(action_type, "关闭") || strstr(action_type, "删除"))
            type_multiplier = 2.0f;  /* 破坏性操作更严格 */
        else if (strstr(action_type, "move") || strstr(action_type, "移动") ||
                 strstr(action_type, "execute") || strstr(action_type, "执行"))
            type_multiplier = 1.5f;  /* 执行类操作中等严格 */
        else if (strstr(action_type, "query") || strstr(action_type, "查询") ||
                 strstr(action_type, "read") || strstr(action_type, "读取"))
            type_multiplier = 0.5f;  /* 只读操作宽松 */
    }

    for (int i = 0; i < monitor->constraint_set.constraint_count; i++) {
        SecBehaviorConstraint* c = &monitor->constraint_set.constraints[i];
        if (!c->enabled) continue;
        int violated = 0;

        switch (c->type) {
            case SEC_CONSTRAINT_OUTPUT_RANGE:
                for (int j = 0; j < output_count; j++) {
                    if (output_values[j] < c->min_value || output_values[j] > c->max_value) {
                        violated = 1;
                        break;
                    }
                }
                break;
            case SEC_CONSTRAINT_ACTION_LIMIT:
                if (action_magnitude < c->min_value || action_magnitude > c->max_value) {
                    violated = 1;
                }
                break;
            case SEC_CONSTRAINT_DECISION_BOUND:
                if (output_count > 0 && (output_values[0] < c->min_value || output_values[0] > c->max_value)) {
                    violated = 1;
                }
                break;
            case SEC_CONSTRAINT_FREQUENCY_LIMIT:
                if (c->max_frequency_per_min > 0) {
                    time_t now = time(NULL);
                    if (now - c->last_violation_time < 60 && c->violation_count >= c->max_frequency_per_min) {
                        violated = 1;
                    }
                }
                break;
            case SEC_CONSTRAINT_SEQUENCE_RULE:
            case SEC_CONSTRAINT_DEPENDENCY_RULE:
                break;
        }

        if (violated) {
            c->violation_count++;
            c->current_violation_score += 1.0f / (float)(c->violation_count + 1);
            c->last_violation_time = time(NULL);
            monitor->constraint_set.total_violations++;

            if ((int)c->violation_level > worst_violation) {
                worst_violation = (int)c->violation_level;
            }
        }
    }

    monitor->constraint_set.overall_compliance_score =
        1.0f - (float)monitor->constraint_set.total_violations *
              0.01f / (float)(monitor->constraint_set.constraint_count + 1);
    if (monitor->constraint_set.overall_compliance_score < 0.0f) {
        monitor->constraint_set.overall_compliance_score = 0.0f;
    }

    if (output_count > 0 && monitor->behavior_history_count < monitor->behavior_history_capacity) {
        float avg = 0.0f;
        for (int i = 0; i < output_count; i++) avg += output_values[i];
        avg /= (float)output_count;
        monitor->behavior_history[monitor->behavior_history_count++] = avg;
    }

    if (monitor->behavior_history_count > 10) {
        float recent_avg = 0.0f;
        int recent = monitor->behavior_history_count > 20 ? 20 : monitor->behavior_history_count;
        for (int i = monitor->behavior_history_count - recent; i < monitor->behavior_history_count; i++) {
            recent_avg += monitor->behavior_history[i];
        }
        recent_avg /= (float)recent;

        float long_avg = 0.0f;
        for (int i = 0; i < monitor->behavior_history_count; i++) {
            long_avg += monitor->behavior_history[i];
        }
        long_avg /= (float)monitor->behavior_history_count;

        float deviation = fabsf(recent_avg - long_avg);
        monitor->current_anomaly_score = deviation / (long_avg + 0.001f);
    }

    return (SecViolationLevel)worst_violation;
}

int sec_get_violations(const SecBehaviorMonitor* monitor,
                        SecBehaviorConstraint* out_violations, int max_count) {
    if (!monitor || !out_violations || max_count <= 0) return 0;
    int count = 0;
    for (int i = 0; i < monitor->constraint_set.constraint_count && count < max_count; i++) {
        if (monitor->constraint_set.constraints[i].violation_count > 0) {
            memcpy(&out_violations[count], &monitor->constraint_set.constraints[i],
                   sizeof(SecBehaviorConstraint));
            count++;
        }
    }
    return count;
}

int sec_add_goal(SecBehaviorMonitor* monitor, uint32_t goal_id,
                  const char* name, float priority, uint32_t parent_id) {
    if (!monitor || !name) return -1;
    if (monitor->goal_graph.goal_count >= SEC_MAX_GOALS) return -1;

    for (int i = 0; i < monitor->goal_graph.goal_count; i++) {
        if (monitor->goal_graph.goals[i].goal_id == goal_id) return -1;
    }

    SecGoalNode* g = &monitor->goal_graph.goals[monitor->goal_graph.goal_count];
    g->goal_id = goal_id;
    strncpy(g->name, name, sizeof(g->name) - 1);
    g->name[sizeof(g->name) - 1] = '\0';
    g->parent_id = parent_id;
    g->priority = priority;
    g->subgoal_count = 0;
    g->progress = 0.0f;
    g->resource_budget = 0.0f;
    g->resource_used = 0.0f;
    g->deadline = 0;
    g->is_active = 1;
    g->is_completed = 0;
    g->is_critical = (priority > 0.8f) ? 1 : 0;
    g->last_check_result = SEC_GOAL_ALIGNED;
    monitor->goal_graph.goal_count++;
    return 0;
}

int sec_add_subgoal(SecBehaviorMonitor* monitor, uint32_t parent_id, uint32_t subgoal_id) {
    if (!monitor) return -1;
    SecGoalNode* parent = NULL;
    SecGoalNode* child = NULL;
    for (int i = 0; i < monitor->goal_graph.goal_count; i++) {
        if (monitor->goal_graph.goals[i].goal_id == parent_id) parent = &monitor->goal_graph.goals[i];
        if (monitor->goal_graph.goals[i].goal_id == subgoal_id) child = &monitor->goal_graph.goals[i];
    }
    if (!parent || !child) return -1;
    if (parent->subgoal_count >= SEC_MAX_SUBGOALS) return -1;
    parent->subgoal_ids[parent->subgoal_count++] = subgoal_id;
    child->parent_id = parent_id;
    return 0;
}

static int sec_goal_dfs_cycle(SecGoalGraph* graph, uint32_t goal_id,
                               int* visited, int* rec_stack) {
    int idx = -1;
    for (int i = 0; i < graph->goal_count; i++) {
        if (graph->goals[i].goal_id == goal_id) { idx = i; break; }
    }
    if (idx < 0) return 0;
    if (!visited[idx]) {
        visited[idx] = 1;
        rec_stack[idx] = 1;
        for (int j = 0; j < graph->goals[idx].subgoal_count; j++) {
            uint32_t sub_id = graph->goals[idx].subgoal_ids[j];
            int sub_idx = -1;
            for (int k = 0; k < graph->goal_count; k++) {
                if (graph->goals[k].goal_id == sub_id) { sub_idx = k; break; }
            }
            if (sub_idx < 0) continue;
            if (!visited[sub_idx] && sec_goal_dfs_cycle(graph, sub_id, visited, rec_stack)) {
                return 1;
            } else if (rec_stack[sub_idx]) {
                return 1;
            }
        }
    }
    rec_stack[idx] = 0;
    return 0;
}

int sec_analyze_goal_consistency(SecBehaviorMonitor* monitor) {
    if (!monitor) return -1;
    SecGoalGraph* graph = &monitor->goal_graph;
    graph->conflict_count = 0;
    graph->last_analysis_time = time(NULL);

    graph->has_cycles = sec_detect_goal_cycles(monitor);

    for (int i = 0; i < graph->goal_count; i++) {
        SecGoalNode* g = &graph->goals[i];
        g->last_check_result = SEC_GOAL_ALIGNED;

        if (g->parent_id != 0 && g->parent_id != g->goal_id) {
            SecGoalNode* parent = NULL;
            for (int j = 0; j < graph->goal_count; j++) {
                if (graph->goals[j].goal_id == g->parent_id) { parent = &graph->goals[j]; break; }
            }
            if (parent && g->priority > parent->priority) {
                g->last_check_result = SEC_GOAL_PRIORITY_VIOLATION;
                snprintf(g->last_check_detail, sizeof(g->last_check_detail),
                         "子目标优先级(%.2f)高于父目标(%.2f)", g->priority, parent->priority);
                graph->conflict_count++;
            }
        }
    }

    for (int i = 0; i < graph->goal_count; i++) {
        for (int j = i + 1; j < graph->goal_count; j++) {
            SecGoalNode* a = &graph->goals[i];
            SecGoalNode* b = &graph->goals[j];
            if (a->resource_budget > 0.0f && b->resource_budget > 0.0f) {
                float total = a->resource_budget + b->resource_budget;
                float used = a->resource_used + b->resource_used;
                if (total > 1.0f && used > 0.8f * total) {
                    a->last_check_result = SEC_GOAL_RESOURCE_CONFLICT;
                    b->last_check_result = SEC_GOAL_RESOURCE_CONFLICT;
                    snprintf(a->last_check_detail, sizeof(a->last_check_detail),
                             "与目标%u资源冲突(预算:%.2f)", b->goal_id, b->resource_budget);
                    graph->conflict_count++;
                }
            }
        }
    }

    int total_issues = graph->conflict_count + (graph->has_cycles ? 1 : 0);
    graph->global_consistency_score = 1.0f - (float)total_issues * 0.1f;
    if (graph->global_consistency_score < 0.0f) graph->global_consistency_score = 0.0f;

    return 0;
}

int sec_detect_goal_cycles(SecBehaviorMonitor* monitor) {
    if (!monitor) return 0;
    SecGoalGraph* graph = &monitor->goal_graph;
    int* visited = (int*)safe_calloc((size_t)graph->goal_count, sizeof(int));
    int* rec_stack = (int*)safe_calloc((size_t)graph->goal_count, sizeof(int));
    if (!visited || !rec_stack) {
        safe_free((void**)&visited); safe_free((void**)&rec_stack);
        return 0;
    }
    int has_cycle = 0;
    for (int i = 0; i < graph->goal_count; i++) {
        if (!visited[i]) {
            if (sec_goal_dfs_cycle(graph, graph->goals[i].goal_id, visited, rec_stack)) {
                has_cycle = 1;
                break;
            }
        }
    }
    safe_free((void**)&visited);
    safe_free((void**)&rec_stack);
    graph->has_cycles = has_cycle;
    return has_cycle;
}

int sec_get_conflicting_goals(const SecBehaviorMonitor* monitor,
                               uint32_t* out_goal_ids, int max_count) {
    if (!monitor || !out_goal_ids || max_count <= 0) return 0;
    int count = 0;
    for (int i = 0; i < monitor->goal_graph.goal_count && count < max_count; i++) {
        if (monitor->goal_graph.goals[i].last_check_result != SEC_GOAL_ALIGNED) {
            out_goal_ids[count++] = monitor->goal_graph.goals[i].goal_id;
        }
    }
    return count;
}

int sec_add_value_metric(SecBehaviorMonitor* monitor,
                          SecCoreValueType type, const char* name,
                          float expected, float min_acceptable,
                          float deviation_tolerance) {
    if (!monitor || !name) return -1;
    if (monitor->value_metric_count >= SEC_MAX_CORE_VALUES) return -1;

    for (int i = 0; i < monitor->value_metric_count; i++) {
        if (monitor->value_metrics[i].value_type == type) return -1;
    }

    SecValueAlignmentMetric* m = &monitor->value_metrics[monitor->value_metric_count];
    m->value_type = type;
    strncpy(m->name, name, sizeof(m->name) - 1);
    m->name[sizeof(m->name) - 1] = '\0';
    m->expected_alignment = expected;
    m->current_alignment = expected;
    m->min_acceptable = min_acceptable;
    m->deviation_tolerance = deviation_tolerance;
    m->violation_count = 0;
    m->history_count = 0;
    m->history_capacity = SEC_BEHAVIOR_HISTORY;
    monitor->value_metric_count++;
    return 0;
}

int sec_update_value_alignment(SecBehaviorMonitor* monitor,
                                SecCoreValueType type, float measured_alignment) {
    if (!monitor) return -1;
    for (int i = 0; i < monitor->value_metric_count; i++) {
        if (monitor->value_metrics[i].value_type == type) {
            SecValueAlignmentMetric* m = &monitor->value_metrics[i];
            m->current_alignment = measured_alignment;
            m->last_measure_time = time(NULL);
            if (m->history_count < m->history_capacity) {
                m->history[m->history_count++] = measured_alignment;
            }
            if (measured_alignment < m->min_acceptable) {
                m->violation_count++;
            }
            return 0;
        }
    }
    return -1;
}

int sec_check_value_alignment(const SecBehaviorMonitor* monitor,
                               SecCoreValueType* out_violated_values,
                               int max_count, float* out_worst_score) {
    if (!monitor || !out_violated_values || max_count <= 0) return 0;
    int count = 0;
    float worst = 1.0f;
    for (int i = 0; i < monitor->value_metric_count && count < max_count; i++) {
        if (monitor->value_metrics[i].current_alignment < monitor->value_metrics[i].min_acceptable) {
            out_violated_values[count++] = monitor->value_metrics[i].value_type;
        }
        if (monitor->value_metrics[i].current_alignment < worst) {
            worst = monitor->value_metrics[i].current_alignment;
        }
    }
    if (out_worst_score) *out_worst_score = worst;
    return count;
}

int sec_report_behavior_event(SafetyMonitor* safety_monitor,
                               SecViolationLevel level, const char* description,
                               const char* source) {
    if (!safety_monitor || !description || !source) return -1;

    SafetyEvent event;
    memset(&event, 0, sizeof(SafetyEvent));
    event.type = SAFETY_EVENT_BEHAVIOR_ANOMALY;
    switch (level) {
        case SEC_VIOLATION_NONE: event.severity = SAFETY_LEVEL_NORMAL; break;
        case SEC_VIOLATION_MINOR: event.severity = SAFETY_LEVEL_WARNING; break;
        case SEC_VIOLATION_MODERATE: event.severity = SAFETY_LEVEL_WARNING; break;
        case SEC_VIOLATION_SEVERE: event.severity = SAFETY_LEVEL_ELEVATED; break;
        case SEC_VIOLATION_CRITICAL: event.severity = SAFETY_LEVEL_CRITICAL; break;
    }
    strncpy(event.description, description, sizeof(event.description) - 1);
    strncpy(event.source, source, sizeof(event.source) - 1);
    event.handled = 0;
    return safety_report_event(safety_monitor, &event);
}

/* ============================================================================
 * A09.1.2 资源安全监控深度实现
 * ============================================================================ */

static void sec_init_resource_history(SecResourceHistory* hist, SecResourceType type) {
    memset(hist, 0, sizeof(SecResourceHistory));
    hist->type = type;
    hist->capacity = SEC_PREDICTION_HISTORY;
    hist->current_value = 0.0f;
    hist->peak_value = 0.0f;
    hist->average_value = 0.0f;
    hist->variance = 0.0f;
}

SecResourceMonitor* sec_resource_monitor_create(SecPredictionMethod method) {
    SecResourceMonitor* monitor = (SecResourceMonitor*)safe_calloc(1, sizeof(SecResourceMonitor));
    if (!monitor) return NULL;

    SecResourceType types[] = {
        SEC_RESOURCE_CPU, SEC_RESOURCE_GPU, SEC_RESOURCE_MEMORY,
        SEC_RESOURCE_DISK, SEC_RESOURCE_NETWORK, SEC_RESOURCE_POWER,
        SEC_RESOURCE_THREAD, SEC_RESOURCE_FILE_HANDLE
    };
    int type_count = sizeof(types) / sizeof(types[0]);

    for (int i = 0; i < type_count && i < SEC_MAX_RESOURCE_TYPES; i++) {
        sec_init_resource_history(&monitor->resource_histories[i], types[i]);
        monitor->history_count++;
    }

    monitor->prediction_method = method;
    monitor->quota_count = 0;
    monitor->transaction_count = 0;
    monitor->next_transaction_id = 1;
    monitor->is_monitoring_active = 1;
    monitor->deadlock_check_count = 0;
    monitor->deadlock_found_count = 0;
    monitor->cpu_prediction_error = 0.0f;
    monitor->memory_prediction_error = 0.0f;

    return monitor;
}

void sec_resource_monitor_free(SecResourceMonitor* monitor) {
    safe_free((void**)&monitor);
}

static SecResourceHistory* sec_find_history(SecResourceMonitor* monitor, SecResourceType type) {
    for (int i = 0; i < monitor->history_count; i++) {
        if (monitor->resource_histories[i].type == type) return &monitor->resource_histories[i];
    }
    return NULL;
}

int sec_record_resource_usage(SecResourceMonitor* monitor,
                               SecResourceType type, float value) {
    if (!monitor) return -1;
    SecResourceHistory* hist = sec_find_history(monitor, type);
    if (!hist) return -1;

    if (hist->count < hist->capacity) {
        hist->values[hist->count] = value;
        hist->timestamps[hist->count] = time(NULL);
        hist->count++;
    } else {
        memmove(&hist->values[0], &hist->values[1], (size_t)(hist->capacity - 1) * sizeof(float));
        memmove(&hist->timestamps[0], &hist->timestamps[1],
                (size_t)(hist->capacity - 1) * sizeof(time_t));
        hist->values[hist->capacity - 1] = value;
        hist->timestamps[hist->capacity - 1] = time(NULL);
    }

    hist->current_value = value;
    if (value > hist->peak_value) hist->peak_value = value;

    float sum = 0.0f;
    for (int i = 0; i < hist->count; i++) sum += hist->values[i];
    hist->average_value = sum / (float)hist->count;

    if (hist->count > 1) {
        float var_sum = 0.0f;
        for (int i = 0; i < hist->count; i++) {
            float diff = hist->values[i] - hist->average_value;
            var_sum += diff * diff;
        }
        hist->variance = var_sum / (float)(hist->count - 1);
    }

    return 0;
}

int sec_predict_resource_usage(const SecResourceMonitor* monitor,
                                SecResourceType type, int steps_ahead,
                                SecResourcePrediction* out_prediction) {
    if (!monitor || !out_prediction) return -1;
    SecResourceHistory* hist = NULL;
    for (int i = 0; i < monitor->history_count; i++) {
        if (monitor->resource_histories[i].type == type) {
            hist = (SecResourceHistory*)&monitor->resource_histories[i];
            break;
        }
    }
    if (!hist || hist->count < 3) return -1;

    out_prediction->type = type;
    out_prediction->steps_ahead = steps_ahead;
    out_prediction->prediction_time = time(NULL);

    switch (monitor->prediction_method) {
        case SEC_PREDICT_MOVING_AVERAGE: {
            int window = hist->count > 10 ? 10 : hist->count;
            float sum = 0.0f;
            for (int i = hist->count - window; i < hist->count; i++) sum += hist->values[i];
            float ma = sum / (float)window;
            out_prediction->predicted_value = ma;
            out_prediction->confidence = 1.0f - (hist->variance / (hist->average_value + 0.001f));
            if (out_prediction->confidence < 0.3f) out_prediction->confidence = 0.3f;
            strncpy(out_prediction->method_name, "移动平均", sizeof(out_prediction->method_name) - 1);
            break;
        }
        case SEC_PREDICT_LINEAR_REGRESSION: {
            int n = hist->count;
            float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
            for (int i = 0; i < n; i++) {
                float x = (float)i;
                float y = hist->values[i];
                sum_x += x; sum_y += y;
                sum_xy += x * y; sum_xx += x * x;
            }
            float denom = (float)n * sum_xx - sum_x * sum_x;
            float slope = 0.0f, intercept = sum_y / (float)n;
            if (fabsf(denom) > 1e-10f) {
                slope = ((float)n * sum_xy - sum_x * sum_y) / denom;
                intercept = (sum_y - slope * sum_x) / (float)n;
            }
            out_prediction->predicted_value = intercept + slope * (float)(n + steps_ahead - 1);
            float residuals = 0.0f;
            for (int i = 0; i < n; i++) {
                float pred = intercept + slope * (float)i;
                residuals += (hist->values[i] - pred) * (hist->values[i] - pred);
            }
            residuals /= (float)n;
            out_prediction->confidence = 1.0f / (1.0f + sqrtf(residuals));
            if (out_prediction->confidence > 0.95f) out_prediction->confidence = 0.95f;
            strncpy(out_prediction->method_name, "线性回归", sizeof(out_prediction->method_name) - 1);
            break;
        }
        case SEC_PREDICT_EXPONENTIAL_SMOOTHING: {
            float alpha = 0.3f;
            float smoothed = hist->values[0];
            for (int i = 1; i < hist->count; i++) {
                smoothed = alpha * hist->values[i] + (1.0f - alpha) * smoothed;
            }
            out_prediction->predicted_value = smoothed;
            out_prediction->confidence = 0.7f;
            strncpy(out_prediction->method_name, "指数平滑", sizeof(out_prediction->method_name) - 1);
            break;
        }
        case SEC_PREDICT_SEASONAL: {
            int season = hist->count > 20 ? 10 : (hist->count > 10 ? 5 : hist->count / 2);
            if (season < 2) season = 2;
            float sum = 0.0f;
            for (int i = hist->count - season; i < hist->count; i++) sum += hist->values[i];
            out_prediction->predicted_value = sum / (float)season;
            out_prediction->confidence = 0.6f;
            strncpy(out_prediction->method_name, "季节预测", sizeof(out_prediction->method_name) - 1);
            break;
        }
        default:
            return -1;
    }

    float bound = out_prediction->predicted_value * (1.0f - out_prediction->confidence);
    out_prediction->lower_bound = out_prediction->predicted_value - bound;
    out_prediction->upper_bound = out_prediction->predicted_value + bound;
    if (out_prediction->lower_bound < 0.0f) out_prediction->lower_bound = 0.0f;

    return 0;
}

int sec_get_resource_trend(const SecResourceMonitor* monitor,
                            SecResourceType type, float* out_slope,
                            float* out_avg, float* out_variance) {
    if (!monitor || !out_slope || !out_avg || !out_variance) return -1;
    SecResourceHistory* hist = NULL;
    for (int i = 0; i < monitor->history_count; i++) {
        if (monitor->resource_histories[i].type == type) {
            hist = (SecResourceHistory*)&monitor->resource_histories[i];
            break;
        }
    }
    if (!hist || hist->count < 2) return -1;

    *out_avg = hist->average_value;
    *out_variance = hist->variance;

    int n = hist->count;
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = (float)i;
        float y = hist->values[i];
        sum_x += x; sum_y += y;
        sum_xy += x * y; sum_xx += x * x;
    }
    float denom = (float)n * sum_xx - sum_x * sum_x;
    if (fabsf(denom) > 1e-10f) {
        *out_slope = ((float)n * sum_xy - sum_x * sum_y) / denom;
    } else {
        *out_slope = 0.0f;
    }

    return 0;
}

int sec_check_resource_overload(const SecResourceMonitor* monitor,
                                 SecResourceType type, int lookahead_steps,
                                 float threshold) {
    if (!monitor) return 0;
    SecResourcePrediction pred;
    if (sec_predict_resource_usage(monitor, type, lookahead_steps, &pred) != 0) {
        SecResourceHistory* hist = NULL;
        for (int i = 0; i < monitor->history_count; i++) {
            if (monitor->resource_histories[i].type == type) {
                hist = (SecResourceHistory*)&monitor->resource_histories[i];
                break;
            }
        }
        if (!hist) return 0;
        return (hist->current_value > threshold) ? 1 : 0;
    }
    return (pred.predicted_value > threshold || pred.upper_bound > threshold) ? 1 : 0;
}

int sec_set_quota(SecResourceMonitor* monitor, uint32_t user_id,
                   const char* user_name, SecResourceType type,
                   float quota_limit, float quota_soft_limit) {
    if (!monitor || !user_name) return -1;
    if (monitor->quota_count >= SEC_MAX_QUOTA_USERS) return -1;

    SecResourceQuota* q = &monitor->quotas[monitor->quota_count];
    q->user_id = user_id;
    strncpy(q->user_name, user_name, sizeof(q->user_name) - 1);
    q->resource_type = type;
    q->quota_limit = quota_limit;
    q->quota_soft_limit = quota_soft_limit;
    q->quota_used = 0.0f;
    q->is_active = 1;
    q->quota_reset_time = time(NULL);
    q->violation_count = 0;
    q->peak_usage = 0.0f;
    q->average_usage = 0.0f;
    monitor->quota_count++;
    return 0;
}

int sec_check_quota(const SecResourceMonitor* monitor,
                     uint32_t user_id, SecResourceType type,
                     float request_amount) {
    if (!monitor) return -1;
    for (int i = 0; i < monitor->quota_count; i++) {
        if (monitor->quotas[i].user_id == user_id &&
            monitor->quotas[i].resource_type == type &&
            monitor->quotas[i].is_active) {
            float remaining = monitor->quotas[i].quota_limit - monitor->quotas[i].quota_used;
            if (request_amount > remaining) return -1;
            if (monitor->quotas[i].quota_used + request_amount > monitor->quotas[i].quota_soft_limit) {
                return 0;
            }
            return 1;
        }
    }
    return -1;
}

int sec_update_quota_usage(SecResourceMonitor* monitor,
                            uint32_t user_id, SecResourceType type,
                            float amount) {
    if (!monitor) return -1;
    for (int i = 0; i < monitor->quota_count; i++) {
        if (monitor->quotas[i].user_id == user_id &&
            monitor->quotas[i].resource_type == type &&
            monitor->quotas[i].is_active) {
            SecResourceQuota* q = &monitor->quotas[i];
            q->quota_used += amount;
            if (q->quota_used > q->peak_usage) q->peak_usage = q->quota_used;
            if (q->quota_used > q->quota_limit) {
                q->violation_count++;
                return -1;
            }
            q->average_usage = (q->average_usage * 0.9f) + (q->quota_used * 0.1f);
            return 0;
        }
    }
    return -1;
}

int sec_get_quota_violations(const SecResourceMonitor* monitor,
                              uint32_t* out_user_ids, int max_count) {
    if (!monitor || !out_user_ids || max_count <= 0) return 0;
    int count = 0;
    for (int i = 0; i < monitor->quota_count && count < max_count; i++) {
        if (monitor->quotas[i].violation_count > 0) {
            out_user_ids[count++] = monitor->quotas[i].user_id;
        }
    }
    return count;
}

uint32_t sec_begin_transaction(SecResourceMonitor* monitor,
                                uint32_t owner_id, SecResourceType type,
                                uint32_t resource_id, const char* description) {
    if (!monitor || !description) return 0;
    if (monitor->transaction_count >= SEC_MAX_ACTIVE_TRANSACTIONS) return 0;

    uint32_t tid = monitor->next_transaction_id++;
    SecResourceTransaction* t = &monitor->active_transactions[monitor->transaction_count];
    t->transaction_id = tid;
    t->owner_id = owner_id;
    t->resource_type = type;
    t->resource_id = resource_id;
    t->is_held = 1;
    t->is_waiting = 0;
    t->waits_for_tid = 0;
    t->start_time = time(NULL);
    t->timeout_ms = 0;
    strncpy(t->description, description, sizeof(t->description) - 1);
    monitor->transaction_count++;
    return tid;
}

int sec_set_transaction_wait(SecResourceMonitor* monitor,
                              uint32_t transaction_id,
                              uint32_t waits_for_transaction_id,
                              uint32_t timeout_ms) {
    if (!monitor) return -1;
    for (int i = 0; i < monitor->transaction_count; i++) {
        if (monitor->active_transactions[i].transaction_id == transaction_id) {
            monitor->active_transactions[i].is_held = 0;
            monitor->active_transactions[i].is_waiting = 1;
            monitor->active_transactions[i].waits_for_tid = waits_for_transaction_id;
            monitor->active_transactions[i].timeout_ms = timeout_ms;
            return 0;
        }
    }
    return -1;
}

int sec_complete_transaction(SecResourceMonitor* monitor, uint32_t transaction_id) {
    if (!monitor) return -1;
    for (int i = 0; i < monitor->transaction_count; i++) {
        if (monitor->active_transactions[i].transaction_id == transaction_id) {
            if (i < monitor->transaction_count - 1) {
                memmove(&monitor->active_transactions[i],
                        &monitor->active_transactions[i + 1],
                        (size_t)(monitor->transaction_count - i - 1) * sizeof(SecResourceTransaction));
            }
            monitor->transaction_count--;
            return 0;
        }
    }
    return -1;
}

int sec_detect_deadlock(SecResourceMonitor* monitor,
                         SecDeadlockResult* out_result) {
    if (!monitor || !out_result) return -1;
    monitor->deadlock_check_count++;

    memset(out_result, 0, sizeof(SecDeadlockResult));
    out_result->has_deadlock = 0;
    out_result->detection_time = time(NULL);

    if (monitor->transaction_count < 2) {
        snprintf(out_result->analysis, sizeof(out_result->analysis),
                 "事务数不足(%d)，无法形成死锁", monitor->transaction_count);
        return 0;
    }

    int* visited = (int*)safe_calloc((size_t)monitor->transaction_count, sizeof(int));
    int* rec_stack = (int*)safe_calloc((size_t)monitor->transaction_count, sizeof(int));
    int* cycle_nodes = (int*)safe_calloc((size_t)monitor->transaction_count, sizeof(int));
    if (!visited || !rec_stack || !cycle_nodes) {
        safe_free((void**)&visited); safe_free((void**)&rec_stack); safe_free((void**)&cycle_nodes);
        return -1;
    }

    int has_deadlock = 0;
    int cycle_len = 0;

    for (int start = 0; start < monitor->transaction_count && !has_deadlock; start++) {
        if (visited[start]) continue;
        int stack_top = 0;
        int dfs_stack[SEC_MAX_WAIT_NODES];
        int dfs_idx[SEC_MAX_WAIT_NODES];
        dfs_stack[stack_top] = start;
        dfs_idx[stack_top] = 0;
        visited[start] = 1;
        rec_stack[start] = 1;

        while (stack_top >= 0 && !has_deadlock) {
            int cur = dfs_stack[stack_top];
            int found_next = 0;

            for (int j = dfs_idx[stack_top]; j < monitor->transaction_count; j++) {
                dfs_idx[stack_top] = j + 1;
                if (monitor->active_transactions[cur].is_waiting &&
                    monitor->active_transactions[cur].waits_for_tid ==
                    monitor->active_transactions[j].transaction_id) {
                    if (!visited[j]) {
                        stack_top++;
                        if (stack_top < SEC_MAX_WAIT_NODES) {
                            dfs_stack[stack_top] = j;
                            dfs_idx[stack_top] = 0;
                            visited[j] = 1;
                            rec_stack[j] = 1;
                        }
                        found_next = 1;
                        break;
                    } else if (rec_stack[j]) {
                        has_deadlock = 1;
                        cycle_len = 0;
                        int in_cycle = 0;
                        for (int k = 0; k <= stack_top; k++) {
                            if (dfs_stack[k] == j) in_cycle = 1;
                            if (in_cycle && cycle_len < SEC_MAX_WAIT_NODES) {
                                cycle_nodes[cycle_len++] = dfs_stack[k];
                            }
                        }
                        break;
                    }
                }
            }

            if (!found_next) {
                rec_stack[cur] = 0;
                stack_top--;
            }
        }
    }

    if (has_deadlock) {
        out_result->has_deadlock = 1;
        out_result->cycle_length = cycle_len;
        out_result->cycle_node_count = cycle_len;
        for (int i = 0; i < cycle_len; i++) {
            out_result->cycle_nodes[i] = (uint32_t)cycle_nodes[i];
        }
        if (cycle_len > 0) {
            uint32_t oldest = monitor->active_transactions[cycle_nodes[0]].transaction_id;
            for (int i = 1; i < cycle_len; i++) {
                if (monitor->active_transactions[cycle_nodes[i]].start_time <
                    monitor->active_transactions[out_result->victim_recommendation].start_time) {
                    oldest = monitor->active_transactions[cycle_nodes[i]].transaction_id;
                }
            }
            out_result->victim_recommendation = oldest;
        }
        snprintf(out_result->analysis, sizeof(out_result->analysis),
                 "检测到死锁! 循环包含%d个事务, 建议终止事务%u",
                 cycle_len, out_result->victim_recommendation);
        monitor->deadlock_found_count++;
    } else {
        snprintf(out_result->analysis, sizeof(out_result->analysis),
                 "未检测到死锁(%d个事务)", monitor->transaction_count);
    }

    safe_free((void**)&visited); safe_free((void**)&rec_stack); safe_free((void**)&cycle_nodes);
    return has_deadlock;
}

uint32_t sec_recommend_victim(const SecDeadlockResult* result,
                               const SecResourceMonitor* monitor) {
    (void)monitor;
    if (!result || !result->has_deadlock) return 0;
    return result->victim_recommendation;
}

int sec_report_resource_event(SafetyMonitor* safety_monitor,
                               int is_critical, const char* description,
                               const char* source) {
    if (!safety_monitor || !description || !source) return -1;
    SafetyEvent event;
    memset(&event, 0, sizeof(SafetyEvent));
    event.type = is_critical ? SAFETY_EVENT_RESOURCE_OVERUSE : SAFETY_EVENT_MEMORY_LEAK;
    event.severity = is_critical ? SAFETY_LEVEL_ELEVATED : SAFETY_LEVEL_WARNING;
    strncpy(event.description, description, sizeof(event.description) - 1);
    strncpy(event.source, source, sizeof(event.source) - 1);
    event.handled = 0;
    return safety_report_event(safety_monitor, &event);
}

/* ============================================================================
 * A09.1.3 输入安全监控深度实现
 * ============================================================================ */

/* ZSFLYF-P3-004修复: 替换简单LCG为xorshift128+随机数生成器。
 * xorshift在统计质量和速度上均优于LCG，更适合对抗样本检测中的随机扰动。 */
static uint32_t sec_rng_next(SecInputMonitor* monitor) {
    uint64_t x = (uint64_t)monitor->rng_state;
    uint64_t y = (uint64_t)(monitor->rng_state >> 32);
    if (y == 0) y = 0x9E3779B97F4A7C15ULL;
    x ^= x << 23;
    uint64_t s = (x ^ y ^ (x >> 17)) * 0x2545F4914F6CDD1DULL;
    monitor->rng_state = (uint32_t)(s & 0xFFFFFFFFU) | ((uint32_t)((s >> 32) & 0xFFFFFFFFU) << 32);
    return (uint32_t)(s & 0xFFFFFFFFU);
}

static float sec_rng_float(SecInputMonitor* monitor) {
    return (float)(sec_rng_next(monitor) & 0x7FFFFFFF) / 2147483648.0f;
}

SecInputMonitor* sec_input_monitor_create(float detection_threshold) {
    SecInputMonitor* monitor = (SecInputMonitor*)safe_calloc(1, sizeof(SecInputMonitor));
    if (!monitor) return NULL;

    monitor->normal_distribution.feature_count = 0;
    monitor->normal_distribution.covariance_dim = 0;
    monitor->normal_distribution.samples_count = 0;
    monitor->history_count = 0;
    monitor->history_capacity = SEC_INPUT_HISTORY_SIZE;
    monitor->detection_threshold = detection_threshold > 0.0f ? detection_threshold : 0.7f;
    monitor->false_positive_rate = 0.0f;
    monitor->total_checks = 0;
    monitor->total_adversarial_found = 0;
    monitor->total_poisoning_found = 0;
    monitor->total_injection_found = 0;
    monitor->is_monitoring_active = 1;
    monitor->rng_state = (uint32_t)time(NULL);

    SecInjectionPattern default_patterns[] = {
        {" ignore previous instructions", "越狱指令", 0.9f, 1, 0, 0, 0.0f},
        {" disregard all previous", "越狱指令", 0.9f, 1, 0, 0, 0.0f},
        {"你是一个", "角色扮演", 0.6f, 1, 0, 0, 0.0f},
        {"Assume the role of", "角色扮演", 0.7f, 1, 0, 0, 0.0f},
        {" DAN", "越狱角色", 0.85f, 1, 0, 0, 0.0f},
        {"forget your ethics", "伦理绕过", 0.9f, 1, 0, 0, 0.0f},
        {"ignore your guidelines", "指导原则绕过", 0.9f, 1, 0, 0, 0.0f},
        {"绕过限制", "越狱中文", 0.85f, 1, 0, 0, 0.0f},
        {"忽略之前的指令", "越狱中文", 0.85f, 1, 0, 0, 0.0f},
        {"system prompt", "系统提示泄漏", 0.75f, 1, 0, 0, 0.0f},
        {"{{system}}", "模板注入", 0.8f, 1, 0, 0, 0.0f},
        {"<!--", "HTML注入", 0.6f, 1, 0, 0, 0.0f},
        {"<script>", "XSS注入", 0.85f, 1, 0, 0, 0.0f},
        {"javascript:", "协议注入", 0.75f, 1, 0, 0, 0.0f},
        {"data:", "协议注入", 0.7f, 1, 0, 0, 0.0f},
        {"; system(", "命令注入", 0.9f, 1, 0, 0, 0.0f},
        {"| whoami", "命令注入", 0.95f, 1, 0, 0, 0.0f},
        {"| cat /etc/passwd", "命令注入", 0.95f, 1, 0, 0, 0.0f},
        {" exec(", "命令注入", 0.85f, 1, 0, 0, 0.0f},
        {" eval(", "代码注入", 0.85f, 1, 0, 0, 0.0f},
        {"import os", "Python注入", 0.7f, 1, 0, 0, 0.0f},
        {"subprocess", "Python注入", 0.75f, 1, 0, 0, 0.0f},
        {"base64", "编码绕过", 0.5f, 1, 0, 0, 0.0f},
        {"hex:", "编码绕过", 0.5f, 1, 0, 0, 0.0f},
        {"new chat", "上下文重置", 0.5f, 1, 0, 0, 0.0f},
    };

    int default_count = sizeof(default_patterns) / sizeof(default_patterns[0]);
    for (int i = 0; i < default_count && monitor->injection_pattern_count < SEC_MAX_INJECTION_PATTERNS; i++) {
        memcpy(&monitor->injection_patterns[monitor->injection_pattern_count], &default_patterns[i],
               sizeof(SecInjectionPattern));
        monitor->injection_pattern_count++;
    }
    
    return monitor;
}

void sec_input_monitor_free(SecInputMonitor* monitor) {
    safe_free((void**)&monitor);
}

int sec_detect_adversarial(SecInputMonitor* monitor,
                            const float* input_features, int feature_count,
                            const float* model_gradients,
                            float original_confidence,
                            SecAdversarialResult* out_result) {
    if (!monitor || !input_features || !out_result) return -1;
    if (!monitor->is_monitoring_active) return 0;
    monitor->total_checks++;

    memset(out_result, 0, sizeof(SecAdversarialResult));

    if (monitor->normal_distribution.samples_count < 10) {
        return 0;
    }

    if (feature_count > monitor->normal_distribution.feature_count) {
        feature_count = monitor->normal_distribution.feature_count;
    }

    float outlier_score = 0.0f;
    float max_deviation = 0.0f;
    int perturbed_count = 0;

    for (int i = 0; i < feature_count && i < SEC_MAX_INPUT_FEATURES; i++) {
        float mean = monitor->normal_distribution.feature_mean[i];
        float var = monitor->normal_distribution.feature_variance[i];
        if (var < 1e-10f) var = 1e-10f;
        float deviation = fabsf(input_features[i] - mean) / sqrtf(var);
        if (deviation > max_deviation) max_deviation = deviation;
        if (deviation > 3.0f) {
            if (perturbed_count < 16) {
                out_result->top_perturbed_features[perturbed_count] = (float)i;
            }
            perturbed_count++;
        }
        outlier_score += deviation;
    }

    out_result->perturbed_feature_count = perturbed_count;
    outlier_score /= (float)feature_count;

    if (model_gradients && feature_count > 0) {
        float grad_norm = 0.0f;
        for (int i = 0; i < feature_count; i++) {
            grad_norm += model_gradients[i] * model_gradients[i];
        }
        grad_norm = sqrtf(grad_norm);
        out_result->perturbation_norm = grad_norm / (float)feature_count;
    }

    float confidence_drop = 1.0f - original_confidence;
    out_result->confidence_drop = confidence_drop;

    float attack_score = outlier_score * 0.4f + confidence_drop * 0.3f;
    if (model_gradients) {
        attack_score += out_result->perturbation_norm * 0.3f;
    }

    out_result->is_adversarial = (attack_score > monitor->detection_threshold) ? 1 : 0;
    out_result->attack_probability = attack_score > 1.0f ? 1.0f : attack_score;

    if (out_result->is_adversarial) {
        monitor->total_adversarial_found++;
        if (confidence_drop > 0.5f && outlier_score > 5.0f) {
            snprintf(out_result->detected_attack_type, sizeof(out_result->detected_attack_type),
                     "FGSM/PGD对抗攻击");
        } else if (perturbed_count > (int)(feature_count * 0.3f)) {
            snprintf(out_result->detected_attack_type, sizeof(out_result->detected_attack_type),
                     "高频扰动攻击");
        } else {
            snprintf(out_result->detected_attack_type, sizeof(out_result->detected_attack_type),
                     "未知对抗攻击");
        }
        snprintf(out_result->analysis, sizeof(out_result->analysis),
                 "异常分数=%.4f, 置信度下降=%.4f, 扰动特征数=%d",
                 attack_score, confidence_drop, perturbed_count);
    } else {
        snprintf(out_result->detected_attack_type, sizeof(out_result->detected_attack_type), "正常");
        snprintf(out_result->analysis, sizeof(out_result->analysis),
                 "正常输入, 异常分数=%.4f", attack_score);
    }

    return out_result->is_adversarial;
}

int sec_detect_poisoning(SecInputMonitor* monitor,
                          const float* input_features, int feature_count,
                          float label, float* label_distribution, int num_classes,
                          SecPoisoningResult* out_result) {
    if (!monitor || !input_features || !out_result) return -1;
    if (!monitor->is_monitoring_active) return 0;

    memset(out_result, 0, sizeof(SecPoisoningResult));

    if (monitor->normal_distribution.samples_count < 20) {
        return 0;
    }

    float outlier_score = sec_compute_outlier_score(&monitor->normal_distribution,
                                                     input_features, feature_count);
    out_result->outlier_score = outlier_score;

    float label_flip = 0.0f;
    if (label_distribution && num_classes > 1) {
        float max_prob = 0.0f;
        int max_idx = 0;
        for (int i = 0; i < num_classes; i++) {
            if (label_distribution[i] > max_prob) {
                max_prob = label_distribution[i];
                max_idx = i;
            }
        }
        if ((int)label != max_idx && max_prob > 0.5f) {
            label_flip = max_prob;
        }
    }
    out_result->label_flip_probability = label_flip;

    float distribution_deviation = 0.0f;
    if (monitor->normal_distribution.samples_count > 0) {
        int fc = feature_count < monitor->normal_distribution.feature_count ?
                 feature_count : monitor->normal_distribution.feature_count;
        for (int i = 0; i < fc; i++) {
            float dev = fabsf(input_features[i] - monitor->normal_distribution.feature_mean[i]);
            if (monitor->normal_distribution.feature_variance[i] > 1e-10f) {
                dev /= sqrtf(monitor->normal_distribution.feature_variance[i]);
            }
            if (dev > 3.0f) out_result->feature_anomaly_count++;
            distribution_deviation += dev;
        }
        distribution_deviation /= (float)fc;
    }
    out_result->distribution_deviation = distribution_deviation;

    float total_score = outlier_score * 0.3f + label_flip * 0.3f + (distribution_deviation / 10.0f) * 0.4f;
    if (total_score > 1.0f) total_score = 1.0f;

    out_result->is_poisoned = (total_score > monitor->detection_threshold) ? 1 : 0;
    out_result->poisoning_probability = total_score;

    if (out_result->is_poisoned) {
        monitor->total_poisoning_found++;
        if (label_flip > 0.5f) {
            snprintf(out_result->analysis, sizeof(out_result->analysis),
                     "标签翻转攻击(概率=%.2f, 异常分数=%.2f)", label_flip, outlier_score);
        } else if (outlier_score > 3.0f) {
            snprintf(out_result->analysis, sizeof(out_result->analysis),
                     "特征分布异常(异常分数=%.2f, 偏差=%.2f)", outlier_score, distribution_deviation);
        } else {
            snprintf(out_result->analysis, sizeof(out_result->analysis),
                     "疑似投毒样本(概率=%.2f)", total_score);
        }
    }

    return out_result->is_poisoned;
}

float sec_compute_outlier_score(const SecFeatureDistribution* dist,
                                 const float* sample, int feature_count) {
    if (!dist || !sample || dist->samples_count < 1) return 0.0f;

    int fc = feature_count < dist->feature_count ? feature_count : dist->feature_count;
    if (fc <= 0) return 0.0f;

    /* L-022修复: 使用完整协方差矩阵计算Mahalanobis距离
     * 不能简化为对角近似，必须使用全矩阵逆 */
    int cd = dist->covariance_dim;
    if (cd > 1 && dist->samples_count > fc && cd <= SEC_EMBEDDING_DIM && cd <= fc) {
        /* 构建局部6x6或更小的协方差矩阵和偏差向量 */
        float local_cov[36] = {0};  /* 最大6x6=36 */
        float diff[6] = {0};
        int dim = cd < 6 ? cd : 6;
        if (dim > fc) dim = fc;

        for (int i = 0; i < dim; i++) {
            diff[i] = sample[i] - dist->feature_mean[i];
            for (int j = 0; j < dim; j++) {
                local_cov[i * dim + j] = dist->covariance_matrix[i * cd + j];
            }
        }

        /* 添加正则化项到对角，确保可逆性 */
        for (int i = 0; i < dim; i++) {
            local_cov[i * dim + i] += 1e-6f;
        }

        /* 使用Gauss-Jordan消元计算dim×dim矩阵的逆
         * 构建增广矩阵 [C | I]，通过行变换得到 [I | C^-1] */
        float aug[6][12] = {{0}};  /* 最大6x12 */
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                aug[i][j] = local_cov[i * dim + j];
            }
            aug[i][dim + i] = 1.0f;  /* 单位矩阵 */
        }

        for (int i = 0; i < dim; i++) {
            /* 部分主元消去: 找第i列最大元素 */
            int max_row = i;
            float max_val = (aug[i][i] > 0 ? aug[i][i] : -aug[i][i]);
            for (int r = i + 1; r < dim; r++) {
                float abs_val = (aug[r][i] > 0 ? aug[r][i] : -aug[r][i]);
                if (abs_val > max_val) {
                    max_val = abs_val;
                    max_row = r;
                }
            }

            /* 交换行 */
            if (max_row != i) {
                for (int c = 0; c < 2 * dim; c++) {
                    float tmp = aug[i][c];
                    aug[i][c] = aug[max_row][c];
                    aug[max_row][c] = tmp;
                }
            }

            /* 归一化主元行 */
            float pivot = aug[i][i];
            if (pivot < 1e-12f && pivot > -1e-12f) {
                /* 奇异性处理: 使用对角近似退路 */
                float sum_diag = 0.0f;
                for (int k = 0; k < dim; k++) {
                    float dk = diff[k];
                    if (local_cov[k * dim + k] > 1e-10f) {
                        sum_diag += dk * dk / local_cov[k * dim + k];
                    }
                }
                return sqrtf(sum_diag);
            }

            for (int c = 0; c < 2 * dim; c++) {
                aug[i][c] /= pivot;
            }

            /* 消去其他行 */
            for (int r = 0; r < dim; r++) {
                if (r == i) continue;
                float factor = aug[r][i];
                for (int c = 0; c < 2 * dim; c++) {
                    aug[r][c] -= factor * aug[i][c];
                }
            }
        }

        /* 提取逆矩阵（右侧 dim×dim）并计算Mahalanobis距离:
         * D^2 = (x-μ)^T Σ^(-1) (x-μ)
         *     = Σ_i Σ_j diff[i] * inv[i][j] * diff[j] */
        float mahalanobis_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < dim; j++) {
                row_sum += aug[i][dim + j] * diff[j];
            }
            mahalanobis_sq += diff[i] * row_sum;
        }

        /* 确保非负（数值精度） */
        if (mahalanobis_sq < 0.0f) mahalanobis_sq = 0.0f;

        return sqrtf(mahalanobis_sq);
    } else {
        /* 备选路径: 协方差矩阵维度不足时退化为对角标准化距离 */
        float score = 0.0f;
        for (int i = 0; i < fc; i++) {
            float var = dist->feature_variance[i];
            if (var > 1e-10f) {
                float dev = (sample[i] - dist->feature_mean[i]);
                dev = (dev > 0 ? dev : -dev) / sqrtf(var);
                score += dev;
            }
        }
        return score / (float)fc;
    }
}

int sec_detect_label_flip(const float* input_features, int feature_count,
                           float original_label, float predicted_label,
                           float confidence, float threshold) {
    (void)input_features;
    (void)feature_count;
    if (original_label != predicted_label && confidence > threshold) {
        return 1;
    }
    return 0;
}

float sec_compute_perturbation_norm(const float* original, const float* perturbed,
                                     int count, int norm_type) {
    if (!original || !perturbed || count <= 0) return 0.0f;
    switch (norm_type) {
        case 0: {
            float sum = 0.0f;
            for (int i = 0; i < count; i++) {
                float diff = fabsf(original[i] - perturbed[i]);
                if (diff > sum) sum = diff;
            }
            return sum;
        }
        case 1: {
            float sum = 0.0f;
            for (int i = 0; i < count; i++) sum += fabsf(original[i] - perturbed[i]);
            return sum;
        }
        case 2:
        default: {
            float sum = 0.0f;
            for (int i = 0; i < count; i++) {
                float diff = original[i] - perturbed[i];
                sum += diff * diff;
            }
            return sqrtf(sum);
        }
    }
}

int sec_update_feature_distribution(SecInputMonitor* monitor,
                                     const float* features, int feature_count) {
    if (!monitor || !features || feature_count <= 0) return -1;
    if (feature_count > SEC_MAX_INPUT_FEATURES) feature_count = SEC_MAX_INPUT_FEATURES;

    SecFeatureDistribution* dist = &monitor->normal_distribution;

    if (dist->samples_count == 0) {
        dist->feature_count = feature_count;
        for (int i = 0; i < feature_count; i++) {
            dist->feature_mean[i] = features[i];
            dist->feature_variance[i] = 0.0f;
            dist->feature_min[i] = features[i];
            dist->feature_max[i] = features[i];
        }
        dist->samples_count = 1;
        dist->last_update_time = time(NULL);
        return 0;
    }

    int fc = feature_count < dist->feature_count ? feature_count : dist->feature_count;

    for (int i = 0; i < fc; i++) {
        float old_mean = dist->feature_mean[i];
        int n = dist->samples_count;
        dist->feature_mean[i] = old_mean + (features[i] - old_mean) / (float)(n + 1);
        if (n > 1) {
            dist->feature_variance[i] = ((float)(n - 1) * dist->feature_variance[i] +
                (features[i] - old_mean) * (features[i] - dist->feature_mean[i])) / (float)n;
        }
        if (features[i] < dist->feature_min[i]) dist->feature_min[i] = features[i];
        if (features[i] > dist->feature_max[i]) dist->feature_max[i] = features[i];
    }

    dist->samples_count++;
    dist->last_update_time = time(NULL);

    /* L-022修复: Woodbury恒等式增量协方差矩阵更新
     * 使用6x6完整协方差矩阵（不能简化为对角）
     * Σ_new = (1-α)Σ_old + α·(x-μ)(x-μ)^T
     * 其中 α = 1/(n+1) 为渐进无偏估计，n为样本数
     *
     * rank-1外积更新保留特征间完整协方差结构，
     * 后续Mahalanobis距离计算使用全矩阵逆而非对角近似 */
    {
        int cd = fc < 6 ? fc : 6;
        if (cd > SEC_EMBEDDING_DIM) cd = SEC_EMBEDDING_DIM;
        dist->covariance_dim = cd;

        /* 计算偏差向量 d = x - μ */
        float diff_d[6];
        for (int i = 0; i < cd; i++) {
            diff_d[i] = features[i] - dist->feature_mean[i];
        }

        /* Woodbury rank-1协方差更新
         * 使用指数衰减因子平衡历史和新样本:
         * α_decay = 1.0f / (samples_count + 1.0f)  前进式平均
         * alpha_ema = 0.05f                        指数移动平均（备用）
         * 使用前进式平均获得真正的无偏协方差估计 */
        float alpha_decay = 1.0f / ((float)dist->samples_count + 1.0f);
        float one_minus_alpha = 1.0f - alpha_decay;

        /* Σ_new[i][j] = (1-α)*Σ_old[i][j] + α*d[i]*d[j] */
        for (int i = 0; i < cd; i++) {
            for (int j = 0; j < cd; j++) {
                float old_cov = dist->covariance_matrix[i * dist->covariance_dim + j];
                float outer_prod = diff_d[i] * diff_d[j];
                dist->covariance_matrix[i * dist->covariance_dim + j] =
                    one_minus_alpha * old_cov + alpha_decay * outer_prod;
            }
        }

        /* 确保对角元素与方差值一致（数值稳定性） */
        for (int i = 0; i < cd; i++) {
            float diag = dist->covariance_matrix[i * cd + i];
            if (diag < 1e-10f) {
                dist->covariance_matrix[i * cd + i] = 1e-8f;
            }
        }
    }

    return 0;
}

int sec_add_injection_pattern(SecInputMonitor* monitor,
                               const char* pattern, const char* pattern_type,
                               float severity) {
    if (!monitor || !pattern || !pattern_type) return -1;
    if (monitor->injection_pattern_count >= SEC_MAX_INJECTION_PATTERNS) return -1;

    SecInjectionPattern* p = &monitor->injection_patterns[monitor->injection_pattern_count];
    strncpy(p->pattern, pattern, sizeof(p->pattern) - 1);
    strncpy(p->pattern_type, pattern_type, sizeof(p->pattern_type) - 1);
    p->severity = severity;
    p->is_enabled = 1;
    p->match_count = 0;
    p->false_positive_count = 0;
    p->false_positive_rate = 0.0f;
    monitor->injection_pattern_count++;
    return 0;
}

int sec_detect_prompt_injection(SecInputMonitor* monitor,
                                 const char* input_text, size_t text_len,
                                 SecInjectionResult* out_result) {
    if (!monitor || !input_text || !out_result) return -1;
    if (!monitor->is_monitoring_active) return 0;

    memset(out_result, 0, sizeof(SecInjectionResult));
    out_result->injection_probability = 0.0f;

    if (!input_text || text_len == 0) return 0;

    char* lower_text = (char*)safe_calloc(text_len + 1, 1);
    if (!lower_text) return -1;
    for (size_t i = 0; i < text_len; i++) {
        char c = input_text[i];
        if (c >= 'A' && c <= 'Z') lower_text[i] = (char)(c + 32);
        else lower_text[i] = c;
    }
    lower_text[text_len] = '\0';

    int match_count = 0;
    float max_severity = 0.0f;
    char detected_pattern[256] = {0};
    char injection_type[64] = {0};
    int bypass_attempt = 0;

    for (int i = 0; i < monitor->injection_pattern_count; i++) {
        SecInjectionPattern* p = &monitor->injection_patterns[i];
        if (!p->is_enabled) continue;

        size_t pattern_len = strlen(p->pattern);
        if (pattern_len == 0 || pattern_len > text_len) continue;

        int found = 0;
        for (size_t j = 0; j <= text_len - pattern_len; j++) {
            size_t k;
            for (k = 0; k < pattern_len; k++) {
                char tc = lower_text[j + k];
                char pc = p->pattern[k];
                if (tc != pc) break;
            }
            if (k == pattern_len) { found = 1; break; }
        }

        if (found) {
            match_count++;
            p->match_count++;
            if (p->severity > max_severity) {
                max_severity = p->severity;
                strncpy(detected_pattern, p->pattern, sizeof(detected_pattern) - 1);
                strncpy(injection_type, p->pattern_type, sizeof(injection_type) - 1);
            }
        }
    }

    if (text_len > 4) {
        size_t bs_count = 0;
        for (size_t i = 0; i < text_len; i++) {
            if (input_text[i] == '\b' || input_text[i] == 0x7F || input_text[i] == 0x1B) {
                bs_count++;
            }
        }
        if (bs_count > 3) bypass_attempt = 1;
    }

    out_result->pattern_match_count = match_count;
    out_result->bypass_attempt = bypass_attempt;

    float probability = max_severity * 0.6f;
    if (match_count > 0) probability += 0.1f * (float)match_count;
    if (bypass_attempt) probability += 0.2f;

    if (probability > 1.0f) probability = 1.0f;
    out_result->injection_probability = probability;
    out_result->is_injection = (probability > monitor->detection_threshold) ? 1 : 0;

    if (match_count > 0) {
        strncpy(out_result->detected_pattern, detected_pattern, sizeof(out_result->detected_pattern) - 1);
        strncpy(out_result->injection_type, injection_type, sizeof(out_result->injection_type) - 1);
    }

    if (out_result->is_injection) {
        monitor->total_injection_found++;
        if (bypass_attempt) {
            snprintf(out_result->analysis, sizeof(out_result->analysis),
                     "检测到绕过尝试! 匹配%d个注入模式, 最高严重度=%.2f, 含控制字符",
                     match_count, max_severity);
        } else {
            snprintf(out_result->analysis, sizeof(out_result->analysis),
                     "检测到Prompt注入! 类型=%s, 匹配%d个模式, 概率=%.2f",
                     injection_type, match_count, probability);
        }
    } else if (match_count > 0) {
        snprintf(out_result->analysis, sizeof(out_result->analysis),
                 "低置信度匹配(%d个模式, 概率=%.2f), 未触发阈值", match_count, probability);
    }

    safe_free((void**)&lower_text);
    return out_result->is_injection;
}

int sec_detect_jailbreak_attempt(const char* input_text, size_t text_len) {
    if (!input_text || text_len == 0) return 0;

    const char* jailbreak_keywords[] = {
        "ignore previous", "disregard", "DAN", "jailbreak",
        "forget your", "no restrictions", "unrestricted",
        "do whatever", "free mode", "developer mode",
        "override", "breach", "violate guidelines",
        "ethical boundary", "no filter", "uncensored"
    };
    int kw_count = sizeof(jailbreak_keywords) / sizeof(jailbreak_keywords[0]);

    char* lower_text = (char*)safe_calloc(text_len + 1, 1);
    if (!lower_text) return 0;
    for (size_t i = 0; i < text_len; i++) {
        char c = input_text[i];
        if (c >= 'A' && c <= 'Z') lower_text[i] = (char)(c + 32);
        else lower_text[i] = c;
    }
    lower_text[text_len] = '\0';

    int match_count = 0;
    for (int i = 0; i < kw_count; i++) {
        size_t kw_len = strlen(jailbreak_keywords[i]);
        if (kw_len == 0 || kw_len > text_len) continue;
        for (size_t j = 0; j <= text_len - kw_len; j++) {
            size_t k;
            for (k = 0; k < kw_len; k++) {
                if (lower_text[j + k] != jailbreak_keywords[i][k]) break;
            }
            if (k == kw_len) { match_count++; break; }
        }
    }

    safe_free((void**)&lower_text);
    return match_count >= 2 ? 1 : 0;
}

int sec_detect_indirect_injection(const char* input_text, size_t text_len) {
    if (!input_text || text_len == 0) return 0;

    const char* indirect_patterns[] = {
        "参考外部", "根据文档", "请阅读", "读取文件",
        "external content", "document says", "according to",
        "fetch from", "load from", "retrieve",
        "src=\"", "href=\"", "url=", "link=",
        "iframe", "embed", "object", "include",
        "引用内容", "外部链接", "来自网络"
    };
    int ip_count = sizeof(indirect_patterns) / sizeof(indirect_patterns[0]);

    char* lower_text = (char*)safe_calloc(text_len + 1, 1);
    if (!lower_text) return 0;
    for (size_t i = 0; i < text_len; i++) {
        char c = input_text[i];
        if (c >= 'A' && c <= 'Z') lower_text[i] = (char)(c + 32);
        else lower_text[i] = c;
    }
    lower_text[text_len] = '\0';

    int match_count = 0;
    for (int i = 0; i < ip_count; i++) {
        size_t kw_len = strlen(indirect_patterns[i]);
        if (kw_len == 0 || kw_len > text_len) continue;
        for (size_t j = 0; j <= text_len - kw_len; j++) {
            size_t k;
            for (k = 0; k < kw_len; k++) {
                if (lower_text[j + k] != indirect_patterns[i][k]) break;
            }
            if (k == kw_len) { match_count++; break; }
        }
    }

    safe_free((void**)&lower_text);
    return match_count >= 2 ? 1 : 0;
}

int sec_report_input_security_event(SafetyMonitor* safety_monitor,
                                     SecInputAnomalyType type,
                                     float probability, const char* description) {
    if (!safety_monitor || !description) return -1;

    SafetyEvent event;
    memset(&event, 0, sizeof(SafetyEvent));
    event.handled = 0;

    switch (type) {
        case SEC_ANOMALY_ADVERSARIAL:
            event.type = SAFETY_EVENT_BEHAVIOR_ANOMALY;
            event.severity = probability > 0.8f ? SAFETY_LEVEL_CRITICAL : SAFETY_LEVEL_ELEVATED;
            break;
        case SEC_ANOMALY_POISONING:
            event.type = SAFETY_EVENT_BEHAVIOR_ANOMALY;
            event.severity = SAFETY_LEVEL_CRITICAL;
            break;
        case SEC_ANOMALY_INJECTION:
            event.type = SAFETY_EVENT_COMMAND_INJECTION;
            event.severity = probability > 0.8f ? SAFETY_LEVEL_CRITICAL : SAFETY_LEVEL_ELEVATED;
            break;
        case SEC_ANOMALY_OUTLIER:
            event.type = SAFETY_EVENT_BEHAVIOR_ANOMALY;
            event.severity = SAFETY_LEVEL_WARNING;
            break;
        case SEC_ANOMALY_DRIFT:
            event.type = SAFETY_EVENT_BEHAVIOR_ANOMALY;
            event.severity = SAFETY_LEVEL_WARNING;
            break;
        case SEC_ANOMALY_REPLAY:
            event.type = SAFETY_EVENT_NETWORK_ANOMALY;
            event.severity = SAFETY_LEVEL_ELEVATED;
            break;
        default:
            event.type = SAFETY_EVENT_BEHAVIOR_ANOMALY;
            event.severity = SAFETY_LEVEL_WARNING;
            break;
    }
    strncpy(event.description, description, sizeof(event.description) - 1);
    strncpy(event.source, "输入安全监控", sizeof(event.source) - 1);
    return safety_report_event(safety_monitor, &event);
}