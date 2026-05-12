/**
 * @file product_design_enhanced.c
 * @brief A09.5 产品设计系统增强实现
 *
 * 两大子模块：
 *   A09.5.1 需求分析引擎（需求验证/追踪/一致性检查）
 *   A09.5.2 设计优化引擎（多目标优化/设计空间探索/敏感性分析）
 */

#include "selflnn/product_design/product_design_enhanced.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/laplace.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 安全字符串复制 */
static void pde_strcpy(char* dst, size_t dst_size, const char* src)
{
    if (!dst || !src || dst_size == 0) return;
    size_t i;
    for (i = 0; i < dst_size - 1 && src[i] != '\0'; ++i)
        dst[i] = src[i];
    dst[i] = '\0';
}

/* 简单线性同余随机数生成器 */
#ifdef _WIN32
#include <windows.h>
static CRITICAL_SECTION g_pde_rng_lock;
static int g_pde_rng_lock_init = 0;
#define PDE_RNG_LOCK() do { if (!g_pde_rng_lock_init) { InitializeCriticalSection(&g_pde_rng_lock); g_pde_rng_lock_init = 1; } EnterCriticalSection(&g_pde_rng_lock); } while(0)
#define PDE_RNG_UNLOCK() LeaveCriticalSection(&g_pde_rng_lock)
#else
#include <pthread.h>
static pthread_mutex_t g_pde_rng_lock = PTHREAD_MUTEX_INITIALIZER;
#define PDE_RNG_LOCK() pthread_mutex_lock(&g_pde_rng_lock)
#define PDE_RNG_UNLOCK() pthread_mutex_unlock(&g_pde_rng_lock)
#endif
static unsigned int pde_rand_state = 12345;
static void pde_srand(unsigned int seed) { PDE_RNG_LOCK(); pde_rand_state = seed; PDE_RNG_UNLOCK(); }
static double pde_rand_double(void)
{
    PDE_RNG_LOCK();
    pde_rand_state = pde_rand_state * 1103515245 + 12345;
    double r = (double)(pde_rand_state & 0x7FFFFFFF) / 2147483648.0;
    PDE_RNG_UNLOCK();
    return r;
}
static int pde_rand_int(int min, int max)
{
    if (min >= max) return min;
    return min + (int)(pde_rand_double() * (max - min + 1));
}

/* 标准正态分布随机数（Box-Muller） */
static double pde_rand_normal(double mean, double stddev)
{
    double u1 = pde_rand_double();
    double u2 = pde_rand_double();
    if (u1 < 1e-15) u1 = 1e-15;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z * stddev;
}

/* 复制参数数组 */
static DesignParameter* pde_copy_params(const DesignParameter* src, size_t count)
{
    if (!src || count == 0) return NULL;
    DesignParameter* dst = (DesignParameter*)malloc(sizeof(DesignParameter) * count);
    if (!dst) return NULL;
    for (size_t i = 0; i < count; ++i) {
        dst[i] = src[i];
        dst[i].name = NULL;
        if (src[i].name) {
            dst[i].name = (char*)malloc(strlen(src[i].name) + 1);
            if (dst[i].name) pde_strcpy(dst[i].name, strlen(src[i].name) + 1, src[i].name);
        }
    }
    return dst;
}

/* 释放参数数组 */
static void pde_free_params(DesignParameter* params, size_t count)
{
    if (!params) return;
    for (size_t i = 0; i < count; ++i)
        free(params[i].name);
    free(params);
}

/* 评估一个设计方案的各目标值 */
static int pde_evaluate_design(const ProductRequirement* req,
                                const DesignParameter* params, size_t param_count,
                                float* objectives, int obj_count)
{
    if (!req || !params || !objectives) return -1;
    float cost = 0.5f, feasibility = 0.7f, innovation = 0.5f, market = 0.6f, complexity = 0.4f;
    for (size_t i = 0; i < param_count && i < 5; ++i) {
        double norm = (params[i].current_value - params[i].min_value) /
                      (params[i].max_value - params[i].min_value + 1e-12);
        if (norm < 0) norm = 0; if (norm > 1) norm = 1;
        cost += (float)norm * 0.3f;
        feasibility += (float)(1.0 - norm) * 0.2f;
        innovation += (float)norm * 0.3f;
        market += (float)(0.5 + norm * 0.5) * 0.2f;
        complexity += (float)norm * 0.4f;
    }
    if (cost > 1.0f) cost = 1.0f;
    if (feasibility > 1.0f) feasibility = 1.0f;
    if (innovation > 1.0f) innovation = 1.0f;
    if (market > 1.0f) market = 1.0f;
    for (int i = 0; i < obj_count && i < PDE_MAX_OBJECTIVES; ++i)
        objectives[i] = 0.5f;
    if (obj_count >= 1) objectives[0] = cost;
    if (obj_count >= 2) objectives[1] = feasibility;
    if (obj_count >= 3) objectives[2] = innovation;
    if (obj_count >= 4) objectives[3] = market;
    if (obj_count >= 5) objectives[4] = complexity;
    return 0;
}

/* 基于需求约束的自适应综合评分
 * 根据需求的最大成本(max_cost)和工期(max_time)动态调整各目标权重，
 * 替代原始的固定 0.5/0.3/0.2 硬编码权重。
 * 权重维度: [成本, 可行性, 创新, 市场, 复杂度]
 */
static float pde_compute_composite_score(const ProductRequirement* req,
                                          const float* obj_vals, int obj_count)
{
    if (!obj_vals || obj_count <= 0) return 0.0f;

    float w[5] = {0.30f, 0.25f, 0.20f, 0.15f, 0.10f};

    if (req) {
        if (req->max_cost > 0.0) {
            w[0] += 0.10f;
            w[3] -= 0.05f;
            w[4] -= 0.05f;
        }
        if (req->max_time > 0.0) {
            if (req->max_time <= 6.0) {
                w[1] += 0.10f;
                w[2] -= 0.10f;
            } else if (req->max_time >= 18.0) {
                w[2] += 0.10f;
                w[1] -= 0.10f;
            }
        }
    }

    float sum = 0.0f;
    int n = (obj_count < 5) ? obj_count : 5;
    for (int i = 0; i < n; ++i) sum += w[i];
    if (sum < 1e-6f) sum = 1.0f;
    for (int i = 0; i < n; ++i) w[i] /= sum;

    float score = 0.0f;
    if (obj_count >= 1) score += w[0] * (1.0f - obj_vals[0]);
    if (obj_count >= 2) score += w[1] * obj_vals[1];
    if (obj_count >= 3) score += w[2] * obj_vals[2];
    if (obj_count >= 4) score += w[3] * obj_vals[3];
    if (obj_count >= 5) score += w[4] * (1.0f - obj_vals[4]);
    return score;
}

/* ============================================================================
 * A09.5.1 需求追踪器内部结构
 * ============================================================================ */

struct PdeRequirementTracker {
    PdeRequirementItem requirements[PDE_MAX_REQUIREMENTS];
    int req_count;
    PdeTraceLink trace_links[PDE_MAX_TRACE_LINKS];
    int link_count;
    int next_req_id;
    int next_link_id;
};

/* ============================================================================
 * A09.5.1 需求追踪器实现
 * ============================================================================ */

PdeRequirementTracker* pde_tracker_create(void)
{
    PdeRequirementTracker* tracker = (PdeRequirementTracker*)
        malloc(sizeof(PdeRequirementTracker));
    if (!tracker) return NULL;
    memset(tracker, 0, sizeof(PdeRequirementTracker));
    tracker->next_req_id = 1;
    tracker->next_link_id = 1;
    
    return tracker;
}

void pde_tracker_destroy(PdeRequirementTracker* tracker)
{
    free(tracker);
}

int pde_tracker_register_requirements(PdeRequirementTracker* tracker,
                                       const char* requirement_text)
{
    if (!tracker || !requirement_text) return -1;

    char text_copy[4096];
    pde_strcpy(text_copy, sizeof(text_copy), requirement_text);

    /* 按行分割需求文本 */
    char* line = strtok(text_copy, "\n");
    while (line && tracker->req_count < PDE_MAX_REQUIREMENTS) {
        while (*line == ' ' || *line == '\t') line++;
        if (*line == '\0' || *line == '#' || *line == '/') {
            line = strtok(NULL, "\n");
            continue;
        }

        PdeRequirementItem* item = &tracker->requirements[tracker->req_count];
        memset(item, 0, sizeof(PdeRequirementItem));
        item->req_id = tracker->next_req_id++;
        item->status = PDE_REQ_DRAFT;
        item->priority = PDE_PRIORITY_MEDIUM;
        item->source_line = tracker->req_count + 1;
        item->parent_id = -1;

        /* 解析行内格式: [ID] 标题 | 描述 | 优先级 */
        char* title_start = line;
        char* desc_sep = strstr(line, "|");
        if (desc_sep) {
            *desc_sep = '\0';
            pde_strcpy(item->title, sizeof(item->title), title_start);
            const char* desc = desc_sep + 1;
            char* prio_sep = strstr(desc, "|");
            if (prio_sep) {
                *prio_sep = '\0';
                pde_strcpy(item->description, sizeof(item->description), desc);
                const char* prio_str = prio_sep + 1;
                while (*prio_str == ' ') prio_str++;
                if (strstr(prio_str, "关键")) item->priority = PDE_PRIORITY_CRITICAL;
                else if (strstr(prio_str, "高")) item->priority = PDE_PRIORITY_HIGH;
                else if (strstr(prio_str, "低")) item->priority = PDE_PRIORITY_LOW;
                else if (strstr(prio_str, "可选")) item->priority = PDE_PRIORITY_OPTIONAL;
                else item->priority = PDE_PRIORITY_MEDIUM;
            } else {
                pde_strcpy(item->description, sizeof(item->description), desc);
            }
        } else {
            pde_strcpy(item->title, sizeof(item->title), line);
            pde_strcpy(item->description, sizeof(item->description), line);
        }

        item->estimated_effort = 1.0f + (float)pde_rand_double() * 9.0f;
        item->business_value = (float)(PDE_PRIORITY_OPTIONAL - item->priority) * 2.5f + 2.0f;
        item->version = 1;
        snprintf(item->source_origin, sizeof(item->source_origin), "需求文本行 %d", item->source_line);
        snprintf(item->change_history, sizeof(item->change_history), "v1: 初始创建");

        tracker->req_count++;
        line = strtok(NULL, "\n");
    }

    return tracker->req_count;
}

int pde_tracker_get_requirement_count(const PdeRequirementTracker* tracker)
{
    return tracker ? tracker->req_count : 0;
}

const PdeRequirementItem* pde_tracker_get_requirement(
    const PdeRequirementTracker* tracker, int req_id)
{
    if (!tracker) return NULL;
    for (int i = 0; i < tracker->req_count; ++i) {
        if (tracker->requirements[i].req_id == req_id)
            return &tracker->requirements[i];
    }
    return NULL;
}

int pde_tracker_set_status(PdeRequirementTracker* tracker,
                            int req_id, PdeRequirementStatus status)
{
    if (!tracker) return -1;
    for (int i = 0; i < tracker->req_count; ++i) {
        if (tracker->requirements[i].req_id == req_id) {
            tracker->requirements[i].status = status;
            tracker->requirements[i].version++;
            char history[1024];
            snprintf(history, sizeof(history), "\nv%d: 状态更新为 %d",
                     tracker->requirements[i].version, (int)status);
            strncat(tracker->requirements[i].change_history, history,
                    sizeof(tracker->requirements[i].change_history) -
                    strlen(tracker->requirements[i].change_history) - 1);
            return 0;
        }
    }
    return -1;
}

int pde_tracker_set_priority(PdeRequirementTracker* tracker,
                              int req_id, PdeRequirementPriority priority)
{
    if (!tracker) return -1;
    for (int i = 0; i < tracker->req_count; ++i) {
        if (tracker->requirements[i].req_id == req_id) {
            tracker->requirements[i].priority = priority;
            return 0;
        }
    }
    return -1;
}

int pde_tracker_add_dependency(PdeRequirementTracker* tracker,
                                int req_id, int depends_on_id)
{
    if (!tracker) return -1;
    for (int i = 0; i < tracker->req_count; ++i) {
        if (tracker->requirements[i].req_id == req_id) {
            PdeRequirementItem* item = &tracker->requirements[i];
            if (item->dependency_count >= 32) return -1;
            item->dependency_ids[item->dependency_count++] = depends_on_id;
            return 0;
        }
    }
    return -1;
}

int pde_tracker_add_trace_link(PdeRequirementTracker* tracker,
                                int source_id, int target_id,
                                const char* link_type)
{
    if (!tracker || tracker->link_count >= PDE_MAX_TRACE_LINKS) return -1;
    PdeTraceLink* link = &tracker->trace_links[tracker->link_count++];
    link->link_id = tracker->next_link_id++;
    link->source_req_id = source_id;
    link->target_req_id = target_id;
    pde_strcpy(link->link_type, sizeof(link->link_type), link_type ? link_type : "依赖");
    snprintf(link->description, sizeof(link->description),
             "需求 #%d -> #%d (%s)", source_id, target_id, link->link_type);
    return 0;
}

/* ============================================================================
 * A09.5.1 需求验证实现
 * ============================================================================ */

int pde_validate_requirements(const PdeRequirementTracker* tracker,
                               PdeValidationReport* report)
{
    if (!tracker || !report) return -1;
    memset(report, 0, sizeof(PdeValidationReport));

    PdeValidationIssue issues[256];
    int total_issues = 0;

    /* 完整性检查 */
    int comp_count = pde_check_completeness(tracker, issues, 256);
    total_issues += comp_count;

    /* 一致性检查 */
    int cons_count = pde_check_consistency(tracker, &issues[total_issues], 256 - total_issues);
    total_issues += cons_count;

    /* 可行性检查 */
    int feas_count = pde_check_feasibility(tracker, &issues[total_issues], 256 - total_issues);
    total_issues += feas_count;

    if (total_issues > 256) total_issues = 256;
    for (int i = 0; i < total_issues; ++i)
        report->issues[i] = issues[i];
    report->issue_count = total_issues;

    /* 计算各项评分 */
    if (tracker->req_count > 0) {
        int complete_count = 0;
        for (int i = 0; i < tracker->req_count; ++i) {
            if (strlen(tracker->requirements[i].title) > 0 &&
                strlen(tracker->requirements[i].description) > 0)
                complete_count++;
        }
        report->completeness_score = (float)complete_count / tracker->req_count * 100.0f;

        int conflict_count = 0;
        for (int i = 0; i < total_issues; ++i) {
            if (strcmp(issues[i].category, "一致性") == 0)
                conflict_count++;
        }
        report->consistency_score = tracker->req_count > 0 ?
            100.0f - (float)conflict_count / tracker->req_count * 50.0f : 100.0f;
        if (report->consistency_score < 0) report->consistency_score = 0;

        float total_effort = 0, total_value = 0;
        for (int i = 0; i < tracker->req_count; ++i) {
            total_effort += tracker->requirements[i].estimated_effort;
            total_value += tracker->requirements[i].business_value;
        }
        report->feasibility_score = total_effort > 0 ?
            (total_value / total_effort) * 50.0f : 50.0f;
        if (report->feasibility_score > 100) report->feasibility_score = 100;
    }

    report->passed = (report->issue_count == 0);
    snprintf(report->summary, sizeof(report->summary),
             "需求验证: %s | 完整性 %.1f%% | 一致性 %.1f%% | 可行性 %.1f%% | %d 个问题",
             report->passed ? "通过" : "未通过",
             report->completeness_score, report->consistency_score,
             report->feasibility_score, report->issue_count);

    return report->passed ? 0 : 1;
}

int pde_check_completeness(const PdeRequirementTracker* tracker,
                            PdeValidationIssue* issues, int max_count)
{
    if (!tracker || !issues || max_count <= 0) return 0;
    int count = 0;

    for (int i = 0; i < tracker->req_count && count < max_count; ++i) {
        const PdeRequirementItem* item = &tracker->requirements[i];

        if (strlen(item->title) == 0) {
            issues[count].issue_id = count + 1;
            issues[count].severity = PDE_SEVERITY_ERROR;
            pde_strcpy(issues[count].category, sizeof(issues[count].category), "完整性");
            snprintf(issues[count].description, sizeof(issues[count].description),
                     "需求 #%d 缺少标题", item->req_id);
            snprintf(issues[count].location, sizeof(issues[count].location),
                     "行 %d", item->source_line);
            snprintf(issues[count].suggestion, sizeof(issues[count].suggestion),
                     "为需求添加简洁明确的标题");
            count++;
        }

        if (strlen(item->description) == 0 && count < max_count) {
            issues[count].issue_id = count + 1;
            issues[count].severity = PDE_SEVERITY_WARNING;
            pde_strcpy(issues[count].category, sizeof(issues[count].category), "完整性");
            snprintf(issues[count].description, sizeof(issues[count].description),
                     "需求 #%d (%s) 缺少详细描述", item->req_id, item->title);
            snprintf(issues[count].location, sizeof(issues[count].location),
                     "行 %d", item->source_line);
            snprintf(issues[count].suggestion, sizeof(issues[count].suggestion),
                     "补充需求的详细描述，包括功能、约束和验收标准");
            count++;
        }

        if (item->estimated_effort <= 0 && count < max_count) {
            issues[count].issue_id = count + 1;
            issues[count].severity = PDE_SEVERITY_INFO;
            pde_strcpy(issues[count].category, sizeof(issues[count].category), "完整性");
            snprintf(issues[count].description, sizeof(issues[count].description),
                     "需求 #%d (%s) 未估算工作量", item->req_id, item->title);
            snprintf(issues[count].location, sizeof(issues[count].location),
                     "行 %d", item->source_line);
            pde_strcpy(issues[count].suggestion, sizeof(issues[count].suggestion),
                       "进行工作量估算以便于排期");
            count++;
        }
    }

    return count;
}

int pde_check_consistency(const PdeRequirementTracker* tracker,
                           PdeValidationIssue* issues, int max_count)
{
    if (!tracker || !issues || max_count <= 0) return 0;
    int count = 0;

    /* 检测循环依赖：DFS 环检测 */
    int visited[PDE_MAX_REQUIREMENTS];
    int rec_stack[PDE_MAX_REQUIREMENTS];
    memset(visited, 0, sizeof(visited));
    memset(rec_stack, 0, sizeof(rec_stack));

    for (int i = 0; i < tracker->req_count && count < max_count; ++i) {
        if (!visited[i]) {
            /* DFS */
            int stack[PDE_MAX_REQUIREMENTS];
            int stack_pos[PDE_MAX_REQUIREMENTS];
            int sp = 0;
            stack[sp] = i;
            stack_pos[sp] = 0;
            visited[i] = 1;
            rec_stack[i] = 1;
            sp++;

            while (sp > 0 && count < max_count) {
                int cur = stack[sp - 1];
                int next_dep = stack_pos[sp - 1];

                if (next_dep >= tracker->requirements[cur].dependency_count) {
                    sp--;
                    rec_stack[cur] = 0;
                } else {
                    int dep = -1;
                    for (int k = 0; k < tracker->req_count; ++k) {
                        if (tracker->requirements[k].req_id ==
                            tracker->requirements[cur].dependency_ids[next_dep]) {
                            dep = k;
                            break;
                        }
                    }
                    stack_pos[sp - 1]++;

                    if (dep >= 0) {
                        if (rec_stack[dep]) {
                            issues[count].issue_id = count + 1;
                            issues[count].severity = PDE_SEVERITY_CRITICAL;
                            pde_strcpy(issues[count].category, sizeof(issues[count].category), "一致性");
                            snprintf(issues[count].description, sizeof(issues[count].description),
                                     "检测到循环依赖: 需求 #%d <-> #%d",
                                     tracker->requirements[cur].req_id,
                                     tracker->requirements[dep].req_id);
                            snprintf(issues[count].location, sizeof(issues[count].location),
                                     "%s -> %s",
                                     tracker->requirements[cur].title,
                                     tracker->requirements[dep].title);
                            pde_strcpy(issues[count].suggestion, sizeof(issues[count].suggestion),
                                       "重新评估需求间的依赖关系，消除循环");
                            count++;
                        }
                        if (!visited[dep]) {
                            stack[sp] = dep;
                            stack_pos[sp] = 0;
                            visited[dep] = 1;
                            rec_stack[dep] = 1;
                            sp++;
                        }
                    }
                }
            }
        }
    }

    /* 检测优先级冲突 */
    for (int i = 0; i < tracker->req_count && count < max_count; ++i) {
        for (int j = i + 1; j < tracker->req_count && count < max_count; ++j) {
            if (tracker->requirements[i].priority == PDE_PRIORITY_CRITICAL &&
                tracker->requirements[j].priority == PDE_PRIORITY_CRITICAL) {
                /* 过多的关键优先级可能导致资源冲突 */
                issues[count].issue_id = count + 1;
                issues[count].severity = PDE_SEVERITY_WARNING;
                pde_strcpy(issues[count].category, sizeof(issues[count].category), "一致性");
                snprintf(issues[count].description, sizeof(issues[count].description),
                         "存在多个关键优先级需求: #%d (%s) 和 #%d (%s)",
                         tracker->requirements[i].req_id, tracker->requirements[i].title,
                         tracker->requirements[j].req_id, tracker->requirements[j].title);
                snprintf(issues[count].location, sizeof(issues[count].location),
                         "需求 #%d 和 #%d", tracker->requirements[i].req_id,
                         tracker->requirements[j].req_id);
                pde_strcpy(issues[count].suggestion, sizeof(issues[count].suggestion),
                           "重新评估各需求的真实优先级，关键级不超过3个");
                count++;
            }
        }
    }

    return count;
}

int pde_check_feasibility(const PdeRequirementTracker* tracker,
                           PdeValidationIssue* issues, int max_count)
{
    if (!tracker || !issues || max_count <= 0) return 0;
    int count = 0;

    float total_effort = 0;
    float total_value = 0;
    for (int i = 0; i < tracker->req_count; ++i) {
        total_effort += tracker->requirements[i].estimated_effort;
        total_value += tracker->requirements[i].business_value;
    }

    /* 检查整体工作量是否过大 */
    if (total_effort > 100 && count < max_count) {
        issues[count].issue_id = count + 1;
        issues[count].severity = PDE_SEVERITY_WARNING;
        pde_strcpy(issues[count].category, sizeof(issues[count].category), "可行性");
        snprintf(issues[count].description, sizeof(issues[count].description),
                 "总工作量 %.1f 人月较大，建议分阶段实施", total_effort);
        pde_strcpy(issues[count].location, sizeof(issues[count].location), "整体");
        pde_strcpy(issues[count].suggestion, sizeof(issues[count].suggestion),
                   "对需求进行优先级排序，MVP阶段只实现关键和高优先级需求");
        count++;
    }

    /* 检查价值/工作量比过低的需求 */
    for (int i = 0; i < tracker->req_count && count < max_count; ++i) {
        const PdeRequirementItem* item = &tracker->requirements[i];
        if (item->estimated_effort > 0) {
            float ratio = item->business_value / item->estimated_effort;
            if (ratio < 0.5f) {
                issues[count].issue_id = count + 1;
                issues[count].severity = PDE_SEVERITY_WARNING;
                pde_strcpy(issues[count].category, sizeof(issues[count].category), "可行性");
                snprintf(issues[count].description, sizeof(issues[count].description),
                         "需求 #%d (%s) 价值/工作量比过低 (%.2f)",
                         item->req_id, item->title, ratio);
                snprintf(issues[count].location, sizeof(issues[count].location),
                         "行 %d", item->source_line);
                pde_strcpy(issues[count].suggestion, sizeof(issues[count].suggestion),
                           "重新评估此需求的商业价值或考虑降低实现复杂度");
                count++;
            }
        }
    }

    return count;
}

int pde_generate_trace_report(const PdeRequirementTracker* tracker,
                               char* output, size_t output_size)
{
    if (!tracker || !output || output_size == 0) return -1;

    char* pos = output;
    size_t remaining = output_size;

    int written = snprintf(pos, remaining,
        "=================================================================\n"
        "  需求追踪报告\n"
        "=================================================================\n"
        "  总需求数: %d\n"
        "  追踪链接: %d\n"
        "  状态分布: ",
        tracker->req_count, tracker->link_count);

    if (written > 0 && (size_t)written < remaining) {
        pos += written;
        remaining -= written;
    }

    /* 统计各状态数量 */
    int draft_count = 0, validated_count = 0, approved_count = 0;
    int implemented_count = 0, verified_count = 0, rejected_count = 0;
    for (int i = 0; i < tracker->req_count; ++i) {
        switch (tracker->requirements[i].status) {
        case PDE_REQ_DRAFT:       draft_count++; break;
        case PDE_REQ_VALIDATED:   validated_count++; break;
        case PDE_REQ_APPROVED:    approved_count++; break;
        case PDE_REQ_IMPLEMENTED: implemented_count++; break;
        case PDE_REQ_VERIFIED:    verified_count++; break;
        case PDE_REQ_REJECTED:    rejected_count++; break;
        default: break;
        }
    }

    written = snprintf(pos, remaining,
        "草稿 %d, 已验证 %d, 已审批 %d, 已实现 %d, 已验证 %d, 已拒绝 %d\n"
        "-----------------------------------------------------------------\n",
        draft_count, validated_count, approved_count,
        implemented_count, verified_count, rejected_count);
    if (written > 0 && (size_t)written < remaining) {
        pos += written;
        remaining -= written;
    }

    /* 需求明细 */
    for (int i = 0; i < tracker->req_count; ++i) {
        const PdeRequirementItem* item = &tracker->requirements[i];
        const char* status_str = "未知";
        switch (item->status) {
        case PDE_REQ_DRAFT:       status_str = "草稿"; break;
        case PDE_REQ_VALIDATED:   status_str = "已验证"; break;
        case PDE_REQ_APPROVED:    status_str = "已审批"; break;
        case PDE_REQ_IMPLEMENTED: status_str = "已实现"; break;
        case PDE_REQ_VERIFIED:    status_str = "已验证"; break;
        case PDE_REQ_REJECTED:    status_str = "已拒绝"; break;
        case PDE_REQ_CHANGED:     status_str = "已变更"; break;
        }
        const char* prio_str = "中";
        switch (item->priority) {
        case PDE_PRIORITY_CRITICAL: prio_str = "关键"; break;
        case PDE_PRIORITY_HIGH:     prio_str = "高"; break;
        case PDE_PRIORITY_LOW:      prio_str = "低"; break;
        case PDE_PRIORITY_OPTIONAL: prio_str = "可选"; break;
        default: break;
        }

        written = snprintf(pos, remaining,
            "  #%03d [%s][%s] %s\n"
            "        %s\n"
            "        工作量: %.1f | 价值: %.1f | 版本: %d\n",
            item->req_id, status_str, prio_str, item->title,
            item->description, item->estimated_effort,
            item->business_value, item->version);
        if (written > 0 && (size_t)written < remaining) {
            pos += written;
            remaining -= written;
        }

        if (item->dependency_count > 0) {
            written = snprintf(pos, remaining, "        依赖: ");
            if (written > 0 && (size_t)written < remaining) {
                pos += written;
                remaining -= written;
            }
            for (int d = 0; d < item->dependency_count; ++d) {
                written = snprintf(pos, remaining, "#%d ", item->dependency_ids[d]);
                if (written > 0 && (size_t)written < remaining) {
                    pos += written;
                    remaining -= written;
                }
            }
            written = snprintf(pos, remaining, "\n");
            if (written > 0 && (size_t)written < remaining) {
                pos += written;
                remaining -= written;
            }
        }
    }

    /* 追踪链接明细 */
    if (tracker->link_count > 0) {
        written = snprintf(pos, remaining,
            "-----------------------------------------------------------------\n"
            "  追踪链接 (%d):\n", tracker->link_count);
        if (written > 0 && (size_t)written < remaining) {
            pos += written;
            remaining -= written;
        }
        for (int i = 0; i < tracker->link_count; ++i) {
            written = snprintf(pos, remaining, "  #%03d: %s\n",
                tracker->trace_links[i].link_id, tracker->trace_links[i].description);
            if (written > 0 && (size_t)written < remaining) {
                pos += written;
                remaining -= written;
            }
        }
    }

    snprintf(pos, remaining, "=================================================================\n");
    return 0;
}

/* ============================================================================
 * A09.5.2 多目标优化器内部结构
 * ============================================================================ */

struct PdeMultiObjectiveOptimizer {
    PdeObjective objectives[PDE_MAX_OBJECTIVES];
    int objective_count;
    int initialized;
};

/* 非支配排序 */
static int pde_dominates(const float* a, const float* b,
                          const PdeObjective* objs, int obj_count)
{
    int better_in_any = 0;
    int worse_in_none = 1;
    for (int i = 0; i < obj_count; ++i) {
        if (objs[i].maximize) {
            if (a[i] > b[i]) better_in_any = 1;
            if (a[i] < b[i]) worse_in_none = 0;
        } else {
            if (a[i] < b[i]) better_in_any = 1;
            if (a[i] > b[i]) worse_in_none = 0;
        }
    }
    return better_in_any && worse_in_none;
}

static void pde_non_dominated_sort(PdeParetoSolution* pop, int pop_size,
                                    const PdeObjective* objs, int obj_count)
{
    /* 初始化支配关系 */
    int dominated_by[PDE_MAX_PARETO_SOLUTIONS];
    int dominates_list[PDE_MAX_PARETO_SOLUTIONS][PDE_MAX_PARETO_SOLUTIONS];
    int dominates_count[PDE_MAX_PARETO_SOLUTIONS];
    memset(dominated_by, 0, sizeof(dominated_by));
    memset(dominates_count, 0, sizeof(dominates_count));

    for (int i = 0; i < pop_size; ++i) {
        for (int j = 0; j < pop_size; ++j) {
            if (i == j) continue;
            if (pde_dominates(pop[i].objective_values, pop[j].objective_values,
                              objs, obj_count)) {
                dominates_list[i][dominates_count[i]++] = j;
                dominated_by[j]++;
            }
        }
    }

    /* 第一前沿 */
    int front[PDE_MAX_PARETO_SOLUTIONS];
    int front_count = 0;
    for (int i = 0; i < pop_size; ++i) {
        if (dominated_by[i] == 0)
            front[front_count++] = i;
    }

    /* 分配前沿等级 */
    int current_rank = 0;
    int* current_front = front;
    int current_count = front_count;
    int* next_front = NULL;
    int next_count = 0;

    while (current_count > 0) {
        int next_buf[PDE_MAX_PARETO_SOLUTIONS];
        next_front = next_buf;
        next_count = 0;

        for (int i = 0; i < current_count; ++i) {
            int idx = current_front[i];
            pop[idx].rank = current_rank;
            for (int j = 0; j < dominates_count[idx]; ++j) {
                int dominated_idx = dominates_list[idx][j];
                dominated_by[dominated_idx]--;
                if (dominated_by[dominated_idx] == 0)
                    next_front[next_count++] = dominated_idx;
            }
        }
        current_front = next_front;
        current_count = next_count;
        current_rank++;
    }
}

/* 拥挤度距离 */
static void pde_crowding_distance(PdeParetoSolution* pop, int pop_size,
                                   const PdeObjective* objs, int obj_count)
{
    for (int i = 0; i < pop_size; ++i)
        pop[i].crowding_distance = 0;

    for (int o = 0; o < obj_count; ++o) {
        /* 按目标值排序 */
        int indices[PDE_MAX_PARETO_SOLUTIONS];
        for (int i = 0; i < pop_size; ++i) indices[i] = i;
        for (int i = 0; i < pop_size - 1; ++i) {
            for (int j = i + 1; j < pop_size; ++j) {
                if ((objs[o].maximize &&
                     pop[indices[i]].objective_values[o] < pop[indices[j]].objective_values[o]) ||
                    (!objs[o].maximize &&
                     pop[indices[i]].objective_values[o] > pop[indices[j]].objective_values[o])) {
                    int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
                }
            }
        }
        /* 边界点无限大距离 */
        pop[indices[0]].crowding_distance = FLT_MAX;
        pop[indices[pop_size - 1]].crowding_distance = FLT_MAX;
        /* 中间点归一化距离 */
        float min_val = pop[indices[0]].objective_values[o];
        float max_val = pop[indices[pop_size - 1]].objective_values[o];
        float range = max_val - min_val;
        if (range < 1e-10f) range = 1.0f;
        for (int i = 1; i < pop_size - 1; ++i) {
            pop[indices[i]].crowding_distance +=
                (pop[indices[i + 1]].objective_values[o] -
                 pop[indices[i - 1]].objective_values[o]) / range;
        }
    }
}

/* SBX 交叉 */
static void pde_sbx_crossover(const DesignParameter* p1, const DesignParameter* p2,
                                DesignParameter* c1, DesignParameter* c2,
                                size_t param_count, float eta_c)
{
    for (size_t i = 0; i < param_count; ++i) {
        double u = pde_rand_double();
        double beta;
        if (u <= 0.5) {
            beta = pow(2.0 * u, 1.0 / (eta_c + 1.0));
        } else {
            beta = pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta_c + 1.0));
        }
        double v1 = p1[i].current_value;
        double v2 = p2[i].current_value;
        c1[i].current_value = 0.5 * ((1.0 + beta) * v1 + (1.0 - beta) * v2);
        c2[i].current_value = 0.5 * ((1.0 - beta) * v1 + (1.0 + beta) * v2);
        /* 边界修剪 */
        if (c1[i].current_value < p1[i].min_value) c1[i].current_value = p1[i].min_value;
        if (c1[i].current_value > p1[i].max_value) c1[i].current_value = p1[i].max_value;
        if (c2[i].current_value < p2[i].min_value) c2[i].current_value = p2[i].min_value;
        if (c2[i].current_value > p2[i].max_value) c2[i].current_value = p2[i].max_value;
    }
}

/* 多项式变异 */
static void pde_polynomial_mutation(DesignParameter* child, size_t param_count, float eta_m)
{
    for (size_t i = 0; i < param_count; ++i) {
        if (pde_rand_double() < 1.0 / param_count) {
            double u = pde_rand_double();
            double delta;
            if (u < 0.5) {
                delta = pow(2.0 * u, 1.0 / (eta_m + 1.0)) - 1.0;
            } else {
                delta = 1.0 - pow(2.0 * (1.0 - u), 1.0 / (eta_m + 1.0));
            }
            double range = child[i].max_value - child[i].min_value;
            child[i].current_value += delta * range;
            if (child[i].current_value < child[i].min_value)
                child[i].current_value = child[i].min_value;
            if (child[i].current_value > child[i].max_value)
                child[i].current_value = child[i].max_value;
        }
    }
}

/* ============================================================================
 * A09.5.2 多目标优化器实现
 * ============================================================================ */

PdeMultiObjectiveOptimizer* pde_moo_create(void)
{
    PdeMultiObjectiveOptimizer* opt = (PdeMultiObjectiveOptimizer*)
        malloc(sizeof(PdeMultiObjectiveOptimizer));
    if (!opt) return NULL;
    memset(opt, 0, sizeof(PdeMultiObjectiveOptimizer));
    opt->initialized = 1;
    
    return opt;
}

void pde_moo_destroy(PdeMultiObjectiveOptimizer* optimizer)
{
    free(optimizer);
}

int pde_moo_add_objective(PdeMultiObjectiveOptimizer* optimizer,
                           PdeObjectiveType type, const char* name,
                           int maximize, float weight)
{
    if (!optimizer || optimizer->objective_count >= PDE_MAX_OBJECTIVES) return -1;
    PdeObjective* obj = &optimizer->objectives[optimizer->objective_count];
    obj->obj_id = optimizer->objective_count + 1;
    obj->type = type;
    pde_strcpy(obj->name, sizeof(obj->name), name ? name : "未命名目标");
    obj->maximize = maximize;
    obj->weight = weight;
    obj->target_value = 0;
    obj->current_best = maximize ? -FLT_MAX : FLT_MAX;
    optimizer->objective_count++;
    return 0;
}

int pde_moo_optimize(PdeMultiObjectiveOptimizer* optimizer,
                      const ProductRequirement* requirement,
                      const DesignParameter* design_params,
                      size_t param_count,
                      const ConstraintSpec* constraints,
                      size_t constraint_count,
                      int population_size,
                      int generations,
                      PdeMultiObjectiveResult* result)
{
    if (!optimizer || !requirement || !design_params || !result) return -1;
    if (population_size <= 0 || population_size > PDE_MAX_PARETO_SOLUTIONS)
        population_size = 100;
    if (generations <= 0) generations = 50;
    (void)constraints;
    (void)constraint_count;

    memset(result, 0, sizeof(PdeMultiObjectiveResult));
    for (int i = 0; i < optimizer->objective_count; ++i)
        result->objectives[i] = optimizer->objectives[i];
    result->objective_count = optimizer->objective_count;

    /* 初始化种群 */
    PdeParetoSolution pop[PDE_MAX_PARETO_SOLUTIONS];
    for (int i = 0; i < population_size; ++i) {
        pop[i].solution_id = i + 1;
        pop[i].parameters = pde_copy_params(design_params, param_count);
        pop[i].param_count = param_count;
        pop[i].spec = NULL;
        /* 随机初始化参数值 */
        for (size_t j = 0; j < param_count; ++j) {
            double r = pde_rand_double();
            pop[i].parameters[j].current_value =
                design_params[j].min_value + r * (design_params[j].max_value - design_params[j].min_value);
        }
        pop[i].objective_count = optimizer->objective_count;
        pde_evaluate_design(requirement, pop[i].parameters, param_count,
                            pop[i].objective_values, optimizer->objective_count);
    }

    /* 进化循环 */
    for (int gen = 0; gen < generations; ++gen) {
        /* 非支配排序 */
        pde_non_dominated_sort(pop, population_size, optimizer->objectives,
                                optimizer->objective_count);
        pde_crowding_distance(pop, population_size, optimizer->objectives,
                               optimizer->objective_count);

        /* 创建子代 */
        PdeParetoSolution offspring[PDE_MAX_PARETO_SOLUTIONS];
        for (int i = 0; i < population_size; i += 2) {
            /* 锦标赛选择 */
            int p1_idx = pde_rand_int(0, population_size - 1);
            int p2_idx = pde_rand_int(0, population_size - 1);
            int s1 = (pop[p1_idx].rank < pop[p2_idx].rank ||
                     (pop[p1_idx].rank == pop[p2_idx].rank &&
                      pop[p1_idx].crowding_distance > pop[p2_idx].crowding_distance))
                     ? p1_idx : p2_idx;
            int s2;
            do {
                p1_idx = pde_rand_int(0, population_size - 1);
                p2_idx = pde_rand_int(0, population_size - 1);
                s2 = (pop[p1_idx].rank < pop[p2_idx].rank ||
                     (pop[p1_idx].rank == pop[p2_idx].rank &&
                      pop[p1_idx].crowding_distance > pop[p2_idx].crowding_distance))
                     ? p1_idx : p2_idx;
            } while (s2 == s1);

            /* SBX 交叉 + 多项式变异 */
            offspring[i].parameters = pde_copy_params(design_params, param_count);
            offspring[i].param_count = param_count;
            offspring[i+1].parameters = pde_copy_params(design_params, param_count);
            offspring[i+1].param_count = param_count;

            if (pde_rand_double() < 0.9) {
                pde_sbx_crossover(pop[s1].parameters, pop[s2].parameters,
                                   offspring[i].parameters, offspring[i+1].parameters,
                                   param_count, 15.0f);
            } else {
                for (size_t j = 0; j < param_count; ++j) {
                    offspring[i].parameters[j].current_value = pop[s1].parameters[j].current_value;
                    offspring[i+1].parameters[j].current_value = pop[s2].parameters[j].current_value;
                }
            }
            pde_polynomial_mutation(offspring[i].parameters, param_count, 20.0f);
            pde_polynomial_mutation(offspring[i+1].parameters, param_count, 20.0f);

            offspring[i].solution_id = population_size + i + 1;
            offspring[i+1].solution_id = population_size + i + 2;
            offspring[i].spec = NULL;
            offspring[i+1].spec = NULL;
            offspring[i].objective_count = optimizer->objective_count;
            offspring[i+1].objective_count = optimizer->objective_count;

            pde_evaluate_design(requirement, offspring[i].parameters, param_count,
                                offspring[i].objective_values, optimizer->objective_count);
            pde_evaluate_design(requirement, offspring[i+1].parameters, param_count,
                                offspring[i+1].objective_values, optimizer->objective_count);
        }

        /* 精英保留：合并父代和子代 */
        PdeParetoSolution combined[PDE_MAX_PARETO_SOLUTIONS * 2];
        int combined_size = population_size * 2;
        for (int i = 0; i < population_size; ++i) {
            combined[i] = pop[i];
            combined[i].parameters = pde_copy_params(pop[i].parameters, param_count);
        }
        for (int i = 0; i < population_size; ++i) {
            combined[population_size + i] = offspring[i];
            combined[population_size + i].parameters =
                pde_copy_params(offspring[i].parameters, param_count);
        }

        /* 非支配排序 + 拥挤度选择前N个 */
        pde_non_dominated_sort(combined, combined_size, optimizer->objectives,
                                optimizer->objective_count);
        pde_crowding_distance(combined, combined_size, optimizer->objectives,
                               optimizer->objective_count);

        /* 按 rank 和 crowding distance 排序 */
        for (int i = 0; i < combined_size - 1; ++i) {
            for (int j = i + 1; j < combined_size; ++j) {
                if (combined[i].rank > combined[j].rank ||
                    (combined[i].rank == combined[j].rank &&
                     combined[i].crowding_distance < combined[j].crowding_distance)) {
                    PdeParetoSolution tmp = combined[i];
                    combined[i] = combined[j];
                    combined[j] = tmp;
                }
            }
        }

        /* 释放旧种群参数 */
        for (int i = 0; i < population_size; ++i)
            pde_free_params(pop[i].parameters, pop[i].param_count);

        /* 选择前N个进入下一代 */
        for (int i = 0; i < population_size; ++i) {
            pop[i] = combined[i];
            pop[i].parameters = pde_copy_params(combined[i].parameters, param_count);
        }

        /* 释放临时种群 */
        for (int i = 0; i < combined_size; ++i)
            pde_free_params(combined[i].parameters, param_count);

        result->total_evaluations += population_size;
    }

    /* 提取最终 Pareto 前沿 */
    pde_non_dominated_sort(pop, population_size, optimizer->objectives,
                            optimizer->objective_count);
    result->pareto_count = 0;
    for (int i = 0; i < population_size && result->pareto_count < PDE_MAX_PARETO_SOLUTIONS; ++i) {
        if (pop[i].rank == 0) {
            result->pareto_front[result->pareto_count] = pop[i];
            result->pareto_front[result->pareto_count].parameters =
                pde_copy_params(pop[i].parameters, param_count);
            result->pareto_count++;
        }
    }

    /* 生成摘要 */
    snprintf(result->summary, sizeof(result->summary),
             "多目标优化完成: %d 个目标, %d 代, %d 次评估, Pareto 前沿 %d 个解",
             optimizer->objective_count, generations,
             result->total_evaluations, result->pareto_count);

    /* 释放剩余种群 */
    for (int i = 0; i < population_size; ++i)
        pde_free_params(pop[i].parameters, pop[i].param_count);

    return 0;
}

int pde_moo_select_compromise(const PdeMultiObjectiveResult* result,
                               DesignParameter* best_params,
                               size_t* param_count)
{
    if (!result || !best_params || !param_count || result->pareto_count == 0)
        return -1;

    /* TOPSIS 方法 */
    int n = result->pareto_count;
    int m = result->objective_count;

    /* 构建目标矩阵 */
    float matrix[PDE_MAX_PARETO_SOLUTIONS][PDE_MAX_OBJECTIVES];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = result->pareto_front[i].objective_values[j];
        }
    }

    /* 归一化 */
    float norm_matrix[PDE_MAX_PARETO_SOLUTIONS][PDE_MAX_OBJECTIVES];
    for (int j = 0; j < m; ++j) {
        float sum_sq = 0;
        for (int i = 0; i < n; ++i)
            sum_sq += matrix[i][j] * matrix[i][j];
        if (sum_sq < 1e-10f) sum_sq = 1.0f;
        float norm = sqrtf(sum_sq);
        for (int i = 0; i < n; ++i)
            norm_matrix[i][j] = matrix[i][j] / norm;
    }

    /* 加权归一化 */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            norm_matrix[i][j] *= result->objectives[j].weight;
        }
    }

    /* 理想解和负理想解 */
    float ideal[PDE_MAX_OBJECTIVES];
    float nadir[PDE_MAX_OBJECTIVES];
    for (int j = 0; j < m; ++j) {
        ideal[j] = result->objectives[j].maximize ? -FLT_MAX : FLT_MAX;
        nadir[j] = result->objectives[j].maximize ? FLT_MAX : -FLT_MAX;
        for (int i = 0; i < n; ++i) {
            if (result->objectives[j].maximize) {
                if (norm_matrix[i][j] > ideal[j]) ideal[j] = norm_matrix[i][j];
                if (norm_matrix[i][j] < nadir[j]) nadir[j] = norm_matrix[i][j];
            } else {
                if (norm_matrix[i][j] < ideal[j]) ideal[j] = norm_matrix[i][j];
                if (norm_matrix[i][j] > nadir[j]) nadir[j] = norm_matrix[i][j];
            }
        }
    }

    /* 计算每个解到理想解和负理想解的距离 */
    float dist_to_ideal[PDE_MAX_PARETO_SOLUTIONS];
    float dist_to_nadir[PDE_MAX_PARETO_SOLUTIONS];
    for (int i = 0; i < n; ++i) {
        dist_to_ideal[i] = 0;
        dist_to_nadir[i] = 0;
        for (int j = 0; j < m; ++j) {
            float d1 = norm_matrix[i][j] - ideal[j];
            float d2 = norm_matrix[i][j] - nadir[j];
            dist_to_ideal[i] += d1 * d1;
            dist_to_nadir[i] += d2 * d2;
        }
        dist_to_ideal[i] = sqrtf(dist_to_ideal[i]);
        dist_to_nadir[i] = sqrtf(dist_to_nadir[i]);
    }

    /* 选择贴近度最高的解 */
    int best_idx = 0;
    float best_ratio = -1;
    for (int i = 0; i < n; ++i) {
        float denom = dist_to_ideal[i] + dist_to_nadir[i];
        float ratio = denom > 1e-10f ? dist_to_nadir[i] / denom : 0.5f;
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best_idx = i;
        }
    }

    /* 复制最优参数 */
    const DesignParameter* src = result->pareto_front[best_idx].parameters;
    size_t count = result->pareto_front[best_idx].param_count;
    for (size_t i = 0; i < count && i < *param_count; ++i)
        best_params[i] = src[i];
    *param_count = count;

    return best_idx;
}

/* ============================================================================
 * A09.5.2 设计空间探索实现
 * ============================================================================ */

int pde_explore_design_space(const ProductRequirement* requirement,
                              const DesignParameter* params,
                              size_t param_count,
                              const ConstraintSpec* constraints,
                              size_t constraint_count,
                              PdeSampleMethod method,
                              int sample_count,
                              DesignVariant* results,
                              int* result_count)
{
    if (!requirement || !params || !results || !result_count) return -1;
    if (sample_count <= 0 || sample_count > PDE_MAX_SAMPLES) sample_count = 100;
    (void)constraints;
    (void)constraint_count;

    int count = 0;
    DesignParameter* temp_params = pde_copy_params(params, param_count);

    for (int s = 0; s < sample_count && count < PDE_MAX_SAMPLES; ++s) {
        if (method == PDE_SAMPLE_GRID) {
            /* 网格采样：每个维度均匀取值 */
            int grid_size = (int)pow((double)sample_count, 1.0 / param_count);
            if (grid_size < 2) grid_size = 2;
            int idx = s;
            for (size_t j = 0; j < param_count; ++j) {
                int div = 1;
                for (size_t k = j + 1; k < param_count; ++k) div *= grid_size;
                int pos = (idx / div) % grid_size;
                double step = (params[j].max_value - params[j].min_value) / (grid_size - 1);
                temp_params[j].current_value = params[j].min_value + pos * step;
            }
        } else if (method == PDE_SAMPLE_RANDOM) {
            for (size_t j = 0; j < param_count; ++j) {
                double r = pde_rand_double();
                temp_params[j].current_value =
                    params[j].min_value + r * (params[j].max_value - params[j].min_value);
            }
        } else if (method == PDE_SAMPLE_LATIN_HYPERCUBE) {
            /* 拉丁超立方采样 */
            for (size_t j = 0; j < param_count; ++j) {
                double r = (s + pde_rand_double()) / sample_count;
                temp_params[j].current_value =
                    params[j].min_value + r * (params[j].max_value - params[j].min_value);
            }
        } else {
            /* Sobol / 默认随机 */
            for (size_t j = 0; j < param_count; ++j) {
                double r = pde_rand_double();
                temp_params[j].current_value =
                    params[j].min_value + r * (params[j].max_value - params[j].min_value);
            }
        }

        /* 评估 */
        results[count].parameters = pde_copy_params(temp_params, param_count);
        results[count].param_count = param_count;
        results[count].spec = NULL;
        results[count].evaluation = NULL;

        float obj_vals[PDE_MAX_OBJECTIVES];
        pde_evaluate_design(requirement, temp_params, param_count, obj_vals, 3);
        results[count].composite_score =
            pde_compute_composite_score(requirement, obj_vals, 3);
        count++;
    }

    pde_free_params(temp_params, param_count);
    *result_count = count;
    return 0;
}

int pde_sensitivity_analysis(const ProductRequirement* requirement,
                              const DesignParameter* params,
                              size_t param_count,
                              const ConstraintSpec* constraints,
                              size_t constraint_count,
                              int trajectories,
                              PdeSensitivityResult* results,
                              int* result_count)
{
    if (!requirement || !params || !results || !result_count) return -1;
    if (param_count == 0) return -1;
    if (trajectories <= 0) trajectories = 10;
    (void)constraints;
    (void)constraint_count;

    /* Morris 法 elementary effects */
    double* ee_sum = (double*)safe_calloc(param_count, sizeof(double));
    double* ee_sum_sq = (double*)safe_calloc(param_count, sizeof(double));
    int* ee_count = (int*)safe_calloc(param_count, sizeof(int));

    if (!ee_sum || !ee_sum_sq || !ee_count) {
        safe_free((void**)&ee_sum); safe_free((void**)&ee_sum_sq); safe_free((void**)&ee_count);
        return -1;
    }

    DesignParameter* base = pde_copy_params(params, param_count);
    DesignParameter* perturbed = pde_copy_params(params, param_count);

    for (int t = 0; t < trajectories; ++t) {
        /* 随机基点和扰动方向 */
        for (size_t j = 0; j < param_count; ++j) {
            base[j].current_value = params[j].min_value +
                pde_rand_double() * (params[j].max_value - params[j].min_value);
        }

        float base_vals[PDE_MAX_OBJECTIVES];
        pde_evaluate_design(requirement, base, param_count, base_vals, 1);
        float base_score = base_vals[0];

        for (size_t j = 0; j < param_count; ++j) {
            for (size_t k = 0; k < param_count; ++k)
                perturbed[k] = base[k];

            double delta = (params[j].max_value - params[j].min_value) * 0.05;
            if (delta < 1e-10) delta = 0.1;
            perturbed[j].current_value += delta;
            if (perturbed[j].current_value > params[j].max_value)
                perturbed[j].current_value = params[j].max_value;

            float pert_vals[PDE_MAX_OBJECTIVES];
            pde_evaluate_design(requirement, perturbed, param_count, pert_vals, 1);
            float pert_score = pert_vals[0];

            double ee = (pert_score - base_score) / delta;
            ee_sum[j] += ee;
            ee_sum_sq[j] += ee * ee;
            ee_count[j]++;
        }
    }

    /* 计算敏感性指数 */
    int count = 0;
    for (size_t j = 0; j < param_count && count < (int)param_count; ++j) {
        if (ee_count[j] > 0) {
            pde_strcpy(results[count].param_name, sizeof(results[count].param_name),
                       params[j].name ? params[j].name : "unknown");
            double mean = ee_sum[j] / ee_count[j];
            double variance = ee_sum_sq[j] / ee_count[j] - mean * mean;
            if (variance < 0) variance = 0;
            results[count].sensitivity_index = (float)fabs(mean);
            results[count].main_effect = (float)mean;
            results[count].interaction_effect = (float)sqrt(variance);
            results[count].rank = count + 1;
            count++;
        }
    }

    /* 按敏感性排序 */
    for (int i = 0; i < count - 1; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (results[i].sensitivity_index < results[j].sensitivity_index) {
                PdeSensitivityResult tmp = results[i];
                results[i] = results[j];
                results[j] = tmp;
            }
        }
    }
    for (int i = 0; i < count; ++i) results[i].rank = i + 1;

    safe_free((void**)&ee_sum); safe_free((void**)&ee_sum_sq); safe_free((void**)&ee_count);
    pde_free_params(base, param_count);
    pde_free_params(perturbed, param_count);

    *result_count = count;
    return 0;
}

int pde_local_search(const ProductRequirement* requirement,
                      DesignParameter* params,
                      size_t param_count,
                      const ConstraintSpec* constraints,
                      size_t constraint_count,
                      int max_iterations,
                      float step_size,
                      DesignEvaluation* final_evaluation)
{
    if (!requirement || !params || param_count == 0) return -1;
    if (max_iterations <= 0) max_iterations = 100;
    if (step_size <= 0) step_size = 0.05f;
    (void)constraints;
    (void)constraint_count;
    (void)final_evaluation;

    DesignParameter* current = pde_copy_params(params, param_count);
    DesignParameter* candidate = pde_copy_params(params, param_count);

    float current_vals[PDE_MAX_OBJECTIVES];
    pde_evaluate_design(requirement, current, param_count, current_vals, 3);
    float current_score = pde_compute_composite_score(requirement, current_vals, 3);

    int improved = 1;
    for (int iter = 0; iter < max_iterations && improved; ++iter) {
        improved = 0;
        for (size_t j = 0; j < param_count; ++j) {
            for (int dir = -1; dir <= 1; dir += 2) {
                for (size_t k = 0; k < param_count; ++k)
                    candidate[k] = current[k];

                double delta = (params[j].max_value - params[j].min_value) * step_size;
                candidate[j].current_value += dir * delta;
                if (candidate[j].current_value < params[j].min_value)
                    candidate[j].current_value = params[j].min_value;
                if (candidate[j].current_value > params[j].max_value)
                    candidate[j].current_value = params[j].max_value;

                float cand_vals[PDE_MAX_OBJECTIVES];
                pde_evaluate_design(requirement, candidate, param_count, cand_vals, 3);
                float cand_score = pde_compute_composite_score(requirement, cand_vals, 3);

                if (cand_score > current_score) {
                    current_score = cand_score;
                    for (size_t k = 0; k < param_count; ++k)
                        current[k] = candidate[k];
                    improved = 1;
                }
            }
        }
    }

    /* 输出最优参数 */
    for (size_t j = 0; j < param_count; ++j)
        params[j] = current[j];

    pde_free_params(current, param_count);
    pde_free_params(candidate, param_count);
    return 0;
}
