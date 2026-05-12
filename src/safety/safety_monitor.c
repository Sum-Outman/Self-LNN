/**
 * @file safety_monitor.c
 * @brief AGI安全监控系统完整实现
 */

#include "selflnn/safety/safety_monitor.h"
#include "selflnn/safety/emergency_stop.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* 最大事件数 */
#define SAFETY_MAX_EVENTS 1000
#define SAFETY_MAX_RULES 128
#define SAFETY_MAX_AUDIT_LOG 5000

/* ========== P1: SafetyMonitor锁宏 ========== */
#define SAFETY_LOCK(m)    mutex_lock((m)->lock)
#define SAFETY_UNLOCK(m)  mutex_unlock((m)->lock)

struct SafetyMonitor {
    SafetyLevel current_level;
    int emergency_stop_active;
    int soft_stop_active;

    ResourceLimits resource_limits;
    PhysicalBoundaries physical_boundaries;

    SafetyRule* rules;
    int rule_count;
    int rule_capacity;

    SafetyEvent* events;
    int event_count;
    int event_capacity;

    AuditLogEntry* audit_log;
    int audit_count;
    int audit_capacity;

    SafetyStats stats;
    time_t start_time;

    /* 关联的紧急停止系统（安全事件触发紧急停止） */
    EmergencyStopSystem* emergency_system;

    ///* ---- 动态自适应调整字段 ---- */
    int violation_history[60];      /* 最近60分钟每分钟违规数循环缓冲区 */
    int violation_index;            /* 当前写入位置 */
    int violation_window_count;     /* 有效窗口条目数 */
    time_t last_dynamic_adjust_time;/* 上次自适应调整时间 */
    float last_cpu_usage;           /* 上次CPU使用率（趋势跟踪） */
    float last_memory_usage;        /* 上次内存使用率（趋势跟踪） */

    /* K-022: 隐私过滤系统 */
    int privacy_filter_level;       /* 0=不过滤 1=日志 2=脱敏 3=隔离 */
    int privacy_filter_data_types;  /* 位掩码: bit0=图像 bit1=音频 bit2=文本 bit3=传感器 */
    int privacy_filter_enabled;     /* 是否启用 */

    /* K-022: 行为约束规则 */
    struct { char name[128]; char condition[256]; int action; int enabled; } 
        behavioral_constraints[32];
    int behavioral_constraint_count;

    /* 线程安全锁 */
    MutexHandle lock;
    
};

SafetyMonitor* safety_monitor_create(void) {
    SafetyMonitor* monitor = (SafetyMonitor*)safe_calloc(1, sizeof(SafetyMonitor));
    if (!monitor) return NULL;

    monitor->lock = mutex_create();
    if (!monitor->lock) {
        safe_free((void**)&monitor);
        return NULL;
    }

    monitor->current_level = SAFETY_LEVEL_NORMAL;
    monitor->emergency_stop_active = 0;
    monitor->soft_stop_active = 0;

    /* 默认资源限制 */
    monitor->resource_limits.cpu_usage_max = 90.0f;
    monitor->resource_limits.gpu_usage_max = 95.0f;
    monitor->resource_limits.memory_usage_max = 85.0f;
    monitor->resource_limits.disk_usage_max = 90.0f;
    monitor->resource_limits.network_bandwidth_max = 1000.0f;
    monitor->resource_limits.max_threads = 256;
    monitor->resource_limits.max_connections = 1000;
    monitor->resource_limits.max_power_watts = 500.0f;

    /* 默认物理边界 */
    monitor->physical_boundaries.max_velocity = 5.0f;
    monitor->physical_boundaries.max_acceleration = 10.0f;
    monitor->physical_boundaries.max_torque = 100.0f;
    monitor->physical_boundaries.safety_zone_radius = 1.0f;
    monitor->physical_boundaries.collision_distance_min = 0.1f;
    monitor->physical_boundaries.joint_count = 16;
    for (int i = 0; i < 16; i++) {
        monitor->physical_boundaries.min_joint_angle[i] = -3.14f;
        monitor->physical_boundaries.max_joint_angle[i] = 3.14f;
    }

    monitor->rules = (SafetyRule*)safe_calloc(SAFETY_MAX_RULES, sizeof(SafetyRule));
    monitor->rule_count = 0;
    monitor->rule_capacity = SAFETY_MAX_RULES;

    monitor->events = (SafetyEvent*)safe_calloc(SAFETY_MAX_EVENTS, sizeof(SafetyEvent));
    monitor->event_count = 0;
    monitor->event_capacity = SAFETY_MAX_EVENTS;

    monitor->audit_log = (AuditLogEntry*)safe_calloc(SAFETY_MAX_AUDIT_LOG, sizeof(AuditLogEntry));
    monitor->audit_count = 0;
    monitor->audit_capacity = SAFETY_MAX_AUDIT_LOG;

    memset(&monitor->stats, 0, sizeof(SafetyStats));
    monitor->stats.current_safety_score = 1.0f;
    monitor->stats.average_safety_score = 1.0f;
    monitor->start_time = time(NULL);

    /* 初始化动态自适应调整字段 */
    memset(monitor->violation_history, 0, sizeof(monitor->violation_history));
    monitor->violation_index = 0;
    monitor->violation_window_count = 0;
    monitor->last_dynamic_adjust_time = time(NULL);
    monitor->last_cpu_usage = 0.0f;
    monitor->last_memory_usage = 0.0f;

    /* 添加默认安全规则 */
    SafetyRule default_rules[] = {
        {"CPU过载保护", "CPU使用率超过阈值时触发警告", SAFETY_EVENT_RESOURCE_OVERUSE, 0.9f, 0, 3, 1, 1, 1},
        {"内存泄漏检测", "内存持续增长且未释放", SAFETY_EVENT_MEMORY_LEAK, 0.85f, 0, 2, 1, 1, 2},
        {"死循环检测", "检测到任务执行时间异常增长", SAFETY_EVENT_LOOP_DETECTED, 0.0f, 0, 1, 1, 1, 3},
        {"物理碰撞预防", "检测到即将发生的物理碰撞", SAFETY_EVENT_PHYSICAL_VIOLATION, 0.8f, 0, 1, 1, 1, 3},
        {"传感器失效检测", "传感器数据长时间无变化或异常", SAFETY_EVENT_SENSOR_INVALID, 0.0f, 0, 2, 1, 1, 2},
        {"网络异常监控", "网络连接异常断开或数据异常", SAFETY_EVENT_NETWORK_ANOMALY, 0.0f, 0, 3, 1, 0, 2},
        {"行为异常检测", "输出模式与预期偏差过大", SAFETY_EVENT_BEHAVIOR_ANOMALY, 0.7f, 0, 3, 1, 1, 2},
        {"决策冲突检测", "多个决策同时冲突", SAFETY_EVENT_DECISION_CONFLICT, 0.0f, 0, 2, 1, 0, 1},
    };

    int rule_count = sizeof(default_rules) / sizeof(default_rules[0]);
    for (int i = 0; i < rule_count; i++) {
        memcpy(&monitor->rules[monitor->rule_count], &default_rules[i], sizeof(SafetyRule));
        monitor->rule_count++;
    }

    return monitor;
}

void safety_monitor_free(SafetyMonitor* monitor) {
    if (!monitor) return;
    SAFETY_LOCK(monitor);
    safe_free((void**)&monitor->rules);
    safe_free((void**)&monitor->events);
    safe_free((void**)&monitor->audit_log);
    SAFETY_UNLOCK(monitor);
    mutex_destroy(monitor->lock);
    safe_free((void**)&monitor);
}

int safety_set_resource_limits(SafetyMonitor* monitor, const ResourceLimits* limits) {
    if (!monitor || !limits) return -1;
    SAFETY_LOCK(monitor);
    memcpy(&monitor->resource_limits, limits, sizeof(ResourceLimits));
    SAFETY_UNLOCK(monitor);
    return 0;
}

int safety_set_physical_boundaries(SafetyMonitor* monitor, const PhysicalBoundaries* boundaries) {
    if (!monitor || !boundaries) return -1;
    SAFETY_LOCK(monitor);
    memcpy(&monitor->physical_boundaries, boundaries, sizeof(PhysicalBoundaries));
    SAFETY_UNLOCK(monitor);
    return 0;
}

int safety_add_rule(SafetyMonitor* monitor, const SafetyRule* rule) {
    if (!monitor || !rule) return -1;
    SAFETY_LOCK(monitor);
    if (monitor->rule_count >= monitor->rule_capacity) {
        SAFETY_UNLOCK(monitor);
        return -1;
    }
    memcpy(&monitor->rules[monitor->rule_count], rule, sizeof(SafetyRule));
    monitor->rule_count++;
    SAFETY_UNLOCK(monitor);
    return 0;
}

int safety_report_event(SafetyMonitor* monitor, const SafetyEvent* event) {
    if (!monitor || !event) return -1;
    SAFETY_LOCK(monitor);

    /* 存储事件 */
    if (monitor->event_count < monitor->event_capacity) {
        memcpy(&monitor->events[monitor->event_count], event, sizeof(SafetyEvent));
        monitor->events[monitor->event_count].timestamp = time(NULL);
        monitor->event_count++;
    } else {
        /* 循环覆盖旧事件 */
        int oldest = 0;
        time_t oldest_time = monitor->events[0].timestamp;
        for (int i = 1; i < monitor->event_count; i++) {
            if (monitor->events[i].timestamp < oldest_time) {
                oldest_time = monitor->events[i].timestamp;
                oldest = i;
            }
        }
        memcpy(&monitor->events[oldest], event, sizeof(SafetyEvent));
        monitor->events[oldest].timestamp = time(NULL);
    }

    /* 查找匹配的安全规则 */
    for (int i = 0; i < monitor->rule_count; i++) {
        if (monitor->rules[i].target_type == event->type && monitor->rules[i].enabled) {
            monitor->rules[i].violation_count++;

            /* 自动处理 */
            if (monitor->rules[i].auto_action && 
                monitor->rules[i].violation_count >= monitor->rules[i].violation_limit) {
                switch (monitor->rules[i].action_type) {
                    case 1: /* 警告 */
                        monitor->current_level = SAFETY_LEVEL_WARNING;
                        break;
                    case 2: /* 限制 */
                        monitor->current_level = SAFETY_LEVEL_ELEVATED;
                        break;
                    case 3: /* 紧急停止 */
                        monitor->current_level = SAFETY_LEVEL_EMERGENCY;
                        safety_emergency_stop(monitor);
                        break;
                }
            }
        }
    }

    /* 更新统计 */
    monitor->stats.total_events++;
    switch (event->severity) {
        case SAFETY_LEVEL_WARNING: monitor->stats.warnings++; break;
        case SAFETY_LEVEL_ELEVATED: monitor->stats.elevated++; break;
        case SAFETY_LEVEL_CRITICAL: monitor->stats.critical++; break;
        case SAFETY_LEVEL_EMERGENCY: monitor->stats.emergencies++; break;
        default: break;
    }
    if (event->auto_resolved) monitor->stats.auto_resolved++;

    /* 安全评分衰减 */
    float severity_penalty = 0.0f;
    switch (event->severity) {
        case SAFETY_LEVEL_WARNING: severity_penalty = 0.02f; break;
        case SAFETY_LEVEL_ELEVATED: severity_penalty = 0.05f; break;
        case SAFETY_LEVEL_CRITICAL: severity_penalty = 0.15f; break;
        case SAFETY_LEVEL_EMERGENCY: severity_penalty = 0.30f; break;
        default: break;
    }
    monitor->stats.current_safety_score -= severity_penalty;
    if (monitor->stats.current_safety_score < 0.0f) monitor->stats.current_safety_score = 0.0f;
    monitor->stats.last_incident_time = time(NULL);

    /* 审计日志 */
    if (monitor->audit_count < monitor->audit_capacity) {
        AuditLogEntry* entry = &monitor->audit_log[monitor->audit_count++];
        entry->timestamp = time(NULL);
        entry->event_type = event->type;
        entry->level = event->severity;
        snprintf(entry->action, sizeof(entry->action), "安全事件: %s", event->description);
        snprintf(entry->decision, sizeof(entry->decision), "自动处理: 级别=%d", event->severity);
        snprintf(entry->result, sizeof(entry->result), "%s", event->handled ? "已处理" : "待处理");
        entry->confidence = monitor->stats.current_safety_score;
    }

    /* 记录违规到历史窗口（用于动态阈值调整） */
    time_t now_t = time(NULL);
    int minute_idx = (int)((now_t - monitor->start_time) / 60) % 60;
    if (minute_idx != monitor->violation_index) {
        monitor->violation_index = minute_idx;
        monitor->violation_history[minute_idx] = 0;
        if (monitor->violation_window_count < 60) monitor->violation_window_count++;
    }
    monitor->violation_history[minute_idx]++;

    SAFETY_UNLOCK(monitor);
    return 0;
}

SafetyLevel safety_check_status(SafetyMonitor* monitor) {
    if (!monitor) return SAFETY_LEVEL_EMERGENCY;
    SAFETY_LOCK(monitor);

    /* 恢复：随时间恢复安全评分 */
    time_t now = time(NULL);
    time_t elapsed = now - monitor->stats.last_incident_time;
    if (elapsed > 60 && monitor->stats.current_safety_score < 1.0f) {
        float recovery = (float)(elapsed - 60) / 3600.0f * 0.01f;
        if (recovery > 0.1f) recovery = 0.1f;
        monitor->stats.current_safety_score += recovery;
        if (monitor->stats.current_safety_score > 1.0f) monitor->stats.current_safety_score = 1.0f;
    }

    SafetyLevel level;
    if (monitor->emergency_stop_active) level = SAFETY_LEVEL_EMERGENCY;
    else if (monitor->soft_stop_active) level = SAFETY_LEVEL_CRITICAL;
    else if (monitor->stats.current_safety_score < 0.3f) level = SAFETY_LEVEL_CRITICAL;
    else if (monitor->stats.current_safety_score < 0.6f) level = SAFETY_LEVEL_ELEVATED;
    else if (monitor->stats.current_safety_score < 0.85f) level = SAFETY_LEVEL_WARNING;
    else level = SAFETY_LEVEL_NORMAL;

    SAFETY_UNLOCK(monitor);
    return level;
}

int safety_approve_action(SafetyMonitor* monitor, const char* action_type, const void* params) {
    if (!monitor || !action_type) return -1;
    (void)params;

    SAFETY_LOCK(monitor);
    if (monitor->emergency_stop_active) {
        SAFETY_UNLOCK(monitor);
        return -1;
    }
    int emergency_stop = monitor->emergency_stop_active;
    float safety_score = monitor->stats.current_safety_score;
    int audit_avail = (monitor->audit_count < monitor->audit_capacity);
    SAFETY_UNLOCK(monitor);

    SafetyLevel level = safety_check_status(monitor);
    if (emergency_stop || level >= SAFETY_LEVEL_CRITICAL) return -1;

    if (audit_avail) {
        SAFETY_LOCK(monitor);
        if (monitor->audit_count < monitor->audit_capacity) {
            AuditLogEntry* entry = &monitor->audit_log[monitor->audit_count++];
            entry->timestamp = time(NULL);
            entry->event_type = SAFETY_EVENT_BOUNDARY_CROSSING;
            entry->level = level;
            snprintf(entry->action, sizeof(entry->action), "动作审批: %s", action_type);
            snprintf(entry->decision, sizeof(entry->decision), level == SAFETY_LEVEL_NORMAL ? "批准" : "审查");
            snprintf(entry->result, sizeof(entry->result), "%s", level == SAFETY_LEVEL_NORMAL ? "通过" : "需审查");
            entry->confidence = safety_score;
        }
        SAFETY_UNLOCK(monitor);
    }

    return (level == SAFETY_LEVEL_NORMAL) ? 0 : -1;
}

int safety_monitor_set_emergency_stop(SafetyMonitor* monitor, EmergencyStopSystem* emergency_system) {
    if (!monitor) return -1;
    SAFETY_LOCK(monitor);
    monitor->emergency_system = emergency_system;
    SAFETY_UNLOCK(monitor);
    return 0;
}

int safety_emergency_stop(SafetyMonitor* monitor) {
    if (!monitor) return -1;
    SAFETY_LOCK(monitor);
    monitor->emergency_stop_active = 1;
    monitor->current_level = SAFETY_LEVEL_EMERGENCY;
    monitor->stats.current_safety_score = 0.0f;
    EmergencyStopSystem* es = monitor->emergency_system;
    SAFETY_UNLOCK(monitor);

    /* 同时触发紧急停止系统 */
    if (es) {
        emergency_stop_system_trigger(es, EMERGENCY_SOURCE_SAFETY_MONITOR,
                                      EMERGENCY_LEVEL_HARD_STOP, NULL, 0);
    }

    SafetyEvent stop_event;
    memset(&stop_event, 0, sizeof(SafetyEvent));
    stop_event.type = SAFETY_EVENT_BOUNDARY_CROSSING;
    stop_event.severity = SAFETY_LEVEL_EMERGENCY;
    snprintf(stop_event.description, sizeof(stop_event.description), "紧急停止已激活");
    snprintf(stop_event.source, sizeof(stop_event.source), "safety_monitor");
    stop_event.handled = 1;
    safety_report_event(monitor, &stop_event);

    return 0;
}

int safety_soft_stop(SafetyMonitor* monitor) {
    if (!monitor) return -1;
    SAFETY_LOCK(monitor);
    monitor->soft_stop_active = 1;
    monitor->current_level = SAFETY_LEVEL_CRITICAL;
    SAFETY_UNLOCK(monitor);

    SafetyEvent stop_event;
    memset(&stop_event, 0, sizeof(SafetyEvent));
    stop_event.type = SAFETY_EVENT_BOUNDARY_CROSSING;
    stop_event.severity = SAFETY_LEVEL_CRITICAL;
    snprintf(stop_event.description, sizeof(stop_event.description), "软停止已激活");
    snprintf(stop_event.source, sizeof(stop_event.source), "safety_monitor");
    stop_event.handled = 1;
    safety_report_event(monitor, &stop_event);

    return 0;
}

int safety_get_stats(const SafetyMonitor* monitor, SafetyStats* stats) {
    if (!monitor || !stats) return -1;

    SafetyMonitor* m = (SafetyMonitor*)monitor;
    SAFETY_LOCK(m);
    SafetyStats* s = &m->stats;
    s->uptime_without_incident = (size_t)(time(NULL) - m->stats.last_incident_time);
    if (s->total_events > 0) {
        s->average_safety_score = 1.0f - (float)(s->warnings * 0.02f + s->elevated * 0.05f +
            s->critical * 0.1f + s->emergencies * 0.3f) / (float)s->total_events;
    }
    s->pending = s->total_events - s->auto_resolved - s->manual_resolved;

    memcpy(stats, s, sizeof(SafetyStats));
    SAFETY_UNLOCK(m);
    return 0;
}

int safety_get_audit_log(const SafetyMonitor* monitor, AuditLogEntry* entries, int max_entries) {
    if (!monitor || !entries || max_entries <= 0) return 0;
    SafetyMonitor* m = (SafetyMonitor*)monitor;
    SAFETY_LOCK(m);
    int count = m->audit_count < max_entries ? m->audit_count : max_entries;
    memcpy(entries, m->audit_log, count * sizeof(AuditLogEntry));
    SAFETY_UNLOCK(m);
    return count;
}

int safety_validate_decision(SafetyMonitor* monitor, const char* decision_type,
                            const float* decision_params, int param_count) {
    if (!monitor || !decision_type) return -1;

    SAFETY_LOCK(monitor);
    if (monitor->emergency_stop_active) {
        SAFETY_UNLOCK(monitor);
        return -1;
    }
    SAFETY_UNLOCK(monitor);

    /* 检查决策参数是否在安全范围内 */
    if (decision_params && param_count > 0) {
        for (int i = 0; i < param_count; i++) {
            if (isnan(decision_params[i]) || isinf(decision_params[i])) {
                SafetyEvent event;
                memset(&event, 0, sizeof(SafetyEvent));
                event.type = SAFETY_EVENT_DECISION_CONFLICT;
                event.severity = SAFETY_LEVEL_WARNING;
                snprintf(event.description, sizeof(event.description),
                        "决策参数异常: %s[%d]=%f", decision_type, i, decision_params[i]);
                snprintf(event.source, sizeof(event.source), "decision_validator");
                safety_report_event(monitor, &event);
                return -1;
            }
        }
    }

    return 0;
}

/* 本地资源监控实现 */
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
static float get_cpu_usage_local(void) {
    static ULARGE_INTEGER last_idle = {0}, last_kernel = {0}, last_user = {0};
    FILETIME idle, kernel, user;
    if (!GetSystemTimes(&idle, &kernel, &user)) return 0.0f;
    ULARGE_INTEGER idle_u, kernel_u, user_u;
    idle_u.LowPart = idle.dwLowDateTime; idle_u.HighPart = idle.dwHighDateTime;
    kernel_u.LowPart = kernel.dwLowDateTime; kernel_u.HighPart = kernel.dwHighDateTime;
    user_u.LowPart = user.dwLowDateTime; user_u.HighPart = user.dwHighDateTime;
    if (last_idle.QuadPart == 0) {
        last_idle = idle_u; last_kernel = kernel_u; last_user = user_u;
        return 0.0f;
    }
    ULONGLONG idle_diff = idle_u.QuadPart - last_idle.QuadPart;
    ULONGLONG kernel_diff = kernel_u.QuadPart - last_kernel.QuadPart;
    ULONGLONG user_diff = user_u.QuadPart - last_user.QuadPart;
    ULONGLONG total_diff = kernel_diff + user_diff;
    last_idle = idle_u; last_kernel = kernel_u; last_user = user_u;
    if (total_diff == 0) return 0.0f;
    return (float)(total_diff - idle_diff) / (float)total_diff * 100.0f;
}
static float get_memory_usage_local(void) {
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    if (!GlobalMemoryStatusEx(&mem_info)) return 0.0f;
    return (float)mem_info.dwMemoryLoad;
}
#else
#include <sys/sysinfo.h>
#include <stdio.h>
#include <pthread.h>
static pthread_mutex_t g_cpu_lock = PTHREAD_MUTEX_INITIALIZER;
static float get_cpu_usage_local(void) {
    static long prev_idle = 0, prev_total = 0;
    pthread_mutex_lock(&g_cpu_lock);
    FILE* fp = popen("grep 'cpu ' /proc/stat 2>/dev/null || echo ''", "r");
    if (!fp) { pthread_mutex_unlock(&g_cpu_lock); return 0.0f; }
    char line[512];
    if (!fgets(line, sizeof(line), fp)) { pclose(fp); pthread_mutex_unlock(&g_cpu_lock); return 0.0f; }
    pclose(fp);
    long user, nice, system, idle, iowait, irq, softirq, steal;
    int n = sscanf(line, "cpu %ld %ld %ld %ld %ld %ld %ld %ld",
                   &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);
    if (n < 4) { pthread_mutex_unlock(&g_cpu_lock); return 0.0f; }
    if (n < 8) { iowait = 0; irq = 0; softirq = 0; steal = 0; }
    long total = user + nice + system + idle + iowait + irq + softirq + steal;
    long total_diff = total - prev_total;
    long idle_diff = idle - prev_idle;
    prev_total = total; prev_idle = idle;
    pthread_mutex_unlock(&g_cpu_lock);
    if (total_diff == 0) return 0.0f;
    return (float)(total_diff - idle_diff) / (float)total_diff * 100.0f;
}
static float get_memory_usage_local(void) {
    struct sysinfo info;
    if (sysinfo(&info) != 0) return 0.0f;
    unsigned long total = info.totalram;
    unsigned long free = info.freeram + info.bufferram;
    if (total == 0) return 0.0f;
    return (float)(total - free) / (float)total * 100.0f;
}
#endif

int safety_monitor_resources(SafetyMonitor* monitor) {
    if (!monitor) return -1;

    float cpu_usage = get_cpu_usage_local();
    float memory_usage = get_memory_usage_local();

    SAFETY_LOCK(monitor);
    float cpu_max = monitor->resource_limits.cpu_usage_max;
    float mem_max = monitor->resource_limits.memory_usage_max;
    SAFETY_UNLOCK(monitor);

    if (cpu_usage > cpu_max) {
        SafetyEvent event;
        memset(&event, 0, sizeof(SafetyEvent));
        event.type = SAFETY_EVENT_RESOURCE_OVERUSE;
        event.severity = SAFETY_LEVEL_WARNING;
        snprintf(event.description, sizeof(event.description),
                "CPU使用率过高: %.1f%% (限制: %.1f%%)", cpu_usage, cpu_max);
        snprintf(event.source, sizeof(event.source), "resource_monitor");
        safety_report_event(monitor, &event);
    }

    if (memory_usage > mem_max) {
        SafetyEvent event;
        memset(&event, 0, sizeof(SafetyEvent));
        event.type = SAFETY_EVENT_MEMORY_LEAK;
        event.severity = SAFETY_LEVEL_WARNING;
        snprintf(event.description, sizeof(event.description),
                "内存使用率过高: %.1f%% (限制: %.1f%%)", memory_usage, mem_max);
        snprintf(event.source, sizeof(event.source), "resource_monitor");
        safety_report_event(monitor, &event);
    }

    return 0;
}

int safety_dynamic_adjust_thresholds(SafetyMonitor* monitor) {
    if (!monitor) return -1;

    time_t now_t = time(NULL);
    double elapsed = difftime(now_t, monitor->last_dynamic_adjust_time);
    if (elapsed < 30.0) return 0;

    float cpu_usage = get_cpu_usage_local();
    float memory_usage = get_memory_usage_local();

    SAFETY_LOCK(monitor);

    int total_violations = 0;
    int window_used = monitor->violation_window_count > 0 ? monitor->violation_window_count : 1;
    for (int i = 0; i < 60; i++) {
        total_violations += monitor->violation_history[i];
    }
    float avg_violations_per_min = (float)total_violations / (float)window_used;

    int adjusted = 0;

    /* 高频违规（>3次/分钟）：收紧阈值 */
    if (avg_violations_per_min > 3.0f) {
        float tighten_factor = 0.90f;
        monitor->resource_limits.cpu_usage_max *= tighten_factor;
        monitor->resource_limits.memory_usage_max *= tighten_factor;
        monitor->resource_limits.max_connections = (int)(monitor->resource_limits.max_connections * 0.9f);
        if (monitor->resource_limits.cpu_usage_max < 50.0f) monitor->resource_limits.cpu_usage_max = 50.0f;
        if (monitor->resource_limits.memory_usage_max < 50.0f) monitor->resource_limits.memory_usage_max = 50.0f;
        if (monitor->resource_limits.max_connections < 50) monitor->resource_limits.max_connections = 50;
        adjusted = 1;
    }
    /* 中等违规（1-3次/分钟）：微调 */
    else if (avg_violations_per_min > 1.0f) {
        float tighten_factor = 0.95f;
        monitor->resource_limits.cpu_usage_max *= tighten_factor;
        monitor->resource_limits.memory_usage_max *= tighten_factor;
        if (monitor->resource_limits.cpu_usage_max < 60.0f) monitor->resource_limits.cpu_usage_max = 60.0f;
        if (monitor->resource_limits.memory_usage_max < 60.0f) monitor->resource_limits.memory_usage_max = 60.0f;
        adjusted = 1;
    }
    /* 低违规且系统稳定（<0.5次/分钟且评分>0.9）：逐步恢复默认值 */
    else if (avg_violations_per_min < 0.5f && monitor->stats.current_safety_score > 0.9f) {
        float relax_factor = 1.02f;
        float new_cpu = monitor->resource_limits.cpu_usage_max * relax_factor;
        float new_mem = monitor->resource_limits.memory_usage_max * relax_factor;
        if (new_cpu <= 95.0f) monitor->resource_limits.cpu_usage_max = new_cpu;
        if (new_mem <= 90.0f) monitor->resource_limits.memory_usage_max = new_mem;
        int new_conn = (int)(monitor->resource_limits.max_connections * 1.02f);
        if (new_conn <= 1000) monitor->resource_limits.max_connections = new_conn;
        adjusted = 1;
    }

    /* 根据CPU趋势调整：持续高负载则收紧 */
    if (cpu_usage > 85.0f && monitor->last_cpu_usage > 80.0f) {
        monitor->resource_limits.cpu_usage_max = (cpu_usage + monitor->last_cpu_usage) * 0.5f + 5.0f;
        if (monitor->resource_limits.cpu_usage_max > 95.0f) monitor->resource_limits.cpu_usage_max = 95.0f;
        adjusted = 1;
    }

    monitor->last_cpu_usage = cpu_usage;
    monitor->last_memory_usage = memory_usage;
    monitor->last_dynamic_adjust_time = time(NULL);

    SAFETY_UNLOCK(monitor);
    return adjusted ? 1 : 0;
}

/* ================================================================
 * K-022: 安全规则增强实现
 * ================================================================ */

int safety_validate_physical_action(SafetyMonitor* monitor,
                                     const float* joint_positions,
                                     const float* joint_velocities,
                                     const float* joint_torques,
                                     int joint_count) {
    if (!monitor || !joint_positions || !joint_velocities || joint_count <= 0)
        return -1;

    SAFETY_LOCK(monitor);

    int violations = 0;
    PhysicalBoundaries* pb = &monitor->physical_boundaries;

    for (int i = 0; i < joint_count && i < 16; i++) {
        /* 关节角度边界检查 */
        float angle = joint_positions[i];
        float angle_limit = pb->max_joint_angle[i] > 0.0f ?
            pb->max_joint_angle[i] : 3.14f;
        if (fabsf(angle) > angle_limit) {
            log_warning("[物理安全] 关节%d角度超限: %.3f > %.3f", i, fabsf(angle), angle_limit);
            violations += 2;
        }

        /* 速度边界检查 */
        float vel = fabsf(joint_velocities[i]);
        if (vel > pb->max_velocity) {
            log_warning("[物理安全] 关节%d速度超限: %.3f > %.3f", i, vel, pb->max_velocity);
            violations++;
        }

        /* 扭矩边界检查 */
        if (joint_torques && pb->max_torque > 0.0f) {
            float torque = fabsf(joint_torques[i]);
            if (torque > pb->max_torque) {
                log_warning("[物理安全] 关节%d扭矩超限: %.3f > %.3f", i, torque, pb->max_torque);
                violations += 2;
            }
        }
    }

    /* 记录违规事件 */
    if (violations > 0) {
        SafetyEvent event;
        memset(&event, 0, sizeof(SafetyEvent));
        event.type = SAFETY_EVENT_PHYSICAL_VIOLATION;
        event.severity = (violations >= 4) ? SAFETY_LEVEL_CRITICAL : SAFETY_LEVEL_WARNING;
        snprintf(event.description, sizeof(event.description),
                "物理动作验证: %d处违规 (关节数=%d)", violations, joint_count);
        snprintf(event.source, sizeof(event.source), "safety_validate_physical_action");
        event.timestamp = time(NULL);
        event.handled = 0;
        if (monitor->event_count < monitor->event_capacity) {
            monitor->events[monitor->event_count++] = event;
        }
    }

    SAFETY_UNLOCK(monitor);

    if (violations >= 4) return 2;
    if (violations > 0) return 1;
    return 0;
}

int safety_set_privacy_filter(SafetyMonitor* monitor, int filter_level, int data_types) {
    if (!monitor) return -1;
    if (filter_level < 0 || filter_level > 3) return -1;

    SAFETY_LOCK(monitor);
    monitor->privacy_filter_level = filter_level;
    monitor->privacy_filter_data_types = data_types;
    monitor->privacy_filter_enabled = (filter_level > 0) ? 1 : 0;
    SAFETY_UNLOCK(monitor);

    const char* level_names[] = {"关闭", "仅日志记录", "数据脱敏", "完全隔离"};
    log_info("[隐私安全] 过滤级别: %s, 数据类型掩码: 0x%02X (图像=%d 音频=%d 文本=%d 传感器=%d)",
             level_names[filter_level], data_types,
             (data_types & 1) ? 1 : 0, (data_types & 2) ? 1 : 0,
             (data_types & 4) ? 1 : 0, (data_types & 8) ? 1 : 0);
    return 0;
}

int safety_add_behavioral_constraint(SafetyMonitor* monitor,
                                      const char* constraint_name,
                                      const char* condition,
                                      int action_on_violation) {
    if (!monitor || !constraint_name || !condition) return -1;
    if (monitor->behavioral_constraint_count >= 32) return -1;

    SAFETY_LOCK(monitor);
    int idx = monitor->behavioral_constraint_count++;
    strncpy(monitor->behavioral_constraints[idx].name, constraint_name, 127);
    strncpy(monitor->behavioral_constraints[idx].condition, condition, 255);
    monitor->behavioral_constraints[idx].action = action_on_violation;
    monitor->behavioral_constraints[idx].enabled = 1;
    SAFETY_UNLOCK(monitor);

    log_info("[行为约束] 添加规则: %s (条件=\"%s\", 动作=%d)",
             constraint_name, condition, action_on_violation);
    return 0;
}

int safety_get_privacy_filter_status(const SafetyMonitor* monitor,
                                      int* filter_level,
                                      int* data_types,
                                      int* enabled) {
    if (!monitor) return -1;

    SAFETY_LOCK((SafetyMonitor*)monitor);
    if (filter_level) *filter_level = monitor->privacy_filter_level;
    if (data_types) *data_types = monitor->privacy_filter_data_types;
    if (enabled) *enabled = monitor->privacy_filter_enabled;
    SAFETY_UNLOCK((SafetyMonitor*)monitor);

    return 0;
}

int safety_check_physics(SafetyMonitor* monitor, const float* position,
                        const float* velocity, const float* joints, int joint_count) {
    if (!monitor) return -1;

    SAFETY_LOCK(monitor);
    float max_vel = monitor->physical_boundaries.max_velocity;
    float pos_limits[6];  /* x_min, x_max, y_min, y_max, z_min, z_max */
    /* 从物理边界获取关节角度限制 */
    int bound_joint_count = monitor->physical_boundaries.joint_count;
    int check_count = (joints && joint_count > 0) ? (joint_count < bound_joint_count ? joint_count : bound_joint_count) : 0;
    float min_joints[256];
    float max_joints[256];
    int copy_count = check_count < 256 ? check_count : 256;
    for (int i = 0; i < copy_count; i++) {
        min_joints[i] = monitor->physical_boundaries.min_joint_angle[i];
        max_joints[i] = monitor->physical_boundaries.max_joint_angle[i];
    }
    SAFETY_UNLOCK(monitor);

    /* 位置边界检查 */
    if (position) {
        if (position[0] < pos_limits[0] || position[0] > pos_limits[1] ||
            position[1] < pos_limits[2] || position[1] > pos_limits[3] ||
            position[2] < pos_limits[4] || position[2] > pos_limits[5]) {
            SafetyEvent event;
            memset(&event, 0, sizeof(SafetyEvent));
            event.type = SAFETY_EVENT_PHYSICAL_VIOLATION;
            event.severity = SAFETY_LEVEL_CRITICAL;
            snprintf(event.description, sizeof(event.description),
                    "位置越界: (%.2f, %.2f, %.2f) 超出安全区域", 
                    position[0], position[1], position[2]);
            snprintf(event.source, sizeof(event.source), "physics_check_position");
            safety_report_event(monitor, &event);
            return -1;
        }
    }

    if (velocity) {
        float speed = sqrtf(velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
        if (speed > max_vel) {
            SafetyEvent event;
            memset(&event, 0, sizeof(SafetyEvent));
            event.type = SAFETY_EVENT_PHYSICAL_VIOLATION;
            event.severity = SAFETY_LEVEL_CRITICAL;
            snprintf(event.description, sizeof(event.description),
                    "速度超限: %.2f m/s (限制: %.2f m/s)", speed, max_vel);
            snprintf(event.source, sizeof(event.source), "physics_check");
            safety_report_event(monitor, &event);
            return -1;
        }
    }

    for (int i = 0; i < copy_count; i++) {
        if (joints[i] < min_joints[i] || joints[i] > max_joints[i]) {
            SafetyEvent event;
            memset(&event, 0, sizeof(SafetyEvent));
            event.type = SAFETY_EVENT_PHYSICAL_VIOLATION;
            event.severity = SAFETY_LEVEL_CRITICAL;
            snprintf(event.description, sizeof(event.description),
                    "关节%d角度超限: %.2f rad", i, joints[i]);
            snprintf(event.source, sizeof(event.source), "physics_check");
            safety_report_event(monitor, &event);
            return -1;
        }
    }

    return 0;
}

int safety_reset(SafetyMonitor* monitor) {
    if (!monitor) return -1;
    SAFETY_LOCK(monitor);
    monitor->emergency_stop_active = 0;
    monitor->soft_stop_active = 0;
    monitor->current_level = SAFETY_LEVEL_NORMAL;
    monitor->stats.current_safety_score = 1.0f;
    SAFETY_UNLOCK(monitor);
    return 0;
}
