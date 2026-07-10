#ifndef SELFLNN_SYSTEM_SCHEDULER_H
#define SELFLNN_SYSTEM_SCHEDULER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "common.h"

#define SCHEDULER_MAX_MODULES 64
#define SCHEDULER_MAX_DEPENDENCIES 16
#define SCHEDULER_NAME_LEN 64

typedef enum {
    MODULE_TYPE_COGNITION = 0,
    MODULE_TYPE_LEARNING = 1,
    MODULE_TYPE_REASONING = 2,
    MODULE_TYPE_KNOWLEDGE = 3,
    MODULE_TYPE_ROBOT = 4,
    MODULE_TYPE_MULTIMODAL = 5,
    MODULE_TYPE_TRAINING = 6,
    MODULE_TYPE_SAFETY = 7,
    MODULE_TYPE_PROGRAMMING = 8,
    MODULE_TYPE_GPU = 9,
    MODULE_TYPE_MEMORY = 10,
    MODULE_TYPE_CONCURRENCY = 11,
    MODULE_TYPE_BACKEND = 12,
    MODULE_TYPE_CUSTOM = 13
} ModuleType;

typedef enum {
    SCHED_PRIORITY_CRITICAL = 0,
    SCHED_PRIORITY_HIGH = 1,
    SCHED_PRIORITY_NORMAL = 2,
    SCHED_PRIORITY_LOW = 3,
    SCHED_PRIORITY_IDLE = 4
} SchedTaskPriority;

typedef enum {
    MODULE_STATE_UNINITIALIZED = 0,
    MODULE_STATE_INITIALIZED = 1,
    MODULE_STATE_RUNNING = 2,
    MODULE_STATE_PAUSED = 3,
    MODULE_STATE_ERROR = 4,
    MODULE_STATE_STOPPED = 5
} ModuleState;

typedef enum {
    SCHEDULER_MODE_ROUND_ROBIN = 0,
    SCHEDULER_MODE_PRIORITY = 1,
    SCHEDULER_MODE_HYBRID = 2
} SchedulerMode;

typedef int (*module_init_fn)(void* config);
typedef int (*module_process_fn)(void* input, void* output);
typedef int (*module_cleanup_fn)(void);
typedef int (*module_status_fn)(char* buffer, size_t buffer_size);

/**
 * @brief 模块健康检查结果（在SystemModule之前定义以避免前向引用）
 */
typedef struct {
    float param_deviation;
    float gradient_vanishing;
    float activation_saturation;
    float weight_norm;
    float grad_norm;
    int needs_recovery;
} ModuleHealth;

struct SystemModule;
typedef int (*module_health_fn)(struct SystemModule* module, ModuleHealth* health);

typedef struct SystemModule {
    char name[SCHEDULER_NAME_LEN];
    ModuleType type;
    module_init_fn init;
    module_process_fn process;
    module_cleanup_fn cleanup;
    module_status_fn get_status;
    SchedTaskPriority priority;
    int enabled;
    int auto_restart;
    int restart_count;
    int max_restarts;
    ModuleState state;
    float urgency_score;
    uint64_t last_run_time;
    uint64_t total_run_time;
    size_t run_count;
    size_t error_count;
    char dependencies[SCHEDULER_MAX_DEPENDENCIES][SCHEDULER_NAME_LEN];
    size_t dependency_count;
    void* module_config;
    void* module_data;
    module_health_fn check_health;
} SystemModule;

/** 已移至SystemModule之前定义，此处保留注释 */
/* ModuleHealth 和 module_health_fn 已前置定义 */

typedef struct {
    char name[SCHEDULER_NAME_LEN];
    char description[256];
    SchedTaskPriority priority;
    int (*task_func)(void* arg);
    void* arg;
    uint64_t max_exec_time_ms;
    uint64_t enqueue_time;
    int recurring;
    uint64_t interval_ms;
    uint64_t last_exec_time;
} ScheduledTask;

typedef struct {
    char module_name[SCHEDULER_NAME_LEN];
    uint64_t timestamp;
    int severity;
    int error_code;
    char message[256];
} SchedulerEvent;

typedef struct {
    uint64_t total_modules;
    uint64_t active_modules;
    uint64_t errored_modules;
    uint64_t total_tasks_queued;
    uint64_t total_tasks_completed;
    uint64_t total_tasks_failed;
    float avg_response_time_ms;
    float cpu_usage;
    float memory_usage_mb;
    uint64_t uptime_ms;
    uint64_t total_run_time_ms;
    SchedulerMode mode;
} SchedulerStats;

typedef void (*scheduler_event_callback)(const SchedulerEvent* event, void* user_data);

typedef struct {
    SchedulerMode mode;                /**< 调度模式（协作式优先级队列调度） */
    int enable_preemptive;             /**< 是否启用模块级监控（非抢占式，当前为协作式） */
    uint64_t max_task_duration_ms;     /**< 单任务最大执行时长阈值（ms），用于健康监控告警 */
    int enable_load_monitoring;        /**< 是否启用负载监控 */
    float max_cpu_threshold;           /**< CPU使用率告警阈值 */
    float max_memory_mb;               /**< 内存使用告警阈值（MB） */
    int enable_event_logging;          /**< 是否启用事件日志 */
    size_t max_events;                 /**< 事件日志最大条目数 */
    int auto_recover;                  /**< 是否启用自动恢复机制 */
} SchedulerConfig;

typedef struct SystemScheduler SystemScheduler;

SystemScheduler* system_scheduler_create(const SchedulerConfig* config);
void system_scheduler_free(SystemScheduler* scheduler);

int system_scheduler_register_module(SystemScheduler* scheduler, const SystemModule* module);
int system_scheduler_unregister_module(SystemScheduler* scheduler, const char* module_name);
int system_scheduler_enable_module(SystemScheduler* scheduler, const char* module_name, int enable);
int system_scheduler_get_module_state(SystemScheduler* scheduler, const char* module_name, ModuleState* state);

int system_scheduler_enqueue_task(SystemScheduler* scheduler, const ScheduledTask* task);
int system_scheduler_cancel_task(SystemScheduler* scheduler, const char* task_name);

int system_scheduler_run_once(SystemScheduler* scheduler, uint64_t timeout_ms);
int system_scheduler_run_cycle(SystemScheduler* scheduler, int iterations);
int system_scheduler_stop(SystemScheduler* scheduler);

int system_scheduler_init_modules(SystemScheduler* scheduler);
int system_scheduler_cleanup_modules(SystemScheduler* scheduler);

int system_scheduler_set_enabled(SystemScheduler* scheduler, int enable);
int system_scheduler_is_enabled(const SystemScheduler* scheduler);
int system_scheduler_get_stats(SystemScheduler* scheduler, SchedulerStats* stats);
int system_scheduler_set_mode(SystemScheduler* scheduler, SchedulerMode mode);

int system_scheduler_register_event_callback(SystemScheduler* scheduler, scheduler_event_callback callback, void* user_data);

/* P3-002: 硬件热插拔管理API */
int system_scheduler_start_hotplug_monitor(SystemScheduler* scheduler);
int system_scheduler_stop_hotplug_monitor(SystemScheduler* scheduler);
int system_scheduler_poll_hotplug_events(SystemScheduler* scheduler);
int system_scheduler_is_hotplug_active(const SystemScheduler* scheduler);
int system_scheduler_get_events(SystemScheduler* scheduler, SchedulerEvent* events, size_t* count, size_t max_count);

int system_scheduler_get_module_by_type(SystemScheduler* scheduler, ModuleType type, SystemModule* modules, size_t* count, size_t max_count);
int system_scheduler_get_module_by_name(SystemScheduler* scheduler, const char* name, SystemModule* module);

const char* system_scheduler_module_type_str(ModuleType type);
const char* system_scheduler_module_state_str(ModuleState state);
const char* system_scheduler_priority_str(SchedTaskPriority priority);

int system_scheduler_get_dependency_order(SystemScheduler* scheduler, char ordered_names[SCHEDULER_MAX_MODULES][SCHEDULER_NAME_LEN], size_t* count);
int system_scheduler_check_deadlock(SystemScheduler* scheduler);

int system_scheduler_save_state(SystemScheduler* scheduler, const char* filepath);
int system_scheduler_load_state(SystemScheduler* scheduler, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif
