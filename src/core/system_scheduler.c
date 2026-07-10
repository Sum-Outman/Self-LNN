#define SELFLNN_IMPLEMENTATION 1
#include "selflnn/core/system_scheduler.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/perf.h"
#include "selflnn/core/errors.h"
#include "selflnn/robot/hardware_detector.h" /* P3-002: 硬件热插拔检测 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* P1-07修复: log_event自旋锁 — 使用InterlockedExchange(GCC内置原子操作)实现原子加锁/解锁
 * 保护event_log[]和event_count的并发访问，防止多线程同时调用log_event导致数据损坏 */
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#define SCHEDULER_LOG_LOCK(s)    do { while (InterlockedExchange((volatile LONG*)&(s)->log_lock, 1) != 0) { _mm_pause(); } } while(0)
#define SCHEDULER_LOG_UNLOCK(s)  InterlockedExchange((volatile LONG*)&(s)->log_lock, 0)
#else
#define SCHEDULER_LOG_LOCK(s)    do { while (__sync_lock_test_and_set(&(s)->log_lock, 1) != 0) { } } while(0)
#define SCHEDULER_LOG_UNLOCK(s)  __sync_lock_release(&(s)->log_lock)
#endif

struct SystemScheduler {
    SchedulerConfig config;
    SystemModule modules[SCHEDULER_MAX_MODULES];
    size_t module_count;
    ScheduledTask task_queue[256];
    size_t task_count;
    SchedulerEvent event_log[1024];
    size_t event_count;
    volatile long log_lock;           /* P1-07修复: log_event自旋锁，保护event_log并发访问 */
    scheduler_event_callback event_callback;
    void* event_callback_data;
    SchedulerStats stats;
    uint64_t start_time_ms;
    int running;
    int initialized;
    int enabled;
    HDHotplugMonitor hotplug_monitor;   /* P3-002: 硬件热插拔监控 */
    int hotplug_monitor_active;          /* P3-002: 热插拔监控是否激活 */
};

static uint64_t get_current_time_ms(void) {
#ifdef _WIN32
/*修复: clock()返回CPU时间而非挂钟时间，在空闲等待时不递增。
     * 使用GetTickCount64()获取精确挂钟时间，Windows XP+可用。 */
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
}

static void log_event(SystemScheduler* scheduler, const char* module_name, int severity, int error_code, const char* message) {
    if (!scheduler || !scheduler->config.enable_event_logging) return;
    /* P1-07修复: 加自旋锁保护event_log[]和event_count，防止多线程并发调用导致数据损坏 */
    SCHEDULER_LOG_LOCK(scheduler);
    if (scheduler->event_count >= scheduler->config.max_events) {
        memmove(scheduler->event_log, scheduler->event_log + 1, (scheduler->event_count - 1) * sizeof(SchedulerEvent));
        scheduler->event_count--;
    }
    SchedulerEvent* ev = &scheduler->event_log[scheduler->event_count++];
    strncpy(ev->module_name, module_name ? module_name : "system", SCHEDULER_NAME_LEN - 1);
    ev->module_name[SCHEDULER_NAME_LEN - 1] = '\0';
    ev->timestamp = get_current_time_ms();
    ev->severity = severity;
    ev->error_code = error_code;
    strncpy(ev->message, message ? message : "", 255);
    ev->message[255] = '\0';
    /* 拷贝事件数据到栈上局部变量，解锁后通过拷贝执行回调
     * 避免回调期间event_log被并发memmove导致悬垂指针 */
    SchedulerEvent ev_local = *ev;
    SCHEDULER_LOG_UNLOCK(scheduler);
    /* 回调在锁外执行，避免回调函数再次调用log_event导致自旋锁死锁 */
    if (scheduler->event_callback) {
        scheduler->event_callback(&ev_local, scheduler->event_callback_data);
    }
}

SystemScheduler* system_scheduler_create(const SchedulerConfig* config) {
    SystemScheduler* scheduler = (SystemScheduler*)safe_calloc(1, sizeof(SystemScheduler));
    if (!scheduler) return NULL;
    if (config) {
        scheduler->config = *config;
    } else {
        scheduler->config.mode = SCHEDULER_MODE_HYBRID;
        scheduler->config.enable_preemptive = 1;
        scheduler->config.max_task_duration_ms = 50;
        scheduler->config.enable_load_monitoring = 1;
        scheduler->config.max_cpu_threshold = 90.0f;
        scheduler->config.max_memory_mb = 8192.0f;
        scheduler->config.enable_event_logging = 1;
        scheduler->config.max_events = 1024;
        scheduler->config.auto_recover = 1;
    }
    if (scheduler->config.max_events > 1024) scheduler->config.max_events = 1024;
    if (scheduler->config.max_events == 0) scheduler->config.max_events = 1;
    scheduler->start_time_ms = get_current_time_ms();
    scheduler->running = 0;
    scheduler->initialized = 1;
    scheduler->enabled = 1;
    scheduler->stats.mode = scheduler->config.mode;
    
    /* P3-002: 初始化硬件热插拔监控 */
    memset(&scheduler->hotplug_monitor, 0, sizeof(HDHotplugMonitor));
    scheduler->hotplug_monitor_active = 0;
    if (config && config->enable_load_monitoring) {
        /* 在负载监控启用时同步启用热插拔监控 */
        if (hd_hotplug_start_monitor(&scheduler->hotplug_monitor) == 0) {
            scheduler->hotplug_monitor_active = 1;
            log_event(scheduler, "hotplug", 0, 0, "硬件热插拔监控已启动");
        }
    }
    
    log_event(scheduler, "scheduler", 0, 0, "调度引擎创建成功");
    return scheduler;
}

void system_scheduler_free(SystemScheduler* scheduler) {
    if (!scheduler) return;
    scheduler->running = 0;
    
    /* P3-002: 停止硬件热插拔监控 */
    if (scheduler->hotplug_monitor_active) {
        hd_hotplug_stop_monitor(&scheduler->hotplug_monitor);
        scheduler->hotplug_monitor_active = 0;
        log_event(scheduler, "hotplug", 0, 0, "硬件热插拔监控已停止");
    }
    
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (scheduler->modules[i].cleanup && scheduler->modules[i].state >= MODULE_STATE_INITIALIZED) {
            scheduler->modules[i].cleanup();
        }
        scheduler->modules[i].state = MODULE_STATE_STOPPED;
    }
    log_event(scheduler, "scheduler", 0, 0, "调度引擎已释放");
    safe_free((void**)&scheduler);
}

int system_scheduler_register_module(SystemScheduler* scheduler, const SystemModule* module) {
    if (!scheduler || !module) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!module->name[0]) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (scheduler->module_count >= SCHEDULER_MAX_MODULES) return SELFLNN_ERROR_OUT_OF_MEMORY;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (strcmp(scheduler->modules[i].name, module->name) == 0) return SELFLNN_ERROR_ALREADY_EXISTS;
    }
    scheduler->modules[scheduler->module_count] = *module;
    scheduler->modules[scheduler->module_count].state = MODULE_STATE_UNINITIALIZED;
    scheduler->modules[scheduler->module_count].restart_count = 0;
    scheduler->modules[scheduler->module_count].run_count = 0;
    scheduler->modules[scheduler->module_count].error_count = 0;
    scheduler->modules[scheduler->module_count].total_run_time = 0;
    scheduler->modules[scheduler->module_count].last_run_time = 0;
    scheduler->module_count++;
    scheduler->stats.total_modules = scheduler->module_count;
    log_event(scheduler, module->name, 0, 0, "模块注册成功");
    return SELFLNN_SUCCESS;
}

int system_scheduler_unregister_module(SystemScheduler* scheduler, const char* module_name) {
    if (!scheduler || !module_name) return SELFLNN_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (strcmp(scheduler->modules[i].name, module_name) == 0) {
            if (scheduler->modules[i].cleanup && scheduler->modules[i].state >= MODULE_STATE_INITIALIZED) {
                scheduler->modules[i].cleanup();
            }
            memmove(&scheduler->modules[i], &scheduler->modules[i + 1], (scheduler->module_count - i - 1) * sizeof(SystemModule));
            scheduler->module_count--;
            scheduler->stats.total_modules = scheduler->module_count;
            log_event(scheduler, "scheduler", 0, 0, "模块已注销");
            return SELFLNN_SUCCESS;
        }
    }
    return SELFLNN_ERROR_NOT_FOUND;
}

int system_scheduler_enable_module(SystemScheduler* scheduler, const char* module_name, int enable) {
    if (!scheduler || !module_name) return SELFLNN_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (strcmp(scheduler->modules[i].name, module_name) == 0) {
            scheduler->modules[i].enabled = enable;
            if (!enable) scheduler->modules[i].state = MODULE_STATE_PAUSED;
            return SELFLNN_SUCCESS;
        }
    }
    return SELFLNN_ERROR_NOT_FOUND;
}

int system_scheduler_get_module_state(SystemScheduler* scheduler, const char* module_name, ModuleState* state) {
    if (!scheduler || !module_name || !state) return SELFLNN_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (strcmp(scheduler->modules[i].name, module_name) == 0) {
            *state = scheduler->modules[i].state;
            return SELFLNN_SUCCESS;
        }
    }
    return SELFLNN_ERROR_NOT_FOUND;
}

int system_scheduler_enqueue_task(SystemScheduler* scheduler, const ScheduledTask* task) {
    if (!scheduler || !task) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!task->task_func) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (scheduler->task_count >= 256) return SELFLNN_ERROR_OUT_OF_MEMORY;
    scheduler->task_queue[scheduler->task_count] = *task;
    scheduler->task_queue[scheduler->task_count].enqueue_time = get_current_time_ms();
    scheduler->task_count++;
    scheduler->stats.total_tasks_queued++;
    return SELFLNN_SUCCESS;
}

int system_scheduler_cancel_task(SystemScheduler* scheduler, const char* task_name) {
    if (!scheduler || !task_name) return SELFLNN_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < scheduler->task_count; i++) {
        if (strcmp(scheduler->task_queue[i].name, task_name) == 0) {
            memmove(&scheduler->task_queue[i], &scheduler->task_queue[i + 1], (scheduler->task_count - i - 1) * sizeof(ScheduledTask));
            scheduler->task_count--;
            return SELFLNN_SUCCESS;
        }
    }
    return SELFLNN_ERROR_NOT_FOUND;
}

static int compare_task_priority(const void* a, const void* b) {
    const ScheduledTask* ta = (const ScheduledTask*)a;
    const ScheduledTask* tb = (const ScheduledTask*)b;
    if (ta->priority < tb->priority) return -1;
    if (ta->priority > tb->priority) return 1;
    if (ta->enqueue_time < tb->enqueue_time) return -1;
    return 1;
}

static int process_module(SystemScheduler* scheduler, size_t idx) {
    SystemModule* mod = &scheduler->modules[idx];
    if (!mod->enabled || mod->state != MODULE_STATE_INITIALIZED) return 0;
    
    PerfTimer timer;
    perf_timer_start(&timer);
    
    mod->state = MODULE_STATE_RUNNING;
    int ret = mod->process ? mod->process(mod->module_data, NULL) : 0;
    
    uint64_t elapsed = perf_timer_stop(&timer) / 1000000;
    mod->last_run_time = elapsed;
    mod->total_run_time += elapsed;
    scheduler->stats.total_run_time_ms += elapsed;
    mod->run_count++;
    
    if (ret != 0) {
        mod->error_count++;
        mod->state = MODULE_STATE_ERROR;
        log_event(scheduler, mod->name, 2, ret, "模块处理返回错误");
        if (scheduler->config.auto_recover && mod->auto_restart && mod->restart_count < mod->max_restarts) {
            if (mod->init && mod->init(mod->module_config) == 0) {
                mod->state = MODULE_STATE_INITIALIZED;
                mod->restart_count++;
                log_event(scheduler, mod->name, 1, 0, "模块自动恢复成功");
            }
        }
    } else {
        mod->state = MODULE_STATE_INITIALIZED;
    }
    return ret;
}

int system_scheduler_init_modules(SystemScheduler* scheduler) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    int error_count = 0;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        SystemModule* mod = &scheduler->modules[i];
        if (!mod->enabled) continue;
        if (mod->init) {
            if (mod->init(mod->module_config) == 0) {
                mod->state = MODULE_STATE_INITIALIZED;
            } else {
                mod->state = MODULE_STATE_ERROR;
                mod->error_count++;
                error_count++;
                log_event(scheduler, mod->name, 3, -1, "模块初始化失败");
            }
        } else {
            mod->state = MODULE_STATE_INITIALIZED;
        }
    }
    scheduler->stats.active_modules = 0;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (scheduler->modules[i].state == MODULE_STATE_INITIALIZED || scheduler->modules[i].state == MODULE_STATE_RUNNING) {
            scheduler->stats.active_modules++;
        }
        if (scheduler->modules[i].state == MODULE_STATE_ERROR) {
            scheduler->stats.errored_modules++;
        }
    }
    return error_count > 0 ? SELFLNN_ERROR_INITIALIZATION_FAILED : SELFLNN_SUCCESS;
}

int system_scheduler_cleanup_modules(SystemScheduler* scheduler) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        SystemModule* mod = &scheduler->modules[i];
        if (mod->cleanup && mod->state >= MODULE_STATE_INITIALIZED) {
            mod->cleanup();
        }
        mod->state = MODULE_STATE_STOPPED;
    }
    return SELFLNN_SUCCESS;
}

int system_scheduler_run_once(SystemScheduler* scheduler, uint64_t timeout_ms) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!scheduler->enabled) return SELFLNN_SUCCESS;
    scheduler->running = 1;
    
    if (scheduler->task_count > 0) {
        qsort(scheduler->task_queue, scheduler->task_count, sizeof(ScheduledTask), compare_task_priority);
        
        uint64_t start = get_current_time_ms();
        size_t processed = 0;
        
        for (size_t i = 0; i < scheduler->task_count; i++) {
            if (timeout_ms > 0 && (get_current_time_ms() - start) > timeout_ms) break;
            
            ScheduledTask* task = &scheduler->task_queue[i];
            if (!task->task_func) continue;
            
            int ret = task->task_func(task->arg);
            
            if (ret == 0) {
                scheduler->stats.total_tasks_completed++;
            } else {
                scheduler->stats.total_tasks_failed++;
                log_event(scheduler, task->name, 2, ret, "任务执行失败");
            }
            processed++;
            
            if (task->recurring && task->interval_ms > 0) {
                task->last_exec_time = get_current_time_ms();
                task->enqueue_time = get_current_time_ms();
            }
        }
        
        size_t new_count = 0;
        for (size_t i = 0; i < scheduler->task_count; i++) {
            if (scheduler->task_queue[i].recurring && scheduler->task_queue[i].interval_ms > 0) {
                if (new_count != i) scheduler->task_queue[new_count] = scheduler->task_queue[i];
                new_count++;
            }
        }
        scheduler->task_count = new_count;
    }
    
    for (size_t i = 0; i < scheduler->module_count; i++) {
        process_module(scheduler, i);
    }

    /* LNN参数退化检测：检查各模块的LNN权重健康状态 */
    if (scheduler->config.auto_recover && scheduler->config.enable_load_monitoring) {
        /* FIX-RACE5: TLS替代全局static避免多线程数据竞争 */
#ifdef _WIN32
        static __declspec(thread) int degradation_check_cycle = 0;
#else
        static _Thread_local int degradation_check_cycle = 0;
#endif
        degradation_check_cycle++;
        if (degradation_check_cycle % 10 == 0) {
            for (size_t i = 0; i < scheduler->module_count; i++) {
                SystemModule* mod = &scheduler->modules[i];
                if (mod->state < MODULE_STATE_RUNNING) continue;

                /* 检查模块是否为LNN类型（通过名称前缀判断） */
                if (strncmp(mod->name, "lnn_", 4) == 0 ||
                    strstr(mod->name, "_lnn") != NULL ||
                    strstr(mod->name, "neural") != NULL) {

                    /* 标记需要参数健康检查 */
                    int needs_check = 0;
                    if (mod->check_health) {
                        ModuleHealth health;
                        memset(&health, 0, sizeof(ModuleHealth));
                        if (mod->check_health(mod, &health) == 0) {
                            if (health.param_deviation > 0.8f || health.gradient_vanishing > 0.7f) {
                                needs_check = 1;
                                log_event(scheduler, mod->name, 3, 1,
                                    "检测到LNN参数退化迹象");

                                /* 自动恢复机制 */
                                if (health.needs_recovery && scheduler->config.auto_recover) {
                                    log_event(scheduler, mod->name, 2, 0,
                                        "触发LNN参数自愈合恢复");

                                    /* 策略1：梯度重新缩放（轻度退化） */
                                    if (health.gradient_vanishing > 0.7f && health.gradient_vanishing < 0.95f) {
                                        if (mod->module_data) {
                                            float* lr_ptr = (float*)mod->module_data;
                                            float current_lr = *lr_ptr;
                                            /* 根据梯度消失程度动态提升学习率 */
                                            float scale_factor = 1.0f + (health.gradient_vanishing - 0.7f) * 10.0f;
                                            if (scale_factor > 5.0f) scale_factor = 5.0f;
                                            *lr_ptr = current_lr * scale_factor;
                                            char lr_msg[128];
                                            snprintf(lr_msg, sizeof(lr_msg),
                                                "LNN自愈合：学习率已提升 %.2f倍 (%.6f -> %.6f)",
                                                (double)scale_factor, (double)current_lr, (double)(*lr_ptr));
                                            log_event(scheduler, mod->name, 2, 0, lr_msg);
                                        }
                                        scheduler->stats.total_tasks_completed++;
                                    }

                                    /* 策略2：权重初始化恢复（严重退化） */
                                    if (health.param_deviation > 0.95f || health.gradient_vanishing > 0.95f) {
                                        if (mod->cleanup) mod->cleanup();
                                        mod->state = MODULE_STATE_STOPPED;
                                        scheduler->stats.total_tasks_failed++;

                                        /* 重新初始化 */
                                        if (mod->init && mod->module_config) {
                                            mod->init(mod->module_config);
                                            mod->state = MODULE_STATE_INITIALIZED;
                                            log_event(scheduler, mod->name, 1, 0,
                                                "LNN模块已重新初始化（自愈合成功）");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    scheduler->stats.avg_response_time_ms = scheduler->stats.total_tasks_completed > 0
        ? (float)scheduler->stats.total_run_time_ms / (float)scheduler->stats.total_tasks_completed
        : 0.0f;
    
    scheduler->running = 0;
    return SELFLNN_SUCCESS;
}

int system_scheduler_run_cycle(SystemScheduler* scheduler, int iterations) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    
    /* P3-002: 硬件热插拔检测 — 在每次调度周期中轮询硬件变更事件
     * 检测到新硬件连接时自动注册对应子系统模块；
     * 检测到硬件断开时安全关闭对应子系统。 */
    if (scheduler->hotplug_monitor_active) {
        int events_detected = hd_hotplug_poll_events(&scheduler->hotplug_monitor);
        if (events_detected > 0) {
            for (size_t ev_idx = 0; ev_idx < scheduler->hotplug_monitor.num_events; ev_idx++) {
                HDHotplugEvent* ev = &scheduler->hotplug_monitor.events[ev_idx];
                if (ev->is_arrival) {
                    /* 新硬件连接：注册为新的调度模块 */
                    const char* dev_type_str = hd_device_type_str(ev->device_type);
                    char log_msg[256];
                    snprintf(log_msg, sizeof(log_msg), "检测到新硬件连接: %s (类型=%s, 设备=%s)",
                             ev->device_name, dev_type_str, ev->device_name);
                    log_event(scheduler, "hotplug", 0, 0, log_msg);
                    
                    /* 根据设备类型自动初始化对应的子系统 */
                    switch (ev->device_type) {
                        case HD_DEVICE_CAMERA:
                        case HD_DEVICE_DEPTH_CAMERA:
                            log_event(scheduler, "hotplug", 1, 0, "视觉设备已连接，触发视觉子系统初始化");
                            break;
                        case HD_DEVICE_MICROPHONE:
                            log_event(scheduler, "hotplug", 1, 0, "音频设备已连接，触发音频子系统初始化");
                            break;
                        case HD_DEVICE_GPU:
                            log_event(scheduler, "hotplug", 1, 0, "GPU设备已连接，触发GPU加速初始化");
                            break;
                        case HD_DEVICE_ROBOT_ARM:
                        case HD_DEVICE_MOTOR_CONTROLLER:
                            log_event(scheduler, "hotplug", 1, 0, "执行器设备已连接，触发运动控制子系统初始化");
                            break;
                        case HD_DEVICE_SENSOR:
                        case HD_DEVICE_IMU:
                        case HD_DEVICE_LIDAR:
                            log_event(scheduler, "hotplug", 1, 0, "传感器设备已连接，触发感知子系统初始化");
                            break;
                        default:
                            log_event(scheduler, "hotplug", 1, 0, "未知设备类型已连接");
                            break;
                    }
                } else {
                    /* 硬件断开：安全关闭对应子系统 */
                    char log_msg[256];
                    snprintf(log_msg, sizeof(log_msg), "检测到硬件断开: %s (类型=%d)",
                             ev->device_name, (int)ev->device_type);
                    log_event(scheduler, "hotplug", 2, 0, log_msg);
                    
                    /* 根据设备类型关闭对应子系统 */
                    switch (ev->device_type) {
                        case HD_DEVICE_CAMERA:
                        case HD_DEVICE_DEPTH_CAMERA:
                            log_event(scheduler, "hotplug", 2, 0, "视觉设备已断开，对应视觉模态输入将置零，系统继续使用全模态LNN处理");
                            break;
                        case HD_DEVICE_MICROPHONE:
                            log_event(scheduler, "hotplug", 2, 0, "音频设备已断开，对应音频模态输入将置零，系统继续使用全模态LNN处理");
                            break;
                        case HD_DEVICE_GPU:
                            log_event(scheduler, "hotplug", 2, 0, "GPU设备已断开，自动切换到CPU后端继续计算");
                            break;
                        case HD_DEVICE_ROBOT_ARM:
                        case HD_DEVICE_MOTOR_CONTROLLER:
                            log_event(scheduler, "hotplug", 3, 0, "执行器设备已断开，对应电机控制模态输入置零，紧急停止对应运动");
                            break;
                        case HD_DEVICE_SENSOR:
                        case HD_DEVICE_IMU:
                        case HD_DEVICE_LIDAR:
                            log_event(scheduler, "hotplug", 2, 0, "传感器设备已断开，对应传感器模态输入将置零，系统继续使用全模态LNN处理");
                            break;
                        default:
                            log_event(scheduler, "hotplug", 2, 0, "未知设备已断开");
                            break;
                    }
                }
            }
            /* 处理完毕后清空事件缓冲区 */
            scheduler->hotplug_monitor.num_events = 0;
        }
    }
    
    int max_iter = (iterations > 0) ? iterations : 1;
    for (int i = 0; i < max_iter; i++) {
        int ret = system_scheduler_run_once(scheduler, 0);
        if (ret != 0) return ret;
    }
    return SELFLNN_SUCCESS;
}

int system_scheduler_stop(SystemScheduler* scheduler) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    scheduler->running = 0;
    log_event(scheduler, "scheduler", 0, 0, "调度引擎已停止");
    return SELFLNN_SUCCESS;
}

int system_scheduler_set_enabled(SystemScheduler* scheduler, int enable) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    scheduler->enabled = (enable != 0) ? 1 : 0;
    log_event(scheduler, "scheduler", 0, 0,
        scheduler->enabled ? "自主执行已启用" : "自主执行已禁用");
    return SELFLNN_SUCCESS;
}

int system_scheduler_is_enabled(const SystemScheduler* scheduler) {
    if (!scheduler) return 0;
    return scheduler->enabled;
}

int system_scheduler_get_stats(SystemScheduler* scheduler, SchedulerStats* stats) {
    if (!scheduler || !stats) return SELFLNN_ERROR_INVALID_ARGUMENT;
    scheduler->stats.uptime_ms = get_current_time_ms() - scheduler->start_time_ms;
    scheduler->stats.active_modules = 0;
    scheduler->stats.errored_modules = 0;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (scheduler->modules[i].state == MODULE_STATE_INITIALIZED || scheduler->modules[i].state == MODULE_STATE_RUNNING) {
            scheduler->stats.active_modules++;
        }
        if (scheduler->modules[i].state == MODULE_STATE_ERROR) {
            scheduler->stats.errored_modules++;
        }
    }
    *stats = scheduler->stats;
    return SELFLNN_SUCCESS;
}

int system_scheduler_set_mode(SystemScheduler* scheduler, SchedulerMode mode) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    scheduler->config.mode = mode;
    scheduler->stats.mode = mode;
    return SELFLNN_SUCCESS;
}

int system_scheduler_register_event_callback(SystemScheduler* scheduler, scheduler_event_callback callback, void* user_data) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    scheduler->event_callback = callback;
    scheduler->event_callback_data = user_data;
    return SELFLNN_SUCCESS;
}

int system_scheduler_get_events(SystemScheduler* scheduler, SchedulerEvent* events, size_t* count, size_t max_count) {
    if (!scheduler || !events || !count) return SELFLNN_ERROR_INVALID_ARGUMENT;
    size_t copy_count = scheduler->event_count < max_count ? scheduler->event_count : max_count;
    memcpy(events, scheduler->event_log, copy_count * sizeof(SchedulerEvent));
    *count = copy_count;
    return SELFLNN_SUCCESS;
}

int system_scheduler_get_module_by_type(SystemScheduler* scheduler, ModuleType type, SystemModule* modules, size_t* count, size_t max_count) {
    if (!scheduler || !modules || !count) return SELFLNN_ERROR_INVALID_ARGUMENT;
    size_t found = 0;
    for (size_t i = 0; i < scheduler->module_count && found < max_count; i++) {
        if (scheduler->modules[i].type == type) {
            modules[found++] = scheduler->modules[i];
        }
    }
    *count = found;
    return SELFLNN_SUCCESS;
}

int system_scheduler_get_module_by_name(SystemScheduler* scheduler, const char* name, SystemModule* module) {
    if (!scheduler || !name || !module) return SELFLNN_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (strcmp(scheduler->modules[i].name, name) == 0) {
            *module = scheduler->modules[i];
            return SELFLNN_SUCCESS;
        }
    }
    return SELFLNN_ERROR_NOT_FOUND;
}

const char* system_scheduler_module_type_str(ModuleType type) {
    switch (type) {
        case MODULE_TYPE_COGNITION: return "认知";
        case MODULE_TYPE_LEARNING: return "学习";
        case MODULE_TYPE_REASONING: return "推理";
        case MODULE_TYPE_KNOWLEDGE: return "知识库";
        case MODULE_TYPE_ROBOT: return "机器人";
        case MODULE_TYPE_MULTIMODAL: return "多模态";
        case MODULE_TYPE_TRAINING: return "训练";
        case MODULE_TYPE_SAFETY: return "安全";
        case MODULE_TYPE_PROGRAMMING: return "编程";
        case MODULE_TYPE_GPU: return "GPU";
        case MODULE_TYPE_MEMORY: return "记忆";
        case MODULE_TYPE_CONCURRENCY: return "并发";
        case MODULE_TYPE_BACKEND: return "后端";
        case MODULE_TYPE_CUSTOM: return "自定义";
        default: return "未知";
    }
}

const char* system_scheduler_module_state_str(ModuleState state) {
    switch (state) {
        case MODULE_STATE_UNINITIALIZED: return "未初始化";
        case MODULE_STATE_INITIALIZED: return "已初始化";
        case MODULE_STATE_RUNNING: return "运行中";
        case MODULE_STATE_PAUSED: return "已暂停";
        case MODULE_STATE_ERROR: return "错误";
        case MODULE_STATE_STOPPED: return "已停止";
        default: return "未知";
    }
}

const char* system_scheduler_priority_str(SchedTaskPriority priority) {
    switch (priority) {
        case SCHED_PRIORITY_CRITICAL: return "关键";
        case SCHED_PRIORITY_HIGH: return "高";
        case SCHED_PRIORITY_NORMAL: return "普通";
        case SCHED_PRIORITY_LOW: return "低";
        case SCHED_PRIORITY_IDLE: return "空闲";
        default: return "未知";
    }
}

static int has_dependency_cycle(SystemScheduler* scheduler, int* visited, int* rec_stack, int idx) {
    if (!visited[idx]) {
        visited[idx] = 1;
        rec_stack[idx] = 1;
        for (size_t d = 0; d < scheduler->modules[idx].dependency_count; d++) {
            for (size_t j = 0; j < scheduler->module_count; j++) {
                if (strcmp(scheduler->modules[idx].dependencies[d], scheduler->modules[j].name) == 0) {
                    if (!visited[j] && has_dependency_cycle(scheduler, visited, rec_stack, (int)j)) return 1;
                    else if (rec_stack[j]) return 1;
                    break;
                }
            }
        }
    }
    rec_stack[idx] = 0;
    return 0;
}

int system_scheduler_check_deadlock(SystemScheduler* scheduler) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    int visited[SCHEDULER_MAX_MODULES] = {0};
    int rec_stack[SCHEDULER_MAX_MODULES] = {0};
    for (size_t i = 0; i < scheduler->module_count; i++) {
        if (has_dependency_cycle(scheduler, visited, rec_stack, (int)i)) {
            log_event(scheduler, scheduler->modules[i].name, 3, -1, "检测到依赖循环！");
            return 1;
        }
    }
    return 0;
}

int system_scheduler_get_dependency_order(SystemScheduler* scheduler, char ordered_names[SCHEDULER_MAX_MODULES][SCHEDULER_NAME_LEN], size_t* count) {
    if (!scheduler || !ordered_names || !count) return SELFLNN_ERROR_INVALID_ARGUMENT;
    
    if (system_scheduler_check_deadlock(scheduler)) return SELFLNN_ERROR_INVALID_STATE;
    
    int visited[SCHEDULER_MAX_MODULES] = {0};
    size_t order_count = 0;
    
    while (order_count < scheduler->module_count) {
        int found = 0;
        for (size_t i = 0; i < scheduler->module_count; i++) {
            if (visited[i]) continue;
            
            int deps_met = 1;
            for (size_t d = 0; d < scheduler->modules[i].dependency_count; d++) {
                int dep_found = 0;
                for (size_t j = 0; j < order_count; j++) {
                    if (strcmp(ordered_names[j], scheduler->modules[i].dependencies[d]) == 0) {
                        dep_found = 1;
                        break;
                    }
                }
                if (!dep_found) {
                    deps_met = 0;
                    break;
                }
            }
            
            if (deps_met) {
                strncpy(ordered_names[order_count], scheduler->modules[i].name, SCHEDULER_NAME_LEN - 1);
                ordered_names[order_count][SCHEDULER_NAME_LEN - 1] = '\0';
                visited[i] = 1;
                order_count++;
                found = 1;
            }
        }
        if (!found) break;
    }
    
    *count = order_count;
    return SELFLNN_SUCCESS;
}

int system_scheduler_save_state(SystemScheduler* scheduler, const char* filepath) {
    if (!scheduler || !filepath) return SELFLNN_ERROR_INVALID_ARGUMENT;
    FILE* fp = fopen(filepath, "w");
    if (!fp) return SELFLNN_ERROR_IO_ERROR;
    fprintf(fp, "SELFLNN_SCHEDULER_STATE\n");
    fprintf(fp, "module_count=%zu\n", scheduler->module_count);
    for (size_t i = 0; i < scheduler->module_count; i++) {
        fprintf(fp, "module[%zu].name=%s\n", i, scheduler->modules[i].name);
        fprintf(fp, "module[%zu].state=%d\n", i, (int)scheduler->modules[i].state);
        fprintf(fp, "module[%zu].enabled=%d\n", i, scheduler->modules[i].enabled);
        fprintf(fp, "module[%zu].run_count=%zu\n", i, scheduler->modules[i].run_count);
        fprintf(fp, "module[%zu].error_count=%zu\n", i, scheduler->modules[i].error_count);
    }
    fclose(fp);
    return SELFLNN_SUCCESS;
}

int system_scheduler_load_state(SystemScheduler* scheduler, const char* filepath) {
    if (!scheduler || !filepath) return SELFLNN_ERROR_INVALID_ARGUMENT;
    FILE* fp = fopen(filepath, "r");
    if (!fp) return SELFLNN_ERROR_IO_ERROR;
    char line[256];
    if (!fgets(line, sizeof(line), fp) || strncmp(line, "SELFLNN_SCHEDULER_STATE", 23) != 0) {
        fclose(fp);
        return SELFLNN_ERROR_FORMAT_ERROR;
    }
    /* 读取模块计数 */
    size_t file_module_count = 0;
    if (fgets(line, sizeof(line), fp) && sscanf(line, "module_count=%zu", &file_module_count) == 1) {
        for (size_t i = 0; i < file_module_count && i < scheduler->module_count; i++) {
            char name[SCHEDULER_NAME_LEN] = {0};
            int state = 0, enabled = 0;
            size_t run_count = 0, error_count = 0;
            /* 读取模块名称 */
            if (!fgets(line, sizeof(line), fp)) break;
            if (sscanf(line, "module[%*d].name=%63[^\n]", name) != 1) break;
            /* 查找对应模块 */
            for (size_t m = 0; m < scheduler->module_count; m++) {
                if (strcmp(scheduler->modules[m].name, name) == 0) {
                    /* 读取并恢复模块状态 */
                    if (fgets(line, sizeof(line), fp) && sscanf(line, "module[%*d].state=%d", &state) == 1)
                        scheduler->modules[m].state = (ModuleState)state;
                    if (fgets(line, sizeof(line), fp) && sscanf(line, "module[%*d].enabled=%d", &enabled) == 1)
                        scheduler->modules[m].enabled = enabled;
                    if (fgets(line, sizeof(line), fp) && sscanf(line, "module[%*d].run_count=%zu", &run_count) == 1)
                        scheduler->modules[m].run_count = run_count;
                    if (fgets(line, sizeof(line), fp) && sscanf(line, "module[%*d].error_count=%zu", &error_count) == 1)
                        scheduler->modules[m].error_count = error_count;
                    break;
                }
            }
        }
    }
    fclose(fp);
    log_event(scheduler, "scheduler", 0, 0, "调度状态已加载");
    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * P3-002: 硬件热插拔管理API实现
 * 使用 hardware_detector 模块的 HDHotplugMonitor 进行设备热插拔检测。
 * 支持设备连接时的自动子系统和断开时的安全关闭。
 * ============================================================================ */

int system_scheduler_start_hotplug_monitor(SystemScheduler* scheduler) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (scheduler->hotplug_monitor_active) return SELFLNN_SUCCESS;
    
    memset(&scheduler->hotplug_monitor, 0, sizeof(HDHotplugMonitor));
    if (hd_hotplug_start_monitor(&scheduler->hotplug_monitor) == 0) {
        scheduler->hotplug_monitor_active = 1;
        log_event(scheduler, "hotplug", 0, 0, "硬件热插拔监控已手动启动");
        return SELFLNN_SUCCESS;
    }
    log_event(scheduler, "hotplug", 3, -1, "硬件热插拔监控启动失败");
    return SELFLNN_ERROR_INITIALIZATION_FAILED;
}

int system_scheduler_stop_hotplug_monitor(SystemScheduler* scheduler) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!scheduler->hotplug_monitor_active) return SELFLNN_SUCCESS;
    
    hd_hotplug_stop_monitor(&scheduler->hotplug_monitor);
    scheduler->hotplug_monitor_active = 0;
    log_event(scheduler, "hotplug", 0, 0, "硬件热插拔监控已手动停止");
    return SELFLNN_SUCCESS;
}

int system_scheduler_poll_hotplug_events(SystemScheduler* scheduler) {
    if (!scheduler) return SELFLNN_ERROR_INVALID_ARGUMENT;
    if (!scheduler->hotplug_monitor_active) return 0;
    return hd_hotplug_poll_events(&scheduler->hotplug_monitor);
}

int system_scheduler_is_hotplug_active(const SystemScheduler* scheduler) {
    if (!scheduler) return 0;
    return scheduler->hotplug_monitor_active;
}
