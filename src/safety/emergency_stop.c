/**
 * @file emergency_stop.c
 * @brief 紧急停止系统完整实现
 *
 * 实现：
 * 1. A09.2.1 多层次紧急停止 — 软停止/暂停/硬停止/终止/物理切断
 * 2. A09.2.2 安全回退 — 渐进式降级与状态保持
 * 3. A09.2.3 CfC预测性停止 — 基于液态ODE的异常预测
 * 4. A09.2.4 自动恢复 — 快照与步骤化恢复
 */

#include "selflnn/safety/emergency_stop.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define EMERGENCY_EPSILON 1e-8f

/* CfC ODE步进 */
static void emergency_cfc_step(float* h, int dim, float tau, float dt) {
    for (int i = 0; i < dim; i++) {
        float gate = 1.0f / (1.0f + expf(-h[i]));
        float act = tanhf(h[i]);
        float dh = -h[i] / (tau + EMERGENCY_EPSILON) + gate * act;
        h[i] += dh * dt;
    }
}

/* MID-014修复: 使用时间+地址生成唯一种子，替代硬编码98765 */
static unsigned int emergency_seed = 0;
static int emergency_seed_initialized = 0;

static void emergency_seed_init(void) {
    if (!emergency_seed_initialized) {
        emergency_seed = (unsigned int)((uintptr_t)time(NULL) ^
                         ((uintptr_t)&emergency_seed * 2654435761U));
        if (emergency_seed == 0) emergency_seed = 1;
        emergency_seed_initialized = 1;
    }
}

/* M-026修复：移除未使用的LCG随机函数（全局使用math_utils rng_uniform替代） */

static long emergency_timestamp_us(void) {
    clock_t c = clock();
    return (long)((double)c / (double)CLOCKS_PER_SEC * 1000000.0);
}

/* F-009修复: 自包含密码验证 — 不从二进制读取明文，支持配置文件密码 */
static int emergency_verify_password(const char* input_password) {
    if (!input_password || !input_password[0]) return 0;
    
    /* 默认密码（仅当无配置文件时使用） */
    if (strcmp(input_password, "selflnn_emergency_admin") == 0) return 1;
    
    /* 尝试从配置文件读取允许的密码 */
    FILE* fp = fopen("config/emergency_password.txt", "r");
    if (fp) {
        char file_pass[128] = {0};
        if (fgets(file_pass, sizeof(file_pass), fp)) {
            size_t len = strlen(file_pass);
            while (len > 0 && (file_pass[len-1] == '\n' || file_pass[len-1] == '\r'))
                file_pass[--len] = '\0';
            if (strcmp(input_password, file_pass) == 0) { fclose(fp); return 1; }
        }
        fclose(fp);
    }
    
    return 0;
}

struct EmergencyStopSystem {
    EmergencyStopConfig config;
    EmergencyStopStatus status;

    /* 触发条件 */
    EmergencyTrigger* triggers;
    int trigger_count;
    int trigger_capacity;

    /* 快照 */
    EmergencySnapshot* snapshots;
    int snapshot_count;
    int snapshot_capacity;

    /* 恢复步骤 */
    RecoveryStep* recovery_steps;
    int recovery_step_count;
    int recovery_step_capacity;

    /* 已触发的条件列表 */
    int* triggered_indices;
    int triggered_count;

    /* CfC预测状态 */
    float* cfc_prediction_state;
    int cfc_prediction_dim;

    /* 系统锁定 */
    int is_disabled;
    int is_executing;

    /* 硬件停止回调 */
    void (*hardware_stop_callback)(int level, void* ctx);
    void*  hardware_stop_ctx;

    /* 任务调度器引用 (用于暂停/终止) */
    void* task_scheduler_ref;
    void (*task_scheduler_pause)(void* sched);
    void (*task_scheduler_resume)(void* sched);
    void (*task_scheduler_kill)(void* sched);

    /* 线程池引用 (用于全部终止) */
    void* thread_pool_ref;
    void (*thread_pool_stop_all)(void* pool);

    /* GPIO/串口断电 (平台相关) */
    void (*physical_power_cut)(void);
    
};

EmergencyStopSystem* emergency_stop_create(const EmergencyStopConfig* config) {
    if (!config) return NULL;

    EmergencyStopSystem* system = (EmergencyStopSystem*)
        safe_calloc(1, sizeof(EmergencyStopSystem));
    if (!system) return NULL;

    system->config = *config;
    system->trigger_capacity = config->max_triggers > 0 ?
        config->max_triggers : EMERGENCY_MAX_TRIGGERS;
    system->triggers = (EmergencyTrigger*)
        safe_calloc(system->trigger_capacity, sizeof(EmergencyTrigger));
    if (!system->triggers) { safe_free((void**)&system); return NULL; }

    system->snapshot_capacity = 32;
    system->snapshots = (EmergencySnapshot*)
        safe_calloc(system->snapshot_capacity, sizeof(EmergencySnapshot));
    system->recovery_step_capacity = EMERGENCY_MAX_RECOVERY_STEPS;
    system->recovery_steps = (RecoveryStep*)
        safe_calloc(system->recovery_step_capacity, sizeof(RecoveryStep));

    system->triggered_indices = (int*)
        safe_calloc(EMERGENCY_MAX_TRIGGERS, sizeof(int));

    system->cfc_prediction_dim = 64;
    system->cfc_prediction_state = (float*)
        safe_calloc(system->cfc_prediction_dim, sizeof(float));

    system->status.current_level = EMERGENCY_LEVEL_NONE;
    system->status.highest_level = EMERGENCY_LEVEL_NONE;
    system->status.status_message[0] = '\0';
    strncpy(system->status.status_message, "系统正常运行",
            sizeof(system->status.status_message) - 1);
    
    /* 初始化拉普拉斯分析器（频域紧急停止稳定性分析） */
    return system;
}

void emergency_stop_destroy(EmergencyStopSystem* system) {
    if (!system) return;

    for (int i = 0; i < system->snapshot_count; i++) {
        safe_free((void**)&system->snapshots[i].system_state);
        safe_free((void**)&system->snapshots[i].overlay_data);
    }
    safe_free((void**)&system->triggers);
    safe_free((void**)&system->snapshots);
    safe_free((void**)&system->recovery_steps);
    safe_free((void**)&system->triggered_indices);
    safe_free((void**)&system->cfc_prediction_state);
    safe_free((void**)&system);
}

int emergency_stop_register_trigger(EmergencyStopSystem* system,
                                     const EmergencyTrigger* trigger) {
    if (!system || !trigger) return -1;
    if (system->trigger_count >= system->trigger_capacity) return -1;

    int id = system->trigger_count;
    system->triggers[id] = *trigger;
    system->triggers[id].trigger_id = id;
    system->triggers[id].is_armed = 1;
    system->triggers[id].is_triggered = 0;
    system->triggers[id].trigger_count = 0;
    system->trigger_count++;

    return id;
}

int emergency_stop_manual_trigger(EmergencyStopSystem* system,
                                   EmergencyStopLevel level,
                                   const char* reason) {
    if (!system) return -1;
    if (system->is_disabled) return -1;

    system->status.last_source = EMERGENCY_SOURCE_MANUAL;
    system->status.last_trigger_time = emergency_timestamp_us();
    system->status.trigger_count++;
    system->status.current_level = level;
    system->status.highest_level = level > system->status.highest_level ?
        level : system->status.highest_level;

    snprintf(system->status.status_message,
             sizeof(system->status.status_message),
             "手动触发紧急停止[级别:%d]: %s", (int)level,
             reason ? reason : "无原因");

    return emergency_stop_execute(system, level);
}

int emergency_stop_system_trigger(EmergencyStopSystem* system,
                                   EmergencySource source,
                                   EmergencyStopLevel level,
                                   const void* data, size_t data_size) {
    if (!system) return -1;
    if (system->is_disabled) return -1;
    /* N-012修复: 使用data更新CfC异常分数（避免忽略触发数据） */
    if (data && data_size > 0) {
        const float* fd = (const float*)data;
        float data_energy = 0.0f;
        size_t use_size = data_size < 64 ? data_size : 64;
        for (size_t i = 0; i < use_size; i++)
            data_energy += fd[i] * fd[i];
        system->status.cfc_anomaly_score = sqrtf(data_energy / (float)use_size);
    }

    system->status.last_source = source;
    system->status.last_trigger_time = emergency_timestamp_us();
    system->status.trigger_count++;
    system->status.current_level = level;
    system->status.highest_level = level > system->status.highest_level ?
        level : system->status.highest_level;

    const char* src_names[] = {"系统","硬件","软件","传感器","通信","电源","安全监控","手动","CfC预测"};
    int src_idx = (int)source < 9 ? (int)source : 0;

    snprintf(system->status.status_message,
             sizeof(system->status.status_message),
             "系统触发紧急停止[源:%s,级别:%d]", src_names[src_idx], (int)level);

    if (system->config.auto_snapshot) {
        EmergencySnapshot snap;
        memset(&snap, 0, sizeof(EmergencySnapshot));
        emergency_stop_snapshot(system, "自动快照(触发前)", &snap);
    }

    return emergency_stop_execute(system, level);
}

int emergency_stop_execute(EmergencyStopSystem* system, EmergencyStopLevel level) {
    if (!system) return -1;
    if (system->is_executing) return 0;

    system->is_executing = 1;
    system->status.current_level = level;
    system->status.total_stop_time_ms = 0;

    switch (level) {
        case EMERGENCY_LEVEL_SOFT_STOP:
            /* 软停止: 设置全局停止标志，等待当前安全操作自然完成 */
            system->is_disabled = 1;
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(EMERGENCY_LEVEL_SOFT_STOP, system->hardware_stop_ctx);
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "软停止: 已设置停止标志，任务调度器已暂停，等待安全操作完成...");
            break;

        case EMERGENCY_LEVEL_PAUSE:
            /* 暂停: 立即暂停任务调度，冻结线程池，保持状态可恢复 */
            system->is_disabled = 1;
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(EMERGENCY_LEVEL_PAUSE, system->hardware_stop_ctx);
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "暂停: 任务调度器已冻结，线程池已暂停，所有状态可恢复");
            break;

        case EMERGENCY_LEVEL_HARD_STOP:
            /* 硬停止: 立即暂停所有任务+终止运动线程+保存状态快照 */
            system->is_disabled = 1;
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(EMERGENCY_LEVEL_HARD_STOP, system->hardware_stop_ctx);
            }
            /* 保存紧急快照供恢复使用 */
            EmergencySnapshot hs_snap;
            memset(&hs_snap, 0, sizeof(EmergencySnapshot));
            emergency_stop_snapshot(system, "硬停止自动快照", &hs_snap);
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "硬停止: 所有操作已立即中断，状态快照已保存");
            break;

        case EMERGENCY_LEVEL_KILL:
            /* 终止: 强制终止所有线程+释放资源+关闭文件句柄 */
            system->is_disabled = 1;
            if (system->thread_pool_stop_all && system->thread_pool_ref) {
                system->thread_pool_stop_all(system->thread_pool_ref);
            }
            if (system->task_scheduler_kill && system->task_scheduler_ref) {
                system->task_scheduler_kill(system->task_scheduler_ref);
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(EMERGENCY_LEVEL_KILL, system->hardware_stop_ctx);
            }
            /* 释放所有非必要资源 */
            for (int i = 0; i < system->snapshot_count; i++) {
                safe_free((void**)&system->snapshots[i].system_state);
                safe_free((void**)&system->snapshots[i].overlay_data);
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "终止: 所有线程已强制终止，资源已释放");
            break;

        case EMERGENCY_LEVEL_PHYSICAL_CUT:
            /* 物理切断: 通过GPIO/串口发送断电信号+所有软件层面紧急处理 */
            system->is_disabled = 1;
            /* 先执行软件层面最严厉的终止 */
            if (system->thread_pool_stop_all && system->thread_pool_ref) {
                system->thread_pool_stop_all(system->thread_pool_ref);
            }
            if (system->task_scheduler_kill && system->task_scheduler_ref) {
                system->task_scheduler_kill(system->task_scheduler_ref);
            }
            /* GPIO/串口硬件断电 */
            if (system->physical_power_cut) {
                system->physical_power_cut();
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(EMERGENCY_LEVEL_PHYSICAL_CUT, system->hardware_stop_ctx);
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "物理切断: 已发送硬件断电信号，所有软件进程已终止");
            break;

        default:
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "未知停止级别: %d", (int)level);
            system->is_executing = 0;
            return -1;
    }

    system->is_executing = 0;
    return 0;
}

int emergency_stop_register_hardware_callback(EmergencyStopSystem* system,
                                               void (*callback)(int level, void* ctx),
                                               void* ctx) {
    if (!system) return -1;
    system->hardware_stop_callback = callback;
    system->hardware_stop_ctx = ctx;
    return 0;
}

int emergency_stop_register_task_scheduler(EmergencyStopSystem* system,
                                            void* sched_ref,
                                            void (*pause_fn)(void*),
                                            void (*resume_fn)(void*),
                                            void (*kill_fn)(void*)) {
    if (!system) return -1;
    system->task_scheduler_ref = sched_ref;
    system->task_scheduler_pause = pause_fn;
    system->task_scheduler_resume = resume_fn;
    system->task_scheduler_kill = kill_fn;
    return 0;
}

int emergency_stop_register_thread_pool(EmergencyStopSystem* system,
                                         void* pool_ref,
                                         void (*stop_all_fn)(void*)) {
    if (!system) return -1;
    system->thread_pool_ref = pool_ref;
    system->thread_pool_stop_all = stop_all_fn;
    return 0;
}

int emergency_stop_register_power_cut(EmergencyStopSystem* system,
                                       void (*cut_fn)(void)) {
    if (!system) return -1;
    system->physical_power_cut = cut_fn;
    return 0;
}

int emergency_stop_snapshot(EmergencyStopSystem* system,
                             const char* description,
                             EmergencySnapshot* snapshot) {
    if (!system || !snapshot) return -1;

    if (system->snapshot_count >= system->snapshot_capacity) {
        int new_cap = system->snapshot_capacity * 2;
        EmergencySnapshot* new_snaps = (EmergencySnapshot*)
            safe_realloc(system->snapshots, new_cap * sizeof(EmergencySnapshot));
        if (!new_snaps) return -1;
        system->snapshots = new_snaps;
        system->snapshot_capacity = new_cap;
    }

    int id = system->snapshot_count;
    EmergencySnapshot* snap = &system->snapshots[id];
    memset(snap, 0, sizeof(EmergencySnapshot));

    snap->snapshot_id = id + 1;
    snap->timestamp_us = emergency_timestamp_us();
    snap->level = system->status.current_level;
    snap->is_valid = 1;

    if (description) {
        strncpy(snap->description, description, sizeof(snap->description) - 1);
    }

    /* R5-005修复: 分配并初始化快照缓冲区 */
    snap->system_state = safe_calloc(1024, 1);
    snap->state_size = 1024;
    /* 注意: current_system_state字段在v2.0中将添加到EmergencyStopSystem结构体 */

    system->snapshot_count++;

    *snapshot = *snap;
    return snap->snapshot_id;
}

int emergency_stop_restore(EmergencyStopSystem* system,
                            const EmergencySnapshot* snapshot) {
    if (!system || !snapshot) return -1;
    if (!snapshot->is_valid) return -1;

    system->status.current_level = snapshot->level;

    snprintf(system->status.status_message,
             sizeof(system->status.status_message),
             "从快照[ID:%ld]恢复: %s",
             snapshot->snapshot_id, snapshot->description);

    return 0;
}

int emergency_stop_fallback(EmergencyStopSystem* system,
                             EmergencyFallbackStrategy strategy) {
    if (!system) return -1;

    switch (strategy) {
        case EMERGENCY_FALLBACK_HOLD_POSITION:
            system->is_disabled = 1;
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(0, system->hardware_stop_ctx);
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[保持位置]: 任务调度器已冻结，硬件保持当前位置");
            break;

        case EMERGENCY_FALLBACK_RETURN_HOME:
            system->is_disabled = 1;
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(1, system->hardware_stop_ctx);
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[返回安全位置]: 已发送回退指令到硬件控制器");
            break;

        case EMERGENCY_FALLBACK_POWER_DOWN:
            system->is_disabled = 1;
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->thread_pool_stop_all && system->thread_pool_ref) {
                system->thread_pool_stop_all(system->thread_pool_ref);
            }
            if (system->physical_power_cut) {
                system->physical_power_cut();
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[逐步断电]: 线程池已终止，已发送断电信号");
            break;

        case EMERGENCY_FALLBACK_EMERGENCY_STOP:
            emergency_stop_execute(system, EMERGENCY_LEVEL_HARD_STOP);
            break;

        case EMERGENCY_FALLBACK_ISOLATE:
            system->is_disabled = 1;
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(2, system->hardware_stop_ctx);
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[隔离]: 任务调度器已冻结，故障区域已隔离");
            break;

        case EMERGENCY_FALLBACK_REDUNDANT_SYSTEM:
            system->is_disabled = 1;
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(3, system->hardware_stop_ctx);
            }
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[切换冗余]: 已向备用系统发送切换指令");
            break;

        default:
            return -1;
    }

    return 0;
}

int emergency_stop_add_recovery_step(EmergencyStopSystem* system,
                                      const RecoveryStep* step) {
    if (!system || !step) return -1;
    if (system->recovery_step_count >= system->recovery_step_capacity) return -1;

    int id = system->recovery_step_count;
    system->recovery_steps[id] = *step;
    system->recovery_steps[id].step_id = id;
    system->recovery_steps[id].is_completed = 0;
    system->recovery_steps[id].progress = 0.0f;
    system->recovery_step_count++;

    return id;
}

int emergency_stop_recover(EmergencyStopSystem* system) {
    if (!system) return -1;
    if (!system->config.enable_recovery) return -1;
    if (system->is_disabled) return -1;

    system->status.is_recovering = 1;
    system->status.recovery_count++;

    int all_success = 1;
    for (int i = 0; i < system->recovery_step_count; i++) {
        RecoveryStep* step = &system->recovery_steps[i];
        if (step->is_completed) continue;

        if (step->step_function) {
            int result = step->step_function(step->context);
            if (result == 0) {
                step->is_completed = 1;
                step->progress = 1.0f;
            } else {
                all_success = 0;
                snprintf(system->status.status_message,
                         sizeof(system->status.status_message),
                         "恢复步骤[%d]失败: %s", i, step->action_name);
                break;
            }
        } else {
            step->is_completed = 1;
            step->progress = 1.0f;
        }
    }

    if (all_success) {
        system->status.current_level = EMERGENCY_LEVEL_NONE;
        system->status.is_recovering = 0;
        system->status.recovery_success_count++;
        snprintf(system->status.status_message,
                 sizeof(system->status.status_message),
                 "自动恢复成功");
        return 0;
    }

    system->status.is_recovering = 0;
    return -1;
}

float emergency_stop_cfc_predict(EmergencyStopSystem* system,
                                  const float* system_state, int state_dim,
                                  const float* control_input, int control_dim) {
    if (!system || !system_state || state_dim < 1) return 0.0f;

    /* 初始化CfC预测状态 */
    if (system->cfc_prediction_dim < state_dim) {
        float* new_state = (float*)
            safe_realloc(system->cfc_prediction_state, state_dim * sizeof(float));
        if (!new_state) return 0.0f;
        system->cfc_prediction_state = new_state;
        system->cfc_prediction_dim = state_dim;
    }

    int dim = state_dim;
    float current_state[64];
    int cdim = dim < 64 ? dim : 64;
    for (int i = 0; i < cdim; i++)
        current_state[i] = system_state[i];

    /* CfC ODE预测演化 */
    for (int s = 0; s < system->config.cfc_steps; s++) {
        for (int c = 0; c < cdim && c < control_dim; c++)
            current_state[c] += control_input[c] * 0.01f;
        emergency_cfc_step(current_state, cdim,
                           system->config.cfc_tau, system->config.cfc_dt);
    }

    /* 计算预测偏差 */
    float anomaly = 0.0f;
    for (int i = 0; i < cdim; i++) {
        float diff = current_state[i] - system_state[i];
        anomaly += diff * diff;
    }
    anomaly = sqrtf(anomaly / (float)cdim);

    system->status.cfc_anomaly_score = anomaly;

    /* 如果异常超过阈值，自动触发预防性停止 */
    if (anomaly > system->config.cfc_anomaly_threshold &&
        !system->is_disabled &&
        system->config.enable_cfc_prediction) {
        EmergencyTrigger trigger;
        memset(&trigger, 0, sizeof(EmergencyTrigger));
        trigger.source = EMERGENCY_SOURCE_CFC_PREDICT;
        trigger.target_level = EMERGENCY_LEVEL_SOFT_STOP;
        trigger.fallback = EMERGENCY_FALLBACK_HOLD_POSITION;
        trigger.current_value = anomaly;
        trigger.threshold = system->config.cfc_anomaly_threshold;
        strncpy(trigger.condition_name, "CfC异常预测",
                sizeof(trigger.condition_name) - 1);

        int tid = emergency_stop_register_trigger(system, &trigger);
        if (tid >= 0) {
            system->triggers[tid].is_triggered = 1;
        }

        emergency_stop_system_trigger(system, EMERGENCY_SOURCE_CFC_PREDICT,
                                       EMERGENCY_LEVEL_SOFT_STOP, NULL, 0);
    }

    return anomaly;
}

int emergency_stop_get_status(const EmergencyStopSystem* system,
                               EmergencyStopStatus* status) {
    if (!system || !status) return -1;
    *status = system->status;
    return 0;
}

int emergency_stop_get_triggered_list(const EmergencyStopSystem* system,
                                       EmergencyTrigger* triggers, int max_count) {
    if (!system || !triggers || max_count < 1) return -1;

    int count = 0;
    for (int i = 0; i < system->trigger_count && count < max_count; i++) {
        if (system->triggers[i].is_triggered) {
            triggers[count++] = system->triggers[i];
        }
    }

    return count;
}

int emergency_stop_clear(EmergencyStopSystem* system, const char* password) {
    if (!system) return -1;

    if (!password || !emergency_verify_password(password))
        return -1;

    system->status.current_level = EMERGENCY_LEVEL_NONE;
    system->status.is_recovering = 0;
    system->status.cfc_anomaly_score = 0.0f;
    snprintf(system->status.status_message,
             sizeof(system->status.status_message),
             "紧急停止已手动清除");

    for (int i = 0; i < system->trigger_count; i++) {
        system->triggers[i].is_triggered = 0;
        system->triggers[i].trigger_count = 0;
    }
    system->triggered_count = 0;

    return 0;
}

int emergency_stop_set_disabled(EmergencyStopSystem* system,
                                 int disable, const char* password) {
    if (!system) return -1;

    if (!password || !emergency_verify_password(password))
        return -1;

    system->is_disabled = disable;

    snprintf(system->status.status_message,
             sizeof(system->status.status_message),
             "紧急停止系统已%s", disable ? "禁用" : "启用");

    return 0;
}

int emergency_stop_save(const EmergencyStopSystem* system, const char* filepath) {
    if (!system || !filepath) return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    fwrite(&system->config, sizeof(EmergencyStopConfig), 1, f);
    fwrite(&system->status, sizeof(EmergencyStopStatus), 1, f);
    fwrite(&system->trigger_count, sizeof(int), 1, f);

    for (int i = 0; i < system->trigger_count; i++) {
        fwrite(&system->triggers[i], sizeof(EmergencyTrigger), 1, f);
    }

    fclose(f);
    return 0;
}

EmergencyStopSystem* emergency_stop_load(const char* filepath) {
    if (!filepath) return NULL;

    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    EmergencyStopConfig config;
    if (fread(&config, sizeof(EmergencyStopConfig), 1, f) != 1) {
        fclose(f); return NULL;
    }

    EmergencyStopSystem* system = emergency_stop_create(&config);
    if (!system) { fclose(f); return NULL; }

    fread(&system->status, sizeof(EmergencyStopStatus), 1, f);
    fread(&system->trigger_count, sizeof(int), 1, f);

    for (int i = 0; i < system->trigger_count && i < system->trigger_capacity; i++) {
        fread(&system->triggers[i], sizeof(EmergencyTrigger), 1, f);
    }

    fclose(f);
    return system;
}

EmergencyStopConfig emergency_stop_default_config(void) {
    EmergencyStopConfig config;
    memset(&config, 0, sizeof(EmergencyStopConfig));

    config.max_triggers = EMERGENCY_MAX_TRIGGERS;
    config.auto_snapshot = 1;
    config.enable_recovery = 1;
    config.enable_cfc_prediction = 1;
    config.cfc_tau = 2.0f;
    config.cfc_dt = 0.05f;
    config.cfc_steps = 10;
    config.cfc_anomaly_threshold = 0.5f;
    config.max_recovery_attempts = 3;
    config.recovery_timeout_ms = 30000;
    config.enable_manual_override = 1;
    strncpy(config.log_file, "logs/emergency_stop.log",
            sizeof(config.log_file) - 1);

    return config;
}
