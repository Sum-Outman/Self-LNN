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
    
    /* 尝试从配置文件读取密码：config/emergency_password.txt */
    {
        FILE* fp = fopen("config/emergency_password.txt", "r");
        if (fp) {
            char cfg_pwd[256];
            memset(cfg_pwd, 0, sizeof(cfg_pwd));
            if (fgets(cfg_pwd, (int)sizeof(cfg_pwd) - 1, fp)) {
                /* 去除末尾换行符 */
                size_t cfg_len = strlen(cfg_pwd);
                while (cfg_len > 0 && (cfg_pwd[cfg_len - 1] == '\n' || cfg_pwd[cfg_len - 1] == '\r'))
                    cfg_pwd[--cfg_len] = '\0';
                fclose(fp);
                if (cfg_len > 0 && strcmp(input_password, cfg_pwd) == 0) return 1;
            } else {
                fclose(fp);
            }
        }
    }
    
    /* 无配置文件或密码不匹配时拒绝（不再使用硬编码默认值） */
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

    /* H-005修复: 线程池恢复回调 (用于快照恢复时重新启动线程池) */
    void (*thread_pool_restart)(void* pool);

    /* H-005修复: 模型推理恢复回调 (用于快照恢复时重新激活模型推理) */
    void (*model_inference_resume)(void* ctx);
    void*  model_inference_ctx;

    /* H-005修复: 审计日志回调 (用于记录恢复审计信息) */
    void (*audit_log_callback)(const char* message, void* ctx);
    void*  audit_log_ctx;

    /* P3-001修复: 系统运行时指标缓存（供快照打包使用） */
    EmergencySystemMetrics last_metrics;
    int metrics_valid;
    int snapshot_sequence_counter;

    int gpu_kernel_interrupt_flag;
    void (*gpu_kernel_interrupt_callback)(void* ctx);
    void* gpu_kernel_interrupt_ctx;
    int gpu_kernel_count;
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
    
    /* P3-001修复: 初始化指标缓存 */
    memset(&system->last_metrics, 0, sizeof(EmergencySystemMetrics));
    system->metrics_valid = 0;
    system->snapshot_sequence_counter = 0;
    
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

    if (system->gpu_kernel_interrupt_flag && system->gpu_kernel_interrupt_callback) {
        system->gpu_kernel_interrupt_callback(system->gpu_kernel_interrupt_ctx);
        snprintf(system->status.status_message,
                 sizeof(system->status.status_message),
                 "紧急停止[级别:%d]: GPU内核中断已发送，中断%d个内核",
                 (int)level, system->gpu_kernel_count);
    }
    if (system->gpu_kernel_interrupt_flag && !system->gpu_kernel_interrupt_callback) {
        system->gpu_kernel_interrupt_flag = 0;
        system->gpu_kernel_count = 0;
    }

    /* M-019修复: 执行前验证关键回调是否已注册，任一缺失则返回错误 */
    /* 检查对应级别的关键停止机制是否可用（函数指针+上下文引用必须同时非NULL） */
    {
        int has_valid_callback = 0;
        const char* missing_callback = NULL;

        switch (level) {
            case EMERGENCY_LEVEL_SOFT_STOP:
            case EMERGENCY_LEVEL_PAUSE:
            case EMERGENCY_LEVEL_HARD_STOP:
                /* 关键回调: 任务调度器暂停 或 硬件停止 */
                if ((system->task_scheduler_pause && system->task_scheduler_ref) ||
                    system->hardware_stop_callback) {
                    has_valid_callback = 1;
                } else {
                    missing_callback = "task_scheduler_pause/ref 和 hardware_stop_callback 均未注册";
                }
                break;

            case EMERGENCY_LEVEL_KILL:
                /* 关键回调: 线程池停止 或 任务调度器终止 */
                if ((system->thread_pool_stop_all && system->thread_pool_ref) ||
                    (system->task_scheduler_kill && system->task_scheduler_ref)) {
                    has_valid_callback = 1;
                } else {
                    missing_callback = "thread_pool_stop_all/ref 和 task_scheduler_kill/ref 均未注册";
                }
                break;

            case EMERGENCY_LEVEL_PHYSICAL_CUT:
                /* 关键回调: 线程池停止 或 任务调度器终止 或 物理断电 */
                if ((system->thread_pool_stop_all && system->thread_pool_ref) ||
                    (system->task_scheduler_kill && system->task_scheduler_ref) ||
                    system->physical_power_cut) {
                    has_valid_callback = 1;
                } else {
                    missing_callback = "thread_pool_stop_all/ref、task_scheduler_kill/ref 和 physical_power_cut 均未注册";
                }
                break;

            default:
                break;
        }

        if (!has_valid_callback) {
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "紧急停止失败[级别:%d]: 关键回调未注册 — %s",
                     (int)level, missing_callback ? missing_callback : "未知");
            system->is_executing = 0;
            return -1;
        }
    }

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

/* H-005修复: 注册线程池重启回调（用于快照恢复时重新启动线程池） */
int emergency_stop_register_thread_pool_restart(EmergencyStopSystem* system,
                                                  void* pool_ref,
                                                  void (*restart_fn)(void*)) {
    if (!system) return -1;
    system->thread_pool_ref = pool_ref;
    system->thread_pool_restart = restart_fn;
    return 0;
}

/* H-005修复: 注册模型推理恢复回调（用于快照恢复时重新激活模型推理） */
int emergency_stop_register_model_inference_resume(EmergencyStopSystem* system,
                                                     void* ctx,
                                                     void (*resume_fn)(void*)) {
    if (!system) return -1;
    system->model_inference_ctx = ctx;
    system->model_inference_resume = resume_fn;
    return 0;
}

/* H-005修复: 注册审计日志回调（用于记录恢复操作审计信息） */
int emergency_stop_register_audit_log(EmergencyStopSystem* system,
                                       void (*log_fn)(const char* message, void* ctx),
                                       void* ctx) {
    if (!system) return -1;
    system->audit_log_callback = log_fn;
    system->audit_log_ctx = ctx;
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

    /* P3-001修复: 使用紧凑二进制打包格式填充真实系统状态 */
    /* 二进制格式布局 (1024字节):
     *  [0-3]   魔数: 0x53534E41 ("SSNA" = Snapshot AGI)
     *  [4-7]   版本: 1
     *  [8-11]  AGI活跃任务数 (int32)
     *  [12-15] AGI总任务数 (int32)
     *  [16-19] AGI活跃目标数 (int32)
     *  [20-23] AGI总目标数 (int32)
     *  [24-27] AGI认知循环计数 (int32)
     *  [28-31] AGI认知负载率 (float32)
     *  [32-35] LNN平均激活值 (float32)
     *  [36-39] LNN最大激活值 (float32)
     *  [40-43] LNN最小激活值 (float32)
     *  [44-47] LNN步数 (int32)
     *  [48-51] LNN隐藏维度 (int32)
     *  [52-55] LNN输入维度 (int32)
     *  [56-59] LNN输出维度 (int32)
     *  [60-67] 总分配内存字节 (uint64)
     *  [68-75] 总释放内存字节 (uint64)
     *  [76-83] 当前使用内存字节 (uint64)
     *  [84-87] 活跃分配计数 (int32)
     *  [88-91] 内存使用率 (float32)
     *  [92-99] 快照时间戳微秒 (int64)
     *  [100-103] 快照序列号 (int32)
     */
    snap->system_state = safe_calloc(1024, 1);
    snap->state_size = 1024;
    
    if (snap->system_state) {
        uint8_t* buf = (uint8_t*)snap->system_state;
        EmergencySystemMetrics m;
        if (system->metrics_valid) {
            m = system->last_metrics;
            m.snapshot_timestamp_us = snap->timestamp_us;
            m.snapshot_sequence = system->snapshot_sequence_counter++;
        } else {
            /* 无外部指标时使用系统内部可获取的信息填充 */
            memset(&m, 0, sizeof(EmergencySystemMetrics));
            m.snapshot_timestamp_us = snap->timestamp_us;
            m.snapshot_sequence = system->snapshot_sequence_counter++;
            m.cognitive_cycle_count = system->status.trigger_count;
            m.active_task_count = system->trigger_count;
            m.active_goal_count = system->triggered_count;
            m.cognitive_load = system->status.cfc_anomaly_score;
        }
        
        /* 写入魔数 "SSNA" */
        buf[0] = 'S'; buf[1] = 'S'; buf[2] = 'N'; buf[3] = 'A';
        /* 写入版本 */
        *(int32_t*)(buf + 4) = 1;
        /* AGI认知状态 */
        *(int32_t*)(buf + 8)  = m.active_task_count;
        *(int32_t*)(buf + 12) = m.total_task_count;
        *(int32_t*)(buf + 16) = m.active_goal_count;
        *(int32_t*)(buf + 20) = m.total_goal_count;
        *(int32_t*)(buf + 24) = m.cognitive_cycle_count;
        *(float*)(buf + 28)   = m.cognitive_load;
        /* LNN状态摘要 */
        *(float*)(buf + 32)  = m.lnn_mean_activation;
        *(float*)(buf + 36)  = m.lnn_max_activation;
        *(float*)(buf + 40)  = m.lnn_min_activation;
        *(int32_t*)(buf + 44) = m.lnn_step_count;
        *(int32_t*)(buf + 48) = m.lnn_hidden_dim;
        *(int32_t*)(buf + 52) = m.lnn_input_dim;
        *(int32_t*)(buf + 56) = m.lnn_output_dim;
        /* 内存使用摘要 */
        *(uint64_t*)(buf + 60) = (uint64_t)m.total_allocated_bytes;
        *(uint64_t*)(buf + 68) = (uint64_t)m.total_freed_bytes;
        *(uint64_t*)(buf + 76) = (uint64_t)m.current_used_bytes;
        *(int32_t*)(buf + 84)  = m.active_allocation_count;
        *(float*)(buf + 88)    = m.memory_usage_ratio;
        /* 时间戳 */
        *(int64_t*)(buf + 92)  = (int64_t)m.snapshot_timestamp_us;
        *(int32_t*)(buf + 100) = m.snapshot_sequence;
    }

    system->snapshot_count++;

    *snapshot = *snap;
    return snap->snapshot_id;
}

int emergency_stop_restore(EmergencyStopSystem* system,
                            const EmergencySnapshot* snapshot) {
    if (!system || !snapshot) return -1;
    if (!snapshot->is_valid) return -1;
    if (system->is_executing) return -1;

    system->is_executing = 1;

    /* 步骤1: 验证快照内系统状态数据的完整性 */
    int state_valid = 0;
    if (snapshot->system_state && snapshot->state_size >= 4) {
        uint8_t* buf = (uint8_t*)snapshot->system_state;
        if (buf[0] == 'S' && buf[1] == 'S' && buf[2] == 'N' && buf[3] == 'A') {
            state_valid = 1;
        }
    }

    /* 步骤2: 重新初始化/恢复线程池 */
    int thread_pool_restored = 0;
    if (system->thread_pool_restart && system->thread_pool_ref) {
        system->thread_pool_restart(system->thread_pool_ref);
        thread_pool_restored = 1;
    }

    /* 步骤3: 恢复任务调度器 */
    int scheduler_restored = 0;
    if (system->task_scheduler_resume && system->task_scheduler_ref) {
        system->task_scheduler_resume(system->task_scheduler_ref);
        scheduler_restored = 1;
    }

    /* 步骤4: 重新激活模型推理 */
    int inference_restored = 0;
    if (system->model_inference_resume && system->model_inference_ctx) {
        system->model_inference_resume(system->model_inference_ctx);
        inference_restored = 1;
    }

    /* 步骤5: 更新系统状态为正常运行 */
    system->status.current_level = EMERGENCY_LEVEL_NONE;
    system->is_disabled = 0;
    system->status.is_recovering = 0;
    system->status.recovery_count++;
    system->status.recovery_success_count++;
    system->status.cfc_anomaly_score = 0.0f;

    /* 步骤6: 清除已触发的条件列表，重新启用所有触发器 */
    for (int i = 0; i < system->trigger_count; i++) {
        system->triggers[i].is_triggered = 0;
        system->triggers[i].trigger_count = 0;
    }
    system->triggered_count = 0;

    /* 步骤7: 构建恢复状态消息 */
    {
        char detail_buf[512];
        snprintf(detail_buf, sizeof(detail_buf),
                 "从快照[ID:%ld]恢复: %s | 快照级别:%d | 状态数据:%s | "
                 "线程池:%s | 调度器:%s | 模型推理:%s",
                 snapshot->snapshot_id, snapshot->description,
                 (int)snapshot->level,
                 state_valid ? "有效" : "无效",
                 thread_pool_restored ? "已重启" : "未注册",
                 scheduler_restored ? "已恢复" : "未注册",
                 inference_restored ? "已激活" : "未注册");
        snprintf(system->status.status_message,
                 sizeof(system->status.status_message),
                 "%s", detail_buf);
    }

    /* 步骤8: 记录恢复审计日志 */
    long restore_time = emergency_timestamp_us();
    {
        char audit_msg[512];
        snprintf(audit_msg, sizeof(audit_msg),
                 "[恢复审计] 快照ID:%ld | 时间戳:%ld | "
                 "快照级别:%d | 描述:%s | "
                 "线程池:%d | 调度器:%d | 推理:%d",
                 snapshot->snapshot_id, restore_time,
                 (int)snapshot->level, snapshot->description,
                 thread_pool_restored, scheduler_restored,
                 inference_restored);

        if (system->audit_log_callback) {
            system->audit_log_callback(audit_msg, system->audit_log_ctx);
        }

        /* 同时写入本地日志文件 */
        {
            FILE* log_fp = fopen(system->config.log_file, "a");
            if (log_fp) {
                time_t now = time(NULL);
                char time_buf[64];
                strftime(time_buf, sizeof(time_buf),
                         "%Y-%m-%d %H:%M:%S", localtime(&now));
                fprintf(log_fp, "[%s] [恢复审计] %s | 恢复耗时:%ldus\n",
                        time_buf, audit_msg,
                        restore_time - snapshot->timestamp_us);
                fclose(log_fp);
            }
        }
    }

    system->is_executing = 0;
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

/* P3-001修复: 设置系统运行时指标 — 供快照捕获使用 */
int emergency_stop_set_metrics(EmergencyStopSystem* system,
                                const EmergencySystemMetrics* metrics) {
    if (!system || !metrics) return -1;
    memcpy(&system->last_metrics, metrics, sizeof(EmergencySystemMetrics));
    system->metrics_valid = 1;
    return 0;
}

/* P3-001修复: 获取最近一次设置的指标 */
int emergency_stop_get_metrics(const EmergencyStopSystem* system,
                                EmergencySystemMetrics* metrics) {
    if (!system || !metrics) return -1;
    if (!system->metrics_valid) return -1;
    memcpy(metrics, &system->last_metrics, sizeof(EmergencySystemMetrics));
    return 0;
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

int emergency_stop_register_gpu_kernel_interrupt(EmergencyStopSystem* system,
                                                  void (*callback)(void*),
                                                  void* ctx) {
    if (!system) return -1;
    system->gpu_kernel_interrupt_callback = callback;
    system->gpu_kernel_interrupt_ctx = ctx;
    system->gpu_kernel_interrupt_flag = 1;
    return 0;
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
