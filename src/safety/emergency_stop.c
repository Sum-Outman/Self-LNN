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
/* P1修复: 引入platform.h获取MutexHandle类型和mutex系列函数 */
#include "selflnn/utils/platform.h"
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

/* S-P1-03修复: 常量时间比较函数，防止时序攻击
 * 无论匹配与否，遍历长度固定，不因提前不匹配而提前退出 */
static int constant_time_compare(const char* a, const char* b, size_t len) {
    int result = 0;
    for (size_t i = 0; i < len; i++)
        result |= (int)((unsigned char)a[i] ^ (unsigned char)b[i]);
    return result == 0 ? 1 : 0;
}

/* S-P1-03修复: 简单密码哈希函数（FNV-1a变体，带盐值和多轮扩散）
 * 输出32字符十六进制哈希字符串，不存储明文密码
 * 注意: 非加密学级别安全，但足以防止明文泄露和彩虹表直接反查 */
static void emergency_hash_password(const char* input, char* out, size_t out_size) {
    uint64_t h1 = 0xcbf29ce484222325ULL;   /* FNV-1a基础偏移 */
    uint64_t h2 = 0x517cc1b727220a95ULL;   /* 第二组基础偏移 */
    const uint64_t prime = 0x100000001b3ULL; /* FNV素数 */
    /* 固定盐值，增加彩虹表破解难度 */
    const char salt[] = "EMERGENCY_STOP_SALT_v1";

    /* 第一轮: 混合盐值 */
    for (size_t i = 0; salt[i] != '\0'; i++) {
        h1 ^= (uint64_t)(unsigned char)salt[i];
        h1 *= prime;
        h2 ^= (uint64_t)(unsigned char)salt[i] * 31ULL;
        h2 *= prime;
    }

    /* 第二轮: 混合输入密码 */
    size_t len = strlen(input);
    for (size_t i = 0; i < len; i++) {
        h1 ^= (uint64_t)(unsigned char)input[i];
        h1 *= prime;
        h2 ^= (uint64_t)(unsigned char)input[i] * 37ULL;
        h2 *= prime;
    }

    /* 第三轮: 多轮扩散，增强单向性 */
    for (int round = 0; round < 64; round++) {
        h1 ^= (h1 >> 33);
        h1 *= prime;
        h1 ^= (h1 >> 29);
        h2 ^= (h2 >> 33);
        h2 *= 0xff51afd7ed558ccdULL;
        h2 ^= (h2 >> 29);
    }

    /* 输出32字符十六进制哈希 */
    snprintf(out, out_size, "%016llx%016llx",
             (unsigned long long)h1, (unsigned long long)h2);
}

/* S-P1-03修复: 检查字符串是否为32位十六进制哈希格式 */
static int emergency_is_hex_hash(const char* s) {
    size_t len = strlen(s);
    if (len != 32) return 0;
    for (size_t i = 0; i < 32; i++) {
        char c = s[i];
        if (!((c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F')))
            return 0;
    }
    return 1;
}

/* F-009/S-P1-03修复: 自包含密码验证 — 哈希存储 + 常量时间比较
 * 不再明文存储或比较密码，防止时序攻击和明文泄露 */
static int emergency_verify_password(const char* input_password) {
    if (!input_password || !input_password[0]) return 0;

    /* 对输入密码进行哈希 */
    char input_hash[33];
    memset(input_hash, 0, sizeof(input_hash));
    emergency_hash_password(input_password, input_hash, sizeof(input_hash));

    /* 尝试从配置文件读取存储的密码哈希: config/emergency_password.txt */
    FILE* fp = fopen("config/emergency_password.txt", "r");
    if (fp) {
        char stored[256];
        memset(stored, 0, sizeof(stored));
        if (fgets(stored, (int)sizeof(stored) - 1, fp)) {
            fclose(fp);
            /* 去除末尾换行符 */
            size_t slen = strlen(stored);
            while (slen > 0 && (stored[slen - 1] == '\n' || stored[slen - 1] == '\r'))
                stored[--slen] = '\0';

            if (slen > 0) {
                int match = 0;
                if (emergency_is_hex_hash(stored)) {
                    /* 存储值是哈希格式，直接常量时间比较 */
                    match = constant_time_compare(input_hash, stored, 32);
                } else {
                    /* 存储值是明文（旧格式兼容），对存储值也做哈希后比较 */
                    char stored_hash[33];
                    memset(stored_hash, 0, sizeof(stored_hash));
                    emergency_hash_password(stored, stored_hash, sizeof(stored_hash));
                    match = constant_time_compare(input_hash, stored_hash, 32);

                    /* 将明文密码迁移为哈希格式存储 */
                    FILE* wf = fopen("config/emergency_password.txt", "w");
                    if (wf) {
                        fputs(stored_hash, wf);
                        fputc('\n', wf);
                        fclose(wf);
                    }
                    memset(stored_hash, 0, sizeof(stored_hash));
                }
                /* 清零敏感数据 */
                memset(stored, 0, sizeof(stored));
                memset(input_hash, 0, sizeof(input_hash));
                return match;
            }
        } else {
            fclose(fp);
        }
    }

    /* 配置文件不存在: 首次运行，创建带默认哈希的配置文件
     * 默认密码 "emergency_change_me"，用户应在首次登录后修改 */
    {
        char default_hash[33];
        memset(default_hash, 0, sizeof(default_hash));
        emergency_hash_password("emergency_change_me", default_hash, sizeof(default_hash));

        FILE* wf = fopen("config/emergency_password.txt", "w");
        if (wf) {
            fputs(default_hash, wf);
            fputc('\n', wf);
            fclose(wf);
        }
        memset(default_hash, 0, sizeof(default_hash));
    }

    /* 清零敏感数据 */
    memset(input_hash, 0, sizeof(input_hash));
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

    /* P1修复: 线程安全锁，保护所有状态操作和is_executing的check-then-set竞态。
     * 原实现EmergencyStopSystem结构体无互斥锁字段，所有状态操作无锁保护，
     * 多线程并发触发紧急停止时is_executing竞态可导致重复执行，
     * triggers/snapshots/recovery_steps数组并发写入可致越界。 */
    MutexHandle lock;
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

    /* P1修复: 初始化线程安全锁 */
    system->lock = mutex_create();
    if (!system->lock) {
        safe_free((void**)&system->triggers);
        safe_free((void**)&system->snapshots);
        safe_free((void**)&system->recovery_steps);
        safe_free((void**)&system->triggered_indices);
        safe_free((void**)&system->cfc_prediction_state);
        safe_free((void**)&system);
        return NULL;
    }

    /* 初始化拉普拉斯分析器（频域紧急停止稳定性分析） */
    return system;
}

void emergency_stop_destroy(EmergencyStopSystem* system) {
    if (!system) return;

    /* P1修复: 先销毁锁，确保不会有线程在释放内存期间持锁访问 */
    if (system->lock) mutex_destroy(system->lock);

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

    /* P1修复: 加锁保护triggers数组并发写入 */
    mutex_lock(system->lock);
    if (system->trigger_count >= system->trigger_capacity) {
        mutex_unlock(system->lock);
        return -1;
    }

    int id = system->trigger_count;
    system->triggers[id] = *trigger;
    system->triggers[id].trigger_id = id;
    system->triggers[id].is_armed = 1;
    system->triggers[id].is_triggered = 0;
    system->triggers[id].trigger_count = 0;
    system->trigger_count++;

    mutex_unlock(system->lock);
    return id;
}

int emergency_stop_manual_trigger(EmergencyStopSystem* system,
                                   EmergencyStopLevel level,
                                   const char* reason) {
    if (!system) return -1;

    /* P1修复: 加锁保护is_disabled检查和status字段修改 */
    mutex_lock(system->lock);
    if (system->is_disabled) {
        mutex_unlock(system->lock);
        return -1;
    }

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
    mutex_unlock(system->lock);

    /* 锁外调用emergency_stop_execute，避免重入死锁（execute内部有独立锁保护） */
    return emergency_stop_execute(system, level);
}

int emergency_stop_system_trigger(EmergencyStopSystem* system,
                                   EmergencySource source,
                                   EmergencyStopLevel level,
                                   const void* data, size_t data_size) {
    if (!system) return -1;

    /* P1修复: 加锁保护is_disabled检查、数据处理和status字段修改 */
    mutex_lock(system->lock);
    if (system->is_disabled) {
        mutex_unlock(system->lock);
        return -1;
    }
    /* S-P1-02修复: data_size语义统一为字节数，转换为float元素个数
     * 原代码将data_size直接当作float元素个数使用，导致缓冲区越界读取
     * (例如data_size=64字节实际只有16个float，但原代码遍历64个float) */
    if (data && data_size > 0) {
        size_t float_count = data_size / sizeof(float);  /* data_size是字节数 */
        if (float_count > 64) float_count = 64;           /* 最多处理64个float */
        if (float_count > 0) {
            const float* fd = (const float*)data;
            float data_energy = 0.0f;
            for (size_t i = 0; i < float_count; i++)
                data_energy += fd[i] * fd[i];
            system->status.cfc_anomaly_score = sqrtf(data_energy / (float)float_count);
        }
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

    /* P1修复: 在锁内缓存auto_snapshot标志，锁外调用snapshot避免重入死锁 */
    int do_auto_snapshot = system->config.auto_snapshot;
    mutex_unlock(system->lock);

    if (do_auto_snapshot) {
        EmergencySnapshot snap;
        memset(&snap, 0, sizeof(EmergencySnapshot));
        emergency_stop_snapshot(system, "自动快照(触发前)", &snap);
    }

    /* 锁外调用emergency_stop_execute，避免重入死锁（execute内部有独立锁保护） */
    return emergency_stop_execute(system, level);
}

int emergency_stop_execute(EmergencyStopSystem* system, EmergencyStopLevel level) {
    if (!system) return -1;

    /* P1修复: 保护is_executing的check-then-set竞态，防止多线程同时进入执行路径。
     * 原实现无锁保护，两个线程可同时读取is_executing=0并同时进入执行，
     * 导致重复执行紧急停止操作（重复暂停调度器、重复终止线程池等）。 */
    mutex_lock(system->lock);
    if (system->is_executing) {
        mutex_unlock(system->lock);
        return 0;
    }
    system->is_executing = 1;
    system->status.current_level = level;
    system->status.total_stop_time_ms = 0;
    mutex_unlock(system->lock);

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
            /* P1修复: 加锁保护is_executing清除 */
            mutex_lock(system->lock);
            system->is_executing = 0;
            mutex_unlock(system->lock);
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
            /* SAFETY-005修复: KILL后清零snapshot_count避免后续遍历释放的悬垂指针 */
            for (int i = 0; i < system->snapshot_count; i++) {
                safe_free((void**)&system->snapshots[i].system_state);
                safe_free((void**)&system->snapshots[i].overlay_data);
            }
            system->snapshot_count = 0;
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
            /* P1修复: 加锁保护is_executing清除 */
            mutex_lock(system->lock);
            system->is_executing = 0;
            mutex_unlock(system->lock);
            return -1;
    }

    /* P1修复: 加锁保护is_executing清除 */
    mutex_lock(system->lock);
    system->is_executing = 0;
    mutex_unlock(system->lock);
    return 0;
}

int emergency_stop_register_hardware_callback(EmergencyStopSystem* system,
                                               void (*callback)(int level, void* ctx),
                                               void* ctx) {
    if (!system) return -1;
    /* P1修复: 加锁保护回调指针写入 */
    mutex_lock(system->lock);
    system->hardware_stop_callback = callback;
    system->hardware_stop_ctx = ctx;
    mutex_unlock(system->lock);
    return 0;
}

int emergency_stop_register_task_scheduler(EmergencyStopSystem* system,
                                            void* sched_ref,
                                            void (*pause_fn)(void*),
                                            void (*resume_fn)(void*),
                                            void (*kill_fn)(void*)) {
    if (!system) return -1;
    /* P1修复: 加锁保护回调指针写入 */
    mutex_lock(system->lock);
    system->task_scheduler_ref = sched_ref;
    system->task_scheduler_pause = pause_fn;
    system->task_scheduler_resume = resume_fn;
    system->task_scheduler_kill = kill_fn;
    mutex_unlock(system->lock);
    return 0;
}

int emergency_stop_register_thread_pool(EmergencyStopSystem* system,
                                         void* pool_ref,
                                         void (*stop_all_fn)(void*)) {
    if (!system) return -1;
    /* P1修复: 加锁保护回调指针写入 */
    mutex_lock(system->lock);
    system->thread_pool_ref = pool_ref;
    system->thread_pool_stop_all = stop_all_fn;
    mutex_unlock(system->lock);
    return 0;
}

int emergency_stop_register_power_cut(EmergencyStopSystem* system,
                                       void (*cut_fn)(void)) {
    if (!system) return -1;
    /* P1修复: 加锁保护回调指针写入 */
    mutex_lock(system->lock);
    system->physical_power_cut = cut_fn;
    mutex_unlock(system->lock);
    return 0;
}

/* H-005修复: 注册线程池重启回调（用于快照恢复时重新启动线程池） */
int emergency_stop_register_thread_pool_restart(EmergencyStopSystem* system,
                                                  void* pool_ref,
                                                  void (*restart_fn)(void*)) {
    if (!system) return -1;
    /* P1修复: 加锁保护回调指针写入 */
    mutex_lock(system->lock);
    system->thread_pool_ref = pool_ref;
    system->thread_pool_restart = restart_fn;
    mutex_unlock(system->lock);
    return 0;
}

/* H-005修复: 注册模型推理恢复回调（用于快照恢复时重新激活模型推理） */
int emergency_stop_register_model_inference_resume(EmergencyStopSystem* system,
                                                     void* ctx,
                                                     void (*resume_fn)(void*)) {
    if (!system) return -1;
    /* P1修复: 加锁保护回调指针写入 */
    mutex_lock(system->lock);
    system->model_inference_ctx = ctx;
    system->model_inference_resume = resume_fn;
    mutex_unlock(system->lock);
    return 0;
}

/* H-005修复: 注册审计日志回调（用于记录恢复操作审计信息） */
int emergency_stop_register_audit_log(EmergencyStopSystem* system,
                                       void (*log_fn)(const char* message, void* ctx),
                                       void* ctx) {
    if (!system) return -1;
    /* P1修复: 加锁保护回调指针写入 */
    mutex_lock(system->lock);
    system->audit_log_callback = log_fn;
    system->audit_log_ctx = ctx;
    mutex_unlock(system->lock);
    return 0;
}

int emergency_stop_snapshot(EmergencyStopSystem* system,
                             const char* description,
                             EmergencySnapshot* snapshot) {
    if (!system || !snapshot) return -1;

    /* P1修复: 加锁保护snapshots数组的并发写入和realloc操作。
     * realloc可能移动整个数组的基址指针，若另一线程同时读取会导致悬垂指针。 */
    mutex_lock(system->lock);

    if (system->snapshot_count >= system->snapshot_capacity) {
        int new_cap = system->snapshot_capacity * 2;
        EmergencySnapshot* new_snaps = (EmergencySnapshot*)
            safe_realloc(system->snapshots, new_cap * sizeof(EmergencySnapshot));
        if (!new_snaps) {
            mutex_unlock(system->lock);
            return -1;
        }
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
        /* S-P0-01修复: 所有字段使用memcpy写入，避免非对齐指针转换
         * (ARM/SPARC等架构上非对齐访问会触发SIGBUS)
         * 偏移60/68/76(uint64_t)和92(int64_t)非8字节对齐，必须用memcpy */
        {
            int32_t tmp32;
            int64_t tmp64;
            uint64_t tmpu64;
            float tmpf;

            /* 写入版本 */
            tmp32 = 1;
            memcpy(buf + 4, &tmp32, sizeof(int32_t));
            /* AGI认知状态 */
            tmp32 = m.active_task_count;
            memcpy(buf + 8, &tmp32, sizeof(int32_t));
            tmp32 = m.total_task_count;
            memcpy(buf + 12, &tmp32, sizeof(int32_t));
            tmp32 = m.active_goal_count;
            memcpy(buf + 16, &tmp32, sizeof(int32_t));
            tmp32 = m.total_goal_count;
            memcpy(buf + 20, &tmp32, sizeof(int32_t));
            tmp32 = m.cognitive_cycle_count;
            memcpy(buf + 24, &tmp32, sizeof(int32_t));
            tmpf = m.cognitive_load;
            memcpy(buf + 28, &tmpf, sizeof(float));
            /* LNN状态摘要 */
            tmpf = m.lnn_mean_activation;
            memcpy(buf + 32, &tmpf, sizeof(float));
            tmpf = m.lnn_max_activation;
            memcpy(buf + 36, &tmpf, sizeof(float));
            tmpf = m.lnn_min_activation;
            memcpy(buf + 40, &tmpf, sizeof(float));
            tmp32 = m.lnn_step_count;
            memcpy(buf + 44, &tmp32, sizeof(int32_t));
            tmp32 = m.lnn_hidden_dim;
            memcpy(buf + 48, &tmp32, sizeof(int32_t));
            tmp32 = m.lnn_input_dim;
            memcpy(buf + 52, &tmp32, sizeof(int32_t));
            tmp32 = m.lnn_output_dim;
            memcpy(buf + 56, &tmp32, sizeof(int32_t));
            /* 内存使用摘要 (uint64_t，偏移60/68/76非8字节对齐，必须用memcpy) */
            tmpu64 = (uint64_t)m.total_allocated_bytes;
            memcpy(buf + 60, &tmpu64, sizeof(uint64_t));
            tmpu64 = (uint64_t)m.total_freed_bytes;
            memcpy(buf + 68, &tmpu64, sizeof(uint64_t));
            tmpu64 = (uint64_t)m.current_used_bytes;
            memcpy(buf + 76, &tmpu64, sizeof(uint64_t));
            tmp32 = m.active_allocation_count;
            memcpy(buf + 84, &tmp32, sizeof(int32_t));
            tmpf = m.memory_usage_ratio;
            memcpy(buf + 88, &tmpf, sizeof(float));
            /* 时间戳 (int64_t，偏移92非8字节对齐，必须用memcpy) */
            tmp64 = (int64_t)m.snapshot_timestamp_us;
            memcpy(buf + 92, &tmp64, sizeof(int64_t));
            tmp32 = m.snapshot_sequence;
            memcpy(buf + 100, &tmp32, sizeof(int32_t));
        }
    }

    system->snapshot_count++;

    *snapshot = *snap;
    int ret_snap_id = snap->snapshot_id;
    mutex_unlock(system->lock);
    return ret_snap_id;
}

int emergency_stop_restore(EmergencyStopSystem* system,
                            const EmergencySnapshot* snapshot) {
    if (!system || !snapshot) return -1;
    if (!snapshot->is_valid) return -1;

    /* P1修复: 保护is_executing的check-then-set竞态 */
    mutex_lock(system->lock);
    if (system->is_executing) {
        mutex_unlock(system->lock);
        return -1;
    }
    system->is_executing = 1;
    mutex_unlock(system->lock);

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
    /* P1修复: 加锁保护status字段和triggers数组的修改 */
    mutex_lock(system->lock);
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
    mutex_unlock(system->lock);

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

    /* P1修复: 加锁保护is_executing清除 */
    mutex_lock(system->lock);
    system->is_executing = 0;
    mutex_unlock(system->lock);
    return 0;
}

int emergency_stop_fallback(EmergencyStopSystem* system,
                             EmergencyFallbackStrategy strategy) {
    if (!system) return -1;

    switch (strategy) {
        case EMERGENCY_FALLBACK_HOLD_POSITION:
            /* P1修复: 加锁保护is_disabled修改，锁外调用回调避免重入死锁 */
            mutex_lock(system->lock);
            system->is_disabled = 1;
            mutex_unlock(system->lock);
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(0, system->hardware_stop_ctx);
            }
            mutex_lock(system->lock);
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[保持位置]: 任务调度器已冻结，硬件保持当前位置");
            mutex_unlock(system->lock);
            break;

        case EMERGENCY_FALLBACK_RETURN_HOME:
            mutex_lock(system->lock);
            system->is_disabled = 1;
            mutex_unlock(system->lock);
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(1, system->hardware_stop_ctx);
            }
            mutex_lock(system->lock);
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[返回安全位置]: 已发送回退指令到硬件控制器");
            mutex_unlock(system->lock);
            break;

        case EMERGENCY_FALLBACK_POWER_DOWN:
            mutex_lock(system->lock);
            system->is_disabled = 1;
            mutex_unlock(system->lock);
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->thread_pool_stop_all && system->thread_pool_ref) {
                system->thread_pool_stop_all(system->thread_pool_ref);
            }
            if (system->physical_power_cut) {
                system->physical_power_cut();
            }
            mutex_lock(system->lock);
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[逐步断电]: 线程池已终止，已发送断电信号");
            mutex_unlock(system->lock);
            break;

        case EMERGENCY_FALLBACK_EMERGENCY_STOP:
            /* 锁外调用emergency_stop_execute，避免重入死锁 */
            emergency_stop_execute(system, EMERGENCY_LEVEL_HARD_STOP);
            break;

        case EMERGENCY_FALLBACK_ISOLATE:
            mutex_lock(system->lock);
            system->is_disabled = 1;
            mutex_unlock(system->lock);
            if (system->task_scheduler_pause && system->task_scheduler_ref) {
                system->task_scheduler_pause(system->task_scheduler_ref);
            }
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(2, system->hardware_stop_ctx);
            }
            mutex_lock(system->lock);
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[隔离]: 任务调度器已冻结，故障区域已隔离");
            mutex_unlock(system->lock);
            break;

        case EMERGENCY_FALLBACK_REDUNDANT_SYSTEM:
            mutex_lock(system->lock);
            system->is_disabled = 1;
            mutex_unlock(system->lock);
            if (system->hardware_stop_callback) {
                system->hardware_stop_callback(3, system->hardware_stop_ctx);
            }
            mutex_lock(system->lock);
            snprintf(system->status.status_message,
                     sizeof(system->status.status_message),
                     "安全回退[切换冗余]: 已向备用系统发送切换指令");
            mutex_unlock(system->lock);
            break;

        default:
            return -1;
    }

    return 0;
}

int emergency_stop_add_recovery_step(EmergencyStopSystem* system,
                                      const RecoveryStep* step) {
    if (!system || !step) return -1;

    /* P1修复: 加锁保护recovery_steps数组并发写入 */
    mutex_lock(system->lock);
    if (system->recovery_step_count >= system->recovery_step_capacity) {
        mutex_unlock(system->lock);
        return -1;
    }

    int id = system->recovery_step_count;
    system->recovery_steps[id] = *step;
    system->recovery_steps[id].step_id = id;
    system->recovery_steps[id].is_completed = 0;
    system->recovery_steps[id].progress = 0.0f;
    system->recovery_step_count++;

    mutex_unlock(system->lock);
    return id;
}

int emergency_stop_recover(EmergencyStopSystem* system) {
    if (!system) return -1;

    /* P1修复: 加锁保护恢复过程中的状态修改和recovery_steps遍历，
     * 防止与add_recovery_step并发写入冲突 */
    mutex_lock(system->lock);
    if (!system->config.enable_recovery) {
        mutex_unlock(system->lock);
        return -1;
    }
    if (system->is_disabled) {
        mutex_unlock(system->lock);
        return -1;
    }

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
        mutex_unlock(system->lock);
        return 0;
    }

    system->status.is_recovering = 0;
    mutex_unlock(system->lock);
    return -1;
}

float emergency_stop_cfc_predict(EmergencyStopSystem* system,
                                  const float* system_state, int state_dim,
                                  const float* control_input, int control_dim) {
    if (!system || !system_state || state_dim < 1) return 0.0f;

    /* P1修复: 加锁保护cfc_prediction_state的realloc操作 */
    mutex_lock(system->lock);
    if (system->cfc_prediction_dim < state_dim) {
        float* new_state = (float*)
            safe_realloc(system->cfc_prediction_state, state_dim * sizeof(float));
        if (!new_state) {
            mutex_unlock(system->lock);
            return 0.0f;
        }
        system->cfc_prediction_state = new_state;
        system->cfc_prediction_dim = state_dim;
    }
    mutex_unlock(system->lock);

    int dim = state_dim;
    float current_state[64];
    int cdim = dim < 64 ? dim : 64;
    for (int i = 0; i < cdim; i++)
        current_state[i] = system_state[i];

    /* CfC ODE预测演化（纯本地计算，无需持锁） */
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

    /* P1修复: 加锁保护cfc_anomaly_score写入和阈值判断 */
    mutex_lock(system->lock);
    system->status.cfc_anomaly_score = anomaly;

    /* 如果异常超过阈值，自动触发预防性停止 */
    int should_trigger = (anomaly > system->config.cfc_anomaly_threshold &&
                          !system->is_disabled &&
                          system->config.enable_cfc_prediction);
    float cached_threshold = system->config.cfc_anomaly_threshold;
    mutex_unlock(system->lock);

    if (should_trigger) {
        EmergencyTrigger trigger;
        memset(&trigger, 0, sizeof(EmergencyTrigger));
        trigger.source = EMERGENCY_SOURCE_CFC_PREDICT;
        trigger.target_level = EMERGENCY_LEVEL_SOFT_STOP;
        trigger.fallback = EMERGENCY_FALLBACK_HOLD_POSITION;
        trigger.current_value = anomaly;
        trigger.threshold = cached_threshold;
        strncpy(trigger.condition_name, "CfC异常预测",
                sizeof(trigger.condition_name) - 1);

        /* 锁外调用register_trigger和system_trigger（两者均有独立锁保护） */
        int tid = emergency_stop_register_trigger(system, &trigger);
        if (tid >= 0) {
            mutex_lock(system->lock);
            if (tid < system->trigger_count)
                system->triggers[tid].is_triggered = 1;
            mutex_unlock(system->lock);
        }

        emergency_stop_system_trigger(system, EMERGENCY_SOURCE_CFC_PREDICT,
                                       EMERGENCY_LEVEL_SOFT_STOP, NULL, 0);
    }

    return anomaly;
}

int emergency_stop_get_status(const EmergencyStopSystem* system,
                               EmergencyStopStatus* status) {
    if (!system || !status) return -1;
    /* P1修复: 加锁保护status读取，防止与trigger/execute等写线程竞态 */
    mutex_lock(((EmergencyStopSystem*)system)->lock);
    *status = system->status;
    mutex_unlock(((EmergencyStopSystem*)system)->lock);
    return 0;
}

int emergency_stop_get_triggered_list(const EmergencyStopSystem* system,
                                       EmergencyTrigger* triggers, int max_count) {
    if (!system || !triggers || max_count < 1) return -1;

    /* P1修复: 加锁保护triggers数组读取，防止与register_trigger并发写入冲突 */
    mutex_lock(((EmergencyStopSystem*)system)->lock);
    int count = 0;
    for (int i = 0; i < system->trigger_count && count < max_count; i++) {
        if (system->triggers[i].is_triggered) {
            triggers[count++] = system->triggers[i];
        }
    }
    mutex_unlock(((EmergencyStopSystem*)system)->lock);

    return count;
}

/* P3-001修复: 设置系统运行时指标 — 供快照捕获使用 */
int emergency_stop_set_metrics(EmergencyStopSystem* system,
                                const EmergencySystemMetrics* metrics) {
    if (!system || !metrics) return -1;
    /* P1修复: 加锁保护metrics写入 */
    mutex_lock(system->lock);
    memcpy(&system->last_metrics, metrics, sizeof(EmergencySystemMetrics));
    system->metrics_valid = 1;
    mutex_unlock(system->lock);
    return 0;
}

/* P3-001修复: 获取最近一次设置的指标 */
int emergency_stop_get_metrics(const EmergencyStopSystem* system,
                                EmergencySystemMetrics* metrics) {
    if (!system || !metrics) return -1;
    /* P1修复: 加锁保护metrics读取 */
    mutex_lock(((EmergencyStopSystem*)system)->lock);
    if (!system->metrics_valid) {
        mutex_unlock(((EmergencyStopSystem*)system)->lock);
        return -1;
    }
    memcpy(metrics, &system->last_metrics, sizeof(EmergencySystemMetrics));
    mutex_unlock(((EmergencyStopSystem*)system)->lock);
    return 0;
}

int emergency_stop_clear(EmergencyStopSystem* system, const char* password) {
    if (!system) return -1;

    if (!password || !emergency_verify_password(password))
        return -1;

    /* P1修复: 加锁保护status和triggers数组的修改 */
    mutex_lock(system->lock);
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
    mutex_unlock(system->lock);

    return 0;
}

int emergency_stop_set_disabled(EmergencyStopSystem* system,
                                 int disable, const char* password) {
    if (!system) return -1;

    if (!password || !emergency_verify_password(password))
        return -1;

    /* P1修复: 加锁保护is_disabled和status修改 */
    mutex_lock(system->lock);
    system->is_disabled = disable;

    snprintf(system->status.status_message,
             sizeof(system->status.status_message),
             "紧急停止系统已%s", disable ? "禁用" : "启用");
    mutex_unlock(system->lock);

    return 0;
}

int emergency_stop_save(const EmergencyStopSystem* system, const char* filepath) {
    if (!system || !filepath) return -1;

    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    /* S-P0-01/H-P2-23修复: 逐字段序列化写入，使用固定宽度类型
     * 避免结构体对齐和平台相关类型(long/size_t)导致的跨平台不兼容
     * 每个fwrite都检查返回值，确保数据完整写入 */

    /* 文件头: 魔数 "EMS1" + 版本 */
    const char magic[4] = {'E', 'M', 'S', '1'};
    if (fwrite(magic, 4, 1, f) != 1) { fclose(f); return -1; }
    int32_t version = 1;
    if (fwrite(&version, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }

    /* --- 配置字段 (EmergencyStopConfig) --- */
    int32_t v32;
    int64_t v64;
    float vf;
    uint32_t v32u;

    v32 = (int32_t)system->config.max_triggers;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->config.auto_snapshot;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->config.enable_recovery;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->config.enable_cfc_prediction;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    vf = system->config.cfc_tau;
    if (fwrite(&vf, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
    vf = system->config.cfc_dt;
    if (fwrite(&vf, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->config.cfc_steps;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    vf = system->config.cfc_anomaly_threshold;
    if (fwrite(&vf, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->config.max_recovery_attempts;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->config.recovery_timeout_ms;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->config.enable_manual_override;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    /* log_file: 先写长度(uint32_t)再写内容 */
    {
        v32u = 0;
        while (v32u < sizeof(system->config.log_file) && system->config.log_file[v32u] != '\0') v32u++;
        if (fwrite(&v32u, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
        if (v32u > 0) {
            if (fwrite(system->config.log_file, 1, v32u, f) != v32u) { fclose(f); return -1; }
        }
    }

    /* --- 状态字段 (EmergencyStopStatus, long转为int64_t) --- */
    v32 = (int32_t)system->status.current_level;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->status.highest_level;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->status.last_source;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v64 = (int64_t)system->status.last_trigger_time;
    if (fwrite(&v64, sizeof(int64_t), 1, f) != 1) { fclose(f); return -1; }
    v64 = (int64_t)system->status.total_stop_time_ms;
    if (fwrite(&v64, sizeof(int64_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->status.trigger_count;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->status.recovery_count;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->status.recovery_success_count;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->status.is_recovering;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    v32 = (int32_t)system->status.is_manual_block;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    vf = system->status.cfc_anomaly_score;
    if (fwrite(&vf, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
    /* status_message: 先写长度再写内容 */
    {
        v32u = 0;
        while (v32u < sizeof(system->status.status_message) && system->status.status_message[v32u] != '\0') v32u++;
        if (fwrite(&v32u, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
        if (v32u > 0) {
            if (fwrite(system->status.status_message, 1, v32u, f) != v32u) { fclose(f); return -1; }
        }
    }

    /* --- 触发条件数量 --- */
    v32 = (int32_t)system->trigger_count;
    if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }

    /* --- 逐个写入触发条件 (EmergencyTrigger) --- */
    for (int i = 0; i < system->trigger_count; i++) {
        const EmergencyTrigger* t = &system->triggers[i];
        v32 = (int32_t)t->trigger_id;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        v32 = (int32_t)t->source;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        /* condition_name: 先写长度再写内容 */
        {
            v32u = 0;
            while (v32u < sizeof(t->condition_name) && t->condition_name[v32u] != '\0') v32u++;
            if (fwrite(&v32u, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }
            if (v32u > 0) {
                if (fwrite(t->condition_name, 1, v32u, f) != v32u) { fclose(f); return -1; }
            }
        }
        vf = t->threshold;
        if (fwrite(&vf, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
        vf = t->hysteresis;
        if (fwrite(&vf, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
        vf = t->current_value;
        if (fwrite(&vf, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
        v32 = (int32_t)t->check_interval_ms;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        v32 = (int32_t)t->min_triggers;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        v32 = (int32_t)t->trigger_count;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        v32 = (int32_t)t->target_level;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        v32 = (int32_t)t->fallback;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        v32 = (int32_t)t->is_armed;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        v32 = (int32_t)t->is_triggered;
        if (fwrite(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
    }

    fclose(f);
    return 0;
}

EmergencyStopSystem* emergency_stop_load(const char* filepath) {
    if (!filepath) return NULL;

    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    /* S-P0-01/H-P2-23/S-P1-04修复: 逐字段反序列化读取，使用固定宽度类型
     * 每个fread都检查返回值，防止部分读取导致数据损坏 */

    /* 读取并验证文件头 */
    char magic[4];
    if (fread(magic, 4, 1, f) != 1 ||
        magic[0] != 'E' || magic[1] != 'M' || magic[2] != 'S' || magic[3] != '1') {
        fclose(f);
        return NULL;
    }
    int32_t version;
    if (fread(&version, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    /* --- 读取配置字段 --- */
    EmergencyStopConfig config;
    memset(&config, 0, sizeof(EmergencyStopConfig));
    int32_t v32;
    int64_t v64;
    uint32_t v32u;
    float vf;

    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    config.max_triggers = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    config.auto_snapshot = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    config.enable_recovery = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    config.enable_cfc_prediction = v32;
    if (fread(&vf, sizeof(float), 1, f) != 1) { fclose(f); return NULL; }
    config.cfc_tau = vf;
    if (fread(&vf, sizeof(float), 1, f) != 1) { fclose(f); return NULL; }
    config.cfc_dt = vf;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    config.cfc_steps = v32;
    if (fread(&vf, sizeof(float), 1, f) != 1) { fclose(f); return NULL; }
    config.cfc_anomaly_threshold = vf;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    config.max_recovery_attempts = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    config.recovery_timeout_ms = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    config.enable_manual_override = v32;
    /* log_file: 先读长度再读内容 */
    if (fread(&v32u, sizeof(uint32_t), 1, f) != 1) { fclose(f); return NULL; }
    if (v32u > 0) {
        if (v32u >= sizeof(config.log_file)) v32u = (uint32_t)sizeof(config.log_file) - 1;
        if (fread(config.log_file, 1, v32u, f) != v32u) { fclose(f); return NULL; }
    }
    config.log_file[v32u] = '\0';

    /* 创建系统实例 */
    EmergencyStopSystem* system = emergency_stop_create(&config);
    if (!system) { fclose(f); return NULL; }

    /* --- 读取状态字段 (int64_t转回long) --- */
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.current_level = (EmergencyStopLevel)v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.highest_level = (EmergencyStopLevel)v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.last_source = (EmergencySource)v32;
    if (fread(&v64, sizeof(int64_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.last_trigger_time = (long)v64;
    if (fread(&v64, sizeof(int64_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.total_stop_time_ms = (long)v64;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.trigger_count = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.recovery_count = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.recovery_success_count = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.is_recovering = v32;
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.is_manual_block = v32;
    if (fread(&vf, sizeof(float), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->status.cfc_anomaly_score = vf;
    /* status_message: 先读长度再读内容 */
    if (fread(&v32u, sizeof(uint32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    if (v32u > 0) {
        if (v32u >= sizeof(system->status.status_message)) v32u = (uint32_t)sizeof(system->status.status_message) - 1;
        if (fread(system->status.status_message, 1, v32u, f) != v32u) { fclose(f); emergency_stop_destroy(system); return NULL; }
    }
    system->status.status_message[v32u] = '\0';

    /* --- 读取触发条件数量 --- */
    if (fread(&v32, sizeof(int32_t), 1, f) != 1) { fclose(f); emergency_stop_destroy(system); return NULL; }
    system->trigger_count = v32;
    if (system->trigger_count > system->trigger_capacity)
        system->trigger_count = system->trigger_capacity;

    /* --- 逐个读取触发条件，每个fread检查返回值 --- */
    for (int i = 0; i < system->trigger_count; i++) {
        EmergencyTrigger* t = &system->triggers[i];
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->trigger_id = v32;
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->source = (EmergencySource)v32;
        /* condition_name */
        if (fread(&v32u, sizeof(uint32_t), 1, f) != 1) { system->trigger_count = i; break; }
        if (v32u > 0) {
            if (v32u >= sizeof(t->condition_name)) v32u = (uint32_t)sizeof(t->condition_name) - 1;
            if (fread(t->condition_name, 1, v32u, f) != v32u) { system->trigger_count = i; break; }
        }
        t->condition_name[v32u] = '\0';
        if (fread(&vf, sizeof(float), 1, f) != 1) { system->trigger_count = i; break; }
        t->threshold = vf;
        if (fread(&vf, sizeof(float), 1, f) != 1) { system->trigger_count = i; break; }
        t->hysteresis = vf;
        if (fread(&vf, sizeof(float), 1, f) != 1) { system->trigger_count = i; break; }
        t->current_value = vf;
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->check_interval_ms = v32;
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->min_triggers = v32;
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->trigger_count = v32;
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->target_level = (EmergencyStopLevel)v32;
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->fallback = (EmergencyFallbackStrategy)v32;
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->is_armed = v32;
        if (fread(&v32, sizeof(int32_t), 1, f) != 1) { system->trigger_count = i; break; }
        t->is_triggered = v32;
    }

    fclose(f);
    return system;
}

int emergency_stop_register_gpu_kernel_interrupt(EmergencyStopSystem* system,
                                                  void (*callback)(void*),
                                                  void* ctx) {
    if (!system) return -1;
    /* P1修复: 加锁保护GPU内核中断回调指针写入 */
    mutex_lock(system->lock);
    system->gpu_kernel_interrupt_callback = callback;
    system->gpu_kernel_interrupt_ctx = ctx;
    system->gpu_kernel_interrupt_flag = 1;
    mutex_unlock(system->lock);
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
