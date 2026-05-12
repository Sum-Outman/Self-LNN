/**
 * @file emergency_stop.h
 * @brief 紧急停止系统接口（A09.2）
 *
 * 基于CfC液态神经网络的安全回退与紧急停止系统。
 * 提供多层次紧急停止机制，确保AGI系统在任何异常情况下都能安全回退。
 *
 * 核心机制：
 * 1. 硬件紧急停止 — 直接切断执行器电源/信号
 * 2. 软件紧急停止 — 暂停所有非关键进程
 * 3. 渐进式安全回退 — 按优先级逐步降级
 * 4. 自动恢复与状态保存 — 故障恢复后恢复执行
 * 5. CfC预测性紧急停止 — 基于液态ODE的异常预测
 *
 * 100% 纯C实现，无第三方库依赖。
 */

#ifndef SELFLNN_EMERGENCY_STOP_H
#define SELFLNN_EMERGENCY_STOP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 配置常量
 * ============================================================================ */
#define EMERGENCY_MAX_TRIGGERS      256     /**< 最大触发条件数 */
#define EMERGENCY_MAX_RECOVERY_STEPS 64     /**< 最大恢复步骤数 */
#define EMERGENCY_MAX_SNAPSHOT_SIZE 1048576 /**< 最大状态快照大小(1MB) */
#define EMERGENCY_MAX_OVERLAY_SIZE  65536   /**< 最大覆盖数据大小 */

/* ============================================================================
 * 类型定义
 * ============================================================================ */

/**
 * @brief 紧急停止级别
 */
typedef enum {
    EMERGENCY_LEVEL_NONE = 0,               /**< 正常运行 */
    EMERGENCY_LEVEL_SOFT_STOP,              /**< 软停止（完成当前安全操作后停止） */
    EMERGENCY_LEVEL_PAUSE,                  /**< 暂停（冻结所有状态，可恢复） */
    EMERGENCY_LEVEL_HARD_STOP,              /**< 硬停止（立即停止所有操作） */
    EMERGENCY_LEVEL_KILL,                   /**< 终止（释放所有资源） */
    EMERGENCY_LEVEL_PHYSICAL_CUT            /**< 物理切断（硬件断电/断气） */
} EmergencyStopLevel;

/**
 * @brief 紧急停止触发源类型
 */
typedef enum {
    EMERGENCY_SOURCE_SYSTEM = 0,            /**< 系统自检触发 */
    EMERGENCY_SOURCE_HARDWARE,              /**< 硬件故障触发 */
    EMERGENCY_SOURCE_SOFTWARE,              /**< 软件异常触发 */
    EMERGENCY_SOURCE_SENSOR,                /**< 传感器数据异常 */
    EMERGENCY_SOURCE_COMMUNICATION,         /**< 通信异常 */
    EMERGENCY_SOURCE_POWER,                 /**< 电源异常 */
    EMERGENCY_SOURCE_SAFETY_MONITOR,        /**< 安全监控触发 */
    EMERGENCY_SOURCE_MANUAL,                /**< 手动触发 */
    EMERGENCY_SOURCE_CFC_PREDICT            /**< CfC预测触发 */
} EmergencySource;

/**
 * @brief 安全回退策略
 */
typedef enum {
    EMERGENCY_FALLBACK_NONE = 0,            /**< 无回退 */
    EMERGENCY_FALLBACK_HOLD_POSITION,       /**< 保持当前位置/状态 */
    EMERGENCY_FALLBACK_RETURN_HOME,         /**< 返回安全位置 */
    EMERGENCY_FALLBACK_POWER_DOWN,          /**< 逐步断电 */
    EMERGENCY_FALLBACK_EMERGENCY_STOP,      /**< 立即停止 */
    EMERGENCY_FALLBACK_ISOLATE,             /**< 隔离故障区域 */
    EMERGENCY_FALLBACK_REDUNDANT_SYSTEM     /**< 切换到冗余系统 */
} EmergencyFallbackStrategy;

/**
 * @brief 触发条件
 */
typedef struct {
    int trigger_id;                         /**< 条件ID */
    EmergencySource source;                 /**< 触发源类型 */
    char condition_name[64];               /**< 条件名称 */
    float threshold;                        /**< 触发阈值 */
    float hysteresis;                       /**< 迟滞值（防止频繁触发） */
    float current_value;                    /**< 当前监测值 */
    int check_interval_ms;                  /**< 检查间隔(毫秒) */
    int min_triggers;                       /**< 最小连续触发次数 */
    int trigger_count;                      /**< 当前触发计数 */
    EmergencyStopLevel target_level;        /**< 目标停止级别 */
    EmergencyFallbackStrategy fallback;     /**< 回退策略 */
    int is_armed;                           /**< 是否启用 */
    int is_triggered;                       /**< 是否已触发 */
} EmergencyTrigger;

/**
 * @brief 状态快照（用于恢复）
 */
typedef struct {
    long snapshot_id;                       /**< 快照ID */
    long timestamp_us;                      /**< 快照时间戳(微秒) */
    EmergencyStopLevel level;               /**< 停止级别 */
    void* system_state;                      /**< 系统状态数据 */
    size_t state_size;                      /**< 状态数据大小 */
    void* overlay_data;                      /**< 覆盖数据 */
    size_t overlay_size;                    /**< 覆盖数据大小 */
    char description[256];                  /**< 快照描述 */
    int is_valid;                           /**< 是否有效 */
} EmergencySnapshot;

/**
 * @brief 恢复步骤
 */
typedef struct {
    int step_id;                            /**< 步骤ID */
    char action_name[64];                   /**< 操作名称 */
    float estimated_time_s;                 /**< 估计恢复时间(秒) */
    int is_critical;                        /**< 是否为关键步骤 */
    int is_completed;                       /**< 是否已完成 */
    float progress;                         /**< 进度(0~1) */
    int (*step_function)(void* context);    /**< 执行函数 */
    void* context;                          /**< 上下文 */
} RecoveryStep;

/**
 * @brief 紧急停止配置
 */
typedef struct {
    int max_triggers;                       /**< 最大触发条件数 */
    int auto_snapshot;                      /**< 触发时自动快照 */
    int enable_recovery;                    /**< 启用自动恢复 */
    int enable_cfc_prediction;              /**< 启用CfC预测性停止 */
    float cfc_tau;                          /**< CfC时间常数 */
    float cfc_dt;                           /**< CfC步长 */
    int cfc_steps;                          /**< CfC积分步数 */
    float cfc_anomaly_threshold;            /**< CfC异常检测阈值 */
    int max_recovery_attempts;              /**< 最大恢复尝试次数 */
    int recovery_timeout_ms;                /**< 恢复超时(毫秒) */
    int enable_manual_override;             /**< 启用手动干预覆盖 */
    char log_file[256];                     /**< 日志文件路径 */
} EmergencyStopConfig;

/**
 * @brief 紧急停止状态
 */
typedef struct {
    EmergencyStopLevel current_level;       /**< 当前停止级别 */
    EmergencyStopLevel highest_level;       /**< 历史最高级别 */
    EmergencySource last_source;            /**< 最后触发源 */
    long last_trigger_time;                 /**< 最后触发时间 */
    long total_stop_time_ms;                /**< 总停止时间(毫秒) */
    int trigger_count;                      /**< 总触发次数 */
    int recovery_count;                     /**< 恢复次数 */
    int recovery_success_count;             /**< 恢复成功次数 */
    int is_recovering;                      /**< 是否正在恢复 */
    int is_manual_block;                    /**< 是否被手动锁定 */
    float cfc_anomaly_score;                /**< CfC异常分数 */
    char status_message[256];               /**< 状态信息 */
} EmergencyStopStatus;

/**
 * @brief 紧急停止系统句柄
 */
typedef struct EmergencyStopSystem EmergencyStopSystem;

/* ============================================================================
 * 核心API
 * ============================================================================ */

/**
 * @brief 创建紧急停止系统
 *
 * @param config 系统配置
 * @return EmergencyStopSystem* 成功返回句柄，失败返回NULL
 */
EmergencyStopSystem* emergency_stop_create(const EmergencyStopConfig* config);

/**
 * @brief 销毁紧急停止系统
 *
 * @param system 系统句柄
 */
void emergency_stop_destroy(EmergencyStopSystem* system);

/**
 * @brief 注册紧急停止触发条件
 *
 * @param system 系统句柄
 * @param trigger 触发条件定义
 * @return int 成功返回触发条件ID，失败返回-1
 */
int emergency_stop_register_trigger(EmergencyStopSystem* system,
                                     const EmergencyTrigger* trigger);

/**
 * @brief 手动触发紧急停止
 *
 * 最高优先级的触发方式，覆盖所有自动检测。
 *
 * @param system 系统句柄
 * @param level 停止级别
 * @param reason 停止原因
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_manual_trigger(EmergencyStopSystem* system,
                                   EmergencyStopLevel level,
                                   const char* reason);

/**
 * @brief 系统自检触发紧急停止
 *
 * 由安全监控系统或其他子系统触发。
 *
 * @param system 系统句柄
 * @param source 触发源
 * @param level 停止级别
 * @param data 触发数据
 * @param data_size 数据大小
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_system_trigger(EmergencyStopSystem* system,
                                   EmergencySource source,
                                   EmergencyStopLevel level,
                                   const void* data, size_t data_size);

/**
 * @brief 执行紧急停止
 *
 * 根据指定级别执行实际停止操作：
 * - SOFT_STOP: 设置停止标志，等待当前安全操作完成
 * - PAUSE: 冻结所有线程和状态
 * - HARD_STOP: 立即中断所有操作
 * - KILL: 释放所有资源，关闭文件句柄
 * - PHYSICAL_CUT: 通过硬件接口发送断电信号
 *
 * @param system 系统句柄
 * @param level 停止级别
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_execute(EmergencyStopSystem* system, EmergencyStopLevel level);

/**
 * @brief 创建状态快照（用于恢复）
 *
 * @param system 系统句柄
 * @param description 快照描述
 * @param snapshot 输出快照
 * @return int 成功返回快照ID，失败返回-1
 */
int emergency_stop_snapshot(EmergencyStopSystem* system,
                             const char* description,
                             EmergencySnapshot* snapshot);

/**
 * @brief 从快照恢复
 *
 * @param system 系统句柄
 * @param snapshot 快照
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_restore(EmergencyStopSystem* system,
                            const EmergencySnapshot* snapshot);

/**
 * @brief 执行渐进式安全回退
 *
 * 按优先级逐步降级操作：
 *   1. 非关键任务暂停
 *   2. 释放非关键资源
 *   3. 保持关键任务
 *   4. 转移状态到安全区域
 *   5. 执行最终停止
 *
 * @param system 系统句柄
 * @param strategy 回退策略
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_fallback(EmergencyStopSystem* system,
                             EmergencyFallbackStrategy strategy);

/**
 * @brief 添加恢复步骤
 *
 * @param system 系统句柄
 * @param step 恢复步骤定义
 * @return int 成功返回步骤ID，失败返回-1
 */
int emergency_stop_add_recovery_step(EmergencyStopSystem* system,
                                      const RecoveryStep* step);

/**
 * @brief 执行自动恢复
 *
 * 按步骤序列执行恢复：
 *   1. 验证快照有效性
 *   2. 依次执行每个恢复步骤
 *   3. 每一步完成后验证状态
 *   4. 如果失败，尝试回滚
 *
 * @param system 系统句柄
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_recover(EmergencyStopSystem* system);

/**
 * @brief CfC预测性紧急停止检测
 *
 * 使用CfC液态ODE监测系统状态，预测潜在异常：
 *   h_pred = CFC_ODE(h_current, u_control)
 *   anomaly = ||h_pred - h_expected||₂
 *   如果anomaly > 阈值，触发预防性停止
 *
 * @param system 系统句柄
 * @param system_state 当前系统状态向量
 * @param state_dim 状态维度
 * @param control_input 控制输入向量
 * @param control_dim 控制维度
 * @return float 异常分数(0~∞，越大越异常)
 */
float emergency_stop_cfc_predict(EmergencyStopSystem* system,
                                  const float* system_state, int state_dim,
                                  const float* control_input, int control_dim);

/**
 * @brief 获取当前紧急停止状态
 *
 * @param system 系统句柄
 * @param status 输出状态
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_get_status(const EmergencyStopSystem* system,
                               EmergencyStopStatus* status);

/**
 * @brief 获取所有已触发的条件列表
 *
 * @param system 系统句柄
 * @param triggers 输出触发条件数组
 * @param max_count 最大数量
 * @return int 实际触发数
 */
int emergency_stop_get_triggered_list(const EmergencyStopSystem* system,
                                       EmergencyTrigger* triggers, int max_count);

/**
 * @brief 清除紧急停止状态（手动重置）
 *
 * @param system 系统句柄
 * @param password 授权密码
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_clear(EmergencyStopSystem* system, const char* password);

/**
 * @brief 禁用/启用紧急停止系统
 *
 * @param system 系统句柄
 * @param disable 1=禁用, 0=启用
 * @param password 授权密码
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_set_disabled(EmergencyStopSystem* system,
                                 int disable, const char* password);

/**
 * @brief 保存紧急停止配置
 *
 * @param system 系统句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int emergency_stop_save(const EmergencyStopSystem* system, const char* filepath);

/**
 * @brief 加载紧急停止配置
 *
 * @param filepath 文件路径
 * @return EmergencyStopSystem* 成功返回句柄，失败返回NULL
 */
EmergencyStopSystem* emergency_stop_load(const char* filepath);

/**
 * @brief 获取默认紧急停止配置
 *
 * @return EmergencyStopConfig 默认配置
 */
EmergencyStopConfig emergency_stop_default_config(void);

/**
 * @brief 注册硬件停止回调
 *
 * @param system 系统句柄
 * @param callback 硬件停止回调函数(level, ctx)
 * @param ctx 回调上下文
 * @return int 成功返回0
 */
int emergency_stop_register_hardware_callback(EmergencyStopSystem* system,
                                               void (*callback)(int level, void* ctx),
                                               void* ctx);

/**
 * @brief 注册任务调度器引用（用于暂停/恢复/终止）
 */
int emergency_stop_register_task_scheduler(EmergencyStopSystem* system,
                                            void* sched_ref,
                                            void (*pause_fn)(void*),
                                            void (*resume_fn)(void*),
                                            void (*kill_fn)(void*));

/**
 * @brief 注册线程池引用（用于全部终止）
 */
int emergency_stop_register_thread_pool(EmergencyStopSystem* system,
                                         void* pool_ref,
                                         void (*stop_all_fn)(void*));

/**
 * @brief 注册物理断电函数（GPIO/串口）
 */
int emergency_stop_register_power_cut(EmergencyStopSystem* system,
                                       void (*cut_fn)(void));

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_EMERGENCY_STOP_H */
