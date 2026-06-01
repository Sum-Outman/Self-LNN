/**
 * @file safety_monitor.h
 * @brief AGI安全监控系统接口
 */

#ifndef SELFLNN_SAFETY_MONITOR_H
#define SELFLNN_SAFETY_MONITOR_H

#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 安全级别 */
typedef enum {
    SAFETY_LEVEL_NORMAL = 0,     /* 正常运行 */
    SAFETY_LEVEL_WARNING = 1,    /* 警告：检测到潜在风险 */
    SAFETY_LEVEL_ELEVATED = 2,   /* 升级：需要人工关注 */
    SAFETY_LEVEL_CRITICAL = 3,   /* 危险：自动限制 */
    SAFETY_LEVEL_EMERGENCY = 4   /* 紧急：立即停止 */
} SafetyLevel;

/* 安全事件类型 */
typedef enum {
    SAFETY_EVENT_BEHAVIOR_ANOMALY = 0,   /* 行为异常 */
    SAFETY_EVENT_RESOURCE_OVERUSE = 1,   /* 资源过度使用 */
    SAFETY_EVENT_PHYSICAL_VIOLATION = 2, /* 物理约束违反 */
    SAFETY_EVENT_BOUNDARY_CROSSING = 3,  /* 安全边界越界 */
    SAFETY_EVENT_COMMAND_INJECTION = 4,  /* 命令注入风险 */
    SAFETY_EVENT_LOOP_DETECTED = 5,      /* 检测到死循环 */
    SAFETY_EVENT_MEMORY_LEAK = 6,        /* 内存泄漏 */
    SAFETY_EVENT_NETWORK_ANOMALY = 7,    /* 网络异常 */
    SAFETY_EVENT_SENSOR_INVALID = 8,     /* 传感器数据无效 */
    SAFETY_EVENT_DECISION_CONFLICT = 9   /* 决策冲突 */
} SafetyEventType;

/* 安全事件 */
typedef struct {
    SafetyEventType type;
    SafetyLevel severity;
    char description[512];
    char source[128];
    time_t timestamp;
    int handled;
    int auto_resolved;
} SafetyEvent;

/* 安全规则 */
typedef struct {
    char name[128];
    char description[256];
    SafetyEventType target_type;
    float threshold;
    float violation_count;
    float violation_limit;
    int enabled;
    int auto_action;
    int action_type;
} SafetyRule;

/* 资源使用限制 */
typedef struct {
    float cpu_usage_max;          /* CPU使用率上限 */
    float gpu_usage_max;          /* GPU使用率上限 */
    float memory_usage_max;       /* 内存使用率上限 */
    float disk_usage_max;         /* 磁盘使用率上限 */
    float network_bandwidth_max;  /* 网络带宽上限 */
    int max_threads;              /* 最大线程数 */
    int max_connections;          /* 最大连接数 */
    float max_power_watts;        /* 最大功率（瓦） */
} ResourceLimits;

/* 物理安全边界 */
typedef struct {
    float max_velocity;           /* 最大速度 m/s */
    float max_acceleration;       /* 最大加速度 m/s² */
    float max_torque;             /* 最大扭矩 Nm */
    float max_joint_angle[16];    /* 各关节最大角度 */
    float min_joint_angle[16];    /* 各关节最小角度 */
    float safety_zone_radius;     /* 安全区域半径 m */
    float collision_distance_min; /* 最小碰撞距离 m */
    int joint_count;              /* 关节数量 */
} PhysicalBoundaries;

/* 安全审计日志条目 */
typedef struct {
    time_t timestamp;
    SafetyEventType event_type;
    SafetyLevel level;
    char action[256];
    char decision[256];
    char result[128];
    float confidence;
} AuditLogEntry;

/* 安全监控统计 */
typedef struct {
    size_t total_events;
    size_t warnings;
    size_t elevated;
    size_t critical;
    size_t emergencies;
    size_t auto_resolved;
    size_t manual_resolved;
    size_t pending;
    float current_safety_score;
    float average_safety_score;
    time_t last_incident_time;
    size_t uptime_without_incident;
} SafetyStats;

/* 安全监控句柄 */
typedef struct SafetyMonitor SafetyMonitor;

/**
 * @brief 安全监控系统内部结构体（完整定义，供直接字段访问使用）
 * safety_monitor.c中定义SELFLNN_SAFETY_IMPL时使用自有完整定义，
 * 其他文件使用此兼容版本（含核心字段用于直接访问）。
 */
#ifndef SELFLNN_SAFETY_IMPL
struct SafetyMonitor {
    SafetyLevel current_level;           /**< 当前安全等级 */
    int emergency_stop_active;           /**< 紧急停止是否激活 */
    int soft_stop_active;                /**< 软停止是否激活 */
    ResourceLimits resource_limits;      /**< 资源限制 */
    PhysicalBoundaries physical_boundaries; /**< 物理边界 */
    void* rules;                         /**< 安全规则数组（不透明） */
    int rule_count;                      /**< 规则数量 */
    void* events;                        /**< 安全事件数组（不透明） */
    int event_count;                     /**< 事件数量 */
};
#endif

/* 紧急停止系统前向声明（避免循环包含） */
struct EmergencyStopSystem;
typedef struct EmergencyStopSystem EmergencyStopSystem;

/**
 * @brief 创建安全监控系统
 */
SafetyMonitor* safety_monitor_create(void);

/**
 * @brief 释放安全监控系统
 */
void safety_monitor_free(SafetyMonitor* monitor);

/* ========== ZSFQQ-DEEP-003: 主动熔断器(Circuit Breaker) ========== */
int safety_circuit_breaker_report_failure(SafetyMonitor* monitor, int subsystem_id);
int safety_circuit_breaker_report_success(SafetyMonitor* monitor, int subsystem_id);
int safety_circuit_breaker_check_allowed(SafetyMonitor* monitor, int subsystem_id);
void safety_circuit_breaker_reset(SafetyMonitor* monitor);
int safety_circuit_breaker_get_state(const SafetyMonitor* monitor);

/**
 * @brief 关联紧急停止系统（安全事件自动触发紧急停止）
 * @param monitor 安全监控句柄
 * @param emergency_system 紧急停止系统句柄
 * @return 0成功，-1失败
 */
int safety_monitor_set_emergency_stop(SafetyMonitor* monitor, EmergencyStopSystem* emergency_system);

/**
 * @brief 设置资源使用限制
 */
int safety_set_resource_limits(SafetyMonitor* monitor, const ResourceLimits* limits);

/**
 * @brief 设置物理安全边界
 */
int safety_set_physical_boundaries(SafetyMonitor* monitor, const PhysicalBoundaries* boundaries);

/**
 * @brief 添加安全规则
 */
int safety_add_rule(SafetyMonitor* monitor, const SafetyRule* rule);

/**
 * @brief 上报安全事件
 */
int safety_report_event(SafetyMonitor* monitor, const SafetyEvent* event);

/**
 * @brief 检查当前安全状态
 */
SafetyLevel safety_check_status(SafetyMonitor* monitor);

/**
 * @brief 请求执行动作前的安全检查
 * @param action_type 动作类型
 * @param params 动作参数
 * @return 0安全，-1危险
 */
int safety_approve_action(SafetyMonitor* monitor, const char* action_type, const void* params);

/**
 * @brief 执行紧急停止
 */
int safety_emergency_stop(SafetyMonitor* monitor);

/**
 * @brief 软停止（允许当前任务完成）
 */
int safety_soft_stop(SafetyMonitor* monitor);

/**
 * @brief 获取安全统计
 */
int safety_get_stats(const SafetyMonitor* monitor, SafetyStats* stats);

/**
 * @brief 获取审计日志
 */
int safety_get_audit_log(const SafetyMonitor* monitor, AuditLogEntry* entries, int max_entries);

/**
 * @brief 验证决策安全性
 */
int safety_validate_decision(SafetyMonitor* monitor, const char* decision_type,
                            const float* decision_params, int param_count);

/**
 * @brief 监控资源使用
 */
int safety_monitor_resources(SafetyMonitor* monitor);

/**
 * @brief 检查物理约束
 */
int safety_check_physics(SafetyMonitor* monitor, const float* position,
                        const float* velocity, const float* joints, int joint_count);

/**
 * @brief 恢复安全状态
 */
int safety_reset(SafetyMonitor* monitor);

/**
 * @brief 动态自适应调整安全阈值
 *
 * 根据近期违规率和系统负载情况自动调整资源限制和物理边界。
 * 高频违规时收紧阈值，系统稳定时逐步放松阈值。
 * 建议周期性调用（如每60秒）。
 *
 * @param monitor 安全监控句柄
 * @return int 成功返回0，失败返回-1
 */
int safety_dynamic_adjust_thresholds(SafetyMonitor* monitor);

/* ================================================================
 * K-022: 安全规则增强 —— 隐私过滤 + 行为约束 + 物理动作验证
 * ================================================================ */

/**
 * @brief K-022: 验证机器人物理动作安全性
 *
 * 综合检查速度、加速度、关节角度、扭矩、碰撞风险。
 * 所有检测基于真实物理边界参数，无虚拟/模拟数据。
 *
 * @param monitor 安全监控句柄
 * @param joint_positions 关节位置数组 [joint_count]
 * @param joint_velocities 关节速度数组 [joint_count]
 * @param joint_torques 关节扭矩数组 [joint_count] (可NULL)
 * @param joint_count 关节数量
 * @return 0=安全，1=警告，2=危险，-1=错误
 */
int safety_validate_physical_action(SafetyMonitor* monitor,
                                     const float* joint_positions,
                                     const float* joint_velocities,
                                     const float* joint_torques,
                                     int joint_count);

/**
 * @brief K-022: 设置数据隐私过滤级别
 *
 * @param monitor 安全监控句柄
 * @param filter_level 过滤级别: 0=不过滤, 1=仅日志, 2=脱敏, 3=完全隔离
 * @param data_types 需过滤的数据类型(位掩码: bit0=图像,bit1=音频,bit2=文本,bit3=传感器)
 * @return 0成功
 */
int safety_set_privacy_filter(SafetyMonitor* monitor, int filter_level, int data_types);

/**
 * @brief K-022: 添加行为约束规则
 *
 * @param monitor 安全监控句柄
 * @param constraint_name 约束名称
 * @param condition 约束条件表达式(如 "velocity < 5.0")
 * @param action_on_violation 违规动作: 0=警告, 1=软停止, 2=紧急停止
 * @return 0成功
 */
int safety_add_behavioral_constraint(SafetyMonitor* monitor,
                                      const char* constraint_name,
                                      const char* condition,
                                      int action_on_violation);

/**
 * @brief K-022: 获取当前隐私过滤状态
 *
 * @param monitor 安全监控句柄
 * @param filter_level [输出] 当前过滤级别
 * @param data_types [输出] 当前过滤数据类型位掩码
 * @param enabled [输出] 过滤是否启用
 * @return 0成功
 */
int safety_get_privacy_filter_status(const SafetyMonitor* monitor,
                                      int* filter_level,
                                      int* data_types,
                                      int* enabled);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SAFETY_MONITOR_H */
