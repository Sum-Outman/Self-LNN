#ifndef SELFLNN_CAPABILITY_SWITCH_H
#define SELFLNN_CAPABILITY_SWITCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CAP_SELF_COGNITION = 0,
    CAP_SELF_DECISION = 1,
    CAP_AUTONOMOUS_EXECUTION = 2,
    CAP_SELF_LEARNING = 3,
    CAP_SELF_EVOLUTION = 4,
    CAP_IMITATION_LEARNING = 5,
    CAP_SELF_CORRECTION = 6,
    CAP_REFLECTION = 7,
    CAP_CURIOSITY = 8,
    CAP_PLANNING = 9,
    CAP_DIALOGUE = 10,
    CAP_CONCURRENCY = 11,
    CAP_COUNT
} CapabilityType;

const char* capability_get_name(CapabilityType type);

const char* capability_get_description(CapabilityType type);

int capability_is_enabled(CapabilityType type);

int capability_set_enabled(CapabilityType type, int enabled);

int capability_enable(CapabilityType type);

int capability_disable(CapabilityType type);

int capability_toggle(CapabilityType type);

int capability_enable_all(void);

int capability_disable_all(void);

int capability_get_states(int* states, size_t max_count);

int capability_set_states(const int* states, size_t count);

int capability_get_enabled_count(void);

int capability_reset_to_defaults(void);

typedef int (*CapabilityCheckFunc)(void);

typedef int (*CapabilitySetFunc)(int enable);

int capability_register_module(CapabilityType type,
                                CapabilityCheckFunc check_func,
                                CapabilitySetFunc set_func);

int capability_force_on(CapabilityType type);

int capability_unforce(CapabilityType type);

/**
 * @brief K-023: 能力开关完整状态诊断
 *
 * 输出所有12种能力开关的完整状态信息到JSON字符串。
 * 包括：名称、启用状态、强制锁定状态、子系统实际可用性、描述。
 *
 * @param json_buffer 输出JSON字符串缓冲区
 * @param buffer_size 缓冲区大小
 * @return 成功返回写入的JSON字节数，失败返回-1
 */
int capability_diagnose_all(char* json_buffer, size_t buffer_size);

/**
 * @brief K-023: 获取能力开关运行时健康报告
 *
 * 验证12种能力的底层子系统是否真正可用（非仅状态位）。
 * 例如"自我学习"要求在线学习器已创建，"自我决策"要求LNN已初始化。
 *
 * @return 返回健康能力数（0-12），-1表示错误
 */
int capability_health_check(void);

/**
 * @brief 获取能力的依赖项列表
 *
 * 某些能力依赖其他能力先启用才能正常工作。
 * 例如"自主执行"依赖"自我决策"先启用以决定执行什么。
 *
 * @param type 能力类型
 * @param dependencies 输出依赖数组（调用者分配CAP_COUNT个int空间）
 * @param max_count 数组最大长度
 * @return 实际依赖数量，-1失败
 */
int capability_get_dependencies(CapabilityType type, int* dependencies, int max_count);

/**
 * @brief 检查能力冲突
 *
 * 检测两个能力之间是否存在不可同时启用的冲突关系。
 * 例如"自我演化"运行时不应同时禁用"自我学习"。
 *
 * @param type_a 能力A
 * @param type_b 能力B
 * @return 1=存在冲突，0=无冲突，-1参数无效
 */
int capability_check_conflict(CapabilityType type_a, CapabilityType type_b);

/**
 * @brief 带依赖检查的能力启用
 *
 * 启用指定能力时自动检查并启用其所有前置依赖，
 * 同时检查是否会引入冲突。如果冲突则操作失败。
 *
 * @param type 能力类型
 * @return 0成功，-1失败（依赖不足或冲突）
 */
int capability_enable_safe(CapabilityType type);

/**
 * @brief 带影响检查的能力禁用
 *
 * 禁用指定能力前检查是否有其他能力依赖它。
 * 如果存在依赖者则操作失败。
 *
 * @param type 能力类型
 * @return 0成功，-1失败（被其他能力依赖）
 */
int capability_disable_safe(CapabilityType type);

#ifdef __cplusplus
}
#endif

#endif
