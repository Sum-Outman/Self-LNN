/**
 * @file short_term.h
 * @brief 短期记忆系统接口
 * 
 * 短期记忆管理，支持快速存取和有限容量记忆存储。
 */

#ifndef SELFLNN_SHORT_TERM_H
#define SELFLNN_SHORT_TERM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 短期记忆配置
 */
typedef struct {
    size_t capacity;           /**< 最大记忆容量 */
    float decay_rate;          /**< 记忆衰减率 */
    float retrieval_speed;     /**< 检索速度 */
    float consolidation_threshold; /**< 巩固阈值 */
} ShortTermMemoryConfig;

/**
 * @brief 短期记忆句柄
 */
typedef struct ShortTermMemory ShortTermMemory;

/**
 * @brief 创建短期记忆实例
 * 
 * @param config 短期记忆配置
 * @return ShortTermMemory* 短期记忆句柄，失败返回NULL
 */
ShortTermMemory* short_term_memory_create(const ShortTermMemoryConfig* config);

/**
 * @brief 释放短期记忆实例
 * 
 * @param memory 短期记忆句柄
 */
void short_term_memory_free(ShortTermMemory* memory);

/**
 * @brief 存储短期记忆
 * 
 * @param memory 短期记忆句柄
 * @param key 记忆键
 * @param data 记忆数据
 * @param data_size 数据大小
 * @param strength 初始强度
 * @return int 成功返回0，失败返回-1
 */
int short_term_memory_store(ShortTermMemory* memory, const char* key, 
                           const float* data, size_t data_size, float strength);

/**
 * @brief 检索短期记忆
 * 
 * @param memory 短期记忆句柄
 * @param key 记忆键
 * @param data 数据输出缓冲区
 * @param data_size 缓冲区大小
 * @param strength 记忆强度输出缓冲区
 * @return int 成功返回0，未找到返回-1
 */
int short_term_memory_retrieve(ShortTermMemory* memory, const char* key,
                              float* data, size_t data_size, float* strength);

/**
 * @brief 更新短期记忆
 * 
 * @param memory 短期记忆句柄
 * @param key 记忆键
 * @param data 新数据
 * @param data_size 数据大小
 * @param strength_delta 强度变化量
 * @return int 成功返回0，失败返回-1
 */
int short_term_memory_update(ShortTermMemory* memory, const char* key,
                            const float* data, size_t data_size, float strength_delta);

/**
 * @brief 遗忘短期记忆
 * 
 * @param memory 短期记忆句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int short_term_memory_forget(ShortTermMemory* memory, const char* key);

/**
 * @brief 巩固短期记忆到长期记忆
 * 
 * @param memory 短期记忆句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int short_term_memory_consolidate(ShortTermMemory* memory, const char* key);

/**
 * @brief 获取短期记忆配置
 * 
 * @param memory 短期记忆句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int short_term_memory_get_config(const ShortTermMemory* memory, ShortTermMemoryConfig* config);

/**
 * @brief 设置短期记忆配置
 * 
 * @param memory 短期记忆句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int short_term_memory_set_config(ShortTermMemory* memory, const ShortTermMemoryConfig* config);

/**
 * @brief 获取短期记忆统计信息
 * 
 * @param memory 短期记忆句柄
 * @param usage 使用率输出缓冲区
 * @param avg_strength 平均强度输出缓冲区
 * @param decay_rate 当前衰减率输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int short_term_memory_get_stats(const ShortTermMemory* memory, float* usage,
                               float* avg_strength, float* decay_rate);

/**
 * @brief 重置短期记忆
 * 
 * @param memory 短期记忆句柄
 */
void short_term_memory_reset(ShortTermMemory* memory);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_SHORT_TERM_H