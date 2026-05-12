/**
 * @file long_term.h
 * @brief 长期记忆系统接口
 * 
 * 长期记忆管理，支持持久存储和大容量记忆。
 */

#ifndef SELFLNN_LONG_TERM_H
#define SELFLNN_LONG_TERM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 长期记忆配置
 */
typedef struct {
    size_t capacity;           /**< 最大记忆容量 */
    float persistence;         /**< 持久性因子 */
    float retrieval_cost;      /**< 检索代价 */
    float consolidation_rate;  /**< 巩固率 */
} LongTermMemoryConfig;

/**
 * @brief 长期记忆句柄
 */
typedef struct LongTermMemory LongTermMemory;

/**
 * @brief 创建长期记忆实例
 * 
 * @param config 长期记忆配置
 * @return LongTermMemory* 长期记忆句柄，失败返回NULL
 */
LongTermMemory* long_term_memory_create(const LongTermMemoryConfig* config);

/**
 * @brief 释放长期记忆实例
 * 
 * @param memory 长期记忆句柄
 */
void long_term_memory_free(LongTermMemory* memory);

/**
 * @brief 存储长期记忆
 * 
 * @param memory 长期记忆句柄
 * @param key 记忆键
 * @param data 记忆数据
 * @param data_size 数据大小
 * @param strength 初始强度
 * @return int 成功返回0，失败返回-1
 */
int long_term_memory_store(LongTermMemory* memory, const char* key, 
                          const float* data, size_t data_size, float strength);

/**
 * @brief 检索长期记忆
 * 
 * @param memory 长期记忆句柄
 * @param key 记忆键
 * @param data 数据输出缓冲区
 * @param data_size 缓冲区大小
 * @param strength 记忆强度输出缓冲区
 * @return int 成功返回0，未找到返回-1
 */
int long_term_memory_retrieve(LongTermMemory* memory, const char* key,
                             float* data, size_t data_size, float* strength);

/**
 * @brief 更新长期记忆
 * 
 * @param memory 长期记忆句柄
 * @param key 记忆键
 * @param data 新数据
 * @param data_size 数据大小
 * @param strength_delta 强度变化量
 * @return int 成功返回0，失败返回-1
 */
int long_term_memory_update(LongTermMemory* memory, const char* key,
                           const float* data, size_t data_size, float strength_delta);

/**
 * @brief 遗忘长期记忆
 * 
 * @param memory 长期记忆句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int long_term_memory_forget(LongTermMemory* memory, const char* key);

/**
 * @brief 强化长期记忆
 * 
 * @param memory 长期记忆句柄
 * @param key 记忆键
 * @param strength_delta 强度变化量
 * @return int 成功返回0，失败返回-1
 */
int long_term_memory_strengthen(LongTermMemory* memory, const char* key, float strength_delta);

/**
 * @brief 获取长期记忆配置
 * 
 * @param memory 长期记忆句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int long_term_memory_get_config(const LongTermMemory* memory, LongTermMemoryConfig* config);

/**
 * @brief 设置长期记忆配置
 * 
 * @param memory 长期记忆句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int long_term_memory_set_config(LongTermMemory* memory, const LongTermMemoryConfig* config);

/**
 * @brief 获取长期记忆统计信息
 * 
 * @param memory 长期记忆句柄
 * @param usage 使用率输出缓冲区
 * @param avg_strength 平均强度输出缓冲区
 * @param persistence 当前持久性输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int long_term_memory_get_stats(const LongTermMemory* memory, float* usage,
                              float* avg_strength, float* persistence);

/**
 * @brief 重置长期记忆
 * 
 * @param memory 长期记忆句柄
 */
void long_term_memory_reset(LongTermMemory* memory);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_LONG_TERM_H