/**
 * @file episodic.h
 * @brief 情景记忆系统接口
 * 
 * 情景记忆管理，支持事件序列和时间戳记忆。
 */

#ifndef SELFLNN_EPISODIC_H
#define SELFLNN_EPISODIC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 情景记忆配置
 */
typedef struct {
    size_t capacity;           /**< 最大事件容量 */
    float temporal_resolution; /**< 时间分辨率 */
    float association_strength; /**< 关联强度 */
    int enable_chaining;       /**< 是否启用事件链 */
} EpisodicMemoryConfig;

/**
 * @brief 情景记忆句柄
 */
typedef struct EpisodicMemory EpisodicMemory;

/**
 * @brief 创建情景记忆实例
 * 
 * @param config 情景记忆配置
 * @return EpisodicMemory* 情景记忆句柄，失败返回NULL
 */
EpisodicMemory* episodic_memory_create(const EpisodicMemoryConfig* config);

/**
 * @brief 释放情景记忆实例
 * 
 * @param memory 情景记忆句柄
 */
void episodic_memory_free(EpisodicMemory* memory);

/**
 * @brief 存储情景记忆事件
 * 
 * @param memory 情景记忆句柄
 * @param event_id 事件ID
 * @param data 事件数据
 * @param data_size 数据大小
 * @param timestamp 时间戳
 * @param strength 初始强度
 * @return int 成功返回0，失败返回-1
 */
int episodic_memory_store_event(EpisodicMemory* memory, const char* event_id,
                               const float* data, size_t data_size,
                               float timestamp, float strength);

/**
 * @brief 检索情景记忆事件
 * 
 * @param memory 情景记忆句柄
 * @param event_id 事件ID
 * @param data 数据输出缓冲区
 * @param data_size 缓冲区大小
 * @param timestamp 时间戳输出缓冲区
 * @param strength 记忆强度输出缓冲区
 * @return int 成功返回0，未找到返回-1
 */
int episodic_memory_retrieve_event(EpisodicMemory* memory, const char* event_id,
                                  float* data, size_t data_size,
                                  float* timestamp, float* strength);

/**
 * @brief 关联两个事件
 * 
 * @param memory 情景记忆句柄
 * @param event_id1 第一个事件ID
 * @param event_id2 第二个事件ID
 * @param association_strength 关联强度
 * @return int 成功返回0，失败返回-1
 */
int episodic_memory_associate_events(EpisodicMemory* memory, const char* event_id1,
                                    const char* event_id2, float association_strength);

/**
 * @brief 检索相关事件
 * 
 * @param memory 情景记忆句柄
 * @param event_id 事件ID
 * @param max_results 最大结果数
 * @param results 结果ID输出缓冲区
 * @param strengths 关联强度输出缓冲区
 * @return int 成功返回相关事件数，失败返回-1
 */
int episodic_memory_retrieve_related(EpisodicMemory* memory, const char* event_id,
                                    size_t max_results, char** results, float* strengths);

/**
 * @brief 按时间范围检索事件
 * 
 * @param memory 情景记忆句柄
 * @param start_time 开始时间
 * @param end_time 结束时间
 * @param max_results 最大结果数
 * @param results 结果ID输出缓冲区
 * @param timestamps 时间戳输出缓冲区
 * @return int 成功返回事件数，失败返回-1
 */
int episodic_memory_retrieve_by_time(EpisodicMemory* memory, float start_time,
                                    float end_time, size_t max_results,
                                    char** results, float* timestamps);

/**
 * @brief 获取情景记忆配置
 * 
 * @param memory 情景记忆句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int episodic_memory_get_config(const EpisodicMemory* memory, EpisodicMemoryConfig* config);

/**
 * @brief 设置情景记忆配置
 * 
 * @param memory 情景记忆句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int episodic_memory_set_config(EpisodicMemory* memory, const EpisodicMemoryConfig* config);

/**
 * @brief 获取情景记忆统计信息
 * 
 * @param memory 情景记忆句柄
 * @param total_events 总事件数输出缓冲区
 * @param avg_association 平均关联强度输出缓冲区
 * @param temporal_density 时间密度输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int episodic_memory_get_stats(const EpisodicMemory* memory, size_t* total_events,
                             float* avg_association, float* temporal_density);

/**
 * @brief 重置情景记忆
 * 
 * @param memory 情景记忆句柄
 */
void episodic_memory_reset(EpisodicMemory* memory);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_EPISODIC_H