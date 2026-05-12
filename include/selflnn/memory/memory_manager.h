/**
 * @file memory_manager.h
 * @brief 记忆管理器接口
 * 
 * 记忆系统总管理器，协调短期、长期、情景和语义记忆。
 */

#ifndef SELFLNN_MEMORY_MANAGER_H
#define SELFLNN_MEMORY_MANAGER_H

#include <stddef.h>

/* 前向声明 */
typedef struct MemorySystem MemorySystem;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 伙伴分配器统计信息
 */
typedef struct {
    size_t pool_size;              /**< 内存池总大小（字节） */
    size_t used_size;              /**< 已使用大小（字节） */
    size_t free_size;              /**< 空闲大小（字节） */
    size_t largest_free_block;     /**< 最大空闲块大小（字节） */
    int total_blocks;              /**< 总块数 */
    int free_blocks;               /**< 空闲块数 */
    float fragmentation_ratio;     /**< 碎片率（0.0~1.0，越小越好） */
    int levels;                    /**< 伙伴级别数 */
    int free_per_level[21];        /**< 每级空闲块数 */
    int defragment_count;          /**< 整理的次数 */
    int page_merge_count;          /**< 页级合并块数 */
    size_t page_threshold;         /**< 页合并阈值（字节） */
    size_t avg_allocation_size;    /**< 平均分配大小（字节） */
    int adaptive_min_shift;        /**< 自适应最小偏移 */
    int overflow_blocks;           /**< 溢出回退块数（malloc回退） */
    size_t overflow_used;          /**< 溢出回退使用量（字节） */
} BuddyAllocatorStats;

/**
 * @brief 记忆管理器配置
 */
typedef struct {
    size_t short_term_capacity;   /**< 短期记忆容量 */
    size_t long_term_capacity;    /**< 长期记忆容量 */
    size_t episodic_capacity;     /**< 情景记忆容量 */
    size_t semantic_capacity;     /**< 语义记忆容量 */
    float consolidation_rate;     /**< 巩固率 */
    int enable_integration;       /**< 是否启用记忆整合 */
    size_t buddy_pool_size;       /**< 伙伴分配器内存池大小（0=使用默认） */
    int enable_buddy_allocator;   /**< 是否启用伙伴分配器 */
} MemoryManagerConfig;

/**
 * @brief 记忆管理器句柄
 */
typedef struct MemoryManager MemoryManager;

/**
 * @brief 创建记忆管理器实例
 * 
 * @param config 记忆管理器配置
 * @return MemoryManager* 记忆管理器句柄，失败返回NULL
 */
MemoryManager* memory_manager_create(const MemoryManagerConfig* config);

/**
 * @brief 释放记忆管理器实例
 * 
 * @param manager 记忆管理器句柄
 */
void memory_manager_free(MemoryManager* manager);

/**
 * @brief 存储记忆（自动选择类型）
 * 
 * @param manager 记忆管理器句柄
 * @param key 记忆键
 * @param data 记忆数据
 * @param data_size 数据大小
 * @param priority 优先级
 * @param strength 初始强度
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_store(MemoryManager* manager, const char* key,
                        const float* data, size_t data_size,
                        int priority, float strength);

/**
 * @brief 检索记忆（自动搜索所有类型）
 * 
 * @param manager 记忆管理器句柄
 * @param key 记忆键
 * @param data 数据输出缓冲区
 * @param data_size 缓冲区大小
 * @param strength 记忆强度输出缓冲区
 * @param memory_type 记忆类型输出缓冲区
 * @return int 成功返回0，未找到返回-1
 */
int memory_manager_retrieve(MemoryManager* manager, const char* key,
                           float* data, size_t data_size,
                           float* strength, int* memory_type);

/**
 * @brief 更新记忆
 * 
 * @param manager 记忆管理器句柄
 * @param key 记忆键
 * @param data 新数据
 * @param data_size 数据大小
 * @param strength_delta 强度变化量
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_update(MemoryManager* manager, const char* key,
                         const float* data, size_t data_size, float strength_delta);

/**
 * @brief 遗忘记忆
 * 
 * @param manager 记忆管理器句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_forget(MemoryManager* manager, const char* key);

/**
 * @brief 巩固记忆（短期到长期）
 * 
 * @param manager 记忆管理器句柄
 * @param key 记忆键
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_consolidate(MemoryManager* manager, const char* key);

/**
 * @brief 整合记忆（跨类型关联）
 * 
 * @param manager 记忆管理器句柄
 * @param key1 第一个记忆键
 * @param key2 第二个记忆键
 * @param association_strength 关联强度
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_integrate(MemoryManager* manager, const char* key1,
                            const char* key2, float association_strength);

/**
 * @brief 获取记忆管理器配置
 * 
 * @param manager 记忆管理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_get_config(const MemoryManager* manager, MemoryManagerConfig* config);

/**
 * @brief 设置记忆管理器配置
 * 
 * @param manager 记忆管理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_set_config(MemoryManager* manager, const MemoryManagerConfig* config);

/**
 * @brief 获取记忆管理器统计信息
 * 
 * @param manager 记忆管理器句柄
 * @param total_memories 总记忆数输出缓冲区
 * @param consolidation_ratio 巩固率输出缓冲区
 * @param integration_level 整合级别输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_get_stats(const MemoryManager* manager, size_t* total_memories,
                            float* consolidation_ratio, float* integration_level);

/**
 * @brief 重置记忆管理器
 * 
 * @param manager 记忆管理器句柄
 */
void memory_manager_reset(MemoryManager* manager);

/**
 * @brief 获取底层记忆系统句柄
 * 
 * 用于需要直接操作记忆系统的场景（如训练采样）。
 * 
 * @param manager 记忆管理器句柄
 * @return MemorySystem* 底层记忆系统句柄，失败返回NULL
 */
MemorySystem* memory_manager_get_system(MemoryManager* manager);

/**
 * @brief 从伙伴分配器分配内存
 * 
 * 从管理器内部的伙伴分配器分配内存块，减少碎片化。
 * 仅当启用伙伴分配器时有效。
 * 
 * @param manager 记忆管理器句柄
 * @param size 请求大小
 * @return void* 成功返回内存地址，失败返回NULL
 */
void* memory_manager_pool_alloc(MemoryManager* manager, size_t size);

/**
 * @brief 释放伙伴分配器内存
 * 
 * 将内存块归还到伙伴分配器，自动合并伙伴块。
 * 
 * @param manager 记忆管理器句柄
 * @param ptr 要释放的内存地址
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_pool_free(MemoryManager* manager, void* ptr);

/**
 * @brief 执行伙伴分配器碎片整理
 * 
 * 扫描所有空闲块，最大可能合并伙伴块以降低碎片率。
 * 
 * @param manager 记忆管理器句柄
 * @return int 成功返回合并的块数，失败返回-1
 */
int memory_manager_pool_defragment(MemoryManager* manager);

/**
 * @brief 获取伙伴分配器统计信息
 * 
 * @param manager 记忆管理器句柄
 * @param stats 统计信息输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int memory_manager_get_buddy_stats(const MemoryManager* manager, BuddyAllocatorStats* stats);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_MEMORY_MANAGER_H