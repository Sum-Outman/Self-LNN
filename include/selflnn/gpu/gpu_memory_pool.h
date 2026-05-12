/**
 * @file gpu_memory_pool.h
 * @brief GPU内存池管理系统
 * 
 * GPU内存池管理，减少内存碎片，提高内存分配效率。
 * 支持多种内存类型：主机内存、设备内存、统一内存。
 * 提供内存统计、碎片整理和智能分配功能。
 */

#ifndef SELFLNN_GPU_MEMORY_POOL_H
#define SELFLNN_GPU_MEMORY_POOL_H

#include "selflnn/gpu/gpu.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 内存池分配策略
 */
typedef enum {
    GPU_POOL_STRATEGY_FIRST_FIT = 0,    /**< 首次适应 */
    GPU_POOL_STRATEGY_BEST_FIT,         /**< 最佳适应 */
    GPU_POOL_STRATEGY_WORST_FIT,        /**< 最差适应 */
    GPU_POOL_STRATEGY_BUDDY             /**< 伙伴系统 */
} GpuPoolStrategy;

/**
 * @brief 内存块状态
 */
typedef enum {
    GPU_BLOCK_FREE = 0,                 /**< 空闲 */
    GPU_BLOCK_USED,                     /**< 使用中 */
    GPU_BLOCK_RESERVED                  /**< 保留 */
} GpuBlockStatus;

/**
 * @brief 内存块信息
 */
typedef struct GpuMemoryBlock {
    void* address;                      /**< 内存地址 */
    size_t size;                        /**< 块大小 */
    size_t actual_size;                 /**< 实际分配大小（包括对齐） */
    GpuBlockStatus status;              /**< 块状态 */
    GpuMemoryType memory_type;          /**< 内存类型 */
    struct GpuMemoryBlock* prev;        /**< 前一个块 */
    struct GpuMemoryBlock* next;        /**< 下一个块 */
    void* user_data;                    /**< 用户数据 */
    unsigned int allocation_id;         /**< 分配ID */
} GpuMemoryBlock;

/**
 * @brief 内存池统计信息
 */
typedef struct {
    size_t total_memory;                /**< 总内存大小 */
    size_t used_memory;                 /**< 已使用内存 */
    size_t free_memory;                 /**< 空闲内存 */
    size_t largest_free_block;          /**< 最大空闲块大小 */
    size_t allocation_count;            /**< 分配次数 */
    size_t free_count;                  /**< 释放次数 */
    size_t fragmentation;               /**< 碎片化程度（0-100） */
    size_t overhead_memory;             /**< 管理开销内存 */
    size_t total_expansions;            /**< 总扩展次数 */
    size_t total_expanded_memory;       /**< 总扩展内存大小 */
    size_t allocation_failures;         /**< 分配失败次数 */
    size_t defrag_count;                /**< 碎片整理次数 */
    size_t defrag_freed_memory;         /**< 碎片整理释放的总内存 */
    size_t compact_move_count;          /**< 压缩移动块次数 */
    double avg_fragmentation;           /**< 平均碎片化程度 */
} GpuPoolStatistics;

/**
 * @brief 内存池配置
 */
typedef struct {
    size_t initial_size;                /**< 初始池大小（0表示动态增长） */
    size_t max_size;                    /**< 最大池大小（0表示无限制） */
    size_t alignment;                   /**< 内存对齐要求 */
    GpuPoolStrategy strategy;           /**< 分配策略 */
    int enable_defragmentation;         /**< 是否启用碎片整理 */
    size_t defragmentation_threshold;   /**< 碎片整理阈值（百分比） */
    int track_statistics;               /**< 是否跟踪统计信息 */
    int zero_memory_on_alloc;           /**< 分配时是否清零内存 */
    int zero_memory_on_free;            /**< 释放时是否清零内存 */
    int auto_expand;                    /**< 是否允许自动扩展内存池 */
    int allow_memory_moving;            /**< 是否允许内存移动以优化碎片 */
    size_t min_merge_size;              /**< 最小合并大小（字节） */
    size_t defrag_threshold;            /**< 碎片整理触发阈值（百分比） */
    int defrag_trigger_on_failure;      /**< 分配失败时自动触发碎片整理 */
    size_t max_failures_before_defrag;  /**< 触发碎片整理的最大失败次数 */
    size_t compact_batch_size;          /**< 每轮压缩移动的最大块数 */
} GpuPoolConfig;

/**
 * @brief 内存池句柄
 */
typedef struct GpuMemoryPool GpuMemoryPool;

/**
 * @brief 创建GPU内存池
 * 
 * @param context GPU上下文
 * @param memory_type 内存类型
 * @param config 内存池配置
 * @return GpuMemoryPool* 内存池句柄，失败返回NULL
 */
GpuMemoryPool* gpu_memory_pool_create(GpuContext* context, 
                                      GpuMemoryType memory_type,
                                      const GpuPoolConfig* config);

/**
 * @brief 销毁GPU内存池
 * 
 * @param pool 内存池句柄
 */
void gpu_memory_pool_destroy(GpuMemoryPool* pool);

/**
 * @brief 从内存池分配内存
 * 
 * @param pool 内存池句柄
 * @param size 请求大小
 * @param alignment 对齐要求（0表示使用池默认对齐）
 * @return void* 分配的内存地址，失败返回NULL
 */
void* gpu_memory_pool_alloc(GpuMemoryPool* pool, size_t size, size_t alignment);

/**
 * @brief 释放内存到内存池
 * 
 * @param pool 内存池句柄
 * @param ptr 要释放的内存地址
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_pool_free(GpuMemoryPool* pool, void* ptr);

/**
 * @brief 重新分配内存
 * 
 * @param pool 内存池句柄
 * @param ptr 原内存地址
 * @param new_size 新大小
 * @return void* 新内存地址，失败返回NULL
 */
void* gpu_memory_pool_realloc(GpuMemoryPool* pool, void* ptr, size_t new_size);

/**
 * @brief 获取内存池统计信息
 * 
 * @param pool 内存池句柄
 * @param stats 统计信息输出
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_pool_get_statistics(GpuMemoryPool* pool, GpuPoolStatistics* stats);

/**
 * @brief 执行内存池碎片整理
 * 
 * @param pool 内存池句柄
 * @return int 整理后释放的内存大小，失败返回-1
 */
size_t gpu_memory_pool_defragment(GpuMemoryPool* pool);

/**
 * @brief 验证内存池完整性
 * 
 * @param pool 内存池句柄
 * @return int 有效返回0，发现错误返回错误码
 */
int gpu_memory_pool_validate(GpuMemoryPool* pool);

/**
 * @brief 清空内存池（释放所有分配）
 * 
 * @param pool 内存池句柄
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_pool_clear(GpuMemoryPool* pool);

/**
 * @brief 获取内存块信息
 * 
 * @param pool 内存池句柄
 * @param ptr 内存地址
 * @param block 块信息输出
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_pool_get_block_info(GpuMemoryPool* pool, void* ptr, GpuMemoryBlock* block);

/**
 * @brief 设置内存池配置
 * 
 * @param pool 内存池句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_pool_set_config(GpuMemoryPool* pool, const GpuPoolConfig* config);

/**
 * @brief 获取默认内存池配置
 * 
 * @param config 配置输出
 */
void gpu_memory_pool_default_config(GpuPoolConfig* config);

/**
 * @brief 创建与GPU后端兼容的内存池包装器
 * 
 * @param backend GPU后端接口
 * @param context GPU上下文
 * @param memory_type 内存类型
 * @return GpuMemoryPool* 内存池句柄
 */
GpuMemoryPool* gpu_memory_pool_create_from_backend(const GpuBackendInterface* backend,
                                                   GpuContext* context,
                                                   GpuMemoryType memory_type);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GPU_MEMORY_POOL_H */