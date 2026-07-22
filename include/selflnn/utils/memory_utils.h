/**
 * @file memory_utils.h
 * @brief 内存工具库
 * 
 * 提供内存管理、分配、池管理和内存安全检查功能。
 */

#ifndef SELFLNN_MEMORY_UTILS_H
#define SELFLNN_MEMORY_UTILS_H

#include <stddef.h>
#include <stdlib.h>

/* 清理释放宏 — 安全释放单字段
 * DEEP-FIX: 使用 safe_free 替代裸 free，与 DEEP_COPY_STRING 等宏保持一致，
 * 防止 mixed allocator (safe_malloc + raw free) 导致的堆损坏 */
#define DEEP_COPY_CLEANUP_FREE(ptr) do { \
    if (ptr) { safe_free((void**)&(ptr)); } \
} while(0)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 内存分配对齐方式
 */
typedef enum {
    MEM_ALIGN_DEFAULT = 0,    /**< 默认对齐（系统默认） */
    MEM_ALIGN_16 = 16,        /**< 16字节对齐 */
    MEM_ALIGN_32 = 32,        /**< 32字节对齐 */
    MEM_ALIGN_64 = 64,        /**< 64字节对齐 */
    MEM_ALIGN_128 = 128,      /**< 128字节对齐 */
    MEM_ALIGN_256 = 256,      /**< 256字节对齐 */
    MEM_ALIGN_512 = 512,      /**< 512字节对齐 */
    MEM_ALIGN_1024 = 1024,    /**< 1024字节对齐 */
    MEM_ALIGN_PAGE = 4096     /**< 页面对齐（4KB） */
} MemAlignType;

/**
 * @brief 内存池配置
 */
typedef struct {
    size_t block_size;        /**< 块大小 */
    size_t num_blocks;        /**< 块数量 */
    int enable_zero_init;     /**< 是否启用零初始化 */
    int enable_guard_pages;   /**< 是否启用保护页 */
    MemAlignType alignment;   /**< 内存对齐 */
} MemoryPoolConfig;

/**
 * @brief 内存池句柄
 */
typedef struct MemoryPool MemoryPool;

/**
 * @brief 内存统计信息
 */
typedef struct {
    size_t total_allocated;   /**< 总分配内存 */
    size_t total_freed;       /**< 总释放内存 */
    size_t peak_usage;        /**< 峰值使用量 */
    size_t current_usage;     /**< 当前使用量 */
    size_t allocation_count;  /**< 分配次数 */
    size_t free_count;        /**< 释放次数 */
    size_t pool_count;        /**< 内存池数量 */
} MemoryStats;

/**
 * @brief 内部安全内存分配（带文件/行号跟踪）
 * 
 * @param size 分配大小
 * @param file 分配文件名（由宏自动传入）
 * @param line 分配行号（由宏自动传入）
 * @return void* 分配的内存指针，失败返回NULL
 */
void* _safe_malloc(size_t size, const char* file, int line);

/**
 * @brief 内部安全内存分配：分配并清零内存（类似calloc，带文件/行号跟踪）
 * 
 * @param num 元素数量
 * @param size 元素大小
 * @param file 分配文件名（由宏自动传入）
 * @param line 分配行号（由宏自动传入）
 * @return void* 分配的内存指针，失败返回NULL
 */
void* _safe_calloc(size_t num, size_t size, const char* file, int line);

/**
 * @brief 内部安全内存分配：分配对齐内存（带文件/行号跟踪）
 * 
 * @param size 分配大小
 * @param alignment 对齐要求
 * @param file 分配文件名（由宏自动传入）
 * @param line 分配行号（由宏自动传入）
 * @return void* 分配的内存指针，失败返回NULL
 */
void* _safe_aligned_malloc(size_t size, size_t alignment, const char* file, int line);

/**
 * @brief 安全内存分配：分配内存并初始化为零
 * 
 * @param size 分配大小
 * @return void* 分配的内存指针，失败返回NULL
 */
#ifdef USE_STANDARD_ALLOC
#  define safe_malloc(size) malloc(size)
#else
#  define safe_malloc(size) _safe_malloc(size, __FILE__, __LINE__)
#endif

/**
 * @brief 安全内存分配：分配并清零内存（类似calloc）
 * 
 * @param num 元素数量
 * @param size 元素大小
 * @return void* 分配的内存指针，失败返回NULL
 */
#ifdef USE_STANDARD_ALLOC
#  define safe_calloc(num, size) calloc(num, size)
#else
#  define safe_calloc(num, size) _safe_calloc(num, size, __FILE__, __LINE__)
#endif

/**
 * @brief 安全内存分配：分配对齐内存
 * 
 * @param size 分配大小
 * @param alignment 对齐要求
 * @return void* 分配的内存指针，失败返回NULL
 */
#ifdef USE_STANDARD_ALLOC
#  define safe_aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#else
#  define safe_aligned_malloc(size, alignment) _safe_aligned_malloc(size, alignment, __FILE__, __LINE__)
#endif

/**
 * @brief 安全内存释放：释放内存并置空指针
 * 
 * @param ptr 内存指针的地址
 */
void safe_free(void** ptr);

/* P6-R90: enable bypass to standard malloc/free for debugging */
void memory_utils_bypass_safe_alloc(int bypass);

/* v9.1: 验证所有已跟踪内存块的完整性，返回损坏块数 */
int memory_utils_validate_all_tracked(void);

/* v9.1: 返回当前已跟踪的分配数量 */
size_t memory_utils_tracked_count(void);

/**
 * @brief 安全内存重新分配
 * 
 * @param ptr 原始内存指针
 * @param size 新大小
 * @return void* 重新分配的内存指针，失败返回NULL
 */
void* safe_realloc(void* ptr, size_t size);

/**
 * @brief 安全内存复制：带边界检查
 * 
 * @param dest 目标缓冲区
 * @param dest_size 目标缓冲区大小
 * @param src 源缓冲区
 * @param count 复制数量
 * @return int 成功返回0，失败返回-1
 */
int safe_memcpy(void* dest, size_t dest_size, const void* src, size_t count);

/**
 * @brief 安全内存设置：带边界检查
 * 
 * @param dest 目标缓冲区
 * @param dest_size 目标缓冲区大小
 * @param value 设置的值
 * @param count 设置数量
 * @return int 成功返回0，失败返回-1
 */
int safe_memset(void* dest, size_t dest_size, int value, size_t count);

/**
 * @brief 创建内存池
 * 
 * @param config 内存池配置
 * @return MemoryPool* 内存池句柄，失败返回NULL
 */
MemoryPool* memory_pool_create(const MemoryPoolConfig* config);

/**
 * @brief 从内存池分配内存
 * 
 * @param pool 内存池句柄
 * @param size 分配大小
 * @return void* 分配的内存指针，失败返回NULL
 */
void* memory_pool_alloc(MemoryPool* pool, size_t size);

/**
 * @brief 释放内存池中的内存
 * 
 * @param pool 内存池句柄
 * @param ptr 内存指针
 * @return int 成功返回0，失败返回-1
 */
int memory_pool_free(MemoryPool* pool, void* ptr);

/**
 * @brief 销毁内存池
 * 
 * @param pool 内存池句柄
 */
void memory_pool_destroy(MemoryPool* pool);

/**
 * @brief 获取内存统计信息
 * 
 * @param stats 统计信息输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int memory_utils_get_stats(MemoryStats* stats);

/**
 * @brief 重置内存统计信息
 */
void memory_reset_stats(void);

/**
 * @brief 内存泄漏检测：检查是否有未释放的内存
 * 
 * @return size_t 未释放的内存大小，0表示无泄漏
 */
size_t memory_check_leaks(void);

/**
 * @brief 内存碎片整理（如果支持）
 * 
 * @return int 成功返回整理的内存大小，失败返回-1
 */
int memory_defragment(void);

/**
 * @brief 内存访问安全检查
 * 
 * @param ptr 内存指针
 * @param size 访问大小
 * @return int 安全返回1，不安全返回0
 */
int memory_safe_access(const void* ptr, size_t size);

/**
 * @brief 验证所有已分配内存块的完整性
 * @return 0 所有块完好，-1 发现损坏
 */
int selflnn_validate_all_allocations(void);

/**
 * @brief 获取内存页大小
 * 
 * @return size_t 内存页大小
 */
size_t memory_get_page_size(void);

/**
 * @brief 锁定内存到物理RAM（防止交换）
 * 
 * @param ptr 内存指针
 * @param size 内存大小
 * @return int 成功返回0，失败返回-1
 */
int memory_lock(void* ptr, size_t size);

/**
 * @brief 解锁内存
 * 
 * @param ptr 内存指针
 * @param size 内存大小
 * @return int 成功返回0，失败返回-1
 */
int memory_unlock(void* ptr, size_t size);

/* 深拷贝宏定义 (用于知识库和决策引擎)
 * DEEP-FIX: 使用 safe_free 替代 raw free，统一分配器，
 * 防止 mixed allocator (safe_malloc + raw free) 导致的堆损坏
 * 使用 memcpy + 手动null终止替代strcpy，防止缓冲区溢出 */
#define DEEP_COPY_STRING(dest, src) do { \
    if (dest) { safe_free((void**)&(dest)); } \
    if (src) { \
        size_t _len = strlen(src); \
        dest = (char*)safe_malloc(_len + 1); \
        if (dest) { memcpy(dest, src, _len); dest[_len] = '\0'; } \
    } \
} while(0)

#define DEEP_COPY_SCALAR(dest, src) do { (dest) = (src); } while(0)

#define DEEP_COPY_BLOB(dest, size_dest, src, size_src) do { \
    if (dest) { safe_free((void**)&(dest)); (size_dest) = 0; } \
    if (src && (size_src) > 0) { \
        dest = (char*)safe_malloc(size_src); \
        if (dest) { memcpy(dest, src, size_src); (size_dest) = (size_src); } \
    } \
} while(0)

#define DEEP_COPY_STRING_SAFE(dest, src, free_fn) do { \
    if (dest) { free_fn((void**)&(dest)); dest = NULL; } \
    if (src) { \
        size_t _len = strlen(src); \
        dest = (char*)safe_malloc(_len + 1); \
        if (dest) { memcpy(dest, src, _len); dest[_len] = '\0'; } \
    } \
} while(0)

/* float数组深拷贝 - DEEP-FIX: 使用safe_malloc/safe_free */
#define DEEP_COPY_FLOAT_ARRAY(dest, dest_count, src, src_count) do { \
    if (dest) { safe_free((void**)&(dest)); (dest_count) = 0; } \
    if (src && (src_count) > 0) { \
        dest = (float*)safe_malloc((src_count) * sizeof(float)); \
        if (dest) { memcpy(dest, src, (src_count) * sizeof(float)); (dest_count) = (src_count); } \
    } \
} while(0)

/* INCON-03: 删除重复的 memory_utils_bypass_safe_alloc 声明（已在145行保留） */

#ifdef __cplusplus
}
#endif

#endif