/**
 * @file gpu_internal.h
 * @brief GPU内部类型定义 - 仅供GPU后端实现使用
 * 
 * 这个头文件包含GPU内部结构体的完整定义，允许GPU后端实现
 * 直接访问GpuContext、GpuMemory、GpuKernel、GpuStream等结构体成员。
 * 
 * 注意：这个头文件仅供GPU后端实现文件使用（gpu.c、gpu_cuda.c、
 * gpu_opencl.c、gpu_vulkan.c、gpu_metal.c等）。
 * 公共API应使用gpu.h中的不透明类型定义。
 * 
 * 重要：根据项目要求"禁止任何降级处理"和" 全部深度实现"，
 * 所有GPU后端必须提供真实的硬件加速实现，不能使用简化或模拟实现。
 * 
 * F-046 重构标注: 各GPU后端(gpu_cuda.c/gpu_opencl.c/gpu_vulkan.c等)
 * 有大量相似的结构体定义(GpuContext/GpuMemory/GpuKernel)和函数签名。
 * 待重构方向：将公共部分提取到本文件作为统一抽象层，各后端仅实现
 * 差异化的硬件API调用部分，减少代码重复。
 */

#ifndef SELFLNN_GPU_INTERNAL_H
#define SELFLNN_GPU_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include "selflnn/gpu/gpu.h"
#include "selflnn/gpu/auto_kernel_optimization.h"
#include "selflnn/concurrency/thread_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * GPU内部结构体完整定义
 * =========================================================================== */

/* 前向声明：GpuKernelCacheEntry（在 struct GpuContext 之前声明以供 MSVC 兼容） */
typedef struct GpuKernelCacheEntry GpuKernelCacheEntry;

/**
 * @brief GPU上下文内部结构体
 * 
 * 注意：这个定义必须与gpu.c中的struct GpuContext完全一致。
 * 所有GPU后端实现都使用这个相同的结构体定义来确保兼容性。
 */
struct GpuContext {
    GpuBackend backend;          /**< 后端类型 */
    int device_index;            /**< 设备索引 */
    ThreadPool* thread_pool;     /**< 线程池（CPU后端使用） */
    size_t total_memory;         /**< 总内存（字节） */
    size_t free_memory;          /**< 空闲内存（字节） */
    int is_initialized;          /**< 是否已初始化 */
    char device_name[256];       /**< 设备名称 */
    size_t device_memory_override; /**< 设备内存覆盖值（0=使用真实值） */
    
    /** 后端私有数据指针（由具体后端实现使用） */
    void* backend_data;
    
    /** 自动内核优化器 */
    AutoKernelOptimizer* kernel_optimizer;
    
    /** 内核缓存 */
    GpuKernelCacheEntry* kernel_cache;    /**< 内核缓存条目数组 */
    int kernel_cache_size;                /**< 当前缓存条目数 */
    int kernel_cache_capacity;            /**< 缓存容量 */
    int kernel_cache_hits;                /**< 缓存命中次数 */
    int kernel_cache_misses;              /**< 缓存未命中次数 */
    int kernel_cache_evictions;           /**< 淘汰次数 */
    long cache_timestamp;                 /**< 全局缓存时间戳（LRU用） */

    /* NPU相关字段 */
    void** npu_device_memory;            /**< NPU设备内存缓冲区数组 */
    int    npu_memory_count;             /**< NPU内存缓冲区数量 */
    int    npu_backend_type;             /**< NPU后端类型 */
    int    npu_initialized;              /**< NPU是否已初始化 */
};

/**
 * @brief GPU内存内部结构体
 * 
 * 注意：这个定义必须与gpu.c中的struct GpuMemory完全一致。
 */
struct GpuMemory {
    GpuContext* context;         /**< 所属上下文 */
    void* data;                  /**< 数据指针 */
    size_t size;                 /**< 数据大小（字节） */
    GpuMemoryType type;          /**< 内存类型 */
    int is_device_memory;        /**< 是否为设备内存 */
    
    /** 后端私有数据指针（由具体后端实现使用） */
    void* backend_data;

/* NPU算子执行上下文
     * op_handle: ACL算子句柄 / 其他NPU算子对象
     * op_attrs : 算子属性列表
     * op_inputs: 算子输入描述符列表
     * op_outputs: 算子输出描述符列表
     * op_stream: 算子执行流（用于异步执行） */
    void* op_handle;
    void** op_attrs;
    void** op_inputs;
    void** op_outputs;
    void* op_stream;
};

/**
 * @brief GPU内核内部结构体
 * 
 * 注意：这个定义必须与gpu.c中的struct GpuKernel完全一致。
 */
struct GpuKernel {
    GpuContext* context;         /**< 所属上下文 */
    char* kernel_source;         /**< 内核源代码 */
    char* kernel_name;           /**< 内核函数名 */
    void (*cpu_function)(void*); /**< CPU实现函数（CPU后端使用） */
    void* user_data;             /**< 用户数据 */
    void** arg_values;           /**< 参数值数组 */
    size_t* arg_sizes;           /**< 参数大小数组 */
    int arg_count;               /**< 参数数量 */
    int arg_capacity;            /**< 参数容量 */
    int work_dim;                /**< 工作维度（用于多维内核执行） */
    int is_compiled;             /**< 内核是否已编译（P0-028修复：NPU后端使用） */
    size_t global_work_size[3];  /**< 全局工作大小（P0-028修复：NPU后端使用） */
    size_t local_work_size[3];   /**< 本地工作组大小（P0-028修复：NPU后端使用） */
    
    /** 后端私有数据指针（由具体后端实现使用） */
    void* backend_data;

/* NPU算子执行上下文（核函数使用）
     * op_handle: ACL算子句柄 / 其他NPU算子对象
     * op_attrs : 算子属性列表
     * op_inputs: 算子输入描述符列表
     * op_outputs: 算子输出描述符列表
     * op_stream: 算子执行流（用于异步执行） */
    void* op_handle;
    void** op_attrs;
    void** op_inputs;
    void** op_outputs;
    void* op_stream;
};

/**
 * @brief GPU流内部结构体
 * 
 * 注意：这个定义必须与gpu.c中的struct GpuStream完全一致。
 */
struct GpuStream {
    GpuContext* context;         /**< 所属上下文 */
    int is_completed;            /**< 是否完成 */
    size_t enqueued_operations;  /**< 已入队操作计数（NPU/TPU后端使用） */
    
    /** 后端私有数据指针（由具体后端实现使用） */
    void* backend_data;
};

/* ============================================================================
 * 内核缓存类型定义
 * =========================================================================== */

/**
 * @brief 内核缓存默认容量
 */
#define GPU_KERNEL_CACHE_DEFAULT_CAPACITY 64

/**
 * @brief 内核缓存条目
 */
struct GpuKernelCacheEntry {
    uint64_t source_hash;          /**< 源码哈希值（用于快速查找） */
    struct GpuKernel* kernel;      /**< 缓存的内核指针 */
    long last_access_time;         /**< 最后访问时间（LRU淘汰用） */
    int use_count;                 /**< 使用次数 */
    int refcount;                  /**< 引用计数 */
    int is_valid;                  /**< 条目是否有效 */
};

/* ============================================================================
 * 辅助宏和类型定义
 * ============================================================================ */

/**
 * @brief 将不透明句柄转换为具体结构体指针的宏
 * 
 * 使用示例：
 * GpuContext* ctx = ...;
 * struct GpuContext* internal_ctx = GPU_TO_INTERNAL(ctx);
 */
#define GPU_TO_INTERNAL(handle) ((struct GpuContext*)(handle))
#define GPU_MEMORY_TO_INTERNAL(memory) ((struct GpuMemory*)(memory))
#define GPU_KERNEL_TO_INTERNAL(kernel) ((struct GpuKernel*)(kernel))
#define GPU_STREAM_TO_INTERNAL(stream) ((struct GpuStream*)(stream))

/**
 * @brief 检查指针是否有效的宏
 */
#define GPU_CHECK_PTR(ptr) ((ptr) != NULL)

/**
 * @brief GPU后端初始化状态
 */
typedef enum {
    GPU_BACKEND_UNINITIALIZED = 0,
    GPU_BACKEND_INITIALIZING = 1,
    GPU_BACKEND_INITIALIZED = 2,
    GPU_BACKEND_ERROR = 3
} GpuBackendState;

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_GPU_INTERNAL_H