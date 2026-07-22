/**
 * @file gpu_cpu_backend_bridge.h
 * @brief GPU CPU后端桥接接口
 *
 * L-002修复：消除gpu.c与gpu_cpu.c之间的CPU后端功能重叠。
 * 通过此桥接接口，gpu.c中的CPU后端调度函数可以委托到
 * gpu_cpu.c中优化的SIMD实现，避免代码重复。
 *
 * 架构分层：
 *   gpu.c (GPU调度层) ──→ gpu_cpu_backend_bridge ──→ gpu_cpu.c (优化CPU实现)
 *
 * 实现日期: 2026-07-22
 */

#ifndef SELFLNN_GPU_CPU_BACKEND_BRIDGE_H
#define SELFLNN_GPU_CPU_BACKEND_BRIDGE_H

#include "selflnn/gpu/gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CPU后端优化函数指针表
 * 用于gpu.c中的CPU后端调度，可选委托到gpu_cpu.c的SIMD优化实现
 * ============================================================================ */

/** @brief CPU后端优化操作表 */
typedef struct {
    /* 上下文管理 */
    void* (*context_create)(GpuBackend backend, int device_index);
    void  (*context_destroy)(void* ctx);

    /* 内存管理 */
    void* (*memory_alloc)(void* ctx, size_t size, GpuMemoryType type);
    void  (*memory_free)(void* mem);

    /* 内核执行 */
    void* (*kernel_create)(void* ctx, const char* source, const char* name);
    void  (*kernel_destroy)(void* kernel);
    int   (*kernel_set_arg)(void* kernel, int index, size_t size, const void* value);
    int   (*kernel_launch)(void* kernel, size_t grid[3], size_t block[3], void* stream);

    /* 流管理 */
    void* (*stream_create)(void* ctx);
    void  (*stream_destroy)(void* stream);
    int   (*stream_sync)(void* stream);

    /* 矩阵运算（SIMD优化） */
    int   (*matmul)(const float* a, const float* b, float* c,
                    int m, int n, int k, int transpose_a, int transpose_b);
    int   (*matmul_batched)(const float* a, const float* b, float* c,
                            int m, int n, int k, int batch_size,
                            int transpose_a, int transpose_b);

    /* 激活函数（SIMD优化） */
    int   (*relu)(float* data, size_t count);
    int   (*sigmoid)(float* data, size_t count);
    int   (*tanh_act)(float* data, size_t count);
    int   (*gelu)(float* data, size_t count);

    /* 优化器步骤（SIMD优化） */
    int   (*sgd_step)(float* weights, const float* grads, size_t count,
                      float lr, float momentum, float weight_decay,
                      float* velocity);
    int   (*adam_step)(float* weights, const float* grads, size_t count,
                       float lr, float beta1, float beta2, float eps,
                       float* m, float* v, int step);

    /* 是否可用（gpu_cpu.c编译链接时可用） */
    int   is_available;

    /* 运行时检测的SIMD能力 */
    int   has_avx2;
    int   has_avx;
    int   has_sse;
    int   has_neon;
    int   has_fma;
} GpuCpuBackendOps;

/**
 * @brief 获取CPU后端优化操作表
 *
 * 返回指向全局CPU后端优化操作表的指针。
 * 如果gpu_cpu.c已编译链接，则is_available=1且所有函数指针有效。
 * 否则is_available=0，调用者应回退到gpu.c中的基本实现。
 *
 * @return const GpuCpuBackendOps* 操作表指针（永不返回NULL）
 */
const GpuCpuBackendOps* gpu_cpu_backend_get_ops(void);

/**
 * @brief 初始化CPU后端桥接
 *
 * 在gpu.c的CPU后端初始化时调用，建立与gpu_cpu.c的连接。
 * 如果gpu_cpu.c已链接，则填充优化操作表；否则标记为不可用。
 *
 * @return int 成功返回0，失败返回-1
 */
int gpu_cpu_backend_bridge_init(void);

/**
 * @brief 检查CPU后端优化是否可用
 *
 * @return int 可用返回1，不可用返回0
 */
int gpu_cpu_backend_is_optimized_available(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GPU_CPU_BACKEND_BRIDGE_H */