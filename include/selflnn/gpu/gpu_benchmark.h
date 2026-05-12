/**
 * @file gpu_benchmark.h
 * @brief GPU 基准测试和内存压力测试接口
 *
 * 提供完整的 GPU 性能基准测试功能，包括：
 * - 内存压力测试：逐级分配显存直到失败或阈值
 * - 计算性能测试：矩阵乘法和卷积 GFLOPS 测量
 * - 带宽测试：主机到设备和设备到主机的传输速度测量
 * - 端到端完整基准测试：所有测试自动运行并生成报告
 */

#ifndef SELFLNN_GPU_BENCHMARK_H
#define SELFLNN_GPU_BENCHMARK_H

#include "selflnn/gpu/gpu.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 基准测试结果类型定义
 * ============================================================================ */

/**
 * @brief 内存压力测试结果
 */
typedef struct {
    size_t total_memory;             /**< 设备总内存（字节） */
    size_t peak_allocated;           /**< 成功分配的最大内存（字节） */
    size_t max_single_allocation;    /**< 单次最大成功分配（字节） */
    int allocation_count;            /**< 成功分配次数 */
    double allocation_success_rate;  /**< 分配成功率（0.0-1.0） */
    double peak_utilization_ratio;   /**< 峰值利用率 peak_allocated/total_memory */
    double average_allocation_time_ms; /**< 平均分配耗时（毫秒） */
    double total_test_time_ms;       /**< 总测试耗时（毫秒） */
    int overflow_detected;           /**< 是否检测到显存溢出 */
    char error_message[256];         /**< 错误信息 */
} GpuMemoryPressureResult;

/**
 * @brief 计算性能测试结果
 */
typedef struct {
    double matrix_mul_gflops;        /**< 矩阵乘法 GFLOPS */
    double convolution_gflops;       /**< 卷积 GFLOPS */
    double elementwise_gflops;       /**< 逐元素操作 GFLOPS */
    double memory_bound_gflops;      /**< 内存受限操作 GFLOPS */
    double average_gflops;           /**< 平均 GFLOPS */
    double theoretical_gflops;       /**< 理论峰值 GFLOPS */
    double utilization_ratio;        /**< 利用率 average_gflops/theoretical_gflops */
    double matrix_mul_time_ms;       /**< 矩阵乘法耗时（毫秒） */
    double convolution_time_ms;      /**< 卷积耗时（毫秒） */
    double elementwise_time_ms;      /**< 逐元素操作耗时（毫秒） */
    int precision_bits;              /**< 测试精度位数 */
    char description[128];           /**< 测试描述 */
} GpuComputeBenchmarkResult;

/**
 * @brief 带宽测试结果
 */
typedef struct {
    double host_to_device_bandwidth_gbps;   /**< 主机到设备带宽（GB/s） */
    double device_to_host_bandwidth_gbps;   /**< 设备到主机带宽（GB/s） */
    double device_to_device_bandwidth_gbps; /**< 设备间带宽（GB/s），无则为0 */
    double host_to_device_time_ms;          /**< 主机到设备耗时（毫秒） */
    double device_to_host_time_ms;          /**< 设备到主机耗时（毫秒） */
    double device_to_device_time_ms;        /**< 设备间耗时（毫秒），无则为0 */
    size_t transfer_size;                   /**< 单次传输大小（字节） */
    int iteration_count;                    /**< 迭代次数 */
    char description[128];                  /**< 测试描述 */
} GpuBandwidthBenchmarkResult;

/**
 * @brief 综合基准测试结果
 */
typedef struct {
    GpuBackend backend;                          /**< 后端类型 */
    int device_index;                            /**< 设备索引 */
    GpuDeviceInfo device_info;                   /**< 设备信息 */
    char backend_name[64];                       /**< 后端名称 */

    GpuMemoryPressureResult memory_pressure;     /**< 内存压力测试结果 */
    GpuComputeBenchmarkResult compute_result;    /**< 计算性能结果 */
    GpuBandwidthBenchmarkResult bandwidth_result; /**< 带宽测试结果 */
    double kernel_launch_latency_us;             /**< 内核启动延迟（微秒） */
    double memory_copy_latency_us;               /**< 小数据拷贝延迟（微秒） */

    double overall_score;                        /**< 综合评分（0-1000） */
    double memory_score;                         /**< 内存性能评分（0-1000） */
    double compute_score;                        /**< 计算性能评分（0-1000） */
    double bandwidth_score;                      /**< 带宽性能评分（0-1000） */
    char test_duration_str[64];                  /**< 总测试耗时描述 */
    double total_test_time_s;                    /**< 总测试耗时（秒） */
    char summary[1024];                          /**< 测试摘要 */
} GpuBenchmarkResult;

/* ============================================================================
 * 基准测试配置
 * ============================================================================ */

/**
 * @brief 基准测试配置
 */
typedef struct {
    int enable_memory_pressure;      /**< 是否启用内存压力测试，默认1 */
    int enable_compute_benchmark;    /**< 是否启用计算性能测试，默认1 */
    int enable_bandwidth_benchmark;  /**< 是否启用带宽测试，默认1 */
    int enable_latency_benchmark;    /**< 是否启用延迟测试，默认1 */

    double memory_pressure_step_ratio;  /**< 内存压力步进比例（相对总显存），默认0.05 */
    double memory_pressure_max_ratio;   /**< 内存压力最大分配比例，默认0.95 */
    int memory_pressure_min_blocks;     /**< 最小分配块数，默认4 */
    int memory_pressure_iterations;     /**< 内存压力测试迭代次数，默认3 */

    int compute_matrix_sizes[4];        /**< 矩阵乘法测试规模，默认{512,1024,2048,4096} */
    int compute_iterations;             /**< 计算测试迭代次数，默认10 */
    int compute_warmup;                 /**< 计算测试预热次数，默认3 */

    size_t bandwidth_min_bytes;         /**< 带宽测试最小字节，默认1MB */
    size_t bandwidth_max_bytes;         /**< 带宽测试最大字节，默认1GB */
    int bandwidth_iterations;           /**< 带宽测试迭代次数，默认10 */

    int latency_iterations;             /**< 延迟测试迭代次数，默认100 */
    size_t latency_transfer_bytes;      /**< 延迟测试传输字节，默认64 */

    int verbose;                        /**< 是否输出详细日志，默认0 */
    double timeout_seconds;             /**< 单测试超时秒数，默认300 */
} GpuBenchmarkConfig;

/**
 * @brief 获取默认基准测试配置
 *
 * @return GpuBenchmarkConfig 默认配置
 */
GpuBenchmarkConfig gpu_benchmark_config_default(void);

/* ============================================================================
 * 基准测试主接口
 * ============================================================================ */

/**
 * @brief 对指定后端和设备运行完整基准测试
 *
 * 依次运行内存压力测试、计算性能测试、带宽测试和延迟测试，
 * 生成综合评分和测试报告。
 *
 * @param backend GPU后端类型
 * @param device_index 设备索引
 * @param config 基准测试配置（为NULL使用默认配置）
 * @param result [out] 测试结果输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int gpu_run_full_benchmark(GpuBackend backend, int device_index,
                           const GpuBenchmarkConfig* config,
                           GpuBenchmarkResult* result);

/**
 * @brief 仅运行内存压力测试
 *
 * 从小块开始逐级增加分配量，记录峰值分配和分配速度。
 * 当分配失败或达到 max_ratio 限制时停止。
 *
 * @param context GPU上下文
 * @param config 基准测试配置（为NULL使用默认配置）
 * @param result [out] 内存压力测试结果
 * @return int 成功返回0，失败返回-1
 */
int gpu_run_memory_pressure_test(GpuContext* context,
                                 const GpuBenchmarkConfig* config,
                                 GpuMemoryPressureResult* result);

/**
 * @brief 仅运行计算性能测试
 *
 * 测试多种规模的矩阵乘法、卷积和逐元素操作，
 * 计算实际 GFLOPS 并与理论峰值比较。
 *
 * @param context GPU上下文
 * @param config 基准测试配置（为NULL使用默认配置）
 * @param result [out] 计算性能测试结果
 * @return int 成功返回0，失败返回-1
 */
int gpu_run_compute_benchmark(GpuContext* context,
                              const GpuBenchmarkConfig* config,
                              GpuComputeBenchmarkResult* result);

/**
 * @brief 仅运行带宽测试
 *
 * 测试不同传输大小的主机到设备和设备到主机带宽，
 * 计算峰值带宽和平均带宽。
 *
 * @param context GPU上下文
 * @param config 基准测试配置（为NULL使用默认配置）
 * @param result [out] 带宽测试结果
 * @return int 成功返回0，失败返回-1
 */
int gpu_run_bandwidth_benchmark(GpuContext* context,
                                const GpuBenchmarkConfig* config,
                                GpuBandwidthBenchmarkResult* result);

/**
 * @brief 对所有可用 GPU 后端运行基准测试
 *
 * 自动检测所有可用后端，对每个后端运行完整基准测试。
 *
 * @param config 基准测试配置（为NULL使用默认配置）
 * @param results [out] 测试结果数组（需分配至少 GPU_BACKEND_COUNT 个元素）
 * @param max_results 结果数组最大容量
 * @return int 成功测试的后端数量，失败返回-1
 */
int gpu_benchmark_all_backends(const GpuBenchmarkConfig* config,
                               GpuBenchmarkResult* results, int max_results);

/**
 * @brief 生成基准测试报告字符串
 *
 * 将基准测试结果格式化为人类可读的字符串。
 * 输出包含设备信息、各子测试结果、评分和优化建议。
 *
 * @param result 基准测试结果
 * @param buffer [out] 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回写入的字符数，失败返回-1
 */
int gpu_benchmark_format_report(const GpuBenchmarkResult* result,
                                char* buffer, size_t buffer_size);

/**
 * @brief 将基准测试结果格式化为 JSON 字符串
 *
 * 便于程序化处理和持久化存储。
 *
 * @param result 基准测试结果
 * @param buffer [out] 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回写入的字符数，失败返回-1
 */
int gpu_benchmark_format_json(const GpuBenchmarkResult* result,
                              char* buffer, size_t buffer_size);

/**
 * @brief 从设备信息估算理论 GFLOPS
 *
 * 根据设备类型、计算单元数、时钟频率和 SIMD/向量宽度
 * 估算理论峰值浮点性能。
 *
 * @param info 设备信息
 * @param precision_bits 精度位数（32=单精度，64=双精度，16=半精度）
 * @return double 估算的理论 GFLOPS
 */
double gpu_estimate_theoretical_gflops(const GpuDeviceInfo* info, int precision_bits);

/**
 * @brief 从设备信息估算理论带宽（GB/s）
 *
 * 根据设备内存类型和规格估算理论峰值带宽。
 *
 * @param info 设备信息
 * @return double 估算的理论带宽（GB/s）
 */
double gpu_estimate_theoretical_bandwidth(const GpuDeviceInfo* info);

/**
 * @brief 获取基准测试引擎版本信息
 *
 * @return const char* 版本字符串
 */
const char* gpu_benchmark_version(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GPU_BENCHMARK_H */
