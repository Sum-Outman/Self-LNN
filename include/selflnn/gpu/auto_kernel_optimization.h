/**
 * @file auto_kernel_optimization.h
 * @brief 自动内核优化系统接口
 *
 * 提供GPU内核自动调优和性能优化功能，支持内核性能分析、
 * 自动参数选择、工作组大小优化、向量化宽度自适应等功能。
 */

#ifndef SELFLNN_AUTO_KERNEL_OPTIMIZATION_H
#define SELFLNN_AUTO_KERNEL_OPTIMIZATION_H

#include <stddef.h>

/* 前向声明 */
typedef struct GpuContext GpuContext;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 内核类型枚举
 */
typedef enum {
    KERNEL_TYPE_MATMUL = 0,       /**< 矩阵乘法 */
    KERNEL_TYPE_CONV2D = 1,       /**< 二维卷积 */
    KERNEL_TYPE_POOLING = 2,      /**< 池化操作 */
    KERNEL_TYPE_NORMALIZATION = 3,/**< 归一化 */
    KERNEL_TYPE_ACTIVATION = 4,   /**< 激活函数 */
    KERNEL_TYPE_SOFTMAX = 5,      /**< Softmax */
    KERNEL_TYPE_LIQUID_GATE = 6,  /**< 液态门控计算 */
    KERNEL_TYPE_REDUCTION = 7,    /**< 归约操作 */
    KERNEL_TYPE_ELEMENTWISE = 8,  /**< 逐元素操作 */
    KERNEL_TYPE_CUSTOM = 9        /**< 自定义内核 */
} KernelType;

/**
 * @brief 优化策略枚举
 */
typedef enum {
    OPT_STRATEGY_OCCUPANCY = 0,  /**< 以占用率为优先 */
    OPT_STRATEGY_LATENCY = 1,    /**< 以延迟为优先 */
    OPT_STRATEGY_BALANCED = 2,   /**< 平衡策略 */
    OPT_STRATEGY_MEMORY = 3      /**< 以内存访问为优先 */
} OptimizationStrategy;

/**
 * @brief 内核优化参数
 */
typedef struct {
    size_t local_work_size[3];   /**< 局部工作组大小 */
    int vector_width;            /**< 向量化宽度 (1/2/4/8) */
    int unroll_factor;           /**< 循环展开因子 */
    int use_shared_memory;       /**< 是否使用共享内存 */
    size_t shared_memory_size;   /**< 共享内存大小（字节） */
    int use_tensor_cores;        /**< 是否使用张量核心（如果可用） */
    int num_workgroups;          /**< 工作组数量（0表示自动） */
    OptimizationStrategy strategy; /**< 优化策略 */
    int prefer_occupancy;        /**< 是否优先占用率 */
    int enable_doubles;          /**< 是否启用双精度 */
} KernelOptimizationParams;

/**
 * @brief 内核性能记录
 */
typedef struct {
    KernelType kernel_type;      /**< 内核类型 */
    char kernel_name[128];       /**< 内核名称 */
    size_t input_size;           /**< 输入数据大小（字节） */
    size_t output_size;          /**< 输出数据大小（字节） */
    size_t global_work_size[3];  /**< 全局工作大小 */
    KernelOptimizationParams params; /**< 使用的优化参数 */
    double execution_time_ms;    /**< 执行时间（毫秒） */
    int device_id;               /**< 设备ID */
    int use_count;               /**< 使用次数 */
    double average_time_ms;      /**< 平均执行时间 */
    double best_time_ms;         /**< 最佳执行时间 */
    double flops;                /**< 计算吞吐量（GFLOPS） */
    double bandwidth;            /**< 内存带宽（GB/s） */
    int is_optimal;              /**< 是否已确认最优 */
} KernelPerformanceRecord;

/**
 * @brief 自动内核优化器句柄
 */
typedef struct AutoKernelOptimizer AutoKernelOptimizer;

/**
 * @brief 创建自动内核优化器
 *
 * @param device_id 设备ID
 * @param device_name 设备名称（可为NULL）
 * @return AutoKernelOptimizer* 优化器句柄，失败返回NULL
 */
AutoKernelOptimizer* auto_kernel_optimizer_create(int device_id, const char* device_name);

/**
 * @brief 销毁自动内核优化器
 *
 * @param optimizer 优化器句柄
 */
void auto_kernel_optimizer_destroy(AutoKernelOptimizer* optimizer);

/**
 * @brief 记录内核执行性能
 *
 * @param optimizer 优化器句柄
 * @param kernel_type 内核类型
 * @param kernel_name 内核名称
 * @param input_size 输入数据大小
 * @param output_size 输出数据大小
 * @param global_work_size 全局工作大小
 * @param params 优化参数
 * @param execution_time_ms 执行时间
 * @return int 成功返回0，失败返回-1
 */
int auto_kernel_optimizer_profile(AutoKernelOptimizer* optimizer,
                                 KernelType kernel_type,
                                 const char* kernel_name,
                                 size_t input_size,
                                 size_t output_size,
                                 const size_t* global_work_size,
                                 const KernelOptimizationParams* params,
                                 double execution_time_ms);

/**
 * @brief 获取最优优化参数
 *
 * @param optimizer 优化器句柄
 * @param kernel_type 内核类型
 * @param kernel_name 内核名称
 * @param input_size 输入数据大小
 * @param output_size 输出数据大小
 * @param params 优化参数输出
 * @return int 成功返回0，失败返回-1
 */
int auto_kernel_optimizer_get_optimal_params(AutoKernelOptimizer* optimizer,
                                            KernelType kernel_type,
                                            const char* kernel_name,
                                            size_t input_size,
                                            size_t output_size,
                                            KernelOptimizationParams* params);

/**
 * @brief 执行自动调优
 *
 * @param optimizer 优化器句柄
 * @param kernel_type 内核类型
 * @param kernel_name 内核名称
 * @param input_size 输入数据大小
 * @param output_size 输出数据大小
 * @return int 成功返回优化后的执行时间（毫秒），失败返回-1
 */
double auto_kernel_optimizer_tune(AutoKernelOptimizer* optimizer,
                                 KernelType kernel_type,
                                 const char* kernel_name,
                                 size_t input_size,
                                 size_t output_size);

/**
 * @brief 获取性能统计信息
 *
 * @param optimizer 优化器句柄
 * @param total_profiles 总分析次数输出
 * @param total_optimizations 总优化次数输出
 * @param average_speedup 平均加速比输出
 * @return int 成功返回0，失败返回-1
 */
int auto_kernel_optimizer_get_statistics(AutoKernelOptimizer* optimizer,
                                        int* total_profiles,
                                        int* total_optimizations,
                                        double* average_speedup);

/**
 * @brief 建议工作组大小
 *
 * @param optimizer 优化器句柄
 * @param global_size 全局工作大小
 * @param max_work_group_size 最大工作组大小
 * @param kernel_type 内核类型
 * @return size_t 建议的工作组大小
 */
size_t auto_kernel_optimizer_suggest_work_group(AutoKernelOptimizer* optimizer,
                                               size_t global_size,
                                               size_t max_work_group_size,
                                               KernelType kernel_type);

/**
 * @brief 建议向量化宽度
 *
 * @param optimizer 优化器句柄
 * @param data_size 数据元素大小（字节）
 * @param supports_half 是否支持半精度
 * @param memory_bandwidth 内存带宽（GB/s）
 * @return int 建议的向量化宽度 (1/2/4/8)
 */
int auto_kernel_optimizer_suggest_vector_width(AutoKernelOptimizer* optimizer,
                                              size_t data_size,
                                              int supports_half,
                                              float memory_bandwidth);

/**
 * @brief 建议循环展开因子
 *
 * @param optimizer 优化器句柄
 * @param loop_iterations 循环迭代次数
 * @param operations_per_iter 每次迭代操作数
 * @return int 建议的展开因子
 */
int auto_kernel_optimizer_suggest_unroll_factor(AutoKernelOptimizer* optimizer,
                                               int loop_iterations,
                                               int operations_per_iter);

/**
 * @brief 清除性能缓存
 *
 * @param optimizer 优化器句柄
 */
void auto_kernel_optimizer_clear_cache(AutoKernelOptimizer* optimizer);

/**
 * @brief 保存优化数据库到文件
 *
 * @param optimizer 优化器句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int auto_kernel_optimizer_save_database(AutoKernelOptimizer* optimizer,
                                       const char* filepath);

/**
 * @brief 从文件加载优化数据库
 *
 * @param optimizer 优化器句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int auto_kernel_optimizer_load_database(AutoKernelOptimizer* optimizer,
                                       const char* filepath);

/**
 * @brief 获取设备最佳内核配置
 *
 * @param optimizer 优化器句柄
 * @param kernel_type 内核类型
 * @param kernel_name 内核名称
 * @param input_size 输入数据大小
 * @param record 性能记录输出
 * @return int 找到返回0，未找到返回-1
 */
int auto_kernel_optimizer_get_best_record(AutoKernelOptimizer* optimizer,
                                         KernelType kernel_type,
                                         const char* kernel_name,
                                         size_t input_size,
                                         KernelPerformanceRecord* record);

/**
 * @brief 获取所有性能记录数量
 *
 * @param optimizer 优化器句柄
 * @return int 记录数量，失败返回-1
 */
int auto_kernel_optimizer_get_record_count(AutoKernelOptimizer* optimizer);

/**
 * @brief 获取指定索引的性能记录
 *
 * @param optimizer 优化器句柄
 * @param index 记录索引
 * @param record 性能记录输出
 * @return int 成功返回0，失败返回-1
 */
int auto_kernel_optimizer_get_record_at(AutoKernelOptimizer* optimizer,
                                       int index,
                                       KernelPerformanceRecord* record);

/**
 * @brief 评估内核性能（基于历史数据预测）
 *
 * @param optimizer 优化器句柄
 * @param kernel_type 内核类型
 * @param kernel_name 内核名称
 * @param input_size 输入数据大小
 * @param output_size 输出数据大小
 * @param predicted_time_ms 预测执行时间输出（毫秒）
 * @return int 成功返回0，失败返回-1
 */
int auto_kernel_optimizer_predict_performance(AutoKernelOptimizer* optimizer,
                                             KernelType kernel_type,
                                             const char* kernel_name,
                                             size_t input_size,
                                             size_t output_size,
                                             double* predicted_time_ms);

/**
 * @brief 真实GPU内核执行计时测量（在线调优模式）
 *
 * 实际创建/执行内核并测量高精度墙钟时间。
 * 跨平台实现：Windows使用QueryPerformanceCounter，Linux使用clock_gettime。
 *
 * @param context GPU上下文
 * @param kernel_source 内核源码
 * @param kernel_name 内核名称
 * @param global_size 全局工作大小
 * @param local_size 局部工作大小
 * @param warmup_iters 预热迭代次数
 * @param bench_iters 基准测试迭代次数
 * @param out_time_ms 输出单次执行时间（毫秒）
 * @return int 成功返回0，失败返回-1
 */
int auto_kernel_optimizer_measure_real(GpuContext* context,
                                        const char* kernel_source,
                                        const char* kernel_name,
                                        size_t global_size, size_t local_size,
                                        int warmup_iters, int bench_iters,
                                        double* out_time_ms);

/* ZSFX-P1修复: 设置GPU在线上下文（先前遗漏声明） */
int auto_kernel_optimizer_set_context(AutoKernelOptimizer* optimizer, GpuContext* context);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_AUTO_KERNEL_OPTIMIZATION_H */
