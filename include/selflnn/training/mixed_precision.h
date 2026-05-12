/**
 * @file mixed_precision.h
 * @brief 混合精度训练支持（FP16/FP32）
 * 
 * 混合精度训练系统，支持FP16和FP32精度的自动混合。
 * 提供精度转换、梯度缩放、损失缩放和自动精度管理功能。
 */

#ifndef SELFLNN_TRAINING_MIXED_PRECISION_H
#define SELFLNN_TRAINING_MIXED_PRECISION_H

#include "selflnn/core/lnn.h"
/* 兼容性定义：NeuralNetwork 是 LNN 的别名 */
#ifndef SELFLNN_NEURAL_NETWORK_DEFINED
#define SELFLNN_NEURAL_NETWORK_DEFINED
typedef struct LNN NeuralNetwork;
#endif
#include "selflnn/training/training.h"
#include "selflnn/gpu/gpu.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 混合精度训练模式
 */
typedef enum {
    MIXED_PRECISION_DISABLED = 0,   /**< 禁用混合精度 */
    MIXED_PRECISION_FP16,           /**< FP16为主，FP32为辅 */
    MIXED_PRECISION_BFLOAT16,       /**< BFLOAT16为主，FP32为辅 */
    MIXED_PRECISION_AUTO,           /**< 自动选择最佳精度 */
    MIXED_PRECISION_DYNAMIC         /**< 动态调整精度 */
} MixedPrecisionMode;

/**
 * @brief 梯度缩放策略
 */
typedef enum {
    GRADIENT_SCALING_NONE = 0,      /**< 无梯度缩放 */
    GRADIENT_SCALING_STATIC,        /**< 静态缩放 */
    GRADIENT_SCALING_DYNAMIC,       /**< 动态缩放 */
    GRADIENT_SCALING_ADAPTIVE       /**< 自适应缩放 */
} GradientScalingStrategy;

/**
 * @brief 精度转换模式
 */
typedef enum {
    PRECISION_CONVERSION_AUTO = 0,  /**< 自动转换 */
    PRECISION_CONVERSION_MANUAL,    /**< 手动转换 */
    PRECISION_CONVERSION_HYBRID     /**< 混合转换 */
} PrecisionConversionMode;

/**
 * @brief 混合精度配置
 */
typedef struct {
    MixedPrecisionMode mode;               /**< 混合精度模式 */
    GradientScalingStrategy scaling;       /**< 梯度缩放策略 */
    PrecisionConversionMode conversion;    /**< 精度转换模式 */
    
    // 精度设置
    int use_fp16_for_forward;              /**< 前向传播使用FP16 */
    int use_fp16_for_backward;             /**< 反向传播使用FP16 */
    int use_fp16_for_weights;              /**< 权重使用FP16 */
    int use_fp32_master_weights;           /**< 使用FP32主权重 */
    
    // 梯度缩放参数
    float initial_scale;                   /**< 初始缩放因子 */
    float max_scale;                       /**< 最大缩放因子 */
    float min_scale;                       /**< 最小缩放因子 */
    float growth_factor;                   /**< 增长因子 */
    float backoff_factor;                  /**< 回退因子 */
    int scale_update_interval;             /**< 缩放更新间隔 */
    
    // 性能优化
    int enable_fp16_arithmetic;            /**< 启用FP16算术运算 */
    int enable_fp16_storage;               /**< 启用FP16存储 */
    int enable_automatic_loss_scaling;     /**< 启用自动损失缩放 */
    int enable_gradient_clipping;          /**< 启用梯度裁剪 */
    
    // 精度控制
    float fp16_min_value;                  /**< FP16最小值保护 */
    float fp16_max_value;                  /**< FP16最大值保护 */
    int check_nan_inf;                     /**< 检查NaN/Inf */
    int handle_underflow_overflow;         /**< 处理下溢/上溢 */
    
    // 调试和监控
    int log_precision_stats;               /**< 记录精度统计 */
    int log_gradient_stats;                /**< 记录梯度统计 */
    int log_scaling_stats;                 /**< 记录缩放统计 */
} MixedPrecisionConfig;

/**
 * @brief 精度统计信息
 */
typedef struct {
    // 精度分布
    size_t fp16_tensor_count;              /**< FP16张量数量 */
    size_t fp32_tensor_count;              /**< FP32张量数量 */
    size_t total_tensor_count;             /**< 总张量数量 */
    
    // 内存使用
    size_t fp16_memory_bytes;              /**< FP16内存使用（字节） */
    size_t fp32_memory_bytes;              /**< FP32内存使用（字节） */
    size_t total_memory_bytes;             /**< 总内存使用（字节） */
    float memory_saving_ratio;             /**< 内存节省比例 */
    
    // 精度转换统计
    size_t fp32_to_fp16_conversions;       /**< FP32到FP16转换次数 */
    size_t fp16_to_fp32_conversions;       /**< FP16到FP32转换次数 */
    float conversion_overhead_ms;          /**< 转换开销（毫秒） */
    
    // 数值稳定性
    size_t underflow_count;                /**< 下溢次数 */
    size_t overflow_count;                 /**< 上溢次数 */
    size_t nan_count;                      /**< NaN次数 */
    size_t inf_count;                      /**< Inf次数 */
    
    // 梯度缩放统计
    float current_scale;                   /**< 当前缩放因子 */
    size_t scale_increases;                /**< 缩放增加次数 */
    size_t scale_decreases;                /**< 缩放减少次数 */
    float average_gradient_magnitude;      /**< 平均梯度幅度 */
} PrecisionStatistics;

/**
 * @brief 混合精度层信息
 */
typedef struct {
    int layer_id;                   /**< 层ID */
    int use_fp16_forward;           /**< 前向传播使用FP16 */
    int use_fp16_backward;          /**< 反向传播使用FP16 */
    int use_fp16_weights;           /**< 权重使用FP16 */
    void* fp16_weights;             /**< FP16权重 */
    void* fp16_gradients;           /**< FP16梯度 */
    void* fp32_master_weights;      /**< FP32主权重 */
    void* fp32_master_gradients;    /**< FP32主梯度 */
    size_t weight_size;             /**< 权重大小（字节） */
    size_t gradient_size;           /**< 梯度大小（字节） */
} MixedPrecisionLayer;

/**
 * @brief 硬件FP16支持检测结果
 */
typedef struct {
    int has_fp16_native;               /**< 原生FP16支持（GPU硬件原生FP16计算） */
    int has_tensor_cores;              /**< Tensor Cores支持（NVIDIA cc>=7.0） */
    int has_bfloat16;                  /**< BFLOAT16支持（NVIDIA cc>=8.0/AMD等） */
    int has_fp16_arithmetic;           /**< FP16算术运算支持（非仅存储） */
    int has_fp16_storage;              /**< FP16存储支持 */
    int has_int8_tensor_cores;         /**< INT8 Tensor Cores支持（NVIDIA cc>=7.0） */
    int has_int4_tensor_cores;         /**< INT4 Tensor Cores支持（NVIDIA cc>=8.0） */
    int max_fp16_flops_per_second;     /**< FP16峰值TFLOPS（0表示未知） */
    float memory_bandwidth_gbps;       /**< 内存带宽GB/s（0表示未知） */
    GpuBackend gpu_backend;            /**< 当前GPU后端类型 */
    char device_name[128];             /**< 设备名称 */
    char backend_version[64];          /**< 后端版本信息 */
    int compute_capability_major;      /**< CUDA计算能力主版本（非CUDA后端为0） */
    int compute_capability_minor;      /**< CUDA计算能力次版本（非CUDA后端为0） */
    int device_count;                  /**< 可用设备数量 */
    int selected_device;               /**< 当前选中的设备索引 */
    int supports_async_fp16_conversion;/**< 支持异步FP16转换 */
    MixedPrecisionMode suggested_mode;         /**< 根据硬件推荐的最优精度模式 */
    GradientScalingStrategy suggested_scaling; /**< 根据硬件推荐的梯度缩放策略 */
} HardwareFP16Support;

/**
 * @brief 混合精度训练上下文
 */
struct MixedPrecisionContext {
    NeuralNetwork* network;         /**< 原始神经网络 */
    NeuralNetwork* fp16_network;    /**< FP16神经网络（可选） */
    MixedPrecisionConfig config;    /**< 配置 */
    PrecisionStatistics stats;      /**< 统计信息 */
    HardwareFP16Support hardware;   /**< 硬件FP16支持检测结果 */
    
    // 层管理
    MixedPrecisionLayer* layers;    /**< 层信息数组 */
    int layer_count;                /**< 层数量 */
    
    // 梯度缩放状态
    float current_scale;            /**< 当前缩放因子 */
    float target_scale;             /**< 目标缩放因子 */
    int scale_update_counter;       /**< 缩放更新计数器 */
    int scale_increase_pending;     /**< 缩放增加待处理 */
    int scale_decrease_pending;     /**< 缩放减少待处理 */
    
    // 缓冲区
    void* fp16_buffer;              /**< FP16缓冲区 */
    void* fp32_buffer;              /**< FP32缓冲区 */
    size_t buffer_size;             /**< 缓冲区大小 */
    
    // GPU加速转换函数指针（根据硬件动态选择最优转换路径）
    int (*gpu_fp32_to_fp16)(const float* src, void* dst, size_t count);
    int (*gpu_fp16_to_fp32)(const void* src, float* dst, size_t count);
    
    // 状态标记
    int enabled;                    /**< 是否启用 */
    int initialized;                /**< 是否已初始化 */
    int training;                   /**< 是否在训练中 */
};

typedef struct MixedPrecisionContext MixedPrecisionContext;

/**
 * @brief 创建混合精度训练上下文
 * 
 * @param network 神经网络
 * @param config 混合精度配置
 * @return MixedPrecisionContext* 上下文指针，失败返回NULL
 */
MixedPrecisionContext* mixed_precision_create(NeuralNetwork* network, 
                                              const MixedPrecisionConfig* config);

/**
 * @brief 销毁混合精度训练上下文
 * 
 * @param context 混合精度上下文
 */
void mixed_precision_destroy(MixedPrecisionContext* context);

/**
 * @brief 应用混合精度到前向传播
 * 
 * @param context 混合精度上下文
 * @param input 输入数据
 * @param output 输出数据
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_forward(MixedPrecisionContext* context, 
                           const void* input, void* output);

/**
 * @brief 应用混合精度到反向传播
 * 
 * @param context 混合精度上下文
 * @param grad_output 输出梯度
 * @param grad_input 输入梯度
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_backward(MixedPrecisionContext* context,
                            const void* grad_output, void* grad_input);

/**
 * @brief 更新混合精度权重
 * 
 * @param context 混合精度上下文
 * @param learning_rate 学习率
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_update_weights(MixedPrecisionContext* context, 
                                  float learning_rate);

/**
 * @brief 转换张量精度
 * 
 * @param src 源数据
 * @param dst 目标数据
 * @param count 元素数量
 * @param src_dtype 源数据类型
 * @param dst_dtype 目标数据类型
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_convert_tensor(const void* src, void* dst, 
                                  size_t count, int src_dtype, int dst_dtype);

/**
 * @brief 应用梯度缩放
 * 
 * @param gradients 梯度数据
 * @param count 梯度数量
 * @param scale 缩放因子
 * @param dtype 数据类型
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_scale_gradients(void* gradients, size_t count, 
                                   float scale, int dtype);

/**
 * @brief 检查数值稳定性
 * 
 * @param data 数据指针
 * @param count 元素数量
 * @param dtype 数据类型
 * @param stats 统计信息输出
 * @return int 发现不稳定返回1，稳定返回0，错误返回-1
 */
int mixed_precision_check_stability(const void* data, size_t count, 
                                   int dtype, PrecisionStatistics* stats);

/**
 * @brief 更新梯度缩放因子
 * 
 * @param context 混合精度上下文
 * @param gradients 梯度数据
 * @param count 梯度数量
 * @return float 新的缩放因子
 */
float mixed_precision_update_scaling(MixedPrecisionContext* context,
                                    const void* gradients, size_t count);

/**
 * @brief 获取精度统计信息
 * 
 * @param context 混合精度上下文
 * @param stats 统计信息输出
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_get_statistics(MixedPrecisionContext* context,
                                  PrecisionStatistics* stats);

/**
 * @brief 重置精度统计信息
 * 
 * @param context 混合精度上下文
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_reset_statistics(MixedPrecisionContext* context);

/**
 * @brief 检测硬件FP16支持能力
 * 
 * 自动检测当前系统的GPU/NPU硬件对FP16计算的支持情况，
 * 包括原生FP16支持、Tensor Cores、BFLOAT16等。
 * 此函数会查询当前激活的GPU后端并获取详细的设备能力信息。
 * 
 * @return HardwareFP16Support 检测结果结构体（包含推荐的精度配置）
 */
HardwareFP16Support mixed_precision_detect_hardware_support(void);

/**
 * @brief 根据硬件能力自动配置混合精度参数
 * 
 * 使用mixed_precision_detect_hardware_support()的检测结果，
 * 自动设置最优的混合精度配置。当has_tensor_cores时启用FP16+动态缩放；
 * 仅有FP16存储时使用FP32主权重；硬件不支持时禁用混合精度。
 * 
 * @param config 配置输出（将被自动填充）
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_configure_from_hardware(MixedPrecisionConfig* config);

/**
 * @brief 获取默认混合精度配置
 * 
 * @param config 配置输出
 */
void mixed_precision_default_config(MixedPrecisionConfig* config);

/**
 * @brief 启用混合精度训练
 * 
 * @param trainer 训练器
 * @param config 混合精度配置
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_enable(Trainer* trainer, const MixedPrecisionConfig* config);

/**
 * @brief 禁用混合精度训练
 * 
 * @param trainer 训练器
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_disable(Trainer* trainer);

/**
 * @brief 创建FP16版本的神经网络
 * 
 * @param network FP32神经网络
 * @return NeuralNetwork* FP16神经网络，失败返回NULL
 */
NeuralNetwork* mixed_precision_create_fp16_network(const NeuralNetwork* network);

/**
 * @brief 同步FP32主权重
 * 
 * @param context 混合精度上下文
 * @return int 成功返回0，失败返回-1
 */
int mixed_precision_sync_master_weights(MixedPrecisionContext* context);

/* ============================================================================
 * BF16 (Brain Float 16) 支持
 * ============================================================================ */

typedef uint16_t bf16_t;

float bf16_to_float(bf16_t v);
bf16_t float_to_bf16(float v);
int mixed_precision_convert_fp32_to_bf16(const float* src, bf16_t* dst, size_t count);
int mixed_precision_convert_bf16_to_fp32(const bf16_t* src, float* dst, size_t count);
int mixed_precision_bf16_matmul(const bf16_t* A, const bf16_t* B, float* C,
                                 int M, int N, int K);
int mixed_precision_bf16_gemm(const bf16_t* A, const bf16_t* B, float* C,
                               int M, int N, int K, float beta);
int mixed_precision_bf16_forward(NeuralNetwork* network, const bf16_t* input,
                                  bf16_t* output, int batch_size);
int mixed_precision_bf16_hardware_support(void);

/* ============================================================================
 * INT8 量化训练 (Quantization-Aware Training)
 * ============================================================================ */

typedef struct {
    float scale;
    int zero_point;
    float min_val;
    float max_val;
    int is_symmetric;
    int bit_width;
} MPQuantConfig;

typedef struct {
    float* weights;
    float* activations;
    float* gradients;
    float weight_scale;
    float activation_scale;
    int weight_zero_point;
    int activation_zero_point;
    int weight_quantized;
    int activation_quantized;
    int num_bits;
    int is_symmetric;
} MPQuantLayer;

#define MP_DEFAULT_QUANT_CONFIG { 1.0f, 0, -128.0f, 127.0f, 1, 8 }

int mp_quantize_tensor(const float* src, int8_t* dst, size_t count,
                        const MPQuantConfig* config);
int mp_dequantize_tensor(const int8_t* src, float* dst, size_t count,
                          const MPQuantConfig* config);
int mp_quantize_weights(const float* weights, int8_t* qweights, size_t count,
                         float* scale, int* zero_point, int num_bits);
int mp_quantize_activations(const float* activations, int8_t* qacts,
                             size_t count, float* scale, int* zero_point);
int mp_simulated_quantize(float* data, size_t count, int num_bits,
                          int is_symmetric);
int mp_simulated_quantize_linear(float* data, size_t count, float scale,
                                  int zero_point);
int mp_quant_aware_forward(const float* input, const float* weights,
                            float* output, int M, int N, int K,
                            int num_bits, int symmetric);
int mp_quant_aware_backward(const float* grad_output, const float* input,
                             const float* weights, float* grad_input,
                             float* grad_weight, int M, int N, int K,
                             int num_bits, int symmetric);
int mp_calibrate_quant_range(const float* samples, size_t sample_count,
                              size_t dim, MPQuantConfig* config);
int mp_compute_quantization_error(const float* original, const float* recovered,
                                   size_t count, float* out_mse, float* out_max_error);
int mixed_precision_enable_quant_aware_training(Trainer* trainer, int num_bits,
                                                  int symmetric);

/* ============================================================================
 * 精度感知训练 (Precision-Aware Training)
 * ============================================================================ */

typedef enum {
    PRECISION_MONITOR_NONE = 0,
    PRECISION_MONITOR_GRADIENT,
    PRECISION_MONITOR_WEIGHT,
    PRECISION_MONITOR_ACTIVATION,
    PRECISION_MONITOR_ALL
} PrecisionMonitorMode;

int mixed_precision_precision_aware_train_step(Trainer* trainer,
                                                const float* input,
                                                const float* target,
                                                float* output, float* loss,
                                                PrecisionMonitorMode monitor);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TRAINING_MIXED_PRECISION_H */