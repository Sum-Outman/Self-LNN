#ifndef SELFLNN_EXTENDED_TRAINING_H
#define SELFLNN_EXTENDED_TRAINING_H

#include <stddef.h>
#include <stdint.h>
#include "selflnn/core/common.h"
#include "selflnn/core/lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SELFLNN_MAX_CHECKPOINTS 256
#define SELFLNN_MAX_MODEL_PARALLEL_DEVICES 64
#define SELFLNN_MAX_GRADIENT_COMPRESSION_GROUP_SIZE 1024
#define SELFLNN_GRADIENT_COMPRESSION_ALIGNMENT 64
#define SELFLNN_MODEL_PARALLEL_MIN_LAYERS_PER_DEVICE 2

/**
 * @brief 梯度检查点配置
 */
typedef struct {
    int enable_checkpointing;                /**< 是否启用检查点 */
    size_t checkpoint_interval;              /**< 检查点间隔层数 */
    size_t max_checkpoints;                  /**< 最大检查点数量 */
    int recompute_layer_outputs;             /**< 是否重计算层输出 */
    int enable_memory_optimization;          /**< 是否启用内存优化 */
    float memory_budget_gb;                  /**< 内存预算（GB） */
} GradientCheckpointConfig;

/**
 * @brief 模型并行配置
 */
typedef struct {
    int enable_model_parallel;               /**< 是否启用模型并行 */
    size_t num_devices;                      /**< 设备数量 */
    size_t device_rank;                      /**< 当前设备排名 */
    size_t* layer_partition;                 /**< 各设备分配层数数组 */
    size_t first_layer_id;                   /**< 当前设备起始层ID */
    size_t last_layer_id;                    /**< 当前设备结束层ID */
    int enable_pipeline_parallel;            /**< 是否启用流水线并行 */
    size_t pipeline_stages;                  /**< 流水线阶段数 */
    int enable_communication_overlap;        /**< 是否启用通信重叠 */
} ModelParallelConfig;

/**
 * @brief 梯度压缩配置
 */
typedef struct {
    int enable_compression;                  /**< 是否启用梯度压缩 */
    float compression_ratio;                 /**< 压缩比（0.0-1.0） */
    size_t top_k;                            /**< Top-K稀疏化数量 */
    int enable_quantization;                 /**< 是否启用量化 */
    int quantization_bits;                   /**< 量化位数 */
    int enable_sparsification;               /**< 是否启用稀疏化 */
    int enable_error_feedback;               /**< 是否启用误差反馈 */
    float* error_feedback_buffer;            /**< 误差反馈缓冲区 */
    size_t error_feedback_size;              /**< 误差反馈缓冲区大小 */
} GradientCompressionConfig;

/**
 * @brief 激活检查点条目
 */
typedef struct {
    float* activation_data;                  /**< 检查点激活数据 */
    size_t activation_size;                  /**< 激活数据大小 */
    size_t layer_index;                      /**< 对应层索引 */
    int is_valid;                            /**< 是否有效 */
} ActivationCheckpointEntry;

/**
 * @brief 梯度压缩上下文（用于Top-K稀疏化压缩/解压）
 */
typedef struct {
    float* compressed_values;                /**< 压缩后的梯度值 */
    int32_t* compressed_indices;             /**< 压缩后的梯度索引 */
    size_t compressed_size;                  /**< 压缩后数据大小 */
    size_t original_size;                    /**< 原始数据大小 */
    float compression_ratio;                 /**< 实际压缩比 */
    float* error_feedback;                   /**< 误差反馈缓冲区 */
    size_t error_feedback_capacity;          /**< 误差反馈缓冲区容量 */
    MutexHandle lock;                        /**< 压缩上下文锁 */
} GradientCompressionContext;

/**
 * @brief 模型并行通信缓冲区（用于设备间激活/梯度传输）
 */
typedef struct {
    float* send_buffer;                      /**< 发送缓冲区 */
    float* recv_buffer;                      /**< 接收缓冲区 */
    size_t buffer_size;                      /**< 缓冲区大小 */
    int is_pipeline_stage;                   /**< 是否为流水线阶段 */
    size_t pipeline_stage_id;                /**< 流水线阶段ID */
} ModelParallelCommBuffer;

/**
 * @brief 带梯度检查点的前向传播
 *
 * 前向传播过程中按 checkpoint_interval 间隔保存激活值检查点，
 * 反向传播时从最近的检查点重计算中间激活值，大幅降低显存占用。
 * 对于万亿级模型，内存占用从 O(L) 降低到 O(L/interval)。
 *
 * @param network 网络句柄
 * @param input 输入向量
 * @param output 输出向量缓冲区
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_forward_with_checkpoint(LNN* network, const float* input, float* output);

/**
 * @brief 带梯度检查点的反向传播
 *
 * 从最近的激活检查点重计算中间层激活值，然后计算梯度。
 * 重计算过程只发生在检查点之间的层，显著减少内存占用。
 *
 * @param network 网络句柄
 * @param target 目标输出向量
 * @param loss 损失值输出缓冲区
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_backward_with_checkpoint(LNN* network, const float* target, float* loss);

/**
 * @brief 保存激活检查点
 *
 * 在指定层保存当前激活值到检查点缓冲区。
 *
 * @param network 网络句柄
 * @param layer_index 层索引
 * @param activation_data 激活数据指针
 * @param activation_size 激活数据大小
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_save_activation_checkpoint(LNN* network, size_t layer_index,
                                                const float* activation_data,
                                                size_t activation_size);

/**
 * @brief 从最近的检查点重计算激活值
 *
 * 从最近的激活检查点开始，前向重计算到指定层之间的所有层。
 *
 * @param network 网络句柄
 * @param target_layer 目标层索引（重计算到此层）
 * @param output_activation 输出激活值缓冲区
 * @param output_size 输出缓冲区大小
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_recompute_from_checkpoint(LNN* network, size_t target_layer,
                                               float* output_activation,
                                               size_t output_size);

/**
 * @brief 清理所有激活检查点
 *
 * @param network 网络句柄
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_clear_activation_checkpoints(LNN* network);

/**
 * @brief 分片感知的前向传播
 *
 * 将参数分片存储在前向传播中透明使用：
 * 1. 收集所有分片参数到统一缓冲区
 * 2. 执行标准前向传播
 * 3. 将更新后的参数重新分布到各分片
 *
 * @param network 网络句柄
 * @param input 输入向量
 * @param output 输出向量缓冲区
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_forward_sharded(LNN* network, const float* input, float* output);

/**
 * @brief 分片感知的反向传播
 *
 * 在反向传播中透明处理参数分片：
 * 1. 收集所有分片参数到统一缓冲区
 * 2. 执行标准反向传播计算梯度
 * 3. 将梯度分布回各分片
 * 4. 如果启用异步同步，触发梯度同步
 *
 * @param network 网络句柄
 * @param target 目标输出向量
 * @param loss 损失值输出缓冲区
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_backward_sharded(LNN* network, const float* target, float* loss);

/**
 * @brief 梯度压缩编码（Top-K稀疏化）
 *
 * 对梯度进行Top-K稀疏化压缩：
 * 1. 计算梯度绝对值
 * 2. 选择绝对值最大的K个元素
 * 3. 只保留这些元素的值和索引
 * 4. 如果启用误差反馈，将未选中的梯度累积到误差缓冲区
 *
 * @param gradients 原始梯度数据
 * @param num_gradients 梯度数量
 * @param ctx 压缩上下文（输出压缩后的数据和索引）
 * @param compression_ratio 压缩比（0.0-1.0）
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_gradient_compress(const float* gradients, size_t num_gradients,
                                       GradientCompressionContext* ctx,
                                       float compression_ratio);

/**
 * @brief 梯度压缩解码（Top-K稀疏化解压）
 *
 * 将压缩后的稀疏梯度恢复为密集梯度向量。
 *
 * @param gradients 输出解压后的梯度数据
 * @param num_gradients 梯度数量
 * @param ctx 压缩上下文（包含压缩后的数据和索引）
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_gradient_decompress(float* gradients, size_t num_gradients,
                                         const GradientCompressionContext* ctx);

/**
 * @brief 创建梯度压缩上下文
 *
 * @param original_size 原始梯度数据大小
 * @param compression_ratio 压缩比
 * @return GradientCompressionContext* 压缩上下文句柄，失败返回NULL
 */
SELFLNN_API GradientCompressionContext* lnn_gradient_compression_create(
    size_t original_size, float compression_ratio);

/**
 * @brief 释放梯度压缩上下文
 *
 * @param ctx 压缩上下文句柄
 */
SELFLNN_API void lnn_gradient_compression_free(GradientCompressionContext* ctx);

/**
 * @brief 模型并行前向传播
 *
 * 将网络层按模型并行配置切分到不同设备：
 * 1. 当前设备计算分配给自己的层（first_layer_id 到 last_layer_id）
 * 2. 前向传播经过层边界时，通过通信缓冲区传递激活值
 * 3. 支持流水线并行（如果启用）
 *
 * @param network 网络句柄
 * @param input 输入向量
 * @param output 输出向量缓冲区
 * @param mp_config 模型并行配置
 * @param comm_buffer 通信缓冲区
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_model_parallel_forward(LNN* network, const float* input,
                                            float* output,
                                            const ModelParallelConfig* mp_config,
                                            ModelParallelCommBuffer* comm_buffer);

/**
 * @brief 模型并行反向传播
 *
 * 反向传播模型并行：
 * 1. 当前设备计算分配给自己的层的梯度
 * 2. 跨设备传递梯度
 * 3. 支持流水线并行反向传播
 *
 * @param network 网络句柄
 * @param target 目标输出向量
 * @param loss 损失值输出缓冲区
 * @param mp_config 模型并行配置
 * @param comm_buffer 通信缓冲区
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_model_parallel_backward(LNN* network, const float* target,
                                             float* loss,
                                             const ModelParallelConfig* mp_config,
                                             ModelParallelCommBuffer* comm_buffer);

/**
 * @brief 计算模型并行层分配方案
 *
 * 根据设备数量和网络总层数，计算每台设备分配的层范围。
 *
 * @param total_layers 网络总层数
 * @param mp_config 模型并行配置（num_devices, device_rank 需要设置）
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_partition_layers_for_model_parallel(size_t total_layers,
                                                         ModelParallelConfig* mp_config);

/**
 * @brief 创建模型并行通信缓冲区
 *
 * @param buffer_size 缓冲区大小（元素数量）
 * @return ModelParallelCommBuffer* 通信缓冲区句柄，失败返回NULL
 */
SELFLNN_API ModelParallelCommBuffer* lnn_model_parallel_comm_create(size_t buffer_size);

/**
 * @brief 释放模型并行通信缓冲区
 *
 * @param comm_buffer 通信缓冲区句柄
 */
SELFLNN_API void lnn_model_parallel_comm_free(ModelParallelCommBuffer* comm_buffer);

/**
 * @brief 万亿级参数规模训练步骤
 *
 * 整合参数分片、梯度检查点和模型并行的完整训练步骤：
 * 1. 如果启用分片：收集分片参数到统一缓冲区
 * 2. 如果启用检查点：带检查点的前向传播
 * 3. 如果启用模型并行：模型并行前向传播
 * 4. 计算损失
 * 5. 如果启用检查点：带重计算的反向传播
 * 6. 如果启用模型并行：模型并行反向传播
 * 7. 如果启用分片：将梯度分布到各分片，触发同步
 * 8. 如果启用梯度压缩：压缩梯度后再同步
 *
 * @param network 网络句柄
 * @param input 输入向量
 * @param target 目标输出向量
 * @param output 输出向量缓冲区
 * @param loss 损失值输出缓冲区
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_trillion_scale_train_step(LNN* network, const float* input,
                                               const float* target, float* output,
                                               float* loss);

/**
 * @brief 异步梯度同步工作函数
 *
 * 在后台线程中执行异步梯度同步操作。
 * 如果启用了梯度压缩，先压缩再同步，大幅减少通信量。
 *
 * @param network 网络句柄
 * @return int 成功返回0，失败返回错误码
 */
SELFLNN_API int lnn_async_gradient_sync_worker(LNN* network);

/**
 * @brief 计算万亿级模型实际参数数量
 *
 * 根据配置计算完整模型（包括所有分片）的总参数数量。
 * 用于评估实际的模型规模是否达到万亿级。
 *
 * @param network 网络句柄
 * @return size_t 总参数数量
 */
SELFLNN_API size_t lnn_calculate_trillion_scale_params(const LNN* network);

/**
 * @brief 估算万亿级模型训练所需内存
 *
 * @param network 网络句柄
 * @param include_optimizer 是否包含优化器状态
 * @param include_activations 是否包含激活值
 * @return double 估算内存需求（GB）
 */
SELFLNN_API double lnn_estimate_trillion_scale_memory(const LNN* network,
                                                       int include_optimizer,
                                                       int include_activations);

/**
 * @brief 对比学习InfoNCE损失
 * 计算anchor-positive对比与多负样本的归一化损失
 */
SELFLNN_API int lnn_contrastive_loss(const float* anchor, const float* positive,
                                      const float* negatives, size_t num_negatives,
                                      size_t feature_dim, float temperature, float* loss);

/**
 * @brief 对比学习图像数据增强（中心裁剪缩放）
 */
SELFLNN_API int lnn_contrastive_augment_image(const float* image, size_t width, size_t height,
                                               size_t channels, float* augmented);

/**
 * @brief 自监督预训练（InfoNCE+数据增强对+负采样）
 */
SELFLNN_API int lnn_self_supervised_pretrain(LNN* network,
                                              const float* data, size_t num_samples,
                                              size_t feature_dim, int epochs);

/**
 * @brief 知识蒸馏（教师网络软标签KL散度+硬损失）
 */
SELFLNN_API int lnn_knowledge_distill(LNN* teacher, LNN* student,
                                       const float* data, size_t num_samples,
                                       size_t feature_dim, float temperature,
                                       float alpha, int epochs);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_EXTENDED_TRAINING_H */
