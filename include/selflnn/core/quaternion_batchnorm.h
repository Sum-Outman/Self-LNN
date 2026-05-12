/**
 * @file quaternion_batchnorm.h
 * @brief 四元数批归一化层
 *
 * 提供四元数批归一化的完整实现。
 * 对四元数(w,x,y,z)的每个分量独立计算统计量后，
 * 使用可学习的缩放γ和偏移β进行变换，保持四元数结构。
 *
 * 数学原理：
 *   对批数据计算每个分量的均值μ_c和方差σ²_c
 *   归一化: q_hat_c = (q_c - μ_c) / sqrt(σ²_c + ε)
 *   缩放平移: y_c = γ_c * q_hat_c + β_c
 *   其中c ∈ {w, x, y, z}
 *
 * 支持：
 *   - 训练模式：使用当前批统计量
 *   - 推理模式：使用运行均值/方差
 *   - 完整反向传播梯度
 *   - Adam优化器更新γ和β
 *   - 序列化保存/加载
 */

#ifndef SELFLNN_QUATERNION_BATCHNORM_H
#define SELFLNN_QUATERNION_BATCHNORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 四元数批归一化配置
 */
typedef struct {
    size_t num_features;           /**< 特征维度（四元数个数） */
    float epsilon;                 /**< 小常数防止除零（默认1e-5f） */
    float momentum;                /**< 运行统计量动量（默认0.9f） */
    float lr;                      /**< 学习率 */
    int use_scale;                 /**< 是否使用可学习缩放γ */
    int use_shift;                 /**< 是否使用可学习偏移β */
} QuaternionBatchNormConfig;

/**
 * @brief 四元数批归一化层句柄
 */
typedef struct QuaternionBatchNorm QuaternionBatchNorm;

/**
 * @brief 创建四元数批归一化层
 * @param config 配置（必填）
 * @return 四元数批归一化层句柄，失败返回NULL
 */
QuaternionBatchNorm* quaternion_batchnorm_create(const QuaternionBatchNormConfig* config);

/**
 * @brief 释放四元数批归一化层
 * @param layer 四元数批归一化层句柄
 */
void quaternion_batchnorm_free(QuaternionBatchNorm* layer);

/**
 * @brief 四元数批归一化前向传播
 *
 * 输入布局：[batch_size * num_features * 4]
 * 每个四元数依次存储4个分量(w,x,y,z)
 *
 * @param layer 四元数批归一化层句柄
 * @param input 输入数据 [batch * features * 4]
 * @param batch_size 批次大小
 * @param output 输出缓冲区 [batch * features * 4]
 * @param is_training 1=训练模式（使用批统计量），0=推理模式（使用运行统计量）
 * @return 成功返回0，失败返回-1
 */
int quaternion_batchnorm_forward(QuaternionBatchNorm* layer,
                                  const float* input,
                                  size_t batch_size,
                                  float* output,
                                  int is_training);

/**
 * @brief 四元数批归一化反向传播
 *
 * @param layer 四元数批归一化层句柄
 * @param output_grad 输出梯度 [batch * features * 4]
 * @param input 输入数据（用于计算梯度）[batch * features * 4]
 * @param batch_size 批次大小
 * @param input_grad 输入梯度缓冲区 [batch * features * 4]，可为NULL
 * @return 成功返回0，失败返回-1
 */
int quaternion_batchnorm_backward(QuaternionBatchNorm* layer,
                                   const float* output_grad,
                                   const float* input,
                                   size_t batch_size,
                                   float* input_grad);

/**
 * @brief 设置四元数批归一化层为推理模式
 * 锁定运行统计量，不再更新
 * @param layer 四元数批归一化层句柄
 */
void quaternion_batchnorm_eval(QuaternionBatchNorm* layer);

/**
 * @brief 设置四元数批归一化层为训练模式
 * 允许更新运行统计量
 * @param layer 四元数批归一化层句柄
 */
void quaternion_batchnorm_train(QuaternionBatchNorm* layer);

/**
 * @brief 重置四元数批归一化层运行统计量
 * @param layer 四元数批归一化层句柄
 */
void quaternion_batchnorm_reset_stats(QuaternionBatchNorm* layer);

/**
 * @brief 重置四元数批归一化层优化器
 * @param layer 四元数批归一化层句柄
 */
void quaternion_batchnorm_reset_optimizer(QuaternionBatchNorm* layer);

/**
 * @brief 设置四元数批归一化层的学习率
 * @param layer 四元数批归一化层句柄
 * @param lr 新学习率
 */
void quaternion_batchnorm_set_lr(QuaternionBatchNorm* layer, float lr);

/**
 * @brief 获取四元数批归一化层可训练参数数量
 * @param layer 四元数批归一化层句柄
 * @return 参数数量（γ和β各4个分量，共8个参数），失败返回-1
 */
int quaternion_batchnorm_param_count(const QuaternionBatchNorm* layer);

/**
 * @brief 保存四元数批归一化层到文件
 * @param layer 四元数批归一化层句柄
 * @param filename 文件名
 * @return 成功返回0，失败返回-1
 */
int quaternion_batchnorm_save(const QuaternionBatchNorm* layer, const char* filename);

/**
 * @brief 从文件加载四元数批归一化层
 * @param filename 文件名
 * @return 四元数批归一化层句柄，失败返回NULL
 */
QuaternionBatchNorm* quaternion_batchnorm_load(const char* filename);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_QUATERNION_BATCHNORM_H */
