/**
 * @file lnn_layer_norm.h
 * @brief 液态神经网络层归一化 (Layer Normalization)
 * 
 * P1-001: 为CfC深层网络提供训练稳定性
 * 
 * 层归一化在特征维度上对每个样本进行归一化，与批归一化不同：
 *   - 不依赖批次大小，适合小批量训练和在线学习
 *   - 训练/推理行为一致，无需运行统计量切换
 *   - 消除内部协变量偏移（Internal Covariate Shift）
 * 
 * 数学公式:
 *   μ = (1/H) Σ x_i           // 均值
 *   σ² = (1/H) Σ (x_i - μ)²   // 方差
 *   x̂_i = (x_i - μ) / √(σ² + ε)  // 归一化
 *   y_i = γ_i * x̂_i + β_i     // 缩放与偏移
 * 
 * 反向传播:
 *   ∂L/∂γ_i = Σ ∂L/∂y_i * x̂_i
 *   ∂L/∂β_i = Σ ∂L/∂y_i
 *   ∂L/∂x̂_i = ∂L/∂y_i * γ_i
 *   通过均值/方差传播: ∂L/∂x = ...
 */

#ifndef SELFLNN_LNN_LAYER_NORM_H
#define SELFLNN_LNN_LAYER_NORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 层归一化配置
 */
typedef struct {
    size_t feature_size;         /**< 特征维度（归一化的维度大小） */
    float epsilon;               /**< 数值稳定性参数（默认1e-5） */
    int use_affine;              /**< 是否使用可学习的缩放γ和偏移β（默认1=启用） */
    float init_gamma;            /**< γ的初始值（默认1.0） */
    float init_beta;             /**< β的初始值（默认0.0） */
    float momentum;              /**< 运行统计量动量（当前版本层归一化不使用，保留接口兼容性） */
} LayerNormConfig;

/**
 * @brief 层归一化句柄（不透明类型）
 */
typedef struct LayerNorm LayerNorm;

/**
 * @brief 获取默认层归一化配置
 */
LayerNormConfig layer_norm_default_config(size_t feature_size);

/**
 * @brief 创建层归一化实例
 * @param config 归一化配置
 * @return LayerNorm* 实例句柄，失败返回NULL
 */
LayerNorm* layer_norm_create(const LayerNormConfig* config);

/**
 * @brief 释放层归一化实例
 */
void layer_norm_free(LayerNorm* norm);

/**
 * @brief 层归一化前向传播
 * @param norm 层归一化实例
 * @param input 输入数组 (长度=feature_size)
 * @param output 输出数组 (长度=feature_size)
 * @param is_training 是否训练模式（1=训练，0=推理）
 * @return int 成功返回0，失败返回-1
 */
int layer_norm_forward(LayerNorm* norm, const float* input, float* output, int is_training);

/**
 * @brief 层归一化反向传播
 * @param norm 层归一化实例
 * @param output_grad 输出梯度 (长度=feature_size)
 * @param input_grad 输入梯度输出缓冲区 (长度=feature_size)
 * @return int 成功返回0，失败返回-1
 */
int layer_norm_backward(LayerNorm* norm, const float* output_grad, float* input_grad);

/**
 * @brief 获取γ（缩放）参数
 */
const float* layer_norm_get_gamma(const LayerNorm* norm);

/**
 * @brief 获取β（偏移）参数
 */
const float* layer_norm_get_beta(const LayerNorm* norm);

/**
 * @brief 获取γ梯度
 */
const float* layer_norm_get_gamma_grad(const LayerNorm* norm);

/**
 * @brief 获取β梯度
 */
const float* layer_norm_get_beta_grad(const LayerNorm* norm);

/**
 * @brief 获取特征维度大小
 */
size_t layer_norm_get_feature_size(const LayerNorm* norm);

/**
 * @brief P0-001: 更新层归一化的可学习参数（γ缩放、β偏移）
 * @param norm 层归一化实例
 * @param learning_rate 学习率（内部自动减半以提高稳定性）
 * @return 0成功，-1失败
 */
int layer_norm_update_params(LayerNorm* norm, float learning_rate);

/**
 * @brief 获取可训练参数数量
 */
size_t layer_norm_get_num_params(const LayerNorm* norm);

/**
 * @brief 获取所有可训练参数（用于优化器批量更新）
 * @param norm 层归一化实例
 * @param params_out 参数输出缓冲区 (gamma[0..N-1], beta[0..N-1])
 * @param grads_out 梯度输出缓冲区（可为NULL）
 * @return int 参数总数
 */
int layer_norm_get_params(LayerNorm* norm, float* params_out, float* grads_out);

/**
 * @brief 使用外部参数更新（由优化器调用）
 */
int layer_norm_set_params(LayerNorm* norm, const float* new_params);

/**
 * @brief 重置归一化层状态（用于模型重置）
 */
void layer_norm_reset(LayerNorm* norm);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LNN_LAYER_NORM_H */
