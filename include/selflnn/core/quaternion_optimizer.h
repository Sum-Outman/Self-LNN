/**
 * @file quaternion_optimizer.h
 * @brief 四元数优化器
 *
 * 提供四元数感知的优化器，在四元数流形上进行梯度优化。
 * 区别于标量优化器，四元数优化器考虑到四元数的4维结构和单位范数约束。
 *
 * 支持的优化器类型：
 * - QSGD: 四元数随机梯度下降（含动量）
 * - QAdam: 四元数Adam（各分量独立动量+自适应学习率）
 * - QAdamW: 四元数AdamW（带解耦权重衰减）
 *
 * 数学原理：
 *   四元数参数 q ∈ H, 梯度 ∇q ∈ H
 *   更新规则保留了四元数的代数结构和几何意义
 */

#ifndef SELFLNN_QUATERNION_OPTIMIZER_H
#define SELFLNN_QUATERNION_OPTIMIZER_H

#include <stddef.h>
#include "selflnn/utils/math_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 四元数优化器类型
 */
typedef enum {
    QUAT_OPTIMIZER_SGD = 0,        /**< 四元数SGD */
    QUAT_OPTIMIZER_MOMENTUM = 1,   /**< 四元数动量SGD */
    QUAT_OPTIMIZER_ADAM = 2,       /**< 四元数Adam */
    QUAT_OPTIMIZER_ADAMW = 3       /**< 四元数AdamW（解耦权重衰减） */
} QuatOptimizerType;

/**
 * @brief 四元数优化器配置
 */
typedef struct {
    QuatOptimizerType type;        /**< 优化器类型 */
    float learning_rate;           /**< 学习率 */
    float beta1;                   /**< Adam beta1（一阶矩衰减） */
    float beta2;                   /**< Adam beta2（二阶矩衰减） */
    float epsilon;                 /**< 数值稳定性常数 */
    float weight_decay;            /**< 权重衰减系数 */
    float momentum;                /**< 动量系数（仅momentum有效） */
    int use_nesterov;              /**< 是否使用Nesterov动量 */
    int clamp_grad_norm;           /**< 是否裁剪四元数梯度范数（防止爆炸） */
    float max_grad_norm;           /**< 最大梯度范数 */
} QuatOptimizerConfig;

/**
 * @brief 四元数优化器句柄（不透明类型）
 */
typedef struct QuatOptimizer QuatOptimizer;

/**
 * @brief 创建四元数优化器
 *
 * @param config 优化器配置
 * @return QuatOptimizer* 优化器句柄，失败返回NULL
 */
QuatOptimizer* quat_optimizer_create(const QuatOptimizerConfig* config);

/**
 * @brief 释放四元数优化器
 *
 * @param optimizer 优化器句柄
 */
void quat_optimizer_free(QuatOptimizer* optimizer);

/**
 * @brief 一步优化：更新四元数参数数组
 *
 * 对四元数数组进行优化更新，保持四元数的代数结构。
 * 每个参数是一个Quaternion结构。
 *
 * @param optimizer 优化器句柄
 * @param params 四元数参数数组（长度 num_quaternions）
 * @param grads 四元数梯度数组（长度 num_quaternions）
 * @param num_quaternions 四元数数量
 * @param step 当前步数（从1开始，用于Adam偏差校正）
 * @return int 成功返回0，失败返回-1
 */
int quat_optimizer_step(QuatOptimizer* optimizer, Quaternion* params,
                         const Quaternion* grads, size_t num_quaternions, size_t step);

/**
 * @brief 标量浮点优化步（用于非四元数部分，如偏置或输出投影）
 *
 * @param optimizer 优化器句柄
 * @param params 浮点参数数组
 * @param grads 浮点梯度数组
 * @param num_params 参数数量
 * @param step 当前步数
 * @return int 成功返回0，失败返回-1
 */
int quat_optimizer_step_scalar(QuatOptimizer* optimizer, float* params,
                                const float* grads, size_t num_params, size_t step);

/**
 * @brief 重置优化器状态
 *
 * @param optimizer 优化器句柄
 */
void quat_optimizer_reset(QuatOptimizer* optimizer);

/**
 * @brief 设置学习率
 *
 * @param optimizer 优化器句柄
 * @param lr 新学习率
 */
void quat_optimizer_set_lr(QuatOptimizer* optimizer, float lr);

/**
 * @brief 获取学习率
 *
 * @param optimizer 优化器句柄
 * @return float 当前学习率
 */
float quat_optimizer_get_lr(const QuatOptimizer* optimizer);

/**
 * @brief 获取四元数感知的默认配置
 *
 * @param type 优化器类型
 * @param learning_rate 学习率
 * @return QuatOptimizerConfig 推荐配置
 */
QuatOptimizerConfig quat_optimizer_default_config(QuatOptimizerType type, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_QUATERNION_OPTIMIZER_H */
