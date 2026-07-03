/**
 * @file bdi_model.h
 * @brief BDI（信念-愿望-意图）模型接口
 *
 * 从 self_cognition.c 中提取的 BDI 核心模型 API。
 * 提供贝叶斯信念更新和从欲望+信念计算意图的基础函数。
 * 这些是心智理论（Theory of Mind）和多智能体交互的构建块。
 *
 * 原始代码位置: src/cognition/self_cognition.c
 * 提取日期: 2026-07-03 (M-7修复)
 */

#ifndef SELFLNN_BDI_MODEL_H
#define SELFLNN_BDI_MODEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 贝叶斯信念更新
 *
 * 使用观测数据以贝叶斯方式更新信念向量。
 * 信念 = 信念 * prior_weight + 观测 * posterior_weight
 * prior_weight = 1.0 - certainty * 0.5
 * posterior_weight = certainty * 0.5
 *
 * @param belief  信念向量（原地更新），每个元素范围 [0, 1]
 * @param dim     信念向量维度
 * @param observation 观测数据向量
 * @param certainty   观测确定性 (0.0 ~ 1.0)，越高则后验权重越大
 */
void bdi_update_belief_bayesian(float* belief, size_t dim,
                                 const float* observation, float certainty);

/**
 * @brief 从欲望和信念计算意图
 *
 * 意图 = desire 逐元素乘以 belief（哈达玛积），然后 L1 归一化。
 *
 * @param intention 输出意图向量（调用方分配）
 * @param desire    欲望向量
 * @param belief    信念向量
 * @param dim       向量维度
 */
void bdi_compute_intention_from_desire_belief(float* intention, const float* desire,
                                               const float* belief, size_t dim);

/**
 * @brief 初始化信念向量为均等分布
 *
 * 将所有元素设为 0.5（最大熵信念）。
 *
 * @param belief 信念向量
 * @param dim    向量维度
 */
void bdi_init_belief_uniform(float* belief, size_t dim);

/**
 * @brief 初始化意图向量为零
 *
 * @param intention 意图向量
 * @param dim       向量维度
 */
void bdi_init_intention_zero(float* intention, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_BDI_MODEL_H */
