/**
 * @file bdi_model.c
 * @brief BDI（信念-愿望-意图）模型实现
 *
 * 从 self_cognition.c 中提取的 BDI 核心模型。
 * 原始代码位置: src/cognition/self_cognition.c (约6424-6717行)
 *
 * M-7修复: 将BDI模型从 self_cognition.c (8400+行) 中拆分出来，
 * 减少单文件代码量，提高模块可维护性。
 */

#include "selflnn/cognition/bdi_model.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * 贝叶斯信念更新
 * ============================================================================ */

/**
 * @brief 贝叶斯信念更新
 *
 * 使用观测数据以贝叶斯方式更新信念向量。
 * prior_weight = 1.0 - certainty * 0.5;
 * posterior_weight = certainty * 0.5;
 */
void bdi_update_belief_bayesian(float* belief, size_t dim,
                                  const float* observation, float certainty) {
    if (!belief || !observation || dim == 0) return;

    /* 确保 certainty 在有效范围 */
    if (certainty < 0.0f) certainty = 0.0f;
    if (certainty > 1.0f) certainty = 1.0f;

    float prior_weight = 1.0f - certainty * 0.5f;
    float posterior_weight = certainty * 0.5f;

    for (size_t i = 0; i < dim; i++) {
        belief[i] = belief[i] * prior_weight + observation[i] * posterior_weight;
        /* 裁剪到 [0, 1] */
        if (belief[i] < 0.0f) belief[i] = 0.0f;
        if (belief[i] > 1.0f) belief[i] = 1.0f;
    }
}

/* ============================================================================
 * 意图计算
 * ============================================================================ */

/**
 * @brief 从欲望和信念计算意图
 *
 * 意图 = desire 逐元素乘以 belief（哈达玛积），然后 L1 归一化。
 */
void bdi_compute_intention_from_desire_belief(float* intention, const float* desire,
                                               const float* belief, size_t dim) {
    if (!intention || !desire || !belief || dim == 0) return;

    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        intention[i] = desire[i] * belief[i];
        sum += intention[i];
    }
    /* L1 归一化 */
    if (sum > 1e-10f) {
        for (size_t i = 0; i < dim; i++) {
            intention[i] /= sum;
        }
    }
    /* 如果 sum 接近 0，保持 intention 为零向量 */
}

/* ============================================================================
 * 初始化辅助函数
 * ============================================================================ */

/**
 * @brief 初始化信念向量为均等分布（最大熵）
 */
void bdi_init_belief_uniform(float* belief, size_t dim) {
    if (!belief || dim == 0) return;
    for (size_t i = 0; i < dim; i++) {
        belief[i] = 0.5f;
    }
}

/**
 * @brief 初始化意图向量为零
 */
void bdi_init_intention_zero(float* intention, size_t dim) {
    if (!intention || dim == 0) return;
    memset(intention, 0, dim * sizeof(float));
}
