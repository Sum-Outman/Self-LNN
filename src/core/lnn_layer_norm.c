/**
 * @file lnn_layer_norm.c
 * @brief 液态神经网络层归一化完整实现
 *
 * P1-001: 为CfC深层网络提供训练稳定性
 *
 * 层归一化 (Ba, Kiros & Hinton, 2016):
 *   在特征维度上对每个样本独立归一化，消除内部协变量偏移
 *   不依赖批次大小，适合小批量训练、在线学习和RNN/液态网络
 *
 * 前向传播（训练模式）:
 *   1. 计算均值: μ = (1/H) Σ x_i
 *   2. 计算方差: σ² = (1/H) Σ (x_i - μ)²
 *   3. 归一化:   x̂_i = (x_i - μ) / σ⁻ 其中 σ⁻ = √(σ² + ε)
 *   4. 缩放偏移: y_i = γ_i * x̂_i + β_i
 *
 * 反向传播（完整解析梯度）:
 *   设 L 为损失函数，已知上游梯度 ∂L/∂y_i
 *   ∂L/∂γ_i = ∂L/∂y_i * x̂_i
 *   ∂L/∂β_i = ∂L/∂y_i
 *   ∂L/∂x̂_i = ∂L/∂y_i * γ_i
 *   ∂L/∂σ²  = Σ_i ∂L/∂x̂_i * (x_i - μ) * (-0.5) * (σ² + ε)^(-3/2)
 *   ∂L/∂μ   = Σ_i ∂L/∂x̂_i * (-1/σ⁻) + ∂L/∂σ² * (-2/H) * Σ_i (x_i - μ)
 *   ∂L/∂x_i = ∂L/∂x̂_i * (1/σ⁻) + ∂L/∂σ² * (2/H)*(x_i - μ) + ∂L/∂μ * (1/H)
 *
 * 推理模式: 训练/推理行为一致（层归一化不使用运行统计量）
 */

#define SELFLNN_IMPLEMENTATION
#include "selflnn/core/lnn_layer_norm.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ========================================================================
 * 内部数据结构
 * ======================================================================== */

struct LayerNorm {
    LayerNormConfig config;

    float* gamma;                /**< 可学习缩放因子 [feature_size] */
    float* beta;                 /**< 可学习偏移 [feature_size] */
    float* gamma_grad;           /**< γ梯度 [feature_size] */
    float* beta_grad;            /**< β梯度 [feature_size] */

    /* 训练缓存（反向传播使用） */
    float* saved_mean;           /**< 当前批均值 [1]——每个样本独立计算，但结构上保留 */
    float* saved_ivar;           /**< 1/√(σ²+ε) [1] */
    float* saved_normalized;     /**< 归一化值 x̂_i [feature_size] */
    float* saved_input;          /**< 前向传播时的原始输入 [feature_size] */

    int is_initialized;
};

/* ========================================================================
 * 公共API实现
 * ======================================================================== */

LayerNormConfig layer_norm_default_config(size_t feature_size) {
    LayerNormConfig config;
    memset(&config, 0, sizeof(config));
    config.feature_size = feature_size;
    config.epsilon = 1e-5f;
    config.use_affine = 1;
    config.init_gamma = 1.0f;
    config.init_beta = 0.0f;
    config.momentum = 0.0f;
    return config;
}

LayerNorm* layer_norm_create(const LayerNormConfig* config) {
    if (!config || config->feature_size == 0) return NULL;

    LayerNorm* norm = (LayerNorm*)safe_calloc(1, sizeof(LayerNorm));
    if (!norm) return NULL;

    norm->config = *config;
    if (norm->config.epsilon <= 0.0f) norm->config.epsilon = 1e-5f;

    size_t N = config->feature_size;
    norm->gamma = (float*)safe_calloc(N, sizeof(float));
    norm->beta = (float*)safe_calloc(N, sizeof(float));
    norm->gamma_grad = (float*)safe_calloc(N, sizeof(float));
    norm->beta_grad = (float*)safe_calloc(N, sizeof(float));
    norm->saved_mean = (float*)safe_calloc(1, sizeof(float));
    norm->saved_ivar = (float*)safe_calloc(1, sizeof(float));
    norm->saved_normalized = (float*)safe_calloc(N, sizeof(float));
    norm->saved_input = (float*)safe_calloc(N, sizeof(float));

    if (!norm->gamma || !norm->beta || !norm->gamma_grad ||
        !norm->beta_grad || !norm->saved_mean || !norm->saved_ivar ||
        !norm->saved_normalized || !norm->saved_input) {
        layer_norm_free(norm);
        return NULL;
    }

    /* 初始化γ=1, β=0 */
    if (config->use_affine) {
        for (size_t i = 0; i < N; i++) {
            norm->gamma[i] = config->init_gamma;
            norm->beta[i] = config->init_beta;
        }
    } else {
        for (size_t i = 0; i < N; i++) {
            norm->gamma[i] = 1.0f;
            norm->beta[i] = 0.0f;
        }
    }

    norm->is_initialized = 1;
    return norm;
}

void layer_norm_free(LayerNorm* norm) {
    if (!norm) return;
    safe_free((void**)&norm->gamma);
    safe_free((void**)&norm->beta);
    safe_free((void**)&norm->gamma_grad);
    safe_free((void**)&norm->beta_grad);
    safe_free((void**)&norm->saved_mean);
    safe_free((void**)&norm->saved_ivar);
    safe_free((void**)&norm->saved_normalized);
    safe_free((void**)&norm->saved_input);
    safe_free((void**)&norm);
}

/**
 * @brief 层归一化前向传播完整实现
 *
 * 层归一化在训练和推理模式行为完全一致，均使用当前样本的统计量。
 * 这与批归一化不同——层归一化不维护运行均值/方差。
 */
int layer_norm_forward(LayerNorm* norm, const float* input, float* output, int is_training) {
    if (!norm || !input || !output) return -1;
    (void)is_training;  /* 层归一化训练/推理行为一致 */

    size_t N = norm->config.feature_size;
    float eps = norm->config.epsilon;

    /* 步骤1: 计算均值 μ = (1/N) Σ x_i */
    float mean = 0.0f;
    for (size_t i = 0; i < N; i++) {
        mean += input[i];
    }
    mean /= (float)N;

    /* 步骤2: 计算方差 σ² = (1/N) Σ (x_i - μ)² */
    float var = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= (float)N;

    /* 步骤3: 计算 1/√(σ²+ε) */
    float ivar = 1.0f / sqrtf(var + eps);

    /* 步骤4: 归一化 x̂_i = (x_i - μ) * ivar，然后缩放偏移 */
    if (norm->config.use_affine) {
        for (size_t i = 0; i < N; i++) {
            float normalized = (input[i] - mean) * ivar;
            output[i] = norm->gamma[i] * normalized + norm->beta[i];
        }
    } else {
        for (size_t i = 0; i < N; i++) {
            output[i] = (input[i] - mean) * ivar;
        }
    }

    /* 保存缓存供反向传播使用 */
    norm->saved_mean[0] = mean;
    norm->saved_ivar[0] = ivar;
    for (size_t i = 0; i < N; i++) {
        norm->saved_normalized[i] = (input[i] - mean) * ivar;
        norm->saved_input[i] = input[i];
    }

    return 0;
}

/**
 * @brief 层归一化反向传播完整解析梯度实现
 *
 * 基于BatchNorm论文的链式法则推导，适用于层归一化的特征维归一化。
 */
int layer_norm_backward(LayerNorm* norm, const float* output_grad, float* input_grad) {
    if (!norm || !output_grad || !input_grad) return -1;

    size_t N = norm->config.feature_size;
    float mean = norm->saved_mean[0];
    float ivar = norm->saved_ivar[0];
    float eps = norm->config.epsilon;

    /* 计算 ∂L/∂γ 和 ∂L/∂β */
    memset(norm->gamma_grad, 0, N * sizeof(float));
    memset(norm->beta_grad, 0, N * sizeof(float));

    if (norm->config.use_affine) {
        for (size_t i = 0; i < N; i++) {
            float dout = output_grad[i];
            norm->gamma_grad[i] += dout * norm->saved_normalized[i];
            norm->beta_grad[i] += dout;
        }
    }

    /* 中间量: 先计算 ∂L/∂x̂_i = ∂L/∂y_i * γ_i */
    /* 对于无affine的情况，γ_i = 1 */

    /* ∂L/∂σ² = Σ_i ∂L/∂x̂_i * (x_i - μ) * (-0.5) * (σ²+ε)^(-3/2) */
    /* 注意: ivar = 1/√(σ²+ε), 所以 (σ²+ε)^(-3/2) = ivar^3 */
    float dvar = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float dx_hat;
        if (norm->config.use_affine) {
            dx_hat = output_grad[i] * norm->gamma[i];
        } else {
            dx_hat = output_grad[i];
        }
        float x_centered = norm->saved_input[i] - mean;
        dvar += dx_hat * x_centered;
    }
    dvar *= -0.5f * ivar * ivar * ivar;

    /* ∂L/∂μ = Σ_i ∂L/∂x̂_i * (-ivar) + ∂L/∂σ² * (-2/N) * Σ_i (x_i - μ) */
    /* 由于 Σ_i (x_i - μ) = 0, 第二项为0 */
    float dmean = 0.0f;
    for (size_t i = 0; i < N; i++) {
        float dx_hat;
        if (norm->config.use_affine) {
            dx_hat = output_grad[i] * norm->gamma[i];
        } else {
            dx_hat = output_grad[i];
        }
        dmean += dx_hat * (-ivar);
    }

    /* ∂L/∂x_i = ∂L/∂x̂_i * ivar + ∂L/∂σ² * (2/N)*(x_i-μ) + ∂L/∂μ * (1/N) */
    float inv_N = 1.0f / (float)N;
    for (size_t i = 0; i < N; i++) {
        float dx_hat;
        if (norm->config.use_affine) {
            dx_hat = output_grad[i] * norm->gamma[i];
        } else {
            dx_hat = output_grad[i];
        }
        float x_centered = norm->saved_input[i] - mean;
        input_grad[i] = dx_hat * ivar
                      + dvar * 2.0f * inv_N * x_centered
                      + dmean * inv_N;
    }

    return 0;
}

const float* layer_norm_get_gamma(const LayerNorm* norm) {
    return norm ? norm->gamma : NULL;
}

const float* layer_norm_get_beta(const LayerNorm* norm) {
    return norm ? norm->beta : NULL;
}

const float* layer_norm_get_gamma_grad(const LayerNorm* norm) {
    return norm ? norm->gamma_grad : NULL;
}

const float* layer_norm_get_beta_grad(const LayerNorm* norm) {
    return norm ? norm->beta_grad : NULL;
}

size_t layer_norm_get_feature_size(const LayerNorm* norm) {
    return norm ? norm->config.feature_size : 0;
}

/**
 * @brief P0-001: 更新层归一化的可学习参数（γ、β）
 * 由外部优化器或cfc_apply_cell_gradients驱动，统一在批量结束后更新。
 * gamma -= lr * gamma_grad
 * beta  -= lr * beta_grad
 */
int layer_norm_update_params(LayerNorm* norm, float learning_rate) {
    if (!norm || !norm->gamma || !norm->beta || !norm->gamma_grad || !norm->beta_grad) return -1;
    size_t N = norm->config.feature_size;
    float lr = learning_rate * 0.5f;
    for (size_t i = 0; i < N; i++) {
        norm->gamma[i] -= lr * norm->gamma_grad[i];
        norm->beta[i]  -= lr * norm->beta_grad[i];
    }
    memset(norm->gamma_grad, 0, N * sizeof(float));
    memset(norm->beta_grad,  0, N * sizeof(float));
    return 0;
}

size_t layer_norm_get_num_params(const LayerNorm* norm) {
    if (!norm) return 0;
    return norm->config.use_affine ? norm->config.feature_size * 2 : 0;
}

int layer_norm_get_params(LayerNorm* norm, float* params_out, float* grads_out) {
    if (!norm || !params_out) return -1;
    size_t N = norm->config.feature_size;
    memcpy(params_out, norm->gamma, N * sizeof(float));
    memcpy(params_out + N, norm->beta, N * sizeof(float));
    if (grads_out) {
        memcpy(grads_out, norm->gamma_grad, N * sizeof(float));
        memcpy(grads_out + N, norm->beta_grad, N * sizeof(float));
    }
    return (int)(N * 2);
}

int layer_norm_set_params(LayerNorm* norm, const float* new_params) {
    if (!norm || !new_params) return -1;
    size_t N = norm->config.feature_size;
    memcpy(norm->gamma, new_params, N * sizeof(float));
    memcpy(norm->beta, new_params + N, N * sizeof(float));
    return 0;
}

void layer_norm_reset(LayerNorm* norm) {
    if (!norm) return;
    size_t N = norm->config.feature_size;
    for (size_t i = 0; i < N; i++) {
        norm->gamma[i] = norm->config.init_gamma;
        norm->beta[i] = norm->config.init_beta;
    }
    memset(norm->gamma_grad, 0, N * sizeof(float));
    memset(norm->beta_grad, 0, N * sizeof(float));
    memset(norm->saved_normalized, 0, N * sizeof(float));
    memset(norm->saved_input, 0, N * sizeof(float));
}
