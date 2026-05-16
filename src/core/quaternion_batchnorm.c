/**
 * @file quaternion_batchnorm.c
 * @brief 四元数批归一化层实现
 *
 * 对四元数(w,x,y,z)的每个分量独立计算批统计量后，
 * 使用可学习的缩放γ和偏移β进行变换。
 *
 * 训练模式：使用当前批的均值和方差
 * 推理模式：使用运行时累积的运行均值/方差
 *
 * 数学原理：
 *   1. μ_c = mean(q_c) over batch
 *   2. σ²_c = var(q_c) over batch
 *   3. q̂_c = (q_c - μ_c) / √(σ²_c + ε)
 *   4. y_c = γ_c * q̂_c + β_c
 *   其中 c ∈ {w, x, y, z}
 *
 * 反向传播：
 *   ∂L/∂γ_c = Σ ∂L/∂y_c * q̂_c
 *   ∂L/∂β_c = Σ ∂L/∂y_c
 *   ∂L/∂q̂_c = ∂L/∂y_c * γ_c
 *   ∂L/∂σ²_c = Σ ∂L/∂q̂_c * (q_c - μ_c) * (-0.5) * (σ²_c + ε)^(-3/2)
 *   ∂L/∂μ_c = Σ ∂L/∂q̂_c * (-1/√(σ²_c+ε)) + ∂L/∂σ²_c * (-2) * mean(q_c - μ_c)
 *   ∂L/∂q_c = ∂L/∂q̂_c / √(σ²_c+ε) + 2 * ∂L/∂σ²_c * (q_c - μ_c) / N + ∂L/∂μ_c / N
 */

#include "selflnn/core/quaternion_batchnorm.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================
 *   内部数据结构
 * ============================ */

struct QuaternionBatchNorm {
    QuaternionBatchNormConfig config;

    /* 可学习参数 [num_features × 4] */
    float* gamma;               /**< 缩放因子 (w,x,y,z) */
    float* beta;                /**< 偏移 (w,x,y,z) */

    /* 运行统计量 [num_features × 4] */
    float* running_mean;        /**< 运行均值 */
    float* running_var;         /**< 运行方差 */

    /* 梯度和Adam状态 */
    float* gamma_grad;
    float* beta_grad;
    float* gamma_m;
    float* gamma_v;
    float* beta_m;
    float* beta_v;
    size_t adam_step;

    /* 训练缓存 */
    float* saved_mean;          /**< 当前批均值 */
    float* saved_var;           /**< 当前批方差 */
    float* saved_std_inv;       /**< 1/sqrt(var + eps) */
    float* saved_normalized;    /**< 归一化后的四元数 */
    size_t saved_normalized_capacity;  /**< saved_normalized已分配容量(元素数) */

    int is_training;            /**< 训练模式标志 */
};

/* ============================
 *   内部辅助函数
 * ============================ */

/**
 * @brief Adam优化器单参数更新
 */
static void adam_update(float* param, float* m, float* v, float grad,
                        float lr, float beta1, float beta2, float eps,
                        float corr1, float corr2)
{
    m[0] = beta1 * m[0] + (1.0f - beta1) * grad;
    v[0] = beta2 * v[0] + (1.0f - beta2) * grad * grad;
    float m_hat = m[0] / corr1;
    float v_hat = v[0] / corr2;
    param[0] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

/* ============================
 *   实现
 * ============================ */

QuaternionBatchNorm* quaternion_batchnorm_create(const QuaternionBatchNormConfig* config)
{
    if (!config || config->num_features == 0) return NULL;

    QuaternionBatchNorm* layer = (QuaternionBatchNorm*)safe_calloc(1, sizeof(QuaternionBatchNorm));
    if (!layer) return NULL;

    layer->config = *config;
    if (config->epsilon <= 0.0f) layer->config.epsilon = 1e-5f;
    if (config->momentum <= 0.0f || config->momentum >= 1.0f) layer->config.momentum = 0.9f;
    if (config->lr <= 0.0f) layer->config.lr = 0.001f;
    layer->config.use_scale = config->use_scale;
    layer->config.use_shift = config->use_shift;
    layer->is_training = 1;

    size_t nf = config->num_features;
    size_t qsize = nf * 4;

    layer->gamma = (float*)safe_calloc(config->use_scale ? qsize : 1, sizeof(float));
    layer->beta = (float*)safe_calloc(config->use_shift ? qsize : 1, sizeof(float));
    layer->running_mean = (float*)safe_calloc(qsize, sizeof(float));
    layer->running_var = (float*)safe_calloc(qsize, sizeof(float));
    layer->gamma_grad = (float*)safe_calloc(config->use_scale ? qsize : 1, sizeof(float));
    layer->beta_grad = (float*)safe_calloc(config->use_shift ? qsize : 1, sizeof(float));
    layer->gamma_m = (float*)safe_calloc(config->use_scale ? qsize : 1, sizeof(float));
    layer->gamma_v = (float*)safe_calloc(config->use_scale ? qsize : 1, sizeof(float));
    layer->beta_m = (float*)safe_calloc(config->use_shift ? qsize : 1, sizeof(float));
    layer->beta_v = (float*)safe_calloc(config->use_shift ? qsize : 1, sizeof(float));
    layer->saved_mean = (float*)safe_calloc(qsize, sizeof(float));
    layer->saved_var = (float*)safe_calloc(qsize, sizeof(float));
    layer->saved_std_inv = (float*)safe_calloc(qsize, sizeof(float));
    layer->saved_normalized = NULL;

    if (!layer->gamma || !layer->beta || !layer->running_mean || !layer->running_var ||
        !layer->gamma_grad || !layer->beta_grad || !layer->gamma_m || !layer->gamma_v ||
        !layer->beta_m || !layer->beta_v || !layer->saved_mean || !layer->saved_var ||
        !layer->saved_std_inv) {
        quaternion_batchnorm_free(layer);
        return NULL;
    }

    /* 初始化γ=1.0, β=0.0 */
    if (config->use_scale) {
        for (size_t i = 0; i < qsize; i++) layer->gamma[i] = 1.0f;
    }
    if (config->use_shift) {
        for (size_t i = 0; i < qsize; i++) layer->beta[i] = 0.0f;
    }

    /* 运行方差初始化为1.0 */
    for (size_t i = 0; i < qsize; i++) layer->running_var[i] = 1.0f;

    /* 初始化容量跟踪 */
    layer->saved_normalized_capacity = 0;

    layer->adam_step = 0;
    return layer;
}

void quaternion_batchnorm_free(QuaternionBatchNorm* layer)
{
    if (!layer) return;
    safe_free((void**)&layer->gamma);
    safe_free((void**)&layer->beta);
    safe_free((void**)&layer->running_mean);
    safe_free((void**)&layer->running_var);
    safe_free((void**)&layer->gamma_grad);
    safe_free((void**)&layer->beta_grad);
    safe_free((void**)&layer->gamma_m);
    safe_free((void**)&layer->gamma_v);
    safe_free((void**)&layer->beta_m);
    safe_free((void**)&layer->beta_v);
    safe_free((void**)&layer->saved_mean);
    safe_free((void**)&layer->saved_var);
    safe_free((void**)&layer->saved_std_inv);
    safe_free((void**)&layer->saved_normalized);
    safe_free((void**)&layer);
}

int quaternion_batchnorm_forward(QuaternionBatchNorm* layer,
                                  const float* input,
                                  size_t batch_size,
                                  float* output,
                                  int is_training)
{
    if (!layer || !input || !output || batch_size == 0) return -1;

    layer->is_training = is_training;
    size_t nf = layer->config.num_features;
    size_t qsize = nf * 4;
    float eps = layer->config.epsilon;
    float momentum = layer->config.momentum;

    if (is_training) {
        /* 分配归一化缓存（支持动态batch_size变化，防止缓冲区溢出） */
        size_t needed = batch_size * qsize;
        if (!layer->saved_normalized || layer->saved_normalized_capacity < needed) {
            safe_free((void**)&layer->saved_normalized);
            layer->saved_normalized = (float*)safe_calloc(needed, sizeof(float));
            if (!layer->saved_normalized) {
                layer->saved_normalized_capacity = 0;
                return -1;
            }
            layer->saved_normalized_capacity = needed;
        }

        /* 计算均值（对每个分量(c)在每个特征(f)上取batch平均） */
        memset(layer->saved_mean, 0, qsize * sizeof(float));
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < qsize; i++) {
                layer->saved_mean[i] += input[b * qsize + i];
            }
        }
        float inv_n = 1.0f / (float)batch_size;
        for (size_t i = 0; i < qsize; i++) {
            layer->saved_mean[i] *= inv_n;
        }

        /* 计算方差 */
        memset(layer->saved_var, 0, qsize * sizeof(float));
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < qsize; i++) {
                float diff = input[b * qsize + i] - layer->saved_mean[i];
                layer->saved_var[i] += diff * diff;
            }
        }
        for (size_t i = 0; i < qsize; i++) {
            layer->saved_var[i] *= inv_n;
        }

        /* 计算 1/sqrt(var + eps) */
        for (size_t i = 0; i < qsize; i++) {
            layer->saved_std_inv[i] = 1.0f / sqrtf(layer->saved_var[i] + eps);
        }

        /* 更新运行统计量 */
        for (size_t i = 0; i < qsize; i++) {
            layer->running_mean[i] = momentum * layer->running_mean[i]
                                     + (1.0f - momentum) * layer->saved_mean[i];
            layer->running_var[i] = momentum * layer->running_var[i]
                                    + (1.0f - momentum) * layer->saved_var[i];
        }

        /* 归一化和缩放/偏移 */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < qsize; i++) {
                float normalized = (input[b * qsize + i] - layer->saved_mean[i]) * layer->saved_std_inv[i];
                layer->saved_normalized[b * qsize + i] = normalized;
                float y = normalized;
                if (layer->config.use_scale) y *= layer->gamma[i];
                if (layer->config.use_shift) y += layer->beta[i];
                output[b * qsize + i] = y;
            }
        }
    } else {
        /* 推理模式：使用运行统计量 */
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < qsize; i++) {
                float running_std_inv = 1.0f / sqrtf(layer->running_var[i] + eps);
                float normalized = (input[b * qsize + i] - layer->running_mean[i]) * running_std_inv;
                float y = normalized;
                if (layer->config.use_scale) y *= layer->gamma[i];
                if (layer->config.use_shift) y += layer->beta[i];
                output[b * qsize + i] = y;
            }
        }
    }

    return 0;
}

/* S-010修复: 四元数协方差矩阵归一化
 * 计算w,x,y,z分量间的4×4协方差矩阵，通过Cholesky分解实现Mahalanobis白化
 * 替代独立分量假设，捕获四元数分量间的相关性 */
int quaternion_batchnorm_forward_covariance(QuaternionBatchNorm* layer,
                                             const float* input,
                                             size_t batch_size,
                                             float* output,
                                             int is_training)
{
    if (!layer || !input || !output || batch_size < 4) {
        return quaternion_batchnorm_forward(layer, input, batch_size, output, is_training);
    }

    size_t nf = layer->config.num_features;
    float eps = layer->config.epsilon;
    float momentum = layer->config.momentum;

    /* 首先执行标准逐分量归一化 */
    if (quaternion_batchnorm_forward(layer, input, batch_size, output, is_training) != 0)
        return -1;

    if (!is_training || !layer->saved_normalized) return 0;

    size_t qsize = nf * 4;

    /* 对每个四元数特征计算4×4协方差矩阵并白化 */
    for (size_t f = 0; f < nf; f++) {
        /* 收集该特征的batch中所有四元数分量 */
        float cov[16] = {0}; /* 4×4协方差矩阵 */
        size_t base = f * 4;

        for (size_t b = 0; b < batch_size; b++) {
            float w = layer->saved_normalized[b * qsize + base + 0] - layer->saved_mean[base + 0];
            float x = layer->saved_normalized[b * qsize + base + 1] - layer->saved_mean[base + 1];
            float y = layer->saved_normalized[b * qsize + base + 2] - layer->saved_mean[base + 2];
            float z = layer->saved_normalized[b * qsize + base + 3] - layer->saved_mean[base + 3];
            float vec[4] = {w, x, y, z};
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    cov[i * 4 + j] += vec[i] * vec[j];
        }

        float inv_n = 1.0f / (float)batch_size;
        for (int i = 0; i < 16; i++) cov[i] *= inv_n;

        /* Cholesky分解 L: cov = L * L^T */
        float L[16] = {0};
        for (int i = 0; i < 4; i++) {
            float diag = cov[i * 4 + i];
            for (int k = 0; k < i; k++)
                diag -= L[i * 4 + k] * L[i * 4 + k];
            if (diag <= 0.0f) diag = eps;
            L[i * 4 + i] = sqrtf(diag);

            for (int j = i + 1; j < 4; j++) {
                float sum = cov[j * 4 + i];
                for (int k = 0; k < i; k++)
                    sum -= L[j * 4 + k] * L[i * 4 + k];
                L[j * 4 + i] = sum / (L[i * 4 + i] + eps);
            }
        }

        /* L的逆矩阵 L_inv */
        float L_inv[16] = {0};
        for (int i = 0; i < 4; i++) {
            L_inv[i * 4 + i] = 1.0f / (L[i * 4 + i] + eps);
            for (int j = 0; j < i; j++) {
                float sum = 0.0f;
                for (int k = j; k < i; k++)
                    sum += L[i * 4 + k] * L_inv[k * 4 + j];
                L_inv[i * 4 + j] = -L_inv[i * 4 + i] * sum;
            }
        }

        /* Mahalanobis白化: y = L_inv^T * x */
        for (size_t b = 0; b < batch_size; b++) {
            float vec[4] = {
                layer->saved_normalized[b * qsize + base + 0] - layer->saved_mean[base + 0],
                layer->saved_normalized[b * qsize + base + 1] - layer->saved_mean[base + 1],
                layer->saved_normalized[b * qsize + base + 2] - layer->saved_mean[base + 2],
                layer->saved_normalized[b * qsize + base + 3] - layer->saved_mean[base + 3]
            };
            float whitened[4] = {0};
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    whitened[i] += L_inv[j * 4 + i] * vec[j];

            output[b * qsize + base + 0] = whitened[0];
            output[b * qsize + base + 1] = whitened[1];
            output[b * qsize + base + 2] = whitened[2];
            output[b * qsize + base + 3] = whitened[3];

            if (layer->config.use_scale) {
                output[b * qsize + base + 0] *= layer->gamma[base + 0];
                output[b * qsize + base + 1] *= layer->gamma[base + 1];
                output[b * qsize + base + 2] *= layer->gamma[base + 2];
                output[b * qsize + base + 3] *= layer->gamma[base + 3];
            }
            if (layer->config.use_shift) {
                output[b * qsize + base + 0] += layer->beta[base + 0];
                output[b * qsize + base + 1] += layer->beta[base + 1];
                output[b * qsize + base + 2] += layer->beta[base + 2];
                output[b * qsize + base + 3] += layer->beta[base + 3];
            }
        }
    }

    return 0;
}

int quaternion_batchnorm_backward(QuaternionBatchNorm* layer,
                                   const float* output_grad,
                                   const float* input,
                                   size_t batch_size,
                                   float* input_grad)
{
    if (!layer || !output_grad || !input || batch_size == 0 || !layer->saved_normalized) return -1;

    size_t nf = layer->config.num_features;
    size_t qsize = nf * 4;
    float inv_n = 1.0f / (float)batch_size;

    /* 清零梯度 */
    if (layer->config.use_scale) memset(layer->gamma_grad, 0, qsize * sizeof(float));
    if (layer->config.use_shift) memset(layer->beta_grad, 0, qsize * sizeof(float));

    /* 计算γ和β的梯度 */
    for (size_t i = 0; i < qsize; i++) {
        float gamma_grad = 0.0f;
        float beta_grad = 0.0f;
        for (size_t b = 0; b < batch_size; b++) {
            float dy = output_grad[b * qsize + i];
            float x_norm = layer->saved_normalized[b * qsize + i];
            if (layer->config.use_scale) gamma_grad += dy * x_norm;
            if (layer->config.use_shift) beta_grad += dy;
        }
        if (layer->config.use_scale) layer->gamma_grad[i] = gamma_grad;
        if (layer->config.use_shift) layer->beta_grad[i] = beta_grad;
    }

    /* 计算输入梯度 */
    if (input_grad) {
        /* 每通道预计算 dvar(∂L/∂σ²) 和 dmean(∂L/∂μ) */
        for (size_t i = 0; i < qsize; i++) {
            float std_inv = layer->saved_std_inv[i];
            float gamma_val = layer->config.use_scale ? layer->gamma[i] : 1.0f;

            float dvar = 0.0f;
            float dmean_first = 0.0f;
            for (size_t b = 0; b < batch_size; b++) {
                float dy = output_grad[b * qsize + i];
                float dx_norm = dy * gamma_val;
                float diff = input[b * qsize + i] - layer->saved_mean[i];

                dvar += dx_norm * diff * (-0.5f) * std_inv * std_inv * std_inv;
                dmean_first += dx_norm * (-std_inv);
            }

            /* dmean第二项: dvar * Σ(-2*(x_b-μ)/N) = dvar * 0 = 0，直接省略以避免数值噪声 */
            float dmean = dmean_first;

            for (size_t b = 0; b < batch_size; b++) {
                float dy = output_grad[b * qsize + i];
                float dx_norm = dy * gamma_val;
                float diff = input[b * qsize + i] - layer->saved_mean[i];

                input_grad[b * qsize + i] = dx_norm * std_inv
                                           + 2.0f * dvar * diff * inv_n
                                           + dmean * inv_n;
            }
        }
    }

    /* Adam参数更新 */
    layer->adam_step++;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float corr1 = 1.0f - powf(beta1, (float)layer->adam_step);
    float corr2 = 1.0f - powf(beta2, (float)layer->adam_step);
    float lr = layer->config.lr;

    if (layer->config.use_scale) {
        for (size_t i = 0; i < qsize; i++) {
            adam_update(&layer->gamma[i], &layer->gamma_m[i], &layer->gamma_v[i],
                        layer->gamma_grad[i], lr, beta1, beta2, eps, corr1, corr2);
        }
    }
    if (layer->config.use_shift) {
        for (size_t i = 0; i < qsize; i++) {
            adam_update(&layer->beta[i], &layer->beta_m[i], &layer->beta_v[i],
                        layer->beta_grad[i], lr, beta1, beta2, eps, corr1, corr2);
        }
    }

    return 0;
}

void quaternion_batchnorm_eval(QuaternionBatchNorm* layer)
{
    if (layer) layer->is_training = 0;
}

void quaternion_batchnorm_train(QuaternionBatchNorm* layer)
{
    if (layer) layer->is_training = 1;
}

void quaternion_batchnorm_reset_stats(QuaternionBatchNorm* layer)
{
    if (!layer) return;
    size_t qsize = layer->config.num_features * 4;
    memset(layer->running_mean, 0, qsize * sizeof(float));
    for (size_t i = 0; i < qsize; i++) layer->running_var[i] = 1.0f;
    memset(layer->saved_mean, 0, qsize * sizeof(float));
    memset(layer->saved_var, 0, qsize * sizeof(float));
    memset(layer->saved_std_inv, 0, qsize * sizeof(float));
    safe_free((void**)&layer->saved_normalized);
}

void quaternion_batchnorm_reset_optimizer(QuaternionBatchNorm* layer)
{
    if (!layer) return;
    size_t qsize = layer->config.num_features * 4;
    if (layer->config.use_scale) {
        memset(layer->gamma_m, 0, qsize * sizeof(float));
        memset(layer->gamma_v, 0, qsize * sizeof(float));
    }
    if (layer->config.use_shift) {
        memset(layer->beta_m, 0, qsize * sizeof(float));
        memset(layer->beta_v, 0, qsize * sizeof(float));
    }
    layer->adam_step = 0;
}

void quaternion_batchnorm_set_lr(QuaternionBatchNorm* layer, float lr)
{
    if (layer) layer->config.lr = lr;
}

int quaternion_batchnorm_param_count(const QuaternionBatchNorm* layer)
{
    if (!layer) return -1;
    int count = 0;
    size_t nf = layer->config.num_features * 4;
    if (layer->config.use_scale) count += (int)nf;
    if (layer->config.use_shift) count += (int)nf;
    return count;
}

int quaternion_batchnorm_save(const QuaternionBatchNorm* layer, const char* filename)
{
    if (!layer || !filename) return -1;
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    fwrite(&layer->config, sizeof(QuaternionBatchNormConfig), 1, f);
    size_t qsize = layer->config.num_features * 4;

    fwrite(layer->gamma, sizeof(float), layer->config.use_scale ? qsize : 1, f);
    fwrite(layer->beta, sizeof(float), layer->config.use_shift ? qsize : 1, f);
    fwrite(layer->running_mean, sizeof(float), qsize, f);
    fwrite(layer->running_var, sizeof(float), qsize, f);

    fclose(f);
    return 0;
}

QuaternionBatchNorm* quaternion_batchnorm_load(const char* filename)
{
    if (!filename) return NULL;
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    QuaternionBatchNormConfig cfg;
    if (fread(&cfg, sizeof(QuaternionBatchNormConfig), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    QuaternionBatchNorm* layer = quaternion_batchnorm_create(&cfg);
    if (!layer) { fclose(f); return NULL; }

    size_t qsize = cfg.num_features * 4;
    size_t br_gamma = fread(layer->gamma, sizeof(float), cfg.use_scale ? qsize : 1, f);
    size_t br_beta = fread(layer->beta, sizeof(float), cfg.use_shift ? qsize : 1, f);
    size_t br_mean = fread(layer->running_mean, sizeof(float), qsize, f);
    size_t br_var = fread(layer->running_var, sizeof(float), qsize, f);

    fclose(f);

    if (br_gamma != (cfg.use_scale ? qsize : 1) || br_beta != (cfg.use_shift ? qsize : 1) ||
        br_mean != qsize || br_var != qsize) {
        quaternion_batchnorm_free(layer);
        return NULL;
    }
    return layer;
}
