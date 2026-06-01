#include "selflnn/core/optimizer.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* C-014修复: 梯度裁剪默认阈值从魔法数字抽出为命名常量
 * 当全局梯度的最大绝对值超过此阈值时，按比例缩放所有梯度
 * 防止梯度爆炸导致训练不稳定 */
#define OPTIMIZER_DEFAULT_GRAD_CLIP_NORM 10.0f

struct Optimizer
{
    OptimizerConfig config;
    float* momentum_buffer;
    float* velocity_buffer;
    float* cache_buffer;
    float* beta1_power;
    float* beta2_power;
    size_t num_params;
    int is_initialized;
    int last_error;              /**< 最后一次错误的错误码，0=无错误 */
};

Optimizer* optimizer_create(const OptimizerConfig* config)
{
    if (!config) return NULL;

    Optimizer* opt = (Optimizer*)safe_calloc(1, sizeof(Optimizer));
    if (!opt) return NULL;

    opt->config = *config;
    opt->is_initialized = 0;

    return opt;
}

void optimizer_free(Optimizer* optimizer)
{
    if (!optimizer) return;

    safe_free((void**)&optimizer->momentum_buffer);
    safe_free((void**)&optimizer->velocity_buffer);
    safe_free((void**)&optimizer->cache_buffer);
    safe_free((void**)&optimizer->beta1_power);
    safe_free((void**)&optimizer->beta2_power);
    safe_free((void**)&optimizer);
}

static int ensure_buffers(Optimizer* optimizer, size_t num_params)
{
    if (!optimizer) return -1;

    if (optimizer->num_params == num_params && optimizer->is_initialized) return 0;

    safe_free((void**)&optimizer->momentum_buffer);
    safe_free((void**)&optimizer->velocity_buffer);
    safe_free((void**)&optimizer->cache_buffer);
    safe_free((void**)&optimizer->beta1_power);
    safe_free((void**)&optimizer->beta2_power);

    optimizer->num_params = num_params;

    switch (optimizer->config.type)
    {
        case OPTIMIZER_SGD:
            break;

        case OPTIMIZER_MOMENTUM:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            if (!optimizer->momentum_buffer) return -1;
            break;

        case OPTIMIZER_ADAGRAD:
            optimizer->cache_buffer = (float*)safe_calloc(num_params, sizeof(float));
            if (!optimizer->cache_buffer) return -1;
            break;

        case OPTIMIZER_RMSPROP:
            optimizer->cache_buffer = (float*)safe_calloc(num_params, sizeof(float));
            if (!optimizer->cache_buffer) return -1;
            break;

        case OPTIMIZER_ADAM:
        case OPTIMIZER_ADAMW:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->beta1_power = (float*)safe_calloc(1, sizeof(float));
            optimizer->beta2_power = (float*)safe_calloc(1, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer ||
                !optimizer->beta1_power || !optimizer->beta2_power) return -1;
            *optimizer->beta1_power = 1.0f;
            *optimizer->beta2_power = 1.0f;
            break;

        case OPTIMIZER_ADADELTA:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer) return -1;
            break;

        case OPTIMIZER_LAMB:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->beta1_power = (float*)safe_calloc(1, sizeof(float));
            optimizer->beta2_power = (float*)safe_calloc(1, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer ||
                !optimizer->beta1_power || !optimizer->beta2_power) return -1;
            *optimizer->beta1_power = 1.0f;
            *optimizer->beta2_power = 1.0f;
            break;

        case OPTIMIZER_LARS:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            if (!optimizer->momentum_buffer) return -1;
            break;

        case OPTIMIZER_RANGER:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->cache_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->beta1_power = (float*)safe_calloc(1, sizeof(float));
            optimizer->beta2_power = (float*)safe_calloc(1, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer ||
                !optimizer->cache_buffer || !optimizer->beta1_power || !optimizer->beta2_power) return -1;
            *optimizer->beta1_power = 1.0f;
            *optimizer->beta2_power = 1.0f;
            break;

        case OPTIMIZER_NOVOGRAD:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->beta1_power = (float*)safe_calloc(1, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer ||
                !optimizer->beta1_power) return -1;
            *optimizer->beta1_power = 1.0f;
            break;

        default:
            return -1;
    }

    optimizer->is_initialized = 1;
    return 0;
}

int optimizer_step(Optimizer* optimizer, float* parameters, float* gradients,
                   size_t num_params, size_t step)
{
    if (!optimizer || !parameters || !gradients || num_params == 0) return -1;

    if (ensure_buffers(optimizer, num_params) != 0) return -1;
    
    /* R2-P9修复: 逐元素NaN/Inf梯度过滤，替代全步丢弃
     * 原实现发现单个NaN梯度就丢弃整步更新，导致偶发NaN时所有参数停止学习。
     * 修正：将NaN/Inf梯度清零，正常参数继续更新。
 *: gradients改为非const以支持原地过滤 */
    {
        float max_abs_g = 0.0f;
        for (size_t k = 0; k < num_params; k++) {
            float g = gradients[k];
            if (!isfinite(g)) {
                gradients[k] = 0.0f;
            } else {
                float abs_g = fabsf(g);
                if (abs_g > max_abs_g) max_abs_g = abs_g;
            }
        }
        if (max_abs_g > OPTIMIZER_DEFAULT_GRAD_CLIP_NORM) {
            float clip_ratio = OPTIMIZER_DEFAULT_GRAD_CLIP_NORM / max_abs_g;
            for (size_t k = 0; k < num_params; k++) {
                gradients[k] *= clip_ratio;
            }
        }
    }
    
    /* P2-004修复：(void)step仅限于非RANGER优化器分支，
       RANGER使用step进行预热调度，需保留该参数 */ 
    if (optimizer->config.type != OPTIMIZER_RANGER) {
        (void)step;
    }

    float lr = optimizer->config.learning_rate;
    float eps = optimizer->config.epsilon > 0.0f ? optimizer->config.epsilon : 1e-8f;
    size_t i;

    switch (optimizer->config.type)
    {
        case OPTIMIZER_SGD:
        {
            float wd = optimizer->config.weight_decay;
            for (i = 0; i < num_params; i++)
            {
                /* FIX-005: 解耦权重衰减，与AdamW一致 */
                parameters[i] *= (1.0f - lr * wd);
                parameters[i] -= lr * gradients[i];
            }
            break;
        }

        case OPTIMIZER_MOMENTUM:
        {
            float mu = optimizer->config.momentum > 0.0f ? optimizer->config.momentum : 0.9f;
            float wd = optimizer->config.weight_decay;
            for (i = 0; i < num_params; i++)
            {
                /* FIX-001: 解耦权重衰减：直接衰减参数，梯度不包含权重衰减项 */
                parameters[i] *= (1.0f - lr * wd);
                float v_old = optimizer->momentum_buffer[i];
                optimizer->momentum_buffer[i] = mu * v_old - lr * gradients[i];
                if (optimizer->config.use_nesterov)
                {
                    /* 正确Nesterov：v = μ·v_old - lr·g; θ += v + μ·(v - v_old) */
                    parameters[i] += optimizer->momentum_buffer[i] +
                                     mu * (optimizer->momentum_buffer[i] - v_old);
                }
                else
                {
                    parameters[i] += optimizer->momentum_buffer[i];
                }
            }
            break;
        }

        case OPTIMIZER_ADAGRAD:
        {
            float wd = optimizer->config.weight_decay;
            for (i = 0; i < num_params; i++)
            {
                /* FIX-005: 解耦权重衰减 */
                parameters[i] *= (1.0f - lr * wd);
                optimizer->cache_buffer[i] += gradients[i] * gradients[i];
                float adjusted_lr = lr / (sqrtf(optimizer->cache_buffer[i]) + eps);
                parameters[i] -= adjusted_lr * gradients[i];
            }
            break;
        }

        case OPTIMIZER_RMSPROP:
        {
            float decay_rate = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.9f;
            float wd = optimizer->config.weight_decay;
            for (i = 0; i < num_params; i++)
            {
/* 添加解耦权重衰减，与其他优化器一致 */
                parameters[i] *= (1.0f - lr * wd);
                optimizer->cache_buffer[i] = decay_rate * optimizer->cache_buffer[i] +
                                             (1.0f - decay_rate) * gradients[i] * gradients[i];
                float adjusted_lr = lr / (sqrtf(optimizer->cache_buffer[i]) + eps);
                parameters[i] -= adjusted_lr * gradients[i];
            }
            break;
        }

        case OPTIMIZER_ADAM:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            float wd = optimizer->config.weight_decay;
            *optimizer->beta1_power *= b1;
            *optimizer->beta2_power *= b2;

            for (i = 0; i < num_params; i++)
            {
/* 添加解耦权重衰减，与其他优化器一致 */
                parameters[i] *= (1.0f - lr * wd);
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * gradients[i];
                optimizer->velocity_buffer[i] = b2 * optimizer->velocity_buffer[i] +
                                                (1.0f - b2) * gradients[i] * gradients[i];

                float m_hat = optimizer->momentum_buffer[i] / (1.0f - *optimizer->beta1_power);
                float v_hat = optimizer->velocity_buffer[i] / (1.0f - *optimizer->beta2_power);

                parameters[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
            }
            break;
        }

        case OPTIMIZER_ADAMW:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            float wd = optimizer->config.weight_decay;
            *optimizer->beta1_power *= b1;
            *optimizer->beta2_power *= b2;

            for (i = 0; i < num_params; i++)
            {
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * gradients[i];
                optimizer->velocity_buffer[i] = b2 * optimizer->velocity_buffer[i] +
                                                (1.0f - b2) * gradients[i] * gradients[i];

                float m_hat = optimizer->momentum_buffer[i] / (1.0f - *optimizer->beta1_power);
                float v_hat = optimizer->velocity_buffer[i] / (1.0f - *optimizer->beta2_power);

                parameters[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * parameters[i]);
            }
            break;
        }

        case OPTIMIZER_ADADELTA:
        {
            float rho = optimizer->config.momentum > 0.0f ? optimizer->config.momentum : 0.95f;
            float wd = optimizer->config.weight_decay;
            for (i = 0; i < num_params; i++)
            {
/* 添加解耦权重衰减，与其他优化器一致 */
                parameters[i] *= (1.0f - lr * wd);
                optimizer->velocity_buffer[i] = rho * optimizer->velocity_buffer[i] +
                                                (1.0f - rho) * gradients[i] * gradients[i];

                float rms_g = sqrtf(optimizer->velocity_buffer[i] + eps);
                float rms_p = sqrtf(optimizer->momentum_buffer[i] + eps);

                float delta = (rms_p / rms_g) * gradients[i];

                optimizer->momentum_buffer[i] = rho * optimizer->momentum_buffer[i] +
                                                (1.0f - rho) * delta * delta;

                parameters[i] -= delta;
            }
            break;
        }

        case OPTIMIZER_LAMB:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            float trust = optimizer->config.lars_trust_coef > 0.0f ? optimizer->config.lars_trust_coef : 0.001f;
            float wd = optimizer->config.weight_decay;
            *optimizer->beta1_power *= b1;
            *optimizer->beta2_power *= b2;

/* LAMB优化器状态破坏修复。
             * velocity_buffer存储的是梯度平方的指数移动平均(v_t = β2*v_{t-1} + (1-β2)*g²)，
             * 第一遍循环中计算完v_hat后不能覆写velocity_buffer，否则下一次step时
             * EMA输入错误。改为分配临时update_buffer存储中间结果。 */
            float* update_tmp = (float*)safe_malloc((size_t)num_params * sizeof(float));
            if (!update_tmp) { optimizer->last_error = -1; return -1; }

            float param_norm = 0.0f, update_norm = 0.0f;
            float inv_b1 = 1.0f / (1.0f - *optimizer->beta1_power);
            float inv_b2 = 1.0f / (1.0f - *optimizer->beta2_power);
            for (i = 0; i < num_params; i++)
            {
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * gradients[i];
                optimizer->velocity_buffer[i] = b2 * optimizer->velocity_buffer[i] +
                                                (1.0f - b2) * gradients[i] * gradients[i];

                float m_hat = optimizer->momentum_buffer[i] * inv_b1;
                float v_hat = optimizer->velocity_buffer[i] * inv_b2;

                float update = m_hat / (sqrtf(v_hat) + eps) + wd * parameters[i];
                update_tmp[i] = update;
                param_norm += parameters[i] * parameters[i];
                update_norm += update * update;
            }
            param_norm = sqrtf(param_norm + 1e-12f);
            update_norm = sqrtf(update_norm + 1e-12f);
            float trust_ratio = (param_norm > 0.0f && update_norm > 0.0f)
                                ? trust * param_norm / update_norm : 1.0f;

            for (i = 0; i < num_params; i++)
            {
                parameters[i] -= lr * trust_ratio * update_tmp[i];
            }
            safe_free((void**)&update_tmp);
            break;
        }

        case OPTIMIZER_LARS:
        {
            float trust = optimizer->config.lars_trust_coef > 0.0f ? optimizer->config.lars_trust_coef : 0.001f;
            float wd = optimizer->config.weight_decay;
            float mu = optimizer->config.momentum > 0.0f ? optimizer->config.momentum : 0.9f;

            float param_norm = 0.0f, grad_norm = 0.0f;
            for (i = 0; i < num_params; i++)
            {
                param_norm += parameters[i] * parameters[i];
                float g = gradients[i] + wd * parameters[i];
                grad_norm += g * g;
            }
            param_norm = sqrtf(param_norm + 1e-12f);
            grad_norm = sqrtf(grad_norm + 1e-12f);
            float local_lr = (param_norm > 0.0f && grad_norm > 0.0f)
                             ? trust * param_norm / grad_norm : 1.0f;
            local_lr *= lr;

            for (i = 0; i < num_params; i++)
            {
                float g = gradients[i] + wd * parameters[i];
                optimizer->momentum_buffer[i] = mu * optimizer->momentum_buffer[i] -
                                                local_lr * g;
                parameters[i] += optimizer->momentum_buffer[i];
            }
            break;
        }

        case OPTIMIZER_RANGER:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            float wd = optimizer->config.weight_decay;
            *optimizer->beta1_power *= b1;
            *optimizer->beta2_power *= b2;

            /* RAdam整流因子计算 */
            float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
            float beta2_t = *optimizer->beta2_power;
            float rho_t = rho_inf - 2.0f * (float)step * beta2_t / (1.0f - beta2_t + 1e-12f);

            /* FIX-002: LookAhead慢权重初始化（首次调用时快照当前参数为慢权重） */
            int k = optimizer->config.lookahead_k > 0 ? optimizer->config.lookahead_k : 5;
            float alpha = optimizer->config.lookahead_alpha > 0.0f
                          ? optimizer->config.lookahead_alpha : 0.5f;
            int is_first_step = (step == 0);
            if (is_first_step) {
                for (i = 0; i < num_params; i++) {
                    optimizer->cache_buffer[i] = parameters[i];
                }
            }

            for (i = 0; i < num_params; i++)
            {
                /* FIX-003: 解耦权重衰减，梯度仅含原始梯度 */
                parameters[i] *= (1.0f - lr * wd);
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * gradients[i];
                optimizer->velocity_buffer[i] = b2 * optimizer->velocity_buffer[i] +
                                                (1.0f - b2) * gradients[i] * gradients[i];

                float m_hat = optimizer->momentum_buffer[i] / (1.0f - *optimizer->beta1_power);
                float v_hat = optimizer->velocity_buffer[i] / (1.0f - *optimizer->beta2_power);

                if (rho_t > 4.0f)
                {
                    float l_t = sqrtf(1.0f - *optimizer->beta2_power) / (sqrtf(v_hat) + eps);
                    float r_num = (rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf;
                    float r_den = (rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t;
                    float r_ratio = r_num / (r_den + 1e-12f);
                    float r_t = sqrtf(r_ratio > 0.0f ? r_ratio : 0.0f);
                    parameters[i] -= lr * r_t * m_hat * l_t;
                }
                else
                {
                    parameters[i] -= lr * m_hat;
                }
                /* 不在每步覆盖慢权重，慢权重仅在同步时更新 */
            }

            /* FIX-002: LookAhead同步：每k步将快权重向慢权重方向移动 */
            if ((step + 1) % k == 0)
            {
                for (i = 0; i < num_params; i++)
                {
                    float slow = optimizer->cache_buffer[i];
                    float fast = parameters[i];
                    float new_slow = slow + alpha * (fast - slow);
                    optimizer->cache_buffer[i] = new_slow;
                    parameters[i] = new_slow;
                }
            }
            break;
        }

        case OPTIMIZER_NOVOGRAD:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            float wd = optimizer->config.weight_decay;
            *optimizer->beta1_power *= b1;

/* 添加解耦权重衰减，与其他优化器一致 */
            for (i = 0; i < num_params; i++) {
                parameters[i] *= (1.0f - lr * wd);
            }

            /* 全局梯度L2范数平方（分层级v） */
            float grad_l2_sq = 0.0f;
            for (i = 0; i < num_params; i++) {
                grad_l2_sq += gradients[i] * gradients[i];
            }
            float grad_norm_v = grad_l2_sq / (float)num_params;
            float v = optimizer->velocity_buffer[0];
            v = b2 * v + (1.0f - b2) * grad_norm_v;
            optimizer->velocity_buffer[0] = v;

            float v_sqrt = sqrtf(v + eps);
            for (i = 0; i < num_params; i++)
            {
                float g_normed = gradients[i] / v_sqrt;
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * g_normed;
                float m_hat = optimizer->momentum_buffer[i] / (1.0f - *optimizer->beta1_power);
                parameters[i] -= lr * m_hat;
            }
            break;
        }

        default:
            return -1;
    }

    return 0;
}

/* 多参数组更新——一次更新多组独立参数（权重/偏置/门控/时间常数等） */
int optimizer_update_multi_group(Optimizer* optimizer, OptimizerParamGroup* groups,
                                  int num_groups, size_t step)
{
    if (!optimizer || !groups || num_groups <= 0) return -1;

    /* 计算总参数数并验证各组指针有效性 */
    size_t total_params = 0;
    for (int g = 0; g < num_groups; g++) {
        if (!groups[g].parameters || !groups[g].gradients || groups[g].num_params == 0) {
            return -1;
        }
        total_params += groups[g].num_params;
    }
    if (total_params == 0) return -1;

    /* 确保优化器内部缓冲区足够大 */
    if (ensure_buffers(optimizer, total_params) != 0) return -1;

    if (optimizer->config.type != OPTIMIZER_RANGER) {
        (void)step;
    }

    float lr = optimizer->config.learning_rate;
    float eps = optimizer->config.epsilon > 0.0f ? optimizer->config.epsilon : 1e-8f;

    /* 使用全局偏移idx在跨组平坦缓冲区上操作 */
    size_t idx = 0;

    switch (optimizer->config.type)
    {
        case OPTIMIZER_SGD:
        {
            float wd = optimizer->config.weight_decay;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++) {
                    params[i] *= (1.0f - lr * wd);
                    params[i] -= lr * grads[i];
                }
            }
            break;
        }

        case OPTIMIZER_MOMENTUM:
        {
            float mu = optimizer->config.momentum > 0.0f ? optimizer->config.momentum : 0.9f;
            float wd = optimizer->config.weight_decay;
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    params[i] *= (1.0f - lr * wd);
                    float v_old = optimizer->momentum_buffer[idx];
                    optimizer->momentum_buffer[idx] = mu * v_old - lr * grads[i];
                    if (optimizer->config.use_nesterov) {
                        params[i] += optimizer->momentum_buffer[idx] +
                                     mu * (optimizer->momentum_buffer[idx] - v_old);
                    } else {
                        params[i] += optimizer->momentum_buffer[idx];
                    }
                }
            }
            break;
        }

        case OPTIMIZER_ADAGRAD:
        {
            float wd = optimizer->config.weight_decay;
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    params[i] *= (1.0f - lr * wd);
                    optimizer->cache_buffer[idx] += grads[i] * grads[i];
                    float adjusted_lr = lr / (sqrtf(optimizer->cache_buffer[idx]) + eps);
                    params[i] -= adjusted_lr * grads[i];
                }
            }
            break;
        }

        case OPTIMIZER_RMSPROP:
        {
            float decay_rate = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.9f;
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    optimizer->cache_buffer[idx] = decay_rate * optimizer->cache_buffer[idx] +
                                                   (1.0f - decay_rate) * grads[i] * grads[i];
                    float adjusted_lr = lr / (sqrtf(optimizer->cache_buffer[idx]) + eps);
                    params[i] -= adjusted_lr * grads[i];
                }
            }
            break;
        }

        case OPTIMIZER_ADAM:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            *optimizer->beta1_power *= b1;
            *optimizer->beta2_power *= b2;
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    optimizer->momentum_buffer[idx] = b1 * optimizer->momentum_buffer[idx] +
                                                      (1.0f - b1) * grads[i];
                    optimizer->velocity_buffer[idx] = b2 * optimizer->velocity_buffer[idx] +
                                                      (1.0f - b2) * grads[i] * grads[i];
                    float m_hat = optimizer->momentum_buffer[idx] / (1.0f - *optimizer->beta1_power);
                    float v_hat = optimizer->velocity_buffer[idx] / (1.0f - *optimizer->beta2_power);
                    params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
                }
            }
            break;
        }

        case OPTIMIZER_ADAMW:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            float wd = optimizer->config.weight_decay;
            *optimizer->beta1_power *= b1;
            *optimizer->beta2_power *= b2;
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    optimizer->momentum_buffer[idx] = b1 * optimizer->momentum_buffer[idx] +
                                                      (1.0f - b1) * grads[i];
                    optimizer->velocity_buffer[idx] = b2 * optimizer->velocity_buffer[idx] +
                                                      (1.0f - b2) * grads[i] * grads[i];
                    float m_hat = optimizer->momentum_buffer[idx] / (1.0f - *optimizer->beta1_power);
                    float v_hat = optimizer->velocity_buffer[idx] / (1.0f - *optimizer->beta2_power);
                    params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * params[i]);
                }
            }
            break;
        }

        case OPTIMIZER_ADADELTA:
        {
            float rho = optimizer->config.momentum > 0.0f ? optimizer->config.momentum : 0.95f;
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    optimizer->velocity_buffer[idx] = rho * optimizer->velocity_buffer[idx] +
                                                      (1.0f - rho) * grads[i] * grads[i];
                    float rms_g = sqrtf(optimizer->velocity_buffer[idx] + eps);
                    float rms_p = sqrtf(optimizer->momentum_buffer[idx] + eps);
                    float delta = (rms_p / rms_g) * grads[i];
                    optimizer->momentum_buffer[idx] = rho * optimizer->momentum_buffer[idx] +
                                                      (1.0f - rho) * delta * delta;
                    params[i] -= delta;
                }
            }
            break;
        }

        case OPTIMIZER_LAMB:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            float trust = optimizer->config.lars_trust_coef > 0.0f ? optimizer->config.lars_trust_coef : 0.001f;
            float wd = optimizer->config.weight_decay;
            *optimizer->beta1_power *= b1;
            *optimizer->beta2_power *= b2;
            float inv_b1 = 1.0f / (1.0f - *optimizer->beta1_power);
            float inv_b2 = 1.0f / (1.0f - *optimizer->beta2_power);

/* 多组LAMB状态保护。分配临时update缓冲区 */
/* total_params已在函数开头计算，此处复用不重新声明 */
            float* update_tmp = (float*)safe_malloc(total_params * sizeof(float));
            if (!update_tmp) { optimizer->last_error = -1; return -1; }

            /* 第一遍：计算动量/速度和范数 */
            float param_norm = 0.0f, update_norm = 0.0f;
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    optimizer->momentum_buffer[idx] = b1 * optimizer->momentum_buffer[idx] +
                                                      (1.0f - b1) * grads[i];
                    optimizer->velocity_buffer[idx] = b2 * optimizer->velocity_buffer[idx] +
                                                      (1.0f - b2) * grads[i] * grads[i];
                    float m_hat = optimizer->momentum_buffer[idx] * inv_b1;
                    float v_hat = optimizer->velocity_buffer[idx] * inv_b2;
                    float update = m_hat / (sqrtf(v_hat) + eps) + wd * params[i];
                    update_tmp[idx] = update;
                    param_norm += params[i] * params[i];
                    update_norm += update * update;
                }
            }
            param_norm = sqrtf(param_norm + 1e-12f);
            update_norm = sqrtf(update_norm + 1e-12f);
            float trust_ratio = (param_norm > 0.0f && update_norm > 0.0f)
                                ? trust * param_norm / update_norm : 1.0f;

            /* 第二遍：应用信任比缩放后的更新 */
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    params[i] -= lr * trust_ratio * update_tmp[idx];
                }
            }
            safe_free((void**)&update_tmp);
            break;
        }

        case OPTIMIZER_LARS:
        {
            float trust = optimizer->config.lars_trust_coef > 0.0f ? optimizer->config.lars_trust_coef : 0.001f;
            float wd = optimizer->config.weight_decay;
            float mu = optimizer->config.momentum > 0.0f ? optimizer->config.momentum : 0.9f;

            /* 计算全局范数 */
            float param_norm = 0.0f, grad_norm = 0.0f;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++) {
                    param_norm += params[i] * params[i];
                    float gv = grads[i] + wd * params[i];
                    grad_norm += gv * gv;
                }
            }
            param_norm = sqrtf(param_norm + 1e-12f);
            grad_norm = sqrtf(grad_norm + 1e-12f);
            float local_lr = (param_norm > 0.0f && grad_norm > 0.0f)
                             ? trust * param_norm / grad_norm : 1.0f;
            local_lr *= lr;

            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    float gv = grads[i] + wd * params[i];
                    optimizer->momentum_buffer[idx] = mu * optimizer->momentum_buffer[idx] -
                                                      local_lr * gv;
                    params[i] += optimizer->momentum_buffer[idx];
                }
            }
            break;
        }

        case OPTIMIZER_RANGER:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            float wd = optimizer->config.weight_decay;
            *optimizer->beta1_power *= b1;
            *optimizer->beta2_power *= b2;

            float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
            float beta2_t = *optimizer->beta2_power;
            float rho_t = rho_inf - 2.0f * (float)step * beta2_t / (1.0f - beta2_t + 1e-12f);

            int k = optimizer->config.lookahead_k > 0 ? optimizer->config.lookahead_k : 5;
            float alpha = optimizer->config.lookahead_alpha > 0.0f
                          ? optimizer->config.lookahead_alpha : 0.5f;

            /* LookAhead慢权重初始化（首次调用时快照当前参数） */
            if (step == 0) {
                idx = 0;
                for (int g = 0; g < num_groups; g++) {
                    float* params = groups[g].parameters;
                    size_t n = groups[g].num_params;
                    for (size_t i = 0; i < n; i++, idx++) {
                        optimizer->cache_buffer[idx] = params[i];
                    }
                }
            }

            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    params[i] *= (1.0f - lr * wd);
                    optimizer->momentum_buffer[idx] = b1 * optimizer->momentum_buffer[idx] +
                                                      (1.0f - b1) * grads[i];
                    optimizer->velocity_buffer[idx] = b2 * optimizer->velocity_buffer[idx] +
                                                      (1.0f - b2) * grads[i] * grads[i];
                    float m_hat = optimizer->momentum_buffer[idx] / (1.0f - *optimizer->beta1_power);
                    float v_hat = optimizer->velocity_buffer[idx] / (1.0f - *optimizer->beta2_power);
                    if (rho_t > 4.0f) {
                        float l_t = sqrtf(1.0f - *optimizer->beta2_power) / (sqrtf(v_hat) + eps);
                        float r_num = (rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf;
                        float r_den = (rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t;
                        float r_ratio = r_num / (r_den + 1e-12f);
                        float r_t = sqrtf(r_ratio > 0.0f ? r_ratio : 0.0f);
                        params[i] -= lr * r_t * m_hat * l_t;
                    } else {
                        params[i] -= lr * m_hat;
                    }
                }
            }

            /* LookAhead同步 */
            if ((step + 1) % k == 0) {
                idx = 0;
                for (int g = 0; g < num_groups; g++) {
                    float* params = groups[g].parameters;
                    size_t n = groups[g].num_params;
                    for (size_t i = 0; i < n; i++, idx++) {
                        float slow = optimizer->cache_buffer[idx];
                        float fast = params[i];
                        float new_slow = slow + alpha * (fast - slow);
                        optimizer->cache_buffer[idx] = new_slow;
                        params[i] = new_slow;
                    }
                }
            }
            break;
        }

        case OPTIMIZER_NOVOGRAD:
        {
            float b1 = optimizer->config.beta1 > 0.0f ? optimizer->config.beta1 : 0.9f;
            float b2 = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.999f;
            *optimizer->beta1_power *= b1;

            /* 全局梯度L2范数平方 */
            float grad_l2_sq = 0.0f;
            for (int g = 0; g < num_groups; g++) {
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++) {
                    grad_l2_sq += grads[i] * grads[i];
                }
            }
            float grad_norm_v = grad_l2_sq / (float)total_params;
            float v = optimizer->velocity_buffer[0];
            v = b2 * v + (1.0f - b2) * grad_norm_v;
            optimizer->velocity_buffer[0] = v;

            float v_sqrt = sqrtf(v + eps);
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                for (size_t i = 0; i < n; i++, idx++) {
                    float g_normed = grads[i] / v_sqrt;
                    optimizer->momentum_buffer[idx] = b1 * optimizer->momentum_buffer[idx] +
                                                      (1.0f - b1) * g_normed;
                    float m_hat = optimizer->momentum_buffer[idx] / (1.0f - *optimizer->beta1_power);
                    params[i] -= lr * m_hat;
                }
            }
            break;
        }

        default:
            return -1;
    }

    return 0;
}

void optimizer_reset(Optimizer* optimizer)
{
    if (!optimizer) return;

    if (optimizer->momentum_buffer)
    {
        memset(optimizer->momentum_buffer, 0, optimizer->num_params * sizeof(float));
    }
    if (optimizer->velocity_buffer)
    {
        memset(optimizer->velocity_buffer, 0, optimizer->num_params * sizeof(float));
    }
    if (optimizer->cache_buffer)
    {
        memset(optimizer->cache_buffer, 0, optimizer->num_params * sizeof(float));
    }
    if (optimizer->beta1_power) *optimizer->beta1_power = 1.0f;
    if (optimizer->beta2_power) *optimizer->beta2_power = 1.0f;
    optimizer->is_initialized = 0;
}

int optimizer_get_config(const Optimizer* optimizer, OptimizerConfig* config)
{
    if (!optimizer || !config) return -1;
    *config = optimizer->config;
    return 0;
}

int optimizer_set_learning_rate(Optimizer* optimizer, float learning_rate)
{
    if (!optimizer) return -1;
    optimizer->config.learning_rate = learning_rate;
    return 0;
}

/* CORE-24: AdamW (Weight Decay解耦) */
int optimizer_adamw_step(float* params, float* grads, float* m, float* v, size_t n, float lr, float beta1, float beta2, float eps, float weight_decay, int step) {
    if (!params || !grads || !m || !v) return -1;
    float b1_corr = 1.0f - powf(beta1, (float)step);
    float b2_corr = 1.0f - powf(beta2, (float)step);
/* 添加逐元素NaN/Inf保护，
     * 防止单次NaN污染动量缓冲区m和v导致永久训练损坏 */
    for (size_t i = 0; i < n; i++) {
        float g = grads[i];
        if (!isfinite(g)) g = 0.0f;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] / b1_corr, v_hat = v[i] / b2_corr;
        params[i] = params[i] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
    }
    return 0;
}

/* TRAIN-21: Cosine Annealing Warm Restarts */
float lr_cosine_annealing(float base_lr, float min_lr, int epoch, int T_0, int T_mult) {
    int T_cur = epoch, T_i = T_0;
    while (T_cur >= T_i) { T_cur -= T_i; T_i *= T_mult; if (T_i == 0) T_i = T_0; }
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf((float)M_PI * (float)T_cur / (float)T_i));
}
