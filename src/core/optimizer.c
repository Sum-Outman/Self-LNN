#include "selflnn/core/optimizer.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

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

int optimizer_step(Optimizer* optimizer, float* parameters, const float* gradients,
                   size_t num_params, size_t step)
{
    if (!optimizer || !parameters || !gradients || num_params == 0) return -1;

    if (ensure_buffers(optimizer, num_params) != 0) return -1;
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
            for (i = 0; i < num_params; i++)
            {
                float decay = optimizer->config.weight_decay * parameters[i];
                parameters[i] -= lr * (gradients[i] + decay);
            }
            break;
        }

        case OPTIMIZER_MOMENTUM:
        {
            float mu = optimizer->config.momentum > 0.0f ? optimizer->config.momentum : 0.9f;
            for (i = 0; i < num_params; i++)
            {
                float decay = optimizer->config.weight_decay * parameters[i];
                optimizer->momentum_buffer[i] = mu * optimizer->momentum_buffer[i] -
                                                lr * (gradients[i] + decay);
                if (optimizer->config.use_nesterov)
                {
                    parameters[i] += optimizer->momentum_buffer[i] - mu * optimizer->momentum_buffer[i];
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
            for (i = 0; i < num_params; i++)
            {
                float decay = optimizer->config.weight_decay * parameters[i];
                optimizer->cache_buffer[i] += gradients[i] * gradients[i];
                float adjusted_lr = lr / (sqrtf(optimizer->cache_buffer[i]) + eps);
                parameters[i] -= adjusted_lr * (gradients[i] + decay);
            }
            break;
        }

        case OPTIMIZER_RMSPROP:
        {
            float decay_rate = optimizer->config.beta2 > 0.0f ? optimizer->config.beta2 : 0.9f;
            for (i = 0; i < num_params; i++)
            {
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
            for (i = 0; i < num_params; i++)
            {
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

            float param_norm = 0.0f, update_norm = 0.0f;
            for (i = 0; i < num_params; i++)
            {
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * gradients[i];
                optimizer->velocity_buffer[i] = b2 * optimizer->velocity_buffer[i] +
                                                (1.0f - b2) * gradients[i] * gradients[i];

                float m_hat = optimizer->momentum_buffer[i] / (1.0f - *optimizer->beta1_power);
                float v_hat = optimizer->velocity_buffer[i] / (1.0f - *optimizer->beta2_power);

                float update = m_hat / (sqrtf(v_hat) + eps) + wd * parameters[i];
                param_norm += parameters[i] * parameters[i];
                update_norm += update * update;
            }
            param_norm = sqrtf(param_norm + 1e-12f);
            update_norm = sqrtf(update_norm + 1e-12f);
            float trust_ratio = (param_norm > 0.0f && update_norm > 0.0f)
                                ? trust * param_norm / update_norm : 1.0f;

            for (i = 0; i < num_params; i++)
            {
                float m_hat = optimizer->momentum_buffer[i] / (1.0f - *optimizer->beta1_power);
                float v_hat = optimizer->velocity_buffer[i] / (1.0f - *optimizer->beta2_power);
                float update = m_hat / (sqrtf(v_hat) + eps) + wd * parameters[i];
                parameters[i] -= lr * trust_ratio * update;
            }
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

            /* RAdam整流 */
            float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
            float beta2_t = *optimizer->beta2_power;
            float rho_t = rho_inf - 2.0f * (float)step * beta2_t / (1.0f - beta2_t + 1e-12f);
            (void)rho_inf;

            for (i = 0; i < num_params; i++)
            {
                float g = gradients[i] + wd * parameters[i];
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * g;
                optimizer->velocity_buffer[i] = b2 * optimizer->velocity_buffer[i] +
                                                (1.0f - b2) * g * g;

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

                /* LookAhead: 保存慢权重到cache_buffer（快照） */
                optimizer->cache_buffer[i] = parameters[i];
            }

            /* LookAhead同步：每lookahead_k步 = 慢 → 快 */
            int k = optimizer->config.lookahead_k > 0 ? optimizer->config.lookahead_k : 5;
            float alpha = optimizer->config.lookahead_alpha > 0.0f
                          ? optimizer->config.lookahead_alpha : 0.5f;
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
            *optimizer->beta1_power *= b1;

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
    for (size_t i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
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
