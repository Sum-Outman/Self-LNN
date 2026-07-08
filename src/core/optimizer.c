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
    int slow_weights_initialized; /**< P0修复: Ranger LookAhead慢权重是否已初始化标志。
                                       0=未初始化（需从当前参数快照），1=已初始化。
                                       在ensure_buffers重分配或optimizer_reset时重置为0。 */
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

    /* P0修复: 缓冲区重分配后，Ranger慢权重(cache_buffer)被calloc清零，
     * 需要重新从当前参数快照初始化，否则LookAhead同步会使用全零慢权重
     * 导致参数被错误地拉向零。 */
    optimizer->slow_weights_initialized = 0;

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
                !optimizer->beta1_power || !optimizer->beta2_power) {
                /* P1-02修复: 分配失败时释放本批次已分配的所有缓冲区，防止内存泄漏 */
                safe_free((void**)&optimizer->momentum_buffer);
                safe_free((void**)&optimizer->velocity_buffer);
                safe_free((void**)&optimizer->beta1_power);
                safe_free((void**)&optimizer->beta2_power);
                return -1;
            }
            *optimizer->beta1_power = 1.0f;
            *optimizer->beta2_power = 1.0f;
            break;

        case OPTIMIZER_ADADELTA:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer) {
                /* P1-02修复: 分配失败时释放本批次已分配的所有缓冲区，防止内存泄漏 */
                safe_free((void**)&optimizer->momentum_buffer);
                safe_free((void**)&optimizer->velocity_buffer);
                return -1;
            }
            break;

        case OPTIMIZER_LAMB:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->beta1_power = (float*)safe_calloc(1, sizeof(float));
            optimizer->beta2_power = (float*)safe_calloc(1, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer ||
                !optimizer->beta1_power || !optimizer->beta2_power) {
                /* P1-02修复: 分配失败时释放本批次已分配的所有缓冲区，防止内存泄漏 */
                safe_free((void**)&optimizer->momentum_buffer);
                safe_free((void**)&optimizer->velocity_buffer);
                safe_free((void**)&optimizer->beta1_power);
                safe_free((void**)&optimizer->beta2_power);
                return -1;
            }
            *optimizer->beta1_power = 1.0f;
            *optimizer->beta2_power = 1.0f;
            break;

        case OPTIMIZER_LARS:
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            if (!optimizer->momentum_buffer) return -1;
            break;

        case OPTIMIZER_RANGER:
            /* P0修复: RANGER分配失败时释放已分配的缓冲区，防止内存泄漏 */
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->cache_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->beta1_power = (float*)safe_calloc(1, sizeof(float));
            optimizer->beta2_power = (float*)safe_calloc(1, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer ||
                !optimizer->cache_buffer || !optimizer->beta1_power || !optimizer->beta2_power) {
                safe_free((void**)&optimizer->momentum_buffer);
                safe_free((void**)&optimizer->velocity_buffer);
                safe_free((void**)&optimizer->cache_buffer);
                safe_free((void**)&optimizer->beta1_power);
                safe_free((void**)&optimizer->beta2_power);
                return -1;
            }
            *optimizer->beta1_power = 1.0f;
            *optimizer->beta2_power = 1.0f;
            break;

        case OPTIMIZER_NOVOGRAD:
            /* P0修复: NOVOGRAD分配失败时释放已分配的缓冲区，防止内存泄漏 */
            optimizer->momentum_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->velocity_buffer = (float*)safe_calloc(num_params, sizeof(float));
            optimizer->beta1_power = (float*)safe_calloc(1, sizeof(float));
            if (!optimizer->momentum_buffer || !optimizer->velocity_buffer ||
                !optimizer->beta1_power) {
                safe_free((void**)&optimizer->momentum_buffer);
                safe_free((void**)&optimizer->velocity_buffer);
                safe_free((void**)&optimizer->beta1_power);
                return -1;
            }
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
     * S-002修复: 梯度裁剪从max_abs改为标准L2范数裁剪。
     *   max_abs裁剪对大模型参数（例如1M+参数）会过度裁剪——单个大梯度
     *   值会触发全量缩放，而L2范数衡量梯度的"整体大小"，更符合标准做法。
     * gradients改为non-const以支持原地过滤 */
    {
        float grad_l2 = 0.0f;
        size_t nan_count = 0;
        for (size_t k = 0; k < num_params; k++) {
            float g = gradients[k];
            if (!isfinite(g)) {
                gradients[k] = 0.0f;
                nan_count++;
            } else {
                grad_l2 += g * g;
            }
        }
        if (nan_count > 0) {
            log_warning("[梯度裁剪] 过滤了%zu个NaN/Inf梯度值(占总数%.2f%%)",
                       nan_count, (float)nan_count * 100.0f / (float)num_params);
        }
        float grad_norm = sqrtf(grad_l2);
        if (grad_norm > OPTIMIZER_DEFAULT_GRAD_CLIP_NORM) {
            float clip_ratio = OPTIMIZER_DEFAULT_GRAD_CLIP_NORM / grad_norm;
            for (size_t k = 0; k < num_params; k++) {
                gradients[k] *= clip_ratio;
            }
            log_debug("[梯度裁剪] L2范数=%.2f > 阈值=%.2f, 裁剪比=%.4f",
                     grad_norm, OPTIMIZER_DEFAULT_GRAD_CLIP_NORM, clip_ratio);
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
                /* P1修复: 使用AdamW风格解耦权重衰减（加法形式），替代原乘法衰减。
                 * 原实现先对参数施加乘法衰减再做Adam更新，虽数学等价但存在数值
                 * 稳定性隐患（大wd时参数先大幅缩小再更新）。AdamW标准形式将
                 * 权重衰减合并到参数更新公式中，更稳定且与AdamW实现一致。 */
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * gradients[i];
                optimizer->velocity_buffer[i] = b2 * optimizer->velocity_buffer[i] +
                                                (1.0f - b2) * gradients[i] * gradients[i];

                /* ZSFJJJ-H003修复: 防止Adam偏差校正分母趋零，已有完整保护
                 * 长期训练(t>10000)后beta1_power下溢为0(float精度~1e-7),
                 * 导致1.0-0=1.0, 偏差校正失去意义。
                 * 使用min epsilon确保分母始终>=1e-8, 避免除零和校正失效。 */
                float b1_corr = (1.0f - *optimizer->beta1_power);
                float b2_corr = (1.0f - *optimizer->beta2_power);
                if (b1_corr < 1e-7f) b1_corr = 1e-7f;
                if (b2_corr < 1e-7f) b2_corr = 1e-7f;
                float m_hat = optimizer->momentum_buffer[i] / b1_corr;
                float v_hat = optimizer->velocity_buffer[i] / b2_corr;

                /* P1修复: 解耦权重衰减，在参数更新中直接加上wd*param项 */
                parameters[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * parameters[i]);
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

                /* P0-H002修复: 将下溢保护合并到主计算中, 消除重复计算 */
                float bc1 = 1.0f - *optimizer->beta1_power;
                float bc2 = 1.0f - *optimizer->beta2_power;
                if (bc1 < 1e-7f) bc1 = 1e-7f;
                if (bc2 < 1e-7f) bc2 = 1e-7f;
                float m_hat = optimizer->momentum_buffer[i] / bc1;
                float v_hat = optimizer->velocity_buffer[i] / bc2;

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
            /* P0修复: 偏置校正除零保护，与Adam保持一致（β^t→1时1/(1-β^t)会溢出） */
            float b1_corr = 1.0f - *optimizer->beta1_power;
            float b2_corr = 1.0f - *optimizer->beta2_power;
            if (b1_corr < 1e-7f) b1_corr = 1e-7f;
            if (b2_corr < 1e-7f) b2_corr = 1e-7f;
            float inv_b1 = 1.0f / b1_corr;
            float inv_b2 = 1.0f / b2_corr;
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
            /* P1修复: 统一使用double计算rho_inf和rho_t，避免float/double混用
             * 导致精度不一致。原实现rho_inf为float但参与double运算，
             * float→double隐式转换可能引入舍入误差，尤其在b2接近1.0时
             * (1-b2)极小，float精度不足导致rho_inf偏差。 */
            double d_b2 = (double)b2;
            double d_rho_inf = 2.0 / (1.0 - d_b2) - 1.0;
            double d_beta2_t = (double)(*optimizer->beta2_power);
            double d_step = (double)step;
            double d_rho_t = d_rho_inf - 2.0 * (d_step * d_beta2_t) / (1.0 - d_beta2_t + 1e-12);
            float rho_inf = (float)d_rho_inf;
            float rho_t = (float)d_rho_t;

            /* P0修复: LookAhead慢权重初始化。
             * 原实现在step==0时初始化，但ensure_buffers重分配缓冲区后
             * step可能>0而cache_buffer已被calloc清零，导致慢权重全零。
             * 现改用slow_weights_initialized标志，确保首次使用或重分配后
             * 都从当前参数快照初始化慢权重。 */
            int k = optimizer->config.lookahead_k > 0 ? optimizer->config.lookahead_k : 5;
            float alpha = optimizer->config.lookahead_alpha > 0.0f
                          ? optimizer->config.lookahead_alpha : 0.5f;
            if (!optimizer->slow_weights_initialized) {
                for (i = 0; i < num_params; i++) {
                    optimizer->cache_buffer[i] = parameters[i];
                }
                optimizer->slow_weights_initialized = 1;
            }

            for (i = 0; i < num_params; i++)
            {
                /* FIX-003: 解耦权重衰减，梯度仅含原始梯度 */
                parameters[i] *= (1.0f - lr * wd);
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * gradients[i];
                optimizer->velocity_buffer[i] = b2 * optimizer->velocity_buffer[i] +
                                                (1.0f - b2) * gradients[i] * gradients[i];

                /* P0修复: 偏置校正除零保护，与Adam保持一致（β^t→1时1/(1-β^t)会溢出） */
                float b1_corr = 1.0f - *optimizer->beta1_power; if (b1_corr < 1e-7f) b1_corr = 1e-7f;
                float b2_corr = 1.0f - *optimizer->beta2_power; if (b2_corr < 1e-7f) b2_corr = 1e-7f;
                float m_hat = optimizer->momentum_buffer[i] / b1_corr;
                float v_hat = optimizer->velocity_buffer[i] / b2_corr;

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
            /* P0修复: 偏置校正除零保护，与Adam保持一致（β1^t→1时1/(1-β1^t)会溢出） */
            float b1_corr = 1.0f - *optimizer->beta1_power;
            if (b1_corr < 1e-7f) b1_corr = 1e-7f;
            for (i = 0; i < num_params; i++)
            {
                float g_normed = gradients[i] / v_sqrt;
                optimizer->momentum_buffer[i] = b1 * optimizer->momentum_buffer[i] +
                                                (1.0f - b1) * g_normed;
                float m_hat = optimizer->momentum_buffer[i] / b1_corr;
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
            float wd = optimizer->config.weight_decay;  /* DEEP-005修复: wd未声明，使用optimizer配置 */
            idx = 0;
            for (int g = 0; g < num_groups; g++) {
                float* params = groups[g].parameters;
                const float* grads = groups[g].gradients;
                size_t n = groups[g].num_params;
                /* DEEP-005修复: groups[g]无weight_decay字段，使用optimizer全局配置 */
                float grp_wd = wd; 
                for (size_t i = 0; i < n; i++, idx++) {
                    /* P-FIX-013: 添加RMSProp multi_group分支的权重衰减（与单参数组版本对齐） */
                    params[i] *= (1.0f - lr * grp_wd);
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
            float wd = optimizer->config.weight_decay;  /* DEEP-FIX: 多组Adam添加权重衰减 */
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
                    /* DEEP-FIX: 多组Adam添加下溢保护 */
                    float b1c = 1.0f - *optimizer->beta1_power; if (b1c < 1e-7f) b1c = 1e-7f;
                    float b2c = 1.0f - *optimizer->beta2_power; if (b2c < 1e-7f) b2c = 1e-7f;
                    float m_hat = optimizer->momentum_buffer[idx] / b1c;
                    float v_hat = optimizer->velocity_buffer[idx] / b2c;
                    /* P1修复: 多组Adam改用AdamW风格解耦权重衰减（加法形式），
                     * 与单参数组版本和AdamW实现保持一致 */
                    params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * params[i]);
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
                    /* P0修复: 偏置校正除零保护，与Adam保持一致（β^t→1时1/(1-β^t)会溢出） */
                    float b1_corr = 1.0f - *optimizer->beta1_power; if (b1_corr < 1e-7f) b1_corr = 1e-7f;
                    float b2_corr = 1.0f - *optimizer->beta2_power; if (b2_corr < 1e-7f) b2_corr = 1e-7f;
                    float m_hat = optimizer->momentum_buffer[idx] / b1_corr;
                    float v_hat = optimizer->velocity_buffer[idx] / b2_corr;
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
            /* P0修复: 偏置校正除零保护，与Adam保持一致（β^t→1时1/(1-β^t)会溢出） */
            float b1_corr = 1.0f - *optimizer->beta1_power;
            float b2_corr = 1.0f - *optimizer->beta2_power;
            if (b1_corr < 1e-7f) b1_corr = 1e-7f;
            if (b2_corr < 1e-7f) b2_corr = 1e-7f;
            float inv_b1 = 1.0f / b1_corr;
            float inv_b2 = 1.0f / b2_corr;

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

            /* P1修复: 统一使用double计算rho_inf和rho_t，避免float/double混用精度不一致 */
            double d_b2 = (double)b2;
            double d_rho_inf = 2.0 / (1.0 - d_b2) - 1.0;
            double d_beta2_t = (double)(*optimizer->beta2_power);
            double d_step = (double)step;
            double d_rho_t = d_rho_inf - 2.0 * (d_step * d_beta2_t) / (1.0 - d_beta2_t + 1e-12);
            float rho_inf = (float)d_rho_inf;
            float rho_t = (float)d_rho_t;

            int k = optimizer->config.lookahead_k > 0 ? optimizer->config.lookahead_k : 5;
            float alpha = optimizer->config.lookahead_alpha > 0.0f
                          ? optimizer->config.lookahead_alpha : 0.5f;

            /* P0修复: LookAhead慢权重初始化，使用标志位替代step==0检查 */
            if (!optimizer->slow_weights_initialized) {
                idx = 0;
                for (int g = 0; g < num_groups; g++) {
                    float* params = groups[g].parameters;
                    size_t n = groups[g].num_params;
                    for (size_t i = 0; i < n; i++, idx++) {
                        optimizer->cache_buffer[idx] = params[i];
                    }
                }
                optimizer->slow_weights_initialized = 1;
            }

            /* P0修复: 偏置校正除零保护，与Adam保持一致（β^t→1时1/(1-β^t)会溢出） */
            float b1_corr = 1.0f - *optimizer->beta1_power;
            float b2_corr = 1.0f - *optimizer->beta2_power;
            if (b1_corr < 1e-7f) b1_corr = 1e-7f;
            if (b2_corr < 1e-7f) b2_corr = 1e-7f;

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
                    float m_hat = optimizer->momentum_buffer[idx] / b1_corr;
                    float v_hat = optimizer->velocity_buffer[idx] / b2_corr;
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
                    /* P0修复: 偏置校正除零保护，与Adam保持一致（β1^t→1时1/(1-β1^t)会溢出） */
                    float b1_corr = 1.0f - *optimizer->beta1_power; if (b1_corr < 1e-7f) b1_corr = 1e-7f;
                    float m_hat = optimizer->momentum_buffer[idx] / b1_corr;
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
    /* P0修复: 重置时cache_buffer被memset清零，慢权重需重新初始化 */
    optimizer->slow_weights_initialized = 0;
    optimizer->is_initialized = 1; /* 缓冲区已通过memset清零，保持已初始化状态，避免不必要的重新分配 */
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
    /* P1修复5: step=0时powf(beta,0)=1，b1_corr/b2_corr=0，后续m_hat=m[i]/b1_corr
     * 除零产生Inf/NaN并污染动量缓冲区m和v，导致永久训练损坏。
     * 添加下限保护(与标准Adam实现一致)，step=0时偏置校正退化为约等于1。 */
    if (b1_corr < 1e-7f) b1_corr = 1e-7f;
    if (b2_corr < 1e-7f) b2_corr = 1e-7f;
/* 添加逐元素NaN/Inf保护，
     * 防止单次NaN污染动量缓冲区m和v导致永久训练损坏 */
    for (size_t i = 0; i < n; i++) {
        float g = grads[i];
        if (!isfinite(g)) g = 0.0f;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] / b1_corr, v_hat = v[i] / b2_corr;
        /* P1修复: 使用标准AdamW解耦权重衰减（加法形式），替代原乘法衰减。
         * 两种形式数学等价，但加法形式数值更稳定且为标准实现。 */
        params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params[i]);
    }
    return 0;
}

/* TRAIN-21: Cosine Annealing Warm Restarts */
float lr_cosine_annealing(float base_lr, float min_lr, int epoch, int T_0, int T_mult) {
    /* P0修复: T_0<=0时while循环永不终止（T_cur恒>=T_i且T_i恒<=0），
     * 且最终除以T_i=0导致除零异常。直接返回基础学习率。
     * T_mult<=0时周期倍增异常（负数导致T_i变负、0导致周期不增长），
     * 退化为固定周期T_mult=1。 */
    if (T_0 <= 0) return base_lr;
    if (T_mult <= 0) T_mult = 1;
    int T_cur = epoch, T_i = T_0;
    while (T_cur >= T_i) {
        T_cur -= T_i;
        T_i *= T_mult;
        /* 防御性保护：T_mult溢出可能导致T_i<=0，回退到初始周期 */
        if (T_i <= 0) T_i = T_0;
    }
    /* 最终除零保护：极端情况下确保T_i>0 */
    if (T_i <= 0) T_i = 1;
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf((float)M_PI * (float)T_cur / (float)T_i));
}
