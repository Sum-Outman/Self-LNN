/**
 * @file quaternion_optimizer.c
 * @brief 四元数优化器实现
 *
 * 在四元数流形上执行梯度优化的完整实现。
 * 支持四元数SGD、四元数动量SGD、四元数Adam和四元数AdamW。
 */

#include "selflnn/core/quaternion_optimizer.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/**
 * @brief 四元数优化器内部结构
 */
struct QuatOptimizer {
    QuatOptimizerConfig config;     /**< 优化器配置 */
    size_t num_quaternions;         /**< 四元数数量 */
    int is_initialized;             /**< 是否已初始化 */
    
    /* 动量缓冲区（四元数数组） */
    Quaternion* momentum;           /**< 一阶动量（SGD动量 / Adam m_t） */
    
    /* Adam专用缓冲区 */
    Quaternion* velocity;           /**< 二阶动量（Adam v_t） */
    
    /* 标量缓冲区（用于标量参数更新） */
    float* momentum_scalar;         /**< 标量动量缓冲区 */
    float* velocity_scalar;         /**< 标量二阶矩缓冲区 */
    size_t num_scalars;             /**< 标量参数数量 */
};

QuatOptimizer* quat_optimizer_create(const QuatOptimizerConfig* config) {
    if (!config) return NULL;
    
    QuatOptimizer* opt = (QuatOptimizer*)safe_calloc(1, sizeof(QuatOptimizer));
    if (!opt) return NULL;
    
    opt->config = *config;
    opt->is_initialized = 0;
    opt->num_quaternions = 0;
    opt->num_scalars = 0;
    opt->momentum = NULL;
    opt->velocity = NULL;
    opt->momentum_scalar = NULL;
    opt->velocity_scalar = NULL;
    
    return opt;
}

void quat_optimizer_free(QuatOptimizer* optimizer) {
    if (!optimizer) return;
    safe_free((void**)&optimizer->momentum);
    safe_free((void**)&optimizer->velocity);
    safe_free((void**)&optimizer->momentum_scalar);
    safe_free((void**)&optimizer->velocity_scalar);
    safe_free((void**)&optimizer);
}

/**
 * @brief 确保四元数缓冲区已分配
 */
static int quat_ensure_buffers(QuatOptimizer* opt, size_t num_quaternions, size_t num_scalars) {
    if (!opt) return -1;
    if (opt->num_quaternions == num_quaternions && opt->num_scalars == num_scalars && opt->is_initialized)
        return 0;
    
    safe_free((void**)&opt->momentum);
    safe_free((void**)&opt->velocity);
    safe_free((void**)&opt->momentum_scalar);
    safe_free((void**)&opt->velocity_scalar);
    
    opt->num_quaternions = num_quaternions;
    opt->num_scalars = num_scalars;
    
    QuatOptimizerType t = opt->config.type;
    
    if (t == QUAT_OPTIMIZER_MOMENTUM || t == QUAT_OPTIMIZER_ADAM || t == QUAT_OPTIMIZER_ADAMW) {
        if (num_quaternions > 0) {
            opt->momentum = (Quaternion*)safe_calloc(num_quaternions, sizeof(Quaternion));
            if (!opt->momentum) return -1;
        }
        if (num_scalars > 0) {
            opt->momentum_scalar = (float*)safe_calloc(num_scalars, sizeof(float));
            if (!opt->momentum_scalar) return -1;
        }
    }
    
    if (t == QUAT_OPTIMIZER_ADAM || t == QUAT_OPTIMIZER_ADAMW) {
        if (num_quaternions > 0) {
            opt->velocity = (Quaternion*)safe_calloc(num_quaternions, sizeof(Quaternion));
            if (!opt->velocity) return -1;
        }
        if (num_scalars > 0) {
            opt->velocity_scalar = (float*)safe_calloc(num_scalars, sizeof(float));
            if (!opt->velocity_scalar) return -1;
        }
    }
    
    opt->is_initialized = 1;
    return 0;
}

/**
 * @brief 裁剪四元数梯度范数
 */
static void clip_quat_grad_norm(Quaternion* grads, size_t n, float max_norm) {
    if (max_norm <= 0.0f) return;
    float total_norm = 0.0f;
    for (size_t i = 0; i < n; i++) {
        total_norm += grads[i].w * grads[i].w;
        total_norm += grads[i].x * grads[i].x;
        total_norm += grads[i].y * grads[i].y;
        total_norm += grads[i].z * grads[i].z;
    }
    total_norm = sqrtf(total_norm);
    if (total_norm > max_norm && total_norm > 1e-12f) {
        float scale = max_norm / total_norm;
        for (size_t i = 0; i < n; i++) {
            grads[i].w *= scale;
            grads[i].x *= scale;
            grads[i].y *= scale;
            grads[i].z *= scale;
        }
    }
}

/**
 * @brief 四元数Adam更新（单个四元数）
 */
static inline void quat_adam_update(Quaternion* param, const Quaternion* grad,
                                    Quaternion* m, Quaternion* v,
                                    float lr, float b1, float b2, float eps,
                                    float wd, size_t step) {
    float step_f = (float)step;
    float b1_t = powf(b1, step_f);
    float b2_t = powf(b2, step_f);
    float m_hat_scale = 1.0f / (1.0f - b1_t + 1e-10f);
    float v_hat_scale = 1.0f / (1.0f - b2_t + 1e-10f);
    
    /* 更新一阶矩: m = b1 * m + (1-b1) * grad */
    m->w = b1 * m->w + (1.0f - b1) * grad->w;
    m->x = b1 * m->x + (1.0f - b1) * grad->x;
    m->y = b1 * m->y + (1.0f - b1) * grad->y;
    m->z = b1 * m->z + (1.0f - b1) * grad->z;
    
    /* 更新二阶矩: v = b2 * v + (1-b2) * grad^2 */
    v->w = b2 * v->w + (1.0f - b2) * grad->w * grad->w;
    v->x = b2 * v->x + (1.0f - b2) * grad->x * grad->x;
    v->y = b2 * v->y + (1.0f - b2) * grad->y * grad->y;
    v->z = b2 * v->z + (1.0f - b2) * grad->z * grad->z;
    
    /* 偏差校正: m_hat = m / (1-b1^t), v_hat = v / (1-b2^t) */
    float mw = m->w * m_hat_scale, mx = m->x * m_hat_scale;
    float my = m->y * m_hat_scale, mz = m->z * m_hat_scale;
    
    float vw = v->w * v_hat_scale, vx = v->x * v_hat_scale;
    float vy = v->y * v_hat_scale, vz = v->z * v_hat_scale;
    
    /* 更新: param -= lr * m_hat / (sqrt(v_hat) + eps) */
    float step_w = lr * mw / (sqrtf(vw) + eps);
    float step_x = lr * mx / (sqrtf(vx) + eps);
    float step_y = lr * my / (sqrtf(vy) + eps);
    float step_z = lr * mz / (sqrtf(vz) + eps);
    
    /* 权重衰减 (AdamW) */
    if (wd > 0.0f) {
        step_w += lr * wd * param->w;
        step_x += lr * wd * param->x;
        step_y += lr * wd * param->y;
        step_z += lr * wd * param->z;
    }
    
    param->w -= step_w;
    param->x -= step_x;
    param->y -= step_y;
    param->z -= step_z;
}

/**
 * @brief 标量Adam更新
 */
static inline void scalar_adam_update(float* param, float grad,
                                      float* m, float* v,
                                      float lr, float b1, float b2, float eps,
                                      float wd, size_t step) {
    float step_f = (float)step;
    float b1_t = powf(b1, step_f);
    float b2_t = powf(b2, step_f);
    float m_hat_s = 1.0f / (1.0f - b1_t + 1e-10f);
    float v_hat_s = 1.0f / (1.0f - b2_t + 1e-10f);
    
    *m = b1 * (*m) + (1.0f - b1) * grad;
    *v = b2 * (*v) + (1.0f - b2) * grad * grad;
    
    float m_hat = (*m) * m_hat_s;
    float v_hat = (*v) * v_hat_s;
    
    float step_v = lr * m_hat / (sqrtf(v_hat) + eps);
    if (wd > 0.0f) step_v += lr * wd * (*param);
    *param -= step_v;
}

int quat_optimizer_step(QuatOptimizer* optimizer, Quaternion* params,
                         const Quaternion* grads, size_t num_quaternions, size_t step) {
    if (!optimizer || !params || !grads || num_quaternions == 0) return -1;
    if (step == 0) step = 1;
    
    int ret = quat_ensure_buffers(optimizer, num_quaternions, 0);
    if (ret != 0) return -1;
    
    QuatOptimizerConfig* cfg = &optimizer->config;
    float lr = cfg->learning_rate;
    
    /* 梯度裁剪 */
    Quaternion* working_grads = (Quaternion*)grads;
    Quaternion* clipped = NULL;
    if (cfg->clamp_grad_norm && cfg->max_grad_norm > 0.0f) {
        clipped = (Quaternion*)safe_malloc(num_quaternions * sizeof(Quaternion));
        if (clipped) {
            memcpy(clipped, grads, num_quaternions * sizeof(Quaternion));
            clip_quat_grad_norm(clipped, num_quaternions, cfg->max_grad_norm);
            working_grads = clipped;
        }
    }
    
    switch (cfg->type) {
        case QUAT_OPTIMIZER_SGD: {
            for (size_t i = 0; i < num_quaternions; i++) {
                float g_norm = quaternion_norm(&working_grads[i]);
                if (g_norm > cfg->max_grad_norm && cfg->max_grad_norm > 0.0f) {
                    Quaternion scaled = quaternion_scale(&working_grads[i],
                                                         cfg->max_grad_norm / g_norm);
                    Quaternion step_q = quaternion_scale(&scaled, lr);
                    params[i] = quaternion_subtract(&params[i], &step_q);
                } else {
                    Quaternion step_q = quaternion_scale(&working_grads[i], lr);
                    params[i] = quaternion_subtract(&params[i], &step_q);
                }
            }
            break;
        }
        
        case QUAT_OPTIMIZER_MOMENTUM: {
            float mu = cfg->momentum;
            for (size_t i = 0; i < num_quaternions; i++) {
                if (cfg->use_nesterov) {
                    float old_m_w = optimizer->momentum[i].w;
                    float old_m_x = optimizer->momentum[i].x;
                    float old_m_y = optimizer->momentum[i].y;
                    float old_m_z = optimizer->momentum[i].z;

                    optimizer->momentum[i].w = mu * optimizer->momentum[i].w + lr * working_grads[i].w;
                    optimizer->momentum[i].x = mu * optimizer->momentum[i].x + lr * working_grads[i].x;
                    optimizer->momentum[i].y = mu * optimizer->momentum[i].y + lr * working_grads[i].y;
                    optimizer->momentum[i].z = mu * optimizer->momentum[i].z + lr * working_grads[i].z;

                    params[i].w -= (mu * optimizer->momentum[i].w + lr * working_grads[i].w);
                    params[i].x -= (mu * optimizer->momentum[i].x + lr * working_grads[i].x);
                    params[i].y -= (mu * optimizer->momentum[i].y + lr * working_grads[i].y);
                    params[i].z -= (mu * optimizer->momentum[i].z + lr * working_grads[i].z);

                    (void)old_m_w; (void)old_m_x; (void)old_m_y; (void)old_m_z;
                } else {
                    optimizer->momentum[i].w = mu * optimizer->momentum[i].w + lr * working_grads[i].w;
                    optimizer->momentum[i].x = mu * optimizer->momentum[i].x + lr * working_grads[i].x;
                    optimizer->momentum[i].y = mu * optimizer->momentum[i].y + lr * working_grads[i].y;
                    optimizer->momentum[i].z = mu * optimizer->momentum[i].z + lr * working_grads[i].z;
                    params[i] = quaternion_subtract(&params[i], &optimizer->momentum[i]);
                }
            }
            break;
        }
        
        case QUAT_OPTIMIZER_ADAM:
        case QUAT_OPTIMIZER_ADAMW: {
            float wd = (cfg->type == QUAT_OPTIMIZER_ADAMW) ? cfg->weight_decay : 0.0f;
            for (size_t i = 0; i < num_quaternions; i++) {
                quat_adam_update(&params[i], &working_grads[i],
                                &optimizer->momentum[i], &optimizer->velocity[i],
                                lr, cfg->beta1, cfg->beta2, cfg->epsilon,
                                wd, step);
            }
            break;
        }

        /* M-003修复: 四元数RMSProp */
        case QUAT_OPTIMIZER_RMSPROP: {
            float rho = cfg->beta2 > 0.0f ? cfg->beta2 : 0.9f;
            for (size_t i = 0; i < num_quaternions; i++) {
                float g_w = working_grads[i].w, g_x = working_grads[i].x;
                float g_y = working_grads[i].y, g_z = working_grads[i].z;
                optimizer->velocity[i].w = rho * optimizer->velocity[i].w + (1.0f - rho) * g_w * g_w;
                optimizer->velocity[i].x = rho * optimizer->velocity[i].x + (1.0f - rho) * g_x * g_x;
                optimizer->velocity[i].y = rho * optimizer->velocity[i].y + (1.0f - rho) * g_y * g_y;
                optimizer->velocity[i].z = rho * optimizer->velocity[i].z + (1.0f - rho) * g_z * g_z;
                params[i].w -= lr * g_w / (sqrtf(optimizer->velocity[i].w) + cfg->epsilon);
                params[i].x -= lr * g_x / (sqrtf(optimizer->velocity[i].x) + cfg->epsilon);
                params[i].y -= lr * g_y / (sqrtf(optimizer->velocity[i].y) + cfg->epsilon);
                params[i].z -= lr * g_z / (sqrtf(optimizer->velocity[i].z) + cfg->epsilon);
            }
            break;
        }

        /* M-003修复: 四元数AdaGrad */
        case QUAT_OPTIMIZER_ADAGRAD: {
            for (size_t i = 0; i < num_quaternions; i++) {
                float g_w = working_grads[i].w, g_x = working_grads[i].x;
                float g_y = working_grads[i].y, g_z = working_grads[i].z;
                optimizer->velocity[i].w += g_w * g_w;
                optimizer->velocity[i].x += g_x * g_x;
                optimizer->velocity[i].y += g_y * g_y;
                optimizer->velocity[i].z += g_z * g_z;
                params[i].w -= lr * g_w / (sqrtf(optimizer->velocity[i].w) + cfg->epsilon);
                params[i].x -= lr * g_x / (sqrtf(optimizer->velocity[i].x) + cfg->epsilon);
                params[i].y -= lr * g_y / (sqrtf(optimizer->velocity[i].y) + cfg->epsilon);
                params[i].z -= lr * g_z / (sqrtf(optimizer->velocity[i].z) + cfg->epsilon);
            }
            break;
        }

        /* M-003修复: 四元数LAMB（层自适应大Batch优化） */
        case QUAT_OPTIMIZER_LAMB: {
            float wd = cfg->weight_decay;
            float b1 = cfg->beta1, b2 = cfg->beta2, eps = cfg->epsilon;
            float step_f = (float)step;
            float b1_t = powf(b1, step_f);
            float b2_t = powf(b2, step_f);
            float m_hat_scale = 1.0f / (1.0f - b1_t + 1e-10f);
            float v_hat_scale = 1.0f / (1.0f - b2_t + 1e-10f);

            /* 第一步：更新动量（不更新参数）并收集更新方向 */
            float p_norm2 = 0.0f, u_norm2 = 0.0f;
            for (size_t i = 0; i < num_quaternions; i++) {
                p_norm2 += params[i].w * params[i].w + params[i].x * params[i].x +
                           params[i].y * params[i].y + params[i].z * params[i].z;

                /* 更新一阶矩和二阶矩 */
                float gw = working_grads[i].w + wd * params[i].w;
                float gx = working_grads[i].x + wd * params[i].x;
                float gy = working_grads[i].y + wd * params[i].y;
                float gz = working_grads[i].z + wd * params[i].z;

                optimizer->momentum[i].w = b1 * optimizer->momentum[i].w + (1.0f - b1) * gw;
                optimizer->momentum[i].x = b1 * optimizer->momentum[i].x + (1.0f - b1) * gx;
                optimizer->momentum[i].y = b1 * optimizer->momentum[i].y + (1.0f - b1) * gy;
                optimizer->momentum[i].z = b1 * optimizer->momentum[i].z + (1.0f - b1) * gz;

                optimizer->velocity[i].w = b2 * optimizer->velocity[i].w + (1.0f - b2) * gw * gw;
                optimizer->velocity[i].x = b2 * optimizer->velocity[i].x + (1.0f - b2) * gx * gx;
                optimizer->velocity[i].y = b2 * optimizer->velocity[i].y + (1.0f - b2) * gy * gy;
                optimizer->velocity[i].z = b2 * optimizer->velocity[i].z + (1.0f - b2) * gz * gz;

                /* 计算更新方向范数（用于信任比） */
                float mw = optimizer->momentum[i].w * m_hat_scale;
                float mx = optimizer->momentum[i].x * m_hat_scale;
                float my = optimizer->momentum[i].y * m_hat_scale;
                float mz = optimizer->momentum[i].z * m_hat_scale;
                float vw = optimizer->velocity[i].w * v_hat_scale;
                float vx = optimizer->velocity[i].x * v_hat_scale;
                float vy = optimizer->velocity[i].y * v_hat_scale;
                float vz = optimizer->velocity[i].z * v_hat_scale;

                float uw = mw / (sqrtf(vw) + eps);
                float ux = mx / (sqrtf(vx) + eps);
                float uy = my / (sqrtf(vy) + eps);
                float uz = mz / (sqrtf(vz) + eps);
                u_norm2 += uw * uw + ux * ux + uy * uy + uz * uz;
            }

            float p_norm = sqrtf(p_norm2);
            float u_norm = sqrtf(u_norm2);
            float trust_ratio = (u_norm > 0.0f && p_norm > 0.0f) ? p_norm / u_norm : 1.0f;
            if (trust_ratio > 10.0f) trust_ratio = 10.0f;
            if (trust_ratio < 0.1f) trust_ratio = 0.1f;

            /* 第二步：应用信任比缩放后的单次更新 */
            for (size_t i = 0; i < num_quaternions; i++) {
                float mw = optimizer->momentum[i].w * m_hat_scale;
                float mx = optimizer->momentum[i].x * m_hat_scale;
                float my = optimizer->momentum[i].y * m_hat_scale;
                float mz = optimizer->momentum[i].z * m_hat_scale;
                float vw = optimizer->velocity[i].w * v_hat_scale;
                float vx = optimizer->velocity[i].x * v_hat_scale;
                float vy = optimizer->velocity[i].y * v_hat_scale;
                float vz = optimizer->velocity[i].z * v_hat_scale;

                params[i].w -= lr * trust_ratio * mw / (sqrtf(vw) + eps);
                params[i].x -= lr * trust_ratio * mx / (sqrtf(vx) + eps);
                params[i].y -= lr * trust_ratio * my / (sqrtf(vy) + eps);
                params[i].z -= lr * trust_ratio * mz / (sqrtf(vz) + eps);
            }
            break;
        }
        
        default:
            safe_free((void**)&clipped);
            return -1;
    }
    
    safe_free((void**)&clipped);
    return 0;
}

int quat_optimizer_step_scalar(QuatOptimizer* optimizer, float* params,
                                const float* grads, size_t num_params, size_t step) {
    if (!optimizer || !params || !grads || num_params == 0) return -1;
    if (step == 0) step = 1;
    
    int ret = quat_ensure_buffers(optimizer, 0, num_params);
    if (ret != 0) return -1;
    
    QuatOptimizerConfig* cfg = &optimizer->config;
    float lr = cfg->learning_rate;
    
    switch (cfg->type) {
        case QUAT_OPTIMIZER_SGD: {
            for (size_t i = 0; i < num_params; i++) {
                params[i] -= lr * grads[i];
            }
            break;
        }
        
        case QUAT_OPTIMIZER_MOMENTUM: {
            float mu = cfg->momentum;
            for (size_t i = 0; i < num_params; i++) {
                if (cfg->use_nesterov) {
                    float old_m = optimizer->momentum_scalar[i];
                    optimizer->momentum_scalar[i] = mu * optimizer->momentum_scalar[i] + lr * grads[i];
                    params[i] -= (mu * optimizer->momentum_scalar[i] + lr * grads[i]);
                    (void)old_m;
                } else {
                    optimizer->momentum_scalar[i] = mu * optimizer->momentum_scalar[i] + lr * grads[i];
                    params[i] -= optimizer->momentum_scalar[i];
                }
            }
            break;
        }
        
        case QUAT_OPTIMIZER_ADAM:
        case QUAT_OPTIMIZER_ADAMW: {
            float wd = (cfg->type == QUAT_OPTIMIZER_ADAMW) ? cfg->weight_decay : 0.0f;
            for (size_t i = 0; i < num_params; i++) {
                scalar_adam_update(&params[i], grads[i],
                                  &optimizer->momentum_scalar[i],
                                  &optimizer->velocity_scalar[i],
                                  lr, cfg->beta1, cfg->beta2, cfg->epsilon,
                                  wd, step);
            }
            break;
        }
        
        default:
            return -1;
    }
    
    return 0;
}

void quat_optimizer_reset(QuatOptimizer* optimizer) {
    if (!optimizer) return;
    
    if (optimizer->momentum) {
        memset(optimizer->momentum, 0, optimizer->num_quaternions * sizeof(Quaternion));
    }
    if (optimizer->velocity) {
        memset(optimizer->velocity, 0, optimizer->num_quaternions * sizeof(Quaternion));
    }
    if (optimizer->momentum_scalar) {
        memset(optimizer->momentum_scalar, 0, optimizer->num_scalars * sizeof(float));
    }
    if (optimizer->velocity_scalar) {
        memset(optimizer->velocity_scalar, 0, optimizer->num_scalars * sizeof(float));
    }
}

void quat_optimizer_set_lr(QuatOptimizer* optimizer, float lr) {
    if (!optimizer) return;
    optimizer->config.learning_rate = lr;
}

float quat_optimizer_get_lr(const QuatOptimizer* optimizer) {
    if (!optimizer) return 0.0f;
    return optimizer->config.learning_rate;
}

QuatOptimizerConfig quat_optimizer_default_config(QuatOptimizerType type, float learning_rate) {
    QuatOptimizerConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.type = type;
    cfg.learning_rate = learning_rate;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.epsilon = 1e-8f;
    cfg.weight_decay = 0.0f;
    cfg.momentum = 0.9f;
    cfg.use_nesterov = 0;
    cfg.clamp_grad_norm = 1;
    cfg.max_grad_norm = 5.0f;
    return cfg;
}
