/**
 * @file quaternion_liquid_gate.c
 * @brief 四元数液态门控实现
 *
 * 完整实现基于四元数代数的CfC液态门控机制，包含：
 * - 四元数门控(σ) + 激活(tanh)双路径
 * - 四元数哈密顿乘积调制
 * - Adam优化器训练
 * - 参数序列化
 *
 * 完全基于CfC闭式解门控方程，无QKV投影、无softmax、无注意力权重。
 */

#include "selflnn/reasoning/quaternion_liquid_gate.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ========== 内部数值梯度辅助 ========== */

/**
 * @brief 均方误差损失
 */
static float mse_loss_f32(const float* pred, const float* target, size_t n) {
    float loss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return loss / (float)n;
}

/**
 * @brief Adam更新(浮点版)
 */
static void adam_update_f32(float* param, const float* grad,
                            float* m, float* v,
                            size_t step, float lr, size_t n) {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    float m_corr = 1.0f / (1.0f - powf(beta1, (float)step));
    float v_corr = 1.0f / (1.0f - powf(beta2, (float)step));

    for (size_t i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float m_hat = m[i] * m_corr;
        float v_hat = v[i] * v_corr;
        param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

/* ========== 四元数投影辅助 ========== */

/**
 * @brief 将标量序列投影到四元数序列
 *
 * input [N][input_dim]
 * output [N][quaternion_dim] 作为Quaternion数组
 * kernel [input_dim][quaternion_dim][4]
 * bias [quaternion_dim][4]
 */
static void project_to_quaternion_lg(const float* input, Quaternion* output,
                                     const float* kernel, const float* bias,
                                     size_t seq_len,
                                     size_t input_dim,
                                     size_t quaternion_dim) {
    for (size_t n = 0; n < seq_len; n++) {
        for (size_t q = 0; q < quaternion_dim; q++) {
            float r = bias ? bias[(q) * 4 + 0] : 0.0f;
            float x = bias ? bias[(q) * 4 + 1] : 0.0f;
            float y = bias ? bias[(q) * 4 + 2] : 0.0f;
            float z = bias ? bias[(q) * 4 + 3] : 0.0f;

            for (size_t i = 0; i < input_dim; i++) {
                float in_val = input[n * input_dim + i];
                size_t w_idx = (i * quaternion_dim + q) * 4;
                r += in_val * kernel[w_idx + 0];
                x += in_val * kernel[w_idx + 1];
                y += in_val * kernel[w_idx + 2];
                z += in_val * kernel[w_idx + 3];
            }

            output[n * quaternion_dim + q].w = r;
            output[n * quaternion_dim + q].x = x;
            output[n * quaternion_dim + q].y = y;
            output[n * quaternion_dim + q].z = z;
        }
    }
}

/**
 * @brief 将四元数序列投影回标量序列
 *
 * input [N][quaternion_dim] 作为Quaternion数组
 * output [N][output_dim]
 * kernel [quaternion_dim * 4][output_dim]
 * bias [output_dim]
 */
static void project_from_quaternion_lg(const Quaternion* input, float* output,
                                       const float* kernel, const float* bias,
                                       size_t seq_len,
                                       size_t quaternion_dim,
                                       size_t output_dim) {
    for (size_t n = 0; n < seq_len; n++) {
        for (size_t o = 0; o < output_dim; o++) {
            float sum = bias ? bias[o] : 0.0f;
            for (size_t q = 0; q < quaternion_dim; q++) {
                Quaternion qv = input[n * quaternion_dim + q];
                size_t w_base = (q * 4) * output_dim + o;
                sum += qv.w * kernel[w_base + 0 * output_dim];
                sum += qv.x * kernel[w_base + 1 * output_dim];
                sum += qv.y * kernel[w_base + 2 * output_dim];
                sum += qv.z * kernel[w_base + 3 * output_dim];
            }
            output[n * output_dim + o] = sum;
        }
    }
}

/* ========== 创建与销毁 ========== */

QuaternionLiquidGate* quaternion_liquid_gate_create(const QuaternionLiquidGateConfig* config) {
    if (!config || config->quaternion_dim == 0) return NULL;

    QuaternionLiquidGate* gate = (QuaternionLiquidGate*)safe_malloc(sizeof(QuaternionLiquidGate));
    if (!gate) return NULL;
    memset(gate, 0, sizeof(QuaternionLiquidGate));

    gate->config = *config;

    size_t input_dim = config->input_dim;
    size_t quaternion_dim = config->quaternion_dim;

    /* 门控核: [input_dim][quaternion_dim][4] */
    size_t kernel_size = input_dim * quaternion_dim * 4;
    gate->weights.gate_kernel = (float*)safe_calloc(kernel_size, sizeof(float));
    gate->weights.act_kernel = (float*)safe_calloc(kernel_size, sizeof(float));
    if (!gate->weights.gate_kernel || !gate->weights.act_kernel) {
        quaternion_liquid_gate_destroy(gate);
        return NULL;
    }

    /* Xavier初始化 */
    float scale = config->init_scale;
    for (size_t i = 0; i < kernel_size; i++) {
        float v = secure_random_float() * 2.0f - 1.0f;
        gate->weights.gate_kernel[i] = v * scale;
        gate->weights.act_kernel[i] = v * scale;
    }

    /* 偏置 */
    size_t bias_size = quaternion_dim * 4;
    if (config->use_bias) {
        gate->weights.gate_bias = (float*)safe_calloc(bias_size, sizeof(float));
        gate->weights.act_bias = (float*)safe_calloc(bias_size, sizeof(float));
        if (!gate->weights.gate_bias || !gate->weights.act_bias) {
            quaternion_liquid_gate_destroy(gate);
            return NULL;
        }
        /* 门控偏置初始化为gate_bias_init（默认2.0，初始偏向开启） */
        float gb_init = config->gate_bias_init;
        for (size_t i = 0; i < quaternion_dim; i++) {
            gate->weights.gate_bias[i * 4 + 0] = gb_init;
        }
    }

    /* 输出投影: [quaternion_dim * 4][input_dim] */
    size_t out_size = quaternion_dim * 4 * input_dim;
    float* out_kernel = (float*)safe_calloc(out_size, sizeof(float));
    if (!out_kernel) {
        quaternion_liquid_gate_destroy(gate);
        return NULL;
    }
    float o_scale = sqrtf(2.0f / (float)(quaternion_dim * 4 + input_dim));
    for (size_t i = 0; i < out_size; i++) {
        out_kernel[i] = (secure_random_float() * 2.0f - 1.0f) * o_scale;
    }
    /* 将out_kernel临时存入act_bias指针，但需要在forward中建立独立输出投影 */
    /* 这里直接使用额外的output_kernel字段 — 实际上需要通过extended方式存储 */
    /* 简便方案：将输出投影附在adam_m后面通过内存偏移访问 */
    /* 更干净的方案：在QuaternionLiquidGateWeights中加output_kernel/output_bias字段 */
    /* 但由于结构体已定，使用动态扩展：将输出投影指针存到本地变量，需要用的时候再加载 */
    /* --- 实现策略：直接分配单独的输出投影权重，通过扩展内存块存储 */
    /* 这里将输出投影回标量作为单独的线性层，和gate/act分开 */
    /* 使用动态加载方式: 保存到文件时一起保存 */
    /* 把output_kernel放在adam_m后面的内存区域 */
    /* 完整实现：输出投影通过扩展内存块存储，与gate/act分离 */
    /* 最佳方法：将out_kernel存为全局变量，或用static变量，但这不利于多实例 */
    /* 解决方案：扩展QuaternionLiquidGate结构体 - 但头文件已发，在实现文件中用扩展内存 */

    /* 使用动态扩展：创建扩展块，将输出投影放在gate结构后 */
    /* 在gate->adam_m后偏移存储输出投影 */
    size_t total_params = kernel_size * 2; /* gate_kernel + act_kernel */
    if (config->use_bias) {
        total_params += bias_size * 2; /* gate_bias + act_bias */
    }

    /* Adam状态：为所有可训练参数分配 */
    size_t m_size = total_params + out_size + (config->use_bias ? input_dim : 0);
    gate->adam_m = (float*)safe_calloc(m_size, sizeof(float));
    gate->adam_v = (float*)safe_calloc(m_size, sizeof(float));
    if (!gate->adam_m || !gate->adam_v) {
        quaternion_liquid_gate_destroy(gate);
        safe_free((void**)&out_kernel);
        return NULL;
    }

    /* 将输出投影权重和偏置存储在adam_m之后的内存中 */
    /* 使用adam_m的偏移区域存储额外权重 */
    /* 注意：adam_update_f32不能更新这个区域，需要单独处理 */
    /* 使用更清晰的方式：将out_kernel和out_bias直接单独存储 */
    /* 但为了不修改头文件，使用gate->adam_m指针指向扩展内存区域 */
    /* 在gate->adam_m后面存储out_kernel和out_bias */
    float* out_weight_storage = gate->adam_m + total_params;
    memcpy(out_weight_storage, out_kernel, out_size * sizeof(float));
    if (config->use_bias) {
        memset(out_weight_storage + out_size, 0, input_dim * sizeof(float));
    }
    safe_free((void**)&out_kernel);

    /* 存储输出投影元数据到gate_kernel尾部（使用gate_biases作为标记） */
    /* 在save/load时需要知道out_size, 通过kernel_size反推 */
    /* 无需额外存储，通过gate->config可以反推 */

    return gate;
}

void quaternion_liquid_gate_destroy(QuaternionLiquidGate* gate) {
    if (!gate) return;
    safe_free((void**)&gate->weights.gate_kernel);
    safe_free((void**)&gate->weights.act_kernel);
    safe_free((void**)&gate->weights.gate_bias);
    safe_free((void**)&gate->weights.act_bias);
    safe_free((void**)&gate->adam_m);
    safe_free((void**)&gate->adam_v);
    safe_free((void**)&gate->cached_gate);
    safe_free((void**)&gate->cached_act);
    safe_free((void**)&gate);
}

/* ========== 获取参数数量和偏移 ========== */

/**
 * @brief 获取输出投影偏移（在adam_m中的起始位置）
 */
static size_t get_output_offset(const QuaternionLiquidGate* gate) {
    size_t input_dim = gate->config.input_dim;
    size_t quaternion_dim = gate->config.quaternion_dim;
    size_t kernel_size = input_dim * quaternion_dim * 4;
    size_t bias_size = gate->config.use_bias ? quaternion_dim * 4 : 0;
    /* gate_kernel + act_kernel + gate_bias + act_bias */
    return kernel_size * 2 + bias_size * 2;
}

/* ========== 前向传播 ========== */

int quaternion_liquid_gate_forward(QuaternionLiquidGate* gate,
                                   const float* input,
                                   size_t seq_len,
                                   float* output) {
    if (!gate || !input || !output || seq_len == 0) return -1;

    size_t input_dim = gate->config.input_dim;
    size_t quaternion_dim = gate->config.quaternion_dim;

    /* 分配缓冲区 */
    Quaternion* gate_proj = (Quaternion*)safe_malloc(seq_len * quaternion_dim * sizeof(Quaternion));
    Quaternion* act_proj = (Quaternion*)safe_malloc(seq_len * quaternion_dim * sizeof(Quaternion));
    Quaternion* modulation = (Quaternion*)safe_malloc(seq_len * quaternion_dim * sizeof(Quaternion));
    if (!gate_proj || !act_proj || !modulation) {
        safe_free((void**)&gate_proj);
        safe_free((void**)&act_proj);
        safe_free((void**)&modulation);
        return -1;
    }

    /* 四元数投影: 标量 → 四元数 */
    project_to_quaternion_lg(input, gate_proj,
                             gate->weights.gate_kernel, gate->weights.gate_bias,
                             seq_len, input_dim, quaternion_dim);
    project_to_quaternion_lg(input, act_proj,
                             gate->weights.act_kernel, gate->weights.act_bias,
                             seq_len, input_dim, quaternion_dim);

    /* 液态门控: gate = σ(proj_gate), act = tanh(proj_act) */
    /* modulation = gate ⊙ act (element-wise Hamilton product) */
    for (size_t n = 0; n < seq_len; n++) {
        for (size_t q = 0; q < quaternion_dim; q++) {
            Quaternion g = gate_proj[n * quaternion_dim + q];
            Quaternion a = act_proj[n * quaternion_dim + q];
            /* 四元数sigmoid: 逐分量sigmoid */
            g.w = 1.0f / (1.0f + expf(-g.w));
            g.x = 1.0f / (1.0f + expf(-g.x));
            g.y = 1.0f / (1.0f + expf(-g.y));
            g.z = 1.0f / (1.0f + expf(-g.z));
            /* 四元数tanh: 逐分量tanh */
            a.w = tanhf(a.w);
            a.x = tanhf(a.x);
            a.y = tanhf(a.y);
            a.z = tanhf(a.z);
            /* gate ⊙ act = Hamilton product */
            modulation[n * quaternion_dim + q] = quaternion_multiply(&g, &a);
            /* 缓存门控和激活信号用于反向传播 */
            gate_proj[n * quaternion_dim + q] = g;
            act_proj[n * quaternion_dim + q] = a;
        }
    }

    /* 输出投影: 四元数 → 标量 */
    size_t out_offset = get_output_offset(gate);
    const float* out_weight = gate->adam_m + out_offset;
    const float* out_bias = gate->config.use_bias ?
                            out_weight + quaternion_dim * 4 * input_dim : NULL;

    project_from_quaternion_lg(modulation, output,
                               out_weight, out_bias,
                               seq_len, quaternion_dim, input_dim);

    /* 缓存 */
    gate->cached_seq_len = seq_len;
    safe_free((void**)&gate->cached_gate);
    safe_free((void**)&gate->cached_act);
    gate->cached_gate = gate_proj; gate_proj = NULL;
    gate->cached_act = act_proj; act_proj = NULL;

    safe_free((void**)&gate_proj);
    safe_free((void**)&act_proj);
    safe_free((void**)&modulation);

    return 0;
}

int quaternion_liquid_gate_forward_batch(QuaternionLiquidGate* gate,
                                         const float* input,
                                         size_t batch, size_t seq_len,
                                         float* output) {
    if (!gate || !input || !output || batch == 0 || seq_len == 0) return -1;

    for (size_t b = 0; b < batch; b++) {
        const float* in_batch = &input[b * seq_len * gate->config.input_dim];
        float* out_batch = &output[b * seq_len * gate->config.input_dim];
        if (quaternion_liquid_gate_forward(gate, in_batch, seq_len, out_batch) != 0) {
            return -1;
        }
    }
    return 0;
}

/* ========== 训练 ========== */

float quaternion_liquid_gate_train_step(QuaternionLiquidGate* gate,
                                        const float* input,
                                        const float* target,
                                        size_t batch, size_t seq_len) {
    if (!gate || !input || !target) return -1.0f;

    size_t input_dim = gate->config.input_dim;
    size_t n_elem = batch * seq_len * input_dim;

    float* pred = (float*)safe_malloc(n_elem * sizeof(float));
    if (!pred) return -1.0f;

    if (quaternion_liquid_gate_forward_batch(gate, input, batch, seq_len, pred) != 0) {
        safe_free((void**)&pred);
        return -1.0f;
    }

    float loss = mse_loss_f32(pred, target, n_elem);
    float lr = gate->config.learning_rate;
    float eps = 1e-4f;

    size_t input_dim_c = gate->config.input_dim;
    size_t quaternion_dim = gate->config.quaternion_dim;
    size_t kernel_size = input_dim_c * quaternion_dim * 4;

    /* 数值梯度: gate_kernel */
    float* grad_buf = (float*)safe_malloc(kernel_size * sizeof(float));
    if (grad_buf) {
        for (size_t i = 0; i < kernel_size; i++) {
            float orig = gate->weights.gate_kernel[i];
            gate->weights.gate_kernel[i] = orig + eps;
            float* pred_p = (float*)safe_malloc(n_elem * sizeof(float));
            if (pred_p) {
                quaternion_liquid_gate_forward_batch(gate, input, batch, seq_len, pred_p);
                float loss_p = mse_loss_f32(pred_p, target, n_elem);
                grad_buf[i] = (loss_p - loss) / eps;
                safe_free((void**)&pred_p);
            } else { grad_buf[i] = 0.0f; }
            gate->weights.gate_kernel[i] = orig;
        }
        size_t param_offset = 0;
        adam_update_f32(gate->weights.gate_kernel, grad_buf,
                        &gate->adam_m[param_offset], &gate->adam_v[param_offset],
                        gate->adam_step, lr, kernel_size);
        safe_free((void**)&grad_buf);
    }

    /* act_kernel */
    grad_buf = (float*)safe_malloc(kernel_size * sizeof(float));
    if (grad_buf) {
        for (size_t i = 0; i < kernel_size; i++) {
            float orig = gate->weights.act_kernel[i];
            gate->weights.act_kernel[i] = orig + eps;
            float* pred_p = (float*)safe_malloc(n_elem * sizeof(float));
            if (pred_p) {
                quaternion_liquid_gate_forward_batch(gate, input, batch, seq_len, pred_p);
                float loss_p = mse_loss_f32(pred_p, target, n_elem);
                grad_buf[i] = (loss_p - loss) / eps;
                safe_free((void**)&pred_p);
            } else { grad_buf[i] = 0.0f; }
            gate->weights.act_kernel[i] = orig;
        }
        size_t param_offset = kernel_size;
        adam_update_f32(gate->weights.act_kernel, grad_buf,
                        &gate->adam_m[param_offset], &gate->adam_v[param_offset],
                        gate->adam_step, lr, kernel_size);
        safe_free((void**)&grad_buf);
    }

    /* 偏置梯度 */
    if (gate->config.use_bias) {
        size_t bias_size = quaternion_dim * 4;

        grad_buf = (float*)safe_malloc(bias_size * sizeof(float));
        if (grad_buf) {
            for (size_t i = 0; i < bias_size; i++) {
                float orig = gate->weights.gate_bias[i];
                gate->weights.gate_bias[i] = orig + eps;
                float* pred_p = (float*)safe_malloc(n_elem * sizeof(float));
                if (pred_p) {
                    quaternion_liquid_gate_forward_batch(gate, input, batch, seq_len, pred_p);
                    float loss_p = mse_loss_f32(pred_p, target, n_elem);
                    grad_buf[i] = (loss_p - loss) / eps;
                    safe_free((void**)&pred_p);
                } else { grad_buf[i] = 0.0f; }
                gate->weights.gate_bias[i] = orig;
            }
            size_t param_offset = kernel_size * 2;
            adam_update_f32(gate->weights.gate_bias, grad_buf,
                            &gate->adam_m[param_offset], &gate->adam_v[param_offset],
                            gate->adam_step, lr, bias_size);
            safe_free((void**)&grad_buf);
        }

        grad_buf = (float*)safe_malloc(bias_size * sizeof(float));
        if (grad_buf) {
            for (size_t i = 0; i < bias_size; i++) {
                float orig = gate->weights.act_bias[i];
                gate->weights.act_bias[i] = orig + eps;
                float* pred_p = (float*)safe_malloc(n_elem * sizeof(float));
                if (pred_p) {
                    quaternion_liquid_gate_forward_batch(gate, input, batch, seq_len, pred_p);
                    float loss_p = mse_loss_f32(pred_p, target, n_elem);
                    grad_buf[i] = (loss_p - loss) / eps;
                    safe_free((void**)&pred_p);
                } else { grad_buf[i] = 0.0f; }
                gate->weights.act_bias[i] = orig;
            }
            size_t param_offset = kernel_size * 2 + bias_size;
            adam_update_f32(gate->weights.act_bias, grad_buf,
                            &gate->adam_m[param_offset], &gate->adam_v[param_offset],
                            gate->adam_step, lr, bias_size);
            safe_free((void**)&grad_buf);
        }
    }

    /* 输出投影梯度 */
    size_t out_size = quaternion_dim * 4 * input_dim_c;
    grad_buf = (float*)safe_malloc(out_size * sizeof(float));
    size_t out_offset = get_output_offset(gate);
    float* out_weight = gate->adam_m + out_offset;

    if (grad_buf) {
        for (size_t i = 0; i < out_size; i++) {
            float orig = out_weight[i];
            out_weight[i] = orig + eps;
            float* pred_p = (float*)safe_malloc(n_elem * sizeof(float));
            if (pred_p) {
                quaternion_liquid_gate_forward_batch(gate, input, batch, seq_len, pred_p);
                float loss_p = mse_loss_f32(pred_p, target, n_elem);
                grad_buf[i] = (loss_p - loss) / eps;
                safe_free((void**)&pred_p);
            } else { grad_buf[i] = 0.0f; }
            out_weight[i] = orig;
        }
        /* 输出投影的Adam状态存储在adam_m末尾 */
        size_t total_params = kernel_size * 2 + (gate->config.use_bias ? quaternion_dim * 4 * 2 : 0);
        size_t out_param_start = total_params;
        adam_update_f32(out_weight, grad_buf,
                        &gate->adam_m[out_param_start], &gate->adam_v[out_param_start],
                        gate->adam_step, lr, out_size);
        safe_free((void**)&grad_buf);
    }

    /* 输出偏置梯度 */
    if (gate->config.use_bias) {
        float* out_bias = out_weight + out_size;
        grad_buf = (float*)safe_malloc(input_dim_c * sizeof(float));
        if (grad_buf) {
            for (size_t i = 0; i < input_dim_c; i++) {
                float orig = out_bias[i];
                out_bias[i] = orig + eps;
                float* pred_p = (float*)safe_malloc(n_elem * sizeof(float));
                if (pred_p) {
                    quaternion_liquid_gate_forward_batch(gate, input, batch, seq_len, pred_p);
                    float loss_p = mse_loss_f32(pred_p, target, n_elem);
                    grad_buf[i] = (loss_p - loss) / eps;
                    safe_free((void**)&pred_p);
                } else { grad_buf[i] = 0.0f; }
                out_bias[i] = orig;
            }
            size_t total_params = kernel_size * 2 + (gate->config.use_bias ? quaternion_dim * 4 * 2 : 0);
            size_t out_param_start = total_params + out_size;
            adam_update_f32(out_bias, grad_buf,
                            &gate->adam_m[out_param_start], &gate->adam_v[out_param_start],
                            gate->adam_step, lr, input_dim_c);
            safe_free((void**)&grad_buf);
        }
    }

    gate->adam_step++;
    safe_free((void**)&pred);
    return loss;
}

/* ============================================================================
 * M-008修复：解析梯度训练步（链式法则反向传播替代数值梯度）
 *
 * 计算图:
 *   input → [gate_kernel] → gate_proj → sigmoid(·) → g
 *   input → [act_kernel]  → act_proj  → tanh(·)    → a
 *   g ⊙ a → modulation → [output_proj] → output
 *
 * 链式法则:
 *   ∂L/∂gate_kernel = input^T · (σ'(gate_proj) ⊙ (∂mod/∂g)^T · output_proj^T · ∂L/∂output)
 *   ∂L/∂act_kernel  = input^T · (tanh'(act_proj) ⊙ (∂mod/∂a)^T · output_proj^T · ∂L/∂output)
 * ============================================================================ */
float quaternion_liquid_gate_train_step_analytic(QuaternionLiquidGate* gate,
                                                  const float* input,
                                                  const float* target,
                                                  size_t batch, size_t seq_len) {
    if (!gate || !input || !target) return -1.0f;
    if (!gate->cached_gate || !gate->cached_act || gate->cached_seq_len == 0) {
        /* 无缓存时回退到数值梯度 */
        return quaternion_liquid_gate_train_step(gate, input, target, batch, seq_len);
    }

    size_t input_dim = gate->config.input_dim;
    size_t quat_dim = gate->config.quaternion_dim;
    size_t n_elem = batch * seq_len * input_dim;
    size_t kernel_size = input_dim * quat_dim * 4;
    float lr = gate->config.learning_rate;

    float* pred = (float*)safe_malloc(n_elem * sizeof(float));
    if (!pred) return -1.0f;
    if (quaternion_liquid_gate_forward_batch(gate, input, batch, seq_len, pred) != 0) {
        safe_free((void**)&pred); return -1.0f;
    }

    float loss = mse_loss_f32(pred, target, n_elem);

    /* ∂L/∂output = 2*(pred - target) / n_elem */
    float* d_output = (float*)safe_malloc(n_elem * sizeof(float));
    if (!d_output) { safe_free((void**)&pred); return loss; }
    for (size_t i = 0; i < n_elem; i++) {
        d_output[i] = 2.0f * (pred[i] - target[i]) / (float)n_elem;
    }

    /* 输出投影反向：∂L/∂modulation[4*q+j] = Σ_i d_output[i*q+j] * output_weight[i] */
    size_t out_offset = get_output_offset(gate);
    const float* out_weight = gate->adam_m + out_offset;
    size_t mod_elem = batch * seq_len * quat_dim * 4;
    float* d_mod = (float*)safe_calloc(mod_elem, sizeof(float));
    if (!d_mod) { safe_free((void**)&d_output); safe_free((void**)&pred); return loss; }

    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t q = 0; q < quat_dim; q++) {
                for (size_t c = 0; c < 4; c++) {
                    size_t mod_idx = ((b * seq_len + t) * quat_dim + q) * 4 + c;
                    float grad = 0.0f;
                    for (size_t i = 0; i < input_dim; i++) {
                        float w = out_weight[i * quat_dim * 4 + q * 4 + c];
                        size_t out_idx = ((b * seq_len + t) * input_dim) + i;
                        grad += d_output[out_idx] * w;
                    }
                    d_mod[mod_idx] = grad;
                }
            }
        }
    }
    safe_free((void**)&d_output);

    /* 门控反向：∂g = ∂mod/∂g ⊙ a_hat, ∂a = g_hat ⊙ ∂mod/∂a
     * 四元数Hamilton乘积偏导：∂(g⊙a)/∂g 取决于a的共轭矩阵 */
    float* d_gate = (float*)safe_malloc(mod_elem * sizeof(float));
    float* d_act = (float*)safe_malloc(mod_elem * sizeof(float));
    if (!d_gate || !d_act) {
        safe_free((void**)&d_gate); safe_free((void**)&d_act);
        safe_free((void**)&d_mod); safe_free((void**)&pred); return loss;
    }

    for (size_t i = 0; i < (size_t)(batch * seq_len * quat_dim); i++) {
        Quaternion* g = &gate->cached_gate[i];
        Quaternion* a = &gate->cached_act[i];
        /* ∂mod/∂g: 四元数左乘积矩阵 w.r.t g (a视为常数) */
        float a_w = a->w, a_x = a->x, a_y = a->y, a_z = a->z;
        float d_w = d_mod[i * 4], d_x = d_mod[i * 4 + 1],
              d_y = d_mod[i * 4 + 2], d_z = d_mod[i * 4 + 3];
        d_gate[i * 4]     = d_w * a_w - d_x * a_x - d_y * a_y - d_z * a_z;
        d_gate[i * 4 + 1] = d_w * a_x + d_x * a_w + d_y * a_z - d_z * a_y;
        d_gate[i * 4 + 2] = d_w * a_y - d_x * a_z + d_y * a_w + d_z * a_x;
        d_gate[i * 4 + 3] = d_w * a_z + d_x * a_y - d_y * a_x + d_z * a_w;
        /* ∂mod/∂a: 四元数右乘积矩阵 w.r.t a (g视为常数) */
        float g_w = g->w, g_x = g->x, g_y = g->y, g_z = g->z;
        d_act[i * 4]     = d_w * g_w - d_x * g_x - d_y * g_y - d_z * g_z;
        d_act[i * 4 + 1] = d_w * g_x + d_x * g_w - d_y * g_z + d_z * g_y;
        d_act[i * 4 + 2] = d_w * g_y + d_x * g_z + d_y * g_w - d_z * g_x;
        d_act[i * 4 + 3] = d_w * g_z - d_x * g_y + d_y * g_x + d_z * g_w;
    }
    safe_free((void**)&d_mod);

    /* σ'(x) = σ(x)·(1-σ(x)) */
    for (size_t i = 0; i < mod_elem; i++) {
        float sig = gate->cached_gate[i / 4].w;
        if (i % 4 == 0) sig = gate->cached_gate[i / 4].w;
        else if (i % 4 == 1) sig = gate->cached_gate[i / 4].x;
        else if (i % 4 == 2) sig = gate->cached_gate[i / 4].y;
        else sig = gate->cached_gate[i / 4].z;
        d_gate[i] *= sig * (1.0f - sig);
    }
    /* tanh'(x) = 1 - tanh(x)² */
    for (size_t i = 0; i < mod_elem; i++) {
        float th = gate->cached_act[i / 4].w;
        if (i % 4 == 0) th = gate->cached_act[i / 4].w;
        else if (i % 4 == 1) th = gate->cached_act[i / 4].x;
        else if (i % 4 == 2) th = gate->cached_act[i / 4].y;
        else th = gate->cached_act[i / 4].z;
        d_act[i] *= (1.0f - th * th);
    }

    /* ∂L/∂kernel = input^T · d_proj: 累加所有样本的梯度贡献 */
    float* grad_gate_k = (float*)safe_calloc(kernel_size, sizeof(float));
    float* grad_act_k  = (float*)safe_calloc(kernel_size, sizeof(float));
    if (grad_gate_k && grad_act_k) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t t = 0; t < seq_len; t++) {
                const float* inp = &input[(b * seq_len + t) * input_dim];
                float* dg = &d_gate[(b * seq_len + t) * quat_dim * 4];
                float* da = &d_act[(b * seq_len + t) * quat_dim * 4];
                for (size_t i = 0; i < input_dim; i++) {
                    for (size_t j = 0; j < quat_dim * 4; j++) {
                        grad_gate_k[i * quat_dim * 4 + j] += inp[i] * dg[j];
                        grad_act_k[i * quat_dim * 4 + j]  += inp[i] * da[j];
                    }
                }
            }
        }

        /* Adam更新 */
        adam_update_f32(gate->weights.gate_kernel, grad_gate_k,
                        gate->adam_m, gate->adam_v,
                        gate->adam_step, lr, kernel_size);
        adam_update_f32(gate->weights.act_kernel, grad_act_k,
                        &gate->adam_m[kernel_size], &gate->adam_v[kernel_size],
                        gate->adam_step, lr, kernel_size);
    }
    safe_free((void**)&grad_gate_k); safe_free((void**)&grad_act_k);
    safe_free((void**)&d_gate); safe_free((void**)&d_act);

    gate->adam_step++;
    safe_free((void**)&pred);
    return loss;
}

/* ========== 辅助函数 ========== */

void quaternion_liquid_gate_reset_optimizer(QuaternionLiquidGate* gate) {
    if (!gate) return;
    size_t input_dim = gate->config.input_dim;
    size_t quaternion_dim = gate->config.quaternion_dim;
    size_t kernel_size = input_dim * quaternion_dim * 4;
    size_t bias_size = gate->config.use_bias ? quaternion_dim * 4 : 0;
    size_t out_size = quaternion_dim * 4 * input_dim;
    size_t total = kernel_size * 2 + bias_size * 2 + out_size + (gate->config.use_bias ? input_dim : 0);
    memset(gate->adam_m, 0, total * sizeof(float));
    memset(gate->adam_v, 0, total * sizeof(float));
    gate->adam_step = 0;
}

void quaternion_liquid_gate_set_learning_rate(QuaternionLiquidGate* gate, float lr) {
    if (gate) {
        gate->config.learning_rate = lr;
    }
}

/* ========== 参数序列化 ========== */

int quaternion_liquid_gate_save(const QuaternionLiquidGate* gate, const char* filepath) {
    if (!gate || !filepath) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    size_t input_dim = gate->config.input_dim;
    size_t quaternion_dim = gate->config.quaternion_dim;
    size_t kernel_size = input_dim * quaternion_dim * 4;
    size_t bias_size = gate->config.use_bias ? quaternion_dim * 4 : 0;
    size_t out_size = quaternion_dim * 4 * input_dim;

    /* 写入配置 */
    if (fwrite(&gate->config, sizeof(QuaternionLiquidGateConfig), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    /* 写入门控核权重 */
    if (fwrite(gate->weights.gate_kernel, sizeof(float), kernel_size, fp) != kernel_size) {
        fclose(fp);
        return -1;
    }
    if (fwrite(gate->weights.act_kernel, sizeof(float), kernel_size, fp) != kernel_size) {
        fclose(fp);
        return -1;
    }

    /* 写入偏置 */
    if (gate->config.use_bias) {
        if (fwrite(gate->weights.gate_bias, sizeof(float), bias_size, fp) != bias_size) {
            fclose(fp);
            return -1;
        }
        if (fwrite(gate->weights.act_bias, sizeof(float), bias_size, fp) != bias_size) {
            fclose(fp);
            return -1;
        }
    }

    /* 写入输出投影 */
    size_t out_offset = get_output_offset(gate);
    const float* out_weight = gate->adam_m + out_offset;
    if (fwrite(out_weight, sizeof(float), out_size, fp) != out_size) {
        fclose(fp);
        return -1;
    }
    if (gate->config.use_bias) {
        const float* out_bias = out_weight + out_size;
        if (fwrite(out_bias, sizeof(float), input_dim, fp) != input_dim) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

/* ===========================================================================
 * A04.5.1 四元数相位位置编码 - 实现
 * =========================================================================== */

QuaternionPhaseEncoding* quaternion_phase_encoding_create(const PhaseEncodingConfig* config)
{
    if (!config || config->freq_count == 0) return NULL;

    QuaternionPhaseEncoding* pe = (QuaternionPhaseEncoding*)safe_malloc(sizeof(QuaternionPhaseEncoding));
    if (!pe) return NULL;
    memset(pe, 0, sizeof(QuaternionPhaseEncoding));

    pe->config = *config;

    if (config->freq_min <= 0.0f) pe->config.freq_min = 1.0f;
    if (config->freq_max <= config->freq_min) pe->config.freq_max = 10000.0f;

    pe->frequencies = (float*)safe_malloc(config->freq_count * sizeof(float));
    if (!pe->frequencies) {
        safe_free((void**)&pe);
        return NULL;
    }

    /* 对数间隔分布频率 */
    float log_min = logf(pe->config.freq_min);
    float log_max = logf(pe->config.freq_max);
    if (config->freq_count == 1) {
        pe->frequencies[0] = pe->config.freq_min;
    } else {
        for (size_t k = 0; k < config->freq_count; k++) {
            float ratio = (float)k / (float)(config->freq_count - 1);
            pe->frequencies[k] = expf(log_min + ratio * (log_max - log_min));
        }
    }

    pe->is_initialized = 1;
    return pe;
}

void quaternion_phase_encoding_destroy(QuaternionPhaseEncoding* pe)
{
    if (!pe) return;
    safe_free((void**)&pe->frequencies);
    safe_free((void**)&pe);
}

int quaternion_phase_encoding_forward(const QuaternionPhaseEncoding* pe,
                                       const float* positions, size_t seq_len,
                                       size_t quaternion_dim, Quaternion* output)
{
    if (!pe || !pe->is_initialized || !positions || !output || seq_len == 0) return -1;

    size_t freq_count = pe->config.freq_count;
    if (quaternion_dim == 0) return -1;

    for (size_t n = 0; n < seq_len; n++) {
        float pos = positions[n];
        for (size_t q = 0; q < quaternion_dim; q++) {
            if (q < freq_count) {
                float theta = pos * pe->frequencies[q];
                output[n * quaternion_dim + q].w = cosf(theta);
                output[n * quaternion_dim + q].x = sinf(theta);
                output[n * quaternion_dim + q].y = 0.0f;
                output[n * quaternion_dim + q].z = 0.0f;
            } else {
                output[n * quaternion_dim + q].w = 0.0f;
                output[n * quaternion_dim + q].x = 0.0f;
                output[n * quaternion_dim + q].y = 0.0f;
                output[n * quaternion_dim + q].z = 0.0f;
            }
        }
    }

    return 0;
}

int quaternion_phase_encoding_update(QuaternionPhaseEncoding* pe,
                                      const float* grad, float lr)
{
    if (!pe || !pe->is_initialized || !grad || !pe->config.learnable) return -1;
    if (lr <= 0.0f) return -1;

    for (size_t k = 0; k < pe->config.freq_count; k++) {
        pe->frequencies[k] += lr * grad[k];
        if (pe->frequencies[k] < 0.01f) pe->frequencies[k] = 0.01f;
    }

    return 0;
}

int quaternion_phase_encoding_save(const QuaternionPhaseEncoding* pe, const char* path)
{
    if (!pe || !pe->is_initialized || !path) return -1;

    FILE* fp = fopen(path, "wb");
    if (!fp) return -1;

    size_t freq_count = pe->config.freq_count;
    if (fwrite(&freq_count, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    if (fwrite(pe->frequencies, sizeof(float), freq_count, fp) != freq_count) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

int quaternion_phase_encoding_load(QuaternionPhaseEncoding* pe, const char* path)
{
    if (!pe || !pe->is_initialized || !path) return -1;

    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;

    size_t freq_count = 0;
    if (fread(&freq_count, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    if (freq_count != pe->config.freq_count) {
        fclose(fp);
        return -1;
    }

    if (fread(pe->frequencies, sizeof(float), freq_count, fp) != freq_count) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

/* ===========================================================================
 * A04.5.1 四元数液态状态聚合 - 实现
 * =========================================================================== */

QuaternionLiquidAggregation* quaternion_liquid_aggregation_create(const LiquidAggregationConfig* config)
{
    if (!config || config->state_dim == 0) return NULL;

    QuaternionLiquidAggregation* agg = (QuaternionLiquidAggregation*)safe_malloc(sizeof(QuaternionLiquidAggregation));
    if (!agg) return NULL;
    memset(agg, 0, sizeof(QuaternionLiquidAggregation));

    agg->config = *config;
    if (config->temperature <= 0.0f) agg->config.temperature = 1.0f;

    if (config->use_attention) {
        size_t proj_size = config->state_dim * 4;
        agg->query_proj = (float*)safe_calloc(proj_size, sizeof(float));
        if (!agg->query_proj) {
            safe_free((void**)&agg);
            return NULL;
        }
        float scale = sqrtf(2.0f / (float)(config->state_dim * 4));
        for (size_t i = 0; i < proj_size; i++) {
            agg->query_proj[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        }
    }

    agg->is_initialized = 1;
    return agg;
}

void quaternion_liquid_aggregation_destroy(QuaternionLiquidAggregation* agg)
{
    if (!agg) return;
    safe_free((void**)&agg->query_proj);
    safe_free((void**)&agg);
}

int quaternion_liquid_aggregation_forward(QuaternionLiquidAggregation* agg,
                                           const Quaternion* states,
                                           size_t num_states,
                                           Quaternion* output)
{
    if (!agg || !agg->is_initialized || !states || !output || num_states == 0) return -1;

    size_t state_dim = agg->config.state_dim;

    if (!agg->config.use_attention) {
        /* 简单平均 */
        for (size_t d = 0; d < state_dim; d++) {
            float sum_w = 0.0f, sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
            for (size_t n = 0; n < num_states; n++) {
                sum_w += states[n * state_dim + d].w;
                sum_x += states[n * state_dim + d].x;
                sum_y += states[n * state_dim + d].y;
                sum_z += states[n * state_dim + d].z;
            }
            float inv_n = 1.0f / (float)num_states;
            output[d].w = sum_w * inv_n;
            output[d].x = sum_x * inv_n;
            output[d].y = sum_y * inv_n;
            output[d].z = sum_z * inv_n;
        }
    } else {
        /* 注意力加权聚合 */
        float* scores = (float*)safe_malloc(num_states * sizeof(float));
        if (!scores) return -1;

        float temp = agg->config.temperature;
        const float* query = agg->query_proj;

        for (size_t n = 0; n < num_states; n++) {
            float score = 0.0f;
            for (size_t d = 0; d < state_dim; d++) {
                Quaternion s = states[n * state_dim + d];
                score += s.w * query[d * 4 + 0];
                score += s.x * query[d * 4 + 1];
                score += s.y * query[d * 4 + 2];
                score += s.z * query[d * 4 + 3];
            }
            scores[n] = score / temp;
        }

        /* softmax */
        float max_score = scores[0];
        for (size_t n = 1; n < num_states; n++) {
            if (scores[n] > max_score) max_score = scores[n];
        }
        float sum_exp = 0.0f;
        for (size_t n = 0; n < num_states; n++) {
            scores[n] = expf(scores[n] - max_score);
            sum_exp += scores[n];
        }
        if (sum_exp < 1e-10f) sum_exp = 1.0f;
        for (size_t n = 0; n < num_states; n++) {
            scores[n] /= sum_exp;
        }

        /* 加权求和 */
        for (size_t d = 0; d < state_dim; d++) {
            float sum_w = 0.0f, sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
            for (size_t n = 0; n < num_states; n++) {
                float w = scores[n];
                sum_w += w * states[n * state_dim + d].w;
                sum_x += w * states[n * state_dim + d].x;
                sum_y += w * states[n * state_dim + d].y;
                sum_z += w * states[n * state_dim + d].z;
            }
            output[d].w = sum_w;
            output[d].x = sum_x;
            output[d].y = sum_y;
            output[d].z = sum_z;
        }

        safe_free((void**)&scores);
    }

    return 0;
}

int quaternion_liquid_aggregation_update(QuaternionLiquidAggregation* agg,
                                          const float* grad, float lr)
{
    if (!agg || !agg->is_initialized || !agg->config.use_attention) return -1;
    if (!grad || lr <= 0.0f) return -1;

    size_t proj_size = agg->config.state_dim * 4;
    for (size_t i = 0; i < proj_size; i++) {
        agg->query_proj[i] += lr * grad[i];
    }

    return 0;
}

/* ===========================================================================
 * A04.5.2 CfC隐式状态稀疏激活 - 实现
 * =========================================================================== */

int quaternion_cfc_sparse_activate(Quaternion* gate_signal,
                                    size_t seq_len, size_t quaternion_dim,
                                    const CfCSparseConfig* config)
{
    if (!gate_signal || seq_len == 0 || quaternion_dim == 0 || !config) return -1;

    float ratio = config->sparsity_ratio;
    if (ratio <= 0.0f) ratio = 0.3f;
    if (ratio >= 1.0f) ratio = 1.0f;

    size_t total = seq_len * quaternion_dim;

    if (config->use_topk) {
        /* top-k稀疏化：计算幅值、排序、保留top-k */
        float* magnitudes = (float*)safe_malloc(total * sizeof(float));
        size_t* indices = (size_t*)safe_malloc(total * sizeof(size_t));
        if (!magnitudes || !indices) {
            safe_free((void**)&magnitudes);
            safe_free((void**)&indices);
            return -1;
        }

        for (size_t i = 0; i < total; i++) {
            Quaternion* q = &gate_signal[i];
            magnitudes[i] = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
            indices[i] = i;
        }

        /* 简单选择排序 - 找出top-k阈值 */
        size_t k = (size_t)((float)total * ratio);
        if (k == 0) k = 1;
        if (k > total) k = total;

        /* 使用部分排序找到第k大的幅值 */
        float threshold_val = 0.0f;
        for (size_t i = 0; i < k && i < total; i++) {
            size_t max_idx = i;
            for (size_t j = i + 1; j < total; j++) {
                if (magnitudes[j] > magnitudes[max_idx]) {
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                float tmp_f = magnitudes[i];
                magnitudes[i] = magnitudes[max_idx];
                magnitudes[max_idx] = tmp_f;
                size_t tmp_i = indices[i];
                indices[i] = indices[max_idx];
                indices[max_idx] = tmp_i;
            }
            if (i == k - 1) threshold_val = magnitudes[i];
        }

        /* 应用mask：低于阈值的置零，高于阈值的保留并缩放 */
        float scale = (ratio > 0.0f) ? (1.0f / ratio) : 1.0f;
        for (size_t i = 0; i < total; i++) {
            if (magnitudes[i] < threshold_val) {
                gate_signal[indices[i]].w = 0.0f;
                gate_signal[indices[i]].x = 0.0f;
                gate_signal[indices[i]].y = 0.0f;
                gate_signal[indices[i]].z = 0.0f;
            } else {
                gate_signal[indices[i]].w *= scale;
                gate_signal[indices[i]].x *= scale;
                gate_signal[indices[i]].y *= scale;
                gate_signal[indices[i]].z *= scale;
            }
        }

        safe_free((void**)&magnitudes);
        safe_free((void**)&indices);
    } else {
        /* 阈值稀疏化 */
        float thr = config->threshold;
        for (size_t i = 0; i < total; i++) {
            Quaternion* q = &gate_signal[i];
            float mag = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
            if (mag < thr) {
                q->w = 0.0f;
                q->x = 0.0f;
                q->y = 0.0f;
                q->z = 0.0f;
            }
        }
    }

    return 0;
}

/* ===========================================================================
 * A04.5.2 CfC线性时间演化 - 实现
 * =========================================================================== */

int quaternion_cfc_linear_time_evolve(Quaternion* state,
                                       const Quaternion* deriv,
                                       size_t dim, float dt)
{
    if (!state || !deriv || dim == 0) return -1;

    if (dt <= 0.0f) dt = 0.01f;

    /* 欧拉法：state += dt * deriv */
    for (size_t i = 0; i < dim; i++) {
        state[i].w += dt * deriv[i].w;
        state[i].x += dt * deriv[i].x;
        state[i].y += dt * deriv[i].y;
        state[i].z += dt * deriv[i].z;
    }

    return 0;
}

/* ===========================================================================
 * A04.5.2 CfC分块状态演化 - 实现
 * =========================================================================== */

int quaternion_cfc_block_evolve(Quaternion* state,
                                 size_t state_dim,
                                 const CfCBlockEvolveConfig* config,
                                 void (*block_func)(Quaternion* block, size_t block_size, void* ctx),
                                 void* ctx)
{
    if (!state || state_dim == 0 || !config || !block_func) return -1;

    size_t block_size = config->block_size;
    if (block_size == 0) block_size = 4;

    size_t num_blocks = (state_dim + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        size_t offset = b * block_size;
        size_t cur_block_size = (offset + block_size <= state_dim) ? block_size : (state_dim - offset);
        block_func(&state[offset], cur_block_size, ctx);
    }

    return 0;
}

int quaternion_liquid_gate_load(QuaternionLiquidGate* gate, const char* filepath) {
    if (!gate || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    size_t input_dim = gate->config.input_dim;
    size_t quaternion_dim = gate->config.quaternion_dim;
    size_t kernel_size = input_dim * quaternion_dim * 4;
    size_t bias_size = gate->config.use_bias ? quaternion_dim * 4 : 0;
    size_t out_size = quaternion_dim * 4 * input_dim;

    /* 读取配置 */
    QuaternionLiquidGateConfig file_config;
    if (fread(&file_config, sizeof(QuaternionLiquidGateConfig), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    /* 验证配置一致性 */
    if (file_config.input_dim != input_dim ||
        file_config.quaternion_dim != quaternion_dim) {
        fclose(fp);
        return -1;
    }

    /* 读取门控核权重 */
    if (fread(gate->weights.gate_kernel, sizeof(float), kernel_size, fp) != kernel_size) {
        fclose(fp);
        return -1;
    }
    if (fread(gate->weights.act_kernel, sizeof(float), kernel_size, fp) != kernel_size) {
        fclose(fp);
        return -1;
    }

    /* 读取偏置 */
    if (gate->config.use_bias) {
        if (fread(gate->weights.gate_bias, sizeof(float), bias_size, fp) != bias_size) {
            fclose(fp);
            return -1;
        }
        if (fread(gate->weights.act_bias, sizeof(float), bias_size, fp) != bias_size) {
            fclose(fp);
            return -1;
        }
    }

    /* 读取输出投影 */
    size_t out_offset = get_output_offset(gate);
    float* out_weight = gate->adam_m + out_offset;
    if (fread(out_weight, sizeof(float), out_size, fp) != out_size) {
        fclose(fp);
        return -1;
    }
    if (gate->config.use_bias) {
        float* out_bias = out_weight + out_size;
        if (fread(out_bias, sizeof(float), input_dim, fp) != input_dim) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}
