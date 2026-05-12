/**
 * @file liquid_vision.c
 * @brief 液态视觉处理组件实现
 *
 * 基于CfC（Closed-form Continuous-time）液态神经网络的视觉处理。
 * 所有视觉处理使用单一微分方程框架：
 * τ dh/dt = -h + σ(W_gx·x + W_gh·h + b_g) ⊙ tanh(W_ax·x + W_ah·h + b_a)
 * 不引入任何Transformer、注意力机制或独立处理器。
 */

#include "selflnn/multimodal/liquid_vision.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

/* ============ 内部辅助函数 ============ */

/* CfC闭式解核心函数
 * τ dh/dt = -h + σ(W_gx·x + W_gh·h + b_g) ⊙ tanh(W_ax·x + W_ah·h + b_a)
 * 闭式解: h(t+Δt) = h(t)·exp(-Δt/τ) + (1-exp(-Δt/τ))·[σ(...) ⊙ tanh(...)]
 */
static void _liquid_cfc_step(const float* input, int input_size,
                              const float* prev_state, int hidden_size,
                              const float* w_gate, const float* w_activation,
                              const float* h_to_gate, const float* h_to_activation,
                              const float* gate_bias, const float* act_bias,
                              const float* time_constants, float delta_t,
                              float min_tau, float max_tau,
                              int use_adaptive_tau, float* output) {
    for (int i = 0; i < hidden_size; i++) {
        float gate_sum = gate_bias ? gate_bias[i] : 0.0f;
        float act_sum = act_bias ? act_bias[i] : 0.0f;

        for (int j = 0; j < input_size; j++) {
            int idx = i * input_size + j;
            gate_sum += w_gate[idx] * input[j];
            act_sum += w_activation[idx] * input[j];
        }

        for (int j = 0; j < hidden_size; j++) {
            int idx = i * hidden_size + j;
            gate_sum += h_to_gate[idx] * prev_state[j];
            act_sum += h_to_activation[idx] * prev_state[j];
        }

        float gate = gate_sum > 10.0f ? 1.0f :
                     (gate_sum < -10.0f ? 0.0f : 1.0f / (1.0f + expf(-gate_sum)));

        float activation = act_sum > 10.0f ? 1.0f :
                           (act_sum < -10.0f ? -1.0f : tanhf(act_sum));

        float driver = gate * activation;

        float tau = use_adaptive_tau && time_constants ? time_constants[i] : 0.1f;
        if (tau < min_tau) tau = min_tau;
        if (tau > max_tau) tau = max_tau;

        float dt_over_tau = delta_t / tau;
        if (dt_over_tau < 1e-8f) dt_over_tau = 1e-8f;
        if (dt_over_tau > 100.0f) dt_over_tau = 100.0f;

        float exp_term;
        if (dt_over_tau > 20.0f) {
            exp_term = 0.0f;
        } else if (dt_over_tau < 1e-4f) {
            exp_term = 1.0f - dt_over_tau;
        } else {
            exp_term = expf(-dt_over_tau);
        }

        float one_minus_exp = dt_over_tau > 20.0f ? 1.0f :
                              (dt_over_tau < 1e-4f ? dt_over_tau : 1.0f - exp_term);

        float new_state = prev_state[i] * exp_term + one_minus_exp * driver;
        if (isnan(new_state) || isinf(new_state)) new_state = 0.0f;

        output[i] = new_state;
    }
}

/* CfC液态步反向传播
 * 计算损失对输入和参数的梯度
 * dL/dx, dL/dh_prev, dL/dW_gate, dL/dW_act, dL/dH_gate, dL/dH_act,
 * dL/db_gate, dL/db_act, dL/dtau
 */
static void _liquid_cfc_backward_step(
    const float* input, int input_size,
    const float* prev_state, int hidden_size,
    const float* w_gate, const float* w_activation,
    const float* h_to_gate, const float* h_to_activation,
    const float* gate_bias, const float* act_bias,
    const float* time_constants, float delta_t,
    float min_tau, float max_tau, int use_adaptive_tau,
    const float* dL_dh,  /* 损失对输出的梯度 */
    float* dL_dx,        /* 输出: 损失对输入的梯度 */
    float* dL_dh_prev,   /* 输出: 损失对前状态的梯度 */
    float* dL_dW_gate,   /* 输出: 损失对门权重梯度 */
    float* dL_dW_act,    /* 输出: 损失对激活权重梯度 */
    float* dL_dH_gate,   /* 输出: 损失对隐藏门权重梯度 */
    float* dL_dH_act,    /* 输出: 损失对隐藏激活权重梯度 */
    float* dL_db_gate,   /* 输出: 损失对门偏置梯度 */
    float* dL_db_act,    /* 输出: 损失对激活偏置梯度 */
    float* dL_dtau       /* 输出: 损失对时间常数梯度 */
) {
    /* 防御性检查：核心参数为空则直接返回 */
    if (!w_gate || !w_activation || !h_to_gate || !h_to_activation || !dL_dh) return;
    if (hidden_size <= 0 || input_size <= 0) return;
    if (delta_t <= 0.0f) return;
    for (int i = 0; i < hidden_size; i++) {
        float gate_sum = gate_bias ? gate_bias[i] : 0.0f;
        float act_sum = act_bias ? act_bias[i] : 0.0f;

        for (int j = 0; j < input_size; j++) {
            int idx = i * input_size + j;
            gate_sum += w_gate[idx] * input[j];
            act_sum += w_activation[idx] * input[j];
        }

        for (int j = 0; j < hidden_size; j++) {
            int idx = i * hidden_size + j;
            gate_sum += h_to_gate[idx] * prev_state[j];
            act_sum += h_to_activation[idx] * prev_state[j];
        }

        /* 前向值（与_cfc_step保持一致） */
        float gate = gate_sum > 10.0f ? 1.0f :
                     (gate_sum < -10.0f ? 0.0f : 1.0f / (1.0f + expf(-gate_sum)));
        float activation = act_sum > 10.0f ? 1.0f :
                           (act_sum < -10.0f ? -1.0f : tanhf(act_sum));
        float driver = gate * activation;

        float tau = use_adaptive_tau && time_constants ? time_constants[i] : 0.1f;
        if (tau < min_tau) tau = min_tau;
        if (tau > max_tau) tau = max_tau;

        float dt_over_tau = delta_t / tau;
        if (dt_over_tau < 1e-8f) dt_over_tau = 1e-8f;
        if (dt_over_tau > 100.0f) dt_over_tau = 100.0f;

        float exp_term, dexp_dtau;
        if (dt_over_tau > 20.0f) {
            exp_term = 0.0f;
            dexp_dtau = 0.0f;
        } else if (dt_over_tau < 1e-4f) {
            exp_term = 1.0f - dt_over_tau;
            dexp_dtau = -delta_t / (tau * tau);
        } else {
            exp_term = expf(-dt_over_tau);
            dexp_dtau = exp_term * (delta_t / (tau * tau));
        }

        float one_minus_exp, done_dtau;
        if (dt_over_tau > 20.0f) {
            one_minus_exp = 1.0f;
            done_dtau = 0.0f;
        } else if (dt_over_tau < 1e-4f) {
            one_minus_exp = dt_over_tau;
            done_dtau = -delta_t / (tau * tau);
        } else {
            one_minus_exp = 1.0f - exp_term;
            done_dtau = -dexp_dtau;
        }

        /* dh/dgate = one_minus_exp * activation
         * dh/dactivation = one_minus_exp * gate
         * dgate/dgate_sum = gate * (1 - gate)
         * dactivation/dact_sum = 1 - tanh^2(act_sum) = 1 - activation^2 */
        float dh_dgate = one_minus_exp * activation;
        float dh_dact = one_minus_exp * gate;
        float dsig = gate * (1.0f - gate);
        float dtanh = 1.0f - activation * activation;

        /* 链路法则：dL/d(gate_sum) = dL/dh * dh/dgate * dsig
         * dL/d(act_sum) = dL/dh * dh/dact * dtanh */
        float dL_dgate_sum = dL_dh[i] * dh_dgate * dsig;
        float dL_dact_sum = dL_dh[i] * dh_dact * dtanh;

        /* 输入梯度 */
        if (dL_dx) {
            for (int j = 0; j < input_size; j++) {
                int idx = i * input_size + j;
                dL_dx[j] += dL_dgate_sum * w_gate[idx] + dL_dact_sum * w_activation[idx];
            }
        }

        /* 前状态梯度 */
        if (dL_dh_prev) {
            /* dL/dh_prev = dL/dh * exp_term + Σ(hidden权重贡献) */
            dL_dh_prev[i] += dL_dh[i] * exp_term;
            for (int j = 0; j < hidden_size; j++) {
                int idx = i * hidden_size + j;
                dL_dh_prev[j] += dL_dgate_sum * h_to_gate[idx] + dL_dact_sum * h_to_activation[idx];
            }
        }

        /* 参数梯度 */
        if (dL_dW_gate) {
            for (int j = 0; j < input_size; j++) {
                int idx = i * input_size + j;
                dL_dW_gate[idx] += dL_dgate_sum * input[j];
            }
        }
        if (dL_dW_act) {
            for (int j = 0; j < input_size; j++) {
                int idx = i * input_size + j;
                dL_dW_act[idx] += dL_dact_sum * input[j];
            }
        }
        if (dL_dH_gate) {
            for (int j = 0; j < hidden_size; j++) {
                int idx = i * hidden_size + j;
                dL_dH_gate[idx] += dL_dgate_sum * prev_state[j];
            }
        }
        if (dL_dH_act) {
            for (int j = 0; j < hidden_size; j++) {
                int idx = i * hidden_size + j;
                dL_dH_act[idx] += dL_dact_sum * prev_state[j];
            }
        }
        if (dL_db_gate) {
            dL_db_gate[i] += dL_dgate_sum;
        }
        if (dL_db_act) {
            dL_db_act[i] += dL_dact_sum;
        }

        /* 时间常数梯度 */
        if (dL_dtau && use_adaptive_tau) {
            float dh_dtau = prev_state[i] * dexp_dtau + driver * done_dtau;
            dL_dtau[i] += dL_dh[i] * dh_dtau;
        }
    }
}

/* 从图像提取补丁
 * 输出: [num_patches x patch_size x patch_size x channels]
 */
static int _extract_patches(const float* image, int width, int height, int channels,
                             int patch_size, int stride,
                             float* patches, int max_patches,
                             int* out_num_patches_h, int* out_num_patches_w) {
    int num_h = (height - patch_size) / stride + 1;
    int num_w = (width - patch_size) / stride + 1;
    if (num_h < 1) num_h = 1;
    if (num_w < 1) num_w = 1;

    int total = num_h * num_w;
    if (total > max_patches) total = max_patches;

    int patch_plane = patch_size * patch_size;
    int patch_elem = patch_plane * channels;

    int patch_idx = 0;
    for (int ph = 0; ph < num_h && patch_idx < total; ph++) {
        for (int pw = 0; pw < num_w && patch_idx < total; pw++) {
            int y_start = ph * stride;
            int x_start = pw * stride;
            float* dst = patches + (size_t)patch_idx * patch_elem;

            for (int c = 0; c < channels; c++) {
                for (int py = 0; py < patch_size; py++) {
                    for (int px = 0; px < patch_size; px++) {
                        int src_y = y_start + py;
                        int src_x = x_start + px;
                        if (src_y >= height) src_y = height - 1;
                        if (src_x >= width) src_x = width - 1;
                        dst[c * patch_plane + py * patch_size + px] =
                            image[c * height * width + src_y * width + src_x];
                    }
                }
            }
            patch_idx++;
        }
    }

    *out_num_patches_h = num_h;
    *out_num_patches_w = num_w;
    return patch_idx;
}

/* Xavier均匀初始化 */
static void _xavier_init(float* data, int rows, int cols, unsigned int* seed) {
    float scale = sqrtf(6.0f / (float)(rows + cols));
    for (int i = 0; i < rows * cols; i++) {
        *seed = *seed * 1103515245 + 12345;
        float r = (float)((*seed >> 16) & 0x7FFF) / 32767.0f;
        data[i] = (r * 2.0f - 1.0f) * scale;
    }
}

/* 常量初始化 */
static void _const_init(float* data, int n, float val) {
    for (int i = 0; i < n; i++) data[i] = val;
}

/* ============ LiquidPatchEncoder 实现 ============ */

struct LiquidPatchEncoder {
    LiquidPatchEncoderConfig config;
    float* patch_w_gate;
    float* patch_w_act;
    float* patch_h_to_gate;
    float* patch_h_to_act;
    float* patch_gate_bias;
    float* patch_act_bias;
    float* time_constants;
    float* patch_state;
    float* cached_patches;
    int cached_patches_count;
    int initialized;
    unsigned int seed;
};

LiquidPatchEncoderConfig liquid_patch_encoder_get_default_config(void) {
    LiquidPatchEncoderConfig cfg;
    cfg.patch_size = 16;
    cfg.stride = 16;
    cfg.max_patches = 256;
    cfg.input_channels = 3;
    cfg.patch_hidden_dim = 64;
    cfg.time_constant = 0.1f;
    cfg.delta_t = 0.05f;
    cfg.use_adaptive_tau = 1;
    cfg.min_tau = 0.01f;
    cfg.max_tau = 1.0f;
    cfg.enable_noise = 0;
    cfg.noise_std = 0.01f;
    return cfg;
}

LiquidPatchEncoder* liquid_patch_encoder_create(const LiquidPatchEncoderConfig* config) {
    if (!config) return NULL;

    LiquidPatchEncoder* enc = (LiquidPatchEncoder*)safe_malloc(sizeof(LiquidPatchEncoder));
    if (!enc) return NULL;

    memcpy(&enc->config, config, sizeof(LiquidPatchEncoderConfig));

    int D = config->patch_hidden_dim;
    int patch_size = config->patch_size;
    int channels = config->input_channels;
    int input_dim = patch_size * patch_size * channels;

    enc->patch_w_gate = (float*)safe_calloc((size_t)D * input_dim, sizeof(float));
    enc->patch_w_act = (float*)safe_calloc((size_t)D * input_dim, sizeof(float));
    enc->patch_h_to_gate = (float*)safe_calloc((size_t)D * D, sizeof(float));
    enc->patch_h_to_act = (float*)safe_calloc((size_t)D * D, sizeof(float));
    enc->patch_gate_bias = (float*)safe_calloc(D, sizeof(float));
    enc->patch_act_bias = (float*)safe_calloc(D, sizeof(float));
    enc->time_constants = (float*)safe_calloc(D, sizeof(float));
    enc->patch_state = (float*)safe_calloc((size_t)D, sizeof(float));
    enc->cached_patches = NULL;
    enc->cached_patches_count = 0;

    if (!enc->patch_w_gate || !enc->patch_w_act ||
        !enc->patch_h_to_gate || !enc->patch_h_to_act ||
        !enc->patch_gate_bias || !enc->patch_act_bias ||
        !enc->time_constants || !enc->patch_state) {
        liquid_patch_encoder_free(enc);
        return NULL;
    }

    enc->seed = 42;
    _xavier_init(enc->patch_w_gate, D, input_dim, &enc->seed);
    _xavier_init(enc->patch_w_act, D, input_dim, &enc->seed);
    _xavier_init(enc->patch_h_to_gate, D, D, &enc->seed);
    _xavier_init(enc->patch_h_to_act, D, D, &enc->seed);
    _const_init(enc->patch_gate_bias, D, 0.0f);
    _const_init(enc->patch_act_bias, D, 0.0f);
    _const_init(enc->time_constants, D, config->time_constant);
    _const_init(enc->patch_state, D, 0.0f);

    enc->initialized = 1;
    return enc;
}

void liquid_patch_encoder_free(LiquidPatchEncoder* encoder) {
    if (!encoder) return;
    safe_free((void**)&encoder->patch_w_gate);
    safe_free((void**)&encoder->patch_w_act);
    safe_free((void**)&encoder->patch_h_to_gate);
    safe_free((void**)&encoder->patch_h_to_act);
    safe_free((void**)&encoder->patch_gate_bias);
    safe_free((void**)&encoder->patch_act_bias);
    safe_free((void**)&encoder->time_constants);
    safe_free((void**)&encoder->patch_state);
    safe_free((void**)&encoder->cached_patches);
    safe_free((void**)&encoder);
}

int liquid_patch_encoder_forward(LiquidPatchEncoder* encoder,
                                  int width, int height, int channels,
                                  const float* image_data,
                                  float* patch_embeddings, int max_patches) {
    if (!encoder || !image_data || !patch_embeddings || !encoder->initialized) return -1;

    int patch_size = encoder->config.patch_size;
    int stride = encoder->config.stride;
    int D = encoder->config.patch_hidden_dim;
    float delta_t = encoder->config.delta_t;

    if (channels != encoder->config.input_channels) return -1;

    int num_h, num_w;
    int patch_elem = patch_size * patch_size * channels;
    int max_patches_alloc = encoder->config.max_patches;
    size_t patches_bytes = (size_t)max_patches_alloc * patch_elem * sizeof(float);

    safe_free((void**)&encoder->cached_patches);
    encoder->cached_patches = (float*)safe_malloc(patches_bytes);
    if (!encoder->cached_patches) return -1;

    int num_patches = _extract_patches(image_data, width, height, channels,
                                        patch_size, stride,
                                        encoder->cached_patches, max_patches_alloc,
                                        &num_h, &num_w);

    if (num_patches > max_patches) num_patches = max_patches;

    float* prev_state = (float*)safe_malloc((size_t)D * sizeof(float));
    if (!prev_state) {
        safe_free((void**)&encoder->cached_patches);
        return -1;
    }
    memcpy(prev_state, encoder->patch_state, (size_t)D * sizeof(float));

    for (int p = 0; p < num_patches; p++) {
        _liquid_cfc_step(encoder->cached_patches + (size_t)p * patch_elem, patch_elem,
                          prev_state, D,
                          encoder->patch_w_gate, encoder->patch_w_act,
                          encoder->patch_h_to_gate, encoder->patch_h_to_act,
                          encoder->patch_gate_bias, encoder->patch_act_bias,
                          encoder->time_constants, delta_t,
                          encoder->config.min_tau, encoder->config.max_tau,
                          encoder->config.use_adaptive_tau,
                          patch_embeddings + (size_t)p * D);
        prev_state = patch_embeddings + (size_t)p * D;
    }

    memcpy(encoder->patch_state, patch_embeddings + (size_t)(num_patches - 1) * D,
           (size_t)D * sizeof(float));
    encoder->cached_patches_count = num_patches;

    safe_free((void**)&prev_state);
    return num_patches;
}

/* LiquidPatchEncoder反向传播 */
int liquid_patch_encoder_backward(LiquidPatchEncoder* encoder,
                                   const float* dL_dembeddings, int num_patches,
                                   float* dL_dimage, int image_size,
                                   float learning_rate) {
    if (!encoder || !dL_dembeddings || !encoder->initialized) return -1;

    int D = encoder->config.patch_hidden_dim;
    int patch_size = encoder->config.patch_size;
    int channels = encoder->config.input_channels;
    int patch_elem = patch_size * patch_size * channels;
    float delta_t = encoder->config.delta_t;

    memset(dL_dimage, 0, (size_t)image_size * sizeof(float));

    float* dL_dW_g = (float*)safe_calloc((size_t)D * patch_elem, sizeof(float));
    float* dL_dW_a = (float*)safe_calloc((size_t)D * patch_elem, sizeof(float));
    float* dL_dH_g = (float*)safe_calloc((size_t)D * D, sizeof(float));
    float* dL_dH_a = (float*)safe_calloc((size_t)D * D, sizeof(float));
    float* dL_db_g = (float*)safe_calloc(D, sizeof(float));
    float* dL_db_a = (float*)safe_calloc(D, sizeof(float));
    float* dL_dtau = (float*)safe_calloc(D, sizeof(float));

    if (!dL_dW_g || !dL_dW_a || !dL_dH_g || !dL_dH_a ||
        !dL_db_g || !dL_db_a || !dL_dtau) {
        safe_free((void**)&dL_dW_g); safe_free((void**)&dL_dW_a); safe_free((void**)&dL_dH_g); safe_free((void**)&dL_dH_a);
        safe_free((void**)&dL_db_g); safe_free((void**)&dL_db_a); safe_free((void**)&dL_dtau);
        return -1;
    }

    float* dL_dh_prev = (float*)safe_calloc(D, sizeof(float));
    float* dL_dx_curr = (float*)safe_calloc(patch_elem, sizeof(float));
    if (!dL_dh_prev || !dL_dx_curr) {
        safe_free((void**)&dL_dh_prev); safe_free((void**)&dL_dx_curr);
        safe_free((void**)&dL_dW_g); safe_free((void**)&dL_dW_a); safe_free((void**)&dL_dH_g); safe_free((void**)&dL_dH_a);
        safe_free((void**)&dL_db_g); safe_free((void**)&dL_db_a); safe_free((void**)&dL_dtau);
        return -1;
    }

    /* 反向遍历所有patches */
    for (int p = num_patches - 1; p >= 0; p--) {
        const float* dL_dh = dL_dembeddings + (size_t)p * D;

        memset(dL_dx_curr, 0, (size_t)patch_elem * sizeof(float));
        memset(dL_dh_prev, 0, (size_t)D * sizeof(float));

        /* prev_state: 第0步使用encoder初始状态, 否则使用上一步的输出 */
        const float* prev_state_ptr = (p == 0) ? encoder->patch_state :
                                      dL_dembeddings + (size_t)(p - 1) * D;
        _liquid_cfc_backward_step(
            encoder->cached_patches + (size_t)p * patch_elem, patch_elem,
            prev_state_ptr, D,
            encoder->patch_w_gate, encoder->patch_w_act,
            encoder->patch_h_to_gate, encoder->patch_h_to_act,
            encoder->patch_gate_bias, encoder->patch_act_bias,
            encoder->time_constants, delta_t,
            encoder->config.min_tau, encoder->config.max_tau,
            encoder->config.use_adaptive_tau,
            dL_dh,
            dL_dx_curr, dL_dh_prev,
            dL_dW_g, dL_dW_a, dL_dH_g, dL_dH_a,
            dL_db_g, dL_db_a, dL_dtau);

        /* 梯度裁剪防止爆炸 */
        float grad_norm = 0.0f;
        for (int j = 0; j < patch_elem; j++) grad_norm += dL_dx_curr[j] * dL_dx_curr[j];
        if (grad_norm > 100.0f) {
            float scale = 10.0f / sqrtf(grad_norm);
            for (int j = 0; j < patch_elem; j++) dL_dx_curr[j] *= scale;
        }
    }

    /* 应用梯度更新 */
    float lr = learning_rate;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < patch_elem; j++) {
            int idx = i * patch_elem + j;
            if (fabsf(dL_dW_g[idx]) < 1e-6f) continue;
            encoder->patch_w_gate[idx] -= lr * dL_dW_g[idx];
            encoder->patch_w_act[idx] -= lr * dL_dW_a[idx];
        }
        for (int j = 0; j < D; j++) {
            int idx = i * D + j;
            if (fabsf(dL_dH_g[idx]) < 1e-6f) continue;
            encoder->patch_h_to_gate[idx] -= lr * dL_dH_g[idx];
            encoder->patch_h_to_act[idx] -= lr * dL_dH_a[idx];
        }
        encoder->patch_gate_bias[i] -= lr * dL_db_g[i];
        encoder->patch_act_bias[i] -= lr * dL_db_a[i];
        if (encoder->config.use_adaptive_tau && fabsf(dL_dtau[i]) > 1e-8f) {
            float new_tau = encoder->time_constants[i] - lr * dL_dtau[i];
            if (new_tau >= encoder->config.min_tau && new_tau <= encoder->config.max_tau) {
                encoder->time_constants[i] = new_tau;
            }
        }
    }

    safe_free((void**)&dL_dW_g); safe_free((void**)&dL_dW_a); safe_free((void**)&dL_dH_g); safe_free((void**)&dL_dH_a);
    safe_free((void**)&dL_db_g); safe_free((void**)&dL_db_a); safe_free((void**)&dL_dtau);
    safe_free((void**)&dL_dh_prev); safe_free((void**)&dL_dx_curr);
    return 0;
}

void liquid_patch_encoder_reset(LiquidPatchEncoder* encoder) {
    if (!encoder) return;
    memset(encoder->patch_state, 0, (size_t)encoder->config.patch_hidden_dim * sizeof(float));
}

int liquid_patch_encoder_get_num_patches(const LiquidPatchEncoder* encoder,
                                          int width, int height) {
    if (!encoder) return -1;
    int num_h = (height - encoder->config.patch_size) / encoder->config.stride + 1;
    int num_w = (width - encoder->config.patch_size) / encoder->config.stride + 1;
    if (num_h < 1) num_h = 1;
    if (num_w < 1) num_w = 1;
    int total = num_h * num_w;
    return total > encoder->config.max_patches ? encoder->config.max_patches : total;
}

int liquid_patch_encoder_get_hidden_dim(const LiquidPatchEncoder* encoder) {
    return encoder ? encoder->config.patch_hidden_dim : 0;
}

/* ============ LiquidVisualCfCEvolver 实现 ============ */

struct LiquidVisualCfCEvolver {
    LiquidVisualCfCEvolverConfig config;
    float* w_gate;
    float* w_act;
    float* h_to_gate;
    float* h_to_act;
    float* gate_bias;
    float* act_bias;
    float* time_constants;
    float* state;
    float* channel_states;
    int initialized;
    unsigned int seed;
};

LiquidVisualCfCEvolverConfig liquid_visual_cfc_evolver_get_default_config(void) {
    LiquidVisualCfCEvolverConfig cfg;
    cfg.visual_state_dim = 128;
    cfg.num_visual_channels = 4;
    cfg.base_time_constant = 0.1f;
    cfg.delta_t = 0.05f;
    cfg.use_adaptive_tau = 1;
    cfg.min_tau = 0.01f;
    cfg.max_tau = 2.0f;
    cfg.enable_cross_channel = 1;
    cfg.enable_noise = 0;
    cfg.noise_std = 0.01f;
    return cfg;
}

LiquidVisualCfCEvolver* liquid_visual_cfc_evolver_create(const LiquidVisualCfCEvolverConfig* config) {
    if (!config) return NULL;

    LiquidVisualCfCEvolver* ev = (LiquidVisualCfCEvolver*)safe_malloc(sizeof(LiquidVisualCfCEvolver));
    if (!ev) return NULL;

    memcpy(&ev->config, config, sizeof(LiquidVisualCfCEvolverConfig));

    int D = config->visual_state_dim;
    int NC = config->num_visual_channels;

    /* 每个通道有独立的CfC参数 */
    size_t param_size = (size_t)D * D;

    ev->w_gate = (float*)safe_calloc((size_t)NC * D * D, sizeof(float));
    ev->w_act = (float*)safe_calloc((size_t)NC * D * D, sizeof(float));
    ev->h_to_gate = (float*)safe_calloc((size_t)NC * param_size, sizeof(float));
    ev->h_to_act = (float*)safe_calloc((size_t)NC * param_size, sizeof(float));
    ev->gate_bias = (float*)safe_calloc((size_t)NC * D, sizeof(float));
    ev->act_bias = (float*)safe_calloc((size_t)NC * D, sizeof(float));
    ev->time_constants = (float*)safe_calloc((size_t)D, sizeof(float));
    ev->state = (float*)safe_calloc((size_t)D, sizeof(float));
    ev->channel_states = (float*)safe_calloc((size_t)NC * D, sizeof(float));

    if (!ev->w_gate || !ev->w_act || !ev->h_to_gate || !ev->h_to_act ||
        !ev->gate_bias || !ev->act_bias || !ev->time_constants ||
        !ev->state || !ev->channel_states) {
        liquid_visual_cfc_evolver_free(ev);
        return NULL;
    }

    ev->seed = 123;
    for (int c = 0; c < NC; c++) {
        _xavier_init(ev->w_gate + (size_t)c * D * D, D, D, &ev->seed);
        _xavier_init(ev->w_act + (size_t)c * D * D, D, D, &ev->seed);
        _xavier_init(ev->h_to_gate + (size_t)c * param_size, D, D, &ev->seed);
        _xavier_init(ev->h_to_act + (size_t)c * param_size, D, D, &ev->seed);
    }
    _const_init(ev->gate_bias, NC * D, 0.0f);
    _const_init(ev->act_bias, NC * D, 0.0f);
    _const_init(ev->time_constants, D, config->base_time_constant);

    ev->initialized = 1;
    return ev;
}

void liquid_visual_cfc_evolver_free(LiquidVisualCfCEvolver* evolver) {
    if (!evolver) return;
    safe_free((void**)&evolver->w_gate);
    safe_free((void**)&evolver->w_act);
    safe_free((void**)&evolver->h_to_gate);
    safe_free((void**)&evolver->h_to_act);
    safe_free((void**)&evolver->gate_bias);
    safe_free((void**)&evolver->act_bias);
    safe_free((void**)&evolver->time_constants);
    safe_free((void**)&evolver->state);
    safe_free((void**)&evolver->channel_states);
    safe_free((void**)&evolver);
}

int liquid_visual_cfc_evolver_forward(LiquidVisualCfCEvolver* evolver,
                                       const float* visual_input, int input_dim,
                                       float* visual_state) {
    if (!evolver || !visual_input || !visual_state || !evolver->initialized) return -1;

    int D = evolver->config.visual_state_dim;
    int NC = evolver->config.num_visual_channels;
    float delta_t = evolver->config.delta_t;
    float* ws_state = (float*)safe_malloc((size_t)D * sizeof(float));
    float* ws_prev = (float*)safe_malloc((size_t)D * sizeof(float));
    if (!ws_state || !ws_prev) {
        safe_free((void**)&ws_state);
        safe_free((void**)&ws_prev);
        return -1;
    }

    memcpy(ws_prev, evolver->state, (size_t)D * sizeof(float));

    for (int c = 0; c < NC; c++) {
        _liquid_cfc_step(visual_input, input_dim,
                          ws_prev, D,
                          evolver->w_gate + (size_t)c * D * D,
                          evolver->w_act + (size_t)c * D * D,
                          evolver->h_to_gate + (size_t)c * (size_t)D * D,
                          evolver->h_to_act + (size_t)c * (size_t)D * D,
                          evolver->gate_bias + (size_t)c * D,
                          evolver->act_bias + (size_t)c * D,
                          evolver->time_constants, delta_t,
                          evolver->config.min_tau, evolver->config.max_tau,
                          evolver->config.use_adaptive_tau,
                          ws_state);

        memcpy(evolver->channel_states + (size_t)c * D, ws_state,
               (size_t)D * sizeof(float));

        if (evolver->config.enable_cross_channel) {
            memcpy(ws_prev, ws_state, (size_t)D * sizeof(float));
        } else {
            memcpy(ws_prev, evolver->state, (size_t)D * sizeof(float));
        }
    }

    memcpy(evolver->state, ws_state, (size_t)D * sizeof(float));
    memcpy(visual_state, ws_state, (size_t)D * sizeof(float));

    safe_free((void**)&ws_state);
    safe_free((void**)&ws_prev);
    return 0;
}

int liquid_visual_cfc_evolver_step(LiquidVisualCfCEvolver* evolver,
                                    const float* visual_inputs, int input_dim,
                                    int num_steps, float* final_state) {
    if (!evolver || !visual_inputs || !final_state || !evolver->initialized) return -1;

    for (int t = 0; t < num_steps; t++) {
        int ret = liquid_visual_cfc_evolver_forward(
            evolver, visual_inputs + (size_t)t * input_dim, input_dim,
            t == num_steps - 1 ? final_state : evolver->state);
        if (ret != 0) return -1;
    }
    return 0;
}

void liquid_visual_cfc_evolver_reset(LiquidVisualCfCEvolver* evolver) {
    if (!evolver) return;
    memset(evolver->state, 0, (size_t)evolver->config.visual_state_dim * sizeof(float));
    memset(evolver->channel_states, 0,
           (size_t)evolver->config.num_visual_channels * evolver->config.visual_state_dim * sizeof(float));
}

int liquid_visual_cfc_evolver_get_state_dim(const LiquidVisualCfCEvolver* evolver) {
    return evolver ? evolver->config.visual_state_dim : 0;
}

/* ============ LiquidSpatialProcessor 实现 ============ */

struct LiquidSpatialProcessor {
    LiquidSpatialProcessorConfig config;
    float* feat_to_hidden_w_gate;
    float* feat_to_hidden_w_act;
    float* hidden_to_hidden_w_gate;
    float* hidden_to_hidden_w_act;
    float* gate_bias;
    float* act_bias;
    float* time_constants;
    float* hidden_state;
    int initialized;
    unsigned int seed;
};

LiquidSpatialProcessorConfig liquid_spatial_processor_get_default_config(void) {
    LiquidSpatialProcessorConfig cfg;
    cfg.num_features = 64;
    cfg.feature_dim = 64;
    cfg.spatial_hidden_dim = 128;
    cfg.time_constant = 0.1f;
    cfg.delta_t = 0.05f;
    cfg.use_adaptive_tau = 1;
    cfg.num_evolution_steps = 3;
    cfg.min_tau = 0.01f;
    cfg.max_tau = 1.0f;
    return cfg;
}

LiquidSpatialProcessor* liquid_spatial_processor_create(const LiquidSpatialProcessorConfig* config) {
    if (!config) return NULL;

    LiquidSpatialProcessor* sp = (LiquidSpatialProcessor*)safe_malloc(sizeof(LiquidSpatialProcessor));
    if (!sp) return NULL;

    memcpy(&sp->config, config, sizeof(LiquidSpatialProcessorConfig));

    int FD = config->feature_dim;
    int HD = config->spatial_hidden_dim;

    sp->feat_to_hidden_w_gate = (float*)safe_calloc((size_t)HD * FD, sizeof(float));
    sp->feat_to_hidden_w_act = (float*)safe_calloc((size_t)HD * FD, sizeof(float));
    sp->hidden_to_hidden_w_gate = (float*)safe_calloc((size_t)HD * HD, sizeof(float));
    sp->hidden_to_hidden_w_act = (float*)safe_calloc((size_t)HD * HD, sizeof(float));
    sp->gate_bias = (float*)safe_calloc(HD, sizeof(float));
    sp->act_bias = (float*)safe_calloc(HD, sizeof(float));
    sp->time_constants = (float*)safe_calloc(HD, sizeof(float));
    sp->hidden_state = (float*)safe_calloc((size_t)HD, sizeof(float));

    if (!sp->feat_to_hidden_w_gate || !sp->feat_to_hidden_w_act ||
        !sp->hidden_to_hidden_w_gate || !sp->hidden_to_hidden_w_act ||
        !sp->gate_bias || !sp->act_bias ||
        !sp->time_constants || !sp->hidden_state) {
        liquid_spatial_processor_free(sp);
        return NULL;
    }

    sp->seed = 789;
    _xavier_init(sp->feat_to_hidden_w_gate, HD, FD, &sp->seed);
    _xavier_init(sp->feat_to_hidden_w_act, HD, FD, &sp->seed);
    _xavier_init(sp->hidden_to_hidden_w_gate, HD, HD, &sp->seed);
    _xavier_init(sp->hidden_to_hidden_w_act, HD, HD, &sp->seed);
    _const_init(sp->gate_bias, HD, 0.0f);
    _const_init(sp->act_bias, HD, 0.0f);
    _const_init(sp->time_constants, HD, config->time_constant);

    sp->initialized = 1;
    return sp;
}

void liquid_spatial_processor_free(LiquidSpatialProcessor* processor) {
    if (!processor) return;
    safe_free((void**)&processor->feat_to_hidden_w_gate);
    safe_free((void**)&processor->feat_to_hidden_w_act);
    safe_free((void**)&processor->hidden_to_hidden_w_gate);
    safe_free((void**)&processor->hidden_to_hidden_w_act);
    safe_free((void**)&processor->gate_bias);
    safe_free((void**)&processor->act_bias);
    safe_free((void**)&processor->time_constants);
    safe_free((void**)&processor->hidden_state);
    safe_free((void**)&processor);
}

static void _liquid_spatial_step(const float* features, int num_features, int feature_dim,
                                  const float* hidden_state, int hidden_dim,
                                  const float* fg_w, const float* fa_w,
                                  const float* hg_w, const float* ha_w,
                                  const float* gb, const float* ab,
                                  const float* time_constants, float delta_t,
                                  float min_tau, float max_tau, int use_adaptive_tau,
                                  float* output) {
    /* 每个隐藏单元接收来自所有特征的输入
     * gate_i = Σ_j W_fg_ij · f_j + Σ_k W_hg_ik · h_k + b_gi
     * act_i  = Σ_j W_fa_ij · f_j + Σ_k W_ha_ik · h_k + b_ai
     */
    for (int i = 0; i < hidden_dim; i++) {
        float gate_sum = gb ? gb[i] : 0.0f;
        float act_sum = ab ? ab[i] : 0.0f;

        for (int j = 0; j < feature_dim; j++) {
            float feat_avg = 0.0f;
            for (int n = 0; n < num_features; n++) {
                feat_avg += features[(size_t)n * feature_dim + j];
            }
            feat_avg /= (float)num_features;

            int idx = i * feature_dim + j;
            gate_sum += fg_w[idx] * feat_avg;
            act_sum += fa_w[idx] * feat_avg;
        }

        for (int k = 0; k < hidden_dim; k++) {
            int idx = i * hidden_dim + k;
            gate_sum += hg_w[idx] * hidden_state[k];
            act_sum += ha_w[idx] * hidden_state[k];
        }

        float gate = gate_sum > 10.0f ? 1.0f :
                     (gate_sum < -10.0f ? 0.0f : 1.0f / (1.0f + expf(-gate_sum)));
        float activation = act_sum > 10.0f ? 1.0f :
                           (act_sum < -10.0f ? -1.0f : tanhf(act_sum));
        float driver = gate * activation;

        float tau = use_adaptive_tau && time_constants ? time_constants[i] : 0.1f;
        if (tau < min_tau) tau = min_tau;
        if (tau > max_tau) tau = max_tau;

        float dt_over_tau = delta_t / tau;
        if (dt_over_tau < 1e-8f) dt_over_tau = 1e-8f;
        if (dt_over_tau > 100.0f) dt_over_tau = 100.0f;

        float exp_term = dt_over_tau > 20.0f ? 0.0f :
                         (dt_over_tau < 1e-4f ? 1.0f - dt_over_tau : expf(-dt_over_tau));
        float one_minus_exp = dt_over_tau > 20.0f ? 1.0f :
                              (dt_over_tau < 1e-4f ? dt_over_tau : 1.0f - exp_term);

        float new_state = hidden_state[i] * exp_term + one_minus_exp * driver;
        if (isnan(new_state) || isinf(new_state)) new_state = 0.0f;
        output[i] = new_state;
    }
}

int liquid_spatial_processor_forward(LiquidSpatialProcessor* processor,
                                      const float* features, int num_features, int feature_dim,
                                      float* output) {
    if (!processor || !features || !output || !processor->initialized) return -1;

    int HD = processor->config.spatial_hidden_dim;
    int num_steps = processor->config.num_evolution_steps;
    float delta_t = processor->config.delta_t;

    float* prev = (float*)safe_malloc((size_t)HD * sizeof(float));
    float* curr = (float*)safe_malloc((size_t)HD * sizeof(float));
    if (!prev || !curr) {
        safe_free((void**)&prev);
        safe_free((void**)&curr);
        return -1;
    }

    memcpy(prev, processor->hidden_state, (size_t)HD * sizeof(float));

    for (int s = 0; s < num_steps; s++) {
        _liquid_spatial_step(features, num_features, feature_dim,
                              prev, HD,
                              processor->feat_to_hidden_w_gate,
                              processor->feat_to_hidden_w_act,
                              processor->hidden_to_hidden_w_gate,
                              processor->hidden_to_hidden_w_act,
                              processor->gate_bias, processor->act_bias,
                              processor->time_constants, delta_t,
                              processor->config.min_tau, processor->config.max_tau,
                              processor->config.use_adaptive_tau,
                              curr);
        memcpy(prev, curr, (size_t)HD * sizeof(float));
    }

    memcpy(processor->hidden_state, curr, (size_t)HD * sizeof(float));
    memcpy(output, curr, (size_t)HD * sizeof(float));

    safe_free((void**)&prev);
    safe_free((void**)&curr);
    return 0;
}

void liquid_spatial_processor_reset(LiquidSpatialProcessor* processor) {
    if (!processor) return;
    memset(processor->hidden_state, 0,
           (size_t)processor->config.spatial_hidden_dim * sizeof(float));
}

int liquid_spatial_processor_get_output_dim(const LiquidSpatialProcessor* processor) {
    return processor ? processor->config.spatial_hidden_dim : 0;
}

/* ============ LiquidVisionManager 实现 ============ */

struct LiquidVisionManager {
    LiquidVisionManagerConfig config;
    LiquidPatchEncoder* patch_encoder;
    LiquidVisualCfCEvolver* cfc_evolver;
    LiquidSpatialProcessor* spatial_processor;
    float* feature_buffer;
    float* temporal_buffer;
    float* temporal_accumulator;
    int feature_buffer_size;
    int temporal_frame_count;
    int is_feature_mode;
    int initialized;
    int max_feature_dim;
    float* cfc_temporal_state;
    int cfc_temporal_initialized;
};

LiquidVisionManagerConfig liquid_vision_manager_get_default_config(void) {
    LiquidVisionManagerConfig cfg;
    cfg.pipeline_type = LIQUID_VISION_FULL;
    cfg.patch_config = liquid_patch_encoder_get_default_config();
    cfg.evolver_config = liquid_visual_cfc_evolver_get_default_config();
    cfg.spatial_config = liquid_spatial_processor_get_default_config();
    cfg.enable_temporal_integration = 0;
    cfg.temporal_window_size = 5;
    return cfg;
}

LiquidVisionManager* liquid_vision_manager_create(const LiquidVisionManagerConfig* config) {
    if (!config) return NULL;

    LiquidVisionManager* mgr = (LiquidVisionManager*)safe_malloc(sizeof(LiquidVisionManager));
    if (!mgr) return NULL;

    memcpy(&mgr->config, config, sizeof(LiquidVisionManagerConfig));

    mgr->patch_encoder = NULL;
    mgr->cfc_evolver = NULL;
    mgr->spatial_processor = NULL;
    mgr->feature_buffer = NULL;
    mgr->temporal_buffer = NULL;
    mgr->temporal_accumulator = NULL;
    mgr->feature_buffer_size = 0;
    mgr->temporal_frame_count = 0;
    mgr->is_feature_mode = 0;
    mgr->initialized = 0;
    mgr->max_feature_dim = 0;
    mgr->cfc_temporal_state = NULL;
    mgr->cfc_temporal_initialized = 0;

    int need_patch = (config->pipeline_type == LIQUID_VISION_PATCH_ENCODER ||
                      config->pipeline_type == LIQUID_VISION_FULL);
    int need_spatial = (config->pipeline_type == LIQUID_VISION_SPATIAL ||
                        config->pipeline_type == LIQUID_VISION_FULL);
    int need_evolver = (config->pipeline_type == LIQUID_VISION_CFC_EVOLVER ||
                        config->pipeline_type == LIQUID_VISION_FULL);

    if (need_patch) {
        mgr->patch_encoder = liquid_patch_encoder_create(&config->patch_config);
        if (!mgr->patch_encoder) {
            liquid_vision_manager_free(mgr);
            return NULL;
        }
    }

    if (need_spatial) {
        mgr->spatial_processor = liquid_spatial_processor_create(&config->spatial_config);
        if (!mgr->spatial_processor) {
            liquid_vision_manager_free(mgr);
            return NULL;
        }
    }

    if (need_evolver) {
        mgr->cfc_evolver = liquid_visual_cfc_evolver_create(&config->evolver_config);
        if (!mgr->cfc_evolver) {
            liquid_vision_manager_free(mgr);
            return NULL;
        }
    }

    if (config->enable_temporal_integration) {
        int max_dim = config->evolver_config.visual_state_dim;
        if (config->pipeline_type == LIQUID_VISION_PATCH_ENCODER)
            max_dim = config->patch_config.patch_hidden_dim;
        if (config->pipeline_type == LIQUID_VISION_SPATIAL)
            max_dim = config->spatial_config.spatial_hidden_dim;

        mgr->temporal_buffer = (float*)safe_calloc(
            (size_t)config->temporal_window_size * max_dim, sizeof(float));
        mgr->temporal_accumulator = (float*)safe_calloc(max_dim, sizeof(float));
        if (!mgr->temporal_buffer || !mgr->temporal_accumulator) {
            liquid_vision_manager_free(mgr);
            return NULL;
        }
    }

    mgr->max_feature_dim = 256;
    mgr->initialized = 1;
    return mgr;
}

void liquid_vision_manager_free(LiquidVisionManager* manager) {
    if (!manager) return;
    liquid_patch_encoder_free(manager->patch_encoder);
    liquid_visual_cfc_evolver_free(manager->cfc_evolver);
    liquid_spatial_processor_free(manager->spatial_processor);
    safe_free((void**)&manager->feature_buffer);
    safe_free((void**)&manager->temporal_buffer);
    safe_free((void**)&manager->temporal_accumulator);
    safe_free((void**)&manager->cfc_temporal_state);
    safe_free((void**)&manager);
}

int liquid_vision_manager_forward(LiquidVisionManager* manager,
                                   int width, int height, int channels,
                                   const float* image_data,
                                   float* output, int output_dim) {
    if (!manager || !image_data || !output || !manager->initialized) return -1;

    float* internal_buf = NULL;
    int result = -1;

    switch (manager->config.pipeline_type) {
        case LIQUID_VISION_PATCH_ENCODER: {
            int D = manager->config.patch_config.patch_hidden_dim;
            int np = liquid_patch_encoder_get_num_patches(manager->patch_encoder, width, height);
            if (np * D > output_dim) {
                np = output_dim / D;
            }
            internal_buf = (float*)safe_malloc((size_t)np * D * sizeof(float));
            if (!internal_buf) return -1;

            int num_patches = liquid_patch_encoder_forward(
                manager->patch_encoder, width, height, channels, image_data,
                internal_buf, np);
            if (num_patches < 0) break;

            int out_len = num_patches * D;
            if (out_len > output_dim) out_len = output_dim;
            memcpy(output, internal_buf, (size_t)out_len * sizeof(float));
            result = out_len;
            break;
        }

        case LIQUID_VISION_CFC_EVOLVER: {
            int D = manager->config.evolver_config.visual_state_dim;
            if (D > output_dim) D = output_dim;

            int input_dim = width * height * channels;
            internal_buf = (float*)safe_malloc((size_t)input_dim * sizeof(float));
            if (!internal_buf) return -1;

            for (int i = 0; i < input_dim; i++) internal_buf[i] = image_data[i];

            int ret = liquid_visual_cfc_evolver_forward(
                manager->cfc_evolver, internal_buf, input_dim, output);
            if (ret != 0) break;

            result = D;
            break;
        }

        case LIQUID_VISION_SPATIAL: {
            int HD = manager->config.spatial_config.spatial_hidden_dim;
            if (HD > output_dim) HD = output_dim;

            int NF = manager->config.spatial_config.num_features;
            int FD = manager->config.spatial_config.feature_dim;
            internal_buf = (float*)safe_malloc((size_t)NF * FD * sizeof(float));
            if (!internal_buf) return -1;

            int input_dim = width * height * channels;
            int copy_len = input_dim < NF * FD ? input_dim : NF * FD;
            memcpy(internal_buf, image_data, (size_t)copy_len * sizeof(float));

            int ret = liquid_spatial_processor_forward(
                manager->spatial_processor, internal_buf, NF, FD, output);
            if (ret != 0) break;

            result = HD;
            break;
        }

        case LIQUID_VISION_FULL: {
            int patch_D = manager->config.patch_config.patch_hidden_dim;
            int spatial_HD = manager->config.spatial_config.spatial_hidden_dim;
            int evolver_D = manager->config.evolver_config.visual_state_dim;

            if (evolver_D > output_dim) evolver_D = output_dim;

            int np = liquid_patch_encoder_get_num_patches(manager->patch_encoder, width, height);
            internal_buf = (float*)safe_malloc((size_t)np * patch_D * sizeof(float));
            if (!internal_buf) return -1;

            int num_patches = liquid_patch_encoder_forward(
                manager->patch_encoder, width, height, channels, image_data,
                internal_buf, np);
            if (num_patches < 0) break;

            float* spatial_buf = (float*)safe_malloc((size_t)spatial_HD * sizeof(float));
            if (!spatial_buf) break;

            int ret_sp = liquid_spatial_processor_forward(
                manager->spatial_processor, internal_buf, num_patches, patch_D, spatial_buf);
            if (ret_sp != 0) {
                safe_free((void**)&spatial_buf);
                break;
            }

            int ret_ev = liquid_visual_cfc_evolver_forward(
                manager->cfc_evolver, spatial_buf, spatial_HD, output);
            safe_free((void**)&spatial_buf);
            if (ret_ev != 0) break;

            result = evolver_D;
            break;
        }

        default:
            break;
    }

    safe_free((void**)&internal_buf);

    if (result > 0 && manager->config.enable_temporal_integration) {
        int window = manager->config.temporal_window_size;
        int frame = manager->temporal_frame_count % window;
        if (frame < 0) frame = 0;

        memcpy(manager->temporal_buffer + (size_t)frame * result,
               output, (size_t)result * sizeof(float));
        manager->temporal_frame_count++;

        int count = manager->temporal_frame_count < window ?
                    manager->temporal_frame_count : window;

        memset(manager->temporal_accumulator, 0, (size_t)result * sizeof(float));
        for (int f = 0; f < count; f++) {
            for (int d = 0; d < result; d++) {
                manager->temporal_accumulator[d] +=
                    manager->temporal_buffer[(size_t)f * result + d];
            }
        }
        float inv_count = 1.0f / (float)count;
        for (int d = 0; d < result; d++) {
            output[d] = manager->temporal_accumulator[d] * inv_count;
        }

        /* CfC视频时序建模：基于隐藏状态的帧间连续性演化
         * 使用CfC微分方程维护帧间隐藏状态连续性：
         * τ dh/dt = -h + σ(W_g·x + b_g) ⊙ tanh(W_a·x + b_a)
         * 其中 h 是跨帧隐藏状态，x 是当前帧的视觉特征
         */
        if (manager->cfc_temporal_state) {
            float dt = manager->temporal_frame_count > 1 ? 0.033f : 0.0f;
            if (dt < 1e-6f) dt = 0.033f;
            float tau = 0.1f;

            for (int d = 0; d < result && d < manager->max_feature_dim; d++) {
                float x = output[d];
                float prev_h = manager->cfc_temporal_state[d];

                float gate = 1.0f / (1.0f + expf(-x));
                float activation = tanhf(x);

                float exp_term = expf(-dt / tau);
                float one_minus_exp = 1.0f - exp_term;

                float new_h = prev_h * exp_term + one_minus_exp * gate * activation;
                if (isnan(new_h) || isinf(new_h)) new_h = prev_h;

                manager->cfc_temporal_state[d] = new_h;
                output[d] = 0.7f * output[d] + 0.3f * new_h;
            }

            manager->cfc_temporal_initialized = 1;
        }
    }

    return result;
}

int liquid_vision_manager_enable_cfc_temporal(LiquidVisionManager* manager, int enable) {
    if (!manager || manager->max_feature_dim <= 0) return -1;

    if (enable) {
        if (!manager->cfc_temporal_state) {
            manager->cfc_temporal_state = (float*)safe_calloc(
                (size_t)manager->max_feature_dim, sizeof(float));
            if (!manager->cfc_temporal_state) return -1;
        }
        manager->cfc_temporal_initialized = 0;
    } else {
        safe_free((void**)&manager->cfc_temporal_state);
        manager->cfc_temporal_initialized = 0;
    }
    return 0;
}

int liquid_vision_manager_reset_temporal(LiquidVisionManager* manager) {
    if (!manager) return -1;
    manager->temporal_frame_count = 0;
    if (manager->cfc_temporal_state) {
        memset(manager->cfc_temporal_state, 0,
               (size_t)manager->max_feature_dim * sizeof(float));
    }
    manager->cfc_temporal_initialized = 0;
    if (manager->temporal_buffer) {
        memset(manager->temporal_buffer, 0,
               (size_t)manager->config.temporal_window_size *
               (size_t)manager->max_feature_dim * sizeof(float));
    }
    return 0;
}

int liquid_vision_manager_set_features(LiquidVisionManager* manager,
                                        const float* features, int feature_dim) {
    if (!manager || !features || feature_dim <= 0) return -1;

    if (manager->feature_buffer_size < feature_dim) {
        float* new_buf = (float*)safe_realloc(manager->feature_buffer,
                                                (size_t)feature_dim * sizeof(float));
        if (!new_buf) return -1;
        manager->feature_buffer = new_buf;
        manager->feature_buffer_size = feature_dim;
    }

    memcpy(manager->feature_buffer, features, (size_t)feature_dim * sizeof(float));
    manager->is_feature_mode = 1;
    return 0;
}

void liquid_vision_manager_reset(LiquidVisionManager* manager) {
    if (!manager) return;
    liquid_patch_encoder_reset(manager->patch_encoder);
    liquid_visual_cfc_evolver_reset(manager->cfc_evolver);
    liquid_spatial_processor_reset(manager->spatial_processor);
    manager->temporal_frame_count = 0;
    if (manager->temporal_buffer) {
        memset(manager->temporal_buffer, 0,
               (size_t)manager->config.temporal_window_size *
               manager->config.evolver_config.visual_state_dim *
               sizeof(float));
    }
    if (manager->temporal_accumulator) {
        memset(manager->temporal_accumulator, 0,
               (size_t)manager->config.evolver_config.visual_state_dim *
               sizeof(float));
    }
    manager->is_feature_mode = 0;
}

int liquid_vision_manager_get_output_dim(const LiquidVisionManager* manager) {
    if (!manager) return 0;
    switch (manager->config.pipeline_type) {
        case LIQUID_VISION_PATCH_ENCODER:
            return manager->config.patch_config.patch_hidden_dim *
                   manager->config.patch_config.max_patches;
        case LIQUID_VISION_CFC_EVOLVER:
            return manager->config.evolver_config.visual_state_dim;
        case LIQUID_VISION_SPATIAL:
            return manager->config.spatial_config.spatial_hidden_dim;
        case LIQUID_VISION_FULL:
            return manager->config.evolver_config.visual_state_dim;
        default:
            return 0;
    }
}

int liquid_vision_manager_save(const LiquidVisionManager* manager, const char* filepath) {
    if (!manager || !filepath) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    fwrite(&manager->config, sizeof(LiquidVisionManagerConfig), 1, fp);

    if (manager->patch_encoder) {
        int has = 1;
        fwrite(&has, sizeof(int), 1, fp);
        fwrite(&manager->patch_encoder->config, sizeof(LiquidPatchEncoderConfig), 1, fp);
        int D = manager->patch_encoder->config.patch_hidden_dim;
        int patch_size = manager->patch_encoder->config.patch_size;
        int channels = manager->patch_encoder->config.input_channels;
        int input_dim = patch_size * patch_size * channels;
        fwrite(manager->patch_encoder->patch_w_gate, sizeof(float), (size_t)D * input_dim, fp);
        fwrite(manager->patch_encoder->patch_w_act, sizeof(float), (size_t)D * input_dim, fp);
        fwrite(manager->patch_encoder->patch_h_to_gate, sizeof(float), (size_t)D * D, fp);
        fwrite(manager->patch_encoder->patch_h_to_act, sizeof(float), (size_t)D * D, fp);
        fwrite(manager->patch_encoder->patch_gate_bias, sizeof(float), D, fp);
        fwrite(manager->patch_encoder->patch_act_bias, sizeof(float), D, fp);
        fwrite(manager->patch_encoder->time_constants, sizeof(float), D, fp);
    } else {
        int has = 0;
        fwrite(&has, sizeof(int), 1, fp);
    }

    if (manager->cfc_evolver) {
        int has = 1;
        fwrite(&has, sizeof(int), 1, fp);
        fwrite(&manager->cfc_evolver->config, sizeof(LiquidVisualCfCEvolverConfig), 1, fp);
        int NC = manager->cfc_evolver->config.num_visual_channels;
        int D = manager->cfc_evolver->config.visual_state_dim;
        fwrite(manager->cfc_evolver->w_gate, sizeof(float), (size_t)NC * D * D, fp);
        fwrite(manager->cfc_evolver->w_act, sizeof(float), (size_t)NC * D * D, fp);
        fwrite(manager->cfc_evolver->h_to_gate, sizeof(float), (size_t)NC * D * D, fp);
        fwrite(manager->cfc_evolver->h_to_act, sizeof(float), (size_t)NC * D * D, fp);
        fwrite(manager->cfc_evolver->gate_bias, sizeof(float), (size_t)NC * D, fp);
        fwrite(manager->cfc_evolver->act_bias, sizeof(float), (size_t)NC * D, fp);
    } else {
        int has = 0;
        fwrite(&has, sizeof(int), 1, fp);
    }

    if (manager->spatial_processor) {
        int has = 1;
        fwrite(&has, sizeof(int), 1, fp);
        fwrite(&manager->spatial_processor->config, sizeof(LiquidSpatialProcessorConfig), 1, fp);
        int HD = manager->spatial_processor->config.spatial_hidden_dim;
        int FD = manager->spatial_processor->config.feature_dim;
        fwrite(manager->spatial_processor->feat_to_hidden_w_gate, sizeof(float), (size_t)HD * FD, fp);
        fwrite(manager->spatial_processor->feat_to_hidden_w_act, sizeof(float), (size_t)HD * FD, fp);
        fwrite(manager->spatial_processor->hidden_to_hidden_w_gate, sizeof(float), (size_t)HD * HD, fp);
        fwrite(manager->spatial_processor->hidden_to_hidden_w_act, sizeof(float), (size_t)HD * HD, fp);
        fwrite(manager->spatial_processor->gate_bias, sizeof(float), HD, fp);
        fwrite(manager->spatial_processor->act_bias, sizeof(float), HD, fp);
    } else {
        int has = 0;
        fwrite(&has, sizeof(int), 1, fp);
    }

    fclose(fp);
    return 0;
}

int liquid_vision_manager_load(LiquidVisionManager* manager, const char* filepath) {
    if (!manager || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    LiquidVisionManagerConfig loaded_config;
    if (fread(&loaded_config, sizeof(LiquidVisionManagerConfig), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    int has;
    if (fread(&has, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    if (has && manager->patch_encoder) {
        LiquidPatchEncoderConfig pcfg;
        if (fread(&pcfg, sizeof(LiquidPatchEncoderConfig), 1, fp) != 1) { fclose(fp); return -1; }
        int D = pcfg.patch_hidden_dim;
        int patch_size = pcfg.patch_size;
        int ch = pcfg.input_channels;
        int input_dim = patch_size * patch_size * ch;
        fread(manager->patch_encoder->patch_w_gate, sizeof(float), (size_t)D * input_dim, fp);
        fread(manager->patch_encoder->patch_w_act, sizeof(float), (size_t)D * input_dim, fp);
        fread(manager->patch_encoder->patch_h_to_gate, sizeof(float), (size_t)D * D, fp);
        fread(manager->patch_encoder->patch_h_to_act, sizeof(float), (size_t)D * D, fp);
        fread(manager->patch_encoder->patch_gate_bias, sizeof(float), D, fp);
        fread(manager->patch_encoder->patch_act_bias, sizeof(float), D, fp);
        fread(manager->patch_encoder->time_constants, sizeof(float), D, fp);
    }

    if (fread(&has, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    if (has && manager->cfc_evolver) {
        LiquidVisualCfCEvolverConfig ecfg;
        if (fread(&ecfg, sizeof(LiquidVisualCfCEvolverConfig), 1, fp) != 1) { fclose(fp); return -1; }
        int NC = ecfg.num_visual_channels;
        int D = ecfg.visual_state_dim;
        fread(manager->cfc_evolver->w_gate, sizeof(float), (size_t)NC * D * D, fp);
        fread(manager->cfc_evolver->w_act, sizeof(float), (size_t)NC * D * D, fp);
        fread(manager->cfc_evolver->h_to_gate, sizeof(float), (size_t)NC * D * D, fp);
        fread(manager->cfc_evolver->h_to_act, sizeof(float), (size_t)NC * D * D, fp);
        fread(manager->cfc_evolver->gate_bias, sizeof(float), (size_t)NC * D, fp);
        fread(manager->cfc_evolver->act_bias, sizeof(float), (size_t)NC * D, fp);
    }

    if (fread(&has, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    if (has && manager->spatial_processor) {
        LiquidSpatialProcessorConfig scfg;
        if (fread(&scfg, sizeof(LiquidSpatialProcessorConfig), 1, fp) != 1) { fclose(fp); return -1; }
        int HD = scfg.spatial_hidden_dim;
        int FD = scfg.feature_dim;
        fread(manager->spatial_processor->feat_to_hidden_w_gate, sizeof(float), (size_t)HD * FD, fp);
        fread(manager->spatial_processor->feat_to_hidden_w_act, sizeof(float), (size_t)HD * FD, fp);
        fread(manager->spatial_processor->hidden_to_hidden_w_gate, sizeof(float), (size_t)HD * HD, fp);
        fread(manager->spatial_processor->hidden_to_hidden_w_act, sizeof(float), (size_t)HD * HD, fp);
        fread(manager->spatial_processor->gate_bias, sizeof(float), HD, fp);
        fread(manager->spatial_processor->act_bias, sizeof(float), HD, fp);
    }

    fclose(fp);
    return 0;
}
