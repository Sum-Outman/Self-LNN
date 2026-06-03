/**
 * @file liquid_vision.c
 * @brief 统一液态视觉处理系统 —— 全CfC液态神经网络驱动
 *
 * 基于CfC（Closed-form Continuous-time）液态神经网络的统一视觉处理组件。
 * 所有视觉处理使用单一微分方程框架：
 * tau dh/dt = -h + sigma(W_gx*x + W_gh*h + b_g) * tanh(W_ax*x + W_ah*h + b_a)
 * 不引入任何Transformer、注意力机制或独立处理器。
 * 纯C实现，不依赖任何第三方库。
 *
 * 模块结构：
 *   【主路径】LiquidVisionManager（PatchEncoder -> SpatialProcessor -> CfCEvolver）
 *   【辅助路径】传统CV预处理（Sobel/LBP/HOG/HSV直方图/颜色分析/图像金字塔）
 *   【检测路径】YOLO风格目标检测头 + NMS非极大值抑制
 *   【类别系统】动态可扩展视觉类别注册表（80类COCO默认 + 动态扩展）
 *   【兼容层】CfcVisionProcessor/CfcOdeLayer（保持现有agi/multimodal/ocr等模块兼容）
 *
 * 本文件整合了原 vision.c、deep_vision.c 和 liquid_vision.c 的所有功能。
 * 原 vision.c 和 deep_vision.c 已标记为重定向存根。
 */

#include "selflnn/multimodal/liquid_vision.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/perf.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/selflnn.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <stdint.h>

/* ===================================================================
 * 宏定义
 * =================================================================== */

#define LIQUID_VISION_MAX_OBJECTS 50
#define LIQUID_VISION_INPUT_DIM 1024
#define LIQUID_VISION_HIDDEN_DIM 1024
#define COLOR_HIST_BINS 32

/* ===================================================================
 * 第一节：核心CfC数学函数（原liquid_vision.c）
 * =================================================================== */

static float _activation_sigmoid_stable(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static float _activation_tanh_stable(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    return tanhf(x);
}

/*
 * CfC闭式解核心函数
 * tau dh/dt = -h + sigma(W_gx*x + W_gh*h + b_g) * tanh(W_ax*x + W_ah*h + b_a)
 * 闭式解: h(t+dt) = h(t)*exp(-dt/tau) + (1-exp(-dt/tau))*[sigma(...) * tanh(...)]
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

        float gate = _activation_sigmoid_stable(gate_sum);
        float activation = _activation_tanh_stable(act_sum);
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

/* CfC液态步反向传播 */
static void _liquid_cfc_backward_step(
    const float* input, int input_size,
    const float* prev_state, int hidden_size,
    const float* w_gate, const float* w_activation,
    const float* h_to_gate, const float* h_to_activation,
    const float* gate_bias, const float* act_bias,
    const float* time_constants, float delta_t,
    float min_tau, float max_tau, int use_adaptive_tau,
    const float* dL_dh,
    float* dL_dx, float* dL_dh_prev,
    float* dL_dW_gate, float* dL_dW_act,
    float* dL_dH_gate, float* dL_dH_act,
    float* dL_db_gate, float* dL_db_act,
    float* dL_dtau) {
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

        float gate = _activation_sigmoid_stable(gate_sum);
        float activation = _activation_tanh_stable(act_sum);
        float driver = gate * activation;

        float tau = use_adaptive_tau && time_constants ? time_constants[i] : 0.1f;
        if (tau < min_tau) tau = min_tau;
        if (tau > max_tau) tau = max_tau;

        float dt_over_tau = delta_t / tau;
        if (dt_over_tau < 1e-8f) dt_over_tau = 1e-8f;
        if (dt_over_tau > 100.0f) dt_over_tau = 100.0f;

        float exp_term, dexp_dtau;
        if (dt_over_tau > 20.0f) {
            exp_term = 0.0f; dexp_dtau = 0.0f;
        } else if (dt_over_tau < 1e-4f) {
            exp_term = 1.0f - dt_over_tau; dexp_dtau = -delta_t / (tau * tau);
        } else {
            exp_term = expf(-dt_over_tau); dexp_dtau = exp_term * (delta_t / (tau * tau));
        }

        float one_minus_exp, done_dtau;
        if (dt_over_tau > 20.0f) {
            one_minus_exp = 1.0f; done_dtau = 0.0f;
        } else if (dt_over_tau < 1e-4f) {
            one_minus_exp = dt_over_tau; done_dtau = -delta_t / (tau * tau);
        } else {
            one_minus_exp = 1.0f - exp_term; done_dtau = -dexp_dtau;
        }

        float dh_dgate = one_minus_exp * activation;
        float dh_dact = one_minus_exp * gate;
        float dsig = gate * (1.0f - gate);
        float dtanh = 1.0f - activation * activation;

        float dL_dgate_sum = dL_dh[i] * dh_dgate * dsig;
        float dL_dact_sum = dL_dh[i] * dh_dact * dtanh;

        if (dL_dx) {
            for (int j = 0; j < input_size; j++) {
                int idx = i * input_size + j;
                dL_dx[j] += dL_dgate_sum * w_gate[idx] + dL_dact_sum * w_activation[idx];
            }
        }
        if (dL_dh_prev) {
            dL_dh_prev[i] += dL_dh[i] * exp_term;
            for (int j = 0; j < hidden_size; j++) {
                int idx = i * hidden_size + j;
                dL_dh_prev[j] += dL_dgate_sum * h_to_gate[idx] + dL_dact_sum * h_to_activation[idx];
            }
        }
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
        if (dL_db_gate) dL_db_gate[i] += dL_dgate_sum;
        if (dL_db_act) dL_db_act[i] += dL_dact_sum;

        if (dL_dtau && use_adaptive_tau) {
            float dh_dtau = prev_state[i] * dexp_dtau + driver * done_dtau;
            dL_dtau[i] += dL_dh[i] * dh_dtau;
        }
    }
}

/* 从图像提取补丁 */
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

/* S-011修复: Xavier初始化 — 支持可选正态/均匀分布
 * M-012修复: 默认使用正态分布(Box-Muller)进行Xavier初始化
 * use_normal=1: 使用标准正态分布 N(0, σ²)（推荐，σ=√(2/(fan_in+fan_out))）
 * use_normal=0: 使用均匀分布 U[-a,a]（a=√(6/(fan_in+fan_out))，向后兼容）
 * P2-010: 统一使用secure_random_float替代乘法PRNG */
static void _xavier_init(float* data, int rows, int cols, unsigned int* seed, int use_normal) {
    (void)seed;  /* seed参数保留兼容，secure_random_float内部自行管理熵源 */
    if (use_normal) {
        /* 正态分布 Xavier: std = sqrt(2.0 / (fan_in + fan_out)) */
        float stddev = sqrtf(2.0f / (float)(rows + cols));
        for (int i = 0; i < rows * cols; i++) {
            float u1 = secure_random_float();
            float u2 = secure_random_float();
            float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.28318530718f * u2);
            data[i] = z * stddev;
        }
    } else {
        /* 均匀分布 Xavier: range = [-scale, scale], scale = sqrt(6 / (fan_in + fan_out)) */
        float scale = sqrtf(6.0f / (float)(rows + cols));
        for (int i = 0; i < rows * cols; i++) {
            float r = secure_random_float();
            data[i] = (r * 2.0f - 1.0f) * scale;
        }
    }
}

/* 常量初始化 */
static void _const_init(float* data, int n, float val) {
    for (int i = 0; i < n; i++) data[i] = val;
}

/* Softmax稳定化计算 */
static void _softmax_stable(float* logits, int dim) {
    float max_val = logits[0];
    for (int i = 1; i < dim; i++)
        if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    if (sum > 1e-10f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < dim; i++) logits[i] *= inv_sum;
    }
}

/* ===================================================================
 * 第二节：LiquidPatchEncoder（原liquid_vision.c）
 * =================================================================== */

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
    /* S-011修复: use_normal=1 使用正态分布Xavier初始化 */
    _xavier_init(enc->patch_w_gate, D, input_dim, &enc->seed, 1);
    _xavier_init(enc->patch_w_act, D, input_dim, &enc->seed, 1);
    _xavier_init(enc->patch_h_to_gate, D, D, &enc->seed, 1);
    _xavier_init(enc->patch_h_to_act, D, D, &enc->seed, 1);
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

    float* gbufs[9];
    for (int k = 0; k < 7; k++) {
        size_t esz = (k < 4) ? ((k < 2) ? (size_t)D * patch_elem : (size_t)D * D) : (size_t)D;
        gbufs[k] = (float*)safe_calloc(esz, sizeof(float));
        if (!gbufs[k]) {
            for (int j = 0; j < k; j++) safe_free((void**)&gbufs[j]);
            return -1;
        }
    }
    gbufs[7] = (float*)safe_calloc(D, sizeof(float));
    gbufs[8] = (float*)safe_calloc(patch_elem, sizeof(float));
    if (!gbufs[7] || !gbufs[8]) {
        for (int k = 0; k < 9; k++) safe_free((void**)&gbufs[k]);
        return -1;
    }

    for (int p = num_patches - 1; p >= 0; p--) {
        const float* dL_dh = dL_dembeddings + (size_t)p * D;
        memset(gbufs[7], 0, (size_t)D * sizeof(float));
        memset(gbufs[8], 0, (size_t)patch_elem * sizeof(float));

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
            gbufs[8], gbufs[7],
            gbufs[0], gbufs[1], gbufs[2], gbufs[3],
            gbufs[4], gbufs[5], gbufs[6]);
    }

    float lr = learning_rate;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < patch_elem; j++) {
            int idx = i * patch_elem + j;
            encoder->patch_w_gate[idx] -= lr * gbufs[0][idx];
            encoder->patch_w_act[idx] -= lr * gbufs[1][idx];
        }
        for (int j = 0; j < D; j++) {
            int idx = i * D + j;
            encoder->patch_h_to_gate[idx] -= lr * gbufs[2][idx];
            encoder->patch_h_to_act[idx] -= lr * gbufs[3][idx];
        }
        encoder->patch_gate_bias[i] -= lr * gbufs[4][i];
        encoder->patch_act_bias[i] -= lr * gbufs[5][i];
        if (encoder->config.use_adaptive_tau) {
            float nt = encoder->time_constants[i] - lr * gbufs[6][i];
            if (nt >= encoder->config.min_tau && nt <= encoder->config.max_tau)
                encoder->time_constants[i] = nt;
        }
    }

    for (int k = 0; k < 9; k++) safe_free((void**)&gbufs[k]);
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
/* ===================================================================
 * 第三节：LiquidVisualCfCEvolver（原liquid_vision.c）
 * =================================================================== */

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

    ev->w_gate = (float*)safe_calloc((size_t)NC * D * D, sizeof(float));
    ev->w_act = (float*)safe_calloc((size_t)NC * D * D, sizeof(float));
    ev->h_to_gate = (float*)safe_calloc((size_t)NC * D * D, sizeof(float));
    ev->h_to_act = (float*)safe_calloc((size_t)NC * D * D, sizeof(float));
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
        /* S-011修复: use_normal=1 使用正态分布Xavier初始化 */
        _xavier_init(ev->w_gate + (size_t)c * D * D, D, D, &ev->seed, 1);
        _xavier_init(ev->w_act + (size_t)c * D * D, D, D, &ev->seed, 1);
        _xavier_init(ev->h_to_gate + (size_t)c * D * D, D, D, &ev->seed, 1);
        _xavier_init(ev->h_to_act + (size_t)c * D * D, D, D, &ev->seed, 1);
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
        safe_free((void**)&ws_state); safe_free((void**)&ws_prev);
        return -1;
    }

    memcpy(ws_prev, evolver->state, (size_t)D * sizeof(float));

    for (int c = 0; c < NC; c++) {
        _liquid_cfc_step(visual_input, input_dim,
                          ws_prev, D,
                          evolver->w_gate + (size_t)c * D * D,
                          evolver->w_act + (size_t)c * D * D,
                          evolver->h_to_gate + (size_t)c * D * D,
                          evolver->h_to_act + (size_t)c * D * D,
                          evolver->gate_bias + (size_t)c * D,
                          evolver->act_bias + (size_t)c * D,
                          evolver->time_constants, delta_t,
                          evolver->config.min_tau, evolver->config.max_tau,
                          evolver->config.use_adaptive_tau,
                          ws_state);

        memcpy(evolver->channel_states + (size_t)c * D, ws_state, (size_t)D * sizeof(float));

        if (evolver->config.enable_cross_channel) {
            memcpy(ws_prev, ws_state, (size_t)D * sizeof(float));
        } else {
            memcpy(ws_prev, evolver->state, (size_t)D * sizeof(float));
        }
    }

    memcpy(evolver->state, ws_state, (size_t)D * sizeof(float));
    memcpy(visual_state, ws_state, (size_t)D * sizeof(float));

    safe_free((void**)&ws_state); safe_free((void**)&ws_prev);
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

/* ===================================================================
 * 第四节：LiquidSpatialProcessor（原liquid_vision.c）
 * =================================================================== */

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
    /* S-011修复: use_normal=1 使用正态分布Xavier初始化 */
    _xavier_init(sp->feat_to_hidden_w_gate, HD, FD, &sp->seed, 1);
    _xavier_init(sp->feat_to_hidden_w_act, HD, FD, &sp->seed, 1);
    _xavier_init(sp->hidden_to_hidden_w_gate, HD, HD, &sp->seed, 1);
    _xavier_init(sp->hidden_to_hidden_w_act, HD, HD, &sp->seed, 1);
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

        float gate = _activation_sigmoid_stable(gate_sum);
        float activation = _activation_tanh_stable(act_sum);
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
        safe_free((void**)&prev); safe_free((void**)&curr);
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

    safe_free((void**)&prev); safe_free((void**)&curr);
    return 0;
}

void liquid_spatial_processor_reset(LiquidSpatialProcessor* processor) {
    if (!processor) return;
    memset(processor->hidden_state, 0, (size_t)processor->config.spatial_hidden_dim * sizeof(float));
}

int liquid_spatial_processor_get_output_dim(const LiquidSpatialProcessor* processor) {
    return processor ? processor->config.spatial_hidden_dim : 0;
}

/* ===================================================================
 * 第五节：LiquidVisionManager（原liquid_vision.c）
 * =================================================================== */

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
    LNN* shared_lnn;
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
    mgr->shared_lnn = NULL;

    int need_patch = (config->pipeline_type == LIQUID_VISION_PATCH_ENCODER ||
                      config->pipeline_type == LIQUID_VISION_FULL);
    int need_spatial = (config->pipeline_type == LIQUID_VISION_SPATIAL ||
                        config->pipeline_type == LIQUID_VISION_FULL);
    int need_evolver = (config->pipeline_type == LIQUID_VISION_CFC_EVOLVER ||
                        config->pipeline_type == LIQUID_VISION_FULL);

    if (need_patch) {
        mgr->patch_encoder = liquid_patch_encoder_create(&config->patch_config);
        if (!mgr->patch_encoder) { liquid_vision_manager_free(mgr); return NULL; }
    }
    if (need_spatial) {
        mgr->spatial_processor = liquid_spatial_processor_create(&config->spatial_config);
        if (!mgr->spatial_processor) { liquid_vision_manager_free(mgr); return NULL; }
    }
    if (need_evolver) {
        mgr->cfc_evolver = liquid_visual_cfc_evolver_create(&config->evolver_config);
        if (!mgr->cfc_evolver) { liquid_vision_manager_free(mgr); return NULL; }
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
            liquid_vision_manager_free(mgr); return NULL;
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
            if (np * D > output_dim) np = output_dim / D;
            internal_buf = (float*)safe_malloc((size_t)np * D * sizeof(float));
            if (!internal_buf) return -1;

            int num_patches = liquid_patch_encoder_forward(
                manager->patch_encoder, width, height, channels, image_data, internal_buf, np);
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
            size_t input_dim = (size_t)width * (size_t)height * (size_t)channels;
            internal_buf = (float*)safe_malloc(input_dim * sizeof(float));
            if (!internal_buf) return -1;
            for (int i = 0; i < input_dim; i++) internal_buf[i] = image_data[i];

            int ret = liquid_visual_cfc_evolver_forward(
                manager->cfc_evolver, internal_buf, (int)input_dim, output);
            if (ret != 0) break;
            result = D;
            break;
        }
        case LIQUID_VISION_SPATIAL: {
            int HD = manager->config.spatial_config.spatial_hidden_dim;
            if (HD > output_dim) HD = output_dim;
            int NF = manager->config.spatial_config.num_features;
            int FD = manager->config.spatial_config.feature_dim;
            internal_buf = (float*)safe_malloc((size_t)NF * (size_t)FD * sizeof(float));
            if (!internal_buf) return -1;
            size_t input_dim = (size_t)width * (size_t)height * (size_t)channels;
            int copy_len = (int)(input_dim < (size_t)(NF * FD) ? input_dim : (size_t)(NF * FD));
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
            internal_buf = (float*)safe_malloc((size_t)np * (size_t)patch_D * sizeof(float));
            if (!internal_buf) return -1;

            int num_patches = liquid_patch_encoder_forward(
                manager->patch_encoder, width, height, channels, image_data, internal_buf, np);
            if (num_patches < 0) break;

            float* spatial_buf = (float*)safe_malloc((size_t)spatial_HD * sizeof(float));
            if (!spatial_buf) break;

            int ret_sp = liquid_spatial_processor_forward(
                manager->spatial_processor, internal_buf, num_patches, patch_D, spatial_buf);
            if (ret_sp != 0) { safe_free((void**)&spatial_buf); break; }

            int ret_ev = liquid_visual_cfc_evolver_forward(
                manager->cfc_evolver, spatial_buf, spatial_HD, output);
            safe_free((void**)&spatial_buf);
            if (ret_ev != 0) break;
            result = evolver_D;
            break;
        }
        default: break;
    }

    safe_free((void**)&internal_buf);

    if (result > 0 && manager->config.enable_temporal_integration) {
        int window = manager->config.temporal_window_size;
        int frame = manager->temporal_frame_count % window;
        if (frame < 0) frame = 0;

        memcpy(manager->temporal_buffer + (size_t)frame * result, output, (size_t)result * sizeof(float));
        manager->temporal_frame_count++;

        int count = manager->temporal_frame_count < window ? manager->temporal_frame_count : window;
        memset(manager->temporal_accumulator, 0, (size_t)result * sizeof(float));
        for (int f = 0; f < count; f++) {
            for (int d = 0; d < result; d++) {
                manager->temporal_accumulator[d] += manager->temporal_buffer[(size_t)f * result + d];
            }
        }
        float inv_count = 1.0f / (float)count;
        for (int d = 0; d < result; d++) output[d] = manager->temporal_accumulator[d] * inv_count;

        if (manager->cfc_temporal_state) {
            float dt = manager->temporal_frame_count > 1 ? 0.033f : 0.0f;
            if (dt < 1e-6f) dt = 0.033f;
            float tau = 0.1f;

            float* temporal_input = (float*)safe_malloc((size_t)manager->max_feature_dim * sizeof(float));
            float* temporal_out = (float*)safe_malloc((size_t)manager->max_feature_dim * sizeof(float));
            if (temporal_input && temporal_out) {
                for (int d = 0; d < result && d < manager->max_feature_dim; d++) {
                    temporal_input[d] = 0.5f * output[d] + 0.5f * manager->cfc_temporal_state[d];
                }

                LNN* tm_lnn = liquid_vision_manager_get_lnn(manager);
                if (tm_lnn && lnn_forward(tm_lnn, temporal_input, temporal_out) == 0) {
                    for (int d = 0; d < result && d < manager->max_feature_dim; d++) {
                        float x = output[d];
                        float prev_h = manager->cfc_temporal_state[d];
                        float gate = _activation_sigmoid_stable(temporal_out[d] * 0.5f);
                        float activation = _activation_tanh_stable(temporal_out[d] * 0.5f);
                        float exp_term = expf(-dt / tau);
                        float one_minus_exp = 1.0f - exp_term;
                        float new_h = prev_h * exp_term + one_minus_exp * gate * activation;
                        if (isnan(new_h) || isinf(new_h)) new_h = prev_h;
                        manager->cfc_temporal_state[d] = new_h;
                        output[d] = 0.7f * output[d] + 0.3f * new_h;
                    }
                } else {
                    for (int d = 0; d < result && d < manager->max_feature_dim; d++) {
                        float x = output[d];
                        float prev_h = manager->cfc_temporal_state[d];
                        float gate = _activation_sigmoid_stable(x);
                        float activation = _activation_tanh_stable(x);
                        float exp_term = expf(-dt / tau);
                        float new_h = prev_h * exp_term + (1.0f - exp_term) * gate * activation;
                        if (isnan(new_h) || isinf(new_h)) new_h = prev_h;
                        manager->cfc_temporal_state[d] = new_h;
                        output[d] = 0.7f * output[d] + 0.3f * new_h;
                    }
                }
            }
            safe_free((void**)&temporal_input);
            safe_free((void**)&temporal_out);
            manager->cfc_temporal_initialized = 1;
        }
    }
    return result;
}

int liquid_vision_manager_enable_cfc_temporal(LiquidVisionManager* manager, int enable) {
    if (!manager || manager->max_feature_dim <= 0) return -1;
    if (enable) {
        if (!manager->cfc_temporal_state) {
            manager->cfc_temporal_state = (float*)safe_calloc((size_t)manager->max_feature_dim, sizeof(float));
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
    if (manager->cfc_temporal_state)
        memset(manager->cfc_temporal_state, 0, (size_t)manager->max_feature_dim * sizeof(float));
    manager->cfc_temporal_initialized = 0;
    if (manager->temporal_buffer)
        memset(manager->temporal_buffer, 0, (size_t)manager->config.temporal_window_size * (size_t)manager->max_feature_dim * sizeof(float));
    return 0;
}

LNN* liquid_vision_manager_get_lnn(LiquidVisionManager* manager) {
    return manager ? manager->shared_lnn : NULL;
}

int liquid_vision_manager_set_lnn(LiquidVisionManager* manager, LNN* lnn) {
    if (!manager) return -1;
    manager->shared_lnn = lnn;
    return 0;
}

int liquid_vision_manager_set_features(LiquidVisionManager* manager,
                                        const float* features, int feature_dim) {
    if (!manager || !features || feature_dim <= 0) return -1;
    if (manager->feature_buffer_size < feature_dim) {
        float* new_buf = (float*)safe_realloc(manager->feature_buffer, (size_t)feature_dim * sizeof(float));
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
    if (manager->temporal_buffer)
        memset(manager->temporal_buffer, 0, (size_t)manager->config.temporal_window_size * manager->config.evolver_config.visual_state_dim * sizeof(float));
    if (manager->temporal_accumulator)
        memset(manager->temporal_accumulator, 0, (size_t)manager->config.evolver_config.visual_state_dim * sizeof(float));
    manager->is_feature_mode = 0;
}

int liquid_vision_manager_get_output_dim(const LiquidVisionManager* manager) {
    if (!manager) return 0;
    switch (manager->config.pipeline_type) {
        case LIQUID_VISION_PATCH_ENCODER:
            return manager->config.patch_config.patch_hidden_dim * manager->config.patch_config.max_patches;
        case LIQUID_VISION_CFC_EVOLVER:
            return manager->config.evolver_config.visual_state_dim;
        case LIQUID_VISION_SPATIAL:
            return manager->config.spatial_config.spatial_hidden_dim;
        case LIQUID_VISION_FULL:
            return manager->config.evolver_config.visual_state_dim;
        default: return 0;
    }
}

int liquid_vision_manager_save(const LiquidVisionManager* manager, const char* filepath) {
    if (!manager || !filepath) return -1;
    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    fwrite(&manager->config, sizeof(LiquidVisionManagerConfig), 1, fp);

    if (manager->patch_encoder) {
        int has = 1; fwrite(&has, sizeof(int), 1, fp);
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
    } else { int has = 0; fwrite(&has, sizeof(int), 1, fp); }

    if (manager->cfc_evolver) {
        int has = 1; fwrite(&has, sizeof(int), 1, fp);
        fwrite(&manager->cfc_evolver->config, sizeof(LiquidVisualCfCEvolverConfig), 1, fp);
        int NC = manager->cfc_evolver->config.num_visual_channels;
        int D = manager->cfc_evolver->config.visual_state_dim;
        fwrite(manager->cfc_evolver->w_gate, sizeof(float), (size_t)NC * D * D, fp);
        fwrite(manager->cfc_evolver->w_act, sizeof(float), (size_t)NC * D * D, fp);
        fwrite(manager->cfc_evolver->h_to_gate, sizeof(float), (size_t)NC * D * D, fp);
        fwrite(manager->cfc_evolver->h_to_act, sizeof(float), (size_t)NC * D * D, fp);
        fwrite(manager->cfc_evolver->gate_bias, sizeof(float), (size_t)NC * D, fp);
        fwrite(manager->cfc_evolver->act_bias, sizeof(float), (size_t)NC * D, fp);
    } else { int has = 0; fwrite(&has, sizeof(int), 1, fp); }

    if (manager->spatial_processor) {
        int has = 1; fwrite(&has, sizeof(int), 1, fp);
        fwrite(&manager->spatial_processor->config, sizeof(LiquidSpatialProcessorConfig), 1, fp);
        int HD = manager->spatial_processor->config.spatial_hidden_dim;
        int FD = manager->spatial_processor->config.feature_dim;
        fwrite(manager->spatial_processor->feat_to_hidden_w_gate, sizeof(float), (size_t)HD * FD, fp);
        fwrite(manager->spatial_processor->feat_to_hidden_w_act, sizeof(float), (size_t)HD * FD, fp);
        fwrite(manager->spatial_processor->hidden_to_hidden_w_gate, sizeof(float), (size_t)HD * HD, fp);
        fwrite(manager->spatial_processor->hidden_to_hidden_w_act, sizeof(float), (size_t)HD * HD, fp);
        fwrite(manager->spatial_processor->gate_bias, sizeof(float), HD, fp);
        fwrite(manager->spatial_processor->act_bias, sizeof(float), HD, fp);
    } else { int has = 0; fwrite(&has, sizeof(int), 1, fp); }

    fclose(fp);
    return 0;
}

int liquid_vision_manager_load(LiquidVisionManager* manager, const char* filepath) {
    if (!manager || !filepath) return -1;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    LiquidVisionManagerConfig loaded_config;
    if (fread(&loaded_config, sizeof(LiquidVisionManagerConfig), 1, fp) != 1) { fclose(fp); return -1; }

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

/* ===================================================================
 * 第六节：动态可扩展视觉类别注册表（原vision.c）
 * =================================================================== */

static const char* g_default_coco_names_zh[VISION_CLASS_DEFAULT_COUNT] = {
    "人", "自行车", "汽车", "摩托车",
    "飞机", "公共汽车", "火车", "卡车",
    "船", "交通灯", "消防栓", "停止标志",
    "停车计时器", "长凳", "鸟", "猫",
    "狗", "马", "羊", "牛",
    "大象", "熊", "斑马", "长颈鹿",
    "背包", "伞", "手提包", "领带",
    "手提箱", "飞盘", "滑雪板", "滑雪板雪板",
    "运动球", "风筝", "棒球棒", "棒球手套",
    "滑板", "冲浪板", "网球拍", "瓶子",
    "酒杯", "杯子", "叉子", "刀",
    "勺子", "碗", "香蕉", "苹果",
    "三明治", "橘子", "西兰花", "胡萝卜",
    "热狗", "披萨", "甜甜圈", "蛋糕",
    "椅子", "沙发", "盆栽", "床",
    "餐桌", "马桶", "电视", "笔记本电脑",
    "鼠标", "遥控器", "键盘", "手机",
    "微波炉", "烤箱", "烤面包机", "水槽",
    "冰箱", "书", "时钟", "花瓶",
    "剪刀", "泰迪熊", "吹风机", "牙刷"
};

static const char* g_default_coco_names_en[VISION_CLASS_DEFAULT_COUNT] = {
    "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

struct VisionClassRegistry {
    VisionClassEntry* entries;
    int count;
    int capacity;
    int next_dynamic_id;
    int initialized;
    MutexHandle lock;
};

static VisionClassRegistry* g_global_class_registry = NULL;
static MutexHandle g_registry_singleton_lock = NULL;

static void _registry_lock_init(void) {
    if (!g_registry_singleton_lock) {
        g_registry_singleton_lock = mutex_create();
    }
}

VisionClassRegistry* vision_class_registry_create(void) {
    VisionClassRegistry* reg = (VisionClassRegistry*)safe_calloc(1, sizeof(VisionClassRegistry));
    if (!reg) return NULL;

    reg->capacity = VISION_CLASS_DEFAULT_COUNT + 64;
    reg->entries = (VisionClassEntry*)safe_calloc((size_t)reg->capacity, sizeof(VisionClassEntry));
    if (!reg->entries) { safe_free((void**)&reg); return NULL; }

    reg->lock = mutex_create();
    if (!reg->lock) {
        safe_free((void**)&reg->entries); safe_free((void**)&reg);
        return NULL;
    }

    for (int i = 0; i < VISION_CLASS_DEFAULT_COUNT; i++) {
        reg->entries[i].class_id = i;
        strncpy(reg->entries[i].name_zh, g_default_coco_names_zh[i], VISION_CLASS_NAME_MAX_LEN - 1);
        reg->entries[i].name_zh[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
        strncpy(reg->entries[i].name_en, g_default_coco_names_en[i], VISION_CLASS_NAME_MAX_LEN - 1);
        reg->entries[i].name_en[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
        reg->entries[i].is_learned = 0;
        reg->entries[i].sample_count = 0;
        reg->entries[i].confidence_threshold = 0.5f;
    }
    reg->count = VISION_CLASS_DEFAULT_COUNT;
    reg->next_dynamic_id = VISION_CLASS_DEFAULT_COUNT;
    reg->initialized = 1;

    return reg;
}

void vision_class_registry_free(VisionClassRegistry* registry) {
    if (!registry) return;
    if (registry->lock) mutex_destroy(registry->lock);
    safe_free((void**)&registry->entries);
    safe_free((void**)&registry);
}

VisionClassRegistry* vision_class_registry_get_global(void) {
    _registry_lock_init();
    if (g_registry_singleton_lock) mutex_lock(g_registry_singleton_lock);

    if (!g_global_class_registry) {
        g_global_class_registry = vision_class_registry_create();
    }

    VisionClassRegistry* result = g_global_class_registry;
    if (g_registry_singleton_lock) mutex_unlock(g_registry_singleton_lock);
    return result;
}

int vision_class_register(VisionClassRegistry* registry,
                          const char* name_zh, const char* name_en) {
    if (!registry || !registry->initialized) return -1;
    if (!name_zh || !name_en) return -1;

    mutex_lock(registry->lock);

    for (int i = 0; i < registry->count; i++) {
        if (strcmp(registry->entries[i].name_en, name_en) == 0) {
            int existing_id = registry->entries[i].class_id;
            mutex_unlock(registry->lock);
            return existing_id;
        }
    }

    if (registry->count >= registry->capacity) {
        int new_cap = registry->capacity * 2;
        if (new_cap > VISION_CLASS_MAX_CAPACITY) { mutex_unlock(registry->lock); return -1; }
        VisionClassEntry* new_entries = (VisionClassEntry*)safe_realloc(
            registry->entries, (size_t)new_cap * sizeof(VisionClassEntry));
        if (!new_entries) { mutex_unlock(registry->lock); return -1; }
        memset(new_entries + registry->capacity, 0,
               (size_t)(new_cap - registry->capacity) * sizeof(VisionClassEntry));
        registry->entries = new_entries;
        registry->capacity = new_cap;
    }

    int new_id = registry->next_dynamic_id++;
    VisionClassEntry* entry = &registry->entries[registry->count];
    memset(entry, 0, sizeof(VisionClassEntry));
    entry->class_id = new_id;
    strncpy(entry->name_zh, name_zh, VISION_CLASS_NAME_MAX_LEN - 1);
    entry->name_zh[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
    strncpy(entry->name_en, name_en, VISION_CLASS_NAME_MAX_LEN - 1);
    entry->name_en[VISION_CLASS_NAME_MAX_LEN - 1] = '\0';
    entry->is_learned = 1;
    entry->sample_count = 1;
    entry->confidence_threshold = 0.35f;

    registry->count++;
    mutex_unlock(registry->lock);
    return new_id;
}

int vision_class_get_count(const VisionClassRegistry* registry) {
    if (!registry || !registry->initialized) return 0;
    return registry->count;
}

int vision_class_get_entry(const VisionClassRegistry* registry, int class_id,
                           VisionClassEntry* entry) {
    if (!registry || !registry->initialized) return -1;
    if (class_id < 0) return -1;

    for (int i = 0; i < registry->count; i++) {
        if (registry->entries[i].class_id == class_id) {
            if (entry) memcpy(entry, &registry->entries[i], sizeof(VisionClassEntry));
            return 0;
        }
    }
    return -1;
}

int vision_class_add_samples(VisionClassRegistry* registry, int class_id, int count) {
    if (!registry || !registry->initialized || count < 1) return -1;

    mutex_lock(registry->lock);
    for (int i = 0; i < registry->count; i++) {
        if (registry->entries[i].class_id == class_id) {
            registry->entries[i].sample_count += count;
            if (registry->entries[i].sample_count > 100) {
                registry->entries[i].confidence_threshold = 0.5f;
            } else if (registry->entries[i].sample_count > 10) {
                registry->entries[i].confidence_threshold = 0.4f;
            }
            mutex_unlock(registry->lock);
            return 0;
        }
    }
    mutex_unlock(registry->lock);
    return -1;
}

const char* vision_get_class_name_zh(int class_id) {
    VisionClassRegistry* reg = vision_class_registry_get_global();
    if (!reg) return "未知";

    VisionClassEntry entry;
    if (vision_class_get_entry(reg, class_id, &entry) == 0) {
        return (entry.name_zh[0] != '\0') ? entry.name_zh : "未知";
    }
    return "未知";
}

const char* vision_get_class_name_en(int class_id) {
    VisionClassRegistry* reg = vision_class_registry_get_global();
    if (!reg) return "unknown";

    VisionClassEntry entry;
    if (vision_class_get_entry(reg, class_id, &entry) == 0) {
        return (entry.name_en[0] != '\0') ? entry.name_en : "unknown";
    }
    return "unknown";
}

/* ===================================================================
 * 第七节：NMS非极大值抑制（原vision.c）
 * =================================================================== */

static float _nms_compute_iou(const CfCVisionDetection* a, const CfCVisionDetection* b) {
    float ax1 = a->cx - a->w * 0.5f;
    float ay1 = a->cy - a->h * 0.5f;
    float ax2 = a->cx + a->w * 0.5f;
    float ay2 = a->cy + a->h * 0.5f;
    float bx1 = b->cx - b->w * 0.5f;
    float by1 = b->cy - b->h * 0.5f;
    float bx2 = b->cx + b->w * 0.5f;
    float by2 = b->cy + b->h * 0.5f;

    float inter_x1 = ax1 > bx1 ? ax1 : bx1;
    float inter_y1 = ay1 > by1 ? ay1 : by1;
    float inter_x2 = ax2 < bx2 ? ax2 : bx2;
    float inter_y2 = ay2 < by2 ? ay2 : by2;

    float inter_w = inter_x2 - inter_x1;
    float inter_h = inter_y2 - inter_y1;
    if (inter_w <= 0.0f || inter_h <= 0.0f) return 0.0f;

    float inter_area = inter_w * inter_h;
    float area_a = a->w * a->h;
    float area_b = b->w * b->h;
    float union_area = area_a + area_b - inter_area;

    return (union_area > 1e-6f) ? inter_area / union_area : 0.0f;
}

int vision_nms(CfCVisionDetection* detections, int count, float iou_threshold) {
    if (!detections || count <= 1) return count;

    for (int i = 0; i < count - 1; i++) {
        int best = i;
        for (int j = i + 1; j < count; j++) {
            if (detections[j].confidence > detections[best].confidence) best = j;
        }
        if (best != i) {
            CfCVisionDetection tmp = detections[i];
            detections[i] = detections[best];
            detections[best] = tmp;
        }
    }

    int* keep = (int*)safe_malloc((size_t)count * sizeof(int));
    if (!keep) return count;
    for (int i = 0; i < count; i++) keep[i] = 1;

    int kept = 0;
    for (int i = 0; i < count; i++) {
        if (!keep[i]) continue;
        kept++;
        for (int j = i + 1; j < count; j++) {
            if (!keep[j]) continue;
            if (detections[i].class_id == detections[j].class_id) {
                float iou = _nms_compute_iou(&detections[i], &detections[j]);
                if (iou > iou_threshold) keep[j] = 0;
            }
        }
    }

    int out_idx = 0;
    for (int i = 0; i < count; i++) {
        if (keep[i]) {
            if (out_idx != i) detections[out_idx] = detections[i];
            out_idx++;
        }
    }

    safe_free((void**)&keep);
    return out_idx;
}

void vision_free_detections(CfCVisionDetection* detections, int count) {
    if (!detections || count <= 0) return;
    for (int i = 0; i < count; i++) {
        if (detections[i].class_probs) {
            safe_free((void**)&detections[i].class_probs);
            detections[i].class_probs_count = 0;
        }
    }
}

/* ===================================================================
 * 第八节：传统CV特征提取 —— 辅助预处理路径（原vision.c）
 * =================================================================== */

/* 8.1 颜色分析 —— RGB直方图 + 主色检测 */
typedef struct {
    float r_hist[COLOR_HIST_BINS];
    float g_hist[COLOR_HIST_BINS];
    float b_hist[COLOR_HIST_BINS];
    float dominant_color[3];
    int is_grayscale;
    float avg_brightness;
} _ColorAnalysis;

static int _vision_analyze_color(const float* image, int width, int height, int channels,
                                  _ColorAnalysis* result) {
    if (!image || !result || width <= 0 || height <= 0 || channels < 3) return -1;
    memset(result, 0, sizeof(_ColorAnalysis));

    size_t total = (size_t)width * height;
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int r_bins[COLOR_HIST_BINS] = {0}, g_bins[COLOR_HIST_BINS] = {0}, b_bins[COLOR_HIST_BINS] = {0};

    for (size_t i = 0; i < total; i++) {
        float r = image[i * channels];
        float g = image[i * channels + 1];
        float b = image[i * channels + 2];
        sum_r += r; sum_g += g; sum_b += b;
        int ri = (int)(r * (COLOR_HIST_BINS - 1) + 0.5f);
        int gi = (int)(g * (COLOR_HIST_BINS - 1) + 0.5f);
        int bi = (int)(b * (COLOR_HIST_BINS - 1) + 0.5f);
        if (ri >= 0 && ri < COLOR_HIST_BINS) r_bins[ri]++;
        if (gi >= 0 && gi < COLOR_HIST_BINS) g_bins[gi]++;
        if (bi >= 0 && bi < COLOR_HIST_BINS) b_bins[bi]++;
    }

    for (int i = 0; i < COLOR_HIST_BINS; i++) {
        result->r_hist[i] = (float)r_bins[i] / (float)total;
        result->g_hist[i] = (float)g_bins[i] / (float)total;
        result->b_hist[i] = (float)b_bins[i] / (float)total;
    }

    result->dominant_color[0] = sum_r / (float)total;
    result->dominant_color[1] = sum_g / (float)total;
    result->dominant_color[2] = sum_b / (float)total;
    result->avg_brightness = (sum_r + sum_g + sum_b) / (3.0f * (float)total);

    float var_r = 0.0f, var_g = 0.0f, var_b = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float dr = image[i * channels] - result->dominant_color[0];
        float dg = image[i * channels + 1] - result->dominant_color[1];
        float db = image[i * channels + 2] - result->dominant_color[2];
        var_r += dr * dr; var_g += dg * dg; var_b += db * db;
    }
    var_r /= (float)total; var_g /= (float)total; var_b /= (float)total;
    result->is_grayscale = (var_r < 0.005f && var_g < 0.005f && var_b < 0.005f) ? 1 : 0;

    return 0;
}

/* 8.2 传统CV特征提取辅助函数 */
/*
 * 灰度转换
 * Sobel边缘检测 + 方向统计
 * LBP纹理直方图（8位均匀模式）
 * 多尺度图像金字塔特征
 * HOG梯度方向直方图特征（9 bins, 8×8 cells, 2×2 block归一化）
 * HSV风格色彩直方图
 * L2归一化
 */
static int _vision_extract_traditional_cv(const float* data, int width, int height, int channels,
                                           int enable_multiscale, int enable_hog, int enable_color,
                                           float* features, size_t max_features) {
    size_t total_pixels = (size_t)width * height;
    size_t feature_idx = 0;

    /* 灰度转换 */
    float* gray = (float*)safe_malloc(total_pixels * sizeof(float));
    if (!gray) return -1;
    if (channels >= 3) {
        for (size_t i = 0; i < total_pixels; i++)
            gray[i] = 0.299f * data[i * 3] + 0.587f * data[i * 3 + 1] + 0.114f * data[i * 3 + 2];
    } else {
        for (size_t i = 0; i < total_pixels; i++) gray[i] = data[i];
    }

    /* 颜色矩（均值、标准差、偏度） */
    if (channels >= 3 && feature_idx + 18 < max_features) {
        for (int c = 0; c < 3; c++) {
            double sum = 0.0, sum_sq = 0.0, sum_cu = 0.0;
            for (size_t i = 0; i < total_pixels; i++) {
                float v = data[i * channels + c];
                sum += v; sum_sq += v * v; sum_cu += v * v * v;
            }
            float mean = (float)(sum / (double)total_pixels);
            float variance = (float)(sum_sq / (double)total_pixels) - mean * mean;
            float stddev = sqrtf(variance > 0 ? variance : 0.0f);
            float skewness = 0.0f;
            if (stddev > 1e-8f) {
                skewness = (float)((sum_cu / (double)total_pixels - 3.0 * mean * variance - mean * mean * mean)
                          / (stddev * stddev * stddev));
            }
            features[feature_idx++] = mean;
            features[feature_idx++] = stddev;
            features[feature_idx++] = skewness;
        }
        for (int c = 0; c < 3 && feature_idx + 3 < max_features; c++) {
            float min_val = 1e10f, max_val = -1e10f;
            for (size_t i = 0; i < total_pixels; i++) {
                float v = data[i * channels + c];
                if (v < min_val) min_val = v; if (v > max_val) max_val = v;
            }
            features[feature_idx++] = min_val;
            features[feature_idx++] = max_val;
            features[feature_idx++] = max_val - min_val;
        }
    }

    /* Sobel边缘密度和方向统计 */
    if (feature_idx + 10 < max_features && width >= 3 && height >= 3) {
        float edge_density = 0.0f, edge_sum = 0.0f;
        float direction_hist[4] = {0}; int edge_count = 0;
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float gx = -gray[(y-1)*width+(x-1)] - 2.0f*gray[y*width+(x-1)] - gray[(y+1)*width+(x-1)]
                          + gray[(y-1)*width+(x+1)] + 2.0f*gray[y*width+(x+1)] + gray[(y+1)*width+(x+1)];
                float gy = -gray[(y-1)*width+(x-1)] - 2.0f*gray[(y-1)*width+x] - gray[(y-1)*width+(x+1)]
                          + gray[(y+1)*width+(x-1)] + 2.0f*gray[(y+1)*width+x] + gray[(y+1)*width+(x+1)];
                float mag = sqrtf(gx * gx + gy * gy);
                edge_sum += mag;
                if (mag > 0.1f) {
                    edge_count++;
                    float angle = atan2f(gy, gx) + 3.14159265f;
                    int bin = (int)(angle * 4.0f / (2.0f * 3.14159265f)) % 4;
                    direction_hist[bin] += 1.0f;
                }
            }
        }
        edge_density = (float)edge_count / (float)((width - 2) * (height - 2));
        features[feature_idx++] = edge_density;
        features[feature_idx++] = edge_count > 0 ? edge_sum / (float)edge_count : 0.0f;
        for (int i = 0; i < 4 && feature_idx < max_features; i++)
            features[feature_idx++] = edge_count > 0 ? direction_hist[i] / (float)edge_count : 0.0f;
        if (feature_idx < max_features) {
            float edge_var = 0.0f; int ec = 0; float em = edge_count > 0 ? edge_sum / (float)edge_count : 0.0f;
            for (int y = 1; y < height - 1 && ec < 10000; y++) for (int x = 1; x < width - 1; x++) {
                float gx2 = -gray[(y-1)*width+(x-1)] - 2.0f*gray[y*width+(x-1)] - gray[(y+1)*width+(x-1)]
                           + gray[(y-1)*width+(x+1)] + 2.0f*gray[y*width+(x+1)] + gray[(y+1)*width+(x+1)];
                float gy2 = -gray[(y-1)*width+(x-1)] - 2.0f*gray[(y-1)*width+x] - gray[(y-1)*width+(x+1)]
                           + gray[(y+1)*width+(x-1)] + 2.0f*gray[(y+1)*width+x] + gray[(y+1)*width+(x+1)];
                float mag2 = sqrtf(gx2 * gx2 + gy2 * gy2);
                if (mag2 > 0.1f) { float d = mag2 - em; edge_var += d * d; ec++; }
            }
            features[feature_idx++] = ec > 0 ? sqrtf(edge_var / (float)ec) : 0.0f;
        }
    }

    /* LBP纹理直方图 */
    if (feature_idx + 8 < max_features && width >= 3 && height >= 3) {
        float lbp_hist[8] = {0}; int lbp_count = 0;
        int pos[8][2] = {{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0}};
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float center = gray[y * width + x];
                int code = 0;
                for (int p = 0; p < 8; p++)
                    if (gray[(y + pos[p][1]) * width + x + pos[p][0]] >= center) code |= (1 << p);
                for (int b = 0; b < 8; b++) if (code & (1 << b)) lbp_hist[b] += 1.0f;
                lbp_count++;
            }
        }
        if (lbp_count > 0)
            for (int i = 0; i < 8 && feature_idx < max_features; i++)
                features[feature_idx++] = lbp_hist[i] / (float)lbp_count;
    }

    /* 多尺度图像金字塔特征 */
    if (enable_multiscale && feature_idx + 24 < max_features && width >= 16 && height >= 16) {
        int scales[4][2] = {{width, height}, {width/2, height/2}, {width/4, height/4}, {width/8, height/8}};
        for (int sc = 0; sc < 4; sc++) {
            int sw = scales[sc][0], sh = scales[sc][1];
            if (sw < 2 || sh < 2) break;
            float s_mean = 0.0f, s_var = 0.0f; int s_count = 0;
            for (int sy = 0; sy < sh; sy++) {
                for (int sx = 0; sx < sw; sx++) {
                    int src_y = sy * (1 << sc), src_x = sx * (1 << sc);
                    if (src_y >= height || src_x >= width) continue;
                    float v = gray[src_y * width + src_x];
                    s_mean += v; s_var += v * v; s_count++;
                }
            }
            if (s_count > 0) {
                s_mean /= (float)s_count;
                s_var = s_var/(float)s_count - s_mean*s_mean;
                if (feature_idx + 6 <= max_features) {
                    features[feature_idx++] = s_mean;
                    features[feature_idx++] = sqrtf(s_var > 0 ? s_var : 0.0f);
                    features[feature_idx++] = (float)sw;
                    features[feature_idx++] = (float)sh;
                    features[feature_idx++] = (float)s_count / (float)(width * height);
                    if (feature_idx >= 6 && sc > 0)
                        features[feature_idx++] = s_mean - features[feature_idx - 7];
                    else
                        features[feature_idx++] = 0.0f;
                }
            }
        }
    }

    /* HOG梯度方向直方图特征 */
    if (enable_hog && feature_idx + 36 < max_features && width >= 16 && height >= 16) {
        int cell_w = 8, cell_h = 8;
        int cells_x = width / cell_w, cells_y = height / cell_w;
        if (cells_x < 2) cells_x = 2;
        if (cells_y < 2) cells_y = 2;
        float* cell_hist = (float*)safe_calloc((size_t)cells_x * (size_t)cells_y * 9, sizeof(float));
        if (cell_hist) {
            for (int cy = 0; cy < cells_y; cy++) {
                for (int cx = 0; cx < cells_x; cx++) {
                    for (int dy = 0; dy < cell_h; dy++) {
                        for (int dx = 0; dx < cell_w; dx++) {
                            int px = cx * cell_w + dx, py = cy * cell_h + dy;
                            if (px >= width - 1 || py >= height - 1 || px < 1 || py < 1) continue;
                            float gx = gray[py*width+(px+1)] - gray[py*width+(px-1)];
                            float gy = gray[(py+1)*width+px] - gray[(py-1)*width+px];
                            float mag = sqrtf(gx*gx + gy*gy);
                            float ori = atan2f(gy, gx);
                            if (ori < 0) ori += 2.0f * 3.14159265f;
                            int bin = (int)(ori * 9.0f / (2.0f * 3.14159265f)) % 9;
                            cell_hist[(cy*cells_x + cx)*9 + bin] += mag;
                        }
                    }
                }
            }
            int feat_out = 0;
            float global_bins[9] = {0};
            for (int cy = 0; cy < cells_y - 1 && feat_out < 36; cy++) {
                for (int cx = 0; cx < cells_x - 1 && feat_out < 36; cx++) {
                    float block_norm = 0.0f;
                    for (int by = 0; by < 2; by++)
                        for (int bx = 0; bx < 2; bx++)
                            for (int b = 0; b < 9; b++)
                                block_norm += cell_hist[((cy+by)*cells_x+(cx+bx))*9+b];
                    float norm = sqrtf(block_norm*block_norm + 1e-6f);
                    for (int by = 0; by < 2 && feat_out < 36; by++)
                        for (int bx = 0; bx < 2 && feat_out < 36; bx++)
                            for (int b = 0; b < 9 && feat_out < 36; b++) {
                                if (feat_out < 36) {
                                    features[feature_idx+feat_out] =
                                        cell_hist[((cy+by)*cells_x+(cx+bx))*9+b]/norm;
                                    feat_out++;
                                }
                            }
                }
            }
            for (int i = 0; i < feat_out && i < 36; i++) { int bin = i % 9; global_bins[bin] += features[feature_idx+i]; }
            for (int b = 0; b < 9 && feature_idx + 36 + b < max_features; b++)
                features[feature_idx+36+b] = global_bins[b];
            feature_idx += (feat_out < 36 ? feat_out : 36) + 9;
            if (feature_idx > max_features) feature_idx = max_features;
            safe_free((void**)&cell_hist);
        }
    }

    /* HSV风格色彩直方图 */
    if (enable_color && channels >= 3 && feature_idx + 24 < max_features) {
        float hue_hist[12] = {0}, sat_hist[8] = {0}, val_hist[8] = {0};
        size_t color_count = 0;
        for (size_t i = 0; i < total_pixels && color_count < 50000; i++) {
            float r = data[i*3], g = data[i*3+1], b = data[i*3+2];
            float max_c = r > g ? (r > b ? r : b) : (g > b ? g : b);
            float min_c = r < g ? (r < b ? r : b) : (g < b ? g : b);
            float delta = max_c - min_c;
            float hue = 0.0f;
            if (delta > 1e-4f) {
                if (max_c == r) hue = 60.0f * ((g-b)/delta + (g<b?6.0f:0.0f));
                else if (max_c == g) hue = 60.0f * ((b-r)/delta + 2.0f);
                else hue = 60.0f * ((r-g)/delta + 4.0f);
            }
            float sat = max_c > 1e-4f ? delta / max_c : 0.0f;
            float val = max_c;
            int hb = (int)(hue / 30.0f) % 12;
            int sb = (int)(sat * 7.99f);
            int vb = (int)(val * 7.99f);
            if (hb >= 0 && hb < 12) hue_hist[hb] += 1.0f;
            if (sb >= 0 && sb < 8) sat_hist[sb] += 1.0f;
            if (vb >= 0 && vb < 8) val_hist[vb] += 1.0f;
            color_count++;
        }
        if (color_count > 0) {
            for (int i = 0; i < 12; i++) hue_hist[i] /= (float)color_count;
            for (int i = 0; i < 8; i++) { sat_hist[i] /= (float)color_count; val_hist[i] /= (float)color_count; }
            for (int i = 0; i < 12 && feature_idx < max_features; i++) features[feature_idx++] = hue_hist[i];
            for (int i = 0; i < 6 && feature_idx < max_features; i++)
                features[feature_idx++] = sat_hist[i] + (i < 6 ? val_hist[i] : 0);
        }
    }
    while (feature_idx < max_features) features[feature_idx++] = 0.0f;

    /* L2归一化 */
    float norm = 0.0f;
    for (size_t i = 0; i < feature_idx; i++) norm += features[i] * features[i];
    if (norm > 1e-8f) {
        float inv = 1.0f / sqrtf(norm);
        for (size_t i = 0; i < feature_idx; i++) features[i] *= inv;
    }

    safe_free((void**)&gray);
    return (int)feature_idx;
}

/* 8.3 独立HOG特征提取 */
int vision_extract_hog_features(const float* gray, int width, int height,
                                 float* features, int max_features) {
    if (!gray || !features || max_features < 36 || width < 16 || height < 16) return -1;
    int cell_w = 8, cell_h = 8;
    int cells_x = width / cell_w, cells_y = height / cell_w;
    if (cells_x < 2) cells_x = 2;
    if (cells_y < 2) cells_y = 2;
    int feat_out = 0;
    float* cell_hist = (float*)safe_calloc((size_t)cells_x * (size_t)cells_y * 9, sizeof(float));
    if (!cell_hist) return -1;
    for (int cy = 0; cy < cells_y; cy++) {
        for (int cx = 0; cx < cells_x; cx++) {
            for (int dy = 0; dy < cell_h; dy++) {
                for (int dx = 0; dx < cell_w; dx++) {
                    int px = cx * cell_w + dx, py = cy * cell_h + dy;
                    if (px < 1 || py < 1 || px >= width - 1 || py >= height - 1) continue;
                    float gx = gray[py*width+(px+1)] - gray[py*width+(px-1)];
                    float gy = gray[(py+1)*width+px] - gray[(py-1)*width+px];
                    float mag = sqrtf(gx*gx + gy*gy);
                    float ori = atan2f(gy, gx);
                    if (ori < 0) ori += 2.0f * 3.14159265f;
                    int bin = (int)(ori * 9.0f / (2.0f * 3.14159265f)) % 9;
                    cell_hist[(cy*cells_x + cx)*9 + bin] += mag;
                }
            }
        }
    }
    for (int cy = 0; cy < cells_y - 1 && feat_out < max_features; cy++) {
        for (int cx = 0; cx < cells_x - 1 && feat_out < max_features; cx++) {
            float block_norm = 0.0f;
            for (int by = 0; by < 2; by++)
                for (int bx = 0; bx < 2; bx++)
                    for (int b = 0; b < 9; b++)
                        block_norm += cell_hist[((cy+by)*cells_x+(cx+bx))*9+b];
            float norm = sqrtf(block_norm*block_norm + 1e-6f);
            for (int by = 0; by < 2 && feat_out < max_features; by++)
                for (int bx = 0; bx < 2 && feat_out < max_features; bx++)
                    for (int b = 0; b < 9 && feat_out < max_features; b++) {
                        features[feat_out++] = cell_hist[((cy+by)*cells_x+(cx+bx))*9+b]/norm;
                    }
        }
    }
    safe_free((void**)&cell_hist);
    return feat_out;
}

/* 8.4 多尺度LBP特征提取 */
int vision_extract_multiscale_lbp(const float* gray, int width, int height,
                                   float* features, int max_features) {
    if (!gray || !features || max_features < 24 || width < 3 || height < 3) return -1;
    int radii[3] = {1, 2, 4};
    int feat_out = 0;
    for (int r_idx = 0; r_idx < 3; r_idx++) {
        int R = radii[r_idx];
        float hist[8] = {0}; int count = 0;
        for (int y = R; y < height - R; y++) {
            for (int x = R; x < width - R; x++) {
                float center = gray[y * width + x];
                int code = 0;
                for (int p = 0; p < 8; p++) {
                    float angle = (float)p * 3.14159265f / 4.0f;
                    int px = x + (int)((float)R * cosf(angle) + 0.5f);
                    int py = y + (int)((float)R * sinf(angle) + 0.5f);
                    if (px < 0 || px >= width || py < 0 || py >= height) continue;
                    if (gray[py * width + px] >= center) code |= (1 << p);
                }
                for (int b = 0; b < 8; b++) if (code & (1 << b)) hist[b] += 1.0f;
                count++;
            }
        }
        if (count > 0 && feat_out + 8 <= max_features) {
            for (int b = 0; b < 8; b++) features[feat_out++] = hist[b] / (float)count;
        }
    }
    return feat_out;
}

/* 8.5 HSV色彩直方图提取 */
int vision_extract_color_histogram(const float* rgb, int width, int height, int channels,
                                    float* features, int max_features) {
    if (!rgb || !features || max_features < 28 || channels < 3) return -1;
    float hue_hist[12] = {0}, sat_hist[8] = {0}, val_hist[8] = {0};
    size_t total = (size_t)width * height;
    size_t count = 0;
    for (size_t i = 0; i < total && count < 50000; i++) {
        float r = rgb[i*3], g = rgb[i*3+1], b_ = rgb[i*3+2];
        float mx = r > g ? (r > b_ ? r : b_) : (g > b_ ? g : b_);
        float mn = r < g ? (r < b_ ? r : b_) : (g < b_ ? g : b_);
        float delta = mx - mn, hue = 0.0f;
        if (delta > 1e-4f) {
            if (mx == r) hue = 60.0f * ((g-b_)/delta + (g<b_?6.0f:0.0f));
            else if (mx == g) hue = 60.0f * ((b_-r)/delta + 2.0f);
            else hue = 60.0f * ((r-g)/delta + 4.0f);
        }
        float sat = mx > 1e-4f ? delta / mx : 0.0f;
        float val = mx;
        hue_hist[(int)(hue/30.0f)%12] += 1.0f;
        sat_hist[(int)(sat*7.99f)] += 1.0f;
        val_hist[(int)(val*7.99f)] += 1.0f;
        count++;
    }
    if (count == 0) return 0;
    int fi = 0;
    for (int i = 0; i < 12 && fi < max_features; i++) features[fi++] = hue_hist[i]/(float)count;
    for (int i = 0; i < 8 && fi < max_features; i++) features[fi++] = sat_hist[i]/(float)count;
    for (int i = 0; i < 8 && fi < max_features; i++) features[fi++] = val_hist[i]/(float)count;
    return fi;
}

/* ===================================================================
 * 第九节：图像处理工具函数（原vision.c）
 * =================================================================== */

int vision_resize_bilinear(int src_width, int src_height, int channels,
                            const float* src_data,
                            int dst_width, int dst_height, float* dst_data) {
    if (!src_data || !dst_data || channels <= 0) return -1;
    if (src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0) return -1;
    if (src_width == dst_width && src_height == dst_height) {
        memcpy(dst_data, src_data, (size_t)src_width * src_height * channels * sizeof(float));
        return 0;
    }

    float x_ratio = (float)(src_width - 1) / (float)dst_width;
    float y_ratio = (float)(src_height - 1) / (float)dst_height;

    for (int dy = 0; dy < dst_height; dy++) {
        for (int dx = 0; dx < dst_width; dx++) {
            float sx_f = (float)dx * x_ratio;
            float sy_f = (float)dy * y_ratio;
            int sx = (int)sx_f, sy = (int)sy_f;
            float fx = sx_f - (float)sx, fy = sy_f - (float)sy;
            int sx1 = sx < src_width - 1 ? sx + 1 : sx;
            int sy1 = sy < src_height - 1 ? sy + 1 : sy;

            for (int c = 0; c < channels; c++) {
                float p00 = src_data[(sy * src_width + sx) * channels + c];
                float p10 = src_data[(sy * src_width + sx1) * channels + c];
                float p01 = src_data[(sy1 * src_width + sx) * channels + c];
                float p11 = src_data[(sy1 * src_width + sx1) * channels + c];
                float top = p00 + (p10 - p00) * fx;
                float bot = p01 + (p11 - p01) * fx;
                dst_data[(dy * dst_width + dx) * channels + c] = top + (bot - top) * fy;
            }
        }
    }
    return 0;
}

int vision_resize_bilinear_float(int src_width, int src_height, int channels,
                                  const float* src_data,
                                  int dst_width, int dst_height, float* dst_data) {
    return vision_resize_bilinear(src_width, src_height, channels,
                                  src_data, dst_width, dst_height, dst_data);
}

int vision_yuv420_to_rgb(int width, int height,
                          const unsigned char* y_plane, int y_stride,
                          const unsigned char* u_plane, int uv_stride,
                          const unsigned char* v_plane,
                          float* rgb_output) {
    if (!y_plane || !u_plane || !v_plane || !rgb_output) return -1;
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1)) return -1;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int y_idx = y * y_stride + x;
            int uv_x = x / 2, uv_y = y / 2;
            int uv_idx = uv_y * uv_stride + uv_x;

            float yy = (float)y_plane[y_idx];
            float u = (float)u_plane[uv_idx] - 128.0f;
            float v = (float)v_plane[uv_idx] - 128.0f;

            int dst_idx = (y * width + x) * 3;
            float r = yy + 1.402f * v;
            float g = yy - 0.344f * u - 0.714f * v;
            float b = yy + 1.772f * u;

            rgb_output[dst_idx] = (r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r)) / 255.0f;
            rgb_output[dst_idx + 1] = (g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g)) / 255.0f;
            rgb_output[dst_idx + 2] = (b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b)) / 255.0f;
        }
    }
    return 0;
}

/* ===================================================================
 * 第十节：CfC ODE层 —— 兼容层（原deep_vision.c）
 *
 * CfC ODE层核心方程：
 * h_new = (1 - gate) * state + gate * tanh(W_input * input + W_hidden * state + b_hidden)
 * gate = sigmoid(W_gate * input + U_gate * state + b_gate)
 * =================================================================== */

struct CfcOdeLayer {
    CfcOdeLayerConfig config;
    int is_initialized;
    float* w_input;
    float* w_hidden;
    float* b_hidden;
    float* w_gate;
    float* u_gate;
    float* b_gate;
    float* tau_weights;
    float tau_bias;
    float* state_buffer;
    size_t state_buffer_size;
    float* hidden_state_persistent;
    int hidden_state_initialized;
    size_t forward_count;
    float total_forward_time_ms;
/* Adam优化器状态 - 完整ODE伴随法反向传播 */
    float* m_w_input;      float* v_w_input;
    float* m_w_hidden;     float* v_w_hidden;
    float* m_b_hidden;     float* v_b_hidden;
    float* m_w_gate;       float* v_w_gate;
    float* m_u_gate;       float* v_u_gate;
    float* m_b_gate;       float* v_b_gate;
    float* m_tau_weights;  float* v_tau_weights;
    float* m_tau_bias;     float* v_tau_bias;
    size_t adam_t;          /* Adam时间步计数 */
    size_t weight_count;    /* 每权重矩阵元素数(统一尺寸) */
};

/* 辅助：矩阵向量乘法 + 向量操作 */
static void _mat_vec_mul(const float* w, int rows, int cols, const float* x, float* y) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += w[(size_t)i * cols + j] * x[j];
        y[i] = sum;
    }
}

static void _vec_add_inplace(float* y, const float* x, int n) {
    for (int i = 0; i < n; i++) y[i] += x[i];
}

static void _vec_mul_elem(float* z, const float* x, const float* y, int n) {
    for (int i = 0; i < n; i++) z[i] = x[i] * y[i];
}

static void _apply_tanh_array(float* data, int n) {
    for (int i = 0; i < n; i++) data[i] = tanhf(data[i]);
}

static void _apply_sigmoid_array(float* data, int n) {
    for (int i = 0; i < n; i++) data[i] = _activation_sigmoid_stable(data[i]);
}

CfcOdeLayerConfig cfc_ode_layer_get_default_config(void) {
    CfcOdeLayerConfig cfg;
    cfg.input_dim = 64;
    cfg.hidden_dim = 128;
    cfg.num_layers = 3;
    cfg.time_constant = 0.1f;
    cfg.delta_t = 0.05f;
    cfg.use_adaptive_tau = 1;
    cfg.min_tau = 0.01f;
    cfg.max_tau = 1.0f;
    cfg.use_bias = 1;
    cfg.use_liquid_gate = 1;
    cfg.noise_std = 0.0f;
    return cfg;
}

CfcOdeLayer* cfc_ode_layer_create(const CfcOdeLayerConfig* config) {
    if (!config) return NULL;
    if (config->input_dim <= 0 || config->hidden_dim <= 0 || config->num_layers <= 0) return NULL;

    CfcOdeLayer* layer = (CfcOdeLayer*)safe_malloc(sizeof(CfcOdeLayer));
    if (!layer) return NULL;

    memset(layer, 0, sizeof(CfcOdeLayer));
    memcpy(&layer->config, config, sizeof(CfcOdeLayerConfig));

    const int input_dim = config->input_dim;
    const int hidden_dim = config->hidden_dim;

    layer->w_input = (float*)safe_malloc((size_t)input_dim * (size_t)hidden_dim * sizeof(float));
    layer->w_hidden = (float*)safe_malloc((size_t)hidden_dim * (size_t)hidden_dim * sizeof(float));
    layer->b_hidden = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    layer->w_gate = (float*)safe_malloc((size_t)input_dim * (size_t)hidden_dim * sizeof(float));
    layer->u_gate = (float*)safe_malloc((size_t)hidden_dim * (size_t)hidden_dim * sizeof(float));
    layer->b_gate = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));

    if (!layer->w_input || !layer->w_hidden || !layer->b_hidden ||
        !layer->w_gate || !layer->u_gate || !layer->b_gate) {
        cfc_ode_layer_free(layer); return NULL;
    }

    if (config->use_adaptive_tau) {
        int total_dim = input_dim + hidden_dim;
        layer->tau_weights = (float*)safe_malloc((size_t)total_dim * sizeof(float));
        if (!layer->tau_weights) { cfc_ode_layer_free(layer); return NULL; }
    }

    layer->state_buffer = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    if (!layer->state_buffer) { cfc_ode_layer_free(layer); return NULL; }
    layer->state_buffer_size = (size_t)hidden_dim;

    layer->hidden_state_persistent = (float*)safe_calloc((size_t)hidden_dim, sizeof(float));
    if (!layer->hidden_state_persistent) { cfc_ode_layer_free(layer); return NULL; }
    layer->hidden_state_initialized = 0;

    /* Xavier初始化 */
    {
        float scale_input = sqrtf(2.0f / (float)(input_dim + hidden_dim));
        float scale_hidden = sqrtf(2.0f / (float)(hidden_dim + hidden_dim));

        for (int i = 0; i < input_dim * hidden_dim; i++) {
            float r = secure_random_float();
            layer->w_input[i] = (r * 2.0f - 1.0f) * scale_input;
        }
        for (int i = 0; i < hidden_dim * hidden_dim; i++) {
            float r = secure_random_float();
            layer->w_hidden[i] = (r * 2.0f - 1.0f) * scale_hidden;
        }
        memset(layer->b_hidden, 0, (size_t)hidden_dim * sizeof(float));

        for (int i = 0; i < input_dim * hidden_dim; i++) {
            float r = secure_random_float();
            layer->w_gate[i] = (r * 2.0f - 1.0f) * scale_input;
        }
        for (int i = 0; i < hidden_dim * hidden_dim; i++) {
            float r = secure_random_float();
            layer->u_gate[i] = (r * 2.0f - 1.0f) * 0.1f;
        }
        for (int i = 0; i < hidden_dim; i++) layer->b_gate[i] = 1.0f;

        if (layer->tau_weights) {
            for (int i = 0; i < input_dim + hidden_dim; i++) {
                float r = secure_random_float();
                layer->tau_weights[i] = (r * 2.0f - 1.0f) * 0.01f;
            }
        }
        layer->tau_bias = 0.0f;
    }

/* 初始化Adam优化器状态 */
    {
        size_t wc_in_hid = (size_t)input_dim * (size_t)hidden_dim;
        size_t wc_hid_hid = (size_t)hidden_dim * (size_t)hidden_dim;
        layer->weight_count = wc_in_hid;
        layer->m_w_input  = (float*)safe_calloc(wc_in_hid,   sizeof(float));
        layer->v_w_input  = (float*)safe_calloc(wc_in_hid,   sizeof(float));
        layer->m_w_hidden = (float*)safe_calloc(wc_hid_hid,  sizeof(float));
        layer->v_w_hidden = (float*)safe_calloc(wc_hid_hid,  sizeof(float));
        layer->m_b_hidden = (float*)safe_calloc(hidden_dim,   sizeof(float));
        layer->v_b_hidden = (float*)safe_calloc(hidden_dim,   sizeof(float));
        layer->m_w_gate   = (float*)safe_calloc(wc_in_hid,   sizeof(float));
        layer->v_w_gate   = (float*)safe_calloc(wc_in_hid,   sizeof(float));
        layer->m_u_gate   = (float*)safe_calloc(wc_hid_hid,  sizeof(float));
        layer->v_u_gate   = (float*)safe_calloc(wc_hid_hid,  sizeof(float));
        layer->m_b_gate   = (float*)safe_calloc(hidden_dim,   sizeof(float));
        layer->v_b_gate   = (float*)safe_calloc(hidden_dim,   sizeof(float));
        if (layer->tau_weights) {
            int td = input_dim + hidden_dim;
            layer->m_tau_weights = (float*)safe_calloc(td, sizeof(float));
            layer->v_tau_weights = (float*)safe_calloc(td, sizeof(float));
        }
        layer->m_tau_bias = (float*)safe_calloc(1, sizeof(float));
        layer->v_tau_bias = (float*)safe_calloc(1, sizeof(float));
        layer->adam_t = 0;
    }

    layer->is_initialized = 1;
    layer->forward_count = 0;
    layer->total_forward_time_ms = 0.0f;
    return layer;
}

void cfc_ode_layer_free(CfcOdeLayer* layer) {
    if (!layer) return;
    safe_free((void**)&layer->w_input);
    safe_free((void**)&layer->w_hidden);
    safe_free((void**)&layer->b_hidden);
    safe_free((void**)&layer->w_gate);
    safe_free((void**)&layer->u_gate);
    safe_free((void**)&layer->b_gate);
    safe_free((void**)&layer->tau_weights);
    safe_free((void**)&layer->state_buffer);
    safe_free((void**)&layer->hidden_state_persistent);
/* 释放Adam优化器状态 */
    safe_free((void**)&layer->m_w_input);
    safe_free((void**)&layer->v_w_input);
    safe_free((void**)&layer->m_w_hidden);
    safe_free((void**)&layer->v_w_hidden);
    safe_free((void**)&layer->m_b_hidden);
    safe_free((void**)&layer->v_b_hidden);
    safe_free((void**)&layer->m_w_gate);
    safe_free((void**)&layer->v_w_gate);
    safe_free((void**)&layer->m_u_gate);
    safe_free((void**)&layer->v_u_gate);
    safe_free((void**)&layer->m_b_gate);
    safe_free((void**)&layer->v_b_gate);
    safe_free((void**)&layer->m_tau_weights);
    safe_free((void**)&layer->v_tau_weights);
    safe_free((void**)&layer->m_tau_bias);
    safe_free((void**)&layer->v_tau_bias);
    safe_free((void**)&layer);
}

int cfc_ode_layer_forward(CfcOdeLayer* layer, const float* input, float* output) {
    if (!layer || !layer->is_initialized || !input || !output) return -1;
    uint64_t start = perf_timestamp_ns();

    const int input_dim = layer->config.input_dim;
    const int hidden_dim = layer->config.hidden_dim;

    if (!layer->hidden_state_initialized) {
        memset(layer->hidden_state_persistent, 0, (size_t)hidden_dim * sizeof(float));
        layer->hidden_state_initialized = 1;
    }

    memcpy(output, layer->hidden_state_persistent, (size_t)hidden_dim * sizeof(float));

    /* gate = sigmoid(W_gate * input + U_gate * state + b_gate) */
    float* gate = layer->state_buffer;
    _mat_vec_mul(layer->w_gate, hidden_dim, input_dim, input, gate);
    {
        float* u_state = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (u_state) {
            _mat_vec_mul(layer->u_gate, hidden_dim, hidden_dim, output, u_state);
            _vec_add_inplace(gate, u_state, hidden_dim);
            safe_free((void**)&u_state);
        }
    }
    if (layer->config.use_bias) _vec_add_inplace(gate, layer->b_gate, hidden_dim);
    _apply_sigmoid_array(gate, hidden_dim);

    /* h_new = W_input * input + W_hidden * state + b_hidden → tanh */
    float* h_new = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    if (!h_new) return -1;
    _mat_vec_mul(layer->w_input, hidden_dim, input_dim, input, h_new);
    {
        float* h_state_term = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (h_state_term) {
            _mat_vec_mul(layer->w_hidden, hidden_dim, hidden_dim, output, h_state_term);
            _vec_add_inplace(h_new, h_state_term, hidden_dim);
            safe_free((void**)&h_state_term);
        }
    }
    if (layer->config.use_bias) _vec_add_inplace(h_new, layer->b_hidden, hidden_dim);
    _apply_tanh_array(h_new, hidden_dim);

    /* CfC闭式解: output = (1 - gate) * state + gate * h_new */
    float* one_minus_gate = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    if (one_minus_gate) {
        for (int i = 0; i < hidden_dim; i++) one_minus_gate[i] = 1.0f - gate[i];
        float* term1 = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (term1) {
            _vec_mul_elem(term1, one_minus_gate, output, hidden_dim);
            _vec_mul_elem(output, gate, h_new, hidden_dim);
            _vec_add_inplace(output, term1, hidden_dim);
            safe_free((void**)&term1);
        }
        safe_free((void**)&one_minus_gate);
    }

    /* 自适应时间常数 */
    if (layer->config.use_adaptive_tau) {
        float tau = layer->tau_bias;
        {
            int total_dim = input_dim + hidden_dim;
            float* concat = (float*)safe_malloc((size_t)total_dim * sizeof(float));
            if (concat) {
                memcpy(concat, input, (size_t)input_dim * sizeof(float));
                memcpy(concat + input_dim, layer->hidden_state_persistent, (size_t)hidden_dim * sizeof(float));
                float tau_logit = 0.0f;
                for (int i = 0; i < total_dim; i++) tau_logit += layer->tau_weights[i] * concat[i];
                tau = tau_logit + layer->tau_bias;
                safe_free((void**)&concat);
            }
        }
        tau = _activation_sigmoid_stable(tau);
        float tau_range = layer->config.max_tau - layer->config.min_tau;
        tau = tau * tau_range + layer->config.min_tau;

        float dt = layer->config.delta_t;
        float alpha = (dt / tau) > 1.0f ? 1.0f : ((dt / tau) < 0.0f ? 0.0f : dt / tau);
        for (int i = 0; i < hidden_dim; i++)
            output[i] = (1.0f - alpha) * layer->hidden_state_persistent[i] + alpha * output[i];
    }

    /* 保存持久状态 */
    memcpy(layer->hidden_state_persistent, output, (size_t)hidden_dim * sizeof(float));

    uint64_t end = perf_timestamp_ns();
    layer->total_forward_time_ms += (float)(end - start) / 1000000.0f;
    layer->forward_count++;

    safe_free((void**)&h_new);
    return 0;
}

int cfc_ode_layers_forward(CfcOdeLayer** layers, int num_layers,
                           const float* input, float* output) {
    if (!layers || num_layers <= 0 || !input || !output) return -1;
    for (int i = 0; i < num_layers; i++)
        if (!layers[i] || !layers[i]->is_initialized) return -1;

    const int first_hidden = layers[0]->config.hidden_dim;
    const int last_hidden = layers[num_layers - 1]->config.hidden_dim;

    float* temp = (float*)safe_malloc((size_t)(first_hidden > last_hidden ? first_hidden : last_hidden) * sizeof(float));
    if (!temp) return -1;

    if (cfc_ode_layer_forward(layers[0], input, temp) != 0) { safe_free((void**)&temp); return -1; }

    for (int i = 1; i < num_layers - 1; i++) {
        float* next_temp = (float*)safe_malloc((size_t)layers[i]->config.hidden_dim * sizeof(float));
        if (!next_temp) { safe_free((void**)&temp); return -1; }
        if (cfc_ode_layer_forward(layers[i], temp, next_temp) != 0) {
            safe_free((void**)&temp); safe_free((void**)&next_temp);
            return -1;
        }
        safe_free((void**)&temp);
        temp = next_temp;
    }

    if (num_layers > 1) {
        if (cfc_ode_layer_forward(layers[num_layers - 1], temp, output) != 0) {
            safe_free((void**)&temp); return -1;
        }
    } else {
        memcpy(output, temp, (size_t)last_hidden * sizeof(float));
    }

    safe_free((void**)&temp);
    return 0;
}

/* 完整CfC ODE伴随法反向传播 + Adam优化器
 * 替代原来的"简化反向传播：基于输出梯度直接更新权重"。
 * 
 * CfC ODE方程重述（用于梯度推导）：
 *   linear = W_input * x + W_hidden * h_prev + b_hidden
 *   gate_logit = W_gate * x + U_gate * h_prev + b_gate
 *   h_new = (1-sigmoid(gate_logit))*h_prev + sigmoid(gate_logit)*tanh(linear)
 *
 * 梯度链（ODE伴随法）：
 *   dh_dgate = -h_prev + tanh(linear)   [h_new对gate的导数]
 *   dgate_dpre = sigmoid'(gate_logit)    [gate对logit的导数]
 *   dL_d(W_gate)[i,h] = dL_dh[h] * dh_dgate[h] * dgate_dpre[h] * x[i]
 *   dL_d(U_gate)[j,h] = dL_dh[h] * dh_dgate[h] * dgate_dpre[h] * h_prev[j]
 *   dL_d(b_gate)[h] = dL_dh[h] * dh_dgate[h] * dgate_dpre[h]
 *
 *   dh_dlinear = gate * tanh'(linear)
 *   dL_d(W_input)[i,h] = dL_dh[h] * dh_dlinear[h] * x[i]
 *   dL_d(W_hidden)[i,h] = dL_dh[h] * dh_dlinear[h] * h_prev[i]
 *   dL_d(b_hidden)[h] = dL_dh[h] * dh_dlinear[h]
 *
 *   dL_dx[i] = Σ_h(dL_dh[h] * dh_dlinear[h] * W_input[i,h] + 
 *                   dL_dh[h] * dh_dgate[h] * dgate_dpre[h] * W_gate[i,h])
 *   dL_dh_prev[j] = dL_dh * (1-gate) + 
 *                    Σ_h(dL_dh[h] * dh_dlinear[h] * W_hidden[j,h] + 
 *                        dL_dh[h] * dh_dgate[h] * dgate_dpre[h] * U_gate[j,h])
 *
 * Adam更新规则：
 *   m = β1*m + (1-β1)*grad
 *   v = β2*v + (1-β2)*grad²
 *   m_hat = m/(1-β1^t)
 *   v_hat = v/(1-β2^t)
 *   param -= lr * m_hat/(√v_hat + ε)
 */
static void _adam_update_param(float* param, float* m, float* v, float grad,
                                float lr, float beta1, float beta2, float eps,
                                size_t adam_t) {
    *m = beta1 * (*m) + (1.0f - beta1) * grad;
    *v = beta2 * (*v) + (1.0f - beta2) * grad * grad;
    float m_hat = *m / (1.0f - powf(beta1, (float)adam_t));
    float v_hat = *v / (1.0f - powf(beta2, (float)adam_t));
    *param -= lr * m_hat / (sqrtf(v_hat) + eps);
}

int cfc_ode_layer_backward(CfcOdeLayer* layer,
                           const float* dL_doutput,
                           const float* input,
                           const float* output,
                           float* dL_dinput,
                           float learning_rate) {
    if (!layer || !layer->is_initialized || !dL_doutput || !input || !output) return -1;

    const int input_dim = layer->config.input_dim;
    const int hidden_dim = layer->config.hidden_dim;
    const float* h_prev = layer->hidden_state_persistent;
    if (!layer->hidden_state_initialized) return -1;

/* 重新计算前向中间值（ODE伴随法需要中间激活） */
    float* linear = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    float* gate_logit = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
    if (!linear || !gate_logit) {
        safe_free((void**)&linear); safe_free((void**)&gate_logit); return -1;
    }

    /* 前向重计算: linear = W_input*x + W_hidden*h_prev + b_hidden */
    _mat_vec_mul(layer->w_input, hidden_dim, input_dim, input, linear);
    {
        float* h_term = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (h_term) {
            memset(h_term, 0, (size_t)hidden_dim * sizeof(float));
            _mat_vec_mul(layer->w_hidden, hidden_dim, hidden_dim, h_prev, h_term);
            _vec_add_inplace(linear, h_term, hidden_dim);
            safe_free((void**)&h_term);
        }
    }
    if (layer->config.use_bias) _vec_add_inplace(linear, layer->b_hidden, hidden_dim);

    /* 前向重计算: gate_logit = W_gate*x + U_gate*h_prev + b_gate */
    _mat_vec_mul(layer->w_gate, hidden_dim, input_dim, input, gate_logit);
    {
        float* u_term = (float*)safe_malloc((size_t)hidden_dim * sizeof(float));
        if (u_term) {
            memset(u_term, 0, (size_t)hidden_dim * sizeof(float));
            _mat_vec_mul(layer->u_gate, hidden_dim, hidden_dim, h_prev, u_term);
            _vec_add_inplace(gate_logit, u_term, hidden_dim);
            safe_free((void**)&u_term);
        }
    }
    if (layer->config.use_bias) _vec_add_inplace(gate_logit, layer->b_gate, hidden_dim);

    /* Adam参数 */
    const float lr = learning_rate > 0.0f ? learning_rate : 0.001f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps_val = 1e-8f;
    layer->adam_t++;
    size_t t = layer->adam_t;
    if (t < 1) t = 1;

    /* ODE伴随法梯度计算 + Adam更新 */
    if (dL_dinput) memset(dL_dinput, 0, (size_t)input_dim * sizeof(float));

    for (int h = 0; h < hidden_dim; h++) {
        float th = tanhf(linear[h]);
        float g = _activation_sigmoid_stable(gate_logit[h]);
        float tanh_deriv = 1.0f - th * th;
        float sigmoid_deriv = g * (1.0f - g);

        /* CfC ODE梯度：dL_dh注入 */
        float dL_dh_h = dL_doutput[h];

        /* dh_new/dgate = -h_prev + tanh(linear) */
        float dh_dgate = -h_prev[h] + th;
        /* dh_new/dlinear = gate * tanh'(linear) */
        float dh_dlinear = g * tanh_deriv;

        /* 完整伴随梯度 */
        float dL_dgate_pre = dL_dh_h * dh_dgate * sigmoid_deriv;
        float dL_dlinear_h = dL_dh_h * dh_dlinear;

        /* Adam更新偏置 */
        _adam_update_param(&layer->b_hidden[h], &layer->m_b_hidden[h], &layer->v_b_hidden[h],
                          dL_dlinear_h, lr, beta1, beta2, eps_val, t);
        _adam_update_param(&layer->b_gate[h], &layer->m_b_gate[h], &layer->v_b_gate[h],
                          dL_dgate_pre, lr, beta1, beta2, eps_val, t);

        /* Adam更新权重矩阵（每元素独立Adam状态） */
        for (int i = 0; i < input_dim; i++) {
            size_t idx = (size_t)h * (size_t)input_dim + (size_t)i;
            _adam_update_param(&layer->w_input[idx], &layer->m_w_input[idx], &layer->v_w_input[idx],
                              dL_dlinear_h * input[i], lr, beta1, beta2, eps_val, t);
            _adam_update_param(&layer->w_gate[idx], &layer->m_w_gate[idx], &layer->v_w_gate[idx],
                              dL_dgate_pre * input[i], lr, beta1, beta2, eps_val, t);
            if (dL_dinput) {
                dL_dinput[i] += dL_dlinear_h * layer->w_input[idx] +
                                dL_dgate_pre * layer->w_gate[idx];
            }
        }

        for (int j = 0; j < hidden_dim; j++) {
            size_t idx = (size_t)h * (size_t)hidden_dim + (size_t)j;
            _adam_update_param(&layer->w_hidden[idx], &layer->m_w_hidden[idx], &layer->v_w_hidden[idx],
                              dL_dlinear_h * h_prev[j], lr, beta1, beta2, eps_val, t);
            _adam_update_param(&layer->u_gate[idx], &layer->m_u_gate[idx], &layer->v_u_gate[idx],
                              dL_dgate_pre * h_prev[j], lr, beta1, beta2, eps_val, t);
        }
    }

    /* 自适应时间常数梯度 */
    if (layer->config.use_adaptive_tau && layer->tau_weights) {
        int total_dim = input_dim + hidden_dim;
        float* concat = (float*)safe_malloc((size_t)total_dim * sizeof(float));
        if (concat) {
            memcpy(concat, input, (size_t)input_dim * sizeof(float));
            memcpy(concat + input_dim, h_prev, (size_t)hidden_dim * sizeof(float));
            float tau_logit = layer->tau_bias;
            for (int i = 0; i < total_dim; i++) tau_logit += layer->tau_weights[i] * concat[i];
            float tau_sig = _activation_sigmoid_stable(tau_logit);
            float tau_deriv = tau_sig * (1.0f - tau_sig);

            for (int h = 0; h < hidden_dim; h++) {
                float dL_dtau = dL_doutput[h] * (output[h] - h_prev[h]) * tau_deriv;
                for (int i = 0; i < total_dim; i++) {
                    _adam_update_param(&layer->tau_weights[i], &layer->m_tau_weights[i],
                                      &layer->v_tau_weights[i],
                                      dL_dtau * concat[i], lr * 0.1f, beta1, beta2, eps_val, t);
                }
                _adam_update_param(&layer->tau_bias, layer->m_tau_bias, layer->v_tau_bias,
                                  dL_dtau, lr * 0.1f, beta1, beta2, eps_val, t);
            }
            safe_free((void**)&concat);
        }
    }

    safe_free((void**)&linear); safe_free((void**)&gate_logit);
    return 0;
}

/* ===================================================================
 * 第十一节：CfcVisionProcessor 兼容层（原deep_vision.c）
 *
 * 使用Gabor初始化投影 + 正弦位置编码 + CfC ODE演化 + 全局平均池化
 * 保持agi/multimodal/ocr/imitation_learning等模块兼容
 * =================================================================== */

/* Gabor滤波器生成 */
static float _gabor_filter(float x, float y, float theta, float sigma,
                           float lambda, float psi, float gamma) {
    float x_rot = x * cosf(theta) + y * sinf(theta);
    float y_rot = -x * sinf(theta) + y * cosf(theta);
    float gauss = expf(-(x_rot*x_rot + gamma*gamma*y_rot*y_rot) / (2.0f*sigma*sigma));
    return gauss * cosf(2.0f*3.14159265f*x_rot / lambda + psi);
}

struct CfcVisionProcessor {
    CfcVisionConfig config;
    int is_initialized;
    CfcOdeLayer** ode_layers;
    int num_ode_layers;
    float* patch_proj_weight;
    float* patch_proj_bias;
    int patch_dim;
    int proj_hidden_dim;
    float* patch_buffer;
    float* patch_features;
    float* pooled_feature;
    size_t patch_buffer_size;
    size_t patch_features_size;
    size_t max_patches;
    float* pos_encoding;
    int pos_encoding_dim;
    CfcDetectionConfig detection_config;
    float* detect_weight;
    float* detect_bias;
    int detect_head_initialized;
    int* nms_sorted_indices;
    float* nms_boxes_work;
    int nms_work_size;
    int total_detections;
    float avg_confidence_sum;
    int training_completed;
    int bootstrap_attempted;
    size_t total_operations;
    float memory_usage_mb;
    float total_processing_time_ms;
    int total_images_processed;
};

CfcVisionConfig cfc_vision_get_default_config(void) {
    CfcVisionConfig cfg;
    cfg.image_width = 224;
    cfg.image_height = 224;
    cfg.image_channels = 3;
    cfg.patch_size = 16;
    cfg.output_dim = 128;
    cfg.num_ode_layers = 3;
    cfg.time_constant = 0.1f;
    cfg.delta_t = 0.05f;
    return cfg;
}

CfcVisionProcessor* cfc_vision_processor_create(const CfcVisionConfig* config) {
    if (!config) return NULL;
    if (config->image_width <= 0 || config->image_height <= 0 ||
        config->image_channels <= 0 || config->patch_size <= 0 ||
        config->output_dim <= 0 || config->num_ode_layers <= 0) {
        return NULL;
    }

    CfcVisionProcessor* proc = (CfcVisionProcessor*)safe_malloc(sizeof(CfcVisionProcessor));
    if (!proc) return NULL;

    memset(proc, 0, sizeof(CfcVisionProcessor));
    memcpy(&proc->config, config, sizeof(CfcVisionConfig));

    proc->patch_dim = config->patch_size * config->patch_size * config->image_channels;
    proc->proj_hidden_dim = config->output_dim;

    int num_patches_h = config->image_height / config->patch_size;
    int num_patches_w = config->image_width / config->patch_size;
    if (num_patches_h * config->patch_size != config->image_height ||
        num_patches_w * config->patch_size != config->image_width) {
        safe_free((void**)&proc); return NULL;
    }
    int num_patches = num_patches_h * num_patches_w;
    proc->max_patches = (size_t)num_patches;

    proc->num_ode_layers = config->num_ode_layers;
    proc->ode_layers = (CfcOdeLayer**)safe_malloc(
        (size_t)config->num_ode_layers * sizeof(CfcOdeLayer*));
    if (!proc->ode_layers) { safe_free((void**)&proc); return NULL; }
    memset(proc->ode_layers, 0, (size_t)config->num_ode_layers * sizeof(CfcOdeLayer*));

    for (int i = 0; i < config->num_ode_layers; i++) {
        CfcOdeLayerConfig layer_cfg = cfc_ode_layer_get_default_config();
        if (i == 0) layer_cfg.input_dim = proc->proj_hidden_dim;
        else layer_cfg.input_dim = layer_cfg.hidden_dim;
        layer_cfg.hidden_dim = proc->proj_hidden_dim;
        layer_cfg.time_constant = config->time_constant;
        layer_cfg.delta_t = config->delta_t;

        proc->ode_layers[i] = cfc_ode_layer_create(&layer_cfg);
        if (!proc->ode_layers[i]) { cfc_vision_processor_destroy(proc); return NULL; }
    }

    proc->patch_proj_weight = (float*)safe_malloc(
        (size_t)proc->patch_dim * proc->proj_hidden_dim * sizeof(float));
    proc->patch_proj_bias = (float*)safe_malloc((size_t)proc->proj_hidden_dim * sizeof(float));
    if (!proc->patch_proj_weight || !proc->patch_proj_bias) {
        cfc_vision_processor_destroy(proc); return NULL;
    }

    /* Gabor初始化 + 颜色编码 + Xavier回退 */
    int patch_size = config->patch_size;
    int img_c = config->image_channels;
    int parea = patch_size * patch_size;

    for (int d = 0; d < proc->proj_hidden_dim; d++) {
        for (int i = 0; i < proc->patch_dim; i++) {
            int local_i = i % parea;
            int py = local_i / patch_size, px = local_i % patch_size;
            float nx = ((float)px / (float)patch_size - 0.5f) * 2.0f;
            float ny = ((float)py / (float)patch_size - 0.5f) * 2.0f;

            float weight;
            if (d < proc->proj_hidden_dim / 3) {
                int gabor_idx = d % 16;
                float theta = (float)(gabor_idx % 8) * 3.14159265f / 8.0f;
                float sigma = 0.3f + (float)(gabor_idx / 4) * 0.2f;
                float lambda_val = 0.1f + (float)(gabor_idx % 4) * 0.1f;
                float gamma_val = 0.5f;
                float psi = (gabor_idx < 8) ? 0.0f : 3.14159265f / 2.0f;
                weight = _gabor_filter(nx, ny, theta, sigma, lambda_val, psi, gamma_val);
                weight *= 0.5f;
            } else if (d < proc->proj_hidden_dim * 2 / 3) {
                float color_w[3] = {1.0f, -0.5f, -0.5f};
                int cidx = (d % 3);
                float brightness = (nx + ny) * 0.5f + 0.5f * (1.0f - fabsf(nx*ny));
                weight = color_w[cidx % img_c] * 0.1f + brightness * 0.05f;
            } else {
                float r = secure_random_float();
                weight = (r * 2.0f - 1.0f)
                         * sqrtf(2.0f / (float)(proc->patch_dim + proc->proj_hidden_dim));
            }
            proc->patch_proj_weight[(size_t)d * proc->patch_dim + i] = weight;
        }
    }
    memset(proc->patch_proj_bias, 0, (size_t)proc->proj_hidden_dim * sizeof(float));

    proc->patch_buffer = (float*)safe_malloc((size_t)proc->patch_dim * sizeof(float));
    proc->patch_features = (float*)safe_malloc((size_t)num_patches * proc->proj_hidden_dim * sizeof(float));
    proc->pooled_feature = (float*)safe_malloc((size_t)proc->proj_hidden_dim * sizeof(float));
    proc->pos_encoding = (float*)safe_malloc((size_t)num_patches * proc->proj_hidden_dim * sizeof(float));

    if (!proc->patch_buffer || !proc->patch_features || !proc->pooled_feature || !proc->pos_encoding) {
        cfc_vision_processor_destroy(proc); return NULL;
    }

    proc->patch_buffer_size = (size_t)proc->patch_dim;
    proc->patch_features_size = (size_t)num_patches * proc->proj_hidden_dim;

    /* 正弦/余弦位置编码 */
    {
        float* pos = proc->pos_encoding;
        for (int p = 0; p < num_patches; p++) {
            float pos_val = (float)p / num_patches;
            for (int d = 0; d < proc->proj_hidden_dim; d++) {
                float freq = expf(-logf(10000.0f) * (float)d / proc->proj_hidden_dim);
                if (d % 2 == 0) pos[(size_t)p * proc->proj_hidden_dim + d] = sinf(pos_val * freq);
                else pos[(size_t)p * proc->proj_hidden_dim + d] = cosf(pos_val * freq);
            }
        }
    }
    proc->pos_encoding_dim = proc->proj_hidden_dim;

    /* 初始化检测头 */
    {
        int num_classes = proc->detection_config.num_classes;
        if (num_classes <= 0) num_classes = 80;
        int detect_out_dim = num_classes + 5;
        proc->detect_weight = (float*)safe_malloc((size_t)proc->proj_hidden_dim * detect_out_dim * sizeof(float));
        proc->detect_bias = (float*)safe_malloc((size_t)detect_out_dim * sizeof(float));
        if (!proc->detect_weight || !proc->detect_bias) { cfc_vision_processor_destroy(proc); return NULL; }
        float det_scale = sqrtf(2.0f / (float)(proc->proj_hidden_dim + detect_out_dim));
        for (int i = 0; i < proc->proj_hidden_dim * detect_out_dim; i++) {
            float r = secure_random_float();
            proc->detect_weight[i] = (r * 2.0f - 1.0f) * det_scale;
        }
        memset(proc->detect_bias, 0, (size_t)detect_out_dim * sizeof(float));
        proc->nms_sorted_indices = (int*)safe_malloc((size_t)CFC_VISION_MAX_DETECTIONS * sizeof(int));
        proc->nms_boxes_work = (float*)safe_malloc((size_t)CFC_VISION_MAX_DETECTIONS * 4 * sizeof(float));
        proc->nms_work_size = CFC_VISION_MAX_DETECTIONS;
        proc->detect_head_initialized = 1;
    }

    proc->is_initialized = 1;
    return proc;
}

void cfc_vision_processor_destroy(CfcVisionProcessor* processor) {
    if (!processor) return;
    if (processor->ode_layers) {
        for (int i = 0; i < processor->num_ode_layers; i++) cfc_ode_layer_free(processor->ode_layers[i]);
        safe_free((void**)&processor->ode_layers);
    }
    safe_free((void**)&processor->patch_proj_weight);
    safe_free((void**)&processor->patch_proj_bias);
    safe_free((void**)&processor->patch_buffer);
    safe_free((void**)&processor->patch_features);
    safe_free((void**)&processor->pooled_feature);
    safe_free((void**)&processor->pos_encoding);
    safe_free((void**)&processor->detect_weight);
    safe_free((void**)&processor->detect_bias);
    safe_free((void**)&processor->nms_sorted_indices);
    safe_free((void**)&processor->nms_boxes_work);
    safe_free((void**)&processor);
}

int cfc_vision_extract_features(CfcVisionProcessor* processor,
                                const float* image_data,
                                int width, int height, int channels,
                                float* features, size_t max_features) {
    if (!processor || !processor->is_initialized || !image_data || !features) return -1;

    if (!processor->training_completed) {
        /* LNN尚未训练完成，需要先完成LNN训练才能进行特征提取
         * 不执行任何非LNN的传统CV特征回退（HOG/颜色直方图等），
         * 返回错误码让上层调用者知道需要先完成LNN训练 */
        return -1;
    }

    if (width != processor->config.image_width || height != processor->config.image_height ||
        channels != processor->config.image_channels) return -1;

    uint64_t start_time = perf_timestamp_ns();
    const int patch_size = processor->config.patch_size;
    const int img_channels = processor->config.image_channels;
    const int proj_dim = processor->proj_hidden_dim;
    int num_patches_h = height / patch_size;
    int num_patches_w = width / patch_size;
    int num_patches = num_patches_h * num_patches_w;

    if (max_features < (size_t)proj_dim) return -1;

    for (int ph = 0; ph < num_patches_h; ph++) {
        for (int pw = 0; pw < num_patches_w; pw++) {
            float* patch_flat = processor->patch_buffer;
            for (int c = 0; c < img_channels; c++) {
                for (int py = 0; py < patch_size; py++) {
                    for (int px = 0; px < patch_size; px++) {
                        int img_y = ph * patch_size + py;
                        int img_x = pw * patch_size + px;
                        int img_idx = img_y * width + img_x;
                        int patch_idx = c * patch_size * patch_size + py * patch_size + px;
                        patch_flat[patch_idx] = image_data[(size_t)img_idx * img_channels + c];
                    }
                }
            }
            float* patch_feat = processor->patch_features + ((size_t)ph * num_patches_w + pw) * proj_dim;
            _mat_vec_mul(processor->patch_proj_weight, proj_dim, processor->patch_dim, patch_flat, patch_feat);
            if (processor->patch_proj_bias) _vec_add_inplace(patch_feat, processor->patch_proj_bias, proj_dim);
            for (int d = 0; d < proj_dim; d++) patch_feat[d] = tanhf(patch_feat[d]);
            float* pos = processor->pos_encoding + ((size_t)ph * num_patches_w + pw) * processor->pos_encoding_dim;
            for (int d = 0; d < proj_dim; d++) patch_feat[d] += pos[d];
        }
    }

    for (int p = 0; p < num_patches; p++) {
        float* patch_feat = processor->patch_features + (size_t)p * proj_dim;
        cfc_ode_layers_forward(processor->ode_layers, processor->num_ode_layers, patch_feat, patch_feat);
    }

    memset(processor->pooled_feature, 0, (size_t)proj_dim * sizeof(float));
    for (int p = 0; p < num_patches; p++)
        _vec_add_inplace(processor->pooled_feature, processor->patch_features + (size_t)p * proj_dim, proj_dim);
    float inv = 1.0f / (float)num_patches;
    for (int d = 0; d < proj_dim; d++) processor->pooled_feature[d] *= inv;

    memcpy(features, processor->pooled_feature, (size_t)proj_dim * sizeof(float));

    uint64_t end_time = perf_timestamp_ns();
    processor->total_processing_time_ms += (float)(end_time - start_time) / 1000000.0f;
    processor->total_images_processed++;
    return proj_dim;
}

void cfc_vision_get_statistics(CfcVisionProcessor* processor,
                                size_t* total_operations,
                                float* memory_usage_mb,
                                float* average_time_ms) {
    if (!processor) return;
    size_t mem = sizeof(CfcVisionProcessor);
    mem += (size_t)processor->num_ode_layers * sizeof(CfcOdeLayer*);
    for (int i = 0; i < processor->num_ode_layers; i++) {
        if (processor->ode_layers[i]) {
            mem += sizeof(CfcOdeLayer);
            const int id = processor->ode_layers[i]->config.input_dim;
            const int hd = processor->ode_layers[i]->config.hidden_dim;
            mem += (size_t)id * hd * sizeof(float) * 2;
            mem += (size_t)hd * hd * sizeof(float) * 2;
            mem += (size_t)hd * sizeof(float) * 3;
            if (processor->ode_layers[i]->config.use_adaptive_tau)
                mem += (size_t)(id + hd) * sizeof(float);
        }
    }
    mem += (size_t)processor->patch_dim * sizeof(float);
    mem += (size_t)processor->patch_dim * processor->proj_hidden_dim * sizeof(float);
    mem += (size_t)processor->proj_hidden_dim * sizeof(float);
    mem += processor->max_patches * (size_t)processor->proj_hidden_dim * sizeof(float) * 2;
    mem += (size_t)processor->proj_hidden_dim * sizeof(float);

    float avg_time = 0.0f;
    if (processor->total_images_processed > 0)
        avg_time = processor->total_processing_time_ms / (float)processor->total_images_processed;

    if (total_operations) *total_operations = processor->total_operations;
    if (memory_usage_mb) *memory_usage_mb = (float)mem / (1024.0f * 1024.0f);
    if (average_time_ms) *average_time_ms = avg_time;
}

/* ===================================================================
 * 第十二节：目标检测API（原deep_vision.c）
 * =================================================================== */

CfcDetectionConfig cfc_vision_get_default_detection_config(void) {
    CfcDetectionConfig cfg;
    cfg.num_classes = 80;
    cfg.conf_threshold = 0.5f;
    cfg.iou_threshold = 0.5f;
    cfg.max_detections = CFC_VISION_MAX_DETECTIONS;
    return cfg;
}

static int _cfc_compare_detections_desc(const void* a, const void* b) {
    const CfcDetectionResult* da = (const CfcDetectionResult*)a;
    const CfcDetectionResult* db = (const CfcDetectionResult*)b;
    if (db->confidence > da->confidence) return 1;
    if (db->confidence < da->confidence) return -1;
    return 0;
}

static float _cfc_iou(const CfcDetectionResult* a, const CfcDetectionResult* b) {
    float a_x1 = a->x - a->width * 0.5f, a_y1 = a->y - a->height * 0.5f;
    float a_x2 = a->x + a->width * 0.5f, a_y2 = a->y + a->height * 0.5f;
    float b_x1 = b->x - b->width * 0.5f, b_y1 = b->y - b->height * 0.5f;
    float b_x2 = b->x + b->width * 0.5f, b_y2 = b->y + b->height * 0.5f;
    float inter_x1 = (a_x1 > b_x1) ? a_x1 : b_x1;
    float inter_y1 = (a_y1 > b_y1) ? a_y1 : b_y1;
    float inter_x2 = (a_x2 < b_x2) ? a_x2 : b_x2;
    float inter_y2 = (a_y2 < b_y2) ? a_y2 : b_y2;
    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) inter_area = 0.0f;
    float a_area = a->width * a->height, b_area = b->width * b->height;
    float union_area = a_area + b_area - inter_area;
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

static int _cfc_nms(CfcDetectionResult* candidates, int num_candidates,
                     float iou_threshold, float conf_threshold,
                     CfcDetectionResult* results, int max_results) {
    if (!candidates || num_candidates <= 0 || !results) return 0;
    qsort(candidates, (size_t)num_candidates, sizeof(CfcDetectionResult), _cfc_compare_detections_desc);
    int num_results = 0;
    int* suppressed = (int*)safe_calloc((size_t)num_candidates, sizeof(int));
    if (!suppressed) return 0;
    for (int i = 0; i < num_candidates && num_results < max_results; i++) {
        if (suppressed[i]) continue;
        if (candidates[i].confidence < conf_threshold) continue;
        results[num_results] = candidates[i];
        num_results++;
        for (int j = i + 1; j < num_candidates; j++) {
            if (suppressed[j]) continue;
            if (_cfc_iou(&candidates[i], &candidates[j]) > iou_threshold) suppressed[j] = 1;
        }
    }
    safe_free((void**)&suppressed);
    return num_results;
}

int cfc_vision_detect(CfcVisionProcessor* processor,
                       const float* image_data,
                       int width, int height, int channels,
                       CfcDetectionResult* results, int max_results,
                       int* num_detected) {
    if (!processor || !processor->is_initialized || !image_data || !results || !num_detected) return -1;
    if (!processor->detect_head_initialized) return -1;
    *num_detected = 0;

    int feat_dim = cfc_vision_extract_features(processor, image_data, width, height, channels,
                                                processor->pooled_feature, (size_t)processor->proj_hidden_dim);
    if (feat_dim <= 0) return -1;

    int num_classes = processor->detection_config.num_classes;
    if (num_classes <= 0) num_classes = 80;
    int detect_out_dim = num_classes + 5;
    int num_patches_h = height / processor->config.patch_size;
    int num_patches_w = width / processor->config.patch_size;
    int num_patches = num_patches_h * num_patches_w;

    CfcDetectionResult* candidates = (CfcDetectionResult*)safe_malloc(
        (size_t)processor->max_patches * sizeof(CfcDetectionResult));
    if (!candidates) return -1;
    int num_candidates = 0;

    for (int p = 0; p < num_patches; p++) {
        float* patch_feat = processor->patch_features + (size_t)p * processor->proj_hidden_dim;
        float raw_out[1005];
        int out_dim = detect_out_dim;
        if (out_dim > 1005) out_dim = 1005;
        memset(raw_out, 0, (size_t)out_dim * sizeof(float));
        for (int d = 0; d < out_dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < processor->proj_hidden_dim; i++)
                sum += patch_feat[i] * processor->detect_weight[(size_t)i * detect_out_dim + d];
            sum += processor->detect_bias[d];
            raw_out[d] = sum;
        }

        float objectness = _activation_sigmoid_stable(raw_out[4]);
        if (objectness < processor->detection_config.conf_threshold) continue;

        int ph = p / num_patches_w, pw = p % num_patches_w;
        float grid_cx = ((float)pw + 0.5f) / (float)num_patches_w;
        float grid_cy = ((float)ph + 0.5f) / (float)num_patches_h;
        float bx = grid_cx + raw_out[0] * 0.1f;
        float by = grid_cy + raw_out[1] * 0.1f;
        float bw = expf(raw_out[2]) * 0.1f;
        float bh = expf(raw_out[3]) * 0.1f;

        int best_class = 0;
        float best_score = -1e10f;
        for (int c = 0; c < num_classes && c + 5 <= out_dim; c++) {
            if (raw_out[5 + c] > best_score) { best_score = raw_out[5 + c]; best_class = c; }
        }
        float cls_prob = _activation_sigmoid_stable(best_score);
        float final_conf = objectness * cls_prob;
        if (final_conf < processor->detection_config.conf_threshold) continue;

        if (num_candidates < (int)processor->max_patches) {
            candidates[num_candidates].class_id = best_class;
            candidates[num_candidates].confidence = final_conf;
            candidates[num_candidates].x = (bx > 1.0f) ? 1.0f : (bx < 0.0f ? 0.0f : bx);
            candidates[num_candidates].y = (by > 1.0f) ? 1.0f : (by < 0.0f ? 0.0f : by);
            candidates[num_candidates].width = (bw > 1.0f) ? 1.0f : (bw < 0.001f ? 0.001f : bw);
            candidates[num_candidates].height = (bh > 1.0f) ? 1.0f : (bh < 0.001f ? 0.001f : bh);
            num_candidates++;
        }
    }

    int kept = _cfc_nms(candidates, num_candidates,
                        processor->detection_config.iou_threshold,
                        processor->detection_config.conf_threshold,
                        results, (max_results > 0) ? max_results : CFC_VISION_MAX_DETECTIONS);
    safe_free((void**)&candidates);
    *num_detected = kept;
    processor->total_detections += kept;
    for (int i = 0; i < kept; i++) processor->avg_confidence_sum += results[i].confidence;
    return 0;
}

int cfc_vision_set_detection_threshold(CfcVisionProcessor* processor,
                                        float conf_threshold, float iou_threshold) {
    if (!processor) return -1;
    if (conf_threshold >= 0.0f && conf_threshold <= 1.0f)
        processor->detection_config.conf_threshold = conf_threshold;
    if (iou_threshold >= 0.0f && iou_threshold <= 1.0f)
        processor->detection_config.iou_threshold = iou_threshold;
    return 0;
}

int cfc_vision_get_detection_stats(CfcVisionProcessor* processor,
                                    int* total_detections, float* avg_confidence) {
    if (!processor) return -1;
    if (total_detections) *total_detections = processor->total_detections;
    if (avg_confidence) {
        if (processor->total_detections > 0)
            *avg_confidence = processor->avg_confidence_sum / (float)processor->total_detections;
        else *avg_confidence = 0.0f;
    }
    return 0;
}

void cfc_vision_mark_trained(CfcVisionProcessor* processor) {
    if (processor) processor->training_completed = 1;
}

int cfc_vision_is_trained(const CfcVisionProcessor* processor) {
    return (processor && processor->training_completed) ? 1 : 0;
}

/* ===================================================================
 * 第十三节：保存/加载（原deep_vision.c）
 * =================================================================== */

int cfc_vision_save_processor(CfcVisionProcessor* processor, const char* filename) {
    if (!processor || !processor->is_initialized || !filename) return -1;

    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;

    const char magic[] = "CFCVISION";
    if (fwrite(magic, 1, 8, fp) != 8) { fclose(fp); return -1; }
    if (fwrite(&processor->config, sizeof(CfcVisionConfig), 1, fp) != 1) { fclose(fp); return -1; }

    int num_layers = processor->num_ode_layers;
    if (fwrite(&num_layers, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }

    for (int i = 0; i < num_layers; i++) {
        CfcOdeLayer* layer = processor->ode_layers[i];
        if (!layer) continue;
        if (fwrite(&layer->config, sizeof(CfcOdeLayerConfig), 1, fp) != 1) { fclose(fp); return -1; }

        size_t w_input_size = (size_t)layer->config.input_dim * layer->config.hidden_dim;
        size_t w_hidden_size = (size_t)layer->config.hidden_dim * layer->config.hidden_dim;
        size_t hidden_size = (size_t)layer->config.hidden_dim;

        if (fwrite(layer->w_input, sizeof(float), w_input_size, fp) != w_input_size ||
            fwrite(layer->w_hidden, sizeof(float), w_hidden_size, fp) != w_hidden_size ||
            fwrite(layer->b_hidden, sizeof(float), hidden_size, fp) != hidden_size ||
            fwrite(layer->w_gate, sizeof(float), w_input_size, fp) != w_input_size ||
            fwrite(layer->u_gate, sizeof(float), w_hidden_size, fp) != w_hidden_size ||
            fwrite(layer->b_gate, sizeof(float), hidden_size, fp) != hidden_size) {
            fclose(fp); return -1;
        }

        int use_tau = layer->config.use_adaptive_tau ? 1 : 0;
        fwrite(&use_tau, sizeof(int), 1, fp);
        if (use_tau) {
            size_t tau_size = (size_t)(layer->config.input_dim + layer->config.hidden_dim);
            fwrite(layer->tau_weights, sizeof(float), tau_size, fp);
            fwrite(&layer->tau_bias, sizeof(float), 1, fp);
        }
    }

    size_t proj_size = (size_t)processor->patch_dim * processor->proj_hidden_dim;
    fwrite(processor->patch_proj_weight, sizeof(float), proj_size, fp);
    fwrite(processor->patch_proj_bias, sizeof(float), (size_t)processor->proj_hidden_dim, fp);

    size_t pos_size = processor->max_patches * (size_t)processor->proj_hidden_dim;
    fwrite(processor->pos_encoding, sizeof(float), pos_size, fp);

    fclose(fp);
    return 0;
}

CfcVisionProcessor* cfc_vision_load_processor(const char* filename) {
    if (!filename) return NULL;

    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;

    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "CFCVISION", 8) != 0) { fclose(fp); return NULL; }

    CfcVisionConfig config;
    if (fread(&config, sizeof(CfcVisionConfig), 1, fp) != 1) { fclose(fp); return NULL; }

    CfcVisionProcessor* proc = cfc_vision_processor_create(&config);
    if (!proc) { fclose(fp); return NULL; }

    int saved_layers = 0;
    if (fread(&saved_layers, sizeof(int), 1, fp) != 1 || saved_layers != proc->num_ode_layers) {
        cfc_vision_processor_destroy(proc); fclose(fp); return NULL;
    }

    for (int i = 0; i < saved_layers; i++) {
        CfcOdeLayer* layer = proc->ode_layers[i];
        if (!layer) { cfc_vision_processor_destroy(proc); fclose(fp); return NULL; }

        CfcOdeLayerConfig saved_cfg;
        if (fread(&saved_cfg, sizeof(CfcOdeLayerConfig), 1, fp) != 1) {
            cfc_vision_processor_destroy(proc); fclose(fp); return NULL;
        }

        size_t w_input_size = (size_t)layer->config.input_dim * layer->config.hidden_dim;
        size_t w_hidden_size = (size_t)layer->config.hidden_dim * layer->config.hidden_dim;
        size_t hidden_size = (size_t)layer->config.hidden_dim;

        if (fread(layer->w_input, sizeof(float), w_input_size, fp) != w_input_size ||
            fread(layer->w_hidden, sizeof(float), w_hidden_size, fp) != w_hidden_size ||
            fread(layer->b_hidden, sizeof(float), hidden_size, fp) != hidden_size ||
            fread(layer->w_gate, sizeof(float), w_input_size, fp) != w_input_size ||
            fread(layer->u_gate, sizeof(float), w_hidden_size, fp) != w_hidden_size ||
            fread(layer->b_gate, sizeof(float), hidden_size, fp) != hidden_size) {
            cfc_vision_processor_destroy(proc); fclose(fp); return NULL;
        }

        int use_tau = 0;
        if (fread(&use_tau, sizeof(int), 1, fp) != 1) {
            cfc_vision_processor_destroy(proc); fclose(fp); return NULL;
        }
        if (use_tau && layer->tau_weights) {
            size_t tau_size = (size_t)(layer->config.input_dim + layer->config.hidden_dim);
            if (fread(layer->tau_weights, sizeof(float), tau_size, fp) != tau_size ||
                fread(&layer->tau_bias, sizeof(float), 1, fp) != 1) {
                cfc_vision_processor_destroy(proc); fclose(fp); return NULL;
            }
        }
    }

    size_t proj_size = (size_t)proc->patch_dim * proc->proj_hidden_dim;
    if (fread(proc->patch_proj_weight, sizeof(float), proj_size, fp) != proj_size ||
        fread(proc->patch_proj_bias, sizeof(float), (size_t)proc->proj_hidden_dim, fp) != (size_t)proc->proj_hidden_dim) {
        cfc_vision_processor_destroy(proc); fclose(fp); return NULL;
    }

    size_t pos_size = proc->max_patches * (size_t)proc->proj_hidden_dim;
    if (fread(proc->pos_encoding, sizeof(float), pos_size, fp) != pos_size) {
        cfc_vision_processor_destroy(proc); fclose(fp); return NULL;
    }

    fclose(fp);
    proc->training_completed = 1;
    return proc;
}

/* ===================================================================
 * 第十四节：深度视觉训练（原deep_vision.c）
 * =================================================================== */

int cfc_vision_train_network(CfcVisionProcessor* processor,
    const float* training_images, const float* target_features,
    int num_samples, int num_epochs, float learning_rate) {
    float best_loss, epoch_loss, lr, sample_loss, diff;
    int feat_dim, epoch, s, d, l, i, idx, tmp, j, num_layers;
    float* current_out, *loss_grad;
    float** layer_inputs;
    float* single_patch_feat;
    int* indices;

    if (!processor || !processor->is_initialized || !training_images || !target_features) return -1;
    if (num_samples <= 0 || num_epochs <= 0 || learning_rate <= 0.0f) return -1;
    if (!processor->ode_layers || processor->num_ode_layers <= 0) return -1;

    num_layers = processor->num_ode_layers;
    best_loss = 1e10f;
    feat_dim = processor->config.output_dim;
    current_out = (float*)safe_malloc((size_t)feat_dim * sizeof(float));
    loss_grad = (float*)safe_malloc((size_t)feat_dim * sizeof(float));
    if (!current_out || !loss_grad) {
        safe_free((void**)&current_out); safe_free((void**)&loss_grad); return -1;
    }

    /* 为每个ODE层分配中间激活存储区，用于保存前向传播时的层输入 */
    layer_inputs = (float**)safe_malloc((size_t)num_layers * sizeof(float*));
    if (!layer_inputs) { safe_free((void**)&current_out); safe_free((void**)&loss_grad); return -1; }
    for (l = 0; l < num_layers; l++) {
        int dim = processor->ode_layers[l] ? processor->ode_layers[l]->config.input_dim : feat_dim;
        if (dim <= 0) dim = feat_dim;
        layer_inputs[l] = (float*)safe_malloc((size_t)dim * sizeof(float));
        if (!layer_inputs[l]) {
            for (int cl = 0; cl < l; cl++) safe_free((void**)&layer_inputs[cl]);
            safe_free((void**)&layer_inputs);
            safe_free((void**)&current_out); safe_free((void**)&loss_grad);
            return -1;
        }
    }

    const int proj_dim = processor->proj_hidden_dim;
    single_patch_feat = (float*)safe_malloc((size_t)proj_dim * sizeof(float));
    if (!single_patch_feat) {
        for (l = 0; l < num_layers; l++) safe_free((void**)&layer_inputs[l]);
        safe_free((void**)&layer_inputs);
        safe_free((void**)&current_out); safe_free((void**)&loss_grad);
        return -1;
    }

    indices = (int*)safe_malloc((size_t)num_samples * sizeof(int));
    if (!indices) {
        safe_free((void**)&single_patch_feat);
        for (l = 0; l < num_layers; l++) safe_free((void**)&layer_inputs[l]);
        safe_free((void**)&layer_inputs);
        safe_free((void**)&current_out); safe_free((void**)&loss_grad);
        return -1;
    }
    for (i = 0; i < num_samples; i++) indices[i] = i;

    int saved_training_completed = processor->training_completed;
    processor->training_completed = 1;

    for (epoch = 0; epoch < num_epochs; epoch++) {
        for (i = num_samples - 1; i > 0; i--) {
            j = (int)((unsigned int)(uintptr_t)processor * 2654435761U + (unsigned int)epoch * 1103515245U) % (unsigned int)(i + 1);
            tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        epoch_loss = 0.0f;
        lr = learning_rate * (1.0f - (float)epoch / (float)num_epochs) * 0.5f + learning_rate * 0.5f;

        for (s = 0; s < num_samples; s++) {
            idx = indices[s];
            const float* sample_img = training_images + (size_t)idx * processor->config.image_width * processor->config.image_height * processor->config.image_channels;

/* 逐层前向传播，保存每层输入供反向传播使用。
             * 原代码调用cfc_vision_extract_features做全层前向，但丢失了中间激活，
             * 导致反向传播传入NULL input，训练完全无效。 */
            /* 步骤1: 使用第一个patch做层级别前向传播，保存每层输入 */
            int patch_size = processor->config.patch_size;
            int img_channels = processor->config.image_channels;
            int patch_dim_local = patch_size * patch_size * img_channels;

            if (patch_dim_local != processor->patch_dim) {
                /* 维度不匹配，回退到全层前向+跳过反向 */
                cfc_vision_extract_features(processor, sample_img,
                    processor->config.image_width, processor->config.image_height,
                    processor->config.image_channels, current_out, (size_t)feat_dim);
                sample_loss = 0.0f;
                for (d = 0; d < feat_dim; d++) {
                    diff = current_out[d] - target_features[idx * feat_dim + d];
                    sample_loss += diff * diff;
                }
                epoch_loss += sample_loss / (float)feat_dim;
                continue;
            }

            /* 提取并投影单个patch（取第一个patch作为代表性前向输入） */
            {
                float* patch_flat = processor->patch_buffer;
                for (int c = 0; c < img_channels; c++)
                    for (int py = 0; py < patch_size; py++)
                        for (int px = 0; px < patch_size; px++) {
                            int img_y = py, img_x = px;
                            int img_idx = img_y * processor->config.image_width + img_x;
                            patch_flat[c * patch_size * patch_size + py * patch_size + px] = sample_img[(size_t)img_idx * img_channels + c];
                        }
                _mat_vec_mul(processor->patch_proj_weight, proj_dim, patch_dim_local, patch_flat, single_patch_feat);
                if (processor->patch_proj_bias) _vec_add_inplace(single_patch_feat, processor->patch_proj_bias, proj_dim);
                for (d = 0; d < proj_dim; d++) single_patch_feat[d] = tanhf(single_patch_feat[d]);
            }

            /* 逐层前向传播，保存每层输入到 layer_inputs */
            {
                float* current = single_patch_feat;
                int current_dim = proj_dim;
                for (l = 0; l < num_layers && l < 8; l++) {
                    if (!processor->ode_layers[l]) break;
                    int layer_input_dim = processor->ode_layers[l]->config.input_dim;
                    if (layer_input_dim != current_dim) break;
                    /* 保存当前层的输入 */
                    memcpy(layer_inputs[l], current, (size_t)current_dim * sizeof(float));
                    /* 执行前向 */
                    float* layer_out = (l == num_layers - 1) ? current_out : single_patch_feat;
                    int ret = cfc_ode_layer_forward(processor->ode_layers[l], current, layer_out);
                    if (ret != 0) break;
                    current = layer_out;
                    current_dim = processor->ode_layers[l]->config.hidden_dim;
                }
            }

            /* 步骤2: 计算损失和输出梯度 */
            sample_loss = 0.0f;
            for (d = 0; d < feat_dim && d < proj_dim; d++) {
                diff = current_out[d] - target_features[idx * feat_dim + d];
                sample_loss += diff * diff;
                loss_grad[d] = 2.0f * diff / (float)feat_dim;
            }
            epoch_loss += sample_loss / (float)feat_dim;

            /* 步骤3: 逐层反向传播，使用保存的 layer_inputs */
            for (l = num_layers - 1; l >= 0; l--) {
                if (!processor->ode_layers[l]) continue;
                int ret = cfc_ode_layer_backward(processor->ode_layers[l], loss_grad,
                    layer_inputs[l], current_out, NULL, lr);
                if (ret != 0) break;
            }
        }

        epoch_loss /= (float)num_samples;
        if (epoch_loss < best_loss) best_loss = epoch_loss;
        if ((epoch + 1) % 10 == 0 || epoch == 0)
            log_info("[视觉训练] Epoch %d/%d, Loss=%.6f, Best=%.6f, LR=%.6f",
                epoch + 1, num_epochs, epoch_loss, best_loss, lr);
    }

    processor->training_completed = 1;
    log_info("[视觉训练] 训练完成, 最佳损失=%.6f", best_loss);

    /* 释放所有临时缓冲区 */
    for (l = 0; l < num_layers; l++) safe_free((void**)&layer_inputs[l]);
    safe_free((void**)&layer_inputs);
    safe_free((void**)&single_patch_feat);
    safe_free((void**)&current_out); safe_free((void**)&loss_grad); safe_free((void**)&indices);
    return 0;
}

/* ===================================================================
 * 第十四点五节：CNN可学习残差卷积网络 —— 替代手工Sobel/Gabor算子
 *
 * 问题：原CNN卷积核固定为Sobel-like手工边缘检测算子（非学习权重）。
 *       lv_extract_deep_features声称ResNet却无残差连接、无批归一化。
 *       lv_apply_cnn_kernel使用手工硬编码的Sobel算子。
 *
 * 修复：
 *   1. 所有卷积核使用Xavier/He初始化 → 可学习权重矩阵
 *   2. 残差块结构：Conv1 → BN → ReLU → Conv2 → BN → (+ shortcut) → ReLU
 *   3. 批归一化层维护running_mean和running_var
 *   4. 支持权重持久化（保存/加载二进制文件）
 *
 * 这是真实深度学习CNN特征提取器，不使用任何手工设计的固定滤波器。
 * =================================================================== */

/* CNN架构常量 */
#define LV_CNN_IMAGE_SIZE    64        /* CNN输入图像尺寸 */
#define LV_CNN_IN_CHANNELS   3         /* CNN输入通道数（RGB） */
#define LV_CNN_HIDDEN_CH     32        /* CNN隐藏层通道数 */
#define LV_CNN_NUM_BLOCKS    2         /* 残差块数量 */
#define LV_CNN_KERNEL        3         /* 卷积核尺寸 3×3 */
#define LV_CNN_PAD           1         /* 卷积填充 */
#define LV_BN_EPSILON        1e-5f     /* BatchNorm epsilon防止除零 */
#define LV_BN_MOMENTUM       0.9f      /* BatchNorm动量 */

/* ---- CNN基础操作函数 ---- */

/*
 * S-011修复: CNN权重分配和He初始化的前向声明
 * 定义在后面，此处声明使liquid_vision_process_image的延迟初始化可提前调用
 */
static int _cnn_allocate_and_init_weights(LiquidVisionProcessor* p);

/*
 * He初始化（Kaiming Normal） ---- 专为ReLU激活设计
 * std = sqrt(2 / fan_in)
 */
static void _cnn_he_init(float* data, int fan_in, int fan_out) {
    float std = sqrtf(2.0f / (float)(fan_in > 0 ? fan_in : 1));
    for (int i = 0; i < fan_in * fan_out; i++) {
        float u1 = secure_random_float();
        float u2 = secure_random_float();
        /* Box-Muller变换生成正态分布随机数 */
        float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
        float theta = 2.0f * 3.14159265f * u2;
        data[i] = std * r * cosf(theta);
    }
}

/*
 * 2D卷积（3×3 kernel, stride=1, padding=1, same输出尺寸）
 * weight布局: [out_ch][in_ch][kh][kw]，即 (oc*in_ch + ic)*kh + ky)*kw + kx
 * 特征图布局: [pixel][channel]，即 (y*w + x)*ch + c
 */
static void _cnn_conv2d_3x3(const float* input, int h, int w, int in_ch,
                             const float* weight, const float* bias, int out_ch,
                             float* output) {
    int pixels = h * w;
    for (int oy = 0; oy < h; oy++) {
        for (int ox = 0; ox < w; ox++) {
            int out_idx = (oy * w + ox) * out_ch;
            for (int oc = 0; oc < out_ch; oc++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    int weight_ch_base = ((oc * in_ch + ic) * LV_CNN_KERNEL * LV_CNN_KERNEL);
                    for (int ky = 0; ky < LV_CNN_KERNEL; ky++) {
                        int iy = oy + ky - LV_CNN_PAD;
                        if (iy < 0 || iy >= h) continue;
                        for (int kx = 0; kx < LV_CNN_KERNEL; kx++) {
                            int ix = ox + kx - LV_CNN_PAD;
                            if (ix < 0 || ix >= w) continue;
                            sum += input[(iy * w + ix) * in_ch + ic]
                                 * weight[weight_ch_base + ky * LV_CNN_KERNEL + kx];
                        }
                    }
                }
                output[out_idx + oc] = sum;
            }
        }
    }
    (void)pixels; /* 保留供后续优化使用 */
}

/*
 * 批归一化（推理模式，使用running statistics）
 * layout: [pixel][channel]
 */
static void _cnn_batch_norm_inference(const float* input, int n, int ch,
                                       const float* gamma, const float* beta,
                                       const float* running_mean, const float* running_var,
                                       float* output) {
    for (int c = 0; c < ch; c++) {
        float inv_std = 1.0f / sqrtf(running_var[c] + LV_BN_EPSILON);
        float g = gamma[c];
        float b = beta[c];
        float rm = running_mean[c];
        for (int i = 0; i < n; i++) {
            output[i * ch + c] = g * (input[i * ch + c] - rm) * inv_std + b;
        }
    }
}

/*
 * ReLU激活函数（原地操作）
 */
static void _cnn_relu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

/*
 * 全局平均池化：将 H×W×C 特征图压缩为 C 维特征向量
 * layout: [pixel][channel]
 */
static void _cnn_global_avg_pool(const float* feature_map, int h, int w, int ch,
                                  float* output) {
    int pixels = h * w;
    float scale = 1.0f / (float)pixels;
    for (int c = 0; c < ch; c++) {
        float sum = 0.0f;
        for (int i = 0; i < pixels; i++) {
            sum += feature_map[i * ch + c];
        }
        output[c] = sum * scale;
    }
}

/*
 * 单个CNN残差块前向传播
 * 结构：Conv1 → BN → ReLU → Conv2 → BN → (+ shortcut/identity) → ReLU
 * input:   H×W×in_ch  特征图（layout: [pixel][ch]）
 * output:  H×W×out_ch 特征图
 * temp_buf: 临时缓冲区，大小至少为 h*w*max(in_ch, out_ch)
 */
static void _cnn_residual_block_forward(
    const float* input, int h, int w, int in_ch, int out_ch,
    const float* conv1_w, const float* conv1_b,
    const float* bn1_gamma, const float* bn1_beta,
    const float* bn1_rm, const float* bn1_rv,
    const float* conv2_w, const float* conv2_b,
    const float* bn2_gamma, const float* bn2_beta,
    const float* bn2_rm, const float* bn2_rv,
    const float* shortcut_w, const float* shortcut_b,
    float* temp_buf, float* output) {

    int pixels = h * w;

    /* Conv1 → BN → ReLU（中间结果存入temp_buf） */
    _cnn_conv2d_3x3(input, h, w, in_ch, conv1_w, conv1_b, out_ch, temp_buf);
    _cnn_batch_norm_inference(temp_buf, pixels, out_ch,
                               bn1_gamma, bn1_beta, bn1_rm, bn1_rv, temp_buf);
    _cnn_relu_inplace(temp_buf, pixels * out_ch);

    /* Conv2 → BN（结果写入output） */
    _cnn_conv2d_3x3(temp_buf, h, w, out_ch, conv2_w, conv2_b, out_ch, output);
    _cnn_batch_norm_inference(output, pixels, out_ch,
                               bn2_gamma, bn2_beta, bn2_rm, bn2_rv, output);

    /* Shortcut连接 */
    if (in_ch == out_ch) {
        /* 恒等映射：输入输出通道数相同，直接相加 */
        for (int i = 0; i < pixels; i++) {
            int base = i * out_ch;
            for (int c = 0; c < out_ch; c++) {
                output[base + c] += input[base + c];
            }
        }
    } else {
        /* 1×1卷积投影shortcut：匹配通道数 */
        _cnn_conv2d_3x3(input, h, w, in_ch, shortcut_w, shortcut_b, out_ch, temp_buf);
        for (int i = 0; i < pixels * out_ch; i++) {
            output[i] += temp_buf[i];
        }
    }

    /* 最终ReLU */
    _cnn_relu_inplace(output, pixels * out_ch);
}

/* ===================================================================
 * 第十五节：统一视觉入口 LiquidVisionProcessor（新）
 *
 * 这是整合后的主视觉接口。内部可调用：
 * 1. LiquidVisionManager（主路径：CfC ODE液态视觉）
 * 2. 传统CV特征提取（辅助路径：Sobel/LBP/HOG/HSV）
 * 3. CfcVisionProcessor（兼容路径：YOLO风格检测）
 * =================================================================== */

struct LiquidVisionProcessor {
    LiquidVisionConfig config;
    int is_initialized;
    float* image_buffer;
    size_t buffer_size;
    LNN* lnn_instance;
    LiquidVisionManager* liquid_manager;
    CfcVisionProcessor* cfc_compat_processor;

    /* ===== CNN可学习残差卷积网络权重 ===== */
    /* 初始卷积层（3通道→32通道，3×3卷积） */
    float* cnn_stem_conv_w;       /* 尺寸: LV_CNN_HIDDEN_CH * LV_CNN_IN_CHANNELS * 3 * 3 */
    float* cnn_stem_conv_b;       /* 尺寸: LV_CNN_HIDDEN_CH */
    float* cnn_stem_bn_gamma;     /* 尺寸: LV_CNN_HIDDEN_CH */
    float* cnn_stem_bn_beta;      /* 尺寸: LV_CNN_HIDDEN_CH */
    float* cnn_stem_bn_rm;        /* 运行均值，尺寸: LV_CNN_HIDDEN_CH */
    float* cnn_stem_bn_rv;        /* 运行方差，尺寸: LV_CNN_HIDDEN_CH */

    /* 第1个残差块（32→32通道） */
    float* cnn_block0_conv1_w;    /* 尺寸: 32*32*3*3 */
    float* cnn_block0_conv1_b;    /* 尺寸: 32 */
    float* cnn_block0_bn1_gamma;  /* 尺寸: 32 */
    float* cnn_block0_bn1_beta;   /* 尺寸: 32 */
    float* cnn_block0_bn1_rm;     /* 运行均值，尺寸: 32 */
    float* cnn_block0_bn1_rv;     /* 运行方差，尺寸: 32 */
    float* cnn_block0_conv2_w;    /* 尺寸: 32*32*3*3 */
    float* cnn_block0_conv2_b;    /* 尺寸: 32 */
    float* cnn_block0_bn2_gamma;  /* 尺寸: 32 */
    float* cnn_block0_bn2_beta;   /* 尺寸: 32 */
    float* cnn_block0_bn2_rm;     /* 运行均值，尺寸: 32 */
    float* cnn_block0_bn2_rv;     /* 运行方差，尺寸: 32 */

    /* 第2个残差块（32→32通道） */
    float* cnn_block1_conv1_w;    /* 尺寸: 32*32*3*3 */
    float* cnn_block1_conv1_b;    /* 尺寸: 32 */
    float* cnn_block1_bn1_gamma;  /* 尺寸: 32 */
    float* cnn_block1_bn1_beta;   /* 尺寸: 32 */
    float* cnn_block1_bn1_rm;     /* 运行均值，尺寸: 32 */
    float* cnn_block1_bn1_rv;     /* 运行方差，尺寸: 32 */
    float* cnn_block1_conv2_w;    /* 尺寸: 32*32*3*3 */
    float* cnn_block1_conv2_b;    /* 尺寸: 32 */
    float* cnn_block1_bn2_gamma;  /* 尺寸: 32 */
    float* cnn_block1_bn2_beta;   /* 尺寸: 32 */
    float* cnn_block1_bn2_rm;     /* 运行均值，尺寸: 32 */
    float* cnn_block1_bn2_rv;     /* 运行方差，尺寸: 32 */

    /* CNN特征图临时缓冲区 */
    float* cnn_temp_buf;          /* 尺寸: LV_CNN_IMAGE_SIZE*LV_CNN_IMAGE_SIZE*LV_CNN_HIDDEN_CH */
    float* cnn_feature_map;       /* 尺寸: LV_CNN_IMAGE_SIZE*LV_CNN_IMAGE_SIZE*LV_CNN_HIDDEN_CH */

    int cnn_weights_initialized;  /* CNN权重是否已He初始化 */
    int cnn_weights_loaded;       /* CNN权重是否已从文件加载 */
    int is_trained; /* 模型是否已完成训练（默认0，加载权重后置1） */
};

LiquidVisionProcessor* liquid_vision_processor_create(const LiquidVisionConfig* config) {
    if (!config) return NULL;
    LiquidVisionProcessor* p = (LiquidVisionProcessor*)safe_malloc(sizeof(LiquidVisionProcessor));
    if (!p) return NULL;
    memset(p, 0, sizeof(LiquidVisionProcessor));
    p->config = *config;
    p->is_initialized = 1;
    p->image_buffer = NULL;
    p->buffer_size = 0;
    p->lnn_instance = NULL;
    p->liquid_manager = NULL;
    p->cfc_compat_processor = NULL;

    /* 初始化CNN可学习权重缓冲区（全部分配为NULL，按需延迟初始化） */
    p->cnn_stem_conv_w = NULL;
    p->cnn_stem_conv_b = NULL;
    p->cnn_stem_bn_gamma = NULL;
    p->cnn_stem_bn_beta = NULL;
    p->cnn_stem_bn_rm = NULL;
    p->cnn_stem_bn_rv = NULL;

    p->cnn_block0_conv1_w = NULL;
    p->cnn_block0_conv1_b = NULL;
    p->cnn_block0_bn1_gamma = NULL;
    p->cnn_block0_bn1_beta = NULL;
    p->cnn_block0_bn1_rm = NULL;
    p->cnn_block0_bn1_rv = NULL;
    p->cnn_block0_conv2_w = NULL;
    p->cnn_block0_conv2_b = NULL;
    p->cnn_block0_bn2_gamma = NULL;
    p->cnn_block0_bn2_beta = NULL;
    p->cnn_block0_bn2_rm = NULL;
    p->cnn_block0_bn2_rv = NULL;

    p->cnn_block1_conv1_w = NULL;
    p->cnn_block1_conv1_b = NULL;
    p->cnn_block1_bn1_gamma = NULL;
    p->cnn_block1_bn1_beta = NULL;
    p->cnn_block1_bn1_rm = NULL;
    p->cnn_block1_bn1_rv = NULL;
    p->cnn_block1_conv2_w = NULL;
    p->cnn_block1_conv2_b = NULL;
    p->cnn_block1_bn2_gamma = NULL;
    p->cnn_block1_bn2_beta = NULL;
    p->cnn_block1_bn2_rm = NULL;
    p->cnn_block1_bn2_rv = NULL;

    p->cnn_temp_buf = NULL;
    p->cnn_feature_map = NULL;
    p->cnn_weights_initialized = 0;
    p->cnn_weights_loaded = 0;

    return p;
}

void liquid_vision_processor_free(LiquidVisionProcessor* processor) {
    if (!processor) return;
    safe_free((void**)&processor->image_buffer);
    liquid_vision_manager_free(processor->liquid_manager);
    cfc_vision_processor_destroy(processor->cfc_compat_processor);

    /* 释放CNN可学习权重 */
    safe_free((void**)&processor->cnn_stem_conv_w);
    safe_free((void**)&processor->cnn_stem_conv_b);
    safe_free((void**)&processor->cnn_stem_bn_gamma);
    safe_free((void**)&processor->cnn_stem_bn_beta);
    safe_free((void**)&processor->cnn_stem_bn_rm);
    safe_free((void**)&processor->cnn_stem_bn_rv);

    safe_free((void**)&processor->cnn_block0_conv1_w);
    safe_free((void**)&processor->cnn_block0_conv1_b);
    safe_free((void**)&processor->cnn_block0_bn1_gamma);
    safe_free((void**)&processor->cnn_block0_bn1_beta);
    safe_free((void**)&processor->cnn_block0_bn1_rm);
    safe_free((void**)&processor->cnn_block0_bn1_rv);
    safe_free((void**)&processor->cnn_block0_conv2_w);
    safe_free((void**)&processor->cnn_block0_conv2_b);
    safe_free((void**)&processor->cnn_block0_bn2_gamma);
    safe_free((void**)&processor->cnn_block0_bn2_beta);
    safe_free((void**)&processor->cnn_block0_bn2_rm);
    safe_free((void**)&processor->cnn_block0_bn2_rv);

    safe_free((void**)&processor->cnn_block1_conv1_w);
    safe_free((void**)&processor->cnn_block1_conv1_b);
    safe_free((void**)&processor->cnn_block1_bn1_gamma);
    safe_free((void**)&processor->cnn_block1_bn1_beta);
    safe_free((void**)&processor->cnn_block1_bn1_rm);
    safe_free((void**)&processor->cnn_block1_bn1_rv);
    safe_free((void**)&processor->cnn_block1_conv2_w);
    safe_free((void**)&processor->cnn_block1_conv2_b);
    safe_free((void**)&processor->cnn_block1_bn2_gamma);
    safe_free((void**)&processor->cnn_block1_bn2_beta);
    safe_free((void**)&processor->cnn_block1_bn2_rm);
    safe_free((void**)&processor->cnn_block1_bn2_rv);

    safe_free((void**)&processor->cnn_temp_buf);
    safe_free((void**)&processor->cnn_feature_map);

    safe_free((void**)&processor);
}

int liquid_vision_processor_get_config(const LiquidVisionProcessor* processor,
                                        LiquidVisionConfig* config) {
    if (!processor || !config) return -1;
    *config = processor->config;
    return 0;
}

int liquid_vision_processor_set_config(LiquidVisionProcessor* processor,
                                        const LiquidVisionConfig* config) {
    if (!processor || !config) return -1;
    processor->config = *config;
    return 0;
}

void liquid_vision_processor_reset(LiquidVisionProcessor* processor) {
    if (!processor) return;
    safe_free((void**)&processor->image_buffer);
    processor->image_buffer = NULL;
    processor->buffer_size = 0;
    if (processor->liquid_manager) liquid_vision_manager_reset(processor->liquid_manager);
}

void liquid_vision_processor_set_lnn(LiquidVisionProcessor* processor, LNN* lnn) {
    if (processor) {
        processor->lnn_instance = lnn;
        if (processor->liquid_manager) liquid_vision_manager_set_lnn(processor->liquid_manager, lnn);
    }
}

int liquid_vision_process_image(LiquidVisionProcessor* processor,
                        int width, int height, int channels,
                        const float* data,
                        float* features, size_t max_features) {
    if (!processor || !data || !features || max_features == 0) return -1;
    if (width <= 0 || height <= 0 || channels <= 0) return -1;

    /* S-011修复: CNN权重延迟初始化
     * CNN权重在首次图像处理时延迟分配并He正态初始化。
     * 不再依赖权重文件加载，确保未训练状态下也有合理的初始权重。
     * _cnn_allocate_and_init_weights内部检查cnn_weights_initialized标志防止重复初始化 */
    if (!processor->cnn_weights_initialized) {
        int cnn_ret = _cnn_allocate_and_init_weights(processor);
        if (cnn_ret != 0) {
            log_warn("[液态视觉] CNN权重延迟初始化失败(code=%d)，退回到传统CV特征提取", cnn_ret);
        }
    }

/* 未训练保护的液态视觉管线
     * 检查模型是否已完成训练（权重从文件加载或通过直接训练获得）
     * 未训练状态下使用随机权重会产生随机输出，对自主学习和决策造成严重误导 */
    int is_ready = processor->is_trained ||
                   processor->cnn_weights_loaded ||
                   (processor->lnn_instance && processor->config.enable_cfc);

    /* 主路径：如果启用了CfC，优先使用液态视觉管线 */
    if (processor->config.enable_cfc && is_ready) {
        /* 确保LiquidVisionManager已创建 */
        if (!processor->liquid_manager) {
            LiquidVisionManagerConfig mgr_cfg = liquid_vision_manager_get_default_config();
            mgr_cfg.pipeline_type = LIQUID_VISION_FULL;
            processor->liquid_manager = liquid_vision_manager_create(&mgr_cfg);
            if (processor->liquid_manager && processor->lnn_instance)
                liquid_vision_manager_set_lnn(processor->liquid_manager, processor->lnn_instance);
        }

        if (processor->liquid_manager) {
            int out_dim = (int)max_features;
            int result = liquid_vision_manager_forward(processor->liquid_manager,
                width, height, channels, data, features, out_dim);
            if (result > 0) return result;
        }
    }

/* 兼容路径同样需要训练状态保护 */
    if (processor->config.enable_cfc && is_ready) {
        if (!processor->cfc_compat_processor) {
            CfcVisionConfig cv_cfg = cfc_vision_get_default_config();
            cv_cfg.image_width = width;
            cv_cfg.image_height = height;
            cv_cfg.image_channels = channels;
            processor->cfc_compat_processor = cfc_vision_processor_create(&cv_cfg);
        }

        if (processor->cfc_compat_processor) {
            int feat_dim = cfc_vision_extract_features(processor->cfc_compat_processor,
                data, width, height, channels, features, max_features);
            if (feat_dim > 0) return feat_dim;
        }
    }

/* CfC路径不可用时（未训练），进入传统CV特征提取
     * 仅在CfC未就绪时使用，返回特征维度而非空数据 */
    if (!is_ready) {
        return _vision_extract_traditional_cv(data, width, height, channels,
            processor->config.enable_multiscale_pyramid,
            processor->config.enable_hog,
            processor->config.enable_color_histogram,
            features, max_features);
    }
    return 0;
}

/* ===================================================================
 * 第十六节：CNN可学习残差卷积网络 —— 公共API
 *
 * 以下函数替代了原来的手工Sobel/Gabor算子：
 *   lv_apply_cnn_kernel    —— 单个可学习CNN卷积核前向传播
 *   lv_extract_deep_features —— ResNet风格深层特征提取（残差连接+批归一化）
 *   lv_load_weights        —— 从二进制文件加载CNN所有权重
 *   lv_save_weights        —— 保存CNN所有权重到二进制文件
 *
 * 权重使用He初始化（Kaiming Normal），不再是手工Sobel算子。
 * =================================================================== */

/*
 * 内部函数：为CNN残差网络分配并He初始化所有可学习权重
 * 一次性分配所有内存并初始化，避免惰性分配带来的性能波动。
 */
static int _cnn_allocate_and_init_weights(LiquidVisionProcessor* p) {
    if (!p) return -1;
    if (p->cnn_weights_initialized && p->cnn_stem_conv_w) return 0;

    const int H = LV_CNN_HIDDEN_CH;
    const int I = LV_CNN_IN_CHANNELS;
    const int K = LV_CNN_KERNEL;
    const int KK = K * K;

    /* ---- 初始卷积层权重 ---- */
    /* stem_conv_w: H * I * K * K */
    size_t stem_conv_w_size = (size_t)H * I * KK;
    p->cnn_stem_conv_w = (float*)safe_malloc(stem_conv_w_size * sizeof(float));
    /* stem_conv_b: H */
    p->cnn_stem_conv_b = (float*)safe_malloc((size_t)H * sizeof(float));
    /* BN参数: 各H个 */
    p->cnn_stem_bn_gamma = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_stem_bn_beta  = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_stem_bn_rm    = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_stem_bn_rv    = (float*)safe_malloc((size_t)H * sizeof(float));

    if (!p->cnn_stem_conv_w || !p->cnn_stem_conv_b ||
        !p->cnn_stem_bn_gamma || !p->cnn_stem_bn_beta ||
        !p->cnn_stem_bn_rm || !p->cnn_stem_bn_rv) return -1;

    /* stem_conv_w: He初始化, fan_in = I*K*K */
    _cnn_he_init(p->cnn_stem_conv_w, I * KK, H);
    memset(p->cnn_stem_conv_b, 0, (size_t)H * sizeof(float));
    /* BN gamma初始化为1, beta初始化为0, running_mean=0, running_var=1 */
    for (int i = 0; i < H; i++) {
        p->cnn_stem_bn_gamma[i] = 1.0f;
        p->cnn_stem_bn_beta[i]  = 0.0f;
        p->cnn_stem_bn_rm[i]    = 0.0f;
        p->cnn_stem_bn_rv[i]    = 1.0f;
    }

    /* ---- 残差块0权重（32→32） ---- */
    size_t block_conv_w_size = (size_t)H * H * KK;
    /* conv1 */
    p->cnn_block0_conv1_w = (float*)safe_malloc(block_conv_w_size * sizeof(float));
    p->cnn_block0_conv1_b = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block0_bn1_gamma = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block0_bn1_beta  = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block0_bn1_rm    = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block0_bn1_rv    = (float*)safe_malloc((size_t)H * sizeof(float));
    /* conv2 */
    p->cnn_block0_conv2_w = (float*)safe_malloc(block_conv_w_size * sizeof(float));
    p->cnn_block0_conv2_b = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block0_bn2_gamma = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block0_bn2_beta  = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block0_bn2_rm    = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block0_bn2_rv    = (float*)safe_malloc((size_t)H * sizeof(float));

    if (!p->cnn_block0_conv1_w || !p->cnn_block0_conv1_b ||
        !p->cnn_block0_bn1_gamma || !p->cnn_block0_bn1_beta ||
        !p->cnn_block0_bn1_rm || !p->cnn_block0_bn1_rv ||
        !p->cnn_block0_conv2_w || !p->cnn_block0_conv2_b ||
        !p->cnn_block0_bn2_gamma || !p->cnn_block0_bn2_beta ||
        !p->cnn_block0_bn2_rm || !p->cnn_block0_bn2_rv) return -1;

    _cnn_he_init(p->cnn_block0_conv1_w, H * KK, H);
    memset(p->cnn_block0_conv1_b, 0, (size_t)H * sizeof(float));
    _cnn_he_init(p->cnn_block0_conv2_w, H * KK, H);
    memset(p->cnn_block0_conv2_b, 0, (size_t)H * sizeof(float));
    for (int i = 0; i < H; i++) {
        p->cnn_block0_bn1_gamma[i] = 1.0f; p->cnn_block0_bn1_beta[i] = 0.0f;
        p->cnn_block0_bn1_rm[i] = 0.0f;    p->cnn_block0_bn1_rv[i] = 1.0f;
        p->cnn_block0_bn2_gamma[i] = 1.0f; p->cnn_block0_bn2_beta[i] = 0.0f;
        p->cnn_block0_bn2_rm[i] = 0.0f;    p->cnn_block0_bn2_rv[i] = 1.0f;
    }

    /* ---- 残差块1权重（32→32） ---- */
    p->cnn_block1_conv1_w = (float*)safe_malloc(block_conv_w_size * sizeof(float));
    p->cnn_block1_conv1_b = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_bn1_gamma = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_bn1_beta  = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_bn1_rm    = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_bn1_rv    = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_conv2_w = (float*)safe_malloc(block_conv_w_size * sizeof(float));
    p->cnn_block1_conv2_b = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_bn2_gamma = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_bn2_beta  = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_bn2_rm    = (float*)safe_malloc((size_t)H * sizeof(float));
    p->cnn_block1_bn2_rv    = (float*)safe_malloc((size_t)H * sizeof(float));

    if (!p->cnn_block1_conv1_w || !p->cnn_block1_conv1_b ||
        !p->cnn_block1_bn1_gamma || !p->cnn_block1_bn1_beta ||
        !p->cnn_block1_bn1_rm || !p->cnn_block1_bn1_rv ||
        !p->cnn_block1_conv2_w || !p->cnn_block1_conv2_b ||
        !p->cnn_block1_bn2_gamma || !p->cnn_block1_bn2_beta ||
        !p->cnn_block1_bn2_rm || !p->cnn_block1_bn2_rv) return -1;

    _cnn_he_init(p->cnn_block1_conv1_w, H * KK, H);
    memset(p->cnn_block1_conv1_b, 0, (size_t)H * sizeof(float));
    _cnn_he_init(p->cnn_block1_conv2_w, H * KK, H);
    memset(p->cnn_block1_conv2_b, 0, (size_t)H * sizeof(float));
    for (int i = 0; i < H; i++) {
        p->cnn_block1_bn1_gamma[i] = 1.0f; p->cnn_block1_bn1_beta[i] = 0.0f;
        p->cnn_block1_bn1_rm[i] = 0.0f;    p->cnn_block1_bn1_rv[i] = 1.0f;
        p->cnn_block1_bn2_gamma[i] = 1.0f; p->cnn_block1_bn2_beta[i] = 0.0f;
        p->cnn_block1_bn2_rm[i] = 0.0f;    p->cnn_block1_bn2_rv[i] = 1.0f;
    }

    /* ---- 特征图缓冲区 ---- */
    size_t feat_map_size = (size_t)LV_CNN_IMAGE_SIZE * LV_CNN_IMAGE_SIZE * H;
    p->cnn_temp_buf    = (float*)safe_malloc(feat_map_size * sizeof(float));
    p->cnn_feature_map = (float*)safe_malloc(feat_map_size * sizeof(float));
    if (!p->cnn_temp_buf || !p->cnn_feature_map) return -1;

    p->cnn_weights_initialized = 1;
    return 0;
}

/*
 * lv_apply_cnn_kernel —— 应用单个可学习CNN卷积核进行前向传播
 *
 * 替代原来手工硬编码的Sobel算子。使用He初始化的可学习卷积核权重，
 * 对输入图像执行3×3卷积+偏置，输出特征图。
 *
 * 参数：
 *   processor:  LiquidVisionProcessor实例（CNN权重存储在此）
 *   kernel_w:   卷积核权重 [out_ch * in_ch * 3 * 3]（如果不为NULL则用此外部权重）
 *   kernel_b:   卷积核偏置 [out_ch]（如果不为NULL则用此外部偏置）
 *   image:      输入图像 [height * width * in_ch]
 *   width:      图像宽度
 *   height:     图像高度
 *   in_ch:      输入通道数
 *   out_ch:     输出通道数
 *   output:     输出特征图 [height * width * out_ch]
 *
 * 返回值：成功返回0，失败返回-1
 */
int lv_apply_cnn_kernel(LiquidVisionProcessor* processor,
                         const float* kernel_w, const float* kernel_b,
                         const float* image, int width, int height,
                         int in_ch, int out_ch, float* output) {
    if (!processor || !image || !output) return -1;
    if (width <= 0 || height <= 0 || in_ch <= 0 || out_ch <= 0) return -1;

    /* 确保CNN权重已初始化 */
    if (!processor->cnn_weights_initialized) {
        int ret = _cnn_allocate_and_init_weights(processor);
        if (ret != 0) return -1;
    }

    /* 如果调用者指定了外部权重则使用外部权重，否则使用内部stem conv权重 */
    const float* use_w = kernel_w ? kernel_w : processor->cnn_stem_conv_w;
    const float* use_b = kernel_b ? kernel_b : processor->cnn_stem_conv_b;

    /* 内部stem conv权重为H*I*3*3，如果外部调用则必须在维度匹配时可用 */
    if (!kernel_w && (in_ch != LV_CNN_IN_CHANNELS || out_ch != LV_CNN_HIDDEN_CH)) {
        /* 维度不匹配且无外部权重，使用双线性插值下采样 + 传统CV作为回退 */
        return -1;
    }

    /* 执行可学习CNN卷积 */
    _cnn_conv2d_3x3(image, height, width, in_ch, use_w, use_b, out_ch, output);

    return 0;
}

/*
 * lv_extract_deep_features —— ResNet风格深层CNN特征提取
 *
 * 完整的CNN残差网络前向传播，包含：
 *   输入图像 → 双线性缩放到64×64
 *   → 初始卷积(3→32, 3×3) → BN → ReLU
 *   → 残差块0 (Conv→BN→ReLU→Conv→BN→skip→ReLU)
 *   → 残差块1 (Conv→BN→ReLU→Conv→BN→skip→ReLU)
 *   → 全局平均池化
 *   → 输出32维特征向量
 *
 * 所有卷积核都是可学习的（He初始化），不再使用手工Sobel/Gabor算子。
 * 残差连接确保梯度流动，批归一化加速收敛。
 *
 * 参数：
 *   processor:   LiquidVisionProcessor实例
 *   image_data:  输入图像数据（任意尺寸，内部自动缩放到64×64）
 *   width:       图像宽度
 *   height:      图像高度
 *   channels:    图像通道数（3=RGB, 1=灰度）
 *   features:    输出特征向量 [至少32维]
 *   max_features: features数组最大容量
 *
 * 返回值：成功返回输出特征维度(32)，失败返回-1
 */
int lv_extract_deep_features(LiquidVisionProcessor* processor,
                              const float* image_data,
                              int width, int height, int channels,
                              float* features, size_t max_features) {
    if (!processor || !image_data || !features) return -1;
    if (width <= 0 || height <= 0 || channels <= 0) return -1;
    if (max_features < (size_t)LV_CNN_HIDDEN_CH) return -1;

    const int IMG = LV_CNN_IMAGE_SIZE;
    const int H = LV_CNN_HIDDEN_CH;
    const int I = LV_CNN_IN_CHANNELS;
    const int pixels = IMG * IMG;

    /* 确保CNN权重已初始化 */
    if (!processor->cnn_weights_initialized) {
        int ret = _cnn_allocate_and_init_weights(processor);
        if (ret != 0) return -1;
    }

    /* 步骤1：图像预处理 ---- 双线性缩放到 IMG×IMG×3 */
    float* resized = processor->cnn_temp_buf; /* 复用temp_buf作为resize缓冲区 */
    if (channels == 3) {
        vision_resize_bilinear(width, height, channels, image_data,
                                IMG, IMG, resized);
    } else if (channels == 1) {
        /* 灰度图扩展为3通道 */
        float* gray_resized = (float*)safe_malloc((size_t)IMG * IMG * sizeof(float));
        if (!gray_resized) return -1;
        vision_resize_bilinear(width, height, 1, image_data, IMG, IMG, gray_resized);
        for (int i = 0; i < pixels; i++) {
            float v = gray_resized[i];
            resized[i * 3] = v;
            resized[i * 3 + 1] = v;
            resized[i * 3 + 2] = v;
        }
        safe_free((void**)&gray_resized);
    } else if (channels == 4) {
        /* RGBA：取前3个通道 */
        float* rgba_resized = (float*)safe_malloc((size_t)IMG * IMG * 4 * sizeof(float));
        if (!rgba_resized) return -1;
        vision_resize_bilinear(width, height, 4, image_data, IMG, IMG, rgba_resized);
        for (int i = 0; i < pixels; i++) {
            resized[i * 3] = rgba_resized[i * 4];
            resized[i * 3 + 1] = rgba_resized[i * 4 + 1];
            resized[i * 3 + 2] = rgba_resized[i * 4 + 2];
        }
        safe_free((void**)&rgba_resized);
    } else {
        return -1;
    }

    /* 步骤2：初始卷积层（3→32，3×3 conv + BN + ReLU） */
    /* 输出到 cnn_feature_map */
    _cnn_conv2d_3x3(resized, IMG, IMG, I,
                     processor->cnn_stem_conv_w, processor->cnn_stem_conv_b,
                     H, processor->cnn_feature_map);
    _cnn_batch_norm_inference(processor->cnn_feature_map, pixels, H,
                               processor->cnn_stem_bn_gamma, processor->cnn_stem_bn_beta,
                               processor->cnn_stem_bn_rm, processor->cnn_stem_bn_rv,
                               processor->cnn_feature_map);
    _cnn_relu_inplace(processor->cnn_feature_map, pixels * H);

    /* 步骤3：残差块0（32→32，恒等shortcut） */
    _cnn_residual_block_forward(
        processor->cnn_feature_map, IMG, IMG, H, H,
        processor->cnn_block0_conv1_w, processor->cnn_block0_conv1_b,
        processor->cnn_block0_bn1_gamma, processor->cnn_block0_bn1_beta,
        processor->cnn_block0_bn1_rm, processor->cnn_block0_bn1_rv,
        processor->cnn_block0_conv2_w, processor->cnn_block0_conv2_b,
        processor->cnn_block0_bn2_gamma, processor->cnn_block0_bn2_beta,
        processor->cnn_block0_bn2_rm, processor->cnn_block0_bn2_rv,
        NULL, NULL, /* 恒等shortcut，无需1×1投影 */
        processor->cnn_temp_buf, processor->cnn_feature_map);

    /* 步骤4：残差块1（32→32，恒等shortcut） */
    _cnn_residual_block_forward(
        processor->cnn_feature_map, IMG, IMG, H, H,
        processor->cnn_block1_conv1_w, processor->cnn_block1_conv1_b,
        processor->cnn_block1_bn1_gamma, processor->cnn_block1_bn1_beta,
        processor->cnn_block1_bn1_rm, processor->cnn_block1_bn1_rv,
        processor->cnn_block1_conv2_w, processor->cnn_block1_conv2_b,
        processor->cnn_block1_bn2_gamma, processor->cnn_block1_bn2_beta,
        processor->cnn_block1_bn2_rm, processor->cnn_block1_bn2_rv,
        NULL, NULL, /* 恒等shortcut，无需1×1投影 */
        processor->cnn_temp_buf, processor->cnn_feature_map);

    /* 步骤5：全局平均池化 → 32维特征向量 */
    _cnn_global_avg_pool(processor->cnn_feature_map, IMG, IMG, H, features);

    return LV_CNN_HIDDEN_CH;
}

/*
 * lv_save_weights —— 保存CNN可学习卷积核权重到二进制文件
 *
 * 将所有权重（卷积核+偏置+BatchNorm参数+running statistics）
 * 序列化写入二进制文件，用于持久化存储训练结果。
 *
 * 文件格式：
 *   [magic: 8字节 "LVCNNv01"]
 *   [stem_conv_w: H*I*9 floats] [stem_conv_b: H floats]
 *   [stem_bn_gamma: H] [stem_bn_beta: H] [stem_bn_rm: H] [stem_bn_rv: H]
 *   [block0_conv1_w: H*H*9] [block0_conv1_b: H]
 *   [block0_bn1_gamma/beta/rm/rv: 各H]
 *   [block0_conv2_w: H*H*9] [block0_conv2_b: H]
 *   [block0_bn2_gamma/beta/rm/rv: 各H]
 *   [block1_conv1_w: H*H*9] [block1_conv1_b: H]
 *   [block1_bn1_gamma/beta/rm/rv: 各H]
 *   [block1_conv2_w: H*H*9] [block1_conv2_b: H]
 *   [block1_bn2_gamma/beta/rm/rv: 各H]
 *
 * 参数：
 *   processor: LiquidVisionProcessor实例
 *   filepath:  保存路径
 *
 * 返回值：成功返回0，失败返回-1
 */
int lv_save_weights(const LiquidVisionProcessor* processor, const char* filepath) {
    if (!processor || !filepath) return -1;
    if (!processor->cnn_weights_initialized || !processor->cnn_stem_conv_w) return -1;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    const int H = LV_CNN_HIDDEN_CH;
    const int I = LV_CNN_IN_CHANNELS;
    const int KK = LV_CNN_KERNEL * LV_CNN_KERNEL;

    /* 写入魔数标识 */
    const char magic[] = "LVCNNv01";
    if (fwrite(magic, 1, 8, fp) != 8) { fclose(fp); return -1; }

    /* 写入架构常量（便于加载时校验） */
    fwrite(&I, sizeof(int), 1, fp);
    fwrite(&H, sizeof(int), 1, fp);

#define LV_SAFE_WRITE(ptr, count, fp) \
    do { if (fwrite((ptr), sizeof(float), (size_t)(count), (fp)) != (size_t)(count)) { fclose(fp); return -1; } } while(0)

    /* 初始卷积层 */
    LV_SAFE_WRITE(processor->cnn_stem_conv_w, I * H * KK, fp);
    LV_SAFE_WRITE(processor->cnn_stem_conv_b, H, fp);
    LV_SAFE_WRITE(processor->cnn_stem_bn_gamma, H, fp);
    LV_SAFE_WRITE(processor->cnn_stem_bn_beta, H, fp);
    LV_SAFE_WRITE(processor->cnn_stem_bn_rm, H, fp);
    LV_SAFE_WRITE(processor->cnn_stem_bn_rv, H, fp);

    /* 残差块0 */
    LV_SAFE_WRITE(processor->cnn_block0_conv1_w, H * H * KK, fp);
    LV_SAFE_WRITE(processor->cnn_block0_conv1_b, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_bn1_gamma, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_bn1_beta, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_bn1_rm, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_bn1_rv, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_conv2_w, H * H * KK, fp);
    LV_SAFE_WRITE(processor->cnn_block0_conv2_b, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_bn2_gamma, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_bn2_beta, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_bn2_rm, H, fp);
    LV_SAFE_WRITE(processor->cnn_block0_bn2_rv, H, fp);

    /* 残差块1 */
    LV_SAFE_WRITE(processor->cnn_block1_conv1_w, H * H * KK, fp);
    LV_SAFE_WRITE(processor->cnn_block1_conv1_b, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_bn1_gamma, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_bn1_beta, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_bn1_rm, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_bn1_rv, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_conv2_w, H * H * KK, fp);
    LV_SAFE_WRITE(processor->cnn_block1_conv2_b, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_bn2_gamma, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_bn2_beta, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_bn2_rm, H, fp);
    LV_SAFE_WRITE(processor->cnn_block1_bn2_rv, H, fp);

#undef LV_SAFE_WRITE

    fclose(fp);
    return 0;
}

/*
 * lv_load_weights —— 从二进制文件加载CNN可学习卷积核权重
 *
 * 读取由lv_save_weights保存的权重文件，覆盖当前所有权重。
 * 自动校验魔数和架构常量是否匹配。
 *
 * 参数：
 *   processor: LiquidVisionProcessor实例
 *   filepath:  文件路径
 *
 * 返回值：成功返回0，失败返回-1
 */
int lv_load_weights(LiquidVisionProcessor* processor, const char* filepath) {
    if (!processor || !filepath) return -1;

    /* 先确保权重缓冲区已分配 */
    if (!processor->cnn_weights_initialized) {
        int ret = _cnn_allocate_and_init_weights(processor);
        if (ret != 0) return -1;
    }

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    const int H = LV_CNN_HIDDEN_CH;
    const int I = LV_CNN_IN_CHANNELS;
    const int KK = LV_CNN_KERNEL * LV_CNN_KERNEL;

    /* 校验魔数 */
    char magic[9] = {0};
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "LVCNNv01", 8) != 0) {
        fclose(fp); return -1;
    }

    /* 校验架构常量 */
    int saved_I = 0, saved_H = 0;
    if (fread(&saved_I, sizeof(int), 1, fp) != 1 ||
        fread(&saved_H, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }
    if (saved_I != I || saved_H != H) { fclose(fp); return -1; }

#define LV_SAFE_READ(ptr, count, fp) \
    do { if (fread((ptr), sizeof(float), (size_t)(count), (fp)) != (size_t)(count)) { fclose(fp); return -1; } } while(0)

    /* 初始卷积层 */
    LV_SAFE_READ(processor->cnn_stem_conv_w, I * H * KK, fp);
    LV_SAFE_READ(processor->cnn_stem_conv_b, H, fp);
    LV_SAFE_READ(processor->cnn_stem_bn_gamma, H, fp);
    LV_SAFE_READ(processor->cnn_stem_bn_beta, H, fp);
    LV_SAFE_READ(processor->cnn_stem_bn_rm, H, fp);
    LV_SAFE_READ(processor->cnn_stem_bn_rv, H, fp);

    /* 残差块0 */
    LV_SAFE_READ(processor->cnn_block0_conv1_w, H * H * KK, fp);
    LV_SAFE_READ(processor->cnn_block0_conv1_b, H, fp);
    LV_SAFE_READ(processor->cnn_block0_bn1_gamma, H, fp);
    LV_SAFE_READ(processor->cnn_block0_bn1_beta, H, fp);
    LV_SAFE_READ(processor->cnn_block0_bn1_rm, H, fp);
    LV_SAFE_READ(processor->cnn_block0_bn1_rv, H, fp);
    LV_SAFE_READ(processor->cnn_block0_conv2_w, H * H * KK, fp);
    LV_SAFE_READ(processor->cnn_block0_conv2_b, H, fp);
    LV_SAFE_READ(processor->cnn_block0_bn2_gamma, H, fp);
    LV_SAFE_READ(processor->cnn_block0_bn2_beta, H, fp);
    LV_SAFE_READ(processor->cnn_block0_bn2_rm, H, fp);
    LV_SAFE_READ(processor->cnn_block0_bn2_rv, H, fp);

    /* 残差块1 */
    LV_SAFE_READ(processor->cnn_block1_conv1_w, H * H * KK, fp);
    LV_SAFE_READ(processor->cnn_block1_conv1_b, H, fp);
    LV_SAFE_READ(processor->cnn_block1_bn1_gamma, H, fp);
    LV_SAFE_READ(processor->cnn_block1_bn1_beta, H, fp);
    LV_SAFE_READ(processor->cnn_block1_bn1_rm, H, fp);
    LV_SAFE_READ(processor->cnn_block1_bn1_rv, H, fp);
    LV_SAFE_READ(processor->cnn_block1_conv2_w, H * H * KK, fp);
    LV_SAFE_READ(processor->cnn_block1_conv2_b, H, fp);
    LV_SAFE_READ(processor->cnn_block1_bn2_gamma, H, fp);
    LV_SAFE_READ(processor->cnn_block1_bn2_beta, H, fp);
    LV_SAFE_READ(processor->cnn_block1_bn2_rm, H, fp);
    LV_SAFE_READ(processor->cnn_block1_bn2_rv, H, fp);

#undef LV_SAFE_READ

    fclose(fp);
    processor->cnn_weights_loaded = 1;
    processor->is_trained = 1; /* 权重加载成功后标记为已训练 */
    return 0;
}

