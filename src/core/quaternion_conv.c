/**
 * @file quaternion_conv.c
 * @brief 四元数卷积层实现
 *
 * 提供四元数1D/2D卷积的完整算法实现，
 * 包括前向传播、反向传播、Adam优化、序列化。
 *
 * 核心算法：
 *   四元数卷积 = Σ (输入四元数 × 卷积核四元数)
 *   其中 × 为Hamilton乘积:
 *   (w1,x1,y1,z1)×(w2,x2,y2,z2) =
 *     w = w1*w2 - x1*x2 - y1*y2 - z1*z2
 *     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
 *     y = w1*y2 - x1*z2 + y1*w2 + z1*x2
 *     z = w1*z2 + x1*y2 - y1*x2 + z1*w2
 *
 * 梯度传播：
 *   ∂(a×b)/∂a = M_plus(b)^T, ∂(a×b)/∂b = M_plus(a)^T
 */

#include "selflnn/core/quaternion_conv.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ============================
 *   内部辅助函数
 * ============================ */

/**
 * @brief Hamilton梯度：输入梯度
/* 修复输入梯度公式。
 * 前向: r = input ⊗ weight (Hamilton积, input在左, weight在右)
 * 正确梯度: ∂L/∂input = M_R(weight)^T · grad_r
 * M_R(b)^T = [  b0   b1   b2   b3 ]
 *            [ -b1   b0  -b3   b2 ]
 *            [ -b2   b3   b0  -b1 ]
 *            [ -b3  -b2   b1   b0 ]
 * 原作错误使用了 M_L(b)^T, 6个off-diagonal项符号相反。 */
static void quat_grad_input(const float* b, const float* grad_r, float* grad_a)
{
    grad_a[0] = b[0] * grad_r[0] + b[1] * grad_r[1] + b[2] * grad_r[2] + b[3] * grad_r[3];
    grad_a[1] = -b[1] * grad_r[0] + b[0] * grad_r[1] - b[3] * grad_r[2] + b[2] * grad_r[3];
    grad_a[2] = -b[2] * grad_r[0] + b[3] * grad_r[1] + b[0] * grad_r[2] - b[1] * grad_r[3];
    grad_a[3] = -b[3] * grad_r[0] - b[2] * grad_r[1] + b[1] * grad_r[2] + b[0] * grad_r[3];
}

/**
 * @brief Hamilton梯度：权重梯度
 * grad_w = M_plus(a)^T * grad_r
 */
static void quat_grad_weight(const float* a, const float* grad_r, float* grad_w)
{
    grad_w[0] = a[0] * grad_r[0] + a[1] * grad_r[1] + a[2] * grad_r[2] + a[3] * grad_r[3];
    grad_w[1] = -a[1] * grad_r[0] + a[0] * grad_r[1] + a[3] * grad_r[2] - a[2] * grad_r[3];
    grad_w[2] = -a[2] * grad_r[0] - a[3] * grad_r[1] + a[0] * grad_r[2] + a[1] * grad_r[3];
    grad_w[3] = -a[3] * grad_r[0] + a[2] * grad_r[1] - a[1] * grad_r[2] + a[0] * grad_r[3];
}

/**
 * @brief Xavier初始化：均匀分布初始化四元数权重
 */
/* K-012修复：使用加密安全随机数替代rand() */
static void quat_xavier_init(float* data, size_t count, float scale)
{
    for (size_t i = 0; i < count; i++) {
        float val = secure_random_float() * 2.0f - 1.0f;
        data[i] = val * scale;
    }
}

/**
 * @brief Adam更新单参数
 */
static void adam_update(float* param, float* m, float* v, float grad,
                        float lr, float beta1, float beta2, float eps, float corr1, float corr2)
{
    m[0] = beta1 * m[0] + (1.0f - beta1) * grad;
    v[0] = beta2 * v[0] + (1.0f - beta2) * grad * grad;
    float m_hat = m[0] / corr1;
    float v_hat = v[0] / corr2;
    param[0] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

/* ============================
 *   四元数1D卷积实现
 * ============================ */

struct QuaternionConv1D {
    QuaternionConv1DConfig config;
    float* weights;          /* [out_channels × in_channels × kernel_size × 4] */
    float* bias;             /* [out_channels × 4] */
    float* weight_grad;      /* 与weights同维度 */
    float* bias_grad;        /* 与bias同维度 */
    float* weight_m;         /* Adam一阶矩 */
    float* weight_v;         /* Adam二阶矩 */
    float* bias_m;
    float* bias_v;
    size_t adam_step;        /* Adam步数 */
    float* workspace;        /* 工作空间 */
};

QuaternionConv1D* quaternion_conv1d_create(const QuaternionConv1DConfig* config)
{
    if (!config) return NULL;
    QuaternionConv1D* layer = (QuaternionConv1D*)safe_calloc(1, sizeof(QuaternionConv1D));
    if (!layer) return NULL;

    layer->config = *config;
    if (config->init_scale <= 0.0f) layer->config.init_scale = 0.1f;

    layer->config.weight_decay = config->weight_decay;
    if (config->lr <= 0.0f) layer->config.lr = 0.001f;
    if (config->kernel_size == 0) layer->config.kernel_size = 3;
    if (config->in_channels == 0 || config->out_channels == 0) {
        safe_free((void**)&layer);
        return NULL;
    }

    size_t wcount = config->out_channels * config->in_channels * config->kernel_size * 4;
    size_t bcount = config->out_channels * 4;

    layer->weights = (float*)safe_calloc(wcount, sizeof(float));
    layer->bias = (float*)safe_calloc(config->use_bias ? bcount : 1, sizeof(float));
    layer->weight_grad = (float*)safe_calloc(wcount, sizeof(float));
    layer->bias_grad = (float*)safe_calloc(config->use_bias ? bcount : 1, sizeof(float));
    layer->weight_m = (float*)safe_calloc(wcount, sizeof(float));
    layer->weight_v = (float*)safe_calloc(wcount, sizeof(float));
    layer->bias_m = (float*)safe_calloc(config->use_bias ? bcount : 1, sizeof(float));
    layer->bias_v = (float*)safe_calloc(config->use_bias ? bcount : 1, sizeof(float));

    if (!layer->weights || !layer->bias || !layer->weight_grad || !layer->bias_grad ||
        !layer->weight_m || !layer->weight_v || !layer->bias_m || !layer->bias_v) {
        quaternion_conv1d_free(layer);
        return NULL;
    }

    float init_scale_val = sqrtf(6.0f / (float)(config->in_channels + config->out_channels)) * config->init_scale;
    quat_xavier_init(layer->weights, wcount, init_scale_val);
    if (config->use_bias) {
        for (size_t i = 0; i < bcount; i++) layer->bias[i] = 0.0f;
    }

    layer->adam_step = 0;
    layer->workspace = NULL;
    return layer;
}

void quaternion_conv1d_free(QuaternionConv1D* layer)
{
    if (!layer) return;
    safe_free((void**)&layer->weights);
    safe_free((void**)&layer->bias);
    safe_free((void**)&layer->weight_grad);
    safe_free((void**)&layer->bias_grad);
    safe_free((void**)&layer->weight_m);
    safe_free((void**)&layer->weight_v);
    safe_free((void**)&layer->bias_m);
    safe_free((void**)&layer->bias_v);
    safe_free((void**)&layer->workspace);
    safe_free((void**)&layer);
}

int quaternion_conv1d_forward(QuaternionConv1D* layer,
                               const float* input,
                               size_t batch_size,
                               size_t seq_len,
                               float* output)
{
    if (!layer || !input || !output) return -1;

    const QuaternionConv1DConfig* cfg = &layer->config;

    /* 计算输出长度: floor((seq_len + 2*pad - kernel) / stride) + 1 */
    size_t padded_len = seq_len + 2 * cfg->padding;
    size_t output_len;
    if (padded_len < cfg->kernel_size) {
        output_len = 1;
    } else {
        output_len = (padded_len - cfg->kernel_size) / cfg->stride + 1;
    }

    size_t oc = cfg->out_channels;
    size_t ic = cfg->in_channels;
    size_t ks = cfg->kernel_size;
    size_t stride = cfg->stride;

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t oi = 0; oi < oc; oi++) {
            for (size_t ol = 0; ol < output_len; ol++) {
                size_t out_idx = ((b * oc + oi) * output_len + ol) * 4;
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (size_t ii = 0; ii < ic; ii++) {
                    for (size_t k = 0; k < ks; k++) {
/* 修正padding处理
                         * 原作: seq_start = ol*stride, 仅检查pos>=seq_len上界, padding完全未实现
                         * 修正: 考虑padding左偏移, 同时检查下界和上界 */
                        int pos = (int)(ol * stride) - (int)(cfg->padding);
                        pos += (int)k;
                        if (pos < 0 || (size_t)pos >= seq_len) {
                            continue;
                        }

                        size_t in_idx = ((b * ic + ii) * seq_len + (size_t)pos) * 4;
                        size_t w_idx = ((oi * ic + ii) * ks + k) * 4;

                        float temp[4];
                        quaternion_hamilton_product(&input[in_idx], &layer->weights[w_idx], temp);
                        quaternion_add_arr(acc, temp, acc);
                    }
                }

                if (cfg->use_bias) {
                    acc[0] += layer->bias[oi * 4 + 0];
                    acc[1] += layer->bias[oi * 4 + 1];
                    acc[2] += layer->bias[oi * 4 + 2];
                    acc[3] += layer->bias[oi * 4 + 3];
                }

                output[out_idx + 0] = acc[0];
                output[out_idx + 1] = acc[1];
                output[out_idx + 2] = acc[2];
                output[out_idx + 3] = acc[3];
            }
        }
    }
    return 0;
}

int quaternion_conv1d_backward(QuaternionConv1D* layer,
                                const float* output_grad,
                                const float* input,
                                size_t batch_size,
                                size_t seq_len,
                                float* input_grad)
{
    if (!layer || !output_grad || !input) return -1;

    const QuaternionConv1DConfig* cfg = &layer->config;
    size_t padded_len = seq_len + 2 * cfg->padding;
    size_t output_len;
    if (padded_len < cfg->kernel_size) {
        output_len = 1;
    } else {
        output_len = (padded_len - cfg->kernel_size) / cfg->stride + 1;
    }

    size_t oc = cfg->out_channels;
    size_t ic = cfg->in_channels;
    size_t ks = cfg->kernel_size;
    size_t stride = cfg->stride;

    /* 清零梯度缓冲区 */
    memset(layer->weight_grad, 0, oc * ic * ks * 4 * sizeof(float));
    memset(layer->bias_grad, 0, oc * 4 * sizeof(float));

    /* 输入梯度清零（如果提供） */
    if (input_grad) {
        memset(input_grad, 0, batch_size * ic * seq_len * 4 * sizeof(float));
    }

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t oi = 0; oi < oc; oi++) {
            for (size_t ol = 0; ol < output_len; ol++) {
                size_t out_idx = ((b * oc + oi) * output_len + ol) * 4;
                const float* grad = &output_grad[out_idx];

                /* 偏置梯度 */
                if (cfg->use_bias) {
                    layer->bias_grad[oi * 4 + 0] += grad[0];
                    layer->bias_grad[oi * 4 + 1] += grad[1];
                    layer->bias_grad[oi * 4 + 2] += grad[2];
                    layer->bias_grad[oi * 4 + 3] += grad[3];
                }

                for (size_t ii = 0; ii < ic; ii++) {
                    for (size_t k = 0; k < ks; k++) {
                        size_t seq_start = ol * stride;
                        size_t pos = seq_start + k;
                        if (pos >= seq_len) continue;

                        size_t in_idx = ((b * ic + ii) * seq_len + pos) * 4;
                        size_t w_idx = ((oi * ic + ii) * ks + k) * 4;
                        float temp_grad[4];

                        /* 权重梯度: ∂loss/∂w = M_plus(input)^T * grad */
                        quat_grad_weight(&input[in_idx], grad, temp_grad);
                        for (int c = 0; c < 4; c++) {
                            layer->weight_grad[w_idx + c] += temp_grad[c];
                        }

                        /* 输入梯度: ∂loss/∂input = M_plus(weight)^T * grad */
                        if (input_grad) {
                            quat_grad_input(&layer->weights[w_idx], grad, temp_grad);
                            for (int c = 0; c < 4; c++) {
                                input_grad[in_idx + c] += temp_grad[c];
                            }
                        }
                    }
                }
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
    float lr = cfg->lr;
    float wd = cfg->weight_decay;
    size_t wcount = oc * ic * ks * 4;

    for (size_t i = 0; i < wcount; i++) {
        float g = layer->weight_grad[i] / (float)batch_size + wd * layer->weights[i];
        adam_update(&layer->weights[i], &layer->weight_m[i], &layer->weight_v[i],
                    g, lr, beta1, beta2, eps, corr1, corr2);
    }

    if (cfg->use_bias) {
        for (size_t i = 0; i < oc * 4; i++) {
            float g = layer->bias_grad[i] / (float)batch_size;
            adam_update(&layer->bias[i], &layer->bias_m[i], &layer->bias_v[i],
                        g, lr, beta1, beta2, eps, corr1, corr2);
        }
    }

    return 0;
}

void quaternion_conv1d_reset_optimizer(QuaternionConv1D* layer)
{
    if (!layer) return;
    const QuaternionConv1DConfig* cfg = &layer->config;
    size_t wcount = cfg->out_channels * cfg->in_channels * cfg->kernel_size * 4;
    size_t bcount = cfg->out_channels * 4;
    memset(layer->weight_m, 0, wcount * sizeof(float));
    memset(layer->weight_v, 0, wcount * sizeof(float));
    memset(layer->bias_m, 0, (cfg->use_bias ? bcount : 1) * sizeof(float));
    memset(layer->bias_v, 0, (cfg->use_bias ? bcount : 1) * sizeof(float));
    layer->adam_step = 0;
}

void quaternion_conv1d_set_lr(QuaternionConv1D* layer, float lr)
{
    if (layer) layer->config.lr = lr;
}

int quaternion_conv1d_param_count(const QuaternionConv1D* layer)
{
    if (!layer) return -1;
    const QuaternionConv1DConfig* cfg = &layer->config;
    size_t count = cfg->out_channels * cfg->in_channels * cfg->kernel_size * 4;
    if (cfg->use_bias) count += cfg->out_channels * 4;
    return (int)count;
}

int quaternion_conv1d_save(const QuaternionConv1D* layer, const char* filename)
{
    if (!layer || !filename) return -1;
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    fwrite(&layer->config, sizeof(QuaternionConv1DConfig), 1, f);
    size_t wcount = layer->config.out_channels * layer->config.in_channels *
                    layer->config.kernel_size * 4;
    size_t bcount = layer->config.out_channels * 4;
    fwrite(layer->weights, sizeof(float), wcount, f);
    fwrite(layer->bias, sizeof(float), layer->config.use_bias ? bcount : 1, f);
    fclose(f);
    return 0;
}

QuaternionConv1D* quaternion_conv1d_load(const char* filename)
{
    if (!filename) return NULL;
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    QuaternionConv1DConfig cfg;
    if (fread(&cfg, sizeof(QuaternionConv1DConfig), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    QuaternionConv1D* layer = quaternion_conv1d_create(&cfg);
    if (!layer) { fclose(f); return NULL; }

    size_t wcount = cfg.out_channels * cfg.in_channels * cfg.kernel_size * 4;
    size_t bcount = cfg.out_channels * 4;
    size_t bread = fread(layer->weights, sizeof(float), wcount, f);
    size_t bread_b = fread(layer->bias, sizeof(float), cfg.use_bias ? bcount : 1, f);
    fclose(f);

    if (bread != wcount || (cfg.use_bias && bread_b != bcount)) {
        quaternion_conv1d_free(layer);
        return NULL;
    }
    return layer;
}

/* ============================
 *   四元数2D卷积实现
 * ============================ */

struct QuaternionConv2D {
    QuaternionConv2DConfig config;
    float* weights;          /* [out_channels × in_channels × kernel_h × kernel_w × 4] */
    float* bias;             /* [out_channels × 4] */
    float* weight_grad;
    float* bias_grad;
    float* weight_m;
    float* weight_v;
    float* bias_m;
    float* bias_v;
    size_t adam_step;
};

QuaternionConv2D* quaternion_conv2d_create(const QuaternionConv2DConfig* config)
{
    if (!config) return NULL;
    QuaternionConv2D* layer = (QuaternionConv2D*)safe_calloc(1, sizeof(QuaternionConv2D));
    if (!layer) return NULL;

    layer->config = *config;
    if (config->init_scale <= 0.0f) layer->config.init_scale = 0.1f;
    layer->config.weight_decay = config->weight_decay;
    if (config->lr <= 0.0f) layer->config.lr = 0.001f;
    if (config->kernel_h == 0 || config->kernel_w == 0) {
        layer->config.kernel_h = 3; layer->config.kernel_w = 3;
    }
    if (config->in_channels == 0 || config->out_channels == 0) {
        safe_free((void**)&layer);
        return NULL;
    }

    size_t wcount = config->out_channels * config->in_channels *
                    config->kernel_h * config->kernel_w * 4;
    size_t bcount = config->out_channels * 4;

    layer->weights = (float*)safe_calloc(wcount, sizeof(float));
    layer->bias = (float*)safe_calloc(config->use_bias ? bcount : 1, sizeof(float));
    layer->weight_grad = (float*)safe_calloc(wcount, sizeof(float));
    layer->bias_grad = (float*)safe_calloc(config->use_bias ? bcount : 1, sizeof(float));
    layer->weight_m = (float*)safe_calloc(wcount, sizeof(float));
    layer->weight_v = (float*)safe_calloc(wcount, sizeof(float));
    layer->bias_m = (float*)safe_calloc(config->use_bias ? bcount : 1, sizeof(float));
    layer->bias_v = (float*)safe_calloc(config->use_bias ? bcount : 1, sizeof(float));

    if (!layer->weights || !layer->bias || !layer->weight_grad || !layer->bias_grad ||
        !layer->weight_m || !layer->weight_v || !layer->bias_m || !layer->bias_v) {
        quaternion_conv2d_free(layer);
        return NULL;
    }

    float init_scale_val = sqrtf(6.0f / (float)(config->in_channels + config->out_channels)) * config->init_scale;
    quat_xavier_init(layer->weights, wcount, init_scale_val);
    layer->adam_step = 0;
    return layer;
}

void quaternion_conv2d_free(QuaternionConv2D* layer)
{
    if (!layer) return;
    safe_free((void**)&layer->weights);
    safe_free((void**)&layer->bias);
    safe_free((void**)&layer->weight_grad);
    safe_free((void**)&layer->bias_grad);
    safe_free((void**)&layer->weight_m);
    safe_free((void**)&layer->weight_v);
    safe_free((void**)&layer->bias_m);
    safe_free((void**)&layer->bias_v);
    safe_free((void**)&layer);
}

int quaternion_conv2d_forward(QuaternionConv2D* layer,
                               const float* input,
                               size_t batch_size,
                               size_t height,
                               size_t width,
                               float* output)
{
    if (!layer || !input || !output) return -1;

    const QuaternionConv2DConfig* cfg = &layer->config;
    size_t out_h = (height + 2 * cfg->pad_h - cfg->kernel_h) / cfg->stride_h + 1;
    size_t out_w = (width + 2 * cfg->pad_w - cfg->kernel_w) / cfg->stride_w + 1;

    size_t oc = cfg->out_channels;
    size_t ic = cfg->in_channels;
    size_t kh = cfg->kernel_h;
    size_t kw = cfg->kernel_w;
    size_t sh = cfg->stride_h;
    size_t sw = cfg->stride_w;

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t oi = 0; oi < oc; oi++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    size_t out_idx = (((b * oc + oi) * out_h + oh) * out_w + ow) * 4;
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                    for (size_t ii = 0; ii < ic; ii++) {
                        for (size_t kh_pos = 0; kh_pos < kh; kh_pos++) {
                            size_t base_h = oh * sh + kh_pos;
                            if (base_h < (size_t)cfg->pad_h) continue;
                            size_t ih = base_h - (size_t)cfg->pad_h;
                            if (ih >= height) continue;
                            for (size_t kw_pos = 0; kw_pos < kw; kw_pos++) {
                                size_t base_w = ow * sw + kw_pos;
                                if (base_w < (size_t)cfg->pad_w) continue;
                                size_t iw = base_w - (size_t)cfg->pad_w;
                                if (iw >= width) continue;

                                size_t in_idx = (((b * ic + ii) * height + ih) * width + iw) * 4;
                                size_t w_idx = (((oi * ic + ii) * kh + kh_pos) * kw + kw_pos) * 4;

                                float temp[4];
                                quaternion_hamilton_product(&input[in_idx], &layer->weights[w_idx], temp);
                                quaternion_add_arr(acc, temp, acc);
                            }
                        }
                    }

                    if (cfg->use_bias) {
                        acc[0] += layer->bias[oi * 4 + 0];
                        acc[1] += layer->bias[oi * 4 + 1];
                        acc[2] += layer->bias[oi * 4 + 2];
                        acc[3] += layer->bias[oi * 4 + 3];
                    }

                    output[out_idx + 0] = acc[0];
                    output[out_idx + 1] = acc[1];
                    output[out_idx + 2] = acc[2];
                    output[out_idx + 3] = acc[3];
                }
            }
        }
    }
    return 0;
}

int quaternion_conv2d_backward(QuaternionConv2D* layer,
                                const float* output_grad,
                                const float* input,
                                size_t batch_size,
                                size_t height,
                                size_t width,
                                float* input_grad)
{
    if (!layer || !output_grad || !input) return -1;

    const QuaternionConv2DConfig* cfg = &layer->config;
    size_t out_h = (height + 2 * cfg->pad_h - cfg->kernel_h) / cfg->stride_h + 1;
    size_t out_w = (width + 2 * cfg->pad_w - cfg->kernel_w) / cfg->stride_w + 1;

    size_t oc = cfg->out_channels;
    size_t ic = cfg->in_channels;
    size_t kh = cfg->kernel_h;
    size_t kw = cfg->kernel_w;
    size_t sh = cfg->stride_h;
    size_t sw = cfg->stride_w;

    size_t wcount = oc * ic * kh * kw * 4;
    memset(layer->weight_grad, 0, wcount * sizeof(float));
    memset(layer->bias_grad, 0, oc * 4 * sizeof(float));

    if (input_grad) {
        memset(input_grad, 0, batch_size * ic * height * width * 4 * sizeof(float));
    }

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t oi = 0; oi < oc; oi++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    size_t out_idx = (((b * oc + oi) * out_h + oh) * out_w + ow) * 4;
                    const float* grad = &output_grad[out_idx];

                    if (cfg->use_bias) {
                        layer->bias_grad[oi * 4 + 0] += grad[0];
                        layer->bias_grad[oi * 4 + 1] += grad[1];
                        layer->bias_grad[oi * 4 + 2] += grad[2];
                        layer->bias_grad[oi * 4 + 3] += grad[3];
                    }

                    for (size_t ii = 0; ii < ic; ii++) {
                        for (size_t kh_pos = 0; kh_pos < kh; kh_pos++) {
                            size_t base_h = oh * sh + kh_pos;
                            if (base_h < (size_t)cfg->pad_h) continue;
                            size_t ih = base_h - (size_t)cfg->pad_h;
                            if (ih >= height) continue;
                            for (size_t kw_pos = 0; kw_pos < kw; kw_pos++) {
                                size_t base_w = ow * sw + kw_pos;
                                if (base_w < (size_t)cfg->pad_w) continue;
                                size_t iw = base_w - (size_t)cfg->pad_w;
                                if (iw >= width) continue;

                                size_t in_idx = (((b * ic + ii) * height + ih) * width + iw) * 4;
                                size_t w_idx = (((oi * ic + ii) * kh + kh_pos) * kw + kw_pos) * 4;
                                float temp_grad[4];

                                quat_grad_weight(&input[in_idx], grad, temp_grad);
                                for (int c = 0; c < 4; c++) {
                                    layer->weight_grad[w_idx + c] += temp_grad[c];
                                }

                                if (input_grad) {
                                    quat_grad_input(&layer->weights[w_idx], grad, temp_grad);
                                    for (int c = 0; c < 4; c++) {
                                        input_grad[in_idx + c] += temp_grad[c];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* Adam更新 */
    layer->adam_step++;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float corr1 = 1.0f - powf(beta1, (float)layer->adam_step);
    float corr2 = 1.0f - powf(beta2, (float)layer->adam_step);
    float lr = cfg->lr;
    float wd = cfg->weight_decay;

    for (size_t i = 0; i < wcount; i++) {
        float g = layer->weight_grad[i] / (float)batch_size + wd * layer->weights[i];
        adam_update(&layer->weights[i], &layer->weight_m[i], &layer->weight_v[i],
                    g, lr, beta1, beta2, eps, corr1, corr2);
    }

    if (cfg->use_bias) {
        for (size_t i = 0; i < oc * 4; i++) {
            float g = layer->bias_grad[i] / (float)batch_size;
            adam_update(&layer->bias[i], &layer->bias_m[i], &layer->bias_v[i],
                        g, lr, beta1, beta2, eps, corr1, corr2);
        }
    }

    return 0;
}

void quaternion_conv2d_reset_optimizer(QuaternionConv2D* layer)
{
    if (!layer) return;
    const QuaternionConv2DConfig* cfg = &layer->config;
    size_t wcount = cfg->out_channels * cfg->in_channels * cfg->kernel_h * cfg->kernel_w * 4;
    size_t bcount = cfg->out_channels * 4;
    memset(layer->weight_m, 0, wcount * sizeof(float));
    memset(layer->weight_v, 0, wcount * sizeof(float));
    memset(layer->bias_m, 0, (cfg->use_bias ? bcount : 1) * sizeof(float));
    memset(layer->bias_v, 0, (cfg->use_bias ? bcount : 1) * sizeof(float));
    layer->adam_step = 0;
}

void quaternion_conv2d_set_lr(QuaternionConv2D* layer, float lr)
{
    if (layer) layer->config.lr = lr;
}

int quaternion_conv2d_param_count(const QuaternionConv2D* layer)
{
    if (!layer) return -1;
    const QuaternionConv2DConfig* cfg = &layer->config;
    size_t count = cfg->out_channels * cfg->in_channels * cfg->kernel_h * cfg->kernel_w * 4;
    if (cfg->use_bias) count += cfg->out_channels * 4;
    return (int)count;
}

int quaternion_conv2d_save(const QuaternionConv2D* layer, const char* filename)
{
    if (!layer || !filename) return -1;
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    fwrite(&layer->config, sizeof(QuaternionConv2DConfig), 1, f);
    size_t wcount = layer->config.out_channels * layer->config.in_channels *
                    layer->config.kernel_h * layer->config.kernel_w * 4;
    size_t bcount = layer->config.out_channels * 4;
    fwrite(layer->weights, sizeof(float), wcount, f);
    fwrite(layer->bias, sizeof(float), layer->config.use_bias ? bcount : 1, f);
    fclose(f);
    return 0;
}

QuaternionConv2D* quaternion_conv2d_load(const char* filename)
{
    if (!filename) return NULL;
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    QuaternionConv2DConfig cfg;
    if (fread(&cfg, sizeof(QuaternionConv2DConfig), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    QuaternionConv2D* layer = quaternion_conv2d_create(&cfg);
    if (!layer) { fclose(f); return NULL; }

    size_t wcount = cfg.out_channels * cfg.in_channels * cfg.kernel_h * cfg.kernel_w * 4;
    size_t bcount = cfg.out_channels * 4;
    size_t bread = fread(layer->weights, sizeof(float), wcount, f);
    size_t bread_b = fread(layer->bias, sizeof(float), cfg.use_bias ? bcount : 1, f);
    fclose(f);

    if (bread != wcount || (cfg.use_bias && bread_b != bcount)) {
        quaternion_conv2d_free(layer);
        return NULL;
    }
    return layer;
}
