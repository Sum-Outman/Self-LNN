/**
 * @file quaternion_enhanced.c
 * @brief 四元数增强功能实现（Dropout + LayerNorm + 注意力机制）
 *
 * M-006修复: 实现四元数空间的Dropout和LayerNorm。
 * 所有inline基础运算在 quaternion_enhanced.h 中定义。
 */

#include "selflnn/core/quaternion_enhanced.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * ============================================================
 * 四元数 Dropout
 * ============================================================
 */

int quat_dropout_forward(const Quaternion* input, size_t num_quats,
                         float dropout_rate, unsigned int seed,
                         Quaternion* output, int* mask) {
    if (!input || !output || num_quats == 0) return -1;
    if (dropout_rate < 0.0f) dropout_rate = 0.0f;
    if (dropout_rate > 1.0f) dropout_rate = 1.0f;

    float scale = 1.0f / (1.0f - dropout_rate + 1e-10f);
    unsigned int rng = seed;

    for (size_t i = 0; i < num_quats; i++) {
        rng = rng * 1103515245u + 12345u;
        float r = (float)(rng & 0x7FFFFFFF) / 2147483648.0f;
        int keep = (r > dropout_rate) ? 1 : 0;

        if (mask) mask[i] = keep;
        if (keep) {
            output[i].w = input[i].w * scale;
            output[i].x = input[i].x * scale;
            output[i].y = input[i].y * scale;
            output[i].z = input[i].z * scale;
        } else {
            output[i].w = 0.0f;
            output[i].x = 0.0f;
            output[i].y = 0.0f;
            output[i].z = 0.0f;
        }
    }
    return 0;
}

int quat_dropout_backward(const Quaternion* output_grad, const int* mask,
                          size_t num_quats, float dropout_rate,
                          Quaternion* input_grad) {
    if (!output_grad || !mask || !input_grad || num_quats == 0) return -1;
    if (dropout_rate < 0.0f) dropout_rate = 0.0f;
    if (dropout_rate >= 1.0f) return -1;

    float scale = 1.0f / (1.0f - dropout_rate + 1e-10f);

    for (size_t i = 0; i < num_quats; i++) {
        if (mask[i]) {
            input_grad[i].w = output_grad[i].w * scale;
            input_grad[i].x = output_grad[i].x * scale;
            input_grad[i].y = output_grad[i].y * scale;
            input_grad[i].z = output_grad[i].z * scale;
        } else {
            input_grad[i].w = 0.0f;
            input_grad[i].x = 0.0f;
            input_grad[i].y = 0.0f;
            input_grad[i].z = 0.0f;
        }
    }
    return 0;
}

/*
 * ============================================================
 * 四元数 Layer Normalization
 * 对四元数的w,x,y,z分量独立计算均值和方差后归一化
 * ============================================================
 */

typedef struct {
    float* gamma;
    float* beta;
    int num_features;
    float epsilon;
    int initialized;
} QuatLayerNorm;

QuatLayerNorm* quat_layernorm_create(int num_features, float epsilon) {
    if (num_features <= 0) return NULL;
    QuatLayerNorm* ln = (QuatLayerNorm*)safe_calloc(1, sizeof(QuatLayerNorm));
    if (!ln) return NULL;

    int qsize = num_features * 4;
    ln->gamma = (float*)safe_calloc(qsize, sizeof(float));
    ln->beta = (float*)safe_calloc(qsize, sizeof(float));
    if (!ln->gamma || !ln->beta) {
        safe_free((void**)&ln->gamma);
        safe_free((void**)&ln->beta);
        safe_free((void**)&ln);
        return NULL;
    }

    for (int i = 0; i < qsize; i++) ln->gamma[i] = 1.0f;
    ln->num_features = num_features;
    ln->epsilon = epsilon > 0.0f ? epsilon : 1e-5f;
    ln->initialized = 1;
    return ln;
}

void quat_layernorm_free(QuatLayerNorm* ln) {
    if (!ln) return;
    safe_free((void**)&ln->gamma);
    safe_free((void**)&ln->beta);
    safe_free((void**)&ln);
}

int quat_layernorm_forward(QuatLayerNorm* ln, const float* input,
                           size_t batch_size, float* output) {
    if (!ln || !input || !output || batch_size == 0 || !ln->initialized) return -1;

    int qsize = ln->num_features * 4;
    float eps = ln->epsilon;

    for (size_t b = 0; b < batch_size; b++) {
        const float* x = input + (size_t)b * qsize;
        float* y = output + (size_t)b * qsize;

        float mean = 0.0f;
        for (int i = 0; i < qsize; i++) mean += x[i];
        mean /= (float)qsize;

        float var = 0.0f;
        for (int i = 0; i < qsize; i++) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= (float)qsize;

        float std_inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < qsize; i++) {
            float norm = (x[i] - mean) * std_inv;
            y[i] = norm * ln->gamma[i] + ln->beta[i];
        }
    }
    return 0;
}

/*
 * ============================================================
 * 四元数注意力机制（缩放点积注意力）
 * ============================================================
 */

int quat_attention_scaled_dot_product(
    const Quaternion* queries, const Quaternion* keys,
    const Quaternion* values, size_t seq_len, size_t head_dim,
    Quaternion* output) {
    if (!queries || !keys || !values || !output) return -1;
    if (seq_len == 0 || head_dim == 0) return -1;

    float scale = 1.0f / sqrtf((float)head_dim);

    float* scores = (float*)safe_calloc(seq_len * seq_len, sizeof(float));
    if (!scores) return -1;

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            float dot = queries[i].w * keys[j].w + queries[i].x * keys[j].x +
                        queries[i].y * keys[j].y + queries[i].z * keys[j].z;
            scores[i * seq_len + j] = dot * scale;
        }

        float max_s = scores[i * seq_len];
        for (size_t j = 1; j < seq_len; j++)
            if (scores[i * seq_len + j] > max_s) max_s = scores[i * seq_len + j];

        float sum = 0.0f;
        for (size_t j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_s);
            sum += scores[i * seq_len + j];
        }
        float inv_sum = 1.0f / (sum + 1e-10f);
        for (size_t j = 0; j < seq_len; j++)
            scores[i * seq_len + j] *= inv_sum;
    }

    for (size_t i = 0; i < seq_len; i++) {
        output[i].w = 0.0f; output[i].x = 0.0f;
        output[i].y = 0.0f; output[i].z = 0.0f;
        for (size_t j = 0; j < seq_len; j++) {
            float w = scores[i * seq_len + j];
            output[i].w += w * values[j].w;
            output[i].x += w * values[j].x;
            output[i].y += w * values[j].y;
            output[i].z += w * values[j].z;
        }
    }

    safe_free((void**)&scores);
    return 0;
}

/*
 * ============================================================
 * 四元数残差连接
 * ============================================================
 */

int quat_residual_connection(const Quaternion* input, const Quaternion* sublayer,
                              size_t num_quats, Quaternion* output) {
    if (!input || !sublayer || !output || num_quats == 0) return -1;
    for (size_t i = 0; i < num_quats; i++) {
        output[i].w = input[i].w + sublayer[i].w;
        output[i].x = input[i].x + sublayer[i].x;
        output[i].y = input[i].y + sublayer[i].y;
        output[i].z = input[i].z + sublayer[i].z;
    }
    return 0;
}
