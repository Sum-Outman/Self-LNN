/**
 * @file vec_ops_shared.h
 * @brief 通用N维向量运算库 —— 统一的数学基础函数
 *
 * P1-015 修复：合并 image_recognition_deep.c 与 object_recognition.c 中重复的 static 向量运算函数。
 * 采用 header-only (static inline) 实现，与 vec3_ops.h 保持一致的设计模式，
 * 无需修改 CMakeLists.txt。
 *
 * 函数清单：
 *   标量激活函数：   vec_sigmoid_f / vec_tanh_f
 *   权重初始化：      vec_init_kaiming / vec_init_he
 *   矩阵-向量乘法：   vec_mat_mul
 *   向量逐元素运算：   vec_add / vec_hadamard / vec_sigmoid / vec_tanh / vec_scale / vec_copy
 *   向量内积/范数：   vec_dot / vec_norm / vec_normalize
 *   向量距离/相似度： vec_l2_dist / vec_cos_sim
 *   概率归一化：      vec_softmax
 */

#ifndef SELFLNN_MATH_VEC_OPS_SHARED_H
#define SELFLNN_MATH_VEC_OPS_SHARED_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <string.h>
#include "selflnn/utils/secure_random.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- 标量激活函数 ---- */

/* Sigmoid激活函数: f(x) = 1 / (1 + e^(-x)) */
static inline float vec_sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Tanh激活函数: f(x) = (e^(2x) - 1) / (e^(2x) + 1) */
static inline float vec_tanh_f(float x) {
    float e2x = expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

/* ---- 权重初始化 ---- */

/* Kaiming/Xavier初始化：适用于tanh/sigmoid激活函数层
 * 公式：std = sqrt(2.0 / (fan_in + fan_out))
 * 使用Box-Muller变换生成正态分布随机数 */
static inline void vec_init_kaiming(float* w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++) {
        float u1 = secure_random_float();
        float u2 = secure_random_float();
        if (u1 < 1e-7f) u1 = 1e-7f;
        w[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * scale;
    }
}

/* He初始化：适用于ReLU及变体激活函数层
 * 公式：std = sqrt(2.0 / fan_in)
 * 对ReLU激活的层使用此初始化可以避免梯度消失，保持前向传播方差 */
static inline void vec_init_he(float* w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        float u1 = secure_random_float();
        float u2 = secure_random_float();
        if (u1 < 1e-7f) u1 = 1e-7f;
        w[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * scale;
    }
}

/* ---- 矩阵-向量乘法 ---- */

/* 矩阵-向量乘法: out(r,1) = mat(r,c) * vec(c,1) */
static inline void vec_mat_mul(const float* mat, const float* vec, float* out, int r, int c) {
    for (int i = 0; i < r; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < c; j++) out[i] += mat[i * c + j] * vec[j];
    }
}

/* ---- 向量逐元素运算（in-place） ---- */

/* 向量逐元素加法: a[i] += b[i] */
static inline void vec_add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

/* 向量逐元素乘法(Hadamard积): a[i] *= b[i] */
static inline void vec_hadamard(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

/* 向量逐元素Sigmoid: a[i] = sigmoid(a[i]) */
static inline void vec_sigmoid(float* a, int n) {
    for (int i = 0; i < n; i++) a[i] = vec_sigmoid_f(a[i]);
}

/* 向量逐元素Tanh: a[i] = tanh(a[i]) */
static inline void vec_tanh(float* a, int n) {
    for (int i = 0; i < n; i++) a[i] = vec_tanh_f(a[i]);
}

/* 向量标量乘法: a[i] *= s */
static inline void vec_scale(float* a, float s, int n) {
    for (int i = 0; i < n; i++) a[i] *= s;
}

/* 向量拷贝: d = s (使用memcpy优化) */
static inline void vec_copy(float* d, const float* s, int n) {
    memcpy(d, s, (size_t)n * sizeof(float));
}

/* ---- 向量内积与范数 ---- */

/* 向量内积: sum(a[i] * b[i]) */
static inline float vec_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* 向量L2范数(norm): sqrt(sum(a[i]^2)) */
static inline float vec_norm(const float* a, int n) {
    return sqrtf(vec_dot(a, a, n));
}

/* 向量L2归一化: a = a / ||a|| */
static inline void vec_normalize(float* a, int n) {
    float norm = vec_norm(a, n);
    if (norm > 1e-10f) vec_scale(a, 1.0f / norm, n);
}

/* ---- 向量距离与相似度 ---- */

/* L2欧式距离: sqrt(sum((a[i]-b[i])^2)) */
static inline float vec_l2_dist(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s);
}

/* 余弦相似度: dot(a,b) / (||a|| * ||b||) */
static inline float vec_cos_sim(const float* a, const float* b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = sqrtf(na * nb);
    return (denom > 1e-10f) ? dot / denom : 0.0f;
}

/* ---- 概率归一化 ---- */

/* Softmax归一化: logits[i] = exp(logits[i] - max) / sum(exp) */
static inline void vec_softmax(float* logits, int n) {
    float mv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mv) mv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - mv);
        sum += logits[i];
    }
    if (sum > 1e-10f) {
        for (int i = 0; i < n; i++) logits[i] /= sum;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MATH_VEC_OPS_SHARED_H */
