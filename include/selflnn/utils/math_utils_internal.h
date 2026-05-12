/**
 * @file math_utils_internal.h
 * @brief 全项目统一内部数学工具函数
 *
 * 消除跨文件重复定义问题（54+处重复）：
 *   sigmoid(8处), softmax(10处), 向量点积(12处), 矩阵乘法(7处),
 *   矩阵转置(5处), Cholesky分解(6处), FFT(6处), 向量叉积(6处)
 *
 * 使用方式：各模块 #include "selflnn/utils/math_utils_internal.h"
 * 所有函数为 static inline，零链接开销。
 */

#ifndef SELFLNN_MATH_UTILS_INTERNAL_H
#define SELFLNN_MATH_UTILS_INTERNAL_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 基础激活函数
 * ================================================================ */

/** @brief Sigmoid 激活函数 σ(x) = 1/(1+e^(-x))，数值稳定 */
static inline float slnn_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

/** @brief Sigmoid 导数 σ'(x) = σ(x)(1-σ(x)) */
static inline float slnn_sigmoid_derivative(float x) {
    float s = slnn_sigmoid(x);
    return s * (1.0f - s);
}

/** @brief Softmax：数值稳定，对数域计算避免溢出 */
static inline void slnn_softmax(float* x, size_t n) {
    if (n == 0) return;
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    if (sum > 1e-30f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < n; i++) {
            x[i] *= inv_sum;
        }
    }
}

/* ================================================================
 * 向量运算
 * ================================================================ */

/** @brief 向量点积（内积） */
static inline float slnn_vec_dot(const float* a, const float* b, size_t n) {
    float result = 0.0f;
    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

/** @brief 3D向量叉积 c = a × b */
static inline void slnn_vec_cross3(const float a[3], const float b[3], float c[3]) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/** @brief 向量L2范数 */
static inline float slnn_vec_norm(const float* v, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return sqrtf(sum);
}

/** @brief 向量归一化（就地） */
static inline void slnn_vec_normalize(float* v, size_t n) {
    float norm = slnn_vec_norm(v, n);
    if (norm > 1e-10f) {
        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < n; i++) {
            v[i] *= inv_norm;
        }
    }
}

/** @brief 向量加法 c = a + b */
static inline void slnn_vec_add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/** @brief 向量减法 c = a - b */
static inline void slnn_vec_sub(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] - b[i];
    }
}

/** @brief 向量标量乘 c = s * v */
static inline void slnn_vec_scale(float* v, float s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        v[i] *= s;
    }
}

/* ================================================================
 * 矩阵运算
 * ================================================================ */

/** @brief 矩阵乘法 C[m×p] = A[m×n] × B[n×p] */
static inline void slnn_mat_mul(const float* A, const float* B, float* C,
                                 size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

/** @brief 矩阵转置 B[n×m] = A[m×n]^T */
static inline void slnn_mat_transpose(const float* A, float* B, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

/** @brief 矩阵-向量乘法 y[m] = A[m×n] × x[n] */
static inline void slnn_mat_vec_mul(const float* A, const float* x, float* y,
                                     size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ================================================================
 * Cholesky 分解（正定矩阵）
 * ================================================================ */

/** @brief Cholesky分解 A = L·L^T，返回0成功，-1失败（非正定） */
static inline int slnn_cholesky(float* A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            float sum = A[i * n + j];
            for (size_t k = 0; k < j; k++) {
                sum -= A[i * n + k] * A[j * n + k];
            }
            if (i == j) {
                if (sum <= 1e-12f) return -1;
                A[i * n + i] = sqrtf(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
    }
    return 0;
}

/* ================================================================
 * FFT（快速傅里叶变换 Radix-2 Cooley-Tukey）
 * ================================================================ */

/** @brief 复数结构体 */
typedef struct {
    float re;
    float im;
} slnn_complex;

/** @brief 基数2 Cooley-Tukey FFT（就地），n必须为2的幂 */
static inline void slnn_fft_radix2(slnn_complex* data, size_t n, int inverse) {
    /* 位逆序置换 */
    size_t j = 0;
    for (size_t i = 1; i < n; i++) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            slnn_complex tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
    
    /* Cooley-Tukey蝶形运算 */
    int sign = inverse ? 1 : -1;
    for (size_t len = 2; len <= n; len <<= 1) {
        float angle = (float)(sign * 2.0 * 3.141592653589793) / (float)len;
        slnn_complex wlen = { cosf(angle), sinf(angle) };
        for (size_t i = 0; i < n; i += len) {
            slnn_complex w = { 1.0f, 0.0f };
            for (size_t k = 0; k < len / 2; k++) {
                slnn_complex u = data[i + k];
                slnn_complex v = {
                    w.re * data[i + k + len / 2].re - w.im * data[i + k + len / 2].im,
                    w.re * data[i + k + len / 2].im + w.im * data[i + k + len / 2].re
                };
                data[i + k].re = u.re + v.re;
                data[i + k].im = u.im + v.im;
                data[i + k + len / 2].re = u.re - v.re;
                data[i + k + len / 2].im = u.im - v.im;
                float w_re_new = w.re * wlen.re - w.im * wlen.im;
                float w_im_new = w.re * wlen.im + w.im * wlen.re;
                w.re = w_re_new;
                w.im = w_im_new;
            }
        }
    }
    
    /* 逆FFT需要除以n */
    if (inverse) {
        float inv_n = 1.0f / (float)n;
        for (size_t i = 0; i < n; i++) {
            data[i].re *= inv_n;
            data[i].im *= inv_n;
        }
    }
}

/* ================================================================
 * Clamp / 数值工具
 * ================================================================ */

#ifndef CLAMP
#define CLAMP(x, min, max) (((x) < (min)) ? (min) : (((x) > (max)) ? (max) : (x)))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/** @brief ReLU激活函数 */
static inline float slnn_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/** @brief ReLU导数 */
static inline float slnn_relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

/** @brief tanh激活函数 */
static inline float slnn_tanh(float x) {
    return tanhf(x);
}

/** @brief tanh导数 */
static inline float slnn_tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MATH_UTILS_INTERNAL_H */
