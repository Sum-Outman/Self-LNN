/**
 * @file simd_utils.h
 * @brief 统一SIMD向量化内联基元
 * 
 * 消除 cfc_enhanced.c 和 gpu_cpu.c 中重复的SIMD辅助函数。
 * 提供统一的SSE/AVX/NEON向量化接口。
 * 纯C实现，零外部依赖。
 */

#ifndef SELFLNN_SIMD_UTILS_H
#define SELFLNN_SIMD_UTILS_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#define SELFLNN_SIMD_X86 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SELFLNN_SIMD_NEON 1
#endif

#ifdef SELFLNN_SIMD_X86

static inline float simd_dot_product(const float* a, const float* b, size_t n) {
    float total = 0.0f;
    size_t i = 0;
#if defined(__AVX__)
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(va, vb));
    }
    float buf[8];
    _mm256_storeu_ps(buf, sum256);
    for (int k = 0; k < 8; k++) total += buf[k];
#endif
    __m128 sum128 = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum128 = _mm_add_ps(sum128, _mm_mul_ps(va, vb));
    }
    float buf4[4];
    _mm_storeu_ps(buf4, sum128);
    for (int k = 0; k < 4; k++) total += buf4[k];
    for (; i < n; i++) total += a[i] * b[i];
    return total;
}

static inline void simd_vector_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
#if defined(__AVX__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
#endif
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(c + i, _mm_add_ps(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

static inline void simd_sigmoid_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
#if defined(__AVX__)
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), vx);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 neg_c = _mm256_max_ps(_mm256_set1_ps(-4.0f), _mm256_min_ps(_mm256_set1_ps(4.0f), neg));
        __m256 x2 = _mm256_mul_ps(neg_c, neg_c);
        __m256 x3 = _mm256_mul_ps(x2, neg_c);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 exp_a = _mm256_add_ps(one, _mm256_add_ps(neg_c,
            _mm256_add_ps(_mm256_mul_ps(x2, _mm256_set1_ps(0.5f)),
            _mm256_add_ps(_mm256_mul_ps(x3, _mm256_set1_ps(1.0f/6.0f)),
                          _mm256_mul_ps(x4, _mm256_set1_ps(1.0f/24.0f))))));
        _mm256_storeu_ps(y + i, _mm256_div_ps(one, _mm256_add_ps(one, exp_a)));
    }
#endif
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 neg = _mm_sub_ps(_mm_setzero_ps(), vx);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 neg_c = _mm_max_ps(_mm_set1_ps(-4.0f), _mm_min_ps(_mm_set1_ps(4.0f), neg));
        __m128 x2 = _mm_mul_ps(neg_c, neg_c);
        __m128 x3 = _mm_mul_ps(x2, neg_c);
        __m128 x4 = _mm_mul_ps(x2, x2);
        __m128 exp_a = _mm_add_ps(one, _mm_add_ps(neg_c,
            _mm_add_ps(_mm_mul_ps(x2, _mm_set1_ps(0.5f)),
            _mm_add_ps(_mm_mul_ps(x3, _mm_set1_ps(1.0f/6.0f)),
                       _mm_mul_ps(x4, _mm_set1_ps(1.0f/24.0f))))));
        _mm_storeu_ps(y + i, _mm_div_ps(one, _mm_add_ps(one, exp_a)));
    }
    for (; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

static inline void simd_tanh_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
#if defined(__AVX__)
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vx2 = _mm256_mul_ps(vx, _mm256_set1_ps(2.0f));
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), vx2);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 neg_c = _mm256_max_ps(_mm256_set1_ps(-4.0f), _mm256_min_ps(_mm256_set1_ps(4.0f), neg));
        __m256 x2 = _mm256_mul_ps(neg_c, neg_c);
        __m256 x3 = _mm256_mul_ps(x2, neg_c);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 exp_a = _mm256_add_ps(one, _mm256_add_ps(neg_c,
            _mm256_add_ps(_mm256_mul_ps(x2, _mm256_set1_ps(0.5f)),
            _mm256_add_ps(_mm256_mul_ps(x3, _mm256_set1_ps(1.0f/6.0f)),
                          _mm256_mul_ps(x4, _mm256_set1_ps(1.0f/24.0f))))));
        __m256 sig2x = _mm256_div_ps(one, _mm256_add_ps(one, exp_a));
        _mm256_storeu_ps(y + i, _mm256_sub_ps(_mm256_mul_ps(two, sig2x), one));
    }
#endif
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vx2 = _mm_mul_ps(vx, _mm_set1_ps(2.0f));
        __m128 neg = _mm_sub_ps(_mm_setzero_ps(), vx2);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 two = _mm_set1_ps(2.0f);
        __m128 neg_c = _mm_max_ps(_mm_set1_ps(-4.0f), _mm_min_ps(_mm_set1_ps(4.0f), neg));
        __m128 x2 = _mm_mul_ps(neg_c, neg_c);
        __m128 x3 = _mm_mul_ps(x2, neg_c);
        __m128 x4 = _mm_mul_ps(x2, x2);
        __m128 exp_a = _mm_add_ps(one, _mm_add_ps(neg_c,
            _mm_add_ps(_mm_mul_ps(x2, _mm_set1_ps(0.5f)),
            _mm_add_ps(_mm_mul_ps(x3, _mm_set1_ps(1.0f/6.0f)),
                       _mm_mul_ps(x4, _mm_set1_ps(1.0f/24.0f))))));
        __m128 sig2x = _mm_div_ps(one, _mm_add_ps(one, exp_a));
        _mm_storeu_ps(y + i, _mm_sub_ps(_mm_mul_ps(two, sig2x), one));
    }
    for (; i < n; i++) y[i] = tanhf(x[i]);
}

static inline void simd_relu_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
#if defined(__AVX__)
    __m256 zero256 = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_max_ps(_mm256_loadu_ps(x + i), zero256));
    }
#endif
    __m128 zero128 = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        _mm_storeu_ps(y + i, _mm_max_ps(_mm_loadu_ps(x + i), zero128));
    }
    for (; i < n; i++) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

static inline void simd_gate_apply(const float* gate, const float* x,
                                    const float* state, float* y, size_t n) {
    size_t i = 0;
#if defined(__AVX__)
    __m256 one256 = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 vg = _mm256_loadu_ps(gate + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vs = _mm256_loadu_ps(state + i);
        __m256 term1 = _mm256_mul_ps(vg, vx);
        __m256 term2 = _mm256_mul_ps(_mm256_sub_ps(one256, vg), vs);
        _mm256_storeu_ps(y + i, _mm256_add_ps(term1, term2));
    }
#endif
    __m128 one128 = _mm_set1_ps(1.0f);
    for (; i + 4 <= n; i += 4) {
        __m128 vg = _mm_loadu_ps(gate + i);
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vs = _mm_loadu_ps(state + i);
        __m128 term1 = _mm_mul_ps(vg, vx);
        __m128 term2 = _mm_mul_ps(_mm_sub_ps(one128, vg), vs);
        _mm_storeu_ps(y + i, _mm_add_ps(term1, term2));
    }
    for (; i < n; i++) y[i] = gate[i] * x[i] + (1.0f - gate[i]) * state[i];
}

static inline void simd_sgd_update(float* restrict w, const float* restrict g,
                                    float lr, float wd, int n) {
    int i = 0;
#if defined(__AVX__)
    __m256 lr256 = _mm256_set1_ps(lr);
    __m256 wd256 = _mm256_set1_ps(wd);
    for (; i <= n - 8; i += 8) {
        __m256 wv = _mm256_loadu_ps(&w[i]);
        __m256 gv = _mm256_loadu_ps(&g[i]);
        __m256 update = _mm256_mul_ps(lr256, _mm256_add_ps(gv, _mm256_mul_ps(wd256, wv)));
        _mm256_storeu_ps(&w[i], _mm256_sub_ps(wv, update));
    }
#endif
    __m128 lr128 = _mm_set1_ps(lr);
    __m128 wd128 = _mm_set1_ps(wd);
    for (; i <= n - 4; i += 4) {
        __m128 wv = _mm_loadu_ps(&w[i]);
        __m128 gv = _mm_loadu_ps(&g[i]);
        __m128 update = _mm_mul_ps(lr128, _mm_add_ps(gv, _mm_mul_ps(wd128, wv)));
        _mm_storeu_ps(&w[i], _mm_sub_ps(wv, update));
    }
    for (; i < n; i++) w[i] -= lr * (g[i] + wd * w[i]);
}

#elif defined(SELFLNN_SIMD_NEON)

static inline float simd_dot_product(const float* a, const float* b, size_t n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }
    float result[4];
    vst1q_f32(result, sum);
    float total = result[0] + result[1] + result[2] + result[3];
    for (; i < n; i++) total += a[i] * b[i];
    return total;
}

static inline void simd_vector_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

static inline void simd_sigmoid_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t neg = vnegq_f32(vx);
        float32x4_t neg_c = vmaxq_f32(vdupq_n_f32(-4.0f), vminq_f32(vdupq_n_f32(4.0f), neg));
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t x2 = vmulq_f32(neg_c, neg_c);
        float32x4_t exp_a = vaddq_f32(one, vaddq_f32(neg_c,
            vaddq_f32(vmulq_f32(x2, vdupq_n_f32(0.5f)),
            vaddq_f32(vmulq_f32(vmulq_f32(x2, neg_c), vdupq_n_f32(1.0f/6.0f)),
                      vmulq_f32(vmulq_f32(x2, x2), vdupq_n_f32(1.0f/24.0f))))));
        float32x4_t recip = vrecpeq_f32(vaddq_f32(one, exp_a));
        recip = vmulq_f32(vrecpsq_f32(vaddq_f32(one, exp_a), recip), recip);
        vst1q_f32(y + i, recip);
    }
    for (; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

static inline void simd_tanh_batch(const float* x, float* y, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vx2 = vmulq_f32(vx, vdupq_n_f32(2.0f));
        float32x4_t neg = vnegq_f32(vx2);
        float32x4_t neg_c = vmaxq_f32(vdupq_n_f32(-4.0f), vminq_f32(vdupq_n_f32(4.0f), neg));
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t two = vdupq_n_f32(2.0f);
        float32x4_t x2 = vmulq_f32(neg_c, neg_c);
        float32x4_t exp_a = vaddq_f32(one, vaddq_f32(neg_c,
            vaddq_f32(vmulq_f32(x2, vdupq_n_f32(0.5f)),
            vaddq_f32(vmulq_f32(vmulq_f32(x2, neg_c), vdupq_n_f32(1.0f/6.0f)),
                      vmulq_f32(vmulq_f32(x2, x2), vdupq_n_f32(1.0f/24.0f))))));
        float32x4_t sig2x = vrecpeq_f32(vaddq_f32(one, exp_a));
        sig2x = vmulq_f32(vrecpsq_f32(vaddq_f32(one, exp_a), sig2x), sig2x);
        vst1q_f32(y + i, vsubq_f32(vmulq_f32(two, sig2x), one));
    }
    for (; i < n; i++) y[i] = tanhf(x[i]);
}

static inline void simd_relu_batch(const float* x, float* y, size_t n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vst1q_f32(y + i, vmaxq_f32(vx, zero));
    }
    for (; i < n; i++) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}

static inline void simd_gate_apply(const float* gate, const float* x,
                                    const float* state, float* y, size_t n) {
    float32x4_t one = vdupq_n_f32(1.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vg = vld1q_f32(gate + i);
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vs = vld1q_f32(state + i);
        float32x4_t term1 = vmulq_f32(vg, vx);
        float32x4_t term2 = vmulq_f32(vsubq_f32(one, vg), vs);
        vst1q_f32(y + i, vaddq_f32(term1, term2));
    }
    for (; i < n; i++) y[i] = gate[i] * x[i] + (1.0f - gate[i]) * state[i];
}

static inline void simd_sgd_update(float* restrict w, const float* restrict g,
                                    float lr, float wd, int n) {
    float32x4_t lrv = vdupq_n_f32(lr);
    float32x4_t wdv = vdupq_n_f32(wd);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t wv = vld1q_f32(&w[i]);
        float32x4_t gv = vld1q_f32(&g[i]);
        float32x4_t update = vmulq_f32(lrv, vaddq_f32(gv, vmulq_f32(wdv, wv)));
        vst1q_f32(&w[i], vsubq_f32(wv, update));
    }
    for (; i < n; i++) w[i] -= lr * (g[i] + wd * w[i]);
}

#else

static inline float simd_dot_product(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline void simd_vector_add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}
static inline void simd_sigmoid_batch(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
}
static inline void simd_tanh_batch(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = tanhf(x[i]);
}
static inline void simd_relu_batch(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}
static inline void simd_gate_apply(const float* gate, const float* x,
                                    const float* state, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = gate[i] * x[i] + (1.0f - gate[i]) * state[i];
}
static inline void simd_sgd_update(float* restrict w, const float* restrict g,
                                    float lr, float wd, int n) {
    for (int i = 0; i < n; i++) w[i] -= lr * (g[i] + wd * w[i]);
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SIMD_UTILS_H */
