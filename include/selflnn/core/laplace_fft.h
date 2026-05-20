#ifndef SELFLNN_LAPLACE_FFT_H
#define SELFLNN_LAPLACE_FFT_H

#include <stddef.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 复数类型 */
typedef struct { float re; float im; } LFFT_Complex;

/* 位反转排序 */
static inline void lfft_bit_reverse_order(LFFT_Complex* data, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < j) { LFFT_Complex tmp = data[i]; data[i] = data[j]; data[j] = tmp; }
        size_t m = n >> 1;
        while (m > 0 && (j & m)) { j &= ~m; m >>= 1; }
        j |= m;
    }
}

/* Cooley-Tukey基-2复数FFT（在位计算）
 * B-005修复: 蝶形运算添加边界检查，防止索引溢出；
 * 增加输入大小为2的幂的验证，n必须满足 (n & (n-1)) == 0 */
static inline void lfft_complex_inplace(LFFT_Complex* data, size_t n, int inverse) {
    /* B-005: 验证输入大小必须是2的幂，非2的幂时FFT索引计算无法保证安全 */
    if (n == 0 || (n & (n - 1)) != 0) return;
    lfft_bit_reverse_order(data, n);
    for (size_t s = 2; s <= n; s <<= 1) {
        float angle = (inverse ? 2.0f : -2.0f) * (float)M_PI / (float)s;
        LFFT_Complex wm = { cosf(angle), sinf(angle) };
        for (size_t k = 0; k < n; k += s) {
            LFFT_Complex w = { 1.0f, 0.0f };
            size_t half = s / 2;
            for (size_t j = 0; j < half; j++) {
                size_t idx_a = k + j;
                size_t idx_b = k + j + half;
                /* B-005: 边界检查防止索引越界 */
                if (idx_b >= n) break;
                LFFT_Complex t = { w.re * data[idx_b].re - w.im * data[idx_b].im,
                                   w.re * data[idx_b].im + w.im * data[idx_b].re };
                data[idx_b].re = data[idx_a].re - t.re;
                data[idx_b].im = data[idx_a].im - t.im;
                data[idx_a].re += t.re;
                data[idx_a].im += t.im;
                float wn_re = w.re * wm.re - w.im * wm.im;
                float wn_im = w.re * wm.im + w.im * wm.re;
                w.re = wn_re; w.im = wn_im;
            }
        }
    }
    if (inverse) {
        float inv_n = 1.0f / (float)n;
        for (size_t i = 0; i < n; i++) { data[i].re *= inv_n; data[i].im *= inv_n; }
    }
}

/* 实部+虚部分开数组的FFT接口 */
static inline void lfft_split_radix2(float* real, float* imag, size_t n, int inverse) {
    LFFT_Complex* tmp = (LFFT_Complex*)malloc(n * sizeof(LFFT_Complex));
    if (!tmp) return;
    for (size_t i = 0; i < n; i++) { tmp[i].re = real[i]; tmp[i].im = imag ? imag[i] : 0.0f; }
    lfft_complex_inplace(tmp, n, inverse);
    for (size_t i = 0; i < n; i++) { real[i] = tmp[i].re; if (imag) imag[i] = tmp[i].im; }
    free(tmp);
}

/* 实数FFT（输入纯实数，输出复谱） */
static inline void lfft_real_forward(const float* input, float* real_out, float* imag_out, size_t n) {
    for (size_t i = 0; i < n; i++) { real_out[i] = input[i]; imag_out[i] = 0.0f; }
    lfft_split_radix2(real_out, imag_out, n, 0);
}

/* 实数IFFT（输入复谱，输出实数） */
static inline void lfft_real_inverse(const float* real_in, const float* imag_in, float* output, size_t n) {
    float* re = (float*)malloc(n * sizeof(float));
    float* im = (float*)malloc(n * sizeof(float));
    if (!re || !im) { free(re); free(im); return; }
    memcpy(re, real_in, n * sizeof(float));
    memcpy(im, imag_in, n * sizeof(float));
    lfft_split_radix2(re, im, n, 1);
    for (size_t i = 0; i < n; i++) output[i] = re[i];
    free(re); free(im);
}

#endif
