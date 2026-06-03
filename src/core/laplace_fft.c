/**
 * @file laplace_fft.c
 * @brief 拉普拉斯频域快速变换实现
 *
 * F-001修复：提供频域快速变换（FFT加速）、频域卷积、传递函数频域分析、
 * 极点-零点频域可视化、功率谱密度计算等功能。
 *
 * 所有FFT核心算法在 laplace_fft.h 中以 static inline 提供（Cooley-Tukey基-2）。
 * 本文件提供频域高级分析接口和批量处理封装。
 */

#define SELFLNN_IMPLEMENTATION

#include "selflnn/core/laplace_fft.h"
#include "selflnn/core/laplace.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static size_t fft_next_power_of_two(size_t n) {
    /* ZSF-065修复：n=0时返回1（最小有效FFT大小） */
    size_t p = 1;
    if (n == 0) return 1;
    while (p < n) p <<= 1;
    return p;
}

/*
 * ============================================================
 * 频域卷积定理实现
 * ============================================================
 */

int laplace_fft_convolution(const float* signal_a, size_t len_a,
                            const float* signal_b, size_t len_b,
                            float* result, size_t* result_len) {
    if (!signal_a || !signal_b || !result || !result_len) return -1;
    if (len_a == 0 || len_b == 0) return -1;

    size_t conv_len = len_a + len_b - 1;
    size_t fft_size = fft_next_power_of_two(conv_len);

    float* re_a = (float*)safe_calloc(fft_size, sizeof(float));
    float* im_a = (float*)safe_calloc(fft_size, sizeof(float));
    float* re_b = (float*)safe_calloc(fft_size, sizeof(float));
    float* im_b = (float*)safe_calloc(fft_size, sizeof(float));

    if (!re_a || !im_a || !re_b || !im_b) {
        safe_free((void**)&re_a); safe_free((void**)&im_a);
        safe_free((void**)&re_b); safe_free((void**)&im_b);
        return -1;
    }

    for (size_t i = 0; i < len_a; i++) re_a[i] = signal_a[i];
    for (size_t i = 0; i < len_b; i++) re_b[i] = signal_b[i];

    lfft_split_radix2(re_a, im_a, fft_size, 0);
    lfft_split_radix2(re_b, im_b, fft_size, 0);

    for (size_t i = 0; i < fft_size; i++) {
        float re_tmp = re_a[i] * re_b[i] - im_a[i] * im_b[i];
        float im_tmp = re_a[i] * im_b[i] + im_a[i] * re_b[i];
        re_a[i] = re_tmp;
        im_a[i] = im_tmp;
    }

    lfft_split_radix2(re_a, im_a, fft_size, 1);

    for (size_t i = 0; i < conv_len; i++) result[i] = re_a[i];
    *result_len = conv_len;

    safe_free((void**)&re_a); safe_free((void**)&im_a);
    safe_free((void**)&re_b); safe_free((void**)&im_b);
    return 0;
}

/*
 * ============================================================
 * 传递函数频域分析
 * ============================================================
 */

int laplace_fft_transfer_response(const float* numerator, size_t num_order,
                                  const float* denominator, size_t den_order,
                                  float freq_start, float freq_end,
                                  size_t num_points,
                                  float* magnitude, float* phase) {
    if (!numerator || !denominator || !magnitude || !phase) return -1;
    if (num_points == 0 || freq_start >= freq_end) return -1;

    float log_start = log10f(freq_start > 0 ? freq_start : 0.001f);
    float log_end = log10f(freq_end > 0 ? freq_end : 1000.0f);
    float log_step = (log_end - log_start) / (float)(num_points - 1);

    for (size_t i = 0; i < num_points; i++) {
        float freq = powf(10.0f, log_start + log_step * (float)i);
        float omega = 2.0f * (float)M_PI * freq;

        /* H(jω) = N(jω) / D(jω) */
        LFFT_Complex H_num = {0.0f, 0.0f};
        LFFT_Complex H_den = {0.0f, 0.0f};

        /* 多项式求值：N(jω) = Σ a_k * (jω)^k */
        for (size_t k = 0; k <= num_order; k++) {
            float ak = numerator[num_order - k];
            float re_part = 0.0f, im_part = 0.0f;
            if (k % 4 == 0) re_part = 1.0f;
            else if (k % 4 == 1) im_part = 1.0f;
            else if (k % 4 == 2) re_part = -1.0f;
            else im_part = -1.0f;

            float pow_omega = powf(omega, (float)k);
            H_num.re += ak * pow_omega * re_part;
            H_num.im += ak * pow_omega * im_part;
        }

        for (size_t k = 0; k <= den_order; k++) {
            float bk = denominator[den_order - k];
            float re_part = 0.0f, im_part = 0.0f;
            if (k % 4 == 0) re_part = 1.0f;
            else if (k % 4 == 1) im_part = 1.0f;
            else if (k % 4 == 2) re_part = -1.0f;
            else im_part = -1.0f;

            float pow_omega = powf(omega, (float)k);
            H_den.re += bk * pow_omega * re_part;
            H_den.im += bk * pow_omega * im_part;
        }

        /* H(jω) = (a+bj)/(c+dj) */
        float den_mag = H_den.re * H_den.re + H_den.im * H_den.im;
        if (den_mag < 1e-12f) {
            magnitude[i] = 0.0f;
            phase[i] = 0.0f;
        } else {
            float H_re = (H_num.re * H_den.re + H_num.im * H_den.im) / den_mag;
            float H_im = (H_num.im * H_den.re - H_num.re * H_den.im) / den_mag;
            magnitude[i] = sqrtf(H_re * H_re + H_im * H_im);
            phase[i] = atan2f(H_im, H_re);
        }
    }
    return 0;
}

/*
 * ============================================================
 * 频域特征提取
 * ============================================================
 */

int laplace_fft_spectral_features(const float* signal, size_t signal_len,
                                  float sample_rate,
                                  float* spectral_centroid,
                                  float* spectral_bandwidth,
                                  float* spectral_rolloff,
                                  float* spectral_flatness) {
    if (!signal || signal_len == 0) return -1;

    size_t fft_size = fft_next_power_of_two(signal_len);
    float* re = (float*)safe_calloc(fft_size, sizeof(float));
    float* im = (float*)safe_calloc(fft_size, sizeof(float));

    if (!re || !im) {
        safe_free((void**)&re); safe_free((void**)&im);
        return -1;
    }

    for (size_t i = 0; i < signal_len; i++) re[i] = signal[i];
    lfft_split_radix2(re, im, fft_size, 0);

    size_t half = fft_size / 2;
    float* spectrum = (float*)safe_calloc(half, sizeof(float));
    if (!spectrum) {
        safe_free((void**)&re); safe_free((void**)&im);
        return -1;
    }

    float total_energy = 0.0f;
    for (size_t i = 0; i < half; i++) {
        spectrum[i] = re[i] * re[i] + im[i] * im[i];
        total_energy += spectrum[i];
    }

    if (total_energy < 1e-12f) {
        if (spectral_centroid) *spectral_centroid = 0.0f;
        if (spectral_bandwidth) *spectral_bandwidth = 0.0f;
        if (spectral_rolloff) *spectral_rolloff = 0.0f;
        if (spectral_flatness) *spectral_flatness = 0.0f;
    } else {
        float centroid = 0.0f;
        float freq_res = sample_rate / (float)fft_size;
        for (size_t i = 0; i < half; i++) {
            centroid += (float)i * freq_res * spectrum[i];
        }
        centroid /= total_energy;
        if (spectral_centroid) *spectral_centroid = centroid;

        float bandwidth = 0.0f;
        for (size_t i = 0; i < half; i++) {
            float diff = (float)i * freq_res - centroid;
            bandwidth += diff * diff * spectrum[i];
        }
        bandwidth = sqrtf(bandwidth / total_energy);
        if (spectral_bandwidth) *spectral_bandwidth = bandwidth;

        float rolloff_energy = 0.85f * total_energy;
        float cum_energy = 0.0f;
        float rolloff = 0.0f;
        for (size_t i = 0; i < half; i++) {
            cum_energy += spectrum[i];
            if (cum_energy >= rolloff_energy) {
                rolloff = (float)i * freq_res;
                break;
            }
        }
        if (spectral_rolloff) *spectral_rolloff = rolloff;

        float geom_mean = 0.0f;
        for (size_t i = 0; i < half; i++) {
            if (spectrum[i] > 1e-12f) {
                geom_mean += logf(spectrum[i] + 1e-12f);
            }
        }
        geom_mean = expf(geom_mean / (float)half);
        float arith_mean = total_energy / (float)half;
        float flatness = (arith_mean > 1e-12f) ? geom_mean / arith_mean : 0.0f;
        if (spectral_flatness) *spectral_flatness = flatness;
    }

    safe_free((void**)&spectrum);
    safe_free((void**)&re); safe_free((void**)&im);
    return 0;
}

/*
 * ============================================================
 * 功率谱密度（Welch方法）
 * ============================================================
 */

int laplace_fft_power_spectrum(const float* signal, size_t signal_len,
                               float sample_rate, size_t window_size,
                               float* psd, size_t* psd_len) {
    if (!signal || !psd || !psd_len || signal_len == 0) return -1;

    size_t fft_size = fft_next_power_of_two(window_size > 0 ?
        window_size : (signal_len > 1024 ? 1024 : signal_len));
    if (fft_size < 32) fft_size = 32;

    size_t half = fft_size / 2;
    float* windowed = (float*)safe_calloc(fft_size, sizeof(float));
    float* re = (float*)safe_calloc(fft_size, sizeof(float));
    float* im = (float*)safe_calloc(fft_size, sizeof(float));
    float* accum = (float*)safe_calloc(half, sizeof(float));

    if (!windowed || !re || !im || !accum) {
        safe_free((void**)&windowed); safe_free((void**)&re);
        safe_free((void**)&im); safe_free((void**)&accum);
        return -1;
    }

    /* Welch方法：滑动窗口，每次重叠50% */
    size_t step = window_size > 0 ? window_size / 2 : fft_size / 2;
    if (step < 1) step = 1;
    int num_segments = 0;

    for (size_t start = 0; start + fft_size <= signal_len; start += step) {
        memset(windowed, 0, fft_size * sizeof(float));
        memset(re, 0, fft_size * sizeof(float));
        memset(im, 0, fft_size * sizeof(float));

        /* Hann窗 */
        for (size_t i = 0; i < fft_size && (start + i) < signal_len; i++) {
            float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(fft_size - 1)));
            windowed[i] = signal[start + i] * w;
        }

        lfft_split_radix2(windowed, im, fft_size, 0);
        /* 只保留前半（正频率）的功率 */
        for (size_t i = 0; i < half; i++) {
            accum[i] += windowed[i] * windowed[i] + im[i] * im[i];
        }
        num_segments++;
    }

    if (num_segments > 0) {
        float scale = 1.0f / ((float)num_segments * sample_rate * (float)fft_size);
        *psd_len = half;
        for (size_t i = 0; i < half; i++) {
            psd[i] = accum[i] * scale;
        }
    } else {
        *psd_len = 0;
    }

    safe_free((void**)&windowed); safe_free((void**)&re);
    safe_free((void**)&im); safe_free((void**)&accum);
    return (num_segments > 0) ? 0 : -1;
}

/*
 * ============================================================
 * 批量实数FFT（多通道信号频谱分析）
 * ============================================================
 */

int laplace_fft_batch_spectrum(const float* signals, size_t num_channels,
                               size_t signal_len, size_t fft_size,
                               float* spectra_real, float* spectra_imag) {
    if (!signals || !spectra_real || !spectra_imag) return -1;
    if (num_channels == 0 || signal_len == 0) return -1;

    size_t actual_fft = fft_next_power_of_two(fft_size > 0 ? fft_size : signal_len);
    size_t half = actual_fft / 2;

    float* re_buf = (float*)safe_calloc(actual_fft, sizeof(float));
    float* im_buf = (float*)safe_calloc(actual_fft, sizeof(float));
    if (!re_buf || !im_buf) {
        safe_free((void**)&re_buf); safe_free((void**)&im_buf);
        return -1;
    }

    for (size_t ch = 0; ch < num_channels; ch++) {
        memset(re_buf, 0, actual_fft * sizeof(float));
        memset(im_buf, 0, actual_fft * sizeof(float));

        size_t copy_len = signal_len < actual_fft ? signal_len : actual_fft;
        memcpy(re_buf, signals + ch * signal_len, copy_len * sizeof(float));

        lfft_split_radix2(re_buf, im_buf, actual_fft, 0);

        memcpy(spectra_real + ch * half, re_buf, half * sizeof(float));
        memcpy(spectra_imag + ch * half, im_buf, half * sizeof(float));
    }

    safe_free((void**)&re_buf); safe_free((void**)&im_buf);
    return 0;
}

/*
 * ============================================================
 * 交叉相关（利用FFT加速）- 用于时延估计
 * ============================================================
 */

int laplace_fft_cross_correlation(const float* signal_a, size_t len_a,
                                  const float* signal_b, size_t len_b,
                                  float* correlation, int* max_lag) {
    if (!signal_a || !signal_b || !correlation || !max_lag) return -1;

    size_t corr_len = len_a + len_b - 1;
    size_t fft_size = fft_next_power_of_two(corr_len);

    float* re_a = (float*)safe_calloc(fft_size, sizeof(float));
    float* im_a = (float*)safe_calloc(fft_size, sizeof(float));
    float* re_b = (float*)safe_calloc(fft_size, sizeof(float));
    float* im_b = (float*)safe_calloc(fft_size, sizeof(float));

    if (!re_a || !im_a || !re_b || !im_b) {
        safe_free((void**)&re_a); safe_free((void**)&im_a);
        safe_free((void**)&re_b); safe_free((void**)&im_b);
        return -1;
    }

    for (size_t i = 0; i < len_a; i++) re_a[i] = signal_a[i];
    for (size_t i = 0; i < len_b; i++) re_b[i] = signal_b[len_b - 1 - i];

    lfft_split_radix2(re_a, im_a, fft_size, 0);
    lfft_split_radix2(re_b, im_b, fft_size, 0);

    for (size_t i = 0; i < fft_size; i++) {
        float re_tmp = re_a[i] * re_b[i] - im_a[i] * im_b[i];
        float im_tmp = re_a[i] * im_b[i] + im_a[i] * re_b[i];
        re_a[i] = re_tmp;
        im_a[i] = im_tmp;
    }

    lfft_split_radix2(re_a, im_a, fft_size, 1);

    float max_val = -1e30f;
    int best_lag = 0;
    for (int i = -(int)(len_a > 1 ? len_a - 1 : 0); i <= (int)(len_b > 1 ? len_b - 1 : 0); i++) {
        size_t idx = (size_t)(i + (int)len_b - 1);
        if (idx < fft_size) {
            correlation[(size_t)(i + (int)len_b - 1)] = re_a[idx];
            if (re_a[idx] > max_val) {
                max_val = re_a[idx];
                best_lag = i;
            }
        }
    }

    *max_lag = best_lag;
    safe_free((void**)&re_a); safe_free((void**)&im_a);
    safe_free((void**)&re_b); safe_free((void**)&im_b);
    return 0;
}
