#include "selflnn/core/laplace_ai_framework.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/laplace_fft.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/learning/reinforcement_learning.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/* ============================================================================
 * ZSFBUILD: 缺失类型定义 —— 原laplace_ai_framework.h为空转发，
 * 以下类型定义从未在任何头文件中声明，在此处补全以确保编译通过。
 * ============================================================================ */

#define LAI_SPECTRUM_SIZE 4096
#define LAI_MAX_FILTER_BANDS 64

typedef enum {
    LAI_FILTER_NONE = -1,       /**< 不过滤 */
    LAI_FILTER_LOWPASS = 0,     /**< 低通 */
    LAI_FILTER_HIGHPASS = 1,    /**< 高通 */
    LAI_FILTER_BANDPASS = 2,    /**< 带通 */
    LAI_FILTER_BANDSTOP = 3,    /**< 带阻 */
    LAI_FILTER_NOTCH = 4,       /**< 陷波 */
    LAI_FILTER_ADAPTIVE = 5     /**< 自适应 */
} LAIFilterType;

typedef enum {
    LAI_WINDOW_NONE = -1,       /**< 不加窗 */
    LAI_WINDOW_HANN = 0,        /**< Hann窗 */
    LAI_WINDOW_HAMMING = 1,     /**< Hamming窗 */
    LAI_WINDOW_BLACKMAN = 2,    /**< Blackman窗 */
    LAI_WINDOW_KAISER = 3,      /**< Kaiser窗 */
    LAI_WINDOW_RECTANGULAR = 4  /**< 矩形窗（等同于NONE） */
} LAIWindowType;

/* 向后兼容别名 */
#define LAI_WINDOW_HANNING LAI_WINDOW_HANN

typedef struct LAIStabilityAnalysis {
    float gain_margin;
    float phase_margin;
    float dominant_pole_real;
    float dominant_pole_imag;
    float stability_margin;
    float stability_score;          /**< 综合稳定性评分 0-1 */
     float noise_level;             /**< 噪声水平 (归一化) */
     float oscillation_index;       /**< 振荡指数 */
     int is_stable;
     int warning_level;              /**< 警告级别 0=正常 1=临界 2=危险 */
     char description[256];          /**< 诊断描述 */
 } LAIStabilityAnalysis;

/* ZSFBUILD: LAIStabilityReport 与 LAIStabilityAnalysis 等价 */
typedef LAIStabilityAnalysis LAIStabilityReport;

typedef struct LaplaceAIConfig {
    int fft_size;
    float sampling_rate;
    int enable_adaptive_filter;
    int window_type;
    float kaiser_beta;
    int num_feature_bands;
    float feature_freq_bands[32];
    int enable_rl;
    int filter_order;              /**< 滤波器阶数 (1-4) */
    float high_cutoff_hz;          /**< 高频截止 (Hz) */
    float lr_min_scale;            /**< 学习率最小缩放 */
    float lr_max_scale;            /**< 学习率最大缩放 */
    float lr_adaptation_speed;     /**< 学习率自适应速度 */
} LaplaceAIConfig;

typedef struct LaplaceAI {
    LaplaceAIConfig config;
    LaplaceAnalyzer* analyzer;
    float* work_buffer;
    float* filter_kernel_real;
    float* filter_kernel_imag;
    int kernel_size;
    float prev_lr;
    float lr_momentum;
    int is_initialized;
} LaplaceAI;

typedef struct LAISpectrumResult {
    float* magnitude;
    float* phase;
    float* frequency;
    float* band_energies;           /**< 各频带能量 */
    int fft_size;
    float sampling_rate;
    int num_frequencies;
    int num_bins;                   /**< 频带数 */
    int num_bands;                  /**< 子带数 */
    float total_energy;             /**< 总能量 */
    float spectral_centroid;        /**< 频谱质心 */
    float spectral_rolloff;         /**< 频谱滚降 */
    float dominant_freq;            /**< 主导频率 */
    float processing_time_ms;       /**< 处理时间(ms) */
} LAISpectrumResult;

 /* 变换缓冲区（时域+频域双域存储） */
 typedef struct {
     size_t signal_length;           /**< 原始信号长度 */
     size_t spectrum_size;           /**< 频谱大小 (fft_n/2+1) */
     float* time_domain;             /**< 时域信号副本 */
     float* frequency_domain_real;   /**< FFT实部 */
     float* frequency_domain_imag;   /**< FFT虚部 */
     float* magnitude_spectrum;      /**< 幅度谱 */
     float* phase_spectrum;          /**< 相位谱 */
     double transform_time_ms;       /**< 变换耗时(ms) */
 } LAITransformBuffer;

/* 默认配置初始化宏 */
#define LAPLACE_AI_CONFIG_DEFAULT_INITIALIZER { \
    512,      /* fft_size */ \
    44000.0f, /* sampling_rate */ \
    1,        /* enable_adaptive_filter */ \
    LAI_WINDOW_HANN, /* window_type */ \
    0.0f,     /* kaiser_beta */ \
    8,        /* num_feature_bands */ \
    {0},      /* feature_freq_bands */ \
    0         /* enable_rl */ \
}

/* ========================================================================= */

/* 线程安全宏：各编译器TLS支持 */
#if defined(_MSC_VER)
#define LAI_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define LAI_THREAD_LOCAL _Thread_local
#else
#define LAI_THREAD_LOCAL __thread
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct LaplaceAI {
    LaplaceAIConfig config;
    LaplaceAnalyzer* analyzer;
    float* work_buffer;
    float* filter_kernel_real;
    float* filter_kernel_imag;
    size_t kernel_size;
    double prev_lr;
    double lr_momentum;
    int is_initialized;
};

#define RL_MAX_AGENTS 8
static RLAgent* g_rl_agents[RL_MAX_AGENTS];
static int g_rl_agent_count = 0;

static int lai_rl_find_slot(void)
{
    for (int i = 0; i < RL_MAX_AGENTS; i++)
        if (!g_rl_agents[i]) return i;
    return -1;
}

static int lai_rl_find_agent(const RLAgent* agent)
{
    for (int i = 0; i < RL_MAX_AGENTS; i++)
        if (g_rl_agents[i] == agent) return i;
    return -1;
}

/* FFT：统一使用 laplace_fft.h，消除重复代码 */
#define lai_fft_radix2(real, imag, n, invert) lfft_split_radix2(real, imag, n, invert)

static void lai_apply_window(float* data, size_t length, LAIWindowType type, float beta) {
    if (type == LAI_WINDOW_NONE) return;
    for (size_t i = 0; i < length; i++) {
        float t = (float)i / (float)(length - 1);
        float w = 1.0f;
        switch (type) {
            case LAI_WINDOW_HANNING:
                w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * t));
                break;
            case LAI_WINDOW_HAMMING:
                w = 0.53836f - 0.46164f * cosf(2.0f * (float)M_PI * t);
                break;
            case LAI_WINDOW_BLACKMAN:
                w = 0.42f - 0.5f * cosf(2.0f * (float)M_PI * t) + 0.08f * cosf(4.0f * (float)M_PI * t);
                break;
            case LAI_WINDOW_KAISER: {
                float alpha = beta;
                float arg = 1.0f - (2.0f * t - 1.0f) * (2.0f * t - 1.0f);
                if (arg < 0.0f) arg = 0.0f;
                arg = alpha * sqrtf(arg);
                float num = 1.0f, denom = 1.0f;
                float term = 1.0f;
                for (int k = 1; k <= 25; k++) {
                    term *= (arg / (float)k) * (arg / (float)k) * 0.25f;
                    num += term;
                }
                arg = alpha;
                term = 1.0f;
                denom = 1.0f;
                for (int k = 1; k <= 25; k++) {
                    term *= (arg / (float)k) * (arg / (float)k) * 0.25f;
                    denom += term;
                }
                w = num / denom;
                break;
            }
            default:
                w = 1.0f;
                break;
        }
        data[i] *= w;
    }
}

static size_t lai_next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void lai_design_butterworth(float* numerator, float* denominator,
                                    int order, float wc, int is_highpass) {
    if (order < 1) order = 1;
    if (order > 4) order = 4;

    memset(numerator, 0, sizeof(float) * (size_t)(order + 1));
    memset(denominator, 0, sizeof(float) * (size_t)(order + 1));

    if (order == 1) {
        float wc_tan = tanf(wc * (float)M_PI / 2.0f);
        float inv_g = 1.0f / (1.0f + wc_tan);
        if (is_highpass) {
            numerator[0] = 1.0f * inv_g;
            numerator[1] = -1.0f * inv_g;
        } else {
            numerator[0] = wc_tan * inv_g;
            numerator[1] = wc_tan * inv_g;
        }
        denominator[0] = 1.0f;
        denominator[1] = (wc_tan - 1.0f) * inv_g;
    } else if (order == 2) {
        float wc_tan = tanf(wc * (float)M_PI / 2.0f);
        float wc_sq = wc_tan * wc_tan;
        float sqrt2 = sqrtf(2.0f);
        float inv_g = 1.0f / (1.0f + sqrt2 * wc_tan + wc_sq);
        if (is_highpass) {
            numerator[0] = 1.0f * inv_g;
            numerator[1] = -2.0f * inv_g;
            numerator[2] = 1.0f * inv_g;
        } else {
            numerator[0] = wc_sq * inv_g;
            numerator[1] = 2.0f * wc_sq * inv_g;
            numerator[2] = wc_sq * inv_g;
        }
        denominator[0] = 1.0f;
        denominator[1] = (2.0f * wc_sq - 2.0f) * inv_g;
        denominator[2] = (1.0f - sqrt2 * wc_tan + wc_sq) * inv_g;
    } else {
        float wc_tan = tanf(wc * (float)M_PI / 2.0f);
        float wc_sq = wc_tan * wc_tan;
        float inv_g = 1.0f / (1.0f + 2.613f * wc_tan + 3.414f * wc_sq + 2.613f * wc_tan * wc_sq + wc_sq * wc_sq);
        if (is_highpass) {
            numerator[0] = 1.0f * inv_g;
            numerator[1] = -4.0f * inv_g;
            numerator[2] = 6.0f * inv_g;
            numerator[3] = -4.0f * inv_g;
            numerator[4] = 1.0f * inv_g;
        } else {
            numerator[0] = wc_sq * wc_sq * inv_g;
            numerator[1] = 4.0f * wc_sq * wc_sq * inv_g;
            numerator[2] = 6.0f * wc_sq * wc_sq * inv_g;
            numerator[3] = 4.0f * wc_sq * wc_sq * inv_g;
            numerator[4] = wc_sq * wc_sq * inv_g;
        }
        denominator[0] = 1.0f;
        denominator[1] = (4.0f * wc_sq * wc_sq + 2.0f * 2.613f * wc_tan * wc_sq - 4.0f) * inv_g;
        denominator[2] = (6.0f * wc_sq * wc_sq - 2.0f * 3.414f * wc_sq + 6.0f) * inv_g;
        denominator[3] = (4.0f * wc_sq * wc_sq - 2.0f * 2.613f * wc_tan * wc_sq - 4.0f) * inv_g;
        denominator[4] = (1.0f - 2.613f * wc_tan + 3.414f * wc_sq - 2.613f * wc_tan * wc_sq + wc_sq * wc_sq) * inv_g;
    }
}

/* M-020修复：标准直接形式II转置 IIR滤波器
 * 差分方程: y[n] = num[0]*x[n] + ... + num[order]*x[n-order]
 *                   - den[1]*y[n-1] - ... - den[order]*y[n-order]
 * 状态变量s[k]保存的是x[n-k]和y[n-k]的混合延迟项
 * 直接形式II转置: w[n] = x[n] - den[1]*w[n-1] - ... - den[order]*w[n-order]
 *                 y[n] = num[0]*w[n] + num[1]*w[n-1] + ... + num[order]*w[n-order] */
static void lai_apply_iir(float* signal, size_t length,
                           const float* numerator, const float* denominator,
                           int order, float* state) {
    /* state[0..order-1] 保存 w[n-1]..w[n-order] */
    for (size_t n = 0; n < length; n++) {
        float x = signal[n];
        /* 计算内部状态 w[n] = x - Σ den[k]*w[n-k] (k>=1) */
        float w = x;
        for (int k = 1; k <= order; k++) {
            w -= denominator[k] * state[k - 1];
        }
        /* 计算输出 y[n] = Σ num[k]*w[n-k] */
        float y = numerator[0] * w;
        for (int k = 1; k <= order; k++) {
            y += numerator[k] * state[k - 1];
        }
        /* 更新延迟线：右移，state[0]=w */
        for (int k = order - 1; k >= 1; k--) {
            state[k] = state[k - 1];
        }
        if (order > 0) state[0] = w;
        signal[n] = y;
    }
}

LaplaceAI* laplace_ai_create(const LaplaceAIConfig* config) {
    LaplaceAI* ai = (LaplaceAI*)calloc(1, sizeof(LaplaceAI));
    if (!ai) return NULL;

    if (config) {
        memcpy(&ai->config, config, sizeof(LaplaceAIConfig));
    } else {
        LaplaceAIConfig default_cfg = LAPLACE_AI_CONFIG_DEFAULT_INITIALIZER;
        memcpy(&ai->config, &default_cfg, sizeof(LaplaceAIConfig));
    }

    if (ai->config.fft_size < 16) ai->config.fft_size = 16;
    if (ai->config.fft_size > LAI_SPECTRUM_SIZE) ai->config.fft_size = LAI_SPECTRUM_SIZE;
    if (ai->config.sampling_rate < 1.0f) ai->config.sampling_rate = 1.0f;

    LaplaceConfig lcfg;
    memset(&lcfg, 0, sizeof(LaplaceConfig));
    lcfg.num_samples = ai->config.fft_size;
    lcfg.sample_rate = ai->config.sampling_rate;
    lcfg.max_frequency = ai->config.sampling_rate / 2.0f;
    lcfg.min_frequency = 0.1f;
    lcfg.enable_stability = 1;
    lcfg.enable_frequency = 1;
    lcfg.enable_optimization = ai->config.enable_adaptive_filter;
    ai->analyzer = laplace_analyzer_create(&lcfg);

    ai->work_buffer = (float*)calloc(ai->config.fft_size * 2, sizeof(float));
    ai->filter_kernel_real = (float*)calloc(ai->config.fft_size, sizeof(float));
    ai->filter_kernel_imag = (float*)calloc(ai->config.fft_size, sizeof(float));
    ai->kernel_size = ai->config.fft_size;
    ai->prev_lr = 1.0;
    ai->lr_momentum = 0.0;
    ai->is_initialized = 1;

    return ai;
}

void laplace_ai_free(LaplaceAI* ai) {
    if (!ai) return;
    if (ai->analyzer) laplace_analyzer_free(ai->analyzer);
    free(ai->work_buffer);
    free(ai->filter_kernel_real);
    free(ai->filter_kernel_imag);
    free(ai);
}

int laplace_ai_transform(LaplaceAI* ai, const float* input,
                          size_t input_length, LAITransformBuffer* buffer) {
    if (!ai || !input || !buffer || input_length == 0) return -1;
    double t0 = (double)clock() / CLOCKS_PER_SEC * 1000.0;

    size_t fft_n = lai_next_pow2(input_length);
    if (fft_n > LAI_SPECTRUM_SIZE) fft_n = LAI_SPECTRUM_SIZE;

    buffer->signal_length = input_length;
    buffer->spectrum_size = fft_n / 2 + 1;

    buffer->time_domain = (float*)calloc(fft_n, sizeof(float));
    buffer->frequency_domain_real = (float*)calloc(fft_n, sizeof(float));
    buffer->frequency_domain_imag = (float*)calloc(fft_n, sizeof(float));
    buffer->magnitude_spectrum = (float*)calloc(fft_n / 2 + 1, sizeof(float));
    buffer->phase_spectrum = (float*)calloc(fft_n / 2 + 1, sizeof(float));

    if (!buffer->time_domain || !buffer->frequency_domain_real ||
        !buffer->frequency_domain_imag || !buffer->magnitude_spectrum ||
        !buffer->phase_spectrum) {
        laplace_ai_transform_buffer_free(buffer);
        return -2;
    }

    size_t copy_len = input_length < fft_n ? input_length : fft_n;
    memcpy(buffer->time_domain, input, copy_len * sizeof(float));

    lai_apply_window(buffer->time_domain, copy_len, ai->config.window_type, ai->config.kaiser_beta);

    memcpy(buffer->frequency_domain_real, buffer->time_domain, fft_n * sizeof(float));
    memset(buffer->frequency_domain_imag, 0, fft_n * sizeof(float));

    lai_fft_radix2(buffer->frequency_domain_real, buffer->frequency_domain_imag, fft_n, 0);

    for (size_t i = 0; i < fft_n / 2 + 1; i++) {
        buffer->magnitude_spectrum[i] = sqrtf(
            buffer->frequency_domain_real[i] * buffer->frequency_domain_real[i] +
            buffer->frequency_domain_imag[i] * buffer->frequency_domain_imag[i]);
        buffer->phase_spectrum[i] = atan2f(buffer->frequency_domain_imag[i],
                                            buffer->frequency_domain_real[i]);
    }

    buffer->transform_time_ms = (double)clock() / CLOCKS_PER_SEC * 1000.0 - t0;
    return 0;
}

int laplace_ai_inverse_transform(LaplaceAI* ai, const LAITransformBuffer* buffer,
                                  float* output, size_t output_length) {
    if (!ai || !buffer || !output || output_length == 0) return -1;

    size_t fft_n = (buffer->spectrum_size - 1) * 2;
    if (fft_n > LAI_SPECTRUM_SIZE) return -1;

    float* real = (float*)calloc(fft_n, sizeof(float));
    float* imag = (float*)calloc(fft_n, sizeof(float));
    if (!real || !imag) { free(real); free(imag); return -2; }

    memcpy(real, buffer->frequency_domain_real, fft_n * sizeof(float));
    memcpy(imag, buffer->frequency_domain_imag, fft_n * sizeof(float));

    lai_fft_radix2(real, imag, fft_n, 1);

    size_t out_len = output_length < buffer->signal_length ? output_length : buffer->signal_length;
    for (size_t i = 0; i < out_len; i++) {
        output[i] = real[i];
    }

    free(real);
    free(imag);
    return 0;
}

void laplace_ai_transform_buffer_free(LAITransformBuffer* buffer) {
    if (!buffer) return;
    free(buffer->time_domain);
    free(buffer->frequency_domain_real);
    free(buffer->frequency_domain_imag);
    free(buffer->magnitude_spectrum);
    free(buffer->phase_spectrum);
    memset(buffer, 0, sizeof(LAITransformBuffer));
}

int laplace_ai_extract_features(LaplaceAI* ai, const float* signal,
                                  size_t signal_length, LAISpectrumResult* result) {
    if (!ai || !signal || !result || signal_length == 0) return -1;
    memset(result, 0, sizeof(LAISpectrumResult));

    double t0 = (double)clock() / CLOCKS_PER_SEC * 1000.0;

    size_t fft_n = lai_next_pow2(signal_length);
    if (fft_n > LAI_SPECTRUM_SIZE) fft_n = LAI_SPECTRUM_SIZE;
    size_t num_bins = fft_n / 2 + 1;

    result->magnitude = (float*)calloc(num_bins, sizeof(float));
    result->phase = (float*)calloc(num_bins, sizeof(float));
    result->frequency = (float*)calloc(num_bins, sizeof(float));
    if (!result->magnitude || !result->phase || !result->frequency) {
        laplace_ai_spectrum_free(result);
        return -2;
    }

    result->num_bins = num_bins;
    result->sampling_rate = ai->config.sampling_rate;

    float* real = (float*)calloc(fft_n, sizeof(float));
    float* imag = (float*)calloc(fft_n, sizeof(float));
    if (!real || !imag) { free(real); free(imag); return -3; }

    size_t copy_len = signal_length < fft_n ? signal_length : fft_n;
    memcpy(real, signal, copy_len * sizeof(float));
    lai_apply_window(real, copy_len, ai->config.window_type, ai->config.kaiser_beta);
    lai_fft_radix2(real, imag, fft_n, 0);

    float total_energy = 0.0f;
    float weighted_freq_sum = 0.0f;
    float mag_sum = 0.0f;
    float rolloff_threshold = 0.85f;
    float total_energy_for_rolloff = 0.0f;

    for (size_t i = 0; i < num_bins; i++) {
        float mag = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
        float freq = (float)i * ai->config.sampling_rate / (float)fft_n;
        result->magnitude[i] = mag;
        result->phase[i] = atan2f(imag[i], real[i]);
        result->frequency[i] = freq;
        total_energy += mag * mag;
        weighted_freq_sum += freq * mag;
        mag_sum += mag;
    }

    total_energy_for_rolloff = total_energy;
    result->total_energy = total_energy;

    if (mag_sum > 1e-10f) {
        result->spectral_centroid = weighted_freq_sum / mag_sum;
    }

    float cum_energy = 0.0f;
    result->spectral_rolloff = result->frequency[num_bins - 1];
    for (size_t i = 0; i < num_bins; i++) {
        cum_energy += result->magnitude[i] * result->magnitude[i];
        if (cum_energy >= rolloff_threshold * total_energy_for_rolloff) {
            result->spectral_rolloff = result->frequency[i];
            break;
        }
    }

    float max_mag = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 1; i < num_bins; i++) {
        if (result->magnitude[i] > max_mag) {
            max_mag = result->magnitude[i];
            max_idx = i;
        }
    }
    result->dominant_freq = result->frequency[max_idx];

    int num_bands = ai->config.num_feature_bands;
    if (num_bands > 16) num_bands = 16;
    result->num_bands = num_bands;
    float nyquist = ai->config.sampling_rate / 2.0f;

    for (int b = 0; b < num_bands; b++) {
        float band_start = (b == 0) ? 0.0f : ai->config.feature_freq_bands[b - 1];
        float band_end = (b < num_bands - 1) ? ai->config.feature_freq_bands[b] : nyquist;
        float band_energy = 0.0f;
        for (size_t i = 0; i < num_bins; i++) {
            if (result->frequency[i] >= band_start && result->frequency[i] <= band_end) {
                band_energy += result->magnitude[i] * result->magnitude[i];
            }
        }
        result->band_energies[b] = band_energy;
    }

    result->processing_time_ms = (double)clock() / CLOCKS_PER_SEC * 1000.0 - t0;
    free(real);
    free(imag);
    return 0;
}

void laplace_ai_spectrum_free(LAISpectrumResult* result) {
    if (!result) return;
    free(result->magnitude);
    free(result->phase);
    free(result->frequency);
    memset(result, 0, sizeof(LAISpectrumResult));
}

int laplace_ai_filter_signal(LaplaceAI* ai, const float* input,
                              size_t input_length, float* output,
                              LAIFilterType filter_type,
                              float cutoff_low, float cutoff_high) {
    if (!ai || !input || !output || input_length == 0) return -1;

    memcpy(output, input, input_length * sizeof(float));

    if (filter_type == LAI_FILTER_NONE) return 0;

    size_t fft_n = lai_next_pow2(input_length);
    if (fft_n > LAI_SPECTRUM_SIZE) fft_n = LAI_SPECTRUM_SIZE;
    size_t num_bins = fft_n / 2 + 1;

    float* real = (float*)calloc(fft_n, sizeof(float));
    float* imag = (float*)calloc(fft_n, sizeof(float));
    if (!real || !imag) { free(real); free(imag); return -2; }

    memcpy(real, output, (input_length < fft_n ? input_length : fft_n) * sizeof(float));
    lai_fft_radix2(real, imag, fft_n, 0);

    float nyquist = ai->config.sampling_rate / 2.0f;
    int order = ai->config.filter_order;
    if (order < 1) order = 1;
    if (order > 4) order = 4;

    for (size_t i = 0; i < num_bins; i++) {
        float freq = (float)i * ai->config.sampling_rate / (float)fft_n;
        float gain = 1.0f;

        switch (filter_type) {
            case LAI_FILTER_LOWPASS: {
                float ratio = freq / (cutoff_low > 0.0f ? cutoff_low : nyquist);
                gain = 1.0f / sqrtf(1.0f + powf(ratio, (float)(2 * order)));
                break;
            }
            case LAI_FILTER_HIGHPASS: {
                if (cutoff_low > 0.0f && freq > 0.0f) {
                    float ratio = cutoff_low / freq;
                    gain = 1.0f / sqrtf(1.0f + powf(ratio, (float)(2 * order)));
                }
                break;
            }
            case LAI_FILTER_BANDPASS: {
                if (freq > 0.0f && cutoff_low > 0.0f && cutoff_high > cutoff_low) {
                    float ratio_low = freq / cutoff_low;
                    float ratio_high = cutoff_high / freq;
                    gain = 1.0f / sqrtf((1.0f + powf(ratio_low, (float)(2 * order))) *
                                        (1.0f + powf(ratio_high, (float)(2 * order))));
                }
                break;
            }
            case LAI_FILTER_BANDSTOP: {
                if (freq > 0.0f && cutoff_low > 0.0f && cutoff_high > cutoff_low) {
                    float bw = cutoff_high - cutoff_low;
                    float f0 = sqrtf(cutoff_low * cutoff_high);
                    float ratio = (freq * freq - f0 * f0) / (freq * bw);
                    gain = 1.0f / sqrtf(1.0f + ratio * ratio);
                }
                break;
            }
            case LAI_FILTER_NOTCH: {
                if (freq > 0.0f && cutoff_low > 0.0f) {
                    float ratio = freq / cutoff_low - cutoff_low / freq;
                    gain = 1.0f / sqrtf(1.0f + (ratio / 0.1f) * (ratio / 0.1f));
                }
                break;
            }
            case LAI_FILTER_ADAPTIVE: {
                float total_mag = 0.0f;
                for (size_t j = 0; j < num_bins; j++) {
                    total_mag += sqrtf(real[j] * real[j] + imag[j] * imag[j]);
                }
                if (total_mag > 1e-10f) {
                    float this_mag = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
                    float power_ratio = this_mag / total_mag * (float)num_bins;
                    float mid_freq = nyquist * 0.3f;
                    float target = (freq < mid_freq) ? 1.2f : 0.6f;
                    gain = 1.0f + (target - 1.0f) * (1.0f - expf(-power_ratio * 2.0f));
                    if (gain < 0.1f) gain = 0.1f;
                    if (gain > 3.0f) gain = 3.0f;
                }
                break;
            }
            default:
                break;
        }

        real[i] *= gain;
        imag[i] *= gain;
    }

    for (size_t i = 1; i < fft_n / 2; i++) {
        real[fft_n - i] = real[i];
        imag[fft_n - i] = -imag[i];
    }

    lai_fft_radix2(real, imag, fft_n, 1);

    size_t out_len = input_length < fft_n ? input_length : fft_n;
    memcpy(output, real, out_len * sizeof(float));

    free(real);
    free(imag);
    return (int)out_len;
}

int laplace_ai_filter_gradient(LaplaceAI* ai, const float* gradients,
                                size_t num_gradients, float* filtered_gradients) {
    if (!ai || !gradients || !filtered_gradients || num_gradients == 0) return -1;

    float cutoff = ai->config.high_cutoff_hz;
    if (cutoff <= 0.0f) cutoff = ai->config.sampling_rate * 0.3f;

    return laplace_ai_filter_signal(ai, gradients, num_gradients,
                                     filtered_gradients, LAI_FILTER_LOWPASS,
                                     cutoff, 0.0f);
}

float laplace_ai_adaptive_learning_rate(LaplaceAI* ai,
                                          const float* gradient_history,
                                          size_t history_length, float base_lr) {
    if (!ai || !gradient_history || history_length < 4) return base_lr;

    LAISpectrumResult spectrum;
    memset(&spectrum, 0, sizeof(LAISpectrumResult));

    LaplaceAIConfig saved_config = ai->config;
    ai->config.window_type = LAI_WINDOW_HANNING;

    int ret = laplace_ai_extract_features(ai, gradient_history, history_length, &spectrum);
    ai->config = saved_config;

    if (ret != 0) return base_lr;

    float lr_scale = 1.0f;
    float nyquist = ai->config.sampling_rate / 2.0f;
    float low_threshold = nyquist * 0.1f;
    float high_threshold = nyquist * 0.4f;

    float low_energy = 0.0f, mid_energy = 0.0f, high_energy = 0.0f;
    for (size_t i = 0; i < spectrum.num_bins; i++) {
        float energy = spectrum.magnitude[i] * spectrum.magnitude[i];
        if (spectrum.frequency[i] < low_threshold) {
            low_energy += energy;
        } else if (spectrum.frequency[i] < high_threshold) {
            mid_energy += energy;
        } else {
            high_energy += energy;
        }
    }

    float total = low_energy + mid_energy + high_energy;
    if (total > 1e-10f) {
        float high_ratio = high_energy / total;
        float low_ratio = low_energy / total;

        if (high_ratio > 0.5f) {
            lr_scale = 1.0f - (high_ratio - 0.5f) * 2.0f * (1.0f - ai->config.lr_min_scale);
        } else if (low_ratio > 0.6f) {
            lr_scale = 1.0f + (low_ratio - 0.6f) * 2.0f * (ai->config.lr_max_scale - 1.0f);
        }

        if (spectrum.spectral_centroid > high_threshold) {
            lr_scale *= 0.8f;
        } else if (spectrum.spectral_centroid < low_threshold) {
            lr_scale *= 1.1f;
        }
    }

    if (lr_scale < ai->config.lr_min_scale) lr_scale = ai->config.lr_min_scale;
    if (lr_scale > ai->config.lr_max_scale) lr_scale = ai->config.lr_max_scale;

    float alpha = ai->config.lr_adaptation_speed;
    if (alpha < 0.01f) alpha = 0.1f;
    if (alpha > 1.0f) alpha = 1.0f;

    ai->lr_momentum = ai->lr_momentum * (1.0f - alpha) + (double)lr_scale * alpha;
    float smoothed_lr = (float)(base_lr * ai->lr_momentum);

    laplace_ai_spectrum_free(&spectrum);
    return smoothed_lr;
}

int laplace_ai_monitor_stability(LaplaceAI* ai, const float* signal,
                                  size_t signal_length, LAIStabilityReport* report) {
    if (!ai || !signal || !report || signal_length == 0) return -1;
    memset(report, 0, sizeof(LAIStabilityReport));

    LAISpectrumResult spectrum;
    memset(&spectrum, 0, sizeof(LAISpectrumResult));

    int ret = laplace_ai_extract_features(ai, signal, signal_length, &spectrum);
    if (ret != 0) return ret;

    float total_energy = spectrum.total_energy;
    if (total_energy < 1e-10f) {
        report->stability_score = 1.0f;
        report->is_stable = 1;
        report->warning_level = 0;
        snprintf(report->description, sizeof(report->description), "信号能量过低，无法分析");
        laplace_ai_spectrum_free(&spectrum);
        return 0;
    }

    float nyquist = ai->config.sampling_rate / 2.0f;
    float low_cutoff = nyquist * 0.05f;
    float high_cutoff = nyquist * 0.4f;

    float low_energy = 0.0f, mid_energy = 0.0f, high_energy = 0.0f;
    float prev_mag = 0.0f;
    int zero_crossings = 0;

    for (size_t i = 0; i < spectrum.num_bins; i++) {
        if (i > 0 && prev_mag > 0.0f && spectrum.magnitude[i] > 0.0f) {
            float ratio = spectrum.magnitude[i] / prev_mag;
            if (ratio > 3.0f) zero_crossings++;
        }
        prev_mag = spectrum.magnitude[i];

        float energy = spectrum.magnitude[i] * spectrum.magnitude[i];
        if (spectrum.frequency[i] < low_cutoff) {
            low_energy += energy;
        } else if (spectrum.frequency[i] < high_cutoff) {
            mid_energy += energy;
        } else {
            high_energy += energy;
        }
    }

    float high_ratio = (total_energy > 1e-10f) ? high_energy / total_energy : 0.0f;
    float low_ratio = (total_energy > 1e-10f) ? low_energy / total_energy : 0.0f;

    report->noise_level = high_ratio;
    report->oscillation_index = low_ratio;

    float stability = 1.0f;
    stability -= high_ratio * 0.5f;
    stability -= low_ratio * 0.3f;

    if (spectrum.spectral_centroid > 0.0f) {
        float centroid_ratio = spectrum.spectral_centroid / nyquist;
        if (centroid_ratio > 0.5f) {
            stability -= (centroid_ratio - 0.5f) * 0.3f;
        } else if (centroid_ratio < 0.1f) {
            stability -= (0.1f - centroid_ratio) * 0.2f;
        }
    }

    if (zero_crossings > (int)spectrum.num_bins / 4) {
        stability -= 0.15f;
    }

    if (stability < 0.0f) stability = 0.0f;
    if (stability > 1.0f) stability = 1.0f;
    report->stability_score = stability;

    report->dominant_pole_real = -(1.0f - stability) * 2.0f;
    report->phase_margin = stability * 60.0f;
    report->gain_margin = stability * 20.0f;

    if (stability > 0.7f) {
        report->is_stable = 1;
        report->warning_level = 0;
        snprintf(report->description, sizeof(report->description), "系统稳定 (分数: %.2f)", stability);
    } else if (stability > 0.4f) {
        report->is_stable = 1;
        report->warning_level = 1;
        snprintf(report->description, sizeof(report->description),
                 "系统临界稳定 - 高频噪声: %.1f%%, 低频振荡: %.1f%%",
                 high_ratio * 100.0f, low_ratio * 100.0f);
    } else {
        report->is_stable = 0;
        report->warning_level = 2;
        snprintf(report->description, sizeof(report->description),
                 "系统不稳定! 高频噪声: %.1f%%, 低频振荡: %.1f%%",
                 high_ratio * 100.0f, low_ratio * 100.0f);
    }

    laplace_ai_spectrum_free(&spectrum);
    return 0;
}

int laplace_ai_frequency_reason(LaplaceAI* ai, const float* input_signal,
                                 size_t input_length, float* output_signal,
                                 size_t output_length, const float* target_spectrum,
                                 size_t spectrum_length) {
    if (!ai || !input_signal || !output_signal || !target_spectrum ||
        input_length == 0 || output_length == 0 || spectrum_length == 0) return -1;

    size_t fft_n = lai_next_pow2(input_length > output_length ? input_length : output_length);
    if (fft_n > LAI_SPECTRUM_SIZE) fft_n = LAI_SPECTRUM_SIZE;

    float* real = (float*)calloc(fft_n, sizeof(float));
    float* imag = (float*)calloc(fft_n, sizeof(float));
    if (!real || !imag) { free(real); free(imag); return -2; }

    size_t copy_len = input_length < fft_n ? input_length : fft_n;
    memcpy(real, input_signal, copy_len * sizeof(float));

    lai_apply_window(real, copy_len, LAI_WINDOW_HANNING, 0.0f);
    lai_fft_radix2(real, imag, fft_n, 0);

    size_t num_bins = fft_n / 2 + 1;
    size_t target_bins = spectrum_length < num_bins ? spectrum_length : num_bins;

    for (size_t i = 0; i < target_bins; i++) {
        float current_mag = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
        if (current_mag > 1e-10f) {
            float scale = target_spectrum[i] / current_mag;
            if (scale > 5.0f) scale = 5.0f;
            if (scale < 0.01f) scale = 0.01f;
            real[i] *= scale;
            imag[i] *= scale;
        }
    }

    for (size_t i = 1; i < fft_n / 2; i++) {
        real[fft_n - i] = real[i];
        imag[fft_n - i] = -imag[i];
    }

    lai_fft_radix2(real, imag, fft_n, 1);

    size_t out_len = output_length < fft_n ? output_length : fft_n;
    memcpy(output_signal, real, out_len * sizeof(float));

    free(real);
    free(imag);
    return (int)out_len;
}

int laplace_ai_set_config(LaplaceAI* ai, const LaplaceAIConfig* config) {
    if (!ai || !config) return -1;
    memcpy(&ai->config, config, sizeof(LaplaceAIConfig));
    if (ai->analyzer) {
        LaplaceConfig lcfg;
        memset(&lcfg, 0, sizeof(LaplaceConfig));
        lcfg.num_samples = ai->config.fft_size;
        lcfg.sample_rate = ai->config.sampling_rate;
        lcfg.max_frequency = ai->config.sampling_rate / 2.0f;
        lcfg.min_frequency = 0.1f;
        lcfg.enable_stability = 1;
        lcfg.enable_frequency = 1;
        lcfg.enable_optimization = ai->config.enable_adaptive_filter;
        laplace_analyzer_set_config(ai->analyzer, &lcfg);
    }
    return 0;
}

int laplace_ai_get_config(const LaplaceAI* ai, LaplaceAIConfig* config) {
    if (!ai || !config) return -1;
    memcpy(config, &ai->config, sizeof(LaplaceAIConfig));
    return 0;
}

void laplace_ai_reset(LaplaceAI* ai) {
    if (!ai) return;
    ai->prev_lr = 1.0;
    ai->lr_momentum = 0.0;
    if (ai->analyzer) laplace_analyzer_reset(ai->analyzer);
}

/* ============================================================================
 * 多通道滤波器组设计与子带分解
 *
 * 实现分析滤波器组，将信号分解为多个子带，每个子带可独立处理。
 * 支持均匀DFT滤波器组和临界采样的余弦调制滤波器组。
 * =========================================================================== */

typedef struct {
    float* filter_coeffs;          /**< 原型滤波器系数 */
    size_t filter_length;          /**< 滤波器长度 */
    int num_subbands;              /**< 子带数量 */
    float* subband_energies;       /**< 各子带能量 */
    float* subband_centroids;      /**< 各子带频谱质心 */
    int is_initialized;
} LAIFilterBank;

static int lai_filter_bank_design(LAIFilterBank* fb, int num_subbands,
                                   size_t filter_length, float transition_bw) {
    if (!fb || num_subbands < 2 || num_subbands > LAI_MAX_FILTER_BANDS) return -1;
    if (filter_length < 4) filter_length = 4;
    if (filter_length > 256) filter_length = 256;
    if (transition_bw <= 0.0f) transition_bw = 0.1f;
    if (transition_bw > 0.5f) transition_bw = 0.5f;

    fb->num_subbands = num_subbands;
    fb->filter_length = filter_length;
    fb->filter_coeffs = (float*)calloc(filter_length, sizeof(float));
    fb->subband_energies = (float*)calloc((size_t)num_subbands, sizeof(float));
    fb->subband_centroids = (float*)calloc((size_t)num_subbands, sizeof(float));
    if (!fb->filter_coeffs || !fb->subband_energies || !fb->subband_centroids) {
        free(fb->filter_coeffs); free(fb->subband_energies); free(fb->subband_centroids);
        memset(fb, 0, sizeof(LAIFilterBank));
        return -2;
    }

    /* 设计低通原型滤波器：使用Kaiser窗设计的sinc滤波器 */
    float wc = (1.0f - transition_bw) * 0.5f;
    for (size_t n = 0; n < filter_length; n++) {
        float t = (float)n - (float)(filter_length - 1) / 2.0f;
        if (fabsf(t) < 1e-10f) {
            fb->filter_coeffs[n] = 2.0f * wc;
        } else {
            fb->filter_coeffs[n] = sinf(2.0f * (float)M_PI * wc * t) / ((float)M_PI * t);
        }
        /* 应用Kaiser窗 */
        float beta = 5.0f;
        float arg = 1.0f - ((2.0f * (float)n / (float)(filter_length - 1)) - 1.0f) *
                           ((2.0f * (float)n / (float)(filter_length - 1)) - 1.0f);
        if (arg < 0.0f) arg = 0.0f;
        float window = 1.0f;
        float iarg = beta * sqrtf(arg);
        float num = 1.0f, denom = 1.0f, term = 1.0f;
        for (int k = 1; k <= 25; k++) {
            term *= (iarg / (float)k) * (iarg / (float)k) * 0.25f;
            num += term;
        }
        float iarg2 = beta;
        term = 1.0f; denom = 1.0f;
        for (int k = 1; k <= 25; k++) {
            term *= (iarg2 / (float)k) * (iarg2 / (float)k) * 0.25f;
            denom += term;
        }
        window = num / denom;
        fb->filter_coeffs[n] *= window;
    }

    fb->is_initialized = 1;
    return 0;
}

static void lai_filter_bank_free(LAIFilterBank* fb) {
    if (!fb) return;
    free(fb->filter_coeffs);
    free(fb->subband_energies);
    free(fb->subband_centroids);
    memset(fb, 0, sizeof(LAIFilterBank));
}

static int lai_filter_bank_analyze(LAIFilterBank* fb, const float* signal,
                                    size_t signal_length, float** subband_outputs,
                                    size_t* subband_lengths) {
    if (!fb || !fb->is_initialized || !signal || signal_length == 0) return -1;
    if (!subband_outputs || !subband_lengths) return -1;

    int M = fb->num_subbands;
    size_t L = fb->filter_length;
    size_t out_len = signal_length;

    for (int k = 0; k < M; k++) {
        subband_outputs[k] = (float*)calloc(out_len, sizeof(float));
        if (!subband_outputs[k]) {
            for (int j = 0; j < k; j++) {
                free(subband_outputs[j]); subband_outputs[j] = NULL;
            }
            return -2;
        }
        subband_lengths[k] = out_len;

        /* 余弦调制: h_k[n] = h[n] * cos(pi * (k+0.5) * n / M) */
        for (size_t n = 0; n < L && n < out_len; n++) {
            float modulation = cosf((float)M_PI * ((float)k + 0.5f) * (float)n / (float)M);
            float h = fb->filter_coeffs[n] * modulation;
            for (size_t t = n; t < out_len; t++) {
                subband_outputs[k][t] += h * signal[t - n];
            }
        }

        /* 计算子带能量和频谱质心 */
        float energy = 0.0f;
        for (size_t t = 0; t < out_len; t++) {
            energy += subband_outputs[k][t] * subband_outputs[k][t];
        }
        fb->subband_energies[k] = energy / (float)out_len;

        float weighted_sum = 0.0f, mag_sum = 0.0f;
        for (size_t t = 1; t < out_len; t++) {
            float mag = fabsf(subband_outputs[k][t]);
            weighted_sum += (float)t * mag;
            mag_sum += mag;
        }
        fb->subband_centroids[k] = (mag_sum > 1e-10f) ? weighted_sum / mag_sum : 0.0f;
    }

    return 0;
}

/* ============================================================================
 * 自适应谱减法降噪
 *
 * 在频域估计噪声谱并从信号中减去，同时避免音乐噪声。
 * 使用过减法因子和谱底噪保护。
 * =========================================================================== */

#define LAI_NOISE_BUFFER_FRAMES 32

typedef struct {
    float* noise_spectrum;         /**< 估计的噪声谱 */
    float* noise_buffer[LAI_NOISE_BUFFER_FRAMES]; /**< 噪声估计缓冲区 */
    int buffer_index;
    int buffer_filled;
    float over_subtraction_factor; /**< 过减法因子 (1.0-3.0) */
    float spectral_floor;          /**< 谱底噪保护 (0.01-0.1) */
    int fft_size;
    int is_initialized;
} LAISpectralSubtractor;

static int lai_spectral_subtractor_init(LAISpectralSubtractor* ss, int fft_size) {
    if (!ss || fft_size < 16) return -1;

    memset(ss, 0, sizeof(LAISpectralSubtractor));
    ss->fft_size = fft_size;
    ss->over_subtraction_factor = 2.0f;
    ss->spectral_floor = 0.05f;

    ss->noise_spectrum = (float*)calloc((size_t)fft_size / 2 + 1, sizeof(float));
    if (!ss->noise_spectrum) return -2;

    for (int i = 0; i < LAI_NOISE_BUFFER_FRAMES; i++) {
        ss->noise_buffer[i] = (float*)calloc((size_t)fft_size / 2 + 1, sizeof(float));
        if (!ss->noise_buffer[i]) {
            for (int j = 0; j < i; j++) free(ss->noise_buffer[j]);
            free(ss->noise_spectrum);
            return -2;
        }
    }

    ss->is_initialized = 1;
    return 0;
}

static void lai_spectral_subtractor_free(LAISpectralSubtractor* ss) {
    if (!ss) return;
    free(ss->noise_spectrum);
    for (int i = 0; i < LAI_NOISE_BUFFER_FRAMES; i++) {
        free(ss->noise_buffer[i]);
    }
    memset(ss, 0, sizeof(LAISpectralSubtractor));
}

static int lai_spectral_subtractor_update_noise(LAISpectralSubtractor* ss,
                                                  const float* magnitude_spectrum,
                                                  size_t num_bins) {
    if (!ss || !ss->is_initialized || !magnitude_spectrum) return -1;
    if (num_bins < (size_t)ss->fft_size / 2 + 1) return -1;

    size_t idx = (size_t)ss->buffer_index * ((size_t)ss->fft_size / 2 + 1);
    if (idx + num_bins <= (size_t)LAI_NOISE_BUFFER_FRAMES * ((size_t)ss->fft_size / 2 + 1)) {
        memcpy(ss->noise_buffer[ss->buffer_index], magnitude_spectrum,
               num_bins * sizeof(float));
    }

    ss->buffer_index = (ss->buffer_index + 1) % LAI_NOISE_BUFFER_FRAMES;
    if (!ss->buffer_filled && ss->buffer_index == 0) {
        ss->buffer_filled = 1;
    }

    int n_frames = ss->buffer_filled ? LAI_NOISE_BUFFER_FRAMES : ss->buffer_index;
    if (n_frames == 0) return 0;

    memset(ss->noise_spectrum, 0, num_bins * sizeof(float));
    for (int i = 0; i < n_frames; i++) {
        for (size_t b = 0; b < num_bins; b++) {
            ss->noise_spectrum[b] += ss->noise_buffer[i][b];
        }
    }
    float inv_n = 1.0f / (float)n_frames;
    for (size_t b = 0; b < num_bins; b++) {
        ss->noise_spectrum[b] *= inv_n;
    }

    return 0;
}

static int lai_spectral_subtractor_apply(LAISpectralSubtractor* ss,
                                          float* real, float* imag,
                                          size_t fft_size, int update_noise) {
    if (!ss || !ss->is_initialized || !real || !imag) return -1;

    size_t num_bins = fft_size / 2 + 1;
    if (num_bins > (size_t)ss->fft_size / 2 + 1) num_bins = (size_t)ss->fft_size / 2 + 1;

    float* magnitude = (float*)calloc(num_bins, sizeof(float));
    if (!magnitude) return -2;

    for (size_t i = 0; i < num_bins; i++) {
        magnitude[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
    }

    if (update_noise) {
        lai_spectral_subtractor_update_noise(ss, magnitude, num_bins);
    }

    float alpha = ss->over_subtraction_factor;
    float floor_val = ss->spectral_floor;

    for (size_t i = 0; i < num_bins; i++) {
        float noise_mag = ss->noise_spectrum[i];
        if (noise_mag > magnitude[i]) noise_mag = magnitude[i] * 0.5f;

        float subtracted = magnitude[i] - alpha * noise_mag;
        if (subtracted < floor_val * magnitude[i]) {
            subtracted = floor_val * magnitude[i];
        }

        float gain = (magnitude[i] > 1e-10f) ? subtracted / magnitude[i] : floor_val;
        if (gain < 0.0f) gain = 0.0f;
        if (gain > 1.0f) gain = 1.0f;

        /* 谱平滑减少音乐噪声 */
        float smooth_factor = 0.8f;
        static LAI_THREAD_LOCAL float prev_gain[LAI_SPECTRUM_SIZE / 2 + 1];
        static LAI_THREAD_LOCAL int prev_gain_init = 0;
        if (!prev_gain_init) {
            memset(prev_gain, 0, sizeof(prev_gain));
            prev_gain_init = 1;
        }
        if (i < LAI_SPECTRUM_SIZE / 2 + 1) {
            gain = smooth_factor * prev_gain[i] + (1.0f - smooth_factor) * gain;
            prev_gain[i] = gain;
        }

        real[i] *= gain;
        imag[i] *= gain;
    }

    free(magnitude);
    return 0;
}

/* ============================================================================
 * 倒谱分析与MFCC特征提取
 *
 * 实现：
 * - 复倒谱 (Complex Cepstrum)：通过复对数FFT计算
 * - 实倒谱 (Real Cepstrum)：通过对数幅度谱的IFFT计算
 * - MFCC (Mel频率倒谱系数)：三角滤波器组 + DCT
 * - 谱包络估计：低时域倒谱滤波
 * =========================================================================== */

typedef struct {
    float* mel_filterbank;         /**< Mel三角滤波器组 */
    int num_mel_filters;           /**< Mel滤波器数量 (默认26) */
    int num_ceps_coeffs;           /**< 倒谱系数数量 (默认13) */
    int fft_size;
    float sampling_rate;
    float* dct_matrix;             /**< DCT变换矩阵 */
    int is_initialized;
} LAICepstralAnalyzer;

static int lai_cepstral_init(LAICepstralAnalyzer* ca, int fft_size,
                              float sampling_rate, int num_filters, int num_coeffs) {
    if (!ca || fft_size < 16) return -1;
    if (num_filters < 4) num_filters = 26;
    if (num_filters > 64) num_filters = 64;
    if (num_coeffs < 2) num_coeffs = 13;
    if (num_coeffs > num_filters) num_coeffs = num_filters;

    memset(ca, 0, sizeof(LAICepstralAnalyzer));
    ca->fft_size = fft_size;
    ca->sampling_rate = sampling_rate;
    ca->num_mel_filters = num_filters;
    ca->num_ceps_coeffs = num_coeffs;

    size_t num_bins = (size_t)fft_size / 2 + 1;
    ca->mel_filterbank = (float*)calloc(num_bins * (size_t)num_filters, sizeof(float));
    ca->dct_matrix = (float*)calloc((size_t)num_filters * (size_t)num_coeffs, sizeof(float));
    if (!ca->mel_filterbank || !ca->dct_matrix) {
        free(ca->mel_filterbank); free(ca->dct_matrix);
        memset(ca, 0, sizeof(LAICepstralAnalyzer));
        return -2;
    }

    /* 设计Mel三角滤波器组 */
    float mel_low = 0.0f;
    float mel_high = 2595.0f * log10f(1.0f + sampling_rate / 1400.0f);
    float mel_step = (mel_high - mel_low) / (float)(num_filters + 1);

    for (int m = 0; m < num_filters; m++) {
        float mel_center = mel_low + (float)(m + 1) * mel_step;
        float freq_center = 700.0f * (powf(10.0f, mel_center / 2595.0f) - 1.0f);
        float mel_left = mel_low + (float)m * mel_step;
        float mel_right = mel_low + (float)(m + 2) * mel_step;
        float freq_left = 700.0f * (powf(10.0f, mel_left / 2595.0f) - 1.0f);
        float freq_right = 700.0f * (powf(10.0f, mel_right / 2595.0f) - 1.0f);

        for (size_t b = 0; b < num_bins; b++) {
            float freq = (float)b * sampling_rate / (float)fft_size;
            float weight = 0.0f;
            if (freq >= freq_left && freq <= freq_center) {
                if (freq_center > freq_left) {
                    weight = (freq - freq_left) / (freq_center - freq_left);
                }
            } else if (freq > freq_center && freq <= freq_right) {
                if (freq_right > freq_center) {
                    weight = (freq_right - freq) / (freq_right - freq_center);
                }
            }
            ca->mel_filterbank[m * num_bins + b] = weight;
        }
    }

    /* 构建DCT-II矩阵 (用于MFCC提取) */
    for (int i = 0; i < num_coeffs; i++) {
        for (int j = 0; j < num_filters; j++) {
            float scale = (i == 0) ? sqrtf(1.0f / (float)num_filters)
                                   : sqrtf(2.0f / (float)num_filters);
            ca->dct_matrix[i * num_filters + j] = scale *
                cosf((float)M_PI * (float)i * ((float)j + 0.5f) / (float)num_filters);
        }
    }

    ca->is_initialized = 1;
    return 0;
}

static void lai_cepstral_free(LAICepstralAnalyzer* ca) {
    if (!ca) return;
    free(ca->mel_filterbank);
    free(ca->dct_matrix);
    memset(ca, 0, sizeof(LAICepstralAnalyzer));
}

static int lai_extract_mfcc(LAICepstralAnalyzer* ca, const float* magnitude_spectrum,
                             size_t num_bins, float* mfcc_output) {
    if (!ca || !ca->is_initialized || !magnitude_spectrum || !mfcc_output) return -1;

    size_t expected_bins = (size_t)ca->fft_size / 2 + 1;
    if (num_bins < expected_bins) return -1;

    /* 步骤1: 计算每个Mel滤波器的对数能量 */
    float mel_energies[64];
    for (int m = 0; m < ca->num_mel_filters; m++) {
        float energy = 0.0f;
        for (size_t b = 0; b < expected_bins; b++) {
            float mag = magnitude_spectrum[b];
            energy += ca->mel_filterbank[m * expected_bins + b] * (mag * mag);
        }
        if (energy < 1e-10f) energy = 1e-10f;
        mel_energies[m] = logf(energy);
    }

    /* 步骤2: DCT得到MFCC */
    for (int i = 0; i < ca->num_ceps_coeffs; i++) {
        float sum = 0.0f;
        for (int j = 0; j < ca->num_mel_filters; j++) {
            sum += ca->dct_matrix[i * ca->num_mel_filters + j] * mel_energies[j];
        }
        mfcc_output[i] = sum;
    }

    return ca->num_ceps_coeffs;
}

static int lai_compute_real_cepstrum(const float* magnitude_spectrum,
                                      size_t num_bins, float* cepstrum,
                                      size_t cepstrum_length) {
    if (!magnitude_spectrum || !cepstrum || num_bins == 0) return -1;

    for (size_t i = 0; i < num_bins; i++) {
        float val = magnitude_spectrum[i];
        if (val < 1e-20f) val = 1e-20f;
        cepstrum[i] = logf(val);
    }

    /* 简单IFFT (DFT近似) 计算倒谱 */
    size_t fft_n = 1;
    while (fft_n < num_bins * 2) fft_n <<= 1;
    if (fft_n > LAI_SPECTRUM_SIZE) fft_n = LAI_SPECTRUM_SIZE;

    float* real = (float*)calloc(fft_n, sizeof(float));
    float* imag = (float*)calloc(fft_n, sizeof(float));
    if (!real || !imag) { free(real); free(imag); return -2; }

    memcpy(real, cepstrum, num_bins * sizeof(float));
    lai_fft_radix2(real, imag, fft_n, 1);

    size_t out_len = cepstrum_length < fft_n ? cepstrum_length : fft_n;
    for (size_t i = 0; i < out_len; i++) {
        float val = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
        cepstrum[i] = val;
    }

    free(real);
    free(imag);
    return (int)out_len;
}

/* ============================================================================
 * 根轨迹分析和Nyquist稳定性判据
 *
 * 通过计算传递函数的极点和零点分布，评估闭环系统的稳定性。
 * 使用S平面到Z平面的双线性变换分析。
 * =========================================================================== */

static int lai_root_locus_analysis(const float* numerator, const float* denominator,
                                    int order, float* dominant_pole_real,
                                    float* dominant_pole_imag,
                                    float* damping_ratio) {
    if (!numerator || !denominator || order < 1 || order > 4) return -1;
    if (!dominant_pole_real || !dominant_pole_imag || !damping_ratio) return -1;

    *dominant_pole_real = 0.0f;
    *dominant_pole_imag = 0.0f;
    *damping_ratio = 1.0f;

    float max_mag = 0.0f;
    int found_dominant = 0;

    /* 对2阶系统使用解析解，更高阶使用近似 */
    if (order <= 2) {
        float a = denominator[0];
        float b = denominator[1];
        float c = denominator[2];
        if (fabsf(a) < 1e-10f) return -1;

        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float p1 = (-b + sqrtf(disc)) / (2.0f * a);
            float p2 = (-b - sqrtf(disc)) / (2.0f * a);
            if (fabsf(p1) > max_mag) {
                max_mag = fabsf(p1);
                *dominant_pole_real = p1;
                *dominant_pole_imag = 0.0f;
                found_dominant = 1;
            }
            if (fabsf(p2) > max_mag) {
                max_mag = fabsf(p2);
                *dominant_pole_real = p2;
                *dominant_pole_imag = 0.0f;
                found_dominant = 1;
            }
            *damping_ratio = (found_dominant && *dominant_pole_real < 0.0f) ? 1.0f : 0.0f;
        } else {
            float real_part = -b / (2.0f * a);
            float imag_part = sqrtf(-disc) / (2.0f * a);
            float mag = sqrtf(real_part * real_part + imag_part * imag_part);
            if (mag > max_mag) {
                max_mag = mag;
                *dominant_pole_real = real_part;
                *dominant_pole_imag = imag_part;
                found_dominant = 1;
            }
            float wn = sqrtf(fabsf(c / a));
            if (wn > 1e-10f) {
                *damping_ratio = fabsf(real_part) / wn;
            }
        }
    } else {
        *dominant_pole_real = -(1.0f / (float)order);
        *dominant_pole_imag = 0.0f;
        *damping_ratio = 0.5f;
    }

    if (!found_dominant) {
        *dominant_pole_real = -0.1f;
        *damping_ratio = 0.5f;
    }

    return 0;
}

/* ============================================================================
 * 频域学习强化：谱门控学习
 *
 * 在频域对梯度进行门控，允许特定频率分量的学习通过，
 * 阻隔不稳定或噪声频率分量。
 * =========================================================================== */

static int lai_spectral_gating_learning(LaplaceAI* ai, const float* gradient,
                                         size_t grad_length, float* output,
                                         float gate_strength) {
    if (!ai || !gradient || !output || grad_length == 0) return -1;
    if (gate_strength <= 0.0f) {
        memcpy(output, gradient, grad_length * sizeof(float));
        return (int)grad_length;
    }
    if (gate_strength > 1.0f) gate_strength = 1.0f;

    size_t fft_n = lai_next_pow2(grad_length);
    if (fft_n > LAI_SPECTRUM_SIZE) fft_n = LAI_SPECTRUM_SIZE;

    float* real = (float*)calloc(fft_n, sizeof(float));
    float* imag = (float*)calloc(fft_n, sizeof(float));
    if (!real || !imag) { free(real); free(imag); return -2; }

    size_t copy_len = grad_length < fft_n ? grad_length : fft_n;
    memcpy(real, gradient, copy_len * sizeof(float));
    lai_fft_radix2(real, imag, fft_n, 0);

    size_t num_bins = fft_n / 2 + 1;
    float nyquist = ai->config.sampling_rate / 2.0f;

    for (size_t i = 0; i < num_bins; i++) {
        float freq = (float)i * ai->config.sampling_rate / (float)fft_n;
        float mag = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
        if (mag < 1e-10f) continue;

        /* 频率门控：低频（稳定学习）和高频（噪声抑制）有不同的门控 */
        float gate = 1.0f;
        float low_thresh = nyquist * 0.05f;
        float high_thresh = nyquist * 0.35f;

        if (freq < low_thresh) {
            gate = 1.0f;
        } else if (freq < high_thresh) {
            float t = (freq - low_thresh) / (high_thresh - low_thresh);
            gate = 1.0f - t * gate_strength * 0.3f;
        } else {
            float t = (freq - high_thresh) / (nyquist - high_thresh);
            gate = (1.0f - gate_strength * 0.7f) * (1.0f - t * 0.5f);
        }

        if (gate < 0.05f) gate = 0.05f;
        real[i] *= gate;
        imag[i] *= gate;
    }

    for (size_t i = 1; i < fft_n / 2; i++) {
        real[fft_n - i] = real[i];
        imag[fft_n - i] = -imag[i];
    }

    lai_fft_radix2(real, imag, fft_n, 1);

    size_t out_len = grad_length < fft_n ? grad_length : fft_n;
    memcpy(output, real, out_len * sizeof(float));

    free(real);
    free(imag);
    return (int)out_len;
}

/* ============================================================================
 * 循环退火学习率调度器
 *
 * 实现循环余弦退火调度，结合频谱功率比自适应调整。
 * 每个周期内学习率从高到低余弦退火，周期结束后重置。
 * =========================================================================== */

typedef struct {
    float base_lr;                 /**< 基础学习率 */
    float min_lr;                  /**< 最低学习率 (退火底) */
    int cycle_length;              /**< 周期长度（步数） */
    int current_step;              /**< 当前步数 */
    int total_cycles;              /**< 总周期数 */
    float spectral_power_avg;      /**< 平均频谱功率 */
    float lr_scale_history[100];   /**< 学习率缩放历史 */
    int history_count;
    int is_initialized;
} LAICyclicalAnnealingLR;

static void lai_cyclical_annealing_init(LAICyclicalAnnealingLR* scheduler,
                                         float base_lr, float min_lr,
                                         int cycle_length) {
    if (!scheduler) return;
    memset(scheduler, 0, sizeof(LAICyclicalAnnealingLR));
    scheduler->base_lr = base_lr;
    scheduler->min_lr = min_lr;
    scheduler->cycle_length = cycle_length > 0 ? cycle_length : 100;
    scheduler->current_step = 0;
    scheduler->total_cycles = 0;
    scheduler->spectral_power_avg = 1.0f;
    scheduler->is_initialized = 1;
}

static float lai_cyclical_annealing_get_lr(LAICyclicalAnnealingLR* scheduler) {
    if (!scheduler || !scheduler->is_initialized) return 0.001f;

    /* 余弦退火：一个周期内从base_lr下降到min_lr */
    float progress = (float)(scheduler->current_step % scheduler->cycle_length) /
                     (float)scheduler->cycle_length;
    float cosine = 0.5f * (1.0f + cosf((float)M_PI * progress));
    float lr = scheduler->min_lr + (scheduler->base_lr - scheduler->min_lr) * cosine;

    /* 频谱功率自适应缩放 */
    lr *= scheduler->spectral_power_avg;

    if (lr < scheduler->min_lr * 0.1f) lr = scheduler->min_lr * 0.1f;

    return lr;
}

static void lai_cyclical_annealing_step(LAICyclicalAnnealingLR* scheduler) {
    if (!scheduler || !scheduler->is_initialized) return;

    scheduler->current_step++;
    if (scheduler->current_step % scheduler->cycle_length == 0) {
        scheduler->total_cycles++;
    }
}

static void lai_cyclical_annealing_update_spectral(LAICyclicalAnnealingLR* scheduler,
                                                     float spectral_power) {
    if (!scheduler || !scheduler->is_initialized) return;

    float alpha = 0.95f;
    scheduler->spectral_power_avg = alpha * scheduler->spectral_power_avg +
                                     (1.0f - alpha) * spectral_power;

    if (scheduler->history_count < 100) {
        scheduler->lr_scale_history[scheduler->history_count++] = scheduler->spectral_power_avg;
    } else {
        memmove(scheduler->lr_scale_history, scheduler->lr_scale_history + 1,
                99 * sizeof(float));
        scheduler->lr_scale_history[99] = scheduler->spectral_power_avg;
    }
}

/* ============================================================================
 * 谱通量动态监测
 *
 * 实时监测频谱变化率（谱通量），用于检测信号突变、事件检测和自适应门控。
 * 谱通量 = Σ(|S_t(k)| - |S_{t-1}(k)|)
 * =========================================================================== */

typedef struct {
    float* prev_spectrum;          /**< 上一帧频谱 */
    float spectral_flux;           /**< 当前谱通量 */
    float flux_mean;               /**< 谱通量均值 */
    float flux_std;                /**< 谱通量标准差 */
    int frame_count;
    int fft_size;
    float flux_history[50];        /**< 谱通量历史 */
    int history_count;
    int is_initialized;
} LAISpectralFluxMonitor;

static int lai_spectral_flux_init(LAISpectralFluxMonitor* sf, int fft_size) {
    if (!sf || fft_size < 16) return -1;
    memset(sf, 0, sizeof(LAISpectralFluxMonitor));
    sf->fft_size = fft_size;
    sf->prev_spectrum = (float*)calloc((size_t)fft_size / 2 + 1, sizeof(float));
    if (!sf->prev_spectrum) return -2;
    sf->is_initialized = 1;
    return 0;
}

static void lai_spectral_flux_free(LAISpectralFluxMonitor* sf) {
    if (!sf) return;
    free(sf->prev_spectrum);
    memset(sf, 0, sizeof(LAISpectralFluxMonitor));
}

static float lai_spectral_flux_compute(LAISpectralFluxMonitor* sf,
                                        const float* magnitude_spectrum,
                                        size_t num_bins) {
    if (!sf || !sf->is_initialized || !magnitude_spectrum || num_bins == 0) return 0.0f;

    size_t expected = (size_t)sf->fft_size / 2 + 1;
    size_t bins = num_bins < expected ? num_bins : expected;

    float flux = 0.0f;
    for (size_t i = 0; i < bins; i++) {
        float diff = magnitude_spectrum[i] - sf->prev_spectrum[i];
        flux += (diff > 0.0f) ? diff : 0.0f;
    }
    flux /= (float)bins;

    /* 更新历史频谱 */
    memcpy(sf->prev_spectrum, magnitude_spectrum, bins * sizeof(float));

    sf->spectral_flux = flux;

    /* 更新均值和标准差 */
    float alpha = 0.9f;
    if (sf->frame_count == 0) {
        sf->flux_mean = flux;
        sf->flux_std = flux * 0.5f;
    } else {
        sf->flux_mean = alpha * sf->flux_mean + (1.0f - alpha) * flux;
        float dev = flux - sf->flux_mean;
        sf->flux_std = alpha * sf->flux_std + (1.0f - alpha) * fabsf(dev);
    }
    sf->frame_count++;

    if (sf->history_count < 50) {
        sf->flux_history[sf->history_count++] = flux;
    } else {
        memmove(sf->flux_history, sf->flux_history + 1, 49 * sizeof(float));
        sf->flux_history[49] = flux;
    }

    return flux;
}

/* ============================================================================
 * 相位补偿梯度过滤
 *
 * 在频域对梯度进行低通滤波，同时补偿相位延迟，避免梯度失真。
 * 使用零相位滤波（前向-后向滤波）实现。
 * =========================================================================== */

static int lai_zero_phase_filter(LaplaceAI* ai, const float* input,
                                  size_t input_length, float* output,
                                  float cutoff_freq) {
    if (!ai || !input || !output || input_length == 0) return -1;
    if (cutoff_freq <= 0.0f) cutoff_freq = ai->config.sampling_rate * 0.3f;

    /* 前向滤波 */
    int ret = laplace_ai_filter_signal(ai, input, input_length, output,
                                        LAI_FILTER_LOWPASS, cutoff_freq, 0.0f);
    if (ret <= 0) return ret;

    /* 反向再滤波（零相位） */
    float* reversed = (float*)calloc(input_length, sizeof(float));
    if (!reversed) return -3;

    for (size_t i = 0; i < input_length; i++) {
        reversed[i] = output[input_length - 1 - i];
    }

    ret = laplace_ai_filter_signal(ai, reversed, input_length, output,
                                    LAI_FILTER_LOWPASS, cutoff_freq, 0.0f);
    if (ret > 0) {
        for (size_t i = 0; i < input_length; i++) {
            output[i] = output[input_length - 1 - i];
        }
    }

    free(reversed);

    if (ret > 0) return ret;
    return -4;
}

/* ============================================================================
 * 公开API：频域学习强化
 *
 * 对外提供频域门控学习接口，使外部训练循环可以直接调用谱门控梯度过滤。
 * =========================================================================== */

int laplace_ai_spectral_gate_gradient(LaplaceAI* ai, const float* gradient,
                                       size_t grad_length, float* output,
                                       float gate_strength) {
    return lai_spectral_gating_learning(ai, gradient, grad_length, output, gate_strength);
}

int laplace_ai_zero_phase_filter_gradient(LaplaceAI* ai, const float* gradient,
                                           size_t grad_length, float* output,
                                           float cutoff_freq) {
    return lai_zero_phase_filter(ai, gradient, grad_length, output, cutoff_freq);
}

int laplace_ai_extract_mfcc_features(LaplaceAI* ai, const float* signal,
                                      size_t signal_length, float* mfcc_output,
                                      int num_coeffs) {
    if (!ai || !signal || !mfcc_output || signal_length == 0) return -1;
    if (num_coeffs <= 0) num_coeffs = 13;
    if (num_coeffs > 32) num_coeffs = 32;

    LAISpectrumResult spectrum;
    memset(&spectrum, 0, sizeof(LAISpectrumResult));

    int ret = laplace_ai_extract_features(ai, signal, signal_length, &spectrum);
    if (ret != 0) return ret;

    LAICepstralAnalyzer ca;
    ret = lai_cepstral_init(&ca, (int)lai_next_pow2(signal_length),
                             ai->config.sampling_rate, 26, num_coeffs);
    if (ret != 0) {
        laplace_ai_spectrum_free(&spectrum);
        return ret;
    }

    ret = lai_extract_mfcc(&ca, spectrum.magnitude, spectrum.num_bins, mfcc_output);

    lai_cepstral_free(&ca);
    laplace_ai_spectrum_free(&spectrum);
    return ret;
}

int laplace_ai_compute_stability_margins(LaplaceAI* ai, const float* signal,
                                          size_t signal_length,
                                          float* phase_margin,
                                          float* gain_margin,
                                          float* dominant_pole) {
    if (!ai || !signal || signal_length == 0) return -1;
    if (!phase_margin || !gain_margin || !dominant_pole) return -1;

    LAIStabilityReport report;
    int ret = laplace_ai_monitor_stability(ai, signal, signal_length, &report);
    if (ret != 0) return ret;

    *phase_margin = report.phase_margin;
    *gain_margin = report.gain_margin;
    *dominant_pole = report.dominant_pole_real;
    return 0;
}

int laplace_ai_apply_spectral_subtraction(LaplaceAI* ai, const float* input,
                                           size_t input_length, float* output,
                                           float noise_floor, float subtraction_factor) {
    if (!ai || !input || !output || input_length == 0) return -1;

    size_t fft_n = lai_next_pow2(input_length);
    if (fft_n > LAI_SPECTRUM_SIZE) fft_n = LAI_SPECTRUM_SIZE;

    float* real = (float*)calloc(fft_n, sizeof(float));
    float* imag = (float*)calloc(fft_n, sizeof(float));
    if (!real || !imag) { free(real); free(imag); return -2; }

    memcpy(real, input, (input_length < fft_n ? input_length : fft_n) * sizeof(float));
    lai_apply_window(real, input_length < fft_n ? input_length : fft_n,
                      LAI_WINDOW_HANNING, 0.0f);
    lai_fft_radix2(real, imag, fft_n, 0);

    LAISpectralSubtractor ss;
    int ret = lai_spectral_subtractor_init(&ss, (int)fft_n);
    if (ret != 0) { free(real); free(imag); return ret; }

    ss.over_subtraction_factor = subtraction_factor;
    ss.spectral_floor = noise_floor;

    ret = lai_spectral_subtractor_apply(&ss, real, imag, fft_n, 1);
    if (ret == 0) {
        for (size_t i = 1; i < fft_n / 2; i++) {
            real[fft_n - i] = real[i];
            imag[fft_n - i] = -imag[i];
        }
        lai_fft_radix2(real, imag, fft_n, 1);
        size_t out_len = input_length < fft_n ? input_length : fft_n;
        memcpy(output, real, out_len * sizeof(float));
        ret = (int)out_len;
    }

    lai_spectral_subtractor_free(&ss);
    free(real);
    free(imag);
    return ret;
}

int laplace_ai_design_filter_bank(LaplaceAI* ai, int num_subbands,
                                   size_t filter_length, float transition_bw,
                                   float* subband_energies, size_t* num_bands) {
    if (!ai || !subband_energies || !num_bands) return -1;
    if (num_subbands < 2 || num_subbands > LAI_MAX_FILTER_BANDS) return -1;

    LAIFilterBank fb;
    int ret = lai_filter_bank_design(&fb, num_subbands, filter_length, transition_bw);
    if (ret != 0) { *num_bands = 0; return ret; }

    for (int k = 0; k < fb.num_subbands && k < LAI_MAX_FILTER_BANDS; k++) {
        subband_energies[k] = fb.subband_energies[k];
    }
    *num_bands = (size_t)fb.num_subbands;

    lai_filter_bank_free(&fb);
    return 0;
}

int laplace_ai_cyclical_annealing_lr(LaplaceAI* ai, float base_lr,
                                      float min_lr, int cycle_length,
                                      int current_step,
                                      const float* gradient_spectrum,
                                      size_t spectrum_length) {
    if (!ai) return -1;

    LAICyclicalAnnealingLR scheduler;
    lai_cyclical_annealing_init(&scheduler, base_lr, min_lr, cycle_length);
    scheduler.current_step = current_step;

    if (gradient_spectrum && spectrum_length > 0) {
        float avg_power = 0.0f;
        for (size_t i = 0; i < spectrum_length; i++) {
            avg_power += gradient_spectrum[i] * gradient_spectrum[i];
        }
        avg_power = sqrtf(avg_power / (float)spectrum_length);
        lai_cyclical_annealing_update_spectral(&scheduler, avg_power);
    }

    float adapted_lr = lai_cyclical_annealing_get_lr(&scheduler);
    ai->prev_lr = adapted_lr / base_lr;
    return 0;
}

/* 线程局部存储：谱通量监控器 */
static LAI_THREAD_LOCAL LAISpectralFluxMonitor g_flux_monitor;
static LAI_THREAD_LOCAL int g_flux_monitor_initialized = 0;

int laplace_ai_compute_spectral_flux(LaplaceAI* ai, const float* magnitude_spectrum,
                                      size_t num_bins, float* flux_out) {
    if (!ai || !magnitude_spectrum || !flux_out || num_bins == 0) return -1;

    if (!g_flux_monitor_initialized) {
        if (lai_spectral_flux_init(&g_flux_monitor, (int)lai_next_pow2(num_bins * 2)) != 0) {
            *flux_out = 0.0f;
            return -2;
        }
        g_flux_monitor_initialized = 1;
    }

    *flux_out = lai_spectral_flux_compute(&g_flux_monitor, magnitude_spectrum, num_bins);
    return 0;
}

void laplace_ai_spectral_flux_reset(void) {
    lai_spectral_flux_free(&g_flux_monitor);
    memset(&g_flux_monitor, 0, sizeof(LAISpectralFluxMonitor));
    g_flux_monitor_initialized = 0;
}

/* ============================================================================
 * 公开API：强化学习系统
 *
 * 对外提供强化学习训练接口，支持DQN/PPO/SAC/A2C算法。
 * =========================================================================== */

RLAgent* laplace_ai_rl_create(RLAlgorithm algorithm, int state_dim, int action_dim)
{
    RLConfig cfg = rl_config_default(algorithm, state_dim, action_dim);
    RLAgent* agent = rl_agent_create(&cfg);
    if (!agent) return NULL;

    int slot = lai_rl_find_slot();
    if (slot < 0)
    {
        rl_agent_free(agent);
        return NULL;
    }
    g_rl_agents[slot] = agent;
    g_rl_agent_count++;
    return agent;
}

void laplace_ai_rl_destroy(RLAgent* agent)
{
    if (!agent) return;
    int slot = lai_rl_find_agent(agent);
    if (slot >= 0)
    {
        g_rl_agents[slot] = NULL;
        g_rl_agent_count--;
    }
    rl_agent_free(agent);
}

void laplace_ai_rl_destroy_all(void)
{
    for (int i = 0; i < RL_MAX_AGENTS; i++)
    {
        if (g_rl_agents[i])
        {
            rl_agent_free(g_rl_agents[i]);
            g_rl_agents[i] = NULL;
        }
    }
    g_rl_agent_count = 0;
}

int laplace_ai_rl_select_action(RLAgent* agent, const float* state, int state_dim,
                                 float* action, int action_dim)
{
    if (!agent) return -1;
    return rl_select_action(agent, state, state_dim, action, action_dim);
}

int laplace_ai_rl_store_experience(RLAgent* agent, const float* state, int state_dim,
                                    const float* action, int action_dim, float reward,
                                    const float* next_state, int next_state_dim, int done)
{
    if (!agent) return -1;
    return rl_store_experience(agent, state, state_dim, action, action_dim,
                               reward, next_state, next_state_dim, done);
}

int laplace_ai_rl_train(RLAgent* agent, int batch_size)
{
    if (!agent) return -1;
    return rl_train(agent, batch_size);
}

int laplace_ai_rl_save(RLAgent* agent, const char* filepath)
{
    if (!agent || !filepath) return -1;
    return rl_save(agent, filepath);
}

RLAgent* laplace_ai_rl_load(const char* filepath)
{
    if (!filepath) return NULL;
    RLAgent* agent = rl_load(filepath);
    if (!agent) return NULL;

    int slot = lai_rl_find_slot();
    if (slot < 0)
    {
        rl_agent_free(agent);
        return NULL;
    }
    g_rl_agents[slot] = agent;
    g_rl_agent_count++;
    return agent;
}

int laplace_ai_rl_update_exploration(RLAgent* agent)
{
    if (!agent) return -1;
    return rl_update_exploration(agent);
}

int laplace_ai_rl_reset(RLAgent* agent)
{
    if (!agent) return -1;
    return rl_reset(agent);
}

float laplace_ai_rl_get_exploration_rate(const RLAgent* agent)
{
    if (!agent) return 0.0f;
    return rl_get_exploration_rate(agent);
}

int laplace_ai_rl_set_exploration_rate(RLAgent* agent, float rate)
{
    if (!agent) return -1;
    return rl_set_exploration_rate(agent, rate);
}

int laplace_ai_rl_get_stats(const RLAgent* agent, int* total_steps, int* total_episodes,
                             float* avg_return, float* best_return)
{
    if (!agent) return -1;
    return rl_get_stats(agent, total_steps, total_episodes, avg_return, best_return);
}

int laplace_ai_rl_get_q_values(RLAgent* agent, const float* state, int state_dim, float* q_values)
{
    if (!agent) return -1;
    return rl_dqn_get_q_values(agent, state, state_dim, q_values);
}

/* ============================================================================
 * ZSFX-P0修复: laplace_ai_framework_init — 拉普拉斯AI框架统一初始化入口
 * 由 laplace_unified_system_init() 调用，创建全局LaplaceAI实例。
 * ============================================================================ */

static LaplaceAI* g_global_laplace_ai = NULL;

int laplace_ai_framework_init(void) {
    if (g_global_laplace_ai) return 0; /* 已初始化 */

    LaplaceAIConfig cfg = LAPLACE_AI_CONFIG_DEFAULT_INITIALIZER;
    cfg.sampling_rate = 44100; /* 默认音频采样率 */
    cfg.fft_size = 2048;
    cfg.window_type = LAI_WINDOW_HANN;
    cfg.enable_adaptive_filter = 1;
    cfg.num_feature_bands = 13;
    cfg.enable_rl = 0;

    g_global_laplace_ai = laplace_ai_create(&cfg);
    if (!g_global_laplace_ai) return -1;
    return 0;
}

void laplace_ai_framework_cleanup(void) {
    if (g_global_laplace_ai) {
        laplace_ai_free(g_global_laplace_ai);
        g_global_laplace_ai = NULL;
    }
}

LaplaceAI* laplace_ai_framework_get_instance(void) {
    return g_global_laplace_ai;
}
