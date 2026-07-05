/**
 * @file audio_tools.c
 * @brief 音频处理辅助工具集
 *
 * 提供四种音频处理工具：
 * 1. CQT时频分析 — 基于稀疏核矩阵的快速CQT
 * 2. STFT短时傅里叶变换 — 分帧/加窗/FFT/重叠相加iSTFT
 * 3. 音频源分离 — 频域掩蔽+液态状态动态系统
 * 4. 音频事件检测 — 液态状态演化+时间后处理
 *
 * 全部纯C实现，无外部依赖。
 */

#include "selflnn/multimodal/audio_tools.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* FIX-EXTERN9: math_utils.h已包含，移除冗余extern声明 */
#include "selflnn/utils/math_utils.h"

/* =============================================================== *
 * 内部辅助函数                                                     *
 * =============================================================== */

/** @brief 获取不小于n的最小2的幂 */
static int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/**
 * @brief 基2复FFT（原地，in-place）
 *
 * @param real 实部数组（大小n）
 * @param imag 虚部数组（大小n）
 * @param n FFT大小（必须为2的幂）
 */
static void fft_complex(float* real, float* imag, int n) {
    /* 位反转重排 */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }
    /* Cooley-Tukey蝶形运算 */
    for (int len = 2; len <= n; len <<= 1) {
        float wlen_r = cosf(2.0f * (float)M_PI / len);
        float wlen_i = -sinf(2.0f * (float)M_PI / len);
        for (int i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int idx = i + j;
                int idx2 = idx + len / 2;
                float tr = wr * real[idx2] - wi * imag[idx2];
                float ti = wr * imag[idx2] + wi * real[idx2];
                real[idx2] = real[idx] - tr;
                imag[idx2] = imag[idx] - ti;
                real[idx] += tr;
                imag[idx] += ti;
                float new_wr = wr * wlen_r - wi * wlen_i;
                float new_wi = wr * wlen_i + wi * wlen_r;
                wr = new_wr;
                wi = new_wi;
            }
        }
    }
}

/**
 * @brief 基2复IFFT（原地）
 */
static void ifft_complex(float* real, float* imag, int n) {
    /* 共轭 */
    for (int i = 0; i < n; i++) imag[i] = -imag[i];
    fft_complex(real, imag, n);
    /* 缩放 */
    float inv_n = 1.0f / n;
    for (int i = 0; i < n; i++) {
        real[i] *= inv_n;
        imag[i] = -imag[i] * inv_n;
    }
}

/* =============================================================== *
 * CQT时频分析                                                      *
 * =============================================================== */

/**
 * @brief CQT内部结构体
 */
struct AudioCQT {
    AudioCQTConfig config;
    int fft_size;
    float* kernel_real;          /* 稀疏核矩阵实部 [num_bins x fft_size] */
    float* kernel_imag;          /* 稀疏核矩阵虚部 [num_bins x fft_size] */
    float* frequencies;          /* 中心频率 [num_bins] */
    float* window_buf;           /* 窗函数缓冲区 */
    float* fft_buf_real;         /* FFT实部缓冲区 */
    float* fft_buf_imag;         /* FFT虚部缓冲区 */
    int initialized;
};

static AudioCQTConfig audio_cqt_default_config(void) {
    AudioCQTConfig cfg;
    cfg.sample_rate = 16000;
    cfg.bins_per_octave = 24;
    cfg.num_bins = 84;
    cfg.min_freq = 65.406f;
    cfg.max_freq = 2093.0f;
    cfg.window_length_factor = 1.0f;
    return cfg;
}

AudioCQT* audio_cqt_create(const AudioCQTConfig* config) {
    AudioCQT* cqt = (AudioCQT*)safe_malloc(sizeof(AudioCQT));
    if (!cqt) return NULL;
    memset(cqt, 0, sizeof(AudioCQT));

    if (config) {
        cqt->config = *config;
    } else {
        cqt->config = audio_cqt_default_config();
    }

    AudioCQTConfig* cfg = &cqt->config;

    if (cfg->num_bins <= 0 || cfg->num_bins > AUDIO_CQT_MAX_BINS) {
        safe_free((void**)&cqt);
        return NULL;
    }
    if (cfg->sample_rate <= 0 || cfg->bins_per_octave <= 0) {
        safe_free((void**)&cqt);
        return NULL;
    }
    if (cfg->min_freq <= 0 || cfg->max_freq <= cfg->min_freq) {
        safe_free((void**)&cqt);
        return NULL;
    }

    /* FFT大小：保证能覆盖最低频率的周期 */
    float q_factor = (float)cfg->bins_per_octave / (powf(2.0f, 1.0f / cfg->bins_per_octave) - 1.0f);
    float min_window = q_factor * cfg->sample_rate / cfg->min_freq;
    cqt->fft_size = next_power_of_two((int)(min_window * cfg->window_length_factor * 2.0f));
    if (cqt->fft_size < 512) cqt->fft_size = 512;
    if (cqt->fft_size > 16384) cqt->fft_size = 16384;

    /* 分配内存 */
    int n_bins = cfg->num_bins;
    cqt->kernel_real = (float*)safe_malloc((size_t)n_bins * cqt->fft_size * sizeof(float));
    cqt->kernel_imag = (float*)safe_malloc((size_t)n_bins * cqt->fft_size * sizeof(float));
    cqt->frequencies = (float*)safe_malloc((size_t)n_bins * sizeof(float));
    cqt->window_buf = (float*)safe_malloc((size_t)cqt->fft_size * sizeof(float));
    cqt->fft_buf_real = (float*)safe_malloc((size_t)cqt->fft_size * sizeof(float));
    cqt->fft_buf_imag = (float*)safe_malloc((size_t)cqt->fft_size * sizeof(float));

    if (!cqt->kernel_real || !cqt->kernel_imag || !cqt->frequencies ||
        !cqt->window_buf || !cqt->fft_buf_real || !cqt->fft_buf_imag) {
        audio_cqt_free(cqt);
        return NULL;
    }

    /* 计算每个频带的中心频率（几何间隔） */
    float ratio = powf(cfg->max_freq / cfg->min_freq, 1.0f / (n_bins - 1));
    for (int k = 0; k < n_bins; k++) {
        cqt->frequencies[k] = cfg->min_freq * powf(ratio, (float)k);
    }

    /* 预计算稀疏核矩阵：对每个频带k，生成频域核 */
    memset(cqt->kernel_real, 0, (size_t)n_bins * cqt->fft_size * sizeof(float));
    memset(cqt->kernel_imag, 0, (size_t)n_bins * cqt->fft_size * sizeof(float));

    for (int k = 0; k < n_bins; k++) {
        float fk = cqt->frequencies[k];
        float qk = q_factor;
        int window_len = (int)(qk * cfg->sample_rate / fk);
        if (window_len > cqt->fft_size) window_len = cqt->fft_size;
        if (window_len < 4) window_len = 4;

        /* 生成汉宁窗+复指数核 */
        for (int n = 0; n < window_len; n++) {
            float t = (float)n / cfg->sample_rate;
            float window_val = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / (window_len - 1));
            float angle = 2.0f * (float)M_PI * fk * t;
            cqt->kernel_real[k * cqt->fft_size + n] = window_val * cosf(angle) / window_len;
            cqt->kernel_imag[k * cqt->fft_size + n] = window_val * sinf(angle) / window_len;
        }

        /* FFT将核变换到频域（用于高效卷积） */
        memcpy(cqt->fft_buf_real, &cqt->kernel_real[k * cqt->fft_size], cqt->fft_size * sizeof(float));
        memcpy(cqt->fft_buf_imag, &cqt->kernel_imag[k * cqt->fft_size], cqt->fft_size * sizeof(float));
        fft_complex(cqt->fft_buf_real, cqt->fft_buf_imag, cqt->fft_size);
        memcpy(&cqt->kernel_real[k * cqt->fft_size], cqt->fft_buf_real, cqt->fft_size * sizeof(float));
        memcpy(&cqt->kernel_imag[k * cqt->fft_size], cqt->fft_buf_imag, cqt->fft_size * sizeof(float));
    }

    /* 计算FFT所需的汉宁窗 */
    for (int i = 0; i < cqt->fft_size; i++) {
        cqt->window_buf[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (cqt->fft_size - 1));
    }

    cqt->initialized = 1;
    return cqt;
}

int audio_cqt_compute(AudioCQT* cqt, const float* samples, int num_samples,
                       float* magnitude, int max_mag) {
    if (!cqt || !cqt->initialized || !samples || !magnitude) return -1;
    if (num_samples <= 0) return -1;

    AudioCQTConfig* cfg = &cqt->config;
    int n_bins = cfg->num_bins;
    int hop_size = cqt->fft_size / 2;
    int num_frames = (num_samples - cqt->fft_size) / hop_size + 1;
    if (num_frames < 1) num_frames = 1;
    if (num_frames > AUDIO_CQT_MAX_FRAMES) num_frames = AUDIO_CQT_MAX_FRAMES;

    if (max_mag < n_bins * num_frames) return -1;

    memset(magnitude, 0, (size_t)n_bins * num_frames * sizeof(float));

    /* 对每帧：FFT→频域相乘（核已FFT）→IFFT→取模 */
    for (int f = 0; f < num_frames; f++) {
        int offset = f * hop_size;

        /* 加窗并填充到fft_buf */
        memset(cqt->fft_buf_imag, 0, cqt->fft_size * sizeof(float));
        for (int i = 0; i < cqt->fft_size; i++) {
            int idx = offset + i;
            cqt->fft_buf_real[i] = (idx < num_samples) ? samples[idx] * cqt->window_buf[i] : 0.0f;
        }

        /* FFT */
        fft_complex(cqt->fft_buf_real, cqt->fft_buf_imag, cqt->fft_size);

        /* 对每个频带k：频域相乘→IFFT→取第0点幅度 */
        for (int k = 0; k < n_bins; k++) {
            float* kr = &cqt->kernel_real[k * cqt->fft_size];
            float* ki = &cqt->kernel_imag[k * cqt->fft_size];

            /* 频域点乘等价于时域卷积 */
            float sum_r = 0.0f, sum_i = 0.0f;
            for (int i = 0; i < cqt->fft_size; i++) {
                float sr = cqt->fft_buf_real[i];
                float si = cqt->fft_buf_imag[i];
                sum_r += sr * kr[i] - si * ki[i];
                sum_i += sr * ki[i] + si * kr[i];
            }

            magnitude[k * num_frames + f] = sqrtf(sum_r * sum_r + sum_i * sum_i);
        }
    }

    return num_frames;
}

int audio_cqt_get_frequencies(AudioCQT* cqt, float* frequencies, int max_freqs) {
    if (!cqt || !cqt->initialized || !frequencies) return -1;
    int n = cqt->config.num_bins;
    if (max_freqs < n) return -1;
    memcpy(frequencies, cqt->frequencies, (size_t)n * sizeof(float));
    return n;
}

int audio_cqt_get_config(AudioCQT* cqt, AudioCQTConfig* config) {
    if (!cqt || !config) return -1;
    *config = cqt->config;
    return 0;
}

void audio_cqt_free(AudioCQT* cqt) {
    if (!cqt) return;
    safe_free((void**)&cqt->kernel_real);
    safe_free((void**)&cqt->kernel_imag);
    safe_free((void**)&cqt->frequencies);
    safe_free((void**)&cqt->window_buf);
    safe_free((void**)&cqt->fft_buf_real);
    safe_free((void**)&cqt->fft_buf_imag);
    safe_free((void**)&cqt);
}

/* =============================================================== *
 * STFT短时傅里叶变换                                               *
 * =============================================================== */

AudioSTFTConfig audio_stft_default_config(void) {
    AudioSTFTConfig cfg;
    cfg.fft_size = 512;
    cfg.hop_size = 128;
    cfg.window_type = AUDIO_WINDOW_HAMMING;
    cfg.use_magnitude_only = 0;
    return cfg;
}

void audio_generate_window(float* window, int size, AudioWindowType type) {
    if (!window || size <= 0) return;
    for (int i = 0; i < size; i++) {
        float a = (float)i / (size - 1);
        switch (type) {
            case AUDIO_WINDOW_HAMMING:
                window[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * a);
                break;
            case AUDIO_WINDOW_HANNING:
                window[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * a);
                break;
            case AUDIO_WINDOW_BLACKMAN:
                window[i] = 0.42f - 0.5f * cosf(2.0f * (float)M_PI * a) +
                            0.08f * cosf(4.0f * (float)M_PI * a);
                break;
            case AUDIO_WINDOW_RECT:
            default:
                window[i] = 1.0f;
                break;
        }
    }
}

int audio_stft_compute(const float* samples, int num_samples,
                        const AudioSTFTConfig* config,
                        float* magnitude, float* phase, int max_frames) {
    if (!samples || !magnitude) return -1;
    if (num_samples <= 0) return -1;

    AudioSTFTConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = audio_stft_default_config();
    }

    if (cfg.fft_size <= 0 || cfg.hop_size <= 0) return -1;

    int fft_size = cfg.fft_size;
    int hop_size = cfg.hop_size;
    int num_bins = fft_size / 2 + 1;
    int num_frames = (num_samples - fft_size) / hop_size + 1;
    if (num_frames < 1) num_frames = 1;
    if (num_frames > max_frames) num_frames = max_frames;
    if (num_frames > AUDIO_STFT_MAX_FRAMES) num_frames = AUDIO_STFT_MAX_FRAMES;

    float* window = (float*)safe_malloc((size_t)fft_size * sizeof(float));
    float* real = (float*)safe_malloc((size_t)fft_size * sizeof(float));
    float* imag = (float*)safe_malloc((size_t)fft_size * sizeof(float));
    if (!window || !real || !imag) {
        safe_free((void**)&window);
        safe_free((void**)&real);
        safe_free((void**)&imag);
        return -1;
    }

    audio_generate_window(window, fft_size, cfg.window_type);

    for (int f = 0; f < num_frames; f++) {
        int offset = f * hop_size;
        memset(real, 0, (size_t)fft_size * sizeof(float));
        memset(imag, 0, (size_t)fft_size * sizeof(float));

        for (int i = 0; i < fft_size; i++) {
            int idx = offset + i;
            if (idx < num_samples) {
                real[i] = samples[idx] * window[i];
            }
        }

        fft_complex(real, imag, fft_size);

        for (int b = 0; b < num_bins; b++) {
            float mag = sqrtf(real[b] * real[b] + imag[b] * imag[b]);
            magnitude[f * num_bins + b] = mag;
            if (phase) {
                phase[f * num_bins + b] = atan2f(imag[b], real[b] + 1e-10f);
            }
        }
    }

    safe_free((void**)&window);
    safe_free((void**)&real);
    safe_free((void**)&imag);
    return num_frames;
}

int audio_istft_compute(const float* magnitude, const float* phase,
                         int num_frames, const AudioSTFTConfig* config,
                         float* output, int max_output) {
    if (!magnitude || !output) return -1;
    if (num_frames <= 0) return -1;

    AudioSTFTConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = audio_stft_default_config();
    }

    int fft_size = cfg.fft_size;
    int hop_size = cfg.hop_size;
    int num_bins = fft_size / 2 + 1;
    int output_len = (num_frames - 1) * hop_size + fft_size;
    if (output_len > max_output) output_len = max_output;

    float* window = (float*)safe_malloc((size_t)fft_size * sizeof(float));
    float* real = (float*)safe_malloc((size_t)fft_size * sizeof(float));
    float* imag = (float*)safe_malloc((size_t)fft_size * sizeof(float));
    float* window_sum = (float*)safe_malloc((size_t)output_len * sizeof(float));
    if (!window || !real || !imag || !window_sum) {
        safe_free((void**)&window);
        safe_free((void**)&real);
        safe_free((void**)&imag);
        safe_free((void**)&window_sum);
        return -1;
    }

    audio_generate_window(window, fft_size, cfg.window_type);
    memset(output, 0, (size_t)output_len * sizeof(float));
    memset(window_sum, 0, (size_t)output_len * sizeof(float));

    for (int f = 0; f < num_frames; f++) {
        memset(real, 0, (size_t)fft_size * sizeof(float));
        memset(imag, 0, (size_t)fft_size * sizeof(float));

        for (int b = 0; b < num_bins; b++) {
            float mag = magnitude[f * num_bins + b];
            float ph = (phase) ? phase[f * num_bins + b] : 0.0f;
            real[b] = mag * cosf(ph);
            imag[b] = mag * sinf(ph);
            if (b > 0 && b < fft_size - b) {
                real[fft_size - b] = real[b];
                imag[fft_size - b] = -imag[b];
            }
        }

        ifft_complex(real, imag, fft_size);

        int offset = f * hop_size;
        int frame_end = offset + fft_size;
        if (frame_end > output_len) frame_end = output_len;
        for (int i = offset; i < frame_end; i++) {
            int wi = i - offset;
            output[i] += real[wi] * window[wi];
            window_sum[i] += window[wi] * window[wi];
        }
    }

    for (int i = 0; i < output_len; i++) {
        if (window_sum[i] > 1e-10f) {
            output[i] /= window_sum[i];
        }
    }

    safe_free((void**)&window);
    safe_free((void**)&real);
    safe_free((void**)&imag);
    safe_free((void**)&window_sum);
    return output_len;
}

/* =============================================================== *
 * 音频源分离（使用液态状态动态系统）                                *
 * =============================================================== */

/**
 * @brief 音频源分离内部结构体
 */
struct AudioSourceSep {
    AudioSourceSepConfig config;
    float* hidden_state;
    float* mask_weights;         /* 掩蔽权重 [num_sources x hidden_size x num_bins] */
    float* mask_biases;          /* 掩蔽偏置 [num_sources x num_bins] */
    float* window;
    float* fft_buf_real;
    float* fft_buf_imag;
    float* spectrum_buf;
    int fft_size;
    int hop_size;
    int num_bins;
    int initialized;
};

static AudioSourceSepConfig audio_source_sep_default_config(void) {
    AudioSourceSepConfig cfg;
    cfg.sample_rate = 16000;
    cfg.num_sources = 2;
    cfg.fft_size = 512;
    cfg.hop_size = 128;
    cfg.hidden_size = 256;
    cfg.time_constant = 0.05f;
    cfg.ode_solver_type = 0;
    cfg.use_gpu = 0;
    return cfg;
}

AudioSourceSep* audio_source_sep_create(const AudioSourceSepConfig* config) {
    AudioSourceSep* sep = (AudioSourceSep*)safe_malloc(sizeof(AudioSourceSep));
    if (!sep) return NULL;
    memset(sep, 0, sizeof(AudioSourceSep));

    if (config) {
        sep->config = *config;
    } else {
        sep->config = audio_source_sep_default_config();
    }

    AudioSourceSepConfig* cfg = &sep->config;
    if (cfg->num_sources <= 0 || cfg->num_sources > AUDIO_SOURCE_SEP_MAX_SOURCES) {
        safe_free((void**)&sep);
        return NULL;
    }

    sep->fft_size = cfg->fft_size;
    sep->hop_size = cfg->hop_size;
    sep->num_bins = cfg->fft_size / 2 + 1;
    int s = cfg->num_sources;
    size_t h = cfg->hidden_size;
    int nb = sep->num_bins;

    sep->hidden_state = (float*)safe_malloc(h * sizeof(float));
    sep->window = (float*)safe_malloc((size_t)cfg->fft_size * sizeof(float));
    sep->fft_buf_real = (float*)safe_malloc((size_t)cfg->fft_size * sizeof(float));
    sep->fft_buf_imag = (float*)safe_malloc((size_t)cfg->fft_size * sizeof(float));
    sep->spectrum_buf = (float*)safe_malloc((size_t)nb * sizeof(float));
    sep->mask_weights = (float*)safe_malloc((size_t)s * h * nb * sizeof(float));
    sep->mask_biases = (float*)safe_malloc((size_t)s * nb * sizeof(float));

    if (!sep->hidden_state || !sep->window ||
        !sep->fft_buf_real || !sep->fft_buf_imag || !sep->spectrum_buf ||
        !sep->mask_weights || !sep->mask_biases) {
        audio_source_sep_free(sep);
        return NULL;
    }

    memset(sep->hidden_state, 0, h * sizeof(float));

    /* 初始化掩蔽权重（Xavier初始化） */
    float scale_w = sqrtf(2.0f / (h + nb));
    for (int i = 0; i < s * (int)h * nb; i++) {
        sep->mask_weights[i] = rng_uniform(-scale_w, scale_w);
    }
    memset(sep->mask_biases, 0, (size_t)s * nb * sizeof(float));

    /* Hamming窗 */
    audio_generate_window(sep->window, cfg->fft_size, AUDIO_WINDOW_HAMMING);

    sep->initialized = 1;
    return sep;
}

int audio_source_sep_separate(AudioSourceSep* separator,
                               const float* mixed_audio, int num_samples,
                               float* separated, int max_per_source) {
    if (!separator || !separator->initialized || !mixed_audio || !separated) return -1;
    if (num_samples <= 0) return -1;

    AudioSourceSepConfig* cfg = &separator->config;
    int s = cfg->num_sources;
    int fft_size = separator->fft_size;
    int hop_size = separator->hop_size;
    int num_bins = separator->num_bins;
    size_t h = cfg->hidden_size;

    int num_frames = (num_samples - fft_size) / hop_size + 1;
    if (num_frames < 1) num_frames = 1;
    int output_len = (num_frames - 1) * hop_size + fft_size;
    if (output_len > max_per_source) output_len = max_per_source;

    /* 清零输出缓冲区 */
    memset(separated, 0, (size_t)s * output_len * sizeof(float));

    /* 每帧处理 */
    for (int f = 0; f < num_frames; f++) {
        int offset = f * hop_size;

        /* 加窗FFT */
        memset(separator->fft_buf_real, 0, (size_t)fft_size * sizeof(float));
        memset(separator->fft_buf_imag, 0, (size_t)fft_size * sizeof(float));
        for (int i = 0; i < fft_size; i++) {
            int idx = offset + i;
            if (idx < num_samples) {
                separator->fft_buf_real[i] = mixed_audio[idx] * separator->window[i];
            }
        }
        fft_complex(separator->fft_buf_real, separator->fft_buf_imag, fft_size);

        /* 提取幅度谱作为液态状态输入 */
        for (int b = 0; b < num_bins; b++) {
            separator->spectrum_buf[b] = sqrtf(
                separator->fft_buf_real[b] * separator->fft_buf_real[b] +
                separator->fft_buf_imag[b] * separator->fft_buf_imag[b]);
        }

        /* 液态状态演化（漏泄积分器） */
        float alpha = 1.0f / (1.0f + separator->config.time_constant);
        for (size_t i = 0; i < h && i < (size_t)num_bins; i++) {
            separator->hidden_state[i] = (1.0f - alpha) * separator->hidden_state[i] +
                                          alpha * separator->spectrum_buf[i];
        }

        /* 为每个源计算频域掩蔽 */
        for (int src = 0; src < s; src++) {
            float* mask = (float*)safe_malloc((size_t)num_bins * sizeof(float));
            if (!mask) continue;

            for (int b = 0; b < num_bins; b++) {
                float sum = separator->mask_biases[src * num_bins + b];
                for (size_t i = 0; i < h; i++) {
                    sum += separator->hidden_state[i] *
                           separator->mask_weights[src * h * num_bins + i * num_bins + b];
                }
                mask[b] = 1.0f / (1.0f + expf(-sum));
            }

            /* 应用掩蔽并IFFT */
            float* src_real = (float*)safe_malloc((size_t)fft_size * sizeof(float));
            float* src_imag = (float*)safe_malloc((size_t)fft_size * sizeof(float));
            if (!src_real || !src_imag) {
                safe_free((void**)&mask);
                safe_free((void**)&src_real);
                safe_free((void**)&src_imag);
                continue;
            }
            memcpy(src_real, separator->fft_buf_real, (size_t)fft_size * sizeof(float));
            memcpy(src_imag, separator->fft_buf_imag, (size_t)fft_size * sizeof(float));
            for (int b = 0; b < num_bins; b++) {
                src_real[b] *= mask[b];
                src_imag[b] *= mask[b];
                if (b > 0 && b < fft_size - b) {
                    src_real[fft_size - b] = src_real[b];
                    src_imag[fft_size - b] = -src_imag[b];
                }
            }
            ifft_complex(src_real, src_imag, fft_size);

            /* 重叠相加法合成时域 */
            int frame_end = offset + fft_size;
            if (frame_end > output_len) frame_end = output_len;
            for (int i = offset; i < frame_end; i++) {
                int wi = i - offset;
                separated[src * output_len + i] += src_real[wi] * separator->window[wi];
            }

            safe_free((void**)&mask);
            safe_free((void**)&src_real);
            safe_free((void**)&src_imag);
        }
    }

    return output_len;
}

void audio_source_sep_reset(AudioSourceSep* separator) {
    if (!separator || !separator->initialized) return;
    size_t h = separator->config.hidden_size;
    memset(separator->hidden_state, 0, h * sizeof(float));
}

void audio_source_sep_free(AudioSourceSep* separator) {
    if (!separator) return;
    safe_free((void**)&separator->hidden_state);
    safe_free((void**)&separator->window);
    safe_free((void**)&separator->fft_buf_real);
    safe_free((void**)&separator->fft_buf_imag);
    safe_free((void**)&separator->spectrum_buf);
    safe_free((void**)&separator->mask_weights);
    safe_free((void**)&separator->mask_biases);
    safe_free((void**)&separator);
}

/* =============================================================== *
 * 音频事件检测（使用液态状态动态系统）                              *
 * =============================================================== */

/**
 * @brief 音频事件检测内部结构体
 */
struct AudioEventDetector {
    AudioEventConfig config;
    float* hidden_state;
    float* output_weights;       /* 输出权重 [hidden_size x num_classes] */
    float* output_biases;        /* 输出偏置 [num_classes] */
    float* frame_probs;          /* 帧级概率缓冲区 [max_frames x num_classes] */
    int frame_probs_count;
    float* window;
    float* fft_buf_real;
    float* fft_buf_imag;
    float* spectrum_buf;
    float* mel_filterbank_buf;
    int fft_size;
    int hop_size;
    int num_bins;
    int initialized;
};

static AudioEventConfig audio_event_detector_default_config(void) {
    AudioEventConfig cfg;
    cfg.sample_rate = 16000;
    cfg.num_classes = 10;
    cfg.fft_size = 512;
    cfg.hop_size = 160;
    cfg.hidden_size = 256;
    cfg.time_constant = 0.05f;
    cfg.ode_solver_type = 0;
    cfg.detection_threshold = 0.5f;
    cfg.min_event_frames = 3;
    memset(cfg.class_names, 0, sizeof(cfg.class_names));
    return cfg;
}

AudioEventDetector* audio_event_detector_create(const AudioEventConfig* config) {
    AudioEventDetector* det = (AudioEventDetector*)safe_malloc(sizeof(AudioEventDetector));
    if (!det) return NULL;
    memset(det, 0, sizeof(AudioEventDetector));

    if (config) {
        det->config = *config;
    } else {
        det->config = audio_event_detector_default_config();
    }

    AudioEventConfig* cfg = &det->config;
    if (cfg->num_classes <= 0 || cfg->num_classes > AUDIO_EVENT_MAX_CLASSES) {
        safe_free((void**)&det);
        return NULL;
    }

    det->fft_size = cfg->fft_size;
    det->hop_size = cfg->hop_size;
    det->num_bins = cfg->fft_size / 2 + 1;
    size_t h = cfg->hidden_size;
    int nc = cfg->num_classes;
    int nb = det->num_bins;

    det->hidden_state = (float*)safe_malloc(h * sizeof(float));
    det->output_weights = (float*)safe_malloc(h * nc * sizeof(float));
    det->output_biases = (float*)safe_malloc((size_t)nc * sizeof(float));
    det->window = (float*)safe_malloc((size_t)cfg->fft_size * sizeof(float));
    det->fft_buf_real = (float*)safe_malloc((size_t)cfg->fft_size * sizeof(float));
    det->fft_buf_imag = (float*)safe_malloc((size_t)cfg->fft_size * sizeof(float));
    det->spectrum_buf = (float*)safe_malloc((size_t)nb * sizeof(float));

    if (!det->hidden_state ||
        !det->output_weights || !det->output_biases ||
        !det->window || !det->fft_buf_real || !det->fft_buf_imag ||
        !det->spectrum_buf) {
        audio_event_detector_free(det);
        return NULL;
    }

    memset(det->hidden_state, 0, h * sizeof(float));

    /* Xavier初始化输出权重 */
    float scale_w = sqrtf(2.0f / (h + nc));
    for (size_t i = 0; i < h * nc; i++) {
        det->output_weights[i] = rng_uniform(-scale_w, scale_w);
    }
    memset(det->output_biases, 0, (size_t)nc * sizeof(float));

    audio_generate_window(det->window, cfg->fft_size, AUDIO_WINDOW_HAMMING);

    det->initialized = 1;
    return det;
}

int audio_event_detector_detect(AudioEventDetector* detector,
                                 const float* samples, int num_samples,
                                 AudioEventResult* results, int max_results) {
    if (!detector || !detector->initialized || !samples || !results) return -1;
    if (num_samples <= 0 || max_results <= 0) return -1;

    AudioEventConfig* cfg = &detector->config;
    int fft_size = detector->fft_size;
    int hop_size = detector->hop_size;
    int num_bins = detector->num_bins;
    size_t h = cfg->hidden_size;
    int nc = cfg->num_classes;

    int num_frames = (num_samples - fft_size) / hop_size + 1;
    if (num_frames < 1) num_frames = 1;

    /* 分配帧级概率缓冲区 */
    float* frame_probs = (float*)safe_malloc((size_t)num_frames * nc * sizeof(float));
    if (!frame_probs) return -1;

    float alpha = 1.0f / (1.0f + cfg->time_constant);

    /* 逐帧处理 */
    for (int f = 0; f < num_frames; f++) {
        int offset = f * hop_size;

        memset(detector->fft_buf_real, 0, (size_t)fft_size * sizeof(float));
        memset(detector->fft_buf_imag, 0, (size_t)fft_size * sizeof(float));
        for (int i = 0; i < fft_size; i++) {
            int idx = offset + i;
            if (idx < num_samples) {
                detector->fft_buf_real[i] = samples[idx] * detector->window[i];
            }
        }
        fft_complex(detector->fft_buf_real, detector->fft_buf_imag, fft_size);

        for (int b = 0; b < num_bins; b++) {
            detector->spectrum_buf[b] = logf(
                sqrtf(detector->fft_buf_real[b] * detector->fft_buf_real[b] +
                      detector->fft_buf_imag[b] * detector->fft_buf_imag[b]) + 1e-10f);
        }

        /* 液态状态演化（漏泄积分器） */
        for (size_t i = 0; i < h && i < (size_t)num_bins; i++) {
            detector->hidden_state[i] = (1.0f - alpha) * detector->hidden_state[i] +
                                         alpha * detector->spectrum_buf[i];
        }

        for (int c = 0; c < nc; c++) {
            float logit = detector->output_biases[c];
            for (size_t i = 0; i < h; i++) {
                logit += detector->hidden_state[i] *
                         detector->output_weights[i * nc + c];
            }
            frame_probs[f * nc + c] = 1.0f / (1.0f + expf(-logit));
        }
    }

    /* 保存到内部缓冲区供 get_frame_probs 使用 */
    safe_free((void**)&detector->frame_probs);
    detector->frame_probs = frame_probs;
    detector->frame_probs_count = num_frames;

    /* 后处理：连续帧合并为事件段 */
    int* active = (int*)safe_malloc((size_t)num_frames * nc * sizeof(int));
    if (!active) return -1;

    for (int f = 0; f < num_frames; f++) {
        for (int c = 0; c < nc; c++) {
            active[f * nc + c] = (frame_probs[f * nc + c] >= cfg->detection_threshold) ? 1 : 0;
        }
    }

    int result_count = 0;
    for (int c = 0; c < nc && result_count < max_results; c++) {
        int seg_start = -1;
        int seg_len = 0;
        for (int f = 0; f <= num_frames; f++) {
            int is_active = (f < num_frames) ? active[f * nc + c] : 0;
            if (is_active && seg_start < 0) {
                seg_start = f;
                seg_len = 1;
            } else if (is_active && seg_start >= 0) {
                seg_len++;
            } else if (!is_active && seg_start >= 0) {
                if (seg_len >= cfg->min_event_frames) {
                    results[result_count].class_id = c;
                    results[result_count].confidence = 0.0f;
                    float max_conf = 0.0f;
                    for (int sf = seg_start; sf < seg_start + seg_len; sf++) {
                        if (frame_probs[sf * nc + c] > max_conf)
                            max_conf = frame_probs[sf * nc + c];
                    }
                    results[result_count].confidence = max_conf;
                    results[result_count].start_time = (float)seg_start * hop_size / cfg->sample_rate;
                    results[result_count].end_time = (float)(seg_start + seg_len) * hop_size / cfg->sample_rate;
                    result_count++;
                    if (result_count >= max_results) {
                        safe_free((void**)&active);
                        return result_count;
                    }
                }
                seg_start = -1;
                seg_len = 0;
            }
        }
    }

    safe_free((void**)&active);
    return result_count;
}

int audio_event_detector_get_frame_probs(AudioEventDetector* detector,
                                          float* frame_probs,
                                          int max_frames, int* num_classes) {
    if (!detector || !detector->initialized || !frame_probs || !num_classes) return -1;
    if (detector->frame_probs_count <= 0 || !detector->frame_probs) return -1;

    int nf = detector->frame_probs_count;
    if (nf > max_frames) nf = max_frames;
    *num_classes = detector->config.num_classes;
    memcpy(frame_probs, detector->frame_probs,
           (size_t)nf * detector->config.num_classes * sizeof(float));
    return nf;
}

void audio_event_detector_reset(AudioEventDetector* detector) {
    if (!detector || !detector->initialized) return;
    size_t h = detector->config.hidden_size;
    memset(detector->hidden_state, 0, h * sizeof(float));
}

void audio_event_detector_free(AudioEventDetector* detector) {
    if (!detector) return;
    safe_free((void**)&detector->hidden_state);
    safe_free((void**)&detector->output_weights);
    safe_free((void**)&detector->output_biases);
    safe_free((void**)&detector->window);
    safe_free((void**)&detector->fft_buf_real);
    safe_free((void**)&detector->fft_buf_imag);
    safe_free((void**)&detector->spectrum_buf);
    safe_free((void**)&detector->frame_probs);
    safe_free((void**)&detector);
}
