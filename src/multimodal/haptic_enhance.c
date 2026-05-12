/**
 * @file haptic_enhance.c
 * @brief CfC全液态神经网络的触觉/力觉感官系统深度增强实现
 *
 * 全部使用CfC液态神经网络模型：
 * - CfC触觉信号深度处理（ODF演化+接触检测+滑移检测）
 * - 触觉纹理深度分析（FFT+小波+分形+统计特征+CfC）
 * - CfC增强物体识别（多抓取融合+连续学习）
 * - CfC增强抓取学习（滑移补偿+自适应力+示教学习）
 * - 视触联合融合（CfC动态融合）
 *
 * 100%纯C实现，不依赖任何第三方库
 */
#include "selflnn/multimodal/haptic_learning.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ============================================================================
 * 内部辅助函数
 * ============================================================================ */

static inline float _hc_sig(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float _hc_tanh(float x) {
    return tanhf(x);
}

static float _hc_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float _hc_norm(const float* a, int n) {
    return sqrtf(_hc_dot(a, a, n) + 1e-30f);
}

static void _hc_vec_add(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

static void _hc_vec_sub(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) dst[i] -= src[i];
}

static void _hc_vec_scale(float* v, float s, int n) {
    for (int i = 0; i < n; i++) v[i] *= s;
}

static void _hc_mat_mul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            float s = 0.0f;
            for (int t = 0; t < n; t++)
                s += A[i * n + t] * B[t * k + j];
            C[i * k + j] = s;
        }
}

static void _hc_mat_transpose(const float* A, float* AT, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            AT[j * m + i] = A[i * n + j];
}

static void _hc_mat_vec_mul(const float* M, const float* v, float* out, int m, int n) {
    for (int i = 0; i < m; i++) {
        float s = 0.0f;
        for (int j = 0; j < n; j++)
            s += M[i * n + j] * v[j];
        out[i] = s;
    }
}

static void _hc_mat_identity(float* M, int n) {
    memset(M, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) M[i * n + i] = 1.0f;
}

/* CfC ODE步进（封闭形式解） */
static void _hc_cfc_ode_step(const float* h_in, const float* input,
                              float* h_out,
                              const float* W_h, const float* W_in,
                              const float* b_g, const float* b_a,
                              const float* tau, int hs, float dt) {
    float* gate = (float*)safe_malloc(hs * sizeof(float));
    float* act = (float*)safe_malloc(hs * sizeof(float));
    if (!gate || !act) { safe_free((void**)&gate); safe_free((void**)&act); return; }
    for (int i = 0; i < hs; i++) {
        float g = b_g[i];
        float a = b_a[i];
        for (int j = 0; j < hs; j++) { g += W_h[i * hs + j] * h_in[j]; a += W_h[i * hs + j] * h_in[j]; }
        for (int j = 0; j < hs; j++) { g += W_in[i * hs + j] * input[j]; }
        gate[i] = _hc_sig(g);
        act[i] = _hc_tanh(a);
    }
    for (int i = 0; i < hs; i++) {
        float exp_term = expf(-dt / (tau[i] + 1e-10f));
        h_out[i] = h_in[i] * exp_term + (1.0f - exp_term) * gate[i] * act[i];
    }
    safe_free((void**)&gate); safe_free((void**)&act);
}

static void _hc_cfc_ode_predict(const float* h_in, const float* input,
                                 float* h_out,
                                 const float* W_h, const float* W_in,
                                 const float* b_g, const float* b_a,
                                 const float* tau, int hs, float dt, int steps) {
    float* h_tmp = (float*)safe_malloc(hs * sizeof(float));
    if (!h_tmp) return;
    memcpy(h_tmp, h_in, hs * sizeof(float));
    float dt_sub = dt / (float)(steps > 0 ? steps : 1);
    for (int s = 0; s < steps; s++) {
        float* h_cur = (s == 0) ? h_tmp : h_tmp;
        _hc_cfc_ode_step(h_cur, input, h_tmp, W_h, W_in, b_g, b_a, tau, hs, dt_sub);
    }
    memcpy(h_out, h_tmp, hs * sizeof(float));
    safe_free((void**)&h_tmp);
}

/* 简单DFT（用于纹理FFT分析） */
static void _hc_dft(const float* signal, int n, float* re_out, float* im_out) {
    for (int k = 0; k < n; k++) {
        re_out[k] = 0.0f;
        im_out[k] = 0.0f;
        for (int i = 0; i < n; i++) {
            float angle = -2.0f * (float)M_PI * k * i / (float)n;
            re_out[k] += signal[i] * cosf(angle);
            im_out[k] += signal[i] * sinf(angle);
        }
    }
}

/* 低通滤波器（一阶IIR） */
static float _hc_lowpass(float prev, float input, float alpha) {
    return prev + alpha * (input - prev);
}

/* ============================================================================
 * CfC触觉信号深度处理器
 * ============================================================================ */

struct HapticCfcProcessor {
    HapticCfcConfig config;
    float* hidden_state;
    float* W_h;
    float* W_in;
    float* b_g;
    float* b_a;
    float* tau;
    float* filter_state;
    float prev_vibration_energy;
    int initialized;
};

HapticCfcConfig haptic_cfc_get_default_config(void) {
    HapticCfcConfig cfg;
    cfg.signal_dim = 16;
    cfg.cfc_hidden_size = 64;
    cfg.cfc_time_constant = 0.1f;
    cfg.cfc_noise_std = 0.01f;
    cfg.enable_adaptive_timestep = 1;
    cfg.min_timestep = 0.001f;
    cfg.max_timestep = 0.1f;
    cfg.enable_signal_filtering = 1;
    cfg.filter_cutoff_freq = 100.0f;
    cfg.enable_contact_detection = 1;
    cfg.contact_threshold = 0.05f;
    cfg.enable_slip_detection = 1;
    cfg.slip_detection_threshold = 0.02f;
    return cfg;
}

HapticCfcProcessor* haptic_cfc_create(const HapticCfcConfig* config) {
    if (!config) return NULL;
    HapticCfcProcessor* proc = (HapticCfcProcessor*)safe_calloc(1, sizeof(HapticCfcProcessor));
    if (!proc) return NULL;
    memcpy(&proc->config, config, sizeof(HapticCfcConfig));
    int hs = config->cfc_hidden_size;
    int sd = config->signal_dim;
    if (hs <= 0 || sd <= 0) { safe_free((void**)&proc); return NULL; }
    proc->hidden_state = (float*)safe_calloc(hs, sizeof(float));
    proc->W_h  = (float*)safe_calloc(hs * hs, sizeof(float));
    proc->W_in = (float*)safe_calloc(hs * sd, sizeof(float));
    proc->b_g = (float*)safe_calloc(hs, sizeof(float));
    proc->b_a = (float*)safe_calloc(hs, sizeof(float));
    proc->tau = (float*)safe_calloc(hs, sizeof(float));
    proc->filter_state = (float*)safe_calloc(sd, sizeof(float));
    if (!proc->hidden_state || !proc->W_h || !proc->W_in ||
        !proc->b_g || !proc->b_a || !proc->tau || !proc->filter_state) {
        haptic_cfc_free(proc);
        return NULL;
    }
    for (int i = 0; i < hs; i++) {
        proc->tau[i] = config->cfc_time_constant;
        float scale = 0.01f;
        /* K-012修复：安全随机数初始化 */
        proc->b_g[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        proc->b_a[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < hs; j++)
            proc->W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < sd; j++)
            proc->W_in[i * sd + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
    }
    proc->prev_vibration_energy = 0.0f;
    proc->initialized = 1;
    return proc;
}

void haptic_cfc_free(HapticCfcProcessor* proc) {
    if (!proc) return;
    safe_free((void**)&proc->hidden_state);
    safe_free((void**)&proc->W_h);
    safe_free((void**)&proc->W_in);
    safe_free((void**)&proc->b_g);
    safe_free((void**)&proc->b_a);
    safe_free((void**)&proc->tau);
    safe_free((void**)&proc->filter_state);
    safe_free((void**)&proc);
}

int haptic_cfc_process(HapticCfcProcessor* proc,
                       const HapticReading* reading,
                       float dt,
                       float* features_out, int feature_dim,
                       int* contact_detected,
                       int* slip_detected) {
    if (!proc || !reading || !features_out) return -1;
    int hs = proc->config.cfc_hidden_size;
    int sd = proc->config.signal_dim;
    /* 构建触觉输入向量 */
    float* input = (float*)safe_malloc(sd * sizeof(float));
    if (!input) return -1;
    memset(input, 0, sd * sizeof(float));
    int pos = 0;
    int np = reading->sensor_count > 0 ? reading->sensor_count : 1;
    float inv_np = 1.0f / (float)np;
    for (int i = 0; i < 16 && pos < sd; i++, pos++)
        input[pos] = reading->pressure[i] * inv_np;
    for (int i = 0; i < 4 && pos < sd; i++, pos++)
        input[pos] = reading->temperature[i] / 100.0f;
    for (int i = 0; i < 8 && pos < sd; i++, pos++)
        input[pos] = reading->vibration[i] * inv_np;
    for (int i = 0; i < 6 && pos < sd; i++, pos++)
        input[pos] = reading->force[i] / 50.0f;
    for (int i = 0; i < 6 && pos < sd; i++, pos++)
        input[pos] = reading->torque[i] / 10.0f;
    /* 信号预滤波（低通） */
    float alpha = 0.0f;
    if (proc->config.enable_signal_filtering && dt > 1e-10f) {
        float rc = 1.0f / (2.0f * (float)M_PI * proc->config.filter_cutoff_freq + 1e-10f);
        alpha = dt / (rc + dt);
        for (int i = 0; i < sd; i++)
            input[i] = proc->filter_state[i] = _hc_lowpass(proc->filter_state[i], input[i], alpha);
    }
    /* 接触检测 */
    int contact = 0;
    float contact_energy = 0.0f;
    if (proc->config.enable_contact_detection) {
        for (int i = 0; i < 16; i++)
            contact_energy += reading->pressure[i] * reading->pressure[i];
        for (int i = 0; i < 6; i++)
            contact_energy += reading->force[i] * reading->force[i];
        contact_energy = sqrtf(contact_energy / 22.0f);
        contact = (contact_energy > proc->config.contact_threshold) ? 1 : 0;
    }
    if (contact_detected) *contact_detected = contact;
    /* 滑移检测（振动信号高频能量变化率） */
    int slip = 0;
    if (proc->config.enable_slip_detection) {
        float vib_energy = 0.0f;
        for (int i = 0; i < 8; i++)
            vib_energy += reading->vibration[i] * reading->vibration[i];
        vib_energy = sqrtf(vib_energy / 8.0f);
        float energy_rate = fabsf(vib_energy - proc->prev_vibration_energy) / (dt + 1e-10f);
        if (contact && energy_rate > proc->config.slip_detection_threshold)
            slip = 1;
        proc->prev_vibration_energy = vib_energy;
    }
    if (slip_detected) *slip_detected = slip;
    /* CfC ODE演化 */
    float adaptive_dt = dt;
    if (proc->config.enable_adaptive_timestep) {
        if (adaptive_dt < proc->config.min_timestep)
            adaptive_dt = proc->config.min_timestep;
        if (adaptive_dt > proc->config.max_timestep)
            adaptive_dt = proc->config.max_timestep;
    }
    int steps = proc->config.enable_adaptive_timestep ? 3 : 1;
    _hc_cfc_ode_predict(proc->hidden_state, input, proc->hidden_state,
                        proc->W_h, proc->W_in,
                        proc->b_g, proc->b_a,
                        proc->tau, hs, adaptive_dt, steps);
    /* 输出特征向量（从隐藏状态投影） */
    int out_dim = feature_dim < hs ? feature_dim : hs;
    for (int i = 0; i < out_dim; i++)
        features_out[i] = _hc_tanh(proc->hidden_state[i]);
    if (out_dim < feature_dim)
        memset(features_out + out_dim, 0, (feature_dim - out_dim) * sizeof(float));
    safe_free((void**)&input);
    return 0;
}

void haptic_cfc_reset(HapticCfcProcessor* proc) {
    if (!proc) return;
    int hs = proc->config.cfc_hidden_size;
    int sd = proc->config.signal_dim;
    memset(proc->hidden_state, 0, hs * sizeof(float));
    memset(proc->filter_state, 0, sd * sizeof(float));
    proc->prev_vibration_energy = 0.0f;
}

int haptic_cfc_get_state(const HapticCfcProcessor* proc,
                         float* hidden_state, int state_dim) {
    if (!proc || !hidden_state) return -1;
    int hs = proc->config.cfc_hidden_size;
    int copy_dim = state_dim < hs ? state_dim : hs;
    memcpy(hidden_state, proc->hidden_state, copy_dim * sizeof(float));
    return 0;
}

/* ============================================================================
 * 触觉纹理深度分析
 * ============================================================================ */

struct HapticTextureAnalyzer {
    HapticTextureConfig config;
    HapticTextureDescriptor* known_textures;
    int num_textures;
    float* cfc_hidden;
    float* cfc_W_h;
    float* cfc_W_in;
    float* cfc_b_g;
    float* cfc_b_a;
    float* cfc_tau;
    float* filter_buffer;
    int buffer_pos;
};

HapticTextureConfig haptic_texture_get_default_config(void) {
    HapticTextureConfig cfg;
    cfg.fft_window_size = 256;
    cfg.fft_overlap = 128;
    cfg.frequency_resolution = 0.5f;
    cfg.enable_wavelet_analysis = 1;
    cfg.wavelet_decomposition_level = 4;
    cfg.enable_statistical_features = 1;
    cfg.enable_fractal_analysis = 1;
    cfg.fractal_box_size_min = 2.0f;
    cfg.fractal_box_size_max = 64.0f;
    cfg.num_texture_classes = 20;
    cfg.cfc_hidden_size = 64;
    return cfg;
}

HapticTextureAnalyzer* haptic_texture_create(const HapticTextureConfig* config) {
    if (!config) return NULL;
    HapticTextureAnalyzer* ta = (HapticTextureAnalyzer*)safe_calloc(1, sizeof(HapticTextureAnalyzer));
    if (!ta) return NULL;
    memcpy(&ta->config, config, sizeof(HapticTextureConfig));
    int hs = config->cfc_hidden_size;
    ta->known_textures = (HapticTextureDescriptor*)safe_calloc(
        config->num_texture_classes, sizeof(HapticTextureDescriptor));
    ta->cfc_hidden = (float*)safe_calloc(hs, sizeof(float));
    ta->cfc_W_h = (float*)safe_calloc(hs * hs, sizeof(float));
    ta->cfc_W_in = (float*)safe_calloc(hs * hs, sizeof(float));
    ta->cfc_b_g = (float*)safe_calloc(hs, sizeof(float));
    ta->cfc_b_a = (float*)safe_calloc(hs, sizeof(float));
    ta->cfc_tau = (float*)safe_calloc(hs, sizeof(float));
    int ws = config->fft_window_size;
    ta->filter_buffer = (float*)safe_calloc(ws, sizeof(float));
    if (!ta->known_textures || !ta->cfc_hidden || !ta->cfc_W_h ||
        !ta->cfc_W_in || !ta->cfc_b_g || !ta->cfc_b_a || !ta->cfc_tau || !ta->filter_buffer) {
        haptic_texture_free(ta);
        return NULL;
    }
    for (int i = 0; i < hs; i++) {
        ta->cfc_tau[i] = 0.1f;
        float scale = 0.01f;
        /* K-012修复：安全随机数初始化 */
        ta->cfc_b_g[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        ta->cfc_b_a[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < hs; j++)
            ta->cfc_W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < hs; j++)
            ta->cfc_W_in[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
    }
    ta->buffer_pos = 0;
    return ta;
}

void haptic_texture_free(HapticTextureAnalyzer* ta) {
    if (!ta) return;
    safe_free((void**)&ta->known_textures);
    safe_free((void**)&ta->cfc_hidden);
    safe_free((void**)&ta->cfc_W_h);
    safe_free((void**)&ta->cfc_W_in);
    safe_free((void**)&ta->cfc_b_g);
    safe_free((void**)&ta->cfc_b_a);
    safe_free((void**)&ta->cfc_tau);
    safe_free((void**)&ta->filter_buffer);
    safe_free((void**)&ta);
}

int haptic_texture_analyze(HapticTextureAnalyzer* ta,
                           const HapticReading* reading,
                           float dt,
                           HapticTextureDescriptor* descriptor) {
    if (!ta || !reading || !descriptor) return -1;
    int ws = ta->config.fft_window_size;
    int hs = ta->config.cfc_hidden_size;
    /* 使用振动信号进行纹理分析 */
    memset(descriptor, 0, sizeof(HapticTextureDescriptor));
    /* 填充FFT缓冲区 */
    int fill = 8 < ws ? 8 : ws;
    if (ta->buffer_pos + fill > ws) {
        int remaining = ws - ta->buffer_pos;
        memcpy(ta->filter_buffer + ta->buffer_pos, reading->vibration, remaining * sizeof(float));
        ta->buffer_pos = 0;
    }
    for (int i = 0; i < fill; i++)
        ta->filter_buffer[(ta->buffer_pos + i) % ws] = reading->vibration[i];
    ta->buffer_pos = (ta->buffer_pos + fill) % ws;
    /* FFT分析 */
    float* fft_re = (float*)safe_malloc(ws * sizeof(float));
    float* fft_im = (float*)safe_malloc(ws * sizeof(float));
    if (!fft_re || !fft_im) { safe_free((void**)&fft_re); safe_free((void**)&fft_im); return -1; }
    _hc_dft(ta->filter_buffer, ws, fft_re, fft_im);
    /* 频谱特征提取 */
    float total_energy = 0.0f;
    float spectral_centroid = 0.0f;
    float peak_freq = 0.0f;
    float peak_mag = 0.0f;
    for (int k = 0; k < ws / 2; k++) {
        float mag = sqrtf(fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k]);
        float freq = (float)k * 1.0f / ((float)ws * dt + 1e-10f);
        total_energy += mag;
        spectral_centroid += freq * mag;
        if (mag > peak_mag) { peak_mag = mag; peak_freq = freq; }
    }
    if (total_energy > 1e-10f) spectral_centroid /= total_energy;
    descriptor->spatial_frequency = peak_freq;
    /* 小波分析（Haar小波包分解） */
    if (ta->config.enable_wavelet_analysis) {
        int level = ta->config.wavelet_decomposition_level;
        int len = ws;
        float* approx = (float*)safe_malloc(len * sizeof(float));
        float* detail = (float*)safe_malloc(len * sizeof(float));
        float* sig = (float*)safe_malloc(len * sizeof(float));
        if (approx && detail && sig) {
            memcpy(sig, ta->filter_buffer, len * sizeof(float));
            float wavelet_energy_ratio = 0.0f;
            for (int l = 0; l < level && len > 1; l++) {
                int half = len / 2;
                for (int i = 0; i < half; i++) {
                    float a = (sig[2 * i] + sig[2 * i + 1]) * 0.5f;
                    float d = (sig[2 * i] - sig[2 * i + 1]) * 0.5f;
                    approx[i] = a;
                    detail[i] = d;
                }
                float d_energy = 0.0f;
                for (int i = 0; i < half; i++)
                    d_energy += detail[i] * detail[i];
                wavelet_energy_ratio += d_energy;
                memcpy(sig, approx, half * sizeof(float));
                len = half;
            }
            descriptor->roughness_index = _hc_tanh(wavelet_energy_ratio / ((float)ws + 1e-10f) * 10.0f);
        }
        safe_free((void**)&approx); safe_free((void**)&detail); safe_free((void**)&sig);
    }
    /* 统计特征 */
    if (ta->config.enable_statistical_features) {
        float mean = 0.0f;
        for (int i = 0; i < ws; i++) mean += ta->filter_buffer[i];
        mean /= (float)ws;
        float variance = 0.0f;
        float skewness = 0.0f;
        float kurtosis = 0.0f;
        for (int i = 0; i < ws; i++) {
            float d = ta->filter_buffer[i] - mean;
            variance += d * d;
            skewness += d * d * d;
            kurtosis += d * d * d * d;
        }
        variance /= (float)ws;
        float stddev = sqrtf(variance + 1e-30f);
        skewness /= (float)ws * stddev * stddev * stddev + 1e-30f;
        kurtosis /= (float)ws * variance * variance + 1e-30f;
        kurtosis -= 3.0f;
        descriptor->amplitude_variance = variance;
        descriptor->feature_count = 0;
        descriptor->feature_vector[descriptor->feature_count++] = mean;
        descriptor->feature_vector[descriptor->feature_count++] = variance;
        descriptor->feature_vector[descriptor->feature_count++] = skewness;
        descriptor->feature_vector[descriptor->feature_count++] = kurtosis;
    }
    /* 分形维度（盒计数法） */
    if (ta->config.enable_fractal_analysis) {
        float bmin = ta->config.fractal_box_size_min;
        float bmax = ta->config.fractal_box_size_max;
        int n_scales = 10;
        float log_N = 0.0f, log_inv_eps = 0.0f;
        int valid_scales = 0;
        for (int s = 0; s < n_scales; s++) {
            float scale = bmin * powf(bmax / bmin, (float)s / (float)(n_scales - 1));
            int box_size = (int)(scale + 0.5f);
            if (box_size < 1) box_size = 1;
            if (box_size > ws) break;
            int n_boxes = (ws + box_size - 1) / box_size;
            float* box_vals = (float*)safe_calloc(n_boxes, sizeof(float));
            if (!box_vals) continue;
            for (int i = 0; i < ws; i++) {
                int b = i / box_size;
                float val = fabsf(ta->filter_buffer[i]);
                if (val > box_vals[b]) box_vals[b] = val;
            }
            int count = 0;
            float threshold = 0.01f * (total_energy / (float)ws + 1e-10f);
            for (int b = 0; b < n_boxes; b++)
                if (box_vals[b] > threshold) count++;
            if (count > 0) {
                log_N += logf((float)count);
                log_inv_eps += logf(1.0f / ((float)box_size / (float)ws + 1e-30f));
                valid_scales++;
            }
            safe_free((void**)&box_vals);
        }
        if (valid_scales > 1) {
            float fd = log_N / (log_inv_eps + 1e-30f);
            if (fd < 1.0f) fd = 1.0f;
            if (fd > 2.0f) fd = 2.0f;
            descriptor->feature_vector[descriptor->feature_count++] = fd;
        }
    }
    /* 摩擦系数估计 */
    float friction = 0.0f;
    for (int i = 0; i < 6; i++)
        friction += fabsf(reading->force[i]);
    for (int i = 0; i < 16; i++)
        friction += fabsf(reading->pressure[i]);
    descriptor->friction_coefficient = _hc_tanh(friction / (16.0f + 6.0f) * 2.0f);
    /* 硬度指数 */
    float hardness = 0.0f;
    for (int i = 0; i < 4; i++)
        hardness += reading->temperature[i];
    hardness /= 400.0f;
    descriptor->hardness_index = hardness > 1.0f ? 1.0f : (hardness < 0.0f ? 0.0f : hardness);
    /* 柔顺性指数 */
    descriptor->compliance_index = 1.0f - descriptor->hardness_index;
    /* 热扩散率 */
    descriptor->thermal_diffusivity = 1e-6f * (1.0f + descriptor->hardness_index * 0.5f);
    /* CfC ODE特征演化 */
    float* cfc_input = (float*)safe_malloc(hs * sizeof(float));
    if (cfc_input) {
        memcpy(cfc_input, ta->cfc_hidden, hs * sizeof(float));
        for (int i = 0; i < hs && i < descriptor->feature_count; i++)
            cfc_input[i] = descriptor->feature_vector[i];
        _hc_cfc_ode_predict(ta->cfc_hidden, cfc_input, ta->cfc_hidden,
                            ta->cfc_W_h, ta->cfc_W_in,
                            ta->cfc_b_g, ta->cfc_b_a,
                            ta->cfc_tau, hs, dt, 1);
        for (int i = 0; i < 64 && i < hs; i++)
            descriptor->feature_vector[i] = _hc_tanh(ta->cfc_hidden[i]);
        if (descriptor->feature_count < 64)
            descriptor->feature_count = 64 > descriptor->feature_count ?
                (64 < hs ? 64 : hs) : descriptor->feature_count;
        safe_free((void**)&cfc_input);
    }
    /* 纹理分类（最近邻匹配） */
    float best_sim = 0.0f;
    int best_idx = -1;
    for (int i = 0; i < ta->num_textures; i++) {
        float sim = 0.0f;
        float n1 = 0.0f, n2 = 0.0f;
        int fd = 64 < ta->known_textures[i].feature_count ?
                 64 : ta->known_textures[i].feature_count;
        fd = fd < descriptor->feature_count ? fd : descriptor->feature_count;
        for (int j = 0; j < fd; j++) {
            sim += descriptor->feature_vector[j] * ta->known_textures[i].feature_vector[j];
            n1 += descriptor->feature_vector[j] * descriptor->feature_vector[j];
            n2 += ta->known_textures[i].feature_vector[j] * ta->known_textures[i].feature_vector[j];
        }
        sim /= (sqrtf(n1 + 1e-30f) * sqrtf(n2 + 1e-30f));
        if (sim > best_sim) { best_sim = sim; best_idx = i; }
    }
    if (best_idx >= 0 && best_sim > 0.5f) {
        strncpy(descriptor->texture_class, ta->known_textures[best_idx].texture_class, 63);
        descriptor->texture_class[63] = '\0';
        descriptor->classification_confidence = best_sim;
    } else {
        descriptor->classification_confidence = 0.0f;
    }
    /* 周期性 */
    if (peak_freq > 0.0f)
        descriptor->periodicity = 1.0f / (peak_freq + 1e-10f);
    safe_free((void**)&fft_re); safe_free((void**)&fft_im);
    return 0;
}

int haptic_texture_learn(HapticTextureAnalyzer* ta,
                         const HapticTextureDescriptor* descriptor,
                         const char* texture_name) {
    if (!ta || !descriptor || !texture_name) return -1;
    int nc = ta->config.num_texture_classes;
    if (ta->num_textures >= nc) return -1;
    int idx = ta->num_textures;
    memcpy(&ta->known_textures[idx], descriptor, sizeof(HapticTextureDescriptor));
    strncpy(ta->known_textures[idx].texture_class, texture_name, 63);
    ta->known_textures[idx].texture_class[63] = '\0';
    ta->known_textures[idx].classification_confidence = 1.0f;
    ta->num_textures++;
    return 0;
}

int haptic_texture_list(const HapticTextureAnalyzer* ta,
                        HapticTextureDescriptor* descriptors,
                        int max_count) {
    if (!ta || !descriptors) return -1;
    int count = ta->num_textures < max_count ? ta->num_textures : max_count;
    memcpy(descriptors, ta->known_textures, count * sizeof(HapticTextureDescriptor));
    return count;
}

void haptic_texture_reset(HapticTextureAnalyzer* ta) {
    if (!ta) return;
    int hs = ta->config.cfc_hidden_size;
    int ws = ta->config.fft_window_size;
    memset(ta->cfc_hidden, 0, hs * sizeof(float));
    memset(ta->filter_buffer, 0, ws * sizeof(float));
    ta->buffer_pos = 0;
    ta->num_textures = 0;
}

/* ============================================================================
 * CfC增强物体识别
 * ============================================================================ */

struct HapticObjectRecognizer {
    HapticObjectRecognitionConfig config;
    int num_objects;
    char(*object_names)[64];
    float** object_embeddings;
    float* cfc_hidden;
    int* num_grasps;
    float* cfc_W_h;
    float* cfc_W_in;
    float* cfc_b_g;
    float* cfc_b_a;
    float* cfc_tau;
};

HapticObjectRecognitionConfig haptic_object_recognition_get_default_config(void) {
    HapticObjectRecognitionConfig cfg;
    cfg.cfc_hidden_size = 64;
    cfg.num_known_objects = 100;
    cfg.recognition_threshold = 0.6f;
    cfg.enable_multi_grasp_fusion = 1;
    cfg.max_grasps_per_object = 10;
    cfg.enable_continuous_learning = 1;
    cfg.learning_rate = 0.01f;
    cfg.num_material_features = 32;
    return cfg;
}

HapticObjectRecognizer* haptic_object_recognizer_create(const HapticObjectRecognitionConfig* config) {
    if (!config) return NULL;
    HapticObjectRecognizer* rec = (HapticObjectRecognizer*)safe_calloc(1, sizeof(HapticObjectRecognizer));
    if (!rec) return NULL;
    memcpy(&rec->config, config, sizeof(HapticObjectRecognitionConfig));
    int hs = config->cfc_hidden_size;
    int max_obj = config->num_known_objects;
    rec->object_names = (char(*)[64])safe_calloc(max_obj, 64);
    rec->object_embeddings = (float**)safe_calloc(max_obj, sizeof(float*));
    rec->num_grasps = (int*)safe_calloc(max_obj, sizeof(int));
    rec->cfc_hidden = (float*)safe_calloc(hs, sizeof(float));
    rec->cfc_W_h = (float*)safe_calloc(hs * hs, sizeof(float));
    rec->cfc_W_in = (float*)safe_calloc(hs * hs, sizeof(float));
    rec->cfc_b_g = (float*)safe_calloc(hs, sizeof(float));
    rec->cfc_b_a = (float*)safe_calloc(hs, sizeof(float));
    rec->cfc_tau = (float*)safe_calloc(hs, sizeof(float));
    if (!rec->object_names || !rec->object_embeddings || !rec->num_grasps ||
        !rec->cfc_hidden || !rec->cfc_W_h || !rec->cfc_W_in ||
        !rec->cfc_b_g || !rec->cfc_b_a || !rec->cfc_tau) {
        haptic_object_recognizer_free(rec);
        return NULL;
    }
    for (int i = 0; i < hs; i++) {
        rec->cfc_tau[i] = 0.1f;
        float scale = 0.01f;
        /* K-012修复：安全随机数初始化 */
        rec->cfc_b_g[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        rec->cfc_b_a[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < hs; j++)
            rec->cfc_W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < hs; j++)
            rec->cfc_W_in[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
    }
    return rec;
}

void haptic_object_recognizer_free(HapticObjectRecognizer* rec) {
    if (!rec) return;
    for (int i = 0; i < rec->num_objects && i < rec->config.num_known_objects; i++)
        safe_free((void**)&rec->object_embeddings[i]);
    safe_free((void**)&rec->object_names);
    safe_free((void**)&rec->object_embeddings);
    safe_free((void**)&rec->num_grasps);
    safe_free((void**)&rec->cfc_hidden);
    safe_free((void**)&rec->cfc_W_h);
    safe_free((void**)&rec->cfc_W_in);
    safe_free((void**)&rec->cfc_b_g);
    safe_free((void**)&rec->cfc_b_a);
    safe_free((void**)&rec->cfc_tau);
    safe_free((void**)&rec);
}

int haptic_object_recognize(HapticObjectRecognizer* rec,
                            const HapticReading* readings, int num_readings,
                            const float* dt,
                            char* object_name, int max_name_len,
                            float* confidence) {
    if (!rec || !readings || !object_name || !confidence) return -1;
    int hs = rec->config.cfc_hidden_size;
    /* 重置CfC隐藏状态 */
    memset(rec->cfc_hidden, 0, hs * sizeof(float));
    /* 通过所有读数演化CfC状态 */
    for (int r = 0; r < num_readings; r++) {
        float* input = (float*)safe_calloc(hs, sizeof(float));
        if (!input) continue;
        int np = readings[r].sensor_count > 0 ? readings[r].sensor_count : 1;
        float inv_np = 1.0f / (float)np;
        for (int i = 0; i < 16 && i < hs; i++)
            input[i] = readings[r].pressure[i] * inv_np;
        for (int i = 0; i < 8 && (i + 16) < hs; i++)
            input[16 + i] = readings[r].vibration[i] * inv_np;
        for (int i = 0; i < 6 && (i + 24) < hs; i++)
            input[24 + i] = readings[r].force[i] / 50.0f;
        float t = dt ? dt[r] : 0.01f;
        _hc_cfc_ode_predict(rec->cfc_hidden, input, rec->cfc_hidden,
                            rec->cfc_W_h, rec->cfc_W_in,
                            rec->cfc_b_g, rec->cfc_b_a,
                            rec->cfc_tau, hs, t, 2);
        safe_free((void**)&input);
    }
    /* 提取嵌入向量 */
    float* embedding = (float*)safe_malloc(hs * sizeof(float));
    if (!embedding) return -1;
    for (int i = 0; i < hs; i++)
        embedding[i] = _hc_tanh(rec->cfc_hidden[i]);
    /* 计算与已知物体的余弦相似度 */
    float best_sim = rec->config.recognition_threshold * 0.5f;
    int best_idx = -1;
    float emb_norm = _hc_norm(embedding, hs);
    for (int i = 0; i < rec->num_objects; i++) {
        if (!rec->object_embeddings[i]) continue;
        float obj_norm = _hc_norm(rec->object_embeddings[i], hs);
        float sim = _hc_dot(embedding, rec->object_embeddings[i], hs) /
                    (emb_norm * obj_norm + 1e-30f);
        if (sim > best_sim && sim > rec->config.recognition_threshold) {
            best_sim = sim;
            best_idx = i;
        }
    }
    /* 持续学习：更新匹配物体的嵌入 */
    if (best_idx >= 0 && rec->config.enable_continuous_learning) {
        float lr = rec->config.learning_rate;
        for (int i = 0; i < hs; i++)
            rec->object_embeddings[best_idx][i] += lr * (embedding[i] - rec->object_embeddings[best_idx][i]);
    }
    if (best_idx >= 0) {
        strncpy(object_name, rec->object_names[best_idx], (size_t)(max_name_len - 1));
        object_name[max_name_len - 1] = '\0';
        *confidence = best_sim;
        safe_free((void**)&embedding);
        return 0;
    }
    *confidence = best_sim;
    safe_free((void**)&embedding);
    return 1;
}

int haptic_object_learn(HapticObjectRecognizer* rec,
                        const HapticReading* readings, int num_readings,
                        const float* dt,
                        const char* object_name) {
    if (!rec || !readings || !object_name) return -1;
    int hs = rec->config.cfc_hidden_size;
    int max_obj = rec->config.num_known_objects;
    if (rec->num_objects >= max_obj) return -1;
    /* 通过读数演化得到物体嵌入 */
    memset(rec->cfc_hidden, 0, hs * sizeof(float));
    for (int r = 0; r < num_readings; r++) {
        float* input = (float*)safe_calloc(hs, sizeof(float));
        if (!input) continue;
        int np = readings[r].sensor_count > 0 ? readings[r].sensor_count : 1;
        float inv_np = 1.0f / (float)np;
        for (int i = 0; i < 16 && i < hs; i++)
            input[i] = readings[r].pressure[i] * inv_np;
        for (int i = 0; i < 8 && (i + 16) < hs; i++)
            input[16 + i] = readings[r].vibration[i] * inv_np;
        for (int i = 0; i < 6 && (i + 24) < hs; i++)
            input[24 + i] = readings[r].force[i] / 50.0f;
        float t = dt ? dt[r] : 0.01f;
        _hc_cfc_ode_predict(rec->cfc_hidden, input, rec->cfc_hidden,
                            rec->cfc_W_h, rec->cfc_W_in,
                            rec->cfc_b_g, rec->cfc_b_a,
                            rec->cfc_tau, hs, t, 2);
        safe_free((void**)&input);
    }
    int idx = rec->num_objects;
    rec->object_embeddings[idx] = (float*)safe_malloc(hs * sizeof(float));
    if (!rec->object_embeddings[idx]) return -1;
    for (int i = 0; i < hs; i++)
        rec->object_embeddings[idx][i] = _hc_tanh(rec->cfc_hidden[i]);
    strncpy(rec->object_names[idx], object_name, 63);
    rec->object_names[idx][63] = '\0';
    rec->num_grasps[idx] = num_readings;
    rec->num_objects++;
    return 0;
}

void haptic_object_recognizer_reset(HapticObjectRecognizer* rec) {
    if (!rec) return;
    int hs = rec->config.cfc_hidden_size;
    memset(rec->cfc_hidden, 0, hs * sizeof(float));
    for (int i = 0; i < rec->num_objects && i < rec->config.num_known_objects; i++)
        safe_free((void**)&rec->object_embeddings[i]);
    rec->num_objects = 0;
}

/* ============================================================================
 * CfC增强抓取学习
 * ============================================================================ */

struct HapticGraspLearner {
    HapticGraspLearningConfig config;
    float* cfc_hidden;
    float* W_h;
    float* W_in_f;
    float* W_in_p;
    float* b_g;
    float* b_a;
    float* tau;
    float* W_out;
    float* b_out;
    float prev_vib_energy;
    int num_demo_trajectories;
    float* demo_buffer;
    int demo_steps;
};

HapticGraspLearningConfig haptic_grasp_learning_get_default_config(void) {
    HapticGraspLearningConfig cfg;
    cfg.cfc_hidden_size = 64;
    cfg.num_fingers = 5;
    cfg.force_control_gain = 0.8f;
    cfg.position_control_gain = 0.6f;
    cfg.enable_slip_compensation = 1;
    cfg.slip_compensation_gain = 0.3f;
    cfg.enable_adaptive_grip = 1;
    cfg.min_grip_force = 0.5f;
    cfg.max_grip_force = 20.0f;
    cfg.enable_learning_from_demo = 1;
    cfg.imitation_learning_rate = 0.05f;
    return cfg;
}

HapticGraspLearner* haptic_grasp_learner_create(const HapticGraspLearningConfig* config) {
    if (!config) return NULL;
    HapticGraspLearner* gl = (HapticGraspLearner*)safe_calloc(1, sizeof(HapticGraspLearner));
    if (!gl) return NULL;
    memcpy(&gl->config, config, sizeof(HapticGraspLearningConfig));
    int hs = config->cfc_hidden_size;
    int nf = config->num_fingers;
    int input_dim = nf * 4 + 8;
    gl->cfc_hidden = (float*)safe_calloc(hs, sizeof(float));
    gl->W_h = (float*)safe_calloc(hs * hs, sizeof(float));
    gl->W_in_f = (float*)safe_calloc(hs * nf, sizeof(float));
    gl->W_in_p = (float*)safe_calloc(hs * nf * 3, sizeof(float));
    gl->b_g = (float*)safe_calloc(hs, sizeof(float));
    gl->b_a = (float*)safe_calloc(hs, sizeof(float));
    gl->tau = (float*)safe_calloc(hs, sizeof(float));
    gl->W_out = (float*)safe_calloc(nf * 4 * hs, sizeof(float));
    gl->b_out = (float*)safe_calloc(nf * 4, sizeof(float));
    gl->demo_buffer = (float*)safe_calloc(1000 * input_dim, sizeof(float));
    if (!gl->cfc_hidden || !gl->W_h || !gl->W_in_f || !gl->W_in_p ||
        !gl->b_g || !gl->b_a || !gl->tau || !gl->W_out || !gl->b_out || !gl->demo_buffer) {
        haptic_grasp_learner_free(gl);
        return NULL;
    }
    for (int i = 0; i < hs; i++) {
        gl->tau[i] = 0.1f;
        float scale = 0.01f;
        /* K-012修复：安全随机数初始化 */
        gl->b_g[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        gl->b_a[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < hs; j++)
            gl->W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < nf; j++)
            gl->W_in_f[i * nf + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < nf * 3; j++)
            gl->W_in_p[i * nf * 3 + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < nf * 4; j++) {
            gl->W_out[j * hs + i] = (secure_random_float() * 2.0f - 1.0f) * scale;
            if (i == 0) gl->b_out[j] = 0.0f;
        }
    }
    gl->prev_vib_energy = 0.0f;
    return gl;
}

void haptic_grasp_learner_free(HapticGraspLearner* gl) {
    if (!gl) return;
    safe_free((void**)&gl->cfc_hidden);
    safe_free((void**)&gl->W_h);
    safe_free((void**)&gl->W_in_f);
    safe_free((void**)&gl->W_in_p);
    safe_free((void**)&gl->b_g);
    safe_free((void**)&gl->b_a);
    safe_free((void**)&gl->tau);
    safe_free((void**)&gl->W_out);
    safe_free((void**)&gl->b_out);
    safe_free((void**)&gl->demo_buffer);
    safe_free((void**)&gl);
}

int haptic_grasp_control(HapticGraspLearner* gl,
                         const HapticReading* reading,
                         const float* finger_positions,
                         const float* contact_forces,
                         float dt,
                         float* target_forces,
                         float* target_positions) {
    if (!gl || !reading || !finger_positions || !contact_forces ||
        !target_forces || !target_positions) return -1;
    int hs = gl->config.cfc_hidden_size;
    int nf = gl->config.num_fingers;
    int input_dim = nf * 4 + 8;
    float* input = (float*)safe_malloc(input_dim * sizeof(float));
    if (!input) return -1;
    memset(input, 0, input_dim * sizeof(float));
    for (int i = 0; i < nf; i++) {
        input[i] = contact_forces[i] / gl->config.max_grip_force;
        input[nf + i * 3]     = finger_positions[i * 3];
        input[nf + i * 3 + 1] = finger_positions[i * 3 + 1];
        input[nf + i * 3 + 2] = finger_positions[i * 3 + 2];
    }
    int vib_offset = nf * 4;
    for (int i = 0; i < 8 && vib_offset + i < input_dim; i++)
        input[vib_offset + i] = reading->vibration[i];
    /* CfC ODE演化 */
    _hc_cfc_ode_predict(gl->cfc_hidden, input, gl->cfc_hidden,
                        gl->W_h, gl->W_h,
                        gl->b_g, gl->b_a,
                        gl->tau, hs, dt, 2);
    /* 从隐藏状态解码控制信号 */
    float* control = (float*)safe_malloc(nf * 4 * sizeof(float));
    if (!control) { safe_free((void**)&input); return -1; }
    for (int i = 0; i < nf * 4; i++) {
        float s = gl->b_out[i];
        for (int j = 0; j < hs; j++)
            s += gl->W_out[i * hs + j] * _hc_tanh(gl->cfc_hidden[j]);
        control[i] = _hc_tanh(s);
    }
    /* 滑移检测与补偿 */
    if (gl->config.enable_slip_compensation) {
        float vib_energy = 0.0f;
        for (int i = 0; i < 8; i++)
            vib_energy += reading->vibration[i] * reading->vibration[i];
        vib_energy = sqrtf(vib_energy / 8.0f);
        float energy_rate = fabsf(vib_energy - gl->prev_vib_energy) / (dt + 1e-10f);
        float slip_factor = _hc_tanh(energy_rate * gl->config.slip_compensation_gain);
        for (int i = 0; i < nf; i++)
            control[i] += slip_factor * 0.2f;
        gl->prev_vib_energy = vib_energy;
    }
    /* 生成目标力和位置 */
    for (int i = 0; i < nf; i++) {
        float f_target = (control[i] + 1.0f) * 0.5f * gl->config.max_grip_force;
        if (gl->config.enable_adaptive_grip) {
            if (f_target < gl->config.min_grip_force)
                f_target = gl->config.min_grip_force;
            if (f_target > gl->config.max_grip_force)
                f_target = gl->config.max_grip_force;
        }
        target_forces[i] = f_target;
        target_positions[i * 3]     = finger_positions[i * 3] + control[nf + i * 3] * gl->config.position_control_gain;
        target_positions[i * 3 + 1] = finger_positions[i * 3 + 1] + control[nf + i * 3 + 1] * gl->config.position_control_gain;
        target_positions[i * 3 + 2] = finger_positions[i * 3 + 2] + control[nf + i * 3 + 2] * gl->config.position_control_gain;
    }
    safe_free((void**)&input); safe_free((void**)&control);
    return 0;
}

int haptic_grasp_learn_from_demo(HapticGraspLearner* gl,
                                 const HapticReading* readings,
                                 const float* finger_positions,
                                 const float* contact_forces,
                                 int num_steps, float dt,
                                 float* grasp_quality) {
    if (!gl || !readings || !finger_positions || !contact_forces || num_steps <= 0) return -1;
    int hs = gl->config.cfc_hidden_size;
    int nf = gl->config.num_fingers;
    int input_dim = nf * 4 + 8;
    float lr = gl->config.imitation_learning_rate;
    float total_loss = 0.0f;
    /* 示教学习：通过梯度下降拟合示教轨迹 */
    for (int epoch = 0; epoch < 5; epoch++) {
        memset(gl->cfc_hidden, 0, hs * sizeof(float));
        for (int s = 0; s < num_steps; s++) {
            float* inp = (float*)safe_malloc(input_dim * sizeof(float));
            if (!inp) continue;
            memset(inp, 0, input_dim * sizeof(float));
            for (int i = 0; i < nf; i++) {
                int fi = s * nf * 3 + i * 3;
                inp[i] = contact_forces[s * nf + i] / gl->config.max_grip_force;
                inp[nf + i * 3]     = finger_positions[fi];
                inp[nf + i * 3 + 1] = finger_positions[fi + 1];
                inp[nf + i * 3 + 2] = finger_positions[fi + 2];
            }
            int vib_offset = nf * 4;
            for (int i = 0; i < 8 && vib_offset + i < input_dim; i++)
                inp[vib_offset + i] = readings[s].vibration[i];
            float* h_before = (float*)safe_malloc(hs * sizeof(float));
            if (h_before) memcpy(h_before, gl->cfc_hidden, hs * sizeof(float));
            _hc_cfc_ode_predict(gl->cfc_hidden, inp, gl->cfc_hidden,
                                gl->W_h, gl->W_h,
                                gl->b_g, gl->b_a,
                                gl->tau, hs, dt, 2);
            float* control_target = (float*)safe_malloc(nf * 4 * sizeof(float));
            if (control_target && h_before) {
                for (int i = 0; i < nf; i++) {
                    control_target[i] = contact_forces[s * nf + i] / gl->config.max_grip_force;
                    control_target[nf + i * 3]     = 0.0f;
                    control_target[nf + i * 3 + 1] = 0.0f;
                    control_target[nf + i * 3 + 2] = 0.0f;
                }
                for (int i = 0; i < nf * 4; i++) {
                    float pred = gl->b_out[i];
                    for (int j = 0; j < hs; j++)
                        pred += gl->W_out[i * hs + j] * _hc_tanh(gl->cfc_hidden[j]);
                    pred = _hc_tanh(pred);
                    float error = pred - control_target[i];
                    total_loss += error * error;
                    float delta = lr * error * (1.0f - pred * pred);
                    for (int j = 0; j < hs; j++)
                        gl->W_out[i * hs + j] -= delta * _hc_tanh(gl->cfc_hidden[j]);
                    gl->b_out[i] -= delta;
                }
            }
            safe_free((void**)&inp); safe_free((void**)&h_before); safe_free((void**)&control_target);
        }
    }
    if (grasp_quality) {
        *grasp_quality = 1.0f / (1.0f + total_loss / (float)(num_steps * nf * 4 + 1));
    }
    gl->demo_steps = num_steps;
    return 0;
}

float haptic_grasp_evaluate(HapticGraspLearner* gl,
                            const HapticReading* reading,
                            const float* contact_forces,
                            int num_fingers) {
    if (!gl || !reading || !contact_forces) return -1.0f;
    int nf = gl->config.num_fingers;
    int fn = num_fingers < nf ? num_fingers : nf;
    float total_force = 0.0f;
    for (int i = 0; i < fn; i++)
        total_force += fabsf(contact_forces[i]);
    float avg_force = total_force / (float)fn;
    float vib_energy = 0.0f;
    for (int i = 0; i < 8; i++)
        vib_energy += reading->vibration[i] * reading->vibration[i];
    vib_energy = sqrtf(vib_energy / 8.0f + 1e-30f);
    float force_score = avg_force / gl->config.max_grip_force;
    if (force_score > 1.0f) force_score = 1.0f;
    float stability_score = 1.0f / (1.0f + vib_energy * 5.0f);
    float quality = 0.6f * force_score + 0.4f * stability_score;
    return quality;
}

void haptic_grasp_learner_reset(HapticGraspLearner* gl) {
    if (!gl) return;
    int hs = gl->config.cfc_hidden_size;
    memset(gl->cfc_hidden, 0, hs * sizeof(float));
    gl->prev_vib_energy = 0.0f;
    gl->demo_steps = 0;
}

/* ============================================================================
 * 视触联合融合
 * ============================================================================ */

struct VisionHapticFusion {
    VisionHapticFusionConfig config;
    float* fusion_hidden;
    float* W_h;
    float* W_vis;
    float* W_hap;
    float* b_g;
    float* b_a;
    float* tau;
    float* W_out;
    float* b_out;
};

VisionHapticFusion* vision_haptic_fusion_create(const VisionHapticFusionConfig* config) {
    if (!config) return NULL;
    VisionHapticFusion* vf = (VisionHapticFusion*)safe_calloc(1, sizeof(VisionHapticFusion));
    if (!vf) return NULL;
    memcpy(&vf->config, config, sizeof(VisionHapticFusionConfig));
    int hs = config->cfc_hidden_size;
    int vd = config->visual_feature_dim;
    int hd = config->haptic_feature_dim;
    int fd = config->fused_feature_dim;
    if (hs <= 0 || vd <= 0 || hd <= 0 || fd <= 0) { safe_free((void**)&vf); return NULL; }
    vf->fusion_hidden = (float*)safe_calloc(hs, sizeof(float));
    vf->W_h  = (float*)safe_calloc(hs * hs, sizeof(float));
    vf->W_vis = (float*)safe_calloc(hs * vd, sizeof(float));
    vf->W_hap = (float*)safe_calloc(hs * hd, sizeof(float));
    vf->b_g = (float*)safe_calloc(hs, sizeof(float));
    vf->b_a = (float*)safe_calloc(hs, sizeof(float));
    vf->tau = (float*)safe_calloc(hs, sizeof(float));
    vf->W_out = (float*)safe_calloc(fd * hs, sizeof(float));
    vf->b_out = (float*)safe_calloc(fd, sizeof(float));
    if (!vf->fusion_hidden || !vf->W_h || !vf->W_vis || !vf->W_hap ||
        !vf->b_g || !vf->b_a || !vf->tau || !vf->W_out || !vf->b_out) {
        vision_haptic_fusion_free(vf);
        return NULL;
    }
    for (int i = 0; i < hs; i++) {
        vf->tau[i] = config->fusion_temporal_constant > 0 ? config->fusion_temporal_constant : 0.05f;
        float scale = 0.01f;
        /* K-012修复：安全随机数初始化 */
        vf->b_g[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        vf->b_a[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < hs; j++)
            vf->W_h[i * hs + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < vd; j++)
            vf->W_vis[i * vd + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < hd; j++)
            vf->W_hap[i * hd + j] = (secure_random_float() * 2.0f - 1.0f) * scale;
        for (int j = 0; j < fd; j++) {
            vf->W_out[j * hs + i] = (secure_random_float() * 2.0f - 1.0f) * scale;
            if (i == 0) vf->b_out[j] = 0.0f;
        }
    }
    return vf;
}

void vision_haptic_fusion_free(VisionHapticFusion* vf) {
    if (!vf) return;
    safe_free((void**)&vf->fusion_hidden);
    safe_free((void**)&vf->W_h);
    safe_free((void**)&vf->W_vis);
    safe_free((void**)&vf->W_hap);
    safe_free((void**)&vf->b_g);
    safe_free((void**)&vf->b_a);
    safe_free((void**)&vf->tau);
    safe_free((void**)&vf->W_out);
    safe_free((void**)&vf->b_out);
    safe_free((void**)&vf);
}

int vision_haptic_fusion_fuse(VisionHapticFusion* vf,
                              const float* visual_features,
                              const float* haptic_features,
                              float* fused_features,
                              float dt) {
    if (!vf || !visual_features || !haptic_features || !fused_features) return -1;
    int hs = vf->config.cfc_hidden_size;
    int vd = vf->config.visual_feature_dim;
    int hd = vf->config.haptic_feature_dim;
    int fd = vf->config.fused_feature_dim;
    float* input = (float*)safe_malloc((vd + hd) * sizeof(float));
    if (!input) return -1;
    memcpy(input, visual_features, vd * sizeof(float));
    memcpy(input + vd, haptic_features, hd * sizeof(float));
    /* CfC ODE融合（联合视觉和触觉输入演化隐藏状态） */
    if (vf->config.enable_cfc_fusion) {
        float* combined_input = (float*)safe_malloc(hs * sizeof(float));
        if (!combined_input) { safe_free((void**)&input); return -1; }
        memset(combined_input, 0, hs * sizeof(float));
        int copy_v = vd < hs ? vd : hs;
        for (int j = 0; j < copy_v; j++)
            combined_input[j] = visual_features[j];
        int copy_h = hd < (hs - copy_v) ? hd : (hs - copy_v);
        for (int j = 0; j < copy_h; j++)
            combined_input[copy_v + j] = haptic_features[j];
        _hc_cfc_ode_predict(vf->fusion_hidden, combined_input, vf->fusion_hidden,
                            vf->W_h, vf->W_h,
                            vf->b_g, vf->b_a,
                            vf->tau, hs, dt, 2);
        safe_free((void**)&combined_input);
    } else {
        for (int i = 0; i < hs; i++)
            vf->fusion_hidden[i] = _hc_tanh(vf->fusion_hidden[i] + input[i % (vd + hd)]);
    }
    /* 解码融合特征 */
    for (int i = 0; i < fd; i++) {
        float s = vf->b_out[i];
        for (int j = 0; j < hs; j++)
            s += vf->W_out[i * hs + j] * _hc_tanh(vf->fusion_hidden[j]);
        fused_features[i] = _hc_tanh(s);
    }
    safe_free((void**)&input);
    return 0;
}

void vision_haptic_fusion_reset(VisionHapticFusion* vf) {
    if (!vf) return;
    int hs = vf->config.cfc_hidden_size;
    memset(vf->fusion_hidden, 0, hs * sizeof(float));
}