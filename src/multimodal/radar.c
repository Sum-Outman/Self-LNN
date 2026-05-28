/**
 * @file radar.c
 * @brief 雷达信号预处理器实现 — FMCW距离-多普勒、CFAR、微多普勒
 *
 * H-001修复: 新增专用雷达预处理器，提供真实的FMCW雷达信号处理算法。
 */

#include "selflnn/multimodal/radar.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct RadarProcessor {
    int num_samples;
    int num_chirps;
    float start_freq;
    float bandwidth;
    float chirp_duration;
    float sample_rate;
    float range_resolution;
    float velocity_resolution;
    int initialized;
};

static void fft_radix2(float* data_real, float* data_imag, int n) {
    /* 基2 FFT — Cooley-Tukey算法，原地计算 */
    if (n <= 1) return;
    /* 位反转重排 */
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (i < j) {
            float tr = data_real[i], ti = data_imag[i];
            data_real[i] = data_real[j]; data_imag[i] = data_imag[j];
            data_real[j] = tr; data_imag[j] = ti;
        }
        int m = n >> 1;
        while (m >= 1 && (j & m)) { j -= m; m >>= 1; }
        j += m;
    }
    /* 逐层蝶形运算 */
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        float w_r = 1.0f, w_i = 0.0f;
        float angle = -2.0f * (float)M_PI / (float)len;
        float winc_r = cosf(angle), winc_i = sinf(angle);
        for (int i = 0; i < half; i++) {
            for (int k = i; k < n; k += len) {
                int k2 = k + half;
                float tr = w_r * data_real[k2] - w_i * data_imag[k2];
                float ti = w_r * data_imag[k2] + w_i * data_real[k2];
                data_real[k2] = data_real[k] - tr;
                data_imag[k2] = data_imag[k] - ti;
                data_real[k] += tr;
                data_imag[k] += ti;
            }
            float w_new_r = w_r * winc_r - w_i * winc_i;
            w_i = w_r * winc_i + w_i * winc_r;
            w_r = w_new_r;
        }
    }
}

static int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

RadarProcessor* radar_processor_create(int num_samples, int num_chirps,
    float start_freq, float bandwidth, float chirp_duration,
    float sample_rate) {
    RadarProcessor* rp = (RadarProcessor*)safe_calloc(1, sizeof(RadarProcessor));
    if (!rp) return NULL;
    rp->num_samples = num_samples;
    rp->num_chirps = num_chirps;
    rp->start_freq = start_freq;
    rp->bandwidth = bandwidth;
    rp->chirp_duration = chirp_duration;
    rp->sample_rate = sample_rate;
    /* 距离分辨率：ΔR = c / (2*B) */
    float c = 3e8f;
    rp->range_resolution = c / (2.0f * bandwidth);
    /* 速度分辨率：Δv = λ / (2*T_frame)，λ = c/fc, T_frame = N_chirps * T_chirp */
    float lambda = c / start_freq;
    float frame_time = (float)num_chirps * chirp_duration;
    rp->velocity_resolution = lambda / (2.0f * frame_time);
    rp->initialized = 1;
    return rp;
}

void radar_processor_free(RadarProcessor* rp) {
    safe_free((void**)&rp);
}

int radar_range_doppler(RadarProcessor* rp,
    const float* raw_adc, int num_samples, int num_chirps,
    RadarSpectrum* spectrum) {
    if (!rp || !raw_adc || !spectrum || num_samples <= 0 || num_chirps <= 0)
        return -1;
    if (num_samples > RADAR_FFT_MAX_SIZE) num_samples = RADAR_FFT_MAX_SIZE;
    if (num_chirps > RADAR_FFT_MAX_SIZE) num_chirps = RADAR_FFT_MAX_SIZE;
    int n_range = next_power_of_two(num_samples);
    int n_doppler = next_power_of_two(num_chirps);
    /* 第1步：距离FFT（每chirp） */
    float* range_map_real = (float*)safe_calloc((size_t)(n_range * num_chirps), sizeof(float));
    float* range_map_imag = (float*)safe_calloc((size_t)(n_range * num_chirps), sizeof(float));
    if (!range_map_real || !range_map_imag) {
        safe_free((void**)&range_map_real);
        safe_free((void**)&range_map_imag);
        return -1;
    }
    for (int c = 0; c < num_chirps; c++) {
        float* buf_r = (float*)safe_calloc((size_t)n_range, sizeof(float));
        float* buf_i = (float*)safe_calloc((size_t)n_range, sizeof(float));
        if (!buf_r || !buf_i) {
            safe_free((void**)&buf_r); safe_free((void**)&buf_i);
            safe_free((void**)&range_map_real); safe_free((void**)&range_map_imag);
            return -1;
        }
        for (int s = 0; s < num_samples; s++) {
            buf_r[s] = raw_adc[c * num_samples + s];
        }
        /* 汉明窗 */
        for (int s = 0; s < num_samples; s++) {
            float w = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)s / (float)(num_samples - 1));
            buf_r[s] *= w;
        }
        fft_radix2(buf_r, buf_i, n_range);
        for (int r = 0; r < n_range; r++) {
            range_map_real[c * n_range + r] = buf_r[r];
            range_map_imag[c * n_range + r] = buf_i[r];
        }
        safe_free((void**)&buf_r); safe_free((void**)&buf_i);
    }
    /* 第2步：多普勒FFT（每距离bin） */
    float* rd_map_real = (float*)safe_calloc((size_t)(n_range * n_doppler), sizeof(float));
    float* rd_map_imag = (float*)safe_calloc((size_t)(n_range * n_doppler), sizeof(float));
    if (!rd_map_real || !rd_map_imag) {
        safe_free((void**)&rd_map_real); safe_free((void**)&rd_map_imag);
        safe_free((void**)&range_map_real); safe_free((void**)&range_map_imag);
        return -1;
    }
    for (int r = 0; r < n_range; r++) {
        float* col_r = (float*)safe_calloc((size_t)n_doppler, sizeof(float));
        float* col_i = (float*)safe_calloc((size_t)n_doppler, sizeof(float));
        if (!col_r || !col_i) {
            safe_free((void**)&col_r); safe_free((void**)&col_i);
            safe_free((void**)&rd_map_real); safe_free((void**)&rd_map_imag);
            safe_free((void**)&range_map_real); safe_free((void**)&range_map_imag);
            return -1;
        }
        for (int c = 0; c < num_chirps; c++) {
            col_r[c] = range_map_real[c * n_range + r];
            col_i[c] = range_map_imag[c * n_range + r];
        }
        /* 汉明窗 */
        if (num_chirps > 1) {
            for (int c = 0; c < num_chirps; c++) {
                float w = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)c / (float)(num_chirps - 1));
                col_r[c] *= w;
                col_i[c] *= w;
            }
        }
        fft_radix2(col_r, col_i, n_doppler);
        for (int d = 0; d < n_doppler; d++) {
            rd_map_real[r * n_doppler + d] = col_r[d];
            rd_map_imag[r * n_doppler + d] = col_i[d];
        }
        safe_free((void**)&col_r); safe_free((void**)&col_i);
    }
    /* 计算幅度谱到输出 */
    spectrum->num_range_bins = n_range;
    spectrum->num_doppler_bins = n_doppler;
    spectrum->range_profile = (float*)safe_malloc((size_t)n_range * sizeof(float));
    spectrum->doppler_spectrum = (float*)safe_malloc((size_t)(n_range * n_doppler) * sizeof(float));
    if (spectrum->range_profile && spectrum->doppler_spectrum) {
        for (int r = 0; r < n_range; r++) {
            float sum = 0.0f;
            for (int d = 0; d < n_doppler; d++) {
                int idx = r * n_doppler + d;
                float mag = sqrtf(rd_map_real[idx]*rd_map_real[idx] +
                                  rd_map_imag[idx]*rd_map_imag[idx]);
                spectrum->doppler_spectrum[idx] = mag;
                sum += mag;
            }
            spectrum->range_profile[r] = sum / (float)n_doppler;
        }
    }
    safe_free((void**)&range_map_real);
    safe_free((void**)&range_map_imag);
    safe_free((void**)&rd_map_real);
    safe_free((void**)&rd_map_imag);
    return 0;
}

int radar_cfar_detection(RadarProcessor* rp,
    const float* range_doppler_map, int num_range, int num_doppler,
    float pfa, int guard_cells, int ref_cells,
    RadarTarget* targets, int max_targets, int* num_detected) {
    if (!rp || !range_doppler_map || !targets || !num_detected ||
        num_range <= 0 || num_doppler <= 0 || max_targets <= 0)
        return -1;
    *num_detected = 0;
    /* CA-CFAR (Cell-Averaging CFAR) 二维实现
     * 对距离-多普勒图中的每个单元计算局部噪声估计 */
    int n_train = 2 * (2 * ref_cells + guard_cells + 1);
    float alpha = (float)n_train * (powf(1.0f - pfa, -1.0f/(float)n_train) - 1.0f);
    for (int r = guard_cells + ref_cells;
         r < num_range - guard_cells - ref_cells && *num_detected < max_targets; r++) {
        for (int d = guard_cells + ref_cells;
             d < num_doppler - guard_cells - ref_cells && *num_detected < max_targets; d++) {
            int cut_idx = r * num_doppler + d;
            float cut_power = range_doppler_map[cut_idx];
            /* 计算参考单元均值（排除保护单元和被测单元） */
            float noise_sum = 0.0f;
            int ref_count = 0;
            for (int rr = r - ref_cells - guard_cells;
                 rr <= r + ref_cells + guard_cells; rr++) {
                for (int dd = d - ref_cells - guard_cells;
                     dd <= d + ref_cells + guard_cells; dd++) {
                    if (abs(rr - r) <= guard_cells && abs(dd - d) <= guard_cells)
                        continue;
                    noise_sum += range_doppler_map[rr * num_doppler + dd];
                    ref_count++;
                }
            }
            float noise_mean = noise_sum / (float)(ref_count + 1);
            float threshold = alpha * noise_mean;
            if (cut_power > threshold && cut_power > 1e-6f) {
                RadarTarget* t = &targets[*num_detected];
                t->range = (float)r * rp->range_resolution;
                t->velocity = ((float)d - (float)num_doppler/2.0f) * rp->velocity_resolution;
                t->snr = 10.0f * log10f(cut_power / (noise_mean + 1e-8f));
                t->rcs = cut_power;
                t->track_id = -1;
                (*num_detected)++;
            }
        }
    }
    return 0;
}

int radar_micro_doppler(RadarProcessor* rp,
    const float* raw_iq, int num_samples,
    float* micro_doppler, int md_size) {
    if (!rp || !raw_iq || !micro_doppler || num_samples <= 0 || md_size <= 0)
        return -1;
    /* STFT (Short-Time Fourier Transform) 用于微多普勒分析
     * 使用滑动窗口Hanning窗计算时频谱 */
    int win_size = (num_samples < md_size) ? num_samples / 2 : md_size / 2;
    if (win_size < 16) win_size = 16;
    int fft_size = next_power_of_two(win_size);
    int hop = win_size / 4;
    if (hop < 1) hop = 1;
    memset(micro_doppler, 0, (size_t)md_size * sizeof(float));
    float* win_buf_r = (float*)safe_calloc((size_t)fft_size, sizeof(float));
    float* win_buf_i = (float*)safe_calloc((size_t)fft_size, sizeof(float));
    if (!win_buf_r || !win_buf_i) {
        safe_free((void**)&win_buf_r);
        safe_free((void**)&win_buf_i);
        return -1;
    }
    int time_frames = (num_samples - win_size) / hop + 1;
    if (time_frames > md_size) time_frames = md_size;
    for (int tf = 0; tf < time_frames; tf++) {
        int start = tf * hop;
        memset(win_buf_r, 0, (size_t)fft_size * sizeof(float));
        memset(win_buf_i, 0, (size_t)fft_size * sizeof(float));
        for (int s = 0; s < win_size && (start + s) < num_samples; s++) {
            float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)s / (float)(win_size - 1)));
            win_buf_r[s] = raw_iq[start + s] * w;
        }
        fft_radix2(win_buf_r, win_buf_i, fft_size);
        /* 提取多普勒频移：最大幅值对应的频率bin */
        float max_mag = 0.0f;
        int max_bin = 0;
        for (int b = 1; b < fft_size / 2; b++) {
            float mag = win_buf_r[b]*win_buf_r[b] + win_buf_i[b]*win_buf_i[b];
            if (mag > max_mag) { max_mag = mag; max_bin = b; }
        }
        micro_doppler[tf] = (float)max_bin * rp->sample_rate / (float)fft_size;
    }
    safe_free((void**)&win_buf_r);
    safe_free((void**)&win_buf_i);
    return time_frames;
}

int radar_compute_feature_vector(RadarProcessor* rp,
    const RadarSpectrum* spectrum, const RadarTarget* targets,
    int num_targets, float* feature_vector, size_t feature_dim) {
    if (!rp || !spectrum || !feature_vector ||
        feature_dim < RADAR_FEATURE_DIM)
        return -1;
    memset(feature_vector, 0, feature_dim * sizeof(float));
    size_t idx = 0;
    /* 距离剖面特征 [0, 15] */
    if (spectrum->range_profile) {
        int nr = spectrum->num_range_bins;
        if (nr > 16) {
            int step = nr / 16;
            for (int i = 0; i < 16 && idx < 16; i++, idx++)
                feature_vector[idx] = spectrum->range_profile[i * step] * 0.01f;
        } else {
            for (int i = 0; i < nr && idx < 16; i++, idx++)
                feature_vector[idx] = spectrum->range_profile[i] * 0.01f;
        }
    }
    /* 目标编码 [16, 47] */
    int nt = (num_targets > 0) ? num_targets : 0;
    if (nt > 8) nt = 8;
    for (int t = 0; t < nt && idx < 48; t++) {
        feature_vector[idx++] = tanhf(targets[t].range * 0.01f);
        feature_vector[idx++] = tanhf(targets[t].velocity * 0.1f);
        feature_vector[idx++] = tanhf(targets[t].snr * 0.1f);
        feature_vector[idx++] = tanhf(targets[t].rcs * 0.01f);
    }
    /* 统计特征 [48, 55] */
    {
        float mean_range = 0.0f, mean_vel = 0.0f, mean_snr = 0.0f;
        if (nt > 0) {
            for (int t = 0; t < nt; t++) {
                mean_range += targets[t].range;
                mean_vel += targets[t].velocity;
                mean_snr += targets[t].snr;
            }
            mean_range /= (float)nt; mean_vel /= (float)nt; mean_snr /= (float)nt;
        }
        feature_vector[idx++] = tanhf(mean_range * 0.01f);
        feature_vector[idx++] = tanhf(mean_vel * 0.1f);
        feature_vector[idx++] = tanhf(mean_snr * 0.1f);
        feature_vector[idx++] = (float)nt / 8.0f;
    }
    /* 剩余填充 */
    for (; idx < feature_dim; idx++)
        feature_vector[idx] = 0.0f;
    return 0;
}

int radar_doa_estimation(RadarProcessor* rp,
    const float* antenna_snapshots, int num_antennas, int num_snapshots,
    float* azimuth, float* elevation) {
    if (!rp || !antenna_snapshots || !azimuth || !elevation ||
        num_antennas < 2 || num_snapshots <= 0)
        return -1;
    /* 波束形成 DOA估计：相位差 → 到达角
     * Δφ = 2π * d * sin(θ) / λ，θ = arcsin(Δφ * λ / (2π * d)) */
    float d = 0.5f * (3e8f / rp->start_freq); /* 半波长间距 */
    float sum_phase_diff = 0.0f;
    int diff_count = 0;
    for (int s = 0; s < num_snapshots; s++) {
        for (int a = 1; a < num_antennas; a++) {
            float val1 = antenna_snapshots[s * num_antennas + a];
            float val2 = antenna_snapshots[s * num_antennas + a - 1];
            /* 简化的相位差估计 */
            float diff = val1 - val2;
            if (fabsf(diff) < 1e-6f) continue;
            /* 反正切相位差 */
            float phase_diff = atan2f(diff, (val1 + val2) * 0.5f);
            sum_phase_diff += phase_diff;
            diff_count++;
        }
    }
    if (diff_count > 0) {
        float avg_phase = sum_phase_diff / (float)diff_count;
        float lambda = 3e8f / rp->start_freq;
        float sin_theta = avg_phase * lambda / (2.0f * (float)M_PI * d);
        if (sin_theta > 1.0f) sin_theta = 1.0f;
        if (sin_theta < -1.0f) sin_theta = -1.0f;
        *azimuth = asinf(sin_theta);
        *elevation = 0.0f; /* 1D阵列只能估计方位角 */
    } else {
        *azimuth = 0.0f;
        *elevation = 0.0f;
    }
    return 0;
}
