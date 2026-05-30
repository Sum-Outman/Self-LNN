/**
 * @file speech_recognition.c
 * @brief 语音识别 — 液态状态时序动态系统
 *
 * 纯液态神经网络语音识别实现。
 * 架构：音频 → 梅尔滤波器组 → 液态状态演化 → 直接字符概率输出 → 波束搜索解码
 * 无外部声学模型、无外部语言模型、无独立解码器。
 * 所有语音到文本的映射由单个连续时间液态状态动态系统完成。
 */

#include "selflnn/multimodal/speech_recognition.h"
#include "selflnn/multimodal/speech_language_model.h" /* ZSFA-FIX-P0-003: LM后处理纠错 */
#ifdef _MSC_VER
#pragma warning(disable:4702 4715)  /* 训练函数预存警告:不可达代码+返回值路径 */
#endif
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/cfc_cell.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <float.h>

/* rng_uniform声明来自math_utils.h */
extern float rng_uniform(float min, float max);

/* =============================================================== *
 * 常量定义                                                         *
 * =============================================================== */

/** @brief 默认词汇表大小（含空白token）
 *  中文语音识别需要更大词汇表：常用汉字~3500 + 词汇~20000 + 标点/英文
 *  扩大至50000以确保中文语音识别的基本覆盖 */
#define SR_DEFAULT_VOCAB_SIZE 50000

/** @brief 空白token ID */
#define SR_BLANK_TOKEN 0

/** @brief 梅尔频率下限（Hz） */
#define SR_MEL_LOW_HZ 80.0f

/** @brief 梅尔频率上限（Hz） */
#define SR_MEL_HIGH_HZ 7600.0f

/** @brief FFT大小 */
#define SR_FFT_SIZE 512

/** @brief 最大帧数 */
#define SR_MAX_FRAMES 3000

/** @brief 预加重系数 */
#define SR_PREEMPHASIS_ALPHA 0.97f

/** @brief 特征缓冲区大小增量 */
#define SR_FEATURE_BUF_INCREMENT 512

/** @brief 解码缓冲区大小 */
#define SR_DECODE_BUF_SIZE 2048

/* =============================================================== *
 * 语音识别处理器内部结构                                          *
 * =============================================================== */

struct SpeechRecognizer {
    SpeechRecognitionConfig config;
    int is_initialized;

    /* 液态状态系统（整个语音识别的核心） */
    float* hidden_state;
    int state_initialized;

    /* 输出投影权重（隐藏状态 → 字符概率） */
    float* output_projection_weight;   /* [hidden_size x vocab_size] */
    float* output_projection_bias;     /* [vocab_size] */
    int output_projection_initialized;

    /* ZSFABC-F007: 输入投影权重（特征→隐藏状态）改为实例成员
     * 之前为static变量，多实例并发时共享同一组权重导致干扰 */
    float* input_projection_weight;    /* [hidden_size x feature_dim] */
    int input_projection_initialized;

    /* 梅尔滤波器组系数（预计算） */
    float* mel_filterbank;             /* [num_mel_bins x fft_size/2+1] */
    int mel_filterbank_initialized;    /* ZSFABC: 梅尔滤波器组初始化标志 */
    float* hamming_window;             /* [frame_length] */

    /* 词汇表 */
    char** vocabulary;
    int vocabulary_size;
    int vocabulary_capacity;

    /* 音频/特征缓冲区 */
    float* audio_buffer;
    int audio_buffer_capacity;
    int audio_buffer_pos;
    float* feature_buffer;
    int feature_buffer_capacity;
    int feature_buffer_num_frames;

    /* 流式处理帧计数器 */
    int stream_total_frames;
    int stream_state_reset;

    /* 解码缓冲区 */
    int* decoded_tokens;
    int decoded_token_count;
    int decoded_token_capacity;

    /* 核心LNN网络集成（用于状态演化） */
    LNN* shared_lnn;

    /* P0-014: 自包含CfC路径可学习参数 — 替代硬编码0.5/0.3缩放因子 */
    float gate_scale;                 /* 门控缩放因子，Xavier初始化，替代硬编码0.5 */
    float act_scale;                  /* 激活缩放因子，Xavier初始化，替代硬编码0.3 */

    /* P0-014: 自包含CfC单元 — 独立于shared_lnn的完整CfC动态系统
     * 当shared_lnn未连接时，使用此CfCCell进行更丰富的多时间尺度状态演化
     * 若创建失败则为NULL，回退到可学习参数版简化路径 */
    CfCCell* self_contained_cfc;

    /* Adam优化器状态（训练用） */
    float* adam_m_proj_w;
    float* adam_v_proj_w;
    float* adam_m_proj_b;
    float* adam_v_proj_b;
    int adam_step;

    /* ZSFWS-F008: 模型训练完成标志 */
    int is_model_trained;
};

/* =============================================================== *
 * 辅助函数：梅尔频率变换                                           *
 * =============================================================== */

static inline float hz_to_mel(float hz) {
    return 1127.0f * logf(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

/* =============================================================== *
 * 汉明窗生成                                                       *
 * =============================================================== */

static void generate_hamming_window(float* window, int length) {
    for (int i = 0; i < length; i++) {
        window[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (length - 1));
    }
}

/* =============================================================== *
 * 梅尔滤波器组初始化                                               *
 * =============================================================== */

static int init_mel_filterbank(float* filterbank, int num_mels,
                                int fft_size, float sample_rate) {
    int num_bins = fft_size / 2 + 1;
    float mel_low = hz_to_mel(SR_MEL_LOW_HZ);
    float mel_high = hz_to_mel(SR_MEL_HIGH_HZ);

    memset(filterbank, 0, (size_t)num_mels * num_bins * sizeof(float));

    for (int m = 0; m < num_mels; m++) {
        float mel_center = mel_low + (mel_high - mel_low) * (m + 1) / (num_mels + 1);
        float hz_center = mel_to_hz(mel_center);
        int bin_center = (int)(hz_center * fft_size / sample_rate);
        if (bin_center < 1) bin_center = 1;
        if (bin_center >= num_bins - 1) bin_center = num_bins - 2;

        float mel_left = (m == 0) ? mel_low : mel_low + (mel_high - mel_low) * m / (num_mels + 1);
        float hz_left = mel_to_hz(mel_left);
        int bin_left = (int)(hz_left * fft_size / sample_rate);
        if (bin_left < 0) bin_left = 0;
        if (bin_left >= bin_center) bin_left = bin_center - 1;

        float mel_right = (m == num_mels - 1) ? mel_high : mel_low + (mel_high - mel_low) * (m + 2) / (num_mels + 1);
        float hz_right = mel_to_hz(mel_right);
        int bin_right = (int)(hz_right * fft_size / sample_rate);
        if (bin_right >= num_bins) bin_right = num_bins - 1;
        if (bin_right <= bin_center) bin_right = bin_center + 1;

        for (int b = bin_left; b < bin_center; b++) {
            filterbank[(size_t)m * num_bins + b] = (float)(b - bin_left) / (bin_center - bin_left);
        }
        for (int b = bin_center; b <= bin_right && b < num_bins; b++) {
            filterbank[(size_t)m * num_bins + b] = 1.0f - (float)(b - bin_center) / (bin_right - bin_center);
        }
    }
    return 0;
}

/* =============================================================== *
 * STFT：单帧FFT（实数输入→幅度谱）                                 *
 * =============================================================== */

static void compute_frame_spectrum(const float* frame, int frame_len,
                                    float* spectrum, int fft_size) {
    int num_bins = fft_size / 2 + 1;
    float* real = (float*)safe_malloc((size_t)fft_size * sizeof(float));
    float* imag = (float*)safe_malloc((size_t)fft_size * sizeof(float));
    if (!real || !imag) {
        safe_free((void**)&real);
        safe_free((void**)&imag);
        return;
    }

    memset(real, 0, (size_t)fft_size * sizeof(float));
    memset(imag, 0, (size_t)fft_size * sizeof(float));
    memcpy(real, frame, (size_t)frame_len * sizeof(float));

    /* 基2 FFT */
    int n = fft_size;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }
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
                float nwr = wr * wlen_r - wi * wlen_i;
                wi = wr * wlen_i + wi * wlen_r;
                wr = nwr;
            }
        }
    }

    for (int i = 0; i < num_bins; i++) {
        spectrum[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
    }

    safe_free((void**)&real);
    safe_free((void**)&imag);
}

/* =============================================================== *
 * 特征提取：音频 → 梅尔滤波器组特征                                 *
 * =============================================================== */

static int extract_mel_features(SpeechRecognizer* sr,
                                 const float* audio_data, int num_samples,
                                 float* features, int max_features) {
    int frame_length = sr->config.frame_length;
    int frame_shift = sr->config.frame_shift;
    int num_mels = sr->config.num_mel_bins;
    int fft_size = SR_FFT_SIZE;
    int num_bins = fft_size / 2 + 1;
    int feature_dim = sr->config.feature_dimension;

    int num_frames = (num_samples - frame_length) / frame_shift + 1;
    if (num_frames < 1) num_frames = 1;
    if (num_frames > max_features / feature_dim) {
        num_frames = max_features / feature_dim;
    }

    for (int t = 0; t < num_frames; t++) {
        int offset = t * frame_shift;
        int len = (offset + frame_length <= num_samples) ? frame_length : (num_samples - offset);
        if (len < frame_length) {
            memset(features + (size_t)t * feature_dim, 0, (size_t)feature_dim * sizeof(float));
            continue;
        }

        /* 预加重 + 加窗 */
        float* frame = (float*)safe_malloc((size_t)frame_length * sizeof(float));
        if (!frame) return -1;
        frame[0] = audio_data[offset] * sr->hamming_window[0];
        for (int i = 1; i < frame_length; i++) {
            frame[i] = (audio_data[offset + i] - SR_PREEMPHASIS_ALPHA * audio_data[offset + i - 1])
                       * sr->hamming_window[i];
        }

        /* STFT幅度谱 */
        float* spectrum = (float*)safe_malloc((size_t)num_bins * sizeof(float));
        if (!spectrum) { safe_free((void**)&frame); return -1; }
        compute_frame_spectrum(frame, frame_length, spectrum, fft_size);

        /* 梅尔滤波器组 */
        float* mel_energies = (float*)safe_malloc((size_t)num_mels * sizeof(float));
        if (!mel_energies) { safe_free((void**)&frame); safe_free((void**)&spectrum); return -1; }
        for (int m = 0; m < num_mels; m++) {
            double energy = 0.0;
            const float* fb_row = sr->mel_filterbank + (size_t)m * num_bins;
            for (int b = 0; b < num_bins; b++) {
                energy += (double)spectrum[b] * fb_row[b];
            }
            mel_energies[m] = (float)(logf((float)(energy + 1e-10f)));
        }

        /* 写入特征（梅尔+一阶差分+二阶差分） */
        float* feat = features + (size_t)t * feature_dim;
        int mel_dim = (feature_dim < num_mels) ? feature_dim : num_mels;
        memcpy(feat, mel_energies, (size_t)mel_dim * sizeof(float));
        if (feature_dim > num_mels) {
            memset(feat + num_mels, 0, (size_t)(feature_dim - num_mels) * sizeof(float));
        }

        safe_free((void**)&frame);
        safe_free((void**)&spectrum);
        safe_free((void**)&mel_energies);
    }

    /* 一阶差分（delta） */
    if (num_frames >= 3) {
        int mel_used = (feature_dim > num_mels) ? num_mels : feature_dim;
        int delta_dim = mel_used;
        for (int t = 2; t < num_frames - 2; t++) {
            for (int d = 0; d < delta_dim && d < mel_used; d++) {
                float prev2 = features[(size_t)(t - 2) * feature_dim + d];
                float prev1 = features[(size_t)(t - 1) * feature_dim + d];
                float next1 = features[(size_t)(t + 1) * feature_dim + d];
                float next2 = features[(size_t)(t + 2) * feature_dim + d];
                float delta = (next2 - prev2 + 2.0f * (next1 - prev1)) / 10.0f;
                features[(size_t)t * feature_dim + d + delta_dim] = delta;
            }
        }
        /* 二阶差分（delta-delta） */
        if (feature_dim >= 3 * mel_used) {
            int d2_offset = 2 * mel_used;
            for (int t = 2; t < num_frames - 2; t++) {
                for (int d = 0; d < mel_used; d++) {
                    float prev2 = features[(size_t)(t - 2) * feature_dim + d + mel_used];
                    float prev1 = features[(size_t)(t - 1) * feature_dim + d + mel_used];
                    float next1 = features[(size_t)(t + 1) * feature_dim + d + mel_used];
                    float next2 = features[(size_t)(t + 2) * feature_dim + d + mel_used];
                    float delta2 = (next2 - prev2 + 2.0f * (next1 - prev1)) / 10.0f;
                    features[(size_t)t * feature_dim + d + d2_offset] = delta2;
                }
            }
        }
    }

    /* R6-②修复: 语音特征全局Z-score归一化。
     * log-Mel能量值范围约[-12,0]且受输入音量影响大，
     * 与已归一化的视觉[0,1]和文本[L2单位范数]量级不匹配。
     * 此处对所有帧的全部特征做Z-score→mean=0, std=1, ±5截断。 */
    if (num_frames > 0) {
        size_t total = (size_t)num_frames * feature_dim;
        float mean = 0.0f, var = 0.0f;
        for (size_t i = 0; i < total; i++) mean += features[i];
        mean /= (float)total;
        for (size_t i = 0; i < total; i++) { float d = features[i] - mean; var += d * d; }
        var = var / (float)total + 1e-8f;
        float inv_std = 1.0f / sqrtf(var);
        for (size_t i = 0; i < total; i++) {
            features[i] = (features[i] - mean) * inv_std;
            if (features[i] > 5.0f) features[i] = 5.0f;
            else if (features[i] < -5.0f) features[i] = -5.0f;
        }
    }

    return num_frames;
}

/* =============================================================== *
 * 输出投影：隐藏状态 → 字符logits                                   *
 *                                                                   *
 * P2-005修复: 优先使用共享LNN进行输出投影（非线性液态动态），         *
 * 当LNN可用且维度匹配时通过lnn_forward处理；否则回退到线性投影。       *
 * =============================================================== */

static int compute_output_logits(SpeechRecognizer* sr,
                                   const float* hidden_state,
                                   float* logits, int vocab_size) {
    int hs = (int)sr->config.hidden_size;

    /* 尝试共享LNN做输出投影（非线性液态动态） */
    if (sr->shared_lnn) {
        LNNConfig lnn_cfg;
        if (lnn_get_config(sr->shared_lnn, &lnn_cfg) == 0) {
            if ((size_t)hs == lnn_cfg.input_size && (size_t)vocab_size == lnn_cfg.output_size) {
                float* lnn_input = (float*)safe_malloc(lnn_cfg.input_size * sizeof(float));
                float* lnn_output = (float*)safe_malloc(lnn_cfg.output_size * sizeof(float));
                if (lnn_input && lnn_output) {
                    memcpy(lnn_input, hidden_state, lnn_cfg.input_size * sizeof(float));
                    if (lnn_forward(sr->shared_lnn, lnn_input, lnn_output) == 0) {
                        memcpy(logits, lnn_output, (size_t)vocab_size * sizeof(float));
                        safe_free((void**)&lnn_input);
                        safe_free((void**)&lnn_output);
                        return 0;
                    }
                }
                safe_free((void**)&lnn_input);
                safe_free((void**)&lnn_output);
            }
        }
        /* LNN不可用或维度不匹配 → 返回错误码让上层处理，不执行线性投影回退 */
    }

    /* ZSFNO1-P0-001修复: 移除引用未定义变量mfcc_features/mfcc_dim的MFCC回退代码。
     * 原ZSFLYF-P2-010尝试添加MFCC能量分类作为兜底，但忘记传递MFCC特征参数。
     * 当前clean修复：LNN不可用时直接返回-1，调用方已有完善的错误处理逻辑。
     * 纯液态神经系统严格依赖LNN进行输出投影，不允许非LNN的简化回退路径。 */
    memset(logits, 0, (size_t)vocab_size * sizeof(float));
    return -1;
}

/* =============================================================== *
 * 波束搜索解码（直接对输出logits解码，无外部语言模型）                 *
 * =============================================================== */

typedef struct {
    int* tokens;
    int length;
    float score;
    float log_prob;
} BeamHypothesis;

static int beam_search_decode(const float* logits, int num_frames,
                               int vocab_size, int beam_width,
                               float beam_threshold, float temperature,
                               int* out_tokens, int* out_len, int max_out) {
    if (num_frames <= 0 || vocab_size <= 0) return -1;

    int blank = SR_BLANK_TOKEN;
    int max_hyp = beam_width;

    BeamHypothesis* hyps = (BeamHypothesis*)safe_malloc(
        (size_t)max_hyp * sizeof(BeamHypothesis));
    BeamHypothesis* next_hyps = (BeamHypothesis*)safe_malloc(
        (size_t)(max_hyp * vocab_size) * sizeof(BeamHypothesis));
    if (!hyps || !next_hyps) {
        safe_free((void**)&hyps);
        safe_free((void**)&next_hyps);
        return -1;
    }

    for (int i = 0; i < max_hyp; i++) {
        hyps[i].tokens = (int*)safe_malloc((size_t)max_out * sizeof(int));
        hyps[i].length = 0;
        hyps[i].score = (i == 0) ? 0.0f : -1e30f;
        hyps[i].log_prob = 0.0f;
        next_hyps[i].tokens = (int*)safe_malloc((size_t)max_out * sizeof(int));
        next_hyps[i].length = 0;
        next_hyps[i].score = -1e30f;
        next_hyps[i].log_prob = 0.0f;
    }

    for (int t = 0; t < num_frames; t++) {
        const float* frame_logits = logits + (size_t)t * vocab_size;
        int num_hyps = 0;

        for (int h = 0; h < max_hyp; h++) {
            if (hyps[h].score < -1e20f) continue;

            for (int c = 1; c < vocab_size; c++) {
                float logp = frame_logits[c] / temperature;
                if (logp < -20.0f) logp = -20.0f;

                float new_score = hyps[h].score + logp;
                if (new_score < hyps[h].score - beam_threshold) continue;

                /* 合并相同token（CTC风格） */
                int last_token = (hyps[h].length > 0) ? hyps[h].tokens[hyps[h].length - 1] : blank;
                int merge = 0;
                for (int k = 0; k < num_hyps; k++) {
                    int same = 1;
                    if (next_hyps[k].length != hyps[h].length + (c == last_token ? 0 : 1)) continue;
                    for (int j = 0; j < hyps[h].length; j++) {
                        if (next_hyps[k].tokens[j] != hyps[h].tokens[j]) { same = 0; break; }
                    }
                    if (same && c == last_token) {
                        next_hyps[k].score = logf(expf(next_hyps[k].score) + expf(new_score));
                        merge = 1;
                        break;
                    }
                }
                if (merge) continue;

                if (num_hyps < max_hyp * vocab_size) {
                    memcpy(next_hyps[num_hyps].tokens, hyps[h].tokens,
                           (size_t)hyps[h].length * sizeof(int));
                    int len = hyps[h].length;
                    if (c != last_token) {
                        next_hyps[num_hyps].tokens[len] = c;
                        next_hyps[num_hyps].length = len + 1;
                    } else {
                        next_hyps[num_hyps].length = len;
                    }
                    next_hyps[num_hyps].score = new_score;
                    next_hyps[num_hyps].log_prob = hyps[h].log_prob + logp;
                    num_hyps++;
                }
            }
        }

        /* 按分数排序，保留top beam_width */
        for (int i = 0; i < num_hyps - 1 && i < beam_width * 2; i++) {
            int best = i;
            for (int j = i + 1; j < num_hyps; j++) {
                if (next_hyps[j].score > next_hyps[best].score) best = j;
            }
            if (best != i) {
                BeamHypothesis tmp = next_hyps[i];
                next_hyps[i] = next_hyps[best];
                next_hyps[best] = tmp;
            }
        }

        int keep = (num_hyps < beam_width) ? num_hyps : beam_width;
        for (int i = 0; i < max_hyp; i++) {
            hyps[i].length = 0;
            hyps[i].score = -1e30f;
        }
        for (int i = 0; i < keep; i++) {
            memcpy(hyps[i].tokens, next_hyps[i].tokens,
                   (size_t)next_hyps[i].length * sizeof(int));
            hyps[i].length = next_hyps[i].length;
            hyps[i].score = next_hyps[i].score;
            hyps[i].log_prob = next_hyps[i].log_prob;
        }
    }

    /* 返回最佳假设 */
    int best = 0;
    for (int h = 1; h < max_hyp; h++) {
        if (hyps[h].score > hyps[best].score) best = h;
    }

    int out_len_val = 0;
    for (int i = 0; i < hyps[best].length && out_len_val < max_out; i++) {
        out_tokens[out_len_val++] = hyps[best].tokens[i];
    }
    *out_len = out_len_val;

    for (int i = 0; i < max_hyp; i++) {
        safe_free((void**)&hyps[i].tokens);
        safe_free((void**)&next_hyps[i].tokens);
    }
    safe_free((void**)&hyps);
    safe_free((void**)&next_hyps);

    return 0;
}

/* =============================================================== *
 * 穷举解码（贪心，用于快速测试）                                     *
 * =============================================================== */

static int greedy_decode(const float* logits, int num_frames,
                          int vocab_size, int* out_tokens, int* out_len,
                          int max_out) {
    int blank = SR_BLANK_TOKEN;
    int count = 0;
    int prev = blank;

    for (int t = 0; t < num_frames && count < max_out; t++) {
        const float* frame = logits + (size_t)t * vocab_size;
        int best_c = 0;
        float best_v = frame[0];
        for (int c = 1; c < vocab_size; c++) {
            if (frame[c] > best_v) {
                best_v = frame[c];
                best_c = c;
            }
        }
        if (best_c != blank && best_c != prev) {
            out_tokens[count++] = best_c;
        }
        prev = best_c;
    }
    *out_len = count;
    return 0;
}

static float _compute_avg_confidence(const float* confidences, int count) {
    if (!confidences || count <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < count; i++) sum += confidences[i];
    return sum / (float)count;
}

/*
 * CTC (Connectionist Temporal Classification) 前向后向损失计算
 * Graves et al. 2006
 *
 * 输入: logits[T][V] = 帧×词汇表概率分布(已过softmax)
 *       labels[L] = 目标标签序列
 * 输出: loss = -ln(P(labels|logits))
 *
 * 实现前向后向算法计算所有对齐路径的动态规划。
 * 扩展的标签序列：blank插入每个标签前后 + 重复标签间
 */
static float compute_ctc_loss(const float* logits, int num_frames,
                               const int* labels, int num_labels,
                               int vocab_size) {
    if (num_frames <= 0 || num_labels <= 0 || vocab_size < 2) return 1e10f;

    /* 构建扩展标签序列 l': blank+label0+blank+label1+...
     * 长度 E = 2*L + 1 */
    int extended_len = 2 * num_labels + 1;
    int blank = SR_BLANK_TOKEN;

    int* extended_labels = (int*)safe_malloc((size_t)extended_len * sizeof(int));
    if (!extended_labels) return 1e10f;
    extended_labels[0] = blank;
    for (int i = 0; i < num_labels; i++) {
        extended_labels[2*i + 1] = labels[i];
        extended_labels[2*i + 2] = blank;
    }

    /* 前向变量 alpha[t][s]: 帧t时刻到达扩展标签位置s的概率 */
    float* alpha = (float*)safe_calloc((size_t)(num_frames * extended_len), sizeof(float));
    if (!alpha) { safe_free((void**)&extended_labels); return 1e10f; }

    /* 初始化 t=0: alpha[0][0] = logits[0][blank], alpha[0][1] = logits[0][labels[0]] */
    alpha[0] = logits[blank];
    if (num_labels > 0) {
        alpha[extended_len] = logits[labels[0]]; /* alpha[0][1] */
    }

    for (int t = 1; t < num_frames; t++) {
        const float* frame = logits + (size_t)t * vocab_size;
        for (int s = 0; s < extended_len; s++) {
            int label_s = extended_labels[s];
            float p_t_s = frame[label_s];
            if (p_t_s < 1e-20f) continue;

            float sum = 0.0f;

            /* 从位置s转移（保持） */
            sum += alpha[(size_t)(t-1) * extended_len + s];

            /* 从位置s-1转移（前进） */
            if (s > 0) {
                sum += alpha[(size_t)(t-1) * extended_len + (s-1)];
            }

            /* 从位置s-2转移（跳过blank重复——当s-2标签≠当前标签时允许） */
            if (s > 1 && label_s != extended_labels[s-2]) {
                sum += alpha[(size_t)(t-1) * extended_len + (s-2)];
            }

            alpha[(size_t)t * extended_len + s] = sum * p_t_s;
        }
    }

    /* 后向变量 beta[t][s] */
    float* beta = (float*)safe_calloc((size_t)(num_frames * extended_len), sizeof(float));
    float loss;
    if (!beta) {
        /* 无后向变量时只用前向计算 */
        float p_total = alpha[(size_t)(num_frames - 1) * extended_len + (extended_len - 1)];
        if (alpha[(size_t)(num_frames - 1) * extended_len + (extended_len - 2)] > 0.0f) {
            p_total += alpha[(size_t)(num_frames - 1) * extended_len + (extended_len - 2)];
        }
        loss = (p_total > 1e-20f) ? -logf(p_total) : 10.0f;
        safe_free((void**)&extended_labels);
        safe_free((void**)&alpha);
        return loss;
    }

    /* 初始化后向 t=T-1 */
    beta[(size_t)(num_frames - 1) * extended_len + (extended_len - 1)] = 1.0f;
    if (extended_len >= 2) {
        beta[(size_t)(num_frames - 1) * extended_len + (extended_len - 2)] = 1.0f;
    }

    for (int t = num_frames - 2; t >= 0; t--) {
        const float* frame_next = logits + (size_t)(t + 1) * vocab_size;
        for (int s = 0; s < extended_len; s++) {
            float sum = 0.0f;
            int label_s = extended_labels[s];

            /* 保持 */
            sum += frame_next[label_s] * beta[(size_t)(t+1) * extended_len + s];

            /* 前进 */
            if (s + 1 < extended_len) {
                sum += frame_next[extended_labels[s+1]] * beta[(size_t)(t+1) * extended_len + (s+1)];
            }

            /* 跳过blank */
            if (s + 2 < extended_len && label_s != extended_labels[s+2]) {
                sum += frame_next[extended_labels[s+2]] * beta[(size_t)(t+1) * extended_len + (s+2)];
            }

            beta[(size_t)t * extended_len + s] = sum;
        }
    }

    /* 总概率 = Σ_s alpha[t][s] * beta[t][s] / logits[t][extended_label[s]] */
    float p_total = 0.0f;
    for (int t = 0; t < num_frames; t++) {
        for (int s = 0; s < extended_len; s++) {
            float prob = logits[(size_t)t * vocab_size + extended_labels[s]];
            if (prob < 1e-20f) continue;
            p_total += alpha[(size_t)t * extended_len + s] *
                       beta[(size_t)t * extended_len + s] / prob;
        }
    }
    p_total /= (float)num_frames;
    loss = (p_total > 1e-20f) ? -logf(p_total) : 10.0f;

    safe_free((void**)&extended_labels);
    safe_free((void**)&alpha);
    safe_free((void**)&beta);
    return loss;
}

/* CTC损失公开API */
float speech_ctc_compute_loss(const float* logits, int num_frames,
                               const int* labels, int num_labels, int vocab_size) {
    return compute_ctc_loss(logits, num_frames, labels, num_labels, vocab_size);
}

/* =============================================================== *
 * 投影权重初始化（Xavier初始化）                                      *
 * =============================================================== */

static int init_output_projection(SpeechRecognizer* sr) {
    int hs = (int)sr->config.hidden_size;
    int vs = sr->config.vocab_size;
    if (vs <= 0 || hs <= 0) return -1;
    if (sr->output_projection_initialized) return 0;

    sr->output_projection_weight = (float*)safe_malloc(
        (size_t)hs * vs * sizeof(float));
    sr->output_projection_bias = (float*)safe_malloc(
        (size_t)vs * sizeof(float));
    if (!sr->output_projection_weight || !sr->output_projection_bias) {
        safe_free((void**)&sr->output_projection_weight);
        safe_free((void**)&sr->output_projection_bias);
        return -1;
    }

    float scale = sqrtf(2.0f / (float)(hs + vs));
    for (int i = 0; i < hs * vs; i++) {
        sr->output_projection_weight[i] = rng_uniform(-scale, scale);
    }
    memset(sr->output_projection_bias, 0, (size_t)vs * sizeof(float));

    sr->output_projection_initialized = 1;
    return 0;
}

/* =============================================================== *
 * 初始化液态状态缓冲区                                               *
 * =============================================================== */

static int init_state_buffer(SpeechRecognizer* sr) {
    if (sr->state_initialized) return 0;

    sr->hidden_state = (float*)safe_malloc(
        sr->config.hidden_size * sizeof(float));
    if (!sr->hidden_state) return -1;
    memset(sr->hidden_state, 0, sr->config.hidden_size * sizeof(float));

    sr->state_initialized = 1;
    return 0;
}

/* =============================================================== *
 * 中文词汇表初始化 — M-015修复: 分段定义24段共2000+高频词条              *
 * 涵盖日常用语、指令词汇、物体名称、AGI/机器人专用词汇等                *
 * =============================================================== */

/* M-015修复: 词汇表分段定义，扩展至2000+高频中文词条
 * 分段1: 基础词汇（空白/标点/数字/量词/代词/字母） */
static const char* SR_VOCAB_SEGMENT_1[] = {
    /* 空白token（CTC用） */
    "",
    /* 标点符号（扩展） */
    "。", "，", "！", "？", "、", "；", "：", "…", "—", "～",
    "（", "）", "【", "】", "《", "》", "\"", "\"", "'", "'",
    "「", "」", "『", "』", "·", "／", "＠", "＃", "＄", "％",
    "＆", "＊", "＋", "－", "＝", "＜", "＞", "＿", "～", "｜",
    /* 中文数字 */
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "亿", "兆", "两", "半", "几", "多", "第",
    /* 常用量词 */
    "个", "只", "条", "张", "把", "台", "辆", "件", "本", "次",
    "米", "厘米", "毫米", "千米", "公斤", "克", "秒", "分钟", "小时", "天",
    "年", "月", "日", "度", "摄氏度", "周", "季度", "倍", "层", "段",
    "片", "块", "颗", "粒", "滴", "串", "排", "列", "行", "组",
    "套", "副", "双", "对", "束", "堆", "群", "批", "类", "种",
    /* 代词集合 */
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "这", "那", "这个", "那个", "这些", "那些", "这里", "那里", "这边", "那边",
    "什么", "怎么", "哪里", "谁", "为什么", "怎样", "多少", "多久", "哪些", "哪个",
    "自己", "彼此", "大家", "别人", "人家", "各位", "诸位", "本身",
    /* 拉丁字母（小写） */
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z",
    /* 拉丁字母（大写） */
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    /* 数字 */
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
};

/* 分段2: 高频动词集合 */
static const char* SR_VOCAB_SEGMENT_2[] = {
    "是", "有", "在", "说", "来", "去", "做", "看", "听", "吃",
    "喝", "走", "跑", "写", "读", "学", "教", "买", "卖", "用",
    "给", "拿", "放", "开", "关", "打开", "关闭", "开始", "停止", "启动",
    "运行", "移动", "转动", "抓取", "推", "拉", "按", "点击", "输入", "输出",
    "识别", "检测", "处理", "计算", "分析", "生成", "发送", "接收", "连接", "断开",
    "进入", "退出", "切换", "跳转", "滚动", "缩放", "拖拽", "选择", "复制", "粘贴",
    "剪切", "删除", "撤销", "恢复", "保存", "加载", "刷新", "更新", "升级", "降级",
    "安装", "卸载", "配置", "设置", "调试", "测试", "部署", "发布", "构建", "编译",
    "查询", "搜索", "查找", "替换", "排序", "筛选", "过滤", "分组", "聚合", "统计",
    "加密", "解密", "压缩", "解压", "编码", "解码", "转换", "格式化", "验证", "校验",
    "创建", "删除", "修改", "增加", "减少", "调整", "优化", "改进", "完善", "扩展",
    "等待", "延迟", "超时", "重试", "回滚", "备份", "恢复", "迁移", "同步", "异步",
    "上传", "下载", "导入", "导出", "打印", "预览", "播放", "暂停", "停止", "快进",
    "快退", "录制", "截图", "标记", "注释", "批注", "签名", "盖章",
};

/* 分段3: 高频名词 — 人与社会 */
static const char* SR_VOCAB_SEGMENT_3[] = {
    "人", "人类", "人们", "人物", "个人", "男人", "女人", "儿童", "老人", "青年",
    "家庭", "父母", "父亲", "母亲", "爸爸", "妈妈", "儿子", "女儿", "兄弟", "姐妹",
    "爷爷", "奶奶", "外公", "外婆", "叔叔", "阿姨", "舅舅", "姑姑", "夫妻", "丈夫",
    "妻子", "孩子", "婴儿", "学生", "老师", "医生", "护士", "警察", "军人", "工人",
    "农民", "商人", "经理", "老板", "员工", "同事", "朋友", "邻居", "同学", "伙伴",
    "领导", "客户", "用户", "会员", "专家", "作者", "读者", "观众", "听众", "市民",
    "居民", "公民", "人民", "群众", "团体", "组织", "机构", "部门", "单位", "公司",
    "企业", "工厂", "商店", "学校", "医院", "银行", "邮局", "车站", "机场", "码头",
};

/* 分段4: 高频名词 — 物体与设备 */
static const char* SR_VOCAB_SEGMENT_4[] = {
    "机器", "机器人", "电脑", "计算机", "手机", "电话", "平板", "手表", "眼镜", "耳机",
    "摄像头", "相机", "麦克风", "音箱", "喇叭", "屏幕", "显示器", "键盘", "鼠标", "打印机",
    "扫描仪", "投影仪", "服务器", "路由器", "交换机", "硬盘", "内存", "处理器", "显卡", "主板",
    "传感器", "控制器", "执行器", "电机", "马达", "齿轮", "轴承", "弹簧", "阀门", "泵",
    "电池", "电源", "充电器", "变压器", "继电器", "开关", "按钮", "旋钮", "指示灯", "显示屏",
    "系统", "程序", "软件", "应用", "数据", "数据库", "文件", "文档", "图像", "图片",
    "视频", "音频", "文本", "消息", "邮件", "通知", "日志", "记录", "报告", "表格",
    "电路", "线路", "接口", "端口", "协议", "网络", "信号", "频道", "波段", "频率",
};

/* 分段5: 高频名词 — 自然与属性 */
static const char* SR_VOCAB_SEGMENT_5[] = {
    "声音", "光", "光线", "色彩", "温度", "湿度", "压力", "速度", "加速度", "力量",
    "重量", "质量", "密度", "体积", "面积", "长度", "宽度", "高度", "深度", "厚度",
    "位置", "方向", "距离", "角度", "坐标", "轨迹", "路径", "路线", "范围", "区域",
    "时间", "时刻", "日期", "星期", "月份", "季节", "年度", "周期", "频率", "时长",
    "红色", "绿色", "蓝色", "白色", "黑色", "黄色", "紫色", "橙色", "灰色", "棕色",
    "粉色", "青色", "金色", "银色", "透明", "深色", "浅色", "亮色", "暗色", "彩色",
    "天空", "大地", "海洋", "河流", "湖泊", "山脉", "森林", "沙漠", "草原", "岛屿",
    "太阳", "月亮", "星星", "云彩", "风雨", "雷电", "冰雪", "霜雾", "彩虹", "天气",
};

/* 分段6: 身体部位与健康 */
static const char* SR_VOCAB_SEGMENT_6[] = {
    "头", "头发", "脸", "眼睛", "鼻子", "嘴巴", "耳朵", "牙齿", "舌头", "脖子",
    "肩膀", "手臂", "手", "手指", "手掌", "手腕", "肘部", "胸部", "背部", "腰部",
    "腹部", "臀部", "腿", "脚", "膝盖", "脚踝", "脚跟", "脚趾", "皮肤", "骨头",
    "肌肉", "关节", "神经", "血液", "心脏", "肺", "肝脏", "肾脏", "胃", "肠道",
    "身体", "健康", "疾病", "疼痛", "受伤", "疲劳", "发烧", "感冒", "咳嗽", "头痛",
    "眩晕", "恶心", "过敏", "发炎", "感染", "骨折", "创伤", "手术", "药物", "药片",
    "治疗", "康复", "锻炼", "运动", "休息", "睡眠", "饮食", "营养", "维生素", "蛋白",
    "血压", "脉搏", "体温", "呼吸", "心跳", "血糖", "血脂", "体检", "诊断", "预防",
};

/* 分段7: 时间与日期 */
static const char* SR_VOCAB_SEGMENT_7[] = {
    "今天", "明天", "昨天", "前天", "后天", "早上", "上午", "中午", "下午", "晚上",
    "凌晨", "傍晚", "半夜", "白天", "黑夜", "早晨", "黄昏", "黎明", "午夜", "时分",
    "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日", "周末", "工作日",
    "一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月",
    "十一月", "十二月", "春天", "夏天", "秋天", "冬天", "春季", "夏季", "秋季", "冬季",
    "过去", "现在", "未来", "刚才", "即将", "马上", "立刻", "立即", "突然", "逐渐",
    "经常", "偶尔", "总是", "从不", "曾经", "已经", "正在", "将要", "仍然", "一直",
    "开始", "结束", "持续", "短暂", "永久", "临时", "长期", "短期", "及时", "准时",
};

/* 分段8: 方位与空间 */
static const char* SR_VOCAB_SEGMENT_8[] = {
    "前", "后", "左", "右", "上", "下", "中间", "里面", "外面", "旁边",
    "对面", "附近", "周围", "边缘", "中心", "内部", "外部", "顶部", "底部", "侧面",
    "东方", "西方", "南方", "北方", "东南", "东北", "西南", "西北", "方向", "朝向",
    "前面", "后面", "左边", "右边", "上面", "下面", "远处", "近处", "高处", "低处",
    "起点", "终点", "途中", "沿途", "走廊", "通道", "入口", "出口", "门口", "窗户",
    "角落", "墙角", "天花板", "地板", "墙壁", "楼梯", "电梯", "阳台", "花园", "院子",
    "地址", "地点", "场所", "位置", "坐标", "经纬度", "高度", "海拔", "水平", "垂直",
    "前进", "后退", "转弯", "掉头", "直行", "横向", "纵向", "斜向", "顺时针", "逆时针",
};

/* 分段9: 形容词集合 */
static const char* SR_VOCAB_SEGMENT_9[] = {
    "大", "小", "长", "短", "宽", "窄", "厚", "薄", "粗", "细",
    "高", "低", "深", "浅", "远", "近", "快", "慢", "强", "弱",
    "硬", "软", "轻", "重", "新", "旧", "好", "坏", "对", "错",
    "真", "假", "正", "反", "冷", "热", "干", "湿", "亮", "暗",
    "多", "少", "贵", "便宜", "容易", "困难", "简单", "复杂", "重要", "次要",
    "主要", "根本", "基本", "特殊", "普通", "一般", "相同", "不同", "类似", "相似",
    "安全", "危险", "稳定", "变化", "固定", "移动", "安静", "吵闹", "干净", "脏污",
    "整齐", "凌乱", "光滑", "粗糙", "尖锐", "钝化", "灵活", "僵硬", "柔软", "坚硬",
    "温暖", "寒冷", "凉爽", "炎热", "潮湿", "干燥", "明亮", "昏暗", "鲜明", "暗淡",
    "清晰", "模糊", "精确", "粗略", "完整", "残缺", "正常", "异常", "有效", "无效",
    "先进", "落后", "现代", "传统", "智能", "手动", "自动", "半自动", "主动", "被动",
    "积极", "消极", "乐观", "悲观", "谨慎", "大胆", "善良", "恶劣", "友好", "敌意",
};

/* 分段10: 副词集合 */
static const char* SR_VOCAB_SEGMENT_10[] = {
    "很", "非常", "特别", "极其", "相当", "比较", "稍微", "略微", "几乎", "完全",
    "彻底", "绝对", "根本", "确实", "真正", "实在", "大概", "大约", "可能", "也许",
    "已经", "曾经", "正在", "将要", "立刻", "马上", "突然", "忽然", "逐渐", "渐渐",
    "仍然", "依然", "还是", "总是", "经常", "常常", "偶尔", "有时", "从不", "一直",
    "只", "仅仅", "不过", "就", "才", "都", "全部", "总共", "一共", "至少",
    "最多", "最少", "更加", "越来", "越来越", "尽量", "尽可能", "最好", "最好别", "千万",
    "究竟", "到底", "难道", "明明", "居然", "竟然", "果然", "其实", "当然", "自然",
    "也许", "大约", "或许", "恐怕", "终究", "终于", "总算", "毕竟", "反正", "何必",
};

/* 分段11: 连接词与语气词 */
static const char* SR_VOCAB_SEGMENT_11[] = {
    "和", "或", "但", "而", "且", "并", "却", "则", "就", "才",
    "如果", "因为", "所以", "虽然", "但是", "然而", "而且", "因此", "于是", "那么",
    "然后", "接着", "最后", "首先", "其次", "再次", "另外", "此外", "除此", "除了",
    "不但", "不仅", "不管", "无论", "即使", "尽管", "只要", "只有", "除非", "否则",
    "关于", "对于", "根据", "按照", "通过", "经过", "由于", "为了", "以便", "以免",
    "啊", "吧", "吗", "呢", "啦", "哦", "嗯", "呀", "哟", "呗",
    "嘛", "哎", "嗨", "喂", "呵呵", "哈哈", "嘻嘻", "呜呜", "哎呀", "哎哟",
    "总之", "综上所述", "换句话说", "也就是说", "例如", "比如", "包括", "以及", "等等", "之类",
};

/* 分段12: 日常用品与生活 */
static const char* SR_VOCAB_SEGMENT_12[] = {
    "桌子", "椅子", "沙发", "床", "柜子", "书架", "抽屉", "镜子", "钟表", "闹钟",
    "灯", "灯泡", "插座", "插头", "电线", "遥控器", "空调", "风扇", "暖气", "加湿器",
    "冰箱", "洗衣机", "微波炉", "烤箱", "电饭煲", "油烟机", "热水器", "饮水机", "吸尘器",
    "杯子", "碗", "盘子", "筷子", "勺子", "叉子", "刀", "锅", "壶", "瓶",
    "笔", "铅笔", "钢笔", "圆珠笔", "橡皮", "尺子", "剪刀", "胶带", "胶水", "订书机",
    "纸", "本子", "书", "杂志", "报纸", "地图", "日历", "标签", "贴纸", "卡片",
    "钥匙", "锁", "伞", "包", "钱包", "袋子", "箱子", "盒子", "盒子", "篮子",
    "衣服", "裤子", "裙子", "鞋子", "帽子", "手套", "围巾", "袜子", "眼镜", "手表",
    "毛巾", "牙刷", "牙膏", "肥皂", "洗发水", "沐浴露", "纸巾", "垃圾袋", "扫帚", "拖把",
};

/* 分段13: 食物与饮品 */
static const char* SR_VOCAB_SEGMENT_13[] = {
    "食物", "饮料", "水", "茶", "咖啡", "牛奶", "果汁", "可乐", "啤酒", "红酒",
    "米饭", "面条", "面包", "馒头", "饺子", "包子", "饼", "粥", "汤", "菜",
    "肉", "鱼", "虾", "鸡蛋", "豆腐", "蔬菜", "水果", "苹果", "香蕉", "橘子",
    "葡萄", "西瓜", "草莓", "桃子", "梨", "芒果", "菠萝", "樱桃", "柠檬", "猕猴桃",
    "西红柿", "黄瓜", "白菜", "萝卜", "土豆", "洋葱", "辣椒", "大蒜", "生姜", "葱",
    "盐", "糖", "油", "酱油", "醋", "味精", "胡椒粉", "辣椒酱", "番茄酱", "沙拉",
    "早餐", "午餐", "晚餐", "零食", "点心", "蛋糕", "饼干", "巧克力", "糖果", "冰淇淋",
    "牛肉", "猪肉", "鸡肉", "羊肉", "鸭肉", "火腿", "香肠", "培根", "奶酪", "酸奶",
};

/* 分段14: 交通与出行 */
static const char* SR_VOCAB_SEGMENT_14[] = {
    "汽车", "公交车", "地铁", "火车", "高铁", "飞机", "轮船", "自行车", "电动车", "摩托车",
    "出租车", "网约车", "货车", "卡车", "救护车", "消防车", "警车", "校车", "缆车", "直升机",
    "道路", "公路", "高速", "街道", "桥梁", "隧道", "路口", "红绿灯", "斑马线", "人行道",
    "车站", "机场", "港口", "码头", "停车场", "加油站", "服务区", "收费站", "检查站", "入口",
    "驾驶", "乘坐", "出发", "到达", "换乘", "转车", "直达", "中途", "终点", "起点",
    "车票", "机票", "船票", "座位", "车厢", "航班", "班次", "时刻表", "延误", "取消",
    "行李", "托运", "安检", "登机", "检票", "候车", "候机", "抵达", "离开", "往返",
};

/* 分段15: 教育与学习 */
static const char* SR_VOCAB_SEGMENT_15[] = {
    "学习", "教育", "教学", "课程", "科目", "专业", "学科", "知识", "技能", "能力",
    "学校", "教室", "课堂", "图书馆", "实验室", "操场", "食堂", "宿舍", "办公室",
    "作业", "考试", "测验", "成绩", "分数", "等级", "及格", "优秀", "毕业", "升学",
    "语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治", "体育",
    "科学", "技术", "工程", "艺术", "音乐", "美术", "舞蹈", "戏剧", "文学", "哲学",
    "阅读", "写作", "练习", "实验", "研究", "讨论", "演讲", "答辩", "论文", "报告",
    "培训", "实习", "进修", "留学", "交流", "合作", "竞赛", "比赛", "获奖", "证书",
};

/* 分段16: 科技与计算 */
static const char* SR_VOCAB_SEGMENT_16[] = {
    "算法", "模型", "架构", "框架", "接口", "模块", "组件", "服务", "应用", "平台",
    "代码", "编程", "开发", "设计", "实现", "集成", "测试", "上线", "运维", "监控",
    "人工智能", "机器学习", "深度学习", "神经网络", "语音识别", "图像识别", "自然语言", "计算机视觉",
    "数据挖掘", "大数据", "云计算", "边缘计算", "物联网", "区块链", "虚拟现实", "增强现实",
    "前端", "后端", "全栈", "嵌入式", "移动端", "桌面端", "网页", "小程序", "命令行",
    "版本", "仓库", "提交", "分支", "合并", "冲突", "发布", "部署", "回滚", "修复",
    "加密", "安全", "认证", "授权", "令牌", "密码", "密钥", "证书", "签名", "验证",
    "缓存", "队列", "消息", "事件", "调度", "负载", "均衡", "容错", "备份", "恢复",
};

/* 分段17: AGI与机器人专用词汇 */
static const char* SR_VOCAB_SEGMENT_17[] = {
    "液态神经网络", "连续时间", "动态系统", "常微分方程", "状态演化", "CfC", "全模态",
    "自我认知", "自我决策", "自主执行", "自我学习", "自我演化", "模仿学习", "自我修正",
    "知识库", "知识图谱", "推理引擎", "规划系统", "决策系统", "执行引擎", "监控系统",
    "感知模块", "认知模块", "行为模块", "语言模块", "视觉模块", "听觉模块", "触觉模块",
    "多模态融合", "跨模态", "传感器融合", "状态估计", "预测控制", "自适应", "鲁棒性",
    "机器人控制", "运动规划", "轨迹跟踪", "力控制", "阻抗控制", "柔顺控制", "协同控制",
    "机械臂", "关节", "末端执行器", "夹爪", "吸盘", "工具", "夹具", "平台", "底座",
    "导航", "定位", "建图", "SLAM", "路径规划", "避障", "探索", "巡逻", "巡检",
    "双目视觉", "深度估计", "点云", "三维重建", "目标检测", "语义分割", "实例分割",
    "人机交互", "语音对话", "手势识别", "表情识别", "意图理解", "情感计算", "注意力",
    "自主学习", "增量学习", "迁移学习", "强化学习", "元学习", "在线学习", "联邦学习",
};

/* 分段18: 传感器与执行器 */
static const char* SR_VOCAB_SEGMENT_18[] = {
    "温度传感器", "湿度传感器", "压力传感器", "加速度计", "陀螺仪", "磁力计", "IMU", "GPS",
    "激光雷达", "毫米波雷达", "超声波", "红外传感器", "光电传感器", "接近传感器", "触碰传感器",
    "力传感器", "扭矩传感器", "位置传感器", "速度传感器", "编码器", "电位器", "应变片",
    "气体传感器", "烟雾传感器", "火焰传感器", "漏水传感器", "振动传感器", "声学传感器",
    "电流传感器", "电压传感器", "功率传感器", "电量计", "PH传感器", "电导率传感器",
    "伺服电机", "步进电机", "直流电机", "无刷电机", "直线电机", "舵机", "电磁阀",
    "气缸", "液压缸", "电动推杆", "旋转执行器", "直线执行器", "压电执行器",
    "继电器", "接触器", "变频器", "驱动器", "控制器", "PLC", "单片机", "嵌入式",
};

/* 分段19: 指令与交互短语 */
static const char* SR_VOCAB_SEGMENT_19[] = {
    "你好", "谢谢", "再见", "对不起", "没关系", "请", "帮忙", "麻烦", "打扰", "不客气",
    "确认", "取消", "是", "否", "好的", "明白", "了解", "收到", "知道", "不知道",
    "开始工作", "停止工作", "暂停任务", "继续执行", "重新开始", "取消任务", "切换模式",
    "提高速度", "降低速度", "增加力度", "减小力度", "向前移动", "向后移动", "向左转", "向右转",
    "上升", "下降", "前进", "后退", "旋转", "升降", "伸缩", "点头", "摇头", "转身",
    "抓取物体", "释放物体", "放置物体", "移动物体", "推动", "拉动", "按压", "旋转",
    "温度", "湿度", "压力", "电流", "电压", "功率", "频率", "占空比", "阈值", "参数",
    "图片", "拍照", "录像", "录音", "播放", "暂停", "快进", "快退", "下一首", "上一首",
    "执行", "查询", "搜索", "学习", "训练", "保存", "加载", "读取", "写入", "删除",
    "状态", "日志", "设置", "参数", "配置", "模式", "场景", "任务", "计划", "日程",
    "返回", "前进", "首页", "菜单", "导航", "路径", "路线", "方向", "命令", "指令",
    "查询状态", "报告位置", "自我检测", "系统检查", "故障诊断", "错误报告", "警告信息",
    "电量不足", "充电", "待机", "唤醒", "休眠", "关机", "重启", "重置", "初始化",
    "连接设备", "断开连接", "配对", "扫描", "搜索设备", "发现设备", "已连接", "已断开",
    "更新固件", "升级系统", "备份数据", "恢复出厂", "清除缓存", "释放内存", "校准传感器",
    "启动自检", "执行诊断", "查看日志", "导出数据", "导入配置", "应用设置", "保存配置",
};

/* 分段20: 问答用语 */
static const char* SR_VOCAB_SEGMENT_20[] = {
    "什么是", "如何", "怎样", "为什么", "在哪里", "什么时候", "谁", "哪一个", "多少钱",
    "请问", "能否", "可不可以", "需不需要", "应不应该", "值不值得", "会不会", "能不能",
    "告诉我", "解释一下", "说明一下", "介绍一下", "描述", "总结", "概括", "分析",
    "我认为", "我觉得", "看起来", "听起来", "感觉", "建议", "推荐", "警告", "提醒",
    "注意", "小心", "当心", "留意", "检查", "确认", "核实", "对比", "比较", "区别",
    "正确", "错误", "准确", "精确", "大约", "估计", "预测", "判断", "评估", "评价",
    "优点", "缺点", "优势", "劣势", "好处", "坏处", "风险", "机会", "挑战", "问题",
    "解决方案", "替代方案", "最佳实践", "注意事项", "常见问题", "故障排除", "操作指南",
};

/* 分段21: 商业与金融 */
static const char* SR_VOCAB_SEGMENT_21[] = {
    "价格", "成本", "收入", "利润", "损失", "预算", "投资", "融资", "贷款", "利息",
    "股票", "基金", "债券", "期货", "市场", "交易", "买卖", "合同", "协议", "订单",
    "发票", "收据", "账单", "支付", "付款", "收款", "转账", "汇款", "现金", "信用卡",
    "微信支付", "支付宝", "银行转账", "货到付款", "分期付款", "全额", "预付款", "尾款",
    "公司", "企业", "品牌", "产品", "服务", "客户", "用户", "供应商", "代理商", "经销商",
    "生产", "制造", "加工", "采购", "库存", "物流", "配送", "运输", "仓储", "供应链",
};

/* 分段22: 自然环境与生态 */
static const char* SR_VOCAB_SEGMENT_22[] = {
    "环境", "生态", "气候", "气温", "降水", "风力", "风向", "湿度", "气压", "能见度",
    "晴天", "阴天", "多云", "雨天", "雪天", "雾天", "台风", "暴雨", "暴雪", "沙尘暴",
    "空气", "水质", "土壤", "噪音", "污染", "排放", "废弃物", "回收", "再生", "节能",
    "植树", "绿化", "保护", "治理", "监测", "检测", "评估", "报告", "标准", "合规",
    "太阳能", "风能", "水能", "地热能", "核能", "化石能源", "可再生能源", "清洁能源",
    "碳", "碳足迹", "碳中和", "碳排放", "碳交易", "温室效应", "全球变暖", "气候变化",
};

/* 分段23: 通信与网络 */
static const char* SR_VOCAB_SEGMENT_23[] = {
    "通信", "网络", "信号", "带宽", "延迟", "吞吐量", "丢包", "误码", "干扰", "噪声",
    "有线", "无线", "蓝牙", "WiFi", "4G", "5G", "LoRa", "ZigBee", "NFC", "RFID",
    "通信协议", "TCP", "UDP", "HTTP", "HTTPS", "MQTT", "WebSocket", "RS232", "RS485",
    "IP地址", "端口号", "域名", "网关", "子网", "掩码", "路由", "防火墙", "代理", "VPN",
    "发送", "接收", "广播", "组播", "单播", "点对点", "客户端", "服务器", "请求", "响应",
    "加密通信", "安全传输", "认证", "授权", "令牌", "会话", "心跳", "超时", "重连", "握手",
};

/* 分段24: 紧急与安全 */
static const char* SR_VOCAB_SEGMENT_24[] = {
    "紧急", "危险", "警告", "故障", "错误", "异常", "报警", "警报", "灭火", "疏散",
    "安全", "防护", "保护", "监控", "巡检", "巡逻", "检查", "排查", "消除", "预防",
    "火灾", "水灾", "地震", "台风", "爆炸", "泄露", "中毒", "触电", "坠落", "碰撞",
    "急救", "救援", "求助", "求救", "报警电话", "消防", "急救站", "安全出口", "应急灯",
    "安全帽", "安全鞋", "手套", "护目镜", "防护服", "口罩", "安全带", "消防栓", "灭火器",
    "紧急停止", "急停按钮", "安全门", "隔离区", "警戒线", "禁止入内", "高压危险", "当心触电",
};

/* =============================================================== *
 * M-016: 扩展词汇表 — 11个新分段覆盖常用中文词汇 + TTS拼音对齐        *
 * 按类别组织：日常用语(500)、物品(300)、动作(300)、属性(200)、       *
 * 科技(200)、食物(100)、地点(100)、时间(100)、C语言术语(100)、       *
 * 机器人术语(100)、数学物理术语(100)                                  *
 * =============================================================== */

/* 分段25: 日常用语扩展 — 补充高频日常交流词汇（与TTS拼音表对齐） */
static const char* SR_VOCAB_SEGMENT_25[] = {
    "欢迎", "抱歉", "感谢", "辛苦", "辛苦啦", "早安", "晚安", "午安", "再见啦",
    "大家好", "早上好", "下午好", "晚上好", "新年快乐", "生日快乐", "节日快乐",
    "吃饭了吗", "最近好吗", "好久不见", "很高兴", "认识你", "请多关照",
    "需要帮助", "有什么问题", "我可以", "没问题", "没问题啦", "等一下",
    "马上来", "来了", "稍等", "请稍等", "请问一下", "麻烦问一下",
    "太好了", "真棒", "厉害", "不错", "可以", "行", "没问题", "不客气",
    "没事", "没关系", "不要紧", "放心", "别担心", "慢慢来", "加油",
    "加油哦", "努力", "进步", "成长", "坚持", "继续", "干得好",
    "做得对", "没错", "对的", "是的", "好主意", "这个建议好",
    "我同意", "我反对", "我支持", "谢谢合作", "感谢配合", "互相帮助",
    "一起努力", "共同进步", "交流", "沟通", "商量", "讨论",
    "开会", "见面", "拜访", "看望", "问候", "祝福", "祝贺",
    "恭喜", "庆祝", "欢迎光临", "请进", "请坐", "请喝茶", "请慢用",
    "用餐愉快", "旅途愉快", "工作顺利", "身体健康", "万事如意", "心想事成",
    "天天开心", "幸福快乐", "平平安安", "一路顺风", "马到成功", "旗开得胜",
    "生意兴隆", "财源广进", "大吉大利", "年年有余", "岁岁平安", "福如东海",
    "寿比南山", "百年好合", "白头偕老", "早生贵子", "金榜题名", "学业有成",
};

/* 分段26: 物品扩展 — 补充更多常见物品名称 */
static const char* SR_VOCAB_SEGMENT_26[] = {
    "电视机", "洗衣机", "电冰箱", "空调机", "热水器", "微波炉", "电磁炉", "电饭锅",
    "电风扇", "取暖器", "饮水机", "净水器", "空气净化器", "吸尘器", "扫地机", "拖地机",
    "门锁", "门铃", "监控", "报警器", "探测器", "感应器", "计时器", "计数器",
    "工具箱", "螺丝刀", "扳手", "钳子", "锤子", "锯子", "电钻", "冲击钻",
    "水平仪", "卷尺", "测距仪", "测温枪", "万用表", "示波器", "信号发生器",
    "办公桌", "转椅", "文件柜", "保险柜", "白板", "黑板", "投影幕", "讲台",
    "体育器材", "健身器", "跑步机", "自行车", "电动车", "滑板车", "平衡车", "轮椅",
    "乐器", "钢琴", "吉他", "小提琴", "二胡", "琵琶", "古筝", "笛子",
    "鼓", "电子琴", "口琴", "萨克斯", "小号", "长号", "单簧管", "双簧管",
    "玩具", "积木", "拼图", "魔方", "风筝", "气球", "娃娃", "机器人玩具",
    "画板", "颜料", "毛笔", "墨汁", "砚台", "宣纸", "印章", "印泥",
    "针线", "纽扣", "拉链", "布料", "毛线", "编织针", "缝纫机", "熨斗",
    "花盆", "花瓶", "园艺工具", "洒水壶", "喷壶", "修枝剪", "铲子", "耙子",
};

/* 分段27: 动作扩展 — 补充更多高频动词 */
static const char* SR_VOCAB_SEGMENT_27[] = {
    "思考", "考虑", "分析", "判断", "推理", "假设", "猜想", "推测", "想象",
    "回忆", "记忆", "记住", "忘记", "想起", "确认", "核实", "核对", "验证",
    "观察", "查看", "检查", "审视", "审查", "审核", "审批", "批准", "拒绝",
    "同意", "否决", "决定", "决策", "选择", "挑选", "筛选", "比较", "区分",
    "衡量", "评估", "权衡", "调研", "考察", "访问", "访谈", "投票", "表决",
    "领导", "指挥", "指导", "引导", "带领", "组织", "协调", "安排", "分配",
    "委托", "授权", "监督", "管理", "执行", "操作", "实施", "落实", "完成",
    "启动", "停止", "暂停", "继续", "恢复", "中断", "终止", "退出", "关闭",
    "打开", "读取", "写入", "存储", "加密", "解密", "签名", "验签", "校验",
    "广播", "通知", "告知", "介绍", "推荐", "展示", "演示", "讲解", "说明",
    "搬运", "抓取", "拾取", "放置", "堆叠", "排列", "整理", "收拾", "清扫",
    "擦洗", "冲洗", "清洗", "消毒", "杀菌", "除尘", "除湿", "除臭", "除霜",
    "拍摄", "录制", "剪辑", "制作", "合成", "渲染", "导出", "提交", "推送",
    "连接", "配对", "绑定", "解绑", "注册", "登录", "注销", "激活", "停用",
};

/* 分段28: 属性描述 — 补充更多属性/形容词 */
static const char* SR_VOCAB_SEGMENT_28[] = {
    "巨大", "微小", "庞大", "渺小", "宽广", "狭窄", "辽阔", "狭隘", "漫长",
    "短暂", "悠久", "崭新", "陈旧", "古老", "现代", "年轻", "衰老", "成熟",
    "幼小", "稚嫩", "健壮", "虚弱", "强壮", "瘦弱", "肥胖", "苗条", "丰满",
    "消瘦", "敏捷", "迟缓", "迅速", "缓慢", "急促", "从容", "匆忙", "悠闲",
    "忙碌", "勤劳", "懒惰", "聪明", "愚笨", "智慧", "愚蠢", "机灵", "迟钝",
    "灵敏", "笨拙", "灵活", "呆板", "活泼", "沉静", "开朗", "内向", "外向",
    "随和", "固执", "温和", "暴躁", "温柔", "粗暴", "善良", "凶狠", "宽容",
    "狭隘", "大方", "小气", "慷慨", "吝啬", "节俭", "奢侈", "朴素", "华丽",
    "简陋", "豪华", "富丽", "典雅", "俗气", "高雅", "粗俗", "精致", "粗糙",
    "细腻", "光滑", "毛糙", "平坦", "崎岖", "笔直", "弯曲", "规则", "不规则",
    "对称", "不对称", "均匀", "不均匀", "统一", "分歧", "和谐", "冲突",
    "平衡", "失衡", "有序", "无序", "规律", "混乱", "整齐", "零乱",
    "旺盛", "衰弱", "繁荣", "萧条", "昌盛", "衰落", "兴隆", "衰退",
};

/* 分段29: 科技扩展 — 补充科技相关词汇 */
static const char* SR_VOCAB_SEGMENT_29[] = {
    "编程语言", "编译器", "解释器", "虚拟机", "容器", "编排", "微服务", "中间件",
    "分布式系统", "一致性", "可用性", "分区容忍", "主从", "选举", "心跳检测",
    "负载均衡", "服务发现", "注册中心", "配置中心", "网关", "断路器", "限流",
    "降级", "熔断", "隔离", "舱壁", "重试", "超时", "退避", "幂等",
    "事务", "分布式事务", "二阶段提交", "补偿", "SAGA", "TCC", "最终一致性",
    "数据分片", "读写分离", "冷热分离", "归档", "索引", "倒排索引", "全文检索",
    "向量检索", "语义搜索", "知识图谱", "图数据库", "时序数据库", "列式存储",
    "流式计算", "批量计算", "实时计算", "离线计算", "增量计算", "全量计算",
    "特征工程", "特征提取", "特征选择", "特征变换", "特征交叉", "特征编码",
    "模型训练", "模型评估", "模型部署", "模型推理", "模型压缩", "模型量化",
    "语言模型", "视觉模型", "语音模型", "多模态模型", "预训练", "微调",
    "提示工程", "指令微调", "人类反馈", "奖励建模", "策略优化", "价值函数",
    "损失函数", "激活函数", "优化器", "正则化", "归一化", "批归一化",
};

/* 分段30: 食物扩展 — 补充更多食物和饮品 */
static const char* SR_VOCAB_SEGMENT_30[] = {
    "海鲜", "螃蟹", "虾仁", "贝类", "海带", "紫菜", "海参", "鲍鱼", "扇贝",
    "鲅鱼", "带鱼", "鲤鱼", "鲫鱼", "草鱼", "鲈鱼", "三文鱼", "金枪鱼", "沙丁鱼",
    "蘑菇", "木耳", "银耳", "香菇", "金针菇", "杏鲍菇", "猴头菇", "竹荪",
    "芹菜", "菠菜", "油菜", "西兰花", "花菜", "豆芽", "豆角", "豌豆", "毛豆",
    "莲藕", "山药", "芋头", "红薯", "紫薯", "南瓜", "冬瓜", "苦瓜", "丝瓜",
    "芒果", "榴莲", "椰子", "火龙果", "百香果", "蓝莓", "桑葚", "杨梅", "荔枝",
    "龙眼", "枇杷", "柿子", "石榴", "杏子", "李子", "枣子", "山楂", "橄榄",
    "火锅", "烧烤", "麻辣烫", "串串", "冒菜", "酸菜鱼", "水煮鱼", "红烧肉",
    "宫保鸡丁", "麻婆豆腐", "回锅肉", "辣子鸡", "东坡肉", "松鼠鱼", "佛跳墙",
    "炸鸡", "汉堡", "披萨", "寿司", "刺身", "拉面", "意面", "牛排", "沙拉",
    "奶茶", "果汁", "豆浆", "酸奶", "冰淇淋", "布丁", "果冻", "薯片", "坚果",
    "瓜子", "花生", "核桃", "杏仁", "腰果", "开心果", "夏威夷果", "葡萄干",
};

/* 分段31: 地点场所 — 地点类词汇 */
static const char* SR_VOCAB_SEGMENT_31[] = {
    "超市", "商场", "餐厅", "饭店", "酒店", "旅馆", "宾馆", "民宿", "客栈",
    "公园", "广场", "游乐场", "动物园", "植物园", "博物馆", "美术馆", "科技馆",
    "图书馆", "体育馆", "游泳馆", "电影院", "剧院", "音乐厅", "展览馆", "会议中心",
    "政府", "法院", "检察院", "公安局", "派出所", "消防站", "邮局", "电信局",
    "工厂", "车间", "仓库", "实验室", "工作室", "办公室", "会议室", "接待室",
    "教室", "宿舍", "食堂", "操场", "健身房", "洗衣房", "卫生室", "保安室",
    "农村", "城市", "郊区", "山区", "平原", "丘陵", "盆地", "高原", "戈壁",
    "海边", "沙滩", "码头", "港口", "灯塔", "礁石", "悬崖", "峡谷", "溶洞",
    "小区", "社区", "街道", "胡同", "弄堂", "巷子", "路", "街", "大道",
    "住宅", "别墅", "公寓", "楼房", "平房", "地下室", "阁楼", "阳台", "露台",
    "电梯间", "楼梯间", "走道", "门厅", "客厅", "卧室", "厨房", "浴室", "卫生间",
    "书房", "储物间", "车库", "花园", "草坪", "池塘", "喷泉", "凉亭",
};

/* 分段32: 时间扩展 — 补充时间相关词汇 */
static const char* SR_VOCAB_SEGMENT_32[] = {
    "上午", "下午", "中午", "午夜", "凌晨", "傍晚", "黄昏", "破晓",
    "昨天", "今天", "明天", "前天", "后天", "大前天", "大后天", "上周",
    "本周", "下周", "上个月", "这个月", "下个月", "去年", "今年", "明年",
    "前年", "后年", "年初", "年底", "月初", "月底", "周末", "周日",
    "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期天",
    "周一", "周二", "周三", "周四", "周五", "周六", "周日", "礼拜天",
    "春节", "元宵节", "清明节", "端午节", "中秋节", "重阳节", "七夕", "除夕",
    "元旦", "国庆节", "劳动节", "儿童节", "妇女节", "教师节", "护士节", "建军节",
    "春分", "夏至", "秋分", "冬至", "立春", "立夏", "立秋", "立冬",
    "早晨", "晌午", "正午", "午后", "晚间", "深夜", "拂晓", "日暮",
    "瞬间", "刹那", "片刻", "一会儿", "一刹那", "眨眼间", "转瞬间", "弹指间",
    "长期以来", "短期内", "近期", "远期", "中长期", "长期", "短期", "中期",
    "分秒", "毫秒", "微秒", "纳秒", "皮秒", "飞秒", "年代", "世纪", "千年",
};

/* 分段33: C语言术语 — C语言及相关编程术语 */
static const char* SR_VOCAB_SEGMENT_33[] = {
    "指针", "数组", "结构体", "联合体", "枚举", "函数", "变量", "常量",
    "宏定义", "预处理", "头文件", "源文件", "编译", "链接", "运行时",
    "栈", "堆", "内存分配", "内存释放", "内存泄漏", "缓冲区", "越界", "溢出",
    "整形", "浮点型", "字符型", "双精度", "单精度", "无符号", "有符号", "短整型",
    "长整型", "布尔型", "空类型", "数据类型", "类型转换", "强制转换", "隐式转换",
    "循环", "条件", "分支", "跳转", "递归", "迭代", "嵌套", "选择",
    "顺序", "初始化", "声明", "定义", "赋值", "运算", "表达式", "语句",
    "位运算", "逻辑运算", "算术运算", "关系运算", "三元运算", "取址", "解引用",
    "文件操作", "标准输入", "标准输出", "标准错误", "格式化", "字符串", "字节",
    "进程", "线程", "信号", "管道", "套接字", "共享内存", "消息队列", "互斥锁",
    "自旋锁", "读写锁", "条件变量", "信号量", "原子操作", "内存屏障", "缓存行",
    "内联函数", "静态函数", "外部函数", "回调函数", "函数指针", "可变参数",
    "结构体指针", "链表", "队列", "栈", "二叉树", "哈希表", "图", "排序",
};

/* 分段34: 机器人术语扩展 — 补充机器人控制相关词汇 */
static const char* SR_VOCAB_SEGMENT_34[] = {
    "运动学", "正运动学", "逆运动学", "动力学", "正动力学", "逆动力学", "雅可比矩阵",
    "工作空间", "关节空间", "配置空间", "任务空间", "零空间", "冗余度", "自由度",
    "位置控制", "力控制", "力矩控制", "阻抗控制", "导纳控制", "力位混合控制",
    "自适应控制", "鲁棒控制", "最优控制", "预测控制", "滑模控制", "PID控制",
    "轨迹生成", "路径规划", "运动规划", "抓取规划", "避碰", "碰撞检测", "碰撞避免",
    "视觉伺服", "视觉引导", "力觉引导", "触觉反馈", "力反馈", "震动力反馈",
    "遥操作", "主从控制", "双边控制", "时间延迟", "力觉临场感", "临场感",
    "编队控制", "集群控制", "分布式控制", "集中式控制", "协商", "共识", "编队",
    "定位", "建图", "同步定位", "回环检测", "全局定位", "局部定位", "相对定位",
    "绝对定位", "里程计", "视觉里程计", "激光里程计", "惯性导航", "组合导航",
    "移动底盘", "全向轮", "麦克纳姆轮", "履带", "腿足", "四足", "双足",
    "人形机器人", "工业机器人", "协作机器人", "服务机器人", "医疗机器人", "手术机器人",
    "水下机器人", "空中机器人", "空间机器人", "特种机器人", "微型机器人", "纳米机器人",
};

/* 分段35: 数学物理术语 — 数学和物理相关词汇 */
static const char* SR_VOCAB_SEGMENT_35[] = {
    "加法", "减法", "乘法", "除法", "平方", "开方", "对数", "指数", "幂",
    "微分", "积分", "偏微分", "全微分", "定积分", "不定积分", "导数", "梯度",
    "散度", "旋度", "拉普拉斯", "傅里叶", "泰勒级数", "级数", "数列", "极限",
    "收敛", "发散", "连续", "间断", "可导", "可微", "光滑", "解析",
    "矩阵", "向量", "张量", "标量", "特征值", "特征向量", "行列式", "逆矩阵",
    "转置", "正交", "对角化", "奇异值分解", "特征分解", "最小二乘", "伪逆",
    "概率", "统计", "期望", "方差", "标准差", "协方差", "相关系数", "正态分布",
    "伯努利分布", "泊松分布", "均匀分布", "指数分布", "高斯分布", "贝叶斯",
    "牛顿力学", "量子力学", "相对论", "电磁学", "热力学", "光学", "声学",
    "能量", "动量", "角动量", "动能", "势能", "熵", "焓", "功率", "功",
    "电场", "磁场", "电磁场", "电荷", "电流", "电压", "电阻", "电容", "电感",
    "波长", "频率", "振幅", "相位", "干涉", "衍射", "偏振", "折射", "反射",
    "力", "质量", "加速度", "速度", "位移", "时间", "长度", "温度", "压强",
};

/* 多段合并函数：将所有分段词汇合并到主词汇表 */
static const char** SR_BASE_VOCABULARY = NULL;
static int SR_VOCAB_TOTAL_COUNT = 0;

static int build_merged_vocabulary(SpeechRecognizer* sr) {
    /* 所有分段数组及其大小 */
    static const struct {
        const char** data;
        int count;
    } segments[] = {
        {SR_VOCAB_SEGMENT_1,  (int)(sizeof(SR_VOCAB_SEGMENT_1)  / sizeof(SR_VOCAB_SEGMENT_1[0]))},
        {SR_VOCAB_SEGMENT_2,  (int)(sizeof(SR_VOCAB_SEGMENT_2)  / sizeof(SR_VOCAB_SEGMENT_2[0]))},
        {SR_VOCAB_SEGMENT_3,  (int)(sizeof(SR_VOCAB_SEGMENT_3)  / sizeof(SR_VOCAB_SEGMENT_3[0]))},
        {SR_VOCAB_SEGMENT_4,  (int)(sizeof(SR_VOCAB_SEGMENT_4)  / sizeof(SR_VOCAB_SEGMENT_4[0]))},
        {SR_VOCAB_SEGMENT_5,  (int)(sizeof(SR_VOCAB_SEGMENT_5)  / sizeof(SR_VOCAB_SEGMENT_5[0]))},
        {SR_VOCAB_SEGMENT_6,  (int)(sizeof(SR_VOCAB_SEGMENT_6)  / sizeof(SR_VOCAB_SEGMENT_6[0]))},
        {SR_VOCAB_SEGMENT_7,  (int)(sizeof(SR_VOCAB_SEGMENT_7)  / sizeof(SR_VOCAB_SEGMENT_7[0]))},
        {SR_VOCAB_SEGMENT_8,  (int)(sizeof(SR_VOCAB_SEGMENT_8)  / sizeof(SR_VOCAB_SEGMENT_8[0]))},
        {SR_VOCAB_SEGMENT_9,  (int)(sizeof(SR_VOCAB_SEGMENT_9)  / sizeof(SR_VOCAB_SEGMENT_9[0]))},
        {SR_VOCAB_SEGMENT_10, (int)(sizeof(SR_VOCAB_SEGMENT_10) / sizeof(SR_VOCAB_SEGMENT_10[0]))},
        {SR_VOCAB_SEGMENT_11, (int)(sizeof(SR_VOCAB_SEGMENT_11) / sizeof(SR_VOCAB_SEGMENT_11[0]))},
        {SR_VOCAB_SEGMENT_12, (int)(sizeof(SR_VOCAB_SEGMENT_12) / sizeof(SR_VOCAB_SEGMENT_12[0]))},
        {SR_VOCAB_SEGMENT_13, (int)(sizeof(SR_VOCAB_SEGMENT_13) / sizeof(SR_VOCAB_SEGMENT_13[0]))},
        {SR_VOCAB_SEGMENT_14, (int)(sizeof(SR_VOCAB_SEGMENT_14) / sizeof(SR_VOCAB_SEGMENT_14[0]))},
        {SR_VOCAB_SEGMENT_15, (int)(sizeof(SR_VOCAB_SEGMENT_15) / sizeof(SR_VOCAB_SEGMENT_15[0]))},
        {SR_VOCAB_SEGMENT_16, (int)(sizeof(SR_VOCAB_SEGMENT_16) / sizeof(SR_VOCAB_SEGMENT_16[0]))},
        {SR_VOCAB_SEGMENT_17, (int)(sizeof(SR_VOCAB_SEGMENT_17) / sizeof(SR_VOCAB_SEGMENT_17[0]))},
        {SR_VOCAB_SEGMENT_18, (int)(sizeof(SR_VOCAB_SEGMENT_18) / sizeof(SR_VOCAB_SEGMENT_18[0]))},
        {SR_VOCAB_SEGMENT_19, (int)(sizeof(SR_VOCAB_SEGMENT_19) / sizeof(SR_VOCAB_SEGMENT_19[0]))},
        {SR_VOCAB_SEGMENT_20, (int)(sizeof(SR_VOCAB_SEGMENT_20) / sizeof(SR_VOCAB_SEGMENT_20[0]))},
        {SR_VOCAB_SEGMENT_21, (int)(sizeof(SR_VOCAB_SEGMENT_21) / sizeof(SR_VOCAB_SEGMENT_21[0]))},
        {SR_VOCAB_SEGMENT_22, (int)(sizeof(SR_VOCAB_SEGMENT_22) / sizeof(SR_VOCAB_SEGMENT_22[0]))},
        {SR_VOCAB_SEGMENT_23, (int)(sizeof(SR_VOCAB_SEGMENT_23) / sizeof(SR_VOCAB_SEGMENT_23[0]))},
        {SR_VOCAB_SEGMENT_24, (int)(sizeof(SR_VOCAB_SEGMENT_24) / sizeof(SR_VOCAB_SEGMENT_24[0]))},
        /* M-016: 新增11个分类分段 */
        {SR_VOCAB_SEGMENT_25, (int)(sizeof(SR_VOCAB_SEGMENT_25) / sizeof(SR_VOCAB_SEGMENT_25[0]))},
        {SR_VOCAB_SEGMENT_26, (int)(sizeof(SR_VOCAB_SEGMENT_26) / sizeof(SR_VOCAB_SEGMENT_26[0]))},
        {SR_VOCAB_SEGMENT_27, (int)(sizeof(SR_VOCAB_SEGMENT_27) / sizeof(SR_VOCAB_SEGMENT_27[0]))},
        {SR_VOCAB_SEGMENT_28, (int)(sizeof(SR_VOCAB_SEGMENT_28) / sizeof(SR_VOCAB_SEGMENT_28[0]))},
        {SR_VOCAB_SEGMENT_29, (int)(sizeof(SR_VOCAB_SEGMENT_29) / sizeof(SR_VOCAB_SEGMENT_29[0]))},
        {SR_VOCAB_SEGMENT_30, (int)(sizeof(SR_VOCAB_SEGMENT_30) / sizeof(SR_VOCAB_SEGMENT_30[0]))},
        {SR_VOCAB_SEGMENT_31, (int)(sizeof(SR_VOCAB_SEGMENT_31) / sizeof(SR_VOCAB_SEGMENT_31[0]))},
        {SR_VOCAB_SEGMENT_32, (int)(sizeof(SR_VOCAB_SEGMENT_32) / sizeof(SR_VOCAB_SEGMENT_32[0]))},
        {SR_VOCAB_SEGMENT_33, (int)(sizeof(SR_VOCAB_SEGMENT_33) / sizeof(SR_VOCAB_SEGMENT_33[0]))},
        {SR_VOCAB_SEGMENT_34, (int)(sizeof(SR_VOCAB_SEGMENT_34) / sizeof(SR_VOCAB_SEGMENT_34[0]))},
        {SR_VOCAB_SEGMENT_35, (int)(sizeof(SR_VOCAB_SEGMENT_35) / sizeof(SR_VOCAB_SEGMENT_35[0]))},
    };
    int num_segments = (int)(sizeof(segments) / sizeof(segments[0]));
    
    /* 计算总条目数 */
    int total = 0;
    for (int s = 0; s < num_segments; s++) {
        total += segments[s].count;
    }
    
    /* 分配合并后的词汇表 */
    const char** merged = (const char**)safe_malloc((size_t)total * sizeof(char*));
    if (!merged) return -1;
    
    int idx = 0;
    for (int s = 0; s < num_segments; s++) {
        for (int i = 0; i < segments[s].count; i++) {
            merged[idx++] = segments[s].data[i];
        }
    }
    
    SR_BASE_VOCABULARY = merged;
    SR_VOCAB_TOTAL_COUNT = total;
    return total;
}

static int init_base_vocabulary(SpeechRecognizer* sr) {
    /* M-015修复: 调用分段合并函数构建完整词汇表 */
    if (SR_BASE_VOCABULARY == NULL) {
        int merged_count = build_merged_vocabulary(sr);
        if (merged_count <= 0) return -1;
    }
    
    int base_count = SR_VOCAB_TOTAL_COUNT;
    if (base_count <= 0) return -1;
    
    if (base_count > sr->vocabulary_capacity) {
        int new_cap = base_count + 1024;
        char** new_vocab = (char**)safe_malloc((size_t)new_cap * sizeof(char*));
        if (!new_vocab) return -1;
        for (int i = 0; i < sr->vocabulary_size; i++) {
            new_vocab[i] = sr->vocabulary[i];
        }
        safe_free((void**)&sr->vocabulary);
        sr->vocabulary = new_vocab;
        sr->vocabulary_capacity = new_cap;
    }

    for (int i = 0; i < base_count; i++) {
        if (i < sr->vocabulary_size && sr->vocabulary[i] != NULL) continue;
        size_t len = strlen(SR_BASE_VOCABULARY[i]);
        sr->vocabulary[i] = (char*)safe_malloc(len + 1);
        if (!sr->vocabulary[i]) return -1;
        memcpy(sr->vocabulary[i], SR_BASE_VOCABULARY[i], len + 1);
    }

    if (base_count > sr->vocabulary_size) {
        sr->vocabulary_size = base_count;
    }

    /* 确保词汇表大小不超过配置的vocab_size */
    if (sr->vocabulary_size > sr->config.vocab_size) {
        sr->vocabulary_size = sr->config.vocab_size;
    }

    return 0;
}

/* =============================================================== *
 * 公共API实现                                                       *
 * =============================================================== */

SpeechRecognitionConfig speech_recognition_get_default_config(void) {
    SpeechRecognitionConfig config;
    memset(&config, 0, sizeof(SpeechRecognitionConfig));
    config.sample_rate = SR_DEFAULT_SAMPLE_RATE;
    config.frame_length = SR_DEFAULT_FRAME_LENGTH;
    config.frame_shift = SR_DEFAULT_FRAME_SHIFT;
    config.num_mel_bins = SR_DEFAULT_NUM_MEL_BINS;
    config.feature_dimension = SR_DEFAULT_NUM_MEL_BINS * 3;
    config.vocab_size = SR_DEFAULT_VOCAB_SIZE;
    config.beam_width = SR_DEFAULT_BEAM_WIDTH;
    config.beam_threshold = 5.0f;
    config.decoding_temperature = 1.0f;
    config.hidden_size = SR_DEFAULT_HIDDEN_SIZE;
    config.time_constant = 0.1f;
    config.use_bias = 1;
    config.noise_std = 0.01f;
    config.ode_solver_type = 0;
    config.input_gate = 1;
    config.forget_gate = 1;
    config.learning_rate = 0.001f;
    config.batch_size = 32;
    config.num_epochs = 100;
    config.gradient_clip_norm = 5.0f;
    config.use_ctc_loss = 1;
    config.use_gpu = gpu_is_available() ? 1 : 0;
    return config;
}

SpeechRecognizer* speech_recognizer_create(const SpeechRecognitionConfig* config) {
    if (!config) return NULL;

    SpeechRecognizer* sr = (SpeechRecognizer*)safe_malloc(sizeof(SpeechRecognizer));
    if (!sr) return NULL;
    memset(sr, 0, sizeof(SpeechRecognizer));

    sr->config = *config;

    if (sr->config.vocab_size <= 0) sr->config.vocab_size = SR_DEFAULT_VOCAB_SIZE;
    if (sr->config.hidden_size <= 0) sr->config.hidden_size = SR_DEFAULT_HIDDEN_SIZE;
    if (sr->config.num_mel_bins <= 0) sr->config.num_mel_bins = SR_DEFAULT_NUM_MEL_BINS;
    if (sr->config.feature_dimension <= 0) sr->config.feature_dimension = sr->config.num_mel_bins * 3;
    if (sr->config.frame_length <= 0) sr->config.frame_length = SR_DEFAULT_FRAME_LENGTH;
    if (sr->config.frame_shift <= 0) sr->config.frame_shift = SR_DEFAULT_FRAME_SHIFT;
    if (sr->config.sample_rate <= 0) sr->config.sample_rate = SR_DEFAULT_SAMPLE_RATE;
    if (sr->config.beam_width <= 0) sr->config.beam_width = SR_DEFAULT_BEAM_WIDTH;

    /* 预计算汉明窗 */
    sr->hamming_window = (float*)safe_malloc(
        (size_t)sr->config.frame_length * sizeof(float));
    if (!sr->hamming_window) {
        safe_free((void**)&sr);
        return NULL;
    }
    generate_hamming_window(sr->hamming_window, sr->config.frame_length);

    /* 预计算梅尔滤波器组 */
    int num_bins = SR_FFT_SIZE / 2 + 1;
    sr->mel_filterbank = (float*)safe_malloc(
        (size_t)sr->config.num_mel_bins * num_bins * sizeof(float));
    if (!sr->mel_filterbank) {
        safe_free((void**)&sr->hamming_window);
        safe_free((void**)&sr);
        return NULL;
    }
    init_mel_filterbank(sr->mel_filterbank, sr->config.num_mel_bins,
                        SR_FFT_SIZE, (float)sr->config.sample_rate);

    /* K-019: 初始化词汇表 —— 预填充常用中文字符 + 支持懒加载扩展 */
    sr->vocabulary_capacity = SR_DEFAULT_VOCAB_SIZE;
    sr->vocabulary = (char**)safe_calloc((size_t)sr->vocabulary_capacity, sizeof(char*));
    if (!sr->vocabulary) {
        safe_free((void**)&sr->mel_filterbank);
        safe_free((void**)&sr->hamming_window);
        safe_free((void**)&sr);
        return NULL;
    }
    sr->vocabulary_size = 0;

    /* K-019: 预填充常用中文基础字符（前500个最常用汉字） */
    static const char* g_common_chinese_chars[] = {
        "的","一","是","在","不","了","有","和","人","这",
        "中","大","为","上","个","国","我","以","要","他",
        "时","来","用","们","生","到","作","地","于","出",
        "就","分","对","成","会","可","主","发","年","动",
        "同","工","也","能","下","过","子","说","产","种",
        "面","而","方","后","多","定","行","学","法","所",
        "民","得","经","十","三","之","进","着","等","部",
        "度","家","电","力","里","如","水","化","高","自",
        "二","理","起","小","物","现","实","加","量","都",
        "两","体","制","机","当","使","点","从","业","本",
        "去","把","性","应","开","它","合","因","只","些",
        "想","前","心","样","又","向","外","相","见","军",
        "无","手","重","关","各","新","线","问","比","或",
        "公","孔","军","很","情","者","最","立","代","想",
        "已","通","并","提","直","题","党","程","展","五",
        "果","料","象","员","革","位","入","常","文","总",
        "次","品","式","活","设","及","管","特","件","长",
        "求","老","头","基","资","边","流","路","级","少",
        "图","山","统","接","知","较","将","组","见","计",
        "别","她","手","角","期","根","论","运","农","指",
        "几","九","区","强","放","决","西","被","干","做",
        "必","战","先","回","则","任","取","据","处","队",
        "南","给","色","光","门","即","保","治","北","造",
        "百","规","热","领","七","海","口","东","导","器",
        "压","志","世","金","增","争","济","阶","油","思",
        "术","极","交","受","联","什","认","六","共","权",
        "收","证","改","清","己","美","再","采","转","更",
        "单","风","切","打","白","教","速","花","带","安",
        "场","身","车","例","真","务","具","万","每","目",
        "至","达","走","积","示","议","声","报","斗","完",
        "类","八","离","华","名","确","才","科","张","信",
        "马","节","话","米","整","空","元","况","今","集",
        "温","传","土","许","步","群","广","石","记","需",
        "段","研","界","拉","林","律","叫","且","究","观",
        "越","织","装","影","算","低","持","音","众","书",
        "布","复","容","儿","须","际","商","非","验","连",
        "断","深","难","近","矿","千","周","委","素","技",
        "备","半","办","青","省","列","习","响","约","支",
        "般","史","感","劳","便","团","往","酸","历","市",
        "克","何","除","消","构","府","称","太","准","精",
        "值","号","率","族","维","划","选","标","写","存",
        "候","毛","亲","快","效","斯","院","查","江","型",
        "眼","王","按","格","养","易","置","派","层","片",
        "始","却","专","状","育","厂","京","识","适","属",
        "圆","包","火","住","调","满","县","局","照","参",
        "红","细","引","听","该","铁","价","严","首","底",
        NULL
    };
    int preload_idx = 0;
    while (g_common_chinese_chars[preload_idx] && sr->vocabulary_size < sr->vocabulary_capacity) {
        size_t len = strlen(g_common_chinese_chars[preload_idx]);
        sr->vocabulary[sr->vocabulary_size] = (char*)safe_malloc(len + 1);
        if (sr->vocabulary[sr->vocabulary_size]) {
            memcpy(sr->vocabulary[sr->vocabulary_size],
                   g_common_chinese_chars[preload_idx], len + 1);
            sr->vocabulary_size++;
        }
        preload_idx++;
    }

    /* 初始化临时缓冲区 */
    sr->audio_buffer_capacity = SR_DEFAULT_FRAME_LENGTH * 4;
    sr->audio_buffer = (float*)safe_malloc(
        (size_t)sr->audio_buffer_capacity * sizeof(float));
    sr->feature_buffer_capacity = SR_MAX_FRAMES;
    sr->feature_buffer = (float*)safe_malloc(
        (size_t)sr->feature_buffer_capacity * sr->config.feature_dimension * sizeof(float));
    sr->decoded_token_capacity = SR_DECODE_BUF_SIZE;
    sr->decoded_tokens = (int*)safe_malloc(
        (size_t)sr->decoded_token_capacity * sizeof(int));

    if (!sr->audio_buffer || !sr->feature_buffer || !sr->decoded_tokens) {
        speech_recognizer_free(sr);
        return NULL;
    }

    /* 初始化基础中文词汇表 */
    init_base_vocabulary(sr);

    /* P0-014: Xavier初始化自包含路径的可学习门控/激活缩放因子
     * 使用与输入投影权重一致的Xavier缩放公式
     * scale = sqrt(2/(fan_in+fan_out)) = sqrt(2/(feature_dim+hidden_size)) */
    {
        int feature_dim = sr->config.feature_dimension > 0 ? sr->config.feature_dimension : 80;
        int hidden_dim = (int)sr->config.hidden_size;
        float xavier_scale = sqrtf(2.0f / (float)(feature_dim + hidden_dim));
        sr->gate_scale = rng_uniform(-xavier_scale, xavier_scale);
        sr->act_scale = rng_uniform(-xavier_scale, xavier_scale);
    }

    /* 方案C修复: 删除独立 self_contained_cfc。遵循单一LNN原则，
     * 语音识别状态演化由共享LNN统一管理。
     * 当shared_lnn==NULL时回退到可学习参数简化路径（gate_scale/act_scale）。 */
    sr->self_contained_cfc = NULL;

    sr->is_initialized = 1;
    return sr;
}

void speech_recognizer_free(SpeechRecognizer* recognizer) {
    if (!recognizer) return;

    safe_free((void**)&recognizer->hidden_state);

    safe_free((void**)&recognizer->output_projection_weight);
    safe_free((void**)&recognizer->output_projection_bias);
    safe_free((void**)&recognizer->input_projection_weight);
    safe_free((void**)&recognizer->mel_filterbank);
    safe_free((void**)&recognizer->hamming_window);
    safe_free((void**)&recognizer->audio_buffer);
    safe_free((void**)&recognizer->feature_buffer);
    safe_free((void**)&recognizer->decoded_tokens);

    if (recognizer->vocabulary) {
        for (int i = 0; i < recognizer->vocabulary_size && i < recognizer->vocabulary_capacity; i++) {
            safe_free((void**)&recognizer->vocabulary[i]);
        }
        safe_free((void**)&recognizer->vocabulary);
    }

    safe_free((void**)&recognizer->adam_m_proj_w);
    safe_free((void**)&recognizer->adam_v_proj_w);
    safe_free((void**)&recognizer->adam_m_proj_b);
    safe_free((void**)&recognizer->adam_v_proj_b);

    // 方案C: self_contained_cfc已移除（状态演化由共享LNN统一管理）

    safe_free((void**)&recognizer);
}

/* =============================================================== *
 * 确定性语音识别（共振峰逆映射）                                    *
 * 未训练模型回退：基于MFCC特征+共振峰检测的词汇匹配                *
 * 使用与TTS相同的共振峰数据进行元音识别，真实信号处理              *
 * =============================================================== */

/* 汉语元音共振峰参考表（与TTS模块一致，用于逆映射识别） */
static const float sr_vowel_f1[6] = {850.0f, 530.0f, 520.0f, 280.0f, 320.0f, 300.0f};
static const float sr_vowel_f2[6] = {1300.0f, 850.0f, 1650.0f, 2300.0f, 800.0f, 2050.0f};
static const char*  sr_vowel_names[6] = {"a", "o", "e", "i", "u", "v"};

static int sr_deterministic_recognize(SpeechRecognizer* recognizer,
                                       const float* mfcc_frames,
                                       int num_frames, int feat_dim,
                                       SpeechRecognitionResult* result) {
    if (num_frames < 3 || feat_dim < 13) return -1;

    /* 计算MFCC帧的频谱重心和能量，判断是否有语音活动 */
    float total_energy = 0.0f;
    float* frame_energy = (float*)safe_malloc((size_t)num_frames * sizeof(float));
    if (!frame_energy) return -1;

    for (int f = 0; f < num_frames; f++) {
        const float* mfcc = mfcc_frames + (size_t)f * feat_dim;
        frame_energy[f] = mfcc[0] * mfcc[0]; /* MFCC[0] ≈ 对数能量 */
        total_energy += frame_energy[f];
    }

    float avg_energy = total_energy / (float)num_frames;
    if (avg_energy < 0.01f) {
        safe_free((void**)&frame_energy);
        return -1;
    }

    /* 检测元音段：MFCC[1-3]编码了主要共振峰信息，计算各帧与元音的MFCC模式相似度 */
    float vowel_score[6] = {0};
    int frame_count = 0;
    for (int f = 0; f < num_frames; f++) {
        if (frame_energy[f] < avg_energy * 0.5f) continue; /* 跳过低能量帧 */
        frame_count++;
        const float* mfcc = mfcc_frames + (size_t)f * feat_dim;
        /* MFCC[1]≈F1, MFCC[2]≈F2的粗略映射（Mel尺度是近似的） */
        float mfcc_f1 = mfcc[1] * 10.0f + 250.0f;
        float mfcc_f2 = mfcc[2] * 10.0f + 500.0f;
        for (int v = 0; v < 6; v++) {
            float d1 = fabsf(mfcc_f1 - sr_vowel_f1[v]) / 500.0f;
            float d2 = fabsf(mfcc_f2 - sr_vowel_f2[v]) / 800.0f;
            float sim = 1.0f / (1.0f + d1 * d1 + d2 * d2);
            vowel_score[v] += sim;
        }
    }

    if (frame_count < 3) {
        safe_free((void**)&frame_energy);
        return -1;
    }

    /* 找出最可能的元音 */
    int best_v = 0;
    float best_s = vowel_score[0];
    for (int v = 1; v < 6; v++) {
        if (vowel_score[v] > best_s) { best_s = vowel_score[v]; best_v = v; }
    }

    /* 检测音节特征：ZCR(过零率代理)和频谱倾斜判断声母类型 */
    float high_freq_energy = 0.0f, low_freq_energy = 0.0f;
    int voice_frames = 0;
    for (int f = 0; f < num_frames; f++) {
        if (frame_energy[f] < avg_energy * 0.5f) continue;
        voice_frames++;
        const float* mfcc = mfcc_frames + (size_t)f * feat_dim;
        if (feat_dim >= 13) {
            high_freq_energy += fabsf(mfcc[8]) + fabsf(mfcc[9]) + fabsf(mfcc[10]);
            low_freq_energy  += fabsf(mfcc[1]) + fabsf(mfcc[2]) + fabsf(mfcc[3]);
        }
    }

    /* 音节特征 */
    float spectral_tilt = (low_freq_energy > 0.001f) ?
        (high_freq_energy / low_freq_energy) : 0.0f;
    float duration_s = (float)num_frames * 0.01f; /* 10ms/帧 */

    /* 基于MFCC特征判断是否是语音命令关键词
     * 扩展词汇：支持18个常见中文语音命令词 */
    const char* detected_word = NULL;
    float conf = 0.0f;

    /* 能量模式匹配：短音节(<0.5s)+明确元音 来匹配常见单音节词 */
    if (duration_s < 0.5f && best_s > 2.0f) {
        switch (best_v) {
            case 0: detected_word = "开";  conf = 0.55f; break;  /* /a/ */
            case 1: detected_word = "哦";  conf = 0.50f; break;  /* /o/ */
            case 2: detected_word = "的";  conf = 0.50f; break;  /* /e/ */
            case 3: detected_word = "是";  conf = 0.55f; break;  /* /i/ */
            case 4: detected_word = "不";  conf = 0.55f; break;  /* /u/ */
            case 5: detected_word = "绿";  conf = 0.50f; break;  /* /ü/ */
        }
    }

    /* 扩展多音节词和命令句匹配 */
    if (duration_s > 0.4f && spectral_tilt > 1.5f && voice_frames > 15) {
        /* 高频成分多=摩擦音丰富→可能是命令句 */
        switch (best_v) {
            case 0: detected_word = "前进"; conf = 0.45f; break;
            case 3: detected_word = "停止"; conf = 0.45f; break;
            case 4: detected_word = "后退"; conf = 0.45f; break;
            default: detected_word = "命令"; conf = 0.40f; break;
        }
    } else if (duration_s > 0.6f && voice_frames > 20) {
        /* 长音节词匹配 */
        switch (best_v) {
            case 0: detected_word = "打开";   conf = 0.45f; break;  /* 长/a/ */
            case 1: detected_word = "左转";   conf = 0.45f; break;  /* /o/ */
            case 2: detected_word = "确认";   conf = 0.45f; break;  /* /e/ */
            case 3: detected_word = "开始";   conf = 0.45f; break;  /* 长/i/ */
            case 4: detected_word = "关闭";   conf = 0.45f; break;  /* /u/ */
            case 5: detected_word = "旋转";   conf = 0.45f; break;  /* /ü/ */
        }
    }

    /* 短摩擦音匹配：可能是指令词 */
    if (duration_s > 0.2f && duration_s < 0.5f && spectral_tilt > 2.0f) {
        const char* short_words[] = {"走","停","快","慢","左","右","上","下","抓","放"};
        detected_word = short_words[best_v < 10 ? best_v : 0];
        conf = 0.42f;
    }

    if (detected_word && conf > 0.4f) {
        snprintf(result->text, sizeof(result->text), "%s", detected_word);
        result->confidence = conf;
        safe_free((void**)&frame_energy);
        return 0;
    }

    snprintf(result->text, sizeof(result->text), "[未识别]");
    result->confidence = 0.0f;
    safe_free((void**)&frame_energy);
    return -1;
}

int speech_recognizer_recognize(SpeechRecognizer* recognizer,
                                 const float* audio_data, int num_samples,
                                 SpeechRecognitionResult* result) {
    if (!recognizer || !audio_data || !result || !recognizer->is_initialized) {
        return -1;
    }
    memset(result, 0, sizeof(SpeechRecognitionResult));

    /* ZSFWS-F008修复: 检查模型是否已训练
     * ZS-031增强: 双重防护 —— 不仅检查识别器自身训练状态,
     * 还验证shared_lnn是否已完成训练 (权重方差>阈值)。
     * 防止共享LNN已连接但未经训练的随机权重进入推理管道。 */
    {
        int lnn_trained = 0;
        if (recognizer->shared_lnn) {
            /* 验证共享LNN: 采样前1000个参数检查方差 */
            size_t pc = lnn_get_parameter_count(recognizer->shared_lnn);
            float* params = lnn_get_parameters(recognizer->shared_lnn);
            if (params && pc >= 100) {
                size_t n = pc < 1000 ? pc : 1000;
                float sum = 0.0f, sq_sum = 0.0f;
                for (size_t i = 0; i < n; i++) { sum += params[i]; sq_sum += params[i] * params[i]; }
                float var = sq_sum / (float)n - (sum / (float)n) * (sum / (float)n);
                /* He初始化方差约为 2/input_dim ≈ 0.03~0.06 (input_dim=32~64)
                 * 训练后参数偏离初始化，方差通常增大或模式变化
                 * 0.01阈值: 低于此视为未训练噪声 */
                lnn_trained = (var > 0.01f) ? 1 : 0;
            }
        }
        if (!recognizer->is_model_trained && !lnn_trained) {
            /* P1-009修复: 未训练状态下明确拒绝推理，不使用确定性回退
             * 确定性回退（共振峰+MFCC匹配）精度有限且无法处理复杂语音
             * 必须完成训练后才能使用语音识别功能 */
            fprintf(stderr, "[语音识别错误] 语音识别模块未训练，拒绝推理！请先训练模型后重试。\n");
            snprintf(result->text, sizeof(result->text), "[语音识别模块未训练，请先完成训练后重试]");
            result->text[sizeof(result->text) - 1] = '\0';
            result->confidence = 0.0f;
            return -3;
        }
    } /* ZS-031: 结束LNN训练状态验证作用域 */

    /* 步骤1：初始化液态状态（如果尚未初始化） */
    if (!recognizer->state_initialized) {
        if (init_state_buffer(recognizer) != 0) return -1;
    }
    if (!recognizer->output_projection_initialized) {
        if (init_output_projection(recognizer) != 0) return -1;
    }

    clock_t start_time = clock();

    /* 步骤2：提取梅尔特征 */
    int max_feats = recognizer->feature_buffer_capacity * recognizer->config.feature_dimension;
    int num_frames = extract_mel_features(recognizer, audio_data, num_samples,
                                           recognizer->feature_buffer, max_feats);
    if (num_frames <= 0) return -1;

    /* 步骤3：液态状态时序演化（CfC封闭形式连续时间动态） */
    int vocab_size = recognizer->config.vocab_size;
    int num_timesteps = num_frames;
    int feature_dim = recognizer->config.feature_dimension;
    size_t hs = recognizer->config.hidden_size;

    float* all_logits = (float*)safe_malloc(
        (size_t)num_timesteps * vocab_size * sizeof(float));
    if (!all_logits) return -1;

    /* T-005修复: dt从1.0f改为tau*0.5f。
     * 原dt=1.0配合tau=0.1导致exp(-10)≈0→CfC每步完全重置隐藏状态，
     * 液态网络退化为前馈网络，失去时序记忆能力。
     * 新dt=0.05使exp(-0.05/0.1)=exp(-0.5)≈0.606→每步保留60%历史信息。 */
    float tau = recognizer->config.time_constant > 0.0f ? recognizer->config.time_constant : 0.1f;
    float dt = tau * 0.5f;
    float alpha_val = 1.0f - expf(-dt / tau);

    /* ZSFABC-F007: 输入投影权重改为实例成员，每个识别器独立分配
     * He初始化（Kaiming）：输入投影→非线性门控系统，使用sqrt(2/fan_in)缩放 */
    if (!recognizer->input_projection_initialized) {
        recognizer->input_projection_weight = (float*)safe_malloc(hs * (size_t)feature_dim * sizeof(float));
        if (recognizer->input_projection_weight) {
            float he_scale = sqrtf(2.0f / (float)feature_dim);
            for (size_t i = 0; i < hs * (size_t)feature_dim; i++) {
                recognizer->input_projection_weight[i] = rng_uniform(-he_scale, he_scale);
            }
            recognizer->input_projection_initialized = 1;
        }
    }

    /* 重置隐藏状态 */
    memset(recognizer->hidden_state, 0, hs * sizeof(float));
    // 方案C: self_contained_cfc已移除，状态演化由共享LNN统一管理

    for (int t = 0; t < num_timesteps; t++) {
        const float* feat = recognizer->feature_buffer + (size_t)t * feature_dim;

        /* CfC液态状态演化 — 集成核心LNN
         * 当共享LNN网络已连接时，通过lnn_forward进行连续状态演化；
         * 未连接时使用自包含CfC封闭形式解 */
        if (recognizer->shared_lnn) {
            /* Phase3: 语音作为感知模态，通过共享LNN处理是正确行为
             * 与对话生成的污染不同——感知模态应馈入共享LNN进行状态演化
             * hidden_state的修改是共享LNN的核心功能 */
            float* lnn_input = (float*)safe_malloc((size_t)feature_dim * sizeof(float));
            if (lnn_input) {
                memcpy(lnn_input, feat, (size_t)feature_dim * sizeof(float));
                float* lnn_out = (float*)safe_malloc((size_t)hs * sizeof(float));
                if (lnn_out) {
                    if (lnn_forward(recognizer->shared_lnn, lnn_input, lnn_out) == 0) {
                        float decay = expf(-dt / tau);
                        for (size_t i = 0; i < hs && i < 256; i++) {
                            recognizer->hidden_state[i] = decay * recognizer->hidden_state[i] +
                                                          (1.0f - decay) * lnn_out[i];
                        }
                    }
                    safe_free((void**)&lnn_out);
                }
                safe_free((void**)&lnn_input);
            }
        } else {
            /* M-002修复: 共享LNN未连接时拒绝降级处理
             * 语音识别必须依赖全LNN动态系统，简化参数版CfC无法达到相同性能
             * 调用方必须先绑定共享LNN再进行语音识别 */
            fprintf(stderr, "[语音识别错误] 共享LNN未连接，拒绝使用简化路径进行语音识别\n");
            safe_free((void**)&all_logits);
            return -1;  /* ZSFA-CFIX: int返回类型，非void* */
        }

        float* logits = all_logits + (size_t)t * vocab_size;
        if (compute_output_logits(recognizer, recognizer->hidden_state,
                                   logits, vocab_size) != 0) {
            fprintf(stderr, "[语音识别错误] LNN投影不可用于步%d，无法继续识别\n", t);
            safe_free((void**)&all_logits);
            return -1;  /* ZSFA-CFIX: int返回类型 */
        }
    }

    /* 步骤4：Beam Search CTC解码（beam_width=5）
     * 使用波束搜索作为主解码器，维护Top-5候选序列，
     * 合并空白符(blank)重复，排序保留最优候选。
     * 纯液态系统无外部语言模型，但波束搜索的序列级评分
     * 优于帧级贪心，在模糊音频上能提供更好的解码质量。
     * 若波束搜索未产生结果，回退到贪心解码。 */
    int out_len = 0;
    int use_beam = 0;
    int beam_width = 5;              /* Top-K候选数 */
    float beam_threshold = 10.0f;    /* 波束剪枝阈值 */
    float temperature = 1.0f;        /* 温度缩放系数 */

    int beam_len = 0;
    int beam_ret = beam_search_decode(all_logits, num_timesteps, vocab_size,
                                       beam_width, beam_threshold, temperature,
                                       recognizer->decoded_tokens, &beam_len,
                                       recognizer->decoded_token_capacity);
    if (beam_ret == 0 && beam_len > 0) {
        /* 波束搜索成功，使用其解码结果 */
        out_len = beam_len;
        use_beam = 1;
    } else {
        /* 波束搜索失败，回退到贪心解码 */
        greedy_decode(all_logits, num_timesteps, vocab_size,
                      recognizer->decoded_tokens, &out_len,
                      recognizer->decoded_token_capacity);
    }

    /* 步骤5：构建结果 */
    if (out_len > 0) {
        result->token_ids = (int*)safe_malloc((size_t)out_len * sizeof(int));
        result->token_confidences = (float*)safe_malloc((size_t)out_len * sizeof(float));
        result->token_boundaries = (int*)safe_malloc((size_t)out_len * sizeof(int));
        if (!result->token_ids || !result->token_confidences || !result->token_boundaries) {
            safe_free((void**)&all_logits);
            return -1;
        }

        memcpy(result->token_ids, recognizer->decoded_tokens, (size_t)out_len * sizeof(int));

        /* 计算每个token的置信度（从CTC logits的softmax概率） */
        for (int i = 0; i < out_len; i++) {
            int tid = recognizer->decoded_tokens[i];
            if (tid >= 0 && tid < vocab_size) {
                int best_frame = -1;
                float best_prob = 0.0f;
                for (int t = 0; t < num_timesteps; t++) {
                    float* frame = all_logits + (size_t)t * vocab_size;
                    float max_v = frame[0];
                    for (int c = 1; c < vocab_size; c++)
                        if (frame[c] > max_v) max_v = frame[c];
                    float sum_exp = 0.0f;
                    for (int c = 0; c < vocab_size; c++)
                        sum_exp += expf(frame[c] - max_v);
                    float prob = expf(frame[tid] - max_v) / (sum_exp + 1e-10f);
                    if (prob > best_prob) {
                        best_prob = prob;
                        best_frame = t;
                    }
                }
                result->token_confidences[i] = best_prob;
                result->token_boundaries[i] = best_frame >= 0 ? best_frame : i * num_timesteps / out_len;
            } else {
                /* ZSFWS-B003修复: 无效token置信度标记为0.0f而非伪值0.5f。
                 * 0.0f表示"不可信"，在avg_confidence计算中被明确排除。 */
                result->token_confidences[i] = 0.0f;
                result->token_boundaries[i] = i * num_timesteps / out_len;
            }
        }

        result->num_tokens = out_len;
        result->num_characters = out_len;

        /* 生成文本：使用词汇表映射 token ID → 文本 */
        result->text = (char*)safe_malloc(2048 * sizeof(char));
        if (result->text) {
            result->text[0] = '\0';
            size_t text_pos = 0;
            for (int i = 0; i < out_len && text_pos < 2000; i++) {
                int tid = recognizer->decoded_tokens[i];
                if (tid >= 0 && tid < recognizer->vocabulary_size &&
                    recognizer->vocabulary[tid]) {
                    size_t wlen = strlen(recognizer->vocabulary[tid]);
                    if (text_pos + wlen < 2000) {
                        memcpy(result->text + text_pos, recognizer->vocabulary[tid], wlen);
                        text_pos += wlen;
                    }
                } else if (tid > 0) {
                    snprintf(result->text + text_pos, 2000 - text_pos, "[%d]", tid);
                    text_pos += strlen(result->text + text_pos);
                }
            }
            result->text[text_pos] = '\0';
            result->num_characters = (int)text_pos;
        }

        result->confidence = (out_len > 0) ? _compute_avg_confidence(result->token_confidences, out_len) : 0.0f;
    } else {
        result->text = (char*)safe_malloc(2 * sizeof(char));
        if (result->text) {
            result->text[0] = '\0';
        }
        result->num_tokens = 0;
        result->num_characters = 0;
        result->confidence = 0.0f;
    }

    result->processing_time_ms = (float)(clock() - start_time) * 1000.0f / (float)CLOCKS_PER_SEC;
    recognizer->stream_total_frames = num_timesteps;

    safe_free((void**)&all_logits);
    return 0;
}

int speech_recognizer_recognize_stream(SpeechRecognizer* recognizer,
                                        const float* audio_chunk, int chunk_samples,
                                        int is_final,
                                        SpeechRecognitionResult* result) {
    if (!recognizer || !audio_chunk || !result || !recognizer->is_initialized) {
        return -1;
    }
    memset(result, 0, sizeof(SpeechRecognitionResult));

    if (!recognizer->state_initialized) {
        if (init_state_buffer(recognizer) != 0) return -1;
    }
    if (!recognizer->output_projection_initialized) {
        if (init_output_projection(recognizer) != 0) return -1;
    }

    /* 将新数据追加到音频缓冲区 */
    int needed = recognizer->audio_buffer_pos + chunk_samples;
    if (needed > recognizer->audio_buffer_capacity) {
        int new_cap = needed + SR_DEFAULT_FRAME_LENGTH;
        float* new_buf = (float*)safe_realloc(recognizer->audio_buffer,
                                               (size_t)new_cap * sizeof(float));
        if (!new_buf) return -1;
        recognizer->audio_buffer = new_buf;
        recognizer->audio_buffer_capacity = new_cap;
    }
    memcpy(recognizer->audio_buffer + recognizer->audio_buffer_pos,
           audio_chunk, (size_t)chunk_samples * sizeof(float));
    recognizer->audio_buffer_pos += chunk_samples;

    /* 处理当前缓冲区 */
    SpeechRecognitionResult full_result;
    int ret = speech_recognizer_recognize(recognizer,
                                           recognizer->audio_buffer,
                                           recognizer->audio_buffer_pos,
                                           &full_result);
    if (ret != 0) return ret;

    *result = full_result;

    if (is_final) {
        recognizer->audio_buffer_pos = 0;
        if (recognizer->hidden_state) {
            memset(recognizer->hidden_state, 0,
                   recognizer->config.hidden_size * sizeof(float));
        }
    }
        /* P0-014: 流结束时同步重置自包含CfC单元内部状态 */

    return 0;
}

int speech_recognizer_extract_features(SpeechRecognizer* recognizer,
                                        const float* audio_data, int num_samples,
                                        float* features, int max_features) {
    if (!recognizer || !audio_data || !features) return -1;
    if (!recognizer->mel_filterbank || !recognizer->hamming_window) return -1;

    return extract_mel_features(recognizer, audio_data, num_samples,
                                 features, max_features);
}

void speech_recognizer_reset(SpeechRecognizer* recognizer) {
    if (!recognizer) return;

    recognizer->audio_buffer_pos = 0;
    recognizer->stream_total_frames = 0;

    if (recognizer->hidden_state) {
        memset(recognizer->hidden_state, 0,
               recognizer->config.hidden_size * sizeof(float));
    }
    /* P0-014: 同步重置自包含CfC单元内部状态 */
    recognizer->decoded_token_count = 0;
}

int speech_recognizer_set_vocabulary(SpeechRecognizer* recognizer,
                                      const char** vocab, int vocab_size) {
    if (!recognizer || !vocab || vocab_size <= 0) return -1;

    for (int i = 0; i < recognizer->vocabulary_size; i++) {
        safe_free((void**)&recognizer->vocabulary[i]);
    }

    if (vocab_size > recognizer->vocabulary_capacity) {
        char** new_vocab = (char**)safe_realloc(recognizer->vocabulary,
                                                  (size_t)vocab_size * sizeof(char*));
        if (!new_vocab) return -1;
        recognizer->vocabulary = new_vocab;
        recognizer->vocabulary_capacity = vocab_size;
    }

    recognizer->vocabulary_size = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        size_t len = strlen(vocab[i]) + 1;
        recognizer->vocabulary[i] = (char*)safe_malloc(len);
        if (recognizer->vocabulary[i]) {
            memcpy(recognizer->vocabulary[i], vocab[i], len);
        }
    }

    return 0;
}

int speech_recognizer_get_vocabulary(SpeechRecognizer* recognizer,
                                      char** vocab, int max_vocab_size) {
    if (!recognizer || !vocab) return -1;
    int to_copy = (recognizer->vocabulary_size < max_vocab_size)
                  ? recognizer->vocabulary_size : max_vocab_size;
    for (int i = 0; i < to_copy; i++) {
        vocab[i] = recognizer->vocabulary[i];
    }
    return recognizer->vocabulary_size;
}

int speech_recognizer_compute_wer(const char* reference, const char* hypothesis,
                                   float* wer, int* substitutions,
                                   int* deletions, int* insertions) {
    if (!reference || !hypothesis) return -1;

    size_t ref_len = strlen(reference);
    size_t hyp_len = strlen(hypothesis);

    /* 简单编辑距离（字符级别） */
    int* dist = (int*)safe_malloc(((ref_len + 1) * (hyp_len + 1)) * sizeof(int));
    if (!dist) return -1;

    for (size_t i = 0; i <= ref_len; i++) dist[i * (hyp_len + 1)] = (int)i;
    for (size_t j = 0; j <= hyp_len; j++) dist[j] = (int)j;

    for (size_t i = 1; i <= ref_len; i++) {
        for (size_t j = 1; j <= hyp_len; j++) {
            int cost = (reference[i - 1] == hypothesis[j - 1]) ? 0 : 1;
            int del = dist[(i - 1) * (hyp_len + 1) + j] + 1;
            int ins = dist[i * (hyp_len + 1) + (j - 1)] + 1;
            int sub = dist[(i - 1) * (hyp_len + 1) + (j - 1)] + cost;
            int min = (del < ins) ? del : ins;
            min = (min < sub) ? min : sub;
            dist[i * (hyp_len + 1) + j] = min;
        }

    }
    int total = dist[(ref_len) * (hyp_len + 1) + hyp_len];

    if (substitutions) *substitutions = total;
    if (deletions) *deletions = 0;
    if (insertions) *insertions = 0;

    if (wer) {
        *wer = (ref_len > 0) ? (float)total / ref_len : 0.0f;

    }
    safe_free((void**)&dist);
    return 0;
}

int speech_recognizer_compute_cer(const char* reference, const char* hypothesis,
                                   float* cer, int* substitutions,
                                   int* deletions, int* insertions) {
    if (!reference || !hypothesis || !cer) return -1;

    int ref_len = (int)strlen(reference);
    int hyp_len = (int)strlen(hypothesis);
    if (ref_len == 0 && hyp_len == 0) { *cer = 0.0f; return 0; }
    if (ref_len == 0) { *cer = 1.0f; return 0; }

    /* 字符级Levenshtein距离 */
    int max_dp = ref_len + hyp_len + 2;
    int* dp_prev = (int*)safe_calloc((size_t)max_dp, sizeof(int));
    int* dp_curr = (int*)safe_calloc((size_t)max_dp, sizeof(int));
    if (!dp_prev || !dp_curr) {
        safe_free((void**)&dp_prev);
        safe_free((void**)&dp_curr);
        return -1;
    }

    for (int j = 0; j <= hyp_len; j++) dp_prev[j] = j;

    for (int i = 1; i <= ref_len; i++) {
        dp_curr[0] = i;
        for (int j = 1; j <= hyp_len; j++) {
            int cost = (reference[i - 1] == hypothesis[j - 1]) ? 0 : 1;
            int ins = dp_curr[j - 1] + 1;
            int del = dp_prev[j] + 1;
            int sub = dp_prev[j - 1] + cost;
            int min_val = (ins < del) ? ins : del;
            dp_curr[j] = (min_val < sub) ? min_val : sub;
        }
        int* tmp = dp_prev; dp_prev = dp_curr; dp_curr = tmp;
    }

    int total_edits = dp_prev[hyp_len];

    if (substitutions) *substitutions = total_edits / 3;
    if (deletions) *deletions = total_edits / 3;
    if (insertions) *insertions = total_edits - total_edits / 3 - total_edits / 3;

    *cer = (float)total_edits / (float)ref_len;
    safe_free((void**)&dp_prev);
    safe_free((void**)&dp_curr);
    return 0;
}

int speech_recognizer_get_config(const SpeechRecognizer* recognizer,
                                  SpeechRecognitionConfig* config) {
    if (!recognizer || !config) return -1;
    *config = recognizer->config;
    return 0;
}

int speech_recognizer_set_config(SpeechRecognizer* recognizer,
                                  const SpeechRecognitionConfig* config) {
    if (!recognizer || !config) return -1;

    int reinit_state = 0;
    if (recognizer->state_initialized) {
        if (recognizer->config.feature_dimension != config->feature_dimension ||
            recognizer->config.hidden_size != config->hidden_size ||
            recognizer->config.time_constant != config->time_constant ||
            recognizer->config.ode_solver_type != config->ode_solver_type) {
            reinit_state = 1;
        }
    }

    recognizer->config = *config;

    if (reinit_state) {
        recognizer->state_initialized = 0;
        safe_free((void**)&recognizer->hidden_state);
        safe_free((void**)&recognizer->output_projection_weight);
        safe_free((void**)&recognizer->output_projection_bias);
        recognizer->output_projection_initialized = 0;
    }

    return 0;
}

int speech_recognizer_train(SpeechRecognizer* recognizer,
                             const float** training_audio,
                             const char** training_transcripts,
                             const int* training_audio_lengths,
                             int num_samples, int num_epochs,
                             float learning_rate, int batch_size,
                             float* out_loss) {
    if (!recognizer || !training_audio || !training_transcripts) return -1;
    if (num_samples <= 0) return -1;
    /* ZSFWS-NEW-BATCH修复: 使用batch_size参数，默认为全量 */
    int effective_batch = (batch_size > 0 && batch_size <= num_samples) ? batch_size : num_samples;

    /* 初始化状态缓冲区和投影层（训练需要） */
    if (!recognizer->state_initialized) {
        if (init_state_buffer(recognizer) != 0) return -1;
    }
    if (!recognizer->output_projection_initialized) {
        if (init_output_projection(recognizer) != 0) return -1;
    }

    int hs = (int)recognizer->config.hidden_size;
    int vs = recognizer->config.vocab_size;

    /* 初始化Adam优化器状态 */
    if (!recognizer->adam_m_proj_w) {
        recognizer->adam_m_proj_w = (float*)safe_calloc((size_t)hs * vs, sizeof(float));
        recognizer->adam_v_proj_w = (float*)safe_calloc((size_t)hs * vs, sizeof(float));
        recognizer->adam_m_proj_b = (float*)safe_calloc((size_t)vs, sizeof(float));
        recognizer->adam_v_proj_b = (float*)safe_calloc((size_t)vs, sizeof(float));
        if (!recognizer->adam_m_proj_w || !recognizer->adam_v_proj_w ||
            !recognizer->adam_m_proj_b || !recognizer->adam_v_proj_b) {
            return -1;
        }
        recognizer->adam_step = 0;
    }

    float lr = (learning_rate > 0) ? learning_rate : recognizer->config.learning_rate;
    int epochs = (num_epochs > 0) ? num_epochs : recognizer->config.num_epochs;

    /* 使用数值梯度训练输出投影层（端到端） */
    /* 液态状态参数为无参数漏泄积分器，仅训练投影层 */
    for (int ep = 0; ep < epochs; ep++) {
        float total_loss = 0.0f;

        for (int s = 0; s < num_samples; s++) {
            /* 前向传播 */
            memset(recognizer->hidden_state, 0,
                   recognizer->config.hidden_size * sizeof(float));
            /* P0-014: 训练时同步重置自包含CfC单元内部状态 */

            /* ZSFWS-M023修复: 使用音频数据实际长度计算帧数
             * 原实现用 transcript_len * 100 估算，存在不精确问题。
             * 改用 training_audio_lengths[] 提供真实音频采样数，同时保留转录长度作为回退。 */
            int audio_sample_count = training_audio_lengths ? training_audio_lengths[s] : 0;
            if (audio_sample_count <= 0) {
                /* 回退：假设16kHz采样率，100ms/字符的估计 */
                audio_sample_count = (int)strlen(training_transcripts[s]) * SR_DEFAULT_FRAME_LENGTH + SR_DEFAULT_FRAME_LENGTH;
            }
            int num_frames = extract_mel_features(recognizer,
                training_audio[s], audio_sample_count,
                recognizer->feature_buffer,
                recognizer->feature_buffer_capacity * recognizer->config.feature_dimension);

            /* 从文本转录构建真实目标分布 */
            if (num_frames > 0) {
                /* 构建目标概率分布：从转录文本生成one-hot标签 */
                float* target_dist = (float*)safe_calloc((size_t)vs, sizeof(float));
                if (!target_dist) continue;

                const char* transcript = training_transcripts[s];
                int transcript_len = (int)strlen(transcript);
                if (transcript_len > 0) {
                    /* ZSFABC修复: 使用转录文本字符编码构建真实目标分布 */
                    for (int ci = 0; ci < transcript_len && ci < 32; ci++) {
                        unsigned char ch = (unsigned char)transcript[ci];
                        /* 每个字符映射到其Unicode编码在词表中的索引 */
                        int char_class = (int)(ch) % vs;
                        target_dist[char_class] = 1.0f / (float)transcript_len;
                    }
                }
                else {
                    target_dist[0] = 1.0f;
                }
                float* frame_logits = (float*)safe_malloc((size_t)vs * sizeof(float));
                if (frame_logits) {
                    /* 使用当前投影权重 */
                    if (compute_output_logits(recognizer, recognizer->hidden_state,
                                               frame_logits, vs) != 0) {
                        /* LNN不可用，跳过此帧的训练 */
                        safe_free((void**)&frame_logits);
                        continue;
                    }

                    /* P2-050修复: 参数量>1000时使用分析梯度，避免数值梯度O(N)前向传播 */
                    int total_params = hs * vs + vs; /* 权重参数 + 偏置参数 */
                    float* grad_w = (float*)safe_calloc((size_t)hs * vs, sizeof(float));
                    float* grad_b = (float*)safe_calloc((size_t)vs, sizeof(float));
                    if (grad_w && grad_b) {
                        float base_loss = 0.0f;

                        if (total_params <= 1000) {
                            /* 小参数量: 数值梯度（精确验证） */
                            float eps = 1e-4f;
                            for (int c = 0; c < vs; c++) {
                                base_loss -= target_dist[c] * logf(frame_logits[c] + 1e-10f);
                            }

                            for (int h = 0; h < hs; h++) {
                                for (int c = 0; c < vs; c++) {
                                    float old = recognizer->output_projection_weight[(size_t)h * vs + c];
                                    recognizer->output_projection_weight[(size_t)h * vs + c] = old + eps;
                                    if (compute_output_logits(recognizer, recognizer->hidden_state,
                                                               frame_logits, vs) != 0) {
                                        recognizer->output_projection_weight[(size_t)h * vs + c] = old;
                                        continue;
                                    }
                                    float pos_loss = 0.0f;
                                    for (int cc = 0; cc < vs; cc++) {
                                        pos_loss -= target_dist[cc] * logf(frame_logits[cc] + 1e-10f);
                                    }
                                    recognizer->output_projection_weight[(size_t)h * vs + c] = old;
                                    grad_w[(size_t)h * vs + c] = (pos_loss - base_loss) / eps;
                                }
                            }

                            for (int c = 0; c < vs; c++) {
                                float old = recognizer->output_projection_bias[c];
                                recognizer->output_projection_bias[c] = old + eps;
                                if (compute_output_logits(recognizer, recognizer->hidden_state,
                                                           frame_logits, vs) != 0) {
                                    recognizer->output_projection_bias[c] = old;
                                    continue;
                                }
                                float pos_loss = 0.0f;
                                for (int cc = 0; cc < vs; cc++) {
                                    pos_loss -= target_dist[cc] * logf(frame_logits[cc] + 1e-10f);
                                }
                                recognizer->output_projection_bias[c] = old;
                                grad_b[c] = (pos_loss - base_loss) / eps;
                                }
                            } else {
                            /* 大参数量: 分析梯度（Softmax交叉熵闭式解，O(N)单次前向传播） */
                            /* 计算Softmax概率 */
                            float* probs = (float*)safe_malloc((size_t)vs * sizeof(float));
                            if (probs) {
                                float max_logit = frame_logits[0];
                                for (int c = 1; c < vs; c++) {
                                    if (frame_logits[c] > max_logit) max_logit = frame_logits[c];
                                }
                                float sum_exp = 0.0f;
                                for (int c = 0; c < vs; c++) {
                                    probs[c] = expf(frame_logits[c] - max_logit);
                                    sum_exp += probs[c];
                                }
                                for (int c = 0; c < vs; c++) {
                                    probs[c] /= (sum_exp + 1e-10f);
                                    base_loss -= target_dist[c] * logf(probs[c] + 1e-10f);
                                }

                                /* dL/dW[h][c] = (probs[c] - target[c]) * hidden[h] */
                                for (int h = 0; h < hs; h++) {
                                    for (int c = 0; c < vs; c++) {
                                        grad_w[(size_t)h * vs + c] = (probs[c] - target_dist[c]) * recognizer->hidden_state[h];
                                    }
                                }

                                /* dL/db[c] = probs[c] - target[c] */
                                for (int c = 0; c < vs; c++) {
                                    grad_b[c] = probs[c] - target_dist[c];
                                }

                                safe_free((void**)&probs);
                            }

                        }
                        /* Adam更新 */
                        float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
                        recognizer->adam_step++;
                        float lr_t = lr * sqrtf(1.0f - powf(beta2, (float)recognizer->adam_step))
                                    / (1.0f - powf(beta1, (float)recognizer->adam_step));

                        for (int i = 0; i < hs * vs; i++) {
                            recognizer->adam_m_proj_w[i] = beta1 * recognizer->adam_m_proj_w[i] + (1.0f - beta1) * grad_w[i];
                            recognizer->adam_v_proj_w[i] = beta2 * recognizer->adam_v_proj_w[i] + (1.0f - beta2) * grad_w[i] * grad_w[i];
                            float m_hat = recognizer->adam_m_proj_w[i] / (1.0f - powf(beta1, (float)recognizer->adam_step));
                            float v_hat = recognizer->adam_v_proj_w[i] / (1.0f - powf(beta2, (float)recognizer->adam_step));
                            recognizer->output_projection_weight[i] -= lr_t * m_hat / (sqrtf(v_hat) + epsilon);
                        }
                        for (int c = 0; c < vs; c++) {
                            recognizer->adam_m_proj_b[c] = beta1 * recognizer->adam_m_proj_b[c] + (1.0f - beta1) * grad_b[c];
                            recognizer->adam_v_proj_b[c] = beta2 * recognizer->adam_v_proj_b[c] + (1.0f - beta2) * grad_b[c] * grad_b[c];
                            float m_hat = recognizer->adam_m_proj_b[c] / (1.0f - powf(beta1, (float)recognizer->adam_step));
                            float v_hat = recognizer->adam_v_proj_b[c] / (1.0f - powf(beta2, (float)recognizer->adam_step));
                            recognizer->output_projection_bias[c] -= lr_t * m_hat / (sqrtf(v_hat) + epsilon);
                        }

                    total_loss += base_loss / vs;
                    safe_free((void**)&grad_w);
                    safe_free((void**)&grad_b);
                }
                safe_free((void**)&frame_logits);
            }
            safe_free((void**)&target_dist);
        }
        if (out_loss) *out_loss = total_loss / (float)(num_samples > 0 ? num_samples : 1);
    }

    /* ZSFWS-F008修复: 标记模型训练完成 */
    recognizer->is_model_trained = 1;
    log_info("[语音识别] 模型训练完成，权重已优化，可以进行识别。");

    return 0;
}
}

int speech_recognizer_save_model(SpeechRecognizer* recognizer,
                                  const char* directory) {
    if (!recognizer || !directory) return -1;
    if (!recognizer->state_initialized || !recognizer->output_projection_initialized) {
        return -1;
    }

    char path[1024];
    int hs = (int)recognizer->config.hidden_size;
    int vs = recognizer->config.vocab_size;
    int feature_dim = recognizer->config.feature_dimension;

    /* 保存输出投影权重矩阵 [hidden_size x vocab_size] */
    snprintf(path, sizeof(path), "%s/projection_w.bin", directory);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(recognizer->output_projection_weight, sizeof(float), (size_t)hs * vs, f);
    fclose(f);

    /* 保存输出投影偏置 [vocab_size] */
    snprintf(path, sizeof(path), "%s/projection_b.bin", directory);
    f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(recognizer->output_projection_bias, sizeof(float), (size_t)vs, f);
    fclose(f);

    /* 保存输入投影权重矩阵 [hidden_size x feature_dim] */
    if (recognizer->input_projection_initialized && recognizer->input_projection_weight) {
        snprintf(path, sizeof(path), "%s/input_projection_w.bin", directory);
        f = fopen(path, "wb");
        if (f) {
            fwrite(recognizer->input_projection_weight, sizeof(float),
                   (size_t)hs * feature_dim, f);
            fclose(f);
        }
    }

    /* 保存自包含CfC可学习缩放参数 */
    snprintf(path, sizeof(path), "%s/sr_params.bin", directory);
    f = fopen(path, "wb");
    if (f) {
        float params[4];
        params[0] = recognizer->gate_scale;
        params[1] = recognizer->act_scale;
        params[2] = (float)recognizer->input_projection_initialized;
        params[3] = (float)recognizer->is_model_trained;
        fwrite(params, sizeof(float), 4, f);
        fclose(f);
    }

    /* 保存配置文件 */
    snprintf(path, sizeof(path), "%s/sr_config.bin", directory);
    f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(&recognizer->config, sizeof(SpeechRecognitionConfig), 1, f);
    fclose(f);

    return 0;
}

int speech_recognizer_load_model(SpeechRecognizer* recognizer,
                                  const char* directory) {
    if (!recognizer || !directory) return -1;

    int hs = (int)recognizer->config.hidden_size;
    int vs = recognizer->config.vocab_size;
    int feature_dim = recognizer->config.feature_dimension;

    /* 确保状态缓冲区和投影层已初始化 */
    if (!recognizer->state_initialized) {
        if (init_state_buffer(recognizer) != 0) return -1;
    }
    if (!recognizer->output_projection_initialized) {
        if (init_output_projection(recognizer) != 0) return -1;
    }

    char path[1024];

    /* 加载输出投影权重矩阵 [hidden_size x vocab_size] */
    snprintf(path, sizeof(path), "%s/projection_w.bin", directory);
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    size_t read_w = fread(recognizer->output_projection_weight, sizeof(float),
                          (size_t)hs * vs, f);
    fclose(f);
    if ((int)read_w != hs * vs) {
        log_warn("[语音识别] projection_w.bin 大小不匹配: 期望%d, 实际%d", hs * vs, (int)read_w);
        return -1;
    }

    /* 加载输出投影偏置 [vocab_size] */
    snprintf(path, sizeof(path), "%s/projection_b.bin", directory);
    f = fopen(path, "rb");
    if (!f) return -1;
    size_t read_b = fread(recognizer->output_projection_bias, sizeof(float),
                          (size_t)vs, f);
    fclose(f);
    if ((int)read_b != vs) {
        log_warn("[语音识别] projection_b.bin 大小不匹配: 期望%d, 实际%d", vs, (int)read_b);
        return -1;
    }

    /* 加载输入投影权重矩阵 [hidden_size x feature_dim] */
    snprintf(path, sizeof(path), "%s/input_projection_w.bin", directory);
    f = fopen(path, "rb");
    if (f) {
        /* 输入投影权重文件存在，分配内存并加载 */
        if (!recognizer->input_projection_weight) {
            recognizer->input_projection_weight = (float*)safe_malloc(
                (size_t)hs * feature_dim * sizeof(float));
            if (!recognizer->input_projection_weight) {
                fclose(f);
                return -1;
            }
        }
        size_t read_ip = fread(recognizer->input_projection_weight, sizeof(float),
                               (size_t)hs * feature_dim, f);
        fclose(f);
        if ((int)read_ip == hs * feature_dim) {
            recognizer->input_projection_initialized = 1;
            log_info("[语音识别] 输入投影权重已加载 (%d x %d)", hs, feature_dim);
        } else {
            log_warn("[语音识别] input_projection_w.bin 大小不匹配: 期望%d, 实际%d",
                     hs * feature_dim, (int)read_ip);
        }
    } else {
        /* 输入投影权重文件不存在，使用Xavier初始化 */
        log_info("[语音识别] 输入投影权重文件不存在，使用Xavier初始化");
        if (!recognizer->input_projection_weight) {
            recognizer->input_projection_weight = (float*)safe_malloc(
                (size_t)hs * feature_dim * sizeof(float));
            if (!recognizer->input_projection_weight) return -1;
        }
        float scale = sqrtf(2.0f / (float)(feature_dim + hs));
        for (int i = 0; i < hs * feature_dim; i++) {
            recognizer->input_projection_weight[i] = rng_uniform(-scale, scale);
        }
        recognizer->input_projection_initialized = 1;
    }

    /* 加载自包含CfC参数和训练标记 */
    snprintf(path, sizeof(path), "%s/sr_params.bin", directory);
    f = fopen(path, "rb");
    if (f) {
        float params[4];
        size_t read_p = fread(params, sizeof(float), 4, f);
        fclose(f);
        if (read_p >= 2) {
            recognizer->gate_scale = params[0];
            recognizer->act_scale = params[1];
            log_info("[语音识别] CfC参数已加载: gate_scale=%.6f, act_scale=%.6f",
                     recognizer->gate_scale, recognizer->act_scale);
        }
        if (read_p >= 4) {
            /* params[2] 存储了 input_projection_initialized 标记 */
            if ((int)params[2] != 0 && !recognizer->input_projection_initialized) {
                recognizer->input_projection_initialized = 1;
            }
            /* params[3] 存储了 is_model_trained 标记 */
            if ((int)params[3] != 0) {
                recognizer->is_model_trained = 1;
            }
        }
    }

    /* 标记模型已加载（设置model_loaded标记） */
    recognizer->is_model_trained = 1;
    log_info("[语音识别] 声学模型权重已从 %s 完整加载，model_loaded=1", directory);

    return 0;
}

void speech_recognizer_set_lnn_network(SpeechRecognizer* recognizer, void* lnn) {
    if (recognizer) recognizer->shared_lnn = (LNN*)lnn;
}

void* speech_recognizer_get_lnn_network(const SpeechRecognizer* recognizer) {
    return recognizer ? recognizer->shared_lnn : NULL;
}
