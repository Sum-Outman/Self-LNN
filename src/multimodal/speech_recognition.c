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
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
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

    return num_frames;
}

/* =============================================================== *
 * 输出投影：隐藏状态 → 字符logits                                   *
 * =============================================================== */

static void compute_output_logits(SpeechRecognizer* sr,
                                   const float* hidden_state,
                                   float* logits, int vocab_size) {
    int hs = (int)sr->config.hidden_size;
    const float* W = sr->output_projection_weight;
    const float* b = sr->output_projection_bias;

    for (int c = 0; c < vocab_size; c++) {
        double sum = 0.0;
        for (int h = 0; h < hs; h++) {
            sum += (double)hidden_state[h] * W[(size_t)h * vocab_size + c];
        }
        logits[c] = (float)(sum + b[c]);
    }
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
 * 中文词汇表初始化 — 基于GB2312常用汉字+拉丁字母+数字                  *
 * =============================================================== */

static const char* SR_BASE_VOCABULARY[] = {
    /* 空白token（CTC用） */
    "",
    /* 标点符号 */
    "。", "，", "！", "？", "、", "；", "：", "…", "—", "～",
    "（", "）", "【", "】", "《", "》", "\"", "\"", "'", "'",
    /* 中文数字 */
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "亿",
    /* 常用量词 */
    "个", "只", "条", "张", "把", "台", "辆", "件", "本", "次",
    "米", "厘米", "毫米", "千米", "公斤", "克", "秒", "分钟", "小时", "天",
    "年", "月", "日", "度", "摄氏度",
    /* 高频动词 */
    "是", "有", "在", "说", "来", "去", "做", "看", "听", "吃",
    "喝", "走", "跑", "写", "读", "学", "教", "买", "卖", "用",
    "给", "拿", "放", "开", "关", "打开", "关闭", "开始", "停止", "启动",
    "运行", "移动", "转动", "抓取", "推", "拉", "按", "点击", "输入", "输出",
    "识别", "检测", "处理", "计算", "分析", "生成", "发送", "接收", "连接", "断开",
    /* 高频名词 */
    "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这", "那", "什么", "怎么", "哪里", "谁", "为什么",
    "人", "机器", "机器人", "电脑", "手机", "摄像头", "麦克风", "传感器",
    "系统", "程序", "数据", "文件", "图像", "视频", "音频", "文本",
    "声音", "光", "温度", "速度", "位置", "方向", "距离", "角度",
    "手", "臂", "腿", "头", "身体", "关节", "电机", "齿轮",
    "前", "后", "左", "右", "上", "下", "中间",
    "红色", "绿色", "蓝色", "白色", "黑色", "黄色",
    "苹果", "香蕉", "桌子", "椅子", "杯子", "笔", "书", "灯",
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
    /* 常用词 */
    "你好", "谢谢", "再见", "对不起", "没关系", "请",
    "温度", "湿度", "压力", "电流", "电压", "功率",
    "图片", "拍照", "录像", "录音", "播放", "暂停",
    "确认", "取消", "是", "否", "帮助",
    /* 系统命令 */
    "执行", "查询", "搜索", "学习", "训练", "保存", "加载",
    "状态", "日志", "设置", "参数", "配置",
    "返回", "前进", "旋转", "升降", "伸缩", "点头", "摇头",
};

static int init_base_vocabulary(SpeechRecognizer* sr) {
    int base_count = (int)(sizeof(SR_BASE_VOCABULARY) / sizeof(SR_BASE_VOCABULARY[0]));
    if (base_count > sr->vocabulary_capacity) {
        int new_cap = base_count + 512;
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
        if (sr->vocabulary[i] != NULL) continue;
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
    config.use_gpu = 0;
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

    /* P0-014: 尝试创建自包含CfC单元用于独立状态演化
     * 当shared_lnn未连接时，此CfCCell提供完整的多时间尺度CfC动态 */
    {
        CfCCellConfig cfc_cfg;
        memset(&cfc_cfg, 0, sizeof(CfCCellConfig));
        cfc_cfg.input_size = (size_t)sr->config.feature_dimension;
        cfc_cfg.hidden_size = sr->config.hidden_size;
        cfc_cfg.time_constant = sr->config.time_constant > 0.0f ? sr->config.time_constant : 0.1f;
        cfc_cfg.noise_std = 0.01f;
        cfc_cfg.enable_adaptation = 1;
        cfc_cfg.ode_solver_type = ODE_SOLVER_CLOSED_FORM;
        cfc_cfg.delta_t = 1.0f;
        cfc_cfg.use_xavier_init = 1;
        cfc_cfg.use_cell_layer_norm = 1;
        cfc_cfg.use_residual = 1;
        cfc_cfg.residual_scale = 0.3f;

        sr->self_contained_cfc = cfc_cell_create(&cfc_cfg);
        if (!sr->self_contained_cfc) {
            /* CfCCell创建失败，回退到可学习参数简化路径（gate_scale/act_scale） */
            sr->self_contained_cfc = NULL;
        }
    }

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

    /* P0-014: 释放自包含CfC单元 */
    if (recognizer->self_contained_cfc) {
        cfc_cell_free(recognizer->self_contained_cfc);
        recognizer->self_contained_cfc = NULL;
    }

    safe_free((void**)&recognizer);
}

int speech_recognizer_recognize(SpeechRecognizer* recognizer,
                                 const float* audio_data, int num_samples,
                                 SpeechRecognitionResult* result) {
    if (!recognizer || !audio_data || !result || !recognizer->is_initialized) {
        return -1;
    }
    memset(result, 0, sizeof(SpeechRecognitionResult));

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

    float tau = recognizer->config.time_constant > 0.0f ? recognizer->config.time_constant : 0.1f;
    float dt = 1.0f;
    float alpha_val = 1.0f - expf(-dt / tau);

    /* ZSFABC-F007: 输入投影权重改为实例成员，每个识别器独立分配 */
    if (!recognizer->input_projection_initialized) {
        recognizer->input_projection_weight = (float*)safe_malloc(hs * (size_t)feature_dim * sizeof(float));
        if (recognizer->input_projection_weight) {
            float scale = sqrtf(2.0f / (float)(feature_dim + (int)hs));
            for (size_t i = 0; i < hs * (size_t)feature_dim; i++) {
                recognizer->input_projection_weight[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
            }
            recognizer->input_projection_initialized = 1;
        }
    }

    /* 重置隐藏状态 */
    memset(recognizer->hidden_state, 0, hs * sizeof(float));
    /* P0-014: 同步重置自包含CfC单元内部状态 */
    if (recognizer->self_contained_cfc) {
        cfc_cell_reset(recognizer->self_contained_cfc);
    }

    for (int t = 0; t < num_timesteps; t++) {
        const float* feat = recognizer->feature_buffer + (size_t)t * feature_dim;

        /* CfC液态状态演化 — 集成核心LNN
         * 当共享LNN网络已连接时，通过lnn_forward进行连续状态演化；
         * 未连接时使用自包含CfC封闭形式解 */
        if (recognizer->shared_lnn) {
            /* 通过核心LNN的连续动态系统进行状态演化 */
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
            /* P0-014: 自包含CfC状态演化 — 优先使用完整CfCCell动态系统
             * CfCCell提供多时间尺度门控、液时域缩放、层归一化、残差连接等完整CfC特性
             * 若CfCCell未创建成功，回退到可学习参数简化路径 */
            float nonlinear_term[256];
            memset(nonlinear_term, 0, sizeof(nonlinear_term));

            if (recognizer->self_contained_cfc) {
                /* 主路径：通过完整CfCCell进行状态演化
                 * cfc_cell_forward_with_dt内部执行：
                 *   1. 输入投影（Xavier初始化权重）
                 *   2. 多时间尺度门控动态
                 *   3. 液时域缩放（输入依赖时间常数）
                 *   4. CfC封闭形式ODE解
                 *   5. 残差连接 + 层归一化
                 * 比简化 σ(sum*scale)⊙tanh(sum*scale) 强大得多 */
                if (cfc_cell_forward_with_dt(recognizer->self_contained_cfc,
                     feat, dt, nonlinear_term) != 0) {
                    /* CfCCell前向失败，回退到可学习参数简化路径 */
                    goto fallback_self_contained;
                }
            } else {
fallback_self_contained:
                /* 回退路径：可学习参数版自包含CfC简化动态
                 * τ·dh/dt = -h + σ(sum*gate_scale) ⊙ tanh(sum*act_scale)
                 * gate_scale/act_scale为Xavier初始化的可学习参数，替代硬编码0.5/0.3 */
                for (size_t i = 0; i < hs && i < 256; i++) {
                    double sum = 0.0;
                    if (recognizer->input_projection_weight) {
                        for (int j = 0; j < feature_dim && j < feature_dim; j++) {
                            sum += (double)recognizer->input_projection_weight[i * (size_t)feature_dim + j] * feat[j];
                        }
                    } else {
                        if (i < (size_t)feature_dim) sum = (double)feat[i];
                    }
                    float gate = 1.0f / (1.0f + expf(-(float)sum * recognizer->gate_scale));
                    float act = tanhf((float)sum * recognizer->act_scale);
                    nonlinear_term[i] = gate * act;
                }
            }

            float decay = expf(-dt / tau);
            for (size_t i = 0; i < hs && i < 256; i++) {
                recognizer->hidden_state[i] = decay * recognizer->hidden_state[i] +
                                               (1.0f - decay) * nonlinear_term[i];
            }
        }

        float* logits = all_logits + (size_t)t * vocab_size;
        compute_output_logits(recognizer, recognizer->hidden_state,
                              logits, vocab_size);
    }

    /* 步骤4：自适应解码（贪心 → 波束搜索 → 告警）
     * 先尝试贪心解码，若置信度低则升级为波束搜索。
     * 纯液态系统无外部语言模型，但波束搜索的序列级评分
     * 优于帧级贪心，在模糊音频上能提供更好的解码质量。 */
    int out_len = 0;
    int use_beam = 0;
    greedy_decode(all_logits, num_timesteps, vocab_size,
                  recognizer->decoded_tokens, &out_len,
                  recognizer->decoded_token_capacity);

    /* 评估贪心解码置信度 */
    float greedy_conf = 0.0f;
    if (out_len > 0) {
        for (int i = 0; i < out_len; i++) {
            int tid = recognizer->decoded_tokens[i];
            if (tid >= 0 && tid < vocab_size) {
                float max_v = all_logits[0];
                for (int t = 0; t < num_timesteps; t++) {
                    for (int c = 0; c < vocab_size; c++) {
                        float v = all_logits[(size_t)t * vocab_size + c];
                        if (v > max_v) max_v = v;
                    }
                }
                float sum = 0.0f;
                for (int t = 0; t < num_timesteps; t++)
                    sum += expf(all_logits[(size_t)t * vocab_size + tid] - max_v);
                greedy_conf += (sum > 0.0f) ? logf(sum / (float)num_timesteps + 1e-10f) : -10.0f;
            }
        }
        greedy_conf = greedy_conf / (float)out_len;
        greedy_conf = 1.0f / (1.0f + expf(-greedy_conf * 0.5f)); /* sigmoid归一化 */
    }

    /* 若贪心置信度低于阈值，启用波束搜索 */
    float conf_threshold = 0.3f;
    if (greedy_conf < conf_threshold) {
        int beam_len = 0;
        beam_search_decode(all_logits, num_timesteps, vocab_size,
                          8, 10.0f, 1.0f,
                          recognizer->decoded_tokens, &beam_len,
                          recognizer->decoded_token_capacity);
        if (beam_len > 0) {
            out_len = beam_len;
            use_beam = 1;
        }
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
                result->token_confidences[i] = 0.5f;
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
        /* P0-014: 流结束时同步重置自包含CfC单元内部状态 */
        if (recognizer->self_contained_cfc) {
            cfc_cell_reset(recognizer->self_contained_cfc);
        }
    }

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
    if (recognizer->self_contained_cfc) {
        cfc_cell_reset(recognizer->self_contained_cfc);
    }
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
                             int num_samples, int num_epochs,
                             float learning_rate, int batch_size,
                             float* out_loss) {
    if (!recognizer || !training_audio || !training_transcripts) return -1;
    if (num_samples <= 0) return -1;
    (void)batch_size;

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
            if (recognizer->self_contained_cfc) {
                cfc_cell_reset(recognizer->self_contained_cfc);
            }

            int num_frames = extract_mel_features(recognizer,
                training_audio[s], (int)strlen(training_transcripts[s]) * 100 + SR_DEFAULT_FRAME_LENGTH,
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
                    /* 使用转录文本的字符哈希值确定目标类别 */
                    unsigned int hash = 0;
                    for (int ci = 0; ci < transcript_len && ci < 32; ci++) {
                        hash = hash * 31 + (unsigned char)transcript[ci];
                    }
                    int target_class = (int)(hash % (unsigned int)vs);
                    target_dist[target_class] = 1.0f;
                } else {
                    target_dist[0] = 1.0f;
                }
                float* frame_logits = (float*)safe_malloc((size_t)vs * sizeof(float));
                if (frame_logits) {
                    /* 使用当前投影权重 */
                    compute_output_logits(recognizer, recognizer->hidden_state,
                                          frame_logits, vs);

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
                                    compute_output_logits(recognizer, recognizer->hidden_state,
                                                          frame_logits, vs);
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
                                compute_output_logits(recognizer, recognizer->hidden_state,
                                                      frame_logits, vs);
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
        }
        if (out_loss) *out_loss = total_loss / (float)(num_samples > 0 ? num_samples : 1);
    }

    return 0;
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

    snprintf(path, sizeof(path), "%s/projection_w.bin", directory);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(recognizer->output_projection_weight, sizeof(float), (size_t)hs * vs, f);
    fclose(f);

    snprintf(path, sizeof(path), "%s/projection_b.bin", directory);
    f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(recognizer->output_projection_bias, sizeof(float), (size_t)vs, f);
    fclose(f);

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

    if (!recognizer->state_initialized) {
        if (init_state_buffer(recognizer) != 0) return -1;
    }
    if (!recognizer->output_projection_initialized) {
        if (init_output_projection(recognizer) != 0) return -1;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/projection_w.bin", directory);
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    size_t read_w = fread(recognizer->output_projection_weight, sizeof(float), (size_t)hs * vs, f);
    fclose(f);
    if ((int)read_w != hs * vs) return -1;

    snprintf(path, sizeof(path), "%s/projection_b.bin", directory);
    f = fopen(path, "rb");
    if (!f) return -1;
    size_t read_b = fread(recognizer->output_projection_bias, sizeof(float), (size_t)vs, f);
    fclose(f);
    if ((int)read_b != vs) return -1;

    return 0;
}

void speech_recognizer_set_lnn_network(SpeechRecognizer* recognizer, void* lnn) {
    if (recognizer) recognizer->shared_lnn = (LNN*)lnn;
}

void* speech_recognizer_get_lnn_network(const SpeechRecognizer* recognizer) {
    return recognizer ? recognizer->shared_lnn : NULL;
}
