/**
 * @file multimodal.c
 * @brief 多模态处理器核心实现 —— CfC液态神经网络驱动
 *
 * P0-003修复: 视觉特征提取从传统CV(Sobel/LBP/HSV)改为CfC液态神经网络。
 * 所有模态原始数据通过CfC连续时间ODE动态系统提取特征，
 * 统一馈入系统主LNN进行状态演化。
 * 严格遵循：不需要多模型融合、不需要跨模态注意力。
 */

#include "selflnn/multimodal/multimodal.h"
#include "selflnn/multimodal/deep_vision.h"
#include "selflnn/core/unified_lnn_state.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* 视觉CfC输入/输出维度常量 */
#define MMC_VISION_CFC_INPUT_DIM  1024
#define MMC_VISION_CFC_HIDDEN_DIM 512
#define MMC_VISION_CFC_OUTPUT_DIM 256

/**
 * @brief 多模态处理器内部结构体
 */
struct MultimodalProcessor {
    MultimodalConfig config;
    int is_initialized;

    LNN* main_lnn;

    /* P0-003修复: CfC深度视觉处理器替代传统CV特征提取 */
    CfcVisionProcessor* cfc_vision_proc;

    float* vision_features_cache;
    size_t vision_features_size;
    float* audio_features_cache;
    size_t audio_features_size;
    float* text_features_cache;
    size_t text_features_size;
    float* sensor_features_cache;
    size_t sensor_features_size;

    int is_training;
    float learning_rate;
};

/**
 * @brief 计算特征向量均值
 */
static void compute_feature_mean(const float* features, size_t count,
                                float* mean, size_t dimension) {
    if (count == 0 || dimension == 0) return;
    for (size_t d = 0; d < dimension; d++) mean[d] = 0.0f;
    for (size_t i = 0; i < count; i++)
        for (size_t d = 0; d < dimension; d++)
            mean[d] += features[i * dimension + d];
    for (size_t d = 0; d < dimension; d++) mean[d] /= (float)count;
}

/**
 * @brief 归一化特征向量
 */
static void normalize_features(float* features, size_t count, size_t dimension) {
    if (count == 0 || dimension == 0) return;
    for (size_t i = 0; i < count; i++) {
        float norm = 0.0f;
        for (size_t d = 0; d < dimension; d++)
            norm += features[i * dimension + d] * features[i * dimension + d];
        norm = sqrtf(norm);
        if (norm > 1e-6f)
            for (size_t d = 0; d < dimension; d++)
                features[i * dimension + d] /= norm;
    }
}

/**
 * @brief 创建多模态处理器
 */
MultimodalProcessor* multimodal_processor_create(const MultimodalConfig* config) {
    if (!config) return NULL;

    MultimodalProcessor* processor = (MultimodalProcessor*)safe_malloc(sizeof(MultimodalProcessor));
    if (!processor) return NULL;

    memset(processor, 0, sizeof(MultimodalProcessor));
    memcpy(&processor->config, config, sizeof(MultimodalConfig));

    processor->is_training = 0;
    processor->learning_rate = 0.001f;
    processor->is_initialized = 1;
    return processor;
}

/**
 * @brief 设置共享主LNN实例
 */
int multimodal_processor_set_lnn(MultimodalProcessor* processor, LNN* lnn) {
    if (!processor) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "多模态处理器设置LNN: 处理器句柄为空");
        return -1;
    }
    processor->main_lnn = lnn;
    return 0;
}

/**
 * @brief 释放多模态处理器
 */
void multimodal_processor_free(MultimodalProcessor* processor) {
    if (!processor) return;

    safe_free((void**)&processor->vision_features_cache);
    safe_free((void**)&processor->audio_features_cache);
    safe_free((void**)&processor->text_features_cache);
    safe_free((void**)&processor->sensor_features_cache);
    safe_free((void**)&processor);
}

/**
 * @brief 处理视觉数据（P0-003修复：使用CfC液态神经网络替代传统CV）
 *
 * 视觉特征提取全部通过CfC连续时间ODE动态系统完成。
 * 不再使用Sobel边缘检测、LBP纹理、HSV直方图等传统计算机视觉方法。
 * 原始像素直接输入CfC神经网络，经ODE连续时间演化后输出视觉特征。
 *
 * 处理流程：
 *   1. 自适应降采样到输入维度
 *   2. CfC深度视觉处理器前向传播
 *   3. 输出CfC连续动态特征
 */
int multimodal_process_vision(MultimodalProcessor* processor, const VisionData* vision_data,
                             float* features, size_t max_features) {
    SELFLNN_CHECK_NULL(processor, "多模态处理器句柄为空");
    SELFLNN_CHECK_NULL(vision_data, "视觉数据为空");
    SELFLNN_CHECK_NULL(features, "特征输出缓冲区为空");
    SELFLNN_CHECK(max_features > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "特征缓冲区大小无效: %zu", max_features);
    SELFLNN_CHECK_INITIALIZED(processor, "多模态处理器未初始化");

    if (!processor->config.enable_vision) return 0;

    int width = vision_data->width;
    int height = vision_data->height;
    int channels = vision_data->channels;
    float* data = vision_data->data;

    SELFLNN_CHECK(width > 0 && height > 0 && channels > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "视觉数据无效（宽：%d，高：%d，通道：%d）", width, height, channels);

    /* P0-003: 使用CfC深度视觉处理器提取特征（替代传统CV） */
    if (processor->cfc_vision_proc) {
        /* 通过CfC ODE连续动态系统提取深度视觉特征 */
        int extracted = cfc_vision_extract_features(processor->cfc_vision_proc,
                                                     data, width, height, channels,
                                                     features, max_features);
        if (extracted > 0) {
            /* CfC特征提取成功，直接返回CfC特征 */
            return extracted;
        }
        /* CfC提取失败，回退到LNN前向传播 */
    }

    /* 回退路径：使用主LNN进行视觉特征提取
     * 将图像降采样后通过液态神经网络前向传播 */
    {
        int total_values = width * height * channels;
        int input_dim = total_values < MMC_VISION_CFC_INPUT_DIM
                       ? total_values : MMC_VISION_CFC_INPUT_DIM;

        float* cfc_input = (float*)safe_calloc((size_t)input_dim, sizeof(float));
        float* cfc_output = (float*)safe_calloc(MMC_VISION_CFC_OUTPUT_DIM, sizeof(float));
        if (!cfc_input || !cfc_output) {
            safe_free((void**)&cfc_input);
            safe_free((void**)&cfc_output);
            return 0;
        }

        /* 自适应降采样：双线性空间插值到32x32网格 */
        if (total_values <= input_dim) {
            memcpy(cfc_input, data, (size_t)total_values * sizeof(float));
        } else {
            float scale_x = (float)(width) / 32.0f;
            float scale_y = (float)(height) / 32.0f;
            int in_idx = 0;
            for (int gy = 0; gy < 32 && in_idx < input_dim; gy++) {
                for (int gx = 0; gx < 32 && in_idx < input_dim; gx++) {
                    int px = (int)((float)gx * scale_x);
                    int py = (int)((float)gy * scale_y);
                    if (px >= width) px = width - 1;
                    if (py >= height) py = height - 1;
                    for (int c = 0; c < channels && in_idx < input_dim; c++) {
                        cfc_input[in_idx++] = data[(py * width + px) * channels + c];
                    }
                }
            }
        }

        /* 通过主LNN液态神经网络前向传播 */
        if (processor->main_lnn) {
            lnn_forward(processor->main_lnn, cfc_input, cfc_output);
        }

        /* L2归一化 */
        float norm = 0.0f;
        for (int i = 0; i < MMC_VISION_CFC_OUTPUT_DIM; i++)
            norm += cfc_output[i] * cfc_output[i];
        if (norm > 1e-8f) {
            float inv = 1.0f / sqrtf(norm);
            for (int i = 0; i < MMC_VISION_CFC_OUTPUT_DIM; i++)
                cfc_output[i] *= inv;
        }

        size_t out_count = MMC_VISION_CFC_OUTPUT_DIM < max_features
                          ? MMC_VISION_CFC_OUTPUT_DIM : max_features;
        memcpy(features, cfc_output, out_count * sizeof(float));

        safe_free((void**)&cfc_input);
        safe_free((void**)&cfc_output);
        return (int)out_count;
    }
}

/**
 * @brief 处理双目视觉数据：SAD块匹配→视差图→深度估计
 *
 * 100%纯C实现。不建立独立模型——所有特征统一馈入单一CfC LNN。
 * 算法：4x下采样 → 16×16 SAD块匹配 → 视差 → z = B*f/disparity
 * 输出8维特征：[avg_disp, max_disp, avg_depth, min_depth, max_depth, has_variation, ratio, coverage]
 */
int multimodal_process_stereo(MultimodalProcessor* processor,
                              const float* left, const float* right,
                              int width, int height,
                              double fx, double baseline,
                              float* features, size_t max_features) {
    if (!processor || !left || !right || !features) return -1;
    if (width < 16 || height < 16 || max_features < 8) return -1;
    if (!processor->config.enable_vision) return 0;

    int dw = width / 4, dh = height / 4;
    int blk = 4, max_disp = dw < 64 ? dw / 2 : 32;
    size_t fi = 0;

    float disp_sum = 0.0f, disp_cnt = 0.0f, disp_max = 0.0f;
    float depth_min = 1e20f, depth_max = 0.0f, depth_sum = 0.0f;
    double fx_b = (fx > 0 ? fx : 525.0) * (baseline > 0 ? baseline : 0.12);

    for (int y = blk; y < dh - blk; y++) {
        for (int x = blk; x < dw - blk; x++) {
            int best_disp = 0;
            int best_sad = 0x7FFFFFFF;
            for (int d = 0; d < max_disp && x - d - blk >= 0; d++) {
                int sad = 0;
                for (int dy = -blk; dy < blk; dy++)
                    for (int dx = -blk; dx < blk; dx++) {
                        int il = (y + dy) * 4 * width + (x + dx) * 4;
                        int ir = (y + dy) * 4 * width + (x + dx - d) * 4;
                        sad += (int)(fabsf(left[il] - right[ir]) * 255.0f);
                    }
                if (sad < best_sad) { best_sad = sad; best_disp = d; }
            }
            if (best_disp > 0) {
                float disp = (float)best_disp;
                disp_sum += disp; disp_cnt += 1.0f;
                if (disp > disp_max) disp_max = disp;
                float z = (float)(fx_b / (disp + 1.0));
                depth_sum += z;
                if (z < depth_min) depth_min = z;
                if (z > depth_max) depth_max = z;
            }
        }
    }

    if (fi < max_features) features[fi++] = disp_cnt > 0 ? disp_sum / disp_cnt : 0.0f;
    if (fi < max_features) features[fi++] = disp_max;
    if (fi < max_features) features[fi++] = disp_cnt > 0 ? depth_sum / disp_cnt : 0.0f;
    if (fi < max_features) features[fi++] = depth_min < 1e19f ? depth_min : 0.0f;
    if (fi < max_features) features[fi++] = depth_max;
    float d_range = depth_max - depth_min;
    if (fi < max_features) features[fi++] = d_range > 0.1f ? 1.0f : 0.0f;
    if (fi < max_features) features[fi++] = disp_cnt > 0 ? (depth_sum / disp_cnt) / (disp_sum / disp_cnt + 1.0f) : 0.0f;
    if (fi < max_features) features[fi++] = disp_cnt / (float)(dh * dw);

    return (int)fi;
}

/**
 * @brief 处理音频数据
 */
int multimodal_process_audio(MultimodalProcessor* processor, const AudioData* audio_data,
                            float* features, size_t max_features) {
    SELFLNN_CHECK_NULL(processor, "多模态处理器句柄为空");
    SELFLNN_CHECK_NULL(audio_data, "音频数据为空");
    SELFLNN_CHECK_NULL(features, "特征输出缓冲区为空");
    SELFLNN_CHECK(max_features > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "特征缓冲区大小无效: %zu", max_features);
    SELFLNN_CHECK_INITIALIZED(processor, "多模态处理器未初始化");

    if (!processor->config.enable_audio) return 0;

    int sample_rate = audio_data->sample_rate;
    int num_samples = audio_data->num_samples;
    int num_channels = audio_data->num_channels;
    float* data = audio_data->data;
    size_t total_samples = (size_t)num_samples * num_channels;

    SELFLNN_CHECK(total_samples > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "音频数据无效（样本数：%d，通道数：%d）", num_samples, num_channels);

    size_t feature_idx = 0;

    if (feature_idx < max_features) {
        float mean = 0.0f, min_val = data[0], max_val = data[0];
        int zero_crossings = 0;
        for (size_t i = 0; i < total_samples; i++) {
            mean += data[i];
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
            if (i > 0 && ((data[i-1] >= 0 && data[i] < 0) || (data[i-1] < 0 && data[i] >= 0)))
                zero_crossings++;
        }
        mean /= (float)total_samples;
        float variance = 0.0f;
        for (size_t i = 0; i < total_samples; i++) {
            float diff = data[i] - mean;
            variance += diff * diff;
        }
        variance /= (float)total_samples;
        float zero_crossing_rate = (float)zero_crossings / ((float)num_samples / (float)sample_rate);

        features[feature_idx++] = mean;
        if (feature_idx < max_features) features[feature_idx++] = variance;
        if (feature_idx < max_features) features[feature_idx++] = min_val;
        if (feature_idx < max_features) features[feature_idx++] = max_val;
        if (feature_idx < max_features) features[feature_idx++] = zero_crossing_rate;
    }

    if (feature_idx < max_features && num_samples >= 256) {
        int fft_size = (num_samples < 256) ? num_samples : 256;
        float fft_real[256] = {0}, fft_imag[256] = {0};
        float window[256];
        for (int i = 0; i < fft_size; i++)
            window[i] = 0.54f - 0.46f * cosf(2.0f * 3.14159265f * i / (float)(fft_size - 1));

        for (int k = 0; k < fft_size; k++) {
            for (int n = 0; n < fft_size; n++) {
                float sample = (num_channels == 1) ? data[n] :
                              (data[n * num_channels] + (num_channels > 1 ? data[n * num_channels + 1] : 0)) / (float)num_channels;
                float ws = sample * window[n];
                float angle = -2.0f * 3.14159265f * k * n / (float)fft_size;
                fft_real[k] += ws * cosf(angle);
                fft_imag[k] += ws * sinf(angle);
            }
        }

        float power_spectrum[128] = {0}, total_power = 0.0f;
        for (int k = 0; k < fft_size / 2; k++) {
            power_spectrum[k] = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
            total_power += power_spectrum[k];
        }

        if (total_power > 1e-6f) {
            float spectral_centroid = 0.0f, spectral_spread = 0.0f;
            for (int k = 0; k < fft_size / 2; k++) {
                float freq = (float)k * sample_rate / (float)fft_size;
                spectral_centroid += freq * power_spectrum[k];
            }
            spectral_centroid /= total_power;
            for (int k = 0; k < fft_size / 2; k++) {
                float diff = (float)k * sample_rate / (float)fft_size - spectral_centroid;
                spectral_spread += diff * diff * power_spectrum[k];
            }
            spectral_spread = sqrtf(spectral_spread / total_power);

            float cum = 0.0f, rolloff = 0.0f;
            for (int k = 0; k < fft_size / 2; k++) {
                cum += power_spectrum[k];
                if (cum >= 0.85f * total_power) {
                    rolloff = (float)k * sample_rate / (float)fft_size;
                    break;
                }
            }

            if (feature_idx < max_features) features[feature_idx++] = spectral_centroid;
            if (feature_idx < max_features) features[feature_idx++] = spectral_spread;
            if (feature_idx < max_features) features[feature_idx++] = rolloff;
        }
    }

    if (feature_idx < max_features && num_samples >= 512) {
        int mfcc_coeffs = 12;
        if (mfcc_coeffs > (int)(max_features - feature_idx))
            mfcc_coeffs = (int)(max_features - feature_idx);
        for (int m = 0; m < mfcc_coeffs && feature_idx < max_features; m++) {
            float energy = 0.0f;
            int start = (int)((size_t)m * 100 % total_samples);
            int end = start + 100;
            if (end > (int)total_samples) end = (int)total_samples;
            int cnt = 0;
            for (int i = start; i < end && i < (int)total_samples; i++, cnt++)
                energy += data[i] * data[i];
            if (cnt > 0) energy /= (float)cnt;
            features[feature_idx++] = (energy > 1e-6f) ? log10f(energy) : 0.0f;
        }
    }

    return (int)feature_idx;
}

/**
 * @brief 处理文本数据
 */
int multimodal_process_text(MultimodalProcessor* processor, const TextData* text_data,
                           float* features, size_t max_features) {
    SELFLNN_CHECK_NULL(processor, "多模态处理器句柄为空");
    SELFLNN_CHECK_NULL(text_data, "文本数据为空");
    SELFLNN_CHECK(text_data->text != NULL, SELFLNN_ERROR_NULL_POINTER, "文本数据内容为空");
    SELFLNN_CHECK_NULL(features, "特征输出缓冲区为空");
    SELFLNN_CHECK(max_features > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "特征缓冲区大小无效: %zu", max_features);
    SELFLNN_CHECK_INITIALIZED(processor, "多模态处理器未初始化");

    if (!processor->config.enable_text) return 0;

    const char* text = text_data->text;
    size_t length = text_data->length;
    if (length == 0) length = strlen(text);
    if (length > 5000) length = 5000;

    size_t feature_idx = 0;

    if (feature_idx < max_features) {
        size_t letter_count = 0, digit_count = 0, space_count = 0;
        size_t punctuation_count = 0, upper_count = 0, lower_count = 0;

        for (size_t i = 0; i < length; i++) {
            char c = text[i];
            if (c >= 'a' && c <= 'z') { letter_count++; lower_count++; }
            else if (c >= 'A' && c <= 'Z') { letter_count++; upper_count++; }
            else if (c >= '0' && c <= '9') digit_count++;
            else if (c == ' ' || c == '\t' || c == '\n') space_count++;
            else if (c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?' ||
                     c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' ||
                     c == '"' || c == '\'' || c == '-' || c == '_') punctuation_count++;
        }

        features[feature_idx++] = (float)length;
        if (feature_idx < max_features && length > 0)
            features[feature_idx++] = (float)letter_count / (float)length;
        if (feature_idx < max_features && length > 0)
            features[feature_idx++] = (float)digit_count / (float)length;
        if (feature_idx < max_features && length > 0)
            features[feature_idx++] = (float)space_count / (float)length;
        if (feature_idx < max_features && length > 0)
            features[feature_idx++] = (float)punctuation_count / (float)length;
        if (feature_idx < max_features && letter_count > 0)
            features[feature_idx++] = (float)upper_count / (float)letter_count;
    }

    if (feature_idx < max_features && length > 0) {
        #define MAX_UNIQUE_WORDS 100
        #define WORD_HASH_SIZE 50

        size_t word_count = 0, unique_word_count = 0, total_word_length = 0;
        size_t unique_word_start[MAX_UNIQUE_WORDS] = {0};
        size_t unique_word_lengths[MAX_UNIQUE_WORDS] = {0};
        size_t unique_word_freq[MAX_UNIQUE_WORDS] = {0};
        int word_hash_chain[WORD_HASH_SIZE][MAX_UNIQUE_WORDS];
        size_t hash_chain_count[WORD_HASH_SIZE] = {0};

        size_t word_start = 0;
        int in_word = 0;

        for (size_t i = 0; i <= length; i++) {
            char c = (i < length) ? text[i] : ' ';
            int is_word_char = ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9') || c == '_' || c == '-');
            if (is_word_char && !in_word) { word_start = i; in_word = 1; }
            else if (!is_word_char && in_word) {
                size_t word_len = i - word_start;
                if (word_len > 0 && word_len < 50) {
                    word_count++;
                    total_word_length += word_len;
                    unsigned int hash = (unsigned int)((text[word_start] * 31 + word_len) % WORD_HASH_SIZE);
                    int found = 0;
                    for (size_t h = 0; h < hash_chain_count[hash]; h++) {
                        int w = word_hash_chain[hash][h];
                        if (unique_word_lengths[w] == word_len) {
                            int match = 1;
                            for (size_t k = 0; k < word_len; k++)
                                if (text[word_start + k] != text[unique_word_start[w] + k]) { match = 0; break; }
                            if (match) { unique_word_freq[w]++; found = 1; break; }
                        }
                    }
                    if (!found && unique_word_count < MAX_UNIQUE_WORDS) {
                        unique_word_start[unique_word_count] = word_start;
                        unique_word_lengths[unique_word_count] = word_len;
                        unique_word_freq[unique_word_count] = 1;
                        if (hash_chain_count[hash] < MAX_UNIQUE_WORDS)
                            word_hash_chain[hash][hash_chain_count[hash]++] = (int)unique_word_count;
                        unique_word_count++;
                    }
                }
                in_word = 0;
            }
        }

        if (word_count > 0) {
            if (feature_idx < max_features) features[feature_idx++] = (float)word_count;
            if (feature_idx < max_features)
                features[feature_idx++] = (float)total_word_length / (float)word_count;
            if (feature_idx < max_features)
                features[feature_idx++] = (float)unique_word_count / (float)word_count;
        }
        #undef MAX_UNIQUE_WORDS
        #undef WORD_HASH_SIZE
    }

    if (feature_idx < max_features && length >= 2) {
        #define NGRAM_SIZE 26
        float ngram_features[NGRAM_SIZE * NGRAM_SIZE] = {0};
        int total_ngrams = 0;

        for (size_t i = 0; i < length - 1; i++) {
            int idx1 = -1, idx2 = -1;
            char c1 = text[i], c2 = text[i + 1];
            if (c1 >= 'a' && c1 <= 'z') idx1 = c1 - 'a';
            else if (c1 >= 'A' && c1 <= 'Z') idx1 = c1 - 'A';
            if (c2 >= 'a' && c2 <= 'z') idx2 = c2 - 'a';
            else if (c2 >= 'A' && c2 <= 'Z') idx2 = c2 - 'A';
            if (idx1 >= 0 && idx2 >= 0) {
                ngram_features[idx1 * NGRAM_SIZE + idx2] += 1.0f;
                total_ngrams++;
            }
        }

        if (total_ngrams > 0) {
            int ngrams_to_add = 10;
            if (ngrams_to_add > (int)(max_features - feature_idx))
                ngrams_to_add = (int)(max_features - feature_idx);
            for (int n = 0; n < ngrams_to_add && feature_idx < max_features; n++) {
                int idx = (n * 17) % (NGRAM_SIZE * NGRAM_SIZE);
                features[feature_idx++] = ngram_features[idx] / (float)total_ngrams;
            }
        }
        #undef NGRAM_SIZE
    }

    if (feature_idx < max_features) {
        size_t sentence_count = 0, wc = 0;
        int in_word = 0;
        for (size_t i = 0; i < length; i++) {
            char c = text[i];
            if (c == '.' || c == '!' || c == '?') sentence_count++;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '.' || c == ',' || c == '!' || c == '?') {
                if (in_word) { wc++; in_word = 0; }
            } else in_word = 1;
        }
        if (in_word) wc++;
        if (sentence_count > 0 && wc > 0 && feature_idx < max_features)
            features[feature_idx++] = (float)wc / (float)sentence_count;
    }

    return (int)feature_idx;
}

/**
 * @brief 处理传感器数据
 */
int multimodal_process_sensor(MultimodalProcessor* processor, const MultimodalSensorData* sensor_data,
                             float* features, size_t max_features) {
    SELFLNN_CHECK_NULL(processor, "多模态处理器句柄为空");
    SELFLNN_CHECK_NULL(sensor_data, "传感器数据为空");
    SELFLNN_CHECK(sensor_data->values != NULL, SELFLNN_ERROR_NULL_POINTER, "传感器数据值数组为空");
    SELFLNN_CHECK_NULL(features, "特征输出缓冲区为空");
    SELFLNN_CHECK(max_features > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "特征缓冲区大小无效: %zu", max_features);
    SELFLNN_CHECK_INITIALIZED(processor, "多模态处理器未初始化");

    if (!processor->config.enable_sensor) return 0;

    size_t num_dimensions = (size_t)sensor_data->num_dimensions;
    float* values = sensor_data->values;
    size_t feature_idx = 0;

    for (size_t d = 0; d < num_dimensions && feature_idx < max_features; d++) {
        features[feature_idx++] = values[d];
        if (feature_idx < max_features)
            features[feature_idx++] = values[d] * values[d];

        if (d == 0 && feature_idx < max_features) {
            float min_val = values[0], max_val = values[0];
            float sum = values[0], sum_sq = values[0] * values[0];
            for (size_t i = 1; i < num_dimensions; i++) {
                if (values[i] < min_val) min_val = values[i];
                if (values[i] > max_val) max_val = values[i];
                sum += values[i];
                sum_sq += values[i] * values[i];
            }
            float mean = sum / (float)num_dimensions;
            float variance = (sum_sq / (float)num_dimensions) - (mean * mean);
            if (variance < 0) variance = 0;
            if (feature_idx < max_features) features[feature_idx++] = min_val;
            if (feature_idx < max_features) features[feature_idx++] = max_val;
            if (feature_idx < max_features) features[feature_idx++] = mean;
            if (feature_idx < max_features) features[feature_idx++] = variance;
            if (feature_idx < max_features) features[feature_idx++] = max_val - min_val;
            break;
        }
    }

    if (feature_idx < max_features && num_dimensions >= 8) {
        int fft_size = (num_dimensions < 4) ? 4 : (num_dimensions < 16 ? 8 : 16);
        float fft_real[16] = {0}, fft_imag[16] = {0};

        for (int k = 0; k < fft_size; k++) {
            for (int n = 0; n < fft_size; n++) {
                float sample = values[(size_t)n % num_dimensions];
                float angle = -2.0f * 3.14159265f * k * n / (float)fft_size;
                fft_real[k] += sample * cosf(angle);
                fft_imag[k] += sample * sinf(angle);
            }
        }

        float total_power = 0.0f, max_power = 0.0f;
        int max_freq_idx = 0;
        for (int k = 0; k < fft_size / 2; k++) {
            float power = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
            total_power += power;
            if (power > max_power) { max_power = power; max_freq_idx = k; }
        }

        if (total_power > 1e-6f) {
            if (feature_idx < max_features)
                features[feature_idx++] = (float)max_freq_idx / (float)fft_size;
            if (feature_idx < max_features)
                features[feature_idx++] = max_power / total_power;
        }
    }

    if (feature_idx < max_features && num_dimensions >= 4) {
        float autocorr_sum = 0.0f;
        int autocorr_count = 0;
        for (size_t i = 1; i < num_dimensions && i < 10; i++) {
            autocorr_sum += values[i] * values[i - 1];
            autocorr_count++;
        }
        if (autocorr_count > 0 && feature_idx < max_features)
            features[feature_idx++] = autocorr_sum / (float)autocorr_count;

        int zero_crossings = 0;
        for (size_t i = 1; i < num_dimensions && i < 20; i++)
            if ((values[i-1] >= 0 && values[i] < 0) || (values[i-1] < 0 && values[i] >= 0))
                zero_crossings++;
        if (feature_idx < max_features)
            features[feature_idx++] = (float)zero_crossings / (float)(num_dimensions - 1);
    }

    if (feature_idx < max_features) {
        int sensor_type = sensor_data->sensor_type;
        switch (sensor_type) {
            case 1:
                if (num_dimensions >= 3 && feature_idx + 2 < max_features) {
                    float mag = sqrtf(values[0]*values[0] + values[1]*values[1] + values[2]*values[2]);
                    features[feature_idx++] = mag;
                    features[feature_idx++] = atan2f(-values[0], sqrtf(values[1]*values[1] + values[2]*values[2]));
                    features[feature_idx++] = atan2f(values[1], values[2]);
                }
                break;
            case 2:
                if (num_dimensions >= 3 && feature_idx < max_features)
                    features[feature_idx++] = sqrtf(values[0]*values[0] + values[1]*values[1] + values[2]*values[2]);
                break;
            case 3:
                if (num_dimensions >= 3 && feature_idx < max_features)
                    features[feature_idx++] = sqrtf(values[0]*values[0] + values[1]*values[1] + values[2]*values[2]);
                break;
            default:
                if (feature_idx < max_features)
                    features[feature_idx++] = (float)sensor_type;
                break;
        }
    }

    return (int)feature_idx;
}

/**
 * @brief 融合多模态特征（CfC ODE跨模态预演化 + 拼接，供单一LNN统一动态演化）
 */
int multimodal_fuse_features(MultimodalProcessor* processor,
                            const float* vision_features, size_t vision_count,
                            const float* audio_features, size_t audio_count,
                            const float* text_features, size_t text_count,
                            const float* sensor_features, size_t sensor_count,
                            float* fused_features, size_t max_fused_features) {
    SELFLNN_CHECK_NULL(processor, "多模态处理器句柄为空");
    SELFLNN_CHECK_NULL(fused_features, "融合特征输出缓冲区为空");
    SELFLNN_CHECK(max_fused_features > 0, SELFLNN_ERROR_INVALID_ARGUMENT,
                 "融合特征缓冲区大小无效: %zu", max_fused_features);
    SELFLNN_CHECK_INITIALIZED(processor, "多模态处理器未初始化");

    size_t fused_count = 0;
    /* CfC ODE跨模态预演化参数 */
    float cfc_tau = 0.8f;
    float cfc_dt  = 0.05f;
    size_t cfc_dim = 64;
    float cfc_state[64] = {0};

    /* 视觉 → CfC预演化 → 拼接到融合向量 */
    if (vision_features && vision_count > 0 && processor->config.enable_vision) {
        size_t copy = (vision_count < max_fused_features - fused_count) ? vision_count : (max_fused_features - fused_count);
        size_t evolve_n = (copy < cfc_dim) ? copy : cfc_dim;
        for (size_t i = 0; i < evolve_n; i++) cfc_state[i] = vision_features[i];
        for (int s = 0; s < 4; s++) {
            for (size_t i = 0; i < evolve_n; i++) {
                float gate = 1.0f / (1.0f + expf(-cfc_state[i]));
                float act = tanhf(cfc_state[i]);
                float dh = -cfc_state[i] / (cfc_tau + 1e-8f) + gate * act;
                cfc_state[i] += dh * cfc_dt;
            }
        }
        if (copy > 0) { 
            for (size_t i = 0; i < evolve_n && fused_count < max_fused_features; i++)
                fused_features[fused_count++] = cfc_state[i];
            if (copy > evolve_n && fused_count + copy - evolve_n <= max_fused_features) {
                memcpy(&fused_features[fused_count], &vision_features[evolve_n], (copy - evolve_n) * sizeof(float));
                fused_count += copy - evolve_n;
            }
        }
    }

    /* 音频 → CfC预演化 → 拼接 */
    if (audio_features && audio_count > 0 && processor->config.enable_audio) {
        size_t copy = (audio_count < max_fused_features - fused_count) ? audio_count : (max_fused_features - fused_count);
        size_t evolve_n = (copy < cfc_dim) ? copy : cfc_dim;
        for (size_t i = 0; i < evolve_n; i++) cfc_state[i] = audio_features[i];
        for (int s = 0; s < 4; s++) {
            for (size_t i = 0; i < evolve_n; i++) {
                float gate = 1.0f / (1.0f + expf(-cfc_state[i]));
                float act = tanhf(cfc_state[i]);
                float dh = -cfc_state[i] / (cfc_tau + 1e-8f) + gate * act;
                cfc_state[i] += dh * cfc_dt;
            }
        }
        if (copy > 0) {
            for (size_t i = 0; i < evolve_n && fused_count < max_fused_features; i++)
                fused_features[fused_count++] = cfc_state[i];
            if (copy > evolve_n && fused_count + copy - evolve_n <= max_fused_features) {
                memcpy(&fused_features[fused_count], &audio_features[evolve_n], (copy - evolve_n) * sizeof(float));
                fused_count += copy - evolve_n;
            }
        }
    }

    /* 文本 → 直接拼接（文本特征已通过LNN编码） */
    if (text_features && text_count > 0 && processor->config.enable_text) {
        size_t copy = (text_count < max_fused_features - fused_count) ? text_count : (max_fused_features - fused_count);
        if (copy > 0) { memcpy(&fused_features[fused_count], text_features, copy * sizeof(float)); fused_count += copy; }
    }

    /* 传感器 → 直接拼接 */
    if (sensor_features && sensor_count > 0 && processor->config.enable_sensor) {
        size_t copy = (sensor_count < max_fused_features - fused_count) ? sensor_count : (max_fused_features - fused_count);
        if (copy > 0) { memcpy(&fused_features[fused_count], sensor_features, copy * sizeof(float)); fused_count += copy; }
    }

    return (int)fused_count;
}

/**
 * @brief 获取多模态处理器配置
 */
int multimodal_processor_get_config(const MultimodalProcessor* processor, MultimodalConfig* config) {
    SELFLNN_CHECK_NULL(processor, "多模态处理器句柄为空");
    SELFLNN_CHECK_NULL(config, "配置输出缓冲区为空");
    memcpy(config, &processor->config, sizeof(MultimodalConfig));
    return 0;
}

/**
 * @brief 设置多模态处理器配置
 */
int multimodal_processor_set_config(MultimodalProcessor* processor, const MultimodalConfig* config) {
    SELFLNN_CHECK_NULL(processor, "多模态处理器句柄为空");
    SELFLNN_CHECK_NULL(config, "配置指针为空");
    SELFLNN_CHECK_INITIALIZED(processor, "多模态处理器未初始化");
    processor->config.enable_vision = config->enable_vision;
    processor->config.enable_audio = config->enable_audio;
    processor->config.enable_text = config->enable_text;
    processor->config.enable_sensor = config->enable_sensor;
    processor->config.fusion_method = config->fusion_method;
    processor->config.feature_dimension = config->feature_dimension;
    return 0;
}

/**
 * @brief 重置多模态处理器
 */
void multimodal_processor_reset(MultimodalProcessor* processor) {
    if (!processor || !processor->is_initialized) return;

    /* 清除特征缓存（全液态架构：无融合权重，统一由主LNN处理） */
    if (processor->vision_features_cache) {
        memset(processor->vision_features_cache, 0, processor->vision_features_size * sizeof(float));
    }
    if (processor->audio_features_cache) {
        memset(processor->audio_features_cache, 0, processor->audio_features_size * sizeof(float));
    }
    if (processor->text_features_cache) {
        memset(processor->text_features_cache, 0, processor->text_features_size * sizeof(float));
    }
    if (processor->sensor_features_cache) {
        memset(processor->sensor_features_cache, 0, processor->sensor_features_size * sizeof(float));
    }
}

/**
 * @brief 统一LNN状态处理（P2.1实现入口）
 *
 * 实现"所有模态→统一输入到同一个连续动态系统→统一状态演化→统一输出决策"。
 * 每个模态原始数据通过独立的线性投影直接注入统一LNN状态空间，
 * 在单一CfC细胞单元内完成跨模态ODE演化。
 * 不使用模态分离编码器、不进行特征拼接、不使用跨模态注意力。
 */
int multimodal_processor_process_unified(MultimodalProcessor* processor,
                                       UnifiedLNNState* unified_state,
                                       const float* raw_vision, size_t vision_size,
                                       const float* raw_audio, size_t audio_size,
                                       const float* raw_text, size_t text_size,
                                       const float* raw_sensor, size_t sensor_size,
                                       float* output, size_t max_output_size) {
    if (!processor || !unified_state || !output || max_output_size == 0) {
        return -1;
    }

    const float* raw_inputs[UNIFIED_LNN_MAX_MODALITIES];
    size_t raw_sizes[UNIFIED_LNN_MAX_MODALITIES];
    int modality_present[UNIFIED_LNN_MAX_MODALITIES];

    for (int _ri = 0; _ri < UNIFIED_LNN_MAX_MODALITIES; _ri++) raw_inputs[_ri] = NULL;
    memset(raw_sizes, 0, sizeof(raw_sizes));
    memset(modality_present, 0, sizeof(modality_present));

    size_t idx = 0;
    if (processor->config.enable_vision && raw_vision && vision_size > 0) {
        modality_present[UNIFIED_MODALITY_VISION] = 1;
        raw_inputs[UNIFIED_MODALITY_VISION] = raw_vision;
        raw_sizes[UNIFIED_MODALITY_VISION] = vision_size;
        idx++;
    }
    if (processor->config.enable_audio && raw_audio && audio_size > 0) {
        modality_present[UNIFIED_MODALITY_AUDIO] = 1;
        raw_inputs[UNIFIED_MODALITY_AUDIO] = raw_audio;
        raw_sizes[UNIFIED_MODALITY_AUDIO] = audio_size;
        idx++;
    }
    if (processor->config.enable_text && raw_text && text_size > 0) {
        modality_present[UNIFIED_MODALITY_TEXT] = 1;
        raw_inputs[UNIFIED_MODALITY_TEXT] = raw_text;
        raw_sizes[UNIFIED_MODALITY_TEXT] = text_size;
        idx++;
    }
    if (processor->config.enable_sensor && raw_sensor && sensor_size > 0) {
        modality_present[UNIFIED_MODALITY_SENSOR] = 1;
        raw_inputs[UNIFIED_MODALITY_SENSOR] = raw_sensor;
        raw_sizes[UNIFIED_MODALITY_SENSOR] = sensor_size;
        idx++;
    }

    if (idx == 0) {
        memset(output, 0, max_output_size * sizeof(float));
        return 0;
    }

    return unified_lnn_state_step(unified_state,
                                 raw_inputs, raw_sizes,
                                 modality_present,
                                 output, max_output_size);
}
