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
#include "selflnn/multimodal/liquid_vision.h"
#include "selflnn/multimodal/image_recognition_deep.h"
#include "selflnn/multimodal/multimodal_integration.h" /* ZSFWS-005: 统一融合管道 */
#include "selflnn/core/unified_lnn_state.h"
#include "selflnn/core/lnn.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/knowledge/knowledge.h"

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

    /* S-NEW-1: 深度图像识别分类器（细粒度/开放集/少样本/零样本） */
    IRDFineClassifier* ird_classifier;

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

    /* P0-01修复: 初始化CfC液态视觉处理器，适配图像特征提取参数 */
    {
        CfcVisionConfig vis_config = cfc_vision_get_default_config();
        vis_config.image_width   = 224;      /* 标准视觉输入分辨率 */
        vis_config.image_height  = 224;
        vis_config.image_channels = 3;       /* RGB三通道 */
        vis_config.patch_size    = 16;       /* 补丁大小（适配14×14补丁网格） */
        vis_config.output_dim    = MMC_VISION_CFC_OUTPUT_DIM;  /* 256维输出 */
        vis_config.num_ode_layers = 3;       /* 3层CfC ODE深度特征提取 */
        vis_config.time_constant = 1.0f;     /* 连续时间常数τ */
        vis_config.delta_t       = 0.1f;     /* ODE数值积分步长 */

        processor->cfc_vision_proc = cfc_vision_processor_create(&vis_config);
        if (!processor->cfc_vision_proc) {
            /* CfC视觉处理器创建失败，整体失败，释放已分配资源 */
            safe_free((void**)&processor);
            return NULL;
        }

        /* S-NEW-1: 创建深度图像识别分类器
         * image_recognition_deep.c之前孤立未被调用 */
        IRDFineConfig ird_cfg;
        memset(&ird_cfg, 0, sizeof(ird_cfg));
        ird_cfg.feature_dim = 256;
        ird_cfg.patch_size = 16;
        ird_cfg.num_parts = 8;
        ird_cfg.discriminative_threshold = 0.5f;
        ird_cfg.num_fine_categories = 100;
        processor->ird_classifier = ird_fine_classifier_create(&ird_cfg);
    }

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

    /* P0-01修复: 释放CfC液态视觉处理器 */
    if (processor->cfc_vision_proc) {
        cfc_vision_processor_destroy(processor->cfc_vision_proc);
        processor->cfc_vision_proc = NULL;
    }
    /* S-NEW-1: 释放深度图像识别分类器 */
    if (processor->ird_classifier) {
        ird_fine_free(processor->ird_classifier);
        processor->ird_classifier = NULL;
    }

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
            /* CfC特征提取成功 */
            return extracted;
        }
        /* M-001修复: CfC提取失败时返回明确错误码，禁止静默降级到LNN回退路径 */
        fprintf(stderr, "[多模态错误] CfC视觉特征提取失败，拒绝降级处理\n");
        return 0;
    }

    /* M-001修复: CfC视觉处理器未初始化时同样拒绝降级
     * 原先的回退路径（LNN降采样+双线性插值）已禁用
     * 调用方必须先初始化CfC视觉处理器再使用视觉功能 */
    fprintf(stderr, "[多模态错误] CfC视觉处理器未初始化，无法提取视觉特征\n");
    return 0;
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

    /* ZSFABC-F004修复: 完整MFCC实现，含梅尔滤波器组+DCT
     * 替代之前的分段log能量近似 */
    if (feature_idx < max_features && num_samples >= 512) {
        int mfcc_coeffs = 12;
        if (mfcc_coeffs > (int)(max_features - feature_idx))
            mfcc_coeffs = (int)(max_features - feature_idx);

        /* 1. 计算功率谱 (取前一半FFT) */
        int fft_n = 512;
        if ((int)num_samples < fft_n) fft_n = (int)num_samples;
        float power_spectrum[256];
        int num_bins = fft_n / 2;
        
        /* ZSFABC修复: 完整DFT功率谱估计（直接傅里叶变换，非简化版本） */
        for (int k = 0; k < num_bins && k < 256; k++) {
            float real = 0.0f, imag = 0.0f;
            for (int n = 0; n < fft_n; n++) {
                float angle = -2.0f * 3.14159265358979f * (float)k * (float)n / (float)fft_n;
                real += data[n] * cosf(angle);
                imag += data[n] * sinf(angle);
            }
            power_spectrum[k] = (real * real + imag * imag) / (float)fft_n;
            if (power_spectrum[k] < 1e-10f) power_spectrum[k] = 1e-10f;
        }

        /* 2. 构建梅尔滤波器组 (26个三角滤波器) */
        int num_mel_filters = 26;
        float mel_points[28];
        float mel_low = 1125.0f * logf(1.0f + 300.0f / 700.0f);
        float mel_high = 1125.0f * logf(1.0f + (float)(sample_rate / 2) / 700.0f);
        for (int m = 0; m < num_mel_filters + 2; m++) {
            float mel = mel_low + (float)m * (mel_high - mel_low) / (float)(num_mel_filters + 1);
            float freq = 700.0f * (expf(mel / 1125.0f) - 1.0f);
            mel_points[m] = freq * (float)num_bins / (float)(sample_rate / 2);
        }

        float mel_energies[26] = {0};
        for (int m = 1; m <= num_mel_filters; m++) {
            float mel_energy = 0.0f;
            for (int k = 0; k < num_bins; k++) {
                float weight = 0.0f;
                if ((float)k <= mel_points[m - 1]) weight = 0.0f;
                else if ((float)k <= mel_points[m])
                    weight = ((float)k - mel_points[m - 1]) / (mel_points[m] - mel_points[m - 1] + 1e-10f);
                else if ((float)k <= mel_points[m + 1])
                    weight = (mel_points[m + 1] - (float)k) / (mel_points[m + 1] - mel_points[m] + 1e-10f);
                else weight = 0.0f;
                mel_energy += weight * power_spectrum[k];
            }
            mel_energies[m - 1] = (mel_energy > 1e-10f) ? logf(mel_energy) : -23.0f;
        }

        /* 3. DCT-II变换: 梅尔频谱 → MFCC系数 */
        for (int c = 0; c < mfcc_coeffs && feature_idx < max_features; c++) {
            float mfcc = 0.0f;
            for (int m = 0; m < num_mel_filters; m++) {
                mfcc += mel_energies[m] * cosf(3.14159265358979f * (float)c * ((float)m + 0.5f) / (float)num_mel_filters);
            }
            /* 归一化 (DCT-II 正交化) */
            if (c == 0) mfcc *= sqrtf(1.0f / (float)num_mel_filters);
            else mfcc *= sqrtf(2.0f / (float)num_mel_filters);
            features[feature_idx++] = mfcc;
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
 * ZSFWS-005修复: 融合多模态特征 —— 真CfC ODE跨模态统一演化
 *
 * 需求要求"所有模态→统一输入到同一个连续动态系统→统一状态演化→统一输出"。
 * 原实现为独立4步Euler+简单拼接，完全无法实现跨模态交互。
 * 现改为调用multimodal_integration.c的方案C——真正的CfC ODE统一演化。
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

    /* ZSFWS-005: 收集活跃模态数据，调用统一CfC ODE融合管道 */
    const float* modality_data[SELFLNN_MAX_MODALITIES] = {NULL};
    int modality_dims[SELFLNN_MAX_MODALITIES] = {0};
    int active_count = 0;

    if (vision_features && vision_count > 0 && processor->config.enable_vision) {
        modality_data[active_count] = vision_features;
        modality_dims[active_count] = (int)vision_count;
        active_count++;
    }
    if (audio_features && audio_count > 0 && processor->config.enable_audio) {
        modality_data[active_count] = audio_features;
        modality_dims[active_count] = (int)audio_count;
        active_count++;
    }
    if (text_features && text_count > 0 && processor->config.enable_text) {
        modality_data[active_count] = text_features;
        modality_dims[active_count] = (int)text_count;
        active_count++;
    }
    if (sensor_features && sensor_count > 0 && processor->config.enable_sensor) {
        modality_data[active_count] = sensor_features;
        modality_dims[active_count] = (int)sensor_count;
        active_count++;
    }

    if (active_count == 0) return 0;

    /* 委托给真正的CfC ODE统一融合管道 */
    int result = multimodal_unified_pipeline(modality_data, modality_dims,
        active_count, NULL, fused_features, (int)max_fused_features);
    
    /* 如果统一融合成功，直接返回 */
    if (result == 0) {
        /* 计算实际非零输出维度 */
        size_t actual_dim = 0;
        for (size_t i = 0; i < max_fused_features; i++) {
            if (fabsf(fused_features[i]) > 1e-10f) actual_dim = i + 1;
        }
        return (int)(actual_dim > 0 ? actual_dim : max_fused_features);
    }

    /* 回退路径：如果CfC ODE融合不可用，执行简单的归一化拼接
     * （此处保留作为极端情况下的安全网，不对真正融合造成影响） */
    size_t fused_count = 0;
    const float* sources[4] = {vision_features, audio_features, text_features, sensor_features};
    size_t counts[4] = {vision_count, audio_count, text_count, sensor_count};
    int enabled[4] = {
        processor->config.enable_vision,
        processor->config.enable_audio,
        processor->config.enable_text,
        processor->config.enable_sensor
    };
    for (int m = 0; m < 4; m++) {
        if (sources[m] && counts[m] > 0 && enabled[m]) {
            size_t copy = (counts[m] < max_fused_features - fused_count) ?
                counts[m] : (max_fused_features - fused_count);
            if (copy > 0) {
                /* 归一化后拼接，避免高维模态主导 */
                float norm = 0.0f;
                for (size_t i = 0; i < copy; i++) norm += sources[m][i] * sources[m][i];
                norm = sqrtf(norm + 1e-12f);
                float inv_norm = 1.0f / norm;
                for (size_t i = 0; i < copy; i++)
                    fused_features[fused_count + i] = sources[m][i] * inv_norm;
                fused_count += copy;
            }
        }
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

/* ============================================================================
 * P1-003修复: 真实TF-IDF关键词提取 + 语义相似度 + 查询扩展
 *
 * 原问题: TF数组固定赋值1.0f，IDF用固定常量，分词用简单最大匹配
 * 修复: 真实TF（词频/总词数）× IDF（log(N/(df+1))+1）+ 词共现语义相似度
 * 保留高频词表作为回退路径
 * ============================================================================ */

/* 关键词提取常量 */
#define MM_MAX_TERM_LEN      64    /* 单个词最大字节长度（支持UTF-8中文） */
#define MM_MAX_TERMS         512   /* 单个文档最大词数 */
#define MM_MAX_KEYWORDS      128   /* 最大提取关键词数量 */
#define MM_COOCCUR_WINDOW    8     /* 词共现窗口大小 */
#define MM_IDF_SMOOTH        1.0f  /* IDF平滑参数 */
#define MM_SIM_THRESHOLD     0.35f /* 相似度最低阈值 */

/* ============================================================================
 * 高频词回退表 — 100+常用中文词汇（当知识库不可用时回退）
 * ============================================================================ */
static const char* mm_high_freq_words[] = {
    /* 虚词/结构词 */
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "我们", "他们", "自己", "什么", "知道", "可以", "没有", "这个",
    "因为", "所以", "但是", "如果", "虽然", "而且", "已经", "还是",
    "起来", "出来", "过来", "回去", "下来", "上去", "进去", "出去",
    /* 实体/概念词 */
    "系统", "数据", "功能", "模块", "文件", "代码", "用户", "程序", "接口",
    "时间", "状态", "任务", "结果", "参数", "配置", "操作", "环境", "设备",
    "网络", "内存", "进程", "服务", "命令", "路径", "结构", "方法", "类型",
    "定义", "变量", "函数", "对象", "事件", "消息", "错误", "模式", "测试",
    /* 知识/语义词 */
    "学习", "训练", "模型", "算法", "计算", "分析", "处理", "识别", "检测",
    "推理", "规划", "决策", "控制", "执行", "评估", "优化", "更新", "搜索",
    "知识", "推理", "语义", "逻辑", "规则", "关系", "概念", "属性", "事实",
    "信息", "内容", "描述", "说明", "示例", "定义", "分类", "关联", "匹配",
    "目标", "方案", "问题", "需求", "设计", "实现", "验证", "修复",
    "当前", "之前", "之后", "同时", "另外", "例如", "包括", "通过",
    "使用", "需要", "提供", "支持", "开始", "结束", "完成",
    /* 高频动词/形容词 */
    "进行", "存在", "发生", "产生", "包括", "包含", "属于", "关系",
    "主要", "重要", "基本", "核心", "高级", "完整", "真实", "有效",
    "快速", "稳定", "自动", "动态", "连续", "统一", "独立",
    NULL
};
static const int mm_high_freq_count = 
    (int)(sizeof(mm_high_freq_words) / sizeof(mm_high_freq_words[0])) - 1;

/* ============================================================================
 * 中文同义词/近义词扩展表（查询语义扩展用）
 * ============================================================================ */
typedef struct {
    const char* word;           /* 原始词 */
    const char* synonyms[8];    /* 近义词列表（最多8个） */
    int synonym_count;
} MmSynonymEntry;

static const MmSynonymEntry mm_synonym_table[] = {
    {"系统",  {"平台", "框架", "环境", "体系", "架构"}, 5},
    {"数据",  {"信息", "资料", "内容", "素材", "记录"}, 5},
    {"功能",  {"特性", "能力", "作用", "效用", "机能"}, 5},
    {"模块",  {"组件", "单元", "部件", "部分", "子系统"}, 5},
    {"文件",  {"文档", "档案", "记录", "资料"}, 4},
    {"代码",  {"程序", "脚本", "源码", "指令"}, 4},
    {"错误",  {"故障", "异常", "bug", "缺陷", "问题"}, 5},
    {"学习",  {"训练", "习得", "掌握", "获取"}, 4},
    {"模型",  {"网络", "框架", "结构", "架构"}, 4},
    {"算法",  {"方法", "策略", "方案", "机制"}, 4},
    {"分析",  {"解析", "检查", "诊断", "评估"}, 4},
    {"处理",  {"操作", "加工", "执行", "管理"}, 4},
    {"推理",  {"推断", "演绎", "推导", "归结"}, 4},
    {"规划",  {"计划", "安排", "调度", "设计"}, 4},
    {"控制",  {"管理", "调节", "操纵", "指挥"}, 4},
    {"搜索",  {"检索", "查找", "查询", "探寻"}, 4},
    {"识别",  {"辨认", "鉴别", "区分", "检测"}, 4},
    {"优化",  {"改进", "提升", "增强", "完善"}, 4},
    {"检测",  {"检查", "探测", "监控", "扫描"}, 4},
    {"执行",  {"运行", "实施", "操作", "完成"}, 4},
    {NULL, {NULL}, 0}
};

/* ============================================================================
 * 内部辅助: UTF-8字符字节长度判断
 * ============================================================================ */
static int mm_utf8_char_len(unsigned char first_byte) {
    if ((first_byte & 0x80) == 0) return 1;
    if ((first_byte & 0xE0) == 0xC0) return 2;
    if ((first_byte & 0xF0) == 0xE0) return 3;
    if ((first_byte & 0xF8) == 0xF0) return 4;
    return 1;
}

/* ============================================================================
 * 内部辅助: 中英文混合分词器（增强版: unigram + bigram + trigram）
 *
 * 对中文按单字→双字→三字组合分词，英文按空格/标点分词。
 * 支持UTF-8多字节编码。返回词条数组和词数。
 * ============================================================================ */
static int mm_tokenize_mixed(const char* text, 
                              char terms[][MM_MAX_TERM_LEN], 
                              int max_terms) {
    if (!text || max_terms <= 0) return 0;
    
    int count = 0;
    int text_len = (int)strlen(text);
    int i = 0;
    
    while (i < text_len && count < max_terms) {
        unsigned char c = (unsigned char)text[i];
        
        /* 跳过空白和标点 */
        if (c <= 0x20 || c == ',' || c == '.' || c == '!' || c == '?' || 
            c == ';' || c == ':' || c == '"' || c == '\'' ||
            c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}') {
            i++;
            continue;
        }
        
        if (c >= 0x80) {
            /* UTF-8多字节字符（中文等） */
            int char_len = mm_utf8_char_len(c);
            if (char_len < 1 || char_len > 4) { i++; continue; }
            
            /* 收集连续中文字符序列 */
            int start_i = i;
            int chinese_count = 0;
            while (i < text_len && chinese_count < 6) {
                unsigned char nc = (unsigned char)text[i];
                if (nc >= 0x80) {
                    int cl = mm_utf8_char_len(nc);
                    if (i + cl > text_len) break;
                    chinese_count++;
                    i += cl;
                } else if (nc == ' ' || nc == '\t' || nc == '\n') {
                    break;
                } else {
                    break;
                }
            }
            
            int seq_len = i - start_i;
            if (seq_len > 0) {
                /* 产出单字（unigram） */
                int pos = start_i;
                while (pos < i) {
                    int cl = mm_utf8_char_len((unsigned char)text[pos]);
                    if (pos + cl > i) break;
                    if (count < max_terms && cl < MM_MAX_TERM_LEN) {
                        memcpy(terms[count], text + pos, (size_t)cl);
                        terms[count][cl] = '\0';
                        count++;
                    }
                    pos += cl;
                }
                /* 产出双字（bigram） */
                if (chinese_count >= 2) {
                    pos = start_i;
                    while (pos < i && count < max_terms) {
                        int c1 = mm_utf8_char_len((unsigned char)text[pos]);
                        if (pos + c1 >= i) break;
                        int c2 = mm_utf8_char_len((unsigned char)text[pos + c1]);
                        int bigram_len = c1 + c2;
                        if (bigram_len < MM_MAX_TERM_LEN) {
                            memcpy(terms[count], text + pos, (size_t)bigram_len);
                            terms[count][bigram_len] = '\0';
                            count++;
                        }
                        pos += c1;
                    }
                }
                /* 产出三字（trigram） */
                if (chinese_count >= 3) {
                    pos = start_i;
                    while (pos < i && count < max_terms) {
                        int c1 = mm_utf8_char_len((unsigned char)text[pos]);
                        if (pos + c1 >= i) break;
                        int c2_i = pos + c1;
                        if (c2_i >= i) break;
                        int c2 = mm_utf8_char_len((unsigned char)text[c2_i]);
                        if (c2_i + c2 >= i) break;
                        int trigram_len = c1 + c2 + mm_utf8_char_len((unsigned char)text[c2_i + c2]);
                        if (trigram_len < MM_MAX_TERM_LEN && c2_i + c2 + mm_utf8_char_len((unsigned char)text[c2_i + c2]) <= i) {
                            memcpy(terms[count], text + pos, (size_t)trigram_len);
                            terms[count][trigram_len] = '\0';
                            count++;
                        }
                        pos += c1;
                    }
                }
            }
        } else {
            /* ASCII字母数字 */
            int start_i = i;
            while (i < text_len) {
                unsigned char nc = (unsigned char)text[i];
                if ((nc >= 'a' && nc <= 'z') || (nc >= 'A' && nc <= 'Z') ||
                    (nc >= '0' && nc <= '9') || nc == '_' || nc == '-') {
                    i++;
                } else {
                    break;
                }
            }
            int word_len = i - start_i;
            if (word_len > 0 && word_len < MM_MAX_TERM_LEN && count < max_terms) {
                memcpy(terms[count], text + start_i, (size_t)word_len);
                terms[count][word_len] = '\0';
                /* 英文转小写 */
                for (int j = 0; j < word_len; j++) {
                    if (terms[count][j] >= 'A' && terms[count][j] <= 'Z')
                        terms[count][j] += 32;
                }
                count++;
            }
        }
    }
    
    return count;
}

/* ============================================================================
 * 内部辅助: Levenshtein编辑距离（O(n*m)动态规划）
 * 用于mm_word2vec_similarity回退路径
 * ============================================================================ */
static int mm_levenshtein_distance(const char* s1, const char* s2) {
    if (!s1 && !s2) return 0;
    if (!s1) return (int)strlen(s2);
    if (!s2) return (int)strlen(s1);
    
    int len1 = (int)strlen(s1);
    int len2 = (int)strlen(s2);
    
    if (len1 == 0) return len2;
    if (len2 == 0) return len1;
    
    /* 使用两行动态规划，O(min(n,m))空间 */
    int* prev = (int*)safe_malloc((len2 + 1) * sizeof(int));
    int* curr = (int*)safe_malloc((len2 + 1) * sizeof(int));
    if (!prev || !curr) {
        safe_free((void**)&prev);
        safe_free((void**)&curr);
        /* 回退: 绝对长度差 */
        return abs(len1 - len2);
    }
    
    for (int j = 0; j <= len2; j++) prev[j] = j;
    
    for (int i = 1; i <= len1; i++) {
        curr[0] = i;
        for (int j = 1; j <= len2; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            int min_val = prev[j] + 1;
            if (curr[j-1] + 1 < min_val) min_val = curr[j-1] + 1;
            if (prev[j-1] + cost < min_val) min_val = prev[j-1] + cost;
            curr[j] = min_val;
        }
        int* tmp = prev;
        prev = curr;
        curr = tmp;
    }
    
    int dist = prev[len2];
    safe_free((void**)&prev);
    safe_free((void**)&curr);
    return dist;
}

/* ============================================================================
 * 内部辅助: 安全获取知识库条目数（不透明KnowledgeBase句柄）
 * ============================================================================ */
static size_t mm_kb_entry_count(KnowledgeBase* kb) {
    size_t total = 0;
    if (!kb) return 0;
    knowledge_base_get_stats(kb, &total, NULL);
    return total;
}

/* ============================================================================
 * 内部辅助: 从知识库中统计某词出现的文档数（用于IDF计算）
 * ============================================================================ */
static int mm_count_docs_with_term(KnowledgeBase* kb, const char* term) {
    if (!kb || !term) return 0;
    
    size_t total_entries = 0;
    if (knowledge_base_get_stats(kb, &total_entries, NULL) != 0) return 0;
    if (total_entries == 0) return 0;
    
    /* 使用知识库查询API获取所有概念类条目 */
    KnowledgeQuery query;
    memset(&query, 0, sizeof(KnowledgeQuery));
    query.type_filter = KNOWLEDGE_CONCEPT;
    
    size_t query_limit = total_entries < 4096 ? total_entries : 4096;
    KnowledgeEntry* results = (KnowledgeEntry*)safe_calloc(query_limit, sizeof(KnowledgeEntry));
    if (!results) return 0;
    
    int found = knowledge_base_query(kb, &query, results, (size_t)(query_limit > 0 ? query_limit : 1));
    if (found < 0) {
        safe_free((void**)&results);
        /* 回退：查询所有类型 */
        query.type_filter = -1;
        found = knowledge_base_query(kb, &query, results, (size_t)(query_limit > 0 ? query_limit : 1));
    }
    
    int doc_count = 0;
    for (int i = 0; i < found && i < (int)query_limit; i++) {
        const char* subj = results[i].subject ? results[i].subject : "";
        const char* obj = results[i].object ? results[i].object : "";
        if (strstr(subj, term) || strstr(obj, term)) {
            doc_count++;
        }
    }
    
    safe_free((void**)&results);
    return doc_count;
}

/* ============================================================================
 * mm_compute_true_tfidf — 真实TF-IDF计算
 *
 * 对输入文本进行分词，统计每个词的词频并计算TF-IDF加权向量。
 * TF = 词在文档中的出现次数 / 文档总词数
 * IDF = log((总文档数 + 1) / (包含该词的文档数 + 1)) + 1 (Laplace平滑)
 * TF-IDF = TF × IDF
 *
 * @param doc_text     输入文档文本
 * @param kb           知识库指针（用于统计IDF，可为NULL则回退到高频词表）
 * @param features     输出的TF-IDF加权特征向量
 * @param max_features 特征向量最大维度
 * @param keywords_out 提取的关键词输出（可为NULL）
 * @param scores_out   关键词TF-IDF分数（可为NULL）
 * @param max_keywords 最大关键词数量
 * @return 成功返回实际特征维度，失败返回-1
 * ============================================================================ */
static int mm_compute_true_tfidf(const char* doc_text,
                                  KnowledgeBase* kb,
                                  float* features, size_t max_features,
                                  char keywords_out[][MM_MAX_TERM_LEN],
                                  float* scores_out,
                                  int max_keywords) {
    if (!doc_text || !features || max_features == 0) return -1;
    if (doc_text[0] == '\0') {
        memset(features, 0, max_features * sizeof(float));
        return 0;
    }
    
    /* 阶段1: 分词 */
    char raw_terms[MM_MAX_TERMS][MM_MAX_TERM_LEN];
    int raw_count = mm_tokenize_mixed(doc_text, raw_terms, MM_MAX_TERMS);
    if (raw_count == 0) {
        memset(features, 0, max_features * sizeof(float));
        return 0;
    }
    
    /* 阶段2: 词频统计 — 合并重复词并计数 */
    typedef struct {
        char term[MM_MAX_TERM_LEN];
        int freq;
        float tf;
    } TermFreq;
    
    TermFreq term_freqs[MM_MAX_TERMS];
    int tf_count = 0;
    
    for (int ti = 0; ti < raw_count; ti++) {
        if (raw_terms[ti][0] == '\0') continue;
        
        int found = 0;
        for (int ui = 0; ui < tf_count; ui++) {
            if (strcmp(term_freqs[ui].term, raw_terms[ti]) == 0) {
                term_freqs[ui].freq++;
                found = 1;
                break;
            }
        }
        if (!found && tf_count < MM_MAX_TERMS) {
            strncpy(term_freqs[tf_count].term, raw_terms[ti], MM_MAX_TERM_LEN - 1);
            term_freqs[tf_count].term[MM_MAX_TERM_LEN - 1] = '\0';
            term_freqs[tf_count].freq = 1;
            term_freqs[tf_count].tf = 0.0f;
            tf_count++;
        }
    }
    
    /* 阶段3: 计算真实TF = freq / total_raw_count */
    float inv_total = 1.0f / (float)raw_count;
    for (int i = 0; i < tf_count; i++) {
        term_freqs[i].tf = (float)term_freqs[i].freq * inv_total;
    }
    
    /* 阶段4: 计算IDF — 从知识库统计，或回退到高频词表估算IDF */
    float* idf_values = (float*)safe_calloc((size_t)tf_count, sizeof(float));
    if (!idf_values) return -1;
    
    float total_docs = (mm_kb_entry_count(kb) > 0) ? (float)mm_kb_entry_count(kb) : 100.0f;
    
    for (int i = 0; i < tf_count; i++) {
        float doc_freq_with_term;
        
        if (mm_kb_entry_count(kb) > 0) {
            /* 从知识库统计真实DF */
            int df = mm_count_docs_with_term(kb, term_freqs[i].term);
            doc_freq_with_term = (float)(df > 0 ? df : 1);
        } else {
            /* 回退: 从高频词表估算 */
            int in_high_freq = 0;
            for (int h = 0; h < mm_high_freq_count; h++) {
                if (strcmp(term_freqs[i].term, mm_high_freq_words[h]) == 0) {
                    in_high_freq = 1;
                    break;
                }
            }
            /* 高频词假设出现在80%文档中，低频词假设出现在5%文档中 */
            doc_freq_with_term = in_high_freq ? (total_docs * 0.8f) : (total_docs * 0.05f);
            if (doc_freq_with_term < 1.0f) doc_freq_with_term = 1.0f;
        }
        
        /* IDF = log((N + 1) / (df + 1)) + 1 (Laplace平滑) */
        idf_values[i] = logf((total_docs + MM_IDF_SMOOTH) / 
                             (doc_freq_with_term + MM_IDF_SMOOTH)) + 1.0f;
    }
    
    /* 阶段5: 计算TF-IDF分数并排序 */
    typedef struct {
        int term_index;
        float score;
    } KeywordScore;
    
    KeywordScore rankings[MM_MAX_TERMS];
    int rank_count = 0;
    
    for (int i = 0; i < tf_count && i < MM_MAX_TERMS; i++) {
        float tfidf_score = term_freqs[i].tf * idf_values[i];
        /* 词长度加权: 更长的词（如复合词）通常携带更多信息 */
        int term_len = (int)strlen(term_freqs[i].term);
        float length_bonus = 1.0f + 0.15f * (float)(term_len > 8 ? 8 : (term_len > 1 ? term_len - 1 : 0));
        tfidf_score *= length_bonus;
        
        rankings[rank_count].term_index = i;
        rankings[rank_count].score = tfidf_score;
        rank_count++;
    }
    
    /* 按TF-IDF分数降序排序（冒泡排序，项数有限） */
    for (int i = 0; i < rank_count - 1; i++) {
        for (int j = 0; j < rank_count - i - 1; j++) {
            if (rankings[j].score < rankings[j + 1].score) {
                KeywordScore tmp = rankings[j];
                rankings[j] = rankings[j + 1];
                rankings[j + 1] = tmp;
            }
        }
    }
    
    /* 阶段6: 输出TF-IDF特征向量与关键词 */
    int feature_idx = 0;
    int keyword_idx = 0;
    
    for (int i = 0; i < rank_count && feature_idx < (int)max_features; i++) {
        int ti = rankings[i].term_index;
        float score = rankings[i].score;
        
        /* 过滤低分噪声词 */
        if (score < 0.01f || term_freqs[ti].term[0] == '\0') continue;
        
        if (feature_idx < (int)max_features) {
            features[feature_idx++] = score;
        }
        
        /* 输出关键词 */
        if (keywords_out && keyword_idx < max_keywords) {
            strncpy(keywords_out[keyword_idx], term_freqs[ti].term, MM_MAX_TERM_LEN - 1);
            keywords_out[keyword_idx][MM_MAX_TERM_LEN - 1] = '\0';
            if (scores_out) scores_out[keyword_idx] = score;
            keyword_idx++;
        }
    }
    
    safe_free((void**)&idf_values);
    return feature_idx;
}

/* ============================================================================
 * 内部辅助: 从知识库构建词共现矩阵
 *
 * 对KB中每个文档，在滑动窗口内统计词对共现频率。
 * 返回全局词共现表（稀疏表示），用于语义相似度计算。
 *
 * @param kb             知识库
 * @param target_word    目标词
 * @param cooc_terms     共现词输出（词条文本）
 * @param cooc_freqs     共现词频率输出
 * @param max_cooc       最大共现词数量
 * @return 实际共现词数量
 * ============================================================================ */
static int mm_build_cooccurrence(KnowledgeBase* kb,
                                  const char* target_word,
                                  char cooc_terms[][MM_MAX_TERM_LEN],
                                  float* cooc_freqs,
                                  int max_cooc) {
    if (!kb || !target_word || !cooc_terms || !cooc_freqs || max_cooc <= 0) {
        return 0;
    }
    
    typedef struct {
        char term[MM_MAX_TERM_LEN];
        float freq;
    } CoocEntry;
    
    CoocEntry entries[MM_MAX_TERMS];
    int entry_count = 0;
    int total_target_occurrences = 0;
    int max_docs = (int)(mm_kb_entry_count(kb) < 2048 ? mm_kb_entry_count(kb) : 2048);
    
    /* 通过知识库API查询所有条目 */
    KnowledgeQuery cooc_query;
    memset(&cooc_query, 0, sizeof(KnowledgeQuery));
    KnowledgeEntry* cooc_results = (KnowledgeEntry*)safe_calloc((size_t)max_docs, sizeof(KnowledgeEntry));
    if (!cooc_results) return 0;
    int cooc_found = knowledge_base_query(kb, &cooc_query, cooc_results, (size_t)max_docs);
    int cooc_limit = (cooc_found > 0 && cooc_found <= max_docs) ? cooc_found : max_docs;
    
    for (int di = 0; di < cooc_limit; di++) {
        KnowledgeEntry* entry = &cooc_results[di];
        
        /* 合并subject和object作为文档文本 */
        char doc[8192] = {0};
        int pos = 0;
        if (entry->subject) {
            int sl = (int)strlen(entry->subject);
            if (sl > 0 && pos + sl + 1 < 8192) {
                memcpy(doc + pos, entry->subject, (size_t)sl);
                pos += sl;
                doc[pos++] = ' ';
            }
        }
        if (entry->object) {
            int ol = (int)strlen(entry->object);
            if (ol > 0 && pos + ol < 8192) {
                memcpy(doc + pos, entry->object, (size_t)ol);
                pos += ol;
            }
        }
        if (pos == 0) continue;
        
        /* 分词 */
        char terms[MM_MAX_TERMS][MM_MAX_TERM_LEN];
        int tc = mm_tokenize_mixed(doc, terms, MM_MAX_TERMS);
        if (tc < 2) continue;
        
        /* 在文档中查找目标词的所有出现位置 */
        int positions[MM_MAX_TERMS];
        int pos_count = 0;
        for (int t = 0; t < tc; t++) {
            if (strcmp(terms[t], target_word) == 0) {
                positions[pos_count++] = t;
            }
        }
        
        if (pos_count == 0) continue;
        total_target_occurrences += pos_count;
        
        /* 对每个目标词位置，在窗口内统计共现词 */
        for (int p = 0; p < pos_count; p++) {
            int center = positions[p];
            int win_start = center - MM_COOCCUR_WINDOW;
            int win_end = center + MM_COOCCUR_WINDOW;
            if (win_start < 0) win_start = 0;
            if (win_end >= tc) win_end = tc - 1;
            
            for (int w = win_start; w <= win_end; w++) {
                if (w == center) continue;
                if (terms[w][0] == '\0') continue;
                
                /* 跳过单字（信息量太低） */
                int wlen = (int)strlen(terms[w]);
                if (wlen < 2) continue;
                
                /* 累加或添加共现词 */
                int found = 0;
                for (int e = 0; e < entry_count; e++) {
                    if (strcmp(entries[e].term, terms[w]) == 0) {
                        /* 加权: 距离越近权重越大 */
                        float weight = 1.0f / (1.0f + (float)abs(w - center));
                        entries[e].freq += weight;
                        found = 1;
                        break;
                    }
                }
                if (!found && entry_count < MM_MAX_TERMS) {
                    strncpy(entries[entry_count].term, terms[w], MM_MAX_TERM_LEN - 1);
                    entries[entry_count].term[MM_MAX_TERM_LEN - 1] = '\0';
                    float weight = 1.0f / (1.0f + (float)abs(w - center));
                    entries[entry_count].freq = weight;
                    entry_count++;
                }
            }
        }
    }
    
    safe_free((void**)&cooc_results);
    
    /* 归一化共现频率 */
    if (total_target_occurrences > 0 && entry_count > 0) {
        float inv_total = 1.0f / (float)total_target_occurrences;
        for (int e = 0; e < entry_count; e++) {
            entries[e].freq *= inv_total;
        }
    }
    
    /* 按频率降序排序 */
    for (int i = 0; i < entry_count - 1; i++) {
        for (int j = 0; j < entry_count - i - 1; j++) {
            if (entries[j].freq < entries[j + 1].freq) {
                CoocEntry tmp = entries[j];
                entries[j] = entries[j + 1];
                entries[j + 1] = tmp;
            }
        }
    }
    
    /* 输出前max_cooc个 */
    int out_count = 0;
    for (int i = 0; i < entry_count && out_count < max_cooc; i++) {
        if (entries[i].freq < 0.01f) break;
        strncpy(cooc_terms[out_count], entries[i].term, MM_MAX_TERM_LEN - 1);
        cooc_terms[out_count][MM_MAX_TERM_LEN - 1] = '\0';
        cooc_freqs[out_count] = entries[i].freq;
        out_count++;
    }
    
    return out_count;
}

/* ============================================================================
 * mm_word2vec_similarity — 词共现语义相似度计算
 *
 * 基于知识库中的词共现矩阵计算两个词的语义相似度。
 * 使用余弦相似度在共现空间比较两个词的上下文分布。
 * 当知识库不可用或共现数据不足时，回退到编辑距离相似度。
 *
 * @param word1  词1
 * @param word2  词2
 * @param kb     知识库（可为NULL则直接使用编辑距离）
 * @return 相似度分数 [0.0, 1.0]
 * ============================================================================ */
static float mm_word2vec_similarity(const char* word1, const char* word2,
                                     KnowledgeBase* kb) {
    if (!word1 || !word2) return 0.0f;
    if (strcmp(word1, word2) == 0) return 1.0f;
    
    /* 当知识库可用时，基于共现矩阵计算语义相似度 */
    if (mm_kb_entry_count(kb) > 0) {
        char cooc1[MM_MAX_TERMS][MM_MAX_TERM_LEN];
        float freq1[MM_MAX_TERMS];
        int n1 = mm_build_cooccurrence(kb, word1, cooc1, freq1, 64);
        
        char cooc2[MM_MAX_TERMS][MM_MAX_TERM_LEN];
        float freq2[MM_MAX_TERMS];
        int n2 = mm_build_cooccurrence(kb, word2, cooc2, freq2, 64);
        
        if (n1 >= 3 && n2 >= 3) {
            /* 构建联合共现空间，计算余弦相似度 */
            typedef struct {
                char term[MM_MAX_TERM_LEN];
                float val1;
                float val2;
            } JointVec;
            
            JointVec joint[256];
            int joint_count = 0;
            
            /* 添加词1的共现向量 */
            for (int i = 0; i < n1 && joint_count < 256; i++) {
                strncpy(joint[joint_count].term, cooc1[i], MM_MAX_TERM_LEN - 1);
                joint[joint_count].term[MM_MAX_TERM_LEN - 1] = '\0';
                joint[joint_count].val1 = freq1[i];
                joint[joint_count].val2 = 0.0f;
                joint_count++;
            }
            
            /* 合并词2的共现向量 */
            for (int i = 0; i < n2; i++) {
                int found = 0;
                for (int j = 0; j < joint_count; j++) {
                    if (strcmp(joint[j].term, cooc2[i]) == 0) {
                        joint[j].val2 = freq2[i];
                        found = 1;
                        break;
                    }
                }
                if (!found && joint_count < 256) {
                    strncpy(joint[joint_count].term, cooc2[i], MM_MAX_TERM_LEN - 1);
                    joint[joint_count].term[MM_MAX_TERM_LEN - 1] = '\0';
                    joint[joint_count].val1 = 0.0f;
                    joint[joint_count].val2 = freq2[i];
                    joint_count++;
                }
            }
            
            /* 余弦相似度 */
            float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
            for (int i = 0; i < joint_count; i++) {
                dot += joint[i].val1 * joint[i].val2;
                norm1 += joint[i].val1 * joint[i].val1;
                norm2 += joint[i].val2 * joint[i].val2;
            }
            
            if (norm1 > 1e-8f && norm2 > 1e-8f) {
                float cos_sim = dot / (sqrtf(norm1) * sqrtf(norm2));
                if (cos_sim >= MM_SIM_THRESHOLD) return cos_sim;
            }
        }
        
        /* 共现数据不足时的次回退: 检查同义词表 */
        for (int s = 0; mm_synonym_table[s].word != NULL; s++) {
            if (strcmp(mm_synonym_table[s].word, word1) == 0) {
                for (int n = 0; n < mm_synonym_table[s].synonym_count; n++) {
                    if (strcmp(mm_synonym_table[s].synonyms[n], word2) == 0) {
                        return 0.85f; /* 同义词高相似度 */
                    }
                }
            }
            if (strcmp(mm_synonym_table[s].word, word2) == 0) {
                for (int n = 0; n < mm_synonym_table[s].synonym_count; n++) {
                    if (strcmp(mm_synonym_table[s].synonyms[n], word1) == 0) {
                        return 0.85f;
                    }
                }
            }
        }
    }
    
    /* 最终回退: 编辑距离相似度 */
    int dist = mm_levenshtein_distance(word1, word2);
    int max_len = (int)strlen(word1) > (int)strlen(word2) 
                  ? (int)strlen(word1) : (int)strlen(word2);
    if (max_len == 0) return 1.0f;
    
    float edit_sim = 1.0f - (float)dist / (float)max_len;
    if (edit_sim < 0.0f) edit_sim = 0.0f;
    
    return edit_sim;
}

/* ============================================================================
 * mm_semantic_query_expansion — 查询语义扩展
 *
 * 将用户原始查询扩展为语义相关的查询变体。
 * 基于三个维度:
 *   1. 同义词/近义词替换（查同义词表）
 *   2. 知识库中高频共现词补充
 *   3. 编辑距离相近词替换（用于纠错）
 *
 * @param query           原始查询文本
 * @param kb              知识库（可为NULL）
 * @param expanded_queries 扩展后查询变体输出数组（每个最大MM_MAX_TERM_LEN长度）
 * @param max_expansions  最大扩展数量
 * @return 实际扩展数量
 * ============================================================================ */
static int mm_semantic_query_expansion(const char* query,
                                        KnowledgeBase* kb,
                                        char expanded_queries[][MM_MAX_TERM_LEN],
                                        int max_expansions) {
    if (!query || !expanded_queries || max_expansions <= 0) return 0;
    
    int exp_count = 0;
    
    /* 1. 从同义词表扩展查询 */
    /* 首先分词原始查询 */
    char query_terms[64][MM_MAX_TERM_LEN];
    int qt_count = mm_tokenize_mixed(query, query_terms, 64);
    if (qt_count == 0) return 0;
    
    /* 对查询中的每个词查找同义词 */
    typedef struct {
        char term[MM_MAX_TERM_LEN];
        char synonyms[8][MM_MAX_TERM_LEN];
        int syn_count;
    } TermExpansion;
    
    TermExpansion expansions[64];
    int exp_idx = 0;
    
    for (int ti = 0; ti < qt_count && ti < 64; ti++) {
        /* 跳过单字 */
        if (strlen(query_terms[ti]) < 2) continue;
        
        /* 查同义词表 */
        for (int s = 0; mm_synonym_table[s].word != NULL; s++) {
            if (strcmp(mm_synonym_table[s].word, query_terms[ti]) == 0) {
                strncpy(expansions[exp_idx].term, query_terms[ti], MM_MAX_TERM_LEN - 1);
                expansions[exp_idx].term[MM_MAX_TERM_LEN - 1] = '\0';
                expansions[exp_idx].syn_count = 0;
                
                for (int sn = 0; sn < mm_synonym_table[s].synonym_count && 
                     expansions[exp_idx].syn_count < 8; sn++) {
                    strncpy(expansions[exp_idx].synonyms[expansions[exp_idx].syn_count],
                            mm_synonym_table[s].synonyms[sn], MM_MAX_TERM_LEN - 1);
                    expansions[exp_idx].synonyms[expansions[exp_idx].syn_count][MM_MAX_TERM_LEN - 1] = '\0';
                    expansions[exp_idx].syn_count++;
                }
                exp_idx++;
                break;
            }
        }
    }
    
    /* 生成扩展查询变体（替换每个可扩展词的近义词） */
    for (int ei = 0; ei < exp_idx && exp_count < max_expansions; ei++) {
        for (int si = 0; si < expansions[ei].syn_count && exp_count < max_expansions; si++) {
            /* 构造扩展查询: 替换原词为近义词 */
            int buf_pos = 0;
            const char* qp = query;
            
            /* 找到原词在query中的位置并替换 */
            const char* found = strstr(qp, expansions[ei].term);
            if (found) {
                /* 复制前缀 */
                int prefix_len = (int)(found - qp);
                if (prefix_len > 0 && buf_pos + prefix_len < MM_MAX_TERM_LEN) {
                    memcpy(expanded_queries[exp_count] + buf_pos, qp, (size_t)prefix_len);
                    buf_pos += prefix_len;
                }
                /* 替换为同义词 */
                int syn_len = (int)strlen(expansions[ei].synonyms[si]);
                if (buf_pos + syn_len < MM_MAX_TERM_LEN) {
                    memcpy(expanded_queries[exp_count] + buf_pos, 
                           expansions[ei].synonyms[si], (size_t)syn_len);
                    buf_pos += syn_len;
                }
                /* 复制后缀 */
                const char* suffix = found + strlen(expansions[ei].term);
                int suffix_len = (int)strlen(suffix);
                if (suffix_len > 0 && buf_pos + suffix_len < MM_MAX_TERM_LEN) {
                    memcpy(expanded_queries[exp_count] + buf_pos, suffix, (size_t)suffix_len);
                    buf_pos += suffix_len;
                }
                expanded_queries[exp_count][buf_pos] = '\0';
                exp_count++;
            }
        }
    }
    
    /* 2. 按知识库中的高频共现词扩展 */
    if (mm_kb_entry_count(kb) > 0 && exp_count < max_expansions) {
        for (int ti = 0; ti < qt_count && ti < 64 && exp_count < max_expansions; ti++) {
            if (strlen(query_terms[ti]) < 2) continue;
            
            char cooc_terms[16][MM_MAX_TERM_LEN];
            float cooc_freqs[16];
            int cn = mm_build_cooccurrence(kb, query_terms[ti], cooc_terms, cooc_freqs, 16);
            
            /* 取前3个最高频共现词追加到查询末尾 */
            for (int ci = 0; ci < cn && ci < 3 && exp_count < max_expansions; ci++) {
                /* 避免添加已在查询中的词 */
                int already_in_query = 0;
                for (int qc = 0; qc < qt_count; qc++) {
                    if (strcmp(query_terms[qc], cooc_terms[ci]) == 0) {
                        already_in_query = 1;
                        break;
                    }
                }
                if (!already_in_query) {
                    /* 构造"原查询 + 共现词"的扩展查询 */
                    int qlen = (int)strlen(query);
                    int clen = (int)strlen(cooc_terms[ci]);
                    if (qlen + clen + 2 < MM_MAX_TERM_LEN) {
                        int buf_pos = 0;
                        memcpy(expanded_queries[exp_count] + buf_pos, query, (size_t)qlen);
                        buf_pos += qlen;
                        expanded_queries[exp_count][buf_pos++] = ' ';
                        memcpy(expanded_queries[exp_count] + buf_pos, cooc_terms[ci], (size_t)clen);
                        buf_pos += clen;
                        expanded_queries[exp_count][buf_pos] = '\0';
                        exp_count++;
                    }
                }
            }
        }
    }
    
    return exp_count;
}

/* ============================================================================
 * mm_extract_keywords_from_query — 提取查询关键词并返回TF-IDF特征向量
 *
 * P1-003修复核心函数: 使用真实TF-IDF替代硬编码高频词表。
 *   TF = 词频 / 查询总词数（当前文档内的真实频率）
 *   IDF = log(总文档数 / (包含该词的文档数 + 1)) + 1（基于知识库文档统计）
 * 分词使用增强版中英文混合分词（unigram + bigram + trigram）
 * 保留高频词表作为回退（当知识库不可用时）
 *
 * @param query       查询文本（UTF-8中英文混合）
 * @param kb          知识库指针（用于IDF统计，可为NULL回退到高频词表）
 * @param features    输出的TF-IDF加权特征向量
 * @param max_features 特征向量最大长度
 * @return 成功返回实际特征维度，失败返回-1
 * ============================================================================ */
int mm_extract_keywords_from_query(const char* query,
                                    KnowledgeBase* kb,
                                    float* features, size_t max_features) {
    if (!query || !features || max_features == 0) return -1;
    
    /* 调用真实TF-IDF计算 */
    int feature_count = mm_compute_true_tfidf(query, kb,
                                               features, max_features,
                                               NULL, NULL, 0);
    
    /* 如果TF-IDF提取的特征过少（<3个），使用高频词表回退增强 */
    if (feature_count < 3 && feature_count >= 0) {
        int remaining = (int)max_features - feature_count;
        if (remaining > 0) {
            /* 从高频词表中匹配查询中出现的词 */
            int added = 0;
            for (int h = 0; h < mm_high_freq_count && added < remaining && 
                 feature_count + added < (int)max_features; h++) {
                const char* hword = mm_high_freq_words[h];
                if (!hword || hword[0] == '\0') continue;
                
                /* 检查高频词是否出现在查询中 */
                if (strstr(query, hword)) {
                    /* 检查是否已通过TF-IDF提取（避免重复） */
                    int already_extracted = 0;
                    for (int fi = 0; fi < feature_count; fi++) {
                        if (features[fi] > 0.5f) {
                            already_extracted = 1;
                            break;
                        }
                    }
                    if (already_extracted) continue;
                    
                    /* 用退火衰减因子：越晚添加的高频词分数越低 */
                    float fallback_score = 0.15f / (float)(added + 1);
                    features[feature_count + added] = fallback_score;
                    added++;
                }
            }
            feature_count += added;
        }
    }
    
    /* 归一化输出特征向量（L2归一化） */
    if (feature_count > 0) {
        float norm = 0.0f;
        for (int i = 0; i < feature_count; i++) {
            norm += features[i] * features[i];
        }
        if (norm > 1e-8f) {
            float inv_norm = 1.0f / sqrtf(norm);
            for (int i = 0; i < feature_count; i++) {
                features[i] *= inv_norm;
            }
        }
    }
    
    return feature_count;
}
