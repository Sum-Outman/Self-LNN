/**
 * @file environment_sound.c
 * @brief K-034: 环境声音CfC液态分类器
 *
 * 通过CfC液态神经网络对常见环境声音进行分类：
 * 火灾报警、玻璃破碎、婴儿哭声、狗叫、枪声、门铃、雷声、警报等。
 *
 * 架构: 音频波形 → 梅尔频谱图 → CfC ODE演化 → softmax分类
 * 100%纯C实现。
 */

#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/errors.h"
#include "selflnn/multimodal/environment_sound.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define ESC_MAX_CLASSES 16
#define ESC_FREQ_BINS 64
#define ESC_TIME_FRAMES 32
#define ESC_HIDDEN_DIM 128
#define ESC_CFC_TIMESTEPS 8

/* 权重文件魔数 "ESSW" = 0x45535357 (Environment Sound System Weights) */
#define ESC_WEIGHTS_MAGIC 0x45535357U
#define ESC_WEIGHTS_VERSION 1U
#define ESC_DEFAULT_WEIGHTS_PATH "env_sound_weights.bin"

static const char* g_esc_class_names[ESC_MAX_CLASSES] = {
    "火灾报警", "玻璃破碎", "婴儿哭声", "狗叫",
    "枪声", "门铃", "雷声", "警报",
    "引擎声", "水流声", "风声", "脚步声",
    "鸟鸣", "猫叫", "键盘敲击", "静音"
};

typedef struct {
    float mel_filters[ESC_FREQ_BINS * 128];
    float cfc_w[ESC_HIDDEN_DIM * ESC_HIDDEN_DIM];
    float cfc_b[ESC_HIDDEN_DIM];
    float fc_w[ESC_MAX_CLASSES * ESC_HIDDEN_DIM];
    float fc_b[ESC_MAX_CLASSES];
    int num_classes;
    int initialized;
    int is_trained; /* 标记是否已加载训练好的权重 */
} EnvironmentSoundClassifier;

static EnvironmentSoundClassifier* g_esc = NULL;

/**
 * @brief 初始化三角梅尔滤波器组
 */
static void esc_build_mel_filters(EnvironmentSoundClassifier* esc) {
    /* 80Hz-8000Hz → 64 bins on Mel scale */
    float mel_low = 1127.0f * logf(1.0f + 80.0f / 700.0f);
    float mel_high = 1127.0f * logf(1.0f + 8000.0f / 700.0f);
    float mel_step = (mel_high - mel_low) / (ESC_FREQ_BINS + 1);

    for (int m = 0; m < ESC_FREQ_BINS; m++) {
        float mel_center = mel_low + (m + 1) * mel_step;
        float freq_center = 700.0f * (expf(mel_center / 1127.0f) - 1.0f);

        for (int f = 0; f < 128; f++) {
            float freq = f * 8000.0f / 128.0f;
            float mel_f = 1127.0f * logf(1.0f + freq / 700.0f);

            /* 三角滤波器 */
            float left = mel_low + m * mel_step;
            float right = mel_low + (m + 2) * mel_step;

            if (mel_f >= left && mel_f <= mel_center) {
                esc->mel_filters[m * 128 + f] = (mel_f - left) / (mel_center - left + 1e-8f);
            } else if (mel_f > mel_center && mel_f <= right) {
                esc->mel_filters[m * 128 + f] = (right - mel_f) / (right - mel_center + 1e-8f);
            }
        }
    }
}

/**
 * @brief K-034: 创建环境声音CfC分类器
 */
void* environment_sound_classifier_create(int num_classes) {
    if (num_classes <= 0 || num_classes > ESC_MAX_CLASSES) return NULL;

    EnvironmentSoundClassifier* esc = (EnvironmentSoundClassifier*)
        safe_calloc(1, sizeof(EnvironmentSoundClassifier));
    if (!esc) return NULL;

    esc->num_classes = num_classes;
    esc->is_trained = 0;

/* 先构建梅尔滤波器组（滤波器不依赖训练数据） */
    esc_build_mel_filters(esc);

/* 优先尝试加载预训练权重，加载成功则跳过随机初始化 */
    if (env_sound_load_weights(esc, ESC_DEFAULT_WEIGHTS_PATH) == 0) {
        esc->is_trained = 1;
    } else {
        /* 权重文件不存在或校验失败，回退到随机初始化（标记为未训练）
         * esc->is_trained = 0 确保调用方能区分已训练和未训练的模型状态 */
        log_debug("[环境声音] 预训练权重加载失败，使用随机初始化(is_trained=0)");
        for (int i = 0; i < ESC_HIDDEN_DIM * ESC_HIDDEN_DIM; i++) {
            esc->cfc_w[i] = (secure_random_float() - 0.5f) * 0.1f;
        }
        memset(esc->cfc_b, 0, sizeof(esc->cfc_b));
        for (int i = 0; i < ESC_MAX_CLASSES * ESC_HIDDEN_DIM; i++) {
            esc->fc_w[i] = (secure_random_float() - 0.5f) * 0.1f;
        }
        memset(esc->fc_b, 0, sizeof(esc->fc_b));
    }

    esc->initialized = 1;

    g_esc = esc;
    return esc;
}

/**
 * @brief K-034: 分类环境声音片段
 *
 * @param classifier 分类器句柄
 * @param audio_samples 音频采样值 [-1,1]
 * @param num_samples 采样数
 * @param sample_rate 采样率(Hz)
 * @param class_name [输出] 分类名称缓冲区(64字节)
 * @param confidence [输出] 置信度
 * @return 类别索引，-1错误
 */
int environment_sound_classify(void* classifier,
                                const float* audio_samples, int num_samples,
                                int sample_rate,
                                char* class_name, int max_name_len,
                                float* confidence) {
    EnvironmentSoundClassifier* esc = (EnvironmentSoundClassifier*)classifier;
    if (!esc || !audio_samples || num_samples <= 0) return -1;

/* 未训练时返回低置信度结果，避免使用随机权重产生虚假预测 */
    if (!esc->is_trained) {
        if (class_name && max_name_len > 0) {
            snprintf(class_name, max_name_len, "未训练");
        }
        if (confidence) *confidence = 0.001f;
        return 0;
    }

    /* 帧大小: 16ms at 16kHz = 256 samples */
    int frame_size = sample_rate * 16 / 1000;
    int hop_size = frame_size / 2;
    if (frame_size <= 0) frame_size = 256;

    /* 计算梅尔频谱: 对每帧做FFT→Mel滤波 */
    float mel_features[ESC_TIME_FRAMES * ESC_FREQ_BINS];
    memset(mel_features, 0, sizeof(mel_features));

    int num_frames = 0;
    for (int f = 0; f < num_samples - frame_size && num_frames < ESC_TIME_FRAMES;
         f += hop_size) {
        /* BUG-007修复: 使用完整基2-FFT替代Goertzel单频近似
         * Radix-2 Cooley-Tukey FFT，输入帧长度补齐到2的幂 */
        int fft_n = 256;
        /* 确保fft_n是2的幂且不小于frame_size */
        while (fft_n < frame_size) fft_n *= 2;
        float real[256], imag[256];
        memset(real, 0, sizeof(real));
        memset(imag, 0, sizeof(imag));
        /* 复制输入并应用Hann窗 */
        for (int i = 0; i < frame_size && i < fft_n; i++) {
            float w = 0.5f - 0.5f * cosf(2.0f * 3.14159265f * i / (frame_size - 1));
            real[i] = audio_samples[f + i] * w;
        }
        /* M-024修复：使用统一fft_real替代内联基2-FFT */
        {
            float* input_saved = (float*)safe_malloc(fft_n * sizeof(float));
            if (input_saved) {
                memcpy(input_saved, real, fft_n * sizeof(float));
                fft_real(input_saved, fft_n, real, imag);
                safe_free((void**)&input_saved);
            }
        }
        /* 获取幅度谱 */
        float spectrum[128] = {0};
        for (int bin = 0; bin < 128 && bin < fft_n / 2; bin++) {
            spectrum[bin] = sqrtf(real[bin] * real[bin] + imag[bin] * imag[bin]) / fft_n;
        }

        /* Mel滤波 */
        for (int m = 0; m < ESC_FREQ_BINS; m++) {
            float mel_val = 0;
            for (int b = 0; b < 128; b++) {
                mel_val += spectrum[b] * esc->mel_filters[m * 128 + b];
            }
            mel_features[num_frames * ESC_FREQ_BINS + m] = logf(mel_val + 1e-8f);
        }
        num_frames++;
    }

    if (num_frames == 0) return -1;

    /* CfC ODE时序演化 */
    float hidden[ESC_HIDDEN_DIM];
    float cell[ESC_HIDDEN_DIM];
    memset(hidden, 0, sizeof(hidden));
    memset(cell, 0, sizeof(cell));

    for (int t = 0; t < num_frames && t < ESC_CFC_TIMESTEPS; t++) {
        float* input = &mel_features[t * ESC_FREQ_BINS];

        /* CfC一步: dh/dt = W·h + b + W_in·x */
        for (int i = 0; i < ESC_HIDDEN_DIM; i++) {
            float sum = esc->cfc_b[i];
            for (int j = 0; j < ESC_HIDDEN_DIM; j++) {
                sum += esc->cfc_w[i * ESC_HIDDEN_DIM + j] * hidden[j];
            }
            /* 输入投影: 使用前64维 */
            if (i < ESC_FREQ_BINS) sum += input[i] * 0.1f;
            cell[i] = cell[i] * 0.9f + sum * 0.1f;
            hidden[i] = tanhf(cell[i]);
        }
    }

    /* FC分类层 */
    float logits[ESC_MAX_CLASSES] = {0};
    float max_logit = -1e30f;
    int best_class = 0;
    for (int c = 0; c < esc->num_classes; c++) {
        logits[c] = esc->fc_b[c];
        for (int i = 0; i < ESC_HIDDEN_DIM; i++) {
            logits[c] += esc->fc_w[c * ESC_HIDDEN_DIM + i] * hidden[i];
        }
        if (logits[c] > max_logit) { max_logit = logits[c]; best_class = c; }
    }

    /* Softmax */
    float exp_sum = 0;
    for (int c = 0; c < esc->num_classes; c++) {
        logits[c] = expf(logits[c] - max_logit);
        exp_sum += logits[c];
    }
    float best_conf = (exp_sum > 1e-10f) ? logits[best_class] / exp_sum : 0.0f;

    if (class_name && max_name_len > 0) {
        snprintf(class_name, max_name_len, "%s", g_esc_class_names[best_class]);
    }
    if (confidence) *confidence = best_conf;

    return best_class;
}

/* ======================================================================== */
/* 环境声音权重保存与加载                                      */
/* 权重文件格式：魔数0x45535357("ESSW") + 版本号(uint32_t) + 类别数(int)   */
/* + mel_filters(ESC_FREQ_BINS*128) + cfc_w(ESC_HIDDEN_DIM*ESC_HIDDEN_DIM)  */
/* + cfc_b(ESC_HIDDEN_DIM) + fc_w(ESC_MAX_CLASSES*ESC_HIDDEN_DIM)          */
/* + fc_b(ESC_MAX_CLASSES)                                                  */
/* ======================================================================== */

int env_sound_save_weights(void* classifier, const char* filepath) {
    if (!classifier || !filepath) return -1;
    EnvironmentSoundClassifier* esc = (EnvironmentSoundClassifier*)classifier;

    FILE* fp = fopen(filepath, "wb");
    if (!fp) return -1;

    uint32_t magic = ESC_WEIGHTS_MAGIC;
    uint32_t version = ESC_WEIGHTS_VERSION;
    int num_classes = esc->num_classes;

    if (fwrite(&magic, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return -1; }
    if (fwrite(&version, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); return -1; }
    if (fwrite(&num_classes, sizeof(int), 1, fp) != 1) { fclose(fp); return -1; }

    if (fwrite(esc->mel_filters, sizeof(float), ESC_FREQ_BINS * 128, fp) != (size_t)(ESC_FREQ_BINS * 128)) { fclose(fp); return -1; }
    if (fwrite(esc->cfc_w, sizeof(float), ESC_HIDDEN_DIM * ESC_HIDDEN_DIM, fp) != (size_t)(ESC_HIDDEN_DIM * ESC_HIDDEN_DIM)) { fclose(fp); return -1; }
    if (fwrite(esc->cfc_b, sizeof(float), ESC_HIDDEN_DIM, fp) != (size_t)ESC_HIDDEN_DIM) { fclose(fp); return -1; }
    if (fwrite(esc->fc_w, sizeof(float), ESC_MAX_CLASSES * ESC_HIDDEN_DIM, fp) != (size_t)(ESC_MAX_CLASSES * ESC_HIDDEN_DIM)) { fclose(fp); return -1; }
    if (fwrite(esc->fc_b, sizeof(float), ESC_MAX_CLASSES, fp) != (size_t)ESC_MAX_CLASSES) { fclose(fp); return -1; }

    fclose(fp);
    return 0;
}

int env_sound_load_weights(void* classifier, const char* filepath) {
    if (!classifier || !filepath) return -1;
    EnvironmentSoundClassifier* esc = (EnvironmentSoundClassifier*)classifier;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    uint32_t magic = 0;
    uint32_t version = 0;
    int loaded_num_classes = 0;

    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 || magic != ESC_WEIGHTS_MAGIC) {
        fclose(fp);
        return -1;
    }
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1 || version != ESC_WEIGHTS_VERSION) {
        fclose(fp);
        return -1;
    }
    if (fread(&loaded_num_classes, sizeof(int), 1, fp) != 1 || loaded_num_classes != esc->num_classes) {
        fclose(fp);
        return -1;
    }

    if (fread(esc->mel_filters, sizeof(float), ESC_FREQ_BINS * 128, fp) != (size_t)(ESC_FREQ_BINS * 128)) { fclose(fp); return -1; }
    if (fread(esc->cfc_w, sizeof(float), ESC_HIDDEN_DIM * ESC_HIDDEN_DIM, fp) != (size_t)(ESC_HIDDEN_DIM * ESC_HIDDEN_DIM)) { fclose(fp); return -1; }
    if (fread(esc->cfc_b, sizeof(float), ESC_HIDDEN_DIM, fp) != (size_t)ESC_HIDDEN_DIM) { fclose(fp); return -1; }
    if (fread(esc->fc_w, sizeof(float), ESC_MAX_CLASSES * ESC_HIDDEN_DIM, fp) != (size_t)(ESC_MAX_CLASSES * ESC_HIDDEN_DIM)) { fclose(fp); return -1; }
    if (fread(esc->fc_b, sizeof(float), ESC_MAX_CLASSES, fp) != (size_t)ESC_MAX_CLASSES) { fclose(fp); return -1; }

    fclose(fp);
    esc->is_trained = 1;
    return 0;
}

int env_sound_is_trained(void* classifier) {
    if (!classifier) return 0;
    EnvironmentSoundClassifier* esc = (EnvironmentSoundClassifier*)classifier;
    return esc->is_trained;
}

void env_sound_classifier_mark_trained(void* classifier) {
    if (!classifier) return;
    EnvironmentSoundClassifier* esc = (EnvironmentSoundClassifier*)classifier;
    esc->is_trained = 1;
}

void environment_sound_classifier_free(void* classifier) {
    if (!classifier) return;
    /* 修复H-1: 先清除全局指针引用，再释放内存
     * 原来的顺序: safe_free(&classifier) 将 classifier 置为 NULL
     * 然后 g_esc == NULL 永远为假，全局指针无法清空，造成悬空指针 */
    if (g_esc == classifier) g_esc = NULL;
    safe_free(&classifier);
}

/**
 * @brief K-034: 获取分类器类别名称列表
 */
const char* environment_sound_class_name(int class_id) {
    if (class_id < 0 || class_id >= ESC_MAX_CLASSES) return "未知";
    return g_esc_class_names[class_id];
}
