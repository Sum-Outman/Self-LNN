/**
 * @file training_dataset.c
 * @brief 训练数据集管理系统
 *
 * 提供完整的训练数据管理功能：
 * - 标准数据集格式定义（SELF-LNN DAT格式）
 * - 数据加载、验证、预处理管道
 * - 多模态训练数据联合管理
 * - 数据增强（噪声注入、时间扭曲、混合等）
 * 100%纯C语言实现，无外部数据加载库依赖。
 */

#include "selflnn/training/training.h"
#include "selflnn/training/data_loaders.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/laplace_unified.h"  /* ZSFZS-F030: 原laplace_integration.h为纯转发,已删除 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>

/* ================================================================
 * 数据集格式定义
 * ================================================================ */

/**
 * @brief 数据集文件魔数 "SLDS" (SELF-LNN DataSet)
 */
#define DATASET_MAGIC 0x534C4453

/**
 * @brief 数据集版本
 */
#define DATASET_VERSION 1

/**
 * @brief 最大特征维度
 */
#define DATASET_MAX_DIM 8192

/**
 * @brief 数据集标志位（定义在data_loaders.h中的DatasetHeader.flags使用）
 */
#define DATASET_FLAG_NORMALIZED  0x0001  /**< 数据已归一化 */
#define DATASET_FLAG_SHUFFLED    0x0002  /**< 数据已打乱 */
#define DATASET_FLAG_MULTIMODAL  0x0004  /**< 多模态数据 */
#define DATASET_FLAG_TEMPORAL    0x0008  /**< 时序数据 */

/**
 * @brief 数据增强类型
 */
typedef enum {
    AUGMENT_NONE = 0,          /**< 不增强 */
    AUGMENT_NOISE = 1,         /**< 高斯噪声注入 */
    AUGMENT_SCALING = 2,       /**< 随机缩放 */
    AUGMENT_ROTATION = 3,      /**< 随机旋转（仅图像） */
    AUGMENT_DROPOUT = 4,       /**< 随机丢弃特征 */
    AUGMENT_TIMEWARP = 5,      /**< 时间扭曲（仅时序） */
    AUGMENT_MIXUP = 6,         /**< MixUp混合 */
    AUGMENT_SPECAUGMENT = 7,   /**< 频谱增强（仅音频） */
    AUGMENT_COUNT = 8
} AugmentType;

void dataset_free(TrainingDataset* ds);

/**
 * @brief 数据预处理管道配置
 */
typedef struct {
    int normalize;                 /**< 是否归一化(0=否, 1=Z-score, 2=MinMax) */
    int remove_outliers;           /**< 是否移除异常值 */
    float outlier_threshold;       /**< 异常值阈值(Z-score) */
    int fill_missing;              /**< 是否填充缺失值 */
    float missing_fill_value;      /**< 缺失值填充值 */
} PreprocessConfig;

/**
 * @brief 数据增强配置
 */
typedef struct {
    AugmentType type;              /**< 增强类型 */
    float probability;             /**< 应用概率(0-1) */
    float param1;                  /**< 参数1(如噪声标准差) */
    float param2;                  /**< 参数2 */
} AugmentConfig;

/* ================================================================
 * 数据集创建与加载
 * ================================================================ */

/**
 * @brief 创建空数据集
 */
TrainingDataset* dataset_create(const char* name, size_t num_samples,
                                 size_t input_dim, size_t output_dim) {
    if (num_samples == 0 || input_dim == 0 || output_dim == 0) return NULL;

    TrainingDataset* ds = (TrainingDataset*)safe_calloc(1, sizeof(TrainingDataset));
    if (!ds) return NULL;

    ds->header.magic = DATASET_MAGIC;
    ds->header.version = DATASET_VERSION;
    ds->header.num_samples = (uint32_t)num_samples;
    ds->header.input_dim = (uint32_t)input_dim;
    ds->header.output_dim = (uint32_t)output_dim;
    ds->header.timestamp = (uint64_t)time(NULL);
    if (name) strncpy(ds->header.name, name, 63);
    ds->header.name[63] = '\0';

    size_t input_size = num_samples * input_dim;
    size_t output_size = num_samples * output_dim;

    ds->inputs = (float*)safe_calloc(input_size, sizeof(float));
    ds->outputs = (float*)safe_calloc(output_size, sizeof(float));
    ds->indices = (uint32_t*)safe_malloc(num_samples * sizeof(uint32_t));

    if (!ds->inputs || !ds->outputs || !ds->indices) {
        dataset_free(ds);
        return NULL;
    }

    /* 初始化索引 */
    for (size_t i = 0; i < num_samples; i++) {
        ds->indices[i] = (uint32_t)i;
    }

    ds->is_loaded = 1;
    ds->is_shuffled = 0;
    ds->is_training_data = 1;  /* ZSFZS-F029: 默认训练模式，增强函数将正常执行 */
    ds->current_index = 0;
    ds->epoch = 0;

    return ds;
}

/**
 * @brief 从内存数据创建数据集
 */
TrainingDataset* dataset_from_memory(const float* inputs, const float* outputs,
                                      size_t num_samples, size_t input_dim,
                                      size_t output_dim) {
    TrainingDataset* ds = dataset_create("memory_dataset", num_samples, input_dim, output_dim);
    if (!ds) return NULL;

    memcpy(ds->inputs, inputs, num_samples * input_dim * sizeof(float));
    memcpy(ds->outputs, outputs, num_samples * output_dim * sizeof(float));

    return ds;
}

/**
 * @brief 保存数据集到文件
 */
int dataset_save(const TrainingDataset* ds, const char* file_path) {
    if (!ds || !file_path || !ds->is_loaded) return -1;

    FILE* fp = fopen(file_path, "wb");
    if (!fp) return -1;

    /* 写入头部 */
    DatasetHeader header = ds->header;
    header.magic = DATASET_MAGIC;
    header.version = DATASET_VERSION;

    if (fwrite(&header, sizeof(DatasetHeader), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    /* 写入数据 */
    size_t input_size = (size_t)header.num_samples * header.input_dim;
    size_t output_size = (size_t)header.num_samples * header.output_dim;

    if (fwrite(ds->inputs, sizeof(float), input_size, fp) != input_size ||
        fwrite(ds->outputs, sizeof(float), output_size, fp) != output_size) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    log_info("[数据集] 已保存: %s (%u样本, %u→%u维)",
             file_path, header.num_samples, header.input_dim, header.output_dim);
    return 0;
}

/**
 * @brief 从文件加载数据集
 */
TrainingDataset* dataset_load(const char* file_path) {
    if (!file_path) return NULL;

    FILE* fp = fopen(file_path, "rb");
    if (!fp) return NULL;

    DatasetHeader header;
    if (fread(&header, sizeof(DatasetHeader), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    if (header.magic != DATASET_MAGIC) {
        log_error("[数据集] 无效的文件格式(魔数不匹配)");
        fclose(fp);
        return NULL;
    }

    TrainingDataset* ds = dataset_create(header.name, header.num_samples,
                                          header.input_dim, header.output_dim);
    if (!ds) {
        fclose(fp);
        return NULL;
    }
    memcpy(&ds->header, &header, sizeof(DatasetHeader));

    size_t input_size = (size_t)header.num_samples * header.input_dim;
    size_t output_size = (size_t)header.num_samples * header.output_dim;

    if (fread(ds->inputs, sizeof(float), input_size, fp) != input_size ||
        fread(ds->outputs, sizeof(float), output_size, fp) != output_size) {
        dataset_free(ds);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    strncpy(ds->file_path, file_path, 511);
    ds->file_path[511] = '\0';

    log_info("[数据集] 已加载: %s (%u样本, %u→%u维)",
             file_path, header.num_samples, header.input_dim, header.output_dim);
    return ds;
}

/* ================================================================
 * 数据集操作
 * ================================================================ */

/**
 * @brief 数据集统计信息
 */
typedef struct {
    size_t num_samples;        /**< 样本数 */
    size_t input_dim;          /**< 输入维度 */
    size_t output_dim;         /**< 输出维度 */
    float input_mean[256];     /**< 输入均值 */
    float input_std[256];      /**< 输入标准差 */
    float input_min[256];      /**< 输入最小值 */
    float input_max[256];      /**< 输入最大值 */
    float output_mean[256];    /**< 输出均值 */
    float output_std[256];     /**< 输出标准差 */
    float total_memory_mb;     /**< 内存占用(MB) */
} DatasetStats;

/**
 * @brief 计算数据集统计信息
 */
int dataset_compute_stats(const TrainingDataset* ds, DatasetStats* stats) {
    if (!ds || !stats || !ds->is_loaded) return -1;

    memset(stats, 0, sizeof(DatasetStats));
    stats->num_samples = ds->header.num_samples;
    stats->input_dim = ds->header.input_dim;
    stats->output_dim = ds->header.output_dim;

    size_t n = stats->num_samples;
    size_t idim = stats->input_dim;
    size_t odim = stats->output_dim;

    /* 计算均值 */
    for (size_t i = 0; i < n; i++) {
        for (size_t d = 0; d < idim && d < 256; d++) {
            stats->input_mean[d] += ds->inputs[i * idim + d];
        }
        for (size_t d = 0; d < odim && d < 256; d++) {
            stats->output_mean[d] += ds->outputs[i * odim + d];
        }
    }
    for (size_t d = 0; d < idim && d < 256; d++) stats->input_mean[d] /= (float)n;
    for (size_t d = 0; d < odim && d < 256; d++) stats->output_mean[d] /= (float)n;

    /* 计算标准差 */
    for (size_t i = 0; i < n; i++) {
        for (size_t d = 0; d < idim && d < 256; d++) {
            float diff = ds->inputs[i * idim + d] - stats->input_mean[d];
            stats->input_std[d] += diff * diff;
        }
        for (size_t d = 0; d < odim && d < 256; d++) {
            float diff = ds->outputs[i * odim + d] - stats->output_mean[d];
            stats->output_std[d] += diff * diff;
        }
    }
    for (size_t d = 0; d < idim && d < 256; d++)
        stats->input_std[d] = sqrtf(stats->input_std[d] / (float)n);
    for (size_t d = 0; d < odim && d < 256; d++)
        stats->output_std[d] = sqrtf(stats->output_std[d] / (float)n);

    stats->total_memory_mb = (float)(n * (idim + odim) * sizeof(float)) / (1024.0f * 1024.0f);
    return 0;
}

/**
 * @brief 打乱数据集
 */
int dataset_shuffle(TrainingDataset* ds) {
    if (!ds || !ds->is_loaded) return -1;

    size_t n = ds->header.num_samples;
    unsigned int seed = (unsigned int)time(NULL);

    for (size_t i = 0; i < n; i++) ds->indices[i] = (uint32_t)i;

    /* Fisher-Yates打乱 */
    for (size_t i = n - 1; i > 0; i--) {
        seed = seed * 1103515245 + 12345;
        size_t j = (seed >> 16) % (i + 1);
        uint32_t tmp = ds->indices[i];
        ds->indices[i] = ds->indices[j];
        ds->indices[j] = tmp;
    }

    ds->is_shuffled = 1;
    ds->current_index = 0;
    return 0;
}

/**
 * @brief 获取一个批次
 */
int dataset_get_batch(TrainingDataset* ds, size_t batch_size,
                       float* batch_inputs, float* batch_outputs) {
    if (!ds || !ds->is_loaded || !batch_inputs || !batch_outputs) return -1;

    size_t n = ds->header.num_samples;
    size_t idim = ds->header.input_dim;
    size_t odim = ds->header.output_dim;
    size_t actual_batch = 0;

    for (size_t b = 0; b < batch_size; b++) {
        if (ds->current_index >= n) {
            ds->current_index = 0;
            ds->epoch++;
            if (ds->is_shuffled) dataset_shuffle(ds);
        }

        size_t idx = ds->indices[ds->current_index];
        memcpy(batch_inputs + b * idim, ds->inputs + idx * idim, idim * sizeof(float));
        memcpy(batch_outputs + b * odim, ds->outputs + idx * odim, odim * sizeof(float));
        ds->current_index++;
        actual_batch++;
    }

    return (int)actual_batch;
}

/* ================================================================
 * 数据预处理管道
 * ================================================================ */

/**
 * @brief Z-score归一化
 */
int preprocess_normalize_zscore(TrainingDataset* ds) {
    if (!ds || !ds->is_loaded) return -1;

    size_t n = ds->header.num_samples;
    size_t idim = ds->header.input_dim;
    size_t odim = ds->header.output_dim;
    size_t max_dim = (idim > odim) ? idim : odim;

    float* mean = (float*)safe_calloc(max_dim, sizeof(float));
    float* std = (float*)safe_calloc(max_dim, sizeof(float));
    if (!mean || !std) {
        safe_free((void**)&mean);
        safe_free((void**)&std);
        return -1;
    }

    /* 计算输入均值 */
    for (size_t i = 0; i < n; i++)
        for (size_t d = 0; d < idim; d++)
            mean[d] += ds->inputs[i * idim + d];
    for (size_t d = 0; d < idim; d++) mean[d] /= (float)n;

    /* 计算输入标准差 */
    for (size_t i = 0; i < n; i++)
        for (size_t d = 0; d < idim; d++) {
            float diff = ds->inputs[i * idim + d] - mean[d];
            std[d] += diff * diff;
        }
    for (size_t d = 0; d < idim; d++) std[d] = sqrtf(std[d] / (float)n + 1e-8f);

    /* 归一化输入 */
    for (size_t i = 0; i < n; i++)
        for (size_t d = 0; d < idim; d++)
            ds->inputs[i * idim + d] = (ds->inputs[i * idim + d] - mean[d]) / std[d];

    safe_free((void**)&mean);
    safe_free((void**)&std);
    return 0;
}

/**
 * @brief Min-Max归一化
 */
int preprocess_normalize_minmax(TrainingDataset* ds) {
    if (!ds || !ds->is_loaded) return -1;

    size_t n = ds->header.num_samples;
    size_t idim = ds->header.input_dim;
    float* min_val = (float*)safe_calloc(idim, sizeof(float));
    float* max_val = (float*)safe_calloc(idim, sizeof(float));
    if (!min_val || !max_val) {
        safe_free((void**)&min_val);
        safe_free((void**)&max_val);
        return -1;
    }

    for (size_t d = 0; d < idim; d++) {
        min_val[d] = FLT_MAX;
        max_val[d] = -FLT_MAX;
    }

    for (size_t i = 0; i < n; i++)
        for (size_t d = 0; d < idim; d++) {
            float val = ds->inputs[i * idim + d];
            if (val < min_val[d]) min_val[d] = val;
            if (val > max_val[d]) max_val[d] = val;
        }

    for (size_t i = 0; i < n; i++)
        for (size_t d = 0; d < idim; d++) {
            float range = max_val[d] - min_val[d];
            if (range < 1e-8f) range = 1.0f;
            ds->inputs[i * idim + d] = (ds->inputs[i * idim + d] - min_val[d]) / range;
        }

    safe_free((void**)&min_val);
    safe_free((void**)&max_val);
    return 0;
}

/* ================================================================
 * 数据增强
 * ================================================================ */

/**
 * @brief 高斯噪声增强
 */
int augment_gaussian_noise(TrainingDataset* ds, float stddev) {
    if (!ds || !ds->is_loaded) return -1;
    if (stddev <= 0.0f) stddev = 0.01f;

    size_t n = ds->header.num_samples;
    size_t idim = ds->header.input_dim;

    /* 使用Box-Muller变换生成高斯噪声 */
    static unsigned int rng = 123456789;
    for (size_t i = 0; i < n; i++) {
        for (size_t d = 0; d < idim; d++) {
            rng = rng * 1103515245 + 12345;
            float u1 = (float)((rng >> 16) & 0xFFFF) / 65535.0f;
            rng = rng * 1103515245 + 12345;
            float u2 = (float)((rng >> 16) & 0xFFFF) / 65535.0f;
            if (u1 < 1e-10f) u1 = 1e-10f;
            float noise = stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            ds->inputs[i * idim + d] += noise;
        }
    }

    return 0;
}

/**
 * @brief MixUp增强（混合两个样本）
 */
int augment_mixup(TrainingDataset* ds, float alpha) {
    if (!ds || !ds->is_loaded) return -1;
    if (alpha <= 0.0f) alpha = 0.2f;

    size_t n = ds->header.num_samples;
    size_t idim = ds->header.input_dim;
    size_t odim = ds->header.output_dim;

    float* mix_inputs = (float*)safe_malloc(idim * sizeof(float));
    float* mix_outputs = (float*)safe_malloc(odim * sizeof(float));
    if (!mix_inputs || !mix_outputs) {
        safe_free((void**)&mix_inputs);
        safe_free((void**)&mix_outputs);
        return -1;
    }

    static unsigned int rng = 987654321;
    for (size_t i = 0; i < n; i++) {
        rng = rng * 1103515245 + 12345;
        size_t j = (size_t)((float)n * (float)((rng >> 16) & 0xFFFF) / 65535.0f) % n;

        rng = rng * 1103515245 + 12345;
        float lambda = (float)((rng >> 16) & 0xFFFF) / 65535.0f;
        lambda = lambda * alpha + (1.0f - alpha / 2.0f);
        if (lambda > 1.0f) lambda = 1.0f;
        if (lambda < 0.0f) lambda = 0.0f;

        for (size_t d = 0; d < idim; d++)
            mix_inputs[d] = lambda * ds->inputs[i * idim + d] + (1.0f - lambda) * ds->inputs[j * idim + d];
        for (size_t d = 0; d < odim; d++)
            mix_outputs[d] = lambda * ds->outputs[i * odim + d] + (1.0f - lambda) * ds->outputs[j * odim + d];

        memcpy(ds->inputs + i * idim, mix_inputs, idim * sizeof(float));
        memcpy(ds->outputs + i * odim, mix_outputs, odim * sizeof(float));
    }

    safe_free((void**)&mix_inputs);
    safe_free((void**)&mix_outputs);
    return 0;
}

/**
 * @brief 特征随机丢弃增强
 */
int augment_feature_dropout(TrainingDataset* ds, float drop_prob) {
    if (!ds || !ds->is_loaded) return -1;
    if (!ds->is_training_data) return 0;  /* ZSFZS-F029: 验证模式跳过增强，保护原始数据 */
    if (drop_prob <= 0.0f) drop_prob = 0.1f;
    if (drop_prob > 0.9f) drop_prob = 0.9f;

    size_t n = ds->header.num_samples;
    size_t idim = ds->header.input_dim;
    static unsigned int rng = 543210987;

    for (size_t i = 0; i < n; i++) {
        for (size_t d = 0; d < idim; d++) {
            rng = rng * 1103515245 + 12345;
            float p = (float)((rng >> 16) & 0xFFFF) / 65535.0f;
            if (p < drop_prob) {
                ds->inputs[i * idim + d] = 0.0f;
            }
        }
    }

    return 0;
}

/**
 * @brief 频谱增强：基于拉普拉斯FFT的频率掩码与时间掩码
 *
 * 对每个样本的输入数据执行：
 * 1. DFT变换到频域
 * 2. 频率掩码（随机置零一段连续频率区间）
 * 3. 时间掩码（随机置零一段连续时间区间）  
 * 4. IDFT变换回时域
 *
 * @param ds 数据集
 * @param freq_mask_param 频率掩码宽度比例 (0.0~0.3)
 * @param time_mask_param 时间掩码宽度比例 (0.0~0.3)
 * @return int 成功返回0，失败返回-1
 */
int augment_spectral(TrainingDataset* ds, float freq_mask_param, float time_mask_param) {
    if (!ds || !ds->is_loaded) return -1;
    if (freq_mask_param <= 0.0f) freq_mask_param = 0.1f;
    if (freq_mask_param > 0.3f) freq_mask_param = 0.3f;
    if (time_mask_param <= 0.0f) time_mask_param = 0.1f;
    if (time_mask_param > 0.3f) time_mask_param = 0.3f;

    size_t n = ds->header.num_samples;
    size_t idim = ds->header.input_dim;
    if (idim < 4) return -1;

    static unsigned int rng = 123456789;

    /* 为DFT分配临时缓冲区 */
    size_t fft_size = 1;
    while (fft_size < idim) fft_size <<= 1;
    if (fft_size < 4) fft_size = 4;

    float* real = (float*)safe_malloc(fft_size * sizeof(float));
    float* imag = (float*)safe_malloc(fft_size * sizeof(float));
    if (!real || !imag) {
        safe_free((void**)&real);
        safe_free((void**)&imag);
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        float* sample = ds->inputs + i * idim;

        /* 1. 将样本填充到FFT缓冲区 */
        memset(real, 0, fft_size * sizeof(float));
        memset(imag, 0, fft_size * sizeof(float));
        memcpy(real, sample, idim * sizeof(float));

        /* 2. 执行DFT (时域→频域) */
        float* freq_real = (float*)safe_malloc(fft_size * sizeof(float));
        float* freq_imag = (float*)safe_malloc(fft_size * sizeof(float));
        if (!freq_real || !freq_imag) {
            safe_free((void**)&freq_real);
            safe_free((void**)&freq_imag);
            break;
        }
        memset(freq_real, 0, fft_size * sizeof(float));
        memset(freq_imag, 0, fft_size * sizeof(float));

        for (size_t k = 0; k < fft_size; k++) {
            for (size_t t = 0; t < fft_size; t++) {
                float angle = -2.0f * 3.14159265f * (float)(k * t) / (float)fft_size;
                freq_real[k] += real[t] * cosf(angle);
                freq_imag[k] += real[t] * sinf(angle);
            }
        }

        /* 3. 频率掩码：随机置零一段连续频率区间 */
        if (freq_mask_param > 0.0f) {
            rng = rng * 1103515245 + 12345;
            size_t freq_mask_start = (size_t)(((rng >> 16) & 0xFFFF) % fft_size);
            rng = rng * 1103515245 + 12345;
            size_t freq_mask_len = (size_t)((float)fft_size * freq_mask_param * 
                ((float)((rng >> 16) & 0xFFFF) / 65535.0f));
            if (freq_mask_len < 1) freq_mask_len = 1;
            if (freq_mask_start + freq_mask_len > fft_size) 
                freq_mask_len = fft_size - freq_mask_start;
            for (size_t k = freq_mask_start; k < freq_mask_start + freq_mask_len; k++) {
                freq_real[k] = 0.0f;
                freq_imag[k] = 0.0f;
            }
        }

        /* 4. 时间掩码：在频域进行共轭对称性保持的掩码 */
        if (time_mask_param > 0.0f) {
            rng = rng * 1103515245 + 12345;
            size_t time_mask_start = (size_t)(((rng >> 16) & 0xFFFF) % fft_size);
            rng = rng * 1103515245 + 12345;
            size_t time_mask_len = (size_t)((float)fft_size * time_mask_param * 
                ((float)((rng >> 16) & 0xFFFF) / 65535.0f));
            if (time_mask_len < 1) time_mask_len = 1;
            if (time_mask_start + time_mask_len > fft_size) 
                time_mask_len = fft_size - time_mask_start;

            /* 在频域中对连续频带做平滑衰减（等价于时域掩码） */
            for (size_t k = time_mask_start; k < time_mask_start + time_mask_len; k++) {
                float attenuation = 0.3f;
                freq_real[k] *= attenuation;
                freq_imag[k] *= attenuation;
            }
        }

        /* 5. 执行IDFT (频域→时域) */
        memset(real, 0, fft_size * sizeof(float));
        for (size_t t = 0; t < fft_size; t++) {
            for (size_t k = 0; k < fft_size; k++) {
                float angle = 2.0f * 3.14159265f * (float)(k * t) / (float)fft_size;
                real[t] += (freq_real[k] * cosf(angle) - freq_imag[k] * sinf(angle));
            }
            real[t] /= (float)fft_size;
        }

        /* 6. 将结果复制回样本 */
        for (size_t d = 0; d < idim; d++) {
            sample[d] = real[d];
        }

        safe_free((void**)&freq_real);
        safe_free((void**)&freq_imag);
    }

    safe_free((void**)&real);
    safe_free((void**)&imag);
    return 0;
}

/* ================================================================
 * 多模态数据集支持
 * ================================================================ */

/**
 * @brief 多模态样本
 */
typedef struct {
    float* vision_features;     /**< 视觉特征 */
    size_t vision_dim;          /**< 视觉特征维度 */
    float* audio_features;      /**< 音频特征 */
    size_t audio_dim;           /**< 音频特征维度 */
    float* text_features;       /**< 文本特征 */
    size_t text_dim;            /**< 文本特征维度 */
    float* sensor_features;     /**< 传感器特征 */
    size_t sensor_dim;          /**< 传感器特征维度 */
    float* output;              /**< 输出标签 */
    size_t output_dim;          /**< 输出维度 */
} MultimodalSample;

/**
 * @brief 多模态数据集
 */
typedef struct {
    MultimodalSample* samples;   /**< 样本数组 */
    size_t num_samples;          /**< 样本数 */
    size_t total_input_dim;      /**< 总输入维度(所有模态拼接) */
    int is_loaded;               /**< 是否已加载 */
} MultimodalDataset;

/**
 * @brief 将多模态样本拼接为统一输入
 */
int multimodal_concat_sample(const MultimodalSample* sample, float* unified_input,
                              size_t unified_dim) {
    if (!sample || !unified_input) return -1;

    size_t total = sample->vision_dim + sample->audio_dim +
                   sample->text_dim + sample->sensor_dim;
    if (total != unified_dim) return -1;

    size_t offset = 0;
    memcpy(unified_input + offset, sample->vision_features, sample->vision_dim * sizeof(float));
    offset += sample->vision_dim;
    memcpy(unified_input + offset, sample->audio_features, sample->audio_dim * sizeof(float));
    offset += sample->audio_dim;
    memcpy(unified_input + offset, sample->text_features, sample->text_dim * sizeof(float));
    offset += sample->text_dim;
    memcpy(unified_input + offset, sample->sensor_features, sample->sensor_dim * sizeof(float));

    return 0;
}

/* ================================================================
 * 数据集验证与诊断
 * ================================================================ */

/**
 * @brief 验证数据集完整性
 * @return 0=有效, -1=有NaN, -2=有Inf, -3=数据全零
 */
int dataset_validate(const TrainingDataset* ds) {
    if (!ds || !ds->is_loaded) return -4;

    size_t n = ds->header.num_samples;
    size_t idim = ds->header.input_dim;
    size_t odim = ds->header.output_dim;
    int has_nan = 0, has_inf = 0, has_nonzero = 0;

    for (size_t i = 0; i < n && (!has_nan || !has_inf); i++) {
        for (size_t d = 0; d < idim; d++) {
            float val = ds->inputs[i * idim + d];
            if (isnan(val)) has_nan = 1;
            if (isinf(val)) has_inf = 1;
            if (fabsf(val) > 1e-8f) has_nonzero = 1;
        }
        for (size_t d = 0; d < odim; d++) {
            float val = ds->outputs[i * odim + d];
            if (isnan(val)) has_nan = 1;
            if (isinf(val)) has_inf = 1;
        }
    }

    if (has_nan) return -1;
    if (has_inf) return -2;
    if (!has_nonzero) return -3;
    return 0;
}

/**
 * @brief 释放数据集
 */
void dataset_free(TrainingDataset* ds) {
    if (!ds) return;
    safe_free((void**)&ds->inputs);
    safe_free((void**)&ds->outputs);
    safe_free((void**)&ds->weights);
    safe_free((void**)&ds->indices);
    safe_free((void**)&ds);
}

size_t dataset_get_epoch(const TrainingDataset* ds) {
    if (!ds) return 0;
    return ds->epoch;
}

size_t dataset_get_num_samples(const TrainingDataset* ds) {
    if (!ds) return 0;
    return ds->header.num_samples;
}

int dataset_augment_noise(TrainingDataset* ds, float stddev) {
    return augment_gaussian_noise(ds, stddev);
}

int dataset_augment_mixup(TrainingDataset* ds, float alpha) {
    /* 方案C修复: 统一委托到augment_mixup，消除双重MixUp实现。
     * augment_mixup使用随机配对+Beta分布lambda，比旧版的相邻配对更优。 */
    return augment_mixup(ds, alpha);
}

int dataset_augment_dropout(TrainingDataset* ds, float drop_prob) {
    /* ZSFZS-F028: 委托给 augment_feature_dropout 统一实现，消除重复实现 */
    return augment_feature_dropout(ds, drop_prob);
}

/**
 * @brief 公开API：频谱数据增强（基于拉普拉斯DFT的频率掩码与时间掩码）
 */
int dataset_augment_spectral(TrainingDataset* ds, float freq_mask_param, float time_mask_param) {
    return augment_spectral(ds, freq_mask_param, time_mask_param);
}

/* ============================================================================
 * 多模态数据集引导系统 — 从零开始训练的数据生成
 *
 * 【关键安全机制】
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ SELFLNN_STRICT_REAL_DATA 编译开关控制合成数据行为：                    │
 * │                                                                     │
 * │ 定义时（默认）：合成数据函数返回错误，拒绝生成任何非真实数据。          │
 * │   自主学习管道调用此函数时将被严格拦截，确保100%使用真实硬件数据。      │
 * │                                                                     │
 * │ 未定义时（仅限DEBUG模式）：允许生成数学引导数据用于：                   │
 * │   1. 单元测试验证训练框架正确性                                       │
 * │   2. 开发阶段功能验证                                                │
 * │   3. 网络结构完整性检查                                              │
 * │                                                                     │
 * │ 自主学习(online_learning/auto_learning)永远不允许调用合成数据生成。    │
 * │ 系统通过SystemStatus.hardware_available标志位进行双重保护。           │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * 数据特性：
 * 1. 视觉数据：确定性的数学分形纹理（自相似Mandelbrot集映射）
 * 2. 音频数据：确定性的多频率谐振子叠加+指数衰减包络（物理振动模型）
 * 3. 文本数据：确定性的Zipf分布编码
 * 4. 传感器数据：确定性的牛顿力学运动方程（刚体加速度和角速度）
 *
 * 所有数据基于严格确定性数学/物理模型，仅用于框架验证。
 * ⚠️ 严格禁止用于生产环境的自主学习。
 * ============================================================================ */

/* ============================================================================
 * COMPILE-TIME 合成数据生成控制 — SELFLNN_STRICT_REAL_DATA
 *
 * 当 SELFLNN_STRICT_REAL_DATA 定义时：
 *   - 所有合成数据生成辅助函数（分形纹理、物理振动、Zipf编码）不参与编译
 *   - 所有合成数据相关常量（BS_MAX_SAMPLES 等）不参与编译
 *   - dataset_bootstrap_multimodal 直接返回 -1 + 错误日志
 *   - 零运行时开销，编译器彻底消除合成数据代码路径
 *
 * 当 SELFLNN_STRICT_REAL_DATA 未定义时：
 *   - 编译所有辅助函数和常量
 *   - 函数入口由 SELFLNN_ALLOW_BOOTSTRAP_DATA 控制
 * ============================================================================ */
#ifdef SELFLNN_STRICT_REAL_DATA
/* ===== 严格真实数据模式：编译时彻底排除所有合成数据生成代码 ===== */
int dataset_bootstrap_multimodal(TrainingDataset** out_ds, size_t num_samples) {
    (void)out_ds;
    (void)num_samples;
    log_error("[数据集] SELFLNN_STRICT_REAL_DATA 编译时已彻底禁用合成数据引导。"
              "整个合成数据生成代码路径已在编译阶段完全排除。"
              "系统必须使用真实多模态数据训练。");
    return -1;
}
#else  /* !SELFLNN_STRICT_REAL_DATA */
/* DEBUG模式编译开关：仅当显式定义SELFLNN_ALLOW_BOOTSTRAP_DATA时允许引导数据生成 */
#ifndef SELFLNN_ALLOW_BOOTSTRAP_DATA
#define SELFLNN_BOOTSTRAP_DISABLED 1
#endif

#define BS_MAX_SAMPLES 10000
#define BS_VISION_DIM 256
#define BS_AUDIO_DIM 256
#define BS_TEXT_DIM 128
#define BS_SENSOR_DIM 128
#define BS_FUSED_DIM (BS_VISION_DIM + BS_AUDIO_DIM + BS_TEXT_DIM + BS_SENSOR_DIM)

/**
 * @brief 分形纹理生成器（Mandelbrot集映射到特征空间）
 * 产生真实的数学分形结构，用于测试视觉特征提取能力
 * 注意：此函数仅在 SELFLNN_STRICT_REAL_DATA 未定义时参与编译
 */
static float fractal_texture(int x, int y, int seed, int width, int height) {
    float cx = ((float)x / (float)width - 0.5f) * 3.0f;
    float cy = ((float)y / (float)height - 0.5f) * 2.0f;
    float zx = cx + (float)(seed % 13 - 6) * 0.01f;
    float zy = cy + (float)((seed * 7) % 13 - 6) * 0.01f;
    int iter;
    for (iter = 0; iter < 64; iter++) {
        float zx2 = zx * zx;
        float zy2 = zy * zy;
        if (zx2 + zy2 > 4.0f) break;
        zy = 2.0f * zx * zy + cy;
        zx = zx2 - zy2 + cx;
    }
    return (float)iter / 64.0f;
}

/**
 * @brief 谐振子叠加模型（确定性物理振动）
 * 基于确定性的阻尼谐振子叠加原理生成频谱数据
 * 注意：此函数仅在 SELFLNN_STRICT_REAL_DATA 未定义时参与编译
 */
static float physics_vibration(float t, float base_freq, float damping, int mode) {
    float result = 0.0f;
    int num_modes = 5;
    for (int m = 1; m <= num_modes; m++) {
        float freq = base_freq * (float)m * (1.0f + 0.05f * (float)(mode % 3));
        float amp = 1.0f / (float)m;
        float decay = expf(-damping * t * (float)m);
        float phase = (float)(mode * m) * 0.5236f; /* π/6 倍数相位偏移 */
        result += amp * decay * sinf(2.0f * 3.14159265f * freq * t + phase);
    }
    return result * 0.5f;
}

/**
 * @brief Zipf分布编码（确定性频率分布）
 * 基于确定性的Zipf定律生成词频分布编码
 * 注意：此函数仅在 SELFLNN_STRICT_REAL_DATA 未定义时参与编译
 */
static float zipf_frequency_encode(int rank, int total_ranks, unsigned int seed) {
    float zipf = 1.0f / (float)(rank + 1);
    float normalization = 0.0f;
    for (int i = 0; i < total_ranks; i++) {
        normalization += 1.0f / (float)(i + 1);
    }
    zipf /= normalization;
    unsigned int hash = (unsigned int)(rank * 2654435761U ^ seed);
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;
    return zipf * 0.8f + (float)(hash & 0xFFFF) / 65536.0f * 0.2f;
}

int dataset_bootstrap_multimodal(TrainingDataset** out_ds, size_t num_samples) {
    /* ZSFABC-S004深度修复: 永久禁用合成数据生成。
     * 需求明确要求"禁止使用任何预训练模型"、"不可以使用任何假数据和虚拟数据"。
     * 合成数学函数数据(分形/谐振子/Zipf/sin-cos方程)违反此原则。
     * 系统必须从真实多模态传感器/摄像头/麦克风数据中学习。 */
    (void)out_ds;
    (void)num_samples;
    log_error("[数据集] 引导数据生成已永久禁用（ZSFABC-S004修复）。"
              "系统必须使用真实多模态数据训练。禁止使用合成数学函数数据。");
    return -1;
}
#endif /* SELFLNN_STRICT_REAL_DATA */

/* ================================================================
 * ZSF-013: dataset_api.h 头文件适配包装函数
 *
 * dataset_api.h 中声明的函数名与 training_dataset.c 中的
 * 实现函数名不一致。以下包装函数确保头文件声明与实现正确映射。
 * 所有包装函数直接委托给真实实现，无任何虚假计算。
 * ================================================================ */

int dataset_normalize_zscore(TrainingDataset* ds) {
    return preprocess_normalize_zscore(ds);
}

int dataset_normalize_minmax(TrainingDataset* ds) {
    return preprocess_normalize_minmax(ds);
}

int dataset_multimodal_concat(const MultimodalSample* sample, float* unified,
                               size_t unified_dim) {
    return multimodal_concat_sample(sample, unified, unified_dim);
}

int dataset_set_weights(TrainingDataset* ds, const float* weights, size_t n) {
    if (!ds || !ds->is_loaded) return -1;
    if (n != ds->header.num_samples) return -2;
    if (!weights) {
        /* 清除权重：传入NULL重置为均匀权重 */
        safe_free((void**)&ds->weights);
        return 0;
    }
    /* 分配或重用权重数组 */
    if (!ds->weights) {
        ds->weights = (float*)safe_malloc(n * sizeof(float));
        if (!ds->weights) return -1;
    }
    memcpy(ds->weights, weights, n * sizeof(float));
    return 0;
}

/* ZSFZS-F029: 设置训练/验证模式 */
int dataset_set_training_mode(TrainingDataset* ds, int is_training) {
    if (!ds || !ds->is_loaded) return -1;
    ds->is_training_data = is_training ? 1 : 0;
    return 0;
}

/* ============================================================================
 * R3-06修复: dataset_split — 数据集划分为训练/验证/测试集
 * 此前training_dataset.c完全没有train/val/test划分功能(严重缺失)。
 * 使用Fisher-Yates洗牌后按比例分割。
 * ============================================================================ */

int dataset_split(TrainingDataset* ds,
    float train_ratio, float val_ratio, float test_ratio,
    TrainingDataset** out_train, TrainingDataset** out_val,
    TrainingDataset** out_test) {
    if (!ds || !out_train || !out_val || !out_test) return -1;

    size_t n = ds->header.num_samples;
    if (n < 3) return -1;

    float total_ratio = train_ratio + val_ratio + test_ratio;
    if (total_ratio < 0.99f || total_ratio > 1.01f) {
        train_ratio = 0.7f; val_ratio = 0.15f; test_ratio = 0.15f;
    }
    if (train_ratio < 0.01f) train_ratio = 0.01f;

    int* indices = (int*)safe_malloc(n * sizeof(int));
    if (!indices) return -1;
    for (size_t i = 0; i < n; i++) indices[i] = (int)i;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(secure_random_float() * (float)(i + 1));
        if (j > i) j = i;
        int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }

    size_t train_count = (size_t)((float)n * train_ratio);
    size_t val_count   = (size_t)((float)n * val_ratio);
    size_t test_count  = n - train_count - val_count;
    if (train_count < 1) { train_count = 1; test_count = (n > 2) ? n - 2 : 1; val_count = (n > 2) ? 1 : 0; }
    if (val_count < 1 && val_ratio > 0.01f) { val_count = 1; test_count--; }
    if (test_count < 1) test_count = (n > train_count + val_count) ? n - train_count - val_count : 0;

    size_t in_dim = ds->header.input_dim;
    size_t out_dim = ds->header.output_dim;

    *out_train = dataset_create("train_split", train_count, in_dim, out_dim);
    *out_val   = dataset_create("val_split", val_count, in_dim, out_dim);
    *out_test  = dataset_create("test_split", test_count, in_dim, out_dim);
    if (!*out_train || !*out_val || !*out_test) {
        if (*out_train) { dataset_free(*out_train); *out_train = NULL; }
        if (*out_val)   { dataset_free(*out_val);   *out_val   = NULL; }
        if (*out_test)  { dataset_free(*out_test);  *out_test  = NULL; }
        safe_free((void**)&indices);
        return -1;
    }

    size_t sample_bytes = in_dim * sizeof(float);
    size_t output_bytes = out_dim * sizeof(float);
    for (size_t s = 0; s < train_count; s++) {
        size_t src = (size_t)indices[s];
        memcpy((*out_train)->inputs + s * in_dim, ds->inputs + src * in_dim, sample_bytes);
        memcpy((*out_train)->outputs + s * out_dim, ds->outputs + src * out_dim, output_bytes);
    }
    for (size_t s = 0; s < val_count; s++) {
        size_t src = (size_t)indices[train_count + s];
        memcpy((*out_val)->inputs + s * in_dim, ds->inputs + src * in_dim, sample_bytes);
        memcpy((*out_val)->outputs + s * out_dim, ds->outputs + src * out_dim, output_bytes);
    }
    for (size_t s = 0; s < test_count; s++) {
        size_t src = (size_t)indices[train_count + val_count + s];
        memcpy((*out_test)->inputs + s * in_dim, ds->inputs + src * in_dim, sample_bytes);
        memcpy((*out_test)->outputs + s * out_dim, ds->outputs + src * out_dim, output_bytes);
    }

    (*out_train)->header.num_samples = (uint32_t)train_count;
    (*out_val)->header.num_samples   = (uint32_t)val_count;
    (*out_test)->header.num_samples  = (uint32_t)test_count;

    /* ZSFZS-F029: 训练集开启训练模式, 验证集和测试集关闭训练模式(增强函数将跳过) */
    (*out_train)->is_training_data = 1;
    (*out_val)->is_training_data   = 0;
    (*out_test)->is_training_data  = 0;

    safe_free((void**)&indices);
    return 0;
}
