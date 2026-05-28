/**
 * @file dataset_api.h
 * @brief 训练数据集API - 与训练流水线的集成接口
 *
 * 提供数据集创建、加载、预处理、增强的高层API，
 * 供训练流水线(training_pipeline.c)和训练器(training.c)直接调用。
 * 100%纯C实现，零外部依赖。
 */
#ifndef SELFLNN_DATASET_API_H
#define SELFLNN_DATASET_API_H

#include <stddef.h>
#include <stdint.h>

#include "selflnn/training/data_loaders.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 前向声明 */
typedef struct TrainingDataset TrainingDataset;

/* 多模态样本结构 */
typedef struct {
    float* vision_features;     /* 视觉特征 */
    size_t vision_dim;
    float* audio_features;      /* 音频特征 */
    size_t audio_dim;
    float* text_features;       /* 文本特征 */
    size_t text_dim;
    float* sensor_features;     /* 传感器特征 */
    size_t sensor_dim;
    float* output;              /* 输出标签 */
    size_t output_dim;
} MultimodalSample;

/* 数据集统计信息 */
typedef struct {
    size_t num_samples;
    size_t input_dim;
    size_t output_dim;
    float input_mean[256];
    float input_std[256];
    float input_min[256];
    float input_max[256];
    float output_mean[256];
    float output_std[256];
    float total_memory_mb;
} DatasetStats;

/* 数据增强类型 */
typedef enum {
    AUGMENT_NONE = 0,
    AUGMENT_GAUSSIAN_NOISE = 1,
    AUGMENT_SCALING = 2,
    AUGMENT_DROPOUT = 4,
    AUGMENT_TIMEWARP = 5,
    AUGMENT_MIXUP = 6,
    AUGMENT_SPECAUGMENT = 7
} AugmentType;

/**
 * @brief 创建空数据集
 * @param name 数据集名称
 * @param num_samples 样本数
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @return 数据集句柄，失败返回NULL
 */
TrainingDataset* dataset_create(const char* name, size_t num_samples,
                                 size_t input_dim, size_t output_dim);

/**
 * @brief 从内存数据创建数据集
 */
TrainingDataset* dataset_from_memory(const float* inputs, const float* outputs,
                                      size_t num_samples, size_t input_dim,
                                      size_t output_dim);

/**
 * @brief 保存数据集到文件(SELF-LNN DAT格式)
 */
int dataset_save(const TrainingDataset* ds, const char* file_path);

/**
 * @brief 从文件加载数据集
 */
TrainingDataset* dataset_load(const char* file_path);

/**
 * @brief 打乱数据集
 */
int dataset_shuffle(TrainingDataset* ds);

/**
 * @brief 获取一个批次
 * @return 实际读取的样本数
 */
int dataset_get_batch(TrainingDataset* ds, size_t batch_size,
                     float* input_batch, float* output_batch);

/**
 * @brief R3-06: 数据集划分为训练集/验证集/测试集
 * @param ds 输入完整数据集
 * @param train_ratio 训练集比例 (默认0.7)
 * @param val_ratio 验证集比例 (默认0.15)
 * @param test_ratio 测试集比例 (默认0.15)
 * @param out_train 输出训练集(调用者负责释放)
 * @param out_val 输出验证集
 * @param out_test 输出测试集
 * @return 成功返回0，失败返回-1
 */
int dataset_split(TrainingDataset* ds,
    float train_ratio, float val_ratio, float test_ratio,
    TrainingDataset** out_train, TrainingDataset** out_val,
    TrainingDataset** out_test);

/**
 * @brief 计算数据集统计信息
 */
int dataset_compute_stats(const TrainingDataset* ds, DatasetStats* stats);

/**
 * @brief Z-score归一化预处理
 */
int dataset_normalize_zscore(TrainingDataset* ds);

/**
 * @brief Min-Max归一化预处理
 */
int dataset_normalize_minmax(TrainingDataset* ds);

/**
 * @brief 数据增强(高斯噪声)
 */
int dataset_augment_noise(TrainingDataset* ds, float stddev);

/**
 * @brief 数据增强(MixUp混合)
 */
int dataset_augment_mixup(TrainingDataset* ds, float alpha);

/**
 * @brief 数据增强(特征随机丢弃)
 */
int dataset_augment_dropout(TrainingDataset* ds, float drop_prob);

/**
 * @brief 数据增强(频谱增强：基于拉普拉斯FFT的频率掩码与时间掩码)
 *
 * 对每个样本执行DFT→频率掩码→时间掩码→IDFT，
 * 在频域进行数据扩充，增强模型对频域扰动的鲁棒性。
 * 需要训练流水线中启用拉普拉斯增强（laplace_enhancement）。
 *
 * @param ds 数据集
 * @param freq_mask_param 频率掩码宽度比例 (0.0~0.3)
 * @param time_mask_param 时间掩码宽度比例 (0.0~0.3)
 * @return int 成功返回0，失败返回-1
 */
int dataset_augment_spectral(TrainingDataset* ds, float freq_mask_param, float time_mask_param);

/**
 * @brief 多模态样本拼接为统一输入向量
 */
int dataset_multimodal_concat(const MultimodalSample* sample, float* unified,
                               size_t unified_dim);

/**
 * @brief 验证数据集完整性
 * @return 0=有效, -1=含NaN, -2=含Inf, -3=全零
 */
int dataset_validate(const TrainingDataset* ds);

/**
 * @brief 释放数据集内存
 */
void dataset_free(TrainingDataset* ds);

/**
 * @brief 获取数据集当前epoch
 */
size_t dataset_get_epoch(const TrainingDataset* ds);

/**
 * @brief 获取数据集样本总数
 */
size_t dataset_get_num_samples(const TrainingDataset* ds);

/**
 * @brief ZSFZS-F029: 设置数据集的训练/验证模式
 *
 * 训练模式(is_training=1)：数据增强函数正常执行
 * 验证模式(is_training=0)：所有增强函数直接跳过，保护原始数据不被污染
 *
 * @param ds 数据集
 * @param is_training 1=训练模式, 0=验证模式
 * @return 成功返回0，失败返回-1
 */
int dataset_set_training_mode(TrainingDataset* ds, int is_training);

/**
 * @brief 设置样本权重数组
 */
int dataset_set_weights(TrainingDataset* ds, const float* weights, size_t n);

/**
 * @brief 从零开始生成多模态引导数据集（视觉+音频+文本+传感器）
 *
 * 需求："禁止使用任何预训练模型，所有模型全部为本地原生模型,从零开始训练"
 * 生成具有数学结构（正弦/余弦/指数衰减）的合成训练数据，用于初始化液态神经网络训练。
 *
 * @param out_ds    输出数据集指针
 * @param num_samples 样本数(max 10000)
 * @return 0成功，-1失败
 */
int dataset_bootstrap_multimodal(TrainingDataset** out_ds, size_t num_samples);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_DATASET_API_H */
