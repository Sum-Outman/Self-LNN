/**
 * @file training_dataset.h
 * @brief 训练数据集管理系统
 *
 * 提供完整的训练数据管理功能：
 * - 标准数据集格式定义（SELF-LNN DAT格式）
 * - 数据加载、验证、预处理管道
 * - 多模态训练数据联合管理
 * - 数据增强（噪声注入、时间扭曲、混合等）
 * 100%纯C语言实现，无外部数据加载库依赖。
 *
 * OR-001修复: 创建独立头文件声明导出函数
 */

#ifndef SELFLNN_TRAINING_DATASET_H
#define SELFLNN_TRAINING_DATASET_H

#include "selflnn/training/data_loaders.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 数据集魔数 "SLDS" */
#define DATASET_MAGIC 0x534C4453
#define DATASET_VERSION 1
#define DATASET_MAX_DIM 8192

/** @brief 数据集标志位 */
#define DATASET_FLAG_NORMALIZED  0x0001
#define DATASET_FLAG_SHUFFLED    0x0002
#define DATASET_FLAG_MULTIMODAL  0x0004
#define DATASET_FLAG_TEMPORAL    0x0008

/** @brief 数据集统计信息
 * DatasetStats 已在 data_loaders.h 中完整定义，此处不再重复定义。
 * 包含字段: num_samples, input_dim, output_dim, input_mean[256],
 *          input_std[256], input_min[256], input_max[256],
 *          output_mean[256], output_std[256], total_memory_mb
 */

/* DatasetSplit已被新的7参数dataset_split替代，移除此未使用的结构体 */

/* 数据集生命周期管理 */
TrainingDataset* dataset_create(const char* name, size_t num_samples,
                               size_t input_dim, size_t output_dim);
void dataset_free(TrainingDataset* ds);

/* 数据集属性查询（修复C4013: 原缺失声明导致编译器假设返回int） */
size_t dataset_get_num_samples(const TrainingDataset* ds);
size_t dataset_get_epoch(const TrainingDataset* ds);

/* 序列化和统计 */
int dataset_save(const TrainingDataset* ds, const char* file_path);
TrainingDataset* dataset_load(const char* file_path);
int dataset_compute_stats(const TrainingDataset* ds, DatasetStats* stats);

/* 数据处理 */
int dataset_shuffle(TrainingDataset* ds);
int dataset_get_batch(TrainingDataset* ds, size_t batch_size,
                      float* batch_data, float* batch_labels);
int dataset_validate(const TrainingDataset* ds);
int dataset_set_weights(TrainingDataset* ds, const float* weights, size_t n);
int dataset_set_training_mode(TrainingDataset* ds, int is_training);

/* 预处理 */
int preprocess_normalize_zscore(TrainingDataset* ds);
int preprocess_normalize_minmax(TrainingDataset* ds);
int dataset_normalize_zscore(TrainingDataset* ds);
int dataset_normalize_minmax(TrainingDataset* ds);

/* 数据增强 */
int augment_gaussian_noise(TrainingDataset* ds, float stddev);
int augment_mixup(TrainingDataset* ds, float alpha);
int augment_feature_dropout(TrainingDataset* ds, float drop_prob);
int augment_spectral(TrainingDataset* ds, float freq_mask_param, float time_mask_param);
int dataset_augment_noise(TrainingDataset* ds, float stddev);
int dataset_augment_mixup(TrainingDataset* ds, float alpha);
int dataset_augment_dropout(TrainingDataset* ds, float drop_prob);
int dataset_augment_spectral(TrainingDataset* ds, float freq_mask_param, float time_mask_param);

/* 多模态处理 */
/* MultimodalSample前向声明(完整定义在training_dataset.c中) */
typedef struct MultimodalSample MultimodalSample;
int multimodal_concat_sample(const MultimodalSample* sample, float* unified_input, size_t unified_dim);
int dataset_multimodal_concat(const MultimodalSample* sample, float* unified, size_t dim);

/* 数据集分割 — 7参数版本与.c实现一致
 * 将数据集按比例分割为训练集、验证集、测试集三个独立数据集
 * @param ds 源数据集
 * @param train_ratio 训练集比例 (如0.7)
 * @param val_ratio 验证集比例 (如0.15)
 * @param test_ratio 测试集比例 (如0.15)
 * @param out_train 输出训练集
 * @param out_val 输出验证集
 * @param out_test 输出测试集
 * @return 0成功，-1失败 */
int dataset_split(TrainingDataset* ds, float train_ratio, float val_ratio, float test_ratio,
                  TrainingDataset** out_train, TrainingDataset** out_val, TrainingDataset** out_test);

/* 引导数据生成（严格模式下禁用） */
int dataset_bootstrap_multimodal(TrainingDataset** out_ds, size_t num_samples);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TRAINING_DATASET_H */
