#ifndef SELFLNN_DATA_LOADERS_H
#define SELFLNN_DATA_LOADERS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * 数据集结构体定义（与 training_dataset.c 布局严格一致）
 * ================================================================ */

/** 数据集文件头 */
typedef struct {
    uint32_t magic;              /**< 魔数: 0x534C4453 ("SLDS") */
    uint32_t version;            /**< 版本号 */
    uint32_t num_samples;        /**< 样本总数 */
    uint32_t input_dim;          /**< 输入特征维度 */
    uint32_t output_dim;         /**< 输出/标签维度 */
    uint32_t flags;              /**< 标志位 */
    uint64_t timestamp;          /**< 创建时间戳 */
    char name[64];               /**< 数据集名称 */
    char description[256];       /**< 数据集描述 */
    uint32_t reserved[8];        /**< 保留字段 */
} DatasetHeader;

/** 训练数据集结构体 */
typedef struct TrainingDataset {
    DatasetHeader header;          /**< 文件头 */
    float* inputs;                 /**< 输入数据 [num_samples × input_dim] */
    float* outputs;                /**< 输出标签 [num_samples × output_dim] */
    float* weights;                /**< 样本权重 [num_samples]（可选） */
    uint32_t* indices;             /**< 打乱索引 */
    size_t current_index;          /**< 当前读取位置 */
    size_t epoch;                  /**< 当前epoch */
    int is_loaded;                 /**< 是否已加载 */
    int is_shuffled;               /**< 是否已打乱 */
    int is_training_data; /**< 训练/验证模式标记(1=训练,0=验证),增强函数据此跳过验证集 */
    char file_path[512];
} TrainingDataset;

/* DatasetStats类型移到头文件供backend.c等多模块使用 */
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

/**
 * @brief 计算数据集统计信息
 */
int dataset_compute_stats(const TrainingDataset* ds, DatasetStats* stats);

int dataset_validate(const TrainingDataset* ds);

/**
 * @brief CSV数据加载器：解析分隔符文本文件为训练数据集
 *
 * 支持引号字段、转义引号、可变行长度
 *
 * @param filepath CSV文件路径
 * @param has_header 第一行是否为列标题（1=跳过标题行）
 * @param delimiter 列分隔符（常用','或'\t'）
 * @param label_columns 标签列的列索引数组（NULL表示全部为输入）
 * @param num_label_cols 标签列数（0表示全部为输入）
 * @return TrainingDataset* 成功返回数据集，失败返回NULL
 */
TrainingDataset* data_load_csv(const char* filepath, int has_header,
                                char delimiter, const int* label_columns,
                                size_t num_label_cols);

/**
 * @brief JSON数据加载器：解析JSON格式数据集
 *
 * 支持格式：
 *   [{"input": [1,2,3], "output": [4]}, ...]
 *   {"data": [[1,2,3,4], ...], "target": [[4], ...]}
 *   [[1,2,3,4], [5,6,7,8], ...] （每行最后output_cols列为标签）
 *
 * @param filepath JSON文件路径
 * @param input_field 输入字段名（NULL表示使用默认"input"）
 * @param output_field 输出字段名（NULL表示使用默认"output"）
 * @param output_cols 数组格式时末尾标签列数（0表示自动检测）
 * @return TrainingDataset* 成功返回数据集，失败返回NULL
 */
TrainingDataset* data_load_json(const char* filepath, const char* input_field,
                                 const char* output_field, size_t output_cols);

/**
 * @brief 加载BMP图像文件为浮点数数组
 *
 * 支持24位和32位无压缩BMP格式
 * 返回的像素值归一化到[0.0, 1.0]范围
 *
 * @param filepath BMP文件路径
 * @param width 输出：图像宽度
 * @param height 输出：图像高度
 * @param channels 输出：通道数（3=RGB, 4=RGBA）
 * @return float* 成功返回像素数组（调用者负责free），失败返回NULL
 */
float* data_load_image_bmp(const char* filepath, int* width, int* height, int* channels);

/**
 * @brief 加载PPM图像文件为浮点数数组
 *
 * 支持P6（二进制RGB）格式
 * 返回的像素值归一化到[0.0, 1.0]范围
 *
 * @param filepath PPM文件路径
 * @param width 输出：图像宽度
 * @param height 输出：图像高度
 * @param channels 输出：通道数（始终为3）
 * @return float* 成功返回像素数组（调用者负责free），失败返回NULL
 */
float* data_load_image_ppm(const char* filepath, int* width, int* height, int* channels);

/**
 * @brief 图像数据集加载器：从一组图像文件创建训练数据集
 *
 * 每张图像被展平为浮点数向量，支持缩放和灰度转换
 *
 * @param filepaths 图像文件路径数组
 * @param num_images 图像数量
 * @param labels 标签数组 [num_images x label_dim]（NULL表示无标签）
 * @param label_dim 标签维度
 * @param target_width 目标宽度（0表示不缩放）
 * @param target_height 目标高度（0表示不缩放）
 * @param grayscale 是否转为灰度（1=灰度单通道，0=RGB三通道）
 * @return TrainingDataset* 成功返回数据集，失败返回NULL
 */
TrainingDataset* data_load_images(const char* const* filepaths, size_t num_images,
                                   const float* labels, size_t label_dim,
                                   int target_width, int target_height, int grayscale);

/**
 * @brief 加载WAV音频文件为浮点数数组
 *
 * 支持16位PCM格式WAV文件
 * 返回的样本值归一化到[-1.0, 1.0]范围
 *
 * @param filepath WAV文件路径
 * @param sample_rate 输出：采样率（Hz）
 * @param num_samples 输出：样本总数
 * @param num_channels 输出：声道数
 * @return float* 成功返回音频样本数组（调用者负责free），失败返回NULL
 */
float* data_load_wav(const char* filepath, int* sample_rate,
                      size_t* num_samples, int* num_channels);

/**
 * @brief WAV音频数据集加载器：从一组WAV文件创建训练数据集
 *
 * 每个音频文件被截断或填充到相同长度，作为特征向量
 *
 * @param filepaths 音频文件路径数组
 * @param num_files 音频文件数
 * @param labels 标签数组 [num_files x label_dim]（NULL表示无标签）
 * @param label_dim 标签维度
 * @param target_sample_rate 目标采样率（0表示保持原始采样率）
 * @param fixed_length 固定特征长度（每个文件截取/填充到此长度，0表示使用最长文件）
 * @return TrainingDataset* 成功返回数据集，失败返回NULL
 */
TrainingDataset* data_load_audio_wav(const char* const* filepaths, size_t num_files,
                                      const float* labels, size_t label_dim,
                                      int target_sample_rate, size_t fixed_length);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_DATA_LOADERS_H */
