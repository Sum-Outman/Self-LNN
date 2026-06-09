/**
 * @file dataset_mnist.h
 * @brief MNIST IDX格式数据集加载器 (纯C, 零外部依赖)
 */

#ifndef SELFLNN_DATASET_MNIST_H
#define SELFLNN_DATASET_MNIST_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 通用数据集结构 */
typedef struct {
    float* inputs;          /**< 输入数据 [num_samples × input_dim] */
    float* outputs;         /**< 输出标签 [num_samples × output_dim] (one-hot) */
    size_t num_samples;     /**< 样本数量 */
    size_t input_dim;       /**< 输入维度 */
    size_t output_dim;      /**< 输出维度 */
    int num_classes;        /**< 类别数 */
} Dataset;

/**
 * @brief 加载MNIST数据集
 * @param images_path IDX图像文件路径 (train-images-idx3-ubyte)
 * @param labels_path IDX标签文件路径 (train-labels-idx1-ubyte)
 * @param max_samples 最大加载样本数 (0=全部)
 * @return Dataset* 成功, NULL 失败
 */
Dataset* mnist_load(const char* images_path, const char* labels_path, size_t max_samples);

/**
 * @brief 释放MNIST数据集
 */
void mnist_free(Dataset* ds);

/**
 * @brief 获取一个batch
 * @param start 起始索引
 * @param count 样本数
 * @param batch_inputs 输出缓冲区 [count × input_dim]
 * @param batch_outputs 输出缓冲区 [count × output_dim]
 * @return 0成功, <0失败
 */
int mnist_get_batch(const Dataset* ds, size_t start, size_t count,
                    float* batch_inputs, float* batch_outputs);

/**
 * @brief Fisher-Yates洗牌
 */
void mnist_shuffle(Dataset* ds, unsigned int seed);

#ifdef __cplusplus
}
#endif
#endif /* SELFLNN_DATASET_MNIST_H */
