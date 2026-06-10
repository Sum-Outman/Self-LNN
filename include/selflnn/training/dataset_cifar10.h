/**
 * @file dataset_cifar10.h
 * @brief CIFAR-10 数据集加载器
 *
 * CIFAR-10 格式: 每个文件 10000 条记录
 * 每条: label(1B) + R(1024B) + G(1024B) + B(1024B)
 * 图像: 32x32x3 = 3072 字节
 */

#ifndef SELFLNN_DATASET_CIFAR10_H
#define SELFLNN_DATASET_CIFAR10_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** 数据集结构体 (与 dataset_mnist.h 兼容) */
typedef struct {
    float* data;          /**< [num_samples * 3072]  float, [0,1] */
    float* labels;        /**< [num_samples * 10]    one-hot */
    size_t num_samples;
    size_t input_dim;     /**< 3072 */
    size_t output_dim;    /**< 10 */
} Cifar10Dataset;

/**
 * @brief 加载单个 CIFAR-10 二进制文件
 * @param file_path  文件路径
 * @param max_samples  最大样本数 (0=全部, 通常10000)
 * @return 数据集句柄 (需用 cifar10_free 释放)
 */
Cifar10Dataset* cifar10_load_file(const char* file_path, size_t max_samples);

/**
 * @brief 加载 CIFAR-10 训练集 (5个文件合并, 共50000)
 * @param data_dir  数据目录
 * @param max_samples  最大样本数 (0=全部)
 * @return 数据集句柄
 */
Cifar10Dataset* cifar10_load_train(const char* data_dir, size_t max_samples);

/**
 * @brief 加载 CIFAR-10 测试集 (1个文件, 10000)
 * @param data_dir  数据目录
 * @return 数据集句柄
 */
Cifar10Dataset* cifar10_load_test(const char* data_dir);

/**
 * @brief 释放数据集
 */
void cifar10_free(Cifar10Dataset* ds);

/**
 * @brief 从数据集中读取一批样本
 * @param ds    数据集
 * @param start_index  起始索引
 * @param batch_size   批次大小
 * @param images_out   输出图像 [batch_size * 3072]
 * @param labels_out   输出标签 [batch_size * 10] one-hot
 */
void cifar10_get_batch(const Cifar10Dataset* ds, size_t start_index,
                       size_t batch_size, float* images_out, float* labels_out);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_DATASET_CIFAR10_H */
