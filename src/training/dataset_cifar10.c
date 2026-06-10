/**
 * @file dataset_cifar10.c
 * @brief CIFAR-10 数据集加载器实现
 *
 * 格式: label(1B) + R(1024B) + G(1024B) + B(1024B) per image
 * 图像归一化到 [0, 1]
 */

#include "selflnn/training/dataset_cifar10.h"
#include "selflnn/utils/memory_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Cifar10Dataset* cifar10_load_file(const char* file_path, size_t max_samples) {
    if (!file_path) return NULL;

    FILE* fp = fopen(file_path, "rb");
    if (!fp) return NULL;

    /* 每个样本: 1 + 3072 = 3073 字节 */
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize < 3073) { fclose(fp); return NULL; }

    size_t total_in_file = (size_t)fsize / 3073;
    size_t n = (max_samples > 0 && max_samples < total_in_file) ? max_samples : total_in_file;

    Cifar10Dataset* ds = (Cifar10Dataset*)safe_calloc(1, sizeof(Cifar10Dataset));
    if (!ds) { fclose(fp); return NULL; }

    ds->num_samples = n;
    ds->input_dim = 3072;
    ds->output_dim = 10;
    ds->data = (float*)safe_calloc(n * 3072, sizeof(float));
    ds->labels = (float*)safe_calloc(n * 10, sizeof(float));
    if (!ds->data || !ds->labels) { cifar10_free(ds); fclose(fp); return NULL; }

    unsigned char* row = (unsigned char*)safe_malloc(3073);
    if (!row) { cifar10_free(ds); fclose(fp); return NULL; }

    for (size_t i = 0; i < n; i++) {
        if (fread(row, 1, 3073, fp) != 3073) break;
        int label = row[0];
        if (label < 0 || label > 9) label = 0;

        /* one-hot labels */
        memset(ds->labels + i * 10, 0, 10 * sizeof(float));
        ds->labels[i * 10 + label] = 1.0f;

        /* 图像: R/G/B 各1024字节, 归一化到 [0,1] */
        for (size_t j = 0; j < 3072; j++) {
            ds->data[i * 3072 + j] = row[j + 1] / 255.0f;
        }
    }

    safe_free((void**)&row);
    fclose(fp);
    return ds;
}

Cifar10Dataset* cifar10_load_train(const char* data_dir, size_t max_samples) {
    if (!data_dir) return NULL;

    /* 每个文件10000样本, 共5个文件 */
    size_t per_file = (max_samples > 0) ? max_samples / 5 + 1 : 0;

    Cifar10Dataset* merged = (Cifar10Dataset*)safe_calloc(1, sizeof(Cifar10Dataset));
    if (!merged) return NULL;
    merged->input_dim = 3072;
    merged->output_dim = 10;

    const char* files[] = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin"
    };

    for (int f = 0; f < 5; f++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", data_dir, files[f]);
        Cifar10Dataset* batch = cifar10_load_file(path, per_file);
        if (!batch) continue;

        size_t old_n = merged->num_samples;
        size_t new_n = old_n + batch->num_samples;
        if (max_samples > 0 && new_n > max_samples) {
            batch->num_samples = max_samples - old_n;
            new_n = max_samples;
        }

        float* new_data = (float*)safe_calloc(new_n * 3072, sizeof(float));
        float* new_labels = (float*)safe_calloc(new_n * 10, sizeof(float));
        if (!new_data || !new_labels) { cifar10_free(batch); cifar10_free(merged); return NULL; }

        if (old_n > 0 && merged->data) {
            memcpy(new_data, merged->data, old_n * 3072 * sizeof(float));
            memcpy(new_labels, merged->labels, old_n * 10 * sizeof(float));
        }
        memcpy(new_data + old_n * 3072, batch->data, batch->num_samples * 3072 * sizeof(float));
        memcpy(new_labels + old_n * 10, batch->labels, batch->num_samples * 10 * sizeof(float));

        safe_free((void**)&merged->data);
        safe_free((void**)&merged->labels);
        merged->data = new_data;
        merged->labels = new_labels;
        merged->num_samples = new_n;
        cifar10_free(batch);

        if (max_samples > 0 && merged->num_samples >= max_samples) break;
    }

    return (merged->num_samples > 0) ? merged : NULL;
}

Cifar10Dataset* cifar10_load_test(const char* data_dir) {
    if (!data_dir) return NULL;
    char path[512];
    snprintf(path, sizeof(path), "%s/test_batch.bin", data_dir);
    return cifar10_load_file(path, 0);
}

void cifar10_free(Cifar10Dataset* ds) {
    if (!ds) return;
    safe_free((void**)&ds->data);
    safe_free((void**)&ds->labels);
    safe_free((void**)&ds);
}

void cifar10_get_batch(const Cifar10Dataset* ds, size_t start_index,
                       size_t batch_size, float* images_out, float* labels_out) {
    if (!ds || !images_out || !labels_out || start_index >= ds->num_samples) return;
    size_t n = (start_index + batch_size <= ds->num_samples) ? batch_size : (ds->num_samples - start_index);
    memcpy(images_out, ds->data + start_index * 3072, n * 3072 * sizeof(float));
    memcpy(labels_out, ds->labels + start_index * 10, n * 10 * sizeof(float));
}
