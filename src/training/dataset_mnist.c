/**
 * @file dataset_mnist.c
 * @brief MNIST IDX格式数据集加载器 (纯C, 零外部依赖)
 * 
 * MNIST文件格式: http://yann.lecun.com/exdb/mnist/
 * - train-images-idx3-ubyte: 训练图像 [60000, 28, 28]
 * - train-labels-idx1-ubyte: 训练标签 [60000]
 * - t10k-images-idx3-ubyte:  测试图像 [10000, 28, 28]
 * - t10k-labels-idx1-ubyte:  测试标签 [10000]
 * 
 * IDX格式: magic(4B) + dimensions[N] + data
 * magic: 0x000008xx (xx=dim_count, xx=数据类型: 0x08=ubyte, 0x0B=short, 0x0C=int, 0x0D=float)
 */

#include "selflnn/training/dataset_mnist.h"
#include "selflnn/training/training.h"
#include "selflnn/utils/memory_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* IDX big-endian 读取 */
static uint32_t idx_read_uint32(const uint8_t* buf) {
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  |  (uint32_t)buf[3];
}

/* 从文件读取全部内容到内存 */
static uint8_t* file_read_all(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    rewind(f);
    uint8_t* buf = (uint8_t*)safe_malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { safe_free((void**)&buf); return NULL; }
    *out_size = (size_t)sz;
    return buf;
}

Dataset* mnist_load(const char* images_path, const char* labels_path, size_t max_samples) {
    if (!images_path || !labels_path) return NULL;

    size_t img_sz = 0, lbl_sz = 0;
    uint8_t* img_buf = file_read_all(images_path, &img_sz);
    uint8_t* lbl_buf = file_read_all(labels_path, &lbl_sz);
    if (!img_buf || !lbl_buf) {
        safe_free((void**)&img_buf);
        safe_free((void**)&lbl_buf);
        return NULL;
    }

    /* 解析图像文件头 */
    if (img_sz < 16) goto fail;
    uint32_t img_magic = idx_read_uint32(img_buf);
    uint32_t img_count = idx_read_uint32(img_buf + 4);
    uint32_t img_rows  = idx_read_uint32(img_buf + 8);
    uint32_t img_cols  = idx_read_uint32(img_buf + 12);

    /* 解析标签文件头 */
    if (lbl_sz < 8) goto fail;
    uint32_t lbl_magic = idx_read_uint32(lbl_buf);
    uint32_t lbl_count = idx_read_uint32(lbl_buf + 4);

    /* 校验 */
    if (img_magic != 0x00000803) { /* images: 3D ubyte */
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "MNIST图像文件magic错误: 0x%08X", img_magic);
        goto fail;
    }
    if (lbl_magic != 0x00000801) { /* labels: 1D ubyte */
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "MNIST标签文件magic错误: 0x%08X", lbl_magic);
        goto fail;
    }
    if (img_count != lbl_count) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "MNIST样本数不匹配: images=%u labels=%u", img_count, lbl_count);
        goto fail;
    }

    size_t pixel_count = img_rows * img_cols;
    size_t total = img_count;
    if (max_samples > 0 && max_samples < total) total = max_samples;

    /* 分配Dataset */
    Dataset* ds = (Dataset*)safe_calloc(1, sizeof(Dataset));
    if (!ds) goto fail;
    ds->num_classes = 10;
    ds->num_samples = total;
    ds->input_dim  = pixel_count;
    ds->output_dim = 10;
    ds->inputs  = (float*)safe_malloc(total * pixel_count * sizeof(float));
    ds->outputs = (float*)safe_malloc(total * 10 * sizeof(float));
    if (!ds->inputs || !ds->outputs) {
        safe_free((void**)&ds->inputs);
        safe_free((void**)&ds->outputs);
        safe_free((void**)&ds);
        goto fail;
    }

    /* 转换数据: ubyte→float32 [0,1], labels→one-hot */
    const uint8_t* img_data = img_buf + 16;
    const uint8_t* lbl_data = lbl_buf + 8;
    for (size_t i = 0; i < total; i++) {
        /* 像素归一化 */
        float* img_dst = ds->inputs + i * pixel_count;
        for (size_t j = 0; j < pixel_count; j++) {
            img_dst[j] = (float)img_data[i * pixel_count + j] / 255.0f;
        }
        /* one-hot标签 */
        float* lbl_dst = ds->outputs + i * 10;
        memset(lbl_dst, 0, 10 * sizeof(float));
        uint8_t label = lbl_data[i];
        if (label < 10) lbl_dst[label] = 1.0f;
    }

    safe_free((void**)&img_buf);
    safe_free((void**)&lbl_buf);
    return ds;

fail:
    safe_free((void**)&img_buf);
    safe_free((void**)&lbl_buf);
    return NULL;
}

void mnist_free(Dataset* ds) {
    if (!ds) return;
    safe_free((void**)&ds->inputs);
    safe_free((void**)&ds->outputs);
    safe_free((void**)&ds);
}

int mnist_get_batch(const Dataset* ds, size_t start, size_t count,
                    float* batch_inputs, float* batch_outputs) {
    if (!ds || !batch_inputs || !batch_outputs) return -1;
    if (start + count > ds->num_samples) return -2;
    memcpy(batch_inputs,  ds->inputs  + start * ds->input_dim,
           count * ds->input_dim  * sizeof(float));
    memcpy(batch_outputs, ds->outputs + start * ds->output_dim,
           count * ds->output_dim * sizeof(float));
    return 0;
}

void mnist_shuffle(Dataset* ds, unsigned int seed) {
    if (!ds || ds->num_samples < 2) return;
    srand(seed);
    size_t pixel_count = ds->input_dim;
    /* Fisher-Yates shuffle */
    for (size_t i = ds->num_samples - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        if (i == j) continue;
        /* swap inputs */
        float* tmp = (float*)safe_malloc(pixel_count * sizeof(float));
        memcpy(tmp, ds->inputs + i * pixel_count, pixel_count * sizeof(float));
        memcpy(ds->inputs + i * pixel_count, ds->inputs + j * pixel_count, pixel_count * sizeof(float));
        memcpy(ds->inputs + j * pixel_count, tmp, pixel_count * sizeof(float));
        safe_free((void**)&tmp);
        /* swap outputs (10 floats) */
        float tmp_out[10];
        memcpy(tmp_out, ds->outputs + i * 10, 10 * sizeof(float));
        memcpy(ds->outputs + i * 10, ds->outputs + j * 10, 10 * sizeof(float));
        memcpy(ds->outputs + j * 10, tmp_out, 10 * sizeof(float));
    }
}
