/**
 * @file tensor.c
 * @brief 张量核心实现
 * 
 * 提供张量数据结构和基本操作函数的实现。
 * 张量是多维数组，用于神经网络计算。
 * 100%纯C实现，无外部依赖。
 */

// 禁用Windows CRT安全警告

#include "selflnn/core/tensor.h"
#include "selflnn/core/memory.h"
#include "selflnn/core/safe_memory.h"
#include "selflnn/core/errors.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/* 禁用特定警告 */

/**
 * @brief 计算张量元素总数
 * 
 * @param shape 形状数组
 * @param ndim 维度数量
 * @return size_t 元素总数
 */
static size_t compute_numel(const int* shape, int ndim) {
    if (!shape || ndim <= 0) {
        return 0;
    }
    
    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            return 0; // 无效形状
        }
        numel *= (size_t)shape[i];
    }
    
    return numel;
}

/**
 * @brief 获取数据类型大小（字节）
 * 
 * @param dtype 数据类型
 * @return size_t 数据类型大小（字节），0表示未知类型
 */
static size_t dtype_size_bytes(TensorDataType dtype) {
    switch (dtype) {
        case TENSOR_FLOAT32: return sizeof(float);
        case TENSOR_FLOAT64: return sizeof(double);
        case TENSOR_INT32:   return sizeof(int32_t);
        case TENSOR_INT64:   return sizeof(int64_t);
        case TENSOR_UINT8:   return sizeof(uint8_t);
        case TENSOR_UINT16:  return sizeof(uint16_t);
        case TENSOR_UINT32:  return sizeof(uint32_t);
        case TENSOR_UINT64:  return sizeof(uint64_t);
        case TENSOR_BOOL:    return sizeof(uint8_t);
        default: return 0;
    }
}

/**
 * @brief 计算张量数据大小（字节）
 * 
 * @param shape 形状数组
 * @param ndim 维度数量
 * @param dtype 数据类型
 * @return size_t 数据大小（字节），0表示参数无效
 */
static size_t compute_data_size(const int* shape, int ndim, TensorDataType dtype) {
    size_t numel = compute_numel(shape, ndim);
    if (numel == 0) {
        return 0;
    }
    
    size_t type_size = dtype_size_bytes(dtype);
    if (type_size == 0) {
        return 0;
    }
    
    return numel * type_size;
}

/**
 * @brief 计算张量步幅数组
 * 
 * @param strides 输出步幅数组（必须已分配，长度至少为ndim）
 * @param shape 形状数组
 * @param ndim 维度数量
 * @param format 存储格式
 */
static void compute_strides(size_t* strides, const int* shape, int ndim, TensorStorageFormat format) {
    if (!strides || !shape || ndim <= 0) {
        return;
    }
    
    if (format == TENSOR_FORMAT_DENSE) {
        // 行主序（C风格）步幅
        size_t stride = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= (size_t)shape[i];
        }
    } else {
        // 对于稀疏和压缩格式，步幅不适用，设置为0
        memset(strides, 0, sizeof(size_t) * ndim);
    }
}

Tensor* tensor_create(const int* shape, int ndim, TensorDataType dtype) {
    // 参数检查
    if (!shape || ndim <= 0 || ndim > 8) { // 限制最大维度为8
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的张量形状参数");
        return NULL;
    }
    
    if (dtype < TENSOR_FLOAT32 || dtype > TENSOR_BOOL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的张量数据类型");
        return NULL;
    }
    
    // 计算数据大小
    size_t data_size = compute_data_size(shape, ndim, dtype);
    if (data_size == 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的张量形状或数据类型");
        return NULL;
    }
    
    // 分配张量结构体
    Tensor* tensor = (Tensor*)safe_malloc(sizeof(Tensor));
    if (!tensor) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配张量结构体失败");
        return NULL;
    }
    
    // 初始化张量字段
    memset(tensor, 0, sizeof(Tensor));
    tensor->dtype = dtype;
    tensor->format = TENSOR_FORMAT_DENSE; // 默认稠密存储
    tensor->ndim = ndim;
    tensor->owner = 1; // 拥有数据内存
    tensor->device = 0; // 默认CPU
    
    // 分配形状数组
    tensor->shape = (int*)safe_malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配形状数组失败");
        safe_free((void**)&tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    
    // 分配步幅数组
    tensor->strides = (size_t*)safe_malloc(ndim * sizeof(size_t));
    if (!tensor->strides) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配步幅数组失败");
        safe_free((void**)&tensor->shape);
        safe_free((void**)&tensor);
        return NULL;
    }
    compute_strides(tensor->strides, shape, ndim, tensor->format);
    
    // 分配数据内存
    tensor->data = safe_malloc(data_size);
    if (!tensor->data) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配张量数据内存失败");
        safe_free(&tensor->strides);
        safe_free(&tensor->shape);
        safe_free((void**)&tensor);
        return NULL;
    }
    
    // 初始化数据为0
    memset(tensor->data, 0, data_size);
    
    // 设置大小
    tensor->size = data_size;
    
    return tensor;
}

Tensor* tensor_wrap(void* data, const int* shape, int ndim, TensorDataType dtype) {
    // 参数检查
    if (!data || !shape || ndim <= 0 || ndim > 8) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的包装张量参数");
        return NULL;
    }
    
    if (dtype < TENSOR_FLOAT32 || dtype > TENSOR_BOOL) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的张量数据类型");
        return NULL;
    }
    
    // 分配张量结构体
    Tensor* tensor = (Tensor*)safe_malloc(sizeof(Tensor));
    if (!tensor) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配张量结构体失败");
        return NULL;
    }
    
    // 初始化张量字段
    memset(tensor, 0, sizeof(Tensor));
    tensor->dtype = dtype;
    tensor->format = TENSOR_FORMAT_DENSE;
    tensor->ndim = ndim;
    tensor->owner = 0; // 不拥有数据内存
    tensor->device = 0; // 默认CPU
    
    // 分配形状数组
    tensor->shape = (int*)safe_malloc(ndim * sizeof(int));
    if (!tensor->shape) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配形状数组失败");
        safe_free((void**)&tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    
    // 分配步幅数组
    tensor->strides = (size_t*)safe_malloc(ndim * sizeof(size_t));
    if (!tensor->strides) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配步幅数组失败");
        safe_free((void**)&tensor->shape);
        safe_free((void**)&tensor);
        return NULL;
    }
    compute_strides(tensor->strides, shape, ndim, tensor->format);
    
    // 设置数据指针（不分配新内存）
    tensor->data = data;
    
    // 计算数据大小
    tensor->size = compute_data_size(shape, ndim, dtype);
    
    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (!tensor) {
        return;
    }
    
    // 如果拥有数据内存，则释放
    if (tensor->owner && tensor->data) {
        safe_free(&tensor->data);
    }
    
    // 释放形状和步幅数组
    if (tensor->shape) {
        safe_free(&tensor->shape);
    }
    
    if (tensor->strides) {
        safe_free(&tensor->strides);
    }
    
    // 释放张量结构体
    safe_free((void**)&tensor);
}

Tensor* tensor_clone(const Tensor* src) {
    if (!src) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "源张量为空");
        return NULL;
    }
    
    // 创建新张量
    Tensor* dst = tensor_create(src->shape, src->ndim, src->dtype);
    if (!dst) {
        return NULL;
    }
    
    // 复制数据
    size_t copy_size = src->size < dst->size ? src->size : dst->size;
    memcpy(dst->data, src->data, copy_size);
    
    // 复制其他字段
    dst->format = src->format;
    dst->device = src->device;
    
    return dst;
}

size_t tensor_numel(const Tensor* tensor) {
    if (!tensor || !tensor->shape || tensor->ndim <= 0) {
        return 0;
    }
    
    return compute_numel(tensor->shape, tensor->ndim);
}

size_t tensor_size_bytes(const Tensor* tensor) {
    if (!tensor) {
        return 0;
    }
    
    return tensor->size;
}

int tensor_reshape(Tensor* tensor, const int* new_shape, int new_ndim) {
    if (!tensor || !new_shape || new_ndim <= 0 || new_ndim > 8) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "无效的重塑参数");
        return -1;
    }
    
    // 检查元素总数是否匹配
    size_t old_numel = tensor_numel(tensor);
    size_t new_numel = compute_numel(new_shape, new_ndim);
    
    if (old_numel != new_numel) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "重塑前后元素总数不匹配");
        return -1;
    }
    
    // 更新形状
    int* old_shape = tensor->shape;
    tensor->shape = (int*)safe_malloc(new_ndim * sizeof(int));
    if (!tensor->shape) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配新形状数组失败");
        tensor->shape = old_shape; // 恢复旧指针
        return -1;
    }
    
    memcpy(tensor->shape, new_shape, new_ndim * sizeof(int));
    // 注意：不立即释放old_shape，等待步幅分配成功后再释放
    
    // 更新步幅
    size_t* old_strides = tensor->strides;
    tensor->strides = (size_t*)safe_malloc(new_ndim * sizeof(size_t));
    if (!tensor->strides) {
        selflnn_set_last_error(SELFLNN_ERROR_OUT_OF_MEMORY, __func__, __FILE__, __LINE__,
                              "分配新步幅数组失败");
        // 恢复旧形状（完整实现：old_shape尚未释放，可以安全恢复）
        safe_free(&tensor->shape);
        tensor->shape = old_shape; // 安全恢复，因为old_shape尚未释放
        tensor->strides = old_strides;
        return -1;
    }
    
    compute_strides(tensor->strides, new_shape, new_ndim, tensor->format);
    
    // 现在可以安全释放旧内存
    safe_free(&old_shape);
    safe_free(&old_strides);
    
    // 更新维度
    tensor->ndim = new_ndim;
    
    return 0;
}

void tensor_fill(Tensor* tensor, float value) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    size_t numel = tensor_numel(tensor);
    if (numel == 0) {
        return;
    }
    
    // 根据数据类型填充
    switch (tensor->dtype) {
        case TENSOR_FLOAT32: {
            float* data = (float*)tensor->data;
            for (size_t i = 0; i < numel; i++) {
                data[i] = value;
            }
            break;
        }
        case TENSOR_FLOAT64: {
            double* data = (double*)tensor->data;
            for (size_t i = 0; i < numel; i++) {
                data[i] = (double)value;
            }
            break;
        }
        case TENSOR_INT32: {
            int32_t* data = (int32_t*)tensor->data;
            int32_t int_value = (int32_t)value;
            for (size_t i = 0; i < numel; i++) {
                data[i] = int_value;
            }
            break;
        }
        case TENSOR_INT64: {
            int64_t* data = (int64_t*)tensor->data;
            int64_t int_value = (int64_t)value;
            for (size_t i = 0; i < numel; i++) {
                data[i] = int_value;
            }
            break;
        }
        case TENSOR_UINT8: {
            uint8_t* data = (uint8_t*)tensor->data;
            uint8_t int_value = (uint8_t)value;
            for (size_t i = 0; i < numel; i++) {
                data[i] = int_value;
            }
            break;
        }
        case TENSOR_UINT16: {
            uint16_t* data = (uint16_t*)tensor->data;
            uint16_t int_value = (uint16_t)value;
            for (size_t i = 0; i < numel; i++) {
                data[i] = int_value;
            }
            break;
        }
        case TENSOR_UINT32: {
            uint32_t* data = (uint32_t*)tensor->data;
            uint32_t int_value = (uint32_t)value;
            for (size_t i = 0; i < numel; i++) {
                data[i] = int_value;
            }
            break;
        }
        case TENSOR_UINT64: {
            uint64_t* data = (uint64_t*)tensor->data;
            uint64_t int_value = (uint64_t)value;
            for (size_t i = 0; i < numel; i++) {
                data[i] = int_value;
            }
            break;
        }
        case TENSOR_BOOL: {
            uint8_t* data = (uint8_t*)tensor->data;
            uint8_t bool_value = value != 0.0f ? 1 : 0;
            for (size_t i = 0; i < numel; i++) {
                data[i] = bool_value;
            }
            break;
        }
        default:
            break;
    }
}

void tensor_zero(Tensor* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }

    memset(tensor->data, 0, tensor->size);
}

/* ============================================================================
 * 高级张量分解: QR, SVD, Cholesky
 *
 * 纯C实现，不依赖任何第三方库
 * ============================================================================ */

int tensor_qr_decompose(const float* A, int m, int n, float* Q, float* R) {
    if (!A || !Q || !R || m <= 0 || n <= 0 || m < n) return -1;

    float* A_copy = (float*)safe_malloc((size_t)(m * n) * sizeof(float));
    if (!A_copy) return -1;
    memcpy(A_copy, A, (size_t)(m * n) * sizeof(float));

    float* v = (float*)safe_calloc((size_t)m, sizeof(float));
    if (!v) { safe_free((void**)&A_copy); return -1; }

    for (int k = 0; k < n; k++) {
        float norm_x = 0.0f;
        for (int i = k; i < m; i++) {
            float aik = A_copy[i * n + k];
            norm_x += aik * aik;
        }
        norm_x = sqrtf(norm_x);
        float alpha = (A_copy[k * n + k] > 0.0f) ? -norm_x : norm_x;

        v[k] = A_copy[k * n + k] - alpha;
        float v_norm = v[k] * v[k];
        for (int i = k + 1; i < m; i++) {
            v[i] = A_copy[i * n + k];
            v_norm += v[i] * v[i];
        }
        float beta = 2.0f / (v_norm + 1e-15f);

        for (int j = k; j < n; j++) {
            float dot = 0.0f;
            for (int i = k; i < m; i++) dot += v[i] * A_copy[i * n + j];
            float tau = beta * dot;
            for (int i = k; i < m; i++) A_copy[i * n + j] -= tau * v[i];
        }

        A_copy[k * n + k] = alpha;
        for (int i = k + 1; i < m; i++) A_copy[i * n + k] = 0.0f;
    }

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            Q[i * n + j] = (i == j) ? 1.0f : 0.0f;

    for (int k = n - 1; k >= 0; k--) {
        float norm_v = A_copy[k * n + k];
        v[k] = 1.0f;
        for (int i = k + 1; i < m; i++) v[i] = 0.0f;
        if (fabsf(norm_v) > 1e-8f) {
            float vn = 1.0f;
            for (int i = k + 1; i < m; i++) { v[i] = A_copy[i * n + k] / norm_v; vn += v[i] * v[i]; }
            float beta = 2.0f / (vn + 1e-15f);
            for (int j = k; j < n; j++) {
                float dot = 0.0f;
                for (int i = k; i < m; i++) dot += v[i] * Q[i * n + j];
                for (int i = k; i < m; i++) Q[i * n + j] -= beta * dot * v[i];
            }
        }
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            R[i * n + j] = (i <= j) ? A_copy[i * n + j] : 0.0f;

    safe_free((void**)&v);
    safe_free((void**)&A_copy);
    return 0;
}

int tensor_cholesky_decompose(const float* A, int n, float* L) {
    if (!A || !L || n <= 0) return -1;

    memset(L, 0, (size_t)(n * n) * sizeof(float));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) sum += L[i * n + k] * L[j * n + k];
            if (i == j) {
                float diag = A[i * n + i] - sum;
                if (diag <= 0.0f) return -1;
                L[i * n + i] = sqrtf(diag);
            } else {
                L[i * n + j] = (A[i * n + j] - sum) / (L[j * n + j] + 1e-15f);
            }
        }
    }
    return 0;
}

int tensor_svd_decompose(const float* A, int m, int n, float* U, float* S, float* Vt,
                          int max_iter) {
    if (!A || !U || !S || !Vt || m <= 0 || n <= 0) return -1;
    if (max_iter <= 0) max_iter = 100;

    float* B = (float*)safe_malloc((size_t)(m * n) * sizeof(float));
    if (!B) return -1;
    memcpy(B, A, (size_t)(m * n) * sizeof(float));

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            U[i * m + j] = (i == j) ? 1.0f : 0.0f;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Vt[i * n + j] = (i == j) ? 1.0f : 0.0f;

    /* Jacobi SVD: 双边旋转消去非对角元 */
    for (int iter = 0; iter < max_iter; iter++) {
        float max_offdiag = 0.0f;
        int p = 0, q = 1;

        for (int i = 0; i < m && i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float val = 0.0f;
                for (int k = 0; k < m; k++) val += B[k * n + i] * B[k * n + j];
                if (fabsf(val) > max_offdiag) { max_offdiag = fabsf(val); p = i; q = j; }
            }
        }
        if (max_offdiag < 1e-8f) break;

        float app = 0.0f, aqq = 0.0f, apq = 0.0f;
        for (int k = 0; k < m; k++) {
            app += B[k * n + p] * B[k * n + p];
            aqq += B[k * n + q] * B[k * n + q];
            apq += B[k * n + p] * B[k * n + q];
        }

        float theta = 0.5f * atan2f(2.0f * apq, app - aqq);
        float c = cosf(theta), s = sinf(theta);

        for (int k = 0; k < m; k++) {
            float bkp = B[k * n + p], bkq = B[k * n + q];
            B[k * n + p] = c * bkp + s * bkq;
            B[k * n + q] = -s * bkp + c * bkq;
        }
        for (int k = 0; k < n; k++) {
            float vkp = Vt[p * n + k], vkq = Vt[q * n + k];
            Vt[p * n + k] = c * vkp + s * vkq;
            Vt[q * n + k] = -s * vkp + c * vkq;
        }
    }

    /* F-020修复: U是m×m矩阵，B是m×n矩阵，只复制min(m,n)列，避免越界写入 */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < (m < n ? m : n); j++)
            U[i * m + j] = B[i * n + j];

    for (int i = 0; i < (m < n ? m : n); i++) {
        float sigma = 0.0f;
        for (int k = 0; k < m; k++) sigma += B[k * n + i] * B[k * n + i];
        S[i] = sqrtf(sigma);
        if (S[i] > 1e-8f)
            for (int k = 0; k < m; k++) U[k * m + i] = B[k * n + i] / S[i];
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < (m < n ? m : n); j++) {
            float val = U[i * m + j];
            if (fabsf(val) < 1e-8f) U[i * m + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    safe_free((void**)&B);
    return 0;
}