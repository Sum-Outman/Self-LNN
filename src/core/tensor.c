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
#include "selflnn/utils/memory_utils.h"
#include "selflnn/core/errors.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

/* ZSFBUILD: -INFINITY在MSVC+WX配置下不可用，使用-FLT_MAX替代 */
#ifndef SELFLNN_NEG_INF
#define SELFLNN_NEG_INF (-FLT_MAX)
#endif
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

    /* P1-030修复: Q初始化为m×m正交矩阵（非m×n），QR分解要求Q∈R^{m×m}
     * A = Q[:,:n] @ R，其中Q的前n列参与分解 */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            Q[i * m + j] = (i == j) ? 1.0f : 0.0f;

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
                for (int i = k; i < m; i++) dot += v[i] * Q[i * m + j];
                for (int i = k; i < m; i++) Q[i * m + j] -= beta * dot * v[i];
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

/* ============================================================================
 * 核心张量运算: 矩阵乘法、逐元素加法、卷积、池化、激活函数、Softmax
 *
 * 纯C实现，支持OpenMP并行加速
 * 主要数据类型为float32（神经网络标准精度）
 * ============================================================================ */

/* ---------------------------------------------------------------------------
 * 内部辅助函数：检查张量是否为float32稠密格式
 * --------------------------------------------------------------------------- */
static int tensor_is_f32_dense(const Tensor* t) {
    return t && t->dtype == TENSOR_FLOAT32 && t->format == TENSOR_FORMAT_DENSE && t->data;
}

/* ---------------------------------------------------------------------------
 * 内部辅助函数：计算广播后的输出形状
 * 从最后一维开始对齐，返回广播维度和广播形状
 * out_ndim: 输出维度数
 * out_shape: 输出形状数组（调用者分配，至少8个int）
 * 返回0成功，-1表示形状不兼容
 * --------------------------------------------------------------------------- */
static int compute_broadcast_shape(const Tensor* a, const Tensor* b,
                                    int* out_ndim, int* out_shape) {
    int ndim_a = a->ndim;
    int ndim_b = b->ndim;
    int max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;

    for (int i = 0; i < max_ndim; i++) {
        int idx_a = ndim_a - 1 - i;
        int idx_b = ndim_b - 1 - i;
        int sa = (idx_a >= 0) ? a->shape[idx_a] : 1;
        int sb = (idx_b >= 0) ? b->shape[idx_b] : 1;

        if (sa == sb) {
            out_shape[max_ndim - 1 - i] = sa;
        } else if (sa == 1) {
            out_shape[max_ndim - 1 - i] = sb;
        } else if (sb == 1) {
            out_shape[max_ndim - 1 - i] = sa;
        } else {
            return -1; /* 形状不兼容 */
        }
    }

    *out_ndim = max_ndim;
    return 0;
}

/* ---------------------------------------------------------------------------
 * 内部辅助函数：获取广播后线性索引对应的原始张量索引
 * broadcast_shape: 广播形状
 * ndim: 广播维度数
 * orig_shape: 原始张量形状
 * orig_ndim: 原始张量维度数
 * index: 广播空间中的索引
 * 返回原始张量中的线性索引（步幅相乘计算）
 * --------------------------------------------------------------------------- */
static size_t broadcast_index(const int* broadcast_shape, int ndim,
                               const int* orig_shape, int orig_ndim,
                               size_t broadcast_linear_idx) {
    size_t remainder = broadcast_linear_idx;
    size_t orig_idx = 0;
    size_t orig_stride = 1;

    for (int d = ndim - 1; d >= 0; d--) {
        int bc_dim = broadcast_shape[d];
        int coord = (int)(remainder % (size_t)bc_dim);
        remainder /= (size_t)bc_dim;

        int orig_d = d - (ndim - orig_ndim);
        if (orig_d >= 0) {
            int os = orig_shape[orig_d];
            if (os == bc_dim) {
                orig_idx += (size_t)coord * orig_stride;
            }
            /* os == 1 时坐标被忽略（广播），orig_idx不变 */
            orig_stride *= (size_t)os;
        }
    }

    return orig_idx;
}

/* ============================================================================
 * 1. tensor_matmul — 矩阵乘法 C[m,n] = A[m,k] × B[k,n]
 *
 * 支持2D矩阵乘法和3D批量矩阵乘法（第一维为批次）
 * OpenMP并行化最外层循环
 * ============================================================================ */

int tensor_matmul(const Tensor* a, const Tensor* b, Tensor* c) {
    if (!a || !b || !c) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "矩阵乘法输入张量为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!tensor_is_f32_dense(a) || !tensor_is_f32_dense(b) || !tensor_is_f32_dense(c)) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "矩阵乘法仅支持float32稠密张量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    /* 解析维度 */
    int ndim_a = a->ndim, ndim_b = b->ndim, ndim_c = c->ndim;
    int m, k_a, k_b, n;
    int batch_a = 1, batch_b = 1, batch_c = 1;
    int use_batch = 0;

    if (ndim_a == 2 && ndim_b == 2) {
        /* 纯2D矩阵乘法 */
        m = a->shape[0]; k_a = a->shape[1];
        k_b = b->shape[0]; n = b->shape[1];
    } else if (ndim_a == 3 && ndim_b == 2) {
        /* A批次 × B共享: [ba, m, k] × [k, n] → [ba, m, n] */
        use_batch = 1;
        batch_a = a->shape[0]; batch_b = 1; batch_c = batch_a;
        m = a->shape[1]; k_a = a->shape[2];
        k_b = b->shape[0]; n = b->shape[1];
    } else if (ndim_a == 2 && ndim_b == 3) {
        /* A共享 × B批次: [m, k] × [bb, k, n] → [bb, m, n] */
        use_batch = 1;
        batch_a = 1; batch_b = b->shape[0]; batch_c = batch_b;
        m = a->shape[0]; k_a = a->shape[1];
        k_b = b->shape[1]; n = b->shape[2];
    } else if (ndim_a == 3 && ndim_b == 3) {
        /* 批量矩阵乘法: [b, m, k] × [b, k, n] → [b, m, n] */
        use_batch = 1;
        batch_a = a->shape[0]; batch_b = b->shape[0];
        if (batch_a != batch_b) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                                  "批量矩阵乘法批次维度不匹配");
            return SELFLNN_ERROR_INVALID_DIMENSION;
        }
        batch_c = batch_a;
        m = a->shape[1]; k_a = a->shape[2];
        k_b = b->shape[1]; n = b->shape[2];
    } else {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "矩阵乘法仅支持2D或3D张量");
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }

    /* 验证内维度匹配 */
    if (k_a != k_b) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "矩阵乘法内维度不匹配 A:[... %d] × B:[%d ...]",
                              k_a, k_b);
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }

    /* 验证输出形状 */
    if (use_batch) {
        if (ndim_c != 3 || c->shape[0] != batch_c || c->shape[1] != m || c->shape[2] != n) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                                  "输出张量形状不匹配，期望[%d,%d,%d]", batch_c, m, n);
            return SELFLNN_ERROR_INVALID_DIMENSION;
        }
    } else {
        if (ndim_c != 2 || c->shape[0] != m || c->shape[1] != n) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                                  "输出张量形状不匹配，期望[%d,%d]", m, n);
            return SELFLNN_ERROR_INVALID_DIMENSION;
        }
    }

    const float* data_a = (const float*)a->data;
    const float* data_b = (const float*)b->data;
    float* data_c = (float*)c->data;

    int k = k_a;
    size_t stride_a_m = (size_t)a->strides[use_batch ? 1 : 0];
    size_t stride_a_b = use_batch ? (size_t)a->strides[0] : 0;
    size_t stride_b_n = (size_t)b->strides[use_batch ? 2 : 1];
    size_t stride_b_k = (size_t)b->strides[use_batch ? 1 : 0];
    size_t stride_b_b = (use_batch && ndim_b == 3) ? (size_t)b->strides[0] : 0;
    size_t stride_c_m = (size_t)c->strides[use_batch ? 1 : 0];
    size_t stride_c_b = use_batch ? (size_t)c->strides[0] : 0;

    /* 初始化输出为0 */
    memset(data_c, 0, c->size);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(batch_c * m > 256)
#endif
    for (int b = 0; b < batch_c; b++) {
        for (int i = 0; i < m; i++) {
            size_t b_offset_a = (use_batch && ndim_a == 3) ? (size_t)b * stride_a_b : 0;
            size_t b_offset_b = (use_batch && ndim_b == 3) ? (size_t)b * stride_b_b : 0;
            size_t b_offset_c = use_batch ? (size_t)b * stride_c_b : 0;

            const float* row_a = data_a + b_offset_a + (size_t)i * stride_a_m;
            float* row_c = data_c + b_offset_c + (size_t)i * stride_c_m;

            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                const float* col_b = data_b + b_offset_b + (size_t)j * stride_b_n;
                /* 内积 */
                for (int p = 0; p < k; p++) {
                    sum += row_a[p] * col_b[p * stride_b_k];
                }
                row_c[j] = sum;
            }
        }
    }

    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * 2. tensor_add — 逐元素加法，支持广播
 *
 * NumPy风格广播：从最后一维对齐，维度为1的可以广播
 * ============================================================================ */

int tensor_add(const Tensor* a, const Tensor* b, Tensor* result) {
    if (!a || !b || !result) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "加法输入张量为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!tensor_is_f32_dense(a) || !tensor_is_f32_dense(b) || !tensor_is_f32_dense(result)) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "加法仅支持float32稠密张量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    /* 计算广播形状 */
    int bc_ndim;
    int bc_shape[8];
    if (compute_broadcast_shape(a, b, &bc_ndim, bc_shape) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "张量加法形状不兼容：左[%dD] 右[%dD]", a->ndim, b->ndim);
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }

    /* 验证输出形状匹配广播结果 */
    if (result->ndim != bc_ndim) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "输出张量维度不匹配广播结果");
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }
    for (int d = 0; d < bc_ndim; d++) {
        if (result->shape[d] != bc_shape[d]) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                                  "输出张量形状不匹配广播结果");
            return SELFLNN_ERROR_INVALID_DIMENSION;
        }
    }

    /* 计算广播总元素数 */
    size_t total = 1;
    for (int d = 0; d < bc_ndim; d++) {
        total *= (size_t)bc_shape[d];
    }

    const float* data_a = (const float*)a->data;
    const float* data_b = (const float*)b->data;
    float* data_r = (float*)result->data;

    /* 逐元素广播加法 */
#ifdef _OPENMP
    #pragma omp parallel for if(total > 4096)
#endif
    for (size_t idx = 0; idx < total; idx++) {
        size_t idx_a = broadcast_index(bc_shape, bc_ndim, a->shape, a->ndim, idx);
        size_t idx_b = broadcast_index(bc_shape, bc_ndim, b->shape, b->ndim, idx);
        data_r[idx] = data_a[idx_a] + data_b[idx_b];
    }

    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * 3. tensor_conv2d — 2D卷积
 *
 * Input:  [N, C_in,  H, W]
 * Kernel: [C_out, C_in, kH, kW]
 * Output: [N, C_out, H_out, W_out]
 * ============================================================================ */

int tensor_conv2d(const Tensor* input, const Tensor* kernel, Tensor* output,
                  int stride_h, int stride_w, int padding_h, int padding_w,
                  int dilation_h, int dilation_w) {
    if (!input || !kernel || !output) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "卷积输入张量为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!tensor_is_f32_dense(input) || !tensor_is_f32_dense(kernel) || !tensor_is_f32_dense(output)) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "卷积仅支持float32稠密张量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    if (input->ndim != 4 || kernel->ndim != 4 || output->ndim != 4) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "卷积仅支持4D张量 [N,C,H,W]");
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }

    int N  = input->shape[0];
    int Ci = input->shape[1];
    int H  = input->shape[2];
    int W  = input->shape[3];
    int Co = kernel->shape[0];
    int Kc = kernel->shape[1];  /* 输入通道数 */
    int kH = kernel->shape[2];
    int kW = kernel->shape[3];

    if (Ci != Kc) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "卷积输入通道数不匹配: input %d vs kernel %d", Ci, Kc);
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }

    if (stride_h <= 0) stride_h = 1;
    if (stride_w <= 0) stride_w = 1;
    if (dilation_h <= 0) dilation_h = 1;
    if (dilation_w <= 0) dilation_w = 1;

    /* 计算输出尺寸 */
    int H_out = (H + 2 * padding_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    if (H_out <= 0 || W_out <= 0) {
        selflnn_set_last_error(SELFLNN_ERROR_COMPUTATION, __func__, __FILE__, __LINE__,
                              "卷积输出尺寸无效: %d×%d", H_out, W_out);
        return SELFLNN_ERROR_COMPUTATION;
    }

    /* 验证输出形状 */
    if (output->shape[0] != N || output->shape[1] != Co ||
        output->shape[2] != H_out || output->shape[3] != W_out) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "输出张量形状不匹配，期望[%d,%d,%d,%d]",
                              N, Co, H_out, W_out);
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }

    const float* data_in  = (const float*)input->data;
    const float* data_ker = (const float*)kernel->data;
    float* data_out = (float*)output->data;

    /* 卷积步幅/膨胀后的有效核尺寸 */
    int eff_kH = (kH - 1) * dilation_h + 1;
    int eff_kW = (kW - 1) * dilation_w + 1;

    size_t stride_in_N  = (size_t)input->strides[0];
    size_t stride_in_C  = (size_t)input->strides[1];
    size_t stride_in_H  = (size_t)input->strides[2];
    size_t stride_in_W  = (size_t)input->strides[3];
    size_t stride_ker_C = (size_t)kernel->strides[0];
    size_t stride_ker_c = (size_t)kernel->strides[1];
    size_t stride_ker_H = (size_t)kernel->strides[2];
    size_t stride_ker_W = (size_t)kernel->strides[3];
    size_t stride_out_N = (size_t)output->strides[0];
    size_t stride_out_C = (size_t)output->strides[1];
    size_t stride_out_H = (size_t)output->strides[2];
    size_t stride_out_W = (size_t)output->strides[3];

    /* 清零输出 */
    memset(data_out, 0, output->size);

    /* 卷积计算 - OpenMP并行化批次和输出通道 */
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(N * Co > 16)
#endif
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < Co; co++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    float sum = 0.0f;

                    /* 感受野起始位置（考虑填充） */
                    int h_start = h * stride_h - padding_h;
                    int w_start = w * stride_w - padding_w;

                    /* 卷积核遍历 */
                    for (int ci = 0; ci < Ci; ci++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int h_in = h_start + kh * dilation_h;
                                int w_in = w_start + kw * dilation_w;

                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    float input_val = data_in[(size_t)n * stride_in_N +
                                                              (size_t)ci * stride_in_C +
                                                              (size_t)h_in * stride_in_H +
                                                              (size_t)w_in * stride_in_W];
                                    float ker_val = data_ker[(size_t)co * stride_ker_C +
                                                            (size_t)ci * stride_ker_c +
                                                            (size_t)kh * stride_ker_H +
                                                            (size_t)kw * stride_ker_W];
                                    sum += input_val * ker_val;
                                }
                            }
                        }
                    }

                    data_out[(size_t)n * stride_out_N +
                            (size_t)co * stride_out_C +
                            (size_t)h * stride_out_H +
                            (size_t)w * stride_out_W] = sum;
                }
            }
        }
    }

    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * 4. tensor_maxpool2d — 2D最大池化
 *
 * Input:  [N, C, H, W]
 * Output: [N, C, H_out, W_out]
 * ============================================================================ */

int tensor_maxpool2d(const Tensor* input, Tensor* output,
                     int kernel_h, int kernel_w, int stride_h, int stride_w,
                     int padding_h, int padding_w) {
    if (!input || !output) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "池化输入张量为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!tensor_is_f32_dense(input) || !tensor_is_f32_dense(output)) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "池化仅支持float32稠密张量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    if (input->ndim != 4 || output->ndim != 4) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "池化仅支持4D张量 [N,C,H,W]");
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }

    if (kernel_h <= 0) kernel_h = 2;
    if (kernel_w <= 0) kernel_w = 2;
    if (stride_h <= 0) stride_h = kernel_h;
    if (stride_w <= 0) stride_w = kernel_w;

    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];

    int H_out = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int W_out = (W + 2 * padding_w - kernel_w) / stride_w + 1;

    if (H_out <= 0 || W_out <= 0) {
        selflnn_set_last_error(SELFLNN_ERROR_COMPUTATION, __func__, __FILE__, __LINE__,
                              "池化输出尺寸无效: %d×%d", H_out, W_out);
        return SELFLNN_ERROR_COMPUTATION;
    }

    if (output->shape[0] != N || output->shape[1] != C ||
        output->shape[2] != H_out || output->shape[3] != W_out) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "输出张量形状不匹配，期望[%d,%d,%d,%d]",
                              N, C, H_out, W_out);
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }

    const float* data_in = (const float*)input->data;
    float* data_out = (float*)output->data;

    size_t stride_in_N = (size_t)input->strides[0];
    size_t stride_in_C = (size_t)input->strides[1];
    size_t stride_in_H = (size_t)input->strides[2];
    size_t stride_in_W = (size_t)input->strides[3];
    size_t stride_out_N = (size_t)output->strides[0];
    size_t stride_out_C = (size_t)output->strides[1];
    size_t stride_out_H = (size_t)output->strides[2];
    size_t stride_out_W = (size_t)output->strides[3];

    /* 最大池化计算 */
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(N * C > 16)
#endif
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    float max_val = SELFLNN_NEG_INF;

                    int h_start = h * stride_h - padding_h;
                    int w_start = w * stride_w - padding_w;

                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int h_in = h_start + kh;
                            int w_in = w_start + kw;

                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                float val = data_in[(size_t)n * stride_in_N +
                                                   (size_t)c * stride_in_C +
                                                   (size_t)h_in * stride_in_H +
                                                   (size_t)w_in * stride_in_W];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }

                    data_out[(size_t)n * stride_out_N +
                            (size_t)c * stride_out_C +
                            (size_t)h * stride_out_H +
                            (size_t)w * stride_out_W] = max_val;
                }
            }
        }
    }

    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * 5. 激活函数 — ReLU / Sigmoid / Tanh（原地操作）
 *
 * 所有操作均为纯C实现，使用数学函数expf/tanhf
 * ============================================================================ */

int tensor_relu(Tensor* tensor) {
    if (!tensor) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "ReLU输入张量为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!tensor_is_f32_dense(tensor)) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "ReLU仅支持float32稠密张量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    size_t numel = tensor_numel(tensor);
    float* data = (float*)tensor->data;

#ifdef _OPENMP
    #pragma omp parallel for if(numel > 16384)
#endif
    for (size_t i = 0; i < numel; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }

    return SELFLNN_SUCCESS;
}

int tensor_sigmoid(Tensor* tensor) {
    if (!tensor) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "Sigmoid输入张量为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!tensor_is_f32_dense(tensor)) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "Sigmoid仅支持float32稠密张量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    size_t numel = tensor_numel(tensor);
    float* data = (float*)tensor->data;

#ifdef _OPENMP
    #pragma omp parallel for if(numel > 16384)
#endif
    for (size_t i = 0; i < numel; i++) {
        /* 数值稳定: 使用expf实现，大负值钳位 */
        if (data[i] >= 0.0f) {
            data[i] = 1.0f / (1.0f + expf(-data[i]));
        } else {
            float exp_x = expf(data[i]);
            data[i] = exp_x / (1.0f + exp_x);
        }
    }

    return SELFLNN_SUCCESS;
}

int tensor_tanh(Tensor* tensor) {
    if (!tensor) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "Tanh输入张量为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!tensor_is_f32_dense(tensor)) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "Tanh仅支持float32稠密张量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    size_t numel = tensor_numel(tensor);
    float* data = (float*)tensor->data;

#ifdef _OPENMP
    #pragma omp parallel for if(numel > 16384)
#endif
    for (size_t i = 0; i < numel; i++) {
        /* 使用标准库tanhf，数值稳定 */
        data[i] = tanhf(data[i]);
    }

    return SELFLNN_SUCCESS;
}

/* ============================================================================
 * 6. tensor_softmax — Softmax归一化
 *
 * softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 * 减去最大值提升数值稳定性
 * 沿指定轴进行归一化
 * ============================================================================ */

int tensor_softmax(const Tensor* input, Tensor* output, int axis) {
    if (!input || !output) {
        selflnn_set_last_error(SELFLNN_ERROR_NULL_POINTER, __func__, __FILE__, __LINE__,
                              "Softmax输入张量为空");
        return SELFLNN_ERROR_NULL_POINTER;
    }

    if (!tensor_is_f32_dense(input) || !tensor_is_f32_dense(output)) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "Softmax仅支持float32稠密张量");
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    /* 验证输出形状与输入一致 */
    if (output->ndim != input->ndim) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                              "Softmax输出维度与输入不一致");
        return SELFLNN_ERROR_INVALID_DIMENSION;
    }
    for (int d = 0; d < input->ndim; d++) {
        if (output->shape[d] != input->shape[d]) {
            selflnn_set_last_error(SELFLNN_ERROR_INVALID_DIMENSION, __func__, __FILE__, __LINE__,
                                  "Softmax输出形状与输入不一致");
            return SELFLNN_ERROR_INVALID_DIMENSION;
        }
    }

    /* 处理负轴索引 */
    int actual_axis = (axis < 0) ? input->ndim + axis : axis;
    if (actual_axis < 0 || actual_axis >= input->ndim) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_ARGUMENT, __func__, __FILE__, __LINE__,
                              "Softmax轴索引无效: %d (ndim=%d)", axis, input->ndim);
        return SELFLNN_ERROR_INVALID_ARGUMENT;
    }

    const float* data_in = (const float*)input->data;
    float* data_out = (float*)output->data;

    int ndim = input->ndim;
    int axis_size = input->shape[actual_axis];

    /* 计算沿轴的元素总数（批次）和轴步幅 */
    size_t outer_size = 1;
    for (int d = 0; d < actual_axis; d++) {
        outer_size *= (size_t)input->shape[d];
    }

    size_t inner_size = 1;
    for (int d = actual_axis + 1; d < ndim; d++) {
        inner_size *= (size_t)input->shape[d];
    }

    size_t axis_stride = inner_size;
    size_t outer_stride = (size_t)axis_size * inner_size;

    /* 先复制输入到输出 */
    memcpy(data_out, data_in, input->size);

#ifdef _OPENMP
    #pragma omp parallel for if(outer_size > 64)
#endif
    for (size_t o = 0; o < outer_size; o++) {
        for (size_t j = 0; j < inner_size; j++) {
            /* 计算沿轴的元素起始偏移 */
            size_t base = o * outer_stride + j;

            /* 1. 找最大值（数值稳定性） */
            float max_val = SELFLNN_NEG_INF;
            for (int ai = 0; ai < axis_size; ai++) {
                float val = data_out[base + (size_t)ai * axis_stride];
                if (val > max_val) max_val = val;
            }

            /* 2. 计算exp并求和 */
            float sum = 0.0f;
            for (int ai = 0; ai < axis_size; ai++) {
                float val = expf(data_out[base + (size_t)ai * axis_stride] - max_val);
                data_out[base + (size_t)ai * axis_stride] = val;
                sum += val;
            }

            /* 3. 归一化 */
            float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            for (int ai = 0; ai < axis_size; ai++) {
                data_out[base + (size_t)ai * axis_stride] *= inv_sum;
            }
        }
    }

    return SELFLNN_SUCCESS;
}