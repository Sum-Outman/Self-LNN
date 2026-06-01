/**
 * @file tensor.h
 * @brief 张量核心头文件
 * 
 * 提供张量数据结构和基本操作函数。
 * 张量是多维数组，用于神经网络计算。
 */

#ifndef SELFLNN_CORE_TENSOR_H
#define SELFLNN_CORE_TENSOR_H

#include <stddef.h>

/** @brief 张量最大维度数（用于VLA数组分配） */
#define SELFLNN_MAX_TENSOR_DIMS 8

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 张量数据类型
 */
typedef enum {
    TENSOR_FLOAT32 = 0,    /**< 32位浮点数 */
    TENSOR_FLOAT64 = 1,    /**< 64位浮点数 */
    TENSOR_INT32 = 2,      /**< 32位整数 */
    TENSOR_INT64 = 3,      /**< 64位整数 */
    TENSOR_UINT8 = 4,      /**< 8位无符号整数 */
    TENSOR_UINT16 = 5,     /**< 16位无符号整数 */
    TENSOR_UINT32 = 6,     /**< 32位无符号整数 */
    TENSOR_UINT64 = 7,     /**< 64位无符号整数 */
    TENSOR_BOOL = 8        /**< 布尔值 */
} TensorDataType;

/**
 * @brief 张量存储格式
 */
typedef enum {
    TENSOR_FORMAT_DENSE = 0,   /**< 稠密存储 */
    TENSOR_FORMAT_SPARSE = 1,  /**< 稀疏存储 */
    TENSOR_FORMAT_COMPRESSED = 2 /**< 压缩存储 */
} TensorStorageFormat;

/**
 * @brief 张量结构体
 */
typedef struct {
    TensorDataType dtype;      /**< 数据类型 */
    TensorStorageFormat format; /**< 存储格式 */
    int ndim;                  /**< 维度数量 */
    int* shape;                /**< 形状数组（长度ndim） */
    size_t* strides;           /**< 步幅数组（长度ndim） */
    void* data;                /**< 数据指针 */
    size_t size;               /**< 数据大小（字节） */
    int owner;                 /**< 是否拥有数据内存（1=是，0=否） */
    int device;                /**< 设备标识（0=CPU，1=GPU） */
} Tensor;

/**
 * @brief 张量描述符
 */
#ifndef SELFLNN_CORE_COMMON_TENSOR_DESC
#define SELFLNN_CORE_COMMON_TENSOR_DESC
typedef struct {
    TensorDataType dtype;
    int ndim;
    int shape[8];  // 最多8维
} TensorDescriptor;
#endif

/**
 * @brief 创建张量
 * 
 * @param shape 形状数组
 * @param ndim 维度数量
 * @param dtype 数据类型
 * @return Tensor* 新张量，失败返回NULL
 */
Tensor* tensor_create(const int* shape, int ndim, TensorDataType dtype);

/**
 * @brief 从现有数据创建张量（不复制数据）
 * 
 * @param data 数据指针
 * @param shape 形状数组
 * @param ndim 维度数量
 * @param dtype 数据类型
 * @return Tensor* 新张量，失败返回NULL
 */
Tensor* tensor_wrap(void* data, const int* shape, int ndim, TensorDataType dtype);

/**
 * @brief 释放张量
 * 
 * @param tensor 要释放的张量
 */
void tensor_free(Tensor* tensor);

/**
 * @brief 复制张量
 * 
 * @param src 源张量
 * @return Tensor* 新张量副本，失败返回NULL
 */
Tensor* tensor_clone(const Tensor* src);

/**
 * @brief 获取张量元素总数
 * 
 * @param tensor 张量
 * @return size_t 元素总数
 */
size_t tensor_numel(const Tensor* tensor);

/**
 * @brief 获取张量数据大小（字节）
 * 
 * @param tensor 张量
 * @return size_t 数据大小（字节）
 */
size_t tensor_size_bytes(const Tensor* tensor);

/**
 * @brief 张量重塑
 * 
 * @param tensor 要重塑的张量
 * @param new_shape 新形状数组
 * @param new_ndim 新维度数量
 * @return int 成功返回0，失败返回错误码
 */
int tensor_reshape(Tensor* tensor, const int* new_shape, int new_ndim);

/**
 * @brief 张量切片
 * 
 * @param tensor 源张量
 * @param start 起始索引数组
 * @param end 结束索引数组
 * @param step 步长数组
 * @param ndim 切片维度数量
 * @return Tensor* 切片张量，失败返回NULL
 */
Tensor* tensor_slice(const Tensor* tensor, const int* start, const int* end, const int* step, int ndim);

/**
 * @brief 张量转置
 * 
 * @param tensor 源张量
 * @param perm 维度排列数组
 * @param ndim 维度数量
 * @return Tensor* 转置张量，失败返回NULL
 */
Tensor* tensor_transpose(const Tensor* tensor, const int* perm, int ndim);

/**
 * @brief 张量填充（用指定值填充）
 * 
 * @param tensor 目标张量
 * @param value 填充值（根据数据类型解释）
 */
void tensor_fill(Tensor* tensor, float value);

/**
 * @brief 张量归零
 * 
 * @param tensor 目标张量
 */
void tensor_zero(Tensor* tensor);

/* ================================================================
 * 核心张量运算 — 矩阵乘法、卷积、池化、激活函数、逐元素操作
 * 100%纯C实现，支持OpenMP多线程并行
 * ================================================================ */

/**
 * @brief 矩阵乘法 C = A × B
 * 
 * 支持2D矩阵乘法和批量矩阵乘法（第一维为批次）。
 * A形状: [m, k] 或 [batch, m, k]
 * B形状: [k, n] 或 [batch, k, n]
 * C必须预先分配正确形状: [m, n] 或 [batch, m, n]
 * 使用OpenMP并行化最外层循环。
 * 
 * @param a 左操作数张量
 * @param b 右操作数张量
 * @param c 输出张量（必须预先分配）
 * @return int 成功返回0，失败返回错误码
 */
int tensor_matmul(const Tensor* a, const Tensor* b, Tensor* c);

/**
 * @brief 逐元素加法（支持广播）
 * 
 * NumPy风格广播规则：从最后一维开始向前对齐，
 * 维度为1的可广播到任意大小，缺失维度视为1。
 * result必须预先分配正确形状（广播结果形状）。
 * 
 * 广播示例:
 *   [3, 4] + [4]    → [3, 4]
 *   [3, 1] + [1, 4] → [3, 4]
 *   [3, 4] + [3, 4] → [3, 4]
 *   [2, 3] + [1, 3] → [2, 3]
 * 
 * @param a 左操作数张量
 * @param b 右操作数张量
 * @param result 输出张量（必须预先分配）
 * @return int 成功返回0，失败返回错误码
 */
int tensor_add(const Tensor* a, const Tensor* b, Tensor* result);

/**
 * @brief 2D卷积运算
 * 
 * 输入形状: [N, C_in, H, W]
 * 卷积核形状: [C_out, C_in, kH, kW]
 * 输出形状: [N, C_out, H_out, W_out]
 * 
 * 输出尺寸计算:
 *   H_out = floor((H + 2*pad_h - dil_h*(kH-1) - 1) / stride_h + 1)
 *   W_out = floor((W + 2*pad_w - dil_w*(kW-1) - 1) / stride_w + 1)
 * 
 * @param input 输入张量 [N, C_in, H, W]
 * @param kernel 卷积核张量 [C_out, C_in, kH, kW]
 * @param output 输出张量（必须预先分配正确形状）
 * @param stride_h 垂直步长
 * @param stride_w 水平步长
 * @param padding_h 垂直填充
 * @param padding_w 水平填充
 * @param dilation_h 垂直膨胀
 * @param dilation_w 水平膨胀
 * @return int 成功返回0，失败返回错误码
 */
int tensor_conv2d(const Tensor* input, const Tensor* kernel, Tensor* output,
                  int stride_h, int stride_w, int padding_h, int padding_w,
                  int dilation_h, int dilation_w);

/**
 * @brief 2D最大池化
 * 
 * 输入形状: [N, C, H, W]
 * 输出形状: [N, C, H_out, W_out]
 * 
 * 输出尺寸计算:
 *   H_out = floor((H + 2*pad_h - kH) / stride_h + 1)
 *   W_out = floor((W + 2*pad_w - kW) / stride_w + 1)
 * 
 * @param input 输入张量 [N, C, H, W]
 * @param output 输出张量（必须预先分配正确形状）
 * @param kernel_h 池化核高度
 * @param kernel_w 池化核宽度
 * @param stride_h 垂直步长
 * @param stride_w 水平步长
 * @param padding_h 垂直填充
 * @param padding_w 水平填充
 * @return int 成功返回0，失败返回错误码
 */
int tensor_maxpool2d(const Tensor* input, Tensor* output,
                     int kernel_h, int kernel_w, int stride_h, int stride_w,
                     int padding_h, int padding_w);

/**
 * @brief ReLU激活函数（原地操作）
 * 
 * relu(x) = max(0, x)
 * 
 * @param tensor 输入/输出张量
 * @return int 成功返回0，失败返回错误码
 */
int tensor_relu(Tensor* tensor);

/**
 * @brief Sigmoid激活函数（原地操作）
 * 
 * sigmoid(x) = 1 / (1 + exp(-x))
 * 
 * @param tensor 输入/输出张量
 * @return int 成功返回0，失败返回错误码
 */
int tensor_sigmoid(Tensor* tensor);

/**
 * @brief Tanh激活函数（原地操作）
 * 
 * tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * 
 * @param tensor 输入/输出张量
 * @return int 成功返回0，失败返回错误码
 */
int tensor_tanh(Tensor* tensor);

/**
 * @brief Softmax归一化
 * 
 * softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
 * 沿指定轴进行归一化。
 * 
 * @param input 输入张量
 * @param output 输出张量（必须预先分配与输入相同形状）
 * @param axis 归一化轴（-1表示最后一维）
 * @return int 成功返回0，失败返回错误码
 */
int tensor_softmax(const Tensor* input, Tensor* output, int axis);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_CORE_TENSOR_H