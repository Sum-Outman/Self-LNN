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

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_CORE_TENSOR_H