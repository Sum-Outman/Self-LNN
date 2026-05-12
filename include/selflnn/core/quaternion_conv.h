/**
 * @file quaternion_conv.h
 * @brief 四元数卷积层
 *
 * 提供四元数卷积（1D和2D）的完整实现。
 * 四元数卷积使用Hamilton乘积替代标量乘法，保持旋转等变性。
 *
 * 数学原理：
 *   四元数卷积核 Q_k = (w_k, x_k, y_k, z_k)
 *   输入四元数 P_i = (w_i, x_i, y_i, z_i)
 *   卷积输出 R_j = Σ_k P_{j*stride + k} * Q_k
 *   其中 * 为Hamilton乘积
 *
 *   两层架构支持：
 *   - quaternion_conv1d_create    : 1D四元数卷积（序列/时间序列）
 *   - quaternion_conv2d_create    : 2D四元数卷积（图像/空间数据）
 *   - 完整的前向/反向传播
 *   - Adam优化器
 *   - 序列化保存/加载
 */

#ifndef SELFLNN_QUATERNION_CONV_H
#define SELFLNN_QUATERNION_CONV_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================
 *   四元数1D卷积层
 * ============================ */

/**
 * @brief 四元数1D卷积配置
 */
typedef struct {
    size_t in_channels;            /**< 输入通道数（特征维度，每个为四元数） */
    size_t out_channels;           /**< 输出通道数 */
    size_t kernel_size;            /**< 卷积核大小 */
    size_t stride;                 /**< 步长 */
    size_t padding;                /**< 填充（0=same, 1=valid模式两端不填充） */
    int use_bias;                  /**< 是否使用偏置 */
    float lr;                      /**< 学习率 */
    float weight_decay;            /**< 权重衰减系数 */
    float init_scale;              /**< 初始化缩放 */
} QuaternionConv1DConfig;

/**
 * @brief 四元数1D卷积层句柄
 */
typedef struct QuaternionConv1D QuaternionConv1D;

/**
 * @brief 创建四元数1D卷积层
 * @param config 配置（必填，不能为NULL）
 * @return 四元数1D卷积层句柄，失败返回NULL
 */
QuaternionConv1D* quaternion_conv1d_create(const QuaternionConv1DConfig* config);

/**
 * @brief 释放四元数1D卷积层
 * @param layer 四元数1D卷积层句柄
 */
void quaternion_conv1d_free(QuaternionConv1D* layer);

/**
 * @brief 四元数1D卷积前向传播
 *
 * 输入布局：[batch_size, in_channels, seq_len]，存储为平坦数组
 * 每个四元数依次存储4个分量(w,x,y,z)
 * 输出布局：[batch_size, out_channels, output_len]
 *
 * @param layer 四元数1D卷积层句柄
 * @param input 输入数据 [batch * in_channels * seq_len * 4]
 * @param batch_size 批次大小
 * @param seq_len 序列长度
 * @param output 输出缓冲区 [batch * out_channels * output_len * 4]
 * @return 成功返回0，失败返回-1
 */
int quaternion_conv1d_forward(QuaternionConv1D* layer,
                               const float* input,
                               size_t batch_size,
                               size_t seq_len,
                               float* output);

/**
 * @brief 四元数1D卷积反向传播
 *
 * @param layer 四元数1D卷积层句柄
 * @param output_grad 输出梯度 [batch * out_channels * output_len * 4]
 * @param input 输入数据（用于计算输入梯度）[batch * in_channels * seq_len * 4]
 * @param batch_size 批次大小
 * @param seq_len 序列长度
 * @param input_grad 输入梯度缓冲区 [batch * in_channels * seq_len * 4]，可以为NULL
 * @return 成功返回0，失败返回-1
 */
int quaternion_conv1d_backward(QuaternionConv1D* layer,
                                const float* output_grad,
                                const float* input,
                                size_t batch_size,
                                size_t seq_len,
                                float* input_grad);

/**
 * @brief 重置四元数1D卷积层优化器状态（学习率不变）
 * @param layer 四元数1D卷积层句柄
 */
void quaternion_conv1d_reset_optimizer(QuaternionConv1D* layer);

/**
 * @brief 设置四元数1D卷积层学习率
 * @param layer 四元数1D卷积层句柄
 * @param lr 新学习率
 */
void quaternion_conv1d_set_lr(QuaternionConv1D* layer, float lr);

/**
 * @brief 获取四元数1D卷积层可训练参数数量
 * @param layer 四元数1D卷积层句柄
 * @return 参数数量（四元数个数×4），失败返回-1
 */
int quaternion_conv1d_param_count(const QuaternionConv1D* layer);

/**
 * @brief 保存四元数1D卷积层到文件
 * @param layer 四元数1D卷积层句柄
 * @param filename 文件名
 * @return 成功返回0，失败返回-1
 */
int quaternion_conv1d_save(const QuaternionConv1D* layer, const char* filename);

/**
 * @brief 从文件加载四元数1D卷积层
 * @param filename 文件名
 * @return 四元数1D卷积层句柄，失败返回NULL
 */
QuaternionConv1D* quaternion_conv1d_load(const char* filename);

/* ============================
 *   四元数2D卷积层
 * ============================ */

/**
 * @brief 四元数2D卷积配置
 */
typedef struct {
    size_t in_channels;            /**< 输入通道数 */
    size_t out_channels;           /**< 输出通道数 */
    size_t kernel_h;               /**< 卷积核高度 */
    size_t kernel_w;               /**< 卷积核宽度 */
    size_t stride_h;               /**< 垂直步长 */
    size_t stride_w;               /**< 水平步长 */
    size_t pad_h;                  /**< 垂直填充 */
    size_t pad_w;                  /**< 水平填充 */
    int use_bias;                  /**< 是否使用偏置 */
    float lr;                      /**< 学习率 */
    float weight_decay;            /**< 权重衰减系数 */
    float init_scale;              /**< 初始化缩放 */
} QuaternionConv2DConfig;

/**
 * @brief 四元数2D卷积层句柄
 */
typedef struct QuaternionConv2D QuaternionConv2D;

/**
 * @brief 创建四元数2D卷积层
 * @param config 配置
 * @return 四元数2D卷积层句柄，失败返回NULL
 */
QuaternionConv2D* quaternion_conv2d_create(const QuaternionConv2DConfig* config);

/**
 * @brief 释放四元数2D卷积层
 * @param layer 四元数2D卷积层句柄
 */
void quaternion_conv2d_free(QuaternionConv2D* layer);

/**
 * @brief 四元数2D卷积前向传播
 *
 * 输入布局：[batch_size, in_channels, height, width]
 * 每个四元数依次存储4个分量(w,x,y,z)
 * 输出布局：[batch_size, out_channels, out_h, out_w]
 *
 * @param layer 四元数2D卷积层句柄
 * @param input 输入数据 [batch * in_channels * height * width * 4]
 * @param batch_size 批次大小
 * @param height 输入高度
 * @param width 输入宽度
 * @param output 输出缓冲区 [batch * out_channels * out_h * out_w * 4]
 * @return 成功返回0，失败返回-1
 */
int quaternion_conv2d_forward(QuaternionConv2D* layer,
                               const float* input,
                               size_t batch_size,
                               size_t height,
                               size_t width,
                               float* output);

/**
 * @brief 四元数2D卷积反向传播
 *
 * @param layer 四元数2D卷积层句柄
 * @param output_grad 输出梯度 [batch * out_channels * out_h * out_w * 4]
 * @param input 输入数据（用于计算输入梯度）
 * @param batch_size 批次大小
 * @param height 输入高度
 * @param width 输入宽度
 * @param input_grad 输入梯度缓冲区，可以为NULL
 * @return 成功返回0，失败返回-1
 */
int quaternion_conv2d_backward(QuaternionConv2D* layer,
                                const float* output_grad,
                                const float* input,
                                size_t batch_size,
                                size_t height,
                                size_t width,
                                float* input_grad);

/**
 * @brief 重置四元数2D卷积层优化器状态
 * @param layer 四元数2D卷积层句柄
 */
void quaternion_conv2d_reset_optimizer(QuaternionConv2D* layer);

/**
 * @brief 设置四元数2D卷积层学习率
 * @param layer 四元数2D卷积层句柄
 * @param lr 新学习率
 */
void quaternion_conv2d_set_lr(QuaternionConv2D* layer, float lr);

/**
 * @brief 获取四元数2D卷积层可训练参数数量
 * @param layer 四元数2D卷积层句柄
 * @return 参数数量（四元数个数×4），失败返回-1
 */
int quaternion_conv2d_param_count(const QuaternionConv2D* layer);

/**
 * @brief 保存四元数2D卷积层到文件
 * @param layer 四元数2D卷积层句柄
 * @param filename 文件名
 * @return 成功返回0，失败返回-1
 */
int quaternion_conv2d_save(const QuaternionConv2D* layer, const char* filename);

/**
 * @brief 从文件加载四元数2D卷积层
 * @param filename 文件名
 * @return 四元数2D卷积层句柄，失败返回NULL
 */
QuaternionConv2D* quaternion_conv2d_load(const char* filename);

/* ============================
 *   四元数卷积工具函数
 * ============================ */

/**
 * @brief Hamilton乘积：两个四元数相乘
 *
 * 公式：
 *   r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
 *   r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
 *   r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
 *   r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
 *
 * @param a 四元数a [4] = {w, x, y, z}
 * @param b 四元数b [4] = {w, x, y, z}
 * @param out 输出四元数 [4] = {w, x, y, z}
 */
static void quaternion_hamilton_product(const float* a, const float* b, float* out)
{
    float w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    float x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    float y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    float z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

/**
 * @brief 四元数标量乘法
 * @param q 四元数 [4]
 * @param s 标量
 * @param out 输出 [4]
 */
static void quaternion_scale(const float* q, float s, float* out)
{
    out[0] = q[0] * s; out[1] = q[1] * s;
    out[2] = q[2] * s; out[3] = q[3] * s;
}

/**
 * @brief 四元数加法
 * @param a 四元数a [4]
 * @param b 四元数b [4]
 * @param out 输出 [4]
 */
static void quaternion_add(const float* a, const float* b, float* out)
{
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1];
    out[2] = a[2] + b[2]; out[3] = a[3] + b[3];
}

/**
 * @brief 四元数共轭
 * @param q 四元数 [4]
 * @param out 输出 [4]
 */
static void quaternion_conj(const float* q, float* out)
{
    out[0] = q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = -q[3];
}

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_QUATERNION_CONV_H */
