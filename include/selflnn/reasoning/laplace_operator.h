/**
 * @file laplace_operator.h
 * @brief Laplace神经算子接口
 *
 * 基于傅里叶神经算子(FNO)框架，实现函数空间到函数空间的学习映射。
 * 用于求解偏微分方程(PDE)、物理场预测、多分辨率信号处理等。
 */

#ifndef SELFLNN_LAPLACE_OPERATOR_H
#define SELFLNN_LAPLACE_OPERATOR_H

#include <stddef.h>
#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 复数类型
 */
typedef struct {
    double real;
    double imag;
} LaplaceComplex;

/**
 * @brief 1D谱卷积层
 */
typedef struct {
    size_t modes_in;          /**< 输入保留的傅里叶模态数 */
    size_t modes_out;         /**< 输出保留的傅里叶模态数 */
    size_t in_channels;       /**< 输入通道数 */
    size_t out_channels;      /**< 输出通道数 */
    LaplaceComplex* weights;  /**< [out_channels][in_channels][modes_out] 可学习复数权重 */
    double* bias;             /**< [out_channels] 偏置 */
} LaplaceSpectralConv1D;

/**
 * @brief 2D谱卷积层
 */
typedef struct {
    size_t modes1_in;         /**< 第一维输入模态数 */
    size_t modes2_in;         /**< 第二维输入模态数 */
    size_t modes1_out;        /**< 第一维输出模态数 */
    size_t modes2_out;        /**< 第二维输出模态数 */
    size_t in_channels;       /**< 输入通道数 */
    size_t out_channels;      /**< 输出通道数 */
    LaplaceComplex* weights;  /**< [out_channels][in_channels][modes1_out][modes2_out] 复数权重 */
    double* bias;             /**< [out_channels] 偏置 */
} LaplaceSpectralConv2D;

/**
 * @brief 激活函数类型
 */
typedef enum {
    LNO_ACTIVATION_GELU = 0,   /**< GELU激活 */
    LNO_ACTIVATION_RELU = 1,   /**< ReLU激活 */
    LNO_ACTIVATION_SILU = 2,   /**< SiLU/Swish激活 */
    LNO_ACTIVATION_TANH = 3    /**< Tanh激活 */
} LNOActivationType;

/**
 * @brief Laplace神经算子配置
 */
typedef struct {
    size_t num_layers;             /**< 谱卷积层数(1~6) */
    size_t in_channels;            /**< 输入通道数 */
    size_t hidden_channels;        /**< 隐藏层通道数 */
    size_t out_channels;           /**< 输出通道数 */
    size_t modes_1d;               /**< 1D算子保留模态数 */
    size_t modes1_2d;              /**< 2D算子第一维模态数 */
    size_t modes2_2d;              /**< 2D算子第二维模态数 */
    double learning_rate;          /**< 学习率 */
    LNOActivationType activation;  /**< 激活函数类型 */
    int use_batch_norm;            /**< 是否使用批量归一化(0/1) */
} LaplaceOperatorConfig;

/**
 * @brief Laplace神经算子主结构
 */
typedef struct LaplaceNeuralOperator_t {
    LaplaceOperatorConfig config;             /**< 算子配置 */
    size_t input_dim;                         /**< 输入维度(1或2) */

    /* 升维层: 将输入通道映射到隐藏维度 */
    double* lifting_weight;                   /**< [in_channels][hidden_channels] */
    double* lifting_bias;                     /**< [hidden_channels] */

    /* 谱卷积层数组 */
    LaplaceSpectralConv1D* spectral_layers_1d; /**< [num_layers] 1D谱卷积层 */
    LaplaceSpectralConv2D* spectral_layers_2d; /**< [num_layers] 2D谱卷积层 */

    /* 降维层: 将隐藏维度映射到输出通道 */
    double* projection_weight;                /**< [hidden_channels][out_channels] */
    double* projection_bias;                  /**< [out_channels] */

    /* 批量归一化参数(每层每通道) */
    double* bn_gamma;                         /**< [num_layers][hidden_channels] */
    double* bn_beta;                          /**< [num_layers][hidden_channels] */
    double* bn_running_mean;                  /**< [num_layers][hidden_channels] */
    double* bn_running_var;                   /**< [num_layers][hidden_channels] */

    /* 优化器状态(Adam) */
    double* adam_m;                           /**< 一阶动量 */
    double* adam_v;                           /**< 二阶动量 */
    size_t adam_step;                         /**< 优化步数 */
} LaplaceNeuralOperator;

/**
 * @brief 创建Laplace神经算子(1D)
 *
 * @param config 算子配置
 * @return LaplaceNeuralOperator* 成功返回指针，失败返回NULL
 */
LaplaceNeuralOperator* laplace_operator_create_1d(const LaplaceOperatorConfig* config);

/**
 * @brief 创建Laplace神经算子(2D)
 *
 * @param config 算子配置
 * @return LaplaceNeuralOperator* 成功返回指针，失败返回NULL
 */
LaplaceNeuralOperator* laplace_operator_create_2d(const LaplaceOperatorConfig* config);

/**
 * @brief 销毁Laplace神经算子
 *
 * @param op 算子指针
 */
void laplace_operator_destroy(LaplaceNeuralOperator* op);

/**
 * @brief 1D算子前向传播
 *
 * @param op 算子指针
 * @param input 输入数据 [batch][in_channels][grid_size]，grid_size必须为2的幂
 * @param batch 批大小
 * @param grid_size 网格点数
 * @param output 输出数据 [batch][out_channels][grid_size]，预先分配
 * @return int 成功返回0，失败返回-1
 */
int laplace_operator_forward_1d(const LaplaceNeuralOperator* op,
                                const double* input,
                                size_t batch, size_t grid_size,
                                double* output);

/**
 * @brief 2D算子前向传播
 *
 * @param op 算子指针
 * @param input 输入数据 [batch][in_channels][grid_size][grid_size]
 * @param batch 批大小
 * @param grid_size 网格点数(每维)
 * @param output 输出数据 [batch][out_channels][grid_size][grid_size]
 * @return int 成功返回0，失败返回-1
 */
int laplace_operator_forward_2d(const LaplaceNeuralOperator* op,
                                const double* input,
                                size_t batch, size_t grid_size,
                                double* output);

/**
 * @brief 训练1D算子(一步梯度下降)
 *
 * @param op 算子指针
 * @param input 输入数据 [batch][in_channels][grid_size]
 * @param target 目标数据 [batch][out_channels][grid_size]
 * @param batch 批大小
 * @param grid_size 网格点数
 * @return double 当前损失值
 */
double laplace_operator_train_step_1d(LaplaceNeuralOperator* op,
                                      const double* input,
                                      const double* target,
                                      size_t batch, size_t grid_size);

/**
 * @brief 训练2D算子(一步梯度下降)
 *
 * @param op 算子指针
 * @param input 输入数据 [batch][in_channels][grid_size][grid_size]
 * @param target 目标数据 [batch][out_channels][grid_size][grid_size]
 * @param batch 批大小
 * @param grid_size 网格点数
 * @return double 当前损失值
 */
double laplace_operator_train_step_2d(LaplaceNeuralOperator* op,
                                      const double* input,
                                      const double* target,
                                      size_t batch, size_t grid_size);

/**
 * @brief 设置学习率
 *
 * @param op 算子指针
 * @param lr 新学习率
 */
void laplace_operator_set_learning_rate(LaplaceNeuralOperator* op, double lr);

/**
 * @brief 重置优化器状态
 *
 * @param op 算子指针
 */
void laplace_operator_reset_optimizer(LaplaceNeuralOperator* op);

/**
 * @brief 保存算子参数到文件
 *
 * @param op 算子指针
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int laplace_operator_save(const LaplaceNeuralOperator* op, const char* filepath);

/**
 * @brief 从文件加载算子参数
 *
 * @param op 算子指针
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int laplace_operator_load(LaplaceNeuralOperator* op, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LAPLACE_OPERATOR_H */
