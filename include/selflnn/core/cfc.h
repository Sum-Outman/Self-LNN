#ifndef SELFLNN_CFC_H
#define SELFLNN_CFC_H

#include <stddef.h>
#include <stdio.h>
#include "selflnn/core/cfc_network.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file cfc.h
 * @brief 封闭形式连续时间（CfC）液态神经网络 —— 统一API入口
 *
 * CfC（Closed-form Continuous-time）是液态神经网络的核心计算单元。
 * 架构层级：
 *   CfCCell（单元级）→ cfc_cell.h / cfc_cell.c   — 单个CfC细胞的前向/反向/ODE求解
 *   CfCNetwork（网络级）→ cfc_network.h / cfc_network.c — 多层CfC单元组成的网络
 *   cfc.h / cfc.c（统一入口）                           — 对外暴露网络级API
 *
 * 本文件作为统一入口，同时引入细胞级和网络级API。
 * 所有外部模块应该包含 cfc.h 获取完整CfC能力。
 *
 * 关键规则：所有其他模块 MUST 通过主LNN（lnn.h）使用CfC能力，
 * 禁止直接创建独立CfC网络绕过LNN统一框架！
 */

/* ================================================================
 * CfC网络级API统一声明（实现在 src/core/cfc_network.c）
 * 类型定义已通过 cfc_network.h 引入
 * ================================================================ */

/**
 * @brief 创建CfC网络实例
 * @param config 网络配置
 * @return CfC网络句柄，失败返回NULL
 */
CfCNetwork* cfc_create(const CfCNetworkConfig* config);

/**
 * @brief 释放CfC网络实例
 * @param network 网络句柄
 */
void cfc_free(CfCNetwork* network);

/**
 * @brief CfC网络前向传播（连续时间动态演化）
 * @param network CfC网络句柄
 * @param input 输入向量 [input_size]
 * @param hidden_state 隐藏状态缓冲区 [hidden_size]（输入/输出）
 * @param cell_state 细胞状态缓冲区 [hidden_size]（输入/输出）
 * @param output 输出向量缓冲区 [output_size]
 * @return 0成功，负值失败
 */
int cfc_forward(CfCNetwork* network, const float* input,
                float* hidden_state, float* cell_state, float* output);

/**
 * @brief CfC网络反向传播（梯度计算与参数更新）
 * @param network CfC网络句柄
 * @param error 误差向量 [output_size]
 * @param gradient 输入梯度输出缓冲区 [input_size]
 * @param learning_rate 学习率
 * @return 0成功，负值失败
 */
int cfc_backward(CfCNetwork* network, const float* error,
                 float* gradient, float learning_rate);

/**
 * @brief CfC网络累积梯度（批量训练专用）
 * @param network CfC网络句柄
 * @param error 误差向量
 * @param gradient 输入梯度输出
 * @param weight_gradients 权重梯度累积缓冲区
 * @param bias_gradients 偏置梯度累积缓冲区
 * @return 0成功，负值失败
 */
int cfc_accumulate_gradients(CfCNetwork* network, const float* error,
                            float* gradient,
                            float* weight_gradients, float* bias_gradients);

/**
 * @brief 保存CfC网络到文件
 * @param network CfC网络句柄
 * @param file 已打开的文件句柄
 * @return 0成功，负值失败
 */
int cfc_save(const CfCNetwork* network, FILE* file);

/**
 * @brief 从文件加载CfC网络
 * @param network CfC网络句柄
 * @param file 已打开的文件句柄
 * @return 0成功，负值失败
 */
int cfc_load(CfCNetwork* network, FILE* file);

/**
 * @brief 设置CfC网络配置
 * @param network CfC网络句柄
 * @param config 新配置
 * @return 0成功，负值失败
 */
int cfc_set_config(CfCNetwork* network, const CfCNetworkConfig* config);

/**
 * @brief 获取CfC网络配置
 * @param network CfC网络句柄
 * @param config 输出配置
 * @return 0成功，负值失败
 */
int cfc_get_config(const CfCNetwork* network, CfCNetworkConfig* config);

/**
 * @brief 重置CfC网络状态
 * @param network CfC网络句柄
 */
void cfc_reset(CfCNetwork* network);

/**
 * @brief 获取CfC网络统计信息
 * @param network CfC网络句柄
 * @param avg_activation 输出平均激活度
 * @param max_activation 输出最大激活度
 * @param gradient_norm 输出梯度范数
 * @return 0成功，负值失败
 */
int cfc_get_stats(const CfCNetwork* network, float* avg_activation,
                  float* max_activation, float* gradient_norm);

/**
 * @brief 获取CfC网络权重矩阵（用于拉普拉斯分析等）
 * @param network CfC网络句柄
 * @param weight_matrix 输出权重矩阵指针
 * @param weight_count 输出权重元素数量
 * @return 0成功，负值失败
 */
int cfc_get_weight_matrix(CfCNetwork* network, float** weight_matrix, size_t* weight_count);

/**
 * @brief 获取CfC网络偏置向量
 * @param network CfC网络句柄
 * @param bias_vector 输出偏置向量指针
 * @param bias_count 输出偏置元素数量
 * @return 0成功，负值失败
 */
int cfc_get_bias_vector(CfCNetwork* network, float** bias_vector, size_t* bias_count);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_CFC_H