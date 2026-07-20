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
 *   cfc.h（统一入口）                                    — 对外暴露网络级API
 *
 * 本文件作为统一入口，同时引入细胞级和网络级API。
 * 所有外部模块应该包含 cfc.h 获取完整CfC能力。
 *
 * 关键规则：所有其他模块 MUST 通过主LNN（lnn.h）使用CfC能力，
 * 禁止直接创建独立CfC网络绕过LNN统一框架！
 */

/* ================================================================
 * CfC网络级API统一入口
 *
 * H-FIX-001: 移除12处重复声明。cfc_network.h（本文件第6行已include）
 * 已提供全部CfC网络级API声明。此处仅保留注释作为API文档索引。
 *
 * 通过 cfc_network.h 提供的函数：
 *   cfc_create()              — 创建CfC网络
 *   cfc_free()                — 释放CfC网络
 *   cfc_forward()             — 前向传播（已通过M-001修复移除重复）
 *   cfc_backward()            — 反向传播
 *   cfc_accumulate_gradients()— 累积梯度
 *   cfc_save() / cfc_load()   — 模型保存/加载
 *   cfc_set_config() / cfc_get_config() — 配置管理
 *   cfc_reset()               — 重置状态
 *   cfc_get_stats()           — 统计信息
 *   cfc_get_weight_matrix()   — 获取权重矩阵
 *   cfc_get_bias_vector()     — 获取偏置向量
 *
 * 外部模块只需包含 cfc.h 即可获得完整CfC能力（经 cfc_network.h 传递）。
 * ================================================================ */

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_CFC_H