/**
 * @file training_data_pipeline.h
 * @brief 多模态训练数据管线 - 桥接多模态CfC/LNN权重到训练器
 *
 * 提供统一的多模态训练数据管道，将各模态液态神经网络接入训练循环。
 * 100%纯C实现，零外部依赖。
 */
#ifndef SELFLNN_TRAINING_DATA_PIPELINE_H
#define SELFLNN_TRAINING_DATA_PIPELINE_H

#include "selflnn/core/lnn.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 训练单个多模态子网络
 * @param network 液态神经网络句柄
 * @param module_name 模块名称（用于日志）
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param num_samples 训练样本数
 * @param epochs 训练轮数
 * @param learning_rate 学习率
 * @return 0成功，-1失败
 */
int training_pipeline_train_multimodal(LNN* network, const char* module_name,
                                        int input_dim, int output_dim,
                                        int num_samples, int epochs, float learning_rate);

/**
 * @brief 预训练所有视觉子网络
 * @param vision_net 视觉LNN
 * @param deep_vision_net 深度视觉LNN
 * @return 0成功，-1失败
 */
int training_pipeline_pretrain_all_vision(LNN* vision_net, LNN* deep_vision_net);

/**
 * @brief 预训练所有音频子网络
 * @param speech_net 语音识别LNN
 * @param audio_semantic_net 音频语义LNN
 * @return 0成功，-1失败
 */
int training_pipeline_pretrain_all_audio(LNN* speech_net, LNN* audio_semantic_net);

/**
 * @brief 预训练所有传感器子网络
 * @param sensor_fusion_net 传感器融合LNN
 * @param slam_net SLAM LNN
 * @return 0成功，-1失败
 */
int training_pipeline_pretrain_all_sensors(LNN* sensor_fusion_net, LNN* slam_net);

/**
 * @brief 预训练全模块管道
 * @param system_context 系统上下文句柄
 * @return 0成功，-1失败
 */
int training_pipeline_pretrain_all_modules(void* system_context);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TRAINING_DATA_PIPELINE_H */
