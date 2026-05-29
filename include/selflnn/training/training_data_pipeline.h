/**
 * @file training_data_pipeline.h
 * @brief 多模态训练数据管线接口
 *
 * ZSFA-FIX-P0-002: 训练数据流水线预处理集成。
 * 桥接多模态CfC/LNN权重到训练器，提供数据预处理和训练调度功能。
 */

#ifndef SELFLNN_TRAINING_DATA_PIPELINE_H
#define SELFLNN_TRAINING_DATA_PIPELINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 训练数据预处理 — 对训练数据进行归一化和验证
 * @param data_buffer 训练数据缓冲区
 * @param data_size 数据缓冲区大小（字节）
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @return int 成功返回0，失败返回-1
 */
int training_data_pipeline_preprocess(float* data_buffer,
                                       size_t data_size,
                                       size_t input_dim,
                                       size_t output_dim);

/**
 * @brief 多模态训练主函数
 * @param network LNN网络实例
 * @param module_name 模块名称
 * @param input_dim 输入维度
 * @param output_dim 输出维度
 * @param num_samples 样本数
 * @param epochs 训练轮数
 * @param learning_rate 学习率
 * @return int 成功返回0，失败返回-1
 */
int training_pipeline_train_multimodal(void* network, const char* module_name,
                                        int input_dim, int output_dim,
                                        int num_samples, int epochs,
                                        float learning_rate);

/**
 * @brief 所有视觉模块预训练
 */
int training_pipeline_pretrain_all_vision(void* vision_net, void* deep_vision_net,
                                            void* image_seg_net, void* depth_est_net);

/**
 * @brief 所有音频模块预训练
 */
int training_pipeline_pretrain_all_audio(void* speech_net, void* audio_semantic_net,
                                           void* acoustic_scene_net, void* sound_localize_net);

/**
 * @brief 所有传感器模块预训练
 */
int training_pipeline_pretrain_all_sensors(void* sensor_fusion_net, void* slam_net,
                                             void* tactile_net, void* thermal_net);

/**
 * @brief 所有模块统一预训练
 */
int training_pipeline_pretrain_all_modules(void* system_context);

/**
 * @brief 统一信号处理器训练
 */
int training_pipeline_train_unified_processor(void* processor,
                                                size_t total_iterations,
                                                float learning_rate);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_TRAINING_DATA_PIPELINE_H */
