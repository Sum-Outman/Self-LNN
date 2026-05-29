/**
 * @file unified_signal_processor_training.h
 * @brief 多模态数据混合训练策略接口
 *
 * 实现需求20.3: 多模态数据混合策略
 * - 课程学习(Curriculum Learning): 难度渐进式增长
 * - 动态比例(Dynamic Ratio): 根据质量指标动态调整采样比例
 * - 模态丢弃(Modality Dropout): 随机丢弃模态增强鲁棒性
 * - 批量混合训练: 统一多模态数据联合训练
 */
#ifndef SELFLNN_UNIFIED_SIGNAL_PROCESSOR_TRAINING_H
#define SELFLNN_UNIFIED_SIGNAL_PROCESSOR_TRAINING_H

#include "selflnn/multimodal/unified_signal_processor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取课程学习当前难度阈值
 * @param current_iteration 当前迭代次数
 * @param total_iterations 总迭代次数
 * @param config 课程学习配置
 * @return 当前难度阈值 [0-1]
 */
float unified_signal_processor_get_curriculum_threshold(
    size_t current_iteration,
    size_t total_iterations,
    const CurriculumConfig* config);

/**
 * @brief 根据模态质量计算动态采样比例
 * @param modality_quality 各模态质量指标数组[4]
 * @param quality_count 质量指标数量（应=4）
 * @param config 动态比例配置
 * @param ratios 输出采样比例数组[4]
 * @param ratio_count 比例数组大小（应=4）
 * @return 0=成功，-1=失败
 */
int unified_signal_processor_compute_dynamic_ratios(
    const float* modality_quality, size_t quality_count,
    const DynamicRatioConfig* config,
    float* ratios, size_t ratio_count);

/**
 * @brief 应用模态丢弃策略
 * @param processor 统一信号处理器句柄
 * @param config 模态丢弃配置
 * @param current_iteration 当前迭代次数
 * @param modality_active 输出各模态活跃标志[4]
 * @param active_count 活跃标志数组大小（应=4）
 * @return 活跃模态数量，-1=失败
 */
int unified_signal_processor_apply_modality_dropout(
    UnifiedSignalProcessor* processor,
    const ModalityDropoutConfig* config,
    size_t current_iteration,
    int* modality_active, size_t active_count);

/**
 * @brief 批量混合训练主函数
 * @param processor 统一信号处理器句柄
 * @param vision 视觉输入数组
 * @param vision_count 视觉样本数
 * @param audio 音频输入数组
 * @param audio_count 音频样本数
 * @param text 文本输入数组
 * @param text_count 文本样本数
 * @param sensor 传感器输入数组
 * @param sensor_count 传感器样本数
 * @param targets 目标输出数组
 * @param target_count 目标输出数量
 * @param mixing_config 数据混合配置
 * @param modality_quality 各模态质量指标
 * @param current_iteration 当前迭代次数
 * @param total_iterations 总迭代次数
 * @param losses 输出各样本损失值
 * @param loss_count 损失值数组大小
 * @return 0=成功，-1=失败
 */
int unified_signal_processor_train_batch_mixed(
    UnifiedSignalProcessor* processor,
    const VisionInput* vision, size_t vision_count,
    const AudioInput* audio, size_t audio_count,
    const TextInput* text, size_t text_count,
    const SensorInput* sensor, size_t sensor_count,
    const float* targets, size_t target_count,
    const DataMixingConfig* mixing_config,
    const float* modality_quality,
    size_t current_iteration,
    size_t total_iterations,
    float* losses, size_t loss_count);

/**
 * @brief 获取默认数据混合配置
 * @return DataMixingConfig 默认配置（课程学习+动态比例+模态丢弃均启用）
 */
DataMixingConfig unified_signal_processor_get_default_mixing_config(void);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_UNIFIED_SIGNAL_PROCESSOR_TRAINING_H */
