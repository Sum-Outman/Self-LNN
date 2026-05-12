/**
 * @file multimodal.h
 * @brief 多模态处理接口
 *
 * 根据"所有模态 → 统一输入到同一个连续动态系统 → 统一状态演化 → 统一输出决策"原则，
 * 多模态处理器提供各模态的原始特征提取入口，所有特征最终通过统一LNN动态系统演化。
 * 严格遵循：不需要多模型融合、不需要跨模态注意力。
 */

#ifndef SELFLNN_MULTIMODAL_H
#define SELFLNN_MULTIMODAL_H

#include "selflnn/core/unified_lnn_state.h"
#include <stddef.h>

#ifndef SELFLNN_LNN_H
typedef struct LNN LNN;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 模态类型枚举
 */
typedef enum {
    MODALITY_VISION = 0,
    MODALITY_AUDIO = 1,
    MODALITY_TEXT = 2,
    MODALITY_SENSOR = 3,
    MODALITY_TACTILE = 4,
    MODALITY_PROPRIOCEPTION = 5,
    MODALITY_THERMAL = 6,
    MODALITY_RADAR = 7,
    MODALITY_FUSED = 8
} ModalityType;

/**
 * @brief 视觉数据格式
 */
typedef struct {
    int width;
    int height;
    int channels;
    float* data;
    float* stereo_right;    /* 双目：右眼图像 (NULL=单目), 同宽高通道 */
    float* depth_gt;        /* 可选深度真值用于评估 (NULL=无), width×height */
    double fx;              /* 焦距x (像素), 默认525.0, 用于B=基线*焦距/z */
    double baseline;        /* 双目基线 (米), 默认0.12, 用于z=B*f/disparity */
} VisionData;

/**
 * @brief 音频数据格式
 */
typedef struct {
    int sample_rate;
    int num_samples;
    int num_channels;
    float* data;
} AudioData;

/**
 * @brief 文本数据格式
 */
typedef struct {
    const char* text;
    size_t length;
    int encoding;
} TextData;

/**
 * @brief 传感器数据格式
 */
typedef struct {
    int sensor_type;
    int num_dimensions;
    float* values;
    long timestamp;
} MultimodalSensorData;

/**
 * @brief 融合特征向量
 */
typedef struct {
    int dimension;
    float* features;
    float confidence;
} FusedFeatures;

/**
 * @brief 多模态处理配置
 */
typedef struct {
    int enable_vision;
    int enable_audio;
    int enable_text;
    int enable_sensor;
    int fusion_method;
    int feature_dimension;
} MultimodalConfig;

/**
 * @brief 多模态处理器句柄
 */
typedef struct MultimodalProcessor MultimodalProcessor;

/**
 * @brief 创建多模态处理器
 */
MultimodalProcessor* multimodal_processor_create(const MultimodalConfig* config);

/**
 * @brief 释放多模态处理器
 */
void multimodal_processor_free(MultimodalProcessor* processor);

/**
 * @brief 处理视觉数据
 */
int multimodal_process_vision(MultimodalProcessor* processor, const VisionData* vision_data,
                             float* features, size_t max_features);

/**
 * @brief 处理双目视觉数据：SAD块匹配→视差图→深度估计
 *
 * 从左右眼图像对中提取立体视觉特征。
 * 100%纯C实现：SAD (Sum of Absolute Differences) 块匹配 → 亚像素精化 → 深度。
 * 所有特征统一馈入单一CfC液态神经网络，不建立独立模型。
 *
 * @param processor 多模态处理器
 * @param left  左眼图像 (宽×高, 单通道灰度)
 * @param right 右眼图像 (宽×高, 单通道灰度)
 * @param width 图像宽度
 * @param height 图像高度
 * @param fx 焦距 (像素)
 * @param baseline 基线 (米)
 * @param features 特征输出 [max_features]
 * @param max_features 最大特征数
 * @return 提取的特征数量，失败返回-1
 */
int multimodal_process_stereo(MultimodalProcessor* processor,
                              const float* left, const float* right,
                              int width, int height,
                              double fx, double baseline,
                              float* features, size_t max_features);

/**
 * @brief 处理音频数据
 */
int multimodal_process_audio(MultimodalProcessor* processor, const AudioData* audio_data,
                            float* features, size_t max_features);

/**
 * @brief 处理文本数据
 */
int multimodal_process_text(MultimodalProcessor* processor, const TextData* text_data,
                           float* features, size_t max_features);

/**
 * @brief 处理传感器数据
 */
int multimodal_process_sensor(MultimodalProcessor* processor, const MultimodalSensorData* sensor_data,
                             float* features, size_t max_features);

/**
 * @brief 融合多模态特征（简单拼接，通过单一LNN动态系统演化）
 */
int multimodal_fuse_features(MultimodalProcessor* processor,
                            const float* vision_features, size_t vision_count,
                            const float* audio_features, size_t audio_count,
                            const float* text_features, size_t text_count,
                            const float* sensor_features, size_t sensor_count,
                            float* fused_features, size_t max_fused_features);

/**
 * @brief 获取多模态处理器配置
 */
int multimodal_processor_get_config(const MultimodalProcessor* processor, MultimodalConfig* config);

/**
 * @brief 设置多模态处理器配置
 */
int multimodal_processor_set_config(MultimodalProcessor* processor, const MultimodalConfig* config);

/**
 * @brief 重置多模态处理器
 */
void multimodal_processor_reset(MultimodalProcessor* processor);

/**
 * @brief 设置共享液态神经网络实例
 *
 * 根据"所有模态→统一输入到同一个连续动态系统"原则，
 * 设置系统共享的LNN实例，替代MultimodalProcessor内部自建的独立LNN。
 */
int multimodal_processor_set_lnn(MultimodalProcessor* processor, LNN* lnn);

/**
 * @brief 统一LNN状态处理（P2.1：真正单一LNN统一模态）
 *
 * 将所有模态原始数据直接映射到同一个CfC连续动态系统的状态空间。
 * 不使用模态分离编码器、不进行特征拼接、不使用跨模态注意力。
 * 每个模态通过独立的线性投影注入同一LNN状态空间，在CfC ODE中统一演化。
 *
 * @param processor 多模态处理器句柄
 * @param raw_vision 原始视觉特征数据（可直接为NULL）
 * @param vision_size 视觉数据维度
 * @param raw_audio 原始音频特征数据（可直接为NULL）
 * @param audio_size 音频数据维度
 * @param raw_text 原始文本特征数据（可直接为NULL）
 * @param text_size 文本数据维度
 * @param raw_sensor 原始传感器数据（可直接为NULL）
 * @param sensor_size 传感器数据维度
 * @param output 统一输出缓冲区
 * @param max_output_size 输出缓冲区长���
 * @return int 成功返回输出维度，失败返回-1
 */
int multimodal_processor_process_unified(MultimodalProcessor* processor,
                                       UnifiedLNNState* unified_state,
                                       const float* raw_vision, size_t vision_size,
                                       const float* raw_audio, size_t audio_size,
                                       const float* raw_text, size_t text_size,
                                       const float* raw_sensor, size_t sensor_size,
                                       float* output, size_t max_output_size);

#ifdef __cplusplus
}
#endif

#endif
