/**
 * @file multimodal_manager.h
 * @brief 多模态管理器接口（兼容适配层）
 *
 * **已弃用**: 新代码应使用 unified_lnn_state.h 作为唯一多模态统一入口。
 * 本文件保留用于向后兼容，内部已路由至 unified_lnn_state_step。
 * 严格遵循：所有模态→统一输入到同一个LNN连续动态系统。
 */

#ifndef SELFLNN_MULTIMODAL_MANAGER_H
#define SELFLNN_MULTIMODAL_MANAGER_H

#include <stddef.h>
#include "selflnn/core/lnn.h"
#include "selflnn/multimodal/unified_signal_processor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 多模态管理器配置
 */
typedef struct {
    int enable_vision;           /**< 是否启用视觉处理 */
    int enable_audio;            /**< 是否启用音频处理 */
    int enable_text;             /**< 是否启用文本处理 */
    int enable_sensor;           /**< 是否启用传感器处理 */
    int fusion_method;           /**< 融合方法 */
    int sync_timestamps;         /**< 是否同步时间戳 */
    int use_unified_signal_processor;     /**< 是否使用统一信号处理器 */
    size_t unified_dimension;    /**< 统一信号维度 */
} MultimodalManagerConfig;

/**
 * @brief 多模态管理器句柄
 */
typedef struct MultimodalManager MultimodalManager;

/**
 * @brief 创建多模态管理器
 *
 * @param config 管理器配置
 * @return MultimodalManager* 管理器句柄，失败返回NULL
 */
MultimodalManager* multimodal_manager_create(const MultimodalManagerConfig* config);

/**
 * @brief 释放多模态管理器
 *
 * @param manager 管理器句柄
 */
void multimodal_manager_free(MultimodalManager* manager);

/**
 * @brief 处理多模态输入
 *
 * 所有模态通过统一信号处理器处理后，输入到同一个LNN连续动态系统进行状态演化，
 * 输出融合特征。严格遵循统一信号→统一演化→统一输出的架构原则。
 *
 * @param manager 管理器句柄
 * @param vision_data 视觉数据 (可为NULL)
 * @param audio_data 音频数据 (可为NULL)
 * @param text_data 文本数据 (可为NULL)
 * @param sensor_data 传感器数据 (可为NULL)
 * @param fused_features 融合特征输出缓冲区
 * @param max_features 最大特征数
 * @return int 成功返回融合特征数，失败返回-1
 */
int multimodal_manager_process(MultimodalManager* manager,
                              const void* vision_data,
                              const void* audio_data,
                              const void* text_data,
                              const void* sensor_data,
                              float* fused_features, size_t max_features);

/**
 * @brief 设置模态权重
 *
 * @param manager 管理器句柄
 * @param modality_type 模态类型
 * @param weight 权重值 (0-1)
 * @return int 成功返回0，失败返回-1
 */
int multimodal_manager_set_weight(MultimodalManager* manager,
                                 int modality_type, float weight);

/**
 * @brief 获取模态权重
 *
 * @param manager 管理器句柄
 * @param modality_type 模态类型
 * @param weight 权重输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int multimodal_manager_get_weight(const MultimodalManager* manager,
                                 int modality_type, float* weight);

/**
 * @brief 获取多模态管理器配置
 *
 * @param manager 管理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int multimodal_manager_get_config(const MultimodalManager* manager,
                                 MultimodalManagerConfig* config);

/**
 * @brief 设置多模态管理器配置
 *
 * @param manager 管理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int multimodal_manager_set_config(MultimodalManager* manager,
                                 const MultimodalManagerConfig* config);

/**
 * @brief 重置多模态管理器
 *
 * @param manager 管理器句柄
 */
void multimodal_manager_reset(MultimodalManager* manager);

/**
 * @brief 处理多模态输入并生成统一信号输出
 *
 * @param manager 管理器句柄
 * @param vision 视觉输入数据
 * @param audio 音频输入数据
 * @param text 文本输入数据
 * @param sensor 传感器输入数据
 * @param unified_output 统一信号输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int multimodal_manager_process_unified(MultimodalManager* manager,
                                      const VisionInput* vision,
                                      const AudioInput* audio,
                                      const TextInput* text,
                                      const SensorInput* sensor,
                                      UnifiedOutput* unified_output);

/**
 * @brief 获取统一信号处理器实例
 *
 * @param manager 管理器句柄
 * @return UnifiedSignalProcessor* 统一信号处理器实例，如果未启用则返回NULL
 */
UnifiedSignalProcessor* multimodal_manager_get_unified_signal_processor(MultimodalManager* manager);

/**
 * @brief 设置统一信号处理器实例
 *
 * @param manager 管理器句柄
 * @param processor 统一信号处理器实例
 * @return int 成功返回0，失败返回-1
 */
int multimodal_manager_set_unified_signal_processor(MultimodalManager* manager,
                                         UnifiedSignalProcessor* processor);

/**
 * @brief 训练多模态管理器中的统一信号处理器
 *
 * @param manager 管理器句柄
 * @param vision 视觉训练数据
 * @param audio 音频训练数据
 * @param text 文本训练数据
 * @param sensor 传感器训练数据
 * @param target 目标信号
 * @param loss 损失输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int multimodal_manager_train_unified_signal_processor(MultimodalManager* manager,
                                           const VisionInput* vision,
                                           const AudioInput* audio,
                                           const TextInput* text,
                                           const SensorInput* sensor,
                                           const float* target,
                                           float* loss);

int multimodal_manager_set_lnn(MultimodalManager* manager, LNN* lnn);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_MULTIMODAL_MANAGER_H
