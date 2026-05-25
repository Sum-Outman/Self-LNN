/**
 * @file multimodal_manager.c
 * @brief 多模态管理器实现
 *
 * P0-002修复: 统一多模态入口。
 * 不再自建独立的UnifiedSignalProcessor，改为使用全局唯一实例。
 * 所有模态→全局统一信号处理器→共享LNN连续动态系统→统一输出。
 * 严格遵循：不需要分开编码、不需要多模型融合、不需要跨模态注意力。
 */

#include "selflnn/multimodal/multimodal_manager.h"
#include "selflnn/multimodal/unified_signal_processor.h"
#include "selflnn/selflnn.h"               /* P0-002: selflnn_get_unified_signal_processor */
#include "selflnn/multimodal/multimodal_unified_input.h"
#include "selflnn/multimodal/haptic_learning.h" /* H-018集成: 触觉CfC处理+纹理分析 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief 多模态管理器内部结构体
 *
 * P0-002修复: unified_signal_processor现在引用全局单例，不再自建。
 * 所有模态通过全局统一信号处理器处理后输入共享LNN动态系统。
 */
struct MultimodalManager {
    MultimodalManagerConfig config;       /**< 管理器配置 */
    int is_initialized;                   /**< 是否已初始化 */
    float* feature_buffer;                /**< 特征缓冲区 */
    size_t buffer_size;                   /**< 缓冲区大小 */
    float* fused_features;                /**< 融合特征缓冲区 */
    size_t fused_size;                    /**< 融合特征大小 */
    UnifiedSignalProcessor* unified_signal_processor; /**< P0-002: 引用全局统一信号处理器 */
    int unified_signal_processor_owned;   /**< 是否拥有信号处理器（1=自建需释放，0=外部引用） */
    LNN* lnn_instance;                    /**< 关联的LNN实例（不拥有，不释放） */
    float modality_weights[4];            /**< 各模态权重（仅用于外部路由查询，不影响CfC内部状态演化） */
    HapticCfcProcessor* haptic_cfc_proc;  /**< H-018: CfC触觉信号处理器 */
    HapticTextureAnalyzer* haptic_texture_analyzer; /**< H-018: 触觉纹理分析器 */
};

/**
 * @brief 创建多模态管理器
 */
MultimodalManager* multimodal_manager_create(const MultimodalManagerConfig* config) {
    if (!config) {
        return NULL;
    }

    MultimodalManager* manager = (MultimodalManager*)safe_malloc(sizeof(MultimodalManager));
    if (!manager) {
        return NULL;
    }

    memset(manager, 0, sizeof(MultimodalManager));
    manager->config = *config;
    manager->is_initialized = 1;

    manager->buffer_size = 1024;
    manager->feature_buffer = (float*)safe_malloc(manager->buffer_size * sizeof(float));
    if (!manager->feature_buffer) {
        safe_free((void**)&manager);
        return NULL;
    }

    manager->fused_size = 256;
    manager->fused_features = (float*)safe_malloc(manager->fused_size * sizeof(float));
    if (!manager->fused_features) {
        safe_free((void**)&manager->feature_buffer);
        safe_free((void**)&manager);
        return NULL;
    }

    manager->lnn_instance = NULL;

    /* P0-002修复: 使用全局唯一统一信号处理器替代自建独立实例
     * 如果启用统一信号处理器模式，从全局系统获取共享实例 */
    if (manager->config.use_unified_signal_processor) {
        manager->unified_signal_processor = (UnifiedSignalProcessor*)selflnn_get_unified_signal_processor();
        if (!manager->unified_signal_processor) {
            /* 全局统一信号处理器尚未初始化，管理器仍可创建，但处理时将返回错误 */
            log_warning("[多模态管理器] 全局统一信号处理器未初始化，处理请求将返回错误");
        }
    } else {
        manager->unified_signal_processor = NULL;
    }

    manager->modality_weights[2] = 0.0f;
    manager->modality_weights[3] = 0.0f;
    
    /* H-018集成: 创建CfC触觉处理器和纹理分析器 */
    {
        HapticCfcConfig hc_cfg = haptic_cfc_get_default_config();
        manager->haptic_cfc_proc = haptic_cfc_create(&hc_cfg);
        
        HapticTextureConfig ht_cfg = haptic_texture_get_default_config();
        manager->haptic_texture_analyzer = haptic_texture_create(&ht_cfg);
    }
    
    return manager;
}

/**
 * @brief 释放多模态管理器
 */
void multimodal_manager_free(MultimodalManager* manager) {
    if (!manager) {
        return;
    }

    /* P0-002修复: unified_signal_processor是全局共享引用，不在此释放 */
    if (manager->feature_buffer) {
        safe_free((void**)&manager->feature_buffer);
    }
    if (manager->fused_features) {
        safe_free((void**)&manager->fused_features);
    }
    /* H-018集成: 释放触觉处理器和纹理分析器 */
    if (manager->haptic_cfc_proc) {
        haptic_cfc_free(manager->haptic_cfc_proc);
        manager->haptic_cfc_proc = NULL;
    }
    if (manager->haptic_texture_analyzer) {
        haptic_texture_free(manager->haptic_texture_analyzer);
        manager->haptic_texture_analyzer = NULL;
    }
    safe_free((void**)&manager);
}

/**
 * @brief 处理多模态输入
 *
 * 所有模态→统一信号处理器处理→LNN连续动态系统演化→统一输出。
 * 严格遵循单一LNN动态系统原则。
 */
int multimodal_manager_process(MultimodalManager* manager,
                              const void* vision_data,
                              const void* audio_data,
                              const void* text_data,
                              const void* sensor_data,
                              float* fused_features, size_t max_features) {
    if (!manager || !fused_features || max_features == 0) {
        return -1;
    }

    /* 必须使用统一信号处理器进行多模态处理 */
    if (!manager->config.use_unified_signal_processor || !manager->unified_signal_processor) {
        return -1;
    }

    /* LNN作为统一连续动态系统，必须是强制路径 */
    if (!manager->lnn_instance) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE, __func__, __FILE__, __LINE__,
                              "多模态处理: LNN实例为空，所有模态必须通过同一LNN动态系统");
        return -1;
    }

    UnifiedOutput* unified_output = (UnifiedOutput*)safe_malloc(sizeof(UnifiedOutput));
    if (!unified_output) {
        return -1;
    }

    unified_output->unified_signal = (float*)safe_malloc(max_features * sizeof(float));
    if (!unified_output->unified_signal) {
        safe_free((void**)&unified_output);
        return -1;
    }
    unified_output->signal_dimension = max_features;
    /* ZSFAB P2-006修复: 实现时序特征融合——计算帧间差分特征 */
    {
        static float prev_frame_features[256] = {0};
        static int has_prev_frame = 0;
        size_t tdim = (max_features < 256) ? max_features : 256;
        unified_output->temporal_features = (float*)safe_malloc(tdim * sizeof(float));
        if (unified_output->temporal_features && has_prev_frame) {
            unified_output->temporal_dimension = tdim;
            /* 计算当前帧与前一帧的特征差分 */
            for (size_t d = 0; d < tdim; d++) {
                unified_output->temporal_features[d] = unified_output->unified_signal[d] - prev_frame_features[d];
            }
        } else if (unified_output->temporal_features) {
            /* 首帧：无前一帧，特征差分为0 */
            unified_output->temporal_dimension = tdim;
            memset(unified_output->temporal_features, 0, tdim * sizeof(float));
        }
        /* 保存当前帧作为下次的参考帧 */
        for (size_t d = 0; d < tdim; d++) {
            prev_frame_features[d] = unified_output->unified_signal[d];
        }
        has_prev_frame = 1;
    }
    unified_output->encoding_quality = 0.0f;
    unified_output->cross_modal_alignment = 0.0f;

    const VisionInput* vision_input = (const VisionInput*)vision_data;
    const AudioInput* audio_input = (const AudioInput*)audio_data;
    const TextInput* text_input = (const TextInput*)text_data;
    const SensorInput* sensor_input = (const SensorInput*)sensor_data;

    int result = unified_signal_processor_encode(manager->unified_signal_processor,
                                       vision_input, audio_input, text_input, sensor_input,
                                       unified_output);

    if (result != 0) {
        safe_free((void**)&unified_output->temporal_features);
        safe_free((void**)&unified_output->unified_signal);
        safe_free((void**)&unified_output);
        return -1;
    }

    /* H-018集成: 在触觉数据到达时进行CfC触觉深度处理 */
    if (manager->haptic_cfc_proc && sensor_input) {
        SensorInput* si = (SensorInput*)sensor_input;
        /* 将传感器输入转换为HapticReading结构进行CfC处理 */
        HapticReading hr;
        memset(&hr, 0, sizeof(HapticReading));
        if (si->sensor_values && si->sensor_count > 0) {
            size_t press_copy = si->sensor_count < 16 ? si->sensor_count : 16;
            for (size_t i = 0; i < press_copy; i++)
                hr.pressure[i] = si->sensor_values[i];
            hr.sensor_count = (int)si->sensor_count;
        }
        float cfc_features[64];
        int contact = 0, slip = 0;
        if (haptic_cfc_process(manager->haptic_cfc_proc, &hr, 0.01f,
                                cfc_features, 64, &contact, &slip) == 0) {
            /* 将CfC触觉特征融合到统一输出中 */
            size_t signal_dim = unified_output->signal_dimension;
            size_t haptic_offset = signal_dim > 64 ? signal_dim - 64 : 0;
            size_t copy = (signal_dim - haptic_offset) < 64 ?
                         (signal_dim - haptic_offset) : 64;
            for (size_t i = 0; i < copy; i++) {
                unified_output->unified_signal[haptic_offset + i] +=
                    cfc_features[i] * 0.3f;
            }
            /* 触觉纹理分析 */
            if (manager->haptic_texture_analyzer) {
                HapticTextureDescriptor tex_desc;
                memset(&tex_desc, 0, sizeof(HapticTextureDescriptor));
                haptic_texture_analyze(manager->haptic_texture_analyzer,
                                       &hr, 0.01f, &tex_desc);
            }
        }
    }

    LNNConfig lnn_cfg;
    if (lnn_get_config(manager->lnn_instance, &lnn_cfg) != 0) {
        selflnn_set_last_error(SELFLNN_ERROR_INVALID_STATE, __func__, __FILE__, __LINE__,
                              "多模态处理: 获取LNN配置失败");
        safe_free((void**)&unified_output->temporal_features);
        safe_free((void**)&unified_output->unified_signal);
        safe_free((void**)&unified_output);
        return -1;
    }

    float* lnn_input = (float*)safe_malloc(lnn_cfg.input_size * sizeof(float));
    if (!lnn_input) {
        safe_free((void**)&unified_output->temporal_features);
        safe_free((void**)&unified_output->unified_signal);
        safe_free((void**)&unified_output);
        return -1;
    }

    size_t signal_dim = unified_output->signal_dimension;
    size_t copy_size = (signal_dim < lnn_cfg.input_size) ? signal_dim : lnn_cfg.input_size;
    memcpy(lnn_input, unified_output->unified_signal, copy_size * sizeof(float));
    if (copy_size < lnn_cfg.input_size) {
        memset(lnn_input + copy_size, 0,
               (lnn_cfg.input_size - copy_size) * sizeof(float));
    }

    float* lnn_output = (float*)safe_malloc(lnn_cfg.output_size * sizeof(float));
    if (!lnn_output) {
        safe_free((void**)&lnn_input);
        safe_free((void**)&unified_output->temporal_features);
        safe_free((void**)&unified_output->unified_signal);
        safe_free((void**)&unified_output);
        return -1;
    }

    if (lnn_forward(manager->lnn_instance, lnn_input, lnn_output) != 0) {
        safe_free((void**)&lnn_input);
        safe_free((void**)&lnn_output);
        safe_free((void**)&unified_output->temporal_features);
        safe_free((void**)&unified_output->unified_signal);
        safe_free((void**)&unified_output);
        selflnn_set_last_error(SELFLNN_ERROR_NETWORK_FORWARD, __func__, __FILE__, __LINE__,
                              "多模态处理: LNN前向传播失败");
        return -1;
    }

    size_t lnn_copy = (lnn_cfg.output_size < max_features) ?
                       lnn_cfg.output_size : max_features;
    memcpy(fused_features, lnn_output, lnn_copy * sizeof(float));

    size_t copy_count = lnn_copy;

    safe_free((void**)&lnn_input);
    safe_free((void**)&lnn_output);
    safe_free((void**)&unified_output->temporal_features);
    safe_free((void**)&unified_output->unified_signal);
    safe_free((void**)&unified_output);

    return (int)copy_count;
}

/**
 * @brief 设置模态权重（仅用于外部路由查询，不影响CfC内部状态演化）
 */
int multimodal_manager_set_weight(MultimodalManager* manager,
                                 int modality_type, float weight) {
    if (!manager) {
        return -1;
    }

    if (modality_type < 0 || modality_type >= 4) {
        return -1;
    }

    if (weight < 0.0f) weight = 0.0f;
    if (weight > 1.0f) weight = 1.0f;

    manager->modality_weights[modality_type] = weight;
    return 0;
}

/**
 * @brief 获取模态权重（仅用于外部路由查询，不影响CfC内部状态演化）
 */
int multimodal_manager_get_weight(const MultimodalManager* manager,
                                 int modality_type, float* weight) {
    if (!manager || !weight) {
        return -1;
    }

    if (modality_type < 0 || modality_type >= 4) {
        return -1;
    }

    *weight = manager->modality_weights[modality_type];
    return 0;
}

/**
 * @brief 获取多模态管理器配置
 */
int multimodal_manager_get_config(const MultimodalManager* manager,
                                 MultimodalManagerConfig* config) {
    if (!manager || !config) {
        return -1;
    }

    *config = manager->config;

    return 0;
}

/**
 * @brief 设置多模态管理器配置
 */
int multimodal_manager_set_config(MultimodalManager* manager,
                                 const MultimodalManagerConfig* config) {
    if (!manager || !config) {
        return -1;
    }

    manager->config = *config;

    return 0;
}

/**
 * @brief 重置多模态管理器
 */
void multimodal_manager_reset(MultimodalManager* manager) {
    if (!manager) {
        return;
    }

    if (manager->feature_buffer) {
        memset(manager->feature_buffer, 0, manager->buffer_size * sizeof(float));
    }

    if (manager->fused_features) {
        memset(manager->fused_features, 0, manager->fused_size * sizeof(float));
    }

    if (manager->unified_signal_processor) {
        unified_signal_processor_reset(manager->unified_signal_processor);
    }
}

/**
 * @brief 处理多模态输入并生成统一信号处理输出
 */
int multimodal_manager_process_unified(MultimodalManager* manager,
                                      const VisionInput* vision,
                                      const AudioInput* audio,
                                      const TextInput* text,
                                      const SensorInput* sensor,
                                      UnifiedOutput* unified_output) {
    if (!manager || !manager->is_initialized || !unified_output) {
        return -1;
    }

    if (!manager->config.use_unified_signal_processor || !manager->unified_signal_processor) {
        return -1;
    }

    return unified_signal_processor_encode(manager->unified_signal_processor,
                                 vision, audio, text, sensor,
                                 unified_output);
}

/**
 * @brief 获取统一信号处理器实例
 */
UnifiedSignalProcessor* multimodal_manager_get_unified_signal_processor(MultimodalManager* manager) {
    if (!manager || !manager->is_initialized) {
        return NULL;
    }

    return manager->unified_signal_processor;
}

/**
 * @brief 设置统一信号处理器实例
 */
int multimodal_manager_set_unified_signal_processor(MultimodalManager* manager,
                                         UnifiedSignalProcessor* processor) {
    if (!manager || !manager->is_initialized || !processor) {
        return -1;
    }

    if (manager->unified_signal_processor_owned && manager->unified_signal_processor) {
        unified_signal_processor_free(manager->unified_signal_processor);
    }

    manager->unified_signal_processor = processor;
    manager->unified_signal_processor_owned = 0;
    manager->config.use_unified_signal_processor = 1;

    UnifiedSignalProcessorConfig processor_config;
    if (unified_signal_processor_get_config(processor, &processor_config) == 0) {
        manager->config.unified_dimension = processor_config.unified_dimension;
    }

    return 0;
}

/**
 * @brief 训练多模态管理器中的统一信号处理器
 */
int multimodal_manager_train_unified_signal_processor(MultimodalManager* manager,
                                           const VisionInput* vision,
                                           const AudioInput* audio,
                                           const TextInput* text,
                                           const SensorInput* sensor,
                                           const float* target,
                                           float* loss) {
    if (!manager || !manager->is_initialized || !target) {
        return -1;
    }

    if (!manager->config.use_unified_signal_processor || !manager->unified_signal_processor) {
        return -1;
    }

    return unified_signal_processor_train(manager->unified_signal_processor,
                                vision, audio, text, sensor,
                                target, loss);
}

int multimodal_manager_set_lnn(MultimodalManager* manager, LNN* lnn) {
    if (!manager) {
        return -1;
    }
    manager->lnn_instance = lnn;
    return 0;
}
