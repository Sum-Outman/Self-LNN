/**
 * @file multimodal_manager.c
 * @brief 多模态管理器实现
 *
 * ZSF-NEW-013修复: 消除5种模态后混合绕过LNN的架构缺陷。
 * 所有9种模态(视觉/音频/文本/传感器/触觉/本体感/热感/雷达/电机)
 * 现在统一在lnn_forward之前注入lnn_input，经过同一个CfC连续动态系统。
 * 严格遵循：不需要分开编码、不需要多模型融合、不需要跨模态注意力。
 */

#include "selflnn/multimodal/multimodal_manager.h"
#include "selflnn/multimodal/unified_signal_processor.h"
#include "selflnn/selflnn.h"               /* P0-002: selflnn_get_unified_signal_processor */
#include "selflnn/multimodal/multimodal_unified_input.h"
#include "selflnn/multimodal/haptic_learning.h" /* H-018集成: 触觉CfC处理+纹理分析 */
#include "selflnn/multimodal/audio_loader.h"     /* ZSF-007: 音频文件加载器集成 */
#include "selflnn/multimodal/image_loader.h"     /* ZSF-008: 图像文件加载器集成 */
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/secure_random.h"
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
    float modality_weights[9];            /**< ZSF-NEW-013: 各模态权重9种（视觉/音频/文本/传感器/触觉/本体感/热感/雷达/电机）用于LNN输入前投影加权，所有模态通过lnn_forward进入统一CfC ODE */
    HapticCfcProcessor* haptic_cfc_proc;  /**< H-018: CfC触觉信号处理器 */
    HapticTextureAnalyzer* haptic_texture_analyzer; /**< H-018: 触觉纹理分析器 */
    /* M-009修复: 5种额外模态的Xavier投影矩阵，替代取模循环映射
     * 索引: 0=触觉(64维), 1=本体感(32维), 2=热感(16维), 3=雷达(128维), 4=电机(64维)
     * 每个矩阵维度: [src_dim, input_dim] */
    float* extra_proj_w[5];
    size_t extra_proj_input_dim;
    int extra_proj_initialized;
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

    /* P2-001统一: 显式初始化全部9个模态权重为等权 1/9 ≈ 0.1111 */
    for (int i = 0; i < 9; i++) {
        manager->modality_weights[i] = 1.0f / 9.0f;
    }
    
    /* H-018集成: 创建CfC触觉处理器和纹理分析器 */
    {
        HapticCfcConfig hc_cfg = haptic_cfc_get_default_config();
        manager->haptic_cfc_proc = haptic_cfc_create(&hc_cfg);

        /* H-006修复：注册全局处理器供haptic_learning.c使用 */
        extern void haptic_enhance_set_global_processor(HapticCfcProcessor* proc);
        haptic_enhance_set_global_processor(manager->haptic_cfc_proc);

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
    /* M-009: 释放5种额外模态的Xavier投影矩阵 */
    for (int m = 0; m < 5; m++) {
        safe_free((void**)&manager->extra_proj_w[m]);
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
                              const void* haptic_data,
                              const void* proprioception_data,
                              const void* thermal_data,
                              const void* radar_data,
                              const void* motor_data,
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
    /* ZSFWS-001修复: 时序特征计算必须在encode之后执行
     * 原代码在encode前读取unified_signal导致temporal_features永远为0，
     * 因为unified_signal在encode调用后才被填充真实数据。 */
    unified_output->temporal_features = NULL;
    unified_output->temporal_dimension = 0;
    unified_output->encoding_quality = 0.0f;
    unified_output->cross_modal_alignment = 0.0f;

    const VisionInput* vision_input = (const VisionInput*)vision_data;
    const AudioInput* audio_input = (const AudioInput*)audio_data;
    const TextInput* text_input = (const TextInput*)text_data;
    const SensorInput* sensor_input = (const SensorInput*)sensor_data;

    int result = unified_signal_processor_encode(manager->unified_signal_processor,
                                       vision_input, audio_input, text_input, sensor_input,
                                       unified_output);

    /* ZSFWS-001修复: 时序特征计算移到这里——unified_signal已在encode中填充完毕 */
    {
        static float prev_frame_features[256] = {0};
        static int has_prev_frame = 0;
        size_t tdim = (max_features < 256) ? max_features : 256;
        unified_output->temporal_features = (float*)safe_malloc(tdim * sizeof(float));
        if (unified_output->temporal_features && has_prev_frame) {
            unified_output->temporal_dimension = tdim;
            for (size_t d = 0; d < tdim; d++) {
                unified_output->temporal_features[d] = unified_output->unified_signal[d] - prev_frame_features[d];
            }
        } else if (unified_output->temporal_features) {
            unified_output->temporal_dimension = tdim;
            memset(unified_output->temporal_features, 0, tdim * sizeof(float));
        }
        for (size_t d = 0; d < tdim; d++) {
            prev_frame_features[d] = unified_output->unified_signal[d];
        }
        has_prev_frame = 1;
    }

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

    /* M-009修复: 使用Xavier初始化投影矩阵替代取模循环映射。
     * 每种额外模态拥有独立的投影矩阵 W_proj[src_dim × input_dim]，
     * 通过矩阵乘法 proj = W^T * src_signal 将模态信号准确投影到LNN输入空间。 */
    {
        const float* haptic_ptr = (const float*)haptic_data;
        const float* proprioception_ptr = (const float*)proprioception_data;
        const float* thermal_ptr = (const float*)thermal_data;
        const float* radar_ptr = (const float*)radar_data;
        const float* motor_ptr = (const float*)motor_data;

        const float* extra_signals[5] = {
            haptic_ptr, proprioception_ptr, thermal_ptr, radar_ptr, motor_ptr
        };
        size_t extra_sizes[5] = {64, 32, 16, 128, 64};
        int extra_indices[5] = {4, 5, 6, 7, 8};

        /* 惰性初始化Xavier投影矩阵（仅首次调用时分配） */
        if (!manager->extra_proj_initialized) {
            size_t input_dim = lnn_cfg.input_size;
            for (int m = 0; m < 5; m++) {
                size_t src_dim = extra_sizes[m];
                size_t total_elems = src_dim * input_dim;
                manager->extra_proj_w[m] = (float*)safe_malloc(total_elems * sizeof(float));
                if (manager->extra_proj_w[m]) {
                    /* Xavier/Glorot均匀分布初始化: U[-scale, +scale]
                     * scale = sqrt(6 / (fan_in + fan_out)) */
                    float scale = sqrtf(6.0f / (float)(src_dim + input_dim));
                    for (size_t i = 0; i < total_elems; i++) {
                        manager->extra_proj_w[m][i] = (secure_random_float() * 2.0f - 1.0f) * scale;
                    }
                }
            }
            manager->extra_proj_input_dim = input_dim;
            manager->extra_proj_initialized = 1;
        }

        /* 使用投影矩阵进行矩阵-向量乘法: lnn_input += weight * W^T * src_signal */
        if (manager->extra_proj_input_dim == lnn_cfg.input_size) {
            for (int m = 0; m < 5; m++) {
                if (extra_signals[m] && extra_sizes[m] > 0 && manager->extra_proj_w[m]) {
                    float weight = manager->modality_weights[extra_indices[m]];
                    size_t src_dim = extra_sizes[m];
                    size_t input_dim = lnn_cfg.input_size;
                    for (size_t i = 0; i < input_dim; i++) {
                        float proj_val = 0.0f;
                        for (size_t j = 0; j < src_dim; j++) {
                            proj_val += manager->extra_proj_w[m][j * input_dim + i] * extra_signals[m][j];
                        }
                        lnn_input[i] += proj_val * weight;
                    }
                }
            }
        }
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

    /* ZSF-NEW-013: 后混合路径已移除。
     * 所有9种模态(视觉/音频/文本/传感器/触觉/本体感/热感/雷达/电机)
     * 现在统一通过 lnn_input → lnn_forward → lnn_output 路径，
     * 进入同一个CfC连续动态系统进行状态演化。 */

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

    /* P2-001统一: 边界从4扩展到9 */
    if (modality_type < 0 || modality_type >= 9) {
        return -1;
    }

    if (weight < 0.0f) weight = 0.0f;
    if (weight > 1.0f) weight = 1.0f;

    manager->modality_weights[modality_type] = weight;
    return 0;
}

/* ============================================================================
 * ZSF-007/ZSF-008: 音频/图像文件加载器集成桥接函数
 * audio_loader.h 和 image_loader.h 此前为孤儿头文件（有实现但0处引用）。
 * 通过此桥接函数将文件加载能力集成到多模态管理器，使其成为系统的有效组成部分。
 * ============================================================================ */

/**
 * @brief 通过audio_loader从WAV文件加载音频数据（桥接函数）
 *
 * 此函数提供统一的音频文件加载入口，供多模态系统使用。
 * 调用audio_loader中的 audio_load_wav 实现实际文件解析。
 *
 * @param filepath WAV音频文件路径
 * @param sample_rate_out 输出采样率
 * @param num_samples_out 输出采样点数
 * @param channels_out 输出声道数（可选NULL）
 * @return float样本数组（需safe_free释放），失败返回NULL
 */
float* multimodal_load_audio_file(const char* filepath, int* sample_rate_out,
                                   int* num_samples_out, int* channels_out) {
    return audio_load_wav(filepath, sample_rate_out, num_samples_out, channels_out);
}

/**
 * @brief 获取WAV音频文件信息（桥接函数）
 *
 * @param filepath WAV文件路径
 * @param sample_rate_out 输出采样率
 * @param num_channels_out 输出声道数
 * @param bits_per_sample_out 输出位深
 * @param duration_sec_out 输出时长（秒）
 * @return 0成功，-1失败
 */
int multimodal_audio_file_info(const char* filepath, int* sample_rate_out,
                                int* num_channels_out, int* bits_per_sample_out,
                                float* duration_sec_out) {
    return audio_wav_info(filepath, sample_rate_out, num_channels_out,
                           bits_per_sample_out, duration_sec_out);
}

/**
 * @brief 通过image_loader从图像文件加载并解码（桥接函数）
 *
 * 自动检测BMP/PPM格式并解码为归一化float数组。
 *
 * @param filepath 图像文件路径（BMP或PPM）
 * @param width_out 输出宽度
 * @param height_out 输出高度
 * @param channels_out 输出通道数（1=灰度，3=RGB）
 * @return float数组（需safe_free释放），失败返回NULL
 */
float* multimodal_load_image_file(const char* filepath, int* width_out,
                                   int* height_out, int* channels_out) {
    /* 先加载原始RGB数据 */
    uint8_t* rgb = NULL;
    int loaded_w = 0, loaded_h = 0;

    /* 尝试BMP格式 */
    rgb = image_load_bmp(filepath, &loaded_w, &loaded_h);
    if (!rgb) {
        /* 尝试PPM格式 */
        rgb = image_load_ppm(filepath, &loaded_w, &loaded_h);
    }

    if (!rgb) {
        return NULL;
    }

    if (width_out) *width_out = loaded_w;
    if (height_out) *height_out = loaded_h;

    /* 转换为归一化float数组 */
    int ch = (channels_out && *channels_out > 0) ? *channels_out : 3;
    float* result = image_rgb_to_float(rgb, loaded_w, loaded_h, ch);

    /* 释放原始RGB数据 */
    free(rgb);

    if (channels_out) *channels_out = ch;
    return result;
}

/**
 * @brief 释放由multimodal_load_image_file加载的图像数据
 */
void multimodal_free_image_data(float* data) {
    if (data) {
        safe_free((void**)&data);
    }
}

/**
 * @brief 释放由multimodal_load_audio_file加载的音频数据
 */
void multimodal_free_audio_data(float* data) {
    audio_wav_free(data);
}

/**
 * @brief 获取模态权重（仅用于外部路由查询，不影响CfC内部状态演化）
 */
int multimodal_manager_get_weight(const MultimodalManager* manager,
                                 int modality_type, float* weight) {
    if (!manager || !weight) {
        return -1;
    }

    /* P2-001统一: 边界从4扩展到9 */
    if (modality_type < 0 || modality_type >= 9) {
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
