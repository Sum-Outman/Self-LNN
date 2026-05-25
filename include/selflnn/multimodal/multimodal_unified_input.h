#ifndef SELFLNN_MULTIMODAL_UNIFIED_INPUT_H
#define SELFLNN_MULTIMODAL_UNIFIED_INPUT_H

/**
 * @file multimodal_unified_input.h
 * @brief 多模态统一输入处理接口（兼容适配层）
 *
 * **已弃用**: 新代码应使用 unified_lnn_state.h 作为唯一多模态统一入口。
 * 本文件保留用于向后兼容。
 * 严格遵循：所有模态直接拼接输入到同一个LNN连续动态系统。
 */

#include <stddef.h>
#include "selflnn/multimodal/multimodal.h"
#include "selflnn/core/lnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SELFLNN_MAX_CONTROL_DIM 64
#define SELFLNN_MAX_MODALITIES 9
#define SELFLNN_UNIFIED_INPUT_DIM (SELFLNN_MAX_MODALITIES * SELFLNN_MAX_CONTROL_DIM)
/* ZSF-ZNB修复S-002: 统一投影维度（所有模态投影到此维度后求和） */
#define SELFLNN_UNIFIED_PROJECTION_DIM 256

/**
 * @brief 统一输入方法枚举
 */
typedef enum {
    UNIFIED_INPUT_DYNAMIC = 6  /**< 统一动态系统输入（通过单个CfC细胞） */
} UnifiedInputMethod;

/**
 * @brief 模态控制信号质量指标
 */
typedef struct {
    float uncertainty;                  /**< 不确定性（0=完全确定，1=完全不确定） */
    float confidence;                   /**< 置信度（0=无信心，1=完全信心） */
    float signal_strength;              /**< 信号强度 */
    int initialized;                    /**< 是否已初始化 */
    size_t dimension;                   /**< 控制信号维度 */
    size_t offset;                      /**< 在统一输入缓冲区中的偏移量 */
    size_t active_size;                 /**< 当前活跃信号的实际大小 */
} ModalityControlMetrics;

/**
 * @brief 统一输入配置
 */
typedef struct {
    UnifiedInputMethod method;          /**< 统一输入方法（默认使用统一动态系统） */
    size_t unified_input_size;          /**< 统一输入大小 */
    size_t unified_output_size;         /**< 统一输出大小 */
} UnifiedInputConfig;

/**
 * @brief 统一输入状态（不透明类型）
 */
typedef struct {
    UnifiedInputConfig config;                      /**< 统一输入配置 */
    ModalityControlMetrics modality_metrics[SELFLNN_MAX_MODALITIES]; /**< 各模态质量指标 */
    float input_weights[SELFLNN_MAX_MODALITIES];    /**< 输入权重 */
    int is_initialized;                             /**< 初始化标志 */
    int total_process_count;                        /**< 总处理次数 */
    float historical_process_quality;               /**< 历史处理质量 */
    float* unified_buffer;                          /**< 统一缓冲区 */
    size_t unified_buffer_size;                     /**< 统一缓冲区大小 */
    LNN* lnn_instance;                              /**< 关联的LNN实例（不拥有，不释放） */
    float* unified_input_buffer;                    /**< 统一输入缓冲区 */
    size_t unified_input_buffer_size;               /**< 统一输入缓冲区大小 */
    size_t total_active_size;                       /**< 当前活跃信号的总维度 */
    /* M-003修复: 保存上一次输出用于时序对比 */
    float prev_output[SELFLNN_MAX_CONTROL_DIM];     /**< 上一次统一输出 */
    size_t prev_output_dim;                         /**< 上一次输出维度 */
    /* ZSF-ZNB修复S-002: 每模态线性投影矩阵W_i + 偏置b_i
     * W_i[modality] 将各模态原始维度映射到 SELFLNN_UNIFIED_PROJECTION_DIM
     * 统一输入 = sum_i (W_i · x_i + b_i) -- element-wise求和 */
    float* projection_matrices[SELFLNN_MAX_MODALITIES]; /**< 投影矩阵 [proj_dim × input_dim_i] */
    float* projection_biases[SELFLNN_MAX_MODALITIES];   /**< 投影偏置 [proj_dim] */
    size_t projection_input_sizes[SELFLNN_MAX_MODALITIES]; /**< 各投影输入维度 */
    int projections_initialized;                         /**< 投影矩阵是否已初始化 */
} UnifiedInputState;

/**
 * @brief 初始化统一输入状态
 * @param state 统一输入状态指针
 * @param config 统一输入配置（为NULL则使用默认配置）
 * @return 0=成功，非0=失败
 */
int multimodal_unified_input_init(UnifiedInputState* state, const UnifiedInputConfig* config);

/**
 * @brief 处理多模态控制信号为统一输入输出
 * @param state 统一输入状态指针
 * @param vision_control 视觉控制信号
 * @param vision_size 视觉控制维度
 * @param audio_control 音频控制信号
 * @param audio_size 音频控制维度
 * @param text_control 文本控制信号
 * @param text_size 文本控制维度
 * @param sensor_control 传感器控制信号
 * @param sensor_size 传感器控制维度
 * @param tactile_control 触觉控制信号
 * @param tactile_size 触觉控制维度
 * @param proprioception_control 本体感觉控制信号
 * @param proprioception_size 本体感觉控制维度
 * @param thermal_control 热感控制信号
 * @param thermal_size 热感控制维度
 * @param radar_control 雷达控制信号
 * @param radar_size 雷达控制维度
 * @param unified_output [输出] 统一输出控制信号
 * @param unified_size [输出] 统一输出信号维度
 * @param max_output_size 输出缓冲区最大容量
 * @return 成功返回输出特征数，失败返回-1
 */
int multimodal_unified_input_process(UnifiedInputState* state,
                                     const float* vision_control, size_t vision_size,
                                     const float* audio_control, size_t audio_size,
                                     const float* text_control, size_t text_size,
                                     const float* sensor_control, size_t sensor_size,
                                     const float* tactile_control, size_t tactile_size,
                                     const float* proprioception_control, size_t proprioception_size,
                                     const float* thermal_control, size_t thermal_size,
                                     const float* radar_control, size_t radar_size,
                                     const float* motor_control, size_t motor_size,
                                     float* unified_output, size_t* unified_size,
                                     size_t max_output_size);

/**
 * @brief 更新统一输入配置
 * @param state 统一输入状态指针
 * @param config 新配置
 * @return 0=成功，非0=失败
 */
int multimodal_unified_input_set_config(UnifiedInputState* state,
                                        const UnifiedInputConfig* config);

/**
 * @brief 获取当前统一输入配置
 * @param state 统一输入状态指针
 * @param config [输出] 配置缓冲区
 * @return 0=成功，非0=失败
 */
int multimodal_unified_input_get_config(const UnifiedInputState* state,
                                        UnifiedInputConfig* config);

/**
 * @brief 获取各模态当前输入权重
 * @param state 统一输入状态指针
 * @param weights [输出] 权重数组
 * @param count [输出] 权重数量
 * @return 0=成功，非0=失败
 */
int multimodal_unified_input_get_weights(const UnifiedInputState* state,
                                         float* weights, size_t* count);

/**
 * @brief 重置统一输入状态
 * @param state 统一输入状态指针
 * @return 0=成功，非0=失败
 */
int multimodal_unified_input_reset(UnifiedInputState* state);

/**
 * @brief 获取统一输入处理统计信息
 * @param state 统一输入状态指针
 * @param active_method [输出] 当前激活的处理方法
 * @param total_count [输出] 总处理次数
 * @param process_quality [输出] 处理质量评分
 * @return 0=成功，非0=失败
 */
int multimodal_unified_input_get_stats(const UnifiedInputState* state,
                                       UnifiedInputMethod* active_method,
                                       int* total_count,
                                       float* process_quality);

/**
 * @brief 设置关联的LNN实例
 * @param state 统一输入状态指针
 * @param lnn LNN实例指针
 * @return 0=成功，非0=失败
 */
int multimodal_unified_input_set_lnn(UnifiedInputState* state, LNN* lnn);

/**
 * @brief K-009: 9模态统一注入状态诊断
 *
 * 返回哪些模态通道当前有数据流入统一LNN系统。
 * 用于验证"视觉+语音+传感器+文本+控制信号+触觉+本体感知+热感+雷达+电机"
 * 所有9种模态是否都在正确注入到单一CfC连续动态系统。
 *
 * @param state 统一输入状态指针
 * @param modality_active [输出] 9个int数组，1=该模态有数据流，0=无数据流
 * @param total_modalities [输出] 总模态数(应为9)
 * @param active_count [输出] 当前活跃模态数
 * @return 0=成功，非0=失败
 */
int multimodal_unified_input_diagnose_modalities(const UnifiedInputState* state,
                                                  int* modality_active,
                                                  int* total_modalities,
                                                  int* active_count);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_MULTIMODAL_UNIFIED_INPUT_H */
