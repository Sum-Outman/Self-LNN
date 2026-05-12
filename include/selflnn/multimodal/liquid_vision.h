/**
 * @file liquid_vision.h
 * @brief 液态视觉处理组件
 *
 * 基于CfC（Closed-form Continuous-time）液态神经网络的视觉处理组件。
 * 所有组件使用唯一的微分方程驱动：τ dh/dt = -h + σ(gate) ⊙ tanh(activation)
 * 不引入任何Transformer、注意力机制或独立处理器。
 * 所有视觉处理在单一液态神经网络框架内完成。
 */

#ifndef SELFLNN_LIQUID_VISION_H
#define SELFLNN_LIQUID_VISION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============ 液态补丁编码器 ============ */

/**
 * @brief 液态补丁编码器配置
 *
 * 将图像划分为补丁，每个补丁通过CfC ODE连续时间动态编码
 * 为统一状态空间中的表示。补丁之间的顺序关系由连续时间
 * 自然建模，无需位置编码。
 */
typedef struct {
    int patch_size;              /**< 补丁大小（像素），默认16 */
    int stride;                  /**< 补丁滑动步长，默认16（无重叠） */
    int max_patches;             /**< 最大补丁数，默认256 */
    int input_channels;          /**< 输入图像通道数，默认3 */
    int patch_hidden_dim;        /**< 每个补丁的隐藏编码维度，默认64 */
    float time_constant;         /**< CfC时间常数，默认0.1 */
    float delta_t;               /**< ODE时间步长，默认0.05 */
    int use_adaptive_tau;        /**< 是否启用自适应时间常数，默认1 */
    float min_tau;               /**< 最小时间常数，默认0.01 */
    float max_tau;               /**< 最大时间常数，默认1.0 */
    int enable_noise;            /**< 是否启用噪声注入，默认0 */
    float noise_std;             /**< 噪声标准差，默认0.01 */
} LiquidPatchEncoderConfig;

/** @brief 液态补丁编码器句柄 */
typedef struct LiquidPatchEncoder LiquidPatchEncoder;

/**
 * @brief 获取默认液态补丁编码器配置
 */
LiquidPatchEncoderConfig liquid_patch_encoder_get_default_config(void);

/**
 * @brief 创建液态补丁编码器
 */
LiquidPatchEncoder* liquid_patch_encoder_create(const LiquidPatchEncoderConfig* config);

/**
 * @brief 释放液态补丁编码器
 */
void liquid_patch_encoder_free(LiquidPatchEncoder* encoder);

/**
 * @brief 液态补丁编码器前向传播
 *
 * 将图像划分为补丁，每个补丁通过CfC ODE动态编码。
 * 输出直接与液态神经网络统一状态空间兼容。
 *
 * @param encoder 编码器句柄
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param image_data 图像数据 [channels x height x width]
 * @param patch_embeddings 输出补丁嵌入 [num_patches x patch_hidden_dim]
 * @param max_patches 输出缓冲区最大补丁数
 * @return int 成功返回实际补丁数，失败返回-1
 */
int liquid_patch_encoder_forward(LiquidPatchEncoder* encoder,
                                 int width, int height, int channels,
                                 const float* image_data,
                                 float* patch_embeddings, int max_patches);

/**
 * @brief 液态补丁编码器反向传播（训练）
 *
 * 使用分析梯度的CfC反向传播。
 * 计算损失对编码器参数的梯度并应用更新。
 *
 * @param encoder 编码器句柄
 * @param dL_dembeddings 损失对补丁嵌入的梯度 [num_patches x patch_hidden_dim]
 * @param num_patches 补丁数
 * @param dL_dimage 输出损失对图像的梯度 [image_size]
 * @param image_size 图像尺寸（总元素数）
 * @param learning_rate 学习率
 * @return int 成功返回0，失败返回-1
 */
int liquid_patch_encoder_backward(LiquidPatchEncoder* encoder,
                                   const float* dL_dembeddings, int num_patches,
                                   float* dL_dimage, int image_size,
                                   float learning_rate);

/**
 * @brief 重置补丁编码器状态
 */
void liquid_patch_encoder_reset(LiquidPatchEncoder* encoder);

/**
 * @brief 获取补丁数
 */
int liquid_patch_encoder_get_num_patches(const LiquidPatchEncoder* encoder,
                                          int width, int height);

/**
 * @brief 获取补丁编码维度
 */
int liquid_patch_encoder_get_hidden_dim(const LiquidPatchEncoder* encoder);

/* ============ 液态视觉CfC演化器 ============ */

/**
 * @brief 液态视觉CfC演化器配置
 *
 * 在液态神经网络框架内，为视觉处理提供专用通道。
 * 每个通道使用独立的CfC ODE演化，具有视觉特异性的
 * 时间常数和门控机制。
 */
typedef struct {
    int visual_state_dim;        /**< 视觉状态维度，默认128 */
    int num_visual_channels;     /**< 视觉处理通道数，默认4 */
    float base_time_constant;    /**< 基础时间常数，默认0.1 */
    float delta_t;               /**< ODE时间步长，默认0.05 */
    int use_adaptive_tau;        /**< 是否启用自适应时间常数，默认1 */
    float min_tau;               /**< 最小时间常数，默认0.01 */
    float max_tau;               /**< 最大时间常数，默认2.0 */
    int enable_cross_channel;    /**< 是否启用通道间交互（通过隐藏到隐藏连接），默认1 */
    int enable_noise;            /**< 是否启用噪声，默认0 */
    float noise_std;             /**< 噪声标准差，默认0.01 */
} LiquidVisualCfCEvolverConfig;

/** @brief 液态视觉CfC演化器句柄 */
typedef struct LiquidVisualCfCEvolver LiquidVisualCfCEvolver;

/**
 * @brief 获取默认配置
 */
LiquidVisualCfCEvolverConfig liquid_visual_cfc_evolver_get_default_config(void);

/**
 * @brief 创建演化器
 */
LiquidVisualCfCEvolver* liquid_visual_cfc_evolver_create(const LiquidVisualCfCEvolverConfig* config);

/**
 * @brief 释放演化器
 */
void liquid_visual_cfc_evolver_free(LiquidVisualCfCEvolver* evolver);

/**
 * @brief 演化器前向传播
 *
 * 将视觉特征通过CfC连续时间动态演化：
 * τ dh/dt = -h + σ(W_gx·x + W_gh·h + b_g) ⊙ tanh(W_ax·x + W_ah·h + b_a)
 *
 * @param evolver 演化器句柄
 * @param visual_input 视觉输入特征
 * @param input_dim 输入维度
 * @param visual_state 输出演化后的视觉状态
 * @return int 成功返回0，失败返回-1
 */
int liquid_visual_cfc_evolver_forward(LiquidVisualCfCEvolver* evolver,
                                       const float* visual_input, int input_dim,
                                       float* visual_state);

/**
 * @brief 多步演化（处理时间序列视觉输入）
 *
 * @param evolver 演化器句柄
 * @param visual_inputs 时间步输入数组 [num_steps x input_dim]
 * @param input_dim 输入维度
 * @param num_steps 时间步数
 * @param final_state 输出最终视觉状态
 * @return int 成功返回0，失败返回-1
 */
int liquid_visual_cfc_evolver_step(LiquidVisualCfCEvolver* evolver,
                                    const float* visual_inputs, int input_dim,
                                    int num_steps, float* final_state);

/**
 * @brief 重置演化器状态
 */
void liquid_visual_cfc_evolver_reset(LiquidVisualCfCEvolver* evolver);

/**
 * @brief 获取视觉状态维度
 */
int liquid_visual_cfc_evolver_get_state_dim(const LiquidVisualCfCEvolver* evolver);

/* ============ 液态空间特征处理器 ============ */

/**
 * @brief 液态空间特征处理器配置
 *
 * 通过CfC隐藏到隐藏连接（W_gh）建模视觉特征之间的空间关系。
 * 每个特征作为一个液态单元，单元间的交互由液态神经微分方程驱动。
 * 不使用任何注意力机制。
 */
typedef struct {
    int num_features;            /**< 特征数量（补丁数），默认64 */
    int feature_dim;             /**< 每个特征的维度，默认64 */
    int spatial_hidden_dim;      /**< 空间处理隐藏维度，默认128 */
    float time_constant;         /**< 时间常数，默认0.1 */
    float delta_t;               /**< ODE时间步长，默认0.05 */
    int use_adaptive_tau;        /**< 是否启用自适应时间常数，默认1 */
    int num_evolution_steps;     /**< 空间演化步数，默认3 */
    float min_tau;               /**< 最小时间常数，默认0.01 */
    float max_tau;               /**< 最大时间常数，默认1.0 */
} LiquidSpatialProcessorConfig;

/** @brief 液态空间特征处理器句柄 */
typedef struct LiquidSpatialProcessor LiquidSpatialProcessor;

/**
 * @brief 获取默认配置
 */
LiquidSpatialProcessorConfig liquid_spatial_processor_get_default_config(void);

/**
 * @brief 创建空间处理器
 */
LiquidSpatialProcessor* liquid_spatial_processor_create(const LiquidSpatialProcessorConfig* config);

/**
 * @brief 释放空间处理器
 */
void liquid_spatial_processor_free(LiquidSpatialProcessor* processor);

/**
 * @brief 空间处理器前向传播
 *
 * 通过CfC隐藏到隐藏动态演化特征间的空间关系：
 * 每个特征 i：τ dh_i/dt = -h_i + σ(Σ_j W_g_ij · h_j + b_gi) ⊙ tanh(Σ_j W_a_ij · h_j + b_ai)
 * 其中 j 遍历所有特征，W_g_ij 和 W_a_ij 为隐藏到隐藏权重
 *
 * @param processor 处理器句柄
 * @param features 输入特征 [num_features x feature_dim]
 * @param num_features 特征数
 * @param feature_dim 特征维度
 * @param output 输出空间感知特征 [num_features x spatial_hidden_dim]
 * @return int 成功返回0，失败返回-1
 */
int liquid_spatial_processor_forward(LiquidSpatialProcessor* processor,
                                      const float* features, int num_features, int feature_dim,
                                      float* output);

/**
 * @brief 重置空间处理器状态
 */
void liquid_spatial_processor_reset(LiquidSpatialProcessor* processor);

/**
 * @brief 获取输出维度
 */
int liquid_spatial_processor_get_output_dim(const LiquidSpatialProcessor* processor);

/* ============ 液态视觉管理器 ============ */

/**
 * @brief 液态视觉处理类型
 */
typedef enum {
    LIQUID_VISION_PATCH_ENCODER = 0,    /**< 仅使用液态补丁编码器 */
    LIQUID_VISION_CFC_EVOLVER = 1,      /**< 仅使用液态视觉CfC演化器 */
    LIQUID_VISION_SPATIAL = 2,          /**< 仅使用液态空间处理器 */
    LIQUID_VISION_FULL = 3              /**< 全流水线：补丁编码 → 空间处理 → CfC演化 */
} LiquidVisionPipelineType;

/**
 * @brief 液态视觉管理器配置
 */
typedef struct {
    LiquidVisionPipelineType pipeline_type;    /**< 流水线类型 */
    LiquidPatchEncoderConfig patch_config;     /**< 补丁编码器配置 */
    LiquidVisualCfCEvolverConfig evolver_config; /**< CfC演化器配置 */
    LiquidSpatialProcessorConfig spatial_config; /**< 空间处理器配置 */
    int enable_temporal_integration;           /**< 是否启用时序集成（多帧），默认0 */
    int temporal_window_size;                  /**< 时序窗口大小，默认5 */
} LiquidVisionManagerConfig;

/** @brief 液态视觉管理器句柄 */
typedef struct LiquidVisionManager LiquidVisionManager;

/**
 * @brief 获取默认管理器配置
 */
LiquidVisionManagerConfig liquid_vision_manager_get_default_config(void);

/**
 * @brief 创建液态视觉管理器
 *
 * 整合所有液态视觉处理组件为一个统一流水线。
 * 所有处理均在液态神经网络框架内完成。
 *
 * @param config 管理器配置
 * @return LiquidVisionManager* 管理器句柄，失败返回NULL
 */
LiquidVisionManager* liquid_vision_manager_create(const LiquidVisionManagerConfig* config);

/**
 * @brief 释放液态视觉管理器
 */
void liquid_vision_manager_free(LiquidVisionManager* manager);

/**
 * @brief 管理器前向传播
 *
 * 根据pipeline_type执行相应流水线：
 * - PATCH_ENCODER: 图像 → 液态补丁编码 → 输出补丁嵌入
 * - CFC_EVOLVER: 视觉特征 → CfC演化 → 输出演化状态
 * - SPATIAL: 特征集 → 空间处理 → 输出空间感知特征
 * - FULL: 图像 → 补丁编码 → 空间处理 → CfC演化 → 输出
 *
 * @param manager 管理器句柄
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param image_data 图像数据
 * @param output 输出缓冲区
 * @param output_dim 输出缓冲区维度
 * @return int 成功返回输出维度，失败返回-1
 */
int liquid_vision_manager_forward(LiquidVisionManager* manager,
                                   int width, int height, int channels,
                                   const float* image_data,
                                   float* output, int output_dim);

/**
 * @brief 设置视觉特征输入（绕过图像处理直接输入特征）
 *
 * 用于从外部（如vision.c提取的SIFT/ORB/HOG特征）接收视觉特征，
 * 然后通过液态流水线进一步处理。
 *
 * @param manager 管理器句柄
 * @param features 视觉特征向量
 * @param feature_dim 特征维度
 * @return int 成功返回0，失败返回-1
 */
int liquid_vision_manager_set_features(LiquidVisionManager* manager,
                                        const float* features, int feature_dim);

/**
 * @brief 重置所有组件状态
 */
void liquid_vision_manager_reset(LiquidVisionManager* manager);

/**
 * @brief 获取管理器输出维度
 */
int liquid_vision_manager_get_output_dim(const LiquidVisionManager* manager);

/**
 * @brief 保存管理器参数到文件
 */
int liquid_vision_manager_save(const LiquidVisionManager* manager, const char* filepath);

/**
 * @brief 从文件加载管理器参数
 */
int liquid_vision_manager_load(LiquidVisionManager* manager, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_LIQUID_VISION_H */
