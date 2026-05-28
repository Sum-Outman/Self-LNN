#ifndef SELFLNN_MULTIMODAL_INTEGRATION_H
#define SELFLNN_MULTIMODAL_INTEGRATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "selflnn/multimodal/unified_signal_processor.h"
#include "selflnn/multimodal/depth_estimation.h"
#include "selflnn/multimodal/point_cloud.h"
#include "selflnn/multimodal/slam.h"
#include "selflnn/multimodal/stereo_calibration.h"
#include "selflnn/multimodal/speech_recognition.h"
#include "selflnn/multimodal/vad.h"
#include "selflnn/multimodal/audio_semantic.h"
#include "selflnn/multimodal/ocr.h"
#include "selflnn/multimodal/character_segmentation.h"
#include "selflnn/multimodal/text_detection.h"
#include "selflnn/core/lnn.h"

/**
 * @file multimodal_integration.h
 * @brief 多模态功能集成接口
 * 
 * 实现所有新功能到统一液态神经网络架构的集成。
 * 提供高级API，将空间识别、语音识别、OCR等功能
 * 集成到统一液态神经网络处理流程中。
 * 遵循"所有模态 → 统一输入到同一个连续动态系统"的架构原则。
 */

/* 组件类型前向声明 */
typedef struct DepthEstimator DepthEstimator;
typedef struct PointCloudProcessor PointCloudProcessor;
typedef struct SlamSystem SlamSystem;
typedef struct SpeechRecognizer SpeechRecognizer;
typedef struct VadProcessor VadProcessor;
typedef struct AudioSemanticProcessor AudioSemanticProcessor;
typedef struct OcrProcessor OcrProcessor;
typedef struct CharSegmentationProcessor CharSegmentationProcessor;
typedef struct TextDetectionProcessor TextDetectionProcessor;
typedef struct UnifiedSignalProcessor UnifiedSignalProcessor;
typedef struct LaplaceAnalyzer LaplaceAnalyzer;

/**
 * @brief 空间识别特征提取配置
 */
typedef struct {
    int enable_depth_estimation;      /**< 启用深度估计 */
    int enable_point_cloud;           /**< 启用点云处理 */
    int enable_slam;                  /**< 启用SLAM系统 */
    int enable_stereo_vision;         /**< 启用立体视觉 */
    int max_features;                 /**< 最大特征数量 */
} SpatialIntegrationConfig;

/**
 * @brief 语音识别特征提取配置
 */
typedef struct {
    int enable_vad;                   /**< 启用语音端点检测 */
    int enable_speech_recognition;    /**< 启用语音识别 */
    int enable_semantic_understanding;/**< 启用语义理解 */
    int max_features;                 /**< 最大特征数量 */
} SpeechIntegrationConfig;

/**
 * @brief OCR特征提取配置
 */
typedef struct {
    int enable_text_detection;        /**< 启用文本检测 */
    int enable_character_segmentation;/**< 启用字符分割 */
    int enable_ocr;                   /**< 启用OCR识别 */
    int max_features;                 /**< 最大特征数量 */
} OcrIntegrationConfig;

/**
 * @brief 多模态集成配置
 */
typedef struct {
    SpatialIntegrationConfig spatial_config;      /**< 空间识别配置 */
    SpeechIntegrationConfig speech_config;        /**< 语音识别配置 */
    OcrIntegrationConfig ocr_config;               /**< OCR处理配置 */
    int unified_feature_dimension;                /**< 统一特征维度 */
    int enable_unified_signal_processor;                   /**< 启用统一信号处理器 */
} MultimodalIntegrationConfig;

/**
 * @brief 多模态集成处理器
 */
typedef struct MultimodalIntegrationProcessor {
    /* 配置 */
    MultimodalIntegrationConfig config;   /**< 当前配置 */
    
    /* 空间识别组件 */
    DepthEstimator* depth_estimator;      /**< 深度估计器 */
    PointCloudProcessor* point_cloud_processor; /**< 点云处理器 */
    SlamSystem* slam_system;              /**< SLAM系统 */
    
    /* 语音识别组件 */
    SpeechRecognizer* speech_recognizer;  /**< 语音识别器 */
    VadProcessor* vad_processor;          /**< 语音端点检测器 */
    AudioSemanticProcessor* audio_semantic_processor; /**< 音频语义处理器 */
    
    /* OCR组件 */
    OcrProcessor* ocr_processor;          /**< OCR处理器 */
    CharSegmentationProcessor* char_seg_processor; /**< 字符分割处理器 */
    TextDetectionProcessor* text_detection_processor; /**< 文本检测处理器 */
    
    /* 统一信号处理器 */
    UnifiedSignalProcessor* unified_signal_processor;      /**< 统一信号处理器 */
    
    /* 状态 */
    int initialized;                      /**< 是否已初始化 */
} MultimodalIntegrationProcessor;

/**
 * @brief 获取默认的多模态集成配置
 * 
 * @return MultimodalIntegrationConfig 默认配置
 */
MultimodalIntegrationConfig multimodal_integration_get_default_config(void);

/**
 * @brief 创建多模态集成处理器
 * 
 * @param config 配置参数
 * @return MultimodalIntegrationProcessor* 处理器指针，失败返回NULL
 */
MultimodalIntegrationProcessor* multimodal_integration_processor_create(
    const MultimodalIntegrationConfig* config);

/**
 * @brief 处理视觉输入并提取空间识别特征
 * 
 * @param processor 处理器
 * @param image_data 图像数据（行优先，通道交错）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param stereo_image 立体图像数据（可选）
 * @param vision_output 视觉输出结构体（用于填充特征）
 * @return int 成功返回0，失败返回错误码
 */
int multimodal_integration_process_vision(
    MultimodalIntegrationProcessor* processor,
    const float* image_data, int width, int height, int channels,
    const float* stereo_image,
    VisionInput* vision_output);

/**
 * @brief 处理音频输入并提取语音识别特征
 * 
 * @param processor 处理器
 * @param audio_data 音频数据（单声道或立体声）
 * @param num_samples 样本数量
 * @param sample_rate 采样率
 * @param num_channels 通道数
 * @param audio_output 音频输出结构体（用于填充特征）
 * @return int 成功返回0，失败返回错误码
 */
int multimodal_integration_process_audio(
    MultimodalIntegrationProcessor* processor,
    const float* audio_data, int num_samples, int sample_rate, int num_channels,
    AudioInput* audio_output);

/**
 * @brief 处理文本输入并提取OCR特征
 * 
 * @param processor 处理器
 * @param image_data 图像数据（包含文本）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param text_output 文本输出结构体（用于填充特征）
 * @return int 成功返回0，失败返回错误码
 */
int multimodal_integration_process_text(
    MultimodalIntegrationProcessor* processor,
    const float* image_data, int width, int height, int channels,
    TextInput* text_output);

/**
 * @brief 处理多模态输入并生成统一信号输出
 * 
 * @param processor 处理器
 * @param vision_input 视觉输入（可选）
 * @param audio_input 音频输入（可选）
 * @param text_input 文本输入（可选）
 * @param sensor_input 传感器输入（可选）
 * @param unified_output 统一信号输出
 * @return int 成功返回0，失败返回错误码
 */
int multimodal_integration_process_unified(
    MultimodalIntegrationProcessor* processor,
    const VisionInput* vision_input,
    const AudioInput* audio_input,
    const TextInput* text_input,
    const SensorInput* sensor_input,
    UnifiedOutput* unified_output);

/**
 * @brief 处理多模态输入并直接传递给液态神经网络
 * 
 * @param processor 处理器
 * @param lnn 液态神经网络实例
 * @param vision_input 视觉输入（可选）
 * @param audio_input 音频输入（可选）
 * @param text_input 文本输入（可选）
 * @param sensor_input 传感器输入（可选）
 * @param lnn_output LNN输出缓冲区
 * @param max_output_size 最大输出大小
 * @return int 成功返回0，失败返回错误码
 */
int multimodal_integration_process_to_lnn(
    MultimodalIntegrationProcessor* processor,
    LNN* lnn_net,
    const VisionInput* vision_input,
    const AudioInput* audio_input,
    const TextInput* text_input,
    const SensorInput* sensor_input,
    float* lnn_output, size_t max_output_size);

/**
 * @brief 释放多模态集成处理器
 * 
 * @param processor 处理器指针
 */
void multimodal_integration_processor_free(MultimodalIntegrationProcessor* processor);

/**
 * @brief 生成离线测试输入数据（无硬件环境下的AGI运行支持）
 * 
 * 当系统未接入物理传感器/摄像头/麦克风等硬件时，
 * 调用此函数生成数学合成数据以保持AGI系统正常运行。
 * 生成的是具有真实数学结构（分形噪声、正弦叠加等）的数据，非虚假占位符。
 * 
 * @param processor 多模态集成处理器
 * @param vision_output 输出：视觉测试数据（调用者负责释放image_data和depth_map）
 * @param audio_output 输出：音频测试数据（调用者负责释放audio_data）
 * @param text_output 输出：文本测试数据（调用者负责释放token_ids）
 * @param sensor_output 输出：传感器测试数据（调用者负责释放sensor_types和sensor_values）
 * @return int 成功返回0，失败返回-1
 */
int multimodal_integration_generate_test_data(
    MultimodalIntegrationProcessor* processor,
    VisionInput* vision_output,
    AudioInput* audio_output,
    TextInput* text_output,
    SensorInput* sensor_output);

/* ============================================================================
 * MmCfcFusionState: CfC ODE 跨模态连续动态融合
 *
 * 各模态通过线性投影映射到统一隐空间，
 * 拼接后通过 CfC ODE 进行连续时间状态演化。
 * 融合权重可通过梯度反向传播学习更新。
 *
 * CfC ODE 融合公式：
 *   τ * dh/dt = -h + σ(W_fusion * [modal_0; ...; modal_n]) ⊙ tanh(U_fusion * h + b)
 *
 * 支持9种模态:
 *   视觉(0) 音频(1) 文本(2) 传感器(3) 触觉(4)
 *   本体感(5) 热感(6) 雷达(7) 电机(8)
 * ============================================================================ */

#define MM_FUSION_MAX_MODALITIES  9

/**
 * @brief CfC跨模态ODE融合状态
 */
typedef struct {
    int num_modalities;               /**< 模态数量（最多9种） */
    int latent_dim;                   /**< 各模态投影后的统一隐空间维度 */
    int hidden_dim;                   /**< CfC ODE 隐状态维度（即融合输出维度） */
    int concat_dim;                   /**< 拼接后总维度 = num_modalities * latent_dim */

    /* CfC ODE 隐状态 h ∈ R^{hidden_dim} */
    float *h;

    /* 各模态线性投影参数
     * W_proj[m] ∈ R^{latent_dim × input_dims[m]}  (行优先存储)
     * b_proj[m] ∈ R^{latent_dim} */
    float **W_proj;
    float **b_proj;
    int *input_dims;

    /* CfC 融合核心参数
     * W_fusion ∈ R^{hidden_dim × concat_dim}  门控投影矩阵
     * U_fusion ∈ R^{hidden_dim × hidden_dim}  循环矩阵
     * b_fusion ∈ R^{hidden_dim}               偏置 */
    float *W_fusion;
    float *U_fusion;
    float *b_fusion;
    float tau;                        /**< ODE时间常数 */
    float dt;                         /**< ODE积分步长 */
    int ode_steps;                    /**< ODE数值积分步数 */

    /* 学习相关 */
    int is_training;
    float learning_rate;

    /* 梯度累积缓冲区（用于参数更新） */
    float *dW_fusion;                 /**< W_fusion 参数梯度 */
    float *dU_fusion;                 /**< U_fusion 参数梯度 */
    float *db_fusion;                 /**< b_fusion 参数梯度 */
    float **dW_proj;                  /**< 投影权重梯度数组 */
    float **db_proj;                  /**< 投影偏置梯度数组 */

    int is_initialized;
} MmCfcFusionState;

/**
 * @brief 创建并初始化 CfC 跨模态ODE融合状态
 *
 * 使用 Xavier 初始化投影矩阵，He 初始化融合矩阵。
 *
 * @param num_modalities 模态数量
 * @param modality_input_dims 各模态原始输入维度数组（长度=num_modalities）
 * @param latent_dim 投影隐空间维度（默认64）
 * @param hidden_dim CfC ODE 隐状态维度（默认256）
 * @param ode_steps ODE数值积分步数（默认10）
 * @param tau ODE时间常数（默认0.5）
 * @param dt 积分步长（默认0.02）
 * @return MmCfcFusionState* 成功返回状态指针，失败返回NULL
 */
MmCfcFusionState* mm_cfc_unified_fusion_init(
    int num_modalities,
    const int *modality_input_dims,
    int latent_dim,
    int hidden_dim,
    int ode_steps,
    float tau,
    float dt);

/**
 * @brief 释放 CfC 跨模态ODE融合状态
 *
 * @param state 融合状态指针
 */
void mm_cfc_unified_fusion_free(MmCfcFusionState *state);

/**
 * @brief 执行 CfC ODE 跨模态连续动态融合
 *
 * 处理流程:
 *   1. 各模态 → 线性投影 → 映射到统一隐空间维度(latent_dim)
 *   2. 所有投影拼接为一维向量 X ∈ R^{concat_dim}
 *   3. CfC ODE 数值求解: τ*dh/dt = -h + σ(W_fusion·X) ⊙ tanh(U_fusion·h + b)
 *   4. 输出 ODE 演化收敛后的隐状态 h 作为跨模态融合特征
 *
 * @param state 融合状态
 * @param modality_data 各模态原始数据（float*数组，对应index的模态缺失则传NULL）
 * @param modality_dims 各模态实际数据维度（用于校验，必须匹配input_dims）
 * @param num_modalities 有效模态数量（<= state->num_modalities）
 * @param fused_output 融合输出缓冲区（调用者分配，大小>=hidden_dim）
 * @param fused_output_dim 输出缓冲区维度
 * @return int 成功返回0，失败返回负值
 */
int mm_cfc_unified_fusion(
    MmCfcFusionState *state,
    const float **modality_data,
    const int *modality_dims,
    int num_modalities,
    float *fused_output,
    int fused_output_dim);

/**
 * @brief 保存融合参数到二进制文件
 *
 * 依次保存: 元数据(num_modalities/latent_dim/hidden_dim等)
 *          + 所有W_proj[] + b_proj[]
 *          + W_fusion + U_fusion + b_fusion
 *
 * @param state 融合状态
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int mm_fusion_save_weights(MmCfcFusionState *state, const char *filepath);

/**
 * @brief 从二进制文件加载融合参数
 *
 * @param state 融合状态（必须已通过 mm_cfc_unified_fusion_init 初始化）
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int mm_fusion_load_weights(MmCfcFusionState *state, const char *filepath);

/**
 * @brief 单步融合训练（通过前向传播 + 数值梯度估计 + SGD更新参数）
 *
 * 使用中央差分法估计核心融合参数 W_fusion 的梯度，
 * 以最小化融合输出与目标信号之间的 MSE 损失。
 *
 * @param state 融合状态
 * @param modality_data 各模态原始数据
 * @param modality_dims 各模态数据维度
 * @param num_modalities 实际模态数量
 * @param target 目标融合输出（监督信号）
 * @param target_dim 目标维度（必须==hidden_dim）
 * @param loss_out 输出：本轮训练损失值
 * @return int 成功返回0，失败返回负值
 */
int mm_cfc_unified_fusion_train(
    MmCfcFusionState *state,
    const float **modality_data,
    const int *modality_dims,
    int num_modalities,
    const float *target,
    int target_dim,
    float *loss_out);

/**
 * ZSFWS-005修复: 多模态统一融合管道
 *
 * 主路径使用方案C的CfC ODE跨模态融合（真正的统一连续动态系统），
 * 回退到共享LNN统一状态处理器。确保"所有模态→统一连续动态系统"。
 *
 * @param modality_data 各模态数据数组（索引=模态类型）
 * @param modality_dims 各模态维度数组
 * @param num_modalities 有效模态数
 * @param main_cfc 主CFC网络（保留兼容，当前未使用）
 * @param unified_output 融合输出缓冲区
 * @param output_dim 输出维度
 * @return int 成功返回0，失败返回-1
 */
int multimodal_unified_pipeline(const float** modality_data, const int* modality_dims,
                                 int num_modalities, void* main_cfc,
                                 float* unified_output, int output_dim);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_MULTIMODAL_INTEGRATION_H