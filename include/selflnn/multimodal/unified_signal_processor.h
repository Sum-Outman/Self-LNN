/**
 * @file unified_signal_processor.h
 * @brief 多模态统一信号处理器
 * 
 * 实现"所有模态 → 统一输入到同一个连续动态系统"的核心组件。
 * 将视觉、语音、文本、传感器等不同模态映射到统一的连续信号空间，
 * 为液态神经网络提供标准化的统一输入。
 */

#ifndef SELFLNN_UNIFIED_SIGNAL_PROCESSOR_H
#define SELFLNN_UNIFIED_SIGNAL_PROCESSOR_H

#include "selflnn/core/common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 状态演化类型枚举
 */
typedef enum {
    STATE_EVOLUTION_NONE = 0,           /**< 无状态演化（简单拼接） */
    STATE_EVOLUTION_LINEAR = 1,         /**< 线性状态演化：dx/dt = A*x + B*u */
    STATE_EVOLUTION_NONLINEAR = 2,      /**< 非线性状态演化：CfC细胞单元 */
    STATE_EVOLUTION_HYBRID = 3          /**< 混合演化：线性+非线性 */
} StateEvolutionType;

/**
 * @brief 统一信号处理器配置
 */
typedef struct {
    size_t vision_dimension;      /**< 视觉信号维度 */
    size_t audio_dimension;       /**< 音频信号维度 */
    size_t text_dimension;        /**< 文本信号维度 */
    size_t sensor_dimension;      /**< 传感器信号维度 */
    size_t unified_dimension;     /**< 统一处理后的维度 */
    size_t state_dimension;       /**< 状态空间维度（用于连续动态系统） */
    StateEvolutionType evolution_type; /**< 状态演化类型 */
    float evolution_delta_t;      /**< 状态演化时间步长（秒） */
    float learning_rate;          /**< 信号处理器学习率 */
    int enable_online_learning;   /**< 是否启用在线学习 */
    int enable_cross_modal_fusion; /**< 是否启用跨模态融合（已弃用，为实现无注意力统一处理，此功能应禁用） */
    int enable_state_evolution;   /**< 是否启用连续状态演化 */
    int enable_end_to_end_cfc;    /**< 是否启用端到端CfC模式（原始信号直入同一CfC ODE） */
} UnifiedSignalProcessorConfig;

/**
 * @brief 视觉输入数据结构
 */
typedef struct {
    const float* features;        /**< 视觉特征向量 */
    size_t feature_count;         /**< 特征数量 */
    int width;                    /**< 图像宽度 */
    int height;                   /**< 图像高度 */
    int channels;                 /**< 通道数 */
    float timestamp;              /**< 时间戳 */
} VisionInput;

/**
 * @brief 音频输入数据结构
 */
typedef struct {
    const float* samples;         /**< 音频样本 */
    size_t sample_count;          /**< 样本数量 */
    float sample_rate;            /**< 采样率 */
    float* mfcc_features;         /**< MFCC特征 */
    size_t mfcc_count;            /**< MFCC特征数量 */
    float timestamp;              /**< 时间戳 */
} AudioInput;

/**
 * @brief 文本输入数据结构
 */
typedef struct {
    const char* text;             /**< 文本字符串 */
    size_t text_length;           /**< 文本长度 */
    float* embeddings;            /**< 词嵌入向量 */
    size_t embedding_dim;         /**< 嵌入维度 */
    float timestamp;              /**< 时间戳 */
} TextInput;

/**
 * @brief 传感器输入数据结构
 */
typedef struct {
    const float* sensor_values;   /**< 传感器数值数组 */
    size_t sensor_count;          /**< 传感器数量 */
    const char** sensor_names;    /**< 传感器名称数组 */
    float timestamp;              /**< 时间戳 */
} SensorInput;

/**
 * @brief 统一输出数据结构
 */
typedef struct {
    float* unified_signal;        /**< 统一连续信号 */
    size_t signal_dimension;      /**< 信号维度 */
    float* temporal_features;     /**< 时序特征 */
    size_t temporal_dimension;    /**< 时序特征维度 */
    float encoding_quality;       /**< 信号处理质量指标 */
    float cross_modal_alignment;  /**< 跨模态对齐度 */
} UnifiedOutput;

/**
 * @brief 统一信号处理器句柄
 */
typedef struct UnifiedSignalProcessor UnifiedSignalProcessor;

/**
 * @brief 获取默认的统一信号处理器配置
 * 
 * @return UnifiedSignalProcessorConfig 默认配置
 */
SELFLNN_API UnifiedSignalProcessorConfig unified_signal_processor_get_default_config(void);

/**
 * @brief 创建统一信号处理器实例
 * 
 * @param config 信号处理器配置
 * @return UnifiedSignalProcessor* 处理器句柄，失败返回NULL
 */
SELFLNN_API UnifiedSignalProcessor* unified_signal_processor_create(const UnifiedSignalProcessorConfig* config);

/**
 * @brief 释放统一信号处理器实例
 * 
 * @param processor 处理器句柄
 */
SELFLNN_API void unified_signal_processor_free(UnifiedSignalProcessor* processor);

/**
 * @brief 处理多模态输入到统一连续信号
 * 
 * @param processor 处理器句柄
 * @param vision 视觉输入（可为NULL）
 * @param audio 音频输入（可为NULL）
 * @param text 文本输入（可为NULL）
 * @param sensor 传感器输入（可为NULL）
 * @param output 统一输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_encode(UnifiedSignalProcessor* processor,
                          const VisionInput* vision,
                          const AudioInput* audio,
                          const TextInput* text,
                          const SensorInput* sensor,
                          UnifiedOutput* output);

/**
 * @brief 获取处理器配置
 * 
 * @param processor 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_get_config(const UnifiedSignalProcessor* processor, UnifiedSignalProcessorConfig* config);

/**
 * @brief 更新处理器配置
 * 
 * @param processor 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_set_config(UnifiedSignalProcessor* processor, const UnifiedSignalProcessorConfig* config);

/**
 * @brief 训练统一信号处理器
 * 
 * @param processor 处理器句柄
 * @param vision 视觉输入训练样本
 * @param audio 音频输入训练样本
 * @param text 文本输入训练样本
 * @param sensor 传感器输入训练样本
 * @param target 目标统一信号
 * @param loss 损失值输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_train(UnifiedSignalProcessor* processor,
                         const VisionInput* vision,
                         const AudioInput* audio,
                         const TextInput* text,
                         const SensorInput* sensor,
                         const float* target,
                         float* loss);

/**
 * @brief 保存处理器到文件
 * 
 * @param processor 处理器句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_save(const UnifiedSignalProcessor* processor, const char* filepath);

/**
 * @brief 从文件加载处理器
 * 
 * @param filepath 文件路径
 * @return UnifiedSignalProcessor* 处理器句柄，失败返回NULL
 */
UnifiedSignalProcessor* unified_signal_processor_load(const char* filepath);

/**
 * @brief 重置处理器状态
 * 
 * @param processor 处理器句柄
 */
void unified_signal_processor_reset(UnifiedSignalProcessor* processor);

/**
 * @brief 获取信号处理质量指标
 * 
 * @param processor 处理器句柄
 * @param quality 质量指标输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_get_quality(const UnifiedSignalProcessor* processor, float* quality);

/**
 * @brief 将多模态输入处理并传递给液态神经网络
 * 
 * 此函数实现了"所有模态 → 统一输入到同一个连续动态系统 → 统一状态演化 → 统一输出决策"的核心流程。
 * 
 * @param processor 统一信号处理器句柄
 * @param lnn 液态神经网络句柄
 * @param vision 视觉输入（可为NULL）
 * @param audio 音频输入（可为NULL）
 * @param text 文本输入（可为NULL）
 * @param sensor 传感器输入（可为NULL）
 * @param lnn_output LNN输出缓冲区
 * @param max_output_size 输出缓冲区最大容量
 * @return int 成功返回输出向量大小，失败返回-1
 */
int unified_signal_processor_encode_to_lnn(UnifiedSignalProcessor* processor,
                                  void* lnn,
                                  const VisionInput* vision,
                                  const AudioInput* audio,
                                  const TextInput* text,
                                  const SensorInput* sensor,
                                  float* lnn_output,
                                  size_t max_output_size);

/* ============================
 * 增强T8：CfC流水线深度跟踪API
 * ============================ */

/**
 * @brief 流水线阶段枚举
 */
typedef enum {
    PIPELINE_STAGE_INPUT = 0,          /**< 输入阶段 */
    PIPELINE_STAGE_ENCODE = 1,         /**< 信号处理阶段 */
    PIPELINE_STAGE_EVOLVE = 2,         /**< 状态演化阶段 */
    PIPELINE_STAGE_OUTPUT = 3,         /**< 输出阶段 */
    PIPELINE_STAGE_COUNT = 4           /**< 阶段总数 */
} PipelineStage;

/**
 * @brief 流水线阶段统计信息
 */
typedef struct {
    PipelineStage stage;               /**< 阶段标识 */
    double avg_latency_ms;             /**< 平均延迟（毫秒） */
    double max_latency_ms;             /**< 最大延迟（毫秒） */
    double min_latency_ms;             /**< 最小延迟（毫秒） */
    size_t total_calls;                /**< 总调用次数 */
    size_t error_count;                /**< 错误次数 */
    double last_latency_ms;            /**< 最近一次延迟 */
} PipelineStageStats;

/**
 * @brief 全模态流水线性能概要
 */
typedef struct {
    PipelineStageStats stages[PIPELINE_STAGE_COUNT]; /**< 各阶段统计 */
    double total_avg_latency_ms;       /**< 总平均延迟 */
    double throughput_per_second;      /**< 每秒吞吐量 */
    size_t total_processed;            /**< 总处理样本数 */
    double pipeline_efficiency;        /**< 流水线效率 (0~1) */
} PipelinePerformanceSummary;

/**
 * @brief 获取指定阶段的流水线统计信息
 * 
 * @param processor 处理器句柄
 * @param stage 流水线阶段
 * @param stats 输出阶段统计结构体
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_get_pipeline_stats(UnifiedSignalProcessor* processor,
                                        PipelineStage stage,
                                        PipelineStageStats* stats);

/**
 * @brief 获取流水线性能概要
 * 
 * 汇总所有阶段的统计信息，生成流水线性能快照。
 *
 * @param processor 处理器句柄
 * @param summary 输出性能概要结构体
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_get_pipeline_summary(UnifiedSignalProcessor* processor,
                                          PipelinePerformanceSummary* summary);

/**
 * @brief 重置流水线统计信息
 * 
 * @param processor 处理器句柄
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_reset_pipeline_stats(UnifiedSignalProcessor* processor);

/* ============================
 * CfC流水线深度跟踪API
 * ============================ */

/**
 * @brief CfC流水线跟踪数据结构
 *
 * 跟踪CfC细胞在流水线中的状态演化全过程。
 * 包括隐藏态、门控激活、梯度流和收敛指标。
 */
typedef struct {
    double hidden_state_norm;            /**< 隐藏态L2范数 */
    double hidden_state_change;          /**< 隐藏态变化幅度 */
    double gate_mean_activation;         /**< 门控平均激活值 */
    double gate_max_activation;          /**< 门控最大激活值 */
    double gradient_norm;                /**< 梯度L2范数 */
    double gradient_variance;            /**< 梯度方差 */
    double state_convergence_rate;       /**< 状态收敛速率 */
    double lyapunov_exponent;            /**< 李雅普诺夫指数（混沌指示器） */
    int cfp_pipeline_depth;              /**< CfC流水线深度 */
    int is_converged;                    /**< 状态是否已收敛 */
} CfCPipelineTrackingData;

/**
 * @brief 获取CfC流水线跟踪数据
 *
 * 获取CfC细胞在信号处理流水线中的状态跟踪数据，
 * 包括隐藏态范数、门控激活统计、梯度流指标和收敛分析。
 *
 * @param processor 处理器句柄
 * @param data CfC跟踪数据输出缓冲区
 * @return int 成功返回0，失败返回-1（如CfC未启用）
 */
int unified_signal_processor_get_cfc_pipeline_tracking(UnifiedSignalProcessor* processor,
                                               CfCPipelineTrackingData* data);

/**
 * @brief 获取处理器中的CfC状态演化收敛指标
 *
 * 分析CfC细胞状态是否已收敛到稳定点。
 * 返回收敛指示和当前李雅普诺夫指数近似值。
 *
 * @param processor 处理器句柄
 * @param convergence_rate 收敛速率输出（0~1，1=完全收敛）
 * @param lyapunov_est 李雅普诺夫指数估计值输出（负值=稳定，正值=混沌）
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_get_cfc_convergence(UnifiedSignalProcessor* processor,
                                         double* convergence_rate,
                                         double* lyapunov_est);

/**
 * @brief 获取处理器的梯度流监控指标
 *
 * 计算处理器梯度缓冲区中的梯度统计信息，
 * 用于检测梯度消失/爆炸。
 *
 * @param processor 处理器句柄
 * @param gradient_norm 梯度L2范数输出
 * @param gradient_variance 梯度方差输出
 * @param gradient_max 梯度最大值输出
 * @return int 成功返回0，失败返回-1
 */
int unified_signal_processor_get_gradient_flow(UnifiedSignalProcessor* processor,
                                       double* gradient_norm,
                                       double* gradient_variance,
                                       double* gradient_max);

/* ================================================================
 * 需求20.3: 多模态数据混合策略
 * 课程学习 + 动态比例 + 模态丢弃
 * ================================================================ */

/**
 * @brief 课程学习配置
 *
 * 控制训练过程中的样本难度渐进式增长。
 * 起始阶段只使用高信噪比的模态样本，随训练进度逐步引入低质量样本。
 */
typedef struct {
    int enable_curriculum;              /**< 是否启用课程学习 */
    int curriculum_stages;              /**< 课程阶段总数(>=1)，每阶段引入更多难度 */
    float difficulty_threshold_start;   /**< 起始难度阈值(0~1)，只训练低于此阈值的样本 */
    float difficulty_threshold_end;     /**< 最终难度阈值(0~1)，训练至此全部样本 */
    float min_modalities_per_stage;     /**< 每阶段最少启用模态数(>=1) */
} CurriculumConfig;

/**
 * @brief 动态比例配置
 *
 * 根据各模态质量指标动态调整数据采样比例。
 * 质量越高的模态获得越多的采样权重。
 */
typedef struct {
    int enable_dynamic_ratio;           /**< 是否启用动态比例 */
    float ratio_update_rate;            /**< 比例更新速率(0~1)，越高变化越快 */
    float min_ratio;                    /**< 最小采样比例(>0)，防止模态饿死 */
    float max_ratio;                    /**< 最大采样比例(<=1)，防止模态占主导 */
    int update_interval;                /**< 更新间隔(迭代次数) */
} DynamicRatioConfig;

/**
 * @brief 模态丢弃配置
 *
 * 训练时随机丢弃部分模态，增强模型鲁棒性。
 * 丢弃率随训练进度递减（退火）。
 */
typedef struct {
    int enable_modality_dropout;        /**< 是否启用模态丢弃 */
    float dropout_rate[4];              /**< 各模态丢弃率[视觉,音频,文本,传感器](0~1) */
    float drop_schedule_decay;          /**< 丢弃率衰减因子(每迭代) */
    float min_dropout_rate;             /**< 最小丢弃率(退火下限) */
    int ensure_min_modalities;          /**< 至少保留的模态数(>=1) */
} ModalityDropoutConfig;

/**
 * @brief 数据混合策略总配置
 */
typedef struct {
    CurriculumConfig curriculum;        /**< 课程学习配置 */
    DynamicRatioConfig dynamic_ratio;   /**< 动态比例配置 */
    ModalityDropoutConfig modality_dropout; /**< 模态丢弃配置 */
    int verbose;                        /**< 是否输出调试信息 */
} DataMixingConfig;

/**
 * @brief 获取默认数据混合策略配置
 *
 * @return DataMixingConfig 默认配置
 */
SELFLNN_API DataMixingConfig unified_signal_processor_get_default_mixing_config(void);

/**
 * @brief 使用数据混合策略训练统一信号处理器
 *
 * 综合应用课程学习、动态比例和模态丢弃三种策略，
 * 对一批多模态样本进行增强训练。
 *
 * @param processor 处理器句柄
 * @param vision 视觉训练样本数组
 * @param vision_count 视觉样本数
 * @param audio 音频训练样本数组
 * @param audio_count 音频样本数
 * @param text 文本训练样本数组
 * @param text_count 文本样本数
 * @param sensor 传感器训练样本数组
 * @param sensor_count 传感器样本数
 * @param targets 目标信号数组
 * @param target_count 目标信号数(须等于各模态样本数之和)
 * @param mixing_config 数据混合策略配置
 * @param modality_quality 各模态质量指标数组(4个浮点数)
 * @param current_iteration 当前迭代次数
 * @param total_iterations 总迭代次数
 * @param losses 逐样本损失输出缓冲区(可选，可为NULL)
 * @param loss_count losses缓冲区大小
 * @return int 成功返回训练的样本数，失败返回-1
 */
SELFLNN_API int unified_signal_processor_train_batch_mixed(
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
 * @brief 获取课程学习的当前难度阈值
 *
 * @param current_iteration 当前迭代次数
 * @param total_iterations 总迭代次数
 * @param config 课程学习配置
 * @return float 当前难度阈值(0~1)
 */
SELFLNN_API float unified_signal_processor_get_curriculum_threshold(
    size_t current_iteration,
    size_t total_iterations,
    const CurriculumConfig* config);

/**
 * @brief 计算动态采样比例
 *
 * 根据模态质量指标计算各模态的采样比例。
 *
 * @param modality_quality 各模态质量指标数组(4个浮点数)
 * @param quality_count quality数组大小(应为4)
 * @param config 动态比例配置
 * @param ratios 输出比例数组(4个浮点数)
 * @param ratio_count ratios数组大小(应为4)
 * @return int 成功返回0，失败返回-1
 */
SELFLNN_API int unified_signal_processor_compute_dynamic_ratios(
    const float* modality_quality, size_t quality_count,
    const DynamicRatioConfig* config,
    float* ratios, size_t ratio_count);

/**
 * @brief 应用模态丢弃
 *
 * 根据配置随机丢弃部分模态，返回各模态是否活跃。
 *
 * @param processor 处理器句柄
 * @param config 模态丢弃配置
 * @param current_iteration 当前迭代次数
 * @param modality_active 输出活跃标记数组(4个int,1=活跃)
 * @param active_count active数组大小(应为4)
 * @return int 成功返回活跃模态数，失败返回-1
 */
SELFLNN_API int unified_signal_processor_apply_modality_dropout(
    UnifiedSignalProcessor* processor,
    const ModalityDropoutConfig* config,
    size_t current_iteration,
    int* modality_active, size_t active_count);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_UNIFIED_SIGNAL_PROCESSOR_H
