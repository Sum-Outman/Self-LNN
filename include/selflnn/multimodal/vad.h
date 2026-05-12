/**
 * @file vad.h
 * @brief 语音端点检测（VAD）接口
 * 
 * 语音活动检测模块，用于检测音频信号中的语音和非语音段。
 * 支持多种VAD算法和配置参数。
 */

#ifndef SELFLNN_VAD_H
#define SELFLNN_VAD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief VAD算法类型枚举
 */
typedef enum {
    VAD_ENERGY_BASED = 0,               /**< 基于能量的VAD */
    VAD_ZERO_CROSSING = 1,              /**< 基于过零率的VAD */
    VAD_SPECTRAL_ENTROPY = 2,           /**< 基于频谱熵的VAD */
    VAD_MACHINE_LEARNING = 3,           /**< 基于机器学习的VAD */
    VAD_HYBRID = 4,                     /**< 混合方法VAD */
    VAD_WEBRTC = 5,                     /**< WebRTC VAD算法 */
} VadAlgorithm;

/**
 * @brief VAD模式枚举
 */
typedef enum {
    VAD_MODE_QUALITY = 0,               /**< 高质量模式（最严格） */
    VAD_MODE_LOW_BITRATE = 1,           /**< 低比特率模式 */
    VAD_MODE_AGGRESSIVE = 2,            /**< 激进模式（最宽松） */
    VAD_MODE_VERY_AGGRESSIVE = 3,       /**< 非常激进模式 */
} VadMode;

/**
 * @brief VAD决策结果枚举
 */
typedef enum {
    VAD_DECISION_SILENCE = 0,           /**< 静音/非语音 */
    VAD_DECISION_SPEECH = 1,            /**< 语音 */
    VAD_DECISION_UNKNOWN = 2,           /**< 未知/不确定 */
} VadDecision;

/**
 * @brief VAD配置结构体
 */
typedef struct {
    VadAlgorithm algorithm;             /**< VAD算法类型 */
    VadMode mode;                       /**< VAD模式 */
    int sample_rate;                    /**< 采样率（Hz） */
    int frame_duration_ms;              /**< 帧持续时间（毫秒） */
    int frame_samples;                  /**< 每帧样本数 */
    float energy_threshold;             /**< 能量阈值（用于能量VAD） */
    float zero_crossing_threshold;      /**< 过零率阈值 */
    float spectral_entropy_threshold;   /**< 频谱熵阈值 */
    float hysteresis_threshold;         /**< 滞后阈值（防止抖动） */
    int min_speech_duration_ms;         /**< 最小语音持续时间（毫秒） */
    int min_silence_duration_ms;        /**< 最小静音持续时间（毫秒） */
    int speech_pad_ms;                  /**< 语音填充时间（毫秒） */
    int aggressiveness;                 /**< 激进度（0-3） */
    int enable_hangover;                /**< 是否启用拖尾效应处理 */
    int hangover_frames;                /**< 拖尾效应帧数 */
    int use_noise_suppression;          /**< 是否使用噪声抑制 */
    int use_auto_threshold;             /**< 是否自动调整阈值 */
    int vad_gate_frames;                /**< VAD门限帧数 */
} VadConfig;

/**
 * @brief VAD帧结果结构体
 */
typedef struct {
    VadDecision decision;               /**< VAD决策 */
    float confidence;                   /**< 置信度（0-1） */
    float energy;                       /**< 帧能量 */
    float zero_crossing_rate;           /**< 过零率 */
    float spectral_entropy;             /**< 频谱熵 */
    int frame_index;                    /**< 帧索引 */
    int sample_start;                   /**< 起始样本索引 */
    int sample_end;                     /**< 结束样本索引 */
} VadFrameResult;

/**
 * @brief VAD段结果结构体
 */
typedef struct {
    VadDecision* decisions;             /**< 每帧决策数组 */
    float* confidences;                 /**< 每帧置信度数组 */
    int num_frames;                     /**< 总帧数 */
    int* speech_segments;               /**< 语音段边界（起始帧，结束帧） */
    int num_speech_segments;            /**< 语音段数量 */
    int* silence_segments;              /**< 静音段边界 */
    int num_silence_segments;           /**< 静音段数量 */
    float speech_ratio;                 /**< 语音比率（0-1） */
    float processing_time_ms;           /**< 处理时间（毫秒） */
} VadResult;

/**
 * @brief 音频段结构体
 */
typedef struct {
    float* audio_data;                  /**< 音频数据 */
    int num_samples;                    /**< 样本数 */
    int sample_rate;                    /**< 采样率 */
    int is_speech;                      /**< 是否为语音段 */
    float confidence;                   /**< 语音置信度 */
    int segment_index;                  /**< 段索引 */
    int start_sample;                   /**< 起始样本（在原始音频中） */
    int end_sample;                     /**< 结束样本（在原始音频中） */
} AudioSegment;

/**
 * @brief 音频分段结果结构体
 */
typedef struct {
    AudioSegment* segments;             /**< 音频段数组 */
    int num_segments;                   /**< 段数量 */
    int num_speech_segments;            /**< 语音段数量 */
    int num_silence_segments;           /**< 静音段数量 */
    float* segment_confidences;         /**< 段置信度数组 */
    int* segment_boundaries;            /**< 段边界（样本索引） */
    float total_duration_ms;            /**< 总持续时间（毫秒） */
    float speech_duration_ms;           /**< 语音持续时间（毫秒） */
    float silence_duration_ms;          /**< 静音持续时间（毫秒） */
} AudioSegmentationResult;

/**
 * @brief VAD处理器句柄
 */
typedef struct VadProcessor VadProcessor;

/**
 * @brief 获取默认的VAD配置
 * 
 * @return VadConfig 默认配置
 */
VadConfig vad_get_default_config(void);

/**
 * @brief 创建VAD处理器
 * 
 * @param config VAD配置
 * @return VadProcessor* 处理器句柄，失败返回NULL
 */
VadProcessor* vad_processor_create(const VadConfig* config);

/**
 * @brief 释放VAD处理器
 * 
 * @param processor 处理器句柄
 */
void vad_processor_free(VadProcessor* processor);

/**
 * @brief 执行VAD处理
 * 
 * @param processor 处理器句柄
 * @param audio_data 音频数据
 * @param num_samples 样本数
 * @param result VAD结果输出
 * @return int 成功返回0，失败返回-1
 */
int vad_process(VadProcessor* processor,
               const float* audio_data, int num_samples,
               VadResult* result);

/**
 * @brief 执行流式VAD处理
 * 
 * @param processor 处理器句柄
 * @param audio_frame 音频帧数据
 * @param frame_samples 帧样本数
 * @param frame_result 帧结果输出
 * @return int 成功返回0，失败返回-1
 */
int vad_process_frame(VadProcessor* processor,
                     const float* audio_frame, int frame_samples,
                     VadFrameResult* frame_result);

/**
 * @brief 重置VAD处理器
 * 
 * @param processor 处理器句柄
 */
void vad_processor_reset(VadProcessor* processor);

/**
 * @brief 获取VAD配置
 * 
 * @param processor 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int vad_processor_get_config(const VadProcessor* processor, VadConfig* config);

/**
 * @brief 设置VAD配置
 * 
 * @param processor 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int vad_processor_set_config(VadProcessor* processor, const VadConfig* config);

/**
 * @brief 自适应调整VAD阈值
 * 
 * @param processor 处理器句柄
 * @param audio_data 音频数据（用于训练）
 * @param num_samples 样本数
 * @param speech_reference 语音参考标签（可选）
 * @return int 成功返回0，失败返回-1
 */
int vad_adaptive_threshold(VadProcessor* processor,
                          const float* audio_data, int num_samples,
                          const int* speech_reference);

/**
 * @brief 分割音频为语音和非语音段
 * 
 * @param processor 处理器句柄
 * @param audio_data 音频数据
 * @param num_samples 样本数
 * @param result 分段结果输出
 * @return int 成功返回0，失败返回-1
 */
int vad_segment_audio(VadProcessor* processor,
                     const float* audio_data, int num_samples,
                     AudioSegmentationResult* result);

/**
 * @brief 从VAD结果提取语音段
 * 
 * @param processor 处理器句柄
 * @param audio_data 原始音频数据
 * @param num_samples 样本数
 * @param vad_result VAD结果
 * @param speech_segments 语音段输出数组
 * @param max_segments 最大段数
 * @return int 成功返回语音段数，失败返回-1
 */
int vad_extract_speech_segments(VadProcessor* processor,
                               const float* audio_data, int num_samples,
                               const VadResult* vad_result,
                               AudioSegment* speech_segments, int max_segments);

/**
 * @brief 从VAD结果提取非语音段
 * 
 * @param processor 处理器句柄
 * @param audio_data 原始音频数据
 * @param num_samples 样本数
 * @param vad_result VAD结果
 * @param silence_segments 非语音段输出数组
 * @param max_segments 最大段数
 * @return int 成功返回非语音段数，失败返回-1
 */
int vad_extract_silence_segments(VadProcessor* processor,
                                const float* audio_data, int num_samples,
                                const VadResult* vad_result,
                                AudioSegment* silence_segments, int max_segments);

/**
 * @brief 合并连续的语音段
 * 
 * @param segments 音频段数组
 * @param num_segments 段数量
 * @param min_gap_ms 最小间隔（毫秒）
 * @param sample_rate 采样率
 * @param merged_segments 合并后段输出数组
 * @param max_merged 最大合并段数
 * @return int 成功返回合并后段数，失败返回-1
 */
int vad_merge_segments(AudioSegment* segments, int num_segments,
                      int min_gap_ms, int sample_rate,
                      AudioSegment* merged_segments, int max_merged);

/**
 * @brief 计算音频帧能量
 * 
 * @param audio_frame 音频帧数据
 * @param frame_samples 帧样本数
 * @return float 帧能量
 */
float vad_compute_frame_energy(const float* audio_frame, int frame_samples);

/**
 * @brief 计算音频帧过零率
 * 
 * @param audio_frame 音频帧数据
 * @param frame_samples 帧样本数
 * @return float 过零率
 */
float vad_compute_zero_crossing_rate(const float* audio_frame, int frame_samples);

/**
 * @brief 计算音频帧频谱熵
 * 
 * @param audio_frame 音频帧数据
 * @param frame_samples 帧样本数
 * @param sample_rate 采样率
 * @return float 频谱熵
 */
float vad_compute_spectral_entropy(const float* audio_frame, int frame_samples,
                                  int sample_rate);

/**
 * @brief 计算音频特征
 * 
 * @param audio_frame 音频帧数据
 * @param frame_samples 帧样本数
 * @param sample_rate 采样率
 * @param features 特征输出数组
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int vad_compute_features(const float* audio_frame, int frame_samples,
                        int sample_rate, float* features, int max_features);

/**
 * @brief 训练基于机器学习的VAD模型
 * 
 * @param processor 处理器句柄
 * @param training_audio 训练音频数组
 * @param training_labels 训练标签数组（0=静音，1=语音）
 * @param num_samples 样本数量
 * @param num_epochs 训练轮数
 * @param learning_rate 学习率
 * @param batch_size 批次大小
 * @return int 成功返回0，失败返回-1
 */
int vad_train_machine_learning(VadProcessor* processor,
                              const float** training_audio,
                              const int* training_labels,
                              int num_samples, int num_epochs,
                              float learning_rate, int batch_size);

/**
 * @brief 保存VAD模型
 * 
 * @param processor 处理器句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int vad_save_model(const VadProcessor* processor, const char* filepath);

/**
 * @brief 加载VAD模型
 * 
 * @param processor 处理器句柄
 * @param filepath 文件路径
 * @return int 成功返回0，失败返回-1
 */
int vad_load_model(VadProcessor* processor, const char* filepath);

/**
 * @brief 获取默认VAD配置
 * 
 * @param config 配置输出
 */
void vad_default_config(VadConfig* config);

/**
 * @brief 验证VAD配置
 * 
 * @param config VAD配置
 * @return int 有效返回0，无效返回-1
 */
int vad_validate_config(const VadConfig* config);

/**
 * @brief 释放VAD结果内存
 * 
 * @param result VAD结果
 */
void vad_result_free(VadResult* result);

/**
 * @brief 释放音频分段结果内存
 * 
 * @param result 音频分段结果
 */
void vad_segmentation_result_free(AudioSegmentationResult* result);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_VAD_H