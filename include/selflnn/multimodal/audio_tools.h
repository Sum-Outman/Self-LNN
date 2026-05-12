/**
 * @file audio_tools.h
 * @brief 音频处理辅助工具集
 *
 * 提供音频处理的高级工具函数：
 * - CQT时频分析（Constant-Q Transform）
 * - STFT短时傅里叶变换及逆变换
 * - 音频源分离（使用液态状态动态系统）
 * - 音频事件检测（使用液态状态动态系统）
 *
 * 所有实现均为纯C、无外部依赖。
 */

#ifndef SELFLNN_AUDIO_TOOLS_H
#define SELFLNN_AUDIO_TOOLS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================== *
 * 常量定义                                                         *
 * =============================================================== */

/** @brief 最大CQT频带数 */
#define AUDIO_CQT_MAX_BINS 336

/** @brief 最大CQT帧数 */
#define AUDIO_CQT_MAX_FRAMES 8192

/** @brief 最大孤立源数 */
#define AUDIO_SOURCE_SEP_MAX_SOURCES 8

/** @brief 最大事件类别数 */
#define AUDIO_EVENT_MAX_CLASSES 64

/** @brief 最大事件检测结果数 */
#define AUDIO_EVENT_MAX_RESULTS 1024

/** @brief STFT最大帧数 */
#define AUDIO_STFT_MAX_FRAMES 65536

/* =============================================================== *
 * CQT时频分析                                                      *
 * =============================================================== */

/**
 * @brief CQT配置结构体
 */
typedef struct {
    int sample_rate;            /**< 采样率（默认16000） */
    int bins_per_octave;        /**< 每八度频带数（默认24） */
    int num_bins;               /**< 总频带数（默认84，约3.5八度） */
    float min_freq;             /**< 最低频率（默认65.4Hz，C2） */
    float max_freq;             /**< 最高频率（默认2093Hz，C7） */
    float window_length_factor; /**< 窗长因子（默认1.0） */
} AudioCQTConfig;

/**
 * @brief CQT句柄（不透明结构体）
 */
typedef struct AudioCQT AudioCQT;

/**
 * @brief 创建CQT处理器（预计算稀疏核矩阵）
 *
 * 基于FFT的快速CQT实现。预计算稀疏核矩阵，
 * 将CQT计算转化为稀疏矩阵乘法。
 *
 * @param config CQT配置（NULL使用默认配置）
 * @return AudioCQT* 成功返回句柄，失败返回NULL
 */
AudioCQT* audio_cqt_create(const AudioCQTConfig* config);

/**
 * @brief 计算CQT时频表示
 *
 * @param cqt CQT句柄
 * @param samples 输入音频样本（float [-1,1]）
 * @param num_samples 样本数
 * @param magnitude 输出幅度谱 [num_bins x num_frames]（调用者分配）
 * @param max_mag 幅度谱缓冲区大小（至少 num_bins * num_frames）
 * @return int 成功返回帧数，失败返回-1
 */
int audio_cqt_compute(AudioCQT* cqt, const float* samples, int num_samples,
                      float* magnitude, int max_mag);

/**
 * @brief 获取CQT中心频率数组
 *
 * @param cqt CQT句柄
 * @param frequencies 输出频率数组（大小至少 num_bins）
 * @param max_freqs 缓冲区大小
 * @return int 成功返回频带数，失败返回-1
 */
int audio_cqt_get_frequencies(AudioCQT* cqt, float* frequencies, int max_freqs);

/**
 * @brief 获取CQT配置
 *
 * @param cqt CQT句柄
 * @param config 输出配置缓冲区
 * @return int 成功返回0，失败返回-1
 */
int audio_cqt_get_config(AudioCQT* cqt, AudioCQTConfig* config);

/**
 * @brief 释放CQT处理器
 *
 * @param cqt CQT句柄
 */
void audio_cqt_free(AudioCQT* cqt);

/* =============================================================== *
 * STFT短时傅里叶变换                                               *
 * =============================================================== */

/**
 * @brief 窗函数类型枚举
 */
typedef enum {
    AUDIO_WINDOW_HAMMING = 0,  /**< Hamming窗（默认） */
    AUDIO_WINDOW_HANNING = 1,  /**< Hanning窗 */
    AUDIO_WINDOW_BLACKMAN = 2, /**< Blackman窗 */
    AUDIO_WINDOW_RECT = 3      /**< 矩形窗 */
} AudioWindowType;

/**
 * @brief STFT配置结构体
 */
typedef struct {
    int fft_size;               /**< FFT大小（必须为2的幂，默认512） */
    int hop_size;               /**< 帧移（默认128，即1/4重叠） */
    AudioWindowType window_type; /**< 窗函数类型（默认Hamming） */
    int use_magnitude_only;     /**< 是否仅输出幅度谱（默认0=复数） */
} AudioSTFTConfig;

/**
 * @brief 计算STFT频谱图
 *
 * 分帧→加窗→FFT→输出频谱。
 * 输出为 [num_frames x (fft_size/2+1)] 的复数幅度谱。
 *
 * @param samples 输入音频样本（float [-1,1]）
 * @param num_samples 样本数
 * @param config STFT配置（NULL使用默认）
 * @param magnitude 输出幅度谱 [num_frames x num_bins]（调用者分配）
 * @param phase 输出相位谱 [num_frames x num_bins]（可为NULL，调用者分配）
 * @param max_frames magnitude缓冲区最大帧数
 * @return int 成功返回帧数，失败返回-1
 */
int audio_stft_compute(const float* samples, int num_samples,
                       const AudioSTFTConfig* config,
                       float* magnitude, float* phase, int max_frames);

/**
 * @brief 计算逆STFT（iSTFT）重建时域信号
 *
 * 幅度+相位 → 复数频谱 → iFFT → 重叠相加法 → 时域信号。
 *
 * @param magnitude 幅度谱 [num_frames x num_bins]
 * @param phase 相位谱 [num_frames x num_bins]（可为NULL，使用默认相位）
 * @param num_frames 帧数
 * @param config STFT配置（NULL使用默认）
 * @param output 输出信号缓冲区（调用者分配）
 * @param max_output 输出缓冲区最大大小
 * @return int 成功返回样本数，失败返回-1
 */
int audio_istft_compute(const float* magnitude, const float* phase,
                        int num_frames, const AudioSTFTConfig* config,
                        float* output, int max_output);

/**
 * @brief 获取STFT默认配置
 *
 * @return AudioSTFTConfig 默认配置
 */
AudioSTFTConfig audio_stft_default_config(void);

/**
 * @brief 生成窗函数系数
 *
 * @param window 输出窗函数数组（大小至少 size）
 * @param size 窗函数大小
 * @param type 窗函数类型
 */
void audio_generate_window(float* window, int size, AudioWindowType type);

/* =============================================================== *
 * 音频源分离                                                      *
 * =============================================================== */

/**
 * @brief 音频源分离配置结构体
 */
typedef struct {
    int sample_rate;            /**< 采样率（默认16000） */
    int num_sources;            /**< 目标源数量（默认2，最大AUDIO_SOURCE_SEP_MAX_SOURCES） */
    int fft_size;               /**< FFT大小（默认512） */
    int hop_size;               /**< 帧移（默认128） */
    size_t hidden_size;         /**< 隐藏状态大小（默认256） */
    float time_constant;        /**< 时间常数（默认0.05） */
    int ode_solver_type;        /**< 求解器类型（保留兼容） */
    int use_gpu;                /**< 是否使用GPU加速 */
} AudioSourceSepConfig;

/**
 * @brief 音频源分离句柄
 */
typedef struct AudioSourceSep AudioSourceSep;

/**
 * @brief 创建音频源分离器
 *
 * 使用液态状态动态系统用于多源分离。
 * 混合音频→频域特征→液态状态更新→频域掩蔽→N个分离源。
 *
 * @param config 分离配置（NULL使用默认）
 * @return AudioSourceSep* 成功返回句柄，失败返回NULL
 */
AudioSourceSep* audio_source_sep_create(const AudioSourceSepConfig* config);

/**
 * @brief 执行音频源分离
 *
 * @param separator 分离器句柄
 * @param mixed_audio 混合音频样本（float [-1,1]）
 * @param num_samples 样本数
 * @param separated 输出分离音频数组 [num_sources x output_samples]
 *                  每个源通道为连续float数组（调用者分配）
 * @param max_per_source 每源输出缓冲区最大样本数
 * @return int 成功返回每源样本数，失败返回-1
 */
int audio_source_sep_separate(AudioSourceSep* separator,
                               const float* mixed_audio, int num_samples,
                               float* separated, int max_per_source);

/**
 * @brief 重置分离器状态
 *
 * @param separator 分离器句柄
 */
void audio_source_sep_reset(AudioSourceSep* separator);

/**
 * @brief 释放音频源分离器
 *
 * @param separator 分离器句柄
 */
void audio_source_sep_free(AudioSourceSep* separator);

/* =============================================================== *
 * 音频事件检测                                                    *
 * =============================================================== */

/**
 * @brief 音频事件检测结果结构体
 */
typedef struct {
    int class_id;               /**< 事件类别ID（0~num_classes-1） */
    float confidence;           /**< 置信度 [0,1] */
    float start_time;           /**< 起始时间（秒） */
    float end_time;             /**< 结束时间（秒） */
} AudioEventResult;

/**
 * @brief 音频事件检测配置结构体
 */
typedef struct {
    int sample_rate;            /**< 采样率（默认16000） */
    int num_classes;            /**< 事件类别数（默认10，最大AUDIO_EVENT_MAX_CLASSES） */
    int fft_size;               /**< FFT大小（默认512） */
    int hop_size;               /**< 帧移（默认160） */
    size_t hidden_size;         /**< 隐藏状态大小（默认256） */
    float time_constant;        /**< 时间常数（默认0.05） */
    int ode_solver_type;        /**< 求解器类型（保留兼容） */
    float detection_threshold;  /**< 检测阈值 [0,1]（默认0.5） */
    int min_event_frames;       /**< 最小事件持续帧数（默认3） */
    float class_names[AUDIO_EVENT_MAX_CLASSES][32]; /**< 事件类别名称 */
} AudioEventConfig;

/**
 * @brief 音频事件检测器句柄
 */
typedef struct AudioEventDetector AudioEventDetector;

/**
 * @brief 创建音频事件检测器
 *
 * 使用液态状态动态系统：音频特征→状态更新→帧级事件分类→
 * 时间连续性后处理→事件段输出。
 *
 * @param config 检测器配置（NULL使用默认）
 * @return AudioEventDetector* 成功返回句柄，失败返回NULL
 */
AudioEventDetector* audio_event_detector_create(const AudioEventConfig* config);

/**
 * @brief 执行音频事件检测
 *
 * @param detector 检测器句柄
 * @param samples 音频样本（float [-1,1]）
 * @param num_samples 样本数
 * @param results 输出检测结果数组（调用者分配）
 * @param max_results 结果缓冲区大小
 * @return int 成功返回检测到的事件数，失败返回-1
 */
int audio_event_detector_detect(AudioEventDetector* detector,
                                 const float* samples, int num_samples,
                                 AudioEventResult* results, int max_results);

/**
 * @brief 获取帧级事件概率矩阵
 *
 * @param detector 检测器句柄
 * @param frame_probs 输出帧级概率 [num_frames x num_classes]（调用者分配）
 * @param max_frames 最大帧数
 * @param num_classes 输出类别数
 * @return int 成功返回帧数，失败返回-1
 */
int audio_event_detector_get_frame_probs(AudioEventDetector* detector,
                                          float* frame_probs,
                                          int max_frames, int* num_classes);

/**
 * @brief 重置检测器状态
 *
 * @param detector 检测器句柄
 */
void audio_event_detector_reset(AudioEventDetector* detector);

/**
 * @brief 释放音频事件检测器
 *
 * @param detector 检测器句柄
 */
void audio_event_detector_free(AudioEventDetector* detector);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_AUDIO_TOOLS_H */
