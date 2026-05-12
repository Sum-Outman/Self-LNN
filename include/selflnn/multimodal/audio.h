/**
 * @file audio.h
 * @brief 音频处理模块接口
 *
 * 音频数据处理接口，支持音频特征提取和设备枚举。
 * 所有频域特征提取由统一的液态神经网络（CfC/LNN）在时域原始信号上完成，
 * 不使用FFT/MFCC等传统频域特征。
 * 严格遵循：所有模态→统一输入到同一个连续动态系统。
 */

#ifndef SELFLNN_AUDIO_H
#define SELFLNN_AUDIO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 音频处理配置
 */
typedef struct {
    int target_sample_rate;    /**< 目标采样率 (0=保持原采样率) */
    int feature_dimension;     /**< 特征维度（输出到LNN的维度） */
    int enable_cfc;            /**< 是否启用CfC连续时间演化 */
    size_t cfc_hidden_size;    /**< CfC隐藏状态维度（默认32） */
    float cfc_time_constant;   /**< CfC时间常数（默认0.1） */
} AudioConfig;

/**
 * @brief 音频设备类型
 */
typedef enum {
    AUDIO_DEVICE_INPUT = 0,   /**< 输入设备（麦克风） */
    AUDIO_DEVICE_OUTPUT = 1   /**< 输出设备（扬声器） */
} AudioDeviceType;

/**
 * @brief 音频设备信息
 */
typedef struct {
    char* device_id;             /**< 设备唯一标识符 */
    char* device_name;           /**< 设备名称 */
    AudioDeviceType device_type; /**< 设备类型（输入/输出） */
    int max_input_channels;      /**< 最大输入通道数（输入设备） */
    int max_output_channels;     /**< 最大输出通道数（输出设备） */
    int supported_sample_rates[16]; /**< 支持的采样率列表，0结尾 */
    int is_default;              /**< 是否为系统默认设备 */
    int is_physical;             /**< 是否为物理设备（而非虚拟设备） */
} AudioDeviceInfo;

/**
 * @brief 音频处理器句柄
 */
typedef struct AudioProcessor AudioProcessor;

/**
 * @brief 创建音频处理器
 *
 * @param config 音频配置
 * @return AudioProcessor* 处理器句柄，失败返回NULL
 */
AudioProcessor* audio_processor_create(const AudioConfig* config);

/**
 * @brief 释放音频处理器
 *
 * @param processor 处理器句柄
 */
void audio_processor_free(AudioProcessor* processor);

/**
 * @brief 处理音频数据
 *
 * 将原始音频时域信号通过液态神经网络转换为统一特征表示。
 * 不使用FFT/MFCC等传统频域特征，全部通过CfC ODE在时域上动态演化。
 *
 * @param processor 处理器句柄
 * @param sample_rate 采样率
 * @param num_samples 样本数
 * @param num_channels 通道数
 * @param data 音频数据
 * @param features 特征输出缓冲区
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int audio_process_samples(AudioProcessor* processor,
                         int sample_rate, int num_samples, int num_channels,
                         const float* data,
                         float* features, size_t max_features);

/**
 * @brief 提取时域特征（基础预处理）
 *
 * 将原始音频时域信号数值归一化到[-1,1]范围，
 * 作为CfC ODE演化的输入信号。
 *
 * @param processor 处理器句柄
 * @param num_samples 样本数
 * @param data 音频数据
 * @param time_features 时域特征输出缓冲区
 * @param max_features 最大特征数
 * @return int 成功返回特征数，失败返回-1
 */
int audio_extract_time_features(AudioProcessor* processor,
                               int num_samples,
                               const float* data,
                               float* time_features, size_t max_features);

/**
 * @brief 获取音频处理器配置
 *
 * @param processor 处理器句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int audio_processor_get_config(const AudioProcessor* processor, AudioConfig* config);

/**
 * @brief 设置音频处理器配置
 *
 * @param processor 处理器句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int audio_processor_set_config(AudioProcessor* processor, const AudioConfig* config);

/**
 * @brief 重置音频处理器
 *
 * @param processor 处理器句柄
 */
void audio_processor_reset(AudioProcessor* processor);

/**
 * @brief 枚举系统中所有音频设备
 *
 * @param devices 输出参数，设备数组指针。使用 audio_free_device_list 释放。
 * @param max_devices 最大设备数
 * @return int 实际发现的设备数，失败返回-1
 */
int audio_enum_devices(AudioDeviceInfo* devices, int max_devices);

/**
 * @brief 获取指定类型的音频设备数量
 *
 * @param type 设备类型（输入/输出）
 * @return int 设备数量，失败返回-1
 */
int audio_get_device_count(AudioDeviceType type);

/**
 * @brief 获取默认音频设备
 *
 * @param type 设备类型（输入/输出）
 * @return AudioDeviceInfo* 设备信息指针（堆分配），失败返回NULL
 */
AudioDeviceInfo* audio_get_default_device(AudioDeviceType type);

/**
 * @brief 释放音频设备列表
 *
 * @param devices 设备数组指针
 * @param count 设备数量
 */
void audio_free_device_list(AudioDeviceInfo* devices, int count);

/**
 * @brief 释放单个音频设备信息
 *
 * @param device 设备信息指针（由 audio_get_default_device 返回）
 */
void audio_free_device_info(AudioDeviceInfo* device);

/**
 * @brief 解码WAV音频文件为PCM浮点样本
 *
 * @param wav_data WAV文件二进制数据
 * @param data_size 数据总大小
 * @param samples 输出参数，解码后的浮点样本数组（调用者通过safe_free释放）
 * @param sample_rate 输出参数，采样率
 * @param num_samples 输出参数，样本总数
 * @param num_channels 输出参数，原始通道数
 * @return int 成功返回0，失败返回-1
 */
int audio_decode_wav(const unsigned char* wav_data, size_t data_size,
                     float** samples, int* sample_rate,
                     int* num_samples, int* num_channels);

/**
 * @brief 语音活动检测（VAD）- 短时能量+过零率双门限法
 *
 * @param samples PCM浮点样本数据（范围[-1,1]）
 * @param num_samples 样本总数
 * @param sample_rate 采样率
 * @param frame_energies 输出每帧短时能量（大小=num_frames，可NULL）
 * @param voice_flags 输出每帧语音标志（1=语音，0=静音，大小=num_frames）
 * @param frame_size 帧大小（样本数，0=自动20ms）
 * @param energy_threshold 能量阈值（0=自动估算）
 * @param zcr_threshold 过零率阈值（0=自动估算）
 * @return int 成功返回帧数，失败返回-1
 */
int audio_vad_detect(const float* samples, int num_samples, int sample_rate,
                     float* frame_energies, int* voice_flags,
                     int frame_size, float energy_threshold, float zcr_threshold);

/* ================================================================
 * 真实音频硬件采集接口（跨平台）
 * ================================================================ */

/**
 * @brief 音频采集设备信息
 */
typedef struct {
    char name[256];            /**< 设备名称 */
    char device_id[128];       /**< 设备标识符（平台相关） */
    int max_channels;          /**< 最大声道数 */
    int default_sample_rate;   /**< 默认采样率(Hz) */
} AudioCaptureDeviceInfo;

/**
 * @brief 音频采集上下文句柄
 */
typedef struct AudioCaptureContext AudioCaptureContext;

/**
 * @brief 枚举所有可用音频采集设备
 *
 * @param devices 输出设备信息数组（调用者管理内存）
 * @param max_devices 最大设备数
 * @return int 实际设备数，0表示无可用设备
 */
int audio_capture_enumerate_devices(AudioCaptureDeviceInfo* devices, int max_devices);

/**
 * @brief 创建音频采集上下文
 *
 * @param device_id 设备标识符（NULL=使用默认设备）
 * @param sample_rate 采样率(Hz)，推荐16000用于语音
 * @param channels 声道数，推荐1(单声道)
 * @param bits_per_sample 位深，推荐16
 * @return AudioCaptureContext* 采集上下文，无设备返回NULL
 */
AudioCaptureContext* audio_capture_create(const char* device_id,
                                           int sample_rate,
                                           int channels,
                                           int bits_per_sample);

/**
 * @brief 开始音频采集
 *
 * @param ctx 采集上下文
 * @param callback 音频数据回调(float样本数组, 样本数, 用户数据)
 * @param user_data 用户数据指针
 * @return int 0成功，-1失败
 */
int audio_capture_start(AudioCaptureContext* ctx,
                         void (*callback)(const float* samples, size_t num_samples, void* user_data),
                         void* user_data);

/**
 * @brief 音频采集处理循环
 *
 * @param ctx 采集上下文
 * @return int 读取的样本数，0=无新数据，-1=错误
 */
int audio_capture_process(AudioCaptureContext* ctx);

/**
 * @brief 停止音频采集
 *
 * @param ctx 采集上下文
 * @return int 0成功，-1失败
 */
int audio_capture_stop(AudioCaptureContext* ctx);

/**
 * @brief 释放音频采集上下文
 *
 * @param ctx 采集上下文
 */
void audio_capture_free(AudioCaptureContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_AUDIO_H
