/**
 * @file audio_loader.h
 * @brief 纯C WAV音频文件加载器头文件
 *
 * K-004补充: WAV格式原生解码，供训练数据加载和音频处理管线使用。
 */

#ifndef SELFLNN_AUDIO_LOADER_H
#define SELFLNN_AUDIO_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 从WAV文件加载音频数据为float数组
 *
 * 自动检测声道数，立体声混合为单声道。
 * 支持8/16/24/32位PCM格式。
 *
 * @param filepath WAV文件路径
 * @param sample_rate_out 输出采样率 (Hz)
 * @param num_samples_out 输出采样点数
 * @param channels_out 输出声道数(可NULL)
 * @return float数组(需调用者以safe_free释放), 失败返回NULL
 */
float* audio_load_wav(const char* filepath, int* sample_rate_out,
                       int* num_samples_out, int* channels_out);

/**
 * @brief 释放audio_load_wav加载的音频数据
 */
void audio_wav_free(float* data);

/**
 * @brief 获取WAV文件信息(不加载全部数据)
 *
 * @param filepath WAV文件路径
 * @param sample_rate_out 输出采样率
 * @param num_channels_out 输出声道数
 * @param bits_per_sample_out 输出位深
 * @param duration_sec_out 输出时长(秒)
 * @return 0成功, -1失败
 */
int audio_wav_info(const char* filepath, int* sample_rate_out,
                    int* num_channels_out, int* bits_per_sample_out,
                    float* duration_sec_out);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_AUDIO_LOADER_H */
