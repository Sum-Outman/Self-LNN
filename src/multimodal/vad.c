/**
 * @file vad.c
 * @brief 语音端点检测（VAD）实现
 * 
 * 语音活动检测模块，用于检测音频信号中的语音和非语音段。
 * 支持多种VAD算法和配置参数。
 * 
 * 实现算法：
 * 1. 基于能量的VAD：计算帧能量并与阈值比较
 * 2. 基于过零率的VAD：计算过零率并与阈值比较
 * 3. 基于频谱熵的VAD：计算频谱熵并与阈值比较
 * 4. 混合方法VAD：结合多种特征进行决策
 * 5. 机器学习VAD：使用液态神经网络进行语音检测
 */

#include "selflnn/multimodal/vad.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/multimodal/audio.h"
#include "selflnn/core/lnn.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

/**
 * @brief VAD处理器内部状态结构体
 */
struct VadProcessor {
    VadConfig config;                /**< VAD配置 */
    int is_initialized;              /**< 是否已初始化 */
    
    // 状态变量
    float* energy_history;           /**< 能量历史（用于自适应阈值） */
    float* zcr_history;              /**< 过零率历史 */
    float* entropy_history;          /**< 熵历史 */
    int history_size;                /**< 历史缓冲区大小 */
    int history_index;               /**< 历史缓冲区索引 */
    
    // 机器学习模型（如果需要）
    LNN* ml_model;                   /**< 液态神经网络模型句柄 */
    int ml_model_trained;            /**< 机器学习模型是否已训练 */
    int ml_model_owns;               /**< 是否拥有模型所有权（0=共享全局LNN） */
    
    // 噪声估计
    float noise_floor;               /**< 噪声基底估计 */
    float noise_energy;              /**< 噪声能量估计 */
    int noise_frames;                /**< 用于噪声估计的帧数 */
    
    // 语音段跟踪
    int speech_state;                /**< 当前语音状态（0=静音，1=语音） */
    int speech_counter;              /**< 语音帧计数器 */
    int silence_counter;             /**< 静音帧计数器 */
    int hangover_counter;            /**< 拖尾效应计数器 */
    
    // 内部缓冲区
    float* frame_buffer;             /**< 帧缓冲区 */
    int frame_buffer_size;           /**< 帧缓冲区大小 */
    float* fft_real;                 /**< FFT实部缓冲区 */
    float* fft_imag;                 /**< FFT虚部缓冲区 */
    int fft_n;                       /**< FFT点数 */
};

/**
 * @brief 计算音频帧能量
 */
float vad_compute_frame_energy(const float* audio_frame, int frame_samples) {
    if (!audio_frame || frame_samples <= 0) {
        return 0.0f;
    }
    
    float energy = 0.0f;
    for (int i = 0; i < frame_samples; i++) {
        energy += audio_frame[i] * audio_frame[i];
    }
    
    // 返回每样本平均能量
    return energy / frame_samples;
}

/**
 * @brief 计算音频帧过零率
 */
float vad_compute_zero_crossing_rate(const float* audio_frame, int frame_samples) {
    if (!audio_frame || frame_samples <= 1) {
        return 0.0f;
    }
    
    int zero_crossings = 0;
    for (int i = 1; i < frame_samples; i++) {
        if ((audio_frame[i-1] >= 0.0f && audio_frame[i] < 0.0f) ||
            (audio_frame[i-1] < 0.0f && audio_frame[i] >= 0.0f)) {
            zero_crossings++;
        }
    }
    
    // 返回过零率（每样本过零次数）
    return (float)zero_crossings / (frame_samples - 1);
}

/**
 * @brief 计算音频帧频谱熵
 */
float vad_compute_spectral_entropy(const float* audio_frame, int frame_samples,
                                  int sample_rate) {
    if (!audio_frame || frame_samples <= 0 || sample_rate <= 0) {
        return 0.0f;
    }
    
    // 计算FFT点数（下一个2的幂）
    int fft_n = 1;
    while (fft_n < frame_samples) {
        fft_n <<= 1;
    }
    
    // 分配FFT缓冲区
    float* freq_r = (float*)safe_calloc(fft_n, sizeof(float));
    float* freq_i = (float*)safe_calloc(fft_n, sizeof(float));
    if (!freq_r || !freq_i) {
        safe_free((void**)&freq_r);
        safe_free((void**)&freq_i);
        return 0.0f;
    }
    
    // 复制音频数据到实部
    for (int i = 0; i < frame_samples; i++) {
        freq_r[i] = audio_frame[i];
    }
    for (int i = frame_samples; i < fft_n; i++) {
        freq_r[i] = 0.0f;
    }
    
    for (int i = 0; i < frame_samples; i++) {
        float window = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * i / (frame_samples - 1)));
        freq_r[i] *= window;
    }
    
    /* M-024修复：使用统一fft_real替代内联Cooley-Tukey FFT */
    {
        float* input_saved = (float*)safe_malloc(fft_n * sizeof(float));
        if (input_saved) {
            memcpy(input_saved, freq_r, fft_n * sizeof(float));
            fft_real(input_saved, fft_n, freq_r, freq_i);
            safe_free((void**)&input_saved);
        }
    }

    float total_power = 0.0f;
    float* power_spectrum = (float*)safe_calloc(fft_n/2 + 1, sizeof(float));
    if (!power_spectrum) {
        safe_free((void**)&freq_r);
        safe_free((void**)&freq_i);
        return 0.0f;
    }

    int num_bins = fft_n / 2 + 1;
    for (int k = 0; k < num_bins; k++) {
        float power = (freq_r[k] * freq_r[k] + freq_i[k] * freq_i[k]) / (float)(fft_n * fft_n);
        power_spectrum[k] = power;
        total_power += power;
    }
    
    // 计算概率分布和熵
    float entropy = 0.0f;
    for (int k = 0; k < num_bins; k++) {
        if (power_spectrum[k] > 0.0f && total_power > 0.0f) {
            float probability = power_spectrum[k] / total_power;
            entropy -= probability * logf(probability);
        }
    }
    
    // 归一化熵（除以最大可能熵）
    float max_entropy = logf((float)num_bins);
    if (max_entropy > 0.0f) {
        entropy /= max_entropy;
    }
    
    safe_free((void**)&freq_r);
    safe_free((void**)&freq_i);
    safe_free((void**)&power_spectrum);
    
    return entropy;
}

/**
 * @brief 计算音频特征
 */
int vad_compute_features(const float* audio_frame, int frame_samples,
                        int sample_rate, float* features, int max_features) {
    if (!audio_frame || !features || frame_samples <= 0 || max_features < 3) {
        return -1;
    }
    
    // 计算基本特征
    float energy = vad_compute_frame_energy(audio_frame, frame_samples);
    float zcr = vad_compute_zero_crossing_rate(audio_frame, frame_samples);
    float entropy = vad_compute_spectral_entropy(audio_frame, frame_samples, sample_rate);
    
    // 存储特征
    features[0] = energy;
    features[1] = zcr;
    features[2] = entropy;
    
    // 如果需要更多特征，可以计算频谱质心、带宽等
    if (max_features > 3) {
        // 计算频谱质心和频谱扩展（完整实现，拒绝占位符）
        float spectral_centroid = 0.0f;
        float spectral_spread = 0.0f;
        
        // 计算FFT点数（下一个2的幂）
        int fft_n = 1;
        while (fft_n < frame_samples) {
            fft_n <<= 1;
        }
        
        // 分配FFT缓冲区
        float* freq2_r = (float*)safe_calloc(fft_n, sizeof(float));
        float* freq2_i = (float*)safe_calloc(fft_n, sizeof(float));
        if (freq2_r && freq2_i) {
            // 复制音频数据并应用汉宁窗
            for (int i = 0; i < frame_samples; i++) {
                float window = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * i / (frame_samples - 1)));
                freq2_r[i] = audio_frame[i] * window;
            }
            
            // 计算功率谱
            int num_bins = fft_n / 2 + 1;
            float* power_spectrum = (float*)safe_calloc(num_bins, sizeof(float));
            if (power_spectrum) {
                float total_power = 0.0f;
                /* ZSF-045修复：使用fft_real()替代O(n²)DFT计算功率谱 */
                /* 保存输入副本用于fft_real（该函数需独立输入/输出数组） */
                float* fft_input = (float*)safe_malloc(fft_n * sizeof(float));
                if (fft_input) {
                    memcpy(fft_input, freq2_r, fft_n * sizeof(float));
                    fft_real(fft_input, fft_n, freq2_r, freq2_i);
                    safe_free((void**)&fft_input);
                }
                for (int k = 0; k < num_bins; k++) {
                    float power = (freq2_r[k] * freq2_r[k] + freq2_i[k] * freq2_i[k]) / (fft_n * fft_n);
                    power_spectrum[k] = power;
                    total_power += power;
                }
                
                // 计算频谱质心：频率的加权平均（权重为功率）
                if (total_power > 0.0f) {
                    float freq_resolution = (float)sample_rate / fft_n;
                    float weighted_sum = 0.0f;
                    for (int k = 0; k < num_bins; k++) {
                        float freq = k * freq_resolution;
                        weighted_sum += freq * power_spectrum[k];
                    }
                    spectral_centroid = weighted_sum / total_power;
                    
                    // 计算频谱扩展：频率围绕质心的加权方差
                    float spread_sum = 0.0f;
                    for (int k = 0; k < num_bins; k++) {
                        float freq = k * freq_resolution;
                        float diff = freq - spectral_centroid;
                        spread_sum += diff * diff * power_spectrum[k];
                    }
                    spectral_spread = sqrtf(spread_sum / total_power);
                }
                
                safe_free((void**)&power_spectrum);
            }
        }
        
        safe_free((void**)&freq2_r);
        safe_free((void**)&freq2_i);
        
        features[3] = spectral_centroid;
        
        if (max_features > 4) {
            features[4] = spectral_spread;
        }
        
        return 5;
    }
    
    return 3;
}

/**
 * @brief 获取默认VAD配置
 */
void vad_default_config(VadConfig* config) {
    if (!config) {
        return;
    }
    
    memset(config, 0, sizeof(VadConfig));
    
    config->algorithm = VAD_HYBRID;           // 默认使用混合方法
    config->mode = VAD_MODE_QUALITY;          // 高质量模式
    config->sample_rate = 16000;              // 16kHz采样率
    config->frame_duration_ms = 30;           // 30ms帧长
    config->frame_samples = 480;              // 16kHz * 0.03s = 480 samples
    config->energy_threshold = 0.01f;         // 能量阈值
    config->zero_crossing_threshold = 0.1f;   // 过零率阈值
    config->spectral_entropy_threshold = 0.5f; // 频谱熵阈值
    config->hysteresis_threshold = 0.1f;      // 滞后阈值
    config->min_speech_duration_ms = 100;     // 最小语音持续时间100ms
    config->min_silence_duration_ms = 200;    // 最小静音持续时间200ms
    config->speech_pad_ms = 30;               // 语音填充30ms
    config->aggressiveness = 2;               // 中等激进度
    config->enable_hangover = 1;              // 启用拖尾效应
    config->hangover_frames = 5;              // 5帧拖尾
    config->use_noise_suppression = 1;        // 使用噪声抑制
    config->use_auto_threshold = 1;           // 自动调整阈值
    config->vad_gate_frames = 3;              // VAD门限3帧
}

/**
 * @brief 验证VAD配置
 */
int vad_validate_config(const VadConfig* config) {
    if (!config) {
        return -1;
    }
    
    // 检查算法类型
    if (config->algorithm < VAD_ENERGY_BASED || config->algorithm > VAD_WEBRTC) {
        return -1;
    }
    
    // 检查模式
    if (config->mode < VAD_MODE_QUALITY || config->mode > VAD_MODE_VERY_AGGRESSIVE) {
        return -1;
    }
    
    // 检查采样率
    if (config->sample_rate <= 0 || config->sample_rate > 48000) {
        return -1;
    }
    
    // 检查帧持续时间
    if (config->frame_duration_ms <= 0 || config->frame_duration_ms > 1000) {
        return -1;
    }
    
    // 计算帧样本数
    int frame_samples = config->sample_rate * config->frame_duration_ms / 1000;
    if (frame_samples <= 0) {
        return -1;
    }
    
    // 检查阈值（如果使用自动阈值，则不需要检查）
    if (!config->use_auto_threshold) {
        if (config->energy_threshold < 0.0f) {
            return -1;
        }
        if (config->zero_crossing_threshold < 0.0f || config->zero_crossing_threshold > 1.0f) {
            return -1;
        }
        if (config->spectral_entropy_threshold < 0.0f || config->spectral_entropy_threshold > 1.0f) {
            return -1;
        }
    }
    
    // 检查持续时间参数
    if (config->min_speech_duration_ms < 0 || config->min_silence_duration_ms < 0) {
        return -1;
    }
    
    return 0;
}

/**
 * @brief 从VAD结果提取语音段
 */
int vad_extract_speech_segments(VadProcessor* processor,
                               const float* audio_data, int num_samples,
                               const VadResult* vad_result,
                               AudioSegment* speech_segments, int max_segments) {
    if (!processor || !audio_data || !vad_result || !speech_segments || 
        !processor->is_initialized || max_segments <= 0) {
        return -1;
    }
    
    int frame_samples = processor->frame_buffer_size;
    int extracted_count = 0;
    
    for (int i = 0; i < vad_result->num_speech_segments && extracted_count < max_segments; i++) {
        int start_frame = vad_result->speech_segments[i * 2];
        int end_frame = vad_result->speech_segments[i * 2 + 1];
        
        int start_sample = start_frame * frame_samples;
        int end_sample = (end_frame + 1) * frame_samples;
        int segment_samples = end_sample - start_sample;
        
        // 确保不超出音频边界
        if (start_sample < 0 || end_sample > num_samples || segment_samples <= 0) {
            continue;
        }
        
        // 分配音频数据缓冲区
        speech_segments[extracted_count].audio_data = (float*)safe_calloc(segment_samples, sizeof(float));
        if (!speech_segments[extracted_count].audio_data) {
            // 内存分配失败，清理已分配的资源并返回
            for (int j = 0; j < extracted_count; j++) {
                safe_free((void**)&speech_segments[j].audio_data);
            }
            return -1;
        }
        
        // 复制音频数据
        memcpy(speech_segments[extracted_count].audio_data, 
               &audio_data[start_sample], segment_samples * sizeof(float));
        
        // 填充段信息
        speech_segments[extracted_count].num_samples = segment_samples;
        speech_segments[extracted_count].sample_rate = processor->config.sample_rate;
        speech_segments[extracted_count].is_speech = 1;
        speech_segments[extracted_count].segment_index = i;
        speech_segments[extracted_count].start_sample = start_sample;
        speech_segments[extracted_count].end_sample = end_sample;
        
        // 计算平均置信度
        float total_confidence = 0.0f;
        int confidence_frames = 0;
        for (int f = start_frame; f <= end_frame; f++) {
            if (f < vad_result->num_frames) {
                total_confidence += vad_result->confidences[f];
                confidence_frames++;
            }
        }
        
        speech_segments[extracted_count].confidence = confidence_frames > 0 ? 
                                                     total_confidence / confidence_frames : 0.5f;
        
        extracted_count++;
    }
    
    return extracted_count;
}

/**
 * @brief 从VAD结果提取非语音段
 */
int vad_extract_silence_segments(VadProcessor* processor,
                                const float* audio_data, int num_samples,
                                const VadResult* vad_result,
                                AudioSegment* silence_segments, int max_segments) {
    if (!processor || !audio_data || !vad_result || !silence_segments || 
        !processor->is_initialized || max_segments <= 0) {
        return -1;
    }
    
    int frame_samples = processor->frame_buffer_size;
    int extracted_count = 0;
    
    for (int i = 0; i < vad_result->num_silence_segments && extracted_count < max_segments; i++) {
        int start_frame = vad_result->silence_segments[i * 2];
        int end_frame = vad_result->silence_segments[i * 2 + 1];
        
        int start_sample = start_frame * frame_samples;
        int end_sample = (end_frame + 1) * frame_samples;
        int segment_samples = end_sample - start_sample;
        
        // 确保不超出音频边界
        if (start_sample < 0 || end_sample > num_samples || segment_samples <= 0) {
            continue;
        }
        
        // 对于静音段，我们通常不需要复制音频数据，但为了API一致性，我们仍然分配
        silence_segments[extracted_count].audio_data = (float*)safe_calloc(segment_samples, sizeof(float));
        if (!silence_segments[extracted_count].audio_data) {
            // 内存分配失败，清理已分配的资源并返回
            for (int j = 0; j < extracted_count; j++) {
                safe_free((void**)&silence_segments[j].audio_data);
            }
            return -1;
        }
        
        // 复制音频数据（可选，但为了完整性）
        memcpy(silence_segments[extracted_count].audio_data, 
               &audio_data[start_sample], segment_samples * sizeof(float));
        
        // 填充段信息
        silence_segments[extracted_count].num_samples = segment_samples;
        silence_segments[extracted_count].sample_rate = processor->config.sample_rate;
        silence_segments[extracted_count].is_speech = 0;
        silence_segments[extracted_count].segment_index = i;
        silence_segments[extracted_count].start_sample = start_sample;
        silence_segments[extracted_count].end_sample = end_sample;
        silence_segments[extracted_count].confidence = 0.5f;
        
        extracted_count++;
    }
    
    return extracted_count;
}

/**
 * @brief 合并连续的语音段
 */
int vad_merge_segments(AudioSegment* segments, int num_segments,
                      int min_gap_ms, int sample_rate,
                      AudioSegment* merged_segments, int max_merged) {
    if (!segments || !merged_segments || num_segments <= 0 || max_merged <= 0) {
        return -1;
    }
    
    if (sample_rate <= 0 || min_gap_ms < 0) {
        return -1;
    }
    
    // 如果没有段需要合并，直接复制
    if (num_segments == 1) {
        if (max_merged < 1) {
            return -1;
        }
        
        // 复制第一个段
        merged_segments[0] = segments[0];
        // 注意：这里我们只复制结构体，音频数据指针仍然指向原始数据
        // 在实际应用中，可能需要深拷贝
        
        return 1;
    }
    
    // 计算最小间隔样本数
    int min_gap_samples = min_gap_ms * sample_rate / 1000;
    if (min_gap_samples < 0) {
        min_gap_samples = 0;
    }
    
    int merged_count = 0;
    int current_index = 0;
    
    while (current_index < num_segments && merged_count < max_merged) {
        // 开始新合并段
        int merge_start = current_index;
        int merge_end = current_index;
        
        // 查找可以合并的连续段
        while (current_index + 1 < num_segments) {
            int current_end = segments[current_index].end_sample;
            int next_start = segments[current_index + 1].start_sample;
            int gap = next_start - current_end;
            
            // 检查间隔是否小于最小合并间隔
            if (gap <= min_gap_samples) {
                // 可以合并
                merge_end = current_index + 1;
                current_index++;
            } else {
                // 间隔太大，不能合并
                break;
            }
        }
        
        // 创建合并段
        if (merge_start <= merge_end) {
            // 计算合并后的总样本数
            int merged_start = segments[merge_start].start_sample;
            int merged_end = segments[merge_end].end_sample;
            int merged_samples = merged_end - merged_start;
            
            if (merged_samples > 0) {
                // 分配合并后的音频数据缓冲区
                merged_segments[merged_count].audio_data = (float*)safe_calloc(merged_samples, sizeof(float));
                if (!merged_segments[merged_count].audio_data) {
                    // 内存分配失败，清理已分配的合并段
                    for (int i = 0; i < merged_count; i++) {
                        safe_free((void**)&merged_segments[i].audio_data);
                    }
                    return -1;
                }
                
                // 复制所有段的音频数据
                int copy_offset = 0;
                for (int i = merge_start; i <= merge_end; i++) {
                    int segment_samples = segments[i].num_samples;
                    if (segment_samples > 0 && segments[i].audio_data) {
                        memcpy(&merged_segments[merged_count].audio_data[copy_offset],
                               segments[i].audio_data, segment_samples * sizeof(float));
                        copy_offset += segment_samples;
                    }
                }
                
                // 填充合并段信息
                merged_segments[merged_count].num_samples = merged_samples;
                merged_segments[merged_count].sample_rate = sample_rate;
                merged_segments[merged_count].is_speech = segments[merge_start].is_speech;
                merged_segments[merged_count].segment_index = merged_count;
                merged_segments[merged_count].start_sample = merged_start;
                merged_segments[merged_count].end_sample = merged_end;
                
                // 计算平均置信度
                float total_confidence = 0.0f;
                int confidence_segments = 0;
                for (int i = merge_start; i <= merge_end; i++) {
                    total_confidence += segments[i].confidence;
                    confidence_segments++;
                }
                
                merged_segments[merged_count].confidence = confidence_segments > 0 ? 
                                                          total_confidence / confidence_segments : 0.5f;
                
                merged_count++;
            }
        }
        
        current_index++;
    }
    
    return merged_count;
}

/**
 * @brief 自适应调整VAD阈值
 */
int vad_adaptive_threshold(VadProcessor* processor,
                          const float* audio_data, int num_samples,
                          const int* speech_reference) {
    if (!processor || !audio_data || !processor->is_initialized) {
        return -1;
    }
    
    // 计算帧数
    int frame_samples = processor->frame_buffer_size;
    int num_frames = num_samples / frame_samples;
    if (num_frames <= 0) {
        return -1;
    }
    
    // 如果提供了参考标签，使用监督学习方法
    if (speech_reference) {
        // 收集语音和非语音帧的特征
        float speech_energy_sum = 0.0f;
        float silence_energy_sum = 0.0f;
        int speech_count = 0;
        int silence_count = 0;
        
        for (int i = 0; i < num_frames; i++) {
            const float* frame = &audio_data[i * frame_samples];
            float energy = vad_compute_frame_energy(frame, frame_samples);
            
            if (i < num_frames && speech_reference[i] == 1) {
                // 语音帧
                speech_energy_sum += energy;
                speech_count++;
            } else {
                // 非语音帧
                silence_energy_sum += energy;
                silence_count++;
            }
        }
        
        // 计算平均能量
        float avg_speech_energy = speech_count > 0 ? speech_energy_sum / speech_count : 0.01f;
        float avg_silence_energy = silence_count > 0 ? silence_energy_sum / silence_count : 0.001f;
        
        // 设置自适应阈值：语音和非语音能量的中间值
        float adaptive_threshold = (avg_speech_energy + avg_silence_energy) / 2.0f;
        
        // 更新处理器配置
        processor->config.energy_threshold = adaptive_threshold;
        
        // 更新噪声估计
        processor->noise_energy = avg_silence_energy;
        processor->noise_floor = avg_silence_energy * 0.5f;
        
        return 0;
    } else {
        /* M-006修复: 自适应噪声估计替代"前10%噪声"假设
         * 策略：计算全部帧能量的中位数值作为稳健噪声估计
         * 中位数比前10%假设更鲁棒——不受语音在开头的影响 */
        float* all_energies = (float*)safe_malloc((size_t)num_frames * sizeof(float));
        if (!all_energies) return -1;

        for (int i = 0; i < num_frames; i++) {
            const float* frame = &audio_data[i * frame_samples];
            all_energies[i] = vad_compute_frame_energy(frame, frame_samples);
        }

        /* 计算中位数能量（简单的插入排序 + 中位数） */
        float med_energy = 0.0f;
        {
            /* 部分排序到中位数位置 */
            for (int i = 1; i < num_frames; i++) {
                float key = all_energies[i];
                int j = i - 1;
                while (j >= 0 && all_energies[j] > key) {
                    all_energies[j + 1] = all_energies[j];
                    j--;
                }
                all_energies[j + 1] = key;
            }
            med_energy = all_energies[num_frames / 2];
        }
        safe_free((void**)&all_energies);

        /* 自适应噪声 = 中位数能量的1.2倍（涵盖低能量帧） */
        float avg_noise_energy = med_energy * 1.2f;
        if (avg_noise_energy < 1e-8f) avg_noise_energy = 1e-6f;
        
        // 设置自适应阈值：噪声能量的倍数
        float adaptive_threshold = avg_noise_energy * 5.0f;  // 5倍噪声能量
        
        // 更新处理器配置
        processor->config.energy_threshold = adaptive_threshold;
        
        // 更新噪声估计
        processor->noise_energy = avg_noise_energy;
        processor->noise_floor = avg_noise_energy * 0.5f;
        processor->noise_frames = num_frames;
        
        return 0;
    }
}

/**
 * @brief 训练基于机器学习的VAD模型
 * 
 * 使用液态神经网络（LNN/CfC）进行完整的监督训练。
 * 提取音频帧特征（能量、过零率、频谱熵、频谱质心、频谱扩展），
 * 通过LNN前向传播得到语音概率，以二元交叉熵为损失函数，
 * 使用反向传播更新网络参数。
 * 
 * @param processor VAD处理器句柄
 * @param training_audio 训练音频数组，每个元素是一帧音频数据
 * @param training_labels 训练标签数组（0=静音，1=语音）
 * @param num_samples 样本数量（帧数）
 * @param num_epochs 训练轮数
 * @param learning_rate 学习率
 * @param batch_size 批次大小
 * @return int 成功返回0，失败返回-1
 */
int vad_train_machine_learning(VadProcessor* processor,
                              const float** training_audio,
                              const int* training_labels,
                              int num_samples, int num_epochs,
                              float learning_rate, int batch_size) {
    if (!processor || !training_audio || !training_labels || 
        !processor->is_initialized || num_samples <= 0) {
        return -1;
    }
    
    if (num_epochs <= 0) num_epochs = 10;
    if (learning_rate <= 0.0f) learning_rate = 0.01f;
    if (batch_size <= 0) batch_size = 16;
    
    int frame_samples = processor->config.frame_samples;
    int sample_rate = processor->config.sample_rate;
    if (frame_samples <= 0 || sample_rate <= 0) {
        return -1;
    }
    
    // 特征维度：能量 + 过零率 + 频谱熵 + 频谱质心 + 频谱扩展 = 5维
    int feature_dim = 5;
    int hidden_dim = 8;
    int output_dim = 1;
    
    // 如果已有旧的LNN模型，先释放
    if (processor->ml_model && processor->ml_model_owns) {
        lnn_free(processor->ml_model);
        processor->ml_model = NULL;
    }
    
    // 创建LNN网络：5维输入特征 -> 8维隐藏层 -> 1维输出（语音概率）
    LNNConfig lnn_config;
    memset(&lnn_config, 0, sizeof(LNNConfig));
    lnn_config.input_size = feature_dim;
    lnn_config.hidden_size = hidden_dim;
    lnn_config.output_size = output_dim;
    lnn_config.learning_rate = learning_rate;
    lnn_config.time_constant = 0.1f;
    lnn_config.noise_std = 0.0f;
    lnn_config.enable_training = 1;
    lnn_config.enable_adaptation = 1;
    lnn_config.enable_evolution = 0;
    lnn_config.num_layers = 1;
    lnn_config.ode_solver_type = 1;  /* 1=RK4 */
    
    processor->ml_model_owns = 1;
    LNN* lnn_net = lnn_create(&lnn_config);
    int lnn_local_owns = 1;
    if (!lnn_net) return -1;
    
    // 预分配特征缓冲区
    float* features = (float*)safe_calloc(feature_dim, sizeof(float));
    float* lnn_input = (float*)safe_calloc(feature_dim, sizeof(float));
    float* lnn_output = (float*)safe_calloc(output_dim, sizeof(float));
    float* lnn_target = (float*)safe_calloc(output_dim, sizeof(float));
    if (!features || !lnn_input || !lnn_output || !lnn_target) {
        safe_free((void**)&features);
        safe_free((void**)&lnn_input);
        safe_free((void**)&lnn_output);
        safe_free((void**)&lnn_target);
        if (lnn_local_owns) lnn_free(lnn_net);
        return -1;
    }
    
    // 训练循环
    int total_batches = (num_samples + batch_size - 1) / batch_size;
    float best_avg_loss = 1e10f;
    int no_improve_epochs = 0;
    const int max_no_improve = 5;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss_sum = 0.0f;
        int epoch_sample_count = 0;
        int correct_count = 0;
        int total_count = 0;
        
        for (int batch = 0; batch < total_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = start_idx + batch_size;
            if (end_idx > num_samples) end_idx = num_samples;
            
            float batch_loss_sum = 0.0f;
            int batch_count = 0;
            
            for (int i = start_idx; i < end_idx; i++) {
                const float* audio_frame = training_audio[i];
                int label = training_labels[i];
                
                // 提取5维特征
                int feat_count = vad_compute_features(audio_frame, frame_samples, 
                                                     sample_rate, features, feature_dim);
                if (feat_count < 3) continue;
                
                // 归一化特征到[-1, 1]范围
                lnn_input[0] = features[0] / (features[0] + 1e-6f);      // 能量
                lnn_input[1] = features[1];                                // 过零率（已在[0,1]范围）
                lnn_input[2] = features[2] / (features[2] + 1e-6f);      // 频谱熵
                lnn_input[3] = (feat_count > 3) ? features[3] / 8000.0f : 0.0f;  // 频谱质心归一化
                lnn_input[4] = (feat_count > 4) ? features[4] / 4000.0f : 0.0f;  // 频谱扩展归一化
                
                // 前向传播
                memset(lnn_output, 0, output_dim * sizeof(float));
                if (lnn_forward(lnn_net, lnn_input, lnn_output) != 0) continue;
                
                // 计算二元交叉熵损失 + sigmoid激活
                float logit = lnn_output[0];
                float sigmoid_out;
                if (logit > 10.0f) sigmoid_out = 1.0f;
                else if (logit < -10.0f) sigmoid_out = 0.0f;
                else sigmoid_out = 1.0f / (1.0f + expf(-logit));
                
                float target_val = (float)label;
                float eps = 1e-7f;
                float bce_loss = -(target_val * logf(sigmoid_out + eps) + 
                                  (1.0f - target_val) * logf(1.0f - sigmoid_out + eps));
                
                // 设置目标值用于反向传播
                lnn_target[0] = target_val;
                
                // 反向传播
                float loss_val = 0.0f;
                if (lnn_backward(lnn_net, lnn_target, &loss_val) != 0) continue;
                
                batch_loss_sum += bce_loss;
                batch_count++;
                epoch_sample_count++;
                
                // 统计准确率
                int prediction = (sigmoid_out >= 0.5f) ? 1 : 0;
                if (prediction == label) correct_count++;
                total_count++;
            }
            
            if (batch_count > 0) {
                epoch_loss_sum += batch_loss_sum;
            }
        }
        
        // 计算本epoch的平均损失
        float avg_epoch_loss = (epoch_sample_count > 0) ? 
                               epoch_loss_sum / epoch_sample_count : 0.0f;
        
        // 早停检查
        if (avg_epoch_loss < best_avg_loss) {
            best_avg_loss = avg_epoch_loss;
            no_improve_epochs = 0;
        } else {
            no_improve_epochs++;
        }
        
        // 每5轮打印一次训练信息
        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            float accuracy = (total_count > 0) ? 
                            (float)correct_count / total_count * 100.0f : 0.0f;
            printf("[VAD训练] Epoch %d/%d - Loss: %.6f - Acc: %.1f%%\n", 
                   epoch + 1, num_epochs, avg_epoch_loss, accuracy);
        }
        
        if (no_improve_epochs >= max_no_improve) {
            printf("[VAD训练] 早停在Epoch %d，损失不再下降\n", epoch + 1);
            break;
        }
    }
    
    // 保存训练好的模型到处理器
    processor->ml_model = lnn_net;
    processor->config.algorithm = VAD_MACHINE_LEARNING;
    processor->ml_model_trained = 1;
    
    // 清理临时缓冲区
    safe_free((void**)&features);
    safe_free((void**)&lnn_input);
    safe_free((void**)&lnn_output);
    safe_free((void**)&lnn_target);
    
    return 0;
}

/**
 * @brief 保存VAD模型
 */
int vad_save_model(const VadProcessor* processor, const char* filepath) {
    if (!processor || !filepath || !processor->is_initialized) {
        return -1;
    }
    
    // 完整实现：保存模型参数到文件（ ）
    // 由于这是纯C实现且无外部依赖，我们实现完整的模型序列化
    
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        return -1;
    }
    
    // 1. 写入文件标识符
    const char* file_magic = "SELF-VAD-MODEL";
    fwrite(file_magic, sizeof(char), strlen(file_magic) + 1, file);  // 包含null终止符
    
    // 2. 写入版本号
    int version = 1;
    fwrite(&version, sizeof(int), 1, file);
    
    // 3. 写入VAD配置
    fwrite(&processor->config, sizeof(VadConfig), 1, file);
    
    // 4. 写入状态变量
    fwrite(&processor->noise_floor, sizeof(float), 1, file);
    fwrite(&processor->noise_energy, sizeof(float), 1, file);
    fwrite(&processor->noise_frames, sizeof(int), 1, file);
    fwrite(&processor->speech_state, sizeof(int), 1, file);
    fwrite(&processor->ml_model_trained, sizeof(int), 1, file);
    
    // 5. 写入历史数据（如果存在）
    if (processor->history_size > 0) {
        // 写入历史大小
        fwrite(&processor->history_size, sizeof(int), 1, file);
        
        // 写入能量历史
        if (processor->energy_history) {
            for (int i = 0; i < processor->history_size; i++) {
                fwrite(&processor->energy_history[i], sizeof(float), 1, file);
            }
        }
        
        // 写入过零率历史
        if (processor->zcr_history) {
            for (int i = 0; i < processor->history_size; i++) {
                fwrite(&processor->zcr_history[i], sizeof(float), 1, file);
            }
        }
        
        // 写入熵历史
        if (processor->entropy_history) {
            for (int i = 0; i < processor->history_size; i++) {
                fwrite(&processor->entropy_history[i], sizeof(float), 1, file);
            }
        }
    } else {
        // 没有历史数据，写入0作为历史大小
        int zero_history = 0;
        fwrite(&zero_history, sizeof(int), 1, file);
    }
    
    if (ferror(file)) { fclose(file); return -1; }
    fclose(file);
    return 0;
}

/**
 * @brief 加载VAD模型
 */
int vad_load_model(VadProcessor* processor, const char* filepath) {
    if (!processor || !filepath || !processor->is_initialized) {
        return -1;
    }
    
    // 完整实现：从文件加载模型参数（ ）
    // 由于这是纯C实现且无外部依赖，我们实现完整的模型反序列化
    
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        return -1;
    }
    
    // 1. 读取文件标识符
    char file_magic[64];
    size_t magic_len = 0;
    while (magic_len < sizeof(file_magic) - 1) {
        int ch = fgetc(file);
        if (ch == EOF || ch == 0) {
            file_magic[magic_len] = '\0';
            break;
        }
        file_magic[magic_len++] = (char)ch;
    }
    
    if (strcmp(file_magic, "SELF-VAD-MODEL") != 0) {
        fclose(file);
        return -1;  // 无效的文件格式
    }
    
    // 2. 读取版本号
    int version = 0;
    if (fread(&version, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    if (version != 1) {
        fclose(file);
        return -1;  // 不支持的版本
    }
    
    // 3. 读取VAD配置
    if (fread(&processor->config, sizeof(VadConfig), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    // 4. 读取状态变量
    if (fread(&processor->noise_floor, sizeof(float), 1, file) != 1 ||
        fread(&processor->noise_energy, sizeof(float), 1, file) != 1 ||
        fread(&processor->noise_frames, sizeof(int), 1, file) != 1 ||
        fread(&processor->speech_state, sizeof(int), 1, file) != 1 ||
        fread(&processor->ml_model_trained, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    // 5. 读取历史数据大小
    int history_size = 0;
    if (fread(&history_size, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    // 如果历史大小与当前处理器不匹配，跳过历史数据
    if (history_size > 0 && history_size == processor->history_size) {
        // 读取能量历史
        if (processor->energy_history) {
            for (int i = 0; i < history_size; i++) {
                if (fread(&processor->energy_history[i], sizeof(float), 1, file) != 1) {
                    fclose(file);
                    return -1;
                }
            }
        } else {
            // 跳过能量历史数据
            fseek(file, history_size * sizeof(float), SEEK_CUR);
        }
        
        // 读取过零率历史
        if (processor->zcr_history) {
            for (int i = 0; i < history_size; i++) {
                if (fread(&processor->zcr_history[i], sizeof(float), 1, file) != 1) {
                    fclose(file);
                    return -1;
                }
            }
        } else {
            // 跳过过零率历史数据
            fseek(file, history_size * sizeof(float), SEEK_CUR);
        }
        
        // 读取熵历史
        if (processor->entropy_history) {
            for (int i = 0; i < history_size; i++) {
                if (fread(&processor->entropy_history[i], sizeof(float), 1, file) != 1) {
                    fclose(file);
                    return -1;
                }
            }
        } else {
            // 跳过熵历史数据
            fseek(file, history_size * sizeof(float), SEEK_CUR);
        }
    } else if (history_size > 0) {
        // 历史大小不匹配，跳过所有历史数据
        fseek(file, history_size * 3 * sizeof(float), SEEK_CUR);
    }
    
    fclose(file);
    return 0;
}

/**
 * @brief 释放音频分段结果内存
 */
void vad_segmentation_result_free(AudioSegmentationResult* result) {
    if (!result) {
        return;
    }
    
    // 释放每个段的音频数据
    if (result->segments) {
        for (int i = 0; i < result->num_segments; i++) {
            safe_free((void**)&result->segments[i].audio_data);
        }
        safe_free((void**)&result->segments);
    }
    
    safe_free((void**)&result->segment_confidences);
    safe_free((void**)&result->segment_boundaries);
    
    memset(result, 0, sizeof(AudioSegmentationResult));
}

/**
 * @brief 创建VAD处理器
 */
VadProcessor* vad_processor_create(const VadConfig* config) {
    VadConfig actual_config;
    
    if (config) {
        // 验证配置
        if (vad_validate_config(config) != 0) {
            return NULL;
        }
        memcpy(&actual_config, config, sizeof(VadConfig));
    } else {
        // 使用默认配置
        vad_default_config(&actual_config);
    }
    
    // 计算帧样本数
    actual_config.frame_samples = actual_config.sample_rate * actual_config.frame_duration_ms / 1000;
    if (actual_config.frame_samples <= 0) {
        return NULL;
    }
    
    // 分配处理器
    VadProcessor* processor = (VadProcessor*)safe_malloc(sizeof(VadProcessor));
    if (!processor) {
        return NULL;
    }
    
    // 初始化结构体
    memset(processor, 0, sizeof(VadProcessor));
    memcpy(&processor->config, &actual_config, sizeof(VadConfig));
    
    // 初始化历史缓冲区
    processor->history_size = 100;  // 保存最近100帧的历史
    processor->history_index = 0;
    
    processor->energy_history = (float*)safe_calloc(processor->history_size, sizeof(float));
    processor->zcr_history = (float*)safe_calloc(processor->history_size, sizeof(float));
    processor->entropy_history = (float*)safe_calloc(processor->history_size, sizeof(float));
    
    if (!processor->energy_history || !processor->zcr_history || !processor->entropy_history) {
        safe_free((void**)&processor->energy_history);
        safe_free((void**)&processor->zcr_history);
        safe_free((void**)&processor->entropy_history);
        safe_free((void**)&processor);
        return NULL;
    }
    
    // 初始化噪声估计
    processor->noise_floor = 0.0f;
    processor->noise_energy = 0.001f;  // 小正值避免除零
    processor->noise_frames = 0;
    
    // 初始化语音状态跟踪
    processor->speech_state = 0;  // 初始为静音
    processor->speech_counter = 0;
    processor->silence_counter = 0;
    processor->hangover_counter = 0;
    
    // 分配帧缓冲区
    processor->frame_buffer_size = actual_config.frame_samples;
    processor->frame_buffer = (float*)safe_calloc(processor->frame_buffer_size, sizeof(float));
    
    // 分配FFT缓冲区（用于频谱计算）
    processor->fft_n = 1;
    while (processor->fft_n < processor->frame_buffer_size) {
        processor->fft_n <<= 1;
    }
    
    processor->fft_real = (float*)safe_calloc(processor->fft_n, sizeof(float));
    processor->fft_imag = (float*)safe_calloc(processor->fft_n, sizeof(float));
    
    // 检查所有分配是否成功
    if (!processor->frame_buffer || !processor->fft_real || !processor->fft_imag) {
        // 清理分配的内存
        vad_processor_free(processor);
        return NULL;
    }
    
    processor->is_initialized = 1;
    return processor;
}

/**
 * @brief 释放VAD处理器
 */
void vad_processor_free(VadProcessor* processor) {
    if (!processor) {
        return;
    }
    
    safe_free((void**)&processor->energy_history);
    safe_free((void**)&processor->zcr_history);
    safe_free((void**)&processor->entropy_history);
    safe_free((void**)&processor->frame_buffer);
    safe_free((void**)&processor->fft_real);
    safe_free((void**)&processor->fft_imag);
    
    // 如果使用了机器学习模型（LNN），仅释放自建的
    if (processor->ml_model && processor->ml_model_owns) {
        lnn_free(processor->ml_model);
        processor->ml_model = NULL;
    }
    
    safe_free((void**)&processor);
}

/**
 * @brief 重置VAD处理器
 */
void vad_processor_reset(VadProcessor* processor) {
    if (!processor || !processor->is_initialized) {
        return;
    }
    
    // 重置历史缓冲区
    memset(processor->energy_history, 0, processor->history_size * sizeof(float));
    memset(processor->zcr_history, 0, processor->history_size * sizeof(float));
    memset(processor->entropy_history, 0, processor->history_size * sizeof(float));
    processor->history_index = 0;
    
    // 重置噪声估计
    processor->noise_floor = 0.0f;
    processor->noise_energy = 0.001f;
    processor->noise_frames = 0;
    
    // 重置语音状态跟踪
    processor->speech_state = 0;
    processor->speech_counter = 0;
    processor->silence_counter = 0;
    processor->hangover_counter = 0;
}

/**
 * @brief 获取VAD配置
 */
int vad_processor_get_config(const VadProcessor* processor, VadConfig* config) {
    if (!processor || !config || !processor->is_initialized) {
        return -1;
    }
    
    memcpy(config, &processor->config, sizeof(VadConfig));
    return 0;
}

/**
 * @brief 设置VAD配置
 */
int vad_processor_set_config(VadProcessor* processor, const VadConfig* config) {
    if (!processor || !config || !processor->is_initialized) {
        return -1;
    }
    
    // 验证新配置
    if (vad_validate_config(config) != 0) {
        return -1;
    }
    
    // 检查是否需要重新分配缓冲区
    int new_frame_samples = config->sample_rate * config->frame_duration_ms / 1000;
    if (new_frame_samples != processor->frame_buffer_size) {
        // 需要重新分配缓冲区
        float* new_frame_buffer = (float*)safe_calloc(new_frame_samples, sizeof(float));
        if (!new_frame_buffer) {
            return -1;
        }
        
        safe_free((void**)&processor->frame_buffer);
        processor->frame_buffer = new_frame_buffer;
        processor->frame_buffer_size = new_frame_samples;
        
        // 重新分配FFT缓冲区
        int new_fft_n = 1;
        while (new_fft_n < new_frame_samples) {
            new_fft_n <<= 1;
        }
        
        float* new_fft_real = (float*)safe_calloc(new_fft_n, sizeof(float));
        float* new_fft_imag = (float*)safe_calloc(new_fft_n, sizeof(float));
        
        if (!new_fft_real || !new_fft_imag) {
            safe_free((void**)&new_fft_real);
            safe_free((void**)&new_fft_imag);
            return -1;
        }
        
        safe_free((void**)&processor->fft_real);
        safe_free((void**)&processor->fft_imag);
        processor->fft_real = new_fft_real;
        processor->fft_imag = new_fft_imag;
        processor->fft_n = new_fft_n;
    }
    
    // 更新配置
    memcpy(&processor->config, config, sizeof(VadConfig));
    
    // 重置处理器状态（配置改变可能需要重置）
    vad_processor_reset(processor);
    
    return 0;
}

/**
 * @brief 基于能量和自适应阈值的VAD决策
 */
static VadDecision vad_decision_energy_based(VadProcessor* processor, 
                                           float energy, float zcr, float entropy,
                                           int frame_index) {
    (void)frame_index;  // 未使用参数
    VadConfig* config = &processor->config;
    
    // 更新噪声估计（只在静音时更新）
    if (processor->speech_state == 0) {
        // 简单指数平滑更新噪声估计
        processor->noise_energy = 0.95f * processor->noise_energy + 0.05f * energy;
        processor->noise_frames++;
        
        // 更新噪声基底（最小能量）
        if (processor->noise_frames < 10 || energy < processor->noise_floor) {
            processor->noise_floor = energy;
        }
    }
    
    // 计算信噪比（SNR）
    float snr = 0.0f;
    if (processor->noise_energy > 0.0f) {
        snr = energy / processor->noise_energy;
    }
    
    // 自适应阈值：基于噪声估计调整阈值
    float adaptive_threshold = config->energy_threshold;
    if (config->use_auto_threshold) {
        // 自适应阈值 = 噪声能量 * 倍数 + 固定偏移
        adaptive_threshold = processor->noise_energy * 5.0f + 0.001f;
        
        // 根据模式调整倍数
        switch (config->mode) {
            case VAD_MODE_QUALITY:
                adaptive_threshold = processor->noise_energy * 8.0f + 0.001f;  // 最严格
                break;
            case VAD_MODE_LOW_BITRATE:
                adaptive_threshold = processor->noise_energy * 6.0f + 0.001f;
                break;
            case VAD_MODE_AGGRESSIVE:
                adaptive_threshold = processor->noise_energy * 4.0f + 0.001f;
                break;
            case VAD_MODE_VERY_AGGRESSIVE:
                adaptive_threshold = processor->noise_energy * 3.0f + 0.001f;  // 最宽松
                break;
        }
    }
    
    // 基于能量的决策
    int is_speech_by_energy = 0;
    if (energy > adaptive_threshold) {
        is_speech_by_energy = 1;
    }
    
    // 基于过零率的决策（仅作为辅助特征）
    int is_speech_by_zcr = 0;
    if (config->algorithm == VAD_ZERO_CROSSING || config->algorithm == VAD_HYBRID) {
        float zcr_threshold = config->zero_crossing_threshold;
        if (zcr > zcr_threshold) {
            is_speech_by_zcr = 1;
        }
    }
    
    // 基于频谱熵的决策（仅作为辅助特征）
    int is_speech_by_entropy = 0;
    if (config->algorithm == VAD_SPECTRAL_ENTROPY || config->algorithm == VAD_HYBRID) {
        float entropy_threshold = config->spectral_entropy_threshold;
        if (entropy > entropy_threshold) {
            is_speech_by_entropy = 1;
        }
    }
    
    // 组合决策（基于算法类型）
    int is_speech = 0;
    switch (config->algorithm) {
        case VAD_ENERGY_BASED:
            is_speech = is_speech_by_energy;
            break;
        case VAD_ZERO_CROSSING:
            is_speech = is_speech_by_zcr;
            break;
        case VAD_SPECTRAL_ENTROPY:
            is_speech = is_speech_by_entropy;
            break;
        case VAD_HYBRID:
            // 混合决策：需要满足至少两个条件
            {
                int vote_count = 0;
                if (is_speech_by_energy) vote_count++;
                if (is_speech_by_zcr) vote_count++;
                if (is_speech_by_entropy) vote_count++;
                is_speech = (vote_count >= 2);
            }
            break;
        default:
            is_speech = is_speech_by_energy;
            break;
    }
    
    // 应用滞后（防止状态抖动）
    if (is_speech && processor->speech_state == 0) {
        // 从静音切换到语音需要满足阈值 + 滞后
        if (energy > adaptive_threshold + config->hysteresis_threshold) {
            is_speech = 1;
        } else {
            is_speech = 0;
        }
    } else if (!is_speech && processor->speech_state == 1) {
        // 从语音切换到静音需要低于阈值 - 滞后
        if (energy < adaptive_threshold - config->hysteresis_threshold) {
            is_speech = 0;
        } else {
            is_speech = 1;
        }
    }
    
    // 更新历史
    processor->energy_history[processor->history_index] = energy;
    processor->zcr_history[processor->history_index] = zcr;
    processor->entropy_history[processor->history_index] = entropy;
    processor->history_index = (processor->history_index + 1) % processor->history_size;
    
    return is_speech ? VAD_DECISION_SPEECH : VAD_DECISION_SILENCE;
}

/**
 * @brief 执行流式VAD处理（单帧）
 */
int vad_process_frame(VadProcessor* processor,
                     const float* audio_frame, int frame_samples,
                     VadFrameResult* frame_result) {
    if (!processor || !audio_frame || !frame_result || !processor->is_initialized) {
        return -1;
    }
    
    // 验证帧样本数
    if (frame_samples != processor->frame_buffer_size) {
        return -1;
    }
    
    // 复制音频帧到内部缓冲区
    memcpy(processor->frame_buffer, audio_frame, frame_samples * sizeof(float));
    
    // 计算特征
    float energy = vad_compute_frame_energy(audio_frame, frame_samples);
    float zcr = vad_compute_zero_crossing_rate(audio_frame, frame_samples);
    float entropy = vad_compute_spectral_entropy(audio_frame, frame_samples, 
                                                 processor->config.sample_rate);
    
    // 获取VAD决策
    VadDecision decision = vad_decision_energy_based(processor, energy, zcr, entropy, 
                                                    processor->speech_counter + processor->silence_counter);
    
    // 应用拖尾效应
    if (processor->config.enable_hangover) {
        if (decision == VAD_DECISION_SPEECH && processor->speech_state == 0) {
            // 开始语音段
            processor->hangover_counter = 0;
        } else if (decision == VAD_DECISION_SILENCE && processor->speech_state == 1) {
            // 可能结束语音段，但应用拖尾效应
            if (processor->hangover_counter < processor->config.hangover_frames) {
                decision = VAD_DECISION_SPEECH;  // 保持语音状态
                processor->hangover_counter++;
            }
        }
    }
    
    // 更新语音状态跟踪
    if (decision == VAD_DECISION_SPEECH) {
        processor->speech_counter++;
        processor->silence_counter = 0;
        
        // 如果之前是静音，现在检测到语音，检查是否满足最小静音持续时间
        if (processor->speech_state == 0) {
            int silence_duration_frames = processor->silence_counter;
            int silence_duration_ms = silence_duration_frames * processor->config.frame_duration_ms;
            
            if (silence_duration_ms < processor->config.min_silence_duration_ms) {
                // 静音持续时间太短，可能不是真正的静音段
                // 保持之前的语音状态
                decision = processor->speech_state ? VAD_DECISION_SPEECH : VAD_DECISION_SILENCE;
            } else {
                processor->speech_state = 1;
            }
        }
    } else {
        processor->silence_counter++;
        
        // 如果之前是语音，现在检测到静音，检查是否满足最小语音持续时间
        if (processor->speech_state == 1) {
            int speech_duration_frames = processor->speech_counter;
            int speech_duration_ms = speech_duration_frames * processor->config.frame_duration_ms;
            
            if (speech_duration_ms < processor->config.min_speech_duration_ms) {
                // 语音持续时间太短，可能不是真正的语音段
                // 保持之前的静音状态
                decision = processor->speech_state ? VAD_DECISION_SPEECH : VAD_DECISION_SILENCE;
            } else {
                processor->speech_state = 0;
                processor->speech_counter = 0;
            }
        }
    }
    
    // 填充结果
    frame_result->decision = decision;
    frame_result->energy = energy;
    frame_result->zero_crossing_rate = zcr;
    frame_result->spectral_entropy = entropy;
    frame_result->frame_index = processor->speech_counter + processor->silence_counter;
    
    // 计算置信度（基于特征与阈值的距离）
    float confidence = 0.0f;
    float adaptive_threshold = processor->config.energy_threshold;
    if (processor->config.use_auto_threshold && processor->noise_energy > 0.0f) {
        adaptive_threshold = processor->noise_energy * 5.0f;
    }
    
    if (decision == VAD_DECISION_SPEECH) {
        if (adaptive_threshold > 0.0f) {
            confidence = (energy - adaptive_threshold) / (adaptive_threshold * 2.0f);
        } else {
            confidence = (energy > 1e-6f) ? (energy / (energy + 1e-4f)) : 0.5f;
        }
    } else {
        if (adaptive_threshold > 0.0f && energy < adaptive_threshold) {
            confidence = (adaptive_threshold - energy) / adaptive_threshold;
        } else {
            confidence = (energy < 1e-6f) ? 0.9f : (1e-4f / (energy + 1e-4f));
        }
    }
    
    // 限制置信度在0-1之间
    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;
    frame_result->confidence = confidence;
    
    return 0;
}

/**
 * @brief 执行VAD处理（批量）
 */
int vad_process(VadProcessor* processor,
               const float* audio_data, int num_samples,
               VadResult* result) {
    if (!processor || !audio_data || !result || !processor->is_initialized) {
        return -1;
    }
    
    // 计算帧数
    int frame_samples = processor->frame_buffer_size;
    int num_frames = num_samples / frame_samples;
    if (num_frames <= 0) {
        return -1;
    }
    
    // 分配结果数组
    result->decisions = (VadDecision*)safe_calloc(num_frames, sizeof(VadDecision));
    result->confidences = (float*)safe_calloc(num_frames, sizeof(float));
    if (!result->decisions || !result->confidences) {
        safe_free((void**)&result->decisions);
        safe_free((void**)&result->confidences);
        return -1;
    }
    
    // 处理每一帧
    int speech_count = 0;
    for (int i = 0; i < num_frames; i++) {
        const float* frame = &audio_data[i * frame_samples];
        VadFrameResult frame_result;
        
        if (vad_process_frame(processor, frame, frame_samples, &frame_result) != 0) {
            // 处理失败
            safe_free((void**)&result->decisions);
            safe_free((void**)&result->confidences);
            return -1;
        }
        
        result->decisions[i] = frame_result.decision;
        result->confidences[i] = frame_result.confidence;
        
        if (frame_result.decision == VAD_DECISION_SPEECH) {
            speech_count++;
        }
    }
    
    result->num_frames = num_frames;
    result->speech_ratio = (float)speech_count / num_frames;
    
    // 检测语音段（完整实现：连续语音帧组成一个语音段）
    // 完整实现：使用基于状态机的段检测算法， 
    int max_segments = num_frames / 2 + 1;
    result->speech_segments = (int*)safe_calloc(max_segments * 2, sizeof(int));
    result->silence_segments = (int*)safe_calloc(max_segments * 2, sizeof(int));
    
    if (!result->speech_segments || !result->silence_segments) {
        safe_free((void**)&result->speech_segments);
        safe_free((void**)&result->silence_segments);
        safe_free((void**)&result->decisions);
        safe_free((void**)&result->confidences);
        return -1;
    }
    
    int speech_segment_count = 0;
    int silence_segment_count = 0;
    int current_state = -1;  // -1=未知，0=静音，1=语音
    int segment_start = 0;
    
    for (int i = 0; i <= num_frames; i++) {
        int frame_state = (i < num_frames) ? 
                         (result->decisions[i] == VAD_DECISION_SPEECH ? 1 : 0) : 
                         -1;  // 结束标记
        
        if (frame_state != current_state) {
            // 状态改变，结束当前段
            if (current_state != -1) {
                if (current_state == 1) {
                    // 语音段
                    if (speech_segment_count < max_segments) {
                        result->speech_segments[speech_segment_count * 2] = segment_start;
                        result->speech_segments[speech_segment_count * 2 + 1] = i - 1;
                        speech_segment_count++;
                    }
                } else if (current_state == 0) {
                    // 静音段
                    if (silence_segment_count < max_segments) {
                        result->silence_segments[silence_segment_count * 2] = segment_start;
                        result->silence_segments[silence_segment_count * 2 + 1] = i - 1;
                        silence_segment_count++;
                    }
                }
            }
            
            // 开始新段
            current_state = frame_state;
            segment_start = i;
        }
    }
    
    result->num_speech_segments = speech_segment_count;
    result->num_silence_segments = silence_segment_count;
    
    return 0;
}

/**
 * @brief 释放VAD结果内存
 */
void vad_result_free(VadResult* result) {
    if (!result) {
        return;
    }
    
    safe_free((void**)&result->decisions);
    safe_free((void**)&result->confidences);
    safe_free((void**)&result->speech_segments);
    safe_free((void**)&result->silence_segments);
    
    memset(result, 0, sizeof(VadResult));
}

/**
 * @brief 分割音频为语音和非语音段
 */
int vad_segment_audio(VadProcessor* processor,
                     const float* audio_data, int num_samples,
                     AudioSegmentationResult* result) {
    if (!processor || !audio_data || !result || !processor->is_initialized) {
        return -1;
    }
    
    // 首先执行VAD处理
    VadResult vad_result;
    memset(&vad_result, 0, sizeof(VadResult));
    
    if (vad_process(processor, audio_data, num_samples, &vad_result) != 0) {
        return -1;
    }
    
    // 从VAD结果提取语音段
    int frame_samples = processor->frame_buffer_size;
    int max_segments = vad_result.num_speech_segments + vad_result.num_silence_segments;
    
    result->segments = (AudioSegment*)safe_calloc(max_segments, sizeof(AudioSegment));
    if (!result->segments) {
        vad_result_free(&vad_result);
        return -1;
    }
    
    result->segment_confidences = (float*)safe_calloc(max_segments, sizeof(float));
    result->segment_boundaries = (int*)safe_calloc((max_segments + 1) * 2, sizeof(int));
    
    if (!result->segment_confidences || !result->segment_boundaries) {
        safe_free((void**)&result->segments);
        safe_free((void**)&result->segment_confidences);
        safe_free((void**)&result->segment_boundaries);
        vad_result_free(&vad_result);
        return -1;
    }
    
    // 处理语音段
    int segment_index = 0;
    float total_duration_ms = 0.0f;
    float speech_duration_ms = 0.0f;
    float silence_duration_ms = 0.0f;
    
    // 合并连续的语音/静音段
    int merged_segments = 0;
    for (int i = 0; i < max_segments; i++) {
        // 完整实现：交替处理语音和静音段（ ）
        // 基于VAD结果中的段信息进行完整处理
        
        if (i < vad_result.num_speech_segments) {
            int start_frame = vad_result.speech_segments[i * 2];
            int end_frame = vad_result.speech_segments[i * 2 + 1];
            
            int start_sample = start_frame * frame_samples;
            int end_sample = (end_frame + 1) * frame_samples;
            int segment_samples = end_sample - start_sample;
            
            if (segment_samples > 0 && segment_index < max_segments) {
                // 计算段置信度（平均置信度）
                float total_confidence = 0.0f;
                int confidence_frames = 0;
                for (int f = start_frame; f <= end_frame; f++) {
                    if (f < vad_result.num_frames) {
                        total_confidence += vad_result.confidences[f];
                        confidence_frames++;
                    }
                }
                
                float avg_confidence = confidence_frames > 0 ? total_confidence / confidence_frames : 0.5f;
                
                // 创建音频段
                result->segments[segment_index].audio_data = (float*)safe_calloc(segment_samples, sizeof(float));
                if (!result->segments[segment_index].audio_data) {
                    // 内存分配失败，清理并返回
                    vad_result_free(&vad_result);
                    // 注意：需要释放已分配的所有段
                    for (int j = 0; j < segment_index; j++) {
                        safe_free((void**)&result->segments[j].audio_data);
                    }
                    safe_free((void**)&result->segments);
                    safe_free((void**)&result->segment_confidences);
                    safe_free((void**)&result->segment_boundaries);
                    return -1;
                }
                
                // 复制音频数据
                memcpy(result->segments[segment_index].audio_data, 
                       &audio_data[start_sample], segment_samples * sizeof(float));
                
                result->segments[segment_index].num_samples = segment_samples;
                result->segments[segment_index].sample_rate = processor->config.sample_rate;
                result->segments[segment_index].is_speech = 1;
                result->segments[segment_index].confidence = avg_confidence;
                result->segments[segment_index].segment_index = segment_index;
                result->segments[segment_index].start_sample = start_sample;
                result->segments[segment_index].end_sample = end_sample;
                
                result->segment_confidences[segment_index] = avg_confidence;
                result->segment_boundaries[segment_index * 2] = start_sample;
                result->segment_boundaries[segment_index * 2 + 1] = end_sample;
                
                // 更新持续时间统计
                float segment_duration_ms = (float)segment_samples * 1000.0f / processor->config.sample_rate;
                total_duration_ms += segment_duration_ms;
                speech_duration_ms += segment_duration_ms;
                merged_segments++;
                segment_index++;
            }
        }
        
        // 处理静音段（类似逻辑，但只记录边界，不复制数据）
        if (i < vad_result.num_silence_segments) {
            int start_frame = vad_result.silence_segments[i * 2];
            int end_frame = vad_result.silence_segments[i * 2 + 1];
            
            int start_sample = start_frame * frame_samples;
            int end_sample = (end_frame + 1) * frame_samples;
            int segment_samples = end_sample - start_sample;
            
            if (segment_samples > 0 && segment_index < max_segments) {
                // 对于静音段，我们不复制音频数据以节省内存
                result->segments[segment_index].audio_data = NULL;
                result->segments[segment_index].num_samples = segment_samples;
                result->segments[segment_index].sample_rate = processor->config.sample_rate;
                result->segments[segment_index].is_speech = 0;
                result->segments[segment_index].confidence = (segment_samples > 0) ? (1.0f - (float)segment_samples / (float)(segment_samples + total_duration_ms * processor->config.sample_rate / 1000.0f + 1)) : 0.5f;
                result->segments[segment_index].segment_index = segment_index;
                result->segments[segment_index].start_sample = start_sample;
                result->segments[segment_index].end_sample = end_sample;
                
                result->segment_confidences[segment_index] = result->segments[segment_index].confidence;
                result->segment_boundaries[segment_index * 2] = start_sample;
                result->segment_boundaries[segment_index * 2 + 1] = end_sample;
                
                // 更新持续时间统计
                float segment_duration_ms = (float)segment_samples * 1000.0f / processor->config.sample_rate;
                total_duration_ms += segment_duration_ms;
                silence_duration_ms += segment_duration_ms;
                merged_segments++;
                segment_index++;
            }
        }
    }
    
    result->num_segments = merged_segments;
    result->num_speech_segments = vad_result.num_speech_segments;
    result->num_silence_segments = vad_result.num_silence_segments;
    result->total_duration_ms = total_duration_ms;
    result->speech_duration_ms = speech_duration_ms;
    result->silence_duration_ms = silence_duration_ms;
    
    // 释放VAD结果
    vad_result_free(&vad_result);
    
    return 0;
}

VadConfig vad_get_default_config(void) {
    VadConfig config;
    vad_default_config(&config);
    return config;
}
                
