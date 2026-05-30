/**
 * @file audio.c
 * @brief 音频处理器实现
 *
 * 音频数据处理实现，支持音频特征提取和时域特征。
 * 所有频域特征提取（FFT/MFCC）已移除，统一由主CfC模型在时域原始信号上完成连续时间动态演化。
 * 严格遵循：所有模态→统一输入到同一个连续动态系统。
 */

#include "selflnn/multimodal/audio.h"
#include "selflnn/multimodal/audio_loader.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/**
 * @brief 音频处理器内部结构体
 */
struct AudioProcessor {
    AudioConfig config;           /**< 处理器配置 */
    int is_initialized;           /**< 是否已初始化 */
    float* sample_buffer;         /**< 样本缓冲区 */
    size_t buffer_size;           /**< 缓冲区大小 */
    /* CfC连续时间演化由multimodal.c中的主CfC统一管理 */
};

/**
 * @brief 创建音频处理器
 */
AudioProcessor* audio_processor_create(const AudioConfig* config) {
    if (!config) {
        return NULL;
    }

    AudioProcessor* processor = (AudioProcessor*)safe_malloc(sizeof(AudioProcessor));
    if (!processor) {
        return NULL;
    }

    memset(processor, 0, sizeof(AudioProcessor));
    processor->config = *config;
    processor->is_initialized = 1;

    processor->sample_buffer = NULL;
    processor->buffer_size = 0;

    return processor;
}

/**
 * @brief 释放音频处理器
 */
void audio_processor_free(AudioProcessor* processor) {
    if (!processor) {
        return;
    }

    safe_free((void**)&processor->sample_buffer);
    safe_free((void**)&processor);
}

/**
 * @brief 处理音频数据
 */
int audio_process_samples(AudioProcessor* processor,
                         int sample_rate, int num_samples, int num_channels,
                         const float* data,
                         float* features, size_t max_features) {
    if (!processor || !data || !features || max_features == 0) {
        return -1;
    }

    if (sample_rate <= 0 || num_samples <= 0 || num_channels <= 0) {
        return -1;
    }

    // 确定目标采样率
    int target_sample_rate = processor->config.target_sample_rate > 0 ?
                             processor->config.target_sample_rate : sample_rate;

    // 执行重采样（如果输入采样率与目标采样率不同）
    float* resampled_data = NULL;
    int resampled_count = num_samples;
    const float* process_data = data;

    if (sample_rate != target_sample_rate && target_sample_rate > 0) {
        float ratio = (float)target_sample_rate / (float)sample_rate;
        resampled_count = (int)(num_samples * ratio);
        if (resampled_count < 1) resampled_count = 1;

        resampled_data = (float*)safe_malloc(resampled_count * sizeof(float));
        if (resampled_data) {
            for (int i = 0; i < resampled_count; i++) {
                float src_pos = (float)i / ratio;
                int src_idx = (int)src_pos;
                float frac = src_pos - src_idx;
                if (src_idx >= num_samples - 1) {
                    resampled_data[i] = data[num_samples - 1];
                } else {
                    resampled_data[i] = data[src_idx] * (1.0f - frac) + data[src_idx + 1] * frac;
                }
            }
            process_data = resampled_data;
        }
    }

    // 提取时域特征（原始时域信号直接作为CfC ODE的输入）
    size_t feature_count = 0;

    if (max_features > feature_count) {
        int time_feat_count = audio_extract_time_features(processor, resampled_count, process_data,
                                                         features + feature_count,
                                                         max_features - feature_count);
        if (time_feat_count > 0) {
            feature_count += time_feat_count;
        }
    }

    safe_free((void**)&resampled_data);

    // 原始时域特征直接返回（由multimodal.c主CfC统一处理时序演化）

    return (int)feature_count;
}

/**
 * @brief 提取时域特征（完整实现）
 *
 * 提取丰富的时域特征供CfC模型处理：
 * 1. 原始信号归一化
 * 2. 过零率（ZCR）
 * 3. 短时能量（STE）
 * 4. 自相关基频估计
 * 5. 统计特征（均值、方差、偏度、峰度、动态范围、均方根）
 *
 * 输出布局：[原始信号(归一化), ZCR, STE, 基频, 均值, 方差, 偏度, 峰度, 动态范围, RMS, ...]
 */
int audio_extract_time_features(AudioProcessor* processor,
                               int num_samples,
                               const float* data,
                               float* time_features, size_t max_features) {
    if (!processor || !data || !time_features || max_features == 0) {
        return -1;
    }

    if (num_samples <= 0) {
        return -1;
    }

    /* === 1. 原始信号归一化输出 === */
    size_t raw_len = (size_t)num_samples;
    size_t to_copy = raw_len < max_features ? raw_len : max_features;
    size_t feature_count = 0;

    for (size_t i = 0; i < to_copy; i++, feature_count++) {
        float v = data[i];
        if (v < -1.0f) v = -1.0f;
        if (v > 1.0f) v = 1.0f;
        time_features[feature_count] = v;
    }

    /* === 2. 过零率（ZCR）=== */
    if (feature_count + 1 <= max_features) {
        float zcr = 0.0f;
        int zcr_count = 0;
        for (int i = 1; i < num_samples; i++) {
            if ((data[i] >= 0.0f && data[i-1] < 0.0f) ||
                (data[i] < 0.0f && data[i-1] >= 0.0f)) {
                zcr_count++;
            }
        }
        zcr = (float)zcr_count / (float)(num_samples > 1 ? num_samples - 1 : 1);
        time_features[feature_count++] = zcr;
    }

    /* === 3. 短时能量（STE）—— 分帧计算 === */
    if (feature_count + 1 <= max_features) {
        int frame_size = num_samples < 256 ? num_samples : 256;
        int num_frames = num_samples / frame_size;
        float total_ste = 0.0f;
        if (num_frames > 0) {
            float* frame_energies = (float*)safe_malloc((size_t)num_frames * sizeof(float));
            if (frame_energies) {
                for (int f = 0; f < num_frames; f++) {
                    float e = 0.0f;
                    for (int s = 0; s < frame_size; s++) {
                        int idx = f * frame_size + s;
                        if (idx < num_samples) e += data[idx] * data[idx];
                    }
                    frame_energies[f] = e / (float)frame_size;
                }
                /* 取平均能量 */
                for (int f = 0; f < num_frames; f++) total_ste += frame_energies[f];
                total_ste /= (float)num_frames;
                safe_free((void**)&frame_energies);
            }
        }
        time_features[feature_count++] = total_ste;
    }

    /* === 4. 自相关基频估计 === */
    if (feature_count + 1 <= max_features) {
        float pitch = 0.0f;
        int min_lag = num_samples / 500 + 1;   /* 最低~50Hz  @16000Hz => 320样本 */
        int max_lag = num_samples / 50 + 1;    /* 最高~500Hz @16000Hz => 32样本 */
        if (min_lag < 2) min_lag = 2;
        if (max_lag >= num_samples) max_lag = num_samples - 1;
        if (max_lag > min_lag) {
            float best_corr = 0.0f;
            int best_lag = 0;
            float energy = 0.0f;
            for (int i = 0; i < num_samples; i++) energy += data[i] * data[i];
            if (energy > 1e-6f) {
                for (int lag = min_lag; lag <= max_lag; lag++) {
                    float corr = 0.0f;
                    for (int i = 0; i < num_samples - lag; i++) {
                        corr += data[i] * data[i + lag];
                    }
                    corr /= energy;
                    if (corr > best_corr) {
                        best_corr = corr;
                        best_lag = lag;
                    }
                }
            }
            if (best_lag > 0 && best_corr > 0.3f) {
                int sample_rate = processor->config.target_sample_rate > 0 ?
                                  processor->config.target_sample_rate : 16000;
                pitch = (float)sample_rate / (float)best_lag;
            }
        }
        time_features[feature_count++] = pitch;
    }

    /* === 5. 统计特征 === */
    if (feature_count + 6 <= max_features) {
        double sum = 0.0, sum_sq = 0.0, sum_cu = 0.0, sum_qu = 0.0;
        float vmin = data[0], vmax = data[0];
        for (int i = 0; i < num_samples; i++) {
            double v = (double)data[i];
            sum += v;
            sum_sq += v * v;
            sum_cu += v * v * v;
            sum_qu += v * v * v * v;
            if (data[i] < vmin) vmin = data[i];
            if (data[i] > vmax) vmax = data[i];
        }
        double n = (double)num_samples;
        double mean = sum / n;
        double variance = (sum_sq / n) - (mean * mean);
        double stddev = variance > 0.0 ? sqrt(variance) : 0.0;
        double skewness = 0.0;
        double kurtosis = 0.0;
        if (stddev > 1e-10) {
            skewness = (sum_cu / n - 3.0 * mean * (sum_sq / n) + 2.0 * mean * mean * mean) /
                       (stddev * stddev * stddev);
            kurtosis = (sum_qu / n - 4.0 * mean * (sum_cu / n) +
                        6.0 * mean * mean * (sum_sq / n) - 3.0 * mean * mean * mean * mean) /
                       (stddev * stddev * stddev * stddev) - 3.0;
        }
        float rms = (float)sqrt(sum_sq / n);
        float dynamic_range = vmax - vmin;
        if (dynamic_range < 0.0f) dynamic_range = 0.0f;

        time_features[feature_count++] = (float)mean;
        time_features[feature_count++] = (float)variance;
        time_features[feature_count++] = (float)skewness;
        time_features[feature_count++] = (float)kurtosis;
        time_features[feature_count++] = dynamic_range;
        time_features[feature_count++] = rms;
    }

    return (int)feature_count;
}

/**
 * @brief 获取音频处理器配置
 */
int audio_processor_get_config(const AudioProcessor* processor, AudioConfig* config) {
    if (!processor || !config) {
        return -1;
    }

    *config = processor->config;
    return 0;
}

/**
 * @brief 设置音频处理器配置
 */
int audio_processor_set_config(AudioProcessor* processor, const AudioConfig* config) {
    if (!processor || !config) {
        return -1;
    }

    processor->config = *config;
    return 0;
}

/**
 * @brief 重置音频处理器
 */
void audio_processor_reset(AudioProcessor* processor) {
    if (!processor) {
        return;
    }

    safe_free((void**)&processor->sample_buffer);
    processor->sample_buffer = NULL;
    processor->buffer_size = 0;
}

/**
 * @brief 辅助函数：复制字符串
 */
static char* audio_strdup(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src) + 1;
    char* dst = (char*)safe_malloc(len);
    if (dst) {
        memcpy(dst, src, len);
    }
    return dst;
}

/**
 * @brief 填充音频设备信息结构体
 */
static void audio_fill_device_info(AudioDeviceInfo* info, const char* id, const char* name,
                                   AudioDeviceType type, int in_ch, int out_ch,
                                   const int* sample_rates, int is_default, int is_physical) {
    memset(info, 0, sizeof(AudioDeviceInfo));
    info->device_id = audio_strdup(id ? id : "");
    info->device_name = audio_strdup(name ? name : "未知设备");
    info->device_type = type;
    info->max_input_channels = in_ch;
    info->max_output_channels = out_ch;
    if (sample_rates) {
        int idx;
        for (idx = 0; idx < 15 && sample_rates[idx] != 0; idx++) {
            info->supported_sample_rates[idx] = sample_rates[idx];
        }
        info->supported_sample_rates[idx] = 0;
    }
    info->is_default = is_default;
    info->is_physical = is_physical;
}

#if defined(_WIN32) || defined(_WIN64)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef COBJMACROS
#define COBJMACROS
#endif
#include <windows.h>
#include <mmdeviceapi.h>
#include <endpointvolume.h>
#include <audioclient.h>
#include <devicetopology.h>
#include <functiondiscoverykeys_devpkey.h>

/* WASAPI GUID本地静态定义（避免链接uuid.lib） */
static const CLSID c_CLSID_MMDeviceEnumerator =
    {0xBCDE0395, 0xE52F, 0x467C, {0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}};
static const IID c_IID_IMMDeviceEnumerator =
    {0xA95664D2, 0x9614, 0x4F35, {0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6}};
static const IID c_IID_IAudioClient =
    {0x1CB9AD4C, 0xDBFA, 0x4C32, {0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2}};
static const IID c_IID_IAudioCaptureClient =
    {0xC8ADBD64, 0xE71E, 0x48A0, {0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17}};
static const GUID c_KSDATAFORMAT_SUBTYPE_PCM =
    {0x00000001, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};
static const GUID c_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT =
    {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};

#ifndef EXCLUDED_FROM_AUDIO_ENUM

/* {0.0.0.0, 0.0.0.0, {0,0,0,0,0,0,0,0,0,0,0}} */
static const GUID GUID_NULL_DEVICE = {0x00000000, 0x0000, 0x0000, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}};

/**
 * @brief 枚举Windows音频设备
 */
int audio_enum_devices(AudioDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return -1;

    HRESULT hr;
    IMMDeviceEnumerator* enumerator = NULL;
    IMMDeviceCollection* collection = NULL;
    IMMDevice* default_device = NULL;
    IPropertyStore* props = NULL;
    LPWSTR def_id = NULL;

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) return -1;

    hr = CoCreateInstance(&c_CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &c_IID_IMMDeviceEnumerator, (void**)&enumerator);
    if (FAILED(hr) || !enumerator) { CoUninitialize(); return -1; }

    hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(enumerator, eRender, eConsole, &default_device);
    if (SUCCEEDED(hr) && default_device) {
        IMMDevice_GetId(default_device, &def_id);
    }

    int count = 0;
    /* 枚举输入设备 */
    hr = IMMDeviceEnumerator_EnumAudioEndpoints(enumerator, eCapture, DEVICE_STATE_ACTIVE, &collection);
    if (SUCCEEDED(hr) && collection) {
        UINT num = 0;
        IMMDeviceCollection_GetCount(collection, &num);
        for (UINT i = 0; i < num && count < max_devices; i++) {
            IMMDevice* device = NULL;
            IMMDeviceCollection_Item(collection, i, &device);
            if (!device) continue;

            LPWSTR dev_id = NULL;
            IMMDevice_GetId(device, &dev_id);
            props = NULL;
            IMMDevice_OpenPropertyStore(device, STGM_READ, &props);

            int is_default = 0;
            PROPVARIANT varName;
            PropVariantInit(&varName);
            char name_buf[256] = "未知设备";
            if (props) {
                IPropertyStore_GetValue(props, &PKEY_Device_FriendlyName, &varName);
                if (varName.vt == VT_LPWSTR && varName.pwszVal) {
                    WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, name_buf, sizeof(name_buf), NULL, NULL);
                }
                if (dev_id && def_id && wcscmp(dev_id, def_id) == 0) is_default = 1;
                PropVariantClear(&varName);
                IPropertyStore_Release(props); props = NULL;
            }

            /* 填充设备信息 */
            char id_buf[128];
            if (dev_id) {
                WideCharToMultiByte(CP_UTF8, 0, dev_id, -1, id_buf, sizeof(id_buf), NULL, NULL);
            } else {
                snprintf(id_buf, sizeof(id_buf), "input_%d", count);
            }

            int sr_list[] = {8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000, 0};
            audio_fill_device_info(&devices[count], id_buf, name_buf,
                                   AUDIO_DEVICE_INPUT, 2, 0, sr_list, is_default, 1);
            count++;

            if (dev_id) CoTaskMemFree(dev_id);
            IMMDevice_Release(device);
        }
        IMMDeviceCollection_Release(collection); collection = NULL;
    }

    /* 枚举输出设备 */
    hr = IMMDeviceEnumerator_EnumAudioEndpoints(enumerator, eRender, DEVICE_STATE_ACTIVE, &collection);
    if (SUCCEEDED(hr) && collection) {
        UINT num = 0;
        IMMDeviceCollection_GetCount(collection, &num);
        for (UINT i = 0; i < num && count < max_devices; i++) {
            IMMDevice* device = NULL;
            IMMDeviceCollection_Item(collection, i, &device);
            if (!device) continue;

            LPWSTR dev_id = NULL;
            IMMDevice_GetId(device, &dev_id);
            props = NULL;
            IMMDevice_OpenPropertyStore(device, STGM_READ, &props);

            int is_default = 0;
            PROPVARIANT varName;
            PropVariantInit(&varName);
            char name_buf[256] = "未知设备";
            if (props) {
                IPropertyStore_GetValue(props, &PKEY_Device_FriendlyName, &varName);
                if (varName.vt == VT_LPWSTR && varName.pwszVal) {
                    WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, name_buf, sizeof(name_buf), NULL, NULL);
                }
                if (dev_id && def_id && wcscmp(dev_id, def_id) == 0) is_default = 1;
                PropVariantClear(&varName);
                IPropertyStore_Release(props); props = NULL;
            }

            char id_buf[128];
            if (dev_id) {
                WideCharToMultiByte(CP_UTF8, 0, dev_id, -1, id_buf, sizeof(id_buf), NULL, NULL);
            } else {
                snprintf(id_buf, sizeof(id_buf), "output_%d", count);
            }

            int sr_list[] = {8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000, 0};
            audio_fill_device_info(&devices[count], id_buf, name_buf,
                                   AUDIO_DEVICE_OUTPUT, 0, 2, sr_list, is_default, 1);
            count++;

            if (dev_id) CoTaskMemFree(dev_id);
            IMMDevice_Release(device);
        }
        IMMDeviceCollection_Release(collection);
    }

    if (default_device) { IMMDevice_Release(default_device); }
    if (enumerator) { IMMDeviceEnumerator_Release(enumerator); }
    if (def_id) CoTaskMemFree(def_id);
    CoUninitialize();

    return count;
}

#else
int audio_enum_devices(AudioDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return -1;
    int sr_list[] = {8000, 16000, 44100, 48000, 0};
    audio_fill_device_info(&devices[0], "default_input", "默认输入设备",
                           AUDIO_DEVICE_INPUT, 2, 0, sr_list, 1, 0);
    audio_fill_device_info(&devices[1], "default_output", "默认输出设备",
                           AUDIO_DEVICE_OUTPUT, 0, 2, sr_list, 1, 0);
    return 2;
}
#endif

#elif defined(__APPLE__)
/* macOS音频设备枚举（通过CoreAudio） */
#include <CoreAudio/CoreAudio.h>
#include <AudioToolbox/AudioToolbox.h>

int audio_enum_devices(AudioDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return -1;

    AudioObjectPropertyAddress propAddr = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    UInt32 dataSize = 0;
    AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propAddr, 0, NULL, &dataSize);
    if (dataSize <= 0) return -1;

    int deviceCount = dataSize / sizeof(AudioDeviceID);
    AudioDeviceID* audioDevices = (AudioDeviceID*)safe_malloc(dataSize);
    if (!audioDevices) return -1;

    AudioObjectGetPropertyData(kAudioObjectSystemObject, &propAddr, 0, NULL, &dataSize, audioDevices);

    AudioDeviceID defaultInput = kAudioDeviceUnknown;
    AudioDeviceID defaultOutput = kAudioDeviceUnknown;
    UInt32 defaultSize = sizeof(AudioDeviceID);

    propAddr.mSelector = kAudioHardwarePropertyDefaultInputDevice;
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &propAddr, 0, NULL, &defaultSize, &defaultInput);

    propAddr.mSelector = kAudioHardwarePropertyDefaultOutputDevice;
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &propAddr, 0, NULL, &defaultSize, &defaultOutput);

    int count = 0;
    for (int i = 0; i < deviceCount && count < max_devices; i++) {
        AudioDeviceID devID = audioDevices[i];

        /* 获取设备名称 */
        CFStringRef deviceName = NULL;
        dataSize = sizeof(CFStringRef);
        propAddr.mSelector = kAudioDevicePropertyDeviceNameCFString;
        AudioObjectGetPropertyData(devID, &propAddr, 0, NULL, &dataSize, &deviceName);

        char nameBuf[256] = "未知设备";
        if (deviceName) {
            CFStringGetCString(deviceName, nameBuf, sizeof(nameBuf), kCFStringEncodingUTF8);
            CFRelease(deviceName);
        }

        /* 判断是输入还是输出 */
        UInt32 safetyOffset;
        propAddr.mSelector = kAudioDevicePropertySafetyOffset;
        propAddr.mScope = kAudioDevicePropertyScopeInput;
        dataSize = sizeof(UInt32);
        OSStatus inputErr = AudioObjectGetPropertyData(devID, &propAddr, 0, NULL, &dataSize, &safetyOffset);

        propAddr.mScope = kAudioDevicePropertyScopeOutput;
        OSStatus outputErr = AudioObjectGetPropertyData(devID, &propAddr, 0, NULL, &dataSize, &safetyOffset);

        int isInput = (inputErr == noErr);
        int isOutput = (outputErr == noErr);

        if (!isInput && !isOutput) continue;

        char idBuf[64];
        snprintf(idBuf, sizeof(idBuf), "%u", (unsigned int)devID);

        int sr_list[] = {8000, 16000, 22050, 44100, 48000, 96000, 0};
        if (isInput) {
            audio_fill_device_info(&devices[count], idBuf, nameBuf,
                                   AUDIO_DEVICE_INPUT, 2, 0, sr_list,
                                   (devID == defaultInput) ? 1 : 0, 1);
            count++;
        }
        if (isOutput && count < max_devices) {
            audio_fill_device_info(&devices[count], idBuf, nameBuf,
                                   AUDIO_DEVICE_OUTPUT, 0, 2, sr_list,
                                   (devID == defaultOutput) ? 1 : 0, 1);
            count++;
        }
    }

    safe_free((void**)&audioDevices);
    return count;
}

#else
/* Linux音频设备枚举（通过ALSA） */
#include <alsa/asoundlib.h>

int audio_enum_devices(AudioDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return -1;

    char** hints;
    int count = 0;

    if (snd_device_name_hint(-1, "pcm", (void***)&hints) < 0) return -1;

    for (int i = 0; hints[i] && count < max_devices; i++) {
        char* name = snd_device_name_get_hint(hints[i], "NAME");
        char* desc = snd_device_name_get_hint(hints[i], "DESC");
        char* ioid = snd_device_name_get_hint(hints[i], "IOID");

        if (name && desc) {
            /* 判断输入输出方向 */
            int isInput = 0, isOutput = 0;
            if (!ioid || strcmp(ioid, "Input") == 0) isOutput = 1;
            if (!ioid || strcmp(ioid, "Output") == 0) isInput = 1;

            /* 过滤掉一些系统设备 */
            if (strstr(name, "surround") || strstr(name, "iec958") ||
                strstr(name, "hdmi") || strstr(name, "modem")) {
                free(name); free(desc); if (ioid) free(ioid);
                continue;
            }

            /* 提取描述的第一行 */
            char descBuf[256];
            const char* nl = strchr(desc, '\n');
            size_t descLen = nl ? (size_t)(nl - desc) : strlen(desc);
            if (descLen >= sizeof(descBuf)) descLen = sizeof(descBuf) - 1;
            memcpy(descBuf, desc, descLen);
            descBuf[descLen] = '\0';

            int sr_list[] = {8000, 16000, 22050, 44100, 48000, 96000, 0};
            if (isOutput) {
                audio_fill_device_info(&devices[count], name, descBuf,
                                       AUDIO_DEVICE_OUTPUT, 0, 2, sr_list,
                                       (count == 0) ? 1 : 0, 1);
                count++;
            }
            if (isInput && count < max_devices) {
                audio_fill_device_info(&devices[count], name, descBuf,
                                       AUDIO_DEVICE_INPUT, 2, 0, sr_list, 0, 1);
                count++;
            }
        }

        if (name) free(name);
        if (desc) free(desc);
        if (ioid) free(ioid);
    }

    snd_device_name_free_hint((void**)hints);
    return count;
}
#endif

/**
 * @brief 获取指定类型的音频设备数量
 */
int audio_get_device_count(AudioDeviceType type) {
    int max_check = 64;
    AudioDeviceInfo* devs = (AudioDeviceInfo*)safe_malloc(max_check * sizeof(AudioDeviceInfo));
    if (!devs) return -1;

    int total = audio_enum_devices(devs, max_check);
    int count = 0;
    for (int i = 0; i < total; i++) {
        if (devs[i].device_type == type) count++;
    }

    for (int i = 0; i < total; i++) {
        safe_free((void**)&devs[i].device_id);
        safe_free((void**)&devs[i].device_name);
    }
    safe_free((void**)&devs);

    return count;
}

/**
 * @brief 获取默认音频设备
 */
AudioDeviceInfo* audio_get_default_device(AudioDeviceType type) {
    int max_check = 64;
    AudioDeviceInfo* devs = (AudioDeviceInfo*)safe_malloc(max_check * sizeof(AudioDeviceInfo));
    if (!devs) return NULL;

    int total = audio_enum_devices(devs, max_check);
    AudioDeviceInfo* result = NULL;

    for (int i = 0; i < total; i++) {
        if (devs[i].device_type == type && devs[i].is_default) {
            result = (AudioDeviceInfo*)safe_malloc(sizeof(AudioDeviceInfo));
            if (result) {
                *result = devs[i];
                /* 复制字符串所有权转移 */
                devs[i].device_id = NULL;
                devs[i].device_name = NULL;
            }
            break;
        }
    }

    for (int i = 0; i < total; i++) {
        safe_free((void**)&devs[i].device_id);
        safe_free((void**)&devs[i].device_name);
    }
    safe_free((void**)&devs);

    return result;
}

/**
 * @brief 释放音频设备列表
 */
void audio_free_device_list(AudioDeviceInfo* devices, int count) {
    if (!devices) return;
    for (int i = 0; i < count; i++) {
        safe_free((void**)&devices[i].device_id);
        safe_free((void**)&devices[i].device_name);
    }
    safe_free((void**)&devices);
}

/**
 * @brief 释放单个音频设备信息
 */
void audio_free_device_info(AudioDeviceInfo* device) {
    if (!device) return;
    safe_free((void**)&device->device_id);
    safe_free((void**)&device->device_name);
    safe_free((void**)&device);
}

/* ===================================================================
 * 增强功能：WAV解码器、VAD语音活动检测
 * =================================================================== */

/**
 * @brief WAV文件头结构（仅用于内部解析）
 */
typedef struct {
    char riff_id[4];
    unsigned int riff_size;
    char wave_id[4];
    char chunk_id[4];
    unsigned int chunk_size;
} WavChunkHeader;

/**
 * @brief WAV格式块结构
 */
typedef struct {
    unsigned short audio_format;
    unsigned short num_channels;
    unsigned int sample_rate;
    unsigned int byte_rate;
    unsigned short block_align;
    unsigned short bits_per_sample;
} WavFormatChunk;

/**
 * @brief 解码WAV音频文件为PCM浮点样本
 */
int audio_decode_wav(const unsigned char* wav_data, size_t data_size,
                     float** samples, int* sample_rate,
                     int* num_samples, int* num_channels) {
    if (!wav_data || !samples || !sample_rate || !num_samples || !num_channels) {
        return -1;
    }

    if (data_size < 44) {
        return -1;
    }

    /* 验证RIFF/WAVE签名 */
    if (memcmp(wav_data, "RIFF", 4) != 0 || memcmp(wav_data + 8, "WAVE", 4) != 0) {
        return -1;
    }

    /* 遍历块查找"fmt "和"data" */
    WavFormatChunk fmt;
    memset(&fmt, 0, sizeof(WavFormatChunk));
    int fmt_found = 0;
    const unsigned char* data_chunk = NULL;
    unsigned int data_chunk_size = 0;

    unsigned int offset = 12;
    while (offset + 8 <= data_size) {
        const WavChunkHeader* chunk = (const WavChunkHeader*)(wav_data + offset);
        unsigned int chunk_data_size = chunk->chunk_size;
        if (offset + 8 + chunk_data_size > data_size) {
            break;
        }

        if (memcmp(chunk->chunk_id, "fmt ", 4) == 0) {
            if (chunk_data_size >= 16) {
                const unsigned char* fmt_data = wav_data + offset + 8;
                fmt.audio_format = (unsigned short)fmt_data[0] | ((unsigned short)fmt_data[1] << 8);
                fmt.num_channels = (unsigned short)fmt_data[2] | ((unsigned short)fmt_data[3] << 8);
                fmt.sample_rate = (unsigned int)fmt_data[4] | ((unsigned int)fmt_data[5] << 8) |
                                  ((unsigned int)fmt_data[6] << 16) | ((unsigned int)fmt_data[7] << 24);
                fmt.byte_rate = (unsigned int)fmt_data[8] | ((unsigned int)fmt_data[9] << 8) |
                                ((unsigned int)fmt_data[10] << 16) | ((unsigned int)fmt_data[11] << 24);
                fmt.block_align = (unsigned short)fmt_data[12] | ((unsigned short)fmt_data[13] << 8);
                fmt.bits_per_sample = (unsigned short)fmt_data[14] | ((unsigned short)fmt_data[15] << 8);
                fmt_found = 1;
            }
        } else if (memcmp(chunk->chunk_id, "data", 4) == 0) {
            data_chunk = wav_data + offset + 8;
            data_chunk_size = chunk_data_size;
        }

        offset += 8 + chunk_data_size;
        if (chunk_data_size & 1) {
            offset++;
        }
    }

    if (!fmt_found || !data_chunk || data_chunk_size == 0) {
        return -1;
    }

    if (fmt.audio_format != 1) {
        return -1;
    }

    *sample_rate = (int)fmt.sample_rate;
    *num_channels = (int)fmt.num_channels;

    int bytes_per_sample = fmt.bits_per_sample / 8;
    if (bytes_per_sample < 1 || bytes_per_sample > 4) {
        return -1;
    }

    int total_samples = (int)(data_chunk_size / bytes_per_sample);
    if (total_samples <= 0) {
        return -1;
    }

    int mono_samples = (fmt.num_channels > 1) ? (total_samples / (int)fmt.num_channels) : total_samples;
    float* output = (float*)safe_malloc(mono_samples * sizeof(float));
    if (!output) {
        return -1;
    }

    int write_idx = 0;
    for (int i = 0; i + fmt.num_channels * bytes_per_sample <= (int)data_chunk_size;
         i += fmt.num_channels * bytes_per_sample) {
        float sum = 0.0f;
        for (unsigned short ch = 0; ch < fmt.num_channels; ch++) {
            int byte_offset = i + ch * bytes_per_sample;
            int32_t raw = 0;

            switch (bytes_per_sample) {
                case 1:
                    raw = (int32_t)data_chunk[byte_offset] - 128;
                    break;
                case 2:
                    raw = (int32_t)(int16_t)(data_chunk[byte_offset] | (data_chunk[byte_offset + 1] << 8));
                    break;
                case 3: {
                    int32_t val = (int32_t)data_chunk[byte_offset] |
                                  ((int32_t)data_chunk[byte_offset + 1] << 8) |
                                  ((int32_t)data_chunk[byte_offset + 2] << 16);
                    if (val & 0x800000) {
                        val |= 0xFF000000;
                    }
                    raw = val;
                    break;
                }
                case 4:
                    raw = (int32_t)data_chunk[byte_offset] |
                          ((int32_t)data_chunk[byte_offset + 1] << 8) |
                          ((int32_t)data_chunk[byte_offset + 2] << 16) |
                          ((int32_t)data_chunk[byte_offset + 3] << 24);
                    break;
                default:
                    break;
            }

            sum += (float)raw;
        }

        float max_val = (float)((1 << (fmt.bits_per_sample - 1)) - 1);
        if (max_val < 1.0f) max_val = 1.0f;
        output[write_idx++] = (sum / (float)fmt.num_channels) / max_val;
    }

    *samples = output;
    *num_samples = write_idx;
    return 0;
}

/**
 * @brief 语音活动检测（VAD）- 短时能量+过零率双门限法
 */
int audio_vad_detect(const float* samples, int num_samples, int sample_rate,
                     float* frame_energies, int* voice_flags,
                     int frame_size, float energy_threshold, float zcr_threshold) {
    if (!samples || !voice_flags || num_samples <= 0 || sample_rate <= 0) {
        return -1;
    }

    if (frame_size <= 0) {
        frame_size = (int)(0.02f * sample_rate + 0.5f);
    }
    if (frame_size < 8) frame_size = 8;

    int num_frames = num_samples / frame_size;
    if (num_frames < 1) {
        voice_flags[0] = 0;
        return 1;
    }

    float* energies = (float*)safe_malloc(num_frames * sizeof(float));
    float* zcrs = (float*)safe_malloc(num_frames * sizeof(float));
    if (!energies || !zcrs) {
        safe_free((void**)&energies);
        safe_free((void**)&zcrs);
        return -1;
    }

    float max_energy = 0.0f;
    float avg_zcr = 0.0f;
    for (int f = 0; f < num_frames; f++) {
        float energy = 0.0f;
        int zcr = 0;
        int start = f * frame_size;

        for (int i = 0; i < frame_size; i++) {
            float val = samples[start + i];
            energy += val * val;
            if (i > 0) {
                if ((samples[start + i - 1] >= 0.0f && samples[start + i] < 0.0f) ||
                    (samples[start + i - 1] < 0.0f && samples[start + i] >= 0.0f)) {
                    zcr++;
                }
            }
        }

        energy /= (float)frame_size;
        energies[f] = energy;
        zcrs[f] = (float)zcr / (float)(frame_size - 1);
        if (energy > max_energy) max_energy = energy;
        avg_zcr += zcrs[f];
    }
    avg_zcr /= (float)num_frames;

    if (energy_threshold <= 0.0f) {
        energy_threshold = max_energy * 0.1f;
        if (energy_threshold < 1e-8f) energy_threshold = 1e-8f;
    }
    if (zcr_threshold <= 0.0f) {
        zcr_threshold = avg_zcr * 1.5f;
        if (zcr_threshold < 0.01f) zcr_threshold = 0.01f;
    }

    for (int f = 0; f < num_frames; f++) {
        voice_flags[f] = (energies[f] >= energy_threshold && zcrs[f] <= zcr_threshold) ? 1 : 0;
    }

    if (frame_energies) {
        memcpy(frame_energies, energies, num_frames * sizeof(float));
    }

    safe_free((void**)&energies);
    safe_free((void**)&zcrs);
    return num_frames;
}

/**
 * @brief 多采样率重采样
 */
int audio_resample(const float* input, int input_len, int input_sr, int output_sr, float* output, int* output_len) {
    if (!input || !output || !output_len || input_sr <= 0 || output_sr <= 0) return -1;
    float ratio = (float)output_sr / (float)input_sr;
    int out_len = (int)((float)input_len * ratio);
    if (out_len > *output_len) out_len = *output_len;
    for (int i = 0; i < out_len; i++) {
        float src_idx = (float)i / ratio;
        int si = (int)src_idx;
        float frac = src_idx - (float)si;
        float v0 = (si >= 0 && si < input_len) ? input[si] : 0.0f;
        float v1 = (si+1 < input_len) ? input[si+1] : v0;
        output[i] = v0 * (1.0f - frac) + v1 * frac;
    }
    *output_len = out_len;
    return 0;
}

/* ============================================================================
 * 真实音频硬件采集接口（跨平台）
 *
 * 所有音频时域原始信号直接送入主CfC统一动态系统处理。
 * 不使用FFT/MFCC等传统频域方法。
 * ============================================================================ */
/* ============================================================================ *
 * 音频数据捕获 - 平台相关实现
 * 支持 Windows (WASAPI), macOS (CoreAudio), Linux (ALSA)
 * ============================================================================ */

#if defined(_WIN32) || defined(_WIN64)
/* ==================== Windows WASAPI 捕获 ==================== */

struct AudioCaptureContext {
    IMMDeviceEnumerator* enumerator;
    IMMDevice* device;
    IAudioClient* audioClient;
    IAudioCaptureClient* captureClient;
    WAVEFORMATEX* format;
    int is_capturing;
    int sample_rate;
    int channels;
    int bits_per_sample;
    HANDLE capture_event;
    HANDLE capture_thread;
    int thread_running;
    void (*data_callback)(const float*, size_t, void*);
    void* user_data;
};

static DWORD WINAPI audio_capture_thread_func(LPVOID arg) {
    AudioCaptureContext* ctx = (AudioCaptureContext*)arg;
    if (!ctx) return 1;

    HANDLE waitArray[2];
    waitArray[0] = ctx->capture_event;
    waitArray[1] = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (!waitArray[1]) waitArray[1] = ctx->capture_event;

    while (ctx->thread_running) {
        DWORD waitResult = WaitForMultipleObjects(2, waitArray, FALSE, 100);
        if (!ctx->thread_running) break;

        if (waitResult == WAIT_OBJECT_0) {
            UINT32 packetLength = 0;
            HRESULT hr = IAudioCaptureClient_GetNextPacketSize(ctx->captureClient, &packetLength);
            if (FAILED(hr) || packetLength == 0) continue;

            BYTE* data = NULL;
            UINT32 framesAvailable = 0;
            DWORD flags = 0;

            hr = IAudioCaptureClient_GetBuffer(ctx->captureClient, &data, &framesAvailable, &flags, NULL, NULL);
            if (SUCCEEDED(hr) && data && framesAvailable > 0) {
                int num_samples = framesAvailable * ctx->channels;
                float* float_data = (float*)safe_malloc(num_samples * sizeof(float));
                if (float_data) {
                    if (ctx->bits_per_sample == 16) {
                        short* src = (short*)data;
                        for (int i = 0; i < num_samples; i++) {
                            float_data[i] = src[i] / 32768.0f;
                        }
                    } else if (ctx->bits_per_sample == 32) {
                        float* src = (float*)data;
                        for (int i = 0; i < num_samples; i++) {
                            float_data[i] = src[i];
                        }
                    } else {
                        for (int i = 0; i < num_samples; i++) {
                            float_data[i] = ((short*)data)[i] / 32768.0f;
                        }
                    }

                    if (ctx->data_callback) {
                        ctx->data_callback(float_data, num_samples, ctx->user_data);
                    }
                    safe_free((void**)&float_data);
                }
                IAudioCaptureClient_ReleaseBuffer(ctx->captureClient, framesAvailable);
            }
        }
    }

    if (waitArray[1] && waitArray[1] != ctx->capture_event) CloseHandle(waitArray[1]);
    return 0;
}

static int audio_capture_enumerate_devices(AudioCaptureDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return 0;

    HRESULT hr;
    IMMDeviceEnumerator* enumerator = NULL;
    IMMDeviceCollection* collection = NULL;

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) return 0;

    hr = CoCreateInstance(&c_CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &c_IID_IMMDeviceEnumerator, (void**)&enumerator);
    if (FAILED(hr) || !enumerator) { CoUninitialize(); return 0; }

    hr = IMMDeviceEnumerator_EnumAudioEndpoints(enumerator, eCapture, DEVICE_STATE_ACTIVE, &collection);
    if (FAILED(hr) || !collection) { IMMDeviceEnumerator_Release(enumerator); CoUninitialize(); return 0; }

    UINT count = 0;
    IMMDeviceCollection_GetCount(collection, &count);
    int actual = 0;

    for (UINT i = 0; i < count && actual < max_devices; i++) {
        IMMDevice* device = NULL;
        IMMDeviceCollection_Item(collection, i, &device);
        if (!device) continue;

        IPropertyStore* props = NULL;
        IMMDevice_OpenPropertyStore(device, STGM_READ, &props);

        AudioCaptureDeviceInfo* info = &devices[actual];
        memset(info, 0, sizeof(AudioCaptureDeviceInfo));

        if (props) {
            PROPVARIANT varName;
            PropVariantInit(&varName);
            hr = IPropertyStore_GetValue(props, &PKEY_Device_FriendlyName, &varName);
            if (SUCCEEDED(hr) && varName.vt == VT_LPWSTR && varName.pwszVal) {
                WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1,
                                    info->name, sizeof(info->name), NULL, NULL);
            } else {
                snprintf(info->name, sizeof(info->name), "麦克风 %d", actual + 1);
            }
            PropVariantClear(&varName);
            IPropertyStore_Release(props);
        } else {
            snprintf(info->name, sizeof(info->name), "麦克风 %d", actual + 1);
        }

        LPWSTR devId = NULL;
        IMMDevice_GetId(device, &devId);
        if (devId) {
            WideCharToMultiByte(CP_UTF8, 0, devId, -1,
                                info->device_id, sizeof(info->device_id), NULL, NULL);
            CoTaskMemFree(devId);
        }

        info->max_channels = 2;
        info->default_sample_rate = 48000;
        actual++;

        IMMDevice_Release(device);
    }

    IMMDeviceCollection_Release(collection);
    IMMDeviceEnumerator_Release(enumerator);
    CoUninitialize();
    return actual;
}

static AudioCaptureContext* audio_capture_create(const char* device_id,
                                           int sample_rate,
                                           int channels,
                                           int bits_per_sample) {
    HRESULT hr;
    AudioCaptureContext* ctx = (AudioCaptureContext*)safe_malloc(sizeof(AudioCaptureContext));
    if (!ctx) return NULL;
    memset(ctx, 0, sizeof(AudioCaptureContext));

    ctx->sample_rate = (sample_rate > 0) ? sample_rate : 48000;
    ctx->channels = (channels > 0) ? channels : 1;
    ctx->bits_per_sample = (bits_per_sample > 0) ? bits_per_sample : 16;
    ctx->is_capturing = 0;
    ctx->thread_running = 0;

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) { safe_free((void**)&ctx); return NULL; }

    hr = CoCreateInstance(&c_CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &c_IID_IMMDeviceEnumerator, (void**)&ctx->enumerator);
    if (FAILED(hr)) { CoUninitialize(); safe_free((void**)&ctx); return NULL; }

    if (device_id && device_id[0]) {
        /* 按设备ID查找 */
        IMMDeviceCollection* collection = NULL;
        hr = IMMDeviceEnumerator_EnumAudioEndpoints(ctx->enumerator, eCapture,
                                                     DEVICE_STATE_ACTIVE, &collection);
        if (SUCCEEDED(hr) && collection) {
            UINT count = 0;
            IMMDeviceCollection_GetCount(collection, &count);
            for (UINT i = 0; i < count && !ctx->device; i++) {
                LPWSTR devId = NULL;
                IMMDeviceCollection_Item(collection, i, &ctx->device);
                if (ctx->device) {
                    IMMDevice_GetId(ctx->device, &devId);
                    char idBuf[256];
                    if (devId) {
                        WideCharToMultiByte(CP_UTF8, 0, devId, -1, idBuf, sizeof(idBuf), NULL, NULL);
                        CoTaskMemFree(devId);
                        if (strcmp(idBuf, device_id) != 0) {
                            IMMDevice_Release(ctx->device);
                            ctx->device = NULL;
                        }
                    } else {
                        IMMDevice_Release(ctx->device);
                        ctx->device = NULL;
                    }
                }
            }
            IMMDeviceCollection_Release(collection);
        }
        if (!ctx->device) {
            IMMDeviceEnumerator_Release(ctx->enumerator);
            CoUninitialize(); safe_free((void**)&ctx); return NULL;
        }
    } else {
        hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(ctx->enumerator,
                                                          eCapture, eConsole, &ctx->device);
        if (FAILED(hr) || !ctx->device) {
            IMMDeviceEnumerator_Release(ctx->enumerator);
            CoUninitialize(); safe_free((void**)&ctx); return NULL;
        }
    }

    hr = IMMDevice_Activate(ctx->device, &c_IID_IAudioClient, CLSCTX_ALL, NULL,
                            (void**)&ctx->audioClient);
    if (FAILED(hr) || !ctx->audioClient) {
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        CoUninitialize(); safe_free((void**)&ctx); return NULL;
    }

    WAVEFORMATEXTENSIBLE wfx;
    memset(&wfx, 0, sizeof(wfx));
    wfx.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    wfx.Format.nChannels = (WORD)ctx->channels;
    wfx.Format.nSamplesPerSec = (DWORD)ctx->sample_rate;
    wfx.Format.wBitsPerSample = (WORD)ctx->bits_per_sample;
    wfx.Format.nBlockAlign = (WORD)((wfx.Format.nChannels * wfx.Format.wBitsPerSample) / 8);
    wfx.Format.nAvgBytesPerSec = wfx.Format.nSamplesPerSec * wfx.Format.nBlockAlign;
    wfx.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
    wfx.Samples.wValidBitsPerSample = (WORD)ctx->bits_per_sample;
    wfx.dwChannelMask = (ctx->channels == 1) ? SPEAKER_FRONT_CENTER : SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
    wfx.SubFormat = c_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;

    hr = IAudioClient_Initialize(ctx->audioClient, AUDCLNT_SHAREMODE_SHARED,
                                  AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                  10000000, 0, (WAVEFORMATEX*)&wfx, NULL);
    if (FAILED(hr)) {
        /* 尝试PCM格式 */
        wfx.SubFormat = c_KSDATAFORMAT_SUBTYPE_PCM;
        hr = IAudioClient_Initialize(ctx->audioClient, AUDCLNT_SHAREMODE_SHARED,
                                      AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                      10000000, 0, (WAVEFORMATEX*)&wfx, NULL);
    }

    if (FAILED(hr)) {
        IAudioClient_Release(ctx->audioClient);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        CoUninitialize(); safe_free((void**)&ctx); return NULL;
    }

    hr = IAudioClient_GetService(ctx->audioClient, &c_IID_IAudioCaptureClient,
                                  (void**)&ctx->captureClient);
    if (FAILED(hr) || !ctx->captureClient) {
        IAudioClient_Release(ctx->audioClient);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        CoUninitialize(); safe_free((void**)&ctx); return NULL;
    }

    ctx->capture_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!ctx->capture_event) {
        IAudioCaptureClient_Release(ctx->captureClient);
        IAudioClient_Release(ctx->audioClient);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        CoUninitialize(); safe_free((void**)&ctx); return NULL;
    }

    IAudioClient_SetEventHandle(ctx->audioClient, ctx->capture_event);

    return ctx;
}

static int audio_capture_start(AudioCaptureContext* ctx,
                         void (*callback)(const float*, size_t, void*),
                         void* user_data) {
    if (!ctx || !callback) return -1;
    ctx->data_callback = callback;
    ctx->user_data = user_data;

    HRESULT hr = IAudioClient_Start(ctx->audioClient);
    if (FAILED(hr)) return -1;

    ctx->thread_running = 1;
    ctx->capture_thread = CreateThread(NULL, 0, audio_capture_thread_func, ctx, 0, NULL);
    if (!ctx->capture_thread) {
        IAudioClient_Stop(ctx->audioClient);
        ctx->thread_running = 0;
        return -1;
    }

    ctx->is_capturing = 1;
    return 0;
}

static int audio_capture_process(AudioCaptureContext* ctx) {
    if (!ctx || !ctx->is_capturing) return 0;
    
    if (!ctx->captureClient || !ctx->data_callback) return 0;
    
    UINT32 packet_length = 0;
    HRESULT hr = IAudioCaptureClient_GetNextPacketSize(ctx->captureClient, &packet_length);
    if (FAILED(hr)) return 0;
    
    int frames_processed = 0;
    while (packet_length > 0) {
        BYTE* data = NULL;
        UINT32 num_frames = 0;
        DWORD flags = 0;
        
        hr = IAudioCaptureClient_GetBuffer(ctx->captureClient, &data, &num_frames, &flags, NULL, NULL);
        if (FAILED(hr)) break;
        
        if (data && num_frames > 0 && !(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
            float* float_data = (float*)safe_malloc(num_frames * ctx->channels * sizeof(float));
            if (float_data) {
                for (UINT32 i = 0; i < num_frames * ctx->channels; i++) {
                    if (ctx->bits_per_sample == 16) {
                        INT16 sample = ((INT16*)data)[i];
                        float_data[i] = (float)sample / 32768.0f;
                    } else if (ctx->bits_per_sample == 32) {
                        INT32 sample = ((INT32*)data)[i];
                        float_data[i] = (float)((double)sample / 2147483648.0);
                    } else {
                        float_data[i] = ((float*)data)[i];
                    }
                }
                ctx->data_callback(float_data, num_frames, ctx->user_data);
                safe_free((void**)&float_data);
                frames_processed += (int)num_frames;
            }
        }
        
        IAudioCaptureClient_ReleaseBuffer(ctx->captureClient, num_frames);
        hr = IAudioCaptureClient_GetNextPacketSize(ctx->captureClient, &packet_length);
        if (FAILED(hr)) break;
    }
    
    return frames_processed;
}

static int audio_capture_stop(AudioCaptureContext* ctx) {
    if (!ctx) return -1;
    if (ctx->is_capturing) {
        ctx->thread_running = 0;
        if (ctx->capture_thread) {
            WaitForSingleObject(ctx->capture_thread, 3000);
            CloseHandle(ctx->capture_thread);
            ctx->capture_thread = NULL;
        }
        IAudioClient_Stop(ctx->audioClient);
    }
    ctx->is_capturing = 0;
    return 0;
}

static void audio_capture_free(AudioCaptureContext* ctx) {
    if (!ctx) return;
    audio_capture_stop(ctx);
    if (ctx->capture_event) CloseHandle(ctx->capture_event);
    if (ctx->captureClient) IAudioCaptureClient_Release(ctx->captureClient);
    if (ctx->audioClient) IAudioClient_Release(ctx->audioClient);
    if (ctx->device) IMMDevice_Release(ctx->device);
    if (ctx->enumerator) IMMDeviceEnumerator_Release(ctx->enumerator);
    CoUninitialize();
    safe_free((void**)&ctx);
    SetLastError(0);
}

#elif defined(__APPLE__)
/* ==================== macOS CoreAudio 捕获 ==================== */
#include <CoreAudio/CoreAudio.h>
#include <AudioToolbox/AudioToolbox.h>
#include <pthread.h>

struct AudioCaptureContext {
    AudioUnit audioUnit;
    int is_capturing;
    int sample_rate;
    int channels;
    int bits_per_sample;
    int thread_running;
    pthread_t capture_thread;
    void (*data_callback)(const float*, size_t, void*);
    void* user_data;
};

static OSStatus audio_capture_render_cb(void* inRefCon,
                                         AudioUnitRenderActionFlags* ioActionFlags,
                                         const AudioTimeStamp* inTimeStamp,
                                         UInt32 inBusNumber,
                                         UInt32 inNumberFrames,
                                         AudioBufferList* ioData) {
    AudioCaptureContext* ctx = (AudioCaptureContext*)inRefCon;
    if (!ctx || !ctx->data_callback) return noErr;

    AudioBufferList bufferList;
    AudioBuffer buffer;
    buffer.mData = NULL;
    buffer.mDataByteSize = inNumberFrames * ctx->channels * sizeof(float);
    buffer.mNumberChannels = (UInt32)ctx->channels;
    bufferList.mNumberBuffers = 1;
    bufferList.mBuffers[0] = buffer;

    float* tempBuf = (float*)safe_malloc(buffer.mDataByteSize);
    if (!tempBuf) return noErr;
    bufferList.mBuffers[0].mData = tempBuf;

    OSStatus status = AudioUnitRender(ctx->audioUnit, ioActionFlags, inTimeStamp,
                                       inBusNumber, inNumberFrames, &bufferList);
    if (status == noErr && tempBuf) {
        ctx->data_callback(tempBuf, inNumberFrames * ctx->channels, ctx->user_data);
    }

    safe_free((void**)&tempBuf);
    return status;
}

static int audio_capture_enumerate_devices(AudioCaptureDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return 0;

    AudioObjectPropertyAddress propAddr = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    UInt32 dataSize = 0;
    OSStatus err = AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propAddr, 0, NULL, &dataSize);
    if (err != noErr || dataSize == 0) return 0;

    int deviceCount = dataSize / sizeof(AudioDeviceID);
    AudioDeviceID* devList = (AudioDeviceID*)safe_malloc(dataSize);
    if (!devList) return 0;

    err = AudioObjectGetPropertyData(kAudioObjectSystemObject, &propAddr, 0, NULL, &dataSize, devList);
    if (err != noErr) { safe_free((void**)&devList); return 0; }

    AudioDeviceID defaultInput = kAudioDeviceUnknown;
    UInt32 defaultSize = sizeof(AudioDeviceID);
    propAddr.mSelector = kAudioHardwarePropertyDefaultInputDevice;
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &propAddr, 0, NULL, &defaultSize, &defaultInput);

    int actual = 0;
    for (int i = 0; i < deviceCount && actual < max_devices; i++) {
        AudioDeviceID devID = devList[i];

        /* 检查是否有输入 */
        propAddr.mSelector = kAudioDevicePropertyStreams;
        propAddr.mScope = kAudioDevicePropertyScopeInput;
        dataSize = 0;
        err = AudioObjectGetPropertyDataSize(devID, &propAddr, 0, NULL, &dataSize);
        if (err != noErr || dataSize == 0) continue;

        AudioCaptureDeviceInfo* info = &devices[actual];
        memset(info, 0, sizeof(AudioCaptureDeviceInfo));

        CFStringRef nameStr = NULL;
        dataSize = sizeof(CFStringRef);
        propAddr.mSelector = kAudioDevicePropertyDeviceNameCFString;
        propAddr.mScope = kAudioObjectPropertyScopeGlobal;
        AudioObjectGetPropertyData(devID, &propAddr, 0, NULL, &dataSize, &nameStr);
        if (nameStr) {
            CFStringGetCString(nameStr, info->name, sizeof(info->name), kCFStringEncodingUTF8);
            CFRelease(nameStr);
        } else {
            snprintf(info->name, sizeof(info->name), "音频设备 %d", actual + 1);
        }

        snprintf(info->device_id, sizeof(info->device_id), "%u", (unsigned int)devID);
        info->max_channels = 2;
        info->default_sample_rate = 44100;
        actual++;
    }

    safe_free((void**)&devList);
    return actual;
}

static AudioCaptureContext* audio_capture_create(const char* device_id,
                                           int sample_rate,
                                           int channels,
                                           int bits_per_sample) {
    AudioCaptureContext* ctx = (AudioCaptureContext*)safe_malloc(sizeof(AudioCaptureContext));
    if (!ctx) return NULL;
    memset(ctx, 0, sizeof(AudioCaptureContext));

    ctx->sample_rate = (sample_rate > 0) ? sample_rate : 44100;
    ctx->channels = (channels > 0) ? channels : 1;
    ctx->bits_per_sample = (bits_per_sample > 0) ? bits_per_sample : 16;
    ctx->is_capturing = 0;
    ctx->thread_running = 0;

    AudioComponentDescription desc;
    memset(&desc, 0, sizeof(desc));
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_HALOutput;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;

    AudioComponent component = AudioComponentFindNext(NULL, &desc);
    if (!component) { safe_free((void**)&ctx); return NULL; }

    OSStatus err = AudioComponentInstanceNew(component, &ctx->audioUnit);
    if (err != noErr) { safe_free((void**)&ctx); return NULL; }

    /* 启用输入，禁用输出 */
    UInt32 enableIO = 1;
    AudioUnitSetProperty(ctx->audioUnit, kAudioOutputUnitProperty_EnableIO,
                         kAudioUnitScope_Input, 1, &enableIO, sizeof(enableIO));
    enableIO = 0;
    AudioUnitSetProperty(ctx->audioUnit, kAudioOutputUnitProperty_EnableIO,
                         kAudioUnitScope_Output, 0, &enableIO, sizeof(enableIO));

    AudioStreamBasicDescription asbd;
    memset(&asbd, 0, sizeof(asbd));
    asbd.mSampleRate = ctx->sample_rate;
    asbd.mFormatID = kAudioFormatLinearPCM;
    asbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    asbd.mBitsPerChannel = 32;
    asbd.mChannelsPerFrame = (UInt32)ctx->channels;
    asbd.mFramesPerPacket = 1;
    asbd.mBytesPerFrame = asbd.mChannelsPerFrame * 4;
    asbd.mBytesPerPacket = asbd.mBytesPerFrame;

    AudioUnitSetProperty(ctx->audioUnit, kAudioUnitProperty_StreamFormat,
                         kAudioUnitScope_Output, 1, &asbd, sizeof(asbd));

    AURenderCallbackStruct callbackStruct;
    callbackStruct.inputProc = audio_capture_render_cb;
    callbackStruct.inputProcRefCon = ctx;
    AudioUnitSetProperty(ctx->audioUnit, kAudioUnitProperty_SetRenderCallback,
                         kAudioUnitScope_Input, 0, &callbackStruct, sizeof(callbackStruct));

    err = AudioUnitInitialize(ctx->audioUnit);
    if (err != noErr) {
        AudioComponentInstanceDispose(ctx->audioUnit);
        safe_free((void**)&ctx); return NULL;
    }

    return ctx;
}

static int audio_capture_start(AudioCaptureContext* ctx,
                         void (*callback)(const float*, size_t, void*),
                         void* user_data) {
    if (!ctx || !callback) return -1;
    ctx->data_callback = callback;
    ctx->user_data = user_data;

    OSStatus err = AudioOutputUnitStart(ctx->audioUnit);
    if (err != noErr) return -1;

    ctx->is_capturing = 1;
    return 0;
}

static int audio_capture_process(AudioCaptureContext* ctx) {
    if (!ctx || !ctx->is_capturing || !ctx->audioUnit) return 0;

    /* 非阻塞轮询：从AudioUnit渲染一帧音频数据并通过回调传递 */
    AudioBufferList bufferList;
    bufferList.mNumberBuffers = 1;
    bufferList.mBuffers[0].mNumberChannels = (UInt32)ctx->channels;
    bufferList.mBuffers[0].mDataByteSize = 0;
    bufferList.mBuffers[0].mData = NULL;

    OSStatus err = AudioUnitRender(ctx->audioUnit, NULL, NULL, 0, 512, &bufferList);
    if (err == noErr && bufferList.mBuffers[0].mData && bufferList.mBuffers[0].mDataByteSize > 0) {
        UInt32 frames = bufferList.mBuffers[0].mDataByteSize / (UInt32)(ctx->channels * sizeof(float));
        const float* data = (const float*)bufferList.mBuffers[0].mData;
        size_t total_samples = (size_t)frames * (size_t)ctx->channels;
        if (ctx->data_callback) {
            ctx->data_callback(data, total_samples, ctx->user_data);
        }
        return (int)total_samples;
    }
    return 0;
}

static int audio_capture_stop(AudioCaptureContext* ctx) {
    if (!ctx) return -1;
    if (ctx->is_capturing) {
        AudioOutputUnitStop(ctx->audioUnit);
    }
    ctx->is_capturing = 0;
    return 0;
}

static void audio_capture_free(AudioCaptureContext* ctx) {
    if (!ctx) return;
    audio_capture_stop(ctx);
    AudioUnitUninitialize(ctx->audioUnit);
    AudioComponentInstanceDispose(ctx->audioUnit);
    safe_free((void**)&ctx);
}

#else
/* ==================== Linux ALSA 捕获 ==================== */
#include <alsa/asoundlib.h>
#include <pthread.h>

struct AudioCaptureContext {
    snd_pcm_t* handle;
    int is_capturing;
    int sample_rate;
    int channels;
    int bits_per_sample;
    pthread_t capture_thread;
    int thread_running;
    void (*data_callback)(const float*, size_t, void*);
    void* user_data;
};

static void* audio_capture_thread_func(void* arg) {
    AudioCaptureContext* ctx = (AudioCaptureContext*)arg;
    if (!ctx || !ctx->handle) return NULL;

    int frame_size = ctx->channels;
    int period_frames = 512;
    int buffer_frames = period_frames * 4;
    int sample_size = ctx->bits_per_sample / 8;
    if (sample_size < 1) sample_size = 2;

    char* raw_buf = (char*)safe_malloc(buffer_frames * frame_size * sample_size);
    float* float_buf = (float*)safe_malloc(period_frames * frame_size * sizeof(float));
    if (!raw_buf || !float_buf) {
        safe_free((void**)&raw_buf);
        safe_free((void**)&float_buf);
        return NULL;
    }

    while (ctx->thread_running) {
        snd_pcm_sframes_t frames = snd_pcm_readi(ctx->handle, raw_buf, period_frames);
        if (frames < 0) {
            snd_pcm_recover(ctx->handle, (int)frames, 0);
            /* 恢复后短暂休眠，避免错误路径上CPU忙等，降低占用 */
            struct timespec ts = {0, 100000};
            nanosleep(&ts, NULL);
            continue;
        }

        int total_samples = (int)frames * frame_size;
        if (ctx->bits_per_sample == 16) {
            short* src = (short*)raw_buf;
            for (int i = 0; i < total_samples; i++) {
                float_buf[i] = src[i] / 32768.0f;
            }
        } else if (ctx->bits_per_sample == 32) {
            int* src = (int*)raw_buf;
            for (int i = 0; i < total_samples; i++) {
                float_buf[i] = src[i] / 2147483648.0f;
            }
        } else {
            for (int i = 0; i < total_samples; i++) {
                float_buf[i] = ((short*)raw_buf)[i] / 32768.0f;
            }
        }

        if (ctx->data_callback) {
            ctx->data_callback(float_buf, total_samples, ctx->user_data);
        }
    }

    safe_free((void**)&raw_buf);
    safe_free((void**)&float_buf);
    return NULL;
}

static int audio_capture_enumerate_devices(AudioCaptureDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return 0;

    char** hints;
    int actual = 0;

    if (snd_device_name_hint(-1, "pcm", (void***)&hints) < 0) return 0;

    for (int i = 0; hints[i] && actual < max_devices; i++) {
        char* name = snd_device_name_get_hint(hints[i], "NAME");
        char* desc = snd_device_name_get_hint(hints[i], "DESC");
        char* ioid = snd_device_name_get_hint(hints[i], "IOID");

        /* 只选择输入设备 */
        if (ioid && strcmp(ioid, "Input") != 0 && strcmp(ioid, "Output") != 0) {
            /* Ioid = "null" 等跳过 */;
        }

        if (name) {
            int is_input = (!ioid || strcmp(ioid, "Input") == 0);
            if (is_input && !strstr(name, "surround") && !strstr(name, "iec958") &&
                !strstr(name, "hdmi") && !strstr(name, "modem")) {
                AudioCaptureDeviceInfo* info = &devices[actual];
                memset(info, 0, sizeof(AudioCaptureDeviceInfo));
                snprintf(info->device_id, sizeof(info->device_id), "%s", name);
                if (desc) {
                    const char* nl = strchr(desc, '
');
                    size_t len = nl ? (size_t)(nl - desc) : strlen(desc);
                    if (len >= sizeof(info->name)) len = sizeof(info->name) - 1;
                    memcpy(info->name, desc, len);
                    info->name[len] = ' ';
                } else {
                    snprintf(info->name, sizeof(info->name), "%s", name);
                }
                info->max_channels = 2;
                info->default_sample_rate = 48000;
                actual++;
            }
        }

        if (name) free(name);
        if (desc) free(desc);
        if (ioid) free(ioid);
    }

    snd_device_name_free_hint((void**)hints);
    return actual;
}

static AudioCaptureContext* audio_capture_create(const char* device_id,
                                           int sample_rate,
                                           int channels,
                                           int bits_per_sample) {
    AudioCaptureContext* ctx = (AudioCaptureContext*)safe_malloc(sizeof(AudioCaptureContext));
    if (!ctx) return NULL;
    memset(ctx, 0, sizeof(AudioCaptureContext));

    ctx->sample_rate = (sample_rate > 0) ? sample_rate : 48000;
    ctx->channels = (channels > 0) ? channels : 1;
    ctx->bits_per_sample = (bits_per_sample > 0) ? bits_per_sample : 16;
    ctx->is_capturing = 0;
    ctx->thread_running = 0;
    ctx->handle = NULL;
    ctx->data_callback = NULL;
    ctx->user_data = NULL;

    int rc = snd_pcm_open(&ctx->handle, device_id ? device_id : "default",
                           SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0) { safe_free((void**)&ctx); return NULL; }

    snd_pcm_hw_params_t* hw_params = NULL;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(ctx->handle, hw_params);
    snd_pcm_hw_params_set_access(ctx->handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);

    snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;
    if (ctx->bits_per_sample == 8) format = SND_PCM_FORMAT_U8;
    else if (ctx->bits_per_sample == 32) format = SND_PCM_FORMAT_S32_LE;

    snd_pcm_hw_params_set_format(ctx->handle, hw_params, format);
    snd_pcm_hw_params_set_channels(ctx->handle, hw_params, (unsigned int)ctx->channels);

    unsigned int rate = (unsigned int)ctx->sample_rate;
    snd_pcm_hw_params_set_rate_near(ctx->handle, hw_params, &rate, 0);
    ctx->sample_rate = (int)rate;

    snd_pcm_uframes_t period_size = 512;
    snd_pcm_hw_params_set_period_size_near(ctx->handle, hw_params, &period_size, 0);

    rc = snd_pcm_hw_params(ctx->handle, hw_params);
    if (rc < 0) { snd_pcm_close(ctx->handle); safe_free((void**)&ctx); return NULL; }

    return ctx;
}

static int audio_capture_start(AudioCaptureContext* ctx,
                         void (*callback)(const float*, size_t, void*),
                         void* user_data) {
    if (!ctx || !callback) return -1;
    ctx->data_callback = callback;
    ctx->user_data = user_data;
    ctx->thread_running = 1;

    int rc = pthread_create(&ctx->capture_thread, NULL, audio_capture_thread_func, ctx);
    if (rc != 0) { ctx->thread_running = 0; return -1; }

    ctx->is_capturing = 1;
    return 0;
}

static int audio_capture_process(AudioCaptureContext* ctx) {
    if (!ctx || !ctx->is_capturing || !ctx->handle) return 0;

    /* 非阻塞轮询：检查ALSA可用帧数并读取 */
    snd_pcm_sframes_t avail = snd_pcm_avail(ctx->handle);
    if (avail <= 0) return 0;

    int max_frames = 512;
    if (avail > max_frames) avail = max_frames;

    int frame_size = ctx->channels;
    int sample_size = ctx->bits_per_sample / 8;
    if (sample_size < 1) sample_size = 2;

    size_t buf_bytes = (size_t)avail * (size_t)frame_size * (size_t)sample_size;
    char* raw_buf = (char*)safe_malloc(buf_bytes);
    float* float_buf = (float*)safe_malloc((size_t)avail * (size_t)frame_size * sizeof(float));
    if (!raw_buf || !float_buf) {
        safe_free((void**)&raw_buf);
        safe_free((void**)&float_buf);
        return 0;
    }

    snd_pcm_sframes_t frames = snd_pcm_readi(ctx->handle, raw_buf, (snd_pcm_uframes_t)avail);
    if (frames < 0) {
        snd_pcm_recover(ctx->handle, (int)frames, 0);
        safe_free((void**)&raw_buf);
        safe_free((void**)&float_buf);
        return 0;
    }

    int total_samples = (int)frames * frame_size;
    if (ctx->bits_per_sample == 16) {
        short* src = (short*)raw_buf;
        for (int i = 0; i < total_samples; i++) {
            float_buf[i] = src[i] / 32768.0f;
        }
    } else if (ctx->bits_per_sample == 32) {
        int* src = (int*)raw_buf;
        for (int i = 0; i < total_samples; i++) {
            float_buf[i] = src[i] / 2147483648.0f;
        }
    } else {
        for (int i = 0; i < total_samples; i++) {
            float_buf[i] = ((short*)raw_buf)[i] / 32768.0f;
        }
    }

    if (ctx->data_callback) {
        ctx->data_callback(float_buf, (size_t)total_samples, ctx->user_data);
    }

    safe_free((void**)&raw_buf);
    safe_free((void**)&float_buf);
    return total_samples;
}

static int audio_capture_stop(AudioCaptureContext* ctx) {
    if (!ctx) return -1;
    if (ctx->is_capturing) {
        ctx->thread_running = 0;
        pthread_join(ctx->capture_thread, NULL);
    }
    ctx->is_capturing = 0;
    return 0;
}

static void audio_capture_free(AudioCaptureContext* ctx) {
    if (!ctx) return;
    audio_capture_stop(ctx);
    if (ctx->handle) {
        snd_pcm_drain(ctx->handle);
        snd_pcm_close(ctx->handle);
    }
    safe_free((void**)&ctx);
}

#endif /* 平台选择 */
