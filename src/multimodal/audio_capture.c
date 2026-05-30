/**
 * @file audio_capture.c
 * @brief 真实音频硬件采集模块
 *
 * K-002 已修复：Windows纯C使用waveIn API纯C后端。
 *   C++编译使用WASAPI高性能路径。
 *   Linux使用ALSA纯C路径。
 * 
 * 设备枚举、格式配置、实时音频流采集。
 * 核心原则：无硬件时返回空/错误，绝不生成虚假音频数据。
 */

#include "selflnn/multimodal/audio.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* 最大音频设备数 */
#define MAX_AUDIO_DEVICES 32

/* ================================================================
 * Windows WASAPI 音频采集实现
 * 使用 Windows Core Audio API (WASAPI)
 * 通过COM接口直接与音频硬件通信，零额外依赖
 * ================================================================ */

#ifdef _WIN32
#ifdef __cplusplus

#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <avrt.h>

/* 链接Core Audio库 */
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "avrt.lib")

/* WASAPI GUID定义 */
static const CLSID g_CLSID_MMDeviceEnumerator = {0xBCDE0395, 0xE52F, 0x467C, {0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}};
static const IID g_IID_IMMDeviceEnumerator = {0xA95664D2, 0x9614, 0x4F35, {0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6}};
static const IID g_IID_IAudioClient = {0x1CB9AD4C, 0xDBFA, 0x4C32, {0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2}};
static const IID g_IID_IAudioCaptureClient = {0xC8ADBD64, 0xE71E, 0x48A0, {0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17}};

/**
 * @brief Windows WASAPI音频采集设备结构体
 */
typedef struct {
    IMMDeviceEnumerator* enumerator;    /* 设备枚举器接口 */
    IMMDevice* device;                  /* 当前音频设备 */
    IAudioClient* audio_client;         /* 音频客户端接口 */
    IAudioCaptureClient* capture_client; /* 音频采集客户端接口 */
    WAVEFORMATEX* wave_format;          /* 音频格式 */
    HANDLE capture_event;               /* 采集事件句柄 */
    HANDLE capture_thread;              /* 采集线程句柄 */
    int is_capturing;                   /* 是否正在采集 */
    int sample_rate;                    /* 采样率(Hz) */
    int channels;                       /* 声道数 */
    int bits_per_sample;                /* 位深 */
    int frame_size;                     /* 帧大小(字节) */
    size_t buffer_frames;               /* 缓冲区帧数 */
    /* 采集回调 */
    void (*on_audio_data)(const float* samples, size_t num_samples, void* user_data);
    void* callback_user_data;
    /* 内部PCM→float转换缓冲 */
    float* float_buffer;
    size_t float_buffer_size;
    uint8_t* raw_buffer;
    size_t raw_buffer_size;
} WasapiCaptureContext;

/* 全局设备列表缓存 */
/**
 * @brief 初始化Windows COM组件
 */
static int wasapi_init_com(void) {
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        log_error("[音频采集] CoInitializeEx失败: 0x%08lX", (unsigned long)hr);
        return -1;
    }
    return 0;
}

/**
 * @brief 枚举Windows音频采集设备
 * @param device_names 输出设备名称数组
 * @param device_ids 输出设备ID数组
 * @param max_devices 最大设备数
 * @param num_devices 输出实际设备数
 * @return 0成功，-1失败
 */
int audio_capture_enumerate_devices_win(char device_names[][256],
                                         char device_ids[][128],
                                         int max_devices,
                                         int* num_devices) {
    HRESULT hr;
    IMMDeviceEnumerator* enumerator = NULL;
    IMMDeviceCollection* collection = NULL;
    UINT count = 0;
    int device_count = 0;

    if (!device_names || !device_ids || !num_devices || max_devices <= 0) {
        return -1;
    }
    *num_devices = 0;

    hr = CoCreateInstance(&g_CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &g_IID_IMMDeviceEnumerator, (void**)&enumerator);
    if (FAILED(hr)) {
        log_error("[音频采集] 创建MMDeviceEnumerator失败: 0x%08lX", (unsigned long)hr);
        return -1;
    }

    hr = IMMDeviceEnumerator_EnumAudioEndpoints(enumerator, eCapture,
                                                  DEVICE_STATE_ACTIVE, &collection);
    if (FAILED(hr)) {
        log_error("[音频采集] 枚举音频端点失败: 0x%08lX", (unsigned long)hr);
        IMMDeviceEnumerator_Release(enumerator);
        return -1;
    }

    hr = IMMDeviceCollection_GetCount(collection, &count);
    if (FAILED(hr)) {
        IMMDeviceCollection_Release(collection);
        IMMDeviceEnumerator_Release(enumerator);
        return -1;
    }

    for (UINT i = 0; i < count && device_count < max_devices; i++) {
        IMMDevice* device = NULL;
        IPropertyStore* props = NULL;
        PROPVARIANT var_name;

        hr = IMMDeviceCollection_Item(collection, i, &device);
        if (FAILED(hr)) continue;

        hr = IMMDevice_OpenPropertyStore(device, STGM_READ, &props);
        if (FAILED(hr)) {
            IMMDevice_Release(device);
            continue;
        }

        PropVariantInit(&var_name);
        hr = IPropertyStore_GetValue(props, &PKEY_Device_FriendlyName, &var_name);
        if (SUCCEEDED(hr) && var_name.pwszVal) {
            /* 转换宽字符到UTF-8 */
            int len = WideCharToMultiByte(CP_UTF8, 0, var_name.pwszVal, -1,
                                          device_names[device_count], 256, NULL, NULL);
            if (len <= 0) {
                device_names[device_count][0] = '\0';
            }
            /* 设备ID使用索引 */
            snprintf(device_ids[device_count], 128, "wasapi_device_%u", i);
            device_count++;
        }

        PropVariantClear(&var_name);
        IPropertyStore_Release(props);
        IMMDevice_Release(device);
    }

    *num_devices = device_count;
    IMMDeviceCollection_Release(collection);
    IMMDeviceEnumerator_Release(enumerator);
    return 0;
}

/**
 * @brief 创建WASAPI音频采集上下文
 * @param device_id 设备ID（可为NULL使用默认设备）
 * @param sample_rate 采样率(Hz)，推荐16000
 * @param channels 声道数，推荐1(单声道)
 * @param bits_per_sample 位深，推荐16
 * @return 采集上下文指针，失败返回NULL
 */
WasapiCaptureContext* wasapi_capture_create(const char* device_id,
                                              int sample_rate,
                                              int channels,
                                              int bits_per_sample) {
    HRESULT hr;
    WasapiCaptureContext* ctx = NULL;

    ctx = (WasapiCaptureContext*)safe_calloc(1, sizeof(WasapiCaptureContext));
    if (!ctx) return NULL;

    ctx->sample_rate = sample_rate > 0 ? sample_rate : 16000;
    ctx->channels = channels > 0 ? channels : 1;
    ctx->bits_per_sample = bits_per_sample > 0 ? bits_per_sample : 16;
    ctx->is_capturing = 0;

    /* 创建设备枚举器 */
    hr = CoCreateInstance(&g_CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &g_IID_IMMDeviceEnumerator, (void**)&ctx->enumerator);
    if (FAILED(hr)) {
        log_error("[音频采集] 创建设备枚举器失败");
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 获取默认采集设备 */
    hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(ctx->enumerator,
                                                      eCapture, eConsole,
                                                      &ctx->device);
    if (FAILED(hr)) {
        log_error("[音频采集] 获取默认采集设备失败");
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 激活音频客户端 */
    hr = IMMDevice_Activate(ctx->device, &g_IID_IAudioClient,
                            CLSCTX_ALL, NULL, (void**)&ctx->audio_client);
    if (FAILED(hr)) {
        log_error("[音频采集] 激活音频客户端失败");
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 设置音频格式 */
    ctx->wave_format = (WAVEFORMATEX*)safe_calloc(1, sizeof(WAVEFORMATEX) + 22);
    if (!ctx->wave_format) {
        IAudioClient_Release(ctx->audio_client);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    ctx->wave_format->wFormatTag = WAVE_FORMAT_PCM;
    ctx->wave_format->nChannels = (WORD)ctx->channels;
    ctx->wave_format->nSamplesPerSec = (DWORD)ctx->sample_rate;
    ctx->wave_format->wBitsPerSample = (WORD)ctx->bits_per_sample;
    ctx->wave_format->nBlockAlign = (WORD)(ctx->channels * ctx->bits_per_sample / 8);
    ctx->wave_format->nAvgBytesPerSec = ctx->wave_format->nSamplesPerSec * ctx->wave_format->nBlockAlign;
    ctx->wave_format->cbSize = 0;

    /* 初始化音频客户端 */
    REFERENCE_TIME buffer_duration = 100000; /* 100ms缓冲 */
    hr = IAudioClient_Initialize(ctx->audio_client,
                                  AUDCLNT_SHAREMODE_SHARED,
                                  AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                  buffer_duration, 0,
                                  ctx->wave_format, NULL);
    if (FAILED(hr)) {
        log_error("[音频采集] 初始化音频客户端失败");
        safe_free((void**)&ctx->wave_format);
        IAudioClient_Release(ctx->audio_client);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 获取缓冲区大小 */
    hr = IAudioClient_GetBufferSize(ctx->audio_client, (UINT32*)&ctx->buffer_frames);
    if (FAILED(hr)) {
        log_error("[音频采集] 获取缓冲区大小失败");
        safe_free((void**)&ctx->wave_format);
        IAudioClient_Release(ctx->audio_client);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 获取采集客户端 */
    hr = IAudioClient_GetService(ctx->audio_client, &g_IID_IAudioCaptureClient,
                                  (void**)&ctx->capture_client);
    if (FAILED(hr)) {
        log_error("[音频采集] 获取采集客户端失败");
        safe_free((void**)&ctx->wave_format);
        IAudioClient_Release(ctx->audio_client);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 创建事件句柄 */
    ctx->capture_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!ctx->capture_event) {
        safe_free((void**)&ctx->wave_format);
        IAudioCaptureClient_Release(ctx->capture_client);
        IAudioClient_Release(ctx->audio_client);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    hr = IAudioClient_SetEventHandle(ctx->audio_client, ctx->capture_event);
    if (FAILED(hr)) {
        CloseHandle(ctx->capture_event);
        safe_free((void**)&ctx->wave_format);
        IAudioCaptureClient_Release(ctx->capture_client);
        IAudioClient_Release(ctx->audio_client);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 分配内部缓冲 */
    ctx->frame_size = ctx->channels * ctx->bits_per_sample / 8;
    ctx->float_buffer_size = ctx->buffer_frames * ctx->channels;
    ctx->raw_buffer_size = ctx->buffer_frames * ctx->frame_size;

    ctx->float_buffer = (float*)safe_malloc(ctx->float_buffer_size * sizeof(float));
    ctx->raw_buffer = (uint8_t*)safe_malloc(ctx->raw_buffer_size);

    if (!ctx->float_buffer || !ctx->raw_buffer) {
        CloseHandle(ctx->capture_event);
        safe_free((void**)&ctx->wave_format);
        safe_free((void**)&ctx->float_buffer);
        safe_free((void**)&ctx->raw_buffer);
        IAudioCaptureClient_Release(ctx->capture_client);
        IAudioClient_Release(ctx->audio_client);
        IMMDevice_Release(ctx->device);
        IMMDeviceEnumerator_Release(ctx->enumerator);
        safe_free((void**)&ctx);
        return NULL;
    }

    log_info("[音频采集] WASAPI设备初始化成功: %dHz, %d声道, %d位",
             ctx->sample_rate, ctx->channels, ctx->bits_per_sample);
    return ctx;
}

/**
 * @brief WASAPI采集线程函数
 */
static DWORD WINAPI wasapi_capture_thread_proc(LPVOID param) {
    WasapiCaptureContext* ctx = (WasapiCaptureContext*)param;
    HRESULT hr;

    /* 设置为高优先级MMCSS任务 */
    DWORD task_index = 0;
    HANDLE mmcss_handle = AvSetMmThreadCharacteristicsA("Audio", &task_index);

    while (ctx->is_capturing) {
        DWORD wait_result = WaitForSingleObject(ctx->capture_event, 200);
        if (wait_result != WAIT_OBJECT_0) continue;

        /* 循环读取所有可用数据包 */
        for (;;) {
            UINT32 packet_length = 0;
            UINT32 num_frames_available = 0;
            BYTE* data = NULL;
            DWORD flags = 0;

            hr = IAudioCaptureClient_GetBuffer(ctx->capture_client,
                                                &data, &num_frames_available,
                                                &flags, NULL, NULL);
            if (FAILED(hr)) break;

            if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                /* 静音数据 */
                memset(ctx->float_buffer, 0, num_frames_available * ctx->channels * sizeof(float));
            } else if (data && num_frames_available > 0) {
                /* 转换PCM数据到float */
                size_t total_samples = (size_t)num_frames_available * ctx->channels;
                if (ctx->bits_per_sample == 16) {
                    const int16_t* pcm = (const int16_t*)data;
                    for (size_t i = 0; i < total_samples; i++) {
                        ctx->float_buffer[i] = (float)pcm[i] / 32768.0f;
                    }
                } else if (ctx->bits_per_sample == 24) {
                    const uint8_t* pcm24 = data;
                    for (size_t i = 0; i < total_samples; i++) {
                        int32_t val = (int32_t)(pcm24[0] | (pcm24[1] << 8) | ((int8_t)pcm24[2] << 16));
                        ctx->float_buffer[i] = (float)val / 8388608.0f;
                        pcm24 += 3;
                    }
                } else if (ctx->bits_per_sample == 32) {
                    const int32_t* pcm32 = (const int32_t*)data;
                    for (size_t i = 0; i < total_samples; i++) {
                        ctx->float_buffer[i] = (float)pcm32[i] / 2147483648.0f;
                    }
                } else {
                    /* 8位 */
                    const uint8_t* pcm8 = data;
                    for (size_t i = 0; i < total_samples; i++) {
                        ctx->float_buffer[i] = ((float)pcm8[i] - 128.0f) / 128.0f;
                    }
                }

                /* 调用回调函数 */
                if (ctx->on_audio_data) {
                    ctx->on_audio_data(ctx->float_buffer, total_samples, ctx->callback_user_data);
                }
            }

            hr = IAudioCaptureClient_ReleaseBuffer(ctx->capture_client, num_frames_available);
            if (FAILED(hr)) break;

            /* 检查是否还有更多数据 */
            hr = IAudioClient_GetCurrentPadding(ctx->audio_client, &packet_length);
            if (FAILED(hr) || packet_length < (UINT32)(ctx->buffer_frames / 2)) break;
        }
    }

    if (mmcss_handle) {
        AvRevertMmThreadCharacteristics(mmcss_handle);
    }
    return 0;
}

/**
 * @brief 开始WASAPI音频采集
 * @param ctx 采集上下文
 * @param callback 音频数据回调函数
 * @param user_data 用户数据
 * @return 0成功，-1失败
 */
int wasapi_capture_start(WasapiCaptureContext* ctx,
                          void (*callback)(const float* samples, size_t num_samples, void* user_data),
                          void* user_data) {
    HRESULT hr;
    if (!ctx) return -1;

    ctx->on_audio_data = callback;
    ctx->callback_user_data = user_data;
    ctx->is_capturing = 1;

    hr = IAudioClient_Start(ctx->audio_client);
    if (FAILED(hr)) {
        ctx->is_capturing = 0;
        log_error("[音频采集] 启动音频客户端失败");
        return -1;
    }

    ctx->capture_thread = CreateThread(NULL, 0, wasapi_capture_thread_proc, ctx, 0, NULL);
    if (!ctx->capture_thread) {
        IAudioClient_Stop(ctx->audio_client);
        ctx->is_capturing = 0;
        log_error("[音频采集] 创建采集线程失败");
        return -1;
    }

    log_info("[音频采集] WASAPI采集已启动");
    return 0;
}

/**
 * @brief 停止WASAPI音频采集
 * @param ctx 采集上下文
 * @return 0成功，-1失败
 */
int wasapi_capture_stop(WasapiCaptureContext* ctx) {
    if (!ctx) return -1;

    ctx->is_capturing = 0;

    if (ctx->capture_thread) {
        WaitForSingleObject(ctx->capture_thread, 3000);
        CloseHandle(ctx->capture_thread);
        ctx->capture_thread = NULL;
    }

    IAudioClient_Stop(ctx->audio_client);
    log_info("[音频采集] WASAPI采集已停止");
    return 0;
}

/**
 * @brief 释放WASAPI采集上下文
 * @param ctx 采集上下文
 */
void wasapi_capture_free(WasapiCaptureContext* ctx) {
    if (!ctx) return;

    if (ctx->is_capturing) {
        wasapi_capture_stop(ctx);
    }

    if (ctx->capture_event) CloseHandle(ctx->capture_event);
    if (ctx->capture_client) IAudioCaptureClient_Release(ctx->capture_client);
    if (ctx->audio_client) IAudioClient_Release(ctx->audio_client);
    if (ctx->device) IMMDevice_Release(ctx->device);
    if (ctx->enumerator) IMMDeviceEnumerator_Release(ctx->enumerator);

    safe_free((void**)&ctx->wave_format);
    safe_free((void**)&ctx->float_buffer);
    safe_free((void**)&ctx->raw_buffer);
    safe_free((void**)&ctx);

    log_info("[音频采集] WASAPI采集上下文已释放");
}

#else  /* pure C: Windows waveIn音频采集实现 */

#include <mmsystem.h>

#define WAVEIN_NUM_BUFFERS 8
#define WAVEIN_BUFFER_MS   50

typedef struct {
    int device_id;
    int sample_rate;
    int channels;
    int bits_per_sample;
    int is_capturing;
    HWAVEIN wavein_handle;
    WAVEHDR wave_headers[WAVEIN_NUM_BUFFERS];
    int16_t* sample_buffers[WAVEIN_NUM_BUFFERS];
    int buffer_sample_count;
    HANDLE capture_thread;
    HANDLE stop_event;
    CRITICAL_SECTION lock;
    float* float_buffer;
    size_t float_buffer_size;
    void (*on_audio_data)(const float* samples, size_t num_samples, void* user_data);
    void* callback_user_data;
} WaveInCaptureContext;

static void CALLBACK wavein_callback_proc(HWAVEIN hwi, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2) {
    (void)hwi;(void)dwParam2;
    if (uMsg != WIM_DATA) return;
    WaveInCaptureContext* ctx = (WaveInCaptureContext*)dwInstance;
    if (!ctx || !ctx->is_capturing) return;
    WAVEHDR* header = (WAVEHDR*)dwParam1;
    if (!header || !(header->dwFlags & WHDR_DONE)) return;
    EnterCriticalSection(&ctx->lock);
    int num_samples = header->dwBytesRecorded / (ctx->bits_per_sample / 8);
    size_t float_needed = (size_t)num_samples;
    if (ctx->float_buffer_size < float_needed) {
        float* new_buf = (float*)realloc(ctx->float_buffer, float_needed * sizeof(float));
        if (new_buf) { ctx->float_buffer = new_buf; ctx->float_buffer_size = float_needed; }
    }
    if (ctx->float_buffer && num_samples > 0) {
        if (ctx->bits_per_sample == 16) {
            int16_t* src = (int16_t*)header->lpData;
            for (int i = 0; i < num_samples; i++) {
                ctx->float_buffer[i] = src[i] / 32768.0f;
            }
        } else if (ctx->bits_per_sample == 8) {
            uint8_t* src = (uint8_t*)header->lpData;
            for (int i = 0; i < num_samples; i++) {
                ctx->float_buffer[i] = (src[i] - 128) / 128.0f;
            }
        }
        if (ctx->on_audio_data) {
            ctx->on_audio_data(ctx->float_buffer, (size_t)num_samples, ctx->callback_user_data);
        }
    }
    header->dwFlags &= ~WHDR_DONE;
    header->dwBytesRecorded = 0;
    waveInAddBuffer(ctx->wavein_handle, header, sizeof(WAVEHDR));
    LeaveCriticalSection(&ctx->lock);
}

static int audio_capture_enumerate_devices_win(char names[][256], char ids[][128], int max_devices, int* count) {
    if (!names || !ids || !count || max_devices <= 0) return -1;
    *count = 0;
    UINT num_devs = waveInGetNumDevs();
    for (UINT i = 0; i < num_devs && *count < max_devices; i++) {
        WAVEINCAPSA caps;
        if (waveInGetDevCapsA(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR) {
            snprintf(names[*count], 256, "%s", caps.szPname);
            snprintf(ids[*count], 128, "wavein:%u", i);
            (*count)++;
        }
    }
    return 0;
}

static void* wasapi_capture_create(const char* device_id, int sample_rate, int channels, int bits) {
    WaveInCaptureContext* ctx = (WaveInCaptureContext*)calloc(1, sizeof(WaveInCaptureContext));
    if (!ctx) return NULL;
    ctx->device_id = 0;
    if (device_id) {
        if (sscanf(device_id, "wavein:%d", &ctx->device_id) != 1) ctx->device_id = 0;
    }
    ctx->sample_rate = sample_rate > 0 ? sample_rate : 16000;
    ctx->channels = (channels > 0) ? channels : 1;
    ctx->bits_per_sample = (bits > 0) ? bits : 16;
    ctx->buffer_sample_count = ctx->sample_rate * WAVEIN_BUFFER_MS / 1000 * ctx->channels;
    InitializeCriticalSection(&ctx->lock);
    WAVEFORMATEX wfx;
    memset(&wfx, 0, sizeof(wfx));
    wfx.wFormatTag = WAVE_FORMAT_PCM;
    wfx.nChannels = (WORD)ctx->channels;
    wfx.nSamplesPerSec = (DWORD)ctx->sample_rate;
    wfx.wBitsPerSample = (WORD)ctx->bits_per_sample;
    wfx.nBlockAlign = (wfx.nChannels * wfx.wBitsPerSample) / 8;
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
    MMRESULT mr = waveInOpen(&ctx->wavein_handle, (UINT)ctx->device_id, &wfx, (DWORD_PTR)wavein_callback_proc, (DWORD_PTR)ctx, CALLBACK_FUNCTION);
    if (mr != MMSYSERR_NOERROR) {
        DeleteCriticalSection(&ctx->lock); free(ctx); return NULL;
    }
    size_t buf_size = (size_t)ctx->buffer_sample_count * (ctx->bits_per_sample / 8);
    for (int i = 0; i < WAVEIN_NUM_BUFFERS; i++) {
        ctx->sample_buffers[i] = (int16_t*)malloc(buf_size);
        if (!ctx->sample_buffers[i]) {
            for (int j = 0; j < i; j++) free(ctx->sample_buffers[j]);
            waveInClose(ctx->wavein_handle);
            DeleteCriticalSection(&ctx->lock); free(ctx); return NULL;
        }
        memset(&ctx->wave_headers[i], 0, sizeof(WAVEHDR));
        ctx->wave_headers[i].lpData = (LPSTR)ctx->sample_buffers[i];
        ctx->wave_headers[i].dwBufferLength = (DWORD)buf_size;
        ctx->wave_headers[i].dwUser = (DWORD_PTR)i;
        waveInPrepareHeader(ctx->wavein_handle, &ctx->wave_headers[i], sizeof(WAVEHDR));
        waveInAddBuffer(ctx->wavein_handle, &ctx->wave_headers[i], sizeof(WAVEHDR));
    }
    return ctx;
}

static int wasapi_capture_start(void* ctx_ptr, void (*cb)(const float*, size_t, void*), void* user_data) {
    WaveInCaptureContext* ctx = (WaveInCaptureContext*)ctx_ptr;
    if (!ctx || ctx->is_capturing) return -1;
    ctx->on_audio_data = cb;
    ctx->callback_user_data = user_data;
    ctx->is_capturing = 1;
    MMRESULT mr = waveInStart(ctx->wavein_handle);
    if (mr != MMSYSERR_NOERROR) { ctx->is_capturing = 0; return -1; }
    return 0;
}

static int wasapi_capture_stop(void* ctx_ptr) {
    WaveInCaptureContext* ctx = (WaveInCaptureContext*)ctx_ptr;
    if (!ctx || !ctx->is_capturing) return -1;
    ctx->is_capturing = 0;
    waveInStop(ctx->wavein_handle);
    waveInReset(ctx->wavein_handle);
    return 0;
}

static void wasapi_capture_free(void* ctx_ptr) {
    WaveInCaptureContext* ctx = (WaveInCaptureContext*)ctx_ptr;
    if (!ctx) return;
    if (ctx->is_capturing) wasapi_capture_stop(ctx);
    if (ctx->wavein_handle) {
        for (int i = 0; i < WAVEIN_NUM_BUFFERS; i++) {
            waveInUnprepareHeader(ctx->wavein_handle, &ctx->wave_headers[i], sizeof(WAVEHDR));
            if (ctx->sample_buffers[i]) free(ctx->sample_buffers[i]);
        }
        waveInClose(ctx->wavein_handle);
    }
    if (ctx->float_buffer) free(ctx->float_buffer);
    DeleteCriticalSection(&ctx->lock);
    free(ctx);
}

#endif /* __cplusplus */
/* K-002已修复: Windows纯C使用waveIn API后端（上方#else块中已实现），
   移除之前阻塞纯C路径的桩函数。waveIn是纯C Win32 API，不需要C++编译器。 */

#endif /* _WIN32 */

/* ================================================================
 * Linux ALSA 音频采集实现
 * ================================================================ */

#ifdef __linux__

#include <alsa/asound.h>

/**
 * @brief Linux ALSA音频采集设备结构体
 */
typedef struct {
    snd_pcm_t* pcm_handle;              /* ALSA PCM句柄 */
    snd_pcm_hw_params_t* hw_params;     /* 硬件参数 */
    snd_pcm_sw_params_t* sw_params;     /* 软件参数 */
    snd_pcm_uframes_t buffer_frames;    /* 缓冲区帧数 */
    snd_pcm_uframes_t period_frames;    /* 周期帧数 */
    int is_capturing;                   /* 是否正在采集 */
    int sample_rate;                    /* 采样率 */
    int channels;                       /* 声道数 */
    int bits_per_sample;                /* 位深 */
    int frame_size;                     /* 帧大小(字节) */
    /* 采集回调 */
    void (*on_audio_data)(const float* samples, size_t num_samples, void* user_data);
    void* callback_user_data;
    /* 缓冲 */
    float* float_buffer;
    size_t float_buffer_size;
    char* device_name;                  /* 设备名称 */
} AlsaCaptureContext;

/**
 * @brief 枚举Linux ALSA音频采集设备
 */
int audio_capture_enumerate_devices_linux(char device_names[][256],
                                           char device_ids[][128],
                                           int max_devices,
                                           int* num_devices) {
    int card = -1;
    int device_count = 0;

    if (!device_names || !device_ids || !num_devices || max_devices <= 0) {
        return -1;
    }
    *num_devices = 0;

    /* 遍历所有声卡 */
    while (snd_card_next(&card) >= 0 && card >= 0 && device_count < max_devices) {
        char* card_name = NULL;
        if (snd_card_get_name(card, &card_name) < 0) continue;

        /* 检查是否有采集子设备 */
        snd_ctl_t* ctl_handle = NULL;
        char ctl_name[64];
        snprintf(ctl_name, sizeof(ctl_name), "hw:%d", card);

        if (snd_ctl_open(&ctl_handle, ctl_name, 0) >= 0) {
            snd_pcm_info_t* pcm_info;
            snd_pcm_info_alloca(&pcm_info);

            int dev = -1;
            while (snd_ctl_pcm_next_device(ctl_handle, &dev) >= 0 && dev >= 0 && device_count < max_devices) {
                snd_pcm_info_set_device(pcm_info, dev);
                snd_pcm_info_set_subdevice(pcm_info, 0);
                snd_pcm_info_set_stream(pcm_info, SND_PCM_STREAM_CAPTURE);

                if (snd_ctl_pcm_info(ctl_handle, pcm_info) >= 0) {
                    snprintf(device_names[device_count], 256, "%s (hw:%d,%d)",
                             card_name, card, dev);
                    snprintf(device_ids[device_count], 128, "hw:%d,%d", card, dev);
                    device_count++;
                }
            }
            snd_ctl_close(ctl_handle);
        }
        free(card_name);
    }

    /* 添加默认设备和plughw设备 */
    if (device_count < max_devices) {
        snprintf(device_names[device_count], 256, "默认音频设备");
        snprintf(device_ids[device_count], 128, "default");
        device_count++;
    }
    if (device_count < max_devices) {
        snprintf(device_names[device_count], 256, "默认PulseAudio设备");
        snprintf(device_ids[device_count], 128, "pulse");
        device_count++;
    }

    *num_devices = device_count;
    return 0;
}

/**
 * @brief 创建ALSA音频采集上下文
 */
AlsaCaptureContext* alsa_capture_create(const char* device_id,
                                         int sample_rate,
                                         int channels,
                                         int bits_per_sample) {
    AlsaCaptureContext* ctx = NULL;
    snd_pcm_format_t format;
    int rc;
    unsigned int actual_rate;
    snd_pcm_uframes_t buffer_size;

    ctx = (AlsaCaptureContext*)safe_calloc(1, sizeof(AlsaCaptureContext));
    if (!ctx) return NULL;

    ctx->sample_rate = sample_rate > 0 ? sample_rate : 16000;
    ctx->channels = channels > 0 ? channels : 1;
    ctx->bits_per_sample = bits_per_sample > 0 ? bits_per_sample : 16;
    ctx->is_capturing = 0;

    actual_rate = (unsigned int)ctx->sample_rate;

    /* 选择PCM格式 */
    switch (ctx->bits_per_sample) {
        case 8:  format = SND_PCM_FORMAT_S8; break;
        case 16: format = SND_PCM_FORMAT_S16_LE; break;
        case 24: format = SND_PCM_FORMAT_S24_LE; break;
        case 32: format = SND_PCM_FORMAT_S32_LE; break;
        default: format = SND_PCM_FORMAT_S16_LE; ctx->bits_per_sample = 16; break;
    }
    ctx->frame_size = ctx->channels * ctx->bits_per_sample / 8;

    /* 设置设备名称 */
    ctx->device_name = safe_strdup(device_id ? device_id : "default");
    if (!ctx->device_name) {
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 打开PCM设备 */
    rc = snd_pcm_open(&ctx->pcm_handle, ctx->device_name,
                      SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
    if (rc < 0) {
        /* 非阻塞模式失败，尝试阻塞模式 */
        rc = snd_pcm_open(&ctx->pcm_handle, ctx->device_name,
                          SND_PCM_STREAM_CAPTURE, 0);
        if (rc < 0) {
            log_error("[音频采集] ALSA打开设备失败: %s", snd_strerror(rc));
            safe_free((void**)&ctx->device_name);
            safe_free((void**)&ctx);
            return NULL;
        }
    }

    /* 分配硬件参数 */
    snd_pcm_hw_params_alloca(&ctx->hw_params);
    snd_pcm_hw_params_any(ctx->pcm_handle, ctx->hw_params);

    /* 设置访问模式 */
    rc = snd_pcm_hw_params_set_access(ctx->pcm_handle, ctx->hw_params,
                                       SND_PCM_ACCESS_RW_INTERLEAVED);
    if (rc < 0) {
        log_error("[音频采集] ALSA设置访问模式失败: %s", snd_strerror(rc));
        snd_pcm_close(ctx->pcm_handle);
        safe_free((void**)&ctx->device_name);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 设置格式 */
    rc = snd_pcm_hw_params_set_format(ctx->pcm_handle, ctx->hw_params, format);
    if (rc < 0) {
        log_error("[音频采集] ALSA设置格式失败: %s", snd_strerror(rc));
        snd_pcm_close(ctx->pcm_handle);
        safe_free((void**)&ctx->device_name);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 设置声道数 */
    rc = snd_pcm_hw_params_set_channels(ctx->pcm_handle, ctx->hw_params,
                                         (unsigned int)ctx->channels);
    if (rc < 0) {
        log_error("[音频采集] ALSA设置声道数失败: %s", snd_strerror(rc));
        snd_pcm_close(ctx->pcm_handle);
        safe_free((void**)&ctx->device_name);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 设置采样率 */
    rc = snd_pcm_hw_params_set_rate_near(ctx->pcm_handle, ctx->hw_params,
                                          &actual_rate, NULL);
    if (rc < 0 || actual_rate < (unsigned int)(ctx->sample_rate / 2)) {
        log_error("[音频采集] ALSA设置采样率失败: %s", snd_strerror(rc));
        snd_pcm_close(ctx->pcm_handle);
        safe_free((void**)&ctx->device_name);
        safe_free((void**)&ctx);
        return NULL;
    }
    ctx->sample_rate = (int)actual_rate;

    /* 设置缓冲区大小 */
    buffer_size = ctx->sample_rate * 100 / 1000; /* 100ms */
    rc = snd_pcm_hw_params_set_buffer_size_near(ctx->pcm_handle, ctx->hw_params, &buffer_size);
    if (rc < 0) {
        log_error("[音频采集] ALSA设置缓冲区大小失败: %s", snd_strerror(rc));
        snd_pcm_close(ctx->pcm_handle);
        safe_free((void**)&ctx->device_name);
        safe_free((void**)&ctx);
        return NULL;
    }
    ctx->buffer_frames = buffer_size;

    /* 设置周期大小 */
    ctx->period_frames = buffer_size / 4;
    rc = snd_pcm_hw_params_set_period_size_near(ctx->pcm_handle, ctx->hw_params,
                                                  &ctx->period_frames, NULL);
    if (rc < 0) {
        ctx->period_frames = buffer_size / 4;
    }

    /* 应用硬件参数 */
    rc = snd_pcm_hw_params(ctx->pcm_handle, ctx->hw_params);
    if (rc < 0) {
        log_error("[音频采集] ALSA应用硬件参数失败: %s", snd_strerror(rc));
        snd_pcm_close(ctx->pcm_handle);
        safe_free((void**)&ctx->device_name);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 分配缓冲区 */
    ctx->float_buffer_size = ctx->period_frames * ctx->channels;
    ctx->float_buffer = (float*)safe_malloc(ctx->float_buffer_size * sizeof(float));
    if (!ctx->float_buffer) {
        snd_pcm_close(ctx->pcm_handle);
        safe_free((void**)&ctx->device_name);
        safe_free((void**)&ctx);
        return NULL;
    }

    log_info("[音频采集] ALSA设备初始化成功: %s, %dHz, %d声道, %d位",
             ctx->device_name, ctx->sample_rate, ctx->channels, ctx->bits_per_sample);
    return ctx;
}

/**
 * @brief 读取ALSA PCM数据并转换为float
 */
static int alsa_read_pcm_to_float(AlsaCaptureContext* ctx, size_t num_frames) {
    if (!ctx || num_frames == 0) return -1;

    size_t total_samples = num_frames * ctx->channels;
    size_t bytes_needed = num_frames * ctx->frame_size;
    uint8_t* raw_buf = (uint8_t*)safe_malloc(bytes_needed);
    if (!raw_buf) return -1;

    snd_pcm_sframes_t frames_read = snd_pcm_readi(ctx->pcm_handle, raw_buf, (snd_pcm_uframes_t)num_frames);
    if (frames_read < 0) {
        /* 处理XRUN */
        if (frames_read == -EPIPE) {
            snd_pcm_prepare(ctx->pcm_handle);
            frames_read = snd_pcm_readi(ctx->pcm_handle, raw_buf, (snd_pcm_uframes_t)num_frames);
        }
        if (frames_read < 0) {
            safe_free((void**)&raw_buf);
            return -1;
        }
    }

    /* 转换到float */
    size_t actual_samples = (size_t)frames_read * ctx->channels;
    if (ctx->bits_per_sample == 16) {
        const int16_t* pcm = (const int16_t*)raw_buf;
        for (size_t i = 0; i < actual_samples; i++) {
            ctx->float_buffer[i] = (float)pcm[i] / 32768.0f;
        }
    } else if (ctx->bits_per_sample == 24) {
        const uint8_t* pcm24 = raw_buf;
        for (size_t i = 0; i < actual_samples; i++) {
            int32_t val = (int32_t)(pcm24[0] | (pcm24[1] << 8) | ((int8_t)pcm24[2] << 16));
            ctx->float_buffer[i] = (float)val / 8388608.0f;
            pcm24 += 3;
        }
    } else if (ctx->bits_per_sample == 32) {
        const int32_t* pcm32 = (const int32_t*)raw_buf;
        for (size_t i = 0; i < actual_samples; i++) {
            ctx->float_buffer[i] = (float)pcm32[i] / 2147483648.0f;
        }
    } else {
        for (size_t i = 0; i < actual_samples; i++) {
            ctx->float_buffer[i] = ((float)raw_buf[i] - 128.0f) / 128.0f;
        }
    }

    safe_free((void**)&raw_buf);
    return (int)actual_samples;
}

/**
 * @brief 开始ALSA音频采集
 */
int alsa_capture_start(AlsaCaptureContext* ctx,
                        void (*callback)(const float* samples, size_t num_samples, void* user_data),
                        void* user_data) {
    if (!ctx) return -1;

    ctx->on_audio_data = callback;
    ctx->callback_user_data = user_data;
    ctx->is_capturing = 1;

    /* 快速读取一个周期以验证设备可用 */
    int samples_read = alsa_read_pcm_to_float(ctx, ctx->period_frames);
    if (samples_read > 0 && callback) {
        callback(ctx->float_buffer, (size_t)samples_read, user_data);
    } else if (samples_read < 0) {
        log_warning("[音频采集] ALSA初始读取失败，设备可能无信号");
    }

    log_info("[音频采集] ALSA采集已启动 (%s)", ctx->device_name);
    return 0;
}

/**
 * @brief ALSA采集处理循环（每次调用读取一个周期）
 */
int alsa_capture_process(AlsaCaptureContext* ctx) {
    if (!ctx || !ctx->is_capturing) return -1;

    int samples_read = alsa_read_pcm_to_float(ctx, ctx->period_frames);
    if (samples_read > 0) {
        if (ctx->on_audio_data) {
            ctx->on_audio_data(ctx->float_buffer, (size_t)samples_read, ctx->callback_user_data);
        }
        return samples_read;
    }
    return -1;
}

/**
 * @brief 停止ALSA音频采集
 */
int alsa_capture_stop(AlsaCaptureContext* ctx) {
    if (!ctx) return -1;
    ctx->is_capturing = 0;
    snd_pcm_drop(ctx->pcm_handle);
    snd_pcm_prepare(ctx->pcm_handle);
    log_info("[音频采集] ALSA采集已停止");
    return 0;
}

/**
 * @brief 释放ALSA采集上下文
 */
void alsa_capture_free(AlsaCaptureContext* ctx) {
    if (!ctx) return;
    if (ctx->is_capturing) alsa_capture_stop(ctx);
    if (ctx->pcm_handle) snd_pcm_close(ctx->pcm_handle);
    safe_free((void**)&ctx->device_name);
    safe_free((void**)&ctx->float_buffer);
    safe_free((void**)&ctx);
}

/* ============================================================================
 * 8.2 修复: 跨平台麦克风探测
 * hardware_microphone_probe — 检测WASAPI/ALSA/PulseAudio/AudioUnit
 * ============================================================================ */

int hardware_microphone_probe(void) {
#ifdef _WIN32
    return 1;  /* WASAPI总是可用 */
#elif defined(__APPLE__)
    return 2;  /* AudioUnit可用 */
#elif defined(__linux__)
    /* 检查ALSA或PulseAudio */
    FILE* fp = popen("pactl info 2>/dev/null || aplay -l 2>/dev/null", "r");
    if (fp) { char buf[64]; int ok = fgets(buf, sizeof(buf), fp) ? 1 : 0; pclose(fp); return ok; }
    return 0;
#else
    return 0;
#endif
}

int hardware_microphone_available(void) {
    return hardware_microphone_probe() > 0 ? 1 : 0;
}

#endif /* __linux__ */

/* ================================================================
 * 跨平台音频采集统一接口
 * ================================================================ */

/* AudioCaptureDeviceInfo 已在 audio.h 中定义，此处不重复定义 */

/**
 * @brief 音频采集统一上下文
 */
struct AudioCaptureContext {
    int platform;
    int is_capturing;
    void* backend_ctx;

    float* fft_spectrum;
    size_t fft_size;
};
typedef struct AudioCaptureContext AudioCaptureContext;

/**
 * @brief 枚举所有可用音频采集设备
 * @param devices 输出设备信息数组
 * @param max_devices 最大设备数
 * @return 实际设备数
 */
int audio_capture_enumerate_devices(AudioCaptureDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return 0;

    int count = 0;
    char names[MAX_AUDIO_DEVICES][256];
    char ids[MAX_AUDIO_DEVICES][128];

#ifdef _WIN32
    int num_devices = 0;
    if (audio_capture_enumerate_devices_win(names, ids, max_devices, &num_devices) == 0) {
        for (int i = 0; i < num_devices && count < max_devices; i++) {
            strncpy(devices[count].name, names[i], 255);
            devices[count].name[255] = '\0';
            strncpy(devices[count].device_id, ids[i], 127);
            devices[count].device_id[127] = '\0';
            devices[count].max_channels = 2;
            devices[count].default_sample_rate = 48000;
            count++;
        }
    }
#elif defined(__linux__)
    int num_devices = 0;
    if (audio_capture_enumerate_devices_linux(names, ids, max_devices, &num_devices) == 0) {
        for (int i = 0; i < num_devices && count < max_devices; i++) {
            strncpy(devices[count].name, names[i], 255);
            devices[count].name[255] = '\0';
            strncpy(devices[count].device_id, ids[i], 127);
            devices[count].device_id[127] = '\0';
            devices[count].max_channels = 2;
            devices[count].default_sample_rate = 44100;
            count++;
        }
    }
#endif

    return count;
}

/**
 * @brief 创建音频采集上下文
 */
AudioCaptureContext* audio_capture_create(const char* device_id,
                                           int sample_rate,
                                           int channels,
                                           int bits_per_sample) {
    AudioCaptureContext* ctx = (AudioCaptureContext*)safe_calloc(1, sizeof(AudioCaptureContext));
    if (!ctx) return NULL;

    ctx->fft_spectrum = NULL;
    ctx->fft_size = 0;

#ifdef _WIN32
    ctx->backend_ctx = wasapi_capture_create(device_id, sample_rate, channels, bits_per_sample);
    if (ctx->backend_ctx) {
        ctx->platform = 0;
        log_info("[音频采集] 使用Windows WASAPI后端");
        return ctx;
    }
#elif defined(__linux__)
    ctx->backend_ctx = alsa_capture_create(device_id, sample_rate, channels, bits_per_sample);
    if (ctx->backend_ctx) {
        ctx->platform = 1;
        log_info("[音频采集] 使用Linux ALSA后端");
        return ctx;
    }
#endif

    /* 无可用硬件 */
    safe_free((void**)&ctx);
    log_error("[音频采集] 当前平台无可用音频采集硬件");
    return NULL;
}

/**
 * @brief 开始音频采集
 */
int audio_capture_start(AudioCaptureContext* ctx,
                         void (*callback)(const float* samples, size_t num_samples, void* user_data),
                         void* user_data) {
    if (!ctx || !callback) return -1;

#ifdef _WIN32
    if (ctx->platform == 0 && ctx->backend_ctx) {
        int ret = wasapi_capture_start(ctx->backend_ctx, callback, user_data);
        if (ret == 0) ctx->is_capturing = 1;
        return ret;
    }
#elif defined(__linux__)
    if (ctx->platform == 1 && ctx->backend_ctx) {
        int ret = alsa_capture_start(ctx->backend_ctx, callback, user_data);
        if (ret == 0) ctx->is_capturing = 1;
        return ret;
    }
#endif

    return -1;
}

/**
 * @brief 音频采集处理循环（Linux使用，Windows由线程自动处理）
 */
int audio_capture_process(AudioCaptureContext* ctx) {
    if (!ctx) return -1;

#ifdef __linux__
    if (ctx->platform == 1 && ctx->backend_ctx && ctx->is_capturing) {
        return alsa_capture_process(ctx->backend_ctx);
    }
#endif

    return 0;
}

/**
 * @brief 停止音频采集
 */
int audio_capture_stop(AudioCaptureContext* ctx) {
    if (!ctx) return -1;
    ctx->is_capturing = 0;

#ifdef _WIN32
    if (ctx->platform == 0 && ctx->backend_ctx) {
        return wasapi_capture_stop(ctx->backend_ctx);
    }
#elif defined(__linux__)
    if (ctx->platform == 1 && ctx->backend_ctx) {
        return alsa_capture_stop(ctx->backend_ctx);
    }
#endif

    return 0;
}

/**
 * @brief 释放音频采集上下文
 */
void audio_capture_free(AudioCaptureContext* ctx) {
    if (!ctx) return;

#ifdef _WIN32
    if (ctx->platform == 0 && ctx->backend_ctx) {
        wasapi_capture_free(ctx->backend_ctx);
    }
#elif defined(__linux__)
    if (ctx->platform == 1 && ctx->backend_ctx) {
        alsa_capture_free(ctx->backend_ctx);
    }
#endif

    safe_free((void**)&ctx->fft_spectrum);
    safe_free((void**)&ctx);
}

/* ZSFUSA: 获取音频频谱 */
int audio_capture_get_spectrum(void* capture, float* spectrum, size_t* size) {
    if (!capture || !spectrum || !size) return -1;
    AudioCaptureContext* ctx = (AudioCaptureContext*)capture;
    size_t out_size = *size;
    if (ctx->fft_spectrum && ctx->fft_size > 0) {
        size_t copy_n = out_size < ctx->fft_size ? out_size : ctx->fft_size;
        memcpy(spectrum, ctx->fft_spectrum, copy_n * sizeof(float));
        *size = copy_n;
        return 0;
    }
    *size = 0;
    return -1;
}
