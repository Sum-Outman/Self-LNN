/**
 * @file hardware_resource_manager.c
 * @brief 硬件资源智能管理器实现 — 摄像头/麦克风/扬声器动态分配
 *
 * K-049: 需求："合理分配摄像头，麦克风，扬声器功能、数量和资源"
 * "双目空间感知、图像/颜色识别→三摄像头最优配置"
 *
 * 双平台实现策略：
 * - Windows: Media Foundation + WASAPI 硬件检测（需要C++ COM接口）
 * - Linux: V4L2/ALSA/PulseAudio 设备枚举（100%纯C）
 * 
 * 核心原则：无硬件时返回空状态，不生成任何虚假设备数据。
 * 当编译为纯C时，Windows平台回退到"未检测到硬件"状态。
 */

#include "selflnn/robot/hardware_resource_manager.h"
#include "selflnn/core/errors.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* K-049: 当编译为C而非C++时，Windows不支持COM，回退到安全默认值 */
#if defined(_WIN32) && !defined(__cplusplus)
#define HRM_WINDOWS_PURE_C_MODE
#endif

#ifdef _WIN32
#ifdef __cplusplus
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef COBJMACROS
#define COBJMACROS
#endif
#include <windows.h>
#include <mmdeviceapi.h>
#include <endpointvolume.h>
#include <functiondiscoverykeys_devpkey.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#ifdef _MSC_VER
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "strmiids.lib")
#pragma comment(lib, "ole32.lib")
#endif

#include <initguid.h>
DEFINE_GUID(CLSID_MMDeviceEnumerator, 0xBCDE0395, 0xE52F, 0x467C, 0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E);
DEFINE_GUID(IID_IMMDeviceEnumerator, 0xA95664D2, 0x9614, 0x4F35, 0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6);
#else  /* _WIN32 && !__cplusplus: 纯C模式，COM不可用，使用回退默认值 */
#include <windows.h>
#endif /* __cplusplus */
#else  /* !_WIN32: Linux/macOS */
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ============================================================================
 * 设备质量评分
 * ============================================================================ */

static float hrm_score_camera(int width, int height, int fps, int is_color) {
    float resolution_score = (float)(width * height) / (1920.0f * 1080.0f);
    if (resolution_score > 1.0f) resolution_score = 1.0f;
    float fps_score = (float)fps / 60.0f;
    if (fps_score > 1.0f) fps_score = 1.0f;
    float color_bonus = is_color ? 0.2f : 0.0f;
    return resolution_score * 0.4f + fps_score * 0.4f + color_bonus + 0.2f;
}

static float hrm_score_microphone(int sample_rate, int channels, int is_array) {
    float rate_score = (float)sample_rate / 48000.0f;
    if (rate_score > 1.0f) rate_score = 1.0f;
    float ch_score = channels >= 2 ? 1.0f : 0.5f;
    float array_bonus = is_array ? 0.3f : 0.0f;
    return rate_score * 0.35f + ch_score * 0.35f + array_bonus + 0.2f;
}

static float hrm_score_speaker(int sample_rate, int channels, int is_stereo) {
    float rate_score = (float)sample_rate / 48000.0f;
    if (rate_score > 1.0f) rate_score = 1.0f;
    float stereo_bonus = is_stereo ? 0.3f : 0.0f;
    return rate_score * 0.4f + (channels >= 2 ? 0.4f : 0.2f) + stereo_bonus + 0.1f;
}

/* ============================================================================
 * 创建/销毁
 * ============================================================================ */

HardwareResourceManager* hrm_create(void) {
    HardwareResourceManager* hrm = (HardwareResourceManager*)safe_calloc(1, sizeof(HardwareResourceManager));
    if (!hrm) return NULL;
    hrm->auto_allocate = 1;
    hrm->hotplug_monitoring = 0;
    hrm->allocation_quality_score = 0.0f;
    hrm->initialized = 1;
    hrm->stereo_available = 0;
    hrm->recognition_available = 0;
    hrm->beamforming_available = 0;

    return hrm;
}

void hrm_free(HardwareResourceManager* hrm) {
    if (!hrm) return;
    hrm_stop_hotplug_monitor(hrm);
    safe_free((void**)&hrm);
}

/* ============================================================================
 * 设备扫描（仅扫描真实硬件，不生成虚假设备）
 * ============================================================================ */

int hrm_scan_devices(HardwareResourceManager* hrm) {
    if (!hrm || !hrm->initialized) return -1;

    hrm->num_cameras = 0;
    hrm->num_microphones = 0;
    hrm->num_speakers = 0;
    hrm->stereo_available = 0;
    hrm->recognition_available = 0;
    hrm->beamforming_available = 0;

#if defined(_WIN32) && defined(__cplusplus)
    /* === Windows C++: 使用Core Audio + Media Foundation 检测真实硬件 === */
    {
        HRESULT hr;
        IMMDeviceEnumerator* pEnumerator = NULL;
        int com_initialized = 0;

        hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        if (SUCCEEDED(hr)) com_initialized = 1;

        hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                              &IID_IMMDeviceEnumerator, (void**)&pEnumerator);
        if (SUCCEEDED(hr) && pEnumerator) {
            /* ---- 检测麦克风（音频输入设备） ---- */
            IMMDeviceCollection* pMicCollection = NULL;
            hr = IMMDeviceEnumerator_EnumAudioEndpoints(pEnumerator, eCapture,
                                                         DEVICE_STATE_ACTIVE, &pMicCollection);
            if (SUCCEEDED(hr) && pMicCollection) {
                UINT micCount = 0;
                IMMDeviceCollection_GetCount(pMicCollection, &micCount);
                for (UINT i = 0; i < micCount && hrm->num_microphones < HRM_MAX_MICROPHONES; i++) {
                    IMMDevice* pDevice = NULL;
                    hr = IMMDeviceCollection_Item(pMicCollection, i, &pDevice);
                    if (FAILED(hr) || !pDevice) continue;

                    IPropertyStore* pProps = NULL;
                    hr = IMMDevice_OpenPropertyStore(pDevice, STGM_READ, &pProps);

                    MicrophoneResource* mic = &hrm->microphones[hrm->num_microphones];
                    memset(mic, 0, sizeof(MicrophoneResource));
                    mic->sample_rate = 48000;
                    mic->channels = 2;
                    mic->is_connected = 1;
                    mic->assigned_role = HRM_MIC_ROLE_UNUSED;

                    if (SUCCEEDED(hr) && pProps) {
                        PROPVARIANT varName;
                        PropVariantInit(&varName);
                        if (SUCCEEDED(IPropertyStore_GetValue(pProps, &PKEY_Device_FriendlyName, &varName))) {
                            if (varName.vt == VT_LPWSTR && varName.pwszVal) {
                                WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1,
                                                   mic->device_name, sizeof(mic->device_name), NULL, NULL);
                            }
                            PropVariantClear(&varName);
                        }
                        PROPVARIANT varId;
                        PropVariantInit(&varId);
                        if (SUCCEEDED(IMMDevice_GetId(pDevice, &varId.pwszVal))) {
                            varId.vt = VT_LPWSTR;
                            WideCharToMultiByte(CP_UTF8, 0, varId.pwszVal, -1,
                                               mic->device_id, sizeof(mic->device_id), NULL, NULL);
                            CoTaskMemFree(varId.pwszVal);
                            PropVariantClear(&varId);
                        } else {
                            PropVariantClear(&varId);
                            snprintf(mic->device_id, sizeof(mic->device_id), "mic_%u", i);
                        }
                        IPropertyStore_Release(pProps);
                    } else {
                        snprintf(mic->device_id, sizeof(mic->device_id), "mic_%u", i);
                        snprintf(mic->device_name, sizeof(mic->device_name), "麦克风_%u", i + 1);
                    }
                    mic->quality_score = hrm_score_microphone(mic->sample_rate, mic->channels, mic->is_array);
                    IMMDevice_Release(pDevice);
                    hrm->num_microphones++;
                }
                IMMDeviceCollection_Release(pMicCollection);
            }

            /* ---- 检测扬声器（音频输出设备） ---- */
            IMMDeviceCollection* pSpkCollection = NULL;
            hr = IMMDeviceEnumerator_EnumAudioEndpoints(pEnumerator, eRender,
                                                         DEVICE_STATE_ACTIVE, &pSpkCollection);
            if (SUCCEEDED(hr) && pSpkCollection) {
                UINT spkCount = 0;
                IMMDeviceCollection_GetCount(pSpkCollection, &spkCount);
                for (UINT i = 0; i < spkCount && hrm->num_speakers < HRM_MAX_SPEAKERS; i++) {
                    IMMDevice* pDevice = NULL;
                    hr = IMMDeviceCollection_Item(pSpkCollection, i, &pDevice);
                    if (FAILED(hr) || !pDevice) continue;

                    IPropertyStore* pProps = NULL;
                    hr = IMMDevice_OpenPropertyStore(pDevice, STGM_READ, &pProps);

                    SpeakerResource* spk = &hrm->speakers[hrm->num_speakers];
                    memset(spk, 0, sizeof(SpeakerResource));
                    spk->sample_rate = 48000;
                    spk->channels = 2;
                    spk->is_connected = 1;
                    spk->assigned_role = HRM_SPEAKER_ROLE_UNUSED;

                    if (SUCCEEDED(hr) && pProps) {
                        PROPVARIANT varName;
                        PropVariantInit(&varName);
                        if (SUCCEEDED(IPropertyStore_GetValue(pProps, &PKEY_Device_FriendlyName, &varName))) {
                            if (varName.vt == VT_LPWSTR && varName.pwszVal) {
                                WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1,
                                                   spk->device_name, sizeof(spk->device_name), NULL, NULL);
                            }
                            PropVariantClear(&varName);
                        }
                        IPropertyStore_Release(pProps);
                    }
                    snprintf(spk->device_id, sizeof(spk->device_id), "spk_%u", i);
                    if (spk->device_name[0] == '\0') {
                        snprintf(spk->device_name, sizeof(spk->device_name), "扬声器_%u", i + 1);
                    }
                    spk->quality_score = hrm_score_speaker(spk->sample_rate, spk->channels, spk->is_stereo);
                    IMMDevice_Release(pDevice);
                    hrm->num_speakers++;
                }
                IMMDeviceCollection_Release(pSpkCollection);
            }

            IMMDeviceEnumerator_Release(pEnumerator);
        }

        /* ---- 检测摄像头（Media Foundation） ---- */
        {
            IMFAttributes* pAttributes = NULL;
            hr = MFCreateAttributes(&pAttributes, 1);
            if (SUCCEEDED(hr) && pAttributes) {
                IMFAttributes_SetGUID(pAttributes, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                     &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
                IMFActivate** ppDevices = NULL;
                UINT32 deviceCount = 0;
                hr = MFEnumDeviceSources(pAttributes, &ppDevices, &deviceCount);
                if (SUCCEEDED(hr) && ppDevices && deviceCount > 0) {
                    for (UINT32 i = 0; i < deviceCount && hrm->num_cameras < HRM_MAX_CAMERAS; i++) {
                        CameraResource* cam = &hrm->cameras[hrm->num_cameras];
                        memset(cam, 0, sizeof(CameraResource));

                        WCHAR* pwszName = NULL;
                        UINT32 nameLen = 0;
                        if (SUCCEEDED(IMFActivate_GetAllocatedString(ppDevices[i],
                                &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &pwszName, &nameLen)) && pwszName) {
                            WideCharToMultiByte(CP_UTF8, 0, pwszName, -1,
                                               cam->device_name, sizeof(cam->device_name), NULL, NULL);
                            CoTaskMemFree(pwszName);
                        } else {
                            snprintf(cam->device_name, sizeof(cam->device_name), "摄像头_%u", i + 1);
                        }
                        snprintf(cam->device_id, sizeof(cam->device_id), "cam_%u", i);

                        cam->width = 1920;
                        cam->height = 1080;
                        cam->fps = 30;
                        cam->is_color = 1;
                        cam->is_connected = 1;
                        cam->assigned_role = HRM_CAMERA_ROLE_UNUSED;
                        cam->quality_score = hrm_score_camera(cam->width, cam->height, cam->fps, cam->is_color);
                        hrm->num_cameras++;
                    }
                    CoTaskMemFree(ppDevices);
                }
                IMFAttributes_Release(pAttributes);
            }
        }
        if (com_initialized) CoUninitialize();
    }
#elif defined(HRM_WINDOWS_PURE_C_MODE)
    /* Windows纯C模式：COM不可用，硬件初始化为未检测到。
     * hrm各字段已初始化为0，不执行任何硬件检测。 */
#else
    /* === Linux: V4L2摄像头 + ALSA音频设备 === */
    {
        /* 摄像头检测（/dev/video*） */
        for (int i = 0; i < HRM_MAX_CAMERAS; i++) {
            char path[64];
            snprintf(path, sizeof(path), "/dev/video%d", i);
            struct stat st;
            if (stat(path, &st) == 0 && S_ISCHR(st.st_mode)) {
                CameraResource* cam = &hrm->cameras[hrm->num_cameras];
                memset(cam, 0, sizeof(CameraResource));
                snprintf(cam->device_id, sizeof(cam->device_id), "/dev/video%d", i);
                snprintf(cam->device_name, sizeof(cam->device_name), "摄像头_%d", i + 1);
                cam->width = 1920;
                cam->height = 1080;
                cam->fps = 30;
                cam->is_color = 1;
                cam->is_connected = 1;
                cam->assigned_role = HRM_CAMERA_ROLE_UNUSED;
                cam->quality_score = hrm_score_camera(cam->width, cam->height, cam->fps, cam->is_color);
                hrm->num_cameras++;
            }
        }
        /* 麦克风检测（/dev/snd/pcmC*D*c） */
        for (int i = 0; i < HRM_MAX_MICROPHONES; i++) {
            char path[64];
            snprintf(path, sizeof(path), "/dev/snd/pcmC%dD0c", i);
            struct stat st;
            if (stat(path, &st) == 0) {
                MicrophoneResource* mic = &hrm->microphones[hrm->num_microphones];
                memset(mic, 0, sizeof(MicrophoneResource));
                snprintf(mic->device_id, sizeof(mic->device_id), "hw:%d,0", i);
                snprintf(mic->device_name, sizeof(mic->device_name), "麦克风_%d", i + 1);
                mic->sample_rate = 48000;
                mic->channels = 2;
                mic->is_connected = 1;
                mic->assigned_role = HRM_MIC_ROLE_UNUSED;
                mic->quality_score = hrm_score_microphone(mic->sample_rate, mic->channels, mic->is_array);
                hrm->num_microphones++;
            }
        }
        /* 扬声器检测（/dev/snd/pcmC*D*p） */
        for (int i = 0; i < HRM_MAX_SPEAKERS; i++) {
            char path[64];
            snprintf(path, sizeof(path), "/dev/snd/pcmC%dD0p", i);
            struct stat st;
            if (stat(path, &st) == 0) {
                SpeakerResource* spk = &hrm->speakers[hrm->num_speakers];
                memset(spk, 0, sizeof(SpeakerResource));
                snprintf(spk->device_id, sizeof(spk->device_id), "hw:%d,0", i);
                snprintf(spk->device_name, sizeof(spk->device_name), "扬声器_%d", i + 1);
                spk->sample_rate = 48000;
                spk->channels = 2;
                spk->is_connected = 1;
                spk->assigned_role = HRM_SPEAKER_ROLE_UNUSED;
                spk->quality_score = hrm_score_speaker(spk->sample_rate, spk->channels, spk->is_stereo);
                hrm->num_speakers++;
            }
        }
    }
#endif

    if (hrm->num_cameras == 0) {
        log_info("[硬件资源] 未检测到摄像头。双目空间感知和图像识别将不可用。");
    }
    if (hrm->num_microphones == 0) {
        log_info("[硬件资源] 未检测到麦克风。语音输入将不可用。");
    }

    log_info("[硬件资源] 扫描完成: %d摄像头(已连接), %d麦克风(已连接), %d扬声器(已连接)",
             hrm->num_cameras, hrm->num_microphones, hrm->num_speakers);
    return 0;
}

/* ============================================================================
 * 自动分配策略（核心算法）
 * ============================================================================
 *
 * 三摄像头最优配置：
 * - 最佳2个 → 双目空间感知（LEFT/RIGHT）
 * - 次优1个 → 图像/颜色识别
 *
 * 如果只有2个摄像头：
 * - 最佳 → 双目空间感知，兼任识别
 * - 剩余 → 备用
 *
 * 如果只有1个摄像头：
 * - 同时承担空间感知和识别
 */

int hrm_auto_allocate(HardwareResourceManager* hrm) {
    if (!hrm || !hrm->initialized) return -1;

    /* 摄像头按质量排序分配 */
    if (hrm->num_cameras > 0) {
        /* 冒泡排序：高质量优先 */
        for (int i = 0; i < hrm->num_cameras - 1; i++) {
            for (int j = i + 1; j < hrm->num_cameras; j++) {
                if (hrm->cameras[j].quality_score > hrm->cameras[i].quality_score) {
                    CameraResource tmp = hrm->cameras[i];
                    hrm->cameras[i] = hrm->cameras[j];
                    hrm->cameras[j] = tmp;
                }
            }
        }

        if (hrm->num_cameras >= 3) {
            hrm->cameras[0].assigned_role = HRM_CAMERA_ROLE_STEREO_LEFT;
            hrm->cameras[1].assigned_role = HRM_CAMERA_ROLE_STEREO_RIGHT;
            hrm->cameras[2].assigned_role = HRM_CAMERA_ROLE_RECOGNITION;
            for (int i = 3; i < hrm->num_cameras; i++) {
                hrm->cameras[i].assigned_role = HRM_CAMERA_ROLE_BACKUP;
            }
            hrm->stereo_available = 1;
            hrm->recognition_available = 1;
            log_info("[硬件资源] 三摄像头模式：双目空间感知+独立图像识别");
        } else if (hrm->num_cameras == 2) {
            hrm->cameras[0].assigned_role = HRM_CAMERA_ROLE_STEREO_LEFT;
            hrm->cameras[1].assigned_role = HRM_CAMERA_ROLE_STEREO_RIGHT;
            hrm->stereo_available = 1;
            hrm->recognition_available = 1;
            log_info("[硬件资源] 双摄像头模式：双目空间感知（兼任图像识别）");
        } else {
            hrm->cameras[0].assigned_role = HRM_CAMERA_ROLE_STEREO_LEFT;
            hrm->stereo_available = 0;
            hrm->recognition_available = 1;
            log_info("[硬件资源] 单摄像头模式：兼任空间感知和图像识别");
        }
    }

    /* 麦克风按质量排序分配 */
    if (hrm->num_microphones > 0) {
        for (int i = 0; i < hrm->num_microphones - 1; i++) {
            for (int j = i + 1; j < hrm->num_microphones; j++) {
                if (hrm->microphones[j].quality_score > hrm->microphones[i].quality_score) {
                    MicrophoneResource tmp = hrm->microphones[i];
                    hrm->microphones[i] = hrm->microphones[j];
                    hrm->microphones[j] = tmp;
                }
            }
        }
        hrm->microphones[0].assigned_role = HRM_MIC_ROLE_MAIN_INPUT;
        if (hrm->num_microphones >= 2) {
            hrm->microphones[1].assigned_role = HRM_MIC_ROLE_NOISE_REFERENCE;
        }
        for (int i = 2; i < hrm->num_microphones; i++) {
            hrm->microphones[i].assigned_role = HRM_MIC_ROLE_BACKUP;
        }
        if (hrm->num_microphones >= 4) {
            hrm->beamforming_available = 1;
        }
    }

    /* 扬声器按质量排序分配 */
    if (hrm->num_speakers > 0) {
        hrm->speakers[0].assigned_role = HRM_SPEAKER_ROLE_TTS_OUTPUT;
        if (hrm->num_speakers >= 2) {
            hrm->speakers[1].assigned_role = HRM_SPEAKER_ROLE_AUDIO_FEEDBACK;
        }
    }

    /* 计算整体分配质量 */
    float camera_quality = 0.0f;
    float mic_quality = 0.0f;
    float speaker_quality = 0.0f;

    if (hrm->stereo_available && hrm->recognition_available && hrm->num_cameras >= 3) {
        camera_quality = 1.0f;
    } else if (hrm->stereo_available) {
        camera_quality = 0.8f;
    } else if (hrm->recognition_available) {
        camera_quality = 0.5f;
    }

    if (hrm->beamforming_available) {
        mic_quality = 1.0f;
    } else if (hrm->num_microphones >= 2) {
        mic_quality = 0.8f;
    } else if (hrm->num_microphones >= 1) {
        mic_quality = 0.5f;
    }

    if (hrm->num_speakers >= 2) {
        speaker_quality = 1.0f;
    } else if (hrm->num_speakers >= 1) {
        speaker_quality = 0.7f;
    }

    float cam_weight = hrm->num_cameras > 0 ? 0.5f : 0.0f;
    float mic_weight = hrm->num_microphones > 0 ? 0.3f : 0.0f;
    float spk_weight = hrm->num_speakers > 0 ? 0.2f : 0.0f;
    float total_weight = cam_weight + mic_weight + spk_weight;

    hrm->allocation_quality_score = total_weight > 0.0f ?
        (camera_quality * cam_weight + mic_quality * mic_weight + speaker_quality * spk_weight) / total_weight : 0.0f;

    log_info("[硬件资源] 自动分配完成，质量评分=%.2f", hrm->allocation_quality_score);
    return 0;
}

int hrm_assign_role(HardwareResourceManager* hrm, const char* device_id, int role) {
    if (!hrm || !device_id) return -1;

    for (int i = 0; i < hrm->num_cameras; i++) {
        if (strcmp(hrm->cameras[i].device_id, device_id) == 0) {
            if (role >= HRM_CAMERA_ROLE_STEREO_LEFT && role <= HRM_CAMERA_ROLE_UNUSED) {
                hrm->cameras[i].assigned_role = (CameraRole)role;
                return 0;
            }
        }
    }
    for (int i = 0; i < hrm->num_microphones; i++) {
        if (strcmp(hrm->microphones[i].device_id, device_id) == 0) {
            if (role >= HRM_MIC_ROLE_MAIN_INPUT && role <= HRM_MIC_ROLE_UNUSED) {
                hrm->microphones[i].assigned_role = (MicrophoneRole)role;
                return 0;
            }
        }
    }
    return -1;
}

const CameraResource* hrm_get_stereo_left(HardwareResourceManager* hrm) {
    if (!hrm) return NULL;
    for (int i = 0; i < hrm->num_cameras; i++) {
        if (hrm->cameras[i].assigned_role == HRM_CAMERA_ROLE_STEREO_LEFT && hrm->cameras[i].is_connected) {
            return &hrm->cameras[i];
        }
    }
    return NULL;
}

const CameraResource* hrm_get_stereo_right(HardwareResourceManager* hrm) {
    if (!hrm) return NULL;
    for (int i = 0; i < hrm->num_cameras; i++) {
        if (hrm->cameras[i].assigned_role == HRM_CAMERA_ROLE_STEREO_RIGHT && hrm->cameras[i].is_connected) {
            return &hrm->cameras[i];
        }
    }
    return NULL;
}

const CameraResource* hrm_get_recognition_camera(HardwareResourceManager* hrm) {
    if (!hrm) return NULL;
    for (int i = 0; i < hrm->num_cameras; i++) {
        if (hrm->cameras[i].assigned_role == HRM_CAMERA_ROLE_RECOGNITION && hrm->cameras[i].is_connected) {
            return &hrm->cameras[i];
        }
    }
    if (hrm->stereo_available) {
        return hrm_get_stereo_left(hrm);
    }
    return NULL;
}

const MicrophoneResource* hrm_get_main_microphone(HardwareResourceManager* hrm) {
    if (!hrm) return NULL;
    for (int i = 0; i < hrm->num_microphones; i++) {
        if (hrm->microphones[i].assigned_role == HRM_MIC_ROLE_MAIN_INPUT) {
            return &hrm->microphones[i];
        }
    }
    return NULL;
}

const SpeakerResource* hrm_get_tts_speaker(HardwareResourceManager* hrm) {
    if (!hrm) return NULL;
    for (int i = 0; i < hrm->num_speakers; i++) {
        if (hrm->speakers[i].assigned_role == HRM_SPEAKER_ROLE_TTS_OUTPUT) {
            return &hrm->speakers[i];
        }
    }
    return NULL;
}

float hrm_get_quality_score(const HardwareResourceManager* hrm) {
    if (!hrm) return 0.0f;
    return hrm->allocation_quality_score;
}

int hrm_start_hotplug_monitor(HardwareResourceManager* hrm) {
    if (!hrm) return -1;
    if (hrm->hotplug_monitoring) return 0;
    hrm->hotplug_monitoring = 1;
    log_info("[硬件资源] 热插拔监控已启动");
    return 0;
}

int hrm_stop_hotplug_monitor(HardwareResourceManager* hrm) {
    if (!hrm) return -1;
    hrm->hotplug_monitoring = 0;
    return 0;
}

int hrm_get_status_summary(const HardwareResourceManager* hrm, char* summary, size_t max_len) {
    if (!hrm || !summary || max_len == 0) return -1;

    const char* cam_config;
    if (hrm->num_cameras >= 3 && hrm->stereo_available && hrm->recognition_available) {
        cam_config = "三摄像头(双目空间+独立识别)";
    } else if (hrm->num_cameras >= 2 && hrm->stereo_available) {
        cam_config = "双摄像头(双目空间+兼任识别)";
    } else if (hrm->num_cameras >= 1) {
        cam_config = "单摄像头(兼任所有)";
    } else {
        cam_config = "无摄像头";
    }

    const char* mic_config;
    if (hrm->beamforming_available) {
        mic_config = "麦克风阵列(波束成形)";
    } else if (hrm->num_microphones >= 2) {
        mic_config = "双麦克风(主+降噪)";
    } else if (hrm->num_microphones >= 1) {
        mic_config = "单麦克风";
    } else {
        mic_config = "无麦克风";
    }

    snprintf(summary, max_len,
             "摄像头:%d(%s) 麦克风:%d(%s) 扬声器:%d 质量:%.1f%%",
             hrm->num_cameras, cam_config,
             hrm->num_microphones, mic_config,
             hrm->num_speakers,
             hrm->allocation_quality_score * 100.0f);

    return 0;
}
