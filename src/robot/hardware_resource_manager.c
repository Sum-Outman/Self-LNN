/**
 * @file hardware_resource_manager.c
 * @brief 硬件资源管理器 —— 摄像头/麦克风/扬声器智能分配
 *
 * 实现策略：
 * - 三个摄像头自动识别：双目(空间感知) + 单目(图像/颜色识别)
 * - 麦克风阵列分配：主输入 + 环境降噪 + 波束成形
 * - 扬声器分配：TTS输出 + 音频反馈
 * - 资源热插拔检测和自动重分配
 *
 * 通过Windows Media Foundation / Linux V4L2/ALSA检测真实硬件设备。
 */

#include "selflnn/robot/hardware_resource_manager.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <dshow.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>

/* ZSFBUILD: MSVC编译为C模式下，COM GUID定义需显式提供 */
const CLSID CLSID_MMDeviceEnumerator = {0xBCDE0395,0xE52F,0x467C,{0x8E,0x3D,0xC4,0x57,0x92,0x91,0x69,0x2E}};
const IID IID_IMMDeviceEnumerator = {0xA95664D2,0x9614,0x4F35,{0xA7,0x46,0xDE,0x8D,0xB6,0x36,0x17,0xE6}};
#else
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>
#endif

HardwareResourceManager* hrm_create(void) {
    HardwareResourceManager* hrm = (HardwareResourceManager*)safe_calloc(1, sizeof(HardwareResourceManager));
    if (!hrm) return NULL;
    hrm->auto_allocate = 1;
    hrm->hotplug_monitoring = 1;
    hrm->allocation_quality_score = 0.0f;
    hrm->initialized = 0;
    hrm->monitor_thread = NULL;
    hrm->monitor_running = 0;
    log_info("[硬件资源管理器] 已创建");
    return hrm;
}

void hrm_free(HardwareResourceManager* hrm) {
    if (!hrm) return;
    hrm_stop_hotplug_monitor(hrm);
    safe_free((void**)&hrm);
}

int hrm_scan_devices(HardwareResourceManager* hrm) {
    if (!hrm) return -1;
    hrm->num_cameras = 0;
    hrm->num_microphones = 0;
    hrm->num_speakers = 0;
    hrm->stereo_available = 0;
    hrm->recognition_available = 0;
    hrm->beamforming_available = 0;

#ifdef _WIN32
    /* Windows: 使用DirectShow枚举摄像头设备 */
    ICreateDevEnum* pDevEnum = NULL;
    IEnumMoniker* pEnum = NULL;
    IMoniker* pMoniker = NULL;
    HRESULT hr;

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) goto scan_microphones;

    hr = CoCreateInstance(&CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC,
                          &IID_ICreateDevEnum, (void**)&pDevEnum);
    if (FAILED(hr)) goto scan_microphones;

    hr = pDevEnum->lpVtbl->CreateClassEnumerator(pDevEnum, &CLSID_VideoInputDeviceCategory, &pEnum, 0);
    if (hr != S_OK || !pEnum) goto scan_microphones;

    while (pEnum->lpVtbl->Next(pEnum, 1, &pMoniker, NULL) == S_OK && hrm->num_cameras < HRM_MAX_CAMERAS) {
        IPropertyBag* pPropBag = NULL;
        hr = pMoniker->lpVtbl->BindToStorage(pMoniker, NULL, NULL, &IID_IPropertyBag, (void**)&pPropBag);
        if (SUCCEEDED(hr)) {
            VARIANT var;
            VariantInit(&var);
            CameraResource* cam = &hrm->cameras[hrm->num_cameras];
            memset(cam, 0, sizeof(CameraResource));
            cam->is_connected = 1;
            cam->quality_score = 0.8f;

            hr = pPropBag->lpVtbl->Read(pPropBag, L"FriendlyName", &var, NULL);
            if (SUCCEEDED(hr) && var.bstrVal) {
                snprintf(cam->device_name, sizeof(cam->device_name), "%S", var.bstrVal);
                snprintf(cam->device_id, sizeof(cam->device_id), "cam_%d", hrm->num_cameras);
                VariantClear(&var);
            }
            cam->width = 640;
            cam->height = 480;
            cam->fps = 30;
            cam->is_color = 1;
            cam->assigned_role = HRM_CAMERA_ROLE_UNUSED;

            pPropBag->lpVtbl->Release(pPropBag);
            hrm->num_cameras++;
        }
        pMoniker->lpVtbl->Release(pMoniker);
    }

scan_microphones:
    if (pEnum) pEnum->lpVtbl->Release(pEnum);
    if (pDevEnum) pDevEnum->lpVtbl->Release(pDevEnum);

    /* Windows: 使用MMDevice API枚举音频设备 */
    IMMDeviceEnumerator* pEnumerator = NULL;
    IMMDeviceCollection* pCollection = NULL;

    hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &IID_IMMDeviceEnumerator, (void**)&pEnumerator);
    if (SUCCEEDED(hr)) {
        /* 麦克风 (eCapture) */
        hr = pEnumerator->lpVtbl->EnumAudioEndpoints(pEnumerator, eCapture,
                                                       DEVICE_STATE_ACTIVE, &pCollection);
        if (SUCCEEDED(hr)) {
            UINT count = 0;
            pCollection->lpVtbl->GetCount(pCollection, &count);
            for (UINT i = 0; i < count && hrm->num_microphones < HRM_MAX_MICROPHONES; i++) {
                IMMDevice* pDevice = NULL;
                hr = pCollection->lpVtbl->Item(pCollection, i, &pDevice);
                if (SUCCEEDED(hr)) {
                    IPropertyStore* pProps = NULL;
                    hr = pDevice->lpVtbl->OpenPropertyStore(pDevice, STGM_READ, &pProps);
                    if (SUCCEEDED(hr)) {
                        PROPVARIANT varName;
                        PropVariantInit(&varName);
                        MicrophoneResource* mic = &hrm->microphones[hrm->num_microphones];
                        memset(mic, 0, sizeof(MicrophoneResource));
                        snprintf(mic->device_id, sizeof(mic->device_id), "mic_%d", hrm->num_microphones);
                        mic->sample_rate = 44100;
                        mic->channels = 1;
                        mic->is_connected = 1;
                        mic->quality_score = 0.75f;
                        mic->assigned_role = HRM_MIC_ROLE_UNUSED;
                        hrm->num_microphones++;

                        PropVariantClear(&varName);
                        pProps->lpVtbl->Release(pProps);
                    }
                    pDevice->lpVtbl->Release(pDevice);
                }
            }
            pCollection->lpVtbl->Release(pCollection);
        }

        /* 扬声器 (eRender) */
        hr = pEnumerator->lpVtbl->EnumAudioEndpoints(pEnumerator, eRender,
                                                       DEVICE_STATE_ACTIVE, &pCollection);
        if (SUCCEEDED(hr)) {
            UINT count = 0;
            pCollection->lpVtbl->GetCount(pCollection, &count);
            for (UINT i = 0; i < count && hrm->num_speakers < HRM_MAX_SPEAKERS; i++) {
                char device_name[256];
                snprintf(device_name, sizeof(device_name), "扬声器_%u", i + 1);
                SpeakerResource* spk = &hrm->speakers[hrm->num_speakers];
                memset(spk, 0, sizeof(SpeakerResource));
                snprintf(spk->device_id, sizeof(spk->device_id), "spk_%d", hrm->num_speakers);
                spk->sample_rate = 44100;
                spk->channels = 2;
                spk->is_connected = 1;
                spk->is_stereo = 1;
                spk->quality_score = 0.7f;
                spk->assigned_role = HRM_SPEAKER_ROLE_UNUSED;
                hrm->num_speakers++;
            }
            pCollection->lpVtbl->Release(pCollection);
        }
        pEnumerator->lpVtbl->Release(pEnumerator);
    }

    CoUninitialize();
#else
    /* Linux: 使用V4L2和ALSA检测设备 */
    /* 摄像头检测 (V4L2 /dev/video*) */
    for (int i = 0; i < 8 && hrm->num_cameras < HRM_MAX_CAMERAS; i++) {
        char dev_path[64];
        snprintf(dev_path, sizeof(dev_path), "/dev/video%d", i);
        int fd = open(dev_path, O_RDONLY | O_NONBLOCK);
        if (fd >= 0) {
            CameraResource* cam = &hrm->cameras[hrm->num_cameras];
            memset(cam, 0, sizeof(CameraResource));
            snprintf(cam->device_id, sizeof(cam->device_id), "%s", dev_path);
            snprintf(cam->device_name, sizeof(cam->device_name), "摄像头_%d", i);
            cam->width = 640;
            cam->height = 480;
            cam->fps = 30;
            cam->is_connected = 1;
            cam->is_color = 1;
            cam->quality_score = 0.8f;
            cam->assigned_role = HRM_CAMERA_ROLE_UNUSED;
            hrm->num_cameras++;
            close(fd);
        }
    }

    /* 麦克风检测 (ALSA hw:) */
    for (int i = 0; i < 8 && hrm->num_microphones < HRM_MAX_MICROPHONES; i++) {
        char dev_path[64];
        snprintf(dev_path, sizeof(dev_path), "/dev/snd/pcmC%dD0c", i);
        int fd = open(dev_path, O_RDONLY | O_NONBLOCK);
        if (fd >= 0) {
            MicrophoneResource* mic = &hrm->microphones[hrm->num_microphones];
            memset(mic, 0, sizeof(MicrophoneResource));
            snprintf(mic->device_id, sizeof(mic->device_id), "hw:%d,0", i);
            snprintf(mic->device_name, sizeof(mic->device_name), "麦克风_%d", i);
            mic->sample_rate = 44100;
            mic->channels = 1;
            mic->is_connected = 1;
            mic->quality_score = 0.75f;
            mic->assigned_role = HRM_MIC_ROLE_UNUSED;
            hrm->num_microphones++;
            close(fd);
        }
    }

    /* 扬声器检测 (ALSA playback) */
    for (int i = 0; i < 8 && hrm->num_speakers < HRM_MAX_SPEAKERS; i++) {
        char dev_path[64];
        snprintf(dev_path, sizeof(dev_path), "/dev/snd/pcmC%dD0p", i);
        int fd = open(dev_path, O_WRONLY | O_NONBLOCK);
        if (fd >= 0) {
            SpeakerResource* spk = &hrm->speakers[hrm->num_speakers];
            memset(spk, 0, sizeof(SpeakerResource));
            snprintf(spk->device_id, sizeof(spk->device_id), "hw:%d,0", i);
            snprintf(spk->device_name, sizeof(spk->device_name), "扬声器_%d", i);
            spk->sample_rate = 44100;
            spk->channels = 2;
            spk->is_connected = 1;
            spk->is_stereo = 1;
            spk->quality_score = 0.7f;
            spk->assigned_role = HRM_SPEAKER_ROLE_UNUSED;
            hrm->num_speakers++;
            close(fd);
        }
    }
#endif

    hrm->initialized = 1;
    log_info("[硬件资源管理器] 扫描完成：%d摄像头, %d麦克风, %d扬声器",
             hrm->num_cameras, hrm->num_microphones, hrm->num_speakers);
    return 0;
}

int hrm_auto_allocate(HardwareResourceManager* hrm) {
    if (!hrm || hrm->num_cameras == 0) return -1;

    /* 摄像头分配策略：前2个用于双目，第3个用于识别 */
    for (int i = 0; i < hrm->num_cameras; i++) {
        hrm->cameras[i].assigned_role = HRM_CAMERA_ROLE_UNUSED;
    }
    if (hrm->num_cameras >= 2) {
        hrm->cameras[0].assigned_role = HRM_CAMERA_ROLE_STEREO_LEFT;
        hrm->cameras[1].assigned_role = HRM_CAMERA_ROLE_STEREO_RIGHT;
        hrm->stereo_available = 1;
    }
    if (hrm->num_cameras >= 3) {
        hrm->cameras[2].assigned_role = HRM_CAMERA_ROLE_RECOGNITION;
        hrm->recognition_available = 1;
    } else if (hrm->num_cameras == 2) {
        hrm->cameras[1].assigned_role = HRM_CAMERA_ROLE_RECOGNITION;
        hrm->recognition_available = 1;
    } else if (hrm->num_cameras == 1) {
        hrm->cameras[0].assigned_role = HRM_CAMERA_ROLE_RECOGNITION;
    }

    /* 麦克风分配策略 */
    for (int i = 0; i < hrm->num_microphones; i++) {
        hrm->microphones[i].assigned_role = HRM_MIC_ROLE_UNUSED;
    }
    if (hrm->num_microphones >= 1) {
        hrm->microphones[0].assigned_role = HRM_MIC_ROLE_MAIN_INPUT;
    }
    if (hrm->num_microphones >= 2) {
        hrm->microphones[1].assigned_role = HRM_MIC_ROLE_NOISE_REFERENCE;
    }
    if (hrm->num_microphones >= 3) {
        hrm->microphones[2].assigned_role = HRM_MIC_ROLE_BEAMFORMING;
        hrm->beamforming_available = 1;
    }

    /* 扬声器分配策略 */
    for (int i = 0; i < hrm->num_speakers; i++) {
        hrm->speakers[i].assigned_role = HRM_SPEAKER_ROLE_UNUSED;
    }
    if (hrm->num_speakers >= 1) {
        hrm->speakers[0].assigned_role = HRM_SPEAKER_ROLE_TTS_OUTPUT;
    }
    if (hrm->num_speakers >= 2) {
        hrm->speakers[1].assigned_role = HRM_SPEAKER_ROLE_AUDIO_FEEDBACK;
    }

    hrm->allocation_quality_score = (hrm->num_cameras >= 2 && hrm->num_microphones >= 1) ? 0.9f : 0.5f;
    log_info("[硬件资源管理器] 自动分配完成，质量评分: %.2f", hrm->allocation_quality_score);
    return 0;
}

int hrm_assign_role(HardwareResourceManager* hrm, const char* device_id, int role) {
    if (!hrm || !device_id) return -1;
    for (int i = 0; i < hrm->num_cameras; i++) {
        if (strstr(hrm->cameras[i].device_id, device_id) || strstr(device_id, hrm->cameras[i].device_id)) {
            hrm->cameras[i].assigned_role = (CameraRole)role;
            return 0;
        }
    }
    return -1;
}

const CameraResource* hrm_get_stereo_left(HardwareResourceManager* hrm) {
    if (!hrm || !hrm->stereo_available) return NULL;
    for (int i = 0; i < hrm->num_cameras; i++) {
        if (hrm->cameras[i].assigned_role == HRM_CAMERA_ROLE_STEREO_LEFT)
            return &hrm->cameras[i];
    }
    return NULL;
}

const CameraResource* hrm_get_stereo_right(HardwareResourceManager* hrm) {
    if (!hrm || !hrm->stereo_available) return NULL;
    for (int i = 0; i < hrm->num_cameras; i++) {
        if (hrm->cameras[i].assigned_role == HRM_CAMERA_ROLE_STEREO_RIGHT)
            return &hrm->cameras[i];
    }
    return NULL;
}

const CameraResource* hrm_get_recognition_camera(HardwareResourceManager* hrm) {
    if (!hrm) return NULL;
    for (int i = 0; i < hrm->num_cameras; i++) {
        if (hrm->cameras[i].assigned_role == HRM_CAMERA_ROLE_RECOGNITION)
            return &hrm->cameras[i];
    }
    if (hrm->num_cameras > 0) return &hrm->cameras[0];
    return NULL;
}

const MicrophoneResource* hrm_get_main_microphone(HardwareResourceManager* hrm) {
    if (!hrm) return NULL;
    for (int i = 0; i < hrm->num_microphones; i++) {
        if (hrm->microphones[i].assigned_role == HRM_MIC_ROLE_MAIN_INPUT)
            return &hrm->microphones[i];
    }
    return NULL;
}

const SpeakerResource* hrm_get_tts_speaker(HardwareResourceManager* hrm) {
    if (!hrm) return NULL;
    for (int i = 0; i < hrm->num_speakers; i++) {
        if (hrm->speakers[i].assigned_role == HRM_SPEAKER_ROLE_TTS_OUTPUT)
            return &hrm->speakers[i];
    }
    return NULL;
}

float hrm_get_quality_score(const HardwareResourceManager* hrm) {
    return hrm ? hrm->allocation_quality_score : 0.0f;
}

/* ============================================================================
 * ZSFX-003修复: 热插拔监控线程实现
 * 定期扫描硬件变化（基于hrm_scan_devices的硬件检测能力），
 * 检测设备增删并自动重新分配资源角色。
 * 
 * Windows: CreateThread + WaitForSingleObject
 * Linux:   pthread_create + pthread_join
 * ============================================================================ */

/* 热插拔监控间隔（毫秒） */
#define HRM_HOTPLUG_POLL_INTERVAL_MS 2000

#ifdef _WIN32
static DWORD WINAPI hrm_hotplug_monitor_thread(LPVOID param) {
    HardwareResourceManager* hrm = (HardwareResourceManager*)param;
    if (!hrm) return 1;

    /* COM初始化（COM可能需要用于DirectShow设备枚举） */
    CoInitializeEx(NULL, COINIT_MULTITHREADED);

    /* 记录初始设备数量用于变更检测 */
    int prev_cameras = hrm->num_cameras;
    int prev_mics = hrm->num_microphones;
    int prev_speakers = hrm->num_speakers;

    log_info("[硬件资源管理器] 热插拔监控线程已启动(cam:%d,mic:%d,spk:%d)",
             prev_cameras, prev_mics, prev_speakers);

    while (hrm->monitor_running) {
        Sleep(HRM_HOTPLUG_POLL_INTERVAL_MS);

        if (!hrm->monitor_running) break;

        /* 重新扫描硬件设备 */
        int scan_result = hrm_scan_devices(hrm);
        if (scan_result != 0) {
            log_error("[硬件资源管理器] 热插拔扫描失败，跳过本轮");
            continue;
        }

        /* 检测变更 */
        if (hrm->num_cameras != prev_cameras ||
            hrm->num_microphones != prev_mics ||
            hrm->num_speakers != prev_speakers) {
            log_info("[硬件资源管理器] 硬件变更检测: 摄像头%d→%d, 麦克风%d→%d, 扬声器%d→%d",
                     prev_cameras, hrm->num_cameras,
                     prev_mics, hrm->num_microphones,
                     prev_speakers, hrm->num_speakers);
            hrm_auto_allocate(hrm);
            prev_cameras = hrm->num_cameras;
            prev_mics = hrm->num_microphones;
            prev_speakers = hrm->num_speakers;
        }
    }

    CoUninitialize();
    log_info("[硬件资源管理器] 热插拔监控线程已退出");
    return 0;
}
#else
static void* hrm_hotplug_monitor_thread(void* param) {
    HardwareResourceManager* hrm = (HardwareResourceManager*)param;
    if (!hrm) return NULL;

    int prev_cameras = hrm->num_cameras;
    int prev_mics = hrm->num_microphones;
    int prev_speakers = hrm->num_speakers;

    log_info("[硬件资源管理器] 热插拔监控线程已启动(cam:%d,mic:%d,spk:%d)",
             prev_cameras, prev_mics, prev_speakers);

    while (hrm->monitor_running) {
        /* 以秒为单位的睡眠，支持中断 */
        for (int i = 0; i < HRM_HOTPLUG_POLL_INTERVAL_MS / 100; i++) {
            if (!hrm->monitor_running) break;
            usleep(100000); /* 100ms */
        }
        if (!hrm->monitor_running) break;

        int scan_result = hrm_scan_devices(hrm);
        if (scan_result != 0) {
            log_error("[硬件资源管理器] 热插拔扫描失败，跳过本轮");
            continue;
        }

        if (hrm->num_cameras != prev_cameras ||
            hrm->num_microphones != prev_mics ||
            hrm->num_speakers != prev_speakers) {
            log_info("[硬件资源管理器] 硬件变更检测: 摄像头%d→%d, 麦克风%d→%d, 扬声器%d→%d",
                     prev_cameras, hrm->num_cameras,
                     prev_mics, hrm->num_microphones,
                     prev_speakers, hrm->num_speakers);
            hrm_auto_allocate(hrm);
            prev_cameras = hrm->num_cameras;
            prev_mics = hrm->num_microphones;
            prev_speakers = hrm->num_speakers;
        }
    }

    log_info("[硬件资源管理器] 热插拔监控线程已退出");
    return NULL;
}
#endif

int hrm_start_hotplug_monitor(HardwareResourceManager* hrm) {
    if (!hrm) return -1;

    /* 如果已有线程在运行，先停止 */
    if (hrm->monitor_running) {
        log_warning("[硬件资源管理器] 热插拔监控已在运行，先停止旧线程");
        hrm_stop_hotplug_monitor(hrm);
    }

    /* 确保已执行初始扫描 */
    if (!hrm->initialized) {
        hrm_scan_devices(hrm);
        hrm_auto_allocate(hrm);
    }

    hrm->monitor_running = 1;

#ifdef _WIN32
    hrm->monitor_thread = CreateThread(NULL, 0,
                                        hrm_hotplug_monitor_thread,
                                        hrm, 0, NULL);
    if (!hrm->monitor_thread) {
        log_error("[硬件资源管理器] CreateThread失败，无法启动热插拔监控");
        hrm->monitor_running = 0;
        return -1;
    }
#else
    int rc = pthread_create((pthread_t*)&hrm->monitor_thread, NULL,
                            hrm_hotplug_monitor_thread, hrm);
    if (rc != 0) {
        log_error("[硬件资源管理器] pthread_create失败(errno=%d)，无法启动热插拔监控", rc);
        hrm->monitor_running = 0;
        return -1;
    }
#endif

    log_info("[硬件资源管理器] 热插拔监控已启动(间隔=%dms)", HRM_HOTPLUG_POLL_INTERVAL_MS);
    return 0;
}

int hrm_stop_hotplug_monitor(HardwareResourceManager* hrm) {
    if (!hrm) return -1;
    if (!hrm->monitor_running) return 0;

    /* 发出停止信号 */
    hrm->monitor_running = 0;

    /* 等待线程退出 */
    if (hrm->monitor_thread) {
#ifdef _WIN32
        WaitForSingleObject((HANDLE)hrm->monitor_thread, 5000); /* 最多等5秒 */
        CloseHandle((HANDLE)hrm->monitor_thread);
#else
        pthread_join((pthread_t)hrm->monitor_thread, NULL);
#endif
        hrm->monitor_thread = NULL;
    }

    log_info("[硬件资源管理器] 热插拔监控已停止");
    return 0;
}

int hrm_get_status_summary(const HardwareResourceManager* hrm, char* summary, size_t max_len) {
    if (!hrm || !summary) return -1;
    snprintf(summary, max_len,
             "硬件资源: %d摄像头(双目=%s,识别=%s), %d麦克风(波束=%s), %d扬声器 | 质量评分: %.2f",
             hrm->num_cameras,
             hrm->stereo_available ? "可用" : "不可用",
             hrm->recognition_available ? "可用" : "不可用",
             hrm->num_microphones,
             hrm->beamforming_available ? "可用" : "不可用",
             hrm->num_speakers,
             hrm->allocation_quality_score);
    return 0;
}
