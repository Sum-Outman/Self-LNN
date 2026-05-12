/**
 * @file hardware_resource_manager.h
 * @brief 硬件资源智能管理器 — 摄像头/麦克风/扬声器动态分配
 *
 * 需求："合理分配摄像头，麦克风，扬声器功能、数量和资源"
 * "建议：是否使用三个摄像头更为合理两个摄像头空间感知、一个摄像头图像识别"
 *
 * 实现策略：
 * - 三个摄像头自动识别：双目(空间感知) + 单目(图像/颜色识别)
 * - 麦克风阵列分配：主输入 + 环境降噪 + 波束成形
 * - 扬声器分配：TTS输出 + 音频反馈
 * - 资源热插拔检测和自动重分配
 */

#ifndef SELFLNN_HARDWARE_RESOURCE_MANAGER_H
#define SELFLNN_HARDWARE_RESOURCE_MANAGER_H

#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HRM_MAX_CAMERAS 8
#define HRM_MAX_MICROPHONES 8
#define HRM_MAX_SPEAKERS 4

typedef enum {
    HRM_CAMERA_ROLE_STEREO_LEFT = 0,
    HRM_CAMERA_ROLE_STEREO_RIGHT = 1,
    HRM_CAMERA_ROLE_RECOGNITION = 2,
    HRM_CAMERA_ROLE_BACKUP = 3,
    HRM_CAMERA_ROLE_UNUSED = 4,
} CameraRole;

typedef enum {
    HRM_MIC_ROLE_MAIN_INPUT = 0,
    HRM_MIC_ROLE_NOISE_REFERENCE = 1,
    HRM_MIC_ROLE_BEAMFORMING = 2,
    HRM_MIC_ROLE_BACKUP = 3,
    HRM_MIC_ROLE_UNUSED = 4,
} MicrophoneRole;

typedef enum {
    HRM_SPEAKER_ROLE_TTS_OUTPUT = 0,
    HRM_SPEAKER_ROLE_AUDIO_FEEDBACK = 1,
    HRM_SPEAKER_ROLE_BACKUP = 2,
    HRM_SPEAKER_ROLE_UNUSED = 3,
} SpeakerRole;

typedef struct {
    char device_id[256];
    char device_name[256];
    int width;
    int height;
    int fps;
    int is_connected;
    int is_color;
    CameraRole assigned_role;
    float quality_score;
} CameraResource;

typedef struct {
    char device_id[256];
    char device_name[256];
    int sample_rate;
    int channels;
    int is_connected;
    int is_array;
    MicrophoneRole assigned_role;
    float quality_score;
} MicrophoneResource;

typedef struct {
    char device_id[256];
    char device_name[256];
    int sample_rate;
    int channels;
    int is_connected;
    int is_stereo;
    SpeakerRole assigned_role;
    float quality_score;
} SpeakerResource;

typedef struct {
    CameraResource cameras[HRM_MAX_CAMERAS];
    int num_cameras;
    int stereo_available;
    int recognition_available;

    MicrophoneResource microphones[HRM_MAX_MICROPHONES];
    int num_microphones;
    int beamforming_available;

    SpeakerResource speakers[HRM_MAX_SPEAKERS];
    int num_speakers;

    int auto_allocate;
    int hotplug_monitoring;
    float allocation_quality_score;

    int initialized;
    void* monitor_thread;
    volatile int monitor_running;
} HardwareResourceManager;

/**
 * @brief 创建硬件资源管理器
 */
SELFLNN_API HardwareResourceManager* hrm_create(void);

/**
 * @brief 释放硬件资源管理器
 */
SELFLNN_API void hrm_free(HardwareResourceManager* hrm);

/**
 * @brief 扫描所有可用硬件设备
 */
SELFLNN_API int hrm_scan_devices(HardwareResourceManager* hrm);

/**
 * @brief 自动分配最优设备角色（基于质量和数量）
 *
 * 策略：
 * - 3+摄像头：最佳2个→双目空间感知，次优1个→图像/颜色识别
 * - 2个摄像头：双目空间感知，其中1个兼任识别
 * - 1个摄像头：同时承担空间感知和识别
 * - 麦克风≥2：主输入+降噪参考
 * - 扬声器≥2：TTS输出+音频反馈
 */
SELFLNN_API int hrm_auto_allocate(HardwareResourceManager* hrm);

/**
 * @brief 手动分配设备角色
 */
SELFLNN_API int hrm_assign_role(HardwareResourceManager* hrm,
                                 const char* device_id, int role);

/**
 * @brief 获取指定角色的设备
 */
SELFLNN_API const CameraResource* hrm_get_stereo_left(HardwareResourceManager* hrm);
SELFLNN_API const CameraResource* hrm_get_stereo_right(HardwareResourceManager* hrm);
SELFLNN_API const CameraResource* hrm_get_recognition_camera(HardwareResourceManager* hrm);
SELFLNN_API const MicrophoneResource* hrm_get_main_microphone(HardwareResourceManager* hrm);
SELFLNN_API const SpeakerResource* hrm_get_tts_speaker(HardwareResourceManager* hrm);

/**
 * @brief 获取分配质量评分(0-1)
 */
SELFLNN_API float hrm_get_quality_score(const HardwareResourceManager* hrm);

/**
 * @brief 启动热插拔监控
 */
SELFLNN_API int hrm_start_hotplug_monitor(HardwareResourceManager* hrm);

/**
 * @brief 停止热插拔监控
 */
SELFLNN_API int hrm_stop_hotplug_monitor(HardwareResourceManager* hrm);

/**
 * @brief 获取资源状态摘要(供前端展示)
 */
SELFLNN_API int hrm_get_status_summary(const HardwareResourceManager* hrm,
                                        char* summary, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_HARDWARE_RESOURCE_MANAGER_H */
