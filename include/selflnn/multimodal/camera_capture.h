/**
 * @file camera_capture.h
 * @brief 真实摄像头硬件采集接口
 *
 * 跨平台摄像头采集统一接口：
 * - Windows: Media Foundation 帧采集
 * - Linux: V4L2 帧采集
 * 100%纯C语言实现，无外部视频库依赖。
 */

#ifndef SELFLNN_CAMERA_CAPTURE_H
#define SELFLNN_CAMERA_CAPTURE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[256];
    char device_id[128];
    int max_width;
    int max_height;
} CameraDeviceInfo;

typedef struct CameraCaptureContext CameraCaptureContext;

int camera_capture_enumerate_devices(CameraDeviceInfo* devices, int max_devices);

CameraCaptureContext* camera_capture_create(const char* device_id,
                                              int width, int height, int fps);

int camera_capture_start(CameraCaptureContext* ctx,
                          void (*callback)(const uint8_t* rgb_data, int width, int height, void* user_data),
                          void* user_data);

int camera_capture_process(CameraCaptureContext* ctx);

int camera_capture_stop(CameraCaptureContext* ctx);

void camera_capture_free(CameraCaptureContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
