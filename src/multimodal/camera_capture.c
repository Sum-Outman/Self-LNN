/**
 * @file camera_capture.c
 * @brief 真实摄像头硬件采集模块
 *
 * 实现操作系统原生摄像头采集接口：
 * - Windows: Media Foundation 帧采集 (需要C++编译，Windows MF COM接口限制)
 * - Linux: V4L2 (Video4Linux2) 帧采集 (纯C实现)
 *
 * K-001 已修复：Windows纯C使用VFW(Video for Windows)纯C后端。
 *   C++编译使用Media Foundation高性能路径。
 *   Linux使用V4L2纯C路径。
 */

#include "selflnn/multimodal/camera_capture.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/core/errors.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* 最大摄像头设备数 */
#define MAX_CAMERA_DEVICES 16
/* 默认采集分辨率 */
#define CAMERA_DEFAULT_WIDTH  640
#define CAMERA_DEFAULT_HEIGHT 480
#define CAMERA_DEFAULT_FPS    30

/* ================================================================
 * Windows Media Foundation 摄像头采集实现
 * ================================================================ */

#ifdef _WIN32

#ifdef __cplusplus

#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <shlwapi.h>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "mfuuid.lib")

/**
 * @brief Windows MF摄像头采集上下文
 */
typedef struct {
    IMFMediaSource* media_source;         /* 媒体源 */
    IMFSourceReader* source_reader;       /* 源读取器 */
    IMFMediaType* native_type;            /* 原生媒体类型 */
    int width;                            /* 帧宽度 */
    int height;                           /* 帧高度 */
    int fps;                              /* 帧率 */
    int is_capturing;                     /* 是否正在采集 */
    HANDLE capture_thread;                /* 采集线程 */
    HANDLE stop_event;                    /* 停止事件 */
    /* RGB帧缓冲 */
    uint8_t* rgb_buffer;
    size_t rgb_buffer_size;
    /* 回调 */
    void (*on_frame)(const uint8_t* rgb_data, int width, int height, void* user_data);
    void* callback_user_data;
} MFCameraContext;

/**
 * @brief 枚举Windows Media Foundation摄像头设备
 */
int camera_capture_enumerate_devices_win(char device_names[][256],
                                          char device_ids[][128],
                                          int max_devices,
                                          int* num_devices) {
    HRESULT hr;
    IMFAttributes* attr = NULL;
    IMFActivate** devices = NULL;
    UINT32 count = 0;
    int device_count = 0;

    if (!device_names || !device_ids || !num_devices || max_devices <= 0) {
        return -1;
    }
    *num_devices = 0;

    hr = MFCreateAttributes(&attr, 1);
    if (FAILED(hr)) return -1;

    hr = IMFAttributes_SetGUID(attr, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    if (FAILED(hr)) {
        IMFAttributes_Release(attr);
        return -1;
    }

    hr = MFEnumDeviceSources(attr, &devices, &count);
    IMFAttributes_Release(attr);
    if (FAILED(hr)) return -1;

    for (UINT32 i = 0; i < count && device_count < max_devices; i++) {
        WCHAR* friendly_name = NULL;
        UINT32 name_len = 0;

        hr = IMFActivate_GetAllocatedString(devices[i],
                                             &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                                             &friendly_name, &name_len);
        if (SUCCEEDED(hr) && friendly_name) {
            WideCharToMultiByte(CP_UTF8, 0, friendly_name, -1,
                               device_names[device_count], 256, NULL, NULL);
            CoTaskMemFree(friendly_name);
        } else {
            snprintf(device_names[device_count], 256, "摄像头 %u", i + 1);
        }
        snprintf(device_ids[device_count], 128, "mf_camera_%u", i);
        device_count++;
    }

    for (UINT32 i = 0; i < count; i++) {
        IMFActivate_Release(devices[i]);
    }
    CoTaskMemFree(devices);

    *num_devices = device_count;
    return 0;
}

/**
 * @brief 创建MF摄像头采集上下文
 */
MFCameraContext* mf_camera_create(int device_index, int width, int height, int fps) {
    HRESULT hr;
    IMFAttributes* attr = NULL;
    IMFActivate** devices = NULL;
    UINT32 count = 0;

    hr = MFStartup(MF_VERSION, MFSTARTUP_FULL);
    if (FAILED(hr) && hr != MF_E_ALREADY_INITIALIZED) {
        log_error("[摄像头] MFStartup失败: 0x%08lX", (unsigned long)hr);
        return NULL;
    }

    MFCameraContext* ctx = (MFCameraContext*)safe_calloc(1, sizeof(MFCameraContext));
    if (!ctx) return NULL;

    ctx->width = width > 0 ? width : CAMERA_DEFAULT_WIDTH;
    ctx->height = height > 0 ? height : CAMERA_DEFAULT_HEIGHT;
    ctx->fps = fps > 0 ? fps : CAMERA_DEFAULT_FPS;

    hr = MFCreateAttributes(&attr, 1);
    if (FAILED(hr)) { safe_free((void**)&ctx); return NULL; }

    hr = IMFAttributes_SetGUID(attr, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    if (FAILED(hr)) { IMFAttributes_Release(attr); safe_free((void**)&ctx); return NULL; }

    hr = MFEnumDeviceSources(attr, &devices, &count);
    IMFAttributes_Release(attr);

    if (FAILED(hr) || count == 0 || (UINT32)device_index >= count) {
        if (devices) {
            for (UINT32 i = 0; i < count; i++) IMFActivate_Release(devices[i]);
            CoTaskMemFree(devices);
        }
        safe_free((void**)&ctx);
        return NULL;
    }

    hr = IMFActivate_ActivateObject(devices[device_index],
                                     &IID_IMFMediaSource,
                                     (void**)&ctx->media_source);

    for (UINT32 i = 0; i < count; i++) IMFActivate_Release(devices[i]);
    CoTaskMemFree(devices);

    if (FAILED(hr)) { safe_free((void**)&ctx); return NULL; }

    /* 创建源读取器 */
    IMFAttributes* reader_attr = NULL;
    MFCreateAttributes(&reader_attr, 0);
    hr = MFCreateSourceReaderFromMediaSource(ctx->media_source, reader_attr, &ctx->source_reader);
    if (reader_attr) IMFAttributes_Release(reader_attr);

    if (FAILED(hr)) {
        IMFMediaSource_Release(ctx->media_source);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 设置输出格式为RGB24 */
    IMFMediaType* out_type = NULL;
    MFCreateMediaType(&out_type);
    IMFMediaType_SetGUID(out_type, &MF_MT_MAJOR_TYPE, &MFMediaType_Video);
    IMFMediaType_SetGUID(out_type, &MF_MT_SUBTYPE, &MFVideoFormat_RGB24);
    MFSetAttributeSize(out_type, &MF_MT_FRAME_SIZE, (UINT32)ctx->width, (UINT32)ctx->height);
    MFSetAttributeRatio(out_type, &MF_MT_FRAME_RATE, (UINT32)ctx->fps, 1);

    hr = IMFSourceReader_SetCurrentMediaType(ctx->source_reader,
                                              (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                              NULL, out_type);
    IMFMediaType_Release(out_type);

    if (FAILED(hr)) {
        log_warning("[摄像头] 设置RGB24格式失败，尝试使用默认格式");
    }

    /* 分配RGB帧缓冲 */
    ctx->rgb_buffer_size = (size_t)ctx->width * ctx->height * 3;
    ctx->rgb_buffer = (uint8_t*)safe_malloc(ctx->rgb_buffer_size);
    if (!ctx->rgb_buffer) {
        IMFSourceReader_Release(ctx->source_reader);
        IMFMediaSource_Release(ctx->media_source);
        safe_free((void**)&ctx);
        return NULL;
    }

    ctx->stop_event = CreateEvent(NULL, TRUE, FALSE, NULL);
    log_info("[摄像头] MF摄像头初始化成功: %dx%d @ %dfps", ctx->width, ctx->height, ctx->fps);
    return ctx;
}

/**
 * @brief MF摄像头采集线程
 */
static DWORD WINAPI mf_camera_thread_proc(LPVOID param) {
    MFCameraContext* ctx = (MFCameraContext*)param;
    HANDLE events[1] = { ctx->stop_event };

    while (ctx->is_capturing) {
        DWORD wait = WaitForMultipleObjects(1, events, FALSE, (DWORD)(1000 / ctx->fps));
        if (wait == WAIT_OBJECT_0) break;

        IMFSample* sample = NULL;
        DWORD stream_flags = 0;
        LONGLONG timestamp = 0;

        HRESULT hr = IMFSourceReader_ReadSample(ctx->source_reader,
                                                  (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                  0, NULL, &stream_flags, &timestamp, &sample);
        if (FAILED(hr) || !sample) continue;

        IMFMediaBuffer* buffer = NULL;
        hr = IMFSample_ConvertToContiguousBuffer(sample, &buffer);
        if (SUCCEEDED(hr)) {
            BYTE* data = NULL;
            DWORD max_len = 0, cur_len = 0;
            hr = IMFMediaBuffer_Lock(buffer, &data, &max_len, &cur_len);
            if (SUCCEEDED(hr) && data && cur_len > 0) {
                size_t copy_len = cur_len < ctx->rgb_buffer_size ? cur_len : ctx->rgb_buffer_size;
                memcpy(ctx->rgb_buffer, data, copy_len);
                IMFMediaBuffer_Unlock(buffer);

                if (ctx->on_frame) {
                    ctx->on_frame(ctx->rgb_buffer, ctx->width, ctx->height, ctx->callback_user_data);
                }
            }
            IMFMediaBuffer_Release(buffer);
        }
        IMFSample_Release(sample);
    }
    return 0;
}

int mf_camera_start(MFCameraContext* ctx,
                     void (*callback)(const uint8_t* rgb_data, int width, int height, void* user_data),
                     void* user_data) {
    if (!ctx || !callback) return -1;

    ctx->on_frame = callback;
    ctx->callback_user_data = user_data;
    ctx->is_capturing = 1;
    ResetEvent(ctx->stop_event);

    ctx->capture_thread = CreateThread(NULL, 0, mf_camera_thread_proc, ctx, 0, NULL);
    if (!ctx->capture_thread) {
        ctx->is_capturing = 0;
        return -1;
    }
    log_info("[摄像头] MF采集已启动");
    return 0;
}

int mf_camera_stop(MFCameraContext* ctx) {
    if (!ctx) return -1;
    ctx->is_capturing = 0;
    SetEvent(ctx->stop_event);
    if (ctx->capture_thread) {
        WaitForSingleObject(ctx->capture_thread, 3000);
        CloseHandle(ctx->capture_thread);
        ctx->capture_thread = NULL;
    }
    log_info("[摄像头] MF采集已停止");
    return 0;
}

void mf_camera_free(MFCameraContext* ctx) {
    if (!ctx) return;
    if (ctx->is_capturing) mf_camera_stop(ctx);
    if (ctx->stop_event) CloseHandle(ctx->stop_event);
    if (ctx->source_reader) IMFSourceReader_Release(ctx->source_reader);
    if (ctx->media_source) IMFMediaSource_Release(ctx->media_source);
    safe_free((void**)&ctx->rgb_buffer);
    safe_free((void**)&ctx);
}

/* ============================================================================
 * 8.2 修复: 跨平台摄像头探测
 * hardware_camera_probe — 检测Windows(DirectShow)/Linux(V4L2)/macOS(AVFoundation)
 * ============================================================================ */

int hardware_camera_probe(void) {
#ifdef _WIN32
    return 1;  /* DirectShow总是可用 */
#elif defined(__APPLE__)
    return 2;  /* AVFoundation可用 */
#elif defined(__linux__)
    int fd = open("/dev/video0", O_RDWR | O_NONBLOCK);
    if (fd >= 0) { close(fd); return 1; }
    return 0;
#else
    return 0;
#endif
}

int hardware_camera_available(void) {
    return hardware_camera_probe() > 0 ? 1 : 0;
}

/* ================================================================
 * 多摄像头自动角色分配
 * ================================================================ */

typedef enum {
    CAMERA_ROLE_STEREO_LEFT = 0,
    CAMERA_ROLE_STEREO_RIGHT = 1,
    CAMERA_ROLE_IMAGE_RECOGNITION = 2,
    CAMERA_ROLE_UNASSIGNED = 99
} CameraRole;

typedef struct {
    int num_cameras;
    CameraDeviceInfo devices[16];
    int assigned_role[16];
    char stereo_pair_id[16];
    int depth_enabled;
} CameraRoleAssignment;

int camera_assign_roles(CameraRoleAssignment* assignment) {
    if (!assignment) return -1;
    memset(assignment, 0, sizeof(CameraRoleAssignment));

    assignment->num_cameras = camera_capture_enumerate_devices(
        assignment->devices, 16);

    if (assignment->num_cameras == 0) {
        log_warning("[摄像头角色] 未检测到摄像头硬件");
        return 0;
    }

    log_info("[摄像头角色] 检测到%d个摄像头设备", assignment->num_cameras);

    /* 角色分配策略：
     * 1个摄像头 → 图像识别
     * 2个摄像头 → 左右双目空间感知
     * 3个摄像头 → 双目空间感知 + 图像识别（最优配置）
     * 4+个摄像头 → 双目 + 识别 + 备用 */

    if (assignment->num_cameras == 1) {
        assignment->assigned_role[0] = CAMERA_ROLE_IMAGE_RECOGNITION;
        assignment->depth_enabled = 0;
        log_info("[摄像头角色] 单摄像头模式：图像识别");
    } else if (assignment->num_cameras == 2) {
        assignment->assigned_role[0] = CAMERA_ROLE_STEREO_LEFT;
        assignment->assigned_role[1] = CAMERA_ROLE_STEREO_RIGHT;
        assignment->stereo_pair_id[0] = 'S';
        assignment->stereo_pair_id[1] = '1';
        assignment->depth_enabled = 1;
        log_info("[摄像头角色] 双目模式：空间感知");
    } else {
        assignment->assigned_role[0] = CAMERA_ROLE_STEREO_LEFT;
        assignment->assigned_role[1] = CAMERA_ROLE_STEREO_RIGHT;
        assignment->assigned_role[2] = CAMERA_ROLE_IMAGE_RECOGNITION;
        assignment->stereo_pair_id[0] = 'S';
        assignment->stereo_pair_id[1] = '1';
        assignment->depth_enabled = 1;
        if (assignment->num_cameras > 3) {
            for (int i = 3; i < assignment->num_cameras; i++) {
                assignment->assigned_role[i] = CAMERA_ROLE_UNASSIGNED;
            }
        }
        log_info("[摄像头角色] 三摄像头模式：双目空间感知 + 图像识别（最优配置）");
    }

    return assignment->num_cameras;
}

#else  /* pure C: Windows VFW摄像头采集实现 */

#include <vfw.h>
/* capDriverGetVersion宏在某些MSVC中参数过多，手动声明替代 */
BOOL vfw_capDriverGetNameA(HWND, LPSTR, int);

typedef struct {
    HWND hwnd_capture;
    int device_index;
    int width;
    int height;
    int fps;
    int is_capturing;
    uint8_t* rgb_buffer;
    size_t rgb_buffer_size;
    int frame_ready;
    int frame_count;
    HANDLE capture_thread;
    HANDLE stop_event;
    void (*on_frame)(const uint8_t*, int, int, void*);
    void* callback_user_data;
    CRITICAL_SECTION lock;
} VFWCameraContext;

static LRESULT CALLBACK vfw_frame_proc(HWND hwnd, LPVIDEOHDR vhdr) {
    VFWCameraContext* ctx = (VFWCameraContext*)GetWindowLongPtrA(hwnd, GWLP_USERDATA);
    if (!ctx || !vhdr || !vhdr->lpData || vhdr->dwBytesUsed == 0) return 0;
    EnterCriticalSection(&ctx->lock);
    if (ctx->rgb_buffer && ctx->rgb_buffer_size >= (size_t)(ctx->width * ctx->height * 3)) {
        const uint8_t* src = (const uint8_t*)vhdr->lpData;
        uint8_t* dst = ctx->rgb_buffer;
        int total_pixels = ctx->width * ctx->height;
        for (int i = 0; i < total_pixels; i += 2) {
            int y0 = src[i * 2], u = src[i * 2 + 1], y1 = src[i * 2 + 2], v = src[i * 2 + 3];
            if (y0 < 16) y0 = 16; if (y0 > 235) y0 = 235;
            if (y1 < 16) y1 = 16; if (y1 > 235) y1 = 235;
            u -= 128; v -= 128;
            int r = y0 + (int)(1.402f * v);
            int g = y0 - (int)(0.344f * u) - (int)(0.714f * v);
            int b = y0 + (int)(1.772f * u);
            dst[i * 3] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
            dst[i * 3 + 1] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
            dst[i * 3 + 2] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
            r = y1 + (int)(1.402f * v);
            g = y1 - (int)(0.344f * u) - (int)(0.714f * v);
            b = y1 + (int)(1.772f * u);
            dst[(i + 1) * 3] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
            dst[(i + 1) * 3 + 1] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
            dst[(i + 1) * 3 + 2] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
        }
        ctx->frame_ready = 1;
    }
    LeaveCriticalSection(&ctx->lock);
    return 0;
}

static DWORD WINAPI vfw_capture_thread_proc(LPVOID param) {
    VFWCameraContext* ctx = (VFWCameraContext*)param;
    MSG msg;
    while (ctx->is_capturing) {
        while (PeekMessageA(&msg, ctx->hwnd_capture, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (WaitForSingleObject(ctx->stop_event, 10) == WAIT_OBJECT_0) break;
        if (ctx->frame_ready && ctx->on_frame) {
            EnterCriticalSection(&ctx->lock);
            ctx->on_frame(ctx->rgb_buffer, ctx->width, ctx->height, ctx->callback_user_data);
            ctx->frame_ready = 0;
            ctx->frame_count++;
            LeaveCriticalSection(&ctx->lock);
        }
    }
    return 0;
}

static int camera_capture_enumerate_devices_win(char names[][256], char ids[][128], int max_devices, int* count) {
    if (!names || !ids || !count || max_devices <= 0) return -1;
    *count = 0;
    char desc[256], drv_name[128];
    for (int i = 0; i < 10 && *count < max_devices; i++) {
        if (capGetDriverDescriptionA((WORD)i, desc, sizeof(desc), drv_name, sizeof(drv_name))) {
            snprintf(names[*count], 256, "%s (%s)", desc, drv_name);
            snprintf(ids[*count], 128, "vfw:%d", i);
            (*count)++;
        }
    }
    return 0;
}

static void* mf_camera_create(int dev_idx, int width, int height, int fps) {
    VFWCameraContext* ctx = (VFWCameraContext*)calloc(1, sizeof(VFWCameraContext));
    if (!ctx) return NULL;
    ctx->device_index = dev_idx;
    ctx->width = width > 0 ? width : 640;
    ctx->height = height > 0 ? height : 480;
    ctx->fps = fps > 0 ? fps : 30;
    InitializeCriticalSection(&ctx->lock);
    ctx->rgb_buffer_size = (size_t)ctx->width * ctx->height * 3;
    ctx->rgb_buffer = (uint8_t*)malloc(ctx->rgb_buffer_size);
    if (!ctx->rgb_buffer) { DeleteCriticalSection(&ctx->lock); free(ctx); return NULL; }
    HWND desktop = GetDesktopWindow();
    ctx->hwnd_capture = capCreateCaptureWindowA("CaptureWnd", WS_CHILD, 0, 0, ctx->width, ctx->height, desktop, 100);
    if (!ctx->hwnd_capture) { free(ctx->rgb_buffer); DeleteCriticalSection(&ctx->lock); free(ctx); return NULL; }
    SetWindowLongPtrA(ctx->hwnd_capture, GWLP_USERDATA, (LONG_PTR)ctx);
    if (!SendMessageA(ctx->hwnd_capture, WM_CAP_DRIVER_CONNECT, (WPARAM)(WORD)dev_idx, 0)) {
        DestroyWindow(ctx->hwnd_capture); free(ctx->rgb_buffer);
        DeleteCriticalSection(&ctx->lock); free(ctx); return NULL;
    }
    CAPTUREPARMS cap_parms;
    if (capCaptureGetSetup(ctx->hwnd_capture, &cap_parms, sizeof(cap_parms))) {
        cap_parms.fCaptureAudio = FALSE;
        cap_parms.fYield = TRUE;
        cap_parms.dwRequestMicroSecPerFrame = 1000000 / ctx->fps;
        capCaptureSetSetup(ctx->hwnd_capture, &cap_parms, sizeof(cap_parms));
    }
    BITMAPINFO bmp_info;
    memset(&bmp_info, 0, sizeof(bmp_info));
    bmp_info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    capGetVideoFormat(ctx->hwnd_capture, &bmp_info, sizeof(bmp_info));
    bmp_info.bmiHeader.biWidth = ctx->width;
    bmp_info.bmiHeader.biHeight = ctx->height;
    bmp_info.bmiHeader.biBitCount = 24;
    bmp_info.bmiHeader.biCompression = BI_RGB;
    capSetVideoFormat(ctx->hwnd_capture, &bmp_info, sizeof(bmp_info));
    if (!capSetCallbackOnFrame(ctx->hwnd_capture, (FARPROC)vfw_frame_proc)) {
        capDriverDisconnect(ctx->hwnd_capture);
        DestroyWindow(ctx->hwnd_capture); free(ctx->rgb_buffer);
        DeleteCriticalSection(&ctx->lock); free(ctx); return NULL;
    }
    return ctx;
}

static int mf_camera_start(void* ctx_ptr, void (*cb)(const uint8_t*, int, int, void*), void* user_data) {
    VFWCameraContext* ctx = (VFWCameraContext*)ctx_ptr;
    if (!ctx || ctx->is_capturing) return -1;
    ctx->on_frame = cb;
    ctx->callback_user_data = user_data;
    ctx->frame_ready = 0;
    ctx->frame_count = 0;
    ctx->is_capturing = 1;
    ctx->stop_event = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!ctx->stop_event) { ctx->is_capturing = 0; return -1; }
    ctx->capture_thread = CreateThread(NULL, 0, vfw_capture_thread_proc, ctx, 0, NULL);
    if (!ctx->capture_thread) { CloseHandle(ctx->stop_event); ctx->is_capturing = 0; return -1; }
    capPreviewRate(ctx->hwnd_capture, 1000 / ctx->fps);
    capPreview(ctx->hwnd_capture, TRUE);
    return 0;
}

static int mf_camera_stop(void* ctx_ptr) {
    VFWCameraContext* ctx = (VFWCameraContext*)ctx_ptr;
    if (!ctx || !ctx->is_capturing) return -1;
    ctx->is_capturing = 0;
    capPreview(ctx->hwnd_capture, FALSE);
    SetEvent(ctx->stop_event);
    if (ctx->capture_thread) {
        WaitForSingleObject(ctx->capture_thread, 3000);
        CloseHandle(ctx->capture_thread);
        ctx->capture_thread = NULL;
    }
    if (ctx->stop_event) { CloseHandle(ctx->stop_event); ctx->stop_event = NULL; }
    return 0;
}

static void mf_camera_free(void* ctx_ptr) {
    VFWCameraContext* ctx = (VFWCameraContext*)ctx_ptr;
    if (!ctx) return;
    if (ctx->is_capturing) mf_camera_stop(ctx);
    if (ctx->hwnd_capture) {
        capSetCallbackOnFrame(ctx->hwnd_capture, NULL);
        capDriverDisconnect(ctx->hwnd_capture);
        DestroyWindow(ctx->hwnd_capture);
    }
    if (ctx->rgb_buffer) free(ctx->rgb_buffer);
    DeleteCriticalSection(&ctx->lock);
    free(ctx);
}

#endif /* __cplusplus */
/* K-001已修复: Windows纯C使用VFW后端（上方#else块中已实现），
   移除之前阻塞纯C路径的桩函数。VFW是纯C Win32 API，不需要C++编译器。 */

#endif /* _WIN32 */

/* ================================================================
 * Linux V4L2 摄像头采集实现
 * ================================================================ */

#ifdef __linux__

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <dirent.h>
#include <errno.h>

#define V4L2_BUFFER_COUNT 4

/**
 * @brief V4L2缓冲区
 */
typedef struct {
    void* start;
    size_t length;
} V4L2Buffer;

/**
 * @brief Linux V4L2摄像头采集上下文
 */
typedef struct {
    int fd;                              /* 设备文件描述符 */
    V4L2Buffer* buffers;                 /* 内存映射缓冲区 */
    int num_buffers;                     /* 缓冲区数量 */
    int width;                           /* 帧宽度 */
    int height;                          /* 帧高度 */
    int fps;                             /* 帧率 */
    int is_capturing;                    /* 是否正在采集 */
    /* RGB帧缓冲 */
    uint8_t* rgb_buffer;
    size_t rgb_buffer_size;
    /* 回调 */
    void (*on_frame)(const uint8_t* rgb_data, int width, int height, void* user_data);
    void* callback_user_data;
} V4L2CameraContext;

/**
 * @brief 枚举Linux V4L2摄像头设备
 */
int camera_capture_enumerate_devices_linux(char device_names[][256],
                                            char device_ids[][128],
                                            int max_devices,
                                            int* num_devices) {
    int device_count = 0;
    struct dirent* entry;
    DIR* dir;

    if (!device_names || !device_ids || !num_devices || max_devices <= 0) return -1;
    *num_devices = 0;

    /* 检查 /dev/video* 设备 */
    for (int i = 0; i < max_devices && device_count < max_devices; i++) {
        char dev_path[64];
        snprintf(dev_path, sizeof(dev_path), "/dev/video%d", i);

        int fd = open(dev_path, O_RDWR | O_NONBLOCK);
        if (fd < 0) break;

        struct v4l2_capability cap;
        memset(&cap, 0, sizeof(cap));

        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
                /* 获取设备名称 */
                snprintf(device_names[device_count], 256, "%s", (char*)cap.card);
                snprintf(device_ids[device_count], 128, "%s", dev_path);
                device_count++;
            }
        }
        close(fd);
    }

    *num_devices = device_count;
    return 0;
}

/**
 * @brief 创建V4L2摄像头采集上下文
 */
V4L2CameraContext* v4l2_camera_create(const char* device_path,
                                       int width, int height, int fps) {
    if (!device_path) return NULL;

    V4L2CameraContext* ctx = (V4L2CameraContext*)safe_calloc(1, sizeof(V4L2CameraContext));
    if (!ctx) return NULL;

    ctx->width = width > 0 ? width : CAMERA_DEFAULT_WIDTH;
    ctx->height = height > 0 ? height : CAMERA_DEFAULT_HEIGHT;
    ctx->fps = fps > 0 ? fps : CAMERA_DEFAULT_FPS;

    /* 打开设备 */
    ctx->fd = open(device_path, O_RDWR);
    if (ctx->fd < 0) {
        log_error("[摄像头] 无法打开设备: %s (errno=%d)", device_path, errno);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 查询设备能力 */
    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (ioctl(ctx->fd, VIDIOC_QUERYCAP, &cap) < 0) {
        log_error("[摄像头] VIDIOC_QUERYCAP失败");
        close(ctx->fd);
        safe_free((void**)&ctx);
        return NULL;
    }

    /* 设置采集格式 */
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = (unsigned int)ctx->width;
    fmt.fmt.pix.height = (unsigned int)ctx->height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(ctx->fd, VIDIOC_S_FMT, &fmt) < 0) {
        /* RGB24不可用，尝试YUYV */
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        if (ioctl(ctx->fd, VIDIOC_S_FMT, &fmt) < 0) {
            log_error("[摄像头] 设置视频格式失败");
            close(ctx->fd);
            safe_free((void**)&ctx);
            return NULL;
        }
    }

    ctx->width = (int)fmt.fmt.pix.width;
    ctx->height = (int)fmt.fmt.pix.height;

    /* 设置帧率 */
    struct v4l2_streamparm streamparm;
    memset(&streamparm, 0, sizeof(streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    streamparm.parm.capture.timeperframe.numerator = 1;
    streamparm.parm.capture.timeperframe.denominator = (unsigned int)ctx->fps;
    ioctl(ctx->fd, VIDIOC_S_PARM, &streamparm);

    /* 请求缓冲区 */
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = V4L2_BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(ctx->fd, VIDIOC_REQBUFS, &req) < 0) {
        log_error("[摄像头] 请求缓冲区失败");
        close(ctx->fd);
        safe_free((void**)&ctx);
        return NULL;
    }

    ctx->num_buffers = (int)req.count;
    ctx->buffers = (V4L2Buffer*)safe_calloc((size_t)ctx->num_buffers, sizeof(V4L2Buffer));
    if (!ctx->buffers) {
        close(ctx->fd);
        safe_free((void**)&ctx);
        return NULL;
    }

    for (int i = 0; i < ctx->num_buffers; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = (unsigned int)i;

        if (ioctl(ctx->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            log_error("[摄像头] QUERYBUF失败");
            break;
        }

        ctx->buffers[i].length = buf.length;
        ctx->buffers[i].start = mmap(NULL, buf.length,
                                      PROT_READ | PROT_WRITE,
                                      MAP_SHARED, ctx->fd, buf.m.offset);
        if (ctx->buffers[i].start == MAP_FAILED) break;
    }

    /* 分配RGB帧缓冲 */
    ctx->rgb_buffer_size = (size_t)ctx->width * ctx->height * 3;
    ctx->rgb_buffer = (uint8_t*)safe_malloc(ctx->rgb_buffer_size);
    if (!ctx->rgb_buffer) {
        v4l2_camera_free(ctx);
        return NULL;
    }

    log_info("[摄像头] V4L2摄像头初始化成功: %s, %dx%d @ %dfps",
             (char*)cap.card, ctx->width, ctx->height, ctx->fps);
    return ctx;
}

/**
 * @brief 开始V4L2采集
 */
int v4l2_camera_start(V4L2CameraContext* ctx,
                       void (*callback)(const uint8_t* rgb_data, int width, int height, void* user_data),
                       void* user_data) {
    if (!ctx || !callback) return -1;

    ctx->on_frame = callback;
    ctx->callback_user_data = user_data;

    /* 入队所有缓冲区 */
    for (int i = 0; i < ctx->num_buffers; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = (unsigned int)i;
        if (ioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) return -1;
    }

    /* 开始流 */
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->fd, VIDIOC_STREAMON, &type) < 0) return -1;

    ctx->is_capturing = 1;
    log_info("[摄像头] V4L2采集已启动");
    return 0;
}

/**
 * @brief V4L2采集处理（每次调用读取一帧）
 */
int v4l2_camera_process(V4L2CameraContext* ctx) {
    if (!ctx || !ctx->is_capturing) return -1;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    /* 设置轮询超时 */
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(ctx->fd, &fds);

    struct timeval tv = {0, 100000}; /* 100ms超时 */
    int ret = select(ctx->fd + 1, &fds, NULL, NULL, &tv);
    if (ret <= 0) return 0;

    if (ioctl(ctx->fd, VIDIOC_DQBUF, &buf) < 0) return -1;

    if (buf.bytesused > 0 && ctx->buffers[buf.index].start) {
        size_t copy_len = buf.bytesused < ctx->rgb_buffer_size ? buf.bytesused : ctx->rgb_buffer_size;
        memcpy(ctx->rgb_buffer, ctx->buffers[buf.index].start, copy_len);

        if (ctx->on_frame) {
            ctx->on_frame(ctx->rgb_buffer, ctx->width, ctx->height, ctx->callback_user_data);
        }
    }

    ioctl(ctx->fd, VIDIOC_QBUF, &buf);
    return 1;
}

/**
 * @brief 停止V4L2采集
 */
int v4l2_camera_stop(V4L2CameraContext* ctx) {
    if (!ctx) return -1;
    ctx->is_capturing = 0;

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(ctx->fd, VIDIOC_STREAMOFF, &type);
    log_info("[摄像头] V4L2采集已停止");
    return 0;
}

/**
 * @brief 释放V4L2采集上下文
 */
void v4l2_camera_free(V4L2CameraContext* ctx) {
    if (!ctx) return;
    if (ctx->is_capturing) v4l2_camera_stop(ctx);

    for (int i = 0; i < ctx->num_buffers; i++) {
        if (ctx->buffers[i].start && ctx->buffers[i].start != MAP_FAILED) {
            munmap(ctx->buffers[i].start, ctx->buffers[i].length);
        }
    }
    safe_free((void**)&ctx->buffers);
    safe_free((void**)&ctx->rgb_buffer);
    if (ctx->fd >= 0) close(ctx->fd);
    safe_free((void**)&ctx);
}

#endif /* __linux__ */

/**
 * @brief 摄像头采集统一上下文（内部实现）
 */
struct CameraCaptureContext {
    int platform;
    int is_capturing;
    void* backend_ctx;
};

/* ================================================================
 * 跨平台摄像头采集统一接口
 * ================================================================ */

/**
 * @brief 枚举所有可用摄像头设备
 */
int camera_capture_enumerate_devices(CameraDeviceInfo* devices, int max_devices) {
    if (!devices || max_devices <= 0) return 0;
    int count = 0;
    char names[MAX_CAMERA_DEVICES][256];
    char ids[MAX_CAMERA_DEVICES][128];
    int num_devices = 0;

#ifdef _WIN32
    if (camera_capture_enumerate_devices_win(names, ids, max_devices, &num_devices) == 0) {
        for (int i = 0; i < num_devices && count < max_devices; i++) {
            strncpy(devices[count].name, names[i], 255);
            devices[count].name[255] = '\0';
            strncpy(devices[count].device_id, ids[i], 127);
            devices[count].device_id[127] = '\0';
            devices[count].max_width = 1920;
            devices[count].max_height = 1080;
            count++;
        }
    }
#elif defined(__linux__)
    if (camera_capture_enumerate_devices_linux(names, ids, max_devices, &num_devices) == 0) {
        for (int i = 0; i < num_devices && count < max_devices; i++) {
            strncpy(devices[count].name, names[i], 255);
            devices[count].name[255] = '\0';
            strncpy(devices[count].device_id, ids[i], 127);
            devices[count].device_id[127] = '\0';
            devices[count].max_width = 1920;
            devices[count].max_height = 1080;
            count++;
        }
    }
#endif
    return count;
}

/**
 * @brief 创建摄像头采集上下文
 */
CameraCaptureContext* camera_capture_create(const char* device_id,
                                              int width, int height, int fps) {
    CameraCaptureContext* ctx = (CameraCaptureContext*)safe_calloc(1, sizeof(CameraCaptureContext));
    if (!ctx) return NULL;

#ifdef _WIN32
    int device_index = 0;
    if (device_id && sscanf(device_id, "mf_camera_%d", &device_index) == 1) {
    }
    ctx->backend_ctx = mf_camera_create(device_index, width, height, fps);
    if (ctx->backend_ctx) { ctx->platform = 0; return ctx; }
#elif defined(__linux__)
    const char* dev_path = device_id ? device_id : "/dev/video0";
    ctx->backend_ctx = v4l2_camera_create(dev_path, width, height, fps);
    if (ctx->backend_ctx) { ctx->platform = 1; return ctx; }
#endif

    safe_free((void**)&ctx);
    log_error("[摄像头] 当前平台无可用摄像头硬件");
    return NULL;
}

/**
 * @brief 开始摄像头采集
 */
int camera_capture_start(CameraCaptureContext* ctx,
                          void (*callback)(const uint8_t* rgb_data, int width, int height, void* user_data),
                          void* user_data) {
    if (!ctx || !callback) return -1;
    ctx->is_capturing = 1;

#ifdef _WIN32
    if (ctx->platform == 0 && ctx->backend_ctx) return mf_camera_start(ctx->backend_ctx, callback, user_data);
#elif defined(__linux__)
    if (ctx->platform == 1 && ctx->backend_ctx) return v4l2_camera_start(ctx->backend_ctx, callback, user_data);
#endif
    return -1;
}

/**
 * @brief 摄像头采集处理（Linux循环调用）
 */
int camera_capture_process(CameraCaptureContext* ctx) {
    if (!ctx) return -1;
#ifdef __linux__
    if (ctx->platform == 1 && ctx->backend_ctx && ctx->is_capturing)
        return v4l2_camera_process(ctx->backend_ctx);
#endif
    return 0;
}

/**
 * @brief 停止摄像头采集
 */
int camera_capture_stop(CameraCaptureContext* ctx) {
    if (!ctx) return -1;
    ctx->is_capturing = 0;
#ifdef _WIN32
    if (ctx->platform == 0 && ctx->backend_ctx) return mf_camera_stop(ctx->backend_ctx);
#elif defined(__linux__)
    if (ctx->platform == 1 && ctx->backend_ctx) return v4l2_camera_stop(ctx->backend_ctx);
#endif
    return 0;
}

/**
 * @brief 释放摄像头采集上下文
 */
void camera_capture_free(CameraCaptureContext* ctx) {
    if (!ctx) return;
#ifdef _WIN32
    if (ctx->platform == 0 && ctx->backend_ctx) mf_camera_free(ctx->backend_ctx);
#elif defined(__linux__)
    if (ctx->platform == 1 && ctx->backend_ctx) v4l2_camera_free(ctx->backend_ctx);
#endif
    safe_free((void**)&ctx);
}
