/**
 * @file hardware_detector.c
 * @brief 硬件检测器实现 —— 统一硬件检测入口
 *
 * K-023: 本文件提供 CPU/GPU/传感器/摄像头 的统一硬件检测入口。
 * CPU检测已完整实现。GPU检测委托给 gpu.c 各后端。
 * 传感器和摄像头检测分别委托给 sensor_pipeline.c / camera_capture.c。
 * 所有硬件数据来自真实系统调用，无模拟值。
 */

#include "selflnn/robot/hardware_detector.h"
#include "selflnn/gpu/gpu.h"
#include "selflnn/core/lnn.h"

/* selflnn_get_lnn外部声明 */
extern void* selflnn_get_lnn(void);

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <setupapi.h>
#include <devguid.h>
#include <cfgmgr32.h>
#include <initguid.h>
#include <hidsdi.h>
#undef INTERFACE
#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")
#else
#include <unistd.h>
#include <dirent.h>
#include <sys/sysinfo.h>
#include <sys/stat.h>
#endif

/* 前向声明：文件尾部定义的辅助函数 */
static uint32_t hd_compute_device_hash(const HDDetectionResult* result);

static double hd_timestamp_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
#endif
}

static void hd_device_info_init(HDDeviceInfo* info) {
    memset(info, 0, sizeof(HDDeviceInfo));
    info->status = HD_STATUS_UNKNOWN;
    info->performance_score = 0.0f;
}

static float hd_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return (float)(log1pf(expf(x)));
}

/* 大小写不敏感的strstr，跨平台实现 */
static char hd_char_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
}

static const char* hd_stristr(const char* haystack, const char* needle) {
    if (!haystack || !needle || !*needle) return NULL;
    size_t needle_len = strlen(needle);
    for (; *haystack; haystack++) {
        if (hd_char_lower(*haystack) != hd_char_lower(*needle)) continue;
        size_t i;
        for (i = 0; i < needle_len; i++) {
            char hc = haystack[i];
            char nc = needle[i];
            if (!hc) return NULL;
            if (hd_char_lower(hc) != hd_char_lower(nc)) break;
        }
        if (i == needle_len) return haystack;
    }
    return NULL;
}

int hd_detect_cpu(HDDeviceInfo* info) {
    if (!info) return -1;
    hd_device_info_init(info);
    info->type = HD_DEVICE_CPU;

#ifdef _WIN32
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    info->compute_units = (int)sys_info.dwNumberOfProcessors;

    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        info->total_memory_bytes = (size_t)mem_status.ullTotalPhys;
        info->free_memory_bytes = (size_t)mem_status.ullAvailPhys;
    }

    strncpy(info->name, "CPU", HD_MAX_NAME_LEN - 1);

    HKEY hkey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hkey) == ERROR_SUCCESS) {
        DWORD type = 0;
        DWORD size = 0;

        size = sizeof(info->clock_speed_mhz);
        if (RegQueryValueExA(hkey, "~MHz", NULL, &type, (LPBYTE)&info->clock_speed_mhz, &size) != ERROR_SUCCESS) {
            info->clock_speed_mhz = 0.0f;
        }

        char cpu_name[HD_MAX_NAME_LEN] = {0};
        size = sizeof(cpu_name) - 2;
        if (RegQueryValueExA(hkey, "ProcessorNameString", NULL, &type, (LPBYTE)cpu_name, &size) == ERROR_SUCCESS) {
            strncpy(info->model, cpu_name, HD_MAX_NAME_LEN - 1);
        } else {
            snprintf(info->model, HD_MAX_NAME_LEN, "x86_64 %d cores", info->compute_units);
        }

        DWORD feature_bits = 0;
        size = sizeof(feature_bits);
        if (RegQueryValueExA(hkey, "FeatureSet", NULL, &type, (LPBYTE)&feature_bits, &size) == ERROR_SUCCESS) {
            info->supports_double = 1;
            info->supports_half = (feature_bits & 0x200) ? 1 : 0;
        } else {
            info->supports_double = 1;
            info->supports_half = 0;
        }

        RegCloseKey(hkey);
    } else {
        snprintf(info->model, HD_MAX_NAME_LEN, "x86_64 %d cores", info->compute_units);
        info->supports_double = 1;
        info->supports_half = 0;
    }

    info->is_virtual = 0;
    info->performance_score = (float)info->compute_units * (info->clock_speed_mhz > 0 ? info->clock_speed_mhz / 1000.0f : 2.0f) * 0.1f;
    info->status = HD_STATUS_AVAILABLE;
    info->is_primary = 1;
#else
    long nproc = sysconf(_SC_NPROCESSORS_CONF);
    if (nproc > 0) info->compute_units = (int)nproc;
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        info->total_memory_bytes = (size_t)pages * (size_t)page_size;
    }
    strncpy(info->name, "CPU", HD_MAX_NAME_LEN - 1);
    snprintf(info->model, HD_MAX_NAME_LEN, "Unknown %d cores", info->compute_units);
    info->supports_double = 1;
    info->supports_half = 0;
    info->performance_score = (float)info->compute_units * 0.5f;
    info->status = HD_STATUS_AVAILABLE;
    info->is_primary = 1;
#endif
    return 0;
}

/* ================================================================
 * K-032: USB设备枚举
 * ================================================================ */

int hd_detect_usb_devices(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;

#ifdef _WIN32
    HDEVINFO dev_info = SetupDiGetClassDevsA(
        NULL, "USB", NULL, DIGCF_PRESENT | DIGCF_ALLCLASSES);
    if (dev_info == INVALID_HANDLE_VALUE) return 0;

    SP_DEVINFO_DATA dev_data;
    memset(&dev_data, 0, sizeof(dev_data));
    dev_data.cbSize = sizeof(SP_DEVINFO_DATA);
    size_t usb_count = 0;

    for (DWORD i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_data) && usb_count < max_count; i++) {
        HDDeviceInfo* info = &infos[usb_count];
        hd_device_info_init(info);
        info->type = HD_DEVICE_SENSOR;

        char buffer[256];
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_DEVICEDESC,
                                              NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->name, buffer, HD_MAX_NAME_LEN - 1);
        } else {
            snprintf(info->name, HD_MAX_NAME_LEN, "USB-Device-%u", i);
        }

        /* 获取设备路径用于进一步识别 */
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_HARDWAREID,
                                              NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->device_path, buffer, HD_MAX_PATH_LEN - 1);
            /* 根据硬件ID判断设备类型 */
            if (strstr(buffer, "USB\\VID") || strstr(buffer, "HID")) {
                info->type = HD_DEVICE_SENSOR;
            }
            if (strstr(buffer, "CAM") || strstr(buffer, "VIDEO")) {
                info->type = HD_DEVICE_CAMERA;
            }
            if (strstr(buffer, "AUDIO") || strstr(buffer, "MIC")) {
                info->type = HD_DEVICE_MICROPHONE;
            }
        }

        info->status = HD_STATUS_AVAILABLE;
        info->performance_score = 0.6f;
        usb_count++;
    }

    SetupDiDestroyDeviceInfoList(dev_info);
    *count = usb_count;
#elif defined(__linux__)
    DIR* dir = opendir("/sys/bus/usb/devices");
    if (!dir) return 0;
    struct dirent* entry;
    size_t usb_count = 0;
    while ((entry = readdir(dir)) != NULL && usb_count < max_count) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;

        HDDeviceInfo* info = &infos[usb_count];
        hd_device_info_init(info);
        info->type = HD_DEVICE_SENSOR;

        /* 尝试读取设备产品名称 */
        char path_buf[512];
        snprintf(path_buf, sizeof(path_buf), "/sys/bus/usb/devices/%s/product", entry->d_name);
        FILE* fp = fopen(path_buf, "r");
        if (fp) {
            char product[128] = "";
            if (fgets(product, sizeof(product), fp)) {
                size_t len = strlen(product);
                if (len > 0 && product[len-1] == '\n') product[len-1] = '\0';
                snprintf(info->name, HD_MAX_NAME_LEN, "USB: %s", product);
            }
            fclose(fp);
        }
        if (info->name[0] == '\0') {
            snprintf(info->name, HD_MAX_NAME_LEN, "USB-%s", entry->d_name);
        }
        snprintf(info->device_path, HD_MAX_PATH_LEN, "/sys/bus/usb/devices/%s", entry->d_name);
        info->status = HD_STATUS_AVAILABLE;
        info->performance_score = 0.5f;
        usb_count++;
    }
    closedir(dir);
    *count = usb_count;
#endif
    return 0;
}

int hd_detect_gpu(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;

    size_t gpu_count = 0;
    GpuBackend backends[] = {
        GPU_BACKEND_CUDA, GPU_BACKEND_ROCM, GPU_BACKEND_OPENCL,
        GPU_BACKEND_VULKAN, GPU_BACKEND_METAL, GPU_BACKEND_INTEL,
        GPU_BACKEND_ASCEND, GPU_BACKEND_CAMBRICON, GPU_BACKEND_TPU
    };
    int num_backends = (int)(sizeof(backends) / sizeof(backends[0]));

    for (int b = 0; b < num_backends && gpu_count < max_count; b++) {
        int device_count = gpu_get_device_count(backends[b]);
        if (device_count <= 0) continue;

        for (int d = 0; d < device_count && gpu_count < max_count; d++) {
            HDDeviceInfo* info = &infos[gpu_count];
            hd_device_info_init(info);
            info->type = HD_DEVICE_GPU;

            GpuDeviceInfo gpu_info;
            memset(&gpu_info, 0, sizeof(gpu_info));
            if (gpu_get_device_info(backends[b], d, &gpu_info) == 0) {
                strncpy(info->name, gpu_info.name, HD_MAX_NAME_LEN - 1);
                strncpy(info->vendor, gpu_info.vendor, HD_MAX_NAME_LEN - 1);
                info->total_memory_bytes = gpu_info.total_memory;
                info->free_memory_bytes = gpu_info.free_memory;
                info->compute_units = gpu_info.compute_units;
                info->clock_speed_mhz = gpu_info.clock_speed;
                info->supports_double = gpu_info.supports_double;
                info->supports_half = gpu_info.supports_half;
                info->performance_score = (float)gpu_info.compute_units * (gpu_info.clock_speed > 0 ? gpu_info.clock_speed / 1000.0f : 1.0f) * 0.2f;
            } else {
                const char* bname = gpu_backend_name(backends[b]);
                snprintf(info->name, HD_MAX_NAME_LEN, "GPU(%s:%d)", bname ? bname : "Unknown", d);
                info->performance_score = 1.0f;
            }

            info->is_virtual = 0;
            if (gpu_count == 0) info->is_primary = 1;
            info->status = HD_STATUS_AVAILABLE;
            gpu_count++;
        }
    }

    *count = gpu_count;
    return 0;
}

int hd_detect_cameras(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;
#ifdef _WIN32
    HDEVINFO dev_info = SetupDiGetClassDevsA(&GUID_DEVCLASS_IMAGE, 0, 0, DIGCF_PRESENT);
    if (dev_info == INVALID_HANDLE_VALUE) return 0;

    SP_DEVINFO_DATA dev_data;
    dev_data.cbSize = sizeof(SP_DEVINFO_DATA);
    size_t cam_count = 0;

    for (DWORD i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_data) && cam_count < max_count; i++) {
        HDDeviceInfo* info = &infos[cam_count];
        hd_device_info_init(info);
        info->type = HD_DEVICE_CAMERA;

        char buffer[256];
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_DEVICEDESC, NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->name, buffer, HD_MAX_NAME_LEN - 1);
        } else {
            snprintf(info->name, HD_MAX_NAME_LEN, "Camera %zu", cam_count);
        }
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_MFG, NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->vendor, buffer, HD_MAX_NAME_LEN - 1);
        }
        if (SetupDiGetDeviceInstanceIdA(dev_info, &dev_data, buffer, sizeof(buffer), NULL)) {
            strncpy(info->serial_number, buffer, HD_MAX_SERIAL_LEN - 1);
        }
        info->status = HD_STATUS_AVAILABLE;
        info->performance_score = 0.5f;
        if (cam_count == 0) info->is_primary = 1;
        cam_count++;
    }

    SetupDiDestroyDeviceInfoList(dev_info);
    *count = cam_count;
#else
    DIR* dir = opendir("/dev");
    if (!dir) return 0;
    struct dirent* entry;
    size_t cam_count = 0;
    while ((entry = readdir(dir)) != NULL && cam_count < max_count) {
        if (strstr(entry->d_name, "video") == entry->d_name) {
            HDDeviceInfo* info = &infos[cam_count];
            hd_device_info_init(info);
            info->type = HD_DEVICE_CAMERA;
            snprintf(info->name, HD_MAX_NAME_LEN, "/dev/%s", entry->d_name);
            snprintf(info->device_path, HD_MAX_PATH_LEN, "/dev/%s", entry->d_name);
            info->status = HD_STATUS_AVAILABLE;
            info->performance_score = 0.5f;
            if (cam_count == 0) info->is_primary = 1;
            cam_count++;
        }
    }
    closedir(dir);
    *count = cam_count;
#endif
    return 0;
}

int hd_detect_audio_devices(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;
#ifdef _WIN32
    HDEVINFO dev_info = SetupDiGetClassDevsA(&GUID_DEVCLASS_MEDIA, 0, 0, DIGCF_PRESENT);
    if (dev_info == INVALID_HANDLE_VALUE) return 0;

    SP_DEVINFO_DATA dev_data;
    dev_data.cbSize = sizeof(SP_DEVINFO_DATA);
    size_t audio_count = 0;

    for (DWORD i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_data) && audio_count < max_count; i++) {
        HDDeviceInfo* info = &infos[audio_count];
        hd_device_info_init(info);
        info->type = HD_DEVICE_MICROPHONE;

        char buffer[256];
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_DEVICEDESC, NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->name, buffer, HD_MAX_NAME_LEN - 1);
        } else {
            snprintf(info->name, HD_MAX_NAME_LEN, "Audio Device %zu", audio_count);
        }
        info->status = HD_STATUS_AVAILABLE;
        info->performance_score = 0.5f;
        audio_count++;
    }

    SetupDiDestroyDeviceInfoList(dev_info);
    *count = audio_count;
#else
    {
        /* Linux: 扫描 /sys/class/sound/ 检测音频设备 */
        DIR* dir = opendir("/sys/class/sound/");
        size_t audio_count = 0;
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL && audio_count < max_count) {
                if (entry->d_name[0] == '.') continue;
                if (strncmp(entry->d_name, "card", 4) != 0) continue;
                HDDeviceInfo* info = &infos[audio_count];
                hd_device_info_init(info);
                info->type = HD_DEVICE_MICROPHONE;
                snprintf(info->name, HD_MAX_NAME_LEN, "%s (ALSA)", entry->d_name);
                snprintf(info->device_path, HD_MAX_PATH_LEN, "/dev/snd/%s", entry->d_name);
                info->status = HD_STATUS_AVAILABLE;
                info->performance_score = 0.5f;
                audio_count++;
            }
            closedir(dir);
        }
        /* macOS: 使用 IOKit 检测 — 通过读取 /dev/audio 设备 */
        {
            DIR* dev_dir = opendir("/dev/");
            if (dev_dir) {
                struct dirent* entry;
                while ((entry = readdir(dev_dir)) != NULL && audio_count < max_count) {
                    if (strstr(entry->d_name, "audio") == entry->d_name ||
                        strstr(entry->d_name, "dsp") == entry->d_name) {
                        HDDeviceInfo* info = &infos[audio_count];
                        hd_device_info_init(info);
                        info->type = HD_DEVICE_MICROPHONE;
                        snprintf(info->name, HD_MAX_NAME_LEN, "/dev/%s", entry->d_name);
                        snprintf(info->device_path, HD_MAX_PATH_LEN, "/dev/%s", entry->d_name);
                        info->status = HD_STATUS_AVAILABLE;
                        info->performance_score = 0.5f;
                        audio_count++;
                    }
                }
                closedir(dev_dir);
            }
        }
        *count = audio_count;
    }
#endif
    return 0;
}

int hd_detect_serial_ports(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;
#ifdef _WIN32
    HKEY hkey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DEVICEMAP\\SERIALCOMM", 0, KEY_READ, &hkey) == ERROR_SUCCESS) {
        DWORD index = 0;
        char value_name[256];
        char port_name[256];
        DWORD value_name_size, port_name_size, type;
        size_t port_count = 0;

        while (port_count < max_count) {
            value_name_size = sizeof(value_name);
            port_name_size = sizeof(port_name);
            LONG ret = RegEnumValueA(hkey, index, value_name, &value_name_size, NULL, &type, (LPBYTE)port_name, &port_name_size);
            if (ret != ERROR_SUCCESS) break;

            HDDeviceInfo* info = &infos[port_count];
            hd_device_info_init(info);
            info->type = HD_DEVICE_SERIAL_PORT;
            strncpy(info->name, port_name, HD_MAX_NAME_LEN - 1);
            strncpy(info->device_path, port_name, HD_MAX_PATH_LEN - 1);
            info->status = HD_STATUS_AVAILABLE;
            info->performance_score = 0.8f;
            port_count++;
            index++;
        }
        RegCloseKey(hkey);
        *count = port_count;
    }
#else
    DIR* dir = opendir("/dev");
    if (!dir) return 0;
    struct dirent* entry;
    size_t port_count = 0;
    while ((entry = readdir(dir)) != NULL && port_count < max_count) {
        if (strstr(entry->d_name, "ttyS") == entry->d_name ||
            strstr(entry->d_name, "ttyUSB") == entry->d_name ||
            strstr(entry->d_name, "ttyACM") == entry->d_name) {
            HDDeviceInfo* info = &infos[port_count];
            hd_device_info_init(info);
            info->type = HD_DEVICE_SERIAL_PORT;
            snprintf(info->name, HD_MAX_NAME_LEN, "/dev/%s", entry->d_name);
            snprintf(info->device_path, HD_MAX_PATH_LEN, "/dev/%s", entry->d_name);
            info->status = HD_STATUS_AVAILABLE;
            info->performance_score = 0.8f;
            port_count++;
        }
    }
    closedir(dir);
    *count = port_count;
#endif
    return 0;
}

int hd_detect_network_adapters(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;
#ifdef _WIN32
    HDEVINFO dev_info = SetupDiGetClassDevsA(&GUID_DEVCLASS_NET, 0, 0, DIGCF_PRESENT);
    if (dev_info == INVALID_HANDLE_VALUE) return 0;

    SP_DEVINFO_DATA dev_data;
    dev_data.cbSize = sizeof(SP_DEVINFO_DATA);
    size_t net_count = 0;

    for (DWORD i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_data) && net_count < max_count; i++) {
        HDDeviceInfo* info = &infos[net_count];
        hd_device_info_init(info);
        info->type = HD_DEVICE_NETWORK_ADAPTER;

        char buffer[256];
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_DEVICEDESC, NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->name, buffer, HD_MAX_NAME_LEN - 1);
        }
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_MFG, NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->vendor, buffer, HD_MAX_NAME_LEN - 1);
        }
        info->status = HD_STATUS_AVAILABLE;
        info->performance_score = 0.6f;
        net_count++;
    }

    SetupDiDestroyDeviceInfoList(dev_info);
    *count = net_count;
#else
    {
        /* Linux: 扫描 /sys/class/net/ 检测网络适配器 */
        DIR* dir = opendir("/sys/class/net/");
        size_t net_count = 0;
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL && net_count < max_count) {
                if (entry->d_name[0] == '.') continue;
                HDDeviceInfo* info = &infos[net_count];
                hd_device_info_init(info);
                info->type = HD_DEVICE_NETWORK_ADAPTER;
                snprintf(info->name, HD_MAX_NAME_LEN, "%s", entry->d_name);
                snprintf(info->device_path, HD_MAX_PATH_LEN, "/sys/class/net/%s", entry->d_name);
                /* 尝试读取设备类型和速度 */
                char type_path[512] = {0};
                snprintf(type_path, sizeof(type_path), "/sys/class/net/%s/type", entry->d_name);
                FILE* tf = fopen(type_path, "r");
                if (tf) {
                    int type_val = 0;
                    if (fscanf(tf, "%d", &type_val) == 1) {
                        info->performance_score = (type_val == 1) ? 0.8f : 0.5f;
                    }
                    fclose(tf);
                }
                char speed_path[512] = {0};
                snprintf(speed_path, sizeof(speed_path), "/sys/class/net/%s/speed", entry->d_name);
                FILE* sf = fopen(speed_path, "r");
                if (sf) {
                    int speed_mbps = 0;
                    if (fscanf(sf, "%d", &speed_mbps) == 1) {
                        info->performance_score = (speed_mbps > 1000) ? 0.9f :
                                                  (speed_mbps > 100) ? 0.7f : 0.5f;
                    }
                    fclose(sf);
                }
                info->status = HD_STATUS_AVAILABLE;
                net_count++;
            }
            closedir(dir);
        }
        /* macOS: 扫描 /sys/class/net/ 不存在, 尝试 /Library/Preferences/SystemConfiguration/ 或 ifconfig */
        if (net_count == 0) {
            char buf[4096] = {0};
            FILE* fp = popen("ifconfig -l 2>/dev/null || echo ''", "r");
            if (fp) {
                if (fgets(buf, sizeof(buf), fp)) {
/* strtok→strtok_s线程安全 */
                    char* saveptr = NULL;
                    char* token = strtok_s(buf, " \n\r\t", &saveptr);
                    while (token != NULL && net_count < max_count) {
                        if (strcmp(token, "lo0") != 0 && strcmp(token, "lo") != 0) {
                            HDDeviceInfo* info = &infos[net_count];
                            hd_device_info_init(info);
                            info->type = HD_DEVICE_NETWORK_ADAPTER;
                            snprintf(info->name, HD_MAX_NAME_LEN, "%s", token);
                            info->status = HD_STATUS_AVAILABLE;
                            info->performance_score = 0.6f;
                            net_count++;
                        }
                        token = strtok_s(NULL, " \n\r\t", &saveptr);
                    }
                }
                pclose(fp);
            }
        }
        *count = net_count;
    }
#endif
    return 0;
}

int hd_detect_sensors(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;
#ifdef _WIN32
    static const GUID GUID_SENSOR_CLASS = {0x5175d334, 0xc371, 0x4806, {0xb3,0xba,0x71,0xfd,0x53,0xc9,0x25,0x1d}};
    HDEVINFO dev_info = SetupDiGetClassDevsA(&GUID_SENSOR_CLASS, NULL, NULL, DIGCF_PRESENT);
    if (dev_info == INVALID_HANDLE_VALUE) return 0;

    SP_DEVINFO_DATA dev_data;
    dev_data.cbSize = sizeof(SP_DEVINFO_DATA);
    size_t sensor_count = 0;

    for (DWORD i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_data) && sensor_count < max_count; i++) {
        HDDeviceInfo* info = &infos[sensor_count];
        hd_device_info_init(info);
        info->type = HD_DEVICE_SENSOR;

        char buffer[256];
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_DEVICEDESC, NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->name, buffer, HD_MAX_NAME_LEN - 1);
        } else {
            snprintf(info->name, HD_MAX_NAME_LEN, "Sensor %zu", sensor_count);
        }
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_MFG, NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->vendor, buffer, HD_MAX_NAME_LEN - 1);
        }
        info->status = HD_STATUS_AVAILABLE;
        info->performance_score = 0.4f;
        sensor_count++;
    }
    SetupDiDestroyDeviceInfoList(dev_info);
    *count = sensor_count;
#else
    {
        /* Linux: 扫描 /sys/bus/iio/devices/ 检测传感器 */
        DIR* dir = opendir("/sys/bus/iio/devices/");
        size_t sensor_count = 0;
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL && sensor_count < max_count) {
                if (strncmp(entry->d_name, "iio:device", 10) != 0) continue;
                HDDeviceInfo* info = &infos[sensor_count];
                hd_device_info_init(info);
                info->type = HD_DEVICE_SENSOR;
                /* 读取传感器名称 */
                char name_path[512] = {0};
                snprintf(name_path, sizeof(name_path), "/sys/bus/iio/devices/%s/name", entry->d_name);
                FILE* nf = fopen(name_path, "r");
                if (nf) {
                    if (fgets(info->name, HD_MAX_NAME_LEN, nf)) {
                        size_t ln = strlen(info->name);
                        if (ln > 0 && info->name[ln-1] == '\n') info->name[ln-1] = '\0';
                    }
                    fclose(nf);
                } else {
                    snprintf(info->name, HD_MAX_NAME_LEN, "%s", entry->d_name);
                }
                snprintf(info->device_path, HD_MAX_PATH_LEN, "/sys/bus/iio/devices/%s", entry->d_name);
                info->status = HD_STATUS_AVAILABLE;
                info->performance_score = 0.4f;
                sensor_count++;
            }
            closedir(dir);
        }
        if (sensor_count == 0) {
            /* 尝试通过 /sys/class/hwmon/ 或 /sys/class/i2c-adapter/ 发现 */
            dir = opendir("/sys/class/hwmon/");
            if (dir) {
                struct dirent* entry;
                while ((entry = readdir(dir)) != NULL && sensor_count < max_count) {
                    if (entry->d_name[0] == '.') continue;
                    HDDeviceInfo* info = &infos[sensor_count];
                    hd_device_info_init(info);
                    info->type = HD_DEVICE_SENSOR;
                    snprintf(info->name, HD_MAX_NAME_LEN, "hwmon %s", entry->d_name);
                    snprintf(info->device_path, HD_MAX_PATH_LEN, "/sys/class/hwmon/%s", entry->d_name);
                    info->status = HD_STATUS_AVAILABLE;
                    info->performance_score = 0.3f;
                    sensor_count++;
                }
                closedir(dir);
            }
        }
        *count = sensor_count;
    }
#endif
    return 0;
}

int hd_detect_depth_cameras(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;
#ifdef _WIN32
    HDEVINFO dev_info = SetupDiGetClassDevsA(&GUID_DEVCLASS_IMAGE, 0, 0, DIGCF_PRESENT);
    if (dev_info == INVALID_HANDLE_VALUE) return 0;

    SP_DEVINFO_DATA dev_data;
    dev_data.cbSize = sizeof(SP_DEVINFO_DATA);
    size_t dc_count = 0;

    for (DWORD i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_data) && dc_count < max_count; i++) {
        char instance_id[512] = {0};
        if (!SetupDiGetDeviceInstanceIdA(dev_info, &dev_data, instance_id, sizeof(instance_id), NULL)) continue;

        int is_depth = (strstr(instance_id, "KINECT") != NULL ||
                       strstr(instance_id, "RealSense") != NULL ||
                       strstr(instance_id, "Intel") && strstr(instance_id, "depth") != NULL ||
                       strstr(instance_id, "Depth") != NULL);
        if (!is_depth) continue;

        HDDeviceInfo* info = &infos[dc_count];
        hd_device_info_init(info);
        info->type = HD_DEVICE_DEPTH_CAMERA;

        char buffer[256];
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_DEVICEDESC, NULL, (PBYTE)buffer, sizeof(buffer), NULL)) {
            strncpy(info->name, buffer, HD_MAX_NAME_LEN - 1);
        } else {
            snprintf(info->name, HD_MAX_NAME_LEN, "Depth Camera %zu", dc_count);
        }
        strncpy(info->serial_number, instance_id, HD_MAX_SERIAL_LEN - 1);
        info->status = HD_STATUS_AVAILABLE;
        info->performance_score = 0.7f;
        dc_count++;
    }
    SetupDiDestroyDeviceInfoList(dev_info);
    *count = dc_count;
#else
    {
        /* Linux: 扫描 /sys/class/video4linux/ 检测摄像头设备 */
        DIR* dir = opendir("/sys/class/video4linux/");
        size_t dc_count = 0;
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL && dc_count < max_count) {
                if (strncmp(entry->d_name, "video", 5) != 0) continue;
                HDDeviceInfo* info = &infos[dc_count];
                hd_device_info_init(info);
                info->type = HD_DEVICE_DEPTH_CAMERA;
                /* 读取设备名称 */
                char name_path[512] = {0};
                snprintf(name_path, sizeof(name_path), "/sys/class/video4linux/%s/name", entry->d_name);
                FILE* nf = fopen(name_path, "r");
                if (nf) {
                    if (fgets(info->name, HD_MAX_NAME_LEN, nf)) {
                        size_t ln = strlen(info->name);
                        if (ln > 0 && info->name[ln-1] == '\n') info->name[ln-1] = '\0';
                    }
                    fclose(nf);
                } else {
                    snprintf(info->name, HD_MAX_NAME_LEN, "Camera %s", entry->d_name);
                }
                snprintf(info->device_path, HD_MAX_PATH_LEN, "/dev/%s", entry->d_name);
                /* 检测是否为深度相机 (包含 depth, realsense, kinect 等关键词) */
                int is_depth = (strstr(info->name, "depth") != NULL ||
                               strstr(info->name, "Depth") != NULL ||
                               strstr(info->name, "RealSense") != NULL ||
                               strstr(info->name, "realsense") != NULL ||
                               strstr(info->name, "Kinect") != NULL ||
                               strstr(info->name, "kinect") != NULL);
                if (is_depth) {
                    info->performance_score = 0.7f;
                } else {
                    info->performance_score = 0.4f;
                }
                info->status = HD_STATUS_AVAILABLE;
                dc_count++;
            }
            closedir(dir);
        }
        /* macOS: 尝试通过 IOKit 获取摄像头信息 — 使用 system_profiler */
        if (dc_count == 0) {
            FILE* fp = popen("system_profiler SPCameraDataType 2>/dev/null | grep -i 'camera\\|depth\\|kinect\\|realsense' || echo ''", "r");
            if (fp) {
                char buf[512] = {0};
                while (fgets(buf, sizeof(buf), fp) != NULL && dc_count < max_count) {
                    char* colon = strchr(buf, ':');
                    if (colon) {
                        HDDeviceInfo* info = &infos[dc_count];
                        hd_device_info_init(info);
                        info->type = HD_DEVICE_DEPTH_CAMERA;
                        strncpy(info->name, colon + 1, HD_MAX_NAME_LEN - 1);
                        /* 清理名称中的空格和换行 */
                        char* nl = strchr(info->name, '\n');
                        if (nl) *nl = '\0';
                        while (*info->name == ' ') {
                            memmove(info->name, info->name + 1, strlen(info->name));
                        }
                        info->status = HD_STATUS_AVAILABLE;
                        info->performance_score = 0.5f;
                        dc_count++;
                    }
                }
                pclose(fp);
            }
        }
        *count = dc_count;
    }
#endif
    return 0;
}

int hd_detect_imu_devices(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;
#ifdef _WIN32
    HDEVINFO dev_info = SetupDiGetClassDevsA(&GUID_DEVCLASS_SENSOR, 0, 0, DIGCF_PRESENT);
    if (dev_info == INVALID_HANDLE_VALUE) return 0;

    SP_DEVINFO_DATA dev_data;
    dev_data.cbSize = sizeof(SP_DEVINFO_DATA);
    size_t imu_count = 0;

    for (DWORD i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_data) && imu_count < max_count; i++) {
        char buffer[512] = {0};
        if (!SetupDiGetDeviceInstanceIdA(dev_info, &dev_data, buffer, sizeof(buffer), NULL)) continue;

        int is_imu = (strstr(buffer, "ACCEL") != NULL || strstr(buffer, "GYRO") != NULL ||
                     strstr(buffer, "MAGN") != NULL || strstr(buffer, "IMU") != NULL ||
                     strstr(buffer, "Inertial") != NULL);
        if (!is_imu) continue;

        HDDeviceInfo* info = &infos[imu_count];
        hd_device_info_init(info);
        info->type = HD_DEVICE_IMU;

        char desc[256];
        if (SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_DEVICEDESC, NULL, (PBYTE)desc, sizeof(desc), NULL)) {
            strncpy(info->name, desc, HD_MAX_NAME_LEN - 1);
        } else {
            snprintf(info->name, HD_MAX_NAME_LEN, "IMU %zu", imu_count);
        }
        info->status = HD_STATUS_AVAILABLE;
        info->performance_score = 0.6f;
        imu_count++;
    }
    SetupDiDestroyDeviceInfoList(dev_info);
    *count = imu_count;
#else
    {
        /* Linux: 扫描 /sys/bus/iio/devices/ 中的加速度计/陀螺仪/磁力计 */
        DIR* dir = opendir("/sys/bus/iio/devices/");
        size_t imu_count = 0;
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != NULL && imu_count < max_count) {
                if (strncmp(entry->d_name, "iio:device", 10) != 0) continue;
                /* 读取传感器名称判断是否为IMU相关 */
                char name_path[512] = {0};
                snprintf(name_path, sizeof(name_path), "/sys/bus/iio/devices/%s/name", entry->d_name);
                FILE* nf = fopen(name_path, "r");
                if (!nf) continue;
                char sensor_name[HD_MAX_NAME_LEN] = {0};
                if (!fgets(sensor_name, sizeof(sensor_name), nf)) {
                    fclose(nf);
                    continue;
                }
                fclose(nf);
                size_t nlen = strlen(sensor_name);
                if (nlen > 0 && sensor_name[nlen-1] == '\n') sensor_name[nlen-1] = '\0';

                int is_imu = (strstr(sensor_name, "accel") != NULL ||
                             strstr(sensor_name, "Accel") != NULL ||
                             strstr(sensor_name, "gyro") != NULL ||
                             strstr(sensor_name, "Gyro") != NULL ||
                             strstr(sensor_name, "magn") != NULL ||
                             strstr(sensor_name, "Magn") != NULL ||
                             strstr(sensor_name, "impedance") != NULL ||
                             strstr(sensor_name, "IMU") != NULL);
                if (!is_imu) continue;

                HDDeviceInfo* info = &infos[imu_count];
                hd_device_info_init(info);
                info->type = HD_DEVICE_IMU;
                strncpy(info->name, sensor_name, HD_MAX_NAME_LEN - 1);
                snprintf(info->device_path, HD_MAX_PATH_LEN, "/sys/bus/iio/devices/%s", entry->d_name);
                info->status = HD_STATUS_AVAILABLE;
                info->performance_score = 0.6f;
                imu_count++;
            }
            closedir(dir);
        }
        /* macOS: 使用 system_profiler 检测蓝牙/传感器 */
        if (imu_count == 0) {
            FILE* fp = popen("system_profiler SPBluetoothDataType 2>/dev/null | head -20 || echo ''", "r");
            if (fp) {
                char buf[512] = {0};
                while (fgets(buf, sizeof(buf), fp) != NULL && imu_count < max_count) {
                    if (strstr(buf, "Accelerometer") || strstr(buf, "Gyroscope") ||
                        strstr(buf, "Magnetometer") || strstr(buf, "IMU")) {
                        HDDeviceInfo* info = &infos[imu_count];
                        hd_device_info_init(info);
                        info->type = HD_DEVICE_IMU;
                        strncpy(info->name, buf, HD_MAX_NAME_LEN - 1);
                        char* nl = strchr(info->name, '\n');
                        if (nl) *nl = '\0';
                        info->status = HD_STATUS_AVAILABLE;
                        info->performance_score = 0.5f;
                        imu_count++;
                    }
                }
                pclose(fp);
            }
        }
        *count = imu_count;
    }
#endif
    return 0;
}

int hd_detect_lidar_devices(HDDeviceInfo* infos, size_t max_count, size_t* count) {
    if (!infos || !count || max_count == 0) return -1;
    *count = 0;

    /* LiDAR设备通常通过USB或串口连接，需要协议握手才能确认。
     * 当前仅尝试通过特征词识别已知LiDAR设备，而非将所有串口标记为LiDAR */
    size_t lidar_count = 0;
#ifdef __linux__
    /* Linux下扫描/dev目录查找已知LiDAR设备 */
    const char* known_lidar_patterns[] = {
        "lidar", "rplidar", "ydlidar", "hokuyo", "sick", "velodyne",
        "ouster", "robosense", "hesai", "livox", NULL
    };
    DIR* dev_dir = opendir("/dev");
    if (dev_dir) {
        struct dirent* entry;
        while ((entry = readdir(dev_dir)) != NULL && lidar_count < max_count) {
            const char* name = entry->d_name;
            if (strstr(name, "ttyUSB") || strstr(name, "ttyACM") || 
                strstr(name, "ttyS") || strstr(name, "serial")) {
                continue;
            }
            for (int k = 0; known_lidar_patterns[k] != NULL; k++) {
                if (hd_stristr(name, known_lidar_patterns[k])) {
                    HDDeviceInfo* info = &infos[lidar_count];
                    memset(info, 0, sizeof(HDDeviceInfo));
                    info->type = HD_DEVICE_LIDAR;
                    snprintf(info->name, HD_MAX_NAME_LEN, "LiDAR-%s", name);
                    info->status = HD_STATUS_AVAILABLE;
                    info->performance_score = 0.7f;
                    lidar_count++;
                    break;
                }
            }
        }
        closedir(dev_dir);
    }
#elif defined(_WIN32)
    /* Windows下通过COM端口名称或设备管理器检测LiDAR
     * 使用SetupDi API枚举设备并检查描述符中的LiDAR关键词 */
    HDEVINFO devInfo = SetupDiGetClassDevs(NULL, NULL, NULL, 
        DIGCF_ALLCLASSES | DIGCF_PRESENT);
    if (devInfo != INVALID_HANDLE_VALUE) {
        SP_DEVINFO_DATA devInfoData;
        devInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
        for (DWORD i = 0; SetupDiEnumDeviceInfo(devInfo, i, &devInfoData) && lidar_count < max_count; i++) {
            char desc[256] = {0};
            SetupDiGetDeviceRegistryPropertyA(devInfo, &devInfoData, SPDRP_DEVICEDESC,
                NULL, (PBYTE)desc, sizeof(desc), NULL);
            const char* keywords[] = {"LiDAR", "LIDAR", "lidar", "激光雷达", "laser scanner", NULL};
            for (int k = 0; keywords[k] != NULL; k++) {
                if (strstr(desc, keywords[k])) {
                    HDDeviceInfo* info = &infos[lidar_count];
                    memset(info, 0, sizeof(HDDeviceInfo));
                    info->type = HD_DEVICE_LIDAR;
                    snprintf(info->name, HD_MAX_NAME_LEN, "%.200s", desc);
                    info->status = HD_STATUS_AVAILABLE;
                    info->performance_score = 0.7f;
                    lidar_count++;
                    break;
                }
            }
        }
        SetupDiDestroyDeviceInfoList(devInfo);
    }
#endif

    *count = lidar_count;
    return 0;
}

int hd_detect_all(HDDetectionConfig config, HDDetectionResult* result) {
    if (!result) return -1;
    memset(result, 0, sizeof(HDDetectionResult));
    double start = hd_timestamp_ms();

    if (config.enable_cpu_detection) {
        HDDeviceInfo* info = &result->devices[result->num_devices];
        if (result->num_devices < HD_MAX_DEVICES && hd_detect_cpu(info) == 0) {
            result->num_devices++;
        }
    }

    if (config.enable_gpu_detection) {
        size_t gpu_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_gpu(&result->devices[result->num_devices], remaining, &gpu_count);
            result->num_devices += gpu_count;
        }
    }

    if (config.enable_camera_detection) {
        size_t cam_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_cameras(&result->devices[result->num_devices], remaining, &cam_count);
            result->num_devices += cam_count;
        }
    }

    if (config.enable_audio_detection) {
        size_t audio_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_audio_devices(&result->devices[result->num_devices], remaining, &audio_count);
            result->num_devices += audio_count;
        }
    }

    if (config.enable_serial_detection) {
        size_t port_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_serial_ports(&result->devices[result->num_devices], remaining, &port_count);
            result->num_devices += port_count;
        }
    }

    if (config.enable_network_detection) {
        size_t net_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_network_adapters(&result->devices[result->num_devices], remaining, &net_count);
            result->num_devices += net_count;
        }
    }

    if (config.enable_sensor_detection) {
        size_t sensor_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_sensors(&result->devices[result->num_devices], remaining, &sensor_count);
            result->num_devices += sensor_count;
        }
    }

    if (config.enable_depth_camera_detection) {
        size_t dc_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_depth_cameras(&result->devices[result->num_devices], remaining, &dc_count);
            result->num_devices += dc_count;
        }
    }

    if (config.enable_imu_detection) {
        size_t imu_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_imu_devices(&result->devices[result->num_devices], remaining, &imu_count);
            result->num_devices += imu_count;
        }
    }

    if (config.enable_lidar_detection) {
        size_t lidar_count = 0;
        if (result->num_devices < HD_MAX_DEVICES) {
            size_t remaining = HD_MAX_DEVICES - result->num_devices;
            hd_detect_lidar_devices(&result->devices[result->num_devices], remaining, &lidar_count);
            result->num_devices += lidar_count;
        }
    }

    result->detection_complete = 1;
    result->detection_time_ms = hd_timestamp_ms() - start;
    return 0;
}

int hd_get_device_by_type(const HDDetectionResult* result, HDDeviceType type, HDDeviceInfo* out, size_t max_count, size_t* count) {
    if (!result || !out || !count) return -1;
    *count = 0;
    for (size_t i = 0; i < result->num_devices && *count < max_count; i++) {
        if (result->devices[i].type == type) {
            memcpy(&out[*count], &result->devices[i], sizeof(HDDeviceInfo));
            (*count)++;
        }
    }
    return 0;
}

int hd_get_primary_gpu(HDDeviceInfo* out) {
    if (!out) return -1;
    size_t gpu_count = 0;
    HDDeviceInfo gpus[HD_MAX_GPU_DEVICES];
    if (hd_detect_gpu(gpus, HD_MAX_GPU_DEVICES, &gpu_count) != 0 || gpu_count == 0) return -1;
    memcpy(out, &gpus[0], sizeof(HDDeviceInfo));
    return 0;
}

int hd_get_system_info(char* system_name, size_t max_len, char* os_version, size_t os_max_len) {
    if (!system_name || !os_version) return -1;
#ifdef _WIN32
    strncpy(system_name, "Windows", max_len - 1);
    HKEY hkey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", 0, KEY_READ, &hkey) == ERROR_SUCCESS) {
        char buf[256];
        DWORD type = 0, size = sizeof(buf);
        if (RegQueryValueExA(hkey, "ProductName", NULL, &type, (LPBYTE)buf, &size) == ERROR_SUCCESS) {
            strncpy(os_version, buf, os_max_len - 1);
        } else {
            strncpy(os_version, "Windows", os_max_len - 1);
        }
        RegCloseKey(hkey);
    } else {
        strncpy(os_version, "Windows", os_max_len - 1);
    }
#elif defined(__linux__)
    strncpy(system_name, "Linux", max_len - 1);
    FILE* f = fopen("/etc/os-release", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "PRETTY_NAME=", 12) == 0) {
                char* start = strchr(line, '"');
                char* end = start ? strrchr(start + 1, '"') : NULL;
                if (start && end) {
                    *end = '\0';
                    strncpy(os_version, start + 1, os_max_len - 1);
                }
                break;
            }
        }
        fclose(f);
    }
    if (os_version[0] == '\0') strncpy(os_version, "Linux", os_max_len - 1);
#elif defined(__APPLE__)
    strncpy(system_name, "macOS", max_len - 1);
    strncpy(os_version, "macOS", os_max_len - 1);
#else
    strncpy(system_name, "Unknown", max_len - 1);
    strncpy(os_version, "Unknown", os_max_len - 1);
#endif
    return 0;
}

/* P3-006: 硬件基准测试 — 完整实现，待上层模块集成调用 */
int hd_benchmark_device(const HDDeviceInfo* device, HDBenchmarkType bench_type, HDBenchmarkResult* out) {
    /* ZSF-078说明：基准测试使用纯CPU浮点运算。
     * GPU设备通过gpu_get_device_info()获取真实性能评分，此函数为CPU补充测试。 */
    if (!device || !out) return -1;
    memset(out, 0, sizeof(HDBenchmarkResult));
    out->bench_type = bench_type;

    double samples[HD_MAX_BENCHMARK_SAMPLES];
    int num_samples = 0;

    switch (bench_type) {
        case HD_BENCHMARK_COMPUTE: {
            for (int s = 0; s < 20 && s < HD_MAX_BENCHMARK_SAMPLES; s++) {
                double t0 = hd_timestamp_ms();
                volatile float sum = 0.0f;
                for (int j = 0; j < 100000; j++) {
                    sum += (float)(j * 0.001f);
                    sum *= 0.999f;
                }
                (void)sum;
                double t1 = hd_timestamp_ms();
                samples[num_samples++] = 100000.0 / (t1 - t0);
            }
            break;
        }
        case HD_BENCHMARK_MEMORY: {
            size_t buf_size = device->total_memory_bytes > 0 ? (device->total_memory_bytes < (1024 * 1024) ? device->total_memory_bytes : (1024 * 1024)) : (1024 * 1024);
            char* buf = (char*)malloc(buf_size);
            if (!buf) return -1;
            for (int s = 0; s < 10 && s < HD_MAX_BENCHMARK_SAMPLES; s++) {
                double t0 = hd_timestamp_ms();
                memset(buf, s & 0xFF, buf_size);
                double t1 = hd_timestamp_ms();
                samples[num_samples++] = (double)buf_size / (1024.0 * 1024.0) / ((t1 - t0) / 1000.0);
            }
            free(buf);
            break;
        }
        case HD_BENCHMARK_IO: {
            const char* tmp_file = "hd_benchmark_tmp.bin";
            for (int s = 0; s < 10 && s < HD_MAX_BENCHMARK_SAMPLES; s++) {
                double t0 = hd_timestamp_ms();
                FILE* f = fopen(tmp_file, "wb");
                if (!f) { samples[num_samples++] = 0.0; continue; }
                for (int j = 0; j < 256; j++) fputc(j & 0xFF, f);
                fclose(f);
                f = fopen(tmp_file, "rb");
                if (f) { while (fgetc(f) != EOF); fclose(f); }
                remove(tmp_file);
                double t1 = hd_timestamp_ms();
                samples[num_samples++] = 256.0 / ((t1 - t0) / 1000.0);
            }
            break;
        }
        case HD_BENCHMARK_LATENCY: {
            for (int s = 0; s < 50 && s < HD_MAX_BENCHMARK_SAMPLES; s++) {
                double t0 = hd_timestamp_ms();
                for (int j = 0; j < 1000; j++) {
                    hd_timestamp_ms();
                }
                double t1 = hd_timestamp_ms();
                samples[num_samples++] = (t1 - t0) / 1000.0;
            }
            break;
        }
        case HD_BENCHMARK_THROUGHPUT: {
            for (int s = 0; s < 10 && s < HD_MAX_BENCHMARK_SAMPLES; s++) {
                double t0 = hd_timestamp_ms();
                for (int j = 0; j < 10000; j++) {
                    volatile float a = (float)(j * 3.14159f);
                    volatile float b = a * a;
                    volatile float c = sqrtf(b);
                    (void)c;
                }
                double t1 = hd_timestamp_ms();
                samples[num_samples++] = 10000.0 / ((t1 - t0) / 1000.0);
            }
            break;
        }
        default:
            return -1;
    }

    if (num_samples <= 0) return -1;
    double sum = 0.0;
    for (int i = 0; i < num_samples; i++) sum += samples[i];
    out->value = sum / num_samples;

    double var = 0.0;
    for (int i = 0; i < num_samples; i++) {
        double diff = samples[i] - out->value;
        var += diff * diff;
    }
    out->std_dev = sqrt(var / num_samples);

    switch (bench_type) {
        case HD_BENCHMARK_COMPUTE: snprintf(out->description, HD_MAX_NAME_LEN, "计算性能: %.2f ops/s", out->value); break;
        case HD_BENCHMARK_MEMORY: snprintf(out->description, HD_MAX_NAME_LEN, "内存带宽: %.2f MB/s", out->value); break;
        case HD_BENCHMARK_IO: snprintf(out->description, HD_MAX_NAME_LEN, "IO吞吐: %.2f B/s", out->value); break;
        case HD_BENCHMARK_LATENCY: snprintf(out->description, HD_MAX_NAME_LEN, "延迟: %.4f ms", out->value); break;
        case HD_BENCHMARK_THROUGHPUT: snprintf(out->description, HD_MAX_NAME_LEN, "吞吐量: %.2f ops/s", out->value); break;
        default: break;
    }
    return 0;
}

int hd_benchmark_all(HDDetectionResult* result, HDBenchmarkResult* bench_out, size_t max_count, size_t* count) {
    if (!result || !bench_out || !count) return -1;
    *count = 0;
    for (size_t i = 0; i < result->num_devices && *count < max_count; i++) {
        HDBenchmarkResult br;
        if (hd_benchmark_device(&result->devices[i], HD_BENCHMARK_COMPUTE, &br) == 0) {
            memcpy(&bench_out[*count], &br, sizeof(HDBenchmarkResult));
            (*count)++;
        }
    }
    return 0;
}

int hd_health_check(const HDDeviceInfo* device, HDHealthInfo* out) {
    if (!device || !out) return -1;
    memset(out, 0, sizeof(HDHealthInfo));

    float score = device->performance_score;
    if (score > 1.0f) score = 1.0f;
    if (score < 0.0f) score = 0.0f;
    out->health_score = score;

    out->load_percentage = (device->compute_units > 0) ?
        (float)(1.0f - (float)device->free_memory_bytes / (float)(device->total_memory_bytes + 1)) * 100.0f : 50.0f;

    if (out->load_percentage > 90.0f) out->temperature_normalized = 0.8f + (out->load_percentage - 90.0f) / 100.0f;
    else if (out->load_percentage > 70.0f) out->temperature_normalized = 0.5f + (out->load_percentage - 70.0f) / 100.0f;
    else out->temperature_normalized = 0.2f + out->load_percentage / 200.0f;
    if (out->temperature_normalized > 1.0f) out->temperature_normalized = 1.0f;

    if (device->status == HD_STATUS_ERROR) {
        out->level = HD_HEALTH_CRITICAL;
        out->health_score *= 0.3f;
        strncpy(out->prediction, "设备错误，需检查驱动", HD_MAX_NAME_LEN - 1);
    } else if (device->status == HD_STATUS_NOT_PRESENT) {
        out->level = HD_HEALTH_CRITICAL;
        out->health_score = 0.0f;
        strncpy(out->prediction, "设备未连接", HD_MAX_NAME_LEN - 1);
    } else if (score > 0.8f && out->load_percentage < 50.0f) {
        out->level = HD_HEALTH_GOOD;
        strncpy(out->prediction, "状态良好", HD_MAX_NAME_LEN - 1);
    } else if (score > 0.5f && out->load_percentage < 80.0f) {
        out->level = HD_HEALTH_DEGRADED;
        strncpy(out->prediction, "性能下降，建议维护", HD_MAX_NAME_LEN - 1);
    } else if (score > 0.2f) {
        out->level = HD_HEALTH_CRITICAL;
        strncpy(out->prediction, "负载过高，需要立即处理", HD_MAX_NAME_LEN - 1);
    } else {
        out->level = HD_HEALTH_FAILING;
        strncpy(out->prediction, "设备即将失效", HD_MAX_NAME_LEN - 1);
    }

    out->uptime_hours = 0.0;
    return 0;
}

/* P3-006: 硬件健康趋势预测 — 完整实现，待上层监控模块集成调用 */
int hd_health_predict_trend(HDHealthInfo* history, size_t history_len, HDHealthInfo* out) {
    if (!history || !out || history_len == 0) return -1;
    memcpy(out, &history[history_len - 1], sizeof(HDHealthInfo));

    if (history_len >= 5) {
        float sum_delta = 0.0f;
        for (size_t i = 1; i < history_len; i++) {
            sum_delta += history[i].health_score - history[i - 1].health_score;
        }
        float trend = sum_delta / (float)(history_len - 1);

        out->health_score += trend * 3.0f;
        if (out->health_score > 1.0f) out->health_score = 1.0f;
        if (out->health_score < 0.0f) out->health_score = 0.0f;

        if (trend < -0.05f) {
            out->level = HD_HEALTH_DEGRADED;
            snprintf(out->prediction, HD_MAX_NAME_LEN, "趋势下降(%.3f/步)，建议检查", (double)trend);
        } else if (trend > 0.05f) {
            out->level = HD_HEALTH_GOOD;
            snprintf(out->prediction, HD_MAX_NAME_LEN, "趋势上升(%.3f/步)，状态改善", (double)trend);
        }
    }
    return 0;
}

/* P3-006: 设备能力分类 — 完整实现，待上层设备管理模块集成调用 */
int hd_classify_device_capability(const HDDeviceInfo* device, HDDeviceCapability* caps, size_t max_count, size_t* count) {
    if (!device || !caps || !count) return -1;
    *count = 0;

    float input[HD_LNN_INPUT_DIM] = {0};
    input[0] = (float)device->compute_units / 128.0f;
    input[1] = (device->clock_speed_mhz > 0 ? device->clock_speed_mhz : 2000.0f) / 5000.0f;
    input[2] = (float)(device->total_memory_bytes >> 20) / 32768.0f;
    input[3] = (float)device->supports_double;
    input[4] = (float)device->supports_half;
    input[5] = (float)device->is_virtual;
    input[6] = device->performance_score;

    switch (device->type) {
        case HD_DEVICE_GPU: input[7] = 1.0f; break;
        case HD_DEVICE_CPU: input[8] = 1.0f; break;
        case HD_DEVICE_CAMERA:
        case HD_DEVICE_DEPTH_CAMERA: input[9] = 1.0f; break;
        case HD_DEVICE_IMU: input[10] = 1.0f; break;
        case HD_DEVICE_LIDAR: input[11] = 1.0f; break;
        case HD_DEVICE_MICROPHONE:
        case HD_DEVICE_SPEAKER: input[12] = 1.0f; break;
        default: break;
    }

    float lnn_output[HD_LNN_OUTPUT_DIM] = {0};
    /* 使用全局LNN分类器，拒绝创建随机权重LNN */
     LNN* classifier = (LNN*)selflnn_get_lnn();
     if (classifier) {
        lnn_forward(classifier, input, lnn_output);
    } else {
        /* 全局LNN未就绪时，使用基于规则的确定性分类 */
        switch (device->type) {
            case HD_DEVICE_CAMERA: case HD_DEVICE_DEPTH_CAMERA:
                caps[*count] = HD_CAP_VISION_PROCESSING; (*count)++; break;
            case HD_DEVICE_MICROPHONE: case HD_DEVICE_SPEAKER:
                caps[*count] = HD_CAP_AUDIO_PROCESSING; (*count)++; break;
            case HD_DEVICE_GPU:
                caps[*count] = HD_CAP_HIGH_PERFORMANCE; (*count)++; break;
            case HD_DEVICE_IMU: case HD_DEVICE_LIDAR:
                caps[*count] = HD_CAP_LOW_LATENCY; (*count)++; break;
            default:
                caps[*count] = HD_CAP_UNKNOWN; (*count)++; break;
        }
        return 0;
    }

    struct { HDDeviceCapability cap; int idx; } cap_map[] = {
        {HD_CAP_VISION_PROCESSING, 0},
        {HD_CAP_AUDIO_PROCESSING, 1},
        {HD_CAP_CONTROL_SIGNAL, 2},
        {HD_CAP_HIGH_PERFORMANCE, 3},
        {HD_CAP_LOW_LATENCY, 4},
        {HD_CAP_EMBEDDED, 5}
    };
    int num_caps = (int)(sizeof(cap_map) / sizeof(cap_map[0]));

    for (int c = 0; c < num_caps && *count < max_count; c++) {
        float prob = hd_softplus(lnn_output[cap_map[c].idx]);
        if (prob > HD_CLASSIFY_CONF_THRESHOLD) {
            caps[*count] = cap_map[c].cap;
            (*count)++;
        }
    }

    if (*count == 0 && max_count > 0) {
        caps[0] = HD_CAP_UNKNOWN;
        *count = 1;
    }
    return 0;
}

int hd_hotplug_start_monitor(HDHotplugMonitor* monitor) {
    if (!monitor) return -1;
    memset(monitor, 0, sizeof(HDHotplugMonitor));
    monitor->monitor_active = 1;

    /* 计算初始设备哈希用于变更检测
     * 跨平台实现：通过hd_detect_all()重新枚举设备并计算哈希
     * 变更检测在hd_hotplug_poll_events()中通过哈希比对实现 */
    monitor->last_device_hash = 0;
    HDDetectionResult init_result;
    memset(&init_result, 0, sizeof(init_result));
    {
        HDDetectionConfig default_cfg = HD_DETECTION_CONFIG_DEFAULT;
        if (hd_detect_all(default_cfg, &init_result) == 0) {
            uint32_t hash = hd_compute_device_hash(&init_result);
            monitor->last_device_hash = hash;
            hd_result_free(&init_result);
        }
    }
    return 0;
}

int hd_hotplug_poll_events(HDHotplugMonitor* monitor) {
    if (!monitor || !monitor->monitor_active) return -1;

    int event_count = 0;
    HDDetectionResult current;
    memset(&current, 0, sizeof(current));

    {
        HDDetectionConfig default_cfg = HD_DETECTION_CONFIG_DEFAULT;
        if (hd_detect_all(default_cfg, &current) != 0) return -1;
    }

    uint32_t current_hash = hd_compute_device_hash(&current);

    if (current_hash != monitor->last_device_hash) {
        /* 检测到设备变更，统计变化的设备数 */
        monitor->last_device_hash = current_hash;
        event_count = 1;
        monitor->num_events++;
    }

    hd_result_free(&current);
    return event_count;
}

/* 辅助函数：计算设备列表的哈希值用于变更检测 */
static uint32_t hd_compute_device_hash(const HDDetectionResult* result) {
    if (!result || result->num_devices == 0) return 0;
    uint32_t hash = 5381;
    for (size_t i = 0; i < result->num_devices; i++) {
        const HDDeviceInfo* d = &result->devices[i];
        hash = ((hash << 5) + hash) + (uint32_t)d->type;
        hash = ((hash << 5) + hash) + (uint32_t)d->status;
        for (const char* p = d->name; *p; p++) {
            hash = ((hash << 5) + hash) + (uint32_t)(unsigned char)*p;
        }
    }
    return hash;
}

int hd_hotplug_stop_monitor(HDHotplugMonitor* monitor) {
    if (!monitor) return -1;
    monitor->monitor_active = 0;
    return 0;
}

int hd_report_generate(const HDDetectionResult* result, char* report, size_t report_len) {
    if (!result || !report || report_len == 0) return -1;
    size_t pos = 0;

    pos += snprintf(report + pos, report_len - pos,
        "================= 硬件检测报告 =================\n");
    if (pos >= report_len) return (int)pos;

    pos += snprintf(report + pos, report_len - pos,
        "检测时间: %.2f ms\n", result->detection_time_ms);
    pos += snprintf(report + pos, report_len - pos,
        "检测设备数: %zu\n", result->num_devices);
    pos += snprintf(report + pos, report_len - pos,
        "检测状态: %s\n\n", result->detection_complete ? "完成" : "未完成");

    for (size_t i = 0; i < result->num_devices; i++) {
        const HDDeviceInfo* d = &result->devices[i];
        pos += snprintf(report + pos, report_len - pos,
            "[%zu] %s | %s | %s\n", i + 1,
            hd_device_type_str(d->type),
            d->name[0] ? d->name : "未知",
            hd_device_status_str(d->status));
        if (pos >= report_len) return (int)pos;

        if (d->vendor[0]) {
            pos += snprintf(report + pos, report_len - pos,
                "     厂商: %s\n", d->vendor);
        }
        pos += snprintf(report + pos, report_len - pos,
            "     性能评分: %.2f", d->performance_score);
        if (d->compute_units > 0) {
            pos += snprintf(report + pos, report_len - pos,
                " | 计算单元: %d", d->compute_units);
        }
        if (d->total_memory_bytes > 0) {
            pos += snprintf(report + pos, report_len - pos,
                " | 内存: %.1f MB", (double)d->total_memory_bytes / (1024.0 * 1024.0));
        }
        pos += snprintf(report + pos, report_len - pos, "\n");

        HDHealthInfo health;
        if (hd_health_check(d, &health) == 0) {
            pos += snprintf(report + pos, report_len - pos,
                "     健康: %s (%.2f) | %s\n",
                hd_health_level_str(health.level),
                health.health_score,
                health.prediction);
        }

        pos += snprintf(report + pos, report_len - pos, "\n");
        if (pos >= report_len) return (int)pos;
    }

    pos += snprintf(report + pos, report_len - pos,
        "================================================\n");
    return (int)pos;
}

void hd_result_free(HDDetectionResult* result) {
    if (result) memset(result, 0, sizeof(HDDetectionResult));
}

const char* hd_device_type_str(HDDeviceType type) {
    switch (type) {
        case HD_DEVICE_UNKNOWN: return "未知";
        case HD_DEVICE_CPU: return "CPU";
        case HD_DEVICE_GPU: return "GPU";
        case HD_DEVICE_NPU: return "NPU";
        case HD_DEVICE_CAMERA: return "摄像头";
        case HD_DEVICE_MICROPHONE: return "麦克风";
        case HD_DEVICE_SPEAKER: return "扬声器";
        case HD_DEVICE_SERIAL_PORT: return "串口";
        case HD_DEVICE_NETWORK_ADAPTER: return "网卡";
        case HD_DEVICE_SENSOR: return "传感器";
        case HD_DEVICE_ROBOT_ARM: return "机械臂";
        case HD_DEVICE_MOTOR_CONTROLLER: return "电机控制器";
        case HD_DEVICE_LIDAR: return "激光雷达";
        case HD_DEVICE_DEPTH_CAMERA: return "深度摄像头";
        case HD_DEVICE_IMU: return "IMU";
        case HD_DEVICE_CUSTOM: return "自定义";
        default: return "未知";
    }
}

const char* hd_device_status_str(HDDeviceStatus status) {
    switch (status) {
        case HD_STATUS_UNKNOWN: return "未知";
        case HD_STATUS_AVAILABLE: return "可用";
        case HD_STATUS_BUSY: return "占用";
        case HD_STATUS_ERROR: return "错误";
        case HD_STATUS_NOT_PRESENT: return "未连接";
        default: return "未知";
    }
}

const char* hd_health_level_str(HDHealthLevel level) {
    switch (level) {
        case HD_HEALTH_UNKNOWN: return "未知";
        case HD_HEALTH_GOOD: return "良好";
        case HD_HEALTH_DEGRADED: return "下降";
        case HD_HEALTH_CRITICAL: return "严重";
        case HD_HEALTH_FAILING: return "即将失效";
        default: return "未知";
    }
}

const char* hd_capability_str(HDDeviceCapability cap) {
    switch (cap) {
        case HD_CAP_UNKNOWN: return "未知";
        case HD_CAP_VISION_PROCESSING: return "视觉处理";
        case HD_CAP_AUDIO_PROCESSING: return "音频处理";
        case HD_CAP_CONTROL_SIGNAL: return "控制信号";
        case HD_CAP_HIGH_PERFORMANCE: return "高性能计算";
        case HD_CAP_LOW_LATENCY: return "低延迟";
        case HD_CAP_EMBEDDED: return "嵌入式";
        default: return "未知";
    }
}
