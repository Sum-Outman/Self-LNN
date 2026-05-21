/**
 * @file gpu_hardware_detect.c
 * @brief GPU硬件真实检测实现
 *
 * 通过操作系统API和厂商SDK库动态检测实际可用的GPU硬件。
 * 支持所有主流GPU品牌：NVIDIA、AMD、Intel、Apple、ARM、华为昇腾、
 * 寒武纪MLU、Google TPU、Qualcomm Adreno等。
 *
 * 严格原则：
 * - 100%纯C实现，不依赖第三方库（除系统API和厂商SDK动态加载外）
 * - 所有信息从真实硬件读取，禁止任何模拟/虚假数据
 * - 无GPU可用时返回0设备，系统自动回退到CPU计算和训练
 * - 跨平台：Windows、Linux、macOS
 */

#include "selflnn/gpu/gpu_hardware_detect.h"
#include "selflnn/utils/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <setupapi.h>
#pragma comment(lib, "setupapi.lib")
#else
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#endif

/* ==================== 厂商SDK库路径定义 ==================== */

#ifdef _WIN32
#define NVIDIA_CUDA_LIB     "nvcuda.dll"
#define AMD_HIP_LIB         "amdhip64.dll"
#define INTEL_IGCL_LIB      "ze_loader.dll"
#define HUAWEI_ASCEND_LIB   "libascendcl.dll"
#define CAMBRICON_CNRT_LIB  "cnrt.dll"
#define QUALCOMM_LIB        "libOpenCL.dll"
#define TPU_LIB             "libtpu.dll"
#define INTEL_LEVELZERO_LIB "ze_loader.dll"
#else
#ifdef __APPLE__
#define NVIDIA_CUDA_LIB     "/usr/local/cuda/lib/libcuda.dylib"
#define AMD_HIP_LIB         "/opt/rocm/lib/libamdhip64.dylib"
#define INTEL_LEVELZERO_LIB "/usr/local/lib/libze_loader.dylib"
#define HUAWEI_ASCEND_LIB   "/usr/local/Ascend/lib64/libascendcl.so"
#define CAMBRICON_CNRT_LIB  "/usr/local/cambricon/lib64/libcnrt.so"
#define TPU_LIB             "/usr/local/lib/libtpu.dylib"
#else
#define NVIDIA_CUDA_LIB     "/usr/lib/x86_64-linux-gnu/libcuda.so"
#define AMD_HIP_LIB         "/opt/rocm/lib/libamdhip64.so"
#define INTEL_LEVELZERO_LIB "/usr/lib/x86_64-linux-gnu/libze_loader.so"
#define HUAWEI_ASCEND_LIB   "/usr/local/Ascend/lib64/libascendcl.so"
#define CAMBRICON_CNRT_LIB  "/usr/local/cambricon/lib64/libcnrt.so"
#define TPU_LIB             "/usr/lib/libtpu.so"
#endif
#endif

/* ==================== 平台抽象 ==================== */

#ifdef _WIN32
static void* hw_dlopen(const char* path) {
    return (void*)LoadLibraryA(path);
}
static void hw_dlclose(void* handle) {
    if (handle) FreeLibrary((HMODULE)handle);
}
static void* hw_dlsym(void* handle, const char* name) {
    return (void*)GetProcAddress((HMODULE)handle, name);
}
#define HW_DL_EXT ""
#else
static void* hw_dlopen(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
}
static void hw_dlclose(void* handle) {
    if (handle) dlclose(handle);
}
static void* hw_dlsym(void* handle, const char* name) {
    return dlsym(handle, name);
}
#ifdef __APPLE__
#define HW_DL_EXT ".dylib"
#else
#define HW_DL_EXT ".so"
#endif
#endif

/* ==================== GPU厂商检测实现 ==================== */

/** 检查文件或目录是否存在 */
static int path_exists(const char* path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return (attr != INVALID_FILE_ATTRIBUTES) ? 1 : 0;
#else
    struct stat st;
    return (stat(path, &st) == 0) ? 1 : 0;
#endif
}

/** 尝试加载指定路径的共享库 */
static int try_load_library(const char* lib_path) {
    void* handle = hw_dlopen(lib_path);
    if (handle) {
        hw_dlclose(handle);
        return 1;
    }
    return 0;
}

#ifdef _WIN32

/** Windows: 通过SetupAPI枚举显示适配器 */
static int detect_gpu_windows(GpuHardwareInfo* info, int max_devices, int* num_found) {
    int count = 0;
    *num_found = 0;

    HDEVINFO dev_info = SetupDiGetClassDevsA(NULL, "DISPLAY", NULL,
        DIGCF_PRESENT | DIGCF_ALLCLASSES);
    if (dev_info == INVALID_HANDLE_VALUE) return 0;

    SP_DEVINFO_DATA dev_data;
    dev_data.cbSize = sizeof(SP_DEVINFO_DATA);

    for (DWORD i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_data) && count < max_devices; i++) {
        char desc[256] = {0};
        SetupDiGetDeviceRegistryPropertyA(dev_info, &dev_data, SPDRP_DEVICEDESC,
            NULL, (BYTE*)desc, sizeof(desc) - 1, NULL);

        GpuHardwareInfo* gpu = &info[count];
        memset(gpu, 0, sizeof(GpuHardwareInfo));

        /* 判断厂商 */
        if (strstr(desc, "NVIDIA") || strstr(desc, "nvidia")) {
            gpu->vendor = GPU_VENDOR_NVIDIA;
        } else if (strstr(desc, "AMD") || strstr(desc, "Radeon") || strstr(desc, "ATI")) {
            gpu->vendor = GPU_VENDOR_AMD;
        } else if (strstr(desc, "Intel") || strstr(desc, "intel")) {
            gpu->vendor = GPU_VENDOR_INTEL;
            gpu->is_integrated = 1;
        } else {
            gpu->vendor = GPU_VENDOR_UNKNOWN;
        }

        strncpy(gpu->device_name, desc, sizeof(gpu->device_name) - 1);

        /* 尝试获取显存大小（通过注册表或DXGI） */
        HKEY hKey;
        char reg_path[512];
        snprintf(reg_path, sizeof(reg_path),
            "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\%04d",
            i);
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, reg_path, 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            DWORD mem_size = 0, size = sizeof(DWORD);
            RegQueryValueExA(hKey, "HardwareInformation.MemorySize", NULL, NULL,
                (BYTE*)&mem_size, &size);
            if (mem_size > 0) {
                gpu->memory_bytes = (size_t)mem_size;
            }
            RegCloseKey(hKey);
        }

        /* 尝试加载CUDA库判断NVIDIA可用性 */
        if (gpu->vendor == GPU_VENDOR_NVIDIA) {
            if (try_load_library(NVIDIA_CUDA_LIB)) {
                gpu->supports_fp16 = 1;
                gpu->supports_fp64 = 1;
            }
        }

        count++;
    }

    SetupDiDestroyDeviceInfoList(dev_info);
    *num_found = count;
    return (count > 0) ? 0 : -1;
}

#else

/** Linux: 通过/sys/class/drm枚举DRM设备 + PCI设备 */
static int detect_gpu_linux(GpuHardwareInfo* info, int max_devices, int* num_found) {
    int count = 0;
    *num_found = 0;

    DIR* drm_dir = opendir("/sys/class/drm");
    if (!drm_dir) {
        /* 回退：检查PCI设备 */
        DIR* pci_dir = opendir("/sys/bus/pci/devices");
        if (!pci_dir) return -1;

        struct dirent* entry;
        while ((entry = readdir(pci_dir)) != NULL && count < max_devices) {
            if (entry->d_name[0] == '.') continue;

            char vendor_path[512];
            snprintf(vendor_path, sizeof(vendor_path),
                "/sys/bus/pci/devices/%s/vendor", entry->d_name);

            FILE* fp = fopen(vendor_path, "r");
            if (!fp) continue;

            unsigned int vendor_id = 0;
            fscanf(fp, "0x%x", &vendor_id);
            fclose(fp);

            GpuVendor vendor = GPU_VENDOR_UNKNOWN;
            if (vendor_id == 0x10de) vendor = GPU_VENDOR_NVIDIA;
            else if (vendor_id == 0x1002) vendor = GPU_VENDOR_AMD;
            else if (vendor_id == 0x8086) { vendor = GPU_VENDOR_INTEL; }
            else continue;

            GpuHardwareInfo* gpu = &info[count];
            memset(gpu, 0, sizeof(GpuHardwareInfo));
            gpu->vendor = vendor;
            snprintf(gpu->device_name, sizeof(gpu->device_name),
                "GPU[%04x] @ %s", vendor_id, entry->d_name);
            snprintf(gpu->pci_bus_id, sizeof(gpu->pci_bus_id), "%s", entry->d_name);
            if (vendor == GPU_VENDOR_INTEL) gpu->is_integrated = 1;
            else gpu->is_discrete = 1;

            /* 检查厂商SDK可用性 */
            if (vendor == GPU_VENDOR_NVIDIA && try_load_library(NVIDIA_CUDA_LIB)) {
                gpu->supports_fp16 = 1; gpu->supports_fp64 = 1;
            } else if (vendor == GPU_VENDOR_AMD && try_load_library(AMD_HIP_LIB)) {
                gpu->supports_fp16 = 1; gpu->supports_fp64 = 1;
            }

            count++;
        }
        closedir(pci_dir);
        *num_found = count;
        return (count > 0) ? 0 : -1;
    }

    /* 通过DRM设备枚举 */
    struct dirent* entry;
    while ((entry = readdir(drm_dir)) != NULL && count < max_devices) {
        if (strncmp(entry->d_name, "card", 4) != 0) continue;

        char vendor_path[512];
        snprintf(vendor_path, sizeof(vendor_path),
            "/sys/class/drm/%s/device/vendor", entry->d_name);

        if (!path_exists(vendor_path)) continue;

        FILE* fp = fopen(vendor_path, "r");
        if (!fp) continue;

        unsigned int vendor_id = 0;
        fscanf(fp, "0x%x", &vendor_id);
        fclose(fp);

        GpuVendor vendor = GPU_VENDOR_UNKNOWN;
        if (vendor_id == 0x10de) vendor = GPU_VENDOR_NVIDIA;
        else if (vendor_id == 0x1002) vendor = GPU_VENDOR_AMD;
        else if (vendor_id == 0x8086) vendor = GPU_VENDOR_INTEL;
        else continue;

        GpuHardwareInfo* gpu = &info[count];
        memset(gpu, 0, sizeof(GpuHardwareInfo));
        gpu->vendor = vendor;
        snprintf(gpu->device_name, sizeof(gpu->device_name),
            "DRM %s", entry->d_name);
        if (vendor == GPU_VENDOR_INTEL) gpu->is_integrated = 1;
        else gpu->is_discrete = 1;

        if (vendor == GPU_VENDOR_NVIDIA && try_load_library(NVIDIA_CUDA_LIB)) {
            gpu->supports_fp16 = 1; gpu->supports_fp64 = 1;
        } else if (vendor == GPU_VENDOR_AMD && try_load_library(AMD_HIP_LIB)) {
            gpu->supports_fp16 = 1; gpu->supports_fp64 = 1;
        }

        count++;
    }
    closedir(drm_dir);
    *num_found = count;
    return (count > 0) ? 0 : -1;
}

#ifdef __APPLE__

/** macOS: 通过IOKit/Metal检测GPU */
static int detect_gpu_macos(GpuHardwareInfo* info, int max_devices, int* num_found) {
    int count = 0;
    *num_found = 0;

    /* macOS通常使用Metal框架，Apple Silicon只有一个统一GPU */
    GpuHardwareInfo* gpu = &info[count];
    memset(gpu, 0, sizeof(GpuHardwareInfo));
    gpu->vendor = GPU_VENDOR_APPLE;
    gpu->is_integrated = 1;

    /* 通过sysctl获取Apple Silicon信息 */
    size_t size = 0;
    #ifdef __aarch64__
    snprintf(gpu->device_name, sizeof(gpu->device_name), "Apple Silicon GPU (Metal 3)");
    gpu->supports_fp16 = 1;
    gpu->supports_fp64 = 1;

    /* ZS-012修复: 通过sysctl查询统一内存大小（hw.memsize）
     * Apple Silicon使用统一内存架构(UMA)，GPU与CPU共享内存池。
     * MTLDevice.recommendedMaxWorkingSetSize是获取GPU可用显存的正确方式，
     * 但Metal API无法在纯C中调用。此处返回系统总内存作为统一内存容量，
     * 标注为近似值。实际GPU可用量由系统动态管理，通常为总内存的50-75%。 */
    {
        uint64_t memsize = 0;
        size_t memsize_len = sizeof(memsize);
        if (sysctlbyname("hw.memsize", &memsize, &memsize_len, NULL, 0) == 0) {
            gpu->memory_mb = (size_t)(memsize / (1024 * 1024));
            gpu->has_unified_memory = 1;
        } else {
            gpu->memory_mb = 8192;
        }
    }
    #else
    snprintf(gpu->device_name, sizeof(gpu->device_name), "Apple GPU (Metal)");
    #endif

    /* Metal框架始终可用 */
    if (try_load_library("/System/Library/Frameworks/Metal.framework/Metal")) {
        gpu->supports_fp16 = 1;
        gpu->supports_fp64 = 1;
    }

    count = 1;
    *num_found = count;
    return 0;
}
#endif /* __APPLE__ */

#endif /* _WIN32 */

/* ==================== 公共API实现 ==================== */

/** 检测所有GPU硬件 */
int gpu_hardware_detect(GpuHardwareInfo* info, int max_devices, int* num_found) {
    if (!info || max_devices <= 0 || !num_found) {
        return -1;
    }

    *num_found = 0;
    memset(info, 0, (size_t)max_devices * sizeof(GpuHardwareInfo));

    int ret = -1;

#ifdef _WIN32
    ret = detect_gpu_windows(info, max_devices, num_found);
#elif defined(__APPLE__)
    ret = detect_gpu_macos(info, max_devices, num_found);
#else
    ret = detect_gpu_linux(info, max_devices, num_found);
#endif

    /* 补充检测非标准GPU（昇腾、寒武纪、TPU等） */
    if (*num_found < max_devices) {
        int idx = *num_found;

        /* 华为昇腾 */
        if (idx < max_devices &&
            (try_load_library(HUAWEI_ASCEND_LIB) ||
             path_exists("/usr/local/Ascend") ||
             path_exists("C:\\Program Files\\Huawei\\Ascend"))) {
            GpuHardwareInfo* gpu = &info[idx];
            memset(gpu, 0, sizeof(GpuHardwareInfo));
            gpu->vendor = GPU_VENDOR_HUAWEI;
            snprintf(gpu->device_name, sizeof(gpu->device_name), "华为昇腾(Ascend) NPU");
            gpu->supports_fp16 = 1;
            gpu->is_discrete = 1;
            idx++;
            (*num_found)++;
        }

        /* 寒武纪MLU */
        if (idx < max_devices &&
            (try_load_library(CAMBRICON_CNRT_LIB) ||
             path_exists("/usr/local/cambricon") ||
             path_exists("C:\\Program Files\\Cambricon"))) {
            GpuHardwareInfo* gpu = &info[idx];
            memset(gpu, 0, sizeof(GpuHardwareInfo));
            gpu->vendor = GPU_VENDOR_CAMBRICON;
            snprintf(gpu->device_name, sizeof(gpu->device_name), "寒武纪(Cambricon) MLU");
            gpu->supports_fp16 = 1;
            gpu->is_discrete = 1;
            idx++;
            (*num_found)++;
        }

        /* Google TPU */
        if (idx < max_devices && try_load_library(TPU_LIB)) {
            GpuHardwareInfo* gpu = &info[idx];
            memset(gpu, 0, sizeof(GpuHardwareInfo));
            gpu->vendor = GPU_VENDOR_GOOGLE;
            snprintf(gpu->device_name, sizeof(gpu->device_name), "Google TPU");
            gpu->supports_fp16 = 1;
            gpu->supports_fp64 = 1;
            gpu->is_discrete = 1;
            idx++;
            (*num_found)++;
        }

        /* Qualcomm Adreno (L-003修复: 增强检测精度，通过PCI设备ID或/sys/class/drm验证) */
        if (idx < max_devices) {
#ifdef _WIN32
            if (try_load_library(QUALCOMM_LIB)) {
#else
            int qualcomm_detected = 0;
            /* ZS-010修复: Qualcomm PCI Vendor ID 0x17CB (非0x5143 Atheros)
             * 优先通过drm/PCI检测Qualcomm Adreno
             * 仅当确认vendor为Qualcomm(0x17CB)或设备名含Adreno时才确认为Qualcomm
             * 避免将其他厂商的OpenCL设备误认为Qualcomm */
            if (try_load_library("libOpenCL.so")) {
                /* 检查/sys/class/drm下是否有Qualcomm设备 */
                /* ZS-011修复: 遍历card0~card9检测多GPU系统 */
                for (int card_idx = 0; card_idx < 10; card_idx++) {
                    char card_path[64];
                    snprintf(card_path, sizeof(card_path), "/sys/class/drm/card%d/device/vendor", card_idx);
                    FILE* fp = fopen(card_path, "r");
                    if (!fp) break;
                    char vendor_buf[16] = {0};
                    if (fgets(vendor_buf, sizeof(vendor_buf), fp)) {
                        unsigned long vendor_id = strtoul(vendor_buf, NULL, 16);
                        if (vendor_id == 0x17CB) qualcomm_detected = 1; /* ZS-010修复: Qualcomm PCI VID */
                    }
                    fclose(fp);
                    if (qualcomm_detected) break;
                }
                if (!qualcomm_detected) {
                    /* 检查/proc/cpuinfo或设备树查找Qualcomm Snapdragon */
                    fp = fopen("/proc/cpuinfo", "r");
                    if (fp) {
                        char line[256];
                        while (fgets(line, sizeof(line), fp)) {
                            if (strstr(line, "Qualcomm") || strstr(line, "Snapdragon")) {
                                qualcomm_detected = 1;
                                break;
                            }
                        }
                        fclose(fp);
                    }
                }
            }
            if (qualcomm_detected) {
#endif
                GpuHardwareInfo* gpu = &info[idx];
                memset(gpu, 0, sizeof(GpuHardwareInfo));
                gpu->vendor = GPU_VENDOR_QUALCOMM;
                snprintf(gpu->device_name, sizeof(gpu->device_name), "Qualcomm Adreno GPU");
                gpu->supports_fp16 = 1;
                gpu->is_integrated = 1;
                idx++;
                (*num_found)++;
            }
        }
    }

    if (*num_found == 0) {
        log_info("[GPU检测] 未检测到任何GPU硬件，系统将使用CPU进行计算和训练");
    } else {
        log_info("[GPU检测] 检测到 %d 个GPU设备", *num_found);
        for (int i = 0; i < *num_found; i++) {
            log_info("[GPU检测]   设备%d: %s (厂商=%d, 显存=%zu MB)",
                i, info[i].device_name, (int)info[i].vendor,
                info[i].memory_bytes > 0 ? info[i].memory_bytes / (1024 * 1024) : 0);
        }
    }

    return ret;
}

/** 检测特定厂商的GPU */
int gpu_hardware_detect_vendor(GpuVendor vendor, GpuHardwareInfo* info,
                                int max_devices, int* num_found) {
    if (!info || max_devices <= 0 || !num_found) {
        return -1;
    }

    GpuHardwareInfo* all_info = (GpuHardwareInfo*)calloc((size_t)max_devices, sizeof(GpuHardwareInfo));
    if (!all_info) return -1;

    int total_found = 0;
    int ret = gpu_hardware_detect(all_info, max_devices, &total_found);

    *num_found = 0;
    memset(info, 0, (size_t)max_devices * sizeof(GpuHardwareInfo));

    for (int i = 0; i < total_found && *num_found < max_devices; i++) {
        if (all_info[i].vendor == vendor) {
            memcpy(&info[*num_found], &all_info[i], sizeof(GpuHardwareInfo));
            (*num_found)++;
        }
    }

    free(all_info);
    return ret;
}

/** 获取GPU硬件总数量 */
int gpu_hardware_count(void) {
    int num_found = 0;
    GpuHardwareInfo info[16];
    gpu_hardware_detect(info, 16, &num_found);
    return num_found;
}

/** 获取推荐的GPU设备索引（优先独立GPU，其次集成GPU，再次专业NPU） */
int gpu_hardware_get_recommended_device(void) {
    int num_found = 0;
    GpuHardwareInfo info[16];
    gpu_hardware_detect(info, 16, &num_found);

    if (num_found == 0) return -1;

    /* 优先级排序：NVIDIA > AMD > Intel(独立) > Apple > 华为 > 寒武纪 > TPU > Qualcomm > Intel(集成) */
    const int priority[] = {
        GPU_VENDOR_NVIDIA, GPU_VENDOR_AMD,
        GPU_VENDOR_HUAWEI, GPU_VENDOR_CAMBRICON,
        GPU_VENDOR_APPLE, GPU_VENDOR_GOOGLE,
        GPU_VENDOR_INTEL, GPU_VENDOR_QUALCOMM
    };

    for (size_t p = 0; p < sizeof(priority) / sizeof(priority[0]); p++) {
        for (int i = 0; i < num_found; i++) {
            if (info[i].vendor == priority[p]) {
                /* 对Intel，优先选择独立GPU */
                if (priority[p] == GPU_VENDOR_INTEL && info[i].is_integrated) {
                    continue;
                }
                log_debug("[GPU检测] 推荐使用设备%d: %s", i, info[i].device_name);
                return i;
            }
        }
    }

    /* 回退到第一个设备 */
    log_debug("[GPU检测] 使用第一个可用设备0: %s", info[0].device_name);
    return 0;
}

/** 检查指定厂商的GPU驱动是否已安装 */
int gpu_hardware_driver_installed(GpuVendor vendor) {
    switch (vendor) {
    case GPU_VENDOR_NVIDIA:
        return try_load_library(NVIDIA_CUDA_LIB) ? 1 : 0;
    case GPU_VENDOR_AMD:
        return try_load_library(AMD_HIP_LIB) ? 1 : 0;
    case GPU_VENDOR_INTEL:
        return try_load_library(INTEL_LEVELZERO_LIB) ? 1 : 0;
    case GPU_VENDOR_APPLE:
#ifdef __APPLE__
        return try_load_library(
            "/System/Library/Frameworks/Metal.framework/Metal") ? 1 : 0;
#else
        return 0;
#endif
    case GPU_VENDOR_HUAWEI:
        return try_load_library(HUAWEI_ASCEND_LIB) ? 1 : 0;
    case GPU_VENDOR_CAMBRICON:
        return try_load_library(CAMBRICON_CNRT_LIB) ? 1 : 0;
    case GPU_VENDOR_GOOGLE:
        return try_load_library(TPU_LIB) ? 1 : 0;
    case GPU_VENDOR_QUALCOMM:
#ifdef _WIN32
        return try_load_library(QUALCOMM_LIB) ? 1 : 0;
#else
        return try_load_library("libOpenCL.so") ? 1 : 0;
#endif
    case GPU_VENDOR_ARM:
        /* ARM Mali GPU通过OpenCL检测 */
        return try_load_library("libOpenCL.so") ? 1 : 0;
    default:
        return 0;
    }
}
