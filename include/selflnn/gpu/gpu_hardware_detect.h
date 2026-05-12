#ifndef SELFLNN_GPU_HARDWARE_DETECT_H
#define SELFLNN_GPU_HARDWARE_DETECT_H

#include "selflnn/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GPU硬件检测接口 */

/* 硬件厂商枚举 */
typedef enum {
    GPU_VENDOR_UNKNOWN    = 0,
    GPU_VENDOR_NVIDIA     = 1,
    GPU_VENDOR_AMD        = 2,
    GPU_VENDOR_INTEL      = 3,
    GPU_VENDOR_APPLE      = 4,
    GPU_VENDOR_ARM        = 5,
    GPU_VENDOR_HUAWEI     = 6,
    GPU_VENDOR_CAMBRICON  = 7,
    GPU_VENDOR_GOOGLE     = 8,
    GPU_VENDOR_QUALCOMM   = 9
} GpuVendor;

/* GPU设备信息 */
typedef struct {
    char device_name[256];
    GpuVendor vendor;
    size_t memory_bytes;
    int compute_units;
    int max_frequency_mhz;
    int supports_fp16;
    int supports_fp64;
    int driver_version_major;
    int driver_version_minor;
    char driver_version_str[64];
    char pci_bus_id[32];
    int is_integrated;
    int is_discrete;
} GpuHardwareInfo;

/* 检测GPU硬件是否可用 */
int gpu_hardware_detect(GpuHardwareInfo* info, int max_devices, int* num_found);

/* 检测特定厂商的GPU */
int gpu_hardware_detect_vendor(GpuVendor vendor, GpuHardwareInfo* info, int max_devices, int* num_found);

/* 获取GPU硬件数量 */
int gpu_hardware_count(void);

/* 获取推荐的GPU设备索引 */
int gpu_hardware_get_recommended_device(void);

/* 检查GPU驱动是否已安装 */
int gpu_hardware_driver_installed(GpuVendor vendor);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GPU_HARDWARE_DETECT_H */
