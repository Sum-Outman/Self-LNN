/**
 * @file gpu_npu.h
 * @brief NPU 驱动路径自动扫描与设备发现
 *
 * 提供统一的 NPU 驱动路径扫描、设备枚举和版本检测功能。
 * 支持华为昇腾 (Ascend)、寒武纪 (Cambricon)、谷歌 TPU 三大 NPU 平台。
 * 自动探测常见安装路径和环境变量，无需用户手动配置。
 */

#ifndef SELFLNN_GPU_NPU_H
#define SELFLNN_GPU_NPU_H

#include "selflnn/core/common.h"
#include "selflnn/gpu/gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/** NPU 供应商类型 */
typedef enum {
    NPU_VENDOR_ASCEND = 0,      /**< 华为昇腾 (Ascend) */
    NPU_VENDOR_CAMBRICON = 1,   /**< 寒武纪 (Cambricon) */
    NPU_VENDOR_TPU = 2,         /**< 谷歌 TPU */
    NPU_VENDOR_UNKNOWN = 3
} NpuVendor;

/** 最大扫描路径数 */
#define NPU_MAX_SCAN_PATHS 16

/** 最大路径长度 */
#define NPU_PATH_MAX 512

/** 最大设备名长度 */
#define NPU_DEVICE_NAME_MAX 128

/** 最大设备数 */
#define NPU_MAX_DEVICES 256

/**
 * @brief NPU 驱动路径扫描结果
 *
 * 记录某个 NPU 供应商的驱动扫描详情。
 */
typedef struct {
    NpuVendor vendor;                           /**< NPU 供应商 */
    char library_path[NPU_PATH_MAX];            /**< 找到的驱动库路径（未找到则为空） */
    char install_path[NPU_PATH_MAX];            /**< 安装根目录（未找到则为空） */
    char version[64];                           /**< 驱动版本字符串 */
    int found;                                  /**< 是否找到驱动 */
    int device_count;                           /**< 检测到的设备数 */
    char device_names[NPU_MAX_DEVICES][NPU_DEVICE_NAME_MAX]; /**< 设备名列表 */
    int paths_checked;                          /**< 已检查的路径数 */
    char checked_paths[NPU_MAX_SCAN_PATHS][NPU_PATH_MAX]; /**< 已检查的路径列表 */
    int paths_found;                            /**< 找到的有效路径数 */
    char found_paths[NPU_MAX_SCAN_PATHS][NPU_PATH_MAX];   /**< 找到的有效路径列表 */
} NpuDriverScanResult;

/**
 * @brief NPU 驱动扫描报告
 *
 * 包含所有 NPU 供应商的扫描结果。
 */
typedef struct {
    NpuDriverScanResult ascend;     /**< 昇腾扫描结果 */
    NpuDriverScanResult cambricon;  /**< 寒武纪扫描结果 */
    NpuDriverScanResult tpu;        /**< TPU 扫描结果 */
    int any_found;                  /**< 是否有任一 NPU 驱动找到 */
    int total_devices;              /**< 所有 NPU 设备总数 */
} NpuDriverScanReport;

/**
 * @brief 自动扫描所有 NPU 驱动路径
 *
 * 遍历所有 NPU 供应商的常见安装路径和环境变量，
 * 检测驱动库、设备节点和版本信息。
 * 此函数只进行扫描探测，不会加载或初始化任何驱动。
 *
 * @param report [out] 扫描结果报告
 * @return int 找到的 NPU 供应商数量，-1 表示参数错误
 */
SELFLNN_API int npu_scan_driver_paths(NpuDriverScanReport* report);

/**
 * @brief 扫描指定 NPU 供应商的驱动路径
 *
 * @param vendor NPU 供应商
 * @param result [out] 扫描结果
 * @return int 1=找到驱动，0=未找到，-1=参数错误
 */
SELFLNN_API int npu_scan_single_driver(NpuVendor vendor, NpuDriverScanResult* result);

/**
 * @brief 获取 NPU 供应商名称
 *
 * @param vendor NPU 供应商
 * @return const char* 中文名称（如"华为昇腾"）
 */
SELFLNN_API const char* npu_vendor_name(NpuVendor vendor);

/**
 * @brief 获取 NPU 推荐安装提示
 *
 * 当某 NPU 驱动未找到时，提供安装建议。
 *
 * @param vendor NPU 供应商
 * @return const char* 安装提示字符串
 */
SELFLNN_API const char* npu_install_guide(NpuVendor vendor);

/**
 * @brief 将 NpuVendor 转换为 GpuBackend
 *
 * @param vendor NPU 供应商
 * @return GpuBackend 对应的 GPU 后端类型
 */
SELFLNN_API GpuBackend npu_vendor_to_backend(NpuVendor vendor);

/**
 * @brief 将 GpuBackend 转换为 NpuVendor
 *
 * @param backend GPU 后端类型
 * @return NpuVendor 对应的 NPU 供应商，非 NPU 后端返回 NPU_VENDOR_UNKNOWN
 */
SELFLNN_API NpuVendor npu_backend_to_vendor(GpuBackend backend);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GPU_NPU_H */
