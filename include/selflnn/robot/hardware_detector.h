#ifndef SELFLNN_HARDWARE_DETECTOR_H
#define SELFLNN_HARDWARE_DETECTOR_H

#include "selflnn/core/common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HD_MAX_DEVICES 256
#define HD_MAX_NAME_LEN 128
#define HD_MAX_PATH_LEN 512
#define HD_MAX_SERIAL_LEN 64
#define HD_MAX_CAMERAS 32
#define HD_MAX_AUDIO_DEVICES 16
#define HD_MAX_SERIAL_PORTS 32
#define HD_MAX_NETWORK_ADAPTERS 16
#define HD_MAX_GPU_DEVICES 32
#define HD_MAX_DEPTH_CAMERAS 16
#define HD_MAX_IMU_DEVICES 16
#define HD_MAX_LIDAR_DEVICES 16
#define HD_MAX_SENSOR_DEVICES 64
#define HD_MAX_BENCHMARK_SAMPLES 128
#define HD_MAX_HEALTH_HISTORY 256
#define HD_DETECT_HOTPLUG_BUFFER 32
#define HD_LNN_INPUT_DIM 64
#define HD_LNN_HIDDEN_DIM 128
#define HD_LNN_OUTPUT_DIM 16
#define HD_CLASSIFY_CONF_THRESHOLD 0.6f

typedef enum {
    HD_DEVICE_UNKNOWN = 0,
    HD_DEVICE_CPU,
    HD_DEVICE_GPU,
    HD_DEVICE_NPU,
    HD_DEVICE_CAMERA,
    HD_DEVICE_MICROPHONE,
    HD_DEVICE_SPEAKER,
    HD_DEVICE_SERIAL_PORT,
    HD_DEVICE_NETWORK_ADAPTER,
    HD_DEVICE_SENSOR,
    HD_DEVICE_ROBOT_ARM,
    HD_DEVICE_MOTOR_CONTROLLER,
    HD_DEVICE_LIDAR,
    HD_DEVICE_DEPTH_CAMERA,
    HD_DEVICE_IMU,
    HD_DEVICE_CUSTOM
} HDDeviceType;

typedef enum {
    HD_STATUS_UNKNOWN = 0,
    HD_STATUS_AVAILABLE,
    HD_STATUS_BUSY,
    HD_STATUS_ERROR,
    HD_STATUS_NOT_PRESENT
} HDDeviceStatus;

typedef enum {
    HD_BENCHMARK_NONE = 0,
    HD_BENCHMARK_COMPUTE,
    HD_BENCHMARK_MEMORY,
    HD_BENCHMARK_IO,
    HD_BENCHMARK_LATENCY,
    HD_BENCHMARK_THROUGHPUT
} HDBenchmarkType;

typedef enum {
    HD_HEALTH_UNKNOWN = 0,
    HD_HEALTH_GOOD,
    HD_HEALTH_DEGRADED,
    HD_HEALTH_CRITICAL,
    HD_HEALTH_FAILING
} HDHealthLevel;

typedef enum {
    HD_CAP_UNKNOWN = 0,
    HD_CAP_VISION_PROCESSING,
    HD_CAP_AUDIO_PROCESSING,
    HD_CAP_CONTROL_SIGNAL,
    HD_CAP_HIGH_PERFORMANCE,
    HD_CAP_LOW_LATENCY,
    HD_CAP_EMBEDDED
} HDDeviceCapability;

typedef struct {
    HDDeviceType type;
    char name[HD_MAX_NAME_LEN];
    char vendor[HD_MAX_NAME_LEN];
    char model[HD_MAX_NAME_LEN];
    char serial_number[HD_MAX_SERIAL_LEN];
    char driver_path[HD_MAX_PATH_LEN];
    char device_path[HD_MAX_PATH_LEN];
    HDDeviceStatus status;
    float performance_score;
    size_t total_memory_bytes;
    size_t free_memory_bytes;
    int compute_units;
    float clock_speed_mhz;
    int supports_double;
    int supports_half;
    int is_virtual;
    int is_primary;
} HDDeviceInfo;

typedef struct {
    HDDeviceInfo devices[HD_MAX_DEVICES];
    size_t num_devices;
    int detection_complete;
    double detection_time_ms;
} HDDetectionResult;

typedef struct {
    int enable_cpu_detection;
    int enable_gpu_detection;
    int enable_camera_detection;
    int enable_audio_detection;
    int enable_serial_detection;
    int enable_network_detection;
    int enable_sensor_detection;
    int enable_depth_camera_detection;
    int enable_imu_detection;
    int enable_lidar_detection;
    int enable_hotplug_monitor;
    int enable_health_monitor;
    int use_wmi_fallback;
    int detection_timeout_ms;
} HDDetectionConfig;

#define HD_DETECTION_CONFIG_DEFAULT { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 5000 }

typedef struct {
    HDBenchmarkType bench_type;
    double value;
    double std_dev;
    char description[HD_MAX_NAME_LEN];
} HDBenchmarkResult;

typedef struct {
    HDHealthLevel level;
    float health_score;
    float temperature_normalized;
    float load_percentage;
    size_t error_count;
    double uptime_hours;
    char prediction[HD_MAX_NAME_LEN];
} HDHealthInfo;

typedef struct {
    int device_index;
    HDDeviceType device_type;
    char device_name[HD_MAX_NAME_LEN];
    double timestamp_ms;
    int is_arrival;
} HDHotplugEvent;

typedef struct {
    HDHotplugEvent events[HD_DETECT_HOTPLUG_BUFFER];
    size_t num_events;
    int monitor_active;
} HDHotplugMonitor;

int hd_detect_all(HDDetectionConfig config, HDDetectionResult* result);

int hd_detect_cpu(HDDeviceInfo* info);
int hd_detect_gpu(HDDeviceInfo* infos, size_t max_count, size_t* count);
int hd_detect_cameras(HDDeviceInfo* infos, size_t max_count, size_t* count);
int hd_detect_audio_devices(HDDeviceInfo* infos, size_t max_count, size_t* count);
int hd_detect_serial_ports(HDDeviceInfo* infos, size_t max_count, size_t* count);
int hd_detect_network_adapters(HDDeviceInfo* infos, size_t max_count, size_t* count);
int hd_detect_sensors(HDDeviceInfo* infos, size_t max_count, size_t* count);
int hd_detect_depth_cameras(HDDeviceInfo* infos, size_t max_count, size_t* count);
int hd_detect_imu_devices(HDDeviceInfo* infos, size_t max_count, size_t* count);
int hd_detect_lidar_devices(HDDeviceInfo* infos, size_t max_count, size_t* count);

/**
 * @brief K-032: 枚举USB设备
 * @param infos 设备信息输出数组
 * @param max_count 最大设备数
 * @param count 实际的设备数量
 * @return 0成功，-1失败
 */
int hd_detect_usb_devices(HDDeviceInfo* infos, size_t max_count, size_t* count);

int hd_get_device_by_type(const HDDetectionResult* result, HDDeviceType type, HDDeviceInfo* out, size_t max_count, size_t* count);
int hd_get_primary_gpu(HDDeviceInfo* out);
int hd_get_system_info(char* system_name, size_t max_len, char* os_version, size_t os_max_len);

int hd_benchmark_device(const HDDeviceInfo* device, HDBenchmarkType bench_type, HDBenchmarkResult* out);
int hd_benchmark_all(HDDetectionResult* result, HDBenchmarkResult* bench_out, size_t max_count, size_t* count);

int hd_health_check(const HDDeviceInfo* device, HDHealthInfo* out);
int hd_health_predict_trend(HDHealthInfo* history, size_t history_len, HDHealthInfo* out);

int hd_classify_device_capability(const HDDeviceInfo* device, HDDeviceCapability* caps, size_t max_count, size_t* count);

int hd_hotplug_start_monitor(HDHotplugMonitor* monitor);
int hd_hotplug_poll_events(HDHotplugMonitor* monitor);
int hd_hotplug_stop_monitor(HDHotplugMonitor* monitor);

int hd_report_generate(const HDDetectionResult* result, char* report, size_t report_len);

void hd_result_free(HDDetectionResult* result);
const char* hd_device_type_str(HDDeviceType type);
const char* hd_device_status_str(HDDeviceStatus status);
const char* hd_health_level_str(HDHealthLevel level);
const char* hd_capability_str(HDDeviceCapability cap);

#ifdef __cplusplus
}
#endif

#endif
