/**
 * @file data_collection_pipeline.h
 * @brief 多模态真实数据采集流水线
 *
 * 统一管理所有硬件数据源的采集、验证和路由。
 * 核心理念：所有数据必须来自真实硬件，无硬件时返回明确错误而非虚假数据。
 * 支持部分硬件连接，连接哪些用哪些，绝不生成虚拟数据。
 *
 * 支持的数据源：
 * - 摄像头（单目/双目）：视觉帧数据
 * - 麦克风：音频波形数据
 * - 传感器：IMU/温度/压力/湿度等
 * - 深度相机：深度图数据
 * - 激光雷达：点云数据
 *
 * 100%纯C实现，无外部依赖。
 */

#ifndef SELFLNN_DATA_COLLECTION_PIPELINE_H
#define SELFLNN_DATA_COLLECTION_PIPELINE_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============ 数据源类型 ============ */

typedef enum {
    DC_SOURCE_CAMERA_RGB = 0,        /**< RGB摄像头 */
    DC_SOURCE_CAMERA_STEREO_L = 1,   /**< 双目左目 */
    DC_SOURCE_CAMERA_STEREO_R = 2,   /**< 双目右目 */
    DC_SOURCE_DEPTH_CAMERA = 3,      /**< 深度相机 */
    DC_SOURCE_MICROPHONE = 4,        /**< 麦克风 */
    DC_SOURCE_IMU = 5,               /**< IMU惯性传感器 */
    DC_SOURCE_TEMPERATURE = 6,       /**< 温度传感器 */
    DC_SOURCE_HUMIDITY = 7,          /**< 湿度传感器 */
    DC_SOURCE_PRESSURE = 8,          /**< 压力传感器 */
    DC_SOURCE_PROXIMITY = 9,         /**< 接近传感器 */
    DC_SOURCE_LIDAR = 10,            /**< 激光雷达 */
    DC_SOURCE_MOTOR_ENCODER = 11,    /**< 电机编码器 */
    DC_SOURCE_FORCE_TORQUE = 12,     /**< 力/力矩传感器 */
    DC_SOURCE_COUNT = 13
} DataCollectionSourceType;

/* ============ 数据源状态 ============ */

typedef enum {
    DC_STATUS_NOT_CONFIGURED = 0,    /**< 未配置 */
    DC_STATUS_HARDWARE_SEARCHING = 1,/**< 正在搜索硬件 */
    DC_STATUS_HARDWARE_NOT_FOUND = 2,/**< 硬件未找到 */
    DC_STATUS_HARDWARE_FOUND = 3,    /**< 硬件已找到 */
    DC_STATUS_CONNECTED = 4,         /**< 已连接 */
    DC_STATUS_STREAMING = 5,         /**< 正在采集数据流 */
    DC_STATUS_ERROR = 6,             /**< 错误状态 */
    DC_STATUS_DISCONNECTED = 7       /**< 已断开 */
} DataCollectionStatus;

/* ============ 采集帧数据 ============ */

typedef struct {
    uint8_t* rgb_data;               /**< RGB像素数据 (width*height*3) */
    int width;
    int height;
    int channels;                    /**< 通道数（通常3） */
    uint64_t timestamp_us;           /**< 时间戳（微秒） */
    int source_id;                   /**< 数据源ID */
    int frame_id;                    /**< 帧序号 */
    int is_valid;                    /**< 数据是否有效（来自真实硬件） */
} CollectedImageFrame;

typedef struct {
    float* samples;                  /**< 音频采样数据 */
    int num_samples;
    int sample_rate;
    int num_channels;
    uint64_t timestamp_us;
    int source_id;
    int is_valid;
} CollectedAudioFrame;

typedef struct {
    float* values;                   /**< 传感器值数组 */
    int num_dimensions;
    int sensor_type;                 /**< 传感器类型 */
    uint64_t timestamp_us;
    int source_id;
    int is_valid;
} CollectedSensorFrame;

typedef struct {
    float* points;                   /**< 点云数据 (num_points * 3) */
    int num_points;
    uint64_t timestamp_us;
    int source_id;
    int is_valid;
} CollectedPointCloud;

/* ============ 采集快照 ============ */

typedef struct {
    CollectedImageFrame* images;     /**< 图像帧数组 */
    int num_images;
    int max_images;

    CollectedAudioFrame* audio_frames;
    int num_audio_frames;
    int max_audio_frames;

    CollectedSensorFrame* sensor_frames;
    int num_sensor_frames;
    int max_sensor_frames;

    CollectedPointCloud* point_clouds;
    int num_point_clouds;
    int max_point_clouds;

    uint64_t snapshot_timestamp_us;
    int is_complete;
} CollectionSnapshot;

/* ============ 硬件健康状态 ============ */

typedef struct {
    DataCollectionSourceType source_type;
    DataCollectionStatus status;
    char device_name[256];
    char device_id[128];
    char status_message[512];
    int is_streaming;
    uint64_t frames_collected;
    uint64_t bytes_collected;
    uint64_t last_frame_timestamp_us;
    float avg_fps;
    float error_rate;
    time_t connected_at;
    time_t last_error_at;
    int error_count;
    int hardware_present;            /**< 1=真实硬件检测到，0=未检测到 */
    int data_is_real;                /**< 1=真实数据，0=无数据（绝不生成虚拟数据） */
} DataSourceHealth;

/* ============ 流水线统计 ============ */

typedef struct {
    int total_sources_configured;
    int total_sources_connected;
    int total_sources_streaming;
    int total_sources_error;
    int total_sources_hardware_not_found;
    uint64_t total_frames_collected;
    uint64_t total_bytes_collected;
    float overall_uptime_seconds;
    float collection_rate_hz;
    time_t pipeline_started_at;
    int has_any_real_data;           /**< 1=至少一个真实数据源连接 */
} PipelineStats;

/* ============ 采集流水线配置 ============ */

typedef struct {
    int enable_camera_rgb;           /**< 启用RGB摄像头采集 */
    int enable_camera_stereo;        /**< 启用双目采集 */
    int enable_depth_camera;         /**< 启用深度相机 */
    int enable_microphone;           /**< 启用麦克风 */
    int enable_imu;                  /**< 启用IMU */
    int enable_environment_sensors;  /**< 启用环境传感器（温度/湿度/压力） */
    int enable_lidar;                /**< 启用激光雷达 */
    int enable_motor_encoders;       /**< 启用电机编码器 */
    int enable_force_torque;         /**< 启用力/力矩传感器 */

    int camera_width;                /**< 摄像头采集宽度 */
    int camera_height;               /**< 摄像头采集高度 */
    int camera_fps;                  /**< 摄像头帧率 */
    int audio_sample_rate;           /**< 音频采样率 */
    int audio_channels;              /**< 音频通道数 */
    int sensor_sample_rate_hz;       /**< 传感器采样率 */

    int max_image_frames_per_snapshot;   /**< 每快照最大图像帧数 */
    int max_audio_frames_per_snapshot;   /**< 每快照最大音频帧数 */
    int max_sensor_frames_per_snapshot;  /**< 每快照最大传感器帧数 */
    int max_point_clouds_per_snapshot;   /**< 每快照最大点云数 */

    int strict_mode;                 /**< 严格模式：任何数据源不可用时立即报错 */
    int self_check_interval_sec;     /**< 自检间隔（秒） */
} PipelineConfig;

/* ============ 采集流水线句柄 ============ */

typedef struct DataCollectionPipeline DataCollectionPipeline;

/* ============ API ============ */

/**
 * @brief 创建数据采集流水线
 * @param config 流水线配置
 * @return 流水线句柄，失败返回NULL
 */
DataCollectionPipeline* dcpipeline_create(const PipelineConfig* config);

/**
 * @brief 释放数据采集流水线
 * @param pipeline 流水线句柄
 */
void dcpipeline_free(DataCollectionPipeline* pipeline);

/**
 * @brief 启动数据采集（异步）
 * @param pipeline 流水线句柄
 * @return 0=成功，-1=失败
 */
int dcpipeline_start(DataCollectionPipeline* pipeline);

/**
 * @brief 停止数据采集
 * @param pipeline 流水线句柄
 * @return 0=成功，-1=失败
 */
int dcpipeline_stop(DataCollectionPipeline* pipeline);

/**
 * @brief 采集一帧快照数据（同步，阻塞直到所有已连接源就绪或超时）
 * @param pipeline 流水线句柄
 * @param snapshot 输出快照（调用者负责通过dcpipeline_free_snapshot释放）
 * @param timeout_ms 超时毫秒数
 * @return 0=成功，-1=超时，-2=无可用数据源
 */
int dcpipeline_collect_snapshot(DataCollectionPipeline* pipeline,
                                CollectionSnapshot* snapshot,
                                int timeout_ms);

/**
 * @brief 释放快照内存
 * @param snapshot 快照指针
 */
void dcpipeline_free_snapshot(CollectionSnapshot* snapshot);

/**
 * @brief 获取单个数据源健康状态
 * @param pipeline 流水线句柄
 * @param source_type 数据源类型
 * @param health 输出健康状态
 * @return 0=成功，-1=该数据源未配置
 */
int dcpipeline_get_source_health(DataCollectionPipeline* pipeline,
                                 DataCollectionSourceType source_type,
                                 DataSourceHealth* health);

/**
 * @brief 获取流水线统计
 * @param pipeline 流水线句柄
 * @param stats 输出统计信息
 * @return 0=成功
 */
int dcpipeline_get_stats(DataCollectionPipeline* pipeline, PipelineStats* stats);

/**
 * @brief 执行硬件自检（D-5：验证所有已配置数据源的真实性和可用性）
 *
 * 遍历所有已配置的数据源，逐一检查：
 * 1. 硬件是否存在
 * 2. 连接是否有效
 * 3. 数据流是否正常
 * 4. 数据是否为真实硬件数据（非模拟/虚拟）
 *
 * @param pipeline 流水线句柄
 * @param results 输出各源健康状态数组（至少DC_SOURCE_COUNT个元素）
 * @return 已检查的数据源数量
 */
int dcpipeline_self_check(DataCollectionPipeline* pipeline,
                          DataSourceHealth* results);

/**
 * @brief 获取默认流水线配置
 * @return 默认配置
 */
PipelineConfig dcpipeline_get_default_config(void);

/**
 * @brief 查询所有可用硬件设备（不启动采集）
 * @param pipeline 流水线句柄
 * @return 检测到的可用设备数
 */
int dcpipeline_detect_hardware(DataCollectionPipeline* pipeline);

/**
 * @brief 检查是否有任何真实硬件数据可用
 * @param pipeline 流水线句柄
 * @return 1=有真实数据，0=无硬件连接
 */
int dcpipeline_has_real_data(DataCollectionPipeline* pipeline);

#ifdef __cplusplus
}
#endif

#endif
