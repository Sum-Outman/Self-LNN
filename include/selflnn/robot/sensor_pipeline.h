#ifndef SELFLNN_SENSOR_PIPELINE_H
#define SELFLNN_SENSOR_PIPELINE_H

#include "selflnn/robot/robot.h"
#include "selflnn/core/port_config.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SENSOR_PIPELINE_MAX_SENSORS          64
#define SENSOR_PIPELINE_MAX_SUBSCRIBERS      32
#define SENSOR_PIPELINE_MAX_CLIENTS          16
#define SENSOR_PIPELINE_RING_BUFFER_SIZE     4096
#define SENSOR_PIPELINE_MAX_DATA_SIZE        (8 * 1024 * 1024)
#define SENSOR_PIPELINE_MAX_CAMERA_DATA      (1920 * 1080 * 3)
#define SENSOR_PIPELINE_MAX_LIDAR_POINTS     4096
#define SENSOR_PIPELINE_NAME_MAX             64
#define SENSOR_PIPELINE_TOPIC_MAX            128
#define SENSOR_PIPELINE_FILTER_MAX           16
#define SENSOR_PIPELINE_NUM_PRIORITIES       5

typedef enum {
    SENSOR_PIPELINE_PRIORITY_CRITICAL = 0,
    SENSOR_PIPELINE_PRIORITY_HIGH = 1,
    SENSOR_PIPELINE_PRIORITY_MEDIUM = 2,
    SENSOR_PIPELINE_PRIORITY_LOW = 3,
    SENSOR_PIPELINE_PRIORITY_BACKGROUND = 4
} SensorPipelinePriority;

typedef enum {
    SENSOR_SOURCE_SIMULATOR = 0,
    SENSOR_SOURCE_HARDWARE = 1,
    SENSOR_SOURCE_FILE = 2,
    SENSOR_SOURCE_NETWORK = 3,
    SENSOR_SOURCE_SYNTHETIC = 4
} SensorSource;

typedef enum {
    SENSOR_PIPELINE_OK = 0,
    SENSOR_PIPELINE_ERROR_GENERAL = -1,
    SENSOR_PIPELINE_ERROR_NO_MEMORY = -2,
    SENSOR_PIPELINE_ERROR_NOT_FOUND = -3,
    SENSOR_PIPELINE_ERROR_FULL = -4,
    SENSOR_PIPELINE_ERROR_INVALID = -5,
    SENSOR_PIPELINE_ERROR_TIMEOUT = -6,
    SENSOR_PIPELINE_ERROR_DISABLED = -7,
    SENSOR_PIPELINE_ERROR_BUFFER_EMPTY = -8,
    SENSOR_PIPELINE_ERROR_BUFFER_FULL = -9,
    SENSOR_PIPELINE_ERROR_COMPRESSION = -10,
    SENSOR_PIPELINE_ERROR_STREAM = -11
} SensorPipelineError;

typedef struct {
    int sensor_id;
    SensorType sensor_type;
    SensorSource source;
    SensorPipelinePriority priority;
    double timestamp;
    double receive_timestamp;
    uint32_t sequence;
    size_t data_size;
    uint8_t* data;
    int is_compressed;
    size_t original_size;
    int is_valid;
    float confidence;
    char sensor_name[SENSOR_PIPELINE_NAME_MAX];
    char topic_name[SENSOR_PIPELINE_TOPIC_MAX];
    float range_min;
    float range_max;
    float resolution;
    float accuracy;
    double sample_rate_hz;
    int frame_width;
    int frame_height;
    int channels;
} SensorPipelineEntry;

typedef struct {
    int max_entries;
    int max_data_size;
    int enable_compression;
    int compression_level;
    int enable_dedup;
    double dedup_threshold;
} SensorRingBufferConfig;

typedef struct {
    int sensor_id;
    SensorType sensor_type;
    uint32_t min_sequence;
    uint32_t max_sequence;
    double min_timestamp;
    double max_timestamp;
    int min_confidence;
    int max_sensors;
    int sensor_ids[SENSOR_PIPELINE_MAX_SENSORS];
} SensorPipelineFilter;

typedef void (*SensorPipelineCallback)(const SensorPipelineEntry* entry, void* user_data);

typedef struct {
    int socket;
    char client_host[64];
    int client_port;
    double connect_time;
    double last_active;
    int subscribed_sensor_ids[SENSOR_PIPELINE_MAX_SENSORS];
    int num_subscriptions;
    int is_active;
    uint8_t recv_buffer[1024];
    int recv_buffer_pos;
} SensorStreamingClient;

typedef struct {
    int max_sensors;
    int max_subscribers;
    int max_streaming_clients;
    int streaming_port;
    int enable_streaming_server;
    int enable_timestamp_sync;
    double timestamp_sync_tolerance;
    int enable_priority_scheduling;
    int thread_pool_size;
    double max_buffer_duration;
    SensorRingBufferConfig ring_buffer_config;
} SensorPipelineConfig;

typedef struct SensorPipeline SensorPipeline;

SensorPipeline* sensor_pipeline_create(const SensorPipelineConfig* config);
void sensor_pipeline_destroy(SensorPipeline* pipeline);

int sensor_pipeline_start(SensorPipeline* pipeline);
int sensor_pipeline_stop(SensorPipeline* pipeline);
int sensor_pipeline_reset(SensorPipeline* pipeline);
int sensor_pipeline_is_running(SensorPipeline* pipeline);

int sensor_pipeline_register_sensor(SensorPipeline* pipeline, int sensor_id,
                                     SensorType sensor_type, SensorSource source,
                                     SensorPipelinePriority priority,
                                     const char* sensor_name, double sample_rate_hz);
int sensor_pipeline_unregister_sensor(SensorPipeline* pipeline, int sensor_id);
int sensor_pipeline_get_sensor_info(SensorPipeline* pipeline, int sensor_id,
                                     SensorPipelineEntry* info);
int sensor_pipeline_get_registered_sensors(SensorPipeline* pipeline, int* sensor_ids,
                                            int* count);
int sensor_pipeline_get_sensor_count(SensorPipeline* pipeline);

int sensor_pipeline_push_data(SensorPipeline* pipeline, const SensorPipelineEntry* entry);
int sensor_pipeline_push_raw(SensorPipeline* pipeline, int sensor_id, SensorType sensor_type,
                              const uint8_t* data, size_t data_size, double timestamp,
                              float confidence);

int sensor_pipeline_push_image(SensorPipeline* pipeline, int sensor_id,
                                const uint8_t* image_data, size_t data_size,
                                int width, int height, int channels,
                                double timestamp, float confidence);

int sensor_pipeline_push_lidar(SensorPipeline* pipeline, int sensor_id,
                                const float* ranges, const float* intensities,
                                int num_points, double timestamp, float confidence);

int sensor_pipeline_push_imu(SensorPipeline* pipeline, int sensor_id,
                              const float* acceleration, const float* angular_velocity,
                              const float* orientation, double timestamp, float confidence);

int sensor_pipeline_push_gnss(SensorPipeline* pipeline, int sensor_id,
                               double latitude, double longitude, double altitude,
                               double timestamp, float confidence);

int sensor_pipeline_push_force_torque(SensorPipeline* pipeline, int sensor_id,
                                       const float* force, const float* torque,
                                       double timestamp, float confidence);

int sensor_pipeline_get_latest(SensorPipeline* pipeline, int sensor_id,
                                SensorPipelineEntry* entry);
int sensor_pipeline_get_latest_by_type(SensorPipeline* pipeline, SensorType sensor_type,
                                        SensorPipelineEntry* entry);
int sensor_pipeline_get_history(SensorPipeline* pipeline, int sensor_id,
                                 SensorPipelineEntry* entries, int* count, int max_entries);
int sensor_pipeline_get_history_by_type(SensorPipeline* pipeline, SensorType sensor_type,
                                         SensorPipelineEntry* entries, int* count,
                                         int max_entries);

int sensor_pipeline_get_filtered(SensorPipeline* pipeline, const SensorPipelineFilter* filter,
                                  SensorPipelineEntry* entries, int* count, int max_entries);

int sensor_pipeline_subscribe(SensorPipeline* pipeline, int sensor_id,
                               SensorPipelineCallback callback, void* user_data);
int sensor_pipeline_subscribe_by_type(SensorPipeline* pipeline, SensorType sensor_type,
                                       SensorPipelineCallback callback, void* user_data);
int sensor_pipeline_unsubscribe(SensorPipeline* pipeline, int sensor_id,
                                 SensorPipelineCallback callback);
int sensor_pipeline_unsubscribe_all(SensorPipeline* pipeline);

int sensor_pipeline_set_priority(SensorPipeline* pipeline, int sensor_id,
                                  SensorPipelinePriority priority);
SensorPipelinePriority sensor_pipeline_get_priority(SensorPipeline* pipeline, int sensor_id);

int sensor_pipeline_set_sample_rate(SensorPipeline* pipeline, int sensor_id,
                                     double sample_rate_hz);
double sensor_pipeline_get_sample_rate(SensorPipeline* pipeline, int sensor_id);

int sensor_pipeline_enable_sensor(SensorPipeline* pipeline, int sensor_id);
int sensor_pipeline_disable_sensor(SensorPipeline* pipeline, int sensor_id);
int sensor_pipeline_is_sensor_enabled(SensorPipeline* pipeline, int sensor_id);

int sensor_pipeline_get_statistics(SensorPipeline* pipeline, int sensor_id,
                                    uint32_t* total_pushed, uint32_t* total_dropped,
                                    double* avg_latency_ms, double* max_latency_ms,
                                    double* last_rate_hz);
int sensor_pipeline_get_pipeline_load(SensorPipeline* pipeline, double* buffer_usage,
                                       double* avg_push_time_us, double* avg_poll_time_us);

SensorPipelineConfig* sensor_pipeline_config_create(void);
void sensor_pipeline_config_destroy(SensorPipelineConfig* config);
void sensor_pipeline_config_set_defaults(SensorPipelineConfig* config);

int sensor_pipeline_start_streaming_server(SensorPipeline* pipeline);
int sensor_pipeline_stop_streaming_server(SensorPipeline* pipeline);
int sensor_pipeline_get_streaming_clients(SensorPipeline* pipeline,
                                           SensorStreamingClient* clients, int* count);
int sensor_pipeline_disconnect_client(SensorPipeline* pipeline, int client_index);

int sensor_pipeline_export_to_file(SensorPipeline* pipeline, int sensor_id,
                                    const char* filename, int max_entries);
int sensor_pipeline_import_from_file(SensorPipeline* pipeline, const char* filename);

int sensor_pipeline_clear_buffer(SensorPipeline* pipeline);
int sensor_pipeline_clear_sensor_buffer(SensorPipeline* pipeline, int sensor_id);

const char* sensor_pipeline_get_last_error(SensorPipeline* pipeline);
const char* sensor_pipeline_priority_string(SensorPipelinePriority priority);
const char* sensor_pipeline_source_string(SensorSource source);
const char* sensor_pipeline_error_string(int error_code);

typedef struct {
    int entry_count;
    int dropped_count;
    double avg_latency;
    double max_latency;
    double current_rate_hz;
    int64_t total_bytes;
    int buffer_entries;
    int buffer_capacity;
} SensorPipelineStats;

int sensor_pipeline_get_global_stats(SensorPipeline* pipeline, SensorPipelineStats* stats);

#ifdef __cplusplus
}
#endif

#endif
