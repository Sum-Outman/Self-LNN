#ifndef SELFLNN_SENSOR_SIMULATION_H
#define SELFLNN_SENSOR_SIMULATION_H

#include "selflnn/robot/robot.h"
#include "selflnn/robot/simulator.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SENSOR_SIM_MAX_LIDAR_POINTS 4096
#define SENSOR_SIM_MAX_DEPTH_PIXELS 307200
#define SENSOR_SIM_MAX_LIDAR_LAYERS 64
#define SENSOR_SIM_MAX_DEPTH_WIDTH 1280
#define SENSOR_SIM_MAX_DEPTH_HEIGHT 720

typedef struct {
    float min_range;
    float max_range;
    int num_rays;
    int num_layers;
    float fov_horizontal;
    float fov_vertical;
    float angle_min;
    float angle_max;
    float scan_rate;
    float noise_stddev;
} SensorSimLidarConfig;

typedef struct {
    int width;
    int height;
    float fov;
    float min_depth;
    float max_depth;
    float noise_stddev;
} SensorSimDepthCameraConfig;

typedef struct {
    float accel_noise_stddev;
    float gyro_noise_stddev;
    float accel_bias_instability;
    float gyro_bias_instability;
    float update_rate;
} SensorSimImuConfig;

typedef struct {
    float range;
    float intensity;
    float point[3];
    int valid;
} SensorSimLidarPoint;

typedef struct {
    float depth;
    float point[3];
    int valid;
} SensorSimDepthPixel;

typedef struct {
    float timestamp;
    float acceleration[3];
    float angular_velocity[3];
    float orientation[4];
    int valid;
} SensorSimImuFrame;

typedef struct {
    SensorSimLidarConfig lidar_config;
    SensorSimDepthCameraConfig depth_config;
    SensorSimImuConfig imu_config;
    float imu_bias_accel[3];
    float imu_bias_gyro[3];
    float last_update_time;
    unsigned int rng_state;
    int initialized;
} SensorSimContext;

void sensor_sim_init(SensorSimContext* ctx);

void sensor_sim_set_lidar_config(SensorSimContext* ctx, const SensorSimLidarConfig* config);

void sensor_sim_set_depth_config(SensorSimContext* ctx, const SensorSimDepthCameraConfig* config);

void sensor_sim_set_imu_config(SensorSimContext* ctx, const SensorSimImuConfig* config);

int sensor_sim_lidar_scan(SensorSimContext* ctx,
                          const float* mount_pos, const float* mount_quat,
                          const SimPhysicsPipeline* pipeline,
                          SensorSimLidarPoint* points, int max_points);

int sensor_sim_depth_camera(SensorSimContext* ctx,
                            const float* mount_pos, const float* mount_quat,
                            const SimPhysicsPipeline* pipeline,
                            SensorSimDepthPixel* pixels, int max_pixels);

int sensor_sim_imu_update(SensorSimContext* ctx,
                          const float* linear_accel, const float* angular_vel,
                          const float* orientation, float dt,
                          SensorSimImuFrame* frame);

int sensor_sim_ray_intersect(const float* ray_origin, const float* ray_dir,
                             const SimCollisionObject* object,
                             float* hit_distance, float* hit_point, float* hit_normal);

#ifdef __cplusplus
}
#endif

#endif
