#include "selflnn/robot/sensor_simulation.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/robot/kinematics.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* quat_multiply/quat_conjugate/quat_rotate: math_utils.h inline versions */

void sensor_sim_init(SensorSimContext* ctx) {
    (void)ctx;
    return;
}

void sensor_sim_set_lidar_config(SensorSimContext* ctx, const SensorSimLidarConfig* config) {
    if (!ctx || !config) return;
    ctx->lidar_config = *config;
}

void sensor_sim_set_depth_config(SensorSimContext* ctx, const SensorSimDepthCameraConfig* config) {
    if (!ctx || !config) return;
    ctx->depth_config = *config;
}

void sensor_sim_set_imu_config(SensorSimContext* ctx, const SensorSimImuConfig* config) {
    if (!ctx || !config) return;
    ctx->imu_config = *config;
}

int sensor_sim_lidar_scan(SensorSimContext* ctx,
                          const float* mount_pos, const float* mount_quat,
                          const SimPhysicsPipeline* pipeline,
                          SensorSimLidarPoint* points, int max_points) {
    (void)ctx; (void)mount_pos; (void)mount_quat; (void)pipeline; (void)points; (void)max_points;
    return -1;
}

int sensor_sim_depth_camera(SensorSimContext* ctx,
                            const float* mount_pos, const float* mount_quat,
                            const SimPhysicsPipeline* pipeline,
                            SensorSimDepthPixel* pixels, int max_pixels) {
    (void)ctx; (void)mount_pos; (void)mount_quat; (void)pipeline; (void)pixels; (void)max_pixels;
    return -1;
}

int sensor_sim_imu_update(SensorSimContext* ctx,
                          const float* linear_accel, const float* angular_vel,
                          const float* orientation, float dt,
                          SensorSimImuFrame* frame) {
    (void)ctx; (void)linear_accel; (void)angular_vel; (void)orientation; (void)dt; (void)frame;
    return -1;
}

int sensor_sim_ray_intersect(const float* ray_origin, const float* ray_dir,
                             const SimCollisionObject* object,
                             float* hit_distance, float* hit_point, float* hit_normal) {
    (void)ray_origin; (void)ray_dir; (void)object; (void)hit_distance; (void)hit_point; (void)hit_normal;
    return -1;
}
