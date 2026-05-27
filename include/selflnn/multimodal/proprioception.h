#ifndef SELFLNN_PROPRIOCEPTION_H
#define SELFLNN_PROPRIOCEPTION_H

/**
 * @file proprioception.h
 * @brief 本体感知预处理器 — 关节编码器解码、IMU姿态解算、力矩传感器融合
 *
 * 将原始本体感知传感器数据（关节角度、角速度、IMU加速度/角速度/磁力计、
 * 力矩/力传感器）解码为统一浮点特征向量，供多模态统一输入层使用。
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PROPRIOCEPTION_MAX_JOINTS    32
#define PROPRIOCEPTION_FEATURE_DIM   128

typedef struct {
    float joint_positions[PROPRIOCEPTION_MAX_JOINTS];
    float joint_velocities[PROPRIOCEPTION_MAX_JOINTS];
    float joint_torques[PROPRIOCEPTION_MAX_JOINTS];
    int num_joints;
} JointState;

typedef struct {
    float accel[3];
    float gyro[3];
    float mag[3];
    float orientation[4];       /* 四元数 (w,x,y,z) */
    float linear_accel[3];      /* 世界坐标系线性加速度 */
    int calibrated;
} IMUState;

typedef struct {
    float forces[6];            /* Fx,Fy,Fz,Mx,My,Mz */
    int sensor_id;
    float temperature;
} ForceTorqueState;

typedef struct ProprioceptionProcessor ProprioceptionProcessor;

ProprioceptionProcessor* proprioception_create(void);
void proprioception_free(ProprioceptionProcessor* pp);

int proprioception_decode_joints(ProprioceptionProcessor* pp,
    const float* raw_encoder_values, int num_encoders,
    JointState* out_joints);
int proprioception_decode_imu(ProprioceptionProcessor* pp,
    const float* raw_imu_data, int imu_channels, IMUState* out_imu);
int proprioception_fuse_force_torque(ProprioceptionProcessor* pp,
    const float* raw_ft_data, int ft_channels,
    ForceTorqueState* out_ft);
int proprioception_compute_feature_vector(ProprioceptionProcessor* pp,
    const JointState* joints, const IMUState* imu,
    const ForceTorqueState* ft, float* feature_vector, size_t feature_dim);
int proprioception_ekf_update(ProprioceptionProcessor* pp,
    const float* measurement, int meas_dim, float dt);
int proprioception_get_filtered_state(ProprioceptionProcessor* pp,
    float* state, int state_dim);

#ifdef __cplusplus
}
#endif

#endif
