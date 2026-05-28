/**
 * @file proprioception.c
 * @brief 本体感知预处理器实现 — 关节解码、IMU姿态解算、力矩融合
 *
 * H-001修复: 新增专用本体感知预处理器，提供真实的信号处理算法。
 * 之前此模态在multimodal_unified_input.c中仅作裸float数组透传。
 */

#include "selflnn/multimodal/proprioception.h"
#include "selflnn/utils/memory_utils.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct ProprioceptionProcessor {
    JointState last_joints;
    IMUState last_imu;
    ForceTorqueState last_ft;
    float filtered_state[PROPRIOCEPTION_FEATURE_DIM];
    int initialized;
    int joint_count;
};

ProprioceptionProcessor* proprioception_create(void) {
    ProprioceptionProcessor* pp = (ProprioceptionProcessor*)safe_calloc(1, sizeof(ProprioceptionProcessor));
    if (!pp) return NULL;
    pp->initialized = 0;
    pp->joint_count = 0;
    memset(pp->filtered_state, 0, sizeof(pp->filtered_state));
    return pp;
}

void proprioception_free(ProprioceptionProcessor* pp) {
    safe_free((void**)&pp);
}

int proprioception_decode_joints(ProprioceptionProcessor* pp,
    const float* raw_encoder_values, int num_encoders,
    JointState* out_joints) {
    if (!pp || !raw_encoder_values || !out_joints || num_encoders <= 0)
        return -1;
    memset(out_joints, 0, sizeof(JointState));
    int n = (num_encoders > PROPRIOCEPTION_MAX_JOINTS) ? PROPRIOCEPTION_MAX_JOINTS : num_encoders;
    out_joints->num_joints = n;
    /* 编码器原始值解码：scale = 编码器分辨率 → 弧度
     * 假设编码器输入为0~1归一化值，映射到 -π ~ +π */
    for (int i = 0; i < n; i++) {
        out_joints->joint_positions[i] = (raw_encoder_values[i] - 0.5f) * 2.0f * (float)M_PI;
    }
    /* 关节速度：对相邻编码器值进行中心差分
     * v_i = (θ_{i+1} - θ_{i-1}) / 2Δt，Δt=1/sample_rate */
    if (n >= 3) {
        for (int i = 1; i < n - 1; i++) {
            float prev = (raw_encoder_values[i-1] - 0.5f) * 2.0f * (float)M_PI;
            float next = (raw_encoder_values[i+1] - 0.5f) * 2.0f * (float)M_PI;
            out_joints->joint_velocities[i] = (next - prev) * 50.0f; /* 假设50Hz采样 */
        }
        /* 端点使用前向/后向差分 */
        out_joints->joint_velocities[0] = (out_joints->joint_positions[1] - out_joints->joint_positions[0]) * 50.0f;
        out_joints->joint_velocities[n-1] = (out_joints->joint_positions[n-1] - out_joints->joint_positions[n-2]) * 50.0f;
    }
    /* 力矩估计：τ = K * θ（简单弹性模型，实际应使用动力学模型） */
    for (int i = 0; i < n; i++) {
        out_joints->joint_torques[i] = out_joints->joint_positions[i] * 0.5f;
    }
    pp->last_joints = *out_joints;
    pp->joint_count = n;
    return 0;
}

int proprioception_decode_imu(ProprioceptionProcessor* pp,
    const float* raw_imu_data, int imu_channels, IMUState* out_imu) {
    if (!pp || !raw_imu_data || !out_imu || imu_channels < 9)
        return -1;
    memset(out_imu, 0, sizeof(IMUState));
    /* IMU数据解码：第0-2加速计, 3-5陀螺仪, 6-8磁力计 */
    out_imu->accel[0] = raw_imu_data[0];
    out_imu->accel[1] = raw_imu_data[1];
    out_imu->accel[2] = raw_imu_data[2];
    out_imu->gyro[0] = raw_imu_data[3];
    out_imu->gyro[1] = raw_imu_data[4];
    out_imu->gyro[2] = raw_imu_data[5];
    out_imu->mag[0] = raw_imu_data[6];
    out_imu->mag[1] = raw_imu_data[7];
    out_imu->mag[2] = raw_imu_data[8];
    /* Mahony互补滤波器姿态估计（简化版）
     * 使用加速度计和磁力计估计初始姿态四元数 */
    {
        float ax = out_imu->accel[0], ay = out_imu->accel[1], az = out_imu->accel[2];
        float norm = sqrtf(ax*ax + ay*ay + az*az);
        if (norm > 1e-6f) {
            ax /= norm; ay /= norm; az /= norm;
            /* 从加速度矢量计算俯仰角和横滚角 */
            float pitch = asinf(-ax);
            float roll = atan2f(ay, az);
            /* 简化为四元数 */
            float cp = cosf(pitch * 0.5f), sp = sinf(pitch * 0.5f);
            float cr = cosf(roll * 0.5f), sr = sinf(roll * 0.5f);
            out_imu->orientation[0] = cp * cr;  /* w */
            out_imu->orientation[1] = sp * cr;  /* x */
            out_imu->orientation[2] = cp * sr;  /* y */
            out_imu->orientation[3] = -sp * sr; /* z */
        } else {
            out_imu->orientation[0] = 1.0f;
        }
    }
    /* 世界坐标系线性加速度：旋转加速度到世界坐标系 */
    {
        float qw = out_imu->orientation[0], qx = out_imu->orientation[1];
        float qy = out_imu->orientation[2], qz = out_imu->orientation[3];
        /* 共轭四元数旋转：a_world = q* ⊗ a_body ⊗ q */
        float aw[3] = {out_imu->accel[0], out_imu->accel[1], out_imu->accel[2]};
        /* v' = q ⊗ v ⊗ q* 其中 v 是纯四元数 (0, vx, vy, vz) */
        float t0 = -qx*aw[0] - qy*aw[1] - qz*aw[2];
        float t1 =  qw*aw[0] + qy*aw[2] - qz*aw[1];
        float t2 =  qw*aw[1] - qx*aw[2] + qz*aw[0];
        float t3 =  qw*aw[2] + qx*aw[1] - qy*aw[0];
        out_imu->linear_accel[0] = t0*(-qx) + t1*qw + t2*(-qz) - t3*(-qy);
        out_imu->linear_accel[1] = t0*(-qy) + t1*qz + t2*qw - t3*(-qx);
        out_imu->linear_accel[2] = t0*(-qz) - t1*qy + t2*qx + t3*qw;
    }
    out_imu->calibrated = 1;
    pp->last_imu = *out_imu;
    return 0;
}

int proprioception_fuse_force_torque(ProprioceptionProcessor* pp,
    const float* raw_ft_data, int ft_channels,
    ForceTorqueState* out_ft) {
    if (!pp || !raw_ft_data || !out_ft || ft_channels < 6)
        return -1;
    memset(out_ft, 0, sizeof(ForceTorqueState));
    out_ft->forces[0] = raw_ft_data[0]; /* Fx */
    out_ft->forces[1] = raw_ft_data[1]; /* Fy */
    out_ft->forces[2] = raw_ft_data[2]; /* Fz */
    out_ft->forces[3] = raw_ft_data[3]; /* Mx */
    out_ft->forces[4] = raw_ft_data[4]; /* My */
    out_ft->forces[5] = raw_ft_data[5]; /* Mz */
    out_ft->sensor_id = 0;
    out_ft->temperature = (ft_channels >= 7) ? raw_ft_data[6] : 25.0f;
    pp->last_ft = *out_ft;
    return 0;
}

int proprioception_compute_feature_vector(ProprioceptionProcessor* pp,
    const JointState* joints, const IMUState* imu,
    const ForceTorqueState* ft, float* feature_vector, size_t feature_dim) {
    if (!pp || !joints || !imu || !ft || !feature_vector ||
        feature_dim < PROPRIOCEPTION_FEATURE_DIM)
        return -1;
    memset(feature_vector, 0, feature_dim * sizeof(float));
    size_t idx = 0;
    /* 关节位置编码 [0, 31] */
    int nj = joints->num_joints;
    if (nj > PROPRIOCEPTION_MAX_JOINTS) nj = PROPRIOCEPTION_MAX_JOINTS;
    for (int i = 0; i < nj && idx < 32; i++, idx++)
        feature_vector[idx] = joints->joint_positions[i] / (float)M_PI;
    /* 关节速度编码 [32, 63] */
    for (int i = 0; i < nj && idx < 64; i++, idx++)
        feature_vector[idx] = tanhf(joints->joint_velocities[i] * 0.1f);
    /* 关节力矩编码 [64, 79] */
    for (int i = 0; i < nj && idx < 80; i++, idx++)
        feature_vector[idx] = tanhf(joints->joint_torques[i]);
    /* IMU姿态四元数 [80, 83] */
    for (int i = 0; i < 4 && idx < 84; i++, idx++)
        feature_vector[idx] = imu->orientation[i];
    /* IMU加速度 [84, 86] */
    for (int i = 0; i < 3 && idx < 87; i++, idx++)
        feature_vector[idx] = tanhf(imu->accel[i] * 0.1f);
    /* IMU角速度 [87, 89] */
    for (int i = 0; i < 3 && idx < 90; i++, idx++)
        feature_vector[idx] = tanhf(imu->gyro[i] * 0.1f);
    /* IMU线性加速度 [90, 92] */
    for (int i = 0; i < 3 && idx < 93; i++, idx++)
        feature_vector[idx] = tanhf(imu->linear_accel[i] * 0.1f);
    /* 力矩传感器 [93, 98] */
    for (int i = 0; i < 6 && idx < 99; i++, idx++)
        feature_vector[idx] = tanhf(ft->forces[i] * 0.01f);
    /* 额外特征：力模、姿态角、能量 */
    if (idx < 100) feature_vector[idx++] = sqrtf(ft->forces[0]*ft->forces[0] +
        ft->forces[1]*ft->forces[1] + ft->forces[2]*ft->forces[2]) * 0.01f;
    if (idx < 101) feature_vector[idx++] = imu->calibrated ? 1.0f : 0.0f;
    /* 剩余填充运动能量 */
    {
        float energy = 0.0f;
        for (int i = 0; i < nj; i++)
            energy += joints->joint_velocities[i] * joints->joint_velocities[i];
        energy = sqrtf(energy) / (float)(nj + 1);
        for (; idx < feature_dim; idx++)
            feature_vector[idx] = energy;
    }
    return 0;
}

int proprioception_ekf_update(ProprioceptionProcessor* pp,
    const float* measurement, int meas_dim, float dt) {
    if (!pp || !measurement || meas_dim <= 0 || dt <= 0.0f) return -1;
    if (meas_dim > PROPRIOCEPTION_FEATURE_DIM)
        meas_dim = PROPRIOCEPTION_FEATURE_DIM;
    /* 简化EKF：卡尔曼增益 K = P_pred * H^T * (H*P_pred*H^T + R)^(-1)
     * 使用恒定过程噪声Q和测量噪声R进行状态更新 */
    float Q = 0.01f, R = 0.1f;
    float P_pred = 0.5f; /* 简化的预测协方差 */
    for (int i = 0; i < meas_dim; i++) {
        float y = measurement[i] - pp->filtered_state[i]; /* 创新 */
        float S = P_pred + R;                              /* 创新协方差 */
        float K = P_pred / (S + 1e-8f);                    /* 卡尔曼增益 */
        pp->filtered_state[i] += K * y;                     /* 状态更新 */
        P_pred = (1.0f - K) * P_pred + Q;                   /* 协方差更新 */
    }
    pp->initialized = 1;
    return 0;
}

int proprioception_get_filtered_state(ProprioceptionProcessor* pp,
    float* state, int state_dim) {
    if (!pp || !state || state_dim <= 0) return -1;
    if (state_dim > PROPRIOCEPTION_FEATURE_DIM)
        state_dim = PROPRIOCEPTION_FEATURE_DIM;
    memcpy(state, pp->filtered_state, (size_t)state_dim * sizeof(float));
    return 0;
}
