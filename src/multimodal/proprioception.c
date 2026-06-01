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

/* 扩展卡尔曼滤波器状态维度 (6维完整协方差矩阵) */
#define PROPRIOCEPTION_EKF_DIM 6

/* ========== 6x6 矩阵运算辅助函数 (行优先存储) ========== */

/* 6x6 矩阵乘法: C = A * B */
static void ekf_mat6_mul(const float A[36], const float B[36], float C[36]) {
    memset(C, 0, sizeof(float) * 36);
    for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
        for (int k = 0; k < PROPRIOCEPTION_EKF_DIM; k++) {
            float aik = A[i * PROPRIOCEPTION_EKF_DIM + k];
            if (fabsf(aik) < 1e-10f) continue;
            for (int j = 0; j < PROPRIOCEPTION_EKF_DIM; j++) {
                C[i * PROPRIOCEPTION_EKF_DIM + j] +=
                    aik * B[k * PROPRIOCEPTION_EKF_DIM + j];
            }
        }
    }
}

/* 6x6 矩阵加法: C = A + B */
static void ekf_mat6_add(const float A[36], const float B[36], float C[36]) {
    for (int i = 0; i < 36; i++) {
        C[i] = A[i] + B[i];
    }
}

/* 6x6 矩阵求逆 (高斯-约旦消元法，带部分主元选择) */
static int ekf_mat6_inv(const float A[36], float invA[36]) {
    float aug[PROPRIOCEPTION_EKF_DIM][PROPRIOCEPTION_EKF_DIM * 2];
    /* 构建增广矩阵 [A | I] */
    for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
        for (int j = 0; j < PROPRIOCEPTION_EKF_DIM; j++) {
            aug[i][j] = A[i * PROPRIOCEPTION_EKF_DIM + j];
            aug[i][j + PROPRIOCEPTION_EKF_DIM] = (i == j) ? 1.0f : 0.0f;
        }
    }
    /* 高斯-约旦消元 */
    for (int col = 0; col < PROPRIOCEPTION_EKF_DIM; col++) {
        /* 部分主元选择 */
        int max_row = col;
        float max_val = fabsf(aug[col][col]);
        for (int row = col + 1; row < PROPRIOCEPTION_EKF_DIM; row++) {
            float val = fabsf(aug[row][col]);
            if (val > max_val) { max_val = val; max_row = row; }
        }
        if (max_val < 1e-12f) return -1; /* 矩阵奇异，不可逆 */
        /* 交换行 */
        if (max_row != col) {
            for (int j = 0; j < PROPRIOCEPTION_EKF_DIM * 2; j++) {
                float tmp = aug[col][j];
                aug[col][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }
        /* 归一化主元行 */
        float pivot = aug[col][col];
        for (int j = 0; j < PROPRIOCEPTION_EKF_DIM * 2; j++) {
            aug[col][j] /= pivot;
        }
        /* 消去其他所有行 */
        for (int row = 0; row < PROPRIOCEPTION_EKF_DIM; row++) {
            if (row == col) continue;
            float factor = aug[row][col];
            for (int j = 0; j < PROPRIOCEPTION_EKF_DIM * 2; j++) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    /* 提取逆矩阵 */
    for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
        for (int j = 0; j < PROPRIOCEPTION_EKF_DIM; j++) {
            invA[i * PROPRIOCEPTION_EKF_DIM + j] = aug[i][j + PROPRIOCEPTION_EKF_DIM];
        }
    }
    return 0;
}

/* EKF测量映射表: 从128维测量向量中提取6个关键维度用于完整协方差矩阵EKF */
static const int ekf_meas_map[PROPRIOCEPTION_EKF_DIM] = {
    0,    /* 关节位置0 (与关节位置1存在耦合) */
    1,    /* 关节位置1 (与关节位置0存在耦合) */
    80,   /* IMU四元数w (姿态分量，与IMU各分量耦合) */
    81,   /* IMU四元数x (姿态分量，与IMU各分量耦合) */
    99,   /* 力模大小 (与力矩传感器各分量耦合) */
    127   /* 运动能量 (全局统计量，与各维度耦合) */
};

struct ProprioceptionProcessor {
    JointState last_joints;
    IMUState last_imu;
    ForceTorqueState last_ft;
    float filtered_state[PROPRIOCEPTION_FEATURE_DIM];
    int initialized;
    int joint_count;
    /* 完整6维协方差矩阵EKF状态 */
    float ekf_P[36];       /* 6x6 状态协方差矩阵 (行优先) */
    float ekf_Q[36];       /* 6x6 过程噪声协方差矩阵 (行优先) */
    float ekf_R[36];       /* 6x6 测量噪声协方差矩阵 (行优先) */
    float ekf_state[6];    /* 6维EKF状态向量 */
    int ekf_initialized;   /* EKF是否已初始化 */
};

ProprioceptionProcessor* proprioception_create(void) {
    ProprioceptionProcessor* pp = (ProprioceptionProcessor*)safe_calloc(1, sizeof(ProprioceptionProcessor));
    if (!pp) return NULL;
    pp->initialized = 0;
    pp->joint_count = 0;
    pp->ekf_initialized = 0;
    memset(pp->filtered_state, 0, sizeof(pp->filtered_state));
    memset(pp->ekf_state, 0, sizeof(pp->ekf_state));
    /* 初始化EKF协方差矩阵 */
    memset(pp->ekf_P, 0, sizeof(pp->ekf_P));
    memset(pp->ekf_Q, 0, sizeof(pp->ekf_Q));
    memset(pp->ekf_R, 0, sizeof(pp->ekf_R));
    for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
        pp->ekf_P[i * PROPRIOCEPTION_EKF_DIM + i] = 1.0f;   /* P0 = I (初始不确定性为单位矩阵) */
        pp->ekf_Q[i * PROPRIOCEPTION_EKF_DIM + i] = 0.01f;   /* Q = 0.01 * I (过程噪声) */
        pp->ekf_R[i * PROPRIOCEPTION_EKF_DIM + i] = 0.1f;    /* R = 0.1 * I (测量噪声) */
    }
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
    /* RNE(递归牛顿-欧拉)算法计算完整关节力矩
     * 通过DH参数连杆模型，前向递推计算连杆运动学(角速度/角加速度/线加速度)
     * 后向递推计算连杆力/力矩，提取关节力矩(包含重力项和惯性项) */
    {
        /* 默认DH连杆参数: a(m), alpha(rad), d(m), mass(kg),
         * com_x,com_y,com_z(m), Ixx,Iyy,Izz(kg·m²)
         * 索引0基座,1+活动关节,超出6关节参数按比缩放 */
        static const float dp[][11] = {
            {0,0,0,5, 0,0,0, 1,1,1, 0},
            {0,-1.5708f,0.3f,2, 0,0,0.15f, 0.1f,0.1f,0.05f, 0},
            {0.35f,0,0,1.5f, 0.175f,0,0, 0.02f,0.15f,0.15f, 0},
            {0.3f,0,0,1, 0.15f,0,0, 0.01f,0.1f,0.1f, 0},
            {0,-1.5708f,0,0.5f, 0,0,0.05f, 0.005f,0.005f,0.002f, 0},
            {0,1.5708f,0,0.3f, 0,0,0.03f, 0.003f,0.003f,0.001f, 0},
            {0,0,0.1f,0.2f, 0,0,0.05f, 0.002f,0.002f,0.001f, 0}
        };
        float dt = 0.02f, g = 9.81f;
        /* 分配中间结果存储(3×N矢量按列分离存储) */
        size_t sz = (size_t)(n + 1) * 3 * sizeof(float);
        float* lw = (float*)safe_calloc(1, sz);
        float* la = (float*)safe_calloc(1, sz);
        float* lv = (float*)safe_calloc(1, sz);
        float* lc = (float*)safe_calloc(1, sz);
        if (!lw || !la || !lv || !lc) {
            safe_free((void**)&lw); safe_free((void**)&la);
            safe_free((void**)&lv); safe_free((void**)&lc);
            pp->last_joints = *out_joints; pp->joint_count = n;
            return -1;
        }
        /* 前向递推初始化: 基座加速度=[0,0,g]模拟重力 */
        float w[3]={0,0,0}, al[3]={0,0,0}, a0[3]={0,0,g};
        for (int i = 0; i < n; i++) {
            int li = (i < 6) ? (i + 1) : 6;
            float sc = (i < 6) ? 1 : 1.0f/(float)(i-4);
            float a_dh=dp[li][0]*sc, alp=dp[li][1], d_dh=dp[li][2]*sc;
            float th = out_joints->joint_positions[i];
            float qd = out_joints->joint_velocities[i];
            float qdd = 0;
            if (pp->joint_count > 0 && i < pp->last_joints.num_joints && dt > 1e-6f)
                qdd = (qd - pp->last_joints.joint_velocities[i]) / dt;
            float ct=cosf(th), st=sinf(th), ca=cosf(alp), sa=sinf(alp);
            /* R_{i-1}^i = (Rz(θ)*Rx(α))^T */
            float R11=ct,R12=st,R13=0;
            float R21=-st*ca,R22=ct*ca,R23=sa;
            float R31=st*sa,R32=-ct*sa,R33=ca;
            float Rw0=R11*w[0]+R12*w[1],Rw1=R21*w[0]+R22*w[1]+R23*w[2];
            float Rw2=R31*w[0]+R32*w[1]+R33*w[2];
            float wi[3]={Rw0,Rw1,Rw2+qd};
            float ali[3]={
                R11*al[0]+R12*al[1]+Rw1*qd,
                R21*al[0]+R22*al[1]+R23*al[2]-Rw0*qd,
                R31*al[0]+R32*al[1]+R33*al[2]+qdd
            };
            /* p_i^{i-1}=[a*cosθ, a*sinθ, d] */
            float px=a_dh*ct, py=a_dh*st, pz=d_dh;
            float acp[3]={al[1]*pz-al[2]*py, al[2]*px-al[0]*pz, al[0]*py-al[1]*px};
            float wcp[3]={w[1]*pz-w[2]*py, w[2]*px-w[0]*pz, w[0]*py-w[1]*px};
            float ww[3]={w[1]*wcp[2]-w[2]*wcp[1], w[2]*wcp[0]-w[0]*wcp[2], w[0]*wcp[1]-w[1]*wcp[0]};
            float ab[3]={a0[0]+acp[0]+ww[0], a0[1]+acp[1]+ww[1], a0[2]+acp[2]+ww[2]};
            float ai[3]={
                R11*ab[0]+R12*ab[1], R21*ab[0]+R22*ab[1]+R23*ab[2],
                R31*ab[0]+R32*ab[1]+R33*ab[2]
            };
            float rx=dp[li][4]*sc, ry=dp[li][5]*sc, rz=dp[li][6]*sc;
            float acr[3]={ali[1]*rz-ali[2]*ry, ali[2]*rx-ali[0]*rz, ali[0]*ry-ali[1]*rx};
            float wcr[3]={wi[1]*rz-wi[2]*ry, wi[2]*rx-wi[0]*rz, wi[0]*ry-wi[1]*rx};
            float wrr[3]={wi[1]*wcr[2]-wi[2]*wcr[1], wi[2]*wcr[0]-wi[0]*wcr[2], wi[0]*wcr[1]-wi[1]*wcr[0]};
            float ac[3]={ai[0]+acr[0]+wrr[0], ai[1]+acr[1]+wrr[1], ai[2]+acr[2]+wrr[2]};
            lw[i+1]=wi[0]; lw[(n+1)+(i+1)]=wi[1]; lw[2*(n+1)+(i+1)]=wi[2];
            la[i+1]=ali[0]; la[(n+1)+(i+1)]=ali[1]; la[2*(n+1)+(i+1)]=ali[2];
            lv[i+1]=ai[0]; lv[(n+1)+(i+1)]=ai[1]; lv[2*(n+1)+(i+1)]=ai[2];
            lc[i+1]=ac[0]; lc[(n+1)+(i+1)]=ac[1]; lc[2*(n+1)+(i+1)]=ac[2];
            w[0]=wi[0]; w[1]=wi[1]; w[2]=wi[2];
            al[0]=ali[0]; al[1]=ali[1]; al[2]=ali[2];
            a0[0]=ai[0]; a0[1]=ai[1]; a0[2]=ai[2];
        }
        /* 后向递推 */
        float fn[3]={0,0,0}, nn[3]={0,0,0};
        for (int i = n - 1; i >= 0; i--) {
            int li = (i < 6) ? (i + 1) : 6;
            float sc = (i < 6) ? 1 : 1.0f/(float)(i-4);
            float m=dp[li][3]*sc, Ix=dp[li][7]*sc, Iy=dp[li][8]*sc, Iz=dp[li][9]*sc;
            float rx=dp[li][4]*sc, ry=dp[li][5]*sc, rz=dp[li][6]*sc;
            float wi0=lw[i+1], wi1=lw[(n+1)+(i+1)], wi2=lw[2*(n+1)+(i+1)];
            float al0=la[i+1], al1=la[(n+1)+(i+1)], al2=la[2*(n+1)+(i+1)];
            float ac0=lc[i+1], ac1=lc[(n+1)+(i+1)], ac2=lc[2*(n+1)+(i+1)];
            float Fi[3]={m*ac0, m*ac1, m*ac2};
            float Iw[3]={Ix*wi0, Iy*wi1, Iz*wi2};
            float wIw[3]={wi1*Iw[2]-wi2*Iw[1], wi2*Iw[0]-wi0*Iw[2], wi0*Iw[1]-wi1*Iw[0]};
            float Ni[3]={Ix*al0+wIw[0], Iy*al1+wIw[1], Iz*al2+wIw[2]};
            float th_n, al_n, a_n, d_n;
            if (i < n - 1) {
                int ni = ((i+1)<6) ? (i+2) : 6;
                float ns = ((i+1)<6) ? 1 : 1.0f/(float)(i+1-4);
                th_n = out_joints->joint_positions[i+1];
                al_n = dp[ni][1]; a_n = dp[ni][0]*ns; d_n = dp[ni][2]*ns;
            } else { th_n=0; al_n=0; a_n=0; d_n=0; }
            float ct2=cosf(th_n), st2=sinf(th_n), ca2=cosf(al_n), sa2=sinf(al_n);
            float Rn11=ct2,Rn12=-st2*ca2,Rn13=st2*sa2;
            float Rn21=st2,Rn22=ct2*ca2,Rn23=-ct2*sa2;
            float Rn31=0,Rn32=sa2,Rn33=ca2;
            float Rf[3]={Rn11*fn[0]+Rn12*fn[1]+Rn13*fn[2],
                         Rn21*fn[0]+Rn22*fn[1]+Rn23*fn[2],
                         Rn31*fn[0]+Rn32*fn[1]+Rn33*fn[2]};
            float Rn[3]={Rn11*nn[0]+Rn12*nn[1]+Rn13*nn[2],
                         Rn21*nn[0]+Rn22*nn[1]+Rn23*nn[2],
                         Rn31*nn[0]+Rn32*nn[1]+Rn33*nn[2]};
            float fi[3]={Rf[0]+Fi[0], Rf[1]+Fi[1], Rf[2]+Fi[2]};
            float rcf[3]={ry*Fi[2]-rz*Fi[1], rz*Fi[0]-rx*Fi[2], rx*Fi[1]-ry*Fi[0]};
            float px2=a_n*ct2, py2=a_n*st2, pz2=d_n;
            float prf[3]={py2*Rf[2]-pz2*Rf[1], pz2*Rf[0]-px2*Rf[2], px2*Rf[1]-py2*Rf[0]};
            float ni[3]={Ni[0]+Rn[0]+rcf[0]+prf[0],
                         Ni[1]+Rn[1]+rcf[1]+prf[1],
                         Ni[2]+Rn[2]+rcf[2]+prf[2]};
            out_joints->joint_torques[i] = ni[2];
            fn[0]=fi[0]; fn[1]=fi[1]; fn[2]=fi[2];
            nn[0]=ni[0]; nn[1]=ni[1]; nn[2]=ni[2];
        }
        safe_free((void**)&lw); safe_free((void**)&la);
        safe_free((void**)&lv); safe_free((void**)&lc);
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
    /* ========== 完整6维协方差矩阵EKF更新 ========== */
    /* 从128维测量向量中提取6维EKF观测 */
    float z[PROPRIOCEPTION_EKF_DIM];
    int valid_meas = 0;
    for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
        int idx = ekf_meas_map[i];
        if (idx < meas_dim) {
            z[i] = measurement[idx];
            valid_meas++;
        } else {
            z[i] = pp->ekf_state[i]; /* 不可观测维度使用当前状态预测 */
        }
    }
    if (!pp->ekf_initialized && valid_meas > 0) {
        /* 首次测量: 用观测值初始化EKF状态 (减少初始收敛时间) */
        for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
            if (ekf_meas_map[i] < meas_dim) {
                pp->ekf_state[i] = measurement[ekf_meas_map[i]];
            }
        }
        pp->ekf_initialized = 1;
    }
    if (pp->ekf_initialized && valid_meas > 0) {
        /* 步骤1: 预测协方差 P_pred = P + Q */
        float P_pred[36];
        ekf_mat6_add(pp->ekf_P, pp->ekf_Q, P_pred);
        /* 步骤2: 计算创新向量 y = z - H*x
         * H = I (单位矩阵), 因为每个观测直接对应一个状态分量 */
        float y[PROPRIOCEPTION_EKF_DIM];
        for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
            y[i] = z[i] - pp->ekf_state[i];
        }
        /* 步骤3: 计算创新协方差 S = H * P_pred * H^T + R
         * 当 H = I 时, S = P_pred + R */
        float S[36];
        ekf_mat6_add(P_pred, pp->ekf_R, S);
        /* 步骤4: 计算卡尔曼增益 K = P_pred * H^T * S^(-1)
         * 当 H = I 时, K = P_pred * S^(-1) */
        float S_inv[36];
        if (ekf_mat6_inv(S, S_inv) == 0) {
            float K[36];
            ekf_mat6_mul(P_pred, S_inv, K);
            /* 步骤5: 状态更新 x = x + K * y */
            for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
                float correction = 0.0f;
                for (int j = 0; j < PROPRIOCEPTION_EKF_DIM; j++) {
                    correction += K[i * PROPRIOCEPTION_EKF_DIM + j] * y[j];
                }
                pp->ekf_state[i] += correction;
            }
            /* 步骤6: 协方差更新 P = (I - K*H) * P_pred
             * 当 H = I 时, P = (I - K) * P_pred */
            float I_KH[36];
            memset(I_KH, 0, sizeof(I_KH));
            for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
                for (int j = 0; j < PROPRIOCEPTION_EKF_DIM; j++) {
                    I_KH[i * PROPRIOCEPTION_EKF_DIM + j] =
                        -K[i * PROPRIOCEPTION_EKF_DIM + j];
                }
                I_KH[i * PROPRIOCEPTION_EKF_DIM + i] += 1.0f; /* I - K */
            }
            ekf_mat6_mul(I_KH, P_pred, pp->ekf_P);
        }
        /* 将EKF状态写回filtered_state的对应维度 */
        for (int i = 0; i < PROPRIOCEPTION_EKF_DIM; i++) {
            int idx = ekf_meas_map[i];
            if (idx < PROPRIOCEPTION_FEATURE_DIM) {
                pp->filtered_state[idx] = pp->ekf_state[i];
            }
        }
    }
    /* ========== 剩余维度: 使用标量卡尔曼滤波 ========== */
    float Q_scalar = 0.01f;   /* 标量过程噪声 */
    float R_scalar = 0.1f;    /* 标量测量噪声 */
    float P_scalar = 0.5f;    /* 标量协方差初值 */
    for (int i = 0; i < meas_dim; i++) {
        /* 跳过已被6维矩阵EKF处理的维度 */
        int is_ekf_dim = 0;
        for (int j = 0; j < PROPRIOCEPTION_EKF_DIM; j++) {
            if (i == ekf_meas_map[j]) { is_ekf_dim = 1; break; }
        }
        if (is_ekf_dim) continue;
        float y = measurement[i] - pp->filtered_state[i];   /* 创新 */
        float S = P_scalar + R_scalar;                       /* 创新协方差 */
        float K = P_scalar / (S + 1e-8f);                    /* 卡尔曼增益 */
        pp->filtered_state[i] += K * y;                      /* 状态更新 */
        P_scalar = (1.0f - K) * P_scalar + Q_scalar;         /* 协方差更新 */
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
