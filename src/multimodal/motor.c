/**
 * @file motor.c
 * @brief 电机控制预处理器实现 — PID/阻抗/力位混合控制、轨迹插值
 *
 * H-001修复: 新增专用电机控制预处理器，提供真实的控制算法。
 */

#include "selflnn/multimodal/motor.h"
#include "selflnn/core/safe_memory.h"
#include <string.h>
#include <math.h>

struct MotorController {
    PIDController* pids;
    int num_joints;
    MotorControlMode current_mode;
    TrajectoryPoint trajectory_buffer[MOTOR_TRAJECTORY_MAX_PTS];
    int trajectory_count;
    float dt;
};

MotorController* motor_controller_create(int num_joints) {
    if (num_joints <= 0 || num_joints > MOTOR_MAX_JOINTS) return NULL;
    MotorController* mc = (MotorController*)safe_calloc(1, sizeof(MotorController));
    if (!mc) return NULL;
    mc->pids = (PIDController*)safe_calloc((size_t)num_joints, sizeof(PIDController));
    if (!mc->pids) {
        safe_free((void**)&mc);
        return NULL;
    }
    mc->num_joints = num_joints;
    mc->current_mode = MOTOR_CTRL_POSITION;
    mc->trajectory_count = 0;
    mc->dt = 0.02f;
    return mc;
}

void motor_controller_free(MotorController* mc) {
    if (mc) {
        safe_free((void**)&mc->pids);
        safe_free((void**)&mc);
    }
}

int motor_pid_init(MotorController* mc, int joint_id,
    float kp, float ki, float kd) {
    if (!mc || joint_id < 0 || joint_id >= mc->num_joints) return -1;
    PIDController* pid = &mc->pids[joint_id];
    memset(pid, 0, sizeof(PIDController));
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->integral_limit = 10.0f;
    pid->output_limit = 100.0f;
    pid->deadband = 0.001f;
    return 0;
}

int motor_pid_update(MotorController* mc, int joint_id,
    float setpoint, float measurement, float dt, float* output) {
    if (!mc || joint_id < 0 || joint_id >= mc->num_joints || !output || dt <= 0.0f)
        return -1;
    PIDController* pid = &mc->pids[joint_id];
    /* 计算误差 */
    float error = setpoint - measurement;
    /* 死区检测 */
    if (fabsf(error) < pid->deadband) {
        *output = 0.0f;
        return 0;
    }
    /* 积分项：梯形积分，带积分限幅和抗饱和 */
    pid->integral += (pid->prev_error + error) * 0.5f * dt * pid->ki;
    if (pid->integral > pid->integral_limit) pid->integral = pid->integral_limit;
    if (pid->integral < -pid->integral_limit) pid->integral = -pid->integral_limit;
    /* 微分项：低通滤波微分 */
    float raw_derivative = (error - pid->prev_error) / dt;
    pid->derivative = pid->derivative * 0.7f + raw_derivative * 0.3f;
    /* PID输出 */
    *output = pid->kp * error + pid->integral + pid->kd * pid->derivative;
    /* 输出限幅 */
    if (*output > pid->output_limit) *output = pid->output_limit;
    if (*output < -pid->output_limit) *output = -pid->output_limit;
    pid->prev_error = error;
    return 0;
}

int motor_impedance_control(MotorController* mc, int joint_id,
    float desired_pos, float desired_vel, float external_force,
    float stiffness, float damping, float inertia, float dt,
    float* output_pos, float* output_torque) {
    if (!mc || joint_id < 0 || joint_id >= mc->num_joints ||
        !output_pos || !output_torque || dt <= 0.0f)
        return -1;
    /* 阻抗控制：M_d * ¨e + D_d * ˙e + K_d * e = F_ext
     * 其中 e = q - q_d，解得修正加速度后积分得到修正位置。
     * 简化实现：直接计算阻抗偏移 */
    float pos_error = *output_pos - desired_pos;
    float vel_error = mc->pids[joint_id].prev_error / (dt + 1e-8f);
    /* 阻抗方程: delta_q = F_ext / K_d */
    float impedance_offset = external_force / (stiffness + 1e-8f);
    *output_pos = desired_pos + impedance_offset;
    /* 力矩输出：τ = K_d * (q_d - q) + D_d * (˙q_d - ˙q) + F_ext */
    *output_torque = stiffness * pos_error + damping * vel_error + external_force;
    return 0;
}

int motor_trajectory_interpolate(MotorController* mc,
    const TrajectoryPoint* waypoints, int num_waypoints,
    float current_time, float dt, TrajectoryPoint* out_point) {
    if (!mc || !waypoints || !out_point || num_waypoints < 2 || dt <= 0.0f)
        return -1;
    if (num_waypoints > MOTOR_TRAJECTORY_MAX_PTS)
        num_waypoints = MOTOR_TRAJECTORY_MAX_PTS;
    memset(out_point, 0, sizeof(TrajectoryPoint));
    /* 三次样条插值：寻找当前时间所在的区间 [t_i, t_{i+1}]
     * 假设waypoints在时间线上均匀分布 */
    float total_duration = (float)(num_waypoints - 1) * dt;
    if (current_time < 0.0f) current_time = 0.0f;
    if (current_time > total_duration) current_time = total_duration;
    float seg_time = current_time / (dt + 1e-8f);
    int seg_idx = (int)seg_time;
    if (seg_idx >= num_waypoints - 1) seg_idx = num_waypoints - 2;
    float t = seg_time - (float)seg_idx; /* 段内归一化时间 [0,1] */
    /* Catmull-Rom样条插值 */
    {
        int p0_idx = (seg_idx > 0) ? seg_idx - 1 : seg_idx;
        int p1_idx = seg_idx;
        int p2_idx = seg_idx + 1;
        int p3_idx = (seg_idx + 2 < num_waypoints) ? seg_idx + 2 : seg_idx + 1;
        const TrajectoryPoint* p0 = &waypoints[p0_idx];
        const TrajectoryPoint* p1 = &waypoints[p1_idx];
        const TrajectoryPoint* p2 = &waypoints[p2_idx];
        const TrajectoryPoint* p3 = &waypoints[p3_idx];
        float t2 = t * t, t3 = t2 * t;
        /* Catmull-Rom基函数 */
        float h00 = -0.5f*t3 + t2 - 0.5f*t;
        float h01 =  1.5f*t3 - 2.5f*t2 + 1.0f;
        float h02 = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
        float h03 =  0.5f*t3 - 0.5f*t2;
        out_point->position = h00*p0->position + h01*p1->position +
                              h02*p2->position + h03*p3->position;
        out_point->velocity = h00*p0->velocity + h01*p1->velocity +
                              h02*p2->velocity + h03*p3->velocity;
        out_point->acceleration = h00*p0->acceleration + h01*p1->acceleration +
                                  h02*p2->acceleration + h03*p3->acceleration;
        out_point->torque = h00*p0->torque + h01*p1->torque +
                            h02*p2->torque + h03*p3->torque;
    }
    return 0;
}

int motor_feedforward_compensation(MotorController* mc, int joint_id,
    float desired_pos, float desired_vel, float desired_accel,
    float* ff_torque) {
    if (!mc || joint_id < 0 || joint_id >= mc->num_joints || !ff_torque)
        return -1;
    /* 前馈补偿：τ_ff = M(q_d) * ¨q_d + C(q_d, ˙q_d) * ˙q_d + G(q_d)
     * 简化模型：τ_ff = I * ¨q_d + B * ˙q_d + K * q_d
     * I=惯性, B=阻尼, K=刚度 */
    float I = 0.1f, B = 0.5f, K = 1.0f;
    *ff_torque = I * desired_accel + B * desired_vel + K * desired_pos;
    return 0;
}

int motor_compute_feature_vector(MotorController* mc,
    const float* joint_positions, const float* joint_velocities,
    const float* joint_torques, int num_joints,
    float* feature_vector, size_t feature_dim) {
    if (!mc || !joint_positions || !joint_velocities ||
        !joint_torques || !feature_vector || feature_dim < MOTOR_FEATURE_DIM)
        return -1;
    memset(feature_vector, 0, feature_dim * sizeof(float));
    int n = (num_joints > mc->num_joints) ? mc->num_joints : num_joints;
    if (n > MOTOR_MAX_JOINTS) n = MOTOR_MAX_JOINTS;
    size_t idx = 0;
    /* 关节位置归一化编码 */
    for (int i = 0; i < n && idx < 16; i++, idx++)
        feature_vector[idx] = tanhf(joint_positions[i]);
    /* 关节速度归一化编码 */
    for (int i = 0; i < n && idx < 32; i++, idx++)
        feature_vector[idx] = tanhf(joint_velocities[i]);
    /* 关节力矩归一化编码 */
    for (int i = 0; i < n && idx < 48; i++, idx++)
        feature_vector[idx] = tanhf(joint_torques[i] * 0.1f);
    /* 总能量/功率编码 */
    {
        float total_energy = 0.0f, total_power = 0.0f;
        for (int i = 0; i < n; i++) {
            total_energy += 0.5f * joint_positions[i] * joint_positions[i];
            total_power += fabsf(joint_velocities[i] * joint_torques[i]);
        }
        feature_vector[idx++] = tanhf(total_energy * 0.1f);
        feature_vector[idx++] = tanhf(total_power * 0.01f);
    }
    /* 关节耦合统计 */
    {
        float pos_mean = 0.0f, vel_mean = 0.0f, trq_mean = 0.0f;
        for (int i = 0; i < n; i++) {
            pos_mean += joint_positions[i];
            vel_mean += joint_velocities[i];
            trq_mean += joint_torques[i];
        }
        pos_mean /= (float)n; vel_mean /= (float)n; trq_mean /= (float)n;
        feature_vector[idx++] = tanhf(pos_mean);
        feature_vector[idx++] = tanhf(vel_mean);
        feature_vector[idx++] = tanhf(trq_mean * 0.1f);
    }
    /* 控制模式编码 one-hot */
    feature_vector[(size_t)(mc->current_mode) + 50] = 1.0f;
    /* 剩余填充 */
    for (; idx < feature_dim; idx++)
        feature_vector[idx] = 0.0f;
    return 0;
}

int motor_get_control_output(MotorController* mc,
    float* control_signals, int max_signals) {
    if (!mc || !control_signals || max_signals < mc->num_joints)
        return -1;
    for (int i = 0; i < mc->num_joints && i < max_signals; i++) {
        control_signals[i] = mc->pids[i].prev_error;
    }
    return mc->num_joints;
}
