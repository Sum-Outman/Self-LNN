#ifndef SELFLNN_MOTOR_H
#define SELFLNN_MOTOR_H

/**
 * @file motor.h
 * @brief 电机控制预处理器 — PID/阻抗/力位混合控制、轨迹插值、前馈补偿
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MOTOR_MAX_JOINTS            32
#define MOTOR_FEATURE_DIM           64
#define MOTOR_TRAJECTORY_MAX_PTS    256

typedef enum {
    MOTOR_CTRL_POSITION = 0,
    MOTOR_CTRL_VELOCITY = 1,
    MOTOR_CTRL_TORQUE = 2,
    MOTOR_CTRL_IMPEDANCE = 3,
    MOTOR_CTRL_HYBRID = 4
} MotorControlMode;

typedef struct {
    float kp;
    float ki;
    float kd;
    float integral_limit;
    float output_limit;
    float deadband;
/* 新增安全限位参数
     * 确保PID输出不超过物理关节的安全范围 */
    float pos_min;          /* 位置下限 (rad或归一化) */
    float pos_max;          /* 位置上限 */
    float vel_limit;        /* 速度上限 (rad/s或归一化) */
    float torque_limit;     /* 力矩上限 (N·m或归一化) */
    int estop_active;       /* 急停标志：1=立即停止 */
    float prev_error;
    float integral;
    float derivative;
} PIDController;

typedef struct {
    float position;
    float velocity;
    float acceleration;
    float torque;
} TrajectoryPoint;

typedef struct MotorController MotorController;

MotorController* motor_controller_create(int num_joints);
void motor_controller_free(MotorController* mc);

int motor_pid_init(MotorController* mc, int joint_id,
    float kp, float ki, float kd);
int motor_pid_update(MotorController* mc, int joint_id,
    float setpoint, float measurement, float dt, float* output);
/* 紧急停止所有关节 — 设置所有PID的estop_active=1 */
int motor_controller_estop(MotorController* mc);
int motor_impedance_control(MotorController* mc, int joint_id,
    float desired_pos, float desired_vel, float external_force,
    float stiffness, float damping, float inertia, float dt,
    float* output_pos, float* output_torque);
int motor_trajectory_interpolate(MotorController* mc,
    const TrajectoryPoint* waypoints, int num_waypoints,
    float current_time, float dt, TrajectoryPoint* out_point);
int motor_feedforward_compensation(MotorController* mc, int joint_id,
    float desired_pos, float desired_vel, float desired_accel,
    float* ff_torque);
int motor_compute_feature_vector(MotorController* mc,
    const float* joint_positions, const float* joint_velocities,
    const float* joint_torques, int num_joints,
    float* feature_vector, size_t feature_dim);
int motor_get_control_output(MotorController* mc,
    float* control_signals, int max_signals);

#ifdef __cplusplus
}
#endif

#endif
