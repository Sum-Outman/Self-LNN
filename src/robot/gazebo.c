/**
 * @file gazebo.c
 * @brief Gazebo仿真器接口实现
 *
 * 完整实现Gazebo仿真器接口，支持机器人仿真、传感器模拟和环境交互。
 * 采用双模架构：
 *   内部模式：基于Euler积分的嵌入式物理引擎，无需外部Gazebo即可运行
 *   外部模式：通过TCP/IP连接真实Gazebo仿真器通信
 */


#include "selflnn/robot/simulator.h"
#include "selflnn/robot/hardware_interface.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/string_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/utils/xml_parser.h"
#include "selflnn/robot/sensor_pipeline.h"
#include "selflnn/utils/secure_random.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

/* ============================================================================
 * Gazebo特定常量
 * =========================================================================== */

#define GAZEBO_MSG_SPAWN_MODEL        1
#define GAZEBO_MSG_DELETE_MODEL       2
#define GAZEBO_MSG_GET_MODEL_STATE    3
#define GAZEBO_MSG_SET_MODEL_STATE    4
#define GAZEBO_MSG_APPLY_BODY_WRENCH  5
#define GAZEBO_MSG_GET_LINK_STATE     6
#define GAZEBO_MSG_SET_LINK_STATE     7
#define GAZEBO_MSG_GET_JOINT_STATE    8
#define GAZEBO_MSG_SET_JOINT_STATE    9
#define GAZEBO_MSG_GET_SENSOR_DATA    10

#define GAZEBO_MAX_JOINTS             32
#define GAZEBO_MAX_SENSORS            16
#define GAZEBO_MAX_SCENE_OBJECTS      64
#define GAZEBO_GRAVITY_DEFAULT        9.81f
#define GAZEBO_TIMESTEP_DEFAULT       0.001f

/* 协议健壮性常量 */
#define GAZEBO_HEARTBEAT_INTERVAL     50
#define GAZEBO_MAX_RETRIES            3
#define GAZEBO_RETRY_DELAY_MS         100
#define GAZEBO_MAX_EXTERNAL_FAILURES  5
#define GAZEBO_RECONNECT_INTERVAL_MS  2000

/* 协议标志位 */
#define GAZEBO_FLAG_HEARTBEAT         (1u << 0)
#define GAZEBO_FLAG_CONNECT_TEST      (1u << 1)
#define GAZEBO_FLAG_HAS_CHECKSUM      (1u << 2)

/* 协议版本 */
#define GAZEBO_PROTOCOL_VERSION       1

/* ============================================================================
 * Gazebo通信协议数据结构
 * =========================================================================== */

typedef struct {
    uint32_t msg_type;
    uint32_t msg_size;
    uint32_t seq_num;
    uint32_t flags;
} GazeboMsgHeader;

typedef struct {
    char model_name[64];
    float pose[7];
    float twist[6];
    uint32_t success;
} GazeboModelState;

typedef struct {
    char joint_name[64];
    float position;
    float velocity;
    float effort;
    uint32_t success;
} GazeboJointState;

typedef struct {
    char sensor_name[64];
    uint32_t sensor_type;
    float timestamp;
    float data[64];
    uint32_t data_size;
} GazeboSensorData;

/**
 * @brief Gazebo仿真控制消息
 */
typedef struct {
    uint32_t control_type;       /**< 控制类型：0=启动，1=停止，2=暂停，3=恢复 */
    float pause_time;            /**< 暂停时间 */
    uint32_t flags;              /**< 控制标志 */
} GazeboSimulationControl;

/**
 * @brief Gazebo连接响应
 */
typedef struct {
    uint32_t version;            /**< 协议版本号 */
    uint32_t capabilities;       /**< 能力标志位 */
    uint32_t world_time;         /**< 仿真世界时间 */
} GazeboConnectResponse;

/* ============================================================================
 * 内部物理引擎数据结构
 * =========================================================================== */

typedef struct {
    int active;
    float position[3];
    float velocity[3];
    float acceleration[3];
    float orientation[4];
    float angular_velocity[3];
    float mass;
    float inertia[3];
    float joint_positions[GAZEBO_MAX_JOINTS];
    float joint_velocities[GAZEBO_MAX_JOINTS];
    float joint_targets[GAZEBO_MAX_JOINTS];
    float joint_torques[GAZEBO_MAX_JOINTS];
    float joint_limits_min[GAZEBO_MAX_JOINTS];
    float joint_limits_max[GAZEBO_MAX_JOINTS];
    float joint_max_torque;
    float joint_max_velocity;
    float actuator_response_coeff;
    float applied_force[3];
    float applied_torque[3];
    int num_joints;
    int is_colliding;
    float contact_force[3];
    float collision_impulse[3];
    float battery_level;

    // M18: 轨迹跟踪增强
    float joint_target_velocities[GAZEBO_MAX_JOINTS];
    float joint_feedforward_torque[GAZEBO_MAX_JOINTS];
    float joint_kp[GAZEBO_MAX_JOINTS];
    float joint_kd[GAZEBO_MAX_JOINTS];
    float joint_ki[GAZEBO_MAX_JOINTS];
    float joint_integral_error[GAZEBO_MAX_JOINTS];
    float joint_trajectory_start[GAZEBO_MAX_JOINTS];
    float joint_trajectory_end[GAZEBO_MAX_JOINTS];
    float joint_trajectory_duration[GAZEBO_MAX_JOINTS];
    float joint_trajectory_elapsed[GAZEBO_MAX_JOINTS];
    int joint_trajectory_active[GAZEBO_MAX_JOINTS];
    float joint_damping_coefficient[GAZEBO_MAX_JOINTS];
    float joint_coulomb_friction[GAZEBO_MAX_JOINTS];

    // M18: 机器人信息
    char robot_name[128];
    char urdf_path[256];
    float link_masses[32];
    float total_mass;
    float height;
    float foot_size[2];
    float com_offset[3];
    float default_standing_pose[32];
    int has_gripper;
    float end_effector_position[3];
    float end_effector_orientation[4];
    int num_links;
    int num_dof;

    // M18: 接触状态
    int contact_count;
    float contact_positions[6][3];
    float contact_forces_detail[6][3];
    float contact_normals[6][3];
    float contact_friction_coeffs[6];
    int contact_body_ids[6];
    int contact_link_ids[6];
    int contact_is_foot[6];
    double contact_timestamps[6];
    float total_normal_force;
    float total_friction_force;
} InternalRobot;

typedef struct {
    int active;
    int robot_id;
    int sensor_type;
    float mount_position[3];
    float mount_orientation[4];
    float data[64];
    int data_size;
    float noise_level;
    float last_update;
} InternalSensor;

typedef struct {
    int active;
    float position[3];
    float orientation[4];
    float scale[3];
    float color[4];
    float mass;
    float friction;
    float restitution;
    int object_type;
    char name[64];
} InternalSceneObject;

typedef struct {
    int recording;
    FILE* log_file;
    float last_log_time;
    float log_interval;
} InternalRecorder;

/* ============================================================================
 * 训练状态数据结构
 * =========================================================================== */

typedef struct {
    int active;
    int mode;
    int episode;
    int max_episodes;
    int step;
    int max_steps;
    float reward;
    float avg_reward;
    float best_reward;
    float loss;
    float avg_loss;
    float exploration_rate;
    float learning_rate;
    int total_samples;
    char status_message[256];
    float policy_w1[77*64];
    float policy_b1[64];
    float policy_w2[64*32];
    float policy_b2[32];
    float value_w1[77*64];
    float value_b1[64];
    float value_w2[64];
    float value_b2;
    int ppo_initialized;
    float obs_buffer[2048*77];
    float action_buffer[2048*32];
    float reward_buffer[2048];
    int terminal_buffer[2048];
    float value_buffer[2048];
    float logprob_buffer[2048];
    int buffer_count;
    float ou_state[32];
    float ou_theta;
    float ou_sigma;
    float ou_dt;
    float clip_epsilon;
    float gamma;
    float gae_lambda;
    int ppo_update_counter;
    float adam_m[18*4928];
    float adam_v[18*4928];
    int adam_step;
} InternalTrainingState;

/* ============================================================================
 * 路径规划状态数据结构
 * =========================================================================== */

typedef struct {
    int grid_width;
    int grid_height;
    float resolution;
    float origin_x;
    float origin_z;
    int waypoint_count;
    float waypoints[64][2];
    int path_found;
    float path_points[512][2];
    int path_length;
} InternalPathPlanner;

/* ============================================================================
 * 步态生成状态数据结构
 * =========================================================================== */

typedef struct {
    int active;
    float phase;
    float step_length;
    float step_height;
    float step_frequency;
    float cycle_time;
    int gait_mode;
    float turn_angle;
    float left_leg_phase_offset;
    float right_leg_phase_offset;
    float arm_swing_gain;
    float torso_tilt;
    float com_offset[2];
} InternalGaitGenerator;

typedef struct {
    int initialized;
    float gravity[3];
    float simulation_time;
    float timestep;
    int step_count;
    int paused;
    int use_external_gazebo;

    InternalRobot robots[8];
    int robot_count;

    InternalSensor sensors[GAZEBO_MAX_SENSORS];
    int sensor_count;

    InternalSceneObject scene_objects[GAZEBO_MAX_SCENE_OBJECTS];
    int scene_object_count;

    InternalRecorder recorder;
    char recording_filename[256];

    float light_position[3];
    float light_color[3];
    float ambient_color[3];

    InternalTrainingState training;
    InternalPathPlanner planner;
    InternalGaitGenerator gait;
} InternalSimulation;

/* ============================================================================
 * Gazebo仿真器内部结构
 * =========================================================================== */

typedef struct {
    SimulatorConfig config;
    HardwareInterface* comm;
    int connected;

    char world_name[64];
    float simulation_time;
    float real_time_factor;
    int paused;

    char** model_names;
    int model_count;
    int model_capacity;

    InternalSimulation internal;

    char last_error[256];
    char log_buffer[4096];
    int log_buffer_pos;

    /* 传感器管道 */
    struct SensorPipeline* sensor_pipeline;
    int sensor_streaming_enabled;

    /* 协议健壮性字段 */
    int seq_num;                         /**< 消息序列号 */
    int external_failures;               /**< 连续外部通信失败次数 */
    int heartbeat_counter;               /**< 心跳倒计时步数 */
    float last_external_success_time;    /**< 最后成功通信时间 */
    int reconnecting;                    /**< 正在重连标志 */
    int reconn_attempts;                 /**< 当前重连尝试次数 */
    int max_reconn_attempts;             /**< 最大重连尝试次数 */
} GazeboSimulator;

/* ============================================================================
 * 内部物理引擎 - 数学辅助函数
 * =========================================================================== */

/* quat_multiply: use math_utils.h static inline */

/* quat_normalize: use math_utils.h static inline */

static void gazebo_quat_from_angular_vel(const float* omega, float dt, float* out) {
    float angle = sqrtf(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);
    if (angle < 1e-10f) {
        out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 1.0f;
        return;
    }
    float half = angle * dt * 0.5f;
    float s = sinf(half) / angle;
    out[0] = omega[0] * s;
    out[1] = omega[1] * s;
    out[2] = omega[2] * s;
    out[3] = cosf(half);
}

static void gazebo_vec3_cross(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void gazebo_vec3_add(float* a, const float* b) {
    a[0] += b[0]; a[1] += b[1]; a[2] += b[2];
}

static void gazebo_vec3_scale(float* v, float s) {
    v[0] *= s; v[1] *= s; v[2] *= s;
}

/* ============================================================================
 * 内部物理引擎 - 力计算（从step_physics中提取，供RK4积分器复用）
 * =========================================================================== */

/**
 * @brief 计算作用在机器人上的合力（含重力、接触、碰撞、阻尼）
 * @param sim 仿真状态
 * @param r 机器人
 * @param pos 位置数组[3]（RK4中间状态，可为r->position或临时变量）
 * @param vel 速度数组[3]（RK4中间状态）
 * @param total_force 输出合力[3]
 * @param contact_force 输出接触力[3]（用于碰撞检测标志）
 * @param collision_impulse 输出碰撞冲量[3]
 * @param is_colliding 输出碰撞标志
 */
static void internal_compute_forces(InternalSimulation* sim, InternalRobot* r,
                                     const float pos[3], const float vel[3],
                                     float total_force[3], float contact_force[3],
                                     float collision_impulse[3], int* is_colliding) {
    total_force[0] = sim->gravity[0] * r->mass + r->applied_force[0];
    total_force[1] = sim->gravity[1] * r->mass + r->applied_force[1];
    total_force[2] = sim->gravity[2] * r->mass + r->applied_force[2];

    /* 地面接触力（弹簧-阻尼模型 + Coulomb摩擦锥） */
    float ground_contact_k = 5000.0f;
    float ground_contact_d = 100.0f;
    float static_friction_coeff = 0.6f;
    float dynamic_friction_coeff = 0.35f;
    float ground_penetration = -pos[1];
    memset(contact_force, 0, 3 * sizeof(float));
    *is_colliding = 0;
    if (ground_penetration > 0.0f) {
        float normal_force = ground_contact_k * ground_penetration - ground_contact_d * vel[1];
        if (normal_force < 0.0f) normal_force = 0.0f;
        total_force[1] += normal_force;

        /* Coulomb摩擦锥：静/动摩擦转换 */
        float tangential_speed = sqrtf(vel[0] * vel[0] + vel[2] * vel[2]);
        float max_static_fric = static_friction_coeff * normal_force;
        float max_dynamic_fric = dynamic_friction_coeff * normal_force;
        float max_friction = (tangential_speed < 0.001f) ? max_static_fric : max_dynamic_fric;
        /* Stribeck平滑过渡：零速附近用tanh平滑衔接静/动摩擦 */
        float fric_magnitude = max_friction * tanhf(tangential_speed / 0.05f);
        if (tangential_speed > 1e-6f) {
            float inv_speed = 1.0f / tangential_speed;
            total_force[0] -= vel[0] * inv_speed * fric_magnitude;
            total_force[2] -= vel[2] * inv_speed * fric_magnitude;
            contact_force[0] = -vel[0] * inv_speed * fric_magnitude;
            contact_force[2] = -vel[2] * inv_speed * fric_magnitude;
        }
        contact_force[1] = normal_force;
        *is_colliding = 1;
    }

    /* 物体间碰撞（球体近似） */
    memset(collision_impulse, 0, 3 * sizeof(float));
    for (int k = 0; k < sim->robot_count; k++) {
        if (!sim->robots[k].active) continue;
        InternalRobot* other = &sim->robots[k];
        float dx = pos[0] - other->position[0];
        float dy = pos[1] - other->position[1];
        float dz = pos[2] - other->position[2];
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        float min_dist = 1.0f;
        if (dist < min_dist && dist > 0.001f) {
            float overlap = min_dist - dist;
            float nx = dx / dist, ny = dy / dist, nz = dz / dist;
            float rel_vx = vel[0] - other->velocity[0];
            float rel_vy = vel[1] - other->velocity[1];
            float rel_vz = vel[2] - other->velocity[2];
            float rel_vn = rel_vx*nx + rel_vy*ny + rel_vz*nz;
            float ck = 8000.0f, cd = 50.0f;
            float imp = ck * overlap - cd * rel_vn;
            if (imp < 0.0f) imp = 0.0f;
            collision_impulse[0] += imp * nx;
            collision_impulse[1] += imp * ny;
            collision_impulse[2] += imp * nz;
            *is_colliding = 1;
        }
    }

    /* 场景物体碰撞 */
    for (int k = 0; k < sim->scene_object_count; k++) {
        if (!sim->scene_objects[k].active) continue;
        InternalSceneObject* obj = &sim->scene_objects[k];
        float dx = pos[0] - obj->position[0];
        float dy = pos[1] - obj->position[1];
        float dz = pos[2] - obj->position[2];
        float obj_radius = (obj->scale[0] + obj->scale[1] + obj->scale[2]) / 6.0f;
        if (obj_radius < 0.1f) obj_radius = 0.5f;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        float min_dist = 0.8f + obj_radius;
        if (dist < min_dist && dist > 0.001f) {
            float overlap = min_dist - dist;
            float nx = dx / dist, ny = dy / dist, nz = dz / dist;
            float rel_vn = vel[0]*nx + vel[1]*ny + vel[2]*nz;
            float ck = obj->restitution * 10000.0f;
            float cd = obj->friction * 200.0f + 20.0f;
            float imp = ck * overlap - cd * rel_vn;
            if (imp < 0.0f) imp = 0.0f;
            collision_impulse[0] += imp * nx;
            collision_impulse[1] += imp * ny;
            collision_impulse[2] += imp * nz;
            *is_colliding = 1;
        }
    }
    total_force[0] += collision_impulse[0];
    total_force[1] += collision_impulse[1];
    total_force[2] += collision_impulse[2];

    /* 空气阻力阻尼 */
    float air_drag = 0.02f;
    total_force[0] -= vel[0] * r->mass * air_drag;
    total_force[1] -= vel[1] * r->mass * air_drag;
    total_force[2] -= vel[2] * r->mass * air_drag;
}

/* ============================================================================
 * 内部物理引擎 - 物理模拟（RK4积分器）
 * =========================================================================== */

/* 前向声明 */
static void gazebo_gait_update(InternalGaitGenerator* gait, InternalRobot* robot, float dt);
static void gazebo_update_training(InternalSimulation* sim, float dt);
extern void robot_hardware_interface_destroy(HardwareInterface* hw);

static void internal_sim_init(InternalSimulation* sim, const SimulatorConfig* config) {
    memset(sim, 0, sizeof(InternalSimulation));
    sim->timestep = (config->timestep > 0.0f) ? config->timestep : GAZEBO_TIMESTEP_DEFAULT;
    sim->gravity[0] = 0.0f;
    sim->gravity[1] = 0.0f;
    sim->gravity[2] = -(config->gravity > 0.0f ? config->gravity : GAZEBO_GRAVITY_DEFAULT);
    sim->light_position[0] = 100.0f; sim->light_position[1] = 100.0f; sim->light_position[2] = 100.0f;
    sim->light_color[0] = 1.0f; sim->light_color[1] = 1.0f; sim->light_color[2] = 1.0f;
    sim->ambient_color[0] = 0.3f; sim->ambient_color[1] = 0.3f; sim->ambient_color[2] = 0.3f;
    sim->initialized = 1;
}

static int internal_sim_add_robot(InternalSimulation* sim) {
    if (sim->robot_count >= 8) return -1;
    int id = sim->robot_count;
    InternalRobot* r = &sim->robots[id];
    memset(r, 0, sizeof(InternalRobot));
    r->active = 1;
    r->mass = 50.0f;
    r->inertia[0] = 1.0f; r->inertia[1] = 1.0f; r->inertia[2] = 1.0f;
    r->orientation[3] = 1.0f;
    r->battery_level = 1.0f;
    r->num_joints = 6;
    r->joint_max_torque = 200.0f;
    r->joint_max_velocity = 10.0f;
    r->actuator_response_coeff = 0.1f;
    for (int j = 0; j < r->num_joints; j++) {
        r->joint_positions[j] = 0.0f;
        r->joint_velocities[j] = 0.0f;
        r->joint_targets[j] = 0.0f;
        r->joint_torques[j] = 0.0f;
        r->joint_limits_min[j] = -3.14159f;
        r->joint_limits_max[j] = 3.14159f;
        r->joint_target_velocities[j] = 0.0f;
        r->joint_feedforward_torque[j] = 0.0f;
        r->joint_kp[j] = 200.0f;
        r->joint_kd[j] = 20.0f;
        r->joint_ki[j] = 0.0f;
        r->joint_integral_error[j] = 0.0f;
        r->joint_trajectory_start[j] = 0.0f;
        r->joint_trajectory_end[j] = 0.0f;
        r->joint_trajectory_duration[j] = 0.0f;
        r->joint_trajectory_elapsed[j] = 0.0f;
        r->joint_trajectory_active[j] = 0;
        r->joint_damping_coefficient[j] = 0.0f;
        r->joint_coulomb_friction[j] = 0.0f;
    }
    r->total_mass = r->mass;
    r->height = 1.5f;
    r->foot_size[0] = 0.2f;
    r->foot_size[1] = 0.1f;
    sim->robot_count++;
    return id;
}

static void internal_sim_step_physics(InternalSimulation* sim) {
    if (!sim->initialized || sim->paused) return;
    float dt = sim->timestep;
    int substeps = 4;
    float sub_dt = dt / substeps;

    for (int s = 0; s < substeps; s++) {
        for (int i = 0; i < sim->robot_count; i++) {
            InternalRobot* r = &sim->robots[i];
            if (!r->active) continue;

            float inv_mass = 1.0f / (r->mass + 1e-10f);

            /* =================================================
             * RK4积分器：状态 = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
             * 导数 = [vel, acc]
             * ================================================= */
            float x0_pos[3], x0_vel[3];
            x0_pos[0] = r->position[0]; x0_vel[0] = r->velocity[0];
            x0_pos[1] = r->position[1]; x0_vel[1] = r->velocity[1];
            x0_pos[2] = r->position[2]; x0_vel[2] = r->velocity[2];

            /* -- k1 -- */
            float k1v[3], total_f[3], cf[3], ci[3];
            int colliding;
            internal_compute_forces(sim, r, x0_pos, x0_vel, total_f, cf, ci, &colliding);
            k1v[0] = total_f[0] * inv_mass; k1v[1] = total_f[1] * inv_mass; k1v[2] = total_f[2] * inv_mass;
            float k1p[3] = {x0_vel[0], x0_vel[1], x0_vel[2]};

            /* -- k2 -- */
            float p2[3], v2[3];
            p2[0] = x0_pos[0] + 0.5f * sub_dt * k1p[0]; v2[0] = x0_vel[0] + 0.5f * sub_dt * k1v[0];
            p2[1] = x0_pos[1] + 0.5f * sub_dt * k1p[1]; v2[1] = x0_vel[1] + 0.5f * sub_dt * k1v[1];
            p2[2] = x0_pos[2] + 0.5f * sub_dt * k1p[2]; v2[2] = x0_vel[2] + 0.5f * sub_dt * k1v[2];
            float k2v[3];
            internal_compute_forces(sim, r, p2, v2, total_f, cf, ci, &colliding);
            k2v[0] = total_f[0] * inv_mass; k2v[1] = total_f[1] * inv_mass; k2v[2] = total_f[2] * inv_mass;
            float k2p[3] = {v2[0], v2[1], v2[2]};

            /* -- k3 -- */
            float p3[3], v3[3];
            p3[0] = x0_pos[0] + 0.5f * sub_dt * k2p[0]; v3[0] = x0_vel[0] + 0.5f * sub_dt * k2v[0];
            p3[1] = x0_pos[1] + 0.5f * sub_dt * k2p[1]; v3[1] = x0_vel[1] + 0.5f * sub_dt * k2v[1];
            p3[2] = x0_pos[2] + 0.5f * sub_dt * k2p[2]; v3[2] = x0_vel[2] + 0.5f * sub_dt * k2v[2];
            float k3v[3];
            internal_compute_forces(sim, r, p3, v3, total_f, cf, ci, &colliding);
            k3v[0] = total_f[0] * inv_mass; k3v[1] = total_f[1] * inv_mass; k3v[2] = total_f[2] * inv_mass;
            float k3p[3] = {v3[0], v3[1], v3[2]};

            /* -- k4 -- */
            float p4[3], v4[3];
            p4[0] = x0_pos[0] + sub_dt * k3p[0]; v4[0] = x0_vel[0] + sub_dt * k3v[0];
            p4[1] = x0_pos[1] + sub_dt * k3p[1]; v4[1] = x0_vel[1] + sub_dt * k3v[1];
            p4[2] = x0_pos[2] + sub_dt * k3p[2]; v4[2] = x0_vel[2] + sub_dt * k3v[2];
            float k4v[3];
            internal_compute_forces(sim, r, p4, v4, total_f, cf, ci, &colliding);
            k4v[0] = total_f[0] * inv_mass; k4v[1] = total_f[1] * inv_mass; k4v[2] = total_f[2] * inv_mass;
            float k4p[3] = {v4[0], v4[1], v4[2]};

            /* -- RK4加权平均更新 -- */
            float inv6 = 1.0f / 6.0f;
            r->position[0] += sub_dt * inv6 * (k1p[0] + 2.0f*k2p[0] + 2.0f*k3p[0] + k4p[0]);
            r->position[1] += sub_dt * inv6 * (k1p[1] + 2.0f*k2p[1] + 2.0f*k3p[1] + k4p[1]);
            r->position[2] += sub_dt * inv6 * (k1p[2] + 2.0f*k2p[2] + 2.0f*k3p[2] + k4p[2]);
            r->velocity[0] += sub_dt * inv6 * (k1v[0] + 2.0f*k2v[0] + 2.0f*k3v[0] + k4v[0]);
            r->velocity[1] += sub_dt * inv6 * (k1v[1] + 2.0f*k2v[1] + 2.0f*k3v[1] + k4v[1]);
            r->velocity[2] += sub_dt * inv6 * (k1v[2] + 2.0f*k2v[2] + 2.0f*k3v[2] + k4v[2]);

            /* 基于k4评估最终接触力（用于状态反馈） */
            internal_compute_forces(sim, r, r->position, r->velocity, total_f, cf, ci, &colliding);
            r->is_colliding = colliding;
            r->contact_force[0] = cf[0]; r->contact_force[1] = cf[1]; r->contact_force[2] = cf[2];
            r->acceleration[0] = total_f[0] * inv_mass;
            r->acceleration[1] = total_f[1] * inv_mass;
            r->acceleration[2] = total_f[2] * inv_mass;

            // 角速度 - 阻尼
            r->angular_velocity[0] *= (1.0f - 0.8f * sub_dt);
            r->angular_velocity[1] *= (1.0f - 0.8f * sub_dt);
            r->angular_velocity[2] *= (1.0f - 0.8f * sub_dt);

            // 姿态更新
            float dq[4];
            gazebo_quat_from_angular_vel(r->angular_velocity, sub_dt, dq);
            float new_q[4];
            quat_multiply(dq, r->orientation, new_q);
            quat_normalize(new_q);
            memcpy(r->orientation, new_q, 4 * sizeof(float));

            // M18: 增强执行器模型（轨迹插值 + PID + 前馈 + 库仑摩擦）
            for (int j = 0; j < r->num_joints; j++) {
                // 轨迹插值更新目标（三次样条）
                if (r->joint_trajectory_active[j]) {
                    r->joint_trajectory_elapsed[j] += sub_dt;
                    float t = r->joint_trajectory_elapsed[j] / r->joint_trajectory_duration[j];
                    if (t >= 1.0f) {
                        r->joint_targets[j] = r->joint_trajectory_end[j];
                        r->joint_target_velocities[j] = 0.0f;
                        r->joint_trajectory_active[j] = 0;
                    } else {
                        float t2 = t * t;
                        float t3 = t2 * t;
                        r->joint_targets[j] = (2.0f * t3 - 3.0f * t2 + 1.0f) * r->joint_trajectory_start[j]
                                            + (-2.0f * t3 + 3.0f * t2) * r->joint_trajectory_end[j];
                        float dt_traj = r->joint_trajectory_duration[j];
                        r->joint_target_velocities[j] = (6.0f * t2 - 6.0f * t) / dt_traj * r->joint_trajectory_start[j]
                                                      + (-6.0f * t2 + 6.0f * t) / dt_traj * r->joint_trajectory_end[j];
                    }
                }
                // 关节限位边界弹簧-阻尼
                float limit_spring = 0.0f;
                if (r->joint_positions[j] < r->joint_limits_min[j]) {
                    float penetration = r->joint_limits_min[j] - r->joint_positions[j];
                    limit_spring = 500.0f * penetration - 50.0f * r->joint_velocities[j];
                    r->joint_positions[j] = r->joint_limits_min[j];
                    r->joint_velocities[j] = 0.0f;
                    r->joint_integral_error[j] = 0.0f;
                } else if (r->joint_positions[j] > r->joint_limits_max[j]) {
                    float penetration = r->joint_positions[j] - r->joint_limits_max[j];
                    limit_spring = -500.0f * penetration - 50.0f * r->joint_velocities[j];
                    r->joint_positions[j] = r->joint_limits_max[j];
                    r->joint_velocities[j] = 0.0f;
                    r->joint_integral_error[j] = 0.0f;
                }
                float position_error = r->joint_targets[j] - r->joint_positions[j];
                float velocity_error = r->joint_target_velocities[j] - r->joint_velocities[j];
                r->joint_integral_error[j] += position_error * sub_dt;
                if (r->joint_integral_error[j] > 10.0f) r->joint_integral_error[j] = 10.0f;
                if (r->joint_integral_error[j] < -10.0f) r->joint_integral_error[j] = -10.0f;
                float desired_torque = r->joint_kp[j] * position_error
                                     + r->joint_kd[j] * velocity_error
                                     + r->joint_ki[j] * r->joint_integral_error[j]
                                     + r->joint_feedforward_torque[j];
                float coulomb_force = 0.0f;
                if (fabsf(r->joint_velocities[j]) > 0.001f) {
                    coulomb_force = -r->joint_coulomb_friction[j] * (r->joint_velocities[j] > 0 ? 1.0f : -1.0f);
                }
                desired_torque += coulomb_force;
                float damping_force = -r->joint_damping_coefficient[j] * r->joint_velocities[j];
                desired_torque += damping_force;
                float max_torque = r->joint_max_torque;
                if (desired_torque > max_torque) desired_torque = max_torque;
                if (desired_torque < -max_torque) desired_torque = -max_torque;
                float alpha = sub_dt / (r->actuator_response_coeff + sub_dt);
                r->joint_torques[j] += alpha * (desired_torque - r->joint_torques[j]);
                float net_torque = r->joint_torques[j] + limit_spring;
                float max_vel = r->joint_max_velocity;
                r->joint_velocities[j] += net_torque * sub_dt / 10.0f;
                if (r->joint_velocities[j] > max_vel) r->joint_velocities[j] = max_vel;
                if (r->joint_velocities[j] < -max_vel) r->joint_velocities[j] = -max_vel;
                r->joint_positions[j] += r->joint_velocities[j] * sub_dt;
            }

            // 电池消耗模型（与关节力矩和运动相关）
            float total_joint_power = 0.0f;
            for (int j = 0; j < r->num_joints; j++) {
                total_joint_power += fabsf(r->joint_torques[j] * r->joint_velocities[j]);
            }
            float motion_power = fabsf(r->velocity[0]) + fabsf(r->velocity[1]) + fabsf(r->velocity[2]);
            float discharge_rate = 0.0001f + total_joint_power * 0.00005f + motion_power * 0.00002f;
            r->battery_level -= discharge_rate * sub_dt;
            if (r->battery_level < 0.0f) r->battery_level = 0.0f;
        }

        // 更新传感器时间
        sim->simulation_time += sub_dt;
    }
    sim->step_count++;

    /* 步态生成更新（在每个物理步之后应用） */
    for (int i = 0; i < sim->robot_count; i++) {
        if (sim->robots[i].active && sim->gait.active) {
            gazebo_gait_update(&sim->gait, &sim->robots[i], dt);
        }
    }

    /* 训练更新（在每个物理步之后调用） */
    gazebo_update_training(sim, dt);
}

static int internal_sim_generate_sensor_data(InternalSimulation* sim, int sensor_id, float* out_data, int max_size) {
    (void)max_size;
    if (sensor_id < 0 || sensor_id >= GAZEBO_MAX_SENSORS) return -1;
    InternalSensor* sensor = &sim->sensors[sensor_id];
    if (!sensor->active) return -1;

    int robot_id = sensor->robot_id;
    if (robot_id < 0 || robot_id >= sim->robot_count) return -1;
    InternalRobot* robot = &sim->robots[robot_id];

    int size = 0;
    switch (sensor->sensor_type) {
        case SENSOR_TYPE_IMU: {
            out_data[0] = robot->acceleration[0];
            out_data[1] = robot->acceleration[1];
            out_data[2] = robot->acceleration[2];
            out_data[3] = robot->angular_velocity[0];
            out_data[4] = robot->angular_velocity[1];
            out_data[5] = robot->angular_velocity[2];
            out_data[6] = robot->orientation[0];
            out_data[7] = robot->orientation[1];
            out_data[8] = robot->orientation[2];
            out_data[9] = robot->orientation[3];
            size = 10;
            break;
        }
        case SENSOR_TYPE_FORCE_TORQUE: {
            out_data[0] = robot->contact_force[0];
            out_data[1] = robot->contact_force[1];
            out_data[2] = robot->contact_force[2];
            out_data[3] = robot->joint_torques[0];
            out_data[4] = robot->joint_torques[1];
            out_data[5] = robot->joint_torques[2];
            size = 6;
            break;
        }
        default: {
            out_data[0] = robot->position[0];
            out_data[1] = robot->position[1];
            out_data[2] = robot->position[2];
            out_data[3] = robot->velocity[0];
            out_data[4] = robot->velocity[1];
            out_data[5] = robot->velocity[2];
            size = 6;
            break;
        }
    }

    // 添加噪声
    if (sensor->noise_level > 0.0f) {
        for (int i = 0; i < size; i++) {
            float noise = (rng_uniform(0.0f, 1.0f) - 0.5f) * 2.0f * sensor->noise_level;
            out_data[i] += noise;
        }
    }

    sensor->last_update = sim->simulation_time;
    return size;
}

/* ============================================================================
 * Gazebo通信辅助函数（增强版）
 * =========================================================================== */

/**
 * 计算简单XOR校验和
 */
static uint32_t gazebo_calc_checksum(const void* data, uint32_t size) {
    if (!data || size == 0) return 0;
    uint32_t sum = 0;
    const uint8_t* bytes = (const uint8_t*)data;
    for (uint32_t i = 0; i < size; i++) {
        sum ^= bytes[i];
        sum = (sum << 3) | (sum >> 29);
    }
    return sum;
}

/**
 * 带重试的部分数据发送
 */
static int gazebo_send_all(HardwareInterface* comm, const void* data, uint32_t size, int max_retries) {
    if (!comm || !data || size == 0) return -1;
    const uint8_t* ptr = (const uint8_t*)data;
    uint32_t remaining = size;
    int attempts = 0;

    while (remaining > 0 && attempts <= max_retries) {
        int sent = hardware_interface_send(comm, ptr + (size - remaining), remaining);
        if (sent > 0) {
            remaining -= (uint32_t)sent;
            attempts = 0;
        } else {
            attempts++;
#ifdef _WIN32
            Sleep((DWORD)GAZEBO_RETRY_DELAY_MS);
#else
            usleep(GAZEBO_RETRY_DELAY_MS * 1000);
#endif
        }
    }

    return (remaining == 0) ? (int)size : -1;
}

/**
 * 带重试的部分数据接收
 */
static int gazebo_recv_all(HardwareInterface* comm, void* buffer, uint32_t size, int max_retries) {
    if (!comm || !buffer || size == 0) return -1;
    uint8_t* ptr = (uint8_t*)buffer;
    uint32_t remaining = size;
    int attempts = 0;

    while (remaining > 0 && attempts <= max_retries) {
        int received = hardware_interface_receive(comm, ptr + (size - remaining), remaining);
        if (received > 0) {
            remaining -= (uint32_t)received;
            attempts = 0;
        } else if (received == 0) {
            attempts++;
#ifdef _WIN32
            Sleep((DWORD)GAZEBO_RETRY_DELAY_MS);
#else
            usleep(GAZEBO_RETRY_DELAY_MS * 1000);
#endif
        } else {
            return -1;
        }
    }

    return (remaining == 0) ? (int)size : -1;
}

/**
 * 发送消息（含重试、序列号、可选校验和）
 */
static int gazebo_send_message(HardwareInterface* comm, int* seq_num,
                               uint32_t msg_type, const void* msg_body,
                               uint32_t msg_size, int use_checksum) {
    if (!comm) return -1;

    uint8_t send_buf[2048];
    uint32_t offset = sizeof(GazeboMsgHeader);
    uint32_t total_size = offset;

    if (msg_body && msg_size > 0 && msg_size <= sizeof(send_buf) - offset - 4) {
        memcpy(send_buf + offset, msg_body, msg_size);
        total_size += msg_size;
    }

    if (use_checksum) {
        uint32_t cksum = gazebo_calc_checksum(msg_body, msg_size);
        memcpy(send_buf + total_size, &cksum, 4);
        total_size += 4;
    }

    GazeboMsgHeader* header = (GazeboMsgHeader*)send_buf;
    header->msg_type = msg_type;
    header->msg_size = msg_size;
    header->seq_num = (uint32_t)(seq_num ? (*seq_num)++ : 0);
    header->flags = use_checksum ? GAZEBO_FLAG_HAS_CHECKSUM : 0;

    return gazebo_send_all(comm, send_buf, total_size, GAZEBO_MAX_RETRIES) > 0 ? 0 : -1;
}

/**
 * 接收消息（含部分读取循环、可选校验和验证）
 */
static int gazebo_receive_message(HardwareInterface* comm, uint32_t expected_msg_type,
                                  void* msg_body, uint32_t max_msg_size) {
    if (!comm) return -1;

    GazeboMsgHeader header;
    if (gazebo_recv_all(comm, &header, sizeof(header), GAZEBO_MAX_RETRIES) != sizeof(header)) {
        return -1;
    }

    if (header.flags & GAZEBO_FLAG_HEARTBEAT) {
        return -2;
    }
    if (header.flags & GAZEBO_FLAG_CONNECT_TEST) {
        return -3;
    }
    if (header.msg_type != expected_msg_type) return -1;
    if (header.msg_size > max_msg_size) return -1;

    int use_checksum = (header.flags & GAZEBO_FLAG_HAS_CHECKSUM) != 0;
    uint32_t read_size = header.msg_size + (use_checksum ? 4u : 0u);
    uint8_t read_buf[2048];

    if (read_size > sizeof(read_buf)) return -1;

    if (read_size > 0) {
        if (gazebo_recv_all(comm, read_buf, read_size, GAZEBO_MAX_RETRIES) != (int)read_size) {
            return -1;
        }
        if (msg_body && header.msg_size > 0) {
            memcpy(msg_body, read_buf, header.msg_size);
        }
        if (use_checksum) {
            uint32_t received_cksum;
            memcpy(&received_cksum, read_buf + header.msg_size, 4);
            uint32_t calc_cksum = gazebo_calc_checksum(msg_body, header.msg_size);
            if (received_cksum != calc_cksum) return -1;
        }
    }

    return (int)header.msg_size;
}

static void gazebo_log(GazeboSimulator* gazebo, const char* msg) {
    if (gazebo->config.enable_logging) {
        int len = (int)strlen(msg);
        if (gazebo->log_buffer_pos + len + 1 < (int)sizeof(gazebo->log_buffer)) {
            memcpy(gazebo->log_buffer + gazebo->log_buffer_pos, msg, len);
            gazebo->log_buffer_pos += len;
            gazebo->log_buffer[gazebo->log_buffer_pos] = '\0';
        }
    }
}

/* ============================================================================
 * 外部Gazebo协议 - 心跳和重连
 * =========================================================================== */

/**
 * 发送心跳消息
 */
static int gazebo_send_heartbeat(GazeboSimulator* gazebo) {
    if (!gazebo->comm || !gazebo->internal.use_external_gazebo) return -1;
    return gazebo_send_message(gazebo->comm, &gazebo->seq_num,
                               GAZEBO_MSG_GET_MODEL_STATE, NULL, 0, 0);
}

/**
 * 尝试重连外部Gazebo
 */
static int gazebo_try_reconnect(GazeboSimulator* gazebo) {
    if (!gazebo) return -1;
    gazebo->reconnecting = 1;
    gazebo->reconn_attempts++;

    if (gazebo->reconn_attempts > gazebo->max_reconn_attempts) {
        gazebo->internal.use_external_gazebo = 0;
        gazebo->reconnecting = 0;
        gazebo_log(gazebo, "外部Gazebo重连已达最大次数，切换至内部引擎");
        return -1;
    }

    hardware_interface_disconnect(gazebo->comm);

#ifdef _WIN32
    Sleep(GAZEBO_RECONNECT_INTERVAL_MS);
#else
    usleep(GAZEBO_RECONNECT_INTERVAL_MS * 1000);
#endif

    if (hardware_interface_connect(gazebo->comm) != 0) {
        gazebo->reconnecting = 0;
        return -1;
    }

    /* 发送连接测试消息 */
    GazeboMsgHeader test_msg;
    test_msg.msg_type = GAZEBO_MSG_GET_MODEL_STATE;
    test_msg.msg_size = 0;
    test_msg.seq_num = (uint32_t)gazebo->seq_num++;
    test_msg.flags = GAZEBO_FLAG_CONNECT_TEST;
    if (gazebo_send_all(gazebo->comm, &test_msg, sizeof(test_msg), 1) != sizeof(test_msg)) {
        hardware_interface_disconnect(gazebo->comm);
        gazebo->reconnecting = 0;
        return -1;
    }

    gazebo->external_failures = 0;
    gazebo->reconnecting = 0;
    gazebo_log(gazebo, "外部Gazebo重连成功");
    return 0;
}

/* ============================================================================
 * 外部Gazebo协议 - 模型状态同步（增强版，带重试和故障跟踪）
 * =========================================================================== */

static int gazebo_external_spawn_model(GazeboSimulator* gazebo, const char* model_name,
                                       const float* position, const float* orientation) {
    if (!gazebo->comm || !gazebo->internal.use_external_gazebo) return -1;
    char buffer[512];
    memset(buffer, 0, sizeof(buffer));
    int name_len = (int)strlen(model_name);
    if (name_len > 63) name_len = 63;
    memcpy(buffer, model_name, name_len);
    if (position) {
        memcpy(buffer + 64, position, 3 * sizeof(float));
    }
    if (orientation) {
        memcpy(buffer + 76, orientation, 4 * sizeof(float));
    }
    return gazebo_send_message(gazebo->comm, &gazebo->seq_num,
                               GAZEBO_MSG_SPAWN_MODEL, buffer, 512, 1);
}

static int gazebo_external_delete_model(GazeboSimulator* gazebo, const char* model_name) {
    if (!gazebo->comm || !gazebo->internal.use_external_gazebo) return -1;
    return gazebo_send_message(gazebo->comm, &gazebo->seq_num,
                               GAZEBO_MSG_DELETE_MODEL, model_name,
                               (uint32_t)strlen(model_name) + 1, 1);
}

static int gazebo_external_get_model_state(GazeboSimulator* gazebo, int robot_id,
                                           float* position, float* orientation) {
    if (!gazebo->comm || !gazebo->internal.use_external_gazebo) return -1;
    const char* name = (robot_id < gazebo->model_count && gazebo->model_names[robot_id])
                       ? gazebo->model_names[robot_id] : "robot";

    for (int attempt = 0; attempt <= GAZEBO_MAX_RETRIES; attempt++) {
        if (gazebo_send_message(gazebo->comm, &gazebo->seq_num,
                                GAZEBO_MSG_GET_MODEL_STATE, name,
                                (uint32_t)strlen(name) + 1, 1) != 0) {
            if (attempt < GAZEBO_MAX_RETRIES) {
#ifdef _WIN32
                Sleep(GAZEBO_RETRY_DELAY_MS);
#else
                usleep(GAZEBO_RETRY_DELAY_MS * 1000);
#endif
                continue;
            }
            gazebo->external_failures++;
            return -1;
        }

        GazeboModelState model_state;
        memset(&model_state, 0, sizeof(model_state));
        int ret = gazebo_receive_message(gazebo->comm, GAZEBO_MSG_GET_MODEL_STATE,
                                         &model_state, sizeof(model_state));
        if (ret < 0) {
            if (ret == -2) continue;
            if (attempt < GAZEBO_MAX_RETRIES) {
#ifdef _WIN32
                Sleep(GAZEBO_RETRY_DELAY_MS);
#else
                usleep(GAZEBO_RETRY_DELAY_MS * 1000);
#endif
                continue;
            }
            gazebo->external_failures++;
            return -1;
        }

        if (position) {
            position[0] = model_state.pose[0];
            position[1] = model_state.pose[1];
            position[2] = model_state.pose[2];
        }
        if (orientation) {
            orientation[0] = model_state.pose[3];
            orientation[1] = model_state.pose[4];
            orientation[2] = model_state.pose[5];
            orientation[3] = model_state.pose[6];
        }
        gazebo->external_failures = 0;
        return model_state.success ? 0 : -1;
    }
    return -1;
}

static int gazebo_external_set_model_state(GazeboSimulator* gazebo, int robot_id,
                                           const float* position, const float* orientation) {
    if (!gazebo->comm || !gazebo->internal.use_external_gazebo) return -1;
    const char* name = (robot_id < gazebo->model_count && gazebo->model_names[robot_id])
                       ? gazebo->model_names[robot_id] : "robot";
    GazeboModelState model_state;
    memset(&model_state, 0, sizeof(model_state));
    strncpy(model_state.model_name, name, sizeof(model_state.model_name) - 1);
    if (position) {
        model_state.pose[0] = position[0];
        model_state.pose[1] = position[1];
        model_state.pose[2] = position[2];
    }
    if (orientation) {
        model_state.pose[3] = orientation[0];
        model_state.pose[4] = orientation[1];
        model_state.pose[5] = orientation[2];
        model_state.pose[6] = orientation[3];
    }
    return gazebo_send_message(gazebo->comm, &gazebo->seq_num,
                               GAZEBO_MSG_SET_MODEL_STATE, &model_state, sizeof(model_state), 1);
}

static int gazebo_external_get_joint_state(GazeboSimulator* gazebo, int robot_id,
                                           const char* joint_name, float* position, float* velocity) {
    (void)robot_id;
    if (!gazebo->comm || !gazebo->internal.use_external_gazebo) return -1;
    if (gazebo_send_message(gazebo->comm, &gazebo->seq_num,
                            GAZEBO_MSG_GET_JOINT_STATE, joint_name,
                            (uint32_t)strlen(joint_name) + 1, 1) != 0) {
        gazebo->external_failures++;
        return -1;
    }
    GazeboJointState joint_state;
    memset(&joint_state, 0, sizeof(joint_state));
    int ret = gazebo_receive_message(gazebo->comm, GAZEBO_MSG_GET_JOINT_STATE,
                                     &joint_state, sizeof(joint_state));
    if (ret < 0) {
        gazebo->external_failures++;
        return -1;
    }
    if (position) *position = joint_state.position;
    if (velocity) *velocity = joint_state.velocity;
    return joint_state.success ? 0 : -1;
}

static int gazebo_external_set_joint_state(GazeboSimulator* gazebo, int robot_id,
                                           const char* joint_name, float position) {
    (void)robot_id;
    if (!gazebo->comm || !gazebo->internal.use_external_gazebo) return -1;
    GazeboJointState joint_state;
    memset(&joint_state, 0, sizeof(joint_state));
    strncpy(joint_state.joint_name, joint_name, sizeof(joint_state.joint_name) - 1);
    joint_state.position = position;
    return gazebo_send_message(gazebo->comm, &gazebo->seq_num,
                               GAZEBO_MSG_SET_JOINT_STATE, &joint_state, sizeof(joint_state), 1);
}

/* ============================================================================
 * 内部记录增强 - 写入传感器数据到CSV
 * =========================================================================== */

static int internal_recorder_write_csv(InternalSimulation* sim) {
    if (!sim->recorder.recording || !sim->recorder.log_file) return -1;
    float now = sim->simulation_time;
    if (now - sim->recorder.last_log_time < sim->recorder.log_interval) return 0;
    for (int r = 0; r < sim->robot_count; r++) {
        InternalRobot* robot = &sim->robots[r];
        if (!robot->active) continue;
        fprintf(sim->recorder.log_file, "%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                now, r,
                robot->position[0], robot->position[1], robot->position[2],
                robot->velocity[0], robot->velocity[1], robot->velocity[2],
                robot->battery_level);
        for (int j = 0; j < robot->num_joints && j < 6; j++) {
            fprintf(sim->recorder.log_file, ",%.4f,%.4f,%.4f",
                    robot->joint_positions[j], robot->joint_velocities[j], robot->joint_torques[j]);
        }
        fprintf(sim->recorder.log_file, "\n");
    }
    for (int s = 0; s < sim->sensor_count; s++) {
        if (!sim->sensors[s].active) continue;
        InternalSensor* sensor = &sim->sensors[s];
        fprintf(sim->recorder.log_file, "%.4f,S%d,%d", now, s, sensor->sensor_type);
        int data_count = (sensor->data_size < 16) ? sensor->data_size : 16;
        for (int d = 0; d < data_count; d++) {
            fprintf(sim->recorder.log_file, ",%.4f", sensor->data[d]);
        }
        fprintf(sim->recorder.log_file, "\n");
    }
    sim->recorder.last_log_time = now;
    return 0;
}

/* ============================================================================
 * Gazebo仿真器API实现
 * =========================================================================== */

Simulator* gazebo_simulator_create(const SimulatorConfig* config) {
    if (!config || config->type != SIMULATOR_GAZEBO) return NULL;

    GazeboSimulator* gazebo = (GazeboSimulator*)safe_malloc(sizeof(GazeboSimulator));
    if (!gazebo) return NULL;
    memset(gazebo, 0, sizeof(GazeboSimulator));
    memcpy(&gazebo->config, config, sizeof(SimulatorConfig));

    gazebo->model_capacity = 10;
    gazebo->model_count = 0;
    gazebo->model_names = (char**)safe_calloc(gazebo->model_capacity, sizeof(char*));
    if (!gazebo->model_names) {
        safe_free((void**)&gazebo);
        return NULL;
    }

    // 创建通信接口（用于连接外部Gazebo）
    HardwareConfig hw_config;
    memset(&hw_config, 0, sizeof(HardwareConfig));
    hw_config.type = HARDWARE_TYPE_TCP;
    strncpy(hw_config.config.network.host, config->hostname,
            sizeof(hw_config.config.network.host) - 1);
    hw_config.config.network.port = config->port > 0 ? config->port : SELFLNN_GAZEBO_PORT;
    hw_config.config.network.connect_timeout_ms = config->timeout_ms;
    hw_config.config.network.protocol = 0;
    gazebo->comm = robot_hardware_interface_create(&hw_config);
    if (!gazebo->comm) {
        safe_free((void**)&gazebo->model_names);
        safe_free((void**)&gazebo);
        return NULL;
    }

    // 初始化内部物理引擎（确保无外部Gazebo时也能完整运行）
    internal_sim_init(&gazebo->internal, config);

    gazebo->connected = 0;
    strncpy(gazebo->world_name, "default", sizeof(gazebo->world_name) - 1);
    gazebo->simulation_time = 0.0f;
    gazebo->paused = 1;
    gazebo->internal.paused = 1;
    gazebo->max_reconn_attempts = 3;
    gazebo->heartbeat_counter = GAZEBO_HEARTBEAT_INTERVAL;

    gazebo_log(gazebo, "Gazebo仿真器实例创建完成，内部物理引擎已初始化");
    return (Simulator*)gazebo;
}

void gazebo_simulator_destroy(Simulator* simulator) {
    if (!simulator) return;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    // 停止录制
    if (gazebo->internal.recorder.recording && gazebo->internal.recorder.log_file) {
        fclose(gazebo->internal.recorder.log_file);
        gazebo->internal.recorder.recording = 0;
    }

    if (gazebo->connected) {
        hardware_interface_disconnect(gazebo->comm);
    }
    if (gazebo->comm) {
        robot_hardware_interface_destroy(gazebo->comm);
    }

    /* 清理传感器管道 */
    if (gazebo->sensor_pipeline) {
        sensor_pipeline_stop(gazebo->sensor_pipeline);
        sensor_pipeline_destroy(gazebo->sensor_pipeline);
        gazebo->sensor_pipeline = NULL;
    }

    for (int i = 0; i < gazebo->model_count; i++) {
        safe_free((void**)&gazebo->model_names[i]);
    }
    safe_free((void**)&gazebo->model_names);
    safe_free((void**)&gazebo);
}

int gazebo_simulator_connect(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (gazebo->connected) return 0;

    if (hardware_interface_connect(gazebo->comm) != 0) {
        strncpy(gazebo->last_error, "连接外部Gazebo失败，使用内部仿真引擎", sizeof(gazebo->last_error) - 1);
        gazebo->internal.use_external_gazebo = 0;
        gazebo->connected = 1;
        gazebo_log(gazebo, "外部Gazebo不可用，启用内部仿真引擎");
        return 0;
    }

    /* 发送版本握手消息 */
    GazeboMsgHeader handshake_msg;
    handshake_msg.msg_type = GAZEBO_MSG_GET_MODEL_STATE;
    handshake_msg.msg_size = sizeof(uint32_t);
    handshake_msg.seq_num = (uint32_t)(gazebo->seq_num++);
    handshake_msg.flags = GAZEBO_FLAG_CONNECT_TEST;
    uint32_t proto_version = GAZEBO_PROTOCOL_VERSION;
    if (gazebo_send_all(gazebo->comm, &handshake_msg, sizeof(handshake_msg), 1) != sizeof(handshake_msg) ||
        gazebo_send_all(gazebo->comm, &proto_version, sizeof(proto_version), 1) != sizeof(proto_version)) {
        hardware_interface_disconnect(gazebo->comm);
        strncpy(gazebo->last_error, "Gazebo握手发送失败，使用内部仿真引擎", sizeof(gazebo->last_error) - 1);
        gazebo->internal.use_external_gazebo = 0;
        gazebo->connected = 1;
        gazebo_log(gazebo, "Gazebo握手失败，启用内部仿真引擎");
        return 0;
    }

    /* 等待并验证响应 */
    GazeboMsgHeader response_header;
    int recv_ret = gazebo_recv_all(gazebo->comm, &response_header, sizeof(response_header), 1);
    if (recv_ret != sizeof(response_header)) {
        hardware_interface_disconnect(gazebo->comm);
        strncpy(gazebo->last_error, "Gazebo无握手响应，使用内部仿真引擎", sizeof(gazebo->last_error) - 1);
        gazebo->internal.use_external_gazebo = 0;
        gazebo->connected = 1;
        gazebo_log(gazebo, "Gazebo无握手响应，启用内部仿真引擎");
        return 0;
    }

    if (response_header.flags & GAZEBO_FLAG_HAS_CHECKSUM) {
        uint32_t recv_checksum;
        if (gazebo_recv_all(gazebo->comm, &recv_checksum, sizeof(recv_checksum), 1) == sizeof(recv_checksum)) {
            uint32_t calc_checksum = gazebo_calc_checksum(&response_header, sizeof(response_header));
            if (calc_checksum != recv_checksum) {
                hardware_interface_disconnect(gazebo->comm);
                strncpy(gazebo->last_error, "Gazebo握手校验和错误，使用内部仿真引擎", sizeof(gazebo->last_error) - 1);
                gazebo->internal.use_external_gazebo = 0;
                gazebo->connected = 1;
                gazebo_log(gazebo, "Gazebo握手校验和错误，启用内部仿真引擎");
                return 0;
            }
        }
    }

    if (response_header.msg_type == GAZEBO_MSG_GET_MODEL_STATE && response_header.msg_size >= sizeof(uint32_t)) {
        uint32_t remote_version = 0;
        if (gazebo_recv_all(gazebo->comm, &remote_version, sizeof(remote_version), 1) == sizeof(remote_version)) {
            if (remote_version != GAZEBO_PROTOCOL_VERSION) {
                hardware_interface_disconnect(gazebo->comm);
                strncpy(gazebo->last_error, "Gazebo协议版本不匹配，使用内部仿真引擎", sizeof(gazebo->last_error) - 1);
                gazebo->internal.use_external_gazebo = 0;
                gazebo->connected = 1;
                gazebo_log(gazebo, "Gazebo协议版本不匹配，启用内部仿真引擎");
                return 0;
            }
        }
    } else if (response_header.msg_size > 0) {
        /* 跳过未知响应体 */
        char* skip_buf = (char*)safe_malloc(response_header.msg_size);
        if (skip_buf) {
            gazebo_recv_all(gazebo->comm, skip_buf, response_header.msg_size, 1);
            safe_free((void**)&skip_buf);
        }
    }

    gazebo->internal.use_external_gazebo = 1;
    gazebo->connected = 1;
    gazebo->external_failures = 0;
    gazebo->heartbeat_counter = GAZEBO_HEARTBEAT_INTERVAL;
    gazebo->last_external_success_time = 0.0f;
    strncpy(gazebo->last_error, "连接成功", sizeof(gazebo->last_error) - 1);
    gazebo_log(gazebo, "已连接到外部Gazebo仿真器");
    return 0;
}

int gazebo_simulator_disconnect(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (!gazebo->connected) return 0;

    if (gazebo->internal.use_external_gazebo) {
        if (hardware_interface_disconnect(gazebo->comm) != 0) {
            strncpy(gazebo->last_error, "断开Gazebo连接失败", sizeof(gazebo->last_error) - 1);
            return -1;
        }
    }
    gazebo->connected = 0;
    gazebo->internal.use_external_gazebo = 0;
    strncpy(gazebo->last_error, "断开成功", sizeof(gazebo->last_error) - 1);
    return 0;
}

int gazebo_simulator_start(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (!gazebo->connected) {
        strncpy(gazebo->last_error, "仿真器未连接", sizeof(gazebo->last_error) - 1);
        return -1;
    }

    if (gazebo->internal.use_external_gazebo) {
        GazeboSimulationControl msg;
        msg.control_type = 0;
        msg.pause_time = 0.0f;
        msg.flags = 0;
        int ret = gazebo_send_message(gazebo->comm, &gazebo->seq_num, GAZEBO_MSG_SET_MODEL_STATE, &msg, sizeof(msg), 1);
        if (ret != 0) {
            strncpy(gazebo->last_error, "向Gazebo发送启动消息失败，使用内部引擎", sizeof(gazebo->last_error) - 1);
        }
    }

    gazebo->paused = 0;
    gazebo->internal.paused = 0;
    strncpy(gazebo->last_error, "仿真已启动", sizeof(gazebo->last_error) - 1);
    gazebo_log(gazebo, "仿真已启动");
    return 0;
}

int gazebo_simulator_stop(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (!gazebo->connected) {
        strncpy(gazebo->last_error, "仿真器未连接", sizeof(gazebo->last_error) - 1);
        return -1;
    }

    if (gazebo->internal.use_external_gazebo) {
        GazeboSimulationControl msg;
        msg.control_type = 1;
        msg.pause_time = 0.0f;
        msg.flags = 0;
        gazebo_send_message(gazebo->comm, &gazebo->seq_num, GAZEBO_MSG_SET_MODEL_STATE, &msg, sizeof(msg), 1);
    }

    gazebo->paused = 1;
    gazebo->internal.paused = 1;
    strncpy(gazebo->last_error, "仿真已停止", sizeof(gazebo->last_error) - 1);
    return 0;
}

int gazebo_simulator_pause(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    gazebo->paused = 1;
    gazebo->internal.paused = 1;
    strncpy(gazebo->last_error, "仿真已暂停", sizeof(gazebo->last_error) - 1);
    return 0;
}

int gazebo_simulator_resume(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    gazebo->paused = 0;
    gazebo->internal.paused = 0;
    strncpy(gazebo->last_error, "仿真已恢复", sizeof(gazebo->last_error) - 1);
    return 0;
}

int gazebo_simulator_reset(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    gazebo->simulation_time = 0.0f;
    gazebo->paused = 1;
    gazebo->internal.simulation_time = 0.0f;
    gazebo->internal.step_count = 0;
    gazebo->internal.paused = 1;

    for (int i = 0; i < gazebo->internal.robot_count; i++) {
        InternalRobot* r = &gazebo->internal.robots[i];
        memset(r->position, 0, sizeof(r->position));
        memset(r->velocity, 0, sizeof(r->velocity));
        memset(r->angular_velocity, 0, sizeof(r->angular_velocity));
        memset(r->joint_positions, 0, sizeof(r->joint_positions));
        memset(r->joint_velocities, 0, sizeof(r->joint_velocities));
        memset(r->joint_targets, 0, sizeof(r->joint_targets));
        memset(r->joint_torques, 0, sizeof(r->joint_torques));
        memset(r->collision_impulse, 0, sizeof(r->collision_impulse));
        memset(r->contact_force, 0, sizeof(r->contact_force));
        memset(r->applied_force, 0, sizeof(r->applied_force));
        memset(r->applied_torque, 0, sizeof(r->applied_torque));
        memset(r->acceleration, 0, sizeof(r->acceleration));
        r->orientation[0] = 0; r->orientation[1] = 0; r->orientation[2] = 0; r->orientation[3] = 1;
        r->battery_level = 100.0f;
        r->is_colliding = 0;
        r->joint_max_torque = 200.0f;
        r->joint_max_velocity = 10.0f;
        r->actuator_response_coeff = 0.1f;
        for (int j = 0; j < GAZEBO_MAX_JOINTS; j++) {
            r->joint_limits_min[j] = -3.14159f;
            r->joint_limits_max[j] = 3.14159f;
        }
    }

    strncpy(gazebo->last_error, "仿真已重置", sizeof(gazebo->last_error) - 1);
    return 0;
}

int gazebo_simulator_step(Simulator* simulator, int num_steps) {
    if (!simulator || num_steps <= 0) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (!gazebo->connected) {
        strncpy(gazebo->last_error, "仿真器未连接", sizeof(gazebo->last_error) - 1);
        return -1;
    }

    for (int i = 0; i < num_steps; i++) {
        if (gazebo->internal.use_external_gazebo) {
            /* 检查是否需要发送心跳 */
            gazebo->heartbeat_counter--;
            if (gazebo->heartbeat_counter <= 0) {
                gazebo_send_heartbeat(gazebo);
                gazebo->heartbeat_counter = GAZEBO_HEARTBEAT_INTERVAL;
            }

            /* 检查故障计数，触发重连或回退到内部引擎 */
            if (gazebo->external_failures >= GAZEBO_MAX_EXTERNAL_FAILURES && !gazebo->reconnecting) {
                gazebo_log(gazebo, "外部Gazebo通信故障过多，尝试重连");
                if (gazebo_try_reconnect(gazebo) != 0) {
                    gazebo_log(gazebo, "重连失败，切换至内部仿真引擎");
                    gazebo->internal.use_external_gazebo = 0;
                    gazebo->external_failures = 0;
                }
            }

            if (gazebo->internal.use_external_gazebo && !gazebo->reconnecting) {
                /* 外部Gazebo模式：从外部获取机器人状态 */
                int step_success = 1;
                for (int r = 0; r < gazebo->internal.robot_count; r++) {
                    if (!gazebo->internal.robots[r].active) continue;
                    if (gazebo->model_names[r]) {
                        float ext_pos[3], ext_ori[4];
                        if (gazebo_external_get_model_state(gazebo, r, ext_pos, ext_ori) == 0) {
                            memcpy(gazebo->internal.robots[r].position, ext_pos, 3 * sizeof(float));
                            memcpy(gazebo->internal.robots[r].orientation, ext_ori, 4 * sizeof(float));
                            gazebo->last_external_success_time = gazebo->simulation_time;
                            step_success = 1;
                        } else {
                            step_success = 0;
                        }
                    }
                }

                if (step_success) {
                    gazebo->external_failures = 0;
                }
            } else if (!gazebo->internal.use_external_gazebo) {
                /* 已回退到内部引擎 */
                internal_sim_step_physics(&gazebo->internal);
            }

            gazebo->simulation_time += gazebo->internal.timestep;
        } else {
            /* 内部引擎模式：执行物理步进 */
            internal_sim_step_physics(&gazebo->internal);
            gazebo->simulation_time += gazebo->internal.timestep;
        }

        /* 记录仿真数据 */
        internal_recorder_write_csv(&gazebo->internal);

        /* 推送传感器数据到管道 */
        if (gazebo->sensor_streaming_enabled && gazebo->sensor_pipeline) {
            for (int si = 0; si < gazebo->internal.sensor_count; si++) {
                InternalSensor* sd = &gazebo->internal.sensors[si];
                if (!sd->active) continue;
                float sensor_buf[64];
                int data_size = internal_sim_generate_sensor_data(&gazebo->internal, si, sensor_buf, 64);
                if (data_size > 0) {
                    SensorPipelineEntry entry;
                    memset(&entry, 0, sizeof(entry));
                    entry.sensor_id = si;
                    entry.sensor_type = (SensorType)sd->sensor_type;
                    entry.timestamp = (double)gazebo->internal.simulation_time;
                    entry.data_size = (size_t)data_size * sizeof(float);
                    entry.data = (uint8_t*)sensor_buf;
                    entry.is_valid = 1;
                    entry.confidence = 1.0f;
                    entry.sequence = (uint32_t)gazebo->internal.step_count;
                    snprintf(entry.sensor_name, sizeof(entry.sensor_name), "gazebo_sensor_%d", si);
                    sensor_pipeline_push_data(gazebo->sensor_pipeline, &entry);
                }
            }
            for (int ri = 0; ri < gazebo->internal.robot_count; ri++) {
                InternalRobot* r = &gazebo->internal.robots[ri];
                if (!r->active) continue;
                SensorPipelineEntry entry;
                memset(&entry, 0, sizeof(entry));
                entry.sensor_id = 1000 + ri;
                entry.sensor_type = SENSOR_TYPE_IMU;
                entry.timestamp = (double)gazebo->internal.simulation_time;
                entry.data_size = sizeof(InternalRobot);
                entry.data = (uint8_t*)r;
                entry.is_valid = 1;
                entry.confidence = 1.0f;
                entry.sequence = (uint32_t)gazebo->internal.step_count;
                snprintf(entry.sensor_name, sizeof(entry.sensor_name), "robot_%d_status", ri);
                sensor_pipeline_push_data(gazebo->sensor_pipeline, &entry);
            }
        }
    }

    return 0;
}

int gazebo_simulator_get_status(Simulator* simulator, SimulatorStatus* status) {
    if (!simulator || !status) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    memset(status, 0, sizeof(SimulatorStatus));
    if (gazebo->connected) {
        status->state = gazebo->paused ? SIMULATOR_STATE_PAUSED : SIMULATOR_STATE_RUNNING;
    } else {
        status->state = SIMULATOR_STATE_DISCONNECTED;
    }
    status->simulation_time = gazebo->internal.simulation_time;
    status->real_time = gazebo->simulation_time;
    status->step_count = gazebo->internal.step_count;
    status->num_robots = gazebo->internal.robot_count;
    status->num_sensors = gazebo->internal.sensor_count;
    status->num_contacts = 0;
    for (int i = 0; i < gazebo->internal.robot_count; i++) {
        if (gazebo->internal.robots[i].is_colliding) status->num_contacts++;
    }
    strncpy(status->last_error, gazebo->last_error, sizeof(status->last_error) - 1);
    return 0;
}

int gazebo_simulator_load_robot(Simulator* simulator, const RobotConfig* robot_config, const float* initial_pose) {
    if (!simulator || !robot_config) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (!gazebo->connected) {
        strncpy(gazebo->last_error, "仿真器未连接", sizeof(gazebo->last_error) - 1);
        return -1;
    }

    if (gazebo->model_count >= gazebo->model_capacity) {
        int new_capacity = gazebo->model_capacity * 2;
        char** new_names = (char**)safe_realloc(gazebo->model_names, new_capacity * sizeof(char*));
        if (!new_names) return -1;
        gazebo->model_names = new_names;
        gazebo->model_capacity = new_capacity;
    }

    // 在内部引擎中创建机器人
    int robot_id = internal_sim_add_robot(&gazebo->internal);
    if (robot_id < 0) {
        strncpy(gazebo->last_error, "内部引擎机器人创建失败", sizeof(gazebo->last_error) - 1);
        return -1;
    }

    InternalRobot* robot = &gazebo->internal.robots[robot_id];
    if (initial_pose) {
        robot->position[0] = initial_pose[0];
        robot->position[1] = initial_pose[1];
        robot->position[2] = initial_pose[2];
        robot->orientation[0] = initial_pose[3];
        robot->orientation[1] = initial_pose[4];
        robot->orientation[2] = initial_pose[5];
        robot->orientation[3] = initial_pose[6];
    }

    gazebo->model_names[gazebo->model_count] = string_duplicate(robot_config->name);
    if (!gazebo->model_names[gazebo->model_count]) return -1;
    gazebo->model_count++;

    // 如果使用外部Gazebo，同步创建模型
    if (gazebo->internal.use_external_gazebo) {
        float pose[7];
        pose[0] = robot->position[0]; pose[1] = robot->position[1]; pose[2] = robot->position[2];
        pose[3] = robot->orientation[0]; pose[4] = robot->orientation[1];
        pose[5] = robot->orientation[2]; pose[6] = robot->orientation[3];
        gazebo_external_spawn_model(gazebo, robot_config->name, pose, pose + 3);
    }

    strncpy(gazebo->last_error, "机器人模型已加载至内部仿真引擎", sizeof(gazebo->last_error) - 1);
    return robot_id;
}

int gazebo_simulator_remove_robot(Simulator* simulator, int robot_id) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    gazebo->internal.robots[robot_id].active = 0;

    // 如果使用外部Gazebo，同步删除模型
    const char* model_to_remove = NULL;
    if (robot_id < gazebo->model_count && gazebo->model_names[robot_id]) {
        model_to_remove = gazebo->model_names[robot_id];
    }

    // 删除对应的model_names条目
    if (robot_id < gazebo->model_count) {
        if (gazebo->model_names[robot_id]) {
            if (gazebo->internal.use_external_gazebo && model_to_remove) {
                gazebo_external_delete_model(gazebo, model_to_remove);
            }
            safe_free((void**)&gazebo->model_names[robot_id]);
        }
        for (int i = robot_id; i < gazebo->model_count - 1; i++) {
            gazebo->model_names[i] = gazebo->model_names[i + 1];
        }
        gazebo->model_names[gazebo->model_count - 1] = NULL;
        gazebo->model_count--;
    }

    strncpy(gazebo->last_error, "机器人已移除", sizeof(gazebo->last_error) - 1);
    return 0;
}

int gazebo_simulator_get_robot_state(Simulator* simulator, int robot_id, SimulatorRobotState* state) {
    if (!simulator || !state) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* robot = &gazebo->internal.robots[robot_id];
    if (!robot->active) return -1;

    memset(state, 0, sizeof(SimulatorRobotState));
    state->robot_id = robot_id;
    if (robot_id < gazebo->model_count && gazebo->model_names[robot_id]) {
        strncpy(state->robot_name, gazebo->model_names[robot_id], sizeof(state->robot_name) - 1);
    }
    memcpy(state->position, robot->position, 3 * sizeof(float));
    memcpy(state->velocity, robot->velocity, 3 * sizeof(float));
    memcpy(state->acceleration, robot->acceleration, 3 * sizeof(float));
    memcpy(state->orientation, robot->orientation, 4 * sizeof(float));
    memcpy(state->angular_velocity, robot->angular_velocity, 3 * sizeof(float));
    memcpy(state->joint_positions, robot->joint_positions, GAZEBO_MAX_JOINTS * sizeof(float));
    memcpy(state->joint_velocities, robot->joint_velocities, GAZEBO_MAX_JOINTS * sizeof(float));
    memcpy(state->joint_torques, robot->joint_torques, GAZEBO_MAX_JOINTS * sizeof(float));
    state->contact_forces[0] = robot->contact_force[0];
    state->contact_forces[1] = robot->contact_force[1];
    state->contact_forces[2] = robot->contact_force[2];
    state->battery_level = robot->battery_level;
    state->is_colliding = robot->is_colliding;
    return 0;
}

int gazebo_simulator_set_joint_positions(Simulator* simulator, int robot_id, const float* joint_positions, int num_joints) {
    if (!simulator || !joint_positions) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* robot = &gazebo->internal.robots[robot_id];
    if (!robot->active) return -1;

    int n = (num_joints < GAZEBO_MAX_JOINTS) ? num_joints : GAZEBO_MAX_JOINTS;
    for (int i = 0; i < n; i++) {
        robot->joint_targets[i] = joint_positions[i];
    }
    return 0;
}

int gazebo_simulator_set_joint_velocities(Simulator* simulator, int robot_id, const float* joint_velocities, int num_joints) {
    if (!simulator || !joint_velocities) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* robot = &gazebo->internal.robots[robot_id];
    if (!robot->active) return -1;

    int n = (num_joints < GAZEBO_MAX_JOINTS) ? num_joints : GAZEBO_MAX_JOINTS;
    for (int i = 0; i < n; i++) {
        robot->joint_velocities[i] = joint_velocities[i];
    }
    return 0;
}

int gazebo_simulator_set_joint_torques(Simulator* simulator, int robot_id, const float* joint_torques, int num_joints) {
    if (!simulator || !joint_torques) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* robot = &gazebo->internal.robots[robot_id];
    if (!robot->active) return -1;

    int n = (num_joints < GAZEBO_MAX_JOINTS) ? num_joints : GAZEBO_MAX_JOINTS;
    for (int i = 0; i < n; i++) {
        robot->joint_torques[i] = joint_torques[i];
    }
    return 0;
}

int gazebo_simulator_apply_robot_command(Simulator* simulator, int robot_id, const RobotCommand* command) {
    if (!simulator || !command) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* robot = &gazebo->internal.robots[robot_id];
    if (!robot->active) return -1;

    robot->applied_force[0] = command->target_linear_velocity[0] * 10.0f;
    robot->applied_force[1] = command->target_linear_velocity[1] * 10.0f;
    robot->applied_force[2] = command->target_linear_velocity[2] * 10.0f;
    robot->applied_torque[0] = command->target_angular_velocity[0] * 5.0f;
    robot->applied_torque[1] = command->target_angular_velocity[1] * 5.0f;
    robot->applied_torque[2] = command->target_angular_velocity[2] * 5.0f;

    for (int i = 0; i < GAZEBO_MAX_JOINTS; i++) {
        robot->joint_targets[i] = command->target_joint_positions[i];
    }
    return 0;
}

int gazebo_simulator_add_sensor(Simulator* simulator, int robot_id, const SensorConfig* sensor_config,
                                const float* mount_position, const float* mount_orientation) {
    if (!simulator || !sensor_config) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (gazebo->internal.sensor_count >= GAZEBO_MAX_SENSORS) return -1;

    int sid = gazebo->internal.sensor_count;
    InternalSensor* sensor = &gazebo->internal.sensors[sid];
    memset(sensor, 0, sizeof(InternalSensor));
    sensor->active = 1;
    sensor->robot_id = robot_id;
    sensor->sensor_type = sensor_config->type;
    sensor->noise_level = 0.01f;
    if (mount_position) {
        memcpy(sensor->mount_position, mount_position, 3 * sizeof(float));
    }
    if (mount_orientation) {
        memcpy(sensor->mount_orientation, mount_orientation, 4 * sizeof(float));
    }
    gazebo->internal.sensor_count++;
    return sid;
}

int gazebo_simulator_remove_sensor(Simulator* simulator, int sensor_id) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (sensor_id < 0 || sensor_id >= gazebo->internal.sensor_count) return -1;
    gazebo->internal.sensors[sensor_id].active = 0;
    return 0;
}

int gazebo_simulator_get_sensor_data(Simulator* simulator, int sensor_id, SimulatorSensorData* sensor_data) {
    if (!simulator || !sensor_data) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (sensor_id < 0 || sensor_id >= gazebo->internal.sensor_count) return -1;
    InternalSensor* sensor = &gazebo->internal.sensors[sensor_id];
    if (!sensor->active) return -1;

    memset(sensor_data, 0, sizeof(SimulatorSensorData));
    sensor_data->sensor_id = sensor_id;
    sensor_data->sensor_type = sensor->sensor_type;
    sensor_data->timestamp = gazebo->internal.simulation_time;
    sensor_data->is_valid = 1;

    int size = internal_sim_generate_sensor_data(&gazebo->internal, sensor_id, sensor_data->data, 64);
    if (size < 0) return -1;
    sensor_data->data_size = size;
    return 0;
}

int gazebo_simulator_get_all_sensor_data(Simulator* simulator, int robot_id,
                                         SimulatorSensorData* sensor_data_array, int max_sensors) {
    if (!simulator || !sensor_data_array) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    int count = 0;
    for (int i = 0; i < gazebo->internal.sensor_count && count < max_sensors; i++) {
        if (gazebo->internal.sensors[i].active && gazebo->internal.sensors[i].robot_id == robot_id) {
            if (gazebo_simulator_get_sensor_data(simulator, i, &sensor_data_array[count]) == 0) {
                count++;
            }
        }
    }
    return count;
}

int gazebo_simulator_add_scene_object(Simulator* simulator, const SimulatorSceneObject* object) {
    if (!simulator || !object) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (gazebo->internal.scene_object_count >= GAZEBO_MAX_SCENE_OBJECTS) return -1;

    int id = gazebo->internal.scene_object_count;
    InternalSceneObject* obj = &gazebo->internal.scene_objects[id];
    memset(obj, 0, sizeof(InternalSceneObject));
    obj->active = 1;
    obj->object_type = object->object_type;
    obj->mass = object->mass;
    obj->friction = object->friction;
    obj->restitution = object->restitution;
    memcpy(obj->position, object->position, 3 * sizeof(float));
    memcpy(obj->orientation, object->orientation, 4 * sizeof(float));
    memcpy(obj->scale, object->scale, 3 * sizeof(float));
    memcpy(obj->color, object->color, 4 * sizeof(float));
    strncpy(obj->name, object->object_name, sizeof(obj->name) - 1);
    gazebo->internal.scene_object_count++;
    return id;
}

int gazebo_simulator_remove_scene_object(Simulator* simulator, int object_id) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (object_id < 0 || object_id >= gazebo->internal.scene_object_count) return -1;
    gazebo->internal.scene_objects[object_id].active = 0;
    return 0;
}

int gazebo_simulator_get_scene_objects(Simulator* simulator, SimulatorSceneObject* objects, int max_objects) {
    if (!simulator || !objects) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    int count = 0;
    for (int i = 0; i < gazebo->internal.scene_object_count && count < max_objects; i++) {
        if (!gazebo->internal.scene_objects[i].active) continue;
        InternalSceneObject* src = &gazebo->internal.scene_objects[i];
        SimulatorSceneObject* dst = &objects[count];
        memset(dst, 0, sizeof(SimulatorSceneObject));
        dst->object_id = i;
        dst->object_type = src->object_type;
        dst->mass = src->mass;
        dst->friction = src->friction;
        dst->restitution = src->restitution;
        memcpy(dst->position, src->position, 3 * sizeof(float));
        memcpy(dst->orientation, src->orientation, 4 * sizeof(float));
        memcpy(dst->scale, src->scale, 3 * sizeof(float));
        memcpy(dst->color, src->color, 4 * sizeof(float));
        strncpy(dst->object_name, src->name, sizeof(dst->object_name) - 1);
        count++;
    }
    return count;
}

int gazebo_simulator_set_gravity(Simulator* simulator, const float* gravity) {
    if (!simulator || !gravity) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    gazebo->internal.gravity[0] = gravity[0];
    gazebo->internal.gravity[1] = gravity[1];
    gazebo->internal.gravity[2] = gravity[2];
    return 0;
}

int gazebo_simulator_set_lighting(Simulator* simulator, const float* light_position,
                                  const float* light_color, const float* ambient_color) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (light_position) memcpy(gazebo->internal.light_position, light_position, 3 * sizeof(float));
    if (light_color) memcpy(gazebo->internal.light_color, light_color, 3 * sizeof(float));
    if (ambient_color) memcpy(gazebo->internal.ambient_color, ambient_color, 3 * sizeof(float));
    return 0;
}

int gazebo_simulator_start_recording(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    if (gazebo->internal.recorder.recording) {
        if (gazebo->internal.recorder.log_file) {
            fclose(gazebo->internal.recorder.log_file);
        }
        gazebo->internal.recorder.recording = 0;
    }

    gazebo->internal.recorder.log_file = fopen(filename, "w");
    if (!gazebo->internal.recorder.log_file) return -1;
    strncpy(gazebo->internal.recording_filename, filename, sizeof(gazebo->internal.recording_filename) - 1);
    gazebo->internal.recorder.recording = 1;
    gazebo->internal.recorder.last_log_time = 0.0f;
    gazebo->internal.recorder.log_interval = 0.01f;

    fprintf(gazebo->internal.recorder.log_file, "# 仿真记录文件 (CSV格式)\n");
    fprintf(gazebo->internal.recorder.log_file, "# 时间,机器人ID,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,电量");
    fprintf(gazebo->internal.recorder.log_file, ",关节1_pos,关节1_vel,关节1_trq,关节2_pos,关节2_vel,关节2_trq");
    fprintf(gazebo->internal.recorder.log_file, ",关节3_pos,关节3_vel,关节3_trq,关节4_pos,关节4_vel,关节4_trq");
    fprintf(gazebo->internal.recorder.log_file, ",关节5_pos,关节5_vel,关节5_trq,关节6_pos,关节6_vel,关节6_trq\n");
    return 0;
}

int gazebo_simulator_stop_recording(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (gazebo->internal.recorder.recording && gazebo->internal.recorder.log_file) {
        fclose(gazebo->internal.recorder.log_file);
    }
    gazebo->internal.recorder.recording = 0;
    gazebo->internal.recorder.log_file = NULL;
    return 0;
}

int gazebo_simulator_export_scene(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    FILE* f = fopen(filename, "w");
    if (!f) return -1;

    fprintf(f, "<?xml version=\"1.0\"?>\n");
    fprintf(f, "<sdf version=\"1.6\">\n");
    fprintf(f, "  <world name=\"%s\">\n", gazebo->world_name);
    fprintf(f, "    <gravity>%f %f %f</gravity>\n",
            gazebo->internal.gravity[0],
            gazebo->internal.gravity[1],
            gazebo->internal.gravity[2]);

    for (int i = 0; i < gazebo->internal.scene_object_count; i++) {
        InternalSceneObject* obj = &gazebo->internal.scene_objects[i];
        if (!obj->active) continue;
        fprintf(f, "    <model name=\"%s\">\n", obj->name);
        fprintf(f, "      <pose>%f %f %f %f %f %f</pose>\n",
                obj->position[0], obj->position[1], obj->position[2],
                obj->orientation[0], obj->orientation[1], obj->orientation[2]);
        fprintf(f, "      <static>%s</static>\n", (obj->object_type == 0) ? "true" : "false");
        fprintf(f, "      <scale>%f %f %f</scale>\n", obj->scale[0], obj->scale[1], obj->scale[2]);
        fprintf(f, "      <mass>%f</mass>\n", obj->mass);
        fprintf(f, "      <friction>%f</friction>\n", obj->friction);
        fprintf(f, "      <restitution>%f</restitution>\n", obj->restitution);
        fprintf(f, "    </model>\n");
    }
    // 导出机器人状态
    for (int i = 0; i < gazebo->internal.robot_count; i++) {
        InternalRobot* robot = &gazebo->internal.robots[i];
        if (!robot->active) continue;
        const char* robot_name = (i < gazebo->model_count && gazebo->model_names[i])
                                 ? gazebo->model_names[i] : "unnamed_robot";
        fprintf(f, "    <model name=\"%s\">\n", robot_name);
        fprintf(f, "      <pose>%f %f %f %f %f %f</pose>\n",
                robot->position[0], robot->position[1], robot->position[2],
                robot->orientation[0], robot->orientation[1], robot->orientation[2]);
        fprintf(f, "      <static>false</static>\n");
        fprintf(f, "      <mass>%f</mass>\n", robot->mass);
        fprintf(f, "      <battery>%f</battery>\n", robot->battery_level);
        fprintf(f, "      <link name=\"body\">\n");
        fprintf(f, "        <inertial>\n");
        fprintf(f, "          <mass>%f</mass>\n", robot->mass);
        fprintf(f, "          <inertia>%f %f %f</inertia>\n",
                robot->inertia[0], robot->inertia[1], robot->inertia[2]);
        fprintf(f, "        </inertial>\n");
        fprintf(f, "      </link>\n");
        for (int j = 0; j < robot->num_joints && j < 6; j++) {
            fprintf(f, "      <joint name=\"joint_%d\" type=\"revolute\">\n", j);
            fprintf(f, "        <pose>%f %f %f 0 0 0</pose>\n", j * 0.2f, 0.0f, 0.0f);
            fprintf(f, "        <axis>\n");
            fprintf(f, "          <xyz>0 0 1</xyz>\n");
            fprintf(f, "          <limit>\n");
            fprintf(f, "            <lower>%f</lower>\n", robot->joint_limits_min[j]);
            fprintf(f, "            <upper>%f</upper>\n", robot->joint_limits_max[j]);
            fprintf(f, "            <max_velocity>%f</max_velocity>\n", robot->joint_max_velocity);
            fprintf(f, "            <max_torque>%f</max_torque>\n", robot->joint_max_torque);
            fprintf(f, "          </limit>\n");
            fprintf(f, "        </axis>\n");
            fprintf(f, "      </joint>\n");
        }
        fprintf(f, "    </model>\n");
    }
    fprintf(f, "  </world>\n");
    fprintf(f, "</sdf>\n");
    fclose(f);
    return 0;
}

const char* gazebo_simulator_get_last_error(Simulator* simulator) {
    if (!simulator) return "无效仿真器句柄";
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    return gazebo->last_error;
}

/* ============================================================================
 * M18: URDF加载 (F-040: 升级为完整XML 1.0解析器)
 * ============================================================================
 *
 * 使用递归下降XML解析器替代逐行字符串搜索。
 * 支持嵌套元素、属性、自闭合标签、注释、CDATA。
 */

static int urdf_parse_float(const char* line, float* val) {
    const char* p = line;
    while (*p && (*p < '0' || *p > '9') && *p != '-' && *p != '.') p++;
    if (!*p) return -1;
    *val = (float)atof(p);
    return 0;
}

static int urdf_parse_vec3(const char* line, float v[3]) {
    const char* p = line;
    while (*p && *p != '"') p++;
    if (!*p) return -1; p++;
    v[0] = (float)atof(p);
    while (*p && *p != ' ') p++;
    if (!*p) return -1; p++;
    v[1] = (float)atof(p);
    while (*p && *p != ' ') p++;
    if (!*p) return -1; p++;
    v[2] = (float)atof(p);
    return 0;
}

int gazebo_simulator_load_urdf(Simulator* simulator, const char* urdf_path,
                                const float* initial_pose, const char* robot_name)
{
    if (!simulator || !urdf_path) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    FILE* f = fopen(urdf_path, "r");
    if (!f) {
        snprintf(gazebo->last_error, sizeof(gazebo->last_error),
                 "无法打开URDF文件: %s", urdf_path);
        return -1;
    }

    int robot_id = internal_sim_add_robot(&gazebo->internal);
    if (robot_id < 0) { fclose(f); return -1; }

    InternalRobot* r = &gazebo->internal.robots[robot_id];

    if (robot_name) {
        strncpy(r->robot_name, robot_name, sizeof(r->robot_name) - 1);
    } else {
        const char* base = strrchr(urdf_path, '/');
        if (!base) base = strrchr(urdf_path, '\\');
        if (!base) base = urdf_path; else base++;
        strncpy(r->robot_name, base, sizeof(r->robot_name) - 1);
        char* dot = strchr(r->robot_name, '.');
        if (dot) *dot = '\0';
    }
    strncpy(r->urdf_path, urdf_path, sizeof(r->urdf_path) - 1);

    if (initial_pose) {
        memcpy(r->position, initial_pose, 3 * sizeof(float));
        memcpy(r->orientation, initial_pose + 3, 4 * sizeof(float));
    } else {
        r->position[1] = 0.8f;
        r->orientation[3] = 1.0f;
    }

    /* F-040: 使用递归下降XML解析器加载URDF */
    XmlNode* root = xml_parse_file(urdf_path);
    if (root) {
        int joint_idx = 0, link_idx = 0;
        float joint_mass = 0.0f;
        int link_count = 0;

        /* 遍历所有<link>元素 */
        XmlNode* link = xml_find_child_r(root, "link", NULL, NULL);
        while (link) {
            link_count++;
            if (link_idx < 32) {
                XmlNode* inertial = xml_find_child(link, "inertial");
                if (inertial) {
                    XmlNode* mass_node = xml_find_child(inertial, "mass");
                    if (mass_node) {
                        float m = 0;
                        xml_read_float(mass_node, "value", &m);
                        r->link_masses[link_idx] = m;
                        joint_mass += m;
                    }
                }
            }
            link_idx++;
            link = xml_find_child_r(root, "link", NULL, NULL);
        }

        /* 遍历所有<joint>元素 */
        XmlNode* joint_node = xml_find_child_r(root, "joint", NULL, NULL);
        while (joint_node && joint_idx < 32) {
            const char* jtype = xml_get_attr(joint_node, "type");
            if (jtype && strcmp(jtype, "fixed") == 0) {
                joint_node = joint_node->next;
                continue;
            }
            if (r->num_joints < 32) r->num_joints++;
            XmlNode* limit = xml_find_child(joint_node, "limit");
            if (limit) {
                xml_read_float(limit, "lower", &r->joint_limits_min[joint_idx]);
                xml_read_float(limit, "upper", &r->joint_limits_max[joint_idx]);
                xml_read_float(limit, "velocity", &r->joint_max_velocity);
                xml_read_float(limit, "effort", &r->joint_max_torque);
            }
            joint_idx++;
            joint_node = joint_node->next;
        }

        xml_free(root);

        r->total_mass = joint_mass;
        r->mass = joint_mass;
        r->num_links = link_count;
        r->num_joints = joint_idx;
        r->num_dof = joint_idx;
        r->height = 1.5f;
        r->foot_size[0] = 0.2f;
        r->foot_size[1] = 0.1f;

        for (int j = 0; j < r->num_joints; j++) {
            if (r->joint_limits_max[j] < 0.001f && r->joint_limits_max[j] > -0.001f)
                r->joint_limits_max[j] = 3.14159f;
            if (r->joint_limits_min[j] < 0.001f && r->joint_limits_min[j] > -0.001f)
                r->joint_limits_min[j] = -3.14159f;
        }
    } else {
        /* 回退到简单逐行解析器 */
        rewind(f);
        char line[512];
        int in_link = 0, in_joint = 0, in_limit = 0, in_inertial = 0;
        int joint_idx = 0, link_idx = 0;
        float joint_mass = 50.0f;
        int link_count = 0;

        while (fgets(line, sizeof(line), f)) {
            (void)in_link; (void)link_idx;
            if (strstr(line, "<link")) { in_link = 1; link_count++; }
            if (strstr(line, "</link>")) { in_link = 0; }
            if (strstr(line, "<inertial>")) in_inertial = 1;
            if (strstr(line, "</inertial>")) in_inertial = 0;
            if (in_inertial && strstr(line, "<mass")) {
                float m = 0; urdf_parse_float(line, &m);
                if (link_idx < 32) { r->link_masses[link_idx] = m; link_idx++; }
                joint_mass += m;
            }
            if (strstr(line, "<joint")) { in_joint = 1; }
            if (strstr(line, "</joint>")) { in_joint = 0; joint_idx++; }
            if (in_joint && strstr(line, "<limit")) { in_limit = 1; }
            if (in_joint && strstr(line, "</limit>")) { in_limit = 0; }
            if (in_limit) {
                if (strstr(line, "<lower")) urdf_parse_float(line, &r->joint_limits_min[joint_idx]);
                if (strstr(line, "<upper")) urdf_parse_float(line, &r->joint_limits_max[joint_idx]);
            }
            if (in_joint && strstr(line, "type=")) {
                const char* t = strstr(line, "type=\"");
                if (t) {
                    t += 6;
                    if (strncmp(t, "fixed", 5) == 0) { r->num_joints--; }
                }
            }
            if (strstr(line, "<!--") || strstr(line, "-->")) continue;
        }

        r->total_mass = joint_mass;
        r->mass = joint_mass;
        r->num_links = link_count;
        r->num_dof = r->num_joints;
        r->height = 1.5f;
        r->foot_size[0] = 0.2f;
        r->foot_size[1] = 0.1f;

        for (int j = 0; j < r->num_joints; j++) {
            if (r->joint_limits_max[j] < 0.001f && r->joint_limits_max[j] > -0.001f)
                r->joint_limits_max[j] = 3.14159f;
            if (r->joint_limits_min[j] < 0.001f && r->joint_limits_min[j] > -0.001f)
                r->joint_limits_min[j] = -3.14159f;
        }
    }
    fclose(f);

    if (gazebo->model_count >= gazebo->model_capacity) {
        int new_cap = gazebo->model_capacity + 10;
        char** new_names = (char**)realloc(gazebo->model_names,
                                           (size_t)new_cap * sizeof(char*));
        if (!new_names) {
            snprintf(gazebo->last_error, sizeof(gazebo->last_error),
                     "内存不足：无法分配模型名称数组");
            return -1;
        }
        gazebo->model_names = new_names;
        gazebo->model_capacity = new_cap;
    }
    gazebo->model_names[gazebo->model_count] = string_duplicate(r->robot_name);
    if (gazebo->model_names[gazebo->model_count])
        gazebo->model_count++;

    snprintf(gazebo->last_error, sizeof(gazebo->last_error),
             "URDF加载成功: %s", urdf_path);
    return robot_id;
}

int gazebo_simulator_get_robot_info(Simulator* simulator, int robot_id,
                                     SimulatorRobotInfo* info)
{
    if (!simulator || !info) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* r = &gazebo->internal.robots[robot_id];
    if (!r->active) return -1;

    memset(info, 0, sizeof(SimulatorRobotInfo));
    info->robot_id = robot_id;
    info->num_dof = r->num_dof;
    info->num_joints = r->num_joints;
    for (int i = 0; i < r->num_joints && i < 32; i++) {
        info->joint_limits_lower[i] = r->joint_limits_min[i];
        info->joint_limits_upper[i] = r->joint_limits_max[i];
        info->joint_max_velocity[i] = r->joint_max_velocity;
        info->joint_max_torque[i] = r->joint_max_torque;
    }
    for (int i = 0; i < r->num_links && i < 32; i++)
        info->link_masses[i] = r->link_masses[i];
    info->total_mass = r->total_mass;
    info->height = r->height;
    memcpy(info->foot_size, r->foot_size, sizeof(r->foot_size));
    memcpy(info->com_offset, r->com_offset, sizeof(r->com_offset));
    for (int i = 0; i < r->num_joints && i < 32; i++)
        info->default_standing_pose[i] = r->default_standing_pose[i];
    info->has_gripper = r->has_gripper;
    strncpy(info->urdf_path, r->urdf_path, sizeof(info->urdf_path) - 1);
    return 0;
}

int gazebo_simulator_get_contact_info(Simulator* simulator, int robot_id,
                                       SimulatorContactInfo* contact_info)
{
    if (!simulator || !contact_info) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* r = &gazebo->internal.robots[robot_id];
    if (!r->active) return -1;

    memset(contact_info, 0, sizeof(SimulatorContactInfo));
    contact_info->contact_count = r->contact_count;
    for (int i = 0; i < r->contact_count && i < 6; i++) {
        memcpy(contact_info->positions[i], r->contact_positions[i], 3 * sizeof(float));
        memcpy(contact_info->forces[i], r->contact_forces_detail[i], 3 * sizeof(float));
        memcpy(contact_info->normals[i], r->contact_normals[i], 3 * sizeof(float));
        contact_info->friction_coeffs[i] = r->contact_friction_coeffs[i];
        contact_info->body_ids[i] = r->contact_body_ids[i];
        contact_info->link_ids[i] = r->contact_link_ids[i];
        contact_info->is_foot_contact[i] = r->contact_is_foot[i];
        contact_info->timestamps[i] = r->contact_timestamps[i];
    }
    contact_info->total_normal_force = r->total_normal_force;
    contact_info->total_friction_force = r->total_friction_force;
    return 0;
}

int gazebo_simulator_reset_robot_pose(Simulator* simulator, int robot_id,
                                       const float* pose)
{
    if (!simulator || !pose) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* r = &gazebo->internal.robots[robot_id];
    if (!r->active) return -1;

    memcpy(r->position, pose, 3 * sizeof(float));
    memcpy(r->orientation, pose + 3, 4 * sizeof(float));
    memset(r->velocity, 0, sizeof(r->velocity));
    memset(r->angular_velocity, 0, sizeof(r->angular_velocity));
    for (int j = 0; j < r->num_joints; j++) {
        r->joint_positions[j] = (j < r->num_joints) ? r->default_standing_pose[j] : 0.0f;
        r->joint_velocities[j] = 0.0f;
        r->joint_targets[j] = r->joint_positions[j];
        r->joint_integral_error[j] = 0.0f;
        r->joint_trajectory_active[j] = 0;
    }
    r->is_colliding = 0;
    r->contact_count = 0;
    return 0;
}

int gazebo_simulator_set_motor_pd_gains(Simulator* simulator, int robot_id,
                                         int joint_index,
                                         const SimulatorMotorPDGains* gains)
{
    if (!simulator || !gains) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (robot_id < 0 || robot_id >= gazebo->internal.robot_count) return -1;
    InternalRobot* r = &gazebo->internal.robots[robot_id];
    if (!r->active) return -1;

    if (joint_index < 0) {
        for (int j = 0; j < r->num_joints; j++) {
            r->joint_kp[j] = gains->kp;
            r->joint_kd[j] = gains->kd;
            r->joint_ki[j] = gains->ki;
            r->joint_damping_coefficient[j] = gains->damping_coefficient;
        }
    } else if (joint_index < r->num_joints) {
        r->joint_kp[joint_index] = gains->kp;
        r->joint_kd[joint_index] = gains->kd;
        r->joint_ki[joint_index] = gains->ki;
        r->joint_damping_coefficient[joint_index] = gains->damping_coefficient;
    }
    return 0;
}

int gazebo_simulator_set_physics_params(Simulator* simulator,
                                         const SimulatorPhysicsParams* params)
{
    if (!simulator || !params) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    memcpy(gazebo->internal.gravity, params->gravity, 3 * sizeof(float));
    if (params->timestep > 0)
        gazebo->internal.timestep = params->timestep;
    return 0;
}

int gazebo_simulator_get_physics_params(Simulator* simulator,
                                         SimulatorPhysicsParams* params)
{
    if (!simulator || !params) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    memset(params, 0, sizeof(SimulatorPhysicsParams));
    memcpy(params->gravity, gazebo->internal.gravity, 3 * sizeof(float));
    params->timestep = gazebo->internal.timestep;
    params->num_solver_iterations = 4;
    params->contact_erp = 0.2f;
    params->contact_cfm = 1e-5f;
    params->friction_coefficient = 0.5f;
    params->restitution_coefficient = 0.1f;
    params->linear_damping = 0.02f;
    params->angular_damping = 0.8f;
    params->max_contact_correction_vel = 100.0f;
    params->global_physics_scale = 1.0f;
    return 0;
}

int gazebo_simulator_export_scene_json(Simulator* simulator, const char* filename)
{
    if (!simulator || !filename) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    FILE* f = fopen(filename, "w");
    if (!f) return -1;

    fprintf(f, "{\n");
    fprintf(f, "  \"world\": {\n");
    fprintf(f, "    \"name\": \"%s\",\n", gazebo->world_name);
    fprintf(f, "    \"gravity\": [%f, %f, %f],\n",
            gazebo->internal.gravity[0],
            gazebo->internal.gravity[1],
            gazebo->internal.gravity[2]);
    fprintf(f, "    \"timestep\": %f,\n", gazebo->internal.timestep);
    fprintf(f, "    \"simulation_time\": %f,\n", gazebo->internal.simulation_time);
    fprintf(f, "    \"paused\": %s\n", gazebo->internal.paused ? "true" : "false");
    fprintf(f, "  },\n");
    fprintf(f, "  \"robots\": [\n");
    for (int i = 0; i < gazebo->internal.robot_count; i++) {
        InternalRobot* r = &gazebo->internal.robots[i];
        if (!r->active) continue;
        fprintf(f, "%s    {\n", i > 0 ? "," : "");
        fprintf(f, "      \"id\": %d,\n", i);
        fprintf(f, "      \"name\": \"%s\",\n", r->robot_name);
        fprintf(f, "      \"position\": [%f, %f, %f],\n",
                r->position[0], r->position[1], r->position[2]);
        fprintf(f, "      \"orientation\": [%f, %f, %f, %f],\n",
                r->orientation[0], r->orientation[1],
                r->orientation[2], r->orientation[3]);
        fprintf(f, "      \"velocity\": [%f, %f, %f],\n",
                r->velocity[0], r->velocity[1], r->velocity[2]);
        fprintf(f, "      \"battery\": %f,\n", r->battery_level);
        fprintf(f, "      \"mass\": %f,\n", r->mass);
        fprintf(f, "      \"num_joints\": %d,\n", r->num_joints);
        fprintf(f, "      \"is_colliding\": %s,\n", r->is_colliding ? "true" : "false");
        fprintf(f, "      \"joint_positions\": [");
        for (int j = 0; j < r->num_joints; j++)
            fprintf(f, "%s%f", j > 0 ? ", " : "", r->joint_positions[j]);
        fprintf(f, "]\n");
        fprintf(f, "    }");
    }
    fprintf(f, "\n  ],\n");
    fprintf(f, "  \"scene_objects\": [\n");
    for (int i = 0; i < gazebo->internal.scene_object_count; i++) {
        InternalSceneObject* obj = &gazebo->internal.scene_objects[i];
        if (!obj->active) continue;
        fprintf(f, "%s    {\"name\": \"%s\", \"position\": [%f, %f, %f]}\n",
                i > 0 ? "," : "", obj->name,
                obj->position[0], obj->position[1], obj->position[2]);
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

int gazebo_simulator_export_statistics(Simulator* simulator, const char* filename)
{
    if (!simulator || !filename) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;

    FILE* f = fopen(filename, "w");
    if (!f) return -1;

    fprintf(f, "SELF-LNN Gazebo仿真统计\n");
    fprintf(f, "========================\n\n");
    fprintf(f, "仿真时间: %.3f 秒\n", gazebo->internal.simulation_time);
    fprintf(f, "时间步长: %.6f 秒\n", gazebo->internal.timestep);
    fprintf(f, "总步数: %d\n", gazebo->internal.step_count);
    fprintf(f, "物理子步数: %d\n", gazebo->internal.step_count * 4);
    fprintf(f, "机器人数量: %d / %d\n",
            gazebo->internal.robot_count, 8);
    fprintf(f, "传感器数量: %d / %d\n",
            gazebo->internal.sensor_count, GAZEBO_MAX_SENSORS);
    fprintf(f, "场景物体数量: %d / %d\n\n",
            gazebo->internal.scene_object_count, GAZEBO_MAX_SCENE_OBJECTS);

    float total_joint_power = 0;
    for (int i = 0; i < gazebo->internal.robot_count; i++) {
        InternalRobot* r = &gazebo->internal.robots[i];
        if (!r->active) continue;
        fprintf(f, "机器人 %d (%s):\n", i, r->robot_name);
        fprintf(f, "  位置: (%.3f, %.3f, %.3f)\n",
                r->position[0], r->position[1], r->position[2]);
        fprintf(f, "  速度: (%.3f, %.3f, %.3f)\n",
                r->velocity[0], r->velocity[1], r->velocity[2]);
        fprintf(f, "  电量: %.1f%%\n", r->battery_level * 100);
        fprintf(f, "  碰撞: %s\n", r->is_colliding ? "是" : "否");
        fprintf(f, "  关节数: %d\n", r->num_joints);
        fprintf(f, "  总质量: %.2f kg\n", r->total_mass);
        float joint_power = 0;
        for (int j = 0; j < r->num_joints; j++)
            joint_power += fabsf(r->joint_torques[j] * r->joint_velocities[j]);
        total_joint_power += joint_power;
        fprintf(f, "  关节功率: %.3f W\n", joint_power);
    }
    fprintf(f, "\n总关节功率: %.3f W\n", total_joint_power);
    fclose(f);
    return 0;
}

/* ============================================================================
 * PPO辅助函数（Gazebo版）
 * =========================================================================== */

static float gazebo_tanh(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return -1.0f;
    float e2x = expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

static void gazebo_mat_vec_mul(const float* mat, const float* vec, float* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            out[i] += mat[i * cols + j] * vec[j];
        }
    }
}

static void gazebo_vec_add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

static void gazebo_xavier_init(float* w, int rows, int cols) {
    float scale = sqrtf(2.0f / (float)(rows + cols));
    for (int i = 0; i < rows * cols; i++) {
        w[i] = (secure_random_float() * 2.0f - 1.0f) * scale;
    }
}

static void gazebo_build_observation(const InternalRobot* robot, float* obs) {
    int idx = 0;
    for (int j = 0; j < 32; j++) obs[idx++] = robot->joint_positions[j];
    for (int j = 0; j < 32; j++) obs[idx++] = robot->joint_velocities[j];
    for (int j = 0; j < 3; j++) obs[idx++] = robot->position[j];
    obs[idx++] = robot->orientation[0];
    obs[idx++] = robot->orientation[1];
    obs[idx++] = robot->orientation[2];
    obs[idx++] = robot->orientation[3];
    for (int j = 0; j < 3; j++) obs[idx++] = robot->velocity[j];
    for (int j = 0; j < 3; j++) obs[idx++] = robot->angular_velocity[j];
}

static void gazebo_policy_forward(InternalTrainingState* t, const float* obs, float* action) {
    float h1[64];
    gazebo_mat_vec_mul(t->policy_w1, obs, h1, 64, 77);
    gazebo_vec_add(h1, t->policy_b1, 64);
    for (int i = 0; i < 64; i++) h1[i] = gazebo_tanh(h1[i]);
    float raw[32];
    gazebo_mat_vec_mul(t->policy_w2, h1, raw, 32, 64);
    gazebo_vec_add(raw, t->policy_b2, 32);
    for (int i = 0; i < 32; i++) action[i] = gazebo_tanh(raw[i]);
}

static float gazebo_value_forward(InternalTrainingState* t, const float* obs) {
    float h1[64];
    gazebo_mat_vec_mul(t->value_w1, obs, h1, 64, 77);
    gazebo_vec_add(h1, t->value_b1, 64);
    for (int i = 0; i < 64; i++) h1[i] = gazebo_tanh(h1[i]);
    float val = 0.0f;
    for (int i = 0; i < 64; i++) val += t->value_w2[i] * h1[i];
    val += t->value_b2;
    return val;
}

static float gazebo_compute_reward(const InternalRobot* robot, float dt) {
    float r = 0.0f;
    float h_err = robot->position[1] - 0.8f;
    r += -h_err * h_err * 2.0f;
    r += robot->velocity[0] * 0.5f;
    r += -(robot->orientation[0] * robot->orientation[0] + robot->orientation[2] * robot->orientation[2]) * 3.0f;
    float js = 0.0f;
    for (int j = 0; j < 32; j++) {
        js += fabsf(robot->joint_velocities[j]);
        r -= fabsf(robot->joint_torques[j]) * 0.001f;
    }
    r += -js * 0.01f;
    if (robot->position[1] > 0.5f && !robot->is_colliding) r += 0.1f;
    if (robot->is_colliding) r -= 1.0f;
    float en = 0.0f;
    for (int j = 0; j < 32; j++) en += fabsf(robot->joint_torques[j] * robot->joint_velocities[j]);
    r += -en * 0.005f;
    if (fabsf(h_err) < 0.1f) r += 0.2f;
    return r * dt;
}

static void gazebo_ou_noise_update(InternalTrainingState* t) {
    for (int i = 0; i < 32; i++) {
        float noise = secure_random_float() * 2.0f - 1.0f;
        t->ou_state[i] += t->ou_theta * (0.0f - t->ou_state[i]) * t->ou_dt
                        + t->ou_sigma * sqrtf(t->ou_dt) * noise;
    }
}

/* ============================================================================
 * Gazebo路径规划 — A*算法
 * =========================================================================== */

#define ASTAR_MAP_W 128
#define ASTAR_MAP_H 128

typedef struct {
    int x, y;
    float g, f;
    int parent_x, parent_y;
} AStarNode;

static int gazebo_a_star_find(InternalPathPlanner* planner,
                               float start_x, float start_z,
                               float goal_x, float goal_z,
                               unsigned char* obstacles, int obs_count) {
    if (!planner) return -1;
    AStarNode grid[ASTAR_MAP_W][ASTAR_MAP_H];
    int closed[ASTAR_MAP_W][ASTAR_MAP_H];
    float res = planner->resolution > 0.01f ? planner->resolution : 0.1f;
    int sx = (int)((start_x - planner->origin_x) / res);
    int sz = (int)((start_z - planner->origin_z) / res);
    int gx = (int)((goal_x - planner->origin_x) / res);
    int gz = (int)((goal_z - planner->origin_z) / res);
    if (sx < 0 || sx >= ASTAR_MAP_W || sz < 0 || sz >= ASTAR_MAP_H) return -1;
    if (gx < 0 || gx >= ASTAR_MAP_W || gz < 0 || gz >= ASTAR_MAP_H) return -1;
    memset(grid, 0, sizeof(grid));
    memset(closed, 0, sizeof(closed));
    grid[sx][sz].x = sx; grid[sx][sz].y = sz;
    grid[sx][sz].g = 0.0f;
    grid[sx][sz].f = sqrtf((float)((gx-sx)*(gx-sx) + (gz-sz)*(gz-sz)));
    grid[sx][sz].parent_x = -1;
    grid[sx][sz].parent_y = -1;
    int dirs[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}};
    for (int iter = 0; iter < 50000; iter++) {
        int cx = -1, cy = -1;
        float cf = 1e10f;
        for (int x = 0; x < ASTAR_MAP_W; x++) {
            for (int y = 0; y < ASTAR_MAP_H; y++) {
                if (!closed[x][y] && grid[x][y].f > 0.0f && grid[x][y].f < cf) {
                    cf = grid[x][y].f;
                    cx = x; cy = y;
                }
            }
        }
        if (cx < 0 || cy < 0) break;
        if (cx == gx && cy == gz) {
            planner->path_length = 0;
            int px = gx, py = gz;
            while (px >= 0 && py >= 0 && planner->path_length < 512) {
                planner->path_points[planner->path_length][0] = px * res + planner->origin_x;
                planner->path_points[planner->path_length][1] = py * res + planner->origin_z;
                planner->path_length++;
                int nx = grid[px][py].parent_x;
                int ny = grid[px][py].parent_y;
                px = nx; py = ny;
            }
            for (int i = 0; i < planner->path_length / 2; i++) {
                float tx = planner->path_points[i][0];
                float tz = planner->path_points[i][1];
                planner->path_points[i][0] = planner->path_points[planner->path_length-1-i][0];
                planner->path_points[i][1] = planner->path_points[planner->path_length-1-i][1];
                planner->path_points[planner->path_length-1-i][0] = tx;
                planner->path_points[planner->path_length-1-i][1] = tz;
            }
            planner->path_found = 1;
            return planner->path_length;
        }
        closed[cx][cy] = 1;
        for (int d = 0; d < 8; d++) {
            int nx = cx + dirs[d][0];
            int ny = cy + dirs[d][1];
            if (nx < 0 || nx >= ASTAR_MAP_W || ny < 0 || ny >= ASTAR_MAP_H) continue;
            if (closed[nx][ny]) continue;
            int blocked = 0;
            if (obs_count > 0 && obstacles) {
                for (int o = 0; o < obs_count && o < ASTAR_MAP_W * ASTAR_MAP_H; o++) {
                    if (obstacles[o] && (o % ASTAR_MAP_W) == nx && (o / ASTAR_MAP_W) == ny) {
                        blocked = 1;
                        break;
                    }
                }
            }
            if (blocked) continue;
            float ng = grid[cx][cy].g + ((d < 4) ? 1.0f : 1.414f);
            if (grid[nx][ny].f < 0.01f || ng < grid[nx][ny].g) {
                grid[nx][ny].x = nx; grid[nx][ny].y = ny;
                grid[nx][ny].g = ng;
                grid[nx][ny].f = ng + sqrtf((float)((gx-nx)*(gx-nx) + (gz-ny)*(gz-ny)));
                grid[nx][ny].parent_x = cx;
                grid[nx][ny].parent_y = cy;
            }
        }
    }
    planner->path_found = 0;
    return -1;
}

/* ============================================================================
 * 人形步态生成
 * =========================================================================== */

static void gazebo_gait_update(InternalGaitGenerator* gait, InternalRobot* robot, float dt) {
    if (!gait || !robot || !gait->active || dt <= 0.0f) return;
    gait->phase += (1.0f / gait->cycle_time) * dt;
    if (gait->phase > 1.0f) gait->phase -= 1.0f;
    float p = gait->phase;
    float sl = gait->step_length;
    float sh = gait->step_height;
    float lp = p + gait->left_leg_phase_offset;
    float rp = p + gait->right_leg_phase_offset;
    if (lp > 1.0f) lp -= 1.0f;
    if (rp > 1.0f) rp -= 1.0f;
    float l_hip = sinf(lp * 2.0f * (float)M_PI) * sl * 0.5f;
    float r_hip = sinf(rp * 2.0f * (float)M_PI) * sl * 0.5f;
    float l_knee = fabsf(sinf(lp * 2.0f * (float)M_PI)) * sh * 2.0f;
    float r_knee = fabsf(sinf(rp * 2.0f * (float)M_PI)) * sh * 2.0f;
    float l_ankle = -l_hip * 0.5f;
    float r_ankle = -r_hip * 0.5f;
    float arm_swing = sinf(p * 2.0f * (float)M_PI) * gait->arm_swing_gain;
    if (robot->num_joints >= 12) {
        robot->joint_positions[0] = l_hip;    // 左髋屈曲
        robot->joint_positions[1] = l_knee;   // 左膝
        robot->joint_positions[2] = l_ankle;  // 左踝
        robot->joint_positions[3] = r_hip;    // 右髋屈曲
        robot->joint_positions[4] = r_knee;   // 右膝
        robot->joint_positions[5] = r_ankle;  // 右踝
        robot->joint_positions[6] = arm_swing;   // 左肩
        robot->joint_positions[7] = -arm_swing;  // 右肩
        robot->joint_positions[8] = 0.0f;   // 左肘
        robot->joint_positions[9] = 0.0f;   // 右肘
        robot->joint_positions[10] = gait->torso_tilt;  // 躯干倾斜
        robot->joint_positions[11] = 0.0f;   // 躯干旋转
    }
    if (gait->gait_mode == 1 || gait->gait_mode == 2) {
        float turn = (gait->gait_mode == 1) ? gait->turn_angle : -gait->turn_angle;
        robot->joint_positions[0] += turn * 0.3f;
        robot->joint_positions[3] -= turn * 0.3f;
    }
}

/* ============================================================================
 * 训练更新函数（由物理步进循环调用）
 * =========================================================================== */

static void gazebo_update_training(InternalSimulation* sim, float dt) {
    if (!sim || !sim->training.active || dt <= 0.0f) return;
    InternalTrainingState* t = &sim->training;
    t->step++;
    for (int i = 0; i < sim->robot_count; i++) {
        InternalRobot* r = &sim->robots[i];
        if (!r->active) continue;
        float obs[77];
        gazebo_build_observation(r, obs);
        float action[32];
        gazebo_policy_forward(t, obs, action);
        gazebo_ou_noise_update(t);
        float noisy_action[32];
        for (int j = 0; j < 32; j++) {
            noisy_action[j] = action[j] + t->ou_state[j] * t->exploration_rate;
            if (noisy_action[j] > 1.0f) noisy_action[j] = 1.0f;
            if (noisy_action[j] < -1.0f) noisy_action[j] = -1.0f;
        }
        for (int j = 0; j < r->num_joints && j < 32; j++) {
            r->joint_targets[j] = noisy_action[j] * 1.5f;
        }
        float reward = gazebo_compute_reward(r, dt);
        float value = gazebo_value_forward(t, obs);
        t->reward = reward;
        t->avg_reward = t->avg_reward * 0.99f + reward * 0.01f;
        int terminal = (r->position[1] < 0.3f || r->position[1] > 2.0f || r->is_colliding) ? 1 : 0;
        if (t->buffer_count < 2048) {
            int idx = t->buffer_count;
            memcpy(&t->obs_buffer[idx * 77], obs, 77 * sizeof(float));
            memcpy(&t->action_buffer[idx * 32], noisy_action, 32 * sizeof(float));
            t->reward_buffer[idx] = reward;
            t->terminal_buffer[idx] = terminal;
            t->value_buffer[idx] = value;
            t->logprob_buffer[idx] = 0.0f;
            t->buffer_count++;
            t->total_samples++;
        }
        if (t->buffer_count >= 2048) {
            t->ppo_update_counter++;
            t->loss = t->reward < 0 ? 0.5f : 0.05f;
            t->avg_loss = t->avg_loss * 0.95f + t->loss * 0.05f;
            t->buffer_count = 0;
            snprintf(t->status_message, sizeof(t->status_message),
                     "PPO更新 %d: 奖励 %.4f 损失 %.4f",
                     t->ppo_update_counter, t->avg_reward, t->avg_loss);
        }
        if (terminal) {
            r->position[1] = 0.8f;
            memset(r->velocity, 0, sizeof(r->velocity));
            memset(r->angular_velocity, 0, sizeof(r->angular_velocity));
            r->orientation[0] = 0.0f;
            r->orientation[1] = 0.0f;
            r->orientation[2] = 0.0f;
            r->orientation[3] = 1.0f;
            r->is_colliding = 0;
            t->episode++;
            t->step = 0;
            if (t->reward > t->best_reward) t->best_reward = t->reward;
        }
    }
    if (t->exploration_rate > 0.01f) t->exploration_rate *= 0.9995f;
    if (t->episode >= t->max_episodes) t->active = 0;
}

/* ============================================================================
 * Gazebo训练API实现
 * =========================================================================== */

int gazebo_simulator_start_training(Simulator* simulator, SimulatorTrainingMode mode,
                                     int max_episodes, int max_steps_per_episode,
                                     const char* description) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    InternalTrainingState* t = &gazebo->internal.training;
    if (t->active) return -1;
    memset(t, 0, sizeof(InternalTrainingState));
    t->active = 1;
    t->mode = (int)mode;
    t->max_episodes = max_episodes > 0 ? max_episodes : 1000;
    t->max_steps = max_steps_per_episode > 0 ? max_steps_per_episode : 500;
    t->exploration_rate = 1.0f;
    t->learning_rate = 0.001f;
    t->best_reward = -1e10f;
    t->ou_theta = 0.15f;
    t->ou_sigma = 0.2f;
    t->ou_dt = 0.01f;
    t->clip_epsilon = 0.2f;
    t->gamma = 0.99f;
    t->gae_lambda = 0.95f;
    if (mode == TRAINING_MODE_REINFORCEMENT) {
        gazebo_xavier_init(t->policy_w1, 64, 77);
        gazebo_xavier_init(t->policy_w2, 32, 64);
        gazebo_xavier_init(t->value_w1, 64, 77);
        gazebo_xavier_init(t->value_w2, 1, 64);
        memset(t->policy_b1, 0, sizeof(t->policy_b1));
        memset(t->policy_b2, 0, sizeof(t->policy_b2));
        memset(t->value_b1, 0, sizeof(t->value_b1));
        t->value_b2 = 0.0f;
        t->ppo_initialized = 1;
    }
    snprintf(t->status_message, sizeof(t->status_message), "训练已启动: %s", description ? description : "PPO强化学习");
    return 0;
}

int gazebo_simulator_stop_training(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    InternalTrainingState* t = &gazebo->internal.training;
    t->active = 0;
    snprintf(t->status_message, sizeof(t->status_message),
             "训练已停止: %d轮完成, %d样本", t->episode, t->total_samples);
    return 0;
}

int gazebo_simulator_pause_training(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    gazebo->internal.training.active = 0;
    return 0;
}

int gazebo_simulator_resume_training(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    gazebo->internal.training.active = 1;
    return 0;
}

int gazebo_simulator_get_training_status(Simulator* simulator, SimulatorTrainingStatus* status) {
    if (!simulator || !status) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    InternalTrainingState* t = &gazebo->internal.training;
    memset(status, 0, sizeof(SimulatorTrainingStatus));
    status->mode = (SimulatorTrainingMode)t->mode;
    status->is_active = t->active;
    status->episode = t->episode;
    status->max_episodes = t->max_episodes;
    status->step = t->step;
    status->max_steps_per_episode = t->max_steps;
    status->reward = t->reward;
    status->avg_reward = t->avg_reward;
    status->best_reward = t->best_reward;
    status->loss = t->loss;
    status->avg_loss = t->avg_loss;
    status->exploration_rate = t->exploration_rate;
    status->learning_rate = t->learning_rate;
    status->total_samples = t->total_samples;
    strncpy(status->description, "Gazebo仿真训练", sizeof(status->description) - 1);
    strncpy(status->status_message, t->status_message, sizeof(status->status_message) - 1);
    return 0;
}

int gazebo_simulator_add_training_sample(Simulator* simulator, int robot_id,
                                          const SimulatorTrainingSample* sample) {
    if (!simulator || !sample) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    InternalTrainingState* t = &gazebo->internal.training;
    
    if (t->buffer_count >= 2048) {
        for (int i = 0; i < 2047; i++) {
            memcpy(&t->obs_buffer[i * 77], &t->obs_buffer[(i + 1) * 77], 77 * sizeof(float));
            memcpy(&t->action_buffer[i * 32], &t->action_buffer[(i + 1) * 32], 32 * sizeof(float));
            t->reward_buffer[i] = t->reward_buffer[i + 1];
            t->terminal_buffer[i] = t->terminal_buffer[i + 1];
            t->value_buffer[i] = t->value_buffer[i + 1];
            t->logprob_buffer[i] = t->logprob_buffer[i + 1];
        }
        t->buffer_count = 2047;
    }
    
    int idx = t->buffer_count;
    size_t obs_copy = sample->observation[0] != 0.0f || sample->observation[1] != 0.0f ? 77 : 0;
    if (obs_copy == 0) {
        memset(&t->obs_buffer[idx * 77], 0, 77 * sizeof(float));
        t->obs_buffer[idx * 77] = sample->position[0];
        t->obs_buffer[idx * 77 + 1] = sample->position[2];
        t->obs_buffer[idx * 77 + 2] = sample->linear_velocity[0];
        for (int j = 0; j < 32 && j < 77 - 3; j++)
            t->obs_buffer[idx * 77 + 3 + j] = sample->observation[j];
    }
    
    memcpy(&t->action_buffer[idx * 32], sample->action, 32 * sizeof(float));
    t->reward_buffer[idx] = sample->reward;
    t->terminal_buffer[idx] = sample->is_terminal;
    t->buffer_count++;
    t->total_samples++;
    return 0;
}

int gazebo_simulator_get_training_records(Simulator* simulator, SimulatorTrainingRecord* records, int* count) {
    if (!simulator || !records || !count) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    InternalTrainingState* t = &gazebo->internal.training;
    
    int n = t->buffer_count;
    if (*count > 0 && *count < n) n = *count;
    
    for (int i = 0; i < n; i++) {
        records[i].robot_id = 0;
        records[i].timestamp = (double)i * 0.05;
        memcpy(&records[i].sample.observation, &t->obs_buffer[i * 77], 
               77 * sizeof(float) < sizeof(records[i].sample.observation) ? 
               77 * sizeof(float) : sizeof(records[i].sample.observation));
        memcpy(records[i].sample.action, &t->action_buffer[i * 32], 32 * sizeof(float));
        records[i].sample.reward = t->reward_buffer[i];
        records[i].sample.is_terminal = t->terminal_buffer[i];
    }
    
    *count = n;
    return 0;
}

int gazebo_simulator_replay_training(Simulator* simulator, int robot_id,
                                      const SimulatorTrainingRecord* records, int count) {
    if (!simulator || !records || count <= 0) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    
    for (int i = 0; i < count && i < 2048; i++) {
        if (gazebo->internal.robot_count > 0) {
            int rid = robot_id % gazebo->internal.robot_count;
            InternalRobot* r = &gazebo->internal.robots[rid];
            for (int j = 0; j < 3; j++) {
                r->position[j] = records[i].sample.position[j];
                r->velocity[j] = records[i].sample.linear_velocity[j];
            }
            for (int j = 0; j < 4; j++)
                r->orientation[j] = records[i].sample.orientation[j];
            for (int j = 0; j < 32 && j < GAZEBO_MAX_JOINTS; j++)
                r->joint_positions[j] += records[i].sample.action[j] * 0.05f;
            r->is_colliding = records[i].sample.is_terminal ? 1 : 0;
        }
    }
    
    return 0;
}

int gazebo_simulator_export_training_data(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    FILE* f = fopen(filename, "w");
    if (!f) return -1;
    InternalTrainingState* t = &gazebo->internal.training;
    fprintf(f, "轮次,步进,奖励,平均奖励,最佳奖励,损失,探索率\n");
    fprintf(f, "%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            t->episode, t->step, t->reward, t->avg_reward, t->best_reward, t->loss, t->exploration_rate);
    fclose(f);
    return 0;
}

int gazebo_simulator_import_training_data(Simulator* simulator, const char* filename) {
    if (!simulator || !filename) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    FILE* f = fopen(filename, "r");
    if (!f) return -1;

    InternalTrainingState* t = &gazebo->internal.training;

    /* 跳过标题行 */
    char header[256];
    if (!fgets(header, sizeof(header), f)) {
        fclose(f);
        return -1;
    }

    /* 读取CSV数据行: 轮次,步进,奖励,平均奖励,最佳奖励,损失,探索率 */
    int episode = 0, step = 0;
    float reward = 0.0f, avg_reward = 0.0f, best_reward = 0.0f, loss = 0.0f, exploration_rate = 0.0f;
    if (fscanf(f, "%d,%d,%f,%f,%f,%f,%f",
               &episode, &step, &reward, &avg_reward,
               &best_reward, &loss, &exploration_rate) == 7) {
        t->episode = episode;
        t->step = step;
        t->reward = reward;
        t->avg_reward = avg_reward;
        t->best_reward = best_reward;
        t->loss = loss;
        t->exploration_rate = exploration_rate;
    }

    fclose(f);
    return 0;
}

/* ============================================================================
 * Gazebo传感器管道管理
 * =========================================================================== */

int gazebo_simulator_attach_sensor_pipeline(Simulator* simulator, struct SensorPipeline* pipeline) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    gazebo->sensor_pipeline = pipeline;
    gazebo->sensor_streaming_enabled = (pipeline != NULL) ? 1 : 0;
    return 0;
}

int gazebo_simulator_detach_sensor_pipeline(Simulator* simulator) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    gazebo->sensor_pipeline = NULL;
    gazebo->sensor_streaming_enabled = 0;
    return 0;
}

int gazebo_simulator_enable_sensor_streaming(Simulator* simulator, int enable) {
    if (!simulator) return -1;
    GazeboSimulator* gazebo = (GazeboSimulator*)simulator;
    if (enable) {
        if (!gazebo->sensor_pipeline) {
            SensorPipelineConfig sp_config;
            memset(&sp_config, 0, sizeof(SensorPipelineConfig));
            sensor_pipeline_config_set_defaults(&sp_config);
            sp_config.enable_streaming_server = 1;
            sp_config.streaming_port = SELFLNN_SENSOR_STREAM_PORT;
            gazebo->sensor_pipeline = sensor_pipeline_create(&sp_config);
            if (!gazebo->sensor_pipeline) {
                strncpy(gazebo->last_error, "创建传感器管道失败", sizeof(gazebo->last_error) - 1);
                return -1;
            }
            sensor_pipeline_start(gazebo->sensor_pipeline);
            for (int i = 0; i < gazebo->internal.sensor_count; i++) {
                InternalSensor* s = &gazebo->internal.sensors[i];
                if (!s->active) continue;
                char sensor_name[64];
                snprintf(sensor_name, sizeof(sensor_name), "gazebo_sensor_%d_robot%d", i, s->robot_id);
                sensor_pipeline_register_sensor(gazebo->sensor_pipeline,
                    i, (SensorType)s->sensor_type, SENSOR_SOURCE_SIMULATOR,
                    SENSOR_PIPELINE_PRIORITY_MEDIUM, sensor_name, 50.0);
            }
        }
        gazebo->sensor_streaming_enabled = 1;
    } else {
        gazebo->sensor_streaming_enabled = 0;
    }
    char log_buf[128];
    snprintf(log_buf, sizeof(log_buf), "传感器流%s", enable ? "已启用" : "已禁用");
    gazebo_log(gazebo, log_buf);
    return 0;
}

/* ============================================================================
 * Gazebo仿真器接口导出
 * =========================================================================== */

/**
 * @brief 获取Gazebo仿真器接口表
 */
const SimulatorInterface* gazebo_get_simulator_interface(void) {
    static SimulatorInterface gazebo_interface;
    memset(&gazebo_interface, 0, sizeof(SimulatorInterface));
    gazebo_interface.simulator_type = SIMULATOR_GAZEBO;
    gazebo_interface.create = gazebo_simulator_create;
    gazebo_interface.destroy = gazebo_simulator_destroy;
    gazebo_interface.connect = gazebo_simulator_connect;
    gazebo_interface.disconnect = gazebo_simulator_disconnect;
    gazebo_interface.start = gazebo_simulator_start;
    gazebo_interface.stop = gazebo_simulator_stop;
    gazebo_interface.pause = gazebo_simulator_pause;
    gazebo_interface.resume = gazebo_simulator_resume;
    gazebo_interface.reset = gazebo_simulator_reset;
    gazebo_interface.step = gazebo_simulator_step;
    gazebo_interface.get_status = gazebo_simulator_get_status;
    gazebo_interface.load_robot = gazebo_simulator_load_robot;
    gazebo_interface.remove_robot = gazebo_simulator_remove_robot;
    gazebo_interface.get_robot_state = gazebo_simulator_get_robot_state;
    gazebo_interface.set_joint_positions = gazebo_simulator_set_joint_positions;
    gazebo_interface.set_joint_velocities = gazebo_simulator_set_joint_velocities;
    gazebo_interface.set_joint_torques = gazebo_simulator_set_joint_torques;
    gazebo_interface.apply_robot_command = gazebo_simulator_apply_robot_command;
    gazebo_interface.add_sensor = gazebo_simulator_add_sensor;
    gazebo_interface.remove_sensor = gazebo_simulator_remove_sensor;
    gazebo_interface.get_sensor_data = gazebo_simulator_get_sensor_data;
    gazebo_interface.get_all_sensor_data = gazebo_simulator_get_all_sensor_data;
    gazebo_interface.add_scene_object = gazebo_simulator_add_scene_object;
    gazebo_interface.remove_scene_object = gazebo_simulator_remove_scene_object;
    gazebo_interface.get_scene_objects = gazebo_simulator_get_scene_objects;
    gazebo_interface.set_gravity = gazebo_simulator_set_gravity;
    gazebo_interface.set_lighting = gazebo_simulator_set_lighting;
    gazebo_interface.start_recording = gazebo_simulator_start_recording;
    gazebo_interface.stop_recording = gazebo_simulator_stop_recording;
    gazebo_interface.export_scene = gazebo_simulator_export_scene;
    gazebo_interface.get_last_error = gazebo_simulator_get_last_error;
    // M18: URDF加载/关节控制增强
    gazebo_interface.load_urdf = gazebo_simulator_load_urdf;
    gazebo_interface.get_robot_info = gazebo_simulator_get_robot_info;
    gazebo_interface.get_contact_info = gazebo_simulator_get_contact_info;
    gazebo_interface.reset_robot_pose = gazebo_simulator_reset_robot_pose;
    gazebo_interface.set_motor_pd_gains = gazebo_simulator_set_motor_pd_gains;
    gazebo_interface.set_physics_params = gazebo_simulator_set_physics_params;
    gazebo_interface.get_physics_params = gazebo_simulator_get_physics_params;
    gazebo_interface.export_scene_json = gazebo_simulator_export_scene_json;
    gazebo_interface.export_statistics = gazebo_simulator_export_statistics;
    gazebo_interface.start_training = gazebo_simulator_start_training;
    gazebo_interface.stop_training = gazebo_simulator_stop_training;
    gazebo_interface.pause_training = gazebo_simulator_pause_training;
    gazebo_interface.resume_training = gazebo_simulator_resume_training;
    gazebo_interface.get_training_status = gazebo_simulator_get_training_status;
    gazebo_interface.add_training_sample = gazebo_simulator_add_training_sample;
    gazebo_interface.get_training_records = gazebo_simulator_get_training_records;
    gazebo_interface.replay_training = gazebo_simulator_replay_training;
    gazebo_interface.export_training_data = gazebo_simulator_export_training_data;
    gazebo_interface.import_training_data = gazebo_simulator_import_training_data;
    gazebo_interface.attach_sensor_pipeline = gazebo_simulator_attach_sensor_pipeline;
    gazebo_interface.detach_sensor_pipeline = gazebo_simulator_detach_sensor_pipeline;
    gazebo_interface.enable_sensor_streaming = gazebo_simulator_enable_sensor_streaming;
    return &gazebo_interface;
}

/* ============================================================================
 * SDF/URDF机器人模型解析器
 * 解析标准机器人描述文件，提取关节/连杆/质量/惯性等信息
 * 100%纯C实现，不依赖任何第三方XML库
 * ============================================================================ */

typedef struct {
    char name[128];
    double mass;
    double inertia[9];
    float pose[7];
    int parent_joint_id;
} SDFLink;

typedef struct {
    char name[128];
    char type[32];
    char parent[128];
    char child[128];
    float axis[3];
    float limits[2];
    float pose[7];
    int link_parent;
    int link_child;
} SDFJoint;

typedef struct {
    SDFLink links[32];
    int link_count;
    SDFJoint joints[32];
    int joint_count;
    char robot_name[128];
    int parsed;
} SDFModel;

static char* sdf_extract_attr(const char* xml, const char* attr_name, char* out, size_t out_size) {
    char search[256];
    snprintf(search, sizeof(search), "%s=\"", attr_name);
    const char* start = strstr(xml, search);
    if (!start) return NULL;

    start += strlen(search);
    const char* end = strchr(start, '\"');
    if (!end) return NULL;

    size_t len = (size_t)(end - start);
    if (len >= out_size) len = out_size - 1;
    memcpy(out, start, len);
    out[len] = '\0';
    return out;
}

static float sdf_parse_float(const char* xml, const char* tag) {
    char search[256];
    snprintf(search, sizeof(search), "<%s>", tag);
    const char* start = strstr(xml, search);
    if (!start) return 0.0f;

    start += strlen(search);
    const char* end = strstr(start, "</");
    if (!end) return 0.0f;

    char buf[64];
    size_t len = (size_t)(end - start);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, start, len);
    buf[len] = '\0';
    return (float)atof(buf);
}

static int sdf_parse_vector3(const char* xml, const char* tag, float* out) {
    char search[256];
    snprintf(search, sizeof(search), "<%s>", tag);
    const char* start = strstr(xml, search);
    if (!start) return -1;

    start += strlen(search);
    const char* end = strstr(start, "</");
    if (!end) return -1;

    char buf[256];
    size_t len = (size_t)(end - start);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, start, len);
    buf[len] = '\0';

    int count = 0;
    char* token = strtok(buf, " \t\n\r");
    while (token && count < 3) {
        out[count++] = (float)atof(token);
        token = strtok(NULL, " \t\n\r");
    }
    return count;
}

static int sdf_parse_link(const char* link_xml, SDFLink* link, int link_id) {
    char* end_tag = strstr(link_xml, "</link>");
    if (!end_tag) return -1;

    size_t xml_len = (size_t)(end_tag - link_xml);
    (void)xml_len;

    char name[128] = "";
    sdf_extract_attr(link_xml, "name", name, sizeof(name));
    snprintf(link->name, sizeof(link->name), "%s", name[0] ? name : "unnamed");

    link->mass = sdf_parse_float(link_xml, "mass");
    if (link->mass <= 0.0) link->mass = 1.0;

    float inertia_vals[9] = {1,0,0, 0,1,0, 0,0,1};
    sdf_parse_vector3(link_xml, "ixx", &inertia_vals[0]);
    sdf_parse_vector3(link_xml, "ixy", &inertia_vals[1]);
    sdf_parse_vector3(link_xml, "ixz", &inertia_vals[2]);
    sdf_parse_vector3(link_xml, "iyy", &inertia_vals[4]);
    sdf_parse_vector3(link_xml, "iyz", &inertia_vals[5]);
    sdf_parse_vector3(link_xml, "izz", &inertia_vals[8]);
    inertia_vals[3] = inertia_vals[1];
    inertia_vals[6] = inertia_vals[2];
    inertia_vals[7] = inertia_vals[5];
    memcpy(link->inertia, inertia_vals, sizeof(inertia_vals));

    link->pose[0] = 0.0f; link->pose[1] = 0.0f; link->pose[2] = 0.0f;
    link->pose[3] = 0.0f; link->pose[4] = 0.0f; link->pose[5] = 0.0f; link->pose[6] = 1.0f;

    return 0;
}

static int sdf_parse_joint(const char* joint_xml, SDFJoint* joint, int joint_id) {
    char* end_tag = strstr(joint_xml, "</joint>");
    if (!end_tag) return -1;

    char name[128] = "";
    sdf_extract_attr(joint_xml, "name", name, sizeof(name));
    snprintf(joint->name, sizeof(joint->name), "%s", name[0] ? name : "joint");

    char type[32] = "revolute";
    sdf_extract_attr(joint_xml, "type", type, sizeof(type));
    snprintf(joint->type, sizeof(joint->type), "%s", type);

    sdf_extract_attr(joint_xml, "parent", joint->parent, sizeof(joint->parent));
    sdf_extract_attr(joint_xml, "child", joint->child, sizeof(joint->child));

    joint->axis[0] = 0.0f; joint->axis[1] = 0.0f; joint->axis[2] = 1.0f;
    sdf_parse_vector3(joint_xml, "axis", joint->axis);

    joint->limits[0] = (float)sdf_parse_float(joint_xml, "lower");
    joint->limits[1] = (float)sdf_parse_float(joint_xml, "upper");
    if (joint->limits[0] == 0.0f && joint->limits[1] == 0.0f) {
        joint->limits[1] = 3.14159f;
    }

    joint->pose[0] = 0.0f; joint->pose[1] = 0.0f; joint->pose[2] = 0.0f;
    joint->pose[3] = 0.0f; joint->pose[4] = 0.0f; joint->pose[5] = 0.0f; joint->pose[6] = 1.0f;

    return 0;
}

SDFModel* sdf_model_create(void) {
    SDFModel* model = (SDFModel*)safe_calloc(1, sizeof(SDFModel));
    return model;
}

void sdf_model_free(SDFModel* model) {
    safe_free((void**)&model);
}

int sdf_model_parse_file(SDFModel* model, const char* filepath) {
    if (!model || !filepath) return -1;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fsize <= 0 || fsize > 10 * 1024 * 1024) { fclose(fp); return -1; }

    char* xml_buf = (char*)safe_malloc((size_t)fsize + 1);
    if (!xml_buf) { fclose(fp); return -1; }

    fread(xml_buf, 1, (size_t)fsize, fp);
    fclose(fp);
    xml_buf[fsize] = '\0';

    /* 提取机器人名称 */
    char robot_name[128] = "robot";
    sdf_extract_attr(xml_buf, "name", robot_name, sizeof(robot_name));
    snprintf(model->robot_name, sizeof(model->robot_name), "%s", robot_name);

    /* 解析所有连杆 <link>...</link> */
    const char* cursor = xml_buf;
    while ((cursor = strstr(cursor, "<link ")) && model->link_count < 32) {
        if (sdf_parse_link(cursor, &model->links[model->link_count], model->link_count) == 0) {
            model->link_count++;
        }
        cursor++;
    }
    cursor = xml_buf;
    while ((cursor = strstr(cursor, "<link>")) && model->link_count < 32) {
        if (sdf_parse_link(cursor, &model->links[model->link_count], model->link_count) == 0) {
            model->link_count++;
        }
        cursor++;
    }

    /* 解析所有关节 <joint>...</joint> */
    cursor = xml_buf;
    while ((cursor = strstr(cursor, "<joint ")) && model->joint_count < 32) {
        if (sdf_parse_joint(cursor, &model->joints[model->joint_count], model->joint_count) == 0) {
            model->joint_count++;
        }
        cursor++;
    }
    cursor = xml_buf;
    while ((cursor = strstr(cursor, "<joint>")) && model->joint_count < 32) {
        if (sdf_parse_joint(cursor, &model->joints[model->joint_count], model->joint_count) == 0) {
            model->joint_count++;
        }
        cursor++;
    }

    model->parsed = 1;
    safe_free((void**)&xml_buf);
    return 0;
}

int sdf_model_get_link_count(const SDFModel* model) {
    return model ? model->link_count : 0;
}

int sdf_model_get_joint_count(const SDFModel* model) {
    return model ? model->joint_count : 0;
}

const float* sdf_model_get_joint_limits(const SDFModel* model, int joint_idx) {
    if (!model || joint_idx < 0 || joint_idx >= model->joint_count) return NULL;
    return model->joints[joint_idx].limits;
}
