/**
 * @file ros_robot_controller.c
 * @brief 真实ROS机器人控制器实现 —— 基于rosbridge WebSocket协议的多机器人控制
 *
 * 通过ros_node进行ROS话题发布/订阅，实现对多个机器人的控制。
 * 支持：速度控制、位置控制、关节空间控制、轨迹执行、紧急停止、训练模式。
 *
 * 纯C实现，通过rosbridge WebSocket与ROS系统通信。
 */

#include "selflnn/robot/ros_robot_controller.h"
#include "selflnn/robot/ros_node.h"
#include "selflnn/robot/ros_gazebo_bridge.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _MSC_VER
#pragma warning(disable:4100 4189 4244 4267 4701 4113)
#endif

#define MAX_ROBOTS 64
#define MAX_JOINTS 32

/* 单个机器人状态 */
typedef struct {
    int active;
    int id;
    char name[ROS_ROBOT_CONTROLLER_MAX_NAME];
    RobotType type;
    RosRobotConnectionState conn_state;
    RosRobotControlMode control_mode;
    RosGazeboBridge* gazebo_bridge;
    RosNode* ros_node;

    /* 位姿和速度 */
    float position[3];
    float orientation[4];
    float linear_velocity[3];
    float angular_velocity[3];

    /* 关节状态 */
    float joint_positions[MAX_JOINTS];
    float joint_velocities[MAX_JOINTS];
    float joint_efforts[MAX_JOINTS];
    int num_joints;

    /* 安全 */
    int emergency_stopped;

    /* 训练 */
    int is_training;
    RosRobotTrainingMode training_mode;
    RosRobotTrainingState training_state;
    float training_progress;
    int training_iterations;
    float learning_rate;
    float exploration_rate;
    float discount_factor;
    int training_episode;
    float reward_sum;
    float avg_reward;

    /* 轨迹 */
    int record_trajectory;
    int num_recorded_points;
    float recorded_times[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX];
    float recorded_positions[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX][3];
    float recorded_orientations[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX][4];
    float recorded_joint_positions[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX][MAX_JOINTS];
    int recorded_num_joints;
    int is_recording;
    int is_replaying;
    int replay_index;
    float replay_speed;

    /* 统计 */
    int command_count;
    int error_count;
    double last_heartbeat;
    char last_error[256];
} RobotEntry;

struct RosRobotController {
    RosRobotControllerConfig config;
    RobotEntry robots[MAX_ROBOTS];
    int robot_count;
    char last_error[256];
    int error_code;
};

RosRobotControllerConfig ros_robot_controller_config_default(void) {
    RosRobotControllerConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_robots = ROS_ROBOT_CONTROLLER_MAX_ROBOTS;
    cfg.enable_auto_reconnect = 1;
    cfg.reconnect_interval_ms = 3000;
    cfg.heartbeat_interval_ms = 1000;
    cfg.enable_multi_robot_coordination = 1;
    cfg.enable_training = 1;
    snprintf(cfg.default_ros_master_host, sizeof(cfg.default_ros_master_host), "127.0.0.1");
    cfg.default_ros_master_port = 11311;
    cfg.control_loop_hz = 100.0f;
    return cfg;
}

/* 真实JSON解析回调前向声明 */
static void rrc_joint_states_cb(const void* msg, size_t msg_size, void* user_data);
static void rrc_odom_cb(const void* msg, size_t msg_size, void* user_data);

/* 简易JSON浮点数提取 */
static float rrc_find_float(const char* json, const char* key) {
    const char* p = strstr(json, key);
    if (!p) return 0.0f;
    p += strlen(key);
    while (*p && (*p == ' ' || *p == ':' || *p == '"')) p++;
    if ((*p >= '0' && *p <= '9') || *p == '-' || *p == '.') {
        char* end = NULL;
        return (float)strtod(p, &end);
    }
    return 0.0f;
}

RosRobotController* ros_robot_controller_create(const RosRobotControllerConfig* config) {
    if (!config) return NULL;
    RosRobotController* ctrl = (RosRobotController*)safe_calloc(1, sizeof(RosRobotController));
    if (!ctrl) return NULL;
    ctrl->config = *config;
    ctrl->robot_count = 0;
    ctrl->error_code = 0;
    ctrl->last_error[0] = '\0';
    log_info("[ROS机器人控制器] 已创建，最大机器人: %d, 控制频率: %.1fHz",
             config->max_robots, config->control_loop_hz);
    return ctrl;
}

void ros_robot_controller_destroy(RosRobotController* controller) {
    if (!controller) return;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].gazebo_bridge) {
            ros_gazebo_bridge_destroy(controller->robots[i].gazebo_bridge);
        }
        if (controller->robots[i].ros_node) {
            ros_node_destroy(controller->robots[i].ros_node);
        }
    }
    safe_free((void**)&controller);
}

int ros_robot_controller_add_robot(RosRobotController* controller, const char* name,
                                    RobotType type, const RosGazeboBridgeConfig* bridge_config) {
    if (!controller || !name || controller->robot_count >= controller->config.max_robots)
        return -1;

    int idx = controller->robot_count;
    RobotEntry* robot = &controller->robots[idx];
    memset(robot, 0, sizeof(RobotEntry));
    robot->active = 1;
    robot->id = idx;
    snprintf(robot->name, sizeof(robot->name), "%s", name);
    robot->type = type;
    robot->conn_state = ROS_ROBOT_CONNECTION_STATE_DISCONNECTED;
    robot->control_mode = ROS_ROBOT_CONTROL_MODE_NONE;
    robot->num_joints = 0;
    robot->emergency_stopped = 0;

    /* 创建ROS节点 */
    RosNodeConfig node_cfg = ros_node_config_default(name);
    snprintf(node_cfg.master_host, sizeof(node_cfg.master_host), "%s",
             controller->config.default_ros_master_host);
    node_cfg.master_port = controller->config.default_ros_master_port;
    robot->ros_node = ros_node_create(&node_cfg);

    /* 创建Gazebo桥接 */
    if (bridge_config) {
        robot->gazebo_bridge = ros_gazebo_bridge_create(bridge_config);
    }

    /* 方向四元数初始化为单位四元数 */
    robot->orientation[0] = 0.0f;
    robot->orientation[1] = 0.0f;
    robot->orientation[2] = 0.0f;
    robot->orientation[3] = 1.0f;

    controller->robot_count++;
    log_info("[ROS机器人控制器] 已添加机器人: %s (ID=%d, 类型=%d)", name, idx, (int)type);
    return idx;
}

int ros_robot_controller_remove_robot(RosRobotController* controller, int robot_id) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (robot->gazebo_bridge) ros_gazebo_bridge_destroy(robot->gazebo_bridge);
    if (robot->ros_node) ros_node_destroy(robot->ros_node);
    robot->active = 0;
    return 0;
}

int ros_robot_controller_connect_robot(RosRobotController* controller, int robot_id) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    robot->conn_state = ROS_ROBOT_CONNECTION_STATE_CONNECTING;

/* 使用真实JSON解析回调替代NULL */
    /* 连接ROS节点 */
    if (robot->ros_node) {
        if (ros_node_connect_to_master(robot->ros_node, NULL) == 0) {
            /* 订阅关键话题 */
            char joint_state_topic[256];
            snprintf(joint_state_topic, sizeof(joint_state_topic),
                     "/%s/joint_states", robot->name);
            ros_node_subscribe(robot->ros_node, joint_state_topic,
                              "sensor_msgs/JointState", rrc_joint_states_cb, robot);

            char odom_topic[256];
            snprintf(odom_topic, sizeof(odom_topic), "/%s/odom", robot->name);
            ros_node_subscribe(robot->ros_node, odom_topic,
                              "nav_msgs/Odometry", rrc_odom_cb, robot);

            /* 广告控制话题 */
            char cmd_vel_topic[256];
            snprintf(cmd_vel_topic, sizeof(cmd_vel_topic),
                     "/%s/cmd_vel", robot->name);
            ros_node_advertise(robot->ros_node, cmd_vel_topic,
                              "geometry_msgs/Twist", NULL);

            char joint_cmd_topic[256];
            snprintf(joint_cmd_topic, sizeof(joint_cmd_topic),
                     "/%s/joint_commands", robot->name);
            ros_node_advertise(robot->ros_node, joint_cmd_topic,
                              "trajectory_msgs/JointTrajectory", NULL);
        }
    }

    /* 连接Gazebo桥接 */
    if (robot->gazebo_bridge) {
        ros_gazebo_bridge_connect(robot->gazebo_bridge);
    }

    robot->conn_state = ROS_ROBOT_CONNECTION_STATE_CONNECTED;
    robot->last_heartbeat = (double)time(NULL);
    log_info("[ROS机器人控制器] 机器人 '%s'(ID=%d) 已连接", robot->name, robot_id);
    return 0;
}

int ros_robot_controller_disconnect_robot(RosRobotController* controller, int robot_id) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (robot->ros_node) ros_node_disconnect_from_master(robot->ros_node);
    if (robot->gazebo_bridge) ros_gazebo_bridge_disconnect(robot->gazebo_bridge);
    robot->conn_state = ROS_ROBOT_CONNECTION_STATE_DISCONNECTED;
    return 0;
}

int ros_robot_controller_connect_all(RosRobotController* controller) {
    if (!controller) return -1;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].active)
            ros_robot_controller_connect_robot(controller, i);
    }
    return 0;
}

int ros_robot_controller_disconnect_all(RosRobotController* controller) {
    if (!controller) return -1;
    for (int i = 0; i < controller->robot_count; i++) {
        ros_robot_controller_disconnect_robot(controller, i);
    }
    return 0;
}

int ros_robot_controller_set_control_mode(RosRobotController* controller, int robot_id,
                                           RosRobotControlMode mode) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    controller->robots[robot_id].control_mode = mode;
    return 0;
}

/* 发送速度控制指令 —— 通过 /{robot_name}/cmd_vel 话题 */
int ros_robot_controller_send_velocity(RosRobotController* controller, int robot_id,
                                        float linear_x, float linear_y, float linear_z,
                                        float angular_x, float angular_y, float angular_z) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (robot->emergency_stopped) return -1;

    if (robot->ros_node && robot->conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        char cmd_topic[256];
        snprintf(cmd_topic, sizeof(cmd_topic), "/%s/cmd_vel", robot->name);

        /* 构建geometry_msgs/Twist JSON消息 */
        char json_buf[512];
        snprintf(json_buf, sizeof(json_buf),
                 "{\"linear\":{\"x\":%.4f,\"y\":%.4f,\"z\":%.4f},"
                 "\"angular\":{\"x\":%.4f,\"y\":%.4f,\"z\":%.4f}}",
                 linear_x, linear_y, linear_z,
                 angular_x, angular_y, angular_z);

        int ret = ros_node_publish(robot->ros_node, cmd_topic, json_buf, strlen(json_buf));
        robot->command_count++;
        robot->control_mode = ROS_ROBOT_CONTROL_MODE_VELOCITY;

        /* 更新本地状态缓存 */
        robot->linear_velocity[0] = linear_x;
        robot->linear_velocity[1] = linear_y;
        robot->linear_velocity[2] = linear_z;
        robot->angular_velocity[0] = angular_x;
        robot->angular_velocity[1] = angular_y;
        robot->angular_velocity[2] = angular_z;

        return ret;
    }

    snprintf(robot->last_error, sizeof(robot->last_error), "ROS节点未连接");
    robot->error_count++;
    return -1;
}

/* 发送位姿控制指令 —— 通过 /{robot_name}/move_base_simple/goal 话题 */
int ros_robot_controller_send_position(RosRobotController* controller, int robot_id,
                                        float x, float y, float z,
                                        float ox, float oy, float oz, float ow) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (robot->emergency_stopped) return -1;

    if (robot->ros_node && robot->conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        char goal_topic[256];
        snprintf(goal_topic, sizeof(goal_topic), "/%s/move_base_simple/goal", robot->name);

        char json_buf[512];
        snprintf(json_buf, sizeof(json_buf),
                 "{\"header\":{\"frame_id\":\"map\"},"
                 "\"pose\":{\"position\":{\"x\":%.4f,\"y\":%.4f,\"z\":%.4f},"
                 "\"orientation\":{\"x\":%.4f,\"y\":%.4f,\"z\":%.4f,\"w\":%.4f}}}",
                 x, y, z, ox, oy, oz, ow);

        int ret = ros_node_publish(robot->ros_node, goal_topic, json_buf, strlen(json_buf));
        robot->command_count++;
        robot->control_mode = ROS_ROBOT_CONTROL_MODE_POSITION;
        return ret;
    }

    return -1;
}

/* 发送关节位置指令 */
int ros_robot_controller_send_joint_positions(RosRobotController* controller, int robot_id,
                                                const float* positions, int num_joints) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    if (!positions || num_joints <= 0) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (robot->emergency_stopped) return -1;

    if (robot->ros_node && robot->conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        char jt_topic[256];
        snprintf(jt_topic, sizeof(jt_topic), "/%s/joint_commands", robot->name);

        char json_buf[4096];
        int off = snprintf(json_buf, sizeof(json_buf),
                          "{\"joint_names\":[");
        for (int i = 0; i < num_joints && i < MAX_JOINTS; i++) {
            off += snprintf(json_buf + off, sizeof(json_buf) - off,
                           "%s\"joint_%d\"", (i > 0 ? "," : ""), i + 1);
        }
        off += snprintf(json_buf + off, sizeof(json_buf) - off,
                       "],\"points\":[{\"positions\":[");
        for (int i = 0; i < num_joints && i < MAX_JOINTS; i++) {
            off += snprintf(json_buf + off, sizeof(json_buf) - off,
                           "%s%.6f", (i > 0 ? "," : ""), positions[i]);
        }
        snprintf(json_buf + off, sizeof(json_buf) - off,
                "],\"time_from_start\":{\"secs\":1,\"nsecs\":0}}]}");

        int ret = ros_node_publish(robot->ros_node, jt_topic, json_buf, strlen(json_buf));
        robot->command_count++;
        robot->control_mode = ROS_ROBOT_CONTROL_MODE_JOINT_POSITION;

        /* 更新本地缓存 */
        int copy_count = num_joints < MAX_JOINTS ? num_joints : MAX_JOINTS;
        memcpy(robot->joint_positions, positions, copy_count * sizeof(float));
        robot->num_joints = copy_count;

        return ret;
    }

    return -1;
}

/* 发送关节速度指令 */
int ros_robot_controller_send_joint_velocities(RosRobotController* controller, int robot_id,
                                                 const float* velocities, int num_joints) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    if (!velocities || num_joints <= 0) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (robot->emergency_stopped) return -1;

    if (robot->ros_node && robot->conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        char jv_topic[256];
        snprintf(jv_topic, sizeof(jv_topic), "/%s/joint_velocity_commands", robot->name);

        char json_buf[2048];
        int off = snprintf(json_buf, sizeof(json_buf),
                          "{\"velocities\":[");
        for (int i = 0; i < num_joints && i < MAX_JOINTS; i++) {
            off += snprintf(json_buf + off, sizeof(json_buf) - off,
                           "%s%.6f", (i > 0 ? "," : ""), velocities[i]);
        }
        snprintf(json_buf + off, sizeof(json_buf) - off, "]}");

        int ret = ros_node_publish(robot->ros_node, jv_topic, json_buf, strlen(json_buf));
        robot->command_count++;
        robot->control_mode = ROS_ROBOT_CONTROL_MODE_JOINT_VELOCITY;
        return ret;
    }

    return -1;
}

/* 发送关节力矩指令 */
int ros_robot_controller_send_joint_torques(RosRobotController* controller, int robot_id,
                                              const float* torques, int num_joints) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    if (!torques || num_joints <= 0) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (robot->emergency_stopped) return -1;

    if (robot->ros_node && robot->conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        char jt_topic[256];
        snprintf(jt_topic, sizeof(jt_topic), "/%s/joint_torque_commands", robot->name);

        char json_buf[2048];
        int off = snprintf(json_buf, sizeof(json_buf),
                          "{\"effort\":[");
        for (int i = 0; i < num_joints && i < MAX_JOINTS; i++) {
            off += snprintf(json_buf + off, sizeof(json_buf) - off,
                           "%s%.6f", (i > 0 ? "," : ""), torques[i]);
        }
        snprintf(json_buf + off, sizeof(json_buf) - off, "]}");

        int ret = ros_node_publish(robot->ros_node, jt_topic, json_buf, strlen(json_buf));
        robot->command_count++;
        robot->control_mode = ROS_ROBOT_CONTROL_MODE_JOINT_TORQUE;
        return ret;
    }

    return -1;
}

/* 执行轨迹 */
int ros_robot_controller_execute_trajectory(RosRobotController* controller, int robot_id,
                                             const float* trajectory, int num_points,
                                             float duration) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    if (!trajectory || num_points <= 0) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (robot->emergency_stopped) return -1;

    if (robot->ros_node && robot->conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        char traj_topic[256];
        snprintf(traj_topic, sizeof(traj_topic), "/%s/joint_path_command", robot->name);

        /* 构建JointTrajectory消息， 每个轨迹点包含7个值(pos3+ori4或直接关节值) */
        int points_per_row = 7;
        char json_buf[8192];
        int off = snprintf(json_buf, sizeof(json_buf),
                          "{\"joint_names\":[\"joint_1\",\"joint_2\",\"joint_3\","
                          "\"joint_4\",\"joint_5\",\"joint_6\",\"joint_7\"],"
                          "\"points\":[");

        float dt = duration / (float)num_points;
        for (int j = 0; j < num_points; j++) {
            float t = dt * (float)j;
            off += snprintf(json_buf + off, sizeof(json_buf) - off,
                           "%s{\"positions\":[", (j > 0 ? "," : ""));
            for (int k = 0; k < points_per_row && k < 7; k++) {
                off += snprintf(json_buf + off, sizeof(json_buf) - off,
                               "%s%.6f", (k > 0 ? "," : ""),
                               trajectory[j * points_per_row + k]);
            }
            off += snprintf(json_buf + off, sizeof(json_buf) - off,
                           "],\"time_from_start\":{\"secs\":%d,\"nsecs\":%d}}",
                           (int)t, (int)((t - (int)t) * 1e9f));
        }
        snprintf(json_buf + off, sizeof(json_buf) - off, "]}");

        int ret = ros_node_publish(robot->ros_node, traj_topic, json_buf, strlen(json_buf));
        robot->command_count++;
        robot->control_mode = ROS_ROBOT_CONTROL_MODE_TRAJECTORY;
        return ret;
    }

    return -1;
}

/* 紧急停止单个机器人 */
int ros_robot_controller_emergency_stop(RosRobotController* controller, int robot_id) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    robot->emergency_stopped = 1;

    /* 发送零速度指令 */
    if (robot->ros_node && robot->conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        char cmd_topic[256];
        snprintf(cmd_topic, sizeof(cmd_topic), "/%s/cmd_vel", robot->name);
        const char* stop_msg = "{\"linear\":{\"x\":0,\"y\":0,\"z\":0},"
                               "\"angular\":{\"x\":0,\"y\":0,\"z\":0}}";
        ros_node_publish(robot->ros_node, cmd_topic, stop_msg, strlen(stop_msg));
    }

    log_warning("[ROS机器人控制器] 机器人 '%s'(ID=%d) 紧急停止", robot->name, robot_id);
    return 0;
}

int ros_robot_controller_emergency_stop_all(RosRobotController* controller) {
    if (!controller) return -1;
    for (int i = 0; i < controller->robot_count; i++) {
        ros_robot_controller_emergency_stop(controller, i);
    }
    log_warning("[ROS机器人控制器] 所有机器人紧急停止");
    return 0;
}

int ros_robot_controller_clear_emergency_stop(RosRobotController* controller, int robot_id) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    controller->robots[robot_id].emergency_stopped = 0;
    return 0;
}

/* 获取机器人信息 */
int ros_robot_controller_get_robot_info(RosRobotController* controller, int robot_id,
                                         RosRobotInfo* info) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count || !info) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    memset(info, 0, sizeof(RosRobotInfo));
    info->robot_id = robot->id;
    snprintf(info->robot_name, sizeof(info->robot_name), "%s", robot->name);
    info->robot_type = robot->type;
    info->connection_state = robot->conn_state;
    info->active_control_mode = robot->control_mode;
    info->is_ros_connected = (robot->ros_node && ros_node_is_connected(robot->ros_node));
    info->is_gazebo_connected = (robot->gazebo_bridge &&
                                  ros_gazebo_bridge_is_connected(robot->gazebo_bridge));
    info->last_heartbeat = robot->last_heartbeat;
    info->command_count = robot->command_count;
    info->error_count = robot->error_count;
    memcpy(info->position, robot->position, sizeof(robot->position));
    memcpy(info->orientation, robot->orientation, sizeof(robot->orientation));
    memcpy(info->linear_velocity, robot->linear_velocity, sizeof(robot->linear_velocity));
    memcpy(info->angular_velocity, robot->angular_velocity, sizeof(robot->angular_velocity));
    int jcount = robot->num_joints < 32 ? robot->num_joints : 32;
    memcpy(info->joint_positions, robot->joint_positions, jcount * sizeof(float));
    memcpy(info->joint_velocities, robot->joint_velocities, jcount * sizeof(float));
    memcpy(info->joint_efforts, robot->joint_efforts, jcount * sizeof(float));
    info->num_joints = robot->num_joints;
    info->is_emergency_stopped = robot->emergency_stopped;
    info->is_training = robot->is_training;
    info->training_mode = robot->training_mode;
    info->training_state = robot->training_state;
    info->training_progress = robot->training_progress;
    snprintf(info->last_error, sizeof(info->last_error), "%s", robot->last_error);
    /* P3-001修复: 电池电量默认-1.0表示"未知"，真实值需从硬件传感器读取 */
    info->battery_level = -1.0f;
    return 0;
}

int ros_robot_controller_get_robot_count(RosRobotController* controller) {
    return controller ? controller->robot_count : 0;
}

int ros_robot_controller_get_connected_count(RosRobotController* controller) {
    if (!controller) return 0;
    int count = 0;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].active &&
            controller->robots[i].conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED)
            count++;
    }
    return count;
}

int ros_robot_controller_get_active_count(RosRobotController* controller) {
    if (!controller) return 0;
    int count = 0;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].active &&
            controller->robots[i].conn_state == ROS_ROBOT_CONNECTION_STATE_ACTIVE)
            count++;
    }
    return count;
}

int ros_robot_controller_get_gazebo_bridge(RosRobotController* controller, int robot_id,
                                            RosGazeboBridge** bridge) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count || !bridge) return -1;
    *bridge = controller->robots[robot_id].gazebo_bridge;
    return 0;
}

int ros_robot_controller_get_bridge(RosRobotController* controller, int robot_id,
                                     RosGazeboBridge** bridge_out) {
    return ros_robot_controller_get_gazebo_bridge(controller, robot_id, bridge_out);
}

/* 训练相关 */
int ros_robot_controller_start_training(RosRobotController* controller, int robot_id,
                                         RosRobotTrainingMode mode) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    robot->is_training = 1;
    robot->training_mode = mode;
    robot->training_state = ROS_ROBOT_TRAINING_STATE_TRAINING;
    robot->training_progress = 0.0f;
    robot->training_episode = 0;
    robot->reward_sum = 0.0f;
    log_info("[ROS机器人控制器] 机器人 '%s' 开始训练 (模式=%d)", robot->name, (int)mode);
    return 0;
}

int ros_robot_controller_stop_training(RosRobotController* controller, int robot_id) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    robot->is_training = 0;
    robot->training_state = ROS_ROBOT_TRAINING_STATE_COMPLETED;
    return 0;
}

int ros_robot_controller_pause_training(RosRobotController* controller, int robot_id) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    controller->robots[robot_id].training_state = ROS_ROBOT_TRAINING_STATE_IDLE;
    return 0;
}

int ros_robot_controller_resume_training(RosRobotController* controller, int robot_id) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    controller->robots[robot_id].training_state = ROS_ROBOT_TRAINING_STATE_TRAINING;
    return 0;
}

/* 轨迹录制与回放 */
int ros_robot_controller_record_trajectory(RosRobotController* controller, int robot_id,
                                            int start_recording) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    if (start_recording) {
        robot->is_recording = 1;
        robot->num_recorded_points = 0;
    } else {
        robot->is_recording = 0;
    }
    return 0;
}

int ros_robot_controller_replay_trajectory(RosRobotController* controller, int robot_id,
                                            float replay_speed) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    robot->is_replaying = 1;
    robot->replay_index = 0;
    robot->replay_speed = replay_speed > 0.0f ? replay_speed : 1.0f;
    return 0;
}

int ros_robot_controller_set_training_params(RosRobotController* controller, int robot_id,
                                              float learning_rate, float exploration_rate,
                                              float discount_factor, int num_iterations) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    robot->learning_rate = learning_rate;
    robot->exploration_rate = exploration_rate;
    robot->discount_factor = discount_factor;
    robot->training_iterations = num_iterations;
    return 0;
}

int ros_robot_controller_get_training_session(RosRobotController* controller, int robot_id,
                                                RosRobotTrainingSession* session) {
    if (!controller || robot_id < 0 || robot_id >= controller->robot_count || !session) return -1;
    RobotEntry* robot = &controller->robots[robot_id];
    memset(session, 0, sizeof(RosRobotTrainingSession));
    session->record_trajectory = robot->record_trajectory;
    session->num_recorded_points = robot->num_recorded_points;
    session->max_recorded_points = ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX;
    memcpy(session->recorded_times, robot->recorded_times, sizeof(robot->recorded_times));
    memcpy(session->recorded_positions, robot->recorded_positions, sizeof(robot->recorded_positions));
    memcpy(session->recorded_orientations, robot->recorded_orientations, sizeof(robot->recorded_orientations));
    memcpy(session->recorded_joint_positions, robot->recorded_joint_positions, sizeof(robot->recorded_joint_positions));
    session->recorded_num_joints = robot->recorded_num_joints;
    int jcount = robot->num_joints < 32 ? robot->num_joints : 32;
    memcpy(session->joint_positions, robot->joint_positions, jcount * sizeof(float));
    memcpy(session->joint_velocities, robot->joint_velocities, jcount * sizeof(float));
    memcpy(session->joint_efforts, robot->joint_efforts, jcount * sizeof(float));
    session->num_joints = robot->num_joints;
    session->replay_speed = robot->replay_speed;
    session->is_recording = robot->is_recording;
    session->is_replaying = robot->is_replaying;
    session->replay_index = robot->replay_index;
    session->training_iterations = robot->training_iterations;
    session->learning_rate = robot->learning_rate;
    session->exploration_rate = robot->exploration_rate;
    session->discount_factor = robot->discount_factor;
    session->training_episode = robot->training_episode;
    session->reward_sum = robot->reward_sum;
    session->avg_reward = robot->avg_reward;
    return 0;
}

/* 控制循环更新 */
int ros_robot_controller_update(RosRobotController* controller) {
    if (!controller) return -1;
    for (int i = 0; i < controller->robot_count; i++) {
        RobotEntry* robot = &controller->robots[i];
        if (!robot->active) continue;
        if (robot->ros_node && robot->conn_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
            ros_node_spin_once(robot->ros_node);
        }
    }
    return 0;
}

int ros_robot_controller_spin(RosRobotController* controller, int timeout_ms) {
    if (!controller) return -1;
    int total = 0;
    for (int i = 0; i < controller->robot_count; i++) {
        RobotEntry* robot = &controller->robots[i];
        if (!robot->active) continue;
        if (robot->ros_node) {
            total += ros_node_spin(robot->ros_node, timeout_ms);
        }
    }
    return total;
}

const char* ros_robot_controller_get_last_error(RosRobotController* controller) {
    return controller ? controller->last_error : "";
}

int ros_robot_controller_get_error_code(RosRobotController* controller) {
    return controller ? controller->error_code : -1;
}

/* 真实JSON解析回调 — 基于RobotEntry结构体直接访问 */

static void rrc_joint_states_cb(const void* msg, size_t msg_size, void* user_data) {
    RobotEntry* robot = (RobotEntry*)user_data;
    if (!robot || !msg || msg_size == 0 || !robot->active) return;

    const char* json = (const char*)msg;
    robot->num_joints = 0;
    const char* pp = strstr(json, "\"position\"");
    if (pp) {
        const char* start = strstr(pp, "[");
        if (start) {
            start++;
            for (int i = 0; i < MAX_JOINTS && *start; i++) {
                while (*start == ' ' || *start == ',') start++;
                if (*start == ']' || *start == 0) break;
                char* end = NULL;
                robot->joint_positions[i] = (float)strtod(start, &end);
                if (end == start) break;
                start = end;
                robot->num_joints = i + 1;
            }
        }
    }
    const char* vp = strstr(json, "\"velocity\"");
    if (vp) {
        const char* start = strstr(vp, "[");
        if (start) {
            start++;
            for (int i = 0; i < robot->num_joints && *start; i++) {
                while (*start == ' ' || *start == ',') start++;
                if (*start == ']' || *start == 0) break;
                char* end = NULL;
                robot->joint_velocities[i] = (float)strtod(start, &end);
                if (end == start) break;
                start = end;
            }
        }
    }
    const char* ep = strstr(json, "\"effort\"");
    if (ep) {
        const char* start = strstr(ep, "[");
        if (start) {
            start++;
            for (int i = 0; i < robot->num_joints && *start; i++) {
                while (*start == ' ' || *start == ',') start++;
                if (*start == ']' || *start == 0) break;
                char* end = NULL;
                robot->joint_efforts[i] = (float)strtod(start, &end);
                if (end == start) break;
                start = end;
            }
        }
    }
}

static void rrc_odom_cb(const void* msg, size_t msg_size, void* user_data) {
    RobotEntry* robot = (RobotEntry*)user_data;
    if (!robot || !msg || msg_size == 0 || !robot->active) return;

    const char* json = (const char*)msg;

/* 修复JSON解析偏移bug
     * rrc_find_float每次都从字符串开头搜索，而Odometry消息中
     * position/orientation/linear/angular都有相同的"x":"y":"z":子字段，
     * 导致不同区域读取到同一个值。修复：先用strstr定位到各区域，
     * 再从该区域的起始位置开始解析对应字段。 */

    /* 解析position区域: 定位到"position"后解析x/y/z */
    const char* pos_ptr = strstr(json, "\"position\"");
    if (pos_ptr) {
        robot->position[0] = rrc_find_float(pos_ptr, "\"x\":");
        robot->position[1] = rrc_find_float(pos_ptr, "\"y\":");
        robot->position[2] = rrc_find_float(pos_ptr, "\"z\":");
    }

    /* 解析orientation区域: 定位到"orientation"后解析x/y/z/w */
    const char* ori_ptr = strstr(json, "\"orientation\"");
    if (ori_ptr) {
        robot->orientation[0] = rrc_find_float(ori_ptr, "\"x\":");
        robot->orientation[1] = rrc_find_float(ori_ptr, "\"y\":");
        robot->orientation[2] = rrc_find_float(ori_ptr, "\"z\":");
        robot->orientation[3] = rrc_find_float(ori_ptr, "\"w\":");
    }

    /* 解析linear速度区域: 定位到"linear"后解析x/y/z */
    const char* lin_ptr = strstr(json, "\"linear\"");
    if (lin_ptr) {
        robot->linear_velocity[0] = rrc_find_float(lin_ptr, "\"x\":");
        robot->linear_velocity[1] = rrc_find_float(lin_ptr, "\"y\":");
        robot->linear_velocity[2] = rrc_find_float(lin_ptr, "\"z\":");
    }

    /* 解析angular速度区域: 定位到"angular"后解析x/y/z */
    const char* ang_ptr = strstr(json, "\"angular\"");
    if (ang_ptr) {
        robot->angular_velocity[0] = rrc_find_float(ang_ptr, "\"x\":");
        robot->angular_velocity[1] = rrc_find_float(ang_ptr, "\"y\":");
        robot->angular_velocity[2] = rrc_find_float(ang_ptr, "\"z\":");
    }

    robot->last_heartbeat = (double)time(NULL);
}
