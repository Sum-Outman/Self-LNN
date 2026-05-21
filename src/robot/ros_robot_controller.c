#include "selflnn/robot/ros_robot_controller.h"
#include "selflnn/robot/ros_gazebo_bridge.h"
#include "selflnn/core/port_config.h"
#include "selflnn/utils/memory_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define ROS_ROBOT_CONTROLLER_MAX_ERROR 256
#define ROS_ROBOT_CONTROLLER_SPIN_INTERVAL_MS 50

typedef struct {
    int active;
    char name[ROS_ROBOT_CONTROLLER_MAX_NAME];
    RobotType type;
    RosRobotInfo info;
    RosGazeboBridge* bridge;
    RosGazeboBridgeConfig bridge_config;
    RosRobotTrainingSession training;
    int bridge_created;
    int bridge_started;
    double last_update_time;
    double last_heartbeat_time;
    int consecutive_timeouts;
} RosRobotEntry;

typedef struct RosRobotController {
    RosRobotControllerConfig config;
    RosRobotEntry robots[ROS_ROBOT_CONTROLLER_MAX_ROBOTS];
    int robot_count;
    double last_spin_time;
    int running;
    char last_error[ROS_ROBOT_CONTROLLER_MAX_ERROR];
    int error_code;
} RosRobotController;

static double controller_get_time(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void controller_set_error(RosRobotController* ctrl, int code, const char* msg) {
    if (!ctrl) return;
    strncpy(ctrl->last_error, msg, sizeof(ctrl->last_error) - 1);
    ctrl->last_error[sizeof(ctrl->last_error) - 1] = '\0';
    ctrl->error_code = code;
}

static int find_robot_entry(RosRobotController* ctrl, int robot_id) {
    if (!ctrl || robot_id < 0 || robot_id >= ctrl->robot_count) return -1;
    if (!ctrl->robots[robot_id].active) return -1;
    return robot_id;
}

static int find_or_create_ros_bridge(RosRobotController* ctrl, int robot_id) {
    RosRobotEntry* entry = &ctrl->robots[robot_id];
    if (!entry->active) return -1;
    if (entry->bridge_created && entry->bridge) return 0;

    if (entry->bridge) {
        ros_gazebo_bridge_destroy(entry->bridge);
        entry->bridge = NULL;
    }

    RosGazeboBridgeConfig bridge_cfg;
    memcpy(&bridge_cfg, &entry->bridge_config, sizeof(RosGazeboBridgeConfig));
    snprintf(bridge_cfg.node_name, sizeof(bridge_cfg.node_name), "selflnn_%s", entry->name);
    strncpy(bridge_cfg.ros_master_host, ctrl->config.default_ros_master_host,
            sizeof(bridge_cfg.ros_master_host) - 1);
    bridge_cfg.ros_master_port = ctrl->config.default_ros_master_port;

    entry->bridge = ros_gazebo_bridge_create(&bridge_cfg);
    if (!entry->bridge) {
        controller_set_error(ctrl, -1, "创建ROS-Gazebo桥接失败");
        entry->bridge_created = 0;
        return -1;
    }

    entry->bridge_created = 1;
    return 0;
}

RosRobotControllerConfig ros_robot_controller_config_default(void) {
    RosRobotControllerConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_robots = ROS_ROBOT_CONTROLLER_MAX_ROBOTS;
    cfg.enable_auto_reconnect = 1;
    cfg.reconnect_interval_ms = 3000;
    cfg.heartbeat_interval_ms = 1000;
    cfg.enable_multi_robot_coordination = 1;
    cfg.enable_training = 1;
    strncpy(cfg.default_ros_master_host, "127.0.0.1", sizeof(cfg.default_ros_master_host) - 1);
    cfg.default_ros_master_port = SELFLNN_ROS_MASTER_PORT;
    cfg.control_loop_hz = 50.0f;
    return cfg;
}

RosRobotController* ros_robot_controller_create(const RosRobotControllerConfig* config) {
    if (!config) return NULL;

    RosRobotController* ctrl = (RosRobotController*)safe_malloc(sizeof(RosRobotController));
    if (!ctrl) return NULL;
    memset(ctrl, 0, sizeof(RosRobotController));

    if (config) {
        memcpy(&ctrl->config, config, sizeof(RosRobotControllerConfig));
    } else {
        ctrl->config = ros_robot_controller_config_default();
    }

    ctrl->robot_count = 0;
    ctrl->running = 1;
    ctrl->last_spin_time = controller_get_time();
    ctrl->last_error[0] = '\0';
    ctrl->error_code = 0;

    for (int i = 0; i < ROS_ROBOT_CONTROLLER_MAX_ROBOTS; i++) {
        ctrl->robots[i].active = 0;
        ctrl->robots[i].bridge = NULL;
        ctrl->robots[i].bridge_created = 0;
        ctrl->robots[i].bridge_started = 0;
    }

    return ctrl;
}

void ros_robot_controller_destroy(RosRobotController* controller) {
    if (!controller) return;
    controller->running = 0;

    for (int i = 0; i < controller->robot_count; i++) {
        RosRobotEntry* entry = &controller->robots[i];
        if (entry->bridge) {
            ros_gazebo_bridge_disconnect((RosGazeboBridge*)entry->bridge);
            ros_gazebo_bridge_destroy(entry->bridge);
            entry->bridge = NULL;
        }
        entry->active = 0;
    }

    controller->robot_count = 0;
    safe_free((void**)&controller);
}

int ros_robot_controller_add_robot(RosRobotController* controller, const char* name,
                                    RobotType type, const RosGazeboBridgeConfig* bridge_config) {
    if (!controller || !name) return -1;
    if (controller->robot_count >= controller->config.max_robots) {
        controller_set_error(controller, -1, "机器人数量已达上限");
        return -1;
    }

    int id = controller->robot_count;
    RosRobotEntry* entry = &controller->robots[id];
    memset(entry, 0, sizeof(RosRobotEntry));

    entry->active = 1;
    strncpy(entry->name, name, sizeof(entry->name) - 1);
    entry->type = type;

    if (bridge_config) {
        memcpy(&entry->bridge_config, bridge_config, sizeof(RosGazeboBridgeConfig));
    } else {
        entry->bridge_config = ros_gazebo_bridge_config_default(name);
    }

    memset(&entry->info, 0, sizeof(RosRobotInfo));
    entry->info.robot_id = id;
    strncpy(entry->info.robot_name, name, sizeof(entry->info.robot_name) - 1);
    entry->info.robot_type = type;
    entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_DISCONNECTED;
    entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_NONE;
    entry->info.is_ros_connected = 0;
    entry->info.is_gazebo_connected = 0;
    entry->info.last_heartbeat = controller_get_time();
    entry->info.battery_level = 100.0f;
    entry->info.num_joints = 0;
    entry->info.is_emergency_stopped = 0;
    entry->info.is_training = 0;
    entry->info.training_mode = ROS_ROBOT_TRAINING_MODE_NONE;
    entry->info.training_state = ROS_ROBOT_TRAINING_STATE_IDLE;
    entry->info.training_progress = 0.0f;
    entry->info.orientation[3] = 1.0f;

    memset(&entry->training, 0, sizeof(RosRobotTrainingSession));
    entry->training.replay_speed = 1.0f;
    entry->training.learning_rate = 0.001f;
    entry->training.exploration_rate = 0.1f;
    entry->training.discount_factor = 0.95f;
    entry->training.training_iterations = 1000;

    entry->last_update_time = controller_get_time();
    entry->last_heartbeat_time = entry->last_update_time;
    entry->consecutive_timeouts = 0;

    controller->robot_count++;
    return id;
}

int ros_robot_controller_remove_robot(RosRobotController* controller, int robot_id) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->bridge) {
        ros_gazebo_bridge_disconnect((RosGazeboBridge*)entry->bridge);
        ros_gazebo_bridge_destroy(entry->bridge);
        entry->bridge = NULL;
    }
    entry->active = 0;
    entry->bridge_created = 0;

    if (idx == controller->robot_count - 1) {
        controller->robot_count--;
    }

    return 0;
}

int ros_robot_controller_connect_robot(RosRobotController* controller, int robot_id) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.connection_state == ROS_ROBOT_CONNECTION_STATE_CONNECTED ||
        entry->info.connection_state == ROS_ROBOT_CONNECTION_STATE_ACTIVE) {
        return 0;
    }

    entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_CONNECTING;

    if (find_or_create_ros_bridge(controller, idx) != 0) {
        entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_ERROR;
        strncpy(entry->info.last_error, "创建ROS-Gazebo桥接失败", sizeof(entry->info.last_error) - 1);
        return -1;
    }

    if (ros_gazebo_bridge_connect((RosGazeboBridge*)entry->bridge) != 0) {
        entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_ERROR;
        strncpy(entry->info.last_error, "ROS-Gazebo桥接连失败", sizeof(entry->info.last_error) - 1);
        return -1;
    }

    float init_pose[7] = {0, 0, 0, 0, 0, 0, 1};
    char model_name[ROS_ROBOT_CONTROLLER_MAX_NAME + 16];
    snprintf(model_name, sizeof(model_name), "%s_model", entry->name);

    if (ros_gazebo_bridge_spawn_model((RosGazeboBridge*)entry->bridge, model_name, init_pose, &init_pose[3]) != 0) {
        controller_set_error(controller, -1, "在Gazebo中生成机器人模型失败");
    }

    if (ros_gazebo_bridge_start((RosGazeboBridge*)entry->bridge) == 0) {
        entry->bridge_started = 1;
    }

    entry->info.is_ros_connected = 1;
    entry->info.is_gazebo_connected = 1;
    entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_CONNECTED;
    entry->info.last_heartbeat = controller_get_time();
    entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_VELOCITY;
    entry->last_heartbeat_time = entry->info.last_heartbeat;
    entry->consecutive_timeouts = 0;

    return 0;
}

int ros_robot_controller_disconnect_robot(RosRobotController* controller, int robot_id) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.connection_state == ROS_ROBOT_CONNECTION_STATE_DISCONNECTED) return 0;

    if (entry->bridge) {
        if (entry->bridge_started) {
            ros_gazebo_bridge_stop((RosGazeboBridge*)entry->bridge);
            entry->bridge_started = 0;
        }

        char model_name[ROS_ROBOT_CONTROLLER_MAX_NAME + 16];
        snprintf(model_name, sizeof(model_name), "%s_model", entry->name);
        ros_gazebo_bridge_delete_model((RosGazeboBridge*)entry->bridge, model_name);
        ros_gazebo_bridge_disconnect((RosGazeboBridge*)entry->bridge);
        ros_gazebo_bridge_destroy(entry->bridge);
        entry->bridge = NULL;
        entry->bridge_created = 0;
    }

    entry->info.is_ros_connected = 0;
    entry->info.is_gazebo_connected = 0;
    entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_DISCONNECTED;

    return 0;
}

int ros_robot_controller_connect_all(RosRobotController* controller) {
    if (!controller) return -1;
    int success_count = 0;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].active) {
            if (ros_robot_controller_connect_robot(controller, i) == 0) {
                success_count++;
            }
        }
    }
    return success_count;
}

int ros_robot_controller_disconnect_all(RosRobotController* controller) {
    if (!controller) return -1;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].active) {
            ros_robot_controller_disconnect_robot(controller, i);
        }
    }
    return 0;
}

int ros_robot_controller_set_control_mode(RosRobotController* controller, int robot_id,
                                           RosRobotControlMode mode) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;
    controller->robots[idx].info.active_control_mode = mode;
    return 0;
}

int ros_robot_controller_send_velocity(RosRobotController* controller, int robot_id,
                                        float linear_x, float linear_y, float linear_z,
                                        float angular_x, float angular_y, float angular_z) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.connection_state < ROS_ROBOT_CONNECTION_STATE_CONNECTED) return -1;
    if (!entry->bridge) return -1;

    RosTwist cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.linear.x = linear_x;
    cmd.linear.y = linear_y;
    cmd.linear.z = linear_z;
    cmd.angular.x = angular_x;
    cmd.angular.y = angular_y;
    cmd.angular.z = angular_z;

    int ret = ros_gazebo_bridge_publish_cmd_vel((RosGazeboBridge*)entry->bridge, &cmd);
    if (ret == 0) {
        entry->info.command_count++;
        entry->info.linear_velocity[0] = linear_x;
        entry->info.linear_velocity[1] = linear_y;
        entry->info.linear_velocity[2] = linear_z;
        entry->info.angular_velocity[0] = angular_x;
        entry->info.angular_velocity[1] = angular_y;
        entry->info.angular_velocity[2] = angular_z;
        entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_VELOCITY;
    }
    return ret;
}

int ros_robot_controller_send_position(RosRobotController* controller, int robot_id,
                                        float x, float y, float z,
                                        float ox, float oy, float oz, float ow) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.connection_state < ROS_ROBOT_CONNECTION_STATE_CONNECTED) return -1;
    if (!entry->bridge) return -1;

    float pos[3] = {x, y, z};
    float ori[4] = {ox, oy, oz, ow};
    entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_POSITION;
    return ros_gazebo_bridge_set_model_state((RosGazeboBridge*)entry->bridge, entry->name, pos, ori);
}

int ros_robot_controller_send_joint_positions(RosRobotController* controller, int robot_id,
                                                const float* positions, int num_joints) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.connection_state < ROS_ROBOT_CONNECTION_STATE_CONNECTED) return -1;
    if (!entry->bridge) return -1;

    int ret = 0;
    for (int j = 0; j < num_joints && j < 32; j++) {
        char joint_name[64];
        snprintf(joint_name, sizeof(joint_name), "joint_%d", j);
        if (ros_gazebo_bridge_set_joint_position((RosGazeboBridge*)entry->bridge, 0, joint_name, positions[j]) != 0) {
            ret = -1;
        }
        entry->info.joint_positions[j] = positions[j];
    }
    entry->info.num_joints = num_joints;
    entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_JOINT_POSITION;
    return ret;
}

int ros_robot_controller_send_joint_velocities(RosRobotController* controller, int robot_id,
                                                 const float* velocities, int num_joints) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.connection_state < ROS_ROBOT_CONNECTION_STATE_CONNECTED) return -1;
    if (!entry->bridge) return -1;

    int ret = 0;
    for (int j = 0; j < num_joints && j < 32; j++) {
        char joint_name[64];
        snprintf(joint_name, sizeof(joint_name), "joint_%d", j);
        if (ros_gazebo_bridge_set_joint_velocity((RosGazeboBridge*)entry->bridge, 0, joint_name, velocities[j]) != 0) {
            ret = -1;
        }
        entry->info.joint_velocities[j] = velocities[j];
    }
    entry->info.num_joints = num_joints;
    entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_JOINT_VELOCITY;
    return ret;
}

int ros_robot_controller_send_joint_torques(RosRobotController* controller, int robot_id,
                                              const float* torques, int num_joints) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.connection_state < ROS_ROBOT_CONNECTION_STATE_CONNECTED) return -1;
    if (!entry->bridge) return -1;

    int ret = 0;
    for (int j = 0; j < num_joints && j < 32; j++) {
        char joint_name[64];
        snprintf(joint_name, sizeof(joint_name), "joint_%d", j);
        if (ros_gazebo_bridge_set_joint_torque((RosGazeboBridge*)entry->bridge, 0, joint_name, torques[j]) != 0) {
            ret = -1;
        }
        entry->info.joint_efforts[j] = torques[j];
    }
    entry->info.num_joints = num_joints;
    entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_JOINT_TORQUE;
    return ret;
}

int ros_robot_controller_execute_trajectory(RosRobotController* controller, int robot_id,
                                             const float* trajectory, int num_points,
                                             float duration) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.connection_state < ROS_ROBOT_CONNECTION_STATE_CONNECTED) return -1;

    if (num_points > 0 && num_points <= ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX) {
        entry->training.num_recorded_points = num_points;
        for (int i = 0; i < num_points && i < ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX; i++) {
            entry->training.recorded_times[i] = duration * (float)i / (float)num_points;
            if (trajectory) {
                entry->training.recorded_positions[i][0] = trajectory[i * 7 + 0];
                entry->training.recorded_positions[i][1] = trajectory[i * 7 + 1];
                entry->training.recorded_positions[i][2] = trajectory[i * 7 + 2];
                entry->training.recorded_orientations[i][0] = trajectory[i * 7 + 3];
                entry->training.recorded_orientations[i][1] = trajectory[i * 7 + 4];
                entry->training.recorded_orientations[i][2] = trajectory[i * 7 + 5];
                entry->training.recorded_orientations[i][3] = trajectory[i * 7 + 6];
            }
        }
        entry->training.is_replaying = 1;
        entry->training.replay_index = 0;
    }

    entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_TRAJECTORY;
    return 0;
}

int ros_robot_controller_emergency_stop(RosRobotController* controller, int robot_id) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];

    RosTwist stop_cmd;
    memset(&stop_cmd, 0, sizeof(stop_cmd));

    if (entry->bridge) {
        ros_gazebo_bridge_publish_cmd_vel((RosGazeboBridge*)entry->bridge, &stop_cmd);
        ros_gazebo_bridge_pause((RosGazeboBridge*)entry->bridge);
    }

    entry->info.is_emergency_stopped = 1;
    entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_CONNECTED;
    entry->info.linear_velocity[0] = 0;
    entry->info.linear_velocity[1] = 0;
    entry->info.linear_velocity[2] = 0;
    entry->info.angular_velocity[0] = 0;
    entry->info.angular_velocity[1] = 0;
    entry->info.angular_velocity[2] = 0;

    return 0;
}

int ros_robot_controller_emergency_stop_all(RosRobotController* controller) {
    if (!controller) return -1;
    int ret = 0;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].active) {
            if (ros_robot_controller_emergency_stop(controller, i) != 0) {
                ret = -1;
            }
        }
    }
    return ret;
}

int ros_robot_controller_clear_emergency_stop(RosRobotController* controller, int robot_id) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    entry->info.is_emergency_stopped = 0;

    if (entry->bridge) {
        ros_gazebo_bridge_resume((RosGazeboBridge*)entry->bridge);
    }

    return 0;
}

int ros_robot_controller_get_robot_info(RosRobotController* controller, int robot_id,
                                         RosRobotInfo* info) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0 || !info) return -1;

    RosRobotEntry* entry = &controller->robots[idx];

    if (entry->bridge && entry->info.connection_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        float pos[3] = {0}, ori[4] = {0,0,0,1}, lin_vel[3] = {0}, ang_vel[3] = {0};
        char model_name[ROS_ROBOT_CONTROLLER_MAX_NAME + 16];
        snprintf(model_name, sizeof(model_name), "%s_model", entry->name);

        if (ros_gazebo_bridge_get_model_state((RosGazeboBridge*)entry->bridge, model_name,
                                                pos, ori, lin_vel, ang_vel) == 0) {
            entry->info.position[0] = pos[0];
            entry->info.position[1] = pos[1];
            entry->info.position[2] = pos[2];
            entry->info.orientation[0] = ori[0];
            entry->info.orientation[1] = ori[1];
            entry->info.orientation[2] = ori[2];
            entry->info.orientation[3] = ori[3];
            entry->info.linear_velocity[0] = lin_vel[0];
            entry->info.linear_velocity[1] = lin_vel[1];
            entry->info.linear_velocity[2] = lin_vel[2];
            entry->info.angular_velocity[0] = ang_vel[0];
            entry->info.angular_velocity[1] = ang_vel[1];
            entry->info.angular_velocity[2] = ang_vel[2];
        }

        float positions[32] = {0}, velocities[32] = {0}, efforts[32] = {0};
        int n = ros_gazebo_bridge_get_joint_state((RosGazeboBridge*)entry->bridge, 0,
                                                    positions, velocities, efforts, 32);
        if (n > 0) {
            entry->info.num_joints = n;
            memcpy(entry->info.joint_positions, positions, n * sizeof(float));
            memcpy(entry->info.joint_velocities, velocities, n * sizeof(float));
            memcpy(entry->info.joint_efforts, efforts, n * sizeof(float));
        }

        RosGazeboWorldState world_state;
        memset(&world_state, 0, sizeof(world_state));
        if (ros_gazebo_bridge_get_world_state((RosGazeboBridge*)entry->bridge, &world_state) == 0) {
            if (world_state.paused) {
                entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_CONNECTED;
            } else {
                entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_ACTIVE;
            }
        }
    }

    memcpy(info, &entry->info, sizeof(RosRobotInfo));
    return 0;
}

int ros_robot_controller_get_robot_count(RosRobotController* controller) {
    if (!controller) return 0;
    return controller->robot_count;
}

int ros_robot_controller_get_connected_count(RosRobotController* controller) {
    if (!controller) return 0;
    int count = 0;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].active &&
            controller->robots[i].info.connection_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
            count++;
        }
    }
    return count;
}

int ros_robot_controller_get_active_count(RosRobotController* controller) {
    if (!controller) return 0;
    int count = 0;
    for (int i = 0; i < controller->robot_count; i++) {
        if (controller->robots[i].active &&
            controller->robots[i].info.connection_state == ROS_ROBOT_CONNECTION_STATE_ACTIVE) {
            count++;
        }
    }
    return count;
}

int ros_robot_controller_get_gazebo_bridge(RosRobotController* controller, int robot_id,
                                            RosGazeboBridge** bridge) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0 || !bridge) return -1;
    *bridge = controller->robots[idx].bridge;
    return (*bridge != NULL) ? 0 : -1;
}

int ros_robot_controller_get_bridge(RosRobotController* controller, int robot_id,
                                     RosGazeboBridge** bridge_out) {
    return ros_robot_controller_get_gazebo_bridge(controller, robot_id, bridge_out);
}

int ros_robot_controller_start_training(RosRobotController* controller, int robot_id,
                                         RosRobotTrainingMode mode) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (!controller->config.enable_training) {
        controller_set_error(controller, -1, "训练功能未启用");
        return -1;
    }

    if (entry->info.connection_state < ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
        controller_set_error(controller, -1, "机器人未连接，无法开始训练");
        return -1;
    }

    entry->info.is_training = 1;
    entry->info.training_mode = mode;
    entry->info.training_state = ROS_ROBOT_TRAINING_STATE_TRAINING;
    entry->info.training_progress = 0.0f;

    memset(&entry->training, 0, sizeof(RosRobotTrainingSession));
    entry->training.replay_speed = 1.0f;
    entry->training.learning_rate = 0.001f;
    entry->training.exploration_rate = 0.1f;
    entry->training.discount_factor = 0.95f;
    entry->training.training_iterations = 1000;
    entry->training.is_recording = 0;
    entry->training.is_replaying = 0;
    entry->training.num_recorded_points = 0;

    entry->info.active_control_mode = ROS_ROBOT_CONTROL_MODE_TRAINING;
    return 0;
}

int ros_robot_controller_stop_training(RosRobotController* controller, int robot_id) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    entry->info.is_training = 0;
    entry->info.training_state = ROS_ROBOT_TRAINING_STATE_IDLE;
    entry->training.is_recording = 0;
    entry->training.is_replaying = 0;

    return 0;
}

int ros_robot_controller_pause_training(RosRobotController* controller, int robot_id) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;
    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.training_state == ROS_ROBOT_TRAINING_STATE_TRAINING) {
        entry->info.training_state = ROS_ROBOT_TRAINING_STATE_IDLE;
        if (entry->bridge) {
            ros_gazebo_bridge_pause((RosGazeboBridge*)entry->bridge);
        }
    }
    return 0;
}

int ros_robot_controller_resume_training(RosRobotController* controller, int robot_id) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;
    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->info.is_training) {
        entry->info.training_state = ROS_ROBOT_TRAINING_STATE_TRAINING;
        if (entry->bridge) {
            ros_gazebo_bridge_resume((RosGazeboBridge*)entry->bridge);
        }
    }
    return 0;
}

int ros_robot_controller_record_trajectory(RosRobotController* controller, int robot_id,
                                            int start_recording) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];

    if (start_recording) {
        memset(&entry->training, 0, sizeof(RosRobotTrainingSession));
        entry->training.is_recording = 1;
        entry->training.is_replaying = 0;
        entry->training.num_recorded_points = 0;
        entry->training.replay_speed = 1.0f;
        entry->training.learning_rate = 0.001f;
        entry->training.exploration_rate = 0.1f;
        entry->training.discount_factor = 0.95f;
        entry->training.training_iterations = 1000;
        entry->info.training_state = ROS_ROBOT_TRAINING_STATE_RECORDING;
    } else {
        entry->training.is_recording = 0;
        entry->info.training_state = ROS_ROBOT_TRAINING_STATE_COMPLETED;
    }

    return 0;
}

int ros_robot_controller_replay_trajectory(RosRobotController* controller, int robot_id,
                                            float replay_speed) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (entry->training.num_recorded_points == 0) {
        controller_set_error(controller, -1, "没有录制的轨迹可回放");
        return -1;
    }

    entry->training.is_replaying = 1;
    entry->training.is_recording = 0;
    entry->training.replay_index = 0;
    entry->training.replay_speed = (replay_speed > 0.0f) ? replay_speed : 1.0f;
    entry->info.training_state = ROS_ROBOT_TRAINING_STATE_EVALUATING;

    return 0;
}

int ros_robot_controller_set_training_params(RosRobotController* controller, int robot_id,
                                              float learning_rate, float exploration_rate,
                                              float discount_factor, int num_iterations) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0) return -1;

    RosRobotEntry* entry = &controller->robots[idx];
    if (learning_rate > 0.0f) entry->training.learning_rate = learning_rate;
    if (exploration_rate >= 0.0f) entry->training.exploration_rate = exploration_rate;
    if (discount_factor >= 0.0f) entry->training.discount_factor = discount_factor;
    if (num_iterations > 0) entry->training.training_iterations = num_iterations;

    return 0;
}

int ros_robot_controller_get_training_session(RosRobotController* controller, int robot_id,
                                                RosRobotTrainingSession* session) {
    int idx = find_robot_entry(controller, robot_id);
    if (idx < 0 || !session) return -1;

    RosRobotEntry* entry = &controller->robots[idx];

    if (entry->bridge && entry->training.is_recording &&
        entry->training.num_recorded_points < ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX) {
        float pos[3], ori[4], lin_vel[3], ang_vel[3];
        char model_name[ROS_ROBOT_CONTROLLER_MAX_NAME + 16];
        snprintf(model_name, sizeof(model_name), "%s_model", entry->name);

        if (ros_gazebo_bridge_get_model_state((RosGazeboBridge*)entry->bridge, model_name,
                                                pos, ori, lin_vel, ang_vel) == 0) {
            int idx_p = entry->training.num_recorded_points;
            entry->training.recorded_times[idx_p] = (float)entry->training.num_recorded_points * 0.05f;
            entry->training.recorded_positions[idx_p][0] = pos[0];
            entry->training.recorded_positions[idx_p][1] = pos[1];
            entry->training.recorded_positions[idx_p][2] = pos[2];
            entry->training.recorded_orientations[idx_p][0] = ori[0];
            entry->training.recorded_orientations[idx_p][1] = ori[1];
            entry->training.recorded_orientations[idx_p][2] = ori[2];
            entry->training.recorded_orientations[idx_p][3] = ori[3];
            entry->training.num_recorded_points++;
        }
    }

    if (entry->training.is_replaying && entry->training.num_recorded_points > 0) {
        int idx_r = entry->training.replay_index;
        if (idx_r < entry->training.num_recorded_points) {
            RosTwist cmd;
            cmd.linear.x = (entry->training.recorded_positions[idx_r][0] - entry->info.position[0]) * 2.0;
            cmd.linear.y = (entry->training.recorded_positions[idx_r][1] - entry->info.position[1]) * 2.0;
            cmd.linear.z = 0;
            cmd.angular.x = 0;
            cmd.angular.y = 0;
            cmd.angular.z = (entry->training.recorded_orientations[idx_r][2] - entry->info.orientation[2]) * 2.0;
            ros_gazebo_bridge_publish_cmd_vel((RosGazeboBridge*)entry->bridge, &cmd);

            entry->training.replay_index++;
            if (entry->training.replay_index >= entry->training.num_recorded_points) {
                entry->training.is_replaying = 0;
                entry->info.training_state = ROS_ROBOT_TRAINING_STATE_COMPLETED;
                entry->info.training_progress = 1.0f;
            } else {
                entry->info.training_progress = (float)entry->training.replay_index /
                                                 (float)entry->training.num_recorded_points;
            }
        }
    }

    memcpy(session, &entry->training, sizeof(RosRobotTrainingSession));
    return 0;
}

int ros_robot_controller_update(RosRobotController* controller) {
    if (!controller) return -1;
    double now = controller_get_time();
    double dt = now - controller->last_spin_time;

    for (int i = 0; i < controller->robot_count; i++) {
        RosRobotEntry* entry = &controller->robots[i];
        if (!entry->active) continue;

        if (entry->info.connection_state >= ROS_ROBOT_CONNECTION_STATE_CONNECTED) {
            double heartbeat_dt = now - entry->last_heartbeat_time;
            if (heartbeat_dt > (double)controller->config.heartbeat_interval_ms / 1000.0) {
                entry->last_heartbeat_time = now;
                entry->info.last_heartbeat = now;

                if (entry->bridge) {
                    RosGazeboBridgeState bridge_state = ros_gazebo_bridge_get_state((RosGazeboBridge*)entry->bridge);
                    if (bridge_state == ROS_GAZEBO_BRIDGE_STATE_RUNNING) {
                        entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_ACTIVE;
                        entry->consecutive_timeouts = 0;
                    } else if (bridge_state >= ROS_GAZEBO_BRIDGE_STATE_CONNECTED) {
                        entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_CONNECTED;
                        entry->consecutive_timeouts = 0;
                    } else {
                        entry->consecutive_timeouts++;
                        if (entry->consecutive_timeouts > 3) {
                            entry->info.connection_state = ROS_ROBOT_CONNECTION_STATE_ERROR;
                            strncpy(entry->info.last_error, "ROS-Gazebo桥接心跳超时",
                                    sizeof(entry->info.last_error) - 1);
                            if (controller->config.enable_auto_reconnect) {
                                ros_gazebo_bridge_connect((RosGazeboBridge*)entry->bridge);
                                entry->consecutive_timeouts = 0;
                            }
                        }
                    }
                }
            }

            if (entry->info.is_training) {
                if (entry->training.is_recording || entry->training.is_replaying) {
                    RosRobotTrainingSession session;
                    ros_robot_controller_get_training_session(controller, i, &session);
                }

                if (entry->info.training_state == ROS_ROBOT_TRAINING_STATE_TRAINING) {
                    entry->info.training_progress += (float)(dt * 0.1);
                    if (entry->info.training_progress >= 1.0f) {
                        entry->info.training_progress = 1.0f;
                        entry->info.training_state = ROS_ROBOT_TRAINING_STATE_COMPLETED;
                    }
                }
            }
        }

        entry->last_update_time = now;
    }

    controller->last_spin_time = now;
    return 0;
}

int ros_robot_controller_spin(RosRobotController* controller, int timeout_ms) {
    if (!controller) return -1;

    double start = controller_get_time();
    int elapsed = 0;

    while (elapsed < timeout_ms || timeout_ms <= 0) {
        ros_robot_controller_update(controller);

        if (timeout_ms > 0) {
            double now = controller_get_time();
            elapsed = (int)((now - start) * 1000.0);
            if (elapsed >= timeout_ms) break;
        }

#ifdef _WIN32
        Sleep(ROS_ROBOT_CONTROLLER_SPIN_INTERVAL_MS);
#else
        usleep(ROS_ROBOT_CONTROLLER_SPIN_INTERVAL_MS * 1000);
#endif
    }

    return 0;
}

const char* ros_robot_controller_get_last_error(RosRobotController* controller) {
    if (!controller) return "控制器未初始化";
    return controller->last_error;
}

int ros_robot_controller_get_error_code(RosRobotController* controller) {
    if (!controller) return -1;
    return controller->error_code;
}
