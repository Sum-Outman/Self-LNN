#ifndef SELFLNN_ROS_GAZEBO_BRIDGE_H
#define SELFLNN_ROS_GAZEBO_BRIDGE_H

#include "selflnn/robot/ros_protocol.h"
#include "selflnn/robot/ros_node.h"
#include "selflnn/robot/simulator.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROS_GAZEBO_BRIDGE_MAX_TOPICS    64
#define ROS_GAZEBO_BRIDGE_MAX_SERVICES  16
#define ROS_GAZEBO_BRIDGE_MAX_ROBOTS    16
#define ROS_GAZEBO_BRIDGE_MAX_LINKS     128
#define ROS_GAZEBO_BRIDGE_MAX_SENSORS   32
#define ROS_GAZEBO_BRIDGE_WORLD_NAME_MAX 64
#define ROS_GAZEBO_BRIDGE_FRAME_ID_MAX  64

typedef enum {
    ROS_GAZEBO_BRIDGE_STATE_DISCONNECTED = 0,
    ROS_GAZEBO_BRIDGE_STATE_CONNECTING = 1,
    ROS_GAZEBO_BRIDGE_STATE_CONNECTED = 2,
    ROS_GAZEBO_BRIDGE_STATE_RUNNING = 3,
    ROS_GAZEBO_BRIDGE_STATE_ERROR = 4
} RosGazeboBridgeState;

typedef enum {
    ROS_GAZEBO_LINK_TYPE_NONE = 0,
    ROS_GAZEBO_LINK_TYPE_BASE = 1,
    ROS_GAZEBO_LINK_TYPE_LEG = 2,
    ROS_GAZEBO_LINK_TYPE_ARM = 3,
    ROS_GAZEBO_LINK_TYPE_HEAD = 4,
    ROS_GAZEBO_LINK_TYPE_GRIPPER = 5,
    ROS_GAZEBO_LINK_TYPE_WHEEL = 6,
    ROS_GAZEBO_LINK_TYPE_SENSOR = 7
} RosGazeboLinkType;

typedef struct {
    char name[ROS_GAZEBO_BRIDGE_FRAME_ID_MAX];
    RosGazeboLinkType type;
    float pose_position[3];
    float pose_orientation[4];
    float velocity_linear[3];
    float velocity_angular[3];
    double inertial_mass;
    float inertial_inertia[9];
    char parent_link[ROS_GAZEBO_BRIDGE_FRAME_ID_MAX];
    int has_collision;
    int is_fixed;
} RosGazeboLinkInfo;

typedef struct {
    char name[ROS_GAZEBO_BRIDGE_FRAME_ID_MAX];
    char type[32];
    float position[3];
    float orientation[4];
    int publish_tf;
    int is_static;
    double last_update_time;
} RosGazeboFrameTransform;

typedef struct {
    int num_names;
    char names[ROS_GAZEBO_BRIDGE_MAX_ROBOTS][ROS_GAZEBO_BRIDGE_FRAME_ID_MAX];
    int num_models;
    int num_lights;
    int num_physics_profiles;
    float simulation_time;
    int paused;
    int step_count;
    float real_time_factor;
} RosGazeboWorldState;

typedef struct {
    int num_links;
    RosGazeboLinkInfo links[ROS_GAZEBO_BRIDGE_MAX_LINKS];
    int num_joints;
    float joint_positions[32];
    float joint_velocities[32];
    float joint_efforts[32];
    char joint_names[32][ROS_GAZEBO_BRIDGE_FRAME_ID_MAX];
    int num_sensors;
    int sensor_ids[ROS_GAZEBO_BRIDGE_MAX_SENSORS];
    char sensor_names[ROS_GAZEBO_BRIDGE_MAX_SENSORS][ROS_GAZEBO_BRIDGE_FRAME_ID_MAX];
} RosGazeboRobotInfo;

typedef struct RosGazeboBridge RosGazeboBridge;

typedef struct {
    char ros_master_host[256];
    int ros_master_port;
    char node_name[ROS_MAX_NODE_NAME];
    char node_host[256];
    int node_xmlrpc_port;
    int node_tcp_port;
    char world_name[ROS_GAZEBO_BRIDGE_WORLD_NAME_MAX];
    float publish_rate;
    int enable_joint_state_pub;
    int enable_odom_pub;
    int enable_scan_pub;
    int enable_imu_pub;
    int enable_camera_pub;
    int enable_tf_pub;
    int enable_sim_control_srv;
    int use_external_gazebo;
    int internal;
} RosGazeboBridgeConfig;

RosGazeboBridgeConfig ros_gazebo_bridge_config_default(const char* node_name);

RosGazeboBridge* ros_gazebo_bridge_create(const RosGazeboBridgeConfig* config);
void ros_gazebo_bridge_destroy(RosGazeboBridge* bridge);

int ros_gazebo_bridge_connect(RosGazeboBridge* bridge);
int ros_gazebo_bridge_disconnect(RosGazeboBridge* bridge);
int ros_gazebo_bridge_is_connected(const RosGazeboBridge* bridge);
RosGazeboBridgeState ros_gazebo_bridge_get_state(const RosGazeboBridge* bridge);

/**
 * @brief 检测Gazebo/ROS Master运行环境是否可用
 *
 * 通过TCP短连接尝试连接本地ROS Master（端口11311），
 * 200ms超时，不阻塞。用于启动时决策是否启用ROS/Gazebo相关功能。
 *
 * @return int 1=Gazebo/ROS环境可用，0=不可用（无Gazebo/ROS安装）
 */
int gazebo_is_environment_available(void);

int ros_gazebo_bridge_start(RosGazeboBridge* bridge);
int ros_gazebo_bridge_stop(RosGazeboBridge* bridge);
int ros_gazebo_bridge_pause(RosGazeboBridge* bridge);
int ros_gazebo_bridge_resume(RosGazeboBridge* bridge);
int ros_gazebo_bridge_step(RosGazeboBridge* bridge, int num_steps);
int ros_gazebo_bridge_reset(RosGazeboBridge* bridge);

int ros_gazebo_bridge_spawn_model(RosGazeboBridge* bridge, const char* model_name,
                                   const float* position, const float* orientation);
int ros_gazebo_bridge_delete_model(RosGazeboBridge* bridge, const char* model_name);
int ros_gazebo_bridge_get_model_state(RosGazeboBridge* bridge, const char* model_name,
                                       float* position, float* orientation, float* linear_vel, float* angular_vel);
int ros_gazebo_bridge_set_model_state(RosGazeboBridge* bridge, const char* model_name,
                                       const float* position, const float* orientation);
int ros_gazebo_bridge_apply_body_wrench(RosGazeboBridge* bridge, const char* model_name,
                                         const char* link_name, const float* force, const float* torque);
int ros_gazebo_bridge_get_joint_state(RosGazeboBridge* bridge, int robot_id,
                                       float* positions, float* velocities, float* efforts, int max_joints);
int ros_gazebo_bridge_set_joint_position(RosGazeboBridge* bridge, int robot_id,
                                          const char* joint_name, float position);
int ros_gazebo_bridge_set_joint_velocity(RosGazeboBridge* bridge, int robot_id,
                                          const char* joint_name, float velocity);
int ros_gazebo_bridge_set_joint_torque(RosGazeboBridge* bridge, int robot_id,
                                        const char* joint_name, float torque);

int ros_gazebo_bridge_get_world_state(RosGazeboBridge* bridge, RosGazeboWorldState* state);
int ros_gazebo_bridge_get_robot_info(RosGazeboBridge* bridge, int robot_id,
                                      RosGazeboRobotInfo* info);
int ros_gazebo_bridge_get_link_info(RosGazeboBridge* bridge, int robot_id,
                                     const char* link_name, RosGazeboLinkInfo* info);

int ros_gazebo_bridge_get_odometry(RosGazeboBridge* bridge, int robot_id,
                                    double* pos_x, double* pos_y, double* pos_z,
                                    double* ori_x, double* ori_y, double* ori_z, double* ori_w,
                                    double* lin_vel_x, double* lin_vel_y, double* lin_vel_z,
                                    double* ang_vel_x, double* ang_vel_y, double* ang_vel_z);

int ros_gazebo_bridge_get_sensor_data(RosGazeboBridge* bridge, int sensor_id,
                                       float* data, int max_size, int* size_out);

int ros_gazebo_bridge_publish_cmd_vel(RosGazeboBridge* bridge, const RosTwist* cmd);
int ros_gazebo_bridge_publish_odometry(RosGazeboBridge* bridge, int robot_id);
int ros_gazebo_bridge_publish_joint_states(RosGazeboBridge* bridge, int robot_id);
int ros_gazebo_bridge_publish_laserscan(RosGazeboBridge* bridge, int sensor_id);
int ros_gazebo_bridge_publish_imu(RosGazeboBridge* bridge, int sensor_id);
int ros_gazebo_bridge_publish_camera(RosGazeboBridge* bridge, int sensor_id);

int ros_gazebo_bridge_get_robot_count(RosGazeboBridge* bridge);
int ros_gazebo_bridge_get_simulator_handle(RosGazeboBridge* bridge, void** sim_out);

const char* ros_gazebo_bridge_get_last_error(RosGazeboBridge* bridge);
int ros_gazebo_bridge_get_error_code(RosGazeboBridge* bridge);

#ifdef __cplusplus
}
#endif

#endif
