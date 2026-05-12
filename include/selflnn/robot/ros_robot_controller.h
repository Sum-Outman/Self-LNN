#ifndef SELFLNN_ROS_ROBOT_CONTROLLER_H
#define SELFLNN_ROS_ROBOT_CONTROLLER_H

#include "selflnn/robot/ros_gazebo_bridge.h"
#include "selflnn/robot/robot.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROS_ROBOT_CONTROLLER_MAX_ROBOTS      16
#define ROS_ROBOT_CONTROLLER_MAX_NAME        64
#define ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX  1024
#define ROS_ROBOT_CONTROLLER_MAX_CONTROL_MODES 8

typedef enum {
    ROS_ROBOT_CONTROL_MODE_NONE = 0,
    ROS_ROBOT_CONTROL_MODE_VELOCITY = 1,
    ROS_ROBOT_CONTROL_MODE_POSITION = 2,
    ROS_ROBOT_CONTROL_MODE_JOINT_POSITION = 3,
    ROS_ROBOT_CONTROL_MODE_JOINT_VELOCITY = 4,
    ROS_ROBOT_CONTROL_MODE_JOINT_TORQUE = 5,
    ROS_ROBOT_CONTROL_MODE_TRAJECTORY = 6,
    ROS_ROBOT_CONTROL_MODE_TRAINING = 7,
    ROS_ROBOT_CONTROL_MODE_TELEOP = 8
} RosRobotControlMode;

typedef enum {
    ROS_ROBOT_CONNECTION_STATE_DISCONNECTED = 0,
    ROS_ROBOT_CONNECTION_STATE_CONNECTING = 1,
    ROS_ROBOT_CONNECTION_STATE_CONNECTED = 2,
    ROS_ROBOT_CONNECTION_STATE_ACTIVE = 3,
    ROS_ROBOT_CONNECTION_STATE_ERROR = 4
} RosRobotConnectionState;

typedef enum {
    ROS_ROBOT_TRAINING_MODE_NONE = 0,
    ROS_ROBOT_TRAINING_MODE_IMITATION = 1,
    ROS_ROBOT_TRAINING_MODE_REINFORCEMENT = 2,
    ROS_ROBOT_TRAINING_MODE_MOVEMENT_PRIMITIVES = 3,
    ROS_ROBOT_TRAINING_MODE_JOINT_SPACE = 4,
    ROS_ROBOT_TRAINING_MODE_TASK_SPACE = 5
} RosRobotTrainingMode;

typedef enum {
    ROS_ROBOT_TRAINING_STATE_IDLE = 0,
    ROS_ROBOT_TRAINING_STATE_RECORDING = 1,
    ROS_ROBOT_TRAINING_STATE_TRAINING = 2,
    ROS_ROBOT_TRAINING_STATE_EVALUATING = 3,
    ROS_ROBOT_TRAINING_STATE_COMPLETED = 4,
    ROS_ROBOT_TRAINING_STATE_ERROR = 5
} RosRobotTrainingState;

typedef struct {
    int robot_id;
    char robot_name[ROS_ROBOT_CONTROLLER_MAX_NAME];
    RobotType robot_type;
    RosRobotConnectionState connection_state;
    RosRobotControlMode active_control_mode;
    int is_ros_connected;
    int is_gazebo_connected;
    double last_heartbeat;
    int command_count;
    int error_count;
    float battery_level;
    float position[3];
    float orientation[4];
    float linear_velocity[3];
    float angular_velocity[3];
    float joint_positions[32];
    float joint_velocities[32];
    float joint_efforts[32];
    int num_joints;
    int is_emergency_stopped;
    int is_training;
    RosRobotTrainingMode training_mode;
    RosRobotTrainingState training_state;
    float training_progress;
    char last_error[256];
} RosRobotInfo;

typedef struct {
    int record_trajectory;
    int num_recorded_points;
    int max_recorded_points;
    float recorded_times[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX];
    float recorded_positions[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX][3];
    float recorded_orientations[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX][4];
    float recorded_joint_positions[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX][32];
    int recorded_num_joints;
    float joint_positions[32];
    float joint_velocities[32];
    float joint_efforts[32];
    int num_joints;
    float replay_speed;
    int is_recording;
    int is_replaying;
    int replay_index;
    int training_iterations;
    float learning_rate;
    float exploration_rate;
    float discount_factor;
    int training_episode;
    float reward_sum;
    float avg_reward;
} RosRobotTrainingSession;

typedef struct {
    int mode;
    float velocity_linear[3];
    float velocity_angular[3];
    float position_target[3];
    float orientation_target[4];
    float joint_targets[32];
    int num_joint_targets;
    float trajectory_points[ROS_ROBOT_CONTROLLER_TRAJECTORY_MAX][7];
    int num_trajectory_points;
    float duration;
    float max_velocity;
    float max_acceleration;
    int flags;
} RosRobotControlCommand;

typedef struct RosRobotController RosRobotController;

typedef struct {
    int max_robots;
    int enable_auto_reconnect;
    int reconnect_interval_ms;
    int heartbeat_interval_ms;
    int enable_multi_robot_coordination;
    int enable_training;
    char default_ros_master_host[256];
    int default_ros_master_port;
    float control_loop_hz;
} RosRobotControllerConfig;

RosRobotControllerConfig ros_robot_controller_config_default(void);

RosRobotController* ros_robot_controller_create(const RosRobotControllerConfig* config);
void ros_robot_controller_destroy(RosRobotController* controller);

int ros_robot_controller_add_robot(RosRobotController* controller, const char* name,
                                    RobotType type, const RosGazeboBridgeConfig* bridge_config);
int ros_robot_controller_remove_robot(RosRobotController* controller, int robot_id);

int ros_robot_controller_connect_robot(RosRobotController* controller, int robot_id);
int ros_robot_controller_disconnect_robot(RosRobotController* controller, int robot_id);

int ros_robot_controller_connect_all(RosRobotController* controller);
int ros_robot_controller_disconnect_all(RosRobotController* controller);

int ros_robot_controller_set_control_mode(RosRobotController* controller, int robot_id,
                                           RosRobotControlMode mode);

int ros_robot_controller_send_velocity(RosRobotController* controller, int robot_id,
                                        float linear_x, float linear_y, float linear_z,
                                        float angular_x, float angular_y, float angular_z);

int ros_robot_controller_send_position(RosRobotController* controller, int robot_id,
                                        float x, float y, float z,
                                        float ox, float oy, float oz, float ow);

int ros_robot_controller_send_joint_positions(RosRobotController* controller, int robot_id,
                                                const float* positions, int num_joints);

int ros_robot_controller_send_joint_velocities(RosRobotController* controller, int robot_id,
                                                 const float* velocities, int num_joints);

int ros_robot_controller_send_joint_torques(RosRobotController* controller, int robot_id,
                                              const float* torques, int num_joints);

int ros_robot_controller_execute_trajectory(RosRobotController* controller, int robot_id,
                                             const float* trajectory, int num_points,
                                             float duration);

int ros_robot_controller_emergency_stop(RosRobotController* controller, int robot_id);
int ros_robot_controller_emergency_stop_all(RosRobotController* controller);
int ros_robot_controller_clear_emergency_stop(RosRobotController* controller, int robot_id);

int ros_robot_controller_get_robot_info(RosRobotController* controller, int robot_id,
                                         RosRobotInfo* info);
int ros_robot_controller_get_robot_count(RosRobotController* controller);
int ros_robot_controller_get_connected_count(RosRobotController* controller);
int ros_robot_controller_get_active_count(RosRobotController* controller);

int ros_robot_controller_get_gazebo_bridge(RosRobotController* controller, int robot_id,
                                            RosGazeboBridge** bridge);

int ros_robot_controller_start_training(RosRobotController* controller, int robot_id,
                                         RosRobotTrainingMode mode);
int ros_robot_controller_stop_training(RosRobotController* controller, int robot_id);
int ros_robot_controller_pause_training(RosRobotController* controller, int robot_id);
int ros_robot_controller_resume_training(RosRobotController* controller, int robot_id);

int ros_robot_controller_record_trajectory(RosRobotController* controller, int robot_id,
                                            int start_recording);
int ros_robot_controller_replay_trajectory(RosRobotController* controller, int robot_id,
                                            float replay_speed);

int ros_robot_controller_set_training_params(RosRobotController* controller, int robot_id,
                                              float learning_rate, float exploration_rate,
                                              float discount_factor, int num_iterations);

int ros_robot_controller_get_training_session(RosRobotController* controller, int robot_id,
                                                RosRobotTrainingSession* session);

int ros_robot_controller_get_bridge(RosRobotController* controller, int robot_id,
                                     RosGazeboBridge** bridge_out);

int ros_robot_controller_update(RosRobotController* controller);
int ros_robot_controller_spin(RosRobotController* controller, int timeout_ms);

const char* ros_robot_controller_get_last_error(RosRobotController* controller);
int ros_robot_controller_get_error_code(RosRobotController* controller);

#ifdef __cplusplus
}
#endif

#endif
