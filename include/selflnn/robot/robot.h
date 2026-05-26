/**
 * @file robot.h
 * @brief 机器人控制接口
 * 
 * 机器人控制模块，支持多机器人协调、传感器融合、运动规划和设备控制。
 */

#ifndef SELFLNN_ROBOT_H
#define SELFLNN_ROBOT_H

#include <stddef.h>
#include "selflnn/robot/hardware_interface.h"
#include "selflnn/robot/hardware_detector.h"
#include "selflnn/robot/pybullet_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 机器人类型枚举
 */
typedef enum {
    ROBOT_TYPE_MOBILE = 0,      /**< 移动机器人 */
    ROBOT_TYPE_MANIPULATOR = 1, /**< 机械臂 */
    ROBOT_TYPE_HUMANOID = 2,    /**< 人形机器人 */
    ROBOT_TYPE_AERIAL = 3,      /**< 空中机器人 */
    ROBOT_TYPE_QUADCOPTER = 4,  /**< 四旋翼无人机 */
    ROBOT_TYPE_AQUATIC = 5,     /**< 水下机器人 */
    ROBOT_TYPE_CUSTOM = 6       /**< 自定义机器人 */
} RobotType;

/**
 * @brief 仿真引擎类型枚举
 */
typedef enum {
    SIM_ENGINE_PYBULLET = 0,    /**< PyBullet物理仿真引擎 */
    SIM_ENGINE_GAZEBO = 1,      /**< Gazebo仿真引擎 */
    SIM_ENGINE_MUJOCO = 2,      /**< MuJoCo物理引擎 */
    SIM_ENGINE_CUSTOM = 3       /**< 自定义仿真引擎 */
} SimEngine;

/**
 * @brief 传感器类型枚举
 */
typedef enum {
    SENSOR_TYPE_LIDAR = 0,      /**< 激光雷达 */
    SENSOR_TYPE_CAMERA = 1,     /**< 摄像头 */
    SENSOR_TYPE_IMU = 2,        /**< 惯性测量单元 */
    SENSOR_TYPE_GNSS = 3,       /**< 卫星导航 */
    SENSOR_TYPE_FORCE_TORQUE = 4, /**< 力扭矩传感器 */
    SENSOR_TYPE_TEMPERATURE = 5, /**< 温度传感器 */
    SENSOR_TYPE_PRESSURE = 6,    /**< 压力传感器 */
    SENSOR_TYPE_PROXIMITY = 7    /**< 接近传感器 */
} SensorType;

/**
 * @brief 运动模式枚举
 */
typedef enum {
    MOTION_MODE_POSITION = 0,   /**< 位置控制 */
    MOTION_MODE_VELOCITY = 1,   /**< 速度控制 */
    MOTION_MODE_TORQUE = 2,     /**< 力矩控制 */
    MOTION_MODE_TRAJECTORY = 3  /**< 轨迹控制 */
} MotionMode;

/**
 * @brief 机器人状态枚举
 */
typedef enum {
    ROBOT_STATE_IDLE = 0,       /**< 空闲 */
    ROBOT_STATE_MOVING = 1,     /**< 移动中 */
    ROBOT_STATE_GRASPING = 2,   /**< 抓取中 */
    ROBOT_STATE_NAVIGATING = 3, /**< 导航中 */
    ROBOT_STATE_ERROR = 4,      /**< 错误状态 */
    ROBOT_STATE_EMERGENCY = 5,  /**< 紧急停止 */
    ROBOT_STATE_READY = 6,      /**< 准备就绪 */
    ROBOT_STATE_EMERGENCY_STOP = 7 /**< 紧急停止状态 */
} RobotState;

/**
 * @brief 机器人配置结构体
 */
typedef struct {
    RobotType type;             /**< 机器人类型 */
    char name[64];              /**< 机器人名称 */
    int robot_id;               /**< 机器人ID */
    float max_linear_velocity;  /**< 最大线速度 (m/s) */
    float max_angular_velocity; /**< 最大角速度 (rad/s) */
    float max_acceleration;     /**< 最大加速度 (m/s²) */
    int num_joints;             /**< 关节数量 */
    int has_gripper;            /**< 是否有夹爪 */
    int enable_safety;          /**< 是否启用安全功能 */
    float safety_distance;      /**< 安全距离 (m) */
    int enable_sync;            /**< 是否启用同步控制 */
} RobotConfig;

/**
 * @brief 传感器配置结构体
 */
typedef struct {
    SensorType type;            /**< 传感器类型 */
    char name[32];              /**< 传感器名称 */
    int sensor_id;              /**< 传感器ID */
    float update_rate;          /**< 更新频率 (Hz) */
    float range_min;            /**< 最小测量范围 */
    float range_max;            /**< 最大测量范围 */
    float resolution;           /**< 分辨率 */
    float accuracy;             /**< 精度 */
} SensorConfig;

/**
 * @brief 机器人状态结构体
 */
typedef struct {
    RobotState state;           /**< 机器人状态 */
    float timestamp;            /**< 时间戳 (秒) */
    float position[3];          /**< 位置 (x, y, z) (m) */
    float orientation[4];       /**< 姿态 (四元数) */
    float linear_velocity[3];   /**< 线速度 (m/s) */
    float angular_velocity[3];  /**< 角速度 (rad/s) */
    float joint_positions[32];  /**< 关节位置 (rad) */
    float joint_velocities[32]; /**< 关节速度 (rad/s) */
    float joint_torques[32];    /**< 关节力矩 (Nm) */
    float gripper_position;     /**< 夹爪位置 (0-1) */
    float gripper_force;        /**< 夹持力 (N) */
    float battery_level;        /**< 电池电量 (0-1) */
    float temperature;          /**< 温度 (°C) */
    int error_code;             /**< 错误码 */
    char error_message[128];    /**< 错误信息 */
} RobotStatus;

/**
 * @brief 传感器数据结构体
 */
typedef struct {
    SensorType type;            /**< 传感器类型 */
    int sensor_id;              /**< 传感器ID */
    float timestamp;            /**< 时间戳 (秒) */
    float* data;                /**< 传感器数据 */
    size_t data_size;           /**< 数据大小 */
    int is_valid;               /**< 数据是否有效 */
} SensorData;

/**
 * @brief 机器人命令标志枚举
 */
typedef enum {
    ROBOT_CMD_FLAG_NONE = 0,            /**< 无标志 */
    ROBOT_CMD_FLAG_SYNC_PREP = 1 << 0,  /**< 同步准备标志 */
    ROBOT_CMD_FLAG_SYNC_EXEC = 1 << 1,  /**< 同步执行标志 */
    ROBOT_CMD_FLAG_EMERGENCY_STOP = 1 << 2, /**< 紧急停止标志 */
    ROBOT_CMD_FLAG_OVERRIDE = 1 << 3,   /**< 手动覆盖标志 */
} RobotCommandFlags;

/**
 * @brief 机器人控制命令结构体
 */
typedef struct {
    MotionMode mode;            /**< 控制模式 */
    float target_position[3];   /**< 目标位置 (x, y, z) (m) */
    float target_orientation[4]; /**< 目标姿态 (四元数) */
    float target_linear_velocity[3]; /**< 目标线速度 (m/s) */
    float target_angular_velocity[3]; /**< 目标角速度 (rad/s) */
    float target_joint_positions[32]; /**< 目标关节位置 (rad) */
    float target_joint_velocities[32]; /**< 目标关节速度 (rad/s) */
    float target_joint_torques[32];   /**< 目标关节力矩 (Nm) */
    float target_gripper_position;    /**< 目标夹爪位置 (0-1) */
    float max_velocity;        /**< 最大速度限制 */
    float max_acceleration;    /**< 最大加速度限制 */
    int use_trajectory;        /**< 是否使用轨迹规划 */
    float trajectory_time;     /**< 轨迹时间 (秒) */
    float max_jerk;            /**< 最大加加速度限制 (m/s³) */
    float trajectory_duration; /**< 轨迹持续时间 (秒) */
    int trajectory_type;       /**< 轨迹类型 */
    int coordination_mode;     /**< 协调模式 */
    int flags;                 /**< 命令标志位 */
} RobotCommand;

/**
 * @brief 控制命令结构体
 */
typedef struct {
    int command_type;               /**< 命令类型 */
    float* target_positions;        /**< 目标位置数组 */
    size_t positions_size;          /**< 位置数组大小 */
    float* target_velocities;       /**< 目标速度数组 */
    size_t velocities_size;         /**< 速度数组大小 */
    float* target_torques;          /**< 目标力矩数组 */
    size_t torques_size;            /**< 力矩数组大小 */
    float* trajectory_points;       /**< 轨迹点数组 */
    size_t trajectory_size;         /**< 轨迹点数量 */
    float duration;                 /**< 持续时间 (秒) */
    int priority;                   /**< 优先级 */
} ControlCommand;

/**
 * @brief 关节状态结构体
 */
typedef struct {
    int joint_id;                   /**< 关节ID */
    float position;                 /**< 关节位置 (rad) */
    float velocity;                 /**< 关节速度 (rad/s) */
    float torque;                   /**< 关节力矩 (Nm) */
} JointState;

/**
 * @brief 机器人姿态结构体
 */
typedef struct {
    float position[3];              /**< 位置 (x, y, z) (m) */
    float linear_velocity[3];       /**< 线速度 (m/s) */
    float angular_velocity[3];      /**< 角速度 (rad/s) */
} RobotPose;

/**
 * @brief 机器人任务结构体
 */
typedef struct {
    const char* task_id;            /**< 任务ID */
    const char* task_type;          /**< 任务类型 */
    float* target_pose;             /**< 目标姿态数组 */
    size_t pose_size;               /**< 姿态数组大小 */
    float* constraints;             /**< 约束条件数组 */
    size_t constraints_size;        /**< 约束数组大小 */
    float timeout;                  /**< 超时时间 (秒) */
    int allow_interruption;         /**< 是否允许中断 */
} RobotTask;

/**
 * @brief 机器人句柄
 */
typedef struct Robot Robot;

/**
 * @brief 多机器人控制器句柄
 */
typedef struct RobotController RobotController;

/**
 * @brief 多机器人控制器配置
 */
typedef struct {
    int max_robots;                 /**< 最大机器人数量 */
    float simulation_time_step;     /**< 仿真时间步长 (秒) */
    int enable_physics;             /**< 是否启用物理仿真 */
    int enable_rendering;           /**< 是否启用渲染 */
    int enable_network;             /**< 是否启用网络通信 */
    const char* environment_file;   /**< 环境文件路径 */
    float gravity[3];               /**< 重力向量 (m/s²) */
} MultiRobotConfig;

/**
 * @brief 创建机器人实例
 * 
 * @param config 机器人配置
 * @return Robot* 机器人句柄，失败返回NULL
 */
Robot* robot_create(const RobotConfig* config);

/**
 * @brief 释放机器人实例
 * 
 * @param robot 机器人句柄
 */
void robot_free(Robot* robot);

/**
 * @brief 添加传感器到机器人
 * 
 * @param robot 机器人句柄
 * @param config 传感器配置
 * @return int 成功返回传感器ID，失败返回-1
 */
int robot_add_sensor(Robot* robot, const SensorConfig* config);

/**
 * @brief 移除机器人上的传感器
 * 
 * @param robot 机器人句柄
 * @param sensor_id 传感器ID
 * @return int 成功返回0，失败返回-1
 */
int robot_remove_sensor(Robot* robot, int sensor_id);

/**
 * @brief 获取机器人状态
 * 
 * @param robot 机器人句柄
 * @param status 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int robot_get_status(Robot* robot, RobotStatus* status);

/**
 * @brief 获取传感器数据
 * 
 * @param robot 机器人句柄
 * @param sensor_id 传感器ID
 * @param data 数据输出缓冲区
 * @return int 成功返回数据大小，失败返回-1
 */
int robot_get_sensor_data(Robot* robot, int sensor_id, SensorData* data);

/**
 * @brief 发送控制命令
 * 
 * @param robot 机器人句柄
 * @param command 控制命令
 * @return int 成功返回0，失败返回-1
 */
int robot_send_command(Robot* robot, const RobotCommand* command);

/**
 * @brief 机器人移动到目标位置
 * 
 * @param robot 机器人句柄
 * @param position 目标位置 (x, y, z)
 * @param orientation 目标姿态 (四元数，可为NULL)
 * @param velocity 移动速度 (m/s)
 * @return int 成功返回0，失败返回-1
 */
int robot_move_to_position(Robot* robot, const float position[3],
                          const float orientation[4], float velocity);

/**
 * @brief 机器人设置关节位置
 * 
 * @param robot 机器人句柄
 * @param joint_positions 关节位置数组
 * @param num_joints 关节数量
 * @param velocity 关节速度 (rad/s)
 * @return int 成功返回0，失败返回-1
 */
int robot_set_joint_positions(Robot* robot, const float* joint_positions,
                             int num_joints, float velocity);

/**
 * @brief 机器人控制夹爪
 * 
 * @param robot 机器人句柄
 * @param position 夹爪位置 (0-1)
 * @param force 夹持力 (N)
 * @return int 成功返回0，失败返回-1
 */
int robot_control_gripper(Robot* robot, float position, float force);

/**
 * @brief 机器人执行轨迹
 * 
 * @param robot 机器人句柄
 * @param waypoints 路径点数组
 * @param num_waypoints 路径点数量
 * @param velocities 速度数组
 * @param num_velocities 速度数量
 * @param total_time 总时间 (秒)
 * @return int 成功返回0，失败返回-1
 */
int robot_execute_trajectory(Robot* robot, const float* waypoints,
                            size_t num_waypoints, const float* velocities,
                            size_t num_velocities, float total_time);

/**
 * @brief 机器人紧急停止
 * 
 * @param robot 机器人句柄
 * @return int 成功返回0，失败返回-1
 */
int robot_emergency_stop(Robot* robot);

/**
 * @brief 清除机器人错误
 * 
 * @param robot 机器人句柄
 * @return int 成功返回0，失败返回-1
 */
int robot_clear_errors(Robot* robot);

/**
 * @brief 机器人重置
 * 
 * @param robot 机器人句柄
 * @return int 成功返回0，失败返回-1
 */
int robot_reset(Robot* robot);

/**
 * @brief 获取机器人配置
 * 
 * @param robot 机器人句柄
 * @param config 配置输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int robot_get_config(const Robot* robot, RobotConfig* config);

/**
 * @brief 更新机器人配置
 * 
 * @param robot 机器人句柄
 * @param config 新配置
 * @return int 成功返回0，失败返回-1
 */
int robot_set_config(Robot* robot, const RobotConfig* config);

/**
 * @brief 多机器人协调：获取所有机器人状态
 * 
 * @param robots 机器人句柄数组
 * @param num_robots 机器人数量
 * @param status_array 状态数组输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int robot_get_all_status(Robot** robots, size_t num_robots,
                        RobotStatus* status_array);

/**
 * @brief 多机器人协调：同步控制
 * 
 * @param robots 机器人句柄数组
 * @param num_robots 机器人数量
 * @param commands 控制命令数组
 * @param sync_time 同步时间 (秒)
 * @return int 成功返回0，失败返回-1
 */
int robot_sync_control(Robot** robots, size_t num_robots,
                      const RobotCommand* commands, float sync_time);

/**
 * @brief 设置机器人硬件接口
 * 
 * @param robot 机器人句柄
 * @param hardware 硬件接口句柄
 * @return int 成功返回0，失败返回-1
 */
int robot_set_hardware_interface(Robot* robot, HardwareInterface* hardware);

/**
 * @brief 启用硬件接口
 * 
 * @param robot 机器人句柄
 * @param enable 启用标志（1启用，0禁用）
 * @return int 成功返回0，失败返回-1
 */
int robot_enable_hardware(Robot* robot, int enable);

/**
 * @brief 检查硬件接口是否启用
 * 
 * @param robot 机器人句柄
 * @return int 已启用返回1，未启用返回0，错误返回-1
 */
int robot_is_hardware_enabled(const Robot* robot);

/**
 * @brief 检查机器人数据是否来自仿真模式（不可用于自主学习训练）
 *
 * P0-007修复: 仿真模式已移除，无硬件连接时所有数据为空，此函数保留用于兼容检查。
 *
 * @param robot 机器人句柄
 * @return int 1=仿真数据不安全，0=真实硬件数据
 */
int robot_is_simulation_data_unsafe(Robot* robot);

/**
 * @brief 通过硬件接口发送命令
 * 
 * @param robot 机器人句柄
 * @param command 控制命令
 * @return int 成功返回0，失败返回错误码
 */
int robot_send_hardware_command(Robot* robot, const RobotCommand* command);

/**
 * @brief 创建多机器人控制器
 * 
 * @param config 控制器配置
 * @return RobotController* 控制器句柄，失败返回NULL
 */
RobotController* robot_controller_create(const MultiRobotConfig* config);

/**
 * @brief 释放多机器人控制器
 * 
 * @param controller 控制器句柄
 */
void robot_controller_free(RobotController* controller);

/**
 * @brief 向控制器添加机器人
 * 
 * @param controller 控制器句柄
 * @param config 机器人配置
 * @return int 成功返回机器人ID，失败返回-1
 */
int robot_add(RobotController* controller, const RobotConfig* config);

/**
 * @brief 初始化机器人
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @return int 成功返回0，失败返回-1
 */
int robot_initialize(RobotController* controller, int robot_id);

/**
 * @brief 启动机器人
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @return int 成功返回0，失败返回-1
 */
int robot_start(RobotController* controller, int robot_id);

/**
 * @brief 获取机器人状态
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @param state 状态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int robot_get_state(RobotController* controller, int robot_id, RobotState* state);

/**
 * @brief 发送控制命令（控制器版本）
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @param command 控制命令
 * @return int 成功返回0，失败返回-1
 */
int robot_controller_send_command(RobotController* controller, int robot_id, const ControlCommand* command);

/**
 * @brief 执行仿真步进
 * 
 * @param controller 控制器句柄
 * @param time_step 时间步长 (秒)
 * @return int 成功返回0，失败返回-1
 */
int robot_step_simulation(RobotController* controller, float time_step);

/**
 * @brief 获取关节状态
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @param joint_states 关节状态输出缓冲区
 * @param max_joints 最大关节数量
 * @return int 成功返回关节数量，失败返回-1
 */
int robot_get_joint_states(RobotController* controller, int robot_id, JointState* joint_states, size_t max_joints);

/**
 * @brief 获取机器人姿态
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @param pose 姿态输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int robot_get_pose(RobotController* controller, int robot_id, RobotPose* pose);

/**
 * @brief 获取传感器数据（控制器版本）
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @param data 传感器数据输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int robot_controller_get_sensor_data(RobotController* controller, int robot_id, SensorData* data);

/**
 * @brief 执行机器人任务
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @param task 任务描述
 * @return int 成功返回0，失败返回-1
 */
int robot_execute_task(RobotController* controller, int robot_id, const RobotTask* task);

/**
 * @brief 获取任务状态
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID
 * @param task_id 任务ID
 * @param progress 进度输出缓冲区 (0-1)
 * @return int 成功返回0，失败返回-1
 */
int robot_get_task_status(RobotController* controller, int robot_id, const char* task_id, float* progress);

/**
 * @brief 紧急停止所有机器人（控制器版本）
 * 
 * @param controller 控制器句柄
 * @return int 成功返回0，失败返回-1
 */
int robot_controller_emergency_stop(RobotController* controller);

/**
 * @brief 重置仿真
 * 
 * @param controller 控制器句柄
 * @param robot_id 机器人ID (-1表示重置所有机器人)
 * @return int 成功返回0，失败返回-1
 */
int robot_reset_simulation(RobotController* controller, int robot_id);

/**
 * @brief 获取控制器统计信息
 * 
 * @param controller 控制器句柄
 * @param num_robots 机器人数量输出缓冲区
 * @param total_steps 总步数输出缓冲区
 * @param avg_step_time 平均步长时间输出缓冲区 (毫秒)
 * @param error_count 错误计数输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int robot_get_stats(RobotController* controller, size_t* num_robots, unsigned long* total_steps, float* avg_step_time, unsigned long* error_count);

/* ============================
 * 增强T5：多机器人健康监控系统API
 * ============================ */

/**
 * @brief 机器人健康状态枚举
 */
typedef enum {
    ROBOT_HEALTH_OK = 0,            /**< 健康正常 */
    ROBOT_HEALTH_WARNING = 1,       /**< 健康警告（延迟增高或轻微异常） */
    ROBOT_HEALTH_ERROR = 2,         /**< 健康错误（通信中断或严重异常） */
    ROBOT_HEALTH_CRITICAL = 3,      /**< 健康危急（即将离线） */
    ROBOT_HEALTH_OFFLINE = 4        /**< 已离线 */
} RobotHealthLevel;

/**
 * @brief 机器人健康状态结构
 */
typedef struct {
    int robot_id;                    /**< 机器人ID */
    RobotHealthLevel health_level;   /**< 健康等级 */
    double last_heartbeat_time;      /**< 上次心跳时间（秒） */
    double response_time_ms;         /**< 最近响应时间（毫秒） */
    double avg_response_time_ms;     /**< 平均响应时间（毫秒） */
    double max_response_time_ms;     /**< 最大响应时间（毫秒） */
    size_t command_success_count;    /**< 命令成功计数 */
    size_t command_failure_count;    /**< 命令失败计数 */
    size_t consecutive_failures;     /**< 连续失败次数 */
    double cpu_usage;                /**< CPU使用率 (0~1) */
    double memory_usage;             /**< 内存使用率 (0~1) */
    int is_responding;               /**< 是否正在响应 */
} RobotHealthStatus;

/**
 * @brief 机器人健康监控器（不透明类型）
 */
typedef struct RobotHealthMonitor RobotHealthMonitor;

/**
 * @brief 创建机器人健康监控器
 * 
 * @param controller 机器人控制器句柄
 * @param heartbeat_timeout 心跳超时阈值（秒）
 * @param max_consecutive_failures 最大连续失败次数（超过则判离线）
 * @param monitor_interval_ms 监控间隔（毫秒）
 * @return RobotHealthMonitor* 成功返回监控器句柄，失败返回NULL
 */
RobotHealthMonitor* robot_health_monitor_create(RobotController* controller,
                                                  double heartbeat_timeout,
                                                  size_t max_consecutive_failures,
                                                  size_t monitor_interval_ms);

/**
 * @brief 销毁健康监控器
 * 
 * @param monitor 健康监控器句柄
 */
void robot_health_monitor_free(RobotHealthMonitor* monitor);

/**
 * @brief 执行一次健康检查（对所有注册机器人）
 * 
 * 遍历所有机器人，检查心跳超时、连续失败次数、响应时间等指标，
 * 更新健康等级并触发故障转移。
 *
 * @param monitor 健康监控器句柄
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_check_all(RobotHealthMonitor* monitor);

/**
 * @brief 获取指定机器人的健康状态
 * 
 * @param monitor 健康监控器句柄
 * @param robot_id 机器人ID
 * @param status 输出健康状态结构体
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_get_status(RobotHealthMonitor* monitor,
                                      int robot_id,
                                      RobotHealthStatus* status);

/**
 * @brief 更新指定机器人的心跳
 * 
 * @param monitor 健康监控器句柄
 * @param robot_id 机器人ID
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_update_heartbeat(RobotHealthMonitor* monitor,
                                            int robot_id);

/**
 * @brief 记录一次命令执行结果（成功/失败）
 * 
 * @param monitor 健康监控器句柄
 * @param robot_id 机器人ID
 * @param response_time_ms 响应时间（毫秒）
 * @param success 成功标志（1=成功，0=失败）
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_record_command(RobotHealthMonitor* monitor,
                                          int robot_id,
                                          double response_time_ms,
                                          int success);

/**
 * @brief 触发故障转移：将任务从故障机器人迁移至备用机器人
 * 
 * @param monitor 健康监控器句柄
 * @param failed_robot_id 故障机器人ID
 * @return int 成功返回目标机器人ID，失败返回-1
 */
int robot_health_monitor_failover(RobotHealthMonitor* monitor,
                                    int failed_robot_id);

/**
 * @brief 获取当前健康或离线的机器人数量
 * 
 * @param monitor 健康监控器句柄
 * @param healthy_count 输出健康机器人数量
 * @param warning_count 输出警告机器人数量
 * @param offline_count 输出离线机器人数量
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_get_counts(RobotHealthMonitor* monitor,
                                      size_t* healthy_count,
                                      size_t* warning_count,
                                      size_t* offline_count);

/* ============================
 * 预测性维护API（增强T5）
 * ============================ */

/**
 * @brief 获取指定机器人的退化趋势指标
 * 
 * 基于历史响应时间的线性回归分析，计算退化趋势。
 * degradation_score: 0~1，越高表示退化越严重
 * trend_slope: 响应时间变化斜率（正值为恶化）
 *
 * @param monitor 健康监控器句柄
 * @param robot_id 机器人ID
 * @param degradation_score 输出退化评分 (0~1)
 * @param trend_slope 输出趋势斜率
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_get_degradation_trend(RobotHealthMonitor* monitor,
                                                 int robot_id,
                                                 float* degradation_score,
                                                 float* trend_slope);

/**
 * @brief 获取指定机器人的异常评分
 * 
 * 基于Z-score统计方法检测异常行为。
 * anomaly_score: 0~1，越高表示越异常（>0.8视为异常）
 *
 * @param monitor 健康监控器句柄
 * @param robot_id 机器人ID
 * @param anomaly_score 输出异常评分 (0~1)
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_get_anomaly_score(RobotHealthMonitor* monitor,
                                             int robot_id,
                                             float* anomaly_score);

/**
 * @brief 获取预测性维护建议
 * 
 * 综合退化趋势和异常检测，给出预测性维护建议。
 * 返回维护优先级：0=无需维护，1=建议维护，2=需要维护，3=紧急维护
 *
 * @param monitor 健康监控器句柄
 * @param robot_id 机器人ID
 * @param maintenance_priority 输出维护优先级 (0~3)
 * @param estimated_hours_to_failure 输出预计故障时间（小时），-1表示无法预测
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_get_predictive_maintenance(RobotHealthMonitor* monitor,
                                                      int robot_id,
                                                      int* maintenance_priority,
                                                      float* estimated_hours_to_failure);

/**
 * @brief 获取所有机器人的预测性维护摘要
 * 
 * 返回退化最严重机器人的维护建议、健康/警告/离线数。
 *
 * @param monitor 健康监控器句柄
 * @param highest_priority 输出最高维护优先级
 * @param worst_robot_id 输出退化最严重机器人ID
 * @param worst_degradation 输出最高退化评分
 * @param total_robots 输出机器人总数
 * @param healthy_count 输出健康机器人数量
 * @param warning_count 输出警告机器人数量
 * @param offline_count 输出离线机器人数量
 * @return int 成功返回0，失败返回-1
 */
int robot_health_monitor_get_maintenance_summary(RobotHealthMonitor* monitor,
                                                    int* highest_priority,
                                                    int* worst_robot_id,
                                                    float* worst_degradation,
                                                    size_t* total_robots,
                                                    size_t* healthy_count,
                                                    size_t* warning_count,
                                                    size_t* offline_count);

/* ============================
 * PyBullet仿真引擎集成（M16）
 * ============================ */

/**
 * @brief 连接到PyBullet仿真引擎（单个机器人）
 *
 * 初始化并连接到PyBullet Python桥接进程。
 * 连接后，机器人仿真状态由PyBullet物理引擎驱动。
 *
 * @param robot 机器人句柄
 * @param config PyBullet配置（可选，NULL使用默认配置）
 * @return int 成功返回PyBullet body ID，失败返回-1
 */
int robot_connect_pybullet(Robot* robot, const PyBulletConfig* config);

/**
 * @brief 断开PyBullet连接
 *
 * @param robot 机器人句柄
 * @return int 成功返回0，失败返回-1
 */
int robot_disconnect_pybullet(Robot* robot);

/**
 * @brief 检查PyBullet是否已连接
 *
 * @param robot 机器人句柄
 * @return int 已连接返回1，未连接返回0，错误返回-1
 */
int robot_is_pybullet_connected(const Robot* robot);

/**
 * @brief 控制器版本：连接到PyBullet
 *
 * @param controller 控制器句柄
 * @param config PyBullet配置（可选，NULL使用默认配置）
 * @return int 成功返回0，失败返回-1
 */
int robot_controller_connect_pybullet(RobotController* controller, const PyBulletConfig* config);

/**
 * @brief 控制器版本：断开PyBullet连接
 *
 * @param controller 控制器句柄
 * @return int 成功返回0，失败返回-1
 */
int robot_controller_disconnect_pybullet(RobotController* controller);

/**
 * @brief 控制器版本：检查PyBullet是否已连接
 *
 * @param controller 控制器句柄
 * @return int 已连接返回1，未连接返回0，错误返回-1
 */
int robot_controller_is_pybullet_connected(const RobotController* controller);

/**
 * @brief 控制器版本：执行PyBullet仿真步进
 *
 * 替代内部简单物理仿真，使用PyBullet精确物理引擎步进。
 * 自动同步所有已加载机器人的仿真状态。
 *
 * @param controller 控制器句柄
 * @param num_steps 步进次数（每个步进=timestep秒）
 * @return int 成功返回0，失败返回-1
 */
int robot_controller_step_pybullet(RobotController* controller, int num_steps);

/* ============================
 * 硬件自动检测集成（M16）
 * ============================ */

/**
 * @brief 运行硬件自动检测
 *
 * 检测当前系统中所有可用的硬件设备（CPU/GPU/摄像头/串口等），
 * 并将结果缓存到机器人实例中供后续查询。
 *
 * @param robot 机器人句柄
 * @param config 检测配置（可选，NULL使用默认配置）
 * @return int 成功返回检测到的设备数量，失败返回-1
 */
int robot_detect_hardware(Robot* robot, const HDDetectionConfig* config);

/**
 * @brief 获取硬件检测结果
 *
 * @param robot 机器人句柄
 * @param result 输出检测结果
 * @return int 成功返回0，失败返回-1
 */
int robot_get_hardware_result(const Robot* robot, HDDetectionResult* result);

/**
 * @brief 按类型获取检测到的硬件设备
 *
 * @param robot 机器人句柄
 * @param type 设备类型
 * @param out 输出设备信息数组
 * @param max_count 最大输出数量
 * @param count 输出实际数量
 * @return int 成功返回0，失败返回-1
 */
int robot_get_hardware_by_type(const Robot* robot, int type, HDDeviceInfo* out, size_t max_count, size_t* count);

/* ZSFWXJ-FIX011修复: 力顺应控制器(FCC)签名与robot.c实现同步 */
int fcc_init(float stiffness, float damping, float force_limit);
int fcc_compute_command(const float* measured_force, const float* current_pos, const float* current_vel, const float* desired_force, float* output_command, float dt);
int fcc_set_desired_trajectory(const float* position, const float* velocity, const float* force);
int fcc_enable_force_axis(int axis_bitmask);

/* ZSFWXJ-FIX011修复: NMPCConfig前向定义 — 从robot.c移至header供签名引用 */
#define NMPC_MAX_HORIZON 20
typedef struct {
    float Q_diag[6];
    float R_diag[6];
    float u_min[6], u_max[6];
    int horizon;
    float dt;
} NMPCConfig;

/* 非线性模型预测控制 (NMPC) */
int nmpc_init(const NMPCConfig* config);
int nmpc_solve(const float* current_state, const float* reference_traj, float* control_output);

/* 语音到运动控制 */
int robot_voice_to_motion(const float* mfcc_features, int feature_dim, void* main_cfc_network, float* joint_commands, int num_joints, float* confidence);

/* 传感器有效性验证 */
int robot_read_sensor_valid(Robot* robot, int sensor_type, float* value, int* valid);
int robot_get_all_sensor_validity(Robot* robot, int* validity_bitmask);

/* 重连管理 */
int robot_update_reconnect_state_impl(int connection_ok);
int robot_get_reconnect_delay_ms(Robot* robot);

/* EKF传感器融合 */
int robot_ekf_fuse_sensors(const float* imu_accel, const float* imu_gyro, const float* joint_positions, const float* joint_velocities, const float* force_torque, float* fused_state, float* state_covariance);


/* ============================
 * 增强任务分配类型定义
 * ============================ */

#define TASK_ALLOCATION_MAX_TASKS 256
#define TASK_ALLOCATION_MAX_ROBOTS 64

typedef enum {
    TASK_PRIORITY_LOW = 0,
    TASK_PRIORITY_NORMAL = 1,
    TASK_PRIORITY_HIGH = 2,
    TASK_PRIORITY_CRITICAL = 3
} TaskPriority;

typedef enum {
    TASK_ALLOCATION_STATUS_PENDING = 0,
    TASK_ALLOCATION_STATUS_ASSIGNED = 1,
    TASK_ALLOCATION_STATUS_IN_PROGRESS = 2,
    TASK_ALLOCATION_STATUS_COMPLETED = 3,
    TASK_ALLOCATION_STATUS_FAILED = 4,
    TASK_ALLOCATION_STATUS_CANCELLED = 5
} TaskAllocationStatus;

typedef struct {
    char task_id[64];
    char task_type[64];
    float position[3];
    float orientation[4];
    TaskPriority priority;
    float deadline;
    float estimated_duration;
    float required_capabilities[8];
    int required_capabilities_count;
    float reward;
    float penalty;
    int is_preemptive;
    char description[256];
} TaskDescriptor;

typedef struct {
    int robot_id;
    float robot_capabilities[8];
    int capabilities_count;
    float current_position[3];
    float battery_level;
    float estimated_cost;
    float estimated_time;
    float bid_value;
    int task_count;
    float total_load;
    int is_available;
} AllocationCandidate;

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ROBOT_H */