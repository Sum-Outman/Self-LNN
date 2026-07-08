/**
 * @file simulator.h
 * @brief 仿真器接口定义
 * 
 * 提供PyBullet和Gazebo仿真器接口，支持人形机器人仿真控制。
 * 通过进程间通信或网络套接字与外部仿真器进程交互。
 * 遵循100%纯C语言原则，不依赖任何第三方库。
 */

#ifndef SELFLNN_SIMULATOR_H
#define SELFLNN_SIMULATOR_H

#include "selflnn/robot/robot.h"
#include "selflnn/robot/kinematics.h"

struct SensorPipeline;

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 类型定义
 * =========================================================================== */

/**
 * @brief 仿真器类型
 */
typedef enum {
    SIMULATOR_NONE = 0,           /**< 无仿真器 */
    SIMULATOR_PYBULLET = 1,       /**< PyBullet物理仿真器 */
    SIMULATOR_GAZEBO = 2,         /**< Gazebo仿真器 */
    SIMULATOR_SIMPLE = 3,         /**< 简单内部仿真器 */
    SIMULATOR_CUSTOM = 4          /**< 自定义仿真器 */
} SimulatorType;

/**
 * @brief 仿真器状态
 */
typedef enum {
    SIMULATOR_STATE_DISCONNECTED = 0, /**< 未连接 */
    SIMULATOR_STATE_CONNECTING = 1,   /**< 连接中 */
    SIMULATOR_STATE_CONNECTED = 2,    /**< 已连接 */
    SIMULATOR_STATE_RUNNING = 3,      /**< 运行中 */
    SIMULATOR_STATE_PAUSED = 4,       /**< 暂停 */
    SIMULATOR_STATE_ERROR = 5         /**< 错误 */
} SimulatorState;

/**
 * @brief 仿真器配置
 */
typedef struct {
    SimulatorType type;           /**< 仿真器类型 */
    char name[64];                /**< 仿真器名称 */
    
    // 连接配置
    char hostname[256];           /**< 主机名或IP地址 */
    int port;                     /**< 端口号 */
    int timeout_ms;               /**< 超时时间（毫秒） */
    int retry_count;              /**< 重试次数 */
    int retry_delay_ms;           /**< 重试延迟（毫秒） */
    
    // 仿真参数
    float timestep;               /**< 时间步长（秒） */
    float gravity;                /**< 重力加速度（m/s²） */
    int enable_visualization;     /**< 是否启用可视化 */
    int enable_gui;               /**< 是否启用GUI */
    
    // 机器人模型
    char robot_model[256];        /**< 机器人模型文件路径 */
    float initial_position[3];    /**< 初始位置（x, y, z） */
    float initial_orientation[4]; /**< 初始姿态（四元数） */
    int use_urdf;                 /**< 是否使用URDF文件 */
    
    // 物理参数
    int enable_real_time_simulation; /**< 是否启用实时仿真 */
    float real_time_factor;       /**< 实时因子 */
    int num_solver_iterations;    /**< 求解器迭代次数 */
    
    // 传感器配置
    int enable_sensors;           /**< 是否启用传感器 */
    int sensor_update_rate;       /**< 传感器更新频率（Hz） */
    
    // 日志和调试
    int enable_logging;           /**< 是否启用日志记录 */
    char log_directory[256];      /**< 日志目录 */
} SimulatorConfig;

/**
 * @brief 仿真器状态信息
 */
typedef struct {
    SimulatorState state;         /**< 仿真器状态 */
    float simulation_time;        /**< 仿真时间（秒） */
    float real_time;              /**< 真实时间（秒） */
    int step_count;               /**< 步进计数 */
    float frame_rate;             /**< 帧率（Hz） */
    float physics_update_time;    /**< 物理更新时间（秒） */
    float render_time;            /**< 渲染时间（秒） */
    int num_robots;               /**< 机器人数量 */
    int num_sensors;              /**< 传感器数量 */
    int num_constraints;          /**< 约束数量 */
    int num_contacts;             /**< 接触点数量 */
    float cpu_usage;              /**< CPU使用率（0.0-1.0） */
    float memory_usage;           /**< 内存使用率（0.0-1.0） */
    char last_error[256];         /**< 最后错误信息 */
} SimulatorStatus;

/**
 * @brief 仿真器机器人状态
 */
typedef struct {
    int robot_id;                 /**< 机器人ID */
    char robot_name[64];          /**< 机器人名称 */
    float position[3];            /**< 位置（x, y, z） */
    float velocity[3];            /**< 速度（m/s） */
    float acceleration[3];        /**< 加速度（m/s²） */
    float orientation[4];         /**< 姿态（四元数） */
    float angular_velocity[3];    /**< 角速度（rad/s） */
    float joint_positions[32];    /**< 关节位置（rad） */
    float joint_velocities[32];   /**< 关节速度（rad/s） */
    float joint_torques[32];      /**< 关节力矩（Nm） */
    float target_joint_positions[32]; /**< PD控制器目标关节位置（rad） */
    float contact_forces[6];      /**< 接触力（N） */
    float contact_positions[18];  /**< 接触位置（x, y, z）x6 */
    int contact_flags[6];         /**< 接触标志 */
    float battery_level;          /**< 电池电量（0.0-1.0） */
    int is_colliding;             /**< 是否发生碰撞 */
    int num_links;                /**< 链接数量 */
    int num_joints;               /**< 关节数量 */
    int motion_mode;              /**< 当前运动控制模式(MotionMode),用于PD控制器与力矩模式切换 */
} SimulatorRobotState;

/**
 * @brief 仿真器传感器数据
 */
typedef struct {
    int sensor_id;                /**< 传感器ID */
    char sensor_name[64];         /**< 传感器名称 */
    RobotSensorType sensor_type;  /**< 传感器类型 */
    float timestamp;              /**< 时间戳（秒） */
    float data[64];               /**< 传感器数据 */
    int data_size;                /**< 数据大小 */
    float noise_level;            /**< 噪声级别 */
    int is_valid;                 /**< 数据是否有效 */
/* 传感器空间位姿 — 存储mount_position/mount_orientation */
    float position[3];            /**< 安装位置 (x,y,z) */
    float orientation[4];         /**< 安装姿态 (四元数 w,x,y,z) */
    int robot_id;                 /**< 所属机器人ID */
} SimulatorSensorData;

/**
 * @brief 仿真器场景对象
 */
typedef struct {
    int object_id;                /**< 对象ID */
    char object_name[64];         /**< 对象名称 */
    int object_type;              /**< 对象类型（0:静态，1:动态，2:传感器） */
    float position[3];            /**< 位置 */
    float orientation[4];         /**< 姿态 */
    float scale[3];               /**< 缩放 */
    float color[4];               /**< 颜色（RGBA） */
    float mass;                   /**< 质量（kg） */
    float friction;               /**< 摩擦系数 */
    float restitution;            /**< 恢复系数 */
    char model_file[256];         /**< 模型文件路径 */
    int is_dynamic;               /**< 是否动态对象（1=动态，0=静态） */
} SimulatorSceneObject;

/* ============================================================================
 * 增强物理引擎类型定义（用于内部物理管道）
 * ============================================================================ */

#define SIM_MAX_COLLISION_SHAPES 128
#define SIM_MAX_CONTACTS 64
#define SIM_MAX_CONSTRAINTS 256
#define SIM_MAX_JOINTS 64
#define SIM_MAX_BROADPHASE_PAIRS 512
#define SIM_SOLVER_ITERATIONS 10
#define SIM_BAUMGARTE_FACTOR 0.2f

typedef struct {
    int object_id;
    int is_robot;
    CollisionShape shape;
    float world_transform[7];
    float aabb_min[3];
    float aabb_max[3];
    float inv_mass;
    float inv_inertia[3];
    float radius;              /* 碰撞体包围球半径（传感器模拟用） */
    float velocity[3];         /* 碰撞体线速度（用于碰撞求解器） */
    float angular_velocity[3]; /* 碰撞体角速度（用于碰撞求解器） */
    int active;
} SimCollisionObject;

typedef struct {
    float position[3];
    float normal[3];
    float penetration;
    float friction_coeff;
    float restitution;
    int body_a;
    int body_b;
    float impulse_normal;
    float impulse_tangent[2];
    int active;
} SimContactPoint;

typedef struct {
    int type;
    int body_a;
    int body_b;
    union {
        SimContactPoint contact;
        struct {
            float pivot_a[3];
            float pivot_b[3];
            float axis_a[3];
            float axis_b[3];
            float limit_lower;
            float limit_upper;
            float max_force;
        } joint_c;
    } data;
    int active;
} SimConstraint;

typedef enum {
    SIM_JOINT_HINGE = 0,
    SIM_JOINT_BALL = 1,
    SIM_JOINT_SLIDER = 2,
    SIM_JOINT_FIXED = 3
} SimJointType;

typedef struct {
    SimJointType type;
    int body_a;
    int body_b;
    float pivot_a[3];
    float pivot_b[3];
    float axis_a[3];
    float axis_b[3];
    float limit_lower;
    float limit_upper;
    float max_force;
    int active;
} SimJoint;

typedef struct {
    int a;
    int b;
} SimBroadphasePair;

typedef struct {
    SimCollisionObject objects[128];
    int object_count;
    SimContactPoint contacts[64];
    int contact_count;
    SimConstraint constraints[256];
    int constraint_count;
    SimBroadphasePair pairs[512];
    int pair_count;
    SimJoint joints[64];
    int joint_count;
} SimPhysicsPipeline;

/**
 * @brief 仿真器句柄
 */
typedef struct Simulator Simulator;

/* ============================================================================
 * 仿真器管理函数
 * =========================================================================== */

/**
 * @brief 创建仿真器实例
 * 
 * @param config 仿真器配置
 * @return Simulator* 仿真器句柄，失败返回NULL
 */
Simulator* simulator_create(const SimulatorConfig* config);

/**
 * @brief 销毁仿真器实例
 * 
 * @param simulator 仿真器句柄
 */
void simulator_destroy(Simulator* simulator);

/**
 * @brief 连接仿真器
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_connect(Simulator* simulator);

/**
 * @brief 断开仿真器连接
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_disconnect(Simulator* simulator);

/**
 * @brief 启动仿真器
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_start(Simulator* simulator);

/**
 * @brief 停止仿真器
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_stop(Simulator* simulator);

/**
 * @brief 暂停仿真器
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_pause(Simulator* simulator);

/**
 * @brief 恢复仿真器
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_resume(Simulator* simulator);

/**
 * @brief 重置仿真器
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_reset(Simulator* simulator);

/**
 * @brief 步进仿真器（单步）
 * 
 * @param simulator 仿真器句柄
 * @param num_steps 步进次数
 * @return int 成功返回0，失败返回-1
 */
int simulator_step(Simulator* simulator, int num_steps);

/**
 * @brief 获取仿真器状态
 * 
 * @param simulator 仿真器句柄
 * @param status 状态信息输出
 * @return int 成功返回0，失败返回-1
 */
int simulator_get_status(Simulator* simulator, SimulatorStatus* status);

/* ============================================================================
 * 机器人管理函数
 * =========================================================================== */

/**
 * @brief 在仿真器中加载机器人
 * 
 * @param simulator 仿真器句柄
 * @param robot_config 机器人配置
 * @param initial_pose 初始位姿
 * @return int 机器人ID，失败返回-1
 */
int simulator_load_robot(Simulator* simulator, const RobotConfig* robot_config, const float* initial_pose);

/**
 * @brief 从仿真器中移除机器人
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @return int 成功返回0，失败返回-1
 */
int simulator_remove_robot(Simulator* simulator, int robot_id);

/**
 * @brief 获取仿真器中的机器人状态
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param state 机器人状态输出
 * @return int 成功返回0，失败返回-1
 */
int simulator_get_robot_state(Simulator* simulator, int robot_id, SimulatorRobotState* state);

/**
 * @brief 设置机器人关节目标位置
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param joint_positions 关节目标位置数组
 * @param num_joints 关节数量
 * @return int 成功返回0，失败返回-1
 */
int simulator_set_joint_positions(Simulator* simulator, int robot_id, const float* joint_positions, int num_joints);

/**
 * @brief 设置机器人关节目标速度
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param joint_velocities 关节目标速度数组
 * @param num_joints 关节数量
 * @return int 成功返回0，失败返回-1
 */
int simulator_set_joint_velocities(Simulator* simulator, int robot_id, const float* joint_velocities, int num_joints);

/**
 * @brief 设置机器人关节目标力矩
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param joint_torques 关节目标力矩数组
 * @param num_joints 关节数量
 * @return int 成功返回0，失败返回-1
 */
int simulator_set_joint_torques(Simulator* simulator, int robot_id, const float* joint_torques, int num_joints);

/**
 * @brief 应用机器人控制命令
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param command 控制命令
 * @return int 成功返回0，失败返回-1
 */
int simulator_apply_robot_command(Simulator* simulator, int robot_id, const RobotCommand* command);

/* ============================================================================
 * 传感器管理函数
 * =========================================================================== */

/**
 * @brief 在仿真器中添加传感器
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param sensor_config 传感器配置
 * @param mount_position 安装位置
 * @param mount_orientation 安装姿态
 * @return int 传感器ID，失败返回-1
 */
int simulator_add_sensor(Simulator* simulator, int robot_id, const RobotSensorConfig* sensor_config, 
                         const float* mount_position, const float* mount_orientation);

/**
 * @brief 从仿真器中移除传感器
 * 
 * @param simulator 仿真器句柄
 * @param sensor_id 传感器ID
 * @return int 成功返回0，失败返回-1
 */
int simulator_remove_sensor(Simulator* simulator, int sensor_id);

/**
 * @brief 获取传感器数据
 * 
 * @param simulator 仿真器句柄
 * @param sensor_id 传感器ID
 * @param sensor_data 传感器数据输出
 * @return int 成功返回0，失败返回-1
 */
int simulator_get_sensor_data(Simulator* simulator, int sensor_id, SimulatorSensorData* sensor_data);

/**
 * @brief 获取机器人所有传感器数据
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param sensor_data_array 传感器数据数组
 * @param max_sensors 最大传感器数量
 * @return int 实际传感器数量，失败返回-1
 */
int simulator_get_all_sensor_data(Simulator* simulator, int robot_id, 
                                  SimulatorSensorData* sensor_data_array, int max_sensors);

/* ============================================================================
 * 场景管理函数
 * =========================================================================== */

/**
 * @brief 在仿真器中添加场景对象
 * 
 * @param simulator 仿真器句柄
 * @param object 场景对象
 * @return int 对象ID，失败返回-1
 */
int simulator_add_scene_object(Simulator* simulator, const SimulatorSceneObject* object);

/**
 * @brief 从仿真器中移除场景对象
 * 
 * @param simulator 仿真器句柄
 * @param object_id 对象ID
 * @return int 成功返回0，失败返回-1
 */
int simulator_remove_scene_object(Simulator* simulator, int object_id);

/**
 * @brief 获取场景对象列表
 * 
 * @param simulator 仿真器句柄
 * @param objects 对象数组输出
 * @param max_objects 最大对象数量
 * @return int 实际对象数量，失败返回-1
 */
int simulator_get_scene_objects(Simulator* simulator, SimulatorSceneObject* objects, int max_objects);

/**
 * @brief 设置场景重力
 * 
 * @param simulator 仿真器句柄
 * @param gravity 重力向量（x, y, z）
 * @return int 成功返回0，失败返回-1
 */
int simulator_set_gravity(Simulator* simulator, const float* gravity);

/**
 * @brief 设置场景光照
 * 
 * @param simulator 仿真器句柄
 * @param light_position 光源位置
 * @param light_color 光源颜色
 * @param ambient_color 环境光颜色
 * @return int 成功返回0，失败返回-1
 */
int simulator_set_lighting(Simulator* simulator, const float* light_position, 
                           const float* light_color, const float* ambient_color);

/* ============================================================================
 * 数据导出和记录函数
 * =========================================================================== */

/**
 * @brief 开始记录仿真数据
 * 
 * @param simulator 仿真器句柄
 * @param filename 记录文件名
 * @return int 成功返回0，失败返回-1
 */
int simulator_start_recording(Simulator* simulator, const char* filename);

/**
 * @brief 停止记录仿真数据
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_stop_recording(Simulator* simulator);

/**
 * @brief 导出仿真场景
 * 
 * @param simulator 仿真器句柄
 * @param filename 导出文件名
 * @return int 成功返回0，失败返回-1
 */
int simulator_export_scene(Simulator* simulator, const char* filename);

/**
 * @brief 获取最后错误信息
 * 
 * @param simulator 仿真器句柄
 * @return const char* 错误信息字符串
 */
const char* simulator_get_last_error(Simulator* simulator);

/* ============================================================================
 * 训练API — 人形机器人仿真训练控制 (PyBullet/Gazebo)
 * =========================================================================== */

/**
 * @brief 训练模式枚举
 */
typedef enum {
    TRAINING_MODE_NONE = 0,                 /**< 无训练 */
    TRAINING_MODE_IMITATION = 1,            /**< 模仿学习 */
    TRAINING_MODE_REINFORCEMENT = 2,        /**< 强化学习 */
    TRAINING_MODE_MOVEMENT_PRIMITIVES = 3,  /**< 运动基元 */
    TRAINING_MODE_JOINT_SPACE = 4,          /**< 关节空间控制 */
    TRAINING_MODE_TASK_SPACE = 5            /**< 任务空间控制 */
} SimulatorTrainingMode;

/**
 * @brief 训练状态结构体
 */
typedef struct {
    SimulatorTrainingMode mode;    /**< 训练模式 */
    int is_active;                 /**< 是否激活 */
    int is_paused;                 /**< 是否暂停 */
    int episode;                   /**< 当前轮次 */
    int max_episodes;              /**< 最大轮次 */
    int step;                      /**< 当前步数 */
    int max_steps_per_episode;     /**< 每轮最大步数 */
    float reward;                  /**< 当前奖励 */
    float avg_reward;              /**< 平均奖励 */
    float best_reward;             /**< 最佳奖励 */
    float loss;                    /**< 当前损失 */
    float avg_loss;                /**< 平均损失 */
    float exploration_rate;        /**< 探索率（强化学习） */
    float learning_rate;           /**< 学习率 */
    int total_samples;             /**< 总样本数 */
    int successful_episodes;       /**< 成功轮次 */
    int failed_episodes;           /**< 失败轮次 */
    double elapsed_time;           /**< 经过时间（秒） */
    double estimated_time_remaining; /**< 预计剩余时间（秒） */
    char description[128];         /**< 训练描述 */
    char status_message[256];      /**< 状态消息 */
} SimulatorTrainingStatus;

/**
 * @brief 训练样本结构体
 */
typedef struct {
    double timestamp;                   /**< 时间戳 */
    float joint_positions[32];          /**< 关节位置 */
    float joint_velocities[32];         /**< 关节速度 */
    float joint_torques[32];            /**< 关节力矩 */
    float position[3];                  /**< 基础位置 */
    float orientation[4];               /**< 基础姿态 */
    float linear_velocity[3];           /**< 线速度 */
    float angular_velocity[3];          /**< 角速度 */
    float contact_forces[6];            /**< 接触力 */
    float com_position[3];              /**< 质心位置 */
    float zmp_position[2];              /**< ZMP位置 */
    float reward;                       /**< 奖励值 */
    float action[32];                   /**< 动作向量 */
    float observation[128];             /**< 观测向量 */
    int is_terminal;                    /**< 是否终止状态 */
    int is_success;                     /**< 是否成功 */
} SimulatorTrainingSample;

/**
 * @brief 机器人详细信息结构体
 */
typedef struct {
    int robot_id;                        /**< 机器人ID */
    char robot_name[64];                 /**< 机器人名称 */
    RobotType type;                      /**< 机器人类型 */
    int num_dof;                         /**< 自由度数量 */
    int num_joints;                      /**< 关节数量 */
    float joint_limits_lower[32];        /**< 关节下限 */
    float joint_limits_upper[32];        /**< 关节上限 */
    float joint_max_velocity[32];        /**< 关节最大速度 */
    float joint_max_torque[32];          /**< 关节最大力矩 */
    float link_masses[32];               /**< 连杆质量 */
    float total_mass;                    /**< 总质量 */
    float height;                        /**< 高度 */
    float foot_size[2];                  /**< 脚底尺寸 (x, y) */
    float com_offset[3];                 /**< 质心偏移 */
    float default_standing_pose[32];     /**< 默认站立姿态 */
    int has_gripper;                     /**< 是否有夹爪 */
    char urdf_path[256];                 /**< URDF文件路径 */
    float end_effector_position[3];      /**< 末端执行器位置 */
    float end_effector_orientation[4];   /**< 末端执行器姿态 */
} SimulatorRobotInfo;

/**
 * @brief 接触信息结构体
 */
typedef struct {
    int contact_count;                   /**< 接触点数量 */
    float positions[6][3];               /**< 接触位置 */
    float forces[6][3];                  /**< 接触力 */
    float normals[6][3];                 /**< 接触法线 */
    float friction_coeffs[6];            /**< 摩擦系数 */
    int body_ids[6];                     /**< 接触体ID */
    int link_ids[6];                     /**< 接触连杆ID */
    int is_foot_contact[6];              /**< 是否为脚底接触 */
    double timestamps[6];                /**< 接触时间戳 */
    float total_normal_force;            /**< 总法向力 */
    float total_friction_force;          /**< 总摩擦力 */
} SimulatorContactInfo;

/**
 * @brief 电机PD增益结构体
 */
typedef struct {
    float kp;                            /**< 比例增益 */
    float kd;                            /**< 微分增益 */
    float ki;                            /**< 积分增益 */
    float max_force;                     /**< 最大力 */
    float max_velocity;                  /**< 最大速度 */
    float target_position_tolerance;     /**< 目标位置容差 */
    float damping_coefficient;           /**< 阻尼系数 */
} SimulatorMotorPDGains;

/**
 * @brief 物理参数结构体
 */
typedef struct {
    float gravity[3];                    /**< 重力向量 */
    float timestep;                      /**< 时间步长 */
    int num_solver_iterations;           /**< 求解器迭代次数 */
    float contact_erp;                   /**< 接触ERP */
    float contact_cfm;                   /**< 接触CFM */
    float friction_coefficient;          /**< 摩擦系数 */
    float restitution_coefficient;       /**< 恢复系数 */
    float linear_damping;                /**< 线性阻尼 */
    float angular_damping;               /**< 角阻尼 */
    float max_contact_correction_vel;    /**< 最大接触修正速度 */
    float global_physics_scale;          /**< 物理缩放 */
} SimulatorPhysicsParams;

/**
 * @brief 训练记录条目
 */
typedef struct {
    double timestamp;
    int robot_id;
    int episode;
    int step;
    SimulatorTrainingSample sample;
} SimulatorTrainingRecord;

/* ============================================================================
 * 二进制高速仿真通信协议
 * ============================================================================ */

#define SIM_BINARY_MAGIC  0x534C534D  /* "SLSM" */
#define SIM_BINARY_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t packet_type;       /* 0=状态更新, 1=命令, 2=传感器, 3=应答 */
    uint32_t packet_size;
    uint64_t timestamp_us;
    uint32_t robot_count;
    uint32_t sensor_count;
    uint32_t object_count;
    uint32_t reserved;
} SimBinaryHeader;

typedef struct {
    SimBinaryHeader header;
    float position[3];
    float orientation[4];
    float linear_velocity[3];
    float angular_velocity[3];
    float joint_positions[32];
    float joint_velocities[32];
    float motor_torques[32];
    float gripper_position;
    float gripper_force;
} SimBinaryRobotState;

int simulator_send_binary_state(Simulator* sim, int robot_index, 
                                SimBinaryRobotState* state);
int simulator_recv_binary_command(Simulator* sim, SimBinaryRobotState* cmd);
int simulator_binary_poll_sensors(Simulator* sim, float* sensor_data, 
                                  size_t* sensor_count);

/* ============================================================================
 * 增强API函数 — 训练控制
 * ============================================================================ */

/**
 * @brief 启动训练
 * 
 * @param simulator 仿真器句柄
 * @param mode 训练模式
 * @param max_episodes 最大轮次
 * @param max_steps_per_episode 每轮最大步数
 * @param description 训练描述
 * @return int 成功返回0，失败返回-1
 */
int simulator_start_training(Simulator* simulator, SimulatorTrainingMode mode,
                              int max_episodes, int max_steps_per_episode,
                              const char* description);

/**
 * @brief 停止训练
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_stop_training(Simulator* simulator);

/**
 * @brief 暂停训练（暂停后可恢复）
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_pause_training(Simulator* simulator);

/**
 * @brief 恢复训练
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_resume_training(Simulator* simulator);

/**
 * @brief 获取训练状态
 * 
 * @param simulator 仿真器句柄
 * @param status 训练状态输出
 * @return int 成功返回0，失败返回-1
 */
int simulator_get_training_status(Simulator* simulator, SimulatorTrainingStatus* status);

/**
 * @brief 添加训练样本到记录
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param sample 训练样本
 * @return int 成功返回0，失败返回-1
 */
int simulator_add_training_sample(Simulator* simulator, int robot_id,
                                   const SimulatorTrainingSample* sample);

/**
 * @brief 获取训练数据记录
 * 
 * @param simulator 仿真器句柄
 * @param records 记录数组输出
 * @param count 输入最大记录数，输出实际记录数
 * @return int 成功返回0，失败返回-1
 */
int simulator_get_training_records(Simulator* simulator, SimulatorTrainingRecord* records, int* count);

/**
 * @brief 回放训练记录
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 目标机器人ID
 * @param records 训练记录数组
 * @param count 记录数量
 * @return int 成功返回0，失败返回-1
 */
int simulator_replay_training(Simulator* simulator, int robot_id,
                               const SimulatorTrainingRecord* records, int count);

/**
 * @brief 导出训练数据到文件
 * 
 * @param simulator 仿真器句柄
 * @param filename 导出文件名
 * @return int 成功返回0，失败返回-1
 */
int simulator_export_training_data(Simulator* simulator, const char* filename);

/**
 * @brief 从文件导入训练数据
 * 
 * @param simulator 仿真器句柄
 * @param filename 导入文件名
 * @return int 成功返回0，失败返回-1
 */
int simulator_import_training_data(Simulator* simulator, const char* filename);

/* ============================================================================
 * 增强API函数 — URDF加载与机器人信息
 * =========================================================================== */

/**
 * @brief 从URDF文件加载机器人到仿真器
 * 
 * @param simulator 仿真器句柄
 * @param urdf_path URDF文件路径
 * @param initial_pose 初始位姿 [x,y,z,qx,qy,qz,qw]
 * @param robot_name 机器人名称
 * @return int 机器人ID，失败返回-1
 */
int simulator_load_urdf(Simulator* simulator, const char* urdf_path,
                         const float* initial_pose, const char* robot_name);

/**
 * @brief 获取机器人详细信息
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param info 机器人信息输出
 * @return int 成功返回0，失败返回-1
 */
int simulator_get_robot_info(Simulator* simulator, int robot_id, SimulatorRobotInfo* info);

/**
 * @brief 获取接触信息
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param contact_info 接触信息输出
 * @return int 成功返回0，失败返回-1
 */
int simulator_get_contact_info(Simulator* simulator, int robot_id, SimulatorContactInfo* contact_info);

/**
 * @brief 重置机器人位姿到初始状态
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param pose 目标位姿 [x,y,z,qx,qy,qz,qw]，NULL则使用默认位姿
 * @return int 成功返回0，失败返回-1
 */
int simulator_reset_robot_pose(Simulator* simulator, int robot_id, const float* pose);

/* ============================================================================
 * 增强API函数 — 物理参数与电机控制
 * =========================================================================== */

/**
 * @brief 设置重力向量（任意方向）
 * 
 * @param simulator 仿真器句柄
 * @param gravity 重力向量 (x, y, z) m/s²
 * @return int 成功返回0，失败返回-1
 */
int simulator_set_gravity_vector(Simulator* simulator, const float* gravity);

/**
 * @brief 设置电机PD增益
 * 
 * @param simulator 仿真器句柄
 * @param robot_id 机器人ID
 * @param joint_index 关节索引，-1表示所有关节
 * @param gains PD增益参数
 * @return int 成功返回0，失败返回-1
 */
int simulator_set_motor_pd_gains(Simulator* simulator, int robot_id,
                                  int joint_index, const SimulatorMotorPDGains* gains);

/**
 * @brief 设置物理参数
 * 
 * @param simulator 仿真器句柄
 * @param params 物理参数
 * @return int 成功返回0，失败返回-1
 */
int simulator_set_physics_params(Simulator* simulator, const SimulatorPhysicsParams* params);

/**
 * @brief 获取物理参数
 * 
 * @param simulator 仿真器句柄
 * @param params 物理参数输出
 * @return int 成功返回0，失败返回-1
 */
int simulator_get_physics_params(Simulator* simulator, SimulatorPhysicsParams* params);

/* ============================================================================
 * 增强API函数 — 传感器流连接（与sensor_pipeline集成）
 * =========================================================================== */

/**
 * @brief 连接仿真器到传感器管道
 * 
 * @param simulator 仿真器句柄
 * @param pipeline 传感器管道句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_attach_sensor_pipeline(Simulator* simulator, struct SensorPipeline* pipeline);

/**
 * @brief 断开传感器管道连接
 * 
 * @param simulator 仿真器句柄
 * @return int 成功返回0，失败返回-1
 */
int simulator_detach_sensor_pipeline(Simulator* simulator);

/**
 * @brief 设置传感器流推送（每次物理步进自动推送传感器数据到管道）
 * 
 * @param simulator 仿真器句柄
 * @param enable 1启用，0禁用
 * @return int 成功返回0，失败返回-1
 */
int simulator_enable_sensor_streaming(Simulator* simulator, int enable);

/* ============================================================================
 * 增强API函数 — 仿真数据导出
 * =========================================================================== */

/**
 * @brief 导出仿真器当前场景和机器人状态到JSON格式
 * 
 * @param simulator 仿真器句柄
 * @param filename 导出文件路径
 * @return int 成功返回0，失败返回-1
 */
int simulator_export_scene_json(Simulator* simulator, const char* filename);

/**
 * @brief 导出仿真器统计数据到CSV
 * 
 * @param simulator 仿真器句柄
 * @param filename 导出文件路径
 * @return int 成功返回0，失败返回-1
 */
int simulator_export_statistics(Simulator* simulator, const char* filename);

/* ============================================================================
 * 仿真器接口表（用于多态仿真器管理）
 * =========================================================================== */

/**
 * @brief 仿真器接口函数表
 * 
 * 提供统一的仿真器操作接口，支持多态（PyBullet/Gazebo/内部仿真器）。
 */
typedef struct {
    SimulatorType simulator_type;                    /**< 仿真器类型 */
    Simulator* (*create)(const SimulatorConfig*);    /**< 创建仿真器 */
    void (*destroy)(Simulator*);                     /**< 销毁仿真器 */
    int (*connect)(Simulator*);                      /**< 连接仿真器 */
    int (*disconnect)(Simulator*);                   /**< 断开仿真器 */
    int (*start)(Simulator*);                        /**< 启动仿真器 */
    int (*stop)(Simulator*);                         /**< 停止仿真器 */
    int (*pause)(Simulator*);                        /**< 暂停仿真器 */
    int (*resume)(Simulator*);                       /**< 恢复仿真器 */
    int (*reset)(Simulator*);                        /**< 重置仿真器 */
    int (*step)(Simulator*, int);                    /**< 步进仿真器 */
    int (*get_status)(Simulator*, SimulatorStatus*); /**< 获取状态 */
    int (*load_robot)(Simulator*, const RobotConfig*, const float*); /**< 加载机器人 */
    int (*remove_robot)(Simulator*, int);            /**< 移除机器人 */
    int (*get_robot_state)(Simulator*, int, SimulatorRobotState*); /**< 获取机器人状态 */
    int (*set_joint_positions)(Simulator*, int, const float*, int); /**< 设置关节位置 */
    int (*set_joint_velocities)(Simulator*, int, const float*, int); /**< 设置关节速度 */
    int (*set_joint_torques)(Simulator*, int, const float*, int); /**< 设置关节力矩 */
    int (*apply_robot_command)(Simulator*, int, const RobotCommand*); /**< 应用控制命令 */
    int (*add_sensor)(Simulator*, int, const RobotSensorConfig*, const float*, const float*); /**< 添加传感器 */
    int (*remove_sensor)(Simulator*, int);           /**< 移除传感器 */
    int (*get_sensor_data)(Simulator*, int, SimulatorSensorData*); /**< 获取传感器数据 */
    int (*get_all_sensor_data)(Simulator*, int, SimulatorSensorData*, int); /**< 获取全部传感器数据 */
    int (*add_scene_object)(Simulator*, const SimulatorSceneObject*); /**< 添加场景对象 */
    int (*remove_scene_object)(Simulator*, int);     /**< 移除场景对象 */
    int (*get_scene_objects)(Simulator*, SimulatorSceneObject*, int); /**< 获取场景对象 */
    int (*set_gravity)(Simulator*, const float*);    /**< 设置重力 */
    int (*set_lighting)(Simulator*, const float*, const float*, const float*); /**< 设置光照 */
    int (*start_recording)(Simulator*, const char*); /**< 开始记录 */
    int (*stop_recording)(Simulator*);               /**< 停止记录 */
    int (*export_scene)(Simulator*, const char*);    /**< 导出场景 */
    const char* (*get_last_error)(Simulator*);       /**< 获取错误信息 */
    
    /* 训练API */
    int (*start_training)(Simulator*, SimulatorTrainingMode, int, int, const char*);
    int (*stop_training)(Simulator*);
    int (*pause_training)(Simulator*);
    int (*resume_training)(Simulator*);
    int (*get_training_status)(Simulator*, SimulatorTrainingStatus*);
    int (*add_training_sample)(Simulator*, int, const SimulatorTrainingSample*);
    int (*get_training_records)(Simulator*, SimulatorTrainingRecord*, int*);
    int (*replay_training)(Simulator*, int, const SimulatorTrainingRecord*, int);
    int (*export_training_data)(Simulator*, const char*);
    int (*import_training_data)(Simulator*, const char*);
    
    /* URDF加载与机器人信息 */
    int (*load_urdf)(Simulator*, const char*, const float*, const char*);
    int (*get_robot_info)(Simulator*, int, SimulatorRobotInfo*);
    int (*get_contact_info)(Simulator*, int, SimulatorContactInfo*);
    int (*reset_robot_pose)(Simulator*, int, const float*);
    
    /* 物理参数与电机控制 */
    int (*set_gravity_vector)(Simulator*, const float*);
    int (*set_motor_pd_gains)(Simulator*, int, int, const SimulatorMotorPDGains*);
    int (*set_physics_params)(Simulator*, const SimulatorPhysicsParams*);
    int (*get_physics_params)(Simulator*, SimulatorPhysicsParams*);
    
    /* 传感器流连接 */
    int (*attach_sensor_pipeline)(Simulator*, struct SensorPipeline*);
    int (*detach_sensor_pipeline)(Simulator*);
    int (*enable_sensor_streaming)(Simulator*, int);
    
    /* 仿真数据导出 */
    int (*export_scene_json)(Simulator*, const char*);
    int (*export_statistics)(Simulator*, const char*);
} SimulatorInterface;

/**
 * @brief 获取Gazebo仿真器接口表
 * 
 * @return const SimulatorInterface* 接口表指针
 */
const SimulatorInterface* gazebo_get_simulator_interface(void);

/* 仿真器辅助操作接口 */
int simulator_set_camera_view(void* sim, float x, float y, float z, float target_x, float target_y, float target_z);
int simulator_toggle_grid_display(void* sim, int enable);
int simulator_add_robot(void* sim, const char* name);
int simulator_clear_scene(void* sim);
int simulator_plan_path(void* sim, const float* start, const float* goal, float* waypoints, int max_wp);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_SIMULATOR_H */