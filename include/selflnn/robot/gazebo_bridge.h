/**
 * @file gazebo_bridge.h
 * @brief SELF-LNN 与 Gazebo 机器人仿真器的桥接接口
 *
 * 根据需求更新（2026-05-11），Gazebo 允许使用 C++ 依赖。
 * 本桥接模块通过 gazebo-ros 或直接 Gazebo C++ API 进行通信，
 * 提供仿真环境的 C 语言接口。
 *
 * 当 Gazebo 环境不可用时，回退到 simulator.c 内部仿真。
 */

#ifndef SELFLNN_GAZEBO_BRIDGE_H
#define SELFLNN_GAZEBO_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Gazebo 连接状态 */
typedef enum {
    GAZEBO_DISCONNECTED = 0,    /**< 未连接 */
    GAZEBO_CONNECTING = 1,      /**< 连接中 */
    GAZEBO_CONNECTED = 2,       /**< 已连接 */
    GAZEBO_PAUSED = 3,          /**< 已暂停 */
    GAZEBO_ERROR = 4            /**< 错误 */
} GazeboConnectionState;

/** @brief Gazebo 仿真配置 */
typedef struct {
    char* world_file;            /**< .world 世界文件路径 */
    int start_paused;            /**< 是否启动时暂停 */
    float real_time_factor;      /**< 实时因子（1.0=实时） */
    float max_step_size;         /**< 最大步长（秒） */
    int use_gui;                 /**< 是否启动GUI */
    int use_gazebo_ros;          /**< 是否通过 gazebo-ros 连接 */
    int server_port;             /**< Gazebo 服务器端口 */
} GazeboConfig;

/** @brief 模型状态 */
typedef struct {
    char model_name[256];        /**< 模型名称 */
    float position[3];           /**< 位置 x,y,z */
    float orientation[4];        /**< 方向四元数 x,y,z,w */
    float linear_velocity[3];    /**< 线速度 */
    float angular_velocity[3];   /**< 角速度 */
} GazeboModelState;

/** @brief Gazebo 桥接句柄 */
typedef struct GazeboBridge GazeboBridge;

/**
 * @brief 检测 Gazebo 是否可用
 * @return 1=可用，0=不可用
 */
int gazebo_is_available(void);

/**
 * @brief 创建 Gazebo 仿真连接
 * @param config 仿真配置
 * @return 桥接句柄，失败返回NULL
 */
GazeboBridge* gazebo_connect(const GazeboConfig* config);

/**
 * @brief 断开 Gazebo 仿真连接
 * @param bridge 桥接句柄
 */
void gazebo_disconnect(GazeboBridge* bridge);

/**
 * @brief 获取连接状态
 * @param bridge 桥接句柄
 * @return 连接状态
 */
GazeboConnectionState gazebo_get_state(GazeboBridge* bridge);

/**
 * @brief 生成模型到仿真中
 * @param bridge 桥接句柄
 * @param model_name 模型名称
 * @param sdf_path SDF模型文件路径
 * @param position 初始位置 [x,y,z]
 * @param orientation 初始方向（欧拉角）[roll,pitch,yaw]
 * @return 0成功，-1失败
 */
int gazebo_spawn_model(GazeboBridge* bridge, const char* model_name,
                       const char* sdf_path,
                       const float position[3],
                       const float orientation[3]);

/**
 * @brief 删除仿真中的模型
 * @param bridge 桥接句柄
 * @param model_name 模型名称
 * @return 0成功，-1失败
 */
int gazebo_delete_model(GazeboBridge* bridge, const char* model_name);

/**
 * @brief 获取模型状态
 * @param bridge 桥接句柄
 * @param model_name 模型名称
 * @param state 输出模型状态
 * @return 0成功，-1失败
 */
int gazebo_get_model_state(GazeboBridge* bridge, const char* model_name,
                           GazeboModelState* state);

/**
 * @brief 设置模型状态
 * @param bridge 桥接句柄
 * @param model_name 模型名称
 * @param state 新状态
 * @return 0成功，-1失败
 */
int gazebo_set_model_state(GazeboBridge* bridge, const char* model_name,
                           const GazeboModelState* state);

/**
 * @brief 施加力到模型
 * @param bridge 桥接句柄
 * @param model_name 模型名称
 * @param link_name 链接名称
 * @param force 力向量 [x,y,z]
 * @param torque 力矩向量 [x,y,z]
 * @return 0成功，-1失败
 */
int gazebo_apply_force(GazeboBridge* bridge, const char* model_name,
                       const char* link_name,
                       const float force[3], const float torque[3]);

/**
 * @brief 施加关节力矩/力
 * @param bridge 桥接句柄
 * @param model_name 模型名称
 * @param joint_name 关节名称
 * @param force 力标量（N）施加在关节运动方向
 * @return 0成功，-1失败
 */
int gazebo_apply_joint_force(GazeboBridge* bridge, const char* model_name,
                             const char* joint_name, float force);

/**
 * @brief 控制关节
 * @param bridge 桥接句柄
 * @param model_name 模型名称
 * @param joint_name 关节名称
 * @param position 目标位置
 * @param velocity 目标速度
 * @param effort 最大力矩
 * @return 0成功，-1失败
 */
int gazebo_set_joint(GazeboBridge* bridge, const char* model_name,
                     const char* joint_name,
                     float position, float velocity, float effort);

/**
 * @brief 步进仿真
 * @param bridge 桥接句柄
 * @param steps 步数
 * @return 0成功，-1失败
 */
int gazebo_step(GazeboBridge* bridge, int steps);

/**
 * @brief 暂停仿真
 * @param bridge 桥接句柄
 * @return 0成功，-1失败
 */
int gazebo_pause(GazeboBridge* bridge);

/**
 * @brief 恢复仿真
 * @param bridge 桥接句柄
 * @return 0成功，-1失败
 */
int gazebo_unpause(GazeboBridge* bridge);

/**
 * @brief 重置仿真
 * @param bridge 桥接句柄
 * @return 0成功，-1失败
 */
int gazebo_reset(GazeboBridge* bridge);

/**
 * @brief 获取仿真时间
 * @param bridge 桥接句柄
 * @param sim_time_sec 输出仿真时间（秒）
 * @return 0成功，-1失败
 */
int gazebo_get_sim_time(GazeboBridge* bridge, double* sim_time_sec);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_GAZEBO_BRIDGE_H */
