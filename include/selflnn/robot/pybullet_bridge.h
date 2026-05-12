/**
 * @file pybullet_bridge.h
 * @brief SELF-LNN 与 PyBullet 物理仿真引擎的桥接接口
 *
 * PyBullet 是 Python 库（Bullet Physics 的 Python 绑定），
 * 根据需求更新（2026-05-11），PyBullet 允许使用 Python 依赖。
 *
 * 本桥接模块通过 Python C API 启动 PyBullet 仿真环境，
 * 提供人形机器人仿真训练的 C 语言接口。
 *
 * 当 PyBullet/Python 环境不可用时，回退到 simulator.c 内部仿真。
 */

#ifndef SELFLNN_PYBULLET_BRIDGE_H
#define SELFLNN_PYBULLET_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief PyBullet 连接状态 */
typedef enum {
    PYBULLET_DISCONNECTED = 0,  /**< 未连接 */
    PYBULLET_CONNECTING = 1,    /**< 连接中 */
    PYBULLET_CONNECTED = 2,     /**< 已连接 */
    PYBULLET_ERROR = 3          /**< 错误状态 */
} PyBulletConnectionState;

/** @brief PyBullet 仿真配置 */
typedef struct {
    int use_gui;                 /**< 是否启用 GUI（1=GUI模式，0=DIRECT模式） */
    float time_step;             /**< 仿真时间步长（秒），默认1/240 */
    int max_steps_per_call;      /**< 每次调用最大步数 */
    char* urdf_path;             /**< 机器人URDF模型路径 */
    float gravity[3];            /**< 重力向量，默认 {0, 0, -9.81} */
    int num_solver_iterations;   /**< 约束求解器迭代次数 */
    int enable_self_collision;   /**< 是否启用自碰撞检测 */
} PyBulletConfig;

/** @brief 关节状态 */
typedef struct {
    float position;              /**< 关节位置（弧度） */
    float velocity;              /**< 关节速度（弧度/秒） */
    float torque;                /**< 关节力矩 */
    float target_position;       /**< 目标位置 */
    float target_velocity;       /**< 目标速度 */
} PyBulletJointState;

/** @brief 机器人基座状态 */
typedef struct {
    float position[3];           /**< 位置 x,y,z */
    float orientation[4];        /**< 四元数方向 x,y,z,w */
    float linear_velocity[3];    /**< 线速度 */
    float angular_velocity[3];   /**< 角速度 */
} PyBulletBaseState;

/** @brief 接触点信息 */
typedef struct {
    int body_a;                  /**< 接触体A的ID */
    int body_b;                  /**< 接触体B的ID */
    float position[3];           /**< 接触点世界坐标 */
    float normal[3];             /**< 接触面法向量 */
    float force;                 /**< 接触力（N） */
    float distance;              /**< 穿透深度 */
} PyBulletContactPoint;

/** @brief 摄像头图像数据 */
typedef struct {
    int width;                   /**< 图像宽度 */
    int height;                  /**< 图像高度 */
    unsigned char* rgb_data;     /**< RGB像素数据 */
    float* depth_data;           /**< 深度数据（可选，可为NULL） */
    float* segmentation_data;    /**< 语义分割数据（可选，可为NULL） */
} PyBulletCameraImage;

/**
 * @brief 检测 PyBullet 环境是否可用
 * @return 1=可用，0=不可用
 */
int pybullet_is_available(void);

/**
 * @brief 创建 PyBullet 仿真连接
 * @param config 仿真配置
 * @return 连接句柄（正整数），失败返回-1
 */
int pybullet_connect(const PyBulletConfig* config);

/**
 * @brief 断开 PyBullet 仿真连接
 * @param connection_id 连接句柄
 * @return 0成功，-1失败
 */
int pybullet_disconnect(int connection_id);

/**
 * @brief 获取连接状态
 * @param connection_id 连接句柄
 * @return 连接状态
 */
PyBulletConnectionState pybullet_get_state(int connection_id);

/**
 * @brief 加载 URDF 机器人模型
 * @param connection_id 连接句柄
 * @param urdf_path URDF文件路径
 * @param base_position 基座初始位置 [x,y,z]
 * @param base_orientation 基座初始方向（四元数）[x,y,z,w]
 * @param use_fixed_base 是否固定基座
 * @return 机器人ID（正整数），失败返回-1
 */
int pybullet_load_urdf(int connection_id, const char* urdf_path,
                       const float base_position[3],
                       const float base_orientation[4],
                       int use_fixed_base);

/**
 * @brief 执行仿真步进
 * @param connection_id 连接句柄
 * @return 0成功，-1失败
 */
int pybullet_step_simulation(int connection_id);

/**
 * @brief 获取关节数量
 * @param connection_id 连接句柄
 * @param robot_id 机器人ID
 * @return 关节数量，失败返回-1
 */
int pybullet_get_num_joints(int connection_id, int robot_id);

/**
 * @brief 获取关节状态
 * @param connection_id 连接句柄
 * @param robot_id 机器人ID
 * @param joint_index 关节索引
 * @param state 输出关节状态
 * @return 0成功，-1失败
 */
int pybullet_get_joint_state(int connection_id, int robot_id,
                             int joint_index, PyBulletJointState* state);

/**
 * @brief 设置关节控制
 * @param connection_id 连接句柄
 * @param robot_id 机器人ID
 * @param joint_indices 关节索引数组
 * @param target_positions 目标位置数组
 * @param num_joints 关节数量
 * @return 0成功，-1失败
 */
int pybullet_set_joint_control(int connection_id, int robot_id,
                               const int* joint_indices,
                               const float* target_positions,
                               int num_joints);

/**
 * @brief 获取基座状态
 * @param connection_id 连接句柄
 * @param robot_id 机器人ID
 * @param state 输出基座状态
 * @return 0成功，-1失败
 */
int pybullet_get_base_state(int connection_id, int robot_id,
                            PyBulletBaseState* state);

/**
 * @brief 获取接触点信息
 * @param connection_id 连接句柄
 * @param robot_id 机器人ID
 * @param contacts 输出接触点数组
 * @param max_contacts 最大接触点数
 * @return 实际接触点数，失败返回-1
 */
int pybullet_get_contacts(int connection_id, int robot_id,
                          PyBulletContactPoint* contacts, int max_contacts);

/**
 * @brief 获取仿真摄像头图像
 * @param connection_id 连接句柄
 * @param width 图像宽度
 * @param height 图像高度
 * @param camera_position 摄像头位置 [x,y,z]
 * @param camera_target 摄像头目标点 [x,y,z]
 * @param camera_up 摄像头上方向 [x,y,z]
 * @param image 输出图像数据（调用者负责释放 rgb_data/depth_data/segmentation_data）
 * @return 0成功，-1失败
 */
int pybullet_get_camera_image(int connection_id, int width, int height,
                              const float camera_position[3],
                              const float camera_target[3],
                              const float camera_up[3],
                              PyBulletCameraImage* image);

/**
 * @brief 释放摄像头图像数据
 * @param image 图像数据
 */
void pybullet_free_camera_image(PyBulletCameraImage* image);

/**
 * @brief 重置仿真
 * @param connection_id 连接句柄
 * @return 0成功，-1失败
 */
int pybullet_reset_simulation(int connection_id);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_PYBULLET_BRIDGE_H */
