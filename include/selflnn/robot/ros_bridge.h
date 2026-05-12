/**
 * @file ros_bridge.h
 * @brief SELF-LNN 与 ROS/ROS2 机器人操作系统的桥接接口
 *
 * 根据需求更新（2026-05-11），ROS 允许使用 C++/Python 依赖。
 * 本桥接模块通过 rosbridge WebSocket 协议与 ROS Master 通信，
 * 提供话题发布/订阅、服务调用等 C 语言接口。
 *
 * 策略：优先使用 rosbridge (WebSocket JSON) 协议，无需编译 ROS 依赖。
 *       需要原生 ROS 性能时，通过 roscpp C++ 子进程通信。
 */

#ifndef SELFLNN_ROS_BRIDGE_H
#define SELFLNN_ROS_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief ROS 版本 */
typedef enum {
    ROS_VERSION_1 = 1,          /**< ROS 1 (Noetic/Melodic) */
    ROS_VERSION_2 = 2           /**< ROS 2 (Humble/Iron) */
} RosVersion;

/** @brief ROS 连接状态 */
typedef enum {
    ROS_STATE_DISCONNECTED = 0, /**< 未连接 */
    ROS_STATE_CONNECTING = 1,   /**< 连接中 */
    ROS_STATE_CONNECTED = 2,    /**< 已连接 */
    ROS_STATE_ERROR = 3         /**< 错误 */
} RosConnectionState;

/** @brief ROS 连接配置 */
typedef struct {
    RosVersion version;          /**< ROS 版本 */
    char* master_uri;            /**< ROS Master URI，如 "http://localhost:11311" */
    char* bridge_host;           /**< rosbridge WebSocket 主机 */
    int bridge_port;             /**< rosbridge WebSocket 端口，默认9090 */
    int use_rosbridge;           /**< 1=使用rosbridge，0=使用原生roscpp子进程 */
    float reconnect_timeout;     /**< 重连超时（秒） */
    int auto_reconnect;          /**< 是否自动重连 */
} RosBridgeConfig;

/** @brief 话题订阅回调函数类型 */
typedef void (*RosTopicCallback)(const char* topic, const char* message_type,
                                  const char* json_data, size_t data_len,
                                  void* user_data);

/** @brief ROS 桥接句柄 */
typedef struct RosBridge RosBridge;

/**
 * @brief 检测 ROS 环境是否可用
 * @return 1=可用，0=不可用
 */
int ros_bridge_is_available(void);

/**
 * @brief 检测 ROS 2 环境是否可用
 * @return 1=可用，0=不可用
 */
int ros2_bridge_is_available(void);

/**
 * @brief 创建 ROS 桥接连接
 * @param config 连接配置
 * @return 桥接句柄，失败返回NULL
 */
RosBridge* ros_bridge_create(const RosBridgeConfig* config);

/**
 * @brief 释放 ROS 桥接连接
 * @param bridge 桥接句柄
 */
void ros_bridge_free(RosBridge* bridge);

/**
 * @brief 获取连接状态
 * @param bridge 桥接句柄
 * @return 连接状态
 */
RosConnectionState ros_bridge_get_state(RosBridge* bridge);

/**
 * @brief 发布话题消息
 * @param bridge 桥接句柄
 * @param topic 话题名称
 * @param message_type 消息类型（如 "std_msgs/String"）
 * @param json_data JSON格式的消息数据
 * @return 0成功，-1失败
 */
int ros_bridge_publish(RosBridge* bridge, const char* topic,
                       const char* message_type, const char* json_data);

/**
 * @brief 订阅话题
 * @param bridge 桥接句柄
 * @param topic 话题名称
 * @param message_type 消息类型
 * @param callback 回调函数
 * @param user_data 用户数据
 * @return 0成功，-1失败
 */
int ros_bridge_subscribe(RosBridge* bridge, const char* topic,
                         const char* message_type,
                         RosTopicCallback callback, void* user_data);

/**
 * @brief 取消订阅话题
 * @param bridge 桥接句柄
 * @param topic 话题名称
 * @return 0成功，-1失败
 */
int ros_bridge_unsubscribe(RosBridge* bridge, const char* topic);

/**
 * @brief 调用 ROS 服务
 * @param bridge 桥接句柄
 * @param service 服务名称
 * @param request_json JSON格式的请求数据
 * @param response_buffer 响应缓冲区
 * @param buffer_size 缓冲区大小
 * @return 0成功，-1失败
 */
int ros_bridge_call_service(RosBridge* bridge, const char* service,
                            const char* request_json,
                            char* response_buffer, size_t buffer_size);

/**
 * @brief 获取 ROS 节点列表
 * @param bridge 桥接句柄
 * @param nodes_json 输出JSON节点列表
 * @param buffer_size 缓冲区大小
 * @return 0成功，-1失败
 */
int ros_bridge_get_nodes(RosBridge* bridge, char* nodes_json, size_t buffer_size);

/**
 * @brief 获取 ROS 话题列表
 * @param bridge 桥接句柄
 * @param topics_json 输出JSON话题列表
 * @param buffer_size 缓冲区大小
 * @return 0成功，-1失败
 */
int ros_bridge_get_topics(RosBridge* bridge, char* topics_json, size_t buffer_size);

/**
 * @brief 刷新连接（处理收到的消息，调用回调）
 * @param bridge 桥接句柄
 * @param timeout_ms 超时（毫秒），0表示非阻塞
 * @return 处理的消息数，-1失败
 */
int ros_bridge_spin_once(RosBridge* bridge, int timeout_ms);

/**
 * @brief 发送机器人控制命令（关节轨迹）
 * @param bridge 桥接句柄
 * @param joint_names 关节名称数组
 * @param positions 目标位置数组
 * @param velocities 目标速度数组
 * @param num_joints 关节数量
 * @param duration_sec 轨迹持续时间
 * @return 0成功，-1失败
 */
int ros_bridge_send_joint_trajectory(RosBridge* bridge,
                                     const char** joint_names,
                                     const float* positions,
                                     const float* velocities,
                                     int num_joints,
                                     float duration_sec);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_ROS_BRIDGE_H */
