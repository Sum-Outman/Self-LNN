/**
 * @file port_config.h
 * @brief 统一端口配置
 *
 * 集中管理系统所有服务端口，确保端口统一管理。
 * 所有模块必须引用此文件获取端口配置，禁止各模块独立定义端口号。
 *
 * ==================================================================
 * 端口分配表（所有端口定义见下方宏）
 * ==================================================================
 * | 宏名称                       | 端口号 | 用途/使用者                    |
 * |-----------------------------|--------|-------------------------------|
 * | SELFLNN_HTTP_PORT           | 8080   | HTTP API服务、前端通信         |
 * | SELFLNN_WEBSOCKET_PORT      | 8081   | WebSocket实时推送服务          |
 * | SELFLNN_GAZEBO_PORT         | 11345  | Gazebo仿真器TCP通信            |
 * | SELFLNN_PYBULLET_PORT       | 6667   | PyBullet仿真器端口             |
 * | SELFLNN_DISTRIBUTED_PORT    | 8765   | 分布式训练节点间通信            |
 * | SELFLNN_SIMULATOR_PORT      | 5555   | 内部仿真器监听端口             |
 * | SELFLNN_SENSOR_STREAM_PORT  | 5556   | 传感器实时数据流推送            |
 * | SELFLNN_RPC_PORT            | 18760  | 多系统RPC控制/负载均衡共用      |
 * | SELFLNN_DISCOVERY_PORT      | 18761  | 多系统节点自动发现服务          |
 * | SELFLNN_ROS_MASTER_PORT     | 11311  | ROS Master服务发现             |
 * | SELFLNN_ROS_XMLRPC_PORT     | 11312  | ROS节点XML-RPC通信            |
 * | SELFLNN_ROS_DATA_PORT       | 11313  | ROS TCPROS数据传输            |
 * | SELFLNN_GAZEBO_ROS_PORT     | 11345  | Gazebo ROS接口（与Gazebo共用） |
 * | SELFLNN_LOCALHOST           | -      | 默认本机地址 "127.0.0.1"      |
 * ==================================================================
 */

#ifndef SELFLNN_PORT_CONFIG_H
#define SELFLNN_PORT_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup port_config 端口配置
 *
 * 系统所有网络端口集中定义。
 * 修改端口时只需修改此文件，所有模块自动生效。
 * 端口号与config/system_config.json保持一致。
 *
 * @{
 */

/**
 * @brief HTTP API服务端口
 *
 * 后端HTTP API服务器端口，用于前端与SELF-LNN系统通信。
 * 对应config/system_config.json中http_port字段。
 */
#define SELFLNN_HTTP_PORT           8080

/**
 * @brief Gazebo仿真器TCP通信端口
 *
 * 连接外部Gazebo仿真器的TCP端口号。
 * Gazebo默认使用11345端口进行gazebo-ROS通信。
 */
#define SELFLNN_GAZEBO_PORT         11345

/**
 * @brief PyBullet仿真器端口
 *
 * PyBullet仿真器默认使用TCP 6667端口进行外部控制。
 * 当前版本中用作预留端口。
 */
#define SELFLNN_PYBULLET_PORT       6667

/**
 * @brief WebSocket服务端口
 *
 * WebSocket实时数据推送端口，用于前端实时接收系统状态更新。
 * 对应config/system_config.json中websocket_port字段。
 * 独立端口，不与HTTP端口(8080)共用。
 */
#define SELFLNN_WEBSOCKET_PORT      8081

/**
 * @brief 分布式训练通信端口
 *
 * 多节点分布式训练时节点间通信端口。
 */
#define SELFLNN_DISTRIBUTED_PORT    8765

/**
 * @brief 内部仿真器默认端口
 *
 * 内部简易仿真器（Simulator）监听端口。
 * 用于进程间通信或网络套接字连接。
 */
#define SELFLNN_SIMULATOR_PORT      5555

/**
 * @brief 多系统RPC通信默认端口
 *
 * 多系统控制器的节点间RPC控制命令传输端口。
 * 用于多机协同场景下的远程过程调用通信。
 * 来源：MultisystemControl内部通信需求。
 */
#define SELFLNN_RPC_PORT            18760

/**
 * @brief 多系统发现服务默认端口
 *
 * 多系统控制器的节点自动发现/广播服务端口。
 * 用于局域网内节点自动发现和注册。
 * 来源：MultisystemControl节点管理需求。
 */
#define SELFLNN_DISCOVERY_PORT      18761

/**
 * @brief ROS Master默认端口
 *
 * ROS Master服务默认端口，用于ROS节点发现和连接管理。
 */
#define SELFLNN_ROS_MASTER_PORT     11311

/**
 * @brief ROS XML-RPC协议端口
 *
 * ROS节点的XML-RPC通信端口，用于节点间的服务调用和参数管理。
 */
#define SELFLNN_ROS_XMLRPC_PORT     11312

/**
 * @brief ROS数据传输端口（F-033修复）
 *
 * ROS发布者与订阅者之间的TCPROS数据传输端口，
 * 与Master的XMLRPC端口(11311)分离，用于实际的
 * 话题消息传输。默认从11313开始。
 */
#define SELFLNN_ROS_DATA_PORT       11313

/**
 * @brief ROS Gazebo接口端口
 *
 * Gazebo的ROS接口通信端口，用于Gazebo与ROS系统之间的数据交换。
 */
#define SELFLNN_GAZEBO_ROS_PORT     SELFLNN_GAZEBO_PORT  /* 与Gazebo共用同一端口 */

/**
 * @brief 传感器数据流端口
 *
 * 多传感器实时数据推送端口，用于高速传感器数据传输。
 */
#define SELFLNN_SENSOR_STREAM_PORT  5556

/**
 * @brief 默认本机IP地址
 *
 * 所有需要连接本机的服务统一使用此宏，替代硬编码"127.0.0.1"。
 * 修改本机地址时只需修改此定义，所有模块自动生效。
 */
#define SELFLNN_LOCALHOST          "127.0.0.1"

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_PORT_CONFIG_H */
