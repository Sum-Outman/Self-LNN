# AGI机器人指南 / AGI Robot Guide

---

## 目录 / Table of Contents
- [简介 / Introduction](#简介--introduction)
- [机器人类型 / Robot Types](#机器人类型--robot-types)
- [系统架构 / System Architecture](#系统架构--system-architecture)
- [传感器系统 / Sensor System](#传感器系统--sensor-system)
- [运动控制 / Motion Control](#运动控制--motion-control)
- [路径规划 / Path Planning](#路径规划--path-planning)
- [视觉感知 / Visual Perception](#视觉感知--visual-perception)
- [多机器人协调 / Multi-Robot Coordination](#多机器人协调--multi-robot-coordination)
- [硬件接口 / Hardware Interface](#硬件接口--hardware-interface)
- [仿真环境 / Simulation Environment](#仿真环境--simulation-environment)
- [部署与维护 / Deployment and Maintenance](#部署与维护--deployment-and-maintenance)

---

## 简介 / Introduction

### 中文
SELF-LNN提供完整的AGI机器人控制框架，支持多种机器人类型、传感器集成、路径规划、视觉感知、多机器人协调、硬件接口和仿真环境。系统支持CPU计算和各种品牌GPU计算（NVIDIA/AMD/Apple GPU完整支持；华为昇腾/寒武纪/Google TPU需安装对应厂商SDK，无SDK时自动回退CPU）。

### English
SELF-LNN provides a complete AGI robot control framework, supporting multiple robot types, sensor integration, path planning, visual perception, multi-robot coordination, hardware interfaces and simulation environments. The system supports CPU and various GPU brands (NVIDIA/AMD/Apple GPUs fully supported; Huawei Ascend/Cambricon/Google TPU require vendor SDKs, auto-fallback to CPU without SDK).

---

## 机器人类型 / Robot Types

### 中文
| 机器人类型 / Robot Type | 描述 / Description | 主要特性 / Key Features |
|----------------------|-------------------|------------------|
| 移动机器人 / Mobile Robot | 轮式/履带式移动机器人 | 自主导航、避障、SLAM |
| 机械臂 / Manipulator | 多自由度机械臂 | 抓取、操作、力控 |
| 人形机器人 / Humanoid Robot | 双足人形机器人 | 步态生成、平衡控制、全身协调 |
| 无人机 / UAV | 多旋翼无人机 | 飞行控制、航拍、导航 |
| 水下机器人 / AUV | 水下自主航行器 | 水下导航、声呐、防水设计 |
| 自定义机器人 / Custom Robot | 用户自定义机器人 | 灵活配置、可扩展 |

### English
| Robot Type | Description | Key Features |
|------------|-------------|--------------|
| Mobile Robot | Wheeled/tracked robot | Autonomous navigation, obstacle avoidance, SLAM |
| Manipulator | Multi-DOF robotic arm | Grasping, manipulation, force control |
| Humanoid Robot | Bipedal humanoid robot | Gait generation, balance control, whole-body coordination |
| UAV | Multi-rotor drone | Flight control, aerial photography, navigation |
| AUV | Underwater autonomous vehicle | Underwater navigation, sonar, waterproof design |
| Custom Robot | User-defined robot | Flexible configuration, extensible |

---

## 系统架构 / System Architecture

### 中文
机器人控制系统采用统一的液态神经网络架构，所有机器人共享同一个CfC核心，统一处理多模态输入。

### English
Robot control system uses unified liquid neural network architecture, all robots share the same CfC core for unified multi-modal input processing.

---

## 传感器系统 / Sensor System

### 中文
| 传感器类型 / Sensor Type | 描述 / Description | 用途 / Use Case |
|----------------------|-------------------|--------------|
| 摄像头 / Camera | RGB相机 | 视觉感知、物体识别 |
| 激光雷达 / LiDAR | 激光测距雷达 | 3D感知、SLAM、避障 |
| 惯性测量单元 / IMU | 加速度计+陀螺仪 | 姿态估计、运动控制 |
| 力传感器 / Force Sensor | 力/力矩传感器 | 力控制、触觉反馈 |
| 触觉传感器 / Tactile Sensor | 触觉阵列 | 物体识别、抓取控制 |
| 距离传感器 / Distance Sensor | 超声波/红外 | 近距离检测、避障 |
| 温度传感器 / Temperature Sensor | 温度测量 | 环境监测、安全保护 |
| 位置传感器 / Position Sensor | 编码器/电位器 | 关节位置反馈 |

### English
| Sensor Type | Description | Use Case |
|-------------|-------------|----------|
| Camera | RGB camera | Visual perception, object recognition |
| LiDAR | Laser ranging radar | 3D perception, SLAM, obstacle avoidance |
| IMU | Accelerometer + Gyroscope | Pose estimation, motion control |
| Force Sensor | Force/Torque sensor | Force control, haptic feedback |
| Tactile Sensor | Tactile array | Object recognition, grasp control |
| Distance Sensor | Ultrasonic/Infrared | Proximity detection, obstacle avoidance |
| Temperature Sensor | Temperature measurement | Environmental monitoring, safety protection |
| Position Sensor | Encoder/Potentiometer | Joint position feedback |

---

## 运动控制 / Motion Control

### 中文
运动控制系统支持多种运动控制模式，提供运动学求解、轨迹规划、安全监测等功能。

### English
Motion control system supports multiple motion control modes, providing kinematics solving, trajectory planning, safety monitoring and other functions.

---

## 路径规划 / Path Planning

### 中文
| 规划算法 / Planning Algorithm | 中文名称 / Chinese Name | 描述 / Description |
|---------------------|------------------------|-------------------|
| A* | A*算法 | 网格路径规划 |
| RRT* | RRT*快速随机树 | 高维空间路径规划 |
| Dijkstra | Dijkstra算法 | 最短路径搜索 |
| 人工势场法 / Artificial Potential Field | 人工势场法 | 实时避障 |

### English
| Planning Algorithm | Name | Description |
|--------------------|------|-------------|
| A* | A* Algorithm | Grid-based path planning |
| RRT* | RRT* Rapidly-exploring Random Tree | High-dimensional space path planning |
| Dijkstra | Dijkstra's Algorithm | Shortest path search |
| Artificial Potential Field | Artificial Potential Field | Real-time obstacle avoidance |

---

## 视觉感知 / Visual Perception

### 中文
视觉感知系统提供图像采集、图像处理、物体识别、深度估计、双目视觉、SLAM等功能。

### English
Visual perception system provides image capture, image processing, object recognition, depth estimation, stereo vision, SLAM and other functions.

---

## 多机器人协调 / Multi-Robot Coordination

### 中文
支持多机器人协调控制系统，支持多种协调算法，支持多机器人协同任务分配、编队控制等功能。

### English
Supports multi-robot coordinated control system, multiple coordination algorithms, multi-robot collaborative task allocation, formation control and other functions.

---

## 硬件接口 / Hardware Interface

### 中文
| 接口类型 / Interface Type | 描述 / Description | 用途 / Use Case |
|----------------------|-------------------|--------------|
| 串口 / Serial Port | UART/RS232/RS485 | 硬件通信 |
| CAN总线 / CAN Bus | CAN 2.0A/B | 实时控制、传感器数据 |
| ROS 1/ROS 2 | Robot Operating System | 标准机器人接口 |
| 以太网 / Ethernet | TCP/IP网络通信 | 远程控制、数据传输 |
| USB | USB通信 | 传感器、执行器 |
| 网络 / GPIO | 通用IO | 简单控制、传感器读取 |

### English
| Interface Type | Description | Use Case |
|----------------|-------------|----------|
| Serial Port | UART/RS232/RS485 | Hardware communication |
| CAN Bus | CAN 2.0A/B | Real-time control, sensor data |
| ROS 1/ROS 2 | Robot Operating System | Standard robot interface |
| Ethernet | TCP/IP network communication | Remote control, data transmission |
| USB | USB communication | Sensors, actuators |
| GPIO | General-purpose IO | Simple control, sensor reading |

---

## 仿真环境 / Simulation Environment

### 中文
| 仿真引擎 / Simulation Engine | 中文名称 / Chinese Name | 描述 / Description |
|------------------------|-----------------------|------------------|
| 内置仿真器 / Internal Simulator | 内置仿真器 | 简单仿真、快速测试 |
| PyBullet | PyBullet仿真引擎 | 物理仿真、关节控制 |
| Gazebo | Gazebo仿真器 | 高保真仿真、插件系统 |

### English
| Simulation Engine | Name | Description |
|------------------|------|-------------|
| Internal Simulator | Internal Simulator | Simple simulation, fast testing |
| PyBullet | PyBullet Simulation Engine | Physics simulation, joint control |
| Gazebo | Gazebo Simulator | High-fidelity simulation, plugin system |

---

## 部署与维护 / Deployment and Maintenance

### 部署建议 / Deployment Recommendations

#### 中文
1. **硬件要求**
   - CPU: 4核心以上，推荐8核心
   - 内存: 4GB以上，推荐8GB
   - 存储: 10GB以上可用空间
   - GPU: 可选，支持NVIDIA/AMD/Intel等

2. **软件环境**
   - Windows 10+ / Ubuntu 18.04+ / macOS 10.14+
   - CMake 3.10+
   - GCC 7+ / Clang 6+ / MSVC 2017+

3. **安全建议**
   - 使用API密钥认证
   - 配置防火墙规则
   - 定期备份配置和数据
   - 启用安全监控

#### English
1. **Hardware Requirements**
   - CPU: 4+ cores, 8 cores recommended
   - Memory: 4GB+, 8GB recommended
   - Storage: 10GB+ available space
   - GPU: Optional, supports NVIDIA/AMD/Intel etc.

2. **Software Environment**
   - Windows 10+ / Ubuntu 18.04+ / macOS 10.14+
   - CMake 3.10+
   - GCC 7+ / Clang 6+ / MSVC 2017+

3. **Security Recommendations**
   - Use API key authentication
   - Configure firewall rules
   - Regularly backup configurations and data
   - Enable security monitoring

### 常见问题 / FAQ

#### 中文
**Q: 机器人连接失败怎么办？**
A: 请检查:
   1. 硬件接口配置是否正确
   2. 串口/网络连接是否正常
   3. 机器人是否已上电
   4. 查看系统日志获取详细错误信息

**Q: 仿真模式和硬件模式如何切换？**
A: 使用相应的API函数进行切换。

**Q: 如何配置多机器人？**
A: 为每个机器人设置不同的机器人ID，然后使用协调控制器进行协调控制。

#### English
**Q: What if robot connection fails?**
A: Please check:
   1. Hardware interface configuration is correct
   2. Serial/network connection is normal
   3. Robot is powered on
   4. Check system logs for detailed error information

**Q: How to switch between simulation and hardware mode?**
A: Use corresponding API functions for switching.

**Q: How to configure multiple robots?**
A: Set different robot ID for each robot, then use coordination controller for coordinated control.

---

> **文档更新 / Updated**: 2026-05-14
> **相关文档 / Related Docs**: [架构图](./Architecture_Diagram.md) | [用户手册](./USER_GUIDE.md) | [训练指南](./Training_Guide.md)
