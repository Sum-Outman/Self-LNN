# SELF-LNN 全液态神经网络 AGI 系统
# SELF-LNN Full Liquid Neural Network AGI System

> **版本 / Version:** 1.4.0 | **语言 / Language:** 100% Pure C (C11) | **许可证 / License:** Apache 2.0
> **构建 / Build:** CMake 3.10+ | **平台 / Platform:** Windows / Linux / macOS
> **开源仓库 / Repository:** https://github.com/Sum-Outman/Self-LNN
> **开发者邮箱 / Developer Email:** silencecrowtom@qq.com

---

## 目录 / Table of Contents
- [项目简介 / Project Overview](#项目简介--project-overview)
- [核心特性 / Core Features](#核心特性--core-features)
- [系统架构 / System Architecture](#系统架构--system-architecture)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [部署指南 / Deployment Guide](#部署指南--deployment-guide)
- [功能说明 / Feature Description](#功能说明--feature-description)

- [AGI 机器人功能 / AGI Robot Features](#agi-机器人功能--agi-robot-features)
- [维护指南 / Maintenance Guide](#维护指南--maintenance-guide)
- [开发指南 / Development Guide](#开发指南--development-guide)
- [常见问题 / FAQ](#常见问题--faq)
- [文档索引 / Document Index](#文档索引--document-index)

---

## 项目简介 / Project Overview

### 中文
SELF-LNN 是一个 100% 纯 C 语言实现的全模态通用人工智能（AGI）实验系统。该系统采用单一 CfC (Closed-form Continuous-time) 液态神经网络作为核心模型，无需使用多模型融合、跨模态注意力机制或独立的编码器，直接将视觉、音频、文本、传感器等多种模态数据通过线性投影注入同一连续动态系统进行处理。

系统具有自我认知、推理、学习、演化、记忆、机器人控制等多种能力，并支持多种 GPU 后端加速。所有功能均采用纯 C 语言实现，不依赖任何第三方 C 库，保证了系统的稳定性和可移植性。

> **项目定位说明**：本项目目前没有图像和视频生成能力，后期开发根据实际应用可能会加入。项目以感知真实世界和工程智能化控制拓展为主。

### English
SELF-LNN is a full-modal Artificial General Intelligence (AGI) experimental system implemented in 100% pure C. The system uses a single CfC (Closed-form Continuous-time) Liquid Neural Network as the core model, eliminating the need for multi-model fusion, cross-modal attention mechanisms, or separate encoders. It directly injects multi-modal data including vision, audio, text, and sensors into the same continuous dynamic system via linear projection.

The system features self-cognition, reasoning, learning, evolution, memory, robot control, and supports multiple GPU backend acceleration. All functions are implemented in pure C without any third-party C library dependencies, ensuring system stability and portability.

> **Project Positioning Note**: This project currently does not have image and video generation capabilities, which may be added in later development based on actual application needs. The project focuses on real-world perception and engineering intelligent control expansion.

---

## 核心特性 / Core Features

### 中文
| 特性类别 | 具体功能 |
|---------|---------|
| **核心模型** | 单一 CfC 液态神经网络，7 种 ODE 求解器（闭式解、RK4、RK45、DP54、Rosenbrock、Symplectic、CTBP） |
| **多模态处理** | 支持 9 种模态统一输入：视觉、音频、文本、传感器、触觉、本体感、热感、雷达、电机 |
| **架构设计** | Token-Free 连续信号架构，无离散 token，无需注意力机制，零外部 C 库依赖 |
| **认知能力** | 自我认知、元认知、迭代式深度反思、多维自我修正 |
| **推理能力** | 因果推理、数学推理、分层规划、长期规划、不确定性推理 |
| **学习能力** | 在线学习、强化学习（PPO/SAC）、模仿学习（BC/DAgger/IRL）、元学习（MAML/Reptile） |
| **训练系统** | 6 阶段训练流水线、混合精度训练、分布式训练、模型版本管理 |
| **演化系统** | CMA-ES 演化策略、帕累托优化、NAS 神经架构搜索 |
| **记忆系统** | 5 层记忆体系（短期、长期、情景、语义、工作记忆）、Ebbinghaus 遗忘曲线、Hebbian 巩固 |
| **知识库** | 知识图谱、三元组存储、多跳推理、本体工程、语义网络 |
| **机器人控制** | DH 运动学、A*/RRT* 路径规划、ROS/ROS2 集成、PyBullet/Gazebo 仿真、多机器人协调 |
| **GPU 加速** | 10 种 GPU 后端接口（CUDA、OpenCL、Vulkan、Metal、ROCm、CPU-SIMD — 完整内核调度+线程池并行；Intel、Ascend、Cambricon、TPU — 通过dlsym动态加载SDK + CPU自动回退；CPU 后端支持 27+ 种真实内核算子） |
| **后端服务** | HTTP REST API（278 端点，254 已实现 + 24 预留）、WebSocket 实时通信、API 密钥认证、安全监控 |
| **并发系统** | 线程池、无锁队列、读写锁、RCU 机制 |
| **安全系统** | 紧急停止、熔断器、审计日志、内容过滤 |
| **前端界面** | 15 个 HTML 页面、17 个 JS 模块，完整的可视化控制台 |

### English
| Feature Category | Specific Functions |
|-----------------|-------------------|
| **Core Model** | Single CfC Liquid Neural Network, 7 ODE solvers (Closed-Form, RK4, RK45, DP54, Rosenbrock, Symplectic, CTBP) |
| **Multimodal Processing** | Unified input for 9 modalities: Vision, Audio, Text, Sensor, Haptic, Proprioception, Thermal, Radar, Motor |
| **Architecture Design** | Token-Free continuous signal architecture, no discrete tokens, no attention mechanisms, zero external C library dependencies |
| **Cognitive Abilities** | Self-cognition, metacognition, iterative deep reflection, multi-dimensional self-correction |
| **Reasoning Abilities** | Causal reasoning, mathematical reasoning, hierarchical planning, long-term planning, uncertainty reasoning |
| **Learning Abilities** | Online learning, reinforcement learning (PPO/SAC), imitation learning (BC/DAgger/IRL), meta-learning (MAML/Reptile) |
| **Training System** | 6-stage training pipeline, mixed precision training, distributed training, model version management |
| **Evolution System** | CMA-ES evolution strategy, Pareto optimization, NAS neural architecture search |
| **Memory System** | 5-layer memory hierarchy (short-term, long-term, episodic, semantic, working memory), Ebbinghaus forgetting curve, Hebbian consolidation |
| **Knowledge Base** | Knowledge graph, triple store, multi-hop reasoning, ontology engineering, semantic network |
| **Robot Control** | DH kinematics, A*/RRT* path planning, ROS/ROS2 integration, PyBullet/Gazebo simulation, multi-robot coordination |
| **GPU Acceleration** | 10 GPU backend interfaces (CUDA, OpenCL, Vulkan, Metal, ROCm, CPU-SIMD — full kernel dispatch + thread pool parallelism; Intel, Ascend, Cambricon, TPU — dlsym-loaded SDK + CPU auto-fallback; CPU backend supports 27+ real kernel operators) |
| **Backend Service** | HTTP REST API (278 endpoints, 254 implemented + 24 reserved), WebSocket real-time communication, API key authentication, security monitoring |
| **Concurrency System** | Thread pool, lock-free queue, read-write lock, RCU mechanism |
| **Safety System** | Emergency stop, circuit breaker, audit log, content filtering |
| **Frontend Interface** | 15 HTML pages, 17 JS modules, complete visual console |

---

## 系统架构 / System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              SELF-LNN AGI 系统总体架构 / System Architecture                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  层1 / Layer 1: 前端交互层 / Frontend Layer — 15 HTML + 17 JS        │  │
│  │  仪表盘/Dashboard | LNN控制台/Console | 训练中心/Training             │  │
│  │  机器人控制/Robot Ctrl | 仿真控制/Simulation | 多模态学习/Multimodal  │  │
│  │  语音控制/Voice Ctrl | 对话界面/Dialogue | 安全面板/Safety Panel      │  │
│  │  知识图谱/Knowledge Graph | API文档/API Docs | 使用记录/Usage Logs   │  │
│  │  通信/Comm: WebSocket (实时推送/Real-time) + HTTP REST, 端口/Port 8080│  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │  ↕ HTTP/WebSocket                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  层2 / Layer 2: 后端服务层 / Backend Layer — main.c + backend.c      │  │
│  │  HTTP路由分发/Routing | WebSocket实时通信 | API密钥认证/Auth          │  │
│  │  熔断器/CircuitBreaker | 日志系统/Logging | 线程池调度/ThreadPool     │  │
│  │  配置管理/Config | 信号处理/Signal | 安全头注入/SecurityHeaders      │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  层3 / Layer 3: 统一多模态输入层 / Unified Multimodal Input          │  │
│  │                                                                      │  │
│  │  视觉/Vision[1024] ───→ W·x+b ──┐                                    │  │
│  │  音频/Audio[256]   ───→ W·x+b ──┤                                    │  │
│  │  文本/Text[512]    ───→ W·x+b ──┤                                    │  │
│  │  (Unicode码点/Codepoint→哈希投影/Hash→float[]) ──┤                    │  │
│  │  传感器/Sensor[128]───→ W·x+b ──┤                                    │  │
│  │  触觉/Haptic[128]  ───→ W·x+b ──┤                                    │  │
│  │  本体感/Proprio[128]──→ W·x+b ──┤                                    │  │
│  │  热感/Thermal[64]  ───→ W·x+b ──┤                                    │  │
│  │  雷达/Radar[128]   ───→ W·x+b ──┤                                    │  │
│  │  电机/Motor[64]    ───→ W·x+b ──┼──→ combined_input[256]            │  │
│  │      线性投影/Linear Projection → 直接求和/Element-wise Sum           │  │
│  │                                  ↓                                    │  │
│  │            单一 CfCCell ODE 连续动态系统 / Continuous Dynamic System │  │
│  │            τ·dh/dt = -h + σ⊙tanh(Wx+Uh+b)                             │  │
│  │            封闭形式解/Closed-Form:                                     │  │
│  │            h(t+Δt)=h(t)·e^(-Δt/τ)+(1-e^(-Δt/τ))·driver                │  │
│  │                                  ↓                                    │  │
│  │            统一输出/Unified Output → 共享主LNN/Shared Main LNN       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  层4 / Layer 4: 核心引擎层 / Core Engine — 唯一共享LNN/Single Shared │  │
│  │                                                                      │  │
│  │  ╔════════════════════════════════════════════════════════════════╗  │  │
│  │  ║  单一CfC液态神经网络 / Single CfC LNN — 128→256→128           ║  │  │
│  │  ║  全局1个实例/1 Global Instance | 20+子系统注册表共享           ║  │  │
│  │  ╚══════════════════════════════════════════════════════════════╝  │  │
│  │                                                                      │  │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐     │  │
│  │  │自我认知  │ 知识库  │ 推理引擎 │ 记忆系统 │ 学习系统 │ 训练系统 │     │  │
│  │  │Cognit.  │Knowledg.│Reasoning│ Memory  │Learning │Training │     │  │
│  │  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤     │  │
│  │  │机器人   │GPU加速  │安全系统 │演化引擎 │AGI核心  │分布式   │     │  │
│  │  │ Robot   │GPU Accel│ Safety  │Evolution│AGI Core │Distrib. │     │  │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 中文
系统分为四层架构：
1. **前端交互层**：15个HTML页面和17个JS模块组成，提供完整的Web控制台界面
2. **后端服务层**：提供HTTP REST API和WebSocket通信，处理路由、认证、安全等功能
3. **统一多模态输入层**：将9种模态数据通过线性投影求和后注入单一CfC动态系统
4. **核心引擎层**：包含唯一的共享LNN实例，所有子系统共享该模型进行处理

### English
The system is divided into four architectural layers:
1. **Frontend Interaction Layer**: Composed of 15 HTML pages and 17 JS modules, providing a complete Web console interface
2. **Backend Service Layer**: Provides HTTP REST API and WebSocket communication, handling routing, authentication, security, and other functions
3. **Unified Multimodal Input Layer**: Injects 9 modalities of data into a single CfC dynamic system after linear projection and summation
4. **Core Engine Layer**: Contains the only shared LNN instance, with all subsystems sharing this model for processing

---

## 快速开始 / Quick Start

### 系统要求 / System Requirements

#### 中文
- **操作系统**：Windows 10+, Linux (Ubuntu 18.04+), macOS 10.14+
- **编译器**：GCC 7+, Clang 6+, MSVC 2017+ (含 VS 2022/2026)
- **构建工具**：CMake 3.10+
- **内存**：最低 2GB，推荐 8GB+
- **磁盘空间**：最低 500MB
- **可选**：NVIDIA GPU (CUDA), AMD GPU (ROCm), Intel GPU, 其他支持的 GPU 设备
- **GPU/NPU专属SDK**（仅自定义内核编译时需要，模型推理无需）：
  - Intel GPU 自定义内核编译：Intel Level Zero SDK
  - 华为昇腾(Ascend)自定义内核编译：AscendCL (CANN)
  - 寒武纪(Cambricon)自定义内核编译：CNRT (Cambricon Neuware)
  - Google TPU 自定义内核编译：libtpu
  - **注意**：上述SDK仅在需要编译GPU原生内核算子时使用。模型推理和通用计算均通过dlsym动态加载SDK自动适配，无SDK时自动使用CPU后端（支持27+种真实内核算子+线程池+SIMD并行）。CUDA/OpenCL/Vulkan/Metal/ROCm无需额外SDK。

#### English
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.14+
- **Compiler**: GCC 7+, Clang 6+, MSVC 2017+ (including VS 2022/2026)
- **Build Tools**: CMake 3.10+
- **Memory**: Minimum 2GB, recommended 8GB+
- **Disk Space**: Minimum 500MB
- **Optional**: NVIDIA GPU (CUDA), AMD GPU (ROCm), Intel GPU, other supported GPU devices
- **GPU/NPU Vendor SDKs** (only needed for custom kernel compilation; model inference works without):
  - Intel GPU custom kernel compilation: Intel Level Zero SDK
  - Huawei Ascend custom kernel compilation: AscendCL (CANN)
  - Cambricon custom kernel compilation: CNRT (Cambricon Neuware)
  - Google TPU custom kernel compilation: libtpu
  - **Note**: These SDKs are only required for compiling native GPU kernel operators. Model inference and general computation auto-adapt via dlsym-loaded SDK, falling back to CPU backend (27+ real kernel operators + thread pool + SIMD parallel) when SDK is absent. CUDA/OpenCL/Vulkan/Metal/ROCm require no additional SDK.

### 编译 / Build

#### 中文
```bash
# Windows (MSVC, VS 2017/2019/2022/2026)
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
# 或 Visual Studio 2026: cmake .. -G "Visual Studio 18 2026" -A x64
cmake --build . --config Release

# Linux (GCC/Clang)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### English
```bash
# Windows (MSVC, VS 2017/2019/2022/2026)
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
# Or Visual Studio 2026: cmake .. -G "Visual Studio 18 2026" -A x64
cmake --build . --config Release

# Linux (GCC/Clang)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 运行 / Run

#### 中文
```bash
# 启动 HTTP 服务器（默认端口 8080）
# Start HTTP Server (default port 8080)
./build/bin/Release/selflnn.exe

# 或指定端口
# Or specify port
./build/bin/Release/selflnn.exe --port 8080

# 运行示例
# Run examples
./build/examples/Release/hello_selflnn.exe
./build/examples/Release/robot_example.exe
./build/examples/Release/knowledge_example.exe

# 运行测试
# Run tests
ctest --test-dir build -C Release
```

#### English
```bash
# Start HTTP Server (default port 8080)
./build/bin/Release/selflnn.exe

# Or specify port
./build/bin/Release/selflnn.exe --port 8080

# Run examples
./build/examples/Release/hello_selflnn.exe
./build/examples/Release/robot_example.exe
./build/examples/Release/knowledge_example.exe

# Run tests
ctest --test-dir build -C Release
```

### 配置 / Configuration

#### 中文
配置文件 `selflnn_config.json`：

```json
{
  "state_dimension": 256,
  "memory_capacity": 10000,
  "ode_solver_type": 0,
  "backend_port": 8080,
  "enable_gpu": true,
  "strict_real_data_mode": true,
  "max_concurrent_tasks": 100,
  "power_mode": "balanced"
}
```

#### English
Configuration file `selflnn_config.json`:

```json
{
  "state_dimension": 256,
  "memory_capacity": 10000,
  "ode_solver_type": 0,
  "backend_port": 8080,
  "enable_gpu": true,
  "strict_real_data_mode": true,
  "max_concurrent_tasks": 100,
  "power_mode": "balanced"
}
```

### 前端访问 / Frontend Access

#### 中文
启动服务器后，在浏览器中访问：
- 主页/仪表盘：http://localhost:8080/index.html (所有功能集成于单一页面)
- API 端点：http://localhost:8080/api/status 等

#### English
After starting the server, visit in your browser:
- Homepage/Dashboard: http://localhost:8080/index.html (all features in single page)
- API Endpoints: http://localhost:8080/api/status etc.

---

## 部署指南 / Deployment Guide

### 生产环境部署 / Production Deployment

#### 中文
1. **环境准备**
   - 安装 CMake 3.10+ 和支持 C11 的编译器
   - 根据需要安装 GPU 驱动（可选）
   - 确保网络端口 8080 可访问

2. **编译部署**
   ```bash
   # 生产环境优化编译
   mkdir build_prod && cd build_prod
   cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPTIMIZATION=ON
   make -j$(nproc)
   ```

3. **配置优化**
   - 根据硬件调整 `state_dimension` 和 `memory_capacity`
   - 设置 API 密钥认证
   - 配置日志级别和日志文件路径
   - 启用安全监控

4. **服务管理**
   - 使用 systemd (Linux) 或 Windows 服务管理程序
   - 配置自动重启策略
   - 设置资源限制

#### English
1. **Environment Preparation**
   - Install CMake 3.10+ and a C11-compliant compiler
   - Install GPU drivers if needed (optional)
   - Ensure network port 8080 is accessible

2. **Build and Deploy**
   ```bash
   # Production optimized build
   mkdir build_prod && cd build_prod
   cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_OPTIMIZATION=ON
   make -j$(nproc)
   ```

3. **Configuration Optimization**
   - Adjust `state_dimension` and `memory_capacity` based on hardware
   - Set up API key authentication
   - Configure log level and log file path
   - Enable security monitoring

4. **Service Management**
   - Use systemd (Linux) or Windows Service Manager
   - Configure auto-restart policy
   - Set resource limits

### 离线部署 / Offline Deployment

#### 中文
详细的离线部署指南请参考 `docs/Offline_Deployment_Guide.md`。

#### English
For detailed offline deployment guide, please refer to `docs/Offline_Deployment_Guide.md`.

### 嵌入式平台部署 / Embedded Platform Deployment

#### 中文
嵌入式平台部署请参考 `docs/Embedded_Deployment_Guide.md`。

#### English
For embedded platform deployment, please refer to `docs/Embedded_Deployment_Guide.md`.

---

## 功能说明 / Feature Description

### 核心液态神经网络 / Core Liquid Neural Network

#### 中文
- **CfC 细胞**：Closed-form Continuous-time 液态神经网络细胞，具有封闭形式解，可高效计算
- **7 种 ODE 求解器**：支持闭式解、RK4、RK45、DP54、Rosenbrock、Symplectic、CTBP 求解器
- **四元数增强**：提供旋转不变性增强
- **拉普拉斯变换**：用于频域分析和稳定性分析
- **单一模型原则**：整个系统仅使用一个全局 LNN 实例，所有子系统共享

#### English
- **CfC Cell**: Closed-form Continuous-time liquid neural network cell with closed-form solution for efficient computation
- **7 ODE Solvers**: Supports Closed-Form, RK4, RK45, DP54, Rosenbrock, Symplectic, CTBP solvers
- **Quaternion Enhancement**: Provides rotation invariance enhancement
- **Laplace Transform**: Used for frequency domain analysis and stability analysis
- **Single Model Principle**: The entire system uses only one global LNN instance, shared by all subsystems

### 多模态处理 / Multimodal Processing

#### 中文
- **9 种模态统一输入**：视觉（1024维）、音频（256维）、文本（512维，Token-Free）、传感器（128维）、触觉（128维）、本体感（128维）、热感（64维）、雷达（128维）、电机（64维）
- **视觉处理**：摄像头捕获、图像加载、双目深度估计、SLAM
- **音频处理**：麦克风捕获、语音识别、TTS 合成
- **文本处理**：Unicode 哈希投影、无需分词器
- **传感器融合**：多传感器数据统一处理

#### English
- **9 Modalities Unified Input**: Vision (1024D), Audio (256D), Text (512D, Token-Free), Sensor (128D), Haptic (128D), Proprioception (128D), Thermal (64D), Radar (128D), Motor (64D)
- **Vision Processing**: Camera capture, image loading, stereo depth estimation, SLAM
- **Audio Processing**: Microphone capture, speech recognition, TTS synthesis
- **Text Processing**: Unicode hash projection, no tokenizer required
- **Sensor Fusion**: Unified processing of multi-sensor data

### 认知系统 / Cognitive System

#### 中文
- **自我认知**：7 维状态监控（CPU、内存、错误率、学习状态、演化状态、推理状态、安全状态）
- **元认知**：认知负荷评估、任务优先级调整、错误率监控
- **深度反思**：迭代式 5 维分析（信念一致性、假设检验、矛盾检测、风险评估、体验质量）
- **自我修正**：多维触发的自我修正机制，包括学习率调整、策略更新、记忆重组

#### English
- **Self-Cognition**: 7-dimensional state monitoring (CPU, memory, error rate, learning state, evolution state, reasoning state, safety state)
- **Metacognition**: Cognitive load assessment, task priority adjustment, error rate monitoring
- **Deep Reflection**: Iterative 5-dimensional analysis (belief consistency, hypothesis testing, contradiction detection, risk assessment, experience quality)
- **Self-Correction**: Multi-dimensional triggered self-correction mechanism, including learning rate adjustment, strategy update, memory reorganization

### 推理与规划 / Reasoning and Planning

#### 中文
- **因果推理**：因果关系识别和推理
- **数学推理**：数学物理公式推导和计算
- **分层规划**：多层次任务分解和规划
- **长期规划**：长周期目标规划和动态重规划
- **不确定性推理**：概率推理和不确定性处理

#### English
- **Causal Reasoning**: Causal relationship identification and reasoning
- **Mathematical Reasoning**: Mathematical and physical formula derivation and calculation
- **Hierarchical Planning**: Multi-level task decomposition and planning
- **Long-Term Planning**: Long-cycle goal planning and dynamic replanning
- **Uncertainty Reasoning**: Probabilistic reasoning and uncertainty handling

### 学习系统 / Learning System

#### 中文
- **在线学习**：实时数据流学习，自适应学习率，概念漂移检测
- **强化学习**：PPO、SAC 算法，GAE 优势函数，经验回放
- **模仿学习**：行为克隆（BC）、DAgger、逆强化学习（IRL）、GAIL
- **元学习**：MAML、Reptile，快速适应新任务
- **人工教学**：人类演示教学、知识注入、反馈学习

#### English
- **Online Learning**: Real-time data stream learning, adaptive learning rate, concept drift detection
- **Reinforcement Learning**: PPO, SAC algorithms, GAE advantage function, experience replay
- **Imitation Learning**: Behavior Cloning (BC), DAgger, Inverse Reinforcement Learning (IRL), GAIL
- **Meta-Learning**: MAML, Reptile, rapid adaptation to new tasks
- **Human Teaching**: Human demonstration teaching, knowledge injection, feedback learning

### 训练系统 / Training System

#### 中文
- **6 阶段训练流水线**：预训练 → 深度训练 → 多模态联合训练 → 微调 → 本地适配 → 评估
- **混合精度训练**：FP16/BF16 支持，减少内存使用加速训练
- **分布式训练**：Ring AllReduce、梯度压缩、故障恢复
- **模型版本管理**：检查点、版本控制、模型选择
- **API 训练**：支持通过外部 API 进行远程训练

#### English
- **6-Stage Training Pipeline**: Pretraining → Deep Training → Multimodal Joint Training → Fine-tuning → Local Adaptation → Evaluation
- **Mixed Precision Training**: FP16/BF16 support, reduces memory usage and accelerates training
- **Distributed Training**: Ring AllReduce, gradient compression, fault recovery
- **Model Version Management**: Checkpoints, version control, model selection
- **API Training**: Supports remote training via external API

### 演化系统 / Evolution System

#### 中文
- **CMA-ES**：协方差矩阵自适应演化策略
- **帕累托优化**：多目标优化，非支配排序
- **NAS 神经架构搜索**：自动网络拓扑设计，IPOP 重启
- **多种变异/交叉策略**：5 种选择策略、4 种交叉策略、5 种变异策略

#### English
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy
- **Pareto Optimization**: Multi-objective optimization, non-dominated sorting
- **NAS Neural Architecture Search**: Automatic network topology design, IPOP restart
- **Multiple Mutation/Crossover Strategies**: 5 selection strategies, 4 crossover strategies, 5 mutation strategies

### 记忆系统 / Memory System

#### 中文
- **5 层记忆体系**：
  - 工作记忆：当前任务处理
  - 短期记忆：临时存储
  - 情景记忆：事件序列记忆
  - 语义记忆：事实知识记忆
  - 长期记忆：持久化存储
- **Ebbinghaus 遗忘曲线**：自然衰减机制
- **Hebbian 巩固**：记忆强化机制

#### English
- **5-Layer Memory Hierarchy**:
  - Working Memory: Current task processing
  - Short-term Memory: Temporary storage
  - Episodic Memory: Event sequence memory
  - Semantic Memory: Factual knowledge memory
  - Long-term Memory: Persistent storage
- **Ebbinghaus Forgetting Curve**: Natural decay mechanism
- **Hebbian Consolidation**: Memory reinforcement mechanism

### 知识库 / Knowledge Base

#### 中文
- **知识图谱**：三元组存储（主语-谓语-宾语）
- **多跳推理**：链式推理、路径查找
- **本体工程**：概念层次、属性定义
- **语义网络**：概念关联网络
- **知识整合**：自动知识学习、知识融合

#### English
- **Knowledge Graph**: Triple store (subject-predicate-object)
- **Multi-hop Reasoning**: Chain reasoning, path finding
- **Ontology Engineering**: Concept hierarchy, attribute definition
- **Semantic Network**: Concept association network
- **Knowledge Integration**: Automatic knowledge learning, knowledge fusion

---

## AGI 机器人功能 / AGI Robot Features

### 机器人控制 / Robot Control

#### 中文
- **运动学**：DH 参数运动学正逆解
- **动力学**：刚体动力学、碰撞检测
- **路径规划**：A*、RRT*、多目标路径优化
- **硬件接口**：
  - ROS 1/2 集成
  - 串口通信
  - 多种硬件协议支持
  - 硬件资源管理器

#### English
- **Kinematics**: DH parameter forward/inverse kinematics
- **Dynamics**: Rigid body dynamics, collision detection
- **Path Planning**: A*, RRT*, multi-objective path optimization
- **Hardware Interface**:
  - ROS 1/2 integration
  - Serial communication
  - Multiple hardware protocol support
  - Hardware resource manager

### 仿真环境 / Simulation Environment

#### 中文
- **PyBullet 集成**：物理仿真、关节控制、传感器模拟
- **Gazebo 集成**：高保真仿真、插件系统
- **传感器仿真**：摄像头、激光雷达、IMU、触觉传感器
- **多机器人协调**：多机器人协同控制、群智能算法

#### English
- **PyBullet Integration**: Physics simulation, joint control, sensor simulation
- **Gazebo Integration**: High-fidelity simulation, plugin system
- **Sensor Simulation**: Camera, LiDAR, IMU, haptic sensors
- **Multi-Robot Coordination**: Multi-robot cooperative control, swarm intelligence algorithms

### 机器人功能特性 / Robot Functional Features

#### 中文
| 功能模块 | 功能描述 |
|---------|---------|
| **人形机器人控制** | 全自由度控制、步态生成、平衡控制 |
| **双目视觉** | 立体匹配、深度估计、三维重建、SLAM |
| **物体识别** | 目标检测、姿态估计、颜色识别 |
| **语音交互** | 语音识别、语音合成、对话系统 |
| **多模态学习** | 视觉-音频-文本多模态联合学习 |
| **模仿学习** | 通过观察人类动作学习技能 |
| **自主导航** | 实时定位与地图构建、路径规划、避障 |
| **操作能力** | 抓取、操纵、工具使用 |
| **紧急停止** | 安全停止机制、碰撞避免 |

#### English
| Feature Module | Description |
|---------------|-------------|
| **Humanoid Robot Control** | Full DOF control, gait generation, balance control |
| **Stereo Vision** | Stereo matching, depth estimation, 3D reconstruction, SLAM |
| **Object Recognition** | Object detection, pose estimation, color recognition |
| **Voice Interaction** | Speech recognition, speech synthesis, dialogue system |
| **Multimodal Learning** | Vision-audio-text multimodal joint learning |
| **Imitation Learning** | Learn skills by observing human actions |
| **Autonomous Navigation** | Real-time localization and mapping, path planning, obstacle avoidance |
| **Manipulation Ability** | Grasping, manipulation, tool use |
| **Emergency Stop** | Safety stop mechanism, collision avoidance |

### 机器人操作模式 / Robot Operation Modes

#### 中文
1. **直接控制模式**：直接发送控制指令到硬件
2. **ROS 控制模式**：通过 ROS/ROS2 进行控制
3. **仿真模式**：在 PyBullet/Gazebo 中仿真
4. **混合模式**：仿真与硬件结合的半实物仿真

#### English
1. **Direct Control Mode**: Send control commands directly to hardware
2. **ROS Control Mode**: Control via ROS/ROS2
3. **Simulation Mode**: Simulate in PyBullet/Gazebo
4. **Hybrid Mode**: Hardware-in-the-loop simulation combining simulation and hardware

---

### 机器人模仿学习 / Robot Imitation Learning

#### 中文
系统提供多种模仿学习算法，支持机器人从专家演示中学习策略：

**模仿学习算法类型**：
- **行为克隆（BC）**：直接监督学习，从状态-动作对中学习策略
- **DAgger**：数据集聚合算法，交互式学习，在策略执行中用专家纠正
- **逆强化学习（IRL）**：从演示中推断奖励函数
- **GAIL**：生成对抗模仿学习
- **增强型行为克隆（BC+）**：带正则化的行为克隆
- **IQ-Learn**：逆Q学习，隐式奖励函数
- **贝叶斯IRL**：贝叶斯逆强化学习

**演示数据格式**：
- 支持连续状态和动作序列
- 可选奖励序列
- 时间戳记录
- 演示描述标注

**视觉模仿学习**：
- 支持从图像/视频帧中学习策略
- 深度视觉特征提取
- 多尺度特征支持
- 图像输入：灰度/RGB，可配置尺寸

**配置参数**：
- 学习率、批次大小、训练轮数
- L2正则化强度、Dropout率
- 混合精度训练、GPU加速
- 算法特定参数（DAgger beta、IRL参数等）

**功能特性**：
- 策略保存/加载
- 策略评估（与专家动作匹配度）
- 损失历史记录
- 检查点保存
- 能力开关控制（可启用/禁用）

#### English
The system provides various imitation learning algorithms, enabling robots to learn policies from expert demonstrations:

**Imitation Learning Algorithm Types**:
- **Behavioral Cloning (BC)**: Direct supervised learning from state-action pairs
- **DAgger**: Dataset aggregation algorithm, interactive learning with expert corrections during policy execution
- **Inverse Reinforcement Learning (IRL)**: Infer reward functions from demonstrations
- **GAIL**: Generative Adversarial Imitation Learning
- **Behavioral Cloning Plus (BC+)**: BC with regularization
- **IQ-Learn**: Inverse Q-learning with implicit reward functions
- **Bayesian IRL**: Bayesian inverse reinforcement learning

**Demonstration Data Format**:
- Supports continuous state and action sequences
- Optional reward sequences
- Timestamp recording
- Demonstration description annotations

**Visual Imitation Learning**:
- Supports policy learning from images/video frames
- Deep visual feature extraction
- Multi-scale feature support
- Image input: grayscale/RGB, configurable dimensions

**Configuration Parameters**:
- Learning rate, batch size, training epochs
- L2 regularization strength, Dropout rate
- Mixed precision training, GPU acceleration
- Algorithm-specific parameters (DAgger beta, IRL parameters, etc.)

**Features**:
- Policy save/load
- Policy evaluation (expert action matching rate)
- Loss history recording
- Checkpoint saving
- Capability switch control (enable/disable)

---

### 机器人训练系统 / Robot Training System

#### 中文
系统提供完整的6阶段训练流水线，支持机器人技能训练：

**训练阶段**：
1. **预训练（Pretrain）**：基础能力初始化
2. **深度训练（Deep Train）**：深度网络优化
3. **多模态联合训练（Multimodal）**：多模态输入联合优化
4. **微调（Fine-Tune）**：特定任务微调
5. **本地适配（Local）**：目标环境适配
6. **评估（Evaluation）**：性能评估与验证

**训练配置**：
- 各阶段独立设置训练轮数和学习率
- 批次大小、优化器类型（SGD/Momentum/Adam等）
- 损失函数选择（MSE/MAE/CrossEntropy等）
- 混合精度训练、GPU加速
- 早停机制、验证集划分
- 拉普拉斯增强（可选）

**数据增强**：
- 高斯噪声、均匀噪声
- 随机缩放、随机遮盖、随机裁剪
- MixUp数据混合
- 可配置增强强度

**异步数据加载**：
- 多线程数据预取
- 预取缓冲区管理
- 异步加载状态监控

**训练监控**：
- 当前阶段、轮数、损失值
- 训练/验证准确率
- GPU利用率
- 估计剩余时间
- 检查点保存

**训练模式**：
- 单阶段训练
- 全流水线自动训练
- 局部模块训练
- 自动超参数搜索

**机器人代理**：
- 状态空间：32维，动作空间：16维
- 经验回放缓冲区（容量10000）
- 技能库（最多50个技能）
- 知识库（最多500条知识）
- 目标设定与规划
- 多种学习模式切换（模仿/强化/自监督/进化/持续）
- 自我学习、自我进化、自我修正、自主执行能力开关

#### English
The system provides a complete 6-stage training pipeline for robot skill training:

**Training Stages**:
1. **Pretrain**: Basic capability initialization
2. **Deep Train**: Deep network optimization
3. **Multimodal**: Multi-modal input joint optimization
4. **Fine-Tune**: Task-specific fine-tuning
5. **Local**: Target environment adaptation
6. **Evaluation**: Performance evaluation and verification

**Training Configuration**:
- Independent epoch and learning rate settings per stage
- Batch size, optimizer type (SGD/Momentum/Adam, etc.)
- Loss function selection (MSE/MAE/CrossEntropy, etc.)
- Mixed precision training, GPU acceleration
- Early stopping mechanism, validation split
- Laplace enhancement (optional)

**Data Augmentation**:
- Gaussian noise, uniform noise
- Random scaling, random masking, random cropping
- MixUp data blending
- Configurable augmentation strength

**Asynchronous Data Loading**:
- Multi-threaded data prefetching
- Prefetch buffer management
- Async loading state monitoring

**Training Monitoring**:
- Current stage, epoch, loss value
- Train/validation accuracy
- GPU utilization
- Estimated time remaining
- Checkpoint saving

**Training Modes**:
- Single-stage training
- Full pipeline automatic training
- Local module training
- Automatic hyperparameter search

**Robot Agent**:
- State space: 32-dimensional, Action space: 16-dimensional
- Experience replay buffer (capacity 10000)
- Skill library (up to 50 skills)
- Knowledge base (up to 500 knowledge entries)
- Goal setting and planning
- Multiple learning mode switching (imitation/reinforcement/self-supervised/evolutionary/continuous)
- Self-learning, self-evolution, self-correction, autonomous execution capability switches

---

## 维护指南 / Maintenance Guide

### 日常维护 / Routine Maintenance

#### 中文
- **日志监控**：定期检查服务器日志，关注错误和警告
- **资源监控**：监控 CPU、内存、磁盘使用情况
- **备份**：定期备份模型权重、知识库、配置文件
- **更新**：及时应用安全更新和功能更新

#### English
- **Log Monitoring**: Regularly check server logs, pay attention to errors and warnings
- **Resource Monitoring**: Monitor CPU, memory, disk usage
- **Backup**: Regularly backup model weights, knowledge base, configuration files
- **Updates**: Apply security updates and feature updates in time

### 故障排查 / Troubleshooting

#### 中文
| 问题现象 | 可能原因 | 解决方法 |
|---------|---------|---------|
| 服务器无法启动 | 端口被占用 | 更换端口或释放被占用端口 |
| 内存不足 | 模型参数过大 | 减小 state_dimension |
| GPU 不可用 | 驱动未安装 | 检查 GPU 驱动或使用 CPU 后端（27+ 真实内核算子+线程池并行） |
| 响应缓慢 | 并发任务过多 | 调整 max_concurrent_tasks |

#### English
| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Server cannot start | Port occupied | Change port or release occupied port |
| Insufficient memory | Model parameters too large | Reduce state_dimension |
| GPU unavailable | Driver not installed | Check GPU driver or use CPU backend (27+ real kernel operators + thread pool) |
| Slow response | Too many concurrent tasks | Adjust max_concurrent_tasks |

### 性能优化 / Performance Optimization

#### 中文
- **GPU 加速**：启用并配置合适的 GPU 后端，或使用 CPU 后端线程池+SIMD 加速
- **批处理**：适当增大批处理大小
- **内存管理**：及时清理不再需要的数据
- **并发优化**：根据硬件调整线程池大小

#### English
- **GPU Acceleration**: Enable and configure appropriate GPU backend, or use CPU backend thread pool + SIMD acceleration
- **Batching**: Increase batch size appropriately
- **Memory Management**: Timely clean up unneeded data
- **Concurrency Optimization**: Adjust thread pool size based on hardware

---

## 开发指南 / Development Guide

### 目录结构 / Directory Structure

#### 中文
```
self-Z/
├── src/                    # 源代码 / Source Code
│   ├── core/              # CfC 核心 / Core (34 files)
│   ├── multimodal/        # 多模态处理 / Multimodal (52 files)
│   ├── cognition/         # 自我认知 / Cognition (6 files)
│   ├── reasoning/         # 推理引擎 / Reasoning (11 files)
│   ├── knowledge/         # 知识库 / Knowledge (18 files)
│   ├── memory/            # 记忆系统 / Memory (7 files)
│   ├── learning/          # 学习系统 / Learning (10 files)
│   ├── training/          # 训练系统 / Training (15 files)
│   ├── robot/             # 机器人控制 / Robot (34 files)
│   ├── gpu/               # GPU 加速 / GPU Backends (10 backends, 17 files)
│   ├── backend/           # HTTP/WS 后端 / Backend (3 files)
│   ├── agi/               # AGI 核心 / AGI Core (3 files)
│   ├── evolution/         # 演化引擎 / Evolution (3 files)
│   ├── safety/            # 安全系统 / Safety (5 files)
│   ├── concurrency/       # 并发 / Concurrency (4 files)
│   ├── distributed/       # 分布式 / Distributed (2 files)
│   ├── programming/       # 自我编程 / Self-Programming (3 files)
│   ├── product_design/    # 产品设计 / Product Design (2 files)
│   ├── multisystem/       # 多系统控制 / Multisystem (3 files)
│   ├── math/              # 数学库 / Math (1 file)
│   └── utils/             # 工具库 / Utils (12 files)
├── include/selflnn/       # 头文件 / Headers (~180 files)
├── frontend/              # 前端 / Frontend (15 HTML + 17 JS)
├── docs/                  # 文档 / Documentation
├── tests/                 # 测试 / Tests (50 files)
├── examples/              # 示例 / Examples (6 files)
├── scripts/               # 脚本 / Scripts (21 files)
├── config/                # 配置模板 / Config Templates
├── CMakeLists.txt         # 构建配置 / Build Config
└── selflnn_config.json    # 运行时配置 / Runtime Config
```

#### English
```
self-Z/
├── src/                    # Source Code
│   ├── core/              # CfC Core (34 files)
│   ├── multimodal/        # Multimodal Processing (52 files)
│   ├── cognition/         # Cognition (6 files)
│   ├── reasoning/         # Reasoning Engine (11 files)
│   ├── knowledge/         # Knowledge Base (18 files)
│   ├── memory/            # Memory System (7 files)
│   ├── learning/          # Learning System (10 files)
│   ├── training/          # Training System (15 files)
│   ├── robot/             # Robot Control (34 files)
│   ├── gpu/               # GPU Acceleration (10 backends, 17 files)
│   ├── backend/           # HTTP/WS Backend (3 files)
│   ├── agi/               # AGI Core (3 files)
│   ├── evolution/         # Evolution Engine (3 files)
│   ├── safety/            # Safety System (5 files)
│   ├── concurrency/       # Concurrency (4 files)
│   ├── distributed/       # Distributed (2 files)
│   ├── programming/       # Self-Programming (3 files)
│   ├── product_design/    # Product Design (2 files)
│   ├── multisystem/       # Multisystem Control (3 files)
│   ├── math/              # Math Library (1 file)
│   └── utils/             # Utilities (12 files)
├── include/selflnn/       # Headers (~180 files)
├── frontend/              # Frontend (15 HTML + 17 JS)
├── docs/                  # Documentation
├── tests/                 # Tests (50 files)
├── examples/              # Examples (6 files)
├── scripts/               # Scripts (21 files)
├── config/                # Config Templates
├── CMakeLists.txt         # Build Config
└── selflnn_config.json    # Runtime Config
```

### 添加新功能 / Adding New Features

#### 中文
1. 在相应的子模块目录中添加源代码
2. 在 `include/selflnn/` 中添加头文件
3. 更新 `CMakeLists.txt`
4. 确保新功能使用共享的全局 LNN 实例
5. 添加测试用例

#### English
1. Add source code in the corresponding submodule directory
2. Add header file in `include/selflnn/`
3. Update `CMakeLists.txt`
4. Ensure new features use the shared global LNN instance
5. Add test cases

---

## 常见问题 / FAQ

### 中文
**Q: SELF-LNN 与其他 AI 框架有什么不同？**

A: SELF-LNN 采用单一 CfC 液态神经网络，不需要多模型融合、注意力机制或独立编码器。所有功能都是 100% 纯 C 实现，零外部 C 库依赖，支持多种 GPU 后端。

**Q: 可以在没有 GPU 的机器上运行吗？**

A: 可以。CPU 后端现已完整实现 27+ 种真实内核算子和线程池并行调度，支持 SIMD 加速（SSE/AVX/AVX2），可在无 GPU 环境下正常运行全部训练和推理任务。

**Q: 如何启用 GPU 加速？**

A: 确保已安装相应的 GPU 驱动，设置配置中的 `enable_gpu` 为 true，系统会自动检测并选择合适的 GPU 后端。NVIDIA(CUDA)、AMD(ROCm)、Apple(Metal — 支持 GPU 原子归约批归一化)、通用(OpenCL/Vulkan — 支持 SVM 共享虚拟内存和 SPIR-V 运行时编译)驱动安装后即可使用。华为昇腾(Ascend)、寒武纪(Cambricon)、Google TPU 需安装对应SDK（AscendCL/CNRT/libtpu），无SDK时通过dlsym自动回退 CPU（27+ 真实内核算子+线程池并行）。

**Q: 在没有连接真实硬件时，系统还能工作吗？**

A: 可以。PyBullet 和 Gazebo 桥接已实现完整物理仿真数据（真实相机渲染、射线碰撞、深度图、点云），ROS/ROS2 节点支持完整 TCP/XMLRPC 通信。系统也内置纯 C 物理引擎作为回退。

**Q: 严格真实数据模式是什么？**

A: Release 构建默认启用 `SELFLNN_STRICT_REAL_DATA` 模式。在该模式下，所有合成数据生成函数返回错误，确保自主学习 100% 使用真实硬件/仿真数据。调试时可设置 `ALLOW_BOOTSTRAP_DATA=ON` 临时启用引导数据用于框架验证。

### English
**Q: How is SELF-LNN different from other AI frameworks?**

A: SELF-LNN uses a single CfC Liquid Neural Network, eliminating the need for multi-model fusion, attention mechanisms, or separate encoders. All functions are 100% pure C with zero external C library dependencies, supporting multiple GPU backends.

**Q: Can it run on machines without a GPU?**

A: Yes. The CPU backend is now fully implemented with 27+ real kernel operators and thread pool parallel scheduling, supporting SIMD acceleration (SSE/AVX/AVX2), capable of running all training and inference tasks without a GPU.

**Q: How to enable GPU acceleration?**

A: Ensure the appropriate GPU drivers are installed, set `enable_gpu` to true in the configuration, and the system will automatically detect and select the appropriate GPU backend. NVIDIA(CUDA), AMD(ROCm), Apple(Metal — GPU atomic reduction for batch norm), and general (OpenCL/Vulkan — SVM shared virtual memory & SPIR-V runtime compilation) work with driver installation only. Huawei Ascend, Cambricon, and Google TPU require vendor SDKs (AscendCL/CNRT/libtpu); without SDK, the system auto-falls back to CPU (27+ real kernel operators + thread pool parallelism) via dlsym.

**Q: Can the system still work without real hardware connected?**

A: Yes. PyBullet and Gazebo bridges provide full physics simulation data (real camera rendering, ray collision, depth maps, point clouds). ROS/ROS2 nodes support full TCP/XMLRPC communication. The system also includes a pure C physics engine as fallback.

**Q: What is Strict Real Data Mode?**

A: Release builds default to `SELFLNN_STRICT_REAL_DATA` mode, where all synthetic data generation functions return errors, ensuring autonomous learning uses 100% real hardware/simulation data. Set `ALLOW_BOOTSTRAP_DATA=ON` in debug to temporarily enable bootstrap data for framework validation.

---

## 文档索引 / Document Index

### 中文
- [架构图与架构详解](docs/Architecture_Diagram.md) - 系统架构图与详细说明
- [API 参考](docs/API_REFERENCE_ZH.md) - API 接口文档
- [部署指南](docs/DEPLOYMENT_GUIDE_ZH.md) - 详细部署说明
- [开发指南](docs/DEVELOPMENT_GUIDE_ZH.md) - 开发者指南
- [AGI 机器人指南](docs/AGI_Robot_Guide.md) - 机器人功能详细说明
- [训练指南](docs/Training_Guide.md) - 训练系统使用指南
- [用户指南](docs/USER_GUIDE.md) - 用户使用手册
- [演示指南](docs/DEMO_GUIDE.md) - 功能演示教程
- [Ubuntu 部署指南](docs/Deployment_Ubuntu.md) - Ubuntu 平台部署
- [离线部署指南](docs/Offline_Deployment_Guide.md) - 离线环境部署
- [嵌入式平台部署指南](docs/Embedded_Deployment_Guide.md) - 嵌入式平台部署
- [Ubuntu 系统恢复指南](docs/Ubuntu_Recovery.md) - 系统恢复
- [WSL 网络故障排除](docs/WSL_Network_Troubleshooting.md) - 网络问题排查

### English
- [Architecture Diagram & Details](docs/Architecture_Diagram.md) - System architecture diagram & detailed explanation
- [API Reference](docs/API_REFERENCE_ZH.md) - API interface documentation
- [Deployment Guide](docs/DEPLOYMENT_GUIDE_ZH.md) - Detailed deployment instructions
- [Development Guide](docs/DEVELOPMENT_GUIDE_ZH.md) - Developer guide
- [AGI Robot Guide](docs/AGI_Robot_Guide.md) - Robot function detailed description
- [Training Guide](docs/Training_Guide.md) - Training system usage guide
- [User Guide](docs/USER_GUIDE.md) - User manual
- [Demo Guide](docs/DEMO_GUIDE.md) - Function demonstration tutorial
- [Ubuntu Deployment Guide](docs/Deployment_Ubuntu.md) - Ubuntu platform deployment
- [Offline Deployment Guide](docs/Offline_Deployment_Guide.md) - Offline environment deployment
- [Embedded Platform Deployment Guide](docs/Embedded_Deployment_Guide.md) - Embedded platform deployment
- [Ubuntu System Recovery Guide](docs/Ubuntu_Recovery.md) - System recovery
- [WSL Network Troubleshooting](docs/WSL_Network_Troubleshooting.md) - Network issue troubleshooting

---

> **文档更新 / Updated**: 2026-05-14
> **开源仓库 / Repository**: https://github.com/Sum-Outman/Self-LNN
> **开发者邮箱 / Developer Email**: silencecrowtom@qq.com
