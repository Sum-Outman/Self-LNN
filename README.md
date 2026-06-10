# SELF-LNN 全液态神经网络 AGI 系统
# SELF-LNN Full Liquid Neural Network AGI System

> **版本 / Version:** 1.5.0 | **语言 / Language:** 100% Pure C (C11) | **许可证 / License:** Apache 2.0
> **构建 / Build:** CMake 3.10+ | **推荐编译器 / Recommended Compiler:** VS 2022 / GCC 7+ / Clang 6+
> **开源仓库 / Repository:** https://github.com/Sum-Outman/Self-LNN
> **开发者邮箱 / Developer Email:** silencecrowtom@qq.com

---

## 目录 / Table of Contents
- [项目简介 / Project Overview](#项目简介--project-overview)
- [核心特性 / Core Features](#核心特性--core-features)
- [系统架构 / System Architecture](#系统架构--system-architecture)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [用户手册 / User Guide](#用户手册--user-guide)
- [部署指南 / Deployment Guide](#部署指南--deployment-guide)
- [功能说明 / Feature Description](#功能说明--feature-description)
- [API 参考 / API Reference](#api-参考--api-reference)
- [训练框架 / Training Framework](#训练框架--training-framework)
- [AGI 机器人功能 / AGI Robot Features](#agi-机器人功能--agi-robot-features)
- [维护指南 / Maintenance Guide](#维护指南--maintenance-guide)
- [开发指南 / Development Guide](#开发指南--development-guide)
- [常见问题 / FAQ](#常见问题--faq)

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
| **核心模型** | 单一 CfC 液态神经网络，8 种 ODE 求解器（闭式解、RK4、RK45、CTBP、DP54、Rosenbrock、Forest-Ruth辛、RHS直接评估） |
| **多模态处理** | 支持 9 种模态统一输入：视觉(512维)、音频(256维)、文本(256维)、传感器(128维)、触觉(64维)、本体感(64维)、热感(32维)、雷达(64维)、电机(128维) |
| **架构设计** | Token-Free 连续信号架构，无离散 token，无需注意力机制，零外部 C 库依赖 |
| **认知能力** | 自我认知、元认知、迭代式深度反思、多维自我修正 |
| **推理能力** | 因果推理、数学推理、分层规划、长期规划、不确定性推理 |
| **学习能力** | 在线学习、强化学习（PPO/SAC）、模仿学习（BC/DAgger/IRL）、元学习（MAML/Reptile） |
| **训练系统** | 7 阶段训练流水线（预训练/深度/多模态/微调/局部/API/评估）、混合精度训练、分布式训练、模型版本管理 |
| **演化系统** | CMA-ES 演化策略、帕累托优化、NAS 神经架构搜索 |
| **动态架构** | 运行时网络结构自我调整：扩展/收缩隐藏层、增删层、知识迁移、安全审批、原子交换 |
| **记忆系统** | 5 层记忆体系（短期、长期、情景、语义、工作记忆）、Ebbinghaus 遗忘曲线、Hebbian 巩固 |
| **知识库** | 知识图谱+知识库双引擎、图推理路径查找(DFS回溯)、PageRank/社区检测/Louvain、SPARQL查询、多跳推理、实体嵌入增强对话、知识图谱→LNN桥接(bridge) |
| **机器人控制** | DH 运动学、A*/RRT* 路径规划、ROS/ROS2 集成、PyBullet/Gazebo 仿真、多机器人协调 |
| **GPU 加速** | 10 种 GPU 后端接口（CUDA、OpenCL、Vulkan、Metal、ROCm、CPU-SIMD — 完整内核调度+线程池并行；Intel、Ascend、Cambricon、TPU — 通过dlsym动态加载SDK + CPU自动回退；CPU 后端支持 27+ 种真实内核算子） |
| **后端服务** | HTTP REST API（290 handler槽位，282个API枚举 + 8个向后兼容别名）、WebSocket 实时通信、API 密钥认证、安全监控 |
| **并发系统** | 线程池、无锁队列、读写锁、RCU 机制 |
| **安全系统** | 紧急停止、熔断器、审计日志、内容过滤 |
| **前端界面** | 单页面应用（SPA）、20 个 JS 模块（19个主模块 + 1个Stereo Worker），完整的可视化控制台 |

### English
| Feature Category | Specific Functions |
|-----------------|-------------------|
| **Core Model** | Single CfC Liquid Neural Network, 8 ODE solvers (Closed-Form, RK4, RK45, CTBP, DP54, Rosenbrock, Forest-Ruth Symplectic, RHS Direct) |
| **Multimodal Processing** | Unified input for 9 modalities: Vision(512D), Audio(256D), Text(256D), Sensor(128D), Tactile(64D), Proprioception(64D), Thermal(32D), Radar(64D), Motor(128D) |
| **Architecture Design** | Token-Free continuous signal architecture, no discrete tokens, no attention mechanisms, zero external C library dependencies |
| **Cognitive Abilities** | Self-cognition, metacognition, iterative deep reflection, multi-dimensional self-correction |
| **Reasoning Abilities** | Causal reasoning, mathematical reasoning, hierarchical planning, long-term planning, uncertainty reasoning |
| **Learning Abilities** | Online learning, reinforcement learning (PPO/SAC), imitation learning (BC/DAgger/IRL), meta-learning (MAML/Reptile) |
| **Training System** | 7-stage training pipeline (Pretrain/Deep/Multimodal/Fine-tune/Local/API/Evaluation), mixed precision training, distributed training, model version management |
| **Evolution System** | CMA-ES evolution strategy, Pareto optimization, NAS neural architecture search |
| **Dynamic Architecture** | Runtime network structure self-adjustment: expand/shrink hidden layers, add/remove layers, knowledge transfer, safety approval, atomic swap |
| **Memory System** | 5-layer memory hierarchy (short-term, long-term, episodic, semantic, working memory), Ebbinghaus forgetting curve, Hebbian consolidation |
| **Knowledge Base** | Knowledge graph, triple store, multi-hop reasoning, ontology engineering, semantic network |
| **Robot Control** | DH kinematics, A*/RRT* path planning, ROS/ROS2 integration, PyBullet/Gazebo simulation, multi-robot coordination |
| **GPU Acceleration** | 10 GPU backend interfaces (CUDA, OpenCL, Vulkan, Metal, ROCm, CPU-SIMD — full kernel dispatch + thread pool parallelism; Intel, Ascend, Cambricon, TPU — dlsym-loaded SDK + CPU auto-fallback; CPU backend supports 27+ real kernel operators) |
| **Backend Service** | HTTP REST API (290 handler slots, 282 API enums + 8 backward-compatible aliases), WebSocket real-time communication, API key authentication, security monitoring |
| **Concurrency System** | Thread pool, lock-free queue, read-write lock, RCU mechanism |
| **Safety System** | Emergency stop, circuit breaker, audit log, content filtering |
| **Frontend Interface** | Single Page Application (SPA), 20 JS modules (19 main modules + 1 Stereo Worker), complete visual console |

---

## 系统架构 / System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              SELF-LNN AGI 系统总体架构 / System Architecture                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  层1 / Layer 1: 前端交互层 / Frontend Layer — 1 HTML (SPA) + 20 JS │  │
│  │  仪表盘/Dashboard | LNN控制台/Console | 训练中心/Training             │  │
│  │  机器人控制/Robot Ctrl | 仿真控制/Simulation | 数据引擎/DataEngine    │  │
│  │  语音控制/Voice Ctrl | 对话界面/Dialogue | AGI控制器/AGI Controller  │  │
│  │  知识图谱/Knowledge Graph | API服务/API Service | 可视化/Visualization│  │
│  │  编程工作台/Programming | 产品设计/ProductDesign | 文本命令/TextCmd   │  │
│  │  通信/Comm: HTTP REST 8080 + WebSocket 8081 (独立端口/Separate Port)   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │  ↕ HTTP:8080 / WS:8081                │
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
│  │  视觉/Vision[512]   ───→ W·x+b ──┐                                    │  │
│  │  音频/Audio[256]   ───→ W·x+b ──┤                                    │  │
│  │  文本/Text[256]    ───→ W·x+b ──┤                                    │  │
│  │  (Unicode码点/Codepoint→哈希投影/Hash→float[]) ──┤                    │  │
│  │  传感器/Sensor[128]───→ W·x+b ──┤                                    │  │
│  │  触觉/Tactile[64]  ───→ W·x+b ──┤                                    │  │
│  │  本体感/Proprio[64]──→ W·x+b ──┤                                    │  │
│  │  热感/Thermal[32]  ───→ W·x+b ──┤                                    │  │
│  │  雷达/Radar[64]    ───→ W·x+b ──┤                                    │  │
│  │  电机/Motor[128]   ───→ W·x+b ──┼──→ combined_input[256]            │  │
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
│  │  ║  全局1个实例/1 Global Instance | 26子系统注册表共享           ║  │  │
│  │  ╚══════════════════════════════════════════════════════════════╝  │  │
│  │                                                                      │  │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐     │  │
│  │  │自我认知  │ 知识库  │ 推理引擎 │ 记忆系统 │ 学习系统 │ 训练系统 │     │  │
│  │  │Cognit.  │Knowledg.│Reasoning│ Memory  │Learning │Training │     │  │
│  │  ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤     │  │
│  │  │机器人   │GPU加速  │安全系统 │演化引擎 │AGI核心  │分布式   │     │  │
│  │  │ Robot   │GPU Accel│ Safety  │Evolution│AGI Core │Distrib. │     │  │
│  │  ├─────────┴─────────┴─────────┴─────────┴─────────┴─────────┤     │  │
│  │  │ 动态架构控制器 / Dynamic Architecture Controller          │     │  │
│  │  │ 扩展·收缩·增层·删层·知识迁移·安全审批·原子交换             │     │  │
│  │  │ Expand·Shrink·Add/Remove Layers·KnowledgeTransfer·Atomic  │     │  │
│  │  └────────────────────────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 中文
系统分为四层架构：
1. **前端交互层**：单页面应用（SPA，1个index.html）+ 20个JS模块（19个主模块 + 1个Worker）组成，提供完整的Web控制台界面
2. **后端服务层**：提供HTTP REST API和WebSocket通信，处理路由、认证、安全等功能
3. **统一多模态输入层**：将9种模态数据通过固定Xavier投影矩阵注入单一CfC动态系统
4. **核心引擎层**：包含唯一的共享LNN实例，所有子系统共享该模型进行处理；动态架构控制器运行于底层，根据性能反馈自动调整网络拓扑（扩展/收缩/增删层/知识迁移/安全审批）

### 渐进分层架构 / Progressive Layering Architecture

```
共享LNN [唯一核心引擎]
    │
    ├─ 感知馈入(lnn_forward): 视觉CfC/语音/传感器/统一信号 → 修改hidden_state ← 正确
    │
    └─ 生成隔离(lnn_get_output): 对话/TTS/后端回退 → 只读输出 → 私有ODE演化
         ↑                                    ↑
    gen_private_hidden (对话私有CfC ODE)    embedding_table (TTS自包含CfC)
    gen_projection_lnn (独立投影LNN)        waveform_projection (独立权重)
```

**关键原则**：
- **感知模态**（视觉/语音/传感器）通过 `lnn_forward` 馈入共享LNN，修改 `hidden_state` — 这是共享LNN的核心功能
- **生成模态**（对话生成/TTS合成）通过 `lnn_get_output` 只读查询共享LNN输出，使用**私有ODE状态**进行自回归生成 — 不修改共享LNN的 `hidden_state`
- **投影矩阵** Xavier初始化后锁定（`projection_locked=1`），不参与反向传播
- **VH融合** 废弃独立CfC ODE，改用投影拼接+指数移动平均
- 对话生成每token需进行512次共享LNN状态重写的问题已消除（→0次）

### English
The system is divided into four architectural layers:
1. **Frontend Interaction Layer**: Single Page Application (SPA, 1 index.html) with 20 JS modules (19 main modules + 1 Worker), providing a complete Web console interface
2. **Backend Service Layer**: Provides HTTP REST API and WebSocket communication, handling routing, authentication, security, and other functions
3. **Unified Multimodal Input Layer**: Injects 9 modalities of data into a single CfC dynamic system after linear projection and summation
4. **Core Engine Layer**: Contains the only shared LNN instance, with all subsystems sharing this model for processing; the Dynamic Architecture Controller runs at the bottom level, automatically adjusting network topology based on performance feedback (expand/shrink/add-remove layers/knowledge transfer/safety approval)

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
# Windows (推荐VS 2022 / Recommended VS 2022)
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release



# 或使用根目录脚本 / or use root scripts
build.bat

# Linux (GCC/Clang)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 或使用根目录脚本
# or use root scripts
bash build.sh
```

#### English
```bash
# Windows (recommended VS 2022)
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release


# Linux (GCC/Clang)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Or use root scripts: bash build.sh
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
```

#### English
```bash
# Start HTTP Server (default port 8080)
./build/bin/Release/selflnn.exe

# Or specify port
./build/bin/Release/selflnn.exe --port 8080
```

### 配置 / Configuration

#### 中文
配置文件 `config/system_config.json`：

```json
{
  "version": "1.6.0",
  "state_dimension": 256,
  "multimodal_channels": 64,
  "memory_capacity": 10000,
  "max_concurrent_tasks": 100,
  "power_mode": "balanced",
  "gpu_backend": "auto",
  "http_port": 8080,
  "websocket_port": 8081,
  "distributed_port": 8765,
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "mixed_precision": "auto"
  },
  "safety": {
    "content_filter_enabled": true,
    "audit_logging_enabled": true,
    "emergency_stop_enabled": true
  }
}
```

#### English
Configuration file `config/system_config.json`:

```json
{
  "version": "1.6.0",
  "state_dimension": 256,
  "multimodal_channels": 64,
  "memory_capacity": 10000,
  "max_concurrent_tasks": 100,
  "power_mode": "balanced",
  "gpu_backend": "auto",
  "http_port": 8080,
  "websocket_port": 8081,
  "distributed_port": 8765,
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "mixed_precision": "auto"
  },
  "safety": {
    "content_filter_enabled": true,
    "audit_logging_enabled": true,
    "emergency_stop_enabled": true
  }
}
```

### 前端访问 / Frontend Access

#### 中文
启动服务器后，在浏览器中访问：
- 主页/仪表盘：http://localhost:8080 (单页面应用，集成所有功能)
- WebSocket 实时推送：ws://localhost:8081
- API 端点：http://localhost:8080/api/status 等

#### English
After starting the server, visit in your browser:
- Homepage/Dashboard: http://localhost:8080 (single-page app with all features)
- WebSocket real-time push: ws://localhost:8081
- API Endpoints: http://localhost:8080/api/status etc.

---

## 用户手册 / User Guide

### 启动选项 / Startup Options

#### 中文
```bash
./build/bin/Release/selflnn --help
# 可用选项 / Available options:
#   --port <port>     服务端口（默认: 8080）/ Service port (default: 8080)
#   --version         显示版本信息 / Show version info
#   --help            显示此帮助 / Show this help
```

#### English
```bash
./build/bin/Release/selflnn --help
# Available options:
#   --port <port>     Service port (default: 8080)
#   --version         Show version info
#   --help            Show this help
```

### Web 控制台 / Web Console

#### 中文
系统提供以下功能页面（SPA单页面应用内导航）：
- **主控制台**：AGI系统总控面板，含系统状态、对话、推理等模块
- **训练中心**：模型训练管理与监控
- **模拟控制**：环境模拟与测试
- **教学系统**：交互式教学界面
- **语音控制**：语音交互控制
- **知识图谱**：知识图谱可视化与查询
- **API文档**：API接口文档

#### English
The system provides the following functional pages (SPA in-app navigation):
- **Main Console**: AGI system control panel with system status, dialogue, reasoning modules
- **Training Center**: Model training management and monitoring
- **Simulation Control**: Environment simulation and testing
- **Teaching System**: Interactive teaching interface
- **Voice Control**: Voice interaction control
- **Knowledge Graph**: Knowledge graph visualization and query
- **API Documentation**: API interface documentation

### 系统配置文件 / System Configuration Files

#### 中文
配置文件位于 `config/` 目录：

| 文件 | 用途 |
|------|------|
| `system_config.json` | 系统配置 |
| `seed_knowledge.json` | 种子知识库 |
| `slnn.service.example` | Linux systemd 服务配置 |
| `slnn_windows_service.xml.example` | Windows 服务配置 |
| `com.selflnn.slnn.plist.example` | macOS 服务配置 |

#### English
Configuration files are located in the `config/` directory:

| File | Purpose |
|------|---------|
| `system_config.json` | System configuration |
| `seed_knowledge.json` | Seed knowledge base |
| `slnn.service.example` | Linux systemd service configuration |
| `slnn_windows_service.xml.example` | Windows service configuration |
| `com.selflnn.slnn.plist.example` | macOS service configuration |

### 后端配置示例 / Backend Configuration Example

```json
{
    "server": {
        "port": 8080,
        "max_connections": 100,
        "log_level": "info"
    },
    "modules": {
        "multimodal": true,
        "reasoning": true,
        "learning": true,
        "self_evolution": true,
        "robotics": false
    }
}
```

### CfC 单元配置 / CfC Cell Configuration

#### 中文
CfC单元通过 `CfCCellConfig` 结构体配置：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_size` | `size_t` | 必需 | 输入向量大小 |
| `hidden_size` | `size_t` | 必需 | 隐藏状态大小 |
| `time_constant` | `float` | 1.0 | 时间常数 |
| `delta_t` | `float` | 1.0 | 时间步长（秒） |
| `noise_std` | `float` | 0.0 | 噪声标准差 |
| `ode_solver_type` | `int` | 0（闭式解） | ODE求解器类型 |
| `enable_adaptation` | `int` | 0 | 启用参数自适应 |
| `use_multi_timescale` | `int` | 0 | 多时间尺度 |

#### English
CfC cell is configured via the `CfCCellConfig` struct:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_size` | `size_t` | Required | Input vector size |
| `hidden_size` | `size_t` | Required | Hidden state size |
| `time_constant` | `float` | 1.0 | Time constant |
| `delta_t` | `float` | 1.0 | Time step (seconds) |
| `noise_std` | `float` | 0.0 | Noise standard deviation |
| `ode_solver_type` | `int` | 0 (Closed-Form) | ODE solver type |
| `enable_adaptation` | `int` | 0 | Enable parameter adaptation |
| `use_multi_timescale` | `int` | 0 | Multi-timescale |

### ODE 求解器枚举 / ODE Solver Enumeration

| 枚举值 / Enum | 名称 / Name | 说明 / Description |
|---------------|-------------|-------------------|
| 0 | `ODE_SOLVER_CLOSED_FORM` | 封闭形式解（默认，最快）/ Closed-form solution (default, fastest) |
| 1 | `ODE_SOLVER_RK4` | 四阶龙格-库塔 / 4th-order Runge-Kutta |
| 2 | `ODE_SOLVER_RK45` | 五阶自适应 / 5th-order adaptive |
| 3 | `ODE_SOLVER_CTBP` | 连续时间反向传播 / Continuous-time backpropagation |
| 4 | `ODE_SOLVER_DP54` | Dormand-Prince 5(4) 自适应 / Dormand-Prince 5(4) adaptive |
| 5 | `ODE_SOLVER_ROSENBROCK` | Rosenbrock 刚性求解器 / Rosenbrock stiff solver |
| 6 | `ODE_SOLVER_SYMPLECTIC` | Forest-Ruth 辛积分器 / Forest-Ruth symplectic integrator |

### CfC 核心 API 示例 / CfC Core API Examples

#### 创建 CfC 单元 / Create CfC Cell

```c
#include "selflnn/core/cfc_cell.h"

// 配置 / Configuration
CfCCellConfig config = {
    .input_size = 256,
    .hidden_size = 128,
    .time_constant = 1.0f,
    .delta_t = 1.0f,
    .ode_solver_type = ODE_SOLVER_CLOSED_FORM,  // 0=闭式解 / Closed-form
    .noise_std = 0.0f,
    .enable_adaptation = 0
};

// 创建网络实例 / Create network instance
CfCCell* network = cfc_cell_create(&config);
if (!network) {
    printf("创建CfC单元失败 / Failed to create CfC cell\n");
    return -1;
}
```

#### 前向传播 / Forward Pass

```c
float input[256];   // 输入 / Input
float output[128];  // 输出隐藏状态 / Output hidden state

// 执行前向传播 / Execute forward pass
int result = cfc_cell_forward(network, input, output);

// 或指定时间步长 / Or specify time step
result = cfc_cell_forward_with_dt(network, input, 0.5f, output);
```

#### 反向传播 / Backward Pass

```c
float gradient[128];    // 输出梯度 / Output gradient
float input_grad[256];  // 输入梯度 / Input gradient

// 执行反向传播 / Execute backward pass
int result = cfc_cell_backward(network, gradient, input_grad);
```

#### 清理资源 / Cleanup

```c
cfc_cell_free(network);
```

### GPU 加速示例 / GPU Acceleration Example

```c
#include "selflnn/gpu/gpu.h"

// 查询可用GPU后端 / Query available GPU backends
int backend_count = gpu_get_available_backends();
printf("可用GPU后端数 / Available GPU backends: %d\n", backend_count);

// 获取后端信息 / Get backend info
GpuBackendInfo info;
gpu_get_backend_info(GPU_BACKEND_CUDA, &info);
printf("后端 / Backend: %s, 可用 / Available: %s\n", info.name, info.available ? "是/Yes" : "否/No");
```

支持的GPU后端 / Supported GPU backends: CPU, CUDA, OpenCL, Vulkan, Metal, ROCm, Ascend（华为）, Cambricon（寒武纪）, TPU（Google）, Intel

### 模型保存与加载 / Model Save and Load

```c
// 保存到文件 / Save to file
FILE* fout = fopen("model.bin", "wb");
if (fout) {
    cfc_cell_save(network, fout);
    fclose(fout);
}

// 从文件加载 / Load from file
FILE* fin = fopen("model.bin", "rb");
if (fin) {
    cfc_cell_load(network, fin);
    fclose(fin);
}

// 重置隐藏状态和内部缓存 / Reset hidden state and internal cache
cfc_cell_reset(network);
```

### 动态架构控制器示例 / Dynamic Architecture Controller Example

```c
#include "selflnn/core/architecture_controller.h"

ArchitectureControllerConfig cfg = arch_controller_default_config();
cfg.min_confidence_threshold = 0.6f;   // 最低置信度 / Min confidence
cfg.max_changes_per_hour = 3;           // 每小时最大变更次数 / Max changes per hour
cfg.enable_auto_approval = 1;           // 启用自动审批 / Enable auto approval
cfg.enable_knowledge_transfer = 1;      // 启用知识迁移 / Enable knowledge transfer
ArchitectureController* ctrl = arch_controller_create(&cfg);
```

**手动提交架构变更 / Manual Architecture Change:**
```c
ArchitectureChangeRequest req = arch_controller_default_request();
req.type = ARCH_CHANGE_EXPAND_HIDDEN;
req.target_hidden_size = 512;
req.confidence = 0.8f;
snprintf(req.source_module, sizeof(req.source_module), "UserCommand");

ArchitectureChangeResult result;
arch_controller_submit_change(ctrl, &lnn, &req, &result);
```

**查询当前架构 / Query Current Architecture:**
```c
size_t neurons, params, hidden;
int layers;
arch_controller_get_architecture_stats(lnn, &neurons, &params, &hidden, &layers);
printf("神经元/Neurons: %zu, 参数/Params: %zu, 隐藏层/Hidden: %zux%d层/layers\n", neurons, params, hidden, layers);
```

### 故障排除 / Troubleshooting

#### 构建失败 / Build Failures

| 问题 / Problem | 解决方案 / Solution |
|---------------|-------------------|
| CMake未找到 / CMake not found | 安装CMake 3.10+ / Install CMake 3.10+ |
| 编译器未找到 / Compiler not found | 安装MSVC/GCC/Clang / Install MSVC/GCC/Clang |
| 链接错误 / Link error | 清理后重新构建 / Clean rebuild: `cmake --build build --clean-first` |

#### 运行错误 / Runtime Errors

| 问题 / Problem | 解决方案 / Solution |
|---------------|-------------------|
| 端口被占用 / Port occupied | 使用 `--port` 指定其他端口 / Use `--port` to specify another port: `selflnn --port 8081` |
| GPU加速不工作 / GPU not working | 检查驱动，程序自动回退到CPU / Check driver, program auto-falls back to CPU |
| 内存不足 / Out of memory | 减小 `hidden_size` 或 `input_size` / Reduce `hidden_size` or `input_size` |

#### 调试技巧 / Debug Tips

```bash
# 启用详细日志 / Enable verbose logging
selflnn -l debug

# 使用Valgrind检查内存（Linux/macOS）/ Check memory with Valgrind (Linux/macOS)
valgrind --leak-check=full ./build/bin/Release/selflnn
```

---

## 部署指南 / Deployment Guide

### 系统要求 / System Requirements

| 组件 / Component | 最低要求 / Minimum | 推荐配置 / Recommended |
|-----------------|-------------------|----------------------|
| CPU | x86-64, SSE4.2 | x86-64, AVX2 |
| 内存 / Memory | 4 GB | 16+ GB |
| 存储 / Storage | 500 MB | 10+ GB |
| 编译器 / Compiler | MSVC 2019 / GCC 9 / Clang 12 | MSVC 2022 / GCC 13 / Clang 16 |
| 构建工具 / Build Tool | CMake 3.10 | CMake 3.20+ |
| GPU（可选 / Optional） | CUDA 10.0+ / OpenCL 2.0+ | CUDA 12.0+ |
| 操作系统 / OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 24.04 / macOS 14 |

### 构建项目 / Build Project

#### Windows
```powershell
# 使用根目录构建脚本（推荐）/ Use root build script (recommended)
build.bat

# 或手动构建 / Or manual build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
```

#### Linux / macOS
```bash
# 使用根目录构建脚本（推荐）/ Use root build script (recommended)
chmod +x build.sh
./build.sh

# 或手动构建 / Or manual build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

#### 构建选项 / Build Options
```bash
# 启用GPU支持 / Enable GPU support
cmake .. -DENABLE_GPU=ON

# 调试构建 / Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

**构建产物位置 / Build Artifacts:**
- Windows: `build/bin/Release/selflnn.exe`
- Linux/macOS: `build/bin/Release/selflnn`


### 部署到生产目录 / Deploy to Production Directory

#### Windows
```powershell
# 先编译项目 / Build first
build.bat

# 手动部署到指定目录 / Manual deploy to target directory
mkdir "C:\Program Files\selflnn\bin"
mkdir "C:\Program Files\selflnn\frontend"
mkdir "C:\Program Files\selflnn\config"
copy build\bin\Release\selflnn.exe "C:\Program Files\selflnn\bin\"
xcopy frontend "C:\Program Files\selflnn\frontend\" /E /I
copy config\*.json.example "C:\Program Files\selflnn\config\"
```

#### Linux
```bash
# 先编译项目 / Build first
./build.sh

# 手动部署到 /opt/slnn / Manual deploy to /opt/slnn
sudo mkdir -p /opt/slnn/{bin,frontend,config,data}
sudo cp build/bin/Release/selflnn /opt/slnn/bin/
sudo cp -r frontend/* /opt/slnn/frontend/
sudo cp config/*.json.example /opt/slnn/config/
```

#### 部署目录结构 / Deployment Directory Structure
```
/opt/slnn/
├── bin/                    # 可执行文件 / Executable
│   └── selflnn            # AGI后端服务 / AGI backend service
├── frontend/              # 前端页面（SPA单页面应用）/ Frontend (SPA)
│   ├── index.html         # 主控制台 / Main console
│   ├── css/
│   └── js/
├── config/                # 配置文件 / Configuration
│   └── system_config.json # 系统配置 / System config
└── data/                  # 数据目录（运行时创建）/ Data dir (created at runtime)
```

### 配置后端服务 / Configure Backend Service

使用 `config/system_config.json` 作为配置文件：

主要配置项 / Main configuration items：
```json
{
    "server": {
        "port": 8080,
        "max_connections": 100,
        "log_level": "info"
    },
    "modules": {
        "multimodal": true,
        "reasoning": true,
        "learning": true,
        "self_evolution": true,
        "robotics": false
    }
}
```

### 安装为系统服务 / Install as System Service

项目提供服务配置文件模板，位于 `config/` 目录：

| 文件 / File | 用途 / Purpose |
|------------|---------------|
| `slnn.service.example` | Linux systemd 服务 / Linux systemd service |
| `slnn_windows_service.xml.example` | Windows 服务配置 / Windows service configuration |
| `com.selflnn.slnn.plist.example` | macOS launchd 服务 / macOS launchd service |

#### Linux systemd 服务 / Linux systemd Service
```bash
# 编辑服务文件 / Edit service file
sudo cp config/slnn.service.example /etc/systemd/system/slnn.service
sudo systemctl daemon-reload

# 启动服务 / Start service
sudo systemctl enable slnn
sudo systemctl start slnn

# 查看状态 / Check status
sudo systemctl status slnn
```

#### Windows 服务 / Windows Service
```powershell
# 使用 sc 命令创建服务 / Create service using sc command
sc create "SELF-LNN" binPath="C:\Program Files\selflnn\bin\selflnn.exe" start=auto
sc start "SELF-LNN"
```

#### macOS launchd 服务 / macOS launchd Service
```bash
cp config/com.selflnn.slnn.plist.example ~/Library/LaunchAgents/com.selflnn.slnn.plist
launchctl load ~/Library/LaunchAgents/com.selflnn.slnn.plist
```

### 离线部署 / Offline Deployment

#### 适用场景 / Applicable Scenarios
- 无网络连接的环境（内网服务器、隔离区）/ No network (intranet servers, isolated zones)
- 网络受限环境（仅允许局域网通信）/ Network-restricted (LAN only)
- 高安全要求环境（禁止外网连接）/ High-security (no external connections)

#### 准备工作 / Preparation

本项目**无任何外部依赖**，所有功能由纯C实现，不需要下载任何第三方库。

仅需编译工具 / Only build tools needed：
- **Linux**: `sudo apt-get install -y build-essential cmake`
- **Windows**: CMake 3.10+ 和 Visual Studio Build Tools 2022
- **macOS**: `xcode-select --install && brew install cmake`

#### 创建离线部署包 / Create Offline Deployment Bundle

```bash
# 创建完整的离线部署包 / Create complete offline bundle
mkdir -p slnn-offline-bundle
cp -r /path/to/self-Z/* slnn-offline-bundle/

# 打包 / Package
tar -czf slnn-offline-$(date +%Y%m%d).tar.gz slnn-offline-bundle/
```

#### 传输与部署 / Transfer and Deploy

```bash
# 方法1: USB设备 / Method 1: USB device
sudo mount /dev/sdX1 /mnt/usb
cp slnn-offline-*.tar.gz /mnt/usb/

# 方法2: 局域网SCP / Method 2: LAN SCP
scp slnn-offline-*.tar.gz user@192.168.1.100:/tmp/

# 解压后构建 / Extract and build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### GPU 加速配置 / GPU Acceleration Configuration

SELF-LNN 支持10种GPU计算后端，无需安装额外深度学习框架。

**注意 / Note：** GPU后端通过动态库加载，无需GPU也能正常运行（自动使用CPU后端，支持27+种真实内核算子+线程池+SIMD并行）。

#### 支持的GPU后端一览 / Supported GPU Backends

| 后端 / Backend | 标识 / Identifier | 需要的运行时 / Required Runtime | 适用硬件 / Applicable Hardware |
|---------------|-------------------|-------------------------------|-------------------------------|
| CPU | `GPU_BACKEND_CPU` | 无（纯C自包含）/ None (pure C self-contained) | 所有x86/ARM处理器 / All x86/ARM processors |
| NVIDIA CUDA | `GPU_BACKEND_CUDA` | CUDA Toolkit 10.0+ | NVIDIA GPU |
| AMD ROCm | `GPU_BACKEND_ROCM` | ROCm 5.0+ | AMD GPU |
| Intel GPU | `GPU_BACKEND_INTEL` | Intel oneAPI | Intel 集成/独立GPU / Intel iGPU/dGPU |
| Vulkan | `GPU_BACKEND_VULKAN` | Vulkan SDK 1.2+ | 兼容GPU / Compatible GPU |
| OpenCL | `GPU_BACKEND_OPENCL` | OpenCL SDK 1.2+ | 兼容设备 / Compatible devices |
| Apple Metal | `GPU_BACKEND_METAL` | macOS | Apple GPU |
| 华为昇腾 / Huawei Ascend | `GPU_BACKEND_ASCEND` | 昇腾CANN | 华为昇腾NPU / Huawei Ascend NPU |
| 寒武纪 / Cambricon | `GPU_BACKEND_CAMBRICON` | 寒武纪CNToolkit | 寒武纪MLU / Cambricon MLU |
| Google TPU | `GPU_BACKEND_TPU` | TPU运行时 / TPU runtime | Google TPU |

#### Ubuntu GPU驱动安装 / Ubuntu GPU Driver Installation

**NVIDIA CUDA:**
```bash
sudo apt update
sudo apt install -y nvidia-driver-535
nvidia-smi  # 验证驱动安装 / Verify driver installation
# 安装CUDA Toolkit（推荐官方run包）/ Install CUDA Toolkit (recommended official run package)
```
最低要求 / Minimum: CUDA 10.0+，推荐 / Recommended: CUDA 12.0+，计算能力 / Compute Capability ≥ 3.5

**AMD ROCm:**
```bash
sudo apt install -y rocm-libs rocm-dev
sudo usermod -aG render $USER
sudo usermod -aG video $USER
rocm-smi  # 验证安装 / Verify installation
```
最低要求 / Minimum: ROCm 5.0+，推荐 / Recommended: ROCm 6.0+

**华为昇腾 / Huawei Ascend:**
```bash
# 从华为官网下载CANN软件包 / Download CANN package from Huawei official site
sudo ./Ascend-hdk-910b-npu-driver_x.x.x_linux-xxx.run --install
sudo ./Ascend-cann-toolkit_x.x.x_linux-xxx.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info  # 验证安装 / Verify installation
```

### 性能优化 / Performance Optimization

#### 系统参数调整（Linux）/ System Parameter Tuning (Linux)

```bash
# 增大文件描述符限制 / Increase file descriptor limit
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# 调整内核网络参数（高并发场景）/ Tune kernel network params (high concurrency)
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# GPU持久化模式（NVIDIA，减少驱动初始化开销）/ GPU persistence mode (NVIDIA, reduce driver init overhead)
sudo nvidia-smi -pm 1
```

#### 构建优化 / Build Optimization

```bash
# 链接时优化（LTO）/ Link-time optimization (LTO)
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-flto=auto" \
    -DCMAKE_EXE_LINKER_FLAGS="-flto=auto"

# 针对本机架构优化（注意：编译产物不可跨架构运行）/ Native arch optimization (note: binary not cross-arch portable)
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=native -mtune=native"
```

### 验证部署 / Verify Deployment

```bash
# 1. 检查可执行文件 / Check executable
./build/bin/Release/selflnn --help

# 2. 启动服务 / Start service
./build/bin/Release/selflnn --port 8080 &

# 3. 验证API / Verify API
curl http://localhost:8080/api/status

# 4. 打开Web控制台 / Open Web console
# 浏览器访问 / Visit in browser: http://localhost:8080/
```

### 故障排除 / Troubleshooting

| 问题 / Problem | 原因 / Cause | 解决方案 / Solution |
|---------------|-------------|-------------------|
| 编译失败 / Build fails | 缺少CMake或编译器 / Missing CMake or compiler | 安装CMake 3.10+ 和 C11 编译器 / Install CMake 3.10+ and C11 compiler |
| 端口被占用 / Port occupied | 8080端口被占用 / Port 8080 in use | `selflnn --port 8081` 使用其他端口 / Use another port |
| GPU未检测到 / GPU not detected | 缺少GPU驱动 / Missing GPU driver | 安装对应GPU驱动，或使用CPU模式 / Install GPU driver or use CPU mode |
| 前端页面打不开 / Frontend not loading | 前端文件未部署 / Frontend files not deployed | 确认 `frontend/` 目录在可执行文件同目录下 / Ensure `frontend/` dir is alongside executable |
| CUDA初始化失败 / CUDA init fails | 驱动版本与CUDA版本不匹配 / Driver/CUDA version mismatch | 更新NVIDIA驱动至与CUDA Toolkit兼容的版本 / Update NVIDIA driver |
| 内存不足 / Out of memory | 模型参数过大 / Model params too large | 减小批大小或启用混合精度训练 / Reduce batch size or enable mixed precision |
| 编译内存溢出 / Build OOM | 并行编译占用过多内存 / Parallel build uses too much memory | 减少 `--parallel` 参数值 / Reduce `--parallel` value |

---

## 功能说明 / Feature Description

### 核心液态神经网络 / Core Liquid Neural Network

#### 中文
- **CfC 细胞**：Closed-form Continuous-time 液态神经网络细胞，具有封闭形式解，可高效计算
- **8 种 ODE 求解器**：支持闭式解、RK4、RK45、DP54、Rosenbrock、Symplectic、CTBP、RHS直接评估求解器
- **四元数增强**：提供旋转不变性增强
- **拉普拉斯变换**：用于频域分析和稳定性分析
- **单一模型原则**：整个系统仅使用一个全局 LNN 实例，所有子系统共享

#### English
- **CfC Cell**: Closed-form Continuous-time liquid neural network cell with closed-form solution for efficient computation
- **8 ODE Solvers**: Supports Closed-Form, RK4, RK45, DP54, Rosenbrock, Symplectic, CTBP, RHS Direct solvers
- **Quaternion Enhancement**: Provides rotation invariance enhancement
- **Laplace Transform**: Used for frequency domain analysis and stability analysis
- **Single Model Principle**: The entire system uses only one global LNN instance, shared by all subsystems

### 多模态处理 / Multimodal Processing

#### 中文
- **9 种模态统一输入**：视觉（512维）、音频（256维）、文本（256维，Token-Free）、传感器（128维）、触觉（64维）、本体感（64维）、热感（32维）、雷达（64维）、电机（128维）
- **视觉处理**：摄像头捕获、图像加载、双目深度估计、SLAM
- **音频处理**：麦克风捕获、语音识别、TTS 合成
- **文本处理**：Unicode 哈希投影、无需分词器
- **传感器融合**：多传感器数据统一处理

#### English
- **9 Modalities Unified Input**: Vision (512D), Audio (256D), Text (256D, Token-Free), Sensor (128D), Tactile (64D), Proprioception (64D), Thermal (32D), Radar (64D), Motor (128D)
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
- **7 阶段训练流水线**：预训练 → 深度训练 → 多模态联合训练 → 微调 → 本地适配 → API训练 → 评估
- **混合精度训练**：FP16/BF16 支持，减少内存使用加速训练
- **分布式训练**：Ring AllReduce、梯度压缩、故障恢复
- **模型版本管理**：检查点、版本控制、模型选择
- **API 训练**：支持通过外部 API 进行远程训练

#### English
- **7-Stage Training Pipeline**: Pretraining → Deep Training → Multimodal Joint Training → Fine-tuning → Local Adaptation → API Training → Evaluation
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

### 动态架构控制器 / Dynamic Architecture Controller

#### 中文
- **运行时网络结构变更**：训练中自动检测网络容量不足或冗余，动态调整隐藏层维度和层数
- **6 种架构变更类型**：扩展隐藏层、收缩隐藏层、增加网络层、删除网络层、完全重建、NAS 架构部署
- **知识迁移**：网络重建时通过左上角 Xavier 策略保留旧网络权重，确保已有知识不丢失
- **安全审批**：置信度阈值检查 + 频率限制（3次/小时）+ 维度合法性校验
- **原子交换**：新旧 LNN 指针交换，推理不中断
- **闭环触发**：
  - **自我认知闭环**：`self_cognition` 检测到容量不足 → `arch_controller_submit_change()` → 重建 + 知识迁移
  - **NAS 闭环**：`nas_search_complete()` 搜索完成 → `nas_deploy_best_architecture()` → 自动部署最优架构
  - **演化闭环**：`evolution_run()` 每10代 → `evolution_engine_structural_mutate()` → 概率性结构变异
- **审计日志**：128 条环形历史记录，记录每次变更的请求、结果和时间戳

#### English
- **Runtime Network Structure Change**: Automatically detects insufficient or redundant network capacity during training, dynamically adjusts hidden layer dimensions and layer count
- **6 Architecture Change Types**: Expand hidden, Shrink hidden, Add layer, Remove layer, Full reshape, NAS architecture deployment
- **Knowledge Transfer**: Preserves old network weights via top-left Xavier strategy during network rebuild, ensuring existing knowledge is not lost
- **Safety Approval**: Confidence threshold check + frequency limit (3/hour) + dimension validity check
- **Atomic Swap**: Old→new LNN pointer swap, no inference interruption
- **Closed-Loop Triggers**:
  - **Self-Cognition Loop**: `self_cognition` detects capacity shortage → `arch_controller_submit_change()` → rebuild + knowledge transfer
  - **NAS Loop**: `nas_search_complete()` finishes search → `nas_deploy_best_architecture()` → auto-deploy optimal architecture
  - **Evolution Loop**: `evolution_run()` every 10 generations → `evolution_engine_structural_mutate()` → probabilistic structural mutation
- **Audit Log**: 128-entry circular history, recording request/result/timestamp for each change

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
- **知识图谱+知识库双引擎**：知识图谱(graph)提供图结构化推理，知识库(kb)提供文本条目检索，互补协作
- **图推理路径查找**：DFS回溯最短路径、多跳关系查询、关系模式匹配
- **图分析算法**：PageRank节点重要性、Louvain社区检测、介数/紧密度中心性、图密度/直径
- **SPARQL查询引擎**：支持SELECT/WHERE/OPTIONAL/LIMIT，内建解析器+执行器
- **知识图谱↔LNN桥接(bridge)**：将知识图谱概念嵌入加权聚合后注入LNN，使对话C​fC演化基于结构化知识
- **对话注入体系(7个注入点)**：覆盖对话入口预注入、图推理检索、实体嵌入特征增强、社区调制、回复实体标注、AGI后台循环、教学实时写图
- **知识图谱API端点(7个)**：/api/kg/stats, pagerank, communities, path, search, sparql, visualize
- **本体工程**：概念层次、属性定义
- **知识整合**：知识库→图谱单向导入(import_from_kb)、记忆巩固时自动同步
- **语义网络**：概念关联网络

#### English
- **Knowledge Graph + Knowledge Base Dual Engine**: Graph for structured reasoning, KB for text retrieval, complementary
- **Graph Reasoning Path Finding**: DFS backtracking shortest path, multi-hop query, relation pattern matching
- **Graph Analysis**: PageRank, Louvain community detection, betweenness/closeness centrality, density/diameter
- **SPARQL Query Engine**: SELECT/WHERE/OPTIONAL/LIMIT support, built-in parser+executor
- **KG↔LNN Bridge**: Weighted aggregation of concept embeddings injected into LNN — dialogue CfC evolution based on structured knowledge
- **7 Injection Points**: Dialogue pre-injection, graph reasoning retrieval, entity embedding enhancement, community modulation, reply entity annotation, AGI background loop, real-time teach-to-graph
- **7 KG API Endpoints**: /api/kg/stats, pagerank, communities, path, search, sparql, visualize
- **Ontology Engineering**: Concept hierarchy, attribute definition
- **Knowledge Integration**: KB→KG import, auto-sync on memory consolidation
- **Semantic Network**: Concept association network

---

## API 参考 / API Reference

### 概述 / Overview

SELF-LNN AGI 基于**单一 CfC 液态神经网络 + 渐进分层架构**。感知模态通过 `lnn_forward` 馈入共享LNN，生成模态通过 `lnn_get_output` 只读查询后使用私有ODE自回归，确保零全局状态污染。
默认服务端口 / Default port：`http://localhost:8080`（可通过 `--port` 参数修改 / configurable via `--port`）

### 系统状态 / System Status

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/status | 系统运行状态 / System running status |
| GET | /api/health | 健康检查（含GPU/CPU/组件状态）/ Health check (GPU/CPU/component status) |
| GET | /api/stats | 服务器统计信息 / Server statistics |
| GET | /api/memory | 内存状态 / Memory status |
| GET | /api/system/diagnostic | 系统诊断 / System diagnostic |
| GET | /api/system/export_diagnostic | 导出诊断数据 / Export diagnostic data |
| POST | /api/reset | 重置系统 / Reset system |
| POST | /api/shutdown | 关闭系统 / Shutdown system |
| POST | /api/backup | 系统备份 / System backup |
| POST | /api/model/load | 模型加载 / Model load |

### AGI 功能 / AGI Features

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/agi/features | AGI功能状态列表 / AGI feature status list |
| GET | /api/agi/feature_list | 所有AGI功能状态 / All AGI feature status |
| GET | /api/agi/cognition/state | 自我认知详细状态 / Self-cognition detailed state |
| POST | /api/agi/feature/toggle | 切换AGI功能 / Toggle AGI feature |
| POST | /api/agi/execute | AGI任务执行 / AGI task execution |
| POST | /api/agi/task/status | 任务状态查询 / Task status query |
| POST | /api/agi/self_correction | 触发自我修正 / Trigger self-correction |
| POST | /api/agi/think | 全模态思考（思维链+反思+认知）/ Full-modal thinking (chain-of-thought + reflection + cognition) |
| POST | /api/agi/decide | 自主决策 / Autonomous decision |
| POST | /api/agi/learn | 在线学习 / Online learning |
| POST | /api/agi/evolve | 进化演化 / Evolution |
| POST | /api/agi/memory | 记忆系统读写 / Memory system read/write |
| POST | /api/agi/plan | 自主规划 / Autonomous planning |

### 多模态 / Multimodal

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/vision | 视觉输入处理 / Vision input processing |
| POST | /api/audio | 音频输入处理 / Audio input processing |
| POST | /api/text | 文本输入处理 / Text input processing |
| POST | /api/sensor | 传感器输入处理 / Sensor input processing |
| POST | /api/multimodal/learn | 多模态学习触发 / Multimodal learning trigger |
| GET | /api/multimodal/status | 多模态处理状态 / Multimodal processing status |
| POST | /api/multimodal/process | 多模态输入处理 / Multimodal input processing |
| POST | /api/multimodal/config | 配置多模态参数 / Configure multimodal params |
| POST | /api/multimodal/reset | 重置多模态配置 / Reset multimodal config |
| POST | /api/multimodal/stop | 停止多模态处理 / Stop multimodal processing |
| POST | /api/multimodal/teach | 多模态教学 / Multimodal teaching |
| POST | /api/multimodal/teach/test | 多模态教学测试 / Multimodal teaching test |
| GET | /api/sensor/pipeline/status | 传感器管道状态 / Sensor pipeline status |

### 对话 / Dialogue

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/dialogue | 对话处理 / Dialogue processing |
| GET | /api/dialogue/history | 获取对话历史 / Get dialogue history |
| POST | /api/dialogue/clear | 清除对话历史 / Clear dialogue history |
| POST | /api/dialogue/multimodal | 多模态对话 / Multimodal dialogue |

### 知识库 / Knowledge Base

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET/POST | /api/knowledge | 知识库查询/添加 / Knowledge query/add |
| POST | /api/learning/from-dialogue | 从对话中学习 / Learn from dialogue |
| POST | /api/learning/from-manual | 从说明书学习 / Learn from manual |

#### 知识图谱专用（v1.6+）/ Knowledge Graph (v1.6+)

| 方法 / Method | 路径 / Path | 描述 / Description | 请求体 / Request Body |
|--------------|------------|-------------------|---------------------|
| GET | /api/kg/stats | 图谱统计（节点/边/内存）/ Graph stats (nodes/edges/memory) | - |
| GET | /api/kg/pagerank | PageRank排名 / PageRank ranking | - |
| GET | /api/kg/communities | 社区检测状态 / Community detection status | - |
| POST | /api/kg/path | 实体间路径查询 / Entity path query | `{"source":"实体A","target":"实体B"}` |
| POST | /api/kg/search | 图谱语义搜索 / Graph semantic search | `{"query":"关键词"}` |
| POST | /api/kg/sparql | SPARQL查询 / SPARQL query | `{"sparql":"SELECT ?x WHERE {...}"}` |
| GET | /api/kg/visualize | 图谱可视化JSON导出 / Graph visualization JSON export | - |

### 训练 / Training

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/training | 训练请求 / Training request |
| POST | /api/training/start | 开始训练 / Start training |
| POST | /api/training/stop | 停止训练 / Stop training |
| POST | /api/training/pause | 暂停训练 / Pause training |
| GET | /api/training/status | 训练状态 / Training status |
| POST | /api/training/pretrain | 预训练 / Pretrain |
| POST | /api/training/fine-tune | 微调 / Fine-tune |
| POST | /api/training/transfer | 迁移学习 / Transfer learning |
| POST | /api/training/continual | 持续学习 / Continual learning |
| POST | /api/training/from-scratch | 从零开始训练 / Train from scratch |
| POST | /api/training/external-api | 外部API训练 / External API training |
| GET | /api/training/history | 训练历史 / Training history |
| POST | /api/training/export | 导出训练结果 / Export training results |
| POST | /api/training/log/clear | 清除训练日志 / Clear training log |

### 推理 / Inference

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/reasoning | 执行推理 / Execute inference |

### 机器人控制 / Robot Control

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/robot/status | 机器人状态 / Robot status |
| POST | /api/robot/command | 发送控制命令 / Send control command |
| GET | /api/robot/sensor | 传感器数据 / Sensor data |
| POST | /api/robot/trajectory | 执行轨迹 / Execute trajectory |
| POST | /api/robot/emergency_stop | 紧急停止 / Emergency stop |
| POST | /api/robot/connect | 连接机器人 / Connect robot |
| POST | /api/robot/disconnect | 断开连接 / Disconnect |
| GET | /api/robot/list | 机器人列表 / Robot list |
| POST | /api/robot/parameters | 设置参数 / Set parameters |
| POST | /api/robot/coordinate | 坐标控制 / Coordinate control |
| POST | /api/robot/training | 训练控制 / Training control |
| POST | /api/robot/calibrate | 标定 / Calibrate |
| POST | /api/robot/execute_task | 执行任务 / Execute task |
| POST | /api/robot/execute_action | 执行动作 / Execute action |
| POST | /api/robot/stop_task | 停止任务 / Stop task |
| POST | /api/robot/learn_from_demo | 从演示中学习 / Learn from demo |
| POST | /api/robot/reboot | 重启机器人 / Reboot robot |
| POST | /api/robot/firmware | 固件升级 / Firmware upgrade |
| POST | /api/robot/analyze_screen | 分析机器人屏幕 / Analyze robot screen |
| POST | /api/robot/config/reset | 重置配置 / Reset config |
| POST | /api/multi_robot/sync | 多机器人同步 / Multi-robot sync |

### ROS 接口 / ROS Interface

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/ros/status | ROS Master状态 / ROS Master status |
| POST | /api/ros/configure | 配置ROS连接 / Configure ROS connection |
| GET | /api/ros/nodes | ROS节点列表 / ROS node list |
| GET | /api/ros/topics | ROS主题列表 / ROS topic list |
| POST | /api/ros/publish | 话题发布 / Topic publish |
| POST | /api/ros/subscribe | 话题订阅 / Topic subscribe |
| POST | /api/ros/service | 服务调用 / Service call |
| POST | /api/gazebo/control | Gazebo仿真控制 / Gazebo simulation control |

### 计算机操作 / Computer Operations

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/computer/launch | 启动应用 / Launch application |
| POST | /api/computer/close | 关闭应用 / Close application |
| POST | /api/computer/type | 键盘输入 / Keyboard input |
| POST | /api/computer/screenshot | 截取屏幕 / Take screenshot |
| POST | /api/computer/execute | 执行命令 / Execute command |
| POST | /api/computer/volume | 音量控制 / Volume control |

### 仿真控制 / Simulation Control

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/simulation/start | 启动仿真 / Start simulation |
| POST | /api/simulation/stop | 停止仿真 / Stop simulation |
| GET | /api/simulation/status | 仿真状态 / Simulation status |
| POST | /api/simulation/reset | 重置仿真 / Reset simulation |
| POST | /api/simulation/plan_path | 路径规划 / Path planning |
| POST | /api/simulation/robot_control | 仿真机器人控制 / Simulation robot control |
| POST | /api/simulation/reconstruct | 3D场景重建 / 3D scene reconstruction |

### 教学 API（Say-Look-Touch-Count）/ Teaching API

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/teach/get_concepts | 获取已教概念 / Get taught concepts |
| POST | /api/teach/test_concept | 测试概念 / Test concept |
| POST | /api/teach/say_and_associate | 说与关联 / Say and associate |
| POST | /api/teach/look_and_learn | 看与学习 / Look and learn |
| POST | /api/teach/touch_and_understand | 触与理解 / Touch and understand |
| POST | /api/teach/count_and_generalize | 计数与泛化 / Count and generalize |
| POST | /api/teach/clear_concept | 清除单个概念 / Clear single concept |
| POST | /api/teach/clear_all_concepts | 清除所有概念 / Clear all concepts |

### 语音 / Voice

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/audio/recognize | 语音识别 / Speech recognition |
| POST | /api/tts/synthesize | 语音合成 / TTS synthesis |
| POST | /api/audio/stream | 实时语音流 / Real-time audio stream |
| POST | /api/audio/command | 语音指令 / Voice command |
| GET | /api/voice/history | 语音历史 / Voice history |

### 设备与硬件 / Devices and Hardware

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/devices/list | 列出设备 / List devices |
| POST | /api/devices/register | 注册设备 / Register device |
| POST | /api/devices/unregister | 注销设备 / Unregister device |
| POST | /api/devices/status | 设备状态 / Device status |
| POST | /api/devices/mode | 设置模式 / Set mode |
| POST | /api/devices/emergency_stop | 紧急停止 / Emergency stop |
| POST | /api/hardware/scan | 扫描硬件 / Scan hardware |
| GET | /api/hardware/info | 硬件信息 / Hardware info |
| POST | /api/hardware/config | 配置硬件 / Configure hardware |

### GPU / 加速器 / GPU / Accelerator

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/gpu/status | GPU加速状态 / GPU acceleration status |
| GET | /api/gpu/diagnostic | GPU完整诊断 / GPU full diagnostic |
| POST | /api/gpu/benchmark | GPU基准测试 / GPU benchmark |

### API 密钥 / API Keys

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/key/list | 密钥列表 / Key list |
| POST | /api/key/create | 创建密钥 / Create key |
| POST | /api/key/delete | 删除密钥 / Delete key |
| POST | /api/key/update | 更新密钥 / Update key |
| POST | /api/key/toggle | 启用/禁用 / Enable/disable |
| POST | /api/key/set | 设置密钥 / Set key |
| GET | /api/key/status | 密钥状态 / Key status |
| GET | /api/key/stats | 调用统计 / Usage statistics |
| GET | /api/key/rate-limit | 限流状态 / Rate limit status |

### 技能库 / Skill Library

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/skills | 技能库列表 / Skill list |
| POST | /api/skills/search | 搜索技能 / Search skill |
| POST | /api/skills/execute | 执行技能 / Execute skill |
| POST | /api/skills/compose | 组合技能 / Compose skills |
| GET | /api/skills/stats | 技能统计 / Skill statistics |

### 自主学习 / Autonomous Learning

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/auto-learn/scan | 扫描学习 / Scan learning |
| GET | /api/auto-learn/stats | 自主学习统计 / Autonomous learning stats |
| POST | /api/auto-learn/export | 导出学习结果 / Export learning results |
| POST | /api/auto-learn/toggle | 开关自主学习 / Toggle autonomous learning |

### 演化 / Evolution

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/evolution | 演化请求 / Evolution request |
| POST | /api/evolution/pareto | 帕累托前沿 / Pareto frontier |

### 模仿学习 / Imitation Learning

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/imitation/demonstration | 提交示范数据 / Submit demonstration data |
| POST | /api/imitation/train | 触发训练 / Trigger training |
| POST | /api/imitation/predict | 策略预测 / Policy prediction |
| GET | /api/imitation/status | 学习状态 / Learning status |
| POST | /api/imitation/algorithm | 切换算法 / Switch algorithm |

### 安全 / Safety

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| GET | /api/safety/status | 安全监控状态 / Safety monitoring status |
| GET | /api/safety/events | 安全事件列表 / Safety event list |
| POST | /api/safety/emergency_stop | 紧急停止 / Emergency stop |
| POST | /api/safety/soft_stop | 软停止 / Soft stop |
| POST | /api/safety/reset | 重置安全状态 / Reset safety state |

### 文件与串口 / Files and Serial

| 方法 / Method | 路径 / Path | 描述 / Description |
|--------------|------------|-------------------|
| POST | /api/files/read | 读取文件 / Read file |
| POST | /api/files/write | 写入文件 / Write file |
| POST | /api/files/delete | 删除文件 / Delete file |
| GET | /api/files/list | 列出目录 / List directory |
| GET | /api/serial/list | 串口列表 / Serial port list |
| POST | /api/serial/open | 打开串口 / Open serial port |
| POST | /api/serial/close | 关闭串口 / Close serial port |
| POST | /api/serial/send | 发送数据 / Send data |

### 动态架构控制器 API / Dynamic Architecture Controller API

#### 架构统计查询 / Architecture Statistics Query

**GET** `/api/lnn/architecture/status`

返回当前 LNN 的神经元数量、参数数量和隐藏层配置。
Returns current LNN neuron count, parameter count, and hidden layer configuration.

**响应 / Response:**
```json
{
  "neuron_count": 768,
  "param_count": 953472,
  "hidden_size": 256,
  "num_layers": 2,
  "input_size": 128,
  "output_size": 128
}
```

**错误码 / Error Codes:**

| 码 / Code | 含义 / Meaning |
|-----------|---------------|
| 200 | 成功 / Success |
| 404 | LNN 未初始化 / LNN not initialized |

#### 架构变更历史 / Architecture Change History

**GET** `/api/lnn/architecture/history`

返回最近的架构变更历史记录（最多 128 条）。
Returns recent architecture change history (up to 128 entries).

**响应 / Response:**
```json
{
  "total_changes": 3,
  "entries": [
    {
      "timestamp": "2026-06-03 12:00:00",
      "type": "EXPAND_HIDDEN",
      "old_neurons": 768,
      "new_neurons": 1280,
      "reason": "Self-cognition detected capacity shortage"
    }
  ]
}
```

### HTTP 状态码 / HTTP Status Codes

| 状态码 / Code | 含义 / Meaning |
|--------------|---------------|
| 200 | 成功 / Success |
| 400 | 请求参数错误 / Bad request |
| 403 | 认证失败 / Authentication failed |
| 413 | 请求体超过10MB限制 / Request body exceeds 10MB limit |
| 500 | 内部服务器错误 / Internal server error |
| 503 | 子模块/硬件不可用（不降级返回虚假数据）/ Submodule/hardware unavailable (no fake data fallback) |

### 安全头 / Security Headers

所有 API 响应包含 / All API responses include：
- `Access-Control-Allow-Origin: *`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Cache-Control: no-store`

---

## 训练框架 / Training Framework

### 训练模式 / Training Modes

| 模式 / Mode | 中文名称 / Chinese Name | 描述 / Description |
|------------|-----------------------|------------------|
| TRAIN_MODE_BATCH | 批量训练 / Batch training | 使用全部数据进行一次参数更新 / One parameter update using all data |
| TRAIN_MODE_MINI_BATCH | 小批量训练 / Mini-batch training | 将数据分成小批次进行训练 / Train in small batches |
| TRAIN_MODE_ONLINE | 在线训练 / Online training | 每个样本进行一次参数更新 / One update per sample |
| TRAIN_MODE_ADAPTIVE | 自适应训练 / Adaptive training | 根据数据特性动态调整训练策略 / Dynamically adjust strategy based on data |

### 优化器 / Optimizers

| 优化器 / Optimizer | 描述 / Description |
|-------------------|-------------------|
| OPTIMIZER_SGD | 随机梯度下降 / Stochastic Gradient Descent |
| OPTIMIZER_MOMENTUM | 带动量的SGD / SGD with momentum |
| OPTIMIZER_ADAM | Adam优化器 / Adam optimizer |
| OPTIMIZER_RMSPROP | RMSprop优化器 / RMSprop optimizer |

### 损失函数 / Loss Functions

| 损失函数 / Loss Function | 中文名称 / Chinese Name | 适用场景 / Applicable Scenario |
|------------------------|-----------------------|-------------------|
| LOSS_MSE | 均方误差 / Mean Squared Error | 回归问题 / Regression |
| LOSS_MAE | 平均绝对误差 / Mean Absolute Error | 回归问题 / Regression |
| LOSS_CROSS_ENTROPY | 交叉熵 / Cross Entropy | 分类问题 / Classification |
| LOSS_BINARY_CROSS_ENTROPY | 二元交叉熵 / Binary Cross Entropy | 二分类问题 / Binary classification |

### 学习率调度 / Learning Rate Schedulers

| 调度器 / Scheduler | 中文名称 / Chinese Name | 描述 / Description |
|-------------------|-----------------------|------------------|
| SCHEDULER_CONSTANT | 恒定学习率 / Constant LR | 学习率保持不变 / Learning rate stays constant |
| SCHEDULER_STEP | 阶梯衰减 / Step decay | 每隔固定步数衰减 / Decay at fixed step intervals |
| SCHEDULER_EXPONENTIAL | 指数衰减 / Exponential decay | 每步指数衰减 / Exponential decay per step |
| SCHEDULER_COSINE | 余弦衰减 / Cosine decay | 余弦函数衰减 / Cosine function decay |
| SCHEDULER_CYCLIC | 循环学习率 / Cyclic LR | 在范围内循环变化 / Cyclic variation within range |

### 训练流程 / Training Workflow

1. **数据准备 / Data Preparation**：加载和预处理训练数据 / Load and preprocess training data
2. **网络创建 / Network Creation**：初始化液态神经网络 / Initialize liquid neural network
3. **训练器创建 / Trainer Creation**：配置训练参数 / Configure training parameters
4. **训练循环 / Training Loop**：前向传播、损失计算、反向传播、参数更新 / Forward, loss, backward, parameter update
5. **验证 / Validation**：在验证集上评估性能 / Evaluate on validation set
6. **模型保存 / Model Saving**：保存最佳模型 / Save best model
7. **测试 / Testing**：在测试集上评估最终性能 / Evaluate final performance on test set

### 动态架构训练 / Dynamic Architecture Training

SELF-LNN 支持在训练过程中动态调整网络结构，无需重启训练：
SELF-LNN supports dynamic network structure adjustment during training without restart:

| 功能 / Feature | 说明 / Description |
|--------------|-------------------|
| **自动扩容 / Auto-expand** | 自我认知系统检测到网络容量不足时，自动提交架构变更请求，扩展隐藏层维度（+25%） / Self-cognition detects capacity shortage, auto-submits architecture change, expands hidden dims (+25%) |
| **自动剪枝 / Auto-prune** | 检测到冗余神经元时，自动收缩隐藏层，减少计算开销 / Detects redundant neurons, auto-shrinks hidden layers, reduces compute overhead |
| **NAS 部署 / NAS deploy** | NAS 搜索完成后，最优架构自动部署到运行中的 LNN（通过 `nas_deploy_best_architecture`）/ After NAS search, optimal architecture auto-deploys to running LNN |
| **演化结构变异 / Evolution structural mutation** | 演化引擎每 10 代触发一次概率性结构变异（扩展/收缩/增删层）/ Evolution engine triggers probabilistic structural mutation every 10 generations |
| **知识迁移 / Knowledge transfer** | 网络重建时通过左上角 Xavier 策略保留旧网络权重，避免从零训练 / Preserves old weights via top-left Xavier strategy during rebuild |
| **安全审批 / Safety approval** | 所有架构变更需通过置信度阈值（0.6）+ 频率限制（3次/小时）+ 维度校验 / All changes require confidence threshold (0.6) + frequency limit (3/hr) + dimension check |
| **原子交换 / Atomic swap** | 新旧 LNN 指针原子替换，推理不中断 / Old→new LNN pointer atomic swap, no inference interruption |
| **变更可开关 / Toggle changes** | 通过 `ArchitectureControllerConfig` 的 `enable_auto_approval` 控制是否自动执行架构变更 / Control auto-approval via `enable_auto_approval` in `ArchitectureControllerConfig` |

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

### 可控制机器人种类 / Controllable Robot Types

#### 中文
| 类型 | 枚举值 | 说明 |
|------|--------|------|
| 移动机器人 | `ROBOT_TYPE_MOBILE` | 轮式/履带式移动底盘，自主导航与避障 |
| 机械臂 | `ROBOT_TYPE_MANIPULATOR` | 多关节机械臂，抓取与精密操作 |
| 人形机器人 | `ROBOT_TYPE_HUMANOID` | 双足/全自由度人形，步态生成与平衡控制 |
| 空中机器人 | `ROBOT_TYPE_AERIAL` | 固定翼/旋翼飞行器，航迹规划与姿态控制 |
| 四旋翼无人机 | `ROBOT_TYPE_QUADCOPTER` | 四旋翼无人机，悬停/航拍/编队飞行 |
| 水下机器人 | `ROBOT_TYPE_AQUATIC` | 水下航行器，深潜探测与水下作业 |
| 自定义机器人 | `ROBOT_TYPE_CUSTOM` | 用户自定义机器人类型，支持任意构型 |

**多机器人集群控制**：最多同时控制 64 个机器人（`SWARM_MAX_ROBOTS=64`），支持蜂拥/一致性/编队/任务分配/群体智能 5 种集群算法，9 种编队类型（线形/V形/圆形/楔形/方形/纵列/菱形/十字/自定义）。

#### English
| Type | Enum Value | Description |
|------|-----------|-------------|
| Mobile Robot | `ROBOT_TYPE_MOBILE` | Wheeled/tracked mobile base, autonomous navigation & obstacle avoidance |
| Manipulator | `ROBOT_TYPE_MANIPULATOR` | Multi-joint robotic arm, grasping & precision operation |
| Humanoid Robot | `ROBOT_TYPE_HUMANOID` | Bipedal/full-DOF humanoid, gait generation & balance control |
| Aerial Robot | `ROBOT_TYPE_AERIAL` | Fixed-wing/rotor aircraft, trajectory planning & attitude control |
| Quadcopter | `ROBOT_TYPE_QUADCOPTER` | Quadrotor drone, hovering/aerial photography/formation flight |
| Aquatic Robot | `ROBOT_TYPE_AQUATIC` | Underwater vehicle, deep diving & underwater operations |
| Custom Robot | `ROBOT_TYPE_CUSTOM` | User-defined robot type, supports arbitrary configurations |

**Multi-Robot Swarm Control**: Up to 64 robots simultaneously (`SWARM_MAX_ROBOTS=64`), supports 5 swarm algorithms (flocking/consensus/formation/task allocation/swarm intelligence) and 9 formation types (line/vee/circle/wedge/square/column/diamond/cross/custom).

### 硬件设备支持 / Hardware Device Support

#### 中文
**设备类型**（15种）：

| 设备类型 | 枚举值 | 说明 |
|---------|--------|------|
| CPU | `HD_DEVICE_CPU` | 处理器算力检测与调度 |
| GPU | `HD_DEVICE_GPU` | 图形处理器加速 |
| NPU | `HD_DEVICE_NPU` | 神经网络专用处理器 |
| 摄像头 | `HD_DEVICE_CAMERA` | 单目/双目/深度摄像头 |
| 麦克风 | `HD_DEVICE_MICROPHONE` | 音频采集 |
| 扬声器 | `HD_DEVICE_SPEAKER` | 语音/音频输出 |
| 串口 | `HD_DEVICE_SERIAL_PORT` | 串行通信接口 |
| 网卡 | `HD_DEVICE_NETWORK_ADAPTER` | 网络通信适配器 |
| 传感器 | `HD_DEVICE_SENSOR` | 通用传感器接口 |
| 机械臂 | `HD_DEVICE_ROBOT_ARM` | 机械臂硬件接口 |
| 电机控制器 | `HD_DEVICE_MOTOR_CONTROLLER` | 电机驱动与控制 |
| 激光雷达 | `HD_DEVICE_LIDAR` | LiDAR点云采集 |
| 深度摄像头 | `HD_DEVICE_DEPTH_CAMERA` | RGB-D深度感知 |
| IMU | `HD_DEVICE_IMU` | 惯性测量单元 |
| 自定义 | `HD_DEVICE_CUSTOM` | 用户自定义设备 |

**通信接口**（11种）：串口、TCP、UDP、CAN总线、Modbus TCP/RTU、WebSocket、I2C、SPI、GPIO、自定义

**通信协议**（8种）：Modbus RTU、Modbus TCP、CAN Bus、OPC-UA、EtherCAT、MQTT、原始串口、自定义

**机器人传感器**（8种）：激光雷达、摄像头、IMU、GNSS卫星导航、力扭矩传感器、温度传感器、压力传感器、接近传感器

**硬件资源角色管理**：
- 摄像头角色：双目左/右、识别、备用
- 麦克风角色：主输入、噪声参考、波束成形、备用
- 扬声器角色：TTS输出、音频反馈、备用

#### English
**Device Types** (15):

| Device Type | Enum Value | Description |
|------------|-----------|-------------|
| CPU | `HD_DEVICE_CPU` | Processor compute detection & scheduling |
| GPU | `HD_DEVICE_GPU` | GPU acceleration |
| NPU | `HD_DEVICE_NPU` | Neural network dedicated processor |
| Camera | `HD_DEVICE_CAMERA` | Monocular/stereo/depth camera |
| Microphone | `HD_DEVICE_MICROPHONE` | Audio capture |
| Speaker | `HD_DEVICE_SPEAKER` | Voice/audio output |
| Serial Port | `HD_DEVICE_SERIAL_PORT` | Serial communication interface |
| Network Adapter | `HD_DEVICE_NETWORK_ADAPTER` | Network communication adapter |
| Sensor | `HD_DEVICE_SENSOR` | Generic sensor interface |
| Robot Arm | `HD_DEVICE_ROBOT_ARM` | Robotic arm hardware interface |
| Motor Controller | `HD_DEVICE_MOTOR_CONTROLLER` | Motor drive & control |
| LiDAR | `HD_DEVICE_LIDAR` | LiDAR point cloud capture |
| Depth Camera | `HD_DEVICE_DEPTH_CAMERA` | RGB-D depth sensing |
| IMU | `HD_DEVICE_IMU` | Inertial Measurement Unit |
| Custom | `HD_DEVICE_CUSTOM` | User-defined device |

**Communication Interfaces** (11): Serial, TCP, UDP, CAN Bus, Modbus TCP/RTU, WebSocket, I2C, SPI, GPIO, Custom

**Communication Protocols** (8): Modbus RTU, Modbus TCP, CAN Bus, OPC-UA, EtherCAT, MQTT, Raw Serial, Custom

**Robot Sensors** (8): LiDAR, Camera, IMU, GNSS, Force/Torque, Temperature, Pressure, Proximity

**Hardware Resource Role Management**:
- Camera roles: stereo left/right, recognition, backup
- Microphone roles: main input, noise reference, beamforming, backup
- Speaker roles: TTS output, audio feedback, backup

### 仿真环境 / Simulation Environment

#### 中文
- **PyBullet 集成**：物理仿真、关节控制、传感器模拟
- **Gazebo 集成**：高保真仿真、插件系统
- **MuJoCo 集成**：高性能物理引擎，接触动力学仿真
- **自定义仿真引擎**：支持用户自定义仿真后端
- **传感器仿真**：摄像头、激光雷达、IMU、触觉传感器
- **多机器人协调**：多机器人协同控制、群智能算法

#### English
- **PyBullet Integration**: Physics simulation, joint control, sensor simulation
- **Gazebo Integration**: High-fidelity simulation, plugin system
- **MuJoCo Integration**: High-performance physics engine, contact dynamics simulation
- **Custom Simulation Engine**: Supports user-defined simulation backends
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
系统提供完整的7阶段训练流水线，支持机器人技能训练：

**训练阶段**：
1. **预训练（Pretrain）**：基础能力初始化
2. **深度训练（Deep Train）**：深度网络优化
3. **多模态联合训练（Multimodal）**：多模态输入联合优化
4. **微调（Fine-Tune）**：特定任务微调
5. **本地适配（Local）**：目标环境适配与一致性正则化
6. **API训练（API）**：外部API知识蒸馏迁移学习
7. **评估（Evaluation）**：性能评估与验证

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
The system provides a complete 7-stage training pipeline for robot skill training:

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

### 项目结构 / Project Structure

```
self-Z/
├── CMakeLists.txt              # 主构建文件（CMake 3.10+，C11标准）
├── build.bat / build.sh        # 根目录构建脚本
├── start.bat / start.sh        # 根目录启动脚本
├── config/                     # 配置文件模板
│   ├── system_config.json      # 系统配置
│   ├── seed_knowledge.json     # 种子知识库
│   ├── slnn.service.example    # Linux systemd服务模板
│   ├── slnn_windows_service.xml.example
│   ├── com.selflnn.slnn.plist.example
│   └── version.h.in
├── include/selflnn/           # 公共头文件
│   ├── core/                  # 核心模块（CfC单元、ODE求解器、动态架构控制器等）
│   ├── multimodal/            # 多模态处理
│   ├── cognition/             # 自我认知
│   ├── reasoning/             # 推理引擎
│   ├── knowledge/             # 知识库
│   ├── memory/                # 记忆系统
│   ├── learning/              # 学习系统
│   ├── evolution/             # 演化引擎
│   ├── training/              # 训练框架
│   ├── robot/                 # 机器人控制
│   ├── gpu/                   # GPU加速
│   ├── backend/               # HTTP后端
│   ├── concurrency/           # 并发工具
│   ├── safety/                # 安全监控
│   ├── distributed/           # 分布式系统
│   ├── programming/           # 自我编程
│   ├── product_design/        # 产品设计
│   ├── multisystem/           # 多系统控制
│   ├── math/                  # 数学库
│   └── utils/                 # 工具函数
├── src/                       # 源代码（与include结构对应）
│   ├── main.c                 # AGI主入口
│   ├── core/                  # CfC单元、ODE求解器、动态架构控制器
│   ├── multimodal/            # 视觉/音频/文本/传感器多模态融合
│   ├── cognition/             # 自我认知、深度反思、元认知
│   ├── reasoning/             # 因果推理、规划、数学物理推理
│   ├── knowledge/             # 知识图谱、语义网络、本体工程
│   ├── memory/                # 工作/短期/长期/情景记忆
│   ├── learning/              # 在线/强化/模仿/元学习
│   ├── evolution/             # 进化引擎、神经架构搜索
│   ├── training/              # 训练流水线、分布式训练、模型管理
│   ├── robot/                 # 机器人控制、运动学、ROS集成
│   ├── gpu/                   # 10种GPU后端接口
│   ├── backend/               # HTTP服务器、WebSocket、API端点
│   ├── agi/                   # AGI核心、能力开关、任务调度
│   ├── safety/                # 安全监控、紧急停止、审计日志
│   ├── concurrency/           # 线程池、无锁结构、RCU
│   ├── distributed/           # 负载均衡、PBFT共识
│   ├── programming/           # 自我编程、C解释器
│   ├── product_design/        # 产品设计
│   ├── multisystem/           # 多系统控制、群智能
│   ├── math/                  # 矩阵运算
│   └── utils/                 # 数学/内存/字符串/日志工具
├── frontend/                  # Web前端
│   ├── index.html             # 主控制台（SPA单页面应用）
│   ├── css/
│   └── js/                    # 19个JS模块（含Worker）
└── build/                     # 构建输出（自动生成）
```

### 模块说明 / Module Description

| 模块 / Module | 路径 / Path | 功能 / Function |
|--------------|------------|----------------|
| core | `src/core/` | CfC单元、ODE求解器（8种）、拉普拉斯分析、LNN核心、动态架构控制器 |
| multimodal | `src/multimodal/` | 视觉/音频/文本/传感器多模态融合 |
| cognition | `src/cognition/` | 自我认知、深度反思、元认知 |
| reasoning | `src/reasoning/` | 因果推理、规划、数学物理推理 |
| knowledge | `src/knowledge/` | 知识图谱、语义网络、本体工程 |
| memory | `src/memory/` | 工作/短期/长期/情景记忆 |
| learning | `src/learning/` | 在线/强化/模仿/元学习 |
| evolution | `src/evolution/` | 进化引擎、神经架构搜索 |
| training | `src/training/` | 训练流水线、分布式训练、模型管理 |
| robot | `src/robot/` | 机器人控制、运动学、ROS集成 |
| gpu | `src/gpu/` | 10种GPU后端接口 |
| backend | `src/backend/` | HTTP服务器、WebSocket、API端点 |
| concurrency | `src/concurrency/` | 线程池、无锁结构、RCU |
| safety | `src/safety/` | 安全监控、紧急停止、审计日志 |
| distributed | `src/distributed/` | 负载均衡、PBFT共识 |
| utils | `src/utils/` | 数学/内存/字符串/日志工具 |

### 环境配置 / Environment Setup

#### 必需工具 / Required Tools
- **编译器 / Compiler**：推荐 Visual Studio 2022（MSVC，已验证100%稳定）；MSVC 2019+（Windows）、GCC 9+（Linux）、Clang 12+（macOS）
- **构建工具 / Build Tool**：CMake 3.10+
- **Git**（可选，用于版本管理 / Optional, for version control）

#### Windows 环境 / Windows Environment
```powershell
# 安装 Visual Studio 2019+（含 C/C++ 开发工具）
# Install Visual Studio 2019+ (with C/C++ development tools)
# 安装 CMake（从 cmake.org 下载）
# Install CMake (download from cmake.org)
# 打开 "Developer Command Prompt for VS" 或使用 PowerShell
# Open "Developer Command Prompt for VS" or use PowerShell
```

#### Linux 环境 / Linux Environment
```bash
sudo apt update
sudo apt install build-essential cmake
```

#### macOS 环境 / macOS Environment
```bash
xcode-select --install
brew install cmake
```

### 构建选项 / Build Options

| 选项 / Option | 默认值 / Default | 说明 / Description |
|--------------|-----------------|-------------------|
| `ENABLE_GPU` | OFF | 启用GPU支持 / Enable GPU support |
| `CMAKE_BUILD_TYPE` | Release | Debug/Release |

> 推荐使用 VS 2022 编译（已验证100%稳定）。如必须使用VS 2026，请在CMakeLists.txt中将 `memory_utils.c`的编译选项设为`/Od`。

### 代码规范 / Code Standards

#### C语言规范 / C Language Standards
- **标准 / Standard**：C11（`CMAKE_C_STANDARD 11`）
- **命名 / Naming**：函数/变量 `snake_case`，类型 `PascalCase`，常量/宏 `UPPER_SNAKE_CASE`
- **缩进 / Indentation**：4个空格，禁止Tab / 4 spaces, no tabs
- **行宽 / Line width**：不超过100字符 / Max 100 characters
- **括号风格 / Brace style**：Allman风格（大括号单独一行）/ Allman style (braces on separate line)
- **编译警告 / Compiler warnings**：MSVC `/W4 /WX`，GCC/Clang `-Wall -Wextra -Werror -pedantic`

#### 文件头注释 / File Header Comment
```c
/**
 * @file filename.c
 * @brief 文件功能说明 / File description
 */
```

#### 内存管理 / Memory Management
- 每个`malloc`对应一个`free` / Every `malloc` has a corresponding `free`
- 优先使用`safe_malloc`/`safe_calloc`等安全函数 / Prefer `safe_malloc`/`safe_calloc` safe functions
- 分配后立即检查NULL / Check NULL immediately after allocation
- 使用`goto cleanup`模式处理错误路径 / Use `goto cleanup` pattern for error paths

### 添加新功能 / Adding New Features

#### 中文
1. 在 `include/selflnn/<module>/` 创建头文件
2. 在 `src/<module>/` 创建实现文件
3. 在 `src/<module>/CMakeLists.txt` 添加构建目标
4. 在主 `CMakeLists.txt` 添加子目录引用
5. 确保新功能使用共享的全局 LNN 实例
6. 添加测试用例

#### English
1. Create header file in `include/selflnn/<module>/`
2. Create implementation file in `src/<module>/`
3. Add build target in `src/<module>/CMakeLists.txt`
4. Add subdirectory reference in main `CMakeLists.txt`
5. Ensure new features use the shared global LNN instance
6. Add test cases

### CfC 单元开发示例 / CfC Cell Development Example

```c
#include "selflnn/core/cfc_cell.h"

// 配置并创建CfC单元 / Configure and create CfC cell
CfCCellConfig config = {
    .input_size = 64,
    .hidden_size = 128,
    .time_constant = 0.1f,
    .delta_t = 0.01f,
    .ode_solver_type = ODE_SOLVER_RK4
};

CfCCell* cell = cfc_cell_create(&config);

// 前向传播 / Forward pass
float input[64];
float hidden[128];
cfc_cell_forward(cell, input, hidden);

// 反向传播训练 / Backward pass training
float gradient[128];
cfc_cell_backward(cell, gradient, NULL);

// 清理 / Cleanup
cfc_cell_free(cell);
```

### 调试 / Debugging

#### Windows (Visual Studio)
在 Visual Studio 中打开 `build/selflnn.sln`，设置 `main.c` 为启动项，按 F5 调试。
Open `build/selflnn.sln` in Visual Studio, set `main.c` as startup item, press F5 to debug.

#### Linux (GDB)
使用调试模式构建，然后用GDB启动程序进行调试。
Build in debug mode, then launch with GDB for debugging.

#### 日志系统 / Logging System
系统提供日志功能，日志级别 / System provides logging with levels：DEBUG < INFO < WARNING < ERROR。

---

## 常见问题 / FAQ

### 中文
**Q: SELF-LNN 与其他 AI 框架有什么不同？**

A: SELF-LNN 采用单一 CfC 液态神经网络，不需要多模型融合、注意力机制或独立编码器。所有功能都是 100% 纯 C 实现，零外部 C 库依赖，支持多种 GPU 后端。

**Q: 可以在没有 GPU 的机器上运行吗？**

A: 可以。CPU 后端现已完整实现 27+ 种真实内核算子和线程池并行调度，支持 SIMD 加速（SSE/AVX/AVX2），可在无 GPU 环境下正常运行全部训练和推理任务。

**Q: 如何启用 GPU 加速？**

A: 确保已安装相应的 GPU 驱动，设置配置中 `gpu_backend` 为 `"auto"` 或指定后端名称（如 `"cuda"`、`"opencl"`），系统会自动检测并选择合适的 GPU 后端。NVIDIA(CUDA)、AMD(ROCm)、Apple(Metal — 支持 GPU 原子归约批归一化)、通用(OpenCL/Vulkan — 支持 SVM 共享虚拟内存和 SPIR-V 运行时编译)驱动安装后即可使用。华为昇腾(Ascend)、寒武纪(Cambricon)、Google TPU 需安装对应SDK（AscendCL/CNRT/libtpu），无SDK时通过dlsym自动回退 CPU（27+ 真实内核算子+线程池并行）。

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

A: Ensure the appropriate GPU drivers are installed, set `gpu_backend` to `"auto"` or specify a backend name (e.g., `"cuda"`, `"opencl"`) in the configuration, and the system will automatically detect and select the appropriate GPU backend. NVIDIA(CUDA), AMD(ROCm), Apple(Metal — GPU atomic reduction for batch norm), and general (OpenCL/Vulkan — SVM shared virtual memory & SPIR-V runtime compilation) work with driver installation only. Huawei Ascend, Cambricon, and Google TPU require vendor SDKs (AscendCL/CNRT/libtpu); without SDK, the system auto-falls back to CPU (27+ real kernel operators + thread pool parallelism) via dlsym.

**Q: Can the system still work without real hardware connected?**

A: Yes. PyBullet and Gazebo bridges provide full physics simulation data (real camera rendering, ray collision, depth maps, point clouds). ROS/ROS2 nodes support full TCP/XMLRPC communication. The system also includes a pure C physics engine as fallback.

**Q: What is Strict Real Data Mode?**

A: Release builds default to `SELFLNN_STRICT_REAL_DATA` mode, where all synthetic data generation functions return errors, ensuring autonomous learning uses 100% real hardware/simulation data. Set `ALLOW_BOOTSTRAP_DATA=ON` in debug to temporarily enable bootstrap data for framework validation.

---

> **文档更新 / Updated**: 2026-06-08
> **开源仓库 / Repository**: https://github.com/Sum-Outman/Self-LNN
> **开发者邮箱 / Developer Email**: silencecrowtom@qq.com
