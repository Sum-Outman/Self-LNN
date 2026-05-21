# SELF-LNN 全液态神经网络 AGI 系统架构图
# SELF-LNN Full Liquid Neural Network AGI System Architecture Diagram

> **版本 / Version:** 1.4.0 | **语言 / Language:** 100% Pure C (C11) | **构建 / Build:** CMake 3.10+
> **核心模型 / Core Model:** CfC (Closed-form Continuous-time) LNN — Token-Free 连续信号架构 / Continuous Signal Architecture
> **ODE求解器 / Solvers:** 7种/7 types (闭式解/Closed-Form, RK4, RK45, DP54, Rosenbrock, Symplectic, CTBP)
> **GPU后端 / Backends:** 10种/10 types | **API Handler槽位 / Slots:** 285 (277已实现/Implemented) | **前端/Frontend:** SPA + 18 JS
> **项目信息 / Info:** [GitHub](https://github.com/Sum-Outman/Self-LNN) | [Email](mailto:silencecrowtom@qq.com)

---

## 一、系统总体架构 / I. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              SELF-LNN AGI 系统总体架构 / System Architecture                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  层1 / Layer 1: 前端交互层 / Frontend Layer — SPA + 18 JS           │  │
│  │  仪表盘/Dashboard | LNN控制台/Console | 训练中心/Training             │  │
│  │  机器人控制/Robot Ctrl | 仿真控制/Simulation | 多模态学习/Multimodal  │  │
│  │  语音控制/Voice Ctrl | 对话界面/Dialogue | 安全面板/Safety Panel      │  │
│  │  知识图谱/Knowledge Graph | API文档/API Docs | 编程工作台/Workbench  │  │
│  │  通信/Comm: WebSocket (实时推送/Real-time) + HTTP REST, 端口/Port 8080│  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │  ↕ HTTP/WebSocket                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  层2 / Layer 2: 后端服务层 / Backend Layer — main.c + backend.c      │  │
│  │  HTTP路由分发/Routing | WebSocket实时通信 | API密钥认证/Auth          │  │
│  │  熔断器/CircuitBreaker | 日志系统/Logging | 线程池调度/ThreadPool     │  │
│  │  配置管理/Config | 信号处理/Signal | 安全头注入/SecurityHeaders      │  │
│  │  285 handler槽位/API slots (277已实现/implemented)                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
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
│  │  ╚════════════════════════════════════════════════════════════════╝  │  │
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
1. **前端交互层**：单页面应用（SPA）+ 18个JS模块，提供完整的Web控制台界面
2. **后端服务层**：提供HTTP REST API和WebSocket通信，处理路由、认证、安全等功能，285个handler槽位（277已实现）
3. **统一多模态输入层**：将9种模态数据通过线性投影求和后注入单一CfC动态系统
4. **核心引擎层**：包含唯一的共享LNN实例，128→256→128网络结构，所有子系统共享该模型

### English
The system is divided into four architectural layers:
1. **Frontend Interaction Layer**: Single Page Application (SPA) with 18 JS modules, providing a complete Web console interface
2. **Backend Service Layer**: Provides HTTP REST API and WebSocket communication, handling routing, authentication, and security; 285 handler slots (277 implemented)
3. **Unified Multimodal Input Layer**: Injects 9 modalities of data into a single CfC dynamic system after linear projection and summation
4. **Core Engine Layer**: Contains the only shared LNN instance, 128→256→128 network structure, with all subsystems sharing this model for processing

---

## 二、模块依赖关系图 / II. Module Dependency Graph

```
                        ┌─────────────────────┐
                        │     main.c           │
                        │    (程序入口)         │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
     │  backend.c       │  │ core/lnn.c     │  │ multimodal/    │
     │  (HTTP/WS服务)   │  │ (LNN核心模型)  │  │ (多模态处理)   │
     └────────┬────────┘  └───────┬────────┘  └───────┬────────┘
              │                    │                    │
    ┌─────────┼──────────┐ ┌──────┼──────────┐ ┌──────┼──────────┐
    │         │          │ │      │          │ │      │          │
    ▼         ▼          ▼ ▼      ▼          ▼ ▼      ▼          ▼
 knowledge  memory  reasoning  evolution  learning  robot  safety
  (知识库) (记忆)  (推理引擎)  (演化)    (学习)  (机器人) (安全)
```

### 中文
`main.c` 作为程序入口，协调三大顶层模块：
- **backend.c**：HTTP/WebSocket服务，管理 API 密钥、安全监控
- **core/lnn.c**：单一 CfC 液态神经网络核心模型
- **multimodal/**：多模态数据统一处理入口

各模块下挂载对应的功能子系统（知识库、记忆、推理、演化、学习、机器人控制、安全）。

### English
`main.c` serves as the program entry point, coordinating three top-level modules:
- **backend.c**: HTTP/WebSocket service, managing API keys and security monitoring
- **core/lnn.c**: The single CfC Liquid Neural Network core model
- **multimodal/**: Unified multimodal data processing entry point

Each module mounts its corresponding functional subsystems (knowledge base, memory, reasoning, evolution, learning, robot control, safety).

---

## 三、数据流架构 / III. Data Flow Architecture

```
  多模态输入 ──────────→ 线性投影 ──────→ CfC ODE动态系统 ──────→ 共享LNN
  (视觉/音频/文本/传感器等)         τ·dh/dt = -h + σ⊙tanh(·)          │
                                                        ┌─────────────┘
                                                        ↓
  输出决策/预测/控制 ←─────────── 子系统注册表 ←─────────── 状态演化
  (21个子系统共享)        (全局1个LNN实例)           h(t+Δt) = ...
```

### 中文
1. **输入阶段**：多模态原始数据（视觉、音频、文本、传感器等）通过各自的线性投影矩阵 `W·x+b` 映射到统一特征空间
2. **演化阶段**：拼接后的统一输入进入单一 CfCCell ODE 连续动态系统，由7种可选求解器进行状态演化
3. **输出阶段**：演化后的状态注入共享主LNN，21个子系统通过注册表读取对应输出进行决策/预测/控制

### English
1. **Input Phase**: Multi-modal raw data (vision, audio, text, sensors, etc.) is mapped into a unified feature space via respective linear projection matrices `W·x+b`
2. **Evolution Phase**: The concatenated unified input enters a single CfCCell ODE continuous dynamic system, with state evolution via 7 selectable ODE solvers
3. **Output Phase**: The evolved state is injected into the shared main LNN; 21 subsystems read corresponding outputs via the registry for decision/prediction/control
