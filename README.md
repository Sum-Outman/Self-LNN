# SELF-LNN 全液态神经网络 AGI   系统

> **版本:** 1.5.0 | **语言:** 100% 纯 C (C11) | **许可证:** Apache 2.0
> **构建:** CMake 3.10+ | **推荐编译器:** VS 2022 / GCC 7+ / Clang 6+
> **开源仓库:** https://github.com/Sum-Outman/Self-LNN
> **开发者邮箱:** silencecrowtom@qq.com

> ⚠️ **注意：本项目为高级AGI探索和研究项目，代码复杂且更新频繁，不适合初学者学习或作为入门参考。**

> ✅ **训练验证：已通过 CPU 和 GPU 多轮独立训练测试，训练系统稳定收敛（平均收敛率 >96%），双配置编译（GPU ON/OFF）均通过，24/24 核心测试全部通过。**

---

## 目录
- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [用户手册](#用户手册)
- [部署指南](#部署指南)
- [API 参考](#api-参考)
- [训练框架](#训练框架)
- [AGI 机器人功能](#agi-机器人功能)
- [常见问题](#常见问题)

---

## 项目简介
SELF-LNN 是一个 100% 纯 C 语言实现的全模态通用人工智能（AGI）实验系统。该系统采用单一 CfC (Closed-form Continuous-time) 液态神经网络作为核心模型，无需使用多模型融合、跨模态注意力机制或独立的编码器，直接将视觉、音频、文本、传感器等多种模态数据通过线性投影注入同一连续动态系统进行处理。

系统具有自我认知、推理、学习、演化、记忆、机器人控制等多种能力，并支持多种 GPU 后端加速。所有功能均采用纯 C 语言实现，不依赖任何第三方 C 库，保证了系统的稳定性和可移植性。

> **项目定位说明**：本项目目前没有图像和视频生成能力，后期开发根据实际应用可能会加入。项目以感知真实世界和工程智能化控制拓展为主。

---

## 核心特性
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

---

## 系统架构


系统分为四层架构：
1. **前端交互层**：单页面应用（SPA，1个index.html）+ 20个JS模块（19个主模块 + 1个Worker）组成，提供完整的Web控制台界面
2. **后端服务层**：提供HTTP REST API和WebSocket通信，处理路由、认证、安全等功能
3. **统一多模态输入层**：将9种模态数据通过固定Xavier投影矩阵注入单一CfC动态系统
4. **核心引擎层**：包含唯一的共享LNN实例，所有子系统共享该模型进行处理；动态架构控制器运行于底层，根据性能反馈自动调整网络拓扑（扩展/收缩/增删层/知识迁移/安全审批）

## 快速开始
### 系统要求
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

### 编译
```bash
# Windows (推荐VS 2022)
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# 或使用根目录脚本
build.bat

# Linux (GCC/Clang)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 或使用根目录脚本
bash build.sh
```

### 运行
```bash
# 启动 HTTP 服务器（默认端口 8080）
./build/bin/Release/selflnn.exe

# 或指定端口
./build/bin/Release/selflnn.exe --port 8080
```

### 配置
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

### 前端访问
启动服务器后，在浏览器中访问：
- 主页/仪表盘：http://localhost:8080 (单页面应用，集成所有功能)
- WebSocket 实时推送：ws://localhost:8081
- API 端点：http://localhost:8080/api/status 等

---

## 用户手册
### 启动选项
```bash
./build/bin/Release/selflnn --help
# 可用选项
#   --port <port>     服务端口（默认: 8080）
#   --version         显示版本信息
#   --help            显示此帮助
```

### Web 控制台
系统提供以下功能页面（SPA单页面应用内导航）：
- **主控制台**：AGI系统总控面板，含系统状态、对话、推理等模块
- **训练中心**：模型训练管理与监控
- **模拟控制**：环境模拟与测试
- **教学系统**：交互式教学界面
- **语音控制**：语音交互控制
- **知识图谱**：知识图谱可视化与查询
- **API文档**：API接口文档

### 系统配置文件
配置文件位于 `config/` 目录：

| 文件 | 用途 |
|------|------|
| `system_config.json` | 系统配置 |
| `seed_knowledge.json` | 种子知识库 |
| `slnn.service.example` | Linux systemd 服务配置 |
| `slnn_windows_service.xml.example` | Windows 服务配置 |
| `com.selflnn.slnn.plist.example` | macOS 服务配置 |

### 后端配置示例
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

### CfC 单元配置
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

### ODE 求解器枚举
| 枚举值 | 名称 | 说明 |
|---------------|-------------|-------------------|
| 0 | `ODE_SOLVER_CLOSED_FORM` | 封闭形式解（默认，最快）/ Closed-form solution (default, fastest) |
| 1 | `ODE_SOLVER_RK4` | 四阶龙格-库塔 / 4th-order Runge-Kutta |
| 2 | `ODE_SOLVER_RK45` | 五阶自适应 / 5th-order adaptive |
| 3 | `ODE_SOLVER_CTBP` | 连续时间反向传播 / Continuous-time backpropagation |
| 4 | `ODE_SOLVER_DP54` | Dormand-Prince 5(4) 自适应 / Dormand-Prince 5(4) adaptive |
| 5 | `ODE_SOLVER_ROSENBROCK` | Rosenbrock 刚性求解器 / Rosenbrock stiff solver |
| 6 | `ODE_SOLVER_SYMPLECTIC` | Forest-Ruth 辛积分器 / Forest-Ruth symplectic integrator |

### CfC 核心 API 示例
#### 创建 CfC 单元
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

#### 前向传播
```c
float input[256];   // 输入 / Input
float output[128];  // 输出隐藏状态 / Output hidden state

// 执行前向传播 / Execute forward pass
int result = cfc_cell_forward(network, input, output);

// 或指定时间步长
result = cfc_cell_forward_with_dt(network, input, 0.5f, output);
```

#### 反向传播
```c
float gradient[128];    // 输出梯度 / Output gradient
float input_grad[256];  // 输入梯度 / Input gradient

// 执行反向传播 / Execute backward pass
int result = cfc_cell_backward(network, gradient, input_grad);
```

#### 清理资源
```c
cfc_cell_free(network);
```

### GPU 加速示例
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

### 模型保存与加载
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

### 动态架构控制器示例
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

### 故障排除
#### 构建失败
| 问题 | 解决方案 |
|---------------|-------------------|
| CMake未找到 / CMake not found | 安装CMake 3.10+ / Install CMake 3.10+ |
| 编译器未找到 | 安装MSVC/GCC/Clang / Install MSVC/GCC/Clang |
| 链接错误 | 清理后重新构建 / Clean rebuild: `cmake --build build --clean-first` |

#### 运行错误
| 问题 | 解决方案 |
|---------------|-------------------|
| 端口被占用 | 使用 `--port` 指定其他端口 / Use `--port` to specify another port: `selflnn --port 8081` |
| GPU加速不工作 / GPU not working | 检查驱动，程序自动回退到CPU / Check driver, program auto-falls back to CPU |
| 内存不足 | 减小 `hidden_size` 或 `input_size` / Reduce `hidden_size` or `input_size` |

#### 调试技巧
```bash
# 启用详细日志
selflnn -l debug

# 使用Valgrind检查内存（Linux/macOS）/ Check memory with Valgrind (Linux/macOS)
valgrind --leak-check=full ./build/bin/Release/selflnn
```

---

## 部署指南
### 系统要求
| 组件 | 最低要求 | 推荐配置 |
|-----------------|-------------------|----------------------|
| CPU | x86-64, SSE4.2 | x86-64, AVX2 |
| 内存 | 4 GB | 16+ GB |
| 存储 | 500 MB | 10+ GB |
| 编译器 | MSVC 2019 / GCC 9 / Clang 12 | MSVC 2022 / GCC 13 / Clang 16 |
| 构建工具 | CMake 3.10 | CMake 3.20+ |
| GPU（可选 / Optional） | CUDA 10.0+ / OpenCL 2.0+ | CUDA 12.0+ |
| 操作系统 | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 24.04 / macOS 14 |

### 构建项目
#### Windows
```powershell
# 使用根目录构建脚本（推荐）
build.bat

# 或手动构建
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
```

#### Linux / macOS
```bash
# 使用根目录构建脚本（推荐）
chmod +x build.sh
./build.sh

# 或手动构建
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

#### 构建选项
```bash
# 启用GPU支持
cmake .. -DENABLE_GPU=ON

# 调试构建
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

**构建产物位置 / Build Artifacts:**
- Windows: `build/bin/Release/selflnn.exe`
- Linux/macOS: `build/bin/Release/selflnn`


### 部署到生产目录
#### Windows
```powershell
# 先编译项目
build.bat

# 手动部署到指定目录
mkdir "C:\Program Files\selflnn\bin"
mkdir "C:\Program Files\selflnn\frontend"
mkdir "C:\Program Files\selflnn\config"
copy build\bin\Release\selflnn.exe "C:\Program Files\selflnn\bin\"
xcopy frontend "C:\Program Files\selflnn\frontend\" /E /I
copy config\*.json.example "C:\Program Files\selflnn\config\"
```

#### Linux
```bash
# 先编译项目
./build.sh

# 手动部署到 /opt/slnn / Manual deploy to /opt/slnn
sudo mkdir -p /opt/slnn/{bin,frontend,config,data}
sudo cp build/bin/Release/selflnn /opt/slnn/bin/
sudo cp -r frontend/* /opt/slnn/frontend/
sudo cp config/*.json.example /opt/slnn/config/
```

#### 部署目录结构
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

### 配置后端服务
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

### 安装为系统服务
项目提供服务配置文件模板，位于 `config/` 目录：

| 文件 | 用途 |
|------------|---------------|
| `slnn.service.example` | Linux systemd 服务 / Linux systemd service |
| `slnn_windows_service.xml.example` | Windows 服务配置 / Windows service configuration |
| `com.selflnn.slnn.plist.example` | macOS launchd 服务 / macOS launchd service |

#### Linux systemd 服务
```bash
# 编辑服务文件
sudo cp config/slnn.service.example /etc/systemd/system/slnn.service
sudo systemctl daemon-reload

# 启动服务
sudo systemctl enable slnn
sudo systemctl start slnn

# 查看状态
sudo systemctl status slnn
```

#### Windows 服务
```powershell
# 使用 sc 命令创建服务
sc create "SELF-LNN" binPath="C:\Program Files\selflnn\bin\selflnn.exe" start=auto
sc start "SELF-LNN"
```

#### macOS launchd 服务
```bash
cp config/com.selflnn.slnn.plist.example ~/Library/LaunchAgents/com.selflnn.slnn.plist
launchctl load ~/Library/LaunchAgents/com.selflnn.slnn.plist
```

### 离线部署
#### 适用场景
- 无网络连接的环境（内网服务器、隔离区）/ No network (intranet servers, isolated zones)
- 网络受限环境（仅允许局域网通信）/ Network-restricted (LAN only)
- 高安全要求环境（禁止外网连接）/ High-security (no external connections)

#### 准备工作
本项目**无任何外部依赖**，所有功能由纯C实现，不需要下载任何第三方库。

仅需编译工具 / Only build tools needed：
- **Linux**: `sudo apt-get install -y build-essential cmake`
- **Windows**: CMake 3.10+ 和 Visual Studio Build Tools 2022
- **macOS**: `xcode-select --install && brew install cmake`

#### 创建离线部署包
```bash
# 创建完整的离线部署包
mkdir -p slnn-offline-bundle
cp -r /path/to/self-Z/* slnn-offline-bundle/

# 打包
tar -czf slnn-offline-$(date +%Y%m%d).tar.gz slnn-offline-bundle/
```

#### 传输与部署
```bash
# 方法1: USB设备
sudo mount /dev/sdX1 /mnt/usb
cp slnn-offline-*.tar.gz /mnt/usb/

# 方法2: 局域网SCP
scp slnn-offline-*.tar.gz user@192.168.1.100:/tmp/

# 解压后构建
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### GPU 加速配置
SELF-LNN 支持10种GPU计算后端，无需安装额外深度学习框架。

**注意 / Note：** GPU后端通过动态库加载，无需GPU也能正常运行（自动使用CPU后端，支持27+种真实内核算子+线程池+SIMD并行）。

#### 支持的GPU后端一览
| 后端 | 标识 | 需要的运行时 | 适用硬件 |
|---------------|-------------------|-------------------------------|-------------------------------|
| CPU | `GPU_BACKEND_CPU` | 无（纯C自包含）/ None (pure C self-contained) | 所有x86/ARM处理器 / All x86/ARM processors |
| NVIDIA CUDA | `GPU_BACKEND_CUDA` | CUDA Toolkit 10.0+ | NVIDIA GPU |
| AMD ROCm | `GPU_BACKEND_ROCM` | ROCm 5.0+ | AMD GPU |
| Intel GPU | `GPU_BACKEND_INTEL` | Intel oneAPI | Intel 集成/独立GPU / Intel iGPU/dGPU |
| Vulkan | `GPU_BACKEND_VULKAN` | Vulkan SDK 1.2+ | 兼容GPU / Compatible GPU |
| OpenCL | `GPU_BACKEND_OPENCL` | OpenCL SDK 1.2+ | 兼容设备 |
| Apple Metal | `GPU_BACKEND_METAL` | macOS | Apple GPU |
| 华为昇腾 | `GPU_BACKEND_ASCEND` | 昇腾CANN | 华为昇腾NPU / Huawei Ascend NPU |
| 寒武纪 | `GPU_BACKEND_CAMBRICON` | 寒武纪CNToolkit | 寒武纪MLU / Cambricon MLU |
| Google TPU | `GPU_BACKEND_TPU` | TPU运行时 / TPU runtime | Google TPU |

#### Ubuntu GPU驱动安装
**NVIDIA CUDA:**
```bash
sudo apt update
sudo apt install -y nvidia-driver-535
nvidia-smi  # 验证驱动安装 / Verify driver installation
# 安装CUDA Toolkit（推荐官方run包）
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
# 从华为官网下载CANN软件包
sudo ./Ascend-hdk-910b-npu-driver_x.x.x_linux-xxx.run --install
sudo ./Ascend-cann-toolkit_x.x.x_linux-xxx.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info  # 验证安装 / Verify installation
```

### 性能优化
#### 系统参数调整（Linux）

```bash
# 增大文件描述符限制
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# 调整内核网络参数（高并发场景）
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# GPU持久化模式（NVIDIA，减少驱动初始化开销）
sudo nvidia-smi -pm 1
```

#### 构建优化
```bash
# 链接时优化（LTO）
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-flto=auto" \
    -DCMAKE_EXE_LINKER_FLAGS="-flto=auto"

# 针对本机架构优化（注意：编译产物不可跨架构运行）
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=native -mtune=native"
```

### 验证部署
```bash
# 1. 检查可执行文件
./build/bin/Release/selflnn --help

# 2. 启动服务
./build/bin/Release/selflnn --port 8080 &

# 3. 验证API
curl http://localhost:8080/api/status

# 4. 打开Web控制台
# 浏览器访问
```

### 故障排除
| 问题 | 原因 | 解决方案 |
|---------------|-------------|-------------------|
| 编译失败 | 缺少CMake或编译器 / Missing CMake or compiler | 安装CMake 3.10+ 和 C11 编译器 / Install CMake 3.10+ and C11 compiler |
| 端口被占用 | 8080端口被占用 / Port 8080 in use | `selflnn --port 8081` 使用其他端口 / Use another port |
| GPU未检测到 / GPU not detected | 缺少GPU驱动 / Missing GPU driver | 安装对应GPU驱动，或使用CPU模式 / Install GPU driver or use CPU mode |
| 前端页面打不开 | 前端文件未部署 / Frontend files not deployed | 确认 `frontend/` 目录在可执行文件同目录下 / Ensure `frontend/` dir is alongside executable |
| CUDA初始化失败 / CUDA init fails | 驱动版本与CUDA版本不匹配 / Driver/CUDA version mismatch | 更新NVIDIA驱动至与CUDA Toolkit兼容的版本 / Update NVIDIA driver |
| 内存不足 | 模型参数过大 / Model params too large | 减小批大小或启用混合精度训练 |
| 编译内存溢出 | 并行编译占用过多内存 / Parallel build uses too much memory | 减少 `--parallel` 参数值 / Reduce `--parallel` value |

---

## API 参考
### 概述
SELF-LNN AGI 基于**单一 CfC 液态神经网络 + 渐进分层架构**。感知模态通过 `lnn_forward` 馈入共享LNN，生成模态通过 `lnn_get_output` 只读查询后使用私有ODE自回归，确保零全局状态污染。
默认服务端口 / Default port：`http://localhost:8080`（可通过 `--port` 参数修改 / configurable via `--port`）

### 系统状态
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/status | 系统运行状态 |
| GET | /api/health | 健康检查（含GPU/CPU/组件状态）/ Health check (GPU/CPU/component status) |
| GET | /api/stats | 服务器统计信息 |
| GET | /api/memory | 内存状态 |
| GET | /api/system/diagnostic | 系统诊断 |
| GET | /api/system/export_diagnostic | 导出诊断数据 |
| POST | /api/reset | 重置系统 |
| POST | /api/shutdown | 关闭系统 |
| POST | /api/backup | 系统备份 |
| POST | /api/model/load | 模型加载 |

### AGI 功能
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/agi/features | AGI功能状态列表 / AGI feature status list |
| GET | /api/agi/feature_list | 所有AGI功能状态 / All AGI feature status |
| GET | /api/agi/cognition/state | 自我认知详细状态 / Self-cognition detailed state |
| POST | /api/agi/feature/toggle | 切换AGI功能 / Toggle AGI feature |
| POST | /api/agi/execute | AGI任务执行 / AGI task execution |
| POST | /api/agi/task/status | 任务状态查询 |
| POST | /api/agi/self_correction | 触发自我修正 / Trigger self-correction |
| POST | /api/agi/think | 全模态思考（思维链+反思+认知）/ Full-modal thinking (chain-of-thought + reflection + cognition) |
| POST | /api/agi/decide | 自主决策 |
| POST | /api/agi/learn | 在线学习 |
| POST | /api/agi/evolve | 进化演化 |
| POST | /api/agi/memory | 记忆系统读写 / Memory system read/write |
| POST | /api/agi/plan | 自主规划 |

### 多模态
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/vision | 视觉输入处理 |
| POST | /api/audio | 音频输入处理 |
| POST | /api/text | 文本输入处理 |
| POST | /api/sensor | 传感器输入处理 |
| POST | /api/multimodal/learn | 多模态学习触发 |
| GET | /api/multimodal/status | 多模态处理状态 |
| POST | /api/multimodal/process | 多模态输入处理 |
| POST | /api/multimodal/config | 配置多模态参数 |
| POST | /api/multimodal/reset | 重置多模态配置 |
| POST | /api/multimodal/stop | 停止多模态处理 |
| POST | /api/multimodal/teach | 多模态教学 |
| POST | /api/multimodal/teach/test | 多模态教学测试 |
| GET | /api/sensor/pipeline/status | 传感器管道状态 |

### 对话
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/dialogue | 对话处理 |
| GET | /api/dialogue/history | 获取对话历史 |
| POST | /api/dialogue/clear | 清除对话历史 |
| POST | /api/dialogue/multimodal | 多模态对话 |

### 知识库
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET/POST | /api/knowledge | 知识库查询/添加 / Knowledge query/add |
| POST | /api/learning/from-dialogue | 从对话中学习 |
| POST | /api/learning/from-manual | 从说明书学习 |

#### 知识图谱专用（v1.6+）

| 方法 | 路径 | 描述 | 请求体 |
|--------------|------------|-------------------|---------------------|
| GET | /api/kg/stats | 图谱统计（节点/边/内存）/ Graph stats (nodes/edges/memory) | - |
| GET | /api/kg/pagerank | PageRank排名 / PageRank ranking | - |
| GET | /api/kg/communities | 社区检测状态 | - |
| POST | /api/kg/path | 实体间路径查询 | `{"source":"实体A","target":"实体B"}` |
| POST | /api/kg/search | 图谱语义搜索 | `{"query":"关键词"}` |
| POST | /api/kg/sparql | SPARQL查询 / SPARQL query | `{"sparql":"SELECT ?x WHERE {...}"}` |
| GET | /api/kg/visualize | 图谱可视化JSON导出 / Graph visualization JSON export | - |

### 训练
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/training | 训练请求 |
| POST | /api/training/start | 开始训练 |
| POST | /api/training/stop | 停止训练 |
| POST | /api/training/pause | 暂停训练 |
| GET | /api/training/status | 训练状态 |
| POST | /api/training/pretrain | 预训练 |
| POST | /api/training/fine-tune | 微调 / Fine-tune |
| POST | /api/training/transfer | 迁移学习 |
| POST | /api/training/continual | 持续学习 |
| POST | /api/training/from-scratch | 从零开始训练 |
| POST | /api/training/external-api | 外部API训练 / External API training |
| GET | /api/training/history | 训练历史 |
| POST | /api/training/export | 导出训练结果 |
| POST | /api/training/log/clear | 清除训练日志 |

### 推理
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/reasoning | 执行推理 |

### 机器人控制
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/robot/status | 机器人状态 |
| POST | /api/robot/command | 发送控制命令 |
| GET | /api/robot/sensor | 传感器数据 |
| POST | /api/robot/trajectory | 执行轨迹 |
| POST | /api/robot/emergency_stop | 紧急停止 |
| POST | /api/robot/connect | 连接机器人 |
| POST | /api/robot/disconnect | 断开连接 |
| GET | /api/robot/list | 机器人列表 |
| POST | /api/robot/parameters | 设置参数 |
| POST | /api/robot/coordinate | 坐标控制 |
| POST | /api/robot/training | 训练控制 |
| POST | /api/robot/calibrate | 标定 |
| POST | /api/robot/execute_task | 执行任务 |
| POST | /api/robot/execute_action | 执行动作 |
| POST | /api/robot/stop_task | 停止任务 |
| POST | /api/robot/learn_from_demo | 从演示中学习 |
| POST | /api/robot/reboot | 重启机器人 |
| POST | /api/robot/firmware | 固件升级 |
| POST | /api/robot/analyze_screen | 分析机器人屏幕 |
| POST | /api/robot/config/reset | 重置配置 |
| POST | /api/multi_robot/sync | 多机器人同步 / Multi-robot sync |

### ROS 接口
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/ros/status | ROS Master状态 / ROS Master status |
| POST | /api/ros/configure | 配置ROS连接 / Configure ROS connection |
| GET | /api/ros/nodes | ROS节点列表 / ROS node list |
| GET | /api/ros/topics | ROS主题列表 / ROS topic list |
| POST | /api/ros/publish | 话题发布 |
| POST | /api/ros/subscribe | 话题订阅 |
| POST | /api/ros/service | 服务调用 |
| POST | /api/gazebo/control | Gazebo仿真控制 / Gazebo simulation control |

### 计算机操作
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/computer/launch | 启动应用 |
| POST | /api/computer/close | 关闭应用 |
| POST | /api/computer/type | 键盘输入 |
| POST | /api/computer/screenshot | 截取屏幕 |
| POST | /api/computer/execute | 执行命令 |
| POST | /api/computer/volume | 音量控制 |

### 仿真控制
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/simulation/start | 启动仿真 |
| POST | /api/simulation/stop | 停止仿真 |
| GET | /api/simulation/status | 仿真状态 |
| POST | /api/simulation/reset | 重置仿真 |
| POST | /api/simulation/plan_path | 路径规划 |
| POST | /api/simulation/robot_control | 仿真机器人控制 |
| POST | /api/simulation/reconstruct | 3D场景重建 / 3D scene reconstruction |

### 教学 API（Say-Look-Touch-Count）
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/teach/get_concepts | 获取已教概念 |
| POST | /api/teach/test_concept | 测试概念 |
| POST | /api/teach/say_and_associate | 说与关联 |
| POST | /api/teach/look_and_learn | 看与学习 |
| POST | /api/teach/touch_and_understand | 触与理解 |
| POST | /api/teach/count_and_generalize | 计数与泛化 |
| POST | /api/teach/clear_concept | 清除单个概念 |
| POST | /api/teach/clear_all_concepts | 清除所有概念 |

### 语音
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/audio/recognize | 语音识别 |
| POST | /api/tts/synthesize | 语音合成 |
| POST | /api/audio/stream | 实时语音流 / Real-time audio stream |
| POST | /api/audio/command | 语音指令 |
| GET | /api/voice/history | 语音历史 |

### 设备与硬件
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/devices/list | 列出设备 |
| POST | /api/devices/register | 注册设备 |
| POST | /api/devices/unregister | 注销设备 |
| POST | /api/devices/status | 设备状态 |
| POST | /api/devices/mode | 设置模式 |
| POST | /api/devices/emergency_stop | 紧急停止 |
| POST | /api/hardware/scan | 扫描硬件 |
| GET | /api/hardware/info | 硬件信息 |
| POST | /api/hardware/config | 配置硬件 |

### GPU / 加速器

| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/gpu/status | GPU加速状态 / GPU acceleration status |
| GET | /api/gpu/diagnostic | GPU完整诊断 / GPU full diagnostic |
| POST | /api/gpu/benchmark | GPU基准测试 / GPU benchmark |

### API 密钥
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/key/list | 密钥列表 |
| POST | /api/key/create | 创建密钥 |
| POST | /api/key/delete | 删除密钥 |
| POST | /api/key/update | 更新密钥 |
| POST | /api/key/toggle | 启用/禁用 / Enable/disable |
| POST | /api/key/set | 设置密钥 |
| GET | /api/key/status | 密钥状态 |
| GET | /api/key/stats | 调用统计 |
| GET | /api/key/rate-limit | 限流状态 |

### 技能库
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/skills | 技能库列表 |
| POST | /api/skills/search | 搜索技能 |
| POST | /api/skills/execute | 执行技能 |
| POST | /api/skills/compose | 组合技能 |
| GET | /api/skills/stats | 技能统计 |

### 自主学习
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/auto-learn/scan | 扫描学习 |
| GET | /api/auto-learn/stats | 自主学习统计 |
| POST | /api/auto-learn/export | 导出学习结果 |
| POST | /api/auto-learn/toggle | 开关自主学习 |

### 演化
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/evolution | 演化请求 |
| POST | /api/evolution/pareto | 帕累托前沿 |

### 模仿学习
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/imitation/demonstration | 提交示范数据 |
| POST | /api/imitation/train | 触发训练 |
| POST | /api/imitation/predict | 策略预测 |
| GET | /api/imitation/status | 学习状态 |
| POST | /api/imitation/algorithm | 切换算法 |

### 安全
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| GET | /api/safety/status | 安全监控状态 |
| GET | /api/safety/events | 安全事件列表 |
| POST | /api/safety/emergency_stop | 紧急停止 |
| POST | /api/safety/soft_stop | 软停止 |
| POST | /api/safety/reset | 重置安全状态 |

### 文件与串口
| 方法 | 路径 | 描述 |
|--------------|------------|-------------------|
| POST | /api/files/read | 读取文件 |
| POST | /api/files/write | 写入文件 |
| POST | /api/files/delete | 删除文件 |
| GET | /api/files/list | 列出目录 |
| GET | /api/serial/list | 串口列表 |
| POST | /api/serial/open | 打开串口 |
| POST | /api/serial/close | 关闭串口 |
| POST | /api/serial/send | 发送数据 |

### 动态架构控制器 API
### HTTP 状态码
| 状态码 | 含义 |
|--------------|---------------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 403 | 认证失败 |
| 413 | 请求体超过10MB限制 / Request body exceeds 10MB limit |
| 500 | 内部服务器错误 |
| 503 | 子模块/硬件不可用（不降级返回虚假数据）/ Submodule/hardware unavailable (no fake data fallback) |

### 安全头
所有 API 响应包含 / All API responses include：
- `Access-Control-Allow-Origin: *`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Cache-Control: no-store`

---

## 训练框架
### 训练模式
| 模式 | 中文名称 / Chinese Name | 描述 |
|------------|-----------------------|------------------|
| TRAIN_MODE_BATCH | 批量训练 | 使用全部数据进行一次参数更新 / One parameter update using all data |
| TRAIN_MODE_MINI_BATCH | 小批量训练 / Mini-batch training | 将数据分成小批次进行训练 |
| TRAIN_MODE_ONLINE | 在线训练 | 每个样本进行一次参数更新 / One update per sample |
| TRAIN_MODE_ADAPTIVE | 自适应训练 | 根据数据特性动态调整训练策略 / Dynamically adjust strategy based on data |

### 优化器
| 优化器 | 描述 |
|-------------------|-------------------|
| OPTIMIZER_SGD | 随机梯度下降 |
| OPTIMIZER_MOMENTUM | 带动量的SGD / SGD with momentum |
| OPTIMIZER_ADAM | Adam优化器 / Adam optimizer |
| OPTIMIZER_RMSPROP | RMSprop优化器 / RMSprop optimizer |

### 损失函数
| 损失函数 | 中文名称 | 适用场景 |
|------------------------|-----------------------|-------------------|
| LOSS_MSE | 均方误差 | 回归问题 / Regression |
| LOSS_MAE | 平均绝对误差 | 回归问题 / Regression |
| LOSS_CROSS_ENTROPY | 交叉熵 | 分类问题 / Classification |
| LOSS_BINARY_CROSS_ENTROPY | 二元交叉熵 | 二分类问题 / Binary classification |

### 学习率调度
| 调度器 | 中文名称 / Chinese Name | 描述 |
|-------------------|-----------------------|------------------|
| SCHEDULER_CONSTANT | 恒定学习率 | 学习率保持不变 / Learning rate stays constant |
| SCHEDULER_STEP | 阶梯衰减 | 每隔固定步数衰减 / Decay at fixed step intervals |
| SCHEDULER_EXPONENTIAL | 指数衰减 | 每步指数衰减 / Exponential decay per step |
| SCHEDULER_COSINE | 余弦衰减 | 余弦函数衰减 / Cosine function decay |
| SCHEDULER_CYCLIC | 循环学习率 | 在范围内循环变化 / Cyclic variation within range |

### 训练流程
1. **数据准备 / Data Preparation**：加载和预处理训练数据 / Load and preprocess training data
2. **网络创建 / Network Creation**：初始化液态神经网络 / Initialize liquid neural network
3. **训练器创建 / Trainer Creation**：配置训练参数 / Configure training parameters
4. **训练循环 / Training Loop**：前向传播、损失计算、反向传播、参数更新 / Forward, loss, backward, parameter update
5. **验证 / Validation**：在验证集上评估性能 / Evaluate on validation set
6. **模型保存 / Model Saving**：保存最佳模型 / Save best model
7. **测试 / Testing**：在测试集上评估最终性能 / Evaluate final performance on test set

### 动态架构训练
SELF-LNN 支持在训练过程中动态调整网络结构，无需重启训练：

| 功能 | 说明 |
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

## AGI 机器人功能
### 机器人控制
- **运动学**：DH 参数运动学正逆解
- **动力学**：刚体动力学、碰撞检测
- **路径规划**：A*、RRT*、多目标路径优化
- **硬件接口**：
  - ROS 1/2 集成
  - 串口通信
  - 多种硬件协议支持
  - 硬件资源管理器

### 可控制机器人种类
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

### 硬件设备支持
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

### 仿真环境
- **PyBullet 集成**：物理仿真、关节控制、传感器模拟
- **Gazebo 集成**：高保真仿真、插件系统
- **MuJoCo 集成**：高性能物理引擎，接触动力学仿真
- **自定义仿真引擎**：支持用户自定义仿真后端
- **传感器仿真**：摄像头、激光雷达、IMU、触觉传感器
- **多机器人协调**：多机器人协同控制、群智能算法

### 机器人功能特性
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

### 机器人操作模式
1. **直接控制模式**：直接发送控制指令到硬件
2. **ROS 控制模式**：通过 ROS/ROS2 进行控制
3. **仿真模式**：在 PyBullet/Gazebo 中仿真
4. **混合模式**：仿真与硬件结合的半实物仿真

---

### 机器人模仿学习
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

---

### 机器人训练系统
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

---


---

## 常见问题
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

---

> **文档更新**: 2026-06-08
> **开源仓库**: https://github.com/Sum-Outman/Self-LNN
> **开发者邮箱**: silencecrowtom@qq.com
