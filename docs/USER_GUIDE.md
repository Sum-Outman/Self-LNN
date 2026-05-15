# SELF-LNN AGI系统用户手册
# SELF-LNN AGI System User Guide

> **版本 (Version):** 1.4.0 | **核心模型 (Core):** CfC液态神经网络 (Liquid Neural Network)
> **构建系统 (Build):** CMake 3.10+ | **默认端口 (Default Port):** 8080

---

## 目录 (Table of Contents)

1. [系统概述 / System Overview](#1-系统概述-system-overview)
2. [系统要求 / System Requirements](#2-系统要求-system-requirements)
3. [快速开始 / Quick Start](#3-快速开始-quick-start)
4. [详细构建指南 / Build Guide](#4-详细构建指南-build-guide)
5. [系统配置 / System Configuration](#5-系统配置-system-configuration)
6. [API使用指南 / API Usage Guide](#6-api使用指南-api-usage-guide)
7. [故障排除 / Troubleshooting](#7-故障排除-troubleshooting)
8. [部署指南 / Deployment Guide](#8-部署指南-deployment-guide)

---

## 1. 系统概述 / System Overview

SELF-LNN 是基于纯C语言实现的单一液态神经网络（CfC - Closed-form Continuous-time Liquid Neural Network）全模态AGI系统。

SELF-LNN is a full-modal AGI system based on a single CfC (Closed-form Continuous-time) Liquid Neural Network, implemented in pure C.

**核心特性 (Core Features):**
- **单一模型 (Single Model):** 全模态统一输入→统一状态演化→统一输出
- **纯C实现 (Pure C):** 无任何第三方库依赖，C11标准
- **连续时间动态 (Continuous Time):** 7种ODE求解器 + 内置Verlet/BDF2
- **GPU加速 (GPU Acceleration):** 10种后端运行时动态加载
- **全双工通信 (Full-Duplex):** HTTP REST + WebSocket 实时通信
- **自我演化 (Self-Evolution):** 在线学习、强化学习、模仿学习、元学习

## 2. 系统要求 / System Requirements

### 硬件要求 / Hardware Requirements
| 组件 Component | 最低要求 Minimum | 推荐要求 Recommended |
|---------------|-----------------|-------------------|
| CPU | x86-64, SSE4.2 | x86-64/ARM64, AVX2 |
| 内存 Memory | 4 GB | 16 GB+ |
| 存储 Storage | 500 MB (仅二进制) | 10 GB (含模型数据) |
| GPU (可选 Optional) | 任意OpenCL 2.0+ | CUDA 10.0+ / ROCm 5.0+ |

### 软件要求 / Software Requirements
| 操作系统 OS | 编译器 Compiler | 构建工具 Build |
|-----------|----------------|---------------|
| Windows 10/11 | MSVC 2019+ / Clang | CMake 3.10+ |
| Ubuntu 20.04+ | GCC 9.0+ / Clang 10.0+ | CMake 3.10+ |
| macOS 12+ | Clang 12.0+ | CMake 3.10+ |

**依赖 (Dependencies):** 无第三方库 (No third-party libraries)

## 3. 快速开始 / Quick Start

### 3.1 构建项目 / Build the Project

**Windows (PowerShell):**
```powershell
.\scripts\build.bat
```

**Linux/macOS:**
```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

### 3.2 运行测试 / Run Tests

**Windows:**
```powershell
.\scripts\run_tests.bat
```

**Linux/macOS:**
```bash
./scripts/run_tests.sh
```

### 3.3 启动AGI系统 / Start AGI System

```bash
# 直接运行二进制 (Run binary directly)
./build/src/Debug/selflnn
# 或 (Or)
./build/src/selflnn

# 使用快速启动脚本 (Use quick-start script)
.\scripts\quick_start.bat    # Windows
./scripts/quick_start.sh     # Linux/macOS
```

### 3.4 访问Web控制台 / Access Web Console

启动后，浏览器访问 (After startup, open browser at):
```
http://localhost:8080
```

## 4. 详细构建指南 / Build Guide

### 4.1 CMake构建选项 / CMake Options

| 选项 Option | 默认值 Default | 描述 Description |
|------------|---------------|-----------------|
| `BUILD_TESTS` | ON | 构建测试程序 Build tests |
| `BUILD_EXAMPLES` | ON | 构建示例程序 Build examples |
| `ENABLE_GPU` | OFF | 启用GPU加速 Enable GPU |
| `INSTALL_EXAMPLES` | OFF | 安装示例 Install examples |
| `BUILD_SHARED_LIBS` | OFF | 构建共享库 Build shared libs |
| `BUILD_FRONTEND` | ON | 构建前端 Build frontend |

### 4.2 自定义构建示例 / Custom Build Examples

```bash
# 启用GPU支持 (Enable GPU)
cmake -B build -DENABLE_GPU=ON

# Release构建 (Release build)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Debug构建 (Debug build)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# 仅核心模块 (Core only, no tests)
cmake -B build -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build build --parallel
```

### 4.3 启动选项 / Startup Options

```bash
./build/bin/selflnn --help
# 可用选项 (Available options):
#   --port <port>     服务端口 (默认: 8080)
#   --version         显示版本信息
#   --help            显示此帮助
```

## 5. 系统配置 / System Configuration

### 5.1 配置文件 / Config Files

配置文件位于项目根目录 `selflnn_config.json`:

All config files are in the `config/` directory:

| 文件 File | 用途 Purpose |
|----------|-------------|
| `backend_config.json.example` | 后端服务配置示例 Backend config example |
| `slnn.service.example` | Linux systemd服务配置 Linux systemd service |
| `slnn_windows_service.xml.example` | Windows服务配置 Windows service |
| `com.selflnn.slnn.plist.example` | macOS服务配置 macOS service |
| `version.h.in` | 版本模板 Version template |

### 5.2 后端配置示例 / Backend Config Example

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

### 5.3 CfC单元配置 / CfC Cell Config

CfC单元通过 `CfCCellConfig` 结构体配置:

The CfC cell is configured via the `CfCCellConfig` struct:

| 字段 Field | 类型 Type | 默认值 Default | 说明 Description |
|-----------|-----------|---------------|-----------------|
| `input_size` | `size_t` | 必需 Required | 输入向量大小 Input size |
| `hidden_size` | `size_t` | 必需 Required | 隐藏状态大小 Hidden size |
| `time_constant` | `float` | 1.0 | 时间常数 Time constant |
| `delta_t` | `float` | 1.0 | 时间步长 Time step (秒) |
| `noise_std` | `float` | 0.0 | 噪声标准差 Noise std |
| `ode_solver_type` | `int` | 0 (闭式解) | ODE求解器类型 ODE solver type |
| `enable_adaptation` | `int` | 0 | 启用参数自适应 Enable adaptation |
| `use_multi_timescale` | `int` | 0 | 多时间尺度 Multi-timescale |

## 6. API使用指南 / API Usage Guide

### 6.1 CfC核心API / Core CfC API

所有API在 `include/selflnn/core/cfc_cell.h` 中声明:

All APIs are declared in `include/selflnn/core/cfc_cell.h`:

#### 创建CfC单元 / Create CfC Cell

```c
#include "selflnn/core/cfc_cell.h"

// 配置 (Configuration)
CfCCellConfig config = {
    .input_size = 256,
    .hidden_size = 128,
    .time_constant = 1.0f,
    .delta_t = 1.0f,
    .ode_solver_type = ODE_SOLVER_CLOSED_FORM,  // 0=闭式解
    .noise_std = 0.0f,
    .enable_adaptation = 0
};

// 创建网络实例 (Create network instance)
CfCCell* network = cfc_cell_create(&config);
if (!network) {
    printf("创建CfC单元失败 (Failed to create CfC cell)\n");
    return -1;
}
```

#### 前向传播 / Forward Pass

```c
float input[256];   // 输入 (Input)
float output[128];  // 输出隐藏状态 (Output hidden state)

// 准备输入数据 (Prepare input data)
for (int i = 0; i < 256; i++) {
    input[i] = (float)rand() / RAND_MAX;
}

// 执行前向传播 (Execute forward pass)
int result = cfc_cell_forward(network, input, output);
if (result != 0) {
    printf("前向传播失败 (Forward pass failed)\n");
}

// 或指定时间步长 (Or specify time step)
result = cfc_cell_forward_with_dt(network, input, 0.5f, output);
```

#### 反向传播 / Backward Pass

```c
float gradient[128];    // 输出梯度 (Output gradient)
float input_grad[256];  // 输入梯度 (Input gradient)

// 准备梯度数据 (Prepare gradient data)
for (int i = 0; i < 128; i++) {
    gradient[i] = output[i] - target[i];  // 误差 Error
}

// 执行反向传播 (Execute backward pass)
int result = cfc_cell_backward(network, gradient, input_grad);
if (result == 0) {
    printf("反向传播完成 (Backward pass complete)\n");
}
```

#### 清理资源 / Cleanup

```c
cfc_cell_free(network);
```

### 6.2 ODE求解器选择 / ODE Solver Selection

通过 `ode_solver_type` 字段在 `CfCCellConfig` 中选择:

Select via the `ode_solver_type` field in `CfCCellConfig`:

| 枚举值 Enum | 名称 Name | 说明 Description |
|------------|----------|-----------------|
| 0 | `ODE_SOLVER_CLOSED_FORM` | 封闭形式解 (默认，最快) Closed-form (default, fastest) |
| 1 | `ODE_SOLVER_RK4` | 四阶龙格-库塔 RK4 fixed-step |
| 2 | `ODE_SOLVER_RK45` | 五阶自适应 RK45 adaptive |
| 3 | `ODE_SOLVER_CTBP` | 连续时间反向传播 CTBP |
| 4 | `ODE_SOLVER_DP54` | Dormand-Prince 5(4)自适应 DP54 adaptive |
| 5 | `ODE_SOLVER_ROSENBROCK` | Rosenbrock刚性求解器 Rosenbrock stiff |
| 6 | `ODE_SOLVER_SYMPLECTIC` | Forest-Ruth辛积分器 Symplectic |

使用特定求解器的前向传播 (Forward with specific solver):

```c
// RK45自适应 (RK45 adaptive)
int steps_used;
cfc_cell_forward_rk45(network, input, 0.1f, output, &steps_used);

// DP54自适应 (DP54 adaptive)
cfc_cell_forward_dp54(network, input, 0.1f, output, &steps_used);

// Rosenbrock刚性求解器 (Rosenbrock stiff)
cfc_cell_forward_rosenbrock(network, input, 0.1f, output, &steps_used);

// Forest-Ruth辛积分器 (Symplectic)
cfc_cell_forward_symplectic(network, input, 0.1f, output, &steps_used);
```

### 6.3 GPU加速 / GPU Acceleration

GPU支持通过运行时动态库加载实现，无需在构建时安装GPU框架:

GPU support uses dynamic library loading at runtime — no GPU frameworks needed at build time:

```c
#include "selflnn/gpu/gpu.h"

// 查询可用GPU后端 (Query available GPU backends)
int backend_count = gpu_get_available_backends();
printf("可用GPU后端数: %d (Available GPU backends)\n", backend_count);

// 获取后端信息 (Get backend info)
GpuBackendInfo info;
gpu_get_backend_info(GPU_BACKEND_CUDA, &info);
printf("后端: %s, 可用: %s\n", info.name, info.available ? "是" : "否");
```

支持的GPU后端 (Supported GPU backends): CPU(参考), CUDA, OpenCL, Vulkan, Metal, ROCm, Ascend(华为), Cambricon(寒武纪), TPU(Google), Intel

### 6.4 保存和加载模型 / Save & Load Model

```c
#include "selflnn/core/cfc_cell.h"

// 保存到文件 (Save to file)
FILE* fout = fopen("model.bin", "wb");
if (fout) {
    cfc_cell_save(network, fout);
    fclose(fout);
}

// 从文件加载 (Load from file)
FILE* fin = fopen("model.bin", "rb");
if (fin) {
    cfc_cell_load(network, fin);
    fclose(fin);
}
```

### 6.5 重置状态 / Reset State

```c
// 重置隐藏状态和内部缓存 (Reset hidden state and internal cache)
cfc_cell_reset(network);
```

## 7. 故障排除 / Troubleshooting

### 7.1 构建失败 / Build Failures

| 问题 Problem | 解决 Solution |
|-------------|--------------|
| CMake未找到 CMake not found | 安装CMake 3.10+ Install CMake 3.10+ |
| 编译器未找到 Compiler not found | 安装MSVC/GCC/Clang |
| 链接错误 Link errors | 清理后重新构建: `cmake --build build --clean-first` |

### 7.2 运行错误 / Runtime Errors

| 问题 Problem | 解决 Solution |
|-------------|--------------|
| 端口被占用 Port in use | 使用 `-p` 指定其他端口: `selflnn -p 8081` |
| GPU加速不工作 GPU not working | 检查驱动，程序自动回退到CPU Check driver, auto-fallback to CPU |
| 内存不足 Out of memory | 减小 `hidden_size` 或 `input_size` |

### 7.3 调试技巧 / Debug Tips

```bash
# 启用详细日志 (Enable verbose logging)
selflnn -l debug

# 使用Valgrind检查内存 (Linux/macOS)
valgrind --leak-check=full ./build/src/selflnn

# 使用Dr. Memory检查内存 (Windows)
drmemory.exe -- build\src\Debug\selflnn.exe
```

## 8. 部署指南 / Deployment Guide

### 8.1 系统服务部署 / Service Deployment

**Linux (systemd):**
```bash
sudo cp config/slnn.service.example /etc/systemd/system/slnn.service
sudo systemctl daemon-reload
sudo systemctl enable slnn
sudo systemctl start slnn
# 查看状态 (Check status)
sudo systemctl status slnn
# 查看日志 (View logs)
sudo journalctl -u slnn -f
```

**Windows (服务 Service):**
```powershell
.\scripts\deploy.bat --install-service
```

**macOS (launchd):**
```bash
cp config/com.selflnn.slnn.plist.example ~/Library/LaunchAgents/com.selflnn.slnn.plist
launchctl load ~/Library/LaunchAgents/com.selflnn.slnn.plist
```

### 8.2 部署脚本 / Deploy Scripts

```bash
# Windows
.\scripts\deploy.bat

# Linux/macOS
sudo ./scripts/deploy.sh -t /opt/slnn
# -t <target_dir>: 目标安装目录 Target directory
# -p <port>: 服务端口 Service port
# -s: 安装为系统服务 Install as system service
```

### 8.3 验证部署 / Verify Deployment

```bash
# 检查健康状态 (Health check)
curl http://localhost:8080/api/health

# 检查前端 (Check frontend)
curl http://localhost:8080/

# 查看进程 (Check process)
ps aux | grep selflnn    # Linux/macOS
tasklist | findstr selflnn  # Windows
```

### 8.4 安全注意事项 / Security Notes

1. **网络安全 (Network Security):** 默认监听 `0.0.0.0:8080`，生产环境建议使用防火墙限制访问 By default listens on `0.0.0.0:8080`, use firewall in production
2. **API认证 (API Auth):** 通过Web控制台生成API密钥 Use web console to generate API keys
3. **数据安全 (Data Security):** 定期备份 `config/` 和模型文件 Regularly backup config and model files

---

> **更新 (Updated):** 2026-05-14
> **版本 (Version):** SELF-LNN v1.4.0
> **相关文档 (Related Docs):** [架构图](./Architecture_Diagram.md) | [AGI机器人指南](./AGI_Robot_Guide.md) | [开发指南](./DEVELOPMENT_GUIDE_ZH.md)
