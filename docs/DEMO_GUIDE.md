# 演示指南 / Demo Guide

---

## 目录 / Table of Contents
- [概述 / Overview](#概述--overview)
- [演示1：运行核心测试套件 / Demo 1: Run Core Test Suite](#演示1运行核心测试套件--demo-1-run-core-test-suite)
- [演示2：CfC液态神经网络基本使用 / Demo 2: CfC Liquid Neural Network Basics](#演示2cfc液态神经网络基本使用--demo-2-cfc-liquid-neural-network-basics)
- [演示3：启动AGI后端服务 / Demo 3: Start AGI Backend Service](#演示3启动agi后端服务--demo-3-start-agi-backend-service)
- [演示4：使用Web控制台 / Demo 4: Using the Web Console](#演示4使用web控制台--demo-4-using-the-web-console)
- [演示5：训练CfC模型 / Demo 5: Training a CfC Model](#演示5训练cfc模型--demo-5-training-a-cfc-model)
- [演示6：GPU加速推理 / Demo 6: GPU-Accelerated Inference](#演示6gpu加速推理--demo-6-gpu-accelerated-inference)
- [API参考摘要 / API Reference Summary](#api参考摘要--api-reference-summary)

---

## 概述 / Overview

### 中文
SELF-LNN 是一个基于封闭形式连续时间（CfC）液态神经网络的纯C全模态AGI系统。本指南提供基于真实代码的演示教程，演示无需外部依赖。

### English
SELF-LNN is a pure-C full-modal AGI system based on Closed-form Continuous-time (CfC) Liquid Neural Networks. This guide provides demo tutorials based on actual code — no external dependencies required.

---

## 演示1：运行核心测试套件 / Demo 1: Run Core Test Suite

### 中文
> **注意：** tests/ 目录尚未建立，核心测试套件待实现。当前可通过编译项目并运行主程序进行手动功能验证。

```powershell
# 编译项目
scripts\build.bat      # Windows
./scripts/build.sh     # Linux/macOS

# 运行主程序
./build/bin/Release/selflnn        # Linux/macOS
.\build\bin\Release\selflnn.exe    # Windows
```

### English
> **Note:** The tests/ directory is not yet established. Manual functional verification can be done by compiling and running the main executable.

---

## 演示2：CfC液态神经网络基本使用 / Demo 2: CfC Liquid Neural Network Basics

### 中文
以下代码展示CfC单元的核心使用模式，代码基于实际 API（`cfc_cell.h`）。

```c
#include "selflnn/core/cfc_cell.h"
#include <stdio.h>
#include <string.h>

int main() {
    // 1. 配置CfC单元
    CfCCellConfig config;
    memset(&config, 0, sizeof(config));
    config.input_size = 4;       // 输入维度
    config.hidden_size = 16;     // 隐藏状态维度
    config.time_constant = 0.1f; // 时间常数
    config.delta_t = 0.01f;      // ODE求解步长
    config.noise_std = 0.001f;   // 噪声标准差

    // 2. 创建CfC单元
    CfCCell* cell = cfc_cell_create(&config);
    if (!cell) {
        printf("创建CfC单元失败\n");
        return -1;
    }

    // 3. 前向传播
    float input[4] = {0.5f, -0.3f, 0.8f, 0.1f};
    float hidden[16];
    memset(hidden, 0, sizeof(hidden));

    for (int t = 0; t < 10; t++) {
        cfc_cell_forward(cell, input, hidden);
        printf("时间步 %d: hidden[0]=%.4f, hidden[1]=%.4f\n",
               t, hidden[0], hidden[1]);
    }

    // 4. 清理
    cfc_cell_free(cell);
    return 0;
}
```

**构建与运行：**
```bash
# 将上述代码保存为 demo_basic.c
gcc -Iinclude -Isrc demo_basic.c src/core/cfc_cell.c \
    src/utils/math_utils.c src/utils/memory_utils.c \
    -lm -o demo_basic
./demo_basic
```

### English
The code below demonstrates the core usage pattern of the CfC cell, based on the actual API (`cfc_cell.h`).

```c
#include "selflnn/core/cfc_cell.h"
#include <stdio.h>
#include <string.h>

int main() {
    // 1. Configure CfC cell
    CfCCellConfig config;
    memset(&config, 0, sizeof(config));
    config.input_size = 4;
    config.hidden_size = 16;
    config.time_constant = 0.1f;
    config.delta_t = 0.01f;
    config.noise_std = 0.001f;

    // 2. Create CfC cell
    CfCCell* cell = cfc_cell_create(&config);
    if (!cell) {
        printf("Failed to create CfC cell\n");
        return -1;
    }

    // 3. Forward propagation
    float input[4] = {0.5f, -0.3f, 0.8f, 0.1f};
    float hidden[16];
    memset(hidden, 0, sizeof(hidden));

    for (int t = 0; t < 10; t++) {
        cfc_cell_forward(cell, input, hidden);
        printf("Step %d: hidden[0]=%.4f, hidden[1]=%.4f\n",
               t, hidden[0], hidden[1]);
    }

    // 4. Cleanup
    cfc_cell_free(cell);
    return 0;
}
```

**Build and run:** compile with `gcc` linking core modules, or use the project's CMake build system.

---

## 演示3：启动AGI后端服务 / Demo 3: Start AGI Backend Service

### 中文
AGI后端服务是系统的核心HTTP服务器，提供REST API接口。

```bash
# 构建项目
./scripts/build.sh

# 直接运行编译后的服务
./build/bin/selflnn --help
```

**启动选项：**
```
用法: selflnn [选项]
选项:
  --port <端口>          HTTP服务端口 (默认: 8080)
  --version             显示版本信息
  --help                显示此帮助
```

**启动服务：**
```bash
# 先编译项目
./scripts/build.sh      # Linux/macOS
.\scripts\build.bat     # Windows

# 默认端口8080启动
./build/bin/Release/selflnn         # Linux/macOS
.\build\bin\Release\selflnn.exe     # Windows
```

**验证服务运行：**
```bash
curl http://localhost:8080/api/status
# 预期返回: {"status":"running","version":"1.0.0","uptime":1234}
```

### English
The AGI backend service is the core HTTP server providing the REST API.

```bash
# Build the project
./scripts/build.sh

# Run the compiled service
./build/bin/selflnn --help
```

**Startup options:**
```
Usage: selflnn [Options]
  --port <port>    HTTP port (default: 8080)
  --version        Show version
  --help           Show this help
```

**Start the service:**
```bash
# Build first
./scripts/build.sh      # Linux/macOS

# Start with default port 8080
./build/bin/Release/selflnn         # Linux/macOS
```

**Verify service:**
```bash
curl http://localhost:8080/api/status
# Expected: {"status":"running","version":"1.0.0","uptime":1234}
```

---

## 演示4：使用Web控制台 / Demo 4: Using the Web Console

### 中文
启动后端后，在浏览器打开前端控制台：

```
http://localhost:8080/
```

系统提供以下功能页面：
- **主控制台** (`/`): AGI系统总控面板，含系统状态、对话、推理等模块
- **训练中心** (`/training-center`): 模型训练管理与监控
- **模拟控制** (`/simulation-control`): 环境模拟与测试
- **教学系统** (`/teach`): 交互式教学界面
- **语音控制** (`/voice-control`): 语音交互控制
- **使用日志** (`/usage-logs`): 系统使用记录

### English
After starting the backend, open the web console in your browser:

```
http://localhost:8080/
```

Available pages:
- **Main Console** (`/`): AGI system master control panel
- **Training Center** (`/training-center`): Model training management
- **Simulation Control** (`/simulation-control`): Environment simulation
- **Teaching System** (`/teach`): Interactive teaching interface
- **Voice Control** (`/voice-control`): Voice interaction
- **Usage Logs** (`/usage-logs`): System usage records

---

## 演示5：训练CfC模型 / Demo 5: Training a CfC Model

### 中文
训练基于实际 API 调用 `cfc_cell_backward` 进行梯度更新。

```c
#include "selflnn/core/cfc_cell.h"
#include "selflnn/core/optimizers.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

int main() {
    // 配置
    CfCCellConfig config;
    memset(&config, 0, sizeof(config));
    config.input_size = 2;
    config.hidden_size = 8;
    config.output_size = 1;
    config.time_constant = 0.1f;
    config.delta_t = 0.01f;

    CfCCell* model = cfc_cell_create(&config);

    // 训练数据: XOR问题
    float inputs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float targets[4] = {0, 1, 1, 0};

    for (int epoch = 0; epoch < 100; epoch++) {
        float total_loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            float hidden[8] = {0};
            cfc_cell_forward(model, inputs[i], hidden);

            float pred = hidden[0];
            float loss = (pred - targets[i]) * (pred - targets[i]);
            total_loss += loss;

            float gradient[8] = {0};
            gradient[0] = 2.0f * (pred - targets[i]);
            cfc_cell_backward(model, gradient, NULL);
        }
        printf("Epoch %d: loss = %.6f\n", epoch, total_loss / 4.0f);
    }

    cfc_cell_free(model);
    return 0;
}
```

### English
Training uses the actual `cfc_cell_backward` API for gradient computation.

Full training pipeline with dataset loading, mini-batch processing, and loss monitoring is available via the project's training module (`src/training/`).

---

## 演示6：GPU加速推理 / Demo 6: GPU-Accelerated Inference

### 中文
SELF-LNN 支持10种计算后端，通过 `gpu模块` 自动调度。

```c
#include "selflnn/gpu/gpu.h"

// 查询可用GPU后端
int backends[GPU_BACKEND_COUNT];
int count = gpu_get_available_backends(backends, GPU_BACKEND_COUNT);

for (int i = 0; i < count; i++) {
    GpuBackendInfo info;
    gpu_get_backend_info(backends[i], &info);
    printf("后端 %d: %s (%s)\n", i, info.name, info.version);
}
```

**支持的GPU后端 / Supported GPU Backends:**
| 后端 / Backend | 类型 / Type |
|---------------|------------|
| CPU | 通用CPU / General CPU |
| CUDA | NVIDIA GPU |
| OpenCL | 跨平台GPU / Cross-platform GPU |
| Vulkan | 跨平台GPU / Cross-platform GPU |
| Metal | Apple GPU |
| ROCm | AMD GPU |
| Ascend | 华为昇腾 / Huawei Ascend |
| Cambricon | 寒武纪 / Cambricon |
| TPU | Google TPU |
| Intel | Intel GPU |

### English
SELF-LNN supports 10 compute backends, dispatched automatically through the `gpu` module.

---

## API参考摘要 / API Reference Summary

### 中文
核心CfC API（完整参考见 [API_REFERENCE_ZH.md](API_REFERENCE_ZH.md)）：

| 函数 / Function | 描述 / Description |
|----------------|-------------------|
| `cfc_cell_create()` | 创建CfC单元实例 / Create CfC cell instance |
| `cfc_cell_forward()` | 前向传播 / Forward propagation |
| `cfc_cell_backward()` | 反向传播（训练）/ Backward propagation (training) |
| `cfc_cell_reset()` | 重置单元状态 / Reset cell state |
| `cfc_cell_save()` | 保存模型参数 / Save model parameters |
| `cfc_cell_load()` | 加载模型参数 / Load model parameters |
| `cfc_cell_free()` | 释放CfC单元 / Free CfC cell |

### English
Core CfC API (full reference at [API_REFERENCE_ZH.md](API_REFERENCE_ZH.md)):

| Function | Description |
|----------|-------------|
| `cfc_cell_create()` | Create CfC cell instance |
| `cfc_cell_forward()` | Forward propagation |
| `cfc_cell_backward()` | Backward propagation (training) |
| `cfc_cell_reset()` | Reset cell state |
| `cfc_cell_save()` | Save model parameters |
| `cfc_cell_load()` | Load model parameters |
| `cfc_cell_free()` | Free CfC cell |
