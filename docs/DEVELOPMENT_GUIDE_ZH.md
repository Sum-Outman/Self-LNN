# 开发指南 / Development Guide

---

## 目录 / Table of Contents
- [概述 / Overview](#概述--overview)
- [项目结构 / Project Structure](#项目结构--project-structure)
- [环境配置 / Environment Setup](#环境配置--environment-setup)
- [编译构建 / Building](#编译构建--building)
- [代码规范 / Code Style](#代码规范--code-style)
- [核心模块开发 / Core Module Development](#核心模块开发--core-module-development)
- [测试 / Testing](#测试--testing)
- [调试 / Debugging](#调试--debugging)

---

## 概述 / Overview

### 中文
SELF-LNN 是一个纯C11实现的全模态AGI系统，基于CfC（Closed-form Continuous-time）液态神经网络。项目无任何第三方库依赖，使用CMake构建系统。

### English
SELF-LNN is a pure-C11 full-modal AGI system based on CfC (Closed-form Continuous-time) Liquid Neural Networks. The project has zero third-party dependencies and uses the CMake build system.

---

## 项目结构 / Project Structure

### 中文
```
self-Z/
├── CMakeLists.txt              # 主构建文件（CMake 3.10+，C11标准）
├── config/                     # 配置文件模板
│   ├── backend_config.json.example
│   ├── slnn.service.example    # Linux systemd服务模板
│   ├── slnn_windows_service.xml.example
│   ├── com.selflnn.slnn.plist.example
│   └── version.h.in
├── scripts/                    # 构建/部署/测试脚本
│   ├── build.bat / build.sh          # 构建脚本
│   ├── run_tests.bat / run_tests.sh  # 测试运行脚本
│   ├── check_code_quality.bat        # 代码质量检查
│   ├── performance_analysis.bat      # 性能分析
│   └── ...
├── include/selflnn/           # 公共头文件
│   ├── core/                  # 核心模块（CfC单元、ODE求解器等）
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
│   └── utils/                 # 工具函数
├── src/                       # 源代码（与include结构对应）
│   ├── main.c                 # AGI主入口
│   ├── core/cfc_cell.c        # CfC单元核心实现（4377行）
│   ├── backend/backend.c      # HTTP服务器（678KB）
│   ├── gpu/gpu.c              # GPU调度层
│   └── ...
├── frontend/                  # Web前端
│   ├── index.html             # 主控制台（SPA单页面应用）
│   ├── css/
│   └── js/
│       ├── api-service.js     # API服务
│       └── ... (共19个JS文件，含Worker)
│       └── main.js
├── tests/                     # 测试代码
│   ├── core/test_core.c       # 核心测试（22项）
│   ├── core/test_cfc_enhanced.c
│   ├── core/test_utils.c
│   ├── core/test_loss.c
│   ├── core/test_memory.c
│   ├── knowledge/
│   ├── learning/
│   ├── reasoning/
│   ├── multimodal/
│   └── robot/
├── docs/                      # 文档
└── build/                     # 构建输出（自动生成）
```

### English
```
self-Z/
├── CMakeLists.txt              # Build file (CMake 3.10+, C11 standard)
├── config/                     # Configuration templates
├── scripts/                    # Build/deploy/test scripts
├── include/selflnn/           # Public headers (19 module dirs)
├── src/                       # Source code (mirrors include structure)
│   ├── main.c                 # AGI entry point
│   ├── core/cfc_cell.c        # CfC cell core (4377 lines)
│   ├── backend/backend.c      # HTTP server (678KB)
│   ├── gpu/gpu.c              # GPU dispatch layer
│   └── ... (19 module dirs)
├── frontend/                  # Web frontend (14 pages)
├── tests/                     # Test code (26 test executables)
├── docs/                      # Documentation
└── build/                     # Build output (auto-generated)
```

### 模块说明 / Module Description

| 模块 / Module | 路径 / Path | 功能 / Function |
|--------------|------------|----------------|
| core | `src/core/` | CfC单元、ODE求解器（10种）、拉普拉斯分析、LNN核心 |
| multimodal | `src/multimodal/` | 视觉/音频/文本/传感器多模态融合 |
| cognition | `src/cognition/` | 自我认知、深度反思、元认知 |
| reasoning | `src/reasoning/` | 因果推理、规划、数学物理推理 |
| knowledge | `src/knowledge/` | 知识图谱、语义网络、本体工程 |
| memory | `src/memory/` | 工作/短期/长期/ episodic 记忆 |
| learning | `src/learning/` | 在线/强化/模仿/元学习 |
| evolution | `src/evolution/` | 进化引擎、神经架构搜索 |
| training | `src/training/` | 训练流水线、分布式训练、模型管理 |
| robot | `src/robot/` | 机器人控制、运动学、ROS集成 |
| gpu | `src/gpu/` | 10种GPU后端接口（CUDA/OpenCL/Vulkan/Metal/ROCm完整实现；Intel/Ascend/Cambricon/TPU需厂商SDK，无SDK通过dlsym+CPU回退）|
| backend | `src/backend/` | HTTP服务器、WebSocket、API端点 |
| concurrency | `src/concurrency/` | 线程池、无锁结构、RCU |
| safety | `src/safety/` | 安全监控、紧急停止、审计日志 |
| distributed | `src/distributed/` | 负载均衡、PBFT共识 |
| utils | `src/utils/` | 数学/内存/字符串/日志工具 |

---

## 环境配置 / Environment Setup

### 中文
#### 必需工具
- **编译器**：MSVC 2019+（Windows）、GCC 9+（Linux）、Clang 12+（macOS）
- **构建工具**：CMake 3.10+
- **Git**（可选，用于版本管理）

#### Windows 环境
```powershell
# 安装 Visual Studio 2019+（含 C/C++ 开发工具）
# 安装 CMake（从 cmake.org 下载）
# 打开 "Developer Command Prompt for VS" 或使用 PowerShell
```

#### Linux 环境
```bash
sudo apt update
sudo apt install build-essential cmake
```

#### macOS 环境
```bash
xcode-select --install
brew install cmake
```

### English
#### Required Tools
- **Compiler**: MSVC 2019+ (Windows), GCC 9+ (Linux), Clang 12+ (macOS)
- **Build Tool**: CMake 3.10+
- **Git** (optional)

---

## 编译构建 / Building

### 中文
#### 使用构建脚本（推荐）
```bash
# Windows
scripts\build.bat

# Linux/macOS
chmod +x scripts/build.sh
./scripts/build.sh
```

#### 手动构建
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

#### 构建选项
| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TESTS` | ON | 构建测试 |
| `ENABLE_GPU` | OFF | 启用GPU支持 |
| `CMAKE_BUILD_TYPE` | Release | Debug/Release |

### English
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

---

## 代码规范 / Code Style

### 中文
#### C语言规范
- **标准**：C11（`CMAKE_C_STANDARD 11`）
- **命名**：函数/变量 `snake_case`，类型 `PascalCase`，常量/宏 `UPPER_SNAKE_CASE`
- **缩进**：4个空格，禁止Tab
- **行宽**：不超过100字符
- **括号风格**：Allman风格（大括号单独一行）
- **编译警告**：MSVC `/W4 /WX`，GCC/Clang `-Wall -Wextra -Werror -pedantic`

#### 文件头注释
```c
/**
 * @file filename.c
 * @brief 文件功能说明
 */
```

#### 内存管理
- 每个`malloc`对应一个`free`
- 优先使用`safe_malloc`/`safe_calloc`等安全函数
- 分配后立即检查NULL
- 使用`goto cleanup`模式处理错误路径

### English
- **Standard**: C11
- **Naming**: `snake_case` functions/variables, `PascalCase` types, `UPPER_SNAKE_CASE` macros
- **Indentation**: 4 spaces, no tabs
- **Line width**: 100 chars max
- **Brace style**: Allman
- **Warnings**: `-Wall -Wextra -Werror` (GCC/Clang), `/W4 /WX` (MSVC)

---

## 核心模块开发 / Core Module Development

### 中文
#### 创建新模块步骤
1. 在 `include/selflnn/<module>/` 创建头文件
2. 在 `src/<module>/` 创建实现文件
3. 在 `src/<module>/CMakeLists.txt` 添加构建目标
4. 在主 `CMakeLists.txt` 添加子目录引用
5. 在 `tests/` 创建对应测试

#### CfC单元开发示例
```c
#include "selflnn/core/cfc_cell.h"

// 配置并创建CfC单元
CfCCellConfig config = {
    .input_size = 64,
    .hidden_size = 128,
    .time_constant = 0.1f,
    .delta_t = 0.01f,
    .ode_solver_type = ODE_SOLVER_RK4
};

CfCCell* cell = cfc_cell_create(&config);

// 前向传播
float input[64];
float hidden[128];
cfc_cell_forward(cell, input, hidden);

// 反向传播训练
float gradient[128];
cfc_cell_backward(cell, gradient, NULL);

// 清理
cfc_cell_free(cell);
```

### English
#### New Module Steps
1. Create header in `include/selflnn/<module>/`
2. Create implementation in `src/<module>/`
3. Add build target in `src/<module>/CMakeLists.txt`
4. Add subdirectory reference in root `CMakeLists.txt`
5. Create tests in `tests/`

#### CfC Cell Example
```c
#include "selflnn/core/cfc_cell.h"

CfCCellConfig config = {
    .input_size = 64,
    .hidden_size = 128,
    .time_constant = 0.1f,
    .delta_t = 0.01f,
    .ode_solver_type = ODE_SOLVER_RK4
};

CfCCell* cell = cfc_cell_create(&config);

float input[64];
float hidden[128];
cfc_cell_forward(cell, input, hidden);

cfc_cell_free(cell);
```

---

## 测试 / Testing

### 中文
项目使用自定义测试框架（非第三方库），基于 `RUN_TEST` 宏。

#### 编写测试
```c
#include "selflnn/core/cfc_cell.h"
#include <stdio.h>
#include <string.h>

// 测试函数，返回0表示通过
static int test_cfc_forward(void) {
    CfCCellConfig config;
    memset(&config, 0, sizeof(config));
    config.input_size = 4;
    config.hidden_size = 8;
    config.time_constant = 0.1f;

    CfCCell* cell = cfc_cell_create(&config);
    if (!cell) return -1;

    float input[4] = {0.5f, -0.3f, 0.8f, 0.1f};
    float hidden[8] = {0};
    int ret = cfc_cell_forward(cell, input, hidden);

    cfc_cell_free(cell);
    return (ret == 0) ? 0 : -1;
}

// 使用RUN_TEST宏注册
// RUN_TEST(test_cfc_forward, "CfC前向传播测试");
```

#### 运行测试
```bash
# 运行所有测试
scripts\run_tests.bat    # Windows
./scripts/run_tests.sh   # Linux

# 运行特定测试
./scripts/run_tests.sh --run test_core

# 详细输出
./scripts/run_tests.sh --verbose
```

### English
The project uses a custom test framework (no third-party library) based on the `RUN_TEST` macro.

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test
./scripts/run_tests.sh --run test_core

# Verbose output
./scripts/run_tests.sh --verbose
```

**Test executables:**
- `test_core` - Core CfC cell, ODE solvers, Laplace (22 tests)
- `test_utils` - Utility functions
- `test_loss` - Loss functions
- `test_memory` - Memory module
- `test_knowledge` - Knowledge base
- `test_learning` - Learning module
- `test_reasoning` - Reasoning engine
- `test_multimodal` - Multimodal processing
- `test_robot` - Robot control
- `test_training` - Training pipeline

---

## 调试 / Debugging

### 中文
#### Windows (Visual Studio)
在 Visual Studio 中打开 `build/selflnn.sln`，设置 `main.c` 为启动项，按 F5 调试。

#### Linux (GDB)
使用调试模式构建，然后用GDB启动程序进行调试。

#### 日志系统
系统提供日志功能，日志级别：DEBUG < INFO < WARNING < ERROR。

### English
#### Windows (Visual Studio)
Open `build/selflnn.sln` in Visual Studio, set `main.c` as startup project, press F5 to debug.

#### Linux (GDB)
Build with debug mode, then start the program with GDB for debugging.

#### Logging System
The system provides logging functionality, log levels: DEBUG < INFO < WARNING < ERROR.

---

> **文档更新 / Updated:** 2026-05-14
> **相关文档 / Related Docs:** [架构图](./Architecture_Diagram.md) | [用户手册](./USER_GUIDE.md) | [部署指南](./DEPLOYMENT_GUIDE_ZH.md)
