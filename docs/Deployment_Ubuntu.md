# Ubuntu 部署指南
# Ubuntu Deployment Guide

> **适用版本 (Applicable Version):** Ubuntu 22.04 LTS / 24.04 LTS
> **SELF-LNN版本 (Version):** 1.0.0 | **架构 (Arch):** x86-64 / ARM64

---

## 目录 (Table of Contents)

1. [系统要求 / System Requirements](#1-系统要求-system-requirements)
2. [快速开始 / Quick Start](#2-快速开始-quick-start)
3. [手动构建 / Manual Build](#3-手动构建-manual-build)
4. [生产部署 / Production Deployment](#4-生产部署-production-deployment)
5. [服务管理 / Service Management](#5-服务管理-service-management)
6. [GPU支持 / GPU Support](#6-gpu支持-gpu-support)
7. [验证部署 / Verify Deployment](#7-验证部署-verify-deployment)
8. [故障排除 / Troubleshooting](#8-故障排除-troubleshooting)
9. [性能优化 / Performance Tuning](#9-性能优化-performance-tuning)

---

## 1. 系统要求 / System Requirements

### 硬件要求 / Hardware Requirements
| 组件 Component | 最低 Minimum | 推荐 Recommended |
|---------------|-------------|-----------------|
| CPU | 2核, x86-64 | 8核+, x86-64/ARM64 |
| 内存 Memory | 4 GB | 16 GB+ |
| 存储 Storage | 500 MB | 10 GB+ |
| GPU (可选) | — | NVIDIA CUDA 11.0+ / AMD ROCm 5.0+ |

### 软件要求 / Software Requirements
| 软件 Software | 版本 Version | 用途 Purpose |
|--------------|-------------|-------------|
| GCC / Clang | GCC 9.0+ / Clang 10.0+ | C编译器 C Compiler |
| CMake | 3.10+ | 构建系统 Build System |
| Make / Ninja | 任意 Any | 构建工具 Build Tool |

**核心项目无外部依赖 (Core project has zero external dependencies).**

#### 可选Python工具 (Optional Python Utilities)
项目提供以下Python辅助脚本 (The project includes these optional Python helper scripts):
- `scripts/verify_project.py` — 项目完整性验证 Project integrity verification
- `scripts/python_env_check.py` — Python环境检查 (如需使用Python脚本) Python env check
- `scripts/fix_line_endings.py` — 行结束符修复 Line ending fix (跨平台传输) (cross-platform transfer)
- `scripts/network_diagnosis.sh` — 网络诊断 Network diagnosis (Shell, 无需Python)

这些脚本需要 Python 3.8+，但 **不是核心项目的运行依赖**。
These scripts require Python 3.8+, but are **not required to build or run the core project**.

## 2. 快速开始 / Quick Start

### 2.1 安装编译工具 / Install Build Tools

```bash
sudo apt update
sudo apt install -y build-essential cmake
# 可选: 使用Ninja加速 (Optional: Ninja for faster builds)
sudo apt install -y ninja-build
```

### 2.2 构建项目 / Build the Project

```bash
# 进入项目目录 (Enter project directory)
cd /path/to/self-lnn

# 创建构建目录并配置 (Create build dir and configure)
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cd ..

# 构建 (Build)
cmake --build build --parallel $(nproc)
```

### 2.3 运行测试 / Run Tests

```bash
# 方法1: 使用CMake的ctest
cd build && ctest --output-on-failure && cd ..

# 方法2: 使用测试脚本
./scripts/run_tests.sh

# 方法3: 直接运行测试二进制
./build/tests/test_core
```

### 2.4 启动服务 / Start the Service

```bash
# 直接运行 (Run directly)
./build/src/selflnn

# 指定端口 (Specify port)
./build/src/selflnn -p 8080

# 后台运行 (Run in background)
nohup ./build/src/selflnn > slnn.log 2>&1 &
```

启动后访问 (After startup, access): `http://localhost:8080`

## 3. 手动构建 / Manual Build

### 3.1 CMake配置选项 / CMake Configuration Options

```bash
# 标准Release构建 (Standard Release build)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Debug构建 (Debug build)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# 启用GPU (Enable GPU)
cmake -B build -DENABLE_GPU=ON
cmake --build build --parallel

# 最小构建 (Minimal build, no tests/examples)
cmake -B build -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build build --parallel

# 使用Ninja (Use Ninja generator)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 3.2 可用构建选项 / Available Build Options

| 选项 Option | 默认值 Default | 描述 Description |
|------------|---------------|-----------------|
| `BUILD_TESTS` | ON | 构建测试程序 Build tests |
| `BUILD_EXAMPLES` | ON | 构建示例程序 Build examples |
| `ENABLE_GPU` | OFF | 启用GPU加速 Enable GPU |
| `BUILD_SHARED_LIBS` | OFF | 构建共享库 Build shared libraries |
| `BUILD_FRONTEND` | ON | 构建前端 Build frontend |

### 3.3 清理构建 / Clean Build

```bash
# 完全清理 (Full clean)
rm -rf build
mkdir build && cd build && cmake .. && cd ..

# 增量清理 (Incremental clean)
cmake --build build --clean-first
```

## 4. 生产部署 / Production Deployment

### 4.1 使用部署脚本 / Using Deploy Script

```bash
# 部署到 /opt/slnn (Deploy to /opt/slnn)
sudo ./scripts/deploy.sh -t /opt/slnn

# 部署并指定端口 (Deploy with custom port)
sudo ./scripts/deploy.sh -t /opt/slnn -p 8080

# 部署并安装为系统服务 (Deploy and install as system service)
sudo ./scripts/deploy.sh -t /opt/slnn -s
```

### 4.2 手动部署 / Manual Deploy

```bash
# 创建目标目录 (Create target directory)
sudo mkdir -p /opt/slnn/{bin,config,frontend,logs,data}

# 复制二进制 (Copy binary)
sudo cp build/src/selflnn /opt/slnn/bin/

# 复制配置 (Copy config)
sudo cp config/backend_config.json.example /opt/slnn/config/backend_config.json
sudo cp config/slnn.service.example /opt/slnn/config/

# 复制前端 (Copy frontend)
sudo cp -r frontend/* /opt/slnn/frontend/

# 创建日志目录 (Create log directory)
sudo mkdir -p /opt/slnn/logs
```

### 4.3 配置文件 / Configuration

编辑 `/opt/slnn/config/backend_config.json`:

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

## 5. 服务管理 / Service Management

### 5.1 安装systemd服务 / Install systemd Service

```bash
# 复制服务配置 (Copy service config)
sudo cp config/slnn.service.example /etc/systemd/system/slnn.service

# 修改配置以匹配安装路径 (Edit to match install path)
sudo nano /etc/systemd/system/slnn.service

# 重新加载配置 (Reload config)
sudo systemctl daemon-reload

# 启用开机自启 (Enable auto-start)
sudo systemctl enable slnn

# 启动服务 (Start service)
sudo systemctl start slnn
```

### 5.2 服务管理命令 / Service Management Commands

```bash
# 状态检查 (Status check)
sudo systemctl status slnn

# 启动 (Start)
sudo systemctl start slnn

# 停止 (Stop)
sudo systemctl stop slnn

# 重启 (Restart)
sudo systemctl restart slnn

# 查看日志 (View logs)
sudo journalctl -u slnn -f
```

## 6. GPU支持 / GPU Support

SELF-LNN使用运行时动态库加载技术，构建时无需安装GPU框架。

SELF-LNN uses runtime dynamic library loading — no GPU frameworks needed at build time.

### 6.1 启用GPU构建 / Enable GPU Build

```bash
cmake -B build -DENABLE_GPU=ON
cmake --build build --parallel
```

### 6.2 NVIDIA CUDA支持 / NVIDIA CUDA Support

如果系统安装了CUDA运行时库，程序启动时自动加载。

If CUDA runtime libraries are installed, the program will auto-detect them at startup.

```bash
# 安装CUDA运行时 (仅运行时，无需SDK)
# Install CUDA runtime only (no SDK needed)
sudo apt install -y nvidia-cuda-toolkit
# 或从NVIDIA官网安装 (Or install from NVIDIA official site)
```

### 6.3 AMD ROCm支持 / AMD ROCm Support

```bash
# 安装ROCm运行时 (Install ROCm runtime)
# 参考: https://rocm.docs.amd.com/
```

### 6.4 Intel GPU 支持 / Intel GPU Support

Intel GPU通过Level Zero API加速。系统检测到Intel GPU驱动后自动启用。

```bash
# 安装Intel GPU驱动和运行时
sudo apt install -y intel-opencl-icd intel-level-zero-gpu
```

### 6.5 华为昇腾(Ascend)支持 / Huawei Ascend Support

需要安装CANN（Ascend Computing）软件包，系统通过dlsym动态加载libascendcl.so。

```bash
# 需从华为官网获取CANN软件包
# https://www.hiascend.com/software/cann
# 无SDK时自动回退CPU计算
```

### 6.6 寒武纪(Cambricon)支持 / Cambricon Support

需要安装Cambricon Neuware SDK，系统通过dlsym动态加载libcnrt.so。

```bash
# 需从寒武纪官网获取Neuware SDK
# https://www.cambricon.com/
# 无SDK时自动回退CPU计算
```

### 6.7 Google TPU支持 / Google TPU Support

需要安装libtpu运行时库，系统通过dlsym动态加载。

```bash
# 需从Google获取libtpu
# 无SDK时自动回退CPU计算
```

### 6.8 验证GPU可用性 / Verify GPU Availability

启动后通过API检查 (Check via API after startup):

```bash
curl http://localhost:8080/api/v1/gpu/status
```

或查看启动日志中的GPU信息 (Or check startup logs for GPU info):

```bash
./build/src/selflnn -l debug 2>&1 | grep -i gpu
```

## 7. 验证部署 / Verify Deployment

### 7.1 基本验证 / Basic Verification

```bash
# 检查服务运行 (Check service is running)
curl http://localhost:8080/api/health

# 检查前端 (Check frontend)
curl -s http://localhost:8080/ | head -20

# 检查进程 (Check process)
ps aux | grep selflnn
```

### 7.2 完整验证 / Full Verification

```bash
# 使用验证脚本 (Use verification script)
./scripts/test_deployment.sh -a

# 运行测试套件 (Run test suite)
cd build && ctest && cd ..
```

### 7.3 监控端点 / Monitoring Endpoints

| 端点 Endpoint | 说明 Description |
|--------------|-----------------|
| `GET /api/health` | 健康检查 Health check |
| `GET /api/v1/system/status` | 系统状态 System status |
| `GET /api/v1/system/metrics` | 性能指标 Performance metrics |

## 8. 故障排除 / Troubleshooting

### 8.1 构建问题 / Build Issues

| 问题 Problem | 解决 Solution |
|-------------|--------------|
| `cmake: command not found` | `sudo apt install -y cmake` |
| `gcc: command not found` | `sudo apt install -y build-essential` |
| `链接错误 Link errors` | `cmake --build build --clean-first` |
| `找不到头文件 Header not found` | 确保在项目根目录运行cmake Run cmake from project root |

### 8.2 运行时问题 / Runtime Issues

| 问题 Problem | 解决 Solution |
|-------------|--------------|
| `端口8080被占用 Port 8080 in use` | `selflnn -p 8081` 或其他端口 Or another port |
| `Address already in use` | `sudo lsof -ti:8080 \| xargs kill -9` |
| `段错误 Segmentation fault` | 使用Debug模式重建获取详细错误 Rebuild in Debug mode |

### 8.3 WSL网络问题 / WSL Network Issues

参见 (See): [WSL_Network_Troubleshooting.md](WSL_Network_Troubleshooting.md)

## 9. 性能优化 / Performance Tuning

### 9.1 系统参数 / System Parameters

```bash
# 调整文件描述符限制 (Adjust file descriptor limit)
echo "fs.file-max = 100000" | sudo tee -a /etc/sysctl.conf

# 调整网络参数 (Adjust network params)
echo "net.core.somaxconn = 1024" | sudo tee -a /etc/sysctl.conf
echo "vm.swappiness = 10" | sudo tee -a /etc/sysctl.conf

# 生效 (Apply)
sudo sysctl -p
```

### 9.2 服务资源限制 / Service Resource Limits

编辑 `/etc/systemd/system/slnn.service`，在 `[Service]` 部分添加:

```ini
LimitNOFILE=65536
LimitNPROC=infinity
LimitCORE=infinity
```

### 9.3 构建优化 / Build Optimization

```bash
# 使用LTO (Use Link-Time Optimization)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

# 使用Native架构优化 (Use native architecture optimization)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-march=native -mtune=native"
```

---

> **更新 (Updated):** 2026-05-04
> **版本 (Version):** SELF-LNN v1.0.0
> **相关文档 (Related Docs):** [部署指南](DEPLOYMENT_GUIDE_ZH.md), [离线部署指南](Offline_Deployment_Guide.md), [WSL网络故障排除](WSL_Network_Troubleshooting.md)
