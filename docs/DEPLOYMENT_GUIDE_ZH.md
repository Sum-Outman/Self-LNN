# 部署指南 / Deployment Guide

---

## 目录 / Table of Contents
- [概述 / Overview](#概述--overview)
- [系统要求 / System Requirements](#系统要求--system-requirements)
- [构建项目 / Building the Project](#构建项目--building-the-project)
- [部署到生产目录 / Deploy to Production Directory](#部署到生产目录--deploy-to-production-directory)
- [配置后端服务 / Configure Backend Service](#配置后端服务--configure-backend-service)
- [安装为系统服务 / Install as System Service](#安装为系统服务--install-as-system-service)
- [GPU加速配置 / GPU Acceleration Configuration](#gpu加速配置--gpu-acceleration-configuration)
- [验证部署 / Verify Deployment](#验证部署--verify-deployment)
- [故障排除 / Troubleshooting](#故障排除--troubleshooting)

---

## 概述 / Overview

### 中文
SELF-LNN 是一个纯C实现的全模态AGI系统，**无任何第三方库依赖**。部署只需编译生成单个可执行文件 `selflnn` 并配置前端文件即可运行。

### English
SELF-LNN is a pure-C full-modal AGI system with **zero third-party library dependencies**. Deployment requires only compiling the `selflnn` executable and serving the frontend files.

---

## 系统要求 / System Requirements

### 中文
| 组件 / Component | 最低要求 / Minimum | 推荐配置 / Recommended |
|-----------------|-------------------|----------------------|
| CPU | x86-64, SSE4.2 | x86-64, AVX2 |
| 内存 / RAM | 4 GB | 16+ GB |
| 存储 / Storage | 500 MB | 10+ GB |
| 编译器 / Compiler | MSVC 2019 / GCC 9 / Clang 12 | MSVC 2022 / GCC 13 / Clang 16 |
| 构建工具 / Build Tool | CMake 3.10 | CMake 3.20+ |
| GPU（可选） | CUDA 10.0+ / OpenCL 2.0+ | CUDA 12.0+ |
| 操作系统 / OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 24.04 / macOS 14 |

### English
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | x86-64, SSE4.2 | x86-64, AVX2 |
| RAM | 4 GB | 16+ GB |
| Storage | 500 MB | 10+ GB |
| Compiler | MSVC 2019 / GCC 9 / Clang 12 | MSVC 2022 / GCC 13 / Clang 16 |
| Build Tool | CMake 3.10 | CMake 3.20+ |
| GPU (optional) | CUDA 10.0+ / OpenCL 2.0+ | CUDA 12.0+ |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 24.04 / macOS 14 |

---

## 构建项目 / Building the Project

### 中文
项目使用 CMake 构建系统，支持 Windows、Linux 和 macOS。

#### Windows 构建
```powershell
# 使用构建脚本（推荐）
scripts\build.bat

# 或手动构建
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
```

#### Linux / macOS 构建
```bash
# 使用构建脚本（推荐）
chmod +x scripts/build.sh
./scripts/build.sh

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

# 跳过测试构建（加快速度）
cmake .. -DBUILD_TESTS=OFF

# 调试构建
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

**构建产物位置 / Build Output:**
- Windows: `build/bin/selflnn.exe`
- Linux/macOS: `build/bin/selflnn`

### English
The project uses the CMake build system, supporting Windows, Linux, and macOS.

```bash
# Using build script (recommended)
./scripts/build.sh

# Or manual build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

---

## 部署到生产目录 / Deploy to Production Directory

### 中文
项目使用 CMake install 功能将编译产物和前端文件部署到指定目录。

#### Windows
```powershell
# 先编译项目
scripts\build.bat

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
./scripts/build.sh

# 手动部署到 /opt/slnn
sudo mkdir -p /opt/slnn/{bin,frontend,config,data}
sudo cp build/bin/Release/selflnn /opt/slnn/bin/
sudo cp -r frontend/* /opt/slnn/frontend/
sudo cp config/*.json.example /opt/slnn/config/
```

#### 部署目录结构
```
/opt/slnn/
├── bin/                    # 可执行文件
│   └── selflnn            # AGI后端服务
├── frontend/              # 前端页面（SPA单页面应用）
│   ├── index.html         # 主控制台
│   ├── css/
│   └── js/
├── config/                # 配置文件
│   ├── backend_config.json.example
│   └── version.h.in
├── scripts/               # 运维脚本
└── data/                  # 数据目录（运行时创建）
```

### English
Deployment uses CMake build followed by manual file copy to the target directory.

```bash
# Deploy to /opt/slnn
sudo mkdir -p /opt/slnn/{bin,frontend,config,data}
sudo cp build/bin/Release/selflnn /opt/slnn/bin/
sudo cp -r frontend/* /opt/slnn/frontend/
sudo cp config/*.json.example /opt/slnn/config/
```

**Deployment directory layout:**
```
/opt/slnn/
├── bin/              # Executables
├── frontend/         # Web frontend
├── config/           # Configuration files
├── scripts/          # Maintenance scripts
└── data/             # Runtime data
```

---

## 配置后端服务 / Configure Backend Service

### 中文
使用 `config/backend_config.json.example` 作为模板创建配置文件：

```bash
cp config/backend_config.json.example config/backend_config.json
```

主要配置项：
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

### English
Use `config/backend_config.json.example` as a template:

```bash
cp config/backend_config.json.example config/backend_config.json
```

Key configuration fields:
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
        "learning": true
    }
}
```

---

## 安装为系统服务 / Install as System Service

### 中文
项目提供服务配置文件模板，位于 `config/` 目录：

| 文件 / File | 用途 / Purpose |
|------------|---------------|
| `slnn.service.example` | Linux systemd 服务 |
| `slnn_windows_service.xml.example` | Windows 服务配置 |
| `com.selflnn.slnn.plist.example` | macOS launchd 服务 |

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

### English
Service configuration templates are provided in `config/`:

| File | Purpose |
|------|---------|
| `slnn.service.example` | Linux systemd service |
| `slnn_windows_service.xml.example` | Windows service config |
| `com.selflnn.slnn.plist.example` | macOS launchd service |

---

## GPU加速配置 / GPU Acceleration Configuration

### 中文
SELF-LNN 支持10种GPU计算后端，无需安装额外深度学习框架。

#### 查询可用后端
```bash
./build/bin/selflnn --help
# GPU相关选项在帮助中列出
```

#### CMake 启用GPU支持
```bash
# 启用CUDA支持
cmake .. -DENABLE_GPU=ON

# 启用OpenCL支持
cmake .. -DENABLE_GPU=ON

# 自动检测可用后端
cmake .. -DENABLE_GPU=ON
```

**注意：** GPU后端通过动态库加载，无需GPU也能正常运行（自动使用CPU后端）。

### English
SELF-LNN supports 10 GPU compute backends without requiring additional deep learning frameworks.

```bash
# Enable CUDA support in CMake
cmake .. -DENABLE_GPU=ON
```

**Note:** GPU backends are loaded dynamically. The system runs normally without GPU (falls back to CPU automatically).

---

## 验证部署 / Verify Deployment

### 中文
部署完成后，通过以下步骤验证：

```bash
# 1. 检查可执行文件
./build/bin/Release/selflnn --help

# 2. 运行测试（tests/目录尚未建立，待完善）
./scripts/run_tests.sh

# 3. 启动服务
./build/bin/Release/selflnn --port 8080 &

# 4. 验证API
curl http://localhost:8080/api/status

# 5. 打开Web控制台
# 浏览器访问 http://localhost:8080/
```

### English
After deployment, verify with:

```bash
# 1. Check executable
./build/bin/Release/selflnn --help

# 2. Run tests (tests/ directory not yet established)
./scripts/run_tests.sh

# 3. Start service
./build/bin/Release/selflnn --port 8080 &

# 4. Verify API
curl http://localhost:8080/api/status

# 5. Open web console at http://localhost:8080/
```

---

## 故障排除 / Troubleshooting

### 中文
| 问题 / Issue | 原因 / Cause | 解决 / Solution |
|-------------|-------------|----------------|
| 编译失败 / Build fails | 缺少CMake或编译器 | 安装CMake 3.10+ 和 C11 编译器 |
| 端口被占用 / Port in use | 8080端口被占用 | `selflnn -p 8081` 使用其他端口 |
| GPU未检测到 / GPU not detected | 缺少GPU驱动 | 安装对应 GPU 驱动，或使用CPU模式 |
| 前端页面打不开 / Frontend not loading | 前端文件未部署 | 确认 `frontend/` 目录在可执行文件同目录下 |

### English
| Issue | Cause | Solution |
|-------|-------|----------|
| Build fails | Missing CMake or compiler | Install CMake 3.10+ and C11 compiler |
| Port in use | Port 8080 occupied | Use `selflnn -p 8081` |
| GPU not detected | Missing GPU driver | Install GPU driver or use CPU mode |
| Frontend not loading | Frontend files not deployed | Ensure `frontend/` directory is alongside the executable |
