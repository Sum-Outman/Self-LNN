# 离线部署指南
# Offline Deployment Guide

> **版本 (Version):** 1.0.0 | **协议 (Protocol):** HTTP REST + WebSocket | **端口 (Port):** 8080
> **核心模型 (Core Model):** CfC (Closed-form Continuous-time Liquid Neural Network)
> **实现语言 (Language):** 纯C (Pure C, C11) | **构建系统 (Build):** CMake 3.10+
> **外部依赖 (Dependencies):** 无 (None) — 100% 自包含 (Self-contained)

---

## 适用场景 / Use Cases

- **无网络连接的环境** — 内网服务器、隔离区
- **网络受限环境** — 仅允许局域网通信
- **高安全要求环境** — 禁止外网连接
- **离线介质传输** — USB、光盘等

Environments without internet, restricted network, high-security zones, or offline media transfer.

---

## 一、准备工作 / Preparation

### 1.1 收集项目文件 / Collect Project Files

在联网机器上下载完整的项目源代码：

```bash
# 从开发机器直接拷贝项目目录
# 或在有网络环境下载完整源码包
# Copy the entire project directory from a development machine
# or download the complete source archive from a connected environment

# 完整项目包含 (Complete project includes):
# - src/                # C源代码 (C source code)
# - include/            # C头文件 (C headers)
# - frontend/           # 前端页面 (Frontend web pages)
# - config/             # 配置文件 (Configuration files)
# - scripts/            # 脚本工具 (Script utilities)
# - tests/              # 测试代码 (Test code)
# - CMakeLists.txt      # CMake构建配置 (CMake build config)
```

> **注意:** 本项目 **无任何外部依赖**，所有功能由纯C实现。不需要下载任何第三方库。
> **Note:** This project has **ZERO external dependencies**. All functionality is implemented in pure C. No third-party libraries required.

### 1.2 编译工具依赖 / Build Tool Dependencies

#### Linux (Ubuntu/Debian)

```bash
# 仅需要基础编译工具 (Only basic build tools needed)
sudo apt-get update
sudo apt-get install -y build-essential cmake

# 如果需要GPU支持（可选），安装对应驱动
# For GPU support (optional), install appropriate drivers
# See: docs/DEPLOYMENT_GUIDE_ZH.md
```

#### Windows

- **CMake**: https://cmake.org/download/ (>= 3.10)
- **Visual Studio Build Tools 2022**: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
  - 或者 (or) **MinGW-w64** (推荐/recommended)
- **Git** (可选/optional，用于版本管理): https://git-scm.com/download/win

#### macOS

```bash
# 使用Homebrew安装 (Install via Homebrew)
brew install cmake
# Xcode Command Line Tools 已包含gcc/clang
# Xcode Command Line Tools include gcc/clang
```

### 1.3 Python工具（可选）/ Python Utilities (Optional)

项目 `scripts/` 目录包含一些Python辅助脚本，但这些 **不是运行必需的**：

| 脚本 (Script) | 功能 (Function) | 必需? (Required?) |
|---|---|---|
| `verify_project.py` | 项目完整性验证 (Project integrity check) | 否 (No) |
| `python_env_check.py` | 环境检查 (Environment check) | 否 (No) |
| `fix_line_endings.py` | 修复行结束符 (Fix line endings) | 否 (No) |

如果需要使用这些脚本，离线传输Python解释器即可：

```bash
# 如果确实需要Python脚本，下载Python安装包
# If Python scripts are needed, download Python installer
# https://www.python.org/downloads/

# 注意：核心系统运行时不需要Python
# Note: Python is NOT required for core system runtime
```

### 1.4 创建离线部署包 / Create Offline Bundle

```bash
# 创建完整的离线部署包
# Create a complete offline deployment bundle
mkdir -p slnn-offline-bundle
cp -r /path/to/self-Z/* slnn-offline-bundle/

# 创建说明文档 (Create README)
cat > slnn-offline-bundle/README_OFFLINE.md << 'EOF'
# SELF-LNN 离线部署包 / Offline Deployment Bundle
# 完全自包含，无外部依赖
# Fully self-contained, zero external dependencies
# 部署步骤见 docs/离线部署指南.md
# See docs/离线部署指南.md for deployment steps
EOF

# 打包 (Package)
# Windows: 使用ZIP压缩 (Use ZIP compression)
# Linux/macOS:
tar -czf slnn-offline-$(date +%Y%m%d).tar.gz slnn-offline-bundle/
```

---

## 二、离线环境部署步骤 / Offline Deployment Steps

### 2.1 传输文件到目标环境 / Transfer Files

```bash
# 方法1: USB设备
# Method 1: USB device
sudo mount /dev/sdX1 /mnt/usb
cp slnn-offline-*.tar.gz /mnt/usb/

# 方法2: 局域网SCP
# Method 2: LAN SCP
scp slnn-offline-*.tar.gz user@192.168.1.100:/tmp/

# Windows: 使用共享文件夹或USB拷贝
# Windows: Use shared folder or USB copy
```

### 2.2 解压项目 / Extract Project

```bash
# Linux/macOS
mkdir -p /opt/slnn-deploy
cd /opt/slnn-deploy
tar -xzf /path/to/slnn-offline-*.tar.gz
cd slnn-offline-bundle

# Windows: 使用ZIP解压到目标目录
# Windows: Extract ZIP to target directory
```

### 2.3 安装编译依赖 / Install Build Dependencies

```bash
# Linux - 如果已离线，使用apt离线安装包
# Linux - if offline, use cached deb packages
# 在有网络环境预先下载 (Pre-download on connected machine):
#   apt-get download build-essential cmake gcc
# 在离线环境安装 (Install on offline machine):
sudo dpkg -i *.deb

# Windows - 直接运行已安装的Visual Studio或MinGW
# Windows - use pre-installed Visual Studio or MinGW
# macOS - Xcode Command Line Tools已预装 (pre-installed)
```

### 2.4 构建项目 / Build Project

```bash
# 创建构建目录 (Create build directory)
mkdir -p build
cd build

# 配置CMake (Configure CMake)
cmake .. -DCMAKE_BUILD_TYPE=Release

# 构建 (Build)
cmake --build . --parallel

# 返回项目根目录 (Return to project root)
cd ..

# 构建完成后，可执行文件在:
# After build, the executable is at:
#   build/bin/selflnn (Linux/macOS)
#   build\bin\selflnn.exe (Windows)
```

> **注意:** 使用 `cmake --build` 而非 `make`。CMake会自动选择系统上可用的构建工具（Make、Ninja、MSBuild等）。
> **Note:** Use `cmake --build` instead of `make`. CMake automatically selects the available build tool.

#### CMake可选编译选项 / CMake Optional Build Options

| 选项 (Option) | 默认值 (Default) | 说明 (Description) |
|---|---|---|
| `BUILD_TESTS=ON` | ON | 编译测试程序 (Build tests) |
| `BUILD_EXAMPLES=ON` | ON | 编译示例程序 (Build examples) |
| `ENABLE_GPU=OFF` | OFF | 启用GPU支持 (Enable GPU support) |
| `INSTALL_EXAMPLES=OFF` | OFF | 安装示例 (Install examples) |
| `BUILD_SHARED_LIBS=OFF` | OFF | 构建共享库 (Build shared libs) |
| `BUILD_FRONTEND=ON` | ON | 构建前端 (Build frontend) |

示例 — 启用GPU支持：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON
cmake --build . --parallel
```

### 2.5 部署到生产目录 / Deploy to Production Directory

```bash
# 使用部署脚本 (Use deploy script)
# Linux/macOS:
chmod +x scripts/deploy.sh
./scripts/deploy.sh

# 或手动部署 (Or manual deployment):
mkdir -p /opt/slnn
cp -r build/bin /opt/slnn/
cp -r config /opt/slnn/
cp -r frontend /opt/slnn/
cp -r scripts /opt/slnn/

# Windows: 拷贝到目标目录即可
# Windows: Copy to target directory
```

---

## 三、配置 / Configuration

### 3.1 后端配置 / Backend Configuration

配置文件位于 `config/backend_config.json.example`，拷贝并修改：

```bash
# Linux/macOS
cp config/backend_config.json.example config/backend_config.json
# 编辑配置文件 (Edit config file)
# vi config/backend_config.json
```

配置示例 (Sample config):

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080
  }
}
```

> **端口默认为 8080**，可在配置文件中修改。
> **Default port is 8080**, configurable in the config file.

### 3.2 服务自启动 / Service Auto-start

#### Linux (systemd)

```bash
# 使用提供的服务文件 (Use provided service file)
sudo cp config/slnn.service.example /etc/systemd/system/slnn.service
sudo systemctl daemon-reload
sudo systemctl enable slnn
sudo systemctl start slnn
```

#### macOS (launchd)

```bash
# 使用提供的plist文件 (Use provided plist file)
cp config/com.selflnn.slnn.plist.example ~/Library/LaunchAgents/com.selflnn.slnn.plist
launchctl load ~/Library/LaunchAgents/com.selflnn.slnn.plist
```

#### Windows (服务)

```bash
# 使用提供的XML配置 (Use provided XML config)
# config/slnn_windows_service.xml.example
# 使用sc命令创建服务 (Use sc to create service):
sc create slnn binPath="C:\slnn\bin\selflnn.exe"
```

---

## 四、验证部署 / Verify Deployment

### 4.1 系统验证 / System Verification

```bash
# 检查必要工具 (Check required tools)
cmake --version
gcc --version
# Python为可选，不必须
# Python is optional, not required
```

### 4.2 项目验证 / Project Verification

```bash
# 源代码统计 (Source code statistics)
# 约208个.c文件，194个.h文件
# ~208 .c files, 194 .h files
find . -name "*.c" | wc -l
find . -name "*.h" | wc -l

# 运行内置测试 (Run built-in tests)
./build/bin/test_core --all
./build/bin/test_backend --all
```

### 4.3 服务验证 / Service Verification

```bash
# 手动启动服务 (Start service manually)
./build/bin/selflnn --help
./build/bin/selflnn --config config/backend_config.json

# 验证HTTP服务 (Verify HTTP service)
# 服务启动后，在浏览器访问:
# After service starts, visit in browser:
#   http://localhost:8080/
#   http://localhost:8080/health

# 验证API (Verify API endpoints)
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/status
```

---

## 五、常见问题解决 / FAQ

### 问题1：缺少编译工具 / Missing Build Tools

```bash
# Linux - 安装build-essential和cmake
sudo apt-get install -y build-essential cmake

# Windows - 安装Visual Studio Build Tools或MinGW
# macOS - 安装Xcode Command Line Tools:
xcode-select --install
```

### 问题2：构建失败 / Build Failure

```bash
# 清理并重新构建 (Clean and rebuild)
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# 查看详细错误 (View detailed errors)
cmake --build . --verbose
```

### 问题3：权限问题 / Permission Issues

```bash
# 修复权限 (Fix permissions)
sudo chown -R $USER:$USER /opt/slnn
chmod +x /opt/slnn/bin/selflnn
chmod +x scripts/*.sh
```

### 问题4：端口冲突 / Port Conflict

```bash
# 检查端口占用 (Check port usage)
# Linux:
netstat -tulpn | grep :8080
# Windows:
netstat -ano | findstr :8080

# 修改端口 (Change port)
# 编辑 config/backend_config.json 修改 port 字段
# Edit port field in config/backend_config.json
```

### 问题5：Windows行结束符问题 / Windows Line Ending Issues

```bash
# 如果脚本无法在Linux运行，使用可选工具修复
# If scripts fail on Linux, use optional fix tool
python3 scripts/fix_line_endings.py
# 或 (or)
sed -i 's/\r$//' scripts/*.sh
```

---

## 六、部署清单 / Deployment Checklist

### 部署前检查 / Pre-deployment Check

- [ ] 项目文件完整 (Complete project files): `src/`, `include/`, `frontend/`, `config/`, `CMakeLists.txt`
- [ ] 编译工具已安装 (Build tools installed): `cmake >= 3.10`, `gcc/clang`
- [ ] 磁盘空间充足 (Sufficient disk space): >= 500MB

### 部署步骤 / Deployment Steps

- [ ] 解压项目到目标目录 (Extract project to target directory)
- [ ] 运行CMake配置 (Run CMake configure): `cmake .. -DCMAKE_BUILD_TYPE=Release`
- [ ] 构建项目 (Build project): `cmake --build . --parallel`
- [ ] 配置服务自启动 (Configure auto-start): `systemd` / `launchd` / `Windows Service`
- [ ] 验证所有测试通过 (Verify all tests pass): `./build/bin/test_core --all`

### 部署后验证 / Post-deployment Verification

- [ ] 服务启动正常 (Service starts successfully)
- [ ] 端口8080监听正常 (Port 8080 listening)
- [ ] 健康检查通过 (Health check passes): `curl http://localhost:8080/health`
- [ ] 前端页面可访问 (Frontend accessible): `http://localhost:8080/`
- [ ] API端点可响应 (API endpoints responding): `curl http://localhost:8080/api/v1/status`

---

## 附：离线包结构 / Appendix: Offline Bundle Structure

```
slnn-offline-bundle/
├── CMakeLists.txt              # CMake构建配置 (Build config)
├── src/                        # C源代码 (208 .c files)
├── include/                    # C头文件 (194 .h files)
├── frontend/                   # 前端页面 (14 HTML pages)
├── config/                     # 配置文件
│   ├── backend_config.json.example
│   ├── slnn.service.example
│   ├── slnn_windows_service.xml.example
│   └── com.selflnn.slnn.plist.example
├── scripts/                    # 脚本工具
│   ├── build.bat / build.sh   # 构建脚本
│   ├── deploy.sh               # 部署脚本
│   ├── run_tests.sh            # 测试脚本
│   ├── verify_project.py       # (可选/Optional)
│   ├── fix_line_endings.py     # (可选/Optional)
│   └── ...
├── tests/                      # 测试代码
├── docs/                       # 文档
└── README.md                   # 项目说明
```

---

> **最后更新 (Last Updated):** 2026-05-04
> **版本 (Version):** 1.0.0 (源码版本/Source version); CMake 0.1.0 (构建版本/Build version)
> **外部依赖 (External Dependencies):** 无 (None)
> **Python依赖 (Python Dependencies):** 无 (None) — Python脚本为可选工具 (Python scripts are optional utilities)
