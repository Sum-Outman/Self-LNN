# WSL 网络故障排除指南
# WSL Network Troubleshooting Guide

> **版本 (Version):** 1.0.0 | **适用环境 (Environment):** WSL 2, Ubuntu 22.04/24.04
> **核心系统 (Core System):** SELF-LNN AGI (CfC Liquid Neural Network)
> **相关文档:** [Deployment_Ubuntu.md](Deployment_Ubuntu.md), [Offline_Deployment_Guide.md](Offline_Deployment_Guide.md)

---

## 问题现象 / Problem Symptoms

从WSL启动日志中常见的网络故障现象：

```
CheckConnection: resolving the name www.msftconnecttest.com [AF_INET]
CheckConnection: connecting to 23.202.34.233
CheckConnection: result.Ipv4Status = ConnCheckStatus::FailureSocketConnect (timeout)
[    7.273575] WSL (275) ERROR: CheckConnection: getaddrinfo() failed: -5
```

### 常见原因 / Common Causes

| 现象 (Symptom) | 可能原因 (Possible Cause) |
|---|---|
| `getaddrinfo() failed: -5` | DNS解析失败 (DNS resolution failure) |
| `timeout` | 网络连接超时 (Connection timeout) |
| WSL自动关机 (Auto shutdown) | 网络初始化失败导致WSL退出 (WSL exits on network init failure) |

---

## 一、快速修复 / Quick Fixes

### 方案1: 重启WSL网络 / Restart WSL Network

在 **Windows PowerShell（管理员权限/Admin）** 中执行：

```powershell
# 1. 关闭所有WSL实例 (Shutdown all WSL instances)
wsl --shutdown

# 2. 重置网络配置 (Reset network configuration)
netsh winsock reset
netsh int ip reset all

# 3. 刷新DNS (Flush DNS)
ipconfig /flushdns

# 4. 重启NAT服务 (Restart NAT service)
net stop winnat
net start winnat

# 5. 重新启动WSL (Restart WSL)
wsl
```

### 方案2: 修改WSL配置 / Modify WSL Configuration

创建或编辑 `%USERPROFILE%\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=false
autoProxy=true
```

保存后重启 (Save and restart):

```powershell
wsl --shutdown
wsl
```

### 方案3: 使用离线部署 / Offline Deployment

如果网络问题无法解决，WSL内仍可离线运行SELF-LNN：

```bash
# WSL内项目已在本地，无需网络
# Project is already local in WSL, no network needed
cd /path/to/self-Z
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
./bin/selflnn --help
```

> SELF-LNN **无外部依赖**，所有C代码离线即可编译运行。
> SELF-LNN has **zero external dependencies**, all C code compiles and runs offline.

---

## 二、详细诊断 / Detailed Diagnostics

### 2.1 WSL网络诊断 / WSL Network Diagnostics

```bash
# 在WSL中测试基本连接 (Test basic connectivity)
ping -c 4 8.8.8.8

# 测试DNS解析 (Test DNS resolution)
nslookup google.com

# 测试HTTP连接 (Test HTTP connection)
curl -I http://www.msftconnecttest.com

# 查看网络接口 (View network interfaces)
ip addr show
ip route show
```

### 2.2 检查WSL配置 / Check WSL Configuration

```bash
# 查看WSL版本 (Check WSL version)
wsl --version

# 查看WSL状态 (Check WSL status)
wsl --status

# 查看DNS配置 (View DNS config)
cat /etc/resolv.conf
cat /etc/hosts
```

### 2.3 修复DNS配置 / Fix DNS Configuration

```bash
# 备份原配置 (Backup original config)
sudo cp /etc/resolv.conf /etc/resolv.conf.backup

# 使用公共DNS (Use public DNS)
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf

# 防止WSL覆盖配置 (Prevent WSL from overwriting)
sudo chattr +i /etc/resolv.conf

# 测试DNS (Test DNS)
ping -c 4 google.com
```

### 2.4 检查Windows防火墙 / Check Windows Firewall

在 **Windows PowerShell（管理员）** 中：

```powershell
# 检查防火墙规则 (Check firewall rules)
Get-NetFirewallRule | Where-Object {$_.Enabled -eq 'True'} | Format-Table Name,Enabled

# 为WSL添加防火墙例外 (Add WSL firewall exception)
New-NetFirewallRule -DisplayName "WSL" -Direction Inbound -InterfaceAlias "vEthernet (WSL)" -Action Allow
```

---

## 三、SELF-LNN 在WSL中的离线部署 / Offline Deployment in WSL

由于SELF-LNN **无任何外部依赖**，WSL网络故障不影响正常运行：

### 3.1 直接编译运行 / Build and Run Directly

```bash
# 项目已在WSL中 (Project already in WSL)
cd ~/self-Z

# 编译 (Build)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# 运行 (Run)
./bin/selflnn --config ../config/backend_config.json.example

# 服务将在 http://localhost:8080 启动
# 从Windows浏览器直接访问
# Service starts at http://localhost:8080, accessible from Windows browser
```

### 3.2 从Windows拷贝项目 / Copy Project from Windows

如果WSL中没有项目文件：

```bash
# WSL中访问Windows文件系统 (Access Windows filesystem from WSL)
# Windows路径 /mnt/c/... 对应 C:\...
cp -r /mnt/c/Users/YourName/self-Z ~/

# 然后编译运行 (Then build and run)
cd ~/self-Z
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### 3.3 验证服务 / Verify Service

```bash
# 服务启动后，从Windows浏览器访问:
# After service starts, access from Windows browser:
#   http://localhost:8080/
#   http://localhost:8080/health

# 或从WSL内验证 (Or verify from within WSL):
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/status
```

---

## 四、紧急恢复方案 / Emergency Recovery

### 情况1: WSL完全无法启动 / WSL Cannot Start

```powershell
# Windows PowerShell（管理员/Admin）:

# 重置WSL (Reset WSL)
wsl --unregister Ubuntu
wsl --install -d Ubuntu

# 从备份恢复 (Restore from backup)
# wsl --import Ubuntu "C:\WSL-New" "C:\Backup\wsl-backup.tar"
```

### 情况2: 网络完全中断 / Complete Network Outage

```bash
# SELF-LNN完全可以离线运行
# SELF-LNN runs fully offline

# 直接使用本地回环地址 (Use localhost directly)
./build/bin/selflnn --port 8080

# 前端可以通过文件系统直接打开:
# Frontend can be opened directly via filesystem:
#   file:///home/user/self-Z/frontend/index.html
```

### 情况3: 系统资源不足 / Insufficient Resources

```bash
# 限制构建并行度 (Limit build parallelism)
cmake --build . --parallel 2  # 只用2个核心 (Use only 2 cores)

# 限制SELF-LNN资源使用 (Limit SELF-LNN resource usage)
# 在config中配置 (Configure in config):
# backend_config.json 中的相关设置
```

---

## 五、预防措施 / Preventive Measures

### 5.1 稳定WSL配置 / Stable WSL Configuration

创建 `%USERPROFILE%\.wslconfig`：

```ini
[wsl2]
memory=8GB
processors=4
localhostForwarding=true
dnsTunneling=true

[experimental]
autoMemoryReclaim=gradual
networkingMode=mirrored
sparseVhd=true
```

### 5.2 网络检查脚本 / Network Check Script

```bash
# 创建网络检查脚本 (Create network check script)
cat > ~/check_network.sh << 'EOF'
#!/bin/bash
echo "=== Network Status Check ==="
date
echo "1. Internet connectivity:"
ping -c 2 8.8.8.8 > /dev/null 2>&1 && echo "  OK" || echo "  FAILED"
echo "2. DNS resolution:"
nslookup google.com > /dev/null 2>&1 && echo "  OK" || echo "  FAILED"
echo "3. Network interface:"
ip addr show eth0 2>/dev/null | grep inet || echo "  No eth0"
echo "=== Check Complete ==="
EOF
chmod +x ~/check_network.sh
```

### 5.3 WSL备份 / WSL Backup

```powershell
# Windows PowerShell（管理员/Admin）:

# 导出WSL备份 (Export WSL backup)
wsl --export Ubuntu "C:\WSL-Backup\ubuntu-backup.tar"

# 从备份恢复 (Restore from backup)
wsl --import Ubuntu "C:\WSL-New" "C:\WSL-Backup\ubuntu-backup.tar"
```

---

## 六、监控和日志 / Monitoring and Logs

### 6.1 SELF-LNN 服务监控 / Service Monitoring

```bash
# 查看SELF-LNN标准输出 (View stdout)
./build/bin/selflnn --config config/backend_config.json.example

# 检查端口监听 (Check port listening)
ss -tulpn | grep 8080
# 或 (or)
netstat -tulpn | grep 8080
```

### 6.2 系统监控 / System Monitoring

```bash
# 查看系统日志 (View system logs)
sudo dmesg | tail -50

# 查看资源使用 (View resource usage)
free -h
df -h
top -bn1 | head -20
```

### 6.3 WSL日志 / WSL Logs

```powershell
# Windows PowerShell中查看WSL事件日志:
# View WSL event logs in Windows PowerShell
Get-WinEvent -LogName Microsoft-Windows-WSL/Operational | Select-Object -First 20
```

---

## 七、相关资源 / Related Resources

### 官方文档 / Official Documentation

- [WSL文档 (Microsoft)](https://docs.microsoft.com/zh-cn/windows/wsl/)
- [WSL网络配置](https://docs.microsoft.com/zh-cn/windows/wsl/networking)
- [WSL故障排除](https://docs.microsoft.com/zh-cn/windows/wsl/troubleshooting)

### 诊断工具 / Diagnostic Tools

| 工具 (Tool) | 用途 (Purpose) |
|---|---|
| `ping` | 网络连通性测试 (Connectivity test) |
| `nslookup` / `dig` | DNS查询 (DNS query) |
| `curl` | HTTP测试 (HTTP test) |
| `ss` / `netstat` | 端口状态 (Port status) |
| `ip` | 网络接口配置 (Network interface) |

### 内置诊断脚本 / Built-in Diagnostic Script

```bash
# 运行网络诊断 (Run network diagnosis)
# scripts/network_diagnosis.sh （如果存在）
# 或使用项目验证脚本 (Or use project verification):
python3 scripts/verify_project.py
```

---

> **最后更新 (Last Updated):** 2026-05-04
> **适用环境 (Environment):** WSL 2, Ubuntu 22.04/24.04
> **SELF-LNN 外部依赖:** 无 (None) — 完全离线可运行 (Fully offline capable)
> **相关文档:** [Deployment_Ubuntu.md](Deployment_Ubuntu.md), [Offline_Deployment_Guide.md](Offline_Deployment_Guide.md)
