# Ubuntu 系统恢复指南 / Ubuntu System Recovery Guide

本文档提供Ubuntu系统（包括WSL环境）遇到启动故障或运行异常时的恢复解决方案。
This document provides recovery solutions for Ubuntu systems (including WSL environments) experiencing boot failures or abnormal operation.

---

## 🔍 问题现象 / Problem Symptoms

您遇到的系统问题表现为：
The system issues you're experiencing appear as:

```
Ubuntu 24.04.2 LTS zsf console
zsf login: 
[  150.729326] EXT4-fs (sdb): shut down requested (2)
[  150.813375] EXT4-fs (sda): shut down requested (2)
[  150.847935] reboot: Power down
```

**问题特征** / **Problem Characteristics**：
1. 系统显示登录提示 `zsf login:`
2. 文件系统收到关机请求
3. 系统自动关机重启
4. 无法正常登录和使用系统

1. System displays login prompt `zsf login:`
2. Filesystems receive shutdown requests
3. System automatically shuts down and reboots
4. Unable to login and use system normally

## 📋 问题分析 / Problem Analysis

### 可能原因 / Possible Causes
1. **文件系统错误**：EXT4文件系统检测到严重错误
2. **系统服务崩溃**：关键系统服务启动失败
3. **资源耗尽**：内存、磁盘空间或inode耗尽
4. **内核问题**：内核模块加载失败
5. **网络配置错误**：网络服务导致系统挂起
6. **WSL特定问题**：WSL网络或配置问题

1. **Filesystem errors**: EXT4 filesystem detects severe errors
2. **System service crashes**: Critical system services fail to start
3. **Resource exhaustion**: Memory, disk space, or inodes exhausted
4. **Kernel issues**: Kernel module loading failures
5. **Network configuration errors**: Network services causing system hangs
6. **WSL-specific problems**: WSL network or configuration issues

### 紧急程度 / Urgency Level
- **高**：系统无法正常启动，需要立即修复
- **影响**：无法部署和运行SELF-LNN项目

- **High**: System cannot boot normally, requires immediate repair
- **Impact**: Unable to deploy and run SELF-LNN project

## 🚀 快速恢复方案 / Quick Recovery Solutions

### 方案A：WSL环境（最可能的情况）/ Solution A: WSL Environment (Most Likely Scenario)

#### 步骤1：使用Windows恢复工具 / Step 1: Use Windows Recovery Tools
```powershell
# 以管理员身份运行PowerShell / Run PowerShell as Administrator

# 1. 完全关闭WSL / Completely shutdown WSL
wsl --shutdown

# 2. 重置WSL网络 / Reset WSL network
netsh winsock reset
netsh int ip reset all
ipconfig /release
ipconfig /renew
ipconfig /flushdns

# 3. 重启网络服务 / Restart network services
net stop winnat
net start winnat

# 4. 以root身份启动WSL / Start WSL as root
wsl --user root
```

#### 步骤2：运行自动恢复脚本 / Step 2: Run Automated Recovery Script
```powershell
# 在Windows中运行恢复工具 / Run recovery tool in Windows
scripts\wsl_recovery.bat

# 选择选项：2（修复网络连接）或 3（重置WSL实例） / Select option: 2 (Fix network connection) or 3 (Reset WSL instance)
```

#### 步骤3：手动修复（如果需要）/ Step 3: Manual Repair (If Needed)
```bash
# 在WSL中以root身份执行 / Execute as root in WSL

# 1. 检查文件系统 / Check filesystem
fsck -f /dev/sda1
fsck -f /dev/sdb1

# 2. 修复DNS配置 / Fix DNS configuration
echo "nameserver 8.8.8.8" > /etc/resolv.conf
echo "nameserver 8.8.4.4" >> /etc/resolv.conf

# 3. 修复WSL配置 / Fix WSL configuration
cat > /etc/wsl.conf << 'EOF'
[network]
generateResolvConf = false
hostname = zsf

[boot]
systemd = true

[interop]
appendWindowsPath = true
EOF

# 4. 重启WSL / Restart WSL
exit
# 返回Windows，然后： / Return to Windows, then:
wsl --shutdown
wsl
```

### 方案B：物理机/虚拟机环境 / Solution B: Physical Machine/Virtual Machine Environment

#### 步骤1：进入恢复模式 / Step 1: Enter Recovery Mode
1. 重启系统 / Reboot system
2. 在GRUB菜单选择 **Advanced options for Ubuntu**
3. 选择 **Recovery mode**
4. 选择 **root** 进入命令行 / Select **root** to enter command line

#### 步骤2：运行恢复命令 / Step 2: Run Recovery Commands
```bash
# 在恢复模式下执行 / Execute in recovery mode

# 1. 挂载文件系统为读写 / Mount filesystem as read-write
mount -o remount,rw /

# 2. 修复文件系统 / Repair filesystem
fsck -f /dev/sda1
fsck -f /dev/sdb1

# 3. 检查磁盘空间 / Check disk space
df -h
df -i

# 4. 修复GRUB引导 / Repair GRUB bootloader
update-grub
grub-install /dev/sda

# 5. 重启系统 / Reboot system
reboot
```

## 🔧 详细恢复步骤 / Detailed Recovery Steps

### 阶段1：诊断问题 / Phase 1: Diagnose Problems

#### 1.1 使用诊断工具 / Use Diagnostic Tools
```bash
# 如果系统能启动到命令行 / If system can boot to command line
sudo ./scripts/ubuntu_recovery.sh -d

# 或使用网络诊断 / Or use network diagnosis
./scripts/network_diagnosis.sh -a
```

#### 1.2 检查关键日志 / Check Critical Logs
```bash
# 查看系统日志 / Check system logs
sudo dmesg | tail -100
sudo journalctl -xb -p err

# 查看启动日志 / Check boot logs
sudo journalctl -b -1  # 上一次启动 / Last boot

# 查看特定服务日志 / Check specific service logs
sudo systemctl status systemd-networkd
sudo systemctl status NetworkManager
```

#### 1.3 检查资源使用 / Check Resource Usage
```bash
# 磁盘空间 / Disk space
df -h
df -i

# 内存使用 / Memory usage
free -h

# 进程资源 / Process resources
top -b -n 1 | head -20
```

### 阶段2：修复文件系统 / Phase 2: Repair Filesystem

#### 2.1 检查文件系统错误 / Check Filesystem Errors
```bash
# 列出所有文件系统 / List all filesystems
lsblk -f

# 检查文件系统错误 / Check filesystem errors
sudo fsck -n /dev/sda1
sudo fsck -n /dev/sdb1

# 修复错误（如果发现）/ Repair errors (if found)
sudo umount /dev/sda1  # 先卸载 / Unmount first
sudo fsck -f -y /dev/sda1
sudo mount /dev/sda1 /mnt
```

#### 2.2 清理磁盘空间 / Clean Disk Space
```bash
# 清理日志文件 / Clean log files
sudo journalctl --vacuum-size=100M
sudo rm -rf /var/log/*.gz

# 清理apt缓存 / Clean apt cache
sudo apt clean
sudo apt autoclean

# 清理临时文件 / Clean temporary files
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*

# 清理旧内核 / Clean old kernels
sudo apt autoremove --purge
```

### 阶段3：修复网络问题 / Phase 3: Repair Network Issues

#### 3.1 重置网络配置 / Reset Network Configuration
```bash
# 停止网络管理器 / Stop network manager
sudo systemctl stop NetworkManager
sudo systemctl stop systemd-networkd

# 清除网络配置 / Clear network configuration
sudo rm -f /etc/netplan/*.yaml
sudo rm -f /etc/network/interfaces.d/*
sudo cp /etc/network/interfaces /etc/network/interfaces.bak

# 恢复默认配置 / Restore default configuration
cat > /etc/network/interfaces << 'EOF'
auto lo
iface lo inet loopback

auto eth0
iface eth0 inet dhcp
EOF

# 重启网络 / Restart network
sudo systemctl start systemd-networkd
sudo systemctl enable systemd-networkd
```

#### 3.2 修复DNS配置 / Fix DNS Configuration
```bash
# 恢复DNS配置 / Restore DNS configuration
sudo rm -f /etc/resolv.conf
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf
sudo chattr +i /etc/resolv.conf
```

#### 3.3 修复hosts文件 / Fix Hosts File
```bash
# 确保localhost解析正常 / Ensure localhost resolution
echo "127.0.0.1 localhost" | sudo tee -a /etc/hosts
echo "127.0.1.1 $(hostname)" | sudo tee -a /etc/hosts
```

### 阶段4：修复系统服务 / Phase 4: Repair System Services

#### 4.1 检查关键服务状态 / Check Critical Service Status
```bash
# 列出所有失败的服务 / List all failed services
sudo systemctl --failed

# 检查关键服务 / Check critical services
sudo systemctl status ssh
sudo systemctl status cron
sudo systemctl status systemd-journald
```

#### 4.2 修复损坏的软件包 / Repair Broken Packages
```bash
# 修复dpkg状态 / Fix dpkg state
sudo dpkg --configure -a

# 修复依赖关系 / Fix dependencies
sudo apt-get install -f -y

# 更新软件包数据库 / Update package database
sudo apt-get update
sudo apt-get upgrade -y

# 清理和自动修复 / Clean and auto-fix
sudo apt-get autoremove -y
sudo apt-get autoclean
```

### 阶段5：验证恢复 / Phase 5: Verify Recovery

#### 5.1 系统功能验证 / System Function Verification
```bash
# 检查磁盘 / Check disk
df -h
df -i

# 检查内存 / Check memory
free -h

# 检查系统正常运行时间 / Check system uptime
uptime

# 检查内核日志 / Check kernel logs
sudo dmesg | tail -20
```

#### 5.2 网络连通性验证 / Network Connectivity Verification
```bash
# 测试本地回环 / Test local loopback
ping -c 4 127.0.0.1

# 测试外网连通（如果有） / Test external connectivity (if available)
ping -c 4 8.8.8.8

# 测试DNS解析 / Test DNS resolution
nslookup google.com
```

#### 5.3 服务验证 / Service Verification
```bash
# 确认所有关键服务正常运行 / Confirm all critical services running
sudo systemctl is-active ssh
sudo systemctl is-active systemd-networkd
sudo systemctl is-active cron
```

---

## 🛡️ 预防措施 / Prevention Measures

### 定期维护 / Regular Maintenance
```bash
# 每周检查磁盘空间 / Weekly disk space check
df -h | grep -v tmpfs

# 每周清理日志 / Weekly log cleanup
sudo journalctl --vacuum-time=7d

# 每月清理旧内核 / Monthly old kernel cleanup
sudo apt autoremove --purge

# 定期备份关键配置 / Regular critical config backup
sudo tar -czf /backup/system-config-$(date +%Y%m%d).tar.gz /etc/
```

### 配置自动恢复 / Configure Automatic Recovery
```bash
# 编辑GRUB配置启用故障保护模式 / Edit GRUB config to enable failsafe mode
sudo sed -i 's/GRUB_TIMEOUT=.*/GRUB_TIMEOUT=10/' /etc/default/grub
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash panic=5"/' /etc/default/grub
sudo update-grub
```

---

> **最后更新 (Last Updated):** 2026-05-04
> **适用环境 (Environment):** Ubuntu 22.04/24.04, WSL 2
> **相关文档 (Related Docs):** [Deployment_Ubuntu.md](Deployment_Ubuntu.md), [WSL_Network_Troubleshooting.md](WSL_Network_Troubleshooting.md)