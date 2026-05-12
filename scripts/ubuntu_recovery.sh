#!/bin/bash

# SELF-LNN Ubuntu系统恢复工具
# 用于解决Ubuntu系统启动后立即关机的问题

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "需要root权限运行此脚本"
        log_info "请使用: sudo $0"
        exit 1
    fi
}

# 检查系统环境
check_environment() {
    log_info "检查系统环境..."
    
    # 检查是否为WSL
    if grep -qi "Microsoft" /proc/version 2>/dev/null; then
        log_info "检测到WSL环境"
        IS_WSL=true
    else
        log_info "检测到原生Linux环境"
        IS_WSL=false
    fi
    
    # 检查系统版本
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log_info "系统: $NAME $VERSION"
    else
        log_warning "无法确定系统版本"
    fi
    
    # 检查内核版本
    log_info "内核: $(uname -r)"
}

# 诊断系统问题
diagnose_system() {
    log_info "开始系统诊断..."
    echo ""
    
    # 1. 检查磁盘空间
    log_info "1. 检查磁盘空间:"
    df -h
    echo ""
    
    # 2. 检查内存使用
    log_info "2. 检查内存使用:"
    free -h
    echo ""
    
    # 3. 检查系统日志
    log_info "3. 检查系统日志:"
    dmesg | tail -50
    echo ""
    
    # 4. 检查失败的systemd服务
    log_info "4. 检查失败的systemd服务:"
    systemctl list-units --failed 2>/dev/null || log_warning "systemd不可用"
    echo ""
    
    # 5. 检查网络连接
    log_info "5. 检查网络连接:"
    ip addr show
    echo ""
    ip route show
    echo ""
    
    # 6. 检查文件系统错误
    log_info "6. 检查文件系统状态:"
    mount | grep -E "^/dev/"
    echo ""
}

# 修复文件系统错误
fix_filesystem() {
    log_info "修复文件系统错误..."
    
    # 检查需要修复的文件系统
    local fs_list=$(mount | grep -E "^/dev/" | awk '{print $1}' | sort -u)
    
    for fs in $fs_list; do
        log_info "检查文件系统: $fs"
        
        # 检查是否需要修复
        if fsck -n $fs 2>/dev/null | grep -q "needs repair"; then
            log_warning "文件系统 $fs 需要修复"
            
            # 卸载文件系统（如果可能）
            local mount_point=$(mount | grep "^$fs " | awk '{print $3}')
            if [ "$mount_point" != "/" ]; then
                log_info "卸载 $fs (挂载点: $mount_point)"
                umount $mount_point 2>/dev/null || log_warning "无法卸载 $mount_point"
                
                # 修复文件系统
                log_info "修复 $fs"
                fsck -f -y $fs
                
                # 重新挂载
                log_info "重新挂载 $mount_point"
                mount $mount_point
            else
                log_warning "无法修复根文件系统，需要重启到恢复模式"
            fi
        else
            log_success "文件系统 $fs 正常"
        fi
    done
}

# 修复网络问题
fix_network() {
    log_info "修复网络问题..."
    
    if [ "$IS_WSL" = true ]; then
        log_info "WSL环境网络修复..."
        
        # 1. 修复DNS配置
        log_info "1. 修复DNS配置"
        echo "nameserver 8.8.8.8" > /etc/resolv.conf
        echo "nameserver 8.8.4.4" >> /etc/resolv.conf
        chattr +i /etc/resolv.conf 2>/dev/null || log_warning "无法设置文件属性"
        
        # 2. 配置WSL网络
        log_info "2. 配置WSL网络"
        cat > /etc/wsl.conf << 'EOF'
[network]
generateResolvConf = false
hostname = zsf

[boot]
systemd = true

[interop]
appendWindowsPath = true
EOF
        
        # 3. 重启网络服务
        log_info "3. 重启网络服务"
        systemctl restart systemd-networkd 2>/dev/null || log_info "使用传统网络配置"
    else
        log_info "原生Linux环境网络修复..."
        
        # 重启网络管理器
        systemctl restart NetworkManager 2>/dev/null || \
        systemctl restart networking 2>/dev/null || \
        log_warning "无法重启网络服务"
    fi
    
    # 测试网络连接
    log_info "测试网络连接..."
    if ping -c 2 8.8.8.8 >/dev/null 2>&1; then
        log_success "网络连接正常"
    else
        log_warning "网络连接仍有问题"
    fi
}

# 修复系统服务
fix_services() {
    log_info "修复系统服务..."
    
    # 禁用可能引起问题的服务
    local problem_services=(
        "systemd-networkd-wait-online"
        "NetworkManager-wait-online"
        "apparmor"
        "modemmanager"
    )
    
    for service in "${problem_services[@]}"; do
        if systemctl is-enabled "$service" 2>/dev/null | grep -q "enabled"; then
            log_info "禁用服务: $service"
            systemctl disable "$service" 2>/dev/null || true
            systemctl mask "$service" 2>/dev/null || true
        fi
    done
    
    # 启用必要服务
    local essential_services=(
        "ssh"
        "cron"
        "systemd-timesyncd"
    )
    
    for service in "${essential_services[@]}"; do
        if systemctl is-enabled "$service" 2>/dev/null | grep -q "disabled"; then
            log_info "启用服务: $service"
            systemctl enable "$service" 2>/dev/null || true
        fi
    done
    
    # 重启systemd
    log_info "重启systemd"
    systemctl daemon-reload
}

# 清理系统资源
cleanup_system() {
    log_info "清理系统资源..."
    
    # 清理临时文件
    log_info "清理临时文件"
    rm -rf /tmp/*
    rm -rf /var/tmp/*
    
    # 清理日志文件
    log_info "清理日志文件"
    find /var/log -type f -name "*.log" -size +10M -delete 2>/dev/null || true
    
    # 清理apt缓存
    log_info "清理apt缓存"
    apt clean 2>/dev/null || true
    apt autoclean 2>/dev/null || true
    
    # 清理系统dumps
    log_info "清理系统dumps"
    rm -f /var/crash/* 2>/dev/null || true
}

# 修复启动问题
fix_boot() {
    log_info "修复启动问题..."
    
    # 1. 修复GRUB配置
    log_info "1. 修复GRUB配置"
    if [ -f /etc/default/grub ]; then
        # 增加启动超时
        sed -i 's/GRUB_TIMEOUT=.*/GRUB_TIMEOUT=10/' /etc/default/grub
        # 禁用quiet启动
        sed -i 's/ quiet//g' /etc/default/grub
        
        # 更新GRUB
        update-grub 2>/dev/null || grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null || log_warning "无法更新GRUB"
    fi
    
    # 2. 修复initramfs
    log_info "2. 修复initramfs"
    update-initramfs -u 2>/dev/null || log_warning "无法更新initramfs"
    
    # 3. 修复fstab
    log_info "3. 检查fstab配置"
    if grep -q "nofail" /etc/fstab; then
        log_success "fstab已配置nofail选项"
    else
        log_warning "建议在fstab中添加nofail选项防止启动失败"
    fi
}

# 设置SELF-LNN环境
setup_slnn_environment() {
    log_info "设置SELF-LNN环境..."
    
    # 创建目录结构
    log_info "创建SELF-LNN目录结构"
    mkdir -p /opt/slnn/{bin,lib,config,data,logs,models,backups}
    mkdir -p /opt/slnn/data/{knowledge,memory,training}
    
    # 设置权限
    log_info "设置目录权限"
    chmod 755 /opt/slnn
    chmod -R 755 /opt/slnn/{bin,lib,config}
    chmod -R 777 /opt/slnn/{data,logs,models,backups}
    
    # 创建服务用户
    log_info "创建服务用户"
    if ! id -u slnn >/dev/null 2>&1; then
        useradd -r -s /bin/false -d /opt/slnn slnn
        chown -R slnn:slnn /opt/slnn
    fi
    
    # 安装依赖
    log_info "安装系统依赖"
    apt update 2>/dev/null || true
    apt install -y build-essential cmake git python3 python3-pip 2>/dev/null || log_warning "无法安装所有依赖"
    
    log_success "SELF-LNN环境设置完成"
}

# 生成恢复报告
generate_report() {
    local report_file="/opt/slnn/reports/recovery_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "生成恢复报告: $report_file"
    
    mkdir -p /opt/slnn/reports
    
    {
        echo "=== SELF-LNN Ubuntu系统恢复报告 ==="
        echo "恢复时间: $(date)"
        echo "系统: $(uname -a)"
        echo ""
        
        echo "=== 系统信息 ==="
        cat /etc/os-release 2>/dev/null || echo "无法获取系统信息"
        echo ""
        
        echo "=== 磁盘空间 ==="
        df -h
        echo ""
        
        echo "=== 内存使用 ==="
        free -h
        echo ""
        
        echo "=== 系统日志 ==="
        dmesg | tail -100
        echo ""
        
        echo "=== 网络配置 ==="
        ip addr show
        echo ""
        ip route show
        echo ""
        
        echo "=== 恢复操作 ==="
        echo "$RECOVERY_LOG"
        echo ""
        
        echo "=== 建议 ==="
        echo "1. 定期备份系统"
        echo "2. 监控磁盘空间使用"
        echo "3. 保持系统更新"
        echo "4. 使用SELF-LNN监控工具"
        
    } > "$report_file"
    
    log_success "恢复报告已保存到: $report_file"
}

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -d, --diagnose      诊断系统问题"
    echo "  -f, --fix-fs        修复文件系统错误"
    echo "  -n, --fix-network   修复网络问题"
    echo "  -s, --fix-services  修复系统服务"
    echo "  -c, --cleanup       清理系统资源"
    echo "  -b, --fix-boot      修复启动问题"
    echo "  -e, --setup-slnn    设置SELF-LNN环境"
    echo "  -r, --report        生成恢复报告"
    echo "  -a, --all           执行所有修复"
    echo ""
    echo "示例:"
    echo "  sudo $0 -d          诊断系统问题"
    echo "  sudo $0 -a          执行所有修复"
    echo "  sudo $0 -e          设置SELF-LNN环境"
}

# 主函数
main() {
    # 全局恢复日志
    RECOVERY_LOG=""
    
    # 显示标题
    echo ""
    echo "================================================================================"
    echo "                     SELF-LNN Ubuntu系统恢复工具"
    echo "================================================================================"
    echo ""
    
    # 检查root权限
    check_root
    
    # 检查环境
    check_environment
    
    # 解析参数
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # 执行选项
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--diagnose)
                RECOVERY_LOG+="[$(date)] 执行系统诊断\n"
                diagnose_system
                ;;
            -f|--fix-fs)
                RECOVERY_LOG+="[$(date)] 修复文件系统\n"
                fix_filesystem
                ;;
            -n|--fix-network)
                RECOVERY_LOG+="[$(date)] 修复网络问题\n"
                fix_network
                ;;
            -s|--fix-services)
                RECOVERY_LOG+="[$(date)] 修复系统服务\n"
                fix_services
                ;;
            -c|--cleanup)
                RECOVERY_LOG+="[$(date)] 清理系统资源\n"
                cleanup_system
                ;;
            -b|--fix-boot)
                RECOVERY_LOG+="[$(date)] 修复启动问题\n"
                fix_boot
                ;;
            -e|--setup-slnn)
                RECOVERY_LOG+="[$(date)] 设置SELF-LNN环境\n"
                setup_slnn_environment
                ;;
            -r|--report)
                generate_report
                ;;
            -a|--all)
                RECOVERY_LOG+="[$(date)] 开始全面系统恢复\n"
                log_info "执行全面系统恢复..."
                
                diagnose_system
                fix_filesystem
                fix_network
                fix_services
                cleanup_system
                fix_boot
                setup_slnn_environment
                generate_report
                
                log_success "全面系统恢复完成"
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
    
    echo ""
    echo "================================================================================"
    log_success "系统恢复操作完成"
    echo ""
    
    # 提供建议
    echo "建议下一步:"
    echo "  1. 重启系统测试修复效果"
    echo "  2. 查看恢复报告: cat /opt/slnn/reports/recovery_*.txt"
    echo "  3. 部署SELF-LNN项目: ./scripts/deploy.sh -t /opt/slnn -s"
    echo "  4. 监控系统状态: ./scripts/network_diagnosis.sh -a"
    echo ""
}

# 执行主函数
main "$@"