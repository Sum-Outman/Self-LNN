#!/bin/bash

# SELF-LNN 网络诊断工具
# 用于诊断WSL/Ubuntu网络连接问题

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

# 检查命令是否存在
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        log_info "$1 已安装: $(which $1)"
        return 0
    else
        log_warning "$1 未安装"
        return 1
    fi
}

# 测试网络连通性
test_ping() {
    local target="$1"
    log_info "测试ping连通性: $target"
    
    if ping -c 3 -W 2 "$target" >/dev/null 2>&1; then
        log_success "ping $target: 成功"
        return 0
    else
        log_error "ping $target: 失败"
        return 1
    fi
}

# 测试DNS解析
test_dns() {
    local domain="$1"
    log_info "测试DNS解析: $domain"
    
    if nslookup "$domain" >/dev/null 2>&1; then
        log_success "DNS解析 $domain: 成功"
        return 0
    elif dig +short "$domain" >/dev/null 2>&1; then
        log_success "DNS解析 $domain: 成功"
        return 0
    else
        log_error "DNS解析 $domain: 失败"
        return 1
    fi
}

# 测试HTTP连接
test_http() {
    local url="$1"
    log_info "测试HTTP连接: $url"
    
    if curl -s -I --connect-timeout 5 "$url" >/dev/null 2>&1; then
        log_success "HTTP连接 $url: 成功"
        return 0
    elif wget -q --spider --timeout=5 "$url" 2>/dev/null; then
        log_success "HTTP连接 $url: 成功"
        return 0
    else
        log_error "HTTP连接 $url: 失败"
        return 1
    fi
}

# 显示网络配置
show_network_config() {
    log_info "网络配置信息:"
    echo ""
    
    # IP地址和接口
    log_info "网络接口:"
    ip addr show 2>/dev/null || ifconfig 2>/dev/null || echo "  无法获取网络接口信息"
    echo ""
    
    # 路由表
    log_info "路由表:"
    ip route show 2>/dev/null || route -n 2>/dev/null || echo "  无法获取路由信息"
    echo ""
    
    # DNS配置
    log_info "DNS配置:"
    if [ -f /etc/resolv.conf ]; then
        cat /etc/resolv.conf
    else
        echo "  /etc/resolv.conf 不存在"
    fi
    echo ""
    
    # 主机名
    log_info "主机信息:"
    echo "  主机名: $(hostname 2>/dev/null || echo '未知')"
    echo "  域名: $(domainname 2>/dev/null || echo '未知')"
    echo ""
}

# 检查WSL特定配置
check_wsl_config() {
    log_info "检查WSL配置..."
    echo ""
    
    # 检查WSL版本
    if command -v wsl >/dev/null 2>&1 || [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        log_info "检测到WSL环境"
        
        # 检查WSL版本
        if [ -f /proc/version ]; then
            grep -i "microsoft" /proc/version && log_info "  WSL 2 (基于Microsoft内核)" || log_info "  WSL 1"
        fi
        
        # 检查网络模式
        if ip link show eth0 2>/dev/null | grep -q "master"; then
            log_info "  网络模式: NAT"
        else
            log_info "  网络模式: 镜像模式或桥接"
        fi
        
        # 检查DNS配置
        if grep -q "nameserver 172." /etc/resolv.conf 2>/dev/null; then
            log_info "  DNS: WSL内置DNS"
        else
            log_info "  DNS: 自定义配置"
        fi
    else
        log_info "非WSL环境 (原生Linux)"
    fi
    echo ""
}

# 检查防火墙
check_firewall() {
    log_info "检查防火墙状态..."
    echo ""
    
    # 检查iptables
    if command -v iptables >/dev/null 2>&1; then
        log_info "iptables规则:"
        sudo iptables -L -n 2>/dev/null | head -20 || log_warning "  需要sudo权限查看iptables"
    fi
    
    # 检查ufw (Ubuntu)
    if command -v ufw >/dev/null 2>&1; then
        log_info "ufw状态:"
        sudo ufw status 2>/dev/null || log_warning "  需要sudo权限查看ufw状态"
    fi
    
    # 检查Windows防火墙（WSL特定）
    if [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        log_info "WSL环境 - 可能需要检查Windows防火墙"
    fi
    echo ""
}

# 测试SELF-LNN所需端口
test_slnn_ports() {
    log_info "测试SELF-LNN所需端口..."
    echo ""
    
    local ports=(8081 8080 9090)
    
    for port in "${ports[@]}"; do
        log_info "端口 $port:"
        
        # 检查端口是否监听
        if netstat -tulpn 2>/dev/null | grep -q ":$port "; then
            log_warning "  端口 $port 已被占用"
            netstat -tulpn 2>/dev/null | grep ":$port " || true
        else
            log_success "  端口 $port 可用"
        fi
        
        # 测试本地连接
        if timeout 2 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
            log_warning "  本地连接端口 $port: 成功 (可能有服务运行)"
        else
            log_success "  本地连接端口 $port: 失败 (端口空闲)"
        fi
        echo ""
    done
}

# 生成诊断报告
generate_report() {
    local report_file="network_diagnosis_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "生成诊断报告: $report_file"
    
    {
        echo "=== SELF-LNN 网络诊断报告 ==="
        echo "生成时间: $(date)"
        echo "系统: $(uname -a)"
        echo ""
        
        echo "=== 网络配置 ==="
        ip addr show 2>/dev/null || echo "无法获取网络接口"
        echo ""
        
        echo "=== DNS配置 ==="
        cat /etc/resolv.conf 2>/dev/null || echo "无法获取DNS配置"
        echo ""
        
        echo "=== 路由表 ==="
        ip route show 2>/dev/null || echo "无法获取路由表"
        echo ""
        
        echo "=== 测试结果 ==="
        echo "ping 8.8.8.8: $(ping -c 2 -W 1 8.8.8.8 >/dev/null 2>&1 && echo '成功' || echo '失败')"
        echo "DNS解析 github.com: $(nslookup github.com >/dev/null 2>&1 && echo '成功' || echo '失败')"
        echo "HTTP连接 raw.githubusercontent.com: $(curl -s -I --connect-timeout 3 https://raw.githubusercontent.com >/dev/null 2>&1 && echo '成功' || echo '失败')"
        echo ""
        
        echo "=== 端口检查 ==="
        netstat -tulpn 2>/dev/null | grep -E ':8081|:8080|:9090' || echo "相关端口未监听"
        echo ""
        
        echo "=== 建议 ==="
        echo "如果网络测试失败，请参考 docs/WSL_网络故障排除.md 文档"
    } > "$report_file"
    
    log_success "诊断报告已保存到: $report_file"
    echo ""
    cat "$report_file"
}

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -t, --test          运行完整网络测试"
    echo "  -c, --config        显示网络配置"
    echo "  -w, --wsl           检查WSL特定配置"
    echo "  -f, --firewall      检查防火墙状态"
    echo "  -p, --ports         测试SELF-LNN端口"
    echo "  -r, --report        生成诊断报告"
    echo "  -a, --all           运行所有检查"
    echo ""
    echo "示例:"
    echo "  $0 -t               运行网络测试"
    echo "  $0 -a               运行所有检查"
    echo "  $0 -r               生成诊断报告"
}

# 主函数
main() {
    # 显示标题
    echo ""
    echo "================================================================================"
    echo "                     SELF-LNN 网络诊断工具"
    echo "================================================================================"
    echo ""
    
    # 检查是否在WSL中
    if [ -f /proc/sys/fs/binfmt_misc/WSLInterop ] || grep -q "Microsoft" /proc/version 2>/dev/null; then
        log_info "检测到WSL环境"
    else
        log_info "检测到原生Linux环境"
    fi
    
    # 检查参数
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--test)
                log_info "运行网络测试..."
                echo ""
                
                # 基本网络测试
                test_ping "8.8.8.8"
                test_ping "1.1.1.1"
                test_dns "google.com"
                test_dns "github.com"
                test_http "http://www.msftconnecttest.com"
                test_http "https://raw.githubusercontent.com"
                ;;
            -c|--config)
                show_network_config
                ;;
            -w|--wsl)
                check_wsl_config
                ;;
            -f|--firewall)
                check_firewall
                ;;
            -p|--ports)
                test_slnn_ports
                ;;
            -r|--report)
                generate_report
                ;;
            -a|--all)
                log_info "运行所有检查..."
                echo ""
                show_network_config
                check_wsl_config
                check_firewall
                test_slnn_ports
                
                log_info "运行网络测试..."
                echo ""
                test_ping "8.8.8.8"
                test_dns "google.com"
                test_http "https://raw.githubusercontent.com"
                
                generate_report
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
    log_info "网络诊断完成"
    echo ""
    
    # 提供建议
    echo "建议下一步:"
    echo "  1. 如果网络有问题，参考 docs/WSL_网络故障排除.md"
    echo "  2. 运行离线部署: ./scripts/deploy.sh -t ~/slnn-local -n"
    echo "  3. 测试部署脚本: ./scripts/test_deployment.sh -a"
    echo ""
}

# 执行主函数
main "$@"