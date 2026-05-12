#!/bin/bash

# SELF-LNN 跨平台部署脚本
# 自动部署项目到目标环境（支持Linux/macOS）

set -e  # 遇到错误时退出脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目信息
PROJECT_NAME="SELF-LNN"
PROJECT_VERSION="0.1.0"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
INSTALL_DIR="${PROJECT_ROOT}/install"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"
CONFIG_DIR="${PROJECT_ROOT}/config"
DATA_DIR="${PROJECT_ROOT}/data"

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

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -t, --target DIR    指定部署目标目录（默认: 当前目录/deploy）"
    echo "  -c, --config DIR    指定配置文件目录（默认: config）"
    echo "  -d, --data DIR      指定数据目录（默认: data）"
    echo "  -s, --service       安装为系统服务（systemd/launchd）"
    echo "  -p, --port PORT     指定服务端口（默认: 8081）"
    echo "  -f, --frontend-port PORT 前端端口（默认: 8080）"
    echo "  -n, --no-build      跳过构建步骤（假设已构建）"
    echo "  -v, --verbose       显示详细输出"
    echo "  -j N                并行构建任务数"
    echo ""
    echo "示例:"
    echo "  $0                  部署到默认目录"
    echo "  $0 -t /opt/slnn     部署到指定目录"
    echo "  $0 -s -p 9000       安装为服务并指定端口"
    echo "  $0 -n               跳过构建，直接部署"
}

# 解析命令行参数
parse_args() {
    DEPLOY_TARGET="$DEPLOY_DIR"
    INSTALL_SERVICE=0
    SERVICE_PORT=8081
    FRONTEND_PORT=8080
    SKIP_BUILD=0
    VERBOSE=0
    PARALLEL_JOBS=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--target)
                if [[ -n "$2" && "$2" != -* ]]; then
                    DEPLOY_TARGET="$2"
                    shift 2
                else
                    log_error "-t 参数需要目录路径"
                    exit 1
                fi
                ;;
            -c|--config)
                if [[ -n "$2" && "$2" != -* ]]; then
                    CONFIG_DIR="$2"
                    shift 2
                else
                    log_error "-c 参数需要目录路径"
                    exit 1
                fi
                ;;
            -d|--data)
                if [[ -n "$2" && "$2" != -* ]]; then
                    DATA_DIR="$2"
                    shift 2
                else
                    log_error "-d 参数需要目录路径"
                    exit 1
                fi
                ;;
            -s|--service)
                INSTALL_SERVICE=1
                shift
                ;;
            -p|--port)
                if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                    SERVICE_PORT="$2"
                    shift 2
                else
                    log_error "-p 参数需要端口号"
                    exit 1
                fi
                ;;
            -f|--frontend-port)
                if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                    FRONTEND_PORT="$2"
                    shift 2
                else
                    log_error "-f 参数需要端口号"
                    exit 1
                fi
                ;;
            -n|--no-build)
                SKIP_BUILD=1
                shift
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -j)
                if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                    PARALLEL_JOBS="$2"
                    shift 2
                else
                    log_error "-j 参数需要数字"
                    exit 1
                fi
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 检查项目结构
check_project_structure() {
    log_info "检查项目结构..."
    
    if [ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
        log_error "项目根目录缺少CMakeLists.txt"
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/README.md" ]; then
        log_warning "项目根目录缺少README.md"
    fi
    
    if [ ! -d "$BUILD_DIR" ]; then
        log_warning "构建目录不存在，需要构建项目"
    fi
    
    log_success "项目结构检查完成"
}

# 构建项目
build_project() {
    log_info "构建项目..."
    
    # 检查是否已安装构建工具
    if ! command -v cmake >/dev/null 2>&1; then
        log_error "CMake 未安装"
        log_info "请安装CMake:"
        log_info "  Ubuntu/Debian: sudo apt-get install cmake"
        log_info "  CentOS/RHEL: sudo yum install cmake"
        log_info "  macOS: brew install cmake"
        exit 1
    fi
    
    # 运行构建脚本
    log_info "运行构建脚本..."
    
    if [ -f "$PROJECT_ROOT/scripts/build.sh" ]; then
        cd "$PROJECT_ROOT"
        
        local build_args=("--install")
        
        if [ $VERBOSE -eq 1 ]; then
            build_args+=("--verbose")
        fi
        
        if [ -n "$PARALLEL_JOBS" ]; then
            build_args+=("-j" "$PARALLEL_JOBS")
        fi
        
        if ! ./scripts/build.sh "${build_args[@]}"; then
            log_error "构建失败"
            exit 1
        fi
    else
        log_error "构建脚本不存在: scripts/build.sh"
        exit 1
    fi
    
    log_success "项目构建完成"
}

# 准备部署目录
prepare_deploy_dir() {
    log_info "准备部署目录: $DEPLOY_TARGET"
    
    # 创建部署目录结构
    local dirs=(
        "$DEPLOY_TARGET"
        "$DEPLOY_TARGET/bin"
        "$DEPLOY_TARGET/lib"
        "$DEPLOY_TARGET/include"
        "$DEPLOY_TARGET/config"
        "$DEPLOY_TARGET/data"
        "$DEPLOY_TARGET/logs"
        "$DEPLOY_TARGET/models"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            if [ $? -ne 0 ]; then
                log_error "创建目录失败: $dir"
                exit 1
            fi
        fi
    done
    
    log_success "部署目录准备完成"
}

# 复制文件
copy_files() {
    log_info "复制文件到部署目录..."
    
    # 复制可执行文件
    if [ -d "$INSTALL_DIR/bin" ]; then
        if cp -r "$INSTALL_DIR/bin/"* "$DEPLOY_TARGET/bin/" 2>/dev/null; then
            log_success "复制可执行文件完成"
        else
            log_warning "复制可执行文件时出错"
        fi
    fi
    
    # 复制库文件
    if [ -d "$INSTALL_DIR/lib" ]; then
        if cp -r "$INSTALL_DIR/lib/"* "$DEPLOY_TARGET/lib/" 2>/dev/null; then
            log_success "复制库文件完成"
        else
            log_warning "复制库文件时出错"
        fi
    fi
    
    # 复制头文件
    if [ -d "$INSTALL_DIR/include" ]; then
        if cp -r "$INSTALL_DIR/include/"* "$DEPLOY_TARGET/include/" 2>/dev/null; then
            log_success "复制头文件完成"
        else
            log_warning "复制头文件时出错"
        fi
    fi
    
    # 复制配置文件
    if [ -d "$CONFIG_DIR" ]; then
        if cp -r "$CONFIG_DIR/"* "$DEPLOY_TARGET/config/" 2>/dev/null; then
            log_success "复制配置文件完成"
        else
            log_warning "复制配置文件时出错"
        fi
    fi
    
    # 复制数据文件
    if [ -d "$DATA_DIR" ]; then
        if cp -r "$DATA_DIR/"* "$DEPLOY_TARGET/data/" 2>/dev/null; then
            log_success "复制数据文件完成"
        else
            log_warning "复制数据文件时出错"
        fi
    fi
    
    # 复制前端文件
    if [ -d "$PROJECT_ROOT/frontend" ]; then
        if cp -r "$PROJECT_ROOT/frontend/"* "$DEPLOY_TARGET/frontend/" 2>/dev/null; then
            log_success "复制前端文件完成"
        else
            log_warning "复制前端文件时出错"
        fi
    fi
    
    log_success "文件复制完成"
}

# 创建配置文件
create_config_files() {
    log_info "创建配置文件..."
    
    # 创建后端配置文件
    local backend_config="$DEPLOY_TARGET/config/backend_config.json"
    if [ ! -f "$backend_config" ]; then
        log_info "创建后端配置文件..."
        cat > "$backend_config" << EOF
{
    "server": {
        "host": "0.0.0.0",
        "port": $SERVICE_PORT,
        "max_connections": 100,
        "timeout": 30
    },
    "database": {
        "path": "data/slnn.db",
        "max_connections": 10
    },
    "logging": {
        "level": "info",
        "file": "logs/slnn.log",
        "max_size": "10MB",
        "backup_count": 5
    },
    "models": {
        "path": "models",
        "auto_save_interval": 300
    }
}
EOF
        
        if [ $? -eq 0 ]; then
            log_success "后端配置文件已创建: $backend_config"
        else
            log_warning "创建后端配置文件失败"
        fi
    fi
    
    # 创建启动脚本
    local start_script="$DEPLOY_TARGET/start_slnn.sh"
    log_info "创建启动脚本..."
    
    cat > "$start_script" << EOF
#!/bin/bash

# SELF-LNN 启动脚本
echo "启动 SELF-LNN AGI 系统..."

# 设置环境变量
PROJECT_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export PATH="\$PROJECT_ROOT/bin:\$PATH"
export LD_LIBRARY_PATH="\$PROJECT_ROOT/lib:\$LD_LIBRARY_PATH"

# 启动后端服务器
echo "[INFO] 启动后端服务器..."
nohup "\$PROJECT_ROOT/bin/backend_server" --config "\$PROJECT_ROOT/config/backend_config.json" > "\$PROJECT_ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=\$!
echo \$BACKEND_PID > "\$PROJECT_ROOT/logs/backend.pid"

# 启动前端服务器（如果存在）
if [ -f "\$PROJECT_ROOT/frontend/index.html" ]; then
    echo "[INFO] 启动前端服务器..."
    cd "\$PROJECT_ROOT/frontend"
    nohup python3 -m http.server $FRONTEND_PORT > "\$PROJECT_ROOT/logs/frontend.log" 2>&1 &
    FRONTEND_PID=\$!
    echo \$FRONTEND_PID > "\$PROJECT_ROOT/logs/frontend.pid"
fi

echo "[SUCCESS] SELF-LNN 系统已启动"
echo "后端地址: http://localhost:$SERVICE_PORT"
echo "前端地址: http://localhost:$FRONTEND_PORT"
echo "查看日志: tail -f \$PROJECT_ROOT/logs/backend.log"
EOF
    
    chmod +x "$start_script"
    if [ $? -eq 0 ]; then
        log_success "启动脚本已创建: $start_script"
    else
        log_warning "创建启动脚本失败"
    fi
    
    # 创建停止脚本
    local stop_script="$DEPLOY_TARGET/stop_slnn.sh"
    
    cat > "$stop_script" << EOF
#!/bin/bash

# SELF-LNN 停止脚本
echo "停止 SELF-LNN AGI 系统..."

PROJECT_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

# 停止后端服务器
if [ -f "\$PROJECT_ROOT/logs/backend.pid" ]; then
    BACKEND_PID=\$(cat "\$PROJECT_ROOT/logs/backend.pid")
    if kill -0 \$BACKEND_PID 2>/dev/null; then
        kill \$BACKEND_PID
        echo "[INFO] 后端服务器已停止 (PID: \$BACKEND_PID)"
    fi
    rm -f "\$PROJECT_ROOT/logs/backend.pid"
fi

# 停止前端服务器
if [ -f "\$PROJECT_ROOT/logs/frontend.pid" ]; then
    FRONTEND_PID=\$(cat "\$PROJECT_ROOT/logs/frontend.pid")
    if kill -0 \$FRONTEND_PID 2>/dev/null; then
        kill \$FRONTEND_PID
        echo "[INFO] 前端服务器已停止 (PID: \$FRONTEND_PID)"
    fi
    rm -f "\$PROJECT_ROOT/logs/frontend.pid"
fi

# 查找并杀死剩余进程
pkill -f "backend_server" 2>/dev/null
pkill -f "http.server.*$FRONTEND_PORT" 2>/dev/null

echo "[SUCCESS] SELF-LNN 系统已停止"
EOF
    
    chmod +x "$stop_script"
    if [ $? -eq 0 ]; then
        log_success "停止脚本已创建: $stop_script"
    else
        log_warning "创建停止脚本失败"
    fi
    
    log_success "配置文件创建完成"
}

# 显示部署总结
show_deployment_summary() {
    echo ""
    echo "================================================================================"
    echo "                         部署总结"
    echo "================================================================================"
    echo ""
    echo "项目名称: $PROJECT_NAME"
    echo "版本: $PROJECT_VERSION"
    echo "部署目录: $DEPLOY_TARGET"
    echo ""
    echo "包含内容:"
    echo "  - 可执行文件: $DEPLOY_TARGET/bin/"
    echo "  - 配置文件: $DEPLOY_TARGET/config/"
    echo "  - 数据文件: $DEPLOY_TARGET/data/"
    echo "  - 模型文件: $DEPLOY_TARGET/models/"
    echo "  - 日志文件: $DEPLOY_TARGET/logs/"
    echo ""
    echo "启动方式:"
    echo "  1. 手动启动: $DEPLOY_TARGET/start_slnn.sh"
    echo "  2. 停止系统: $DEPLOY_TARGET/stop_slnn.sh"
    echo ""
    echo "访问地址:"
    echo "  - 后端API: http://localhost:$SERVICE_PORT/api/v1/"
    echo "  - 前端界面: http://localhost:$FRONTEND_PORT/"
    echo ""
    echo "配置文件:"
    echo "  - 后端配置: $DEPLOY_TARGET/config/backend_config.json"
    echo ""
    echo "下一步:"
    echo "  1. 确保防火墙允许端口 $SERVICE_PORT 和 $FRONTEND_PORT"
    echo "  2. 根据需要修改配置文件"
    echo "  3. 查看日志文件: $DEPLOY_TARGET/logs/slnn.log"
    echo ""
    echo "================================================================================"
}

# 主函数
main() {
    parse_args "$@"
    
    echo ""
    echo "================================================================================"
    echo "                         部署 $PROJECT_NAME v$PROJECT_VERSION"
    echo "================================================================================"
    echo ""
    
    # 检查项目结构
    check_project_structure
    
    # 构建项目（如果需要）
    if [ $SKIP_BUILD -eq 0 ]; then
        build_project
    fi
    
    # 准备部署目录
    prepare_deploy_dir
    
    # 复制文件
    copy_files
    
    # 创建配置文件
    create_config_files
    
    # 显示部署总结
    show_deployment_summary
    
    echo ""
    log_success "$PROJECT_NAME 部署完成！"
    echo ""
}

# 执行主函数
main "$@"