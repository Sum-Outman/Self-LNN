#!/bin/bash

# SELF-LNN 快速启动脚本
# 自动检查依赖项、构建项目并运行测试

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
CONFIG_DIR="${PROJECT_ROOT}/config"
DATA_DIR="${PROJECT_ROOT}/data"
MODELS_DIR="${PROJECT_ROOT}/models"

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
        log_error "$1 未安装"
        return 1
    fi
}

# 检查依赖项
check_dependencies() {
    log_info "检查系统依赖项..."
    
    local missing_deps=0
    
    # 检查C编译器
    if check_command "gcc"; then
        log_info "GCC 版本: $(gcc --version | head -n1)"
    elif check_command "clang"; then
        log_info "Clang 版本: $(clang --version | head -n1)"
    elif check_command "cc"; then
        log_info "C编译器: $(cc --version | head -n1)"
    else
        log_error "未找到C编译器 (gcc/clang/cc)"
        missing_deps=$((missing_deps + 1))
    fi
    
    # 检查CMake
    if check_command "cmake"; then
        log_info "CMake 版本: $(cmake --version | head -n1)"
    else
        log_error "CMake 未安装"
        missing_deps=$((missing_deps + 1))
    fi
    
    # 检查Python (可选)
    if check_command "python3"; then
        log_info "Python3 版本: $(python3 --version 2>&1)"
    elif check_command "python"; then
        log_info "Python 版本: $(python --version 2>&1)"
    else
        log_warning "Python 未安装 (前端运行可能需要)"
    fi
    
    # 检查Git
    if check_command "git"; then
        log_info "Git 版本: $(git --version)"
    else
        log_warning "Git 未安装 (可选)"
    fi
    
    if [ $missing_deps -gt 0 ]; then
        log_error "缺少 $missing_deps 个必需依赖项"
        log_info "请安装缺失的依赖项后重试:"
        log_info "  - Ubuntu/Debian: sudo apt-get install build-essential cmake python3 git"
        log_info "  - CentOS/RHEL: sudo yum install gcc-c++ cmake python3 git"
        log_info "  - macOS: brew install gcc cmake python3 git"
        log_info "  - Windows: 请安装 MinGW-w64 或 MSVC 和 CMake"
        return 1
    fi
    
    log_success "所有必需依赖项已满足"
    return 0
}

# 准备项目目录
prepare_directories() {
    log_info "准备项目目录..."
    
    # 创建必要的目录
    for dir in "$BUILD_DIR" "$INSTALL_DIR" "$DATA_DIR" "$MODELS_DIR"; do
        if [ ! -d "$dir" ]; then
            log_info "创建目录: $dir"
            mkdir -p "$dir"
        else
            log_info "目录已存在: $dir"
        fi
    done
    
    # 检查配置文件
    if [ ! -f "$CONFIG_DIR/train_config.json" ]; then
        log_warning "训练配置文件不存在: $CONFIG_DIR/train_config.json"
        log_info "创建示例配置文件..."
        cat > "$CONFIG_DIR/train_config.json" << EOF
{
    "network": {
        "input_size": 512,
        "hidden_size": 256,
        "output_size": 128,
        "time_step": 0.01
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_split": 0.2
    },
    "memory": {
        "short_term_capacity": 1000,
        "long_term_capacity": 10000
    }
}
EOF
    fi
    
    log_success "项目目录准备完成"
}

# 配置CMake项目
configure_project() {
    log_info "配置CMake项目..."
    
    cd "$BUILD_DIR"
    
    # 检查是否已配置
    if [ -f "CMakeCache.txt" ]; then
        log_warning "CMake缓存已存在，如需重新配置请删除 $BUILD_DIR 目录"
        log_info "跳过配置步骤..."
        return 0
    fi
    
    # CMake配置选项
    local cmake_options=(
        "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
        "-DBUILD_TESTS=ON"
        "-DBUILD_FRONTEND=ON"
        "-DBUILD_EXAMPLES=ON"
        "-DCMAKE_BUILD_TYPE=Release"
    )
    
    # 如果是Windows，添加特定选项
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        cmake_options+=("-G" "MinGW Makefiles")
    fi
    
    log_info "运行: cmake ${PROJECT_ROOT} ${cmake_options[*]}"
    
    if ! cmake "${PROJECT_ROOT}" "${cmake_options[@]}"; then
        log_error "CMake配置失败"
        return 1
    fi
    
    log_success "CMake配置成功"
}

# 构建项目
build_project() {
    log_info "构建项目..."
    
    cd "$BUILD_DIR"
    
    # 获取CPU核心数用于并行构建
    local cpu_cores=4
    if command -v nproc >/dev/null 2>&1; then
        cpu_cores=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    fi
    
    log_info "使用 $cpu_cores 个并行任务构建"
    
    if ! cmake --build . --config Release --parallel "$cpu_cores"; then
        log_error "构建失败"
        return 1
    fi
    
    # 安装到本地目录
    log_info "安装到: $INSTALL_DIR"
    if ! cmake --install . --config Release; then
        log_warning "安装失败，但构建成功"
    fi
    
    log_success "项目构建成功"
}

# 运行测试
run_tests() {
    log_info "运行测试..."
    
    cd "$BUILD_DIR"
    
    if [ -f "CTestTestfile.cmake" ]; then
        if ! ctest --output-on-failure -C Release; then
            log_warning "部分测试失败"
            return 1
        fi
        log_success "所有测试通过"
    else
        log_warning "未找到测试配置，跳过测试"
    fi
}

# 运行验证脚本
run_verification() {
    log_info "运行项目验证..."
    
    if [ -f "${PROJECT_ROOT}/scripts/verify_project.py" ]; then
        if command -v python3 >/dev/null 2>&1; then
            if ! python3 "${PROJECT_ROOT}/scripts/verify_project.py"; then
                log_warning "验证脚本发现问题，请检查验证报告"
            else
                log_success "项目验证通过"
            fi
        elif command -v python >/dev/null 2>&1; then
            if ! python "${PROJECT_ROOT}/scripts/verify_project.py"; then
                log_warning "验证脚本发现问题，请检查验证报告"
            else
                log_success "项目验证通过"
            fi
        else
            log_warning "Python未安装，跳过验证"
        fi
    else
        log_warning "验证脚本不存在: ${PROJECT_ROOT}/scripts/verify_project.py"
    fi
}

# 显示启动指南
show_quickstart_guide() {
    log_info "快速启动指南"
    echo ""
    echo "================================================================================"
    echo "                          SELF-LNN 快速启动完成"
    echo "================================================================================"
    echo ""
    echo "项目已成功构建并安装到:"
    echo "  - 构建目录: $BUILD_DIR"
    echo "  - 安装目录: $INSTALL_DIR"
    echo ""
    echo "可执行文件位置:"
    echo "  - 核心测试程序: $BUILD_DIR/tests/core/test_core"
    echo "  - 后端服务器: $BUILD_DIR/src/backend/backend_server"
    echo "  - 训练程序: $BUILD_DIR/src/training/pretrain"
    echo ""
    echo "启动前端界面:"
    echo "  cd $PROJECT_ROOT/frontend"
    echo "  python3 -m http.server 8080  # 或使用其他HTTP服务器"
    echo ""
    echo "启动后端服务器:"
    echo "  $BUILD_DIR/src/backend/backend_server --port 8081 --config $CONFIG_DIR/backend_config.json"
    echo ""
    echo "运行示例程序:"
    echo "  $BUILD_DIR/examples/basic_lnn_example"
    echo ""
    echo "更多信息请参阅:"
    echo "  - README.md: 项目详细文档"
    echo "  - docs/: 设计文档和API文档"
    echo "  - examples/: 示例代码"
    echo ""
    echo "================================================================================"
}

# 主函数
main() {
    echo ""
    echo "================================================================================"
    echo "                        启动 $PROJECT_NAME v$PROJECT_VERSION"
    echo "================================================================================"
    echo ""
    
    # 检查依赖项
    if ! check_dependencies; then
        log_error "依赖项检查失败，请先安装缺失的依赖项"
        exit 1
    fi
    
    # 准备目录
    prepare_directories
    
    # 配置项目
    if ! configure_project; then
        log_error "项目配置失败"
        exit 1
    fi
    
    # 构建项目
    if ! build_project; then
        log_error "项目构建失败"
        exit 1
    fi
    
    # 运行测试
    if ! run_tests; then
        log_warning "测试阶段发现问题，但继续执行"
    fi
    
    # 运行验证
    run_verification
    
    # 显示指南
    show_quickstart_guide
    
    echo ""
    log_success "$PROJECT_NAME 快速启动完成！"
    echo ""
}

# 执行主函数
main "$@"