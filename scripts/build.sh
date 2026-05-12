#!/bin/bash

# SELF-LNN 构建脚本
# 快速构建项目

set -e  # 遇到错误时退出脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目信息
PROJECT_NAME="SELF-LNN"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

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
    echo "  -c, --clean         清理构建目录"
    echo "  -d, --debug         构建调试版本 (默认: Release)"
    echo "  -j N                并行构建任务数 (默认: 自动检测)"
    echo "  -t, --test          构建后运行测试"
    echo "  -i, --install       构建后安装到系统"
    echo "  -v, --verbose       显示详细输出"
    echo ""
    echo "示例:"
    echo "  $0                  构建Release版本"
    echo "  $0 -d -j4           使用4个任务构建调试版本"
    echo "  $0 -c               清理构建目录"
    echo "  $0 -t               构建并运行测试"
}

# 解析命令行参数
parse_args() {
    BUILD_TYPE="Release"
    CLEAN_BUILD=0
    RUN_TESTS=0
    RUN_INSTALL=0
    VERBOSE=0
    PARALLEL_JOBS=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--clean)
                CLEAN_BUILD=1
                shift
                ;;
            -d|--debug)
                BUILD_TYPE="Debug"
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
            -t|--test)
                RUN_TESTS=1
                shift
                ;;
            -i|--install)
                RUN_INSTALL=1
                shift
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 检查依赖项
check_dependencies() {
    log_info "检查构建依赖项..."
    
    # 检查CMake
    if ! command -v cmake >/dev/null 2>&1; then
        log_error "CMake 未安装"
        log_info "请先安装CMake:"
        log_info "  Ubuntu/Debian: sudo apt-get install cmake"
        log_info "  CentOS/RHEL: sudo yum install cmake"
        log_info "  macOS: brew install cmake"
        log_info "  Windows: 从 cmake.org 下载安装"
        exit 1
    fi
    
    # 检查C编译器
    if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
        log_error "未找到C编译器 (gcc/clang/cc)"
        log_info "请安装C编译器:"
        log_info "  Ubuntu/Debian: sudo apt-get install build-essential"
        log_info "  CentOS/RHEL: sudo yum install gcc-c++"
        log_info "  macOS: xcode-select --install"
        log_info "  Windows: 安装 MinGW-w64 或 MSVC"
        exit 1
    fi
    
    log_success "构建依赖项检查通过"
}

# 清理构建目录
clean_build() {
    log_info "清理构建目录: $BUILD_DIR"
    
    if [ -d "$BUILD_DIR" ]; then
        if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
            log_info "删除CMake缓存文件"
            rm -f "$BUILD_DIR/CMakeCache.txt"
        fi
        
        log_info "删除构建目录内容"
        rm -rf "$BUILD_DIR"/*
        
        if [ $? -eq 0 ]; then
            log_success "清理完成"
        else
            log_error "清理失败"
            exit 1
        fi
    else
        log_warning "构建目录不存在: $BUILD_DIR"
    fi
}

# 准备构建目录
prepare_build_dir() {
    log_info "准备构建目录: $BUILD_DIR"
    
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        log_info "创建构建目录"
    fi
    
    cd "$BUILD_DIR"
}

# 配置CMake
configure_cmake() {
    log_info "配置CMake (构建类型: $BUILD_TYPE)..."
    
    local cmake_options=(
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DBUILD_TESTS=ON"
        "-DBUILD_FRONTEND=ON"
        "-DBUILD_EXAMPLES=ON"
    )
    
    # 如果是Windows，添加特定选项
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        cmake_options+=("-G" "MinGW Makefiles")
    fi
    
    # 设置详细输出
    local cmake_output=""
    if [ $VERBOSE -eq 1 ]; then
        cmake_output=""
    else
        cmake_output="> cmake_config.log 2>&1"
    fi
    
    log_info "运行: cmake ${PROJECT_ROOT} ${cmake_options[*]}"
    
    if ! eval "cmake \"${PROJECT_ROOT}\" \"${cmake_options[@]}\" $cmake_output"; then
        if [ $VERBOSE -eq 0 ] && [ -f "cmake_config.log" ]; then
            log_error "CMake配置失败，查看日志: $BUILD_DIR/cmake_config.log"
            tail -20 "$BUILD_DIR/cmake_config.log"
        fi
        exit 1
    fi
    
    log_success "CMake配置成功"
}

# 构建项目
build_project() {
    log_info "构建项目 (类型: $BUILD_TYPE)..."
    
    # 获取并行任务数
    local parallel_arg=""
    if [ -n "$PARALLEL_JOBS" ]; then
        parallel_arg="--parallel $PARALLEL_JOBS"
        log_info "使用指定的并行任务数: $PARALLEL_JOBS"
    else
        # 自动检测CPU核心数
        local cpu_cores=4
        if command -v nproc >/dev/null 2>&1; then
            cpu_cores=$(nproc)
        elif command -v sysctl >/dev/null 2>&1; then
            cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
        fi
        parallel_arg="--parallel $cpu_cores"
        log_info "自动检测CPU核心数: $cpu_cores"
    fi
    
    # 设置构建输出
    local build_output=""
    if [ $VERBOSE -eq 1 ]; then
        build_output=""
    else
        build_output="> build.log 2>&1"
    fi
    
    log_info "运行: cmake --build . --config $BUILD_TYPE $parallel_arg"
    
    if ! eval "cmake --build . --config $BUILD_TYPE $parallel_arg $build_output"; then
        if [ $VERBOSE -eq 0 ] && [ -f "build.log" ]; then
            log_error "构建失败，查看日志: $BUILD_DIR/build.log"
            tail -20 "$BUILD_DIR/build.log"
        fi
        exit 1
    fi
    
    log_success "项目构建成功"
}

# 运行测试
run_tests() {
    log_info "运行测试..."
    
    if [ ! -f "CTestTestfile.cmake" ]; then
        log_warning "未找到测试配置，跳过测试"
        return 0
    fi
    
    local test_output=""
    if [ $VERBOSE -eq 1 ]; then
        test_output="--output-on-failure"
    else
        test_output="> test.log 2>&1"
    fi
    
    if [ $VERBOSE -eq 1 ]; then
        if ! ctest --output-on-failure -C "$BUILD_TYPE"; then
            log_warning "部分测试失败"
            return 1
        fi
    else
        if ! eval "ctest -C \"$BUILD_TYPE\" $test_output"; then
            if [ -f "test.log" ]; then
                log_warning "测试失败，查看日志: $BUILD_DIR/test.log"
                tail -20 "$BUILD_DIR/test.log"
            fi
            return 1
        fi
    fi
    
    log_success "所有测试通过"
    return 0
}

# 安装项目
install_project() {
    log_info "安装项目..."
    
    if ! cmake --install . --config "$BUILD_TYPE"; then
        log_warning "安装失败"
        return 1
    fi
    
    log_success "项目安装成功"
    return 0
}

# 显示构建摘要
show_summary() {
    echo ""
    echo "================================================================================"
    echo "                          $PROJECT_NAME 构建完成"
    echo "================================================================================"
    echo ""
    echo "构建类型: $BUILD_TYPE"
    echo "构建目录: $BUILD_DIR"
    echo ""
    
    # 显示生成的可执行文件
    if [ -d "$BUILD_DIR/tests" ]; then
        echo "测试程序:"
        find "$BUILD_DIR/tests" -name "test_*" -type f -executable | head -5 | while read test; do
            echo "  - $test"
        done
    fi
    
    if [ -d "$BUILD_DIR/src" ]; then
        echo ""
        echo "主要程序:"
        find "$BUILD_DIR/src" -type f -executable | grep -v "\.so\|\.dll\|\.dylib" | head -5 | while read exe; do
            echo "  - $exe"
        done
    fi
    
    if [ -d "$BUILD_DIR/examples" ]; then
        echo ""
        echo "示例程序:"
        find "$BUILD_DIR/examples" -type f -executable | head -5 | while read exe; do
            echo "  - $exe"
        done
    fi
    
    echo ""
    echo "下一步:"
    echo "  1. 运行测试: $0 -t"
    echo "  2. 运行示例: $BUILD_DIR/examples/basic_lnn_example"
    echo "  3. 启动后端: $BUILD_DIR/src/backend/backend_server"
    echo "  4. 更多信息: cat $PROJECT_ROOT/README.md"
    echo ""
    echo "================================================================================"
}

# 主函数
main() {
    parse_args "$@"
    
    echo ""
    echo "================================================================================"
    echo "                        构建 $PROJECT_NAME"
    echo "================================================================================"
    echo ""
    
    # 如果需要清理
    if [ $CLEAN_BUILD -eq 1 ]; then
        clean_build
        # 如果只需要清理，则退出
        if [ $CLEAN_BUILD -eq 1 ] && [ $RUN_TESTS -eq 0 ] && [ $RUN_INSTALL -eq 0 ]; then
            exit 0
        fi
    fi
    
    # 检查依赖项
    check_dependencies
    
    # 准备构建目录
    prepare_build_dir
    
    # 配置CMake (如果需要)
    if [ ! -f "CMakeCache.txt" ] || [ $CLEAN_BUILD -eq 1 ]; then
        configure_cmake
    else
        log_info "使用现有的CMake配置"
    fi
    
    # 构建项目
    build_project
    
    # 运行测试
    if [ $RUN_TESTS -eq 1 ]; then
        if ! run_tests; then
            log_warning "测试失败，但构建成功"
        fi
    fi
    
    # 安装项目
    if [ $RUN_INSTALL -eq 1 ]; then
        if ! install_project; then
            log_warning "安装失败，但构建成功"
        fi
    fi
    
    # 显示摘要
    show_summary
    
    echo ""
    log_success "$PROJECT_NAME 构建完成！"
    echo ""
}

# 执行主函数
main "$@"