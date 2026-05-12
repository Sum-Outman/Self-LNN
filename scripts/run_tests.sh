#!/bin/bash

# SELF-LNN 测试运行脚本
# 运行项目测试套件

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
TEST_DIR="${PROJECT_ROOT}/tests"

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
    echo "  -l, --list          列出所有测试"
    echo "  -r, --run [测试名]   运行指定测试（支持通配符）"
    echo "  -a, --all           运行所有测试（默认）"
    echo "  -v, --verbose       显示详细输出"
    echo "  -c, --coverage      生成测试覆盖率报告（需要gcov/lcov）"
    echo "  -b, --build-dir DIR 指定构建目录（默认: build）"
    echo "  -j N                并行运行测试数（默认: 自动检测）"
    echo ""
    echo "示例:"
    echo "  $0                  运行所有测试"
    echo "  $0 -l               列出所有可用测试"
    echo "  $0 -r test_core     运行核心测试"
    echo "  $0 -r \"test_*\"      运行所有匹配test_*的测试"
    echo "  $0 -v -j4           详细输出，并行运行4个测试"
}

# 解析命令行参数
parse_args() {
    RUN_ALL=1
    LIST_TESTS=0
    RUN_SPECIFIC=""
    VERBOSE=0
    COVERAGE=0
    PARALLEL_JOBS=""
    BUILD_DIR="${PROJECT_ROOT}/build"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -l|--list)
                LIST_TESTS=1
                RUN_ALL=0
                shift
                ;;
            -r|--run)
                if [[ -n "$2" && "$2" != -* ]]; then
                    RUN_SPECIFIC="$2"
                    RUN_ALL=0
                    shift 2
                else
                    log_error "-r 参数需要测试名称"
                    exit 1
                fi
                ;;
            -a|--all)
                RUN_ALL=1
                shift
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -c|--coverage)
                COVERAGE=1
                shift
                ;;
            -b|--build-dir)
                if [[ -n "$2" && "$2" != -* ]]; then
                    BUILD_DIR="$2"
                    shift 2
                else
                    log_error "-b 参数需要目录路径"
                    exit 1
                fi
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

# 检查构建目录
check_build_dir() {
    if [ ! -d "$BUILD_DIR" ]; then
        log_error "构建目录不存在: $BUILD_DIR"
        log_info "请先构建项目: ./scripts/build.sh"
        exit 1
    fi
    
    if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        log_warning "构建目录未配置CMake，测试可能不可用"
    fi
}

# 使用ctest运行测试
run_ctest() {
    local test_filter=""
    local ctest_args=()
    
    if [ $VERBOSE -eq 1 ]; then
        ctest_args+=("--output-on-failure")
    fi
    
    if [ -n "$PARALLEL_JOBS" ]; then
        ctest_args+=("-j" "$PARALLEL_JOBS")
    else
        # 自动检测CPU核心数
        local cpu_cores=4
        if command -v nproc >/dev/null 2>&1; then
            cpu_cores=$(nproc)
        elif command -v sysctl >/dev/null 2>&1; then
            cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
        fi
        ctest_args+=("-j" "$cpu_cores")
    fi
    
    if [ -n "$RUN_SPECIFIC" ]; then
        ctest_args+=("-R" "$RUN_SPECIFIC")
    fi
    
    log_info "运行CTest: ctest ${ctest_args[*]}"
    
    cd "$BUILD_DIR"
    
    if ! ctest "${ctest_args[@]}"; then
        log_error "CTest运行失败"
        return 1
    fi
    
    return 0
}

# 直接运行测试可执行文件
run_test_executables() {
    log_info "查找测试可执行文件..."
    
    # 查找所有测试可执行文件
    local test_executables=()
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        while IFS= read -r -d '' file; do
            test_executables+=("$file")
        done < <(find "$BUILD_DIR" -name "*test*.exe" -type f -print0)
    else
        # Linux/macOS
        while IFS= read -r -d '' file; do
            test_executables+=("$file")
        done < <(find "$BUILD_DIR" -name "*test*" -type f -executable -print0)
    fi
    
    if [ ${#test_executables[@]} -eq 0 ]; then
        log_error "未找到测试可执行文件"
        return 1
    fi
    
    log_info "找到 ${#test_executables[@]} 个测试可执行文件"
    
    # 过滤特定测试
    local filtered_executables=()
    if [ -n "$RUN_SPECIFIC" ]; then
        for test_exe in "${test_executables[@]}"; do
            local test_name=$(basename "$test_exe")
            # 移除扩展名
            test_name="${test_name%.*}"
            test_name="${test_name%.exe}"
            
            if [[ "$test_name" == $RUN_SPECIFIC ]] || [[ "$test_name" == *"$RUN_SPECIFIC"* ]]; then
                filtered_executables+=("$test_exe")
            fi
        done
        
        if [ ${#filtered_executables[@]} -eq 0 ]; then
            log_warning "未找到匹配的测试: $RUN_SPECIFIC"
            log_info "可用测试:"
            for test_exe in "${test_executables[@]}"; do
                local test_name=$(basename "$test_exe")
                test_name="${test_name%.*}"
                test_name="${test_name%.exe}"
                echo "  - $test_name"
            done
            return 1
        fi
        
        test_executables=("${filtered_executables[@]}")
    fi
    
    # 运行测试
    local passed=0
    local failed=0
    
    for test_exe in "${test_executables[@]}"; do
        local test_name=$(basename "$test_exe")
        test_name="${test_name%.*}"
        test_name="${test_name%.exe}"
        
        log_info "运行测试: $test_name"
        
        if [ $VERBOSE -eq 1 ]; then
            if ! "$test_exe"; then
                log_error "测试失败: $test_name"
                failed=$((failed + 1))
            else
                log_success "测试通过: $test_name"
                passed=$((passed + 1))
            fi
        else
            if ! "$test_exe" >/dev/null 2>&1; then
                log_error "测试失败: $test_name"
                failed=$((failed + 1))
            else
                log_success "测试通过: $test_name"
                passed=$((passed + 1))
            fi
        fi
    done
    
    echo ""
    echo "测试结果:"
    echo "  通过: $passed"
    echo "  失败: $failed"
    echo "  总计: $((passed + failed))"
    
    if [ $failed -gt 0 ]; then
        log_error "有 $failed 个测试失败"
        return 1
    fi
    
    log_success "所有测试通过"
    return 0
}

# 列出所有测试
list_tests() {
    log_info "查找测试..."
    
    # 查找测试源文件
    local test_files=()
    while IFS= read -r -d '' file; do
        test_files+=("$file")
    done < <(find "$TEST_DIR" -name "*.c" -o -name "*.cpp" -print0 2>/dev/null)
    
    if [ ${#test_files[@]} -eq 0 ]; then
        log_warning "未找到测试源文件"
        return 0
    fi
    
    log_info "找到 ${#test_files[@]} 个测试源文件:"
    
    for test_file in "${test_files[@]}"; do
        local rel_path="${test_file#$PROJECT_ROOT/}"
        local test_name=$(basename "$test_file")
        test_name="${test_name%.*}"
        
        echo "  - $test_name (${rel_path})"
    done
    
    # 查找测试可执行文件（如果存在构建目录）
    if [ -d "$BUILD_DIR" ]; then
        echo ""
        log_info "构建目录中的测试可执行文件:"
        
        local test_executables=()
        if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
            while IFS= read -r -d '' file; do
                test_executables+=("$file")
            done < <(find "$BUILD_DIR" -name "*test*.exe" -type f -print0 2>/dev/null)
        else
            while IFS= read -r -d '' file; do
                test_executables+=("$file")
            done < <(find "$BUILD_DIR" -name "*test*" -type f -executable -print0 2>/dev/null)
        fi
        
        if [ ${#test_executables[@]} -eq 0 ]; then
            echo "  未找到测试可执行文件（可能需要编译）"
        else
            for test_exe in "${test_executables[@]}"; do
                local rel_path="${test_exe#$PROJECT_ROOT/}"
                local test_name=$(basename "$test_exe")
                test_name="${test_name%.*}"
                test_name="${test_name%.exe}"
                
                echo "  - $test_name (${rel_path})"
            done
        fi
    fi
}

# 生成覆盖率报告
generate_coverage_report() {
    log_info "生成测试覆盖率报告..."
    
    # 检查必要的工具
    if ! command -v gcov >/dev/null 2>&1; then
        log_error "gcov 未安装，无法生成覆盖率报告"
        return 1
    fi
    
    if ! command -v lcov >/dev/null 2>&1; then
        log_warning "lcov 未安装，将使用基本覆盖率报告"
        # 仍可生成基本报告
    fi
    
    # 创建覆盖率目录
    local coverage_dir="${PROJECT_ROOT}/coverage"
    mkdir -p "$coverage_dir"
    
    # 运行测试以生成gcda文件
    log_info "运行测试以收集覆盖率数据..."
    run_test_executables >/dev/null 2>&1
    
    # 收集覆盖率数据
    cd "$BUILD_DIR"
    
    if command -v lcov >/dev/null 2>&1; then
        # 使用lcov生成详细报告
        lcov --capture --directory . --output-file "$coverage_dir/coverage.info"
        lcov --remove "$coverage_dir/coverage.info" "/usr/*" "*/tests/*" --output-file "$coverage_dir/coverage.filtered.info"
        genhtml "$coverage_dir/coverage.filtered.info" --output-directory "$coverage_dir/html"
        
        log_success "覆盖率报告已生成: $coverage_dir/html/index.html"
    else
        # 基本gcov报告
        log_info "生成基本gcov报告..."
        gcov -b -c -p *.c 2>/dev/null | head -50 > "$coverage_dir/gcov_report.txt"
        
        log_success "基本覆盖率报告已生成: $coverage_dir/gcov_report.txt"
    fi
    
    return 0
}

# 主函数
main() {
    parse_args "$@"
    
    echo ""
    echo "================================================================================"
    echo "                        运行 $PROJECT_NAME 测试"
    echo "================================================================================"
    echo ""
    
    # 列出测试
    if [ $LIST_TESTS -eq 1 ]; then
        list_tests
        exit 0
    fi
    
    # 检查构建目录
    check_build_dir
    
    # 生成覆盖率报告
    if [ $COVERAGE -eq 1 ]; then
        generate_coverage_report
        if [ $? -ne 0 ]; then
            exit 1
        fi
        exit 0
    fi
    
    # 运行测试
    log_info "开始运行测试..."
    
    # 优先使用ctest（如果可用）
    if command -v ctest >/dev/null 2>&1 && [ -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
        log_info "使用CTest运行测试"
        if ! run_ctest; then
            exit 1
        fi
    else
        log_info "使用测试可执行文件运行测试"
        if ! run_test_executables; then
            exit 1
        fi
    fi
    
    echo ""
    log_success "测试运行完成"
    echo ""
}

# 执行主函数
main "$@"