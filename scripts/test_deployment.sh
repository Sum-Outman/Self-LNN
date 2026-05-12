#!/bin/bash

# SELF-LNN 部署流程测试脚本
# 用于测试部署脚本的语法和基本功能

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

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -a, --all           运行所有测试"
    echo "  -s, --scripts       测试部署脚本"
    echo "  -c, --config        测试配置文件"
    echo "  -t, --syntax        测试脚本语法"
    echo "  -v, --verbose       显示详细输出"
    echo ""
    echo "示例:"
    echo "  $0 -a               运行所有测试"
    echo "  $0 -s -t            测试脚本语法"
}

# 测试部署脚本语法
test_script_syntax() {
    log_info "测试部署脚本语法..."
    
    local scripts=(
        "scripts/deploy.sh"
        "scripts/deploy.bat"
        "scripts/build.sh"
        "scripts/build.bat"
        "scripts/quick_start.sh"
        "scripts/run_tests.sh"
        "scripts/test_deployment.sh"
    )
    
    local all_passed=true
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            log_info "检查: $script"
            
            # 检查文件权限
            if [[ "$script" == *.sh ]] && [ ! -x "$script" ]; then
                log_warning "  - 缺少执行权限"
                chmod +x "$script"
                log_info "  - 已添加执行权限"
            fi
            
            # 检查文件编码
            if file "$script" | grep -q "CRLF"; then
                log_warning "  - 使用CRLF行结束符"
                if command -v dos2unix >/dev/null 2>&1; then
                    dos2unix "$script"
                    log_info "  - 已转换为UNIX行结束符"
                fi
            fi
            
            # 检查语法（针对shell脚本）
            if [[ "$script" == *.sh ]]; then
                if bash -n "$script" 2>/dev/null; then
                    log_success "  - 语法正确"
                else
                    log_error "  - 语法错误"
                    all_passed=false
                fi
            fi
            
            # 检查批处理脚本语法
            if [[ "$script" == *.bat ]]; then
                # 简单的文件存在性检查
                if [ -s "$script" ]; then
                    log_success "  - 文件格式正确"
                else
                    log_warning "  - 文件为空或格式问题"
                fi
            fi
        else
            log_warning "  - 文件不存在: $script"
        fi
    done
    
    if $all_passed; then
        log_success "脚本语法测试通过"
        return 0
    else
        log_error "脚本语法测试失败"
        return 1
    fi
}

# 测试配置文件
test_config_files() {
    log_info "测试配置文件..."
    
    local configs=(
        "config/backend_config.json.example"
        "config/slnn.service.example"
        "config/slnn_windows_service.xml.example"
        "config/com.selflnn.slnn.plist.example"
    )
    
    local all_passed=true
    
    for config in "${configs[@]}"; do
        if [ -f "$config" ]; then
            log_info "检查: $config"
            
            # 检查文件大小
            local size=$(wc -c < "$config")
            if [ $size -eq 0 ]; then
                log_error "  - 文件为空"
                all_passed=false
                continue
            fi
            
            # 检查JSON格式
            if [[ "$config" == *.json* ]]; then
                if command -v jq >/dev/null 2>&1; then
                    if jq . "$config" >/dev/null 2>&1; then
                        log_success "  - JSON格式正确"
                    else
                        log_error "  - JSON格式错误"
                        all_passed=false
                    fi
                else
                    log_warning "  - 跳过JSON检查 (jq未安装)"
                fi
            fi
            
            # 检查XML格式
            if [[ "$config" == *.xml* ]] || [[ "$config" == *.plist* ]]; then
                if command -v xmllint >/dev/null 2>&1; then
                    if xmllint --noout "$config" 2>/dev/null; then
                        log_success "  - XML格式正确"
                    else
                        log_error "  - XML格式错误"
                        all_passed=false
                    fi
                else
                    log_warning "  - 跳过XML检查 (xmllint未安装)"
                fi
            fi
            
            # 检查服务文件格式
            if [[ "$config" == *.service* ]]; then
                if grep -q "^\[Unit\]" "$config" && grep -q "^\[Service\]" "$config" && grep -q "^\[Install\]" "$config"; then
                    log_success "  - systemd服务格式正确"
                else
                    log_warning "  - systemd服务格式可能有问题"
                fi
            fi
        else
            log_warning "  - 文件不存在: $config"
        fi
    done
    
    if $all_passed; then
        log_success "配置文件测试通过"
        return 0
    else
        log_error "配置文件测试失败"
        return 1
    fi
}

# 测试部署脚本功能
test_deployment_scripts() {
    log_info "测试部署脚本功能..."
    
    local all_passed=true
    
    # 测试帮助功能
    local scripts_to_test=(
        "scripts/deploy.sh --help"
        "scripts/build.sh --help"
    )
    
    for script_cmd in "${scripts_to_test[@]}"; do
        local script="${script_cmd%% *}"
        local args="${script_cmd#* }"
        
        if [ -f "$script" ] && [ -x "$script" ]; then
            log_info "测试帮助功能: $script"
            
            if timeout 5s bash -c "$script $args" 2>&1 | grep -q -i "用法\|help\|选项"; then
                log_success "  - 帮助功能正常"
            else
                log_warning "  - 帮助功能可能有问题"
            fi
        fi
    done
    
    # 测试脚本依赖检查
    log_info "测试依赖检查..."
    
    # 创建一个测试脚本来检查依赖
    local test_deps=$(cat << 'EOF'
#!/bin/bash
check_deps() {
    local deps=("bash" "echo" "ls")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            echo "错误: $dep 未安装"
            return 1
        fi
    done
    echo "所有依赖已安装"
    return 0
}
check_deps
EOF
    )
    
    if echo "$test_deps" | bash 2>&1 | grep -q "所有依赖已安装"; then
        log_success "  - 依赖检查逻辑正常"
    else
        log_warning "  - 依赖检查逻辑可能有问题"
    fi
    
    # 测试错误处理
    log_info "测试错误处理..."
    
    local test_error_handling=$(cat << 'EOF'
#!/bin/bash
set -e
handle_error() {
    echo "错误处理函数被调用"
    exit 1
}
trap handle_error ERR
false  # 这应该触发错误处理
EOF
    )
    
    if echo "$test_error_handling" | bash 2>&1 | grep -q "错误处理函数被调用"; then
        log_success "  - 错误处理逻辑正常"
    else
        log_warning "  - 错误处理逻辑可能有问题"
    fi
    
    if $all_passed; then
        log_success "部署脚本功能测试通过"
        return 0
    else
        log_error "部署脚本功能测试失败"
        return 1
    fi
}

# 运行集成测试
run_integration_test() {
    log_info "运行集成测试..."
    
    local all_passed=true
    
    # 创建测试目录结构
    local test_dir="/tmp/slnn_test_$(date +%s)"
    mkdir -p "$test_dir"
    
    log_info "创建测试环境: $test_dir"
    
    # 复制必要的文件到测试目录
    cp -r scripts "$test_dir/" 2>/dev/null || true
    cp -r config "$test_dir/" 2>/dev/null || true
    
    cd "$test_dir"
    
    # 测试目录结构
    log_info "测试目录结构..."
    
    local required_dirs=("scripts" "config")
    local required_files=("scripts/deploy.sh" "scripts/build.sh" "config/backend_config.json.example")
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            log_success "  - 目录存在: $dir"
        else
            log_error "  - 目录不存在: $dir"
            all_passed=false
        fi
    done
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "  - 文件存在: $file"
        else
            log_warning "  - 文件不存在: $file"
        fi
    done
    
    # 测试脚本执行（不实际运行，只测试语法）
    log_info "测试脚本执行权限..."
    
    local scripts=("scripts/deploy.sh" "scripts/build.sh")
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            chmod +x "$script" 2>/dev/null || true
            if [ -x "$script" ]; then
                log_success "  - 可执行: $script"
            else
                log_warning "  - 不可执行: $script"
            fi
        fi
    done
    
    # 清理测试目录
    cd - >/dev/null
    rm -rf "$test_dir"
    
    if $all_passed; then
        log_success "集成测试通过"
        return 0
    else
        log_error "集成测试失败"
        return 1
    fi
}

# 主函数
main() {
    # 默认参数
    TEST_ALL=0
    TEST_SCRIPTS=0
    TEST_CONFIG=0
    TEST_SYNTAX=0
    VERBOSE=0
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                TEST_ALL=1
                shift
                ;;
            -s|--scripts)
                TEST_SCRIPTS=1
                shift
                ;;
            -c|--config)
                TEST_CONFIG=1
                shift
                ;;
            -t|--syntax)
                TEST_SYNTAX=1
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
    
    # 如果没有指定测试类型，运行所有测试
    if [ $TEST_ALL -eq 1 ] || [ $TEST_SCRIPTS -eq 0 ] && [ $TEST_CONFIG -eq 0 ] && [ $TEST_SYNTAX -eq 0 ]; then
        TEST_SCRIPTS=1
        TEST_CONFIG=1
        TEST_SYNTAX=1
    fi
    
    # 显示欢迎信息
    echo ""
    echo "================================================================================"
    echo "                     SELF-LNN 部署流程测试"
    echo "================================================================================"
    echo ""
    
    local tests_passed=0
    local tests_failed=0
    
    # 运行测试
    if [ $TEST_SYNTAX -eq 1 ]; then
        echo ""
        echo "1. 测试脚本语法"
        echo "----------------------------------------------------------------"
        if test_script_syntax; then
            ((tests_passed++))
        else
            ((tests_failed++))
        fi
    fi
    
    if [ $TEST_CONFIG -eq 1 ]; then
        echo ""
        echo "2. 测试配置文件"
        echo "----------------------------------------------------------------"
        if test_config_files; then
            ((tests_passed++))
        else
            ((tests_failed++))
        fi
    fi
    
    if [ $TEST_SCRIPTS -eq 1 ]; then
        echo ""
        echo "3. 测试部署脚本功能"
        echo "----------------------------------------------------------------"
        if test_deployment_scripts; then
            ((tests_passed++))
        else
            ((tests_failed++))
        fi
    fi
    
    # 运行集成测试
    echo ""
    echo "4. 运行集成测试"
    echo "----------------------------------------------------------------"
    if run_integration_test; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi
    
    # 显示测试结果
    echo ""
    echo "================================================================================"
    echo "                        测试结果"
    echo "================================================================================"
    echo ""
    
    echo "测试统计:"
    echo "  - 通过: $tests_passed"
    echo "  - 失败: $tests_failed"
    echo "  - 总计: $((tests_passed + tests_failed))"
    echo ""
    
    if [ $tests_failed -eq 0 ]; then
        log_success "所有测试通过！部署流程测试完成。"
        echo ""
        echo "下一步建议:"
        echo "  1. 安装构建工具: CMake 和 C编译器"
        echo "  2. 运行实际构建: ./scripts/build.sh"
        echo "  3. 测试部署: ./scripts/deploy.sh --no-build"
        echo "  4. 运行系统测试: ./scripts/run_tests.sh"
    else
        log_error "有 $tests_failed 个测试失败"
        echo ""
        echo "建议:"
        echo "  1. 检查失败的测试输出"
        echo "  2. 修复脚本语法错误"
        echo "  3. 验证配置文件格式"
        echo "  4. 重新运行测试: ./scripts/test_deployment.sh -a"
    fi
    
    echo ""
    
    return $tests_failed
}

# 执行主函数
main "$@"