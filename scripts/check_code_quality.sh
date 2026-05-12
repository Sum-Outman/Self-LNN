#!/bin/bash
# 代码质量检查脚本
# 执行静态分析、代码格式检查、复杂度分析等

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo -e "${GREEN}开始代码质量检查...${NC}"

# 检查clang-format是否安装
if command -v clang-format >/dev/null 2>&1; then
    echo -e "${GREEN}执行代码格式检查...${NC}"
    
    # 检查C文件格式
    find "${PROJECT_ROOT}/src" -name "*.c" -o -name "*.h" | while read file; do
        echo "检查格式: $file"
        clang-format -style=file -n -Werror "$file"
    done
    
    echo -e "${GREEN}代码格式检查通过！${NC}"
else
    echo -e "${YELLOW}警告: clang-format未安装，跳过格式检查${NC}"
fi

# 检查cppcheck是否安装
if command -v cppcheck >/dev/null 2>&1; then
    echo -e "${GREEN}执行静态分析...${NC}"
    
    # 基本检查
    cppcheck \
        --enable=warning,performance,portability,style \
        --suppress=missingIncludeSystem \
        --suppress=unusedFunction \
        --std=c11 \
        -I "${PROJECT_ROOT}/include" \
        -I "${PROJECT_ROOT}/src" \
        "${PROJECT_ROOT}/src" 2> cppcheck_report.txt
    
    if [ -s cppcheck_report.txt ]; then
        echo -e "${YELLOW}静态分析发现以下问题:${NC}"
        cat cppcheck_report.txt
        echo -e "${YELLOW}详细信息已保存到 cppcheck_report.txt${NC}"
    else
        echo -e "${GREEN}静态分析通过！${NC}"
        rm -f cppcheck_report.txt
    fi
else
    echo -e "${YELLOW}警告: cppcheck未安装，跳过静态分析${NC}"
fi

# 检查代码复杂度（使用pmccabe）
if command -v pmccabe >/dev/null 2>&1; then
    echo -e "${GREEN}分析代码复杂度...${NC}"
    
    # 查找所有C文件
    find "${PROJECT_ROOT}/src" -name "*.c" -exec pmccabe -v {} + > pmccabe_report.txt 2>&1 || true
    
    if [ -s pmccabe_report.txt ]; then
        echo -e "${YELLOW}代码复杂度分析结果:${NC}"
        head -20 pmccabe_report.txt
        echo -e "${YELLOW}完整报告已保存到 pmccabe_report.txt${NC}"
    fi
else
    echo -e "${YELLOW}警告: pmccabe未安装，跳过复杂度分析${NC}"
fi

# 检查内存泄漏工具（valgrind）
if command -v valgrind >/dev/null 2>&1; then
    echo -e "${GREEN}检查内存泄漏工具可用性...${NC}"
    # 这里可以添加valgrind测试
else
    echo -e "${YELLOW}提示: 可安装valgrind进行内存泄漏检测${NC}"
fi

# 运行单元测试
if [ -d "${BUILD_DIR}" ] && [ -f "${BUILD_DIR}/CTestTestfile.cmake" ]; then
    echo -e "${GREEN}运行单元测试...${NC}"
    
    cd "${BUILD_DIR}"
    if command -v ctest >/dev/null 2>&1; then
        ctest --output-on-failure
        echo -e "${GREEN}单元测试完成！${NC}"
    else
        echo -e "${YELLOW}警告: ctest未找到，跳过单元测试${NC}"
    fi
    cd - >/dev/null
else
    echo -e "${YELLOW}提示: 构建目录不存在，请先运行构建脚本${NC}"
fi

echo -e "${GREEN}代码质量检查完成！${NC}"
echo -e "${YELLOW}建议:${NC}"
echo "  1. 定期运行代码质量检查"
echo "  2. 修复所有静态分析警告"
echo "  3. 保持代码复杂度在合理范围内"
echo "  4. 确保所有测试通过"

exit 0