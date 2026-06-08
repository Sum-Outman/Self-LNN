#!/bin/bash
# ============================================================
# SELF-LNN v1.6.0 Linux/macOS编译脚本
# 支持: GCC / Clang / CMake+Ninja
# ============================================================

set -e

echo "================================================================="
echo "  SELF-LNN 工程编译脚本 (Linux/macOS)"
echo "  版本: v1.6.0-engineering"
echo "================================================================="
echo ""

# 检查CMake
if ! command -v cmake &> /dev/null; then
    echo "[错误] CMake未安装。请执行: sudo apt install cmake (Ubuntu) 或 brew install cmake (macOS)"
    exit 1
fi

# 检测可用编译器
BUILD_DIR="build"
GENERATOR="Unix Makefiles"
CONFIG="Release"

echo "[检测] 正在扫描可用的编译工具链..."

if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
    echo "[检测] 发现: Ninja构建系统"
fi

if command -v gcc &> /dev/null; then
    GCC_VER=$(gcc -dumpversion | cut -d. -f1)
    echo "[检测] 发现: GCC $GCC_VER"
elif command -v clang &> /dev/null; then
    CLANG_VER=$(clang --version | head -1)
    echo "[检测] 发现: Clang ($CLANG_VER)"
else
    echo "[错误] 未找到GCC或Clang编译器!"
    echo "请安装: sudo apt install build-essential (Ubuntu) 或 xcode-select --install (macOS)"
    exit 1
fi

echo "[构建] 使用生成器: $GENERATOR"
echo "[构建] 构建目录: $BUILD_DIR"
echo ""

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMake配置
echo "[CMake] 正在配置项目..."
cmake .. \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE="$CONFIG" \
    -DBUILD_TESTS=ON \
    -DENABLE_GPU=OFF \
    -DSELFLNN_ENABLE_PURE_C_SIM=ON \
    -DSELFLNN_ENABLE_EXTERNAL_SIM=OFF

if [ $? -ne 0 ]; then
    echo "[错误] CMake配置失败!"
    cd ..
    exit 1
fi

echo ""
echo "[编译] 正在编译项目 ($(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) 个并行任务)..."

cmake --build . --config "$CONFIG" --parallel "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

if [ $? -ne 0 ]; then
    echo "[错误] 编译失败!"
    cd ..
    exit 1
fi

cd ..

echo ""
echo "================================================================="
echo "  编译成功!"
echo "  可执行文件: build/bin/selflnn"
echo "  启动命令: ./build/bin/selflnn --port 8080"
echo "  测试程序: ./build/bin/lnn_verify"
echo "================================================================="
