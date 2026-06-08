#!/bin/bash
# ============================================================
# SELF-LNN v1.6.0 Linux/macOS启动脚本
# 启动后端服务器，自动检测端口并打开前端页面
# ============================================================

set -e

PORT=8080
BIN="build/bin/selflnn"

echo "================================================================="
echo "  SELF-LNN AGI 系统启动"
echo "  版本: v1.6.0-engineering"
echo "================================================================="
echo ""

# 检查可执行文件
if [ ! -f "$BIN" ]; then
    echo "[错误] 可执行文件不存在: $BIN"
    echo "请先运行 ./build.sh 编译项目"
    exit 1
fi

# 解析命令行参数
if [ -n "$1" ]; then
    PORT="$1"
fi

echo "[启动] HTTP API端口: $PORT"
echo "[启动] WebSocket端口: 8081"
echo "[启动] 前端页面: frontend/index.html"
echo ""

echo "[启动] 正在启动SELF-LNN后端服务器..."
echo ""

# 启动后端服务器
"$BIN" --port "$PORT"

echo ""
echo "[退出] SELF-LNN系统已关闭"
