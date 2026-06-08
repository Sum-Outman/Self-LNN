@echo off
REM ============================================================
REM SELF-LNN v1.6.0 Windows启动脚本
REM 启动后端服务器，自动检测端口并打开前端页面
REM ============================================================

setlocal enabledelayedexpansion

set PORT=8080
set BIN=build\bin\selflnn.exe

echo ================================================================
echo   SELF-LNN AGI 系统启动
echo   版本: v1.6.0-engineering
echo ================================================================
echo.

REM 检查可执行文件
if not exist "%BIN%" (
    echo [错误] 可执行文件不存在: %BIN%
    echo 请先运行 build.bat 编译项目
    exit /b 1
)

REM 解析命令行参数
if not "%1"=="" set PORT=%1

echo [启动] HTTP API端口: %PORT%
echo [启动] WebSocket端口: 8081
echo [启动] 前端页面: frontend\index.html
echo.

echo [启动] 正在启动SELF-LNN后端服务器...
echo.

REM 启动后端服务器
"%BIN%" --port %PORT%

echo.
echo [退出] SELF-LNN系统已关闭
exit /b 0
