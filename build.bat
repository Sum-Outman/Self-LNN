@echo off
REM ============================================================
REM SELF-LNN v1.6.0 Windows编译脚本
REM 支持: Visual Studio 2022 / MinGW-w64 / CMake+Ninja
REM ============================================================

setlocal enabledelayedexpansion

echo ================================================================
echo   SELF-LNN 工程编译脚本 (Windows)
echo   版本: v1.6.0-engineering
echo ================================================================
echo.

REM 检查CMake是否可用
where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] CMake未安装或不在PATH中。
    echo 请从 https://cmake.org/download/ 下载安装CMake。
    exit /b 1
)

REM 设置默认构建目录
set BUILD_DIR=build
set GENERATOR=
set CONFIG=Release

REM 检测可用的生成器
echo [检测] 正在扫描可用的构建工具链...

REM 优先级: Visual Studio 2022 > MinGW > Ninja > 默认
where ninja >nul 2>&1
if %ERRORLEVEL% equ 0 (
    where cl >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "GENERATOR=Ninja"
        echo [检测] 发现: Ninja + MSVC编译器
    ) else (
        where gcc >nul 2>&1
        if %ERRORLEVEL% equ 0 (
            set "GENERATOR=Ninja"
            echo [检测] 发现: Ninja + GCC编译器
        )
    )
)

REM 如果没有Ninja但找到MSVC，使用VS生成器
if "%GENERATOR%"=="" (
    where cl >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "GENERATOR=Visual Studio 17 2022"
        echo [检测] 发现: Visual Studio 2022
    ) else (
        where gcc >nul 2>&1
        if %ERRORLEVEL% equ 0 (
            set "GENERATOR=MinGW Makefiles"
            echo [检测] 发现: MinGW-w64
        ) else (
            echo [错误] 未找到可用的编译器!
            echo 请安装以下任一工具链:
            echo   - Visual Studio 2022 (含C++桌面开发)
            echo   - MinGW-w64 (gcc)
            echo   - Ninja + MSVC
            exit /b 1
        )
    )
)

echo [构建] 使用生成器: %GENERATOR%
echo [构建] 构建目录: %BUILD_DIR%
echo.

REM 创建构建目录
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

REM CMake配置
echo [CMake] 正在配置项目...
if "%GENERATOR%"=="Ninja" (
    cmake .. -G "%GENERATOR%" -DCMAKE_BUILD_TYPE=%CONFIG% -DBUILD_TESTS=ON -DENABLE_GPU=ON -DSELFLNN_ENABLE_PURE_C_SIM=ON
) else (
    cmake .. -G "%GENERATOR%" -DBUILD_TESTS=ON -DENABLE_GPU=ON -DSELFLNN_ENABLE_PURE_C_SIM=ON
)

if %ERRORLEVEL% neq 0 (
    echo [错误] CMake配置失败!
    cd ..
    exit /b 1
)

echo.
echo [编译] 正在编译项目...

if "%GENERATOR%"=="Visual Studio 17 2022" (
    cmake --build . --config %CONFIG% --parallel
) else if "%GENERATOR%"=="MinGW Makefiles" (
    cmake --build . --parallel
    REM MinGW不需要--config，始终是定义的类型
) else (
    cmake --build . --config %CONFIG% --parallel
)

if %ERRORLEVEL% neq 0 (
    echo [错误] 编译失败!
    cd ..
    exit /b 1
)

cd ..

echo.
echo ================================================================
echo   编译成功!
echo   可执行文件: build\bin\selflnn.exe
echo   启动命令: build\bin\selflnn.exe --port 8080
echo   测试程序: build\bin\lnn_verify.exe
echo ================================================================
exit /b 0
