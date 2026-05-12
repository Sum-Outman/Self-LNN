@echo off
setlocal enabledelayedexpansion

REM SELF-LNN Windows快速启动脚本
REM 自动检查依赖项、构建项目并运行测试

set PROJECT_NAME=SELF-LNN
set PROJECT_VERSION=0.1.0
set "PROJECT_ROOT=%~dp0.."
set BUILD_DIR=%PROJECT_ROOT%\build
set INSTALL_DIR=%PROJECT_ROOT%\install
set CONFIG_DIR=%PROJECT_ROOT%\config
set DATA_DIR=%PROJECT_ROOT%\data
set MODELS_DIR=%PROJECT_ROOT%\models
set VERIFY_SCRIPT=%PROJECT_ROOT%\scripts\verify_project.py
set RUN_TESTS_SCRIPT=%PROJECT_ROOT%\scripts\run_tests.bat

REM 颜色定义（Windows控制台）
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
  set "DEL=%%a"
)

call :ColorInit
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM 日志函数
:log_info
    echo [INFO] %*
    goto :eof

:log_success
    call :ColorPrint %GREEN% "[SUCCESS] %*"
    goto :eof

:log_warning
    call :ColorPrint %YELLOW% "[WARNING] %*"
    goto :eof

:log_error
    call :ColorPrint %RED% "[ERROR] %*"
    goto :eof

:ColorPrint
    echo %DEL% >nul
    <nul set /p ".=%DEL%%1%2%DEL%%NC%"
    echo.
    goto :eof

:ColorInit
    for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
        set "DEL=%%a"
    )
    goto :eof

REM 检查命令是否存在
:check_command
    set "cmd_name=%~1"
    where %cmd_name% >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        for /f "delims=" %%i in ('where %cmd_name%') do (
            call :log_info "%cmd_name% 已安装: %%i"
        )
        exit /b 0
    ) else (
        call :log_error "%cmd_name% 未安装"
        exit /b 1
    )

REM 检查依赖项
:check_dependencies
    call :log_info "检查系统依赖项..."
    
    call :check_command cmake
    if errorlevel 1 (
        call :log_error "CMake 未安装"
        call :log_info "请安装CMake: https://cmake.org/download/"
        exit /b 1
    )
    
    call :check_command cl
    if errorlevel 1 (
        call :check_command gcc
        if errorlevel 1 (
            call :check_command g++
            if errorlevel 1 (
                call :log_error "C/C++编译器未安装"
                call :log_info "请安装Visual Studio Build Tools或MinGW"
                call :log_info "Visual Studio: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022"
                call :log_info "MinGW: https://sourceforge.net/projects/mingw-w64/"
                exit /b 1
            )
        )
    )
    
    call :check_command python
    if errorlevel 1 (
        call :check_command python3
        if errorlevel 1 (
            call :log_warning "Python未安装，某些功能可能受限"
        )
    )
    
    call :log_success "所有依赖项检查通过"
    exit /b 0

REM 检查项目结构
:check_project_structure
    call :log_info "检查项目结构..."
    
    if not exist "%PROJECT_ROOT%\CMakeLists.txt" (
        call :log_error "项目根目录缺少CMakeLists.txt"
        exit /b 1
    )
    
    if not exist "%PROJECT_ROOT%\README.md" (
        call :log_warning "项目根目录缺少README.md"
    )
    
    if not exist "%BUILD_DIR%" (
        call :log_info "创建构建目录..."
        mkdir "%BUILD_DIR%"
    )
    
    if not exist "%INSTALL_DIR%" (
        call :log_info "创建安装目录..."
        mkdir "%INSTALL_DIR%"
    )
    
    call :log_success "项目结构检查完成"
    exit /b 0

REM 配置项目
:configure_project
    call :log_info "配置项目..."
    
    cd "%BUILD_DIR%"
    
    REM 检查是否已配置
    if exist CMakeCache.txt (
        call :log_info "项目已配置，跳过配置步骤"
        exit /b 0
    )
    
    REM 配置CMake
    call :log_info "运行CMake配置..."
    
    REM 尝试不同的生成器
    cmake "%PROJECT_ROOT%" -G "Visual Studio 17 2022" -A x64
    if errorlevel 1 (
        call :log_info "尝试使用默认生成器..."
        cmake "%PROJECT_ROOT%"
        if errorlevel 1 (
            call :log_error "CMake配置失败"
            exit /b 1
        )
    )
    
    call :log_success "项目配置完成"
    exit /b 0

REM 构建项目
:build_project
    call :log_info "构建项目..."
    
    if not exist "%BUILD_DIR%\CMakeCache.txt" (
        call :log_error "项目未配置，请先运行配置步骤"
        exit /b 1
    )
    
    cd "%BUILD_DIR%"
    
    REM 检测CPU核心数
    set CPU_COUNT=4
    for /f "tokens=2 delims==" %%i in ('wmic cpu get NumberOfCores /value') do (
        if not "%%i"=="" set CPU_COUNT=%%i
    )
    
    call :log_info "使用 %CPU_COUNT% 个并行任务进行构建..."
    
    REM 构建项目
    cmake --build . --config Release --target install -- /m:%CPU_COUNT%
    if errorlevel 1 (
        call :log_error "构建失败"
        exit /b 1
    )
    
    call :log_success "项目构建完成"
    exit /b 0

REM 运行测试
:run_tests
    call :log_info "运行测试..."
    
    if not exist "%RUN_TESTS_SCRIPT%" (
        call :log_warning "测试脚本不存在: %RUN_TESTS_SCRIPT%"
        call :log_info "跳过测试步骤"
        exit /b 0
    )
    
    call "%RUN_TESTS_SCRIPT%"
    if errorlevel 1 (
        call :log_warning "测试运行失败"
        exit /b 1
    )
    
    call :log_success "测试运行完成"
    exit /b 0

REM 验证项目
:verify_project
    call :log_info "验证项目完整性..."
    
    if not exist "%VERIFY_SCRIPT%" (
        call :log_warning "验证脚本不存在: %VERIFY_SCRIPT%"
        call :log_info "跳过验证步骤"
        exit /b 0
    )
    
    python "%VERIFY_SCRIPT%"
    if errorlevel 1 (
        call :log_error "项目验证失败"
        exit /b 1
    )
    
    call :log_success "项目验证通过"
    exit /b 0

REM 显示总结
:show_summary
    echo.
    echo ================================================================================
    echo                     SELF-LNN 快速启动完成
    echo ================================================================================
    echo.
    echo 项目名称: %PROJECT_NAME%
    echo 版本: %PROJECT_VERSION%
    echo.
    echo 已完成的步骤:
    echo   1. 依赖项检查
    echo   2. 项目结构检查
    echo   3. 项目配置
    echo   4. 项目构建
    echo   5. 测试运行
    echo   6. 项目验证
    echo.
    echo 构建输出:
    echo   - 可执行文件: %INSTALL_DIR%\bin
    echo   - 库文件: %INSTALL_DIR%\lib
    echo   - 头文件: %INSTALL_DIR%\include
    echo.
    echo 下一步:
    echo   1. 运行部署脚本: scripts\deploy.bat
    echo   2. 启动服务: scripts\run_tests.bat --list
    echo   3. 查看文档: README.md
    echo.
    echo ================================================================================
    exit /b 0

REM 主流程
:main
    echo.
    echo ================================================================================
    echo                     %PROJECT_NAME% v%PROJECT_VERSION% 快速启动
    echo ================================================================================
    echo.
    
    REM 检查依赖项
    call :check_dependencies
    if errorlevel 1 exit /b 1
    
    REM 检查项目结构
    call :check_project_structure
    if errorlevel 1 exit /b 1
    
    REM 配置项目
    call :configure_project
    if errorlevel 1 exit /b 1
    
    REM 构建项目
    call :build_project
    if errorlevel 1 exit /b 1
    
    REM 运行测试
    call :run_tests
    if errorlevel 1 (
        call :log_warning "测试步骤出现问题，但继续执行"
    )
    
    REM 验证项目
    call :verify_project
    if errorlevel 1 exit /b 1
    
    REM 显示总结
    call :show_summary
    
    call :log_success "%PROJECT_NAME% 快速启动完成！"
    echo.
    pause
    exit /b 0

REM 解析命令行参数
set "ARG_HELP="
set "ARG_SKIP_TESTS="
set "ARG_SKIP_VERIFY="
set "ARG_VERBOSE="

:parse_args
    if "%~1"=="" goto :run_main
    
    if /i "%~1"=="-h" (
        set "ARG_HELP=1"
        shift
        goto :parse_args
    )
    
    if /i "%~1"=="--help" (
        set "ARG_HELP=1"
        shift
        goto :parse_args
    )
    
    if /i "%~1"=="--skip-tests" (
        set "ARG_SKIP_TESTS=1"
        shift
        goto :parse_args
    )
    
    if /i "%~1"=="--skip-verify" (
        set "ARG_SKIP_VERIFY=1"
        shift
        goto :parse_args
    )
    
    if /i "%~1"=="--verbose" (
        set "ARG_VERBOSE=1"
        shift
        goto :parse_args
    )
    
    shift
    goto :parse_args

:show_help
    echo 用法: %~n0 [选项]
    echo.
    echo 选项:
    echo   -h, --help          显示此帮助信息
    echo   --skip-tests        跳过测试步骤
    echo   --skip-verify       跳过验证步骤
    echo   --verbose           显示详细输出
    echo.
    echo 示例:
    echo   %~n0                 运行完整快速启动
    echo   %~n0 --skip-tests    跳过测试步骤
    echo   %~n0 --verbose       显示详细输出
    echo.
    exit /b 0

:run_main
    if defined ARG_HELP (
        call :show_help
        exit /b 0
    )
    
    if defined ARG_SKIP_TESTS (
        set "RUN_TESTS_SCRIPT="
    )
    
    if defined ARG_SKIP_VERIFY (
        set "VERIFY_SCRIPT="
    )
    
    call :main
    exit /b %ERRORLEVEL%