@echo off
setlocal enabledelayedexpansion

REM SELF-LNN Windows部署脚本
REM 自动部署项目到目标环境

set PROJECT_NAME=SELF-LNN
set PROJECT_VERSION=0.1.0
set PROJECT_ROOT=%~dp0..
set BUILD_DIR=%PROJECT_ROOT%\build
set INSTALL_DIR=%PROJECT_ROOT%\install
set DEPLOY_DIR=%PROJECT_ROOT%\deploy
set CONFIG_DIR=%PROJECT_ROOT%\config
set DATA_DIR=%PROJECT_ROOT%\data

REM 显示帮助
:show_help
    echo 用法: %~n0 [选项]
    echo.
    echo 选项:
    echo   -h, --help          显示此帮助信息
    echo   -t, --target DIR    指定部署目标目录（默认: 当前目录\deploy）
    echo   -c, --config DIR    指定配置文件目录（默认: config）
    echo   -d, --data DIR      指定数据目录（默认: data）
    echo   -s, --service       安装为Windows服务
    echo   -p, --port PORT     指定服务端口（默认: 8081）
    echo   -f, --frontend-port PORT 前端端口（默认: 8080）
    echo   -n, --no-build      跳过构建步骤（假设已构建）
    echo   -v, --verbose       显示详细输出
    echo.
    echo 示例:
    echo   %~n0                  部署到默认目录
    echo   %~n0 -t C:\SELF-LNN  部署到指定目录
    echo   %~n0 -s -p 9000      安装为服务并指定端口
    echo.
    exit /b 0

REM 日志函数
:log_info
    echo [INFO] %*
    exit /b 0

:log_success
    echo [SUCCESS] %*
    exit /b 0

:log_warning
    echo [WARNING] %*
    exit /b 0

:log_error
    echo [ERROR] %*
    exit /b 1

REM 解析命令行参数
set DEPLOY_TARGET=%DEPLOY_DIR%
set INSTALL_SERVICE=0
set SERVICE_PORT=8081
set FRONTEND_PORT=8080
set SKIP_BUILD=0
set VERBOSE=0

:parse_args
if "%~1"=="" goto :parse_done
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help
if "%~1"=="-t" (
    if not "%~2"=="" (
        set DEPLOY_TARGET=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] -t 参数需要目录路径
        exit /b 1
    )
)
if "%~1"=="--target" (
    if not "%~2"=="" (
        set DEPLOY_TARGET=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] --target 参数需要目录路径
        exit /b 1
    )
)
if "%~1"=="-c" (
    if not "%~2"=="" (
        set CONFIG_DIR=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] -c 参数需要目录路径
        exit /b 1
    )
)
if "%~1"=="--config" (
    if not "%~2"=="" (
        set CONFIG_DIR=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] --config 参数需要目录路径
        exit /b 1
    )
)
if "%~1"=="-d" (
    if not "%~2"=="" (
        set DATA_DIR=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] -d 参数需要目录路径
        exit /b 1
    )
)
if "%~1"=="--data" (
    if not "%~2"=="" (
        set DATA_DIR=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] --data 参数需要目录路径
        exit /b 1
    )
)
if "%~1"=="-s" (
    set INSTALL_SERVICE=1
    shift
    goto parse_args
)
if "%~1"=="--service" (
    set INSTALL_SERVICE=1
    shift
    goto parse_args
)
if "%~1"=="-p" (
    if not "%~2"=="" (
        set SERVICE_PORT=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] -p 参数需要端口号
        exit /b 1
    )
)
if "%~1"=="--port" (
    if not "%~2"=="" (
        set SERVICE_PORT=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] --port 参数需要端口号
        exit /b 1
    )
)
if "%~1"=="-f" (
    if not "%~2"=="" (
        set FRONTEND_PORT=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] -f 参数需要端口号
        exit /b 1
    )
)
if "%~1"=="--frontend-port" (
    if not "%~2"=="" (
        set FRONTEND_PORT=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] --frontend-port 参数需要端口号
        exit /b 1
    )
)
if "%~1"=="-n" (
    set SKIP_BUILD=1
    shift
    goto parse_args
)
if "%~1"=="--no-build" (
    set SKIP_BUILD=1
    shift
    goto parse_args
)
if "%~1"=="-v" (
    set VERBOSE=1
    shift
    goto parse_args
)
if "%~1"=="--verbose" (
    set VERBOSE=1
    shift
    goto parse_args
)
echo [ERROR] 未知选项: %~1
call :show_help
exit /b 1

:parse_done

REM 主函数
:main
    echo.
    echo ================================================================================
    echo                         部署 %PROJECT_NAME% v%PROJECT_VERSION%
    echo ================================================================================
    echo.
    
    REM 检查项目结构
    call :check_project_structure
    if errorlevel 1 exit /b 1
    
    REM 构建项目（如果需要）
    if %SKIP_BUILD%==0 (
        call :build_project
        if errorlevel 1 exit /b 1
    )
    
    REM 准备部署目录
    call :prepare_deploy_dir
    if errorlevel 1 exit /b 1
    
    REM 复制文件
    call :copy_files
    if errorlevel 1 exit /b 1
    
    REM 创建配置文件
    call :create_config_files
    if errorlevel 1 exit /b 1
    
    REM 安装服务（如果需要）
    if %INSTALL_SERVICE%==1 (
        call :install_windows_service
        if errorlevel 1 exit /b 1
    )
    
    REM 显示部署总结
    call :show_deployment_summary
    if errorlevel 1 exit /b 1
    
    echo.
    echo [SUCCESS] %PROJECT_NAME% 部署完成！
    echo.
    exit /b 0

REM 检查项目结构
:check_project_structure
    echo [INFO] 检查项目结构...
    
    if not exist "%PROJECT_ROOT%\CMakeLists.txt" (
        echo [ERROR] 项目根目录缺少CMakeLists.txt
        exit /b 1
    )
    
    if not exist "%PROJECT_ROOT%\README.md" (
        echo [WARNING] 项目根目录缺少README.md
    )
    
    if not exist "%BUILD_DIR%" (
        echo [WARNING] 构建目录不存在，需要构建项目
    )
    
    echo [SUCCESS] 项目结构检查完成
    exit /b 0

REM 构建项目
:build_project
    echo [INFO] 构建项目...
    
    REM 检查是否已安装构建工具
    where cmake >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] CMake 未安装
        echo [INFO] 请从 cmake.org 下载并安装CMake
        exit /b 1
    )
    
    REM 运行构建脚本
    echo [INFO] 运行构建脚本...
    
    if exist "%PROJECT_ROOT%\scripts\build.bat" (
        cd /d "%PROJECT_ROOT%"
        if %VERBOSE%==1 (
            call scripts\build.bat --install
        ) else (
            call scripts\build.bat --install >nul 2>&1
        )
        
        if errorlevel 1 (
            echo [ERROR] 构建失败
            exit /b 1
        )
    ) else (
        echo [ERROR] 构建脚本不存在: scripts\build.bat
        exit /b 1
    )
    
    echo [SUCCESS] 项目构建完成
    exit /b 0

REM 准备部署目录
:prepare_deploy_dir
    echo [INFO] 准备部署目录: %DEPLOY_TARGET%
    
    REM 创建部署目录结构
    for %%d in (
        "%DEPLOY_TARGET%"
        "%DEPLOY_TARGET%\bin"
        "%DEPLOY_TARGET%\lib"
        "%DEPLOY_TARGET%\include"
        "%DEPLOY_TARGET%\config"
        "%DEPLOY_TARGET%\data"
        "%DEPLOY_TARGET%\logs"
        "%DEPLOY_TARGET%\models"
    ) do (
        if not exist %%d (
            mkdir %%d
            if errorlevel 1 (
                echo [ERROR] 创建目录失败: %%d
                exit /b 1
            )
        )
    )
    
    echo [SUCCESS] 部署目录准备完成
    exit /b 0

REM 复制文件
:copy_files
    echo [INFO] 复制文件到部署目录...
    
    REM 复制可执行文件
    if exist "%INSTALL_DIR%\bin" (
        xcopy "%INSTALL_DIR%\bin\*" "%DEPLOY_TARGET%\bin\" /E /Y /I >nul 2>&1
        if errorlevel 1 (
            echo [WARNING] 复制可执行文件时出错
        ) else (
            echo [SUCCESS] 复制可执行文件完成
        )
    )
    
    REM 复制库文件
    if exist "%INSTALL_DIR%\lib" (
        xcopy "%INSTALL_DIR%\lib\*" "%DEPLOY_TARGET%\lib\" /E /Y /I >nul 2>&1
        if errorlevel 1 (
            echo [WARNING] 复制库文件时出错
        ) else (
            echo [SUCCESS] 复制库文件完成
        )
    )
    
    REM 复制头文件
    if exist "%INSTALL_DIR%\include" (
        xcopy "%INSTALL_DIR%\include\*" "%DEPLOY_TARGET%\include\" /E /Y /I >nul 2>&1
        if errorlevel 1 (
            echo [WARNING] 复制头文件时出错
        ) else (
            echo [SUCCESS] 复制头文件完成
        )
    )
    
    REM 复制配置文件
    if exist "%CONFIG_DIR%" (
        xcopy "%CONFIG_DIR%\*" "%DEPLOY_TARGET%\config\" /E /Y /I >nul 2>&1
        if errorlevel 1 (
            echo [WARNING] 复制配置文件时出错
        ) else (
            echo [SUCCESS] 复制配置文件完成
        )
    )
    
    REM 复制数据文件
    if exist "%DATA_DIR%" (
        xcopy "%DATA_DIR%\*" "%DEPLOY_TARGET%\data\" /E /Y /I >nul 2>&1
        if errorlevel 1 (
            echo [WARNING] 复制数据文件时出错
        ) else (
            echo [SUCCESS] 复制数据文件完成
        )
    )
    
    REM 复制前端文件
    if exist "%PROJECT_ROOT%\frontend" (
        xcopy "%PROJECT_ROOT%\frontend\*" "%DEPLOY_TARGET%\frontend\" /E /Y /I >nul 2>&1
        if errorlevel 1 (
            echo [WARNING] 复制前端文件时出错
        ) else (
            echo [SUCCESS] 复制前端文件完成
        )
    )
    
    echo [SUCCESS] 文件复制完成
    exit /b 0

REM 创建配置文件
:create_config_files
    echo [INFO] 创建配置文件...
    
    REM 创建后端配置文件
    set BACKEND_CONFIG=%DEPLOY_TARGET%\config\backend_config.json
    if not exist "%BACKEND_CONFIG%" (
        echo [INFO] 创建后端配置文件...
        (
            echo {
            echo     "server": {
            echo         "host": "0.0.0.0",
            echo         "port": %SERVICE_PORT%,
            echo         "max_connections": 100,
            echo         "timeout": 30
            echo     },
            echo     "database": {
            echo         "path": "data/slnn.db",
            echo         "max_connections": 10
            echo     },
            echo     "logging": {
            echo         "level": "info",
            echo         "file": "logs/slnn.log",
            echo         "max_size": "10MB",
            echo         "backup_count": 5
            echo     },
            echo     "models": {
            echo         "path": "models",
            echo         "auto_save_interval": 300
            echo     }
            echo }
        ) > "%BACKEND_CONFIG%"
        
        if errorlevel 1 (
            echo [WARNING] 创建后端配置文件失败
        ) else (
            echo [SUCCESS] 后端配置文件已创建: %BACKEND_CONFIG%
        )
    )
    
    REM 创建启动脚本
    set START_SCRIPT=%DEPLOY_TARGET%\start_slnn.bat
    echo [INFO] 创建启动脚本...
    (
        echo @echo off
        echo setlocal enabledelayedexpansion
        echo.
        echo REM SELF-LNN 启动脚本
        echo echo 启动 SELF-LNN AGI 系统...
        echo.
        echo REM 设置环境变量
        echo set PROJECT_ROOT=%~dp0
        echo set PATH=!PROJECT_ROOT!\bin;!PATH!
        echo.
        echo REM 启动后端服务器
        echo echo [INFO] 启动后端服务器...
        echo start "SELF-LNN Backend" /B cmd /c "!PROJECT_ROOT!\bin\backend_server.exe --config !PROJECT_ROOT!\config\backend_config.json"
        echo.
        echo REM 启动前端服务器（如果存在）
        echo if exist "!PROJECT_ROOT!\frontend\index.html" (
        echo     echo [INFO] 启动前端服务器...
        echo     start "SELF-LNN Frontend" /B cmd /c "python -m http.server %FRONTEND_PORT% --directory !PROJECT_ROOT!\frontend"
        echo )
        echo.
        echo echo [SUCCESS] SELF-LNN 系统已启动
        echo echo 后端地址: http://localhost:%SERVICE_PORT%
        echo echo 前端地址: http://localhost:%FRONTEND_PORT%
        echo pause
    ) > "%START_SCRIPT%"
    
    if errorlevel 1 (
        echo [WARNING] 创建启动脚本失败
    ) else (
        echo [SUCCESS] 启动脚本已创建: %START_SCRIPT%
    )
    
    REM 创建停止脚本
    set STOP_SCRIPT=%DEPLOY_TARGET%\stop_slnn.bat
    (
        echo @echo off
        echo setlocal enabledelayedexpansion
        echo.
        echo REM SELF-LNN 停止脚本
        echo echo 停止 SELF-LNN AGI 系统...
        echo.
        echo REM 停止后端服务器
        echo taskkill /F /IM backend_server.exe /T ^>nul 2^>^&1
        echo.
        echo REM 停止前端服务器
        echo for /f "tokens=2" %%i in ('tasklist ^| findstr http.server') do taskkill /F /PID %%i ^>nul 2^>^&1
        echo.
        echo echo [SUCCESS] SELF-LNN 系统已停止
        echo pause
    ) > "%STOP_SCRIPT%"
    
    if errorlevel 1 (
        echo [WARNING] 创建停止脚本失败
    ) else (
        echo [SUCCESS] 停止脚本已创建: %STOP_SCRIPT%
    )
    
    echo [SUCCESS] 配置文件创建完成
    exit /b 0

REM 安装Windows服务
:install_windows_service
    echo [INFO] 安装Windows服务...
    
    REM 检查是否以管理员身份运行
    net session >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] 请以管理员身份运行此脚本以安装服务
        exit /b 1
    )
    
    REM 检查服务是否已存在
    sc query "SELF-LNN" >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] 服务已存在，正在停止并删除...
        sc stop "SELF-LNN" >nul 2>&1
        sc delete "SELF-LNN" >nul 2>&1
        timeout /t 2 /nobreak >nul
    )
    
    REM 创建服务
    echo [INFO] 创建SELF-LNN服务...
    
    set SERVICE_BIN=%DEPLOY_TARGET%\bin\backend_server.exe
    if not exist "%SERVICE_BIN%" (
        echo [ERROR] 服务可执行文件不存在: %SERVICE_BIN%
        exit /b 1
    )
    
    REM 创建服务配置文件
    set SERVICE_CONFIG=%DEPLOY_TARGET%\config\service_config.json
    (
        echo {
        echo     "service_name": "SELF-LNN",
        echo     "display_name": "SELF-LNN AGI System",
        echo     "description": "全液态神经网络AGI大模型系统服务",
        echo     "executable": "%SERVICE_BIN%",
        echo     "arguments": "--config %DEPLOY_TARGET%\config\backend_config.json",
        echo     "working_directory": "%DEPLOY_TARGET%",
        echo     "start_type": "automatic",
        echo     "log_on_as": "LocalSystem"
        echo }
    ) > "%SERVICE_CONFIG%"
    
    REM 使用sc创建服务
    sc create "SELF-LNN" binPath= "\"%SERVICE_BIN%\" --config \"%DEPLOY_TARGET%\config\backend_config.json\"" DisplayName= "SELF-LNN AGI System" start= auto >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] 创建服务失败
        exit /b 1
    )
    
    REM 设置服务描述
    sc description "SELF-LNN" "全液态神经网络AGI大模型系统服务，提供多模态认知、自主决策、机器人控制等功能。" >nul 2>&1
    
    REM 启动服务
    echo [INFO] 启动SELF-LNN服务...
    sc start "SELF-LNN" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] 启动服务失败，但服务已创建
    ) else (
        echo [SUCCESS] 服务启动成功
    )
    
    echo [SUCCESS] Windows服务安装完成
    exit /b 0

REM 显示部署总结
:show_deployment_summary
    echo.
    echo ================================================================================
    echo                         部署总结
    echo ================================================================================
    echo.
    echo 项目名称: %PROJECT_NAME%
    echo 版本: %PROJECT_VERSION%
    echo 部署目录: %DEPLOY_TARGET%
    echo.
    echo 包含内容:
    echo   - 可执行文件: %DEPLOY_TARGET%\bin\
    echo   - 配置文件: %DEPLOY_TARGET%\config\
    echo   - 数据文件: %DEPLOY_TARGET%\data\
    echo   - 模型文件: %DEPLOY_TARGET%\models\
    echo   - 日志文件: %DEPLOY_TARGET%\logs\
    echo.
    
    if %INSTALL_SERVICE%==1 (
        echo Windows服务:
        echo   - 服务名称: SELF-LNN
        echo   - 显示名称: SELF-LNN AGI System
        echo   - 启动类型: 自动
        echo   - 状态: 已安装并启动
        echo.
    )
    
    echo 启动方式:
    echo   1. 手动启动: 运行 %DEPLOY_TARGET%\start_slnn.bat
    echo   2. 停止系统: 运行 %DEPLOY_TARGET%\stop_slnn.bat
    echo.
    echo 访问地址:
    echo   - 后端API: http://localhost:%SERVICE_PORT%/api/v1/
    echo   - 前端界面: http://localhost:%FRONTEND_PORT%/
    echo.
    echo 配置文件:
    echo   - 后端配置: %DEPLOY_TARGET%\config\backend_config.json
    echo.
    echo 下一步:
    echo   1. 确保防火墙允许端口 %SERVICE_PORT% 和 %FRONTEND_PORT%
    echo   2. 根据需要修改配置文件
    echo   3. 查看日志文件: %DEPLOY_TARGET%\logs\slnn.log
    echo.
    echo ================================================================================
    exit /b 0

REM 执行主函数
call :main %*
exit /b %errorlevel%