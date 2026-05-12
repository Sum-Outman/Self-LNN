@echo off
setlocal enabledelayedexpansion

REM SELF-LNN Windows构建脚本
REM 适用于Windows环境的构建脚本

set PROJECT_NAME=SELF-LNN
set PROJECT_ROOT=%~dp0..
set BUILD_DIR=%PROJECT_ROOT%\build
set INSTALL_DIR=%PROJECT_ROOT%\install

REM 颜色定义
set "RED="
set "GREEN="
set "YELLOW="
set "BLUE="
set "NC="

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

REM 显示帮助
:show_help
    echo 用法: %~n0 [选项]
    echo.
    echo 选项:
    echo   -h, --help          显示此帮助信息
    echo   -c, --clean         清理构建目录
    echo   -d, --debug         构建调试版本 (默认: Release)
    echo   -j N                并行构建任务数 (默认: 自动检测)
    echo   -t, --test          构建后运行测试
    echo   -i, --install       构建后安装到系统
    echo   -v, --verbose       显示详细输出
    echo.
    echo 示例:
    echo   %~n0                  构建Release版本
    echo   %~n0 -d -j4           使用4个任务构建调试版本
    echo   %~n0 -c               清理构建目录
    echo   %~n0 -t               构建并运行测试
    exit /b 0

REM 解析命令行参数
set BUILD_TYPE=Release
set CLEAN_BUILD=0
set RUN_TESTS=0
set RUN_INSTALL=0
set VERBOSE=0
set PARALLEL_JOBS=

:parse_args
if "%~1"=="" goto :parse_done
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help
if "%~1"=="-c" (
    set CLEAN_BUILD=1
    shift
    goto parse_args
)
if "%~1"=="--clean" (
    set CLEAN_BUILD=1
    shift
    goto parse_args
)
if "%~1"=="-d" (
    set BUILD_TYPE=Debug
    shift
    goto parse_args
)
if "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto parse_args
)
if "%~1"=="-j" (
    if not "%~2"=="" (
        echo %~2 | findstr /r "^[0-9][0-9]*$" >nul
        if not errorlevel 1 (
            set PARALLEL_JOBS=%~2
            shift
            shift
            goto parse_args
        )
    )
    echo [ERROR] -j 参数需要数字
    exit /b 1
)
if "%~1"=="-t" (
    set RUN_TESTS=1
    shift
    goto parse_args
)
if "%~1"=="--test" (
    set RUN_TESTS=1
    shift
    goto parse_args
)
if "%~1"=="-i" (
    set RUN_INSTALL=1
    shift
    goto parse_args
)
if "%~1"=="--install" (
    set RUN_INSTALL=1
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

REM 检查依赖项
:check_dependencies
    echo [INFO] 检查构建依赖项...
    
    REM 检查CMake
    where cmake >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] CMake 未安装
        echo [INFO] 请从 cmake.org 下载并安装CMake
        exit /b 1
    )
    
    REM 检查C编译器
    where cl >nul 2>&1
    if errorlevel 1 (
        where gcc >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] 未找到C编译器 (cl/gcc)
            echo [INFO] 请安装Visual Studio或MinGW-w64
            exit /b 1
        )
    )
    
    echo [SUCCESS] 构建依赖项检查通过
    exit /b 0

REM 清理构建目录
:clean_build
    echo [INFO] 清理构建目录: %BUILD_DIR%
    
    if exist "%BUILD_DIR%" (
        if exist "%BUILD_DIR%\CMakeCache.txt" (
            echo [INFO] 删除CMake缓存文件
            del /f /q "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
        )
        
        echo [INFO] 删除构建目录内容
        rmdir /s /q "%BUILD_DIR%" >nul 2>&1
        
        if not errorlevel 1 (
            echo [SUCCESS] 清理完成
        ) else (
            echo [ERROR] 清理失败
            exit /b 1
        )
    ) else (
        echo [WARNING] 构建目录不存在: %BUILD_DIR%
    )
    exit /b 0

REM 准备构建目录
:prepare_build_dir
    echo [INFO] 准备构建目录: %BUILD_DIR%
    
    if not exist "%BUILD_DIR%" (
        mkdir "%BUILD_DIR%"
        echo [INFO] 创建构建目录
    )
    
    cd /d "%BUILD_DIR%"
    exit /b 0

REM 配置CMake
:configure_cmake
    echo [INFO] 配置CMake (构建类型: %BUILD_TYPE%)...
    
    set CMAKE_OPTIONS=-DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_TESTS=ON -DBUILD_FRONTEND=ON -DBUILD_EXAMPLES=ON
    
    REM 设置生成器
    where cl >nul 2>&1
    if not errorlevel 1 (
        REM 使用Visual Studio生成器
        set CMAKE_OPTIONS=%CMAKE_OPTIONS% -G "Visual Studio 17 2022" -A x64
    ) else (
        where gcc >nul 2>&1
        if not errorlevel 1 (
            REM 使用MinGW生成器
            set CMAKE_OPTIONS=%CMAKE_OPTIONS% -G "MinGW Makefiles"
        )
    )
    
    echo [INFO] 运行: cmake %PROJECT_ROOT% %CMAKE_OPTIONS%
    
    if %VERBOSE%==1 (
        cmake "%PROJECT_ROOT%" %CMAKE_OPTIONS%
    ) else (
        cmake "%PROJECT_ROOT%" %CMAKE_OPTIONS% > cmake_config.log 2>&1
        if errorlevel 1 (
            echo [ERROR] CMake配置失败，查看日志: %BUILD_DIR%\cmake_config.log
            type "%BUILD_DIR%\cmake_config.log" | tail -20
            exit /b 1
        )
    )
    
    echo [SUCCESS] CMake配置成功
    exit /b 0

REM 构建项目
:build_project
    echo [INFO] 构建项目 (类型: %BUILD_TYPE%)...
    
    REM 设置并行任务数
    if not "%PARALLEL_JOBS%"=="" (
        set PARALLEL_ARG=--parallel %PARALLEL_JOBS%
        echo [INFO] 使用指定的并行任务数: %PARALLEL_JOBS%
    ) else (
        REM 自动检测CPU核心数
        for /f "tokens=2 delims==" %%i in ('wmic cpu get NumberOfCores /value ^| find "="') do set CPU_CORES=%%i
        if "%CPU_CORES%"=="" set CPU_CORES=4
        set PARALLEL_ARG=--parallel %CPU_CORES%
        echo [INFO] 自动检测CPU核心数: %CPU_CORES%
    )
    
    if %VERBOSE%==1 (
        cmake --build . --config %BUILD_TYPE% %PARALLEL_ARG%
    ) else (
        cmake --build . --config %BUILD_TYPE% %PARALLEL_ARG% > build.log 2>&1
        if errorlevel 1 (
            echo [ERROR] 构建失败，查看日志: %BUILD_DIR%\build.log
            type "%BUILD_DIR%\build.log" | tail -20
            exit /b 1
        )
    )
    
    echo [SUCCESS] 项目构建成功
    exit /b 0

REM 运行测试
:run_tests
    echo [INFO] 运行测试...
    
    if not exist "CTestTestfile.cmake" (
        echo [WARNING] 未找到测试配置，跳过测试
        exit /b 0
    )
    
    if %VERBOSE%==1 (
        ctest --output-on-failure -C %BUILD_TYPE%
    ) else (
        ctest -C %BUILD_TYPE% > test.log 2>&1
        if errorlevel 1 (
            echo [WARNING] 测试失败，查看日志: %BUILD_DIR%\test.log
            type "%BUILD_DIR%\test.log" | tail -20
            exit /b 1
        )
    )
    
    echo [SUCCESS] 所有测试通过
    exit /b 0

REM 安装项目
:install_project
    echo [INFO] 安装项目...
    
    cmake --install . --config %BUILD_TYPE%
    if errorlevel 1 (
        echo [WARNING] 安装失败
        exit /b 1
    )
    
    echo [SUCCESS] 项目安装成功
    exit /b 0

REM 显示构建摘要
:show_summary
    echo.
    echo ================================================================================
    echo                          %PROJECT_NAME% 构建完成
    echo ================================================================================
    echo.
    echo 构建类型: %BUILD_TYPE%
    echo 构建目录: %BUILD_DIR%
    echo.
    
    REM 显示生成的可执行文件
    if exist "%BUILD_DIR%\tests" (
        echo 测试程序:
        dir /b "%BUILD_DIR%\tests\test_*.exe" 2>nul | head -5 | (
            setlocal enabledelayedexpansion
            set count=0
            for /f "tokens=*" %%i in ('more') do (
                echo   - %%i
                set /a count+=1
            )
            if !count!==0 echo   (无测试程序)
        )
    )
    
    if exist "%BUILD_DIR%\src" (
        echo.
        echo 主要程序:
        dir /b /s "%BUILD_DIR%\src\*.exe" 2>nul | head -5 | (
            setlocal enabledelayedexpansion
            set count=0
            for /f "tokens=*" %%i in ('more') do (
                echo   - %%i
                set /a count+=1
            )
            if !count!==0 echo   (无主要程序)
        )
    )
    
    if exist "%BUILD_DIR%\examples" (
        echo.
        echo 示例程序:
        dir /b /s "%BUILD_DIR%\examples\*.exe" 2>nul | head -5 | (
            setlocal enabledelayedexpansion
            set count=0
            for /f "tokens=*" %%i in ('more') do (
                echo   - %%i
                set /a count+=1
            )
            if !count!==0 echo   (无示例程序)
        )
    )
    
    if exist "%PROJECT_ROOT%\frontend" (
        echo.
        echo 前端文件:
        dir /b "%PROJECT_ROOT%\frontend\js\*.js" 2>nul | (
            setlocal enabledelayedexpansion
            echo   JS文件:
            for /f "tokens=*" %%i in ('more') do (
                echo     - %%i
            )
        )
        dir /b "%PROJECT_ROOT%\frontend\css\*.css" 2>nul | (
            setlocal enabledelayedexpansion
            echo   CSS文件:
            for /f "tokens=*" %%i in ('more') do (
                echo     - %%i
            )
        )
    )
    
    echo.
    echo 下一步:
    echo   1. 运行测试: %~n0 -t
    echo   2. 运行示例: %BUILD_DIR%\examples\basic_lnn_example.exe
    echo   3. 启动后端: %BUILD_DIR%\src\backend\backend_server.exe
    echo   4. 更多信息: type %PROJECT_ROOT%\README.md
    echo.
    echo ================================================================================
    exit /b 0

REM 主函数
:main
    echo.
    echo ================================================================================
    echo                         构建 %PROJECT_NAME%
    echo ================================================================================
    echo.
    
    REM 如果需要清理
    if %CLEAN_BUILD%==1 (
        call :clean_build
        REM 如果只需要清理，则退出
        if %CLEAN_BUILD%==1 if %RUN_TESTS%==0 if %RUN_INSTALL%==0 (
            exit /b 0
        )
    )
    
    REM 检查依赖项
    call :check_dependencies
    if errorlevel 1 exit /b 1
    
    REM 准备构建目录
    call :prepare_build_dir
    if errorlevel 1 exit /b 1
    
    REM 配置CMake (如果需要)
    if not exist "CMakeCache.txt" (
        call :configure_cmake
        if errorlevel 1 exit /b 1
    ) else (
        if %CLEAN_BUILD%==1 (
            call :configure_cmake
            if errorlevel 1 exit /b 1
        ) else (
            echo [INFO] 使用现有的CMake配置
        )
    )
    
    REM 构建项目
    call :build_project
    if errorlevel 1 exit /b 1
    
    REM 运行测试
    if %RUN_TESTS%==1 (
        call :run_tests
        if errorlevel 1 (
            echo [WARNING] 测试失败，但构建成功
        )
    )
    
    REM 安装项目
    if %RUN_INSTALL%==1 (
        call :install_project
        if errorlevel 1 (
            echo [WARNING] 安装失败，但构建成功
        )
    )
    
    REM 显示摘要
    call :show_summary
    
    echo.
    echo [SUCCESS] %PROJECT_NAME% 构建完成！
    echo.
    exit /b 0

REM 执行主函数
call :main %*
exit /b %errorlevel%