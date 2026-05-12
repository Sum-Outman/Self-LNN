@echo off
setlocal enabledelayedexpansion

REM SELF-LNN Windows测试运行脚本
REM 运行项目测试套件

set PROJECT_NAME=SELF-LNN
set PROJECT_ROOT=%~dp0..
set BUILD_DIR=%PROJECT_ROOT%\build
set TEST_DIR=%PROJECT_ROOT%\tests

REM 显示帮助
:show_help
    echo 用法: %~n0 [选项]
    echo.
    echo 选项:
    echo   -h, --help          显示此帮助信息
    echo   -l, --list          列出所有测试
    echo   -r, --run [测试名]   运行指定测试（支持通配符）
    echo   -a, --all           运行所有测试（默认）
    echo   -v, --verbose       显示详细输出
    echo   -b, --build-dir DIR 指定构建目录（默认: build）
    echo.
    echo 示例:
    echo   %~n0                  运行所有测试
    echo   %~n0 -l               列出所有可用测试
    echo   %~n0 -r test_core     运行核心测试
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
set RUN_ALL=1
set LIST_TESTS=0
set RUN_SPECIFIC=
set VERBOSE=0
set BUILD_DIR=%PROJECT_ROOT%\build

:parse_args
if "%~1"=="" goto :parse_done
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help
if "%~1"=="-l" (
    set LIST_TESTS=1
    set RUN_ALL=0
    shift
    goto parse_args
)
if "%~1"=="--list" (
    set LIST_TESTS=1
    set RUN_ALL=0
    shift
    goto parse_args
)
if "%~1"=="-r" (
    if not "%~2"=="" (
        set RUN_SPECIFIC=%~2
        set RUN_ALL=0
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] -r 参数需要测试名称
        exit /b 1
    )
)
if "%~1"=="--run" (
    if not "%~2"=="" (
        set RUN_SPECIFIC=%~2
        set RUN_ALL=0
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] --run 参数需要测试名称
        exit /b 1
    )
)
if "%~1"=="-a" (
    set RUN_ALL=1
    shift
    goto parse_args
)
if "%~1"=="--all" (
    set RUN_ALL=1
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
if "%~1"=="-b" (
    if not "%~2"=="" (
        set BUILD_DIR=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] -b 参数需要目录路径
        exit /b 1
    )
)
if "%~1"=="--build-dir" (
    if not "%~2"=="" (
        set BUILD_DIR=%~2
        shift
        shift
        goto parse_args
    ) else (
        echo [ERROR] --build-dir 参数需要目录路径
        exit /b 1
    )
)
echo [ERROR] 未知选项: %~1
call :show_help
exit /b 1

:parse_done

REM 检查构建目录
:check_build_dir
    echo [INFO] 检查构建目录: %BUILD_DIR%
    
    if not exist "%BUILD_DIR%" (
        echo [ERROR] 构建目录不存在: %BUILD_DIR%
        echo [INFO] 请先构建项目: scripts\build.bat
        exit /b 1
    )
    
    if not exist "%BUILD_DIR%\CMakeCache.txt" (
        echo [WARNING] 构建目录未配置CMake，测试可能不可用
    )
    exit /b 0

REM 列出所有测试
:list_tests
    echo [INFO] 查找测试...
    
    REM 查找测试源文件
    if exist "%TEST_DIR%" (
        echo [INFO] 测试源文件:
        dir /b /s "%TEST_DIR%\*.c" 2>nul | (
            setlocal enabledelayedexpansion
            set count=0
            for /f "tokens=*" %%i in ('more') do (
                set /a count+=1
                echo   - %%~ni (%%~fi)
            )
            if !count!==0 echo   (无测试源文件)
        )
    ) else (
        echo [WARNING] 测试目录不存在
    )
    
    REM 查找测试可执行文件
    if exist "%BUILD_DIR%" (
        echo.
        echo [INFO] 构建目录中的测试可执行文件:
        dir /b /s "%BUILD_DIR%\*test*.exe" 2>nul | (
            setlocal enabledelayedexpansion
            set count=0
            for /f "tokens=*" %%i in ('more') do (
                set /a count+=1
                echo   - %%~ni (%%~fi)
            )
            if !count!==0 echo   (未找到测试可执行文件，可能需要编译)
        )
    )
    exit /b 0

REM 运行测试可执行文件
:run_test_executables
    echo [INFO] 查找测试可执行文件...
    
    REM 查找所有测试可执行文件
    set TEST_COUNT=0
    set PASSED=0
    set FAILED=0
    
    for /f "delims=" %%i in ('dir /b /s "%BUILD_DIR%\*test*.exe" 2^>nul') do (
        set /a TEST_COUNT+=1
        set "TEST_EXE=%%i"
        set "TEST_NAME=%%~ni"
        
        REM 过滤特定测试
        if not "%RUN_SPECIFIC%"=="" (
            echo !TEST_NAME! | findstr /i "%RUN_SPECIFIC%" >nul
            if errorlevel 1 (
                REM 不匹配，跳过
                set /a TEST_COUNT-=1
                goto :skip_test
            )
        )
        
        echo [INFO] 运行测试: !TEST_NAME!
        
        if %VERBOSE%==1 (
            !TEST_EXE!
            if !errorlevel!==0 (
                echo [SUCCESS] 测试通过: !TEST_NAME!
                set /a PASSED+=1
            ) else (
                echo [ERROR] 测试失败: !TEST_NAME!
                set /a FAILED+=1
            )
        ) else (
            !TEST_EXE! >nul 2>&1
            if !errorlevel!==0 (
                echo [SUCCESS] 测试通过: !TEST_NAME!
                set /a PASSED+=1
            ) else (
                echo [ERROR] 测试失败: !TEST_NAME!
                set /a FAILED+=1
            )
        )
        
        :skip_test
    )
    
    if %TEST_COUNT%==0 (
        echo [ERROR] 未找到测试可执行文件
        if not "%RUN_SPECIFIC%"=="" (
            echo [INFO] 请检查测试名称: %RUN_SPECIFIC%
        )
        exit /b 1
    )
    
    echo.
    echo 测试结果:
    echo   通过: %PASSED%
    echo   失败: %FAILED%
    echo   总计: %TEST_COUNT%
    
    if %FAILED% gtr 0 (
        echo [ERROR] 有 %FAILED% 个测试失败
        exit /b 1
    )
    
    echo [SUCCESS] 所有测试通过
    exit /b 0

REM 使用ctest运行测试
:run_ctest
    echo [INFO] 使用CTest运行测试...
    
    cd /d "%BUILD_DIR%"
    
    if not exist "CTestTestfile.cmake" (
        echo [WARNING] CTest测试文件不存在，将使用可执行文件
        exit /b 0
    )
    
    set CTEST_ARGS=
    if %VERBOSE%==1 (
        set CTEST_ARGS=--output-on-failure
    )
    
    if not "%RUN_SPECIFIC%"=="" (
        set CTEST_ARGS=%CTEST_ARGS% -R %RUN_SPECIFIC%
    )
    
    ctest %CTEST_ARGS%
    if errorlevel 1 (
        echo [ERROR] CTest运行失败
        exit /b 1
    )
    
    exit /b 0

REM 主函数
:main
    echo.
    echo ================================================================================
    echo                         运行 %PROJECT_NAME% 测试
    echo ================================================================================
    echo.
    
    REM 列出测试
    if %LIST_TESTS%==1 (
        call :list_tests
        exit /b 0
    )
    
    REM 检查构建目录
    call :check_build_dir
    if errorlevel 1 exit /b 1
    
    REM 运行测试
    echo [INFO] 开始运行测试...
    
    REM 优先使用ctest（如果可用）
    where ctest >nul 2>&1
    if not errorlevel 1 (
        if exist "%BUILD_DIR%\CTestTestfile.cmake" (
            call :run_ctest
            if errorlevel 1 exit /b 1
        ) else (
            call :run_test_executables
            if errorlevel 1 exit /b 1
        )
    ) else (
        call :run_test_executables
        if errorlevel 1 exit /b 1
    )
    
    echo.
    echo [SUCCESS] 测试运行完成
    echo.
    exit /b 0

REM 执行主函数
call :main %*
exit /b %errorlevel%