@echo off
REM 代码质量检查脚本（Windows版本）
REM 执行静态分析、代码格式检查等

setlocal enabledelayedexpansion

REM 项目根目录
set "PROJECT_ROOT=%~dp0.."
set "BUILD_DIR=%PROJECT_ROOT%\build"

echo 开始代码质量检查...

REM 检查clang-format是否安装
where clang-format >nul 2>nul
if %errorlevel% equ 0 (
    echo 执行代码格式检查...
    
    REM 查找所有C和H文件
    for /r "%PROJECT_ROOT%\src" %%f in (*.c *.h) do (
        echo 检查格式: %%f
        clang-format -style=file -n -Werror "%%f"
        if !errorlevel! neq 0 (
            echo 格式检查失败: %%f
            exit /b 1
        )
    )
    
    echo 代码格式检查通过！
) else (
    echo 警告: clang-format未安装，跳过格式检查
)

REM 检查cppcheck是否安装
where cppcheck >nul 2>nul
if %errorlevel% equ 0 (
    echo 执行静态分析...
    
    REM 基本检查
    cppcheck ^
        --enable=warning,performance,portability,style ^
        --suppress=missingIncludeSystem ^
        --suppress=unusedFunction ^
        --std=c11 ^
        -I "%PROJECT_ROOT%\include" ^
        -I "%PROJECT_ROOT%\src" ^
        "%PROJECT_ROOT%\src" 2> cppcheck_report.txt
    
    if exist cppcheck_report.txt (
        for %%F in (cppcheck_report.txt) do if %%~zF gtr 0 (
            echo 静态分析发现以下问题:
            type cppcheck_report.txt
            echo 详细信息已保存到 cppcheck_report.txt
        ) else (
            del cppcheck_report.txt
            echo 静态分析通过！
        )
    ) else (
        echo 静态分析通过！
    )
) else (
    echo 警告: cppcheck未安装，跳过静态分析
)

REM 检查构建目录和单元测试
if exist "%BUILD_DIR%" (
    if exist "%BUILD_DIR%\CTestTestfile.cmake" (
        echo 运行单元测试...
        
        pushd "%BUILD_DIR%"
        where ctest >nul 2>nul
        if %errorlevel% equ 0 (
            ctest --output-on-failure
            if !errorlevel! equ 0 (
                echo 单元测试通过！
            ) else (
                echo 单元测试失败！
                exit /b 1
            )
        ) else (
            echo 警告: ctest未找到，跳过单元测试
        )
        popd
    ) else (
        echo 提示: CTest测试文件不存在
    )
) else (
    echo 提示: 构建目录不存在，请先运行构建脚本
)

echo 代码质量检查完成！
echo.
echo 建议:
echo   1. 定期运行代码质量检查
echo   2. 修复所有静态分析警告
echo   3. 保持代码复杂度在合理范围内
echo   4. 确保所有测试通过

exit /b 0