@echo off
chcp 65001 >nul
title SELF-LNN WSL 系统恢复工具

:: 颜色定义
set "ESC="
for /f "tokens=2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "ESC=%%a"
set "RED=%ESC%[91m"
set "GREEN=%ESC%[92m"
set "YELLOW=%ESC%[93m"
set "BLUE=%ESC%[94m"
set "RESET=%ESC%[0m"

:: 日志函数
echo %BLUE%=======================================================%RESET%
echo %BLUE%            SELF-LNN WSL 系统恢复工具               %RESET%
echo %BLUE%=======================================================%RESET%
echo.

:: 检查管理员权限
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo %RED%错误：需要管理员权限运行此脚本%RESET%
    echo 请右键点击脚本，选择"以管理员身份运行"
    pause
    exit /b 1
)

:: 主菜单
:menu
cls
echo %BLUE%=======================================================%RESET%
echo %BLUE%               WSL 系统恢复选项                    %RESET%
echo %BLUE%=======================================================%RESET%
echo.
echo %GREEN%[1]%RESET% 诊断WSL状态和网络问题
echo %GREEN%[2]%RESET% 修复WSL网络连接（最常见问题）
echo %GREEN%[3]%RESET% 重置WSL实例（保留数据）
echo %GREEN%[4]%RESET% 完全重建WSL（全新安装）
echo %GREEN%[5]%RESET% 修复文件系统错误
echo %GREEN%[6]%RESET% 设置SELF-LNN部署环境
echo %GREEN%[7]%RESET% 查看恢复日志和报告
echo %GREEN%[8]%RESET% 退出
echo.
set /p choice="请选择操作 (1-8): "

if "%choice%"=="1" goto diagnose
if "%choice%"=="2" goto fix_network
if "%choice%"=="3" goto reset_wsl
if "%choice%"=="4" goto rebuild_wsl
if "%choice%"=="5" goto fix_fs
if "%choice%"=="6" goto setup_slnn
if "%choice%"=="7" goto show_logs
if "%choice%"=="8" goto exit_script
goto menu

:: 1. 诊断WSL状态
:diagnose
echo.
echo %BLUE%[1] 诊断WSL状态和网络问题%RESET%
echo %YELLOW%=======================================================%RESET%
echo.

:: 检查WSL状态
echo %GREEN%1. 检查WSL版本和状态...%RESET%
wsl --version
echo.

:: 检查WSL运行状态
echo %GREEN%2. 检查WSL运行状态...%RESET%
wsl --status
echo.

:: 检查已安装的发行版
echo %GREEN%3. 检查已安装的WSL发行版...%RESET%
wsl -l -v
echo.

:: 检查网络适配器
echo %GREEN%4. 检查WSL网络适配器...%RESET%
powershell -Command "Get-NetAdapter -Name *WSL* -ErrorAction SilentlyContinue | Format-Table Name, Status, LinkSpeed"
echo.

:: 检查防火墙规则
echo %GREEN%5. 检查防火墙规则...%RESET%
powershell -Command "Get-NetFirewallRule -DisplayName *WSL* -ErrorAction SilentlyContinue | Format-Table DisplayName, Enabled, Direction"
echo.

:: 生成诊断报告
set "report_file=wsl_diagnosis_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%.txt"
(
    echo ========================================
    echo WSL诊断报告
    echo 生成时间: %date% %time%
    echo ========================================
    echo.
    echo [系统信息]
    systeminfo | findstr /B /C:"OS 名称" /C:"OS 版本" /C:"系统类型"
    echo.
    echo [WSL信息]
    wsl --version
    echo.
    wsl --status
    echo.
    echo [WSL发行版]
    wsl -l -v
    echo.
    echo [网络适配器]
    powershell -Command "Get-NetAdapter -Name *WSL* -ErrorAction SilentlyContinue | Format-Table Name, Status, LinkSpeed"
    echo.
    echo [防火墙规则]
    powershell -Command "Get-NetFirewallRule -DisplayName *WSL* -ErrorAction SilentlyContinue | Format-Table DisplayName, Enabled, Direction"
) > "%report_file%"

echo %GREEN%诊断报告已保存到: %report_file%%RESET%
echo.
pause
goto menu

:: 2. 修复WSL网络连接
:fix_network
echo.
echo %BLUE%[2] 修复WSL网络连接%RESET%
echo %YELLOW%=======================================================%RESET%
echo.
echo %YELLOW%正在修复WSL网络连接，这可能需要几分钟...%RESET%
echo.

:: 步骤1：关闭WSL
echo %GREEN%步骤1: 关闭所有WSL实例...%RESET%
wsl --shutdown
timeout /t 3 /nobreak >nul

:: 步骤2：重置网络栈
echo %GREEN%步骤2: 重置网络栈...%RESET%
netsh winsock reset
netsh int ip reset all
ipconfig /release
ipconfig /renew
ipconfig /flushdns

:: 步骤3：重启网络服务
echo %GREEN%步骤3: 重启网络服务...%RESET%
net stop winnat
net start winnat

:: 步骤4：修复WSL网络配置
echo %GREEN%步骤4: 修复WSL网络配置...%RESET%
powershell -Command "
    # 检查并修复WSL网络适配器
    \$adapter = Get-NetAdapter -Name '*WSL*' -ErrorAction SilentlyContinue
    if (\$adapter) {
        Write-Host '找到WSL网络适配器: ' \$adapter.Name
        Enable-NetAdapter -Name \$adapter.Name -Confirm:\$false
    } else {
        Write-Host '未找到WSL网络适配器，可能需要重启WSL'
    }
    
    # 设置防火墙规则
    New-NetFirewallRule -DisplayName 'WSL Inbound' -Direction Inbound -InterfaceAlias '*WSL*' -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName 'WSL Outbound' -Direction Outbound -InterfaceAlias '*WSL*' -Action Allow -ErrorAction SilentlyContinue
"

:: 步骤5：重新启动WSL
echo %GREEN%步骤5: 重新启动WSL...%RESET%
wsl --shutdown
timeout /t 2 /nobreak >nul

echo.
echo %GREEN%网络修复完成！%RESET%
echo 现在尝试启动WSL：wsl
echo.
pause
goto menu

:: 3. 重置WSL实例
:reset_wsl
echo.
echo %BLUE%[3] 重置WSL实例（保留数据）%RESET%
echo %YELLOW%=======================================================%RESET%
echo.
echo %YELLOW%警告：此操作将重置WSL配置但保留用户数据%RESET%
echo.
set /p confirm="确定要继续吗？(y/n): "
if /i not "%confirm%"=="y" goto menu

:: 获取发行版名称
echo %GREEN%获取WSL发行版信息...%RESET%
for /f "tokens=1" %%i in ('wsl -l -q') do set "distro=%%i"
if "%distro%"=="" (
    echo %RED%未找到WSL发行版%RESET%
    pause
    goto menu
)

echo 找到发行版: %distro%
echo.

:: 创建备份
set "backup_dir=%USERPROFILE%\WSL-Backups"
if not exist "%backup_dir%" mkdir "%backup_dir%"
set "backup_file=%backup_dir%\%distro%_backup_%date:~0,4%%date:~5,2%%date:~8,2%.tar"

echo %GREEN%创建备份: %backup_file%%RESET%
wsl --export %distro% "%backup_file%"
if %errorLevel% neq 0 (
    echo %RED%备份失败！请检查WSL状态%RESET%
    pause
    goto menu
)

:: 注销并重新导入
echo %GREEN%注销WSL发行版...%RESET%
wsl --unregister %distro%

echo %GREEN%重新导入WSL发行版...%RESET%
wsl --import %distro% "%USERPROFILE%\WSL-New" "%backup_file%" --version 2

:: 设置默认用户
echo %GREEN%设置默认用户...%RESET%
%distro% config --default-user root

echo.
echo %GREEN%WSL重置完成！%RESET%
echo 备份文件: %backup_file%
echo 现在可以使用命令启动：wsl
echo.
pause
goto menu

:: 4. 完全重建WSL
:rebuild_wsl
echo.
echo %BLUE%[4] 完全重建WSL（全新安装）%RESET%
echo %YELLOW%=======================================================%RESET%
echo.
echo %RED%警告：此操作将完全删除当前WSL实例和数据%RESET%
echo %YELLOW%建议先备份重要数据%RESET%
echo.
set /p confirm="确定要完全重建WSL吗？(y/n): "
if /i not "%confirm%"=="y" goto menu

:: 步骤1：完全卸载
echo %GREEN%步骤1: 完全卸载WSL...%RESET%
wsl --unregister Ubuntu
wsl --unregister Ubuntu-22.04
wsl --unregister Ubuntu-24.04

:: 步骤2：清理旧文件
echo %GREEN%步骤2: 清理旧文件...%RESET%
if exist "%USERPROFILE%\AppData\Local\Packages\CanonicalGroupLimited*" (
    rmdir /s /q "%USERPROFILE%\AppData\Local\Packages\CanonicalGroupLimited*" 2>nul
)

:: 步骤3：重新安装
echo %GREEN%步骤3: 重新安装WSL...%RESET%
wsl --install -d Ubuntu-24.04

echo.
echo %GREEN%WSL重建完成！%RESET%
echo 安装完成后需要重启计算机
echo.
pause
goto menu

:: 5. 修复文件系统错误
:fix_fs
echo.
echo %BLUE%[5] 修复文件系统错误%RESET%
echo %YELLOW%=======================================================%RESET%
echo.
echo %GREEN%此操作需要启动WSL到恢复模式...%RESET%
echo.

:: 尝试启动WSL到恢复模式
echo 尝试启动WSL恢复模式...
wsl --shutdown
timeout /t 2 /nobreak >nul

:: 创建修复脚本
set "fix_script=%TEMP%\wsl_fix_fs.sh"
(
    echo #!/bin/bash
    echo echo "正在检查文件系统..."
    echo 
    echo # 检查根文件系统
    echo if [ -b /dev/sda1 ]; then
    echo     echo "检查 /dev/sda1..."
    echo     fsck -f /dev/sda1
    echo fi
    echo 
    echo if [ -b /dev/sdb1 ]; then
    echo     echo "检查 /dev/sdb1..."
    echo     fsck -f /dev/sdb1
    echo fi
    echo 
    echo # 检查磁盘空间
    echo echo "检查磁盘空间..."
    echo df -h
    echo 
    echo # 检查inode使用
    echo echo "检查inode使用..."
    echo df -i
    echo 
    echo echo "文件系统检查完成"
) > "%fix_script%"

:: 执行修复
echo 正在执行文件系统检查...
wsl -d Ubuntu --user root bash -c "bash %fix_script:\=\\%"

del "%fix_script%" 2>nul

echo.
echo %GREEN%文件系统检查完成！%RESET%
echo.
pause
goto menu

:: 6. 设置SELF-LNN部署环境
:setup_slnn
echo.
echo %BLUE%[6] 设置SELF-LNN部署环境%RESET%
echo %YELLOW%=======================================================%RESET%
echo.
echo %GREEN%正在设置SELF-LNN部署环境...%RESET%
echo.

:: 创建部署脚本
set "deploy_script=%TEMP%\setup_slnn.sh"
(
    echo #!/bin/bash
    echo 
    echo # SELF-LNN 环境设置脚本
    echo echo "=== 设置SELF-LNN部署环境 ==="
    echo 
    echo # 更新系统
    echo echo "1. 更新系统包..."
    echo apt update && apt upgrade -y
    echo 
    echo # 安装基础依赖
    echo echo "2. 安装基础依赖..."
    echo apt install -y build-essential cmake git python3 python3-pip
    echo 
    echo # 创建项目目录
    echo echo "3. 创建项目目录..."
    echo mkdir -p /opt/slnn
    echo mkdir -p /opt/slnn/{bin,lib,config,data,logs,models}
    echo 
    echo # 设置权限
    echo echo "4. 设置权限..."
    echo chmod 755 /opt/slnn
    echo 
    echo # 安装Python依赖
    echo echo "5. 安装Python依赖..."
    echo pip3 install flask flask-cors psutil requests
    echo 
    echo echo "=== 环境设置完成 ==="
    echo echo "项目目录: /opt/slnn"
    echo echo "现在可以部署SELF-LNN项目"
) > "%deploy_script%"

:: 执行部署脚本
echo 正在执行环境设置脚本...
wsl -d Ubuntu --user root bash -c "bash %deploy_script:\=\\%"

del "%deploy_script%" 2>nul

echo.
echo %GREEN%SELF-LNN部署环境设置完成！%RESET%
echo 现在可以运行部署脚本：scripts\deploy.bat
echo.
pause
goto menu

:: 7. 查看恢复日志和报告
:show_logs
echo.
echo %BLUE%[7] 查看恢复日志和报告%RESET%
echo %YELLOW%=======================================================%RESET%
echo.
echo %GREEN%可用的日志文件：%RESET%
echo.

dir "%USERPROFILE%\WSL-Backups\*.txt" /b 2>nul
dir "wsl_diagnosis_*.txt" /b 2>nul

echo.
set /p log_file="输入要查看的日志文件名（或按回车返回）: "
if "%log_file%"=="" goto menu

if not exist "%log_file%" (
    echo %RED%文件不存在: %log_file%%RESET%
    pause
    goto show_logs
)

echo.
echo %BLUE%日志内容: %log_file%%RESET%
echo %YELLOW%=======================================================%RESET%
type "%log_file%"
echo %YELLOW%=======================================================%RESET%
echo.
pause
goto show_logs

:: 8. 退出
:exit_script
echo.
echo %GREEN%感谢使用SELF-LNN WSL恢复工具%RESET%
echo 更多帮助请参考 docs\WSL_网络故障排除.md
echo.
pause
exit /b 0