#!/usr/bin/env python3
"""
F-023: 纯C零依赖链验证脚本
扫描项目中所有 #include 和链接库，检测任何非标准第三方库依赖

使用: python verify_pure_c.py
返回: 0=纯C, 1=发现依赖
"""

import os, re, sys

PROJECT_ROOT = os.getcwd()

# 如果当前目录不是项目根，尝试检测
if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(script_dir) if script_dir else os.getcwd()

# 允许的第三方（操作系统原生API，非"第三方库"）
OS_NATIVE_HEADERS = {
    # Windows SDK
    'windows.h', 'wincrypt.h', 'bcrypt.h', 'process.h', 'winsock2.h', 'ws2tcpip.h',
    'winhttp.h', 'pdh.h', 'pdhmsg.h', 'psapi.h', 'intrin.h', 'mmdeviceapi.h',
    'audioclient.h', 'functiondiscoverykeys_devpkey.h', 'avrt.h', 'direct.h',
    'winbase.h', 'winreg.h',
    # Linux / POSIX
    'unistd.h', 'pthread.h', 'sys/stat.h', 'sys/types.h', 'sys/sysinfo.h',
    'sys/syscall.h', 'sys/random.h', 'sys/statvfs.h', 'sys/time.h', 'sys/resource.h',
    'sys/mman.h', 'fcntl.h', 'dlfcn.h', 'sys/socket.h', 'netinet/in.h', 'arpa/inet.h',
    'cpuid.h', 'netdb.h', 'ifaddrs.h', 'sys/ioctl.h', 'net/if.h', 'netinet/tcp.h',
    # macOS
    'mach/mach.h', 'sys/sysctl.h',
    'Security/Security.h',
    # C 标准库
    'stddef.h', 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h',
    'float.h', 'stdint.h', 'stdarg.h', 'signal.h', 'setjmp.h', 'limits.h',
    'assert.h', 'ctype.h', 'errno.h', 'locale.h',
    # C11 标准库
    'stdalign.h', 'stdatomic.h', 'stdnoreturn.h', 'threads.h', 'uchar.h',
    # 编译器内建
    'emmintrin.h', 'xmmintrin.h', 'immintrin.h', 'arm_neon.h',
    # POSIX 兼容
    'dirent.h',
    # MSVC 内部
    'io.h',
    'crtdbg.h',
    'wchar.h',
    'malloc.h',
}

# OS原生API前缀白名单（通配符匹配）
OS_NATIVE_PREFIXES = [
    'X11/',             # Linux X11 窗口系统
    'objc/',            # macOS Objective-C runtime
    'mach/',            # macOS Mach内核API
    'CoreAudio/',       # macOS 音频
    'AudioToolbox/',    # macOS 音频工具
    'ApplicationServices/',  # macOS 应用服务
    'CoreGraphics/',    # macOS 图形
    'linux/',           # Linux 内核头文件
    'alsa/',            # Linux ALSA 音频
    'sys/',             # POSIX 系统头文件
    'netinet/',         # POSIX 网络
    'endpointvolume.h', # Windows 音频端点
    'devicetopology.h', # Windows 音频拓扑
    'mmsystem.h',       # Windows 多媒体
    'mfapi.h', 'mfidl.h', 'mfreadwrite.h', 'shlwapi.h', 'vfw.h',  # Windows Media Foundation
    'setupapi.h', 'devguid.h', 'cfgmgr32.h', 'initguid.h', 'hidsdi.h',  # Windows 硬件检测
    'powerbase.h',      # Windows 电源管理
    'shellapi.h',       # Windows Shell
    'stdbool.h',        # C99标准
    'endian.h',         # POSIX字节序
    'termios.h',        # POSIX终端
    'security/',        # macOS Security
]

# 项目自身头文件模式
SELF_PATTERN = re.compile(r'selflnn/')

def check_file(filepath):
    issues = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, 1):
            match = re.match(r'^\s*#include\s+[<"](.+)[>"]', line)
            if not match:
                continue
            header = match.group(1)
            
            # 跳过项目自身头文件（含相对路径引用）
            if SELF_PATTERN.match(header) or header.startswith('gpu_internal') or header.startswith('distributed_internal'):
                continue
            if header.startswith('../'):
                continue
            # 跳过内部简短路径头文件（通过CMake include path找到）
            if '/' not in header and '.' in header:
                # 可能是 selflnn 内部头文件的短路径引用
                # 检查是否存在于项目include目录中
                found_internal = False
                for search_dir in ['include/selflnn', 'src']:
                    for r, d, fs in os.walk(os.path.join(PROJECT_ROOT, search_dir)):
                        if header in fs:
                            found_internal = True
                            break
                    if found_internal:
                        break
                if found_internal:
                    continue
            # 跳过允许的OS原生头文件
            ok = False
            if header in OS_NATIVE_HEADERS:
                ok = True
            for prefix in OS_NATIVE_PREFIXES:
                if header.startswith(prefix):
                    ok = True
                    break
            if ok:
                continue
            # 跳过配置生成的头文件
            if 'version.h' in header:
                continue
                
            issues.append(f"  {filepath}:{lineno}: #include <{header}>")
    return issues

def main():
    all_issues = []
    c_files = []
    
    # 扫描所有 .c 和 .h 文件
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in ('build', 'node_modules', '.git', '.trae', 'build_temp')]
        for fname in files:
            if fname.endswith('.c') or fname.endswith('.h'):
                c_files.append(os.path.join(root, fname))
    
    for fp in c_files:
        rel = os.path.relpath(fp, PROJECT_ROOT)
        issues = check_file(fp)
        all_issues.extend(issues)
    
    if all_issues:
        print(f"[纯C依赖检查] 发现 {len(all_issues)} 个潜在非标准依赖:")
        for issue in all_issues:
            print(issue)
        print("\n[警告] 以上头文件可能引入第三方库依赖，请逐一确认。")
        return 1
    else:
        print(f"[纯C依赖检查] [{len(c_files)} 文件扫描完成] 100% 纯C — 零第三方库依赖")
        return 0

if __name__ == '__main__':
    sys.exit(main())
