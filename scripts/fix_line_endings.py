#!/usr/bin/env python3
"""
SELF-LNN 脚本文件行结束符修复工具
将Windows CRLF (\r\n) 转换为Unix LF (\n)
同时修复脚本执行权限
"""

import os
import sys
import stat
import glob

def check_file_format(filepath):
    """检查文件行结束符格式"""
    with open(filepath, 'rb') as f:
        content = f.read()
    
    crlf_count = content.count(b'\r\n')
    cr_count = content.count(b'\r') - crlf_count  # 单独的CR
    lf_count = content.count(b'\n') - crlf_count  # 单独的LF
    
    return {
        'path': filepath,
        'size': len(content),
        'crlf': crlf_count,
        'cr': cr_count,
        'lf': lf_count,
        'has_crlf': crlf_count > 0 or cr_count > 0
    }

def convert_to_unix(filepath):
    """将文件转换为Unix格式(LF)"""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # 替换所有CRLF和单独的CR为LF
        content = content.replace(b'\r\n', b'\n')
        content = content.replace(b'\r', b'\n')
        
        with open(filepath, 'wb') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"  转换失败: {e}")
        return False

def set_executable(filepath):
    """设置文件为可执行"""
    try:
        # 获取当前权限
        current_mode = os.stat(filepath).st_mode
        
        # 添加执行权限 (所有者、组、其他)
        new_mode = current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        
        os.chmod(filepath, new_mode)
        return True
    except Exception as e:
        print(f"  设置权限失败: {e}")
        return False

def check_shebang(filepath):
    """检查文件是否有正确的shebang"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
        
        return first_line.startswith('#!')
    except:
        return False

def process_files():
    """处理所有脚本文件"""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    
    # 需要处理的文件模式
    patterns = [
        os.path.join(scripts_dir, '*.sh'),
        os.path.join(scripts_dir, '*.py'),
        os.path.join(project_root, '*.sh'),
        os.path.join(project_root, '*.py'),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    # 去重
    all_files = list(set(all_files))
    all_files.sort()
    
    print("=" * 70)
    print("SELF-LNN 脚本文件格式检查与修复")
    print("=" * 70)
    print()
    
    if not all_files:
        print("未找到脚本文件")
        return
    
    # 检查阶段
    file_stats = []
    for filepath in all_files:
        stats = check_file_format(filepath)
        file_stats.append(stats)
    
    # 显示检查结果
    print("文件检查结果:")
    print("-" * 70)
    
    needs_fix = []
    for stats in file_stats:
        status = "✓" if not stats['has_crlf'] else "✗"
        format_type = "Unix (LF)" if not stats['has_crlf'] else "Windows (CRLF)"
        
        print(f"  {status} {os.path.basename(stats['path']):30} {format_type:20} "
              f"大小: {stats['size']:,} bytes")
        
        if stats['has_crlf']:
            needs_fix.append(stats['path'])
    
    print()
    
    # 修复阶段
    if needs_fix:
        print("需要修复的文件:")
        print("-" * 70)
        
        for filepath in needs_fix:
            print(f"  修复: {os.path.basename(filepath)}")
            
            # 备份原文件
            backup_path = filepath + '.bak'
            try:
                import shutil
                shutil.copy2(filepath, backup_path)
                print(f"    已备份: {os.path.basename(backup_path)}")
            except Exception as e:
                print(f"    备份失败: {e}")
            
            # 转换格式
            if convert_to_unix(filepath):
                print(f"    格式转换完成")
            
            # 检查并设置执行权限
            if filepath.endswith('.sh') or filepath.endswith('.py'):
                if set_executable(filepath):
                    print(f"    设置执行权限完成")
            
            # 检查shebang
            if filepath.endswith('.sh'):
                if not check_shebang(filepath):
                    print(f"    ⚠️  警告: 缺少shebang (#!)")
            
            print()
    else:
        print("所有文件格式正确，无需修复")
        print()
    
    # 设置执行权限阶段
    print("设置脚本执行权限:")
    print("-" * 70)
    
    for filepath in all_files:
        if filepath.endswith('.sh'):
            if set_executable(filepath):
                print(f"  ✓ {os.path.basename(filepath)}")
    
    print()
    
    # 最终验证
    print("最终验证:")
    print("-" * 70)
    
    all_good = True
    for filepath in all_files:
        stats = check_file_format(filepath)
        
        if stats['has_crlf']:
            print(f"  ✗ {os.path.basename(filepath)}: 仍有CRLF问题")
            all_good = False
        elif filepath.endswith('.sh') and not check_shebang(filepath):
            print(f"  ⚠️  {os.path.basename(filepath)}: 缺少shebang")
        else:
            print(f"  ✓ {os.path.basename(filepath)}: 格式正确")
    
    print()
    print("=" * 70)
    
    if all_good:
        print("✅ 所有脚本文件格式检查通过")
    else:
        print("❌ 部分文件仍有问题，请手动检查")
    
    # 显示使用说明
    print()
    print("使用说明:")
    print("  1. Linux/macOS下直接运行: python3 scripts/fix_line_endings.py")
    print("  2. Windows下运行: python scripts/fix_line_endings.py")
    print("  3. 在提交代码前运行此脚本确保跨平台兼容性")
    print()
    print("注意事项:")
    print("  - 备份文件以 .bak 扩展名保存")
    print("  - 修复后建议运行测试: ./scripts/test_deployment.sh -t")
    print("  - 在Linux/macOS环境下验证脚本可执行性")

def main():
    """主函数"""
    try:
        process_files()
        return 0
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())