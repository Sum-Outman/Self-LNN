#!/usr/bin/env python3
"""
SELF-LNN Python环境检查工具
检查Python版本、依赖包和系统兼容性
"""

import sys
import platform
import subprocess
import importlib.metadata
import importlib.util
import os
import json
from typing import Dict, List, Tuple, Optional

class PythonEnvironmentChecker:
    """Python环境检查器"""
    
    def __init__(self):
        self.results = {
            "system": {},
            "python": {},
            "dependencies": {},
            "compatibility": {},
            "recommendations": []
        }
        
    def check_system_info(self):
        """检查系统信息"""
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_build": platform.python_build(),
            "python_compiler": platform.python_compiler(),
            "python_branch": platform.python_branch(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        }
        self.results["system"] = info
        return info
    
    def check_python_version(self):
        """检查Python版本"""
        version_info = sys.version_info
        requirements = {
            "min_major": 3,
            "min_minor": 8,
            "recommended_major": 3,
            "recommended_minor": 10
        }
        
        version_check = {
            "current": f"{version_info.major}.{version_info.minor}.{version_info.micro}",
            "version_tuple": (version_info.major, version_info.minor, version_info.micro),
            "requirements": requirements,
            "meets_minimum": version_info.major >= requirements["min_major"] and 
                            version_info.minor >= requirements["min_minor"],
            "meets_recommended": version_info.major >= requirements["recommended_major"] and 
                                version_info.minor >= requirements["recommended_minor"],
            "is_64bit": sys.maxsize > 2**32
        }
        
        self.results["python"] = version_check
        return version_check
    
    def check_dependencies(self):
        """检查依赖包"""
        required_packages = {
            "flask": {"min_version": "2.0.0", "required": True},
            "flask-cors": {"min_version": "3.0.0", "required": True},
            "numpy": {"min_version": "1.20.0", "required": False},
            "scipy": {"min_version": "1.7.0", "required": False},
            "psutil": {"min_version": "5.8.0", "required": True},
            "requests": {"min_version": "2.25.0", "required": True},
            "sqlite3": {"min_version": "3.0.0", "required": True, "builtin": True},
            "json": {"min_version": "1.0", "required": True, "builtin": True},
        }
        
        dependencies = {}
        
        for package, req_info in required_packages.items():
            package_info = {"required": req_info["required"]}
            
            if req_info.get("builtin", False):
                # 内置模块
                spec = importlib.util.find_spec(package)
                package_info["installed"] = spec is not None
                package_info["version"] = "built-in"
                package_info["meets_requirements"] = True
            else:
                # 第三方包
                try:
                    version = importlib.metadata.version(package)
                    package_info["installed"] = True
                    package_info["version"] = version
                    
                    # 检查版本是否符合要求
                    if "min_version" in req_info:
                        package_info["meets_requirements"] = self._compare_versions(
                            version, req_info["min_version"]
                        )
                        package_info["min_required"] = req_info["min_version"]
                    else:
                        package_info["meets_requirements"] = True
                        
                except importlib.metadata.PackageNotFoundError:
                    package_info["installed"] = False
                    package_info["version"] = None
                    package_info["meets_requirements"] = False
                    if "min_version" in req_info:
                        package_info["min_required"] = req_info["min_version"]
            
            dependencies[package] = package_info
        
        self.results["dependencies"] = dependencies
        return dependencies
    
    def _compare_versions(self, v1: str, v2: str) -> bool:
        """比较版本号 v1 >= v2 返回True"""
        def parse_version(v: str):
            parts = []
            for part in v.split('.'):
                try:
                    parts.append(int(part))
                except ValueError:
                    # 处理非数字部分
                    for i, c in enumerate(part):
                        if not c.isdigit():
                            parts.append(int(part[:i]) if part[:i] else 0)
                            break
                    else:
                        parts.append(int(part))
            return parts
        
        v1_parts = parse_version(v1)
        v2_parts = parse_version(v2)
        
        # 补全长度
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts += [0] * (max_len - len(v1_parts))
        v2_parts += [0] * (max_len - len(v2_parts))
        
        for p1, p2 in zip(v1_parts, v2_parts):
            if p1 > p2:
                return True
            elif p1 < p2:
                return False
        
        return True  # 相等
    
    def check_compatibility(self):
        """检查兼容性"""
        python_info = self.results["python"]
        deps_info = self.results["dependencies"]
        
        issues = []
        warnings = []
        
        # 检查Python版本
        if not python_info["meets_minimum"]:
            issues.append(f"Python版本 {python_info['current']} 低于最低要求 3.8")
        elif not python_info["meets_recommended"]:
            warnings.append(f"Python版本 {python_info['current']} 低于推荐版本 3.10")
        
        # 检查64位
        if not python_info["is_64bit"]:
            warnings.append("使用32位Python，建议使用64位版本以获得更好性能")
        
        # 检查依赖
        missing_required = []
        outdated_packages = []
        
        for package, info in deps_info.items():
            if info["required"] and not info["installed"]:
                missing_required.append(package)
            elif info["installed"] and not info.get("meets_requirements", True):
                outdated_packages.append(f"{package} (当前: {info['version']}, 需要: {info.get('min_required', '未知')})")
        
        if missing_required:
            issues.append(f"缺少必需包: {', '.join(missing_required)}")
        
        if outdated_packages:
            warnings.append(f"包版本过低: {', '.join(outdated_packages)}")
        
        # 检查虚拟环境
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        compatibility = {
            "has_issues": len(issues) > 0,
            "has_warnings": len(warnings) > 0,
            "issues": issues,
            "warnings": warnings,
            "in_virtualenv": in_venv,
            "python_path": sys.executable,
            "path": sys.path
        }
        
        self.results["compatibility"] = compatibility
        return compatibility
    
    def generate_recommendations(self):
        """生成建议"""
        recommendations = []
        compat = self.results["compatibility"]
        python_info = self.results["python"]
        deps_info = self.results["dependencies"]
        
        # Python版本建议
        if not python_info["meets_minimum"]:
            recommendations.append("升级Python到3.8或更高版本")
        elif not python_info["meets_recommended"]:
            recommendations.append("建议升级Python到3.10或更高版本以获得更好性能")
        
        # 虚拟环境建议
        if not compat["in_virtualenv"]:
            recommendations.append("建议使用虚拟环境隔离项目依赖")
        
        # 依赖安装建议
        missing_packages = []
        for package, info in deps_info.items():
            if info["required"] and not info["installed"]:
                missing_packages.append(package)
        
        if missing_packages:
            recommendations.append(f"安装缺少的包: pip install {' '.join(missing_packages)}")
        
        # 性能建议
        if not python_info["is_64bit"]:
            recommendations.append("建议安装64位Python版本")
        
        # SELF-LNN特定建议
        recommendations.append("对于SELF-LNN项目，确保有足够的RAM（至少8GB）")
        recommendations.append("建议使用SSD存储以获得更好的I/O性能")
        recommendations.append("配置合适的交换空间（swap space）")
        
        self.results["recommendations"] = recommendations
        return recommendations
    
    def run_all_checks(self):
        """运行所有检查"""
        print("=" * 70)
        print("SELF-LNN Python环境检查")
        print("=" * 70)
        print()
        
        # 运行检查
        self.check_system_info()
        self.check_python_version()
        self.check_dependencies()
        self.check_compatibility()
        self.generate_recommendations()
        
        # 显示结果
        self.print_results()
        
        # 保存报告
        self.save_report()
        
        return self.results
    
    def print_results(self):
        """打印检查结果"""
        # 系统信息
        print("📋 系统信息:")
        print(f"  平台: {self.results['system']['platform']}")
        print(f"  Python: {self.results['python']['current']}")
        print(f"  架构: {'64位' if self.results['python']['is_64bit'] else '32位'}")
        print(f"  虚拟环境: {'是' if self.results['compatibility']['in_virtualenv'] else '否'}")
        print()
        
        # 依赖状态
        print("📦 依赖检查:")
        deps = self.results['dependencies']
        installed_count = sum(1 for info in deps.values() if info['installed'])
        required_count = sum(1 for info in deps.values() if info['required'])
        required_installed = sum(1 for info in deps.values() if info['required'] and info['installed'])
        
        print(f"  总依赖: {len(deps)}")
        print(f"  已安装: {installed_count}")
        print(f"  必需依赖: {required_count}")
        print(f"  必需已安装: {required_installed}")
        print()
        
        # 详细依赖信息
        for package, info in deps.items():
            status = "✓" if info['installed'] else "✗"
            version = info['version'] if info['version'] else "未安装"
            req_status = ""
            
            if info['required']:
                req_status = "必需"
                if not info['installed']:
                    req_status += " ❌ 缺失"
                elif not info.get('meets_requirements', True):
                    req_status += " ⚠️ 版本低"
                else:
                    req_status += " ✓"
            else:
                req_status = "可选"
            
            print(f"  {status} {package:15} {version:15} [{req_status}]")
        print()
        
        # 问题和警告
        compat = self.results['compatibility']
        if compat['issues']:
            print("❌ 发现问题:")
            for issue in compat['issues']:
                print(f"  • {issue}")
            print()
        
        if compat['warnings']:
            print("⚠️  警告:")
            for warning in compat['warnings']:
                print(f"  • {warning}")
            print()
        
        if not compat['issues'] and not compat['warnings']:
            print("✅ 所有检查通过！")
            print()
        
        # 建议
        if self.results['recommendations']:
            print("💡 建议:")
            for rec in self.results['recommendations']:
                print(f"  • {rec}")
            print()
    
    def save_report(self):
        """保存检查报告"""
        import datetime
        
        report_dir = "reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"python_env_check_{timestamp}.json")
        
        # 转换非JSON可序列化对象
        serializable_results = json.loads(json.dumps(self.results, default=str))
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 报告已保存到: {report_file}")
        print()
    
    def check_ubuntu_specific(self):
        """检查Ubuntu特定项目（如果在Ubuntu中运行）"""
        ubuntu_checks = {}
        
        # 检查是否为Ubuntu
        if platform.system() == "Linux":
            try:
                # 检查Ubuntu版本
                with open('/etc/os-release', 'r') as f:
                    content = f.read()
                    if 'Ubuntu' in content:
                        ubuntu_checks['is_ubuntu'] = True
                        
                        # 提取版本信息
                        import re
                        version_match = re.search(r'VERSION_ID="([^"]+)"', content)
                        if version_match:
                            ubuntu_checks['ubuntu_version'] = version_match.group(1)
                    else:
                        ubuntu_checks['is_ubuntu'] = False
            except:
                ubuntu_checks['is_ubuntu'] = False
        
        # 检查WSL
        try:
            with open('/proc/version', 'r') as f:
                if 'Microsoft' in f.read():
                    ubuntu_checks['is_wsl'] = True
                else:
                    ubuntu_checks['is_wsl'] = False
        except:
            ubuntu_checks['is_wsl'] = False
        
        return ubuntu_checks

def main():
    """主函数"""
    try:
        checker = PythonEnvironmentChecker()
        results = checker.run_all_checks()
        
        # 检查Ubuntu特定项目
        ubuntu_info = checker.check_ubuntu_specific()
        if ubuntu_info.get('is_ubuntu'):
            print("🐧 Ubuntu环境检测:")
            print(f"  Ubuntu版本: {ubuntu_info.get('ubuntu_version', '未知')}")
            print(f"  WSL环境: {'是' if ubuntu_info.get('is_wsl') else '否'}")
            print()
        
        # 返回退出码
        if results['compatibility']['has_issues']:
            print("❌ 环境检查失败，请解决上述问题")
            return 1
        else:
            print("✅ 环境检查完成，可以继续SELF-LNN部署")
            return 0
            
    except Exception as e:
        print(f"❌ 检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())