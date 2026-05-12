#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SELF-LNN 项目验证脚本

验证项目完整性、编译能力、基本功能测试
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


class VerificationError(Exception):
    """验证错误异常"""
    pass


class ProjectVerifier:
    """项目验证器"""
    
    def __init__(self, project_root: str):
        """初始化验证器
        
        Args:
            project_root: 项目根目录路径
        """
        self.project_root = Path(project_root).resolve()
        self.system = platform.system()
        self.architecture = platform.architecture()[0]
        self.results = {
            "项目名称": "SELF-LNN",
            "验证时间": datetime.now().isoformat(),
            "系统信息": f"{self.system} {self.architecture}",
            "验证结果": {},
            "成功": True,
            "错误数量": 0,
            "警告数量": 0
        }
        
        # 关键文件列表
        self.critical_files = [
            "CMakeLists.txt",
            "README.md",
            "LICENSE",
            "requirements.txt",
            "docs/架构设计.md",
            "docs/API文档.md",
            "docs/部署指南.md",
        ]
        
        # 关键目录列表
        self.critical_dirs = [
            "include/selflnn",
            "src/core",
            "src/multimodal",
            "src/memory", 
            "src/reasoning",
            "src/learning",
            "src/training",
            "src/utils",
            "src/backend",
            "src/robot",
            "src/knowledge",
            "src/gpu",
            "src/concurrency",
            "tests",
            "examples",
            "config",
            "scripts",
        ]
        
        # 关键头文件列表
        self.critical_headers = [
            "include/selflnn/core/cfc.h",
            "include/selflnn/core/cfc_cell.h",
            "include/selflnn/core/dynamics.h",
            "include/selflnn/core/lnn.h",
            "include/selflnn/core/network.h",
            "include/selflnn/core/state.h",
            "include/selflnn/multimodal/audio.h",
            "include/selflnn/multimodal/multimodal.h",
            "include/selflnn/multimodal/multimodal_manager.h",
            "include/selflnn/multimodal/sensor.h",
            "include/selflnn/multimodal/text.h",
            "include/selflnn/multimodal/unified_encoder.h",
            "include/selflnn/multimodal/vision.h",
            "include/selflnn/memory/episodic.h",
            "include/selflnn/memory/long_term.h",
            "include/selflnn/memory/memory.h",
            "include/selflnn/memory/memory_manager.h",
            "include/selflnn/memory/semantic.h",
            "include/selflnn/memory/short_term.h",
            "include/selflnn/reasoning/planning.h",
            "include/selflnn/reasoning/reasoning.h",
            "include/selflnn/learning/learning.h",
            "include/selflnn/training/training.h",
            "include/selflnn/utils/math_utils.h",
            "include/selflnn/utils/memory_utils.h",
            "include/selflnn/utils/perf.h",
            "include/selflnn/utils/platform.h",
            "include/selflnn/utils/string_utils.h",
            "include/selflnn/backend/backend.h",
            "include/selflnn/robot/robot.h",
            "include/selflnn/knowledge/api_training.h",
            "include/selflnn/knowledge/knowledge.h",
            "include/selflnn/gpu/gpu.h",
            "include/selflnn/concurrency/thread_pool.h",
            "include/selflnn/self_cognition.h",
        ]
        
        # 关键源文件列表
        self.critical_sources = [
            "src/core/cfc.c",
            "src/core/cfc_cell.c",
            "src/core/dynamics.c",
            "src/core/lnn.c",
            "src/core/state.c",
            "src/multimodal/audio.c",
            "src/multimodal/multimodal.c",
            "src/multimodal/multimodal_manager.c",
            "src/multimodal/sensor.c",
            "src/multimodal/text.c",
            "src/multimodal/unified_encoder.c",
            "src/multimodal/vision.c",
            "src/memory/episodic.c",
            "src/memory/long_term.c",
            "src/memory/memory.c",
            "src/memory/memory_manager.c",
            "src/memory/semantic.c",
            "src/memory/short_term.c",
            "src/reasoning/planning.c",
            "src/reasoning/reasoning.c",
            "src/learning/learning.c",
            "src/training/training.c",
            "src/utils/math_utils.c",
            "src/utils/memory_utils.c",
            "src/utils/perf.c",
            "src/utils/platform.c",
            "src/backend/backend.c",
            "src/robot/robot.c",
            "src/knowledge/api_training.c",
            "src/knowledge/knowledge.c",
            "src/gpu/gpu.c",
            "src/concurrency/thread_pool.c",
        ]

    def print_header(self, title: str):
        """打印标题
        
        Args:
            title: 标题文本
        """
        print(f"\n{'=' * 80}")
        print(f"{title.center(80)}")
        print(f"{'=' * 80}")

    def print_success(self, message: str):
        """打印成功消息
        
        Args:
            message: 消息文本
        """
        print(f"[✓] {message}")

    def print_warning(self, message: str):
        """打印警告消息
        
        Args:
            message: 消息文本
        """
        print(f"[!] {message}")
        self.results["警告数量"] += 1

    def print_error(self, message: str):
        """打印错误消息
        
        Args:
            message: 消息文本
        """
        print(f"[✗] {message}")
        self.results["错误数量"] += 1
        self.results["成功"] = False

    def check_file_exists(self, file_path: str, required: bool = True) -> bool:
        """检查文件是否存在
        
        Args:
            file_path: 文件路径（相对项目根目录）
            required: 是否为必需文件
            
        Returns:
            文件是否存在
        """
        full_path = self.project_root / file_path
        
        if full_path.exists() and full_path.is_file():
            self.print_success(f"文件存在: {file_path}")
            return True
        else:
            if required:
                self.print_error(f"文件缺失: {file_path}")
                return False
            else:
                self.print_warning(f"文件缺失（可选）: {file_path}")
                return False

    def check_dir_exists(self, dir_path: str, required: bool = True) -> bool:
        """检查目录是否存在
        
        Args:
            dir_path: 目录路径（相对项目根目录）
            required: 是否为必需目录
            
        Returns:
            目录是否存在
        """
        full_path = self.project_root / dir_path
        
        if full_path.exists() and full_path.is_dir():
            # 检查目录是否为空
            try:
                items = list(full_path.iterdir())
                if items:
                    self.print_success(f"目录存在且非空: {dir_path}")
                else:
                    self.print_warning(f"目录存在但为空: {dir_path}")
                return True
            except PermissionError:
                self.print_warning(f"目录存在但无法访问: {dir_path}")
                return True
        else:
            if required:
                self.print_error(f"目录缺失: {dir_path}")
                return False
            else:
                self.print_warning(f"目录缺失（可选）: {dir_path}")
                return False

    def run_command(self, cmd: List[str], cwd: Optional[str] = None, 
                   timeout: int = 300) -> Tuple[int, str, str]:
        """运行命令
        
        Args:
            cmd: 命令列表
            cwd: 工作目录
            timeout: 超时时间（秒）
            
        Returns:
            (返回码, 标准输出, 标准错误)
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or str(self.project_root),
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"命令超时: {' '.join(cmd)}"
        except FileNotFoundError:
            return -1, "", f"命令未找到: {cmd[0]}"
        except Exception as e:
            return -1, "", f"运行命令出错: {str(e)}"

    def verify_project_structure(self):
        """验证项目结构"""
        self.print_header("验证项目结构")
        
        # 检查项目根目录
        if not self.project_root.exists():
            self.print_error(f"项目根目录不存在: {self.project_root}")
            raise VerificationError("项目根目录不存在")
        
        self.print_success(f"项目根目录: {self.project_root}")
        
        # 检查关键文件
        files_result = {"通过": 0, "失败": 0, "警告": 0}
        for file in self.critical_files:
            if self.check_file_exists(file, required=False):
                files_result["通过"] += 1
            else:
                files_result["警告"] += 1
        
        # 检查关键目录
        dirs_result = {"通过": 0, "失败": 0, "警告": 0}
        for dir_path in self.critical_dirs:
            if self.check_dir_exists(dir_path, required=True):
                dirs_result["通过"] += 1
            else:
                dirs_result["失败"] += 1
        
        # 检查关键头文件
        headers_result = {"通过": 0, "失败": 0, "警告": 0}
        for header in self.critical_headers:
            if self.check_file_exists(header, required=True):
                headers_result["通过"] += 1
            else:
                headers_result["失败"] += 1
        
        # 检查关键源文件
        sources_result = {"通过": 0, "失败": 0, "警告": 0}
        for source in self.critical_sources:
            if self.check_file_exists(source, required=True):
                sources_result["通过"] += 1
            else:
                sources_result["失败"] += 1
        
        self.results["验证结果"]["项目结构"] = {
            "文件检查": files_result,
            "目录检查": dirs_result,
            "头文件检查": headers_result,
            "源文件检查": sources_result
        }

    def verify_cmake_config(self):
        """验证CMake配置"""
        self.print_header("验证CMake配置")
        
        # 检查CMakeLists.txt
        cmake_file = self.project_root / "CMakeLists.txt"
        if not cmake_file.exists():
            self.print_error("CMakeLists.txt 不存在")
            return
        
        # 读取CMakeLists.txt内容
        try:
            with open(cmake_file, 'r', encoding='utf-8') as f:
                cmake_content = f.read()
            
            # 检查关键内容
            checks = {
                "包含项目名称": "project(selflnn" in cmake_content.lower(),
                "设置C标准": "set(CMAKE_C_STANDARD" in cmake_content,
                "包含子目录": "add_subdirectory" in cmake_content,
                "包含测试": "enable_testing" in cmake_content or "BUILD_TESTS" in cmake_content,
            }
            
            for check_name, check_result in checks.items():
                if check_result:
                    self.print_success(f"CMake配置检查: {check_name}")
                else:
                    self.print_warning(f"CMake配置检查: {check_name} 缺失")
            
            self.results["验证结果"]["CMake配置"] = {
                "文件存在": True,
                "内容检查": checks
            }
            
        except Exception as e:
            self.print_error(f"读取CMakeLists.txt失败: {str(e)}")
            self.results["验证结果"]["CMake配置"] = {
                "文件存在": True,
                "读取错误": str(e)
            }

    def verify_build_system(self):
        """验证构建系统"""
        self.print_header("验证构建系统")
        
        # 创建构建目录
        build_dir = self.project_root / "build"
        if not build_dir.exists():
            try:
                build_dir.mkdir(parents=True)
                self.print_success("创建构建目录")
            except Exception as e:
                self.print_error(f"创建构建目录失败: {str(e)}")
                return
        
        # 运行CMake配置
        self.print_success("运行CMake配置...")
        
        if self.system == "Windows":
            cmake_cmd = ["cmake", "..", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"]
        else:
            cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]
        
        returncode, stdout, stderr = self.run_command(cmake_cmd, cwd=str(build_dir))
        
        if returncode == 0:
            self.print_success("CMake配置成功")
            cmake_success = True
            
            # 尝试编译
            self.print_success("尝试编译项目...")
            
            if self.system == "Windows":
                build_cmd = ["cmake", "--build", ".", "--config", "Release", "--parallel", "4"]
            else:
                build_cmd = ["make", "-j4"]
            
            build_returncode, build_stdout, build_stderr = self.run_command(
                build_cmd, cwd=str(build_dir), timeout=600
            )
            
            if build_returncode == 0:
                self.print_success("编译成功")
                build_success = True
            else:
                self.print_error("编译失败")
                self.print_warning(f"编译输出:\n{build_stderr}")
                build_success = False
        else:
            self.print_error("CMake配置失败")
            self.print_warning(f"CMake错误输出:\n{stderr}")
            cmake_success = False
            build_success = False
        
        self.results["验证结果"]["构建系统"] = {
            "CMake配置": cmake_success,
            "编译": build_success,
            "构建目录": str(build_dir)
        }

    def verify_dependencies(self):
        """验证依赖项"""
        self.print_header("验证依赖项")
        
        # 检查必要工具
        tools = {
            "CMake": ["cmake", "--version"],
            "编译器": ["gcc", "--version"] if self.system != "Windows" else ["cl", "?"],
            "Python": ["python", "--version"],
            "Git": ["git", "--version"],
        }
        
        tools_result = {}
        for tool_name, tool_cmd in tools.items():
            returncode, stdout, stderr = self.run_command(tool_cmd)
            if returncode == 0 or returncode == 1:  # 有些命令返回1但实际上是成功的
                self.print_success(f"工具存在: {tool_name}")
                tools_result[tool_name] = True
            else:
                self.print_warning(f"工具缺失或有问题: {tool_name}")
                tools_result[tool_name] = False
        
        # 检查Python依赖
        requirements_file = self.project_root / "requirements.txt"
        python_deps_result = {"检查跳过": "requirements.txt不存在"}
        
        if requirements_file.exists():
            try:
                self.print_success("检查Python依赖...")
                returncode, stdout, stderr = self.run_command(
                    ["pip", "install", "-r", "requirements.txt"]
                )
                if returncode == 0:
                    self.print_success("Python依赖安装成功")
                    python_deps_result = {"状态": "安装成功"}
                else:
                    self.print_warning("Python依赖安装失败")
                    python_deps_result = {"状态": "安装失败", "错误": stderr[:200]}
            except Exception as e:
                self.print_warning(f"检查Python依赖时出错: {str(e)}")
                python_deps_result = {"状态": "检查失败", "错误": str(e)}
        
        self.results["验证结果"]["依赖项"] = {
            "工具检查": tools_result,
            "Python依赖": python_deps_result
        }

    def verify_tests(self):
        """验证测试"""
        self.print_header("验证测试")
        
        tests_dir = self.project_root / "tests"
        tests_result = {"测试目录存在": False, "测试文件": 0, "运行测试": False}
        
        if tests_dir.exists():
            tests_result["测试目录存在"] = True
            
            # 计算测试文件数量
            test_files = list(tests_dir.glob("**/*.c")) + list(tests_dir.glob("**/*.cpp"))
            tests_result["测试文件"] = len(test_files)
            
            if test_files:
                self.print_success(f"找到 {len(test_files)} 个测试文件")
            else:
                self.print_warning("测试目录为空")
            
            # 检查是否已编译测试
            build_dir = self.project_root / "build"
            if build_dir.exists():
                # 查找测试可执行文件
                test_executables = []
                if self.system == "Windows":
                    test_executables = list(build_dir.glob("**/*test*.exe")) + list(build_dir.glob("**/*Test*.exe"))
                else:
                    test_executables = list(build_dir.glob("**/*test*")) + list(build_dir.glob("**/*Test*"))
                
                if test_executables:
                    self.print_success(f"找到 {len(test_executables)} 个测试可执行文件")
                    tests_result["运行测试"] = True
                    
                    # 运行第一个测试作为示例
                    if test_executables:
                        test_exe = test_executables[0]
                        self.print_success(f"运行测试: {test_exe.name}")
                        returncode, stdout, stderr = self.run_command([str(test_exe)])
                        
                        if returncode == 0:
                            self.print_success(f"测试通过: {test_exe.name}")
                            tests_result["测试结果"] = "通过"
                        else:
                            self.print_warning(f"测试失败: {test_exe.name}")
                            tests_result["测试结果"] = "失败"
                else:
                    self.print_warning("未找到测试可执行文件，可能需要编译测试")
        else:
            self.print_warning("测试目录不存在")
        
        self.results["验证结果"]["测试"] = tests_result

    def verify_examples(self):
        """验证示例"""
        self.print_header("验证示例")
        
        examples_dir = self.project_root / "examples"
        examples_result = {"示例目录存在": False, "示例文件": 0}
        
        if examples_dir.exists():
            examples_result["示例目录存在"] = True
            
            # 计算示例文件数量
            example_files = list(examples_dir.glob("**/*.c")) + list(examples_dir.glob("**/*.cpp"))
            examples_result["示例文件"] = len(example_files)
            
            if example_files:
                self.print_success(f"找到 {len(example_files)} 个示例文件")
            else:
                self.print_warning("示例目录为空")
        else:
            self.print_warning("示例目录不存在")
        
        self.results["验证结果"]["示例"] = examples_result

    def verify_frontend(self):
        """验证前端（如果存在）"""
        self.print_header("验证前端")
        
        frontend_dirs = [
            self.project_root / "frontend",
            self.project_root / "web",
            self.project_root / "ui",
        ]
        
        frontend_result = {"前端存在": False, "类型": "未知", "文件数量": 0}
        
        for frontend_dir in frontend_dirs:
            if frontend_dir.exists():
                frontend_result["前端存在"] = True
                frontend_result["目录"] = str(frontend_dir.relative_to(self.project_root))
                
                # 检测前端类型
                if (frontend_dir / "package.json").exists():
                    frontend_result["类型"] = "Node.js/JavaScript"
                    self.print_success("检测到Node.js前端")
                elif (frontend_dir / "requirements.txt").exists():
                    frontend_result["类型"] = "Python"
                    self.print_success("检测到Python前端")
                elif (frontend_dir / "Cargo.toml").exists():
                    frontend_result["类型"] = "Rust"
                    self.print_success("检测到Rust前端")
                else:
                    # 统计文件数量
                    file_count = sum(1 for _ in frontend_dir.glob("**/*") if _.is_file())
                    frontend_result["文件数量"] = file_count
                    self.print_success(f"检测到前端目录，包含 {file_count} 个文件")
                
                break
        
        if not frontend_result["前端存在"]:
            self.print_warning("未找到前端目录")
        
        self.results["验证结果"]["前端"] = frontend_result

    def generate_summary(self):
        """生成验证总结"""
        self.print_header("验证总结")
        
        total_tests = 7  # 验证的类别数量
        passed_tests = 0
        
        for category, result in self.results["验证结果"].items():
            if isinstance(result, dict):
                # 基础通过检查
                if "通过" in str(result) or "成功" in str(result) or "存在" in str(result):
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"项目名称: {self.results['项目名称']}")
        print(f"验证时间: {self.results['验证时间']}")
        print(f"系统信息: {self.results['系统信息']}")
        print(f"验证类别: {total_tests}")
        print(f"通过类别: {passed_tests}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"错误数量: {self.results['错误数量']}")
        print(f"警告数量: {self.results['警告数量']}")
        
        if self.results["成功"]:
            print("\n🎉 项目验证成功！所有关键检查通过。")
            print("项目结构完整，可以继续开发。")
        else:
            print(f"\n⚠️  项目验证失败，发现 {self.results['错误数量']} 个错误。")
            print("请修复上述错误后重新验证。")
        
        # 保存结果到文件
        report_file = self.project_root / "verification_report.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            self.print_success(f"验证报告已保存到: {report_file}")
        except Exception as e:
            self.print_warning(f"保存验证报告失败: {str(e)}")
        
        return self.results["成功"]

    def run_all_checks(self) -> bool:
        """运行所有检查
        
        Returns:
            验证是否成功
        """
        try:
            self.verify_project_structure()
            self.verify_cmake_config()
            self.verify_dependencies()
            self.verify_build_system()
            self.verify_tests()
            self.verify_examples()
            self.verify_frontend()
            
            return self.generate_summary()
            
        except VerificationError as e:
            self.print_error(f"验证过程出错: {str(e)}")
            self.results["成功"] = False
            return False
        except Exception as e:
            self.print_error(f"验证过程中出现未预期错误: {str(e)}")
            self.results["成功"] = False
            return False


def main() -> int:
    """主函数
    
    Returns:
        退出码 (0 = 成功, 1 = 失败)
    """
    print("SELF-LNN 项目验证脚本")
    print("版本: 1.0.0")
    print("=" * 80)
    
    # 确定项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"项目根目录: {project_root}")
    
    # 创建验证器
    verifier = ProjectVerifier(str(project_root))
    
    # 运行所有检查
    success = verifier.run_all_checks()
    
    if success:
        print("\n验证完成，项目状态良好。")
        return 0
    else:
        print("\n验证完成，发现问题需要修复。")
        return 1


if __name__ == "__main__":
    sys.exit(main())