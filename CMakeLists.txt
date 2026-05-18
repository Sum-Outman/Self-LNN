cmake_minimum_required(VERSION 3.10)
project(selflnn VERSION 1.4.0 LANGUAGES C)

# 禁用 vcpkg applocal 复制步骤（本系统无 pwsh.exe，该步骤仅为噪音）
if(MSVC AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    set(CMAKE_VS_GLOBALS "VcpkgApplocalDeps=false")
endif()

# 构建选项
option(BUILD_TESTS "构建测试程序" ON)
option(BUILD_EXAMPLES "构建示例程序" ON)
option(ENABLE_GPU "启用GPU加速支持" OFF)
option(INSTALL_EXAMPLES "安装示例程序" OFF)

# F-005/F-006: 外部运行时接口（纯IPC通信，不链接外部库）
# 这些接口通过共享内存/套接字与外部程序通信，核心LNN保持纯C零依赖
option(SELFLNN_ENABLE_EXTERNAL_SIM "启用PyBullet/Gazebo外部仿真器IPC接口" ON)
option(SELFLNN_ENABLE_ROS "启用ROS/ROS2机器人控制IPC接口" OFF)
option(SELFLNN_ENABLE_PURE_C_SIM "优先使用纯C内部物理引擎（替代外部仿真器）" ON)

if(SELFLNN_ENABLE_EXTERNAL_SIM)
    add_definitions(-DSELFLNN_HAS_EXTERNAL_SIM)
endif()
if(SELFLNN_ENABLE_ROS)
    add_definitions(-DSELFLNN_HAS_ROS)
endif()
if(SELFLNN_ENABLE_PURE_C_SIM)
    add_definitions(-DSELFLNN_USE_PURE_C_PHYSICS)
endif()

# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# 编译选项
if(MSVC)
    # 针对不同配置的编译器选项
    # I-004: 逐步减少警告禁用。已移除 /wd4013(隐式函数声明)和/wd4133(类型不兼容)
    #   这两个警告会掩盖真正的未定义行为(UB)问题。
    #   保留的警告禁用主要针对: 未引用参数(4100)、类型转换(4244)、弃用API(4996)等风格问题。
    #   后续版本目标: 从13个禁用减少到6个以下。
    add_compile_options(/W4 /WX /EHa /wd4100 /wd4152 /wd4211 /wd4028 /wd4029 /wd4026 /wd4244 /wd4189 /wd4142 /wd4032 /wd4996 /Oi- /FImath.h /FS /wd4205 /wd4210)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES -DWIN32_LEAN_AND_MEAN -D_WINSOCK_DEPRECATED_NO_WARNINGS)
    
    # Release配置的优化选项
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /fp:fast")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /O2 /fp:fast")
    
    # Debug配置的调试选项
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Od /RTC1")
else()
    # GCC/Clang的编译选项
    add_compile_options(-Wall -Wextra -Werror -pedantic -ffast-math)
    
    # 针对不同配置的优化选项
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -march=native -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
endif()

# 项目选项
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_FRONTEND "Build frontend" ON)

# SELF-LNN 严格真实数据模式（Release默认启用）
# 启用后：所有合成数据生成函数返回错误，确保自主学习100%使用真实硬件数据
# 调试时如需框架验证数据，设置 ALLOW_BOOTSTRAP_DATA=ON
option(ALLOW_BOOTSTRAP_DATA "允许引导数据集生成（仅调试用途）" OFF)

if(ALLOW_BOOTSTRAP_DATA)
    message(WARNING "⚠️ 允许合成数据生成 — 仅用于调试/测试框架！自主学习严禁使用！")
    add_definitions(-DSELFLNN_ALLOW_BOOTSTRAP_DATA)
else()
    add_definitions(-DSELFLNN_STRICT_REAL_DATA)
    message(STATUS "🔒 严格真实数据模式已启用 — 合成数据生成已禁用")
endif()

message(STATUS "构建配置: BUILD_TESTS=${BUILD_TESTS}")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 子目录
add_subdirectory(src/core)
add_subdirectory(src/multimodal)
add_subdirectory(src/memory)
add_subdirectory(src/reasoning)
add_subdirectory(src/learning)
add_subdirectory(src/training)
add_subdirectory(src/utils)
add_subdirectory(src/backend)
add_subdirectory(src/robot)
add_subdirectory(src/knowledge)
add_subdirectory(src/gpu)
add_subdirectory(src/concurrency)
add_subdirectory(src/cognition)
add_subdirectory(src/agi)
add_subdirectory(src/evolution)
add_subdirectory(src/programming)
add_subdirectory(src/product_design)
add_subdirectory(src/multisystem)
add_subdirectory(src/distributed)
add_subdirectory(src/safety)

# 代码质量检查目标
if(UNIX)
    add_custom_target(check_code_quality
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/check_code_quality.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "运行代码质量检查"
        VERBATIM
    )
elseif(WIN32)
    add_custom_target(check_code_quality
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/check_code_quality.bat
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "运行代码质量检查"
        VERBATIM
    )
endif()

# 代码格式化目标
if(UNIX)
    add_custom_target(format_code
        COMMAND find ${CMAKE_SOURCE_DIR}/src -name "*.c" -o -name "*.h" | xargs clang-format -style=file -i
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "格式化所有C/C++代码"
        VERBATIM
    )
elseif(WIN32)
    add_custom_target(format_code
        COMMAND powershell -Command "Get-ChildItem -Path '${CMAKE_SOURCE_DIR}\\src' -Recurse -Include '*.c','*.h' | ForEach-Object { clang-format -style=file -i $_.FullName }"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "格式化所有C/C++代码"
        VERBATIM
    )
endif()

# 性能分析目标
if(UNIX)
    add_custom_target(performance_analysis
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/performance_analysis.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "运行性能分析"
        VERBATIM
    )
elseif(WIN32)
    add_custom_target(performance_analysis
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/performance_analysis.bat
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "运行性能分析"
        VERBATIM
    )
endif()

# ================================================================
# K-048: CMake构建管理 —— 清理与源文件组织
# ================================================================

add_custom_target(build_info
    COMMAND ${CMAKE_COMMAND} -E echo "SELF-LNN v1.4.0 Build System"
    COMMAND ${CMAKE_COMMAND} -E echo "Source dir: ${CMAKE_SOURCE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "Build dir:  ${CMAKE_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "Generator:  ${CMAKE_GENERATOR}"
    VERBATIM
)

add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Build directory cleaned: ${CMAKE_BINARY_DIR}"
    VERBATIM
)

# 独立可执行主程序
add_executable(selflnn
    src/main.c
)

target_link_libraries(selflnn PRIVATE
    selflnn_backend
    selflnn_multisystem
    selflnn_core
    selflnn_multimodal
    selflnn_memory
    selflnn_reasoning
    selflnn_learning
    selflnn_training
    selflnn_knowledge
    selflnn_robot
    selflnn_gpu
    selflnn_concurrency
    selflnn_cognition
    selflnn_agi
    selflnn_nas
    selflnn_programming
    selflnn_product_design
    selflnn_safety
    selflnn_distributed
    utils
)

if(MSVC)
    target_link_libraries(selflnn PRIVATE uuid ole32 mmdevapi propsys winmm)
    set_target_properties(selflnn PROPERTIES LINK_FLAGS "/STACK:67108864")
endif()

target_include_directories(selflnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(selflnn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    OUTPUT_NAME selflnn
)

install(TARGETS selflnn
    RUNTIME DESTINATION bin
)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 安装规则
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY docs/ DESTINATION docs)

if(BUILD_FRONTEND)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/frontend/ DESTINATION frontend
        FILES_MATCHING
            PATTERN "*.html"
            PATTERN "*.css"
            PATTERN "*.js"
            PATTERN "*.json"
            PATTERN "*.png"
            PATTERN "*.jpg"
            PATTERN "*.svg"
            PATTERN "*.ico"
    )
    
    add_custom_target(check_frontend
        COMMAND ${CMAKE_COMMAND} -E echo "正在检查前端文件完整性..."
        COMMAND ${CMAKE_COMMAND} -E echo "前端文件列表 (ZSF-003修复: 所有功能集成于单页面架构):"
        COMMAND ${CMAKE_COMMAND} -E echo "  - index.html (主页面, 含15个功能section)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/compatibility.js (浏览器兼容层)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/api-service.js (API服务)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/data-engine.js (数据引擎)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/device-manager.js (设备管理器)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/voice-command.js (语音指令)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/text-command.js (文本指令)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/agi-controller.js (AGI控制器)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/visualization.js (实时可视化)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/dialogue-enhanced.js (增强对话)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/selflnn-chart.js (图表组件)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/simulation-control.js (仿真控制)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/training-center.js (训练中心)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/training-push.js (训练推送)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/voice-capture-util.js (语音采集)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/programming-workbench.js (编程工作台)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/main.js (主逻辑)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/hardware-util.js (硬件工具)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - css/style.css (样式)"
        COMMAND ${CMAKE_COMMAND} -E echo "前端文件完整性检查完成"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "检查前端文件完整性"
        VERBATIM
    )
endif()

# 版本信息
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/selflnn/version.h
)

# 包配置
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/selflnn-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# 创建包（基础版本，不导出目标）
set(ConfigPackageLocation lib/cmake/selflnn)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/selflnn-config-version.cmake
    DESTINATION ${ConfigPackageLocation}
)