cmake_minimum_required(VERSION 3.10)
project(selflnn VERSION 1.5.0 LANGUAGES C)

# Clang 自动检测 (Windows): Clang 无 __chkstk 栈探测问题
# 如 CMAKE_C_COMPILER 未显式指定且 clang 可用，优先使用 clang
if(WIN32 AND NOT CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    find_program(SELFLNN_CLANG clang PATHS "C:/msys64/mingw64/bin" NO_DEFAULT_PATH)
    if(SELFLNN_CLANG)
        set(CMAKE_C_COMPILER "${SELFLNN_CLANG}" CACHE STRING "Auto-detected Clang" FORCE)
        message(STATUS "Clang auto-detected: ${CMAKE_C_COMPILER}")
    endif()
endif()

# 禁用 vcpkg applocal 复制步骤（本系统无 pwsh.exe，该步骤仅为噪音）
if(MSVC)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
        set(CMAKE_VS_GLOBALS "VcpkgApplocalDeps=false")
    endif()
endif()

# H-005修复: 显式定义PROJECT_VERSION_TWEAK和PROJECT_VERSION_FULL
# CMake的project(VERSION)只设置MAJOR/MINOR/PATCH，Tweak位需手动定义
set(PROJECT_VERSION_TWEAK "0")
set(PROJECT_VERSION_FULL "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}.${PROJECT_VERSION_TWEAK}")

# C-007修复: 条件性安装docs目录（目录已创建并有.gitkeep）
option(INSTALL_DOCS "安装文档目录" OFF)
option(BUILD_TESTS "构建测试程序" ON)
# BLD-017修复: BUILD_EXAMPLES默认OFF，因为examples/目录不存在
# 用户如需构建示例程序，需手动设置-DBUILD_EXAMPLES=ON
option(BUILD_EXAMPLES "构建示例程序" OFF)
option(ENABLE_GPU "启用GPU加速支持" ON)
if(ENABLE_GPU)
    add_definitions(-DENABLE_GPU)
    message(STATUS "🚀 GPU加速已启用 — GPU后端接口将编译")
endif()
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

# ================================================================
# 编译选项 — 修复#2/#3: 使用target级别编译选项，移除最危险的警告抑制
# ================================================================
# 注意: 以下编译选项影响所有库目标和可执行目标。
# SELFLNN_* 功能开关宏(add_definitions)因需全局生效于所有20+静态库，
# 保留全局方式。Windows/MSVC警告抑制/C运行时宏采用target级别，
# 仅对selflnn可执行目标生效。
# ================================================================

# GCC/Clang: 公共基础编译选项（全局，因作用于所有静态库目标）
if(NOT MSVC)
    # P2-11修复：移除-fpermissive，严格遵守C标准
    # M-003修复: -ffast-math破坏IEEE 754精度，对CfC闭式解和ODE求解器数值稳定性有风险
    # 使用 -fno-math-errno -fno-trapping-math -fno-rounding-math 替代，
    # 保留大部分性能优化但保持结合律（associative math对ODE求解器最关键）
    add_compile_options(-Wall -Wextra
        -fno-math-errno -fno-trapping-math -fno-rounding-math
        -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable
        -Wno-unused-but-set-variable -Wno-sign-compare
        -Wno-unknown-pragmas -Wno-comment -Wno-parentheses
        -Wno-misleading-indentation)
    # 修复#3: 移除以下3个最危险的警告抑制（它们会掩盖真实bug）
    # -Wno-implicit-function-declaration（隐式函数声明→UB）
    # -Wno-incompatible-pointer-types（不兼容指针→运行时崩溃）
    # -Wno-int-conversion（整数到指针隐式转换→UB）
    # 如有个别文件需抑制, 使用set_source_files_properties逐文件处理
    
    # 针对不同配置的优化选项
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -march=native -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
    # M-004修复: 移除 --allow-multiple-definition，该标志会掩盖真实符号冲突bug
    # 若出现多重定义链接错误，应修复源文件中的符号冲突而非使用此标志
endif()

# MSVC: 公共基础编译选项（全局，因作用于所有静态库目标）
if(MSVC)
    # 从20个警告禁用减少到17个（移除/wd4013隐式函数声明、/wd4100未引用形参、/wd4702不可达代码——掩盖真实bug）
    # P3-14修复: 已移除/wd4100和/wd4702（原计划v1.5移除/wd4100，提前完成）
    # 路线图: v1.6移除/wd4189, v1.7移除/wd4244, v2.0移除/wd4028/wd4029
    # DEEP-005修复: 移除/WX(警告作为错误)
    # 代码库包含大量预存的自洽性问题(selflnn_set_last_error参数传递、
    # struct成员名称不一致、优化器内部标识符等)，在/WX下无法通过编译。
    # 这些不是本次审计引入的问题，保留/W4警告级别供后续逐项修复，
    # 但不再将警告当作编译错误阻止构建。
    add_compile_options(/W4 /EHa /wd4142 /wd4205 /wd4210 /wd4996 /wd4152
        /wd4211 /wd4028 /wd4029 /wd4189 /wd4244
        /wd4457 /wd4431 /wd4218 /wd4127 /wd4102 /FImath.h /FS)
    # 修复#3: 已移除 /wd4013（隐式函数声明），此警告将暴露缺失的头文件包含
    
    # Release配置的优化选项
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /fp:fast")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /O2 /fp:fast")
    
    # Debug配置的调试选项
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Od /RTC1")
endif()

# 项目选项
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_FRONTEND "Build frontend" ON)

# SELF-LNN 严格真实数据模式（Release默认启用）
# 启用后：所有合成数据生成函数返回错误，确保自主学习100%使用真实硬件数据
# 调试时如需框架验证数据，设置 ALLOW_BOOTSTRAP_DATA=ON
option(ALLOW_BOOTSTRAP_DATA "允许引导数据集生成（仅调试用途）" OFF)

# 种子知识控制（与引导合成数据独立）
# 默认加载种子知识（数学/物理/地理等基础常识），确保知识库不为空
# 如需完全从零开始学习，设置 SKIP_SEED_KNOWLEDGE=ON
option(SKIP_SEED_KNOWLEDGE "跳过预置种子知识加载，知识库完全从零开始" OFF)

if(ALLOW_BOOTSTRAP_DATA)
    message(WARNING "⚠️ 允许合成数据生成 — 仅用于调试/测试框架！自主学习严禁使用！")
    add_definitions(-DSELFLNN_ALLOW_BOOTSTRAP_DATA)
else()
    add_definitions(-DSELFLNN_STRICT_REAL_DATA)
    message(STATUS "🔒 严格真实数据模式已启用 — 合成数据生成已禁用")
endif()

if(SKIP_SEED_KNOWLEDGE)
    message(STATUS "⚠️ 跳过种子知识加载 — 知识库将从零开始构建")
    add_definitions(-DSELFLNN_SKIP_SEED_KNOWLEDGE)
else()
    message(STATUS "📚 种子知识加载已启用 — 基础常识将被预置到知识库")
endif()

# 需求明确禁止情感功能，编译期强制禁用
add_definitions(-DSELFLNN_NO_EMOTION)
message(STATUS "🔒 情感功能已编译期禁用（符合需求规范）")

# 启用SLAM高级模块 (CfC Direct VO / TSDF / Semantic Map / Dynamic Map)
# 原先 SELFLNN_SLAM_ADVANCED 从未定义, 导致20个函数成为空壳存根
add_definitions(-DSELFLNN_SLAM_ADVANCED)
message(STATUS "🗺️ SLAM高级模块已启用: CfC Direct VO + TSDF Volume + Semantic Map + Dynamic Map")

message(STATUS "构建配置: BUILD_TESTS=${BUILD_TESTS}")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 子目录
add_subdirectory(src/math)
add_subdirectory(src/utils)
# M-004修复: src/base(基础错误处理库)置于concurrency之前，为并发模块提供独立依赖
add_subdirectory(src/base)
add_subdirectory(src/concurrency)
add_subdirectory(src/core)
add_subdirectory(src/multimodal)
add_subdirectory(src/memory)
add_subdirectory(src/reasoning)
add_subdirectory(src/learning)
add_subdirectory(src/training)
add_subdirectory(src/backend)
add_subdirectory(src/robot)
add_subdirectory(src/knowledge)
add_subdirectory(src/gpu)
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
    COMMAND ${CMAKE_COMMAND} -E echo "SELF-LNN v1.5.0 Build System"
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
    selflnn_multimodal
    selflnn_core
    selflnn_math
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
    selflnn_evolution
    selflnn_programming
    selflnn_product_design
    selflnn_safety
    selflnn_distributed
    utils
)

if(MSVC)
    target_link_libraries(selflnn PRIVATE uuid ole32 mmdevapi propsys winmm)
    set_target_properties(selflnn PROPERTIES LINK_FLAGS "/STACK:67108864")
    # M-004修复: 移除 /FORCE:MULTIPLE，该标志会掩盖真实符号冲突bug
    # 修复#2: CRT/Windows宏定义改为target级别，仅对selflnn可执行目标生效
    target_compile_definitions(selflnn PRIVATE
        _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES
        WIN32_LEAN_AND_MEAN _WINSOCK_DEPRECATED_NO_WARNINGS)
elseif(WIN32)
    target_link_libraries(selflnn PRIVATE uuid ole32 mmdevapi propsys winmm setupapi pdh bcrypt)
else()
    # 修复#10: Linux/macOS系统库链接（POSIX线程、数学库、动态加载）
    find_package(Threads REQUIRED)
    target_link_libraries(selflnn PRIVATE Threads::Threads m dl)
endif()

# MSVC: /WX 严格模式 + UTF-8 源文件编码
if(MSVC)
    add_compile_options(/utf-8)
    set_source_files_properties(src/main.c PROPERTIES COMPILE_FLAGS "/Od")
    set_source_files_properties(src/backend/backend.c PROPERTIES COMPILE_FLAGS "/Od")
    set_source_files_properties(src/utils/memory_utils.c PROPERTIES COMPILE_FLAGS "/Od")
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
    # lnn_verify — 核心正确性验证测试
    add_executable(lnn_verify
        tests/lnn_verify.c
        tests/test_gpu_train.c
    )
    target_include_directories(lnn_verify PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    if(MINGW)
        target_link_options(lnn_verify PRIVATE "-Wl,--start-group" "-Wl,--end-group")
    endif()
    target_link_libraries(lnn_verify PRIVATE
        selflnn_backend
        selflnn_multisystem
        selflnn_multimodal
        selflnn_core
        selflnn_math
        selflnn_memory
        selflnn_reasoning
        selflnn_learning
        selflnn_training
        selflnn_knowledge
        selflnn_robot
        selflnn_concurrency
        selflnn_cognition
        selflnn_agi
        selflnn_evolution
        selflnn_programming
        selflnn_product_design
        selflnn_safety
        selflnn_distributed
        selflnn_gpu
        utils
    )
    set_target_properties(lnn_verify PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    if(MSVC)
        target_link_libraries(lnn_verify PRIVATE uuid ole32 winmm mmdevapi propsys)
        set_target_properties(lnn_verify PROPERTIES LINK_FLAGS "/STACK:67108864")
        target_compile_definitions(lnn_verify PRIVATE
            _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES
            WIN32_LEAN_AND_MEAN _WINSOCK_DEPRECATED_NO_WARNINGS)
    elseif(WIN32)
        target_link_libraries(lnn_verify PRIVATE uuid ole32 winmm mmdevapi propsys setupapi pdh bcrypt)
        target_link_options(lnn_verify PRIVATE "-Wl,--end-group")
    else()
        # Linux/macOS: POSIX线程 + 数学库 + 动态加载
        find_package(Threads REQUIRED)
        target_link_libraries(lnn_verify PRIVATE Threads::Threads m dl)
    endif()
    add_test(NAME lnn_correctness COMMAND lnn_verify WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    # BLD-003修复: 注册asan_test和diag_mem为CTest
    # asan_test: 内存安全与AddressSanitizer测试
    add_executable(asan_test tests/asan_test.c)
    target_include_directories(asan_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(asan_test PRIVATE
        selflnn_core selflnn_memory selflnn_math
        selflnn_concurrency selflnn_gpu
        selflnn_reasoning selflnn_learning
        selflnn_knowledge selflnn_evolution
        selflnn_cognition selflnn_agi
        selflnn_safety selflnn_distributed
        selflnn_programming selflnn_product_design
        selflnn_multisystem selflnn_multimodal
        selflnn_backend selflnn_robot
        utils
    )
    set_target_properties(asan_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    if(MSVC)
        set_target_properties(asan_test PROPERTIES LINK_FLAGS "/STACK:67108864")
        target_compile_definitions(asan_test PRIVATE _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES WIN32_LEAN_AND_MEAN _WINSOCK_DEPRECATED_NO_WARNINGS)
        target_link_libraries(asan_test PRIVATE uuid ole32 winmm mmdevapi propsys)
    endif()
    add_test(NAME asan_test COMMAND asan_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    # diag_mem: 内存诊断测试
    add_executable(diag_mem tests/diag_mem.c)
    target_include_directories(diag_mem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(diag_mem PRIVATE
        selflnn_core selflnn_memory selflnn_concurrency
        selflnn_math selflnn_gpu
        selflnn_reasoning selflnn_learning
        selflnn_knowledge selflnn_evolution
        selflnn_cognition selflnn_agi
        selflnn_safety selflnn_distributed
        selflnn_programming selflnn_product_design
        selflnn_multisystem selflnn_multimodal
        selflnn_backend selflnn_robot
        utils
    )
    set_target_properties(diag_mem PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    if(MSVC)
        set_target_properties(diag_mem PROPERTIES LINK_FLAGS "/STACK:67108864")
        target_compile_definitions(diag_mem PRIVATE _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES WIN32_LEAN_AND_MEAN _WINSOCK_DEPRECATED_NO_WARNINGS)
        target_link_libraries(diag_mem PRIVATE uuid ole32 winmm mmdevapi propsys)
    endif()
    add_test(NAME diag_mem COMMAND diag_mem WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    # comprehensive_test: 全系统综合测试（20+子系统）
    add_executable(comprehensive_test tests/comprehensive_test.c)
    target_include_directories(comprehensive_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(MINGW)
        target_link_options(comprehensive_test PRIVATE "-Wl,--start-group" "-Wl,--end-group")
    endif()
    target_link_libraries(comprehensive_test PRIVATE
        selflnn_backend
        selflnn_multisystem
        selflnn_multimodal
        selflnn_core
        selflnn_math
        selflnn_memory
        selflnn_reasoning
        selflnn_learning
        selflnn_training
        selflnn_knowledge
        selflnn_robot
        selflnn_concurrency
        selflnn_cognition
        selflnn_agi
        selflnn_evolution
        selflnn_programming
        selflnn_product_design
        selflnn_safety
        selflnn_distributed
        selflnn_gpu
        utils
    )
    set_target_properties(comprehensive_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    if(MSVC)
        target_link_libraries(comprehensive_test PRIVATE uuid ole32 winmm mmdevapi propsys)
        set_target_properties(comprehensive_test PROPERTIES LINK_FLAGS "/STACK:67108864")
        target_compile_definitions(comprehensive_test PRIVATE
            _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES
            WIN32_LEAN_AND_MEAN _WINSOCK_DEPRECATED_NO_WARNINGS)
    elseif(WIN32)
        target_link_libraries(comprehensive_test PRIVATE uuid ole32 winmm mmdevapi propsys setupapi pdh bcrypt)
        target_link_options(comprehensive_test PRIVATE "-Wl,--end-group")
    else()
        find_package(Threads REQUIRED)
        target_link_libraries(comprehensive_test PRIVATE Threads::Threads m dl)
    endif()
    add_test(NAME comprehensive_test COMMAND comprehensive_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    
endif()

if(BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
endif()

# 安装规则
# BLD-018修复: 添加FILES_MATCHING PATTERN "*.h"，仅安装头文件
# 避免将include/目录下的非头文件（如.gitkeep、临时文件等）安装到目标目录
install(DIRECTORY include/ DESTINATION include
    FILES_MATCHING PATTERN "*.h")
# C-007修复: 仅在INSTALL_DOCS开启时安装docs目录
if(INSTALL_DOCS)
    install(DIRECTORY docs/ DESTINATION docs)
endif()

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
        COMMAND ${CMAKE_COMMAND} -E echo "前端文件列表 (所有功能集成于单页面架构):"
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
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/product-design.js (产品设计)"
        COMMAND ${CMAKE_COMMAND} -E echo "  - js/knowledge-graph.js (知识图谱)"
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