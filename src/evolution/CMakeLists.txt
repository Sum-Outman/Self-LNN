# 神经架构搜索 & 帕累托多目标演化优化模块CMake配置

# 添加神经架构搜索与帕累托优化源文件
set(NAS_SOURCES
    pareto_optimization.c
    evolution_engine.c
    neural_architecture_search.c
)

# 添加头文件路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 创建演化优化库（含帕累托优化与神经架构搜索）
add_library(selflnn_evolution STATIC ${NAS_SOURCES})

# 设置目标属性
set_target_properties(selflnn_evolution PROPERTIES
    OUTPUT_NAME "selflnn_evolution"
    VERSION 1.0.0
    SOVERSION 1
)

# C-002修复完成: 循环依赖 selflnn_evolution ↔ selflnn_cognition 已彻底打破。
# 代码层面确认: evolution的3个.c文件零引用cognition的任何符号。
# cognition通过回调模式调用evolution,不再编译期依赖。
target_link_libraries(selflnn_evolution
    PUBLIC utils selflnn_core
)

# 安装规则（如果需要）
install(TARGETS selflnn_evolution
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/evolution/neural_architecture_search.h
    DESTINATION include/selflnn/evolution
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/evolution/pareto_optimization.h
    DESTINATION include/selflnn/evolution
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/evolution/evolution_engine.h
    DESTINATION include/selflnn/evolution
)