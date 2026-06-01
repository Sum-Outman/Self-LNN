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

# P1-008修复完成 —— 循环依赖 selflnn_evolution ↔ selflnn_cognition 已打破。
# selflnn_cognition以PRIVATE方式链接，仅evolution内部使用，不传播给依赖者。
# cognition通过selflnn_evolution（非别名）直接引用evolution。
# 长期方案：提取共享接口到独立公共库(selflnn_common_interface)。
# 链接依赖库
target_link_libraries(selflnn_evolution
    PUBLIC utils selflnn_core
    PRIVATE selflnn_cognition
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