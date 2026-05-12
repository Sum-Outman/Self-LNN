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

# 创建神经架构搜索库（含帕累托优化）
add_library(selflnn_nas STATIC ${NAS_SOURCES})

# 设置目标属性
set_target_properties(selflnn_nas PROPERTIES
    OUTPUT_NAME "selflnn_nas"
    VERSION 1.0.0
    SOVERSION 1
)

# 链接依赖库
target_link_libraries(selflnn_nas
    utils
    selflnn_core
    selflnn_cognition
)

# 兼容性别名
add_library(selflnn_evolution ALIAS selflnn_nas)

# 安装规则（如果需要）
install(TARGETS selflnn_nas
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/neural_architecture_search.h
    DESTINATION include/selflnn
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/evolution/pareto_optimization.h
    DESTINATION include/selflnn/evolution
)