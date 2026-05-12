# 推理与计划引擎库
message(STATUS "构建推理库: 包含源文件 reasoning.c 和 planning.c")

set(REASONING_SOURCES
    reasoning.c
    planning.c
    long_term_planning.c
    causal_reasoning.c
    laplace_operator.c
    reasoning_version.c
    planning_enhanced.c
    cfc_uncertainty_reasoning.c
    hierarchical_planning.c
    math_physics_reasoning.c
    quaternion_liquid_gate.c
)

message(STATUS "推理库源文件: ${REASONING_SOURCES}")

add_library(selflnn_reasoning ${REASONING_SOURCES})

target_include_directories(selflnn_reasoning PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_reasoning PRIVATE selflnn_core selflnn_memory)

# 安装规则
install(TARGETS selflnn_reasoning
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/reasoning/
    DESTINATION include/selflnn/reasoning
    FILES_MATCHING PATTERN "*.h"
)