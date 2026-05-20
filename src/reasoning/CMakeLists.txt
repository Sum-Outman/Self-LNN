# 推理与计划引擎库
message(STATUS "构建推理库")

# ZSF-001修复: reasoning.c中reasoning_infer_with_knowledge等函数签名
# 与头文件声明不兼容（char** vs float*参数类型差异），MSVC无法编译。
# reasoning_internal.c 为MSVC平台提供真实推理引擎实现。
if(MSVC)
    set(REASONING_SOURCES
        reasoning_internal.c planning.c long_term_planning.c
        causal_reasoning.c laplace_operator.c
        planning_enhanced.c cfc_uncertainty_reasoning.c
        hierarchical_planning.c math_physics_reasoning.c
        quaternion_liquid_gate.c
    )
else()
    set(REASONING_SOURCES
        reasoning.c planning.c long_term_planning.c causal_reasoning.c
        laplace_operator.c planning_enhanced.c
        cfc_uncertainty_reasoning.c hierarchical_planning.c
        math_physics_reasoning.c quaternion_liquid_gate.c
    )
endif()

add_library(selflnn_reasoning ${REASONING_SOURCES})

if(MSVC)
    target_compile_options(selflnn_reasoning PRIVATE /bigobj /wd4013 /wd4146 /wd4028 /wd4029 /wd4267 /wd4090 /wd4133 /wd4047)
endif()

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