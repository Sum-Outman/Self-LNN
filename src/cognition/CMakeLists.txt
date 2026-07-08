# 自我认知模块CMake配置

# 添加自我认知源文件
set(SELF_COGNITION_SOURCES
    self_cognition.c
    bdi_model.c
    abstraction.c
    metacognition.c
    deep_reflection.c
    deep_thought_chain.c
    deep_correction.c
)

# 添加自我认知头文件路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 创建自我认知库
add_library(selflnn_cognition STATIC ${SELF_COGNITION_SOURCES})

if(MSVC)
    target_compile_options(selflnn_cognition PRIVATE /FImath.h)
endif()

# 设置目标属性
# BLD-014修复: SOVERSION/VERSION对静态库无意义，静态库在编译时链接，不涉及运行时版本解析
# 保留VERSION和SOVERSION仅为元数据记录，实际链接时CMake会忽略这些属性
# 如需动态库版本，需将add_library的STATIC改为SHARED
set_target_properties(selflnn_cognition PROPERTIES
    OUTPUT_NAME "selflnn_cognition"
    VERSION 1.0.0
    SOVERSION 1
)

# 添加cognition头文件独立install规则
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/cognition/
    DESTINATION include/selflnn/cognition
    FILES_MATCHING PATTERN "*.h")

# 链接依赖库（显式添加所有实际使用的模块依赖）
# C-002修复: 移除 selflnn_evolution 依赖, cognition通过回调模式运行时绑定evolution
# cognition是底层认知模块,不应直接依赖上层evolution模块。
target_link_libraries(selflnn_cognition PRIVATE 
    selflnn_core
    selflnn_knowledge
    selflnn_gpu
    selflnn_memory
    selflnn_reasoning
    selflnn_learning
    selflnn_safety
)
