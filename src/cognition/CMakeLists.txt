# 自我认知模块CMake配置

# 添加自我认知源文件
set(SELF_COGNITION_SOURCES
    self_cognition.c
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
endif

# 设置目标属性
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
# 移除selflnn_agi依赖，打破selflnn_agi ↔ selflnn_cognition循环依赖
# cognition是底层认知模块，不应依赖上层agi模块。如需共享类型，应提取到public头文件。
target_link_libraries(selflnn_cognition PRIVATE 
    selflnn_core
    selflnn_evolution
    selflnn_knowledge
    selflnn_gpu
    selflnn_memory
    selflnn_reasoning
    selflnn_learning
    selflnn_safety
)
