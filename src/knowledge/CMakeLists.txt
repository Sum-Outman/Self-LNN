# 知识库系统库
set(KNOWLEDGE_SOURCES
    knowledge.c
    api_training.c
    knowledge_graph.c
    semantic_network.c
    logic_reasoning.c
    knowledge_self_check.c
    skill_library.c
    auto_learning.c
    knowledge_version.c
    knowledge_inference.c
    graph_storage.c
    graph_query.c
    graph_reasoning.c
    semantic_parsing.c
    ontology_engineering.c
    cfc_knowledge_embedding.c
    knowledge_integration.c
    uncertainty_reasoning.c
)

add_library(selflnn_knowledge ${KNOWLEDGE_SOURCES})

target_include_directories(selflnn_knowledge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# M-005(K-CYCLE-002)修复: 移除selflnn_cognition链接依赖，
# 知识库通过knowledge_bridge.h运行时注入抽象能力函数指针，
# 打破cognition↔knowledge双向循环依赖。
target_link_libraries(selflnn_knowledge PRIVATE selflnn_core selflnn_memory selflnn_learning)

if(MSVC)
    set_source_files_properties(semantic_network.c PROPERTIES COMPILE_FLAGS "/FImath.h")
endif()

# 安装规则
install(TARGETS selflnn_knowledge
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/knowledge/
    DESTINATION include/selflnn/knowledge
    FILES_MATCHING PATTERN "*.h"
)