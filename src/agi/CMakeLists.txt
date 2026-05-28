# AGI高级认知系统CMake配置

set(AGI_SOURCES
    agi.c
    capability_switch.c
    task_scheduler.c
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

add_library(selflnn_agi STATIC ${AGI_SOURCES})

target_include_directories(selflnn_agi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(selflnn_agi PROPERTIES
    OUTPUT_NAME "selflnn_agi"
    VERSION 1.0.0
    SOVERSION 1
)

target_link_libraries(selflnn_agi PRIVATE
    selflnn_cognition
    selflnn_core
    selflnn_reasoning
    selflnn_knowledge
    selflnn_learning
    selflnn_memory
    selflnn_multimodal
)

install(TARGETS selflnn_agi
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
