# 多系统控制能力库
message(STATUS "构建多系统控制库: 包含源文件 multisystem_control.c, swarm_intelligence.c")

set(MULTISYSTEM_CONTROL_SOURCES
    multisystem_control.c
    swarm_intelligence.c
)

message(STATUS "多系统控制库源文件: ${MULTISYSTEM_CONTROL_SOURCES}")

add_library(selflnn_multisystem ${MULTISYSTEM_CONTROL_SOURCES})

target_compile_definitions(selflnn_multisystem PRIVATE WIN32_LEAN_AND_MEAN)

target_include_directories(selflnn_multisystem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_multisystem PRIVATE 
    selflnn_core 
    selflnn_memory
    selflnn_utils
    selflnn_concurrency
)

# Windows平台需要链接ws2_32（Winsock2）
if(WIN32)
    target_link_libraries(selflnn_multisystem PRIVATE ws2_32)
endif()

# 安装规则
install(TARGETS selflnn_multisystem
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/multisystem/
    DESTINATION include/selflnn/multisystem
    FILES_MATCHING PATTERN "*.h"
)