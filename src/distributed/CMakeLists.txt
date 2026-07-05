# 分布式系统能力库
message(STATUS "构建分布式系统库: 包含源文件 pbft.c 和 load_balancer.c")

set(DISTRIBUTED_SOURCES
    pbft.c
    load_balancer.c
)

add_library(selflnn_distributed ${DISTRIBUTED_SOURCES})

target_include_directories(selflnn_distributed PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_distributed PRIVATE
    selflnn_core
    utils
    selflnn_concurrency
)

# Windows平台需要链接ws2_32
if(WIN32)
    target_link_libraries(selflnn_distributed PRIVATE ws2_32)
endif()

# 安装规则
install(TARGETS selflnn_distributed
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/distributed/
    DESTINATION include/selflnn/distributed
    FILES_MATCHING PATTERN "*.h"
)
