# 并发支持模块

# 源文件
set(CONCURRENCY_SOURCES
    thread_pool.c
    lock_free.c
    rcu.c
    rw_lock_map.c
)

# 头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# 创建库
add_library(selflnn_concurrency ${CONCURRENCY_SOURCES})

# 链接库
target_link_libraries(selflnn_concurrency selflnn_core)
if(WIN32)
    target_link_libraries(selflnn_concurrency)
else()
    target_link_libraries(selflnn_concurrency pthread)
endif()

# 安装规则
install(TARGETS selflnn_concurrency
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/concurrency/
    DESTINATION include/selflnn/concurrency
    FILES_MATCHING PATTERN "*.h"
)