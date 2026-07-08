# 并发支持模块

# 源文件
set(CONCURRENCY_SOURCES
    thread_pool.c
    lock_free.c
    rcu.c
    rw_lock_map.c
)

# BLD-021修复: 使用target_include_directories替代include_directories
# include_directories为全局影响，target_include_directories按需作用当前目标，更精确

# 创建库
add_library(selflnn_concurrency ${CONCURRENCY_SOURCES})

# BLD-021修复: 目标级别include目录，替代全局include_directories
target_include_directories(selflnn_concurrency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 链接库
# M-004(K-CYCLE-001)修复: 使用selflnn_base（仅错误报告）替代selflnn_core，打破循环依赖
# concurrency只需要errors.h中的错误码枚举+selflnn_set_last_error()，
# 不需要core的LNN/CFC/Laplace等重型功能。
target_link_libraries(selflnn_concurrency PRIVATE selflnn_base)
if(NOT WIN32)
    target_link_libraries(selflnn_concurrency PRIVATE pthread)
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