# 记忆管理系统库
add_library(selflnn_memory
    memory.c
    memory_manager.c
    short_term.c
    long_term.c
    episodic.c
    semantic.c
    working_memory.c
)

target_include_directories(selflnn_memory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_memory PRIVATE selflnn_core)

# 安装规则
install(TARGETS selflnn_memory
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/memory/
    DESTINATION include/selflnn/memory
    FILES_MATCHING PATTERN "*.h"
)