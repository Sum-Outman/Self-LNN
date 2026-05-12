# 自我编程能力库
message(STATUS "构建自我编程库: 包含源文件 self_programming.c")

set(SELF_PROGRAMMING_SOURCES
    self_programming.c
    programming_enhanced.c
    c_interpreter.c
)

message(STATUS "自我编程库源文件: ${SELF_PROGRAMMING_SOURCES}")

add_library(selflnn_programming ${SELF_PROGRAMMING_SOURCES})

target_include_directories(selflnn_programming PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_programming PRIVATE 
    selflnn_core 
    selflnn_memory
    selflnn_utils
)

# 安装规则
install(TARGETS selflnn_programming
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/programming/
    DESTINATION include/selflnn/programming
    FILES_MATCHING PATTERN "*.h"
)