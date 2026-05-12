# 安全监控系统库
add_library(selflnn_safety
    safety_monitor.c
    emergency_stop.c
    security_monitor_deep.c
    audit_logger.c
    content_filter.c
)

target_include_directories(selflnn_safety PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_safety PUBLIC selflnn_core utils)

install(TARGETS selflnn_safety
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/safety/
    DESTINATION include/selflnn/safety
    FILES_MATCHING PATTERN "*.h"
)
