# 后端服务库
add_library(selflnn_backend
    backend.c
    auth.c
    websocket_push.c
)

target_include_directories(selflnn_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_backend PRIVATE selflnn_core selflnn_multimodal selflnn_memory selflnn_reasoning selflnn_learning selflnn_robot selflnn_programming selflnn_safety selflnn_knowledge selflnn_training selflnn_gpu selflnn_concurrency selflnn_cognition selflnn_nas selflnn_product_design selflnn_multisystem selflnn_distributed utils)

# 安装规则
install(TARGETS selflnn_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/backend/
    DESTINATION include/selflnn/backend
    FILES_MATCHING PATTERN "*.h"
)