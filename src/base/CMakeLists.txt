# ============================================================
# 自感知液态神经网络 - 基础类型与错误处理库 (selflnn_base)
# ============================================================
# M-004(K-CYCLE-001)修复: 从core分离errors.c为独立基础库，
# 打破 core -> gpu -> concurrency -> core 循环依赖。
# selflnn_base是第1层基础设施，仅依赖utils，供所有模块安全引用。
# ============================================================

set(BASE_SOURCES
    errors.c
)

add_library(selflnn_base STATIC ${BASE_SOURCES})

target_include_directories(selflnn_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# errors.c内部使用安全内存分配和日志，需utils库
target_link_libraries(selflnn_base PUBLIC utils)

# 安装规则
install(TARGETS selflnn_base
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
