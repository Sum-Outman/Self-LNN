# 液态神经网络核心库
add_library(selflnn_core
    lnn.c
    cfc_network.c
    loss.c
    optimizer.c
    cfc_cell.c
    dynamics.c
    state.c
    errors.c
    laplace.c
    laplace_fft.c
    quaternion_lnn.c
    quaternion_conv.c
    quaternion_batchnorm.c
    lnn_layer_norm.c

    quaternion_optimizer.c
    graph_optimization.c
    graph_optimization_ext.c
    tensor.c
    selflnn.c
    decision_engine.c
    quaternion_cfc.c
    quaternion_enhanced.c
    evolutionary_algorithms.c
    cma_es.c
    laplace_features.c
    system_scheduler.c
    parameter_shard.c
    extended_training.c
    ode_solvers.c
    cfc_enhanced.c
    unified_lnn_state.c
    laplace_unified.c
)

# ZSFBUILD: MSVC桩文件 - 为非MSVC平台的子系统(Gazebo/PyBullet/HW检测/Laplace/GPU)提供最小实现
if(MSVC)
    target_sources(selflnn_core PRIVATE msvc_stubs.c)
endif()

target_compile_definitions(selflnn_core PRIVATE WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)

target_include_directories(selflnn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# quaternion_lnn.c、selflnn.c需要GPU函数，所以始终链接GPU桩库并传播给所有使用者
# ZSFABC-C001修复: 移除matrix_ops.c双重编译，改为链接selflnn_math库
target_link_libraries(selflnn_core PUBLIC selflnn_gpu selflnn_math utils)
target_link_libraries(selflnn_core PRIVATE selflnn_multimodal)

# 安装规则
install(TARGETS selflnn_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/lnn.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/cfc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/cfc_cell.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/cfc_enhanced.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/cfc_network.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/dynamics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/state.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/errors.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/laplace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/laplace_fft.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/laplace_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/laplace_unified.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/loss.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/memory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/ode_solvers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/optimizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/port_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/tensor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/decision_engine.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/system_scheduler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/parameter_shard.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/extended_training.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/unified_lnn_state.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/evolutionary_algorithms.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/cma_es.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/quaternion_lnn.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/quaternion_conv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/quaternion_batchnorm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/lnn_layer_norm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/quaternion_cfc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/quaternion_enhanced.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/quaternion_optimizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/quaternion_lnn_kernels.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/graph_optimization.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/core/graph_optimization_ext.h
    DESTINATION include/selflnn/core
)
# M-010修复: websocket_push.h 属于 backend 模块，不应由 core 模块安装
# install(FILES
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/backend/websocket_push.h
#     DESTINATION include/selflnn/backend
# )