# 训练框架库
add_library(selflnn_training
    training.c
    mixed_precision.c
    regularization.c
    model_version.c
    distributed_training.c
    training_pipeline.c
    model_registry.c
    model_parallel.c
    training_monitor.c
    training_dataset.c
    training_enhanced.c
    data_loaders.c
    training_data_pipeline.c
)

target_include_directories(selflnn_training PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(selflnn_training PRIVATE selflnn_core selflnn_learning selflnn_utils selflnn_concurrency)

# MSVC下使用/Z7避免PDB锁定问题
if(MSVC)
    target_compile_options(selflnn_training PRIVATE /Z7)
endif()

# 安装规则
install(TARGETS selflnn_training
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/training/
    DESTINATION include/selflnn/training
    FILES_MATCHING PATTERN "*.h"
)