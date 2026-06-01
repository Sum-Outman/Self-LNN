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
    # dataset_api.c已删除(20个重复定义+0有效代码,垃圾文件)
)

target_include_directories(selflnn_training PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 添加selflnn_memory链接
# training.c直接引用memory_manager.h、调用memory_manager_retrieve等函数
# 原配置缺少此依赖，作为独立静态库时符号不完整
target_link_libraries(selflnn_training PRIVATE selflnn_core selflnn_learning selflnn_utils selflnn_concurrency selflnn_memory)

# MSVC下使用/Z7避免PDB锁定问题
if(MSVC)
    target_compile_options(selflnn_training PRIVATE /Z7 /wd4098 /wd4267)
    # /wd4098(void函数返回-1预存bug) /wd4267(size_t→int转换)
endif

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