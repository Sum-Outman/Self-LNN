# GPU计算抽象层

# BLD-021修复: 使用target_include_directories替代include_directories
# include_directories为全局影响，target_include_directories按需作用当前目标，更精确

if(ENABLE_GPU AND NOT MINGW)
    # 完整GPU后端源文件 + CPU后端（含完整SSE/AVX SIMD降级路径）
    # gpu.c已包含gpu_cpu.c的全部函数(调度层), 不重复编译
    set(GPU_SOURCES
        gpu.c
        gpu_hardware_detect.c
        gpu_memory_pool.c
        auto_kernel_optimization.c
        gpu_opencl.c
        gpu_vulkan.c
        gpu_cuda.c
        gpu_metal.c
        npu.c
        npu_common.c
        gpu_npu.c
        gpu_ascend.c
        gpu_cambricon.c
        gpu_tpu.c
        gpu_rocm.c
        gpu_intel.c
        gpu_benchmark.c
    )
else()
    # GPU禁用时仅使用CPU计算后端（完整SSE/AVX SIMD实现）
    # gpu_cpu.c已包含所有CPU降级路径，不编译gpu.c避免重复符号
    set(GPU_SOURCES
        gpu_cpu.c
    )
endif()

# 创建库
add_library(selflnn_gpu ${GPU_SOURCES})

# BLD-021修复: 目标级别include目录，替代全局include_directories
target_include_directories(selflnn_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 链接库
target_link_libraries(selflnn_gpu PRIVATE utils selflnn_concurrency)

# 安装规则
install(TARGETS selflnn_gpu
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/selflnn/gpu/
    DESTINATION include/selflnn/gpu
    FILES_MATCHING PATTERN "*.h"
)