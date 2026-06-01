# GPU计算抽象层

# 头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

if(ENABLE_GPU)
    # 完整GPU后端源文件 + CPU后端（含完整SSE/AVX SIMD降级路径）
    # FIX-006修复: ENABLE_GPU=ON时仍包含gpu_cpu.c，确保无GPU时CPU降级路径完整
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
        gpu_cpu.c
    )
else
    # GPU禁用时使用CPU计算后端（完整SSE/AVX SIMD实现）
    # 添加gpu.c以提供统一的CPU硬件检测接口gpu_hardware_get_cpu_info
    set(GPU_SOURCES
        gpu.c
        gpu_cpu.c
    )
endif

# 创建库
add_library(selflnn_gpu ${GPU_SOURCES})

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