/**
 * @file gpu.h
 * @brief GPU加速接口 —— 统一计算后端抽象层
 *
 * 提供多种GPU后端的统一抽象接口，调度到不同后端实现。
 * 支持CPU计算和多种品牌GPU计算，支持CPU训练和多种品牌GPU训练。
 *
 * ================================================================
 * 关键声明：CPU模式 100% 自包含，零外部依赖
 * ================================================================
 * 当系统检测不到任何GPU硬件或显式选择 GPU_BACKEND_CPU 时：
 *   - 使用 CPU SIMD 向量化计算（SSE/SSE2/SSE3/AVX/AVX2/NEON）
 *   - CPU硬件检测通过 CPUID（x86）或 sysfs（Linux/ARM）实现
 *   - 所有运算均为纯C数学实现（expf/sinf/cosf/sqrtf 等 C99 标准库函数）
 *   - 不依赖 NVIDIA CUDA Toolkit / AMD ROCm / Intel oneAPI / Vulkan SDK / OpenCL SDK
 *   - 不依赖任何第三方数学库（BLAS/LAPACK/MKL 等）
 *   - 线程池通过 pthread / Win32 原生线程实现（不依赖 OpenMP/TBB）
 *   - 100% 纯 C 语言自包含，可直接在任何 C99 编译器上编译运行
 *
 * GPU加速模式（可选）：
 *   - CUDA:   需要 NVIDIA CUDA Toolkit + NVIDIA GPU 硬件
 *   - ROCm:   需要 AMD ROCm 软件栈 + AMD GPU 硬件
 *   - Intel:  需要 Intel oneAPI + Intel GPU
 *   - Vulkan:  需要 Vulkan SDK 1.2+ + 兼容 GPU
 *   - OpenCL:  需要 OpenCL SDK 1.2+ + 兼容设备
 *   - Metal:   需要 macOS + Apple GPU
 *   - 昇腾:    需要华为昇腾 NPU 驱动
 *   - 寒武纪:  需要寒武纪 MLU 驱动
 *   - TPU:     需要 Google TPU 运行时
 * 以上GPU后端通过动态加载系统驱动库实现硬件加速，
 * 是可选增强功能，不影响纯CPU模式的自包含性。
 * ================================================================
 */

#ifndef SELFLNN_GPU_H
#define SELFLNN_GPU_H

#include <stddef.h>
#include "selflnn/core/common.h"
#include "selflnn/gpu/auto_kernel_optimization.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GPU后端类型
 * 
 * @note GpuBackend枚举定义位于 core/common.h
 * 
 * 可用后端类型：
 * - GPU_BACKEND_CPU(0): CPU并行计算（生产就绪）
 * - GPU_BACKEND_CUDA(1): NVIDIA CUDA GPU加速（需要CUDA工具包）
 * - GPU_BACKEND_OPENCL(2): OpenCL GPU加速（需要OpenCL 1.2+）
 * - GPU_BACKEND_VULKAN(3): Vulkan GPU加速（需要Vulkan 1.2+）
 * - GPU_BACKEND_METAL(4): Apple Metal GPU加速（需要macOS/iOS）
 * 
 * 所有GPU后端均提供真实的硬件加速实现，不会自动回退到CPU模拟。
 */

/**
 * @brief GPU设备类型
 */
typedef enum {
    GPU_DEVICE_TYPE_INTEGRATED = 0, /**< 集成GPU */
    GPU_DEVICE_TYPE_DISCRETE = 1,   /**< 独立GPU */
    GPU_DEVICE_TYPE_CPU = 2,        /**< CPU设备 */
    GPU_DEVICE_TYPE_UNKNOWN = 3     /**< 未知设备类型 */
} GpuDeviceType;

/**
 * @brief GPU内存类型
 */
typedef enum {
    GPU_MEMORY_HOST = 0,        /**< 主机内存 */
    GPU_MEMORY_DEVICE = 1,      /**< 设备内存 */
    GPU_MEMORY_UNIFIED = 2      /**< 统一内存 */
} GpuMemoryType;

/**
 * @brief CPU SIMD指令集支持标志位
 * @anchor CPU_SIMD_FLAG
 * @{
 */
#define CPU_SIMD_SSE      (1 << 0)   /**< SSE指令集 */
#define CPU_SIMD_SSE2     (1 << 1)   /**< SSE2指令集 */
#define CPU_SIMD_SSE3     (1 << 2)   /**< SSE3指令集 */
#define CPU_SIMD_SSSE3    (1 << 3)   /**< SSSE3指令集 */
#define CPU_SIMD_SSE41    (1 << 4)   /**< SSE4.1指令集 */
#define CPU_SIMD_SSE42    (1 << 5)   /**< SSE4.2指令集 */
#define CPU_SIMD_AVX      (1 << 6)   /**< AVX指令集 */
#define CPU_SIMD_AVX2     (1 << 7)   /**< AVX2指令集 */
#define CPU_SIMD_AVX512F  (1 << 8)   /**< AVX-512 Foundation指令集 */
#define CPU_SIMD_AVX512DQ (1 << 9)   /**< AVX-512 DQ指令集 */
#define CPU_SIMD_AVX512BW (1 << 10)  /**< AVX-512 BW指令集 */
#define CPU_SIMD_AVX512VL (1 << 13)  /**< AVX-512 VL指令集 */
#define CPU_SIMD_AVX512ER (1 << 14)  /**< AVX-512 ER指令集 */
#define CPU_SIMD_AVX512CD (1 << 15)  /**< AVX-512 CD指令集 */
#define CPU_SIMD_AVX512VBMI (1 << 16) /**< AVX-512 VBMI指令集 */
#define CPU_SIMD_AVX512VNNI (1 << 17) /**< AVX-512 VNNI指令集 */
#define CPU_SIMD_AVX512VPCLMULQDQ (1 << 18) /**< AVX-512 VPCLMULQDQ */
#define CPU_SIMD_AVX512BITALG (1 << 19) /**< AVX-512 BITALG指令集 */
#define CPU_SIMD_AVX512VBMI2 (1 << 20) /**< AVX-512 VBMI2指令集 */
#define CPU_SIMD_FMA     (1 << 21)  /**< FMA融合乘加指令集 */
#define CPU_SIMD_NEON     (1 << 11)  /**< ARM NEON指令集 */
#define CPU_SIMD_SVE      (1 << 12)  /**< ARM SVE指令集 */
/** @} */

/**
 * @brief GPU设备信息
 *
 * 存储计算设备的完整硬件信息。
 * 对于CPU后端，包含CPU品牌、架构、缓存、SIMD支持等真实硬件数据。
 * 对于GPU后端，包含GPU型号、显存、计算单元数等真实硬件数据。
 * 所有字段均从真实硬件检测获取，禁止使用模拟/硬编码值。
 */
typedef struct {
    int device_id;              /**< 设备ID */
    char name[256];             /**< 设备名称（真实型号，如"Intel(R) Core(TM) i7-10700K"） */
    GpuDeviceType type;         /**< 设备类型 */
    size_t total_memory;        /**< 总内存（字节），CPU：物理内存总量，GPU：显存总量 */
    size_t free_memory;         /**< 空闲内存（字节），CPU：可用物理内存，GPU：空闲显存 */
    int compute_units;          /**< 计算单元数，CPU：逻辑核心数，GPU：计算单元数 */
    int max_work_group_size;    /**< 最大工作组大小 */
    float clock_speed;          /**< 时钟速度（MHz），从真实硬件检测获取 */
    int supports_double;        /**< 是否支持双精度 */
    int supports_half;          /**< 是否支持半精度 */

    /* CPU特有的硬件信息（从真实硬件检测获取，禁止模拟值） */
    char vendor[64];             /**< CPU制造商，如"GenuineIntel"、"AuthenticAMD"、"Apple" */
    char architecture[32];       /**< CPU架构，如"x86_64"、"ARM64"、"ARMv8" */
    int physical_cores;          /**< 物理核心数（非超线程逻辑核心） */
    int logical_cores;           /**< 逻辑核心数（含超线程） */
    size_t l1_cache;             /**< L1数据缓存大小（字节） */
    size_t l2_cache;             /**< L2缓存大小（字节） */
    size_t l3_cache;             /**< L3缓存大小（字节），没有则填0 */
    unsigned int simd_flags;     /**< SIMD指令集支持标志位（CPU_SIMD_*位的组合） */
    char driver_version[64];     /**< 驱动版本号 */
    char runtime_version[64];    /**< 运行时版本号 */
} GpuDeviceInfo;

/**
 * @brief GPU内存句柄
 */
typedef struct GpuMemory GpuMemory;

/**
 * @brief GPU内核句柄
 */
typedef struct GpuKernel GpuKernel;

/**
 * @brief GPU流句柄
 */
typedef struct GpuStream GpuStream;

/** GPU上下文句柄前向声明位于文件头部（第42行），此处无需重复 */
/**
 * @brief 后端可用性标志位
 * @anchor GPU_BACKEND_FLAG
 * @{
 */
#define GPU_BACKEND_FLAG_CPU      (1 << 0)  /**< CPU后端可用 */
#define GPU_BACKEND_FLAG_CUDA     (1 << 1)  /**< CUDA后端可用 */
#define GPU_BACKEND_FLAG_OPENCL   (1 << 2)  /**< OpenCL后端可用 */
#define GPU_BACKEND_FLAG_VULKAN   (1 << 3)  /**< Vulkan后端可用 */
#define GPU_BACKEND_FLAG_METAL    (1 << 4)  /**< Metal后端可用 */
#define GPU_BACKEND_FLAG_ASCEND   (1 << 5)  /**< 华为昇腾NPU后端可用 */
#define GPU_BACKEND_FLAG_CAMBRICON (1 << 6) /**< 寒武纪MLU后端可用 */
#define GPU_BACKEND_FLAG_TPU      (1 << 7)  /**< 谷歌TPU后端可用 */
#define GPU_BACKEND_FLAG_ROCM     (1 << 8)  /**< AMD ROCm GPU后端可用 */
#define GPU_BACKEND_FLAG_INTEL    (1 << 9)  /**< Intel GPU后端可用 */
/** @} */

/**
 * @brief GPU后端不可用原因码
 *
 * 与 diagnostic 字符串配合使用，提供结构化的不可用原因。
 */
typedef enum {
    GPU_FAILURE_NONE = 0,               /**< 无故障，后端可用 */
    GPU_FAILURE_INTERFACE_NOT_FOUND,    /**< 后端接口未注册（后端未编译到系统中） */
    GPU_FAILURE_LIBRARY_NOT_FOUND,      /**< 运行时库未找到（如cudart.dll/cuda.dll不存在） */
    GPU_FAILURE_LIBRARY_VERSION_MISMATCH, /**< 运行时库版本不兼容（需CUDA 11+但仅检测到10.x） */
    GPU_FAILURE_DRIVER_NOT_INSTALLED,   /**< 驱动程序未安装（NVIDIA驱动/AMD驱动等缺失） */
    GPU_FAILURE_DRIVER_VERSION_MISMATCH,/**< 驱动程序版本不满足最低要求 */
    GPU_FAILURE_DEVICE_NOT_FOUND,       /**< 无兼容设备（无GPU/设备被禁用） */
    GPU_FAILURE_DEVICE_INCOMPATIBLE,    /**< 设备计算能力不足（如CUDA Compute Capability < 3.5） */
    GPU_FAILURE_INIT_FAILED,            /**< 后端初始化失败（内部错误） */
    GPU_FAILURE_OUT_OF_MEMORY,          /**< 显存或系统内存不足 */
    GPU_FAILURE_API_NOT_SUPPORTED,      /**< 所需API版本不被系统支持 */
    GPU_FAILURE_PERMISSION_DENIED,      /**< 权限不足（如无root/管理员权限） */
    GPU_FAILURE_ARCHITECTURE_MISMATCH,  /**< 架构不匹配（如32位进程无法加载64位驱动库） */
    GPU_FAILURE_PLATFORM_UNSUPPORTED    /**< 当前操作系统不支持此后端 */
} GpuBackendFailureCode;

/**
 * @brief 后端可用性信息
 *
 * 描述系统上每个GPU后端的实际可用状态，
 * 包括初始化状态、设备数量和详细诊断信息。
 */
typedef struct {
    GpuBackend backend;                 /**< 后端类型 */
    int is_available;                   /**< 是否可用（运行时库可加载） */
    int device_count;                   /**< 可用设备数量 */
    GpuBackendFailureCode failure_code; /**< 结构化不可用原因码 */
    char device_name[256];              /**< 首个设备的名称（如有） */
    char diagnostic[512];               /**< 诊断信息（人类可读的失败原因） */
    char driver_version[128];           /**< 驱动版本字符串（如 "555.99", "24.10"） */
    char runtime_version[128];          /**< 运行时库版本（如 "12.5", "2.1.23"） */
} GpuBackendAvailability;

/**
 * @brief 获取后端人类可读名称
 *
 * 返回GPU后端类型的字符串名称。
 *
 * @param backend 后端类型
 * @return const char* 后端名称字符串（如"CUDA(NVIDIA)"、"OpenCL"等）
 */
const char* gpu_backend_name(GpuBackend backend);

/* K-009: GPU后端硬件验证状态查询 */
int gpu_backend_validation_level(GpuBackend backend);
const char* gpu_backend_validation_string(GpuBackend backend);

/**
 * @brief 检测单个GPU后端的可用性（不初始化）
 *
 * 探测指定GPU后端在当前系统中是否可用。
 * 此函数只会尝试检测，不会实际初始化后端。
 * 如要初始化请调用 gpu_init。
 *
 * @param backend 要探测的后端类型
 * @param info [out] 可用性信息输出缓冲区（可为NULL）
 * @return int 1=可用，0=不可用，-1=参数错误
 */
int gpu_probe_backend(GpuBackend backend, GpuBackendAvailability* info);

/**
 * @brief 检测所有GPU后端的可用性
 *
 * 探测所有GPU后端在当前系统中的可用状态，
 * 以位掩码形式返回可用后端集合。
 *
 * @param infos [out] 可用性信息数组（长度至少为5），可为NULL
 * @param max_infos infos数组最大容量
 * @return unsigned int 可用后端位掩码（GPU_BACKEND_FLAG_*位的组合）
 */
unsigned int gpu_get_available_backends(GpuBackendAvailability* infos, int max_infos);

/**
 * @brief 自动检测并选择最佳可用GPU后端
 * 
 * 按优先级检测系统可用的GPU后端：CUDA → Vulkan → OpenCL → Metal
 * 返回检测到的最优可用后端，如果没有任何GPU后端可用则返回CPU。
 * 此函数会实际尝试加载后端运行时库来验证可用性。
 * 
 * @return GpuBackend 检测到的最佳可用后端
 */
GpuBackend gpu_auto_select(void);

/**
 * @brief 初始化GPU系统（自动检测模式）
 * 
 * 自动检测顺序：CUDA → ROCm → Intel → Vulkan → OpenCL → Metal → 昇腾 → 寒武纪 → TPU → CPU
 * 
 * - 如果backend为GPU_BACKEND_CPU（自动模式），按检测顺序依次尝试每个后端，使用首个成功初始化的后端
 * - 如果指定了具体后端（如GPU_BACKEND_CUDA），仅初始化该后端，失败直接返回-1
 * - 硬件自适应原则：需求明确要求无GPU时使用CPU，CUDA/ROCm/Vulkan等后端初始化失败
 *   会自动回退到CPU SIMD计算，这是正常的硬件配置自适应行为，而非降级。
 * 
 * @param backend 首选后端（传入GPU_BACKEND_CPU触发自动检测模式）
 * @return int 成功返回0，失败返回-1
 */
int gpu_init(GpuBackend backend);

/**
 * @brief GPU自动初始化（检测→选择→初始化，硬件自适应）
 *
 * 自动检测可用硬件：CUDA → ROCm → Intel → Vulkan → OpenCL → Metal → 昇腾 → 寒武纪 → TPU
 * 所有GPU加速后端不可用时自动回退到CPU后端。
 * 这是需求明确要求的硬件配置自适应行为（非降级），确保系统可在低端硬件运行。
 *
 * @param backend_out 输出实际选中的后端 (可为NULL)
 * @return 0=成功, -1=连CPU都不可用（极罕见）
 */
int gpu_auto_init(GpuBackend* backend_out);

/**
 * @brief 查询当前使用后端是否为CPU
 *
 * 仅在初始化阶段因无GPU加速硬件可用而回退到CPU时返回1。
 * 这是正常的硬件配置自适应行为，非降级处理。
 *
 * @return 1=当前使用CPU后端, 0=当前使用GPU后端或未初始化
 */
int gpu_is_cpu_backend(void);

/* GPU可用性快速检测 */
int gpu_is_available(void);

/**
 * @brief 清理GPU系统
 */
void gpu_cleanup(void);

/**
 * @brief 获取可用设备数量
 * 
 * @param backend 后端类型
 * @return int 设备数量，失败返回-1
 */
int gpu_get_device_count(GpuBackend backend);

/**
 * @brief 获取设备信息
 * 
 * @param backend 后端类型
 * @param device_index 设备索引
 * @param info 设备信息输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int gpu_get_device_info(GpuBackend backend, int device_index, GpuDeviceInfo* info);

/**
 * @brief 统一CPU硬件检测接口
 *
 * 从真实硬件检测获取CPU的完整信息（品牌、架构、核心数、缓存、SIMD等）。
 * 此函数是项目中所有CPU硬件检测的唯一权威入口。
 * gpu_memory_pool.c 和 gpu_cpu.c 中的CPU检测代码统一调用此接口，
 * 消除三重重复。
 *
 * 支持 x86/x64（CPUID）和 ARM64 架构。
 * 所有数据均从真实硬件读取，禁止使用模拟值。
 *
 * @param info 设备信息输出缓冲区（GpuDeviceInfo，至少256+字节）
 * @return int 成功返回0，失败返回-1
 */
int gpu_hardware_get_cpu_info(GpuDeviceInfo* info);

/**
 * @brief 创建GPU上下文
 * 
 * @param backend 后端类型
 * @param device_index 设备索引
 * @return GpuContext* 上下文句柄，失败返回NULL
 */
GpuContext* gpu_context_create(GpuBackend backend, int device_index);

/**
 * @brief 释放GPU上下文
 * 
 * @param context 上下文句柄
 */
void gpu_context_free(GpuContext* context);

/**
 * @brief 分配GPU内存
 * 
 * @param context 上下文句柄
 * @param size 内存大小（字节）
 * @param memory_type 内存类型
 * @return GpuMemory* 内存句柄，失败返回NULL
 */
GpuMemory* gpu_memory_alloc(GpuContext* context, size_t size, GpuMemoryType memory_type);

/**
 * @brief 释放GPU内存
 * 
 * @param memory 内存句柄
 */
void gpu_memory_free(GpuMemory* memory);

/**
 * @brief 复制数据到GPU内存
 * 
 * @param dst 目标内存句柄
 * @param src 源数据指针
 * @param size 数据大小（字节）
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_copy_to_device(GpuMemory* dst, const void* src, size_t size);

/**
 * @brief 从GPU内存复制数据
 * 
 * @param dst 目标数据指针
 * @param src 源内存句柄
 * @param size 数据大小（字节）
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_copy_from_device(void* dst, GpuMemory* src, size_t size);

/**
 * @brief 在GPU内存之间复制数据
 * 
 * @param dst 目标内存句柄
 * @param src 源内存句柄
 * @param size 数据大小（字节）
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_copy_device_to_device(GpuMemory* dst, GpuMemory* src, size_t size);

/**
 * @brief 异步复制数据到GPU内存（带流同步）
 * 
 * 将主机内存中的数据异步复制到GPU设备内存。
 * 操作在指定的GPU流中执行，调用后立即返回。
 * 使用gpu_stream_synchronize等待操作完成。
 * 
 * @param dst 目标GPU内存句柄
 * @param src 源数据指针（主机内存）
 * @param size 数据大小（字节）
 * @param stream GPU流句柄（为NULL时使用默认流）
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_copy_to_device_async(GpuMemory* dst, const void* src, size_t size, GpuStream* stream);

/**
 * @brief 异步从GPU内存复制数据（带流同步）
 * 
 * 将GPU设备内存中的数据异步复制到主机内存。
 * 操作在指定的GPU流中执行，调用后立即返回。
 * 使用gpu_stream_synchronize等待操作完成。
 * 
 * @param dst 目标数据指针（主机内存）
 * @param src 源GPU内存句柄
 * @param size 数据大小（字节）
 * @param stream GPU流句柄（为NULL时使用默认流）
 * @return int 成功返回0，失败返回-1
 */
int gpu_memory_copy_from_device_async(void* dst, GpuMemory* src, size_t size, GpuStream* stream);

/**
 * @brief 创建GPU内核
 * 
 * @param context 上下文句柄
 * @param kernel_source 内核源代码
 * @param kernel_name 内核函数名
 * @return GpuKernel* 内核句柄，失败返回NULL
 */
GpuKernel* gpu_kernel_create(GpuContext* context, const char* kernel_source, const char* kernel_name);

/**
 * @brief 释放GPU内核
 * 
 * @param kernel 内核句柄
 */
void gpu_kernel_free(GpuKernel* kernel);

/**
 * @brief 设置内核参数
 * 
 * @param kernel 内核句柄
 * @param arg_index 参数索引
 * @param arg_size 参数大小
 * @param arg_value 参数值指针
 * @return int 成功返回0，失败返回-1
 */
int gpu_kernel_set_arg(GpuKernel* kernel, int arg_index, size_t arg_size, const void* arg_value);

/**
 * @brief 执行GPU内核
 * 
 * @param kernel 内核句柄
 * @param global_work_size 全局工作大小
 * @param local_work_size 本地工作大小（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int gpu_kernel_execute(GpuKernel* kernel, size_t global_work_size, size_t local_work_size);

/**
 * @brief 执行GPU内核（多维）
 * 
 * @param kernel 内核句柄
 * @param work_dim 工作维度（1-3）
 * @param global_work_size 全局工作大小数组
 * @param local_work_size 本地工作大小数组（可为NULL）
 * @return int 成功返回0，失败返回-1
 */
int gpu_kernel_execute_nd(GpuKernel* kernel, int work_dim,
                         const size_t* global_work_size,
                         const size_t* local_work_size);

/**
 * @brief 创建GPU流
 * 
 * @param context 上下文句柄
 * @return GpuStream* 流句柄，失败返回NULL
 */
GpuStream* gpu_stream_create(GpuContext* context);

/**
 * @brief 释放GPU流
 * 
 * @param stream 流句柄
 */
void gpu_stream_free(GpuStream* stream);

/**
 * @brief 同步GPU流
 * 
 * @param stream 流句柄
 * @return int 成功返回0，失败返回-1
 */
int gpu_stream_synchronize(GpuStream* stream);

/**
 * @brief 查询GPU流是否完成
 * 
 * @param stream 流句柄
 * @return int 完成返回1，未完成返回0，失败返回-1
 */
int gpu_stream_query(GpuStream* stream);

/**
 * @brief 获取GPU错误信息
 * 
 * @return const char* 错误信息字符串
 */
const char* gpu_get_error_string(void);

/**
 * @brief 获取当前使用的后端
 * 
 * @return GpuBackend 当前后端
 */
GpuBackend gpu_get_current_backend(void);

/**
 * @brief 获取GPU内存信息
 * 
 * @param context 上下文句柄
 * @param total_memory 总内存输出（字节）
 * @param free_memory 空闲内存输出（字节）
 * @return int 成功返回0，失败返回-1
 */
int gpu_get_memory_info(GpuContext* context, size_t* total_memory, size_t* free_memory);

/**
 * @brief 重置GPU设备
 * 
 * @param context 上下文句柄
 * @return int 成功返回0，失败返回-1
 */
int gpu_device_reset(GpuContext* context);

/**
 * @brief GPU后端接口结构体
 * 
 * 每个GPU后端需要实现这个接口中的所有函数。
 * 这允许系统在运行时动态选择和切换后端。
 */
typedef struct {
    /** 后端名称 */
    const char* name;
    
    /** 后端类型 */
    GpuBackend backend_type;
    
    /** 初始化后端 */
    int (*init)(void);
    
    /** 清理后端 */
    void (*cleanup)(void);
    
    /** 获取设备数量 */
    int (*get_device_count)(void);
    
    /** 获取设备信息 */
    int (*get_device_info)(int device_index, GpuDeviceInfo* info);
    
    /** 创建设备上下文 */
    GpuContext* (*context_create)(int device_index);
    
    /** 释放设备上下文 */
    void (*context_free)(GpuContext* context);
    
    /** 分配内存 */
    GpuMemory* (*memory_alloc)(GpuContext* context, size_t size, GpuMemoryType memory_type);
    
    /** 释放内存 */
    void (*memory_free)(GpuMemory* memory);
    
    /** 复制数据到设备 */
    int (*memory_copy_to_device)(GpuMemory* dst, const void* src, size_t size);
    
    /** 从设备复制数据 */
    int (*memory_copy_from_device)(void* dst, GpuMemory* src, size_t size);
    
    /** 设备间复制数据 */
    int (*memory_copy_device_to_device)(GpuMemory* dst, GpuMemory* src, size_t size);
    
    /** 异步复制数据到设备（带流同步） */
    int (*memory_copy_to_device_async)(GpuMemory* dst, const void* src, size_t size, GpuStream* stream);
    
    /** 异步从设备复制数据（带流同步） */
    int (*memory_copy_from_device_async)(void* dst, GpuMemory* src, size_t size, GpuStream* stream);
    
    /** 创建内核 */
    GpuKernel* (*kernel_create)(GpuContext* context, const char* kernel_source, const char* kernel_name);
    
    /** 释放内核 */
    void (*kernel_free)(GpuKernel* kernel);
    
    /** 设置内核参数 */
    int (*kernel_set_arg)(GpuKernel* kernel, int arg_index, size_t arg_size, const void* arg_value);
    
    /** 执行内核 */
    int (*kernel_execute)(GpuKernel* kernel, size_t global_work_size, size_t local_work_size);
    
    /** 执行内核（多维） */
    int (*kernel_execute_nd)(GpuKernel* kernel, int work_dim,
                            const size_t* global_work_size,
                            const size_t* local_work_size);
    
    /** 创建流 */
    GpuStream* (*stream_create)(GpuContext* context);
    
    /** 释放流 */
    void (*stream_free)(GpuStream* stream);
    
    /** 同步流 */
    int (*stream_synchronize)(GpuStream* stream);
    
    /** 查询流状态 */
    int (*stream_query)(GpuStream* stream);
    
    /** 获取内存信息 */
    int (*get_memory_info)(GpuContext* context, size_t* total_memory, size_t* free_memory);
    
    /** 重置设备 */
    int (*device_reset)(GpuContext* context);
    
    /** 获取错误信息 */
    const char* (*get_error_string)(void);
} GpuBackendInterface;

// ==================== 自动内核优化集成接口 ====================

/**
 * @brief 记录内核执行性能数据
 * 
 * @param context GPU上下文
 * @param kernel_type 内核类型
 * @param kernel_name 内核名称
 * @param input_size 输入数据大小
 * @param output_size 输出数据大小
 * @param global_work_size 全局工作大小
 * @param params 使用的优化参数
 * @param execution_time_ms 执行时间（毫秒）
 * @return int 成功返回0，失败返回-1
 */
int gpu_kernel_profile(GpuContext* context,
                       KernelType kernel_type,
                       const char* kernel_name,
                       size_t input_size,
                       size_t output_size,
                       const size_t* global_work_size,
                       const KernelOptimizationParams* params,
                       double execution_time_ms);

/**
 * @brief 获取最优内核优化参数
 * 
 * @param context GPU上下文
 * @param kernel_type 内核类型
 * @param kernel_name 内核名称
 * @param input_size 输入数据大小
 * @param output_size 输出数据大小
 * @param params 优化参数输出
 * @return int 成功返回0，失败返回-1
 */
int gpu_kernel_get_optimal_params(GpuContext* context,
                                  KernelType kernel_type,
                                  const char* kernel_name,
                                  size_t input_size,
                                  size_t output_size,
                                  KernelOptimizationParams* params);

/**
 * @brief 执行内核自动调优
 * 
 * @param context GPU上下文
 * @param kernel_type 内核类型
 * @param kernel_name 内核名称
 * @param input_size 输入数据大小
 * @param output_size 输出数据大小
 * @return double 优化后的执行时间（毫秒），失败返回-1
 */
double gpu_kernel_tune(GpuContext* context,
                       KernelType kernel_type,
                       const char* kernel_name,
                       size_t input_size,
                       size_t output_size);

/**
 * @brief 获取内核优化统计信息
 * 
 * @param context GPU上下文
 * @param total_profiles 总分析次数输出
 * @param total_optimizations 总优化次数输出
 * @param average_speedup 平均加速比输出
 * @return int 成功返回0，失败返回-1
 */
int gpu_kernel_optimizer_get_stats(GpuContext* context,
                                   int* total_profiles,
                                   int* total_optimizations,
                                   double* average_speedup);

/**
 * @brief 建议工作组大小
 * 
 * @param context GPU上下文
 * @param global_size 全局工作大小
 * @param max_work_group_size 最大工作组大小
 * @param kernel_type 内核类型
 * @return size_t 建议的工作组大小
 */
size_t gpu_suggest_work_group(GpuContext* context,
                              size_t global_size,
                              size_t max_work_group_size,
                              KernelType kernel_type);

/* 自动kernel优化器数据库管理API */

/** @brief 清空自动kernel优化器的性能缓存 */
int gpu_kernel_optimizer_clear_cache(GpuContext* context);

/** @brief 保存kernel优化数据库到文件 */
int gpu_kernel_optimizer_save_db(GpuContext* context, const char* filepath);

/** @brief 从文件加载kernel优化数据库 */
int gpu_kernel_optimizer_load_db(GpuContext* context, const char* filepath);

/** @brief 获取历史最优kernel执行记录 */
int gpu_kernel_optimizer_get_best(GpuContext* context, size_t* out_input,
                                   size_t* out_output, double* out_time);

/** @brief 预测指定输入输出尺寸的kernel执行时间 */
int gpu_kernel_optimizer_predict(GpuContext* context, size_t input_size,
                                  size_t output_size, double* predicted_time);

// ==================== GPU内核缓存管理接口 ====================

/**
 * @brief 内核缓存统计信息
 */
typedef struct {
    int total_entries;             /**< 总条目数 */
    int active_entries;            /**< 有效条目数 */
    int cache_hits;                /**< 缓存命中次数 */
    int cache_misses;              /**< 缓存未命中次数 */
    int eviction_count;            /**< 淘汰次数 */
    size_t total_source_bytes;     /**< 缓存的源码总大小（字节） */
} GpuKernelCacheStats;

/**
 * @brief 清除内核缓存
 *
 * 释放所有缓存的内核对象。当GPU上下文重置或显存不足时应调用此函数。
 *
 * @param context GPU上下文
 */
void gpu_kernel_cache_clear(GpuContext* context);

/**
 * @brief 获取内核缓存统计信息
 *
 * @param context GPU上下文
 * @param stats 统计信息输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int gpu_kernel_cache_get_stats(GpuContext* context, GpuKernelCacheStats* stats);

/**
 * @brief 设置内核缓存容量
 *
 * @param context GPU上下文
 * @param capacity 新的缓存容量（小于1则不调整）
 * @return int 成功返回0，失败返回-1
 */
int gpu_kernel_cache_set_capacity(GpuContext* context, int capacity);

// ==================== GPU混合精度训练优化接口 ====================

/**
 * @brief GPU混合精度训练模式
 */
typedef enum {
    GPU_MIXED_PRECISION_DISABLED = 0,
    GPU_MIXED_PRECISION_FP16 = 1,
    GPU_MIXED_PRECISION_DYNAMIC = 2,
    GPU_MIXED_PRECISION_AUTO = 3
} GpuMixedPrecisionMode;

/**
 * @brief GPU混合精度缩放策略
 */
typedef enum {
    GPU_LOSS_SCALE_STATIC = 0,
    GPU_LOSS_SCALE_DYNAMIC = 1,
    GPU_LOSS_SCALE_ADAPTIVE = 2
} GpuLossScaleStrategy;

typedef enum {
    GPU_LOSS_MSE = 0,       /**< 均方误差 */
    GPU_LOSS_MAE = 1,       /**< 平均绝对误差 */
    GPU_LOSS_CROSS_ENTROPY = 2, /**< 交叉熵 */
    GPU_LOSS_KL_DIVERGENCE = 3  /**< KL散度 */
} GpuLossType;

/**
 * @brief GPU混合精度训练配置
 */
typedef struct {
    GpuMixedPrecisionMode mode;
    GpuLossScaleStrategy scale_strategy;
    float initial_loss_scale;
    float max_loss_scale;
    float min_loss_scale;
    int scale_update_interval;
    int overflow_check_interval;
    float scale_growth_factor;
    float scale_decay_factor;
    int enable_fp16_storage;
    int enable_fp16_arithmetic;
    int check_nan_inf;
} GpuMixedPrecisionConfig;

/**
 * @brief GPU混合精度训练上下文（不透明句柄）
 */
typedef struct GpuMixedPrecisionContext GpuMixedPrecisionContext;

/**
 * @brief 创建GPU混合精度训练上下文
 * 
 * @param context GPU上下文
 * @param config 混合精度配置（为NULL时使用默认配置）
 * @return GpuMixedPrecisionContext* 成功返回上下文指针，失败返回NULL
 */
GpuMixedPrecisionContext* gpu_mixed_precision_create(GpuContext* context,
                                                     const GpuMixedPrecisionConfig* config);

/**
 * @brief 销毁GPU混合精度训练上下文
 * 
 * @param mp_ctx 混合精度上下文
 */
void gpu_mixed_precision_destroy(GpuMixedPrecisionContext* mp_ctx);

/**
 * @brief 在GPU上应用损失缩放
 * 
 * 对损失值张量应用缩放，防止FP16梯度下溢。
 * 
 * @param mp_ctx 混合精度上下文
 * @param loss_gpu GPU上的损失值指针
 * @param size 元素数量
 * @return int 成功返回0，失败返回-1
 */
int gpu_mixed_precision_scale_loss(GpuMixedPrecisionContext* mp_ctx,
                                   float* loss_gpu,
                                   size_t size);

/**
 * @brief 在GPU上缩放梯度
 * 
 * 将梯度按当前缩放因子缩放，用于反向传播后的梯度调整。
 * 
 * @param mp_ctx 混合精度上下文
 * @param gradients_gpu GPU上的梯度指针
 * @param size 梯度元素数量
 * @return int 成功返回0，失败返回-1
 */
int gpu_mixed_precision_scale_gradients(GpuMixedPrecisionContext* mp_ctx,
                                        float* gradients_gpu,
                                        size_t size);

/**
 * @brief 在GPU上取消梯度缩放
 * 
 * 将梯度除以缩放因子，恢复原始梯度值用于参数更新。
 * 
 * @param mp_ctx 混合精度上下文
 * @param gradients_gpu GPU上的梯度指针
 * @param size 梯度元素数量
 * @return int 成功返回0，失败返回-1
 */
int gpu_mixed_precision_unscale_gradients(GpuMixedPrecisionContext* mp_ctx,
                                          float* gradients_gpu,
                                          size_t size);

/**
 * @brief 检查GPU张量是否溢出（NaN/Inf）
 * 
 * @param mp_ctx 混合精度上下文
 * @param data_gpu GPU上的数据指针
 * @param size 元素数量
 * @param overflow_flag 溢出标志输出（1=溢出，0=正常）
 * @return int 成功返回0，失败返回-1
 */
int gpu_mixed_precision_check_overflow(GpuMixedPrecisionContext* mp_ctx,
                                       const float* data_gpu,
                                       size_t size,
                                       int* overflow_flag);

/**
 * @brief 更新损失缩放因子
 * 
 * 根据最近的溢出统计动态调整损失缩放因子。
 * 
 * @param mp_ctx 混合精度上下文
 * @param overflow_detected 是否检测到溢出
 * @return float 更新后的缩放因子
 */
float gpu_mixed_precision_update_loss_scale(GpuMixedPrecisionContext* mp_ctx,
                                            int overflow_detected);

/**
 * @brief 将FP32张量转换为FP16张量（GPU内存操作）
 * 
 * @param mp_ctx 混合精度上下文
 * @param fp32_gpu GPU上的FP32数据指针
 * @param fp16_gpu GPU上的FP16数据指针（输出）
 * @param size 元素数量
 * @return int 成功返回0，失败返回-1
 */
int gpu_mixed_precision_convert_fp32_to_fp16(GpuMixedPrecisionContext* mp_ctx,
                                             const float* fp32_gpu,
                                             void* fp16_gpu,
                                             size_t size);

/**
 * @brief 将FP16张量转换为FP32张量（GPU内存操作）
 * 
 * @param mp_ctx 混合精度上下文
 * @param fp16_gpu GPU上的FP16数据指针
 * @param fp32_gpu GPU上的FP32数据指针（输出）
 * @param size 元素数量
 * @return int 成功返回0，失败返回-1
 */
int gpu_mixed_precision_convert_fp16_to_fp32(GpuMixedPrecisionContext* mp_ctx,
                                             const void* fp16_gpu,
                                             float* fp32_gpu,
                                             size_t size);

/**
 * @brief 获取混合精度训练统计信息
 * 
 * @param mp_ctx 混合精度上下文
 * @param current_scale 当前缩放因子输出
 * @param overflow_count 溢出次数输出
 * @param total_steps 总步数输出
 * @return int 成功返回0，失败返回-1
 */
int gpu_mixed_precision_get_stats(GpuMixedPrecisionContext* mp_ctx,
                                  float* current_scale,
                                  int* overflow_count,
                                  int* total_steps);

// ==================== 多GPU协同计算接口 ====================

/**
 * @brief 多GPU通信模式
 */
typedef enum {
    GPU_COMM_ALL_REDUCE = 0,
    GPU_COMM_BROADCAST = 1,
    GPU_COMM_SCATTER = 2,
    GPU_COMM_GATHER = 3,
    GPU_COMM_REDUCE = 4
} GpuCommMode;

/**
 * @brief 多GPU协同配置
 */
typedef struct {
    int num_devices;
    int* device_indices;
    GpuBackend* device_backends;
    GpuCommMode default_comm_mode;
    int enable_peer_access;
    int sync_interval;
    float timeout_seconds;
    int enable_fault_tolerance;
    int max_retries;
} GpuMultiGpuConfig;

/**
 * @brief 多GPU协同上下文（不透明句柄）
 */
typedef struct GpuMultiGpuContext GpuMultiGpuContext;

/**
 * @brief 初始化多GPU协同系统
 * 
 * @param config 多GPU配置
 * @return GpuMultiGpuContext* 成功返回上下文指针，失败返回NULL
 */
GpuMultiGpuContext* gpu_multi_gpu_init(const GpuMultiGpuConfig* config);

/**
 * @brief 清理多GPU协同系统
 * 
 * @param mg_ctx 多GPU上下文
 */
void gpu_multi_gpu_cleanup(GpuMultiGpuContext* mg_ctx);

/**
 * @brief 获取参与协同的GPU数量
 * 
 * @param mg_ctx 多GPU上下文
 * @return int GPU数量，失败返回-1
 */
int gpu_multi_gpu_get_device_count(GpuMultiGpuContext* mg_ctx);

/**
 * @brief 获取指定索引的GPU上下文
 * 
 * @param mg_ctx 多GPU上下文
 * @param device_index 设备索引（0到num_devices-1）
 * @return GpuContext* GPU上下文指针，失败返回NULL
 */
GpuContext* gpu_multi_gpu_get_context(GpuMultiGpuContext* mg_ctx,
                                      int device_index);

/**
 * @brief 执行多GPU All-Reduce操作
 * 
 * 在所有参与的GPU上对梯度进行求和归约，确保每个GPU获得完整的梯度。
 * 
 * @param mg_ctx 多GPU上下文
 * @param data_per_device 每个GPU上的数据指针数组
 * @param size 每个设备上的元素数量
 * @param comm_mode 通信模式
 * @return int 成功返回0，失败返回-1
 */
int gpu_multi_gpu_all_reduce(GpuMultiGpuContext* mg_ctx,
                             float** data_per_device,
                             size_t size,
                             GpuCommMode comm_mode);

/**
 * @brief 在多GPU之间广播数据
 * 
 * 将源设备上的数据广播到所有其他设备。
 * 
 * @param mg_ctx 多GPU上下文
 * @param data 数据指针
 * @param size 数据元素数量
 * @param root_device 根设备索引（广播源）
 * @param comm_mode 通信模式
 * @return int 成功返回0，失败返回-1
 */
int gpu_multi_gpu_broadcast(GpuMultiGpuContext* mg_ctx,
                            float* data, size_t size,
                            int root_device, GpuCommMode comm_mode);

/**
 * @brief 同步所有GPU设备
 * 
 * 等待所有参与的GPU设备完成当前操作。
 * 
 * @param mg_ctx 多GPU上下文
 * @return int 成功返回0，失败返回-1
 */
int gpu_multi_gpu_synchronize(GpuMultiGpuContext* mg_ctx);

/**
 * @brief 在设备间分配工作负载
 * 
 * 根据设备计算能力将总工作量分配到各个GPU。
 * 
 * @param mg_ctx 多GPU上下文
 * @param total_work 总工作量
 * @param work_assignments 各设备分配的工作量数组（输出，长度=设备数）
 * @return int 成功返回0，失败返回-1
 */
int gpu_multi_gpu_distribute_work(GpuMultiGpuContext* mg_ctx,
                                  size_t total_work,
                                  size_t* work_assignments);

/**
 * @brief 获取多GPU协同统计信息
 * 
 * @param mg_ctx 多GPU上下文
 * @param total_communication_rounds 总通信轮次输出
 * @param total_bytes_transferred 总传输字节数输出
 * @param average_sync_time_ms 平均同步时间（毫秒）输出
 * @return int 成功返回0，失败返回-1
 */
int gpu_multi_gpu_get_stats(GpuMultiGpuContext* mg_ctx,
                            int* total_communication_rounds,
                            size_t* total_bytes_transferred,
                            float* average_sync_time_ms);

// ==================== GPU双缓冲数据传输接口 ====================

/**
 * @brief GPU双缓冲结构
 * 
 * 双缓冲机制实现计算与数据传输的重叠（overlap）。
 * front_buffer: 当前用于计算的数据缓冲区
 * back_buffer: 后台正在传输的数据缓冲区
 * stream: 用于异步传输的GPU流
 * 
 * 典型使用模式：
 * 1. 将下一批数据异步传输到back_buffer
 * 2. 在GPU上处理front_buffer中的数据
 * 3. 交换缓冲区：front <-> back
 * 4. 同步等待back_buffer传输完成
 * 5. 重复步骤1-4
 */
typedef struct {
    GpuMemory* front_buffer;      /**< 前端缓冲区（用于计算） */
    GpuMemory* back_buffer;       /**< 后端缓冲区（用于传输） */
    GpuStream* stream;            /**< 异步传输流 */
    size_t size;                  /**< 每个缓冲区大小（字节） */
    GpuMemoryType memory_type;    /**< 内存类型 */
    int is_initialized;           /**< 是否已初始化 */
    int is_swapped;               /**< 是否已交换（用于跟踪状态） */
    GpuContext* context;          /**< 所属GPU上下文 */
} GpuDoubleBuffer;

/**
 * @brief 创建GPU双缓冲
 * 
 * 分配两个大小相同的GPU内存缓冲区和一个异步传输流。
 * 
 * @param context GPU上下文
 * @param size 每个缓冲区大小（字节）
 * @param memory_type 内存类型（推荐GPU_MEMORY_DEVICE）
 * @return GpuDoubleBuffer* 成功返回双缓冲指针，失败返回NULL
 */
GpuDoubleBuffer* gpu_double_buffer_create(GpuContext* context, size_t size, GpuMemoryType memory_type);

/**
 * @brief 销毁GPU双缓冲
 * 
 * 释放两个缓冲区和一个传输流。
 * 
 * @param db 双缓冲指针
 */
void gpu_double_buffer_destroy(GpuDoubleBuffer* db);

/**
 * @brief 交换双缓冲
 * 
 * 交换front和back缓冲区指针。调用后：
 * 新的front_buffer = 原来的back_buffer（最新传输完成的数据）
 * 新的back_buffer = 原来的front_buffer（可写入新数据）
 * 
 * @param db 双缓冲指针
 * @return int 成功返回0，失败返回-1
 */
int gpu_double_buffer_swap(GpuDoubleBuffer* db);

/**
 * @brief 获取前端缓冲区（用于计算）
 * 
 * @param db 双缓冲指针
 * @return GpuMemory* 前端缓冲区指针，失败返回NULL
 */
GpuMemory* gpu_double_buffer_get_front(GpuDoubleBuffer* db);

/**
 * @brief 获取后端缓冲区（用于异步传输）
 * 
 * @param db 双缓冲指针
 * @return GpuMemory* 后端缓冲区指针，失败返回NULL
 */
GpuMemory* gpu_double_buffer_get_back(GpuDoubleBuffer* db);

/**
 * @brief 同步等待双缓冲传输完成
 * 
 * 等待后端缓冲区的异步传输操作完成。
 * 
 * @param db 双缓冲指针
 * @return int 成功返回0，失败返回-1
 */
int gpu_double_buffer_sync(GpuDoubleBuffer* db);

/**
 * @brief 异步传输数据到双缓冲的后端缓冲区
 * 
 * 将主机数据异步复制到back_buffer中。
 * 传输完成后，可通过gpu_double_buffer_swap交换到前端。
 * 
 * @param db 双缓冲指针
 * @param src 源数据指针（主机内存）
 * @param size 数据大小（字节，不能超过缓冲区大小）
 * @return int 成功返回0，失败返回-1
 */
int gpu_double_buffer_transfer_async(GpuDoubleBuffer* db, const void* src, size_t size);

// ==================== SDK可用性诊断接口 ====================

/**
 * @brief 检查GPU后端SDK文件在系统中是否存在
 *
 * 在不加载运行时库的前提下，检查指定GPU后端所需的SDK动态库文件
 * 是否存在于系统搜索路径中。此函数仅做文件存在性检查，不会初始化任何后端。
 *
 * @param backend 要检查的后端类型
 * @return int 1=SDK文件存在，0=未找到SDK文件，-1=参数错误或无需SDK的后端（如CPU）
 */
int gpu_check_sdk_file(GpuBackend backend);

/**
 * @brief 获取GPU后端SDK可用性的详细诊断报告
 *
 * 对每个GPU后端检查SDK文件存在性，并提供可操作的安装建议。
 * 生成的报告包含：
 * - 每个后端的SDK文件存在状态
 * - 已尝试的SDK文件列表
 * - 可行的安装/配置建议
 * - 当前系统上可用的GPU后端
 *
 * @param buffer 输出缓冲区
 * @param buffer_size 缓冲区大小
 * @return int 成功返回0，失败返回-1
 */
int gpu_get_sdk_diagnostic(char* buffer, size_t buffer_size);

// ==================== GPU训练运算接口 ====================

/**
 * @brief GPU优化器类型
 */
typedef enum {
    GPU_OPTIMIZER_SGD = 0,        /**< 随机梯度下降 */
    GPU_OPTIMIZER_MOMENTUM = 1,   /**< 带动量的SGD */
    GPU_OPTIMIZER_ADAM = 2,       /**< Adam优化器 */
    GPU_OPTIMIZER_RMSPROP = 3     /**< RMSProp优化器 */
} GpuOptimizerType;

/**
 * @brief GPU训练配置
 */
typedef struct {
    GpuOptimizerType optimizer;    /**< 优化器类型 */
    float learning_rate;           /**< 学习率 */
    float momentum;                /**< 动量系数（Momentum优化器） */
    float beta1;                   /**< Adam beta1 */
    float beta2;                   /**< Adam beta2 */
    float epsilon;                 /**< Adam epsilon */
    float weight_decay;            /**< 权重衰减 */
    float gradient_clip_norm;      /**< 梯度裁剪范数（0=不裁剪） */
    int loss_type;                 /**< 损失函数类型（GpuLossType枚举值） */
    int batch_size;                /**< 批大小 */
    int max_iterations;            /**< 最大迭代次数 */
    int enable_learning_rate_decay; /**< 是否启用学习率衰减 */
    float learning_rate_decay;     /**< 学习率衰减系数 */
    int decay_steps;               /**< 衰减步数间隔 */
    int enable_mixed_precision;    /**< 是否启用混合精度训练 */
    int enable_gradient_checkpointing; /**< 是否启用梯度检查点 */
    int backend_flags;               /**< 后端训练标志位（GPU_TRAIN_USE_*组合） */
    int precision_mode;              /**< 精度模式 */
} GpuTrainConfig;

typedef enum {
    GPU_TRAIN_USE_SGD               = 1 << 0,
    GPU_TRAIN_USE_ADAM              = 1 << 1,
    GPU_TRAIN_USE_MIXED_PRECISION   = 1 << 2,
    GPU_TRAIN_USE_GRADIENT_CLIP     = 1 << 3
} GpuTrainFlags;

/**
 * @brief 创建GPU训练配置（默认值）
 * 
 * @return GpuTrainConfig 默认训练配置
 */
GpuTrainConfig gpu_train_config_default(void);

/**
 * @brief 在GPU上编译训练内核（SGD权重更新）
 * 
 * 编译并缓存训练内核源代码，用于在GPU上执行梯度下降权重更新。
 * 内核根据指定的训练配置，自动选择合适的算法（SGD/Adam等）。
 * 
 * @param context GPU上下文
 * @param config 训练配置
 * @return int 成功返回0，失败返回-1
 */
int gpu_train_compile_kernels(GpuContext* context, const GpuTrainConfig* config);

/**
 * @brief 在GPU上执行SGD权重更新
 * 
 * 对权重张量执行随机梯度下降更新: w = w - lr * (grad + wd * w)
 * 所有权重、梯度数据均已在GPU设备内存中。
 * 
 * @param context GPU上下文
 * @param weights GPU权重指针
 * @param gradients GPU梯度指针
 * @param num_params 参数数量
 * @param learning_rate 学习率
 * @param weight_decay 权重衰减系数
 * @return int 成功返回0，失败返回-1
 */
int gpu_sgd_update(GpuContext* context, float* weights, const float* gradients,
                   size_t num_params, float learning_rate, float weight_decay);

/**
 * @brief 在GPU上执行带动量的SGD更新
 * 
 * 对权重张量执行带动量的SGD更新:
 * v = momentum * v - lr * (grad + wd * w)
 * w = w + v
 * 
 * @param context GPU上下文
 * @param weights GPU权重指针
 * @param gradients GPU梯度指针
 * @param velocity GPU速度/动量缓冲区指针
 * @param num_params 参数数量
 * @param learning_rate 学习率
 * @param momentum 动量系数
 * @param weight_decay 权重衰减系数
 * @return int 成功返回0，失败返回-1
 */
int gpu_momentum_update(GpuContext* context, float* weights, const float* gradients,
                        float* velocity, size_t num_params,
                        float learning_rate, float momentum, float weight_decay);

/**
 * @brief 在GPU上执行Adam权重更新
 * 
 * 对权重张量执行Adam优化器更新:
 * m = beta1 * m + (1 - beta1) * g
 * v = beta2 * v + (1 - beta2) * g^2
 * w = w - lr * m / (sqrt(v) + eps) - lr * wd * w
 * 
 * @param context GPU上下文
 * @param weights GPU权重指针
 * @param gradients GPU梯度指针
 * @param first_moment GPU一阶动量缓冲区指针
 * @param second_moment GPU二阶动量缓冲区指针
 * @param num_params 参数数量
 * @param learning_rate 学习率
 * @param beta1 Adam beta1
 * @param beta2 Adam beta2
 * @param epsilon Adam epsilon
 * @param weight_decay 权重衰减系数
 * @param timestep 当前时间步（从1开始）
 * @return int 成功返回0，失败返回-1
 */
int gpu_adam_update(GpuContext* context, float* weights, const float* gradients,
                    float* first_moment, float* second_moment,
                    size_t num_params, float learning_rate,
                    float beta1, float beta2, float epsilon,
                    float weight_decay, int timestep);

/**
 * @brief 在GPU上计算L2损失梯度
 * 
 * 在GPU上并行计算预测值和目标值之间的损失梯度。
 * loss = 0.5 * sum((pred - target)^2) / n
 * grad = (pred - target) / n
 * 
 * @param context GPU上下文
 * @param predictions GPU预测值指针
 * @param targets GPU目标值指针
 * @param gradients GPU梯度输出指针
 * @param num_elements 元素数量
 * @return int 成功返回0，失败返回-1
 */
int gpu_compute_l2_loss_gradient(GpuContext* context,
                                 const float* predictions, const float* targets,
                                 float* gradients, size_t num_elements);

/**
 * @brief 在GPU上执行矩阵乘法（用于训练前向传播）
 * 
 * C = alpha * A * B + beta * C
 * A: [M x K], B: [K x N], C: [M x N]
 * 
 * @param context GPU上下文
 * @param A 矩阵A（GPU指针）
 * @param B 矩阵B（GPU指针）
 * @param C 矩阵C输出（GPU指针）
 * @param M A行数
 * @param N B列数
 * @param K A列数/B行数
 * @param alpha 缩放因子
 * @param beta 累加因子
 * @return int 成功返回0，失败返回-1
 */
int gpu_matmul_train(GpuContext* context,
                     const float* A, const float* B, float* C,
                     size_t M, size_t N, size_t K,
                     float alpha, float beta);

/**
 * @brief 在GPU上执行训练内核（完整训练步骤）
 * 
 * 在一个内核调用中完成：前向传播 → 损失计算 → 反向传播 → 权重更新
 * 所有权重、偏置、数据均已在设备内存中。
 * 
 * @param context GPU上下文
 * @param input 输入数据指针
 * @param target 目标数据指针
 * @param output 输出数据指针
 * @param weights 权重张量（一维展开）
 * @param input_size 输入维度
 * @param output_size 输出维度
 * @param config 训练配置
 * @param step 当前训练步数
 * @return int 成功返回0，失败返回-1
 */
int gpu_train_step(GpuContext* context,
                   const float* input, const float* target,
                   float* output, float* weights,
                   size_t input_size, size_t output_size,
                   const GpuTrainConfig* config, int step);

// ==================== GPU高级训练算子接口 ====================

/**
 * @brief GPU激活函数类型
 */
typedef enum {
    GPU_ACTIVATION_RELU = 0,          /**< ReLU: max(0, x) */
    GPU_ACTIVATION_LEAKY_RELU = 1,    /**< LeakyReLU: x>0?x:alpha*x */
    GPU_ACTIVATION_SIGMOID = 2,       /**< Sigmoid: 1/(1+exp(-x)) */
    GPU_ACTIVATION_TANH = 3,          /**< Tanh: (exp(x)-exp(-x))/(exp(x)+exp(-x)) */
    GPU_ACTIVATION_GELU = 4,          /**< GELU: x*0.5*(1+erf(x/sqrt(2))) */
    GPU_ACTIVATION_SOFTPLUS = 5       /**< Softplus: ln(1+exp(x)) */
} GpuActivationType;

/**
 * @brief GPU学习率调度类型
 */
typedef enum {
    GPU_LR_SCHEDULE_STEP = 0,           /**< 阶梯衰减 */
    GPU_LR_SCHEDULE_EXPONENTIAL = 1,    /**< 指数衰减 */
    GPU_LR_SCHEDULE_POLYNOMIAL = 2,     /**< 多项式衰减 */
    GPU_LR_SCHEDULE_COSINE = 3,         /**< 余弦退火 */
    GPU_LR_SCHEDULE_LINEAR = 4          /**< 线性衰减 */
} GpuLRScheduleType;

/**
 * @brief GPU批归一化配置
 */
typedef struct {
    float epsilon;                       /**< 数值稳定性常数 */
    float momentum;                      /**< 运行统计动量 */
    int affine;                          /**< 是否使用可学习缩放/偏移 */
    int track_running_stats;             /**< 是否跟踪运行统计 */
} GpuBatchNormConfig;

/**
 * @brief GPU学习率调度配置
 */
typedef struct {
    GpuLRScheduleType schedule_type;     /**< 调度类型 */
    float initial_lr;                    /**< 初始学习率 */
    float decay_rate;                    /**< 衰减率 */
    int decay_steps;                     /**< 衰减步数间隔 */
    int total_steps;                     /**< 总步数 */
    float min_lr;                        /**< 最小学习率 */
    float gamma;                         /**< 指数衰减gamma因子 */
    int warmup_steps;                    /**< 预热步数 */
    float warmup_lr;                     /**< 预热初始学习率 */
    int step_size;                       /**< 步长（用于Step/多项式衰减） */
    float power;                         /**< 多项式幂次 */
} GpuLRConfig;

/**
 * @brief 创建默认GPU批归一化配置
 *
 * @return GpuBatchNormConfig 默认配置
 */
GpuBatchNormConfig gpu_batch_norm_config_default(void);

/**
 * @brief 创建默认GPU学习率调度配置
 *
 * @return GpuLRConfig 默认配置
 */
GpuLRConfig gpu_lr_config_default(void);

/**
 * @brief 在GPU上执行RMSProp权重更新
 *
 * 对权重张量执行RMSProp优化器更新:
 * s = decay * s + (1-decay) * g^2
 * w = w - lr * g / (sqrt(s) + eps) - lr * wd * w
 *
 * @param context GPU上下文
 * @param weights GPU权重指针
 * @param gradients GPU梯度指针
 * @param square_avg GPU平方梯度均值缓冲区指针
 * @param num_params 参数数量
 * @param learning_rate 学习率
 * @param decay RMSProp衰减系数
 * @param epsilon 数值稳定性常数
 * @param weight_decay 权重衰减系数
 * @return int 成功返回0，失败返回-1
 */
int gpu_rmsprop_update(GpuContext* context, float* weights, const float* gradients,
                       float* square_avg, size_t num_params,
                       float learning_rate, float decay, float epsilon,
                       float weight_decay);

/**
 * @brief 在GPU上计算交叉熵损失梯度（用于分类训练）
 *
 * 计算softmax交叉熵损失和梯度:
 * softmax: p_i = exp(x_i) / sum(exp(x_j))
 * loss = -sum(t_i * log(p_i))
 * grad = (p - t) / n
 *
 * @param context GPU上下文
 * @param logits GPU原始logits指针
 * @param targets GPU目标标签指针（one-hot或整数标签）
 * @param loss GPU损失值输出指针（标量）
 * @param gradients GPU梯度输出指针
 * @param num_elements 总元素数（batch_size * num_classes）
 * @param num_classes 类别数
 * @param is_integer_label 是否为整数标签（非one-hot）
 * @return int 成功返回0，失败返回-1
 */
int gpu_cross_entropy_loss_gradient(GpuContext* context,
                                     const float* logits, const float* targets,
                                     float* loss, float* gradients,
                                     size_t num_elements, int num_classes,
                                     int is_integer_label);

/**
 * @brief 在GPU上执行批归一化前向传播
 *
 * 训练模式: y = gamma * (x - mean) / sqrt(var + eps) + beta
 * 推理模式: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
 *
 * @param context GPU上下文
 * @param input GPU输入数据指针
 * @param output GPU输出数据指针
 * @param gamma GPU缩放参数指针
 * @param beta GPU偏移参数指针
 * @param running_mean GPU运行均值指针（推理使用，训练时更新）
 * @param running_var GPU运行方差指针（推理使用，训练时更新）
 * @param batch_mean GPU批均值输出指针（训练使用）
 * @param batch_var GPU批方差输出指针（训练使用）
 * @param num_elements 总元素数（batch_size * num_features）
 * @param num_features 特征数
 * @param config 批归一化配置
 * @param is_training 是否为训练模式
 * @return int 成功返回0，失败返回-1
 */
int gpu_batch_norm_forward(GpuContext* context,
                           const float* input, float* output,
                           const float* gamma, const float* beta,
                           float* running_mean, float* running_var,
                           float* batch_mean, float* batch_var,
                           size_t num_elements, size_t num_features,
                           const GpuBatchNormConfig* config, int is_training);

/**
 * @brief 在GPU上执行批归一化反向传播
 *
 * 计算批归一化层的梯度:
 * grad_gamma = sum(grad_output * x_hat, axis=0)
 * grad_beta = sum(grad_output, axis=0)
 * grad_input = gamma * (grad_output - grad_beta/N - x_hat*grad_gamma/N) / sqrt(var+eps)
 *
 * @param context GPU上下文
 * @param input GPU输入数据指针
 * @param grad_output GPU输出梯度指针
 * @param grad_input GPU输入梯度输出指针
 * @param grad_gamma GPU缩放参数梯度输出指针
 * @param grad_beta GPU偏移参数梯度输出指针
 * @param mean GPU批均值指针
 * @param var GPU批方差指针
 * @param gamma GPU缩放参数指针（用于反向传播）
 * @param num_elements 总元素数（batch_size * num_features）
 * @param num_features 特征数
 * @param config 批归一化配置
 * @return int 成功返回0，失败返回-1
 */
int gpu_batch_norm_backward(GpuContext* context,
                            const float* input, const float* grad_output,
                            float* grad_input, float* grad_gamma, float* grad_beta,
                            const float* mean, const float* var,
                            const float* gamma,
                            size_t num_elements, size_t num_features,
                            const GpuBatchNormConfig* config);

/**
 * @brief 在GPU上执行激活函数前向传播
 *
 * 对输入数据逐元素应用指定的激活函数。
 * 支持类型: ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Softplus
 *
 * @param context GPU上下文
 * @param input GPU输入数据指针
 * @param output GPU输出数据指针（可为同一指针就地操作）
 * @param num_elements 元素数量
 * @param act_type 激活函数类型
 * @param alpha LeakyReLU的负斜率系数
 * @return int 成功返回0，失败返回-1
 */
int gpu_activation_forward(GpuContext* context,
                           const float* input, float* output,
                           size_t num_elements, GpuActivationType act_type, float alpha);

/**
 * @brief 在GPU上执行激活函数反向传播
 *
 * 计算激活函数的梯度:
 * ReLU: grad_input = grad_output * (x > 0)
 * LeakyReLU: grad_input = grad_output * (x > 0 ? 1 : alpha)
 * Sigmoid: grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
 * Tanh: grad_input = grad_output * (1 - tanh(x)^2)
 * GELU: grad_input = grad_output * (0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi))
 *
 * @param context GPU上下文
 * @param input GPU原始输入数据指针（激活前）
 * @param grad_output GPU输出梯度指针（激活后）
 * @param grad_input GPU输入梯度输出指针
 * @param num_elements 元素数量
 * @param act_type 激活函数类型
 * @param alpha LeakyReLU的负斜率系数
 * @return int 成功返回0，失败返回-1
 */
int gpu_activation_backward(GpuContext* context,
                            const float* input, const float* grad_output,
                            float* grad_input, size_t num_elements,
                            GpuActivationType act_type, float alpha);

/**
 * @brief 在GPU上执行Dropout前向传播
 *
 * 训练模式: 以概率dropout_rate随机置零输入元素，并缩放保留元素: output = mask * input / (1-rate)
 * 推理模式: output = input（恒等映射）
 *
 * @param context GPU上下文
 * @param input GPU输入数据指针
 * @param output GPU输出数据指针
 * @param mask GPU掩码输出指针（训练时记录置零模式）
 * @param num_elements 元素数量
 * @param dropout_rate Dropout概率（0.0~1.0）
 * @param is_training 是否为训练模式
 * @return int 成功返回0，失败返回-1
 */
int gpu_dropout_forward(GpuContext* context,
                        const float* input, float* output,
                        float* mask, size_t num_elements,
                        float dropout_rate, int is_training);

/**
 * @brief 在GPU上执行Dropout反向传播
 *
 * 在训练模式的反向传播中，应用与forward相同的mask:
 * grad_input = grad_output * mask / (1 - rate)
 *
 * @param context GPU上下文
 * @param grad_output GPU输出梯度指针
 * @param grad_input GPU输入梯度输出指针
 * @param mask GPU掩码指针（由forward生成）
 * @param num_elements 元素数量
 * @param dropout_rate Dropout概率
 * @return int 成功返回0，失败返回-1
 */
int gpu_dropout_backward(GpuContext* context,
                         const float* grad_output, float* grad_input,
                         const float* mask, size_t num_elements,
                         float dropout_rate);

/**
 * @brief 执行GPU学习率调度步进
 *
 * 根据当前步数和调度配置计算当前学习率:
 * Step: lr = initial_lr * decay_rate ^ floor(step / decay_steps)
 * Exponential: lr = initial_lr * exp(-decay_rate * step)
 * Polynomial: lr = (initial_lr - min_lr) * (1 - step/total)^power + min_lr
 * Cosine: lr = min_lr + 0.5*(initial_lr-min_lr)*(1+cos(pi*step/total))
 * Linear: lr = initial_lr - (initial_lr-min_lr) * min(step, total) / total
 *
 * @param current_step 当前训练步数（从0开始）
 * @param config 学习率调度配置
 * @return float 当前步数的学习率
 */
float gpu_lr_scheduler_step(int current_step, const GpuLRConfig* config);

/**
 * @brief 在GPU上执行全连接层前向传播
 *
 * 完成一个全连接层的完整前向计算:
 * 1. 矩阵乘法: output = input * weights^T + bias
 * 2. 激活函数: output = activation(output)
 *
 * @param context GPU上下文
 * @param input GPU输入数据指针 [batch_size, input_size]
 * @param weights GPU权重指针 [output_size, input_size]
 * @param bias GPU偏置指针 [output_size]（可为NULL）
 * @param output GPU输出数据指针 [batch_size, output_size]
 * @param batch_size 批大小
 * @param input_size 输入维度
 * @param output_size 输出维度
 * @param act_type 激活函数类型（使用GPU_ACTIVATION_RELU为无激活）
 * @param alpha 激活函数参数（LeakyReLU负斜率）
 * @return int 成功返回0，失败返回-1
 */
int gpu_forward_dense(GpuContext* context,
                      const float* input, float* output,
                      const float* weights, const float* bias,
                      size_t batch_size, size_t input_size, size_t output_size,
                      GpuActivationType act_type, float alpha);

// ==================== NPU推理加速接口 ====================

/**
 * @brief NPU模型句柄（不透明类型）
 */
typedef struct NpuModel NpuModel;

/**
 * @brief NPU推理配置
 */
typedef struct {
    int device_index;                    /**< NPU设备索引 */
    int input_count;                     /**< 输入张量数量 */
    int output_count;                    /**< 输出张量数量 */
    int input_dim;                       /**< 输入特征维度（单张量时有效） */
    int output_dim;                      /**< 输出特征维度（单张量时有效） */
    size_t* input_sizes;                 /**< 每个输入张量的大小（字节） */
    size_t* output_sizes;                /**< 每个输出张量的大小（字节） */
    int enable_fp16;                     /**< 启用FP16推理（默认0=FP32） */
    int enable_profiling;                /**< 启用性能分析（默认0） */
    int batch_size;                      /**< 推理批次大小（默认1） */
    int timeout_ms;                      /**< 推理超时（毫秒，0=无限等待） */
    const char* model_input_name;        /**< 模型输入节点名称 */
    const char* model_output_name;       /**< 模型输出节点名称 */
} NpuInferenceConfig;

/**
 * @brief NPU模型信息
 */
typedef struct {
    char model_name[256];                /**< 模型名称 */
    char model_path[512];                /**< 模型文件路径 */
    int is_loaded;                       /**< 是否已加载 */
    size_t model_size_bytes;             /**< 模型文件大小（字节） */
    int input_count;                     /**< 输入张量数量 */
    int output_count;                    /**< 输出张量数量 */
    size_t* input_dims[4];               /**< 每个输入张量的维度（最大4维） */
    size_t* output_dims[4];              /**< 每个输出张量的维度（最大4维） */
    float estimated_inference_time_ms;   /**< 估计推理时间（毫秒） */
} NpuModelInfo;

/**
 * @brief 初始化NPU子系统
 *
 * 在指定GPU上下文上初始化NPU推理子系统。
 * 需要GPU后端类型为GPU_BACKEND_ASCEND/CAMBRICON/TPU之一。
 *
 * @param context GPU上下文（必须为NPU后端类型）
 * @return int 成功返回0，失败返回-1
 */
int gpu_npu_init(GpuContext* context);

/**
 * @brief 清理NPU子系统
 *
 * 释放所有NPU资源，包括已加载的模型。
 *
 * @param context GPU上下文
 */
void gpu_npu_cleanup(GpuContext* context);

/**
 * @brief 加载NPU模型
 *
 * 将模型文件加载到NPU设备内存中。
 * 支持.om（昇腾）、.cambricon（寒武纪）、.tpu（谷歌TPU）等格式。
 *
 * @param context GPU上下文
 * @param model_path 模型文件路径
 * @param config 推理配置（可为NULL使用默认配置）
 * @return NpuModel* 成功返回模型句柄，失败返回NULL
 */
NpuModel* gpu_npu_load_model(GpuContext* context, const char* model_path,
                             const NpuInferenceConfig* config);

/**
 * @brief 卸载NPU模型
 *
 * 从NPU设备内存中卸载指定模型。
 *
 * @param model 模型句柄
 */
void gpu_npu_unload_model(NpuModel* model);

/**
 * @brief 执行NPU推理
 *
 * 在NPU设备上对输入数据执行模型推理。
 * 输入和输出数据均为主机内存指针，函数内部处理设备内存传输。
 *
 * @param model 模型句柄
 * @param inputs 输入数据数组指针（每个元素为float*）
 * @param outputs 输出数据数组指针（每个元素为float*）
 * @param batch_size 实际推理批次大小（<=配置中的batch_size）
 * @return int 成功返回0，失败返回-1
 */
int gpu_npu_infer(NpuModel* model, const float** inputs, float** outputs,
                  int batch_size);

/**
 * @brief 异步执行NPU推理
 *
 * 在NPU设备上异步执行模型推理，不阻塞调用线程。
 * 通过gpu_npu_infer_wait等待推理完成。
 *
 * @param model 模型句柄
 * @param inputs 输入数据数组指针（必须保持有效直到推理完成）
 * @param outputs 输出数据数组指针（必须保持有效直到推理完成）
 * @param batch_size 实际推理批次大小
 * @return int 成功返回0，失败返回-1
 */
int gpu_npu_infer_async(NpuModel* model, const float** inputs, float** outputs,
                        int batch_size);

/**
 * @brief 等待异步NPU推理完成
 *
 * 阻塞等待异步推理操作完成。
 *
 * @param model 模型句柄
 * @param timeout_ms 超时时间（毫秒，0=无限等待）
 * @return int 成功返回0，超时返回1，失败返回-1
 */
int gpu_npu_infer_wait(NpuModel* model, int timeout_ms);

/**
 * @brief 获取NPU模型信息
 *
 * 查询已加载模型的信息。
 *
 * @param model 模型句柄
 * @param info [out] 模型信息输出缓冲区
 * @return int 成功返回0，失败返回-1
 */
int gpu_npu_get_model_info(NpuModel* model, NpuModelInfo* info);

/**
 * @brief 获取NPU后端设备数量
 *
 * 查询系统中可用的NPU设备数量。
 *
 * @param context GPU上下文
 * @return int NPU设备数量，失败返回-1
 */
int gpu_npu_get_device_count(GpuContext* context);

/**
 * @brief 获取NPU后端名称
 *
 * 返回当前NPU后端的名称字符串。
 *
 * @param context GPU上下文
 * @return const char* 后端名称（如"Ascend310"、"MLU370"等），失败返回NULL
 */
const char* gpu_npu_get_backend_name(GpuContext* context);

#ifdef __cplusplus
}
#endif

#endif // SELFLNN_GPU_H