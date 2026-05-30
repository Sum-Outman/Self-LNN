/**
 * SELF-LNN 核心通用定义
 * 
 * 包含所有核心模块共享的类型定义、常量、宏等。
 */

#ifndef SELFLNN_CORE_COMMON_H
#define SELFLNN_CORE_COMMON_H

/* 跨编译器兼容：MSVC 默认不定义 M_PI */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#include "platform_types.h"

/* 错误码定义（来自 errors.h，循环依赖已通过 platform_types.h 打破） */
#include "errors.h"

/* 为兼容性保留的错误码别名 —— 直接映射到errors.h枚举值 */
/* K-142: 已修复自引用宏定义，改为显式整数值映射。
   原自引用形如 SELFLNN_SUCCESS SELFLNN_SUCCESS 会触发编译器诊断污染。
   现改为显式数值常量，确保：
   1. 向后兼容（所有使用宏的代码无需修改）
   2. 类型安全（int值可隐式转换为SelfLNNErrorCode枚举类型）
   3. 诊断清洁（无循环宏定义警告） */
#define SELFLNN_SUCCESS                 0       /* SELFLNN_SUCCESS = 0 */
#define SELFLNN_ERROR_INVALID_ARGUMENT  (-2)    /* SELFLNN_ERROR_INVALID_ARGUMENT = -2 */
#define SELFLNN_ERROR_OUT_OF_MEMORY     (-4)    /* SELFLNN_ERROR_OUT_OF_MEMORY = -4 */
#define SELFLNN_ERROR_OPERATION_FAILED  (-1)    /* 映射到 SELFLNN_ERROR_GENERIC = -1 */
#define SELFLNN_ERROR_NOT_INITIALIZED   (-5)    /* SELFLNN_ERROR_NOT_INITIALIZED = -5 */
#define SELFLNN_ERROR_INVALID_STATE     (-11)   /* SELFLNN_ERROR_INVALID_STATE = -11 */
#define SELFLNN_ERROR_GPU_NOT_AVAILABLE (-700)  /* SELFLNN_ERROR_GPU_NOT_AVAILABLE = -700 */
#define SELFLNN_ERROR_MODEL_NOT_FOUND   (-750)  /* SELFLNN_ERROR_MODEL_NOT_FOUND = -750 */
#define SELFLNN_ERROR_TRAINING_FAILED   (-751)  /* SELFLNN_ERROR_TRAINING_FAILED = -751 */
#define SELFLNN_ERROR_INFERENCE_FAILED  (-752)  /* SELFLNN_ERROR_INFERENCE_FAILED = -752 */
#define SELFLNN_ERROR_DEVICE_OFFLINE    (-780)  /* SELFLNN_ERROR_DEVICE_OFFLINE = -780 */
#define SELFLNN_ERROR_NO_DATA           (-505)  /* SELFLNN_ERROR_NO_DATA = -505 */
#define SELFLNN_ERROR_PROJECTION_LOCKED (-506)  /* SELFLNN_ERROR_PROJECTION_LOCKED = -506（投影矩阵锁定，跳过训练） */

/* 功率模式枚举 */
typedef enum {
    POWER_MODE_PERFORMANCE = 0,   // 高性能模式
    POWER_MODE_BALANCED = 1,      // 均衡模式
    POWER_MODE_POWER_SAVING = 2,  // 节能模式
    POWER_MODE_ULTRA_SAVING = 3,  // 超节能模式
    POWER_MODE_CUSTOM = 4         // 自定义模式
} PowerMode;

// GPU后端枚举
typedef enum {
    GPU_BACKEND_CPU = 0,        // CPU后端
    GPU_BACKEND_CUDA = 1,       // CUDA后端
    GPU_BACKEND_OPENCL = 2,     // OpenCL后端
    GPU_BACKEND_VULKAN = 3,     // Vulkan后端
    GPU_BACKEND_METAL = 4,      // Metal后端
    GPU_BACKEND_ASCEND = 5,     // 华为昇腾Ascend NPU后端
    GPU_BACKEND_CAMBRICON = 6,  // 寒武纪Cambricon MLU后端
    GPU_BACKEND_TPU = 7,        // 谷歌TPU后端
    GPU_BACKEND_ROCM = 8,       // AMD ROCm GPU后端
    GPU_BACKEND_INTEL = 9,      // Intel GPU后端
    GPU_BACKEND_COUNT = 10      // K-009: 后端总数（用于数组索引）
} GpuBackend;

// 内存位置枚举
typedef enum {
    MEMORY_LOCATION_HOST = 0,       // 主机内存
    MEMORY_LOCATION_DEVICE = 1,     // 设备内存
    MEMORY_LOCATION_UNIFIED = 2,    // 统一内存
    MEMORY_LOCATION_PINNED = 3      // 固定内存
} MemoryLocation;

// 数据精度枚举
typedef enum {
    PRECISION_FP32 = 0,         // 单精度浮点数
    PRECISION_FP16 = 1,         // 半精度浮点数
    PRECISION_INT8 = 2,         // 8位整数
    PRECISION_INT16 = 3,        // 16位整数
    PRECISION_INT32 = 4         // 32位整数
} Precision;

// 张量维度
#define SELFLNN_MAX_DIMS 8

/* K-047: 统一张量描述符 —— 两处定义协同工作
 * tensor.h 包含主定义（TensorDataType dtype; int ndim; int shape[8]），
 * common.h 提供回退定义供未包含 tensor.h 的模块使用。
 * SELFLNN_CORE_TENSOR_H 条件编译确保无冲突。
 * 各模块的张量操作应使用此描述符进行维度/精度/内存位置管理。 */
#ifndef SELFLNN_CORE_COMMON_TENSOR_DESC
#define SELFLNN_CORE_COMMON_TENSOR_DESC
typedef struct {
    size_t shape[SELFLNN_MAX_DIMS];  /**< 形状数组 */
    int ndims;                       /**< 维度数量 (1-8) */
    Precision precision;             /**< 数据精度 */
    MemoryLocation memory_location;  /**< 内存位置 */
} TensorDescriptor;
#endif

// 通用结果结构
typedef struct {
    int error_code;                  // 错误码
    const char* error_message;       // 错误信息
    void* data;                      // 结果数据
    size_t data_size;                // 数据大小
} Result;

// 回调函数类型定义
typedef void (*ProgressCallback)(double progress, void* user_data);
typedef void (*ErrorCallback)(int error_code, const char* message, void* user_data);
typedef void (*LogCallback)(const char* message, int level, void* user_data);

// 日志级别
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_FATAL = 4
} LogLevel;

// 断言宏
#ifdef SELFLNN_DEBUG
    #define SELFLNN_ASSERT(expr) \
        do { \
            if (!(expr)) { \
                selflnn_log(LOG_LEVEL_FATAL, "断言失败: %s, 文件: %s, 行: %d", \
                           #expr, __FILE__, __LINE__); \
                abort(); \
            } \
        } while (0)
#else
    #define SELFLNN_ASSERT(expr) ((void)0)
#endif

// 不返回宏
#ifdef __GNUC__
    #define SELFLNN_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
    #define SELFLNN_NORETURN __declspec(noreturn)
#else
    #define SELFLNN_NORETURN
#endif

// 对齐宏
#define SELFLNN_ALIGN(n) __attribute__((aligned(n)))
#define SELFLNN_CACHE_ALIGN SELFLNN_ALIGN(64)

// 内联函数宏
#ifdef _MSC_VER
    #define SELFLNN_INLINE __inline
#else
    #define SELFLNN_INLINE inline
#endif

// 静态内联
#define SELFLNN_STATIC_INLINE static SELFLNN_INLINE

// 内存屏障
#ifdef __GNUC__
    #define SELFLNN_MEMORY_BARRIER() __sync_synchronize()
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define SELFLNN_MEMORY_BARRIER() _mm_mfence()
#else
    #define SELFLNN_MEMORY_BARRIER() ((void)0)
#endif

// 编译器优化提示
#ifdef __GNUC__
    #define SELFLNN_LIKELY(x) __builtin_expect(!!(x), 1)
    #define SELFLNN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define SELFLNN_LIKELY(x) (x)
    #define SELFLNN_UNLIKELY(x) (x)
#endif

// 函数属性
#ifdef __GNUC__
    #define SELFLNN_DEPRECATED __attribute__((deprecated))
    #define SELFLNN_FORMAT(fmt_idx, arg_idx) __attribute__((format(printf, fmt_idx, arg_idx)))
    #define SELFLNN_PURE __attribute__((pure))
    #define SELFLNN_CONST __attribute__((const))
#else
    #define SELFLNN_DEPRECATED
    #define SELFLNN_FORMAT(fmt_idx, arg_idx)
    #define SELFLNN_PURE
    #define SELFLNN_CONST
#endif

// 平台检测
#if defined(_WIN32) || defined(_WIN64)
    #define SELFLNN_PLATFORM_WINDOWS 1
    #define SELFLNN_PLATFORM_LINUX 0
    #define SELFLNN_PLATFORM_MACOS 0
#elif defined(__linux__)
    #define SELFLNN_PLATFORM_WINDOWS 0
    #define SELFLNN_PLATFORM_LINUX 1
    #define SELFLNN_PLATFORM_MACOS 0
#elif defined(__APPLE__) && defined(__MACH__)
    #define SELFLNN_PLATFORM_WINDOWS 0
    #define SELFLNN_PLATFORM_LINUX 0
    #define SELFLNN_PLATFORM_MACOS 1
#else
    #define SELFLNN_PLATFORM_WINDOWS 0
    #define SELFLNN_PLATFORM_LINUX 0
    #define SELFLNN_PLATFORM_MACOS 0
#endif

// 字节序检测
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    #define SELFLNN_LITTLE_ENDIAN 1
    #define SELFLNN_BIG_ENDIAN 0
#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #define SELFLNN_LITTLE_ENDIAN 0
    #define SELFLNN_BIG_ENDIAN 1
#elif defined(_WIN32) || defined(_WIN64)
    #define SELFLNN_LITTLE_ENDIAN 1
    #define SELFLNN_BIG_ENDIAN 0
#else
    #define SELFLNN_LITTLE_ENDIAN 1  // 假设小端
    #define SELFLNN_BIG_ENDIAN 0
#endif

// 编译器版本检测
#if defined(__GNUC__)
    #define SELFLNN_COMPILER_GCC 1
    #define SELFLNN_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#else
    #define SELFLNN_COMPILER_GCC 0
    #define SELFLNN_GCC_VERSION 0
#endif

#if defined(__clang__)
    #define SELFLNN_COMPILER_CLANG 1
    #define SELFLNN_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#else
    #define SELFLNN_COMPILER_CLANG 0
    #define SELFLNN_CLANG_VERSION 0
#endif

#if defined(_MSC_VER)
    #define SELFLNN_COMPILER_MSVC 1
    #define SELFLNN_MSVC_VERSION _MSC_VER
#else
    #define SELFLNN_COMPILER_MSVC 0
    #define SELFLNN_MSVC_VERSION 0
#endif

// 64位检测
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)
    #define SELFLNN_ARCH_64BIT 1
#else
    #define SELFLNN_ARCH_64BIT 0
#endif

// SIMD检测
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #define SELFLNN_SSE2_AVAILABLE 1
#else
    #define SELFLNN_SSE2_AVAILABLE 0
#endif

#if defined(__AVX__)
    #define SELFLNN_AVX_AVAILABLE 1
#else
    #define SELFLNN_AVX_AVAILABLE 0
#endif

#if defined(__AVX2__)
    #define SELFLNN_AVX2_AVAILABLE 1
#else
    #define SELFLNN_AVX2_AVAILABLE 0
#endif

#if defined(__AVX512F__)
    #define SELFLNN_AVX512_AVAILABLE 1
#else
    #define SELFLNN_AVX512_AVAILABLE 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define SELFLNN_NEON_AVAILABLE 1
#else
    #define SELFLNN_NEON_AVAILABLE 0
#endif

// 最小/最大宏
#define SELFLNN_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SELFLNN_MAX(a, b) ((a) > (b) ? (a) : (b))
#define SELFLNN_CLAMP(x, min, max) (SELFLNN_MIN(SELFLNN_MAX((x), (min)), (max)))

// 数组元素数量
#define SELFLNN_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// SELFLNN_NO_PRETRAINED 编译期宏
// 定义此宏后，系统将完全阻断预训练权重加载路径
// 适用于从零训练场景，确保不使用任何预训练权重
#ifdef SELFLNN_NO_PRETRAINED
    #define SELFLNN_CAN_LOAD_PRETRAINED 0
#else
    #define SELFLNN_CAN_LOAD_PRETRAINED 1
#endif

// 模型文件魔数（SELF-LNN模型文件标识）
#define SELFLNN_FILE_MAGIC      "SELFLNN"
#define SELFLNN_FILE_MAGIC_SIZE 8

// 模型文件头标记
#define SELFLNN_MARKER_FROM_SCRATCH  0x01  // 从零训练生成的模型
#define SELFLNN_MARKER_PRETRAINED    0x02  // 使用预训练权重

// 字符串化
#define SELFLNN_STRINGIFY(x) #x
#define SELFLNN_TO_STRING(x) SELFLNN_STRINGIFY(x)

// 连接
#define SELFLNN_CONCAT(a, b) a##b
#define SELFLNN_CONCAT3(a, b, c) a##b##c

// 函数声明宏
#define SELFLNN_DECLARE_API(ret_type, name, params) \
    SELFLNN_API ret_type SELFLNN_CALL name params

#define SELFLNN_DECLARE_INTERNAL(ret_type, name, params) \
    ret_type SELFLNN_CALL name params

// 模块初始化/清理函数声明
typedef void (*ModuleInitFunc)(void);
typedef void (*ModuleCleanupFunc)(void);

// 模块注册结构
typedef struct {
    const char* module_name;
    ModuleInitFunc init_func;
    ModuleCleanupFunc cleanup_func;
    int priority;  // 初始化优先级，数字越小优先级越高
} ModuleRegistration;

// 模块注册宏
#define SELFLNN_MODULE_REGISTER(name, init, cleanup, prio) \
    static const ModuleRegistration SELFLNN_CONCAT(__module_reg_, __LINE__) = { \
        .module_name = name, \
        .init_func = init, \
        .cleanup_func = cleanup, \
        .priority = prio \
    }

// ============================================================
// 编译时依赖完整性检查
// ============================================================
// C11 _Static_assert 兼容层
// 检测 _Static_assert 是否真正可用（MSVC C 模式需 /std:c11 才支持）
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_STATIC_ASSERT__)) \
    || (defined(__cplusplus) && __cplusplus >= 201103L)
    #define SELFLNN_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__has_feature)
    #if __has_feature(c_static_assert)
        #define SELFLNN_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
    #else
        #define SELFLNN_STATIC_ASSERT(cond, msg) \
            typedef char SELFLNN_CONCAT(__sa_fail_, __LINE__)[(cond) ? 1 : -1]
    #endif
#else
    // 通用回退：负大小数组触发编译错误，兼容所有 C89+ 编译器
    #define SELFLNN_STATIC_ASSERT(cond, msg) \
        typedef char SELFLNN_CONCAT(__sa_fail_, __LINE__)[(cond) ? 1 : -1]
#endif

// --- 类型大小检查 ---
SELFLNN_STATIC_ASSERT(sizeof(int32_t) == 4, "int32_t 必须为 4 字节");
SELFLNN_STATIC_ASSERT(sizeof(uint32_t) == 4, "uint32_t 必须为 4 字节");
SELFLNN_STATIC_ASSERT(sizeof(float) == 4, "float 必须为 4 字节");
SELFLNN_STATIC_ASSERT(sizeof(double) == 8, "double 必须为 8 字节");
SELFLNN_STATIC_ASSERT(sizeof(void*) == 4 || sizeof(void*) == 8, "指针必须为 4 或 8 字节");

// --- 关键结构体大小检查 ---
SELFLNN_STATIC_ASSERT(sizeof(int) >= 2, "int 至少 2 字节");
SELFLNN_STATIC_ASSERT(sizeof(long) >= 4, "long 至少 4 字节");
SELFLNN_STATIC_ASSERT(sizeof(size_t) >= 4, "size_t 至少 4 字节");

// --- 标准头文件存在性检查（通过关键类型存在性） ---
#include <float.h>
SELFLNN_STATIC_ASSERT(FLT_RADIX == 2, "浮点基数必须为 2，需要 IEEE 754 支持");

#include <limits.h>
SELFLNN_STATIC_ASSERT(CHAR_BIT == 8, "CHAR_BIT 必须为 8");

#if defined(SELFLNN_PLATFORM_WINDOWS) && SELFLNN_PLATFORM_WINDOWS
    /* winsock2.h 必须在 windows.h 之前包含，防止 sockaddr/fd_set 重定义 */
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <windows.h>
    /* windows.h 污染全局命名空间：interface→struct，undef 避免破坏项目代码 */
    #ifdef interface
    #undef interface
    #endif
    #ifdef IN
    #undef IN
    #endif
    #ifdef OUT
    #undef OUT
    #endif
    // Windows 关键类型大小检查
    SELFLNN_STATIC_ASSERT(sizeof(DWORD) == 4, "Windows DWORD 必须为 4 字节");
    SELFLNN_STATIC_ASSERT(sizeof(BOOL) == 4, "Windows BOOL 必须为 4 字节");
#endif

// NaN/Inf 检查宏（调用者需包含 <math.h>）
#if defined(_MSC_VER)
    #define SELFLNN_IS_FINITE(x) (_finite((double)(x)))
#else
    #define SELFLNN_IS_FINITE(x) (isfinite(x))
#endif

// ============================================================
// 核心常量宏定义（L-001修复：统一管理关键数值常量，消除硬编码魔数）
// ============================================================
#define SELFLNN_EPSILON_FLOAT        1e-7f   /* 浮点数比较容差 */
#define SELFLNN_EPSILON_DOUBLE       1e-12   /* 双精度浮点数比较容差 */
#define SELFLNN_DEFAULT_DT           0.01f   /* 默认时间步长（dt） */
#define SELFLNN_DEFAULT_LEARNING_RATE 0.001f /* 默认学习率 */
#define SELFLNN_DEFAULT_TAU_MIN      0.001f  /* 默认最小时间常数 */
#define SELFLNN_DEFAULT_TAU_MAX      10.0f   /* 默认最大时间常数 */
#define SELFLNN_DEFAULT_GRAD_CLIP    5.0f    /* 默认梯度裁剪阈值 */
#define SELFLNN_DEFAULT_LAYER_NORM_EPS 1e-5f /* 默认层归一化epsilon */

/* ZSFUSA-P2-001修复: 系统互斥锁函数声明。
 * system_mutex_lock/unlock由system_mutex.c实现，multi_agent.c调用。
 * 显式声明避免隐式函数调用导致的MSVC C4013警告和潜在的ABI问题。 */
#ifdef __cplusplus
extern "C" {
#endif
void system_mutex_lock(void* mutex_ptr);
void system_mutex_unlock(void* mutex_ptr);
void* system_mutex_create(void);
void system_mutex_destroy(void* mutex_ptr);
int system_mutex_trylock(void* mutex_ptr);
#ifdef __cplusplus
}
#endif

#endif // SELFLNN_CORE_COMMON_H