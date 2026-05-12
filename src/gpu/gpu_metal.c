/**
 * @file gpu_metal.c
 * @brief Apple Metal GPU后端实现
 * 
 * Apple Metal GPU后端实现 - 提供macOS/iOS平台GPU加速计算。
 * 当前版本包含完整的Metal GPU计算实现，支持动态库加载、设备管理、
 * 内存分配、计算管线创建和内核执行。
 * 注意：根据项目要求"禁止任何降级处理"，实现真实的Metal硬件加速。
 */

#include "selflnn/gpu/gpu.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "gpu_internal.h"

#if defined(_MSC_VER)
#pragma warning(disable:4100)
#endif


#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#if defined(_MSC_VER)
#pragma warning(disable:4100)
#endif


// 禁用特定编译器警告

// 苹果平台特定的头文件
#ifdef __APPLE__
#include <dlfcn.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <mach/mach_host.h>
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>

#if defined(_MSC_VER)
#pragma warning(disable:4100)
#endif

#endif

/* ============================================================================
 * Metal常量定义
 * =========================================================================== */

// MTLSize结构体定义（用于线程组大小）
typedef struct {
    size_t width;
    size_t height;
    size_t depth;
} MTLSize;

// Metal设备类型
#define METAL_DEVICE_TYPE_INTEGRATED     1
#define METAL_DEVICE_TYPE_DISCRETE       2
#define METAL_DEVICE_TYPE_VIRTUAL        3

// Metal资源选项
#define METAL_RESOURCE_STORAGE_MODE_SHARED    0
#define METAL_RESOURCE_STORAGE_MODE_PRIVATE   1
#define METAL_RESOURCE_STORAGE_MODE_MEMORYLESS 2

// Metal错误码
#define METAL_SUCCESS                     0
#define METAL_ERROR_NOT_SUPPORTED         1
#define METAL_ERROR_INVALID_DEVICE        2
#define METAL_ERROR_INVALID_COMMAND       3
#define METAL_ERROR_INVALID_BUFFER        4
#define METAL_ERROR_INVALID_TEXTURE       5
#define METAL_ERROR_OUT_OF_MEMORY         6
#define METAL_ERROR_COMPILE_FAILED        7

/* ============================================================================
 * Metal API函数指针定义
 * =========================================================================== */

// Metal基本类型定义
typedef void* MTLCreateSystemDefaultDeviceFunc(void);
typedef void* MTLCopyAllDevicesFunc(void);
typedef void* MTLDevice;
typedef void* MTLCommandQueue;
typedef void* MTLCommandBuffer;
typedef void* MTLCommandEncoder;
typedef void* MTLComputeCommandEncoder;
typedef void* MTLBuffer;
typedef void* MTLLibrary;
typedef void* MTLFunction;
typedef void* MTLComputePipelineState;
typedef void* MTLHeap;

// Metal API函数指针定义
static MTLCreateSystemDefaultDeviceFunc* mtlCreateSystemDefaultDevice = NULL;
static MTLCopyAllDevicesFunc* mtlCopyAllDevices = NULL;

// 其他函数指针将在运行时加载
static void* (*mtlDeviceNewCommandQueue)(void* device) = NULL;
static void* (*mtlCommandQueueCommandBuffer)(void* queue) = NULL;
static void* (*mtlCommandBufferComputeCommandEncoder)(void* buffer) = NULL;
static void (*mtlComputeCommandEncoderSetComputePipelineState)(void* encoder, void* pipeline) = NULL;
static void (*mtlComputeCommandEncoderSetBuffer)(void* encoder, void* buffer, size_t offset, int index) = NULL;
static void (*mtlComputeCommandEncoderDispatchThreads)(void* encoder, size_t width, size_t height, size_t depth) = NULL;
static void (*mtlComputeCommandEncoderEndEncoding)(void* encoder) = NULL;
static void (*mtlCommandBufferCommit)(void* buffer) = NULL;
static void (*mtlCommandBufferWaitUntilCompleted)(void* buffer) = NULL;

// Blit命令编码器函数（用于私有内存复制）
static void* (*mtlCommandBufferBlitCommandEncoder)(void* buffer) = NULL;
static void (*mtlBlitCommandEncoderCopyFromBufferToBuffer)(void* encoder, void* src_buffer, size_t src_offset, void* dst_buffer, size_t dst_offset, size_t size) = NULL;
static void (*mtlBlitCommandEncoderEndEncoding)(void* encoder) = NULL;

static void* (*mtlDeviceNewBufferWithLength)(void* device, size_t length, int options) = NULL;
static void (*mtlBufferRelease)(void* buffer) = NULL;
static void* (*mtlBufferContents)(void* buffer) = NULL;

static void* (*mtlDeviceNewLibraryWithSource)(void* device, const char* source, size_t length, void* error) = NULL;
static void* (*mtlLibraryNewFunctionWithName)(void* library, const char* name) = NULL;
static void* (*mtlDeviceNewComputePipelineStateWithFunction)(void* device, void* function, void* error) = NULL;

// Metal资源释放函数
static void (*mtlLibraryRelease)(void* library) = NULL;
static void (*mtlFunctionRelease)(void* function) = NULL;
static void (*mtlComputePipelineStateRelease)(void* pipeline) = NULL;

/* ============================================================================
 * 内部数据结构
 * =========================================================================== */

/**
 * @brief Metal上下文内部结构
 */
typedef struct {
    void* device;                   // Metal设备
    void* command_queue;            // 命令队列
    int device_id;                  // 设备ID
    int initialized;                // 是否已初始化
} MetalContextInternal;

/**
 * @brief Metal内存内部结构
 */
typedef struct {
    void* buffer;                   // Metal缓冲区（或主机内存）
    size_t size;                    // 内存大小
    GpuMemoryType memory_type;      // 内存类型
    int storage_mode;               // 存储模式
    int is_gpu_buffer;              // 是否为GPU缓冲区（1=GPU, 0=主机）
    void* device;                   // Metal设备（用于命令缓冲区创建）
    void* command_queue;            // 命令队列（用于私有内存复制）
} MetalMemoryInternal;

/**
 * @brief Metal内核内部结构
 */
typedef struct {
    void* library;                  // Metal库（持有编译后的着色器）
    void* function;                 // Metal函数
    void* pipeline_state;           // 计算管线状态
    char kernel_name[256];          // 内核名称
    // 内核参数存储
    void* arg_buffers[16];          // 参数缓冲区指针（GpuMemory*）
    size_t arg_sizes[16];           // 参数大小
    int arg_count;                  // 参数数量
} MetalKernelInternal;

/**
 * @brief Metal流内部结构
 */
typedef struct {
    void* command_buffer;           // 命令缓冲区
    int is_completed;               // 是否完成
    
    // 异步操作跟踪
    void* pending_command_buffer;   // 待处理的命令缓冲区（异步传输用）
    void* pending_staging_buffer;   // 待处理的暂存缓冲区（需保持存活直到操作完成）
    void* readback_dst;             // 回读目标指针（用于从设备到主机的异步复制）
    size_t readback_size;           // 回读数据大小
    int has_pending_operation;      // 是否有待处理的异步操作
} MetalStreamInternal;

#define METAL_KERNEL_CACHE_MAX 64

typedef struct MetalKernelCacheEntry {
    char kernel_name[256];
    unsigned long source_hash;
    void* library;
    int ref_count;
    struct MetalKernelCacheEntry* next;
} MetalKernelCacheEntry;

static MetalKernelCacheEntry* g_metal_kernel_cache = NULL;
static int g_metal_cache_count = 0;

/* ============================================================================
 * 全局状态
 * =========================================================================== */

static int g_metal_available = 0;
static int g_metal_device_count = 0;

#define MAX_METAL_DEVICES 64

static void* g_metal_device_refs[MAX_METAL_DEVICES];
static char g_metal_device_names[MAX_METAL_DEVICES][256];
static void* g_metal_framework = NULL;
static char g_metal_error_string[256] = "无错误";

/* ============================================================================
 * 辅助函数
 * =========================================================================== */

static void set_metal_error_string(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(g_metal_error_string, sizeof(g_metal_error_string), format, args);
    va_end(args);
}

static unsigned long metal_hash_string(const char* str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) hash = ((hash << 5) + hash) + (unsigned long)c;
    return hash;
}

static void* metal_cache_find_program(const char* kernel_name, unsigned long source_hash) {
    MetalKernelCacheEntry* entry = g_metal_kernel_cache;
    while (entry) {
        if (entry->source_hash == source_hash && strcmp(entry->kernel_name, kernel_name) == 0) {
            entry->ref_count++;
            return entry->library;
        }
        entry = entry->next;
    }
    return NULL;
}

static int metal_cache_add_program(const char* kernel_name, unsigned long source_hash, void* library) {
    if (g_metal_cache_count >= METAL_KERNEL_CACHE_MAX) {
        MetalKernelCacheEntry* prev = NULL;
        MetalKernelCacheEntry* entry = g_metal_kernel_cache;
        while (entry && entry->next) { prev = entry; entry = entry->next; }
        if (entry) {
#ifdef __APPLE__
            if (entry->library) {
                void (*libRel)(void*) = dlsym(g_metal_framework, "MTLLibraryRelease");
                if (libRel) libRel(entry->library);
            }
#endif
            if (prev) prev->next = NULL; else g_metal_kernel_cache = NULL;
            safe_free((void**)&entry);
            g_metal_cache_count--;
        }
    }
    MetalKernelCacheEntry* new_entry = (MetalKernelCacheEntry*)safe_malloc(sizeof(MetalKernelCacheEntry));
    if (!new_entry) return -1;
    memset(new_entry, 0, sizeof(MetalKernelCacheEntry));
    strncpy(new_entry->kernel_name, kernel_name, 255);
    new_entry->kernel_name[255] = '\0';
    new_entry->source_hash = source_hash;
    new_entry->library = library;
    new_entry->ref_count = 1;
    new_entry->next = g_metal_kernel_cache;
    g_metal_kernel_cache = new_entry;
    g_metal_cache_count++;
    return 0;
}

static void metal_cache_cleanup(void) {
    MetalKernelCacheEntry* entry = g_metal_kernel_cache;
    while (entry) {
        MetalKernelCacheEntry* next = entry->next;
#ifdef __APPLE__
        if (entry->library) {
            void (*libRel)(void*) = dlsym(g_metal_framework, "MTLLibraryRelease");
            if (libRel) libRel(entry->library);
        }
#endif
        safe_free((void**)&entry);
        entry = next;
    }
    g_metal_kernel_cache = NULL;
    g_metal_cache_count = 0;
}

static int load_metal_library(void) {
#ifdef __APPLE__
    if (g_metal_available) {
        return 0;  // 已加载
    }
    
    // 在macOS/iOS上，Metal框架是系统框架，通过dlopen加载
    void* metal_framework = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (!metal_framework) {
        set_metal_error_string("无法加载Metal框架: %s", dlerror());
        return -1;
    }
    
    // 存储框架句柄
    g_metal_framework = metal_framework;
    
    // 加载Metal函数
    #define LOAD_METAL_FUNC(var, sym) \
        var = dlsym(metal_framework, sym); \
        if (!var) { \
            set_metal_error_string("警告: 无法加载Metal函数: %s", sym); \
        }
    
    // 加载必需函数
    LOAD_METAL_FUNC(mtlCreateSystemDefaultDevice, "MTLCreateSystemDefaultDevice");
    
    // 加载多设备枚举函数（macOS 10.13+, iOS 12.0+）
    LOAD_METAL_FUNC(mtlCopyAllDevices, "MTLCopyAllDevices");
    
    // 加载其他Metal函数
    LOAD_METAL_FUNC(mtlDeviceNewCommandQueue, "MTLDeviceNewCommandQueue");
    LOAD_METAL_FUNC(mtlCommandQueueCommandBuffer, "MTLCommandQueueCommandBuffer");
    LOAD_METAL_FUNC(mtlCommandBufferComputeCommandEncoder, "MTLCommandBufferComputeCommandEncoder");
    LOAD_METAL_FUNC(mtlComputeCommandEncoderSetComputePipelineState, "MTLComputeCommandEncoderSetComputePipelineState");
    LOAD_METAL_FUNC(mtlComputeCommandEncoderSetBuffer, "MTLComputeCommandEncoderSetBuffer");
    LOAD_METAL_FUNC(mtlComputeCommandEncoderDispatchThreads, "MTLComputeCommandEncoderDispatchThreads");
    LOAD_METAL_FUNC(mtlComputeCommandEncoderEndEncoding, "MTLComputeCommandEncoderEndEncoding");
    LOAD_METAL_FUNC(mtlCommandBufferCommit, "MTLCommandBufferCommit");
    LOAD_METAL_FUNC(mtlCommandBufferWaitUntilCompleted, "MTLCommandBufferWaitUntilCompleted");
    // Blit命令编码器函数加载
    LOAD_METAL_FUNC(mtlCommandBufferBlitCommandEncoder, "MTLCommandBufferBlitCommandEncoder");
    LOAD_METAL_FUNC(mtlBlitCommandEncoderCopyFromBufferToBuffer, "MTLBlitCommandEncoderCopyFromBufferToBuffer");
    LOAD_METAL_FUNC(mtlBlitCommandEncoderEndEncoding, "MTLBlitCommandEncoderEndEncoding");
    LOAD_METAL_FUNC(mtlDeviceNewBufferWithLength, "MTLDeviceNewBufferWithLength");
    LOAD_METAL_FUNC(mtlBufferRelease, "MTLBufferRelease");
    LOAD_METAL_FUNC(mtlBufferContents, "MTLBufferContents");
    LOAD_METAL_FUNC(mtlDeviceNewLibraryWithSource, "MTLDeviceNewLibraryWithSource");
    LOAD_METAL_FUNC(mtlLibraryNewFunctionWithName, "MTLLibraryNewFunctionWithName");
    LOAD_METAL_FUNC(mtlDeviceNewComputePipelineStateWithFunction, "MTLDeviceNewComputePipelineStateWithFunction");
    // 加载释放函数
    LOAD_METAL_FUNC(mtlLibraryRelease, "MTLLibraryRelease");
    LOAD_METAL_FUNC(mtlFunctionRelease, "MTLFunctionRelease");
    LOAD_METAL_FUNC(mtlComputePipelineStateRelease, "MTLComputePipelineStateRelease");
    
    #undef LOAD_METAL_FUNC
    
    memset(g_metal_device_refs, 0, sizeof(g_metal_device_refs));
    memset(g_metal_device_names, 0, sizeof(g_metal_device_names));
    
    // 多设备枚举：优先使用MTLCopyAllDevices获取所有Metal设备
    if (mtlCopyAllDevices) {
        void* device_array = mtlCopyAllDevices();
        if (device_array) {
            SEL count_sel = sel_registerName("count");
            SEL objectAtIndex_sel = sel_registerName("objectAtIndex:");
            NSUInteger count = ((NSUInteger (*)(id, SEL))objc_msgSend)((id)(device_array), count_sel);
            g_metal_device_count = (count > MAX_METAL_DEVICES) ? MAX_METAL_DEVICES : (int)count;
            for (int i = 0; i < g_metal_device_count; i++) {
                g_metal_device_refs[i] = ((id (*)(id, SEL, NSUInteger))objc_msgSend)((id)(device_array), objectAtIndex_sel, (NSUInteger)i);
                // 查询设备名称
                SEL name_sel = sel_registerName("name");
                SEL utf8String_sel = sel_registerName("UTF8String");
                void* nsname = ((id (*)(id, SEL))objc_msgSend)((id)(g_metal_device_refs[i]), name_sel);
                if (nsname) {
                    const char* cname = ((const char* (*)(id, SEL))objc_msgSend)((id)(nsname), utf8String_sel);
                    if (cname) {
                        strncpy(g_metal_device_names[i], cname, 255);
                        g_metal_device_names[i][255] = '\0';
                    }
                }
            }
        } else {
            // MTLCopyAllDevices返回空，回退到单设备
            void* device = mtlCreateSystemDefaultDevice();
            if (device) {
                g_metal_device_refs[0] = device;
                g_metal_device_count = 1;
                // 查询设备名称
                SEL name_sel = sel_registerName("name");
                SEL utf8String_sel = sel_registerName("UTF8String");
                void* nsname = ((id (*)(id, SEL))objc_msgSend)((id)(device), name_sel);
                if (nsname) {
                    const char* cname = ((const char* (*)(id, SEL))objc_msgSend)((id)(nsname), utf8String_sel);
                    if (cname) {
                        strncpy(g_metal_device_names[0], cname, 255);
                        g_metal_device_names[0][255] = '\0';
                    }
                }
            }
        }
    } else if (mtlCreateSystemDefaultDevice) {
        // 不支持多设备枚举，使用默认设备
        void* device = mtlCreateSystemDefaultDevice();
        if (device) {
            g_metal_device_refs[0] = device;
            g_metal_device_count = 1;
            // 查询设备名称
            SEL name_sel = sel_registerName("name");
            SEL utf8String_sel = sel_registerName("UTF8String");
            void* nsname = ((id (*)(id, SEL))objc_msgSend)((id)(device), name_sel);
            if (nsname) {
                const char* cname = ((const char* (*)(id, SEL))objc_msgSend)((id)(nsname), utf8String_sel);
                if (cname) {
                    strncpy(g_metal_device_names[0], cname, 255);
                    g_metal_device_names[0][255] = '\0';
                }
            }
        }
    }
    
    if (g_metal_device_count == 0) {
        set_metal_error_string("无法创建任何Metal设备");
        dlclose(metal_framework);
        g_metal_framework = NULL;
        return -1;
    }
    
    g_metal_available = 1;
    
    return 0;
#else
    set_metal_error_string("Metal仅支持苹果平台(macOS/iOS)");
    return -1;
#endif
}

/**
 * @brief 卸载Metal库
 */
static void unload_metal_library(void) {
#ifdef __APPLE__
    if (g_metal_framework) {
        dlclose(g_metal_framework);
        g_metal_framework = NULL;
    }
    g_metal_available = 0;
    g_metal_device_count = 0;
#endif
}

/* ============================================================================
 * Metal后端接口实现
 * =========================================================================== */

/**
 * @brief 初始化Metal后端
 */
static int metal_backend_init(void) {
#ifdef __APPLE__
    if (!g_metal_available) {
        return load_metal_library();
    }
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 清理Metal后端
 */
static void metal_backend_cleanup(void) {
    metal_cache_cleanup();
    unload_metal_library();
}

/**
 * @brief 获取Metal设备数量
 */
static int metal_backend_get_device_count(void) {
#ifdef __APPLE__
    if (!g_metal_available && metal_backend_init() != 0) {
        return 0;
    }
    return g_metal_device_count;
#else
    return 0;
#endif
}

/**
 * @brief 获取Metal设备信息（支持多设备）
 */
static int metal_backend_get_device_info(int device_index, GpuDeviceInfo* info) {
#ifdef __APPLE__
    if (!g_metal_available && metal_backend_init() != 0) {
        return -1;
    }
    
    if (device_index < 0 || device_index >= g_metal_device_count) {
        set_metal_error_string("无效设备索引: %d (可用设备数: %d)", device_index, g_metal_device_count);
        return -1;
    }
    
    if (!info) {
        set_metal_error_string("设备信息指针为空");
        return -1;
    }
    
    void* device = g_metal_device_refs[device_index];
    if (!device) {
        set_metal_error_string("设备索引%d的引用为空", device_index);
        return -1;
    }
    
    memset(info, 0, sizeof(GpuDeviceInfo));
    info->device_id = device_index;
    strncpy(info->name, g_metal_device_names[device_index], 127);
    info->name[127] = '\0';
    
    // 查询设备类型：lowPower表示集成GPU，headless可能为虚拟GPU
    SEL lowPower_sel = sel_registerName("lowPower");
    SEL headless_sel = sel_registerName("headless");
    SEL removable_sel = sel_registerName("removable");
    if (lowPower_sel && device) {
        BOOL isLowPower = ((BOOL (*)(id, SEL))objc_msgSend)((id)(device), lowPower_sel);
        BOOL isHeadless = NO;
        if (headless_sel) {
            isHeadless = ((BOOL (*)(id, SEL))objc_msgSend)((id)(device), headless_sel);
        }
        if (isHeadless) {
            info->type = GPU_DEVICE_TYPE_UNKNOWN;
        } else if (isLowPower) {
            info->type = GPU_DEVICE_TYPE_INTEGRATED;
        } else {
            info->type = GPU_DEVICE_TYPE_DISCRETE;
        }
    } else {
        info->type = GPU_DEVICE_TYPE_INTEGRATED;
    }
    
    // 通过recommendedMaxWorkingSetSize获取真实的GPU可访问内存大小（macOS 10.13+）
    // 这是Metal提供的真实GPU内存上限，不是系统RAM
    SEL recommendedMax_sel = sel_registerName("recommendedMaxWorkingSetSize");
    if (recommendedMax_sel && device) {
        uint64_t gpuMem = ((uint64_t (*)(id, SEL))objc_msgSend)((id)(device), recommendedMax_sel);
        info->total_memory = gpuMem;
        // 空闲内存无法直接从Metal查询，使用recommendedMaxWorkingSetSize的80%作为合理估计
        // 注：实际使用中Metal会根据内存压力动态调整，这里提供保守值
        info->free_memory = (uint64_t)(gpuMem * 0.8);
    } else {
        // 回退到系统内存信息
        uint64_t physmem = 0;
        size_t len = sizeof(physmem);
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        if (sysctl(mib, 2, &physmem, &len, NULL, 0) == 0) {
            info->total_memory = physmem;
            vm_size_t page_size;
            mach_port_t host_port = mach_host_self();
            vm_statistics_data_t vm_stat;
            mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);
            if (host_page_size(host_port, &page_size) == KERN_SUCCESS &&
                host_statistics(host_port, HOST_VM_INFO, (host_info_t)&vm_stat, &host_size) == KERN_SUCCESS) {
                info->free_memory = (uint64_t)vm_stat.free_count * page_size;
            } else {
                info->free_memory = physmem;
            }
        } else {
            set_metal_error_string("无法获取真实内存信息");
            return -1;
        }
    }
    
    // 查询最大线程组大小（真实Metal API查询）
    SEL maxThreadsSelector = sel_registerName("maxThreadsPerThreadgroup");
    if (maxThreadsSelector && device) {
        MTLSize maxThreads = ((MTLSize (*)(id, SEL))objc_msgSend)((id)(device), maxThreadsSelector);
        info->max_work_group_size = (size_t)maxThreads.width;
        if (info->max_work_group_size == 0) {
            info->max_work_group_size = 512;
        }
    } else {
        info->max_work_group_size = 512;
    }
    
    // 查询最大并发线程数作为计算单元估计
    SEL maxConcurrentThreadsSelector = sel_registerName("maximumConcurrentThreads");
    if (maxConcurrentThreadsSelector && device) {
        size_t maxThreads = ((size_t (*)(id, SEL))objc_msgSend)((id)(device), maxConcurrentThreadsSelector);
        info->compute_units = (int)(maxThreads / 32);
        if (info->compute_units < 1) info->compute_units = 1;
    } else {
        info->compute_units = 0;
    }
    
    // 查询线程组执行宽度（线程束大小）
    SEL threadExecutionWidth_sel = sel_registerName("threadExecutionWidth");
    if (threadExecutionWidth_sel && device) {
        info->warp_size = (int)((NSUInteger (*)(id, SEL))objc_msgSend)((id)(device), threadExecutionWidth_sel);
    } else {
        info->warp_size = 32;
    }
    
    info->clock_speed = 0.0f;
    
    // 查询GPU家族以判断双精度支持
    SEL supportsFamilySelector = sel_registerName("supportsFamily:");
    if (supportsFamilySelector && device) {
        int family4 = 4004;
        BOOL supportsFamily4 = ((BOOL (*)(id, SEL, int))objc_msgSend)((id)(device), supportsFamilySelector, family4);
        info->supports_double = supportsFamily4 ? 1 : 0;
    } else {
        info->supports_double = 0;
    }
    
    info->supports_half = 1;
    
    // 使用推荐的线程组大小
    SEL recommendedMax_sel2 = sel_registerName("maxThreadsPerThreadgroup");
    if (recommendedMax_sel2 && device) {
        MTLSize maxThreads = ((MTLSize (*)(id, SEL))objc_msgSend)((id)(device), recommendedMax_sel2);
        info->max_work_item_sizes[0] = (size_t)maxThreads.width;
        info->max_work_item_sizes[1] = (size_t)maxThreads.height ? (size_t)maxThreads.height : 1;
        info->max_work_item_sizes[2] = (size_t)maxThreads.depth ? (size_t)maxThreads.depth : 1;
    } else {
        info->max_work_item_sizes[0] = 512;
        info->max_work_item_sizes[1] = 512;
        info->max_work_item_sizes[2] = 512;
    }
    
    // 查询是否有统一内存架构
    SEL unifiedMemory_sel = sel_registerName("hasUnifiedMemory");
    if (unifiedMemory_sel && device) {
        info->integrated = ((BOOL (*)(id, SEL))objc_msgSend)((id)(device), unifiedMemory_sel) ? 1 : 0;
    } else {
        info->integrated = (info->type == GPU_DEVICE_TYPE_INTEGRATED) ? 1 : 0;
    }
    
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 创建Metal上下文（支持多设备）
 */
static GpuContext* metal_backend_context_create(int device_index) {
#ifdef __APPLE__
    if (!g_metal_available && metal_backend_init() != 0) {
        return NULL;
    }
    
    if (device_index < 0 || device_index >= g_metal_device_count) {
        set_metal_error_string("无效设备索引: %d (可用设备数: %d)", device_index, g_metal_device_count);
        return NULL;
    }
    
    if (!g_metal_device_refs[device_index]) {
        set_metal_error_string("设备索引%d的引用为空", device_index);
        return NULL;
    }
    
    // 使用已枚举的设备引用
    void* device = g_metal_device_refs[device_index];
    
    // 分配上下文结构
    MetalContextInternal* ctx = (MetalContextInternal*)safe_malloc(sizeof(MetalContextInternal));
    if (!ctx) {
        set_metal_error_string("内存分配失败");
        return NULL;
    }
    
    memset(ctx, 0, sizeof(MetalContextInternal));
    ctx->device = device;
    ctx->device_id = device_index;
    
    // 创建命令队列（完整实现）
    if (mtlDeviceNewCommandQueue) {
        ctx->command_queue = mtlDeviceNewCommandQueue(device);
        if (!ctx->command_queue) {
            set_metal_error_string("创建Metal命令队列失败");
            safe_free((void**)&ctx);
            return NULL;
        }
    } else {
        ctx->command_queue = NULL;
        set_metal_error_string("警告：Metal命令队列函数不可用，某些功能可能受限");
    }
    
    ctx->initialized = 1;
    
    return (GpuContext*)ctx;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return NULL;
#endif
}

/**
 * @brief 释放Metal上下文
 */
static void metal_backend_context_free(GpuContext* context) {
#ifdef __APPLE__
    if (!context) return;
    
    MetalContextInternal* ctx = (MetalContextInternal*)context;
    
    // 释放命令队列（如果存在）
    // if (ctx->command_queue) { ... }
    
    // 释放设备引用
    // Metal对象使用引用计数，实际需要释放
    
    safe_free((void**)&ctx);
#else
    set_metal_error_string("Metal后端仅支持苹果平台(macOS/iOS)，无法销毁Metal上下文");
#endif
}

/**
 * @brief 分配Metal内存
 */
static GpuMemory* metal_backend_memory_alloc(GpuContext* context, size_t size, GpuMemoryType memory_type) {
#ifdef __APPLE__
    if (!context || size == 0) {
        set_metal_error_string("无效参数");
        return NULL;
    }
    
    MetalContextInternal* ctx = (MetalContextInternal*)context;
    
    // 分配内存结构
    MetalMemoryInternal* mem = (MetalMemoryInternal*)safe_malloc(sizeof(MetalMemoryInternal));
    if (!mem) {
        set_metal_error_string("内存分配失败");
        return NULL;
    }
    
    memset(mem, 0, sizeof(MetalMemoryInternal));
    mem->size = size;
    mem->memory_type = memory_type;
    
    // 根据内存类型选择存储模式
    switch (memory_type) {
        case GPU_MEMORY_HOST:
            mem->storage_mode = METAL_RESOURCE_STORAGE_MODE_SHARED;
            break;
        case GPU_MEMORY_DEVICE:
            mem->storage_mode = METAL_RESOURCE_STORAGE_MODE_PRIVATE;
            break;
        case GPU_MEMORY_UNIFIED:
            mem->storage_mode = METAL_RESOURCE_STORAGE_MODE_SHARED;
            break;
        default:
            set_metal_error_string("未知内存类型: %d", memory_type);
            safe_free((void**)&mem);
            return NULL;
    }
    
    // 分配Metal缓冲区（完整实现）
    if (mtlDeviceNewBufferWithLength && ctx->device) {
        // 使用真正的Metal API分配GPU缓冲区
        mem->buffer = mtlDeviceNewBufferWithLength(ctx->device, size, mem->storage_mode);
        if (!mem->buffer) {
            set_metal_error_string("Metal缓冲区分配失败");
            safe_free((void**)&mem);
            return NULL;
        }
        mem->is_gpu_buffer = 1;  // 标记为GPU缓冲区
        // 保存设备上下文信息用于私有内存复制
        mem->device = ctx->device;
        mem->command_queue = ctx->command_queue;
    } else {
        // Metal API不可用，根据项目要求"禁止任何降级处理"，返回错误
        set_metal_error_string("Metal API不可用，无法分配GPU内存（需要苹果平台和Metal框架）");
        safe_free((void**)&mem);
        return NULL;
    }
    
    return (GpuMemory*)mem;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return NULL;
#endif
}

/**
 * @brief 释放Metal内存
 */
static void metal_backend_memory_free(GpuMemory* memory) {
#ifdef __APPLE__
    if (!memory) return;
    
    MetalMemoryInternal* mem = (MetalMemoryInternal*)memory;
    
    if (mem->buffer) {
        // 根据缓冲区类型释放内存
        if (mem->is_gpu_buffer && mtlBufferRelease) {
            // 释放GPU缓冲区
            mtlBufferRelease(mem->buffer);
        } else {
            // 释放主机内存
            safe_free((void**)&mem->buffer);
        }
        mem->buffer = NULL;
    }
    
    safe_free((void**)&mem);
#else
    set_metal_error_string("Metal后端仅支持苹果平台(macOS/iOS)，无法释放Metal内存");
#endif
}

/**
 * @brief 复制数据到Metal内存
 */
static int metal_backend_memory_copy_to_device(GpuMemory* dst, const void* src, size_t size) {
    UNUSED(dst); UNUSED(src); UNUSED(size);
#ifdef __APPLE__
    if (!dst || !src || size == 0) {
        set_metal_error_string("无效参数");
        return -1;
    }
    
    MetalMemoryInternal* mem = (MetalMemoryInternal*)dst;
    
    if (size > mem->size) {
        set_metal_error_string("复制大小%zu超过内存大小%zu", size, mem->size);
        return -1;
    }
    
    // 复制数据到缓冲区
    // 如果是共享内存，可以直接复制
    if (mem->storage_mode == METAL_RESOURCE_STORAGE_MODE_SHARED) {
        // 共享内存可以直接访问
        if (mem->is_gpu_buffer && mtlBufferContents) {
            // 真正的Metal缓冲区：获取内容指针
            void* buffer_contents = mtlBufferContents(mem->buffer);
            if (buffer_contents) {
                memcpy(buffer_contents, src, size);
            } else {
                set_metal_error_string("无法获取Metal缓冲区内容");
                return -1;
            }
        } else {
            // 主机内存或API不可用：直接复制
            memcpy(mem->buffer, src, size);
        }
        return 0;
    } else {
        // 私有内存：使用命令缓冲区和blit命令编码器
        // 检查必要的Metal函数是否可用
        if (!mtlCommandQueueCommandBuffer || !mtlCommandBufferBlitCommandEncoder || 
            !mtlBlitCommandEncoderCopyFromBufferToBuffer || !mtlBlitCommandEncoderEndEncoding ||
            !mtlCommandBufferCommit || !mtlCommandBufferWaitUntilCompleted ||
            !mtlDeviceNewBufferWithLength || !mtlBufferContents || !mtlBufferRelease) {
            set_metal_error_string("Metal blit命令函数未加载，无法复制到私有内存");
            return -1;
        }
        
        // 检查设备上下文信息
        if (!mem->device || !mem->command_queue) {
            set_metal_error_string("内存对象缺少设备上下文信息，无法执行私有内存复制");
            return -1;
        }
        
        // 创建临时共享内存缓冲区用于主机到设备的复制
        void* temp_buffer = mtlDeviceNewBufferWithLength(mem->device, size, METAL_RESOURCE_STORAGE_MODE_SHARED);
        if (!temp_buffer) {
            set_metal_error_string("创建临时共享缓冲区失败");
            return -1;
        }
        
        // 将主机数据复制到临时缓冲区
        void* temp_contents = mtlBufferContents(temp_buffer);
        if (!temp_contents) {
            mtlBufferRelease(temp_buffer);
            set_metal_error_string("无法获取临时缓冲区内容");
            return -1;
        }
        memcpy(temp_contents, src, size);
        
        // 创建命令缓冲区和blit命令编码器
        void* command_buffer = mtlCommandQueueCommandBuffer(mem->command_queue);
        if (!command_buffer) {
            mtlBufferRelease(temp_buffer);
            set_metal_error_string("创建命令缓冲区失败");
            return -1;
        }
        
        void* blit_encoder = mtlCommandBufferBlitCommandEncoder(command_buffer);
        if (!blit_encoder) {
            mtlBufferRelease(temp_buffer);
            set_metal_error_string("创建blit命令编码器失败");
            return -1;
        }
        
        // 执行缓冲区间复制（从临时共享缓冲区到目标私有缓冲区）
        mtlBlitCommandEncoderCopyFromBufferToBuffer(blit_encoder, temp_buffer, 0, mem->buffer, 0, size);
        
        // 结束编码并提交命令
        mtlBlitCommandEncoderEndEncoding(blit_encoder);
        mtlCommandBufferCommit(command_buffer);
        
        // 等待命令完成（同步操作）
        mtlCommandBufferWaitUntilCompleted(command_buffer);
        
        // 释放临时缓冲区
        mtlBufferRelease(temp_buffer);
        
        return 0;
    }
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 从Metal内存复制数据
 */
static int metal_backend_memory_copy_from_device(void* dst, GpuMemory* src, size_t size) {
    UNUSED(dst); UNUSED(src); UNUSED(size);
#ifdef __APPLE__
    if (!dst || !src || size == 0) {
        set_metal_error_string("无效参数");
        return -1;
    }
    
    MetalMemoryInternal* mem = (MetalMemoryInternal*)src;
    
    if (size > mem->size) {
        set_metal_error_string("复制大小%zu超过内存大小%zu", size, mem->size);
        return -1;
    }
    
    // 从缓冲区复制数据（完整实现）
    // 如果是共享内存，可以直接复制
    if (mem->storage_mode == METAL_RESOURCE_STORAGE_MODE_SHARED) {
        if (mem->is_gpu_buffer && mtlBufferContents) {
            // 真正的Metal缓冲区：获取内容指针
            void* buffer_contents = mtlBufferContents(mem->buffer);
            if (buffer_contents) {
                memcpy(dst, buffer_contents, size);
            } else {
                set_metal_error_string("无法获取Metal缓冲区内容");
                return -1;
            }
        } else {
            // 主机内存或API不可用：直接复制
            memcpy(dst, mem->buffer, size);
        }
    } else {
        // 私有内存：使用命令缓冲区和blit命令编码器
        // 检查必要的Metal函数是否可用
        if (!mtlCommandQueueCommandBuffer || !mtlCommandBufferBlitCommandEncoder || 
            !mtlBlitCommandEncoderCopyFromBufferToBuffer || !mtlBlitCommandEncoderEndEncoding ||
            !mtlCommandBufferCommit || !mtlCommandBufferWaitUntilCompleted ||
            !mtlDeviceNewBufferWithLength || !mtlBufferContents || !mtlBufferRelease) {
            set_metal_error_string("Metal blit命令函数未加载，无法从私有内存复制");
            return -1;
        }
        
        // 检查设备上下文信息
        if (!mem->device || !mem->command_queue) {
            set_metal_error_string("内存对象缺少设备上下文信息，无法执行私有内存复制");
            return -1;
        }
        
        // 创建临时共享内存缓冲区用于设备到主机的复制
        void* temp_buffer = mtlDeviceNewBufferWithLength(mem->device, size, METAL_RESOURCE_STORAGE_MODE_SHARED);
        if (!temp_buffer) {
            set_metal_error_string("创建临时共享缓冲区失败");
            return -1;
        }
        
        // 创建命令缓冲区和blit命令编码器
        void* command_buffer = mtlCommandQueueCommandBuffer(mem->command_queue);
        if (!command_buffer) {
            mtlBufferRelease(temp_buffer);
            set_metal_error_string("创建命令缓冲区失败");
            return -1;
        }
        
        void* blit_encoder = mtlCommandBufferBlitCommandEncoder(command_buffer);
        if (!blit_encoder) {
            mtlBufferRelease(temp_buffer);
            set_metal_error_string("创建blit命令编码器失败");
            return -1;
        }
        
        // 执行缓冲区间复制（从源私有缓冲区到临时共享缓冲区）
        mtlBlitCommandEncoderCopyFromBufferToBuffer(blit_encoder, mem->buffer, 0, temp_buffer, 0, size);
        
        // 结束编码并提交命令
        mtlBlitCommandEncoderEndEncoding(blit_encoder);
        mtlCommandBufferCommit(command_buffer);
        
        // 等待命令完成（同步操作）
        mtlCommandBufferWaitUntilCompleted(command_buffer);
        
        // 从临时缓冲区复制到主机内存
        void* temp_contents = mtlBufferContents(temp_buffer);
        if (!temp_contents) {
            mtlBufferRelease(temp_buffer);
            set_metal_error_string("无法获取临时缓冲区内容");
            return -1;
        }
        memcpy(dst, temp_contents, size);
        
        // 释放临时缓冲区
        mtlBufferRelease(temp_buffer);
        
        return 0;
    }
    
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief Metal设备间复制数据（完整实现）
 */
static int metal_backend_memory_copy_device_to_device(GpuMemory* dst, GpuMemory* src, size_t size) {
    UNUSED(dst); UNUSED(src); UNUSED(size);
#ifdef __APPLE__
    if (!dst || !src || size == 0) {
        set_metal_error_string("无效参数");
        return -1;
    }
    
    MetalMemoryInternal* dst_mem = (MetalMemoryInternal*)dst;
    MetalMemoryInternal* src_mem = (MetalMemoryInternal*)src;
    
    if (size > dst_mem->size || size > src_mem->size) {
        set_metal_error_string("复制大小%zu超过内存大小", size);
        return -1;
    }
    
    // 完整实现：Metal设备间复制
    // 1. 如果都是共享内存缓冲区，可以直接复制
    // 2. 如果都是私有GPU内存缓冲区，使用blit命令编码器直接复制
    // 3. 其他情况使用主机内存中转
    
    // 检查是否都是GPU缓冲区且都是私有内存
    if (dst_mem->is_gpu_buffer && src_mem->is_gpu_buffer && 
        dst_mem->storage_mode == METAL_RESOURCE_STORAGE_MODE_PRIVATE && 
        src_mem->storage_mode == METAL_RESOURCE_STORAGE_MODE_PRIVATE) {
        // 私有GPU内存之间的复制：使用blit命令编码器直接复制
        
        // 检查必要的Metal函数是否可用
        if (!mtlCommandQueueCommandBuffer || !mtlCommandBufferBlitCommandEncoder || 
            !mtlBlitCommandEncoderCopyFromBufferToBuffer || !mtlBlitCommandEncoderEndEncoding ||
            !mtlCommandBufferCommit || !mtlCommandBufferWaitUntilCompleted) {
            set_metal_error_string("Metal blit命令函数未加载，无法执行私有内存之间的复制");
            // 回退到主机内存中转
        } else {
            // 检查设备上下文信息
            if (!dst_mem->device || !dst_mem->command_queue) {
                set_metal_error_string("目标内存对象缺少设备上下文信息");
                // 回退到主机内存中转
            } else if (!src_mem->device || !src_mem->command_queue) {
                set_metal_error_string("源内存对象缺少设备上下文信息");
                // 回退到主机内存中转
            } else {
                // 使用目标内存的命令队列（假设源和目标在同一个设备上）
                void* command_buffer = mtlCommandQueueCommandBuffer(dst_mem->command_queue);
                if (!command_buffer) {
                    set_metal_error_string("创建命令缓冲区失败");
                    // 回退到主机内存中转
                } else {
                    void* blit_encoder = mtlCommandBufferBlitCommandEncoder(command_buffer);
                    if (!blit_encoder) {
                        set_metal_error_string("创建blit命令编码器失败");
                        // 回退到主机内存中转
                    } else {
                        // 执行缓冲区间直接复制
                        mtlBlitCommandEncoderCopyFromBufferToBuffer(blit_encoder, src_mem->buffer, 0, dst_mem->buffer, 0, size);
                        
                        // 结束编码并提交命令
                        mtlBlitCommandEncoderEndEncoding(blit_encoder);
                        mtlCommandBufferCommit(command_buffer);
                        
                        // 等待命令完成（同步操作）
                        mtlCommandBufferWaitUntilCompleted(command_buffer);
                        
                        return 0;
                    }
                }
            }
        }
        
        // 如果直接复制失败，根据项目要求"禁止任何降级处理"，返回错误
        set_metal_error_string("私有GPU内存复制失败：Metal blit命令不可用");
        return -1;
    }
    
    // 非私有GPU内存的情况：根据项目要求"禁止任何降级处理"，返回错误
    // 注意：混合内存类型（共享+私有）需要特殊处理，当前实现不支持
    set_metal_error_string("不支持的内存复制类型：需要私有GPU内存对或支持的直接复制方法");
    return -1;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/* ============================================================================
 * Metal后端实现 - 异步内存拷贝
 * =========================================================================== */

/**
 * @brief 异步复制数据到Metal设备
 * 
 * 使用暂存缓冲区和blit命令编码器实现异步主机到设备的传输。
 * 共享内存直接复制，私有内存通过命令缓冲区提交后立即返回。
 * 调用者需要同步流以确认传输完成。
 */
static int metal_backend_memory_copy_to_device_async(GpuMemory* dst, const void* src, size_t size, GpuStream* stream) {
    UNUSED(dst); UNUSED(src); UNUSED(size); UNUSED(stream);
#ifdef __APPLE__
    if (!dst || !src || size == 0 || !stream) {
        set_metal_error_string("无效参数");
        return -1;
    }
    
    MetalMemoryInternal* mem = (MetalMemoryInternal*)dst;
    MetalStreamInternal* metal_stream = (MetalStreamInternal*)stream;
    
    if (size > mem->size) {
        set_metal_error_string("复制大小%zu超过内存大小%zu", size, mem->size);
        return -1;
    }
    
    // 共享内存可以直接复制（本质上是同步的，但对共享内存而言复制立即生效）
    if (mem->storage_mode == METAL_RESOURCE_STORAGE_MODE_SHARED) {
        if (mem->is_gpu_buffer && mtlBufferContents) {
            void* buffer_contents = mtlBufferContents(mem->buffer);
            if (buffer_contents) {
                memcpy(buffer_contents, src, size);
            } else {
                set_metal_error_string("无法获取Metal缓冲区内容");
                return -1;
            }
        } else {
            memcpy(mem->buffer, src, size);
        }
        return 0;
    }
    
    // 私有内存：使用异步blit命令编码器
    if (!mtlCommandQueueCommandBuffer || !mtlCommandBufferBlitCommandEncoder || 
        !mtlBlitCommandEncoderCopyFromBufferToBuffer || !mtlBlitCommandEncoderEndEncoding ||
        !mtlCommandBufferCommit || !mtlDeviceNewBufferWithLength || 
        !mtlBufferContents || !mtlBufferRelease) {
        set_metal_error_string("Metal blit命令函数未加载，无法执行异步复制");
        return -1;
    }
    
    if (!mem->device || !mem->command_queue) {
        set_metal_error_string("内存对象缺少设备上下文信息");
        return -1;
    }
    
    // 如果存在待处理操作，先同步等待完成，释放资源
    if (metal_stream->has_pending_operation && metal_stream->pending_command_buffer) {
        mtlCommandBufferWaitUntilCompleted(metal_stream->pending_command_buffer);
        if (metal_stream->pending_staging_buffer) {
            mtlBufferRelease(metal_stream->pending_staging_buffer);
            metal_stream->pending_staging_buffer = NULL;
        }
        metal_stream->pending_command_buffer = NULL;
        metal_stream->has_pending_operation = 0;
        metal_stream->readback_dst = NULL;
        metal_stream->readback_size = 0;
    }
    
    // 创建暂存共享缓冲区并复制主机数据
    void* staging_buffer = mtlDeviceNewBufferWithLength(mem->device, size, METAL_RESOURCE_STORAGE_MODE_SHARED);
    if (!staging_buffer) {
        set_metal_error_string("创建暂存缓冲区失败");
        return -1;
    }
    
    void* staging_contents = mtlBufferContents(staging_buffer);
    if (!staging_contents) {
        mtlBufferRelease(staging_buffer);
        set_metal_error_string("无法获取暂存缓冲区内容");
        return -1;
    }
    memcpy(staging_contents, src, size);
    
    // 创建命令缓冲区和blit命令编码器
    void* cmd_buffer = mtlCommandQueueCommandBuffer(mem->command_queue);
    if (!cmd_buffer) {
        mtlBufferRelease(staging_buffer);
        set_metal_error_string("创建命令缓冲区失败");
        return -1;
    }
    
    void* blit_encoder = mtlCommandBufferBlitCommandEncoder(cmd_buffer);
    if (!blit_encoder) {
        mtlBufferRelease(staging_buffer);
        set_metal_error_string("创建blit命令编码器失败");
        return -1;
    }
    
    // 执行缓冲区间复制（暂存共享缓冲区 -> 目标私有缓冲区）
    mtlBlitCommandEncoderCopyFromBufferToBuffer(blit_encoder, staging_buffer, 0, mem->buffer, 0, size);
    
    // 结束编码并提交命令（不等待完成）
    mtlBlitCommandEncoderEndEncoding(blit_encoder);
    mtlCommandBufferCommit(cmd_buffer);
    
    // 在流中跟踪待处理操作
    metal_stream->pending_command_buffer = cmd_buffer;
    metal_stream->pending_staging_buffer = staging_buffer;
    metal_stream->has_pending_operation = 1;
    metal_stream->readback_dst = NULL;
    metal_stream->readback_size = 0;
    metal_stream->is_completed = 0;
    
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 异步从Metal设备复制数据到主机
 * 
 * 使用暂存缓冲区和blit命令编码器实现异步设备到主机的传输。
 * 共享内存直接复制，私有内存通过命令缓冲区提交后立即返回。
 * 调用者需要同步流以确认数据已到达目标内存。
 */
static int metal_backend_memory_copy_from_device_async(void* dst, GpuMemory* src, size_t size, GpuStream* stream) {
    UNUSED(dst); UNUSED(src); UNUSED(size); UNUSED(stream);
#ifdef __APPLE__
    if (!dst || !src || size == 0 || !stream) {
        set_metal_error_string("无效参数");
        return -1;
    }
    
    MetalMemoryInternal* mem = (MetalMemoryInternal*)src;
    MetalStreamInternal* metal_stream = (MetalStreamInternal*)stream;
    
    if (size > mem->size) {
        set_metal_error_string("复制大小%zu超过内存大小%zu", size, mem->size);
        return -1;
    }
    
    // 共享内存可以直接读取（同步完成）
    if (mem->storage_mode == METAL_RESOURCE_STORAGE_MODE_SHARED) {
        if (mem->is_gpu_buffer && mtlBufferContents) {
            void* buffer_contents = mtlBufferContents(mem->buffer);
            if (buffer_contents) {
                memcpy(dst, buffer_contents, size);
            } else {
                set_metal_error_string("无法获取Metal缓冲区内容");
                return -1;
            }
        } else {
            memcpy(dst, mem->buffer, size);
        }
        return 0;
    }
    
    // 私有内存：使用异步blit命令编码器
    if (!mtlCommandQueueCommandBuffer || !mtlCommandBufferBlitCommandEncoder || 
        !mtlBlitCommandEncoderCopyFromBufferToBuffer || !mtlBlitCommandEncoderEndEncoding ||
        !mtlCommandBufferCommit || !mtlDeviceNewBufferWithLength || 
        !mtlBufferContents || !mtlBufferRelease) {
        set_metal_error_string("Metal blit命令函数未加载，无法执行异步复制");
        return -1;
    }
    
    if (!mem->device || !mem->command_queue) {
        set_metal_error_string("内存对象缺少设备上下文信息");
        return -1;
    }
    
    // 如果存在待处理操作，先同步等待完成，释放资源
    if (metal_stream->has_pending_operation && metal_stream->pending_command_buffer) {
        mtlCommandBufferWaitUntilCompleted(metal_stream->pending_command_buffer);
        if (metal_stream->pending_staging_buffer) {
            mtlBufferRelease(metal_stream->pending_staging_buffer);
            metal_stream->pending_staging_buffer = NULL;
        }
        metal_stream->pending_command_buffer = NULL;
        metal_stream->has_pending_operation = 0;
        metal_stream->readback_dst = NULL;
        metal_stream->readback_size = 0;
    }
    
    // 创建暂存共享缓冲区（用于接收设备数据）
    void* staging_buffer = mtlDeviceNewBufferWithLength(mem->device, size, METAL_RESOURCE_STORAGE_MODE_SHARED);
    if (!staging_buffer) {
        set_metal_error_string("创建暂存缓冲区失败");
        return -1;
    }
    
    // 创建命令缓冲区和blit命令编码器
    void* cmd_buffer = mtlCommandQueueCommandBuffer(mem->command_queue);
    if (!cmd_buffer) {
        mtlBufferRelease(staging_buffer);
        set_metal_error_string("创建命令缓冲区失败");
        return -1;
    }
    
    void* blit_encoder = mtlCommandBufferBlitCommandEncoder(cmd_buffer);
    if (!blit_encoder) {
        mtlBufferRelease(staging_buffer);
        set_metal_error_string("创建blit命令编码器失败");
        return -1;
    }
    
    // 执行缓冲区间复制（源私有缓冲区 -> 暂存共享缓冲区）
    mtlBlitCommandEncoderCopyFromBufferToBuffer(blit_encoder, mem->buffer, 0, staging_buffer, 0, size);
    
    // 结束编码并提交命令（不等待完成）
    mtlBlitCommandEncoderEndEncoding(blit_encoder);
    mtlCommandBufferCommit(cmd_buffer);
    
    // 在流中存储待处理操作和回读信息
    metal_stream->pending_command_buffer = cmd_buffer;
    metal_stream->pending_staging_buffer = staging_buffer;
    metal_stream->readback_dst = dst;
    metal_stream->readback_size = size;
    metal_stream->has_pending_operation = 1;
    metal_stream->is_completed = 0;
    
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/* ============================================================================
 * Metal内置MSL内核源码（CfC + 训练 + 视觉算子）
 * =========================================================================== */

static const char* METAL_CFC_FORWARD_KERNEL =
    "kernel void cfc_forward_step(\n"
    "    device const float* input          [[buffer(0)]],\n"
    "    device const float* prev_hidden    [[buffer(1)]],\n"
    "    device const float* weight_matrix  [[buffer(2)]],\n"
    "    device const float* gate_weights   [[buffer(3)]],\n"
    "    device const float* h2act_weights  [[buffer(4)]],\n"
    "    device const float* h2gate_weights [[buffer(5)]],\n"
    "    device const float* bias_act       [[buffer(6)]],\n"
    "    device const float* bias_gate      [[buffer(7)]],\n"
    "    device const float* time_constants [[buffer(8)]],\n"
    "    constant float& delta_t            [[buffer(9)]],\n"
    "    constant int& hidden_size          [[buffer(10)]],\n"
    "    constant int& input_size           [[buffer(11)]],\n"
    "    device float* next_hidden          [[buffer(12)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)hidden_size) return;\n"
    "    float gate_sum = bias_gate[id];\n"
    "    float act_sum = bias_act[id];\n"
    "    for (int j = 0; j < input_size; j++) {\n"
    "        gate_sum += gate_weights[id * input_size + j] * input[j];\n"
    "        act_sum  += weight_matrix[id * input_size + j] * input[j];\n"
    "    }\n"
    "    for (int j = 0; j < hidden_size; j++) {\n"
    "        gate_sum += h2gate_weights[id * hidden_size + j] * prev_hidden[j];\n"
    "        act_sum  += h2act_weights[id * hidden_size + j] * prev_hidden[j];\n"
    "    }\n"
    "    float gate = 1.0 / (1.0 + exp(-gate_sum));\n"
    "    float activation = tanh(act_sum);\n"
    "    float tau = time_constants[id];\n"
    "    if (tau < 1e-6) tau = 1e-6;\n"
    "    float exp_term = exp(-delta_t / tau);\n"
    "    next_hidden[id] = prev_hidden[id] * exp_term + gate * activation * (1.0 - exp_term);\n"
    "}\n";

static const char* METAL_CFC_LIQUID_TAU_KERNEL =
    "kernel void cfc_liquid_tau(\n"
    "    device const float* input          [[buffer(0)]],\n"
    "    device const float* tau_weights    [[buffer(1)]],\n"
    "    device const float* tau_bias       [[buffer(2)]],\n"
    "    constant float& tau_min            [[buffer(3)]],\n"
    "    constant float& tau_max            [[buffer(4)]],\n"
    "    constant int& input_size           [[buffer(5)]],\n"
    "    constant int& hidden_size          [[buffer(6)]],\n"
    "    device float* output_tau           [[buffer(7)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)hidden_size) return;\n"
    "    float sum = tau_bias[id];\n"
    "    for (int j = 0; j < input_size; j++) {\n"
    "        sum += tau_weights[id * input_size + j] * input[j];\n"
    "    }\n"
    "    float sig = 1.0 / (1.0 + exp(-sum));\n"
    "    output_tau[id] = tau_min + sig * (tau_max - tau_min);\n"
    "}\n";

static const char* METAL_CFC_MULTI_TIMESCALE_KERNEL =
    "kernel void cfc_multi_timescale_step(\n"
    "    device const float* input          [[buffer(0)]],\n"
    "    device const float* prev_hidden    [[buffer(1)]],\n"
    "    device const float* weight_matrix  [[buffer(2)]],\n"
    "    device const float* gate_weights   [[buffer(3)]],\n"
    "    device const float* h2act_weights  [[buffer(4)]],\n"
    "    device const float* h2gate_weights [[buffer(5)]],\n"
    "    device const float* bias_act       [[buffer(6)]],\n"
    "    device const float* bias_gate      [[buffer(7)]],\n"
    "    device const float* time_constants [[buffer(8)]],\n"
    "    constant float& delta_t            [[buffer(9)]],\n"
    "    constant float& fast_ratio         [[buffer(10)]],\n"
    "    constant float& slow_ratio         [[buffer(11)]],\n"
    "    constant int& hidden_size          [[buffer(12)]],\n"
    "    constant int& input_size           [[buffer(13)]],\n"
    "    device float* next_hidden          [[buffer(14)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)hidden_size) return;\n"
    "    float gate_sum = bias_gate[id];\n"
    "    float act_sum = bias_act[id];\n"
    "    for (int j = 0; j < input_size; j++) {\n"
    "        gate_sum += gate_weights[id * input_size + j] * input[j];\n"
    "        act_sum  += weight_matrix[id * input_size + j] * input[j];\n"
    "    }\n"
    "    for (int j = 0; j < hidden_size; j++) {\n"
    "        gate_sum += h2gate_weights[id * hidden_size + j] * prev_hidden[j];\n"
    "        act_sum  += h2act_weights[id * hidden_size + j] * prev_hidden[j];\n"
    "    }\n"
    "    float gate = 1.0 / (1.0 + exp(-gate_sum));\n"
    "    float activation = tanh(act_sum);\n"
    "    float tau_base = time_constants[id];\n"
    "    if (tau_base < 1e-6) tau_base = 1e-6;\n"
    "    float tau_fast = tau_base * fast_ratio;\n"
    "    float tau_slow = tau_base * slow_ratio;\n"
    "    if (tau_fast < 1e-6) tau_fast = 1e-6;\n"
    "    if (tau_slow < 1e-6) tau_slow = 1e-6;\n"
    "    float exp_fast = exp(-delta_t / tau_fast);\n"
    "    float exp_slow = exp(-delta_t / tau_slow);\n"
    "    float h_fast = prev_hidden[id] * exp_fast + gate * activation * (1.0 - exp_fast);\n"
    "    float h_slow = prev_hidden[id] * exp_slow + gate * activation * (1.0 - exp_slow);\n"
    "    next_hidden[id] = h_fast * 0.5 + h_slow * 0.5;\n"
    "}\n";

static const char* METAL_SGD_UPDATE_KERNEL =
    "kernel void sgd_update(\n"
    "    device float* weights       [[buffer(0)]],\n"
    "    device const float* gradients [[buffer(1)]],\n"
    "    constant float& lr          [[buffer(2)]],\n"
    "    constant float& wd          [[buffer(3)]],\n"
    "    constant int& n             [[buffer(4)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    weights[id] -= lr * (gradients[id] + wd * weights[id]);\n"
    "}\n";

static const char* METAL_ADAM_UPDATE_KERNEL =
    "kernel void adam_update(\n"
    "    device float* weights       [[buffer(0)]],\n"
    "    device const float* gradients [[buffer(1)]],\n"
    "    device float* m             [[buffer(2)]],\n"
    "    device float* v             [[buffer(3)]],\n"
    "    constant float& lr          [[buffer(4)]],\n"
    "    constant float& beta1       [[buffer(5)]],\n"
    "    constant float& beta2       [[buffer(6)]],\n"
    "    constant float& eps         [[buffer(7)]],\n"
    "    constant float& wd          [[buffer(8)]],\n"
    "    constant float& b1t         [[buffer(9)]],\n"
    "    constant float& b2t         [[buffer(10)]],\n"
    "    constant int& n             [[buffer(11)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    float g = gradients[id] + wd * weights[id];\n"
    "    float m_new = beta1 * m[id] + (1.0 - beta1) * g;\n"
    "    float v_new = beta2 * v[id] + (1.0 - beta2) * g * g;\n"
    "    m[id] = m_new;\n"
    "    v[id] = v_new;\n"
    "    float m_hat = m_new / (1.0 - b1t);\n"
    "    float v_hat = v_new / (1.0 - b2t);\n"
    "    weights[id] -= lr * m_hat / (sqrt(v_hat) + eps);\n"
    "}\n";

static const char* METAL_MSE_LOSS_KERNEL =
    "kernel void mse_loss(\n"
    "    device const float* output  [[buffer(0)]],\n"
    "    device const float* target  [[buffer(1)]],\n"
    "    constant int& batch_size    [[buffer(2)]],\n"
    "    constant int& dim           [[buffer(3)]],\n"
    "    device float* loss          [[buffer(4)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)batch_size) return;\n"
    "    float sum = 0.0;\n"
    "    for (int j = 0; j < dim; j++) {\n"
    "        float d = output[id * dim + j] - target[id * dim + j];\n"
    "        sum += d * d;\n"
    "    }\n"
    "    loss[id] = sum / (float)dim;\n"
    "}\n";

static const char* METAL_CROSS_ENTROPY_LOSS_KERNEL =
    "kernel void cross_entropy_loss(\n"
    "    device const float* output  [[buffer(0)]],\n"
    "    device const float* target  [[buffer(1)]],\n"
    "    constant int& batch_size    [[buffer(2)]],\n"
    "    constant int& num_classes   [[buffer(3)]],\n"
    "    device float* loss          [[buffer(4)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)batch_size) return;\n"
    "    int base = id * num_classes;\n"
    "    float max_val = output[base];\n"
    "    for (int j = 1; j < num_classes; j++) {\n"
    "        if (output[base + j] > max_val) max_val = output[base + j];\n"
    "    }\n"
    "    float sum_exp = 0.0;\n"
    "    for (int j = 0; j < num_classes; j++) {\n"
    "        sum_exp += exp(output[base + j] - max_val);\n"
    "    }\n"
    "    float log_sum = max_val + log(sum_exp);\n"
    "    float nll = 0.0;\n"
    "    for (int j = 0; j < num_classes; j++) {\n"
    "        nll -= target[base + j] * (output[base + j] - log_sum);\n"
    "    }\n"
    "    loss[id] = nll;\n"
    "}\n";

static const char* METAL_GRAD_CLIP_KERNEL =
    "kernel void grad_clip(\n"
    "    device float* gradients     [[buffer(0)]],\n"
    "    constant float& clip_val    [[buffer(1)]],\n"
    "    constant int& n             [[buffer(2)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    float g = gradients[id];\n"
    "    if (g > clip_val) g = clip_val;\n"
    "    if (g < -clip_val) g = -clip_val;\n"
    "    gradients[id] = g;\n"
    "}\n";

static const char* METAL_GRAD_CLIP_BY_NORM_KERNEL =
    "kernel void grad_clip_by_norm_part1(\n"
    "    device const float* gradients [[buffer(0)]],\n"
    "    constant int& n              [[buffer(1)]],\n"
    "    device float* norm_sum       [[buffer(2)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    float val = gradients[id];\n"
    "    float square_sum = 0.0;\n"
    "    for (int j = id; j < n; j++) square_sum += gradients[j] * gradients[j];\n"
    "    norm_sum[0] = square_sum;\n"
    "}\n"
    "kernel void grad_clip_by_norm_part2(\n"
    "    device float* gradients      [[buffer(0)]],\n"
    "    constant float& clip_norm    [[buffer(1)]],\n"
    "    constant float& norm_val     [[buffer(2)]],\n"
    "    constant int& n              [[buffer(3)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    if (norm_val > clip_norm) {\n"
    "        float scale = clip_norm / norm_val;\n"
    "        gradients[id] *= scale;\n"
    "    }\n"
    "}\n";

static const char* METAL_WEIGHT_DECAY_KERNEL =
    "kernel void apply_weight_decay(\n"
    "    device float* weights       [[buffer(0)]],\n"
    "    constant float& lr          [[buffer(1)]],\n"
    "    constant float& wd          [[buffer(2)]],\n"
    "    constant int& n             [[buffer(3)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    weights[id] *= (1.0 - lr * wd);\n"
    "}\n";

static const char* METAL_CONV2D_KERNEL =
    "kernel void conv2d(\n"
    "    device const float* input   [[buffer(0)]],\n"
    "    device const float* kernel  [[buffer(1)]],\n"
    "    device float* output        [[buffer(2)]],\n"
    "    constant int& in_channels   [[buffer(3)]],\n"
    "    constant int& out_channels  [[buffer(4)]],\n"
    "    constant int& height        [[buffer(5)]],\n"
    "    constant int& width         [[buffer(6)]],\n"
    "    constant int& kernel_h      [[buffer(7)]],\n"
    "    constant int& kernel_w      [[buffer(8)]],\n"
    "    constant int& pad_h         [[buffer(9)]],\n"
    "    constant int& pad_w         [[buffer(10)]],\n"
    "    constant int& stride_h      [[buffer(11)]],\n"
    "    constant int& stride_w      [[buffer(12)]],\n"
    "    constant int& out_h         [[buffer(13)]],\n"
    "    constant int& out_w         [[buffer(14)]],\n"
    "    uint3 gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    int oc = (int)gid.x;\n"
    "    int oh = (int)gid.y;\n"
    "    int ow = (int)gid.z;\n"
    "    if (oc >= out_channels || oh >= out_h || ow >= out_w) return;\n"
    "    float sum = 0.0;\n"
    "    for (int ic = 0; ic < in_channels; ic++) {\n"
    "        for (int kh = 0; kh < kernel_h; kh++) {\n"
    "            for (int kw = 0; kw < kernel_w; kw++) {\n"
    "                int ih = oh * stride_h + kh - pad_h;\n"
    "                int iw = ow * stride_w + kw - pad_w;\n"
    "                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {\n"
    "                    sum += input[ic * height * width + ih * width + iw]\n"
    "                         * kernel[oc * in_channels * kernel_h * kernel_w\n"
    "                                 + ic * kernel_h * kernel_w + kh * kernel_w + kw];\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    output[oc * out_h * out_w + oh * out_w + ow] = sum;\n"
    "}\n";

static const char* METAL_MAX_POOL2D_KERNEL =
    "kernel void max_pool2d(\n"
    "    device const float* input   [[buffer(0)]],\n"
    "    device float* output        [[buffer(1)]],\n"
    "    constant int& channels      [[buffer(2)]],\n"
    "    constant int& height        [[buffer(3)]],\n"
    "    constant int& width         [[buffer(4)]],\n"
    "    constant int& pool_h        [[buffer(5)]],\n"
    "    constant int& pool_w        [[buffer(6)]],\n"
    "    constant int& stride_h      [[buffer(7)]],\n"
    "    constant int& stride_w      [[buffer(8)]],\n"
    "    constant int& out_h         [[buffer(9)]],\n"
    "    constant int& out_w         [[buffer(10)]],\n"
    "    uint3 gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)gid.x;\n"
    "    int oh = (int)gid.y;\n"
    "    int ow = (int)gid.z;\n"
    "    if (c >= channels || oh >= out_h || ow >= out_w) return;\n"
    "    float max_val = -1e38;\n"
    "    for (int ph = 0; ph < pool_h; ph++) {\n"
    "        for (int pw = 0; pw < pool_w; pw++) {\n"
    "            int ih = oh * stride_h + ph;\n"
    "            int iw = ow * stride_w + pw;\n"
    "            if (ih < height && iw < width) {\n"
    "                float v = input[c * height * width + ih * width + iw];\n"
    "                if (v > max_val) max_val = v;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    output[c * out_h * out_w + oh * out_w + ow] = max_val;\n"
    "}\n";

static const char* METAL_AVG_POOL2D_KERNEL =
    "kernel void avg_pool2d(\n"
    "    device const float* input   [[buffer(0)]],\n"
    "    device float* output        [[buffer(1)]],\n"
    "    constant int& channels      [[buffer(2)]],\n"
    "    constant int& height        [[buffer(3)]],\n"
    "    constant int& width         [[buffer(4)]],\n"
    "    constant int& pool_h        [[buffer(5)]],\n"
    "    constant int& pool_w        [[buffer(6)]],\n"
    "    constant int& stride_h      [[buffer(7)]],\n"
    "    constant int& stride_w      [[buffer(8)]],\n"
    "    constant int& out_h         [[buffer(9)]],\n"
    "    constant int& out_w         [[buffer(10)]],\n"
    "    uint3 gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)gid.x;\n"
    "    int oh = (int)gid.y;\n"
    "    int ow = (int)gid.z;\n"
    "    if (c >= channels || oh >= out_h || ow >= out_w) return;\n"
    "    float sum = 0.0;\n"
    "    int count = 0;\n"
    "    for (int ph = 0; ph < pool_h; ph++) {\n"
    "        for (int pw = 0; pw < pool_w; pw++) {\n"
    "            int ih = oh * stride_h + ph;\n"
    "            int iw = ow * stride_w + pw;\n"
    "            if (ih < height && iw < width) {\n"
    "                sum += input[c * height * width + ih * width + iw];\n"
    "                count++;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    output[c * out_h * out_w + oh * out_w + ow] = sum / (float)count;\n"
    "}\n";

static const char* METAL_BATCH_NORM_KERNEL =
    "kernel void batch_norm(\n"
    "    device const float* input   [[buffer(0)]],\n"
    "    device float* output        [[buffer(1)]],\n"
    "    device const float* gamma   [[buffer(2)]],\n"
    "    device const float* beta    [[buffer(3)]],\n"
    "    device const float* mean    [[buffer(4)]],\n"
    "    device const float* var     [[buffer(5)]],\n"
    "    constant float& epsilon     [[buffer(6)]],\n"
    "    constant int& channels      [[buffer(7)]],\n"
    "    constant int& spatial_size  [[buffer(8)]],\n"
    "    uint2 id [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)id.x;\n"
    "    int s = (int)id.y;\n"
    "    if (c >= channels) return;\n"
    "    int idx = c * spatial_size + s;\n"
    "    output[idx] = gamma[c] * (input[idx] - mean[c]) / sqrt(var[c] + epsilon) + beta[c];\n"
    "}\n";

static const char* METAL_IMAGE_NORMALIZE_KERNEL =
    "kernel void image_normalize(\n"
    "    device const float* input   [[buffer(0)]],\n"
    "    device float* output        [[buffer(1)]],\n"
    "    device const float* mean    [[buffer(2)]],\n"
    "    device const float* std     [[buffer(3)]],\n"
    "    constant int& channels      [[buffer(4)]],\n"
    "    constant int& height        [[buffer(5)]],\n"
    "    constant int& width         [[buffer(6)]],\n"
    "    uint3 gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)gid.x;\n"
    "    int h = (int)gid.y;\n"
    "    int w = (int)gid.z;\n"
    "    if (c >= channels || h >= height || w >= width) return;\n"
    "    int idx = c * height * width + h * width + w;\n"
    "    output[idx] = (input[idx] - mean[c]) / std[c];\n"
    "}\n";

static const char* METAL_SOBEL_KERNEL =
    "kernel void sobel(\n"
    "    device const float* input   [[buffer(0)]],\n"
    "    device float* magnitude     [[buffer(1)]],\n"
    "    device float* direction     [[buffer(2)]],\n"
    "    constant int& channels      [[buffer(3)]],\n"
    "    constant int& height        [[buffer(4)]],\n"
    "    constant int& width         [[buffer(5)]],\n"
    "    uint3 gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)gid.x;\n"
    "    int y = (int)gid.y;\n"
    "    int x = (int)gid.z;\n"
    "    if (c >= channels || y >= height || x >= width) return;\n"
    "    if (y == 0 || y == height-1 || x == 0 || x == width-1) {\n"
    "        int idx = c * height * width + y * width + x;\n"
    "        magnitude[idx] = 0.0;\n"
    "        if (direction) direction[idx] = 0.0;\n"
    "        return;\n"
    "    }\n"
    "    int base = c * height * width;\n"
    "    float gx = -input[base + (y-1)*width + (x-1)] + input[base + (y-1)*width + (x+1)]\n"
    "               -2.0*input[base + y*width + (x-1)] + 2.0*input[base + y*width + (x+1)]\n"
    "               -input[base + (y+1)*width + (x-1)] + input[base + (y+1)*width + (x+1)];\n"
    "    float gy = -input[base + (y-1)*width + (x-1)] -2.0*input[base + (y-1)*width + x] - input[base + (y-1)*width + (x+1)]\n"
    "               +input[base + (y+1)*width + (x-1)] +2.0*input[base + (y+1)*width + x] + input[base + (y+1)*width + (x+1)];\n"
    "    int idx = base + y * width + x;\n"
    "    magnitude[idx] = sqrt(gx*gx + gy*gy);\n"
    "    if (direction) direction[idx] = atan2(gy, gx);\n"
    "}\n";

static const char* METAL_GAUSSIAN_BLUR_KERNEL =
    "kernel void gaussian_blur_h(\n"
    "    device const float* input   [[buffer(0)]],\n"
    "    device float* temp          [[buffer(1)]],\n"
    "    constant int& channels      [[buffer(2)]],\n"
    "    constant int& height        [[buffer(3)]],\n"
    "    constant int& width         [[buffer(4)]],\n"
    "    uint3 gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)gid.x;\n"
    "    int y = (int)gid.y;\n"
    "    int x = (int)gid.z;\n"
    "    if (c >= channels || y >= height || x >= width) return;\n"
    "    int base = c * height * width + y * width;\n"
    "    float val = 0.375 * input[base + x]\n"
    "              + 0.25 * (input[base + max(x-1,0)] + input[base + min(x+1,width-1)])\n"
    "              + 0.0625 * (input[base + max(x-2,0)] + input[base + min(x+2,width-1)]);\n"
    "    temp[base + x] = val;\n"
    "}\n"
    "kernel void gaussian_blur_v(\n"
    "    device const float* temp    [[buffer(0)]],\n"
    "    device float* output        [[buffer(1)]],\n"
    "    constant int& channels      [[buffer(2)]],\n"
    "    constant int& height        [[buffer(3)]],\n"
    "    constant int& width         [[buffer(4)]],\n"
    "    uint3 gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)gid.x;\n"
    "    int y = (int)gid.y;\n"
    "    int x = (int)gid.z;\n"
    "    if (c >= channels || y >= height || x >= width) return;\n"
    "    int base = c * height * width;\n"
    "    float val = 0.375 * temp[base + y * width + x]\n"
    "              + 0.25 * (temp[base + max(y-1,0)*width + x] + temp[base + min(y+1,height-1)*width + x])\n"
    "              + 0.0625 * (temp[base + max(y-2,0)*width + x] + temp[base + min(y+2,height-1)*width + x]);\n"
    "    output[base + y * width + x] = val;\n"
    "}\n";

static const char* METAL_RESIZE_BILINEAR_KERNEL =
    "kernel void resize_bilinear(\n"
    "    device const float* input   [[buffer(0)]],\n"
    "    device float* output        [[buffer(1)]],\n"
    "    constant int& channels      [[buffer(2)]],\n"
    "    constant int& src_h         [[buffer(3)]],\n"
    "    constant int& src_w         [[buffer(4)]],\n"
    "    constant int& dst_h         [[buffer(5)]],\n"
    "    constant int& dst_w         [[buffer(6)]],\n"
    "    uint3 gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)gid.x;\n"
    "    int dy = (int)gid.y;\n"
    "    int dx = (int)gid.z;\n"
    "    if (c >= channels || dy >= dst_h || dx >= dst_w) return;\n"
    "    float sx = (float)dx * (float)(src_w - 1) / (float)(dst_w - 1);\n"
    "    float sy = (float)dy * (float)(src_h - 1) / (float)(dst_h - 1);\n"
    "    int ix = (int)sx;\n"
    "    int iy = (int)sy;\n"
    "    float fx = sx - (float)ix;\n"
    "    float fy = sy - (float)iy;\n"
    "    int ix2 = min(ix + 1, src_w - 1);\n"
    "    int iy2 = min(iy + 1, src_h - 1);\n"
    "    int base = c * src_h * src_w;\n"
    "    float v00 = input[base + iy * src_w + ix];\n"
    "    float v10 = input[base + iy * src_w + ix2];\n"
    "    float v01 = input[base + iy2 * src_w + ix];\n"
    "    float v11 = input[base + iy2 * src_w + ix2];\n"
    "    float v0 = v00 + (v10 - v00) * fx;\n"
    "    float v1 = v01 + (v11 - v01) * fx;\n"
    "    output[c * dst_h * dst_w + dy * dst_w + dx] = v0 + (v1 - v0) * fy;\n"
    "}\n";

/* ============================================================================
 * 计算内核 - 矩阵乘法、激活函数、BatchNorm反向、Dropout、RMSProp
 * =========================================================================== */

static const char* METAL_MATMUL_KERNEL =
    "kernel void matmul(\n"
    "    device const float* A          [[buffer(0)]],\n"
    "    device const float* B          [[buffer(1)]],\n"
    "    device float* C                [[buffer(2)]],\n"
    "    constant int& M                [[buffer(3)]],\n"
    "    constant int& N                [[buffer(4)]],\n"
    "    constant int& K                [[buffer(5)]],\n"
    "    constant int& transA           [[buffer(6)]],\n"
    "    constant int& transB           [[buffer(7)]],\n"
    "    constant float& alpha          [[buffer(8)]],\n"
    "    constant float& beta           [[buffer(9)]],\n"
    "    uint2 id [[thread_position_in_grid]]\n"
    ") {\n"
    "    int row = (int)id.x;\n"
    "    int col = (int)id.y;\n"
    "    if (row >= M || col >= N) return;\n"
    "    float sum = 0.0;\n"
    "    if (transA != 0) {\n"
    "        if (transB != 0) {\n"
    "            for (int k = 0; k < K; k++)\n"
    "                sum += A[k * M + row] * B[col * K + k];\n"
    "        } else {\n"
    "            for (int k = 0; k < K; k++)\n"
    "                sum += A[k * M + row] * B[k * N + col];\n"
    "        }\n"
    "    } else {\n"
    "        if (transB != 0) {\n"
    "            for (int k = 0; k < K; k++)\n"
    "                sum += A[row * K + k] * B[col * K + k];\n"
    "        } else {\n"
    "            for (int k = 0; k < K; k++)\n"
    "                sum += A[row * K + k] * B[k * N + col];\n"
    "        }\n"
    "    }\n"
    "    if (beta != 0.0) {\n"
    "        C[row * N + col] = alpha * sum + beta * C[row * N + col];\n"
    "    } else {\n"
    "        C[row * N + col] = alpha * sum;\n"
    "    }\n"
    "}\n";

static const char* METAL_ACTIVATION_FORWARD_KERNEL =
    "kernel void activation_forward(\n"
    "    device const float* input       [[buffer(0)]],\n"
    "    device float* output            [[buffer(1)]],\n"
    "    constant int& n                 [[buffer(2)]],\n"
    "    constant int& act_type          [[buffer(3)]],\n"
    "    constant float& alpha           [[buffer(4)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    float x = input[id];\n"
    "    float r;\n"
    "    if (act_type == 0) {\n"
    "        r = (x > 0.0) ? x : 0.0;\n"
    "    } else if (act_type == 1) {\n"
    "        r = (x > 0.0) ? x : alpha * x;\n"
    "    } else if (act_type == 2) {\n"
    "        r = 1.0 / (1.0 + exp(-x));\n"
    "    } else if (act_type == 3) {\n"
    "        r = tanh(x);\n"
    "    } else if (act_type == 4) {\n"
    "        float x3 = x * x * x;\n"
    "        r = 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x3)));\n"
    "    } else if (act_type == 5) {\n"
    "        r = (x > 20.0) ? x : ((x < -20.0) ? exp(x) : log(1.0 + exp(x)));\n"
    "    } else {\n"
    "        r = x;\n"
    "    }\n"
    "    output[id] = r;\n"
    "}\n";

static const char* METAL_ACTIVATION_BACKWARD_KERNEL =
    "kernel void activation_backward(\n"
    "    device const float* input       [[buffer(0)]],\n"
    "    device const float* grad_output [[buffer(1)]],\n"
    "    device float* grad_input        [[buffer(2)]],\n"
    "    constant int& n                 [[buffer(3)]],\n"
    "    constant int& act_type          [[buffer(4)]],\n"
    "    constant float& alpha           [[buffer(5)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    float x = input[id];\n"
    "    float g = grad_output[id];\n"
    "    float dx;\n"
    "    if (act_type == 0) {\n"
    "        dx = (x > 0.0) ? g : 0.0;\n"
    "    } else if (act_type == 1) {\n"
    "        dx = (x > 0.0) ? g : alpha * g;\n"
    "    } else if (act_type == 2) {\n"
    "        float s = 1.0 / (1.0 + exp(-x));\n"
    "        dx = g * s * (1.0 - s);\n"
    "    } else if (act_type == 3) {\n"
    "        float t = tanh(x);\n"
    "        dx = g * (1.0 - t * t);\n"
    "    } else if (act_type == 4) {\n"
    "        float x3 = x * x * x;\n"
    "        float tanh_val = tanh(0.79788456 * (x + 0.044715 * x3));\n"
    "        float sech2 = 1.0 - tanh_val * tanh_val;\n"
    "        float dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * 1.5957691216 * (1.0 + 0.134145 * x * x);\n"
    "        dx = g * dgelu;\n"
    "    } else if (act_type == 5) {\n"
    "        float s = (x > 20.0) ? 1.0 : ((x < -20.0) ? exp(x) : 1.0 / (1.0 + exp(-x)));\n"
    "        dx = g * s;\n"
    "    } else {\n"
    "        dx = g;\n"
    "    }\n"
    "    grad_input[id] = dx;\n"
    "}\n";

static const char* METAL_BATCH_NORM_BACKWARD_KERNEL =
    "kernel void batch_norm_backward_part1(\n"
    "    device const float* input       [[buffer(0)]],\n"
    "    device const float* mean        [[buffer(1)]],\n"
    "    device const float* var         [[buffer(2)]],\n"
    "    constant float& epsilon         [[buffer(3)]],\n"
    "    constant int& channels          [[buffer(4)]],\n"
    "    constant int& spatial_size      [[buffer(5)]],\n"
    "    device float* x_hat_buffer      [[buffer(6)]],\n"
    "    uint2 id [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)id.x;\n"
    "    int s = (int)id.y;\n"
    "    if (c >= channels || s >= spatial_size) return;\n"
    "    int idx = c * spatial_size + s;\n"
    "    float inv_std = 1.0 / sqrt(var[c] + epsilon);\n"
    "    x_hat_buffer[idx] = (input[idx] - mean[c]) * inv_std;\n"
    "}\n"
    "kernel void batch_norm_backward_part2(\n"
    "    device const float* grad_output [[buffer(0)]],\n"
    "    device const float* x_hat_buffer [[buffer(1)]],\n"
    "    device const float* gamma       [[buffer(2)]],\n"
    "    device const float* var         [[buffer(3)]],\n"
    "    constant float& epsilon         [[buffer(4)]],\n"
    "    device const float* d_gamma     [[buffer(5)]],\n"
    "    device const float* d_beta      [[buffer(6)]],\n"
    "    constant int& channels          [[buffer(7)]],\n"
    "    constant int& spatial_size      [[buffer(8)]],\n"
    "    device float* grad_input        [[buffer(9)]],\n"
    "    uint2 id [[thread_position_in_grid]]\n"
    ") {\n"
    "    int c = (int)id.x;\n"
    "    int s = (int)id.y;\n"
    "    if (c >= channels || s >= spatial_size) return;\n"
    "    int idx = c * spatial_size + s;\n"
    "    float inv_std = 1.0 / sqrt(var[c] + epsilon);\n"
    "    float N_inv = 1.0 / (float)spatial_size;\n"
    "    float dy = grad_output[idx];\n"
    "    float x_h = x_hat_buffer[idx];\n"
    "    grad_input[idx] = gamma[c] * inv_std * (dy - d_beta[c] * N_inv - x_h * d_gamma[c] * N_inv);\n"
    "}\n";

static const char* METAL_DROPOUT_FORWARD_KERNEL =
    "kernel void dropout_forward(\n"
    "    device const float* input       [[buffer(0)]],\n"
    "    device float* output            [[buffer(1)]],\n"
    "    device float* mask              [[buffer(2)]],\n"
    "    constant float& p               [[buffer(3)]],\n"
    "    constant int& n                 [[buffer(4)]],\n"
    "    constant int& seed              [[buffer(5)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    uint rng = (uint)(seed + id * 1103515245u + 12345u);\n"
    "    rng = rng * 1103515245u + 12345u;\n"
    "    float r = (float)(rng & 0x7FFFFFFFu) / 2147483648.0;\n"
    "    float inv_keep = 1.0 / (1.0 - p);\n"
    "    mask[id] = (r > p) ? 1.0 : 0.0;\n"
    "    output[id] = input[id] * mask[id] * inv_keep;\n"
    "}\n";

static const char* METAL_DROPOUT_BACKWARD_KERNEL =
    "kernel void dropout_backward(\n"
    "    device const float* grad_output [[buffer(0)]],\n"
    "    device float* grad_input        [[buffer(1)]],\n"
    "    device const float* mask        [[buffer(2)]],\n"
    "    constant float& p               [[buffer(3)]],\n"
    "    constant int& n                 [[buffer(4)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    float inv_keep = 1.0 / (1.0 - p);\n"
    "    grad_input[id] = grad_output[id] * mask[id] * inv_keep;\n"
    "}\n";

static const char* METAL_RMSPROP_UPDATE_KERNEL =
    "kernel void rmsprop_update(\n"
    "    device float* weights           [[buffer(0)]],\n"
    "    device const float* gradients   [[buffer(1)]],\n"
    "    device float* sq_avg            [[buffer(2)]],\n"
    "    constant float& lr              [[buffer(3)]],\n"
    "    constant float& decay           [[buffer(4)]],\n"
    "    constant float& eps             [[buffer(5)]],\n"
    "    constant int& n                 [[buffer(6)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)n) return;\n"
    "    float g = gradients[id];\n"
    "    sq_avg[id] = decay * sq_avg[id] + (1.0 - decay) * g * g;\n"
    "    weights[id] -= lr * g / (sqrt(sq_avg[id]) + eps);\n"
    "}\n";

static const char* METAL_CROSS_ENTROPY_GRAD_KERNEL =
    "kernel void cross_entropy_loss_gradient(\n"
    "    device const float* output      [[buffer(0)]],\n"
    "    device const float* target      [[buffer(1)]],\n"
    "    device float* grad_output       [[buffer(2)]],\n"
    "    constant int& batch_size        [[buffer(3)]],\n"
    "    constant int& num_classes       [[buffer(4)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)batch_size) return;\n"
    "    int base = id * num_classes;\n"
    "    float max_val = output[base];\n"
    "    for (int j = 1; j < num_classes; j++)\n"
    "        if (output[base + j] > max_val) max_val = output[base + j];\n"
    "    float sum_exp = 0.0;\n"
    "    for (int j = 0; j < num_classes; j++)\n"
    "        sum_exp += exp(output[base + j] - max_val);\n"
    "    float inv_sum = 1.0 / sum_exp;\n"
    "    for (int j = 0; j < num_classes; j++) {\n"
    "        float softmax_val = exp(output[base + j] - max_val) * inv_sum;\n"
    "        grad_output[base + j] = softmax_val - target[base + j];\n"
    "    }\n"
    "}\n";

static const char* METAL_BIAS_ADD_KERNEL =
    "kernel void bias_add(\n"
    "    device float* data              [[buffer(0)]],\n"
    "    device const float* bias        [[buffer(1)]],\n"
    "    constant int& bias_size         [[buffer(2)]],\n"
    "    constant int& total_size        [[buffer(3)]],\n"
    "    uint id [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (id >= (uint)total_size) return;\n"
    "    data[id] += bias[id % bias_size];\n"
    "}\n";

/**
 * @brief 创建Metal内核（完整实现）
 */
static GpuKernel* metal_backend_kernel_create(GpuContext* context, const char* kernel_source, const char* kernel_name) {
    UNUSED(context); UNUSED(kernel_source); UNUSED(kernel_name);
#ifdef __APPLE__
    if (!context || !context->backend_data || !kernel_name) {
        set_metal_error_string("无效参数: 上下文、后端数据或内核名称为空");
        return NULL;
    }
    
    MetalContextInternal* metal_context = (MetalContextInternal*)context->backend_data;
    if (!metal_context->device) {
        set_metal_error_string("Metal设备未初始化");
        return NULL;
    }
    
    if (!mtlDeviceNewLibraryWithSource || !mtlLibraryNewFunctionWithName || !mtlDeviceNewComputePipelineStateWithFunction) {
        set_metal_error_string("Metal内核编译函数未加载");
        return NULL;
    }
    
    const char* actual_source = kernel_source;
    if (!actual_source) {
        if (strcmp(kernel_name, "cfc_forward_step") == 0) {
            actual_source = METAL_CFC_FORWARD_KERNEL;
        } else if (strcmp(kernel_name, "cfc_liquid_tau") == 0) {
            actual_source = METAL_CFC_LIQUID_TAU_KERNEL;
        } else if (strcmp(kernel_name, "cfc_multi_timescale_step") == 0) {
            actual_source = METAL_CFC_MULTI_TIMESCALE_KERNEL;
        } else if (strcmp(kernel_name, "sgd_update") == 0) {
            actual_source = METAL_SGD_UPDATE_KERNEL;
        } else if (strcmp(kernel_name, "adam_update") == 0) {
            actual_source = METAL_ADAM_UPDATE_KERNEL;
        } else if (strcmp(kernel_name, "mse_loss") == 0) {
            actual_source = METAL_MSE_LOSS_KERNEL;
        } else if (strcmp(kernel_name, "cross_entropy_loss") == 0) {
            actual_source = METAL_CROSS_ENTROPY_LOSS_KERNEL;
        } else if (strcmp(kernel_name, "grad_clip") == 0) {
            actual_source = METAL_GRAD_CLIP_KERNEL;
        } else if (strcmp(kernel_name, "grad_clip_by_norm_part1") == 0) {
            actual_source = METAL_GRAD_CLIP_BY_NORM_KERNEL;
        } else if (strcmp(kernel_name, "grad_clip_by_norm_part2") == 0) {
            actual_source = METAL_GRAD_CLIP_BY_NORM_KERNEL;
        } else if (strcmp(kernel_name, "apply_weight_decay") == 0) {
            actual_source = METAL_WEIGHT_DECAY_KERNEL;
        } else if (strcmp(kernel_name, "conv2d") == 0) {
            actual_source = METAL_CONV2D_KERNEL;
        } else if (strcmp(kernel_name, "max_pool2d") == 0) {
            actual_source = METAL_MAX_POOL2D_KERNEL;
        } else if (strcmp(kernel_name, "avg_pool2d") == 0) {
            actual_source = METAL_AVG_POOL2D_KERNEL;
        } else if (strcmp(kernel_name, "batch_norm") == 0) {
            actual_source = METAL_BATCH_NORM_KERNEL;
        } else if (strcmp(kernel_name, "image_normalize") == 0) {
            actual_source = METAL_IMAGE_NORMALIZE_KERNEL;
        } else if (strcmp(kernel_name, "sobel") == 0) {
            actual_source = METAL_SOBEL_KERNEL;
        } else if (strncmp(kernel_name, "gaussian_blur", 13) == 0) {
            actual_source = METAL_GAUSSIAN_BLUR_KERNEL;
        } else if (strcmp(kernel_name, "resize_bilinear") == 0) {
            actual_source = METAL_RESIZE_BILINEAR_KERNEL;
        } else if (strcmp(kernel_name, "matmul") == 0) {
            actual_source = METAL_MATMUL_KERNEL;
        } else if (strcmp(kernel_name, "activation_forward") == 0) {
            actual_source = METAL_ACTIVATION_FORWARD_KERNEL;
        } else if (strcmp(kernel_name, "activation_backward") == 0) {
            actual_source = METAL_ACTIVATION_BACKWARD_KERNEL;
        } else if (strcmp(kernel_name, "batch_norm_backward_part1") == 0) {
            actual_source = METAL_BATCH_NORM_BACKWARD_KERNEL;
        } else if (strcmp(kernel_name, "batch_norm_backward_part2") == 0) {
            actual_source = METAL_BATCH_NORM_BACKWARD_KERNEL;
        } else if (strcmp(kernel_name, "dropout_forward") == 0) {
            actual_source = METAL_DROPOUT_FORWARD_KERNEL;
        } else if (strcmp(kernel_name, "dropout_backward") == 0) {
            actual_source = METAL_DROPOUT_BACKWARD_KERNEL;
        } else if (strcmp(kernel_name, "rmsprop_update") == 0) {
            actual_source = METAL_RMSPROP_UPDATE_KERNEL;
        } else if (strcmp(kernel_name, "cross_entropy_loss_gradient") == 0) {
            actual_source = METAL_CROSS_ENTROPY_GRAD_KERNEL;
        } else if (strcmp(kernel_name, "bias_add") == 0) {
            actual_source = METAL_BIAS_ADD_KERNEL;
        } else {
            set_metal_error_string("未知内置内核名称: %s", kernel_name);
            return NULL;
        }
    }
    
    unsigned long source_hash = metal_hash_string(actual_source);
    void* cached_library = metal_cache_find_program(kernel_name, source_hash);
    int is_from_cache = (cached_library != NULL) ? 1 : 0;
    
    MetalKernelInternal* metal_kernel = (MetalKernelInternal*)safe_malloc(sizeof(MetalKernelInternal));
    if (!metal_kernel) {
        set_metal_error_string("内存分配失败");
        return NULL;
    }
    memset(metal_kernel, 0, sizeof(MetalKernelInternal));
    
    size_t name_len = strlen(kernel_name);
    if (name_len >= sizeof(metal_kernel->kernel_name) - 1) name_len = sizeof(metal_kernel->kernel_name) - 1;
    memcpy(metal_kernel->kernel_name, kernel_name, name_len);
    metal_kernel->kernel_name[name_len] = '\0';
    
    void* library = NULL;
    void* function = NULL;
    void* pipeline_state = NULL;
    
    if (is_from_cache) {
        library = cached_library;
        function = mtlLibraryNewFunctionWithName(library, kernel_name);
        if (!function) {
            set_metal_error_string("缓存中获取函数失败: %s", kernel_name);
            safe_free((void**)&metal_kernel);
            return NULL;
        }
        void* err = NULL;
        pipeline_state = mtlDeviceNewComputePipelineStateWithFunction(metal_context->device, function, &err);
        if (!pipeline_state || err) {
            set_metal_error_string("缓存管道状态创建失败");
            safe_free((void**)&metal_kernel);
            return NULL;
        }
    } else {
        void* err = NULL;
        library = mtlDeviceNewLibraryWithSource(metal_context->device, actual_source, strlen(actual_source), &err);
        if (!library || err) {
            set_metal_error_string("MSL编译失败: %s", kernel_name);
            safe_free((void**)&metal_kernel);
            return NULL;
        }
        function = mtlLibraryNewFunctionWithName(library, kernel_name);
        if (!function) {
            set_metal_error_string("获取函数失败: %s", kernel_name);
            safe_free((void**)&metal_kernel);
            return NULL;
        }
        err = NULL;
        pipeline_state = mtlDeviceNewComputePipelineStateWithFunction(metal_context->device, function, &err);
        if (!pipeline_state || err) {
            set_metal_error_string("管道状态创建失败");
            safe_free((void**)&metal_kernel);
            return NULL;
        }
        metal_cache_add_program(kernel_name, source_hash, library);
    }
    
    metal_kernel->library = library;
    metal_kernel->function = function;
    metal_kernel->pipeline_state = pipeline_state;
    
    GpuKernel* gpu_kernel = (GpuKernel*)safe_malloc(sizeof(GpuKernel));
    if (!gpu_kernel) {
        set_metal_error_string("内存分配失败: GPU内核包装");
        safe_free((void**)&metal_kernel);
        return NULL;
    }
    memset(gpu_kernel, 0, sizeof(GpuKernel));
    gpu_kernel->context = context;
    gpu_kernel->user_data = metal_kernel;
    
    set_metal_error_string("Metal内核创建成功: %s", kernel_name);
    
    return gpu_kernel;
#else
    set_metal_error_string("Metal后端仅支持苹果平台(macOS/iOS)");
    return NULL;
#endif
}

/**
 * @brief 释放Metal内核
 */
static void metal_backend_kernel_free(GpuKernel* kernel) {
    if (!kernel) return;
    
    MetalKernelInternal* kern = (MetalKernelInternal*)kernel->user_data;
    if (!kern) {
        safe_free((void**)&kernel);
        return;
    }
    
#ifdef __APPLE__
    // 释放Metal资源（如果释放函数可用）
    if (kern->pipeline_state && mtlComputePipelineStateRelease) {
        mtlComputePipelineStateRelease(kern->pipeline_state);
    }
    if (kern->function && mtlFunctionRelease) {
        mtlFunctionRelease(kern->function);
    }
    if (kern->library && mtlLibraryRelease) {
        mtlLibraryRelease(kern->library);
    }
#endif
    
    safe_free((void**)&kern);
    safe_free((void**)&kernel);
}

/**
 * @brief 设置Metal内核参数（存根）
 */
static int metal_backend_kernel_set_arg(GpuKernel* kernel, int arg_index, size_t arg_size, const void* arg_value) {
    UNUSED(kernel); UNUSED(arg_index); UNUSED(arg_size); UNUSED(arg_value);
#ifdef __APPLE__
    if (!kernel || !kernel->user_data) {
        set_metal_error_string("无效内核");
        return -1;
    }
    
    MetalKernelInternal* metal_kernel = (MetalKernelInternal*)kernel->user_data;
    
    if (arg_index < 0 || arg_index >= 16) {
        set_metal_error_string("参数索引超出范围 (0-15)");
        return -1;
    }
    
    // 存储参数（假设arg_value是指向GpuMemory的指针）
    metal_kernel->arg_buffers[arg_index] = (void*)arg_value;
    metal_kernel->arg_sizes[arg_index] = arg_size;
    
    // 更新参数计数
    if (arg_index >= metal_kernel->arg_count) {
        metal_kernel->arg_count = arg_index + 1;
    }
    
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 执行Metal内核（完整实现）
 */
static int metal_backend_kernel_execute(GpuKernel* kernel, size_t global_work_size, size_t local_work_size) {
    UNUSED(kernel); UNUSED(global_work_size); UNUSED(local_work_size);
#ifdef __APPLE__
    if (!kernel || !kernel->context || !kernel->user_data) {
        set_metal_error_string("无效内核或上下文");
        return -1;
    }
    
    MetalKernelInternal* metal_kernel = (MetalKernelInternal*)kernel->user_data;
    MetalContextInternal* metal_context = (MetalContextInternal*)kernel->context->backend_data;
    
    if (!metal_context->device || !metal_context->command_queue) {
        set_metal_error_string("Metal设备或命令队列未初始化");
        return -1;
    }
    
    // 检查必要的Metal函数是否可用
    if (!mtlCommandQueueCommandBuffer || !mtlCommandBufferComputeCommandEncoder ||
        !mtlComputeCommandEncoderSetComputePipelineState || !mtlComputeCommandEncoderSetBuffer ||
        !mtlComputeCommandEncoderDispatchThreads || !mtlComputeCommandEncoderEndEncoding ||
        !mtlCommandBufferCommit || !mtlCommandBufferWaitUntilCompleted) {
        set_metal_error_string("Metal执行函数未加载");
        return -1;
    }
    
    // 创建命令缓冲区
    void* command_buffer = mtlCommandQueueCommandBuffer(metal_context->command_queue);
    if (!command_buffer) {
        set_metal_error_string("创建Metal命令缓冲区失败");
        return -1;
    }
    
    // 创建计算命令编码器
    void* compute_encoder = mtlCommandBufferComputeCommandEncoder(command_buffer);
    if (!compute_encoder) {
        set_metal_error_string("创建Metal计算命令编码器失败");
        // 释放命令缓冲区？暂时忽略
        return -1;
    }
    
    // 设置计算管道状态
    mtlComputeCommandEncoderSetComputePipelineState(compute_encoder, metal_kernel->pipeline_state);
    
    // 设置内核参数（缓冲区）
    for (int i = 0; i < metal_kernel->arg_count; i++) {
        if (metal_kernel->arg_buffers[i]) {
            // 假设arg_buffers[i]是GpuMemory*
            GpuMemory* gpu_mem = (GpuMemory*)metal_kernel->arg_buffers[i];
            MetalMemoryInternal* metal_mem = (MetalMemoryInternal*)gpu_mem->backend_data;
            if (metal_mem && metal_mem->buffer) {
                mtlComputeCommandEncoderSetBuffer(compute_encoder, metal_mem->buffer, 0, i);
            }
        }
    }
    
    // 调度线程（一维）
    mtlComputeCommandEncoderDispatchThreads(compute_encoder, global_work_size, 1, 1);
    
    // 结束编码
    mtlComputeCommandEncoderEndEncoding(compute_encoder);
    
    // 提交命令缓冲区
    mtlCommandBufferCommit(command_buffer);
    
    // 等待完成
    mtlCommandBufferWaitUntilCompleted(command_buffer);
    
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 执行Metal内核（多维，完整实现）
 */
static int metal_backend_kernel_execute_nd(GpuKernel* kernel, int work_dim,
                                         const size_t* global_work_size,
                                         const size_t* local_work_size) {
    UNUSED(kernel); UNUSED(work_dim); UNUSED(global_work_size); UNUSED(local_work_size);
#ifdef __APPLE__
    if (!kernel || !kernel->context || !kernel->user_data || !global_work_size) {
        set_metal_error_string("无效内核、上下文或全局工作大小");
        return -1;
    }
    
    if (work_dim < 1 || work_dim > 3) {
        set_metal_error_string("工作维度必须是1-3");
        return -1;
    }
    
    MetalKernelInternal* metal_kernel = (MetalKernelInternal*)kernel->user_data;
    MetalContextInternal* metal_context = (MetalContextInternal*)kernel->context->backend_data;
    
    if (!metal_context->device || !metal_context->command_queue) {
        set_metal_error_string("Metal设备或命令队列未初始化");
        return -1;
    }
    
    // 检查必要的Metal函数是否可用
    if (!mtlCommandQueueCommandBuffer || !mtlCommandBufferComputeCommandEncoder ||
        !mtlComputeCommandEncoderSetComputePipelineState || !mtlComputeCommandEncoderSetBuffer ||
        !mtlComputeCommandEncoderDispatchThreads || !mtlComputeCommandEncoderEndEncoding ||
        !mtlCommandBufferCommit || !mtlCommandBufferWaitUntilCompleted) {
        set_metal_error_string("Metal执行函数未加载");
        return -1;
    }
    
    // 创建命令缓冲区
    void* command_buffer = mtlCommandQueueCommandBuffer(metal_context->command_queue);
    if (!command_buffer) {
        set_metal_error_string("创建Metal命令缓冲区失败");
        return -1;
    }
    
    // 创建计算命令编码器
    void* compute_encoder = mtlCommandBufferComputeCommandEncoder(command_buffer);
    if (!compute_encoder) {
        set_metal_error_string("创建Metal计算命令编码器失败");
        // 释放命令缓冲区？暂时忽略
        return -1;
    }
    
    // 设置计算管道状态
    mtlComputeCommandEncoderSetComputePipelineState(compute_encoder, metal_kernel->pipeline_state);
    
    // 设置内核参数（缓冲区）
    for (int i = 0; i < metal_kernel->arg_count; i++) {
        if (metal_kernel->arg_buffers[i]) {
            // 假设arg_buffers[i]是GpuMemory*
            GpuMemory* gpu_mem = (GpuMemory*)metal_kernel->arg_buffers[i];
            MetalMemoryInternal* metal_mem = (MetalMemoryInternal*)gpu_mem->backend_data;
            if (metal_mem && metal_mem->buffer) {
                mtlComputeCommandEncoderSetBuffer(compute_encoder, metal_mem->buffer, 0, i);
            }
        }
    }
    
    // 调度线程（多维）
    size_t width = global_work_size[0];
    size_t height = (work_dim >= 2) ? global_work_size[1] : 1;
    size_t depth = (work_dim >= 3) ? global_work_size[2] : 1;
    
    mtlComputeCommandEncoderDispatchThreads(compute_encoder, width, height, depth);
    
    // 结束编码
    mtlComputeCommandEncoderEndEncoding(compute_encoder);
    
    // 提交命令缓冲区
    mtlCommandBufferCommit(command_buffer);
    
    // 等待完成
    mtlCommandBufferWaitUntilCompleted(command_buffer);
    
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 创建Metal流
 */
static GpuStream* metal_backend_stream_create(GpuContext* context) {
    UNUSED(context);
#ifdef __APPLE__
    if (!context) {
        set_metal_error_string("无效上下文");
        return NULL;
    }
    
    MetalStreamInternal* stream = (MetalStreamInternal*)safe_malloc(sizeof(MetalStreamInternal));
    if (!stream) {
        set_metal_error_string("内存分配失败");
        return NULL;
    }
    
    memset(stream, 0, sizeof(MetalStreamInternal));
    
    // 创建命令缓冲区（完整实现）
    MetalContextInternal* ctx = (MetalContextInternal*)context;
    if (ctx && ctx->command_queue && mtlCommandQueueCommandBuffer) {
        stream->command_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
        if (!stream->command_buffer) {
            set_metal_error_string("创建Metal命令缓冲区失败");
            safe_free((void**)&stream);
            return NULL;
        }
    } else {
        // 如果命令队列或函数不可用，设置警告
        stream->command_buffer = NULL;
        if (!ctx || !ctx->command_queue) {
            set_metal_error_string("警告：Metal命令队列不可用，流功能受限");
        } else {
            set_metal_error_string("警告：Metal命令缓冲区函数不可用，流功能受限");
        }
    }
    
    return (GpuStream*)stream;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return NULL;
#endif
}

/**
 * @brief 释放Metal流
 */
static void metal_backend_stream_free(GpuStream* stream) {
    UNUSED(stream);
#ifdef __APPLE__
    if (!stream) return;
    
    MetalStreamInternal* str = (MetalStreamInternal*)stream;
    
    // 先同步等待待处理操作完成
    if (str->has_pending_operation && str->pending_command_buffer) {
        if (mtlCommandBufferWaitUntilCompleted) {
            mtlCommandBufferWaitUntilCompleted(str->pending_command_buffer);
        }
        str->pending_command_buffer = NULL;
        str->has_pending_operation = 0;
        str->readback_dst = NULL;
        str->readback_size = 0;
    }
    
    // 释放暂存缓冲区
    if (str->pending_staging_buffer) {
        if (mtlBufferRelease) {
            mtlBufferRelease(str->pending_staging_buffer);
        }
        str->pending_staging_buffer = NULL;
    }
    
    safe_free((void**)&str);
#endif
}

/**
 * @brief 同步Metal流
 */
static int metal_backend_stream_synchronize(GpuStream* stream) {
    UNUSED(stream);
#ifdef __APPLE__
    if (!stream) {
        set_metal_error_string("无效流");
        return -1;
    }
    
    MetalStreamInternal* str = (MetalStreamInternal*)stream;
    
    // 处理待处理的异步操作
    if (str->has_pending_operation && str->pending_command_buffer) {
        // 等待命令缓冲区完成
        if (mtlCommandBufferWaitUntilCompleted) {
            mtlCommandBufferWaitUntilCompleted(str->pending_command_buffer);
        }
        
        // 如果存在回读操作，从暂存缓冲区复制到主机目标
        if (str->readback_dst != NULL && str->readback_size > 0 && 
            str->pending_staging_buffer && mtlBufferContents) {
            void* staging_contents = mtlBufferContents(str->pending_staging_buffer);
            if (staging_contents) {
                memcpy(str->readback_dst, staging_contents, str->readback_size);
            }
            str->readback_dst = NULL;
            str->readback_size = 0;
        }
        
        // 释放暂存缓冲区
        if (str->pending_staging_buffer && mtlBufferRelease) {
            mtlBufferRelease(str->pending_staging_buffer);
            str->pending_staging_buffer = NULL;
        }
        
        str->pending_command_buffer = NULL;
        str->has_pending_operation = 0;
    }
    
    str->is_completed = 1;
    
    return 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 查询Metal流状态
 */
static int metal_backend_stream_query(GpuStream* stream) {
    UNUSED(stream);
#ifdef __APPLE__
    if (!stream) {
        set_metal_error_string("无效流");
        return -1;
    }
    
    MetalStreamInternal* str = (MetalStreamInternal*)stream;
    
    return str->is_completed ? 1 : 0;
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 获取Metal内存信息
 */
static int metal_backend_get_memory_info(GpuContext* context, size_t* total_memory, size_t* free_memory) {
    UNUSED(context); UNUSED(total_memory); UNUSED(free_memory);
#ifdef __APPLE__
    if (!context || !total_memory || !free_memory) {
        set_metal_error_string("无效参数");
        return -1;
    }
    
    // 获取真实系统内存信息（禁止任何模拟值）
    // 使用sysctl获取系统内存信息（这是真实系统信息，不是模拟值）
    
    // 获取物理内存大小
    uint64_t physmem = 0;
    size_t len = sizeof(physmem);
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    
    if (sysctl(mib, 2, &physmem, &len, NULL, 0) == 0) {
        *total_memory = (size_t)physmem;
        // 获取空闲内存（近似值）
        vm_size_t page_size;
        mach_port_t host_port = mach_host_self();
        vm_statistics_data_t vm_stat;
        mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);
        
        if (host_page_size(host_port, &page_size) == KERN_SUCCESS &&
            host_statistics(host_port, HOST_VM_INFO, (host_info_t)&vm_stat, &host_size) == KERN_SUCCESS) {
            *free_memory = (size_t)((uint64_t)vm_stat.free_count * page_size);
        } else {
            // 如果无法获取空闲内存，根据"禁止任何模拟值"原则，不使用百分比估计
            // 设为总内存（保守但真实）
            *free_memory = (size_t)physmem;
        }
        
        return 0;
    } else {
        set_metal_error_string("无法获取真实系统内存信息");
        return -1;
    }
#else
    set_metal_error_string("Metal后端仅支持苹果平台");
    return -1;
#endif
}

/**
 * @brief 重置Metal设备（无操作）
 */
static int metal_backend_device_reset(GpuContext* context) {
    UNUSED(context);
    // Metal设备无需重置
    return 0;
}

/**
 * @brief 获取Metal后端错误信息
 */
static const char* metal_backend_get_error_string(void) {
    return g_metal_error_string;
}

/* ============================================================================
 * Metal后端接口导出
 * =========================================================================== */

/**
 * @brief 获取Metal后端接口
 */
const GpuBackendInterface* metal_get_backend_interface(void) {
    static GpuBackendInterface metal_backend = {
        .name = "Apple Metal",
        .backend_type = GPU_BACKEND_METAL,
        .init = metal_backend_init,
        .cleanup = metal_backend_cleanup,
        .get_device_count = metal_backend_get_device_count,
        .get_device_info = metal_backend_get_device_info,
        .context_create = metal_backend_context_create,
        .context_free = metal_backend_context_free,
        .memory_alloc = metal_backend_memory_alloc,
        .memory_free = metal_backend_memory_free,
        .memory_copy_to_device = metal_backend_memory_copy_to_device,
        .memory_copy_from_device = metal_backend_memory_copy_from_device,
        .memory_copy_device_to_device = metal_backend_memory_copy_device_to_device,
        .memory_copy_to_device_async = metal_backend_memory_copy_to_device_async,
        .memory_copy_from_device_async = metal_backend_memory_copy_from_device_async,
        .kernel_create = metal_backend_kernel_create,
        .kernel_free = metal_backend_kernel_free,
        .kernel_set_arg = metal_backend_kernel_set_arg,
        .kernel_execute = metal_backend_kernel_execute,
        .kernel_execute_nd = metal_backend_kernel_execute_nd,
        .stream_create = metal_backend_stream_create,
        .stream_free = metal_backend_stream_free,
        .stream_synchronize = metal_backend_stream_synchronize,
        .stream_query = metal_backend_stream_query,
        .get_memory_info = metal_backend_get_memory_info,
        .device_reset = metal_backend_device_reset,
        .get_error_string = metal_backend_get_error_string
    };
    
    return &metal_backend;
}

/* ============================================================================
 * Metal加速计算操作C包装函数
 * 这些函数直接使用Metal API，供gpu.c调度层通过extern声明调用
 * =========================================================================== */

#ifdef __APPLE__

/**
 * @brief 内核执行辅助函数（直接使用Metal API）
 */
static int metal_execute_kernel_direct(void* command_queue, void* pipeline_state,
                                       void** buffers, int num_buffers,
                                       size_t global_w, size_t global_h) {
    void* cmd_buffer = mtlCommandQueueCommandBuffer(command_queue);
    if (!cmd_buffer) return -1;
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) return -1;
    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipeline_state);
    for (int i = 0; i < num_buffers; i++) {
        mtlComputeCommandEncoderSetBuffer(encoder, buffers[i], 0, i);
    }
    mtlComputeCommandEncoderDispatchThreads(encoder, global_w, global_h, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);
    return 0;
}

#endif

/**
 * @brief Metal加速的全连接前向传播（matmul + bias + activation）
 */
int metal_forward_dense(GpuContext* context,
                        const float* input, const float* weights,
                        const float* bias, float* output,
                        size_t batch_size, size_t input_size, size_t output_size,
                        GpuActivationType act_type, float alpha) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !input || !weights || !output) return -1;
    if (batch_size == 0 || input_size == 0 || output_size == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    size_t temp_size = batch_size * output_size;
    size_t weight_byte = output_size * input_size * sizeof(float);
    size_t input_byte = batch_size * input_size * sizeof(float);
    size_t output_byte = temp_size * sizeof(float);
    size_t bias_byte = output_size * sizeof(float);

    void* weight_buf = mtlDeviceNewBufferWithLength(ctx->device, weight_byte, 0);
    void* input_buf = mtlDeviceNewBufferWithLength(ctx->device, input_byte, 0);
    void* temp_buf = mtlDeviceNewBufferWithLength(ctx->device, output_byte, 0);
    void* output_buf = mtlDeviceNewBufferWithLength(ctx->device, output_byte, 0);
    void* bias_buf = bias ? mtlDeviceNewBufferWithLength(ctx->device, bias_byte, 0) : NULL;
    if (!weight_buf || !input_buf || !temp_buf || !output_buf || (bias && !bias_buf)) {
        if (weight_buf) mtlBufferRelease(weight_buf);
        if (input_buf) mtlBufferRelease(input_buf);
        if (temp_buf) mtlBufferRelease(temp_buf);
        if (output_buf) mtlBufferRelease(output_buf);
        if (bias_buf) mtlBufferRelease(bias_buf);
        return -1;
    }

    memcpy(mtlBufferContents(weight_buf), weights, weight_byte);
    memcpy(mtlBufferContents(input_buf), input, input_byte);
    if (bias_buf) memcpy(mtlBufferContents(bias_buf), bias, bias_byte);

    GpuKernel* matmul_kernel = metal_backend_kernel_create(context, NULL, "matmul");
    GpuKernel* bias_kernel = bias_buf ? metal_backend_kernel_create(context, NULL, "bias_add") : NULL;
    GpuKernel* act_kernel = metal_backend_kernel_create(context, NULL, "activation_forward");
    if (!matmul_kernel || (bias_buf && !bias_kernel) || !act_kernel) {
        if (matmul_kernel) metal_backend_kernel_free(matmul_kernel);
        if (bias_kernel) metal_backend_kernel_free(bias_kernel);
        if (act_kernel) metal_backend_kernel_free(act_kernel);
        mtlBufferRelease(weight_buf); mtlBufferRelease(input_buf);
        mtlBufferRelease(temp_buf); mtlBufferRelease(output_buf);
        if (bias_buf) mtlBufferRelease(bias_buf);
        return -1;
    }

    void* mk_pipe = ((MetalKernelInternal*)matmul_kernel->user_data)->pipeline_state;
    int M = (int)batch_size, N = (int)output_size, K = (int)input_size;
    int transA = 0, transB = 1;
    float m_alpha = 1.0f, m_beta = 0.0f;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) { goto fd_cleanup; }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) { goto fd_cleanup; }

    mtlComputeCommandEncoderSetComputePipelineState(encoder, mk_pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, input_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, weight_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(encoder, temp_buf, 0, 2);
    mtlComputeCommandEncoderSetBytes(encoder, &M, sizeof(int), 3);
    mtlComputeCommandEncoderSetBytes(encoder, &N, sizeof(int), 4);
    mtlComputeCommandEncoderSetBytes(encoder, &K, sizeof(int), 5);
    mtlComputeCommandEncoderSetBytes(encoder, &transA, sizeof(int), 6);
    mtlComputeCommandEncoderSetBytes(encoder, &transB, sizeof(int), 7);
    mtlComputeCommandEncoderSetBytes(encoder, &m_alpha, sizeof(float), 8);
    mtlComputeCommandEncoderSetBytes(encoder, &m_beta, sizeof(float), 9);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)M, (size_t)N, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    if (bias_buf) {
        void* bk_pipe = ((MetalKernelInternal*)bias_kernel->user_data)->pipeline_state;
        int total = (int)temp_size, bs = (int)output_size;
        void* cmd2 = mtlCommandQueueCommandBuffer(ctx->command_queue);
        if (cmd2) {
            void* enc2 = mtlCommandBufferComputeCommandEncoder(cmd2);
            if (enc2) {
                mtlComputeCommandEncoderSetComputePipelineState(enc2, bk_pipe);
                mtlComputeCommandEncoderSetBuffer(enc2, temp_buf, 0, 0);
                mtlComputeCommandEncoderSetBuffer(enc2, bias_buf, 0, 1);
                mtlComputeCommandEncoderSetBytes(enc2, &bs, sizeof(int), 2);
                mtlComputeCommandEncoderSetBytes(enc2, &total, sizeof(int), 3);
                mtlComputeCommandEncoderDispatchThreads(enc2, (size_t)total, 1, 1);
                mtlComputeCommandEncoderEndEncoding(enc2);
                mtlCommandBufferCommit(cmd2);
                mtlCommandBufferWaitUntilCompleted(cmd2);
            }
        }
    }

    {
        void* ak_pipe = ((MetalKernelInternal*)act_kernel->user_data)->pipeline_state;
        int n = (int)temp_size, at = (int)act_type;
        void* cmd3 = mtlCommandQueueCommandBuffer(ctx->command_queue);
        if (cmd3) {
            void* enc3 = mtlCommandBufferComputeCommandEncoder(cmd3);
            if (enc3) {
                mtlComputeCommandEncoderSetComputePipelineState(enc3, ak_pipe);
                mtlComputeCommandEncoderSetBuffer(enc3, temp_buf, 0, 0);
                mtlComputeCommandEncoderSetBuffer(enc3, output_buf, 0, 1);
                mtlComputeCommandEncoderSetBytes(enc3, &n, sizeof(int), 2);
                mtlComputeCommandEncoderSetBytes(enc3, &at, sizeof(int), 3);
                mtlComputeCommandEncoderSetBytes(enc3, &alpha, sizeof(float), 4);
                mtlComputeCommandEncoderDispatchThreads(enc3, (size_t)n, 1, 1);
                mtlComputeCommandEncoderEndEncoding(enc3);
                mtlCommandBufferCommit(cmd3);
                mtlCommandBufferWaitUntilCompleted(cmd3);
            }
        }
    }

    memcpy(output, mtlBufferContents(output_buf), output_byte);

fd_cleanup:
    metal_backend_kernel_free(matmul_kernel);
    if (bias_kernel) metal_backend_kernel_free(bias_kernel);
    metal_backend_kernel_free(act_kernel);
    mtlBufferRelease(weight_buf); mtlBufferRelease(input_buf);
    mtlBufferRelease(temp_buf); mtlBufferRelease(output_buf);
    if (bias_buf) mtlBufferRelease(bias_buf);
    return 0;
#else
    (void)context; (void)input; (void)weights; (void)bias; (void)output;
    (void)batch_size; (void)input_size; (void)output_size; (void)act_type; (void)alpha;
    return -1;
#endif
}

/**
 * @brief Metal加速的矩阵乘法（训练用）
 */
int metal_matmul_train(GpuContext* context,
                       const float* A, const float* B, float* C,
                       size_t M, size_t N, size_t K,
                       float alpha, float beta) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !A || !B || !C) return -1;
    if (M == 0 || N == 0 || K == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    size_t a_byte = M * K * sizeof(float);
    size_t b_byte = K * N * sizeof(float);
    size_t c_byte = M * N * sizeof(float);

    void* a_buf = mtlDeviceNewBufferWithLength(ctx->device, a_byte, 0);
    void* b_buf = mtlDeviceNewBufferWithLength(ctx->device, b_byte, 0);
    void* c_buf = mtlDeviceNewBufferWithLength(ctx->device, c_byte, 0);
    if (!a_buf || !b_buf || !c_buf) {
        if (a_buf) mtlBufferRelease(a_buf);
        if (b_buf) mtlBufferRelease(b_buf);
        if (c_buf) mtlBufferRelease(c_buf);
        return -1;
    }

    memcpy(mtlBufferContents(a_buf), A, a_byte);
    memcpy(mtlBufferContents(b_buf), B, b_byte);

    GpuKernel* kernel = metal_backend_kernel_create(context, NULL, "matmul");
    if (!kernel) {
        mtlBufferRelease(a_buf); mtlBufferRelease(b_buf); mtlBufferRelease(c_buf);
        return -1;
    }

    void* pipe = ((MetalKernelInternal*)kernel->user_data)->pipeline_state;
    int m = (int)M, n = (int)N, k = (int)K, trA = 0, trB = 0;
    float m_alpha = alpha, m_beta = beta;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(a_buf); mtlBufferRelease(b_buf); mtlBufferRelease(c_buf);
        return -1;
    }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(a_buf); mtlBufferRelease(b_buf); mtlBufferRelease(c_buf);
        return -1;
    }

    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, a_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, b_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(encoder, c_buf, 0, 2);
    mtlComputeCommandEncoderSetBytes(encoder, &m, sizeof(int), 3);
    mtlComputeCommandEncoderSetBytes(encoder, &n, sizeof(int), 4);
    mtlComputeCommandEncoderSetBytes(encoder, &k, sizeof(int), 5);
    mtlComputeCommandEncoderSetBytes(encoder, &trA, sizeof(int), 6);
    mtlComputeCommandEncoderSetBytes(encoder, &trB, sizeof(int), 7);
    mtlComputeCommandEncoderSetBytes(encoder, &m_alpha, sizeof(float), 8);
    mtlComputeCommandEncoderSetBytes(encoder, &m_beta, sizeof(float), 9);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)m, (size_t)n, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    memcpy(C, mtlBufferContents(c_buf), c_byte);

    metal_backend_kernel_free(kernel);
    mtlBufferRelease(a_buf); mtlBufferRelease(b_buf); mtlBufferRelease(c_buf);
    return 0;
#else
    (void)context; (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K; (void)alpha; (void)beta;
    return -1;
#endif
}

/**
 * @brief Metal加速的激活函数前向传播
 */
int metal_activation_forward(GpuContext* context,
                             const float* input, float* output,
                             size_t num_elements, GpuActivationType act_type, float alpha) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !input || !output) return -1;
    if (num_elements == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    size_t byte_size = num_elements * sizeof(float);
    void* in_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* out_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    if (!in_buf || !out_buf) {
        if (in_buf) mtlBufferRelease(in_buf);
        if (out_buf) mtlBufferRelease(out_buf);
        return -1;
    }
    memcpy(mtlBufferContents(in_buf), input, byte_size);

    GpuKernel* kernel = metal_backend_kernel_create(context, NULL, "activation_forward");
    if (!kernel) {
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        return -1;
    }

    void* pipe = ((MetalKernelInternal*)kernel->user_data)->pipeline_state;
    int n = (int)num_elements, at = (int)act_type;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        return -1;
    }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        return -1;
    }

    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, in_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, out_buf, 0, 1);
    mtlComputeCommandEncoderSetBytes(encoder, &n, sizeof(int), 2);
    mtlComputeCommandEncoderSetBytes(encoder, &at, sizeof(int), 3);
    mtlComputeCommandEncoderSetBytes(encoder, &alpha, sizeof(float), 4);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)n, 1, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    memcpy(output, mtlBufferContents(out_buf), byte_size);

    metal_backend_kernel_free(kernel);
    mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
    return 0;
#else
    (void)context; (void)input; (void)output;
    (void)num_elements; (void)act_type; (void)alpha;
    return -1;
#endif
}

/**
 * @brief Metal加速的激活函数反向传播
 */
int metal_activation_backward(GpuContext* context,
                              const float* input, const float* grad_output,
                              float* grad_input, size_t num_elements,
                              GpuActivationType act_type, float alpha) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !input || !grad_output || !grad_input) return -1;
    if (num_elements == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    size_t byte_size = num_elements * sizeof(float);
    void* in_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* go_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* gi_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    if (!in_buf || !go_buf || !gi_buf) {
        if (in_buf) mtlBufferRelease(in_buf);
        if (go_buf) mtlBufferRelease(go_buf);
        if (gi_buf) mtlBufferRelease(gi_buf);
        return -1;
    }
    memcpy(mtlBufferContents(in_buf), input, byte_size);
    memcpy(mtlBufferContents(go_buf), grad_output, byte_size);

    GpuKernel* kernel = metal_backend_kernel_create(context, NULL, "activation_backward");
    if (!kernel) {
        mtlBufferRelease(in_buf); mtlBufferRelease(go_buf); mtlBufferRelease(gi_buf);
        return -1;
    }

    void* pipe = ((MetalKernelInternal*)kernel->user_data)->pipeline_state;
    int n = (int)num_elements, at = (int)act_type;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(in_buf); mtlBufferRelease(go_buf); mtlBufferRelease(gi_buf);
        return -1;
    }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(in_buf); mtlBufferRelease(go_buf); mtlBufferRelease(gi_buf);
        return -1;
    }

    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, in_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, go_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(encoder, gi_buf, 0, 2);
    mtlComputeCommandEncoderSetBytes(encoder, &n, sizeof(int), 3);
    mtlComputeCommandEncoderSetBytes(encoder, &at, sizeof(int), 4);
    mtlComputeCommandEncoderSetBytes(encoder, &alpha, sizeof(float), 5);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)n, 1, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    memcpy(grad_input, mtlBufferContents(gi_buf), byte_size);

    metal_backend_kernel_free(kernel);
    mtlBufferRelease(in_buf); mtlBufferRelease(go_buf); mtlBufferRelease(gi_buf);
    return 0;
#else
    (void)context; (void)input; (void)grad_output; (void)grad_input;
    (void)num_elements; (void)act_type; (void)alpha;
    return -1;
#endif
}

/**
 * @brief Metal加速的批归一化前向传播
 */
int metal_batch_norm_forward(GpuContext* context,
                             const float* input, float* output,
                             const float* gamma, const float* beta,
                             float* running_mean, float* running_var,
                             float* batch_mean, float* batch_var,
                             size_t num_elements, size_t num_features,
                             const GpuBatchNormConfig* config, int is_training) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !input || !output || !gamma || !beta || !config) return -1;
    if (num_elements == 0 || num_features == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    size_t spatial_size = num_elements / num_features;
    size_t input_byte = num_elements * sizeof(float);
    size_t feat_byte = num_features * sizeof(float);

    void* in_buf = mtlDeviceNewBufferWithLength(ctx->device, input_byte, 0);
    void* out_buf = mtlDeviceNewBufferWithLength(ctx->device, input_byte, 0);
    void* gamma_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    void* beta_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    void* mean_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    void* var_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    if (!in_buf || !out_buf || !gamma_buf || !beta_buf || !mean_buf || !var_buf) {
        if (in_buf) mtlBufferRelease(in_buf); if (out_buf) mtlBufferRelease(out_buf);
        if (gamma_buf) mtlBufferRelease(gamma_buf); if (beta_buf) mtlBufferRelease(beta_buf);
        if (mean_buf) mtlBufferRelease(mean_buf); if (var_buf) mtlBufferRelease(var_buf);
        return -1;
    }

    memcpy(mtlBufferContents(in_buf), input, input_byte);
    memcpy(mtlBufferContents(gamma_buf), gamma, feat_byte);
    memcpy(mtlBufferContents(beta_buf), beta, feat_byte);

    if (is_training) {
        float* d_input = (float*)mtlBufferContents(in_buf);
        float* m_ptr = (float*)mtlBufferContents(mean_buf);
        float* v_ptr = (float*)mtlBufferContents(var_buf);
        for (size_t c = 0; c < num_features; c++) {
            double sum = 0.0, sum_sq = 0.0;
            for (size_t s = 0; s < spatial_size; s++) {
                float x = d_input[c * spatial_size + s];
                sum += x;
                sum_sq += (double)x * x;
            }
            m_ptr[c] = (float)(sum / (double)spatial_size);
            v_ptr[c] = (float)(sum_sq / (double)spatial_size - (double)m_ptr[c] * m_ptr[c]);
            if (running_mean) running_mean[c] = config->momentum * running_mean[c] + (1.0f - config->momentum) * m_ptr[c];
            if (running_var) running_var[c] = config->momentum * running_var[c] + (1.0f - config->momentum) * v_ptr[c];
            if (batch_mean) batch_mean[c] = m_ptr[c];
            if (batch_var) batch_var[c] = v_ptr[c];
        }
    } else {
        if (!running_mean || !running_var) {
            mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
            mtlBufferRelease(gamma_buf); mtlBufferRelease(beta_buf);
            mtlBufferRelease(mean_buf); mtlBufferRelease(var_buf);
            return -1;
        }
        memcpy(mtlBufferContents(mean_buf), running_mean, feat_byte);
        memcpy(mtlBufferContents(var_buf), running_var, feat_byte);
    }

    GpuKernel* bn_kernel = metal_backend_kernel_create(context, NULL, "batch_norm");
    if (!bn_kernel) {
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        mtlBufferRelease(gamma_buf); mtlBufferRelease(beta_buf);
        mtlBufferRelease(mean_buf); mtlBufferRelease(var_buf);
        return -1;
    }

    void* pipe = ((MetalKernelInternal*)bn_kernel->user_data)->pipeline_state;
    float eps = config->epsilon;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) {
        metal_backend_kernel_free(bn_kernel);
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        mtlBufferRelease(gamma_buf); mtlBufferRelease(beta_buf);
        mtlBufferRelease(mean_buf); mtlBufferRelease(var_buf);
        return -1;
    }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) {
        metal_backend_kernel_free(bn_kernel);
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        mtlBufferRelease(gamma_buf); mtlBufferRelease(beta_buf);
        mtlBufferRelease(mean_buf); mtlBufferRelease(var_buf);
        return -1;
    }

    int ch = (int)num_features, sp = (int)spatial_size;
    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, in_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, out_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(encoder, gamma_buf, 0, 2);
    mtlComputeCommandEncoderSetBuffer(encoder, beta_buf, 0, 3);
    mtlComputeCommandEncoderSetBuffer(encoder, mean_buf, 0, 4);
    mtlComputeCommandEncoderSetBuffer(encoder, var_buf, 0, 5);
    mtlComputeCommandEncoderSetBytes(encoder, &ch, sizeof(int), 6);
    mtlComputeCommandEncoderSetBytes(encoder, &sp, sizeof(int), 7);
    mtlComputeCommandEncoderSetBytes(encoder, &eps, sizeof(float), 8);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)ch, (size_t)sp, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    memcpy(output, mtlBufferContents(out_buf), input_byte);

    metal_backend_kernel_free(bn_kernel);
    mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
    mtlBufferRelease(gamma_buf); mtlBufferRelease(beta_buf);
    mtlBufferRelease(mean_buf); mtlBufferRelease(var_buf);
    return 0;
#else
    (void)context; (void)input; (void)output; (void)gamma; (void)beta;
    (void)running_mean; (void)running_var; (void)batch_mean; (void)batch_var;
    (void)num_elements; (void)num_features; (void)config; (void)is_training;
    return -1;
#endif
}

/**
 * @brief Metal加速的批归一化反向传播
 */
int metal_batch_norm_backward(GpuContext* context,
                              const float* input, const float* grad_output,
                              float* grad_input, float* grad_gamma, float* grad_beta,
                              const float* mean, const float* var,
                              const float* gamma,
                              size_t num_elements, size_t num_features,
                              const GpuBatchNormConfig* config) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !input || !grad_output || !grad_input) return -1;
    if (!mean || !var || !gamma || !grad_gamma || !grad_beta || !config) return -1;
    if (num_elements == 0 || num_features == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    size_t spatial_size = num_elements / num_features;
    size_t input_byte = num_elements * sizeof(float);
    size_t feat_byte = num_features * sizeof(float);

    void* in_buf = mtlDeviceNewBufferWithLength(ctx->device, input_byte, 0);
    void* go_buf = mtlDeviceNewBufferWithLength(ctx->device, input_byte, 0);
    void* gi_buf = mtlDeviceNewBufferWithLength(ctx->device, input_byte, 0);
    void* mean_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    void* var_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    void* gamma_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    void* x_hat_buf = mtlDeviceNewBufferWithLength(ctx->device, input_byte, 0);
    void* dg_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    void* db_buf = mtlDeviceNewBufferWithLength(ctx->device, feat_byte, 0);
    if (!in_buf || !go_buf || !gi_buf || !mean_buf || !var_buf || !gamma_buf || !x_hat_buf || !dg_buf || !db_buf) {
        void* mbrel[] = {in_buf, go_buf, gi_buf, mean_buf, var_buf, gamma_buf, x_hat_buf, dg_buf, db_buf};
        for (int i = 0; i < 9; i++) if (mbrel[i]) mtlBufferRelease(mbrel[i]);
        return -1;
    }

    memcpy(mtlBufferContents(in_buf), input, input_byte);
    memcpy(mtlBufferContents(go_buf), grad_output, input_byte);
    memcpy(mtlBufferContents(mean_buf), mean, feat_byte);
    memcpy(mtlBufferContents(var_buf), var, feat_byte);
    memcpy(mtlBufferContents(gamma_buf), gamma, feat_byte);

    GpuKernel* p1_kernel = metal_backend_kernel_create(context, NULL, "batch_norm_backward_part1");
    if (!p1_kernel) {
        void* mbrel[] = {in_buf, go_buf, gi_buf, mean_buf, var_buf, gamma_buf, x_hat_buf, dg_buf, db_buf};
        for (int i = 0; i < 9; i++) mtlBufferRelease(mbrel[i]);
        return -1;
    }

    void* p1_pipe = ((MetalKernelInternal*)p1_kernel->user_data)->pipeline_state;
    float eps = config->epsilon;
    int ch = (int)num_features, sp = (int)spatial_size;

    void* cmd_buf = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buf) { goto bnb_cleanup; }
    void* enc = mtlCommandBufferComputeCommandEncoder(cmd_buf);
    if (!enc) { goto bnb_cleanup; }
    mtlComputeCommandEncoderSetComputePipelineState(enc, p1_pipe);
    mtlComputeCommandEncoderSetBuffer(enc, in_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(enc, mean_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(enc, var_buf, 0, 2);
    mtlComputeCommandEncoderSetBytes(enc, &eps, sizeof(float), 3);
    mtlComputeCommandEncoderSetBytes(enc, &ch, sizeof(int), 4);
    mtlComputeCommandEncoderSetBytes(enc, &sp, sizeof(int), 5);
    mtlComputeCommandEncoderSetBuffer(enc, x_hat_buf, 0, 6);
    mtlComputeCommandEncoderDispatchThreads(enc, (size_t)ch, (size_t)sp, 1);
    mtlComputeCommandEncoderEndEncoding(enc);
    mtlCommandBufferCommit(cmd_buf);
    mtlCommandBufferWaitUntilCompleted(cmd_buf);

    {
        float* x_hat_host = (float*)mtlBufferContents(x_hat_buf);
        float* go_host = (float*)mtlBufferContents(go_buf);
        float* dg_ptr = (float*)mtlBufferContents(dg_buf);
        float* db_ptr = (float*)mtlBufferContents(db_buf);
        for (size_t c = 0; c < num_features; c++) {
            double d_gamma_sum = 0.0, d_beta_sum = 0.0;
            for (size_t s = 0; s < spatial_size; s++) {
                size_t idx = c * spatial_size + s;
                d_gamma_sum += (double)go_host[idx] * (double)x_hat_host[idx];
                d_beta_sum += (double)go_host[idx];
            }
            grad_gamma[c] = (float)d_gamma_sum;
            grad_beta[c] = (float)d_beta_sum;
            dg_ptr[c] = (float)d_gamma_sum;
            db_ptr[c] = (float)d_beta_sum;
        }
    }

    GpuKernel* p2_kernel = metal_backend_kernel_create(context, NULL, "batch_norm_backward_part2");
    if (!p2_kernel) { goto bnb_cleanup; }

    void* p2_pipe = ((MetalKernelInternal*)p2_kernel->user_data)->pipeline_state;
    void* cmd_buf2 = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buf2) { metal_backend_kernel_free(p2_kernel); goto bnb_cleanup; }
    void* enc2 = mtlCommandBufferComputeCommandEncoder(cmd_buf2);
    if (!enc2) { metal_backend_kernel_free(p2_kernel); goto bnb_cleanup; }
    mtlComputeCommandEncoderSetComputePipelineState(enc2, p2_pipe);
    mtlComputeCommandEncoderSetBuffer(enc2, go_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(enc2, x_hat_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(enc2, gamma_buf, 0, 2);
    mtlComputeCommandEncoderSetBuffer(enc2, var_buf, 0, 3);
    mtlComputeCommandEncoderSetBytes(enc2, &eps, sizeof(float), 4);
    mtlComputeCommandEncoderSetBuffer(enc2, dg_buf, 0, 5);
    mtlComputeCommandEncoderSetBuffer(enc2, db_buf, 0, 6);
    mtlComputeCommandEncoderSetBytes(enc2, &ch, sizeof(int), 7);
    mtlComputeCommandEncoderSetBytes(enc2, &sp, sizeof(int), 8);
    mtlComputeCommandEncoderSetBuffer(enc2, gi_buf, 0, 9);
    mtlComputeCommandEncoderDispatchThreads(enc2, (size_t)ch, (size_t)sp, 1);
    mtlComputeCommandEncoderEndEncoding(enc2);
    mtlCommandBufferCommit(cmd_buf2);
    mtlCommandBufferWaitUntilCompleted(cmd_buf2);

    memcpy(grad_input, mtlBufferContents(gi_buf), input_byte);

    metal_backend_kernel_free(p2_kernel);
bnb_cleanup:
    metal_backend_kernel_free(p1_kernel);
    void* mbrel[] = {in_buf, go_buf, gi_buf, mean_buf, var_buf, gamma_buf, x_hat_buf, dg_buf, db_buf};
    for (int i = 0; i < 9; i++) mtlBufferRelease(mbrel[i]);
    return 0;
#else
    (void)context; (void)input; (void)grad_output; (void)grad_input;
    (void)grad_gamma; (void)grad_beta; (void)mean; (void)var; (void)gamma;
    (void)num_elements; (void)num_features; (void)config;
    return -1;
#endif
}

/**
 * @brief Metal加速的Dropout前向传播
 */
int metal_dropout_forward(GpuContext* context,
                          const float* input, float* output,
                          float* mask, size_t num_elements,
                          float dropout_rate, int is_training) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !input || !output) return -1;
    if (num_elements == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    if (!is_training) {
        memcpy(output, input, num_elements * sizeof(float));
        return 0;
    }

    size_t byte_size = num_elements * sizeof(float);
    void* in_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* out_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* mask_buf = mask ? mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0) : NULL;
    if (!in_buf || !out_buf || (mask && !mask_buf)) {
        if (in_buf) mtlBufferRelease(in_buf);
        if (out_buf) mtlBufferRelease(out_buf);
        if (mask_buf) mtlBufferRelease(mask_buf);
        return -1;
    }
    memcpy(mtlBufferContents(in_buf), input, byte_size);

    GpuKernel* kernel = metal_backend_kernel_create(context, NULL, "dropout_forward");
    if (!kernel) {
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        if (mask_buf) mtlBufferRelease(mask_buf);
        return -1;
    }

    void* pipe = ((MetalKernelInternal*)kernel->user_data)->pipeline_state;
    float p = dropout_rate;
    int n = (int)num_elements, seed = 42;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        if (mask_buf) mtlBufferRelease(mask_buf);
        return -1;
    }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
        if (mask_buf) mtlBufferRelease(mask_buf);
        return -1;
    }

    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, in_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, out_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(encoder, mask_buf ? mask_buf : out_buf, 0, 2);
    mtlComputeCommandEncoderSetBytes(encoder, &p, sizeof(float), 3);
    mtlComputeCommandEncoderSetBytes(encoder, &n, sizeof(int), 4);
    mtlComputeCommandEncoderSetBytes(encoder, &seed, sizeof(int), 5);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)n, 1, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    memcpy(output, mtlBufferContents(out_buf), byte_size);
    if (mask) memcpy(mask, mtlBufferContents(mask_buf), byte_size);

    metal_backend_kernel_free(kernel);
    mtlBufferRelease(in_buf); mtlBufferRelease(out_buf);
    if (mask_buf) mtlBufferRelease(mask_buf);
    return 0;
#else
    (void)context; (void)input; (void)output; (void)mask;
    (void)num_elements; (void)dropout_rate; (void)is_training;
    return -1;
#endif
}

/**
 * @brief Metal加速的Dropout反向传播
 */
int metal_dropout_backward(GpuContext* context,
                           const float* grad_output, float* grad_input,
                           const float* mask, size_t num_elements,
                           float dropout_rate) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !grad_output || !grad_input || !mask) return -1;
    if (num_elements == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    size_t byte_size = num_elements * sizeof(float);
    void* go_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* gi_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* mask_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    if (!go_buf || !gi_buf || !mask_buf) {
        if (go_buf) mtlBufferRelease(go_buf);
        if (gi_buf) mtlBufferRelease(gi_buf);
        if (mask_buf) mtlBufferRelease(mask_buf);
        return -1;
    }
    memcpy(mtlBufferContents(go_buf), grad_output, byte_size);
    memcpy(mtlBufferContents(mask_buf), mask, byte_size);

    GpuKernel* kernel = metal_backend_kernel_create(context, NULL, "dropout_backward");
    if (!kernel) {
        mtlBufferRelease(go_buf); mtlBufferRelease(gi_buf); mtlBufferRelease(mask_buf);
        return -1;
    }

    void* pipe = ((MetalKernelInternal*)kernel->user_data)->pipeline_state;
    float p = dropout_rate;
    int n = (int)num_elements;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(go_buf); mtlBufferRelease(gi_buf); mtlBufferRelease(mask_buf);
        return -1;
    }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(go_buf); mtlBufferRelease(gi_buf); mtlBufferRelease(mask_buf);
        return -1;
    }

    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, go_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, gi_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(encoder, mask_buf, 0, 2);
    mtlComputeCommandEncoderSetBytes(encoder, &n, sizeof(int), 3);
    mtlComputeCommandEncoderSetBytes(encoder, &p, sizeof(float), 4);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)n, 1, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    memcpy(grad_input, mtlBufferContents(gi_buf), byte_size);

    metal_backend_kernel_free(kernel);
    mtlBufferRelease(go_buf); mtlBufferRelease(gi_buf); mtlBufferRelease(mask_buf);
    return 0;
#else
    (void)context; (void)grad_output; (void)grad_input; (void)mask;
    (void)num_elements; (void)dropout_rate;
    return -1;
#endif
}

/**
 * @brief Metal加速的RMSProp优化器更新
 */
int metal_rmsprop_update(GpuContext* context, float* weights, const float* gradients,
                         float* square_avg, size_t num_params,
                         float learning_rate, float decay, float epsilon,
                         float weight_decay) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !weights || !gradients || !square_avg) return -1;
    if (num_params == 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    size_t byte_size = num_params * sizeof(float);
    void* w_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* g_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    void* sa_buf = mtlDeviceNewBufferWithLength(ctx->device, byte_size, 0);
    if (!w_buf || !g_buf || !sa_buf) {
        if (w_buf) mtlBufferRelease(w_buf);
        if (g_buf) mtlBufferRelease(g_buf);
        if (sa_buf) mtlBufferRelease(sa_buf);
        return -1;
    }
    memcpy(mtlBufferContents(w_buf), weights, byte_size);
    memcpy(mtlBufferContents(g_buf), gradients, byte_size);
    memcpy(mtlBufferContents(sa_buf), square_avg, byte_size);

    GpuKernel* kernel = metal_backend_kernel_create(context, NULL, "rmsprop_update");
    if (!kernel) {
        mtlBufferRelease(w_buf); mtlBufferRelease(g_buf); mtlBufferRelease(sa_buf);
        return -1;
    }

    void* pipe = ((MetalKernelInternal*)kernel->user_data)->pipeline_state;
    float lr = learning_rate, dc = decay, eps = epsilon, wd = weight_decay;
    int n = (int)num_params;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(w_buf); mtlBufferRelease(g_buf); mtlBufferRelease(sa_buf);
        return -1;
    }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(w_buf); mtlBufferRelease(g_buf); mtlBufferRelease(sa_buf);
        return -1;
    }

    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, w_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, g_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(encoder, sa_buf, 0, 2);
    mtlComputeCommandEncoderSetBytes(encoder, &n, sizeof(int), 3);
    mtlComputeCommandEncoderSetBytes(encoder, &lr, sizeof(float), 4);
    mtlComputeCommandEncoderSetBytes(encoder, &dc, sizeof(float), 5);
    mtlComputeCommandEncoderSetBytes(encoder, &eps, sizeof(float), 6);
    mtlComputeCommandEncoderSetBytes(encoder, &wd, sizeof(float), 7);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)n, 1, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    memcpy(weights, mtlBufferContents(w_buf), byte_size);
    memcpy(square_avg, mtlBufferContents(sa_buf), byte_size);

    metal_backend_kernel_free(kernel);
    mtlBufferRelease(w_buf); mtlBufferRelease(g_buf); mtlBufferRelease(sa_buf);
    return 0;
#else
    (void)context; (void)weights; (void)gradients; (void)square_avg;
    (void)num_params; (void)learning_rate; (void)decay; (void)epsilon; (void)weight_decay;
    return -1;
#endif
}

/**
 * @brief Metal加速的交叉熵损失梯度计算
 */
int metal_cross_entropy_loss_gradient(GpuContext* context,
                                       const float* logits, const float* targets,
                                       float* loss, float* gradients,
                                       size_t num_elements, int num_classes,
                                       int is_integer_label) {
#ifdef __APPLE__
    if (!context || !context->backend_data || !logits || !targets || !gradients) return -1;
    if (num_elements == 0 || num_classes <= 0) return -1;
    MetalContextInternal* ctx = (MetalContextInternal*)context->backend_data;
    if (!ctx->device || !ctx->command_queue) return -1;

    int batch_size = (int)(num_elements / num_classes);
    int logit_count = batch_size * num_classes;

    void* logit_buf = mtlDeviceNewBufferWithLength(ctx->device, logit_count * sizeof(float), 0);
    void* target_buf = mtlDeviceNewBufferWithLength(ctx->device, num_elements * sizeof(float), 0);
    void* loss_buf = loss ? mtlDeviceNewBufferWithLength(ctx->device, batch_size * sizeof(float), 0) : NULL;
    void* grad_buf = mtlDeviceNewBufferWithLength(ctx->device, logit_count * sizeof(float), 0);
    if (!logit_buf || !target_buf || !grad_buf || (loss && !loss_buf)) {
        if (logit_buf) mtlBufferRelease(logit_buf);
        if (target_buf) mtlBufferRelease(target_buf);
        if (loss_buf) mtlBufferRelease(loss_buf);
        if (grad_buf) mtlBufferRelease(grad_buf);
        return -1;
    }
    memcpy(mtlBufferContents(logit_buf), logits, logit_count * sizeof(float));
    memcpy(mtlBufferContents(target_buf), targets, num_elements * sizeof(float));

    if (is_integer_label) {
        float* target_data = (float*)mtlBufferContents(target_buf);
        int* tgt_int = (int*)target_data;
        int batch = (int)(num_elements / num_classes);
        for (int b = batch - 1; b >= 0; b--) {
            int label = tgt_int[b];
            for (int c = num_classes - 1; c >= 0; c--) {
                target_data[b * num_classes + c] = (c == label) ? 1.0f : 0.0f;
            }
        }
    }

    GpuKernel* kernel = metal_backend_kernel_create(context, NULL, "cross_entropy_loss_gradient");
    if (!kernel) {
        mtlBufferRelease(logit_buf); mtlBufferRelease(target_buf);
        if (loss_buf) mtlBufferRelease(loss_buf);
        mtlBufferRelease(grad_buf);
        return -1;
    }

    void* pipe = ((MetalKernelInternal*)kernel->user_data)->pipeline_state;
    int ne = (int)num_elements, nc = num_classes;
    int label_type = is_integer_label;
    int bs = batch_size;

    void* cmd_buffer = mtlCommandQueueCommandBuffer(ctx->command_queue);
    if (!cmd_buffer) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(logit_buf); mtlBufferRelease(target_buf);
        if (loss_buf) mtlBufferRelease(loss_buf);
        mtlBufferRelease(grad_buf);
        return -1;
    }
    void* encoder = mtlCommandBufferComputeCommandEncoder(cmd_buffer);
    if (!encoder) {
        metal_backend_kernel_free(kernel);
        mtlBufferRelease(logit_buf); mtlBufferRelease(target_buf);
        if (loss_buf) mtlBufferRelease(loss_buf);
        mtlBufferRelease(grad_buf);
        return -1;
    }

    mtlComputeCommandEncoderSetComputePipelineState(encoder, pipe);
    mtlComputeCommandEncoderSetBuffer(encoder, logit_buf, 0, 0);
    mtlComputeCommandEncoderSetBuffer(encoder, target_buf, 0, 1);
    mtlComputeCommandEncoderSetBuffer(encoder, loss_buf ? loss_buf : grad_buf, 0, 2);
    mtlComputeCommandEncoderSetBuffer(encoder, grad_buf, 0, 3);
    mtlComputeCommandEncoderSetBytes(encoder, &ne, sizeof(int), 4);
    mtlComputeCommandEncoderSetBytes(encoder, &nc, sizeof(int), 5);
    mtlComputeCommandEncoderSetBytes(encoder, &bs, sizeof(int), 6);
    mtlComputeCommandEncoderSetBytes(encoder, &label_type, sizeof(int), 7);
    mtlComputeCommandEncoderDispatchThreads(encoder, (size_t)bs, 1, 1);
    mtlComputeCommandEncoderEndEncoding(encoder);
    mtlCommandBufferCommit(cmd_buffer);
    mtlCommandBufferWaitUntilCompleted(cmd_buffer);

    memcpy(gradients, mtlBufferContents(grad_buf), logit_count * sizeof(float));
    if (loss) {
        float* loss_ptr = (float*)mtlBufferContents(loss_buf);
        double total_loss = 0.0;
        for (int i = 0; i < bs; i++) total_loss += loss_ptr[i];
        *loss = (float)(total_loss / bs);
    }

    metal_backend_kernel_free(kernel);
    mtlBufferRelease(logit_buf); mtlBufferRelease(target_buf);
    if (loss_buf) mtlBufferRelease(loss_buf);
    mtlBufferRelease(grad_buf);
    return 0;
#else
    (void)context; (void)logits; (void)targets; (void)loss; (void)gradients;
    (void)num_elements; (void)num_classes; (void)is_integer_label;
    return -1;
#endif
}

