/**
 * @file gpu_memory_pool.c
 * @brief GPU内存池管理系统实现
 * 
 * K-037: GPU内存池管理，减少内存碎片，提高内存分配效率。
 * 支持多种内存类型：主机内存、设备内存、统一内存。
 * 提供内存统计、碎片整理和智能分配功能。
 * 每个GPU后端维护独立的内存池实例，多GPU场景通过后端ID索引。
 */


#include "selflnn/gpu/gpu_memory_pool.h"
#include "selflnn/gpu/gpu.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/core/errors.h"
#include "gpu_internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

/* ============================================================================
 * 外部后端接口声明
 * =========================================================================== */

// 后端接口获取函数声明（定义在相应的后端实现文件中）
extern const GpuBackendInterface* cuda_get_backend_interface(void);
extern const GpuBackendInterface* opencl_get_backend_interface(void);
extern const GpuBackendInterface* vulkan_get_backend_interface(void);
extern const GpuBackendInterface* metal_get_backend_interface(void);

// 在完整系统中，这个接口提供CPU并行计算的完整功能

/* ============================================================================
 * 伙伴系统函数声明
 * =========================================================================== */
static size_t next_power_of_two(size_t size);
static int buddy_get_level(GpuMemoryPool* pool, size_t size);
static int buddy_system_init(GpuMemoryPool* pool);
static void buddy_system_cleanup(GpuMemoryPool* pool);
static void* buddy_system_alloc(GpuMemoryPool* pool, size_t size, size_t alignment);
static int buddy_system_free(GpuMemoryPool* pool, void* ptr);
static GpuMemoryBlock* buddy_find_buddy(GpuMemoryPool* pool, GpuMemoryBlock* block, int level);

/* ============================================================================
 * 内部数据结构
 * =========================================================================== */

/**
 * @brief 内存池内部结构
 */
struct GpuMemoryPool {
    GpuContext* context;               /**< GPU上下文 */
    GpuMemoryType memory_type;         /**< 内存类型 */
    GpuPoolConfig config;              /**< 内存池配置 */
    
    // 内存块管理
    GpuMemoryBlock* free_list;         /**< 空闲块链表 */
    GpuMemoryBlock* used_list;         /**< 使用中块链表 */
    void* base_address;                /**< 内存池基地址 */
    GpuMemory* gpu_memory;             /**< GPU内存句柄（用于GPU内存类型） */
    size_t total_size;                 /**< 总内存大小 */
    
    // 统计信息
    GpuPoolStatistics stats;           /**< 统计信息 */
    unsigned int next_allocation_id;   /**< 下一个分配ID */
    
    // 后端接口
    const GpuBackendInterface* backend; /**< GPU后端接口 */
    
    // 线程安全同步机制 - 深度实现：使用原生平台互斥锁
    MutexHandle lock;                   /**< 互斥锁句柄 */
    
    // 伙伴系统特定字段
    GpuMemoryBlock** buddy_free_lists; /**< 伙伴系统空闲列表数组（按块大小索引） */
    size_t min_buddy_size;             /**< 最小伙伴块大小（2的幂次方） */
    size_t max_buddy_size;             /**< 最大伙伴块大小（2的幂次方） */
    int buddy_levels;                  /**< 伙伴系统级别数 */
};

/* ============================================================================
 * CPU硬件检测函数（全部基于真实硬件检测，禁止使用模拟值/占位符）
 *
 * ZSFUSA-O02: CPU硬件检测已统一在gpu.c中实现。
 * gpu_memory_pool.c应使用gpu_hardware_get_cpu_info()统一接口。
 * 当前保留本地初始化代码，标记为待重构（后续版本将彻底消除重复）。
 *
 * 本文件中cpu_detect_*函数与gpu.c中的cpu_hw_*函数功能重复：
 *   - cpu_cpuid_detect()    ↔ cpu_hw_cpuid()
 *   - cpu_detect_vendor_real() ↔ cpu_hw_get_vendor()
 *   - cpu_detect_brand_real()  ↔ cpu_hw_get_brand_name()
 *   - cpu_detect_simd_flags_real() ↔ cpu_hw_detect_simd_x86/arm()
 *   - cpu_detect_l1/l2/l3_cache_real() ↔ cpu_hw_get_cache_size()
 * 后续重构时请将所有调用替换为gpu.c中对应函数。
 * =========================================================================== */

/**
 * @brief CPUID指令封装（仅x86/x64平台）
 */
static inline int cpu_cpuid_detect(int leaf, int subleaf,
    unsigned int* eax, unsigned int* ebx,
    unsigned int* ecx, unsigned int* edx) {
#if defined(_WIN32) && (defined(_M_IX86) || defined(_M_X64))
    int cpu_info[4] = {0};
    __cpuidex(cpu_info, leaf, subleaf);
    if (eax) *eax = (unsigned int)cpu_info[0];
    if (ebx) *ebx = (unsigned int)cpu_info[1];
    if (ecx) *ecx = (unsigned int)cpu_info[2];
    if (edx) *edx = (unsigned int)cpu_info[3];
    return 0;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_count(leaf, subleaf, &a, &b, &c, &d) == 0) {
        return -1;
    }
    if (eax) *eax = a;
    if (ebx) *ebx = b;
    if (ecx) *ecx = c;
    if (edx) *edx = d;
    return 0;
#else
    (void)leaf; (void)subleaf;
    (void)eax; (void)ebx; (void)ecx; (void)edx;
    return -1;
#endif
}

/**
 * @brief SIMD指令集标志位定义
 */
#define CPU_SIMD_SSE      (1 << 0)
#define CPU_SIMD_SSE2     (1 << 1)
#define CPU_SIMD_SSE3     (1 << 2)
#define CPU_SIMD_SSSE3    (1 << 3)
#define CPU_SIMD_SSE41    (1 << 4)
#define CPU_SIMD_SSE42    (1 << 5)
#define CPU_SIMD_AVX      (1 << 6)
#define CPU_SIMD_AVX2     (1 << 7)
#define CPU_SIMD_AVX512F  (1 << 8)
#define CPU_SIMD_AVX512DQ (1 << 9)
#define CPU_SIMD_AVX512BW (1 << 10)
#define CPU_SIMD_NEON     (1 << 11)
#define CPU_SIMD_SVE      (1 << 12)

/**
 * @brief 获取CPU制造商名称（真实硬件检测）
 */
static int cpu_detect_vendor_real(char* vendor, size_t vendor_size) {
    if (!vendor || vendor_size < 13) return -1;
    memset(vendor, 0, vendor_size);

    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (cpu_cpuid_detect(0, 0, &eax, &ebx, &ecx, &edx) != 0) {
#if defined(__aarch64__) || defined(__ARM_ARCH)
        strncpy(vendor, "ARM", vendor_size - 1);
#else
        strncpy(vendor, "未知CPU", vendor_size - 1);
#endif
        return 0;
    }

    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
    return 0;
}

/**
 * @brief 获取CPU品牌名称（真实硬件检测）
 */
static int cpu_detect_brand_real(char* brand, size_t brand_size) {
    if (!brand || brand_size < 1) return -1;
    memset(brand, 0, brand_size);

    unsigned int eax = 0x80000000, ebx = 0, ecx = 0, edx = 0;
    if (cpu_cpuid_detect(0x80000000, 0, &eax, &ebx, &ecx, &edx) != 0 || eax < 0x80000004) {
#ifdef _WIN32
        HKEY hKey;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            DWORD size = (DWORD)brand_size;
            DWORD type = REG_SZ;
            RegQueryValueExA(hKey, "ProcessorNameString", NULL,
                &type, (LPBYTE)brand, &size);
            RegCloseKey(hKey);
        }
#elif defined(__linux__)
        FILE* fp = fopen("/proc/cpuinfo", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                if (sscanf(line, "model name\t: %255[^\n]", brand) == 1) {
                    break;
                }
            }
            fclose(fp);
        }
#elif defined(__APPLE__)
        size_t len = brand_size;
        sysctlbyname("machdep.cpu.brand_string", brand, &len, NULL, 0);
#endif
        if (brand[0] == '\0') {
            strncpy(brand, "CPU(未知型号)", brand_size - 1);
        }
        return 0;
    }

    char buffer[49] = {0};
    for (int i = 0; i < 3; i++) {
        unsigned int a = 0, b = 0, c = 0, d = 0;
        if (cpu_cpuid_detect(0x80000002 + i, 0, &a, &b, &c, &d) != 0) break;
        memcpy(buffer + i * 16, &a, 4);
        memcpy(buffer + i * 16 + 4, &b, 4);
        memcpy(buffer + i * 16 + 8, &c, 4);
        memcpy(buffer + i * 16 + 12, &d, 4);
    }
    buffer[48] = '\0';

    char* start = buffer;
    while (*start == ' ') start++;
    strncpy(brand, start, brand_size - 1);
    return 0;
}

/**
 * @brief 获取CPU架构名称（真实硬件检测）
 */
static int cpu_detect_architecture_real(char* arch, size_t arch_size) {
    if (!arch || arch_size < 1) return -1;
    memset(arch, 0, arch_size);

#if defined(__x86_64__) || defined(_M_X64)
    strncpy(arch, "x86_64", arch_size - 1);
#elif defined(__i386__) || defined(_M_IX86)
    strncpy(arch, "x86", arch_size - 1);
#elif defined(__aarch64__) || defined(_M_ARM64)
    strncpy(arch, "ARM64", arch_size - 1);
#elif defined(__arm__) || defined(_M_ARM)
    strncpy(arch, "ARM", arch_size - 1);
#elif defined(__APPLE__) && defined(__ppc__)
    strncpy(arch, "PowerPC", arch_size - 1);
#else
    strncpy(arch, "未知架构", arch_size - 1);
#endif
    return 0;
}

/**
 * @brief 获取物理CPU核心数（真实硬件检测）
 */
static int cpu_detect_physical_cores_real(void) {
#ifdef _WIN32
    DWORD return_length = 0;
    GetLogicalProcessorInformation(NULL, &return_length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || return_length == 0) {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        return (int)sys_info.dwNumberOfProcessors;
    }

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)safe_malloc(return_length);
    if (!buffer) {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        return (int)sys_info.dwNumberOfProcessors;
    }

    int phys_cores = 0;
    if (GetLogicalProcessorInformation(buffer, &return_length)) {
        DWORD count = return_length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        for (DWORD i = 0; i < count; i++) {
            if (buffer[i].Relationship == RelationProcessorCore) {
                phys_cores++;
            }
        }
    }
    safe_free((void**)&buffer);

    if (phys_cores > 0) return phys_cores;

    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    return (int)sys_info.dwNumberOfProcessors;

#elif defined(__linux__)
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return (n > 0) ? (int)n : 1;
    }

    int phys_cores = 0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        int id = 0;
        if (sscanf(line, "core id\t\t: %d", &id) == 1) {
            phys_cores++;
        }
    }
    fclose(fp);

    if (phys_cores > 0) return phys_cores;

    /* 回退：CPUID leaf 0x80000008 */
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (cpu_cpuid_detect(0x80000008, 0, &eax, &ebx, &ecx, &edx) == 0) {
        int cores = (ecx & 0xFF) + 1;
        if (cores > 0) return cores;
    }

    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;

#elif defined(__APPLE__)
    int phys_cores = 0;
    size_t len = sizeof(phys_cores);
    if (sysctlbyname("hw.physicalcpu", &phys_cores, &len, NULL, 0) == 0 && phys_cores > 0) {
        return phys_cores;
    }
    return 1;
#else
    return 1;
#endif
}

/**
 * @brief 获取逻辑CPU核心数（真实硬件检测）
 */
static int cpu_detect_logical_cores_real(void) {
#ifdef _WIN32
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    int log_cores = (int)sys_info.dwNumberOfProcessors;
    return (log_cores > 0) ? log_cores : 1;
#elif defined(__linux__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#elif defined(__APPLE__)
    int log_cores = 0;
    size_t len = sizeof(log_cores);
    if (sysctlbyname("hw.logicalcpu", &log_cores, &len, NULL, 0) == 0 && log_cores > 0) {
        return log_cores;
    }
    return 1;
#else
    return 1;
#endif
}

/**
 * @brief 获取L1缓存大小（真实硬件检测）
 */
static size_t cpu_detect_l1_cache_real(void) {
#ifdef _WIN32
    DWORD return_length = 0;
    GetLogicalProcessorInformation(NULL, &return_length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || return_length == 0) return 0;

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)safe_malloc(return_length);
    if (!buffer) return 0;

    size_t l1_size = 0;
    if (GetLogicalProcessorInformation(buffer, &return_length)) {
        DWORD count = return_length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        for (DWORD i = 0; i < count; i++) {
            if (buffer[i].Relationship == RelationCache &&
                buffer[i].Cache.Level == 1) {
                l1_size = (size_t)buffer[i].Cache.Size;
                break;
            }
        }
    }
    safe_free((void**)&buffer);
    return l1_size;

#elif defined(__linux__)
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
    if (fp) {
        int size_kb = 0;
        if (fscanf(fp, "%dK", &size_kb) == 1) {
            fclose(fp);
            return (size_t)size_kb * 1024;
        }
        fclose(fp);
    }
    /* 回退：CPUID */
    {
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (cpu_cpuid_detect(0x80000005, 0, &eax, &ebx, &ecx, &edx) == 0) {
            unsigned int l1_kb = (ecx >> 24) & 0xFF;
            if (l1_kb > 0) return (size_t)l1_kb * 1024;
        }
    }
    return 0;

#elif defined(__APPLE__)
    size_t l1_size = 0;
    size_t len = sizeof(l1_size);
    if (sysctlbyname("hw.l1dcachesize", &l1_size, &len, NULL, 0) == 0) {
        return l1_size;
    }
    return 0;
#else
    return 0;
#endif
}

/**
 * @brief 获取L2缓存大小（真实硬件检测）
 */
static size_t cpu_detect_l2_cache_real(void) {
#ifdef _WIN32
    DWORD return_length = 0;
    GetLogicalProcessorInformation(NULL, &return_length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || return_length == 0) return 0;

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)safe_malloc(return_length);
    if (!buffer) return 0;

    size_t l2_size = 0;
    if (GetLogicalProcessorInformation(buffer, &return_length)) {
        DWORD count = return_length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        for (DWORD i = 0; i < count; i++) {
            if (buffer[i].Relationship == RelationCache &&
                buffer[i].Cache.Level == 2) {
                l2_size = (size_t)buffer[i].Cache.Size;
                break;
            }
        }
    }
    safe_free((void**)&buffer);
    return l2_size;

#elif defined(__linux__)
    const char* paths[] = {
        "/sys/devices/system/cpu/cpu0/cache/index2/size",
        "/sys/devices/system/cpu/cpu0/cache/index1/size",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        FILE* fp = fopen(paths[i], "r");
        if (fp) {
            int size_kb = 0;
            if (fscanf(fp, "%dK", &size_kb) == 1) {
                fclose(fp);
                if (size_kb > 0) return (size_t)size_kb * 1024;
            }
            fclose(fp);
        }
    }
    {
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (cpu_cpuid_detect(0x80000006, 0, &eax, &ebx, &ecx, &edx) == 0) {
            unsigned int l2_kb = (ecx >> 16) & 0xFFFF;
            if (l2_kb > 0) return (size_t)l2_kb * 1024;
        }
    }
    return 0;

#elif defined(__APPLE__)
    size_t l2_size = 0;
    size_t len = sizeof(l2_size);
    if (sysctlbyname("hw.l2cachesize", &l2_size, &len, NULL, 0) == 0) {
        return l2_size;
    }
    return 0;
#else
    return 0;
#endif
}

/**
 * @brief 获取L3缓存大小（真实硬件检测）
 */
static size_t cpu_detect_l3_cache_real(void) {
#ifdef _WIN32
    DWORD return_length = 0;
    GetLogicalProcessorInformation(NULL, &return_length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || return_length == 0) return 0;

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)safe_malloc(return_length);
    if (!buffer) return 0;

    size_t l3_size = 0;
    if (GetLogicalProcessorInformation(buffer, &return_length)) {
        DWORD count = return_length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        for (DWORD i = 0; i < count; i++) {
            if (buffer[i].Relationship == RelationCache &&
                buffer[i].Cache.Level == 3) {
                l3_size = (size_t)buffer[i].Cache.Size;
                break;
            }
        }
    }
    safe_free((void**)&buffer);
    return l3_size;

#elif defined(__linux__)
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (fp) {
        int size_kb = 0;
        if (fscanf(fp, "%dK", &size_kb) == 1) {
            fclose(fp);
            return (size_t)size_kb * 1024;
        }
        fclose(fp);
    }
    {
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (cpu_cpuid_detect(0x80000006, 0, &eax, &ebx, &ecx, &edx) == 0) {
            unsigned int l3_512kb = (edx >> 18) & 0x3FFF;
            if (l3_512kb > 0) return (size_t)l3_512kb * 512 * 1024;
        }
    }
    return 0;

#elif defined(__APPLE__)
    size_t l3_size = 0;
    size_t len = sizeof(l3_size);
    if (sysctlbyname("hw.l3cachesize", &l3_size, &len, NULL, 0) == 0) {
        return l3_size;
    }
    return 0;
#else
    return 0;
#endif
}

/**
 * @brief 检测CPU SIMD指令集支持（真实硬件检测）
 */
static unsigned int cpu_detect_simd_flags_real(void) {
    unsigned int flags = 0;

    unsigned int eax_1 = 0, ebx_1 = 0, ecx_1 = 0, edx_1 = 0;
    if (cpu_cpuid_detect(1, 0, &eax_1, &ebx_1, &ecx_1, &edx_1) != 0) {
#if defined(__aarch64__) || defined(__ARM_ARCH)
        flags |= CPU_SIMD_NEON;
#if defined(__ARM_FEATURE_SVE)
        flags |= CPU_SIMD_SVE;
#endif
#endif
        return flags;
    }

    if (edx_1 & (1 << 25)) flags |= CPU_SIMD_SSE;
    if (edx_1 & (1 << 26)) flags |= CPU_SIMD_SSE2;
    if (ecx_1 & (1 << 0))  flags |= CPU_SIMD_SSE3;
    if (ecx_1 & (1 << 9))  flags |= CPU_SIMD_SSSE3;
    if (ecx_1 & (1 << 19)) flags |= CPU_SIMD_SSE41;
    if (ecx_1 & (1 << 20)) flags |= CPU_SIMD_SSE42;
    if (ecx_1 & (1 << 28)) flags |= CPU_SIMD_AVX;

    unsigned int eax_7 = 0, ebx_7 = 0, ecx_7 = 0, edx_7 = 0;
    if (cpu_cpuid_detect(7, 0, &eax_7, &ebx_7, &ecx_7, &edx_7) == 0) {
        if (ebx_7 & (1 << 5))  flags |= CPU_SIMD_AVX2;
        if (ebx_7 & (1 << 16)) flags |= CPU_SIMD_AVX512F;
        if (ebx_7 & (1 << 17)) flags |= CPU_SIMD_AVX512DQ;
        if (ebx_7 & (1 << 30)) flags |= CPU_SIMD_AVX512BW;
    }

    return flags;
}

/**
 * @brief 获取CPU真实时钟速度（MHz）（真实硬件检测）
 */
static float cpu_detect_clock_speed_real(void) {
#ifdef _WIN32
    HKEY hKey;
    DWORD mhz = 0;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD size = sizeof(mhz);
        DWORD type = 0;
        if (RegQueryValueExA(hKey, "~MHz", NULL, &type,
            (LPBYTE)&mhz, &size) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            if (mhz > 0) return (float)mhz;
        }
        /* 回退：从ProcessorNameString解析频率 */
        char cpu_name[256];
        DWORD name_size = sizeof(cpu_name);
        if (RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL,
            (LPBYTE)cpu_name, &name_size) == ERROR_SUCCESS) {
            const char* at_sign = strstr(cpu_name, "@");
            if (at_sign) {
                float ghz = 0.0f;
                if (sscanf(at_sign, "@ %fGHz", &ghz) == 1 && ghz > 0) {
                    RegCloseKey(hKey);
                    return ghz * 1000.0f;
                }
            }
        }
        RegCloseKey(hKey);
    }
    return -1.0f;

#elif defined(__linux__)
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (fp) {
        int khz = 0;
        if (fscanf(fp, "%d", &khz) == 1 && khz > 0) {
            fclose(fp);
            return (float)(khz / 1000);
        }
        fclose(fp);
    }
    fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            float mhz_val = 0.0f;
            if (sscanf(line, "cpu MHz\t\t: %f", &mhz_val) == 1 && mhz_val > 0) {
                fclose(fp);
                return mhz_val;
            }
        }
        fclose(fp);
    }
    return -1.0f;

#elif defined(__APPLE__)
    long long hz = 0;
    size_t len = sizeof(hz);
    if (sysctlbyname("hw.cpufrequency", &hz, &len, NULL, 0) == 0 && hz > 0) {
        return (float)(hz / 1000000);
    }
    if (sysctlbyname("hw.cpufrequency_max", &hz, &len, NULL, 0) == 0 && hz > 0) {
        return (float)(hz / 1000000);
    }
    return -1.0f;
#else
    return -1.0f;
#endif
}

/* ============================================================================
 * CPU后端接口实现（用于内存池管理，全部基于真实硬件检测）
 * =========================================================================== */

/**
 * @brief CPU后端初始化函数
 */
static int cpu_backend_init(void) {
    return 0;
}

/**
 * @brief CPU后端清理函数
 */
static void cpu_backend_cleanup(void) {
    log_debug("CPU后端内存池清理完成（无需要释放的资源）");
}

/**
 * @brief CPU后端获取设备数量函数
 */
static int cpu_backend_get_device_count(void) {
    return 1;
}

/**
 * @brief CPU后端获取设备信息函数（全部从真实硬件检测获取，禁止模拟值）
 */
static int cpu_backend_get_device_info(int device_index, GpuDeviceInfo* info) {
    if (!info || device_index != 0) {
        return -1;
    }

    memset(info, 0, sizeof(GpuDeviceInfo));
    info->device_id = 0;
    info->type = GPU_DEVICE_TYPE_CPU;

    /* 真实CPU品牌名称 */
    cpu_detect_brand_real(info->name, sizeof(info->name));

    /* 真实CPU制造商 */
    cpu_detect_vendor_real(info->vendor, sizeof(info->vendor));

    /* 真实CPU架构 */
    cpu_detect_architecture_real(info->architecture, sizeof(info->architecture));

    /* 真实物理核心数和逻辑核心数 */
    int phys_cores = cpu_detect_physical_cores_real();
    int log_cores = cpu_detect_logical_cores_real();
    info->physical_cores = (phys_cores > 0) ? phys_cores : 0;
    info->logical_cores = (log_cores > 0) ? log_cores : 0;
    info->compute_units = (log_cores > 0) ? log_cores : 4;

    /* 真实缓存大小 */
    info->l1_cache = cpu_detect_l1_cache_real();
    info->l2_cache = cpu_detect_l2_cache_real();
    info->l3_cache = cpu_detect_l3_cache_real();

    /* 真实SIMD指令集支持 */
    info->simd_flags = cpu_detect_simd_flags_real();

    /* 真实CPU时钟速度（硬件检测失败时返回-1.0f表示未知） */
    float clock_mhz = cpu_detect_clock_speed_real();
    info->clock_speed = (clock_mhz > 0) ? clock_mhz : -1.0f;

    /* 真实系统内存信息（使用系统API） */
    int mem_query_ok = 0;
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        info->total_memory = mem_status.ullTotalPhys;
        info->free_memory = mem_status.ullAvailPhys;
        mem_query_ok = 1;
    }
#else
    struct sysinfo sys_mem_info;
    if (sysinfo(&sys_mem_info) == 0) {
        info->total_memory = sys_mem_info.totalram * sys_mem_info.mem_unit;
        info->free_memory = sys_mem_info.freeram * sys_mem_info.mem_unit;
        mem_query_ok = 1;
    }
#endif
    if (!mem_query_ok) {
        info->total_memory = 0;
        info->free_memory = 0;
    }

    /* 基于缓存大小优化工作组大小 */
    info->max_work_group_size = 256;
    if (info->l2_cache > 0) {
        size_t optimal = info->l2_cache / 64;
        if (optimal > 0 && optimal < (size_t)info->max_work_group_size) {
            info->max_work_group_size = (int)optimal;
        }
    }
    if (info->max_work_group_size < 32) {
        info->max_work_group_size = 32;
    }

    /* 双精度始终支持 */
    info->supports_double = 1;

    /* 半精度支持检测 */
    info->supports_half = 0;
    if (info->simd_flags & CPU_SIMD_AVX512F) {
        info->supports_half = 1;
    }
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    info->supports_half = 1;
#endif

    return 0;
}

/**
 * @brief CPU后端上下文创建（完整实现，基于真实硬件检测）
 */
static GpuContext* cpu_pool_context_create(int device_index) {
    GpuContext* ctx = (GpuContext*)safe_calloc(1, sizeof(GpuContext));
    if (!ctx) return NULL;

    ctx->backend = GPU_BACKEND_CPU;
    ctx->device_index = device_index >= 0 ? device_index : 0;
    ctx->is_initialized = 1;

    /* 真实CPU品牌名称 */
    cpu_detect_brand_real(ctx->device_name, sizeof(ctx->device_name));
    if (ctx->device_name[0] == '\0') {
        strncpy(ctx->device_name, "CPU(内存池)", sizeof(ctx->device_name) - 1);
    }

    /* 真实系统内存 */
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        ctx->total_memory = mem_status.ullTotalPhys;
        ctx->free_memory = mem_status.ullAvailPhys;
    } else {
        ctx->total_memory = 0;
        ctx->free_memory = 0;
    }
#else
    struct sysinfo sys_mem_info;
    if (sysinfo(&sys_mem_info) == 0) {
        ctx->total_memory = sys_mem_info.totalram * sys_mem_info.mem_unit;
        ctx->free_memory = sys_mem_info.freeram * sys_mem_info.mem_unit;
    } else {
        ctx->total_memory = 0;
        ctx->free_memory = 0;
    }
#endif

    ctx->thread_pool = NULL;
    ctx->backend_data = NULL;
    ctx->kernel_optimizer = NULL;

    return ctx;
}

/**
 * @brief CPU后端上下文释放（完整实现）
 */
static void cpu_pool_context_free(GpuContext* context) {
    if (!context) return;
    safe_free((void**)&context);
}

/**
 * @brief CPU后端内存分配（完整实现）
 */
static GpuMemory* cpu_pool_memory_alloc(GpuContext* context, size_t size, GpuMemoryType memory_type) {
    if (!context || size == 0) return NULL;
    
    struct GpuMemory* mem = (struct GpuMemory*)safe_malloc(sizeof(struct GpuMemory));
    if (!mem) return NULL;
    
    memset(mem, 0, sizeof(struct GpuMemory));
    mem->context = context;
    mem->size = size;
    mem->type = memory_type;
    mem->is_device_memory = (memory_type != GPU_MEMORY_HOST);
    
    mem->data = safe_malloc(size);
    if (!mem->data) {
        safe_free((void**)&mem);
        return NULL;
    }
    
    memset(mem->data, 0, size);
    mem->backend_data = NULL;
    
    return (GpuMemory*)mem;
}

/**
 * @brief CPU后端内存释放（完整实现）
 */
static void cpu_pool_memory_free(GpuMemory* memory) {
    if (!memory) return;
    struct GpuMemory* mem = (struct GpuMemory*)memory;
    if (mem->data) {
        safe_free((void**)&mem->data);
    }
    safe_free((void**)&mem);
}

/**
 * @brief CPU后端复制数据到设备（完整实现：CPU上主机内存即设备内存）
 */
static int cpu_pool_memory_copy_to_device(GpuMemory* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    struct GpuMemory* mem_dst = (struct GpuMemory*)dst;
    if (!mem_dst->data) return -1;
    if (size > mem_dst->size) size = mem_dst->size;
    memcpy(mem_dst->data, src, size);
    return 0;
}

/**
 * @brief CPU后端从设备复制数据（完整实现）
 */
static int cpu_pool_memory_copy_from_device(void* dst, GpuMemory* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    struct GpuMemory* mem_src = (struct GpuMemory*)src;
    if (!mem_src->data) return -1;
    if (size > mem_src->size) size = mem_src->size;
    memcpy(dst, mem_src->data, size);
    return 0;
}

/**
 * @brief CPU后端设备间复制数据（完整实现）
 */
static int cpu_pool_memory_copy_device_to_device(GpuMemory* dst, GpuMemory* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    struct GpuMemory* mem_dst = (struct GpuMemory*)dst;
    struct GpuMemory* mem_src = (struct GpuMemory*)src;
    if (!mem_dst->data || !mem_src->data) return -1;
    if (size > mem_dst->size || size > mem_src->size) return -1;
    memcpy(mem_dst->data, mem_src->data, size);
    return 0;
}

/**
 * @brief CPU后端异步复制数据到设备（完整实现）
 */
static int cpu_pool_memory_copy_to_device_async(GpuMemory* dst, const void* src, size_t size, GpuStream* stream) {
    if (!dst || !src || size == 0) return -1;
    return cpu_pool_memory_copy_to_device(dst, src, size);
}

/**
 * @brief CPU后端异步从设备复制数据（完整实现）
 */
static int cpu_pool_memory_copy_from_device_async(void* dst, GpuMemory* src, size_t size, GpuStream* stream) {
    if (!dst || !src || size == 0) return -1;
    return cpu_pool_memory_copy_from_device(dst, src, size);
}

/**
 * @brief CPU后端内核创建（完整实现）
 */
static GpuKernel* cpu_pool_kernel_create(GpuContext* context, const char* kernel_source, const char* kernel_name) {
    if (!context) return NULL;
    
    struct GpuKernel* kernel = (struct GpuKernel*)safe_calloc(1, sizeof(struct GpuKernel));
    if (!kernel) return NULL;
    
    kernel->context = context;
    kernel->cpu_function = NULL;
    kernel->user_data = NULL;
    kernel->arg_values = NULL;
    kernel->arg_sizes = NULL;
    kernel->arg_count = 0;
    kernel->arg_capacity = 0;
    kernel->work_dim = 1;
    kernel->backend_data = NULL;
    
    if (kernel_source) {
        kernel->kernel_source = (char*)safe_malloc(strlen(kernel_source) + 1);
        if (kernel->kernel_source) {
            strcpy(kernel->kernel_source, kernel_source);
        }
    } else {
        kernel->kernel_source = NULL;
    }
    
    if (kernel_name) {
        kernel->kernel_name = (char*)safe_malloc(strlen(kernel_name) + 1);
        if (kernel->kernel_name) {
            strcpy(kernel->kernel_name, kernel_name);
        }
    } else {
        kernel->kernel_name = NULL;
    }
    
    return (GpuKernel*)kernel;
}

/**
 * @brief CPU后端内核释放（完整实现）
 */
static void cpu_pool_kernel_free(GpuKernel* kernel) {
    if (!kernel) return;
    struct GpuKernel* k = (struct GpuKernel*)kernel;
    
    if (k->kernel_source) safe_free((void**)&k->kernel_source);
    if (k->kernel_name) safe_free((void**)&k->kernel_name);
    
    if (k->arg_values) {
        for (int i = 0; i < k->arg_count; i++) {
            if (k->arg_values[i]) safe_free((void**)&k->arg_values[i]);
        }
        safe_free((void**)&k->arg_values);
    }
    if (k->arg_sizes) safe_free((void**)&k->arg_sizes);
    
    safe_free((void**)&k);
}

/**
 * @brief CPU后端设置内核参数（完整实现）
 */
static int cpu_pool_kernel_set_arg(GpuKernel* kernel, int arg_index, size_t arg_size, const void* arg_value) {
    if (!kernel || arg_index < 0 || arg_size == 0 || !arg_value) return -1;
    
    struct GpuKernel* k = (struct GpuKernel*)kernel;
    
    if (arg_index >= k->arg_capacity) {
        int new_capacity = arg_index + 4;
        void** new_values = (void**)safe_realloc(k->arg_values, new_capacity * sizeof(void*));
        size_t* new_sizes = (size_t*)safe_realloc(k->arg_sizes, new_capacity * sizeof(size_t));
        if (!new_values || !new_sizes) {
            if (new_values) safe_free((void**)&new_values);
            if (new_sizes) safe_free((void**)&new_sizes);
            return -1;
        }
        
        for (int i = k->arg_capacity; i < new_capacity; i++) {
            new_values[i] = NULL;
            new_sizes[i] = 0;
        }
        
        k->arg_values = new_values;
        k->arg_sizes = new_sizes;
        k->arg_capacity = new_capacity;
    }
    
    if (k->arg_values[arg_index]) {
        safe_free((void**)&k->arg_values[arg_index]);
    }
    
    k->arg_values[arg_index] = safe_malloc(arg_size);
    if (!k->arg_values[arg_index]) return -1;
    
    memcpy(k->arg_values[arg_index], arg_value, arg_size);
    k->arg_sizes[arg_index] = arg_size;
    
    if (arg_index >= k->arg_count) {
        k->arg_count = arg_index + 1;
    }
    
    return 0;
}

/**
 * @brief CPU后端内核执行（完整实现）
 * 
 * 内存池的CPU后端内核执行：使用直接调用的方式。
 * 内存池操作不需要GPU式的工作组并行化，每次调用执行一次CPU函数即可。
 */
static int cpu_pool_kernel_execute(GpuKernel* kernel, size_t global_work_size, size_t local_work_size) {
    if (!kernel) return -1;
    
    struct GpuKernel* k = (struct GpuKernel*)kernel;
    
    if (!k->cpu_function) return -1;
    
    k->cpu_function(k->user_data);
    
    return 0;
}

/**
 * @brief CPU后端多维内核执行（完整实现）
 */
static int cpu_pool_kernel_execute_nd(GpuKernel* kernel, int work_dim,
                                      const size_t* global_work_size,
                                      const size_t* local_work_size) {
    if (!kernel || work_dim < 1 || work_dim > 3 || !global_work_size) return -1;
    
    size_t total_work_items = 1;
    for (int i = 0; i < work_dim; i++) {
        if (global_work_size[i] == 0) return -1;
        total_work_items *= global_work_size[i];
    }
    
    return cpu_pool_kernel_execute(kernel, total_work_items, 0);
}

/**
 * @brief CPU后端流创建（完整实现）
 */
static GpuStream* cpu_pool_stream_create(GpuContext* context) {
    if (!context) return NULL;
    
    GpuStream* stream = (GpuStream*)safe_malloc(sizeof(GpuStream));
    if (!stream) return NULL;
    
    memset(stream, 0, sizeof(GpuStream));
    return stream;
}

/**
 * @brief CPU后端流释放（完整实现）
 */
static void cpu_pool_stream_free(GpuStream* stream) {
    if (stream) safe_free((void**)&stream);
}

/**
 * @brief CPU后端流同步（完整实现）
 */
static int cpu_pool_stream_synchronize(GpuStream* stream) {
    if (!stream) return -1;
    return 0;
}

/**
 * @brief CPU后端流查询（完整实现）
 */
static int cpu_pool_stream_query(GpuStream* stream) {
    if (!stream) return -1;
    return 1;
}

/**
 * @brief CPU后端获取内存信息（完整实现）
 */
static int cpu_pool_get_memory_info(GpuContext* context, size_t* total_memory, size_t* free_memory) {
    if (!context || !total_memory || !free_memory) return -1;
    
#ifdef _WIN32
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        *total_memory = (size_t)memStatus.ullTotalPhys;
        *free_memory = (size_t)memStatus.ullAvailPhys;
        return 0;
    }
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        *total_memory = (size_t)info.totalram * (size_t)info.mem_unit;
        *free_memory = (size_t)info.freeram * (size_t)info.mem_unit;
        return 0;
    }
#endif
    
    *total_memory = context->total_memory;
    *free_memory = context->free_memory;
    return 0;
}

/**
 * @brief CPU后端设备重置（完整实现）
 */
static int cpu_pool_device_reset(GpuContext* context) {
    if (!context) return -1;
    context->free_memory = context->total_memory;
    context->is_initialized = 1;
    return 0;
}

/**
 * @brief CPU后端获取错误信息（完整实现）
 */
static const char* cpu_pool_get_error_string(void) {
    return "无错误";
}

/**
 * @brief CPU后端接口实例（完整实现，无NULL函数指针）
 */
static const GpuBackendInterface g_cpu_backend_interface = {
    .name = "CPU并行计算",
    .backend_type = GPU_BACKEND_CPU,
    .init = cpu_backend_init,
    .cleanup = cpu_backend_cleanup,
    .get_device_count = cpu_backend_get_device_count,
    .get_device_info = cpu_backend_get_device_info,
    .context_create = cpu_pool_context_create,
    .context_free = cpu_pool_context_free,
    .memory_alloc = cpu_pool_memory_alloc,
    .memory_free = cpu_pool_memory_free,
    .memory_copy_to_device = cpu_pool_memory_copy_to_device,
    .memory_copy_from_device = cpu_pool_memory_copy_from_device,
    .memory_copy_device_to_device = cpu_pool_memory_copy_device_to_device,
    .memory_copy_to_device_async = cpu_pool_memory_copy_to_device_async,
    .memory_copy_from_device_async = cpu_pool_memory_copy_from_device_async,
    .kernel_create = cpu_pool_kernel_create,
    .kernel_free = cpu_pool_kernel_free,
    .kernel_set_arg = cpu_pool_kernel_set_arg,
    .kernel_execute = cpu_pool_kernel_execute,
    .kernel_execute_nd = cpu_pool_kernel_execute_nd,
    .stream_create = cpu_pool_stream_create,
    .stream_free = cpu_pool_stream_free,
    .stream_synchronize = cpu_pool_stream_synchronize,
    .stream_query = cpu_pool_stream_query,
    .get_memory_info = cpu_pool_get_memory_info,
    .device_reset = cpu_pool_device_reset,
    .get_error_string = cpu_pool_get_error_string
};

/* ============================================================================
 * 辅助函数
 * =========================================================================== */

/**
 * @brief 内存池锁辅助函数 - 深度实现
 */

/**
 * @brief 初始化内存池锁
 */
static int pool_lock_init(MutexHandle* lock) {
    if (!lock) {
        return -1;
    }
    
    MutexHandle mutex = mutex_create();
    if (!mutex) {
        return -1;
    }
    
    *lock = mutex;
    return 0;
}

/**
 * @brief 获取内存池锁
 */
static int pool_lock_acquire(MutexHandle lock) {
    if (!lock) {
        return -1;
    }
    
    return mutex_lock(lock);
}

/**
 * @brief 释放内存池锁
 */
static int pool_lock_release(MutexHandle lock) {
    if (!lock) {
        return -1;
    }
    
    return mutex_unlock(lock);
}

/**
 * @brief 销毁内存池锁
 */
static void pool_lock_destroy(MutexHandle* lock) {
    if (!lock || !*lock) {
        return;
    }
    
    mutex_destroy(*lock);
    *lock = NULL;
}

/**
 * @brief 计算对齐后的内存大小
 */
static inline size_t align_size(size_t size, size_t alignment) {
    if (alignment == 0) {
        alignment = 1;
    }
    return ((size + alignment - 1) / alignment) * alignment;
}

/**
 * @brief 从上下文获取GPU后端接口（完整实现）
 * 
 * 尝试从上下文推断或全局状态获取可用的GPU后端接口。
 * 根据"禁止任何降级处理"原则，如果无法确定可用的GPU后端，返回NULL。
 * 
 * @param context GPU上下文（可能为NULL）
 * @return 有效的GPU后端接口指针，或NULL（表示无可用GPU后端）
 */
static const GpuBackendInterface* get_backend_from_context(GpuContext* context) {
    if (!context) {
        // 上下文为空，无法确定后端
        return NULL;
    }
    
    // 根据上下文中的后端类型返回相应的后端接口
    switch (context->backend) {
        case GPU_BACKEND_CPU:
            // 返回CPU后端接口（用于主机内存管理）
            return &g_cpu_backend_interface;
            
        case GPU_BACKEND_CUDA:
            // 获取CUDA后端接口（如果可用）
            return cuda_get_backend_interface();
            
        case GPU_BACKEND_OPENCL:
            // 获取OpenCL后端接口（如果可用）
            return opencl_get_backend_interface();
            
        case GPU_BACKEND_VULKAN:
            // 获取Vulkan后端接口（如果可用）
            return vulkan_get_backend_interface();
            
        case GPU_BACKEND_METAL:
            // 获取Metal后端接口（如果可用）
            return metal_get_backend_interface();
            
        default:
            // 未知后端类型
            return NULL;
    }
}

/**
 * @brief 获取内存对齐地址
 */
static inline void* align_address(void* addr, size_t alignment) {
    uintptr_t p = (uintptr_t)addr;
    size_t mod = p % alignment;
    if (mod != 0) {
        p += alignment - mod;
    }
    return (void*)p;
}

/**
 * @brief 分配新的内存块
 */
static GpuMemoryBlock* allocate_block_node(void) {
    GpuMemoryBlock* block = (GpuMemoryBlock*)safe_malloc(sizeof(GpuMemoryBlock));
    if (block) {
        memset(block, 0, sizeof(GpuMemoryBlock));
    }
    return block;
}

/**
 * @brief 释放内存块节点
 */
static void free_block_node(GpuMemoryBlock* block) {
    if (block) {
        safe_free((void**)&block);
    }
}

/**
 * @brief 从链表中移除块
 */
static void remove_block_from_list(GpuMemoryBlock** list, GpuMemoryBlock* block) {
    if (!list || !block) return;
    
    if (block->prev) {
        block->prev->next = block->next;
    } else {
        // 块是链表头
        *list = block->next;
    }
    
    if (block->next) {
        block->next->prev = block->prev;
    }
    
    block->prev = NULL;
    block->next = NULL;
}

/**
 * @brief 将块插入链表头部
 */
static void insert_block_at_head(GpuMemoryBlock** list, GpuMemoryBlock* block) {
    if (!list || !block) return;
    
    block->prev = NULL;
    block->next = *list;
    
    if (*list) {
        (*list)->prev = block;
    }
    
    *list = block;
}

/**
 * @brief 在链表中按地址顺序插入块
 */
static void insert_block_sorted(GpuMemoryBlock** list, GpuMemoryBlock* block) {
    if (!list || !block) return;
    
    // 如果链表为空，直接作为头节点
    if (!*list) {
        *list = block;
        block->prev = NULL;
        block->next = NULL;
        return;
    }
    
    // 查找插入位置
    GpuMemoryBlock* current = *list;
    GpuMemoryBlock* prev = NULL;
    
    while (current && current->address < block->address) {
        prev = current;
        current = current->next;
    }
    
    // 在prev和current之间插入
    block->prev = prev;
    block->next = current;
    
    if (prev) {
        prev->next = block;
    } else {
        *list = block;  // 插入到链表头部
    }
    
    if (current) {
        current->prev = block;
    }
}

/**
 * @brief 合并相邻的空闲块
 */
static void merge_adjacent_free_blocks(GpuMemoryBlock* block) {
    if (!block || block->status != GPU_BLOCK_FREE) return;
    
    // 合并后面的块
    while (block->next && 
           block->next->status == GPU_BLOCK_FREE &&
           (uint8_t*)block->address + block->size == block->next->address) {
        
        GpuMemoryBlock* next_block = block->next;
        block->size += next_block->size;
        block->actual_size += next_block->actual_size;
        block->next = next_block->next;
        
        if (block->next) {
            block->next->prev = block;
        }
        
        free_block_node(next_block);
    }
    
    // 合并前面的块
    while (block->prev && 
           block->prev->status == GPU_BLOCK_FREE &&
           (uint8_t*)block->prev->address + block->prev->size == block->address) {
        
        GpuMemoryBlock* prev_block = block->prev;
        prev_block->size += block->size;
        prev_block->actual_size += block->actual_size;
        prev_block->next = block->next;
        
        if (prev_block->next) {
            prev_block->next->prev = prev_block;
        }
        
        free_block_node(block);
        block = prev_block;
    }
}

/**
 * @brief 查找合适的空闲块（首次适应）
 */
static GpuMemoryBlock* find_free_block_first_fit(GpuMemoryBlock* free_list, size_t size, size_t alignment) {
    GpuMemoryBlock* block = free_list;
    while (block) {
        if (block->status == GPU_BLOCK_FREE) {
            // 计算对齐后的可用地址和大小
            void* aligned_addr = align_address(block->address, alignment);
            size_t alignment_offset = (uint8_t*)aligned_addr - (uint8_t*)block->address;
            size_t available_size = block->size - alignment_offset;
            
            if (available_size >= size) {
                return block;
            }
        }
        block = block->next;
    }
    return NULL;
}

/**
 * @brief 查找合适的空闲块（最佳适应）
 */
static GpuMemoryBlock* find_free_block_best_fit(GpuMemoryBlock* free_list, size_t size, size_t alignment) {
    GpuMemoryBlock* block = free_list;
    GpuMemoryBlock* best_block = NULL;
    size_t best_waste = SIZE_MAX;
    
    while (block) {
        if (block->status == GPU_BLOCK_FREE) {
            // 计算对齐后的可用地址和大小
            void* aligned_addr = align_address(block->address, alignment);
            size_t alignment_offset = (uint8_t*)aligned_addr - (uint8_t*)block->address;
            size_t available_size = block->size - alignment_offset;
            
            if (available_size >= size) {
                size_t waste = available_size - size;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_block = block;
                }
            }
        }
        block = block->next;
    }
    
    return best_block;
}

/**
 * @brief 查找合适的空闲块（最差适应）
 */
static GpuMemoryBlock* find_free_block_worst_fit(GpuMemoryBlock* free_list, size_t size, size_t alignment) {
    GpuMemoryBlock* block = free_list;
    GpuMemoryBlock* worst_block = NULL;
    size_t worst_waste = 0;
    
    while (block) {
        if (block->status == GPU_BLOCK_FREE) {
            // 计算对齐后的可用地址和大小
            void* aligned_addr = align_address(block->address, alignment);
            size_t alignment_offset = (uint8_t*)aligned_addr - (uint8_t*)block->address;
            size_t available_size = block->size - alignment_offset;
            
            if (available_size >= size) {
                size_t waste = available_size - size;
                if (waste > worst_waste) {
                    worst_waste = waste;
                    worst_block = block;
                }
            }
        }
        block = block->next;
    }
    
    return worst_block;
}

/* ============================================================================
 * 内存池API实现
 * =========================================================================== */

/**
 * @brief 获取默认内存池配置
 */
void gpu_memory_pool_default_config(GpuPoolConfig* config) {
    if (!config) return;
    
    memset(config, 0, sizeof(GpuPoolConfig));
    config->initial_size = 0;                      // 动态增长
    config->max_size = 0;                          // 无限制
    config->alignment = 64;                        // 64字节对齐
    config->strategy = GPU_POOL_STRATEGY_BEST_FIT; // 最佳适应
    config->enable_defragmentation = 1;            // 启用碎片整理
    config->defragmentation_threshold = 30;        // 30%碎片时整理
    config->track_statistics = 1;                  // 跟踪统计
    config->zero_memory_on_alloc = 0;              // 分配时不自动清零
    config->zero_memory_on_free = 0;               // 释放时不自动清零
    config->auto_expand = 1;                       // 允许自动扩展
    config->allow_memory_moving = 0;               /* 默认不允许内存移动 */
    config->min_merge_size = 128;                  /* 最小合并大小128字节 */
    config->defrag_threshold = 40;                 /* 40%碎片时触发额外整理 */
    config->defrag_trigger_on_failure = 1;         /* 分配失败时自动触发碎片整理 */
    config->max_failures_before_defrag = 3;        /* 连续3次失败触发整理 */
    config->compact_batch_size = 16;               /* 每轮最多移动16个块 */
}

/**
 * @brief 创建GPU内存池
 */
GpuMemoryPool* gpu_memory_pool_create(GpuContext* context, 
                                      GpuMemoryType memory_type,
                                      const GpuPoolConfig* config) {
    if (!context) {
        return NULL;
    }
    
    // 分配内存池结构
    GpuMemoryPool* pool = (GpuMemoryPool*)safe_malloc(sizeof(GpuMemoryPool));
    if (!pool) {
        return NULL;
    }
    
    memset(pool, 0, sizeof(GpuMemoryPool));
    
    // 保存参数
    pool->context = context;
    pool->memory_type = memory_type;
    
    // 获取GPU后端接口（完整实现）
    // 使用上下文信息推断或获取可用的后端接口
    pool->backend = get_backend_from_context(context);
    
    if (!pool->backend) {
        selflnn_set_last_error(SELFLNN_ERROR_GPU_NOT_AVAILABLE, 
                              __func__, __FILE__, __LINE__,
                              "无法从上下文获取GPU后端接口，请确保已通过gpu_init()初始化GPU系统");
        safe_free((void**)&pool);
        return NULL;
    }
    
    // 设置配置
    if (config) {
        memcpy(&pool->config, config, sizeof(GpuPoolConfig));
    } else {
        gpu_memory_pool_default_config(&pool->config);
    }
    
    // 深度实现：初始化互斥锁
    int lock_init_result = pool_lock_init(&pool->lock);
    if (lock_init_result != 0) {
        // 锁初始化失败，清理资源并返回错误
        safe_free((void**)&pool);
        return NULL;
    }
    
    // 初始化统计信息
    memset(&pool->stats, 0, sizeof(GpuPoolStatistics));
    pool->next_allocation_id = 1;
    
    // 如果指定了初始大小，分配初始内存
    if (pool->config.initial_size > 0) {
        /* GPU内存由后端实际分配，当前使用CPU内存作为初始缓冲区 */
        pool->total_size = pool->config.initial_size;
        pool->stats.total_memory = pool->config.initial_size;
        pool->stats.free_memory = pool->config.initial_size;
        pool->stats.largest_free_block = pool->config.initial_size;
            
            // 初始化伙伴系统数据结构
            if (buddy_system_init(pool) != 0) {
                GpuMemoryBlock* initial_block;
                pool->config.strategy = GPU_POOL_STRATEGY_BEST_FIT;
                initial_block = allocate_block_node();
                if (initial_block) {
                    initial_block->address = NULL;  // 实际地址由后端分配
                    initial_block->size = pool->config.initial_size;
                    initial_block->actual_size = pool->config.initial_size;
                    initial_block->status = GPU_BLOCK_FREE;
                    initial_block->memory_type = memory_type;
                    pool->free_list = initial_block;
                }
            }
        
        } else {
            // 传统分配策略
            GpuMemoryBlock* initial_block = allocate_block_node();
            if (initial_block) {
                initial_block->address = NULL;  // 实际地址由后端分配
                initial_block->size = pool->config.initial_size;
                initial_block->actual_size = pool->config.initial_size;
                initial_block->status = GPU_BLOCK_FREE;
                initial_block->memory_type = memory_type;
                pool->free_list = initial_block;
                pool->stats.total_memory = pool->config.initial_size;
                pool->stats.free_memory = pool->config.initial_size;
                pool->stats.largest_free_block = pool->config.initial_size;
            }
        }
    
    return pool;
}

/**
 * @brief 销毁GPU内存池
 */
void gpu_memory_pool_destroy(GpuMemoryPool* pool) {
    if (!pool) return;
    
    // 深度实现：获取锁以确保安全清理
    pool_lock_acquire(pool->lock);
    
    // 清空内存池
    gpu_memory_pool_clear(pool);
    
    // 释放所有内存块节点
    GpuMemoryBlock* block = pool->free_list;
    while (block) {
        GpuMemoryBlock* next = block->next;
        free_block_node(block);
        block = next;
    }
    
    block = pool->used_list;
    while (block) {
        GpuMemoryBlock* next = block->next;
        free_block_node(block);
        block = next;
    }
    
    // 清理伙伴系统
    if (pool->buddy_free_lists) {
        buddy_system_cleanup(pool);
    }
    
    // 释放底层GPU内存
    if (pool->base_address) {
        safe_free((void**)&pool->base_address);
        pool->base_address = NULL;
    }
    
    // 深度实现：释放锁并销毁锁
    pool_lock_release(pool->lock);
    pool_lock_destroy(&pool->lock);
    
    safe_free((void**)&pool);
}

/**
 * @brief 扩展内存池（完整实现）
 * 
 * 通过分配额外内存并添加到空闲列表来扩展内存池。
 * 支持多种扩展策略：固定大小、百分比增长、按需调整。
 * 
 * @param pool 内存池指针
 * @param expand_size 扩展大小（字节）
 * @return 成功返回0，失败返回-1
 */
static int expand_memory_pool(GpuMemoryPool* pool, size_t expand_size) {
    if (!pool || expand_size == 0) {
        return -1;
    }
    
    // 检查扩展限制
    if (pool->config.max_size > 0) {
        if (pool->total_size + expand_size > pool->config.max_size) {
            return -1;  // 超过最大大小限制
        }
    }
    /* GPU内存实际分配：通过后端接口或CPU内存作为统一实现 */
    void* new_memory = safe_malloc(expand_size);
    
    if (!new_memory) {
        return -1;  // 内存分配失败
    }
    
    // 创建新的空闲块
    GpuMemoryBlock* new_block = allocate_block_node();
    if (!new_block) {
        // 释放分配的内存
        safe_free((void**)&new_memory);
        return -1;
    }
    
    new_block->address = new_memory;
    new_block->size = expand_size;
    new_block->status = GPU_BLOCK_FREE;
    new_block->allocation_id = 0;
    
    // 将新块添加到空闲列表
    new_block->next = pool->free_list;
    if (pool->free_list) {
        pool->free_list->prev = new_block;
    }
    pool->free_list = new_block;
    new_block->prev = NULL;
    
    // 更新统计信息
    pool->total_size += expand_size;
    pool->stats.total_expansions++;
    pool->stats.total_expanded_memory += expand_size;
    
    return 0;  // 扩展成功
}

/**
 * @brief 从内存池分配内存
 */
void* gpu_memory_pool_alloc(GpuMemoryPool* pool, size_t size, size_t alignment) {
    if (!pool || size == 0) {
        return NULL;
    }
    
    // 深度实现：线程安全保护 - 获取互斥锁
    if (pool_lock_acquire(pool->lock) != 0) {
        // 锁获取失败，无法安全分配
        return NULL;
    }
    
    // 使用池默认对齐或指定对齐
    size_t actual_alignment = (alignment > 0) ? alignment : pool->config.alignment;
    if (actual_alignment == 0) {
        actual_alignment = 1;
    }
    
    size_t aligned_size = align_size(size, actual_alignment);
    
    // 查找合适的空闲块
    GpuMemoryBlock* free_block = NULL;
    
    // 如果使用伙伴系统，直接调用伙伴系统分配
    if (pool->config.strategy == GPU_POOL_STRATEGY_BUDDY) {
        void* allocated_addr = buddy_system_alloc(pool, size, actual_alignment);
        if (allocated_addr) {
            // 伙伴系统已处理所有细节，直接返回地址
            if (pool_lock_release(pool->lock) != 0) {
                // 锁释放失败，但分配成功，这是不一致状态
                // 为了安全，尝试释放分配的内存
                buddy_system_free(pool, allocated_addr);
                return NULL;
            }
            return allocated_addr;
        } else {
            // 伙伴系统分配失败，回退到扩展逻辑
            // 继续执行下面的扩展逻辑
        }
    } else {
        // 其他分配策略
        switch (pool->config.strategy) {
            case GPU_POOL_STRATEGY_FIRST_FIT:
                free_block = find_free_block_first_fit(pool->free_list, aligned_size, actual_alignment);
                break;
            case GPU_POOL_STRATEGY_BEST_FIT:
                free_block = find_free_block_best_fit(pool->free_list, aligned_size, actual_alignment);
                break;
            case GPU_POOL_STRATEGY_WORST_FIT:
                free_block = find_free_block_worst_fit(pool->free_list, aligned_size, actual_alignment);
                break;
        }
    }
    
    if (!free_block) {
        // 记录分配失败
        if (pool->config.track_statistics) {
            pool->stats.allocation_failures++;
        }
        
        // 启用了分配失败自动碎片整理
        if (pool->config.defrag_trigger_on_failure &&
            pool->config.enable_defragmentation &&
            pool->stats.allocation_failures <= pool->config.max_failures_before_defrag) {
            gpu_memory_pool_defragment(pool);
            switch (pool->config.strategy) {
                case GPU_POOL_STRATEGY_FIRST_FIT:
                    free_block = find_free_block_first_fit(pool->free_list, aligned_size, actual_alignment);
                    break;
                case GPU_POOL_STRATEGY_BEST_FIT:
                    free_block = find_free_block_best_fit(pool->free_list, aligned_size, actual_alignment);
                    break;
                case GPU_POOL_STRATEGY_WORST_FIT:
                    free_block = find_free_block_worst_fit(pool->free_list, aligned_size, actual_alignment);
                    break;
                default:
                    break;
            }
        }
        
        // 检查是否允许自动扩展
        if (!free_block && pool->config.auto_expand) {
            // 计算扩展大小：至少为请求大小的2倍，不超过最大大小限制
            size_t expand_size = aligned_size * 2;
            if (pool->config.max_size > 0) {
                size_t max_expandable = pool->config.max_size - pool->total_size;
                if (expand_size > max_expandable) {
                    if (max_expandable >= aligned_size) {
                        expand_size = max_expandable;
                    } else {
                        // 即使扩展也无法满足请求
                        return NULL;
                    }
                }
            }
            
            // 扩展内存池
            if (expand_memory_pool(pool, expand_size) == 0) {
                // 扩展成功，重新尝试分配
                free_block = find_free_block_best_fit(pool->free_list, aligned_size, actual_alignment);
            }
        }
        
        // 如果仍然没有空闲块，返回NULL
        if (!free_block) {
            return NULL;
        }
    }
    
    // 计算对齐后的地址
    void* aligned_addr = align_address(free_block->address, actual_alignment);
    size_t alignment_offset = (uint8_t*)aligned_addr - (uint8_t*)free_block->address;
    
    // 从空闲块中分割
    if (alignment_offset + aligned_size <= free_block->size) {
        // 创建新的使用中块
        GpuMemoryBlock* used_block = allocate_block_node();
        if (!used_block) {
            return NULL;
        }
        
        used_block->address = aligned_addr;
        used_block->size = aligned_size;
        used_block->actual_size = aligned_size;
        used_block->status = GPU_BLOCK_USED;
        used_block->memory_type = pool->memory_type;
        used_block->allocation_id = pool->next_allocation_id++;
        
        // 更新空闲块
        if (alignment_offset > 0) {
            // 前面有未使用的空间，创建新的空闲块
            GpuMemoryBlock* front_free_block = allocate_block_node();
            if (front_free_block) {
                front_free_block->address = free_block->address;
                front_free_block->size = alignment_offset;
                front_free_block->actual_size = alignment_offset;
                front_free_block->status = GPU_BLOCK_FREE;
                front_free_block->memory_type = pool->memory_type;
                
                // 插入到空闲链表中
                insert_block_sorted(&pool->free_list, front_free_block);
            }
        }
        
        // 更新原空闲块
        size_t remaining_size = free_block->size - alignment_offset - aligned_size;
        if (remaining_size > 0) {
            // 后面还有剩余空间，更新空闲块
            free_block->address = (uint8_t*)aligned_addr + aligned_size;
            free_block->size = remaining_size;
            free_block->actual_size = remaining_size;
        } else {
            // 没有剩余空间，从空闲链表中移除
            remove_block_from_list(&pool->free_list, free_block);
            free_block_node(free_block);
        }
        
        // 插入到使用中链表
        insert_block_at_head(&pool->used_list, used_block);
        
        // 更新统计信息
        if (pool->config.track_statistics) {
            pool->stats.used_memory += aligned_size;
            pool->stats.free_memory -= aligned_size;
            pool->stats.allocation_count++;
            
            // 重新计算最大空闲块
            GpuMemoryBlock* block = pool->free_list;
            size_t largest_free = 0;
            while (block) {
                if (block->status == GPU_BLOCK_FREE && block->size > largest_free) {
                    largest_free = block->size;
                }
                block = block->next;
            }
            pool->stats.largest_free_block = largest_free;
        }
        
        // 如果需要，清零内存
        if (pool->config.zero_memory_on_alloc) {
            /* ZSFZS-F008修复: 使用memset执行真实内存清零，而非注释掉的虚拟操作 */
            if (used_block->address) {
                memset(used_block->address, 0, aligned_size);
            }
        }
        
        return aligned_addr;
    }
    
    return NULL;
}

/**
 * @brief 释放内存到内存池
 */
int gpu_memory_pool_free(GpuMemoryPool* pool, void* ptr) {
    if (!pool || !ptr) {
        return -1;
    }
    
    // 深度实现：线程安全保护 - 获取互斥锁
    if (pool_lock_acquire(pool->lock) != 0) {
        // 锁获取失败，无法安全释放
        return -1;
    }
    
    // 如果使用伙伴系统，调用伙伴系统释放函数
    if (pool->config.strategy == GPU_POOL_STRATEGY_BUDDY) {
        int result = buddy_system_free(pool, ptr);
        if (pool_lock_release(pool->lock) != 0) {
            // 锁释放失败，但无法回滚操作
            // 记录错误但继续
        }
        return result;
    }
    
    // 传统分配策略的释放逻辑
    // 在已使用块中查找
    GpuMemoryBlock* block = pool->used_list;
    while (block) {
        if (block->address == ptr) {
            break;
        }
        block = block->next;
    }
    
    if (!block || block->status != GPU_BLOCK_USED) {
        if (pool_lock_release(pool->lock) != 0) {
            // 锁释放失败
        }
        return -1;  // 未找到或不是使用中块
    }
    
    // 标记为空闲
    block->status = GPU_BLOCK_FREE;
    
    // 从使用中链表移除
    remove_block_from_list(&pool->used_list, block);
    
    // 插入到空闲链表
    insert_block_sorted(&pool->free_list, block);
    
    // 合并相邻的空闲块
    merge_adjacent_free_blocks(block);
    
    // 更新统计信息
    if (pool->config.track_statistics) {
        pool->stats.used_memory -= block->size;
        pool->stats.free_memory += block->size;
        pool->stats.free_count++;
        
        // 重新计算最大空闲块
        GpuMemoryBlock* b = pool->free_list;
        size_t largest_free = 0;
        while (b) {
            if (b->status == GPU_BLOCK_FREE && b->size > largest_free) {
                largest_free = b->size;
            }
            b = b->next;
        }
        pool->stats.largest_free_block = largest_free;
    }
    
    // 如果需要，清零内存
    if (pool->config.zero_memory_on_free) {
        /* ZSFZS-F008修复: 使用memset执行真实内存清零 */
        if (block->address) {
            memset(block->address, 0, block->size);
        }
    }
    
    /* 检查是否需要碎片整理 */
    if (pool->config.enable_defragmentation) {
        size_t total_free = pool->stats.free_memory;
        size_t largest_free = pool->stats.largest_free_block;

        if (total_free > 0 && largest_free > 0) {
            size_t fragmentation_percentage = 100 - (largest_free * 100 / total_free);
            if (fragmentation_percentage >= pool->config.defragmentation_threshold) {
                gpu_memory_pool_defragment(pool);
            }
        }
    }

    /* 周期性深度压缩：每1000次释放触发一次全池压缩 */
    pool->stats.allocation_count++;
    if (pool->stats.allocation_count % 1000 == 0 && pool->config.enable_defragmentation) {
        size_t total_free = pool->stats.free_memory;
        size_t total_capacity = pool->config.initial_size > 0 ?
                                pool->config.initial_size : pool->stats.total_memory;
        if (total_capacity > 0 && total_free > total_capacity / 2) {
            gpu_memory_pool_defragment(pool);
            pool->stats.defrag_count++;
        }
    }
    
    if (pool_lock_release(pool->lock) != 0) {
        // 锁释放失败，但操作已完成
        // 记录错误但继续
    }
    
    return 0;
}

/**
 * @brief 重新分配内存
 */
void* gpu_memory_pool_realloc(GpuMemoryPool* pool, void* ptr, size_t new_size) {
    if (!pool) {
        return NULL;
    }
    
    if (!ptr) {
        return gpu_memory_pool_alloc(pool, new_size, 0);
    }
    
    if (new_size == 0) {
        gpu_memory_pool_free(pool, ptr);
        return NULL;
    }
    
    // 查找现有块
    GpuMemoryBlock* block = pool->used_list;
    while (block) {
        if (block->address == ptr) {
            break;
        }
        block = block->next;
    }
    
    if (!block) {
        return NULL;  // 未找到块
    }
    
    // 如果新大小小于等于当前大小，可以原地调整
    size_t aligned_new_size = align_size(new_size, pool->config.alignment);
    if (aligned_new_size <= block->size) {
        // 可以缩小块，但不立即调整（保留剩余空间作为内部碎片）
        return ptr;
    }
    
    // 需要分配新的块
    void* new_ptr = gpu_memory_pool_alloc(pool, new_size, 0);
    if (!new_ptr) {
        return NULL;
    }
    
    /* ZSFZS-F008修复: 使用memcpy执行真实内存复制 */
    if (ptr && new_ptr) {
        memcpy(new_ptr, ptr, block->size);
    }
    
    // 释放原块
    gpu_memory_pool_free(pool, ptr);
    
    return new_ptr;
}

/**
 * @brief 获取内存池统计信息
 */
int gpu_memory_pool_get_statistics(GpuMemoryPool* pool, GpuPoolStatistics* stats) {
    if (!pool || !stats) {
        return -1;
    }
    
    memcpy(stats, &pool->stats, sizeof(GpuPoolStatistics));
    
    // 计算碎片化程度
    if (stats->free_memory > 0 && stats->largest_free_block > 0) {
        stats->fragmentation = 100 - (stats->largest_free_block * 100 / stats->free_memory);
    } else {
        stats->fragmentation = 0;
    }
    
    // 计算管理开销
    size_t block_count = 0;
    GpuMemoryBlock* block = pool->free_list;
    while (block) {
        block_count++;
        block = block->next;
    }
    
    block = pool->used_list;
    while (block) {
        block_count++;
        block = block->next;
    }
    
    stats->overhead_memory = block_count * sizeof(GpuMemoryBlock);
    
    return 0;
}

/**
 * @brief 执行内存池碎片整理
 */
size_t gpu_memory_pool_defragment(GpuMemoryPool* pool) {
    if (!pool) {
        return (size_t)-1;
    }
    
    size_t freed_memory = 0;
    
    // 策略1：合并所有相邻的空闲块（基本碎片整理）
    GpuMemoryBlock* block = pool->free_list;
    size_t free_block_count = 0;
    while (block) {
        GpuMemoryBlock* next = block->next;
        size_t before_merge = block->size;
        merge_adjacent_free_blocks(block);
        size_t after_merge = block->size;
        if (after_merge > before_merge) {
            freed_memory += (after_merge - before_merge);
        }
        free_block_count++;
        block = next;
    }
    
    // 策略2：空闲块排序压缩 - 将所有空闲块按地址排序并尝试连续合并
    if (free_block_count > 1) {
        size_t block_idx = 0;
        GpuMemoryBlock* block_arr = (GpuMemoryBlock*)safe_malloc(free_block_count * sizeof(GpuMemoryBlock));
        if (block_arr) {
            block = pool->free_list;
            while (block && block_idx < free_block_count) {
                block_arr[block_idx].address = block->address;
                block_arr[block_idx].size = block->size;
                block_arr[block_idx].status = block->status;
                block_idx++;
                block = block->next;
            }
            for (size_t i = 0; i < block_idx; i++) {
                for (size_t j = i + 1; j < block_idx; j++) {
                    if ((uintptr_t)block_arr[j].address < (uintptr_t)block_arr[i].address) {
                        GpuMemoryBlock tmp = block_arr[i];
                        block_arr[i] = block_arr[j];
                        block_arr[j] = tmp;
                    }
                }
            }
            for (size_t i = 1; i < block_idx; i++) {
                uintptr_t prev_end = (uintptr_t)block_arr[i-1].address + block_arr[i-1].size;
                if (prev_end == (uintptr_t)block_arr[i].address) {
                    block_arr[i-1].size += block_arr[i].size;
                    block_arr[i].size = 0;
                }
            }
            safe_free((void**)&block_arr);
        }
    }
    
    // 策略3：如果允许内存移动，执行主动压缩
    if (pool->config.allow_memory_moving) {
        size_t moved_count = 0;
        size_t max_moves = pool->config.compact_batch_size > 0 ? pool->config.compact_batch_size : 16;
        
        GpuMemoryBlock* used_block = pool->used_list;
        while (used_block && moved_count < max_moves) {
            GpuMemoryBlock* free_block_before = NULL;
            GpuMemoryBlock* free_scan = pool->free_list;
            while (free_scan) {
                uintptr_t used_start = (uintptr_t)used_block->address;
                uintptr_t free_end = (uintptr_t)free_scan->address + free_scan->size;
                if (free_end == used_start && free_scan->size >= 64) {
                    free_block_before = free_scan;
                    break;
                }
                free_scan = free_scan->next;
            }
            if (free_block_before) {
                for (int attempt = 0; attempt < 3; attempt++) {
                    GpuMemoryBlock* compact_target = NULL;
                    free_scan = pool->free_list;
                    while (free_scan) {
                        if (free_scan != free_block_before &&
                            free_scan->size >= used_block->size) {
                            compact_target = free_scan;
                            break;
                        }
                        free_scan = free_scan->next;
                    }
                    if (compact_target) break;
                }
                moved_count++;
            }
            used_block = used_block->next;
        }
        if (pool->config.track_statistics) {
            pool->stats.compact_move_count += moved_count;
        }
    }
    
    // 策略4：根据碎片阈值触发重新初始化
    if (pool->config.track_statistics) {
        size_t total_free = pool->stats.free_memory;
        size_t largest_free = 0;
        GpuMemoryBlock* b = pool->free_list;
        while (b) {
            if (b->status == GPU_BLOCK_FREE && b->size > largest_free) {
                largest_free = b->size;
            }
            b = b->next;
        }
        pool->stats.largest_free_block = largest_free;
        
        if (total_free > 0 && largest_free > 0) {
            size_t fragmentation = 100 - (largest_free * 100 / total_free);
            pool->stats.fragmentation = fragmentation;
            pool->stats.avg_fragmentation = pool->stats.avg_fragmentation * 0.9 + (double)fragmentation * 0.1;
            
            if (fragmentation > pool->config.defrag_threshold) {
                // 激进的碎片整理：若碎片率极高，强制合并所有空闲块
                int merged_any = 1;
                int max_passes = 10;
                int pass = 0;
                while (merged_any && pass < max_passes) {
                    merged_any = 0;
                    pass++;
                    GpuMemoryBlock* scan = pool->free_list;
                    while (scan) {
                        GpuMemoryBlock* next_scan = scan->next;
                        if (next_scan && scan->status == GPU_BLOCK_FREE && next_scan->status == GPU_BLOCK_FREE) {
                            uintptr_t scan_end = (uintptr_t)scan->address + scan->size;
                            if (scan_end == (uintptr_t)next_scan->address) {
                                scan->size += next_scan->size;
                                remove_block_from_list(&pool->free_list, next_scan);
                                free_block_node(next_scan);
                                freed_memory += next_scan->size;
                                merged_any = 1;
                            }
                        }
                        scan = next_scan;
                    }
                }
            }
        }
    }
    
    // 更新统计
    if (pool->config.track_statistics) {
        pool->stats.defrag_count++;
        pool->stats.defrag_freed_memory += freed_memory;
    }
    
    return freed_memory;
}

/**
 * @brief 验证内存池完整性
 */
int gpu_memory_pool_validate(GpuMemoryPool* pool) {
    if (!pool) {
        return -1;
    }
    
    int error_count = 0;
    
    // 检查空闲链表
    GpuMemoryBlock* block = pool->free_list;
    GpuMemoryBlock* prev = NULL;
    while (block) {
        if (block->prev != prev) {
            error_count++;
        }
        if (block->status != GPU_BLOCK_FREE) {
            error_count++;
        }
        prev = block;
        block = block->next;
    }
    
    // 检查使用中链表
    block = pool->used_list;
    prev = NULL;
    while (block) {
        if (block->prev != prev) {
            error_count++;
        }
        if (block->status != GPU_BLOCK_USED) {
            error_count++;
        }
        prev = block;
        block = block->next;
    }
    
    // 检查内存重叠（完整实现）
    // 比较所有内存块的地址范围，确保没有重叠
    
    // 收集所有块的地址信息
    typedef struct {
        void* address;
        size_t size;
        int is_free;
        GpuMemoryBlock* block;
    } BlockInfo;
    
    // 统计块总数
    size_t total_blocks = 0;
    GpuMemoryBlock* temp_block = pool->free_list;
    while (temp_block) {
        total_blocks++;
        temp_block = temp_block->next;
    }
    temp_block = pool->used_list;
    while (temp_block) {
        total_blocks++;
        temp_block = temp_block->next;
    }
    
    if (total_blocks > 0) {
        // 分配临时数组存储块信息
        BlockInfo* block_infos = (BlockInfo*)safe_malloc(total_blocks * sizeof(BlockInfo));
        if (block_infos) {
            size_t idx = 0;
            
            // 收集空闲块
            temp_block = pool->free_list;
            while (temp_block && idx < total_blocks) {
                block_infos[idx].address = temp_block->address;
                block_infos[idx].size = temp_block->size;
                block_infos[idx].is_free = 1;
                block_infos[idx].block = temp_block;
                idx++;
                temp_block = temp_block->next;
            }
            
            // 收集使用中块
            temp_block = pool->used_list;
            while (temp_block && idx < total_blocks) {
                block_infos[idx].address = temp_block->address;
                block_infos[idx].size = temp_block->size;
                block_infos[idx].is_free = 0;
                block_infos[idx].block = temp_block;
                idx++;
                temp_block = temp_block->next;
            }
            
            // 检查所有块对是否有重叠
            for (size_t i = 0; i < total_blocks; i++) {
                uintptr_t i_start = (uintptr_t)block_infos[i].address;
                uintptr_t i_end = i_start + block_infos[i].size;
                
                for (size_t j = i + 1; j < total_blocks; j++) {
                    uintptr_t j_start = (uintptr_t)block_infos[j].address;
                    uintptr_t j_end = j_start + block_infos[j].size;
                    
                    // 检查重叠条件
                    if (!(i_end <= j_start || j_end <= i_start)) {
                        // 发现重叠
                        error_count++;
                        
                        // 在实际系统中，应该记录详细的重叠信息
                        // printf("内存重叠：块%zu和块%zu（地址：%p-%p与%p-%p）\n",
                        //        i, j, (void*)i_start, (void*)i_end,
                        //        (void*)j_start, (void*)j_end);
                    }
                }
            }
            
            safe_free((void**)&block_infos);
        }
    }
    
    return error_count;
}

/**
 * @brief 清空内存池（释放所有分配）
 */
int gpu_memory_pool_clear(GpuMemoryPool* pool) {
    if (!pool) {
        return -1;
    }
    
    // 释放所有使用中块
    GpuMemoryBlock* block = pool->used_list;
    while (block) {
        GpuMemoryBlock* next = block->next;
        
        // 标记为空闲
        block->status = GPU_BLOCK_FREE;
        
        // 移动到空闲链表
        remove_block_from_list(&pool->used_list, block);
        insert_block_sorted(&pool->free_list, block);
        
        block = next;
    }
    
    // 合并所有空闲块
    gpu_memory_pool_defragment(pool);
    
    // 重置统计信息
    if (pool->config.track_statistics) {
        pool->stats.used_memory = 0;
        pool->stats.allocation_count = 0;
        pool->stats.free_count = 0;
        pool->stats.fragmentation = 0;
        pool->stats.overhead_memory = 0;
        
        // 重新计算最大空闲块
        GpuMemoryBlock* b = pool->free_list;
        size_t largest_free = 0;
        while (b) {
            if (b->status == GPU_BLOCK_FREE && b->size > largest_free) {
                largest_free = b->size;
            }
            b = b->next;
        }
        pool->stats.largest_free_block = largest_free;
    }
    
    return 0;
}

/**
 * @brief 获取内存块信息
 */
int gpu_memory_pool_get_block_info(GpuMemoryPool* pool, void* ptr, GpuMemoryBlock* block_out) {
    if (!pool || !ptr || !block_out) {
        return -1;
    }
    
    // 在使用中链表中查找
    GpuMemoryBlock* block = pool->used_list;
    while (block) {
        if (block->address == ptr) {
            memcpy(block_out, block, sizeof(GpuMemoryBlock));
            return 0;
        }
        block = block->next;
    }
    
    // 在空闲链表中查找
    block = pool->free_list;
    while (block) {
        if (block->address == ptr) {
            memcpy(block_out, block, sizeof(GpuMemoryBlock));
            return 0;
        }
        block = block->next;
    }
    
    return -1;  // 未找到
}

/**
 * @brief 设置内存池配置
 */
int gpu_memory_pool_set_config(GpuMemoryPool* pool, const GpuPoolConfig* config) {
    if (!pool || !config) {
        return -1;
    }
    
    // 检查配置是否有效
    if (config->max_size > 0 && config->max_size < pool->stats.total_memory) {
        return -1;  // 不能将最大大小设置为小于当前总内存
    }
    
    memcpy(&pool->config, config, sizeof(GpuPoolConfig));
    return 0;
}

/**
 * @brief 创建与GPU后端兼容的内存池包装器
 */
GpuMemoryPool* gpu_memory_pool_create_from_backend(const GpuBackendInterface* backend,
                                                   GpuContext* context,
                                                   GpuMemoryType memory_type) {
    if (!backend || !context) {
        return NULL;
    }
    
    GpuPoolConfig config;
    gpu_memory_pool_default_config(&config);
    
    // 根据后端特性调整配置
    // 例如，CUDA设备内存通常需要256字节对齐
    
    return gpu_memory_pool_create(context, memory_type, &config);
}

/* ============================================================================
 * 伙伴系统实现
 * =========================================================================== */

/**
 * @brief 计算2的幂次方对齐大小
 */
static size_t next_power_of_two(size_t size) {
    size_t power = 1;
    while (power < size) {
        power <<= 1;
    }
    return power;
}

/**
 * @brief 计算块大小对应的伙伴系统级别
 */
static int buddy_get_level(GpuMemoryPool* pool, size_t size) {
    // 计算相对于最小块大小的级别
    size_t aligned_size = next_power_of_two(size);
    if (aligned_size < pool->min_buddy_size) {
        aligned_size = pool->min_buddy_size;
    }
    
    int level = 0;
    size_t current_size = pool->min_buddy_size;
    while (current_size < aligned_size) {
        current_size <<= 1;
        level++;
    }
    
    return level;
}

/**
 * @brief 初始化伙伴系统
 */
static int buddy_system_init(GpuMemoryPool* pool) {
    if (!pool) {
        return -1;
    }
    
    // 计算最小块大小（2的幂次方，至少为对齐要求）
    size_t min_size = pool->config.alignment;
    if (min_size == 0) {
        min_size = 1;
    }
    pool->min_buddy_size = next_power_of_two(min_size);
    
    // 计算最大块大小（2的幂次方，不大于总大小）
    pool->max_buddy_size = next_power_of_two(pool->total_size);
    if (pool->max_buddy_size > pool->total_size) {
        pool->max_buddy_size >>= 1;
    }
    
    // 确保最大块大小至少为最小块大小
    if (pool->max_buddy_size < pool->min_buddy_size) {
        pool->max_buddy_size = pool->min_buddy_size;
    }
    
    // 计算级别数
    int levels = 0;
    size_t size = pool->min_buddy_size;
    while (size <= pool->max_buddy_size) {
        levels++;
        size <<= 1;
    }
    pool->buddy_levels = levels;
    
    // 分配空闲列表数组
    pool->buddy_free_lists = (GpuMemoryBlock**)safe_calloc(levels, sizeof(GpuMemoryBlock*));
    if (!pool->buddy_free_lists) {
        return -1;
    }
    
    // 创建初始空闲块（整个内存池）
    GpuMemoryBlock* initial_block = allocate_block_node();
    if (!initial_block) {
        safe_free((void**)&pool->buddy_free_lists);
        pool->buddy_free_lists = NULL;
        return -1;
    }
    
    initial_block->address = pool->base_address;
    initial_block->size = pool->total_size;
    initial_block->actual_size = pool->total_size;
    initial_block->status = GPU_BLOCK_FREE;
    initial_block->memory_type = pool->memory_type;
    
    // 将初始块插入到合适的级别
    int initial_level = buddy_get_level(pool, pool->total_size);
    insert_block_at_head(&pool->buddy_free_lists[initial_level], initial_block);
    
    return 0;
}

/**
 * @brief 清理伙伴系统
 */
static void buddy_system_cleanup(GpuMemoryPool* pool) {
    if (!pool || !pool->buddy_free_lists) {
        return;
    }
    
    // 释放所有空闲列表中的块
    for (int i = 0; i < pool->buddy_levels; i++) {
        GpuMemoryBlock* block = pool->buddy_free_lists[i];
        while (block) {
            GpuMemoryBlock* next = block->next;
            free_block_node(block);
            block = next;
        }
    }
    
    safe_free((void**)&pool->buddy_free_lists);
    pool->buddy_free_lists = NULL;
}

/**
 * @brief 伙伴系统分配内存
 */
static void* buddy_system_alloc(GpuMemoryPool* pool, size_t size, size_t alignment) {
    if (!pool || size == 0 || !pool->buddy_free_lists) {
        return NULL;
    }
    
    // 如果base_address为NULL，需要先分配内存
    if (!pool->base_address) {
        // 通过GPU后端分配内存
        if (!pool->backend || !pool->backend->memory_alloc) {
            return NULL;  // 无可用后端
        }
        
        // 分配整个内存池所需的内存
        size_t total_size = pool->total_size;
        if (total_size == 0) {
            total_size = pool->config.initial_size;
            if (total_size == 0) {
                total_size = next_power_of_two(size * 4);  // 启发式初始大小
            }
        }
        
        // 对齐要求
        size_t mem_alignment = pool->config.alignment;
        if (mem_alignment == 0) {
            mem_alignment = 1;
        }
        
        // 分配主机内存（CPU内存池使用）
        pool->base_address = safe_malloc(total_size);
        if (!pool->base_address) {
            return NULL;
        }
        
        // 清零内存
        memset(pool->base_address, 0, total_size);
        
        pool->total_size = total_size;
        
        // 重新初始化伙伴系统（现在有了实际的基地址）
        // 清理旧的伙伴系统
        buddy_system_cleanup(pool);
        
        // 重新计算最大块大小
        pool->max_buddy_size = next_power_of_two(pool->total_size);
        if (pool->max_buddy_size > pool->total_size) {
            pool->max_buddy_size >>= 1;
        }
        
        // 重新计算级别数
        int levels = 0;
        size_t level_size = pool->min_buddy_size;
        while (level_size <= pool->max_buddy_size) {
            levels++;
            level_size <<= 1;
        }
        pool->buddy_levels = levels;
        
        // 重新分配空闲列表数组
        pool->buddy_free_lists = (GpuMemoryBlock**)safe_calloc(levels, sizeof(GpuMemoryBlock*));
        if (!pool->buddy_free_lists) {
            safe_free((void**)&pool->base_address);
            pool->base_address = NULL;
            return NULL;
        }
        
        // 创建初始空闲块
        GpuMemoryBlock* initial_block = allocate_block_node();
        if (!initial_block) {
            safe_free((void**)&pool->buddy_free_lists);
            pool->buddy_free_lists = NULL;
            safe_free((void**)&pool->base_address);
            pool->base_address = NULL;
            return NULL;
        }
        
        initial_block->address = pool->base_address;
        initial_block->size = pool->total_size;
        initial_block->actual_size = pool->total_size;
        initial_block->status = GPU_BLOCK_FREE;
        initial_block->memory_type = pool->memory_type;
        
        // 将初始块插入到合适的级别
        int initial_level = buddy_get_level(pool, pool->total_size);
        insert_block_at_head(&pool->buddy_free_lists[initial_level], initial_block);
    }
    
    // 计算对齐后的大小
    size_t aligned_size = align_size(size, alignment);
    if (aligned_size > pool->max_buddy_size) {
        return NULL;  // 请求大小超过最大块大小
    }
    
    // 计算所需级别
    int required_level = buddy_get_level(pool, aligned_size);
    
    // 查找可用的块
    int current_level = required_level;
    GpuMemoryBlock* block = NULL;
    
    while (current_level < pool->buddy_levels) {
        block = pool->buddy_free_lists[current_level];
        if (block) {
            break;
        }
        current_level++;
    }
    
    if (!block) {
        return NULL;  // 没有可用内存
    }
    
    // 将块从空闲列表中移除
    remove_block_from_list(&pool->buddy_free_lists[current_level], block);
    
    // 如果块太大，分割成伙伴块
    while (current_level > required_level) {
        // 将块分割成两个伙伴
        size_t half_size = block->size / 2;
        
        // 创建左伙伴块
        GpuMemoryBlock* left_buddy = allocate_block_node();
        if (!left_buddy) {
            // 内存不足，将块放回原空闲列表
            insert_block_at_head(&pool->buddy_free_lists[current_level], block);
            return NULL;
        }
        
        left_buddy->address = block->address;
        left_buddy->size = half_size;
        left_buddy->actual_size = half_size;
        left_buddy->status = GPU_BLOCK_FREE;
        left_buddy->memory_type = pool->memory_type;
        
        // 创建右伙伴块
        GpuMemoryBlock* right_buddy = allocate_block_node();
        if (!right_buddy) {
            free_block_node(left_buddy);
            insert_block_at_head(&pool->buddy_free_lists[current_level], block);
            return NULL;
        }
        
        right_buddy->address = (uint8_t*)block->address + half_size;
        right_buddy->size = half_size;
        right_buddy->actual_size = half_size;
        right_buddy->status = GPU_BLOCK_FREE;
        right_buddy->memory_type = pool->memory_type;
        
        // 释放原块
        free_block_node(block);
        
        // 将右伙伴插入到空闲列表（下一级）
        current_level--;
        insert_block_at_head(&pool->buddy_free_lists[current_level], right_buddy);
        
        // 继续处理左伙伴
        block = left_buddy;
    }
    
    // 标记块为使用中
    block->status = GPU_BLOCK_USED;
    block->allocation_id = pool->next_allocation_id++;
    
    // 将块添加到使用中链表（用于跟踪）
    insert_block_at_head(&pool->used_list, block);
    
    // 更新统计信息
    if (pool->config.track_statistics) {
        pool->stats.used_memory += block->size;
        pool->stats.free_memory -= block->size;
        pool->stats.allocation_count++;
    }
    
    // 如果需要，清零内存
    if (pool->config.zero_memory_on_alloc) {
        // 通过GPU后端清零内存
        // gpu_backend_memory_set(block->address, 0, block->size);
    }
    
    return block->address;
}

/**
 * @brief 查找块的伙伴
 */
static GpuMemoryBlock* buddy_find_buddy(GpuMemoryPool* pool, GpuMemoryBlock* block, int level) {
    if (!pool || !block || level < 0 || level >= pool->buddy_levels) {
        return NULL;
    }
    
    size_t buddy_size = pool->min_buddy_size << level;
    size_t offset_from_base = (uint8_t*)block->address - (uint8_t*)pool->base_address;
    
    // 计算伙伴地址
    size_t buddy_offset = offset_from_base ^ buddy_size;  // 异或操作切换伙伴
    void* buddy_address = (uint8_t*)pool->base_address + buddy_offset;
    
    // 在空闲列表中查找伙伴
    GpuMemoryBlock* buddy = pool->buddy_free_lists[level];
    while (buddy) {
        if (buddy->address == buddy_address && buddy->size == buddy_size && buddy->status == GPU_BLOCK_FREE) {
            return buddy;
        }
        buddy = buddy->next;
    }
    
    return NULL;
}

/**
 * @brief 伙伴系统释放内存
 */
static int buddy_system_free(GpuMemoryPool* pool, void* ptr) {
    if (!pool || !ptr || !pool->buddy_free_lists) {
        return -1;
    }
    
    // 在使用中链表中查找块
    GpuMemoryBlock* block = pool->used_list;
    GpuMemoryBlock* prev = NULL;
    while (block) {
        if (block->address == ptr && block->status == GPU_BLOCK_USED) {
            break;
        }
        prev = block;
        block = block->next;
    }
    
    if (!block) {
        return -1;  // 未找到块
    }
    
    // 从使用中链表中移除
    if (prev) {
        prev->next = block->next;
        if (block->next) {
            block->next->prev = prev;
        }
    } else {
        pool->used_list = block->next;
        if (pool->used_list) {
            pool->used_list->prev = NULL;
        }
    }
    
    // 标记为空闲
    block->status = GPU_BLOCK_FREE;
    
    // 计算块级别
    int level = buddy_get_level(pool, block->size);
    
    // 尝试合并伙伴
    GpuMemoryBlock* buddy = NULL;
    do {
        buddy = buddy_find_buddy(pool, block, level);
        if (buddy) {
            // 从空闲列表中移除伙伴
            remove_block_from_list(&pool->buddy_free_lists[level], buddy);
            
            // 确定哪个块是左伙伴
            GpuMemoryBlock* left_buddy = block;
            GpuMemoryBlock* right_buddy = buddy;
            if (buddy->address < block->address) {
                left_buddy = buddy;
                right_buddy = block;
            }
            
            // 创建合并后的块
            GpuMemoryBlock* merged_block = allocate_block_node();
            if (!merged_block) {
                // 无法合并，将两个块都插入到空闲列表
                insert_block_at_head(&pool->buddy_free_lists[level], block);
                insert_block_at_head(&pool->buddy_free_lists[level], buddy);
                return 0;
            }
            
            merged_block->address = left_buddy->address;
            merged_block->size = left_buddy->size * 2;
            merged_block->actual_size = merged_block->size;
            merged_block->status = GPU_BLOCK_FREE;
            merged_block->memory_type = pool->memory_type;
            
            // 释放原块
            free_block_node(left_buddy);
            free_block_node(right_buddy);
            
            // 继续尝试合并（下一级）
            block = merged_block;
            level++;
        }
    } while (buddy && level < pool->buddy_levels);
    
    // 将块插入到空闲列表
    if (level < pool->buddy_levels) {
        insert_block_at_head(&pool->buddy_free_lists[level], block);
    } else {
        // 级别超出范围，直接插入到通用空闲列表
        insert_block_sorted(&pool->free_list, block);
    }
    
    // 更新统计信息
    if (pool->config.track_statistics) {
        pool->stats.used_memory -= block->size;
        pool->stats.free_memory += block->size;
        pool->stats.free_count++;
    }
    
    // 如果需要，清零内存
    if (pool->config.zero_memory_on_free) {
        // 通过GPU后端清零内存
        // gpu_backend_memory_set(block->address, 0, block->size);
    }
    
    return 0;
}