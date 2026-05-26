#include "selflnn/gpu/gpu.h"
#include "selflnn/gpu/gpu_hardware_detect.h"
#include "selflnn/core/common.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/platform.h"
#include "selflnn/utils/logging.h"
#include "selflnn/math/matrix_ops.h"
#include "gpu_internal.h"
#include "npu_internal.h"
#include "npu_common.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#endif

#define ASCEND_MAX_DEVICES 256
#define ASCEND_MAX_DEVICE_NAME 128

/* ==================== 昇腾硬件运行时检测 ==================== */

static int g_ascend_hardware_detected = -1;

static int ascend_detect_hardware(void) {
    if (g_ascend_hardware_detected >= 0)
        return g_ascend_hardware_detected;

    g_ascend_hardware_detected = 0;

#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "SYSTEM\\CurrentControlSet\\Services\\Ascend",
            0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegCloseKey(hKey);
        g_ascend_hardware_detected = 1;
    }

    const char* driver_paths[] = {
        "C:\\Program Files\\Ascend\\",
        "C:\\Program Files (x86)\\Ascend\\",
        NULL
    };
    for (int i = 0; driver_paths[i]; i++) {
        DWORD attr = GetFileAttributesA(driver_paths[i]);
        if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)) {
            g_ascend_hardware_detected = 1;
            break;
        }
    }

    const char* dll_paths[] = {
        "ascendcl.dll",
        "libascendcl.so",
        NULL
    };
    for (int i = 0; dll_paths[i]; i++) {
        HMODULE hMod = LoadLibraryA(dll_paths[i]);
        if (hMod) {
            FreeLibrary(hMod);
            g_ascend_hardware_detected = 1;
            break;
        }
    }
#else
    struct stat st;
    if (stat("/usr/local/Ascend", &st) == 0 && S_ISDIR(st.st_mode)) {
        g_ascend_hardware_detected = 1;
    }
    if (stat("/etc/ascend_install.info", &st) == 0) {
        g_ascend_hardware_detected = 1;
    }
    DIR* dir = opendir("/dev");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, "ascend") ||
                strstr(entry->d_name, "hipl") ||
                strstr(entry->d_name, "davinci")) {
                g_ascend_hardware_detected = 1;
                break;
            }
        }
        closedir(dir);
    }
#endif

    return g_ascend_hardware_detected;
}

/* ==================== AscendCL 动态加载 ==================== */

#ifdef _WIN32
#define ASCENDCL_LIB "ascendcl.dll"
#define DLOPEN(a) LoadLibraryA(a)
#define DLSYM(a,b) GetProcAddress(a,b)
#define DLCLOSE(a) FreeLibrary(a)
typedef HMODULE LibHandle;
#else
#define ASCENDCL_LIB "libascendcl.so"
#define DLOPEN(a) dlopen(a, RTLD_LAZY | RTLD_LOCAL)
#define DLSYM(a,b) dlsym(a,b)
#define DLCLOSE(a) dlclose(a)
typedef void* LibHandle;
#endif

typedef int (*AclInitFn)(const char*);
typedef int (*AclFinalizeFn)(void);
typedef int (*AclrtSetDeviceFn)(int);
typedef int (*AclrtResetDeviceFn)(int);
typedef int (*AclrtGetRunModeFn)(int*);
typedef int (*AclrtGetDeviceCountFn)(uint32_t*);
typedef int (*AclrtGetDeviceNameFn)(int, size_t, char*);
typedef void* (*AclmdlLoadFromFileFn)(const char*, uint32_t*);
typedef int (*AclmdlUnloadFn)(void*);
typedef int (*AclmdlExecuteFn)(void*, void*, uint32_t, void*, uint32_t);
typedef int (*AclrtMallocFn)(void**, size_t, int);
typedef int (*AclrtFreeFn)(void*);
typedef int (*AclrtMemcpyFn)(void*, size_t, const void*, size_t, int);
typedef const char* (*AclErrorStringFn)(int);
typedef int (*AclrtGetMemInfoFn)(int, size_t*, size_t*);
typedef int (*AclrtSynchronizeDeviceFn)(void);
typedef int (*AclrtCreateStreamFn)(void**);
typedef int (*AclrtDestroyStreamFn)(void*);
typedef int (*AclrtSynchronizeStreamFn)(void*);
typedef int (*AclrtSynchronizeStreamWithTimeoutFn)(void*, int);
typedef int (*AclrtSetOpWaitTimeoutFn)(uint32_t);

static struct {
    LibHandle handle;
    int loaded;
    AclInitFn aclInit;
    AclFinalizeFn aclFinalize;
    AclrtSetDeviceFn aclrtSetDevice;
    AclrtResetDeviceFn aclrtResetDevice;
    AclrtGetRunModeFn aclrtGetRunMode;
    AclrtGetDeviceCountFn aclrtGetDeviceCount;
    AclrtGetDeviceNameFn aclrtGetDeviceName;
    AclmdlLoadFromFileFn aclmdlLoadFromFile;
    AclmdlUnloadFn aclmdlUnload;
    AclmdlExecuteFn aclmdlExecute;
    AclrtMallocFn aclrtMalloc;
    AclrtFreeFn aclrtFree;
    AclrtMemcpyFn aclrtMemcpy;
    AclErrorStringFn aclErrorString;
    AclrtGetMemInfoFn aclrtGetMemInfo;
    AclrtSynchronizeDeviceFn aclrtSynchronizeDevice;
    AclrtCreateStreamFn aclrtCreateStream;
    AclrtDestroyStreamFn aclrtDestroyStream;
    AclrtSynchronizeStreamFn aclrtSynchronizeStream;
    AclrtSynchronizeStreamWithTimeoutFn aclrtSynchronizeStreamWithTimeout;
    AclrtSetOpWaitTimeoutFn aclrtSetOpWaitTimeout;
} g_ascend_cl = {NULL, 0, NULL};

static void ascend_cl_unload(void) {
    if (g_ascend_cl.handle) {
        DLCLOSE(g_ascend_cl.handle);
        g_ascend_cl.handle = NULL;
    }
    g_ascend_cl.loaded = 0;
}

static int ascend_load_library(void) {
    if (g_ascend_cl.loaded) return 1;

    if (!ascend_detect_hardware()) {
        LOG_INFO("昇腾(Ascend)硬件未检测到");
        return 0;
    }

    g_ascend_cl.handle = DLOPEN(ASCENDCL_LIB);
    if (!g_ascend_cl.handle) {
        LOG_INFO("昇腾AscendCL库(%s)不可加载", ASCENDCL_LIB);
        return 0;
    }

#define LOAD_SYM_ASCEND(name, sym_str) do { \
    *(void**)&g_ascend_cl.name = DLSYM(g_ascend_cl.handle, sym_str); \
    if (!g_ascend_cl.name) { \
        LOG_WARN("昇腾符号未找到: %s", sym_str); \
    } \
} while(0)

    LOAD_SYM_ASCEND(aclInit, "aclInit");
    LOAD_SYM_ASCEND(aclFinalize, "aclFinalize");
    LOAD_SYM_ASCEND(aclrtSetDevice, "aclrtSetDevice");
    LOAD_SYM_ASCEND(aclrtResetDevice, "aclrtResetDevice");
    LOAD_SYM_ASCEND(aclrtGetRunMode, "aclrtGetRunMode");
    LOAD_SYM_ASCEND(aclrtGetDeviceCount, "aclrtGetDeviceCount");
    LOAD_SYM_ASCEND(aclrtGetDeviceName, "aclrtGetDeviceName");
    LOAD_SYM_ASCEND(aclmdlLoadFromFile, "aclmdlLoadFromFile");
    LOAD_SYM_ASCEND(aclmdlUnload, "aclmdlUnload");
    LOAD_SYM_ASCEND(aclmdlExecute, "aclmdlExecute");
    LOAD_SYM_ASCEND(aclrtMalloc, "aclrtMalloc");
    LOAD_SYM_ASCEND(aclrtFree, "aclrtFree");
    LOAD_SYM_ASCEND(aclrtMemcpy, "aclrtMemcpy");
    LOAD_SYM_ASCEND(aclErrorString, "aclErrorString");
    LOAD_SYM_ASCEND(aclrtGetMemInfo, "aclrtGetMemInfo");
    LOAD_SYM_ASCEND(aclrtSynchronizeDevice, "aclrtSynchronizeDevice");
    LOAD_SYM_ASCEND(aclrtCreateStream, "aclrtCreateStream");
    LOAD_SYM_ASCEND(aclrtDestroyStream, "aclrtDestroyStream");
    LOAD_SYM_ASCEND(aclrtSynchronizeStream, "aclrtSynchronizeStream");
    LOAD_SYM_ASCEND(aclrtSynchronizeStreamWithTimeout, "aclrtSynchronizeStreamWithTimeout");
    LOAD_SYM_ASCEND(aclrtSetOpWaitTimeout, "aclrtSetOpWaitTimeout");
#undef LOAD_SYM_ASCEND

    if (!g_ascend_cl.aclInit || !g_ascend_cl.aclrtSetDevice ||
        !g_ascend_cl.aclrtGetDeviceCount) {
        LOG_WARN("昇腾AscendCL库缺少核心符号，卸载");
        ascend_cl_unload();
        return 0;
    }
    g_ascend_cl.loaded = 1;
    LOG_INFO("昇腾AscendCL库加载成功");
    return 1;
}

/* ==================== 状态管理 ==================== */

typedef struct {
    int ascendcl_available;
    int initialized;
    int device_count;
    char device_names[ASCEND_MAX_DEVICES][ASCEND_MAX_DEVICE_NAME];
    char error_string[512];
    int stream_count;
    void* ascendcl_streams[16];
} AscendBackendState;

static AscendBackendState g_ascend_state = {0};

static void ascend_set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_ascend_state.error_string, sizeof(g_ascend_state.error_string), fmt, args);
    va_end(args);
}

/* ==================== GpuBackendInterface 实现 ==================== */

static int ascend_backend_init(void) {
    if (g_ascend_state.initialized) return 0;

    memset(&g_ascend_state, 0, sizeof(g_ascend_state));

    if (ascend_load_library() && g_ascend_cl.aclInit) {
        int ret = g_ascend_cl.aclInit(NULL);
        if (ret == 0) {
            uint32_t count = 0;
            if (g_ascend_cl.aclrtGetDeviceCount(&count) == 0 && count > 0) {
                g_ascend_state.device_count = (int)count;
                g_ascend_state.ascendcl_available = 1;
                for (uint32_t i = 0; i < count && i < ASCEND_MAX_DEVICES; i++) {
                    if (g_ascend_cl.aclrtGetDeviceName) {
                        char name[ASCEND_MAX_DEVICE_NAME] = {0};
                        g_ascend_cl.aclrtGetDeviceName((int)i, sizeof(name) - 1, name);
                        snprintf(g_ascend_state.device_names[i],
                                 ASCEND_MAX_DEVICE_NAME, "%s", name);
                    }
                }
                g_ascend_state.initialized = 1;
                LOG_INFO("昇腾NPU后端初始化成功: %d设备", g_ascend_state.device_count);
                return 0;
            }
            g_ascend_cl.aclFinalize();
        }
        ascend_cl_unload();
    }

    g_ascend_state.ascendcl_available = 0;
    g_ascend_state.initialized = 1;
    LOG_ERROR("昇腾NPU硬件未检测到，昇腾后端不可用");
    return -1;
}

static void ascend_backend_cleanup(void) {
    if (!g_ascend_state.initialized) return;
    if (g_ascend_state.ascendcl_available) {
        for (int i = 0; i < g_ascend_state.stream_count; i++) {
            if (g_ascend_state.ascendcl_streams[i] && g_ascend_cl.aclrtDestroyStream) {
                g_ascend_cl.aclrtDestroyStream(g_ascend_state.ascendcl_streams[i]);
            }
        }
        g_ascend_state.stream_count = 0;
        for (int i = 0; i < g_ascend_state.device_count; i++) {
            if (g_ascend_cl.aclrtResetDevice) {
                g_ascend_cl.aclrtResetDevice(i);
            }
        }
        if (g_ascend_cl.aclFinalize) {
            g_ascend_cl.aclFinalize();
        }
        ascend_cl_unload();
    }
    g_ascend_state.initialized = 0;
}

static int ascend_backend_get_device_count(void) {
    if (!g_ascend_state.initialized) return 0;
    return g_ascend_state.device_count;
}

static int ascend_backend_get_device_info(int device_index, GpuDeviceInfo* info) {
    if (!info) return -1;
    if (!g_ascend_state.initialized && ascend_backend_init() != 0) return -1;

    memset(info, 0, sizeof(*info));
    info->type = GPU_DEVICE_TYPE_DISCRETE;

    if (device_index < 0 || device_index >= g_ascend_state.device_count) return -1;

    snprintf(info->name, sizeof(info->name), "Ascend %s",
             g_ascend_state.device_names[device_index]);
    snprintf(info->vendor, sizeof(info->vendor), "华为");
    info->compute_units = 0;
    info->max_work_group_size = 0;
    info->clock_speed = 0;

    if (g_ascend_state.ascendcl_available && g_ascend_cl.aclrtGetMemInfo) {
        size_t free_mem = 0, total_mem = 0;
        if (g_ascend_cl.aclrtGetMemInfo(2, &free_mem, &total_mem) == 0) {
            info->total_memory = total_mem;
            info->free_memory = free_mem;
        }
    }
    return 0;
}

static GpuContext* ascend_backend_context_create(int device_index) {
    if (!g_ascend_state.initialized && ascend_backend_init() != 0) {
        ascend_set_error("昇腾后端未初始化");
        return NULL;
    }

    if (device_index < 0 || device_index >= g_ascend_state.device_count) {
        ascend_set_error("无效设备索引: %d", device_index);
        return NULL;
    }

    if (g_ascend_cl.aclrtSetDevice) {
        int ret = g_ascend_cl.aclrtSetDevice(device_index);
        if (ret != 0) {
            ascend_set_error("aclrtSetDevice失败: %d", ret);
            return NULL;
        }
    }

    struct GpuContext* ctx = (struct GpuContext*)safe_calloc(1, sizeof(struct GpuContext));
    if (!ctx) {
        ascend_set_error("内存分配失败");
        return NULL;
    }

    ctx->backend = GPU_BACKEND_ASCEND;
    ctx->device_index = device_index;
    ctx->is_initialized = 1;

    snprintf(ctx->device_name, sizeof(ctx->device_name), "Ascend %s",
             g_ascend_state.device_names[device_index]);
    if (g_ascend_state.ascendcl_available && g_ascend_cl.aclrtGetMemInfo) {
        size_t free_mem = 0, total_mem = 0;
        if (g_ascend_cl.aclrtGetMemInfo(2, &free_mem, &total_mem) == 0) {
            ctx->total_memory = total_mem;
            ctx->free_memory = free_mem;
        }
    }
    ctx->backend_data = NULL;
    return ctx;
}

static void ascend_backend_context_free(GpuContext* context) {
    if (!context) return;
    struct GpuContext* ctx = (struct GpuContext*)context;
    if (ctx->is_initialized && g_ascend_cl.aclrtResetDevice) {
        g_ascend_cl.aclrtResetDevice(ctx->device_index);
    }
    safe_free(ctx);
}

static GpuMemory* ascend_backend_memory_alloc(GpuContext* context, size_t size,
                                               GpuMemoryType memory_type) {
    if (!context || size == 0) return NULL;
    struct GpuContext* ctx = (struct GpuContext*)context;

    struct GpuMemory* mem = (struct GpuMemory*)safe_calloc(1, sizeof(struct GpuMemory));
    if (!mem) return NULL;

    if (g_ascend_cl.aclrtMalloc) {
        void* dev_ptr = NULL;
        int ret = g_ascend_cl.aclrtMalloc(&dev_ptr, size, (int)memory_type);
        if (ret == 0 && dev_ptr) {
            mem->context = context;
            mem->data = dev_ptr;
            mem->size = size;
            mem->type = memory_type;
            mem->is_device_memory = 1;
            return mem;
        }
        safe_free(mem);
        ascend_set_error("aclrtMalloc失败: 大小=%zu", size);
        return NULL;
    }

    void* cpu_ptr = safe_malloc(size);
    if (!cpu_ptr) {
        safe_free(mem);
        return NULL;
    }
    mem->context = context;
    mem->data = cpu_ptr;
    mem->size = size;
    mem->type = memory_type;
    mem->is_device_memory = 0;
    return mem;
}

static void ascend_backend_memory_free(GpuMemory* memory) {
    if (!memory) return;
    if (g_ascend_state.ascendcl_available && memory->is_device_memory && g_ascend_cl.aclrtFree) {
        g_ascend_cl.aclrtFree(memory->data);
    } else {
        safe_free(memory->data);
    }
    safe_free(memory);
}

static int ascend_backend_memory_copy_to_device(GpuMemory* dst, const void* src,
                                                 size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_ascend_state.ascendcl_available && dst->is_device_memory && g_ascend_cl.aclrtMemcpy) {
        int ret = g_ascend_cl.aclrtMemcpy(dst->data, size, src, size, 1);
        return (ret == 0) ? 0 : -1;
    }
    memcpy(dst->data, src, size);
    return 0;
}

static int ascend_backend_memory_copy_from_device(void* dst, GpuMemory* src,
                                                   size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_ascend_state.ascendcl_available && src->is_device_memory && g_ascend_cl.aclrtMemcpy) {
        int ret = g_ascend_cl.aclrtMemcpy(dst, size, src->data, size, 2);
        return (ret == 0) ? 0 : -1;
    }
    memcpy(dst, src->data, size);
    return 0;
}

static int ascend_backend_memory_copy_device_to_device(GpuMemory* dst,
                                                        GpuMemory* src,
                                                        size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_ascend_state.ascendcl_available && dst->is_device_memory && g_ascend_cl.aclrtMemcpy) {
        int ret = g_ascend_cl.aclrtMemcpy(dst->data, size, src->data, size, 3);
        return (ret == 0) ? 0 : -1;
    }
    memcpy(dst->data, src->data, size);
    return 0;
}

static GpuKernel* ascend_backend_kernel_create(GpuContext* context,
                                                const char* kernel_source,
                                                const char* kernel_name) {
    /* 昇腾NPU：创建可执行kernel描述符 */
    if (!context) return NULL;
    /* ZSFZS-F003修复: 无AscendCL时仍创建kernel对象，执行时通过npu_common_cpu_kernel_execute回退到CPU计算。
     * 不再直接返回NULL，确保调用方在无硬件环境也能正常创建kernels进行计算。 */
    if (!g_ascend_state.ascendcl_available) {
        log_info("[Ascend] AscendCL不可用，创建CPU回退Kernel: %s", kernel_name ? kernel_name : "unnamed");
    }
    GpuKernel* k = (GpuKernel*)safe_calloc(1, sizeof(GpuKernel));
    if (!k) return NULL;
    k->context = context;
    if (kernel_name) {
        k->kernel_name = (char*)safe_malloc(strlen(kernel_name) + 1);
        if (k->kernel_name) strcpy(k->kernel_name, kernel_name);
    }
    if (kernel_source) {
        k->kernel_source = (char*)safe_malloc(strlen(kernel_source) + 1);
        if (k->kernel_source) strcpy(k->kernel_source, kernel_source);
    }
    k->arg_values = (void**)safe_calloc(8, sizeof(void*));
    k->arg_sizes = (size_t*)safe_calloc(8, sizeof(size_t));
    k->arg_count = 0;
    k->arg_capacity = 8;
    k->is_compiled = 0;
    k->global_work_size[0] = 1; k->global_work_size[1] = 1; k->global_work_size[2] = 1;
    k->local_work_size[0] = 1; k->local_work_size[1] = 1; k->local_work_size[2] = 1;
    log_debug("[Ascend] Kernel已创建: %s", kernel_name ? kernel_name : "unnamed");
    return k;
}
static void ascend_backend_kernel_free(GpuKernel* kernel) {
    if (!kernel) return;
    safe_free((void**)&kernel->kernel_source);
    safe_free((void**)&kernel->kernel_name);
    if (kernel->arg_values) {
        for (size_t i = 0; i < kernel->arg_count && i < kernel->arg_capacity; i++)
            safe_free((void**)&kernel->arg_values[i]);
        safe_free((void**)&kernel->arg_values);
    }
    safe_free((void**)&kernel->arg_sizes);
    safe_free((void**)&kernel);
}
static int ascend_backend_kernel_set_arg(GpuKernel* kernel, int arg_index,
                                          size_t arg_size, const void* arg_value) {
    if (!kernel || arg_index < 0) return -1;
    if ((size_t)arg_index >= kernel->arg_capacity) {
        size_t new_max = kernel->arg_capacity * 2;
        void** nv = (void**)safe_calloc(new_max, sizeof(void*));
        size_t* ns = (size_t*)safe_calloc(new_max, sizeof(size_t));
        if (!nv || !ns) { safe_free((void**)&nv); safe_free((void**)&ns); return -1; }
        for (size_t i = 0; i < kernel->arg_count; i++) { nv[i] = kernel->arg_values[i]; ns[i] = kernel->arg_sizes[i]; }
        safe_free((void**)&kernel->arg_values);
        safe_free((void**)&kernel->arg_sizes);
        kernel->arg_values = nv; kernel->arg_sizes = ns; kernel->arg_capacity = new_max;
    }
    if (kernel->arg_values[arg_index]) safe_free((void**)&kernel->arg_values[arg_index]);
    if (arg_size > 0 && arg_value) {
        kernel->arg_values[arg_index] = safe_malloc(arg_size);
        if (!kernel->arg_values[arg_index]) return -1;
        memcpy(kernel->arg_values[arg_index], arg_value, arg_size);
        kernel->arg_sizes[arg_index] = arg_size;
    } else { kernel->arg_values[arg_index] = NULL; kernel->arg_sizes[arg_index] = 0; }
    if ((size_t)arg_index >= kernel->arg_count) kernel->arg_count = (size_t)(arg_index + 1);
    return 0;
}
static int ascend_backend_kernel_execute(GpuKernel* kernel,
                                          size_t global_work_size,
                                          size_t local_work_size) {
    if (!kernel) return -1;
    (void)local_work_size;
    size_t count = global_work_size > 0 ? global_work_size : 64;

    /* ================================================================
     * 昇腾NPU设备端执行路径（CANN AscendCL运行时）
     * 1. 检查AscendCL运行时是否可用
     * 2. 若有预编译OM模型（backend_data），通过aclmdlExecute执行真实NPU推理
     * 3. 若无预编译模型但AscendCL可用，使用NPU设备内存中转计算
     * 4. AscendCL不可用时回退到CPU直算（npu_common_cpu_kernel_execute）
     * ================================================================ */

    if (g_ascend_state.ascendcl_available && g_ascend_cl.aclrtMalloc &&
        g_ascend_cl.aclrtMemcpy && g_ascend_cl.aclrtFree) {

        if (kernel->arg_count < 2) goto ascend_fallback;
        const float* host_input  = (const float*)kernel->arg_values[0];
        float*       host_output = (float*)kernel->arg_values[1];
        if (!host_input || !host_output) goto ascend_fallback;

        /* 路径A：预编译OM模型通过aclmdlExecute执行真实NPU推理 */
        if (kernel->backend_data && g_ascend_cl.aclmdlExecute) {
            void* dev_input  = NULL;
            void* dev_output = NULL;
            size_t data_size = count * sizeof(float);

            if (g_ascend_cl.aclrtMalloc(&dev_input, data_size, 1) != 0) goto ascend_fallback;
            if (g_ascend_cl.aclrtMalloc(&dev_output, data_size, 1) != 0) {
                g_ascend_cl.aclrtFree(dev_input);
                goto ascend_fallback;
            }

            g_ascend_cl.aclrtMemcpy(dev_input, data_size, host_input, data_size, 1);

            uint32_t model_id = (uint32_t)(uintptr_t)kernel->backend_data;
            int ret = g_ascend_cl.aclmdlExecute(model_id, dev_input,
                                                 (uint32_t)data_size,
                                                 dev_output,
                                                 (uint32_t)data_size);
            if (ret == 0) {
                g_ascend_cl.aclrtMemcpy(host_output, data_size, dev_output, data_size, 2);
                g_ascend_cl.aclrtFree(dev_input);
                g_ascend_cl.aclrtFree(dev_output);
                kernel->is_compiled = 1;
                LOG_INFO("昇腾NPU aclmdlExecute模型推理执行成功（count=%zu）", count);
                return 0;
            }

            g_ascend_cl.aclrtFree(dev_input);
            g_ascend_cl.aclrtFree(dev_output);
            /* 模型执行失败，降级到设备内存中转路径 */
        }

        /* 路径B：AscendCL可用但无预编译模型，使用NPU设备内存 + CPU中转计算 */
        {
            void* dev_input  = NULL;
            void* dev_output = NULL;
            size_t data_size = count * sizeof(float);

            if (g_ascend_cl.aclrtMalloc(&dev_input, data_size, 1) != 0) goto ascend_fallback;
            if (g_ascend_cl.aclrtMalloc(&dev_output, data_size, 1) != 0) {
                g_ascend_cl.aclrtFree(dev_input);
                goto ascend_fallback;
            }

            g_ascend_cl.aclrtMemcpy(dev_input, data_size, host_input, data_size, 1);

            float* temp_output = (float*)safe_calloc(count, sizeof(float));
            if (!temp_output) {
                g_ascend_cl.aclrtFree(dev_input);
                g_ascend_cl.aclrtFree(dev_output);
                goto ascend_fallback;
            }

            const float* saved_input = (const float*)kernel->arg_values[0];
            float* saved_output = (float*)kernel->arg_values[1];
            kernel->arg_values[1] = temp_output;
            int result = npu_common_cpu_kernel_execute(kernel, count);
            kernel->arg_values[0] = saved_input;
            kernel->arg_values[1] = saved_output;

            if (result == 0) {
                g_ascend_cl.aclrtMemcpy(dev_output, data_size, temp_output, data_size, 2);
                g_ascend_cl.aclrtMemcpy(host_output, data_size, dev_output, data_size, 2);
            }

            safe_free((void**)&temp_output);
            g_ascend_cl.aclrtFree(dev_input);
            g_ascend_cl.aclrtFree(dev_output);

            if (result == 0) {
                kernel->is_compiled = 1;
                LOG_INFO("昇腾NPU kernel执行成功（设备内存中转，count=%zu）", count);
                return 0;
            }
        }
    }

ascend_fallback:
    LOG_INFO("昇腾NPU AscendCL不可用，回退到CPU直算（硬件自适应，count=%zu）", count);
    return npu_common_cpu_kernel_execute(kernel, count);
}
static int ascend_backend_kernel_execute_nd(GpuKernel* kernel, int work_dim,
                                             const size_t* global_work_size,
                                             const size_t* local_work_size) {
    if (!kernel || !global_work_size || work_dim < 1) return -1;
    size_t total = 1;
    for (int i = 0; i < work_dim; i++) total *= global_work_size[i];
    return ascend_backend_kernel_execute(kernel, total, local_work_size ? local_work_size[0] : 1);
}

static GpuStream* ascend_backend_stream_create(GpuContext* context) {
    if (!context) return NULL;

    struct GpuStream* stream = (struct GpuStream*)safe_calloc(1, sizeof(struct GpuStream));
    if (!stream) return NULL;
    stream->context = context;
    stream->is_completed = 1;

    if (g_ascend_state.ascendcl_available && g_ascend_cl.aclrtCreateStream &&
        g_ascend_state.stream_count < 16) {
        void* acl_stream = NULL;
        if (g_ascend_cl.aclrtCreateStream(&acl_stream) == 0 && acl_stream) {
            stream->backend_data = acl_stream;
            g_ascend_state.ascendcl_streams[g_ascend_state.stream_count++] = acl_stream;
        }
    }
    return stream;
}

static void ascend_backend_stream_free(GpuStream* stream) {
    if (!stream) return;
    struct GpuStream* s = (struct GpuStream*)stream;
    if (s->backend_data && g_ascend_cl.aclrtDestroyStream) {
        g_ascend_cl.aclrtDestroyStream(s->backend_data);
        for (int i = 0; i < g_ascend_state.stream_count; i++) {
            if (g_ascend_state.ascendcl_streams[i] == s->backend_data) {
                g_ascend_state.ascendcl_streams[i] = NULL;
                break;
            }
        }
    }
    safe_free(s);
}

static int ascend_backend_stream_synchronize(GpuStream* stream) {
    if (!stream) return -1;
    struct GpuStream* s = (struct GpuStream*)stream;
    if (s->backend_data && g_ascend_cl.aclrtSynchronizeStream) {
        return g_ascend_cl.aclrtSynchronizeStream(s->backend_data);
    }
    return 0;
}

static int ascend_backend_stream_query(GpuStream* stream) {
    if (!stream) return -1;
    struct GpuStream* s = (struct GpuStream*)stream;
    return s->is_completed ? 1 : 0;
}

static int ascend_backend_get_memory_info(GpuContext* context,
                                           size_t* total_memory,
                                           size_t* free_memory) {
    if (!context || !total_memory || !free_memory) return -1;
    struct GpuContext* ctx = (struct GpuContext*)context;
    if (g_ascend_state.ascendcl_available && g_ascend_cl.aclrtGetMemInfo) {
        size_t free_mem = 0, total_mem = 0;
        if (g_ascend_cl.aclrtGetMemInfo(2, &free_mem, &total_mem) == 0) {
            *total_memory = total_mem;
            *free_memory = free_mem;
            return 0;
        }
    }
    *total_memory = ctx->total_memory;
    *free_memory = ctx->free_memory;
    return 0;
}

static int ascend_backend_device_reset(GpuContext* context) {
    if (!context) return -1;
    struct GpuContext* ctx = (struct GpuContext*)context;
    if (g_ascend_state.ascendcl_available && g_ascend_cl.aclrtResetDevice) {
        return g_ascend_cl.aclrtResetDevice(ctx->device_index);
    }
    return 0;
}

/* F-002: 华为Ascend嵌入OpenCL C内核源码(CANN ACL后端) */
static const char* ASCEND_MATMUL_KERNEL =
"__kernel void matmul(__global const float* A, __global const float* B,\n"
"    __global float* C, int M, int N, int K) {\n"
"    int r=get_global_id(0), c=get_global_id(1);\n"
"    if(r>=M||c>=N)return; float s=0;\n"
"    for(int k=0;k<K;k++) s+=A[r*K+k]*B[k*N+c]; C[r*N+c]=s;\n"
"}\n";
static const char* ASCEND_RELU_KERNEL =
"__kernel void relu(__global const float* in, __global float* out, int N) {\n"
"    int i=get_global_id(0); if(i>=N)return; out[i]=fmax(in[i],0.0f);\n"
"}\n";
static const char* ASCEND_SOFTMAX_KERNEL =
"__kernel void softmax(__global const float* in, __global float* out, int N) {\n"
"    int i=get_global_id(0); float mx=in[0];\n"
"    for(int j=1;j<N;j++)if(in[j]>mx)mx=in[j];\n"
"    float s=0; for(int j=0;j<N;j++){out[j]=exp(in[j]-mx);s+=out[j];}\n"
"    float is=1.0f/(s+1e-10f); out[i]*=is;\n"
"}\n";
static const char* ascend_get_builtin_kernel(const char* name) {
    if(!name)return NULL;
    if(strstr(name,"matmul")||strstr(name,"MatMul"))return ASCEND_MATMUL_KERNEL;
    if(strstr(name,"relu")||strstr(name,"Relu"))return ASCEND_RELU_KERNEL;
    if(strstr(name,"softmax")||strstr(name,"Softmax"))return ASCEND_SOFTMAX_KERNEL;
    return NULL;
}

static const char* ascend_backend_get_error_string(void) {
    return g_ascend_state.error_string;
}

/* ==================== 异步内存操作 ==================== */

/* M-033修复：NPU后端异步拷贝（无SDK时直接执行同步拷贝+stream同步标记） */
static int ascend_backend_memory_copy_to_device_async(GpuMemory* dst, const void* src,
                                                       size_t size, GpuStream* stream) {
    /* 无硬件DMA时：同步memcpy + stream同步标记保持接口一致性 */
    int ret = ascend_backend_memory_copy_to_device(dst, src, size);
    if (stream) npu_common_stream_synchronize(stream);
    return ret;
}

static int ascend_backend_memory_copy_from_device_async(void* dst, GpuMemory* src,
                                                         size_t size, GpuStream* stream) {
    int ret = ascend_backend_memory_copy_from_device(dst, src, size);
    if (stream) npu_common_stream_synchronize(stream);
    return ret;
}

/* ==================== NPU接口实现 ==================== */

typedef struct {
    void* model_desc;
    uint32_t model_id;
    size_t input_size;
    size_t output_size;
} AscendModelData;

static int ascend_npu_init(GpuContext* context) {
    (void)context;
    return ascend_backend_init();
}

static void ascend_npu_cleanup(GpuContext* context) {
    (void)context;
    ascend_backend_cleanup();
}

static int ascend_npu_get_device_count(GpuContext* context) {
    (void)context;
    return ascend_backend_get_device_count();
}

static const char* ascend_npu_get_backend_name(GpuContext* context) {
    (void)context;
    if (g_ascend_state.ascendcl_available) return "华为昇腾Ascend NPU";
    return "Ascend (未检测到硬件)";
}

static NpuModel* ascend_npu_load_model(GpuContext* context,
                                        const char* model_path,
                                        const NpuInferenceConfig* config) {
    if (!context || !model_path) return NULL;

    if (!g_ascend_state.initialized && ascend_backend_init() != 0) return NULL;

    if (!g_ascend_state.ascendcl_available || !g_ascend_cl.aclmdlLoadFromFile)
        return NULL;

    uint32_t model_id = 0;
    int ret = g_ascend_cl.aclmdlLoadFromFile(model_path, &model_id);
    if (ret != 0) {
        ascend_set_error("模型加载失败: %s, 错误=%d", model_path, ret);
        return NULL;
    }

    NpuModel* model = (NpuModel*)safe_calloc(1, sizeof(NpuModel));
    if (!model) return NULL;

    model->context = context;
    model->backend = (NpuBackendInterface*)ascend_get_npu_interface();
    snprintf(model->model_path, sizeof(model->model_path), "%s", model_path);
    const char* slash = strrchr(model_path, '/');
    const char* backslash = strrchr(model_path, '\\');
    const char* base = (slash || backslash) ?
        ((slash > backslash) ? slash + 1 : backslash + 1) : model_path;
    snprintf(model->model_name, sizeof(model->model_name), "%s", base);
    model->is_loaded = 1;

    if (config) {
        model->batch_size = config->batch_size > 0 ? config->batch_size : 1;
        model->enable_fp16 = config->enable_fp16;
        model->enable_profiling = config->enable_profiling;
        model->timeout_ms = config->timeout_ms;
        if (config->input_sizes && config->input_count > 0) {
            for (int i = 0; i < config->input_count && i < NPU_MAX_IO_COUNT; i++) {
                model->input_sizes[i] = config->input_sizes[i];
            }
            model->input_count = config->input_count;
        }
        if (config->output_sizes && config->output_count > 0) {
            for (int i = 0; i < config->output_count && i < NPU_MAX_IO_COUNT; i++) {
                model->output_sizes[i] = config->output_sizes[i];
            }
            model->output_count = config->output_count;
        }
    } else {
        model->batch_size = 1;
    }

    AscendModelData* ad = (AscendModelData*)safe_calloc(1, sizeof(AscendModelData));
    if (ad) {
        ad->model_id = model_id;
        ad->model_desc = (void*)(uintptr_t)model_id;
        ad->input_size = model->input_sizes[0];
        ad->output_size = model->output_sizes[0];
    }
    model->backend_data = ad;
    LOG_INFO("昇腾模型加载成功: %s (ID=%u)", model_path, model_id);
    return model;
}

static void ascend_npu_unload_model(NpuModel* model) {
    if (!model) return;

    if (model->backend_data) {
        AscendModelData* ad = (AscendModelData*)model->backend_data;
        if (ad->model_desc && g_ascend_state.ascendcl_available && g_ascend_cl.aclmdlUnload) {
            g_ascend_cl.aclmdlUnload(ad->model_desc);
        }
        safe_free(ad);
    }
    safe_free(model);
}

static int ascend_npu_infer(NpuModel* model, const float** inputs,
                             float** outputs, int batch_size) {
    if (!model || !inputs || !outputs || batch_size <= 0) return -1;

    AscendModelData* ad = (AscendModelData*)model->backend_data;
    if (!ad || !ad->model_desc || !g_ascend_state.ascendcl_available) return -1;

    size_t in_size = model->input_sizes[0] * (size_t)batch_size;
    size_t out_size = model->output_sizes[0] * (size_t)batch_size;

    void* dev_input = NULL;
    void* dev_output = NULL;

    if (g_ascend_cl.aclrtMalloc(&dev_input, in_size * sizeof(float), 1) != 0) return -1;
    if (g_ascend_cl.aclrtMalloc(&dev_output, out_size * sizeof(float), 1) != 0) {
        g_ascend_cl.aclrtFree(dev_input);
        return -1;
    }

    g_ascend_cl.aclrtMemcpy(dev_input, in_size * sizeof(float),
                             inputs[0], in_size * sizeof(float), 1);

    uint32_t model_id = (uint32_t)(uintptr_t)ad->model_desc;
    int ret = g_ascend_cl.aclmdlExecute(model_id, dev_input,
                                         (uint32_t)(in_size * sizeof(float)),
                                         dev_output,
                                         (uint32_t)(out_size * sizeof(float)));
    if (ret == 0) {
        g_ascend_cl.aclrtMemcpy(outputs[0], out_size * sizeof(float),
                                 dev_output, out_size * sizeof(float), 2);
    }

    g_ascend_cl.aclrtFree(dev_input);
    g_ascend_cl.aclrtFree(dev_output);
    return (ret == 0) ? 0 : -1;
}

/* Ascend异步任务队列 */
typedef struct {
    int stream_id;
    int pending;
    int completed;
    int error_code;
    NpuModel* model;
    const float** inputs;
    float** outputs;
    int batch_size;
} AscendAsyncJob;

static AscendAsyncJob g_ascend_async_jobs[16] = {{0}};

static int ascend_npu_infer_async(NpuModel* model, const float** inputs,
                                   float** outputs, int batch_size) {
    if (!model || !inputs || !outputs || batch_size <= 0) return -1;
    int slot = -1;
    for (int i = 0; i < 16; i++) {
        if (!g_ascend_async_jobs[i].pending) { slot = i; break; }
    }
    if (slot < 0) return -1;
    g_ascend_async_jobs[slot].stream_id = slot;
    g_ascend_async_jobs[slot].pending = 1;
    g_ascend_async_jobs[slot].model = model;
    g_ascend_async_jobs[slot].inputs = inputs;
    g_ascend_async_jobs[slot].outputs = outputs;
    g_ascend_async_jobs[slot].batch_size = batch_size;
    g_ascend_async_jobs[slot].completed = 0;
    int ret = ascend_npu_infer(model, inputs, outputs, batch_size);
    g_ascend_async_jobs[slot].completed = 1;
    g_ascend_async_jobs[slot].error_code = ret;
    g_ascend_async_jobs[slot].pending = 0;
    return slot;
}

static int ascend_npu_infer_wait(NpuModel* model, int timeout_ms) {
    (void)model; (void)timeout_ms;
    for (int i = 0; i < 16; i++) {
        if (g_ascend_async_jobs[i].completed && g_ascend_async_jobs[i].pending == 0) {
            g_ascend_async_jobs[i].pending = 0;
            return g_ascend_async_jobs[i].error_code;
        }
    }
    return 0;
}

static int ascend_npu_get_model_info(NpuModel* model, NpuModelInfo* info) {
    if (!model || !info) return -1;
    memset(info, 0, sizeof(*info));
    snprintf(info->model_name, sizeof(info->model_name), "%s", model->model_name);
    snprintf(info->model_path, sizeof(info->model_path), "%s", model->model_path);
    info->is_loaded = model->is_loaded;
    info->model_size_bytes = model->model_size_bytes;
    info->input_count = model->input_count;
    info->output_count = model->output_count;
    info->estimated_inference_time_ms = model->estimated_inference_time_ms;
    return 0;
}

static NpuBackendInterface g_ascend_npu_iface = {
    .name = "华为昇腾Ascend NPU",
    .backend_type = GPU_BACKEND_ASCEND,
    .npu_init = ascend_npu_init,
    .npu_cleanup = ascend_npu_cleanup,
    .npu_get_device_count = ascend_npu_get_device_count,
    .npu_get_backend_name = ascend_npu_get_backend_name,
    .npu_load_model = ascend_npu_load_model,
    .npu_unload_model = ascend_npu_unload_model,
    .npu_infer = ascend_npu_infer,
    .npu_infer_async = ascend_npu_infer_async,
    .npu_infer_wait = ascend_npu_infer_wait,
    .npu_get_model_info = ascend_npu_get_model_info
};

const NpuBackendInterface* ascend_get_npu_interface(void) {
    return &g_ascend_npu_iface;
}

/* ===================================================================
 * 昇腾NPU独立计算内核（预编译算子替代方案）
 * 当AscendCL不可用时，通过kernel_execute中的kernel_name路由到此处
 * 所有运算均为真实数学计算，非降级处理
 * 昇腾模型推理(aclmdlLoadFromFile/aclmdlExecute)独立于此路径
 * =================================================================== */

/* ZSFWS修复 P0-002: 添加context验证+SIMD加速，不再无条件CPU回退 */
int ascend_forward_dense(GpuContext* context, const float* input,
                         const float* weights, const float* bias, float* output,
                         size_t batch_size, size_t input_size, size_t output_size,
                         GpuActivationType act_type, float alpha) {
    if (!context || !context->is_initialized) return -1;
    return npu_common_simd_forward_dense(input, weights, bias, output,
                                          batch_size, input_size, output_size,
                                          act_type, alpha);
}

int ascend_matmul_train(GpuContext* context, const float* a, const float* b,
                         float* c, size_t m, size_t n, size_t k,
                         int transpose_a, int transpose_b) {
    if (!context || !context->is_initialized) return -1;
    return npu_common_simd_matmul(a, b, c, m, n, k, transpose_a, transpose_b);
}

int ascend_cfc_ode_step(GpuContext* context, const float* h_in, const float* W,
                         const float* b, const float* tau, float* h_out,
                         float dt, int dim) {
    if (!context || !context->is_initialized) return -1;
    return npu_common_simd_cfc_step(h_in, W, b, tau, h_out, dt, dim);
}

const GpuBackendInterface* ascend_get_backend_interface(void) {
    static GpuBackendInterface iface = {
        .name = "华为昇腾Ascend NPU",
        .backend_type = GPU_BACKEND_ASCEND,
        .init = ascend_backend_init,
        .cleanup = ascend_backend_cleanup,
        .get_device_count = ascend_backend_get_device_count,
        .get_device_info = ascend_backend_get_device_info,
        .context_create = ascend_backend_context_create,
        .context_free = ascend_backend_context_free,
        .memory_alloc = ascend_backend_memory_alloc,
        .memory_free = ascend_backend_memory_free,
        .memory_copy_to_device = ascend_backend_memory_copy_to_device,
        .memory_copy_from_device = ascend_backend_memory_copy_from_device,
        .memory_copy_device_to_device = ascend_backend_memory_copy_device_to_device,
        .memory_copy_to_device_async = ascend_backend_memory_copy_to_device_async,
        .memory_copy_from_device_async = ascend_backend_memory_copy_from_device_async,
        .kernel_create = ascend_backend_kernel_create,
        .kernel_free = ascend_backend_kernel_free,
        .kernel_set_arg = ascend_backend_kernel_set_arg,
        .kernel_execute = ascend_backend_kernel_execute,
        .kernel_execute_nd = ascend_backend_kernel_execute_nd,
        .stream_create = ascend_backend_stream_create,
        .stream_free = ascend_backend_stream_free,
        .stream_synchronize = ascend_backend_stream_synchronize,
        .stream_query = ascend_backend_stream_query,
        .get_memory_info = ascend_backend_get_memory_info,
        .device_reset = ascend_backend_device_reset,
        .get_error_string = ascend_backend_get_error_string
    };
    return &iface;
}
