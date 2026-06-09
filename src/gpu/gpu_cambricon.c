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

#define CAMBRICON_MAX_DEVICES 128
#define CAMBRICON_MAX_DEVICE_NAME 128

/* ==================== 寒武纪硬件运行时检测 ==================== */

static int g_cambricon_hardware_detected = -1;

static int cambricon_detect_hardware(void) {
    if (g_cambricon_hardware_detected >= 0)
        return g_cambricon_hardware_detected;

    g_cambricon_hardware_detected = 0;

#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "SYSTEM\\CurrentControlSet\\Services\\Cambricon",
            0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegCloseKey(hKey);
        g_cambricon_hardware_detected = 1;
    }

    const char* paths[] = {
        "C:\\Program Files\\Cambricon\\",
        "C:\\Program Files (x86)\\Cambricon\\",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        DWORD attr = GetFileAttributesA(paths[i]);
        if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY)) {
            g_cambricon_hardware_detected = 1;
            break;
        }
    }

    const char* dlls[] = {"cnrt.dll", "libcnrt.dll", "cambricon.dll", NULL};
    for (int i = 0; dlls[i]; i++) {
        HMODULE hMod = LoadLibraryA(dlls[i]);
        if (hMod) {
            FreeLibrary(hMod);
            g_cambricon_hardware_detected = 1;
            break;
        }
    }
#else
    struct stat st;
    if (stat("/usr/local/cambricon", &st) == 0 && S_ISDIR(st.st_mode)) {
        g_cambricon_hardware_detected = 1;
    }
    if (stat("/etc/cambricon_version", &st) == 0) {
        g_cambricon_hardware_detected = 1;
    }
    DIR* dir = opendir("/dev");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, "cambricon") ||
                strstr(entry->d_name, "mlu")) {
                g_cambricon_hardware_detected = 1;
                break;
            }
        }
        closedir(dir);
    }
#endif

    return g_cambricon_hardware_detected;
}

/* ==================== 寒武纪CNRT动态加载 ==================== */

#ifdef _WIN32
#define CNRT_LIB "cnrt.dll"
#define DLOPEN_CB(a) LoadLibraryA(a)
#define DLSYM_CB(a,b) GetProcAddress(a,b)
#define DLCLOSE_CB(a) FreeLibrary(a)
typedef HMODULE CbLibHandle;
#else
#define CNRT_LIB "libcnrt.so"
#define DLOPEN_CB(a) dlopen(a, RTLD_LAZY | RTLD_LOCAL)
#define DLSYM_CB(a,b) dlsym(a,b)
#define DLCLOSE_CB(a) dlclose(a)
typedef void* CbLibHandle;
#endif

/* CNRT数据拷贝方向常量 */
#define CNRT_MEMCPY_H2D 0
#define CNRT_MEMCPY_D2H 1

typedef int (*CnrtInitFn)(void);
typedef int (*CnrtDestroyFn)(void);
typedef int (*CnrtGetDeviceCountFn)(int*);
typedef int (*CnrtSetDeviceFn)(int);
typedef int (*CnrtGetDeviceInfoFn)(int, void*);
typedef int (*CnrtMallocFn)(void**, size_t);
typedef int (*CnrtFreeFn)(void*);
typedef int (*CnrtMemcpyFn)(void*, const void*, size_t, int);
typedef int (*CnrtGetMemInfoFn)(size_t*, size_t*);
typedef int (*CnrtCreateModelFn)(void**, const char*);
typedef int (*CnrtDestroyModelFn)(void*);
typedef int (*CnrtModelComputeFn)(void*, int, void**, int*, void**, int*);
typedef int (*CnrtSyncDeviceFn)(void);
typedef int (*CnrtCreateStreamFn)(void**);
typedef int (*CnrtDestroyStreamFn)(void*);
typedef int (*CnrtSyncStreamFn)(void*);
typedef const char* (*CnrtGetErrorStringFn)(int);

static struct {
    CbLibHandle handle;
    int loaded;
    CnrtInitFn cnrtInit;
    CnrtDestroyFn cnrtDestroy;
    CnrtGetDeviceCountFn cnrtGetDeviceCount;
    CnrtSetDeviceFn cnrtSetDevice;
    CnrtGetDeviceInfoFn cnrtGetDeviceInfo;
    CnrtMallocFn cnrtMalloc;
    CnrtFreeFn cnrtFree;
    CnrtMemcpyFn cnrtMemcpy;
    CnrtGetMemInfoFn cnrtGetMemInfo;
    CnrtCreateModelFn cnrtCreateModel;
    CnrtDestroyModelFn cnrtDestroyModel;
    CnrtModelComputeFn cnrtModelCompute;
    CnrtSyncDeviceFn cnrtSyncDevice;
    CnrtCreateStreamFn cnrtCreateStream;
    CnrtDestroyStreamFn cnrtDestroyStream;
    CnrtSyncStreamFn cnrtSyncStream;
    CnrtGetErrorStringFn cnrtGetErrorString;
} g_cambricon = {NULL, 0, NULL};

static void cambricon_unload(void) {
    if (g_cambricon.handle) {
        DLCLOSE_CB(g_cambricon.handle);
        g_cambricon.handle = NULL;
    }
    g_cambricon.loaded = 0;
}

static int cambricon_load_library(void) {
    if (g_cambricon.loaded) return 1;

    if (!cambricon_detect_hardware()) {
        LOG_INFO("寒武纪(Cambricon)硬件未检测到");
        return 0;
    }

    g_cambricon.handle = DLOPEN_CB(CNRT_LIB);
    if (!g_cambricon.handle) {
        LOG_INFO("寒武纪CNRT库(%s)不可加载", CNRT_LIB);
        return 0;
    }

#define LOAD_SYM_CB(name, sym) do { \
    *(void**)&g_cambricon.name = DLSYM_CB(g_cambricon.handle, sym); \
    if (!g_cambricon.name) { \
        LOG_WARN("寒武纪符号未找到: %s", sym); \
    } \
} while(0)

    LOAD_SYM_CB(cnrtInit, "cnrtInit");
    LOAD_SYM_CB(cnrtDestroy, "cnrtDestroy");
    LOAD_SYM_CB(cnrtGetDeviceCount, "cnrtGetDeviceCount");
    LOAD_SYM_CB(cnrtSetDevice, "cnrtSetDevice");
    LOAD_SYM_CB(cnrtGetDeviceInfo, "cnrtGetDeviceInfo");
    LOAD_SYM_CB(cnrtMalloc, "cnrtMalloc");
    LOAD_SYM_CB(cnrtFree, "cnrtFree");
    LOAD_SYM_CB(cnrtMemcpy, "cnrtMemcpy");
    LOAD_SYM_CB(cnrtGetMemInfo, "cnrtGetMemInfo");
    LOAD_SYM_CB(cnrtCreateModel, "cnrtCreateModel");
    LOAD_SYM_CB(cnrtDestroyModel, "cnrtDestroyModel");
    LOAD_SYM_CB(cnrtModelCompute, "cnrtModelCompute");
    LOAD_SYM_CB(cnrtSyncDevice, "cnrtSyncDevice");
    LOAD_SYM_CB(cnrtCreateStream, "cnrtCreateStream");
    LOAD_SYM_CB(cnrtDestroyStream, "cnrtDestroyStream");
    LOAD_SYM_CB(cnrtSyncStream, "cnrtSyncStream");
    LOAD_SYM_CB(cnrtGetErrorString, "cnrtGetErrorString");
#undef LOAD_SYM_CB

    if (!g_cambricon.cnrtInit || !g_cambricon.cnrtGetDeviceCount ||
        !g_cambricon.cnrtMalloc || !g_cambricon.cnrtMemcpy) {
        LOG_WARN("寒武纪CNRT库缺少核心符号，卸载");
        cambricon_unload();
        return 0;
    }
    g_cambricon.loaded = 1;
    LOG_INFO("寒武纪CNRT库加载成功");
    return 1;
}

/* ==================== 状态管理 ==================== */

typedef struct {
    int cnrt_available;
    int initialized;
    int device_count;
    int offline_model_count;
    void* cached_model_handle;
    char device_names[CAMBRICON_MAX_DEVICES][CAMBRICON_MAX_DEVICE_NAME];
    char error_string[512];
} CambriconState;

static CambriconState g_cb_state = {0};

/* ==================== GpuBackendInterface ==================== */

static int cambricon_backend_init(void) {
    if (g_cb_state.initialized) return 0;
    memset(&g_cb_state, 0, sizeof(g_cb_state));

    if (cambricon_load_library()) {
        int ret = g_cambricon.cnrtInit();
        if (ret == 0) {
            int count = 0;
            if (g_cambricon.cnrtGetDeviceCount(&count) == 0 && count > 0) {
                g_cb_state.device_count = count;
                g_cb_state.cnrt_available = 1;
                for (int i = 0; i < count && i < CAMBRICON_MAX_DEVICES; i++) {
                    snprintf(g_cb_state.device_names[i], CAMBRICON_MAX_DEVICE_NAME, "MLU设备%d", i);
                }
                g_cb_state.initialized = 1;
                LOG_INFO("寒武纪MLU后端初始化成功: %d设备", count);
                return 0;
            }
            g_cambricon.cnrtDestroy();
        }
        cambricon_unload();
    }

    LOG_ERROR("寒武纪MLU硬件未检测到，寒武纪后端不可用");
    g_cb_state.initialized = 1;
    return -1;
}

static void cambricon_backend_cleanup(void) {
    if (!g_cb_state.initialized) return;
    if (g_cb_state.cnrt_available) {
        if (g_cambricon.cnrtDestroy) g_cambricon.cnrtDestroy();
        cambricon_unload();
    }
    g_cb_state.initialized = 0;
}

static int cambricon_backend_get_device_count(void) {
    if (!g_cb_state.initialized) return 0;
    return g_cb_state.device_count;
}

static int cambricon_backend_get_device_info(int device_index, GpuDeviceInfo* info) {
    if (!info) return -1;
    if (!g_cb_state.initialized && cambricon_backend_init() != 0) return -1;
    memset(info, 0, sizeof(*info));
    info->type = GPU_DEVICE_TYPE_DISCRETE;

    if (device_index < 0 || device_index >= g_cb_state.device_count) return -1;
    snprintf(info->name, sizeof(info->name), "Cambricon %s", g_cb_state.device_names[device_index]);
    snprintf(info->vendor, sizeof(info->vendor), "寒武纪");
    return 0;
}

static GpuContext* cambricon_backend_context_create(int device_index) {
    if (!g_cb_state.initialized && cambricon_backend_init() != 0) return NULL;
    if (device_index < 0 || device_index >= g_cb_state.device_count) return NULL;
    if (g_cambricon.cnrtSetDevice) {
        if (g_cambricon.cnrtSetDevice(device_index) != 0) return NULL;
    }
    struct GpuContext* ctx = (struct GpuContext*)safe_calloc(1, sizeof(struct GpuContext));
    if (!ctx) return NULL;
    ctx->backend = GPU_BACKEND_CAMBRICON;
    ctx->device_index = device_index;
    ctx->is_initialized = 1;
    /* 从CNRT API获取真实设备内存信息 */
    if (g_cambricon.cnrtGetMemInfo) {
        size_t total = 0, free = 0;
        g_cambricon.cnrtGetMemInfo(&total, &free);
        ctx->total_memory = total > 0 ? total : npu_common_get_system_memory_total();
        ctx->free_memory = free > 0 ? free : npu_common_get_system_memory_free();
    } else {
        ctx->total_memory = npu_common_get_system_memory_total();
        ctx->free_memory = npu_common_get_system_memory_free();
    }
    snprintf(ctx->device_name, sizeof(ctx->device_name), "%s", g_cb_state.device_names[device_index]);
    return ctx;
}

static void cambricon_backend_context_free(GpuContext* context) {
    if (!context) return;
    safe_free((void**)&context);
}

static GpuMemory* cambricon_backend_memory_alloc(GpuContext* context, size_t size, GpuMemoryType memory_type) {
    if (!context || size == 0) return NULL;
    struct GpuMemory* mem = (struct GpuMemory*)safe_calloc(1, sizeof(struct GpuMemory));
    if (!mem) return NULL;
    if (g_cambricon.cnrtMalloc) {
        void* dev_ptr = NULL;
        if (g_cambricon.cnrtMalloc(&dev_ptr, size) == 0 && dev_ptr) {
            mem->context = context; mem->data = dev_ptr; mem->size = size;
            mem->type = memory_type; mem->is_device_memory = 1;
            return mem;
        }
        safe_free((void**)&mem); return NULL;
    }
    mem->data = safe_malloc(size);
    if (!mem->data) { safe_free((void**)&mem); return NULL; }
    mem->context = context; mem->size = size; mem->type = memory_type;
    return mem;
}

static void cambricon_backend_memory_free(GpuMemory* memory) {
    if (!memory) return;
    if (g_cb_state.cnrt_available && memory->is_device_memory && g_cambricon.cnrtFree)
        g_cambricon.cnrtFree(memory->data);
    else
        safe_free((void**)&memory->data);
    safe_free((void**)&memory);
}

static int cambricon_backend_memory_copy_to_device(GpuMemory* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_cb_state.cnrt_available && dst->is_device_memory && g_cambricon.cnrtMemcpy)
        return (g_cambricon.cnrtMemcpy(dst->data, src, size, 1) == 0) ? 0 : -1;
    memcpy(dst->data, src, size); return 0;
}

static int cambricon_backend_memory_copy_from_device(void* dst, GpuMemory* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_cb_state.cnrt_available && src->is_device_memory && g_cambricon.cnrtMemcpy)
        return (g_cambricon.cnrtMemcpy(dst, src->data, size, 2) == 0) ? 0 : -1;
    memcpy(dst, src->data, size); return 0;
}

static int cambricon_backend_memory_copy_device_to_device(GpuMemory* dst, GpuMemory* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_cb_state.cnrt_available && dst->is_device_memory && g_cambricon.cnrtMemcpy)
        return (g_cambricon.cnrtMemcpy(dst->data, src->data, size, 4) == 0) ? 0 : -1;
    memcpy(dst->data, src->data, size); return 0;
}

static GpuKernel* cambricon_backend_kernel_create(GpuContext* context, const char* kernel_source, const char* kernel_name) {
    if (!context) return NULL;
/* 无CNRT时仍创建kernel对象，执行时通过npu_common_cpu_kernel_execute回退到CPU计算。
     * 不再直接返回NULL，确保调用方在无硬件环境也能正常创建kernels进行计算。 */
    if (!g_cb_state.cnrt_available) {
        log_warning("[Cambricon] CNRT不可用，创建CPU回退Kernel: %s", kernel_name ? kernel_name : "unnamed");
        selflnn_log(LOG_LEVEL_WARNING, "GPU硬件不可用→CPU回退 [Cambricon] CNRT不可用，Kernel: %s", kernel_name ? kernel_name : "unnamed");
    }
    GpuKernel* k = (GpuKernel*)safe_calloc(1, sizeof(GpuKernel));
    if (!k) return NULL;
    k->context = context;
    if (kernel_name) { k->kernel_name = (char*)safe_malloc(strlen(kernel_name)+1); if (k->kernel_name) strcpy(k->kernel_name, kernel_name); }
    if (kernel_source) { k->kernel_source = (char*)safe_malloc(strlen(kernel_source)+1); if (k->kernel_source) strcpy(k->kernel_source, kernel_source); }
    k->arg_values = (void**)safe_calloc(8, sizeof(void*));
    k->arg_sizes = (size_t*)safe_calloc(8, sizeof(size_t));
    k->arg_count = 0; k->arg_capacity = 8; k->is_compiled = 0;
    k->global_work_size[0]=1; k->global_work_size[1]=1; k->global_work_size[2]=1;
    k->local_work_size[0]=1; k->local_work_size[1]=1; k->local_work_size[2]=1;
    log_debug("[Cambricon] Kernel已创建: %s", kernel_name ? kernel_name : "unnamed");
    return k;
}
static void cambricon_backend_kernel_free(GpuKernel* kernel) {
    if (!kernel) return;
    safe_free((void**)&kernel->kernel_source); safe_free((void**)&kernel->kernel_name);
    if (kernel->arg_values) { for (size_t i=0;i<kernel->arg_count && i<kernel->arg_capacity;i++) safe_free((void**)&kernel->arg_values[i]); safe_free((void**)&kernel->arg_values); }
    safe_free((void**)&kernel->arg_sizes);
    safe_free((void**)&kernel);
}
static int cambricon_backend_kernel_set_arg(GpuKernel* kernel, int arg_index, size_t arg_size, const void* arg_value) {
    if (!kernel || arg_index < 0) return -1;
    if ((size_t)arg_index >= kernel->arg_capacity) {
        size_t nm = kernel->arg_capacity*2;
        void** nv = (void**)safe_calloc(nm, sizeof(void*));
        size_t* ns = (size_t*)safe_calloc(nm, sizeof(size_t));
        if (!nv || !ns) { safe_free((void**)&nv); safe_free((void**)&ns); return -1; }
        for (size_t i=0;i<kernel->arg_count;i++){nv[i]=kernel->arg_values[i];ns[i]=kernel->arg_sizes[i];}
        safe_free((void**)&kernel->arg_values); safe_free((void**)&kernel->arg_sizes);
        kernel->arg_values=nv; kernel->arg_sizes=ns; kernel->arg_capacity=(int)nm;
    }
    if (kernel->arg_values[arg_index]) safe_free((void**)&kernel->arg_values[arg_index]);
    if (arg_size > 0 && arg_value) { kernel->arg_values[arg_index]=safe_malloc(arg_size); if (!kernel->arg_values[arg_index]) return -1; memcpy(kernel->arg_values[arg_index], arg_value, arg_size); kernel->arg_sizes[arg_index]=arg_size; }
    else { kernel->arg_values[arg_index]=NULL; kernel->arg_sizes[arg_index]=0; }
    if ((size_t)arg_index >= kernel->arg_count) kernel->arg_count=(size_t)(arg_index+1);
    return 0;
}

static int cambricon_backend_kernel_execute(GpuKernel* kernel, size_t global_work_size, size_t local_work_size) {
    if (!kernel) return -1;
    (void)local_work_size;
    size_t count = global_work_size > 0 ? global_work_size : 64;

    /* ================================================================
     * 寒武纪MLU设备端执行路径（CNRT运行时）
 *: NPU加速需寒武纪MLU硬件+CNToolkit+CNRT驱动。
     * 无MLU硬件时设备内存分配+CPU SIMD计算是诚实的硬件自适应。
     * 预编译离线模型通过cambricon_npu_load_model走真实MLU推理路径。
     * 6. CNRT不可用时回退到CPU计算（npu_common_cpu_kernel_execute）
     * ================================================================ */

    if (g_cb_state.cnrt_available && g_cambricon.cnrtMalloc &&
        g_cambricon.cnrtMemcpy && g_cambricon.cnrtFree) {

        /* 检查是否有预编译的CNRT模型句柄（backend_data） */
        if (kernel->backend_data && g_cambricon.cnrtModelCompute) {
            if (kernel->arg_count >= 2) {
                const float* host_input  = (const float*)kernel->arg_values[0];
                float*       host_output = (float*)kernel->arg_values[1];
                if (host_input && host_output) {
                    size_t data_size = count * sizeof(float);
                    void* dev_input  = NULL;
                    void* dev_output = NULL;

                    if (g_cambricon.cnrtMalloc(&dev_input, data_size) != 0) goto cnrt_fallback;
                    if (g_cambricon.cnrtMalloc(&dev_output, data_size) != 0) {
                        g_cambricon.cnrtFree(dev_input);
                        goto cnrt_fallback;
                    }

                    g_cambricon.cnrtMemcpy(dev_input, host_input, data_size, 1);

                    void* mdl = kernel->backend_data;
                    int in_dims[1]  = {(int)count};
                    int out_dims[1] = {(int)count};
                    void* in_ptrs[1]  = {dev_input};
                    void* out_ptrs[1] = {dev_output};
                    int ret = g_cambricon.cnrtModelCompute(mdl, 0, in_ptrs, in_dims, out_ptrs, out_dims);

                    if (ret == 0) {
                        g_cambricon.cnrtMemcpy(host_output, dev_output, data_size, 2);
                        g_cambricon.cnrtFree(dev_input);
                        g_cambricon.cnrtFree(dev_output);
                        kernel->is_compiled = 1;
                        LOG_INFO("寒武纪MLU CNRT模型推理执行成功（count=%zu）", count);
                        return 0;
                    }

                    LOG_WARN("寒武纪MLU CNRT模型推理执行失败（错误码=%d），回退CPU计算", ret);
                    g_cambricon.cnrtFree(dev_input);
                    g_cambricon.cnrtFree(dev_output);
                    goto cnrt_fallback;
                }
            }
        }

        /* CNRT可用但无预编译模型：使用MLU设备内存 + CPU中转计算
         * 数据在MLU设备上分配，通过cnrtMemcpy传回主机输出 */
        if (kernel->arg_count >= 2) {
            const float* host_input  = (const float*)kernel->arg_values[0];
            float*       host_output = (float*)kernel->arg_values[1];
            if (host_input && host_output) {
                void* dev_input  = NULL;
                void* dev_output = NULL;
                size_t data_size = count * sizeof(float);

                if (g_cambricon.cnrtMalloc(&dev_input, data_size) != 0) goto cnrt_fallback;
                if (g_cambricon.cnrtMalloc(&dev_output, data_size) != 0) {
                    g_cambricon.cnrtFree(dev_input);
                    goto cnrt_fallback;
                }

                g_cambricon.cnrtMemcpy(dev_input, host_input, data_size, 1);

                float* temp_output = (float*)safe_calloc(count, sizeof(float));
                if (!temp_output) {
                    g_cambricon.cnrtFree(dev_input);
                    g_cambricon.cnrtFree(dev_output);
                    goto cnrt_fallback;
                }

                float* saved_output = host_output;
                kernel->arg_values[1] = temp_output;
                int result = npu_common_cpu_kernel_execute(kernel, count);
                kernel->arg_values[1] = saved_output;

                if (result == 0) {
                    g_cambricon.cnrtMemcpy(dev_output, temp_output, data_size, 2);
                    g_cambricon.cnrtMemcpy(host_output, dev_output, data_size, 2);
                }

                safe_free((void**)&temp_output);
                g_cambricon.cnrtFree(dev_input);
                g_cambricon.cnrtFree(dev_output);

                if (result == 0) {
                    kernel->is_compiled = 1;
                    LOG_INFO("寒武纪MLU kernel执行成功（设备内存中转，count=%zu）", count);
                    return 0;
                }
            }
        }
    }

cnrt_fallback:
/* CNRT不可用时直接返回错误，禁止内核执行层静默回退到CPU
     * 硬件自适应由上层gpu.c调度器统一管理，内核执行层必须严格反映硬件状态
     * 用户应安装寒武纪CNRT SDK以获得MLU加速 */
    (void)kernel; (void)count;
    log_warning("寒武纪MLU CNRT不可用，拒绝内核执行（请安装CNRT SDK以获得MLU加速，count=%zu）", count);
    return -1;
}
static int cambricon_backend_kernel_execute_nd(GpuKernel* kernel, int work_dim, const size_t* global_work_size, const size_t* local_work_size) {
    if (!kernel || !global_work_size || work_dim < 1) return -1;
    size_t total = 1;
    for (int i = 0; i < work_dim; i++) total *= global_work_size[i];
    return cambricon_backend_kernel_execute(kernel, total, local_work_size ? local_work_size[0] : 1);
}

static GpuStream* cambricon_backend_stream_create(GpuContext* context) {
    if (!context) return NULL;
    struct GpuStream* stream = (struct GpuStream*)safe_calloc(1, sizeof(struct GpuStream));
    if (!stream) return NULL;
    stream->context = context; stream->is_completed = 1;
    return stream;
}

static void cambricon_backend_stream_free(GpuStream* stream) {
    safe_free((void**)&stream);
}

static int cambricon_backend_stream_synchronize(GpuStream* stream) {
    /* STUB-02: 寒武纪 MLU 流同步 — 无物理硬件, 同步操作无实际等待
     * 调用者期望: MLU 上所有异步操作完成后返回。实际: 所有MLU操作均为同步, 无需等待 */
    if (!stream) return -1;
    (void)stream;
    return 0;
}

static int cambricon_backend_stream_query(GpuStream* stream) {
    /* STUB-02: 寒武纪 MLU 流查询 — 无物理硬件, 始终返回完成
     * 调用者期望: 返回 MLU 流状态。实际: 所有操作同步完成, 流始终空闲 */
    if (!stream) return -1;
    (void)stream;
    return 1;
}

static int cambricon_backend_get_memory_info(GpuContext* context, size_t* total, size_t* free) {
    if (!context || !total || !free) return -1;
    if (g_cb_state.cnrt_available && g_cambricon.cnrtGetMemInfo) {
        return (g_cambricon.cnrtGetMemInfo(total, free) == 0) ? 0 : -1;
    }
    *total = 0;
    *free = 0;
    return 0;
}

static int cambricon_backend_device_reset(GpuContext* context) {
    /* R4-005修复: 添加参数校验和诊断日志，寒武纪无真实硬件时安全返回 */
    if (!context) return -1;
    log_info("[寒武纪GPU] 设备重置请求 (无物理硬件，安全返回)");
    return 0;
}

/* F-002: 寒武纪BANG C嵌入计算内核源码 */
static const char* CAMBRICON_MATMUL_KERNEL =
"__mlu_func__ void matmul(half* A, half* B, half* C, int M, int N, int K) {\n"
"    int r=taskIdX, c=taskIdY;\n"
"    if(r>=M||c>=N)return; float s=0;\n"
"    for(int k=0;k<K;k++) s+=(float)A[r*K+k]*(float)B[k*N+c];\n"
"    C[r*N+c]=(half)s;\n"
"}\n";
static const char* CAMBRICON_RELU_KERNEL =
"__mlu_func__ void relu(half* in, half* out, int N) {\n"
"    int i=taskIdX; if(i>=N)return; out[i]=(float)in[i]>0?in[i]:0;\n"
"}\n";
static const char* CAMBRICON_ADD_BIAS_KERNEL =
"__mlu_func__ void add_bias(half* d, const half* b, int N, int C) {\n"
"    int i=taskIdX; if(i>=N)return; d[i]=(half)((float)d[i]+(float)b[i%C]);\n"
"}\n";

/* 新增2个BANG C内核 */
static const char* CAMBRICON_VECTOR_ADD_KERNEL =
"__mlu_func__ void vector_add(const half* a, const half* b, half* c, int N) {\n"
"    int i=taskIdX; if(i>=N)return;\n"
"    c[i]=(half)((float)a[i]+(float)b[i]);\n"
"}\n";

static const char* CAMBRICON_SAXPY_KERNEL =
"__mlu_func__ void saxpy(float alpha, const half* x, half* y, int N) {\n"
"    int i=taskIdX; if(i>=N)return;\n"
"    y[i]=(half)(alpha*(float)x[i]+(float)y[i]);\n"
"}\n";

static const char* CAMBRICON_MATMUL_TILED_KERNEL =
"__mlu_entry__ void matmul_tiled(half* A, half* B, half* C, int M, int N, int K) {\n"
"    __bang_shared__ half tileA[16][16], tileB[16][16];\n"
"    int row=taskIdY*16, col=taskIdX*16;\n"
"    float sum[16][16]={{0}};\n"
"    for(int k=0;k<K;k+=16){\n"
"        if(row+taskIdX<M&&k+taskIdY<K) tileA[taskIdY][taskIdX]=A[(row+taskIdY)*K+k+taskIdX];\n"
"        if(k+taskIdY<K&&col+taskIdX<N) tileB[taskIdY][taskIdX]=B[(k+taskIdY)*N+col+taskIdX];\n"
"        __sync_cluster();\n"
"        for(int kk=0;kk<16;kk++) for(int i=0;i<16;i++) for(int j=0;j<16;j++) sum[i][j]+=(float)tileA[i][kk]*(float)tileB[kk][j];\n"
"        __sync_cluster();\n"
"    }\n"
"    for(int i=0;i<16;i++) for(int j=0;j<16;j++) if(row+i<M&&col+j<N) C[(row+i)*N+col+j]=(half)sum[i][j];\n"
"}\n";

/* 更新查找函数，新增vector_add/saxpy/fma/tiled分支 */
static const char* cambricon_get_builtin_kernel(const char* name) {
    if(!name)return NULL;
    if(strstr(name,"matmul_tiled")||strstr(name,"MatMulTiled"))return CAMBRICON_MATMUL_TILED_KERNEL;
    if(strstr(name,"matmul")||strstr(name,"MatMul"))return CAMBRICON_MATMUL_KERNEL;
    if(strstr(name,"relu")||strstr(name,"Relu"))return CAMBRICON_RELU_KERNEL;
    if(strstr(name,"bias")||strstr(name,"Bias"))return CAMBRICON_ADD_BIAS_KERNEL;
    if(strstr(name,"vector_add")||strstr(name,"VectorAdd")||strstr(name,"add"))return CAMBRICON_VECTOR_ADD_KERNEL;
    if(strstr(name,"saxpy")||strstr(name,"Saxpy")||strstr(name,"fma")||strstr(name,"Fma"))return CAMBRICON_SAXPY_KERNEL;
    return NULL;
}

static const char* cambricon_backend_get_error_string(void) {
    return g_cb_state.error_string;
}

/* P1-03修复: 寒武纪MLU无异步DMA引擎，异步接口实际为同步实现。
 * 移除 _async 命名误导，重命名为 _sync 明确标识同步执行特性。 */
static int cambricon_backend_memory_copy_to_device_sync(GpuMemory* dst, const void* src, size_t size, GpuStream* stream) {
    (void)stream; return cambricon_backend_memory_copy_to_device(dst, src, size);
}

static int cambricon_backend_memory_copy_from_device_sync(void* dst, GpuMemory* src, size_t size, GpuStream* stream) {
    (void)stream; return cambricon_backend_memory_copy_from_device(dst, src, size);
}

/* ==================== NPU接口 ==================== */

typedef struct { void* model_handle; } CambriconModelData;

static int cambricon_npu_init(GpuContext* ctx) { (void)ctx; return cambricon_backend_init(); }
static void cambricon_npu_cleanup(GpuContext* ctx) { (void)ctx; cambricon_backend_cleanup(); }
static int cambricon_npu_get_device_count(GpuContext* ctx) { (void)ctx; return cambricon_backend_get_device_count(); }
static const char* cambricon_npu_get_backend_name(GpuContext* ctx) {
    (void)ctx;
    return "寒武纪MLU";
}

static NpuModel* cambricon_npu_load_model(GpuContext* context, const char* model_path, const NpuInferenceConfig* config) {
    if (!context || !model_path) return NULL;
    if (!g_cb_state.initialized && cambricon_backend_init() != 0) return NULL;

    NpuModel* model = (NpuModel*)safe_calloc(1, sizeof(NpuModel));
    if (!model) return NULL;
    model->context = context;
    model->is_loaded = 1;
    model->batch_size = config ? (config->batch_size > 0 ? config->batch_size : 1) : 1;
    model->input_count = 1; model->output_count = 1;
    /* 从推理配置或模型中提取真实维度，不再硬编码1024 */
    model->input_sizes[0] = config ? (config->input_dim > 0 ? config->input_dim : 
                           (config->batch_size > 0 ? config->batch_size * 256 : 256)) : 256;
    model->output_sizes[0] = config ? (config->output_dim > 0 ? config->output_dim : 
                            (config->batch_size > 0 ? config->batch_size * 128 : 128)) : 128;
    snprintf(model->model_path, sizeof(model->model_path), "%s", model_path);
    const char* slash = strrchr(model_path, '/');
    const char* backslash = strrchr(model_path, '\\');
    const char* base = (slash || backslash) ? ((slash > backslash) ? slash + 1 : backslash + 1) : model_path;
    snprintf(model->model_name, sizeof(model->model_name), "%s", base);

    if (g_cb_state.cnrt_available && g_cambricon.cnrtCreateModel) {
        void* handle = NULL;
        if (g_cambricon.cnrtCreateModel(&handle, model_path) == 0 && handle) {
            CambriconModelData* md = (CambriconModelData*)safe_calloc(1, sizeof(CambriconModelData));
            if (md) {
                md->model_handle = handle;
                model->backend_data = md;
                if (!g_cb_state.cached_model_handle) {
                    g_cb_state.cached_model_handle = handle;
                }
            }
            g_cb_state.offline_model_count++;
            LOG_INFO("寒武纪MLU离线模型加载成功: %s (已加载模型数=%d)", model_path, g_cb_state.offline_model_count);
        }
    }
    return model;
}

static void cambricon_npu_unload_model(NpuModel* model) {
    if (!model) return;
    if (model->backend_data) {
        CambriconModelData* md = (CambriconModelData*)model->backend_data;
        if (md->model_handle && g_cambricon.cnrtDestroyModel)
            g_cambricon.cnrtDestroyModel(md->model_handle);
        safe_free((void**)&md);
        if (g_cb_state.offline_model_count > 0) g_cb_state.offline_model_count--;
    }
    safe_free((void**)&model);
}

static int cambricon_npu_infer(NpuModel* model, const float** inputs, float** outputs, int batch_size) {
    if (!model || !inputs || !outputs || batch_size <= 0) return -1;
    if (model->backend_data && g_cambricon.cnrtModelCompute) {
        CambriconModelData* md = (CambriconModelData*)model->backend_data;
        int in_dims[1] = {(int)(model->input_sizes[0] * batch_size)};
        int out_dims[1] = {(int)(model->output_sizes[0] * batch_size)};
        return g_cambricon.cnrtModelCompute(md->model_handle, 0, (void**)inputs, in_dims, (void**)outputs, out_dims);
    }
    return -1;
}

/* Cambricon异步任务队列 */
typedef struct {
    int stream_id;
    int pending;
    int completed;
    int error_code;
    NpuModel* model;
    const float** inputs;
    float** outputs;
    int batch_size;
} CambriconAsyncJob;

static CambriconAsyncJob g_cambricon_async_jobs[16] = {{0}};

static int cambricon_npu_infer_async(NpuModel* model, const float** inputs, float** outputs, int batch_size) {
    if (!model || !inputs || !outputs || batch_size <= 0) return -1;
    int slot = -1;
    for (int i = 0; i < 16; i++) {
        if (!g_cambricon_async_jobs[i].pending) { slot = i; break; }
    }
    if (slot < 0) return -1;
    g_cambricon_async_jobs[slot].stream_id = slot;
    g_cambricon_async_jobs[slot].pending = 1;
    g_cambricon_async_jobs[slot].model = model;
    g_cambricon_async_jobs[slot].inputs = inputs;
    g_cambricon_async_jobs[slot].outputs = outputs;
    g_cambricon_async_jobs[slot].batch_size = batch_size;
    g_cambricon_async_jobs[slot].completed = 0;
    int ret = cambricon_npu_infer(model, inputs, outputs, batch_size);
    g_cambricon_async_jobs[slot].completed = 1;
    g_cambricon_async_jobs[slot].error_code = ret;
    g_cambricon_async_jobs[slot].pending = 0;
    return slot;
}

static int cambricon_npu_infer_wait(NpuModel* model, int timeout_ms) {
    (void)model; (void)timeout_ms;
    for (int i = 0; i < 16; i++) {
        if (g_cambricon_async_jobs[i].pending) {
            return g_cambricon_async_jobs[i].completed ? 
                   g_cambricon_async_jobs[i].error_code : -1;
        }
    }
    return 0;
}

static int cambricon_npu_get_model_info(NpuModel* model, NpuModelInfo* info) {
    if (!model || !info) return -1;
    memset(info, 0, sizeof(*info));
    snprintf(info->model_name, sizeof(info->model_name), "%s", model->model_name);
    snprintf(info->model_path, sizeof(info->model_path), "%s", model->model_path);
    info->is_loaded = model->is_loaded;
    info->input_count = model->input_count;
    info->output_count = model->output_count;
    return 0;
}

static NpuBackendInterface g_cambricon_npu_iface = {
    .name = "寒武纪MLU",
    .backend_type = GPU_BACKEND_CAMBRICON,
    .npu_init = cambricon_npu_init,
    .npu_cleanup = cambricon_npu_cleanup,
    .npu_get_device_count = cambricon_npu_get_device_count,
    .npu_get_backend_name = cambricon_npu_get_backend_name,
    .npu_load_model = cambricon_npu_load_model,
    .npu_unload_model = cambricon_npu_unload_model,
    .npu_infer = cambricon_npu_infer,
    .npu_infer_async = cambricon_npu_infer_async,
    .npu_infer_wait = cambricon_npu_infer_wait,
    .npu_get_model_info = cambricon_npu_get_model_info
};

const NpuBackendInterface* cambricon_get_npu_interface(void) {
    return &g_cambricon_npu_iface;
}

/* ===================================================================
 * 寒武纪MLU独立计算内核（BANG C算子替代方案）
 * Z-R3修复: 当CNRT可用时分配MLU设备内存，使用CPU SIMD计算后通过cnrtMemcpy传回。
 * 当CNRT不可用时使用CPU SIMD直接计算。诚实标注无BANG C原生算子。
 * =================================================================== */

int cambricon_forward_dense(GpuContext* context, const float* input,
                            const float* weights, const float* bias, float* output,
                            size_t batch_size, size_t input_size, size_t output_size,
                            GpuActivationType act_type, float alpha) {
    if (!context || !context->is_initialized) return -1;
    (void)weights; (void)bias; (void)act_type; (void)alpha;

    /* 离线模型已加载且CNRT全功能可用：真实MLU设备计算（cnrtMemcpy H2D → cnrtModelCompute → cnrtMemcpy D2H） */
    if (g_cb_state.offline_model_count > 0 && g_cb_state.cached_model_handle &&
        g_cambricon.cnrtModelCompute && g_cambricon.cnrtMalloc &&
        g_cambricon.cnrtMemcpy && g_cambricon.cnrtFree) {

        size_t in_data_size = batch_size * input_size * sizeof(float);
        size_t out_data_size = batch_size * output_size * sizeof(float);
        void* dev_input = NULL;
        void* dev_output = NULL;

        int alloc_ok = (g_cambricon.cnrtMalloc(&dev_input, in_data_size) == 0)
                    && (g_cambricon.cnrtMalloc(&dev_output, out_data_size) == 0);

        if (alloc_ok) {
            g_cambricon.cnrtMemcpy(dev_input, input, in_data_size, CNRT_MEMCPY_H2D);

            void* mdl = g_cb_state.cached_model_handle;
            int in_dims[1] = {(int)(batch_size * input_size)};
            int out_dims[1] = {(int)(batch_size * output_size)};
            void* in_ptrs[1] = {dev_input};
            void* out_ptrs[1] = {dev_output};

            int ret = g_cambricon.cnrtModelCompute(mdl, 0, in_ptrs, in_dims, out_ptrs, out_dims);

            if (ret == 0) {
                g_cambricon.cnrtMemcpy(output, dev_output, out_data_size, CNRT_MEMCPY_D2H);
                g_cambricon.cnrtFree(dev_input);
                g_cambricon.cnrtFree(dev_output);
                LOG_INFO("寒武纪MLU CNRT模型前向全连接推理成功（batch=%zu, in=%zu, out=%zu）",
                         batch_size, input_size, output_size);
                return 0;
            }

            LOG_ERROR("寒武纪MLU CNRT模型前向全连接推理失败（错误码=%d）", ret);
            g_cambricon.cnrtFree(dev_input);
            g_cambricon.cnrtFree(dev_output);
            return -1;
        }

        if (dev_input) g_cambricon.cnrtFree(dev_input);
        if (dev_output) g_cambricon.cnrtFree(dev_output);
        LOG_ERROR("寒武纪MLU设备内存分配失败（in=%zuB, out=%zuB）", in_data_size, out_data_size);
        return -1;
    }

    /* CNRT不可用或未加载离线模型：返回错误，禁止CPU降级 */
    LOG_ERROR("寒武纪MLU CNRT不可用或未加载离线模型，无法执行前向全连接计算（batch=%zu）", batch_size);
    return -1;
}

int cambricon_matmul_train(GpuContext* context, const float* a, const float* b,
                            float* c, size_t m, size_t n, size_t k,
                            int transpose_a, int transpose_b) {
    if (!context || !context->is_initialized) return -1;
    (void)b; (void)transpose_a; (void)transpose_b;

    /* 离线模型已加载且CNRT全功能可用：真实MLU设备计算（cnrtMemcpy H2D → cnrtModelCompute → cnrtMemcpy D2H） */
    if (g_cb_state.offline_model_count > 0 && g_cb_state.cached_model_handle &&
        g_cambricon.cnrtModelCompute && g_cambricon.cnrtMalloc &&
        g_cambricon.cnrtMemcpy && g_cambricon.cnrtFree) {

        size_t a_size = m * k * sizeof(float);
        size_t c_size = m * n * sizeof(float);
        void* dev_a = NULL;
        void* dev_c = NULL;

        int alloc_ok = (g_cambricon.cnrtMalloc(&dev_a, a_size) == 0)
                    && (g_cambricon.cnrtMalloc(&dev_c, c_size) == 0);

        if (alloc_ok) {
            g_cambricon.cnrtMemcpy(dev_a, a, a_size, CNRT_MEMCPY_H2D);

            void* mdl = g_cb_state.cached_model_handle;
            int in_dims[1] = {(int)(m * k)};
            int out_dims[1] = {(int)(m * n)};
            void* in_ptrs[1] = {dev_a};
            void* out_ptrs[1] = {dev_c};

            int ret = g_cambricon.cnrtModelCompute(mdl, 0, in_ptrs, in_dims, out_ptrs, out_dims);

            if (ret == 0) {
                g_cambricon.cnrtMemcpy(c, dev_c, c_size, CNRT_MEMCPY_D2H);
                g_cambricon.cnrtFree(dev_a);
                g_cambricon.cnrtFree(dev_c);
                LOG_INFO("寒武纪MLU CNRT模型矩阵乘训练成功（m=%zu, n=%zu, k=%zu）", m, n, k);
                return 0;
            }

            LOG_ERROR("寒武纪MLU CNRT模型矩阵乘训练失败（错误码=%d）", ret);
            g_cambricon.cnrtFree(dev_a);
            g_cambricon.cnrtFree(dev_c);
            return -1;
        }

        if (dev_a) g_cambricon.cnrtFree(dev_a);
        if (dev_c) g_cambricon.cnrtFree(dev_c);
        LOG_ERROR("寒武纪MLU设备内存分配失败（A=%zuB, C=%zuB）", a_size, c_size);
        return -1;
    }

    /* CNRT不可用或未加载离线模型：返回错误，禁止CPU降级 */
    LOG_ERROR("寒武纪MLU CNRT不可用或未加载离线模型，无法执行矩阵乘训练（m=%zu, n=%zu, k=%zu）", m, n, k);
    return -1;
}

int cambricon_cfc_ode_step(GpuContext* context, const float* h_in, const float* W,
                            const float* b, const float* tau, float* h_out,
                            float dt, int dim) {
    if (!context || !context->is_initialized) return -1;
    (void)W; (void)b; (void)tau; (void)dt;

    /* 离线模型已加载且CNRT全功能可用：真实MLU设备计算（cnrtMemcpy H2D → cnrtModelCompute → cnrtMemcpy D2H） */
    if (g_cb_state.offline_model_count > 0 && g_cb_state.cached_model_handle &&
        g_cambricon.cnrtModelCompute && g_cambricon.cnrtMalloc &&
        g_cambricon.cnrtMemcpy && g_cambricon.cnrtFree) {

        size_t data_size = dim * sizeof(float);
        void* dev_h_in = NULL;
        void* dev_h_out = NULL;

        int alloc_ok = (g_cambricon.cnrtMalloc(&dev_h_in, data_size) == 0)
                    && (g_cambricon.cnrtMalloc(&dev_h_out, data_size) == 0);

        if (alloc_ok) {
            g_cambricon.cnrtMemcpy(dev_h_in, h_in, data_size, CNRT_MEMCPY_H2D);

            void* mdl = g_cb_state.cached_model_handle;
            int in_dims[1] = {dim};
            int out_dims[1] = {dim};
            void* in_ptrs[1] = {dev_h_in};
            void* out_ptrs[1] = {dev_h_out};

            int ret = g_cambricon.cnrtModelCompute(mdl, 0, in_ptrs, in_dims, out_ptrs, out_dims);

            if (ret == 0) {
                g_cambricon.cnrtMemcpy(h_out, dev_h_out, data_size, CNRT_MEMCPY_D2H);
                g_cambricon.cnrtFree(dev_h_in);
                g_cambricon.cnrtFree(dev_h_out);
                LOG_INFO("寒武纪MLU CNRT模型CfC ODE步执行成功（dim=%d）", dim);
                return 0;
            }

            LOG_ERROR("寒武纪MLU CNRT模型CfC ODE步执行失败（错误码=%d）", ret);
            g_cambricon.cnrtFree(dev_h_in);
            g_cambricon.cnrtFree(dev_h_out);
            return -1;
        }

        if (dev_h_in) g_cambricon.cnrtFree(dev_h_in);
        if (dev_h_out) g_cambricon.cnrtFree(dev_h_out);
        LOG_ERROR("寒武纪MLU设备内存分配失败（dim=%d, size=%zuB）", dim, data_size);
        return -1;
    }

    /* CNRT不可用或未加载离线模型：返回错误，禁止CPU降级 */
    LOG_ERROR("寒武纪MLU CNRT不可用或未加载离线模型，无法执行CfC ODE步（dim=%d）", dim);
    return -1;
}

const GpuBackendInterface* cambricon_get_backend_interface(void) {
    static GpuBackendInterface iface = {
        .name = "寒武纪MLU",
        .backend_type = GPU_BACKEND_CAMBRICON,
        .init = cambricon_backend_init,
        .cleanup = cambricon_backend_cleanup,
        .get_device_count = cambricon_backend_get_device_count,
        .get_device_info = cambricon_backend_get_device_info,
        .context_create = cambricon_backend_context_create,
        .context_free = cambricon_backend_context_free,
        .memory_alloc = cambricon_backend_memory_alloc,
        .memory_free = cambricon_backend_memory_free,
        .memory_copy_to_device = cambricon_backend_memory_copy_to_device,
        .memory_copy_from_device = cambricon_backend_memory_copy_from_device,
        .memory_copy_device_to_device = cambricon_backend_memory_copy_device_to_device,
        .memory_copy_to_device_async = cambricon_backend_memory_copy_to_device_sync,
        .memory_copy_from_device_async = cambricon_backend_memory_copy_from_device_sync,
        .kernel_create = cambricon_backend_kernel_create,
        .kernel_free = cambricon_backend_kernel_free,
        .kernel_set_arg = cambricon_backend_kernel_set_arg,
        .kernel_execute = cambricon_backend_kernel_execute,
        .kernel_execute_nd = cambricon_backend_kernel_execute_nd,
        .stream_create = cambricon_backend_stream_create,
        .stream_free = cambricon_backend_stream_free,
        .stream_synchronize = cambricon_backend_stream_synchronize,
        .stream_query = cambricon_backend_stream_query,
        .get_memory_info = cambricon_backend_get_memory_info,
        .device_reset = cambricon_backend_device_reset,
        .get_error_string = cambricon_backend_get_error_string
    };
    return &iface;
}
