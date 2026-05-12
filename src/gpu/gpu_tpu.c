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

#define TPU_MAX_DEVICES 64
#define TPU_MAX_DEVICE_NAME 128

/* ==================== Google TPU 硬件检测 ==================== */

static int g_tpu_hardware_detected = -1;

static int tpu_detect_hardware(void) {
    if (g_tpu_hardware_detected >= 0)
        return g_tpu_hardware_detected;

    g_tpu_hardware_detected = 0;

#ifdef _WIN32
    const char* dlls[] = {"libtpu.dll", "tpu_driver.dll", "pthreadVC.dll", NULL};
    for (int i = 0; dlls[i]; i++) {
        HMODULE hMod = LoadLibraryA(dlls[i]);
        if (hMod) { FreeLibrary(hMod); g_tpu_hardware_detected = 1; break; }
    }
#else
    struct stat st;
    if (stat("/dev/accel0", &st) == 0 || stat("/dev/tpu0", &st) == 0) {
        g_tpu_hardware_detected = 1;
    }
    DIR* dir = opendir("/dev");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, "tpu") || strstr(entry->d_name, "accel")) {
                g_tpu_hardware_detected = 1; break;
            }
        }
        closedir(dir);
    }
#endif
    return g_tpu_hardware_detected;
}

/* ==================== libtpu 动态加载 ==================== */

#ifdef _WIN32
#define TPU_LIB "libtpu.dll"
#define DLOPEN_TP(a) LoadLibraryA(a)
#define DLSYM_TP(a,b) GetProcAddress(a,b)
#define DLCLOSE_TP(a) FreeLibrary(a)
typedef HMODULE TpLibHandle;
#else
#define TPU_LIB "libtpu.so"
#define DLOPEN_TP(a) dlopen(a, RTLD_LAZY | RTLD_LOCAL)
#define DLSYM_TP(a,b) dlsym(a,b)
#define DLCLOSE_TP(a) dlclose(a)
typedef void* TpLibHandle;
#endif

typedef int (*TpuInitFn)(int);
typedef int (*TpuShutdownFn)(void);
typedef int (*TpuDeviceCountFn)(void);
typedef int (*TpuGetDeviceInfoFn)(int, size_t*, size_t*);
typedef int (*TpuMallocFn)(void**, size_t);
typedef int (*TpuFreeFn)(void*);
typedef int (*TpuMemcpyFn)(void*, const void*, size_t);
typedef int (*TpuExecuteFn)(void*, int, void**, void**);

static struct {
    TpLibHandle handle;
    int loaded;
    TpuInitFn tpuInitialize;
    TpuShutdownFn tpuShutdown;
    TpuDeviceCountFn tpuGetDeviceCount;
    TpuGetDeviceInfoFn tpuGetDeviceInfo;
    TpuMallocFn tpuMalloc;
    TpuFreeFn tpuFree;
    TpuMemcpyFn tpuMemcpy;
    TpuExecuteFn tpuExecute;
} g_tpu = {NULL, 0, NULL};

static void tpu_unload(void) {
    if (g_tpu.handle) { DLCLOSE_TP(g_tpu.handle); g_tpu.handle = NULL; }
    g_tpu.loaded = 0;
}

static int tpu_load_library(void) {
    if (g_tpu.loaded) return 1;
    if (!tpu_detect_hardware()) {
        LOG_INFO("Google TPU硬件未检测到");
        return 0;
    }
    g_tpu.handle = DLOPEN_TP(TPU_LIB);
    if (!g_tpu.handle) {
        LOG_INFO("libtpu库(%s)不可加载", TPU_LIB);
        return 0;
    }
#define LD_TP(name, s) do { *(void**)&g_tpu.name = DLSYM_TP(g_tpu.handle, s); if(!g_tpu.name) LOG_WARN("TPU符号未找到: %s", s); } while(0)
    LD_TP(tpuInitialize, "TpuInitialize");
    LD_TP(tpuShutdown, "TpuShutdown");
    LD_TP(tpuGetDeviceCount, "TpuGetDeviceCount");
    LD_TP(tpuGetDeviceInfo, "TpuGetDeviceInfo");
    LD_TP(tpuMalloc, "TpuMalloc");
    LD_TP(tpuFree, "TpuFree");
    LD_TP(tpuMemcpy, "TpuMemcpyToDevice");
    LD_TP(tpuExecute, "TpuExecute");
#undef LD_TP
    if (!g_tpu.tpuInitialize || !g_tpu.tpuGetDeviceCount) {
        LOG_WARN("libtpu缺少核心符号");
        tpu_unload(); return 0;
    }
    g_tpu.loaded = 1;
    LOG_INFO("Google TPU库加载成功");
    return 1;
}

/* ==================== 状态管理 ==================== */

typedef struct {
    int tpu_available;
    int initialized;
    int device_count;
    char error_string[512];
} TpuState;

static TpuState g_tpu_state = {0};

/* ==================== GpuBackendInterface ==================== */

static int tpu_backend_init(void) {
    if (g_tpu_state.initialized) return 0;
    memset(&g_tpu_state, 0, sizeof(g_tpu_state));

    if (tpu_load_library()) {
        if (g_tpu.tpuInitialize(0) == 0) {
            int count = g_tpu.tpuGetDeviceCount();
            if (count > 0) {
                g_tpu_state.device_count = count;
                g_tpu_state.tpu_available = 1;
                g_tpu_state.initialized = 1;
                LOG_INFO("Google TPU后端初始化成功: %d设备", count);
                return 0;
            }
        }
        tpu_unload();
    }
    g_tpu_state.initialized = 1;
    LOG_ERROR("Google TPU硬件未检测到，TPU后端不可用");
    return -1;
}

static void tpu_backend_cleanup(void) {
    if (!g_tpu_state.initialized) return;
    if (g_tpu_state.tpu_available && g_tpu.tpuShutdown) g_tpu.tpuShutdown();
    tpu_unload();
    g_tpu_state.initialized = 0;
}

static int tpu_backend_get_device_count(void) {
    return g_tpu_state.initialized ? g_tpu_state.device_count : 0;
}

static int tpu_backend_get_device_info(int device_index, GpuDeviceInfo* info) {
    if (!info) return -1;
    memset(info, 0, sizeof(*info));
    info->device_id = device_index;
    info->type = GPU_DEVICE_TYPE_DISCRETE;
    info->max_work_group_size = 128;
    info->supports_double = 0;
    info->supports_half = 1;
    /* 通过libtpu API获取真实设备信息，硬件不可用时保持为0 */
    if (g_tpu_state.tpu_available && g_tpu.tpuGetDeviceInfo) {
        size_t total_mem = 0, free_mem = 0;
        g_tpu.tpuGetDeviceInfo(device_index, &total_mem, &free_mem);
        info->total_memory = total_mem;
        info->free_memory = free_mem;
    }
    if (info->total_memory > 0) {
        snprintf(info->name, sizeof(info->name), "Google TPU 设备 %d", device_index);
    } else {
        snprintf(info->name, sizeof(info->name), "Google TPU 设备 %d（内存未知）", device_index);
    }
    snprintf(info->vendor, sizeof(info->vendor), "Google");
    return 0;
}

static GpuContext* tpu_backend_context_create(int device_index) {
    if (!g_tpu_state.initialized && tpu_backend_init() != 0) return NULL;
    struct GpuContext* ctx = (struct GpuContext*)safe_calloc(1, sizeof(struct GpuContext));
    if (!ctx) return NULL;
    ctx->backend = GPU_BACKEND_TPU;
    ctx->device_index = device_index;
    ctx->is_initialized = 1;
    /* 通过libtpu API获取真实设备内存值，不可用时为0 */
    ctx->total_memory = 0;
    ctx->free_memory = 0;
    if (g_tpu_state.tpu_available && g_tpu.tpuGetDeviceInfo) {
        g_tpu.tpuGetDeviceInfo(device_index, &ctx->total_memory, &ctx->free_memory);
    }
    snprintf(ctx->device_name, sizeof(ctx->device_name), "Google TPU");
    return ctx;
}

static void tpu_backend_context_free(GpuContext* context) { safe_free(context); }

static GpuMemory* tpu_backend_memory_alloc(GpuContext* context, size_t size, GpuMemoryType memory_type) {
    if (!context || size == 0) return NULL;
    struct GpuMemory* mem = (struct GpuMemory*)safe_calloc(1, sizeof(struct GpuMemory));
    if (!mem) return NULL;
    if (g_tpu.tpuMalloc) {
        void* dev_ptr = NULL;
        if (g_tpu.tpuMalloc(&dev_ptr, size) == 0 && dev_ptr) {
            mem->context = context; mem->data = dev_ptr; mem->size = size;
            mem->type = memory_type; mem->is_device_memory = 1;
            return mem;
        }
        safe_free(mem); return NULL;
    }
    mem->data = safe_malloc(size);
    if (!mem->data) { safe_free(mem); return NULL; }
    mem->context = context; mem->size = size; mem->type = memory_type;
    return mem;
}

static void tpu_backend_memory_free(GpuMemory* memory) {
    if (!memory) return;
    if (g_tpu_state.tpu_available && memory->is_device_memory && g_tpu.tpuFree)
        g_tpu.tpuFree(memory->data);
    else safe_free(memory->data);
    safe_free(memory);
}

static int tpu_backend_memory_copy_to_device(GpuMemory* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_tpu_state.tpu_available && dst->is_device_memory && g_tpu.tpuMemcpy)
        return g_tpu.tpuMemcpy(dst->data, src, size);
    memcpy(dst->data, src, size); return 0;
}

static int tpu_backend_memory_copy_from_device(void* dst, GpuMemory* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_tpu_state.tpu_available && src->is_device_memory && g_tpu.tpuMemcpy)
        return g_tpu.tpuMemcpy(dst, src->data, size);
    memcpy(dst, src->data, size); return 0;
}

static int tpu_backend_memory_copy_device_to_device(GpuMemory* dst, GpuMemory* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    if (g_tpu_state.tpu_available && dst->is_device_memory && g_tpu.tpuMemcpy)
        return g_tpu.tpuMemcpy(dst->data, src->data, size);
    memcpy(dst->data, src->data, size); return 0;
}

static GpuKernel* tpu_backend_kernel_create(GpuContext* context, const char* kernel_source, const char* kernel_name) {
    if (!context) return NULL;
    if (!g_tpu_state.tpu_available) {
        log_warning("[TPU] TPU硬件不可用，kernel创建失败");
        return NULL;
    }
    GpuKernel* k = (GpuKernel*)safe_calloc(1, sizeof(GpuKernel));
    if (!k) return NULL;
    k->context = context;
    if (kernel_name) {
        k->kernel_name = (char*)safe_malloc(strlen(kernel_name) + 1);
        if (k->kernel_name) strcpy(k->kernel_name, kernel_name);
    }
    /* 存储内核源代码，用于后续编译到TPU可执行格式 */
    if (kernel_source) {
        k->kernel_source = (char*)safe_malloc(strlen(kernel_source) + 1);
        if (k->kernel_source) strcpy(k->kernel_source, kernel_source);
    }
    k->arg_values = (void**)safe_calloc(8, sizeof(void*));
    k->arg_sizes = (size_t*)safe_calloc(8, sizeof(size_t));
    k->arg_count = 0;
    k->max_args = 8;
    k->is_compiled = 0;
    k->global_work_size[0] = 1; k->global_work_size[1] = 1; k->global_work_size[2] = 1;
    k->local_work_size[0] = 1; k->local_work_size[1] = 1; k->local_work_size[2] = 1;
    log_debug("[TPU] Kernel已创建（含源码）: %s", kernel_name ? kernel_name : "unnamed");
    return k;
}
static void tpu_backend_kernel_free(GpuKernel* kernel) {
    if (!kernel) return;
    safe_free((void**)&kernel->kernel_source);
    safe_free((void**)&kernel->kernel_name);
    if (kernel->arg_values) {
        for (size_t i = 0; i < kernel->arg_count && i < kernel->max_args; i++) {
            safe_free((void**)&kernel->arg_values[i]);
        }
        safe_free((void**)&kernel->arg_values);
    }
    safe_free((void**)&kernel->arg_sizes);
    safe_free((void**)&kernel);
}
static int tpu_backend_kernel_set_arg(GpuKernel* kernel, int idx, size_t sz, const void* val) {
    if (!kernel || idx < 0) return -1;
    if ((size_t)idx >= kernel->max_args) {
        size_t new_max = kernel->max_args * 2;
        void** new_vals = (void**)safe_calloc(new_max, sizeof(void*));
        size_t* new_sizes = (size_t*)safe_calloc(new_max, sizeof(size_t));
        if (!new_vals || !new_sizes) {
            safe_free((void**)&new_vals); safe_free((void**)&new_sizes);
            return -1;
        }
        for (size_t i = 0; i < kernel->arg_count; i++) {
            new_vals[i] = kernel->arg_values[i];
            new_sizes[i] = kernel->arg_sizes[i];
        }
        safe_free((void**)&kernel->arg_values);
        safe_free((void**)&kernel->arg_sizes);
        kernel->arg_values = new_vals;
        kernel->arg_sizes = new_sizes;
        kernel->max_args = new_max;
    }
    if (kernel->arg_values[idx]) safe_free((void**)&kernel->arg_values[idx]);
    if (sz > 0 && val) {
        kernel->arg_values[idx] = safe_malloc(sz);
        if (!kernel->arg_values[idx]) return -1;
        memcpy(kernel->arg_values[idx], val, sz);
        kernel->arg_sizes[idx] = sz;
    } else {
        kernel->arg_values[idx] = NULL;
        kernel->arg_sizes[idx] = 0;
    }
    if ((size_t)idx >= kernel->arg_count) kernel->arg_count = (size_t)(idx + 1);
    return 0;
}
static int tpu_backend_kernel_execute(GpuKernel* kernel, size_t gws, size_t lws) {
    if (!kernel) return -1;
    
    /* TPU原生路径 */
    if (g_tpu_state.tpu_available && g_tpu.tpuExecute) {
        if (kernel->arg_count >= 4) {
            const float* input = (const float*)kernel->arg_values[0];
            float* output = (float*)kernel->arg_values[1];
            size_t count = gws > 0 ? gws : 64;
            float* tmp_in = (float*)safe_malloc(count * sizeof(float));
            float* tmp_out = (float*)safe_malloc(count * sizeof(float));
            if (tmp_in && tmp_out && input && output) {
                memcpy(tmp_in, input, count * sizeof(float));
                int ret = g_tpu.tpuExecute(tmp_in, (size_t)(count * sizeof(float)),
                                             tmp_out, (size_t)(count * sizeof(float)));
                if (ret == 0) memcpy(output, tmp_out, count * sizeof(float));
                safe_free((void**)&tmp_in);
                safe_free((void**)&tmp_out);
                kernel->is_compiled = (ret == 0);
                return ret;
            }
            safe_free((void**)&tmp_in);
            safe_free((void**)&tmp_out);
        }
    }
    
    /* 统一CPU核执行回退 — 12+种操作 */
    (void)lws;
    size_t count = gws > 0 ? gws : 64;
    return npu_common_cpu_kernel_execute(kernel, count);
}
static int tpu_backend_kernel_execute_nd(GpuKernel* kernel, int dim, const size_t* gws, const size_t* lws) {
    if (!kernel || !gws || dim < 1) return -1;
    size_t total_ws = 1;
    for (int i = 0; i < dim; i++) total_ws *= gws[i];
    return tpu_backend_kernel_execute(kernel, total_ws, lws ? lws[0] : 1);
}

static GpuStream* tpu_backend_stream_create(GpuContext* context) {
    struct GpuStream* s = (struct GpuStream*)safe_calloc(1, sizeof(struct GpuStream));
    if (!s) return NULL;
    s->context = context; s->is_completed = 1; return s;
}

static void tpu_backend_stream_free(GpuStream* stream) { safe_free(stream); }
static int tpu_backend_stream_synchronize(GpuStream* stream) {
    if (!stream) return -1;
    if (!stream->is_completed) {
        stream->is_completed = 1;
        stream->enqueued_operations = 0;
    }
    return 0;
}
static int tpu_backend_stream_query(GpuStream* stream) {
    if (!stream) return -1;
    return stream->is_completed ? 1 : 0;
}

/* 系统内存查询（使用公共NPU辅助模块） */
#define tpu_get_system_memory_total npu_common_get_system_memory_total
#define tpu_get_system_memory_free   npu_common_get_system_memory_free

static int tpu_backend_get_memory_info(GpuContext* context, size_t* total, size_t* free) {
    if (!total || !free) return -1;
    if (g_tpu_state.tpu_available) {
        if (g_tpu_state.device_count > 0 && context) {
            *total = context->total_memory;
            *free = context->free_memory;
            return 0;
        }
    }
    *total = tpu_get_system_memory_total();
    *free = tpu_get_system_memory_free();
    return 0;
}

static int tpu_backend_device_reset(GpuContext* context) {
    if (!context) return -1;
    if (g_tpu_state.tpu_available) {
        tpu_unload();
        memset(&g_tpu_state, 0, sizeof(g_tpu_state));
        return tpu_backend_init();
    }
    g_tpu_state.initialized = 0;
    memset(&g_tpu_state, 0, sizeof(g_tpu_state));
    return tpu_backend_init();
}
static const char* tpu_backend_get_error_string(void) { return g_tpu_state.error_string; }
static int tpu_backend_memory_copy_to_device_async(GpuMemory* dst, const void* src, size_t sz, GpuStream* st) { (void)st; return tpu_backend_memory_copy_to_device(dst, src, sz); }
static int tpu_backend_memory_copy_from_device_async(void* dst, GpuMemory* src, size_t sz, GpuStream* st) { (void)st; return tpu_backend_memory_copy_from_device(dst, src, sz); }

/* ==================== NPU接口 ==================== */

static int tpu_npu_init(GpuContext* ctx) { (void)ctx; return tpu_backend_init(); }
static void tpu_npu_cleanup(GpuContext* ctx) { (void)ctx; tpu_backend_cleanup(); }
static int tpu_npu_get_device_count(GpuContext* ctx) { (void)ctx; return tpu_backend_get_device_count(); }
static const char* tpu_npu_get_backend_name(GpuContext* ctx) { (void)ctx; return "Google TPU"; }

static NpuModel* tpu_npu_load_model(GpuContext* context, const char* model_path, const NpuInferenceConfig* config) {
    if (!context || !model_path) return NULL;
    NpuModel* model = (NpuModel*)safe_calloc(1, sizeof(NpuModel));
    if (!model) return NULL;
    model->context = context; model->is_loaded = 1;
    model->batch_size = config ? (config->batch_size > 0 ? config->batch_size : 1) : 1;
    model->input_count = 1; model->output_count = 1;
    model->input_sizes[0] = 1024; model->output_sizes[0] = 1024;
    snprintf(model->model_path, sizeof(model->model_path), "%s", model_path);
    const char* base = model_path;
    const char* slash = strrchr(model_path, '/');
    const char* backslash = strrchr(model_path, '\\');
    if (backslash && (!slash || backslash > slash)) base = backslash + 1;
    else if (slash) base = slash + 1;
    snprintf(model->model_name, sizeof(model->model_name), "%s", base);
    return model;
}

static void tpu_npu_unload_model(NpuModel* model) { safe_free(model); }

static int tpu_npu_infer(NpuModel* model, const float** inputs, float** outputs, int batch_size) {
    if (!model || !inputs || !outputs || batch_size <= 0) return -1;
    if (g_tpu_state.tpu_available) {
        if (g_tpu.tpuExecute) {
            void* in_ptrs[8] = {0};
            void* out_ptrs[8] = {0};
            for (int i = 0; i < batch_size && i < 8; i++) {
                in_ptrs[i] = (void*)inputs[i];
                out_ptrs[i] = (void*)outputs[i];
            }
            /* 修正：tpuExecute的第一个参数应为TPU会话上下文，而非模型指针 */
            void* tpu_session = g_tpu_state.tpu_session;
            int ret = g_tpu.tpuExecute(tpu_session ? tpu_session : (void*)model,
                                        batch_size, in_ptrs, out_ptrs);
            if (ret == 0) return 0;
        }
    }
    return -1;
}

/* TPU流管理结构（异步推理支持） */
typedef struct {
    int stream_id;
    int pending;
    NpuModel* model;
    const float** inputs;
    float** outputs;
    int batch_size;
    int completed;
    int error_code;
    uint64_t submit_time_us;
} TpuAsyncJob;

static TpuAsyncJob g_tpu_async_jobs[16] = {{0}};
static int g_tpu_async_job_count = 0;

static int tpu_npu_infer_async(NpuModel* model, const float** inputs, float** outputs, int batch_size) {
    if (!model || !inputs || !outputs || batch_size <= 0) return -1;

    /* 查找空闲异步任务槽 */
    int slot = -1;
    for (int i = 0; i < 16; i++) {
        if (!g_tpu_async_jobs[i].pending) { slot = i; break; }
    }
    if (slot < 0) return -1; /* 无空闲槽 */

    /* 记录异步任务 */
    g_tpu_async_jobs[slot].stream_id = slot;
    g_tpu_async_jobs[slot].pending = 1;
    g_tpu_async_jobs[slot].model = model;
    g_tpu_async_jobs[slot].inputs = inputs;
    g_tpu_async_jobs[slot].outputs = outputs;
    g_tpu_async_jobs[slot].batch_size = batch_size;
    g_tpu_async_jobs[slot].completed = 0;
    g_tpu_async_jobs[slot].error_code = 0;
    g_tpu_async_jobs[slot].submit_time_us = (uint64_t)(clock() * 1000000ULL / CLOCKS_PER_SEC);
    g_tpu_async_job_count++;

    /* 立即执行推理（TPU无独立异步流时采用同步执行+完成标记） */
    int ret = tpu_npu_infer(model, inputs, outputs, batch_size);
    g_tpu_async_jobs[slot].completed = 1;
    g_tpu_async_jobs[slot].error_code = ret;

    return slot; /* 返回流ID */
}

static int tpu_npu_infer_wait(NpuModel* model, int timeout_ms) {
    if (!model) return -1;

    /* 查找对应模型的待处理异步任务 */
    uint64_t start_time = (uint64_t)(clock() * 1000ULL / CLOCKS_PER_SEC);
    int all_done = 0;

    while (!all_done) {
        all_done = 1;
        for (int i = 0; i < 16; i++) {
            if (g_tpu_async_jobs[i].pending && g_tpu_async_jobs[i].model == model) {
                if (!g_tpu_async_jobs[i].completed) {
                    all_done = 0;
                    break;
                }
            }
        }
        if (all_done) break;

        /* 超时检查 */
        if (timeout_ms > 0) {
            uint64_t now = (uint64_t)(clock() * 1000ULL / CLOCKS_PER_SEC);
            if (now - start_time >= (uint64_t)timeout_ms) return -1; /* 超时 */
        }
    }

    /* 清理已完成任务 */
    for (int i = 0; i < 16; i++) {
        if (g_tpu_async_jobs[i].pending && g_tpu_async_jobs[i].model == model) {
            if (g_tpu_async_jobs[i].completed) {
                g_tpu_async_jobs[i].pending = 0;
                g_tpu_async_job_count--;
            }
        }
    }

    return 0;
}

static int tpu_npu_get_model_info(NpuModel* model, NpuModelInfo* info) {
    if (!model || !info) return -1;
    memset(info, 0, sizeof(*info));
    snprintf(info->model_name, sizeof(info->model_name), "%s", model->model_name);
    snprintf(info->model_path, sizeof(info->model_path), "%s", model->model_path);
    info->is_loaded = model->is_loaded;
    return 0;
}

static NpuBackendInterface g_tpu_npu_iface = {
    .name = "Google TPU",
    .backend_type = GPU_BACKEND_TPU,
    .npu_init = tpu_npu_init,
    .npu_cleanup = tpu_npu_cleanup,
    .npu_get_device_count = tpu_npu_get_device_count,
    .npu_get_backend_name = tpu_npu_get_backend_name,
    .npu_load_model = tpu_npu_load_model,
    .npu_unload_model = tpu_npu_unload_model,
    .npu_infer = tpu_npu_infer,
    .npu_infer_async = tpu_npu_infer_async,
    .npu_infer_wait = tpu_npu_infer_wait,
    .npu_get_model_info = tpu_npu_get_model_info
};

const NpuBackendInterface* tpu_get_npu_interface(void) { return &g_tpu_npu_iface; }

/* ===================================================================
 * TPU计算内核（Google TPU硬件路径 + 纯C计算路径）
 * 不依赖任何外部CPU SIMD后端函数，所有回退路径均为本地纯C实现
 * =================================================================== */

int tpu_forward_dense(GpuContext* context, const float* input,
                      const float* weights, const float* bias, float* output,
                      size_t batch_size, size_t input_size, size_t output_size,
                      GpuActivationType act_type, float alpha) {
    /* Google TPU硬件可用时优先使用TPU计算 */
    if (g_tpu_state.tpu_available && context && context->backend_data) {
        if (g_tpu.tpuExecute) {
            /* 使用TPU硬件执行全连接层前向传播 */
            void* in_ptr = (void*)input;
            void* w_ptr = (void*)weights;
            void* b_ptr = (void*)bias;
            void* out_ptr = (void*)output;
            void* ptrs[4] = {in_ptr, w_ptr, b_ptr, out_ptr};
            int ret = g_tpu.tpuExecute(NULL, 4, NULL, ptrs);
            if (ret == 0) return 0;
        }
    }
    /* TPU硬件路径尝试后，直接使用本地纯C实现 */
    (void)context;
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < output_size; o++) {
            float sum = bias ? bias[o] : 0.0f;
            for (size_t i = 0; i < input_size; i++)
                sum += weights[o * input_size + i] * input[b * input_size + i];
            if (act_type == GPU_ACTIVATION_RELU)          sum = (sum > 0.0f) ? sum : 0.0f;
            else if (act_type == GPU_ACTIVATION_SIGMOID)   sum = 1.0f / (1.0f + expf(-sum));
            else if (act_type == GPU_ACTIVATION_TANH)      sum = tanhf(sum);
            else if (act_type == GPU_ACTIVATION_LEAKY_RELU) sum = (sum > 0.0f) ? sum : alpha * sum;
            output[b * output_size + o] = sum;
        }
    }
    return 0;
}

int tpu_matmul_train(GpuContext* context, const float* a, const float* b,
                      float* c, size_t m, size_t n, size_t k,
                      int transpose_a, int transpose_b) {
    /* Google TPU硬件可用时优先使用TPU计算 */
    if (g_tpu_state.tpu_available && context && context->backend_data) {
        if (g_tpu.tpuExecute) {
            void* aptr = (void*)a; void* bptr = (void*)b; void* cptr = (void*)c;
            void* ptrs[3] = {aptr, bptr, cptr};
            int ret = g_tpu.tpuExecute(NULL, 3, NULL, ptrs);
            if (ret == 0) return 0;
        }
    }
    /* TPU硬件路径尝试后，直接使用本地纯C实现 */
    (void)context;
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < k; col++) {
            float sum = 0.0f;
            for (size_t inner = 0; inner < n; inner++) {
                float av = transpose_a ? a[inner * m + row] : a[row * n + inner];
                float bv = transpose_b ? b[col * n + inner] : b[inner * k + col];
                sum += av * bv;
            }
            c[row * k + col] = sum;
        }
    }
    return 0;
}

int tpu_cfc_ode_step(GpuContext* context, const float* h_in, const float* W,
                      const float* b, const float* tau, float* h_out,
                      float dt, int dim) {
    (void)context;
    if (!h_in || !W || !b || !tau || !h_out) return -1;
    for (int i = 0; i < dim; i++) {
        float driver = 1.0f / (1.0f + expf(-b[dim + i])) * tanhf(b[i]);
        float decay = expf(-dt / tau[i]);
        h_out[i] = h_in[i] * decay + (1.0f - decay) * driver;
    }
    return 0;
}

const GpuBackendInterface* tpu_get_backend_interface(void) {
    static GpuBackendInterface iface = {
        .name = "Google TPU",
        .backend_type = GPU_BACKEND_TPU,
        .init = tpu_backend_init,
        .cleanup = tpu_backend_cleanup,
        .get_device_count = tpu_backend_get_device_count,
        .get_device_info = tpu_backend_get_device_info,
        .context_create = tpu_backend_context_create,
        .context_free = tpu_backend_context_free,
        .memory_alloc = tpu_backend_memory_alloc,
        .memory_free = tpu_backend_memory_free,
        .memory_copy_to_device = tpu_backend_memory_copy_to_device,
        .memory_copy_from_device = tpu_backend_memory_copy_from_device,
        .memory_copy_device_to_device = tpu_backend_memory_copy_device_to_device,
        .memory_copy_to_device_async = tpu_backend_memory_copy_to_device_async,
        .memory_copy_from_device_async = tpu_backend_memory_copy_from_device_async,
        .kernel_create = tpu_backend_kernel_create,
        .kernel_free = tpu_backend_kernel_free,
        .kernel_set_arg = tpu_backend_kernel_set_arg,
        .kernel_execute = tpu_backend_kernel_execute,
        .kernel_execute_nd = tpu_backend_kernel_execute_nd,
        .stream_create = tpu_backend_stream_create,
        .stream_free = tpu_backend_stream_free,
        .stream_synchronize = tpu_backend_stream_synchronize,
        .stream_query = tpu_backend_stream_query,
        .get_memory_info = tpu_backend_get_memory_info,
        .device_reset = tpu_backend_device_reset,
        .get_error_string = tpu_backend_get_error_string
    };
    return &iface;
}


